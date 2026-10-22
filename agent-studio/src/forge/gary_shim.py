"""Gary4juce-compatible local endpoints (DESIGN 3.9, SPEC 6.8).

Two small FastAPI apps, one per service port (Foundation 8015, SA3 8006), that speak the wire
contract Gary's plugin uses for its per-service backends and run the work as ordinary Forge jobs:

* ``POST /generate`` takes a JSON request and answers ``{"success": true, "session_id": ...}``.
* ``POST /audio2audio`` is the same request plus ``audio_data`` (base64 WAV) and ``init_noise_level``.
* ``GET /poll_status/{session_id}`` answers ``{success, generation_in_progress, progress (0-100),
  queue_status: {position, estimated_seconds}, status, audio_data (base64 WAV when done), error?}``.
* ``GET /health`` answers ``{"status": "ok"}``.

Written from that contract only (field families and reply shape as documented in the spec). Field
names the spec does not pin down are read by convention and unknown fields are ignored; nothing
here is copied from Gary4juce, which is AGPL.

The wire contract has no authentication, so these apps must only listen on loopback (the CLI
binds them to 127.0.0.1; reach them from a laptop through an SSH tunnel). As defence in depth
against web pages talking to localhost, a request is refused when it carries an ``Origin`` header
(browsers always send one cross-origin; native plugins do not), when its ``Host`` is not a loopback
name, or when a body is sent with a content type other than JSON.

Foundation request -> Forge job (``generate.loop``, or ``transform.audio`` for ``/audio2audio``):
``bars``, ``host_bpm`` and ``key_root`` + ``key_mode`` become ``bars``, ``bpm`` and ``key``;
``family``, ``subfamily``, ``descriptor_knob_a..c`` + ``descriptors_extra`` and ``behavior_tags``
become the tags ``family``, ``subfamily``, ``descriptors`` and ``behaviour`` that
``forge.workers._prompts`` turns into a Foundation prompt; effect toggles and amounts become
``effects`` tokens in that vocabulary (``Medium Reverb``, ``Delay``); ``custom_prompt_override`` is
the job prompt; a negative ``seed`` means random.

Not forwarded: ``guidance_scale`` and ``steps`` (Forge job params have no such knobs; they are the
worker's ``options``). ``init_noise_level`` is clamped to 0..1 and becomes the ``transform.audio``
strength.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import math
import os
import tempfile
import uuid
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api import SafeJSONResponse, read_json_object
from .config import ForgeConfig
from .errors import ForgeError, ValidationFailed
from .manager import WorkerError
from .music import parse_key
from .service import ForgeService

log = logging.getLogger("forge.gary_shim")

MAX_JSON_BYTES = 1024 * 1024
MAX_SESSIONS = 256
MAX_AUDIO_OUT_BYTES = 64 * 1024 * 1024
MAX_TEXT = 200
MAX_LIST = 32
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
# Foundation's effects. A toggle is ``<effect>`` itself, ``<effect>_enabled``/``_on``/``_toggle`` or
# ``enable_<effect>``/``use_<effect>``; an amount is ``<effect>_amount``/``_level``/``_mix``.
EFFECTS = ("reverb", "delay", "distortion", "phaser", "bitcrush")
_TOGGLE_FORMS = (("", ""), ("", "_enabled"), ("", "_on"), ("", "_toggle"), ("enable_", ""), ("use_", ""))
_AMOUNT_FORMS = (("", "_amount"), ("", "_level"), ("", "_mix"))
STATUS_NAMES = {
    "queued": "queued", "running": "processing", "succeeded": "completed", "failed": "failed", "cancelled": "cancelled",
}
# Which model a request prefers when several in the family run the job type.
PREFERRED_MODELS = {
    "foundation": {"generate.loop": "foundation-1", "transform.audio": "foundation-1"},
    "sa3": {"generate.sfx": "sa3-small-sfx", "generate.loop": "sa3-small-music", "transform.audio": "sa3-small-music"},
}


class ShimRequestError(ValueError):
    """The request body is not acceptable; ``args[0]`` is the message returned to the client."""


class JobSpec:
    """What a request turns into: a Forge job type, its params and the model it wants (if named)."""

    def __init__(self, job_type: str, params: dict[str, Any], wanted_model: str | None = None) -> None:
        self.job_type = job_type
        self.params = params
        self.wanted_model = wanted_model


# ---- field readers (every value from the client is hostile) ----


class Fields:
    """Typed, bounded access to a JSON request body; collects every problem instead of stopping at the first."""

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body
        self.problems: list[str] = []

    def first(self, *names: str) -> Any:
        for name in names:
            if self.body.get(name) is not None:
                return self.body[name]
        return None

    def text(self, *names: str, limit: int = MAX_TEXT) -> str:
        value = self.first(*names)
        if value is None:
            return ""
        if not isinstance(value, str) or len(value) > limit:
            self.problems.append(f"{names[0]} must be text of at most {limit} characters")
            return ""
        return value.strip()

    def optional_number(self, *names: str, lo: float, hi: float) -> float | None:
        value = self.first(*names)
        if value is None or value == "":
            return None
        try:
            number = float(value) if isinstance(value, (int, float, str)) and not isinstance(value, bool) else math.nan
        except ValueError:
            number = math.nan
        if not math.isfinite(number) or not lo <= number <= hi:
            self.problems.append(f"{names[0]} must be a number from {lo:g} to {hi:g}")
            return None
        return number

    def number(self, *names: str, lo: float, hi: float, default: float) -> float:
        value = self.optional_number(*names, lo=lo, hi=hi)
        return default if value is None else value

    def seed(self) -> int | None:
        """A seed, or ``None`` to let Forge pick one (also for a negative seed, Gary's "random")."""
        raw = self.first("seed")
        if isinstance(raw, int) and not isinstance(raw, bool) and abs(raw) <= 2**62:
            return None if raw < 0 else raw  # exact for big integers; floats would round them
        value = self.optional_number("seed", lo=-(2**62), hi=2**62)
        return None if value is None or value < 0 else int(value)

    def words(self, *names: str) -> list[str]:
        """A list of short strings, given as a JSON list or a comma-separated string."""
        value = self.first(*names)
        if value is None:
            return []
        items = value.split(",") if isinstance(value, str) else value
        if not isinstance(items, list) or len(items) > MAX_LIST or not all(isinstance(i, str) and len(i) <= MAX_TEXT for i in items):
            self.problems.append(f"{names[0]} must be a list of at most {MAX_LIST} short strings")
            return []
        return [i.strip() for i in items if i.strip()]

    def key(self) -> str | None:
        root, mode = self.text("key_root", limit=8), self.text("key_mode", limit=16)
        if not root:
            return None
        name = f"{root} {mode}".strip()
        if parse_key(name) is None:
            self.problems.append(f"key_root/key_mode {name!r} is not a musical key")
            return None
        return name

    def done(self) -> None:
        if self.problems:
            raise ShimRequestError("; ".join(self.problems))


def _effect_of(key: str, forms: tuple[tuple[str, str], ...]) -> str | None:
    low = key.lower()
    for prefix, suffix in forms:
        if low.startswith(prefix) and low.endswith(suffix) and low[len(prefix) : len(low) - len(suffix)] in EFFECTS:
            return low[len(prefix) : len(low) - len(suffix)]
    return None


def _level(amount: Any) -> str:
    """``Low``/``Medium``/``High`` from a word, a 0..1 amount or a 0..100 amount; ``""`` when there is none."""
    if isinstance(amount, str):
        word = amount.strip().lower()
        if word in ("low", "medium", "high"):
            return word.title()
        try:
            amount = float(word)
        except ValueError:
            return ""
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount):
        return ""
    share = amount / 100 if amount > 1 else amount
    return "Low" if share < 1 / 3 else "Medium" if share < 2 / 3 else "High"


def effect_tags(body: dict[str, Any]) -> dict[str, list[str]]:
    """``{"effects": [...]}`` in Foundation's vocabulary (``Medium Reverb``, ``Delay``, ``High Bitcrush``) from toggle and amount fields."""
    on: set[str] = set()
    levels: dict[str, str] = {}
    for key, value in body.items():
        if (name := _effect_of(key, _TOGGLE_FORMS)) and value is True:
            on.add(name)
        elif name := _effect_of(key, _AMOUNT_FORMS):
            levels[name] = _level(value)
    tokens = []
    for name in (n for n in EFFECTS if n in on):
        level = levels.get(name) or ("Medium" if name == "reverb" else "")  # the vocabulary has no bare "Reverb"
        if name == "bitcrush":
            level = "High" if level == "High" else ""  # ...and only a "High Bitcrush"
        tokens.append(f"{level} {name.title()}".strip())
    return {"effects": tokens} if tokens else {}


# ---- request -> job, per service ----


def foundation_request(body: dict[str, Any], with_audio: bool) -> JobSpec:
    """``/generate`` and ``/audio2audio`` for the Foundation service."""
    f = Fields(body)
    bars = int(f.number("bars", lo=1, hi=16, default=4))
    bpm = f.number("host_bpm", "bpm", lo=40, hi=240, default=120)
    key = f.key()
    tags: dict[str, str | list[str]] = {}
    for name in ("family", "subfamily"):
        if value := f.text(name, limit=64):
            tags[name] = value
    knobs = [f.text(f"descriptor_knob_{letter}", limit=64) for letter in "abc"]
    if descriptors := [d for d in [*knobs, *f.words("descriptors_extra")] if d]:
        tags["descriptors"] = descriptors[:MAX_LIST]
    if behaviours := f.words("behavior_tags", "behaviour_tags"):
        tags["behaviour"] = behaviours
    tags.update(effect_tags(body))
    prompt, seed = f.text("custom_prompt_override", limit=2000), f.seed()
    strength = f.number("init_noise_level", lo=0, hi=1e9, default=0.5)
    f.done()
    if with_audio:
        # transform.audio has no bars/bpm/key params, so they ride along as tags for the prompt builder
        tags.update(bars=str(bars), bpm=f"{bpm:g}", **({"key": key} if key else {}))
        return JobSpec("transform.audio", {"prompt": prompt, "tags": tags, "seed": seed, "strength": min(1.0, strength)})
    params: dict[str, Any] = {"prompt": prompt, "tags": tags, "seed": seed, "bars": bars, "bpm": bpm}
    return JobSpec("generate.loop", {**params, **({"key": key} if key else {})})


def sa3_request(body: dict[str, Any], with_audio: bool) -> JobSpec:
    """``/generate`` and ``/audio2audio`` for the SA3 service: a sound effect, or a loop when a tempo is given."""
    f = Fields(body)
    prompt = f.text("prompt", limit=2000)
    seconds = f.number("duration", "duration_s", "seconds_total", "seconds", lo=0.1, hi=47, default=4)
    bpm = f.optional_number("host_bpm", "bpm", lo=40, hi=240)
    bars = int(f.number("bars", lo=1, hi=16, default=4))
    key = f.key()
    wanted = f.text("model", limit=64) or None
    strength = f.number("init_noise_level", lo=0, hi=1e9, default=0.5)
    seed = f.seed()
    f.done()
    params: dict[str, Any] = {"prompt": prompt, "seed": seed}
    if with_audio:
        return JobSpec("transform.audio", {**params, "strength": min(1.0, strength)}, wanted)
    if bpm is not None:
        return JobSpec("generate.loop", {**params, "bars": bars, "bpm": bpm, **({"key": key} if key else {})}, wanted)
    return JobSpec("generate.sfx", {**params, "duration_s": seconds}, wanted)


# ---- plumbing ----


def pick_model(service: ForgeService, family: str, spec: JobSpec) -> str:
    """The model of ``family`` that will run ``spec`` (the one it names, else the preferred, else the first by id)."""
    candidates = [
        m for m in service.models()
        if m["family"] == family and m["available"] and spec.job_type in m["job_types"]
        and spec.wanted_model in (None, m["id"])
    ]
    if not candidates:
        wanted = f" {spec.wanted_model!r}" if spec.wanted_model else ""
        raise ForgeError(f"no available {family} model{wanted} runs {spec.job_type}", code="no_model", status=503)
    preferred = PREFERRED_MODELS.get(family, {}).get(spec.job_type)
    return str(min(candidates, key=lambda m: (m["id"] != preferred, m["id"]))["id"])


def decode_audio(value: Any, max_bytes: int) -> bytes:
    """The bytes of a base64 field (a ``data:`` URI prefix and line breaks are tolerated), size-capped before decoding."""
    if not isinstance(value, str) or not value:
        raise ShimRequestError("audio_data (base64 WAV) is required")
    text = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    text = "".join(text.split())
    if len(text) > max_bytes * 4 // 3 + 8:
        raise ShimRequestError(f"audio_data larger than {max_bytes} bytes")
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ShimRequestError("audio_data is not valid base64") from exc


async def ingest_audio(service: ForgeService, value: Any, max_bytes: int) -> str:
    """Decode ``audio_data``, validate it as a WAV through the normal upload path and return the asset id."""
    raw = decode_audio(value, max_bytes)
    handle, name = tempfile.mkstemp(prefix=".gary-", suffix=".bin", dir=service.store.uploads_dir)
    tmp = Path(name)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(raw)
        return (await service.upload(tmp, "audio_data.wav"))["asset_id"]
    finally:
        tmp.unlink(missing_ok=True)


class Sessions:
    """``session_id`` -> ``job_id`` for the most recent submissions."""

    def __init__(self, cap: int = MAX_SESSIONS) -> None:
        self._cap = cap
        self._jobs: OrderedDict[str, str] = OrderedDict()

    def add(self, job_id: str) -> str:
        session_id = uuid.uuid4().hex
        self._jobs[session_id] = job_id
        while len(self._jobs) > self._cap:
            self._jobs.popitem(last=False)
        return session_id

    def get(self, session_id: str) -> str | None:
        return self._jobs.get(session_id)


def host_name(header: str) -> str:
    """The host part of a ``Host`` header (port removed, IPv6 brackets kept)."""
    host = header.strip().lower()
    if host.startswith("["):
        return host.split("]", 1)[0] + "]"
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


async def local_only(request: Request) -> None:
    """Refuse browsers (``Origin``), non-loopback ``Host`` names (DNS rebinding) and non-JSON POST bodies."""
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if "origin" in request.headers or host_name(request.headers.get("host", "")) not in LOOPBACK_HOSTS:
        raise ForgeError("this endpoint only serves local, non-browser clients", code="forbidden", status=403)
    if request.method == "POST" and content_type not in ("", "application/json"):
        raise ForgeError("send application/json", code="unsupported_media_type", status=415)


def shim_error(status: int, message: str) -> SafeJSONResponse:
    return SafeJSONResponse({"success": False, "error": message}, status_code=status)


def _describe(exc: ForgeError) -> str:
    if isinstance(exc, ValidationFailed):
        return "; ".join(f"{e['path']}: {e['message']}" for e in exc.errors) or exc.message
    return exc.message


async def poll_payload(service: ForgeService, job_id: str) -> dict[str, Any]:
    """The ``poll_status`` reply for a job; the first output is attached as base64 once it has succeeded."""
    rec = service.get_job(job_id)
    reply: dict[str, Any] = {
        "success": rec.state not in ("failed", "cancelled"),
        "generation_in_progress": rec.state in ("queued", "running"),
        "progress": int(round(100 * rec.progress)),
        "queue_status": {"position": rec.queue_position or 0, "estimated_seconds": rec.eta_s or 0},
        "status": STATUS_NAMES.get(rec.state, rec.state),
    }
    if rec.state in ("failed", "cancelled"):
        reply["error"] = rec.error or rec.state
    elif rec.state == "succeeded":
        reply["progress"] = 100
        ids = rec.result.asset_ids if rec.result else []
        if not ids:
            return {**reply, "success": False, "status": "failed", "error": "the job produced no audio"}
        path = service.asset_path(ids[0])
        if path.suffix.lower() != ".wav" or path.stat().st_size > MAX_AUDIO_OUT_BYTES:
            return {**reply, "success": False, "status": "failed", "error": "the result is not a WAV file of a sensible size"}
        reply["audio_data"] = await asyncio.to_thread(lambda: base64.b64encode(path.read_bytes()).decode("ascii"))
    return reply


def create_shim_app(
    service: ForgeService, config: ForgeConfig, *, name: str, family: str,
    parse: Callable[[dict[str, Any], bool], JobSpec],
) -> FastAPI:
    """One Gary-compatible service. ``family`` is the Forge family whose models do the work."""
    app = FastAPI(
        title=f"Forge Gary shim ({name})", docs_url=None, redoc_url=None, openapi_url=None,
        dependencies=[Depends(local_only)],
    )
    sessions = Sessions()
    audio_limit = config.uploads.max_bytes * 4 // 3 + 64 * 1024

    @app.exception_handler(ForgeError)
    async def forge_error(request: Request, exc: ForgeError) -> Response:
        return shim_error(exc.status, _describe(exc))

    @app.exception_handler(ShimRequestError)
    async def bad_request(request: Request, exc: ShimRequestError) -> Response:
        return shim_error(400, str(exc))

    @app.exception_handler(WorkerError)
    async def worker_error(request: Request, exc: WorkerError) -> Response:
        return shim_error(502, str(exc)[:500])

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> Response:
        return shim_error(422, "invalid request")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        return shim_error(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> Response:
        log.exception("shim %s failed on %s", name, request.url.path)
        return shim_error(500, "internal error")

    async def submit(request: Request, with_audio: bool) -> SafeJSONResponse:
        body = await read_json_object(request, audio_limit if with_audio else MAX_JSON_BYTES)
        spec = parse(body, with_audio)
        model = pick_model(service, family, spec)
        inputs = None
        if with_audio:
            inputs = {"audio": await ingest_audio(service, body.get("audio_data"), config.uploads.max_bytes)}
        rec = await service.submit(spec.job_type, spec.params, model=model, inputs=inputs)
        return SafeJSONResponse({"success": True, "session_id": sessions.add(rec.job_id)})

    @app.get("/health")
    async def health() -> Any:
        return {"status": "ok", "service": name}

    @app.post("/generate")
    async def generate(request: Request) -> Any:
        return await submit(request, False)

    @app.post("/audio2audio")
    async def audio2audio(request: Request) -> Any:
        return await submit(request, True)

    @app.get("/poll_status/{session_id}")
    async def poll_status(session_id: str) -> Any:
        job_id = sessions.get(session_id)
        if job_id is None:
            return shim_error(404, "unknown session")
        return SafeJSONResponse(await poll_payload(service, job_id))

    app.state.sessions = sessions
    return app


def create_foundation_app(service: ForgeService, config: ForgeConfig, *, family: str = "foundation") -> FastAPI:
    return create_shim_app(service, config, name="foundation", family=family, parse=foundation_request)


def create_sa3_app(service: ForgeService, config: ForgeConfig, *, family: str = "sa3") -> FastAPI:
    return create_shim_app(service, config, name="sa3", family=family, parse=sa3_request)


def create_shim_apps(service: ForgeService, config: ForgeConfig) -> dict[str, tuple[int, FastAPI]]:
    """``{service name: (port, app)}`` for the services listed in ``gary_shim.ports``."""
    makers = {"foundation": create_foundation_app, "sa3": create_sa3_app}
    return {
        name: (port, makers[name](service, config))
        for name, port in config.gary_shim.ports.items()
        if name in makers
    }


__all__ = ["create_foundation_app", "create_sa3_app", "create_shim_app", "create_shim_apps"]
