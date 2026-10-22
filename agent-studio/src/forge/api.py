"""The Forge HTTP gateway (DESIGN 3.7): bearer auth, JSON routes, uploads, SSE and asset downloads.

``create_app(service, config)`` returns a FastAPI app. Every route lives under ``/v1`` and needs
``Authorization: Bearer <token>`` (constant-time compare) except ``GET /v1/health``. Every error
is ``{"error": {"code", "message", "details"?}}`` with the matching HTTP status. Request bodies are
read with hard byte caps; nothing here trusts a header, a filename or a number from the client.

List endpoints (``/v1/models``, ``/v1/jobs``, ``/v1/assets``) answer with a plain JSON array.
``GET /v1/assets`` takes ``tags`` either as a JSON object (``tags={"family":"synth"}``) or as
repeated ``key:value`` entries (``tags=family:synth&tags=family:bass`` matches either value).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.background import BackgroundTask
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect

from .config import ForgeConfig
from .errors import BadRequest, ConfigError, ForgeError, NotFound, ValidationFailed
from .manager import WorkerError
from .service import ForgeService, Subscription
from .store import json_safe

log = logging.getLogger("forge.api")

MAX_JSON_BYTES = 4 * 1024 * 1024
MULTIPART_OVERHEAD = 256 * 1024  # boundaries, headers and the small form fields around the file
MAX_EVENT_STREAMS = 64
SSE_KEEPALIVE_S = 15.0
CHUNK_BYTES = 256 * 1024
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MEDIA_TYPES = {".wav": "audio/wav", ".mid": "audio/midi", ".zip": "application/zip", ".json": "application/json"}
_STATUS_CODES = {
    400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
    409: "conflict", 413: "payload_too_large", 415: "unsupported_media_type", 416: "range_not_satisfiable",
    422: "validation_error", 429: "too_many_requests", 503: "unavailable",
}
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


# ---- responses and errors ----


class SafeJSONResponse(JSONResponse):
    """JSON that can never contain NaN or Infinity (they become ``null``)."""

    def render(self, content: Any) -> bytes:
        return json.dumps(json_safe(content), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def error_response(
    status: int, code: str, message: str, details: Any = None, headers: dict[str, str] | None = None
) -> SafeJSONResponse:
    body: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return SafeJSONResponse({"error": body}, status_code=status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    """Map every failure to the error envelope."""

    async def forge_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, ForgeError)
        headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
        return error_response(exc.status, exc.code, exc.message, exc.details, headers)

    async def worker_error(request: Request, exc: Exception) -> Response:
        log.warning("worker failure on %s: %s", request.url.path, exc)
        return error_response(502, "worker_error", str(exc)[:500])

    async def request_invalid(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, RequestValidationError)
        details = [
            {"path": ".".join(str(p) for p in e["loc"][1:]) or str(e["loc"][0]), "message": str(e["msg"])}
            for e in exc.errors()
        ]
        return error_response(422, "validation_error", "invalid request", details)

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, StarletteHTTPException)
        return error_response(exc.status_code, _STATUS_CODES.get(exc.status_code, "error"), str(exc.detail), headers=exc.headers)

    async def disconnected(request: Request, exc: Exception) -> Response:
        log.debug("client went away during %s %s", request.method, request.url.path)
        return error_response(400, "client_disconnected", "the client closed the connection")

    async def unexpected(request: Request, exc: Exception) -> Response:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return error_response(500, "internal_error", "internal error")

    app.add_exception_handler(ForgeError, forge_error)
    app.add_exception_handler(WorkerError, worker_error)
    app.add_exception_handler(RequestValidationError, request_invalid)
    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(ClientDisconnect, disconnected)
    app.add_exception_handler(Exception, unexpected)


# ---- request bodies ----


def declared_length(request: Request) -> int | None:
    value = request.headers.get("content-length", "")
    return int(value) if value.isascii() and value.isdigit() and len(value) < 19 else None


async def read_body(request: Request, limit: int) -> bytes:
    """The request body, refusing (413) as soon as it exceeds ``limit`` bytes."""
    declared = declared_length(request)
    if declared is not None and declared > limit:
        raise BadRequest(f"request body larger than {limit} bytes", code="payload_too_large", status=413)
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise BadRequest(f"request body larger than {limit} bytes", code="payload_too_large", status=413)
        chunks.append(chunk)
    return b"".join(chunks)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


async def read_json_object(request: Request, limit: int = MAX_JSON_BYTES) -> dict[str, Any]:
    raw = await read_body(request, limit)
    try:
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:  # includes UnicodeDecodeError
        raise BadRequest("the request body is not valid JSON", code="invalid_json") from exc
    if not isinstance(data, dict):
        raise ValidationFailed([{"path": "<body>", "message": "must be a JSON object"}])
    return data


def _clean_filename(name: str | None) -> str:
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    return "".join(ch for ch in base if ch.isprintable())[:255]


async def receive_upload(request: Request, service: ForgeService, config: ForgeConfig) -> dict[str, str]:
    """Stream a multipart ``file`` field to disk (byte-capped) and register it as an asset."""
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise ForgeError("uploads must be multipart/form-data with a 'file' field", code="unsupported_media_type", status=415)
    limit = config.uploads.max_bytes + MULTIPART_OVERHEAD
    declared = declared_length(request)
    if declared is not None and declared > limit:
        raise BadRequest(f"upload larger than {config.uploads.max_bytes} bytes", code="upload_too_large", status=413)
    too_large = False

    async def capped() -> AsyncIterator[bytes]:
        nonlocal too_large
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > limit:
                too_large = True
                raise MultiPartException("upload too large")
            yield chunk

    parser = MultiPartParser(request.headers, capped(), max_files=1, max_fields=8, max_part_size=64 * 1024)
    try:
        form = await parser.parse()
    except MultiPartException as exc:
        if too_large:
            raise BadRequest(f"upload larger than {config.uploads.max_bytes} bytes", code="upload_too_large", status=413) from exc
        raise BadRequest(f"malformed multipart body: {exc.message}", code="invalid_multipart") from exc
    try:
        part = form.get("file")
        if not isinstance(part, UploadFile):
            raise ValidationFailed([{"path": "file", "message": "a file field named 'file' is required"}])
        uploads_dir = service.store.uploads_dir
        handle, name = tempfile.mkstemp(prefix=".incoming-", suffix=".bin", dir=uploads_dir)
        tmp = Path(name)
        try:
            with os.fdopen(handle, "wb") as out:
                await asyncio.to_thread(shutil.copyfileobj, part.file, out, CHUNK_BYTES)
            return await service.upload(tmp, _clean_filename(part.filename))
        finally:
            tmp.unlink(missing_ok=True)
    finally:
        await form.close()


# ---- authentication ----


def make_auth(token: str | None) -> Callable[[Request], Awaitable[None]]:
    """A FastAPI dependency that demands ``Authorization: Bearer <token>`` (none when ``token`` is ``None``)."""
    expected = None if token is None else hashlib.sha256(token.encode("utf-8")).digest()

    async def require_auth(request: Request) -> None:
        if expected is None:
            return
        scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
        given = hashlib.sha256(supplied.strip().encode("latin-1", "replace")).digest()
        # hashing first makes the compare length-independent as well as constant-time
        if not (hmac.compare_digest(given, expected) and scheme.lower() == "bearer"):
            raise ForgeError("missing or invalid bearer token", code="unauthorized", status=401)

    return require_auth


# ---- downloads ----


def etag_for(asset_id: str) -> str:
    return f'"{asset_id}"'  # assets are content-addressed, so the id is a strong validator


def etag_matches(header: str, etag: str) -> bool:
    wanted = etag.strip('"')
    for item in header.split(","):
        item = item.strip()
        if item == "*" or item.removeprefix("W/").strip('"') == wanted:
            return True
    return False


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """The inclusive byte span for a single-range header; ``None`` means serve the whole file.

    Raises :class:`ForgeError` 416 when the range cannot be satisfied. Unknown units, malformed
    values and multi-range requests are ignored (RFC 9110 allows serving the full representation).
    """
    if not header:
        return None
    match = _RANGE.match(header.strip())
    if match is None:
        return None
    first, last = match.groups()
    if not first and not last:
        return None
    unsatisfiable = ForgeError(
        f"range not satisfiable for a {size}-byte asset", code="range_not_satisfiable", status=416,
        details={"size": size},
    )
    if not first:  # suffix: the last N bytes
        count = int(last)
        if count == 0 or size == 0:
            raise unsatisfiable
        return max(0, size - count), size - 1
    start = int(first)
    end = size - 1 if not last else min(int(last), size - 1)
    if last and int(last) < start:
        return None  # syntactically invalid: ignore it
    if start >= size:
        raise unsatisfiable
    return start, end


async def iter_file(path: Path, start: int, length: int) -> AsyncIterator[bytes]:
    try:
        async with await anyio.open_file(path, "rb") as handle:
            await handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = await handle.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
    except OSError as exc:
        log.warning("reading %s failed: %s", path, exc)


def asset_response(request: Request, asset_id: str, path: Path) -> Response:
    """Stream ``path`` with ETag, ``Accept-Ranges`` and single-range (206) support."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise NotFound("no such asset", code="asset_not_found") from exc
    etag = etag_for(asset_id)
    headers = {
        "ETag": etag, "Accept-Ranges": "bytes", "Cache-Control": "private, max-age=86400",
        "X-Content-Type-Options": "nosniff",
        "Content-Disposition": f'attachment; filename="{asset_id}{path.suffix.lower()}"',
    }
    media_type = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    none_match = request.headers.get("if-none-match")
    if none_match and etag_matches(none_match, etag):
        return Response(status_code=304, headers=headers)
    span = None
    if_range = request.headers.get("if-range")
    if if_range is None or if_range.strip() == etag:
        try:
            span = parse_range(request.headers.get("range"), size)
        except ForgeError as exc:
            return error_response(416, exc.code, exc.message, exc.details, {**headers, "Content-Range": f"bytes */{size}"})
    status, start, length = 200, 0, size
    if span is not None:
        status, start = 206, span[0]
        length = span[1] - span[0] + 1
        headers["Content-Range"] = f"bytes {span[0]}-{span[1]}/{size}"
    headers["Content-Length"] = str(length)
    if request.method == "HEAD":
        return Response(status_code=status, headers=headers, media_type=media_type)
    return StreamingResponse(iter_file(path, start, length), status_code=status, headers=headers, media_type=media_type)


# ---- server-sent events ----


_END: dict[str, Any] = {"end": True}  # pushed into a subscription to make its stream finish


class EventStream:
    """One ``GET /v1/events`` connection: a service subscription rendered as SSE frames.

    The subscription is taken when the object is built (before the response starts), so no event
    published after the request was accepted is missed. ``close`` is idempotent and releases the
    slot in the per-app stream cap; ``end`` asks the stream to finish (see :func:`end_event_streams`).
    """

    def __init__(self, app: FastAPI, service: ForgeService, job_id: str | None) -> None:
        if len(app.state.event_streams) >= MAX_EVENT_STREAMS:
            raise ForgeError("too many open event streams", code="too_many_streams", status=503)
        self._app = app
        self._job_id = job_id
        self._sub: Subscription = service.subscribe()
        app.state.event_streams.add(self)

    def close(self) -> None:
        self._sub.close()
        self._app.state.event_streams.discard(self)

    def end(self) -> None:
        self._sub.push(_END)

    async def frames(self) -> AsyncIterator[bytes]:
        try:
            yield b"retry: 3000\n: connected\n\n"
            number = 0
            while True:
                try:
                    event = await asyncio.wait_for(self._sub.get(), SSE_KEEPALIVE_S)
                except asyncio.TimeoutError:
                    yield b": keepalive\n\n"
                    continue
                if event is _END:
                    return
                if self._job_id is not None and event.get("job_id") != self._job_id:
                    continue
                number += 1
                data = json.dumps(json_safe(event), ensure_ascii=False, allow_nan=False)
                yield f"id: {number}\nevent: job\ndata: {data}\n\n".encode()
        finally:
            self.close()


def end_event_streams(app: FastAPI) -> None:
    """Make every open ``/v1/events`` connection of ``app`` finish, so a server can shut down at once."""
    for stream in list(app.state.event_streams):
        stream.end()


# ---- queries ----


def parse_tag_filters(raw: list[str]) -> dict[str, list[str]]:
    """``tags`` query values: JSON objects and/or ``key:value`` pairs; repeated keys match any of their values."""
    out: dict[str, list[str]] = {}

    def fail(message: str) -> ValidationFailed:
        return ValidationFailed([{"path": "tags", "message": message}])

    for item in raw[:64]:
        pairs: list[tuple[str, str]] = []
        if item.lstrip().startswith("{"):
            try:
                obj = json.loads(item)
            except (ValueError, RecursionError) as exc:
                raise fail("must be a JSON object or key:value") from exc
            if not isinstance(obj, dict):
                raise fail("must be a JSON object or key:value")
            for key, value in obj.items():
                values = [value] if isinstance(value, str) else value
                if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                    raise fail("tag values must be strings or lists of strings")
                pairs += [(str(key), v) for v in values]
        else:
            key, sep, value = item.partition(":")
            if not sep:
                key, sep, value = item.partition("=")
            if not sep or not key.strip() or not value.strip():
                raise fail("must be a JSON object or key:value")
            pairs.append((key.strip(), value.strip()))
        for key, value in pairs:
            if len(key) > 64 or len(value) > 200:
                raise fail("tag name or value too long")
            out.setdefault(key, []).append(value)
    if sum(len(v) for v in out.values()) > 64:
        raise fail("too many tag filters")
    return out


def _json(data: Any, status: int = 200) -> SafeJSONResponse:
    return SafeJSONResponse(data, status_code=status)


def _reject_unknown(body: dict[str, Any], allowed: set[str]) -> None:
    unknown = [{"path": key, "message": "unknown field"} for key in body if key not in allowed]
    if unknown:
        raise ValidationFailed(unknown)


# ---- the app ----


def create_app(
    service: ForgeService, config: ForgeConfig, *, token: str | None = None, manage_lifecycle: bool = False
) -> FastAPI:
    """Build the gateway app.

    ``token`` overrides the bearer token (default: ``config.require_token()``, which raises
    :class:`~forge.errors.ConfigError` for an empty token unless ``insecure_no_auth`` is set).
    ``manage_lifecycle`` starts and stops ``service`` with the app's lifespan; leave it off when
    the caller owns the service (the CLI, the tests).
    """
    bearer = token if token is not None else config.require_token()
    if bearer == "":
        raise ConfigError("the bearer token must not be empty")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if manage_lifecycle:
            await service.start()
        try:
            yield
        finally:
            if manage_lifecycle:
                await service.stop()

    app = FastAPI(
        title="Forge", docs_url=None, redoc_url=None, openapi_url=None,
        default_response_class=SafeJSONResponse, lifespan=lifespan,
    )
    app.state.service = service
    app.state.config = config
    app.state.event_streams = set()
    install_error_handlers(app)

    public = APIRouter(prefix="/v1")
    router = APIRouter(prefix="/v1", dependencies=[Depends(make_auth(bearer))])

    @public.get("/health")
    async def health() -> Any:
        return await service.health()

    # -- models --

    def model_view(model_id: str) -> dict[str, Any]:
        if not MODEL_ID.match(model_id):
            raise NotFound(f"unknown model {model_id!r}", code="model_not_found")
        for info in service.models():
            if info["id"] == model_id:
                return info
        raise NotFound(f"unknown model {model_id!r}", code="model_not_found")

    @router.get("/models")
    async def list_models() -> Any:
        return service.models()

    @router.post("/models/{model_id}/load")
    async def load_model(model_id: str) -> Any:
        model_view(model_id)
        await service.load_model(model_id)
        return model_view(model_id)

    @router.post("/models/{model_id}/unload")
    async def unload_model(model_id: str) -> Any:
        model_view(model_id)
        await service.unload_model(model_id)
        return model_view(model_id)

    # -- uploads --

    @router.post("/uploads")
    async def upload(request: Request) -> Any:
        return _json(await receive_upload(request, service, config))

    # -- jobs --

    @router.post("/jobs")
    async def submit_job(request: Request) -> Any:
        body = await read_json_object(request)
        _reject_unknown(body, {"type", "model", "params", "inputs", "priority", "idempotency_key", "candidates"})
        params = body.get("params")
        if "candidates" in body:  # the spec lists candidates beside type and params: accept it there too
            if params is None:
                params = {}
            if isinstance(params, dict):
                if params.get("candidates", body["candidates"]) != body["candidates"]:
                    raise ValidationFailed([{"path": "candidates", "message": "conflicts with params.candidates"}])
                params = {**params, "candidates": body["candidates"]}
        rec = await service.submit(
            body.get("type"), params, model=body.get("model"), inputs=body.get("inputs"),
            priority="normal" if body.get("priority") is None else body["priority"],
            idempotency_key=body.get("idempotency_key"),
        )
        return _json({"job_id": rec.job_id, "state": rec.state}, 202)

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> Any:
        return service.get_job(job_id).model_dump(mode="json")

    @router.get("/jobs")
    async def list_jobs(state: str | None = Query(None, max_length=16), limit: int = Query(50, ge=1, le=200)) -> Any:
        return [rec.model_dump(mode="json") for rec in service.list_jobs(state, limit)]

    @router.delete("/jobs/{job_id}")
    async def cancel_job(job_id: str) -> Any:
        return (await service.cancel(job_id)).model_dump(mode="json")

    # -- events --

    @router.get("/events")
    async def events(job_id: str | None = Query(None, max_length=64)) -> Any:
        stream = EventStream(app, service, job_id)
        return StreamingResponse(
            stream.frames(), media_type="text/event-stream", background=BackgroundTask(stream.close),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- assets --

    @router.api_route("/assets/{asset_id}", methods=["GET", "HEAD"])
    async def download_asset(asset_id: str, request: Request) -> Any:
        return asset_response(request, asset_id, service.asset_path(asset_id))

    @router.get("/assets/{asset_id}/meta")
    async def asset_meta(asset_id: str) -> Any:
        return await asyncio.to_thread(service.asset_meta, asset_id)

    @router.get("/assets")
    async def search_assets(
        tags: list[str] = Query(default_factory=list, max_length=64),
        text: str | None = Query(None, max_length=200),
        type: str | None = Query(None, max_length=64),
        model: str | None = Query(None, max_length=64),
        job_id: str | None = Query(None, max_length=64),
        rated: bool | None = None,
        min_rating: int | None = Query(None, ge=0, le=5),
        since: str | None = Query(None, max_length=64),
        limit: int = Query(50, ge=1, le=200),
    ) -> Any:
        return await asyncio.to_thread(
            lambda: service.search_assets(
                tags=parse_tag_filters(tags), text=text, type=type, model=model, job_id=job_id,
                rated=rated, min_rating=min_rating, since=since, limit=limit,
            )
        )

    @router.post("/assets/{asset_id}/rate")
    async def rate_asset(asset_id: str, request: Request) -> Any:
        body = await read_json_object(request, 64 * 1024)
        _reject_unknown(body, {"rating", "note"})
        if "rating" not in body:
            raise ValidationFailed([{"path": "rating", "message": "required"}])
        side = await asyncio.to_thread(service.rate_asset, asset_id, body["rating"], body.get("note"))
        return {"asset_id": side["asset_id"], "rating": side["rating"], "note": side.get("note")}

    app.include_router(public)
    app.include_router(router)
    return app


__all__ = ["create_app", "end_event_streams", "EventStream", "SafeJSONResponse", "error_response", "install_error_handlers", "read_body"]
