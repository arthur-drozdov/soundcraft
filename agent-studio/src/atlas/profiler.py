"""The Plugin Atlas profiler (DESIGN 6.2 steps 1-5): a draft profile from a running SoundCraft.

For one plugin it

1. opens a scratch session (never the user's real one, see ``ScratchSession``) with the plugin
   on an audio track (effects) or an instrument track,
2. reads identity and the raw parameter table with ``plugin.describe`` (provenance ``host``),
3. calibrates each parameter with ``plugin.param_text`` (11 points, 21 for continuous
   parameters with units) and infers a unit and taper from the texts,
4. measures behaviour offline: every parameter at min / mid / max with the others at their
   defaults, bounce a synthesised test signal (pink noise, log sine sweep and a drum pattern for
   effects; a sustained chord and a staccato note for instruments), analyse with ``audiokit``
   (level, centroid, bandwidth, decay) and summarise what changed (provenance ``measured``),
5. returns a draft that validates against the schema; ``write_draft`` saves it under the
   ``atlas.profiles`` naming scheme.

Only ``execute`` and ``inspect`` of a ``studio_client.Studio`` are used, so any object with those
(plus ``kind``) works. Nothing is swept during live playback: every measurement is a bounce.

Conventions worth knowing:

* ``calibration[].value`` is the parameter value in the units ``mix.insert_param`` takes (the
  ``used`` value from ``plugin.param_text``); ``text`` is what the plugin shows for it.
* ``taper`` describes how the displayed value changes along the control travel. The host's taper
  is kept when the displayed texts agree or say nothing; when they clearly follow the other
  shape (e.g. a hosted plugin that the host maps linearly but whose Hz texts form a geometric
  progression), the text-derived taper is recorded as ``inferred`` and a warning is added.
* A parameter with no measurable change gets ``does: "no measurable effect in default mode"``
  with provenance ``measured`` and confidence 0.3.
"""

from __future__ import annotations

import datetime
import logging
import math
import platform
import re
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from audiokit import analysis, dsp, wav
from studio_client import StudioError

from . import profiles, testsignals
from .validate import ProfileError, check_profile

log = logging.getLogger(__name__)

MAX_PARAMS = 4096
"""The schema's cap on ``params``; a plugin with more is truncated (with a warning)."""
MAX_PLUGIN_ID = 300
CALIBRATION_POINTS = 11
CALIBRATION_POINTS_CONTINUOUS = 21
MAX_TEXT = 200
SILENT_DB = -80.0
CLIP_DB = -0.1
ABORT_AFTER_FAILURES = 5
MEASURE_REF = "offline bounces of synthesised test signals at min/mid/max, other parameters at default (atlas profile)"
HOST = {"source": "host", "confidence": 1.0}

_UNIT_NAMES = {
    "none": "none", "db": "dB", "hz": "Hz", "ms": "ms", "percent": "percent", "ratio": "ratio",
    "semitones": "semitones", "cents": "cents", "seconds": "seconds", "toggle": "bool", "choice": "choice",
}
_UNIT_SUFFIX = {
    "dB": " dB", "Hz": " Hz", "ms": " ms", "percent": "%", "ratio": ":1", "semitones": " st", "cents": " ct", "seconds": " s",
}
_CONTINUOUS_NO_UNIT = {"none", "bool", "choice"}


class ProfilerError(Exception):
    """The profiler could not finish (bad request, unsafe session, plugin missing, studio gone)."""


class StudioLike(Protocol):
    """The slice of ``studio_client.Studio`` the profiler uses."""

    @property
    def kind(self) -> str: ...

    def execute(self, command: str, params: Mapping[str, Any] | None = None) -> Any: ...

    def inspect(self, detail: str = "summary") -> Any: ...


# ---- options and results ---------------------------------------------------------------------------


@dataclass
class ProfilerOptions:
    """Knobs of one profiling run."""

    measure: bool = True
    max_params: int = 128
    """Most parameters measured (calibration still covers all); the rest are left unmeasured."""
    max_bounce_s: float = 9.0
    """Cap on the length of every bounce, which also sets the length of the test signals (3..60)."""
    only_params: Sequence[str] | None = None
    """Measure only these parameter ids."""
    allow_current_session: bool = False
    """Allow working inside the open session of a desktop app (see ``ScratchSession``)."""
    date: str | None = None
    """``identity.profiled`` (ISO date); default today."""
    progress: Callable[[str], None] | None = None
    work_dir: str | Path | None = None
    """Folder for the test signals (default: a temp folder removed afterwards; a given folder is left in place)."""


@dataclass
class ProfileResult:
    profile: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    measured: int = 0
    unmeasured: int = 0
    bounces: int = 0
    elapsed_s: float = 0.0
    scratch_mode: str = ""
    path: Path | None = None


# ---- what plugin.describe says -----------------------------------------------------------------------


@dataclass(frozen=True)
class HostParam:
    id: str
    name: str
    min: float
    max: float
    default: float
    unit: str
    """The host's unit name: ``None``, ``Db``, ``Hz``, ``Ms``, ``Percent``, ``Ratio``, ``Semitones``, ``Cents``, ``Seconds``, ``Toggle`` or ``Choice``."""
    taper: str
    choices: tuple[str, ...]
    value: float
    text: str

    @property
    def profile_unit(self) -> str:
        return profile_unit(self.unit)

    @property
    def stepped(self) -> bool:
        return self.profile_unit in ("bool", "choice")


@dataclass(frozen=True)
class HostPlugin:
    plugin_id: str
    format: str
    name: str
    vendor: str | None
    version: str | None
    category: str | None
    is_instrument: bool
    params: tuple[HostParam, ...]
    warnings: tuple[str, ...] = ()


def profile_unit(raw: str) -> str:
    """The profile vocabulary for a host unit name (``Db`` -> ``dB``, ``Toggle`` -> ``bool``)."""
    key = str(raw).strip().lower()
    return _UNIT_NAMES.get(key, key[:32] or "none")


def profile_taper(raw: str) -> str:
    key = str(raw).strip().lower()
    return key[:32]


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _text(value: Any, limit: int = MAX_TEXT) -> str:
    return str(value).replace("\x00", "")[:limit] if value is not None else ""


def parse_describe(result: Any) -> HostPlugin:
    """Validate and normalise a ``plugin.describe`` result. Malformed parameter rows are skipped."""
    if not isinstance(result, Mapping):
        raise ProfilerError("plugin.describe returned something that is not an object")
    plugin = result.get("plugin")
    if not isinstance(plugin, Mapping):
        raise ProfilerError("plugin.describe returned no `plugin` identity")
    plugin_id = _text(plugin.get("id"), MAX_PLUGIN_ID)
    if not plugin_id:
        raise ProfilerError("plugin.describe returned no plugin id")
    if result.get("available") is False:
        raise ProfilerError(f"plugin `{plugin_id}` is not installed on this machine (the host could not load it)")
    warnings: list[str] = []
    rows = result.get("params")
    if not isinstance(rows, list):
        raise ProfilerError("plugin.describe returned no `params` list")
    params: list[HostParam] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        if len(params) >= MAX_PARAMS:
            warnings.append(f"the plugin has more than {MAX_PARAMS} parameters; only the first {MAX_PARAMS} are profiled")
            break
        if not isinstance(row, Mapping) or isinstance(row.get("id"), bool) or not isinstance(row.get("id"), (str, int)):
            warnings.append(f"params[{i}] skipped: no usable id")
            continue
        pid = _text(row["id"], 200)
        lo, hi, default = _finite(row.get("min")), _finite(row.get("max")), _finite(row.get("default"))
        if not pid or lo is None or hi is None or default is None or lo > hi:
            warnings.append(f"param `{pid}` skipped: min/max/default are missing, not finite or inverted")
            continue
        if pid in seen:
            warnings.append(f"param `{pid}` appears twice in the host table; only the first is profiled")
            continue
        seen.add(pid)
        default = min(max(default, lo), hi)
        value = _finite(row.get("value"))
        choices = row.get("choices")
        params.append(
            HostParam(
                id=pid,
                name=_text(row.get("name")),
                min=lo,
                max=hi,
                default=default,
                unit=_text(row.get("unit"), 32) or "None",
                taper=_text(row.get("taper"), 32),
                choices=tuple(_text(c) for c in choices[:1024]) if isinstance(choices, list) else (),
                value=min(max(value, lo), hi) if value is not None else default,
                text=_text(row.get("text")),
            )
        )
    return HostPlugin(
        plugin_id=plugin_id,
        format=_text(plugin.get("format"), 16) or _format_of(plugin_id),
        name=_text(plugin.get("name")) or plugin_id,
        vendor=_text(plugin.get("vendor")) or None,
        version=_text(plugin.get("version"), 64) or None,
        category=_text(plugin.get("category"), 64) or None,
        is_instrument=bool(plugin.get("is_instrument")),
        params=tuple(params),
        warnings=tuple(warnings),
    )


def _format_of(plugin_id: str) -> str:
    prefix = plugin_id.split(":", 1)[0]
    return prefix if prefix in ("clap", "vst3", "au") else "builtin"


def current_os() -> str:
    """The profile ``os`` name of the machine running the profiler."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system.startswith("win"):
        return "windows"
    if "bsd" in system:
        return "freebsd"
    return "linux"


# ---- the host-derived profile ------------------------------------------------------------------------


def build_host_profile(
    host: HostPlugin,
    *,
    profiled: str,
    os_names: Sequence[str] | None = None,
    os_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A profile holding only what the host reported (identity and the raw parameter table)."""
    prov: dict[str, Any] = {}
    identity: dict[str, Any] = {
        "name": host.name[:200] or host.plugin_id[:200],
        "vendor": host.vendor or "unknown",
        "format": host.format if host.format in ("builtin", "clap", "vst3", "au") else _format_of(host.plugin_id),
        "plugin_id": host.plugin_id,
        "version": host.version or "unknown",
        "os": list(os_names) if os_names else [current_os()],
        "profiled": profiled,
    }
    for key in ("name", "format", "plugin_id"):
        prov[f"identity.{key}"] = dict(HOST)
    for key, value in (("vendor", host.vendor), ("version", host.version)):
        if value:
            prov[f"identity.{key}"] = dict(HOST)
        else:
            prov[f"identity.{key}"] = {"source": "inferred", "ref": f"the host did not report a {key}", "confidence": 0.2}
    prov["identity.os"] = dict(os_provenance) if os_provenance else {"source": "host", "ref": f"profiled on {current_os()}", "confidence": 1.0}

    params: list[dict[str, Any]] = []
    for hp in host.params:
        entry: dict[str, Any] = {"id": hp.id, "host_name": hp.name}
        pp: dict[str, Any] = {"host_name": dict(HOST)}
        entry["unit"] = hp.profile_unit
        pp["unit"] = dict(HOST)
        entry["min"], entry["max"], entry["default"] = hp.min, hp.max, hp.default
        for field_name in ("min", "max", "default"):
            pp[field_name] = dict(HOST)
        taper = profile_taper(hp.taper)
        if taper:
            entry["taper"] = taper
            pp["taper"] = dict(HOST)
        if hp.choices:
            entry["choices"] = list(hp.choices)
            pp["choices"] = dict(HOST)
        entry["provenance"] = pp
        params.append(entry)

    kind = "instrument" if host.is_instrument else "effect"
    category = host.category if host.category and host.category != "Other" and host.category.lower() != kind else None
    label = f"{category} {kind}" if category else kind
    count = f"{len(params)} parameter" + ("" if len(params) == 1 else "s")
    profile: dict[str, Any] = {
        "identity": identity,
        "summary": f"{label[0].upper()}{label[1:]} with {count}.",
        "params": params,
        "provenance": prov,
    }
    profile["provenance"]["summary"] = dict(HOST)
    return profile


# ---- calibration and inference -----------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationPoint:
    t: float
    """Control travel 0..1 this point was asked at."""
    value: float
    text: str


_TEXT_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(.*?)\s*$")
_TOKENS: tuple[tuple[tuple[str, ...], str, float], ...] = (
    (("khz",), "Hz", 1000.0),
    (("hz",), "Hz", 1.0),
    (("ms",), "ms", 1.0),
    (("s", "sec", "secs", "second", "seconds"), "seconds", 1.0),
    (("db", "dbfs", "dbu", "dbv", "db fs"), "dB", 1.0),
    (("%", "percent"), "percent", 1.0),
    ((":1", ": 1", ":1.0"), "ratio", 1.0),
    (("st", "semi", "semis", "semitone", "semitones"), "semitones", 1.0),
    (("ct", "cts", "cent", "cents"), "cents", 1.0),
    (("oct", "octave", "octaves"), "octaves", 1.0),
    (("",), "", 1.0),
)


def parse_value_text(text: str) -> tuple[float, str] | None:
    """``(number in base units, unit name)`` from a parameter text like ``2.40 kHz`` or ``4.0:1``.

    The unit is ``""`` for a bare number and ``"?"`` for a suffix this parser does not know.
    ``None`` when the text does not start with a number (``-inf dB``, ``Off``, ``C#3``).
    """
    m = _TEXT_RE.match(text)
    if m is None:
        return None
    try:
        number = float(m.group(1))
    except ValueError:
        return None
    suffix = m.group(2).strip().lower()
    for names, unit, factor in _TOKENS:
        if suffix in names:
            return number * factor, unit
    return number, "?"


@dataclass(frozen=True)
class TextInference:
    unit: str | None = None
    unit_confidence: float = 0.0
    taper: str | None = None
    taper_confidence: float = 0.0
    r2_log: float | None = None
    r2_linear: float | None = None


def _r2(t: np.ndarray, y: np.ndarray) -> float | None:
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0.0:
        return None
    a = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    ss_res = float(np.sum((y - a @ coef) ** 2))
    return 1.0 - ss_res / ss_tot


def infer_from_texts(points: Sequence[CalibrationPoint]) -> TextInference:
    """Infer a unit and a taper from the calibration texts.

    Unit: the unit suffix shared by at least 70 % of at least 3 parsable texts. Taper: over the
    points of that unit, log if the values form a geometric progression along the travel
    (constant ratios: ``ln(value)`` is a straight line of ``t``), linear if they form an
    arithmetic one; nothing when neither fits clearly (R^2 >= 0.995 and 0.002 better than
    the other) or the values barely move.
    """
    parsed = [(p.t, parse_value_text(p.text)) for p in points]
    parsed_ok = [(t, pv) for t, pv in parsed if pv is not None]
    if len(parsed_ok) < 3:
        return TextInference()
    counts = Counter(pv[1] for _, pv in parsed_ok)
    token, n = counts.most_common(1)[0]
    if n < 3 or n < 0.7 * len(parsed_ok) or token == "?":
        return TextInference()
    unit = token if token else None
    t = np.array([t for t, pv in parsed_ok if pv[1] == token], dtype=np.float64)
    x = np.array([pv[0] for _, pv in parsed_ok if pv[1] == token], dtype=np.float64)
    unit_conf = 0.7 if unit else 0.0
    if len(x) < 5 or float(np.ptp(x)) <= 1e-9 * max(1.0, float(np.max(np.abs(x)))):
        return TextInference(unit, unit_conf)
    r2_lin = _r2(t, x)
    r2_log = _r2(t, np.log(x)) if bool(np.all(x > 0)) and float(x.max() / x.min()) >= 1.2 else None
    taper: str | None = None
    conf = 0.0
    if r2_lin is not None:
        if r2_log is not None and r2_log >= 0.995 and r2_log > r2_lin + 0.002:
            taper, conf = "log", 0.85 if r2_log >= 0.999 else 0.7
        elif r2_lin >= 0.995 and (r2_log is None or r2_lin > r2_log + 0.002):
            taper, conf = "linear", 0.85 if r2_lin >= 0.999 else 0.7
    return TextInference(unit, unit_conf, taper, conf, r2_log, r2_lin)


def plain_from_travel(hp: HostParam, t: float) -> float:
    """The parameter value at control travel ``t`` (0..1) under the host's taper."""
    t = min(max(t, 0.0), 1.0)
    if profile_taper(hp.taper) == "log" and hp.min > 0:
        value = hp.min * (hp.max / hp.min) ** t
    else:
        value = hp.min + (hp.max - hp.min) * t
    if hp.stepped:
        value = math.floor(value + 0.5)
    return min(max(value, hp.min), hp.max)


def _rig_params(rig: Rig) -> dict[str, Any]:
    return {"track": rig.track, "instrument": True} if rig.instrument else {"track": rig.track, "slot": rig.slot}


def calibrate(studio: StudioLike, rig: Rig, hp: HostParam) -> list[CalibrationPoint]:
    """Ask ``plugin.param_text`` for the parameter's text along its travel.

    11 points (21 for continuous parameters with units). Toggles are asked at their two values;
    choice parameters that already carry their labels are not asked at all. The request uses
    ``normalized: true`` (control travel, honouring the taper) and falls back to explicit values
    computed here if the host rejects that field.
    """
    unit = hp.profile_unit
    if unit == "choice" and hp.choices:
        return []
    if hp.max <= hp.min:
        return []
    if unit == "bool":
        ts = [0.0, 1.0]
    else:
        n = CALIBRATION_POINTS if unit in _CONTINUOUS_NO_UNIT else CALIBRATION_POINTS_CONTINUOUS
        ts = [i / (n - 1) for i in range(n)]
    base = {**_rig_params(rig), "param": hp.id}
    result: Any
    if hp.stepped:
        values = [plain_from_travel(hp, t) for t in ts]
        result = studio.execute("plugin.param_text", {**base, "values": values})
    else:
        try:
            result = studio.execute("plugin.param_text", {**base, "values": ts, "normalized": True})
        except StudioError as exc:
            log.debug("plugin.param_text with normalized failed for %s (%s); using explicit values", hp.id, exc)
            result = studio.execute("plugin.param_text", {**base, "values": [plain_from_travel(hp, t) for t in ts]})
    return _calibration_points(result, ts, hp)


def _calibration_points(result: Any, ts: Sequence[float], hp: HostParam) -> list[CalibrationPoint]:
    if not isinstance(result, Mapping):
        return []
    rows = result.get("results")
    texts = result.get("texts")
    points: list[CalibrationPoint] = []
    for i, t in enumerate(ts):
        used: float | None = None
        text: Any = None
        if isinstance(rows, list) and i < len(rows) and isinstance(rows[i], Mapping):
            used, text = _finite(rows[i].get("used")), rows[i].get("text")
        if text is None and isinstance(texts, list) and i < len(texts):
            text = texts[i]
        if used is None:
            used = plain_from_travel(hp, t)
        if not isinstance(text, str) or not text.strip():
            continue
        point = CalibrationPoint(t, round(used, 9), text.strip()[:MAX_TEXT])
        if points and points[-1].value == point.value and points[-1].text == point.text:
            continue
        points.append(point)
    return points


# ---- scratch session --------------------------------------------------------------------------------


@dataclass
class Rig:
    """The track and slot the plugin under test sits on."""

    track: int
    instrument: bool
    slot: int = 0
    plugin_id: str = ""


class ScratchSession:
    """A throw-away session to profile in; leaves the user's real session as it found it.

    * A headless ``Studio`` is our own process: ``session.new`` starts a scratch session and
      ``session.close`` ends it.
    * A control-channel ``Studio`` is the desktop app, which has exactly one session and no
      command to open a second one (``session.new`` would replace the user's work). So:
      an *untouched* session (no tracks, not modified, never saved) is used as scratch and
      replaced by an identical fresh one at the end; anything else is refused with a clear
      error unless ``allow_current_session`` is set, in which case the scratch tracks are
      created in the open session and then undone with ``edit.undo``. That leaves the content
      as it was, but the session is marked modified and its redo list holds the scratch steps.
    """

    def __init__(self, studio: StudioLike, *, allow_current_session: bool = False, name: str = "Atlas scratch") -> None:
        self.studio = studio
        self.allow_current_session = allow_current_session
        self.name = name
        self.mode = ""
        self.problems: list[str] = []
        self._baseline_undo = 0
        self._restore: dict[str, Any] = {}
        self._entered = False

    def __enter__(self) -> ScratchSession:
        kind = getattr(self.studio, "kind", "")
        if kind == "headless":
            self.studio.execute("session.new", {"name": self.name})
            self.mode = "new"
        else:
            self._enter_shared()
        self._entered = True
        return self

    def _enter_shared(self) -> None:
        state = self.studio.inspect("summary")
        if not isinstance(state, Mapping):
            raise ProfilerError("session.inspect returned something unexpected; cannot tell whether the open session is safe to use")
        tracks = state.get("tracks")
        pristine = isinstance(tracks, list) and not tracks and state.get("dirty") is False and not state.get("path")
        if pristine:
            rate = _finite(state.get("sample_rate"))
            self._restore = {"name": str(state.get("name") or "Untitled")}
            if rate:
                self._restore["sample_rate"] = int(rate)
            self.mode = "pristine"
            return
        if not self.allow_current_session:
            raise ProfilerError(
                "the connected SoundCraft app has a session open and the control channel cannot create a separate scratch "
                "session (session.new would replace your work). Profile headless instead (--headless soundcraft-cli), or save "
                "and close your session, or pass allow_current_session (CLI: --allow-current-session) to work inside the open "
                "session: the scratch tracks are removed afterwards with edit.undo, but the session is left marked as modified"
            )
        history = self.studio.execute("engine.history")
        undo = history.get("undo") if isinstance(history, Mapping) else None
        if not isinstance(undo, list):
            raise ProfilerError("engine.history returned something unexpected; cannot undo scratch work safely")
        self._baseline_undo = len(undo)
        self.mode = "current"

    def _undo_depth(self) -> int:
        history = self.studio.execute("engine.history")
        undo = history.get("undo") if isinstance(history, Mapping) else None
        return len(undo) if isinstance(undo, list) else -1

    def __exit__(self, *exc: object) -> None:
        if not self._entered:
            return
        try:
            if self.mode == "new":
                self.studio.execute("session.close")
            elif self.mode == "pristine":
                self.studio.execute("session.new", self._restore)
            elif self.mode == "current":
                depth = self._undo_depth()
                for _ in range(max(0, depth - self._baseline_undo)):
                    self.studio.execute("edit.undo")
                after = self._undo_depth()
                if after != self._baseline_undo:
                    self.problems.append(
                        f"could not undo all scratch work: the undo list has {after} entries, expected {self._baseline_undo}; "
                        "check the open session for leftover 'Atlas scratch' tracks"
                    )
        except StudioError as err:
            self.problems.append(f"cleaning up the scratch session failed: {err}")
            log.warning("scratch cleanup failed: %s", err)


# ---- finding and inserting the plugin ----------------------------------------------------------------


def _lookup_instrument(studio: StudioLike, plugin_id: str) -> bool | None:
    """Whether ``plugin_id`` is an instrument according to the plugin lists; ``None`` if unknown."""
    try:
        if plugin_id.startswith("vst3:"):
            rows = studio.execute("engine.vst3_plugins")
        elif plugin_id.startswith("clap:"):
            rows = studio.execute("engine.clap_plugins")
        elif plugin_id.startswith("au:"):
            return None
        else:
            rows = studio.execute("engine.plugins")
    except StudioError as exc:
        log.debug("plugin list unavailable: %s", exc)
        return None
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, Mapping) and row.get("id") == plugin_id:
            flag = row.get("instrument", row.get("is_instrument"))
            return bool(flag) if isinstance(flag, bool) else None
    return None


def _track_id(result: Any) -> int:
    tracks = result.get("tracks") if isinstance(result, Mapping) else None
    if isinstance(tracks, list) and tracks and isinstance(tracks[0], int) and not isinstance(tracks[0], bool):
        return tracks[0]
    raise ProfilerError("track.new did not return a track id")


def _try_rig(studio: StudioLike, plugin_id: str, instrument: bool) -> Rig:
    if instrument:
        track = _track_id(studio.execute("track.new", {"kind": "instrument", "format": "stereo", "name": "Atlas scratch"}))
        studio.execute("mix.instrument", {"track": track, "plugin": plugin_id})
    else:
        track = _track_id(studio.execute("track.new", {"kind": "audio", "format": "stereo", "name": "Atlas scratch"}))
        studio.execute("mix.insert", {"track": track, "slot": 0, "plugin": plugin_id})
    return Rig(track=track, instrument=instrument, slot=0, plugin_id=plugin_id)


def build_rig(studio: StudioLike, plugin_id: str, prefer_instrument: bool | None = None) -> Rig:
    """Put the plugin on a fresh scratch track: an instrument track for instruments, an audio
    track (insert slot 0) for effects. With no hint, effect first, then instrument."""
    order = [True, False] if prefer_instrument else [False, True]
    errors: list[str] = []
    for instrument in order:
        try:
            return _try_rig(studio, plugin_id, instrument)
        except StudioError as exc:
            errors.append(str(exc))
    raise ProfilerError(f"cannot load `{plugin_id}` into a scratch track: {errors[0]}")


def discard_rig(studio: StudioLike, rig: Rig) -> None:
    """Remove a scratch track that turned out to be the wrong kind (best effort)."""
    try:
        studio.execute("track.delete", {"tracks": [rig.track]})
    except StudioError as exc:
        log.debug("could not remove scratch track %s: %s", rig.track, exc)


def isolate_rig(studio: StudioLike, rig: Rig) -> list[str]:
    """In a session that is not ours: solo the scratch track so a bounce holds only its audio.

    Returns warnings. The solo is one more undoable step, removed with the rest of the scratch work.
    """
    notes = [
        "measured inside the open session: processing on the master or busses it feeds also colours the measurements"
    ]
    try:
        studio.execute("mix.solo", {"track": rig.track, "value": True, "exclusive": True})
    except StudioError as exc:
        notes.append(f"could not solo the scratch track ({exc}); the bounces may contain the audio of your tracks")
    return notes


def describe_plugin(studio: StudioLike, rig: Rig) -> HostPlugin:
    return parse_describe(studio.execute("plugin.describe", _rig_params(rig)))


def display_text(studio: StudioLike, rig: Rig, hp: HostParam, value: float) -> str:
    """The plugin's own text for ``value`` of ``hp`` (empty if it cannot be read)."""
    try:
        result = studio.execute("plugin.param_text", {**_rig_params(rig), "param": hp.id, "values": [value]})
    except StudioError:
        return ""
    texts = result.get("texts") if isinstance(result, Mapping) else None
    if isinstance(texts, list) and texts and isinstance(texts[0], str):
        return texts[0].strip()[:MAX_TEXT]
    return ""


def set_param(studio: StudioLike, rig: Rig, param: str, value: float) -> None:
    if rig.instrument:
        studio.execute("mix.instrument_param", {"track": rig.track, "param": param, "value": value})
    else:
        studio.execute("mix.insert_param", {"track": rig.track, "slot": rig.slot, "param": param, "value": value})


# ---- measurement -------------------------------------------------------------------------------------

Features = dict[str, dict[str, float]]
"""signal name -> metric name -> value (dB for ``*_db`` metrics, Hz for centroid/bandwidth, s for decay)."""

LEVEL_METRICS = ("rms_db", "peak_db")
DECAY_SEGMENTS = {"clicks", "staccato", "chord"}


@dataclass(frozen=True)
class MeasurePlan:
    """What a bounce covers and how to read it."""

    sample_rate: int
    end_samples: int
    windows: dict[str, tuple[float, float]]
    labels: dict[str, str]
    bands: tuple[tuple[float, float, float, float], ...] = ()
    band_segment: str = ""


def _window(audio: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    a = int(max(0.0, lo) * sr)
    b = int(max(lo, hi) * sr)
    return audio[:, a:b]


def analyze_bounce(audio: np.ndarray, sr: int, plan: MeasurePlan) -> Features:
    """Level, peak, centroid, bandwidth and decay per test signal window, plus sweep band levels."""
    out: Features = {}
    for name, (lo, hi) in plan.windows.items():
        seg = _window(audio, sr, lo, hi)
        if seg.shape[1] < 64:
            continue
        rms = dsp.rms_dbfs(seg)
        metrics: dict[str, float] = {"rms_db": round(rms, 3), "peak_db": round(dsp.peak_dbfs(seg), 3)}
        if rms > SILENT_DB:
            metrics["centroid_hz"] = analysis.centroid_hz(seg, sr)
            metrics["bandwidth_hz"] = analysis.bandwidth_hz(seg, sr)
            if name in DECAY_SEGMENTS:
                metrics["decay_s"] = analysis.decay_time_s(seg, sr)
        out[name] = metrics
    if plan.bands and plan.band_segment in out:
        for lo_hz, hi_hz, t0, t1 in plan.bands:
            seg = _window(audio, sr, t0, t1)
            if seg.shape[1] >= 64:
                out[plan.band_segment][band_key(lo_hz, hi_hz)] = round(dsp.rms_dbfs(seg), 3)
    return out


def band_key(lo: float, hi: float) -> str:
    return f"band_{_hz(lo)}-{_hz(hi)}"


def _hz(f: float) -> str:
    return f"{f / 1000:g}k" if f >= 1000 else f"{f:g}"


@dataclass(frozen=True)
class Tolerances:
    level_db: float = 0.25
    octaves: float = 0.05
    decay_s: float = 0.02
    decay_frac: float = 0.15


@dataclass(frozen=True)
class Effect:
    """One measured difference from the default-settings bounce."""

    segment: str
    metric: str
    base: float
    value: float
    delta: float
    """dB for level metrics, octaves for centroid/bandwidth, seconds for decay."""
    significance: float
    """``|delta|`` over the tolerance it had to beat."""
    clipped: bool = False
    """The output of that signal reaches full scale, so a level change may be larger than measured."""


def _is_level(metric: str) -> bool:
    return metric in LEVEL_METRICS or metric.startswith("band_")


def derive_tolerances(*runs: Features) -> tuple[Tolerances, bool]:
    """Tolerances from bounces at identical settings: 4x the largest run-to-run difference, with floors.

    The flag says the bounces differed noticeably (the plugin is not deterministic: noise,
    random modulation), so the tolerances were widened.
    """
    floor = Tolerances()
    level = octave = decay = 0.0
    zero = Tolerances(level_db=0.0, octaves=0.0, decay_s=0.0, decay_frac=0.0)
    for i, first in enumerate(runs):
        for second in runs[i + 1 :]:
            for effect in compare_features(first, second, zero, report_all=True):
                if effect.metric in ("centroid_hz", "bandwidth_hz"):
                    octave = max(octave, abs(effect.delta))
                elif effect.metric == "decay_s":
                    decay = max(decay, abs(effect.delta))
                else:
                    level = max(level, abs(effect.delta))
    tol = Tolerances(
        level_db=max(floor.level_db, 4 * level),
        octaves=max(floor.octaves, 4 * octave),
        decay_s=max(floor.decay_s, 4 * decay),
        decay_frac=floor.decay_frac,
    )
    noisy = level > floor.level_db / 4 or octave > floor.octaves / 4 or decay > floor.decay_s / 4
    return tol, noisy


def compare_features(base: Features, other: Features, tol: Tolerances, *, report_all: bool = False) -> list[Effect]:
    """The differences between two bounces that exceed ``tol``, most significant first."""
    effects: list[Effect] = []
    for segment, metrics in base.items():
        cur = other.get(segment)
        if cur is None:
            continue
        silent = metrics.get("rms_db", 0.0) <= SILENT_DB and cur.get("rms_db", 0.0) <= SILENT_DB
        for metric, b in metrics.items():
            v = cur.get(metric)
            if v is None:
                continue
            if _is_level(metric):
                if silent or (b <= -100.0 and v <= -100.0):
                    continue
                delta, limit = v - b, tol.level_db
            elif metric in ("centroid_hz", "bandwidth_hz"):
                if b <= 0.0 or v <= 0.0:
                    continue
                delta, limit = math.log2(v / b), tol.octaves
            else:
                delta, limit = v - b, max(tol.decay_s, tol.decay_frac * abs(b))
            if abs(delta) >= limit or report_all:
                clipped = _is_level(metric) and cur.get("peak_db", -200.0) >= CLIP_DB
                effects.append(Effect(segment, metric, b, v, delta, abs(delta) / limit if limit > 0 else math.inf, clipped))
    # A segment that went silent (or woke up) shows in the level metrics; centroid etc. are then absent.
    return sorted(effects, key=lambda e: -e.significance)


@dataclass(frozen=True)
class PointResult:
    label: str
    value: float
    effects: tuple[Effect, ...]
    text: str = ""
    """The plugin's own display text for ``value`` (``2.40 kHz``), when it could be read."""


def probe_points(hp: HostParam) -> list[tuple[str, float]]:
    """``min`` / ``mid`` / ``max`` values to try, without duplicates and without the current value.

    Stepped parameters round the middle to a whole step; when that lands on an end it is the
    end's point (labelled ``min`` or ``max``), not a ``mid``.
    """
    if hp.taper.lower() == "log" and hp.min > 0:
        mid = math.sqrt(hp.min * hp.max)
    else:
        mid = (hp.min + hp.max) / 2
    out: list[tuple[str, float]] = []
    for label, value in (("min", hp.min), ("max", hp.max), ("mid", mid)):
        if hp.stepped:
            value = math.floor(value + 0.5)
        value = min(max(value, hp.min), hp.max)
        if math.isclose(value, hp.value, rel_tol=1e-9, abs_tol=1e-12):
            continue
        if any(math.isclose(value, v, rel_tol=1e-9, abs_tol=1e-12) for _, v in out):
            continue
        out.append((label, value))
    return sorted(out, key=lambda lv: lv[1])


def _num_text(value: float) -> str:
    """Four significant digits, no exponent, no trailing zeros (``0.2``, ``1789``, ``16000``)."""
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else str(value)
    decimals = max(0, 3 - int(math.floor(math.log10(abs(value)))))
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_value(hp: HostParam, value: float) -> str:
    """A short human reading of a parameter value."""
    unit = hp.profile_unit
    if unit == "choice" and hp.choices:
        idx = int(value)
        if 0 <= idx < len(hp.choices):
            return hp.choices[idx]
    if unit == "bool":
        return "on" if value >= 0.5 else "off"
    if unit == "Hz" and abs(value) >= 1000:
        return f"{_num_text(value / 1000)} kHz"
    return f"{_num_text(value)}{_UNIT_SUFFIX.get(unit, '')}"


def _point_text(hp: HostParam, point: PointResult) -> str:
    return point.text or format_value(hp, point.value)


def _level_text(effect: Effect) -> str:
    if effect.value <= -100.0:
        return "silent"
    if effect.base <= -100.0:
        return f"audible ({effect.value:.1f} dBFS)"
    return f"{effect.delta:+.1f} dB" + (" (reaches full scale)" if effect.clipped else "")


def _describe_key(effect: Effect, labels: Mapping[str, str]) -> tuple[str, Callable[[Effect], str]]:
    """The heading of a trend and how to print one point of it, for the kind of metric ``effect`` is."""
    sig = labels.get(effect.segment, effect.segment)
    m = effect.metric
    if m in LEVEL_METRICS:
        kind = "RMS" if m == "rms_db" else "peak"
        return f"output level ({sig} {kind}, default {effect.base:.1f} dBFS)", _level_text
    if m.startswith("band_"):
        lo, hi = m[len("band_"):].split("-", 1)
        return f"level at {lo}-{hi} Hz ({sig})", _level_text
    if m == "centroid_hz":
        return f"spectral centroid ({sig}, default {effect.base:.0f} Hz)", lambda e: f"{e.value:.0f} Hz"
    if m == "bandwidth_hz":
        return f"spectral bandwidth ({sig}, default {effect.base:.0f} Hz)", lambda e: f"{e.value:.0f} Hz"
    return f"decay to -40 dB ({sig}, default {effect.base:.2f} s)", lambda e: f"{e.value:.2f} s"


def _family(effect: Effect) -> str:
    if effect.metric in LEVEL_METRICS:
        return "level"
    if effect.metric.startswith("band_"):
        return "bands"
    return effect.metric


def _is_regional(effect: Effect, point: PointResult) -> bool:
    """A sweep-band change that is more than the overall level change at the same point."""
    level = next((e for e in point.effects if e.metric in LEVEL_METRICS), None)
    return level is None or abs(effect.delta - level.delta) >= 1.0


def summarize_measurement(hp: HostParam, points: Sequence[PointResult], labels: Mapping[str, str]) -> tuple[str, float]:
    """The ``does`` text and its confidence for a measured parameter.

    The text lists, for up to three kinds of change (output level, level in one frequency region
    beyond the overall change, brightness, bandwidth, decay), what the measurement was at each
    of min / mid / max relative to the default setting. Strongest change first.
    """
    if not any(p.effects for p in points):
        return "no measurable effect in default mode", 0.3
    best: dict[str, tuple[Effect, ...]] = {}
    strongest: dict[str, float] = {}
    for point in points:
        for effect in point.effects:
            family = _family(effect)
            if family == "bands" and not _is_regional(effect, point):
                continue
            if effect.significance > strongest.get(family, 0.0):
                strongest[family] = effect.significance
                best[family] = (effect,)
    parts: list[str] = []
    for family in sorted(best, key=lambda f: -strongest[f])[:3]:
        key = best[family][0]
        head, fmt = _describe_key(key, labels)
        entries = []
        for point in points:
            hit = next((e for e in point.effects if (e.segment, e.metric) == (key.segment, key.metric)), None)
            entries.append(f"{point.label} ({_point_text(hp, point)}) " + (fmt(hit) if hit else "no change"))
        parts.append(f"{head}: " + ", ".join(entries))
    sig = max(strongest.values()) if strongest else 0.0
    confidence = 0.55 if sig < 2 else 0.65 if sig < 4 else 0.75 if sig < 16 else 0.85
    return ("Measured with the other parameters at default: " + "; ".join(parts) + ".")[:3900], confidence


class _Bouncer:
    """Renders the scratch session over a range and analyses the file."""

    def __init__(self, studio: StudioLike, plan: MeasurePlan, folder: Path) -> None:
        self.studio = studio
        self.plan = plan
        self.folder = folder
        self.count = 0

    def render(self) -> Features:
        self.count += 1
        path = self.folder / "bounce.wav"
        path.unlink(missing_ok=True)
        self.studio.execute(
            "file.bounce_mix",
            {"path": str(path), "start": 0, "end": self.plan.end_samples, "bit_depth": "32f", "dither": False},
        )
        try:
            audio, sr = wav.read(path, max_frames=self.plan.end_samples * 2 + 4096)
        except (OSError, ValueError) as exc:
            raise ProfilerError(f"the bounce could not be read back: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)
        if sr != self.plan.sample_rate:
            raise ProfilerError(f"the bounce has sample rate {sr}, expected {self.plan.sample_rate}")
        if audio.shape[1] < 0.5 * self.plan.end_samples:
            raise ProfilerError(f"the bounce is {audio.shape[1]} frames long, expected about {self.plan.end_samples}")
        return analyze_bounce(audio, sr, self.plan)


def _session_facts(studio: StudioLike) -> tuple[int, float]:
    """(sample rate, bpm) of the scratch session, with safe fallbacks."""
    state = studio.inspect("summary")
    rate, bpm = 48000, 120.0
    if isinstance(state, Mapping):
        r = _finite(state.get("sample_rate"))
        if r and testsignals.MIN_SAMPLE_RATE <= r <= testsignals.MAX_SAMPLE_RATE:
            rate = int(r)
        tempo = state.get("tempo")
        if isinstance(tempo, list) and tempo and isinstance(tempo[0], Mapping):
            b = _finite(tempo[0].get("bpm"))
            if b and 30 <= b <= 300:
                bpm = b
    return rate, bpm


def prepare_measurement(studio: StudioLike, rig: Rig, folder: Path, max_bounce_s: float) -> tuple[MeasurePlan, str]:
    """Load the test material onto the scratch track and return how to read a bounce of it."""
    rate, bpm = _session_facts(studio)
    if rig.instrument:
        iplan = testsignals.build_instrument_plan(bpm, min(max(max_bounce_s, testsignals.MIN_DURATION_S), 6.0))
        end = int(round(iplan.bounce_end_s * rate))
        clip = studio.execute("midi.new_clip", {"track": rig.track, "start": 0, "end": end})
        clip_id = clip.get("clip") if isinstance(clip, Mapping) else None
        if isinstance(clip_id, bool) or not isinstance(clip_id, int):
            raise ProfilerError("midi.new_clip did not return a clip id")
        for call in iplan.note_calls(clip_id):
            studio.execute("midi.note_add", call)
        return (
            MeasurePlan(rate, end, dict(iplan.segments), dict(iplan.names)),
            "instrument",
        )
    signals = testsignals.build_effect_signals(folder, rate, min(max(max_bounce_s, testsignals.MIN_DURATION_S), testsignals.MAX_DURATION_S))
    studio.execute("file.import_audio", {"path": str(signals.path), "track": rig.track, "at": 0})
    eplan = MeasurePlan(
        sample_rate=rate,
        end_samples=int(round(signals.bounce_end_s * rate)),
        windows=dict(signals.segments),
        labels={"noise": "pink noise", "sweep": "sine sweep", "clicks": "drum pattern"},
        bands=tuple(testsignals.sweep_band_windows(signals)),
        band_segment="sweep",
    )
    return eplan, "effect"


@dataclass
class Measurement:
    """What measuring one plugin produced."""

    does: dict[str, tuple[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    bounces: int = 0
    unmeasured: list[str] = field(default_factory=list)
    noisy: bool = False


def measure_params(
    studio: StudioLike,
    rig: Rig,
    host: HostPlugin,
    folder: Path,
    opts: ProfilerOptions,
    progress: Callable[[str], None],
) -> Measurement:
    """Bounce the test material with every parameter at min/mid/max and summarise the changes."""
    out = Measurement()
    wanted = [hp for hp in host.params if hp.max > hp.min]
    if opts.only_params is not None:
        keep = set(opts.only_params)
        wanted = [hp for hp in wanted if hp.id in keep]
    cap = max(0, int(opts.max_params))
    chosen, rest = wanted[:cap], wanted[cap:]
    out.unmeasured = [hp.id for hp in rest]
    if rest:
        out.warnings.append(f"measured {len(chosen)} of {len(wanted)} parameters (max_params={cap}); the rest have no measured effect")
    if not chosen:
        return out
    try:
        plan, kind = prepare_measurement(studio, rig, folder, opts.max_bounce_s)
        bouncer = _Bouncer(studio, plan, folder)
        base = bouncer.render()
        repeats = [bouncer.render(), bouncer.render()]
    except (StudioError, ProfilerError) as exc:
        out.warnings.append(f"measurement skipped: {exc}")
        out.unmeasured = [hp.id for hp in wanted]
        return out
    tol, noisy = derive_tolerances(base, *repeats)
    out.noisy = noisy
    if noisy:
        out.warnings.append("the plugin's output differs between identical bounces (noise or random modulation); measurement tolerances were widened")
    if not any(m.get("rms_db", -200.0) > SILENT_DB for m in base.values()):
        out.warnings.append(f"the {kind} produced silence with default settings; effects can only show as the output becoming audible")
    failures = 0
    for index, hp in enumerate(chosen, 1):
        progress(f"measuring {hp.id} ({index}/{len(chosen)})")
        points: list[PointResult] = []
        try:
            for label, value in probe_points(hp):
                text = display_text(studio, rig, hp, value)
                set_param(studio, rig, hp.id, value)
                feats = bouncer.render()
                points.append(PointResult(label, value, tuple(compare_features(base, feats, tol)), text))
            if points:
                out.does[hp.id] = summarize_measurement(hp, points, plan.labels)
            else:
                out.unmeasured.append(hp.id)
            failures = 0
        except (StudioError, ProfilerError) as exc:
            failures += 1
            out.unmeasured.append(hp.id)
            out.warnings.append(f"param `{hp.id}` not measured: {exc}")
            if "not a built-in instrument" in str(exc):
                out.warnings.append("hosted instruments cannot be set through mix.instrument_param; no parameters of this instrument can be measured")
                out.unmeasured.extend(h.id for h in chosen[index:])
                break
        finally:
            try:
                set_param(studio, rig, hp.id, hp.value)
            except StudioError as exc:
                out.warnings.append(f"could not restore `{hp.id}` to its default: {exc}")
        if failures >= ABORT_AFTER_FAILURES:
            out.warnings.append(f"measurement stopped after {failures} failures in a row")
            out.unmeasured.extend(h.id for h in chosen[index:])
            break
    out.bounces = bouncer.count
    return out


# ---- assembling the draft ---------------------------------------------------------------------------


def _apply_calibration(entry: dict[str, Any], hp: HostParam, points: Sequence[CalibrationPoint], warnings: list[str]) -> None:
    if not points:
        return
    prov = entry["provenance"]
    entry["calibration"] = [{"value": p.value, "text": p.text} for p in points[:256]]
    prov["calibration"] = {"source": "host", "ref": "plugin.param_text", "confidence": 1.0}
    inferred = infer_from_texts(points)
    if inferred.unit and entry.get("unit") == "none":
        entry["unit"] = inferred.unit
        prov["unit"] = {"source": "inferred", "ref": "parsed from the plugin.param_text texts", "confidence": inferred.unit_confidence}
    elif inferred.unit and entry.get("unit") not in (None, inferred.unit) and hp.profile_unit not in ("bool", "choice"):
        warnings.append(f"param `{hp.id}`: host unit `{entry.get('unit')}` but its texts read `{inferred.unit}`; kept the host's")
    if inferred.taper and inferred.taper != entry.get("taper"):
        if "taper" in entry:
            warnings.append(f"param `{hp.id}`: host taper `{entry['taper']}` but the texts follow `{inferred.taper}`; recorded the texts' shape")
        entry["taper"] = inferred.taper
        prov["taper"] = {
            "source": "inferred",
            "ref": "shape of the plugin.param_text texts along the control travel",
            "confidence": inferred.taper_confidence,
        }


def assemble_draft(
    host: HostPlugin,
    calibrations: Mapping[str, Sequence[CalibrationPoint]],
    measurement: Measurement | None,
    *,
    profiled: str,
) -> tuple[dict[str, Any], list[str]]:
    """The draft profile (host fields, calibration, inferred unit/taper, measured effects)."""
    warnings: list[str] = list(host.warnings)
    profile = build_host_profile(host, profiled=profiled)
    by_id = {hp.id: hp for hp in host.params}
    for entry in profile["params"]:
        pid = str(entry["id"])
        hp = by_id[pid]
        _apply_calibration(entry, hp, calibrations.get(pid, ()), warnings)
        if measurement is not None and pid in measurement.does:
            text, confidence = measurement.does[pid]
            entry["does"] = text
            entry["provenance"]["does"] = {"source": "measured", "ref": MEASURE_REF, "confidence": round(confidence, 2)}
    if measurement is not None:
        warnings.extend(measurement.warnings)
    check_profile(profile)
    return profile, warnings


# ---- entry points ------------------------------------------------------------------------------------


def _check_request(plugin_id: Any, opts: ProfilerOptions) -> str:
    if not isinstance(plugin_id, str) or not plugin_id.strip() or len(plugin_id) > MAX_PLUGIN_ID or any(ord(c) < 32 for c in plugin_id):
        raise ProfilerError("plugin id must be a non-empty string without control characters")
    if isinstance(opts.max_params, bool) or not isinstance(opts.max_params, int) or opts.max_params < 0:
        raise ProfilerError("max_params must be a non-negative integer")
    if isinstance(opts.max_bounce_s, bool) or not isinstance(opts.max_bounce_s, (int, float)) or not math.isfinite(opts.max_bounce_s):
        raise ProfilerError("max_bounce_s must be a finite number")
    if not testsignals.MIN_DURATION_S <= opts.max_bounce_s <= testsignals.MAX_DURATION_S:
        raise ProfilerError(f"max_bounce_s must be in {testsignals.MIN_DURATION_S:g}..{testsignals.MAX_DURATION_S:g}")
    if opts.date is not None:
        try:
            datetime.date.fromisoformat(opts.date)
        except ValueError as exc:
            raise ProfilerError(f"date must be YYYY-MM-DD, got {opts.date!r}") from exc
    return plugin_id.strip()


def profile_plugin(studio: StudioLike, plugin_id: str, options: ProfilerOptions | None = None) -> ProfileResult:
    """Profile one plugin on a scratch session of ``studio`` and return the draft.

    Raises ``ProfilerError`` for an unsafe target (see ``ScratchSession``), a plugin the host
    cannot load, or a studio that stops answering. Measurement problems on single parameters
    become warnings, not errors.
    """
    opts = options or ProfilerOptions()
    plugin_id = _check_request(plugin_id, opts)
    progress = opts.progress or (lambda message: log.info("%s", message))
    started = time.monotonic()
    folder = Path(opts.work_dir) if opts.work_dir else Path(tempfile.mkdtemp(prefix="atlas-profile-"))
    folder.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    try:
        try:
            with ScratchSession(studio, allow_current_session=opts.allow_current_session) as scratch:
                progress(f"scratch session ready ({scratch.mode}); loading {plugin_id}")
                guess = _lookup_instrument(studio, plugin_id)
                rig = build_rig(studio, plugin_id, guess)
                host = describe_plugin(studio, rig)
                if host.is_instrument != rig.instrument:
                    progress("the plugin is " + ("an instrument" if host.is_instrument else "an effect") + "; rebuilding the scratch track")
                    discard_rig(studio, rig)
                    rig = build_rig(studio, plugin_id, host.is_instrument)
                    host = describe_plugin(studio, rig)
                if scratch.mode == "current":
                    problems.extend(isolate_rig(studio, rig))
                calibrations: dict[str, list[CalibrationPoint]] = {}
                for i, hp in enumerate(host.params, 1):
                    progress(f"calibrating {hp.id} ({i}/{len(host.params)})")
                    try:
                        calibrations[hp.id] = calibrate(studio, rig, hp)
                    except StudioError as exc:
                        problems.append(f"param `{hp.id}` not calibrated: {exc}")
                measurement = measure_params(studio, rig, host, folder, opts, progress) if opts.measure else None
        except StudioError as exc:
            raise ProfilerError(str(exc)) from exc
        problems.extend(scratch.problems)
        profile, warnings = assemble_draft(host, calibrations, measurement, profiled=opts.date or datetime.date.today().isoformat())
    finally:
        if not opts.work_dir:
            shutil.rmtree(folder, ignore_errors=True)
    warnings = problems + warnings
    measured = len(measurement.does) if measurement else 0
    return ProfileResult(
        profile=profile,
        warnings=warnings,
        measured=measured,
        unmeasured=len(host.params) - measured,
        bounces=measurement.bounces if measurement else 0,
        elapsed_s=round(time.monotonic() - started, 2),
        scratch_mode=scratch.mode,
    )


def write_draft(profile: dict[str, Any], out: str | Path | None = None) -> Path:
    """Write the draft and return its path.

    ``out`` ending in ``.yaml``/``.yml`` is the file; anything else is a root folder and the
    file goes to its ``<format>/<vendor>/<name>@<major>.yaml`` place. Default root: ``atlas-drafts``
    in the current folder.
    """
    target = Path(out) if out else Path("atlas-drafts")
    if target.suffix.lower() in (".yaml", ".yml"):
        return profiles.write_profile(target, profile)
    return profiles.save_profile(target, profile)


__all__ = [
    "CalibrationPoint", "Effect", "Features", "HostParam", "HostPlugin", "MeasurePlan", "Measurement", "PointResult",
    "ProfileError", "ProfileResult", "ProfilerError", "ProfilerOptions", "Rig", "ScratchSession", "StudioLike",
    "TextInference", "Tolerances", "analyze_bounce", "assemble_draft", "build_host_profile", "build_rig", "calibrate",
    "compare_features", "current_os", "derive_tolerances", "describe_plugin", "infer_from_texts", "measure_params",
    "parse_describe", "parse_value_text", "plain_from_travel", "probe_points", "profile_plugin", "profile_unit",
    "summarize_measurement", "write_draft",
]
