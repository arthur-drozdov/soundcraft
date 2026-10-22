"""A fake SoundCraft for the profiler tests: the command surface the profiler uses, with real audio.

``FakeStudio`` implements ``execute`` / ``inspect`` / ``kind`` / ``close`` like
``studio_client.Studio``. Its plugins really process audio (imported WAV files for effects,
MIDI notes for instruments), so bounces carry measurable effects:

* ``fake_gain``   - ``gain`` (dB, scales the level), ``mute`` (toggle, silences), ``dither`` (does nothing)
* ``fake_filter`` - ``cutoff`` (Hz, log taper, a low-pass: the spectral centroid follows it), ``smoothing`` (does nothing)
* ``vst3:FAKEHOSTED0001`` - a "hosted" low-pass whose host range is 0..1 with no unit and a linear
  taper, while its own value text is a geometric progression of Hz (what a VST3 filter looks like)
* ``fake_synth``  - instrument: ``level`` (dB), ``cutoff`` (Hz), ``release`` (seconds), ``glide`` (does nothing)

Command shapes follow ``docs/control-protocol.md`` and the Rust command specs.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from audiokit import dsp, wav
from studio_client import StudioError

SR = 48000
STEPPED = ("Toggle", "Choice")


@dataclass
class FakeParam:
    id: str
    name: str
    min: float
    max: float
    default: float
    unit: str = "None"
    taper: str = "Linear"
    choices: tuple[str, ...] = ()
    plugin_text: Callable[[float], str] | None = None
    """The plugin's own value-to-text (hosted plugins); built-ins use the generic format."""

    def clamp(self, v: float) -> float:
        v = min(max(v, self.min), self.max)
        return float(round(v)) if self.unit in STEPPED else v

    def from_normalized(self, t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        if self.taper == "Log" and self.min > 0:
            return self.clamp(self.min * (self.max / self.min) ** t)
        return self.clamp(self.min + (self.max - self.min) * t)

    def format(self, v: float) -> str:
        v = self.clamp(v)
        if self.plugin_text is not None:
            return self.plugin_text(v)
        if self.unit == "Choice":
            return self.choices[int(v)] if 0 <= int(v) < len(self.choices) else f"{v}"
        if self.unit == "Toggle":
            return "On" if v >= 0.5 else "Off"
        if self.unit == "Db":
            return "-inf dB" if v <= -95.9 else f"{v:.1f} dB"
        if self.unit == "Hz":
            return f"{v / 1000:.2f} kHz" if v >= 1000 else f"{v:.1f} Hz"
        if self.unit == "Ms":
            return f"{v:.1f} ms"
        if self.unit == "Percent":
            return f"{v:.0f}%"
        if self.unit == "Seconds":
            return f"{v:.2f} s"
        return f"{v:.2f}"


@dataclass
class FakePlugin:
    id: str
    name: str
    category: str
    params: list[FakeParam]
    instrument: bool = False
    vendor: str | None = "SoundCraft"
    version: str | None = "0.10.0"
    process: Callable[[np.ndarray, int, Mapping[str, float]], np.ndarray] | None = None
    synth: Callable[[list[tuple[int, float, float, int]], int, int, Mapping[str, float]], np.ndarray] | None = None
    """Instruments: ``(notes[(pitch, start_s, length_s, velocity)], frames, sr, params) -> mono audio``."""

    @property
    def fmt(self) -> str:
        return self.id.split(":", 1)[0] if ":" in self.id else "builtin"

    def info_row(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "category": self.category, "instrument": self.instrument,
            "params": [_param_json(p) for p in self.params],
        }


def _param_json(p: FakeParam) -> dict[str, Any]:
    return {
        "id": p.id, "name": p.name, "min": p.min, "max": p.max, "default": p.default, "unit": p.unit,
        "taper": p.taper, "choices": list(p.choices),
    }


# ---- the fake plugins' DSP --------------------------------------------------------------------------


def _gain(audio: np.ndarray, sr: int, p: Mapping[str, float]) -> np.ndarray:
    if p["mute"] >= 0.5:
        return np.zeros_like(audio)
    return (audio * 10.0 ** (p["gain"] / 20.0)).astype(np.float32)


def _lowpass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    cutoff = min(max(cutoff, 20.0), 0.45 * sr)
    sos = signal.butter(2, cutoff / (sr / 2), btype="lowpass", output="sos")
    return signal.sosfilt(sos, audio, axis=1).astype(np.float32)


def _filter(audio: np.ndarray, sr: int, p: Mapping[str, float]) -> np.ndarray:
    return _lowpass(audio, sr, p["cutoff"])


def _hosted_cutoff(t: float) -> float:
    return 20.0 * 1000.0**t


def _hosted(audio: np.ndarray, sr: int, p: Mapping[str, float]) -> np.ndarray:
    return _lowpass(audio, sr, _hosted_cutoff(p["freq"])) * np.float32(10.0 ** ((-24.0 + 36.0 * p["level"]) / 20.0))


def _synth(notes: list[tuple[int, float, float, int]], frames: int, sr: int, p: Mapping[str, float]) -> np.ndarray:
    out = np.zeros(frames, dtype=np.float64)
    release = max(p["release"], 0.005)
    for pitch, start, length, velocity in notes:
        f0 = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
        a = int(start * sr)
        b = min(frames, int((start + length + release * 8) * sr))
        if a >= frames or b <= a:
            continue
        t = np.arange(b - a) / sr
        tone = sum(np.sin(2 * math.pi * f0 * h * t) / h for h in range(1, 8) if f0 * h < 0.45 * sr)
        env = np.minimum(1.0, t / 0.005)
        env = np.where(t <= length, env, np.exp(-(t - length) / (release / 4.6)))
        out[a:b] += tone * env * (velocity / 127.0)
    mono = _lowpass(out[np.newaxis, :].astype(np.float32), sr, p["cutoff"])[0]
    return (mono * 0.1 * 10.0 ** (p["level"] / 20.0)).astype(np.float32)


def make_plugins() -> dict[str, FakePlugin]:
    plugins = [
        FakePlugin(
            "fake_gain", "Fake Gain", "Other",
            [
                FakeParam("gain", "Gain", -60.0, 24.0, 0.0, "Db"),
                FakeParam("mute", "Mute", 0.0, 1.0, 0.0, "Toggle"),
                FakeParam("dither", "Dither", 0.0, 100.0, 50.0, "Percent"),
            ],
            process=_gain,
        ),
        FakePlugin(
            "fake_filter", "Fake Filter", "Eq",
            [
                FakeParam("cutoff", "Cutoff", 200.0, 16000.0, 8000.0, "Hz", "Log"),
                FakeParam("smoothing", "Smoothing", 0.0, 100.0, 20.0, "Percent"),
            ],
            process=_filter,
        ),
        FakePlugin(
            "vst3:FAKEHOSTED0001", "Hosted LP", "Fx|Filter",
            [
                FakeParam("freq", "Freq", 0.0, 1.0, 1.0, "None", "Linear", plugin_text=lambda v: _hz_text(_hosted_cutoff(v))),
                FakeParam("level", "Level", 0.0, 1.0, 0.667, "None", "Linear", plugin_text=lambda v: f"{-24.0 + 36.0 * v:.1f} dB"),
            ],
            vendor="Acme Audio", version="2.3.1",
            process=_hosted,
        ),
        FakePlugin(
            "fake_synth", "Fake Synth", "Instrument",
            [
                FakeParam("level", "Level", -24.0, 12.0, 0.0, "Db"),
                FakeParam("cutoff", "Cutoff", 100.0, 12000.0, 4000.0, "Hz", "Log"),
                FakeParam("release", "Release", 0.02, 2.0, 0.2, "Seconds", "Log"),
                FakeParam("glide", "Glide", 0.0, 100.0, 0.0, "Percent"),
            ],
            instrument=True,
            synth=_synth,
        ),
    ]
    return {p.id: p for p in plugins}


def _hz_text(hz: float) -> str:
    return f"{hz / 1000:.2f} kHz" if hz >= 1000 else f"{hz:.1f} Hz"


# ---- the studio ---------------------------------------------------------------------------------------


@dataclass
class _Track:
    id: int
    name: str
    kind: str
    insert: dict[str, Any] | None = None
    instrument: dict[str, Any] | None = None
    clips: list[dict[str, Any]] = field(default_factory=list)
    solo: bool = False


@dataclass
class _State:
    name: str = "Untitled"
    sample_rate: int = SR
    tracks: list[_Track] = field(default_factory=list)
    next_id: int = 1
    path: str | None = None


class FakeStudio:
    """Stands in for ``studio_client.Studio`` in the profiler tests."""

    def __init__(
        self,
        kind: str = "headless",
        *,
        plugins: Mapping[str, FakePlugin] | None = None,
        supports_normalized: bool = True,
        bounce_dir_must_exist: bool = True,
    ) -> None:
        self._kind = kind
        self.plugins = dict(plugins) if plugins is not None else make_plugins()
        self.supports_normalized = supports_normalized
        self.state = _State()
        self.dirty = False
        self.undo: list[tuple[str, _State]] = []
        self.redo: list[tuple[str, _State]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self.bounces = 0
        self.fail: dict[str, str] = {}
        """command id -> error message: that command always fails (for failure-path tests)."""
        self.fail_param_value: dict[str, str] = {}
        """param id -> error message for setting that parameter."""
        self.bounce_noise = 0.0
        """Add this much noise (linear) to every bounce: a non-deterministic plugin."""
        self._rng = np.random.default_rng(3)

    # ---- the Studio interface ---------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return self._kind

    def close(self) -> None:
        self.closed = True

    def inspect(self, detail: str = "summary") -> dict[str, Any]:
        s = self.state
        return {
            "name": s.name, "path": s.path, "dirty": self.dirty, "sample_rate": s.sample_rate,
            "tempo": [{"tick": 0, "bpm": 120.0}],
            "tracks": [{"id": t.id, "name": t.name, "kind": t.kind} for t in s.tracks],
        }

    def commands(self, filter: str | None = None) -> list[dict[str, Any]]:
        return []

    def execute(self, command: str, params: Mapping[str, Any] | None = None) -> Any:
        p = dict(params or {})
        self.calls.append((command, copy.deepcopy(p)))
        if command in self.fail:
            raise StudioError(self.fail[command])
        handler = getattr(self, "cmd_" + command.replace(".", "_"), None)
        if handler is None:
            raise StudioError(f"unknown command `{command}`")
        return handler(p)

    # ---- helpers ------------------------------------------------------------------------------------

    def _push(self, label: str) -> None:
        self.undo.append((label, copy.deepcopy(self.state)))
        self.redo.clear()
        self.dirty = True

    def _track(self, p: Mapping[str, Any]) -> _Track:
        ref = p.get("track")
        for t in self.state.tracks:
            if t.id == ref or t.name == ref:
                return t
        raise StudioError(f"bad parameters: no track `{ref}`")

    def _plugin(self, plugin_id: Any) -> FakePlugin:
        plugin = self.plugins.get(str(plugin_id))
        if plugin is None:
            raise StudioError(f"unknown plugin `{plugin_id}`")
        return plugin

    def _target(self, p: Mapping[str, Any], cmd: str) -> tuple[_Track, dict[str, Any], FakePlugin]:
        t = self._track(p)
        if p.get("instrument") is True or p.get("slot") == "instrument":
            ins = t.instrument
            if ins is None:
                raise StudioError(f"{cmd}: the track has no instrument")
        else:
            if p.get("slot") not in (0, "a"):
                raise StudioError(f"{cmd}: insert slot {p.get('slot')} is empty")
            ins = t.insert
            if ins is None:
                raise StudioError(f"{cmd}: insert slot a is empty")
        return t, ins, self._plugin(ins["plugin"])

    # ---- session / history -------------------------------------------------------------------------

    def cmd_session_new(self, p: dict[str, Any]) -> dict[str, Any]:
        self.state = _State(name=str(p.get("name", "Untitled")), sample_rate=int(p.get("sample_rate", SR)))
        self.undo.clear()
        self.redo.clear()
        self.dirty = False
        return {"name": self.state.name, "sample_rate": self.state.sample_rate}

    def cmd_session_close(self, p: dict[str, Any]) -> dict[str, Any]:
        self.state = _State()
        self.undo.clear()
        self.redo.clear()
        self.dirty = False
        return {}

    def cmd_engine_history(self, p: dict[str, Any]) -> dict[str, Any]:
        return {"undo": [label for label, _ in self.undo], "can_redo": bool(self.redo)}

    def cmd_edit_undo(self, p: dict[str, Any]) -> dict[str, Any]:
        if not self.undo:
            return {"undone": False}
        label, before = self.undo.pop()
        self.redo.append((label, self.state))
        self.state = before
        self.dirty = True
        return {"undone": True}

    # ---- plugin lists ---------------------------------------------------------------------------------

    def cmd_engine_plugins(self, p: dict[str, Any]) -> list[dict[str, Any]]:
        return [pl.info_row() for pl in self.plugins.values() if pl.fmt == "builtin"]

    def cmd_engine_vst3_plugins(self, p: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"id": pl.id, "name": pl.name, "vendor": pl.vendor, "version": pl.version, "is_instrument": pl.instrument}
            for pl in self.plugins.values() if pl.fmt == "vst3"
        ]

    def cmd_engine_clap_plugins(self, p: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    # ---- tracks and plugins ----------------------------------------------------------------------------

    def cmd_track_new(self, p: dict[str, Any]) -> dict[str, Any]:
        kind = str(p.get("kind", "audio"))
        if kind not in ("audio", "instrument", "aux", "midi"):
            raise StudioError(f"unknown kind `{kind}`")
        self._push("New...")
        t = _Track(self.state.next_id, str(p.get("name") or f"Track {self.state.next_id}"), kind)
        self.state.next_id += 1
        self.state.tracks.append(t)
        return {"tracks": [t.id]}

    def cmd_track_delete(self, p: dict[str, Any]) -> dict[str, Any]:
        ids = p.get("tracks")
        if not isinstance(ids, list) or not ids:
            raise StudioError("track.delete: `tracks` required")
        self._push("Delete")
        self.state.tracks = [t for t in self.state.tracks if t.id not in ids]
        return {}

    def cmd_mix_solo(self, p: dict[str, Any]) -> dict[str, Any]:
        t = self._track(p)
        self._push("Solo")
        if p.get("exclusive") and p.get("value", True):
            for other in self.state.tracks:
                other.solo = False
        t.solo = bool(p.get("value", not t.solo))
        return {"value": t.solo}

    def cmd_mix_insert(self, p: dict[str, Any]) -> dict[str, Any]:
        t = self._track(p)
        plugin = self._plugin(p.get("plugin"))
        slot = p.get("slot", 0)
        if slot not in (0, "a"):
            raise StudioError("only slot 0 is supported by the fake")
        self._push("Insert Plugin")
        t.insert = {"plugin": plugin.id, "params": {q.id: q.default for q in plugin.params}}
        return {"track": t.id, "slot": 0, "plugin": plugin.id}

    def cmd_mix_instrument(self, p: dict[str, Any]) -> dict[str, Any]:
        t = self._track(p)
        plugin = self._plugin(p.get("plugin"))
        if not plugin.instrument:
            raise StudioError(f"mix.instrument: `{plugin.id}` is not an instrument")
        if t.kind != "instrument":
            raise StudioError("mix.instrument: only instrument tracks have an instrument")
        self._push("Set Instrument")
        t.instrument = {"plugin": plugin.id, "params": {q.id: q.default for q in plugin.params}}
        return {"track": t.id, "plugin": plugin.id}

    def _set_param(self, ins: dict[str, Any], plugin: FakePlugin, p: dict[str, Any], label: str) -> dict[str, Any]:
        pid = str(p.get("param"))
        if pid in self.fail_param_value:
            raise StudioError(self.fail_param_value[pid])
        param = next((q for q in plugin.params if q.id == pid), None)
        if param is None:
            raise StudioError(f"`{plugin.id}` has no parameter `{pid}`")
        value = p.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise StudioError("`value` must be a number")
        self._push(label)
        ins["params"][pid] = param.clamp(float(value))
        return {"value": ins["params"][pid]}

    def cmd_mix_insert_param(self, p: dict[str, Any]) -> dict[str, Any]:
        _, ins, plugin = self._target({**p, "slot": p.get("slot", 0)}, "mix.insert_param")
        return self._set_param(ins, plugin, p, "Set Plugin Parameter")

    def cmd_mix_instrument_param(self, p: dict[str, Any]) -> dict[str, Any]:
        _, ins, plugin = self._target({**p, "instrument": True}, "mix.instrument_param")
        return self._set_param(ins, plugin, p, "Set Instrument Parameter")

    def cmd_plugin_describe(self, p: dict[str, Any]) -> dict[str, Any]:
        t, ins, plugin = self._target(p, "plugin.describe")
        rows = []
        for q in plugin.params:
            v = ins["params"][q.id]
            rows.append({**_param_json(q), "value": v, "text": q.format(v), "text_source": "plugin" if q.plugin_text else "soundcraft"})
        return {
            "track": t.id, "slot": None if p.get("instrument") else 0, "instrument": bool(p.get("instrument")), "available": True,
            "plugin": {
                "id": plugin.id, "format": plugin.fmt, "name": plugin.name, "short_name": plugin.name[:8], "vendor": plugin.vendor,
                "version": plugin.version, "category": plugin.category, "is_instrument": plugin.instrument,
            },
            "bypass": False, "preset": "<factory default>", "params": rows, "state_bytes": 0, "stored_state_bytes": 0,
            "state_refused": False, "editor": False, "instance": False,
        }

    def cmd_plugin_param_text(self, p: dict[str, Any]) -> dict[str, Any]:
        _, _, plugin = self._target(p, "plugin.param_text")
        param = next((q for q in plugin.params if q.id == p.get("param")), None)
        if param is None:
            raise StudioError(f"`{plugin.id}` has no parameter `{p.get('param')}` (parameters: ...)")
        if "normalized" in p and not self.supports_normalized:
            raise StudioError("plugin.param_text: unknown field `normalized`")
        values = p.get("values")
        if not isinstance(values, list) or not 1 <= len(values) <= 64:
            raise StudioError("`values` needs 1..64 numbers")
        results = []
        for v in values:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise StudioError("`values` must be finite numbers")
            used = param.from_normalized(v) if p.get("normalized") else param.clamp(float(v))
            results.append({"value": v, "used": used, "clamped": False, "text": param.format(used), "source": "plugin" if param.plugin_text else "soundcraft"})
        return {"track": p["track"], "param": param.id, "name": param.name, "unit": param.unit, "normalized": bool(p.get("normalized")), "texts": [r["text"] for r in results], "results": results}

    # ---- audio and MIDI ----------------------------------------------------------------------------------

    def cmd_file_import_audio(self, p: dict[str, Any]) -> dict[str, Any]:
        t = self._track(p)
        path = str(p.get("path"))
        try:
            audio, sr = wav.read(path)
        except (OSError, ValueError) as exc:
            raise StudioError(f"cannot read {path}: {exc}") from exc
        if sr != self.state.sample_rate:
            raise StudioError(f"the fake only imports {self.state.sample_rate} Hz files")
        self._push("Audio...")
        t.clips.append({"id": 1000 + len(t.clips), "at": int(p.get("at", 0)), "audio": dsp.to_channels(audio, 2)})
        return {"imported": [{"track": t.id, "clip": t.clips[-1]["id"], "channels": 2, "frames": audio.shape[1]}]}

    def cmd_midi_new_clip(self, p: dict[str, Any]) -> dict[str, Any]:
        t = self._track(p)
        if t.kind not in ("instrument", "midi"):
            raise StudioError("midi.new_clip: not a MIDI or instrument track")
        self._push("New MIDI Clip")
        clip = {"id": 2000 + len(t.clips), "start": int(p.get("start", 0)), "end": int(p.get("end", 0)), "notes": []}
        t.clips.append(clip)
        return {"clip": clip["id"], "start": clip["start"], "end": clip["end"]}

    def cmd_midi_note_add(self, p: dict[str, Any]) -> dict[str, Any]:
        for t in self.state.tracks:
            for clip in t.clips:
                if clip["id"] == p.get("clip") and "notes" in clip:
                    self._push("Add Note")
                    clip["notes"].append(
                        (int(p.get("pitch", 60)), int(p.get("start_ticks", 0)), int(p.get("length_ticks", 480)), int(p.get("velocity", 100)))
                    )
                    return {"index": len(clip["notes"]) - 1, "notes": len(clip["notes"])}
        raise StudioError("midi.note_add: no such clip")

    def cmd_file_bounce_mix(self, p: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(p.get("path")))
        start, end = int(p.get("start", 0)), int(p.get("end", 0))
        if end <= start:
            raise StudioError("empty range")
        sr = self.state.sample_rate
        frames = end
        mix = np.zeros((2, frames), dtype=np.float32)
        any_solo = any(t.solo for t in self.state.tracks)
        for t in self.state.tracks:
            if any_solo and not t.solo:
                continue
            if t.insert is not None:
                src = np.zeros((2, frames), dtype=np.float32)
                for clip in t.clips:
                    a = clip["at"]
                    block = clip["audio"][:, : max(0, frames - a)]
                    src[:, a : a + block.shape[1]] += block
                plugin = self._plugin(t.insert["plugin"])
                assert plugin.process is not None
                mix += plugin.process(src, sr, t.insert["params"])
            elif t.instrument is not None:
                plugin = self._plugin(t.instrument["plugin"])
                assert plugin.synth is not None
                notes = []
                for clip in t.clips:
                    for pitch, start_ticks, length_ticks, vel in clip.get("notes", []):
                        notes.append((pitch, start_ticks / 960 * 0.5, length_ticks / 960 * 0.5, vel))
                mono = plugin.synth(notes, frames, sr, t.instrument["params"])
                mix += np.tile(mono, (2, 1))
        mix = mix[:, start:end]
        if self.bounce_noise:
            mix = mix + (self._rng.standard_normal(mix.shape) * self.bounce_noise).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        wav.write(path, mix, sr, "32f")
        self.bounces += 1
        return {"path": str(path), "bytes": int(mix.size * 4), "seconds": (end - start) / sr, "peak_db": dsp.peak_dbfs(mix), "lufs": -20.0, "true_peak_db": dsp.true_peak_dbfs(mix)}
