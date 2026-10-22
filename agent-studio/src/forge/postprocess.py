"""Post-processing of worker outputs (DESIGN 3.6): audio pipeline, instrument bundles, MIDI, sidecars."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from audiokit import analysis, dsp, instruments, midi, pitch, wav

from .jobs import MAX_ZONES, CommonParams, PostParams
from .music import hz_to_midi, key_name, parse_key
from .store import json_safe

BPM_TOLERANCE = 0.05
NOTE_TOLERANCE = 0.5
SILENT_DBFS = -80.0
MIN_LOOP_SEAM = 0.25
MAX_SIDECAR_ONSETS = 2000
MAX_NOTES_IN_SIDECAR = 20_000
MAX_JSON_BYTES = 1024 * 1024
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_ZONE_KEYS = ("root_note", "key_low", "key_high", "vel_low", "vel_high", "tune_cents", "gain_db", "loop_mode", "loop_start", "loop_end")
_ZIP_TIME = (2020, 1, 1, 0, 0, 0)


class PostprocessError(ValueError):
    """A worker output could not be post-processed (the job fails with this message)."""


@dataclass(frozen=True)
class ModelMeta:
    id: str
    family: str
    checkpoint: str = ""
    checkpoint_hash: str | None = None
    licence: str = ""


@dataclass(frozen=True)
class PostContext:
    job_id: str
    job_type: str
    model: ModelMeta
    params: CommonParams
    seed: int
    parents: tuple[str, ...] = ()


@dataclass
class Processed:
    """A finished file in ``work_dir`` plus the sidecar fields ``AssetStore.add_asset`` completes."""

    path: Path
    kind: str
    sidecar: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def safe_name(text: str, fallback: str = "file") -> str:
    cleaned = _SAFE_NAME.sub("_", text).strip("._")[:64]
    return cleaned or fallback


# ---- the audio pipeline ----

def process_audio(
    audio: np.ndarray, sr: int, post: PostParams, *, channels: int, sample_rate: int, instrument: bool = False
) -> tuple[np.ndarray, list[str]]:
    """resample/convert -> trim -> fades -> loop crossfade -> normalise. For instrument zones the
    attack is never trimmed or faded in, and loops are not crossfaded."""
    warnings: list[str] = []
    out = dsp.as_audio(audio)
    if sr != sample_rate:
        out = dsp.resample(out, sr, sample_rate)
    out = dsp.to_channels(out, channels)
    if post.trim_silence:
        trimmed = dsp.trim_silence(out, sample_rate, head=not instrument, tail=True)
        if trimmed.shape[1] == 0:
            warnings.append("audio is silent; nothing to trim")
        else:
            out = trimmed
    out = dsp.fade(out, sample_rate, 0.0 if instrument else post.fade_in_ms, post.fade_out_ms)
    if post.loop_crossfade_ms > 0 and not instrument:
        out = dsp.loop_crossfade(out, sample_rate, post.loop_crossfade_ms)
    if post.normalize == "peak":
        out = dsp.normalize_peak(out, post.target)
    elif post.normalize == "lufs":
        out = dsp.normalize_lufs(out, sample_rate, post.target, ceiling_dbfs=-1.0)
    return out, warnings


def measure_note(audio: np.ndarray, sr: int) -> float | None:
    """Median pitch of the audio as a fractional MIDI note, or ``None`` if no pitch is found."""
    mono = dsp.as_audio(audio).mean(axis=0, keepdims=True, dtype=np.float32)
    if sr > 22050:
        mono, sr = dsp.resample(mono, sr, 22050), 22050
    voiced = [f0 for _, f0, _ in pitch.track(mono, sr) if f0]
    return round(hz_to_midi(float(np.median(voiced))), 2) if len(voiced) >= 3 else None


def measure_full(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """``audiokit.analysis.summary`` shaped for the sidecar (``onsets`` capped, ``onsets_s`` dropped)."""
    summary = dict(analysis.summary(audio, sr))
    summary.pop("onsets_s", None)
    onsets = summary.get("onsets")
    if isinstance(onsets, list):
        summary["onsets"] = onsets[:MAX_SIDECAR_ONSETS]
    return summary


def _measure_light(audio: np.ndarray, sr: int) -> dict[str, Any]:
    return {
        "duration_s": round(audio.shape[1] / sr, 4),
        "peak_dbfs": round(dsp.peak_dbfs(audio), 2),
        "true_peak_dbfs": round(dsp.true_peak_dbfs(audio), 2),
        "rms_dbfs": round(dsp.rms_dbfs(audio), 2),
        "centroid_hz": analysis.centroid_hz(audio, sr),
        "clipped_samples": dsp.clipped_samples(audio),
        "dc_offset": round(dsp.dc_offset(audio), 6),
    }


def requested_values(ctx: PostContext) -> dict[str, Any]:
    """What the job asked for that can be checked against the audio."""
    p = ctx.params
    return {
        "bpm": getattr(p, "bpm", None) if ctx.job_type == "generate.loop" else None,
        "key": getattr(p, "key", None) if ctx.job_type == "generate.loop" else None,
        "note": getattr(p, "note", None) if ctx.job_type == "generate.sample" else None,
    }


def compare(requested: dict[str, Any], measured: dict[str, Any], label: str = "") -> list[str]:
    """Warnings for every requested value the measurement contradicts, e.g. ``"bpm requested 130, measured 65 (0.71)"``."""
    prefix = f"{label}: " if label else ""
    warnings: list[str] = []
    want_bpm, got_bpm = requested.get("bpm"), measured.get("bpm")
    if want_bpm:
        got = float(got_bpm or 0.0)
        if abs(got - float(want_bpm)) > BPM_TOLERANCE * float(want_bpm):
            warnings.append(f"{prefix}bpm requested {float(want_bpm):g}, measured {got:g} ({float(measured.get('bpm_confidence') or 0.0):.2f})")
    if requested.get("key"):
        want, got_name = parse_key(str(requested["key"])), str(measured.get("key") or "")
        if want is None or parse_key(got_name) != want:
            shown = key_name(*want) if want else requested["key"]
            warnings.append(f"{prefix}key requested {shown}, measured {got_name or 'none'} ({float(measured.get('key_confidence') or 0.0):.2f})")
    if requested.get("note") is not None:
        got_note = measured.get("note")
        if got_note is None:
            warnings.append(f"{prefix}note requested {requested['note']}, no pitch detected")
        elif abs(float(got_note) - float(requested["note"])) > NOTE_TOLERANCE:
            warnings.append(f"{prefix}note requested {requested['note']}, measured {float(got_note):.1f}")
    return warnings


def _level_warnings(measured: dict[str, Any], label: str = "") -> list[str]:
    prefix = f"{label}: " if label else ""
    warnings = []
    if float(measured.get("peak_dbfs", 0.0)) <= SILENT_DBFS:
        warnings.append(f"{prefix}output is silent")
    if int(measured.get("clipped_samples", 0)) > 0:
        warnings.append(f"{prefix}{measured['clipped_samples']} clipped samples")
    return warnings


# ---- sidecars ----

def _base_sidecar(ctx: PostContext, kind: str, candidate: int, post: PostParams, sr: int | None, channels: int | None) -> dict[str, Any]:
    p = ctx.params
    return {
        "job_id": ctx.job_id, "type": ctx.job_type, "kind": kind,
        "model": ctx.model.id, "family": ctx.model.family, "checkpoint": ctx.model.checkpoint,
        "checkpoint_hash": ctx.model.checkpoint_hash, "licence": ctx.model.licence,
        "prompt": p.prompt, "tags": p.tags, "seed": (ctx.seed + candidate) % 2**63,
        "params": p.model_dump(mode="json"), "requested": requested_values(ctx),
        "sample_rate": sr, "channels": channels, "bit_depth": post.bit_depth if kind in ("audio", "instrument") else None,
        "parents": list(ctx.parents), "candidate": candidate, "rating": None, "note": None,
    }


def _extra_fields(sidecar: dict[str, Any], extra: dict[str, Any]) -> None:
    stem = extra.get("stem")
    if isinstance(stem, str):
        sidecar["stem"] = stem[:64]
    rest = {k: v for k, v in extra.items() if k not in ("zones", "notes", "stem")}
    clean = json_safe(rest)
    if clean and len(json.dumps(clean)) <= 10_000:
        sidecar["extra"] = clean


# ---- entry point ----

def process_output(output: dict[str, Any], ctx: PostContext, work_dir: Path) -> Processed:
    """Post-process one validated worker output into a file in ``work_dir`` ready for the store."""
    work_dir.mkdir(parents=True, exist_ok=True)
    handlers = {
        "audio": _process_audio_output, "instrument": _process_instrument,
        "midi": _process_midi, "json": _process_json,
    }
    handler = handlers.get(output["kind"])
    if handler is None:
        raise PostprocessError(f"unsupported output kind {output['kind']!r}")
    try:
        return handler(output, ctx, work_dir)
    except (wav.WavError, instruments.InstrumentError, midi.MidiError, OSError) as exc:
        raise PostprocessError(f"{output['kind']} output {Path(output['path']).name!r}: {exc}") from exc


def _process_audio_output(output: dict[str, Any], ctx: PostContext, work_dir: Path) -> Processed:
    post, p, candidate = ctx.params.post, ctx.params, int(output["candidate"])
    audio, sr = wav.read(output["path"])
    if audio.shape[1] == 0:
        raise PostprocessError("worker produced an empty audio file")
    processed, warnings = process_audio(audio, sr, post, channels=p.channels, sample_rate=p.sample_rate)
    out_path = work_dir / f"audio_{candidate}_{len(list(work_dir.glob('*.wav')))}.wav"
    wav.write(out_path, processed, p.sample_rate, post.bit_depth)
    measured = measure_full(processed, p.sample_rate)
    if ctx.job_type == "generate.sample":
        measured["note"] = measure_note(processed, p.sample_rate)
    side = _base_sidecar(ctx, "audio", candidate, post, p.sample_rate, p.channels)
    warnings += compare(side["requested"], measured) + _level_warnings(measured)
    if ctx.job_type == "generate.loop" and float(measured.get("loop_seam", 1.0)) < MIN_LOOP_SEAM and post.loop_crossfade_ms == 0:
        warnings.append(f"loop seam score {float(measured['loop_seam']):.2f} is low; consider post.loop_crossfade_ms")
    side.update(measured=measured, warnings=warnings)
    _extra_fields(side, output["extra"])
    return Processed(out_path, "audio", side, warnings)


def _zone_source(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PostprocessError("zone has no sample path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PostprocessError(f"zone sample {relative!r} is outside the instrument folder") from exc
    if not resolved.is_file():
        raise PostprocessError(f"zone sample {relative!r} does not exist")
    return resolved


def _process_instrument(output: dict[str, Any], ctx: PostContext, work_dir: Path) -> Processed:
    post, p, candidate = ctx.params.post, ctx.params, int(output["candidate"])
    raw = output["extra"].get("zones")
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_ZONES:
        raise PostprocessError(f"instrument output needs 1..{MAX_ZONES} zones")
    name = safe_name(str(getattr(p, "name", "") or output["extra"].get("name") or "instrument"), "instrument")
    stage = work_dir / f"bundle_{candidate}"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "samples").mkdir(parents=True)
    zones: list[dict[str, Any]] = []
    warnings: list[str] = []
    used: set[str] = set()
    peaks, durations = [], []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PostprocessError("zones must be objects")
        audio, sr = wav.read(_zone_source(Path(output["path"]), item.get("sample")))
        processed, zone_warnings = process_audio(audio, sr, post, channels=p.channels, sample_rate=p.sample_rate, instrument=True)
        fname = safe_name(Path(str(item["sample"])).stem, f"zone{index}") + ".wav"
        while fname in used:
            fname = f"{index:03d}_{fname}"
        used.add(fname)
        wav.write(stage / "samples" / fname, processed, p.sample_rate, post.bit_depth)
        zone: dict[str, Any] = {"sample": f"samples/{fname}"}
        zone.update({k: item[k] for k in _ZONE_KEYS if item.get(k) is not None})
        measured = _measure_light(processed, p.sample_rate)
        measured["note"] = measure_note(processed, p.sample_rate)
        label = f"zone {zone.get('root_note')}"
        warnings += [f"{label}: {w}" for w in zone_warnings]
        warnings += compare({"note": zone.get("root_note")}, measured, label) + _level_warnings(measured, label)
        zones.append({**zone, "measured": measured})
        peaks.append(measured["peak_dbfs"])
        durations.append(measured["duration_s"])
    plain = [{k: v for k, v in z.items() if k != "measured"} for z in zones]
    instruments.write_sfz(stage / f"{name}.sfz", plain, stage)
    instruments.write_dspreset(stage / f"{name}.dspreset", plain, stage)
    manifest = {"name": name, "sample_rate": p.sample_rate, "channels": p.channels, "bit_depth": post.bit_depth, "zones": plain}
    (stage / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2, allow_nan=False), encoding="utf-8")
    bundle = _zip_bundle(stage, work_dir / f"{name}_{candidate}.zip")
    side = _base_sidecar(ctx, "instrument", candidate, post, p.sample_rate, p.channels)
    side.update(
        name=name, files=zones, warnings=warnings,
        measured={"zones": len(zones), "peak_dbfs": max(peaks), "duration_s": max(durations)},
    )
    _extra_fields(side, output["extra"])
    shutil.rmtree(stage, ignore_errors=True)
    return Processed(bundle, "instrument", side, warnings)


def _zip_bundle(stage: Path, target: Path) -> Path:
    """Zip ``stage`` deterministically (sorted entries, fixed timestamps) so equal bundles hash equally."""
    files = sorted(f for f in stage.rglob("*") if f.is_file())
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            info = zipfile.ZipInfo(file.relative_to(stage).as_posix(), _ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, file.read_bytes())
    return target


def _process_midi(output: dict[str, Any], ctx: PostContext, work_dir: Path) -> Processed:
    candidate = int(output["candidate"])
    source = Path(output["path"])
    parsed = midi.read_smf(source)  # rejects malformed or oversized files
    notes = output["extra"].get("notes")
    if not isinstance(notes, list):
        notes = [{"pitch": n.pitch, "start_s": n.start_s, "end_s": n.end_s, "velocity": n.velocity} for n in parsed]
    out_path = work_dir / f"midi_{candidate}.mid"
    shutil.copyfile(source, out_path)
    side = _base_sidecar(ctx, "midi", candidate, ctx.params.post, None, None)
    warnings = [] if notes else ["no notes detected"]
    end = max((float(n.get("end_s", 0.0)) for n in notes if isinstance(n, dict)), default=0.0)
    side.update(
        notes=json_safe(notes[:MAX_NOTES_IN_SIDECAR]), warnings=warnings,
        measured={"note_count": len(notes), "duration_s": round(end, 4)},
    )
    bpm = output["extra"].get("bpm")
    if isinstance(bpm, (int, float)) and not isinstance(bpm, bool):
        side["tempo_bpm"] = float(bpm)
    return Processed(out_path, "midi", side, warnings)


def _process_json(output: dict[str, Any], ctx: PostContext, work_dir: Path) -> Processed:
    source = Path(output["path"])
    if source.stat().st_size > MAX_JSON_BYTES:
        raise PostprocessError("json output too large")
    try:
        json.loads(source.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PostprocessError(f"json output is not valid JSON: {exc}") from exc
    out_path = work_dir / f"json_{int(output['candidate'])}.json"
    shutil.copyfile(source, out_path)
    side = _base_sidecar(ctx, "json", int(output["candidate"]), ctx.params.post, None, None)
    side.update(measured={}, warnings=[])
    return Processed(out_path, "json", side, [])
