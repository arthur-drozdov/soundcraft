"""The ``testtone`` family: a deterministic test generator, not a music model (DESIGN 3.3).

Pitched tones, noise bursts and clicks at tempo, driven only by the job's params and seed, so the
same job always yields the same audio. It labels itself as a test generator in its ``ModelInfo``.

Test hooks (used by the Forge tests and the smoke check), all set through ``tags``:

* ``x-delay_s``: take that many seconds, reporting progress and honouring cancellation.
* ``x-fault``: ``crash`` (the process dies mid-job), ``error`` (the job fails),
  ``off_tempo`` (loops are rendered 1.5x too fast), ``wrong_pitch`` (samples and zones are 3
  semitones sharp) or ``silent``.

``hello`` options: ``extra_models`` is a list of ``{"id", "vram_gb"}`` that adds copies of the
model with a declared memory footprint, to exercise the gateway's memory budget.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import lfilter

from audiokit import wav
from forge.music import midi_to_hz, parse_key

from .protocol import ModelInfo, Output, RunJob
from .runtime import CancelCheck, Family, Progress, check_cancel, main

NATIVE_SR = 44100
JOB_TYPES = [
    "generate.loop",
    "generate.sample",
    "generate.instrument",
    "generate.sfx",
    "transform.audio",
    "extend.audio",
    "inpaint.audio",
    "layer.add",
]
LICENCE = "CC0-1.0 (test generator, not a music model)"
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_FAULTS = ("crash", "error", "off_tempo", "wrong_pitch", "silent")


def _tag(params: dict[str, Any], name: str) -> str:
    value = (params.get("tags") or {}).get(name, "")
    return value[0] if isinstance(value, list) and value else str(value)


def _time(n: int, sr: int = NATIVE_SR) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / sr


def _harmonic(freq: float, seconds: float, rng: np.random.Generator, harmonics: int, sr: int = NATIVE_SR) -> np.ndarray:
    """A decaying harmonic tone with a short attack; the seed picks the overtone balance."""
    n = max(1, int(round(seconds * sr)))
    t = _time(n, sr)
    weights = rng.uniform(0.5, 1.0, harmonics) / np.arange(1, harmonics + 1)
    out = np.zeros(n)
    for k in range(1, harmonics + 1):
        if freq * k < sr / 2.2:
            out += weights[k - 1] * np.sin(2 * np.pi * freq * k * t)
    env = np.minimum(1.0, t / 0.01) * np.exp(-t / max(seconds, 0.05) * 1.5)
    env *= np.minimum(1.0, (seconds - t) / 0.02)
    return out * env / weights.sum()


def _stereo(mono: np.ndarray) -> np.ndarray:
    """Two channels that differ by a 1 ms delay (so stereo handling is exercised)."""
    delay = NATIVE_SR // 1000
    right = np.concatenate([np.zeros(delay), mono[:-delay]]) if mono.size > delay else mono
    return np.stack([mono, right]).astype(np.float32)


def _loop(params: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    bpm = float(params["bpm"]) * (1.5 if _tag(params, "x-fault") == "off_tempo" else 1.0)
    beat = 60.0 / bpm
    n = int(round(params["bars"] * 4 * 60.0 / float(params["bpm"]) * NATIVE_SR))
    out = np.zeros(n)
    kick_t = _time(int(0.12 * NATIVE_SR))
    kick = np.sin(2 * np.pi * (50 * kick_t + 110 * (1 - np.exp(-kick_t * 40)) / 40)) * np.exp(-kick_t / 0.05)
    tick = rng.standard_normal(int(0.004 * NATIVE_SR)) * np.exp(-_time(int(0.004 * NATIVE_SR)) / 0.001)
    index = 0
    while (start := int(round(index * beat * NATIVE_SR))) < n:
        gain = 1.0 if index % 4 == 0 else 0.7
        for sound, level in ((kick, 0.9 * gain), (tick, 0.5 * gain)):
            seg = sound[: n - start]
            out[start : start + seg.size] += level * seg
        index += 1
    parsed = parse_key(params.get("key") or "") or (9, "minor")
    root = 48 + parsed[0] + 12 * int(rng.integers(0, 2))
    t = _time(n)
    for semitones in (0, 3 if parsed[1] == "minor" else 4, 7):
        f = midi_to_hz(root + semitones)
        out += 0.12 * (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t))
    return _stereo(0.7 * out / max(1.0, float(np.max(np.abs(out)))))


def _sample(params: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    note = int(params["note"]) + (3 if _tag(params, "x-fault") == "wrong_pitch" else 0)
    return _stereo(0.6 * _harmonic(midi_to_hz(note), float(params["length_s"]), rng, 5))


def _sfx(params: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    seconds = float(params["duration_s"])
    n = int(round(seconds * NATIVE_SR))
    t = _time(n)
    noise = rng.standard_normal(n)
    # blend fixed one-pole low-passes so the brightness falls as the burst decays
    bright, mid, dark = (lfilter([a], [1.0, a - 1.0], noise) for a in (0.9, 0.25, 0.04))
    w = np.exp(-t / (seconds * 0.3))
    filtered = w * w * bright + 2 * w * (1 - w) * mid + (1 - w) * (1 - w) * dark
    env = np.exp(-t / (seconds * 0.25)) * np.minimum(1.0, t / 0.002)
    return _stereo(0.8 * filtered * env / max(1e-9, float(np.max(np.abs(filtered * env)))))


def _read_input(job: RunJob) -> tuple[np.ndarray, int]:
    path = job["inputs"].get("audio")
    if not path or not Path(path).is_file():
        raise ValueError("missing audio input")
    return wav.read(path)


def _transform(audio: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    s = float(params["strength"])
    coef = 1.0 - 0.9 * s  # one-pole low-pass, darker with strength
    smooth = lfilter([coef], [1.0, coef - 1.0], audio, axis=1)
    return ((1.0 - s) * audio + s * smooth).astype(np.float32)


def _extend(audio: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    extra = int(round(float(params["seconds"]) * sr))
    window = audio[:, -min(audio.shape[1], sr) :]
    reps = -(-extra // window.shape[1])
    tail = np.tile(window, (1, reps))[:, :extra] * np.linspace(1.0, 0.0, extra, dtype=np.float32)
    return np.concatenate([audio, tail], axis=1)


def _inpaint(audio: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    start, end = int(float(params["start_s"]) * sr), min(audio.shape[1], int(float(params["end_s"]) * sr))
    if start >= audio.shape[1] or end <= start:
        raise ValueError("the region to inpaint lies outside the audio")
    out = audio.copy()
    t = _time(end - start, sr)
    fill = (0.3 * np.sin(2 * np.pi * 440.0 * t) * np.hanning(end - start)).astype(np.float32)
    out[:, start:end] = 0.5 * out[:, start:end] * (1 - np.hanning(end - start)) + fill
    return out


def _layer(audio: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    note = 48 + zlib.crc32(str(params["description"]).encode("utf-8")) % 25
    t = _time(audio.shape[1], sr)
    return (audio + 0.2 * np.sin(2 * np.pi * midi_to_hz(note) * t).astype(np.float32)).astype(np.float32)


def _zones(params: dict[str, Any]) -> list[dict[str, Any]]:
    from audiokit.instruments import plan_keybed

    layers = int(params["velocity_layers"])
    edges = [round(i * 128 / layers) for i in range(layers + 1)]
    safe = _SAFE.sub("_", str(params["name"]))
    zones = []
    for root, low, high in plan_keybed(int(params["note_low"]), int(params["note_high"]), int(params["step"])):
        for layer in range(layers):
            zones.append(
                {
                    "sample": f"samples/{safe}_{root:03d}_v{layer + 1}.wav",
                    "root_note": root, "key_low": low, "key_high": high,
                    "vel_low": edges[layer], "vel_high": edges[layer + 1] - 1,
                    "layer": layer,
                }
            )
    return zones


class TestToneFamily(Family):
    __test__ = False  # not a pytest class
    name = "testtone"

    def __init__(self) -> None:
        self._extra: dict[str, float] = {}
        self._loaded: set[str] = set()
        self._digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

    def configure(self, options: dict[str, Any]) -> None:
        extra = options.get("extra_models") or []
        self._extra = {str(m["id"]): float(m.get("vram_gb", 0.0)) for m in extra if isinstance(m, dict) and "id" in m}

    def models(self) -> list[ModelInfo]:
        def info(model_id: str, vram: float) -> ModelInfo:
            return {
                "id": model_id, "family": self.name, "job_types": list(JOB_TYPES), "available": True,
                "licence": LICENCE, "checkpoint": "builtin:testtone", "checkpoint_hash": self._digest,
                "vram_gb": vram, "max_seconds": 300.0,
                "description": "Deterministic test generator (tones, noise bursts, clicks); not a music model",
                "test_generator": True,
            }  # type: ignore[typeddict-unknown-key]

        return [info("testtone", 0.0)] + [info(i, v) for i, v in self._extra.items()]

    def load(self, model: str) -> None:
        if model != "testtone" and model not in self._extra:
            raise ValueError(f"unknown model {model!r}")
        self._loaded.add(model)

    def unload(self, model: str) -> None:
        self._loaded.discard(model)

    def run(self, job: RunJob, progress: Progress, cancelled: CancelCheck) -> list[Output]:
        if job["model"] not in self._loaded:
            raise RuntimeError(f"model {job['model']!r} is not loaded")
        params, out_dir = job["params"], Path(job["out_dir"])
        fault = _tag(params, "x-fault")
        if fault and fault not in _FAULTS:
            raise ValueError(f"unknown x-fault {fault!r}")
        self._delay(params, progress, cancelled, crash=fault == "crash")
        if fault == "error":
            raise RuntimeError("injected failure")
        outputs: list[Output] = []
        candidates = int(params.get("candidates", 1))
        for candidate in range(candidates):
            check_cancel(cancelled)
            rng = np.random.default_rng((int(job["seed"]) + candidate) % 2**63)
            base = candidate / candidates
            if job["type"] == "generate.instrument":
                out = self._instrument(job, params, rng, candidate, out_dir, progress, cancelled, base, 1 / candidates)
            else:
                audio, sr = self._render(job, params, rng)
                if fault == "silent":
                    audio = np.zeros_like(audio)
                path = out_dir / f"candidate_{candidate}.wav"
                wav.write(path, audio, sr, "32f")
                out = {"path": str(path), "kind": "audio", "candidate": candidate, "extra": {}}
            outputs.append(out)  # type: ignore[arg-type]
            progress((candidate + 1) / candidates, f"candidate {candidate + 1}/{candidates}")
        return outputs

    @staticmethod
    def _delay(params: dict[str, Any], progress: Progress, cancelled: CancelCheck, crash: bool) -> None:
        try:
            total = max(0.0, min(300.0, float(_tag(params, "x-delay_s") or 0)))
        except ValueError:
            total = 0.0
        started = time.monotonic()
        progress(0.0, "started")
        if crash:
            time.sleep(min(total, 0.05))
            os._exit(3)
        while (elapsed := time.monotonic() - started) < total:
            check_cancel(cancelled)
            progress(0.5 * elapsed / total, "working")
            time.sleep(0.02)

    def _render(self, job: RunJob, params: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, int]:
        kind = job["type"]
        if kind == "generate.loop":
            return _loop(params, rng), NATIVE_SR
        if kind == "generate.sample":
            return _sample(params, rng), NATIVE_SR
        if kind == "generate.sfx":
            return _sfx(params, rng), NATIVE_SR
        audio, sr = _read_input(job)
        if kind == "transform.audio":
            return _transform(audio, params), sr
        if kind == "extend.audio":
            return _extend(audio, sr, params), sr
        if kind == "inpaint.audio":
            return _inpaint(audio, sr, params), sr
        if kind == "layer.add":
            return _layer(audio, sr, params), sr
        raise ValueError(f"testtone does not run {kind!r}")

    def _instrument(
        self, job: RunJob, params: dict[str, Any], rng: np.random.Generator, candidate: int, out_dir: Path,
        progress: Progress, cancelled: CancelCheck, base: float, span: float,
    ) -> dict[str, Any]:
        bundle = out_dir / f"instrument_{candidate}"
        zones = _zones(params)
        layers = int(params["velocity_layers"])
        shift = 3 if _tag(params, "x-fault") == "wrong_pitch" else 0
        seed = int(rng.integers(0, 2**31))  # one timbre for every pitch, as in a keybed with a shared seed
        for i, zone in enumerate(zones):
            check_cancel(cancelled)
            layer = zone.pop("layer")
            level = 0.3 + 0.4 * (layer + 1) / layers
            tone = _harmonic(midi_to_hz(zone["root_note"] + shift), float(params["length_s"]), np.random.default_rng(seed), 3 + 2 * layer)
            wav.write(bundle / zone["sample"], _stereo(level * tone), NATIVE_SR, "32f")
            progress(base + span * (i + 1) / len(zones), f"zone {i + 1}/{len(zones)}")
        return {
            "path": str(bundle), "kind": "instrument", "candidate": candidate,
            "extra": {"name": params["name"], "zones": zones},
        }


if __name__ == "__main__":
    main(TestToneFamily())
