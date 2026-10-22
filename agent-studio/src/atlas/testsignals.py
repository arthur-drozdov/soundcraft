"""Synthesised test signals for the profiler's offline measurements (DESIGN 6.2 step 4).

Everything is generated from a seeded random generator, so a run is repeatable and nothing
audio is ever committed. Effects get pink noise, a logarithmic sine sweep and a drum/click
pattern, written as WAV files to a temp folder (the three separately, and one composite the
profiler imports so a single bounce measures all three). Instruments get a MIDI plan: a
sustained C major chord and one staccato note.

All audio is ``float32`` shaped ``(channels, frames)``. Durations and sample rates are bounded
so a hostile option cannot ask for a gigabyte of noise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

from audiokit import wav

MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 384_000
MIN_DURATION_S = 3.0
MAX_DURATION_S = 60.0
DEFAULT_DURATION_S = 9.0
TICKS_PER_QUARTER = 960
"""SoundCraft's MIDI resolution (``soundcraft_time::TICKS_PER_QUARTER``)."""

# Layout of the composite effect signal in seconds at DEFAULT_DURATION_S (scaled for shorter caps):
# name -> (signal start, signal end, analysis window start, analysis window end).
_LAYOUT: Mapping[str, tuple[float, float, float, float]] = {
    "noise": (0.0, 2.0, 0.3, 2.3),
    "sweep": (2.5, 4.5, 2.5, 4.8),
    "clicks": (5.0, 7.5, 5.0, 8.5),
}
_LAYOUT_FILE_END_S = 8.5
_LAYOUT_BOUNCE_END_S = 9.0

SWEEP_BAND_EDGES_HZ: tuple[float, ...] = (20.0, 150.0, 600.0, 2400.0, 9600.0, 20000.0)
"""Frequency regions (Hz) the sweep analysis reports levels for."""


class SignalError(ValueError):
    """A signal could not be built from the given arguments."""


@dataclass(frozen=True)
class EffectSignals:
    """The test signals for effects, as written to disk.

    ``path`` is the composite file to import; ``segments`` maps each signal name to the window
    ``(start_s, end_s)`` to analyse in a bounce that starts at 0; ``bounce_end_s`` is where the
    bounce range ends (a little after the file, to catch latency and tails).
    """

    path: Path
    parts: dict[str, Path]
    sample_rate: int
    file_duration_s: float
    bounce_end_s: float
    segments: dict[str, tuple[float, float]]
    sweep_range_hz: tuple[float, float]
    sweep_start_s: float = 0.0
    sweep_length_s: float = 0.0


@dataclass(frozen=True)
class PlanNote:
    """One note of the instrument test clip, in seconds."""

    pitch: int
    start_s: float
    length_s: float
    velocity: int = 100


@dataclass(frozen=True)
class InstrumentPlan:
    """A MIDI test clip for instruments and the windows to analyse after bouncing it."""

    notes: tuple[PlanNote, ...]
    segments: dict[str, tuple[float, float]]
    bounce_end_s: float
    bpm: float = 120.0
    ticks_per_quarter: int = TICKS_PER_QUARTER
    names: dict[str, str] = field(default_factory=dict)

    def note_calls(self, clip: int) -> list[dict[str, object]]:
        """``midi.note_add`` parameter dicts for ``clip``."""
        calls: list[dict[str, object]] = []
        for n in self.notes:
            calls.append(
                {
                    "clip": clip,
                    "pitch": n.pitch,
                    "start_ticks": seconds_to_ticks(n.start_s, self.bpm, self.ticks_per_quarter),
                    "length_ticks": max(1, seconds_to_ticks(n.length_s, self.bpm, self.ticks_per_quarter)),
                    "velocity": n.velocity,
                }
            )
        return calls


def _check_sr(sr: int) -> int:
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or not MIN_SAMPLE_RATE <= int(sr) <= MAX_SAMPLE_RATE:
        raise SignalError(f"sample rate must be an integer in {MIN_SAMPLE_RATE}..{MAX_SAMPLE_RATE}")
    return int(sr)


def _check_duration(value: float, name: str = "duration_s", lo: float = 0.05, hi: float = MAX_DURATION_S) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise SignalError(f"{name} must be a number in {lo:g}..{hi:g}")
    return float(value)


def seconds_to_ticks(seconds: float, bpm: float = 120.0, ticks_per_quarter: int = TICKS_PER_QUARTER) -> int:
    """MIDI ticks for ``seconds`` at ``bpm`` (quarter-note tempo)."""
    if not math.isfinite(seconds) or not math.isfinite(bpm) or bpm <= 0 or seconds < 0:
        raise SignalError("seconds and bpm must be finite, bpm positive")
    return int(round(seconds * bpm / 60.0 * ticks_per_quarter))


def _scale_to_peak(x: np.ndarray, peak_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0.0:
        return x.astype(np.float32)
    return (x * (10.0 ** (peak_dbfs / 20.0) / peak)).astype(np.float32)


def _scale_to_rms(x: np.ndarray, rms_dbfs: float) -> np.ndarray:
    rms = math.sqrt(float(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    if rms <= 0.0:
        return x.astype(np.float32)
    return (x * (10.0 ** (rms_dbfs / 20.0) / rms)).astype(np.float32)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Mono pink (1/f power) noise by shaping white noise in the frequency domain."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:] * n)
    shape[0] = 0.0
    return np.fft.irfft(spectrum * shape, n)


def pink_noise(duration_s: float, sr: int, *, seed: int = 1, level_dbfs: float = -18.0, channels: int = 2) -> np.ndarray:
    """Pink noise at ``level_dbfs`` RMS. Stereo noise is 80 % shared between the channels, so
    width-type parameters have something to change without the signal being dual mono."""
    sr = _check_sr(sr)
    duration_s = _check_duration(duration_s)
    if channels not in (1, 2):
        raise SignalError("channels must be 1 or 2")
    n = max(2, int(round(duration_s * sr)))
    rng = np.random.default_rng(seed)
    left = _pink(n, rng)
    if channels == 1:
        out = left[np.newaxis, :]
    else:
        right = 0.8 * left + 0.2 * _pink(n, rng)
        out = np.stack([left, right])
    fade = min(n // 4, int(0.005 * sr))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade)
        out[:, :fade] *= ramp
        out[:, -fade:] *= ramp[::-1]
    return _scale_to_rms(out, level_dbfs)


def log_sweep(
    duration_s: float,
    sr: int,
    *,
    f_lo: float = 20.0,
    f_hi: float | None = None,
    level_dbfs: float = -12.0,
    channels: int = 2,
) -> np.ndarray:
    """Logarithmic sine sweep from ``f_lo`` to ``f_hi`` (default 20 kHz, below Nyquist) at ``level_dbfs`` peak."""
    sr = _check_sr(sr)
    duration_s = _check_duration(duration_s)
    top = min(20000.0, 0.45 * sr) if f_hi is None else float(f_hi)
    if not (math.isfinite(f_lo) and math.isfinite(top) and 0 < f_lo < top < sr / 2):
        raise SignalError("sweep needs 0 < f_lo < f_hi < Nyquist")
    n = max(2, int(round(duration_s * sr)))
    t = np.arange(n) / sr
    k = math.log(top / f_lo)
    phase = 2.0 * math.pi * f_lo * duration_s / k * (np.exp(t * k / duration_s) - 1.0)
    x = np.sin(phase)
    fade = min(n // 4, int(0.01 * sr))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade)
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
    mono = _scale_to_peak(x, level_dbfs)
    return np.tile(mono, (channels, 1)) if channels == 2 else mono[np.newaxis, :]


def _decay_env(n: int, sr: int, seconds: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (seconds * sr))


def drum_pattern(duration_s: float, sr: int, *, bpm: float = 120.0, peak_dbfs: float = -6.0, seed: int = 7, channels: int = 2) -> np.ndarray:
    """A short kick/snare/hat pattern (kick on beat 1 and 3, snare on 2 and 4, hats on the offbeats).

    Sharp transients and decaying bodies make dynamics, gating, envelope and ambience
    parameters visible in level, peak and decay.
    """
    sr = _check_sr(sr)
    duration_s = _check_duration(duration_s)
    if not math.isfinite(bpm) or not 30 <= bpm <= 300:
        raise SignalError("bpm must be 30..300")
    rng = np.random.default_rng(seed)
    n = max(2, int(round(duration_s * sr)))
    out = np.zeros(n)
    hp = signal.butter(2, 6000.0 / (sr / 2), btype="highpass", output="sos") if sr > 14000 else None

    def kick() -> np.ndarray:
        m = int(0.30 * sr)
        t = np.arange(m) / sr
        freq = 50.0 + 100.0 * np.exp(-t / 0.03)
        return np.sin(2 * math.pi * np.cumsum(freq) / sr) * _decay_env(m, sr, 0.10)

    def snare() -> np.ndarray:
        m = int(0.20 * sr)
        t = np.arange(m) / sr
        body = np.sin(2 * math.pi * 190.0 * t) * _decay_env(m, sr, 0.04)
        noise = rng.standard_normal(m) * _decay_env(m, sr, 0.06)
        return 0.6 * body + 0.5 * noise

    def hat() -> np.ndarray:
        m = int(0.05 * sr)
        noise = rng.standard_normal(m)
        if hp is not None:
            noise = signal.sosfilt(hp, noise)
        return 0.3 * noise * _decay_env(m, sr, 0.012)

    beat = 60.0 / bpm
    step = 0
    while True:
        at = int(round(step * beat / 2 * sr))
        if at >= n - int(0.05 * sr):
            break
        if step % 2 == 0:
            voice = kick() if (step // 2) % 2 == 0 else snare()
        else:
            voice = hat()
        end = min(n, at + voice.size)
        out[at:end] += voice[: end - at]
        step += 1
    mono = _scale_to_peak(out, peak_dbfs)
    return np.tile(mono, (channels, 1)) if channels == 2 else mono[np.newaxis, :]


def _check_cap(duration_s: float) -> float:
    return _check_duration(duration_s, "duration_s", MIN_DURATION_S, MAX_DURATION_S)


def build_effect_signals(folder: str | Path, sr: int = 48000, duration_s: float = DEFAULT_DURATION_S) -> EffectSignals:
    """Write the effect test signals into ``folder`` and describe where each one sits.

    ``duration_s`` is the length of the bounce this profile may use (the profiler's per-bounce
    cap): the layout is compressed proportionally when it is shorter than the default 9 s.
    Files: ``pink_noise.wav``, ``sine_sweep.wav``, ``clicks.wav`` and ``composite.wav``
    (32-bit float stereo).
    """
    sr = _check_sr(sr)
    cap = _check_cap(duration_s)
    scale = min(1.0, cap / _LAYOUT_BOUNCE_END_S)
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    def s(x: float) -> float:
        return x * scale

    lay = {name: tuple(s(v) for v in vals) for name, vals in _LAYOUT.items()}
    total = int(round(s(_LAYOUT_FILE_END_S) * sr))
    composite = np.zeros((2, total), dtype=np.float32)
    parts: dict[str, np.ndarray] = {
        "pink_noise": pink_noise(lay["noise"][1] - lay["noise"][0], sr),
        "sine_sweep": log_sweep(lay["sweep"][1] - lay["sweep"][0], sr),
        "clicks": drum_pattern(lay["clicks"][1] - lay["clicks"][0], sr),
    }
    placement = {"pink_noise": "noise", "sine_sweep": "sweep", "clicks": "clicks"}
    for part, name in placement.items():
        start = int(round(lay[name][0] * sr))
        block = parts[part]
        end = min(total, start + block.shape[1])
        composite[:, start:end] = block[:, : end - start]

    paths: dict[str, Path] = {}
    for part, block in parts.items():
        paths[part] = out_dir / f"{part}.wav"
        wav.write(paths[part], block, sr, "32f")
    composite_path = out_dir / "composite.wav"
    wav.write(composite_path, composite, sr, "32f")
    paths["composite"] = composite_path

    bounce_end = s(_LAYOUT_BOUNCE_END_S)
    segments = {name: (lay[name][2], min(lay[name][3], bounce_end)) for name in lay}
    return EffectSignals(
        path=composite_path,
        parts=paths,
        sample_rate=sr,
        file_duration_s=total / sr,
        bounce_end_s=bounce_end,
        segments=segments,
        sweep_range_hz=(20.0, min(20000.0, 0.45 * sr)),
        sweep_start_s=lay["sweep"][0],
        sweep_length_s=lay["sweep"][1] - lay["sweep"][0],
    )


def sweep_band_windows(signals: EffectSignals) -> list[tuple[float, float, float, float]]:
    """``(f_lo, f_hi, start_s, end_s)`` for each region of ``SWEEP_BAND_EDGES_HZ`` the sweep covers.

    The window of a region is the part of the sweep that passes through it (a log sweep spends
    equal time per octave), so the level in that window is the response around those frequencies.
    """
    f0, f1 = signals.sweep_range_hz
    k = math.log(f1 / f0)
    edges = [e for e in SWEEP_BAND_EDGES_HZ if f0 < e < f1]
    bounds = [f0, *edges, f1]
    out: list[tuple[float, float, float, float]] = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        t0 = signals.sweep_start_s + signals.sweep_length_s * math.log(lo / f0) / k
        t1 = signals.sweep_start_s + signals.sweep_length_s * math.log(hi / f0) / k
        out.append((lo, hi, t0, t1))
    return out


def build_instrument_plan(bpm: float = 120.0, duration_s: float = 6.0) -> InstrumentPlan:
    """A MIDI clip: a sustained C major chord (C4 E4 G4, 2 s) then one staccato note (C5, 0.12 s).

    ``duration_s`` caps the bounce length; the plan is compressed in time when it is shorter
    than the default 6 s (never below 3 s).
    """
    if not math.isfinite(bpm) or not 30 <= bpm <= 300:
        raise SignalError("bpm must be 30..300")
    cap = _check_cap(duration_s)
    scale = min(1.0, cap / 6.0)
    chord = tuple(PlanNote(p, 0.0, 2.0 * scale, 100) for p in (60, 64, 67))
    staccato = (PlanNote(72, 3.5 * scale, 0.12 * scale, 110),)
    return InstrumentPlan(
        notes=chord + staccato,
        segments={"chord": (0.05 * scale, 3.2 * scale), "staccato": (3.5 * scale, min(5.5 * scale, cap))},
        bounce_end_s=min(5.5 * scale, cap),
        bpm=float(bpm),
        names={"chord": "sustained C major chord", "staccato": "staccato C5"},
    )
