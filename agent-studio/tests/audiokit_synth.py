"""Audio synthesis helpers shared by the audiokit tests (no audio files are committed)."""

from __future__ import annotations

import numpy as np

SR = 44100


def sine(freq: float, seconds: float, sr: int = SR, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(round(seconds * sr))) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :]


def click_track(
    bpm: float, seconds: float, sr: int = SR, offset: float = 0.1, noise_db: float = -50.0
) -> tuple[np.ndarray, list[float]]:
    """Decaying 1.2 kHz blips on every beat over a quiet noise floor; also the beat times."""
    rng = np.random.default_rng(7)
    n = int(seconds * sr)
    out = rng.standard_normal(n) * 10 ** (noise_db / 20)
    blip_t = np.arange(int(0.05 * sr)) / sr
    blip = np.sin(2 * np.pi * 1200 * blip_t) * np.exp(-blip_t / 0.008)
    times: list[float] = []
    t = offset
    while t + 0.05 < seconds:
        i = int(round(t * sr))
        out[i : i + blip.size] += 0.8 * blip
        times.append(t)
        t += 60.0 / bpm
    return out.astype(np.float32)[None, :], times


def tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.3, harmonics: int = 4) -> np.ndarray:
    """A harmonic tone with a short attack and release, shaped (frames,)."""
    t = np.arange(int(seconds * sr)) / sr
    y = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, harmonics + 1))
    env = np.minimum(1.0, np.minimum(t / 0.01, (seconds - t) / 0.02))
    return (amp * y * env / harmonics * 2).astype(np.float32)


def midi_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


def chord(midi_notes: list[int], seconds: float, sr: int = SR) -> np.ndarray:
    return sum(tone(midi_hz(m), seconds, sr) for m in midi_notes)
