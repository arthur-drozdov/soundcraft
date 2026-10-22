"""Test-signal synthesis for the audiokit tests (audio is never committed; everything is generated)."""

from __future__ import annotations

import numpy as np

SR = 44100


def sine(freq: float, seconds: float, sr: int = SR, amp: float = 1.0, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(round(seconds * sr))) / sr
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)[None, :]


def click_track(bpm: float, seconds: float, sr: int = SR, offset: float = 0.1, noise_db: float = -50.0) -> np.ndarray:
    """A decaying 1 kHz blip on every beat over a little noise."""
    rng = np.random.default_rng(1)
    n = int(seconds * sr)
    out = 10 ** (noise_db / 20) * rng.standard_normal(n)
    blip_t = np.arange(int(0.03 * sr)) / sr
    blip = np.sin(2 * np.pi * 1000 * blip_t) * np.exp(-blip_t / 0.006)
    beat = 60.0 / bpm
    k = 0
    while int((offset + k * beat) * sr) + blip.size < n:
        start = int((offset + k * beat) * sr)
        out[start : start + blip.size] += 0.8 * blip
        k += 1
    return out.astype(np.float32)[None, :]


def beat_times(bpm: float, seconds: float, sr: int = SR, offset: float = 0.1) -> list[float]:
    beat, n = 60.0 / bpm, int(seconds * sr)
    out, k = [], 0
    while int((offset + k * beat) * sr) + int(0.03 * sr) < n:
        out.append(offset + k * beat)
        k += 1
    return out


def tone_note(freq: float, seconds: float, sr: int = SR, harmonics: int = 4, amp: float = 0.2) -> np.ndarray:
    """A harmonic tone with a short attack and release, shaped (frames,)."""
    t = np.arange(int(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * freq * (h + 1) * t) / (h + 1) for h in range(harmonics))
    env = np.minimum(1.0, np.minimum(t / 0.01, (seconds - t) / 0.02))
    return amp * x * env


def chord_progression(chords: list[list[float]], seconds_each: float = 1.0, sr: int = SR) -> np.ndarray:
    parts = [sum(tone_note(f, seconds_each, sr) for f in chord) for chord in chords]
    return np.concatenate(parts).astype(np.float32)[None, :]


def midi_to_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)
