"""Monophonic pitch tracking (YIN) and note segmentation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sfft

from . import dsp

WINDOW = 2048
HOP = 256
FMIN_HZ = 40.0
FMAX_HZ = 1500.0
MAX_DURATION_S = 1800.0
_ABS_THRESHOLD = 0.15  # YIN absolute threshold for the first dip
_VOICED_MAX = 0.30  # frames whose best normalised difference exceeds this are unvoiced
_SILENCE_RMS = 1e-3  # about -60 dBFS
_SWITCH_FRAMES = 3  # frames that must agree before a new pitch starts a new note
_MAX_GAP_S = 0.03  # unvoiced gaps this short do not end a note

Frame = tuple[float, float | None, float]


@dataclass(frozen=True)
class Note:
    """A detected or parsed note. ``pitch`` is a MIDI note number, times are in seconds."""

    pitch: int
    start_s: float
    end_s: float
    velocity: int = 80
    confidence: float = 1.0


def hz_to_midi(freq: float) -> float:
    """Fractional MIDI note number of ``freq`` (A4 = 440 Hz = 69)."""
    return 69.0 + 12.0 * math.log2(freq / 440.0)


def track(audio: np.ndarray, sr: int) -> list[Frame]:
    """YIN pitch tracker: ``[(time_s, f0_hz or None, confidence)]`` for every 256-sample hop,
    2048-sample window, 40-1500 Hz. ``None`` marks unvoiced or silent frames."""
    a = dsp.as_audio(audio)
    sr = dsp.check_sr(sr)
    if a.shape[1] > MAX_DURATION_S * sr:
        raise ValueError(f"audio longer than {MAX_DURATION_S:.0f} s")
    x = a.mean(axis=0, dtype=np.float32).astype(np.float64)
    n = x.size
    if n == 0:
        return []
    tau_min = max(2, int(sr / FMAX_HZ))
    tau_max = int(math.ceil(sr / FMIN_HZ))
    length = WINDOW + tau_max
    pad = length // 2
    xp = np.pad(x, (pad, length - pad))
    view = sliding_window_view(xp, length)[::HOP][: n // HOP + 1]
    nfft = sfft.next_fast_len(length + WINDOW)
    taus = np.arange(1, tau_max + 1)
    frames: list[Frame] = []
    for s in range(0, view.shape[0], 64):
        seg = view[s : s + 64]
        head = seg[:, :WINDOW]
        cross = sfft.irfft(np.conj(sfft.rfft(head, nfft, axis=1)) * sfft.rfft(seg, nfft, axis=1), nfft, axis=1)
        cs = np.concatenate([np.zeros((seg.shape[0], 1)), np.cumsum(seg * seg, axis=1)], axis=1)
        e_head = cs[:, WINDOW : WINDOW + 1]
        e_lag = cs[:, WINDOW : WINDOW + tau_max + 1] - cs[:, : tau_max + 1]
        diff = np.maximum(e_head + e_lag - 2.0 * cross[:, : tau_max + 1], 0.0)
        csum = np.cumsum(diff[:, 1:], axis=1)
        cmnd = np.ones_like(diff)
        cmnd[:, 1:] = np.where(csum > 1e-12, diff[:, 1:] * taus / np.maximum(csum, 1e-12), 1.0)
        rms = np.sqrt(e_head[:, 0] / WINDOW)
        for k in range(seg.shape[0]):
            t = (s + k) * HOP / sr
            frames.append(_pick(cmnd[k], rms[k], tau_min, tau_max, sr, t))
    return frames


def _pick(cmnd: np.ndarray, rms: float, tau_min: int, tau_max: int, sr: int, t: float) -> Frame:
    if rms < _SILENCE_RMS:
        return (t, None, 0.0)
    region = cmnd[tau_min : tau_max + 1]
    below = np.flatnonzero(region < _ABS_THRESHOLD)
    if below.size:
        tau = tau_min + int(below[0])
        while tau + 1 <= tau_max and cmnd[tau + 1] < cmnd[tau]:
            tau += 1
    else:
        tau = tau_min + int(np.argmin(region))
    best = float(cmnd[tau])
    confidence = float(min(1.0, max(0.0, 1.0 - best)))
    if best > _VOICED_MAX:
        return (t, None, confidence)
    refined = float(tau)
    if tau_min < tau < tau_max:
        y0, y1, y2 = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom > 1e-12:
            refined = tau + float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    f0 = sr / refined
    if not FMIN_HZ <= f0 <= FMAX_HZ:
        return (t, None, confidence)
    return (t, float(f0), confidence)


def _velocity_from_audio(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> int:
    seg = audio[:, int(start_s * sr) : max(int(end_s * sr), int(start_s * sr) + 1)]
    level = dsp.rms_dbfs(seg)
    return int(np.clip(round(30 + (level + 40.0) / 34.0 * 90.0), 1, 127))


def notes(
    frames: Sequence[Frame],
    min_note_ms: float = 60.0,
    merge_glide_cents: float = 80.0,
    audio: np.ndarray | None = None,
    sr: int | None = None,
) -> list[Note]:
    """Turn pitch frames into notes.

    Frames within ``merge_glide_cents`` of the current note's median pitch (vibrato, glides)
    belong to it; a different pitch must persist for 3 frames to start a new note, so
    one-frame octave glitches are ignored. Unvoiced gaps up to 30 ms are bridged and notes
    shorter than ``min_note_ms`` are dropped. Velocity is a constant 80 unless ``audio`` and
    ``sr`` are given, in which case it follows the note's loudness.
    """
    if len(frames) < 2:
        return []
    step = float(np.median(np.diff([f[0] for f in frames])))
    max_gap = max(1, int(round(_MAX_GAP_S / step))) if step > 0 else 1
    glide = float(merge_glide_cents) / 100.0
    segments: list[list[tuple[float, float, float]]] = []  # (time, midi, conf) per note
    cur: list[tuple[float, float, float]] = []
    pend: list[tuple[float, float, float]] = []
    gap = 0

    def close() -> None:
        nonlocal cur, pend
        if cur:
            segments.append(cur)
        cur, pend = [], []

    for t, f0, conf in frames:
        if f0 is None or f0 <= 0:
            gap += 1
            if gap > max_gap:
                close()
            continue
        gap = 0
        midi = hz_to_midi(f0)
        if not cur:
            cur = [(t, midi, conf)]
            continue
        ref = float(np.median([m for _, m, _ in cur]))
        if abs(midi - ref) <= glide:
            cur.append((t, midi, conf))
            pend = []
        else:
            if pend and abs(midi - float(np.median([m for _, m, _ in pend]))) > glide:
                pend = []
            pend.append((t, midi, conf))
            if len(pend) >= _SWITCH_FRAMES:
                segments.append(cur)
                cur, pend = pend, []
    close()

    result: list[Note] = []
    for seg in segments:
        start = max(0.0, seg[0][0] - step / 2.0)
        end = seg[-1][0] + step / 2.0
        if (end - start) * 1000.0 < min_note_ms:
            continue
        pitch = int(np.clip(round(float(np.median([m for _, m, _ in seg]))), 0, 127))
        velocity = 80
        if audio is not None and sr is not None:
            velocity = _velocity_from_audio(dsp.as_audio(audio), dsp.check_sr(sr), start, end)
        confidence = float(np.mean([c for _, _, c in seg]))
        result.append(Note(pitch, round(start, 4), round(end, 4), velocity, round(confidence, 3)))
    return result
