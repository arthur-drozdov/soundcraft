"""Audio analysis: onsets, tempo, key, spectral shape, envelope, decay, loop seam, summary.

Spectral analyses look at no more than the first ``MAX_ANALYSIS_S`` seconds of the input and
work through the signal in chunks, so memory stays bounded for long files.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sfft
from scipy import signal
from scipy.ndimage import uniform_filter1d

from . import dsp

MAX_ANALYSIS_S = 300.0
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_MIN_FLUX = 0.05


def _mono(audio: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 mix of at most ``MAX_ANALYSIS_S`` seconds."""
    a = dsp.as_audio(audio)
    dsp.check_sr(sr)
    a = a[:, : int(MAX_ANALYSIS_S * sr)]
    return a.mean(axis=0, dtype=np.float32).astype(np.float32)


def _pow2_near(value: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, 1 << int(round(math.log2(max(value, 1.0)))))))


def _mag_chunks(x: np.ndarray, n_fft: int, hop: int, chunk: int = 512) -> Iterator[np.ndarray]:
    """Yield Hann-windowed STFT magnitudes ``(frames, bins)`` for centred frames."""
    if x.size == 0:
        return
    pad = n_fft // 2
    mode = "reflect" if x.size > pad else "constant"
    xp = np.pad(x, (pad, pad), mode=mode)
    view = sliding_window_view(xp, n_fft)[::hop]
    win = signal.get_window("hann", n_fft).astype(np.float32)
    for s in range(0, view.shape[0], chunk):
        yield np.abs(sfft.rfft(view[s : s + chunk] * win, axis=1)).astype(np.float32)


def _onset_strength(x: np.ndarray, sr: int) -> tuple[np.ndarray, int, int, int]:
    """Spectral-flux onset strength. Returns ``(flux, hop, n_fft, frame_count)``."""
    n_fft = _pow2_near(0.023 * sr, 256, 4096)
    hop = max(1, int(round(sr / 200.0)))
    parts: list[np.ndarray] = []
    prev: np.ndarray | None = None
    scale = np.float32(1000.0 / (n_fft / 4.0))
    for mags in _mag_chunks(x, n_fft, hop):
        logm = np.log1p(mags * scale)
        stacked = logm if prev is None else np.vstack([prev, logm])
        diff = np.maximum(np.diff(stacked, axis=0), 0.0).mean(axis=1)
        parts.append(np.concatenate([[0.0], diff]) if prev is None else diff)
        prev = logm[-1:]
    flux = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return flux.astype(np.float64), hop, n_fft, flux.size


def _refine_onset(x: np.ndarray, sr: int, centre: int, half: int) -> int:
    """Move an onset estimate to the steepest rise of the smoothed envelope near ``centre``."""
    lo, hi = max(0, centre - half), min(x.size, centre + 2 * half)
    seg = np.abs(x[lo:hi])
    width = max(1, int(sr * 0.002))
    if seg.size <= width + 1:
        return centre
    env = uniform_filter1d(seg.astype(np.float64), size=width, mode="nearest")
    rise = env[width:] - env[:-width]
    return lo + int(np.argmax(rise)) + width // 2


def onsets(audio: np.ndarray, sr: int) -> list[float]:
    """Onset times in seconds (spectral flux with an adaptive threshold)."""
    x = _mono(audio, sr)
    flux, hop, n_fft, _ = _onset_strength(x, sr)
    if flux.size < 3 or float(flux.max()) < _MIN_FLUX:
        return []
    frame_rate = sr / hop
    local = uniform_filter1d(flux, size=max(3, int(0.4 * frame_rate)), mode="nearest")
    thr = local + 0.15 * float(np.percentile(flux, 99))
    peaks, _ = signal.find_peaks(flux, distance=max(1, int(0.05 * frame_rate)))
    times: list[float] = []
    for p in peaks:
        if flux[p] > thr[p] and flux[p] > _MIN_FLUX:
            sample = _refine_onset(x, sr, int(p) * hop, n_fft // 2)
            times.append(round(sample / sr, 4))
    lead = x[: max(1, int(0.02 * sr))]
    starts_loud = float(np.sqrt(np.mean(lead.astype(np.float64) ** 2))) > 0.1 * float(np.max(np.abs(x)))
    if starts_loud and (not times or times[0] > 0.05):
        times.insert(0, 0.0)  # audio that begins mid-sound has no flux peak at the very start
    return times


def _fold_bpm(value: float, lo: float, hi: float) -> float:
    while value < 80.0 and value * 2.0 <= hi:
        value *= 2.0
    while value > 160.0 and value / 2.0 >= lo:
        value /= 2.0
    return value


def bpm(audio: np.ndarray, sr: int, lo: float = 60.0, hi: float = 200.0) -> tuple[float, float]:
    """Estimate tempo as ``(bpm, confidence 0..1)``.

    Autocorrelation of the onset envelope scored over beat multiples, then octave-corrected
    toward 80-160 BPM within ``[lo, hi]``. Returns ``(0.0, 0.0)`` when there is no rhythm.
    """
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi <= 1000):
        raise ValueError("bpm range must satisfy 0 < lo < hi <= 1000")
    x = _mono(audio, sr)
    flux, hop, _, _ = _onset_strength(x, sr)
    n = flux.size
    frame_rate = sr / hop
    if n < 8 or float(flux.max()) < _MIN_FLUX:
        return 0.0, 0.0
    env = np.maximum(flux - uniform_filter1d(flux, size=max(3, int(frame_rate)), mode="nearest"), 0.0)
    env -= env.mean()
    spec = sfft.rfft(env, 2 * n)
    acf = sfft.irfft(spec * np.conj(spec))[:n]
    acf = acf / np.maximum(n - np.arange(n), 1)
    if acf[0] <= 1e-12:
        return 0.0, 0.0
    acf /= acf[0]
    grid = np.arange(lo, hi + 1e-9, 0.1)
    lags = 60.0 * frame_rate / grid
    lag_axis = np.arange(n)
    score = np.zeros_like(grid)
    total = np.zeros_like(grid)
    for mult, weight in ((1, 1.0), (2, 0.5), (4, 0.25)):
        ok = mult * lags <= n / 2
        score[ok] += weight * np.interp(mult * lags[ok], lag_axis, acf)
        total[ok] += weight
    valid = total >= 1.0  # the fundamental lag must fit
    if not valid.any():
        return 0.0, 0.0
    avg = np.where(valid, score / np.maximum(total, 1e-9), -np.inf)
    best = int(np.argmax(avg))
    confidence = float(np.clip(avg[best], 0.0, 1.0))
    if confidence <= 0.0:
        return 0.0, 0.0
    return round(_fold_bpm(float(grid[best]), lo, hi), 2), round(confidence, 3)


def key(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """Estimate the musical key as ``("F minor", confidence 0..1)`` from an STFT chroma vector
    correlated with the Krumhansl-Schmuckler profiles over 24 keys. Returns ``("", 0.0)`` for
    silence."""
    x = _mono(audio, sr)
    n_fft = _pow2_near(0.093 * sr, 512, 16384)
    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for mags in _mag_chunks(x, n_fft, n_fft // 2):
        acc += np.sqrt(mags).sum(axis=0)
    freqs = np.arange(acc.size) * sr / n_fft
    sel = (freqs >= 100.0) & (freqs <= min(4500.0, sr / 2.0))
    if not sel.any() or acc[sel].sum() <= 1e-9:
        return "", 0.0
    pitch_class = np.round(69.0 + 12.0 * np.log2(freqs[sel] / 440.0)).astype(int) % 12
    chroma = np.bincount(pitch_class, weights=acc[sel], minlength=12)
    corr = np.full(24, -1.0)
    for tonic in range(12):
        for mode, prof in enumerate((_KS_MAJOR, _KS_MINOR)):
            rolled = np.roll(prof, tonic)
            if chroma.std() > 0:
                corr[tonic * 2 + mode] = float(np.corrcoef(chroma, rolled)[0, 1])
    order = np.argsort(corr)[::-1]
    best, second = float(corr[order[0]]), float(corr[order[1]])
    tonic, mode = divmod(int(order[0]), 2)
    confidence = float(np.clip(best, 0.0, 1.0) * min(1.0, 0.5 + 5.0 * (best - second)))
    return f"{NOTE_NAMES[tonic]} {'minor' if mode else 'major'}", round(confidence, 3)


def _mean_spectrum(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    x = _mono(audio, sr)
    n_fft = 2048
    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    count = 0
    for mags in _mag_chunks(x, n_fft, n_fft // 2):
        acc += mags.sum(axis=0)
        count += mags.shape[0]
    freqs = np.arange(acc.size) * sr / n_fft
    return freqs, acc / max(count, 1)


def centroid_hz(audio: np.ndarray, sr: int) -> float:
    """Spectral centroid (Hz) of the mean magnitude spectrum; 0 for silence."""
    freqs, mag = _mean_spectrum(audio, sr)
    total = float(mag.sum())
    return round(float((freqs * mag).sum() / total), 2) if total > 1e-9 else 0.0


def bandwidth_hz(audio: np.ndarray, sr: int) -> float:
    """Spectral bandwidth (Hz): magnitude-weighted standard deviation around the centroid."""
    freqs, mag = _mean_spectrum(audio, sr)
    total = float(mag.sum())
    if total <= 1e-9:
        return 0.0
    c = float((freqs * mag).sum() / total)
    return round(math.sqrt(float((mag * (freqs - c) ** 2).sum() / total)), 2)


def envelope(audio: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    """Linear RMS amplitude over consecutive, non-overlapping ``ms`` windows (all channels)."""
    a = dsp.as_audio(audio)
    sr = dsp.check_sr(sr)
    n = a.shape[1]
    win = max(1, int(round(float(ms) * sr / 1000.0)))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    energy = np.sum(np.square(a, dtype=np.float64), axis=0)
    starts = np.arange(0, n, win)
    counts = np.diff(np.append(starts, n)) * a.shape[0]
    return np.sqrt(np.add.reduceat(energy, starts) / counts).astype(np.float32)


def decay_time_s(audio: np.ndarray, sr: int, drop_db: float = -40.0) -> float:
    """Seconds from the envelope peak until it first falls ``drop_db`` below it. If it never
    does, the time remaining after the peak is returned."""
    ms = 5.0
    env = envelope(audio, sr, ms).astype(np.float64)
    if env.size == 0 or float(env.max()) <= 0.0:
        return 0.0
    peak = int(np.argmax(env))
    thr = env[peak] * 10.0 ** (-abs(float(drop_db)) / 20.0)
    below = np.flatnonzero(env[peak:] <= thr)
    idx = int(below[0]) if below.size else env.size - peak
    return round(idx * ms / 1000.0, 4)


def loop_seam_score(audio: np.ndarray, sr: int) -> float:
    """0..1 score for how well the end flows into the start when looped: spectral similarity of
    the last and first ~50 ms, level match, and waveform continuity across the boundary."""
    a = dsp.as_audio(audio)
    sr = dsp.check_sr(sr)
    n = a.shape[1]
    if n < 128:
        return 0.0
    w = int(np.clip(int(0.05 * sr), 64, n // 2))
    tail, head = a[:, n - w :].astype(np.float64), a[:, :w].astype(np.float64)
    hann = np.hanning(w)
    spec_t = np.sqrt(np.abs(np.fft.rfft(tail * hann, axis=1))).ravel()
    spec_h = np.sqrt(np.abs(np.fft.rfft(head * hann, axis=1))).ravel()
    norm = float(np.linalg.norm(spec_t) * np.linalg.norm(spec_h))
    spectral = float(np.dot(spec_t, spec_h) / norm) if norm > 1e-12 else 1.0
    k = max(8, int(0.02 * sr))
    rms_t = math.sqrt(float(np.mean(tail[:, -k:] ** 2)))
    rms_h = math.sqrt(float(np.mean(head[:, :k] ** 2)))
    if max(rms_t, rms_h) < 1e-6:
        level = 1.0
    else:
        diff_db = abs(20.0 * math.log10(max(rms_t, 1e-6) / max(rms_h, 1e-6)))
        level = math.exp(-diff_db / 6.0)
    # linear extrapolation across the boundary in both directions (catches steps and kinks)
    fwd = np.abs(head[:, 0] - (2.0 * tail[:, -1] - tail[:, -2]))
    bwd = np.abs(tail[:, -1] - (2.0 * head[:, 0] - head[:, 1]))
    err = float(np.max(np.maximum(fwd, bwd)))
    slopes = np.concatenate([np.abs(np.diff(tail[:, -k:], axis=1)).ravel(), np.abs(np.diff(head[:, :k], axis=1)).ravel()])
    typical = float(np.sqrt(np.mean(slopes**2)))
    jump = 1.0 / (1.0 + (err / (0.5 * typical + 1e-4)) ** 2)
    return round(float(np.clip(0.4 * spectral + 0.3 * level + 0.3 * jump, 0.0, 1.0)), 3)


def summary(audio: np.ndarray, sr: int) -> dict[str, object]:
    """Everything in the ``analyze.audio`` result as JSON-safe values (``onsets_s`` and its
    alias ``onsets`` are lists of seconds)."""
    a = dsp.as_audio(audio)
    sr = dsp.check_sr(sr)
    onset_times = onsets(a, sr)
    tempo, tempo_conf = bpm(a, sr)
    key_name, key_conf = key(a, sr)
    return {
        "duration_s": round(a.shape[1] / sr, 4),
        "sample_rate": sr,
        "channels": int(a.shape[0]),
        "peak_dbfs": round(dsp.peak_dbfs(a), 2),
        "true_peak_dbfs": round(dsp.true_peak_dbfs(a), 2),
        "rms_dbfs": round(dsp.rms_dbfs(a), 2),
        "lufs": round(dsp.lufs_integrated(a, sr), 2),
        "onsets_s": onset_times,
        "onsets": list(onset_times),
        "bpm": tempo,
        "bpm_confidence": tempo_conf,
        "key": key_name,
        "key_confidence": key_conf,
        "centroid_hz": centroid_hz(a, sr),
        "bandwidth_hz": bandwidth_hz(a, sr),
        "clipped_samples": dsp.clipped_samples(a),
        "dc_offset": round(dsp.dc_offset(a), 6),
        "loop_seam": loop_seam_score(a, sr),
    }
