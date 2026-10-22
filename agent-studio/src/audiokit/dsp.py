"""Core DSP: resampling, levels, loudness (ITU-R BS.1770-4), trimming, fades, normalisation.

Audio is ``float32`` shaped ``(channels, frames)``. All functions are pure and return new
arrays. Level measurements of silent or empty audio return ``FLOOR_DB`` (a finite value, so
results stay JSON-serialisable) instead of ``-inf``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

FLOOR_DB = -200.0
MAX_SAMPLE_RATE = 768_000
_MAX_POLYPHASE_FACTOR = 2000
_MAX_NORMALIZE_GAIN_DB = 60.0


def check_sr(sr: int) -> int:
    """Validate a sample rate and return it as a plain ``int``."""
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)):
        raise ValueError("sample rate must be an integer")
    if not 1 <= int(sr) <= MAX_SAMPLE_RATE:
        raise ValueError(f"sample rate out of range: {sr}")
    return int(sr)


def as_audio(audio: np.ndarray) -> np.ndarray:
    """Return ``audio`` as finite ``float32`` ``(channels, frames)`` (1-D input becomes mono)."""
    arr = np.asarray(audio)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or arr.shape[0] < 1:
        raise ValueError("audio must be shaped (channels, frames)")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _finite_ms(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be a finite, non-negative number")
    return v


def _to_db(linear: float) -> float:
    return FLOOR_DB if linear <= 1e-10 else max(FLOOR_DB, 20.0 * math.log10(linear))


def _gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return (audio * np.float32(10.0 ** (gain_db / 20.0))).astype(np.float32)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample from ``sr_in`` to ``sr_out`` (Kaiser window, ~90 dB stopband)."""
    a = as_audio(audio)
    sr_in, sr_out = check_sr(sr_in), check_sr(sr_out)
    if sr_in == sr_out or a.shape[1] == 0:
        return a.copy()
    g = math.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    if max(up, down) > _MAX_POLYPHASE_FACTOR:  # odd ratio: avoid an enormous filter
        n_out = max(1, int(round(a.shape[1] * sr_out / sr_in)))
        return signal.resample(a, n_out, axis=1).astype(np.float32)
    out = signal.resample_poly(a, up, down, axis=1, window=("kaiser", 9.0))
    return np.ascontiguousarray(out, dtype=np.float32)


def to_channels(audio: np.ndarray, n: int) -> np.ndarray:
    """Convert to ``n`` channels: downmix to mono by averaging, keep the first ``n`` channels
    when reducing to more than one, and cycle existing channels when expanding."""
    a = as_audio(audio)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= n <= 64:
        raise ValueError("channel count must be an integer from 1 to 64")
    c = a.shape[0]
    if c == n:
        return a.copy()
    if n == 1:
        return a.mean(axis=0, keepdims=True, dtype=np.float32).astype(np.float32)
    if n < c:
        return a[:n].copy()
    return np.ascontiguousarray(a[np.arange(n) % c])


def peak_dbfs(audio: np.ndarray) -> float:
    """Sample peak in dBFS."""
    a = as_audio(audio)
    return _to_db(float(np.max(np.abs(a)))) if a.size else FLOOR_DB


def rms_dbfs(audio: np.ndarray) -> float:
    """RMS over all channels in dBFS."""
    a = as_audio(audio)
    if not a.size:
        return FLOOR_DB
    return _to_db(math.sqrt(float(np.mean(np.square(a, dtype=np.float64)))))


def true_peak_dbfs(audio: np.ndarray) -> float:
    """True peak in dBFS using 4x oversampling (never below the sample peak)."""
    a = as_audio(audio)
    if not a.size:
        return FLOOR_DB
    peak = float(np.max(np.abs(a)))
    for ch in a:
        up = signal.resample_poly(ch.astype(np.float64), 4, 1)
        peak = max(peak, float(np.max(np.abs(up))))
    return _to_db(peak)


def _k_weighting(sr: int) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """BS.1770-4 K-weighting biquads (shelf, high-pass) for any sample rate."""
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf_b = np.array([(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0])
    shelf_a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / sr)
    a0 = 1.0 + k / q + k * k
    hp_b = np.array([1.0, -2.0, 1.0])
    hp_a = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    return (shelf_b, shelf_a), (hp_b, hp_a)


def _channel_weights(channels: int) -> np.ndarray:
    if channels == 5:
        return np.array([1.0, 1.0, 1.0, 1.41, 1.41])
    if channels == 6:  # 5.1: LFE excluded
        return np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41])
    return np.ones(channels)


def lufs_integrated(audio: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS per ITU-R BS.1770-4: K-weighting, 400 ms blocks with 75 %
    overlap, -70 LUFS absolute gate and -10 LU relative gate. Audio shorter than one block is
    measured as a single block. Silence returns ``FLOOR_DB``."""
    a = as_audio(audio)
    sr = check_sr(sr)
    n = a.shape[1]
    if n == 0:
        return FLOOR_DB
    (sb, sa), (hb, ha) = _k_weighting(sr)
    y = a.astype(np.float64)
    y = signal.lfilter(hb, ha, signal.lfilter(sb, sa, y, axis=1), axis=1)
    block = max(1, int(round(0.4 * sr)))
    step = max(1, int(round(0.1 * sr)))
    cs = np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(y * y, axis=1)], axis=1)
    if n >= block:
        starts = np.arange(0, n - block + 1, step)
        ends = starts + block
    else:
        starts, ends = np.array([0]), np.array([n])
    ms = (cs[:, ends] - cs[:, starts]) / (ends - starts)
    z = np.maximum((_channel_weights(a.shape[0])[:, None] * ms).sum(axis=0), 0.0)
    loud = -0.691 + 10.0 * np.log10(np.maximum(z, 1e-30))
    above_abs = z[loud > -70.0]
    if above_abs.size == 0:
        return FLOOR_DB
    rel_threshold = -0.691 + 10.0 * math.log10(float(above_abs.mean())) - 10.0
    gated = z[(loud > -70.0) & (loud > rel_threshold)]
    if gated.size == 0:
        return FLOOR_DB
    return max(FLOOR_DB, -0.691 + 10.0 * math.log10(float(gated.mean())))


def trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -60.0,
    pad_ms: float = 5.0,
    head: bool = True,
    tail: bool = True,
) -> np.ndarray:
    """Remove leading/trailing audio below ``threshold_db``, keeping ``pad_ms`` of padding.
    ``head``/``tail`` select which ends to trim. All-silent audio returns zero frames."""
    a = as_audio(audio)
    sr = check_sr(sr)
    pad = int(round(_finite_ms(pad_ms, "pad_ms") * sr / 1000.0))
    thr = 10.0 ** (float(threshold_db) / 20.0)
    loud = np.flatnonzero(np.max(np.abs(a), axis=0) > thr)
    if loud.size == 0:
        return a[:, :0].copy()
    start = max(0, int(loud[0]) - pad) if head else 0
    end = min(a.shape[1], int(loud[-1]) + 1 + pad) if tail else a.shape[1]
    return a[:, start:end].copy()


def fade(audio: np.ndarray, sr: int, in_ms: float, out_ms: float) -> np.ndarray:
    """Apply equal-power (sine / cosine) fade-in and fade-out."""
    a = as_audio(audio).copy()
    sr = check_sr(sr)
    n = a.shape[1]
    n_in = min(n, int(round(_finite_ms(in_ms, "in_ms") * sr / 1000.0)))
    n_out = min(n, int(round(_finite_ms(out_ms, "out_ms") * sr / 1000.0)))
    if n_in > 0:
        a[:, :n_in] *= np.sin((np.arange(n_in) + 0.5) / n_in * (np.pi / 2)).astype(np.float32)
    if n_out > 0:
        a[:, n - n_out :] *= np.cos((np.arange(n_out) + 0.5) / n_out * (np.pi / 2)).astype(np.float32)
    return a


def normalize_peak(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale so the sample peak equals ``target_dbfs`` (silence is returned unchanged)."""
    a = as_audio(audio)
    target = float(target_dbfs)
    if not math.isfinite(target):
        raise ValueError("target_dbfs must be finite")
    peak = peak_dbfs(a)
    if peak <= FLOOR_DB:
        return a.copy()
    return _gain(a, max(-_MAX_NORMALIZE_GAIN_DB, min(_MAX_NORMALIZE_GAIN_DB, target - peak)))


def normalize_lufs(audio: np.ndarray, sr: int, target_lufs: float, ceiling_dbfs: float = -1.0) -> np.ndarray:
    """Scale to ``target_lufs`` integrated loudness, reducing the gain if that would push the
    true peak above ``ceiling_dbfs``. Silence is returned unchanged."""
    a = as_audio(audio)
    target, ceiling = float(target_lufs), float(ceiling_dbfs)
    if not (math.isfinite(target) and math.isfinite(ceiling)):
        raise ValueError("target_lufs and ceiling_dbfs must be finite")
    loudness = lufs_integrated(a, sr)
    if loudness <= FLOOR_DB:
        return a.copy()
    gain_db = target - loudness
    tp = true_peak_dbfs(a)
    if tp + gain_db > ceiling:
        gain_db = ceiling - tp
    return _gain(a, max(-_MAX_NORMALIZE_GAIN_DB, min(_MAX_NORMALIZE_GAIN_DB, gain_db)))


def loop_crossfade(audio: np.ndarray, sr: int, ms: float) -> np.ndarray:
    """Make a loop seamless: the last ``ms`` fade out under the first ``ms`` fading in
    (equal-power). The result is that much shorter, and its end flows into its start."""
    a = as_audio(audio)
    sr = check_sr(sr)
    n = a.shape[1]
    length = min(int(round(_finite_ms(ms, "ms") * sr / 1000.0)), n // 2)
    if length <= 0:
        return a.copy()
    t = (np.arange(length) + 0.5) / length * (np.pi / 2)
    out = a[:, : n - length].copy()
    out[:, :length] = a[:, :length] * np.sin(t).astype(np.float32) + a[:, n - length :] * np.cos(t).astype(np.float32)
    return out


def dc_offset(audio: np.ndarray) -> float:
    """Mean sample value of the channel with the largest absolute mean (signed)."""
    a = as_audio(audio)
    if not a.size:
        return 0.0
    means = a.mean(axis=1, dtype=np.float64)
    return float(means[int(np.argmax(np.abs(means)))])


def clipped_samples(audio: np.ndarray, threshold: float = 0.999) -> int:
    """Count samples whose magnitude is at least ``threshold``."""
    a = as_audio(audio)
    return int(np.count_nonzero(np.abs(a) >= float(threshold)))
