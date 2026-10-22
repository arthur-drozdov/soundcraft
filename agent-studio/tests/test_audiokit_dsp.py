from __future__ import annotations

import numpy as np
import pytest
from scipy.signal.windows import tukey

from ak_signals import SR, sine
from audiokit import dsp


def _peak_freq(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x[0] * np.hanning(x.shape[1]), 1 << 18))
    k = int(np.argmax(spec))
    y0, y1, y2 = np.log(spec[k - 1 : k + 2] + 1e-12)
    return (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * sr / (1 << 18)


def test_lufs_full_scale_sine() -> None:
    assert dsp.lufs_integrated(sine(997, 5), SR) == pytest.approx(-3.01, abs=0.1)
    assert dsp.lufs_integrated(sine(997, 5, sr=48000), 48000) == pytest.approx(-3.01, abs=0.1)


def test_lufs_minus_20_dbfs_sine() -> None:
    assert dsp.lufs_integrated(sine(997, 5, amp=0.1), SR) == pytest.approx(-23.01, abs=0.1)


def test_lufs_stereo_sums_channels() -> None:
    mono = sine(997, 5, amp=0.1)
    assert dsp.lufs_integrated(np.vstack([mono, mono]), SR) == pytest.approx(-20.0, abs=0.1)


def test_lufs_gating_drops_silence_and_quiet_parts() -> None:
    tone = sine(997, 10, amp=0.1)
    ref = dsp.lufs_integrated(tone, SR)
    padded = np.concatenate([tone, np.zeros((1, 10 * SR), np.float32)], axis=1)
    assert dsp.lufs_integrated(padded, SR) == pytest.approx(ref, abs=0.1)
    # a part more than 10 LU below the loud part is dropped by the relative gate
    quiet = sine(997, 10, amp=0.1 * 10 ** (-30 / 20))
    assert dsp.lufs_integrated(np.concatenate([tone, quiet], axis=1), SR) == pytest.approx(ref, abs=0.1)
    assert dsp.lufs_integrated(np.zeros((1, SR), np.float32), SR) == dsp.FLOOR_DB
    assert dsp.lufs_integrated(np.zeros((1, 0), np.float32), SR) == dsp.FLOOR_DB


def test_level_measures() -> None:
    x = sine(1000, 1, amp=0.5)
    assert dsp.peak_dbfs(x) == pytest.approx(-6.02, abs=0.01)
    assert dsp.rms_dbfs(x) == pytest.approx(-9.03, abs=0.01)
    assert dsp.peak_dbfs(np.zeros((1, 10), np.float32)) == dsp.FLOOR_DB


def test_true_peak_exceeds_sample_peak_for_intersample_overs() -> None:
    # fs/4 sine sampled 45 degrees off its crest: every sample is 0.707 of the true peak
    x = sine(SR / 4, 0.2, amp=1.0, phase=np.pi / 4) * tukey(int(0.2 * SR), 0.3).astype(np.float32)
    assert dsp.peak_dbfs(x) == pytest.approx(-3.01, abs=0.05)
    assert dsp.true_peak_dbfs(x) == pytest.approx(0.0, abs=0.1)
    assert dsp.true_peak_dbfs(x) >= dsp.peak_dbfs(x)


@pytest.mark.parametrize("sr_out", [48000, 22050, 96000, 8000])
def test_resample_preserves_tone_frequency(sr_out: int) -> None:
    x = sine(1000, 1.0, amp=0.5)
    y = dsp.resample(x, SR, sr_out)
    assert y.dtype == np.float32 and y.shape == (1, int(np.ceil(SR and sr_out)))
    assert _peak_freq(y[:, sr_out // 10 : -sr_out // 10], sr_out) == pytest.approx(1000, abs=1.0)
    assert dsp.peak_dbfs(y[:, sr_out // 10 : -sr_out // 10]) == pytest.approx(-6.02, abs=0.1)


def test_resample_awkward_ratio_and_identity() -> None:
    x = sine(500, 0.5)
    y = dsp.resample(x, 44101, 48000)
    assert y.shape[1] == pytest.approx(0.5 * SR * 48000 / 44101 * 1.0, rel=0.01)
    same = dsp.resample(x, SR, SR)
    assert np.array_equal(same, x) and same is not x
    with pytest.raises(ValueError):
        dsp.resample(x, 0, SR)


def test_to_channels() -> None:
    stereo = np.vstack([sine(100, 0.01), sine(200, 0.01)])
    assert dsp.to_channels(stereo, 1)[0] == pytest.approx((stereo[0] + stereo[1]) / 2)
    mono = dsp.to_channels(stereo[:1], 2)
    assert mono.shape == (2, stereo.shape[1]) and np.array_equal(mono[0], mono[1])
    assert dsp.to_channels(stereo, 2).shape == stereo.shape
    assert dsp.to_channels(stereo, 3).shape[0] == 3
    with pytest.raises(ValueError):
        dsp.to_channels(stereo, 0)


def test_trim_silence_keeps_pad() -> None:
    lead, body, tail = np.zeros((1, SR // 2), np.float32), sine(440, 0.2, amp=0.5), np.zeros((1, SR), np.float32)
    x = np.concatenate([lead, body, tail], axis=1)
    y = dsp.trim_silence(x, SR, threshold_db=-60, pad_ms=5)
    pad = round(0.005 * SR)
    assert y.shape[1] == pytest.approx(body.shape[1] + 2 * pad, abs=3)
    assert dsp.trim_silence(x, SR, head=False).shape[1] > y.shape[1] + SR // 4
    assert dsp.trim_silence(np.zeros((1, 100), np.float32), SR).shape == (1, 0)


def test_fade_is_equal_power_and_clamped() -> None:
    ones = np.ones((1, SR), np.float32)
    y = dsp.fade(ones, SR, 100, 200)
    n_in, n_out = round(0.1 * SR), round(0.2 * SR)
    assert y[0, 0] < 0.01 and y[0, n_in - 1] > 0.99 and y[0, -1] < 0.01
    assert y[0, n_in + 10] == 1.0
    # equal power: sin^2 + cos^2 over the same positions
    assert (y[0, :n_in] ** 2 + y[0, :n_in][::-1] ** 2) == pytest.approx(np.ones(n_in), abs=1e-4)
    assert y[0, SR - n_out - 1] == 1.0
    short = dsp.fade(np.ones((1, 10), np.float32), SR, 5000, 5000)
    assert short.shape == (1, 10) and np.all(np.isfinite(short))
    with pytest.raises(ValueError):
        dsp.fade(ones, SR, -1, 0)
    with pytest.raises(ValueError):
        dsp.fade(ones, SR, float("nan"), 0)


def test_normalize_peak() -> None:
    y = dsp.normalize_peak(sine(440, 0.1, amp=0.1), -3.0)
    assert dsp.peak_dbfs(y) == pytest.approx(-3.0, abs=0.01)
    silent = np.zeros((1, 10), np.float32)
    assert np.array_equal(dsp.normalize_peak(silent, -3.0), silent)


def test_normalize_lufs_hits_target_and_respects_ceiling() -> None:
    quiet = sine(997, 3, amp=0.05)
    y = dsp.normalize_lufs(quiet, SR, -16.0, ceiling_dbfs=-1.0)
    assert dsp.lufs_integrated(y, SR) == pytest.approx(-16.0, abs=0.05)
    # asking for a very loud target is limited by the true-peak ceiling instead
    loud = dsp.normalize_lufs(quiet, SR, -1.0, ceiling_dbfs=-3.0)
    assert dsp.true_peak_dbfs(loud) == pytest.approx(-3.0, abs=0.05)
    assert dsp.lufs_integrated(loud, SR) < -1.0


def test_loop_crossfade_removes_the_click() -> None:
    x = sine(441.5, 1.0, amp=0.8)  # non-integer number of cycles in 1 s -> a jump at the loop point
    ms = 50
    y = dsp.loop_crossfade(x, SR, ms)
    assert y.shape[1] == x.shape[1] - round(ms / 1000 * SR)
    slope = float(np.max(np.abs(np.diff(x[0]))))
    assert abs(float(y[0, 0] - y[0, -1])) <= 1.5 * slope
    assert np.array_equal(dsp.loop_crossfade(x, SR, 0), x)


def test_dc_offset_and_clipping() -> None:
    x = sine(440, 1.0, amp=0.1) + 0.2
    assert dsp.dc_offset(x) == pytest.approx(0.2, abs=1e-3)
    assert dsp.dc_offset(np.vstack([x - 0.2, -(x - 0.2) - 0.05])) == pytest.approx(-0.05, abs=1e-3)
    clipped = np.clip(sine(100, 1.0, amp=2.0), -1, 1)
    count = dsp.clipped_samples(clipped)
    assert count > 0 and dsp.clipped_samples(sine(100, 1.0, amp=0.5)) == 0
    assert dsp.clipped_samples(clipped, threshold=1.0) <= count


def test_as_audio_sanitises() -> None:
    a = dsp.as_audio(np.array([1.0, np.nan, np.inf], dtype=np.float64))
    assert a.dtype == np.float32 and a.shape == (1, 3) and a[0, 1] == 0.0 and a[0, 2] == 0.0
    with pytest.raises(ValueError):
        dsp.as_audio(np.zeros((2, 2, 2)))
