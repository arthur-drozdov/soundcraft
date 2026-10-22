from __future__ import annotations

import json

import numpy as np
import pytest

from ak_signals import SR, beat_times, chord_progression, click_track, midi_to_hz, sine
from audiokit import analysis


@pytest.mark.parametrize("tempo,tol", [(120, 1.0), (87, 1.5), (100, 1.5), (140, 1.5)])
def test_bpm_of_click_tracks(tempo: int, tol: float) -> None:
    value, confidence = analysis.bpm(click_track(tempo, 14), SR)
    assert value == pytest.approx(tempo, abs=tol)
    assert 0.3 < confidence <= 1.0


def test_bpm_octave_correction_and_no_rhythm() -> None:
    assert analysis.bpm(click_track(65, 16), SR)[0] == pytest.approx(130, abs=2.0)
    assert analysis.bpm(np.zeros((1, 5 * SR), np.float32), SR) == (0.0, 0.0)
    assert analysis.bpm(sine(440, 0.05), SR) == (0.0, 0.0)
    with pytest.raises(ValueError):
        analysis.bpm(sine(440, 1), SR, lo=100, hi=50)


def test_onsets_find_the_clicks() -> None:
    expected = beat_times(120, 12)
    found = analysis.onsets(click_track(120, 12), SR)
    assert abs(len(found) - len(expected)) <= 1
    for t in expected:
        assert min(abs(f - t) for f in found) < 0.02
    assert analysis.onsets(np.zeros((1, SR), np.float32), SR) == []


def _triad(root: int, third: int, fifth: int) -> list[float]:
    return [midi_to_hz(root), midi_to_hz(root + third), midi_to_hz(root + fifth)]


def test_key_of_a_minor_progression() -> None:
    am, dm, e = _triad(57, 3, 7), _triad(62, 3, 7), _triad(64, 4, 7)
    name, confidence = analysis.key(chord_progression([am, dm, e, am, am, dm, e, am]), SR)
    assert name == "A minor" and confidence > 0.5


def test_key_of_c_major_and_silence() -> None:
    c, f, g = _triad(60, 4, 7), _triad(65, 4, 7), _triad(67, 4, 7)
    assert analysis.key(chord_progression([c, f, g, c, c, f, g, c]), SR)[0] == "C major"
    assert analysis.key(np.zeros((1, SR), np.float32), SR) == ("", 0.0)


def test_spectral_shape() -> None:
    assert analysis.centroid_hz(sine(1000, 1), SR) == pytest.approx(1000, abs=80)
    assert analysis.centroid_hz(sine(4000, 1), SR) > analysis.centroid_hz(sine(500, 1), SR)
    noise = np.random.default_rng(0).standard_normal((1, SR)).astype(np.float32) * 0.1
    assert analysis.bandwidth_hz(noise, SR) > 3000
    assert analysis.bandwidth_hz(sine(1000, 1), SR) < analysis.bandwidth_hz(noise, SR) / 3
    assert analysis.centroid_hz(np.zeros((1, 100), np.float32), SR) == 0.0


def test_envelope_and_decay_time() -> None:
    t = np.arange(2 * SR) / SR
    decaying = (np.sin(2 * np.pi * 300 * t) * 10 ** (-60.0 * t / 20.0)).astype(np.float32)[None, :]
    assert analysis.decay_time_s(decaying, SR, -40) == pytest.approx(40 / 60, abs=0.03)
    env = analysis.envelope(sine(1000, 1, amp=0.5), SR, ms=10)
    assert env.dtype == np.float32 and len(env) == 100
    assert env[10] == pytest.approx(0.5 / np.sqrt(2), abs=0.01)
    assert analysis.decay_time_s(np.zeros((1, 100), np.float32), SR) == 0.0
    assert analysis.envelope(np.zeros((1, 0), np.float32), SR).size == 0


def test_loop_seam_score_separates_good_and_bad_loops() -> None:
    seamless = sine(441, 1.0, amp=0.8)  # exactly 441 cycles
    cut = sine(441.5, 1.0, amp=0.8)  # kink at the loop point
    loud_to_quiet = np.concatenate([sine(441, 0.5, amp=0.8), sine(441, 0.5, amp=0.01)], axis=1)
    good = analysis.loop_seam_score(seamless, SR)
    assert good > 0.9
    assert analysis.loop_seam_score(cut, SR) < good - 0.1
    assert analysis.loop_seam_score(loud_to_quiet, SR) < 0.8
    assert 0.0 <= analysis.loop_seam_score(sine(440, 0.001), SR) <= 1.0


def test_summary_is_complete_and_json_safe() -> None:
    audio = click_track(120, 8)
    s = analysis.summary(audio, SR)
    for key in ("duration_s", "sample_rate", "channels", "peak_dbfs", "rms_dbfs", "lufs", "onsets_s", "bpm",
                "bpm_confidence", "key", "key_confidence", "centroid_hz", "clipped_samples", "dc_offset",
                "true_peak_dbfs", "loop_seam"):
        assert key in s
    assert s["duration_s"] == pytest.approx(8.0, abs=0.01) and s["channels"] == 1 and s["sample_rate"] == SR
    assert s["bpm"] == pytest.approx(120, abs=1.0)
    json.dumps(s, allow_nan=False)
    silent = analysis.summary(np.zeros((2, SR), np.float32), SR)
    json.dumps(silent, allow_nan=False)
    assert silent["onsets_s"] == [] and silent["bpm"] == 0.0 and silent["lufs"] <= -100
