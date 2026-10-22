"""The profiler's synthesised test signals: shape, levels, determinism, layout and input checks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from atlas import testsignals as ts
from audiokit import analysis, dsp, wav

SR = 48000


@pytest.fixture(scope="module")
def signals(tmp_path_factory: pytest.TempPathFactory) -> ts.EffectSignals:
    return ts.build_effect_signals(tmp_path_factory.mktemp("signals"), SR)


def _octave_energy(x: np.ndarray, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x[0] * np.hanning(x.shape[1]))) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], 1 / SR)
    return float(spec[(freqs >= lo) & (freqs < hi)].sum())


def test_pink_noise_has_equal_energy_per_octave_and_the_requested_level() -> None:
    x = ts.pink_noise(4.0, SR, level_dbfs=-18.0)
    assert x.shape == (2, 4 * SR) and x.dtype == np.float32
    assert dsp.rms_dbfs(x) == pytest.approx(-18.0, abs=0.1)
    low, high = _octave_energy(x, 250, 500), _octave_energy(x, 4000, 8000)
    assert abs(10 * math.log10(low / high)) < 1.5


def test_pink_noise_is_deterministic_per_seed_and_mono_on_request() -> None:
    a, b = ts.pink_noise(1.0, SR, seed=5), ts.pink_noise(1.0, SR, seed=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, ts.pink_noise(1.0, SR, seed=6))
    assert ts.pink_noise(0.5, SR, channels=1).shape[0] == 1
    # stereo noise is mostly but not exactly dual mono
    corr = float(np.corrcoef(a[0], a[1])[0, 1])
    assert 0.9 < corr < 0.999


def test_sweep_rises_logarithmically_from_20_hz_towards_20_khz() -> None:
    x = ts.log_sweep(2.0, SR, level_dbfs=-12.0)
    assert dsp.peak_dbfs(x) == pytest.approx(-12.0, abs=0.1)

    def dominant(start: float, length: float = 0.05) -> float:
        seg = x[0, int(start * SR) : int((start + length) * SR)] * np.hanning(int(length * SR))
        spec = np.abs(np.fft.rfft(seg, 1 << 16))
        return float(np.argmax(spec) * SR / (1 << 16))

    early, middle, late = dominant(0.05), dominant(1.0), dominant(1.9)
    assert early < 100 < middle < 3000 < late
    # a log sweep from 20 Hz to 20 kHz over 2 s is at the geometric mean of the ends after 1 s
    assert middle == pytest.approx(math.sqrt(20 * 20000), rel=0.2)


def test_drum_pattern_has_transients_a_decaying_kick_and_the_requested_peak() -> None:
    x = ts.drum_pattern(2.5, SR, peak_dbfs=-6.0)
    assert dsp.peak_dbfs(x) == pytest.approx(-6.0, abs=0.1)
    onset_times = analysis.onsets(x, SR)
    assert len(onset_times) >= 4
    assert onset_times[0] < 0.05  # the kick on beat one
    assert analysis.decay_time_s(x[:, : int(0.3 * SR)], SR) > 0.05


def test_effect_signals_are_written_as_wav_files_with_a_consistent_layout(signals: ts.EffectSignals) -> None:
    assert set(signals.parts) == {"pink_noise", "sine_sweep", "clicks", "composite"}
    for name, path in signals.parts.items():
        assert path.suffix == ".wav" and path.is_file(), name
    audio, sr = wav.read(signals.path)
    assert sr == SR and audio.shape[0] == 2
    assert audio.shape[1] == pytest.approx(signals.file_duration_s * SR, abs=2)
    assert signals.bounce_end_s > signals.file_duration_s
    assert set(signals.segments) == {"noise", "sweep", "clicks"}
    last_end = 0.0
    for name in ("noise", "sweep", "clicks"):
        lo, hi = signals.segments[name]
        assert 0 <= lo < hi <= signals.bounce_end_s
        assert lo >= last_end - 1.0  # windows follow one another (tails may reach into the gaps)
        last_end = hi
    # the windows contain the signals: audible inside, silent in the gaps between
    for name, (lo, hi) in signals.segments.items():
        assert dsp.rms_dbfs(audio[:, int(lo * SR) : int(hi * SR)]) > -40, name
    gap = audio[:, int(4.9 * SR) : int(4.99 * SR)]
    assert dsp.peak_dbfs(gap) < -100
    assert dsp.peak_dbfs(audio) < -3.0  # headroom for gain-type parameters


def test_signal_files_are_identical_between_runs(tmp_path: Path) -> None:
    a = ts.build_effect_signals(tmp_path / "a", SR, 4.0)
    b = ts.build_effect_signals(tmp_path / "b", SR, 4.0)
    assert a.path.read_bytes() == b.path.read_bytes()


def test_a_shorter_cap_compresses_the_layout_and_never_exceeds_it(tmp_path: Path) -> None:
    short = ts.build_effect_signals(tmp_path, SR, 4.5)
    assert short.bounce_end_s == pytest.approx(4.5)
    assert short.file_duration_s < 4.5
    assert all(hi <= short.bounce_end_s for _, hi in short.segments.values())
    longer = ts.build_effect_signals(tmp_path / "long", SR, 30.0)
    assert longer.bounce_end_s == pytest.approx(9.0)  # the layout never stretches


def test_sweep_band_windows_cover_the_sweep_in_order(signals: ts.EffectSignals) -> None:
    bands = ts.sweep_band_windows(signals)
    assert len(bands) == 5
    assert bands[0][0] == pytest.approx(20.0) and bands[-1][1] == pytest.approx(20000.0)
    for (lo, hi, t0, t1), nxt in zip(bands, [*bands[1:], None], strict=True):
        assert lo < hi and t0 < t1
        if nxt is not None:
            assert hi == nxt[0] and t1 == pytest.approx(nxt[2])
    assert bands[0][2] == pytest.approx(signals.sweep_start_s)
    assert bands[-1][3] == pytest.approx(signals.sweep_start_s + signals.sweep_length_s)


def test_instrument_plan_is_a_sustained_c_major_chord_then_a_staccato_note() -> None:
    plan = ts.build_instrument_plan(120.0)
    chord = [n for n in plan.notes if n.start_s == 0.0]
    assert sorted(n.pitch for n in chord) == [60, 64, 67]
    assert all(n.length_s >= 1.5 for n in chord)
    staccato = [n for n in plan.notes if n.start_s > 0]
    assert len(staccato) == 1 and staccato[0].length_s < 0.25 and staccato[0].start_s > max(n.length_s for n in chord)
    assert set(plan.segments) == {"chord", "staccato"}
    assert plan.bounce_end_s >= staccato[0].start_s + staccato[0].length_s
    calls = plan.note_calls(clip=9)
    assert {c["clip"] for c in calls} == {9}
    # 120 bpm = 2 quarters per second = 1920 ticks per second at 960 ticks per quarter
    assert calls[0]["length_ticks"] == round(2.0 * 1920)
    assert calls[-1]["start_ticks"] == round(3.5 * 1920)


def test_instrument_plan_follows_the_session_tempo_and_the_cap() -> None:
    slow = ts.build_instrument_plan(60.0)
    assert slow.note_calls(1)[0]["length_ticks"] == round(2.0 * 960)
    short = ts.build_instrument_plan(120.0, duration_s=3.0)
    assert short.bounce_end_s <= 3.0
    assert max(n.start_s + n.length_s for n in short.notes) <= short.bounce_end_s


@pytest.mark.parametrize(
    "call",
    [
        lambda: ts.pink_noise(1.0, 100),
        lambda: ts.pink_noise(1.0, True),
        lambda: ts.pink_noise(float("nan"), SR),
        lambda: ts.pink_noise(10_000.0, SR),
        lambda: ts.pink_noise(1.0, SR, channels=3),
        lambda: ts.log_sweep(1.0, SR, f_lo=0.0),
        lambda: ts.log_sweep(1.0, SR, f_lo=100.0, f_hi=50.0),
        lambda: ts.drum_pattern(1.0, SR, bpm=0.0),
        lambda: ts.seconds_to_ticks(-1.0),
        lambda: ts.seconds_to_ticks(1.0, bpm=float("inf")),
        lambda: ts.build_instrument_plan(bpm=1000.0),
    ],
)
def test_hostile_arguments_are_rejected(call) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ts.SignalError):
        call()


def test_build_rejects_out_of_range_caps_and_sample_rates(tmp_path: Path) -> None:
    with pytest.raises(ts.SignalError):
        ts.build_effect_signals(tmp_path, SR, 1.0)
    with pytest.raises(ts.SignalError):
        ts.build_effect_signals(tmp_path, SR, 1000.0)
    with pytest.raises(ts.SignalError):
        ts.build_effect_signals(tmp_path, 1, 9.0)
    assert not list(tmp_path.glob("*.wav"))
