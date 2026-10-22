from __future__ import annotations

import numpy as np
import pytest

from ak_signals import SR, midi_to_hz, sine, tone_note
from audiokit import pitch


@pytest.mark.parametrize("freq", [110.0, 440.0, 880.0])
def test_yin_tracks_sines_within_10_cents(freq: float) -> None:
    frames = pitch.track(sine(freq, 1.0, amp=0.5), SR)
    voiced = [f0 for _, f0, _ in frames if f0 is not None]
    assert len(voiced) >= 0.95 * len(frames)
    cents = 1200 * np.log2(np.median(voiced) / freq)
    assert abs(cents) < 10
    assert max(abs(1200 * np.log2(f / freq)) for f in voiced) < 25
    assert np.median([c for _, f0, c in frames if f0 is not None]) > 0.95


def test_track_times_hops_and_silence() -> None:
    frames = pitch.track(np.zeros((1, SR // 2), np.float32), SR)
    assert all(f0 is None for _, f0, _ in frames)
    assert frames[1][0] - frames[0][0] == pytest.approx(256 / SR)
    assert pitch.track(np.zeros((1, 0), np.float32), SR) == []


def test_track_rejects_out_of_range_and_noise() -> None:
    assert all(f is None for _, f, _ in pitch.track(sine(3000, 0.3, amp=0.5), SR)[3:-3])
    noise = np.random.default_rng(2).standard_normal((1, SR // 2)).astype(np.float32) * 0.3
    voiced = [f for _, f, _ in pitch.track(noise, SR) if f is not None]
    assert len(voiced) < 0.2 * len(pitch.track(noise, SR))


def _melody(midi_notes: list[int], gap_s: float) -> np.ndarray:
    parts: list[np.ndarray] = []
    for m in midi_notes:
        parts.append(tone_note(midi_to_hz(m), 0.4))
        if gap_s:
            parts.append(np.zeros(int(gap_s * SR)))
    return np.concatenate(parts).astype(np.float32)[None, :]


@pytest.mark.parametrize("gap", [0.05, 0.0])
def test_notes_from_a_four_note_melody(gap: float) -> None:
    pitches = [60, 64, 67, 72]
    found = pitch.notes(pitch.track(_melody(pitches, gap), SR))
    assert [n.pitch for n in found] == pitches
    for i, n in enumerate(found):
        assert n.start_s == pytest.approx(i * (0.4 + gap), abs=0.06)
        assert 0.3 < n.end_s - n.start_s < 0.5
        assert 1 <= n.velocity <= 127 and n.confidence > 0.8


def test_notes_velocity_follows_level_when_audio_given() -> None:
    audio = np.concatenate([_melody([60], 0.05), 0.1 * _melody([64], 0.05)], axis=1)
    found = pitch.notes(pitch.track(audio, SR), audio=audio, sr=SR)
    assert [n.pitch for n in found] == [60, 64] and found[0].velocity > found[1].velocity


def test_notes_merge_vibrato_and_drop_short_blips() -> None:
    t = np.arange(SR) / SR
    vibrato = np.sin(2 * np.pi * np.cumsum(midi_to_hz(69) * 2 ** (0.5 * np.sin(2 * np.pi * 5 * t) / 12) / SR))
    found = pitch.notes(pitch.track((0.5 * vibrato).astype(np.float32)[None, :], SR))
    assert [n.pitch for n in found] == [69]
    blip = np.concatenate([_melody([60], 0.0), 0.5 * sine(midi_to_hz(72), 0.03), _melody([60], 0.0)], axis=1)
    assert {n.pitch for n in pitch.notes(pitch.track(blip, SR))} == {60}  # the 30 ms blip is not a note
    assert pitch.notes([]) == [] and pitch.notes([(0.0, None, 0.0)]) == []
