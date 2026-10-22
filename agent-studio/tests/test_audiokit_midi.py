from __future__ import annotations

import struct
from pathlib import Path

import pytest

from audiokit import midi
from audiokit.pitch import Note


def test_smf_round_trip(tmp_path: Path) -> None:
    notes = [Note(60, 0.0, 0.5, 90), Note(64, 0.5, 1.0, 70), Note(67, 0.25, 1.5, 100), Note(72, 2.0, 2.125, 1)]
    path = tmp_path / "a.mid"
    midi.write_smf(path, notes, bpm=120, ppq=480)
    back = midi.read_smf(path)
    assert [n.pitch for n in back] == [60, 67, 64, 72]  # ordered by start time
    by_pitch = {n.pitch: n for n in back}
    for original in notes:
        got = by_pitch[original.pitch]
        assert got.start_s == pytest.approx(original.start_s, abs=0.002)
        assert got.end_s == pytest.approx(original.end_s, abs=0.002)
        assert got.velocity == original.velocity


def test_header_and_tempo_bytes(tmp_path: Path) -> None:
    path = tmp_path / "h.mid"
    midi.write_smf(path, [Note(60, 0, 1)], bpm=90, ppq=96)
    raw = path.read_bytes()
    assert raw[:4] == b"MThd" and struct.unpack(">IHHH", raw[4:14]) == (6, 0, 1, 96)
    assert struct.pack(">I", round(60_000_000 / 90))[1:] in raw
    assert raw.count(b"MTrk") == 1


def test_tempo_changes_the_tick_scale(tmp_path: Path) -> None:
    path = tmp_path / "t.mid"
    midi.write_smf(path, [Note(60, 1.0, 2.0)], bpm=60, ppq=480)
    ticks = [n for n in midi.read_smf(path)]
    assert ticks[0].start_s == pytest.approx(1.0, abs=0.003) and ticks[0].end_s == pytest.approx(2.0, abs=0.003)


def test_accepts_dicts_and_clamps_same_pitch_overlap(tmp_path: Path) -> None:
    path = tmp_path / "d.mid"
    midi.write_smf(path, [{"pitch": 60, "start_s": 0, "end_s": 1, "velocity": 80}, {"pitch": 60, "start_s": 0.5, "end_s": 1.5}])
    back = midi.read_smf(path)
    assert len(back) == 2
    assert back[0].end_s == pytest.approx(0.5, abs=0.003) and back[1].start_s == pytest.approx(0.5, abs=0.003)


def test_rejects_bad_notes_and_params(tmp_path: Path) -> None:
    p = tmp_path / "bad.mid"
    for bad in (Note(128, 0, 1), Note(60, -1, 1), Note(60, 1, 1), Note(60, 0, float("inf")), Note(60, 0, 1, 0)):
        with pytest.raises(midi.MidiError):
            midi.write_smf(p, [bad])
    with pytest.raises(midi.MidiError):
        midi.write_smf(p, [], bpm=0)
    with pytest.raises(midi.MidiError):
        midi.write_smf(p, [], ppq=0)
    with pytest.raises(midi.MidiError):
        midi.write_smf(p, [{"pitch": "x", "start_s": 0, "end_s": 1}])
    assert not p.exists()


def test_empty_file_is_valid(tmp_path: Path) -> None:
    p = tmp_path / "e.mid"
    midi.write_smf(p, [])
    assert midi.read_smf(p) == []


def test_read_hostile_files(tmp_path: Path) -> None:
    for name, blob in {
        "junk.mid": b"hello world, definitely not midi",
        "smpte.mid": b"MThd" + struct.pack(">IHHH", 6, 0, 1, 0xE728),
        "trunc.mid": b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480) + b"MTrk" + struct.pack(">I", 50) + b"\x00\x90\x3c",
    }.items():
        p = tmp_path / name
        p.write_bytes(blob)
        with pytest.raises(midi.MidiError):
            midi.read_smf(p)


def test_read_running_status_and_note_on_zero(tmp_path: Path) -> None:
    track = b"\x00\x90\x3c\x50" + b"\x60\x3c\x00" + b"\x00\xff\x2f\x00"  # note on, then velocity-0 via running status
    p = tmp_path / "rs.mid"
    p.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, 96) + b"MTrk" + struct.pack(">I", len(track)) + track)
    notes = midi.read_smf(p)
    assert len(notes) == 1 and notes[0].pitch == 60 and notes[0].velocity == 0x50
    assert notes[0].end_s == pytest.approx(0.5)  # 96 ticks at 96 ppq, default 120 bpm
