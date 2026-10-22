"""Minimal Standard MIDI File writer (type 0) and note reader."""

from __future__ import annotations

import math
import os
import struct
from collections.abc import Mapping, Sequence
from typing import Any

from .pitch import Note
from .wav import atomic_write_bytes

MAX_NOTES = 200_000
MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_TRACKS = 64
_DEFAULT_TEMPO_US = 500_000


class MidiError(ValueError):
    """Raised for invalid note data or malformed / unsupported MIDI files."""


def _vlq(value: int) -> bytes:
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def _coerce(note: Any) -> Note:
    """Accept a :class:`Note`, a mapping with the same keys, or any object with those attributes."""
    get = note.get if isinstance(note, Mapping) else lambda k, d=None: getattr(note, k, d)
    try:
        pitch, start, end = int(get("pitch")), float(get("start_s")), float(get("end_s"))
        velocity = int(get("velocity", 80))
    except (TypeError, ValueError) as exc:
        raise MidiError(f"invalid note: {note!r}") from exc
    if not (0 <= pitch <= 127 and 1 <= velocity <= 127):
        raise MidiError(f"note pitch/velocity out of range: {note!r}")
    if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end <= start:
        raise MidiError(f"note times must be finite with 0 <= start < end: {note!r}")
    return Note(pitch, start, end, velocity)


def write_smf(
    path: str | os.PathLike[str], notes: Sequence[Any], bpm: float = 120.0, ppq: int = 480
) -> None:
    """Write ``notes`` (``pitch``, ``start_s``, ``end_s``, ``velocity``; seconds) as a type-0 SMF at ``bpm``.

    Overlapping notes of the same pitch are shortened so each ends where the next begins."""
    if not math.isfinite(bpm) or not 1.0 <= bpm <= 1000.0:
        raise MidiError("bpm must be between 1 and 1000")
    if isinstance(ppq, bool) or not isinstance(ppq, int) or not 1 <= ppq <= 32767:
        raise MidiError("ppq must be an integer in 1..32767")
    if len(notes) > MAX_NOTES:
        raise MidiError(f"too many notes (cap {MAX_NOTES})")
    ticks_per_s = bpm / 60.0 * ppq
    spans: list[tuple[int, int, int, int]] = []  # (pitch, start_tick, end_tick, velocity)
    for raw in sorted((_coerce(n) for n in notes), key=lambda n: (n.start_s, n.pitch)):
        start = int(round(raw.start_s * ticks_per_s))
        end = max(start + 1, int(round(raw.end_s * ticks_per_s)))
        spans.append((raw.pitch, start, end, raw.velocity))
    last_start: dict[int, int] = {}
    for i in range(len(spans) - 1, -1, -1):
        pitch, start, end, vel = spans[i]
        if pitch in last_start and last_start[pitch] < end:
            end = max(start + 1, last_start[pitch])
        spans[i] = (pitch, start, end, vel)
        last_start[pitch] = start
    events: list[tuple[int, int, bytes]] = []  # (tick, order: off before on, message)
    for pitch, start, end, vel in spans:
        events.append((start, 1, bytes([0x90, pitch, vel])))
        events.append((end, 0, bytes([0x80, pitch, 0])))
    events.sort(key=lambda e: (e[0], e[1]))
    track = bytearray()
    track += b"\x00\xff\x51\x03" + int(round(60_000_000 / bpm)).to_bytes(3, "big")
    track += b"\x00\xff\x58\x04\x04\x02\x18\x08"
    tick = 0
    for at, _, message in events:
        track += _vlq(at - tick) + message
        tick = at
    track += b"\x00\xff\x2f\x00"
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq)
    atomic_write_bytes(path, header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track))


def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if pos >= len(data):
            raise MidiError("truncated variable-length quantity")
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos
    raise MidiError("variable-length quantity too long")


def _parse_track(data: bytes) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int]], int]:
    """Return ``(note_events, tempo_changes, end_tick)``; note events are ``(tick, channel, pitch, velocity)``
    with velocity 0 meaning note-off."""
    pos, tick, status = 0, 0, 0
    notes_out: list[tuple[int, int, int, int]] = []
    tempos: list[tuple[int, int]] = []
    while pos < len(data):
        delta, pos = _read_vlq(data, pos)
        tick += delta
        if pos >= len(data):
            raise MidiError("truncated track")
        byte = data[pos]
        if byte >= 0x80:
            status = byte
            pos += 1
        elif status == 0:
            raise MidiError("data byte without running status")
        if status == 0xFF:
            if pos >= len(data):
                raise MidiError("truncated meta event")
            meta = data[pos]
            length, pos = _read_vlq(data, pos + 1)
            body = data[pos : pos + length]
            if len(body) != length:
                raise MidiError("truncated meta event")
            pos += length
            if meta == 0x51 and length == 3:
                tempos.append((tick, int.from_bytes(body, "big")))
            elif meta == 0x2F:
                break
        elif status in (0xF0, 0xF7):
            length, pos = _read_vlq(data, pos)
            pos += length
        else:
            kind, channel = status & 0xF0, status & 0x0F
            size = 1 if kind in (0xC0, 0xD0) else 2
            args = data[pos : pos + size]
            if len(args) != size:
                raise MidiError("truncated channel event")
            pos += size
            if kind == 0x90 and args[1] > 0:
                notes_out.append((tick, channel, args[0], args[1]))
            elif kind == 0x80 or kind == 0x90:
                notes_out.append((tick, channel, args[0], 0))
    return notes_out, tempos, tick


def read_smf(path: str | os.PathLike[str]) -> list[Note]:
    """Read the notes of a type-0 or type-1 SMF as :class:`Note` objects in seconds (tempo changes honoured)."""
    with open(path, "rb") as handle:
        data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise MidiError("MIDI file too large")
    if len(data) < 14 or data[:4] != b"MThd":
        raise MidiError("not a MIDI file")
    length, fmt, ntracks, division = struct.unpack(">IHHH", data[4:14])
    if length < 6 or fmt > 1 or ntracks > _MAX_TRACKS:
        raise MidiError("unsupported MIDI file layout")
    if division & 0x8000 or division == 0:
        raise MidiError("SMPTE time division is not supported")
    pos = 8 + length
    note_events: list[tuple[int, int, int, int]] = []
    tempos: list[tuple[int, int]] = []
    end_tick = 0
    for _ in range(ntracks):
        if pos + 8 > len(data):
            break
        chunk_id, size = struct.unpack(">4sI", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + size]
        pos += 8 + size
        if chunk_id != b"MTrk":
            continue
        events, changes, last = _parse_track(body)
        note_events += events
        tempos += changes
        end_tick = max(end_tick, last)
    tempo_map = sorted(tempos) or [(0, _DEFAULT_TEMPO_US)]
    if tempo_map[0][0] != 0:
        tempo_map.insert(0, (0, _DEFAULT_TEMPO_US))
    marks: list[tuple[int, float, int]] = []  # (tick, seconds at tick, tempo)
    seconds = 0.0
    for i, (tick, tempo) in enumerate(tempo_map):
        if i:
            prev_tick, prev_seconds, prev_tempo = marks[-1]
            seconds = prev_seconds + (tick - prev_tick) * prev_tempo / 1e6 / division
        marks.append((tick, seconds, tempo))

    def to_seconds(tick: int) -> float:
        base_tick, base_s, tempo = max((m for m in marks if m[0] <= tick), key=lambda m: m[0])
        return base_s + (tick - base_tick) * tempo / 1e6 / division

    open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    result: list[Note] = []
    for tick, channel, pitch, velocity in sorted(note_events, key=lambda e: (e[0], e[3] > 0)):
        key = (channel, pitch)
        if velocity > 0:
            open_notes.setdefault(key, []).append((tick, velocity))
        elif open_notes.get(key):
            start, vel = open_notes[key].pop(0)
            result.append(Note(pitch, to_seconds(start), to_seconds(tick), vel, 1.0))
    for (_, pitch), pending in open_notes.items():
        for start, vel in pending:
            result.append(Note(pitch, to_seconds(start), to_seconds(max(end_tick, start + 1)), vel, 1.0))
    return sorted(result, key=lambda n: (n.start_s, n.pitch))
