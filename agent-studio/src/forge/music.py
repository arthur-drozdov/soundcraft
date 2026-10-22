"""Tiny music-theory helpers shared by the gateway and the (dependency-free) workers."""

from __future__ import annotations

import math
import re

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLATS = {"DB": 1, "EB": 3, "GB": 6, "AB": 8, "BB": 10, "CB": 11, "FB": 4}
_KEY_RE = re.compile(r"^\s*([A-Ga-g])\s*([#b]?)\s*(major|minor|maj|min|m)?\s*$", re.IGNORECASE)
KEY_PATTERN = r"^\s*[A-Ga-g]\s*[#b]?\s*(?i:major|minor|maj|min|m)?\s*$"


def midi_to_hz(note: float) -> float:
    """Frequency of a (fractional) MIDI note number (A4 = 69 = 440 Hz)."""
    return 440.0 * 2.0 ** ((float(note) - 69.0) / 12.0)


def hz_to_midi(freq: float) -> float:
    if not math.isfinite(freq) or freq <= 0:
        raise ValueError("frequency must be positive and finite")
    return 69.0 + 12.0 * math.log2(freq / 440.0)


def parse_key(text: str) -> tuple[int, str] | None:
    """``"F minor"`` -> ``(5, "minor")``; flats are folded to sharps; no mode means major."""
    match = _KEY_RE.match(text or "")
    if not match:
        return None
    letter, accidental, mode = match.group(1).upper(), match.group(2), (match.group(3) or "major").lower()
    if accidental == "b" and (letter + "B") in _FLATS:
        pc = _FLATS[letter + "B"]
    else:
        pc = NOTE_NAMES.index(letter)
        if accidental == "#":
            pc = (pc + 1) % 12
    return pc, "minor" if mode in ("minor", "min", "m") else "major"


def key_name(pc: int, mode: str) -> str:
    return f"{NOTE_NAMES[pc % 12]} {mode}"
