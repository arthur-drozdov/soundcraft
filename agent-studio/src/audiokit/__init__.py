"""audiokit: pure numpy/scipy audio utilities (WAV I/O, DSP, analysis, pitch, MIDI, instruments)."""

from . import analysis, dsp, instruments, midi, pitch, wav
from .analysis import bpm, centroid_hz, key, onsets, summary
from .dsp import (
    fade,
    loop_crossfade,
    lufs_integrated,
    normalize_lufs,
    normalize_peak,
    peak_dbfs,
    resample,
    rms_dbfs,
    to_channels,
    trim_silence,
    true_peak_dbfs,
)
from .instruments import plan_keybed, write_dspreset, write_sfz
from .midi import read_smf, write_smf
from .pitch import Note, notes, track
from .wav import read, write

__all__ = [
    "analysis", "dsp", "instruments", "midi", "pitch", "wav",
    "bpm", "centroid_hz", "key", "onsets", "summary",
    "fade", "loop_crossfade", "lufs_integrated", "normalize_lufs", "normalize_peak", "peak_dbfs",
    "resample", "rms_dbfs", "to_channels", "trim_silence", "true_peak_dbfs",
    "plan_keybed", "write_dspreset", "write_sfz", "read_smf", "write_smf", "Note", "notes", "track",
    "read", "write",
]
