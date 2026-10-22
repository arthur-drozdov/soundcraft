"""The ``yin`` family: monophonic melody transcription with ``audiokit.pitch`` (CPU, always available)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from audiokit import midi, pitch, wav

from .protocol import ModelInfo, Output, RunJob
from .runtime import CancelCheck, Family, Progress, check_cancel, main

MAX_NOTES = 20_000


class YinFamily(Family):
    name = "yin"

    def __init__(self) -> None:
        self._digest = hashlib.sha256(Path(pitch.__file__).read_bytes()).hexdigest()[:16]

    def models(self) -> list[ModelInfo]:
        return [
            {
                "id": "yin", "family": self.name, "job_types": ["transcribe.melody"], "available": True,
                "licence": "MIT OR Apache-2.0 (audiokit)", "checkpoint": "builtin:audiokit.pitch",
                "checkpoint_hash": self._digest, "vram_gb": 0.0, "max_seconds": pitch.MAX_DURATION_S,
            }
        ]

    def run(self, job: RunJob, progress: Progress, cancelled: CancelCheck) -> list[Output]:
        if job["type"] != "transcribe.melody":
            raise ValueError(f"yin does not run {job['type']!r}")
        path = job["inputs"].get("audio")
        if not path or not Path(path).is_file():
            raise ValueError("missing audio input")
        params: dict[str, Any] = job["params"]
        progress(0.05, "reading audio")
        audio, sr = wav.read(path)
        check_cancel(cancelled)
        progress(0.2, "tracking pitch")
        frames = pitch.track(audio, sr)
        check_cancel(cancelled)
        progress(0.8, "segmenting notes")
        notes = pitch.notes(frames, min_note_ms=float(params.get("min_note_ms", 60.0)), audio=audio, sr=sr)[:MAX_NOTES]
        out = Path(job["out_dir"]) / "melody.mid"
        midi.write_smf(out, notes, bpm=float(params.get("bpm", 120.0)))
        listing = [
            {"pitch": n.pitch, "start_s": n.start_s, "end_s": n.end_s, "velocity": n.velocity, "confidence": n.confidence}
            for n in notes
        ]
        progress(1.0, f"{len(notes)} notes")
        return [{"path": str(out), "kind": "midi", "candidate": 0, "extra": {"notes": listing, "bpm": float(params.get("bpm", 120.0))}}]


if __name__ == "__main__":
    main(YinFamily())
