"""The ``basicpitch`` family: Spotify Basic Pitch audio-to-MIDI (CPU).

Run as ``python -m forge.workers.basicpitch`` in an environment with ``pip install basic-pitch``.

``hello`` options: none (``deploy/forge.example.yaml`` gives ``{}``). Optional extras: ``onset_threshold``
(0.5), ``frame_threshold`` (0.3), ``melodia_trick`` (true).

Network: none; the model ships inside the ``basic_pitch`` package.

Upstream calls (source: basic-pitch @ fa5997a, ``basic_pitch/inference.py``; none run here)::

    basic_pitch.inference.Model(basic_pitch.ICASSP_2022_MODEL_PATH)          # preload once in load()
    basic_pitch.inference.predict(audio_path, model_or_model_path, onset_threshold=, frame_threshold=,
        minimum_note_length=<ms>, midi_tempo=<bpm>)
        -> (model_output, pretty_midi.PrettyMIDI, [(start_s, end_s, pitch, amplitude 0..1, pitch_bends | None)])

``basic_pitch`` raises ``NameError`` on import when none of TensorFlow, CoreML, TFLite or ONNX runtime is
installed; that is reported as the model being unavailable.

Output (same shape as the ``yin`` family): ``melody.mid`` written with ``audiokit.midi.write_smf`` at the
job's ``bpm``, and ``extra.notes = [{pitch, start_s, end_s, velocity, confidence}]``. Velocity is
``round(127 * amplitude)`` clamped to 1..127; pitch bends are not written (audiokit's MIDI writer has none).
Basic Pitch is polyphonic, so notes may overlap; ``write_smf`` shortens same-pitch overlaps.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from audiokit import midi, pitch, wav

from . import _upstream as up
from .protocol import ModelInfo, Output, RunJob
from .runtime import CancelCheck, Family, Progress, check_cancel, main

log = logging.getLogger("forge.worker.basicpitch")

LICENCE = "Apache-2.0 (Spotify Basic Pitch)"
MAX_NOTES = 20_000
MAX_SECONDS = pitch.MAX_DURATION_S


class BasicPitchFamily(Family):
    name = "basicpitch"

    def __init__(self) -> None:
        self._onset = 0.5
        self._frame = 0.3
        self._melodia = True
        self._config_error: str | None = None
        self._model: Any = None
        self._inference: Any = None

    def configure(self, options: dict[str, Any]) -> None:
        self._config_error = None
        try:
            self._onset = up.number(options, "onset_threshold", 0.5, 0.0, 1.0)
            self._frame = up.number(options, "frame_threshold", 0.3, 0.0, 1.0)
            self._melodia = bool(options.get("melodia_trick", True))
        except ValueError as exc:
            self._config_error = f"invalid options: {exc}"

    def models(self) -> list[ModelInfo]:
        reason = self._config_error or up.probe_module("basic_pitch")
        if reason is None:
            try:  # importing it is the only way to learn that no inference runtime is installed
                up.import_module("basic_pitch.inference")
            except up.MissingDependency as exc:
                reason = str(exc)
        info: ModelInfo = {
            "id": "basic-pitch", "family": self.name, "job_types": ["transcribe.melody"], "available": reason is None,
            "licence": LICENCE, "checkpoint": "builtin:basic_pitch.ICASSP_2022", "checkpoint_hash": None,
            "vram_gb": 0.0, "max_seconds": MAX_SECONDS,
        }
        if reason:
            info["reason"] = reason if reason.startswith(("invalid", "cannot import")) else f"{reason}; pip install basic-pitch"
        return [info]

    def load(self, model: str) -> None:
        if model != "basic-pitch":
            raise ValueError(f"unknown model {model!r}")
        if self._config_error:
            raise up.MissingDependency(self._config_error)
        inference = up.import_module("basic_pitch.inference")
        package = up.import_module("basic_pitch")
        path = getattr(package, "ICASSP_2022_MODEL_PATH", None)
        loader = getattr(inference, "Model", None)
        self._inference = inference
        self._model = path  # predict() also accepts the path and loads it on every call
        if loader is not None and path is not None:
            try:
                self._model = loader(path)
            except Exception as exc:  # fall back to the path: slower per call, same result
                log.warning("could not preload the Basic Pitch model (%s); it will load on every call", exc)

    def unload(self, model: str) -> None:
        self._model = None
        self._inference = None
        up.free_accelerator_memory()

    def run(self, job: RunJob, progress: Progress, cancelled: CancelCheck) -> list[Output]:
        params, out_dir = up.check_job(job, ["transcribe.melody"])
        if self._inference is None:
            raise RuntimeError("model 'basic-pitch' is not loaded")
        min_note_ms = up.number(params, "min_note_ms", 60.0, 1.0, 5000.0)
        bpm = up.number(params, "bpm", 120.0, 1.0, 1000.0)
        path = up.input_path(job, "audio")
        progress(0.05, "checking audio")
        audio, sr = wav.read(path)  # bounded by audiokit's frame cap; Basic Pitch decodes the file itself
        if audio.shape[1] == 0 or audio.shape[1] / sr > MAX_SECONDS:
            raise ValueError(f"audio must be between 0 and {MAX_SECONDS:g} s long")
        del audio
        check_cancel(cancelled)
        progress(0.15, "running Basic Pitch")
        _, _, events = self._inference.predict(
            str(path), self._model, onset_threshold=self._onset, frame_threshold=self._frame,
            minimum_note_length=min_note_ms, midi_tempo=bpm, melodia_trick=self._melodia,
        )
        check_cancel(cancelled)
        progress(0.8, "writing MIDI")
        notes = notes_from_events(events)
        target = up.confine(out_dir, "melody.mid")
        midi.write_smf(target, notes, bpm=bpm)
        listing = [
            {"pitch": n["pitch"], "start_s": n["start_s"], "end_s": n["end_s"], "velocity": n["velocity"], "confidence": n["confidence"]}
            for n in notes
        ]
        progress(1.0, f"{len(notes)} notes")
        return [{"path": str(target), "kind": "midi", "candidate": 0, "extra": {"notes": listing, "bpm": bpm}}]


def notes_from_events(events: Any) -> list[dict[str, Any]]:
    """Validated note dictionaries from Basic Pitch ``note_events``
    (``(start_s, end_s, pitch, amplitude, pitch_bends)``), sorted by start, at most ``MAX_NOTES``.

    Malformed or non-finite events are dropped rather than failing the job."""
    notes: list[dict[str, Any]] = []
    for event in events or []:
        try:
            start, end, midi_pitch, amplitude = float(event[0]), float(event[1]), int(event[2]), float(event[3])
        except (TypeError, ValueError, IndexError):
            continue
        if not all(math.isfinite(v) for v in (start, end, amplitude)) or start < 0 or end <= start or not 0 <= midi_pitch <= 127:
            continue
        confidence = min(1.0, max(0.0, amplitude))
        notes.append({
            "pitch": midi_pitch, "start_s": round(start, 4), "end_s": round(end, 4),
            "velocity": min(127, max(1, int(round(127 * confidence)))), "confidence": round(confidence, 3),
        })
    notes.sort(key=lambda n: (n["start_s"], n["pitch"]))
    return notes[:MAX_NOTES]


if __name__ == "__main__":
    main(BasicPitchFamily())
