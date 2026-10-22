"""The ``basicpitch`` worker against a fake ``basic_pitch`` package."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from audiokit import midi
from forge.workers import basicpitch as bp
from forge.workers import yin
from forge.workers.protocol import validate_model_info, validate_outputs
from forge.workers.runtime import Cancelled

from fworkers_fakes import FakeBasicPitch, Harness, block, inside, install_basic_pitch, make_job, make_wav, put, module, run_job


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[bp.BasicPitchFamily, FakeBasicPitch, Path]:
    state = install_basic_pitch(monkeypatch)
    family = bp.BasicPitchFamily()
    family.configure({})
    family.load("basic-pitch")
    return family, state, tmp_path


def audio_input(tmp: Path, seconds: float = 1.0) -> dict[str, str]:
    return {"audio": str(make_wav(tmp / "in" / "a.wav", seconds=seconds, sr=22050, channels=1))}


# ---- availability ---------------------------------------------------------------------------------

def test_unavailable_without_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    block(monkeypatch, "basic_pitch", "basic_pitch.inference")
    family = bp.BasicPitchFamily()
    family.configure({})
    (info,) = family.models()
    assert info["id"] == "basic-pitch" and not info["available"] and "pip install basic-pitch" in info["reason"]
    assert info["licence"].startswith("Apache-2.0") and info["job_types"] == ["transcribe.melody"] and info["vram_gb"] == 0.0
    validate_model_info(info, "basicpitch")
    with pytest.raises(Exception, match="basic_pitch"):
        family.load("basic-pitch")


def test_unavailable_when_no_inference_runtime_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    # basic_pitch/__init__.py raises NameError at import when TF, CoreML, TFLite and ONNX are all missing
    put(monkeypatch, module("basic_pitch"))
    monkeypatch.delitem(__import__("sys").modules, "basic_pitch.inference", raising=False)

    def explode(name: str) -> Any:
        raise NameError("name '_default_model_type' is not defined")

    monkeypatch.setattr("forge.workers._upstream.importlib.import_module", explode)
    family = bp.BasicPitchFamily()
    family.configure({})
    (info,) = family.models()
    assert not info["available"] and "NameError" in info["reason"]


def test_available_and_invalid_options(monkeypatch: pytest.MonkeyPatch) -> None:
    install_basic_pitch(monkeypatch)
    family = bp.BasicPitchFamily()
    family.configure({})
    (info,) = family.models()
    assert info["available"] is True and "reason" not in info and info["checkpoint"].startswith("builtin:")
    family.configure({"onset_threshold": 2})
    (info,) = family.models()
    assert not info["available"] and "invalid options" in info["reason"]


# ---- transcribe.melody ----------------------------------------------------------------------------

def test_load_preloads_the_model_and_run_calls_predict(env: tuple[bp.BasicPitchFamily, FakeBasicPitch, Path]) -> None:
    family, state, tmp = env
    assert len(state.models) == 1 and state.models[0].path == "/fake/nmp"
    inputs = audio_input(tmp)
    job = make_job(tmp, "transcribe.melody", "basic-pitch", {"min_note_ms": 90.0, "bpm": 100.0}, inputs)
    outputs, events = run_job(family, job)
    call = state.calls[0]
    assert call["audio_path"] == inputs["audio"] and call["model"] is state.models[0]
    assert (call["minimum_note_length"], call["midi_tempo"], call["onset_threshold"], call["frame_threshold"]) == (90.0, 100.0, 0.5, 0.3)
    assert call["melodia_trick"] is True
    assert len(outputs) == 1 and outputs[0]["kind"] == "midi" and outputs[0]["candidate"] == 0
    assert inside(job["out_dir"], outputs[0]["path"]) and outputs[0]["path"].endswith("melody.mid")
    notes = outputs[0]["extra"]["notes"]
    assert [n["pitch"] for n in notes] == [60, 67, 64]  # sorted by start time
    assert notes[0] == {"pitch": 60, "start_s": 0.0, "end_s": 0.5, "velocity": 102, "confidence": 0.8}
    assert outputs[0]["extra"]["bpm"] == 100.0
    parsed = midi.read_smf(outputs[0]["path"])
    assert sorted(n.pitch for n in parsed) == [60, 64, 67]
    assert events[-1] == (1.0, "3 notes")
    validate_outputs(outputs, Path(job["out_dir"]))


def test_output_has_the_same_shape_as_the_yin_family(env: tuple[bp.BasicPitchFamily, FakeBasicPitch, Path]) -> None:
    family, _, tmp = env
    inputs = audio_input(tmp, seconds=1.5)
    mine, _ = run_job(family, make_job(tmp, "transcribe.melody", "basic-pitch", {"min_note_ms": 60.0, "bpm": 120.0}, inputs))
    other = yin.YinFamily()
    theirs, _ = run_job(other, make_job(tmp / "y", "transcribe.melody", "yin", {"min_note_ms": 60.0, "bpm": 120.0}, inputs))
    for a, b in zip(mine, theirs, strict=True):
        assert set(a) == set(b) and a["kind"] == b["kind"] == "midi" and set(a["extra"]) == set(b["extra"])
    assert set(mine[0]["extra"]["notes"][0]) == set(theirs[0]["extra"]["notes"][0]) if theirs[0]["extra"]["notes"] else True


def test_events_are_validated_and_capped() -> None:
    nan, inf = float("nan"), float("inf")
    events = [
        (0.0, 0.5, 60, 0.5, None), (0.1, 0.2, 200, 0.5, None), (-1.0, 0.2, 60, 0.5, None), (0.5, 0.5, 60, 0.5, None),
        (0.0, nan, 60, 0.5, None), (0.0, 1.0, 61, inf, None), ("a", 1.0, 62, 0.5, None), (0.0,), None, (0.2, 0.9, 62, 7.0, None),
        (0.3, 0.9, 63, 0.0, None),
    ]
    notes = bp.notes_from_events(events)
    assert [(n["pitch"], n["velocity"], n["confidence"]) for n in notes] == [(60, 64, 0.5), (62, 127, 1.0), (63, 1, 0.0)]
    many = [(i * 0.01, i * 0.01 + 0.005, 60 + i % 12, 0.5, None) for i in range(bp.MAX_NOTES + 500)]
    assert len(bp.notes_from_events(many)) == bp.MAX_NOTES
    assert bp.notes_from_events(None) == []
    assert all(math.isfinite(n["start_s"]) for n in notes)


def test_no_notes_still_yields_a_midi_file(env: tuple[bp.BasicPitchFamily, FakeBasicPitch, Path]) -> None:
    family, state, tmp = env
    state.events = []
    outputs, _ = run_job(family, make_job(tmp, "transcribe.melody", "basic-pitch", {}, audio_input(tmp)))
    assert outputs[0]["extra"]["notes"] == [] and midi.read_smf(outputs[0]["path"]) == []


def test_input_validation(env: tuple[bp.BasicPitchFamily, FakeBasicPitch, Path]) -> None:
    family, state, tmp = env
    with pytest.raises(ValueError, match="missing audio input"):
        run_job(family, make_job(tmp, "transcribe.melody", "basic-pitch", {}))
    for params in ({"min_note_ms": 0}, {"min_note_ms": float("nan")}, {"bpm": 5000}, {"bpm": "fast"}):
        with pytest.raises(ValueError):
            run_job(family, make_job(tmp, "transcribe.melody", "basic-pitch", params, audio_input(tmp)))
    with pytest.raises(ValueError, match="does not run"):
        run_job(family, make_job(tmp, "generate.loop", "basic-pitch", {}, audio_input(tmp)))
    assert state.calls == []
    family.unload("basic-pitch")
    with pytest.raises(RuntimeError, match="not loaded"):
        run_job(family, make_job(tmp, "transcribe.melody", "basic-pitch", {}, audio_input(tmp)))


def test_cancellation(env: tuple[bp.BasicPitchFamily, FakeBasicPitch, Path]) -> None:
    family, state, tmp = env
    with pytest.raises(Cancelled):
        run_job(family, make_job(tmp, "transcribe.melody", "basic-pitch", {}, audio_input(tmp)), lambda: True)
    assert state.calls == []
    # cancelled while predict() was running: the result is discarded and nothing is written
    flag = {"on": False}
    original = state.calls

    class Flip(list):  # type: ignore[type-arg]
        def append(self, item: Any) -> None:
            flag["on"] = True
            super().append(item)

    state.calls = Flip(original)
    job = make_job(tmp / "b", "transcribe.melody", "basic-pitch", {}, audio_input(tmp))
    with pytest.raises(Cancelled):
        run_job(family, job, lambda: flag["on"])
    assert not list(Path(job["out_dir"]).glob("*.mid"))


def test_falls_back_to_the_model_path_when_preloading_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = install_basic_pitch(monkeypatch)

    def broken(path: Any) -> None:
        raise RuntimeError("no runtime")

    __import__("sys").modules["basic_pitch.inference"].Model = broken
    family = bp.BasicPitchFamily()
    family.configure({})
    family.load("basic-pitch")
    run_job(family, make_job(tmp_path, "transcribe.melody", "basic-pitch", {}, audio_input(tmp_path)))
    assert state.calls[0]["model"] == "/fake/nmp"


def test_protocol_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = install_basic_pitch(monkeypatch)
    with Harness(bp.BasicPitchFamily()) as h:
        _, hello = h.call("hello", options={})
        assert hello["ok"] and hello["result"]["models"][0]["available"]
        assert h.call("load", model="basic-pitch")[1]["ok"]
        job = make_job(tmp_path, "transcribe.melody", "basic-pitch", {"bpm": 90.0}, audio_input(tmp_path))
        events, final = h.call("run", job=job)
        assert final["ok"] and events
        validate_outputs(final["result"]["outputs"], Path(job["out_dir"]))
        assert h.call("unload", model="basic-pitch")[1]["ok"]
    assert len(state.calls) == 1
