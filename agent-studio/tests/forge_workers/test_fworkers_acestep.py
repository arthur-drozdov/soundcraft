"""The ``acestep`` worker against fake ``acestep`` modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from audiokit import wav
from forge.workers import acestep as ace
from forge.workers.protocol import validate_model_info, validate_outputs
from forge.workers.runtime import Cancelled

from fworkers_fakes import FakeAce, Harness, block, inside, install_ace, listing, make_job, make_wav, run_job

LM = "acestep-5Hz-lm-1.7B"


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ace.AceStepFamily, FakeAce, Path]:
    state = install_ace(monkeypatch)
    monkeypatch.delenv("ACESTEP_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("ACESTEP_CHECKPOINTS_DIR", raising=False)
    family = ace.AceStepFamily()
    family.configure({"device": "cpu", "lm": LM, "project_root": str(tmp_path / "proj"), "steps": 4})
    family.load("ace-step-1.5-base")
    return family, state, tmp_path


def source(tmp: Path, seconds: float = 3.0) -> dict[str, str]:
    return {"audio": str(make_wav(tmp / "in" / "a.wav", seconds=seconds, sr=22050))}


# ---- availability and loading ---------------------------------------------------------------------

def test_models_report_missing_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    block(monkeypatch, "torch", "acestep")
    family = ace.AceStepFamily()
    family.configure({"device": "cuda", "checkpoints": {"ace-step-1.5-base": "ACE-Step/acestep-v15-base"}, "lm": LM})
    models = {m["id"]: m for m in family.models()}
    assert set(models) == {"ace-step-1.5-base", "ace-step-1.5-xl-base"}
    for info in models.values():
        assert not info["available"] and "ACE-Step-1.5" in info["reason"] and info["licence"].startswith("MIT")
        assert info["job_types"] == ["layer.add", "stems.split", "generate.loop"] and info["max_seconds"] == 600.0
        validate_model_info(info, "acestep")
    assert models["ace-step-1.5-base"]["checkpoint"] == "ACE-Step/acestep-v15-base"
    assert models["ace-step-1.5-xl-base"]["checkpoint"] == "ACE-Step/acestep-v15-xl-base"  # the default
    assert models["ace-step-1.5-xl-base"]["vram_gb"] > models["ace-step-1.5-base"]["vram_gb"]


def test_available_when_installed_and_option_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    install_ace(monkeypatch)
    family = ace.AceStepFamily()
    family.configure({})
    assert all(m["available"] for m in family.models())
    for bad in ({"lm": "x; rm -rf /"}, {"lm_backend": "tensorrt"}, {"device": "npu"}, {"checkpoints": {"a": 1}}, {"steps": 0}):
        family.configure(bad)
        assert all("invalid options" in m["reason"] for m in family.models()), bad


def test_variant_names() -> None:
    assert ace.variant_of("ACE-Step/acestep-v15-xl-base") == "acestep-v15-xl-base"
    assert ace.variant_of("acestep-v15-base") == "acestep-v15-base"
    for bad in ("", "a/..", "x\\y", "a/\x00"):
        with pytest.raises(ValueError):
            ace.variant_of(bad)


def test_load_initialises_the_handlers_like_upstream(env: tuple[ace.AceStepFamily, FakeAce, Path]) -> None:
    family, state, tmp = env
    assert state.init_calls == [{"project_root": str(tmp / "proj"), "config_path": "acestep-v15-base", "device": "cpu"}]
    assert state.lm_calls == [{"checkpoint_dir": str(tmp / "proj" / "checkpoints"), "lm_model_path": LM, "backend": "pt", "device": "cpu"}]
    family.load("ace-step-1.5-base")  # idempotent
    assert len(state.init_calls) == 1
    family.load("ace-step-1.5-xl-base")
    assert state.init_calls[1]["config_path"] == "acestep-v15-xl-base"
    family.unload("ace-step-1.5-base")
    family.unload("ace-step-1.5-base")
    with pytest.raises(RuntimeError, match="not loaded"):
        run_job(family, make_job(tmp, "generate.loop", "ace-step-1.5-base", {"bars": 4, "bpm": 120}))


def test_checkpoint_directory_follows_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = install_ace(monkeypatch)
    monkeypatch.setenv("ACESTEP_CHECKPOINTS_DIR", str(tmp_path / "shared"))
    family = ace.AceStepFamily()
    family.configure({"device": "cpu", "lm": LM})
    family.load("ace-step-1.5-base")
    assert state.lm_calls[0]["checkpoint_dir"] == str(tmp_path / "shared")


def test_failed_initialisation_and_optional_lm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = install_ace(monkeypatch)
    state.init_ok = False
    family = ace.AceStepFamily()
    family.configure({"device": "cpu", "lm": LM, "project_root": str(tmp_path)})
    with pytest.raises(RuntimeError, match="could not initialise acestep-v15-base: boom"):
        family.load("ace-step-1.5-base")
    state.init_ok, state.lm_ok = True, False  # the language model is optional: DiT-only is fine
    family.load("ace-step-1.5-base")
    job = make_job(tmp_path, "generate.loop", "ace-step-1.5-base", {"bars": 4, "bpm": 120})
    run_job(family, job)
    assert state.calls[0]["params"].thinking is False
    family.unload("ace-step-1.5-base")
    family.configure({"device": "cpu", "project_root": str(tmp_path)})  # no lm configured at all
    family.load("ace-step-1.5-base")
    assert len(state.lm_calls) == 1  # only the failed attempt before


# ---- generate.loop --------------------------------------------------------------------------------

def test_loop_mapping(env: tuple[ace.AceStepFamily, FakeAce, Path]) -> None:
    family, state, tmp = env
    job = make_job(tmp, "generate.loop", "ace-step-1.5-base", {"bars": 4, "bpm": 120, "key": "Bb minor", "prompt": "deep house", "tags": {"genre": "House"}, "candidates": 2}, seed=2**40 + 5)
    outputs, events = run_job(family, job)
    first = state.calls[0]
    params, config = first["params"], first["config"]
    assert params.task_type == "text2music" and params.lyrics == "[Instrumental]" and params.instrumental is True
    assert params.caption == "deep house, instrumental, seamless loop, Genre: House"
    assert (params.bpm, params.keyscale, params.timesignature) == (120, "A# minor", "4")
    assert params.duration == 16.0  # 4 bars at 120 BPM are 8 s; ACE-Step needs 10 s, so two repetitions are generated
    assert (params.inference_steps, params.guidance_scale, params.thinking) == (4, 7.0, True)
    assert not params.use_cot_caption and not params.use_cot_metas and not params.use_cot_language and not params.enable_normalization
    assert (config.batch_size, config.use_random_seed, config.audio_format) == (1, False, "wav32")
    assert first["save_dir"] is None  # the library writes nothing; the worker writes inside out_dir
    assert [c["config"].seeds for c in state.calls] == [[(2**40 + 5) % 2**32], [(2**40 + 6) % 2**32]]
    for output in outputs:
        audio, sr = wav.read(output["path"])
        assert sr == 48000 and audio.shape == (2, 8 * 48000)  # cut back to the loop length
        assert inside(job["out_dir"], output["path"])
    assert outputs[0]["extra"]["generated_s"] == 16.0 and events[-1][0] == 1.0
    validate_outputs(outputs, Path(job["out_dir"]))
    run_job(family, make_job(tmp / "b", "generate.loop", "ace-step-1.5-base", {"bars": 16, "bpm": 60}))
    assert state.calls[2]["params"].duration == 64.0  # long enough already: no repetition


def test_loop_plan() -> None:
    assert ace.loop_generation_plan(4, 120) == (8.0, 16.0)
    assert ace.loop_generation_plan(1, 240) == (1.0, 10.0)
    assert ace.loop_generation_plan(16, 40) == (96.0, 96.0)
    assert ace.loop_generation_plan(3, 119)[1] >= 10.0


def test_loop_needs_bpm(env: tuple[ace.AceStepFamily, FakeAce, Path]) -> None:
    family, state, tmp = env
    with pytest.raises(ValueError, match="needs bpm"):
        run_job(family, make_job(tmp, "generate.loop", "ace-step-1.5-base", {"bars": 4}))
    with pytest.raises(ValueError, match="bpm"):
        run_job(family, make_job(tmp, "generate.loop", "ace-step-1.5-base", {"bars": 4, "bpm": 1000}))
    assert state.calls == []


# ---- layer.add and stems.split --------------------------------------------------------------------

def test_layer_add_is_lego_with_the_track_from_the_description(env: tuple[ace.AceStepFamily, FakeAce, Path]) -> None:
    family, state, tmp = env
    inputs = source(tmp)
    outputs, _ = run_job(family, make_job(tmp, "layer.add", "ace-step-1.5-base", {"description": "a warm analog pad", "prompt": "slow attack"}, inputs))
    params = state.calls[0]["params"]
    assert params.task_type == "lego" and params.instruction == "Generate the synth track based on the audio context:"
    assert params.src_audio == inputs["audio"] and params.caption == "a warm analog pad, slow attack" and params.duration == -1.0
    assert (params.repainting_start, params.repainting_end, params.thinking) == (0.0, -1, False)  # the LM is skipped for lego
    assert outputs[0]["extra"]["track"] == "synth"
    run_job(family, make_job(tmp, "layer.add", "ace-step-1.5-base", {"description": "pad", "tags": {"track": "strings"}}, inputs))
    assert state.calls[1]["params"].instruction == "Generate the strings track based on the audio context:"
    run_job(family, make_job(tmp, "layer.add", "ace-step-1.5-base", {"description": "something indescribable"}, inputs))
    assert state.calls[2]["params"].instruction == ace.LEGO_DEFAULT_INSTRUCTION
    with pytest.raises(ValueError, match="needs a description"):
        run_job(family, make_job(tmp, "layer.add", "ace-step-1.5-base", {"description": "  "}, inputs))
    with pytest.raises(ValueError, match="missing audio input"):
        run_job(family, make_job(tmp, "layer.add", "ace-step-1.5-base", {"description": "pad"}))


def test_stems_split_is_extract_per_stem(env: tuple[ace.AceStepFamily, FakeAce, Path]) -> None:
    family, state, tmp = env
    inputs = source(tmp)
    job = make_job(tmp, "stems.split", "ace-step-1.5-base", {"stems": ["piano", "Drums", "keyboard"], "candidates": 2}, inputs)
    outputs, _ = run_job(family, job)
    assert [c["params"].instruction for c in state.calls] == [
        "Extract the keyboard track from the audio:", "Extract the drums track from the audio:",
    ] * 2
    assert all(c["params"].task_type == "extract" and c["params"].src_audio == inputs["audio"] for c in state.calls)
    assert [(o["candidate"], o["extra"]["stem"]) for o in outputs] == [(0, "keyboard"), (0, "drums"), (1, "keyboard"), (1, "drums")]
    assert len({o["path"] for o in outputs}) == 4 and all(inside(job["out_dir"], o["path"]) for o in outputs)
    assert state.calls[0]["config"].seeds == [42] and state.calls[2]["config"].seeds == [43]
    validate_outputs(outputs, Path(job["out_dir"]))


def test_stems_split_defaults_and_validation(env: tuple[ace.AceStepFamily, FakeAce, Path]) -> None:
    family, state, tmp = env
    inputs = source(tmp)
    outputs, _ = run_job(family, make_job(tmp, "stems.split", "ace-step-1.5-base", {}, inputs))
    assert [o["extra"]["stem"] for o in outputs] == list(ace.DEFAULT_STEMS)
    for bad in (["kazoo"], "vocals", ["vocals"] * 17):
        with pytest.raises(ValueError):
            run_job(family, make_job(tmp, "stems.split", "ace-step-1.5-base", {"stems": bad}, inputs))
    too_long = make_wav(tmp / "long.wav", seconds=601.0, sr=2000, channels=1)
    with pytest.raises(ValueError, match="accepts at most 600 s"):
        run_job(family, make_job(tmp, "stems.split", "ace-step-1.5-base", {}, {"audio": str(too_long)}))
    assert len(state.calls) == 3


# ---- failures and cancellation --------------------------------------------------------------------

def test_failures_are_reported(env: tuple[ace.AceStepFamily, FakeAce, Path]) -> None:
    family, state, tmp = env
    state.fail_with = "CUDA out of memory"
    with pytest.raises(RuntimeError, match="ACE-Step failed: CUDA out of memory"):
        run_job(family, make_job(tmp, "generate.loop", "ace-step-1.5-base", {"bars": 4, "bpm": 120}))
    assert listing(tmp / "out") == []


def test_cancel_from_the_progress_callback(env: tuple[ace.AceStepFamily, FakeAce, Path]) -> None:
    family, state, tmp = env
    seen: list[int] = []

    def cancelled() -> bool:
        seen.append(1)
        return len(seen) > 2

    job = make_job(tmp, "generate.loop", "ace-step-1.5-base", {"bars": 4, "bpm": 120})
    with pytest.raises(Cancelled):
        run_job(family, job, cancelled)
    # upstream turns exceptions into a failed result: the worker must still report a cancellation, not a failure
    state.swallow_exceptions = True
    seen.clear()
    with pytest.raises(Cancelled):
        run_job(family, make_job(tmp / "b", "generate.loop", "ace-step-1.5-base", {"bars": 4, "bpm": 120}), cancelled)
    assert listing(tmp) == []


def test_cancel_between_candidates(env: tuple[ace.AceStepFamily, FakeAce, Path]) -> None:
    family, state, tmp = env
    job = make_job(tmp, "generate.loop", "ace-step-1.5-base", {"bars": 4, "bpm": 120, "candidates": 3})
    with pytest.raises(Cancelled):
        run_job(family, job, (Path(job["out_dir"]) / "candidate_0.wav").exists)
    assert len(state.calls) == 1


# ---- protocol -------------------------------------------------------------------------------------

def test_protocol_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = install_ace(monkeypatch)
    inputs = source(tmp_path)
    with Harness(ace.AceStepFamily()) as h:
        _, hello = h.call("hello", options={"device": "cpu", "project_root": str(tmp_path / "p")})
        assert hello["ok"] and all(m["available"] for m in hello["result"]["models"])
        assert h.call("load", model="ace-step-1.5-base")[1]["ok"]
        job = make_job(tmp_path, "stems.split", "ace-step-1.5-base", {"stems": ["vocals"]}, inputs)
        events, final = h.call("run", job=job)
        assert final["ok"] and final["result"]["outputs"][0]["extra"] == {"stem": "vocals"}
        assert events and events[-1]["progress"] <= 1.0
        validate_outputs(final["result"]["outputs"], Path(job["out_dir"]))
        bad = make_job(tmp_path / "b", "stems.split", "ace-step-1.5-base", {"stems": ["kazoo"]}, inputs)
        failed = h.call("run", job=bad)[1]
        assert failed["ok"] is False and "unknown ACE-Step track" in failed["error"]
        assert h.call("unload", model="ace-step-1.5-base")[1]["ok"]
    assert len(state.calls) == 1
