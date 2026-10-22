"""The ``sa3`` worker against a fake ``stable_audio_3`` package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audiokit import wav
from forge.workers import sa3
from forge.workers.protocol import validate_model_info, validate_outputs
from forge.workers.runtime import Cancelled

from fworkers_fakes import SR, FakeSA3, Harness, block, inside, install_sa3, make_job, make_wav, run_job

OPTIONS = {"device": "cpu", "steps": 3}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[sa3.SA3Family, FakeSA3, Path]:
    state = install_sa3(monkeypatch, cuda=True)  # medium needs CUDA
    family = sa3.SA3Family()
    family.configure({"device": "cuda", "steps": 3})
    for model in sa3.MODELS:
        family.load(model)
    return family, state, tmp_path


# ---- availability ---------------------------------------------------------------------------------

def test_models_report_missing_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    block(monkeypatch, "torch", "stable_audio_3")
    family = sa3.SA3Family()
    family.configure(dict(OPTIONS))
    models = {m["id"]: m for m in family.models()}
    assert set(models) == {"sa3-small-music", "sa3-small-sfx", "sa3-medium"}
    for info in models.values():
        assert not info["available"] and "stable-audio-3" in info["reason"]
        assert "Community Licence" in info["licence"]
        validate_model_info(info, "sa3")
    assert [models[m]["max_seconds"] for m in ("sa3-small-music", "sa3-small-sfx", "sa3-medium")] == [120.0, 120.0, 380.0]
    assert "generate.sfx" in models["sa3-small-sfx"]["job_types"] and "generate.sfx" not in models["sa3-small-music"]["job_types"]
    assert "generate.loop" in models["sa3-small-music"]["job_types"] and "generate.loop" not in models["sa3-small-sfx"]["job_types"]
    assert sorted(models["sa3-medium"]["job_types"]) == sorted(sa3.ALL_JOBS)
    assert models["sa3-medium"]["vram_gb"] > models["sa3-small-music"]["vram_gb"]


def test_checkpoint_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_sa3(monkeypatch)
    family = sa3.SA3Family()
    family.configure({"device": "cpu"})  # no checkpoints: the model names are the defaults
    infos = {m["id"]: m for m in family.models()}
    assert all(m["available"] for m in infos.values()) and infos["sa3-medium"]["checkpoint"] == "medium"
    family.configure({"device": "cpu", "checkpoints": {"sa3-medium": "/nope/where"}})
    assert "neither a model name" in {m["id"]: m for m in family.models()}["sa3-medium"]["reason"]
    local = tmp_path / "local"
    local.mkdir()
    family.configure({"device": "cpu", "checkpoints": {"sa3-medium": str(local)}})
    assert "model_config.json" in {m["id"]: m for m in family.models()}["sa3-medium"]["reason"]
    (local / "model_config.json").write_text("{}")
    (local / "model.safetensors").write_bytes(b"weights")
    info = {m["id"]: m for m in family.models()}["sa3-medium"]
    assert info["available"] and info["checkpoint"] == str(local) and info["checkpoint_hash"].startswith("fp:")
    family.configure({"device": "nonsense"})
    assert all("invalid options" in m["reason"] for m in family.models())
    family.configure({"device": "cpu", "checkpoints": {"sa3-medium": 3}})
    assert all("invalid options" in m["reason"] for m in family.models())


# ---- load -----------------------------------------------------------------------------------------

def test_load_named_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = install_sa3(monkeypatch, cuda=True)
    family = sa3.SA3Family()
    family.configure({"device": "cuda"})
    family.load("sa3-small-music")
    family.load("sa3-medium")
    assert state.loads == [("small-music", {"device": "cuda", "model_half": True}), ("medium", {"device": "cuda", "model_half": True})]
    family.load("sa3-medium")  # idempotent
    assert len(state.loads) == 2
    family.unload("sa3-medium")
    family.unload("sa3-medium")
    with pytest.raises(RuntimeError, match="not loaded"):
        run_job(family, make_job(tmp_path, "generate.sfx", "sa3-medium", {"duration_s": 1.0}))
    with pytest.raises(ValueError, match="unknown model"):
        family.load("nope")


def test_medium_needs_cuda_but_small_runs_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    state = install_sa3(monkeypatch, cuda=False)
    family = sa3.SA3Family()
    family.configure({"device": "cuda"})  # asked for cuda, there is none
    family.load("sa3-small-sfx")
    assert state.loads == [("small-sfx", {"device": "cpu", "model_half": False})]
    with pytest.raises(Exception, match="needs a CUDA GPU"):
        family.load("sa3-medium")


def test_local_directories_load_through_loading_utils(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = install_sa3(monkeypatch)
    local = tmp_path / "m"
    local.mkdir()
    (local / "model_config.json").write_text(json.dumps({"model_type": "diffusion_cond"}))
    (local / "model.safetensors").write_bytes(b"w")
    family = sa3.SA3Family()
    family.configure({"device": "cpu", "checkpoints": {"sa3-small-music": str(local)}})
    family.load("sa3-small-music")
    assert state.loads == []
    assert state.local_loads == [{"config": {"model_type": "diffusion_cond"}, "ckpt": str(local / "model.safetensors"), "device": "cpu", "model_half": False}]
    outputs, _ = run_job(family, make_job(tmp_path, "generate.sample", "sa3-small-music", {"note": 60, "length_s": 0.5}))
    assert outputs and state.calls


# ---- text to audio --------------------------------------------------------------------------------

def test_sfx_mapping(env: tuple[sa3.SA3Family, FakeSA3, Path]) -> None:
    family, state, tmp = env
    job = make_job(tmp, "generate.sfx", "sa3-small-sfx", {"duration_s": 4.0, "prompt": "heavy door slam", "tags": {"genre": "Cinematic"}, "candidates": 2}, seed=7)
    outputs, events = run_job(family, job)
    first = state.calls[0]
    assert first["prompt"] == "TrackType: SFX, heavy door slam, Genre: Cinematic"
    assert (first["duration"], first["steps"], first["cfg_scale"], first["batch_size"]) == (4.0, 3, 1.0, 1)
    assert first["sample_size"] == 5292032 and "init_audio" not in first and "inpaint_audio" not in first
    assert [c["seed"] for c in state.calls] == [7, 8]
    for output in outputs:
        audio, sr = wav.read(output["path"])
        assert sr == SR and audio.shape == (2, 4 * SR)  # SA3 returned a little extra; cut to the duration
        assert inside(job["out_dir"], output["path"])
    assert events[-1][0] == 1.0 and any("step" in m for _, m in events)
    validate_outputs(outputs, Path(job["out_dir"]))


def test_loop_mapping(env: tuple[sa3.SA3Family, FakeSA3, Path]) -> None:
    family, state, tmp = env
    outputs, _ = run_job(family, make_job(tmp, "generate.loop", "sa3-small-music", {"bars": 4, "bpm": 124, "key": "Bb minor", "prompt": "funky bass"}))
    call = state.calls[0]
    assert call["prompt"] == "TrackType: Music, VocalType: Instrumental, funky bass, Key: A# minor, 124 BPM, 4 bars"
    assert call["duration"] == pytest.approx(60 / 124 * 16)
    audio, _ = wav.read(outputs[0]["path"])
    assert audio.shape[1] == round(60 / 124 * 16 * SR)
    with pytest.raises(ValueError, match="needs bpm"):
        run_job(family, make_job(tmp, "generate.loop", "sa3-small-music", {"bars": 4}))
    with pytest.raises(ValueError, match="does not run"):
        run_job(family, make_job(tmp, "generate.loop", "sa3-small-sfx", {"bars": 4, "bpm": 120}))


def test_sample_mapping(env: tuple[sa3.SA3Family, FakeSA3, Path]) -> None:
    family, state, tmp = env
    run_job(family, make_job(tmp, "generate.sample", "sa3-small-sfx", {"note": 60, "length_s": 0.2, "prompt": "pluck"}))
    run_job(family, make_job(tmp / "b", "generate.sample", "sa3-medium", {"note": 36, "length_s": 3.0}))
    assert state.calls[0]["prompt"] == "TrackType: SFX, pluck, single note C4"
    assert state.calls[0]["duration"] == 1.0  # generated at least 1 s, then cut to length_s
    assert state.calls[1]["prompt"] == "TrackType: Instrument, single note C2" and state.calls[1]["duration"] == 3.0
    audio, _ = wav.read(next((tmp / "out").glob("*.wav")))
    assert audio.shape[1] == round(0.2 * SR) and audio[0, -1] == 0.0  # short fade-out on the cut


def test_the_model_maximum_is_enforced(env: tuple[sa3.SA3Family, FakeSA3, Path]) -> None:
    family, state, tmp = env
    short = make_wav(tmp / "in.wav", seconds=119.0, sr=8000, channels=1)
    with pytest.raises(ValueError, match="exceeds the model maximum"):
        run_job(family, make_job(tmp, "extend.audio", "sa3-small-music", {"seconds": 5.0}, {"audio": str(short)}))
    too_long = make_wav(tmp / "long.wav", seconds=121.0, sr=8000, channels=1)
    with pytest.raises(ValueError, match="accepts at most 120 s"):
        run_job(family, make_job(tmp, "transform.audio", "sa3-small-music", {"strength": 0.5}, {"audio": str(too_long)}))
    assert state.calls == []
    run_job(family, make_job(tmp / "m", "transform.audio", "sa3-medium", {"strength": 0.5}, {"audio": str(too_long)}))  # medium goes to 380 s
    assert state.calls[0]["sample_size"] == 380 * SR


# ---- audio inputs ---------------------------------------------------------------------------------

def test_transform_maps_strength_to_init_noise_level(env: tuple[sa3.SA3Family, FakeSA3, Path]) -> None:
    family, state, tmp = env
    source = make_wav(tmp / "in" / "a.wav", seconds=2.0, sr=22050)
    outputs, _ = run_job(family, make_job(tmp, "transform.audio", "sa3-small-music", {"strength": 0.8, "prompt": "bossa nova bassline"}, {"audio": str(source)}))
    call = state.calls[0]
    sr_in, tensor = call["init_audio"]
    assert sr_in == 22050 and tensor.shape == (2, 44100)
    assert call["init_noise_level"] == 0.8 and call["prompt"] == "bossa nova bassline" and call["duration"] == pytest.approx(2.0)
    assert "inpaint_audio" not in call
    audio, _ = wav.read(outputs[0]["path"])
    assert audio.shape[1] == 2 * SR
    for bad in ({"strength": 1.5}, {"strength": float("nan")}, {"strength": "high"}):
        with pytest.raises(ValueError, match="strength"):
            run_job(family, make_job(tmp, "transform.audio", "sa3-small-music", bad, {"audio": str(source)}))


def test_inpaint_passes_the_mask_seconds(env: tuple[sa3.SA3Family, FakeSA3, Path]) -> None:
    family, state, tmp = env
    source = make_wav(tmp / "in.wav", seconds=10.0, sr=16000)
    run_job(family, make_job(tmp, "inpaint.audio", "sa3-small-music", {"start_s": 4.0, "end_s": 8.0, "prompt": "punchy kick drum fill"}, {"audio": str(source)}))
    call = state.calls[0]
    assert call["inpaint_mask_start_seconds"] == 4.0 and call["inpaint_mask_end_seconds"] == 8.0
    assert call["inpaint_audio"][0] == 16000 and call["duration"] == pytest.approx(10.0) and "init_audio" not in call
    run_job(family, make_job(tmp / "b", "inpaint.audio", "sa3-small-music", {"start_s": 8.0, "end_s": 30.0}, {"audio": str(source)}))
    assert state.calls[1]["inpaint_mask_end_seconds"] == pytest.approx(10.0)  # clipped to the input
    for bad in ({"start_s": 12.0, "end_s": 14.0}, {"start_s": 5.0, "end_s": 5.0}):
        with pytest.raises(ValueError, match="lies outside"):
            run_job(family, make_job(tmp / "c", "inpaint.audio", "sa3-small-music", bad, {"audio": str(source)}))


def test_extend_is_continuation_from_the_end_of_the_input(env: tuple[sa3.SA3Family, FakeSA3, Path]) -> None:
    family, state, tmp = env
    source = make_wav(tmp / "in.wav", seconds=10.0, sr=16000)
    outputs, _ = run_job(family, make_job(tmp, "extend.audio", "sa3-small-music", {"seconds": 8.0, "prompt": "dream-like synthpop outro"}, {"audio": str(source)}))
    call = state.calls[0]
    assert call["inpaint_mask_start_seconds"] == pytest.approx(10.0) and call["inpaint_mask_end_seconds"] == pytest.approx(18.0)
    assert call["duration"] == pytest.approx(18.0)
    audio, _ = wav.read(outputs[0]["path"])
    assert audio.shape[1] == round(18.0 * SR)  # original plus extension
    assert outputs[0]["extra"]["extended_from_s"] == 10.0


def test_missing_or_bad_audio_input(env: tuple[sa3.SA3Family, FakeSA3, Path]) -> None:
    family, state, tmp = env
    with pytest.raises(ValueError, match="missing audio input"):
        run_job(family, make_job(tmp, "transform.audio", "sa3-small-music", {"strength": 0.5}))
    junk = tmp / "junk.wav"
    junk.write_bytes(b"not a wav")
    with pytest.raises(wav.WavError):
        run_job(family, make_job(tmp, "transform.audio", "sa3-small-music", {"strength": 0.5}, {"audio": str(junk)}))
    assert state.calls == []


# ---- cancellation and protocol --------------------------------------------------------------------

def test_cancellation(env: tuple[sa3.SA3Family, FakeSA3, Path]) -> None:
    family, state, tmp = env
    seen: list[int] = []

    def cancelled() -> bool:
        seen.append(1)
        return len(seen) > 2

    job = make_job(tmp, "generate.sfx", "sa3-small-sfx", {"duration_s": 2.0})
    with pytest.raises(Cancelled):
        run_job(family, job, cancelled)
    assert len(state.calls) == 1 and not list(Path(job["out_dir"]).glob("*.wav"))
    two = make_job(tmp / "b", "generate.sfx", "sa3-small-sfx", {"duration_s": 2.0, "candidates": 3})
    with pytest.raises(Cancelled):
        run_job(family, two, (Path(two["out_dir"]) / "candidate_0.wav").exists)
    assert len(state.calls) == 2 and [p.name for p in Path(two["out_dir"]).glob("*.wav")] == ["candidate_0.wav"]


def test_protocol_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = install_sa3(monkeypatch)
    with Harness(sa3.SA3Family()) as h:
        _, hello = h.call("hello", options=dict(OPTIONS))
        assert hello["ok"] and all(m["available"] for m in hello["result"]["models"])
        assert h.call("load", model="sa3-small-sfx")[1]["ok"]
        job = make_job(tmp_path, "generate.sfx", "sa3-small-sfx", {"duration_s": 1.5})
        events, final = h.call("run", job=job)
        assert final["ok"] and events and final["result"]["outputs"][0]["kind"] == "audio"
        validate_outputs(final["result"]["outputs"], Path(job["out_dir"]))
        bad = make_job(tmp_path / "b", "generate.sfx", "sa3-small-sfx", {"duration_s": "soon"})
        reply = h.call("run", job=bad)[1]
        assert reply["ok"] is False and "duration_s" in reply["error"]  # a bad job fails cleanly; the worker lives on
        assert h.call("unload", model="sa3-small-sfx")[1]["ok"]
    assert state.calls
