"""The ``foundation`` worker against fake RC Stable Audio Tools modules."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from audiokit import wav
from audiokit.instruments import plan_keybed
from forge.jobs import validate_job
from forge.postprocess import ModelMeta, PostContext, process_output
from forge.workers import foundation as fnd
from forge.workers.protocol import validate_model_info, validate_outputs
from forge.workers.runtime import Cancelled

from fworkers_fakes import (
    SR, FakeSat, Harness, block, foundation_options, inside, install_sat, install_torch, listing, make_job, make_wav, run_job,
)

LOOP = {"bars": 8, "bpm": 128, "key": "F minor", "tags": {"family": "Synth", "subfamily": "Synth Lead", "descriptors": ["warm"]}}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[fnd.FoundationFamily, FakeSat, Path]:
    sat = install_sat(monkeypatch)
    family = fnd.FoundationFamily()
    family.configure(foundation_options(tmp_path))
    return family, sat, tmp_path


def loaded(env: tuple[fnd.FoundationFamily, FakeSat, Path], model: str) -> fnd.FoundationFamily:
    family = env[0]
    family.load(model)
    return family


# ---- availability ---------------------------------------------------------------------------------

def test_models_report_missing_packages_with_a_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    block(monkeypatch, "torch", "stable_audio_tools")
    family = fnd.FoundationFamily()
    family.configure(foundation_options(tmp_path))
    models = {m["id"]: m for m in family.models()}
    assert set(models) == {"foundation-1", "foundation-1.2-samples", "foundation-1.2-keybeds"}
    for info in models.values():
        assert info["available"] is False and "torch" in info["reason"] and "RC-stable-audio-tools" in info["reason"]
        assert "Community Licence" in info["licence"] and info["vram_gb"] == 7.0 and info["family"] == "foundation"
        validate_model_info(info, "foundation")
    assert models["foundation-1"]["job_types"] == ["generate.loop", "transform.audio"]
    assert models["foundation-1.2-samples"]["job_types"] == ["generate.sample", "generate.loop", "transform.audio"]
    assert models["foundation-1.2-keybeds"]["job_types"] == ["generate.instrument"]


def test_models_report_checkpoint_problems(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_sat(monkeypatch)
    family = fnd.FoundationFamily()
    family.configure({"device": "cpu"})  # nothing configured
    for info in family.models():
        assert not info["available"] and f"options.checkpoints.{info['id']} is not configured" in info["reason"]

    options = foundation_options(tmp_path, models=("foundation-1",))
    Path(options["checkpoints"]["foundation-1"]["ckpt"]).unlink()
    family.configure(options)
    info = {m["id"]: m for m in family.models()}["foundation-1"]
    assert not info["available"] and "ckpt file" in info["reason"] and "does not exist" in info["reason"]

    options = foundation_options(tmp_path, models=("foundation-1",))
    Path(options["checkpoints"]["foundation-1"]["config"]).write_text("{not json")
    family.configure(options)
    assert "not valid JSON" in {m["id"]: m for m in family.models()}["foundation-1"]["reason"]

    Path(options["checkpoints"]["foundation-1"]["config"]).write_text(json.dumps({"sample_rate": 44100}))
    assert "sample_size" in {m["id"]: m for m in family.models()}["foundation-1"]["reason"]

    family.configure({"device": "tpu"})
    assert all("invalid options" in m["reason"] for m in family.models())
    family.configure({"device": "cpu", "checkpoints": ["not", "a", "mapping"]})
    assert all("invalid options" in m["reason"] for m in family.models())
    family.configure({"device": "cpu", "checkpoints": {"foundation-1": {"config": "/x\x00y", "ckpt": "z"}}})
    assert "is not set" in {m["id"]: m for m in family.models()}["foundation-1"]["reason"]


def test_available_models_carry_checkpoint_details(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, _, tmp = env
    models = {m["id"]: m for m in family.models()}
    for model_id, info in models.items():
        assert info["available"] is True and "reason" not in info
        assert info["checkpoint"] == str(tmp / "models" / model_id / "model.ckpt")
        assert info["checkpoint_hash"].startswith("fp:")
        assert info["max_seconds"] == 47.0  # sample_size / sample_rate from the model config
        validate_model_info(info, "foundation")


# ---- pure helpers ---------------------------------------------------------------------------------

def test_plan_window_follows_rcs_sample_size_rule() -> None:
    assert fnd.plan_window(15.0, 44100, None) == (15, 15 * 44100)
    seconds, samples = fnd.plan_window(14.2, 44100, 4096)
    assert seconds == 15 and samples % 4096 == 0 and 15 * 44100 <= samples < 15 * 44100 + 4096
    assert fnd.plan_window(2.0, 32000, 2048) == (2, 65536)
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            fnd.plan_window(bad, 44100, None)
    assert fnd.loop_seconds(8, 128) == pytest.approx(15.0)
    assert fnd.loop_seconds(4, 140) == pytest.approx(60 / 140 * 16)


def test_noise_level_mapping() -> None:
    assert fnd.noise_level(0.0, "v") == pytest.approx(0.1) and fnd.noise_level(1.0, "v") == pytest.approx(5.0)
    assert 0.1 < fnd.noise_level(0.56, "v") < 1.0  # RC's own default (0.9) sits at about 0.56
    assert fnd.noise_level(0.4, "rectified_flow") == 0.4 and fnd.noise_level(0.0, "rectified_flow") == 0.01
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            fnd.noise_level(bad, "v")


def test_trim_tail_silence_keeps_the_head_and_a_pad() -> None:
    sr = 1000
    audio = np.zeros((2, 4000), np.float32)
    audio[:, :500] = 0.5
    out = fnd.trim_tail_silence(audio, sr)
    assert out.shape[1] == 500 + 60  # last sound + 60 ms pad
    assert out[0, 0] == 0.5 and out[0, -1] == 0.0  # faded
    assert fnd.trim_tail_silence(np.zeros((1, 4000), np.float32), sr).shape[1] == 120  # at least 120 ms stays
    assert fnd.trim_tail_silence(np.full((1, 100), 0.5, np.float32), sr).shape[1] == 100  # nothing to trim


def test_velocity_edges_and_layer_seeds() -> None:
    assert fnd.velocity_edges(1) == [0, 128] and fnd.velocity_edges(2) == [0, 64, 128] and fnd.velocity_edges(3) == [0, 43, 85, 128]
    assert fnd.layer_seed(5, 0) == 5 and fnd.layer_seed(5, 1) != fnd.layer_seed(6, 0)
    assert fnd.layer_seed(2**63 - 1, 1) < 2**63


# ---- load and unload ------------------------------------------------------------------------------

def test_load_builds_the_model_like_rcs_gradio(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, sat, tmp = env
    family.load("foundation-1")
    model = sat.models[0]
    assert sat.states[0]["path"] == str(tmp / "models" / "foundation-1" / "model.ckpt")
    assert [e[0] for e in model.events] == ["load_state_dict", "to", "eval", "requires_grad_"]
    assert model.events[1] == ("to", "cpu") and model.events[3] == ("requires_grad_", False)
    family.load("foundation-1")  # idempotent
    assert len(sat.models) == 1
    family.unload("foundation-1")
    family.unload("foundation-1")  # unloading twice is fine
    family.load("foundation-1")
    assert len(sat.models) == 2


def test_load_picks_half_precision_like_rc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sat = install_sat(monkeypatch, cuda=True)
    family = fnd.FoundationFamily()
    family.configure(foundation_options(tmp_path, device="cuda"))
    family.load("foundation-1")
    assert ("to", "cuda") in sat.models[0].events and sat.models[0].events[-1] == ("to", "bfloat16")
    family.unload("foundation-1")
    install_torch(monkeypatch, cuda=True, bf16=False)
    family.load("foundation-1")
    assert sat.models[1].events[-1] == ("to", "float16")
    family.unload("foundation-1")
    family.configure(foundation_options(tmp_path, device="cuda", dtype="float32"))
    family.load("foundation-1")
    assert sat.models[2].events[-1] == ("to", "float32")


def test_half_precision_checkpoints_load_into_a_half_model_and_cuda_falls_back(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, sat, tmp = env
    sat.state_dtype = "float16"
    family.load("foundation-1")
    assert sat.models[0].events[0] == ("to", "float16")  # before load_state_dict, as RC does
    assert sat.models[0].events[1][0] == "load_state_dict"
    family.configure(foundation_options(tmp, device="cuda"))  # no CUDA in the fake torch
    family.load("foundation-1.2-samples")
    assert ("to", "cpu") in sat.models[1].events and ("to", "cuda") not in sat.models[1].events


def test_load_errors(env: tuple[fnd.FoundationFamily, FakeSat, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    family, _, tmp = env
    with pytest.raises(ValueError, match="unknown model"):
        family.load("nope")
    block(monkeypatch, "stable_audio_tools")
    with pytest.raises(Exception, match="stable_audio_tools"):
        family.load("foundation-1")
    job = make_job(tmp, "generate.loop", "foundation-1", dict(LOOP))
    with pytest.raises(RuntimeError, match="not loaded"):
        run_job(family, job)


def test_load_rejects_a_hostile_sample_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_sat(monkeypatch)
    options = foundation_options(tmp_path, models=("foundation-1",))
    Path(options["checkpoints"]["foundation-1"]["config"]).write_text(json.dumps({"sample_rate": 10**9, "sample_size": 10**9}))
    family = fnd.FoundationFamily()
    family.configure(options)
    with pytest.raises(ValueError, match="sample_rate"):
        family.load("foundation-1")


# ---- generate.loop --------------------------------------------------------------------------------

def test_loop_maps_params_to_the_sampler_call(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, sat, tmp = env
    family.load("foundation-1")
    job = make_job(tmp, "generate.loop", "foundation-1", {**LOOP, "candidates": 2, "prompt": "dark, 99 BPM"}, seed=42)
    outputs, events = run_job(family, job)
    assert len(sat.calls) == 2
    call = sat.calls[0]
    assert call["conditioning"] == [{"prompt": "Synth, Synth Lead, warm, dark, Dry, F minor, 8 Bars, 128 BPM", "seconds_start": 0.0, "seconds_total": 15.0}]
    assert call["negative_conditioning"] is None
    expected_samples = fnd.plan_window(15.0, SR, 4096)[1]
    assert call["sample_size"] == expected_samples and expected_samples % 4096 == 0
    assert (call["seed"], sat.calls[1]["seed"]) == (42, 43)  # seed + candidate, as the sidecar records
    assert (call["steps"], call["cfg_scale"], call["sampler_type"]) == (3, 7.0, "dpmpp-3m-sde")
    assert (call["sigma_min"], call["sigma_max"], call["scale_phi"], call["batch_size"], call["sample_rate"]) == (0.03, 500.0, 0.0, 1, SR)
    assert call["init_audio"] is None and call["mask_args"] is None and callable(call["callback"])
    assert call["device"] == "cpu"
    assert [o["candidate"] for o in outputs] == [0, 1] and all(o["kind"] == "audio" for o in outputs)
    assert outputs[0]["extra"]["conditioning_prompt"].endswith("128 BPM")
    for output in outputs:
        audio, sr = wav.read(output["path"])
        assert sr == SR and audio.shape == (2, round(15.0 * SR))  # cut to the exact loop length
        assert inside(job["out_dir"], output["path"])
    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0
    assert any("step 3/3" in m for _, m in events)
    validate_outputs(outputs, Path(job["out_dir"]))


def test_loop_validation(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family = loaded(env, "foundation-1")
    tmp = env[2]
    with pytest.raises(ValueError, match="needs bpm"):
        run_job(family, make_job(tmp, "generate.loop", "foundation-1", {"bars": 4}))
    with pytest.raises(ValueError, match="exceeds this model's maximum"):
        run_job(family, make_job(tmp, "generate.loop", "foundation-1", {"bars": 16, "bpm": 40}))
    with pytest.raises(ValueError, match="bpm"):
        run_job(family, make_job(tmp, "generate.loop", "foundation-1", {"bars": 4, "bpm": float("nan")}))
    with pytest.raises(ValueError, match="unrecognised key"):
        run_job(family, make_job(tmp, "generate.loop", "foundation-1", {"bars": 4, "bpm": 120, "key": "H major"}))
    with pytest.raises(ValueError, match="does not run"):
        run_job(family, make_job(tmp, "generate.instrument", "foundation-1", {"name": "x"}))
    with pytest.raises(ValueError, match="seed"):
        run_job(family, make_job(tmp, "generate.loop", "foundation-1", {"bars": 4, "bpm": 120}, seed=-5))
    assert not list((tmp / "out").glob("*.wav"))  # nothing was written for rejected jobs


# ---- generate.sample ------------------------------------------------------------------------------

def test_sample_uses_the_oneshot_grammar_and_trims_the_tail(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, sat, tmp = env
    family.load("foundation-1.2-samples")
    job = make_job(tmp, "generate.sample", "foundation-1.2-samples", {"note": 60, "length_s": 1.0, "tags": {"family": "Mallet", "subfamily": "Marimba"}})
    outputs, _ = run_job(family, job)
    call = sat.calls[0]
    assert call["conditioning"][0]["prompt"] == "One Shot, Mallet, Marimba, Dry, Medium Register, C4"
    assert call["conditioning"][0]["seconds_total"] == 2.0  # the one-shot window the model was trained on
    audio, sr = wav.read(outputs[0]["path"])
    assert 0.5 * sr < audio.shape[1] < 0.7 * sr  # the fake tone is silent after 0.5 s: tail trimmed, never longer than length_s
    assert outputs[0]["extra"]["note"] == 60
    long_job = make_job(tmp / "b", "generate.sample", "foundation-1.2-samples", {"note": 72, "length_s": 5.0})
    run_job(family, long_job)
    assert sat.calls[1]["conditioning"][0]["seconds_total"] == 5.0
    with pytest.raises(ValueError, match="one-shot range"):
        run_job(family, make_job(tmp / "c", "generate.sample", "foundation-1.2-samples", {"note": 3}))


# ---- transform.audio ------------------------------------------------------------------------------

def test_transform_passes_init_audio_and_noise_level(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, sat, tmp = env
    family.load("foundation-1")
    source = make_wav(tmp / "in" / "a.wav", seconds=2.0, sr=22050, freq=330)
    job = make_job(tmp, "transform.audio", "foundation-1", {"strength": 0.5, "tags": {"family": "Keys"}}, {"audio": str(source)})
    outputs, _ = run_job(family, job)
    call = sat.calls[0]
    sr_in, tensor = call["init_audio"]
    assert sr_in == 22050 and tensor.shape == (2, 44100) and tensor.dtype == "float32"
    assert call["init_noise_level"] == pytest.approx(0.1 * 50**0.5)
    assert call["conditioning"][0]["seconds_total"] == 2.0 and call["conditioning"][0]["prompt"] == "Keys, Dry"
    audio, sr = wav.read(outputs[0]["path"])
    assert sr == SR and audio.shape[1] == 2 * SR
    assert outputs[0]["extra"]["init_noise_level"] == pytest.approx(0.7071, abs=1e-3)


def test_transform_with_rectified_flow_and_bad_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sat = install_sat(monkeypatch)
    options = foundation_options(tmp_path, models=("foundation-1",))
    cfg = Path(options["checkpoints"]["foundation-1"]["config"])
    cfg.write_text(json.dumps({"sample_rate": SR, "sample_size": SR * 10, "diffusion_objective": "rectified_flow"}))
    family = fnd.FoundationFamily()
    family.configure(options)
    family.load("foundation-1")
    source = make_wav(tmp_path / "a.wav", seconds=1.0)
    run_job(family, make_job(tmp_path, "transform.audio", "foundation-1", {"strength": 0.25}, {"audio": str(source)}))
    assert sat.calls[0]["init_noise_level"] == 0.25
    with pytest.raises(ValueError, match="missing audio input"):
        run_job(family, make_job(tmp_path, "transform.audio", "foundation-1", {"strength": 0.5}))
    long = make_wav(tmp_path / "long.wav", seconds=11.0, sr=8000)
    with pytest.raises(ValueError, match="accepts at most 10 s"):
        run_job(family, make_job(tmp_path, "transform.audio", "foundation-1", {"strength": 0.5}, {"audio": str(long)}))
    with pytest.raises(ValueError, match="strength"):
        run_job(family, make_job(tmp_path, "transform.audio", "foundation-1", {"strength": 1.5}, {"audio": str(source)}))


# ---- generate.instrument (keybed) ------------------------------------------------------------------

INSTRUMENT = {
    "note_low": 48, "note_high": 59, "step": 3, "velocity_layers": 2, "length_s": 1.0, "name": "My Pad", "candidates": 2,
    "tags": {"family": "Synth", "subfamily": "Pad", "descriptors": ["warm", "wide"]},
}


def test_keybed_generates_one_clip_per_root_and_layer_with_shared_seeds(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, sat, tmp = env
    family.load("foundation-1.2-keybeds")
    job = make_job(tmp, "generate.instrument", "foundation-1.2-keybeds", dict(INSTRUMENT), seed=42)
    outputs, events = run_job(family, job)
    roots = [r for r, _, _ in plan_keybed(48, 59, 3)]
    assert roots == [49, 52, 55, 58]
    assert len(sat.calls) == len(roots) * 2 * 2  # roots x layers x candidates
    # shared seed within a (candidate, layer); different across layers and candidates
    by_seed: dict[int, list[int]] = {}
    for call in sat.calls:
        note = call["conditioning"][0]["prompt"].rsplit(", ", 1)[1]
        by_seed.setdefault(call["seed"], []).append(note)  # type: ignore[arg-type]
    assert sorted(by_seed) == sorted({42, fnd.layer_seed(42, 1), 43, fnd.layer_seed(43, 1)})
    assert all(notes == ["C#3", "E3", "G3", "A#3"] for notes in by_seed.values())
    assert sat.calls[0]["conditioning"][0]["prompt"] == "Keybed, Timbre Profile, Synth, Pad, Warm, Wide, Dry, Target Note, C#3"
    assert (sat.calls[0]["steps"], sat.calls[0]["cfg_scale"]) == (2, 6.0)  # the keybed defaults
    assert sat.calls[0]["conditioning"][0]["seconds_total"] == 1.0

    assert [o["kind"] for o in outputs] == ["instrument", "instrument"] and [o["candidate"] for o in outputs] == [0, 1]
    first, second = outputs
    assert Path(first["path"]) == (tmp / "out" / "My_Pad").resolve() and Path(second["path"]) == (tmp / "out" / "My_Pad_1").resolve()
    zones = first["extra"]["zones"]
    assert len(zones) == 8
    assert first["extra"]["name"] == "My_Pad" and first["extra"]["keybed_seed"] == 42 and second["extra"]["keybed_seed"] == 43
    assert zones[0] == {
        "sample": "samples/My_Pad_049_v1.wav", "root_note": 49, "key_low": 48, "key_high": 50,
        "vel_low": 0, "vel_high": 63, "gain_db": -6.0,
    }
    assert zones[-1] == {
        "sample": "samples/My_Pad_058_v2.wav", "root_note": 58, "key_low": 57, "key_high": 59, "vel_low": 64, "vel_high": 127,
    }
    # keys covered exactly once per layer
    for layer_zones in (zones[:4], zones[4:]):
        covered = [k for z in layer_zones for k in range(z["key_low"], z["key_high"] + 1)]
        assert covered == list(range(48, 60))
    files = listing(tmp / "out")
    assert files.count("My_Pad/samples/My_Pad_049_v1.wav") == 1 and sum(f.startswith("My_Pad_1/samples/") for f in files) == 8
    assert len(files) == 16
    audio, sr = wav.read(Path(first["path"]) / zones[0]["sample"])
    assert sr == SR and 0.5 * sr < audio.shape[1] <= sr  # tail-trimmed, at most length_s
    assert events[-1][0] == 1.0 and any("zone 16/16" in m for _, m in events)
    validate_outputs(outputs, Path(job["out_dir"]))


def test_keybed_output_is_what_postprocessing_expects(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, _, tmp = env
    family.load("foundation-1.2-keybeds")
    params = {**INSTRUMENT, "candidates": 1}
    job = make_job(tmp, "generate.instrument", "foundation-1.2-keybeds", params)
    outputs, _ = run_job(family, job)
    validated = validate_outputs(outputs, Path(job["out_dir"]))
    ctx = PostContext(
        "j_test", "generate.instrument", ModelMeta("foundation-1.2-keybeds", "foundation"),
        validate_job("generate.instrument", params).params, seed=42,
    )
    processed = process_output(dict(validated[0]), ctx, tmp / "post")
    assert processed.kind == "instrument"
    with zipfile.ZipFile(processed.path) as archive:
        names = sorted(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        sfz = archive.read("My_Pad.sfz").decode()
    assert "My_Pad.dspreset" in names and sfz.count("<region>") == 8
    assert sum(n.startswith("samples/") and n.endswith(".wav") for n in names) == 8
    assert [z["root_note"] for z in manifest["zones"]][:4] == [49, 52, 55, 58]
    assert math.isfinite(processed.sidecar["measured"]["peak_dbfs"])


def test_keybed_validation_and_confinement(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, sat, tmp = env
    family.load("foundation-1.2-keybeds")
    with pytest.raises(ValueError, match="keybed range"):  # note_low 0 step 12 puts a root at 5, below C0
        run_job(family, make_job(tmp, "generate.instrument", "foundation-1.2-keybeds", {"note_low": 0, "note_high": 11, "step": 12, "name": "x"}))
    assert sat.calls == []
    # a hostile name stays inside out_dir
    hostile = make_job(tmp / "h", "generate.instrument", "foundation-1.2-keybeds", {"note_low": 60, "note_high": 62, "step": 3, "length_s": 1.0, "name": "../../escape"})
    outputs, _ = run_job(family, hostile)
    assert inside(hostile["out_dir"], outputs[0]["path"]) and not (tmp / "escape").exists()
    # a pre-planted link in out_dir must not let the worker write outside it
    elsewhere = tmp / "elsewhere"
    elsewhere.mkdir()
    planted = make_job(tmp / "p", "generate.instrument", "foundation-1.2-keybeds", {"note_low": 60, "note_high": 62, "step": 3, "name": "pad"})
    (Path(planted["out_dir"]) / "pad").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="escapes"):
        run_job(family, planted)
    assert listing(elsewhere) == []


def test_all_outputs_stay_in_out_dir(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, _, tmp = env
    before = set(listing(tmp))
    for model, jtype, params in (
        ("foundation-1", "generate.loop", {**LOOP, "candidates": 2}),
        ("foundation-1.2-samples", "generate.sample", {"note": 48}),
        ("foundation-1.2-keybeds", "generate.instrument", {"note_low": 60, "note_high": 63, "step": 2, "name": "k", "length_s": 1.0}),
    ):
        family.load(model)
        job = make_job(tmp / jtype, jtype, model, params)
        run_job(family, job)
        new = set(listing(tmp)) - before
        assert all(f.startswith(f"{jtype}/out/") for f in new), new
        before |= new


# ---- cancellation ---------------------------------------------------------------------------------

def test_cancel_inside_the_sampler_callback(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, sat, tmp = env
    family.load("foundation-1")
    steps_seen: list[int] = []

    def cancelled() -> bool:
        steps_seen.append(1)
        return len(steps_seen) > 3  # let the pre-flight check and two sampler steps pass

    job = make_job(tmp, "generate.loop", "foundation-1", dict(LOOP))
    with pytest.raises(Cancelled):
        run_job(family, job, cancelled)
    assert len(sat.calls) == 1 and not list(Path(job["out_dir"]).glob("*.wav"))


def test_cancel_between_candidates(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, sat, tmp = env
    family.load("foundation-1")
    job = make_job(tmp, "generate.loop", "foundation-1", {**LOOP, "candidates": 3})
    first_done = Path(job["out_dir"]) / "candidate_0.wav"
    with pytest.raises(Cancelled):
        run_job(family, job, first_done.exists)
    assert len(sat.calls) == 1  # the second candidate never started
    assert [p.name for p in Path(job["out_dir"]).glob("*.wav")] == ["candidate_0.wav"]


def test_cancel_inside_keybed_zone_loop(env: tuple[fnd.FoundationFamily, FakeSat, Path]) -> None:
    family, sat, tmp = env
    family.load("foundation-1.2-keybeds")
    events: list[tuple[float, str]] = []
    job = make_job(tmp, "generate.instrument", "foundation-1.2-keybeds", {"note_low": 48, "note_high": 59, "step": 3, "name": "k", "length_s": 1.0})
    with pytest.raises(Cancelled):
        family.run(job, lambda f, m="": events.append((f, m)), lambda: any("zone 2/" in m for _, m in events))
    assert len(sat.calls) == 2


# ---- the protocol end to end ----------------------------------------------------------------------

def test_protocol_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sat = install_sat(monkeypatch)
    options = foundation_options(tmp_path)
    with Harness(fnd.FoundationFamily()) as h:
        _, hello = h.call("hello", options=options)
        assert hello["ok"] and hello["result"]["protocol"] == 1 and hello["result"]["family"] == "foundation"
        models = {m["id"]: validate_model_info(m, "foundation") for m in hello["result"]["models"]}
        assert all(m["available"] for m in models.values())
        _, reply = h.call("load", model="foundation-1")
        assert reply == {"id": reply["id"], "ok": True}
        job = make_job(tmp_path, "generate.loop", "foundation-1", {**LOOP, "candidates": 2})
        events, final = h.call("run", job=job)
        assert final["ok"] and len(final["result"]["outputs"]) == 2
        assert events and all(e["event"] == "progress" and 0 <= e["progress"] <= 1 for e in events)
        validate_outputs(final["result"]["outputs"], Path(job["out_dir"]))
        _, unload = h.call("unload", model="foundation-1")
        assert unload["ok"]
        _, again = h.call("run", job=job)  # the model is gone: a clean job failure, not a dead worker
        assert again["ok"] is False and "not loaded" in again["error"]
        _, ping = h.call("hello", options=options)
        assert ping["ok"]
        h.send("shutdown")
    assert len(sat.calls) == 2


def test_protocol_cancel_reaches_the_sampler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sat = install_sat(monkeypatch)
    sat.step_delay = 0.02
    options = foundation_options(tmp_path, steps=200)
    with Harness(fnd.FoundationFamily()) as h:
        h.call("hello", options=options)
        h.call("load", model="foundation-1")
        job = make_job(tmp_path, "generate.loop", "foundation-1", dict(LOOP))
        run_id = h.send("run", job=job)
        first = h.recv()
        assert first["id"] == run_id and first["event"] == "progress"
        cancel_id = h.send("cancel", job_id=job["job_id"])
        seen: dict[int, dict[str, Any]] = {}
        while run_id not in seen or cancel_id not in seen:
            message = h.recv()
            if message.get("event") != "progress":
                seen[message["id"]] = message
        assert seen[cancel_id]["ok"] is True
        assert seen[run_id] == {"id": run_id, "ok": False, "error": "cancelled"}
        _, after = h.call("hello", options=options)  # the worker survived and is idle
        assert after["ok"]
    assert len(sat.calls) == 1
