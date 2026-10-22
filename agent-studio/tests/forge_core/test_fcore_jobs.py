from __future__ import annotations

import math
from typing import Any

import pytest

from forge.errors import ValidationFailed
from forge.jobs import JOB_TYPES, OUTPUT_KINDS, PARAM_MODELS, validate_job

AID = "a_" + "0" * 24


def paths(exc: pytest.ExceptionInfo[ValidationFailed]) -> set[str]:
    return {e["path"] for e in exc.value.errors}


def test_every_job_type_has_a_model_and_output_kind() -> None:
    assert set(JOB_TYPES) == set(PARAM_MODELS) == set(OUTPUT_KINDS)


def test_common_defaults() -> None:
    p = validate_job("generate.sfx").params
    assert (p.prompt, p.tags, p.seed, p.candidates, p.sample_rate, p.channels) == ("", {}, None, 1, 48000, 2)
    assert p.post.model_dump() == {
        "trim_silence": False, "fade_in_ms": 0, "fade_out_ms": 0, "normalize": "peak", "target": -1.0,
        "loop_crossfade_ms": 0, "bit_depth": "24",
    }
    assert p.duration_s == 4  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("job_type", "params", "expected"),
    [
        ("generate.loop", {"bpm": 120}, {"bars": 4, "key": None}),
        ("generate.sample", {}, {"note": 60, "length_s": 2}),
        ("generate.instrument", {"name": "Pad"}, {"note_low": 36, "note_high": 84, "step": 3, "velocity_layers": 1, "length_s": 3}),
        ("transform.audio", {}, {"strength": 0.5}),
        ("extend.audio", {}, {"seconds": 8}),
        ("stems.split", {}, {"stems": []}),
        ("transcribe.melody", {}, {"min_note_ms": 60, "bpm": 120}),
        ("render.session", {}, {"commands": [], "start": 0}),
    ],
)
def test_specific_defaults(job_type: str, params: dict[str, Any], expected: dict[str, Any]) -> None:
    inputs = {"audio": AID} if job_type in ("transform.audio", "extend.audio", "stems.split", "transcribe.melody") else None
    if job_type == "render.session":
        inputs = {"session": AID}
    got = validate_job(job_type, params, inputs).params
    assert {k: getattr(got, k) for k in expected} == expected


@pytest.mark.parametrize(
    ("job_type", "params", "bad_path"),
    [
        ("generate.loop", {"bpm": 120, "bars": 17}, "params.bars"),
        ("generate.loop", {"bpm": 39}, "params.bpm"),
        ("generate.loop", {}, "params.bpm"),
        ("generate.loop", {"bpm": 120, "key": "H major"}, "params.key"),
        ("generate.loop", {"bpm": 120, "extra": 1}, "params.extra"),
        ("generate.loop", {"bpm": math.nan}, "params.bpm"),
        ("generate.loop", {"bpm": math.inf}, "params.bpm"),
        ("generate.loop", {"bpm": 120, "candidates": 9}, "params.candidates"),
        ("generate.loop", {"bpm": 120, "sample_rate": 8000}, "params.sample_rate"),
        ("generate.loop", {"bpm": 120, "channels": 3}, "params.channels"),
        ("generate.loop", {"bpm": 120, "prompt": "x" * 2001}, "params.prompt"),
        ("generate.loop", {"bpm": 120, "seed": -1}, "params.seed"),
        ("generate.loop", {"bpm": 120, "post": {"normalize": "loud"}}, "params.post.normalize"),
        ("generate.loop", {"bpm": 120, "post": {"fade_out_ms": 5001}}, "params.post.fade_out_ms"),
        ("generate.loop", {"bpm": 120, "post": {"bit_depth": "8"}}, "params.post.bit_depth"),
        ("generate.loop", {"bpm": 120, "post": {"loop_crossfade_ms": 501}}, "params.post.loop_crossfade_ms"),
        ("generate.loop", {"bpm": 120, "tags": {"a": 3}}, "params.tags.a"),
        ("generate.sample", {"note": 128}, "params.note"),
        ("generate.sample", {"length_s": 0.01}, "params.length_s"),
        ("generate.instrument", {}, "params.name"),
        ("generate.instrument", {"name": "../evil"}, "params.name"),
        ("generate.instrument", {"name": "x", "step": 13}, "params.step"),
        ("generate.instrument", {"name": "x", "velocity_layers": 4}, "params.velocity_layers"),
        ("generate.sfx", {"duration_s": 48}, "params.duration_s"),
        ("generate.loop", "not an object", "params"),
    ],
)
def test_invalid_params_report_field_paths(job_type: str, params: Any, bad_path: str) -> None:
    with pytest.raises(ValidationFailed) as exc:
        validate_job(job_type, params)
    assert bad_path in paths(exc)


def test_cross_field_rules() -> None:
    with pytest.raises(ValidationFailed):
        validate_job("generate.instrument", {"name": "x", "note_low": 80, "note_high": 40})
    with pytest.raises(ValidationFailed, match="invalid request"):
        validate_job("generate.instrument", {"name": "x", "note_low": 0, "note_high": 127, "step": 1, "velocity_layers": 3})
    with pytest.raises(ValidationFailed):
        validate_job("inpaint.audio", {"start_s": 3, "end_s": 2}, {"audio": AID})
    with pytest.raises(ValidationFailed):
        validate_job("generate.sfx", {"post": {"normalize": "lufs", "target": -1.0 + 5}})
    assert validate_job("generate.sfx", {"post": {"normalize": "lufs", "target": -14}}).params.post.target == -14


def test_all_errors_are_collected() -> None:
    with pytest.raises(ValidationFailed) as exc:
        validate_job("generate.loop", {"bpm": 1000, "bars": 0, "nope": True})
    assert {"params.bpm", "params.bars", "params.nope"} <= paths(exc)
    assert exc.value.status == 422 and exc.value.to_dict()["error"]["code"] == "validation_error"


def test_unknown_type() -> None:
    with pytest.raises(ValidationFailed) as exc:
        validate_job("generate.symphony", {})
    assert paths(exc) == {"type"}
    with pytest.raises(ValidationFailed):
        validate_job(None, {})  # type: ignore[arg-type]


def test_inputs_are_checked() -> None:
    assert validate_job("transform.audio", {}, {"audio": AID}).inputs == {"audio": AID}
    for inputs, bad in [(None, "inputs.audio"), ({}, "inputs.audio"), ({"audio": "nope"}, "inputs.audio"),
                        ({"audio": AID, "other": AID}, "inputs.other"), ({"audio": 5}, "inputs.audio"),
                        ({"audio": "a_" + "G" * 24}, "inputs.audio")]:
        with pytest.raises(ValidationFailed) as exc:
            validate_job("transform.audio", {}, inputs)
        assert bad in paths(exc)
    with pytest.raises(ValidationFailed) as exc:
        validate_job("generate.loop", {"bpm": 100}, {"audio": AID})
    assert "inputs.audio" in paths(exc)
    with pytest.raises(ValidationFailed):
        validate_job("transform.audio", {}, ["x"])  # type: ignore[arg-type]
    assert validate_job("render.session", {"commands": [{"command": "a.b", "params": {"x": 1}}]}, {"session": AID}).kind == "audio"


def test_render_commands_are_capped() -> None:
    cmds = [{"command": "x"}] * 1001
    with pytest.raises(ValidationFailed) as exc:
        validate_job("render.session", {"commands": cmds}, {"session": AID})
    assert "params.commands" in paths(exc)


def test_tags_accept_strings_and_lists() -> None:
    p = validate_job("generate.sfx", {"tags": {"family": "synth", "descriptors": ["warm", "wide"]}}).params
    assert p.tags == {"family": "synth", "descriptors": ["warm", "wide"]}
    with pytest.raises(ValidationFailed):
        validate_job("generate.sfx", {"tags": {f"t{i}": "x" for i in range(65)}})
