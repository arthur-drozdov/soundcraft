"""Schema and semantic validation of profiles."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
from jsonschema import Draft202012Validator

from atlas import yamlio
from atlas.validate import Issue, ProfileError, check_profile, load_schema, validate_file, validate_profile
from atlas_core_samples import sample_profile


def messages(profile: Any) -> str:
    return "\n".join(str(i) for i in validate_profile(profile))


def mutate(fn: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    p = sample_profile()
    fn(p)
    return p


def test_schema_is_valid_draft_2020_12_and_packaged() -> None:
    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    on_disk = json.loads((Path(__import__("atlas").__file__).parent / "schema" / "profile.schema.json").read_text())
    assert on_disk == schema


def test_good_profile_is_valid() -> None:
    assert validate_profile(sample_profile()) == []
    check_profile(sample_profile())


def test_minimal_profile_is_valid() -> None:
    p = sample_profile()
    p = {"identity": p["identity"], "params": []}
    assert validate_profile(p) == []


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        (lambda p: p.pop("identity"), "'identity' is a required property"),
        (lambda p: p.pop("params"), "'params' is a required property"),
        (lambda p: p.update(surprise=1), "unknown field"),
        (lambda p: p["identity"].update(format="vst2"), "identity.format"),
        (lambda p: p["identity"].update(os=["beos"]), "identity.os"),
        (lambda p: p["identity"].update(os=[]), "identity.os"),
        (lambda p: p["identity"].pop("version"), "'version' is a required property"),
        (lambda p: p["identity"].update(profiled="22/10/2026"), "identity.profiled"),
        (lambda p: p["identity"].update(profiled="2026-02-30"), "not a real date"),
        (lambda p: p["identity"].update(plugin_id="ABCDEF"), "must start with `vst3:`"),
        (lambda p: p["params"][0].pop("id"), "'id' is a required property"),
        (lambda p: p["params"][0].update(bogus=1), "params[0]: unknown field"),
        (lambda p: p["params"][0].update(min="low"), "params[0].min"),
        (lambda p: p["params"][0].update(automatable="yes"), "params[0].automatable"),
        (lambda p: p["params"][1].update(calibration=[{"value": 0.1}]), "'text' is a required property"),
        (lambda p: p["params"][0].update(does="x" * 4001), "params[0].does"),
        (lambda p: p["params"].append(copy.deepcopy(p["params"][0])), "duplicate param id `thresh`"),
        (lambda p: p["params"][2].update(choices=["a", "a"]), "duplicate choices"),
        (lambda p: p["params"][0].update(min=5.0), "min 5 is above max 0"),
        (lambda p: p["params"][0].update(default=10.0), "default 10 is outside"),
        (lambda p: p["params"][0].update(safe_range=[-90.0, -10.0], provenance={**p["params"][0]["provenance"], "safe_range": {"source": "manual", "confidence": 1.0}}), "safe_range lies outside"),
        (lambda p: p["params"][0].update(safe_range=[1.0], provenance={**p["params"][0]["provenance"], "safe_range": {"source": "manual", "confidence": 1.0}}), "params[0].safe_range"),
        (lambda p: p["params"][0].update(min=float("nan")), "numbers must be finite"),
        (lambda p: p["params"][0]["provenance"].update(ghost={"source": "host", "confidence": 1.0}), "params[0].provenance"),
        (lambda p: p["provenance"].update(nonsense={"source": "host", "confidence": 1}), "provenance"),
        (lambda p: p.update(modes=[{"description": "no name"}]), "'name' is a required property"),
        (lambda p: p.update(ui_only=[{"feature": "Sidechain"}]), "'procedure' is a required property"),
    ],
)
def test_bad_profiles_are_rejected(change: Callable[[dict[str, Any]], None], needle: str) -> None:
    assert needle in messages(mutate(change))


def test_not_a_mapping() -> None:
    assert validate_profile([1, 2]) == [Issue("", "a profile must be a mapping")]


def test_knowledge_field_without_provenance_is_rejected() -> None:
    p = sample_profile()
    p["params"][0]["does"] = "Sets the level above which gain reduction starts."
    assert "`does` is set but has no provenance" in messages(p)
    p["params"][0]["provenance"]["does"] = {"source": "manual", "ref": "p.4", "confidence": 1.0}
    assert validate_profile(p) == []


@pytest.mark.parametrize("confidence", [0.9, 0.95, 1.0])
def test_inferred_provenance_needs_confidence_below_0_9(confidence: float) -> None:
    p = sample_profile()
    p["params"][0]["provenance"]["taper"] = {"source": "inferred", "confidence": confidence}
    out = messages(p)
    assert "params[0].provenance.taper.confidence" in out
    assert "inferred provenance must have confidence < 0.9" in out


@pytest.mark.parametrize("confidence", [0.0, 0.5, 0.89])
def test_inferred_below_limit_is_fine(confidence: float) -> None:
    p = sample_profile()
    p["params"][0]["provenance"]["taper"] = {"source": "inferred", "confidence": confidence}
    assert validate_profile(p) == []


def test_firm_sources_may_be_fully_confident() -> None:
    p = sample_profile()
    for source in ("host", "manual", "source", "measured"):
        p["params"][0]["provenance"]["taper"] = {"source": source, "confidence": 1.0, "ref": "x"}
        assert validate_profile(p) == []


def test_inferred_rule_also_applies_to_top_level_provenance() -> None:
    p = sample_profile()
    p["provenance"]["summary"] = {"source": "inferred", "confidence": 0.99}
    assert "inferred provenance must have confidence < 0.9" in messages(p)


@pytest.mark.parametrize(
    "prov",
    [
        {"source": "guess", "confidence": 0.5},
        {"source": "host"},
        {"confidence": 0.5},
        {"source": "host", "confidence": 1.5},
        {"source": "host", "confidence": -0.1},
        {"source": "host", "confidence": 1.0, "extra": 1},
        {"source": "host", "confidence": 1.0, "ref": ""},
    ],
)
def test_malformed_provenance_is_rejected(prov: dict[str, Any]) -> None:
    p = sample_profile()
    p["params"][0]["provenance"]["taper"] = prov
    assert validate_profile(p) != []


def test_check_profile_raises_with_issues() -> None:
    with pytest.raises(ProfileError) as info:
        check_profile({"identity": {}})
    assert info.value.issues


def test_validate_file_reports_unreadable_and_hostile_files(tmp_path: Path) -> None:
    assert "No such file" in str(validate_file(tmp_path / "nope.yaml")[0])
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed\n")
    assert validate_file(bad)
    bomb = tmp_path / "bomb.yaml"
    bomb.write_text("a: &a [1,2,3]\nb: &b [*a,*a,*a]\nc: [*b,*b,*b]\n")
    assert "aliases are not allowed" in str(validate_file(bomb)[0])
    big = tmp_path / "big.yaml"
    big.write_bytes(b"#" * (yamlio.MAX_YAML_BYTES + 1))
    assert "larger than" in str(validate_file(big)[0])
    notutf = tmp_path / "bin.yaml"
    notutf.write_bytes(b"\xff\xfe\x00bad")
    assert validate_file(notutf)


def test_yaml_pitfalls_do_not_corrupt_values() -> None:
    data = yamlio.loads("choices: [Off, On, yes]\ntext: 4:1\nwhen: 2026-10-22\nx: 1e3\n")
    assert data == {"choices": ["Off", "On", "yes"], "text": "4:1", "when": "2026-10-22", "x": 1000.0}
