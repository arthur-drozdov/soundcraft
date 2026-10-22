"""Merge semantics: keyed params, provenance recording, protection of firmer knowledge."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from atlas import profiles
from atlas.merge import MergeError, merge_files, merge_patch
from atlas.validate import validate_profile
from atlas_core_samples import sample_profile

MANUAL = {"source": "manual", "ref": "Manual p.12", "confidence": 0.95}


def test_patch_sets_fields_keyed_by_id_and_records_provenance() -> None:
    base = sample_profile()
    snapshot = copy.deepcopy(base)
    patch = {
        "default_provenance": MANUAL,
        "params": [
            {"id": "thresh", "does": "Level above which compression starts.", "role": "threshold", "safe_range": [-40, -6]},
            {"id": "ratio", "musical_use": "2:1 glues, 8:1 and up squashes.", "provenance": {"musical_use": {"ref": "Manual p.14", "confidence": 1.0}}},
        ],
    }
    result = merge_patch(base, patch)
    assert base == snapshot  # input untouched
    out = result.profile
    thresh = out["params"][0]
    assert thresh["does"] == "Level above which compression starts."
    assert thresh["provenance"]["does"] == MANUAL
    assert thresh["provenance"]["role"] == MANUAL
    assert thresh["provenance"]["safe_range"] == MANUAL
    assert thresh["provenance"]["unit"] == {"source": "host", "confidence": 1.0}  # untouched
    ratio = out["params"][1]
    assert ratio["provenance"]["musical_use"] == {"source": "manual", "ref": "Manual p.14", "confidence": 1.0}
    assert [c.path for c in result.changes] == ["params[thresh].role", "params[thresh].does", "params[thresh].safe_range", "params[ratio].musical_use"]  # canonical field order
    assert validate_profile(out) == []


def test_provenance_recorded_for_every_field_set_including_plain_ones() -> None:
    patch = {"default_provenance": MANUAL, "params": [{"id": "ratio", "name": "Ratio", "section": "Dynamics", "automatable": True}]}
    out = merge_patch(sample_profile(), patch).profile["params"][1]
    assert {"name", "section", "automatable"} <= set(out["provenance"])
    assert all(out["provenance"][f] == MANUAL for f in ("name", "section", "automatable"))


def test_numeric_ids_and_top_level_fields() -> None:
    patch = {
        "default_provenance": MANUAL,
        "summary": "A compressor with a Mode switch.",
        "modes": [{"name": "Auto", "description": "Program-dependent release."}],
        "known_issues": ["State recall drops Mode on VST3 in some hosts."],
        "provenance": {"summary": {"ref": "Manual p.2"}},
        "params": [{"id": 7, "does": "Chooses detector behaviour."}],
    }
    out = merge_patch(sample_profile(), patch).profile
    assert out["summary"] == "A compressor with a Mode switch."
    assert out["provenance"]["summary"] == {"source": "manual", "ref": "Manual p.2", "confidence": 0.95}
    assert out["provenance"]["modes"] == MANUAL
    assert out["provenance"]["identity.version"]["source"] == "host"  # kept
    assert out["params"][2]["does"] == "Chooses detector behaviour."


def test_merging_twice_is_idempotent() -> None:
    patch = {"default_provenance": MANUAL, "params": [{"id": "thresh", "does": "x"}]}
    once = merge_patch(sample_profile(), patch).profile
    assert merge_patch(once, patch).profile == once


@pytest.mark.parametrize(
    ("patch", "needle"),
    [
        ([], "must be a mapping"),
        ({"bogus": 1}, "bogus: unknown patch key"),
        ({"default_provenance": MANUAL, "params": [{"id": "nope", "does": "x"}]}, "unknown param id `nope`"),
        ({"default_provenance": MANUAL, "params": [{"does": "x"}]}, "mapping with an `id`"),
        ({"default_provenance": MANUAL, "params": [{"id": "ratio", "host_name": "Hax"}]}, "params[0].host_name: cannot be set"),
        ({"default_provenance": MANUAL, "params": [{"id": "ratio", "wat": 1}]}, "params[0].wat: cannot be set"),
        ({"default_provenance": MANUAL, "params": [{"id": "ratio", "does": "a"}, {"id": "ratio", "does": "b"}]}, "appears twice"),
        ({"params": [{"id": "ratio", "does": "x"}]}, "no provenance"),
        ({"default_provenance": {"source": "manual"}, "params": [{"id": "ratio", "does": "x"}]}, "no provenance"),
        ({"default_provenance": {**MANUAL, "colour": "red"}, "summary": "x"}, "unknown provenance key"),
        ({"default_provenance": MANUAL, "params": [{"id": "ratio", "does": "x", "provenance": {"musical_use": {"ref": "y"}}}]}, "provenance for a field the patch does not set"),
        ({"default_provenance": MANUAL, "params": "ratio"}, "params: must be a list"),
        ({"default_provenance": MANUAL, "identity": {"version": "9"}}, "identity: unknown patch key"),
    ],
)
def test_bad_patches_are_refused(patch: Any, needle: str) -> None:
    with pytest.raises(MergeError) as info:
        merge_patch(sample_profile(), patch)
    assert needle in str(info.value)


def test_patch_that_makes_the_profile_invalid_is_refused() -> None:
    patch = {"default_provenance": MANUAL, "params": [{"id": "thresh", "safe_range": [-90, -10]}]}
    with pytest.raises(MergeError, match="safe_range lies outside"):
        merge_patch(sample_profile(), patch)
    patch = {"default_provenance": MANUAL, "params": [{"id": "thresh", "min": float("nan")}]}
    with pytest.raises(MergeError):
        merge_patch(sample_profile(), patch)


def test_inferred_patch_needs_confidence_below_0_9() -> None:
    patch = {"default_provenance": {"source": "inferred", "confidence": 0.95}, "params": [{"id": "ratio", "does": "x"}]}
    with pytest.raises(MergeError, match="inferred provenance must have confidence < 0.9"):
        merge_patch(sample_profile(), patch)
    patch["default_provenance"]["confidence"] = 0.5  # type: ignore[index]
    assert merge_patch(sample_profile(), patch).profile["params"][1]["does"] == "x"


@pytest.mark.parametrize("existing_source", ["host", "source", "measured"])
def test_inferred_never_overwrites_firmer_sources(existing_source: str) -> None:
    base = sample_profile()
    base["params"][0]["does"] = "Measured level where gain reduction begins."
    base["params"][0]["provenance"]["does"] = {"source": existing_source, "confidence": 0.7}
    patch = {
        "default_provenance": {"source": "inferred", "confidence": 0.5},
        "params": [{"id": "thresh", "does": "Probably a threshold.", "role": "threshold"}],
    }
    with pytest.raises(MergeError) as info:
        merge_patch(base, patch)
    assert [c.path for c in info.value.conflicts] == ["params[thresh].does"]
    assert "refusing inferred" in str(info.value)

    skipped = merge_patch(base, patch, skip_conflicts=True)
    assert skipped.profile["params"][0]["does"] == "Measured level where gain reduction begins."
    assert skipped.profile["params"][0]["provenance"]["does"]["source"] == existing_source
    assert skipped.profile["params"][0]["role"] == "threshold"  # the non-conflicting field still applied
    assert [c.path for c in skipped.conflicts] == ["params[thresh].does"]


def test_protected_value_with_host_provenance_cannot_be_replaced_by_inferred_unit() -> None:
    patch = {"default_provenance": {"source": "inferred", "confidence": 0.8}, "params": [{"id": "thresh", "unit": "dBFS"}]}
    with pytest.raises(MergeError) as info:
        merge_patch(sample_profile(), patch)
    assert info.value.conflicts[0].existing == {"source": "host", "confidence": 1.0}


def test_conflicting_patch_is_atomic() -> None:
    patch = {
        "default_provenance": {"source": "inferred", "confidence": 0.5},
        "summary": "A fresh summary.",
        "params": [{"id": "thresh", "unit": "dBFS"}],
    }
    base = sample_profile()
    snapshot = copy.deepcopy(base)
    with pytest.raises(MergeError):
        merge_patch(base, patch)
    assert base == snapshot


def test_inferred_may_replace_weaker_or_unprotected_knowledge() -> None:
    base = sample_profile()
    # host taper was inferred 0.6 (unprotected); a manual-sourced value was set earlier too
    base["params"][1]["does"] = "old"
    base["params"][1]["provenance"]["does"] = {"source": "manual", "ref": "p.1", "confidence": 1.0}
    inferred = {"default_provenance": {"source": "inferred", "confidence": 0.4}, "params": [{"id": "thresh", "taper": "log"}, {"id": "ratio", "does": "new"}]}
    out = merge_patch(base, inferred).profile
    assert out["params"][0]["taper"] == "log"
    assert out["params"][1]["does"] == "new"
    # a measured value with LOWER confidence than the inferred one can be replaced
    base["params"][0]["provenance"]["taper"] = {"source": "measured", "confidence": 0.2}
    assert merge_patch(base, {"default_provenance": {"source": "inferred", "confidence": 0.4}, "params": [{"id": "thresh", "taper": "exp"}]}).profile["params"][0]["taper"] == "exp"


def test_manual_and_measured_may_replace_host_values() -> None:
    patch = {"default_provenance": {"source": "measured", "ref": "bounce 12", "confidence": 0.8}, "params": [{"id": "thresh", "unit": "dBFS"}]}
    out = merge_patch(sample_profile(), patch).profile
    assert out["params"][0]["unit"] == "dBFS"
    assert out["params"][0]["provenance"]["unit"]["source"] == "measured"


def test_merge_files_in_place_out_and_dry_run(tmp_path: Path) -> None:
    root = tmp_path / "atlas"
    path = profiles.save_profile(root, sample_profile())
    patch = tmp_path / "patch.yaml"
    patch.write_text(
        "default_provenance: {source: manual, ref: 'Manual p.12', confidence: 0.95}\n"
        "params:\n  - id: ratio\n    does: 'Sets compression ratio.'\n    calibration: [{value: 0.5, text: 4:1}]\n"
    )
    before = path.read_text()
    result, written = merge_files(path, patch, dry_run=True)
    assert written is None and len(result.changes) == 2 and path.read_text() == before

    out = tmp_path / "copy.yaml"
    _, written = merge_files(path, patch, out_path=out)
    assert written == out and path.read_text() == before
    assert profiles.load_profile(out)["params"][1]["calibration"] == [{"value": 0.5, "text": "4:1"}]

    _, written = merge_files(path, patch)
    assert written == path
    assert profiles.load_profile(path)["params"][1]["does"] == "Sets compression ratio."


def test_merge_files_failure_leaves_the_file_alone(tmp_path: Path) -> None:
    path = profiles.save_profile(tmp_path, sample_profile())
    before = path.read_text()
    patch = tmp_path / "patch.yaml"
    patch.write_text("default_provenance: {source: inferred, confidence: 0.5}\nparams:\n  - id: thresh\n    unit: dBFS\n")
    with pytest.raises(MergeError):
        merge_files(path, patch)
    assert path.read_text() == before
    patch.write_text("a: &x [1]\nb: *x\n")
    with pytest.raises(MergeError, match="unreadable patch"):
        merge_files(path, patch)
