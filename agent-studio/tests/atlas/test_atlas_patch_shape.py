"""Patch shape checks (atlas.patch) and an end-to-end load -> merge -> CLI flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from atlas import cli, profiles, yamlio
from atlas.merge import merge_patch
from atlas.patch import check_patch, load_patch
from atlas.validate import ProfileError, validate_profile

HOST = {"source": "host", "confidence": 1.0}


def _profile() -> dict[str, Any]:
    return {
        "identity": {
            "name": "Tiny EQ", "vendor": "Acme", "format": "clap", "plugin_id": "clap:acme.tiny-eq",
            "version": "1.0.0", "os": ["linux"], "profiled": "2026-10-22",
        },
        "params": [
            {"id": "cutoff", "host_name": "Cutoff", "min": 20.0, "max": 20000.0, "provenance": {"min": HOST, "max": HOST}},
        ],
    }


def test_well_formed_patch_has_no_issues() -> None:
    patch = {
        "default_provenance": {"source": "manual", "ref": "p.1", "confidence": 0.9},
        "summary": "x",
        "provenance": {"summary": {"confidence": 1.0}},
        "params": [{"id": "cutoff", "does": "d", "provenance": {"does": {"ref": "p.2"}}}],
    }
    assert check_patch(patch) == []


@pytest.mark.parametrize(
    "patch, fragment",
    [
        ([], "mapping"),
        ({"identity": {}}, "unknown patch key"),
        ({"default_provenance": "manual"}, "must be a mapping"),
        ({"default_provenance": {"src": "x"}}, "unknown provenance key"),
        ({"provenance": {"summary": {}}}, "does not set"),
        ({"params": "cutoff"}, "must be a list"),
        ({"params": [{"does": "no id"}]}, "`id`"),
        ({"params": [{"id": True}]}, "string or integer"),
        ({"params": [{"id": "a"}, {"id": "a"}]}, "twice"),
        ({"params": [{"id": "a", "host_name": "x"}]}, "cannot be set"),
        ({"params": [{"id": "a", "provenance": {"does": {}}}]}, "does not set"),
        ({"params": [{"id": "a", "provenance": []}]}, "must be a mapping"),
    ],
)
def test_malformed_patches_are_reported(patch: Any, fragment: str) -> None:
    messages = " | ".join(str(i) for i in check_patch(patch))
    assert fragment in messages


def test_load_patch_rejects_unreadable_and_malformed(tmp_path: Path) -> None:
    with pytest.raises(ProfileError):
        load_patch(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("[unclosed", encoding="utf-8")
    with pytest.raises(ProfileError):
        load_patch(bad)
    alias = tmp_path / "alias.yaml"
    alias.write_text("a: &x [1]\nb: *x\n", encoding="utf-8")
    with pytest.raises(ProfileError):
        load_patch(alias)
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text("nonsense: 1\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="unknown patch key"):
        load_patch(wrong)


def test_yaml_one_point_one_booleans_stay_strings(tmp_path: Path) -> None:
    patch_file = tmp_path / "p.yaml"
    patch_file.write_text(
        "default_provenance: {source: manual, confidence: 0.9}\n"
        "params:\n  - id: cutoff\n    choices: [Off, On, 4:1]\n",
        encoding="utf-8",
    )
    patch = load_patch(patch_file)
    assert patch["params"][0]["choices"] == ["Off", "On", "4:1"]


def test_end_to_end_file_merge_then_cli_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "profiles"
    path = profiles.save_profile(root, _profile())
    assert path.relative_to(root).as_posix() == "clap/acme/tiny-eq@1.yaml"
    patch_file = tmp_path / "patch.yaml"
    patch_file.write_text(
        "default_provenance: {source: manual, ref: 'Manual p.4', confidence: 0.95}\n"
        "params:\n  - id: cutoff\n    does: Sets the corner frequency.\n    safe_range: [100, 8000]\n",
        encoding="utf-8",
    )
    assert cli.main(["merge", str(path), str(patch_file)]) == 0
    merged = profiles.load_profile(path)
    assert validate_profile(merged) == []
    prov = merged["params"][0]["provenance"]
    assert prov["does"] == {"source": "manual", "ref": "Manual p.4", "confidence": 0.95}
    assert prov["safe_range"]["source"] == "manual"
    assert prov["min"] == HOST  # untouched knowledge keeps its provenance
    assert cli.main(["validate", str(root)]) == 0
    assert "ok" in capsys.readouterr().out


def test_inferred_patch_cannot_override_host_knowledge() -> None:
    patch = {
        "default_provenance": {"source": "inferred", "confidence": 0.5},
        "params": [{"id": "cutoff", "min": 0.0}],
    }
    with pytest.raises(ProfileError):
        merge_patch(_profile(), patch)
    result = merge_patch(_profile(), patch, skip_conflicts=True)
    assert result.profile["params"][0]["min"] == 20.0
    assert [c.path for c in result.conflicts] == ["params[cutoff].min"]


def test_inferred_confidence_limit_is_enforced_on_merge() -> None:
    patch = {
        "default_provenance": {"source": "inferred", "confidence": 0.95},
        "summary": "guess",
    }
    with pytest.raises(ProfileError, match="confidence"):
        merge_patch(_profile(), patch)


def test_yamlio_is_importable_for_roundtrip() -> None:
    text = profiles.dump_profile(_profile())
    assert yamlio.loads(text) == profiles.canonical(_profile())
