"""version_check, `atlas check --json`, and hostile-input edges of merge/validate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from atlas import cli, profiles
from atlas.merge import MergeError, merge_patch
from atlas.validate import validate_profile

HOST = {"source": "host", "confidence": 1.0}


def _profile(version: str = "2.1.0") -> dict[str, Any]:
    return {
        "identity": {
            "name": "Tiny EQ", "vendor": "Acme", "format": "vst3", "plugin_id": "vst3:ABCD1234",
            "version": version, "os": ["linux"], "profiled": "2026-10-22",
        },
        "params": [{"id": 0, "host_name": "Cutoff", "min": 0.0, "max": 1.0, "provenance": {"min": HOST, "max": HOST}}],
    }


@pytest.mark.parametrize(
    "actual, matches, status",
    [("2.1.0", True, "exact"), ("v2.1.0", True, "exact"), ("2.3.0", False, "same-major"), ("3.0.0", False, "different"), ("", False, "different")],
)
def test_version_check_shape(actual: str, matches: bool, status: str) -> None:
    result = profiles.version_check(_profile(), actual)
    assert result["matches"] is matches and result["status"] == status
    assert result["profile_version"] == "2.1.0" and result["actual_version"] == actual


def test_cli_check_json_and_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "profiles"
    path = profiles.save_profile(root, _profile())
    assert path.relative_to(root).as_posix() == "vst3/acme/tiny-eq@2.yaml"
    base = ["check", "--root", str(root), "--plugin", "vst3:ABCD1234"]
    assert cli.main([*base, "--version", "2.1.0", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["matches"] is True and out["profile_version"] == "2.1.0"
    assert cli.main([*base, "--version", "2.9", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["matches"] is False
    assert cli.main(["check", "--root", str(root), "--plugin", "vst3:NOPE", "--version", "1", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "missing"


def test_cli_get_json_carries_version_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "profiles"
    profiles.save_profile(root, _profile())
    assert cli.main(["get", "--root", str(root), "vst3:ABCD1234", "--version", "2.2.0", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["version"]["matches"] is False and data["version"]["status"] == "same-major"
    assert cli.main(["get", "--root", str(root), "vst3:ABCD1234", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] is None


def test_nan_and_inf_are_rejected_everywhere() -> None:
    bad = _profile()
    bad["params"][0]["provenance"]["min"] = {"source": "measured", "confidence": float("nan")}
    assert validate_profile(bad)
    bad = _profile()
    bad["params"][0]["max"] = float("inf")
    assert validate_profile(bad)


def test_merge_does_not_mutate_input_and_rejects_unknown_param() -> None:
    profile = _profile()
    snapshot = json.dumps(profile, sort_keys=True)
    patch = {"default_provenance": {"source": "manual", "confidence": 0.9}, "params": [{"id": "nope", "does": "x"}]}
    with pytest.raises(MergeError, match="unknown param"):
        merge_patch(profile, patch)
    ok = {"default_provenance": {"source": "manual", "confidence": 0.9}, "params": [{"id": 0, "does": "Sets the corner."}]}
    merged = merge_patch(profile, ok)
    assert json.dumps(profile, sort_keys=True) == snapshot
    assert merged.changes[0].path == "params[0].does" and merged.changes[0].old is None
