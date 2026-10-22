"""The `atlas` command line: validate, merge, list, get, check, and the extension registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from atlas import cli, profiles
from atlas_core_samples import sample_profile

PATCH = (
    "default_provenance: {source: manual, ref: 'Manual p.12', confidence: 0.95}\n"
    "params:\n  - id: ratio\n    does: 'Sets compression ratio.'\n"
)


@pytest.fixture
def atlas_root(tmp_path: Path) -> Path:
    root = tmp_path / "profiles"
    profiles.save_profile(root, sample_profile())
    profiles.save_profile(root, sample_profile(format="clap", vendor="Zed", name="Zap", plugin_id="clap:zed.zap", version="1.0"))
    return root


def run(*argv: str) -> int:
    return cli.main([str(a) for a in argv])


def test_validate_ok_and_failure(atlas_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = atlas_root / "vst3" / "acme-audio" / "acme-squash@2.yaml"
    assert run("validate", good) == 0
    assert "acme-squash@2.yaml: ok" in capsys.readouterr().out

    bad = tmp_path / "bad.yaml"
    bad.write_text("identity: {name: x}\nparams: []\nwhat: 1\n")
    assert run("validate", good, bad) == 1
    captured = capsys.readouterr()
    assert "acme-squash@2.yaml: ok" in captured.out
    assert "bad.yaml: " in captured.err and "unknown field" in captured.err
    assert run("validate", tmp_path / "missing.yaml") == 1
    assert "no such file" in capsys.readouterr().err


def test_validate_folder_checks_layout(atlas_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("validate", atlas_root) == 0
    capsys.readouterr()
    wrong = atlas_root / "vst3" / "acme-audio" / "misnamed@2.yaml"
    (atlas_root / "vst3" / "acme-audio" / "acme-squash@2.yaml").rename(wrong)
    assert run("validate", atlas_root) == 1
    assert "expected vst3/acme-audio/acme-squash@2.yaml" in capsys.readouterr().err


def test_validate_empty_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("validate", tmp_path) == 0
    assert "no profiles found" in capsys.readouterr().err


def test_merge_writes_and_reports(atlas_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = atlas_root / "vst3" / "acme-audio" / "acme-squash@2.yaml"
    patch = tmp_path / "patch.yaml"
    patch.write_text(PATCH)
    assert run("merge", path, patch, "--dry-run") == 0
    assert "(dry run)" in capsys.readouterr().out
    assert "does" not in profiles.load_profile(path)["params"][1]

    assert run("merge", path, patch) == 0
    out = capsys.readouterr().out
    assert "set params[ratio].does (manual, 0.95)" in out and f"wrote {path}" in out
    ratio = profiles.load_profile(path)["params"][1]
    assert ratio["does"] == "Sets compression ratio."
    assert ratio["provenance"]["does"]["ref"] == "Manual p.12"


def test_merge_conflict_is_refused_unless_skipped(atlas_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = atlas_root / "vst3" / "acme-audio" / "acme-squash@2.yaml"
    before = path.read_text()
    patch = tmp_path / "patch.yaml"
    patch.write_text(
        "default_provenance: {source: inferred, confidence: 0.5}\n"
        "params:\n  - id: thresh\n    unit: dBFS\n    role: threshold\n"
    )
    assert run("merge", path, patch) == 1
    err = capsys.readouterr().err
    assert "merge refused" in err and "conflict: params[thresh].unit: refusing inferred" in err
    assert path.read_text() == before

    out = tmp_path / "out.yaml"
    assert run("merge", path, patch, "--skip-conflicts", "--out", out) == 0
    captured = capsys.readouterr()
    assert "skipped params[thresh].unit" in captured.err
    merged = profiles.load_profile(out)
    assert merged["params"][0]["unit"] == "dB" and merged["params"][0]["role"] == "threshold"
    assert path.read_text() == before


def test_merge_missing_files_exit_1(atlas_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = atlas_root / "vst3" / "acme-audio" / "acme-squash@2.yaml"
    assert run("merge", path, tmp_path / "nope.yaml") == 1
    assert "unreadable patch" in capsys.readouterr().err
    assert run("merge", tmp_path / "nope.yaml", path) == 1


def test_list_text_filter_and_json(atlas_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("list", "--root", atlas_root) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and lines[0].startswith("clap:zed.zap\tZap\tZed\t1.0\t3 params\tclap/zed/zap@1.yaml")

    assert run("list", "--root", atlas_root, "--format", "vst3", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["plugin_id"] for r in rows] == ["vst3:ABCDEF0123456789"]

    assert run("list", "--root", atlas_root, "-q", "COMPRESSOR", "--json") == 0
    assert len(json.loads(capsys.readouterr().out)) == 2  # matched in the summary, case-insensitively
    assert run("list", "--root", atlas_root, "-q", "zed", "--json") == 0
    assert [r["name"] for r in json.loads(capsys.readouterr().out)] == ["Zap"]
    assert run("list", "--root", atlas_root, "-q", "nothing-like-this", "--json") == 0
    assert json.loads(capsys.readouterr().out) == []


def test_get_prints_profile_with_version_flag(atlas_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("get", "vst3:ABCDEF0123456789", "--root", atlas_root, "--version", "2.9.0", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["version_match"] is False and data["version_status"] == "same-major"
    assert data["profile"]["identity"]["name"] == "Acme Squash"

    assert run("get", "vst3:ABCDEF0123456789", "--root", atlas_root, "--version", "2.3.1", "--json") == 0
    assert json.loads(capsys.readouterr().out)["version_match"] is True

    assert run("get", "vst3:ABCDEF0123456789", "--root", atlas_root) == 0
    out = capsys.readouterr().out
    assert out.startswith("# ") and "version_status: unchecked" in out and "identity:\n  name: Acme Squash" in out

    assert run("get", "vst3:NOPE", "--root", atlas_root) == 1
    assert "no profile for vst3:NOPE" in capsys.readouterr().err


def test_check_reports_version_drift(atlas_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("check", "--root", atlas_root, "--plugin", "vst3:ABCDEF0123456789", "--version", "2.3.1") == 0
    assert "matches version 2.3.1" in capsys.readouterr().out
    assert run("check", "--root", atlas_root, "--plugin", "vst3:ABCDEF0123456789", "--version", "2.4.0") == 1
    assert "(same-major)" in capsys.readouterr().out
    assert run("check", "--root", atlas_root, "--plugin", "vst3:ABCDEF0123456789", "--version", "3.0") == 1
    assert "(different)" in capsys.readouterr().out
    assert run("check", "--root", atlas_root, "--plugin", "vst3:NOPE", "--version", "1") == 1
    assert "no profile" in capsys.readouterr().out
    with pytest.raises(SystemExit) as info:
        run("check", "--plugin", "x")  # --version is required
    assert info.value.code == 2


def test_root_comes_from_the_environment(atlas_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("ATLAS_PROFILES", str(atlas_root))
    assert cli.default_profiles_root() == atlas_root
    assert run("list", "--json") == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as info:
        cli.main([])
    assert info.value.code == 2


def test_registry_extension_points(capsys: pytest.CaptureFixture[str]) -> None:
    assert {"validate", "merge", "list", "get", "check"} <= set(cli.SUBCOMMANDS)

    seen: list[str] = []
    hello = cli.Subcommand("say hello", lambda p: p.add_argument("who"), lambda a: (seen.append(a.who), 0)[1])
    registry = dict(cli.SUBCOMMANDS)
    cli.register("hello", hello, registry)
    with pytest.raises(ValueError, match="already registered"):
        cli.register("hello", hello, registry)
    args = cli.build_parser(registry).parse_args(["hello", "world"])
    assert registry[args.command].run(args) == 0 and seen == ["world"]
    assert "hello" not in cli.SUBCOMMANDS  # the module registry was not touched


def test_extension_modules_are_loaded_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "fake_atlas_ext.py").write_text(
        "from atlas.cli import Subcommand\n"
        "def register_commands(registry):\n"
        "    registry['ext'] = Subcommand('extension', lambda p: None, lambda a: print('ext ran') or 0)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = dict(cli.SUBCOMMANDS)
    cli.load_extensions(registry, ("atlas.not_written_yet", "fake_atlas_ext"))  # missing modules are skipped
    assert "ext" in registry
    ns = argparse.Namespace()
    assert registry["ext"].run(ns) == 0 and "ext ran" in capsys.readouterr().out


def test_broken_extension_is_not_hidden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "broken_atlas_ext.py").write_text("import module_that_does_not_exist\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="module_that_does_not_exist"):
        cli.load_extensions({}, ("broken_atlas_ext",))
