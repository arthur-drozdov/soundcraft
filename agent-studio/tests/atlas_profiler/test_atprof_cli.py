"""`atlas profile` and `atlas builtins` through the real CLI entry point, with a FakeStudio behind Studio."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from atlas import cli, cli_builtins, cli_profile, profiles
from atlas.validate import validate_file
from atprof_fake_studio import FakeStudio
from studio_client import Studio, StudioError
from test_atprof_builtins import CURATED, fake_builtins


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Make Studio.headless / Studio.connect hand out one FakeStudio and record how they were called."""
    state: dict[str, Any] = {"studio": FakeStudio(), "headless": [], "connect": []}

    def headless(cls: Any, binary: str = "soundcraft-cli", args: Any = (), **options: Any) -> FakeStudio:
        state["headless"].append(binary)
        return state["studio"]  # type: ignore[no-any-return]

    def connect(cls: Any, port: int, host: str = "127.0.0.1", **options: Any) -> FakeStudio:
        state["connect"].append(port)
        return state["studio"]  # type: ignore[no-any-return]

    monkeypatch.setattr(Studio, "headless", classmethod(headless))
    monkeypatch.setattr(Studio, "connect", classmethod(connect))
    monkeypatch.delenv("SOUNDCRAFT_CLI", raising=False)
    return state


def atlas(*argv: Any) -> int:
    return cli.main([str(a) for a in argv])


# ---- wiring -----------------------------------------------------------------------------------------------------


def test_the_extension_commands_are_part_of_the_atlas_cli(capsys: pytest.CaptureFixture[str]) -> None:
    registry = dict(cli.SUBCOMMANDS)
    cli.load_extensions(registry)
    assert {"profile", "builtins", "validate", "merge", "list", "get", "check"} <= set(registry)
    with pytest.raises(SystemExit) as exit_info:
        atlas("--help")
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "profile" in out and "builtins" in out


def test_register_commands_fills_a_registry_and_refuses_duplicates() -> None:
    registry: dict[str, cli.Subcommand] = {}
    cli_profile.register_commands(registry)
    cli_builtins.register_commands(registry)
    assert sorted(registry) == ["builtins", "profile"]
    with pytest.raises(ValueError, match="already registered"):
        cli_profile.register_commands(registry)


def test_the_new_commands_do_not_make_plain_atlas_commands_slow() -> None:
    code = "import sys; from atlas import cli; r = dict(cli.SUBCOMMANDS); cli.load_extensions(r); cli.build_parser(r); print('numpy' in sys.modules, 'scipy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"], "numpy/scipy are imported only when `profile` or `builtins` run"


def test_profile_requires_a_plugin_and_the_two_targets_exclude_each_other(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e1:
        atlas("profile")
    assert e1.value.code == 2
    with pytest.raises(SystemExit) as e2:
        atlas("profile", "--plugin", "x", "--studio", "7000", "--headless", "soundcraft-cli")
    assert e2.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


# ---- atlas profile ---------------------------------------------------------------------------------------------------


def test_profile_headless_writes_a_validated_draft(fake: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "draft.yaml"
    code = atlas("profile", "--plugin", "fake_filter", "--headless", "my-soundcraft-cli", "--out", out, "--param", "cutoff", "--max-bounce-s", "3", "--date", "2026-10-22", "-q")
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert captured.out.strip() == str(out)
    assert fake["headless"] == ["my-soundcraft-cli"] and fake["connect"] == []
    assert validate_file(out) == []
    profile = profiles.load_profile(out)
    assert profile["identity"]["plugin_id"] == "fake_filter"
    cutoff = profile["params"][0]
    assert "spectral centroid" in cutoff["does"] and cutoff["provenance"]["does"]["source"] == "measured"
    assert "draft written" in captured.err
    assert fake["studio"].closed


def test_profile_out_folder_uses_the_canonical_layout_and_no_measure_skips_the_bounces(fake: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = atlas("profile", "--plugin", "fake_gain", "--headless", "x", "--out", tmp_path / "drafts", "--no-measure", "-q")
    assert code == 0
    path = Path(capsys.readouterr().out.strip())
    assert path == tmp_path / "drafts" / "builtin" / "soundcraft" / "fake-gain@0.yaml"
    assert fake["studio"].bounces == 0
    assert all("does" not in p for p in profiles.load_profile(path)["params"])


def test_profile_without_a_target_uses_soundcraft_cli_from_the_environment(fake: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert atlas("profile", "--plugin", "fake_gain", "--out", tmp_path / "a.yaml", "--no-measure", "-q") == 0
    monkeypatch.setenv("SOUNDCRAFT_CLI", "/opt/sc/soundcraft-cli")
    fake["studio"] = FakeStudio()
    assert atlas("profile", "--plugin", "fake_gain", "--out", tmp_path / "b.yaml", "--no-measure", "-q") == 0
    assert fake["headless"] == ["soundcraft-cli", "/opt/sc/soundcraft-cli"]


def test_profile_refuses_a_desktop_app_that_has_a_session_open(fake: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    studio = FakeStudio("control")
    studio.execute("track.new", {"kind": "audio", "name": "Vocals"})
    fake["studio"] = studio
    code = atlas("profile", "--plugin", "fake_gain", "--studio", "7801", "--out", tmp_path / "a.yaml", "-q")
    err = capsys.readouterr().err
    assert code == 1 and "--allow-current-session" in err and fake["connect"] == [7801]
    assert not (tmp_path / "a.yaml").exists() and [t.name for t in studio.state.tracks] == ["Vocals"] and studio.closed
    code = atlas("profile", "--plugin", "fake_gain", "--studio", "7801", "--out", tmp_path / "a.yaml", "--allow-current-session", "--no-measure", "-q")
    assert code == 0 and (tmp_path / "a.yaml").exists() and [t.name for t in studio.state.tracks] == ["Vocals"]


def test_profile_reports_errors_with_exit_code_1(fake: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert atlas("profile", "--plugin", "nope", "--headless", "x", "--out", tmp_path / "a.yaml", "-q") == 1
    assert "error:" in capsys.readouterr().err and not (tmp_path / "a.yaml").exists()
    assert atlas("profile", "--plugin", "fake_gain", "--headless", "x", "--out", tmp_path / "a.yaml", "--date", "tomorrow", "-q") == 1
    assert atlas("profile", "--plugin", "fake_gain", "--headless", "x", "--out", tmp_path / "a.yaml", "--max-bounce-s", "1", "-q") == 1


def test_profile_prints_warnings_and_progress_on_stderr(fake: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake["studio"] = FakeStudio()
    fake["studio"].fail["plugin.param_text"] = "boom"
    assert atlas("profile", "--plugin", "fake_gain", "--headless", "x", "--out", tmp_path / "a.yaml", "--no-measure") == 0
    err = capsys.readouterr().err
    assert "calibrating gain" in err and "warning: param `gain` not calibrated" in err


def test_a_studio_that_cannot_start_is_a_clean_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def refuse(cls: Any, *a: Any, **k: Any) -> Any:
        raise StudioError("cannot connect to SoundCraft at 127.0.0.1:1: refused")

    monkeypatch.setattr(Studio, "connect", classmethod(refuse))
    assert atlas("profile", "--plugin", "fake_gain", "--studio", "1", "--out", tmp_path / "a.yaml") == 1
    assert "cannot connect" in capsys.readouterr().err


# ---- atlas builtins --------------------------------------------------------------------------------------------------------


def test_builtins_writes_profiles_and_a_rerun_changes_nothing(fake: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake["studio"] = FakeStudio(plugins=fake_builtins())
    out = tmp_path / "builtin" / "soundcraft"
    assert atlas("builtins", "--binary", "sc", "--out", out, "--curated", CURATED) == 0
    first = capsys.readouterr()
    assert first.out.count("wrote ") == 5 and "5 plugins: 5 written, 0 unchanged, 0 failed" in first.out
    assert "plugins without curated knowledge: fake_filter, fake_gain, fake_synth" in first.err
    assert "curated files that match no plugin: " in first.err and "compressor" in first.err
    assert fake["headless"] == ["sc"]
    assert cli.main(["validate", str(tmp_path)]) == 0
    fake["studio"] = FakeStudio(plugins=fake_builtins())
    assert atlas("builtins", "--binary", "sc", "--out", out, "--curated", CURATED) == 0
    second = capsys.readouterr()
    assert "5 plugins: 0 written, 5 unchanged, 0 failed" in second.out and "wrote " not in second.out


def test_builtins_strict_fails_on_mismatches_and_errors_always_fail(fake: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake["studio"] = FakeStudio(plugins=fake_builtins())
    assert atlas("builtins", "--binary", "sc", "--out", tmp_path / "o", "--curated", CURATED, "--strict") == 1
    fake["studio"] = FakeStudio(plugins=fake_builtins())
    bad = tmp_path / "curated"
    bad.mkdir()
    (bad / "gain.yaml").write_text("default_provenance: {source: source, confidence: 0.9}\nparams:\n  - id: ghost\n    does: x\n")
    capsys.readouterr()
    assert atlas("builtins", "--binary", "sc", "--out", tmp_path / "o2", "--curated", bad) == 1
    assert "error: gain:" in capsys.readouterr().err


def test_builtins_date_flag_and_default_output_folder(fake: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake["studio"] = FakeStudio(plugins=fake_builtins())
    monkeypatch.setenv("ATLAS_PROFILES", str(tmp_path / "root"))
    assert atlas("builtins", "--binary", "sc", "--curated", CURATED, "--date", "2031-01-02") == 0
    files = sorted((tmp_path / "root" / "builtin" / "soundcraft").glob("*.yaml"))
    assert len(files) == 5 and {profiles.load_profile(f)["identity"]["profiled"] for f in files} == {"2031-01-02"}
    assert cli.main(["validate", str(tmp_path / "root")]) == 0


def test_builtins_reports_a_studio_failure(fake: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake["studio"] = FakeStudio(plugins=fake_builtins())
    fake["studio"].fail["engine.plugins"] = "no engine"
    assert atlas("builtins", "--binary", "sc", "--out", tmp_path, "--curated", CURATED) == 1
    assert "no engine" in capsys.readouterr().err
    assert atlas("builtins", "--binary", "sc", "--out", tmp_path, "--date", "never") == 1
