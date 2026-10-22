"""Built-in profile generation from engine.plugins + plugin.describe, merged with the curated knowledge."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from atlas import builtins, profiles
from atlas.profiler import ProfilerError
from atlas.validate import validate_file, validate_profile
from atprof_fake_studio import FakeParam, FakePlugin, FakeStudio, make_plugins

CURATED = Path(builtins.__file__).resolve().parents[2] / "atlas" / "curated" / "builtin"


def fake_builtins() -> dict[str, FakePlugin]:
    """Built-ins whose ids/params match real curated files (gain, invert) plus ones without any."""
    plugins = make_plugins()
    plugins["gain"] = FakePlugin("gain", "Gain", "Other", [FakeParam("gain", "Gain", -96.0, 24.0, 0.0, "Db")], process=lambda a, sr, p: a)
    plugins["invert"] = FakePlugin("invert", "Invert", "Other", [], process=lambda a, sr, p: -a)
    del plugins["vst3:FAKEHOSTED0001"]
    return plugins


@pytest.fixture
def studio() -> FakeStudio:
    return FakeStudio(plugins=fake_builtins())


def generate(studio: FakeStudio, out: Path, **kw: Any) -> builtins.BuiltinsReport:
    kw.setdefault("curated_dir", CURATED)
    kw.setdefault("today", "2026-10-22")
    return builtins.generate_builtins(studio, out, **kw)


def test_one_profile_per_builtin_named_after_the_plugin(studio: FakeStudio, tmp_path: Path) -> None:
    report = generate(studio, tmp_path)
    assert report.ok and report.plugin_count == 5
    names = sorted(p.name for p in tmp_path.glob("*.yaml"))
    assert names == ["fake-filter@0.yaml", "fake-gain@0.yaml", "fake-synth@0.yaml", "gain@0.yaml", "invert@0.yaml"]
    assert sorted(report.profiles) == ["fake_filter", "fake_gain", "fake_synth", "gain", "invert"]
    assert report.written and not report.unchanged and len(report.written) == 5


def test_host_fields_come_from_describe_and_validate(studio: FakeStudio, tmp_path: Path) -> None:
    generate(studio, tmp_path)
    profile = profiles.load_profile(tmp_path / "fake-filter@0.yaml")
    ident = profile["identity"]
    assert (ident["name"], ident["vendor"], ident["format"], ident["plugin_id"], ident["version"]) == ("Fake Filter", "SoundCraft", "builtin", "fake_filter", "0.10.0")
    assert ident["os"] == ["macos", "windows", "linux", "freebsd", "web"] and ident["profiled"] == "2026-10-22"
    cutoff = profile["params"][0]
    assert (cutoff["id"], cutoff["host_name"], cutoff["min"], cutoff["max"], cutoff["default"], cutoff["unit"], cutoff["taper"]) == ("cutoff", "Cutoff", 200.0, 16000.0, 8000.0, "Hz", "log")
    assert cutoff["provenance"]["min"] == {"source": "host", "confidence": 1.0}
    assert "calibration" not in cutoff
    assert validate_profile(profile) == []
    assert validate_file(tmp_path / "fake-synth@0.yaml") == []


def test_curated_knowledge_is_merged_over_the_host_fields(studio: FakeStudio, tmp_path: Path) -> None:
    generate(studio, tmp_path)
    gain = profiles.load_profile(tmp_path / "gain@0.yaml")
    param = gain["params"][0]
    assert param["min"] == -96.0 and param["provenance"]["min"]["source"] == "host"
    assert param["safe_range"] == [-60.0, 12.0] and param["automatable"] is True
    assert param["provenance"]["safe_range"]["source"] == "source" and "utility.rs" in param["provenance"]["safe_range"]["ref"]
    assert gain["summary"].startswith("Smoothed gain change")
    invert = profiles.load_profile(tmp_path / "invert@0.yaml")
    assert invert["params"] == [] and "signal_flow" in invert
    # a plugin with no curated file keeps the host-only summary
    assert profiles.load_profile(tmp_path / "fake-gain@0.yaml")["summary"] == "Effect with 3 parameters."


def test_the_report_names_curated_files_without_a_plugin_and_plugins_without_curated_knowledge(studio: FakeStudio, tmp_path: Path) -> None:
    report = generate(studio, tmp_path)
    assert report.plugins_without_curated == ["fake_filter", "fake_gain", "fake_synth"]
    assert "compressor" in report.curated_without_plugin and "gain" not in report.curated_without_plugin and "invert" not in report.curated_without_plugin
    assert report.curated_without_plugin == sorted(report.curated_without_plugin)
    assert len(report.curated_without_plugin) == len(builtins.curated_ids(CURATED)) - 2


def test_output_is_deterministic_and_a_rerun_rewrites_nothing(studio: FakeStudio, tmp_path: Path) -> None:
    generate(studio, tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.yaml")}
    mtimes = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("*.yaml")}
    again = generate(FakeStudio(plugins=fake_builtins()), tmp_path, today="2027-01-01")
    assert again.written == [] and len(again.unchanged) == 5
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.yaml")} == before
    assert {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("*.yaml")} == mtimes
    assert all(r["identity"]["profiled"] == "2026-10-22" for r in again.profiles.values()), "an unchanged profile keeps its date"


def test_two_fresh_runs_write_identical_bytes(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    generate(FakeStudio(plugins=fake_builtins()), a)
    generate(FakeStudio(plugins=fake_builtins()), b)
    assert {p.name: p.read_bytes() for p in a.glob("*.yaml")} == {p.name: p.read_bytes() for p in b.glob("*.yaml")}


def test_the_date_changes_only_when_the_content_does(tmp_path: Path) -> None:
    generate(FakeStudio(plugins=fake_builtins()), tmp_path)
    plugins = fake_builtins()
    plugins["fake_filter"].params[0].default = 4000.0
    report = generate(FakeStudio(plugins=plugins), tmp_path, today="2027-02-03")
    assert [p.name for p in report.written] == ["fake-filter@0.yaml"]
    assert profiles.load_profile(tmp_path / "fake-filter@0.yaml")["identity"]["profiled"] == "2027-02-03"
    assert profiles.load_profile(tmp_path / "gain@0.yaml")["identity"]["profiled"] == "2026-10-22"


def test_a_new_plugin_version_is_new_content(tmp_path: Path) -> None:
    generate(FakeStudio(plugins=fake_builtins()), tmp_path)
    plugins = fake_builtins()
    plugins["gain"].version = "0.11.0"
    report = generate(FakeStudio(plugins=plugins), tmp_path, today="2027-02-03")
    assert [p.name for p in report.written] == ["gain@0.yaml"]


def test_an_explicit_date_is_applied_to_every_profile(studio: FakeStudio, tmp_path: Path) -> None:
    generate(studio, tmp_path)
    report = generate(FakeStudio(plugins=fake_builtins()), tmp_path, date="2030-05-06")
    assert len(report.written) == 5
    assert {profiles.load_profile(p)["identity"]["profiled"] for p in tmp_path.glob("*.yaml")} == {"2030-05-06"}
    again = generate(FakeStudio(plugins=fake_builtins()), tmp_path, date="2030-05-06")
    assert again.written == [] and len(again.unchanged) == 5


def test_a_bad_curated_patch_is_reported_and_its_file_is_not_written(studio: FakeStudio, tmp_path: Path) -> None:
    curated = tmp_path / "curated"
    curated.mkdir()
    (curated / "gain.yaml").write_text("default_provenance: {source: source, confidence: 0.9}\nparams:\n  - id: no_such_param\n    does: x\n")
    shutil.copy(CURATED / "invert.yaml", curated / "invert.yaml")
    out = tmp_path / "out"
    report = generate(studio, out, curated_dir=curated)
    assert not report.ok and list(report.errors) == ["gain"] and "no_such_param" in report.errors["gain"]
    assert not (out / "gain@0.yaml").exists() and (out / "invert@0.yaml").exists()
    assert report.curated_without_plugin == []


def test_stale_files_are_reported_not_deleted(studio: FakeStudio, tmp_path: Path) -> None:
    (tmp_path / "old-plugin@0.yaml").write_text("identity: {}\n")
    report = generate(studio, tmp_path)
    assert report.stale_files == [tmp_path / "old-plugin@0.yaml"] and (tmp_path / "old-plugin@0.yaml").exists()


def test_a_missing_curated_folder_means_every_plugin_lacks_knowledge(studio: FakeStudio, tmp_path: Path) -> None:
    report = generate(studio, tmp_path / "out", curated_dir=tmp_path / "nowhere")
    assert report.ok and len(report.plugins_without_curated) == 5 and report.curated_without_plugin == []
    for p in (tmp_path / "out").glob("*.yaml"):
        assert validate_file(p) == []


def test_calibration_can_be_added_on_request(studio: FakeStudio, tmp_path: Path) -> None:
    generate(studio, tmp_path, calibrate_params=True)
    cutoff = profiles.load_profile(tmp_path / "fake-filter@0.yaml")["params"][0]
    assert len(cutoff["calibration"]) == 21 and cutoff["provenance"]["calibration"]["source"] == "host"


def test_effects_and_instruments_are_each_described_on_the_right_kind_of_track(studio: FakeStudio, tmp_path: Path) -> None:
    generate(studio, tmp_path)
    created = [p["kind"] for c, p in studio.calls if c == "track.new"]
    assert sorted(created) == ["audio", "instrument"]
    assert any(c == "mix.instrument" for c, _ in studio.calls)
    assert studio.state.tracks == [] and studio.calls[-1][0] == "session.close"


def test_a_plugin_whose_description_disagrees_with_the_list_is_an_error(studio: FakeStudio, tmp_path: Path) -> None:
    real = studio.cmd_engine_plugins

    def lying(p: dict[str, Any]) -> list[dict[str, Any]]:
        rows = real(p)
        for row in rows:
            if row["id"] == "fake_gain":
                row["params"] = row["params"][:-1]
        return rows

    studio.cmd_engine_plugins = lying  # type: ignore[method-assign]
    report = generate(studio, tmp_path)
    assert list(report.errors) == ["fake_gain"] and "disagree" in report.errors["fake_gain"]
    assert len(report.written) == 4


def test_hostile_listings_stop_the_run(tmp_path: Path) -> None:
    studio = FakeStudio(plugins=fake_builtins())
    studio.cmd_engine_plugins = lambda p: {"not": "a list"}  # type: ignore[method-assign]
    with pytest.raises(ProfilerError, match="did not return a list"):
        generate(studio, tmp_path)
    assert studio.calls[-1][0] == "session.close"
    dup = FakeStudio(plugins=fake_builtins())
    dup.cmd_engine_plugins = lambda p: [{"id": "x", "params": []}, {"id": "x", "params": []}, "junk", {"id": 5}]  # type: ignore[method-assign]
    with pytest.raises(ProfilerError, match="twice"):
        generate(dup, tmp_path)


def test_a_bad_date_is_refused_before_the_studio_is_touched(studio: FakeStudio, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate(studio, tmp_path, date="yesterday")
    assert studio.calls == []


def test_curated_ids_lists_only_yaml_patches() -> None:
    ids = builtins.curated_ids(CURATED)
    assert "gain" in ids and "README" not in ids and ids == sorted(ids)
    assert builtins.curated_ids(CURATED / "nowhere") == []
