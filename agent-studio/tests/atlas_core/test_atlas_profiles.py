"""Path naming, stable YAML, lookup and version checks."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import pytest

from atlas import profiles, yamlio
from atlas.validate import ProfileError
from atlas_core_samples import sample_profile


@pytest.mark.parametrize(
    ("text", "slug"),
    [
        ("Acme Audio", "acme-audio"),
        ("FabFilter", "fabfilter"),
        ("Pro-Q 3", "pro-q-3"),
        ("  Über  Köln!! ", "uber-koln"),
        ("../../etc/passwd", "etc-passwd"),
        ("///", "unnamed"),
        ("日本語", "unnamed"),
        ("a" * 200, "a" * 64),
    ],
)
def test_slugify(text: str, slug: str) -> None:
    assert profiles.slugify(text) == slug


@pytest.mark.parametrize(("version", "major"), [("2.3.1", 2), ("v3", 3), ("10.0-beta", 10), ("beta", 0), ("", 0)])
def test_major_of(version: str, major: int) -> None:
    assert profiles.major_of(version) == major


def test_relpath_naming() -> None:
    assert profiles.profile_relpath(sample_profile()).as_posix() == "vst3/acme-audio/acme-squash@2.yaml"
    p = sample_profile(format="builtin", vendor="SoundCraft", name="Compressor", version="1.0.0", plugin_id="compressor")
    assert profiles.profile_relpath(p).as_posix() == "builtin/soundcraft/compressor@1.yaml"


def test_relpath_cannot_escape_the_root(tmp_path: Path) -> None:
    p = sample_profile(vendor="../../..", name="..", version="1")
    path = profiles.profile_path(tmp_path, p)
    assert path.resolve().is_relative_to(tmp_path.resolve())


def test_relpath_needs_identity() -> None:
    with pytest.raises(ProfileError):
        profiles.profile_relpath({})
    with pytest.raises(ProfileError, match="missing vendor"):
        profiles.profile_relpath({"identity": {"format": "vst3", "name": "x", "version": "1"}})


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    p = sample_profile()
    path = profiles.save_profile(tmp_path, p)
    assert path == tmp_path / "vst3" / "acme-audio" / "acme-squash@2.yaml"
    assert profiles.load_profile(path) == p
    assert [f.name for f in path.parent.iterdir()] == [path.name]  # no temp files left behind


def test_yaml_is_stable_regardless_of_key_order(tmp_path: Path) -> None:
    p = sample_profile()
    shuffled: dict[str, Any] = {}
    keys = list(p)
    random.Random(1).shuffle(keys)
    for k in keys:
        shuffled[k] = p[k]
    shuffled["params"] = [dict(reversed(list(x.items()))) for x in p["params"]]
    assert profiles.dump_profile(shuffled) == profiles.dump_profile(p)
    text = profiles.dump_profile(p)
    assert text.index("identity:") < text.index("summary:") < text.index("params:") < text.rindex("provenance:")
    assert text.index("id: thresh") < text.index("host_name: Threshold") < text.index("min: -60.0")
    assert profiles.dump_profile(profiles.load_profile(profiles.save_profile(tmp_path, p))) == text


def test_tricky_strings_survive_a_round_trip(tmp_path: Path) -> None:
    p = sample_profile()
    p["params"][2]["choices"] = ["Off", "On", "yes", "no", "4:1", "2026-10-22", "null", "1e3", ""]
    p["params"][2]["provenance"]["choices"] = {"source": "host", "confidence": 1.0}
    back = profiles.load_profile(profiles.save_profile(tmp_path, p))
    assert back["params"][2]["choices"] == p["params"][2]["choices"]
    assert back["identity"]["profiled"] == "2026-10-22"


def test_invalid_profile_is_not_written(tmp_path: Path) -> None:
    p = sample_profile()
    p["params"][0]["provenance"]["taper"] = {"source": "inferred", "confidence": 0.95}
    with pytest.raises(ProfileError):
        profiles.save_profile(tmp_path, p)
    assert list(tmp_path.rglob("*")) == []


def test_load_rejects_invalid_files(tmp_path: Path) -> None:
    f = tmp_path / "x.yaml"
    f.write_text("- not\n- a mapping\n")
    with pytest.raises(ProfileError, match="mapping"):
        profiles.load_profile(f)
    f.write_text("identity: {}\nparams: []\n")
    with pytest.raises(ProfileError):
        profiles.load_profile(f)
    assert profiles.load_profile(f, validate=False)["params"] == []
    with pytest.raises(ProfileError):
        profiles.load_profile(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("profiled", "actual", "status"),
    [
        ("2.3.1", "2.3.1", "exact"),
        ("v2.3.1", " 2.3.1 ", "exact"),
        ("2.3.1", "2.4.0", "same-major"),
        ("2.3.1", "3.0.0", "different"),
        ("1.0", "10.0", "different"),
    ],
)
def test_version_status(profiled: str, actual: str, status: str) -> None:
    assert profiles.version_status(profiled, actual) == status
    assert profiles.version_matches({"identity": {"version": profiled}}, actual) is (status == "exact")


def make_atlas(root: Path) -> None:
    profiles.save_profile(root, sample_profile(version="1.4.0"))
    profiles.save_profile(root, sample_profile(version="2.3.1"))
    profiles.save_profile(root, sample_profile(version="2.5.0", name="Acme Squash Two", plugin_id="vst3:OTHER"))
    profiles.save_profile(root, sample_profile(format="clap", vendor="Zed", name="Zap", plugin_id="clap:zed.zap", version="1.0"))


def test_find_profile_by_id_and_version(tmp_path: Path) -> None:
    make_atlas(tmp_path)
    pid = "vst3:ABCDEF0123456789"
    exact = profiles.find_profile(tmp_path, pid, "2.3.1")
    assert exact and exact.status == "exact" and exact.version_match is True
    assert exact.profile["identity"]["version"] == "2.3.1"

    same_major = profiles.find_profile(tmp_path, pid, "2.9.9")
    assert same_major and same_major.status == "same-major" and same_major.version_match is False
    assert same_major.profile["identity"]["version"] == "2.3.1"

    old = profiles.find_profile(tmp_path, pid, "1.4.0")
    assert old and old.status == "exact" and old.profile["identity"]["version"] == "1.4.0"

    far = profiles.find_profile(tmp_path, pid, "7.0")
    assert far and far.status == "different" and far.profile["identity"]["version"] == "2.3.1"  # newest wins

    latest = profiles.find_profile(tmp_path, pid)
    assert latest and latest.status == "unchecked" and latest.version_match is None
    assert latest.profile["identity"]["version"] == "2.3.1"

    assert profiles.find_profile(tmp_path, "vst3:NOPE") is None
    assert profiles.find_profile(tmp_path / "missing", pid) is None


def test_list_profiles_skips_invalid_and_symlinked_files(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    make_atlas(tmp_path)
    broken = tmp_path / "vst3" / "acme-audio" / "broken@1.yaml"
    broken.write_text("identity: {}\n")
    (tmp_path / "vst3" / "acme-audio" / "link@1.yaml").symlink_to(tmp_path / "vst3" / "acme-audio" / "acme-squash@2.yaml")
    (tmp_path / "stray.yaml").write_text("x: 1\n")
    with caplog.at_level(logging.WARNING):
        found = profiles.list_profiles(tmp_path)
    assert sorted(r.path.name for r in found) == ["acme-squash-two@2.yaml", "acme-squash@1.yaml", "acme-squash@2.yaml", "zap@1.yaml"]
    assert any("broken@1.yaml" in r.message for r in caplog.records)


def test_check_layout(tmp_path: Path) -> None:
    p = sample_profile()
    good = profiles.save_profile(tmp_path, p)
    assert profiles.check_layout(good, tmp_path, p) == []
    moved = tmp_path / "vst3" / "acme-audio" / "wrong-name@2.yaml"
    good.rename(moved)
    [issue] = profiles.check_layout(moved, tmp_path, p)
    assert "expected vst3/acme-audio/acme-squash@2.yaml" in issue.message
    assert profiles.check_layout(Path("/elsewhere/x.yaml"), tmp_path, p)


def test_yamlio_helper_is_what_load_uses(tmp_path: Path) -> None:
    f = tmp_path / "a.yaml"
    f.write_text(profiles.dump_profile(sample_profile()))
    assert yamlio.read_file(f)["identity"]["name"] == "Acme Squash"
