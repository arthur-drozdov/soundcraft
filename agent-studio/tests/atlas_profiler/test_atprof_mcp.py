"""atlas-mcp: the five tools (called directly, through the SDK in memory, and over real stdio)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from atlas import mcp_server, profiler, profiles
from atlas.mcp_server import AtlasToolError, AtlasTools
from atlas.validate import validate_file
from atprof_fake_studio import FakeStudio

pytest.importorskip("mcp")

PATCH = {"default_provenance": {"source": "manual", "ref": "Manual p.4", "confidence": 0.95}, "params": [{"id": "cutoff", "does": "Corner frequency of the low-pass."}]}


def make_profile(**identity: Any) -> dict[str, Any]:
    result = profiler.profile_plugin(FakeStudio(), "fake_filter", profiler.ProfilerOptions(measure=False, date="2026-10-22", progress=lambda m: None))
    profile = result.profile
    profile["identity"].update(identity)
    return profile


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Any]:
    root, project = tmp_path / "profiles", tmp_path / "project"
    project.mkdir()
    exact = profiles.save_profile(root, make_profile())
    other = profiles.save_profile(root, make_profile(plugin_id="vst3:ABCDEF", format="vst3", vendor="Acme", name="Acme Squash", version="2.3.1"))
    return {"root": root, "project": project, "exact": exact, "other": other, "tmp": tmp_path}


@pytest.fixture
def tools(env: dict[str, Any]) -> AtlasTools:
    return AtlasTools(env["root"], env["project"], studio_factory=lambda port: FakeStudio())


# ---- atlas_list / atlas_get ------------------------------------------------------------------------------------


def test_atlas_list_returns_rows_and_filters_by_text(tools: AtlasTools) -> None:
    everything = tools.atlas_list()
    assert everything["count"] == 2
    row = next(r for r in everything["profiles"] if r["plugin_id"] == "fake_filter")
    assert row == {"plugin_id": "fake_filter", "name": "Fake Filter", "vendor": "SoundCraft", "format": "builtin", "version": "0.10.0", "params": 2, "path": "builtin/soundcraft/fake-filter@0.yaml"}
    assert [r["plugin_id"] for r in tools.atlas_list("SQUASH")["profiles"]] == ["vst3:ABCDEF"]
    assert tools.atlas_list("acme")["count"] == 1 and tools.atlas_list("zzz")["count"] == 0
    assert tools.atlas_list("  ")["count"] == 2
    for bad in (5, "x" * 201):
        with pytest.raises(AtlasToolError):
            tools.atlas_list(bad)  # type: ignore[arg-type]


def test_atlas_get_flags_whether_the_profile_matches_the_installed_version(tools: AtlasTools) -> None:
    exact = tools.atlas_get("vst3:ABCDEF", "2.3.1")
    assert exact["found"] and exact["version_match"] is True and exact["version_status"] == "exact"
    assert exact["profile"]["identity"]["name"] == "Acme Squash" and exact["path"].endswith("acme-squash@2.yaml")
    same_major = tools.atlas_get("vst3:ABCDEF", "2.4.0")
    assert same_major["version_match"] is False and same_major["version_status"] == "same-major"
    assert tools.atlas_get("vst3:ABCDEF", "3.0")["version_status"] == "different"
    unchecked = tools.atlas_get("vst3:ABCDEF")
    assert unchecked["version_match"] is None and unchecked["version_status"] == "unchecked"


def test_atlas_get_for_an_unprofiled_plugin_is_not_an_error(tools: AtlasTools) -> None:
    result = tools.atlas_get("vst3:NOPE")
    assert result["found"] is False and "atlas_profile" in result["message"]
    for bad in ("", "  ", "x" * 301, 5):
        with pytest.raises(AtlasToolError):
            tools.atlas_get(bad)  # type: ignore[arg-type]
    with pytest.raises(AtlasToolError):
        tools.atlas_get("fake_filter", "v" * 65)


# ---- atlas_validate ---------------------------------------------------------------------------------------------


def test_atlas_validate_a_file_and_a_folder(tools: AtlasTools, env: dict[str, Any]) -> None:
    ok = tools.atlas_validate(str(env["exact"]))
    assert ok == {"ok": True, "checked": 1, "files": [{"path": "builtin/soundcraft/fake-filter@0.yaml", "ok": True, "issues": []}]}
    broken = env["project"] / "broken.yaml"
    broken.write_text("identity: {name: x}\nparams: []\n")
    bad = tools.atlas_validate("broken.yaml")
    assert bad["ok"] is False and bad["files"][0]["issues"] and "path" in bad["files"][0]["issues"][0]
    folder = tools.atlas_validate(str(env["root"]))
    assert folder["ok"] and folder["checked"] == 2
    (env["root"] / "builtin" / "soundcraft" / "junk.yaml").write_text("not: [valid")
    mixed = tools.atlas_validate(str(env["root"]))
    assert mixed["ok"] is False and mixed["checked"] == 3


def test_atlas_validate_needs_something_to_check(tools: AtlasTools, env: dict[str, Any]) -> None:
    (env["project"] / "empty").mkdir()
    with pytest.raises(AtlasToolError, match="no profile files"):
        tools.atlas_validate("empty")


# ---- path confinement ---------------------------------------------------------------------------------------------


def test_paths_are_confined_to_the_profiles_root_and_the_project_folder(tools: AtlasTools, env: dict[str, Any]) -> None:
    outside = env["tmp"] / "elsewhere.yaml"
    outside.write_text("x: 1\n")
    for hostile in (str(outside), "/etc/passwd", "../elsewhere.yaml", "../../../../etc/passwd", "builtin/../../elsewhere.yaml"):
        with pytest.raises(AtlasToolError):
            tools.atlas_validate(hostile)
        with pytest.raises(AtlasToolError):
            tools.atlas_merge(hostile, PATCH)
    for bad in ("", "  ", "a\x00b", "x" * 5000, None, 5):
        with pytest.raises(AtlasToolError):
            tools.atlas_validate(bad)  # type: ignore[arg-type]


def test_a_symlink_inside_the_project_cannot_lead_out(tools: AtlasTools, env: dict[str, Any]) -> None:
    secret = env["tmp"] / "secret.yaml"
    secret.write_text("x: 1\n")
    link = env["project"] / "link.yaml"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("symlinks not available")
    with pytest.raises(AtlasToolError):
        tools.atlas_validate("link.yaml")
    folder_link = env["project"] / "out"
    folder_link.symlink_to(env["tmp"], target_is_directory=True)
    with pytest.raises(AtlasToolError):
        tools.atlas_validate("out/secret.yaml")


def test_relative_paths_are_found_in_the_project_folder_then_the_profiles_root(tools: AtlasTools, env: dict[str, Any]) -> None:
    assert tools.atlas_validate("builtin/soundcraft/fake-filter@0.yaml")["ok"]  # profiles-root relative
    (env["project"] / "mine.yaml").write_text(env["exact"].read_text())
    assert tools.atlas_validate("mine.yaml")["files"][0]["path"].endswith("mine.yaml")


# ---- atlas_profile -------------------------------------------------------------------------------------------------


def test_atlas_profile_runs_the_profiler_and_returns_the_draft_path(tools: AtlasTools, env: dict[str, Any]) -> None:
    result = tools.atlas_profile("fake_gain", measure=False)
    draft = Path(result["draft_path"])
    assert draft == env["project"].resolve() / "atlas-drafts" / "builtin" / "soundcraft" / "fake-gain@0.yaml"
    assert draft.is_file() and validate_file(draft) == []
    assert result["plugin_id"] == "fake_gain" and result["params"] == 3 and result["scratch_session"] == "new"
    assert result["measured"] == 0 and "atlas_merge" in result["next"]
    assert tools.atlas_list()["count"] == 2, "drafts do not appear among the reviewed profiles"


def test_atlas_profile_measures_unless_told_not_to(tools: AtlasTools) -> None:
    result = tools.atlas_profile("fake_gain", max_params=1)
    assert result["measured"] == 1 and result["bounces"] > 0
    assert any("measured 1 of 3" in w for w in result["warnings"])


def test_atlas_profile_uses_the_port_or_the_headless_binary(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[Any] = []

    def factory(port: int | None) -> FakeStudio:
        asked.append(port)
        return FakeStudio()

    t = AtlasTools(env["root"], env["project"], studio_factory=factory)
    monkeypatch.delenv("SOUNDCRAFT_CONTROL_PORT", raising=False)
    t.atlas_profile("fake_gain", measure=False)
    t.atlas_profile("fake_gain", port=7801, measure=False)
    monkeypatch.setenv("SOUNDCRAFT_CONTROL_PORT", "7802")
    t.atlas_profile("fake_gain", measure=False)
    assert asked == [None, 7801, 7802]


def test_atlas_profile_refuses_a_desktop_session_in_use_and_always_closes_the_studio(env: dict[str, Any]) -> None:
    studio = FakeStudio("control")
    studio.execute("track.new", {"kind": "audio", "name": "Vocals"})
    t = AtlasTools(env["root"], env["project"], studio_factory=lambda port: studio)
    with pytest.raises(AtlasToolError, match="allow_current_session"):
        t.atlas_profile("fake_gain", port=7801)
    assert studio.closed and [x.name for x in studio.state.tracks] == ["Vocals"]
    assert not (env["project"] / "atlas-drafts").exists()
    studio2 = FakeStudio("control")
    studio2.execute("track.new", {"kind": "audio", "name": "Vocals"})
    t2 = AtlasTools(env["root"], env["project"], studio_factory=lambda port: studio2)
    result = t2.atlas_profile("fake_gain", port=7801, measure=False, allow_current_session=True)
    assert result["scratch_session"] == "current" and [x.name for x in studio2.state.tracks] == ["Vocals"]


def test_atlas_profile_rejects_bad_arguments_and_unknown_plugins(tools: AtlasTools) -> None:
    for kwargs in ({"port": 0}, {"port": 70000}, {"port": True}, {"port": "7801"}, {"max_params": -1}, {"max_params": 5000}, {"max_params": True}):
        with pytest.raises(AtlasToolError):
            tools.atlas_profile("fake_gain", **kwargs)
    with pytest.raises(AtlasToolError, match="no_such"):
        tools.atlas_profile("no_such")
    with pytest.raises(AtlasToolError):
        tools.atlas_profile("")


def test_atlas_profile_reports_a_studio_that_cannot_start(env: dict[str, Any]) -> None:
    from studio_client import StudioError

    def boom(port: int | None) -> Any:
        raise StudioError("cannot connect to SoundCraft at 127.0.0.1:1")

    with pytest.raises(AtlasToolError, match="cannot connect"):
        AtlasTools(env["root"], env["project"], studio_factory=boom).atlas_profile("fake_gain", port=1)


# ---- atlas_merge ----------------------------------------------------------------------------------------------------


def test_atlas_merge_applies_a_mapping_patch_in_place_with_provenance(tools: AtlasTools, env: dict[str, Any]) -> None:
    result = tools.atlas_merge(str(env["exact"]), PATCH)
    assert result["written"] is True and result["skipped"] == []
    assert result["changes"] == [{"path": "params[cutoff].does", "provenance": {"source": "manual", "ref": "Manual p.4", "confidence": 0.95}}]
    cutoff = profiles.load_profile(env["exact"])["params"][0]
    assert cutoff["does"] == "Corner frequency of the low-pass." and cutoff["provenance"]["does"]["source"] == "manual"
    assert validate_file(env["exact"]) == []


def test_atlas_merge_takes_yaml_or_json_text_and_can_dry_run(tools: AtlasTools, env: dict[str, Any]) -> None:
    before = env["exact"].read_bytes()
    text = "default_provenance: {source: manual, confidence: 0.9}\nsummary: A low-pass.\n"
    dry = tools.atlas_merge("builtin/soundcraft/fake-filter@0.yaml", text, dry_run=True)
    assert dry["written"] is False and dry["changes"][0]["path"] == "summary"
    assert env["exact"].read_bytes() == before
    tools.atlas_merge("builtin/soundcraft/fake-filter@0.yaml", json.dumps({"default_provenance": {"source": "manual", "confidence": 0.9}, "summary": "A JSON low-pass."}))
    assert profiles.load_profile(env["exact"])["summary"] == "A JSON low-pass."


def test_atlas_merge_refuses_what_atlas_merge_refuses_and_leaves_the_file_alone(tools: AtlasTools, env: dict[str, Any]) -> None:
    before = env["exact"].read_bytes()
    inferred_over_host = {"default_provenance": {"source": "inferred", "confidence": 0.5}, "params": [{"id": "cutoff", "min": 1.0}]}
    with pytest.raises(AtlasToolError, match="merge refused"):
        tools.atlas_merge(str(env["exact"]), inferred_over_host)
    skipped = tools.atlas_merge(str(env["exact"]), {**inferred_over_host, "known_issues": ["ringing at high resonance"]}, skip_conflicts=True)
    assert len(skipped["skipped"]) == 1 and "min" in skipped["skipped"][0]
    merged = profiles.load_profile(env["exact"])
    assert merged["params"][0]["min"] == 200.0 and merged["known_issues"] == ["ringing at high resonance"]
    env["exact"].write_bytes(before)
    for bad in (
        {"nonsense": 1}, {"params": [{"id": "ghost", "does": "x"}], "default_provenance": {"source": "manual", "confidence": 0.9}},
        {"summary": "no provenance"}, "a: [unclosed", "just text", 5, None, "x" * (mcp_server.MAX_PATCH_BYTES + 1),
    ):
        with pytest.raises(AtlasToolError):
            tools.atlas_merge(str(env["exact"]), bad)  # type: ignore[arg-type]
    assert env["exact"].read_bytes() == before


def test_atlas_merge_needs_an_existing_valid_profile_file(tools: AtlasTools, env: dict[str, Any]) -> None:
    with pytest.raises(AtlasToolError):
        tools.atlas_merge("nowhere.yaml", PATCH)
    with pytest.raises(AtlasToolError, match="not a file"):
        tools.atlas_merge(str(env["root"]), PATCH)
    (env["project"] / "bad.yaml").write_text("identity: {}\n")
    with pytest.raises(AtlasToolError, match="invalid"):
        tools.atlas_merge("bad.yaml", PATCH)


# ---- through the SDK ---------------------------------------------------------------------------------------------------


async def sdk_call(server: Any, name: str | None, arguments: dict[str, Any] | None = None) -> tuple[bool, str, Any]:
    """(is_error, text, structured) of one tool call (or the tool names when ``name`` is None), in memory."""
    try:
        from mcp.client import Client  # mcp >= 2

        async with Client(server) as client:
            if name is None:
                return False, ",".join(t.name for t in (await client.list_tools()).tools), None
            r = await client.call_tool(name, arguments or {})
            return bool(r.is_error), r.content[0].text, getattr(r, "structured_content", None)
    except ImportError:
        from mcp.shared.memory import create_connected_server_and_client_session  # mcp 1.x

        async with create_connected_server_and_client_session(server._mcp_server) as session:
            if name is None:
                return False, ",".join(t.name for t in (await session.list_tools()).tools), None
            r = await session.call_tool(name, arguments or {})
            return bool(r.isError), r.content[0].text, getattr(r, "structuredContent", None)


async def test_the_server_registers_the_five_tools(tools: AtlasTools) -> None:
    _, names, _ = await sdk_call(mcp_server.build_server(tools), None)
    assert names.split(",") == ["atlas_list", "atlas_get", "atlas_validate", "atlas_profile", "atlas_merge"]


async def test_tools_answer_through_the_sdk(tools: AtlasTools, env: dict[str, Any]) -> None:
    server = mcp_server.build_server(tools)
    is_error, text, structured = await sdk_call(server, "atlas_get", {"plugin_id": "vst3:ABCDEF", "version": "2.3.1"})
    assert not is_error and json.loads(text)["version_match"] is True
    is_error, text, _ = await sdk_call(server, "atlas_list", {"query": "filter"})
    assert not is_error and json.loads(text)["count"] == 1
    is_error, text, _ = await sdk_call(server, "atlas_validate", {"path": str(env["exact"])})
    assert not is_error and json.loads(text)["ok"] is True


async def test_failures_reach_the_agent_with_their_message(tools: AtlasTools, env: dict[str, Any]) -> None:
    server = mcp_server.build_server(tools)
    is_error, text, _ = await sdk_call(server, "atlas_validate", {"path": "/etc/passwd"})
    assert is_error and "outside the profiles folder" in text
    is_error, text, _ = await sdk_call(server, "atlas_merge", {"profile_path": str(env["exact"]), "patch": {"summary": "x"}})
    assert is_error and "no provenance" in text
    is_error, text, _ = await sdk_call(server, "atlas_get", {"plugin_id": 5})
    assert is_error


async def test_atlas_profile_runs_off_the_event_loop_and_merge_accepts_a_mapping(tools: AtlasTools, env: dict[str, Any]) -> None:
    server = mcp_server.build_server(tools)
    is_error, text, _ = await sdk_call(server, "atlas_profile", {"plugin_id": "fake_filter", "max_params": 1})
    result = json.loads(text)
    assert not is_error and result["measured"] == 1 and Path(result["draft_path"]).is_file()
    is_error, text, _ = await sdk_call(server, "atlas_merge", {"profile_path": result["draft_path"], "patch": PATCH})
    assert not is_error and json.loads(text)["written"] is True
    assert profiles.load_profile(result["draft_path"])["params"][0]["provenance"]["does"]["source"] == "manual"


# ---- over real stdio ------------------------------------------------------------------------------------------------------


async def test_atlas_mcp_main_serves_over_stdio(env: dict[str, Any]) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "atlas.mcp_server"],
        env={**os.environ, "ATLAS_PROFILES": str(env["root"]), "AGENT_STUDIO_PROJECT": str(env["project"])},
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        names = [t.name for t in (await session.list_tools()).tools]
        assert names == ["atlas_list", "atlas_get", "atlas_validate", "atlas_profile", "atlas_merge"]
        r = await session.call_tool("atlas_list", {})
        assert not getattr(r, "is_error", getattr(r, "isError", False))
        assert json.loads(r.content[0].text)["count"] == 2
        r = await session.call_tool("atlas_validate", {"path": "/etc/hostname"})
        assert getattr(r, "is_error", getattr(r, "isError", False))


def test_defaults_come_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ATLAS_PROFILES", str(tmp_path / "p"))
    monkeypatch.setenv("AGENT_STUDIO_PROJECT", str(tmp_path / "proj"))
    t = AtlasTools()
    assert t.profiles_root == tmp_path / "p" and t.project_dir == tmp_path / "proj"
    monkeypatch.delenv("ATLAS_PROFILES")
    monkeypatch.delenv("AGENT_STUDIO_PROJECT")
    d = AtlasTools()
    assert d.profiles_root.name == "profiles" and d.project_dir == Path.cwd()
