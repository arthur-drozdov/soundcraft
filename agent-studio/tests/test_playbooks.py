"""Playbooks: skill front matter, command ids named in the docs, the MCP template and new_project.py.

The command-id check reads the Rust sources next to this project (``crates/engine/src/cmd``,
``crates/ui-egui/src/menus.rs``) and DESIGN.md section 9, so it is skipped when the project is
checked out without the SoundCraft workspace. The real-binary test needs ``SOUNDCRAFT_CLI``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
PLAYBOOKS = ROOT / "playbooks"
SKILLS = PLAYBOOKS / "skills"
TEMPLATES = PLAYBOOKS / "templates"
DESIGN = ROOT / "DESIGN.md"
CMD_DIR = REPO / "crates" / "engine" / "src" / "cmd"
MENUS = REPO / "crates" / "ui-egui" / "src" / "menus.rs"
CLI_MAIN = REPO / "apps" / "soundcraft-cli" / "src" / "main.rs"

EXPECTED_SKILLS = [
    "studio-verified-edit",
    "studio-sample-composition",
    "studio-playable-instrument",
    "studio-sing-to-midi",
    "studio-section-ops",
    "studio-plugin-profiling",
    "studio-session-conventions",
]
SKILL_FILES = [SKILLS / name / "SKILL.md" for name in EXPECTED_SKILLS]
DOC_FILES = [*SKILL_FILES, TEMPLATES / "CLAUDE.md", PLAYBOOKS / "README.md"]
FILE_SUFFIXES = {"json", "md", "yaml", "yml", "py", "wav", "sfz", "dspreset", "scraft", "mid", "zip", "toml", "txt", "pdf", "script"}
# Fields of Forge sidecars and results that read like ids ("requested.note").
FIELD_PREFIXES = ("requested.", "measured.", "result.")

ID_PATTERN = r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+"
ID_LITERAL = re.compile(rf'"({ID_PATTERN})"')
ID_TOKEN = re.compile(rf"(?<![\w./<>$-])({ID_PATTERN})(?![\w/-])")
FENCE = re.compile(r"^```[^\n]*\n(.*?)^```[ \t]*$", re.S | re.M)
INLINE = re.compile(r"`([^`\n]+)`")
URL = re.compile(r"https?://\S+")


def _load_new_project() -> ModuleType:
    spec = importlib.util.spec_from_file_location("agent_studio_new_project", PLAYBOOKS / "new_project.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


np = _load_new_project()


# ---- helpers -------------------------------------------------------------------------------------


def code_spans(text: str) -> list[str]:
    """Fenced-block contents and inline code spans of a Markdown text."""
    fenced = FENCE.findall(text)
    rest = FENCE.sub("", text)
    return [*fenced, *INLINE.findall(rest)]


def dotted_tokens(text: str) -> set[str]:
    """Everything in code spans that looks like a command id (``track.new``)."""
    found: set[str] = set()
    for span in code_spans(text):
        found.update(ID_TOKEN.findall(URL.sub("", span)))
    return found


def design_section(start: str, end: str) -> str:
    text = DESIGN.read_text(encoding="utf-8")
    return text.split(start, 1)[1].split(end, 1)[0]


def rust_ids() -> set[str]:
    ids: set[str] = set()
    for path in [*sorted(CMD_DIR.glob("*.rs")), MENUS]:
        ids.update(ID_LITERAL.findall(path.read_text(encoding="utf-8")))
    return ids


def design_ids() -> set[str]:
    """Command ids named in DESIGN.md section 9 (the Rust additions the playbooks depend on)."""
    section = DESIGN.read_text(encoding="utf-8").split("## 9. Studio command contracts", 1)[1]
    return dotted_tokens(section)


def forge_job_types() -> set[str]:
    return set(re.findall(r"^\| `([a-z]+\.[a-z]+)` \|", design_section("### 3.4", "### 3.5"), flags=re.M))


def is_file_name(token: str) -> bool:
    return token.rsplit(".", 1)[-1] in FILE_SUFFIXES


def unknown_ids(text: str, known: set[str]) -> list[str]:
    allowed = known | forge_job_types()
    return sorted(
        t for t in dotted_tokens(text)
        if t not in allowed and not is_file_name(t) and not t.startswith(FIELD_PREFIXES)
    )


needs_rust = pytest.mark.skipif(not CMD_DIR.is_dir(), reason="the SoundCraft Rust workspace is not next to this project")


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    assert match, "a SKILL.md starts with a --- front matter block"
    data = yaml.safe_load(match.group(1))
    assert isinstance(data, dict)
    return data, match.group(2)


# ---- skills --------------------------------------------------------------------------------------


def test_the_seven_skills_exist() -> None:
    folders = sorted(p.name for p in SKILLS.iterdir() if p.is_dir())
    assert folders == sorted(EXPECTED_SKILLS)
    for path in SKILL_FILES:
        assert path.is_file(), path


@pytest.mark.parametrize("path", SKILL_FILES, ids=EXPECTED_SKILLS)
def test_front_matter_is_valid(path: Path) -> None:
    meta, body = split_front_matter(path.read_text(encoding="utf-8"))
    assert set(meta) == {"name", "description"}
    assert meta["name"] == path.parent.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"]) and len(meta["name"]) <= 64
    description = meta["description"]
    assert isinstance(description, str) and description.strip()
    assert len(description) <= 1024
    assert description.startswith("Use "), "the description says when to use the skill"
    assert body.lstrip().startswith("# "), "the body starts with a heading"


@pytest.mark.parametrize("path", SKILL_FILES, ids=EXPECTED_SKILLS)
def test_skill_is_concise_and_clean(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 120
    assert "\t" not in text
    assert "{{" not in text
    assert "—" not in text, "no em dashes"


@pytest.mark.parametrize("path", DOC_FILES, ids=[p.name if p.name != "SKILL.md" else p.parent.name for p in DOC_FILES])
def test_skill_references_resolve(path: Path) -> None:
    """Every `studio-...` name in backticks is a skill that exists."""
    names = {n for span in code_spans(path.read_text(encoding="utf-8")) for n in re.findall(r"(?<![\w-])studio-[a-z-]+", span)}
    assert names <= set(EXPECTED_SKILLS), sorted(names - set(EXPECTED_SKILLS))


@needs_rust
@pytest.mark.parametrize("path", DOC_FILES, ids=[p.name if p.name != "SKILL.md" else p.parent.name for p in DOC_FILES])
def test_command_ids_exist(path: Path) -> None:
    known = rust_ids() | design_ids()
    assert unknown_ids(path.read_text(encoding="utf-8"), known) == []


@needs_rust
def test_id_sources_are_read_correctly() -> None:
    """Guards the check above: if the extraction broke, everything would pass or fail for the wrong reason."""
    ids = rust_ids()
    assert len(ids) > 300
    assert {"track.new", "mix.insert", "file.bounce_mix", "plugin.describe", "plugin.param_text", "audio.analyze"} <= ids
    assert {"midi.from_audio", "mix.sampler_zones_set", "file.import_sfz", "session.checkpoint", "mix.insert_sidechain"} <= design_ids()
    assert {"generate.loop", "transcribe.melody", "analyze.audio", "layer.add"} <= forge_job_types()
    assert unknown_ids("`track.nwe {}` and `mix.volume {}`", ids) == ["track.nwe"]
    assert unknown_ids("`forge.yaml`, `generate.sfx` and `requested.note`", ids) == []


@needs_rust
def test_skills_use_the_commands_they_promise() -> None:
    """The playbooks lean on the Agent Studio commands; each is named by at least one skill."""
    text = "\n".join(p.read_text(encoding="utf-8") for p in SKILL_FILES)
    used = dotted_tokens(text)
    for needed in ["plugin.describe", "plugin.param_text", "audio.analyze", "midi.from_audio", "mix.sampler_zones_set",
                   "file.import_sfz", "file.import_dspreset", "session.checkpoint", "session.checkpoints", "view.reveal",
                   "mix.insert_sidechain", "file.bounce_mix", "edit.separate_at_transients", "mix.sampler_load"]:
        assert needed in used, needed


def test_skills_name_the_forge_and_atlas_tools() -> None:
    text = "\n".join(p.read_text(encoding="utf-8") for p in SKILL_FILES)
    for tool in ["forge_health", "forge_models", "forge_submit", "forge_job", "forge_cancel", "forge_fetch",
                 "forge_upload", "forge_search", "forge_rate", "atlas_list", "atlas_get", "atlas_validate",
                 "atlas_profile", "atlas_merge"]:
        assert tool in text, tool


def test_plugin_profiling_documents_param_text_units() -> None:
    text = (SKILLS / "studio-plugin-profiling" / "SKILL.md").read_text(encoding="utf-8")
    assert "normalized" in text and "text_source" in text
    assert "parameter's own units" in text


# ---- templates -----------------------------------------------------------------------------------


def test_mcp_template_is_valid_json_with_three_servers() -> None:
    cfg = json.loads((TEMPLATES / "mcp.json").read_text(encoding="utf-8"))
    assert set(cfg["mcpServers"]) == {"soundcraft", "forge", "atlas"}


def test_mcp_template_renders_with_explicit_values(tmp_path: Path) -> None:
    project = tmp_path / "my song"
    cfg = json.loads(np.render_mcp_json(project, 7801, "http://spark.tailnet:7700"))
    servers = cfg["mcpServers"]
    assert servers["soundcraft"]["command"] == "soundcraft-cli"
    assert servers["soundcraft"]["args"] == ["mcp", "--connect", "7801"]
    assert servers["forge"]["command"] == "forge-mcp"
    assert servers["forge"]["env"]["FORGE_URL"] == "http://spark.tailnet:7700"
    assert servers["forge"]["env"]["FORGE_TOKEN"] == "${FORGE_TOKEN}"
    assert servers["forge"]["env"]["AGENT_STUDIO_PROJECT"] == str(project)
    assert servers["atlas"]["command"] == "atlas-mcp"


def test_mcp_template_defaults_read_the_environment(tmp_path: Path) -> None:
    cfg = json.loads(np.render_mcp_json(tmp_path, None, None))
    assert cfg["mcpServers"]["soundcraft"]["args"] == ["mcp", "--connect", "${SOUNDCRAFT_PORT:-7801}"]
    assert cfg["mcpServers"]["forge"]["env"]["FORGE_URL"] == "${FORGE_URL:-http://127.0.0.1:7700}"
    assert "{{" not in json.dumps(cfg)


def test_mcp_template_never_holds_a_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_TOKEN", "s3cret-token-value")
    text = np.render_mcp_json(tmp_path, 7801, "http://spark:7700")
    assert "s3cret-token-value" not in text
    env = json.loads(text)["mcpServers"]["forge"]["env"]
    assert env["FORGE_TOKEN"] == "${FORGE_TOKEN}"
    assert [k for k, v in env.items() if "token" in k.lower() and v != "${FORGE_TOKEN}"] == []
    assert "bearer" not in text.lower()


def test_soundcraft_cli_really_has_connect() -> None:
    if not CLI_MAIN.is_file():
        pytest.skip("soundcraft-cli sources not found")
    source = CLI_MAIN.read_text(encoding="utf-8")
    assert '"--connect"' in source and 'fn mcp(' in source


def test_claude_md_template_covers_the_rules() -> None:
    text = (TEMPLATES / "CLAUDE.md").read_text(encoding="utf-8")
    for placeholder in ["{{PROJECT}}", "{{SOUNDCRAFT_PORT}}", "{{FORGE_URL}}"]:
        assert placeholder in text
    for phrase in ["-1 dBFS", "checkpoint", "No live sweeps", "Profile first", "old to new", "Do not claim"]:
        assert phrase.lower() in text.lower(), phrase
    rendered = np.render_claude_md(Path("/music/song"), 7801, "http://spark:7700")
    assert "{{" not in rendered and "/music/song" in rendered and "7801" in rendered and "http://spark:7700" in rendered


# ---- new_project.py ------------------------------------------------------------------------------


def test_template_session_uses_real_commands() -> None:
    commands = np.template_session_commands()
    ids = {cid for cid, _ in commands}
    assert {"track.new", "track.color", "markers.add", "mix.insert"} <= ids
    names = [p["name"] for cid, p in commands if cid == "track.new"]
    assert names[:5] == ["DRM", "BAS", "SYN", "FX", "VOX"]
    assert [p["name"] for cid, p in commands if cid == "markers.add"] == ["Intro", "Verse", "Chorus", "Drop", "Outro"]
    limiter = [p for cid, p in commands if cid == "mix.insert"]
    assert len(limiter) == 1 and limiter[0]["plugin"] == "maximizer" and limiter[0]["track"] == "Master"
    assert limiter[0]["params"]["ceiling"] <= -1.0
    if CMD_DIR.is_dir():
        assert ids <= rust_ids()


def test_script_text_has_one_command_per_line() -> None:
    steps = np.template_steps()
    lines = np.script_text(steps).splitlines()
    assert len(lines) == len(steps)
    for line, step in zip(lines, steps):
        cid, params = line.split(" ", 1)
        assert cid == step.command and json.loads(params) == step.params


def test_creates_the_project_tree(tmp_path: Path) -> None:
    project = tmp_path / "song"
    result = np.create_project(project, studio_port=7801, forge_url="http://spark.tailnet:7700/")
    assert result["template"] is None and result["skipped"] == []
    assert result["skills"] == sorted(EXPECTED_SKILLS)
    for name in EXPECTED_SKILLS:
        copied = project / ".claude" / "skills" / name / "SKILL.md"
        assert copied.read_text(encoding="utf-8") == (SKILLS / name / "SKILL.md").read_text(encoding="utf-8")
    for sub in ["Forge Inbox", "Previews", "Checkpoints"]:
        assert (project / sub).is_dir()
    claude_md = (project / "CLAUDE.md").read_text(encoding="utf-8")
    assert str(project.resolve()) in claude_md and "http://spark.tailnet:7700" in claude_md and "7801" in claude_md
    assert "{{" not in claude_md
    cfg = json.loads((project / ".mcp.json").read_text(encoding="utf-8"))
    assert set(cfg["mcpServers"]) == {"soundcraft", "forge", "atlas"}
    assert cfg["mcpServers"]["forge"]["env"]["FORGE_URL"] == "http://spark.tailnet:7700"
    assert cfg["mcpServers"]["forge"]["env"]["AGENT_STUDIO_PROJECT"] == str(project.resolve())
    assert not (project / "Template.scraft").exists()
    assert not list(project.rglob(".*.tmp*")), "no temp files are left behind"


def test_main_creates_a_project_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert np.main([str(tmp_path / "song"), "--studio-port", "7802"]) == 0
    out = capsys.readouterr().out
    assert "skills:" in out and "CLAUDE.md" in out and "not created" in out
    cfg = json.loads((tmp_path / "song" / ".mcp.json").read_text(encoding="utf-8"))
    assert cfg["mcpServers"]["soundcraft"]["args"][-1] == "7802"


def test_an_empty_existing_folder_is_fine(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    np.create_project(tmp_path / "empty")
    assert (tmp_path / "empty" / "CLAUDE.md").is_file()


def test_refuses_a_non_empty_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "busy"
    project.mkdir()
    (project / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(np.ProjectError, match="not empty"):
        np.create_project(project)
    assert sorted(p.name for p in project.iterdir()) == ["notes.txt"], "nothing was written"
    assert np.main([str(project)]) == 1
    assert "not empty" in capsys.readouterr().err
    assert (project / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_a_second_run_is_refused_too(tmp_path: Path) -> None:
    np.create_project(tmp_path / "song")
    with pytest.raises(np.ProjectError, match="not empty"):
        np.create_project(tmp_path / "song")


def test_force_overwrites_only_its_own_files(tmp_path: Path) -> None:
    project = tmp_path / "song"
    np.create_project(project, studio_port=7801)
    (project / "CLAUDE.md").write_text("old", encoding="utf-8")
    (project / "my-take.wav").write_bytes(b"RIFF")
    np.create_project(project, studio_port=7900, force=True)
    assert "7900" in (project / "CLAUDE.md").read_text(encoding="utf-8")
    assert json.loads((project / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["soundcraft"]["args"][-1] == "7900"
    assert (project / "my-take.wav").read_bytes() == b"RIFF"


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks")
def test_force_replaces_a_symlink_instead_of_writing_through_it(tmp_path: Path) -> None:
    project = tmp_path / "song"
    project.mkdir()
    target = tmp_path / "elsewhere.md"
    target.write_text("untouched", encoding="utf-8")
    (project / "CLAUDE.md").symlink_to(target)
    np.create_project(project, force=True)
    assert target.read_text(encoding="utf-8") == "untouched"
    assert not (project / "CLAUDE.md").is_symlink()


def test_refuses_a_file_in_the_way(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x", encoding="utf-8")
    with pytest.raises(np.ProjectError, match="not a folder"):
        np.create_project(tmp_path / "file", force=True)


@pytest.mark.parametrize("kwargs, message", [
    ({"studio_port": 0}, "1..65535"),
    ({"studio_port": 70000}, "1..65535"),
    ({"forge_url": "ftp://spark"}, "http"),
    ({"forge_url": "http://user:pw@spark:7700"}, "credentials"),
    ({"forge_url": 'http://spark/"x'}, "quotes"),
    ({"forge_url": "http://${FORGE_TOKEN}@spark"}, "\\$"),
    ({"soundcraft_cli": "/no/such/soundcraft-cli"}, "is not a file"),
])
def test_rejects_bad_arguments_before_writing(tmp_path: Path, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(np.ProjectError, match=message):
        np.create_project(tmp_path / "song", **kwargs)
    assert not (tmp_path / "song").exists()


# ---- the headless build, against a stand-in binary -----------------------------------------------

FAKE_CLI = """#!{python}
import sys
mode = {mode!r}
args = sys.argv[1:]
if args[0] == "script":
    script, out = args[1], args[args.index("--save") + 1]
    if mode == "broken":
        print("soundcraft-cli: boom", file=sys.stderr)
        sys.exit(1)
    for n, line in enumerate(open(script, encoding="utf-8").read().splitlines(), 1):
        cid = line.split(" ", 1)[0]
        if (cid == "markers.add" and '"Drop"' in line) or cid == "track.rename":
            print(f"{{script}}:{{n}}: {{cid}}: bad parameters for `{{cid}}`: pretend failure", file=sys.stderr)
    open(out, "w", encoding="utf-8").write('{{"fake": true}}')
elif args[0] == "run":
    if mode == "unopenable":
        print("soundcraft-cli: bad session", file=sys.stderr)
        sys.exit(1)
    print("{{}}")
"""


def fake_cli(tmp_path: Path, mode: str) -> Path:
    path = tmp_path / f"fake-soundcraft-cli-{mode}"
    path.write_text(FAKE_CLI.format(python=sys.executable, mode=mode), encoding="utf-8")
    path.chmod(0o755)
    return path


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in binary is a script with a shebang")


@posix_only
def test_template_build_reports_skipped_steps(tmp_path: Path) -> None:
    result = np.create_project(tmp_path / "song", soundcraft_cli=fake_cli(tmp_path, "partial"))
    assert result["template"] == str((tmp_path / "song" / "Template.scraft").resolve())
    assert (tmp_path / "song" / "Template.scraft").read_text(encoding="utf-8") == '{"fake": true}'
    # the failed required step is reported, the failed optional rename is not
    assert len(result["skipped"]) == 1 and result["skipped"][0].startswith("markers.add: bad parameters")


@posix_only
def test_main_prints_skipped_steps(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert np.main([str(tmp_path / "song"), "--soundcraft-cli", str(fake_cli(tmp_path, "partial"))]) == 0
    out = capsys.readouterr().out
    assert "template session:" in out and "template step skipped: markers.add" in out


@posix_only
@pytest.mark.parametrize("mode, message", [("broken", "boom"), ("unopenable", "does not open")])
def test_a_failed_template_build_leaves_no_project(tmp_path: Path, mode: str, message: str) -> None:
    with pytest.raises(np.ProjectError, match=message):
        np.create_project(tmp_path / "song", soundcraft_cli=fake_cli(tmp_path, mode))
    assert not (tmp_path / "song").exists()


# ---- the real binary -----------------------------------------------------------------------------

REAL_CLI = os.environ.get("SOUNDCRAFT_CLI", "")


@pytest.mark.skipif(not REAL_CLI, reason="set SOUNDCRAFT_CLI to a soundcraft-cli binary")
def test_real_binary_builds_a_template_session_that_opens(tmp_path: Path) -> None:
    project = tmp_path / "song"
    result = np.create_project(project, studio_port=7801, soundcraft_cli=REAL_CLI)
    session = project / "Template.scraft"
    assert session.is_file() and result["template"] == str(session.resolve())
    assert result["skipped"] == [], result["skipped"]

    proc = subprocess.run([REAL_CLI, "run", "--in", str(session), "--inspect"], capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    state = json.loads(proc.stdout[proc.stdout.index("{"):])
    tracks = {t["name"]: t for t in state["tracks"]}
    assert list(tracks)[:5] == ["DRM", "BAS", "SYN", "FX", "VOX"]
    assert all(tracks[n]["kind"] == "audio" for n in ["DRM", "BAS", "SYN", "FX", "VOX"])
    saved = json.loads(session.read_text(encoding="utf-8"))  # a .scraft file is JSON
    assert saved["name"] == "Template"
    assert [t["color"] for t in saved["tracks"][:5]] == [colour for _, _, colour in np.TRACKS]
    assert tracks["Master"]["kind"] == "master"
    assert [i["plugin"] for i in tracks["Master"]["inserts"] if i] == ["maximizer"]
    assert [m["name"] for m in state["markers"]] == ["Intro", "Verse", "Chorus", "Drop", "Outro"]

    desc = subprocess.run(
        [REAL_CLI, "run", "--in", str(session), "--cmd", 'plugin.describe={"track":"Master","slot":0}'],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert desc.returncode == 0, desc.stderr
    body = json.loads(desc.stdout.split("plugin.describe: ", 1)[1])
    values = {p["id"]: p["value"] for p in body["params"]}
    assert values["ceiling"] == pytest.approx(-1.2) and values["threshold"] == pytest.approx(0.0)
