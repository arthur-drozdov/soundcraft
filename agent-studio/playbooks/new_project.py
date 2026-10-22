#!/usr/bin/env python3
"""Create an Agent Studio music project folder.

    python new_project.py <folder> [--studio-port N] [--forge-url URL]
                          [--soundcraft-cli PATH] [--force]

The folder gets ``.claude/skills/`` (copied from ``skills/``), ``CLAUDE.md`` and ``.mcp.json``
(rendered from ``templates/``), ``Forge Inbox/``, ``Previews/``, ``Checkpoints/`` and, when
``--soundcraft-cli`` is given, ``Template.scraft`` built headlessly (role-named tracks, section
markers, a limiter on the master). The Forge token is never written: ``.mcp.json`` keeps the
``${FORGE_TOKEN}`` reference. An existing folder that is not empty is refused unless ``--force``
is given; ``--force`` overwrites only the files this script creates and deletes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
SKILLS_DIR = HERE / "skills"
TEMPLATES_DIR = HERE / "templates"
ATLAS_PROFILES_DIR = HERE.parent / "atlas" / "profiles"
SESSION_FILE = "Template.scraft"
SUBDIRS = ("Forge Inbox", "Previews", "Checkpoints")
CLI_TIMEOUT_S = 120
DEFAULT_PORT = 7801
DEFAULT_FORGE_URL = "http://127.0.0.1:7700"


class ProjectError(Exception):
    """A user-facing problem (bad argument, folder in the way, headless build failed)."""


# ---- the template session ---------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One headless command. ``optional`` steps are cosmetic: their failure is not reported."""

    command: str
    params: dict[str, Any] = field(default_factory=dict)
    optional: bool = False


# Role-named tracks: (name, channel format, colour). Audio tracks, so generated loops and
# one-shots drop straight in; instrument tracks are added on demand (studio-playable-instrument).
TRACKS: list[tuple[str, str, list[int]]] = [
    ("DRM", "stereo", [220, 80, 70]),
    ("BAS", "stereo", [80, 120, 220]),
    ("SYN", "stereo", [240, 190, 60]),
    ("FX", "stereo", [150, 90, 200]),
    ("VOX", "mono", [90, 190, 120]),
]
# Section markers: (name, bar). Whole-bar sections of 4 or 8 bars.
SECTIONS: list[tuple[str, int]] = [("Intro", 1), ("Verse", 5), ("Chorus", 13), ("Drop", 21), ("Outro", 29)]
LIMITER_PLUGIN = "maximizer"
# Ceiling 0.2 dB under the -1 dBFS preview cap, so inter-sample (true) peaks stay under the cap too.
LIMITER_PARAMS: dict[str, float] = {"threshold": 0.0, "ceiling": -1.2}


def template_steps() -> list[Step]:
    """The commands that build the template session, in order."""
    steps: list[Step] = []
    for name, fmt, colour in TRACKS:
        steps.append(Step("track.new", {"kind": "audio", "name": name, "format": fmt}))
        steps.append(Step("track.color", {"tracks": [name], "color": colour}))
    for name, bar in SECTIONS:
        steps.append(Step("markers.add", {"name": name, "at": f"bars_beats:{bar}|1|000"}))
    # "Master" is a reserved base name, so the new track comes out as "Master 1": rename it.
    steps.append(Step("track.new", {"kind": "master", "name": "Master", "format": "stereo"}))
    steps.append(Step("track.rename", {"track": "Master 1", "name": "Master"}, optional=True))
    steps.append(Step("mix.insert", {"track": "Master", "slot": 0, "plugin": LIMITER_PLUGIN, "params": LIMITER_PARAMS}))
    return steps


def template_session_commands() -> list[tuple[str, dict[str, Any]]]:
    """``(command id, params)`` pairs of :func:`template_steps`."""
    return [(s.command, s.params) for s in template_steps()]


def script_text(steps: list[Step]) -> str:
    """``soundcraft-cli script`` format: one ``ID JSON`` per line (no comments, so line n is step n)."""
    return "".join(f"{s.command} {json.dumps(s.params, separators=(',', ':'))}\n" for s in steps)


def resolve_binary(value: str | Path) -> Path:
    """The ``soundcraft-cli`` executable named by ``--soundcraft-cli`` (a path, or a name on PATH)."""
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    found = shutil.which(str(value))
    if found:
        return Path(found).resolve()
    raise ProjectError(f"--soundcraft-cli {value} is not a file")


def _run_cli(binary: Path, args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(binary), *args], capture_output=True, text=True, timeout=CLI_TIMEOUT_S, check=False, cwd=cwd,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProjectError(f"could not run {binary}: {exc}") from exc


def build_template_session(binary: Path, workdir: Path) -> tuple[Path, list[str]]:
    """Build ``Template.scraft`` in ``workdir`` with the headless CLI.

    Returns ``(session path, skipped)`` where ``skipped`` lists the required steps the binary
    rejected (a command it does not have, for example); they are left out of the session.
    """
    steps = template_steps()
    script = workdir / "template.script"
    script.write_text(script_text(steps), encoding="utf-8")
    out = workdir / SESSION_FILE
    proc = _run_cli(binary, ["script", str(script), "--keep-going", "--save", str(out)], workdir)
    if proc.returncode != 0 or not out.is_file():
        detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit code {proc.returncode}"
        raise ProjectError(f"soundcraft-cli failed to build the template session: {detail}")
    skipped: list[str] = []
    failed = re.compile(rf"^{re.escape(str(script))}:(\d+): (.*)$")
    for line in proc.stderr.splitlines():
        match = failed.match(line.strip())
        if match is None:
            if line.strip():
                skipped.append(line.strip())
            continue
        index = int(match.group(1)) - 1
        if 0 <= index < len(steps) and steps[index].optional:
            continue
        skipped.append(match.group(2))
    verify = _run_cli(binary, ["run", "--in", str(out), "--inspect"], workdir)
    if verify.returncode != 0:
        raise ProjectError(f"the template session does not open: {verify.stderr.strip() or verify.returncode}")
    return out, skipped


# ---- rendering ----------------------------------------------------------------------------------


def validate_port(port: int) -> int:
    if not 1 <= port <= 65535:
        raise ProjectError(f"--studio-port must be 1..65535, got {port}")
    return port


def validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or any(c in url for c in " \t\r\n\"'$"):
        raise ProjectError(f"--forge-url must be an http(s) URL without spaces, quotes or $, got {url!r}")
    if parsed.username or parsed.password:
        raise ProjectError("--forge-url must not contain credentials; use the FORGE_TOKEN environment variable")
    return url.rstrip("/")


def _substitute(node: Any, values: dict[str, str]) -> Any:
    if isinstance(node, str):
        for key, value in values.items():
            node = node.replace("{{" + key + "}}", value)
        return node
    if isinstance(node, list):
        return [_substitute(item, values) for item in node]
    if isinstance(node, dict):
        return {key: _substitute(value, values) for key, value in node.items()}
    return node


def render_claude_md(project: Path, port: int | None, forge_url: str | None) -> str:
    text = (TEMPLATES_DIR / "CLAUDE.md").read_text(encoding="utf-8")
    values = {
        "PROJECT": str(project),
        "SOUNDCRAFT_PORT": str(port) if port is not None else f"$SOUNDCRAFT_PORT, default {DEFAULT_PORT}",
        "FORGE_URL": forge_url if forge_url else f"$FORGE_URL, default {DEFAULT_FORGE_URL}",
    }
    return _substitute(text, values)


def render_mcp_json(project: Path, port: int | None, forge_url: str | None) -> str:
    """``.mcp.json`` text. Unset port / URL become ``${VAR:-default}`` so Claude Code reads the environment."""
    cfg = json.loads((TEMPLATES_DIR / "mcp.json").read_text(encoding="utf-8"))
    values = {
        "PROJECT": str(project),
        "SOUNDCRAFT_PORT": str(port) if port is not None else f"${{SOUNDCRAFT_PORT:-{DEFAULT_PORT}}}",
        "FORGE_URL": forge_url if forge_url else f"${{FORGE_URL:-{DEFAULT_FORGE_URL}}}",
        "ATLAS_PROFILES": str(ATLAS_PROFILES_DIR) if ATLAS_PROFILES_DIR.is_dir() else "",
    }
    cfg = _substitute(cfg, values)
    atlas = cfg["mcpServers"]["atlas"]
    if not atlas.get("env", {}).get("ATLAS_PROFILES"):
        atlas.pop("env", None)
    return json.dumps(cfg, indent=2) + "\n"


# ---- writing ------------------------------------------------------------------------------------


def _replace_file(tmp: Path, dest: Path) -> None:
    """Move ``tmp`` over ``dest``; a symlink at ``dest`` is replaced, never followed."""
    os.replace(tmp, dest)


def atomic_write(dest: Path, text: str) -> None:
    tmp = dest.with_name(f".{dest.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    _replace_file(tmp, dest)


def atomic_copy(src: Path, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.tmp{os.getpid()}")
    shutil.copyfile(src, tmp)
    _replace_file(tmp, dest)


def skill_names() -> list[str]:
    return sorted(p.name for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file())


def copy_skills(project: Path) -> list[str]:
    names = skill_names()
    for name in names:
        dest = project / ".claude" / "skills" / name
        dest.mkdir(parents=True, exist_ok=True)
        atomic_copy(SKILLS_DIR / name / "SKILL.md", dest / "SKILL.md")
    return names


def check_target(project: Path, force: bool) -> None:
    """Refuse a path that is a file, or a non-empty folder without ``force``."""
    if project.exists() and not project.is_dir():
        raise ProjectError(f"{project} exists and is not a folder")
    if project.is_dir() and not force and any(project.iterdir()):
        raise ProjectError(f"{project} is not empty; pick a new folder, or pass --force to write the project files into it")


def create_project(
    folder: str | Path,
    studio_port: int | None = None,
    forge_url: str | None = None,
    soundcraft_cli: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create the project tree. Returns ``{project, skills, written, template, skipped}``."""
    project = Path(folder).expanduser().resolve()
    port = validate_port(studio_port) if studio_port is not None else None
    url = validate_url(forge_url) if forge_url else None
    binary = resolve_binary(soundcraft_cli) if soundcraft_cli else None
    check_target(project, force)
    claude_md = render_claude_md(project, port, url)
    mcp_json = render_mcp_json(project, port, url)

    template: Path | None = None
    skipped: list[str] = []
    with tempfile.TemporaryDirectory(prefix="new-project-") as tmp:
        built: Path | None = None
        if binary is not None:  # build first, so a failure leaves nothing behind
            built, skipped = build_template_session(binary, Path(tmp))
        try:
            project.mkdir(parents=True, exist_ok=True)
            for sub in SUBDIRS:
                (project / sub).mkdir(exist_ok=True)
            skills = copy_skills(project)
            atomic_write(project / "CLAUDE.md", claude_md)
            atomic_write(project / ".mcp.json", mcp_json)
            if built is not None:
                template = project / SESSION_FILE
                atomic_copy(built, template)
        except OSError as exc:
            raise ProjectError(f"cannot write {project}: {exc}") from exc
    written = ["CLAUDE.md", ".mcp.json"] + ([SESSION_FILE] if template else [])
    return {
        "project": str(project), "skills": skills, "written": written,
        "template": str(template) if template else None, "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create an Agent Studio music project folder.")
    ap.add_argument("folder", help="project folder (created if missing; must be empty or new unless --force)")
    ap.add_argument("--studio-port", type=int, help=f"SoundCraft control port (default: $SOUNDCRAFT_PORT, else {DEFAULT_PORT}, read when Claude Code starts)")
    ap.add_argument("--forge-url", help=f"Forge base URL (default: $FORGE_URL, else {DEFAULT_FORGE_URL})")
    ap.add_argument("--soundcraft-cli", help="path to soundcraft-cli; builds Template.scraft headlessly")
    ap.add_argument("--force", action="store_true", help="write into a non-empty folder, overwriting the files this script creates")
    args = ap.parse_args(argv)
    try:
        result = create_project(args.folder, args.studio_port, args.forge_url, args.soundcraft_cli, args.force)
    except ProjectError as exc:
        print(f"new_project: {exc}", file=sys.stderr)
        return 1
    print(f"project: {result['project']}")
    print(f"skills: {', '.join(result['skills'])}")
    print(f"written: {', '.join(result['written'])}")
    if result["template"]:
        print(f"template session: {result['template']}")
    else:
        print("template session: not created (pass --soundcraft-cli PATH to build it)")
    for item in result["skipped"]:
        print(f"template step skipped: {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
