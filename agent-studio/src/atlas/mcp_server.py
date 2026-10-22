"""``atlas-mcp``: the Plugin Atlas as a stdio MCP server (DESIGN 6.3).

Tools:

* ``atlas_list {query?}`` - the profiles under the profiles root.
* ``atlas_get {plugin_id, version?}`` - the profile plus whether it was made for that plugin version.
* ``atlas_validate {path}`` - schema and rule check of a profile file or a folder of them.
* ``atlas_profile {plugin_id, port?}`` - run the profiler on a scratch session of a running
  SoundCraft and return the path of the draft (written under ``<project>/atlas-drafts``).
* ``atlas_merge {profile_path, patch}`` - apply a knowledge patch (mapping or YAML text) to a profile.

Environment: ``ATLAS_PROFILES`` (profiles root, default the checkout's ``atlas/profiles``),
``AGENT_STUDIO_PROJECT`` (project folder, default the working directory), ``SOUNDCRAFT_CONTROL_PORT``
(default control port for ``atlas_profile``) and ``SOUNDCRAFT_CLI`` (binary for headless profiling,
default ``soundcraft-cli``). Every path an agent passes must resolve (links followed) inside the
profiles root or the project folder.

The tool logic lives in ``AtlasTools`` (plain methods, usable and testable without the MCP SDK);
``build_server`` wraps it with the SDK (``MCPServer`` in mcp 2.x, ``FastMCP`` in 1.x).
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import profiles, validate, yamlio
from .cli import default_profiles_root
from .merge import MergeError, merge_patch
from .patch import check_patch
from .validate import ProfileError

log = logging.getLogger("atlas.mcp")

MAX_PATH_CHARS = 4096
MAX_PATCH_BYTES = 1024 * 1024
MAX_VALIDATE_FILES = 500
MAX_WARNINGS = 40
DEFAULT_PORT_ENV = "SOUNDCRAFT_CONTROL_PORT"
BINARY_ENV = "SOUNDCRAFT_CLI"

StudioFactory = Callable[[int | None], Any]
"""``port`` (or ``None`` for headless) -> an object with ``execute``, ``inspect``, ``kind`` and ``close``."""


class AtlasToolError(ValueError):
    """A tool call that cannot be done; the message tells the agent what to change."""


def default_studio(port: int | None) -> Any:
    """A control-channel Studio for ``port``, else a headless one from ``$SOUNDCRAFT_CLI``."""
    from studio_client import Studio

    if port is not None:
        return Studio.connect(port)
    return Studio.headless(os.environ.get(BINARY_ENV) or "soundcraft-cli")


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class AtlasTools:
    """The five Atlas tools as methods, with path confinement."""

    def __init__(
        self,
        profiles_root: str | Path | None = None,
        project_dir: str | Path | None = None,
        studio_factory: StudioFactory | None = None,
    ) -> None:
        self.profiles_root = Path(profiles_root) if profiles_root else default_profiles_root()
        env_project = os.environ.get("AGENT_STUDIO_PROJECT")
        self.project_dir = Path(project_dir) if project_dir else Path(env_project) if env_project else Path.cwd()
        self.studio_factory = studio_factory or default_studio

    # ---- paths -----------------------------------------------------------------------------------

    @property
    def roots(self) -> list[Path]:
        return [self.profiles_root.resolve(), self.project_dir.resolve()]

    @property
    def drafts_dir(self) -> Path:
        return self.project_dir.resolve() / "atlas-drafts"

    def resolve_path(self, path: Any) -> Path:
        """An existing path inside the profiles root or the project folder (links followed)."""
        if not isinstance(path, str) or not path.strip() or len(path) > MAX_PATH_CHARS or "\x00" in path:
            raise AtlasToolError("path must be a non-empty string")
        given = Path(path)
        candidates = [given] if given.is_absolute() else [self.project_dir / given, self.profiles_root / given]
        roots = self.roots
        outside = False
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if any(_inside(resolved, root) for root in roots):
                return resolved
            outside = True
        if outside or given.is_absolute():
            raise AtlasToolError(f"{path}: outside the profiles folder ({roots[0]}) and the project folder ({roots[1]}), or not found there")
        raise AtlasToolError(f"{path}: not found in the project folder or the profiles folder")

    def _shown(self, path: Path) -> str:
        for root in self.roots:
            if _inside(path, root):
                return path.relative_to(root).as_posix() if root == self.roots[0] else str(path)
        return str(path)

    # ---- atlas_list / atlas_get --------------------------------------------------------------------

    def atlas_list(self, query: str | None = None) -> dict[str, Any]:
        """List profiles; ``query`` is a case-insensitive text matched against name, vendor, id and summary."""
        if query is not None and (not isinstance(query, str) or len(query) > 200):
            raise AtlasToolError("query must be a string of at most 200 characters")
        needle = query.strip().lower() if query else ""
        rows: list[dict[str, Any]] = []
        for ref in profiles.list_profiles(self.profiles_root):
            ident = ref.profile["identity"]
            hay = " ".join([ident["name"], ident["vendor"], ident["plugin_id"], str(ref.profile.get("summary", ""))]).lower()
            if needle and needle not in hay:
                continue
            rows.append(
                {
                    "plugin_id": ident["plugin_id"], "name": ident["name"], "vendor": ident["vendor"], "format": ident["format"],
                    "version": ident["version"], "params": len(ref.profile["params"]), "path": self._shown(ref.path),
                }
            )
        return {"count": len(rows), "profiles": rows}

    def atlas_get(self, plugin_id: str, version: str | None = None) -> dict[str, Any]:
        """The profile of a plugin and, when ``version`` is given, whether it matches that version.

        ``version_match`` is true only for an exact version match; ``version_status`` adds
        ``same-major`` / ``different``. A plugin with no profile returns ``found: false`` (read
        its values only, or profile it with ``atlas_profile``).
        """
        if not isinstance(plugin_id, str) or not plugin_id.strip() or len(plugin_id) > 300:
            raise AtlasToolError("plugin_id must be a non-empty string")
        if version is not None and (not isinstance(version, str) or len(version) > 64):
            raise AtlasToolError("version must be a string of at most 64 characters")
        match = profiles.find_profile(self.profiles_root, plugin_id.strip(), version)
        if match is None:
            return {"found": False, "plugin_id": plugin_id, "message": "no profile for this plugin; read its values only, or run atlas_profile"}
        return {
            "found": True,
            "path": self._shown(match.path),
            "profile": match.profile,
            "version_match": match.version_match,
            "version_status": match.status,
            "profile_version": match.profile["identity"]["version"],
            "actual_version": version,
        }

    # ---- atlas_validate ------------------------------------------------------------------------------

    def atlas_validate(self, path: str) -> dict[str, Any]:
        """Validate a profile file, or every ``*.yaml`` profile file in a folder, against the schema and rules."""
        target = self.resolve_path(path)
        if target.is_dir():
            files = sorted(p for p in target.rglob("*.yaml") if p.is_file() and not p.is_symlink())[:MAX_VALIDATE_FILES]
        else:
            files = [target]
        results = []
        for file in files:
            issues = validate.validate_file(file)
            results.append({"path": self._shown(file), "ok": not issues, "issues": [{"path": i.path, "message": i.message} for i in issues]})
        if not results:
            raise AtlasToolError(f"{path}: no profile files found")
        return {"ok": all(r["ok"] for r in results), "checked": len(results), "files": results}

    # ---- atlas_profile ---------------------------------------------------------------------------------

    def atlas_profile(
        self,
        plugin_id: str,
        port: int | None = None,
        measure: bool = True,
        max_params: int = 128,
        allow_current_session: bool = False,
    ) -> dict[str, Any]:
        """Profile a plugin on a scratch session of SoundCraft and return the path of the draft.

        ``port`` is the control port of a running SoundCraft app (default ``$SOUNDCRAFT_CONTROL_PORT``);
        without any port a headless ``soundcraft-cli`` is started. A desktop app with a session
        open is refused unless ``allow_current_session`` is true (the app has one session only).
        """
        from studio_client import StudioError

        from .profiler import ProfilerError, ProfilerOptions, profile_plugin, write_draft

        if port is None and os.environ.get(DEFAULT_PORT_ENV):
            try:
                port = int(os.environ[DEFAULT_PORT_ENV])
            except ValueError as exc:
                raise AtlasToolError(f"${DEFAULT_PORT_ENV} is not a port number") from exc
        if port is not None and (isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535):
            raise AtlasToolError("port must be an integer in 1..65535")
        if isinstance(max_params, bool) or not isinstance(max_params, int) or not 0 <= max_params <= 4096:
            raise AtlasToolError("max_params must be an integer in 0..4096")
        options = ProfilerOptions(measure=bool(measure), max_params=max_params, allow_current_session=bool(allow_current_session))
        try:
            studio = self.studio_factory(port)
        except StudioError as exc:
            raise AtlasToolError(str(exc)) from exc
        try:
            result = profile_plugin(studio, plugin_id, options)
            draft = write_draft(result.profile, self.drafts_dir)
        except (ProfilerError, StudioError, ProfileError, OSError) as exc:
            raise AtlasToolError(str(exc)) from exc
        finally:
            try:
                studio.close()
            except StudioError as exc:
                log.warning("closing the studio failed: %s", exc)
        ident = result.profile["identity"]
        return {
            "draft_path": str(draft),
            "plugin_id": ident["plugin_id"],
            "name": ident["name"],
            "version": ident["version"],
            "params": len(result.profile["params"]),
            "measured": result.measured,
            "bounces": result.bounces,
            "scratch_session": result.scratch_mode,
            "elapsed_s": result.elapsed_s,
            "warnings": result.warnings[:MAX_WARNINGS],
            "next": "review the draft, then apply manual knowledge with atlas_merge; copy it into the profiles folder when approved",
        }

    # ---- atlas_merge -----------------------------------------------------------------------------------

    def atlas_merge(
        self,
        profile_path: str,
        patch: Mapping[str, Any] | str,
        skip_conflicts: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Apply a knowledge patch to a profile file in place, recording provenance for every field it sets.

        ``patch`` is a mapping, or YAML/JSON text, shaped like ``atlas merge`` patch files
        (``default_provenance``, top-level fields, ``params`` keyed by host param id). The merge is
        all-or-nothing and the result must validate.
        """
        path = self.resolve_path(profile_path)
        if not path.is_file():
            raise AtlasToolError(f"{profile_path}: not a file")
        if isinstance(patch, str):
            if len(patch.encode("utf-8", errors="replace")) > MAX_PATCH_BYTES:
                raise AtlasToolError(f"patch text is larger than {MAX_PATCH_BYTES} bytes")
            try:
                data = yamlio.loads(patch)
            except yamlio.YamlError as exc:
                raise AtlasToolError(f"patch is not valid YAML/JSON: {exc}") from exc
        else:
            data = patch
        issues = check_patch(data)
        if issues:
            raise AtlasToolError("invalid patch: " + "; ".join(str(i) for i in issues[:10]))
        try:
            profile = profiles.load_profile(path)
            result = merge_patch(profile, data, skip_conflicts=bool(skip_conflicts))
            if not dry_run:
                profiles.write_profile(path, result.profile)
        except MergeError as exc:
            raise AtlasToolError(f"merge refused: {exc}") from exc
        except (ProfileError, OSError) as exc:
            raise AtlasToolError(str(exc)) from exc
        return {
            "path": self._shown(path),
            "written": not dry_run,
            "changes": [{"path": c.path, "provenance": c.provenance} for c in result.changes],
            "skipped": [str(c) for c in result.conflicts],
        }


def _sdk_server_class() -> Any:
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2

        return MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # type: ignore[attr-defined]  # mcp 1.x

        return FastMCP


def _sdk_tool_error() -> type[Exception]:
    """The SDK's "expected failure" exception: its message reaches the agent (others are reported as crashes)."""
    try:
        from mcp.server.mcpserver.exceptions import ToolError as error_class  # mcp >= 2
    except ImportError:
        from mcp.server.fastmcp.exceptions import ToolError as error_class  # type: ignore[no-redef]  # mcp 1.x
    return error_class  # type: ignore[no-any-return]


def build_server(tools: AtlasTools | None = None) -> Any:
    """The MCP server object with the five Atlas tools registered."""
    import anyio.to_thread

    tools = tools or AtlasTools()
    tool_error = _sdk_tool_error()
    server = _sdk_server_class()(
        "atlas",
        instructions=(
            "Plugin Atlas: profiles of what plugins' parameters do. Call atlas_get before touching a plugin; with no "
            "profile, only read its values or use a built-in. Never trust a profile whose version_match is false: re-profile."
        ),
    )

    def call(fn: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
        try:
            return fn(*args)
        except AtlasToolError as exc:
            raise tool_error(str(exc)) from None

    @server.tool(name="atlas_list", description=AtlasTools.atlas_list.__doc__)
    def atlas_list(query: str | None = None) -> dict[str, Any]:
        return call(tools.atlas_list, query)

    @server.tool(name="atlas_get", description=AtlasTools.atlas_get.__doc__)
    def atlas_get(plugin_id: str, version: str | None = None) -> dict[str, Any]:
        return call(tools.atlas_get, plugin_id, version)

    @server.tool(name="atlas_validate", description=AtlasTools.atlas_validate.__doc__)
    def atlas_validate(path: str) -> dict[str, Any]:
        return call(tools.atlas_validate, path)

    @server.tool(name="atlas_profile", description=AtlasTools.atlas_profile.__doc__)
    async def atlas_profile(
        plugin_id: str,
        port: int | None = None,
        measure: bool = True,
        max_params: int = 128,
        allow_current_session: bool = False,
    ) -> dict[str, Any]:
        return await anyio.to_thread.run_sync(lambda: call(tools.atlas_profile, plugin_id, port, measure, max_params, allow_current_session))

    @server.tool(name="atlas_merge", description=AtlasTools.atlas_merge.__doc__)
    def atlas_merge(
        profile_path: str,
        patch: dict[str, Any] | str,
        skip_conflicts: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return call(tools.atlas_merge, profile_path, patch, skip_conflicts, dry_run)

    return server


def main() -> int:
    """Entry point of ``atlas-mcp``: serve over stdio (stdout carries the protocol; logs go to stderr)."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(name)s: %(message)s")
    try:
        server = build_server()
    except ImportError as exc:
        print(f"atlas-mcp needs the MCP SDK (pip install 'agent-studio[mcp]'): {exc}", file=sys.stderr)
        return 1
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
