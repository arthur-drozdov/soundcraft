"""A tiny MCP server for tests: execute / batch / inspect_session / list_commands / bounce_mix.

Run through a wrapper as ``<wrapper> mcp [args]``; the first argv is ignored, later ones are
recorded so tests can see the args were passed through.
"""

import json
import sys

try:
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

from mcp import types

ARGS = sys.argv[1:]
mcp = Server("fake-soundcraft")
VOLUMES: dict[str, float] = {}


def error(text: str) -> types.CallToolResult:
    """Like SoundCraft: the error text is the whole content, isError is set."""
    return reply(text, True)


def reply(text: str, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=is_error)


def run_one(command: str, params: dict) -> dict:
    if command == "mix.volume":
        track = params.get("track")
        if track not in ("Kick", "Bass"):
            raise ValueError(f"bad parameters for `mix.volume`: no track named `{track}`")
        VOLUMES[track] = params["db"]
        return {"track": track, "db": params["db"]}
    if command == "args.echo":
        return {"argv": ARGS}
    if command == "sleep":
        import time

        time.sleep(float(params.get("s", 1)))
        return {}
    if command == "crash":
        import os

        os._exit(3)
    raise ValueError(f"unknown command `{command}`")


@mcp.tool()
def execute(command: str, params: dict | None = None) -> types.CallToolResult:
    """Run one command."""
    try:
        return reply(json.dumps(run_one(command, params or {}), indent=2))
    except ValueError as e:
        return error(str(e))


@mcp.tool()
def batch(calls: list[dict], keep_going: bool = False) -> types.CallToolResult:
    """Run commands in order."""
    if len(calls) > 1000:
        return error("`calls` must contain at most 1000 commands; no commands were executed")
    out, failed = [], False
    for c in calls:
        try:
            out.append({"command": c["command"], "ok": True, "result": run_one(c["command"], c.get("params") or {})})
        except ValueError as e:
            out.append({"command": c["command"], "ok": False, "error": str(e)})
            failed = True
            if not keep_going:
                break
    if failed:
        return error(json.dumps(out))
    return reply(json.dumps(out, indent=2))


@mcp.tool()
def inspect_session(detail: str = "summary") -> types.CallToolResult:
    """The session."""
    return reply(json.dumps({"detail": detail, "volumes": VOLUMES}))


@mcp.tool()
def list_commands(filter: str = "") -> types.CallToolResult:
    """Commands."""
    return reply(json.dumps([{"id": "mix.volume"}] if "mix" in filter or not filter else []))


@mcp.tool()
def bounce_mix(path: str, format: str = "wav") -> types.CallToolResult:
    """Bounce."""
    return reply(json.dumps({"path": path, "format": format, "peak_dbfs": -3.0, "lufs": -14.0}))


if __name__ == "__main__":
    mcp.run("stdio")
