"""Headless transport: ``soundcraft-cli mcp`` over stdio, via the official ``mcp`` SDK client.

The SDK client is async; this class runs it on a private event-loop thread and exposes a
synchronous interface. The server process keeps the session in memory, so a server that dies
is *not* restarted behind the caller's back: calls raise ``StudioError`` until a new
``HeadlessMcp`` is made.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .base import (
    MAX_BATCH_CALLS,
    BatchCall,
    check_command,
    check_detail,
    check_json,
    check_params,
    check_timeout,
    finish_batch,
    normalize_calls,
)
from .errors import StudioError

log = logging.getLogger(__name__)

DEFAULT_MAX_RESULT_CHARS = 64 * 1024 * 1024


@dataclass(frozen=True)
class ToolReply:
    """A tool call result reduced to what the client needs."""

    text: str
    is_error: bool


class HeadlessMcp:
    """Spawns ``<binary> mcp [args]`` and calls its tools synchronously."""

    kind = "headless"

    def __init__(
        self,
        binary: str = "soundcraft-cli",
        args: Sequence[str] = (),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 300.0,
        start_timeout: float = 30.0,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    ) -> None:
        if not isinstance(binary, str) or not binary:
            raise StudioError("binary must be a non-empty string")
        if isinstance(args, str) or not all(isinstance(a, str) for a in args):
            raise StudioError("args must be a list of strings")
        self.binary = binary
        self.args = list(args)
        self._env = dict(env) if env is not None else None
        self._cwd = cwd
        self._timeout = check_timeout(timeout, "timeout") or 300.0
        self._start_timeout = check_timeout(start_timeout, "start_timeout") or 30.0
        self._max_chars = max_result_chars
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._serve_future: concurrent.futures.Future[None] | None = None
        self._session: Any = None
        self._stop: asyncio.Event | None = None
        self._closed = False

    # ---- lifecycle --------------------------------------------------------------

    def start(self) -> None:
        """Spawn the server and run the MCP handshake (idempotent; also done on first call)."""
        with self._lock:
            if self._closed:
                raise StudioError("the Studio is closed")
            if self._serve_future is not None:
                return
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=self._run_loop, args=(loop,), name="studio-mcp", daemon=True)
            thread.start()
            ready: concurrent.futures.Future[None] = concurrent.futures.Future()
            self._loop, self._thread = loop, thread
            self._serve_future = asyncio.run_coroutine_threadsafe(self._serve(ready), loop)
            try:
                ready.result(timeout=self._start_timeout)
            except concurrent.futures.TimeoutError as exc:
                self._teardown()
                raise StudioError(f"`{self.binary} mcp` did not start within {self._start_timeout:g}s") from exc
            except StudioError:
                self._teardown()
                raise

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _serve(self, ready: concurrent.futures.Future[None]) -> None:
        # The SDK's context managers must be entered and left by the same task, so one
        # coroutine owns the whole session lifetime and waits for the stop event.
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.binary, args=["mcp", *self.args], env=self._env, cwd=self._cwd
        )
        self._stop = asyncio.Event()
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    ready.set_result(None)
                    await self._stop.wait()
        except Exception as exc:  # boundary: SDK, subprocess and anyio failures all end up here
            log.debug("headless MCP session ended: %r", exc)
            if not ready.done():
                ready.set_exception(StudioError(f"cannot start `{self.binary} mcp`: {_describe(exc)}"))
        finally:
            self._session = None
            if not ready.done():
                ready.set_exception(StudioError(f"`{self.binary} mcp` exited during start-up"))

    def _teardown(self) -> None:
        loop, thread, fut = self._loop, self._thread, self._serve_future
        if loop is not None and self._stop is not None:
            loop.call_soon_threadsafe(self._stop.set)
        if fut is not None:
            try:
                fut.result(timeout=10)
            except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError):
                fut.cancel()
            except Exception as exc:  # already reported through `ready`
                log.debug("headless MCP shutdown: %r", exc)
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)
        if loop is not None and not loop.is_running():
            loop.close()
        self._loop = self._thread = self._serve_future = self._stop = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._teardown()

    def __enter__(self) -> HeadlessMcp:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- calling tools ----------------------------------------------------------

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None, *, timeout: float | None = None) -> ToolReply:
        """Call an MCP tool and return its text content and error flag."""
        args = check_params(arguments)
        wait = check_timeout(timeout, "timeout") or self._timeout
        self.start()
        loop, session = self._loop, self._session
        if loop is None or session is None:
            raise StudioError(f"`{self.binary} mcp` is not running (it exited); create a new Studio")
        if threading.current_thread() is self._thread:
            raise StudioError("synchronous Studio calls cannot be made from the MCP event loop thread")
        fut = asyncio.run_coroutine_threadsafe(session.call_tool(name, args), loop)
        try:
            result = fut.result(timeout=wait)
        except concurrent.futures.TimeoutError as exc:
            fut.cancel()
            raise StudioError(f"`{name}` did not finish within {wait:g}s") from exc
        except Exception as exc:  # boundary: protocol error, closed pipes, dead server
            raise StudioError(f"`{name}` failed: {_describe(exc)}") from exc
        is_error = bool(getattr(result, "is_error", getattr(result, "isError", False)))
        parts: list[str] = []
        size = 0
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                size += len(text)
                if size > self._max_chars:
                    raise StudioError(f"`{name}` result exceeds {self._max_chars} characters")
                parts.append(text)
        return ToolReply("\n".join(parts), is_error)

    def _call_json(self, name: str, arguments: Mapping[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        reply = self.call_tool(name, arguments, timeout=timeout)
        if reply.is_error:
            raise StudioError(reply.text or f"`{name}` failed")
        return _parse(reply.text)

    # ---- Studio interface -------------------------------------------------------

    def execute(self, command: str, params: Mapping[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        return self._call_json(
            "execute", {"command": check_command(command), "params": check_params(params)}, timeout=timeout
        )

    def batch(self, calls: Iterable[BatchCall], keep_going: bool = False) -> list[dict[str, Any]]:
        wire = normalize_calls(calls)
        if len(wire) > MAX_BATCH_CALLS:  # normalize_calls already enforces this
            raise StudioError(f"a batch holds at most {MAX_BATCH_CALLS} commands")
        reply = self.call_tool("batch", {"calls": wire, "keep_going": bool(keep_going)})
        parsed = _parse(reply.text)
        if not isinstance(parsed, list):
            # Rejected before running anything (e.g. too many calls): plain error text.
            raise StudioError(reply.text or "batch failed")
        results = [r for r in parsed if isinstance(r, dict)]
        if len(results) != len(parsed):
            raise StudioError("malformed batch result from SoundCraft")
        return finish_batch(results, keep_going)

    def inspect(self, detail: str = "summary") -> Any:
        return self._call_json("inspect_session", {"detail": check_detail(detail)})

    def commands(self, filter: str | None = None) -> Any:
        return self._call_json("list_commands", {} if filter is None else {"filter": filter})

    def screenshot(self, path: str) -> Any:
        raise StudioError("screenshots need the desktop app: use Studio.connect(port)")

    def bounce_mix(self, path: str, **options: Any) -> Any:
        check_json(options, "options")
        return self._call_json("bounce_mix", {"path": path, **options})


def _parse(text: str) -> Any:
    """Tool results are pretty-printed JSON; fall back to the raw text for anything else."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def _describe(exc: BaseException) -> str:
    """One readable line for an exception, unwrapping anyio exception groups."""
    inner = getattr(exc, "exceptions", None)
    if inner:
        return "; ".join(_describe(e) for e in inner)
    return str(exc) or type(exc).__name__
