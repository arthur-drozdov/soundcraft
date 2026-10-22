"""Control channel: newline-delimited JSON over TCP to the SoundCraft desktop app.

Wire format and methods: ``docs/control-protocol.md``. One request per line, one reply per
line, replies echo the request ``id``. Failures raise ``StudioError``.
"""

from __future__ import annotations

import json
import logging
import select
import socket
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from .base import (
    BatchCall,
    check_command,
    check_detail,
    check_json,
    check_params,
    check_timeout,
    finish_batch,
    normalize_calls,
)
from .errors import CommandError, StudioError

log = logging.getLogger(__name__)

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_MAX_REPLY_BYTES = 64 * 1024 * 1024
# Read-only methods: safe to resend once if the connection drops mid-request.
IDEMPOTENT_METHODS = frozenset(
    {"engine.commands", "engine.parity", "session.inspect", "ui.inspect", "ui.menu.list"}
)


class _ConnectionLost(Exception):
    """Internal: the socket failed; the request may or may not have been executed."""


class ControlChannel:
    """Synchronous, thread-safe client for ``soundcraft --control <port>``.

    The connection is opened on first use. A connection the app closed between requests is
    detected before sending and reopened. A connection lost *during* a request raises
    ``StudioError`` (the command may have run); only read-only methods are resent.
    """

    kind = "control"

    def __init__(
        self,
        port: int,
        host: str = "127.0.0.1",
        *,
        timeout: float = 120.0,
        connect_timeout: float = 5.0,
        max_reply_bytes: int = DEFAULT_MAX_REPLY_BYTES,
        allow_remote: bool = False,
    ) -> None:
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise StudioError("port must be an integer in 1..65535")
        if host not in LOOPBACK_HOSTS and not allow_remote:
            raise StudioError(
                f"refusing non-loopback host {host!r}: the control channel is unauthenticated "
                "(pass allow_remote=True to override)"
            )
        if max_reply_bytes < 1024:
            raise StudioError("max_reply_bytes must be at least 1024")
        self.host = host
        self.port = port
        self._timeout = check_timeout(timeout, "timeout") or 120.0
        self._connect_timeout = check_timeout(connect_timeout, "connect_timeout") or 5.0
        self._max_reply = max_reply_bytes
        self._lock = threading.Lock()
        self._next_id = 1
        self._sock: socket.socket | None = None
        self._reader: Any = None
        self._closed = False

    # ---- connection -------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _drop(self) -> None:
        reader, sock = self._reader, self._sock
        self._reader = None
        self._sock = None
        for resource in (reader, sock):
            if resource is not None:
                try:
                    resource.close()
                except OSError as exc:
                    log.debug("error closing control socket: %s", exc)

    def _stale(self) -> bool:
        """True when the peer closed the connection (or sent unsolicited bytes)."""
        sock = self._sock
        if sock is None:
            return True
        try:
            readable, _, _ = select.select([sock], [], [], 0)
            # Nothing is ever sent unprompted, so readable means EOF, a reset or garbage.
            return bool(readable)
        except (OSError, ValueError):
            return True

    def _ensure_connected(self) -> None:
        if self._sock is not None and self._stale():
            log.debug("control connection went away; reconnecting")
            self._drop()
        if self._sock is not None:
            return
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self._connect_timeout)
        except OSError as exc:
            raise StudioError(f"cannot connect to SoundCraft at {self.host}:{self.port}: {exc}") from exc
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._reader = sock.makefile("rb")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._drop()

    def __enter__(self) -> ControlChannel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- requests ---------------------------------------------------------------

    def _roundtrip(self, rid: int, line: bytes, timeout: float) -> Any:
        self._ensure_connected()
        sock, reader = self._sock, self._reader
        if sock is None or reader is None:
            raise _ConnectionLost("not connected")
        try:
            sock.settimeout(timeout)
            sock.sendall(line)
            while True:
                raw = reader.readline(self._max_reply + 1)
                if not raw:
                    raise _ConnectionLost("connection closed by SoundCraft")
                if len(raw) > self._max_reply:
                    self._drop()
                    raise StudioError(f"reply exceeds {self._max_reply} bytes")
                if raw.strip():
                    break
        except socket.timeout as exc:
            self._drop()
            raise StudioError(f"no reply from SoundCraft within {timeout:g}s") from exc
        except OSError as exc:
            raise _ConnectionLost(str(exc)) from exc
        return self._decode(rid, raw)

    def _decode(self, rid: int, raw: bytes) -> Any:
        try:
            reply = json.loads(raw)
        except ValueError as exc:
            self._drop()
            raise StudioError(f"malformed reply from SoundCraft: {exc}") from exc
        if not isinstance(reply, dict) or not isinstance(reply.get("ok"), bool):
            self._drop()
            raise StudioError("malformed reply from SoundCraft: expected an object with `ok`")
        if reply.get("id") != rid:
            self._drop()
            raise StudioError(f"reply id {reply.get('id')!r} does not match request id {rid}")
        if not reply["ok"]:
            error = reply.get("error")
            raise CommandError(error if isinstance(error, str) else json.dumps(error))
        return reply.get("result")

    def request(self, method: str, params: Mapping[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        """Send one method call and return its ``result``."""
        if not isinstance(method, str) or not method:
            raise StudioError("method must be a non-empty string")
        body = check_params(params)
        wait = check_timeout(timeout, "timeout") or self._timeout
        attempts = 2 if method in IDEMPOTENT_METHODS else 1
        with self._lock:
            if self._closed:
                raise StudioError("the Studio is closed")
            rid = self._next_id
            self._next_id += 1
            line = (json.dumps({"id": rid, "method": method, "params": body}, allow_nan=False) + "\n").encode()
            for attempt in range(attempts):
                try:
                    return self._roundtrip(rid, line, wait)
                except _ConnectionLost as exc:
                    self._drop()
                    if attempt + 1 == attempts:
                        raise StudioError(f"connection to SoundCraft lost during `{method}`: {exc}") from exc
                    log.debug("retrying read-only `%s` after connection loss", method)
        raise StudioError("unreachable")  # pragma: no cover

    # ---- Studio interface -------------------------------------------------------

    def execute(self, command: str, params: Mapping[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        return self.request(
            "engine.execute", {"command": check_command(command), "params": check_params(params)}, timeout=timeout
        )

    def batch(self, calls: Iterable[BatchCall], keep_going: bool = False) -> list[dict[str, Any]]:
        """Run commands in order (the control protocol has no batch method, so one request each)."""
        wire = normalize_calls(calls)
        results: list[dict[str, Any]] = []
        for call in wire:
            try:
                value = self.execute(call["command"], call["params"])
            except CommandError as exc:
                results.append({"command": call["command"], "ok": False, "error": exc.message})
                if not keep_going:
                    break
                continue
            results.append({"command": call["command"], "ok": True, "result": value})
        return finish_batch(results, keep_going)

    def inspect(self, detail: str = "summary") -> Any:
        return self.request("session.inspect", {"detail": check_detail(detail)})

    def commands(self, filter: str | None = None) -> Any:
        return self.request("engine.commands", {} if filter is None else {"filter": filter})

    def screenshot(self, path: str) -> Any:
        if not isinstance(path, str) or not path:
            raise StudioError("path must be a non-empty string")
        return self.request("ui.screenshot", {"path": path})

    def bounce_mix(self, path: str, **options: Any) -> Any:
        check_json(options, "options")
        return self.execute("file.bounce_mix", {"path": path, **options})
