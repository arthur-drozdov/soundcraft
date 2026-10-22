"""``Studio``: one interface over the control channel and the headless MCP server."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .base import BatchCall, StudioTransport
from .control import ControlChannel
from .headless import HeadlessMcp


class Studio:
    """Drive SoundCraft. Build one with ``Studio.connect`` (desktop app) or ``Studio.headless``.

    Every failure raises ``StudioError``. Transports connect lazily on first use.
    """

    def __init__(self, transport: StudioTransport) -> None:
        self.transport = transport

    @classmethod
    def connect(cls, port: int, host: str = "127.0.0.1", **options: Any) -> Studio:
        """Attach to a running ``soundcraft --control <port>``."""
        return cls(ControlChannel(port, host, **options))

    @classmethod
    def headless(cls, binary: str = "soundcraft-cli", args: Sequence[str] = (), **options: Any) -> Studio:
        """Spawn ``<binary> mcp [args]`` (e.g. ``args=["--demo"]``)."""
        return cls(HeadlessMcp(binary, args, **options))

    @property
    def kind(self) -> str:
        """``"control"`` or ``"headless"``."""
        return self.transport.kind

    def execute(self, command: str, params: Mapping[str, Any] | None = None) -> Any:
        """Run one command by id and return its result."""
        return self.transport.execute(command, params)

    def batch(self, calls: Iterable[BatchCall], keep_going: bool = False) -> list[dict[str, Any]]:
        """Run up to 1000 commands in order; returns ``[{command, ok, result | error}]``.

        Raises ``StudioError`` (with ``.results``) if a call failed, unless ``keep_going``.
        """
        return self.transport.batch(calls, keep_going)

    def inspect(self, detail: str = "summary") -> Any:
        """The session document (``detail`` is ``"summary"`` or ``"full"``)."""
        return self.transport.inspect(detail)

    def commands(self, filter: str | None = None) -> Any:
        """Every command (id, label, params doc), optionally filtered by substring."""
        return self.transport.commands(filter)

    def screenshot(self, path: str) -> Any:
        """Save a PNG of the app window (control channel only)."""
        return self.transport.screenshot(path)

    def bounce_mix(self, path: str, **options: Any) -> Any:
        """Render the mix to ``path``; returns peak dBFS and integrated LUFS."""
        return self.transport.bounce_mix(path, **options)

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> Studio:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
