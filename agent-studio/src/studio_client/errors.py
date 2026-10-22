"""Errors raised by the Studio client."""

from __future__ import annotations

from typing import Any


class StudioError(Exception):
    """Any failure talking to SoundCraft: transport, protocol, or a command that failed.

    ``results`` carries the per-call results of a failed ``batch`` (each entry is
    ``{"command", "ok", "result" | "error"}``), when the server reported them.
    """

    def __init__(self, message: str, results: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.results = results


class CommandError(StudioError):
    """SoundCraft ran the request and reported a failure (as opposed to a transport problem)."""
