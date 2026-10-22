"""Shared types and helpers for the two Studio transports."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from .errors import StudioError

MAX_BATCH_CALLS = 1000
"""SoundCraft rejects larger batches before running any command."""

BatchCall = Mapping[str, Any] | tuple[str, Mapping[str, Any] | None]


class StudioTransport(Protocol):
    """The interface both ``ControlChannel`` and ``HeadlessMcp`` implement."""

    kind: str

    def execute(self, command: str, params: Mapping[str, Any] | None = None) -> Any: ...

    def batch(self, calls: Iterable[BatchCall], keep_going: bool = False) -> list[dict[str, Any]]: ...

    def inspect(self, detail: str = "summary") -> Any: ...

    def commands(self, filter: str | None = None) -> Any: ...

    def screenshot(self, path: str) -> Any: ...

    def bounce_mix(self, path: str, **options: Any) -> Any: ...

    def close(self) -> None: ...


def check_json(value: Any, what: str) -> None:
    """Reject values that cannot go on the wire as strict JSON (NaN, inf, odd types)."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StudioError(f"{what} is not valid JSON: {exc}") from exc


def check_command(command: Any) -> str:
    if not isinstance(command, str) or not command.strip():
        raise StudioError("command must be a non-empty string")
    return command.strip()


def check_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise StudioError("params must be an object")
    out = dict(params)
    check_json(out, "params")
    return out


def normalize_calls(calls: Iterable[BatchCall]) -> list[dict[str, Any]]:
    """Turn ``{"command", "params"}`` dicts or ``(command, params)`` tuples into wire form."""
    out: list[dict[str, Any]] = []
    for call in calls:
        if isinstance(call, tuple):
            if len(call) != 2:
                raise StudioError("a batch tuple must be (command, params)")
            command, params = call
        elif isinstance(call, Mapping):
            command, params = call.get("command"), call.get("params")
        else:
            raise StudioError("a batch call must be an object or a (command, params) tuple")
        out.append({"command": check_command(command), "params": check_params(params)})
        if len(out) > MAX_BATCH_CALLS:
            raise StudioError(f"a batch holds at most {MAX_BATCH_CALLS} commands; none were run")
    return out


def check_detail(detail: str) -> str:
    if detail not in ("summary", "full"):
        raise StudioError("detail must be 'summary' or 'full'")
    return detail


def check_timeout(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise StudioError(f"{name} must be a positive finite number")
    return float(value)


def finish_batch(results: list[dict[str, Any]], keep_going: bool) -> list[dict[str, Any]]:
    """Return ``results``, or raise ``StudioError`` if a call failed and ``keep_going`` is off."""
    failed = [r for r in results if not r.get("ok")]
    if failed and not keep_going:
        first = failed[0]
        raise StudioError(
            f"batch stopped at `{first.get('command', '?')}`: {first.get('error', 'failed')}",
            results=results,
        )
    return results
