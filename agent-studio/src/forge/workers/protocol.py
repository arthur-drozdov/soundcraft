"""Wire shapes of the worker protocol (DESIGN 3.2) and the checks the gateway applies to replies.

Plain dictionaries and no third-party imports, so a worker environment needs only numpy/scipy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

from typing_extensions import NotRequired

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_OUTPUTS = 256
OUTPUT_KINDS = ("audio", "midi", "instrument", "json")


class ModelInfo(TypedDict):
    id: str
    family: str
    job_types: list[str]
    available: bool
    reason: NotRequired[str]
    licence: str
    checkpoint: str
    checkpoint_hash: NotRequired[str | None]
    vram_gb: float
    max_seconds: NotRequired[float | None]
    params_schema: NotRequired[dict[str, Any] | None]


class RunJob(TypedDict):
    job_id: str
    type: str
    model: str
    params: dict[str, Any]
    inputs: dict[str, str]
    out_dir: str
    seed: int


class Output(TypedDict):
    path: str
    kind: str
    candidate: int
    extra: dict[str, Any]


class ProtocolError(ValueError):
    """A worker sent something that violates the protocol."""


def _text(value: Any, name: str, limit: int = 4096) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ProtocolError(f"model info field {name!r} must be a string")
    return value


def validate_model_info(raw: Any, family: str) -> ModelInfo:
    """Check one ``ModelInfo`` from ``hello`` and return a normalised copy."""
    if not isinstance(raw, Mapping):
        raise ProtocolError("model info must be an object")
    types = raw.get("job_types")
    if not isinstance(types, list) or not all(isinstance(t, str) for t in types) or len(types) > 64:
        raise ProtocolError("model info job_types must be a list of strings")
    vram = raw.get("vram_gb", 0.0)
    if isinstance(vram, bool) or not isinstance(vram, (int, float)) or not math.isfinite(vram) or vram < 0:
        raise ProtocolError("model info vram_gb must be a non-negative number")
    if raw.get("family") != family:
        raise ProtocolError(f"model {raw.get('id')!r} claims family {raw.get('family')!r}, expected {family!r}")
    info: dict[str, Any] = dict(raw)
    info.update(
        id=_text(raw.get("id"), "id", 128),
        family=family,
        job_types=list(types),
        available=bool(raw.get("available")),
        licence=_text(raw.get("licence", ""), "licence"),
        checkpoint=_text(raw.get("checkpoint", ""), "checkpoint"),
        vram_gb=float(vram),
    )
    if not info["id"]:
        raise ProtocolError("model info id must not be empty")
    if "reason" in info:
        info["reason"] = _text(info["reason"], "reason")
    return info  # type: ignore[return-value]


def validate_outputs(raw: Any, out_dir: Path) -> list[Output]:
    """Check a ``run`` result: every output path must exist inside ``out_dir`` and the kind be known."""
    if not isinstance(raw, list) or len(raw) > MAX_OUTPUTS:
        raise ProtocolError("run result outputs must be a list")
    root = out_dir.resolve()
    outputs: list[Output] = []
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ProtocolError("each output needs a path")
        if item.get("kind") not in OUTPUT_KINDS:
            raise ProtocolError(f"unknown output kind {item.get('kind')!r}")
        candidate = item.get("candidate", 0)
        extra = item.get("extra", {})
        if isinstance(candidate, bool) or not isinstance(candidate, int) or not 0 <= candidate < 1000:
            raise ProtocolError("output candidate must be a small non-negative integer")
        if not isinstance(extra, dict):
            raise ProtocolError("output extra must be an object")
        path = Path(item["path"])
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ProtocolError(f"output path {item['path']!r} is not inside the job directory") from exc
        outputs.append({"path": str(resolved), "kind": item["kind"], "candidate": candidate, "extra": extra})
    return outputs
