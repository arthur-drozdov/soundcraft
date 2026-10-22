"""Profile validation: the JSON Schema plus the rules a schema cannot express."""

from __future__ import annotations

import datetime
import functools
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from . import yamlio

# Param fields that carry knowledge: each must have a sibling provenance entry (DESIGN 6.1).
KNOWLEDGE_FIELDS = ("does", "role", "musical_use", "interactions", "safe_range", "calibration", "unit")
INFERRED_CONFIDENCE_LIMIT = 0.9
PREFIXED_FORMATS = ("clap", "vst3", "au")


@dataclass(frozen=True)
class Issue:
    """One validation problem. ``path`` is a dotted/indexed location, ``""`` for the document."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path or '<root>'}: {self.message}"


class ProfileError(Exception):
    """A profile, patch or merge was rejected. ``issues`` lists every problem found."""

    def __init__(self, message: str, issues: list[Issue] | None = None) -> None:
        self.issues = issues or []
        detail = "; ".join(str(i) for i in self.issues[:10])
        super().__init__(f"{message}: {detail}" if detail else message)


@functools.lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """The packaged ``schema/profile.schema.json``."""
    text = resources.files("atlas").joinpath("schema/profile.schema.json").read_text(encoding="utf-8")
    schema = json.loads(text)
    Draft202012Validator.check_schema(schema)
    return schema  # type: ignore[no-any-return]


@functools.lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(load_schema())


def _fmt_path(parts: Any) -> str:
    out = ""
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _describe(error: ValidationError) -> str:
    path = list(error.absolute_path)
    if error.validator == "exclusiveMaximum" and path and path[-1] == "confidence":
        return (
            f"inferred provenance must have confidence < {INFERRED_CONFIDENCE_LIMIT} "
            f"(got {error.instance}); use source host/manual/source/measured for firmer knowledge"
        )
    if error.validator == "additionalProperties":
        return error.message.replace("Additional properties are not allowed", "unknown field(s)")
    return error.message


def _schema_issues(profile: Any) -> list[Issue]:
    errors = sorted(_validator().iter_errors(profile), key=lambda e: [str(p) for p in e.absolute_path])
    return [Issue(_fmt_path(e.absolute_path), _describe(e)) for e in errors]


def _walk_numbers(obj: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], float]]:
    if isinstance(obj, float):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_numbers(v, (*path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_numbers(v, (*path, i))


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _param_issues(index: int, param: dict[str, Any]) -> list[Issue]:
    here = f"params[{index}]"
    issues: list[Issue] = []
    prov = param.get("provenance")
    prov = prov if isinstance(prov, dict) else {}
    for field in KNOWLEDGE_FIELDS:
        if field in param and field not in prov:
            issues.append(Issue(f"{here}.provenance", f"`{field}` is set but has no provenance"))
    for field in prov:
        if field not in param:
            issues.append(Issue(f"{here}.provenance.{field}", f"provenance for `{field}`, which the param does not have"))
    lo, hi = _num(param.get("min")), _num(param.get("max"))
    if lo is not None and hi is not None and lo > hi:
        issues.append(Issue(f"{here}.min", f"min {lo:g} is above max {hi:g}"))
    default = _num(param.get("default"))
    if default is not None and lo is not None and hi is not None and not lo <= default <= hi:
        issues.append(Issue(f"{here}.default", f"default {default:g} is outside {lo:g}..{hi:g}"))
    sr = param.get("safe_range")
    if isinstance(sr, list) and len(sr) == 2 and all(_num(v) is not None for v in sr):
        s_lo, s_hi = float(sr[0]), float(sr[1])
        if s_lo > s_hi:
            issues.append(Issue(f"{here}.safe_range", "safe_range low is above high"))
        elif (lo is not None and s_lo < lo) or (hi is not None and s_hi > hi):
            issues.append(Issue(f"{here}.safe_range", "safe_range lies outside min..max"))
    choices = param.get("choices")
    if isinstance(choices, list) and len(set(map(str, choices))) != len(choices):
        issues.append(Issue(f"{here}.choices", "duplicate choices"))
    return issues


def _semantic_issues(profile: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    for path, value in _walk_numbers(profile):
        if not math.isfinite(value):
            issues.append(Issue(_fmt_path(path), "numbers must be finite"))
    identity = profile.get("identity")
    if isinstance(identity, dict):
        fmt, pid = identity.get("format"), identity.get("plugin_id")
        if fmt in PREFIXED_FORMATS and isinstance(pid, str) and not pid.startswith(f"{fmt}:"):
            issues.append(Issue("identity.plugin_id", f"a {fmt} plugin id must start with `{fmt}:`"))
        profiled = identity.get("profiled")
        if isinstance(profiled, str):
            try:
                datetime.date.fromisoformat(profiled)
            except ValueError:
                issues.append(Issue("identity.profiled", f"`{profiled}` is not a real date"))
    params = profile.get("params")
    if isinstance(params, list):
        seen: dict[str, int] = {}
        for i, param in enumerate(params):
            if not isinstance(param, dict):
                continue
            key = str(param.get("id"))
            if key in seen:
                issues.append(Issue(f"params[{i}].id", f"duplicate param id `{key}` (first at params[{seen[key]}])"))
            seen.setdefault(key, i)
            issues.extend(_param_issues(i, param))
    return issues


def validate_profile(profile: Any) -> list[Issue]:
    """Return every problem with ``profile`` (empty list = valid).

    Schema violations come first; the semantic pass only runs when the shape is sound.
    """
    if not isinstance(profile, dict):
        return [Issue("", "a profile must be a mapping")]
    issues = _schema_issues(profile)
    if not issues:
        issues = _semantic_issues(profile)
    return issues


def check_profile(profile: Any) -> None:
    """Raise ``ProfileError`` unless ``profile`` is valid."""
    issues = validate_profile(profile)
    if issues:
        raise ProfileError("invalid profile", issues)


def validate_file(path: str | Path) -> list[Issue]:
    """Validate a YAML profile file; unreadable or unparsable files are reported as issues."""
    try:
        data = yamlio.read_file(path)
    except yamlio.YamlError as exc:
        return [Issue("", str(exc))]
    return validate_profile(data)
