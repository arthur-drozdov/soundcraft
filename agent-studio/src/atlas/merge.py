"""Apply a knowledge patch (agent- or human-written YAML) to a profile.

Patch format::

    default_provenance: {source: manual, ref: "Manual p.12", confidence: 0.95}   # applies to every field set
    summary: "A two-band compressor."                       # top-level: summary, signal_flow, modes,
    provenance: {summary: {ref: "Manual p.3"}}              #   ui_only, presets, known_issues
    params:
      - id: cutoff                                          # keyed by the host param id
        does: "Sets the filter corner frequency."
        safe_range: [200, 8000]
        provenance: {does: {ref: "Manual p.14", confidence: 1.0}}   # per-field overrides of the default

Every field the patch sets gets a provenance entry (the default overlaid with the per-field
one). A field whose recorded provenance is ``host``, ``source`` or ``measured`` is never
overwritten by an ``inferred`` value of equal or lower confidence. The merge is all-or-nothing:
the input is not modified and nothing is returned unless the merged profile validates.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .patch import PARAM_FIELDS, PROVENANCE_KEYS, TOP_FIELDS, check_patch, load_patch
from .profiles import load_profile, write_profile
from .validate import Issue, ProfileError, check_profile

PROTECTED_SOURCES = frozenset({"host", "source", "measured"})


@dataclass(frozen=True)
class Change:
    """A field the patch set. ``path`` is e.g. ``params[cutoff].does`` or ``summary``."""

    path: str
    old: Any
    new: Any
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Conflict:
    """A field the patch tried to set but may not: it would replace firmer knowledge."""

    path: str
    existing: dict[str, Any]
    proposed: dict[str, Any]

    @property
    def message(self) -> str:
        return (
            f"refusing {self.proposed.get('source')} (confidence {self.proposed.get('confidence')}) "
            f"over {self.existing.get('source')} (confidence {self.existing.get('confidence')})"
        )

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass
class MergeResult:
    profile: dict[str, Any]
    changes: list[Change] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)


class MergeError(ProfileError):
    """The patch is malformed, names unknown params, conflicts, or yields an invalid profile."""

    def __init__(self, message: str, issues: list[Issue] | None = None, conflicts: list[Conflict] | None = None) -> None:
        self.conflicts = conflicts or []
        extra = [Issue(c.path, c.message) for c in self.conflicts]
        super().__init__(message, [*(issues or []), *extra])


def _provenance_for(field_name: str, default: dict[str, Any], overrides: Any, where: str, issues: list[Issue]) -> dict[str, Any] | None:
    override = overrides.get(field_name, {}) if isinstance(overrides, dict) else {}
    if not isinstance(override, dict):
        issues.append(Issue(f"{where}.provenance.{field_name}", "must be a mapping"))
        return None
    merged = {**default, **override}
    unknown = set(merged) - PROVENANCE_KEYS
    if unknown:
        issues.append(Issue(f"{where}.provenance.{field_name}", f"unknown provenance key(s): {sorted(unknown)}"))
        return None
    if "source" not in merged or "confidence" not in merged:
        issues.append(Issue(f"{where}.{field_name}", "no provenance: give default_provenance or a per-field entry with source and confidence"))
        return None
    return {k: merged[k] for k in ("source", "ref", "confidence") if k in merged}


def _protected_conflict(existing: Any, proposed: dict[str, Any]) -> bool:
    if not isinstance(existing, dict) or proposed.get("source") != "inferred":
        return False
    if existing.get("source") not in PROTECTED_SOURCES:
        return False
    old_c, new_c = existing.get("confidence"), proposed.get("confidence")
    if not isinstance(old_c, (int, float)) or not isinstance(new_c, (int, float)):
        return True  # cannot show the inferred value is firmer: keep what the host/source/measurement said
    return float(new_c) <= float(old_c)


def _param_key(value: Any) -> str:
    return str(value)


def merge_patch(profile: dict[str, Any], patch: Any, *, skip_conflicts: bool = False) -> MergeResult:
    """Return the profile with ``patch`` applied (``profile`` itself is not modified).

    Conflicts raise ``MergeError`` unless ``skip_conflicts`` is set, in which case the
    conflicting fields are left as they were and listed in ``MergeResult.conflicts``.
    """
    check_profile(profile)
    shape = check_patch(patch)
    if shape:
        raise MergeError("invalid patch", shape)
    issues: list[Issue] = []
    default = patch.get("default_provenance", {})

    result = copy.deepcopy(profile)
    changes: list[Change] = []
    conflicts: list[Conflict] = []

    def apply(target: dict[str, Any], prov_map: dict[str, Any], name: str, value: Any, prov: dict[str, Any], path: str) -> None:
        existing = prov_map.get(name)
        if _protected_conflict(existing, prov):
            conflicts.append(Conflict(path, dict(existing), prov))
            return
        changes.append(Change(path, copy.deepcopy(target.get(name)), copy.deepcopy(value), dict(prov)))
        target[name] = copy.deepcopy(value)
        prov_map[name] = dict(prov)

    top_overrides = patch.get("provenance", {})
    for name in TOP_FIELDS:
        if name not in patch:
            continue
        prov = _provenance_for(name, default, top_overrides, "", issues)
        if prov is not None:
            apply(result, result.setdefault("provenance", {}), name, patch[name], prov, name)

    entries = patch.get("params", [])
    by_id = {_param_key(p["id"]): p for p in result["params"]}
    for i, entry in enumerate(entries):
        here = f"params[{i}]"
        key = _param_key(entry["id"])
        target = by_id.get(key)
        if target is None:
            issues.append(Issue(f"{here}.id", f"unknown param id `{key}` (patches change existing params; re-run the profiler to add new ones)"))
            continue
        overrides = entry.get("provenance", {})
        prov_map = target.setdefault("provenance", {}) if any(f in entry for f in PARAM_FIELDS) else {}
        for name in PARAM_FIELDS:
            if name not in entry:
                continue
            prov = _provenance_for(name, default, overrides, here, issues)
            if prov is not None:
                apply(target, prov_map, name, entry[name], prov, f"params[{key}].{name}")

    if issues:
        raise MergeError("invalid patch", issues)
    if conflicts and not skip_conflicts:
        raise MergeError("patch conflicts with firmer knowledge", conflicts=conflicts)
    try:
        check_profile(result)
    except ProfileError as exc:
        raise MergeError("merge would produce an invalid profile", exc.issues) from exc
    return MergeResult(result, changes, conflicts)


def merge_files(
    profile_path: str | Path,
    patch_path: str | Path,
    *,
    out_path: str | Path | None = None,
    skip_conflicts: bool = False,
    dry_run: bool = False,
) -> tuple[MergeResult, Path | None]:
    """Merge a patch file into a profile file; writes to ``out_path`` (default: in place).

    Returns the result and the path written (``None`` for a dry run).
    """
    profile = load_profile(profile_path)
    try:
        patch = load_patch(patch_path)
    except ProfileError as exc:
        raise MergeError(exc.label, exc.issues) from exc
    result = merge_patch(profile, patch, skip_conflicts=skip_conflicts)
    if dry_run:
        return result, None
    written = write_profile(out_path or profile_path, result.profile)
    return result, written
