"""Knowledge patches: the YAML shape, loading, and shape-only validation.

A patch has the shape of a profile with every field optional and ``params`` keyed by the host
param ``id``. Applying one to a profile is ``atlas.merge``'s job; this module only says what a
well-formed patch looks like, independent of any profile::

    default_provenance: {source: manual, ref: "Manual p.12", confidence: 0.95}
    summary: "A two-band compressor."            # TOP_FIELDS may be set
    provenance: {summary: {ref: "Manual p.3"}}   # per-field overrides of the default
    params:
      - id: cutoff                               # must name an existing param (checked at merge)
        does: "Sets the filter corner frequency."
        safe_range: [200, 8000]
        provenance: {does: {confidence: 1.0}}

Every field a patch sets needs ``source`` and ``confidence`` from the default overlaid with the
per-field entry; ``ref`` is optional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import yamlio
from .validate import Issue, ProfileError

TOP_FIELDS = ("summary", "signal_flow", "modes", "ui_only", "presets", "known_issues")
PARAM_FIELDS = (
    "name", "section", "role", "unit", "min", "max", "default", "taper", "choices",
    "calibration", "does", "musical_use", "interactions", "safe_range", "automatable",
)
PATCH_KEYS = frozenset({"default_provenance", "provenance", "params", *TOP_FIELDS})
PROVENANCE_KEYS = frozenset({"source", "ref", "confidence"})
MAX_PATCH_PARAMS = 4096


def check_patch(patch: Any) -> list[Issue]:
    """Every shape problem in ``patch`` (empty list = well formed); no profile needed."""
    if not isinstance(patch, dict):
        return [Issue("", "a patch must be a mapping")]
    issues: list[Issue] = []
    for key in sorted(set(patch) - PATCH_KEYS, key=str):
        issues.append(Issue(str(key), "unknown patch key"))

    default = patch.get("default_provenance", {})
    if not isinstance(default, dict):
        issues.append(Issue("default_provenance", "must be a mapping"))
    else:
        bad = sorted(set(default) - PROVENANCE_KEYS, key=str)
        if bad:
            issues.append(Issue("default_provenance", f"unknown provenance key(s): {bad}"))

    top = patch.get("provenance", {})
    if not isinstance(top, dict):
        issues.append(Issue("provenance", "must be a mapping"))
    else:
        for name in sorted(set(top) - {f for f in TOP_FIELDS if f in patch}, key=str):
            issues.append(Issue(f"provenance.{name}", "provenance for a field the patch does not set"))

    entries = patch.get("params", [])
    if "params" in patch and not isinstance(entries, list):
        issues.append(Issue("params", "must be a list"))
        return issues
    if len(entries) > MAX_PATCH_PARAMS:
        issues.append(Issue("params", f"more than {MAX_PATCH_PARAMS} param entries"))
        return issues
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        here = f"params[{i}]"
        if not isinstance(entry, dict) or "id" not in entry:
            issues.append(Issue(here, "each param patch is a mapping with an `id`"))
            continue
        if isinstance(entry["id"], bool) or not isinstance(entry["id"], (str, int)):
            issues.append(Issue(f"{here}.id", "must be a string or integer"))
            continue
        key = str(entry["id"])
        if key in seen:
            issues.append(Issue(f"{here}.id", f"param `{key}` appears twice in the patch"))
        seen.add(key)
        for bad_field in sorted(set(entry) - {"id", "provenance", *PARAM_FIELDS}, key=str):
            issues.append(Issue(f"{here}.{bad_field}", "cannot be set by a patch"))
        overrides = entry.get("provenance", {})
        if not isinstance(overrides, dict):
            issues.append(Issue(f"{here}.provenance", "must be a mapping"))
            continue
        for name in sorted(set(overrides) - {f for f in PARAM_FIELDS if f in entry}, key=str):
            issues.append(Issue(f"{here}.provenance.{name}", "provenance for a field the patch does not set"))
    return issues


def load_patch(path: str | Path) -> dict[str, Any]:
    """Read a patch file and check its shape; raises ``ProfileError`` listing the problems."""
    try:
        data = yamlio.read_file(path)
    except yamlio.YamlError as exc:
        raise ProfileError("unreadable patch", [Issue("", str(exc))]) from exc
    issues = check_patch(data)
    if issues:
        raise ProfileError(f"{path}: invalid patch", issues)
    assert isinstance(data, dict)
    return data
