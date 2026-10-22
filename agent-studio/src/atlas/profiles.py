"""Profile files: load/save with stable key order, path naming, lookup and version checks.

Layout: ``<root>/<format>/<vendor-slug>/<name-slug>@<major>.yaml``.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import yamlio
from .validate import Issue, ProfileError, check_profile, validate_profile

log = logging.getLogger(__name__)

TOP_ORDER = ("identity", "summary", "signal_flow", "modes", "params", "ui_only", "presets", "known_issues", "provenance")
IDENTITY_ORDER = ("name", "vendor", "format", "plugin_id", "version", "os", "profiled")
PARAM_ORDER = (
    "id", "host_name", "name", "section", "role", "unit", "min", "max", "default", "taper", "choices",
    "calibration", "does", "musical_use", "interactions", "safe_range", "automatable", "provenance",
)
PROVENANCE_ORDER = ("source", "ref", "confidence")
CALIBRATION_ORDER = ("value", "text")
MAX_SLUG = 64


# ---- naming ---------------------------------------------------------------------


def slugify(text: str) -> str:
    """Lower-case ASCII slug: accents folded, runs of other characters become one ``-``."""
    folded = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")[:MAX_SLUG].strip("-")
    return slug or "unnamed"


def version_numbers(version: str) -> tuple[int, ...]:
    """The numeric parts of a version string (``"v2.3.1-beta"`` -> ``(2, 3, 1)``)."""
    return tuple(int(n) for n in re.findall(r"\d{1,9}", str(version)))


def major_of(version: str) -> int:
    """Major version: the first number in ``version``, or 0 when there is none."""
    nums = version_numbers(version)
    return nums[0] if nums else 0


def profile_relpath(profile: dict[str, Any]) -> Path:
    """``<format>/<vendor-slug>/<name-slug>@<major>.yaml`` for a profile's identity."""
    ident = profile.get("identity")
    if not isinstance(ident, dict):
        raise ProfileError("profile has no identity")
    try:
        fmt, vendor, name, version = ident["format"], ident["vendor"], ident["name"], ident["version"]
    except KeyError as exc:
        raise ProfileError(f"identity is missing {exc.args[0]}") from exc
    return Path(slugify(fmt)) / slugify(vendor) / f"{slugify(name)}@{major_of(version)}.yaml"


def profile_path(root: str | Path, profile: dict[str, Any]) -> Path:
    return Path(root) / profile_relpath(profile)


def check_layout(path: str | Path, root: str | Path, profile: dict[str, Any]) -> list[Issue]:
    """Issue when ``path`` is not where the profile's identity says it belongs under ``root``."""
    try:
        expected = profile_relpath(profile)
        actual = Path(path).resolve().relative_to(Path(root).resolve())
    except (ProfileError, ValueError):
        return [Issue("", f"cannot check file placement for {path}")]
    if actual != expected:
        return [Issue("", f"file is at {actual.as_posix()}, expected {expected.as_posix()}")]
    return []


# ---- versions -------------------------------------------------------------------


def version_status(profiled_version: str, actual_version: str) -> str:
    """``"exact"``, ``"same-major"`` or ``"different"``; a leading ``v`` and spaces are ignored."""

    def norm(v: str) -> str:
        return str(v).strip().lower().removeprefix("v")

    if norm(profiled_version) == norm(actual_version):
        return "exact"
    if major_of(profiled_version) == major_of(actual_version):
        return "same-major"
    return "different"


def version_matches(profile: dict[str, Any], actual_version: str) -> bool:
    """True when the profile was made for exactly this plugin version (re-profile if not)."""
    return version_status(profile["identity"]["version"], actual_version) == "exact"


# ---- ordering and file I/O --------------------------------------------------------


def _ordered(mapping: dict[str, Any], order: tuple[str, ...]) -> dict[str, Any]:
    out = {k: mapping[k] for k in order if k in mapping}
    out.update({k: mapping[k] for k in sorted(set(mapping) - set(order), key=str)})
    return out


def canonical(profile: dict[str, Any]) -> dict[str, Any]:
    """A copy with keys in the stable order used on disk (provenance keys in field order)."""
    out = _ordered(profile, TOP_ORDER)
    if isinstance(out.get("identity"), dict):
        out["identity"] = _ordered(out["identity"], IDENTITY_ORDER)
    for key in ("modes", "ui_only", "presets"):
        if isinstance(out.get(key), list):
            out[key] = [_ordered(e, ("name", "description", "feature", "procedure", "load")) if isinstance(e, dict) else e for e in out[key]]
    params = []
    for param in out.get("params") or []:
        if not isinstance(param, dict):
            params.append(param)
            continue
        p = _ordered(param, PARAM_ORDER)
        if isinstance(p.get("calibration"), list):
            p["calibration"] = [_ordered(c, CALIBRATION_ORDER) if isinstance(c, dict) else c for c in p["calibration"]]
        if isinstance(p.get("provenance"), dict):
            p["provenance"] = {
                f: _ordered(v, PROVENANCE_ORDER) if isinstance(v, dict) else v
                for f, v in _ordered(p["provenance"], PARAM_ORDER).items()
            }
        params.append(p)
    if "params" in out:
        out["params"] = params
    if isinstance(out.get("provenance"), dict):
        out["provenance"] = {
            f: _ordered(v, PROVENANCE_ORDER) if isinstance(v, dict) else v
            for f, v in _ordered(out["provenance"], TOP_ORDER).items()
        }
    return out


def dump_profile(profile: dict[str, Any]) -> str:
    """YAML text for a profile with stable key order (same profile -> same bytes)."""
    return yamlio.dumps(canonical(profile))


def load_profile(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    """Read a profile file. Raises ``ProfileError`` if it is unreadable or (by default) invalid."""
    try:
        data = yamlio.read_file(path)
    except yamlio.YamlError as exc:
        raise ProfileError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ProfileError(f"{path}: a profile must be a mapping")
    if validate:
        issues = validate_profile(data)
        if issues:
            raise ProfileError(f"{path}: invalid profile", issues)
    return data


def write_profile(path: str | Path, profile: dict[str, Any], *, validate: bool = True) -> Path:
    """Atomically write ``profile`` to ``path`` (temp file + rename), creating folders."""
    if validate:
        check_profile(profile)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = dump_profile(profile)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def save_profile(root: str | Path, profile: dict[str, Any], *, validate: bool = True) -> Path:
    """Write ``profile`` to its canonical place under ``root`` and return the path."""
    return write_profile(profile_path(root, profile), profile, validate=validate)


# ---- lookup -----------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileRef:
    path: Path
    profile: dict[str, Any]


@dataclass(frozen=True)
class ProfileMatch:
    """A found profile and how it relates to the requested plugin version.

    ``status`` is ``"exact"``, ``"same-major"``, ``"different"``, or ``"unchecked"`` when no
    version was asked for. ``version_match`` is True only for ``"exact"`` (``None`` if unchecked).
    """

    path: Path
    profile: dict[str, Any]
    status: str

    @property
    def version_match(self) -> bool | None:
        return None if self.status == "unchecked" else self.status == "exact"


def iter_profile_paths(root: str | Path) -> Iterator[Path]:
    """Every ``<format>/<vendor>/<file>.yaml`` under ``root``, sorted, symlinks not followed."""
    base = Path(root)
    if not base.is_dir():
        return
    for fmt in sorted(p for p in base.iterdir() if p.is_dir() and not p.is_symlink()):
        for vendor in sorted(p for p in fmt.iterdir() if p.is_dir() and not p.is_symlink()):
            for f in sorted(vendor.glob("*.yaml")):
                if f.is_file() and not f.is_symlink():
                    yield f


def list_profiles(root: str | Path) -> list[ProfileRef]:
    """All valid profiles under ``root``; invalid files are logged and skipped."""
    found: list[ProfileRef] = []
    for path in iter_profile_paths(root):
        try:
            found.append(ProfileRef(path, load_profile(path)))
        except ProfileError as exc:
            log.warning("skipping %s: %s", path, exc)
    return found


def find_profile(root: str | Path, plugin_id: str, version: str | None = None) -> ProfileMatch | None:
    """The profile for ``plugin_id``; with ``version``, the closest one by version.

    Preference: exact version, then same major, then the highest major. Without ``version``,
    the highest major (status ``"unchecked"``).
    """
    candidates = [r for r in list_profiles(root) if r.profile["identity"]["plugin_id"] == plugin_id]
    if not candidates:
        return None

    def rank(ref: ProfileRef) -> tuple[int, tuple[int, ...]]:
        v = ref.profile["identity"]["version"]
        order = {"exact": 0, "same-major": 1, "different": 2}
        status = order[version_status(v, version)] if version is not None else 0
        return (status, tuple(-n for n in version_numbers(v)))

    best = min(candidates, key=rank)
    status = "unchecked" if version is None else version_status(best.profile["identity"]["version"], version)
    return ProfileMatch(best.path, best.profile, status)
