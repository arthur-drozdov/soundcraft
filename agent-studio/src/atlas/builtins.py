"""Regenerate the profiles of SoundCraft's built-in plugins (DESIGN 6.2, ``atlas builtins``).

For every plugin in ``engine.plugins`` the host fields (identity, the raw parameter table)
come from ``plugin.describe`` on a scratch session, and the curated knowledge in
``atlas/curated/builtin/<plugin id>.yaml`` is merged over them with ``atlas.merge``. The result
is written to ``<out>/<name>@<major>.yaml``.

The output is deterministic: plugins are handled in id order, key order is fixed by
``atlas.profiles``, and ``identity.profiled`` only changes when the rest of the content does
(or when a date is passed explicitly). A re-run on an unchanged build rewrites nothing.
"""

from __future__ import annotations

import copy
import datetime
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from studio_client import StudioError

from . import profiles
from .merge import merge_patch
from .patch import load_patch
from .profiler import (
    CalibrationPoint,
    HostPlugin,
    ProfilerError,
    Rig,
    ScratchSession,
    StudioLike,
    _apply_calibration,
    build_host_profile,
    calibrate,
    parse_describe,
)
from .validate import ProfileError

log = logging.getLogger(__name__)

BUILTIN_OS = ["macos", "windows", "linux", "freebsd", "web"]
BUILTIN_OS_PROVENANCE = {
    "source": "source",
    "ref": "AGENTS.md (SoundCraft runs on macOS, Windows, Linux, FreeBSD and the web)",
    "confidence": 0.9,
}


@dataclass
class BuiltinsReport:
    """What one ``generate_builtins`` run did."""

    written: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    """plugin id -> why its profile was not written."""
    curated_without_plugin: list[str] = field(default_factory=list)
    plugins_without_curated: list[str] = field(default_factory=list)
    stale_files: list[Path] = field(default_factory=list)
    """Files in the output folder that no current plugin produces (left alone)."""
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    plugin_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def default_curated_dir() -> Path:
    """``atlas/curated/builtin`` of the checkout."""
    return Path(__file__).resolve().parents[2] / "atlas" / "curated" / "builtin"


def curated_ids(curated_dir: str | Path) -> list[str]:
    """Plugin ids with a curated patch file (``<id>.yaml``), sorted."""
    base = Path(curated_dir)
    if not base.is_dir():
        return []
    return sorted(p.stem for p in base.glob("*.yaml") if p.is_file() and not p.is_symlink())


def _plugin_rows(studio: StudioLike) -> list[dict[str, Any]]:
    rows = studio.execute("engine.plugins")
    if not isinstance(rows, list):
        raise ProfilerError("engine.plugins did not return a list")
    out = [r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str) and r["id"]]
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids):
        raise ProfilerError("engine.plugins lists a plugin id twice")
    return sorted(out, key=lambda r: str(r["id"]))


def _describe_builtin(studio: StudioLike, row: Mapping[str, Any], effect_track: int, instrument_track: int) -> tuple[HostPlugin, Rig]:
    plugin_id = str(row["id"])
    if row.get("instrument"):
        studio.execute("mix.instrument", {"track": instrument_track, "plugin": plugin_id})
        rig = Rig(track=instrument_track, instrument=True, plugin_id=plugin_id)
    else:
        studio.execute("mix.insert", {"track": effect_track, "slot": 0, "plugin": plugin_id})
        rig = Rig(track=effect_track, instrument=False, slot=0, plugin_id=plugin_id)
    target: dict[str, Any] = {"track": rig.track, "instrument": True} if rig.instrument else {"track": rig.track, "slot": rig.slot}
    host = parse_describe(studio.execute("plugin.describe", target))
    if host.plugin_id != plugin_id:
        raise ProfilerError(f"plugin.describe returned `{host.plugin_id}` for `{plugin_id}`")
    listed = [str(p.get("id")) for p in row.get("params", []) if isinstance(p, Mapping)] if isinstance(row.get("params"), list) else []
    described = [p.id for p in host.params]
    if listed and listed != described:
        raise ProfilerError(f"engine.plugins and plugin.describe disagree about the parameters of `{plugin_id}`")
    return host, rig


def _text_with_date(profile: dict[str, Any], profiled: str) -> str:
    clone = copy.deepcopy(profile)
    clone["identity"]["profiled"] = profiled
    return profiles.dump_profile(clone)


def _existing(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return profiles.load_profile(path, validate=False)
    except ProfileError as exc:
        log.warning("cannot read the existing %s: %s", path, exc)
        return None


def generate_builtins(
    studio: StudioLike,
    out_dir: str | Path,
    curated_dir: str | Path | None = None,
    *,
    date: str | None = None,
    today: str | None = None,
    calibrate_params: bool = False,
) -> BuiltinsReport:
    """Write one profile per built-in plugin to ``out_dir`` and report what did not line up.

    ``date`` forces ``identity.profiled`` on every profile; without it the date of an existing
    file is kept while the content is unchanged and ``today`` (default: the current date) is
    used when the content changed or the file is new. ``calibrate_params`` adds the
    ``plugin.param_text`` calibration (large; off by default for built-ins, whose text is a
    fixed format of the value).
    """
    if date is not None:
        datetime.date.fromisoformat(date)
    stamp = date or today or datetime.date.today().isoformat()
    datetime.date.fromisoformat(stamp)
    out = Path(out_dir)
    curated = Path(curated_dir) if curated_dir is not None else default_curated_dir()
    report = BuiltinsReport()

    with ScratchSession(studio, name="Atlas builtins"):
        rows = _plugin_rows(studio)
        report.plugin_count = len(rows)
        effect_track = _track(studio, "audio")
        instrument_track = _track(studio, "instrument")
        for row in rows:
            plugin_id = str(row["id"])
            try:
                profile = _build_one(studio, row, effect_track, instrument_track, curated, stamp, calibrate_params)
            except (ProfilerError, ProfileError, StudioError, ValueError) as exc:
                report.errors[plugin_id] = str(exc)
                continue
            report.profiles[plugin_id] = profile
            _store(report, out, profile, date)

    wanted = {str(r["id"]) for r in rows}
    have = set(curated_ids(curated))
    report.curated_without_plugin = sorted(have - wanted)
    report.plugins_without_curated = sorted(wanted - have)
    produced = {profiles.profile_relpath(p).name for p in report.profiles.values()}
    if out.is_dir():
        report.stale_files = sorted(f for f in out.glob("*.yaml") if f.is_file() and f.name not in produced)
    return report


def _track(studio: StudioLike, kind: str) -> int:
    result = studio.execute("track.new", {"kind": kind, "format": "stereo", "name": "Atlas builtins"})
    tracks = result.get("tracks") if isinstance(result, Mapping) else None
    if not (isinstance(tracks, list) and tracks and isinstance(tracks[0], int) and not isinstance(tracks[0], bool)):
        raise ProfilerError("track.new did not return a track id")
    return tracks[0]


def _build_one(
    studio: StudioLike,
    row: Mapping[str, Any],
    effect_track: int,
    instrument_track: int,
    curated: Path,
    stamp: str,
    calibrate_params: bool,
) -> dict[str, Any]:
    plugin_id = str(row["id"])
    host, rig = _describe_builtin(studio, row, effect_track, instrument_track)
    profile = build_host_profile(host, profiled=stamp, os_names=BUILTIN_OS, os_provenance=BUILTIN_OS_PROVENANCE)
    if calibrate_params:
        by_id = {hp.id: hp for hp in host.params}
        warnings: list[str] = []
        for entry in profile["params"]:
            hp = by_id[str(entry["id"])]
            points: list[CalibrationPoint] = calibrate(studio, rig, hp)
            _apply_calibration(entry, hp, points, warnings)
        for w in warnings:
            log.info("%s: %s", plugin_id, w)
    patch_path = curated / f"{plugin_id}.yaml"
    if patch_path.is_file():
        profile = merge_patch(profile, load_patch(patch_path)).profile
    return profile


def _store(report: BuiltinsReport, out: Path, profile: dict[str, Any], forced_date: str | None) -> None:
    path = out / profiles.profile_relpath(profile).name
    plugin_id = profile["identity"]["plugin_id"]
    existing = _existing(path)
    if existing is not None:
        ident = existing.get("identity")
        old_stamp = ident.get("profiled") if isinstance(ident, dict) else None
        if forced_date is None:
            same = isinstance(old_stamp, str) and _text_with_date(profile, old_stamp) == profiles.dump_profile(existing)
        else:
            same = profiles.dump_profile(profile) == profiles.dump_profile(existing)
        if same:
            report.unchanged.append(path)
            if forced_date is None:
                report.profiles[plugin_id] = existing
            return
    try:
        profiles.write_profile(path, profile)
    except (ProfileError, OSError) as exc:
        report.errors[plugin_id] = str(exc)
        return
    report.written.append(path)
