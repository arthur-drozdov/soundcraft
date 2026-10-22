"""``atlas builtins``: regenerate the profiles of SoundCraft's built-in plugins (DESIGN 6.2).

    atlas builtins --binary soundcraft-cli [--out atlas/profiles/builtin/soundcraft]

Host fields come from ``engine.plugins`` + ``plugin.describe`` on a headless scratch session;
the curated knowledge in ``atlas/curated/builtin/<id>.yaml`` is merged over them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cli import Subcommand

DEFAULT_BINARY = "soundcraft-cli"


def _configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("--binary", default=None, help=f"the soundcraft-cli to spawn (default: $SOUNDCRAFT_CLI or {DEFAULT_BINARY})")
    p.add_argument("--out", type=Path, default=None, help="folder for the profile files (default: <profiles root>/builtin/soundcraft)")
    p.add_argument("--curated", type=Path, default=None, help="folder of curated patches (default: atlas/curated/builtin of the checkout)")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD", help="set identity.profiled on every profile (default: keep the date of unchanged profiles, else today)")
    p.add_argument("--calibrate", action="store_true", help="also record plugin.param_text calibration points (large)")
    p.add_argument("--strict", action="store_true", help="exit 1 also when curated files match no plugin or plugins have no curated file")


def _run(args: argparse.Namespace) -> int:
    from studio_client import Studio, StudioError

    from .builtins import generate_builtins
    from .cli import default_profiles_root
    from .profiler import ProfilerError

    out = args.out if args.out is not None else default_profiles_root() / "builtin" / "soundcraft"
    try:
        studio = Studio.headless(args.binary or os.environ.get("SOUNDCRAFT_CLI") or DEFAULT_BINARY)
    except StudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        report = generate_builtins(studio, out, args.curated, date=args.date, calibrate_params=args.calibrate)
    except (ProfilerError, StudioError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        studio.close()

    for path in report.written:
        print(f"wrote {path}")
    for path in report.unchanged:
        print(f"unchanged {path}")
    for plugin_id, why in sorted(report.errors.items()):
        print(f"error: {plugin_id}: {why}", file=sys.stderr)
    if report.curated_without_plugin:
        print("curated files that match no plugin: " + ", ".join(report.curated_without_plugin), file=sys.stderr)
    if report.plugins_without_curated:
        print("plugins without curated knowledge: " + ", ".join(report.plugins_without_curated), file=sys.stderr)
    for path in report.stale_files:
        print(f"note: {path} matches no current plugin (left alone)", file=sys.stderr)
    print(f"{report.plugin_count} plugins: {len(report.written)} written, {len(report.unchanged)} unchanged, {len(report.errors)} failed")
    if report.errors:
        return 1
    if args.strict and (report.curated_without_plugin or report.plugins_without_curated):
        return 1
    return 0


def register_commands(registry: dict[str, Subcommand]) -> None:
    """Add ``builtins`` to the CLI's subcommand registry."""
    from .cli import Subcommand, register

    register("builtins", Subcommand("regenerate the built-in plugin profiles from a headless SoundCraft", _configure, _run), registry)
