"""``atlas profile``: draft a plugin profile from a running SoundCraft (DESIGN 6.2).

    atlas profile --plugin <id> [--studio PORT | --headless BINARY] [--out PATH] [--no-measure]

Registered into the ``atlas`` CLI through ``register_commands`` (see ``atlas.cli``). The heavy
imports (numpy, scipy, the Studio client) happen when the command runs, so ``atlas validate``
stays quick.
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
    p.add_argument("--plugin", required=True, help="plugin id: a built-in id (eq_7band) or clap:/vst3:/au: id")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--studio", type=int, metavar="PORT", help="profile through the control channel of a running SoundCraft app")
    where.add_argument(
        "--headless",
        metavar="BINARY",
        help=f"spawn `BINARY mcp` and profile there (default: $SOUNDCRAFT_CLI or {DEFAULT_BINARY})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="a .yaml file to write, or a root folder (default ./atlas-drafts) that gets <format>/<vendor>/<name>@<major>.yaml",
    )
    p.add_argument("--no-measure", action="store_true", help="host data and calibration only: skip the offline bounces")
    p.add_argument("--max-params", type=int, default=128, metavar="N", help="measure at most N parameters (default 128)")
    p.add_argument("--max-bounce-s", type=float, default=9.0, metavar="S", help="longest bounce, which also sets the test signal length (3..60, default 9)")
    p.add_argument("--param", action="append", default=None, metavar="ID", help="measure only this parameter id (repeatable)")
    p.add_argument(
        "--allow-current-session",
        action="store_true",
        help="with --studio: allow working inside the app's open session (scratch tracks are undone afterwards; the session ends up marked modified)",
    )
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD", help="identity.profiled (default today)")
    p.add_argument("--quiet", "-q", action="store_true", help="no progress on stderr")


def _run(args: argparse.Namespace) -> int:
    from studio_client import Studio, StudioError

    from .profiler import ProfilerError, ProfilerOptions, profile_plugin, write_draft
    from .validate import ProfileError

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr)

    options = ProfilerOptions(
        measure=not args.no_measure,
        max_params=args.max_params,
        max_bounce_s=args.max_bounce_s,
        only_params=args.param,
        allow_current_session=args.allow_current_session,
        date=args.date,
        progress=progress,
    )
    try:
        if args.studio is not None:
            studio = Studio.connect(args.studio)
        else:
            studio = Studio.headless(args.headless or os.environ.get("SOUNDCRAFT_CLI") or DEFAULT_BINARY)
    except StudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        result = profile_plugin(studio, args.plugin, options)
        path = write_draft(result.profile, args.out)
    except (ProfilerError, StudioError, ProfileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        studio.close()
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    params = len(result.profile["params"])
    print(f"{path}")
    print(
        f"{result.profile['identity']['name']}: {params} parameters, {result.measured} measured in {result.bounces} bounces, "
        f"{result.elapsed_s:g}s; draft written (review it, then `atlas merge` knowledge from the manual)",
        file=sys.stderr,
    )
    return 0


def register_commands(registry: dict[str, Subcommand]) -> None:
    """Add ``profile`` to the CLI's subcommand registry."""
    from .cli import Subcommand, register

    register("profile", Subcommand("draft a profile of a plugin from a running SoundCraft", _configure, _run), registry)
