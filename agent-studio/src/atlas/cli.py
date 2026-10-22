"""``atlas`` command line.

Subcommands live in the ``SUBCOMMANDS`` registry. To add one (the profiler's ``profile`` and
``builtins`` commands do this), either call ``register(...)`` / add an entry below, or ship a
module listed in ``EXTENSION_MODULES`` that defines ``register_commands(registry)`` and
fills the registry; it is imported (if present) every time the CLI starts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import profiles, validate
from .merge import MergeError, merge_files
from .validate import Issue, ProfileError


@dataclass(frozen=True)
class Subcommand:
    """``configure`` adds arguments to the subparser; ``run`` returns the exit code."""

    help: str
    configure: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]


# ---- EXTENSION POINT: the subcommand registry --------------------------------------
# name -> Subcommand. Other modules add entries (e.g. "profile", "builtins") via register()
# or by being listed in EXTENSION_MODULES.
SUBCOMMANDS: dict[str, Subcommand] = {}
EXTENSION_MODULES: tuple[str, ...] = ("atlas.cli_profile", "atlas.cli_builtins")
# -------------------------------------------------------------------------------------


def register(name: str, command: Subcommand, registry: dict[str, Subcommand] | None = None) -> None:
    """Add a subcommand; refuses to replace an existing one."""
    target = SUBCOMMANDS if registry is None else registry
    if name in target:
        raise ValueError(f"subcommand `{name}` is already registered")
    target[name] = command


def default_profiles_root() -> Path:
    """``$ATLAS_PROFILES``, else the checkout's ``atlas/profiles``, else ``./atlas/profiles``."""
    env = os.environ.get("ATLAS_PROFILES")
    if env:
        return Path(env)
    checkout = Path(__file__).resolve().parents[2] / "atlas" / "profiles"
    return checkout if checkout.is_dir() else Path.cwd() / "atlas" / "profiles"


def _add_root(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", type=Path, default=None, help="profiles folder (default: $ATLAS_PROFILES or atlas/profiles)")


def _root(args: argparse.Namespace) -> Path:
    return Path(args.root) if args.root else default_profiles_root()


def _print_issues(label: str, issues: Sequence[Issue]) -> None:
    for issue in issues:
        print(f"{label}: {issue}", file=sys.stderr)


# ---- validate -------------------------------------------------------------------------


def _validate_configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("paths", nargs="+", type=Path, help="profile files, or folders laid out as <format>/<vendor>/<name>@<major>.yaml")


def _validate_one(path: Path, layout_root: Path | None) -> list[Issue]:
    issues = validate.validate_file(path)
    if not issues and layout_root is not None:
        issues = profiles.check_layout(path, layout_root, profiles.load_profile(path, validate=False))
    return issues


def _validate_run(args: argparse.Namespace) -> int:
    failed = checked = 0
    for target in args.paths:
        if target.is_dir():
            items = [(p, target) for p in profiles.iter_profile_paths(target)]
        elif target.is_file():
            items = [(target, None)]
        else:
            print(f"{target}: no such file or folder", file=sys.stderr)
            failed += 1
            continue
        for path, layout_root in items:
            checked += 1
            issues = _validate_one(path, layout_root)
            if issues:
                failed += 1
                _print_issues(str(path), issues)
            else:
                print(f"{path}: ok")
    if checked == 0 and not failed:
        print("no profiles found", file=sys.stderr)
    return 1 if failed else 0


# ---- merge ----------------------------------------------------------------------------


def _merge_configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("profile", type=Path)
    p.add_argument("patch", type=Path)
    p.add_argument("--out", type=Path, default=None, help="write here instead of in place")
    p.add_argument("--skip-conflicts", action="store_true", help="apply the rest of the patch when some fields would overwrite firmer knowledge")
    p.add_argument("--dry-run", action="store_true", help="report the changes without writing")


def _merge_run(args: argparse.Namespace) -> int:
    try:
        result, written = merge_files(
            args.profile, args.patch, out_path=args.out, skip_conflicts=args.skip_conflicts, dry_run=args.dry_run
        )
    except MergeError as exc:
        print(f"merge refused: {exc}", file=sys.stderr)
        for conflict in exc.conflicts:
            print(f"  conflict: {conflict}", file=sys.stderr)
        return 1
    except ProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for change in result.changes:
        print(f"set {change.path} ({change.provenance['source']}, {change.provenance['confidence']})")
    for conflict in result.conflicts:
        print(f"skipped {conflict}", file=sys.stderr)
    print(f"{len(result.changes)} field(s) set, {len(result.conflicts)} skipped" + (f"; wrote {written}" if written else " (dry run)"))
    return 0


# ---- list / get / check ---------------------------------------------------------------


def _row(ref: profiles.ProfileRef, root: Path) -> dict[str, Any]:
    ident = ref.profile["identity"]
    try:
        shown = ref.path.relative_to(root).as_posix()
    except ValueError:
        shown = str(ref.path)
    return {
        "plugin_id": ident["plugin_id"], "name": ident["name"], "vendor": ident["vendor"],
        "format": ident["format"], "version": ident["version"], "params": len(ref.profile["params"]), "path": shown,
    }


def _list_configure(p: argparse.ArgumentParser) -> None:
    _add_root(p)
    p.add_argument("--query", "-q", default=None, help="case-insensitive text in name, vendor, id or summary")
    p.add_argument("--format", dest="fmt", default=None, choices=["builtin", "clap", "vst3", "au"])
    p.add_argument("--json", action="store_true")


def _list_run(args: argparse.Namespace) -> int:
    root = _root(args)
    rows = []
    for ref in profiles.list_profiles(root):
        ident = ref.profile["identity"]
        if args.fmt and ident["format"] != args.fmt:
            continue
        if args.query:
            hay = " ".join([ident["name"], ident["vendor"], ident["plugin_id"], str(ref.profile.get("summary", ""))]).lower()
            if args.query.lower() not in hay:
                continue
        rows.append(_row(ref, root))
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"{r['plugin_id']}\t{r['name']}\t{r['vendor']}\t{r['version']}\t{r['params']} params\t{r['path']}")
    return 0


def _lookup_configure(p: argparse.ArgumentParser) -> None:
    _add_root(p)
    p.add_argument("plugin_id")
    p.add_argument("--version", default=None, help="the installed plugin version to compare against")
    p.add_argument("--json", action="store_true")


def _get_run(args: argparse.Namespace) -> int:
    root = _root(args)
    match = profiles.find_profile(root, args.plugin_id, args.version)
    if match is None:
        print(f"no profile for {args.plugin_id} under {root}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"path": str(match.path), "version_match": match.version_match, "version_status": match.status, "version": None if args.version is None else profiles.version_check(match.profile, args.version), "profile": match.profile}, indent=2))
    else:
        print(f"# {match.path}  version_status: {match.status}")
        print(profiles.dump_profile(match.profile), end="")
    return 0


def _check_configure(p: argparse.ArgumentParser) -> None:
    _add_root(p)
    p.add_argument("--plugin", required=True, help="plugin id, e.g. vst3:ABCD...")
    p.add_argument("--version", required=True, help="the installed plugin version")
    p.add_argument("--json", action="store_true", help="print {matches, profile_version, actual_version, status, path}")


def _check_run(args: argparse.Namespace) -> int:
    match = profiles.find_profile(_root(args), args.plugin, args.version)
    if match is None:
        if args.json:
            print(json.dumps({"matches": False, "profile_version": None, "actual_version": args.version, "status": "missing", "path": None}))
        else:
            print(f"{args.plugin}: no profile; profile it")
        return 1
    check = profiles.version_check(match.profile, args.version)
    if args.json:
        print(json.dumps({**check, "path": str(match.path)}))
    elif check["matches"]:
        print(f"{args.plugin}: profile matches version {check['profile_version']}")
    else:
        print(f"{args.plugin}: profile is for {check['profile_version']} ({check['status']}), plugin is {args.version}: re-profile")
    return 0 if check["matches"] else 1


SUBCOMMANDS.update(
    {
        "validate": Subcommand("validate profile files or folders against the schema", _validate_configure, _validate_run),
        "merge": Subcommand("apply a knowledge patch to a profile, recording provenance", _merge_configure, _merge_run),
        "list": Subcommand("list profiles", _list_configure, _list_run),
        "get": Subcommand("print the profile for a plugin id", _lookup_configure, _get_run),
        "check": Subcommand("exit 1 when the profile's version differs from the installed plugin's", _check_configure, _check_run),
    }
)


# ---- entry point ----------------------------------------------------------------------


def load_extensions(registry: dict[str, Subcommand], modules: Sequence[str] = EXTENSION_MODULES) -> None:
    """Import each extension module that exists and let it add its subcommands."""
    for name in modules:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name == name:
                continue  # not written yet
            raise
        module.register_commands(registry)


def build_parser(registry: dict[str, Subcommand]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas", description="Plugin Atlas: profiles of what plugins' parameters do")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name in sorted(registry):
        command = registry[name]
        command.configure(sub.add_parser(name, help=command.help, description=command.help))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    registry = dict(SUBCOMMANDS)
    load_extensions(registry)
    args = build_parser(registry).parse_args(argv)
    try:
        return registry[args.command].run(args)
    except BrokenPipeError:
        return 1


if __name__ == "__main__":
    sys.exit(main())
