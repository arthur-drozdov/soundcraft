"""The ``forge`` command (DESIGN 3.8): ``serve``, ``check``, ``purge`` and ``models``.

Exit codes: 0 success, 1 a check failed or a server could not run, 2 bad usage or configuration.
``check`` and ``models`` run on a throwaway data directory, so they never touch the job database or
assets of a Forge that is serving from the configured ``data_dir``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from audiokit import wav

from .config import ForgeConfig, load_config
from .errors import ConfigError, ForgeError
from .jobs import FINAL_STATES
from .service import ForgeService
from .store import AssetStore

if TYPE_CHECKING:
    import uvicorn

log = logging.getLogger("forge.cli")

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_CHECK_TIMEOUT_S = 900.0
CHECK_INPUT_SECONDS = 1.5
# The tiny job `forge check` runs, by job type: (params, name of the input it needs). Cheapest first;
# a model gets the first of these it supports. render.session needs a session archive and is skipped.
CHECK_JOBS: dict[str, tuple[dict[str, Any], str | None]] = {
    "generate.sfx": ({"duration_s": 0.5}, None),
    "generate.sample": ({"note": 60, "length_s": 0.5}, None),
    "generate.loop": ({"bars": 1, "bpm": 120}, None),
    "generate.instrument": ({"name": "check", "note_low": 60, "note_high": 60, "length_s": 0.5}, None),
    "transcribe.melody": ({}, "audio"),
    "analyze.audio": ({}, "audio"),
    "transform.audio": ({"strength": 0.3}, "audio"),
    "extend.audio": ({"seconds": 1}, "audio"),
    "inpaint.audio": ({"start_s": 0.1, "end_s": 0.3}, "audio"),
    "layer.add": ({"description": "check"}, "audio"),
    "stems.split": ({}, "audio"),
}


# ---- serving ----


def make_server(app: Any, host: str, port: int, *, log_level: str = "info") -> uvicorn.Server:
    """A uvicorn server that leaves signal handling and the app lifespan to the caller."""
    import uvicorn

    class Server(uvicorn.Server):
        def install_signal_handlers(self) -> None:  # older uvicorn
            return None

        @contextlib.contextmanager
        def capture_signals(self) -> Iterator[None]:  # newer uvicorn
            yield

    config = uvicorn.Config(
        app, host=host, port=port, log_level=log_level.lower(), log_config=None, lifespan="off",
        server_header=False, timeout_graceful_shutdown=5, ws="none",
    )
    return Server(config)


async def run_servers(
    servers: Sequence[uvicorn.Server], *, handle_signals: bool = True, on_stop: Callable[[], None] | None = None
) -> int:
    """Serve until a signal arrives (second signal: stop at once). Returns 1 if any server failed to run.

    ``on_stop`` runs when shutdown starts, e.g. to end long-lived responses that would delay it."""
    failures: list[BaseException] = []

    def stop(force: bool = False) -> None:
        if on_stop is not None:
            on_stop()
        for server in servers:
            server.should_exit = True
            server.force_exit = server.force_exit or force

    async def run(server: uvicorn.Server) -> None:
        try:
            await server.serve()
        except SystemExit as exc:  # uvicorn exits on startup failures such as a port in use
            failures.append(exc)
            log.error("a server could not start (exit code %s)", exc.code)
            stop()

    if handle_signals:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, lambda: stop(force=any(s.should_exit for s in servers)))
    await asyncio.gather(*(run(s) for s in servers))
    return 1 if failures else 0


async def serve(config: ForgeConfig, host: str | None = None, port: int | None = None, log_level: str = "info") -> int:
    """Run the gateway (and the Gary shim apps when enabled) around one shared :class:`ForgeService`."""
    from .api import create_app, end_event_streams
    from .gary_shim import create_shim_apps

    token = config.require_token()  # fails before anything is started
    bind_host, bind_port = host or config.listen.host, port or config.listen.port
    if token is None:
        if bind_host not in LOOPBACK:
            raise ConfigError("insecure_no_auth is only allowed when listening on a loopback address")
        log.warning("authentication is DISABLED (insecure_no_auth)")
    service = ForgeService(config)
    gateway = create_app(service, config, token=token)
    servers = [make_server(gateway, bind_host, bind_port, log_level=log_level)]
    if config.gary_shim.enabled:  # the shim has no authentication, so it only ever listens on loopback
        servers += [make_server(app, "127.0.0.1", p, log_level=log_level) for p, app in create_shim_apps(service, config).values()]
    await service.start()
    try:
        log.info("Forge serving on %s:%d (families: %s)", bind_host, bind_port, ", ".join(config.enabled_families()))
        return await run_servers(servers, on_stop=lambda: end_event_streams(gateway))
    finally:
        await service.stop()


# ---- check and models ----


def _scratch_service(config: ForgeConfig, data_dir: Path) -> ForgeService:
    return ForgeService(config.model_copy(update={"data_dir": data_dir}), purge_interval_s=None)


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows)


def model_rows(models: list[dict[str, Any]]) -> list[list[str]]:
    rows = [["FAMILY", "MODEL", "AVAILABLE", "VRAM_GB", "LICENCE / REASON"]]
    for m in sorted(models, key=lambda m: (m["family"], m["id"])):
        detail = m.get("licence", "") if m["available"] else f"unavailable: {m.get('reason', 'no reason given')}"
        rows.append([m["family"], m["id"], "yes" if m["available"] else "no", f"{m['vram_gb']:g}", str(detail)])
    return rows


async def list_models(config: ForgeConfig, as_json: bool = False) -> int:
    """Spawn every enabled family and print its models. Fails if an enabled family could not start."""
    with tempfile.TemporaryDirectory(prefix="forge-models-") as tmp:
        async with _scratch_service(config, Path(tmp)) as service:
            models = service.models()
            down = {f: s for f, s in service.manager.family_status().items() if s["state"] != "ready"}
    print(json.dumps(models, indent=2) if as_json else _table(model_rows(models)))
    for family, status in down.items():
        print(f"family {family} is down: {status['error'] or 'unknown error'}", file=sys.stderr)
    return 1 if down else 0


def _check_input(path: Path) -> Path:
    t = np.arange(int(CHECK_INPUT_SECONDS * 44100)) / 44100
    tone = 0.4 * np.sin(2 * np.pi * 440.0 * t) * np.minimum(1.0, np.minimum(t / 0.01, (CHECK_INPUT_SECONDS - t) / 0.02))
    wav.write(path, tone.astype(np.float32)[None, :], 44100, "16")
    return path


def pick_check_job(model: dict[str, Any]) -> str | None:
    return next((t for t in CHECK_JOBS if t in model["job_types"]), None)


async def run_check_job(service: ForgeService, model: dict[str, Any], audio_asset: str, timeout_s: float) -> tuple[str, str]:
    """Run the tiny job for ``model``. Returns ``(job type, "")`` on success or ``(job type, why it failed)``."""
    job_type = pick_check_job(model)
    if job_type is None:
        return "-", "skipped"
    params, needs = CHECK_JOBS[job_type]
    try:
        rec = await service.submit(job_type, dict(params), model=model["id"], inputs={needs: audio_asset} if needs else None)
        deadline = time.monotonic() + timeout_s
        while rec.state not in FINAL_STATES:
            if time.monotonic() > deadline:
                await service.cancel(rec.job_id)
                return job_type, f"no result after {timeout_s:g} s"
            await asyncio.sleep(0.1)
            rec = service.get_job(rec.job_id)
    except ForgeError as exc:
        return job_type, exc.message if not exc.details else f"{exc.message}: {exc.details}"
    if rec.state != "succeeded":
        return job_type, rec.error or rec.state
    if not (rec.result and (rec.result.asset_ids or rec.result.analysis)):
        return job_type, "the job succeeded but produced nothing"
    return job_type, ""


async def check(config: ForgeConfig, only: Sequence[str] = (), timeout_s: float = DEFAULT_CHECK_TIMEOUT_S) -> int:
    """Spawn every enabled family, print models and availability, run one tiny job per available model."""
    failed = 0
    with tempfile.TemporaryDirectory(prefix="forge-check-") as tmp:
        async with _scratch_service(config, Path(tmp)) as service:
            models = [m for m in service.models() if not only or m["id"] in only]
            print(_table(model_rows(models)))
            shown = {m["family"] for m in models}
            for family, status in service.manager.family_status().items():
                if status["state"] != "ready" and (not only or family in shown):
                    failed += 1
                    print(f"FAIL  family {family} is down: {status['error'] or 'unknown error'}")
            unknown = sorted(set(only) - {m["id"] for m in models})
            for model_id in unknown:
                failed += 1
                print(f"FAIL  {model_id}: no such model")
            asset = (await service.upload(_check_input(Path(tmp) / "check-input.wav"), "check-input.wav"))["asset_id"]
            print()
            for model in (m for m in models if m["available"]):
                started = time.monotonic()
                job_type, problem = await run_check_job(service, model, asset, timeout_s)
                label = f"{model['id']} ({job_type})"
                seconds = time.monotonic() - started
                if problem == "skipped":
                    print(f"skip  {model['id']}: no tiny job for {', '.join(model['job_types'])}")
                elif problem:
                    failed += 1
                    print(f"FAIL  {label}: {problem}")
                else:
                    print(f"ok    {label}  {seconds:.1f} s")
    print(f"\n{'check failed' if failed else 'check passed'}" + (f": {failed} problem(s)" if failed else ""))
    return 1 if failed else 0


# ---- purge ----


def purge(config: ForgeConfig, days: int | None = None) -> int:
    """Delete unrated, unused assets older than the retention window."""
    store = AssetStore(config.data_dir)
    try:
        removed = store.purge(config.retention_days if days is None else days)
    finally:
        store.close()
    print(f"purged {len(removed)} asset(s)")
    return 0


# ---- argument parsing ----


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="path to forge.yaml (default: $FORGE_CONFIG, else built-in defaults)")
    common.add_argument("--log-level", choices=["debug", "info", "warning", "error"], help="default: info for serve, warning otherwise")
    parser = argparse.ArgumentParser(prog="forge", description="Forge: the Agent Studio generation service")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("serve", parents=[common], help="run the HTTP gateway")
    p.add_argument("--host", help="override listen.host")
    p.add_argument("--port", type=int, help="override listen.port")
    p = sub.add_parser("check", parents=[common], help="spawn every enabled family and run one tiny job per available model")
    p.add_argument("--model", action="append", default=[], help="only check this model id (repeatable)")
    p.add_argument("--timeout", type=float, default=DEFAULT_CHECK_TIMEOUT_S, help="seconds to wait for each job (default %(default)g)")
    p = sub.add_parser("purge", parents=[common], help="delete unrated, unused assets older than retention_days")
    p.add_argument("--days", type=int, help="override retention_days")
    p = sub.add_parser("models", parents=[common], help="spawn every enabled family and list its models")
    p.add_argument("--json", action="store_true", help="print the model list as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = args.log_level or ("info" if args.command == "serve" else "warning")
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(args.config)
        if args.command == "serve":
            return asyncio.run(serve(config, args.host, args.port, level))
        if args.command == "check":
            return asyncio.run(check(config, args.model, args.timeout))
        if args.command == "models":
            return asyncio.run(list_models(config, args.json))
        return purge(config, args.days)
    except ConfigError as exc:
        print(f"forge: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
