"""Spawns and supervises the worker subprocesses (DESIGN 3.2): load/unload/run/cancel, restarts with
backoff, and a memory budget with least-recently-used eviction of idle models."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ForgeConfig
from .errors import BadRequest, Conflict, NotFound
from .workers.protocol import MAX_LINE_BYTES, PROTOCOL_VERSION, ModelInfo, ProtocolError, validate_model_info, validate_outputs

log = logging.getLogger("forge.manager")

ProgressCallback = Callable[[float, str], None]
IN_PROCESS_FAMILY = "analysis"


class WorkerError(RuntimeError):
    """A worker reported a failure for a request."""


class WorkerDied(WorkerError):
    """The worker process exited, timed out or broke the protocol."""


class JobCancelled(Exception):
    """The job ended because it was cancelled."""


@dataclass
class _Pending:
    future: asyncio.Future[dict[str, Any]]
    on_progress: ProgressCallback | None


class _Worker:
    """One family's subprocess and its request/reply plumbing."""

    def __init__(self, family: str) -> None:
        self.family = family
        self.proc: asyncio.subprocess.Process | None = None
        self.ready = False
        self.failures = 0
        self.next_start_at = 0.0
        self.last_error = ""
        self.intentional_stop = False
        self.on_exit: Callable[[], None] | None = None
        self.op_lock = asyncio.Lock()  # one load/unload/run at a time
        self.start_lock = asyncio.Lock()
        self._next_id = 0
        self._pending: dict[int, _Pending] = {}
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def spawn(self, argv: list[str], env: dict[str, str], cwd: Path) -> None:
        self.intentional_stop = False
        cwd.mkdir(parents=True, exist_ok=True)
        self.proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=env, cwd=cwd, limit=MAX_LINE_BYTES,
        )
        self._tasks = [asyncio.create_task(self._read_stdout(self.proc)), asyncio.create_task(self._forward_stderr(self.proc))]

    async def _forward_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        while True:
            try:
                line = await proc.stderr.readline()
            except (ValueError, asyncio.LimitOverrunError):
                continue
            if not line:
                return
            log.info("[%s] %s", self.family, line.decode("utf-8", "replace").rstrip()[:2000])

    async def _read_stdout(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        reason = "worker exited"
        try:
            while True:
                try:
                    line = await proc.stdout.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    reason = "worker sent an oversized line"
                    proc.kill()
                    break
                if not line:
                    break
                self._dispatch(line)
        finally:
            await proc.wait()
            self.ready = False
            if self.on_exit is not None:
                self.on_exit()
            self._fail_pending(WorkerDied(f"{reason} (code {proc.returncode})"))

    def _dispatch(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except ValueError:
            log.warning("[%s] ignoring non-JSON line from worker", self.family)
            return
        req_id = message.get("id") if isinstance(message, dict) else None
        pending = self._pending.get(req_id) if isinstance(req_id, int) else None
        if pending is None:
            log.warning("[%s] reply for unknown request: %.200r", self.family, message)
            return
        if message.get("event") == "progress":
            if pending.on_progress is not None:
                try:
                    pending.on_progress(float(message.get("progress", 0.0)), str(message.get("message", "")))
                except (TypeError, ValueError):
                    log.warning("[%s] malformed progress event", self.family)
            return
        if not pending.future.done():
            pending.future.set_result(message)

    def _fail_pending(self, error: Exception) -> None:
        for pending in list(self._pending.values()):
            if not pending.future.done():
                pending.future.set_exception(error)
        self._pending.clear()

    async def request(self, op: str, timeout: float, on_progress: ProgressCallback | None = None, **fields: Any) -> dict[str, Any]:
        """Send one request and wait for its reply; raises :class:`WorkerDied` if the process goes away."""
        if not self.alive or self.proc is None or self.proc.stdin is None:
            raise WorkerDied("worker is not running")
        self._next_id += 1
        req_id = self._next_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = _Pending(future, on_progress)
        try:
            self.proc.stdin.write((json.dumps({"id": req_id, "op": op, **fields}, allow_nan=False) + "\n").encode("utf-8"))
            await self.proc.stdin.drain()
            return await asyncio.wait_for(future, timeout)
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise WorkerDied("worker closed its input") from exc
        except asyncio.TimeoutError as exc:
            await self.kill(intentional=False)
            raise WorkerDied(f"worker did not answer {op!r} within {timeout:g} s") from exc
        finally:
            self._pending.pop(req_id, None)

    async def kill(self, intentional: bool = True) -> None:
        self.intentional_stop = intentional
        if self.proc is not None and self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()
        await self._join()

    async def shutdown(self, grace_s: float = 3.0) -> None:
        self.intentional_stop = True
        if self.alive and self.proc is not None and self.proc.stdin is not None:
            try:
                self.proc.stdin.write(b'{"id":0,"op":"shutdown"}\n')
                await self.proc.stdin.drain()
                await asyncio.wait_for(self.proc.wait(), grace_s)
            except (BrokenPipeError, ConnectionResetError, asyncio.TimeoutError):
                pass
        await self.kill()

    async def _join(self) -> None:
        tasks, self._tasks = self._tasks, []
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=2.0)
            for task in pending:
                task.cancel()


@dataclass
class _ModelState:
    info: ModelInfo
    loaded: bool = False
    in_use: int = 0
    last_used: float | None = None  # wall clock
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    tick: int = 0  # LRU ordering that survives clock resolution


class WorkerManager:
    """Owns one subprocess per enabled family (``analysis`` runs in the gateway and has none)."""

    def __init__(
        self,
        config: ForgeConfig,
        *,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 30.0,
        hello_timeout_s: float = 120.0,
        load_timeout_s: float = 900.0,
        run_timeout_s: float = 3600.0,
        cancel_grace_s: float = 10.0,
    ) -> None:
        self.config = config
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.hello_timeout_s = hello_timeout_s
        self.load_timeout_s = load_timeout_s
        self.run_timeout_s = run_timeout_s
        self.cancel_grace_s = cancel_grace_s
        self._workers: dict[str, _Worker] = {
            f: _Worker(f) for f in config.enabled_families() if f != IN_PROCESS_FAMILY
        }
        for family, worker in self._workers.items():
            worker.on_exit = lambda family=family: self._forget_loaded(family)
        self._background: set[asyncio.Task[None]] = set()
        self._models: dict[str, _ModelState] = {}
        self._active: dict[str, _Worker] = {}
        self._cancelled: set[str] = set()
        self._mem_lock = asyncio.Lock()
        self._tick = 0
        self._closed = False

    # ---- lifecycle ----

    async def start(self) -> None:
        """Spawn every enabled family and read its models. A family that fails is reported, not fatal."""
        await asyncio.gather(*(self._try_ensure(f) for f in self._workers))

    async def _try_ensure(self, family: str) -> None:
        try:
            await self._ensure(family)
        except WorkerDied as exc:
            log.warning("family %s is unavailable: %s", family, exc)

    async def close(self) -> None:
        self._closed = True
        await asyncio.gather(*(w.shutdown() for w in self._workers.values()))

    def _backoff(self, failures: int) -> float:
        return min(self.backoff_max_s, self.backoff_base_s * 2 ** max(0, failures - 1))

    async def _ensure(self, family: str) -> _Worker:
        worker = self._workers[family]
        async with worker.start_lock:
            if worker.alive and worker.ready:
                return worker
            if self._closed:
                raise WorkerDied("forge is shutting down")
            if worker.proc is not None:  # died since last use
                unexpected = not worker.intentional_stop
                await worker.kill()
                self._mark_down(worker, unexpected)
            delay = worker.next_start_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                await self._spawn(worker)
            except WorkerDied as exc:
                worker.failures += 1
                worker.next_start_at = time.monotonic() + self._backoff(worker.failures)
                worker.last_error = str(exc)
                await worker.kill()
                raise
            return worker

    def _forget_loaded(self, family: str) -> None:
        for state in self._models.values():
            if state.info["family"] == family:
                state.loaded = False

    def _mark_down(self, worker: _Worker, unexpected: bool) -> None:
        """Record a death: models are no longer loaded and, if unexpected, the restart is delayed (backoff)."""
        self._forget_loaded(worker.family)
        if unexpected:
            worker.failures += 1
            worker.next_start_at = time.monotonic() + self._backoff(worker.failures)
        worker.proc = None

    async def _spawn(self, worker: _Worker) -> None:
        cfg = self.config.family(worker.family)
        env = {k: v for k, v in os.environ.items() if k != self.config.token_env}  # workers never see the bearer token
        env.update(cfg.env)
        env["PYTHONUNBUFFERED"] = "1"
        cwd = self.config.data_dir / "workers" / worker.family
        try:
            await worker.spawn([cfg.python or "python", "-m", f"forge.workers.{worker.family}"], env, cwd)
        except OSError as exc:
            raise WorkerDied(f"cannot start {cfg.python}: {exc.strerror or exc}") from exc
        reply = await worker.request("hello", self.hello_timeout_s, options=cfg.options)
        if not reply.get("ok"):
            raise WorkerDied(f"hello failed: {reply.get('error')}")
        result = reply.get("result")
        try:
            if not isinstance(result, dict) or result.get("protocol") != PROTOCOL_VERSION or result.get("family") != worker.family:
                raise ProtocolError("unexpected hello reply")
            models = [validate_model_info(m, worker.family) for m in result.get("models", [])]
        except ProtocolError as exc:
            raise WorkerDied(f"bad hello reply: {exc}") from exc
        for info in models:
            state = self._models.get(info["id"])
            if state is None:
                self._models[info["id"]] = _ModelState(info)
            else:
                state.info, state.loaded = info, False
        worker.ready = True
        worker.last_error = ""

    # ---- inspection ----

    def models(self) -> list[dict[str, Any]]:
        """Every known model with ``loaded``, ``warm``, ``last_used`` and ``recent_seconds``."""
        out = []
        for state in self._models.values():
            out.append(
                {
                    **state.info, "loaded": state.loaded, "warm": state.loaded, "in_use": state.in_use > 0,
                    "last_used": state.last_used, "recent_seconds": self._recent(state),
                    "lane": self.config.family(state.info["family"]).lane,
                }
            )
        return out

    def model(self, model_id: str) -> dict[str, Any] | None:
        state = self._models.get(model_id)
        return None if state is None else next(m for m in self.models() if m["id"] == model_id)

    def family_status(self) -> dict[str, dict[str, Any]]:
        return {
            f: {"state": "ready" if w.alive and w.ready else "down", "failures": w.failures, "error": w.last_error}
            for f, w in self._workers.items()
        }

    def lane_of(self, model_id: str) -> str:
        state = self._models.get(model_id)
        if state is None:
            raise NotFound(f"unknown model {model_id!r}", code="model_not_found")
        return self.config.family(state.info["family"]).lane or "gpu"

    @staticmethod
    def _recent(state: _ModelState) -> float | None:
        return round(sum(state.recent) / len(state.recent), 3) if state.recent else None

    def recent_seconds(self, model_id: str) -> float | None:
        state = self._models.get(model_id)
        return None if state is None else self._recent(state)

    @property
    def budget_gb(self) -> float:
        return self.config.memory_budget_gb

    @property
    def used_gb(self) -> float:
        return round(sum(s.info["vram_gb"] for s in self._models.values() if s.loaded), 3)

    def resolve_model(self, job_type: str, model_id: str | None = None) -> dict[str, Any]:
        """The requested model, or the first available one that runs ``job_type``."""
        if model_id is not None:
            info = self.model(model_id)
            if info is None:
                raise NotFound(f"unknown model {model_id!r}", code="model_not_found")
            if job_type not in info["job_types"]:
                raise BadRequest(f"model {model_id!r} does not run {job_type}", code="model_mismatch")
            if not info["available"]:
                raise Conflict(f"model {model_id!r} is unavailable: {info.get('reason', 'unknown reason')}", code="model_unavailable")
            return info
        for info in self.models():
            if info["available"] and job_type in info["job_types"]:
                return info
        raise NotFound(f"no available model runs {job_type}", code="no_model")

    # ---- model memory ----

    async def load(self, model_id: str) -> None:
        """Load a model, evicting least-recently-used idle models if the budget needs it (409 if it cannot fit)."""
        state = self._state(model_id)
        worker = await self._ensure(state.info["family"])
        async with worker.op_lock:
            await self._ensure_loaded(worker, state)

    async def unload(self, model_id: str) -> None:
        state = self._state(model_id)
        if state.in_use:
            raise Conflict(f"model {model_id!r} is running a job", code="model_busy")
        if not state.loaded:
            return
        worker = await self._ensure(state.info["family"])
        async with worker.op_lock, self._mem_lock:
            await self._unload_locked(worker, state)

    def _state(self, model_id: str) -> _ModelState:
        state = self._models.get(model_id)
        if state is None:
            raise NotFound(f"unknown model {model_id!r}", code="model_not_found")
        return state

    def _touch(self, state: _ModelState) -> None:
        self._tick += 1
        state.tick = self._tick
        state.last_used = time.time()

    async def _ensure_loaded(self, worker: _Worker, state: _ModelState) -> None:
        """Caller holds ``worker.op_lock``."""
        async with self._mem_lock:
            if state.loaded:
                self._touch(state)
                return
            if not state.info["available"]:
                raise Conflict(f"model {state.info['id']!r} is unavailable: {state.info.get('reason', '')}", code="model_unavailable")
            need = state.info["vram_gb"]
            if need > self.budget_gb:
                raise Conflict(f"model needs {need:g} GB, more than the {self.budget_gb:g} GB budget", code="over_budget")
            while self.used_gb + need > self.budget_gb:
                victim = self._lru_victim(worker, state)
                if victim is None:
                    raise Conflict(
                        f"cannot fit {need:g} GB: {self.used_gb:g} of {self.budget_gb:g} GB are in use by running or busy models",
                        code="over_budget",
                    )
                await self._unload_locked(self._workers[victim.info["family"]], victim)
            reply = await worker.request("load", self.load_timeout_s, model=state.info["id"])
            if not reply.get("ok"):
                raise WorkerError(f"loading {state.info['id']!r} failed: {reply.get('error')}")
            state.loaded = True
            self._touch(state)

    def _lru_victim(self, own: _Worker, wanted: _ModelState) -> _ModelState | None:
        candidates = [
            s for s in self._models.values()
            if s.loaded and s is not wanted and s.in_use == 0
            and (self._workers[s.info["family"]] is own or not self._workers[s.info["family"]].op_lock.locked())
        ]
        return min(candidates, key=lambda s: s.tick, default=None)

    async def _unload_locked(self, worker: _Worker, state: _ModelState) -> None:
        """Caller holds ``_mem_lock`` and either owns ``worker.op_lock`` or found it idle."""
        own = worker.op_lock.locked()
        if not own:
            await worker.op_lock.acquire()
        try:
            try:
                reply = await worker.request("unload", self.load_timeout_s, model=state.info["id"])
            except WorkerDied:
                state.loaded = False  # a dead worker holds no memory
                return
            if not reply.get("ok"):
                raise WorkerError(f"unloading {state.info['id']!r} failed: {reply.get('error')}")
            state.loaded = False
        finally:
            if not own:
                worker.op_lock.release()

    # ---- jobs ----

    async def run(self, job: dict[str, Any], on_progress: ProgressCallback | None = None) -> list[dict[str, Any]]:
        """Run ``job`` (a ``RunJob``) on its model's worker, loading the model first. Returns validated outputs.

        Raises :class:`JobCancelled`, :class:`WorkerDied` (the worker is restarted on next use) or
        :class:`WorkerError` (the worker reported a failure)."""
        job_id = str(job["job_id"])
        state = self._state(str(job["model"]))
        try:
            worker = await self._ensure(state.info["family"])
            started = time.monotonic()
            async with worker.op_lock:
                state.in_use += 1
                self._active[job_id] = worker
                try:
                    await self._ensure_loaded(worker, state)
                    if job_id in self._cancelled:  # cancelled while the model was loading
                        raise JobCancelled()
                    reply = await worker.request("run", self.run_timeout_s, on_progress, job=job)
                except WorkerDied as exc:
                    if job_id in self._cancelled:
                        raise JobCancelled() from exc
                    if not worker.intentional_stop:
                        self._mark_down(worker, unexpected=True)
                    raise
                finally:
                    state.in_use -= 1
                    self._active.pop(job_id, None)
                    self._touch(state)
            if not reply.get("ok"):
                error = str(reply.get("error", "unknown error"))
                if error == "cancelled" or job_id in self._cancelled:
                    raise JobCancelled()
                raise WorkerError(error)
            try:
                outputs = validate_outputs((reply.get("result") or {}).get("outputs"), Path(job["out_dir"]))
            except ProtocolError as exc:
                raise WorkerError(f"bad worker output: {exc}") from exc
            worker.failures = 0
            state.recent.append(time.monotonic() - started)
            return [dict(o) for o in outputs]
        finally:
            self._cancelled.discard(job_id)

    async def cancel(self, job_id: str) -> bool:
        """Ask the worker running ``job_id`` to stop; kill it if it has not stopped after the grace period."""
        worker = self._active.get(job_id)
        if worker is None:
            return False
        self._cancelled.add(job_id)
        try:
            await worker.request("cancel", max(self.cancel_grace_s, 1.0), job_id=job_id)
        except WorkerDied:
            return True
        task = asyncio.get_running_loop().create_task(self._kill_after_grace(job_id, worker))
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return True

    async def _kill_after_grace(self, job_id: str, worker: _Worker) -> None:
        await asyncio.sleep(self.cancel_grace_s)
        if self._active.get(job_id) is worker:
            log.warning("job %s ignored cancel for %.1f s; killing %s", job_id, self.cancel_grace_s, worker.family)
            await worker.kill()
