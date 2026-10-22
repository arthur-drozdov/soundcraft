"""The Forge core service: job submission, lane queues, execution, post-processing and storage."""

from __future__ import annotations

import asyncio
import logging
import secrets
import shutil
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from audiokit import analysis, wav

from .config import ForgeConfig
from .errors import BadRequest, ForgeError, NotFound, ValidationFailed
from .jobdb import JobDB
from .jobs import (
    FINAL_STATES,
    INPUT_KINDS,
    PRIORITIES,
    PRIORITY_RANK,
    JobRecord,
    JobResult,
    ValidatedJob,
    validate_job,
)
from .manager import JobCancelled, WorkerDied, WorkerError, WorkerManager
from .postprocess import ModelMeta, PostContext, PostprocessError, process_output
from .store import AssetStore, json_safe, utc_now

log = logging.getLogger("forge.service")

ANALYSIS_MODEL = "analysis"
IDEMPOTENCY_WINDOW = timedelta(hours=24)
EVENT_QUEUE_SIZE = 1000
WORKER_PROGRESS_SHARE = 0.95  # the rest is post-processing and storage


class Subscription:
    """A stream of job events: ``{"job_id", "state", "progress", "message"}``. Use ``async for`` or ``get()``."""

    def __init__(self, service: ForgeService) -> None:
        self._service = service
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(EVENT_QUEUE_SIZE)

    def push(self, event: dict[str, Any]) -> None:
        if self.queue.full():
            self.queue.get_nowait()  # a slow consumer loses the oldest event, never blocks the service
        self.queue.put_nowait(event)

    async def get(self) -> dict[str, Any]:
        return await self.queue.get()

    def close(self) -> None:
        self._service._subscribers.discard(self)

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[dict[str, Any]]:
        try:
            while True:
                yield await self.queue.get()
        finally:
            self.close()

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class _Lane:
    name: str
    queued: dict[str, tuple[int, int]] = field(default_factory=dict)  # job_id -> (priority rank, sequence)
    running: str | None = None
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


class ForgeService:
    """Owns the asset store, the worker manager, the job database and the two lane queues."""

    def __init__(
        self,
        config: ForgeConfig,
        *,
        manager: WorkerManager | None = None,
        store: AssetStore | None = None,
        jobdb: JobDB | None = None,
        purge_interval_s: float | None = 86400.0,
    ) -> None:
        self.config = config
        self.store = store or AssetStore(config.data_dir)
        self.jobdb = jobdb or JobDB(config.data_dir)
        self.manager = manager or WorkerManager(config)
        self.purge_interval_s = purge_interval_s
        self._lanes = {"gpu": _Lane("gpu"), "cpu": _Lane("cpu")}
        self._active: dict[str, JobRecord] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancel_requested: set[str] = set()
        self._subscribers: set[Subscription] = set()
        self._analysis_recent: deque[float] = deque(maxlen=5)
        self._seq = 0
        self._started_at = time.monotonic()
        self._background: list[asyncio.Task[None]] = []
        self._running = False
        self._gpu_cache: tuple[float, dict[str, Any]] = (0.0, {"available": False})

    # ---- lifecycle ----

    async def start(self) -> None:
        failed = self.jobdb.fail_unfinished("forge restarted", utc_now())
        if failed:
            log.warning("marked %d unfinished jobs failed after restart", failed)
        shutil.rmtree(self.config.data_dir / "scratch", ignore_errors=True)
        await self.manager.start()
        self._running = True
        for lane in self._lanes.values():
            lane.task = asyncio.create_task(self._lane_loop(lane), name=f"forge-lane-{lane.name}")
        if self.purge_interval_s:
            self._background.append(asyncio.create_task(self._purge_loop()))

    async def stop(self) -> None:
        self._running = False
        for task in self._background:
            task.cancel()
        for lane in self._lanes.values():
            if lane.task is not None:
                lane.task.cancel()
        tasks = [t for t in [*self._background, *(lane.task for lane in self._lanes.values()), *self._tasks.values()] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for rec in list(self._active.values()):  # whatever never ran
            self._finish(rec, "failed", error="forge stopped")
        await self.manager.close()
        self.store.close()
        self.jobdb.close()

    async def __aenter__(self) -> ForgeService:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ---- events ----

    def subscribe(self) -> Subscription:
        sub = Subscription(self)
        self._subscribers.add(sub)
        return sub

    def _emit(self, rec: JobRecord) -> None:
        event = {"job_id": rec.job_id, "state": rec.state, "progress": rec.progress, "message": rec.message}
        for sub in list(self._subscribers):
            sub.push(event)

    # ---- submission ----

    async def submit(
        self,
        type: str,
        params: dict[str, Any] | None = None,
        *,
        model: str | None = None,
        inputs: dict[str, str] | None = None,
        priority: str = "normal",
        idempotency_key: str | None = None,
    ) -> JobRecord:
        """Validate and queue a job. Raises :class:`ValidationFailed` (422) listing field paths."""
        errors: list[dict[str, str]] = []
        if priority not in PRIORITIES:
            errors.append({"path": "priority", "message": f"must be one of {', '.join(PRIORITIES)}"})
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 200):
            errors.append({"path": "idempotency_key", "message": "must be a string of 1..200 characters"})
        if model is not None and not isinstance(model, str):
            errors.append({"path": "model", "message": "must be a model id"})
        try:
            job = validate_job(type, params, inputs)
        except ValidationFailed as exc:
            raise ValidationFailed(errors + exc.errors) from exc
        if errors:
            raise ValidationFailed(errors)
        if idempotency_key is not None:
            existing = self._find_idempotent(idempotency_key)
            if existing is not None:
                return self._view(existing)
        info = self._resolve_model(job, model)
        self._check_inputs(job)
        stored = job.params.model_dump(mode="json")
        if stored.get("seed") is None:
            stored["seed"] = secrets.randbelow(2**31)
        rec = JobRecord(
            job_id="j_" + secrets.token_hex(8), type=job.type, model=info["id"], family=info["family"],
            lane=info["lane"], priority=priority, params=stored, inputs=dict(job.inputs),  # type: ignore[arg-type]
            idempotency_key=idempotency_key, created_at=utc_now(),
        )
        for asset_id in job.inputs.values():
            self.store.mark_used(asset_id)
        self.jobdb.insert(rec)
        self._active[rec.job_id] = rec
        lane = self._lanes[rec.lane or "gpu"]
        self._seq += 1
        lane.queued[rec.job_id] = (PRIORITY_RANK[priority], self._seq)
        lane.wake.set()
        self._emit(rec)
        return self._view(rec)

    def _find_idempotent(self, key: str) -> JobRecord | None:
        since = (datetime.now(timezone.utc) - IDEMPOTENCY_WINDOW).strftime("%Y-%m-%dT%H:%M:%SZ")
        for rec in self._active.values():
            if rec.idempotency_key == key and rec.created_at >= since:
                return rec
        return self.jobdb.find_by_key(key, since)

    def _resolve_model(self, job: ValidatedJob, model: str | None) -> dict[str, Any]:
        try:
            if job.type == "analyze.audio":
                if model not in (None, ANALYSIS_MODEL):
                    raise NotFound(f"analyze.audio runs only on model {ANALYSIS_MODEL!r}")
                if not self.config.family(ANALYSIS_MODEL).enabled:
                    raise NotFound("the analysis family is disabled", code="no_model")
                return self._analysis_info()
            return self.manager.resolve_model(job.type, model)
        except (NotFound, BadRequest) as exc:
            raise ValidationFailed([{"path": "model", "message": exc.message}]) from exc

    def _analysis_info(self) -> dict[str, Any]:
        return {
            "id": ANALYSIS_MODEL, "family": ANALYSIS_MODEL, "job_types": ["analyze.audio"], "available": True,
            "licence": "MIT OR Apache-2.0 (audiokit)", "checkpoint": "builtin:audiokit.analysis",
            "checkpoint_hash": None, "vram_gb": 0.0, "loaded": True, "warm": True, "last_used": None,
            "recent_seconds": self._recent_analysis(), "lane": self.config.family(ANALYSIS_MODEL).lane,
        }

    def _recent_analysis(self) -> float | None:
        return round(sum(self._analysis_recent) / len(self._analysis_recent), 3) if self._analysis_recent else None

    def _check_inputs(self, job: ValidatedJob) -> None:
        errors = []
        for name, asset_id in job.inputs.items():
            if not self.store.exists(asset_id):
                errors.append({"path": f"inputs.{name}", "message": f"unknown asset {asset_id}"})
            elif self.store.kind(asset_id) != INPUT_KINDS[job.type][name]:
                errors.append({"path": f"inputs.{name}", "message": f"asset {asset_id} is not a {INPUT_KINDS[job.type][name]} asset"})
        if errors:
            raise ValidationFailed(errors)

    # ---- reading jobs ----

    def get_job(self, job_id: str) -> JobRecord:
        rec = self._active.get(job_id) or self.jobdb.get(job_id)
        if rec is None:
            raise NotFound("no such job", code="job_not_found")
        return self._view(rec)

    def list_jobs(self, state: str | None = None, limit: int = 50) -> list[JobRecord]:
        if state is not None and state not in ("queued", "running", *FINAL_STATES):
            raise ValidationFailed([{"path": "state", "message": "unknown job state"}])
        out = []
        for rec in self.jobdb.list(state, limit):
            out.append(self._view(self._active.get(rec.job_id, rec)))
        return out

    def _view(self, rec: JobRecord) -> JobRecord:
        """A copy of ``rec`` with the computed ``queue_position`` and ``eta_s``."""
        view = rec.model_copy(deep=True)
        lane = self._lanes.get(rec.lane or "")
        avg = self._average(rec.model)
        if rec.state == "running":
            view.queue_position = 0
            view.eta_s = None if avg is None else round(avg * (1.0 - rec.progress), 2)
        elif rec.state == "queued" and lane is not None and rec.job_id in lane.queued:
            mine = lane.queued[rec.job_id]
            position = 1 + sum(1 for key in lane.queued.values() if key < mine)
            view.queue_position = position
            if avg is not None:
                running = self._active.get(lane.running or "")
                left = avg * (1.0 - running.progress) if running is not None else 0.0
                view.eta_s = round(left + avg * position, 2)
        return view

    def _average(self, model: str | None) -> float | None:
        if model is None:
            return None
        return self._recent_analysis() if model == ANALYSIS_MODEL else self.manager.recent_seconds(model)

    # ---- cancellation ----

    async def cancel(self, job_id: str) -> JobRecord:
        """Cancel a job: queued jobs are removed, running ones are stopped. Finished jobs are returned unchanged."""
        rec = self._active.get(job_id)
        if rec is None:
            return self.get_job(job_id)
        if rec.state == "queued":
            self._lanes[rec.lane or "gpu"].queued.pop(job_id, None)
            self._finish(rec, "cancelled")
        else:
            self._cancel_requested.add(job_id)
            if rec.model == ANALYSIS_MODEL:
                task = self._tasks.get(job_id)
                if task is not None:
                    task.cancel()
            else:
                await self.manager.cancel(job_id)
        return self._view(self._active.get(job_id) or rec)

    # ---- lanes ----

    async def _lane_loop(self, lane: _Lane) -> None:
        while True:
            if not lane.queued:
                lane.wake.clear()
                await lane.wake.wait()
                continue
            job_id = min(lane.queued, key=lambda k: lane.queued[k])
            del lane.queued[job_id]
            rec = self._active.get(job_id)
            if rec is None:
                continue
            lane.running = job_id
            task = asyncio.create_task(self._execute(rec), name=f"forge-job-{job_id}")
            self._tasks[job_id] = task
            try:
                await task
            finally:
                lane.running = None
                self._tasks.pop(job_id, None)

    async def _execute(self, rec: JobRecord) -> None:
        rec.state = "running"
        rec.started_at = utc_now()
        self.jobdb.save(rec)
        self._emit(rec)
        started = time.monotonic()
        try:
            if rec.job_id in self._cancel_requested:
                raise JobCancelled()
            if rec.type == "analyze.audio":
                result = await self._run_analysis(rec)
                self._analysis_recent.append(time.monotonic() - started)
            else:
                result = await self._run_worker_job(rec)
        except (JobCancelled, asyncio.CancelledError) as exc:
            if isinstance(exc, asyncio.CancelledError) and rec.job_id not in self._cancel_requested:
                self._finish(rec, "failed", error="forge stopped")
                raise
            self._finish(rec, "cancelled")
        except WorkerDied as exc:
            self._finish(rec, "failed", error=f"worker died: {exc}")
        except (WorkerError, PostprocessError, ForgeError) as exc:
            self._finish(rec, "failed", error=str(exc))
        except Exception as exc:  # a bug must fail this job, never the lane
            log.exception("job %s crashed", rec.job_id)
            self._finish(rec, "failed", error=f"internal error: {type(exc).__name__}")
        else:
            if rec.job_id in self._cancel_requested:
                self._finish(rec, "cancelled")
            else:
                rec.progress = 1.0
                self._finish(rec, "succeeded", result=result)

    def _finish(self, rec: JobRecord, state: str, *, error: str | None = None, result: JobResult | None = None) -> None:
        rec.state = state  # type: ignore[assignment]
        rec.error = error
        rec.result = result
        rec.finished_at = utc_now()
        rec.message = error or rec.message
        self.jobdb.save(rec)
        self._active.pop(rec.job_id, None)
        self._cancel_requested.discard(rec.job_id)
        self._emit(rec)

    def _progress(self, rec: JobRecord, fraction: float, message: str) -> None:
        rec.progress = round(min(1.0, max(0.0, fraction)) * WORKER_PROGRESS_SHARE, 4)
        rec.message = message
        self._emit(rec)

    # ---- running ----

    async def _run_analysis(self, rec: JobRecord) -> JobResult:
        path = self.store.path(rec.inputs["audio"])

        def work() -> dict[str, Any]:
            audio, sr = wav.read(path)
            if audio.shape[1] == 0:
                raise PostprocessError("the audio asset is empty")
            return json_safe(analysis.summary(audio, sr))  # type: ignore[no-any-return]

        self._progress(rec, 0.1, "analysing")
        return JobResult(analysis=await asyncio.to_thread(work))

    async def _run_worker_job(self, rec: JobRecord) -> JobResult:
        scratch = self.config.data_dir / "scratch" / rec.job_id
        out_dir, post_dir = scratch / "out", scratch / "post"
        out_dir.mkdir(parents=True)
        try:
            run_job = {
                "job_id": rec.job_id, "type": rec.type, "model": rec.model, "params": rec.params,
                "inputs": {name: str(self.store.path(asset_id)) for name, asset_id in rec.inputs.items()},
                "out_dir": str(out_dir), "seed": rec.params["seed"],
            }
            outputs = await self.manager.run(run_job, lambda f, m: self._progress(rec, f, m))
            if rec.job_id in self._cancel_requested:
                raise JobCancelled()
            info = self.manager.model(rec.model or "") or {}
            ctx = PostContext(
                job_id=rec.job_id, job_type=rec.type,
                model=ModelMeta(
                    id=rec.model or "", family=rec.family or "", checkpoint=str(info.get("checkpoint", "")),
                    checkpoint_hash=info.get("checkpoint_hash"), licence=str(info.get("licence", "")),
                ),
                params=validate_job(rec.type, rec.params, rec.inputs).params,
                seed=int(rec.params["seed"]), parents=tuple(rec.inputs.values()),
            )
            return await asyncio.to_thread(self._postprocess_and_store, outputs, ctx, post_dir)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _postprocess_and_store(self, outputs: list[dict[str, Any]], ctx: PostContext, post_dir: Path) -> JobResult:
        processed = [process_output(o, ctx, post_dir) for o in outputs]  # all or nothing before anything is stored
        ids: list[str] = []
        warnings: list[str] = []
        for item in processed:
            side = self.store.add_asset(item.path, item.sidecar)
            if side["asset_id"] not in ids:
                ids.append(side["asset_id"])
            warnings += item.warnings
        return JobResult(asset_ids=ids, warnings=warnings)

    # ---- assets ----

    async def upload(self, path: str | Path, original_name: str | None = None) -> dict[str, str]:
        """Register an uploaded file (already on disk) as an asset: ``{"asset_id", "kind"}``."""
        side = await asyncio.to_thread(
            self.store.register_upload, Path(path), original_name=original_name,
            max_bytes=self.config.uploads.max_bytes, max_seconds=self.config.uploads.max_seconds,
        )
        return {"asset_id": side["asset_id"], "kind": side["kind"]}

    def asset_path(self, asset_id: str) -> Path:
        return self.store.path(asset_id)

    def asset_meta(self, asset_id: str) -> dict[str, Any]:
        return self.store.meta(asset_id)

    def search_assets(self, **filters: Any) -> list[dict[str, Any]]:
        return self.store.search(**filters)

    def rate_asset(self, asset_id: str, rating: int, note: str | None = None) -> dict[str, Any]:
        return self.store.rate(asset_id, rating, note)

    async def purge(self) -> list[str]:
        """Delete unrated, unused assets older than ``retention_days``."""
        removed = await asyncio.to_thread(self.store.purge, self.config.retention_days)
        if removed:
            log.info("purged %d assets", len(removed))
        return removed

    async def _purge_loop(self) -> None:
        assert self.purge_interval_s
        while True:
            await asyncio.sleep(self.purge_interval_s)
            try:
                await self.purge()
            except Exception:  # keep the daily task alive
                log.exception("purge failed")

    # ---- models and health ----

    def models(self) -> list[dict[str, Any]]:
        return [*self.manager.models(), self._analysis_info()] if self.config.family(ANALYSIS_MODEL).enabled else self.manager.models()

    async def load_model(self, model_id: str) -> None:
        """Load a model; raises :class:`Conflict` (409) if it cannot fit the memory budget."""
        if model_id == ANALYSIS_MODEL:
            return
        await self.manager.load(model_id)

    async def unload_model(self, model_id: str) -> None:
        if model_id == ANALYSIS_MODEL:
            return
        await self.manager.unload(model_id)

    async def health(self) -> dict[str, Any]:
        try:
            version = metadata.version("agent-studio")
        except metadata.PackageNotFoundError:
            version = "0.0.0"
        return {
            "status": "ok", "version": version, "uptime_s": round(time.monotonic() - self._started_at, 1),
            "lanes": {
                name: {"running": 1 if lane.running else 0, "queued": len(lane.queued)}
                for name, lane in self._lanes.items()
            },
            "memory": {"budget_gb": self.manager.budget_gb, "used_gb": self.manager.used_gb},
            "gpu": await self._gpu(),
        }

    async def _gpu(self) -> dict[str, Any]:
        now = time.monotonic()
        if now - self._gpu_cache[0] < 5.0:
            return self._gpu_cache[1]
        info: dict[str, Any] = {"available": False}
        if shutil.which("nvidia-smi"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                )
                out, _ = await asyncio.wait_for(proc.communicate(), 3.0)
                name, used, total = [p.strip() for p in out.decode().splitlines()[0].split(",")]
                info = {
                    "available": True, "name": name,
                    "memory_used_gb": round(float(used) / 1024, 2), "memory_total_gb": round(float(total) / 1024, 2),
                }
            except (OSError, ValueError, IndexError, asyncio.TimeoutError):
                info = {"available": False}
        self._gpu_cache = (now, info)
        return info
