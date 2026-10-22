from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from forge.config import ForgeConfig
from forge.errors import BadRequest, Conflict, NotFound
from forge.manager import JobCancelled, WorkerDied, WorkerError, WorkerManager

from fcore_helpers import make_config

EXTRA = [{"id": "tt-a", "vram_gb": 6}, {"id": "tt-b", "vram_gb": 6}, {"id": "tt-c", "vram_gb": 3}, {"id": "tt-huge", "vram_gb": 40}]


def config(tmp_path: Path, **kw: Any) -> ForgeConfig:
    return make_config(
        tmp_path, memory_budget_gb=10,
        families={
            "testtone": {"python": sys.executable, "lane": "gpu", "options": {"extra_models": EXTRA}},
            "yin": {"python": sys.executable},
        }, **kw,
    )


@pytest.fixture
async def manager(tmp_path: Path) -> AsyncIterator[WorkerManager]:
    m = WorkerManager(config(tmp_path), backoff_base_s=0.3, backoff_max_s=1.0, cancel_grace_s=2.0)
    await m.start()
    yield m
    await m.close()


def job(tmp_path: Path, job_id: str = "j1", model: str = "testtone", job_type: str = "generate.sfx", **params: Any) -> dict[str, Any]:
    out = tmp_path / job_id
    out.mkdir(exist_ok=True)
    base = {"prompt": "", "tags": {}, "candidates": 1, "duration_s": 0.2}
    base.update(params)
    return {"job_id": job_id, "type": job_type, "model": model, "params": base, "inputs": {}, "out_dir": str(out), "seed": 3}


def loaded(m: WorkerManager) -> set[str]:
    return {x["id"] for x in m.models() if x["loaded"]}


async def test_start_reads_models_from_hello(manager: WorkerManager) -> None:
    ids = {m["id"] for m in manager.models()}
    assert ids == {"testtone", "tt-a", "tt-b", "tt-c", "tt-huge", "yin"}
    assert manager.lane_of("testtone") == "gpu" and manager.lane_of("yin") == "cpu"
    assert manager.family_status()["testtone"]["state"] == "ready"
    assert manager.resolve_model("generate.loop")["id"] == "testtone"
    assert manager.resolve_model("transcribe.melody")["id"] == "yin"
    assert manager.resolve_model("generate.loop", "tt-a")["id"] == "tt-a"
    with pytest.raises(NotFound):
        manager.resolve_model("generate.loop", "nope")
    with pytest.raises(BadRequest):
        manager.resolve_model("generate.loop", "yin")
    with pytest.raises(NotFound, match="no available model"):
        manager.resolve_model("stems.split")


async def test_run_loads_model_reports_progress_and_validates_outputs(manager: WorkerManager, tmp_path: Path) -> None:
    seen: list[tuple[float, str]] = []
    outputs = await manager.run(job(tmp_path), lambda f, m: seen.append((f, m)))
    assert Path(outputs[0]["path"]).is_file() and outputs[0]["kind"] == "audio"
    assert seen and seen[-1][0] == 1.0
    assert loaded(manager) == {"testtone"}
    info = manager.model("testtone")
    assert info is not None and info["recent_seconds"] is not None and info["last_used"] is not None and info["warm"]


async def test_lru_eviction_within_budget(manager: WorkerManager) -> None:
    await manager.load("tt-a")
    assert manager.used_gb == 6 and loaded(manager) == {"tt-a"}
    await manager.load("tt-b")  # 6 + 6 > 10: tt-a is the least recently used
    assert loaded(manager) == {"tt-b"}
    await manager.load("tt-c")
    assert loaded(manager) == {"tt-b", "tt-c"} and manager.used_gb == 9
    await manager.load("tt-b")  # already loaded: becomes most recently used
    await manager.load("tt-a")  # needs 6 of 10 with 9 in use: evicts tt-c (LRU) first, then tt-b
    assert loaded(manager) == {"tt-a"}  # tt-c was the least recently used, then tt-b
    assert manager.used_gb <= manager.budget_gb


async def test_lru_order_is_by_use_not_by_load(manager: WorkerManager) -> None:
    await manager.load("tt-c")  # 3
    await manager.load("tt-a")  # 9
    await manager.load("tt-c")  # touch: tt-a is now the LRU
    await manager.load("tt-b")  # needs 6, used 9: evict tt-a, leaving tt-c + tt-b = 9
    assert loaded(manager) == {"tt-c", "tt-b"}


async def test_model_larger_than_budget_is_a_conflict(manager: WorkerManager) -> None:
    with pytest.raises(Conflict) as exc:
        await manager.load("tt-huge")
    assert exc.value.status == 409 and exc.value.code == "over_budget"
    assert loaded(manager) == set()


async def test_unload(manager: WorkerManager) -> None:
    await manager.load("tt-a")
    await manager.unload("tt-a")
    assert loaded(manager) == set() and manager.used_gb == 0
    await manager.unload("tt-a")  # idempotent
    with pytest.raises(NotFound):
        await manager.unload("nope")


async def test_running_model_cannot_be_evicted_or_unloaded(manager: WorkerManager, tmp_path: Path) -> None:
    slow = asyncio.create_task(manager.run(job(tmp_path, model="tt-a", tags={"x-delay_s": "1.0"})))
    for _ in range(100):
        if (m := manager.model("tt-a")) and m["in_use"]:
            break
        await asyncio.sleep(0.02)
    with pytest.raises(Conflict, match="running"):
        await manager.unload("tt-a")
    await slow
    await manager.unload("tt-a")


async def test_worker_error_is_reported_without_restart(manager: WorkerManager, tmp_path: Path) -> None:
    with pytest.raises(WorkerError, match="injected failure"):
        await manager.run(job(tmp_path, tags={"x-fault": "error"}))
    assert manager.family_status()["testtone"] == {"state": "ready", "failures": 0, "error": ""}
    await manager.run(job(tmp_path, "j2"))


async def test_worker_death_fails_the_job_and_restarts_with_backoff(manager: WorkerManager, tmp_path: Path) -> None:
    await manager.run(job(tmp_path))
    assert loaded(manager) == {"testtone"}
    with pytest.raises(WorkerDied):
        await manager.run(job(tmp_path, "j2", tags={"x-fault": "crash"}))
    status = manager.family_status()["testtone"]
    assert status["state"] == "down" and status["failures"] == 1
    assert loaded(manager) == set()  # the dead worker's models are forgotten
    started = time.monotonic()
    outputs = await manager.run(job(tmp_path, "j3"))  # restarts on next use, after the 0.3 s backoff
    assert time.monotonic() - started >= 0.25 and Path(outputs[0]["path"]).is_file()
    assert manager.family_status()["testtone"]["failures"] == 0 and loaded(manager) == {"testtone"}


async def test_backoff_grows_with_repeated_deaths(manager: WorkerManager, tmp_path: Path) -> None:
    for i in range(2):
        with pytest.raises(WorkerDied):
            await manager.run(job(tmp_path, f"c{i}", tags={"x-fault": "crash"}))
    assert manager.family_status()["testtone"]["failures"] == 2
    started = time.monotonic()
    await manager.run(job(tmp_path, "ok"))
    assert time.monotonic() - started >= 0.55  # 0.3 * 2


async def test_worker_killed_while_idle_is_restarted(manager: WorkerManager, tmp_path: Path) -> None:
    await manager.run(job(tmp_path))
    proc = manager._workers["testtone"].proc
    assert proc is not None
    os.kill(proc.pid, signal.SIGKILL)
    await proc.wait()
    await asyncio.sleep(0.1)
    assert loaded(manager) == set()
    outputs = await manager.run(job(tmp_path, "again"))
    assert Path(outputs[0]["path"]).is_file()


async def test_cancel_running_job(manager: WorkerManager, tmp_path: Path) -> None:
    task = asyncio.create_task(manager.run(job(tmp_path, tags={"x-delay_s": "30"})))
    await asyncio.sleep(0.5)
    assert await manager.cancel("j1") is True
    with pytest.raises(JobCancelled):
        await asyncio.wait_for(task, 5)
    assert manager.family_status()["testtone"]["state"] == "ready"  # the worker stopped the job itself
    assert await manager.cancel("j1") is False  # nothing running any more
    await manager.run(job(tmp_path, "next"))


async def test_cancel_kills_a_worker_that_ignores_it(tmp_path: Path) -> None:
    m = WorkerManager(config(tmp_path), backoff_base_s=0.05, cancel_grace_s=0.3)
    await m.start()
    try:
        worker = m._workers["testtone"]
        # make the worker deaf to cancel by pausing the process
        task = asyncio.create_task(m.run(job(tmp_path, tags={"x-delay_s": "30"})))
        await asyncio.sleep(0.5)
        assert worker.proc is not None
        os.kill(worker.proc.pid, signal.SIGSTOP)
        cancel = asyncio.create_task(m.cancel("j1"))
        with pytest.raises(JobCancelled):
            await asyncio.wait_for(task, 8)
        cancel.cancel()
        assert m.family_status()["testtone"]["failures"] == 0  # a deliberate kill is not a crash
        await m.run(job(tmp_path, "next"))
    finally:
        await m.close()


async def test_missing_interpreter_marks_family_down_not_fatal(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, families={"testtone": {"python": "/nonexistent/python"}, "yin": {"python": sys.executable}})
    m = WorkerManager(cfg, backoff_base_s=0.05)
    await m.start()
    try:
        assert m.family_status()["testtone"]["state"] == "down" and "cannot start" in m.family_status()["testtone"]["error"]
        assert {x["id"] for x in m.models()} == {"yin"}
        with pytest.raises(NotFound):
            await m.run(job(tmp_path))
    finally:
        await m.close()


async def test_worker_environment_hides_the_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_TOKEN", "super-secret")
    cfg = make_config(tmp_path, families={"testtone": {"python": sys.executable, "env": {"CUSTOM": "1"}}, "yin": {"enabled": False}})
    m = WorkerManager(cfg)
    await m.start()
    try:
        proc = m._workers["testtone"].proc
        assert proc is not None
        environ = Path(f"/proc/{proc.pid}/environ")
        env = environ.read_bytes().split(b"\0") if environ.exists() else None
        if env is not None:
            assert b"FORGE_TOKEN=super-secret" not in env and b"CUSTOM=1" in env
    finally:
        await m.close()


async def test_close_stops_workers(tmp_path: Path) -> None:
    m = WorkerManager(config(tmp_path))
    await m.start()
    procs = [w.proc for w in m._workers.values()]
    await m.close()
    assert all(p is not None and p.returncode is not None for p in procs)
