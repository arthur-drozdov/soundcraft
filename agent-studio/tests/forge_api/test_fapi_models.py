"""``/v1/models`` and the load/unload routes."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from forge.api import create_app
from forge.manager import WorkerManager
from forge.service import ForgeService

from fapi_helpers import AUTH, SLOW, TOKEN, assert_envelope, make_config, submit, wait_job


@asynccontextmanager
async def budget_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """A gateway whose memory budget (10 GB) fits one of two 6 GB test models, and not a 50 GB one."""
    extra = [{"id": "tt-a", "vram_gb": 6}, {"id": "tt-b", "vram_gb": 6}, {"id": "tt-x", "vram_gb": 50}]
    cfg = make_config(
        tmp_path, memory_budget_gb=10,
        families={"testtone": {"python": sys.executable, "lane": "gpu", "options": {"extra_models": extra}}},
    )
    svc = ForgeService(cfg, manager=WorkerManager(cfg), purge_interval_s=None)
    await svc.start()
    try:
        app = create_app(svc, cfg, token=TOKEN)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://forge.test", headers=AUTH) as c:
            yield c
    finally:
        await svc.stop()


async def test_models_lists_every_family_with_runtime_state(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    models = {m["id"]: m for m in resp.json()}
    assert {"testtone", "yin", "analysis"} <= set(models)
    for info in models.values():
        assert {"id", "family", "job_types", "available", "licence", "checkpoint", "vram_gb", "loaded", "warm", "last_used", "recent_seconds"} <= set(info)
    assert "test generator" in models["testtone"]["licence"] and models["testtone"]["available"] is True
    assert "generate.loop" in models["testtone"]["job_types"] and models["yin"]["job_types"] == ["transcribe.melody"]
    assert models["analysis"]["job_types"] == ["analyze.audio"]


async def test_load_and_unload_round_trip(client: httpx.AsyncClient) -> None:
    resp = await client.post("/v1/models/testtone/load")
    assert resp.status_code == 200 and resp.json()["id"] == "testtone" and resp.json()["loaded"] is True
    listed = {m["id"]: m for m in (await client.get("/v1/models")).json()}
    assert listed["testtone"]["loaded"] and listed["testtone"]["warm"] and listed["testtone"]["last_used"]
    assert (await client.post("/v1/models/testtone/load")).status_code == 200  # idempotent
    resp = await client.post("/v1/models/testtone/unload")
    assert resp.status_code == 200 and resp.json()["loaded"] is False


async def test_unknown_and_malformed_model_ids_are_404(client: httpx.AsyncClient) -> None:
    assert_envelope(await client.post("/v1/models/nope/load"), 404, "model_not_found")
    assert_envelope(await client.post("/v1/models/nope/unload"), 404, "model_not_found")
    assert (await client.post("/v1/models/..%2f..%2fetc/load")).status_code == 404
    assert_envelope(await client.post("/v1/models/" + "a" * 200 + "/load"), 404, "model_not_found")


async def test_loading_evicts_idle_models_and_refuses_what_cannot_fit(tmp_path: Path) -> None:
    async with budget_client(tmp_path) as c:
        assert (await c.post("/v1/models/tt-a/load")).json()["loaded"] is True
        assert (await c.post("/v1/models/tt-b/load")).json()["loaded"] is True  # tt-a was idle: evicted
        state = {m["id"]: m["loaded"] for m in (await c.get("/v1/models")).json()}
        assert state["tt-a"] is False and state["tt-b"] is True
        assert_envelope(await c.post("/v1/models/tt-x/load"), 409, "over_budget")
        health = (await c.get("/v1/health")).json()
        assert health["memory"] == {"budget_gb": 10, "used_gb": 6}


async def test_unloading_a_busy_model_is_a_conflict(client: httpx.AsyncClient) -> None:
    job_id = await submit(client, "generate.sfx", {"tags": SLOW})
    await wait_job(client, job_id, "running")
    assert_envelope(await client.post("/v1/models/testtone/unload"), 409, "model_busy")
    await client.delete(f"/v1/jobs/{job_id}")
    await wait_job(client, job_id)
