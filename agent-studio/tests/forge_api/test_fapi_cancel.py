"""Cancelling jobs over HTTP, and the queue fields a client sees while it waits."""

from __future__ import annotations

import httpx

from fapi_helpers import SLOW, assert_envelope, run_job, submit, wait_job


async def test_cancelling_a_running_job(client: httpx.AsyncClient) -> None:
    job_id = await submit(client, "generate.sfx", {"tags": SLOW})
    running = await wait_job(client, job_id, "running")
    assert running["queue_position"] == 0 and running["started_at"]
    resp = await client.delete(f"/v1/jobs/{job_id}")
    assert resp.status_code == 200 and resp.json()["job_id"] == job_id
    done = await wait_job(client, job_id)
    assert done["state"] == "cancelled" and done["result"] is None
    follow_up = await run_job(client, "generate.sfx", {"duration_s": 0.3})  # the lane is free again
    assert follow_up["state"] == "succeeded"


async def test_cancelling_a_queued_job_removes_it(client: httpx.AsyncClient) -> None:
    blocker = await submit(client, "generate.sfx", {"tags": SLOW})
    await wait_job(client, blocker, "running")
    queued = await submit(client, "generate.sfx", {"duration_s": 0.3})
    waiting = (await client.get(f"/v1/jobs/{queued}")).json()
    assert waiting["state"] == "queued" and waiting["queue_position"] == 1
    resp = await client.delete(f"/v1/jobs/{queued}")
    assert resp.status_code == 200 and resp.json()["state"] == "cancelled"
    assert (await client.get(f"/v1/jobs/{queued}")).json()["state"] == "cancelled"
    assert (await client.get("/v1/health")).json()["lanes"]["gpu"] == {"running": 1, "queued": 0}
    await client.delete(f"/v1/jobs/{blocker}")
    await wait_job(client, blocker)


async def test_high_priority_jobs_wait_ahead_of_normal_ones(client: httpx.AsyncClient) -> None:
    blocker = await submit(client, "generate.sfx", {"tags": SLOW})
    await wait_job(client, blocker, "running")
    normal = await submit(client, "generate.sfx", {"duration_s": 0.3}, priority="normal")
    high = await submit(client, "generate.sfx", {"duration_s": 0.3}, priority="high")
    low = await submit(client, "generate.sfx", {"duration_s": 0.3}, priority="low")
    positions = {name: (await client.get(f"/v1/jobs/{jid}")).json()["queue_position"] for name, jid in (("normal", normal), ("high", high), ("low", low))}
    assert positions == {"high": 1, "normal": 2, "low": 3}
    for job_id in (blocker, normal, high, low):
        await client.delete(f"/v1/jobs/{job_id}")
    for job_id in (blocker, normal, high, low):
        await wait_job(client, job_id)


async def test_cancelling_a_finished_job_changes_nothing(client: httpx.AsyncClient) -> None:
    job = await run_job(client, "generate.sfx", {"duration_s": 0.3})
    resp = await client.delete(f"/v1/jobs/{job['job_id']}")
    assert resp.status_code == 200 and resp.json()["state"] == "succeeded"


async def test_cancelling_an_unknown_job_is_404(client: httpx.AsyncClient) -> None:
    assert_envelope(await client.delete("/v1/jobs/j_ffffffffffffffff"), 404, "job_not_found")
