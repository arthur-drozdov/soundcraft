"""``GET /v1/events``: server-sent events, over a real socket (the ASGI test transport buffers streams)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from forge import api

from fapi_helpers import AUTH, SLOW, assert_envelope, submit, wait_job


class Sse:
    """Reads one event stream frame by frame; ``until`` returns every job event seen up to a job's final state."""

    def __init__(self, resp: httpx.Response) -> None:
        self._frames = self._parse(resp)

    @staticmethod
    async def _parse(resp: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        frame: dict[str, Any] = {}
        async for line in resp.aiter_lines():
            if line == "":
                if frame:
                    yield frame
                frame = {}
            elif not line.startswith(":"):  # comments (keepalives) carry nothing
                name, _, value = line.partition(":")
                frame[name] = value.lstrip(" ")

    async def until(self, job_id: str, timeout: float = 30) -> list[dict[str, Any]]:
        seen: list[dict[str, Any]] = []

        async def read() -> None:
            async for frame in self._frames:
                if "event" not in frame:  # the opening `retry:` frame
                    assert frame == {"retry": "3000"}, frame
                    continue
                assert frame["event"] == "job", frame
                data = {"id": int(frame["id"]), **json.loads(frame["data"])}
                seen.append(data)
                if data["job_id"] == job_id and data["state"] in ("succeeded", "failed", "cancelled"):
                    return

        await asyncio.wait_for(read(), timeout)
        return seen


async def test_events_require_the_token(live: str) -> None:
    async with httpx.AsyncClient(base_url=live) as c:
        assert_envelope(await c.get("/v1/events"), 401, "unauthorized")


async def test_stream_delivers_the_events_of_a_job(live: str) -> None:
    async with httpx.AsyncClient(base_url=live, headers=AUTH, timeout=60) as c:
        async with c.stream("GET", "/v1/events") as resp:
            assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
            assert resp.headers["cache-control"] == "no-cache"
            job_id = await submit(c, "generate.loop", {"bpm": 120, "bars": 1, "candidates": 2})
            seen = [e for e in await Sse(resp).until(job_id) if e["job_id"] == job_id]
    states = [e["state"] for e in seen]
    assert states[0] == "queued" and states[-1] == "succeeded" and "running" in states
    assert [e["id"] for e in seen] == sorted(e["id"] for e in seen) and len({e["id"] for e in seen}) == len(seen)
    assert all(set(e) == {"id", "job_id", "state", "progress", "message"} for e in seen)
    assert seen[-1]["progress"] == 1.0 and min(e["progress"] for e in seen) >= 0.0


async def test_stream_can_follow_one_job(live: str) -> None:
    async with httpx.AsyncClient(base_url=live, headers=AUTH, timeout=60) as c:
        first = await submit(c, "generate.sfx", {"duration_s": 0.3, "tags": {"x-delay_s": "1"}})
        async with c.stream("GET", f"/v1/events?job_id={first}") as resp:
            second = await submit(c, "generate.sfx", {"duration_s": 0.3})  # queued behind `first`: its events must not appear
            seen = await Sse(resp).until(first)
        await wait_job(c, second)
    assert seen and {e["job_id"] for e in seen} == {first} and seen[-1]["state"] == "succeeded"


async def test_stream_reports_failures_and_cancellation(live: str) -> None:
    async with httpx.AsyncClient(base_url=live, headers=AUTH, timeout=60) as c:
        async with c.stream("GET", "/v1/events") as resp:
            sse = Sse(resp)
            failing = await submit(c, "generate.sfx", {"tags": {"x-fault": "error"}})
            failed = (await sse.until(failing))[-1]
            assert failed["job_id"] == failing and failed["state"] == "failed" and "injected failure" in failed["message"]
            slow = await submit(c, "generate.sfx", {"tags": SLOW})
            await wait_job(c, slow, "running")
            await c.delete(f"/v1/jobs/{slow}")
            cancelled = (await sse.until(slow))[-1]
            assert cancelled["job_id"] == slow and cancelled["state"] == "cancelled"


async def test_open_streams_are_capped_and_released(live: str, app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "MAX_EVENT_STREAMS", 1)
    async with httpx.AsyncClient(base_url=live, headers=AUTH, timeout=60) as c:
        async with c.stream("GET", "/v1/events") as first:
            assert first.status_code == 200 and len(app.state.event_streams) == 1
            refused = await c.get("/v1/events")
            assert_envelope(refused, 503, "too_many_streams")
        for _ in range(200):  # the server notices the disconnect
            if not app.state.event_streams:
                break
            await asyncio.sleep(0.05)
        assert not app.state.event_streams
        async with c.stream("GET", "/v1/events") as again:
            assert again.status_code == 200


async def test_ending_the_streams_lets_a_server_shut_down(live: str, app: FastAPI) -> None:
    async with httpx.AsyncClient(base_url=live, headers=AUTH, timeout=60) as c:
        async with c.stream("GET", "/v1/events") as resp:
            assert len(app.state.event_streams) == 1
            api.end_event_streams(app)
            body = await asyncio.wait_for(resp.aread(), 10)  # the response completes instead of hanging
        assert body.startswith(b"retry: 3000") and b"event:" not in body
        assert not app.state.event_streams
