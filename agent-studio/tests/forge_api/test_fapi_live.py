"""Downloads and uploads through a real socket (uvicorn), where streaming, HEAD and 304 behave for real."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import httpx
import numpy as np

from forge.config import ForgeConfig

from fapi_helpers import AUTH, assert_envelope, tone, wav_bytes

BIG = wav_bytes(np.tile(tone(220, 4.0), (2, 1)))  # ~700 KB: several streaming chunks


async def test_large_upload_and_streamed_download_round_trip(live: str) -> None:
    assert len(BIG) > 600_000
    async with httpx.AsyncClient(base_url=live, headers=AUTH, timeout=60) as c:
        up = await c.post("/v1/uploads", files={"file": ("big.wav", BIG, "audio/wav")})
        assert up.status_code == 200 and up.json()["asset_id"] == "a_" + hashlib.sha256(BIG).hexdigest()[:24]
        asset = up.json()["asset_id"]
        async with c.stream("GET", f"/v1/assets/{asset}") as resp:
            chunks = [chunk async for chunk in resp.aiter_raw()]
        assert b"".join(chunks) == BIG and resp.headers["content-length"] == str(len(BIG))


async def test_range_head_and_conditional_requests(live: str) -> None:
    async with httpx.AsyncClient(base_url=live, headers=AUTH, timeout=60) as c:
        asset = (await c.post("/v1/uploads", files={"file": ("big.wav", BIG, "audio/wav")})).json()["asset_id"]
        url = f"/v1/assets/{asset}"
        part = await c.get(url, headers={"Range": "bytes=1000-1999"})
        assert part.status_code == 206 and part.content == BIG[1000:2000]
        assert part.headers["content-range"] == f"bytes 1000-1999/{len(BIG)}" and part.headers["content-length"] == "1000"
        tail = await c.get(url, headers={"Range": "bytes=-16"})
        assert tail.status_code == 206 and tail.content == BIG[-16:]
        assert_envelope(await c.get(url, headers={"Range": f"bytes={len(BIG)}-"}), 416, "range_not_satisfiable")
        head = await c.head(url)
        assert head.status_code == 200 and head.content == b"" and head.headers["content-length"] == str(len(BIG))
        assert head.headers["etag"] == f'"{asset}"'
        cached = await c.get(url, headers={"If-None-Match": f'"{asset}"'})
        assert cached.status_code == 304 and cached.content == b""
        # a resumed download: first half, then the rest from where it stopped
        first = await c.get(url, headers={"Range": f"bytes=0-{len(BIG) // 2 - 1}"})
        rest = await c.get(url, headers={"Range": f"bytes={len(BIG) // 2}-", "If-Range": first.headers["etag"]})
        assert first.content + rest.content == BIG


async def test_streaming_upload_over_the_cap_is_cut_off(live: str, config: ForgeConfig) -> None:
    config.uploads.max_bytes = 50_000

    async def body() -> AsyncIterator[bytes]:
        yield b'--B\r\nContent-Disposition: form-data; name="file"; filename="x.wav"\r\n\r\n'
        for _ in range(60):
            yield b"\x00" * 65536
        yield b"\r\n--B--\r\n"

    async with httpx.AsyncClient(base_url=live, headers=AUTH, timeout=60) as c:
        resp = await c.post("/v1/uploads", content=body(), headers={"Content-Type": "multipart/form-data; boundary=B"})
        assert_envelope(resp, 413, "upload_too_large")
        assert (await c.get("/v1/models")).status_code == 200  # the gateway is unharmed


async def test_a_client_that_hangs_up_mid_download_does_not_hurt_the_server(live: str) -> None:
    async with httpx.AsyncClient(base_url=live, headers=AUTH, timeout=60) as c:
        asset = (await c.post("/v1/uploads", files={"file": ("big.wav", BIG, "audio/wav")})).json()["asset_id"]
        for _ in range(5):
            async with c.stream("GET", f"/v1/assets/{asset}") as resp:
                async for _chunk in resp.aiter_raw():
                    break  # hang up after the first chunk
        assert (await c.get(f"/v1/assets/{asset}")).content == BIG


async def test_the_app_can_own_the_service_lifecycle(config: ForgeConfig) -> None:
    from forge.api import create_app
    from forge.manager import WorkerManager
    from forge.service import ForgeService

    svc = ForgeService(config, manager=WorkerManager(config), purge_interval_s=None)
    app = create_app(svc, config, manage_lifecycle=True)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://forge.test", headers=AUTH) as c:
            models = {m["id"] for m in (await c.get("/v1/models")).json()}
            assert {"testtone", "yin"} <= models  # the workers were started by the lifespan
    assert svc._running is False  # and stopped with it
