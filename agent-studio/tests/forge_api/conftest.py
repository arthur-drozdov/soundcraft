"""Fixtures. ``client``/``anon``/``live`` give every test its own service; the ``s``-prefixed ones
(``sclient``, ``sanon``) share one service per module, for tests that only add independent jobs/assets.
A module using the shared ones is marked ``pytestmark = SHARED_LOOP`` (fapi_helpers)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from forge.api import create_app
from forge.cli import make_server
from forge.config import ForgeConfig
from forge.manager import WorkerManager
from forge.service import ForgeService

from fapi_helpers import AUTH, TOKEN, TOKEN_ENV, make_config


def new_service(config: ForgeConfig) -> ForgeService:
    manager = WorkerManager(config, backoff_base_s=0.05, backoff_max_s=0.2, cancel_grace_s=3.0)
    return ForgeService(config, manager=manager, purge_interval_s=None)


def new_client(app: FastAPI, headers: dict[str, str]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://forge.test", headers=headers, timeout=60)


# ---- one service per test ----


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ForgeConfig:
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    return make_config(tmp_path)


@pytest.fixture
async def service(config: ForgeConfig) -> AsyncIterator[ForgeService]:
    """A service with the real testtone and yin worker subprocesses."""
    svc = new_service(config)
    await svc.start()
    try:
        yield svc
    finally:
        await svc.stop()


@pytest.fixture
def app(service: ForgeService, config: ForgeConfig) -> FastAPI:
    return create_app(service, config)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Authenticated client talking to the app in-process."""
    async with new_client(app, AUTH) as c:
        yield c


@pytest.fixture
async def anon(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """The same app, no credentials."""
    async with new_client(app, {}) as c:
        yield c


@pytest.fixture
async def live(app: FastAPI) -> AsyncIterator[str]:
    """The app on a real socket (httpx's ASGI transport cannot stream server-sent events); yields its base URL."""
    server = make_server(app, "127.0.0.1", 0, log_level="warning")
    task = asyncio.create_task(server.serve())
    for _ in range(500):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started, "uvicorn did not start"
    port: Any = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 15)


# ---- one service per module ----


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sservice(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[ForgeService]:
    svc = new_service(make_config(tmp_path_factory.mktemp("shared")))
    await svc.start()
    try:
        yield svc
    finally:
        await svc.stop()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sapp(sservice: ForgeService) -> FastAPI:
    return create_app(sservice, sservice.config, token=TOKEN)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sclient(sapp: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with new_client(sapp, AUTH) as c:
        yield c


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sanon(sapp: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with new_client(sapp, {}) as c:
        yield c
