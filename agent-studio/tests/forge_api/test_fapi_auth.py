"""Bearer authentication, the public health route, the error envelope and routing errors."""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Any

import httpx
import pytest

from forge import api
from forge.api import create_app
from forge.errors import ConfigError
from forge.service import ForgeService

from fapi_helpers import AUTH, SHARED_LOOP, TOKEN, TOKEN_ENV, assert_envelope, make_config

pytestmark = SHARED_LOOP

PROTECTED = [
    ("GET", "/v1/models"),
    ("POST", "/v1/models/testtone/load"),
    ("POST", "/v1/models/testtone/unload"),
    ("POST", "/v1/uploads"),
    ("POST", "/v1/jobs"),
    ("GET", "/v1/jobs"),
    ("GET", "/v1/jobs/j_0123456789abcdef"),
    ("DELETE", "/v1/jobs/j_0123456789abcdef"),
    ("GET", "/v1/events"),
    ("GET", "/v1/assets"),
    ("GET", "/v1/assets/a_000000000000000000000000"),
    ("GET", "/v1/assets/a_000000000000000000000000/meta"),
    ("POST", "/v1/assets/a_000000000000000000000000/rate"),
]


async def test_health_needs_no_token_and_has_the_documented_shape(sanon: httpx.AsyncClient) -> None:
    resp = await sanon.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"status", "version", "uptime_s", "lanes", "memory", "gpu"}
    assert body["status"] == "ok" and body["uptime_s"] >= 0
    assert set(body["lanes"]) == {"gpu", "cpu"} and set(body["lanes"]["gpu"]) == {"running", "queued"}
    assert set(body["memory"]) == {"budget_gb", "used_gb"} and body["memory"]["budget_gb"] == 96
    assert isinstance(body["gpu"]["available"], bool)
    assert AUTH["Authorization"].split()[1] not in resp.text


async def test_health_also_works_with_a_wrong_token(sanon: httpx.AsyncClient) -> None:
    assert (await sanon.get("/v1/health", headers={"Authorization": "Bearer nope"})).status_code == 200


@pytest.mark.parametrize(("method", "path"), PROTECTED)
async def test_every_other_route_requires_the_token(sanon: httpx.AsyncClient, method: str, path: str) -> None:
    resp = await sanon.request(method, path)
    error = assert_envelope(resp, 401, "unauthorized")
    assert resp.headers["www-authenticate"] == "Bearer" and TOKEN not in error["message"]


@pytest.mark.parametrize(
    "header",
    [
        "Bearer wrong",
        "Bearer ",
        "Bearer",
        f"Bearer {TOKEN}x",
        f"Bearer {TOKEN[:-1]}",
        f"Bearer {TOKEN} extra",
        f"Basic {TOKEN}",
        f"Token {TOKEN}",
        TOKEN,
        f"Bearer {TOKEN.upper()}",
        "Bearer éè",
    ],
)
async def test_wrong_credentials_are_rejected(sanon: httpx.AsyncClient, header: str) -> None:
    assert_envelope(await sanon.get("/v1/models", headers={"Authorization": header.encode("latin-1")}), 401, "unauthorized")


async def test_the_right_token_is_accepted_in_any_scheme_case(sanon: httpx.AsyncClient) -> None:
    assert (await sanon.get("/v1/models", headers={"Authorization": f"Bearer {TOKEN}"})).status_code == 200
    assert (await sanon.get("/v1/models", headers={"Authorization": f"bearer {TOKEN}"})).status_code == 200


async def test_token_comparison_is_constant_time(sanon: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[bytes, bytes]] = []
    real = hmac.compare_digest

    def spy(a: Any, b: Any) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(api.hmac, "compare_digest", spy)
    await sanon.get("/v1/models", headers={"Authorization": "Bearer x"})
    await sanon.get("/v1/models", headers={"Authorization": f"Bearer {TOKEN}"})
    assert len(calls) == 2 and all(len(a) == len(b) == 32 for a, b in calls)  # digests, so length never leaks


async def test_auth_runs_before_validation_so_nothing_leaks_to_strangers(sanon: httpx.AsyncClient) -> None:
    assert_envelope(await sanon.get("/v1/jobs?limit=-4"), 401, "unauthorized")
    assert_envelope(await sanon.post("/v1/jobs", content=b"{not json"), 401, "unauthorized")


async def test_unknown_routes_and_methods_use_the_envelope(sclient: httpx.AsyncClient, sanon: httpx.AsyncClient) -> None:
    assert_envelope(await sclient.get("/v1/nope"), 404, "not_found")
    assert_envelope(await sclient.get("/nope"), 404, "not_found")
    assert_envelope(await sclient.put("/v1/jobs"), 405, "method_not_allowed")
    assert (await sclient.get("/docs")).status_code == 404 and (await sclient.get("/openapi.json")).status_code == 404


async def test_insecure_no_auth_skips_the_check(sservice: ForgeService, tmp_path: Path) -> None:
    cfg = make_config(tmp_path, insecure_no_auth=True)
    app = create_app(sservice, cfg)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://forge.test") as open_client:
        assert (await open_client.get("/v1/models")).status_code == 200


async def test_startup_fails_without_a_token(sservice: ForgeService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_ENV, "")
    with pytest.raises(ConfigError, match=TOKEN_ENV):
        create_app(sservice, make_config(tmp_path))
    monkeypatch.delenv(TOKEN_ENV)
    with pytest.raises(ConfigError):
        create_app(sservice, make_config(tmp_path))
    with pytest.raises(ConfigError):
        create_app(sservice, make_config(tmp_path), token="")


async def test_explicit_token_overrides_the_environment(sservice: ForgeService) -> None:
    app = create_app(sservice, sservice.config, token="another-token")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://forge.test") as c:
        assert (await c.get("/v1/models", headers=AUTH)).status_code == 401
        assert (await c.get("/v1/models", headers={"Authorization": "Bearer another-token"})).status_code == 200
