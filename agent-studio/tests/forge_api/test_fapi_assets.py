"""Asset downloads (ETag, Range), metadata, search and ratings."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from fapi_helpers import SHARED_LOOP, assert_envelope, run_job, tone, upload, wav_bytes

pytestmark = SHARED_LOOP


async def uploaded(client: httpx.AsyncClient) -> tuple[str, bytes]:
    data = wav_bytes(tone(300 + secrets.randbelow(5000) / 10, 1.0))
    return (await upload(client, data))["asset_id"], data


async def make_assets(client: httpx.AsyncClient, marker: str) -> dict[str, str]:
    """Three assets from three jobs, all carrying ``marker`` in their prompt."""
    out = {}
    for name, job_type, params in (
        ("synth", "generate.sfx", {"prompt": f"{marker} alpha", "tags": {"family": "synth", "mood": ["dark", "wide"]}}),
        ("bass", "generate.sample", {"prompt": f"{marker} beta", "tags": {"family": "bass"}, "note": 40, "length_s": 0.3}),
        ("loop", "generate.loop", {"prompt": f"{marker} gamma", "bpm": 100, "bars": 1}),
    ):
        job = await run_job(client, job_type, params)
        out[name] = job["result"]["asset_ids"][0]
        out[name + "_job"] = job["job_id"]
    return out


async def search(client: httpx.AsyncClient, **query: Any) -> list[dict[str, Any]]:
    resp = await client.get("/v1/assets", params=query)
    assert resp.status_code == 200, resp.text
    return resp.json()  # type: ignore[no-any-return]


# ---- downloads ----


async def test_download_headers(sclient: httpx.AsyncClient) -> None:
    asset_id, data = await uploaded(sclient)
    resp = await sclient.get(f"/v1/assets/{asset_id}")
    assert resp.status_code == 200 and resp.content == data
    assert resp.headers["etag"] == f'"{asset_id}"' and resp.headers["accept-ranges"] == "bytes"
    assert resp.headers["content-length"] == str(len(data)) and resp.headers["content-type"] == "audio/wav"
    assert resp.headers["x-content-type-options"] == "nosniff" and asset_id in resp.headers["content-disposition"]
    assert asset_id == "a_" + hashlib.sha256(resp.content).hexdigest()[:24]


async def test_head_returns_the_headers_only(sclient: httpx.AsyncClient) -> None:
    asset_id, data = await uploaded(sclient)
    resp = await sclient.head(f"/v1/assets/{asset_id}")
    assert resp.status_code == 200 and resp.content == b"" and resp.headers["content-length"] == str(len(data))
    assert resp.headers["etag"] == f'"{asset_id}"'


async def test_single_ranges(sclient: httpx.AsyncClient) -> None:
    asset_id, data = await uploaded(sclient)
    n = len(data)
    cases = {
        "bytes=0-99": (0, 99),
        "bytes=100-199": (100, 199),
        "bytes=1000-": (1000, n - 1),
        "bytes=-50": (n - 50, n - 1),
        f"bytes={n - 10}-{n + 5000}": (n - 10, n - 1),  # an end past the file is clamped
        f"bytes=-{n * 3}": (0, n - 1),  # a suffix longer than the file is the whole file
        f"bytes={n - 1}-{n - 1}": (n - 1, n - 1),
    }
    for header, (first, last) in cases.items():
        resp = await sclient.get(f"/v1/assets/{asset_id}", headers={"Range": header})
        assert resp.status_code == 206, header
        assert resp.headers["content-range"] == f"bytes {first}-{last}/{n}", header
        assert resp.headers["content-length"] == str(last - first + 1) and resp.content == data[first : last + 1], header
        assert resp.headers["etag"] == f'"{asset_id}"'


@pytest.mark.parametrize("header", ["bytes={n}-", "bytes={n}-{m}", "bytes=999999999-", "bytes=-0"])
async def test_unsatisfiable_ranges_are_416(sclient: httpx.AsyncClient, header: str) -> None:
    asset_id, data = await uploaded(sclient)
    resp = await sclient.get(f"/v1/assets/{asset_id}", headers={"Range": header.format(n=len(data), m=len(data) + 10)})
    err = assert_envelope(resp, 416, "range_not_satisfiable")
    assert resp.headers["content-range"] == f"bytes */{len(data)}" and err["details"] == {"size": len(data)}


@pytest.mark.parametrize("header", ["bytes=5-2", "bytes=0-1,5-9", "bytes=abc", "items=0-5", "bytes=", "bytes=-", "0-5", "bytes=0-5-9", "bytes=\x7f"])
async def test_ranges_that_cannot_be_honoured_serve_the_whole_file(sclient: httpx.AsyncClient, header: str) -> None:
    asset_id, data = await uploaded(sclient)
    resp = await sclient.get(f"/v1/assets/{asset_id}", headers={"Range": header})
    assert resp.status_code == 200 and resp.content == data and "content-range" not in resp.headers


async def test_if_range_guards_partial_responses(sclient: httpx.AsyncClient) -> None:
    asset_id, data = await uploaded(sclient)
    ok = await sclient.get(f"/v1/assets/{asset_id}", headers={"Range": "bytes=0-9", "If-Range": f'"{asset_id}"'})
    assert ok.status_code == 206 and ok.content == data[:10]
    stale = await sclient.get(f"/v1/assets/{asset_id}", headers={"Range": "bytes=0-9", "If-Range": '"a_stale"'})
    assert stale.status_code == 200 and stale.content == data


async def test_conditional_requests_answer_304(sclient: httpx.AsyncClient) -> None:
    asset_id, data = await uploaded(sclient)
    etag = f'"{asset_id}"'
    for value in (etag, f"W/{etag}", f'"other", {etag}', "*"):
        resp = await sclient.get(f"/v1/assets/{asset_id}", headers={"If-None-Match": value})
        assert resp.status_code == 304 and resp.content == b"" and resp.headers["etag"] == etag, value
    resp = await sclient.get(f"/v1/assets/{asset_id}", headers={"If-None-Match": '"a_something_else"'})
    assert resp.status_code == 200 and resp.content == data
    both = await sclient.get(f"/v1/assets/{asset_id}", headers={"If-None-Match": etag, "Range": "bytes=0-9"})
    assert both.status_code == 304


async def test_unknown_assets_are_404(sclient: httpx.AsyncClient) -> None:
    for path in ("a_000000000000000000000000", "a_zz", "a_" + "0" * 5000, "j_0123456789abcdef"):
        for suffix in ("", "/meta"):
            assert_envelope(await sclient.get(f"/v1/assets/{path}{suffix}"), 404, "asset_not_found")
    assert (await sclient.get("/v1/assets/../../etc/passwd")).status_code == 404  # normalised away by the client, or no route


async def test_meta_is_the_sidecar(sclient: httpx.AsyncClient) -> None:
    ids = await make_assets(sclient, "meta" + secrets.token_hex(3))
    side = (await sclient.get(f"/v1/assets/{ids['bass']}/meta")).json()
    assert side["asset_id"] == ids["bass"] and side["job_id"] == ids["bass_job"] and side["type"] == "generate.sample"
    assert side["requested"]["note"] == 40 and side["tags"] == {"family": "bass"} and side["rating"] is None
    assert {"model", "family", "checkpoint", "licence", "seed", "params", "measured", "sample_rate", "channels", "bit_depth", "created_at"} <= set(side)


# ---- search ----


async def test_search_by_text_type_model_and_job(sclient: httpx.AsyncClient) -> None:
    marker = "txt" + secrets.token_hex(3)
    ids = await make_assets(sclient, marker)
    found = await search(sclient, text=marker)
    assert [a["asset_id"] for a in found] == [ids["loop"], ids["bass"], ids["synth"]]  # newest first
    assert {"asset_id", "job_id", "type", "kind", "model", "prompt", "tags", "rating", "note", "size_bytes", "created_at"} == set(found[0])
    assert [a["asset_id"] for a in await search(sclient, text=marker.upper() + " ALPHA")] == [ids["synth"]]
    assert [a["asset_id"] for a in await search(sclient, text=marker, type="generate.sample")] == [ids["bass"]]
    assert len(await search(sclient, text=marker, model="testtone")) == 3 and await search(sclient, text=marker, model="yin") == []
    assert [a["asset_id"] for a in await search(sclient, job_id=ids["loop_job"])] == [ids["loop"]]
    assert [a["asset_id"] for a in await search(sclient, text=marker, limit=2)] == [ids["loop"], ids["bass"]]
    assert await search(sclient, text="%_" + marker) == []  # LIKE wildcards in the query are literals


async def test_search_by_tags(sclient: httpx.AsyncClient) -> None:
    marker = "tag" + secrets.token_hex(3)
    ids = await make_assets(sclient, marker)
    ours = lambda **q: search(sclient, text=marker, **q)  # noqa: E731
    assert [a["asset_id"] for a in await ours(tags='{"family":"synth"}')] == [ids["synth"]]
    assert [a["asset_id"] for a in await ours(tags="family:synth")] == [ids["synth"]]
    assert [a["asset_id"] for a in await ours(tags="FAMILY:SYNTH")] == [ids["synth"]]  # tags are case-insensitive
    assert [a["asset_id"] for a in await ours(tags="mood:wide")] == [ids["synth"]]
    both = await sclient.get("/v1/assets", params=[("text", marker), ("tags", "family:synth"), ("tags", "family:bass")])
    assert {a["asset_id"] for a in both.json()} == {ids["synth"], ids["bass"]}  # several values of one key: any of them
    and_ = await sclient.get("/v1/assets", params=[("text", marker), ("tags", "family:synth"), ("tags", "mood:dark")])
    assert [a["asset_id"] for a in and_.json()] == [ids["synth"]]  # different keys: all of them
    none = await sclient.get("/v1/assets", params=[("text", marker), ("tags", "family:synth"), ("tags", "mood:happy")])
    assert none.json() == []
    obj = await ours(tags='{"family": ["synth", "bass"], "mood": "dark"}')
    assert [a["asset_id"] for a in obj] == [ids["synth"]]


async def test_search_by_rating_and_time(sclient: httpx.AsyncClient) -> None:
    marker = "rt" + secrets.token_hex(3)
    ids = await make_assets(sclient, marker)
    assert (await sclient.post(f"/v1/assets/{ids['synth']}/rate", json={"rating": 5})).status_code == 200
    assert (await sclient.post(f"/v1/assets/{ids['bass']}/rate", json={"rating": 2})).status_code == 200
    assert {a["asset_id"] for a in await search(sclient, text=marker, rated="true")} == {ids["synth"], ids["bass"]}
    assert [a["asset_id"] for a in await search(sclient, text=marker, rated="false")] == [ids["loop"]]
    assert [a["asset_id"] for a in await search(sclient, text=marker, min_rating=3)] == [ids["synth"]]
    assert len(await search(sclient, text=marker, min_rating=0)) == 2
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    assert len(await search(sclient, text=marker, since=past)) == 3 and await search(sclient, text=marker, since=future) == []
    assert len(await search(sclient, text=marker, since="2000-01-01")) == 3


@pytest.mark.parametrize(
    ("query", "path"),
    [
        ({"limit": "0"}, "limit"),
        ({"limit": "201"}, "limit"),
        ({"limit": "many"}, "limit"),
        ({"since": "yesterday-ish"}, "since"),
        ({"min_rating": "9"}, "min_rating"),
        ({"min_rating": "-1"}, "min_rating"),
        ({"rated": "maybe"}, "rated"),
        ({"tags": "no-colon"}, "tags"),
        ({"tags": "{broken"}, "tags"),
        ({"tags": "[1,2]"}, "tags"),
        ({"tags": '{"a": [1]}'}, "tags"),
        ({"tags": "k:" + "v" * 300}, "tags"),
        ({"text": "x" * 201}, "text"),
    ],
)
async def test_bad_search_queries_are_422(sclient: httpx.AsyncClient, query: dict[str, str], path: str) -> None:
    err = assert_envelope(await sclient.get("/v1/assets", params=query), 422, "validation_error")
    assert path in [d["path"] for d in err["details"]], err


async def test_tag_filter_count_is_capped(sclient: httpx.AsyncClient) -> None:
    many = [("tags", f"k{i}:v") for i in range(65)]
    assert_envelope(await sclient.get("/v1/assets", params=many), 422, "validation_error")


async def test_sql_metacharacters_in_queries_are_inert(sclient: httpx.AsyncClient) -> None:
    for hostile in ("' OR 1=1 --", "'; DROP TABLE assets; --", '" OR ""="'):
        assert await search(sclient, text=hostile) == []
        assert await search(sclient, type=hostile) == [] and await search(sclient, model=hostile) == []
        assert await search(sclient, tags=f"k:{hostile}") == []
    assert (await sclient.get("/v1/assets")).status_code == 200


# ---- rating ----


async def test_rate_an_asset(sclient: httpx.AsyncClient) -> None:
    asset_id, _ = await uploaded(sclient)
    resp = await sclient.post(f"/v1/assets/{asset_id}/rate", json={"rating": 4, "note": "good low end"})
    assert resp.status_code == 200 and resp.json() == {"asset_id": asset_id, "rating": 4, "note": "good low end"}
    side = (await sclient.get(f"/v1/assets/{asset_id}/meta")).json()
    assert side["rating"] == 4 and side["note"] == "good low end"
    again = await sclient.post(f"/v1/assets/{asset_id}/rate", json={"rating": 0})  # a rating of 0 is still a verdict; the note stays
    assert again.json() == {"asset_id": asset_id, "rating": 0, "note": "good low end"}
    assert [a for a in await search(sclient, rated="true", limit=200) if a["asset_id"] == asset_id][0]["rating"] == 0


@pytest.mark.parametrize("rating", [6, -1, 3.5, "3", None, True, [4], {"v": 1}])
async def test_invalid_ratings_are_422(sclient: httpx.AsyncClient, rating: Any) -> None:
    asset_id, _ = await uploaded(sclient)
    err = assert_envelope(await sclient.post(f"/v1/assets/{asset_id}/rate", json={"rating": rating}), 422, "validation_error")
    assert err["details"][0]["path"] == "rating"
    assert (await sclient.get(f"/v1/assets/{asset_id}/meta")).json()["rating"] is None


async def test_rate_validates_the_body(sclient: httpx.AsyncClient) -> None:
    asset_id, _ = await uploaded(sclient)
    url = f"/v1/assets/{asset_id}/rate"
    assert_envelope(await sclient.post(url, json={}), 422, "validation_error")
    assert_envelope(await sclient.post(url, json={"rating": 3, "stars": 3}), 422, "validation_error")
    assert_envelope(await sclient.post(url, json={"rating": 3, "note": "x" * 2001}), 422, "validation_error")
    assert_envelope(await sclient.post(url, json={"rating": 3, "note": 7}), 422, "validation_error")
    assert_envelope(await sclient.post(url, content=b'{"rating": NaN}', headers={"Content-Type": "application/json"}), 400, "invalid_json")
    assert_envelope(await sclient.post(url, content=b"x" * 70_000, headers={"Content-Type": "application/json"}), 413, "payload_too_large")
    assert_envelope(await sclient.post("/v1/assets/a_000000000000000000000000/rate", json={"rating": 3}), 404, "asset_not_found")
    assert_envelope(await sclient.post("/v1/assets/not-an-id/rate", json={"rating": 3}), 404, "asset_not_found")
