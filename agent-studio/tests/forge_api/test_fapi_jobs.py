"""Submitting jobs over HTTP, polling them and downloading what they make."""

from __future__ import annotations

import io
import json
import zipfile

import httpx
import numpy as np
import pytest

from fapi_helpers import SHARED_LOOP, assert_envelope, decode_wav, melody, run_job, submit, upload, wait_job, wav_bytes

pytestmark = SHARED_LOOP


async def test_generate_loop_submit_poll_download(sclient: httpx.AsyncClient) -> None:
    resp = await sclient.post(
        "/v1/jobs",
        json={"type": "generate.loop", "params": {"bpm": 120, "bars": 2, "key": "F minor", "prompt": "warm pad", "seed": 5}},
    )
    assert resp.status_code == 202 and set(resp.json()) == {"job_id", "state"} and resp.json()["state"] == "queued"
    job_id = resp.json()["job_id"]
    job = await wait_job(sclient, job_id)
    assert job["state"] == "succeeded" and job["progress"] == 1.0 and job["error"] is None
    assert job["model"] == "testtone" and job["type"] == "generate.loop" and job["params"]["seed"] == 5
    assert {"queue_position", "eta_s", "created_at", "started_at", "finished_at", "priority", "lane"} <= set(job)
    (asset_id,) = job["result"]["asset_ids"]

    audio = await sclient.get(f"/v1/assets/{asset_id}")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    samples, sr = decode_wav(audio.content)
    assert sr == 48000 and samples.shape[0] == 2 and samples.shape[1] / sr == pytest.approx(4.0, abs=0.02)  # 2 bars at 120 bpm

    meta = (await sclient.get(f"/v1/assets/{asset_id}/meta")).json()
    assert meta["asset_id"] == asset_id and meta["job_id"] == job_id and meta["prompt"] == "warm pad"
    assert meta["requested"]["bpm"] == 120 and meta["requested"]["key"] == "F minor"
    assert meta["measured"]["bpm"] == pytest.approx(120, abs=3) and "lufs" in meta["measured"]


async def test_candidates_may_sit_beside_the_params(sclient: httpx.AsyncClient) -> None:
    job = await run_job(sclient, "generate.sfx", {"duration_s": 0.5}, candidates=3)
    assert len(job["result"]["asset_ids"]) == 3 and job["params"]["candidates"] == 3
    assert_envelope(
        await sclient.post("/v1/jobs", json={"type": "generate.sfx", "params": {"candidates": 2}, "candidates": 3}), 422, "validation_error"
    )


async def test_generate_instrument_bundle(sclient: httpx.AsyncClient) -> None:
    job = await run_job(sclient, "generate.instrument", {"name": "Test Keys", "note_low": 48, "note_high": 59, "step": 6, "length_s": 0.4})
    (asset_id,) = job["result"]["asset_ids"]
    meta = (await sclient.get(f"/v1/assets/{asset_id}/meta")).json()
    assert meta["kind"] == "instrument"
    resp = await sclient.get(f"/v1/assets/{asset_id}")
    assert resp.status_code == 200 and resp.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        names = archive.namelist()
        assert all(not n.startswith("/") and ".." not in n for n in names)
        assert {"Test_Keys.sfz", "Test_Keys.dspreset", "manifest.json"} <= set(names)  # the name is made filesystem-safe
        samples = [n for n in names if n.startswith("samples/") and n.endswith(".wav")]
        assert len(samples) == 2  # 12 keys / step 6
        manifest = json.loads(archive.read("manifest.json"))
        zones = manifest["zones"] if isinstance(manifest, dict) else manifest
        assert len(zones) == 2 and {z["sample"] for z in zones} == set(samples)
        audio, sr = decode_wav(archive.read(samples[0]))
        assert audio.shape[1] > 0 and sr == 48000
        sfz = archive.read("Test_Keys.sfz").decode()
        assert sfz.count("<region>") == 2 and all(f"sample={n}" in sfz for n in samples)
        assert archive.read("Test_Keys.dspreset").count(b"<sample ") == 2


async def test_transcribe_melody_returns_a_midi_asset(sclient: httpx.AsyncClient) -> None:
    asset = (await upload(sclient, wav_bytes(melody([60, 64, 67, 72])), "tune.wav"))["asset_id"]
    job = await run_job(sclient, "transcribe.melody", {"bpm": 100}, inputs={"audio": asset})
    (midi_id,) = job["result"]["asset_ids"]
    resp = await sclient.get(f"/v1/assets/{midi_id}")
    assert resp.headers["content-type"] == "audio/midi" and resp.content[:4] == b"MThd"


async def test_transform_uses_an_uploaded_input_and_records_the_parent(sclient: httpx.AsyncClient) -> None:
    asset = (await upload(sclient, wav_bytes(melody([57, 60]))))["asset_id"]
    job = await run_job(sclient, "transform.audio", {"strength": 0.7}, inputs={"audio": asset})
    meta = (await sclient.get(f"/v1/assets/{job['result']['asset_ids'][0]}/meta")).json()
    assert meta["parents"] == [asset] and job["inputs"] == {"audio": asset}


async def test_a_failing_worker_fails_the_job_not_the_gateway(sclient: httpx.AsyncClient) -> None:
    job = await wait_job(sclient, await submit(sclient, "generate.sfx", {"tags": {"x-fault": "error"}}))
    assert job["state"] == "failed" and "injected failure" in job["error"] and job["result"] is None
    assert (await sclient.get("/v1/health")).status_code == 200
    assert (await run_job(sclient, "generate.sfx", {"duration_s": 0.3}))["state"] == "succeeded"


async def test_idempotency_key_returns_the_existing_job(sclient: httpx.AsyncClient) -> None:
    first = await submit(sclient, "generate.sfx", {"duration_s": 0.3}, idempotency_key="idem-1")
    second = await submit(sclient, "generate.sfx", {"duration_s": 0.3}, idempotency_key="idem-1")
    other = await submit(sclient, "generate.sfx", {"duration_s": 0.3}, idempotency_key="idem-2")
    assert first == second != other
    await wait_job(sclient, first)
    assert await submit(sclient, "generate.sfx", {"duration_s": 0.3}, idempotency_key="idem-1") == first


async def test_validation_errors_are_422_with_field_paths(sclient: httpx.AsyncClient) -> None:
    resp = await sclient.post("/v1/jobs", json={"type": "generate.loop", "params": {"bpm": 999, "bars": 0, "post": {"fade_in_ms": -1}}})
    err = assert_envelope(resp, 422, "validation_error")
    paths = {d["path"]: d["message"] for d in err["details"]}
    assert {"params.bpm", "params.bars", "params.post.fade_in_ms"} <= set(paths) and all(paths.values())


async def test_unknown_fields_are_rejected(sclient: httpx.AsyncClient) -> None:
    err = assert_envelope(await sclient.post("/v1/jobs", json={"type": "generate.sfx", "params": {"bogus": 1}}), 422, "validation_error")
    assert [d["path"] for d in err["details"]] == ["params.bogus"]
    err = assert_envelope(await sclient.post("/v1/jobs", json={"type": "generate.sfx", "params": {}, "bogus": 1}), 422, "validation_error")
    assert err["details"] == [{"path": "bogus", "message": "unknown field"}]
    err = assert_envelope(
        await sclient.post("/v1/jobs", json={"type": "generate.sfx", "params": {"post": {"warp": 1}}}), 422, "validation_error"
    )
    assert [d["path"] for d in err["details"]] == ["params.post.warp"]


@pytest.mark.parametrize(
    ("body", "path"),
    [
        ({}, "type"),
        ({"type": "make.coffee"}, "type"),
        ({"type": 7}, "type"),
        ({"type": "generate.sfx", "params": [1]}, "params"),
        ({"type": "generate.sfx", "priority": "urgent"}, "priority"),
        ({"type": "generate.sfx", "priority": ["high"]}, "priority"),
        ({"type": "generate.sfx", "model": "nope"}, "model"),
        ({"type": "generate.sfx", "model": "yin"}, "model"),
        ({"type": "generate.sfx", "idempotency_key": ""}, "idempotency_key"),
        ({"type": "transform.audio"}, "inputs.audio"),
        ({"type": "transform.audio", "inputs": {"audio": "a_000000000000000000000000"}}, "inputs.audio"),
        ({"type": "transform.audio", "inputs": {"audio": "../../etc/passwd"}}, "inputs.audio"),
        ({"type": "generate.sfx", "inputs": {"audio": "a_000000000000000000000000"}}, "inputs.audio"),
        ({"type": "generate.sfx", "inputs": "x"}, "inputs"),
    ],
)
async def test_bad_submissions_name_the_offending_field(sclient: httpx.AsyncClient, body: dict, path: str) -> None:  # type: ignore[type-arg]
    err = assert_envelope(await sclient.post("/v1/jobs", json=body), 422, "validation_error")
    assert path in [d["path"] for d in err["details"]], err


async def test_wrong_kind_of_input_asset_is_a_422(sclient: httpx.AsyncClient) -> None:
    from fapi_helpers import zip_bytes

    session = (await upload_session(sclient, zip_bytes(["s.scraft"])))["asset_id"]
    err = assert_envelope(await sclient.post("/v1/jobs", json={"type": "transform.audio", "inputs": {"audio": session}}), 422, "validation_error")
    assert err["details"][0]["path"] == "inputs.audio"


async def upload_session(client: httpx.AsyncClient, data: bytes) -> dict:  # type: ignore[type-arg]
    return await upload(client, data, "s.zip")


@pytest.mark.parametrize(
    ("content", "status", "code"),
    [
        (b"{not json", 400, "invalid_json"),
        (b"", 400, "invalid_json"),
        (b"\xff\xfe\x00", 400, "invalid_json"),
        (b'{"type": "generate.sfx", "params": {"seed": NaN}}', 400, "invalid_json"),
        (b'{"type": "generate.sfx", "params": {"seed": Infinity}}', 400, "invalid_json"),
        (b'{"type": "generate.sfx", "params": {"duration_s": 1e999}}', 422, "validation_error"),
        (b'["generate.sfx"]', 422, "validation_error"),
        (b'"text"', 422, "validation_error"),
        (b"[" * 100_000, 400, "invalid_json"),
    ],
)
async def test_malformed_json_bodies(sclient: httpx.AsyncClient, content: bytes, status: int, code: str) -> None:
    assert_envelope(await sclient.post("/v1/jobs", content=content, headers={"Content-Type": "application/json"}), status, code)


async def test_oversized_json_bodies_are_refused(sclient: httpx.AsyncClient) -> None:
    huge = b'{"type":"generate.sfx","params":{"prompt":"' + b"a" * (5 * 1024 * 1024) + b'"}}'
    assert_envelope(await sclient.post("/v1/jobs", content=huge, headers={"Content-Type": "application/json"}), 413, "payload_too_large")


async def test_get_and_list_jobs(sclient: httpx.AsyncClient) -> None:
    job = await run_job(sclient, "generate.sfx", {"duration_s": 0.3})
    assert (await sclient.get(f"/v1/jobs/{job['job_id']}")).json()["state"] == "succeeded"
    assert_envelope(await sclient.get("/v1/jobs/j_doesnotexist"), 404, "job_not_found")
    assert_envelope(await sclient.get("/v1/jobs/" + "x" * 5000), 404, "job_not_found")
    recent = (await sclient.get("/v1/jobs?limit=200")).json()
    assert isinstance(recent, list) and job["job_id"] in [j["job_id"] for j in recent]
    done = (await sclient.get("/v1/jobs?state=succeeded&limit=3")).json()
    assert 1 <= len(done) <= 3 and all(j["state"] == "succeeded" for j in done)
    assert (await sclient.get("/v1/jobs?state=failed")).status_code == 200


@pytest.mark.parametrize("query", ["state=exploded", "limit=0", "limit=201", "limit=abc", "state=" + "x" * 40])
async def test_bad_list_queries_are_422(sclient: httpx.AsyncClient, query: str) -> None:
    assert_envelope(await sclient.get(f"/v1/jobs?{query}"), 422, "validation_error")


async def test_analysis_job_result_has_no_assets(sclient: httpx.AsyncClient) -> None:
    asset = (await upload(sclient, wav_bytes(melody([69, 69]))))["asset_id"]
    job = await run_job(sclient, "analyze.audio", inputs={"audio": asset})
    assert job["model"] == "analysis" and job["result"]["asset_ids"] == [] and job["result"]["analysis"]["peak_dbfs"] < 0


async def test_responses_never_contain_nan_or_infinity(sclient: httpx.AsyncClient) -> None:
    silent = (await upload(sclient, wav_bytes(np.zeros((1, 22050), np.float32)), "silence.wav"))["asset_id"]
    job_id = await submit(sclient, "analyze.audio", inputs={"audio": silent})
    await wait_job(sclient, job_id)
    for path in (f"/v1/jobs/{job_id}", "/v1/jobs?limit=5", f"/v1/assets/{silent}/meta", "/v1/assets?limit=5"):
        text = (await sclient.get(path)).text
        json.loads(text, parse_constant=lambda name: pytest.fail(f"{name} in {path}"))
        assert "NaN" not in text and "Infinity" not in text
    analysis = (await sclient.get(f"/v1/jobs/{job_id}")).json()["result"]["analysis"]
    assert analysis["peak_dbfs"] <= -100  # digital silence
