"""The Gary-compatible shim apps (DESIGN 3.9): /generate, /audio2audio and /poll_status over testtone."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from forge.gary_shim import create_foundation_app, create_sa3_app, poll_payload
from forge.service import ForgeService

from fapi_helpers import SHARED_LOOP, SLOW, decode_wav, melody, tone, wav_bytes

pytestmark = SHARED_LOOP

FOUNDATION_BODY: dict[str, Any] = {
    "seed": 42, "bars": 1, "host_bpm": 120, "key_root": "F", "key_mode": "minor",
    "family": "Synth", "subfamily": "Pad", "descriptor_knob_a": "Warm", "descriptor_knob_b": "Wide", "descriptor_knob_c": "",
    "descriptors_extra": ["Gritty", "Airy"], "reverb_enabled": True, "reverb_amount": 0.6, "delay_enabled": False, "delay_amount": 0.3,
    "behavior_tags": ["Pad", "Slow"], "custom_prompt_override": "", "guidance_scale": 7.0, "steps": 75,
}


def local_client(app: FastAPI, **headers: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8015", headers=headers, timeout=60)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def foundation_app(sservice: ForgeService) -> FastAPI:
    return create_foundation_app(sservice, sservice.config, family="testtone")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def foundation(foundation_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with local_client(foundation_app) as c:
        yield c


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sa3(sservice: ForgeService) -> AsyncIterator[httpx.AsyncClient]:
    async with local_client(create_sa3_app(sservice, sservice.config, family="testtone")) as c:
        yield c


async def poll_until_done(client: httpx.AsyncClient, session_id: str, timeout: float = 30) -> dict[str, Any]:
    async def poll() -> dict[str, Any]:
        while True:
            resp = await client.get(f"/poll_status/{session_id}")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert {"success", "generation_in_progress", "progress", "queue_status", "status"} <= set(body)
            assert set(body["queue_status"]) == {"position", "estimated_seconds"} and 0 <= body["progress"] <= 100
            if not body["generation_in_progress"]:
                return body  # type: ignore[no-any-return]
            await asyncio.sleep(0.02)

    return await asyncio.wait_for(poll(), timeout)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---- the wire contract ----


async def test_foundation_generate_then_poll_returns_base64_wav(foundation: httpx.AsyncClient) -> None:
    resp = await foundation.post("/generate", json={**FOUNDATION_BODY, "bars": 2})
    assert resp.status_code == 200 and set(resp.json()) == {"success", "session_id"} and resp.json()["success"] is True
    done = await poll_until_done(foundation, resp.json()["session_id"])
    assert done["success"] is True and done["status"] == "completed" and done["progress"] == 100
    assert "error" not in done and done["queue_status"]["position"] == 0
    audio, sr = decode_wav(base64.b64decode(done["audio_data"], validate=True))
    assert audio.shape[0] == 2 and audio.shape[1] / sr == pytest.approx(4.0, abs=0.02)  # 2 bars at 120 bpm


async def test_the_gary_fields_become_forge_tags_and_params(foundation: httpx.AsyncClient, foundation_app: FastAPI, sservice: ForgeService) -> None:
    session_id = (await foundation.post("/generate", json=FOUNDATION_BODY)).json()["session_id"]
    await poll_until_done(foundation, session_id)
    job = sservice.get_job(foundation_app.state.sessions.get(session_id))
    assert job.model == "testtone" and job.state == "succeeded"
    assert job.params["bars"] == 1 and job.params["bpm"] == 120 and job.params["key"] == "F minor" and job.params["prompt"] == ""
    assert job.params["tags"] == {
        "family": "Synth", "subfamily": "Pad", "descriptors": ["Warm", "Wide", "Gritty", "Airy"],
        "behaviour": ["Pad", "Slow"], "effects": ["Medium Reverb"],
    }
    assert "guidance_scale" not in job.params and "steps" not in job.params


async def test_a_custom_prompt_and_a_random_seed(foundation: httpx.AsyncClient, foundation_app: FastAPI, sservice: ForgeService) -> None:
    body = {"bars": 1, "host_bpm": 90, "custom_prompt_override": "my own prompt", "seed": -1}
    session_id = (await foundation.post("/generate", json=body)).json()["session_id"]
    await poll_until_done(foundation, session_id)
    job = sservice.get_job(foundation_app.state.sessions.get(session_id))
    assert job.params["prompt"] == "my own prompt" and job.params["seed"] >= 0 and job.params["key"] is None and job.params["bpm"] == 90


async def test_audio2audio_transforms_the_posted_audio(foundation: httpx.AsyncClient, foundation_app: FastAPI, sservice: ForgeService) -> None:
    body = {**FOUNDATION_BODY, "audio_data": b64(wav_bytes(melody([60, 64, 67]))), "init_noise_level": 0.35}
    resp = await foundation.post("/audio2audio", json=body)
    assert resp.status_code == 200, resp.text
    done = await poll_until_done(foundation, resp.json()["session_id"])
    assert done["status"] == "completed"
    audio, sr = decode_wav(base64.b64decode(done["audio_data"]))
    assert audio.shape[1] > 0
    job = sservice.get_job(foundation_app.state.sessions.get(resp.json()["session_id"]))
    assert job.type == "transform.audio" and job.params["strength"] == pytest.approx(0.35) and job.params["tags"]["bpm"] == "120" and job.params["tags"]["key"] == "F minor"
    assert job.inputs["audio"].startswith("a_")


async def test_audio_data_may_be_a_data_uri_with_line_breaks(foundation: httpx.AsyncClient) -> None:
    text = b64(wav_bytes(tone(440, 0.5)))
    wrapped = "data:audio/wav;base64," + "\n".join(text[i : i + 76] for i in range(0, len(text), 76))
    resp = await foundation.post("/audio2audio", json={"audio_data": wrapped, "init_noise_level": 7})  # >1 is clamped to 1
    assert resp.status_code == 200, resp.text
    assert (await poll_until_done(foundation, resp.json()["session_id"]))["status"] == "completed"


async def test_sa3_generates_effects_and_loops(sa3: httpx.AsyncClient, sservice: ForgeService) -> None:
    sfx = (await sa3.post("/generate", json={"prompt": "metal door slam", "duration": 1.0, "seed": 9})).json()["session_id"]
    done = await poll_until_done(sa3, sfx)
    audio, sr = decode_wav(base64.b64decode(done["audio_data"]))
    assert done["status"] == "completed" and audio.shape[1] / sr == pytest.approx(1.0, abs=0.05)
    loop = (await sa3.post("/generate", json={"prompt": "drums", "bpm": 120, "bars": 1, "steps": 8, "cfg_scale": 6})).json()["session_id"]
    done = await poll_until_done(sa3, loop)
    audio, sr = decode_wav(base64.b64decode(done["audio_data"]))
    assert audio.shape[1] / sr == pytest.approx(2.0, abs=0.05)
    a2a = await sa3.post("/audio2audio", json={"prompt": "darker", "audio_data": b64(wav_bytes(tone())), "init_noise_level": 0.5})
    assert (await poll_until_done(sa3, a2a.json()["session_id"]))["status"] == "completed"


async def test_health_answers_without_credentials(foundation: httpx.AsyncClient) -> None:
    resp = await foundation.get("/health")
    assert resp.status_code == 200 and resp.json() == {"status": "ok", "service": "foundation"}


# ---- errors ----


async def test_unknown_and_malformed_sessions(foundation: httpx.AsyncClient) -> None:
    for session in ("deadbeef", "x" * 3000, "a%00b"):
        resp = await foundation.get(f"/poll_status/{session}")
        assert resp.status_code == 404 and resp.json() == {"success": False, "error": "unknown session"}


@pytest.mark.parametrize(
    ("patch", "needle"),
    [
        ({"bars": 99}, "bars"),
        ({"bars": "lots"}, "bars"),
        ({"bars": True}, "bars"),
        ({"host_bpm": 10}, "host_bpm"),
        ({"host_bpm": [120]}, "host_bpm"),
        ({"seed": "NaN"}, "seed"),
        ({"key_root": "H"}, "key"),
        ({"key_root": "C" * 50}, "key_root"),
        ({"family": "x" * 500}, "family"),
        ({"behavior_tags": ["a"] * 100}, "behavior_tags"),
        ({"behavior_tags": [1, 2]}, "behavior_tags"),
        ({"custom_prompt_override": "p" * 3000}, "custom_prompt_override"),
        ({"init_noise_level": "x"}, "init_noise_level"),
    ],
)
async def test_bad_foundation_requests_are_400_with_the_reason(foundation: httpx.AsyncClient, patch: dict[str, Any], needle: str) -> None:
    resp = await foundation.post("/generate", json={**FOUNDATION_BODY, **patch})
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["success"] is False and needle in body["error"] and set(body) == {"success", "error"}


async def test_bad_audio2audio_requests(foundation: httpx.AsyncClient) -> None:
    for audio in (None, "", 7, "not base64 !!!", b64(b"definitely not a wav"), b64(b"RIFF\x00\x00\x00\x00WAVE")):
        resp = await foundation.post("/audio2audio", json={**FOUNDATION_BODY, "audio_data": audio})
        assert resp.status_code == 400 and resp.json()["success"] is False, audio


async def test_audio_data_is_size_capped(foundation: httpx.AsyncClient, sservice: ForgeService, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sservice.config.uploads, "max_bytes", 1000)
    resp = await foundation.post("/audio2audio", json={"audio_data": b64(wav_bytes(tone(440, 0.5)))})
    assert resp.status_code in (400, 413) and resp.json()["success"] is False


async def test_non_json_bodies(foundation: httpx.AsyncClient) -> None:
    for content in (b"{nope", b"[1]", b'"s"', b'{"bars": NaN}'):
        resp = await foundation.post("/generate", content=content, headers={"Content-Type": "application/json"})
        assert resp.status_code in (400, 422) and resp.json()["success"] is False, content
    resp = await foundation.post("/generate", content=b"x" * (2 * 1024 * 1024), headers={"Content-Type": "application/json"})
    assert resp.status_code == 413 and resp.json()["success"] is False


async def test_a_job_that_forge_rejects_is_reported_in_the_shim_shape(sa3: httpx.AsyncClient) -> None:
    resp = await sa3.post("/generate", json={"prompt": "x", "duration": 100})
    assert resp.status_code == 400 and "duration" in resp.json()["error"]


async def test_no_model_in_the_family_is_a_503(sservice: ForgeService) -> None:
    async with local_client(create_foundation_app(sservice, sservice.config)) as c:  # the real foundation family is not configured here
        resp = await c.post("/generate", json={"bars": 1})
    assert resp.status_code == 503 and "foundation" in resp.json()["error"] and resp.json()["success"] is False


async def test_a_named_model_that_does_not_exist(sa3: httpx.AsyncClient) -> None:
    resp = await sa3.post("/generate", json={"prompt": "x", "model": "sa3-medium"})
    assert resp.status_code == 503 and resp.json()["success"] is False


# ---- local-only guard ----


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({"Origin": "https://evil.example"}, 403),
        ({"Origin": "null"}, 403),
        ({"Host": "evil.example:8015"}, 403),
        ({"Host": "forge.test"}, 403),
        ({"Host": "127.0.0.1.evil.example"}, 403),
        ({"Content-Type": "text/plain"}, 415),
        ({"Content-Type": "application/x-www-form-urlencoded"}, 415),
        ({"Host": "localhost:8015"}, 200),
        ({"Host": "[::1]:8015"}, 200),
        ({"Host": "127.0.0.1"}, 200),
        ({"Content-Type": "application/json; charset=utf-8"}, 200),
    ],
)
async def test_browsers_and_rebinding_are_refused(sservice: ForgeService, headers: dict[str, str], status: int) -> None:
    async with local_client(create_sa3_app(sservice, sservice.config, family="testtone")) as c:
        resp = await c.post("/generate", content=b'{"prompt": "x", "duration": 0.2}', headers={"Content-Type": "application/json", **headers})
        assert resp.status_code == status, resp.text
        if status != 200:
            assert resp.json()["success"] is False
        assert (await c.get("/health", headers={k: v for k, v in headers.items() if k in ("Origin", "Host")})).status_code == (403 if status == 403 else 200)


# ---- polling states ----


async def test_poll_reports_queue_progress_failure_and_cancellation(sservice: ForgeService) -> None:
    slow = await sservice.submit("generate.sfx", {"tags": SLOW})
    queued = await sservice.submit("generate.sfx", {"duration_s": 0.3})
    while sservice.get_job(slow.job_id).state != "running":
        await asyncio.sleep(0.02)
    running = await poll_payload(sservice, slow.job_id)
    assert running["generation_in_progress"] is True and running["status"] == "processing" and "audio_data" not in running
    waiting = await poll_payload(sservice, queued.job_id)
    assert waiting["status"] == "queued" and waiting["queue_status"]["position"] == 1 and waiting["success"] is True
    await sservice.cancel(queued.job_id)
    await sservice.cancel(slow.job_id)
    while sservice.get_job(slow.job_id).state != "cancelled":
        await asyncio.sleep(0.02)
    gone = await poll_payload(sservice, slow.job_id)
    assert gone["success"] is False and gone["status"] == "cancelled" and gone["generation_in_progress"] is False and gone["error"]
    failing = await sservice.submit("generate.sfx", {"tags": {"x-fault": "error"}})
    while sservice.get_job(failing.job_id).state not in ("failed", "succeeded"):
        await asyncio.sleep(0.02)
    failed = await poll_payload(sservice, failing.job_id)
    assert failed["success"] is False and failed["status"] == "failed" and "injected failure" in failed["error"] and "audio_data" not in failed


async def test_midi_results_are_not_returned_as_audio(sservice: ForgeService, tmp_path: Any) -> None:
    from fapi_helpers import write_input

    asset = await write_input(sservice, tmp_path, melody([60, 64]))
    job = await sservice.submit("transcribe.melody", {}, inputs={"audio": asset})
    while sservice.get_job(job.job_id).state not in ("failed", "succeeded"):
        await asyncio.sleep(0.02)
    payload = await poll_payload(sservice, job.job_id)
    assert payload["success"] is False and "WAV" in payload["error"] and "audio_data" not in payload
