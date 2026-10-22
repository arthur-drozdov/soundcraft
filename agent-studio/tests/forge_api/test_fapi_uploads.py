"""``POST /v1/uploads``: multipart handling, byte and duration caps, WAV and session-zip validation."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import httpx
import numpy as np
import pytest

from forge import store
from forge.service import ForgeService

from fapi_helpers import SHARED_LOOP, assert_envelope, melody, run_job, tone, upload, wav_bytes, zip_bytes

pytestmark = SHARED_LOOP


async def post_file(client: httpx.AsyncClient, data: bytes, name: str = "x.wav", field: str = "file") -> httpx.Response:
    return await client.post("/v1/uploads", files={field: (name, data, "application/octet-stream")})


async def test_wav_upload_becomes_an_audio_asset(sclient: httpx.AsyncClient) -> None:
    data = wav_bytes(tone(330, 1.0))
    body = await upload(sclient, data)
    assert body["kind"] == "audio" and body["asset_id"] == "a_" + hashlib.sha256(data).hexdigest()[:24]
    download = await sclient.get(f"/v1/assets/{body['asset_id']}")
    assert download.content == data and download.headers["content-type"] == "audio/wav"
    meta = (await sclient.get(f"/v1/assets/{body['asset_id']}/meta")).json()
    assert meta["type"] == "upload" and meta["kind"] == "audio" and meta["measured"]["duration_s"] == pytest.approx(1.0, abs=0.01)


async def test_uploading_the_same_bytes_twice_gives_the_same_asset(sclient: httpx.AsyncClient) -> None:
    data = wav_bytes(tone(550, 0.5))
    assert (await upload(sclient, data))["asset_id"] == (await upload(sclient, data, "other-name.wav"))["asset_id"]


@pytest.mark.parametrize(("sent", "kept"), [("../../etc/passwd.wav", "passwd.wav"), ("..\\..\\evil.wav", "evil.wav"), ("C:\\x\\y.wav", "y.wav")])
async def test_original_name_is_reduced_to_a_harmless_base_name(sclient: httpx.AsyncClient, sent: str, kept: str) -> None:
    body = await upload(sclient, wav_bytes(tone(600 + 7 * len(sent), 0.5)), sent)  # distinct audio, or the first upload's sidecar would win
    meta = (await sclient.get(f"/v1/assets/{body['asset_id']}/meta")).json()
    assert meta["original_name"] == kept


async def test_an_uploaded_wav_is_usable_as_a_job_input(sclient: httpx.AsyncClient) -> None:
    asset = (await upload(sclient, wav_bytes(melody([60, 64, 67]))))["asset_id"]
    job = await run_job(sclient, "analyze.audio", inputs={"audio": asset})
    assert job["result"]["analysis"]["duration_s"] == pytest.approx(1.5, abs=0.05)
    midi_job = await run_job(sclient, "transcribe.melody", inputs={"audio": asset})
    assert midi_job["model"] == "yin" and len(midi_job["result"]["asset_ids"]) == 1


@pytest.mark.parametrize(
    "data",
    [
        b"just some text, not audio",
        b"",
        b"RIFF\x00\x00\x00\x00WAVEfmt ",  # looks like a WAV and is truncated
        b"RIFF" + b"\x00" * 200,
        bytes(range(256)) * 40,
    ],
)
async def test_non_wav_uploads_are_rejected(sclient: httpx.AsyncClient, data: bytes) -> None:
    assert_envelope(await post_file(sclient, data), 400, "invalid_upload")


async def test_a_wav_with_no_audio_is_rejected(sclient: httpx.AsyncClient) -> None:
    empty = wav_bytes(tone(440, 1.0)[:, :0])
    assert_envelope(await post_file(sclient, empty), 400, "invalid_upload")


@pytest.fixture
def tiny_limits(sservice: ForgeService, monkeypatch: pytest.MonkeyPatch) -> ForgeService:
    monkeypatch.setattr(sservice.config.uploads, "max_seconds", 1.0)
    monkeypatch.setattr(sservice.config.uploads, "max_bytes", 100_000)
    return sservice


async def test_too_long_audio_is_rejected(sclient: httpx.AsyncClient, tiny_limits: ForgeService) -> None:
    ok = wav_bytes(tone(440, 0.9, sr=8000), 8000)
    assert (await post_file(sclient, ok)).status_code == 200
    err = assert_envelope(await post_file(sclient, wav_bytes(tone(440, 1.5, sr=8000), 8000)), 413, "upload_too_long")
    assert "1 s" in err["message"]


async def test_oversized_uploads_are_rejected(sclient: httpx.AsyncClient, tiny_limits: ForgeService) -> None:
    big = wav_bytes(tone(440, 0.9, sr=44100))  # ~79 KB: fits
    assert (await post_file(sclient, big)).status_code == 200
    bigger = wav_bytes(np.full((1, 60_000), 0.1, np.float32))  # 120 KB of PCM: over the cap, under the multipart allowance
    assert len(bigger) > 100_000
    assert_envelope(await post_file(sclient, bigger), 413, "upload_too_large")


async def test_declared_size_far_over_the_cap_is_refused_up_front(sclient: httpx.AsyncClient, tiny_limits: ForgeService) -> None:
    resp = await post_file(sclient, b"\x00" * 600_000)
    assert_envelope(resp, 413, "upload_too_large")


async def test_oversized_upload_without_content_length_is_cut_off(sclient: httpx.AsyncClient, tiny_limits: ForgeService) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        boundary_open = b'--XBOUND\r\nContent-Disposition: form-data; name="file"; filename="x.wav"\r\nContent-Type: audio/wav\r\n\r\n'
        yield boundary_open
        for _ in range(40):  # 40 x 64 KiB, far beyond the 100 000 byte cap plus multipart overhead allowance
            yield b"\x00" * 65536
        yield b"\r\n--XBOUND--\r\n"

    resp = await sclient.post("/v1/uploads", content=chunks(), headers={"Content-Type": "multipart/form-data; boundary=XBOUND"})
    assert_envelope(resp, 413, "upload_too_large")


async def test_missing_file_field_is_a_422_with_the_field_path(sclient: httpx.AsyncClient) -> None:
    err = assert_envelope(await post_file(sclient, wav_bytes(tone()), field="audio"), 422, "validation_error")
    assert err["details"] == [{"path": "file", "message": "a file field named 'file' is required"}]
    resp = await sclient.post("/v1/uploads", data={"file": "not a file"}, files={"x": ("x.bin", b"1")})
    assert_envelope(resp, 422, "validation_error")


async def test_non_multipart_bodies_are_refused(sclient: httpx.AsyncClient) -> None:
    assert_envelope(await sclient.post("/v1/uploads", content=wav_bytes(tone()), headers={"Content-Type": "audio/wav"}), 415, "unsupported_media_type")
    assert_envelope(await sclient.post("/v1/uploads", json={"file": "x"}), 415, "unsupported_media_type")


async def test_malformed_multipart_and_extra_files_are_400(sclient: httpx.AsyncClient) -> None:
    resp = await sclient.post("/v1/uploads", content=b"garbage", headers={"Content-Type": "multipart/form-data; boundary=B"})
    assert resp.status_code in (400, 422)
    resp = await sclient.post("/v1/uploads", content=b"x", headers={"Content-Type": "multipart/form-data"})
    assert_envelope(resp, 400, "invalid_multipart")
    two = [("file", ("a.wav", wav_bytes(tone()), "audio/wav")), ("file", ("b.wav", wav_bytes(tone(500)), "audio/wav"))]
    assert_envelope(await sclient.post("/v1/uploads", files=two), 400, "invalid_multipart")


async def test_session_zip_with_one_scraft_is_accepted(sclient: httpx.AsyncClient) -> None:
    body = await upload(sclient, zip_bytes({"My Song/session.scraft": b"{}", "My Song/Audio/kick.wav": b"x"}), "song.zip")
    assert body["kind"] == "session"
    download = await sclient.get(f"/v1/assets/{body['asset_id']}")
    assert download.headers["content-type"] == "application/zip"


@pytest.mark.parametrize(
    "members",
    [
        pytest.param(["../evil.scraft"], id="parent traversal"),
        pytest.param(["a/../../evil.scraft"], id="nested traversal"),
        pytest.param(["/etc/session.scraft"], id="absolute path"),
        pytest.param(["\\windows\\session.scraft"], id="backslash absolute path"),
        pytest.param(["C:/session.scraft"], id="drive letter"),
        pytest.param(["song/session.scraft", "../x.txt"], id="one good member and one hostile"),
        pytest.param(["a.scraft", "b.scraft"], id="two sessions"),
        pytest.param(["notes.txt", "audio.wav"], id="no session"),
        pytest.param([], id="empty archive"),
        pytest.param(["session.scraft/"], id="a directory named like a session"),
    ],
)
async def test_hostile_or_malformed_session_zips_are_rejected(sclient: httpx.AsyncClient, members: list[str]) -> None:
    assert_envelope(await post_file(sclient, zip_bytes(members), "song.zip"), 400, "invalid_upload")


async def test_zip_member_count_is_capped(sclient: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "MAX_ZIP_ENTRIES", 5)
    assert (await post_file(sclient, zip_bytes(["s.scraft", "1", "2", "3", "4"]), "s.zip")).status_code == 200
    assert_envelope(await post_file(sclient, zip_bytes(["s.scraft", "1", "2", "3", "4", "5"]), "s.zip"), 400, "invalid_upload")


async def test_zip_uncompressed_size_is_capped(sclient: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "MAX_ZIP_UNCOMPRESSED", 1000)
    assert_envelope(await post_file(sclient, zip_bytes({"s.scraft": b"{}", "big.bin": b"\0" * 5000}), "s.zip"), 400, "invalid_upload")


async def test_corrupt_zip_is_rejected(sclient: httpx.AsyncClient) -> None:
    good = zip_bytes(["s.scraft"])
    assert_envelope(await post_file(sclient, good[: len(good) // 2], "s.zip"), 400, "invalid_upload")
    assert_envelope(await post_file(sclient, b"PK\x03\x04" + b"\xff" * 100, "s.zip"), 400, "invalid_upload")
