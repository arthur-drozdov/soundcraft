"""Helpers shared by the Forge HTTP tests. Audio is synthesised here, never committed."""

from __future__ import annotations

import asyncio
import io
import socket
import sys
import zipfile
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from audiokit import wav
from forge.config import ForgeConfig, parse_config

TOKEN = "test-token-4f9c1d2e8a7b"
TOKEN_ENV = "FORGE_API_TEST_TOKEN"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SR = 44100
SLOW = {"x-delay_s": "30"}
SHARED_LOOP = pytest.mark.asyncio(loop_scope="module")  # for modules that use the shared (s-prefixed) fixtures


def config_data(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "data_dir": str(tmp_path / "data"),
        "token_env": TOKEN_ENV,
        "families": {
            "testtone": {"python": sys.executable, "lane": "gpu"},
            "yin": {"python": sys.executable, "lane": "cpu"},
        },
    }
    data.update(overrides)
    return data


def make_config(tmp_path: Path, **overrides: Any) -> ForgeConfig:
    return parse_config(config_data(tmp_path, **overrides))


def tone(freq: float = 440.0, seconds: float = 1.0, sr: int = SR, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    env = np.minimum(1.0, np.minimum(t / 0.01, (seconds - t) / 0.02))
    return (amp * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)[None, :]


def melody(notes: list[int], note_s: float = 0.4, gap_s: float = 0.1) -> np.ndarray:
    parts: list[np.ndarray] = []
    for n in notes:
        parts += [tone(440.0 * 2 ** ((n - 69) / 12), note_s), np.zeros((1, int(gap_s * SR)), np.float32)]
    return np.concatenate(parts, axis=1)


def wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.wav"
        wav.write(path, audio, sr, "16")
        return path.read_bytes()


def zip_bytes(names: list[str] | dict[str, bytes]) -> bytes:
    members = names if isinstance(names, dict) else {n: b"x" for n in names}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buf.getvalue()


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.wav"
        path.write_bytes(data)
        return wav.read(path)


async def upload(client: httpx.AsyncClient, data: bytes, name: str = "up.wav") -> dict[str, Any]:
    resp = await client.post("/v1/uploads", files={"file": (name, data, "application/octet-stream")})
    assert resp.status_code == 200, resp.text
    return resp.json()  # type: ignore[no-any-return]


async def submit(client: httpx.AsyncClient, job_type: str, params: dict[str, Any] | None = None, **body: Any) -> str:
    resp = await client.post("/v1/jobs", json={"type": job_type, "params": params or {}, **body})
    assert resp.status_code == 202, resp.text
    return str(resp.json()["job_id"])


async def wait_job(client: httpx.AsyncClient, job_id: str, state: str | None = None, timeout: float = 30.0) -> dict[str, Any]:
    """Poll until the job is in ``state`` (default: any final state)."""

    async def poll() -> dict[str, Any]:
        while True:
            job = (await client.get(f"/v1/jobs/{job_id}")).json()
            if (job["state"] == state) if state else job["state"] in ("succeeded", "failed", "cancelled"):
                return job  # type: ignore[no-any-return]
            await asyncio.sleep(0.02)

    return await asyncio.wait_for(poll(), timeout)


async def run_job(client: httpx.AsyncClient, job_type: str, params: dict[str, Any] | None = None, **body: Any) -> dict[str, Any]:
    job = await wait_job(client, await submit(client, job_type, params, **body))
    assert job["state"] == "succeeded", job
    return job


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def assert_envelope(resp: httpx.Response, status: int, code: str) -> dict[str, Any]:
    """Check the ``{"error": {code, message, details?}}`` envelope and return its inner object."""
    assert resp.status_code == status, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert set(body) == {"error"} and body["error"]["code"] == code and isinstance(body["error"]["message"], str), body
    return body["error"]  # type: ignore[no-any-return]


async def write_input(service: Any, tmp_path: Path, audio: np.ndarray, name: str = "in.wav") -> str:
    """Register ``audio`` with the service as an uploaded asset and return its id."""
    path = tmp_path / name
    wav.write(path, audio, SR, "16")
    return str((await service.upload(path, name))["asset_id"])
