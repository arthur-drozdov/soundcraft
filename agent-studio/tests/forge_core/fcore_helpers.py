"""Helpers shared by the Forge core tests. Audio is synthesised here, never committed."""

from __future__ import annotations

import asyncio
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from audiokit import wav
from forge.config import ForgeConfig, parse_config
from forge.jobs import JobRecord
from forge.manager import WorkerManager
from forge.service import ForgeService

SR = 44100


def make_config(tmp_path: Path, **overrides: Any) -> ForgeConfig:
    data: dict[str, Any] = {
        "data_dir": str(tmp_path / "data"),
        "insecure_no_auth": True,
        "families": {
            "testtone": {"python": sys.executable, "lane": "gpu"},
            "yin": {"python": sys.executable, "lane": "cpu"},
        },
    }
    data.update(overrides)
    return parse_config(data)


def make_service(cfg: ForgeConfig, **manager_kwargs: Any) -> ForgeService:
    kwargs: dict[str, Any] = {"backoff_base_s": 0.05, "backoff_max_s": 0.2, "cancel_grace_s": 3.0}
    kwargs.update(manager_kwargs)
    return ForgeService(cfg, manager=WorkerManager(cfg, **kwargs), purge_interval_s=None)


async def wait_for(svc: ForgeService, job_id: str, timeout: float = 30.0) -> JobRecord:
    """Poll until the job reaches a final state."""
    async def poll() -> JobRecord:
        while True:
            rec = svc.get_job(job_id)
            if rec.state in ("succeeded", "failed", "cancelled"):
                return rec
            await asyncio.sleep(0.02)

    return await asyncio.wait_for(poll(), timeout)


async def wait_state(svc: ForgeService, job_id: str, state: str, timeout: float = 15.0) -> JobRecord:
    async def poll() -> JobRecord:
        while svc.get_job(job_id).state != state:
            await asyncio.sleep(0.02)
        return svc.get_job(job_id)

    return await asyncio.wait_for(poll(), timeout)


def tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    y = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    env = np.minimum(1.0, np.minimum(t / 0.01, (seconds - t) / 0.02))
    return (amp * y * env / 1.3).astype(np.float32)


def melody(notes: list[int], note_s: float = 0.4, gap_s: float = 0.1, sr: int = SR) -> np.ndarray:
    parts: list[np.ndarray] = []
    for n in notes:
        parts += [tone(440.0 * 2 ** ((n - 69) / 12), note_s, sr), np.zeros(int(gap_s * sr), np.float32)]
    return np.concatenate(parts)[None, :]


def clicks(bpm: float, seconds: float, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(3)
    out = rng.standard_normal(int(seconds * sr)) * 10 ** (-50 / 20)
    blip_t = np.arange(int(0.05 * sr)) / sr
    blip = np.sin(2 * np.pi * 1200 * blip_t) * np.exp(-blip_t / 0.008)
    t = 0.1
    while t + 0.05 < seconds:
        i = int(round(t * sr))
        out[i : i + blip.size] += 0.8 * blip
        t += 60.0 / bpm
    return out.astype(np.float32)[None, :]


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> Path:
    wav.write(path, audio, sr, "16")
    return path


def make_zip(path: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"x")
    return path
