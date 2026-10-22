"""The ``forge`` command: check, models, purge and serve."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI

from audiokit import wav
from forge.cli import main, make_server, run_servers
from forge.store import AssetStore

from fapi_helpers import AUTH, TOKEN, TOKEN_ENV, config_data, free_port, tone


def write_config(tmp_path: Path, **overrides: Any) -> Path:
    path = tmp_path / "forge.yaml"
    path.write_text(yaml.safe_dump(config_data(tmp_path, **overrides)))
    return path


def run_main(*argv: str) -> int:
    return main(list(argv))


# ---- check ----


def test_check_passes_on_testtone_and_yin(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = write_config(tmp_path)
    assert run_main("check", "--config", str(cfg)) == 0
    out = capsys.readouterr().out
    for model in ("testtone", "yin", "analysis"):
        assert model in out
    assert "AVAILABLE" in out and "test generator" in out
    assert "ok    testtone (generate.sfx)" in out and "ok    yin (transcribe.melody)" in out and "ok    analysis (analyze.audio)" in out
    assert out.rstrip().endswith("check passed") and "FAIL" not in out
    assert not (tmp_path / "data" / "index.sqlite").exists()  # ran on a scratch directory


def test_check_can_be_limited_to_some_models(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = write_config(tmp_path)
    assert run_main("check", "--config", str(cfg), "--model", "yin") == 0
    out = capsys.readouterr().out
    assert "ok    yin" in out and "testtone (" not in out


def test_check_fails_for_an_unknown_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = write_config(tmp_path)
    assert run_main("check", "--config", str(cfg), "--model", "nope") == 1
    assert "FAIL  nope: no such model" in capsys.readouterr().out


def test_check_fails_when_an_enabled_family_cannot_start(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    families = {
        "testtone": {"python": sys.executable},
        "yin": {"python": sys.executable},
        "foundation": {"enabled": True, "python": str(tmp_path / "no-such-python")},
    }
    cfg = write_config(tmp_path, families=families)
    assert run_main("check", "--config", str(cfg)) == 1
    out = capsys.readouterr().out
    assert "FAIL  family foundation is down" in out and "check failed" in out
    assert "ok    testtone" in out  # the healthy families were still checked


def test_check_reports_unavailable_models_without_failing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = tmp_path / "fakefam.py"
    script.write_text(
        "from forge.workers.protocol import ModelInfo\n"
        "from forge.workers.runtime import Family, main\n"
        "class F(Family):\n"
        "    name = 'foundation'\n"
        "    def models(self):\n"
        "        return [{'id': 'foundation-1', 'family': 'foundation', 'job_types': ['generate.loop'], 'available': False,\n"
        "                 'reason': 'missing package stable_audio_tools', 'licence': 'x', 'checkpoint': '', 'vram_gb': 7.0}]\n"
        "    def run(self, job, progress, cancelled):\n"
        "        raise RuntimeError('never runs')\n"
        "main(F())\n"
    )
    wrapper = tmp_path / "py.sh"
    wrapper.write_text(f"#!/bin/sh\nif [ \"$2\" = forge.workers.foundation ]; then exec {sys.executable} {script}; fi\nexec {sys.executable} \"$@\"\n")
    wrapper.chmod(0o755)
    families = {"testtone": {"python": sys.executable}, "yin": {"python": sys.executable}, "foundation": {"enabled": True, "python": str(wrapper)}}
    assert run_main("check", "--config", str(write_config(tmp_path, families=families))) == 0
    out = capsys.readouterr().out
    assert "unavailable: missing package stable_audio_tools" in out and "foundation-1" in out and out.rstrip().endswith("check passed")


# ---- models ----


def test_models_prints_a_table_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = write_config(tmp_path)
    assert run_main("models", "--config", str(cfg)) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["FAMILY", "MODEL", "AVAILABLE", "VRAM_GB", "LICENCE", "/", "REASON"]
    assert {line.split()[1] for line in out.splitlines()[1:]} == {"analysis", "testtone", "yin"}
    assert run_main("models", "--config", str(cfg), "--json") == 0
    listed = json.loads(capsys.readouterr().out)
    assert {m["id"] for m in listed} == {"analysis", "testtone", "yin"} and all("job_types" in m for m in listed)


def test_models_exits_non_zero_when_a_family_is_down(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    families = {"testtone": {"python": sys.executable}, "stems": {"enabled": True, "python": str(tmp_path / "missing")}}
    assert run_main("models", "--config", str(write_config(tmp_path, families=families))) == 1
    assert "family stems is down" in capsys.readouterr().err


# ---- purge ----


def test_purge_removes_old_unrated_assets_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg_path = write_config(tmp_path, retention_days=30)
    store = AssetStore(tmp_path / "data")
    old = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ids = {}
    for name, freq, created in (("old", 300.0, old), ("old_rated", 400.0, old), ("fresh", 500.0, None)):
        src = tmp_path / f"{name}.wav"
        wav.write(src, tone(freq, 0.2), 44100, "16")
        side = {"type": "upload", "kind": "audio", "prompt": name, "tags": {}, **({"created_at": created} if created else {})}
        ids[name] = store.add_asset(src, side)["asset_id"]
    store.rate(ids["old_rated"], 5)
    store.close()
    assert run_main("purge", "--config", str(cfg_path), "--days", "100000") == 0
    assert capsys.readouterr().out.strip() == "purged 0 asset(s)"
    assert run_main("purge", "--config", str(cfg_path)) == 0
    assert capsys.readouterr().out.strip() == "purged 1 asset(s)"
    store = AssetStore(tmp_path / "data")
    try:
        assert store.exists(ids["old_rated"]) and store.exists(ids["fresh"]) and not store.exists(ids["old"])
    finally:
        store.close()


# ---- serve ----


def test_serve_refuses_to_start_without_a_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = write_config(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, "")
    assert run_main("serve", "--config", str(cfg)) == 2
    assert TOKEN_ENV in capsys.readouterr().err
    monkeypatch.delenv(TOKEN_ENV)
    assert run_main("serve", "--config", str(cfg)) == 2
    assert not (tmp_path / "data").exists()  # nothing was started or created


def test_serve_refuses_no_auth_on_a_public_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    cfg = write_config(tmp_path, insecure_no_auth=True, listen={"host": "0.0.0.0", "port": free_port()})
    assert run_main("serve", "--config", str(cfg)) == 2
    assert "loopback" in capsys.readouterr().err


def test_bad_configuration_and_usage_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("listen: {port: 99999}\n")
    assert run_main("check", "--config", str(bad)) == 2
    assert "listen.port" in capsys.readouterr().err
    assert run_main("models", "--config", str(tmp_path / "missing.yaml")) == 2
    with pytest.raises(SystemExit) as caught:
        run_main("frobnicate")
    assert caught.value.code == 2


def wait_for_health(base: str, proc: subprocess.Popen[bytes], timeout: float = 40.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert proc.poll() is None, f"forge serve exited early with {proc.returncode}"
        try:
            if httpx.get(f"{base}/v1/health", timeout=2).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.1)
    raise AssertionError("forge serve did not become healthy")


def test_serve_end_to_end_with_the_gary_shim(tmp_path: Path) -> None:
    port, shim_port, sa3_port = free_port(), free_port(), free_port()
    cfg = write_config(
        tmp_path, listen={"host": "127.0.0.1", "port": port},
        gary_shim={"enabled": True, "ports": {"foundation": shim_port, "sa3": sa3_port}},
    )
    log = tmp_path / "serve.log"
    env = {**os.environ, TOKEN_ENV: TOKEN}
    with open(log, "wb") as sink:
        proc = subprocess.Popen([sys.executable, "-m", "forge", "serve", "--config", str(cfg)], env=env, stdout=sink, stderr=sink)
    base = f"http://127.0.0.1:{port}"
    try:
        wait_for_health(base, proc)
        assert httpx.get(f"{base}/v1/models").status_code == 401
        models = httpx.get(f"{base}/v1/models", headers=AUTH).json()
        assert {"testtone", "yin", "analysis"} <= {m["id"] for m in models}
        job_id = httpx.post(f"{base}/v1/jobs", headers=AUTH, json={"type": "generate.sfx", "params": {"duration_s": 0.3}}).json()["job_id"]
        for _ in range(300):
            job = httpx.get(f"{base}/v1/jobs/{job_id}", headers=AUTH).json()
            if job["state"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        assert job["state"] == "succeeded"
        asset = httpx.get(f"{base}/v1/assets/{job['result']['asset_ids'][0]}", headers={**AUTH, "Range": "bytes=0-3"})
        assert asset.status_code == 206 and asset.content == b"RIFF"
        for shim, name in ((shim_port, "foundation"), (sa3_port, "sa3")):
            assert httpx.get(f"http://127.0.0.1:{shim}/health").json() == {"status": "ok", "service": name}
        no_family = httpx.post(f"http://127.0.0.1:{shim_port}/generate", json={"bars": 1})
        assert no_family.status_code == 503 and no_family.json()["success"] is False  # no foundation family configured
        stream = socket.create_connection(("127.0.0.1", port), timeout=10)  # an event stream left open while the server stops
        stream.sendall(f"GET /v1/events HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {TOKEN}\r\n\r\n".encode())
        assert b"200 OK" in stream.recv(4096)
    finally:
        stopped = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        try:
            code = proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
    assert code == 0, log.read_text(errors="replace")[-3000:]
    assert time.monotonic() - stopped < 4.5, "open event streams must not hold up shutdown"
    stream.close()


def test_python_dash_m_forge_is_the_cli() -> None:
    done = subprocess.run([sys.executable, "-m", "forge", "--help"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and all(word in done.stdout for word in ("serve", "check", "purge", "models"))
    assert subprocess.run([sys.executable, "-m", "forge", "nope"], capture_output=True, timeout=60).returncode == 2


async def test_a_server_that_cannot_bind_makes_run_servers_return_1() -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        server = make_server(FastAPI(), "127.0.0.1", port, log_level="critical")
        assert await asyncio.wait_for(run_servers([server], handle_signals=False), 20) == 1
