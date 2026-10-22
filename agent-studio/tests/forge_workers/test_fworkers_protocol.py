"""Cross-family checks: nothing heavy at import, real ``python -m forge.workers.<family>`` processes, the manager."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from forge.config import parse_config
from forge.manager import WorkerManager
from forge.workers import acestep, basicpitch, foundation, render, runtime, sa3, stems
from forge.workers.protocol import validate_model_info

from fworkers_fakes import Harness, hello_models, in_memory

FAMILIES = {
    "foundation": foundation.FoundationFamily, "sa3": sa3.SA3Family, "acestep": acestep.AceStepFamily,
    "basicpitch": basicpitch.BasicPitchFamily, "stems": stems.StemsFamily, "render": render.RenderFamily,
}
HEAVY = ("torch", "torchaudio", "tensorflow", "stable_audio_tools", "stable_audio_3", "acestep", "basic_pitch", "onnxruntime")


def test_importing_the_workers_imports_no_model_package() -> None:
    code = (
        "import sys\n"
        "import forge.workers.foundation, forge.workers.sa3, forge.workers.acestep, forge.workers.basicpitch\n"
        "import forge.workers.stems, forge.workers.render, forge.workers._prompts, forge.workers._upstream\n"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('clean')\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert done.stdout.strip() == "clean", done.stderr


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_family_names_match_their_module_and_models_are_valid(name: str) -> None:
    family = FAMILIES[name]()
    assert family.name == name
    family.configure({})
    infos = family.models()
    assert infos and all(validate_model_info(i, name) for i in infos)
    for info in infos:
        assert info["licence"] and info["job_types"] and info["vram_gb"] >= 0
        if not info["available"]:
            assert info["reason"]


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_protocol_basics_without_the_real_packages(name: str) -> None:
    """hello answers with every model unavailable and a reason; bad requests get errors, not crashes."""
    models = hello_models(FAMILIES[name](), {})
    for info in models.values():
        assert info["available"] is False and info["reason"]
    with Harness(FAMILIES[name]()) as h:
        h.call("hello", options={})
        bad_load = h.call("load", model="no-such-model")[1]
        assert bad_load["ok"] is False and "unknown model" in bad_load["error"]
        assert h.call("load")[1]["ok"] is False  # model must be a string
        first = next(iter(models))
        loaded = h.call("load", model=first)[1]
        assert loaded["ok"] is False and loaded["error"]  # cannot load what is not installed: an error reply, not a dead worker
        job = {"job_id": "j", "type": "generate.sfx", "model": first, "params": {}, "inputs": {}, "out_dir": "/nonexistent", "seed": 1}
        failed = h.call("run", job=job)[1]
        assert failed["ok"] is False and failed["error"]
        assert h.call("hello", options={})[1]["ok"]  # still alive
        assert h.call("hello", options="nonsense")[1]["ok"] is False


def test_hello_options_of_the_wrong_type_never_kill_the_worker() -> None:
    for name, family in FAMILIES.items():
        replies = in_memory([{"id": 1, "op": "hello", "options": {"checkpoints": 5, "device": 7, "binary": 3, "model": [], "lm": {}, "timeout_s": "x"}}], family())
        assert replies[0]["ok"] is True, (name, replies)
        for info in replies[0]["result"]["models"]:
            assert info["available"] is False


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_real_worker_process_answers_hello(name: str, tmp_path: Path) -> None:
    """``python -m forge.workers.<family>`` as the gateway starts it: JSON lines on stdout, logs on stderr."""
    proc = subprocess.Popen(
        [sys.executable, "-m", f"forge.workers.{name}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=tmp_path,
    )
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps({"id": 1, "op": "hello", "options": {}}) + "\n")
        proc.stdin.flush()
        reply = json.loads(proc.stdout.readline())
        assert reply["ok"] and reply["result"]["family"] == name and reply["result"]["protocol"] == 1
        for info in reply["result"]["models"]:
            validate_model_info(info, name)
        proc.stdin.write('{"id":2,"op":"shutdown"}\n')
        proc.stdin.flush()
        assert proc.wait(timeout=30) == 0
        assert proc.stdout.read() == ""  # nothing but protocol lines ever reached stdout
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()


async def test_the_manager_starts_every_family_and_reports_why_models_are_unavailable(tmp_path: Path) -> None:
    families: dict[str, Any] = {name: {"python": sys.executable, "enabled": True} for name in FAMILIES}
    cfg = parse_config({"data_dir": str(tmp_path / "data"), "insecure_no_auth": True, "families": families})
    manager = WorkerManager(cfg, hello_timeout_s=60.0)
    await manager.start()
    try:
        status = manager.family_status()
        assert set(FAMILIES) <= set(status)  # plus the always-on testtone and yin
        assert all(status[name]["state"] == "ready" for name in FAMILIES)
        models = {m["id"]: m for m in manager.models()}
        assert {"foundation-1", "foundation-1.2-samples", "foundation-1.2-keybeds", "sa3-small-music", "sa3-small-sfx", "sa3-medium",
                "ace-step-1.5-base", "ace-step-1.5-xl-base", "basic-pitch", "stems-cpp", "soundcraft"} <= set(models)
        assert all(not m["available"] and m["reason"] for m in models.values() if m["family"] in FAMILIES)
        assert models["soundcraft"]["lane"] == "cpu" and models["stems-cpp"]["lane"] == "cpu" and models["basic-pitch"]["lane"] == "cpu"
        assert models["sa3-medium"]["lane"] == "gpu" and models["foundation-1"]["lane"] == "gpu"
        from forge.errors import Conflict

        with pytest.raises(Conflict, match="unavailable"):
            manager.resolve_model("render.session", "soundcraft")
    finally:
        await manager.close()


def test_every_family_module_has_a_main_guard() -> None:
    for name in FAMILIES:
        source = (Path(__file__).parents[2] / "src" / "forge" / "workers" / f"{name}.py").read_text()
        assert 'if __name__ == "__main__":' in source and "main(" in source
    assert runtime.main  # the shared entry point used by all of them
