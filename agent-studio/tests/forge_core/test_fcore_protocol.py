"""The worker protocol: the runtime loop in-process, and the real testtone / yin subprocesses."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from audiokit import midi, wav
from forge.workers import runtime
from forge.workers.protocol import ProtocolError, validate_model_info, validate_outputs
from forge.workers.runtime import Family, check_cancel

from fcore_helpers import SR, melody, write_wav


class Echo(Family):
    name = "echo"

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.options: dict[str, Any] = {}

    def configure(self, options: dict[str, Any]) -> None:
        self.options = options

    def models(self) -> list[Any]:
        return [{"id": "echo", "family": "echo", "job_types": ["generate.sfx"], "available": True, "licence": "x", "checkpoint": "c", "vram_gb": 1.0}]

    def load(self, model: str) -> None:
        if model == "bad":
            raise RuntimeError("no such model")
        self.loaded.append(model)

    def run(self, job: Any, progress: Any, cancelled: Any) -> list[Any]:
        progress(0.25, "quarter")
        if job["params"].get("fail"):
            raise ValueError("boom")
        if job["params"].get("nan"):
            return [{"path": "x", "kind": "audio", "candidate": float("nan"), "extra": {}}]
        while job["params"].get("wait") and not cancelled():
            time.sleep(0.01)
        check_cancel(cancelled)
        return [{"path": str(Path(job["out_dir"]) / "o.wav"), "kind": "audio", "candidate": 0, "extra": {}}]


def converse(family: Family, lines: list[str]) -> list[dict[str, Any]]:
    out = io.StringIO()
    code = runtime.serve(family, io.StringIO("\n".join(lines) + "\n"), out)
    assert code == 0
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_runtime_hello_load_run_roundtrip() -> None:
    fam = Echo()
    job = {"job_id": "j1", "type": "generate.sfx", "model": "echo", "params": {}, "inputs": {}, "out_dir": "/o", "seed": 1}
    replies = converse(fam, [
        json.dumps({"id": 1, "op": "hello", "options": {"device": "cpu"}}),
        json.dumps({"id": 2, "op": "load", "model": "echo"}),
        json.dumps({"id": 3, "op": "run", "job": job}),
        json.dumps({"id": 4, "op": "unload", "model": "echo"}),
    ])
    # the run thread may finish after the unload request is read: look replies up by id
    by_id = {r["id"]: r for r in replies if "event" not in r}
    assert by_id[1]["result"]["protocol"] == 1 and by_id[1]["result"]["family"] == "echo"
    assert by_id[1]["result"]["models"][0]["id"] == "echo" and fam.options == {"device": "cpu"}
    assert by_id[2] == {"id": 2, "ok": True} and fam.loaded == ["echo"]
    events = [r for r in replies if r.get("event") == "progress"]
    assert events and events[0] == {"id": 3, "event": "progress", "progress": 0.25, "message": "quarter"}


def test_runtime_errors_do_not_kill_the_loop() -> None:
    replies = converse(Echo(), [
        "this is not json",
        json.dumps([1, 2]),
        json.dumps({"id": "x", "op": "hello"}),
        json.dumps({"id": 1, "op": "bogus"}),
        json.dumps({"id": 2, "op": "load", "model": "bad"}),
        json.dumps({"id": 3, "op": "load"}),
        json.dumps({"id": 4, "op": "hello", "options": 5}),
        json.dumps({"id": 5, "op": "run", "job": "nope"}),
        json.dumps({"id": 6, "op": "hello"}),
    ])
    assert replies[0]["error"] == "invalid JSON" and replies[0]["id"] is None
    assert replies[1]["ok"] is False and replies[2]["ok"] is False
    assert "unknown op" in replies[3]["error"] and replies[3]["id"] == 1
    assert replies[4]["ok"] is False and "no such model" in replies[4]["error"]
    assert all(r["ok"] is False for r in replies[5:8])
    assert replies[8]["ok"] is True  # still serving


class Gated(io.StringIO):
    """Stdin that serves its lines, then waits (up to 5 s) for ``release`` before reporting end of input."""

    def __init__(self, text: str, release: threading.Event) -> None:
        super().__init__(text)
        self.release = release

    def readline(self, size: int = -1) -> str:
        line = super().readline(size)
        if not line:
            self.release.wait(5.0)
        return line


class Collect(io.StringIO):
    """Stdout that fires ``release`` once the final reply (the one with ``ok``) to request 1 is written."""

    def __init__(self, release: threading.Event) -> None:
        super().__init__()
        self.release = release

    def write(self, text: str) -> int:
        count = super().write(text)
        if '"id":1,' in text and '"ok"' in text:
            self.release.set()
        return count


def run_one(job_params: dict[str, Any]) -> dict[str, Any]:
    release = threading.Event()
    job = {"job_id": "j1", "type": "t", "model": "echo", "params": job_params, "inputs": {}, "out_dir": "/o", "seed": 1}
    out = Collect(release)
    runtime.serve(Echo(), Gated(json.dumps({"id": 1, "op": "run", "job": job}) + "\n", release), out)
    return [json.loads(line) for line in out.getvalue().splitlines() if '"ok"' in line][-1]  # type: ignore[no-any-return]


def test_runtime_job_failures_are_replies() -> None:
    assert run_one({"fail": True}) == {"id": 1, "ok": False, "error": "ValueError: boom"}


def test_runtime_unserialisable_result_is_an_error_reply() -> None:
    final = run_one({"nan": True})
    assert final["ok"] is False and "unserialisable" in final["error"]


class Worker:
    """A real worker subprocess driven synchronously."""

    def __init__(self, family: str) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", f"forge.workers.{family}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
        )
        self._id = 0

    def send(self, op: str, **fields: Any) -> int:
        self._id += 1
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"id": self._id, "op": op, **fields}) + "\n")
        self.proc.stdin.flush()
        return self._id

    def read(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        return json.loads(self.proc.stdout.readline())  # type: ignore[no-any-return]

    def call(self, op: str, **fields: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        req = self.send(op, **fields)
        events = []
        while True:
            msg = self.read()
            if msg.get("id") == req and "event" not in msg:
                return msg, events
            events.append(msg)

    def close(self) -> int:
        self.send("shutdown")
        return self.proc.wait(10)


def run_job(w: Worker, tmp_path: Path, job_type: str, params: dict[str, Any], inputs: dict[str, str] | None = None, seed: int = 7, model: str = "testtone"):  # type: ignore[no-untyped-def]
    base = {"prompt": "", "tags": {}, "seed": seed, "candidates": 1}
    base.update(params)
    return w.call("run", job={"job_id": "j1", "type": job_type, "model": model, "params": base, "inputs": inputs or {}, "out_dir": str(tmp_path), "seed": seed})


@pytest.fixture
def testtone() -> Any:
    w = Worker("testtone")
    reply, _ = w.call("hello", options={})
    assert reply["ok"]
    yield w
    w.close()


def test_testtone_subprocess_roundtrip(testtone: Worker, tmp_path: Path) -> None:
    models = testtone.call("hello", options={})[0]["result"]["models"]
    assert [m["id"] for m in models] == ["testtone"] and models[0]["test_generator"] is True and models[0]["available"]
    assert set(models[0]["job_types"]) == {
        "generate.loop", "generate.sample", "generate.instrument", "generate.sfx", "transform.audio", "extend.audio", "inpaint.audio", "layer.add",
    }
    assert models[0]["licence"] and models[0]["checkpoint_hash"]
    assert testtone.call("load", model="testtone")[0] == {"id": 3, "ok": True}
    reply, events = run_job(testtone, tmp_path, "generate.sample", {"note": 60, "length_s": 0.5})
    assert reply["ok"] and reply["result"]["outputs"][0]["kind"] == "audio"
    assert events and all(e["event"] == "progress" and 0 <= e["progress"] <= 1 for e in events)
    audio, sr = wav.read(reply["result"]["outputs"][0]["path"])
    assert sr == 44100 and audio.shape == (2, 22050) and abs(audio).max() < 1.0


def test_testtone_requires_load_and_survives_bad_requests(testtone: Worker, tmp_path: Path) -> None:
    reply, _ = run_job(testtone, tmp_path, "generate.sample", {"note": 60, "length_s": 0.1})
    assert reply["ok"] is False and "not loaded" in reply["error"]
    assert testtone.call("bogus")[0]["ok"] is False
    testtone.proc.stdin.write("garbage\n")  # type: ignore[union-attr]
    testtone.proc.stdin.flush()  # type: ignore[union-attr]
    assert testtone.read()["error"] == "invalid JSON"
    assert testtone.call("hello", options={})[0]["ok"]


def test_testtone_is_deterministic_and_seed_driven(testtone: Worker, tmp_path: Path) -> None:
    testtone.call("load", model="testtone")
    params = {"bpm": 120, "bars": 1, "key": "A minor"}
    paths = []
    for i, seed in enumerate((5, 5, 6)):
        (tmp_path / str(i)).mkdir()
        reply, _ = run_job(testtone, tmp_path / str(i), "generate.loop", params, seed=seed)
        paths.append(Path(reply["result"]["outputs"][0]["path"]).read_bytes())
    assert paths[0] == paths[1] and paths[0] != paths[2]


def test_testtone_candidates_use_consecutive_seeds(testtone: Worker, tmp_path: Path) -> None:
    testtone.call("load", model="testtone")
    reply, events = run_job(testtone, tmp_path, "generate.sfx", {"duration_s": 0.3, "candidates": 3})
    outs = reply["result"]["outputs"]
    assert [o["candidate"] for o in outs] == [0, 1, 2]
    assert len({Path(o["path"]).read_bytes() for o in outs}) == 3
    assert events[-1]["progress"] == 1.0


def test_testtone_every_job_type(testtone: Worker, tmp_path: Path) -> None:
    testtone.call("load", model="testtone")
    src = write_wav(tmp_path / "in.wav", melody([60, 62], 0.5))
    inputs = {"audio": str(src)}
    cases = [
        ("generate.loop", {"bpm": 90, "bars": 1}, None), ("generate.sample", {"note": 48, "length_s": 0.3}, None),
        ("generate.sfx", {"duration_s": 0.3}, None), ("transform.audio", {"strength": 0.8}, inputs),
        ("extend.audio", {"seconds": 1}, inputs), ("inpaint.audio", {"start_s": 0.2, "end_s": 0.6}, inputs),
        ("layer.add", {"description": "airy pad"}, inputs),
    ]
    for i, (job_type, params, ins) in enumerate(cases):
        out = tmp_path / f"o{i}"
        out.mkdir()
        reply, _ = run_job(testtone, out, job_type, params, ins)
        assert reply["ok"], (job_type, reply)
        audio, sr = wav.read(reply["result"]["outputs"][0]["path"])
        assert audio.shape[1] > 0
    original, _ = wav.read(src)
    ext, _ = wav.read(run_job(testtone, tmp_path, "extend.audio", {"seconds": 2}, inputs)[0]["result"]["outputs"][0]["path"])
    assert ext.shape[1] == original.shape[1] + 2 * SR
    bad, _ = run_job(testtone, tmp_path, "transform.audio", {"strength": 0.5}, None)
    assert bad["ok"] is False and "audio input" in bad["error"]
    outside, _ = run_job(testtone, tmp_path, "inpaint.audio", {"start_s": 50, "end_s": 60}, inputs)
    assert outside["ok"] is False


def test_testtone_instrument_zones_at_the_right_pitch(testtone: Worker, tmp_path: Path) -> None:
    from forge.postprocess import measure_note

    testtone.call("load", model="testtone")
    params = {"note_low": 48, "note_high": 59, "step": 4, "velocity_layers": 2, "length_s": 0.5, "name": "Keys"}
    reply, events = run_job(testtone, tmp_path, "generate.instrument", params)
    out = reply["result"]["outputs"][0]
    assert out["kind"] == "instrument" and Path(out["path"]).is_dir()
    zones = out["extra"]["zones"]
    assert [(z["root_note"], z["key_low"], z["key_high"]) for z in zones[::2]] == [(49, 48, 51), (53, 52, 55), (57, 56, 59)]
    assert [(z["vel_low"], z["vel_high"]) for z in zones[:2]] == [(0, 63), (64, 127)]
    for z in zones:
        audio, sr = wav.read(Path(out["path"]) / z["sample"])
        assert measure_note(audio, sr) == pytest.approx(z["root_note"], abs=0.2)
    assert events[-1]["progress"] == 1.0


def test_testtone_faults_and_delay(testtone: Worker, tmp_path: Path) -> None:
    testtone.call("load", model="testtone")
    reply, _ = run_job(testtone, tmp_path, "generate.sfx", {"tags": {"x-fault": "error"}})
    assert reply["ok"] is False and "injected failure" in reply["error"]
    reply, _ = run_job(testtone, tmp_path, "generate.sfx", {"tags": {"x-fault": "unheard-of"}})
    assert reply["ok"] is False
    started = time.monotonic()
    reply, events = run_job(testtone, tmp_path, "generate.sfx", {"duration_s": 0.2, "tags": {"x-delay_s": "0.3"}})
    assert reply["ok"] and time.monotonic() - started >= 0.3 and len(events) > 3


def test_cancel_stops_a_running_job(testtone: Worker, tmp_path: Path) -> None:
    testtone.call("load", model="testtone")
    run_id = testtone.send("run", job={
        "job_id": "slow", "type": "generate.sfx", "model": "testtone", "inputs": {}, "out_dir": str(tmp_path), "seed": 1,
        "params": {"duration_s": 0.2, "candidates": 1, "tags": {"x-delay_s": "30"}},
    })
    assert testtone.read()["event"] == "progress"
    busy = testtone.call("hello", options={})[0]
    assert busy == {"id": run_id + 1, "ok": False, "error": "worker is busy"}
    assert testtone.call("cancel", job_id="someone-else")[0]["ok"]  # a different job is not affected
    started = time.monotonic()
    ack, _ = testtone.call("cancel", job_id="slow")
    assert ack["ok"]
    while True:
        msg = testtone.read()
        if msg.get("id") == run_id and "event" not in msg:
            break
    assert msg == {"id": run_id, "ok": False, "error": "cancelled"} and time.monotonic() - started < 2.0
    assert testtone.call("hello", options={})[0]["ok"]  # and it is idle again


def test_extra_models_option(tmp_path: Path) -> None:
    w = Worker("testtone")
    models = w.call("hello", options={"extra_models": [{"id": "tt-big", "vram_gb": 12.5}]})[0]["result"]["models"]
    assert {m["id"]: m["vram_gb"] for m in models} == {"testtone": 0.0, "tt-big": 12.5}
    assert w.close() == 0


def test_shutdown_and_eof_exit_cleanly() -> None:
    w = Worker("testtone")
    assert w.close() == 0
    w = Worker("yin")
    assert w.proc.stdin is not None
    w.proc.stdin.close()
    assert w.proc.wait(10) == 0


def test_stray_prints_cannot_corrupt_the_protocol(tmp_path: Path) -> None:
    script = tmp_path / "noisy.py"
    script.write_text(
        "import json, sys\nfrom forge.workers import runtime\nfrom forge.workers.testtone import TestToneFamily\n"
        "class Noisy(TestToneFamily):\n    def models(self):\n        print('library banner on stdout')\n        return super().models()\n"
        "runtime.main(Noisy())\n", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write('{"id":1,"op":"hello"}\n{"id":2,"op":"shutdown"}\n')
    proc.stdin.flush()
    assert json.loads(proc.stdout.readline())["ok"] is True
    assert proc.wait(10) == 0
    assert proc.stderr is not None and "library banner on stdout" in proc.stderr.read()


def test_yin_transcribes_a_melody(tmp_path: Path) -> None:
    w = Worker("yin")
    models = w.call("hello", options={})[0]["result"]["models"]
    assert models[0]["id"] == "yin" and models[0]["job_types"] == ["transcribe.melody"] and models[0]["vram_gb"] == 0
    w.call("load", model="yin")
    notes = [60, 64, 67, 72]
    src = write_wav(tmp_path / "m.wav", melody(notes, 0.4, 0.1))
    reply, _ = run_job(w, tmp_path, "transcribe.melody", {"min_note_ms": 60, "bpm": 100}, {"audio": str(src)}, model="yin")
    out = reply["result"]["outputs"][0]
    assert out["kind"] == "midi" and [n["pitch"] for n in out["extra"]["notes"]] == notes
    parsed = midi.read_smf(out["path"])
    assert [n.pitch for n in parsed] == notes
    assert out["extra"]["notes"][1]["start_s"] == pytest.approx(0.5, abs=0.06)
    wrong, _ = run_job(w, tmp_path, "generate.sfx", {}, model="yin")
    assert wrong["ok"] is False
    missing, _ = run_job(w, tmp_path, "transcribe.melody", {}, None, model="yin")
    assert missing["ok"] is False and "audio input" in missing["error"]
    assert w.close() == 0


def test_validate_model_info() -> None:
    good = {"id": "m", "family": "f", "job_types": ["generate.sfx"], "available": True, "licence": "L", "checkpoint": "c", "vram_gb": 2}
    assert validate_model_info(good, "f")["vram_gb"] == 2.0
    for patch in ({"family": "other"}, {"vram_gb": -1}, {"vram_gb": float("nan")}, {"vram_gb": True}, {"job_types": "x"}, {"id": ""}, {"id": 5}):
        with pytest.raises(ProtocolError):
            validate_model_info({**good, **patch}, "f")
    with pytest.raises(ProtocolError):
        validate_model_info("nope", "f")


def test_validate_outputs_confines_paths(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    inside = tmp_path / "out" / "a.wav"
    inside.write_bytes(b"x")
    outside = tmp_path / "secret.wav"
    outside.write_bytes(b"x")
    (tmp_path / "out" / "link.wav").symlink_to(outside)
    ok = validate_outputs([{"path": str(inside), "kind": "audio", "candidate": 2, "extra": {"a": 1}}], tmp_path / "out")
    assert ok[0]["candidate"] == 2
    bad: list[Any] = [
        {"path": str(outside), "kind": "audio"}, {"path": str(tmp_path / "out" / "link.wav"), "kind": "audio"},
        {"path": str(tmp_path / "out" / ".." / "secret.wav"), "kind": "audio"}, {"path": str(tmp_path / "out" / "nope.wav"), "kind": "audio"},
        {"path": str(inside), "kind": "exe"}, {"path": str(inside), "kind": "audio", "candidate": -1},
        {"path": str(inside), "kind": "audio", "candidate": True}, {"path": str(inside), "kind": "audio", "extra": []}, {"kind": "audio"}, "x",
    ]
    for item in bad:
        with pytest.raises(ProtocolError):
            validate_outputs([item], tmp_path / "out")
    with pytest.raises(ProtocolError):
        validate_outputs("nope", tmp_path / "out")
    with pytest.raises(ProtocolError):
        validate_outputs([{"path": str(inside), "kind": "audio"}] * 300, tmp_path / "out")
