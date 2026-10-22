"""The ``stems`` worker against a fake ``stems-split`` executable."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from audiokit import wav
from forge.workers import stems as stems_worker
from forge.workers import _upstream as up
from forge.workers.protocol import validate_model_info, validate_outputs
from forge.workers.runtime import Cancelled

from fworkers_fakes import Harness, inside, listing, make_job, make_wav, pid_alive, run_job, write_script

FAKE = """
    import json, os, sys, time, wave
    from pathlib import Path

    me = Path(sys.argv[0])
    args = sys.argv[1:]
    mode = me.with_suffix(".mode").read_text().strip() if me.with_suffix(".mode").exists() else "ok"
    if args == ["--version"]:
        if mode == "badversion":
            sys.stderr.write("illegal instruction\\n")
            sys.exit(4)
        print("stems-split 0.1-test")
        sys.exit(0)
    me.with_suffix(".argv").write_text(json.dumps(args))
    me.with_suffix(".pid").write_text(str(os.getpid()))
    opts = {}
    it = iter(args)
    for a in it:
        opts[a] = next(it) if a in ("--model", "--input", "--out", "--stems", "--device", "--threads", "--two-stems") else True
    out = Path(opts["--out"])
    if mode == "fail":
        sys.stderr.write("error: model file is corrupt\\n")
        sys.exit(1)
    if mode == "sleep":
        time.sleep(60)
    sources = ["drums", "bass", "other", "vocals"]
    if "--stems" in opts:
        sources = [s for s in sources if s in opts["--stems"].split(",")]
    if mode == "nostems":
        sys.exit(0)

    def write(path):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
            w.writeframes(b"\\x10\\x10" * 2205)

    out.mkdir(exist_ok=True)
    for i, name in enumerate(sources, 1):
        sys.stderr.write(f"\\r[stems] segment {i}/{len(sources)}")
        sys.stderr.flush()
        time.sleep(0.01)
        if mode == "empty":
            (out / f"{name}.wav").write_bytes(b"")
        else:
            write(out / f"{name}.wav")
        print(out / f"{name}.wav")
    if mode == "outside":
        os.symlink(out.parents[1] / "secret.wav", out / "link.wav")
    sys.stderr.write("\\n")
"""


def setup(tmp: Path, mode: str = "ok", **options: Any) -> tuple[stems_worker.StemsFamily, Path]:
    script = write_script(tmp / "bin" / "stems-split", FAKE)
    script.with_suffix(".mode").write_text(mode)
    model = tmp / "models" / "htdemucs.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"GGUF" + b"\0" * 100)
    family = stems_worker.StemsFamily()
    family.configure({"binary": str(script), "model": str(model), **options})
    return family, script


def argv_of(script: Path) -> list[str]:
    return json.loads(script.with_suffix(".argv").read_text())  # type: ignore[no-any-return]


def audio_input(tmp: Path) -> dict[str, str]:
    return {"audio": str(make_wav(tmp / "in" / "song.wav", seconds=1.0, sr=22050))}


# ---- availability ---------------------------------------------------------------------------------

def test_available_when_binary_and_model_exist(tmp_path: Path) -> None:
    family, script = setup(tmp_path)
    (info,) = family.models()
    assert info["id"] == "stems-cpp" and info["available"] is True and "reason" not in info
    assert info["checkpoint"] == str(tmp_path / "models" / "htdemucs.gguf") and info["checkpoint_hash"].startswith("fp:")
    assert info["job_types"] == ["stems.split"] and info["vram_gb"] == 0.0 and info["licence"].startswith("MIT")
    validate_model_info(info, "stems")


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, "options.binary is not set"),
        ({"binary": "/nonexistent/stems-split", "model": "/x"}, "does not exist"),
        ({"binary": "STEMS", "model": None}, "options.model is not set"),
        ({"binary": "STEMS", "model": "/nonexistent/model.gguf"}, "model file"),
        ({"binary": "STEMS", "model": "MODEL", "timeout_s": -1}, "invalid options"),
        ({"binary": "STEMS", "model": "MODEL", "device": "tpu"}, "invalid options"),
    ],
)
def test_unavailable_reasons(tmp_path: Path, options: dict[str, Any], expected: str) -> None:
    family, script = setup(tmp_path)
    fixed = {k: (str(script) if v == "STEMS" else str(tmp_path / "models" / "htdemucs.gguf") if v == "MODEL" else v) for k, v in options.items()}
    family.configure(fixed)
    (info,) = family.models()
    assert info["available"] is False and expected in info["reason"]
    validate_model_info(info, "stems")
    with pytest.raises(up.MissingDependency):
        family.load("stems-cpp")


def test_a_binary_that_cannot_run_is_unavailable(tmp_path: Path) -> None:
    family, _ = setup(tmp_path, mode="badversion")
    (info,) = family.models()
    assert not info["available"] and "exited with code 4" in info["reason"] and "illegal instruction" in info["reason"]


# ---- runs -----------------------------------------------------------------------------------------

def test_all_stems_by_default(tmp_path: Path) -> None:
    family, script = setup(tmp_path, device="cpu", threads=4)
    family.load("stems-cpp")
    inputs = audio_input(tmp_path)
    job = make_job(tmp_path, "stems.split", "stems-cpp", {}, inputs)
    outputs, events = run_job(family, job)
    out = (tmp_path / "out" / "stems").resolve()
    assert argv_of(script) == ["--model", str(tmp_path / "models" / "htdemucs.gguf"), "--input", inputs["audio"], "--out", str(out), "--device", "cpu", "--threads", "4"]
    assert [o["extra"]["stem"] for o in outputs] == ["bass", "drums", "other", "vocals"]
    assert all(o["kind"] == "audio" and o["candidate"] == 0 and inside(job["out_dir"], o["path"]) for o in outputs)
    audio, sr = wav.read(outputs[0]["path"])
    assert sr == 22050 and audio.shape[0] == 1
    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0 and any("separating" in m for _, m in events)
    validate_outputs(outputs, Path(job["out_dir"]))


def test_requested_stems_are_passed_and_checked(tmp_path: Path) -> None:
    family, script = setup(tmp_path)
    family.load("stems-cpp")
    inputs = audio_input(tmp_path)
    outputs, _ = run_job(family, make_job(tmp_path, "stems.split", "stems-cpp", {"stems": ["vocals", "drums", "vocals"]}, inputs))
    assert argv_of(script)[argv_of(script).index("--stems") + 1] == "vocals,drums"
    assert [o["extra"]["stem"] for o in outputs] == ["vocals", "drums"]  # in the order asked, once each
    with pytest.raises(ValueError, match="no stem\\(s\\) guitar; it produced vocals"):
        run_job(family, make_job(tmp_path / "b", "stems.split", "stems-cpp", {"stems": ["guitar", "vocals"]}, inputs))


def test_hostile_stem_names_never_reach_the_tool(tmp_path: Path) -> None:
    family, script = setup(tmp_path)
    family.load("stems-cpp")
    inputs = audio_input(tmp_path)
    for bad in (["../x"], ["a b"], ["vocals,drums"], ["x" * 40], [1], "vocals", ["a"] * 33, ["$(touch pwned)"]):
        with pytest.raises(ValueError, match="stems must be"):
            run_job(family, make_job(tmp_path, "stems.split", "stems-cpp", {"stems": bad}, inputs))
    assert not script.with_suffix(".argv").exists() and not (tmp_path / "pwned").exists()


def test_input_must_be_a_wav(tmp_path: Path) -> None:
    family, script = setup(tmp_path)
    family.load("stems-cpp")
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"nope")
    with pytest.raises(wav.WavError):
        run_job(family, make_job(tmp_path, "stems.split", "stems-cpp", {}, {"audio": str(junk)}))
    with pytest.raises(ValueError, match="missing audio input"):
        run_job(family, make_job(tmp_path, "stems.split", "stems-cpp", {}))
    assert not script.with_suffix(".argv").exists()
    family.unload("stems-cpp")
    with pytest.raises(RuntimeError, match="not loaded"):
        run_job(family, make_job(tmp_path, "stems.split", "stems-cpp", {}, audio_input(tmp_path)))


# ---- failures -------------------------------------------------------------------------------------

def test_nonzero_exit_reports_stderr(tmp_path: Path) -> None:
    family, _ = setup(tmp_path, mode="fail")
    family.load("stems-cpp")
    with pytest.raises(up.ProcessError, match="exited with code 1: error: model file is corrupt"):
        run_job(family, make_job(tmp_path, "stems.split", "stems-cpp", {}, audio_input(tmp_path)))


def test_timeout_kills_the_tool(tmp_path: Path) -> None:
    family, script = setup(tmp_path, mode="sleep", timeout_s=1)
    family.load("stems-cpp")
    started = time.monotonic()
    with pytest.raises(up.ProcessError, match="did not finish within 1 s"):
        run_job(family, make_job(tmp_path, "stems.split", "stems-cpp", {}, audio_input(tmp_path)))
    assert time.monotonic() - started < 20
    pid = int(script.with_suffix(".pid").read_text())
    deadline = time.monotonic() + 5
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_alive(pid)


def test_cancel_kills_the_tool(tmp_path: Path) -> None:
    family, script = setup(tmp_path, mode="sleep")
    family.load("stems-cpp")
    job = make_job(tmp_path, "stems.split", "stems-cpp", {}, audio_input(tmp_path))
    with pytest.raises(Cancelled):
        run_job(family, job, lambda: script.with_suffix(".pid").exists())  # cancel once the tool is running
    pid = int(script.with_suffix(".pid").read_text())
    deadline = time.monotonic() + 5
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_alive(pid)
    with pytest.raises(Cancelled):  # already cancelled before it starts: the tool never runs
        script.with_suffix(".argv").unlink()
        run_job(family, make_job(tmp_path / "b", "stems.split", "stems-cpp", {}, audio_input(tmp_path)), lambda: True)
    assert not script.with_suffix(".argv").exists()


def test_missing_or_unusable_outputs(tmp_path: Path) -> None:
    for mode in ("nostems", "empty"):
        family, _ = setup(tmp_path / mode, mode=mode)
        family.load("stems-cpp")
        with pytest.raises(up.ProcessError, match="wrote no stems"):
            run_job(family, make_job(tmp_path / mode, "stems.split", "stems-cpp", {}, audio_input(tmp_path / mode)))


def test_files_that_resolve_outside_the_job_directory_are_ignored(tmp_path: Path) -> None:
    family, _ = setup(tmp_path, mode="outside")
    family.load("stems-cpp")
    secret = tmp_path / "secret.wav"
    make_wav(secret)
    job = make_job(tmp_path, "stems.split", "stems-cpp", {}, audio_input(tmp_path))
    outputs, _ = run_job(family, job)
    assert "link" not in [o["extra"]["stem"] for o in outputs]
    assert all(inside(job["out_dir"], o["path"]) for o in outputs)
    assert (tmp_path / "out" / "stems" / "link.wav").is_symlink()  # left in place, never reported


# ---- protocol -------------------------------------------------------------------------------------

def test_protocol_end_to_end(tmp_path: Path) -> None:
    _, script = setup(tmp_path)
    options = {"binary": str(script), "model": str(tmp_path / "models" / "htdemucs.gguf")}
    with Harness(stems_worker.StemsFamily()) as h:
        _, hello = h.call("hello", options=options)
        assert hello["ok"] and hello["result"]["models"][0]["available"]
        assert h.call("load", model="stems-cpp")[1]["ok"]
        job = make_job(tmp_path, "stems.split", "stems-cpp", {"stems": ["drums"]}, audio_input(tmp_path))
        events, final = h.call("run", job=job)
        assert final["ok"] and [o["extra"] for o in final["result"]["outputs"]] == [{"stem": "drums"}]
        assert events
        validate_outputs(final["result"]["outputs"], Path(job["out_dir"]))
        assert h.call("unload", model="stems-cpp")[1]["ok"]


def test_protocol_cancel_kills_the_running_tool(tmp_path: Path) -> None:
    _, script = setup(tmp_path, mode="sleep")
    options = {"binary": str(script), "model": str(tmp_path / "models" / "htdemucs.gguf")}
    with Harness(stems_worker.StemsFamily()) as h:
        h.call("hello", options=options)
        h.call("load", model="stems-cpp")
        job = make_job(tmp_path, "stems.split", "stems-cpp", {}, audio_input(tmp_path))
        run_id = h.send("run", job=job)
        deadline = time.monotonic() + 10
        while not script.with_suffix(".pid").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        pid = int(script.with_suffix(".pid").read_text())
        cancel_id = h.send("cancel", job_id=job["job_id"])
        replies: dict[int, dict[str, Any]] = {}
        while len(replies) < 2:
            message = h.recv()
            if message.get("event") != "progress":
                replies[message["id"]] = message
        assert replies[cancel_id]["ok"] and replies[run_id] == {"id": run_id, "ok": False, "error": "cancelled"}
        deadline = time.monotonic() + 5
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not pid_alive(pid)
    assert listing(tmp_path / "out") == []


def test_parse_segment_and_argv() -> None:
    assert stems_worker.parse_segment("[stems] segment 3/10") == 0.3
    assert stems_worker.parse_segment("[stems] segment 10/10") == 1.0
    assert stems_worker.parse_segment("[stems] separated in 2.00 s") is None
    assert stems_worker.parse_segment("segment 1/0") is None
    argv = stems_worker.build_argv(Path("/b"), Path("/m.gguf"), Path("/in.wav"), Path("/o"), ["a", "b"], None, None)
    assert argv == ["/b", "--model", "/m.gguf", "--input", "/in.wav", "--out", "/o", "--stems", "a,b"]
