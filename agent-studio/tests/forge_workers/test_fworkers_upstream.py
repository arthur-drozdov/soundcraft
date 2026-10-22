"""Shared worker helpers: availability probing, validation, path confinement, subprocess runner."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from forge.workers import _upstream as up
from forge.workers.runtime import Cancelled

from fworkers_fakes import FakeTensor, block, install_torch, module, pid_alive, put, write_script


# ---- availability ---------------------------------------------------------------------------------

def test_probe_module_never_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    assert up.probe_module("json") is None
    reason = up.probe_module("definitely_not_installed_pkg")
    assert reason and "not installed" in reason
    assert up.probe_module("not_a_pkg.sub.module")  # missing parent: reported, not raised
    put(monkeypatch, module("fake_pkg_for_probe"))  # __spec__ is None, still counts as present
    assert up.probe_module("fake_pkg_for_probe") is None
    block(monkeypatch, "blocked_pkg_for_probe")
    assert "blocked" in (up.probe_module("blocked_pkg_for_probe") or "")
    assert "definitely_not_installed_pkg" not in sys.modules
    assert up.first_missing("json", "definitely_not_installed_pkg", "os")


def test_import_module_wraps_every_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    assert up.import_module("json").dumps
    with pytest.raises(up.MissingDependency, match="cannot import definitely_not_installed_pkg"):
        up.import_module("definitely_not_installed_pkg")

    class Exploding:
        def __getattr__(self, name: str) -> None:
            raise NameError("name '_default_model_type' is not defined")

    # basic_pitch raises NameError (not ImportError) when no inference runtime is installed
    monkeypatch.setattr("importlib.import_module", lambda name: (_ for _ in ()).throw(NameError("_default_model_type")))
    with pytest.raises(up.MissingDependency, match="NameError"):
        up.import_module("basic_pitch")


def test_free_accelerator_memory_touches_cuda_only_when_torch_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    up.free_accelerator_memory()  # no torch: nothing to do, no error
    torch = install_torch(monkeypatch, cuda=True)
    up.free_accelerator_memory()
    assert torch.cuda.empty_cache_calls == 1  # type: ignore[attr-defined]
    cpu = install_torch(monkeypatch, cuda=False)
    up.free_accelerator_memory()
    assert cpu.cuda.empty_cache_calls == 0  # type: ignore[attr-defined]


def test_device_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    assert up.check_device("CUDA") == "cuda" and up.check_device("cuda:1") == "cuda:1" and up.check_device(None) == "cuda"
    for bad in ("cuda; rm -rf", "gpu", "cuda:abc", "../x"):
        with pytest.raises(ValueError, match="unsupported device"):
            up.check_device(bad)
    torch = install_torch(monkeypatch, cuda=False)
    assert up.resolve_device(torch, "cuda") == "cpu"
    assert up.resolve_device(torch, "cpu") == "cpu"
    assert up.resolve_device(install_torch(monkeypatch, cuda=True), "cuda") == "cuda"


# ---- tensors -------------------------------------------------------------------------------------

def test_tensor_to_audio_shapes_and_sanitising() -> None:
    batch = FakeTensor(np.ones((1, 2, 5), np.float16), "float16")
    out = up.tensor_to_audio(batch)
    assert out.shape == (2, 5) and out.dtype == np.float32
    assert up.tensor_to_audio(np.ones(7)).shape == (1, 7)
    dirty = np.array([[0.5, np.nan, np.inf, -np.inf]], np.float32)
    assert np.all(np.isfinite(up.tensor_to_audio(dirty)))
    for bad in (np.ones((1, 2, 0)), np.ones((0, 4)), np.ones((2, 2, 2, 2))):
        with pytest.raises(ValueError, match="unexpected shape"):
            up.tensor_to_audio(bad)


# ---- validation ----------------------------------------------------------------------------------

def test_number_and_integer_reject_hostile_values() -> None:
    assert up.number({"x": 3}, "x", 1.0, 0, 10) == 3.0
    assert up.number({}, "x", 1.5, 0, 10) == 1.5 and up.number({"x": None}, "x", 1.5, 0, 10) == 1.5
    for bad in (float("nan"), float("inf"), True, "3", [1], 11, -1):
        with pytest.raises(ValueError):
            up.number({"x": bad}, "x", 1.0, 0, 10)
    assert up.integer({"n": 4.0}, "n", 1, 1, 8) == 4
    for bad in (4.5, True, "4", 0, 9, float("nan")):
        with pytest.raises(ValueError):
            up.integer({"n": bad}, "n", 1, 1, 8)


def test_check_job_validates_the_shape(tmp_path: Path) -> None:
    job = {"type": "a", "model": "m", "params": {}, "out_dir": str(tmp_path), "inputs": {}}
    params, out = up.check_job(job, ["a"])
    assert params == {} and out == tmp_path.resolve()
    with pytest.raises(ValueError, match="does not run"):
        up.check_job(job, ["b"])
    for patch in ({"params": []}, {"out_dir": None}, {"out_dir": str(tmp_path / "missing")}, {"inputs": []}, {"out_dir": "a\x00b"}):
        with pytest.raises(ValueError):
            up.check_job({**job, **patch}, ["a"])
    with pytest.raises(ValueError):
        up.check_job([], ["a"])  # type: ignore[arg-type]


def test_seeds_and_inputs(tmp_path: Path) -> None:
    assert up.job_seed({"seed": 5}) == 5
    for bad in (-1, 2**63, True, "5", None):
        with pytest.raises(ValueError, match="seed"):
            up.job_seed({"seed": bad})
    assert up.candidate_seed(2**63 - 1, 1) == 0  # wraps instead of overflowing
    assert up.candidate_seed(10, 3) == 13
    file = tmp_path / "a.wav"
    file.write_bytes(b"x")
    assert up.input_path({"inputs": {"audio": str(file)}}, "audio") == file
    for inputs in ({}, {"audio": ""}, {"audio": str(tmp_path)}, {"audio": str(tmp_path / "gone")}, {"audio": "a\x00"}, {"audio": 3}):
        with pytest.raises(ValueError):
            up.input_path({"inputs": inputs}, "audio")


# ---- files ---------------------------------------------------------------------------------------

def test_confine_blocks_traversal_and_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    assert up.confine(root, "a", "b.wav") == (root / "a" / "b.wav").resolve()
    assert (root / "a").is_dir()  # parents are created
    for parts in (("..", "evil.wav"), ("a", "..", "..", "evil"), ("/etc/passwd",)):
        with pytest.raises(ValueError, match="escapes"):
            up.confine(root, *parts)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        up.confine(root, "link", "file.wav")
    assert not (outside / "file.wav").exists()
    assert up.confine_dir(root, "stems").is_dir()


def test_write_audio_refuses_non_finite_and_stays_inside(tmp_path: Path) -> None:
    audio = np.zeros((1, 10), np.float32)
    path = up.write_audio(tmp_path, "sub/x.wav", audio, 8000)
    assert path.is_file() and path.parent == (tmp_path / "sub").resolve()
    with pytest.raises(ValueError, match="non-finite"):
        up.write_audio(tmp_path, "bad.wav", np.array([[np.nan]], np.float32), 8000)
    with pytest.raises(ValueError, match="escapes"):
        up.write_audio(tmp_path, "../bad.wav", audio, 8000)


def test_safe_name_and_fingerprint(tmp_path: Path) -> None:
    assert up.safe_name("../../etc/passwd") == "etc_passwd"
    assert up.safe_name("   ", "fallback") == "fallback" and len(up.safe_name("x" * 200)) == 48
    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(3 << 20))
    a = up.fingerprint(big)
    assert a and a.startswith("fp:")
    big.write_bytes(big.read_bytes()[:-1] + b"\x00")
    assert up.fingerprint(big) != a  # the last MiB is part of the fingerprint
    assert up.fingerprint(tmp_path / "missing") is None


def test_read_json_file_caps_and_reports(tmp_path: Path) -> None:
    good = tmp_path / "c.json"
    good.write_text('{"a": 1}')
    assert up.read_json_file(good) == {"a": 1}
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(up.MissingDependency, match="not valid JSON"):
        up.read_json_file(bad)
    with pytest.raises(up.MissingDependency, match="larger than"):
        up.read_json_file(good, max_bytes=3)
    with pytest.raises(up.MissingDependency, match="cannot read"):
        up.read_json_file(tmp_path / "missing.json")


def test_fade_out_keeps_length() -> None:
    audio = np.ones((2, 1000), np.float32)
    out = up.fade_out(audio, 1000, 100)
    assert out.shape == audio.shape and out[0, -1] == 0.0 and out[0, 0] == 1.0 and audio[0, -1] == 1.0
    assert up.fade_out(audio, 1000, 0) is audio


def test_span_maps_into_its_slice() -> None:
    seen: list[tuple[float, str]] = []
    top = up.Span(lambda f, m: seen.append((f, m)))
    top.part(1, 4)(0.5, "half")
    top.part(3, 4)(2.0, "clamped")
    top(0.0, "start")
    assert seen == [(0.375, "half"), (1.0, "clamped"), (0.0, "start")]


# ---- executables and subprocesses -----------------------------------------------------------------

def test_resolve_executable(tmp_path: Path) -> None:
    script = write_script(tmp_path / "tool", "print('hi')\n")
    assert up.resolve_executable(str(script), "tool") == script
    assert up.resolve_executable("sh", "shell").name == "sh"  # bare names use PATH
    plain = tmp_path / "plain"
    plain.write_text("x")
    for value, match in ((None, "not set"), ("", "not set"), (str(tmp_path / "gone"), "does not exist"), (str(plain), "not executable"),
                         ("no-such-binary-xyz", "not found on PATH"), ("a\x00b", "not set")):
        with pytest.raises(up.MissingDependency, match=match):
            up.resolve_executable(value, "tool")


def test_run_process_captures_output_and_streams_cr_progress(tmp_path: Path) -> None:
    script = write_script(
        tmp_path / "tool",
        """
        import sys, time
        print("out line")
        for i in range(1, 4):
            sys.stderr.write(f"\\r[x] segment {i}/3")
            sys.stderr.flush()
            time.sleep(0.02)
        sys.stderr.write("\\ndone\\n")
        sys.exit(3)
        """,
    )
    seen: list[str] = []
    result = up.run_process([str(script)], timeout_s=20, cancelled=lambda: False, on_stderr_line=seen.append)
    assert result.returncode == 3  # a non-zero exit is returned, the caller decides
    assert result.stdout == "out line"
    assert seen[:3] == ["[x] segment 1/3", "[x] segment 2/3", "[x] segment 3/3"] and seen[-1] == "done"
    assert "segment 3/3" in result.stderr


def test_run_process_times_out_and_kills_the_group(tmp_path: Path) -> None:
    marker = tmp_path / "pids"
    script = write_script(
        tmp_path / "tool",
        f"""
        import os, subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(marker)!r}, "w").write(f"{{os.getpid()}} {{child.pid}}")
        time.sleep(60)
        """,
    )
    started = time.monotonic()
    with pytest.raises(up.ProcessError, match="did not finish within 1 s"):
        up.run_process([str(script)], timeout_s=1.0, cancelled=lambda: False)
    assert time.monotonic() - started < 15
    parent, child = (int(x) for x in marker.read_text().split())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and (pid_alive(parent) or pid_alive(child)):
        time.sleep(0.05)
    assert not pid_alive(parent) and not pid_alive(child)  # the whole process group died


def test_run_process_cancellation(tmp_path: Path) -> None:
    script = write_script(tmp_path / "tool", "import time\ntime.sleep(60)\n")
    flag = threading.Event()
    threading.Timer(0.3, flag.set).start()
    started = time.monotonic()
    with pytest.raises(Cancelled):
        up.run_process([str(script)], timeout_s=60, cancelled=flag.is_set)
    assert time.monotonic() - started < 10


def test_run_process_caps_captured_output_and_rejects_bad_input(tmp_path: Path) -> None:
    script = write_script(tmp_path / "tool", "import sys\nsys.stdout.write(('line\\n') * 50000)\n")
    result = up.run_process([str(script)], timeout_s=30, cancelled=lambda: False, capture_chars=2000)
    assert result.returncode == 0 and len(result.stdout) <= 2100
    flood = write_script(tmp_path / "flood", "import sys\nsys.stdout.write('x' * 500000)\n")
    assert len(up.run_process([str(flood)], timeout_s=30, cancelled=lambda: False).stdout) <= 70_000
    for argv in ([], ["a\x00b"], [3]):
        with pytest.raises(ValueError):
            up.run_process(argv, timeout_s=1, cancelled=lambda: False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timeout"):
        up.run_process([str(script)], timeout_s=0, cancelled=lambda: False)
    with pytest.raises(up.ProcessError, match="cannot start"):
        up.run_process([str(tmp_path / "does-not-exist")], timeout_s=1, cancelled=lambda: False)


def test_run_process_does_not_use_a_shell(tmp_path: Path) -> None:
    script = write_script(tmp_path / "tool", "import sys\nprint(sys.argv[1:])\n")
    nasty = ["; touch pwned", "$(touch pwned)", "`touch pwned`", "a b"]
    result = up.run_process([str(script), *nasty], timeout_s=20, cancelled=lambda: False, cwd=tmp_path)
    assert result.stdout == str(nasty)
    assert not (tmp_path / "pwned").exists()


def test_tail_flattens_and_limits() -> None:
    assert up.tail("a\n\n b \nc") == "a | b | c"
    assert len(up.tail("x" * 5000)) == 600
