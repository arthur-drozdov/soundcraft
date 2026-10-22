"""Helpers shared by the real model-family workers (foundation, sa3, acestep, basicpitch, stems, render).

Nothing here imports a model package at import time. ``probe_module`` answers "is it installed?"
without importing it (``hello`` must stay cheap and side-effect free), ``import_module`` imports
lazily and turns every failure into :class:`MissingDependency`, and the torch helpers only touch
torch when it is already imported.

Also here: job-input validation (workers treat the job as hostile even though the gateway
validated it), path confinement (a worker only writes inside ``job["out_dir"]``), seed derivation
and a subprocess runner (no shell, timeout, cancellation kills the whole process group) used by the
``stems`` and ``render`` families.
"""

from __future__ import annotations

import collections
import contextlib
import gc
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audiokit import wav

from .runtime import CancelCheck, Cancelled

log = logging.getLogger("forge.worker")

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_SEED = 2**63
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_DEVICE = re.compile(r"^(cpu|mps|cuda(:\d{1,2})?)$")


class MissingDependency(RuntimeError):
    """A model package, binary or checkpoint is not available (the message says which)."""


# ---------------------------------------------------------------- availability

def probe_module(name: str) -> str | None:
    """``None`` when ``name`` looks importable, else the reason. Never imports the module.

    A module already in ``sys.modules`` counts as present (``None`` there blocks it, as Python does)."""
    if name in sys.modules:
        return None if sys.modules[name] is not None else f"package {name!r} is blocked"
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError) as exc:
        return f"package {name!r} cannot be located ({exc})"
    return None if spec is not None else f"package {name!r} is not installed"


def import_module(name: str) -> Any:
    """Import ``name`` or raise :class:`MissingDependency`.

    Third-party packages fail in many ways at import (``basic_pitch`` raises ``NameError`` when no
    inference runtime is installed, CUDA libraries raise ``OSError``), so any ``Exception`` is
    reported as a missing dependency with the original message."""
    try:
        return importlib.import_module(name)
    except Exception as exc:  # third-party import side effects: report, never crash the worker
        raise MissingDependency(f"cannot import {name}: {type(exc).__name__}: {exc}") from exc


def first_missing(*names: str) -> str | None:
    """The reason for the first module in ``names`` that is not installed, or ``None``."""
    for name in names:
        reason = probe_module(name)
        if reason:
            return reason
    return None


# ---------------------------------------------------------------- torch helpers

def free_accelerator_memory() -> None:
    """``gc.collect()`` and, when torch is already imported and has CUDA, ``torch.cuda.empty_cache()``."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # a broken CUDA runtime must not fail unload
        log.debug("empty_cache failed: %s", exc)


def check_device(requested: Any) -> str:
    """A validated device string (``cpu``, ``mps``, ``cuda`` or ``cuda:N``); anything else is an error."""
    text = str(requested or "cuda").strip().lower()
    if not _DEVICE.match(text):
        raise ValueError(f"unsupported device {requested!r} (use cpu, mps, cuda or cuda:N)")
    return text


def resolve_device(torch: Any, requested: str) -> str:
    """``requested``, or ``cpu`` (with a warning) when CUDA is asked for but unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("device %s requested but CUDA is unavailable; falling back to cpu", requested)
        return "cpu"
    return requested


def tensor_to_audio(result: Any) -> np.ndarray:
    """Model output (torch tensor ``[batch, channels, n]``, ``[channels, n]`` or ``[n]``, or an array) to
    finite ``float32`` ``(channels, frames)``. The first batch item is taken."""
    value = result
    for step in ("detach", "float", "cpu"):
        method = getattr(value, step, None)
        if callable(method):
            value = method()
    if hasattr(value, "numpy"):
        value = value.numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"model returned audio of unexpected shape {tuple(arr.shape)}")
    return np.ascontiguousarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)


# ---------------------------------------------------------------- job validation

def number(params: Mapping[str, Any], name: str, default: float, lo: float, hi: float) -> float:
    """``params[name]`` as a finite float within ``lo..hi`` (``default`` when absent or ``None``)."""
    value = params.get(name, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"param {name!r} must be a finite number")
    if not lo <= value <= hi:
        raise ValueError(f"param {name!r} must be between {lo:g} and {hi:g}, got {value:g}")
    return float(value)


def integer(params: Mapping[str, Any], name: str, default: int, lo: int, hi: int) -> int:
    value = params.get(name, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
        raise ValueError(f"param {name!r} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"param {name!r} must be between {lo} and {hi}, got {int(value)}")
    return int(value)


def check_job(job: Mapping[str, Any], allowed_types: Sequence[str]) -> tuple[Mapping[str, Any], Path]:
    """Validate the shape of a ``RunJob``; returns ``(params, resolved out_dir)``."""
    if not isinstance(job, Mapping):
        raise ValueError("job must be an object")
    job_type = job.get("type")
    if job_type not in allowed_types:
        raise ValueError(f"model {job.get('model')!r} does not run {job_type!r}")
    params = job.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("job params must be an object")
    out_dir = job.get("out_dir")
    if not isinstance(out_dir, str) or "\x00" in out_dir:
        raise ValueError("job needs an out_dir")
    root = Path(out_dir).resolve()
    if not root.is_dir():
        raise ValueError("job out_dir is not a directory")
    if not isinstance(job.get("inputs", {}), Mapping):
        raise ValueError("job inputs must be an object")
    return params, root


def job_seed(job: Mapping[str, Any]) -> int:
    seed = job.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < MAX_SEED:
        raise ValueError("job seed must be a non-negative integer below 2**63")
    return seed


def candidate_seed(seed: int, candidate: int) -> int:
    """The seed of one candidate: ``seed + candidate`` (the value Forge records in the sidecar)."""
    return (seed + candidate) % MAX_SEED


def candidate_count(params: Mapping[str, Any]) -> int:
    return integer(params, "candidates", 1, 1, 8)


def input_path(job: Mapping[str, Any], name: str) -> Path:
    """The existing file the gateway staged as input ``name``."""
    inputs = job.get("inputs") or {}
    raw = inputs.get(name) if isinstance(inputs, Mapping) else None
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError(f"missing {name} input")
    path = Path(raw)
    if not path.is_file():
        raise ValueError(f"{name} input is not a file")
    return path


def read_audio_input(job: Mapping[str, Any], name: str = "audio", max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Read the WAV input ``name`` (frame-capped by ``audiokit.wav``), optionally capping its duration."""
    audio, sr = wav.read(input_path(job, name))
    if audio.shape[1] == 0:
        raise ValueError(f"the {name} input is empty")
    if max_seconds is not None and audio.shape[1] / sr > max_seconds:
        raise ValueError(f"the {name} input is {audio.shape[1] / sr:.1f} s long; this model accepts at most {max_seconds:g} s")
    return audio, sr


# ---------------------------------------------------------------- files and confinement

def safe_name(text: str, fallback: str = "file", limit: int = 48) -> str:
    cleaned = _SAFE_NAME.sub("_", str(text)).strip("._")[:limit]
    return cleaned or fallback


def confine(root: Path, *parts: str) -> Path:
    """``root/parts`` resolved, guaranteed to lie inside ``root`` (symbolic links and ``..`` are followed
    and rejected when they escape). Parent directories are created."""
    base = root.resolve()
    target = (base.joinpath(*parts)).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path {'/'.join(parts)!r} escapes the job directory") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def confine_dir(root: Path, *parts: str) -> Path:
    """Like :func:`confine` but creates the directory itself."""
    target = confine(root, *parts, "x").parent
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_audio(root: Path, relative: str, audio: np.ndarray, sr: int) -> Path:
    """Write float WAV ``audio`` at ``root/relative`` (confined). Post-processing re-encodes it."""
    if not np.all(np.isfinite(audio)):
        raise ValueError("refusing to write non-finite audio")
    path = confine(root, relative)
    wav.write(path, audio, int(sr), "32f")
    return path


def fingerprint(path: Path) -> str | None:
    """A cheap identity of a (possibly multi-GB) file: size plus its first and last MiB, hashed.

    It is a fingerprint, not a content hash; it is what ``checkpoint_hash`` reports for local files."""
    try:
        size = path.stat().st_size
        digest = hashlib.sha256(str(size).encode())
        with open(path, "rb") as handle:
            digest.update(handle.read(1 << 20))
            if size > 2 << 20:
                handle.seek(-(1 << 20), os.SEEK_END)
                digest.update(handle.read(1 << 20))
        return "fp:" + digest.hexdigest()[:24]
    except OSError:
        return None


def read_json_file(path: Path, max_bytes: int = MAX_JSON_BYTES) -> Any:
    """Parse a (size-capped) JSON file; raises :class:`MissingDependency` with a readable message."""
    try:
        if path.stat().st_size > max_bytes:
            raise MissingDependency(f"{path.name} is larger than {max_bytes} bytes")
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MissingDependency(f"cannot read {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise MissingDependency(f"{path.name} is not valid JSON: {exc}") from exc


def fade_out(audio: np.ndarray, sr: int, ms: float) -> np.ndarray:
    """A short linear fade-out on the last ``ms`` milliseconds (avoids clicks after a hard cut)."""
    n = min(audio.shape[1], int(round(ms / 1000.0 * sr)))
    if n <= 1:
        return audio
    out = audio.copy()
    out[:, -n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


# ---------------------------------------------------------------- executables and subprocesses

def resolve_executable(value: Any, what: str) -> Path:
    """The configured binary as an absolute executable file (bare names are looked up on ``PATH``)."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise MissingDependency(f"options.binary is not set ({what})")
    text = value.strip()
    if os.sep not in text:
        import shutil

        found = shutil.which(text)
        if found is None:
            raise MissingDependency(f"{what} binary {text!r} was not found on PATH")
        text = found
    path = Path(text).expanduser()
    if not path.is_file():
        raise MissingDependency(f"{what} binary {str(path)!r} does not exist")
    if not os.access(path, os.X_OK):
        raise MissingDependency(f"{what} binary {str(path)!r} is not executable")
    return Path(os.path.abspath(path))


class ProcessError(RuntimeError):
    """An external tool failed (non-zero exit) or timed out."""


@dataclass
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    seconds: float


def tail(text: str, limit: int = 600) -> str:
    """The last ``limit`` characters of ``text`` on one line, for error messages."""
    flat = " | ".join(line.strip() for line in text.strip().splitlines() if line.strip())
    return flat[-limit:]


def _pump(stream: Any, sink: collections.deque[str], on_line: Callable[[str], None] | None, cap: int) -> None:
    """Read a pipe in chunks, split it on CR and LF (progress bars use CR), keep the last ``cap`` characters."""
    pending = b""
    kept = 0
    try:
        while True:
            chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
            if not chunk:
                break
            pending += chunk
            if len(pending) > 65536 and b"\n" not in pending and b"\r" not in pending:
                pending = pending[-4096:]  # one runaway line: keep only its end
            *lines, pending = re.split(rb"[\r\n]", pending)
            for raw in lines:
                line = raw.decode("utf-8", "replace")
                if not line:
                    continue
                if on_line is not None:
                    on_line(line)
                sink.append(line)
                kept += len(line) + 1
                while kept > cap and len(sink) > 1:
                    kept -= len(sink.popleft()) + 1
        if pending:
            line = pending.decode("utf-8", "replace")
            if on_line is not None:
                on_line(line)
            sink.append(line)
    except (OSError, ValueError):
        pass  # the pipe closed under us (process killed)
    finally:
        with contextlib.suppress(OSError):
            stream.close()


def kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the child's process group (it was started in its own session); falls back to ``kill``."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # pragma: no cover - the workers run on Linux
            proc.kill()
    except OSError:  # includes ProcessLookupError and PermissionError
        with contextlib.suppress(OSError):
            proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL cannot be ignored
        log.error("process %s did not die after SIGKILL", proc.pid)


def run_process(
    argv: Sequence[str],
    *,
    timeout_s: float,
    cancelled: CancelCheck,
    cwd: Path | None = None,
    on_stderr_line: Callable[[str], None] | None = None,
    capture_chars: int = 64 * 1024,
) -> ProcessResult:
    """Run ``argv`` without a shell. Stdin is closed; stdout and stderr are captured (last ``capture_chars``).

    Polls every 50 ms: when ``cancelled()`` turns true the whole process group is killed and
    :class:`Cancelled` is raised; after ``timeout_s`` it is killed and :class:`ProcessError` raised.
    A non-zero exit is *returned*; the caller decides what it means."""
    if not argv or any((not isinstance(a, str)) or "\x00" in a for a in argv):
        raise ValueError("invalid command line")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be positive")
    started = time.monotonic()
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv list, shell=False, binary comes from the operator's config
            list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(cwd) if cwd else None, shell=False, close_fds=True, start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise ProcessError(f"cannot start {Path(argv[0]).name}: {exc.strerror or exc}") from exc
    out_lines: collections.deque[str] = collections.deque()
    err_lines: collections.deque[str] = collections.deque()
    assert proc.stdout is not None and proc.stderr is not None
    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, out_lines, None, capture_chars), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, err_lines, on_stderr_line, capture_chars), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        while True:
            try:
                proc.wait(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancelled():
                kill_process_tree(proc)
                raise Cancelled("cancelled")
            if time.monotonic() - started > timeout_s:
                kill_process_tree(proc)
                raise ProcessError(f"{Path(argv[0]).name} did not finish within {timeout_s:g} s and was killed")
    finally:
        if proc.poll() is None:
            kill_process_tree(proc)
        for thread in threads:
            thread.join(timeout=2.0)
    return ProcessResult(proc.returncode, "\n".join(out_lines), "\n".join(err_lines), time.monotonic() - started)


# ---------------------------------------------------------------- progress

class Span:
    """A slice ``[start, end]`` of the job's 0..1 progress. ``span(fraction, message)`` reports inside the
    slice; ``span.part(i, n)`` is the i-th of n equal sub-slices."""

    def __init__(self, report: Callable[[float, str], None], start: float = 0.0, end: float = 1.0) -> None:
        self._report = report
        self.start = start
        self.end = end

    def __call__(self, fraction: float, message: str = "") -> None:
        value = self.start + (self.end - self.start) * min(1.0, max(0.0, float(fraction)))
        self._report(value, message)

    def part(self, index: int, count: int) -> Span:
        width = (self.end - self.start) / max(1, count)
        return Span(self._report, self.start + width * index, self.start + width * (index + 1))
