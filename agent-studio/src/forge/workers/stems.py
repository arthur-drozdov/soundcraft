"""The ``stems`` family: source separation with the stems.cpp command-line tool (CPU or GPU build).

Run as ``python -m forge.workers.stems``. The worker needs only numpy/scipy; the heavy lifting is the
``stems-split`` binary built from https://github.com/betweentwomidnights/stems.cpp (MIT).

``hello`` options (``deploy/forge.example.yaml``)::

    binary: /opt/forge/bin/stems-split
    model:  /opt/forge/models/stems/htdemucs-42M-v1.0-F32.gguf

Optional extras: ``timeout_s`` (1800), ``device`` (``cpu`` or ``gpu``, passed as ``--device``), ``threads``.

Network: none. The binary and the model are local files.

Command line (source: stems.cpp @ a9d2b6c, ``tools/stems-split.cpp``; run without a shell)::

    stems-split --model M.gguf --input IN.wav --out DIR [--stems a,b,...] [--device cpu|gpu] [--threads N]

It writes ``DIR/<source>.wav`` for every source (16 bit; sources are named by the model: ``drums``, ``bass``,
``other``, ``vocals`` and for the 6-stem model ``guitar``, ``piano``), prints each path on stdout and
reports ``[stems] segment i/n`` on stderr, which this worker turns into progress. A requested stem the model
does not have is an error that lists the stems it did produce. ``hello`` also runs ``stems-split --version``
so a binary that cannot execute is reported unavailable.

Each stem becomes one ``audio`` output with ``extra.stem``. ``candidates`` has no effect (separation is
deterministic) and one set of stems is returned.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from . import _upstream as up
from .protocol import ModelInfo, Output, RunJob
from .runtime import CancelCheck, Family, Progress, check_cancel, main

log = logging.getLogger("forge.worker.stems")

LICENCE = (
    "MIT (stems.cpp runtime; the licence of each model file applies: htdemucs variants are MIT, "
    "some UVR/RoFormer files have no upstream licence or are GPL-3.0, see deploy/README.md)"
)
DEFAULT_TIMEOUT_S = 1800.0
MAX_SECONDS = 1800.0
MAX_STEMS = 32
_STEM_NAME = re.compile(r"^[A-Za-z0-9_]{1,32}$")
_SEGMENT = re.compile(r"segment (\d+)/(\d+)")


def parse_segment(line: str) -> float | None:
    """Progress 0..1 from a ``[stems] segment 3/10`` line, else ``None``."""
    match = _SEGMENT.search(line)
    if not match:
        return None
    done, total = int(match.group(1)), int(match.group(2))
    return min(1.0, done / total) if total > 0 else None


def build_argv(binary: Path, model: Path, source: Path, out: Path, stems: list[str], device: str | None, threads: int | None) -> list[str]:
    argv = [str(binary), "--model", str(model), "--input", str(source), "--out", str(out)]
    if stems:
        argv += ["--stems", ",".join(stems)]
    if device:
        argv += ["--device", device]
    if threads:
        argv += ["--threads", str(threads)]
    return argv


class StemsFamily(Family):
    name = "stems"

    def __init__(self) -> None:
        self._binary: Any = None
        self._model: Any = None
        self._timeout = DEFAULT_TIMEOUT_S
        self._device: str | None = None
        self._threads: int | None = None
        self._config_error: str | None = None
        self._loaded = False

    def configure(self, options: dict[str, Any]) -> None:
        self._config_error = None
        self._binary, self._model = options.get("binary"), options.get("model")
        try:
            self._timeout = up.number(options, "timeout_s", DEFAULT_TIMEOUT_S, 1.0, 86400.0)
            device = options.get("device")
            if device is not None and device not in ("cpu", "gpu"):
                raise ValueError("device must be 'cpu' or 'gpu'")
            self._device = device
            threads = options.get("threads")
            self._threads = None if threads is None else up.integer(options, "threads", 0, 1, 256)
        except ValueError as exc:
            self._config_error = f"invalid options: {exc}"

    def _resolve(self) -> tuple[Path, Path]:
        """The binary and the model file, or :class:`MissingDependency` saying which is missing."""
        if self._config_error:
            raise up.MissingDependency(self._config_error)
        binary = up.resolve_executable(self._binary, "stems-split")
        if not isinstance(self._model, str) or not self._model.strip() or "\x00" in self._model:
            raise up.MissingDependency("options.model is not set (a stems.cpp .gguf file)")
        model = Path(self._model).expanduser()
        if not model.is_file():
            raise up.MissingDependency(f"model file {str(model)!r} does not exist")
        return binary, model.resolve()

    def models(self) -> list[ModelInfo]:
        reason: str | None = None
        checkpoint, digest = str(self._model or ""), None
        try:
            binary, model = self._resolve()
            checkpoint, digest = str(model), up.fingerprint(model)
            probe = up.run_process([str(binary), "--version"], timeout_s=15.0, cancelled=lambda: False)
            if probe.returncode != 0:
                reason = f"{binary.name} --version exited with code {probe.returncode}: {up.tail(probe.stderr)}"
            else:
                log.info("stems-split %s", probe.stdout.strip()[:80])
        except up.MissingDependency as exc:
            reason = str(exc)
        except up.ProcessError as exc:
            reason = str(exc)
        info: ModelInfo = {
            "id": "stems-cpp", "family": self.name, "job_types": ["stems.split"], "available": reason is None,
            "licence": LICENCE, "checkpoint": checkpoint, "checkpoint_hash": digest, "vram_gb": 0.0,
            "max_seconds": MAX_SECONDS,
        }
        if reason:
            info["reason"] = reason
        return [info]

    def load(self, model: str) -> None:
        if model != "stems-cpp":
            raise ValueError(f"unknown model {model!r}")
        self._resolve()
        self._loaded = True

    def unload(self, model: str) -> None:
        self._loaded = False

    def run(self, job: RunJob, progress: Progress, cancelled: CancelCheck) -> list[Output]:
        params, out_dir = up.check_job(job, ["stems.split"])
        if not self._loaded:
            raise RuntimeError("model 'stems-cpp' is not loaded")
        requested = params.get("stems") or []
        if not isinstance(requested, list) or len(requested) > MAX_STEMS or not all(isinstance(s, str) and _STEM_NAME.match(s) for s in requested):
            raise ValueError("stems must be a list of names made of letters, digits and underscores")
        source = up.input_path(job, "audio")
        up.read_audio_input(job, "audio", max_seconds=MAX_SECONDS)  # a WAV the tool can read, and a length cap
        binary, model = self._resolve()
        stem_dir = up.confine_dir(out_dir, "stems")
        argv = build_argv(binary, model, source, stem_dir, list(dict.fromkeys(requested)), self._device, self._threads)
        progress(0.02, "starting stems-split")
        check_cancel(cancelled)

        def on_line(line: str) -> None:
            fraction = parse_segment(line)
            if fraction is not None:
                progress(0.05 + 0.9 * fraction, f"separating ({line.strip()[-24:]})")

        log.info("running %s", " ".join(argv))
        result = up.run_process(argv, timeout_s=self._timeout, cancelled=cancelled, cwd=out_dir, on_stderr_line=on_line)
        if result.returncode != 0:
            raise up.ProcessError(f"stems-split exited with code {result.returncode}: {up.tail(result.stderr)}")
        outputs = collect_stems(stem_dir, out_dir, list(dict.fromkeys(requested)))
        progress(1.0, f"{len(outputs)} stems")
        return outputs


def collect_stems(stem_dir: Path, out_dir: Path, requested: list[str]) -> list[Output]:
    """One ``audio`` output per ``*.wav`` the tool left in ``stem_dir`` (confined to ``out_dir``).

    Raises when nothing was produced or a requested stem is missing."""
    root = out_dir.resolve()
    found: dict[str, Path] = {}
    for path in sorted(stem_dir.glob("*.wav")):
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            log.warning("ignoring %s: outside the job directory", path)
            continue
        if resolved.is_file() and resolved.stat().st_size > 44:
            found[path.stem] = resolved
    if not found:
        raise up.ProcessError("stems-split finished but wrote no stems")
    missing = [s for s in requested if s not in found]
    if missing:
        raise ValueError(f"the model has no stem(s) {', '.join(missing)}; it produced {', '.join(sorted(found))}")
    wanted = requested or sorted(found)
    return [
        {"path": str(found[stem]), "kind": "audio", "candidate": 0, "extra": {"stem": stem}}
        for stem in wanted
    ]


if __name__ == "__main__":
    main(StemsFamily())
