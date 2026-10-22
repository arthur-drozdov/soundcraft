"""The worker side of the JSON-lines protocol (DESIGN 3.2), implemented once for every family."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TextIO

from .protocol import MAX_LINE_BYTES, PROTOCOL_VERSION, ModelInfo, Output, RunJob

log = logging.getLogger("forge.worker")

Progress = Callable[[float, str], None]
CancelCheck = Callable[[], bool]


class Cancelled(Exception):
    """Raised by a family when it notices the job was cancelled."""


def check_cancel(cancelled: CancelCheck) -> None:
    if cancelled():
        raise Cancelled("cancelled")


class Family(ABC):
    """A model family. Subclass, set ``name`` and implement ``models`` and ``run``."""

    name: str = ""

    def configure(self, options: dict[str, Any]) -> None:
        """Receive ``families.<name>.options`` from the ``hello`` request."""

    @abstractmethod
    def models(self) -> list[ModelInfo]:
        """Describe every model, reporting ``available: False`` with a ``reason`` when its packages are missing."""

    def load(self, model: str) -> None:
        """Bring the model into memory."""

    def unload(self, model: str) -> None:
        """Free the model."""

    @abstractmethod
    def run(self, job: RunJob, progress: Progress, cancelled: CancelCheck) -> list[Output]:
        """Run a job, writing only inside ``job["out_dir"]``."""


class _Server:
    def __init__(self, family: Family, stdin: TextIO, stdout: TextIO) -> None:
        self.family = family
        self.stdin = stdin
        self.stdout = stdout
        self._write_lock = threading.Lock()
        self._cancel = threading.Event()
        self._current_job: str | None = None

    def send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, allow_nan=False, separators=(",", ":"))
        with self._write_lock:
            self.stdout.write(line + "\n")
            self.stdout.flush()

    def reply(self, req_id: Any, ok: bool = True, **fields: Any) -> None:
        self.send({"id": req_id, "ok": ok, **fields})

    def busy(self) -> bool:
        return self._current_job is not None

    def serve(self) -> int:
        while True:
            line = self.stdin.readline(MAX_LINE_BYTES + 1)
            if not line:
                self._cancel.set()
                return 0  # the gateway went away
            if len(line) > MAX_LINE_BYTES:
                self.reply(None, False, error="request too large")
                self._cancel.set()
                return 2
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except ValueError:
                self.reply(None, False, error="invalid JSON")
                continue
            req_id = request.get("id") if isinstance(request, dict) else None
            if isinstance(req_id, bool) or not isinstance(req_id, int):
                self.reply(None, False, error="request needs an integer id")
                continue
            if request.get("op") == "shutdown":
                self._cancel.set()
                return 0
            self.handle(request)

    def handle(self, request: dict[str, Any]) -> None:
        req_id, op = request["id"], request.get("op")
        try:
            if op == "cancel":
                if self._current_job is not None and request.get("job_id") == self._current_job:
                    self._cancel.set()
                self.reply(req_id)
            elif self.busy():
                self.reply(req_id, False, error="worker is busy")
            elif op == "hello":
                options = request.get("options") or {}
                if not isinstance(options, dict):
                    raise ValueError("options must be an object")
                self.family.configure(options)
                models = self.family.models()
                self.reply(req_id, result={"protocol": PROTOCOL_VERSION, "family": self.family.name, "models": models})
            elif op in ("load", "unload"):
                model = request.get("model")
                if not isinstance(model, str):
                    raise ValueError("model must be a string")
                (self.family.load if op == "load" else self.family.unload)(model)
                self.reply(req_id)
            elif op == "run":
                self.start_run(req_id, request.get("job"))
            else:
                self.reply(req_id, False, error=f"unknown op {op!r}")
        except Exception as exc:  # protocol boundary: a bad request must not kill the worker
            log.exception("request %r failed", op)
            self.reply(req_id, False, error=f"{type(exc).__name__}: {exc}")

    def start_run(self, req_id: int, job: Any) -> None:
        if not isinstance(job, dict) or not isinstance(job.get("job_id"), str):
            raise ValueError("run needs a job object")
        self._cancel.clear()
        self._current_job = job["job_id"]
        threading.Thread(target=self._run, args=(req_id, job), daemon=True).start()

    def _run(self, req_id: int, job: Any) -> None:
        def progress(fraction: float, message: str = "") -> None:
            value = min(1.0, max(0.0, float(fraction)))
            self.send({"id": req_id, "event": "progress", "progress": value, "message": str(message)[:500]})

        final: dict[str, Any]
        try:
            outputs = self.family.run(job, progress, self._cancel.is_set)
            if self._cancel.is_set():
                raise Cancelled("cancelled")
            final = {"id": req_id, "ok": True, "result": {"outputs": outputs}}
        except Cancelled:
            final = {"id": req_id, "ok": False, "error": "cancelled"}
        except Exception as exc:  # the job fails, the worker survives
            log.exception("job %s failed", job.get("job_id"))
            final = {"id": req_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # Become idle before replying, so the gateway's next request is never refused as "busy".
        self._current_job = None
        try:
            self.send(final)
        except (TypeError, ValueError) as exc:  # result not JSON-serialisable
            self.send({"id": req_id, "ok": False, "error": f"unserialisable result: {exc}"})


def serve(family: Family, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    """Run the protocol loop until ``shutdown`` or end of input. Returns the process exit code."""
    if stdin is None and stdout is None:
        # Keep the protocol on a private copy of fd 1 so stray prints from model libraries cannot corrupt it.
        stdout = os.fdopen(os.dup(1), "w", encoding="utf-8")
        os.dup2(2, 1)
        sys.stdout = sys.stderr
    return _Server(family, stdin if stdin is not None else sys.stdin, stdout if stdout is not None else sys.stdout).serve()


def main(family: Family) -> None:
    """Entry point for ``python -m forge.workers.<family>``."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format=f"[{family.name}] %(levelname)s %(message)s")
    code = serve(family)
    sys.stderr.flush()
    os._exit(code)  # do not wait for daemon threads or model libraries' atexit hooks
