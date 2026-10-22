"""The ``render`` family: bounce a SoundCraft session with the headless ``soundcraft-cli``.

Run as ``python -m forge.workers.render``. Needs only numpy/scipy plus the ``soundcraft-cli`` binary.

``hello`` options (``deploy/forge.example.yaml``)::

    binary: /opt/forge/bin/soundcraft-cli

Optional extras: ``timeout_s`` (1800), ``invocation`` (``auto``, ``script`` or ``cmd-file``; see below),
``skip_capability_check`` (false).

Network: none.

Job ``render.session`` (``model`` ``soundcraft``):

1. The ``session`` input is a zip holding exactly one ``.scraft`` (and, usually, its ``Audio Files/``). It is
   extracted into ``<out_dir>/session/`` with hostile-archive checks: at most 20 000 members and 2 GiB
   uncompressed (counted while copying, not trusted from the headers), no absolute or ``..`` paths, no drive
   letters or backslash tricks, no symbolic links, no encrypted members, no duplicate names.
2. The commands (at most 1000, no NaN or infinity) are written to a file in ``out_dir``.
3. The tool runs with no shell, in one of two forms, chosen by ``invocation``:

   ``script`` (what ``soundcraft-cli`` has today; chosen by ``auto`` when ``--cmd-file`` is not in its help)::

       <binary> script <out_dir>/commands.txt --in <out_dir>/session/<x>.scraft \
                       --bounce <out_dir>/render.wav [--start <seconds> [--end <seconds>]]

   where ``commands.txt`` has one ``<command id> <params JSON>`` line per command (the CLI's own script format;
   the first failing command stops the run with exit code 1 and ``file:line: id: error`` on stderr).

   ``cmd-file`` (the invocation the Rust side may add for Forge; chosen by ``auto`` when ``--cmd-file`` is
   listed in the help)::

       <binary> run --session <out_dir>/session/<x>.scraft --cmd-file <out_dir>/commands.jsonl \
                    --bounce <out_dir>/render.wav [--start <seconds> [--end <seconds>]]

   where ``commands.jsonl`` has one ``{"command": "<id>", "params": {...}}`` object per line.

   ``hello`` runs ``--version`` and ``--help`` and reports the model unavailable when the binary does not run or
   lacks the chosen form (``skip_capability_check`` skips the help check; ``auto`` then means ``script``).
4. A non-zero exit, a timeout, cancellation (the whole process group is killed) or a missing/non-WAV
   ``render.wav`` fails the job; the tail of stderr is in the error.

The output is one ``audio`` output (``render.wav``) with ``extra.commands`` (the number of commands run) and
``extra.invocation``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from . import _upstream as up
from .protocol import ModelInfo, Output, RunJob
from .runtime import CancelCheck, Family, Progress, check_cancel, main

log = logging.getLogger("forge.worker.render")

LICENCE = "MIT OR Apache-2.0 (SoundCraft)"
DEFAULT_TIMEOUT_S = 1800.0
MAX_SECONDS = 86400.0
MAX_ENTRIES = 20_000
MAX_UNCOMPRESSED = 2 * 1024**3
MAX_COMMANDS = 1000
MAX_COMMAND_FILE = 8 * 1024 * 1024
MAX_NAME_CHARS = 1024
MAX_PART_CHARS = 255
COPY_CHUNK = 1 << 20
REQUIRED_FLAG = "--cmd-file"
INVOCATIONS = ("auto", "script", "cmd-file")
_COMMAND_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


# ---- archive extraction (unit tested) -------------------------------------------------------------

def clean_member_name(info: zipfile.ZipInfo) -> PurePosixPath:
    """The safe relative path of an archive member, or ``ValueError`` naming the problem."""
    raw = info.filename
    if not raw or len(raw) > MAX_NAME_CHARS or "\x00" in raw or any(ord(c) < 32 for c in raw):
        raise ValueError("session archive has an invalid member name")
    name = raw.replace("\\", "/")
    if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        raise ValueError(f"session archive has an absolute path: {raw!r}")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts) or any(len(p) > MAX_PART_CHARS or ":" in p for p in parts):
        raise ValueError(f"session archive has an unsafe path: {raw!r}")
    kind = stat.S_IFMT(info.external_attr >> 16)  # 0 when the archiver stored no file type
    if kind and kind not in (stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"session archive member {raw!r} is a link or special file")
    if info.flag_bits & 0x1:
        raise ValueError(f"session archive member {raw!r} is encrypted")
    return PurePosixPath(*parts)


def extract_session(
    archive: Path, dest: Path, *, max_entries: int = MAX_ENTRIES, max_bytes: int = MAX_UNCOMPRESSED,
) -> Path:
    """Safely extract a session zip into ``dest`` (created) and return the path of its single ``.scraft``.

    Sizes are enforced while copying, so a member that lies about its size cannot exceed ``max_bytes``."""
    try:
        zf = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError(f"the session input is not a readable zip file: {exc}") from exc
    with zf:
        infos = zf.infolist()
        if not infos or len(infos) > max_entries:
            raise ValueError(f"session archive must have 1..{max_entries} members, has {len(infos)}")
        if sum(i.file_size for i in infos) > max_bytes:
            raise ValueError("session archive is too large when extracted")
        members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        seen: set[str] = set()
        scraft: list[PurePosixPath] = []
        for info in infos:
            path = clean_member_name(info)
            key = str(path).casefold()
            if key in seen:
                raise ValueError(f"session archive lists {str(path)!r} twice")
            seen.add(key)
            members.append((info, path))
            if not info.is_dir() and path.name.lower().endswith(".scraft"):
                scraft.append(path)
        if len(scraft) != 1:
            raise ValueError(f"a session archive must contain exactly one .scraft file, found {len(scraft)}")
        total = 0
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for info, path in members:
                target = up.confine(dest, *path.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                with zf.open(info) as src, open(target, "xb") as out:
                    while chunk := src.read(COPY_CHUNK):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError("session archive is too large when extracted")
                        out.write(chunk)
        except (OSError, zipfile.BadZipFile, RuntimeError, EOFError) as exc:
            raise ValueError(f"cannot extract the session archive: {exc}") from exc
        return up.confine(dest, *scraft[0].parts)


def command_lines(commands: Any, mode: str = "cmd-file") -> str:
    """``commands`` as the text of a command file; validated and size-capped.

    ``cmd-file``: one ``{"command", "params"}`` JSON object per line. ``script``: one ``<id> <params JSON>`` line
    (the ids ``soundcraft-cli script`` accepts have no whitespace, so they are restricted to ``[A-Za-z0-9_.-]``)."""
    if mode not in ("cmd-file", "script"):
        raise ValueError(f"unknown command file format {mode!r}")
    if commands is None:
        commands = []
    if not isinstance(commands, list) or len(commands) > MAX_COMMANDS:
        raise ValueError(f"commands must be a list of at most {MAX_COMMANDS} items")
    lines = []
    for index, item in enumerate(commands):
        if not isinstance(item, dict):
            raise ValueError(f"commands[{index}] must be an object")
        name, params = item.get("command"), item.get("params", {})
        if not isinstance(name, str) or not 1 <= len(name) <= 128 or any(ord(c) < 32 for c in name):
            raise ValueError(f"commands[{index}].command must be a command id")
        if mode == "script" and not _COMMAND_ID.match(name):
            raise ValueError(f"commands[{index}].command {name!r} may only contain letters, digits, '_', '.' and '-'")
        if not isinstance(params, dict):
            raise ValueError(f"commands[{index}].params must be an object")
        try:
            if mode == "script":
                lines.append(f"{name} {json.dumps(params, allow_nan=False, separators=(',', ':'))}")
            else:
                lines.append(json.dumps({"command": name, "params": params}, allow_nan=False, separators=(",", ":")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"commands[{index}] is not valid JSON: {exc}") from exc
    text = "\n".join(lines) + ("\n" if lines else "")
    if len(text.encode("utf-8")) > MAX_COMMAND_FILE:
        raise ValueError("commands are too large")
    return text


def time_arg(seconds: float) -> str:
    return f"{seconds:.6f}"


def build_argv(
    binary: Path, mode: str, session: Path, cmd_file: Path, bounce: Path, start: float, end: float | None,
) -> list[str]:
    """The command line for ``mode`` (``script`` or ``cmd-file``). ``--start`` and ``--end`` are independent
    in ``soundcraft-cli``; they are only passed when the job narrows the range."""
    if mode == "script":
        argv = [str(binary), "script", str(cmd_file), "--in", str(session), "--bounce", str(bounce)]
    elif mode == "cmd-file":
        argv = [str(binary), "run", "--session", str(session), "--cmd-file", str(cmd_file), "--bounce", str(bounce)]
    else:
        raise ValueError(f"unknown invocation {mode!r}")
    if end is not None:
        argv += ["--start", time_arg(start), "--end", time_arg(end)]
    elif start > 0:
        argv += ["--start", time_arg(start)]
    return argv


# ---- the family -----------------------------------------------------------------------------------

class RenderFamily(Family):
    name = "render"

    def __init__(self) -> None:
        self._binary: Any = None
        self._timeout = DEFAULT_TIMEOUT_S
        self._invocation = "auto"
        self._skip_check = False
        self._config_error: str | None = None
        self._mode: str | None = None  # the invocation in use, settled by hello/load
        self._loaded = False

    def configure(self, options: dict[str, Any]) -> None:
        self._config_error = None
        self._binary = options.get("binary")
        self._skip_check = bool(options.get("skip_capability_check", False))
        self._mode = None
        try:
            self._timeout = up.number(options, "timeout_s", DEFAULT_TIMEOUT_S, 1.0, 86400.0)
            self._invocation = str(options.get("invocation", "auto"))
            if self._invocation not in INVOCATIONS:
                raise ValueError(f"invocation must be one of {', '.join(INVOCATIONS)}")
        except ValueError as exc:
            self._config_error = f"invalid options: {exc}"

    def _resolve(self) -> Path:
        if self._config_error:
            raise up.MissingDependency(self._config_error)
        return up.resolve_executable(self._binary, "soundcraft-cli")

    def _detect(self, binary: Path) -> str:
        """The invocation to use with ``binary``; raises :class:`MissingDependency` saying why it cannot render."""
        try:
            version = up.run_process([str(binary), "--version"], timeout_s=15.0, cancelled=lambda: False)
        except up.ProcessError as exc:
            raise up.MissingDependency(str(exc)) from exc
        if version.returncode != 0:
            raise up.MissingDependency(f"{binary.name} --version exited with code {version.returncode}: {up.tail(version.stderr)}")
        if self._skip_check:
            return "script" if self._invocation == "auto" else self._invocation
        helptext = ""
        for args in (["--help"], ["run", "--help"]):
            try:
                result = up.run_process([str(binary), *args], timeout_s=15.0, cancelled=lambda: False)
            except up.ProcessError as exc:
                raise up.MissingDependency(str(exc)) from exc
            helptext += f"\n{result.stdout}\n{result.stderr}"
        has_cmd_file = REQUIRED_FLAG in helptext
        has_script = re.search(r"\bscript\b", helptext) is not None and "--bounce" in helptext
        wanted = self._invocation
        if wanted == "auto":
            wanted = "cmd-file" if has_cmd_file else "script"
        if wanted == "cmd-file" and not has_cmd_file:
            raise up.MissingDependency(f"{binary.name} does not list `run {REQUIRED_FLAG}` in its help (invocation=cmd-file)")
        if wanted == "script" and not has_script:
            raise up.MissingDependency(
                f"{binary.name} lists neither `run {REQUIRED_FLAG}` nor `script FILE --in S --bounce OUT` in its help; "
                "it cannot render sessions headlessly. Set options.skip_capability_check to bypass"
            )
        return wanted

    def models(self) -> list[ModelInfo]:
        reason: str | None = None
        checkpoint, digest = str(self._binary or ""), None
        try:
            binary = self._resolve()
            checkpoint, digest = str(binary), up.fingerprint(binary)
            self._mode = self._detect(binary)
        except up.MissingDependency as exc:
            reason = str(exc)
        info: ModelInfo = {
            "id": "soundcraft", "family": self.name, "job_types": ["render.session"], "available": reason is None,
            "licence": LICENCE, "checkpoint": checkpoint, "checkpoint_hash": digest, "vram_gb": 0.0,
            "max_seconds": MAX_SECONDS,
        }
        if reason:
            info["reason"] = reason
        return [info]

    def load(self, model: str) -> None:
        if model != "soundcraft":
            raise ValueError(f"unknown model {model!r}")
        binary = self._resolve()
        if self._mode is None:
            self._mode = self._detect(binary)
        self._loaded = True

    def unload(self, model: str) -> None:
        self._loaded = False

    def run(self, job: RunJob, progress: Progress, cancelled: CancelCheck) -> list[Output]:
        params, out_dir = up.check_job(job, ["render.session"])
        if not self._loaded or self._mode is None:
            raise RuntimeError("model 'soundcraft' is not loaded")
        mode = self._mode
        start = up.number(params, "start", 0.0, 0.0, 86400.0)
        end_raw = params.get("end")
        end = None if end_raw is None else up.number(params, "end", 0.0, 0.0, 86400.0)
        if end is not None and end <= start:
            raise ValueError("end must be greater than start")
        text = command_lines(params.get("commands"), mode)
        binary = self._resolve()
        archive = up.input_path(job, "session")
        progress(0.02, "unpacking session")
        session_dir = up.confine(out_dir, "session", "x").parent
        if session_dir.exists():
            shutil.rmtree(session_dir)
        session = extract_session(archive, session_dir)
        check_cancel(cancelled)
        cmd_file = up.confine(out_dir, "commands.txt" if mode == "script" else "commands.jsonl")
        cmd_file.write_text(text, encoding="utf-8")
        bounce = up.confine(out_dir, "render.wav")
        argv = build_argv(binary, mode, session, cmd_file, bounce, start, end)
        count = text.count("\n")
        progress(0.1, f"rendering ({count} commands, {mode})")
        log.info("running %s", " ".join(argv))
        result = up.run_process(argv, timeout_s=self._timeout, cancelled=cancelled, cwd=out_dir)
        if result.returncode != 0:
            raise up.ProcessError(f"soundcraft-cli exited with code {result.returncode}: {up.tail(result.stderr or result.stdout)}")
        check_render(bounce)
        progress(1.0, "rendered")
        extra = {"commands": count, "invocation": mode, "seconds": round(result.seconds, 3)}
        return [{"path": str(bounce), "kind": "audio", "candidate": 0, "extra": extra}]


def check_render(path: Path) -> None:
    """The bounce must exist inside the job directory and look like a WAV file."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            head = handle.read(12)
    except OSError as exc:
        raise up.ProcessError("soundcraft-cli finished but wrote no bounce file") from exc
    if size < 44 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise up.ProcessError("soundcraft-cli wrote a bounce that is not a WAV file")


if __name__ == "__main__":
    main(RenderFamily())
