"""The ``render`` worker: safe session extraction and the ``soundcraft-cli run --session ... --cmd-file ...`` call."""

from __future__ import annotations

import json
import os
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

from audiokit import wav
from forge.workers import _upstream as up
from forge.workers import render
from forge.workers.protocol import validate_model_info, validate_outputs
from forge.workers.runtime import Cancelled

from fworkers_fakes import Harness, listing, make_job, pid_alive, run_job, write_script

FAKE_CLI = """
    import json, os, sys, time, wave
    from pathlib import Path

    me = Path(sys.argv[0])
    args = sys.argv[1:]
    mode = me.with_suffix(".mode").read_text().strip() if me.with_suffix(".mode").exists() else "ok"
    flavor = me.with_suffix(".flavor").read_text().strip() if me.with_suffix(".flavor").exists() else "script"
    if args == ["--version"]:
        if mode == "badversion":
            sys.exit(5)
        print("SoundCraft CLI 0.0-test")
        sys.exit(0)
    if args == ["--help"]:
        print("soundcraft-cli run [--in S.scraft | --demo] [--cmd ID[=JSON]]... [--bounce OUT] [--start T --end T]")
        if flavor == "script":
            print("soundcraft-cli script FILE [--in S | --demo] [--save S] [--bounce OUT]   (one `ID JSON` per line)")
        if flavor == "cmdfile":
            print("soundcraft-cli run --session S.scraft --cmd-file F.jsonl --bounce OUT [--start T --end T]")
        sys.exit(0)
    if args == ["run", "--help"]:
        sys.exit(0)
    me.with_suffix(".argv").write_text(json.dumps(args))
    me.with_suffix(".pid").write_text(str(os.getpid()))
    if args[0] == "script":
        cmd_file, rest = args[1], args[2:]
        opts = dict(zip(rest[0::2], rest[1::2]))
        session = Path(opts["--in"])
    else:
        assert args[0] == "run", args
        opts = dict(zip(args[1::2], args[2::2]))
        session, cmd_file = Path(opts["--session"]), opts["--cmd-file"]
    assert session.is_file() and session.suffix == ".scraft", session
    me.with_suffix(".session").write_text(json.dumps(sorted(str(p.relative_to(session.parent)) for p in session.parent.rglob("*") if p.is_file())))
    me.with_suffix(".cmds").write_text(Path(cmd_file).read_text())
    if mode == "fail":
        sys.stderr.write("commands.txt:1: bogus.command: unknown command `bogus.command`\\n")
        sys.exit(1)
    if mode == "sleep":
        time.sleep(60)
    if mode == "nowav":
        sys.exit(0)
    out = Path(opts["--bounce"])
    if mode == "badwav":
        out.write_bytes(b"this is not a wav file at all, not even close, really not.....")
        sys.exit(0)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes(b"\\x00\\x10\\x00\\x10" * 4800)
    print(json.dumps({"path": str(out), "seconds": 0.1}))
"""


def make_session_zip(path: Path, extra: dict[str, bytes] | None = None, scraft_name: str = "My Song.scraft") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(scraft_name, json.dumps({"version": 1}))
        zf.writestr("Audio Files/", b"")
        zf.writestr("Audio Files/kick 1.wav", b"RIFFxxxxWAVE")
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return path


def setup(tmp: Path, mode: str = "ok", flavor: str = "script", **options: Any) -> tuple[render.RenderFamily, Path]:
    """A fake soundcraft-cli. ``flavor`` is what its help lists: ``script`` (the CLI today), ``cmdfile`` or ``none``."""
    script = write_script(tmp / "bin" / "soundcraft-cli", FAKE_CLI)
    script.with_suffix(".mode").write_text(mode)
    script.with_suffix(".flavor").write_text(flavor)
    family = render.RenderFamily()
    family.configure({"binary": str(script), **options})
    return family, script


def render_job(tmp: Path, params: dict[str, Any] | None = None) -> dict[str, Any]:
    archive = make_session_zip(tmp / "in" / "session.zip")
    return make_job(tmp, "render.session", "soundcraft", params or {}, {"session": str(archive)})


# ---- safe extraction ------------------------------------------------------------------------------

def crafted(tmp: Path, members: list[tuple[str, bytes, dict[str, Any]]]) -> Path:
    path = tmp / "evil.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data, attrs in members:
            info = zipfile.ZipInfo(name)
            info.external_attr = attrs.get("external_attr", 0o644 << 16)
            info.flag_bits |= attrs.get("flag_bits", 0)
            zf.writestr(info, data)
    return path


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("../evil.scraft", "unsafe path"),
        ("a/../../evil.scraft", "unsafe path"),
        ("..\\evil.scraft", "unsafe path"),
        ("/etc/evil.scraft", "absolute path"),
        ("\\evil.scraft", "absolute path"),
        ("C:\\evil.scraft", "absolute path"),
        ("a/b:c/evil.scraft", "unsafe path"),
        ("bad\nname.scraft", "invalid member name"),
        ("x" * 300 + ".scraft", "unsafe path"),
    ],
)
def test_zip_slip_and_other_hostile_names_are_rejected(tmp_path: Path, name: str, message: str) -> None:
    archive = crafted(tmp_path, [("fine.scraft", b"{}", {}), (name, b"owned", {})])
    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match=message):
        render.extract_session(archive, dest)
    assert not dest.exists() or listing(dest) == []  # validated before anything was written
    assert listing(tmp_path) == ["evil.zip"]  # and nothing escaped


def test_links_special_files_and_encrypted_members_are_rejected(tmp_path: Path) -> None:
    symlink = crafted(tmp_path, [("a.scraft", b"{}", {}), ("link", b"/etc/passwd", {"external_attr": 0o120777 << 16})])
    with pytest.raises(ValueError, match="link or special file"):
        render.extract_session(symlink, tmp_path / "d1")
    device = crafted(tmp_path, [("a.scraft", b"{}", {}), ("dev", b"", {"external_attr": 0o060666 << 16})])
    with pytest.raises(ValueError, match="link or special file"):
        render.extract_session(device, tmp_path / "d2")
    info = zipfile.ZipInfo("a.scraft")
    info.flag_bits |= 0x1  # zipfile.writestr would clear the bit, so check the per-member test directly
    with pytest.raises(ValueError, match="encrypted"):
        render.clean_member_name(info)
    nul = zipfile.ZipInfo("ok.scraft")
    nul.filename = "bad\x00name.scraft"  # ZipInfo() truncates at NUL, an archive read from disk may not
    with pytest.raises(ValueError, match="invalid member name"):
        render.clean_member_name(nul)


def test_duplicates_and_scraft_count(tmp_path: Path) -> None:
    dup = crafted(tmp_path, [("a.scraft", b"{}", {}), ("A.SCRAFT", b"{}", {})])
    with pytest.raises(ValueError, match="twice"):
        render.extract_session(dup, tmp_path / "d1")
    two = crafted(tmp_path, [("a.scraft", b"{}", {}), ("sub/b.scraft", b"{}", {})])
    with pytest.raises(ValueError, match="exactly one .scraft file, found 2"):
        render.extract_session(two, tmp_path / "d2")
    none = crafted(tmp_path, [("readme.txt", b"hi", {})])
    with pytest.raises(ValueError, match="found 0"):
        render.extract_session(none, tmp_path / "d3")
    with pytest.raises(ValueError, match="not a readable zip"):
        render.extract_session(tmp_path / "missing.zip", tmp_path / "d4")
    garbage = tmp_path / "garbage.zip"
    garbage.write_bytes(b"PK\x03\x04 definitely not a zip")
    with pytest.raises(ValueError, match="not a readable zip"):
        render.extract_session(garbage, tmp_path / "d5")
    empty = tmp_path / "empty.zip"
    zipfile.ZipFile(empty, "w").close()
    with pytest.raises(ValueError, match="1..20000 members"):
        render.extract_session(empty, tmp_path / "d6")


def test_member_and_size_caps(tmp_path: Path) -> None:
    many = crafted(tmp_path, [(f"f{i}.txt", b"x", {}) for i in range(5)] + [("a.scraft", b"{}", {})])
    with pytest.raises(ValueError, match="1..3 members, has 6"):
        render.extract_session(many, tmp_path / "d1", max_entries=3)
    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.scraft", b"{}")
        zf.writestr("zeros.bin", b"\0" * (3 << 20))
    assert bomb.stat().st_size < 20_000  # tiny on disk, 3 MiB when extracted
    with pytest.raises(ValueError, match="too large when extracted"):
        render.extract_session(bomb, tmp_path / "d2", max_bytes=1 << 20)
    assert not (tmp_path / "d2" / "zeros.bin").exists()


def test_a_normal_session_extracts_with_its_layout(tmp_path: Path) -> None:
    archive = make_session_zip(tmp_path / "s.zip", extra={"Audio Files\\snare.wav": b"RIFFxxxxWAVE", "Notes/readme.txt": b"hi"})
    dest = tmp_path / "dest"
    scraft = render.extract_session(archive, dest)
    assert scraft == (dest / "My Song.scraft").resolve() and scraft.is_file()
    assert listing(dest) == ["Audio Files/kick 1.wav", "Audio Files/snare.wav", "My Song.scraft", "Notes/readme.txt"]
    nested = make_session_zip(tmp_path / "n.zip", scraft_name="folder/inner.scraft")
    assert render.extract_session(nested, tmp_path / "dest2") == (tmp_path / "dest2" / "folder" / "inner.scraft").resolve()


def test_extraction_cannot_follow_a_planted_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "Audio Files").symlink_to(outside)
    archive = make_session_zip(tmp_path / "s.zip")
    with pytest.raises(ValueError):
        render.extract_session(archive, dest)
    assert listing(outside) == []


# ---- commands and argv ----------------------------------------------------------------------------

def test_command_lines_in_both_formats() -> None:
    commands = [{"command": "mix.volume", "params": {"track": "Kick", "db": -6.5}}, {"command": "clip.split"}]
    assert render.command_lines(commands, "cmd-file") == '{"command":"mix.volume","params":{"track":"Kick","db":-6.5}}\n{"command":"clip.split","params":{}}\n'
    assert render.command_lines(commands, "script") == 'mix.volume {"track":"Kick","db":-6.5}\nclip.split {}\n'
    assert render.command_lines(commands) == render.command_lines(commands, "cmd-file")
    for mode in ("script", "cmd-file"):
        assert render.command_lines([], mode) == "" and render.command_lines(None, mode) == ""
    # a newline inside a string stays escaped, so one command is always one line
    assert render.command_lines([{"command": "a.b", "params": {"text": "x\ny"}}], "script") == 'a.b {"text":"x\\ny"}\n'
    with pytest.raises(ValueError, match="command file format"):
        render.command_lines([], "yaml")


def test_command_validation() -> None:
    for mode in ("script", "cmd-file"):
        for bad in (
            "nope", [1], [{"params": {}}], [{"command": ""}], [{"command": "x" * 200}], [{"command": "a\nb"}], [{"command": "a", "params": []}],
            [{"command": "a", "params": {"v": float("nan")}}], [{"command": "a", "params": {"v": float("inf")}}],
            [{"command": "a", "params": {"v": object()}}], [{"command": "a"}] * 1001,
        ):
            with pytest.raises(ValueError):
                render.command_lines(bad, mode)
        with pytest.raises(ValueError, match="too large"):
            render.command_lines([{"command": "a", "params": {"blob": "x" * 5_000_000}}] * 2, mode)
    # the script format splits on whitespace, so ids with spaces (or anything exotic) cannot be smuggled in
    for name in ("a b", "a\tb", "a;b", "a$(b)", "é"):
        with pytest.raises(ValueError):
            render.command_lines([{"command": name}], "script")
    render.command_lines([{"command": "a b"}], "cmd-file")  # JSON lines have no such limit


def test_build_argv() -> None:
    b, s, c, o = Path("/bin/cli"), Path("/s/x.scraft"), Path("/o/commands.x"), Path("/o/render.wav")
    script = ["/bin/cli", "script", "/o/commands.x", "--in", "/s/x.scraft", "--bounce", "/o/render.wav"]
    cmdfile = ["/bin/cli", "run", "--session", "/s/x.scraft", "--cmd-file", "/o/commands.x", "--bounce", "/o/render.wav"]
    for mode, base in (("script", script), ("cmd-file", cmdfile)):
        assert render.build_argv(b, mode, s, c, o, 0.0, None) == base
        assert render.build_argv(b, mode, s, c, o, 2.5, None) == [*base, "--start", "2.500000"]
        assert render.build_argv(b, mode, s, c, o, 0.0, 8.0) == [*base, "--start", "0.000000", "--end", "8.000000"]
        assert render.build_argv(b, mode, s, c, o, 1.0, 2.0)[-4:] == ["--start", "1.000000", "--end", "2.000000"]
    with pytest.raises(ValueError, match="unknown invocation"):
        render.build_argv(b, "gui", s, c, o, 0.0, None)


# ---- availability ---------------------------------------------------------------------------------

def test_available_and_the_invocation_follows_the_cli(tmp_path: Path) -> None:
    family, script = setup(tmp_path)
    (info,) = family.models()
    assert info["id"] == "soundcraft" and info["available"] is True and info["checkpoint"] == str(script)
    assert info["checkpoint_hash"].startswith("fp:") and info["job_types"] == ["render.session"] and info["licence"].startswith("MIT OR Apache-2.0")
    validate_model_info(info, "render")
    assert family._mode == "script"  # today's soundcraft-cli: `script FILE --in S --bounce OUT`
    family2, _ = setup(tmp_path / "b", flavor="cmdfile")
    assert family2.models()[0]["available"] and family2._mode == "cmd-file"  # once `run --cmd-file` exists it is preferred


def test_forced_invocations_and_capability_checks(tmp_path: Path) -> None:
    family, script = setup(tmp_path, flavor="script", invocation="cmd-file")
    (info,) = family.models()
    assert not info["available"] and "--cmd-file" in info["reason"] and "invocation=cmd-file" in info["reason"]
    family.configure({"binary": str(script), "invocation": "script"})
    assert family.models()[0]["available"] and family._mode == "script"
    family.configure({"binary": str(script), "invocation": "teleport"})
    assert "invocation must be one of" in family.models()[0]["reason"]
    none, nscript = setup(tmp_path / "n", flavor="none")
    reason = none.models()[0]["reason"]
    assert not none.models()[0]["available"] and "--cmd-file" in reason and "script FILE" in reason and "skip_capability_check" in reason
    none.configure({"binary": str(nscript), "skip_capability_check": True})
    assert none.models()[0]["available"] and none._mode == "script"


@pytest.mark.parametrize(("options", "expected"), [({}, "not set"), ({"binary": "/nonexistent/cli"}, "does not exist"), ({"binary": "CLI", "timeout_s": 0}, "invalid options")])
def test_unavailable_reasons(tmp_path: Path, options: dict[str, Any], expected: str) -> None:
    family, script = setup(tmp_path)
    family.configure({k: str(script) if v == "CLI" else v for k, v in options.items()})
    (info,) = family.models()
    assert not info["available"] and expected in info["reason"]
    validate_model_info(info, "render")
    with pytest.raises(up.MissingDependency):
        family.load("soundcraft")


def test_a_binary_that_fails_its_version_check(tmp_path: Path) -> None:
    family, _ = setup(tmp_path, mode="badversion")
    (info,) = family.models()
    assert not info["available"] and "exited with code 5" in info["reason"]


# ---- runs -----------------------------------------------------------------------------------------

COMMANDS = [{"command": "mix.volume", "params": {"track": "Kick", "db": -6}}, {"command": "transport.seek", "params": {"seconds": 1.5}}]


def test_render_session_uses_the_script_subcommand_of_todays_cli(tmp_path: Path) -> None:
    family, script = setup(tmp_path)
    family.load("soundcraft")
    job = render_job(tmp_path, {"commands": COMMANDS, "start": 1.0, "end": 3.0})
    outputs, events = run_job(family, job)
    out = (tmp_path / "out").resolve()
    assert json.loads(script.with_suffix(".argv").read_text()) == [
        "script", str(out / "commands.txt"), "--in", str(out / "session" / "My Song.scraft"), "--bounce", str(out / "render.wav"),
        "--start", "1.000000", "--end", "3.000000",
    ]
    assert script.with_suffix(".cmds").read_text() == (out / "commands.txt").read_text() == 'mix.volume {"track":"Kick","db":-6}\ntransport.seek {"seconds":1.5}\n'
    assert json.loads(script.with_suffix(".session").read_text()) == ["Audio Files/kick 1.wav", "My Song.scraft"]
    assert len(outputs) == 1 and outputs[0]["kind"] == "audio" and outputs[0]["candidate"] == 0
    assert outputs[0]["path"] == str(out / "render.wav")
    assert outputs[0]["extra"]["commands"] == 2 and outputs[0]["extra"]["invocation"] == "script"
    audio, sr = wav.read(outputs[0]["path"])
    assert sr == 48000 and audio.shape == (2, 4800)
    assert events[0][0] == 0.02 and events[-1] == (1.0, "rendered") and any("script" in m for _, m in events)
    validate_outputs(outputs, Path(job["out_dir"]))


def test_render_session_with_the_cmd_file_form(tmp_path: Path) -> None:
    family, script = setup(tmp_path, flavor="cmdfile")
    family.load("soundcraft")
    outputs, _ = run_job(family, render_job(tmp_path, {"commands": COMMANDS, "start": 1.0, "end": 3.0}))
    out = (tmp_path / "out").resolve()
    assert json.loads(script.with_suffix(".argv").read_text()) == [
        "run", "--session", str(out / "session" / "My Song.scraft"), "--cmd-file", str(out / "commands.jsonl"),
        "--bounce", str(out / "render.wav"), "--start", "1.000000", "--end", "3.000000",
    ]
    assert script.with_suffix(".cmds").read_text() == render.command_lines(COMMANDS, "cmd-file")
    assert outputs[0]["extra"]["invocation"] == "cmd-file"


def test_defaults_pass_no_range(tmp_path: Path) -> None:
    family, script = setup(tmp_path)
    family.load("soundcraft")
    run_job(family, render_job(tmp_path))
    argv = json.loads(script.with_suffix(".argv").read_text())
    assert "--start" not in argv and "--end" not in argv and argv[0] == "script"
    assert script.with_suffix(".cmds").read_text() == ""
    run_job(family, render_job(tmp_path / "b", {"start": 2.0}))
    assert json.loads(script.with_suffix(".argv").read_text())[-2:] == ["--start", "2.000000"]


def test_invalid_jobs_never_start_the_tool(tmp_path: Path) -> None:
    family, script = setup(tmp_path)
    family.load("soundcraft")
    cases: list[dict[str, Any]] = [
        {"end": 1.0, "start": 2.0}, {"start": -1}, {"end": float("nan")}, {"commands": "x"}, {"commands": [{"command": "a", "params": {"v": float("nan")}}]},
        {"commands": [{"command": "two words"}]},
    ]
    for params in cases:
        with pytest.raises(ValueError):
            run_job(family, render_job(tmp_path / "x", params))
    with pytest.raises(ValueError, match="missing session input"):
        run_job(family, make_job(tmp_path / "y", "render.session", "soundcraft", {}))
    not_a_zip = tmp_path / "plain.zip"
    not_a_zip.write_text("hello")
    with pytest.raises(ValueError, match="not a readable zip"):
        run_job(family, make_job(tmp_path / "z", "render.session", "soundcraft", {}, {"session": str(not_a_zip)}))
    assert not script.with_suffix(".argv").exists()
    family.unload("soundcraft")
    with pytest.raises(RuntimeError, match="not loaded"):
        run_job(family, render_job(tmp_path / "w"))


def test_a_hostile_archive_fails_the_job_and_writes_nothing_outside(tmp_path: Path) -> None:
    family, script = setup(tmp_path)
    family.load("soundcraft")
    archive = crafted(tmp_path, [("ok.scraft", b"{}", {}), ("../../../escaped.txt", b"owned", {})])
    job = make_job(tmp_path, "render.session", "soundcraft", {}, {"session": str(archive)})
    with pytest.raises(ValueError, match="unsafe path"):
        run_job(family, job)
    assert not script.with_suffix(".argv").exists()
    assert not [f for f in listing(tmp_path) if "escaped" in f] and not (tmp_path.parent / "escaped.txt").exists()
    assert not (tmp_path / "out" / "session").exists() or listing(tmp_path / "out" / "session") == []


def test_tool_failures(tmp_path: Path) -> None:
    family, _ = setup(tmp_path, mode="fail")
    family.load("soundcraft")
    with pytest.raises(up.ProcessError, match="exited with code 1: commands.txt:1: bogus.command: unknown command"):
        run_job(family, render_job(tmp_path))
    for mode, message in (("nowav", "wrote no bounce"), ("badwav", "not a WAV")):
        fam, _ = setup(tmp_path / mode, mode=mode)
        fam.load("soundcraft")
        with pytest.raises(up.ProcessError, match=message):
            run_job(fam, render_job(tmp_path / mode))


def test_timeout_and_cancel_kill_the_tool(tmp_path: Path) -> None:
    family, script = setup(tmp_path, mode="sleep", timeout_s=1)
    family.load("soundcraft")
    with pytest.raises(up.ProcessError, match="did not finish within 1 s"):
        run_job(family, render_job(tmp_path))
    pid = int(script.with_suffix(".pid").read_text())
    deadline = time.monotonic() + 5
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_alive(pid)
    script.with_suffix(".pid").unlink()
    family.configure({"binary": str(script), "timeout_s": 600})
    family.load("soundcraft")
    with pytest.raises(Cancelled):
        run_job(family, render_job(tmp_path / "c"), script.with_suffix(".pid").exists)  # cancel once the tool is running
    pid = int(script.with_suffix(".pid").read_text())
    deadline = time.monotonic() + 5
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_alive(pid)


def test_protocol_end_to_end(tmp_path: Path) -> None:
    _, script = setup(tmp_path)
    with Harness(render.RenderFamily()) as h:
        _, hello = h.call("hello", options={"binary": str(script)})
        assert hello["ok"] and hello["result"]["models"][0]["available"]
        assert h.call("load", model="soundcraft")[1]["ok"]
        job = render_job(tmp_path, {"commands": [{"command": "mix.volume", "params": {"track": "Kick", "db": -3}}]})
        events, final = h.call("run", job=job)
        assert final["ok"] and events and final["result"]["outputs"][0]["extra"]["invocation"] == "script"
        validate_outputs(final["result"]["outputs"], Path(job["out_dir"]))
        assert h.call("unload", model="soundcraft")[1]["ok"]


# ---- opt-in integration with a real soundcraft-cli -------------------------------------------------

def real_cli() -> Path:
    value = os.environ.get("SOUNDCRAFT_CLI")
    if not value or not Path(value).is_file():
        pytest.skip("set SOUNDCRAFT_CLI to a soundcraft-cli binary to run the integration tests")
    return Path(value)


def demo_session_zip(cli: Path, tmp: Path) -> Path:
    """A real session (``demo.scraft`` + ``Audio Files/``) written by the CLI itself, zipped like an upload."""
    folder = tmp / "demo"
    folder.mkdir()
    subprocess.run([str(cli), "run", "--demo", "--save", str(folder / "demo.scraft")], check=True, capture_output=True, timeout=120)
    archive = tmp / "demo.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(folder).as_posix())
    return archive


def rms(audio: Any) -> float:
    return float((audio.astype("float64") ** 2).mean() ** 0.5)


def test_integration_with_the_real_cli(tmp_path: Path) -> None:
    """Uses whichever invocation the binary supports (``script`` today). Run with SOUNDCRAFT_CLI=<binary>."""
    cli = real_cli()
    family = render.RenderFamily()
    family.configure({"binary": str(cli), "timeout_s": 300})
    (info,) = family.models()
    assert info["available"], info.get("reason")
    family.load("soundcraft")
    archive = demo_session_zip(cli, tmp_path)
    plain = make_job(tmp_path / "plain", "render.session", "soundcraft", {"start": 10.0, "end": 12.0}, {"session": str(archive)})
    quiet = make_job(
        tmp_path / "quiet", "render.session", "soundcraft",
        {"commands": [{"command": "mix.volume", "params": {"track": "Kick", "db": -60}}], "start": 10.0, "end": 12.0}, {"session": str(archive)},
    )
    (plain_out,), _ = run_job(family, plain)
    (quiet_out,), _ = run_job(family, quiet)
    a, sr = wav.read(plain_out["path"])
    b, _ = wav.read(quiet_out["path"])
    assert abs(a.shape[1] / sr - 2.0) < 0.5 and abs(b.shape[1] / sr - 2.0) < 0.5
    assert rms(a) > 0.0 and rms(b) < rms(a)  # the command ran: the demo's kick enters at 9.2 s, so turning it down makes the bounce quieter
    assert quiet_out["extra"]["commands"] == 1 and plain_out["extra"]["invocation"] in ("script", "cmd-file")
    bad = make_job(tmp_path / "bad", "render.session", "soundcraft", {"commands": [{"command": "no.such.command"}]}, {"session": str(archive)})
    with pytest.raises(up.ProcessError, match="no.such.command"):
        run_job(family, bad)
