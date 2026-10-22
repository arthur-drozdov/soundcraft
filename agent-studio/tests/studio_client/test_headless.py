"""HeadlessMcp against a fake MCP server (built on the mcp SDK) run as a subprocess."""

from __future__ import annotations

import stat
import sys
import threading
from pathlib import Path

import pytest

from studio_client import HeadlessMcp, Studio, StudioError

SERVER = Path(__file__).with_name("fake_mcp_server.py")


@pytest.fixture
def binary(tmp_path: Path) -> str:
    """A stand-in for `soundcraft-cli`: `<binary> mcp [args]` runs the fake server."""
    wrapper = tmp_path / "fake-soundcraft-cli"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{SERVER}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return str(wrapper)


@pytest.fixture
def studio(binary: str):
    s = Studio.headless(binary, ["--demo"], start_timeout=60, timeout=30)
    yield s
    s.close()


def test_execute_inspect_commands_bounce(studio: Studio) -> None:
    assert studio.kind == "headless"
    assert studio.execute("mix.volume", {"track": "Kick", "db": -6}) == {"track": "Kick", "db": -6}
    assert studio.inspect("full") == {"detail": "full", "volumes": {"Kick": -6}}
    assert studio.commands("mix") == [{"id": "mix.volume"}]
    assert studio.bounce_mix("/tmp/out.wav", format="wav")["lufs"] == -14.0


def test_args_are_passed_after_mcp(studio: Studio) -> None:
    assert studio.execute("args.echo") == {"argv": ["mcp", "--demo"]}


def test_command_error_raises_studio_error(studio: Studio) -> None:
    with pytest.raises(StudioError, match="no track named `Nope`"):
        studio.execute("mix.volume", {"track": "Nope", "db": 0})
    assert studio.execute("mix.volume", {"track": "Bass", "db": -1})["track"] == "Bass"  # still usable


def test_batch_success_and_failure(studio: Studio) -> None:
    ok = studio.batch([("mix.volume", {"track": "Kick", "db": -3}), {"command": "mix.volume", "params": {"track": "Bass", "db": -4}}])
    assert [r["ok"] for r in ok] == [True, True]

    calls = [("mix.volume", {"track": "Kick", "db": -9}), ("mix.volume", {"track": "Nope", "db": 0}), ("mix.volume", {"track": "Bass", "db": -9})]
    with pytest.raises(StudioError, match="batch stopped at `mix.volume`") as info:
        studio.batch(calls)
    assert [r["ok"] for r in info.value.results or []] == [True, False]

    results = studio.batch(calls, keep_going=True)
    assert [r["ok"] for r in results] == [True, False, True]


def test_oversize_batch_is_rejected_client_side(studio: Studio) -> None:
    with pytest.raises(StudioError, match="at most 1000"):
        studio.batch([("mix.volume", {})] * 1001)


def test_screenshot_needs_the_desktop_app(studio: Studio) -> None:
    with pytest.raises(StudioError, match="desktop app"):
        studio.screenshot("/tmp/x.png")


def test_timeout_raises_but_server_survives(binary: str) -> None:
    with Studio.headless(binary, start_timeout=60) as s:
        with pytest.raises(StudioError, match="did not finish within"):
            s.transport.execute("sleep", {"s": 5}, timeout=0.3)  # type: ignore[attr-defined]
        assert s.inspect()["detail"] == "summary"


def test_calls_from_many_threads(studio: Studio) -> None:
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            for _ in range(5):
                assert studio.inspect()["detail"] == "summary"
        except BaseException as exc:  # noqa: BLE001 - collected and re-asserted below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_server_death_raises_and_is_not_restarted(binary: str) -> None:
    with Studio.headless(binary, start_timeout=60, timeout=15) as s:
        s.inspect()
        with pytest.raises(StudioError):
            s.execute("crash")
        with pytest.raises(StudioError):
            s.inspect()


def test_missing_binary_is_a_studio_error(tmp_path: Path) -> None:
    s = Studio.headless(str(tmp_path / "does-not-exist"), start_timeout=10)
    with pytest.raises(StudioError, match="cannot start"):
        s.inspect()
    s.close()


def test_closed_studio_refuses_calls_and_close_is_idempotent(binary: str) -> None:
    s = Studio.headless(binary, start_timeout=60)
    s.inspect()
    s.close()
    s.close()
    with pytest.raises(StudioError, match="closed"):
        s.inspect()


def test_close_without_use_does_not_spawn(binary: str) -> None:
    h = HeadlessMcp(binary)
    h.close()
    assert h._thread is None


def test_argument_validation() -> None:
    with pytest.raises(StudioError):
        HeadlessMcp("")
    with pytest.raises(StudioError):
        HeadlessMcp("x", "--demo")  # type: ignore[arg-type]
    with pytest.raises(StudioError):
        HeadlessMcp("x", timeout=float("nan"))
