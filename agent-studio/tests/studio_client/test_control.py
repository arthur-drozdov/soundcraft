"""ControlChannel against a fake TCP server speaking the control protocol."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from studio_client import CommandError, ControlChannel, Studio, StudioError

Handler = Callable[[dict[str, Any]], dict[str, Any] | None]


class FakeApp:
    """Accepts connections; ``handler(request)`` returns the reply body (without id) or None to hang up."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[dict[str, Any]] = []
        self.connections = 0
        self.close_after_reply = False
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(8)
        self.port: int = self._srv.getsockname()[1]
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        with conn, conn.makefile("rwb") as f:
            for line in f:
                req = json.loads(line)
                self.requests.append(req)
                body = self.handler(req)
                if body is None:
                    return
                f.write((json.dumps({"id": req["id"], **body}) + "\n").encode())
                f.flush()
                if self.close_after_reply:
                    return

    def stop(self) -> None:
        self._stop = True
        self._srv.close()


def echo(req: dict[str, Any]) -> dict[str, Any]:
    if req["method"] == "engine.execute" and req["params"]["command"] == "mix.volume" and req["params"]["params"].get("track") == "Nope":
        return {"ok": False, "error": "bad parameters for `mix.volume`: no track named `Nope`"}
    return {"ok": True, "result": {"method": req["method"], "params": req["params"]}}


@pytest.fixture
def app():
    a = FakeApp(echo)
    yield a
    a.stop()


def test_execute_sends_engine_execute_with_incrementing_ids(app: FakeApp) -> None:
    with ControlChannel(app.port) as ch:
        r1 = ch.execute("track.new", {"count": 2})
        r2 = ch.execute("transport.play")
    assert r1 == {"method": "engine.execute", "params": {"command": "track.new", "params": {"count": 2}}}
    assert r2["params"] == {"command": "transport.play", "params": {}}
    assert [r["id"] for r in app.requests] == [1, 2]
    assert app.connections == 1


def test_error_reply_raises_command_error(app: FakeApp) -> None:
    with ControlChannel(app.port) as ch:
        with pytest.raises(StudioError, match="no track named `Nope`") as info:
            ch.execute("mix.volume", {"track": "Nope", "db": -6})
        assert isinstance(info.value, CommandError)
        assert ch.execute("transport.stop")  # channel still usable after an error


def test_inspect_commands_screenshot_methods(app: FakeApp) -> None:
    with Studio.connect(app.port) as studio:
        assert studio.kind == "control"
        assert studio.inspect("full")["method"] == "session.inspect"
        assert studio.commands("mix.")["params"] == {"filter": "mix."}
        assert studio.commands()["params"] == {}
        assert studio.screenshot("/tmp/x.png")["params"] == {"path": "/tmp/x.png"}
    assert [r["method"] for r in app.requests] == ["session.inspect", "engine.commands", "engine.commands", "ui.screenshot"]
    with pytest.raises(StudioError):
        ControlChannel(app.port).inspect("everything")


def test_batch_stops_at_first_error_by_default(app: FakeApp) -> None:
    calls = [("a.one", None), ("mix.volume", {"track": "Nope"}), ("a.three", {})]
    with ControlChannel(app.port) as ch:
        with pytest.raises(StudioError, match="batch stopped at `mix.volume`") as info:
            ch.batch(calls)
    assert [r["ok"] for r in info.value.results or []] == [True, False]
    assert len(app.requests) == 2


def test_batch_keep_going_returns_every_result(app: FakeApp) -> None:
    calls = [{"command": "a.one"}, {"command": "mix.volume", "params": {"track": "Nope"}}, {"command": "a.three"}]
    with ControlChannel(app.port) as ch:
        results = ch.batch(calls, keep_going=True)
    assert [r["ok"] for r in results] == [True, False, True]
    assert "no track named" in results[1]["error"]


def test_batch_rejects_oversize_and_bad_calls_before_sending(app: FakeApp) -> None:
    with ControlChannel(app.port) as ch:
        with pytest.raises(StudioError, match="at most 1000"):
            ch.batch([("x.y", None)] * 1001)
        with pytest.raises(StudioError):
            ch.batch([{"params": {}}])
    assert app.requests == []


def test_non_finite_numbers_are_refused_locally(app: FakeApp) -> None:
    with ControlChannel(app.port) as ch:
        with pytest.raises(StudioError, match="not valid JSON"):
            ch.execute("mix.volume", {"track": "A", "db": float("nan")})
    assert app.requests == []


def test_reconnects_after_the_app_closes_the_connection(app: FakeApp) -> None:
    app.close_after_reply = True
    with ControlChannel(app.port) as ch:
        ch.execute("a.one")
        deadline = time.time() + 2
        while ch.connected and time.time() < deadline:
            time.sleep(0.02)
            if ch._stale():  # the FIN has arrived
                break
        assert ch.execute("a.two")["params"]["command"] == "a.two"
    assert app.connections == 2


def test_connection_lost_mid_request_raises_then_recovers() -> None:
    dropped = {"n": 0}

    def handler(req: dict[str, Any]) -> dict[str, Any] | None:
        if req["params"].get("command") == "boom" and dropped["n"] == 0:
            dropped["n"] += 1
            return None  # hang up without replying
        return {"ok": True, "result": "fine"}

    fake = FakeApp(handler)
    try:
        with ControlChannel(fake.port) as ch:
            with pytest.raises(StudioError, match="connection to SoundCraft lost"):
                ch.execute("boom")
            assert not ch.connected
            assert ch.execute("ok") == "fine"
        assert len([r for r in fake.requests if r["params"].get("command") == "boom"]) == 1  # never resent
    finally:
        fake.stop()


def test_read_only_requests_are_resent_once_after_connection_loss() -> None:
    state = {"n": 0}

    def handler(req: dict[str, Any]) -> dict[str, Any] | None:
        state["n"] += 1
        if state["n"] == 1:
            return None
        return {"ok": True, "result": {"tracks": []}}

    fake = FakeApp(handler)
    try:
        with ControlChannel(fake.port) as ch:
            assert ch.inspect() == {"tracks": []}
        assert fake.connections == 2
    finally:
        fake.stop()


def test_mismatched_reply_id_is_a_protocol_error() -> None:
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve() -> None:
        conn, _ = srv.accept()
        with conn, conn.makefile("rwb") as f:
            f.readline()
            f.write(b'{"id": 99, "ok": true, "result": 1}\n')
            f.flush()

    threading.Thread(target=serve, daemon=True).start()
    with ControlChannel(srv.getsockname()[1]) as ch:
        with pytest.raises(StudioError, match="does not match request id"):
            ch.execute("a.one")
    srv.close()


def test_malformed_and_oversize_replies_are_rejected() -> None:
    replies = iter([b"not json\n", b'{"id": 2, "ok": true, "result": "' + b"x" * 5000 + b'"}\n'])
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)

    def serve() -> None:
        for reply in replies:
            conn, _ = srv.accept()
            with conn, conn.makefile("rwb") as f:
                f.readline()
                f.write(reply)
                f.flush()

    threading.Thread(target=serve, daemon=True).start()
    port = srv.getsockname()[1]
    with ControlChannel(port) as ch:
        with pytest.raises(StudioError, match="malformed reply"):
            ch.execute("a.one")
        with pytest.raises(StudioError, match="exceeds 1024 bytes"):
            ControlChannel(port, max_reply_bytes=1024).execute("a.two")
    srv.close()


def test_timeout_raises_and_drops_connection() -> None:
    fake = FakeApp(lambda req: (time.sleep(0.5), {"ok": True, "result": 1})[1])
    try:
        with ControlChannel(fake.port, timeout=0.1) as ch:
            with pytest.raises(StudioError, match="no reply"):
                ch.execute("slow")
            assert not ch.connected
    finally:
        fake.stop()


def test_connect_failure_and_argument_validation() -> None:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    free = s.getsockname()[1]
    s.close()
    with pytest.raises(StudioError, match="cannot connect"):
        ControlChannel(free, connect_timeout=1).execute("a.one")
    for bad in (0, 70000, "7801", True):
        with pytest.raises(StudioError):
            ControlChannel(bad)  # type: ignore[arg-type]
    with pytest.raises(StudioError, match="non-loopback"):
        ControlChannel(7801, host="10.0.0.5")
    ControlChannel(7801, host="10.0.0.5", allow_remote=True)


def test_closed_channel_refuses_requests(app: FakeApp) -> None:
    ch = ControlChannel(app.port)
    ch.close()
    with pytest.raises(StudioError, match="closed"):
        ch.execute("a.one")
