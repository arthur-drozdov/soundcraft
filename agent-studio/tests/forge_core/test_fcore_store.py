from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forge.errors import BadRequest, NotFound, ValidationFailed
from forge.store import AssetStore, file_asset_id, json_safe

from fcore_helpers import make_zip, melody, write_wav


@pytest.fixture
def store(tmp_path: Path) -> AssetStore:
    s = AssetStore(tmp_path / "data")
    yield s  # type: ignore[misc]
    s.close()


def add(store: Path | AssetStore, tmp_path: Path, name: str, payload: bytes, **side: Any) -> dict[str, Any]:
    assert isinstance(store, AssetStore)
    src = tmp_path / name
    src.write_bytes(payload)
    base = {"type": "generate.loop", "kind": "audio", "model": "testtone", "job_id": "j_1", "prompt": "warm pad"}
    base.update(side)
    return store.add_asset(src, base)


def test_id_layout_and_sidecar(store: AssetStore, tmp_path: Path) -> None:
    payload = b"RIFFdata-1"
    side = add(store, tmp_path, "x.wav", payload, created_at="2026-03-05T10:00:00Z")
    digest = hashlib.sha256(payload).hexdigest()
    assert side["asset_id"] == "a_" + digest[:24]
    path = store.path(side["asset_id"])
    assert path == store.root / "2026" / "03" / f"{side['asset_id']}.wav"
    assert path.read_bytes() == payload and not (tmp_path / "x.wav").exists()  # moved in
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar == store.meta(side["asset_id"]) == side
    assert sidecar["created_at"] == "2026-03-05T10:00:00Z" and sidecar["rating"] is None and sidecar["size_bytes"] == len(payload)
    assert file_asset_id(path) == side["asset_id"]


def test_same_bytes_are_one_asset(store: AssetStore, tmp_path: Path) -> None:
    a = add(store, tmp_path, "a.wav", b"same", job_id="j_1")
    b = add(store, tmp_path, "b.wav", b"same", job_id="j_2")
    assert a["asset_id"] == b["asset_id"] and b["job_id"] == "j_1" and store.count() == 1
    assert not (tmp_path / "b.wav").exists()


def test_unsupported_extension_and_bad_ids(store: AssetStore, tmp_path: Path) -> None:
    with pytest.raises(BadRequest):
        add(store, tmp_path, "x.exe", b"1")
    for bad in ["../../etc/passwd", "a_" + "0" * 23, "a_" + "G" * 24, "", "a_" + "0" * 24 + "/..", None, 5]:
        with pytest.raises(NotFound):
            store.path(bad)  # type: ignore[arg-type]
    with pytest.raises(NotFound):
        store.meta("a_" + "0" * 24)
    assert not store.exists("../x")


def test_search_filters(store: AssetStore, tmp_path: Path) -> None:
    a = add(store, tmp_path, "a.wav", b"a", prompt="Warm analog PAD", tags={"family": "synth", "descriptors": ["warm", "wide"]},
            created_at="2026-01-01T00:00:00Z")
    b = add(store, tmp_path, "b.wav", b"b", prompt="plucky_bass 100%", tags={"family": "bass"}, model="sa3", type="generate.sample",
            job_id="j_2", created_at="2026-02-01T00:00:00Z")
    c = add(store, tmp_path, "c.mid", b"c", prompt="", kind="midi", type="transcribe.melody", model="yin", job_id="j_3",
            created_at="2026-03-01T00:00:00Z")
    ids = lambda **kw: [r["asset_id"] for r in store.search(**kw)]  # noqa: E731
    assert ids() == [c["asset_id"], b["asset_id"], a["asset_id"]]  # newest first
    assert ids(tags={"family": "synth"}) == [a["asset_id"]]
    assert ids(tags={"FAMILY": "SYNTH"}) == [a["asset_id"]]
    assert ids(tags={"descriptors": "wide"}) == [a["asset_id"]]
    assert ids(tags={"family": ["synth", "bass"]}) == [b["asset_id"], a["asset_id"]]
    assert ids(tags={"family": "synth", "descriptors": "nope"}) == []
    assert ids(text="analog pad") == [a["asset_id"]]
    assert ids(text="100%") == [b["asset_id"]] and ids(text="%") == [b["asset_id"]] and ids(text="_") == [b["asset_id"]]
    assert ids(type="transcribe.melody") == [c["asset_id"]]
    assert ids(model="sa3") == [b["asset_id"]] and ids(job_id="j_3") == [c["asset_id"]]
    assert ids(since="2026-02-01T00:00:00Z") == [c["asset_id"], b["asset_id"]]
    assert ids(since="2026-02-15") == [c["asset_id"]]
    assert ids(limit=1) == [c["asset_id"]] and len(store.search(limit=10**9)) == 3 and len(store.search(limit=-5)) == 1
    with pytest.raises(ValidationFailed):
        store.search(since="yesterday-ish")
    store.rate(a["asset_id"], 4)
    store.rate(b["asset_id"], 0)
    assert ids(rated=True) == [b["asset_id"], a["asset_id"]] and ids(rated=False) == [c["asset_id"]]
    assert ids(min_rating=3) == [a["asset_id"]] and ids(min_rating=0, rated=True) == [b["asset_id"], a["asset_id"]]
    assert store.search(type="generate.loop")[0]["tags"]["family"] == "synth"


def test_rate_updates_sidecar_and_validates(store: AssetStore, tmp_path: Path) -> None:
    a = add(store, tmp_path, "a.wav", b"a")
    updated = store.rate(a["asset_id"], 5, "perfect")
    assert (updated["rating"], updated["note"]) == (5, "perfect")
    assert store.meta(a["asset_id"])["rating"] == 5
    assert store.rate(a["asset_id"], 2)["note"] == "perfect"  # note kept when not given
    for bad in (6, -1, 2.5, True, "3"):
        with pytest.raises(ValidationFailed):
            store.rate(a["asset_id"], bad)  # type: ignore[arg-type]
    with pytest.raises(ValidationFailed):
        store.rate(a["asset_id"], 1, "x" * 2001)
    with pytest.raises(NotFound):
        store.rate("a_" + "1" * 24, 3)


def test_purge_keeps_rated_and_used_and_recent(store: AssetStore, tmp_path: Path) -> None:
    old = "2026-01-01T00:00:00Z"
    unrated = add(store, tmp_path, "1.wav", b"1", created_at=old)
    rated = add(store, tmp_path, "2.wav", b"2", created_at=old)
    used = add(store, tmp_path, "3.wav", b"3", created_at=old)
    fresh = add(store, tmp_path, "4.wav", b"4")
    store.rate(rated["asset_id"], 3)
    store.mark_used(used["asset_id"])
    removed = store.purge(30)
    assert removed == [unrated["asset_id"]]
    assert not store.exists(unrated["asset_id"]) and store.exists(rated["asset_id"]) and store.exists(fresh["asset_id"])
    assert not list(store.root.rglob(f"{unrated['asset_id']}.*"))  # file and sidecar are gone
    later = datetime.now(timezone.utc) + timedelta(days=31)
    assert set(store.purge(30, now=later)) == {used["asset_id"], fresh["asset_id"]}
    assert [r["asset_id"] for r in store.search()] == [rated["asset_id"]]


def test_register_wav_upload(store: AssetStore, tmp_path: Path) -> None:
    src = write_wav(tmp_path / "in.wav", melody([60, 64]))
    side = store.register_upload(src, original_name="in.wav", max_bytes=10**7, max_seconds=60)
    assert side["kind"] == "audio" and side["type"] == "upload" and side["original_name"] == "in.wav"
    assert side["sample_rate"] == 44100 and side["channels"] == 1 and side["measured"]["duration_s"] == pytest.approx(1.0, abs=0.01)
    assert store.path(side["asset_id"]).read_bytes() == src.read_bytes() and src.exists()
    assert store.search(type="upload")[0]["asset_id"] == side["asset_id"]


def test_upload_rejections(store: AssetStore, tmp_path: Path) -> None:
    good = write_wav(tmp_path / "g.wav", np.zeros((1, 44100), np.float32))
    with pytest.raises(BadRequest) as big:
        store.register_upload(good, max_bytes=100, max_seconds=60)
    assert big.value.status == 413
    with pytest.raises(BadRequest) as long:
        store.register_upload(good, max_bytes=10**7, max_seconds=0.5)
    assert long.value.status == 413
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio at all")
    with pytest.raises(BadRequest, match="WAV"):
        store.register_upload(junk, max_bytes=10**7, max_seconds=60)
    trunc = tmp_path / "trunc.wav"
    trunc.write_bytes(good.read_bytes()[:30])
    with pytest.raises(BadRequest):
        store.register_upload(trunc, max_bytes=10**7, max_seconds=60)
    with pytest.raises(BadRequest):
        store.register_upload(tmp_path / "missing.wav", max_bytes=10**7, max_seconds=60)
    assert store.count() == 0


def test_session_zip_uploads(store: AssetStore, tmp_path: Path) -> None:
    ok = make_zip(tmp_path / "ok.zip", ["My Session/a.scraft", "My Session/audio/take.wav"])
    side = store.register_upload(ok, max_bytes=10**7, max_seconds=60)
    assert side["kind"] == "session" and store.path(side["asset_id"]).suffix == ".zip"
    for name, entries in {"none": ["a.txt"], "two": ["a.scraft", "b.scraft"], "slip": ["../evil.scraft"], "abs": ["/etc/x.scraft"]}.items():
        with pytest.raises(BadRequest):
            store.register_upload(make_zip(tmp_path / f"{name}.zip", entries), max_bytes=10**7, max_seconds=60)
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"PK\x03\x04garbage")
    with pytest.raises(BadRequest):
        store.register_upload(bad, max_bytes=10**7, max_seconds=60)


def test_json_safe() -> None:
    out = json_safe({"a": float("nan"), "b": [np.float32(1.5), np.int64(3)], "c": (1, 2), 4: float("inf")})
    assert out == {"a": None, "b": [1.5, 3], "c": [1, 2], "4": None}
    json.dumps(out, allow_nan=False)


def test_index_survives_reopen(tmp_path: Path) -> None:
    s1 = AssetStore(tmp_path / "data")
    a = add(s1, tmp_path, "a.wav", b"a", tags={"k": "v"})
    s1.close()
    s2 = AssetStore(tmp_path / "data")
    assert [r["asset_id"] for r in s2.search(tags={"k": "v"})] == [a["asset_id"]]
    s2.close()
