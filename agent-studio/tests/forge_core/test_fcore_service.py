"""End-to-end: submit jobs to the service and check assets, sidecars, queueing, cancellation and recovery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from audiokit import midi, wav
from forge.errors import BadRequest, Conflict, NotFound, ValidationFailed
from forge.jobdb import JobDB
from forge.jobs import JobRecord
from forge.service import ForgeService
from forge.store import utc_now

from fcore_helpers import clicks, make_config, make_service, make_zip, melody, wait_for, wait_state, write_wav

SLOW = {"x-delay_s": "30"}


async def upload_wav(svc: ForgeService, tmp_path: Path, audio: Any, name: str = "up.wav") -> str:
    res = await svc.upload(write_wav(tmp_path / name, audio), name)
    assert res["kind"] == "audio"
    return res["asset_id"]


async def run_ok(svc: ForgeService, *args: Any, **kw: Any) -> JobRecord:
    job = await svc.submit(*args, **kw)
    done = await wait_for(svc, job.job_id)
    assert done.state == "succeeded", done.error
    return done


async def test_generate_loop_end_to_end(service: ForgeService) -> None:
    done = await run_ok(service, "generate.loop", {"bpm": 120, "bars": 4, "key": "F minor", "candidates": 2, "prompt": "warm pad", "tags": {"family": "synth"}, "seed": 7})
    assert done.result is not None and len(done.result.asset_ids) == 2 and done.progress == 1.0 and done.queue_position is None
    assert done.params["seed"] == 7 and done.model == "testtone" and done.lane == "gpu"
    for candidate, asset_id in enumerate(done.result.asset_ids):
        path = service.asset_path(asset_id)
        assert asset_id == "a_" + hashlib.sha256(path.read_bytes()).hexdigest()[:24]
        assert path.parent.parent.parent == service.store.root and path.suffix == ".wav" and path.with_suffix(".json").is_file()
        side = service.asset_meta(asset_id)
        assert side["asset_id"] == asset_id and side["job_id"] == done.job_id and side["type"] == "generate.loop" and side["kind"] == "audio"
        assert side["model"] == side["family"] == "testtone" and side["checkpoint"] == "builtin:testtone" and side["checkpoint_hash"]
        assert "test generator" in side["licence"] and side["prompt"] == "warm pad" and side["tags"] == {"family": "synth"}
        assert side["seed"] == 7 + candidate and side["candidate"] == candidate and side["parents"] == []
        assert side["requested"] == {"bpm": 120.0, "key": "F minor", "note": None}
        m = side["measured"]
        assert m["bpm"] == pytest.approx(120, abs=2) and m["key"] == "F minor" and m["duration_s"] == pytest.approx(8.0, abs=0.01)
        assert m["peak_dbfs"] == pytest.approx(-1.0, abs=0.1)
        assert {"bpm_confidence", "key_confidence", "onsets", "true_peak_dbfs", "lufs", "rms_dbfs", "centroid_hz", "clipped_samples", "dc_offset", "loop_seam"} <= set(m)
        assert side["warnings"] == [] and side["rating"] is None and side["note"] is None
        assert (side["sample_rate"], side["channels"], side["bit_depth"]) == (48000, 2, "24")
        assert datetime.fromisoformat(side["created_at"].replace("Z", "+00:00")).tzinfo is not None
        audio, sr = wav.read(path)
        assert sr == 48000 and audio.shape == (2, 8 * 48000)
        json.dumps(side, allow_nan=False)
    first, second = (service.asset_path(a).read_bytes() for a in done.result.asset_ids)
    assert first != second
    assert {r["asset_id"] for r in service.search_assets(tags={"family": "synth"})} == set(done.result.asset_ids)
    assert service.search_assets(text="warm", job_id=done.job_id)
    assert not list((service.config.data_dir / "scratch").glob("*"))  # scratch space is cleaned up


async def test_requested_versus_measured_mismatch_is_flagged(service: ForgeService) -> None:
    done = await run_ok(service, "generate.loop", {"bpm": 100, "bars": 2, "tags": {"x-fault": "off_tempo"}})
    assert done.result is not None
    side = service.asset_meta(done.result.asset_ids[0])
    assert side["requested"]["bpm"] == 100.0 and side["measured"]["bpm"] == pytest.approx(150, abs=3)
    assert any(w.startswith("bpm requested 100, measured 15") for w in side["warnings"])
    assert done.result.warnings == side["warnings"]


async def test_sample_pitch_and_wrong_pitch_warning(service: ForgeService) -> None:
    ok = await run_ok(service, "generate.sample", {"note": 60, "length_s": 1.0, "post": {"bit_depth": "16"}, "channels": 1, "sample_rate": 32000})
    assert ok.result is not None
    side = service.asset_meta(ok.result.asset_ids[0])
    assert side["requested"]["note"] == 60 and side["measured"]["note"] == pytest.approx(60, abs=0.2) and side["warnings"] == []
    assert (side["sample_rate"], side["channels"], side["bit_depth"]) == (32000, 1, "16")
    audio, sr = wav.read(service.asset_path(ok.result.asset_ids[0]))
    assert sr == 32000 and audio.shape[0] == 1
    bad = await run_ok(service, "generate.sample", {"note": 60, "length_s": 1.0, "tags": {"x-fault": "wrong_pitch"}})
    assert bad.result is not None
    assert service.asset_meta(bad.result.asset_ids[0])["warnings"] == ["note requested 60, measured 63.0"]


async def test_silent_output_is_flagged(service: ForgeService) -> None:
    done = await run_ok(service, "generate.sfx", {"duration_s": 0.5, "tags": {"x-fault": "silent"}})
    assert done.result is not None and "output is silent" in service.asset_meta(done.result.asset_ids[0])["warnings"]


async def test_generate_instrument_bundle(service: ForgeService) -> None:
    done = await run_ok(service, "generate.instrument", {"name": "Test Keys", "note_low": 48, "note_high": 71, "step": 6, "velocity_layers": 2, "length_s": 0.6})
    assert done.result is not None and len(done.result.asset_ids) == 1
    asset_id = done.result.asset_ids[0]
    path = service.asset_path(asset_id)
    assert path.suffix == ".zip" and asset_id == "a_" + hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    side = service.asset_meta(asset_id)
    assert side["kind"] == "instrument" and side["name"] == "Test_Keys" and side["warnings"] == []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        manifest = json.loads(z.read("manifest.json"))
        sfz = z.read("Test_Keys.sfz").decode()
        preset = ET.fromstring(z.read("Test_Keys.dspreset"))
        samples = sorted(n for n in names if n.startswith("samples/"))
        assert len(samples) == 8 and {"Test_Keys.sfz", "Test_Keys.dspreset", "manifest.json"} <= set(names)
        for n in samples:
            z.extract(n, path.parent / "unz")
    assert [z["root_note"] for z in manifest["zones"][::2]] == [50, 56, 62, 68]
    covered = sorted({k for z in manifest["zones"][::2] for k in range(z["key_low"], z["key_high"] + 1)})
    assert covered == list(range(48, 72))  # every key exactly once
    assert sfz.count("<region>") == 8 and len(list(preset.iter("sample"))) == 8
    assert [f["sample"] for f in side["files"]] == [z["sample"] for z in manifest["zones"]]
    for f in side["files"]:
        assert abs(f["measured"]["note"] - f["root_note"]) < 0.3
        audio, sr = wav.read(path.parent / "unz" / f["sample"])
        assert sr == 48000 and audio.shape[0] == 2 and audio.shape[1] == pytest.approx(0.6 * 48000, abs=2)


async def test_transcribe_melody_end_to_end(service: ForgeService, tmp_path: Path) -> None:
    notes = [60, 64, 67, 72]
    source = await upload_wav(service, tmp_path, melody(notes, 0.4, 0.1))
    done = await run_ok(service, "transcribe.melody", {"bpm": 100}, inputs={"audio": source})
    assert done.model == "yin" and done.lane == "cpu" and done.result is not None
    side = service.asset_meta(done.result.asset_ids[0])
    path = service.asset_path(done.result.asset_ids[0])
    assert path.suffix == ".mid" and side["kind"] == "midi" and side["parents"] == [source]
    assert [n["pitch"] for n in side["notes"]] == notes and side["measured"]["note_count"] == 4 and side["tempo_bpm"] == 100.0
    assert [n.pitch for n in midi.read_smf(path)] == notes
    assert source in {a["asset_id"] for a in service.search_assets(type="upload")}


async def test_analyze_audio_runs_in_process_on_the_cpu_lane(service: ForgeService, tmp_path: Path) -> None:
    source = await upload_wav(service, tmp_path, clicks(120, 8.0))
    queued = await service.submit("analyze.audio", inputs={"audio": source})
    assert queued.model == "analysis" and queued.family == "analysis" and queued.lane == "cpu"
    done = await wait_for(service, queued.job_id)
    assert done.state == "succeeded" and done.result is not None and done.result.asset_ids == []
    a = done.result.analysis
    assert a is not None and a["bpm"] == pytest.approx(120, abs=2) and a["duration_s"] == pytest.approx(8.0, abs=0.01)
    assert {"peak_dbfs", "lufs", "onsets_s", "key", "centroid_hz", "loop_seam"} <= set(a)
    assert service.get_job(done.job_id).result == done.result  # persisted
    assert service.manager.model("analysis") is None and "analysis" in {m["id"] for m in service.models()}


async def test_input_assets_chain_and_record_parents(service: ForgeService, tmp_path: Path) -> None:
    base = await run_ok(service, "generate.sfx", {"duration_s": 1.0})
    assert base.result is not None
    src = base.result.asset_ids[0]
    for job_type, params in (("transform.audio", {"strength": 0.7}), ("extend.audio", {"seconds": 2}), ("inpaint.audio", {"start_s": 0.2, "end_s": 0.5}), ("layer.add", {"description": "airy pad"})):
        done = await run_ok(service, job_type, params, inputs={"audio": src})
        assert done.result is not None
        side = service.asset_meta(done.result.asset_ids[0])
        assert side["parents"] == [src] and side["type"] == job_type
        if job_type == "extend.audio":
            assert side["measured"]["duration_s"] == pytest.approx(3.0, abs=0.01)


async def test_submit_validation_errors_carry_field_paths(service: ForgeService, tmp_path: Path) -> None:
    with pytest.raises(ValidationFailed) as exc:
        await service.submit("generate.loop", {"bpm": 999, "bars": 0})
    assert {e["path"] for e in exc.value.errors} == {"params.bpm", "params.bars"}
    with pytest.raises(ValidationFailed) as exc:
        await service.submit("generate.sfx", {}, priority="urgent", idempotency_key="")
    assert {e["path"] for e in exc.value.errors} == {"priority", "idempotency_key"}
    with pytest.raises(ValidationFailed) as exc:
        await service.submit("generate.sfx", {}, model="nope")
    assert [e["path"] for e in exc.value.errors] == ["model"]
    with pytest.raises(ValidationFailed) as exc:
        await service.submit("generate.sfx", {}, model="yin")
    assert [e["path"] for e in exc.value.errors] == ["model"]
    with pytest.raises(ValidationFailed) as exc:
        await service.submit("stems.split", {}, inputs={"audio": "a_" + "1" * 24})
    assert [e["path"] for e in exc.value.errors] == ["model"]  # no stems model is enabled
    with pytest.raises(ValidationFailed) as exc:
        await service.submit("transform.audio", {}, inputs={"audio": "a_" + "1" * 24})
    assert [e["path"] for e in exc.value.errors] == ["inputs.audio"]
    midi_job = await run_ok(service, "transcribe.melody", {}, inputs={"audio": await upload_wav(service, tmp_path, melody([60]))})
    assert midi_job.result is not None
    with pytest.raises(ValidationFailed, match="invalid request") as exc:
        await service.submit("transform.audio", {}, inputs={"audio": midi_job.result.asset_ids[0]})
    assert "not a audio asset" in exc.value.errors[0]["message"]
    assert service.jobdb.list() == [j for j in service.jobdb.list() if j.state == "succeeded"] and len(service.jobdb.list()) == 1


async def test_bad_worker_input_fails_the_job_not_the_service(service: ForgeService, tmp_path: Path) -> None:
    src = await upload_wav(service, tmp_path, melody([60]))
    done = await wait_for(service, (await service.submit("inpaint.audio", {"start_s": 50, "end_s": 60}, inputs={"audio": src})).job_id)
    assert done.state == "failed" and "outside the audio" in (done.error or "") and done.result is None
    assert not service.search_assets(job_id=done.job_id)
    await run_ok(service, "generate.sfx", {"duration_s": 0.2})


async def test_cancel_queued_job(service: ForgeService) -> None:
    blocker = await service.submit("generate.sfx", {"tags": SLOW})
    await wait_state(service, blocker.job_id, "running")
    queued = await service.submit("generate.sfx", {"duration_s": 0.2})
    assert queued.state == "queued" and queued.queue_position == 1
    cancelled = await service.cancel(queued.job_id)
    assert cancelled.state == "cancelled" and cancelled.queue_position is None and cancelled.finished_at
    assert service.get_job(blocker.job_id).state == "running"
    await service.cancel(blocker.job_id)
    await wait_for(service, blocker.job_id)
    await asyncio.sleep(0.2)
    final = service.get_job(queued.job_id)
    assert final.state == "cancelled" and final.started_at is None and final.result is None  # it never ran
    assert service.manager.family_status()["testtone"]["failures"] == 0


async def test_cancel_running_job_discards_output_and_frees_the_lane(service: ForgeService) -> None:
    running = await service.submit("generate.sfx", {"tags": SLOW})
    await wait_state(service, running.job_id, "running")
    after = await service.submit("generate.sfx", {"duration_s": 0.2})
    started = asyncio.get_running_loop().time()
    await service.cancel(running.job_id)
    done = await wait_for(service, running.job_id, 10)
    assert done.state == "cancelled" and done.result is None and done.error is None
    assert asyncio.get_running_loop().time() - started < 5
    assert (await wait_for(service, after.job_id)).state == "succeeded"
    assert not service.search_assets(job_id=running.job_id)
    assert (await service.cancel(done.job_id)).state == "cancelled"  # cancelling a finished job changes nothing
    with pytest.raises(NotFound):
        await service.cancel("j_nope")


async def test_cancel_running_analysis(service: ForgeService, tmp_path: Path) -> None:
    src = await upload_wav(service, tmp_path, clicks(120, 20.0))
    job = await service.submit("analyze.audio", inputs={"audio": src})
    await wait_state(service, job.job_id, "running")
    await service.cancel(job.job_id)
    done = await wait_for(service, job.job_id)
    assert done.state in ("cancelled", "succeeded")  # it may have finished before the cancel landed
    if done.state == "cancelled":
        assert done.result is None


async def test_worker_crash_fails_the_job_then_recovers(service: ForgeService) -> None:
    crashed = await wait_for(service, (await service.submit("generate.sfx", {"tags": {"x-fault": "crash"}})).job_id)
    assert crashed.state == "failed" and "worker died" in (crashed.error or "") and crashed.result is None
    nxt = await service.submit("generate.sfx", {"duration_s": 0.3})
    assert (await wait_for(service, nxt.job_id)).state == "succeeded"
    assert service.manager.family_status()["testtone"]["state"] == "ready"
    # the yin lane never noticed
    src = await upload_wav(service, Path(service.config.data_dir), melody([60, 62]), "m.wav")
    await run_ok(service, "transcribe.melody", {}, inputs={"audio": src})


async def test_failed_job_does_not_break_a_following_queued_job(service: ForgeService) -> None:
    jobs = [await service.submit("generate.sfx", {"tags": {"x-fault": "error"}}), await service.submit("generate.sfx", {"duration_s": 0.2})]
    states = [(await wait_for(service, j.job_id)).state for j in jobs]
    assert states == ["failed", "succeeded"]
    assert "injected failure" in (service.get_job(jobs[0].job_id).error or "")


async def test_priority_then_fifo_ordering(service: ForgeService) -> None:
    blocker = await service.submit("generate.sfx", {"tags": {"x-delay_s": "1.0"}})
    await wait_state(service, blocker.job_id, "running")
    order: list[str] = []

    async def collect() -> None:
        with service.subscribe() as sub:
            async for ev in sub:
                if ev["state"] == "running" and ev["job_id"] not in (blocker.job_id, *order):
                    order.append(ev["job_id"])

    collector = asyncio.create_task(collect())
    await asyncio.sleep(0)
    names = {}
    for name, prio in [("low", "low"), ("n1", "normal"), ("high", "high"), ("n2", "normal"), ("high2", "high")]:
        names[(await service.submit("generate.sfx", {"duration_s": 0.2}, priority=prio)).job_id] = name
    positions = {names[j.job_id]: j.queue_position for j in service.list_jobs(state="queued")}
    assert positions == {"high": 1, "high2": 2, "n1": 3, "n2": 4, "low": 5}
    for job_id in names:
        await wait_for(service, job_id, 30)
    collector.cancel()
    assert [names[j] for j in order] == ["high", "high2", "n1", "n2", "low"]


async def test_lanes_run_in_parallel_and_each_one_job_at_a_time(service: ForgeService, tmp_path: Path) -> None:
    src = await upload_wav(service, tmp_path, melody([60, 62, 64]))
    gpu1 = await service.submit("generate.sfx", {"tags": {"x-delay_s": "1.5"}})
    gpu2 = await service.submit("generate.sfx", {"duration_s": 0.2})
    cpu = await service.submit("transcribe.melody", {}, inputs={"audio": src})
    await wait_state(service, gpu1.job_id, "running")
    assert (await wait_for(service, cpu.job_id)).state == "succeeded"
    assert service.get_job(gpu1.job_id).state == "running" and service.get_job(gpu2.job_id).state == "queued"
    health = await service.health()
    assert health["lanes"] == {"gpu": {"running": 1, "queued": 1}, "cpu": {"running": 0, "queued": 0}}
    await wait_for(service, gpu2.job_id)


async def test_idempotency_keys(service: ForgeService) -> None:
    first = await service.submit("generate.sfx", {"duration_s": 0.3}, idempotency_key="abc")
    again = await service.submit("generate.sfx", {"duration_s": 0.3}, idempotency_key="abc")
    assert again.job_id == first.job_id
    other = await service.submit("generate.sfx", {"duration_s": 0.3}, idempotency_key="def")
    assert other.job_id != first.job_id
    done = await wait_for(service, first.job_id)
    await wait_for(service, other.job_id)
    after = await service.submit("generate.sfx", {"duration_s": 0.3}, idempotency_key="abc")
    assert after.job_id == first.job_id and after.state == "succeeded" and after.result == done.result
    assert len([j for j in service.list_jobs(limit=50)]) == 2
    # a key older than 24 h is free again
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).strftime("%Y-%m-%dT%H:%M:%SZ")
    service.jobdb._db.execute("UPDATE jobs SET created_at = ? WHERE job_id = ?", (old, first.job_id))
    fresh = await service.submit("generate.sfx", {"duration_s": 0.3}, idempotency_key="abc")
    assert fresh.job_id != first.job_id
    await wait_for(service, fresh.job_id)


async def test_events_report_progress_and_completion(service: ForgeService) -> None:
    with service.subscribe() as sub:
        job = await service.submit("generate.sfx", {"duration_s": 0.3, "candidates": 2})
        events = []
        while not events or events[-1]["state"] not in ("succeeded", "failed"):
            events.append(await asyncio.wait_for(sub.get(), 10))
    assert all(set(e) == {"job_id", "state", "progress", "message"} and e["job_id"] == job.job_id for e in events)
    assert [e["state"] for e in events][:2] == ["queued", "running"] and events[-1]["progress"] == 1.0
    progress = [e["progress"] for e in events]
    assert progress == sorted(progress) and any(0 < p < 1 for p in progress)


async def test_eta_uses_recent_durations(service: ForgeService) -> None:
    # the seeded job carries a delay so its measured duration is well above the
    # rounding floor on fast machines, otherwise the average is 0.0 and every eta is 0.0
    await run_ok(service, "generate.sfx", {"duration_s": 0.3, "tags": {"x-delay_s": "0.2"}})
    blocker = await service.submit("generate.sfx", {"tags": {"x-delay_s": "1.0"}})
    await wait_state(service, blocker.job_id, "running")
    queued = await service.submit("generate.sfx", {"duration_s": 0.2})
    assert queued.eta_s is not None and queued.eta_s > 0 and service.get_job(blocker.job_id).eta_s is not None
    await wait_for(service, queued.job_id)


async def test_job_listing_and_lookup(service: ForgeService) -> None:
    a = await run_ok(service, "generate.sfx", {"duration_s": 0.2})
    b = await service.submit("generate.sfx", {"tags": {"x-fault": "error"}})
    await wait_for(service, b.job_id)
    listed = service.list_jobs(limit=10)
    assert [j.job_id for j in listed] == [b.job_id, a.job_id]
    assert [j.job_id for j in service.list_jobs(state="failed")] == [b.job_id]
    assert len(service.list_jobs(limit=1)) == 1
    with pytest.raises(ValidationFailed):
        service.list_jobs(state="weird")
    with pytest.raises(NotFound):
        service.get_job("j_missing")


async def test_uploads(service: ForgeService, tmp_path: Path) -> None:
    ok = await service.upload(write_wav(tmp_path / "a.wav", melody([60])), "a.wav")
    assert ok["asset_id"].startswith("a_") and ok["kind"] == "audio"
    session = await service.upload(make_zip(tmp_path / "s.zip", ["proj/x.scraft"]), "s.zip")
    assert session["kind"] == "session"
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"hello world")
    with pytest.raises(BadRequest):
        await service.upload(junk)
    long = write_wav(tmp_path / "long.wav", melody([60] * 3, 1.0, 0.0))
    service.config.uploads.max_seconds = 2.0
    with pytest.raises(BadRequest):
        await service.upload(long)


async def test_ratings_search_and_purge(service: ForgeService) -> None:
    one = await run_ok(service, "generate.sfx", {"duration_s": 0.2, "seed": 1, "prompt": "kick"})
    two = await run_ok(service, "generate.sfx", {"duration_s": 0.2, "seed": 2, "prompt": "snare"})
    assert one.result is not None and two.result is not None
    keep, drop = one.result.asset_ids[0], two.result.asset_ids[0]
    assert service.rate_asset(keep, 5, "great")["rating"] == 5
    assert [a["asset_id"] for a in service.search_assets(rated=True, min_rating=4)] == [keep]
    assert await service.purge() == []  # nothing is old enough
    service.config.__dict__["retention_days"] = 0
    far_future = datetime.now(timezone.utc) + timedelta(days=1)
    assert service.store.purge(0, now=far_future) == [drop]
    assert [a["asset_id"] for a in service.search_assets()] == [keep]
    with pytest.raises(NotFound):
        service.asset_path(drop)


async def test_models_load_unload_and_health(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, memory_budget_gb=10, families={
        "testtone": {"python": __import__("sys").executable, "options": {"extra_models": [{"id": "tt-a", "vram_gb": 6}, {"id": "tt-b", "vram_gb": 6}, {"id": "tt-x", "vram_gb": 50}]}},
        "yin": {"python": __import__("sys").executable},
    })
    svc = make_service(cfg)
    await svc.start()
    try:
        assert {m["id"] for m in svc.models()} >= {"testtone", "yin", "analysis", "tt-a"}
        await svc.load_model("tt-a")
        await svc.load_model("tt-b")
        state = {m["id"]: m for m in svc.models()}
        assert state["tt-b"]["loaded"] and not state["tt-a"]["loaded"] and state["tt-b"]["warm"]
        health = await svc.health()
        assert health["status"] == "ok" and health["memory"] == {"budget_gb": 10, "used_gb": 6.0} and "available" in health["gpu"]
        assert set(health) >= {"version", "uptime_s", "lanes"}
        with pytest.raises(Conflict) as exc:
            await svc.load_model("tt-x")
        assert exc.value.status == 409
        await svc.unload_model("tt-b")
        assert (await svc.health())["memory"]["used_gb"] == 0
        done = await wait_for(svc, (await svc.submit("generate.sfx", {"duration_s": 0.2}, model="tt-a")).job_id)
        assert done.state == "succeeded" and done.model == "tt-a"
    finally:
        await svc.stop()


class _FakeNvidiaSmi:
    """Stands in for `nvidia-smi --query-gpu=name,memory.used,memory.total` output."""

    def __init__(self, out: bytes, returncode: int = 0) -> None:
        self._out, self.returncode = out, returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._out, b""


async def _health_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, out: bytes, returncode: int = 0, present: bool = True) -> dict[str, Any]:
    from forge import service as service_mod

    monkeypatch.setattr(service_mod.shutil, "which", lambda exe: "/usr/bin/nvidia-smi" if present else None)

    async def fake_exec(*args: Any, **kw: Any) -> _FakeNvidiaSmi:
        return _FakeNvidiaSmi(out, returncode)

    monkeypatch.setattr(service_mod.asyncio, "create_subprocess_exec", fake_exec)
    return (await make_service(make_config(tmp_path)).health())["gpu"]


async def test_health_gpu_without_memory_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """DGX Spark's GB10 (and other drivers) answer [N/A] for memory: the GPU is still there."""
    assert await _health_gpu(monkeypatch, tmp_path, b"NVIDIA GB10, [N/A], [N/A]\n") == {"available": True, "name": "NVIDIA GB10"}


async def test_health_gpu_with_memory_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    info = await _health_gpu(monkeypatch, tmp_path, b"NVIDIA GeForce RTX 4090, 2048, 24576\n")
    assert info == {"available": True, "name": "NVIDIA GeForce RTX 4090", "memory_used_gb": 2.0, "memory_total_gb": 24.0}


@pytest.mark.parametrize("out,returncode", [(b"", 0), (b"NVIDIA GB10, 1, 2\n", 1)])
async def test_health_gpu_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, out: bytes, returncode: int) -> None:
    assert await _health_gpu(monkeypatch, tmp_path, out, returncode) == {"available": False}
    assert await _health_gpu(monkeypatch, tmp_path, b"NVIDIA GB10, 1, 2\n", present=False) == {"available": False}


async def test_restart_marks_unfinished_jobs_failed(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    db = JobDB(cfg.data_dir)
    for job_id, state in (("j_running", "running"), ("j_queued", "queued"), ("j_done", "succeeded")):
        db.insert(JobRecord(job_id=job_id, type="generate.sfx", state=state, created_at=utc_now(), lane="gpu", model="testtone"))  # type: ignore[arg-type]
    db.close()
    svc = make_service(cfg)
    await svc.start()
    try:
        assert svc.get_job("j_running").state == svc.get_job("j_queued").state == "failed"
        assert svc.get_job("j_running").error == "forge restarted" and svc.get_job("j_done").state == "succeeded"
        assert svc.get_job("j_queued").queue_position is None
    finally:
        await svc.stop()


async def test_jobs_and_assets_persist_across_restarts(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    svc = make_service(cfg)
    await svc.start()
    done = await run_ok(svc, "generate.sfx", {"duration_s": 0.2, "prompt": "boom"})
    assert done.result is not None
    await svc.stop()
    svc2 = make_service(cfg)
    await svc2.start()
    try:
        again = svc2.get_job(done.job_id)
        assert again.state == "succeeded" and again.result == done.result and again.params == done.params
        assert svc2.asset_meta(done.result.asset_ids[0])["prompt"] == "boom"
        assert svc2.search_assets(text="boom")
    finally:
        await svc2.stop()


async def test_stop_fails_whatever_was_running(tmp_path: Path) -> None:
    svc = make_service(make_config(tmp_path))
    await svc.start()
    running = await svc.submit("generate.sfx", {"tags": SLOW})
    await wait_state(svc, running.job_id, "running")
    queued = await svc.submit("generate.sfx", {})
    await svc.stop()
    db = JobDB(svc.config.data_dir)
    assert {db.get(running.job_id).state, db.get(queued.job_id).state} == {"failed"}  # type: ignore[union-attr]
    assert db.get(queued.job_id).error == "forge stopped"  # type: ignore[union-attr]
    db.close()
