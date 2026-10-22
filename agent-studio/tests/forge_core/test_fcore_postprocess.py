from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from audiokit import dsp, midi, wav
from forge.jobs import validate_job
from forge.postprocess import (
    ModelMeta, PostContext, PostprocessError, compare, measure_note, process_audio, process_output, safe_name,
)

from fcore_helpers import SR, clicks, tone, write_wav

MODEL = ModelMeta("testtone", "testtone", "builtin", "abc", "CC0")


def ctx(job_type: str, params: dict[str, Any], inputs: dict[str, str] | None = None) -> PostContext:
    return PostContext("j_1", job_type, MODEL, validate_job(job_type, params, inputs).params, seed=10, parents=tuple((inputs or {}).values()))


def post(**kw: Any):  # type: ignore[no-untyped-def]
    return validate_job("generate.sfx", {"post": kw}).params.post


def test_pipeline_converts_format_and_normalises_peak() -> None:
    mono = tone(440, 1.0, amp=0.1)
    out, warnings = process_audio(mono, SR, post(), channels=2, sample_rate=48000)
    assert out.shape == (2, 48000) and warnings == []
    assert dsp.peak_dbfs(out) == pytest.approx(-1.0, abs=0.05)
    out, _ = process_audio(np.vstack([mono, mono]), SR, post(target=-6.0), channels=1, sample_rate=22050)
    assert out.shape[0] == 1 and out.shape[1] == 22050 and dsp.peak_dbfs(out) == pytest.approx(-6.0, abs=0.05)


def test_pipeline_normalise_none_and_lufs() -> None:
    quiet = (np.random.default_rng(1).standard_normal((2, SR)) * 0.02).astype(np.float32)
    same, _ = process_audio(quiet, SR, post(normalize="none"), channels=2, sample_rate=SR)
    assert np.allclose(same, quiet, atol=1e-6)
    loud, _ = process_audio(quiet, SR, post(normalize="lufs", target=-23.0), channels=2, sample_rate=SR)
    assert dsp.lufs_integrated(loud, SR) == pytest.approx(-23.0, abs=0.5)


def test_pipeline_trim_fades_and_loop_crossfade() -> None:
    body = tone(440, 0.5, amp=0.5)
    padded = np.concatenate([np.zeros(SR // 2, np.float32), body, np.zeros(SR // 2, np.float32)])[None, :]
    trimmed, _ = process_audio(padded, SR, post(trim_silence=True), channels=1, sample_rate=SR)
    assert 0.5 <= trimmed.shape[1] / SR < 0.6
    faded, _ = process_audio(np.ones((1, SR), np.float32), SR, post(fade_in_ms=100, fade_out_ms=100, normalize="none"), channels=1, sample_rate=SR)
    assert abs(faded[0, 0]) < 0.01 and abs(faded[0, -1]) < 0.01 and faded[0, SR // 2] == pytest.approx(1.0)
    looped, _ = process_audio(np.ones((1, SR), np.float32), SR, post(loop_crossfade_ms=100, normalize="none"), channels=1, sample_rate=SR)
    assert looped.shape[1] == SR - int(0.1 * SR)


def test_pipeline_silence_is_reported_not_emptied() -> None:
    out, warnings = process_audio(np.zeros((1, SR), np.float32), SR, post(trim_silence=True), channels=1, sample_rate=SR)
    assert out.shape[1] == SR and warnings == ["audio is silent; nothing to trim"]


def test_instrument_zones_keep_the_attack() -> None:
    body = tone(220, 0.4, amp=0.5)
    audio = np.concatenate([np.zeros(2000, np.float32), body, np.zeros(SR, np.float32)])[None, :]
    kw = dict(channels=1, sample_rate=SR)
    zone, _ = process_audio(audio, SR, post(trim_silence=True, fade_in_ms=50, loop_crossfade_ms=50), instrument=True, **kw)
    plain, _ = process_audio(audio, SR, post(trim_silence=True, fade_in_ms=50, loop_crossfade_ms=50), **kw)
    assert zone.shape[1] > plain.shape[1]  # tail trimmed only; head and attack untouched, no crossfade shortening
    assert np.argmax(np.abs(zone[0]) > 0.01) > 1500  # the leading silence is still there


def test_compare_wording_and_tolerances() -> None:
    m = {"bpm": 65.0, "bpm_confidence": 0.71, "key": "A minor", "key_confidence": 0.5}
    assert compare({"bpm": 130, "key": None, "note": None}, m) == ["bpm requested 130, measured 65 (0.71)"]
    assert compare({"bpm": 66, "key": None, "note": None}, m) == []  # within 5 %
    assert compare({"bpm": None, "key": "F minor", "note": None}, m) == ["key requested F minor, measured A minor (0.50)"]
    assert compare({"bpm": None, "key": "Bb major", "note": None}, {"key": "A# major", "key_confidence": 0.9}) == []  # enharmonic
    assert compare({"bpm": None, "key": "c", "note": None}, {"key": "C major", "key_confidence": 0.9}) == []
    assert compare({"bpm": None, "key": "C", "note": None}, {"key": "", "key_confidence": 0.0}) == ["key requested C major, measured none (0.00)"]
    assert compare({"bpm": None, "key": None, "note": 60}, {"note": 63.04}) == ["note requested 60, measured 63.0"]
    assert compare({"bpm": None, "key": None, "note": 60}, {"note": 60.3}) == []
    assert compare({"bpm": None, "key": None, "note": 60}, {"note": None}) == ["note requested 60, no pitch detected"]
    assert compare({"bpm": 100, "key": None, "note": None}, {"bpm": 150.0, "bpm_confidence": 0.9}, "zone 60") == ["zone 60: bpm requested 100, measured 150 (0.90)"]
    assert compare({}, {}) == []


def test_measure_note() -> None:
    assert measure_note(tone(440, 1.0)[None, :], SR) == pytest.approx(69.0, abs=0.1)
    assert measure_note(np.zeros((1, SR), np.float32), SR) is None


def test_safe_name() -> None:
    assert safe_name("Pad: warm/../x!") == "Pad_warm_.._x"
    assert safe_name("../..") == "file" and safe_name("", "z") == "z"


def test_audio_output_sidecar_has_requested_and_measured(tmp_path: Path) -> None:
    src = write_wav(tmp_path / "raw.wav", clicks(100, 8.0))
    out = {"path": str(src), "kind": "audio", "candidate": 1, "extra": {"stem": "drums"}}
    done = process_output(out, ctx("generate.loop", {"bpm": 130, "key": "F minor", "bars": 4}), tmp_path / "work")
    side = done.sidecar
    assert done.kind == "audio" and done.path.suffix == ".wav"
    assert side["requested"] == {"bpm": 130.0, "key": "F minor", "note": None}
    assert side["measured"]["bpm"] == pytest.approx(100.0, abs=3) and "onsets" in side["measured"] and "onsets_s" not in side["measured"]
    assert any(w.startswith("bpm requested 130, measured 100") for w in side["warnings"])
    assert any(w.startswith("key requested F minor") for w in side["warnings"])
    assert (side["sample_rate"], side["channels"], side["bit_depth"], side["candidate"], side["seed"]) == (48000, 2, "24", 1, 11)
    assert side["stem"] == "drums" and side["parents"] == [] and side["model"] == "testtone" and side["licence"] == "CC0"
    audio, sr = wav.read(done.path)
    assert sr == 48000 and audio.shape[0] == 2
    json.dumps(side, allow_nan=False)


def test_bit_depth_is_honoured(tmp_path: Path) -> None:
    src = write_wav(tmp_path / "raw.wav", tone(440, 0.5)[None, :])
    sizes = {}
    for depth, width in (("16", 2), ("24", 3), ("32f", 4)):
        done = process_output({"path": str(src), "kind": "audio", "candidate": 0, "extra": {}},
                              ctx("generate.sfx", {"post": {"bit_depth": depth}, "channels": 1}), tmp_path / depth)
        sizes[depth] = done.path.stat().st_size
        assert abs(sizes[depth] - 44 - 24000 * width) < 64
    assert sizes["16"] < sizes["24"] < sizes["32f"]


def test_unreadable_or_empty_audio_fails_cleanly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF....nope")
    with pytest.raises(PostprocessError):
        process_output({"path": str(bad), "kind": "audio", "candidate": 0, "extra": {}}, ctx("generate.sfx", {}), tmp_path / "w")
    empty = write_wav(tmp_path / "empty.wav", np.zeros((1, 0), np.float32))
    with pytest.raises(PostprocessError, match="empty"):
        process_output({"path": str(empty), "kind": "audio", "candidate": 0, "extra": {}}, ctx("generate.sfx", {}), tmp_path / "w")


def _bundle_inputs(tmp_path: Path, notes: list[int]) -> dict[str, Any]:
    folder = tmp_path / "raw_instrument"
    zones = []
    for n in notes:
        write_wav(folder / "samples" / f"z{n}.wav", tone(440 * 2 ** ((n - 69) / 12), 0.6)[None, :])
        zones.append({"sample": f"samples/z{n}.wav", "root_note": n, "key_low": n - 1, "key_high": n + 1, "vel_low": 0, "vel_high": 127})
    return {"path": str(folder), "kind": "instrument", "candidate": 0, "extra": {"zones": zones, "name": "ignored"}}


def test_instrument_bundle(tmp_path: Path) -> None:
    done = process_output(_bundle_inputs(tmp_path, [48, 60, 72]), ctx("generate.instrument", {"name": "Test Pad"}), tmp_path / "w")
    assert done.kind == "instrument" and done.path.suffix == ".zip"
    with zipfile.ZipFile(done.path) as z:
        names = z.namelist()
        assert names == sorted(names)
        assert {"Test_Pad.sfz", "Test_Pad.dspreset", "manifest.json"} <= set(names)
        samples = [n for n in names if n.startswith("samples/")]
        assert len(samples) == 3
        manifest = json.loads(z.read("manifest.json"))
        sfz = z.read("Test_Pad.sfz").decode()
        dsp_xml = ET.fromstring(z.read("Test_Pad.dspreset"))
        for n in samples:
            assert z.getinfo(n).file_size > 1000
    assert [zone["root_note"] for zone in manifest["zones"]] == [48, 60, 72] and manifest["sample_rate"] == 48000
    assert sfz.count("<region>") == 3 and all(f"sample={s}" in sfz for s in samples)
    assert [e.get("path") for e in dsp_xml.iter("sample")] == samples
    side = done.sidecar
    assert [f["root_note"] for f in side["files"]] == [48, 60, 72] and side["name"] == "Test_Pad"
    assert all(abs(f["measured"]["note"] - f["root_note"]) < 0.3 for f in side["files"]) and side["warnings"] == []
    assert side["measured"]["zones"] == 3


def test_instrument_bundle_is_deterministic(tmp_path: Path) -> None:
    raw = _bundle_inputs(tmp_path, [60, 64])
    a = process_output(raw, ctx("generate.instrument", {"name": "X"}), tmp_path / "a")
    b = process_output(raw, ctx("generate.instrument", {"name": "X"}), tmp_path / "b")
    assert a.path.read_bytes() == b.path.read_bytes()


def test_instrument_flags_wrong_pitch(tmp_path: Path) -> None:
    raw = _bundle_inputs(tmp_path, [60])
    raw["extra"]["zones"][0]["root_note"] = 65
    done = process_output(raw, ctx("generate.instrument", {"name": "X"}), tmp_path / "w")
    assert done.warnings and done.warnings[0].startswith("zone 65: note requested 65, measured 60")


@pytest.mark.parametrize("bad_sample", ["../outside.wav", "/etc/passwd", "samples/missing.wav", "", None, 5])
def test_instrument_zone_paths_are_confined(tmp_path: Path, bad_sample: Any) -> None:
    raw = _bundle_inputs(tmp_path, [60])
    write_wav(tmp_path / "outside.wav", tone(440, 0.2)[None, :])
    raw["extra"]["zones"][0]["sample"] = bad_sample
    with pytest.raises(PostprocessError):
        process_output(raw, ctx("generate.instrument", {"name": "X"}), tmp_path / "w")


def test_instrument_needs_valid_zones(tmp_path: Path) -> None:
    raw = _bundle_inputs(tmp_path, [60])
    raw["extra"]["zones"][0]["key_low"] = 100  # low > high
    with pytest.raises(PostprocessError):
        process_output(raw, ctx("generate.instrument", {"name": "X"}), tmp_path / "w")
    for zones in ([], "x", None):
        raw["extra"]["zones"] = zones
        with pytest.raises(PostprocessError):
            process_output(raw, ctx("generate.instrument", {"name": "X"}), tmp_path / "w2")


def test_midi_output(tmp_path: Path) -> None:
    mid = tmp_path / "m.mid"
    midi.write_smf(mid, [{"pitch": 60, "start_s": 0.0, "end_s": 0.5, "velocity": 90}])
    notes = [{"pitch": 60, "start_s": 0.0, "end_s": 0.5, "velocity": 90, "confidence": 0.9}]
    done = process_output({"path": str(mid), "kind": "midi", "candidate": 0, "extra": {"notes": notes, "bpm": 100}},
                          ctx("transcribe.melody", {}, {"audio": "a_" + "1" * 24}), tmp_path / "w")
    assert done.path.read_bytes() == mid.read_bytes()
    assert done.sidecar["notes"] == notes and done.sidecar["parents"] == ["a_" + "1" * 24] and done.sidecar["tempo_bpm"] == 100.0
    assert done.sidecar["kind"] == "midi" and done.sidecar["bit_depth"] is None and done.warnings == []
    empty = tmp_path / "e.mid"
    midi.write_smf(empty, [])
    done = process_output({"path": str(empty), "kind": "midi", "candidate": 0, "extra": {"notes": []}}, ctx("transcribe.melody", {}, {"audio": "a_" + "1" * 24}), tmp_path / "w")
    assert done.warnings == ["no notes detected"]
    broken = tmp_path / "b.mid"
    broken.write_bytes(b"MThd garbage")
    with pytest.raises(PostprocessError):
        process_output({"path": str(broken), "kind": "midi", "candidate": 0, "extra": {}}, ctx("transcribe.melody", {}, {"audio": "a_" + "1" * 24}), tmp_path / "w")


def test_json_output(tmp_path: Path) -> None:
    good = tmp_path / "r.json"
    good.write_text('{"a": 1}', encoding="utf-8")
    assert process_output({"path": str(good), "kind": "json", "candidate": 0, "extra": {}}, ctx("generate.sfx", {}), tmp_path / "w").path.suffix == ".json"
    good.write_text("{nope", encoding="utf-8")
    with pytest.raises(PostprocessError):
        process_output({"path": str(good), "kind": "json", "candidate": 0, "extra": {}}, ctx("generate.sfx", {}), tmp_path / "w")
    with pytest.raises(PostprocessError):
        process_output({"path": str(good), "kind": "nonsense", "candidate": 0, "extra": {}}, ctx("generate.sfx", {}), tmp_path / "w")


def test_silent_and_clipped_outputs_warn(tmp_path: Path) -> None:
    silent = write_wav(tmp_path / "s.wav", np.zeros((1, SR), np.float32))
    done = process_output({"path": str(silent), "kind": "audio", "candidate": 0, "extra": {}}, ctx("generate.sfx", {}), tmp_path / "w")
    assert "output is silent" in done.warnings
    square = write_wav(tmp_path / "q.wav", np.where(np.arange(SR) % 100 < 50, 1.0, -1.0).astype(np.float32)[None, :])
    done = process_output({"path": str(square), "kind": "audio", "candidate": 0, "extra": {}},
                          ctx("generate.sfx", {"post": {"normalize": "none"}}), tmp_path / "w2")
    assert any("clipped samples" in w for w in done.warnings)
