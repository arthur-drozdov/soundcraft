from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from audiokit import instruments as ins


def _zones() -> list[dict]:
    plan = ins.plan_keybed(36, 48, 4)
    return [
        {"sample": f"n{root}.wav", "root_note": root, "key_low": lo, "key_high": hi, "vel_low": 0, "vel_high": 127,
         "tune_cents": 12.5 if i == 0 else 0, "gain_db": -3.0 if i == 1 else 0,
         "loop_mode": "loop_continuous" if i == 0 else None, "loop_start": 100 if i == 0 else None,
         "loop_end": 5000 if i == 0 else None}
        for i, (root, lo, hi) in enumerate(plan)
    ]


@pytest.mark.parametrize("low,high,step", [(36, 84, 3), (0, 127, 1), (60, 60, 5), (36, 48, 4), (10, 20, 12), (21, 108, 7)])
def test_plan_keybed_covers_each_key_exactly_once(low: int, high: int, step: int) -> None:
    plan = ins.plan_keybed(low, high, step)
    covered = [k for _, lo, hi in plan for k in range(lo, hi + 1)]
    assert covered == list(range(low, high + 1))
    for root, lo, hi in plan:
        assert lo <= root <= hi and hi - lo + 1 <= step


def test_plan_keybed_roots() -> None:
    assert ins.plan_keybed(36, 41, 3) == [(37, 36, 38), (40, 39, 41)]
    assert ins.plan_keybed(36, 37, 5) == [(36, 36, 37)]


def test_plan_keybed_validation() -> None:
    for args in ((50, 40, 3), (-1, 10, 3), (0, 128, 3), (0, 10, 0), (0, 10, True)):
        with pytest.raises(ins.InstrumentError):
            ins.plan_keybed(*args)  # type: ignore[arg-type]


def test_write_sfz_one_region_per_zone(tmp_path: Path) -> None:
    zones = _zones()
    out = tmp_path / "inst" / "kit.sfz"
    ins.write_sfz(out, zones, tmp_path / "inst" / "samples")
    text = out.read_text()
    regions = text.split("<region>")[1:]
    assert len(regions) == len(zones) == text.count("<region>")
    for region, zone in zip(regions, zones):
        opcodes = dict(line.split("=", 1) for line in region.strip().splitlines() if "=" in line)
        assert opcodes["sample"] == f"samples/n{zone['root_note']}.wav"
        assert int(opcodes["pitch_keycenter"]) == zone["root_note"]
        assert (int(opcodes["lokey"]), int(opcodes["hikey"])) == (zone["key_low"], zone["key_high"])
        assert (int(opcodes["lovel"]), int(opcodes["hivel"])) == (0, 127)
    assert "tune=12.5" in regions[0] and "loop_mode=loop_continuous" in regions[0]
    assert "loop_start=100" in regions[0] and "loop_end=5000" in regions[0]
    assert "volume=-3" in regions[1] and "tune" not in regions[1]


def test_write_dspreset_one_sample_per_zone(tmp_path: Path) -> None:
    zones = _zones()
    out = tmp_path / "kit.dspreset"
    ins.write_dspreset(out, zones, tmp_path / "samples")
    root = ET.parse(out).getroot()
    assert root.tag == "DecentSampler"
    samples = root.findall("./groups/group/sample")
    assert len(samples) == len(zones)
    for s, zone in zip(samples, zones):
        assert s.get("path") == f"samples/n{zone['root_note']}.wav"
        assert int(s.get("rootNote")) == zone["root_note"]
        assert (int(s.get("loNote")), int(s.get("hiNote"))) == (zone["key_low"], zone["key_high"])
    assert float(samples[0].get("tuning")) == pytest.approx(0.125)
    assert samples[0].get("loopEnabled") == "true" and samples[0].get("loopStart") == "100"
    assert samples[1].get("volume") == "-3dB" and samples[1].get("loopEnabled") is None


def test_xml_escaping_of_sample_names(tmp_path: Path) -> None:
    zone = {"sample": 'a&b "<c>".wav', "root_note": 60, "key_low": 60, "key_high": 60}
    out = tmp_path / "x.dspreset"
    ins.write_dspreset(out, [zone], tmp_path)
    assert ET.parse(out).getroot().find("./groups/group/sample").get("path") == 'a&b "<c>".wav'


@pytest.mark.parametrize(
    "patch",
    [
        {"sample": "bad\nname.wav"}, {"sample": ""}, {"root_note": 128}, {"key_low": 70, "key_high": 60},
        {"vel_low": 100, "vel_high": 10}, {"tune_cents": float("nan")}, {"gain_db": float("inf")},
        {"loop_mode": "weird"}, {"loop_start": 10, "loop_end": 5}, {"root_note": True},
    ],
)
def test_invalid_zones_are_rejected(tmp_path: Path, patch: dict) -> None:
    zone = {"sample": "a.wav", "root_note": 60, "key_low": 60, "key_high": 60} | patch
    with pytest.raises(ins.InstrumentError):
        ins.write_sfz(tmp_path / "k.sfz", [zone], tmp_path)
    assert not (tmp_path / "k.sfz").exists()


def test_empty_zone_list_rejected(tmp_path: Path) -> None:
    with pytest.raises(ins.InstrumentError):
        ins.write_sfz(tmp_path / "k.sfz", [], tmp_path)
