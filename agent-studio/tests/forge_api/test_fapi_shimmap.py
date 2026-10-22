"""How Gary4juce-style request bodies map to Forge job params (no service needed)."""

from __future__ import annotations

import pytest

from forge.config import parse_config
from forge.gary_shim import Sessions, create_shim_apps, effect_tags, foundation_request, sa3_request, Fields, ShimRequestError


def test_effect_tags_use_the_foundation_vocabulary() -> None:
    body = {
        "reverb_enabled": True, "reverb_amount": 0.5, "delay_on": True, "delay_level": "high", "phaser_toggle": True,
        "distortion_enabled": False, "distortion_amount": 0.9, "unknown_enabled": True, "bitcrush_enabled": "yes", "chorus_amount": 3,
    }
    assert effect_tags(body) == {"effects": ["Medium Reverb", "High Delay", "Phaser"]}
    assert effect_tags({}) == {}
    assert effect_tags({"reverb": True}) == {"effects": ["Medium Reverb"]}  # the vocabulary has no bare "Reverb"
    assert effect_tags({"enable_delay": True, "use_distortion": True, "distortion_mix": 10}) == {"effects": ["Delay", "Low Distortion"]}


@pytest.mark.parametrize(
    ("amount", "token"),
    [(0.0, "Low Reverb"), (0.2, "Low Reverb"), (0.34, "Medium Reverb"), (0.66, "Medium Reverb"), (0.7, "High Reverb"), (1, "High Reverb"),
     (80, "High Reverb"), (50, "Medium Reverb"), ("Low", "Low Reverb"), ("0.9", "High Reverb"), ("loud", "Medium Reverb"), (None, "Medium Reverb"),
     (float("nan"), "Medium Reverb"), (True, "Medium Reverb")],
)
def test_effect_amounts_pick_low_medium_or_high(amount: object, token: str) -> None:
    assert effect_tags({"reverb_enabled": True, "reverb_amount": amount}) == {"effects": [token]}


def test_bitcrush_only_has_a_high_variant() -> None:
    assert effect_tags({"bitcrush_enabled": True, "bitcrush_amount": 0.9}) == {"effects": ["High Bitcrush"]}
    assert effect_tags({"bitcrush_enabled": True, "bitcrush_amount": 0.1}) == {"effects": ["Bitcrush"]}


def test_effect_tags_reach_the_foundation_prompt() -> None:
    from forge.workers import _prompts as prompts

    spec = foundation_request(
        {"family": "Synth", "subfamily": "Pad", "descriptor_knob_a": "warm", "behavior_tags": ["slow speed"], "reverb_enabled": True,
         "reverb_amount": 0.9, "host_bpm": 100, "bars": 4, "key_root": "F", "key_mode": "minor"}, False,
    )
    prompt = prompts.build_loop_prompt(prompts.spec_from_tags(spec.params["tags"], spec.params["prompt"]), bars=spec.params["bars"], bpm=spec.params["bpm"], key=spec.params["key"])
    assert prompt == "Synth, Pad, warm, slow speed, Wet, High Reverb, F minor, 4 Bars, 100 BPM"
    audio = foundation_request({"family": "Synth", "host_bpm": 100, "bars": 4, "key_root": "F", "key_mode": "minor"}, True)
    assert prompts.build_descriptor_prompt(prompts.spec_from_tags(audio.params["tags"])) == "Synth, Dry, F minor, 4 Bars, 100 BPM"


def test_foundation_request_defaults_and_audio_variant() -> None:
    spec = foundation_request({}, False)
    assert spec.job_type == "generate.loop" and spec.params == {"prompt": "", "tags": {}, "seed": None, "bars": 4, "bpm": 120.0}
    spec = foundation_request({"host_bpm": "128", "bars": 8.0, "key_root": "Eb", "key_mode": "major", "seed": 5}, True)
    assert spec.job_type == "transform.audio" and spec.params["strength"] == 0.5 and spec.params["seed"] == 5
    assert spec.params["tags"] == {"bars": "8", "bpm": "128", "key": "Eb major"}


def test_sa3_request_picks_the_job_type() -> None:
    assert sa3_request({"prompt": "boom"}, False).job_type == "generate.sfx"
    assert sa3_request({"prompt": "boom", "seconds_total": 12}, False).params["duration_s"] == 12
    spec = sa3_request({"prompt": "beat", "host_bpm": 100, "bars": 2, "key_root": "A", "key_mode": "minor"}, False)
    assert spec.job_type == "generate.loop" and spec.params["bpm"] == 100 and spec.params["key"] == "A minor"
    assert sa3_request({"model": "sa3-medium"}, False).wanted_model == "sa3-medium"


def test_sessions_are_capped_and_unguessable() -> None:
    sessions = Sessions(cap=3)
    ids = [sessions.add(f"j_{i}") for i in range(5)]
    assert len(set(ids)) == 5 and all(len(i) == 32 for i in ids)
    assert [sessions.get(i) for i in ids] == [None, None, "j_2", "j_3", "j_4"]


def test_create_shim_apps_follows_the_configured_ports() -> None:
    cfg = parse_config({"gary_shim": {"enabled": True, "ports": {"foundation": 9015, "sa3": 9006, "other": 1}}})
    apps = create_shim_apps(None, cfg)  # type: ignore[arg-type]
    assert {name: port for name, (port, _) in apps.items()} == {"foundation": 9015, "sa3": 9006}
    assert parse_config({}).gary_shim.ports == {"foundation": 8015, "sa3": 8006}


@pytest.mark.parametrize(
    ("value", "ok"),
    [(1, True), (1.5, True), ("2", True), (True, False), (None, True), ("abc", False), ("nan", False), ("inf", False), (1e400, False), ([1], False), (999, False)],
)
def test_numbers_are_finite_bounded_and_not_booleans(value: object, ok: bool) -> None:
    f = Fields({"n": value})
    result = f.number("n", lo=0, hi=100, default=7)
    assert (not f.problems) is ok
    assert result == 7 or ok


def test_problems_are_reported_together() -> None:
    with pytest.raises(ShimRequestError) as caught:
        foundation_request({"bars": 99, "host_bpm": "x", "key_root": "Q"}, False)
    message = str(caught.value)
    assert "bars" in message and "host_bpm" in message and "key" in message
