"""The Foundation tag prompt builder and the other prompt helpers (pure functions)."""

from __future__ import annotations

import pytest

from forge.workers import _prompts as p


def spec(**kw: object) -> p.FoundationSpec:
    return p.FoundationSpec(**kw)  # type: ignore[arg-type]


# ---- loops --------------------------------------------------------------------------------------

def test_loop_prompt_order_matches_the_foundation_grammar() -> None:
    s = spec(
        family="Synth", subfamily="Synth Lead", descriptors=("warm", "wide"), behaviours=("arp", "rising"),
        effects=("Medium Reverb", "Ping Pong Delay"), wet=True,
    )
    text = p.build_loop_prompt(s, bars=8, bpm=128, key="F minor")
    assert text == "Synth, Synth Lead, warm, wide, arp, rising, Wet, Medium Reverb, Ping Pong Delay, F minor, 8 Bars, 128 BPM"


def test_dry_prompt_has_no_effects_and_wet_defaults_follow_the_effects() -> None:
    dry = p.build_loop_prompt(spec(family="Keys", effects=("Low Reverb",), wet=False), bars=4, bpm=100)
    assert dry == "Keys, Dry, 4 Bars, 100 BPM"
    # no explicit wet/dry: wet exactly when effects are present (an "empty wet" is out of distribution)
    assert p.build_loop_prompt(spec(family="Keys"), bars=4, bpm=100).split(", ")[1] == "Dry"
    assert p.build_loop_prompt(spec(family="Keys", effects=("Low Reverb",)), bars=4, bpm=100) == "Keys, Wet, Low Reverb, 4 Bars, 100 BPM"


def test_numbers_and_key_are_formatted_like_the_training_data() -> None:
    assert p.build_loop_prompt(spec(), bars=4, bpm=127.98, key="Bb minor").endswith("A# minor, 4 Bars, 128 BPM")
    assert p.build_loop_prompt(spec(), bars=4, bpm=127.5).endswith("4 Bars, 127.5 BPM")
    assert p.format_key("fm") == "F minor"
    assert p.format_key("c") == "C major"
    assert p.format_key(None) is None
    with pytest.raises(ValueError):
        p.format_key("H lydian")
    with pytest.raises(ValueError):
        p.format_bpm(float("nan"))
    with pytest.raises(ValueError):
        p.build_loop_prompt(spec(), bars=0, bpm=120)


def test_authoritative_tokens_in_free_text_are_replaced_not_duplicated() -> None:
    s = p.spec_from_tags({"family": "Bass"}, "dark, 99 BPM, 2 bars, Wet, C major, C4, punchy")
    text = p.build_loop_prompt(s, bars=4, bpm=140, key="F minor")
    assert text == "Bass, dark, punchy, Dry, F minor, 4 Bars, 140 BPM"


def test_vocabulary_is_rewritten_to_canonical_case_and_unknown_tokens_are_reported() -> None:
    s = p.spec_from_tags({"family": "synth", "subfamily": "synth lead", "descriptors": ["WARM", "glittery"], "fx": "medium reverb"}, "")
    text = p.build_loop_prompt(s, bars=4, bpm=120)
    assert text == "Synth, Synth Lead, warm, glittery, Wet, Medium Reverb, 4 Bars, 120 BPM"
    assert p.unknown_tokens(s) == ["glittery"]


def test_tag_aliases_gary_style_fields_and_test_hooks() -> None:
    tags = {
        "Family": "Keys", "sub_family": "Rhodes Piano", "descriptor_knob_a": "warm", "descriptor_knob_b": ["soft", "round"],
        "descriptors_extra": "wide, clean", "behavior_tags": "arp, slow speed", "effect": ["Low Delay"], "wetdry": "Wet",
        "key": "A minor", "x-fault": "crash", "mood": "dreamy",
    }
    s = p.spec_from_tags(tags, "")
    assert s.family == "Keys" and s.subfamily == "Rhodes Piano"
    assert s.descriptors == ("warm", "soft", "round", "wide", "clean", "dreamy")
    assert s.behaviours == ("arp", "slow speed")
    assert s.effects == ("Low Delay",) and s.wet is True and s.key == "A minor"
    assert "crash" not in p.build_loop_prompt(s, bars=4, bpm=120)
    assert p.build_loop_prompt(s, bars=4, bpm=120).endswith("Wet, Low Delay, A minor, 4 Bars, 120 BPM")


def test_hostile_tag_values_are_sanitised() -> None:
    s = p.spec_from_tags({"descriptors": ["ok", "bad\x00\x1b[31m, injected", "x" * 500, 12, None]}, "line\nbreak,  spaced   out")
    text = p.build_loop_prompt(s, bars=4, bpm=120)
    assert "\x00" not in text and "\x1b" not in text and "\n" not in text
    assert all(len(token) <= p.MAX_TOKEN_CHARS for token in text.split(", "))
    assert "injected" in text  # the comma split it into separate tokens rather than smuggling a delimiter
    many = p.spec_from_tags({"descriptors": [f"t{i}" for i in range(500)]}, "")
    assert len(many.descriptors) == p.MAX_TOKENS


def test_descriptor_prompt_has_no_numbers_unless_tags_give_them() -> None:
    assert p.build_descriptor_prompt(spec(family="Wind")) == "Wind, Dry"
    s = p.spec_from_tags({"family": "Wind", "bpm": "120 BPM", "bars": "4", "key": "D minor"}, "")
    assert p.build_descriptor_prompt(s) == "Wind, Dry, D minor, 4 Bars, 120 BPM"


# ---- one shots and keybeds -----------------------------------------------------------------------

@pytest.mark.parametrize(
    ("note", "register", "name"),
    [(12, "Sub Register", "C0"), (35, "Sub Register", "B1"), (36, "Low Register", "C2"), (59, "Medium Register", "B3"),
     (60, "Medium Register", "C4"), (72, "High Register", "C5"), (84, "Top Register", "C6"), (104, "Top Register", "G#7")],
)
def test_oneshot_register_and_note(note: int, register: str, name: str) -> None:
    text = p.build_oneshot_prompt(spec(family="Mallet", subfamily="Marimba", descriptors=("warm",)), midi_note=note)
    assert text == f"One Shot, Mallet, Marimba, warm, Dry, {register}, {name}"


def test_oneshot_wet_and_range() -> None:
    assert p.build_oneshot_prompt(spec(family="Keys", effects=("Plate Reverb",)), midi_note=60) == (
        "One Shot, Keys, Wet, Plate Reverb, Medium Register, C4"
    )
    for bad in (0, 11, 105, 127):
        with pytest.raises(ValueError, match="one-shot range"):
            p.build_oneshot_prompt(spec(), midi_note=bad)


def test_keybed_prompt_uses_the_single_note_grammar_and_title_case_tags() -> None:
    s = p.spec_from_tags({"family": "keys", "subfamily": "digital piano", "descriptors": ["warm", "bright"]}, "")
    assert p.build_keybed_prompt(s, midi_note=60) == "Keybed, Timbre Profile, Keys, Digital Piano, Warm, Bright, Dry, Target Note, C4"
    wet = p.spec_from_tags({"family": "Bass", "fx": "Medium Reverb"}, "")
    assert p.build_keybed_prompt(wet, midi_note=36) == "Keybed, Timbre Profile, Bass, Wet, Medium Reverb, Target Note, C2"
    # behaviours (melody words) are not part of the keybed grammar
    assert "arp" not in p.build_keybed_prompt(p.spec_from_tags({"family": "Synth", "behaviour": "arp"}, ""), midi_note=48)
    for bad in (11, 109):
        with pytest.raises(ValueError, match="keybed range"):
            p.build_keybed_prompt(spec(), midi_note=bad)
    assert p.unknown_tokens(s, keybed=True) == []


def test_midi_to_note_names() -> None:
    assert [p.midi_to_note(n) for n in (0, 21, 60, 61, 127)] == ["C-1", "A0", "C4", "C#4", "G9"]
    for bad in (-1, 128, True, 1.5):
        with pytest.raises(ValueError):
            p.midi_to_note(bad)  # type: ignore[arg-type]


# ---- Stable Audio 3 ------------------------------------------------------------------------------

def test_sa3_prompts() -> None:
    assert p.sa3_sfx_prompt("a heavy door slam", {"genre": "Cinematic"}) == "TrackType: SFX, a heavy door slam, Genre: Cinematic"
    assert p.sa3_loop_prompt("funky bass", {"instruments": ["Bass", "Drums"]}, bpm=124, key="F minor", bars=4) == (
        "TrackType: Music, VocalType: Instrumental, funky bass, Instruments: Bass, Instruments: Drums, Key: F minor, 124 BPM, 4 bars"
    )
    assert p.sa3_sample_prompt("pluck", None, midi_note=60, sfx_model=False) == "TrackType: Instrument, pluck, single note C4"
    assert p.sa3_sample_prompt("", None, midi_note=60, sfx_model=True) == "TrackType: SFX, single note C4"
    assert p.sa3_edit_prompt("", None) == ""
    assert p.sa3_edit_prompt("bossa nova bassline", {"x-fault": "crash"}) == "bossa nova bassline"


# ---- ACE-Step ------------------------------------------------------------------------------------

def test_ace_track_names_and_aliases() -> None:
    assert p.ace_track("Drums") == "drums"
    assert p.ace_track("piano") == "keyboard"
    assert p.ace_track("backing vocals") == "backing_vocals"
    with pytest.raises(ValueError, match="unknown ACE-Step track"):
        p.ace_track("kazoo")


def test_ace_track_from_description_and_tags() -> None:
    assert p.ace_track_from_description("a warm analog pad") == "synth"
    assert p.ace_track_from_description("tight sub bass") == "bass"
    assert p.ace_track_from_description("Backing Vocals with air") == "backing_vocals"
    assert p.ace_track_from_description("something indescribable") is None
    assert p.ace_track_from_description("warm pad", {"track": "strings"}) == "strings"
    with pytest.raises(ValueError):
        p.ace_track_from_description("pad", {"stem": "kazoo"})


def test_ace_caption_is_limited_and_tags_are_included() -> None:
    caption = p.ace_caption("x" * 900, {"genre": "House"}, extra=["instrumental"])
    assert len(caption) == 512
    assert p.ace_caption("deep house", {"genre": "House"}, extra=["instrumental"]) == "deep house, instrumental, Genre: House"
    assert p.ace_key("bb major") == "A# major" and p.ace_key(None) == ""
