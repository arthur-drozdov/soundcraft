"""The profiler against FakeStudio: describe -> draft, calibration, inference, measurement, safety."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from atlas import merge, profiler, profiles, yamlio
from atlas.profiler import (
    CalibrationPoint,
    ProfilerError,
    ProfilerOptions,
    ScratchSession,
    infer_from_texts,
    parse_describe,
    parse_value_text,
    profile_plugin,
)
from atlas.validate import validate_profile
from atprof_fake_studio import FakeStudio, make_plugins

OPTS = {"max_bounce_s": 3.0, "date": "2026-10-22", "progress": lambda message: None}


def run(plugin_id: str, studio: FakeStudio | None = None, **overrides: Any) -> tuple[profiler.ProfileResult, FakeStudio]:
    studio = studio or FakeStudio()
    result = profile_plugin(studio, plugin_id, ProfilerOptions(**{**OPTS, **overrides}))
    return result, studio


def param(result: profiler.ProfileResult, pid: str) -> dict[str, Any]:
    return next(p for p in result.profile["params"] if p["id"] == pid)


@pytest.fixture(scope="module")
def gain() -> tuple[profiler.ProfileResult, FakeStudio]:
    return run("fake_gain")


@pytest.fixture(scope="module")
def filt() -> tuple[profiler.ProfileResult, FakeStudio]:
    return run("fake_filter")


@pytest.fixture(scope="module")
def hosted() -> tuple[profiler.ProfileResult, FakeStudio]:
    return run("vst3:FAKEHOSTED0001")


@pytest.fixture(scope="module")
def synth() -> tuple[profiler.ProfileResult, FakeStudio]:
    return run("fake_synth")


# ---- step 2: describe -> draft ---------------------------------------------------------------------------


def test_the_draft_carries_identity_and_the_raw_table_with_host_provenance() -> None:
    result, _ = run("fake_gain", measure=False)
    profile = result.profile
    assert validate_profile(profile) == []
    assert profile["identity"] == {
        "name": "Fake Gain", "vendor": "SoundCraft", "format": "builtin", "plugin_id": "fake_gain",
        "version": "0.10.0", "os": [profiler.current_os()], "profiled": "2026-10-22",
    }
    gain = param(result, "gain")
    assert (gain["host_name"], gain["min"], gain["max"], gain["default"]) == ("Gain", -60.0, 24.0, 0.0)
    assert (gain["unit"], gain["taper"]) == ("dB", "linear")
    for field in ("host_name", "unit", "min", "max", "default", "taper"):
        assert gain["provenance"][field] == {"source": "host", "confidence": 1.0}
    assert param(result, "mute")["unit"] == "bool"
    assert profile["provenance"]["identity.vendor"] == {"source": "host", "confidence": 1.0}
    assert not any("does" in p for p in profile["params"]), "nothing is measured with measure=False"


def test_a_hosted_plugin_without_vendor_or_version_gets_a_low_confidence_placeholder() -> None:
    plugins = make_plugins()
    plugins["fake_gain"].vendor = None
    plugins["fake_gain"].version = None
    result, _ = run("fake_gain", FakeStudio(plugins=plugins), measure=False)
    ident = result.profile["identity"]
    assert (ident["vendor"], ident["version"]) == ("unknown", "unknown")
    assert result.profile["provenance"]["identity.vendor"]["source"] == "inferred"
    assert result.profile["provenance"]["identity.vendor"]["confidence"] < 0.9
    assert validate_profile(result.profile) == []


def test_unit_names_follow_the_profile_vocabulary(synth: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    result, _ = synth
    assert [p["unit"] for p in result.profile["params"]] == ["dB", "Hz", "seconds", "percent"]
    assert result.profile["identity"]["plugin_id"] == "fake_synth"
    assert result.profile["summary"] == "Instrument with 4 parameters."


def test_an_unknown_plugin_is_a_clear_error_and_the_scratch_session_is_closed() -> None:
    studio = FakeStudio()
    with pytest.raises(ProfilerError, match="no_such_plugin"):
        profile_plugin(studio, "no_such_plugin", ProfilerOptions(**OPTS))
    assert studio.calls[0][0] == "session.new" and studio.calls[-1][0] == "session.close"


@pytest.mark.parametrize("bad", ["", "   ", "x" * 301, "a\x00b", None, 5])
def test_hostile_plugin_ids_are_refused_before_touching_the_studio(bad: Any) -> None:
    studio = FakeStudio()
    with pytest.raises(ProfilerError):
        profile_plugin(studio, bad, ProfilerOptions(**OPTS))
    assert studio.calls == []


@pytest.mark.parametrize(
    "overrides",
    [{"max_bounce_s": 1.0}, {"max_bounce_s": 600.0}, {"max_bounce_s": float("nan")}, {"max_params": -1}, {"max_params": True}, {"date": "22/10/2026"}],
)
def test_hostile_options_are_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(ProfilerError):
        profile_plugin(FakeStudio(), "fake_gain", ProfilerOptions(**{**OPTS, **overrides}))


def test_the_draft_is_written_with_atlas_profiles_naming(tmp_path: Path, filt: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    result, _ = filt
    path = profiler.write_draft(result.profile, tmp_path / "drafts")
    assert path == tmp_path / "drafts" / "builtin" / "soundcraft" / "fake-filter@0.yaml"
    assert profiles.load_profile(path) == profiles.load_profile(path)  # valid on reload
    explicit = profiler.write_draft(result.profile, tmp_path / "x" / "mine.yaml")
    assert explicit.read_text() == path.read_text()
    assert yamlio.read_file(path)["identity"]["plugin_id"] == "fake_filter"


def test_a_draft_takes_a_knowledge_patch_like_any_profile(tmp_path: Path, filt: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    result, _ = filt
    patch = {"default_provenance": {"source": "manual", "ref": "Manual p.4", "confidence": 0.95}, "params": [{"id": "cutoff", "does": "Corner frequency of the low-pass."}]}
    merged = merge.merge_patch(result.profile, patch).profile
    cutoff = param(profiler.ProfileResult(merged), "cutoff")
    assert cutoff["does"] == "Corner frequency of the low-pass."
    assert cutoff["provenance"]["does"]["source"] == "manual"  # manual knowledge replaces the measured text
    assert cutoff["provenance"]["calibration"]["source"] == "host"


# ---- step 3: calibration ---------------------------------------------------------------------------------


def test_calibration_uses_21_points_for_continuous_params_with_units_and_11_otherwise(
    gain: tuple[profiler.ProfileResult, FakeStudio], hosted: tuple[profiler.ProfileResult, FakeStudio]
) -> None:
    g, _ = gain
    cal = param(g, "gain")["calibration"]
    assert len(cal) == 21
    assert cal[0] == {"value": -60.0, "text": "-60.0 dB"} and cal[-1] == {"value": 24.0, "text": "24.0 dB"}
    assert cal[10]["value"] == pytest.approx(-18.0) and cal[10]["text"] == "-18.0 dB"
    assert len(param(g, "dither")["calibration"]) == 21
    h, _ = hosted
    assert len(param(h, "level")["calibration"]) == 11, "11 points when the host reports no unit... until the texts say dB"
    assert param(g, "gain")["provenance"]["calibration"]["source"] == "host"


def test_toggles_are_asked_at_both_values_and_labelled_choices_not_at_all() -> None:
    plugins = make_plugins()
    from atprof_fake_studio import FakeParam

    plugins["fake_gain"].params.append(FakeParam("curve", "Curve", 0.0, 2.0, 0.0, "Choice", choices=("Soft", "Hard", "Wild")))
    result, studio = run("fake_gain", FakeStudio(plugins=plugins), measure=False)
    assert [c["text"] for c in param(result, "mute")["calibration"]] == ["Off", "On"]
    assert "calibration" not in param(result, "curve")
    assert param(result, "curve")["choices"] == ["Soft", "Hard", "Wild"]
    asked = [p["param"] for c, p in studio.calls if c == "plugin.param_text"]
    assert "curve" not in asked


def test_calibration_uses_the_hosts_control_travel_and_falls_back_without_it() -> None:
    with_norm, studio = run("fake_filter", measure=False)
    calls = [p for c, p in studio.calls if c == "plugin.param_text" and p["param"] == "cutoff"]
    assert calls[0]["normalized"] is True and calls[0]["values"][0] == 0.0 and calls[0]["values"][-1] == 1.0
    old = FakeStudio(supports_normalized=False)
    without, studio2 = run("fake_filter", old, measure=False)
    explicit = [p for c, p in studio2.calls if c == "plugin.param_text" and p["param"] == "cutoff"][-1]
    assert "normalized" not in explicit and explicit["values"][0] == pytest.approx(200.0)
    assert param(without, "cutoff")["calibration"] == param(with_norm, "cutoff")["calibration"]


def test_a_failing_calibration_is_a_warning_not_a_failure() -> None:
    studio = FakeStudio()
    studio.fail["plugin.param_text"] = "boom"
    result, _ = run("fake_gain", studio, measure=False)
    assert validate_profile(result.profile) == []
    assert any("not calibrated" in w and "boom" in w for w in result.warnings)
    assert "calibration" not in param(result, "gain")


# ---- unit and taper inference -----------------------------------------------------------------------------


def pts(values: list[float], texts: list[str]) -> list[CalibrationPoint]:
    n = len(values)
    return [CalibrationPoint(i / (n - 1), v, t) for i, (v, t) in enumerate(zip(values, texts, strict=True))]


def test_geometric_hz_texts_are_a_log_taper_and_arithmetic_ones_a_linear_taper() -> None:
    hz = [20.0 * 1000.0 ** (i / 10) for i in range(11)]
    inferred = infer_from_texts(pts(hz, [f"{v:.1f} Hz" if v < 1000 else f"{v / 1000:.2f} kHz" for v in hz]))
    assert (inferred.unit, inferred.taper) == ("Hz", "log")
    ms = [1.0 + 9.0 * i for i in range(11)]
    inferred = infer_from_texts(pts(ms, [f"{v:.0f} ms" for v in ms]))
    assert (inferred.unit, inferred.taper) == ("ms", "linear")
    db = [-60.0 + 7.2 * i for i in range(11)]
    inferred = infer_from_texts(pts(db, [f"{v:.1f} dB" for v in db]))
    assert (inferred.unit, inferred.taper) == ("dB", "linear")


def test_inference_says_nothing_when_the_texts_do_not_settle_it() -> None:
    assert infer_from_texts(pts([1, 2], ["a", "b"])) == profiler.TextInference()
    flat = infer_from_texts(pts([5.0] * 6, ["5 ms"] * 6))
    assert flat.unit == "ms" and flat.taper is None
    odd = infer_from_texts(pts([0, 1, 2, 3, 4, 5, 6, 7], ["1 Hz", "50 Hz", "51 Hz", "52 Hz", "400 Hz", "401 Hz", "2 Hz", "900 Hz"]))
    assert odd.taper is None
    mixed = infer_from_texts(pts([0, 1, 2, 3, 4, 5], ["1 Hz", "2 ms", "3 dB", "4 %", "5 s", "6 st"]))
    assert mixed.unit is None
    assert infer_from_texts([]) == profiler.TextInference()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2.40 kHz", (2400.0, "Hz")), ("894.4 Hz", (894.4, "Hz")), ("-6.0 dB", (-6.0, "dB")), ("12.5 ms", (12.5, "ms")),
        ("1.20 s", (1.2, "seconds")), ("4.0:1", (4.0, "ratio")), ("50%", (50.0, "percent")), ("+3 st", (3.0, "semitones")),
        ("-12 ct", (-12.0, "cents")), ("0.37", (0.37, "")), ("3 widgets", (3.0, "?")),
        ("-inf dB", None), ("Off", None), ("C#3", None), ("", None), ("  ", None),
    ],
)
def test_parse_value_text(text: str, expected: Any) -> None:
    assert parse_value_text(text) == expected


def test_a_hosted_plugin_with_no_unit_gets_an_inferred_unit_and_taper_from_its_texts(hosted: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    result, _ = hosted
    freq = param(result, "freq")
    assert freq["unit"] == "Hz"
    assert freq["provenance"]["unit"]["source"] == "inferred" and freq["provenance"]["unit"]["confidence"] < 0.9
    assert freq["taper"] == "log"  # the host maps linearly, the plugin's texts are geometric
    assert freq["provenance"]["taper"]["source"] == "inferred"
    assert any("freq" in w and "taper" in w for w in result.warnings)
    assert validate_profile(result.profile) == []
    level = param(result, "level")
    assert level["unit"] == "dB" and level["taper"] == "linear"
    assert level["provenance"]["taper"]["source"] == "host", "linear texts agree with the host: keep the host"


def test_the_hosts_own_unit_and_taper_stay_when_the_texts_agree(filt: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    result, _ = filt
    cutoff = param(result, "cutoff")
    assert (cutoff["unit"], cutoff["taper"]) == ("Hz", "log")
    assert cutoff["provenance"]["unit"]["source"] == "host" and cutoff["provenance"]["taper"]["source"] == "host"
    assert not any("cutoff" in w for w in result.warnings)


# ---- step 4: measurement ---------------------------------------------------------------------------------


def test_a_gain_parameter_scales_the_measured_level(gain: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    result, _ = gain
    g = param(result, "gain")
    assert "output level" in g["does"] and "-60.0 dB" in g["does"] and "-18.0 dB" in g["does"] and "+24.0 dB" in g["does"]
    prov = g["provenance"]["does"]
    assert prov["source"] == "measured" and 0.75 <= prov["confidence"] <= 0.85 and "bounce" in prov["ref"]
    mute = param(result, "mute")
    assert "silent" in mute["does"] and "max (On)" in mute["does"]


def test_a_parameter_that_changes_nothing_says_so_with_low_confidence(gain: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    result, _ = gain
    d = param(result, "dither")
    assert d["does"] == "no measurable effect in default mode"
    assert d["provenance"]["does"]["source"] == "measured" and d["provenance"]["does"]["confidence"] == 0.3
    assert validate_profile(result.profile) == []


def test_a_filter_cutoff_moves_the_spectral_centroid(filt: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    result, _ = filt
    text = param(result, "cutoff")["does"]
    assert "spectral centroid" in text and "min (200.0 Hz)" in text and "max (16.00 kHz)" in text
    import re

    centroids = [int(n) for n in re.findall(r"(\d+) Hz", text.split("spectral centroid")[1].split(";")[0])]
    # centroid is darker at min than at mid and mid than at max
    assert centroids[-3] < centroids[-2] < centroids[-1] or len(centroids) >= 3
    assert param(result, "smoothing")["does"] == "no measurable effect in default mode"


def test_a_hosted_param_is_described_with_the_plugins_own_value_text(hosted: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    result, _ = hosted
    assert "min (20.0 Hz)" in param(result, "freq")["does"]
    assert "mid (632.5 Hz)" in param(result, "freq")["does"]


def test_an_instrument_is_measured_with_a_chord_and_a_staccato_note(synth: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    result, studio = synth
    cmds = [c for c, _ in studio.calls]
    assert "mix.instrument" in cmds and "midi.new_clip" in cmds and cmds.count("midi.note_add") == 4
    pitches = sorted(p["pitch"] for c, p in studio.calls if c == "midi.note_add")
    assert pitches == [60, 64, 67, 72]
    assert "file.import_audio" not in cmds
    assert "mix.instrument_param" in cmds and "mix.insert_param" not in cmds
    release = param(result, "release")["does"]
    assert "decay" in release and "staccato" in release
    assert "output level" in param(result, "level")["does"]
    assert "spectral" in param(result, "cutoff")["does"]
    assert param(result, "glide")["does"] == "no measurable effect in default mode"


def test_effects_are_measured_with_noise_sweep_and_drums_on_an_audio_track(filt: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    _, studio = filt
    cmds = [c for c, _ in studio.calls]
    imports = [p for c, p in studio.calls if c == "file.import_audio"]
    assert len(imports) == 1 and imports[0]["path"].endswith("composite.wav")
    assert "mix.insert" in cmds and "midi.new_clip" not in cmds
    kinds = [p["kind"] for c, p in studio.calls if c == "track.new"]
    assert kinds == ["audio"]


def test_everything_is_offline_and_the_other_params_stay_at_default(filt: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    _, studio = filt
    assert not [c for c, _ in studio.calls if c.startswith("transport.")], "no playback commands"
    sets = [(p["param"], p["value"]) for c, p in studio.calls if c == "mix.insert_param"]
    # every set is followed by a bounce; each parameter is restored to its default after its points
    assert sets[-1] == ("smoothing", 20.0)
    assert [v for pid, v in sets if pid == "cutoff"][-1] == 8000.0
    bounce_ends = {p["end"] for c, p in studio.calls if c == "file.bounce_mix"}
    assert max(bounce_ends) <= int(3.0 * 48000), "the per-bounce cap is honoured"
    assert all(p["bit_depth"] == "32f" and p["dither"] is False for c, p in studio.calls if c == "file.bounce_mix")


def test_max_params_caps_the_measurement_and_says_so() -> None:
    result, _ = run("fake_gain", max_params=1)
    assert result.measured == 1
    assert "does" in param(result, "gain") and "does" not in param(result, "mute")
    assert any("measured 1 of 3 parameters" in w for w in result.warnings)
    assert len(param(result, "mute")["calibration"]) == 2, "calibration still covers every parameter"
    zero, studio = run("fake_gain", max_params=0)
    assert zero.measured == 0 and studio.bounces == 0


def test_only_params_restricts_the_measurement() -> None:
    result, studio = run("fake_gain", only_params=["mute"])
    assert [p["id"] for p in result.profile["params"] if "does" in p] == ["mute"]
    assert {p["param"] for c, p in studio.calls if c == "mix.insert_param"} == {"mute"}


def test_the_bounce_cap_sets_the_length_of_the_signals() -> None:
    _, studio = run("fake_gain", max_bounce_s=3.0, only_params=["mute"])
    _, studio_long = run("fake_gain", max_bounce_s=9.0, only_params=["mute"])
    short = {p["end"] for c, p in studio.calls if c == "file.bounce_mix"}
    long = {p["end"] for c, p in studio_long.calls if c == "file.bounce_mix"}
    assert max(short) < max(long) <= 9 * 48000


def test_a_parameter_that_cannot_be_set_is_skipped_with_a_warning() -> None:
    studio = FakeStudio()
    studio.fail_param_value["gain"] = "the plugin refused"
    result, _ = run("fake_gain", studio)
    assert "does" not in param(result, "gain")
    assert "does" in param(result, "mute") and "does" in param(result, "dither")
    assert any("`gain` not measured" in w and "refused" in w for w in result.warnings)
    assert validate_profile(result.profile) == []


def test_measurement_stops_after_repeated_failures() -> None:
    from atprof_fake_studio import FakeParam, FakePlugin

    plugins = {
        "many": FakePlugin(
            "many", "Many", "Other", [FakeParam(f"p{i}", f"P{i}", 0.0, 1.0, 0.5, "None") for i in range(8)],
            process=lambda audio, sr, p: audio,
        )
    }
    studio = FakeStudio(plugins=plugins)
    studio.fail["mix.insert_param"] = "gone"
    result, _ = run("many", studio)
    assert result.measured == 0
    assert any("stopped after 5 failures" in w for w in result.warnings)
    assert validate_profile(result.profile) == []


def test_a_failing_bounce_skips_measurement_but_keeps_the_host_data() -> None:
    studio = FakeStudio()
    studio.fail["file.bounce_mix"] = "no audio device"
    result, _ = run("fake_gain", studio)
    assert result.measured == 0 and any("measurement skipped" in w for w in result.warnings)
    assert len(param(result, "gain")["calibration"]) == 21
    assert validate_profile(result.profile) == []


def test_a_non_deterministic_plugin_widens_the_tolerances() -> None:
    studio = FakeStudio()
    studio.bounce_noise = 0.03
    result, _ = run("fake_gain", studio, only_params=["dither", "gain"])
    assert "output level" in param(result, "gain")["does"]
    assert any("differs between identical bounces" in w for w in result.warnings)
    assert validate_profile(result.profile) == []


def test_deterministic_runs_give_identical_drafts() -> None:
    a, _ = run("fake_filter", only_params=["cutoff"])
    b, _ = run("fake_filter", only_params=["cutoff"])
    assert profiles.dump_profile(a.profile) == profiles.dump_profile(b.profile)


def test_the_work_folder_is_removed_unless_kept(tmp_path: Path) -> None:
    studio = FakeStudio()
    profile_plugin(studio, "fake_gain", ProfilerOptions(**OPTS, only_params=["mute"]))
    import_path = Path(next(p["path"] for c, p in studio.calls if c == "file.import_audio"))
    assert not import_path.exists() and not import_path.parent.exists()
    studio2 = FakeStudio()
    profile_plugin(studio2, "fake_gain", ProfilerOptions(**OPTS, only_params=["mute"], work_dir=tmp_path / "w"))
    assert (tmp_path / "w" / "composite.wav").is_file()


# ---- the scratch session -----------------------------------------------------------------------------------


def test_a_headless_studio_gets_a_fresh_scratch_session_that_is_closed(gain: tuple[profiler.ProfileResult, FakeStudio]) -> None:
    result, studio = gain
    assert result.scratch_mode == "new"
    assert [c for c, _ in studio.calls][:2] == ["session.new", "engine.plugins"]
    assert studio.calls[-1][0] == "session.close"
    assert studio.state.tracks == []


def test_a_desktop_app_with_a_session_open_is_refused_without_touching_it() -> None:
    studio = FakeStudio("control")
    studio.execute("track.new", {"kind": "audio", "name": "Vocals"})
    studio.state.path = "/music/song.scraft"
    before = [c for c, _ in studio.calls]
    with pytest.raises(ProfilerError, match="--allow-current-session"):
        profile_plugin(studio, "fake_gain", ProfilerOptions(**OPTS))
    assert [c for c, _ in studio.calls][len(before) :] == []
    assert [t.name for t in studio.state.tracks] == ["Vocals"]


def test_a_desktop_app_with_an_unsaved_but_empty_session_still_counts_as_in_use() -> None:
    studio = FakeStudio("control")
    studio.dirty = True  # e.g. tempo changed in an empty session
    with pytest.raises(ProfilerError, match="session"):
        profile_plugin(studio, "fake_gain", ProfilerOptions(**OPTS))


def test_an_untouched_desktop_session_is_used_as_scratch_and_replaced_by_an_identical_one() -> None:
    studio = FakeStudio("control")
    studio.state.name = "Untitled"
    result, _ = run("fake_gain", studio, measure=False)
    assert result.scratch_mode == "pristine"
    assert studio.state.tracks == [] and studio.dirty is False and studio.state.path is None
    assert studio.calls[-1] == ("session.new", {"name": "Untitled", "sample_rate": 48000})


def test_allow_current_session_works_in_the_open_session_and_undoes_every_scratch_step() -> None:
    studio = FakeStudio("control")
    studio.execute("track.new", {"kind": "audio", "name": "Vocals"})
    studio.execute("track.new", {"kind": "audio", "name": "Drums"})
    studio.state.path = "/music/song.scraft"
    history_before = studio.execute("engine.history")["undo"]
    snapshot = [(t.id, t.name, t.kind) for t in studio.state.tracks]
    result = profile_plugin(studio, "fake_filter", ProfilerOptions(**OPTS, allow_current_session=True, only_params=["cutoff"]))
    assert result.scratch_mode == "current"
    assert [(t.id, t.name, t.kind) for t in studio.state.tracks] == snapshot
    assert studio.execute("engine.history")["undo"] == history_before
    assert studio.state.path == "/music/song.scraft"
    assert not [c for c, _ in studio.calls if c in ("session.new", "session.close")]
    assert studio.bounces > 0 and "does" in param(result, "cutoff")


def test_working_in_the_open_session_isolates_the_scratch_track_from_the_users_audio(tmp_path: Path) -> None:
    """The user's own (loud, processed) tracks must not leak into the bounces, and nothing of theirs changes."""
    import numpy as np

    from audiokit import wav

    reference, _ = run("fake_filter", only_params=["cutoff"])
    loud = tmp_path / "loud.wav"
    wav.write(loud, (np.random.default_rng(1).standard_normal((2, 48000 * 3)) * 0.3).astype(np.float32), 48000, "32f")
    studio = FakeStudio("control")
    studio.execute("track.new", {"kind": "audio", "name": "Vocals"})
    studio.execute("file.import_audio", {"path": str(loud), "track": "Vocals", "at": 0})
    studio.execute("mix.insert", {"track": "Vocals", "slot": 0, "plugin": "fake_gain"})
    studio.state.path = "/music/song.scraft"
    result = profile_plugin(studio, "fake_filter", ProfilerOptions(**OPTS, allow_current_session=True, only_params=["cutoff"]))
    assert param(result, "cutoff")["does"] == param(reference, "cutoff")["does"]
    assert [t.solo for t in studio.state.tracks] == [False]
    assert any("master or busses" in w for w in result.warnings)


def test_leftover_scratch_work_is_reported_when_the_undo_fails() -> None:
    studio = FakeStudio("control")
    studio.execute("track.new", {"kind": "audio", "name": "Vocals"})
    studio.fail["edit.undo"] = "nothing to undo"
    result = profile_plugin(studio, "fake_gain", ProfilerOptions(**OPTS, allow_current_session=True, measure=False))
    assert any("cleaning up the scratch session failed" in w for w in result.warnings)


def test_a_failed_run_still_cleans_up_the_scratch_session() -> None:
    studio = FakeStudio("control")
    studio.execute("track.new", {"kind": "audio", "name": "Vocals"})
    studio.fail["plugin.describe"] = "plugin crashed"
    with pytest.raises(ProfilerError, match="crashed"):
        profile_plugin(studio, "fake_gain", ProfilerOptions(**OPTS, allow_current_session=True))
    assert [t.name for t in studio.state.tracks] == ["Vocals"]
    assert studio.execute("engine.history")["undo"] == ["New..."]


def test_scratch_session_context_reports_its_mode() -> None:
    with ScratchSession(FakeStudio()) as scratch:
        assert scratch.mode == "new"


# ---- instruments and hosted plugins are found -------------------------------------------------------------------


def test_an_instrument_the_lists_do_not_know_is_found_by_its_description() -> None:
    studio = FakeStudio()
    studio.cmd_engine_plugins = lambda p: []  # type: ignore[method-assign]  # e.g. an Audio Unit: no scan list to ask
    result, _ = run("fake_synth", studio, only_params=["level"])
    cmds = [c for c, _ in studio.calls]
    assert "mix.instrument" in cmds and "mix.instrument_param" in cmds
    assert "track.delete" in cmds, "the wrong-kind scratch track is removed"
    assert result.profile["summary"] == "Instrument with 4 parameters."
    assert "output level" in param(result, "level")["does"]


def test_an_effect_id_with_an_instrument_hint_falls_back_to_an_effect_track() -> None:
    studio = FakeStudio()
    rig = profiler.build_rig(studio, "fake_gain", prefer_instrument=True)
    assert rig.instrument is False


def test_hosted_plugins_are_looked_up_in_the_scan_list() -> None:
    studio = FakeStudio()
    assert profiler._lookup_instrument(studio, "vst3:FAKEHOSTED0001") is False
    assert profiler._lookup_instrument(studio, "fake_synth") is True
    assert profiler._lookup_instrument(studio, "au:aufx:xxxx:appl") is None
    assert profiler._lookup_instrument(studio, "clap:nothing") is None


# ---- hostile output from the studio ---------------------------------------------------------------------------


def good_describe() -> dict[str, Any]:
    return {
        "available": True,
        "plugin": {"id": "p", "format": "vst3", "name": "P", "vendor": "V", "version": "1", "category": "Fx", "is_instrument": False},
        "params": [{"id": "a", "name": "A", "min": 0.0, "max": 1.0, "default": 0.5, "unit": "None", "taper": "Linear", "choices": [], "value": 0.5, "text": "0.50"}],
    }


def test_parse_describe_skips_malformed_rows_and_keeps_the_good_ones() -> None:
    d = good_describe()
    d["params"] += [
        "garbage", {"id": "nan", "min": float("nan"), "max": 1, "default": 0}, {"id": "inv", "min": 5, "max": 1, "default": 2},
        {"id": True, "min": 0, "max": 1, "default": 0}, {"id": "a", "min": 0, "max": 1, "default": 0}, {"min": 0, "max": 1, "default": 0},
        {"id": "far", "min": 0, "max": 1, "default": 9, "value": "x", "choices": "no"},
    ]
    host = parse_describe(d)
    assert [p.id for p in host.params] == ["a", "far"]
    assert host.params[1].default == 1.0 and host.params[1].choices == ()
    assert len(host.warnings) == 6


@pytest.mark.parametrize(
    "bad",
    [None, [], {}, {"plugin": []}, {"plugin": {"name": "x"}, "params": []}, {"plugin": {"id": "x"}}, {"plugin": {"id": "x"}, "params": {}}, {"available": False, "plugin": {"id": "x"}, "params": []}],
)
def test_parse_describe_rejects_nonsense(bad: Any) -> None:
    with pytest.raises(ProfilerError):
        parse_describe(bad)


def test_more_params_than_the_schema_allows_are_truncated_with_a_warning() -> None:
    d = good_describe()
    d["params"] = [{"id": f"p{i}", "name": "x", "min": 0, "max": 1, "default": 0} for i in range(profiler.MAX_PARAMS + 5)]
    host = parse_describe(d)
    assert len(host.params) == profiler.MAX_PARAMS and any("more than" in w for w in host.warnings)


def test_text_fields_are_bounded_and_stripped_of_nuls() -> None:
    d = good_describe()
    d["plugin"]["id"] = "vst3:P"
    d["plugin"]["name"] = "N" * 500 + "\x00"
    d["params"][0]["name"] = "x" * 500
    host = parse_describe(d)
    assert len(host.name) == 200 and "\x00" not in host.name and len(host.params[0].name) == 200
    assert validate_profile(profiler.build_host_profile(host, profiled="2026-01-01")) == []


# ---- the comparison and summary helpers ---------------------------------------------------------------------------


def feats(rms: float, centroid: float = 1000.0, decay: float = 0.5, peak: float | None = None) -> profiler.Features:
    return {"noise": {"rms_db": rms, "peak_db": rms + 12 if peak is None else peak, "centroid_hz": centroid, "bandwidth_hz": 500.0}, "clicks": {"rms_db": rms, "peak_db": rms + 12, "decay_s": decay}}


def test_compare_features_applies_the_tolerances() -> None:
    tol = profiler.Tolerances()
    base = feats(-20.0)
    assert profiler.compare_features(base, feats(-20.1), tol) == []
    gone_up = profiler.compare_features(base, feats(-14.0, centroid=2000.0, decay=1.5), tol)
    metrics = {(e.segment, e.metric) for e in gone_up}
    assert ("noise", "rms_db") in metrics and ("noise", "centroid_hz") in metrics and ("clicks", "decay_s") in metrics
    centroid = next(e for e in gone_up if e.metric == "centroid_hz")
    assert centroid.delta == pytest.approx(1.0)  # one octave
    assert gone_up == sorted(gone_up, key=lambda e: -e.significance)


def test_silence_to_silence_is_not_an_effect_and_silence_to_sound_is() -> None:
    tol = profiler.Tolerances()
    assert profiler.compare_features(feats(-200.0), feats(-190.0), tol) == []
    woke = profiler.compare_features(feats(-200.0), feats(-20.0), tol)
    assert any(e.metric == "rms_db" and e.base <= -100 for e in woke)


def test_derive_tolerances_follow_the_run_to_run_difference() -> None:
    tol, noisy = profiler.derive_tolerances(feats(-20.0), feats(-20.0), feats(-20.0))
    assert tol == profiler.Tolerances() and not noisy
    tol, noisy = profiler.derive_tolerances(feats(-20.0), feats(-19.0, centroid=1100.0), feats(-20.2))
    assert noisy and tol.level_db == pytest.approx(4.8) and tol.octaves > 0.05


def test_probe_points_are_min_mid_max_without_the_current_value_or_duplicates() -> None:
    def hp(**kw: Any) -> profiler.HostParam:
        base = dict(id="x", name="X", min=0.0, max=10.0, default=5.0, unit="None", taper="Linear", choices=(), value=5.0, text="")
        return profiler.HostParam(**{**base, **kw})

    assert profiler.probe_points(hp(value=0.0)) == [("mid", 5.0), ("max", 10.0)]
    assert profiler.probe_points(hp(value=3.0)) == [("min", 0.0), ("mid", 5.0), ("max", 10.0)]
    log = profiler.probe_points(hp(min=100.0, max=10000.0, taper="Log", value=100.0))
    assert log[0] == ("mid", pytest.approx(1000.0))
    toggle = profiler.probe_points(hp(unit="Toggle", min=0.0, max=1.0, value=0.0))
    assert toggle == [("max", 1.0)], "the rounded middle of a toggle is its top, not a third point"
    three = profiler.probe_points(hp(unit="Choice", min=0.0, max=2.0, value=0.0, choices=("a", "b", "c")))
    assert three == [("mid", 1.0), ("max", 2.0)]
    assert math.isclose(profiler.plain_from_travel(hp(min=100.0, max=10000.0, taper="Log"), 0.5), 1000.0)


def test_summaries_pick_the_strongest_point_and_cap_the_confidence() -> None:
    hp = profiler.HostParam("g", "Gain", -60.0, 24.0, 0.0, "Db", "Linear", (), 0.0, "")
    quiet = profiler.Effect("noise", "rms_db", -18.0, -19.0, -1.0, 4.0)
    loud = profiler.Effect("noise", "rms_db", -18.0, 6.0, 24.0, 96.0, clipped=True)
    text, conf = profiler.summarize_measurement(hp, [profiler.PointResult("min", -60.0, (quiet,)), profiler.PointResult("max", 24.0, (loud,))], {"noise": "pink noise"})
    assert "-1.0 dB" in text and "+24.0 dB (reaches full scale)" in text and conf == 0.85
    weak = profiler.Effect("noise", "rms_db", -18.0, -18.4, -0.4, 1.6)
    _, conf = profiler.summarize_measurement(hp, [profiler.PointResult("max", 24.0, (weak,))], {})
    assert conf == 0.55
    assert profiler.summarize_measurement(hp, [profiler.PointResult("max", 24.0, ())], {}) == ("no measurable effect in default mode", 0.3)
