"""Opt-in: the profiler and `atlas builtins` against the real headless `soundcraft-cli`.

    SOUNDCRAFT_CLI=/path/to/soundcraft-cli python -m pytest tests/atlas_profiler/test_atprof_integration.py

Skipped when `SOUNDCRAFT_CLI` is unset or not an executable file (it needs a build with the Agent
Studio commands: plugin.describe, plugin.param_text).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from atlas import builtins, cli, profiler, profiles
from atlas.validate import validate_profile
from studio_client import Studio

BINARY = os.environ.get("SOUNDCRAFT_CLI", "")

pytestmark = pytest.mark.skipif(
    not BINARY or not Path(BINARY).is_file() or not os.access(BINARY, os.X_OK),
    reason="set SOUNDCRAFT_CLI to a soundcraft-cli binary to run the integration tests",
)


@pytest.fixture(scope="module")
def studio() -> Iterator[Studio]:
    s = Studio.headless(BINARY, start_timeout=60, timeout=120)
    try:
        yield s
    finally:
        s.close()


def run(studio: Studio, plugin_id: str, **opts: Any) -> profiler.ProfileResult:
    return profiler.profile_plugin(studio, plugin_id, profiler.ProfilerOptions(progress=lambda m: None, **opts))


def param(result: profiler.ProfileResult, pid: str) -> dict[str, Any]:
    return next(p for p in result.profile["params"] if p["id"] == pid)


def test_gain_is_profiled_and_measured_headlessly(studio: Studio) -> None:
    result = run(studio, "gain")
    assert validate_profile(result.profile) == []
    assert result.scratch_mode == "new" and result.measured == 1 and result.bounces >= 5
    ident = result.profile["identity"]
    assert (ident["plugin_id"], ident["format"], ident["vendor"]) == ("gain", "builtin", "SoundCraft")
    gain = param(result, "gain")
    assert (gain["unit"], gain["min"], gain["max"], gain["default"]) == ("dB", -96.0, 24.0, 0.0)
    assert len(gain["calibration"]) == 21 and gain["calibration"][0]["text"] == "-inf dB" and gain["calibration"][-1]["text"] == "24.0 dB"
    assert "output level" in gain["does"] and "silent" in gain["does"], gain["does"]
    assert gain["provenance"]["does"]["source"] == "measured"
    assert result.warnings == []


def test_an_eq_band_at_zero_gain_has_no_measurable_frequency_effect(studio: Studio) -> None:
    result = run(studio, "eq_1band")
    assert validate_profile(result.profile) == []
    assert param(result, "freq")["does"] == "no measurable effect in default mode"
    assert param(result, "freq")["provenance"]["does"]["confidence"] == 0.3
    gain = param(result, "gain")
    assert "output level" in gain["does"] or "level at" in gain["does"]
    freq = param(result, "freq")
    assert freq["taper"] == "log" and freq["provenance"]["taper"]["source"] == "host"
    assert any(c["text"].endswith("kHz") for c in freq["calibration"])
    assert param(result, "type")["choices"][0] == "Peak" and "calibration" not in param(result, "type")


def test_a_built_in_instrument_is_measured_with_a_chord_and_a_staccato_note(studio: Studio) -> None:
    result = run(studio, "subtractive_synth", only_params=["cutoff", "amp_release", "level"])
    assert validate_profile(result.profile) == [] and result.measured == 3
    assert "spectral centroid" in param(result, "cutoff")["does"]
    assert "decay" in param(result, "amp_release")["does"]
    assert "output level" in param(result, "level")["does"]


def test_every_builtin_profile_regenerates_and_validates(studio: Studio, tmp_path: Path) -> None:
    out = tmp_path / "builtin" / "soundcraft"
    report = builtins.generate_builtins(studio, out)
    assert report.ok, report.errors
    assert report.plugin_count == len(report.profiles) >= 27
    assert report.plugins_without_curated == [] and report.curated_without_plugin == []
    assert cli.main(["validate", str(tmp_path)]) == 0
    for ref in profiles.list_profiles(tmp_path):
        assert ref.profile["identity"]["vendor"] == "SoundCraft"
    again = builtins.generate_builtins(studio, out)
    assert again.written == [] and len(again.unchanged) == report.plugin_count
