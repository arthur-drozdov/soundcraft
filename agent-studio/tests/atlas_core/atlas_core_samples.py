"""Sample profiles for the Atlas core tests."""

from __future__ import annotations

import copy
from typing import Any

HOST = {"source": "host", "confidence": 1.0}


def sample_profile(**identity: Any) -> dict[str, Any]:
    ident = {
        "name": "Acme Squash",
        "vendor": "Acme Audio",
        "format": "vst3",
        "plugin_id": "vst3:ABCDEF0123456789",
        "version": "2.3.1",
        "os": ["macos", "windows"],
        "profiled": "2026-10-22",
    }
    ident.update(identity)
    return copy.deepcopy(
        {
            "identity": ident,
            "summary": "A feed-forward compressor.",
            "params": [
                {
                    "id": "thresh",
                    "host_name": "Threshold",
                    "min": -60.0,
                    "max": 0.0,
                    "default": -18.0,
                    "unit": "dB",
                    "taper": "linear",
                    "provenance": {"unit": dict(HOST), "min": dict(HOST), "max": dict(HOST), "default": dict(HOST), "taper": {"source": "inferred", "confidence": 0.6}},
                },
                {
                    "id": "ratio",
                    "host_name": "Ratio",
                    "min": 1.0,
                    "max": 20.0,
                    "default": 4.0,
                    "calibration": [{"value": 0.0, "text": "1:1"}, {"value": 0.5, "text": "4:1"}],
                    "provenance": {"min": dict(HOST), "max": dict(HOST), "default": dict(HOST), "calibration": {"source": "host", "confidence": 1.0}},
                },
                {
                    "id": 7,
                    "host_name": "Mode",
                    "choices": ["Off", "On", "Auto"],
                    "provenance": {"choices": dict(HOST)},
                },
            ],
            "provenance": {"identity.version": dict(HOST)},
        }
    )
