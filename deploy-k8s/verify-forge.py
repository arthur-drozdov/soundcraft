#!/usr/bin/env python3
# Copyright 2026 The SoundCraft Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify a Forge deployment end to end (stdlib only, no venv needed).

Run it after each rollout: it checks health and the GPU report, the model
roster, then does real work on the box - an ACE-Step XL loop on the GPU lane,
a stems.cpp split of that loop, and an SA3 SFX - and confirms the returned
WAVs decode, are the expected length and are not silent.

    python3 deploy-k8s/verify-forge.py http://100.100.7.29:7700 "$FORGE_TOKEN" /tmp/forge-verify

Exits non-zero on the first failed check (all checks still run).
"""
from __future__ import annotations

import json
import math
import struct
import sys
import time
import urllib.request
from array import array
from pathlib import Path

PROBE_BYTES = 8 * 1024 * 1024  # peak is measured over the first ~30 s at most
EXPECTED = ("ace-step-1.5-xl-base", "ace-step-1.5-base", "sa3-small-sfx", "sa3-small-music",
            "sa3-medium", "stems-cpp", "yin", "testtone", "analysis", "soundcraft")


class Forge:
    def __init__(self, base: str, token: str) -> None:
        self.base, self.token = base.rstrip("/"), token

    def req(self, method: str, path: str, body: object = None, raw: bool = False):
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.base + path, data=data, method=method)
        r.add_header("Authorization", "Bearer " + self.token)
        if data:
            r.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(r, timeout=600) as resp:
            blob = resp.read()
            return blob if raw else json.loads(blob or b"null")

    def submit(self, jtype: str, params: dict, model: str | None = None, inputs: dict | None = None) -> dict:
        body: dict = {"type": jtype, "params": params, "priority": "normal"}
        if model:
            body["model"] = model
        if inputs:
            body["inputs"] = inputs
        return {"type": jtype, **self.req("POST", "/v1/jobs", body)}

    def wait(self, rec: dict, timeout_s: int = 1800) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            job = self.req("GET", f"/v1/jobs/{rec['job_id']}")
            rec["state"], rec["model"], rec["family"] = job["state"], job.get("model"), job.get("family")
            if job["state"] in ("succeeded", "failed", "cancelled"):
                rec["result"] = job.get("result") or {}
                rec["error"] = job.get("error")
                return rec
            time.sleep(5)
        rec.update(state="timeout", error="timeout waiting for job")
        return rec

    def download(self, rec: dict, out: Path) -> list[Path]:
        files = []
        for asset_id in (rec.get("result") or {}).get("asset_ids", []):
            blob = self.req("GET", f"/v1/assets/{asset_id}", raw=True)
            stem = ""
            try:
                meta = self.req("GET", f"/v1/assets/{asset_id}/meta")
                stem = "_" + (meta.get("tags") or {}).get("stem", "")
            except Exception:  # noqa: BLE001 - metadata only names the file nicer
                pass
            path = out / f"{rec['type'].replace('.', '-')}_{rec['job_id']}{stem}.wav"
            path.write_bytes(blob)
            files.append(path)
        return files


def describe(path: Path) -> tuple[float, float, int, int] | None:
    """(seconds, peak, channels, bit depth) for a PCM WAV, or None if not one."""
    blob = path.read_bytes()
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        return None
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(blob):  # walk chunks: metadata (LIST, bext) may sit anywhere
        cid, size = blob[pos:pos + 4], struct.unpack_from("<I", blob, pos + 4)[0]
        body = blob[pos + 8: pos + 8 + size]
        if cid == b"fmt " and len(body) >= 16:
            fmt = body
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)  # chunks are word-aligned
    if fmt is None or data is None:
        return None
    channels, = struct.unpack_from("<H", fmt, 2)
    rate, = struct.unpack_from("<I", fmt, 4)
    bits, = struct.unpack_from("<H", fmt, 14)
    if channels < 1 or rate < 1 or bits not in (16, 24, 32):
        return None
    frames = len(data) // ((bits // 8) * channels)
    probe = data[:PROBE_BYTES]
    if bits == 16:
        values = array("h")
        values.frombytes(probe[: len(probe) - len(probe) % 2])
        peak = max((abs(v) for v in values), default=0.0) / 32768
    elif bits == 32:
        values = array("i")
        values.frombytes(probe[: len(probe) - len(probe) % 4])
        peak = max((abs(v) for v in values), default=0.0) / 2147483648
    else:  # 24-bit: three bytes per sample, little-endian signed
        probe = probe[: len(probe) - len(probe) % 3]
        peak = max((abs(int.from_bytes(probe[i:i + 3], "little", signed=True))
                    for i in range(0, len(probe), 3)), default=0.0) / 8388608
    return frames / rate, peak, channels, bits


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    forge, out = Forge(argv[1], argv[2]), Path(argv[3])
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {label}{': ' + detail if detail else ''}")
        if not ok:
            failures.append(label)

    def check_wav(path: Path, want: tuple[float, float] | None) -> None:
        info = describe(path)
        if info is None:
            check(f"{path.name} is a PCM WAV", False)
            return
        seconds, peak, channels, bits = info
        detail = f"{seconds:.2f}s {channels}ch {bits}bit peak {20 * math.log10(max(peak, 1e-9)):.1f} dBFS"
        check(f"{path.name} decodes", True, detail)
        if want:
            check(f"{path.name} length {want[0]}-{want[1]}s", want[0] <= seconds <= want[1], f"{seconds:.2f}s")
        check(f"{path.name} not silent", peak > 0.01, f"peak {peak:.3f}")

    health = forge.req("GET", "/v1/health")
    check("health status", health.get("status") == "ok", f"version {health.get('version')}")
    gpu = health.get("gpu") or {}
    # GB10 answers [N/A] for memory; availability is the point, the numbers are a bonus
    check("health reports the GPU", bool(gpu.get("available")), str(gpu))
    check("lanes idle", health["lanes"]["gpu"]["running"] == 0 and health["lanes"]["cpu"]["running"] == 0)

    roster = forge.req("GET", "/v1/models")
    roster = roster if isinstance(roster, list) else roster.get("models", [])
    have = {m["id"] for m in roster if m.get("available")}
    check("model roster", set(EXPECTED) <= have, f"{len(have)} available, missing {sorted(set(EXPECTED) - have) or 'none'}")

    loop = forge.wait(forge.submit(
        "generate.loop",
        {"prompt": "warm Rhodes chords, soulful house, vinyl dust", "bpm": 124, "bars": 4,
         "key": "A minor", "seed": 11, "candidates": 1},
        model="ace-step-1.5-xl-base"))
    check("ACE-Step XL loop (GPU lane)", loop.get("state") == "succeeded", f"{loop.get('model')} {loop.get('error') or ''}")
    assets = (loop.get("result") or {}).get("asset_ids") or []
    check("loop produced one asset", len(assets) == 1, str(assets))

    jobs = [loop]
    if assets:
        jobs.append(forge.wait(forge.submit("stems.split", {"stems": ["bass", "drums", "other"]},
                                            inputs={"audio": assets[0]})))
        check("stems.cpp split", jobs[-1].get("state") == "succeeded", f"{jobs[-1].get('family')} {jobs[-1].get('error') or ''}")
        check("split produced stems", len((jobs[-1].get("result") or {}).get("asset_ids") or []) >= 2)

    jobs.append(forge.wait(forge.submit(
        "generate.sfx", {"prompt": "short test whoosh", "duration_s": 1.0,
                         "post": {"normalize": "peak", "target": -3.0}}, model="sa3-small-sfx")))
    check("SA3 SFX", jobs[-1].get("state") == "succeeded", jobs[-1].get("error") or "")

    wants = {0: (7.0, 8.6), 2: (0.7, 1.5)}  # 4 bars @ 124 BPM, and the 1 s SFX
    for index, rec in enumerate(jobs):
        if rec.get("state") != "succeeded":
            continue
        for path in forge.download(rec, out):
            check_wav(path, wants.get(index))

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
