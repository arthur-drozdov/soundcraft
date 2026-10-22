---
name: studio-playable-instrument
description: Use when the user wants generated or imported sound to be played from MIDI. Covers a single sample on the built-in Sampler and multi-sample keybed instruments from Forge via SFZ import, with root notes, key ranges, loops and a bounce that proves every note plays.
---

# Playable sample instruments

Follow `studio-verified-edit`. Create instrument tracks with a role prefix (usually `SYN`, `BAS` or `DRM`) and tag them (`studio-session-conventions`). Note numbers are MIDI: 60 is C4, 36 is C2.

## A. Single sample (one pitched note or one-shot)
1. Generate: `forge_submit {"type": "generate.sample", "params": {"prompt": "warm electric piano, single note", "note": 60, "length_s": 3}, "candidates": 3, "wait_s": 60, "fetch": true}`. For a drum hit use a short `length_s` and `"post": {"trim_silence": true}`.
2. Read the sidecar. The root note is `requested.note`. The measured `key` of a single note is only a weak cross-check. If the user hears it out of tune, correct it with `fine_tune` (cents) or `root_note`, not by regenerating.
3. Create the track: `track.new {"kind": "instrument", "name": "SYN Keys"}`. `mix.sampler_load` turns it into a Sampler track.
4. Load: `mix.sampler_load {"track": "SYN Keys", "path": "<Forge Inbox path>.wav", "params": {"root_note": 60, "key_low": 48, "key_high": 72, "attack": 2, "release": 300}}`. It returns `{track, source, name, frames, channels, sample_rate}`; check `frames` against the sidecar duration.
5. Sampler parameter ids (set with `params` above or `mix.instrument_param {"track": "SYN Keys", "param": "release", "value": 300}`):
   - `root_note` 0..127, `transpose` semitones, `fine_tune` cents, `key_low` and `key_high` 0..127, `vel_low` and `vel_high` 1..127, `velocity` 0..100 percent.
   - `attack`, `decay`, `release` in milliseconds (attack minimum 0.1), `sustain` percent, `level` dB.
   - `play_mode` is a choice: 0 Gate, 1 One Shot. Use 1 for drum hits and plucks that should ring out.
   - `loop_mode` is a choice: 0 Off, 1 Forward, 2 Ping-Pong. `loop_start` and `loop_end` are percent of the sample length (0..100). For a sustained sound set `{"loop_mode": 1, "loop_start": 40, "loop_end": 90}` from a steady region away from the attack, and check `loop_seam` in the sidecar.
   - List everything with ranges and current values: `plugin.describe {"track": "SYN Keys", "instrument": true}`.
6. The built-in Sampler holds one sample per track, so wide pitch ranges sound unnatural. Keep `key_low` and `key_high` within about 12 semitones of the root, or build a keybed (section B). `mix.sampler_clear {"track": "SYN Keys"}` removes the sample.

## B. Multi-sample keybed
1. Generate: `forge_submit {"type": "generate.instrument", "params": {"prompt": "warm electric piano", "note_low": 36, "note_high": 84, "step": 3, "velocity_layers": 1, "length_s": 3, "name": "ep_keys"}, "wait_s": 0}`. It is a long job: poll `forge_job {"job_id": "..."}` and tell the user the progress and ETA.
2. `forge_fetch {"asset_id": "<id>"}` extracts the bundle into the inbox and returns the path of the `.sfz`. Check the zone list in the sidecar: every key from `note_low` to `note_high` must be covered exactly once, and each zone's root note should sit near the middle of its key range.
3. Create the track (`track.new {"kind": "instrument", "name": "SYN Keys"}`), then import: `file.import_sfz {"path": "<.sfz path>", "track": "SYN Keys"}` returns `{track, zones, skipped}`. Report each `skipped` reason to the user. DecentSampler presets: `file.import_dspreset {"path": "<.dspreset path>", "track": "SYN Keys"}`.
4. Or set zones directly: `mix.sampler_zones_set {"track": "SYN Keys", "zones": [{"path": "<project>/Forge Inbox/2026-10-22/ep_keys/samples/c4.wav", "root_note": 60, "key_low": 59, "key_high": 61, "vel_low": 1, "vel_high": 127, "tune_cents": 0, "gain_db": 0}]}`. Zone fields: `path`, `root_note`, `key_low`, `key_high`, `vel_low`, `vel_high`, `tune_cents`, `gain_db`, `loop_mode`, `loop_start`, `loop_end`. Shared parameters (envelope, filter) go in the call's `params`. Check the loop units with `list_commands` on your build before setting loop points.
5. Verify: `zones` in the result matches the sidecar zone list, and `track.inspect {"track": "SYN Keys"}` shows the instrument.

## C. Prove it plays
1. `midi.new_clip {"track": "SYN Keys", "start": "bars_beats:1|1|000", "end": "bars_beats:3|1|000"}` returns the clip id. Add notes with `midi.note_add {"clip": 5, "pitch": 40, "start_ticks": 0, "length_ticks": 960, "velocity": 100}`. Ticks are relative to the clip start, 960 per quarter note, 3840 per bar. Add the bottom, the middle and the top of the key range, one per beat, and a chord.
2. `file.bounce_mix {"path": "<project>/Previews/20261022-keys-test.wav", "start": "bars_beats:1|1|000", "end": "bars_beats:3|1|000", "bit_depth": "24"}`. It must not be silent, and `true_peak_db` must be at or below -1.0. If the instrument is too loud, lower the Sampler `level` (dB) or `mix.volume`.
3. If a note is silent, a zone is missing or the key range is wrong. Fix it, bounce again.
4. For a quick tuning check import the bounce onto a scratch track, `audio.analyze {"clip": id}` and compare `key`; delete the scratch track afterwards. Otherwise leave tuning to the user's ears.
5. Report: instrument name, root notes and key ranges as note names (60 is C4), envelope and loop settings, source asset ids with model and licence, the preview path with `true_peak_db` and `lufs`. Ask the user to listen for tuning and loop clicks.

## Use the same MIDI on another instrument
A MIDI clip is independent of the instrument. To audition a different sound, duplicate the track (`track.duplicate {"tracks": ["SYN Keys"]}`; read the copy's id from `session.inspect`), change its instrument with `mix.instrument {"track": "<copy id>", "plugin": "subtractive_synth", "params": {"cutoff": 2400}}`, and compare. Third-party instruments only with an Atlas profile (`studio-plugin-profiling`).
