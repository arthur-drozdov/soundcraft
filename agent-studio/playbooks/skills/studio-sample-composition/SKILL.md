---
name: studio-sample-composition
description: Use when the user wants to build music from generated or imported audio loops and one-shots. Covers generating candidates with Forge, auditioning them, importing, chopping at transients, resequencing on the grid, processing and bouncing a preview.
---

# Sample-first composition

Follow `studio-verified-edit` for every session change, and `studio-session-conventions` for names, colours and tags. This skill adds the sourcing, chopping and sequencing steps. Workflow: generate, audition, import, chop, resequence, process, bounce.

## 0. Set up
1. `session.inspect {"detail": "summary"}`. Note the tempo, meter and markers. Set the tempo if the user gave one: `event.tempo {"bpm": 130}`. Key and bar count come from the request (for example 130 BPM, F minor, 16 bars).
2. `session.checkpoint {"label": "start-composition"}`.
3. Bar length in samples = `4 * 60 / bpm * sample_rate` in 4/4 (130 BPM at 48 kHz: 88615). Use it to check clip lengths later.

## 1. Generate candidates (Forge)
1. `forge_health`. If Forge is offline, say so and offer to work with audio the user already has (import it and skip to step 3).
2. `forge_search {"text": "garage bass", "type": "generate.loop"}` first: reuse a rated earlier asset before generating a new one. `forge_models` shows what is available and each model's licence.
3. Submit with `forge_submit`. Always ask for 2 to 4 `candidates`, and always give `bpm` (loops) and `key` (tonal material) so the audio fits the session:
   - Loop: `{"type": "generate.loop", "params": {"prompt": "dark garage bass, sub heavy, dry", "bars": 4, "bpm": 130, "key": "F minor", "post": {"loop_crossfade_ms": 10}}, "candidates": 3, "wait_s": 120, "fetch": true}`.
   - One-shot: `{"type": "generate.sample", "params": {"prompt": "tight punchy kick", "note": 36, "length_s": 1.0, "post": {"trim_silence": true, "fade_out_ms": 20}}, "candidates": 3, "wait_s": 60, "fetch": true}`. Do at least three percussion one-shots when asked for a drum pattern (kick, snare or clap, hat).
   - Texture or riser: `{"type": "generate.sfx", "params": {"prompt": "noise riser, rising filter", "duration_s": 6}, "candidates": 2, "wait_s": 60, "fetch": true}`.
   - Change existing audio: `forge_upload {"path": "<project>/Previews/...wav"}` returns an `asset_id`; then `{"type": "transform.audio", "inputs": {"audio": "<asset_id>"}, "params": {"prompt": "...", "strength": 0.5}}`, or `"type": "layer.add"` with `"params": {"description": "..."}`.
4. If a job is still running after `wait_s`, poll `forge_job {"job_id": "..."}`. Use `forge_cancel` if the user changes their mind. Do not block longer than `wait_s`.

## 2. Audition
- Each result has a sidecar summary with `requested` and `measured` values: `bpm`, `key`, `peak_dbfs`, `true_peak_dbfs`, `lufs`, `centroid_hz`, `loop_seam`, `duration_s`, plus `warnings`. Drop candidates with a bpm mismatch (for example half time), clipping or a poor loop seam.
- Present a short table: candidate, measured BPM and key, duration, peak, LUFS, warnings. The user listens and picks; do not claim a sound is good.
- After the user chooses, record it: `forge_rate {"asset_id": "a_1234abcd", "rating": 4, "note": "tight kick, used in drop"}`.

## 3. Import the chosen candidates
- `file.import_audio {"path": "<Forge Inbox path>", "at": "bars_beats:1|1|000", "track": "DRM", "new_tracks": false}` puts the clip on an existing role track. Without `track` and with `new_tracks: true` it makes a new track named after the file. The result lists `track`, `clip`, `source` and `frames`.
- Rename and tag new tracks (`track.rename`, `track.color`, `track.comments`, see `studio-session-conventions`). Put the Forge asset id, model and licence from the sidecar into the track comment.
- Check the clip length against the bar length: a 4-bar loop should be `4 * bar length` samples within about 1 percent. If it is a little off, use `clip.conform_to_tempo {"clips": [12]}` or trim the end with `edit.trim_clip {"clip": 12, "end": "bars_beats:5|1|000"}`. If it is far off (more than 8 percent), regenerate with the right `bpm`.
- Verify the material: `audio.analyze {"clip": 12}` and compare `bpm`, `key` and `onsets_s` with the sidecar.

## 4. Chop
- On transients: `edit.separate_at_transients {"tracks": ["BAS"], "start": "bars_beats:1|1|000", "end": "bars_beats:5|1|000", "sensitivity": 0.5}` returns `{"separated": n}`. Raise `sensitivity` towards 0.8 for more slices, lower it for fewer. Check the slice count with `track.inspect`; it should match the number of hits you expect.
- On the grid: `edit.separate_on_grid {"tracks": ["DRM"], "start": "bars_beats:1|1|000", "end": "bars_beats:5|1|000"}` after setting the grid with `view.grid {"value": "1/16"}`.
- Slices begin a fraction of a millisecond before the detected onset. Put every slice you keep on an exact grid position in the next step.
- Clean up: `edit.strip_silence {"tracks": ["DRM"], "threshold_db": -48}`, `edit.heal` over a gap you want closed, and `edit.fades_create {"tracks": ["BAS"], "shape": "equal power", "start": ..., "end": ...}` to avoid clicks.

## 5. Resequence
- Place a slice: `edit.move_clips {"clips": [14], "to": "bars_beats:2|3|000", "track": "BAS"}`. Several clips in one call keep their distance from each other, so use one call per clip to place each on its own grid position. Put every clip start and end on a beat or 16th so bounces are exact.
- Two-step drum pattern from one-shots: import each one-shot once (`at` bar 1 beat 1 for the kick, bar 1 beat 3 for the snare), then `edit.repeat {"tracks": ["DRM"], "start": "bars_beats:1|1|000", "end": "bars_beats:2|1|000", "count": 7}` copies that bar seven more times (8 bars in all).
- Copy a range: `edit.duplicate {"tracks": ["DRM"], "start": "bars_beats:1|1|000", "end": "bars_beats:9|1|000"}` places the copy straight after the range. For another position use `edit.copy {...}` then `edit.paste {"tracks": ["DRM"], "at": "bars_beats:9|1|000"}`.
- Loop one clip: `clip.loop {"clips": [12], "count": 8}`.
- After placing, check with `track.inspect` that starts are on the grid (a bar at 130 BPM and 48 kHz is 88615 samples; positions in samples must be within 48 of the grid, which is 1 ms) and that clips on the same track do not overlap unless you meant them to.

## 6. Process and automate
- Level: `clip.gain {"clips": [12], "db": -3}`; track level `mix.volume {"tracks": ["BAS"], "db": -4}`.
- Inserts: `mix.insert {"track": "DRM", "slot": 0, "plugin": "compressor", "params": {"threshold": -18, "ratio": 3}}`. Load the plugin's Atlas profile first (`studio-plugin-profiling`); without a profile use only built-ins you can read back with `plugin.describe`.
- Offline processing of a clip: `audiosuite.process {"process": "reverse", "clips": [12]}`.
- Automation, for example a filter opening on the texture: insert `eq_1band` as a low-pass (`mix.insert {"track": "SYN", "slot": 0, "plugin": "eq_1band", "params": {"type": 4, "freq": 600}}`), then `automation.set_point {"track": "SYN", "param": "plugin:a:freq", "at": "bars_beats:1|1|000", "value": 600}` and a second point at `bars_beats:9|1|000` with value 12000. Put breakpoints on beat boundaries and verify by bouncing.

## 7. Bounce a preview and report
Bounce with `file.bounce_mix` and measure as in `studio-verified-edit` step 5. Report: the tracks and clips made, positions in bars, the Forge asset ids with model and licence from each sidecar, parameters old to new, `true_peak_db` and `lufs` of the preview, the checkpoint path and the `edit.undo` count. Ask the user to listen.
