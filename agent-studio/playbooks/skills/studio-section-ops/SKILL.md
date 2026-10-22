---
name: studio-section-ops
description: Use for arrangement-level work. Covers duplicating or moving a section, making a chorus bigger, and building transitions such as risers, filter sweeps, reverses, drum cuts and ducking, using the section markers to find bar positions.
---

# Section operations

Follow `studio-verified-edit` and `studio-session-conventions`. Checkpoint first: `session.checkpoint {"label": "before-section-ops"}`.

## 0. Read the sections
Sections are the markers. Read them from `session.inspect {"detail": "summary"}` (name and position). Convert positions with `session.format_time {"at": 1152000, "format": "bars_beats"}`. If there are no markers, add them first: `markers.add {"name": "Verse", "at": "bars_beats:5|1|000"}`. Work out section lengths in bars from the marker positions, and do all later arithmetic in bars.

## 1. Duplicate a section
Example: copy the verse (bars 5 to 13) so it plays again right after itself. The summary shows `edit_mode`; keep it on Slip (the default), because in Shuffle a paste inserts time and shifts later clips.
1. Open space so nothing is overwritten: `event.insert_time {"start": "bars_beats:13|1|000", "end": "bars_beats:21|1|000"}` (the length is the section length: 8 bars). It inserts silence on every track, shifts later clips and automation, and moves later markers.
2. Copy: `edit.duplicate {"tracks": ["DRM", "BAS", "SYN", "FX", "VOX"], "start": "bars_beats:5|1|000", "end": "bars_beats:13|1|000"}`. The copy lands directly after the range and automation inside it is copied too.
3. Name it: `markers.add {"name": "Verse 2", "at": "bars_beats:13|1|000"}`.
4. Verify: per track, the clip count in bars 13 to 21 equals the count in bars 5 to 13 (`track.inspect`). `audio.analyze {"track": "DRM", "start": "bars_beats:5|1|000", "end": "bars_beats:13|1|000"}` and the same for 13 to 21 should match in level and onset count.

To put the copy somewhere else, open space at the destination with `event.insert_time`, then `edit.copy {"tracks": [...], "start": ..., "end": ...}` and `edit.paste {"tracks": [...], "at": "bars_beats:29|1|000"}`. To move a section, copy it to the destination, then remove the original with `event.cut_time {"start": "bars_beats:5|1|000", "end": "bars_beats:13|1|000"}`, which closes the gap and shifts everything after it (markers inside the removed range are deleted). Recompute marker positions afterwards and fix them with `markers.edit`.

## 2. Make the second chorus bigger
Pick one or two of these; do not stack everything.
- Layer: `forge_submit {"type": "layer.add", "inputs": {"audio": "<chorus asset id>"}, "params": {"description": "wide detuned synth stack, same key"}, "candidates": 3, "wait_s": 90, "fetch": true}`. Upload the chorus first if it is not a Forge asset (`forge_upload`). Import on a new `SYN` track at the chorus start (`file.import_audio {"path": "...", "at": "bars_beats:21|1|000", "new_tracks": true}`) and trim it to the chorus with `edit.trim_clip`.
- Add a sample: a crash or impact from `generate.sfx` on `FX`, placed on the chorus downbeat.
- Mix: `mix.volume {"tracks": ["SYN"], "delta_db": 1.5}` for the whole track, `clip.gain {"clips": [30], "delta_db": 1.5}` for one clip, and widen doubled parts with `mix.pan {"tracks": ["SYN 2"], "pan": 0.4}`.
- Ducking so the drums stay clear: put a compressor on the pad (`mix.insert {"track": "SYN", "slot": 0, "plugin": "compressor", "params": {"threshold": -24, "ratio": 4, "attack": 5, "release": 120}}`) and feed its detector from the kick: `mix.insert_sidechain {"track": "SYN", "slot": 0, "source": "DRM"}`. If `mix.insert_sidechain` is not available in this build, duck with volume automation instead (`automation.set_point` on `volume`, dips on the beat boundaries).
- Compare the bounces of the first and second chorus (`file.bounce_mix` each, 8 bars). A bigger chorus should read about 1 to 2 LU louder in `lufs`, never clipped. Report the difference.

## 3. Transitions
Place them in the last bar or beats before the next section. Positions below assume the next section starts at bar 13.
- Riser: `forge_submit {"type": "generate.sfx", "params": {"prompt": "noise riser, rising filter", "duration_s": 4}, "candidates": 2, "wait_s": 60, "fetch": true}`. Import on `FX` with `file.import_audio`, then `edit.move_clips {"clips": [31], "to": "<marker minus clip length>"}` so it ends exactly on the downbeat of bar 13 (clip length in samples comes from `track.inspect`; marker position minus length gives the start in samples), and `edit.fades_create {"clip": 31, "side": "out", "length": 480, "shape": "equal power"}` for a clean end.
- Filter sweep: put a low-pass on the track (`mix.insert {"track": "SYN", "slot": 1, "plugin": "eq_1band", "params": {"type": 4, "freq": 12000}}`), then two breakpoints: `automation.set_point {"track": "SYN", "param": "plugin:b:freq", "at": "bars_beats:12|1|000", "value": 12000}` and `{"at": "bars_beats:13|1|000", "value": 400}`. Slot letters: slot 0 is `a`, slot 1 is `b`. Read the parameter range from `plugin.describe` and the Atlas profile first. Values are in the parameter's units (Hz here).
- Reverse: `audiosuite.process {"process": "reverse", "clips": [32]}`. Reverse a crash or vocal tail so that it ends on the downbeat, and position it with `edit.move_clips`.
- Drum cut on the last beat: `edit.clear {"tracks": ["DRM"], "start": "bars_beats:12|4|000", "end": "bars_beats:13|1|000"}` leaves silence on the last beat and keeps the rest.
- Silence before a drop: `edit.insert_silence {"tracks": ["DRM", "BAS", "SYN"], "start": "bars_beats:12|4|000", "end": "bars_beats:13|1|000"}` pushes later material back; use `edit.clear` when the length must not change.
- Put breakpoints on beat boundaries and confirm by bouncing, because interior breakpoints can depend on the audio block size.

## 4. Verify and report
Bounce the transition with a bar on each side (`file.bounce_mix` from bar 11 to bar 14) and read `true_peak_db` and `lufs`. Re-inspect the tracks. Report in bars: what was copied or added and where, parameters old to new, the level change versus the section before, the checkpoint path, the `edit.undo` count and the preview path.
