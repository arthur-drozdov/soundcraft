---
name: studio-sing-to-midi
description: Use when the user sings, hums or whistles a melody (a recording or voice memo) and wants it as editable MIDI on an instrument. Covers transcription, conservative cleanup, light quantising, keeping the original as a muted reference, and auditioning instruments.
---

# Sing to MIDI to instrument

Follow `studio-verified-edit` for all session changes. This works for monophonic melodies only (one note at a time); say so if the take is chordal. Checkpoint first: `session.checkpoint {"label": "before-transcribe"}`.

## 1. Get the take
- The user records in the desktop app, or points you at a voice memo inside the project folder. Import it: `file.import_audio {"path": "<project>/take1.wav", "at": "bars_beats:1|1|000", "track": "VOX", "new_tracks": false}`. Then rename the track `VOX Ref` (`track.rename`) and tag it.
- Measure it: `audio.analyze {"clip": 12}`. Report `peak_dbfs`, `clipped_samples`, `key` and `bpm` with their confidences. A take that is clipped, very quiet (peak under -30 dBFS) or noisy will transcribe badly: ask for a better take instead of fixing the notes by hand.
- If the take's tempo differs from the session, do not time-stretch it. Either set the session tempo to the measured `bpm` (`event.tempo {"bpm": 96}`) when the take was sung freely, or tell the user.

## 2. Transcribe
1. Create the target first: `track.new {"kind": "instrument", "name": "SYN Lead", "instrument": "subtractive_synth"}`.
2. Preferred: `midi.from_audio {"track": "VOX Ref", "start": "bars_beats:1|1|000", "end": "bars_beats:9|1|000", "to_track": "SYN Lead", "min_note_ms": 60}`. The source must be an audio track; `to_track` must be an existing MIDI or instrument track (leave it out and a new instrument track is created, reported as `created_track`). It writes one MIDI clip and returns `{track, clip, count, notes}` where every note has `index`, `pitch`, `name`, `start_ticks`, `length_ticks`, `velocity` and `confidence`; use those `index` values with `midi.note_delete` and `midi.note_edit`. One `edit.undo` removes everything it did. It fails with "found no notes" on silence, noise or chords.
3. If the result is poor (many notes under 0.5 confidence, octave jumps, fragmented notes), fall back to Forge:
   - `forge_upload {"path": "<project>/take1.wav"}` returns `asset_id`.
   - `forge_submit {"type": "transcribe.melody", "inputs": {"audio": "<asset_id>"}, "params": {"min_note_ms": 60, "bpm": 120}, "wait_s": 60, "fetch": true}`. Set `bpm` to the session tempo. The model is `basic-pitch` when available, else `yin`; say which one ran.
   - `file.import_midi {"path": "<midi path from the result>", "at": "bars_beats:1|1|000"}`.
4. Read the notes: `midi.notes {"clip": 12}` (fields `pitch`, `start`, `length` in ticks, `velocity`). Report the count and the mean confidence.

## 3. Clean up (conservatively)
Work on the notes, never on the vocal. Delete or merge only what is clearly wrong, and report every change.
- Drop notes shorter than 60 ms or with confidence under 0.3: `midi.note_delete {"clip": 12, "indices": [4, 9, 10]}`. Give all indices in one call; indices shift after a delete.
- Merge glides and repeated fragments: extend the first note with `midi.note_edit {"clip": 12, "index": 3, "length_ticks": 1440}`, then delete the extras.
- Snap to the key: pass `"key": "F minor"` to `midi.from_audio` when the key is known (ask the user, or use the measured `key` if its confidence is above 0.6; scale names like `Bb major`, `D dorian` and `minor pentatonic` work too). Notes that snap onto each other are merged. To rerun with a key, `edit.undo` the first transcription. Fix a few stray notes by hand with `midi.notes_move {"clip": 12, "indices": [7], "semitones": -1}`. Transpose a whole clip with `event.transpose {"clips": [12], "semitones": 12}`.
- Quantise lightly, 50 percent strength or less unless the user asks for tight timing: `event.quantize {"clips": [12], "grid": "1/16", "strength": 50}` (or `"quantize": 0.5` with `"grid": "1/16"` in `midi.from_audio`). Say what strength and grid you used.
- Remove double notes: `event.remove_duplicate_notes {"clips": [12]}`.

## 4. Keep the reference
Keep the original vocal on its own track and mute it: `mix.mute {"tracks": ["VOX Ref"], "value": true}`. Do not delete it. The MIDI clip and the audio stay aligned at the same bar.

## 5. Assign instruments and audition
- Built-in synth: `mix.instrument {"track": "SYN Lead", "plugin": "subtractive_synth", "params": {"osc1_wave": 1, "cutoff": 2400, "amp_release": 200}}`. List ids and ranges with `engine.plugins`.
- Sampler or keybed: see `studio-playable-instrument`.
- Third-party instrument: only if it has an Atlas profile (`studio-plugin-profiling`).
- Try two or three alternatives on duplicates of the track (`track.duplicate {"tracks": ["SYN Lead"]}`), then solo and mute so the user can compare. Bounce the same 4 bars for each: `file.bounce_mix {"path": "<project>/Previews/20261022-lead-a.wav", "start": "bars_beats:1|1|000", "end": "bars_beats:5|1|000"}`, and report `true_peak_db` and `lufs` for each.
- The MIDI stays editable on every version. Never render it to audio unless the user asks.

## 6. Report
Notes found, dropped and kept, mean confidence, which transcriber ran, the key used, grid and quantise strength, the state of the reference track, the checkpoint path, and the preview paths. Ask the user to listen for wrong notes and offer to fix them by note index.
