---
name: studio-session-conventions
description: Use whenever creating tracks, markers, buses or previews, or when starting from the project's Template.scraft. Defines role prefixes and colours, section markers, tagging of agent-created tracks, the master limiter, the -1 dBFS preview ceiling and where files go.
---

# Session conventions

## Start from the template
- `Template.scraft` in the project folder has five empty stereo or mono audio tracks (`DRM`, `BAS`, `SYN`, `FX`, `VOX`), section markers (Intro bar 1, Verse bar 5, Chorus bar 13, Drop bar 21, Outro bar 29) and a limiter on the `Master` track.
- Do not work in the template itself. Open it (`session.open {"path": "<project>/Template.scraft"}`), then `session.save_as {"path": "<project>/<Song name>.scraft"}` and work in that file. Saving it in the project folder also lets `session.checkpoint` write to `<project>/Checkpoints/`; checkpoints need a saved session.
- If the project has no template, build the same layout with the commands below.
- Set the tempo first: `event.tempo {"bpm": 130}`. Move the section markers if the song is shorter or longer (`markers.edit`).

## Track names and colours
Prefix every track with its role, then a short description:

| Role | Prefix | Example | Colour `[r,g,b]` |
|---|---|---|---|
| Drums | `DRM` | `DRM Kick` | [220, 80, 70] |
| Bass | `BAS` | `BAS Sub` | [80, 120, 220] |
| Synth, keys, pads | `SYN` | `SYN Keys` | [240, 190, 60] |
| Effects, risers, impacts | `FX` | `FX Risers` | [150, 90, 200] |
| Vocals and references | `VOX` | `VOX Ref` | [90, 190, 120] |

- Create: `track.new {"kind": "audio", "name": "DRM Kick", "format": "stereo"}` or `{"kind": "instrument", "name": "SYN Keys", "instrument": "subtractive_synth"}`. Kinds are `audio`, `instrument`, `midi`, `aux`, `master`, `vca`, `folder`. The result lists the new track ids (`{"tracks": [7]}`); use them, because imports and renames can change names.
- Names like `Audio`, `Master` or `Inst` get a number appended on creation. Rename afterwards with `track.rename {"track": 7, "name": "FX Risers"}`.
- Colour: `track.color {"tracks": ["FX Risers"], "color": [150, 90, 200]}`.
- Tag every track you create: `track.comments {"track": "FX Risers", "comments": "agent: created by Claude, source forge a_1234abcd"}`. Never rename, recolour, delete or restructure tracks the user made unless asked.
- Groups of tracks go on a bus: `mix.new_bus {"name": "Drum Bus"}`, then route each track with `mix.send {"track": "DRM Kick", "slot": 0, "bus": "Drum Bus", "level_db": 0}`.

## Markers
- One marker per section: `markers.add {"name": "Intro", "at": "bars_beats:1|1|000"}`; also Verse, Chorus, Drop, Bridge, Outro. Move or rename with `markers.edit {"number": 2, "name": "Verse 1", "at": "bars_beats:9|1|000"}`; remove with `markers.delete {"number": 2}`.
- Keep section lengths in whole bars (4, 8 or 16). Read the markers back from `session.inspect` before any section work.

## Loudness and safety
- The master limiter is the built-in `maximizer` in slot 0 of `Master` (threshold 0 dB, ceiling -1.2 dB: a margin under the -1 dBFS cap so inter-sample peaks stay under it too). Keep it. Check it with `plugin.describe {"track": "Master", "slot": 0}`.
- Previews never exceed -1 dBFS true peak: read `true_peak_db` from every `file.bounce_mix` result. Fix levels at the source tracks, not by pushing the limiter.
- Previews go in `<project>/Previews/`, named `<yyyymmdd>-<what>.wav`, 24-bit. Never overwrite an earlier preview; add a suffix.
- Generated audio stays in `Forge Inbox/`. Do not move, rename or edit those files: the sidecar `.json` next to each one records the model, seed and licence.
- Checkpoints go in `<project>/Checkpoints/` through `session.checkpoint`.

## Safety rules
- Checkpoint before batches and plugin work.
- No live parameter sweeps. Measure by bouncing.
- Only touch files inside the project folder.
- Save (`session.save {}`) before loading a third-party plugin; a plugin crash can close the session.
