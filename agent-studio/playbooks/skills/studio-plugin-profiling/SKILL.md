---
name: studio-plugin-profiling
description: Use before touching any plugin parameter (built-in, VST3, CLAP or AU), when explaining or tweaking a loaded plugin or instrument patch, and when asked to profile a new plugin or ingest its manual. Loads the Atlas profile, changes values with display units and verification, or builds and merges a new profile.
---

# Plugin profiles (Atlas)

The Plugin Atlas holds verified knowledge about plugins: what each parameter does, its unit and taper, calibration points, a safe range, and where each fact came from. Follow `studio-verified-edit` for every change.

## Rule: load the profile before touching a plugin
1. Read the live plugin: `plugin.describe {"track": "SYN", "slot": 0}` for an insert (slot `0..9` or `"a".."j"`), `{"track": "SYN Keys", "instrument": true}` for an instrument. Note the `plugin` block (`id`, `format`, `version`) and, for every parameter, `id`, `min`, `max`, `unit`, `taper`, `choices`, `value`, `text` and `text_source`.
2. Find the profile: `atlas_list {"query": "eq"}`, then `atlas_get {"plugin_id": "eq_7band", "version": "0.10.0"}`. Built-in plugins use their plain id (`eq_7band`, `maximizer`, `subtractive_synth`); hosted plugins use `vst3:<hex>`, `clap:<id>` or `au:<id>`. Pass the installed plugin `version` from step 1 as `version`; the result flags whether the profile matches it. If the version differs, say so and treat the profile as possibly stale.
3. Compare the profile with the live table: same parameter ids, same ranges, `value` and `text` inside the profile's `safe_range`. Report any mismatch before changing anything.
4. No profile: only read values, or use a built-in instead, or run the profiler (below). Do not write parameters of an unprofiled third-party plugin.

## Understand `text` and `text_source`
- `text` is the value as the plugin shows it ("2.40 kHz").
- `text_source: "plugin"` means the plugin produced the text itself, so it carries plugin-specific meaning (for example a stepped label). `"soundcraft"` means SoundCraft's generic format for the unit (built-ins and plugins without value-to-text): correct, but it adds no plugin knowledge. Do not treat generic text as a calibration of the plugin's own scale.

## Explain a patch
1. `atlas_get` the profile, `plugin.describe` the live values.
2. Explain in plain terms using `does` and `musical_use`, quoting the current display values ("Cutoff 1.2 kHz: the filter is fairly closed, so the sound is dark"). Mark anything the profile gives as `inferred` or with low confidence as a guess.

## Change parameters
- Change only parameters that the profile lists. Stay inside each `safe_range`. Respect `interactions` (a parameter that only works when a mode is on) and `automatable: false` (set it, never ramp it).
- Convert a musical request to a value with the profile's `calibration` table. To check or extend it, ask the plugin itself, without changing the session: `plugin.param_text {"track": "SYN", "slot": 0, "param": "mid_freq", "values": [1000, 2400, 5000]}`. `values` are in the parameter's own units (here Hz). Add `"normalized": true` to pass control travel `0..1` instead (honouring the taper): `"values": [0, 0.25, 0.5, 0.75, 1], "normalized": true`. The result has `texts` (and `results` with `used`, `clamped` and `source` per value). Up to 64 values per call.
- Make the change with one `batch`: `mix.insert_params {"track": "SYN", "slot": 0, "params": {"mid_freq": 2400, "mid_gain": 3}}` for an insert (or `mix.insert_param {"track": "SYN", "slot": 0, "param": "mid_freq", "value": 2400}` one at a time), and `mix.instrument_param {"track": "SYN Keys", "param": "cutoff", "value": 2400}` for a built-in instrument (or `mix.instrument {"track": "SYN Keys", "plugin": "subtractive_synth", "params": {...}}`). Values are in display units and are clamped into range.
- Read back with `plugin.describe` and confirm both `value` and `text` for every parameter you set. If the plugin quantised or refused a value, say so.
- Never sweep parameters while the transport plays. Measure by bouncing: bounce the same range before and after (`file.bounce_mix`), upload both (`forge_upload`), run `forge_submit {"type": "analyze.audio", "inputs": {"audio": "<asset_id>"}, "wait_s": 30}` on each and compare `centroid_hz`, level and decay.
- Report old and new display values for each parameter: "Cutoff 1.2 kHz to 2.4 kHz: brighter attack" and "Amp Decay 200 ms to 90 ms: shorter notes". Include the measured change ("spectral centroid 1.1 kHz to 1.9 kHz").

## Profile a new plugin
1. Use a scratch session; never the user's working session. `atlas_profile {"plugin_id": "vst3:<hex>", "port": 7801}` creates a scratch session in the running Studio (and closes it afterwards), runs the profiler and returns the draft path. Save the user's session first (`session.save {}`) and tell them before you do this, or profile headlessly instead, which touches no open session: `atlas profile --plugin <id> --headless soundcraft-cli`. The profiler reads the parameter table with `plugin.describe`, calibrates every continuous parameter at 11 to 21 points with `plugin.param_text`, and measures effects offline by bouncing noise, a sine sweep and a drum loop (instruments: a held chord) at low, mid and high settings. Find the id first with `engine.vst3_plugins {}`, `engine.clap_plugins {}` or `engine.plugins {}`. The command-line equivalent is `atlas profile --plugin <id> --studio 7801`.
2. `atlas_validate {"path": "<draft path>"}` and fix every error it lists.
3. Review the draft with the user. Anything not measured or read from the host stays `inferred` with low confidence (below 0.9). A parameter that shows no measurable effect is marked so, not removed.

## Ingest the manual
1. The user supplies the manual (a PDF in the project folder). Read it section by section.
2. For each parameter or mode, write a patch with provenance `manual` and a `ref` such as "section 4.2, p. 31". Summarise; do not copy long passages.

```json
{"default_provenance": {"source": "manual", "ref": "section 4.2, p. 31", "confidence": 0.9},
 "params": [{"id": "cutoff", "role": "filter cutoff",
             "does": "Sets the corner frequency of the low-pass filter.",
             "musical_use": "Lower for dark pads, raise for bright plucks.",
             "interactions": ["only active when Filter Enable is on"],
             "safe_range": [200, 8000]}]}
```

3. `atlas_merge {"profile_path": "<draft path>", "patch": {...}}`, then `atlas_validate` again.
4. A control that is not a host parameter goes under `ui_only` with "manual only", unless you have a tested procedure.
5. If a measured value disagrees with the manual, keep the measured value and record the conflict under `known_issues`.

## Reprofile
When `plugin.describe` reports a version different from the profile (`atlas_get` flags it, `atlas check --plugin <id> --version <v>` reports it too), reprofile and diff the result against the old profile before relying on it.

## Report
Plugin and version, the profile used (version-match result), what you changed with old and new display values, what you measured, and any profile gaps or guesses.
