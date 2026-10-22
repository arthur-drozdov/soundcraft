# Control protocol

SoundCraft can be driven completely by other programs: scripts, test harnesses and AI agents.
Every menu item, button, mouse gesture and keystroke maps to a **command** (or to synthetic
input), and every command is reachable from:

| Surface | How |
|---|---|
| Desktop app control channel | `soundcraft --control <port>` (`0` = pick a free port; printed as `SOUNDCRAFT_CONTROL_PORT=…`) |
| CLI against a running app | `soundcraft-cli app --port <port> <method or command id> [JSON]` |
| Headless CLI | `soundcraft-cli run --demo --cmd 'mix.volume={"track":"Kick","db":-6}' --bounce out.wav` |
| MCP (headless) | `soundcraft-cli mcp [--demo]` |
| MCP (bridged to the app) | `soundcraft-cli mcp --connect <port>` |

## Isolated automation profiles

Set `SOUNDCRAFT_CONFIG_DIR` to a fresh directory to isolate UI preferences, `Autosave`, `Presets`, and `logs`.
The nonempty value is used as the complete settings directory, even in portable mode (`portable.txt` beside the executable, which otherwise uses `SoundCraftData` next to it). Unset retains the normal platform paths;
an explicit empty value disables the settings directory instead of falling back to a personal profile. Save synthetic sessions to explicit scratch paths. Do not use `SOUNDCRAFT_NO_PREFS`
when testing preference persistence across restarts.

Run UI acceptance with `--no-audio`. This disables the output player; it does not disable recording
input opened by arm/monitor commands. An automated UI smoke run must never arm a track, enable input
monitoring, or start recording. Control input injects egui events; it does not establish operating-system
input delivery or screen-reader acceptance. Native screenshots use the presented viewport frame.

## Wire format

Newline-delimited JSON over TCP on `127.0.0.1`. One request per line, one reply per line:

```json
{"id": 1, "method": "engine.execute", "params": {"command": "track.new", "params": {"count": 2, "format": "Stereo"}}}
{"id": 1, "ok": true, "result": {"tracks": [12, 13]}}
```

Errors: `{"id": 1, "ok": false, "error": "bad parameters for `mix.volume`: no track named `Nope`"}`.

## Methods

| Method | Params | What it does |
|---|---|---|
| `engine.execute` (alias `command`) | `{command, params}` | Run a command programmatically. **Never opens a dialog**; empty params use defaults or the current selection. |
| `engine.commands` | `{filter?}` | Every command: id, label, menu path, shortcut, params doc, enabled + reason. Includes UI-layer commands (`window.*`). |
| `engine.parity` | `{}` | Parity report against the incumbent's menu catalog. |
| `session.inspect` | `{detail?: "summary"|"full"}` | The document: tracks, clips (with times), inserts (+params when full), sends, routing, automation, markers, groups, busses, sources (`loaded` true when decoded audio is in the pool), selection, zoom (`samples_per_px`, `scroll`), transport, undo label. |
| `ui.inspect` | `{}` | UI state (`window`, panels, dialogs), window size, playing/position, audio device, and `edit_layout` (timeline rect and every track row's rect, for clicking). |
| `ui.set` | any `UiState` fields | e.g. `{"window": "Mix", "narrow_mix": true}`. |
| `ui.menu.list` | `{}` | Every catalog menu path and the command that implements it (or null). |
| `ui.menu.invoke` | `{path}` or `{id}` | Like clicking the menu item: may open its dialog (e.g. `"Track > New..."`). |
| `ui.click` | `{x, y, button?, count?, cmd?, shift?, alt?, ctrl?}` | Synthetic click (window points). |
| `ui.drag` | `{x, y, to_x, to_y, steps?}` | Synthetic drag with the primary button. |
| `ui.scroll` | `{x, y, dx?, dy?, unit?: "point"\|"line", steps?, cmd?, shift?, alt?, ctrl?}` | A two-finger trackpad swipe over `(x, y)`: `dx`/`dy` points spread over `steps` frames (default 8). Positive values move the content right/down (natural scrolling). `unit: "line"` sends mouse-wheel notches instead. With `cmd` it zooms; with `shift` it scrolls sideways. |
| `ui.zoom` | `{x, y, factor, steps?}` | A trackpad pinch around `(x, y)`: `factor` > 1 spreads the fingers (zooms in). |
| `ui.move` | `{x, y}` | Move the pointer. |
| `ui.key` | `{key, cmd?, shift?, alt?, ctrl?}` | Key press (`Space`, `Enter`, `A`, `F7`, `=`…). |
| `ui.text` | `{text}` | Type text into the focused field. |
| `ui.screenshot` | `{path?}` | PNG of the window (base64 in `png_base64` when no path). |
| `app.quit` | `{}` | Quit. |

## Positions and references

- Tracks: id (number) or name (`"Kick"`).
- Clips: id (from `session.inspect`).
- Positions: samples (`96000`), seconds (`{"seconds": 2.5}`), or a time string in any timebase,
  optionally prefixed: `"bars_beats:5|1|000"`, `"min_secs:0:12.500"`, `"timecode:00:00:10:00"`,
  `"samples:48000"`.
- Ranges: `start` + `end` (or `start` + `length`); omitted = the current edit selection.

## Examples

```sh
P=$(soundcraft --demo --control 0 | sed -n 's/SOUNDCRAFT_CONTROL_PORT=//p')   # or read stderr
soundcraft-cli app --port $P mix.solo '{"track":"Bass"}'
soundcraft-cli app --port $P transport.play
soundcraft-cli app --port $P ui.screenshot '{"path":"/tmp/shot.png"}'
soundcraft-cli app --port $P ui.menu.invoke '{"path":"Track > New..."}'
```

Offscreen (no window): `cargo run -p soundcraft-ui-egui --example ui_shot -- out.png script.jsonl`,
where each script line is a request (`{"method": …, "params": …}`), `{"shot": "path.png"}` or
`{"steps": n}`. Use `--system-theme light|dark|none` to supply the initial OS appearance, and
script events such as `{"system_theme": "dark"}` to change it on subsequent frames.

Appearance defaults to Dark. `ui.theme {"mode":"system"}` follows the appearance supplied by
the native or web integration at launch and on later frames, while the saved choice stays System.
When the integration cannot report an appearance, System uses Dark. Manual Light and Dark stay
fixed when the OS appearance changes. An offscreen render without `--system-theme` supplies no
OS appearance and therefore resolves System to Dark.

## Panel docking

Tracks and Groups, Clips, MIDI Editor, Memory Locations, and Undo History use movable panels. Commands target the active Edit or Mix view; floating panels stay inside the current viewport. Edit and Mix retain separate arrangements; Clips and MIDI Editor belong to Edit. Transport, plug-in, and specialist utility windows retain their existing hosts. The document/session view is protected. Layouts, active tabs, panel visibility, and floating rectangles are saved with the app's existing UI preferences. These operations do not edit document contents.

Panel IDs: `tracks`, `clips`, `midi`, `memory`, `history`. The protected center is `canvas`.

`ui.dock` accepts `operation` and `panel`. Operations are `open`, `close`, `activate`, `float` (with finite positive `rect: [x, y, width, height]`), or `move` (with `target` and `zone: center|left|right|top|bottom`). Center groups with the target; side zones split beside it. `ui.dock.reset` restores the default arrangement. Invalid moves return an error without changing the layout.

Invoke these UI commands through `engine.execute`, for example `{"command":"ui.dock","params":{"operation":"float","panel":"tracks","rect":[100,100,320,440]}}`.

Advanced `ui.dock` operations use the same validated transaction as pointer gestures:
`moveFloating` updates `rect`, `resizeSplit` accepts a boolean child `path` and
`size` (`{"Ratio":0.5}`, `{"FixedFirst":280}`, or `{"FixedSecond":280}`),
`setStackOpen` takes `open`, and `resizeStack` takes a point `height` or null.
A center `move` can include `before` (panel ID or null to append) for tab order.
The equivalent `{"action": ...}` form accepts a serialized shared docking action;
this is also the form emitted by the renderer. Protected-canvas and app-specific panel
rules apply equally to both forms.

## Agent Studio commands

Commands added for agent-driven sessions (see `docs/agent-studio/SPEC.md` §5.3). Run them through `engine.execute` (or the MCP `execute` tool) like any command.

### `plugin.describe`

`{track, slot: 0..9 | "a".."j" | "instrument"}` or `{track, instrument: true}`: everything an agent needs to know about one plugin before touching it. A query: it does not change the session, is not journaled and leaves no undo step. `track` is an id or a name (required, no selection fallback); `slot` is the insert slot (a number or its letter), `"instrument"` or `instrument: true` the track's instrument.

Result:

```json
{
  "track": 3, "slot": 1, "instrument": false, "available": true,
  "plugin": {"id": "eq_7band", "format": "builtin", "name": "EQ 7-Band", "short_name": "EQ7",
             "vendor": "SoundCraft", "version": "0.10.0", "category": "Eq", "is_instrument": false},
  "bypass": false, "preset": "<factory default>",
  "params": [{"id": "mid_freq", "name": "MF Freq", "min": 100.0, "max": 8000.0, "default": 1000.0,
              "unit": "Hz", "taper": "Log", "choices": [], "value": 2400.0, "text": "2.40 kHz",
              "text_source": "soundcraft"}],
  "state_bytes": 0, "stored_state_bytes": 0, "state_refused": false, "editor": false, "instance": false
}
```

- `slot` is the insert index (`null` for the instrument, where `instrument` is `true`).
- `plugin.format` is `builtin`, `clap`, `vst3` or `au`, from the id prefix. `vendor` and `version` come from the host's plugin scan for hosted plugins (`null` when the scan has none) and are `"SoundCraft"` and the application version for built-ins. `category` is the `engine.plugins` category name.
- `params` lists every parameter the plugin exposes, in the plugin's order, with its range in SoundCraft units (`min`..`max`), `default`, `unit` (`None`, `Db`, `Hz`, `Ms`, `Percent`, `Ratio`, `Semitones`, `Cents`, `Seconds`, `Toggle`, `Choice`), `taper` (`Linear` or `Log`) and, for `Choice` parameters, the `choices` labels (the value is the index). `value` is the session's value (clamped into range), else the value the plugin itself holds (a hosted plugin's restored state), else the default. `text` is that value as the plugin shows it; `text_source` says whether it is the plugin's own (`"plugin"`) or SoundCraft's generic format (`"soundcraft"`, for built-ins and for hosted plugins without value-to-text).
- `state_bytes` is the size of the state a save would store: for a hosted plugin, what a temporary instance reports after the stored state is loaded; `stored_state_bytes` is the state currently held by the session (it is only refreshed from the live plugin when the application saves, so it can be stale or 0). Built-ins keep no state beyond their parameters (0). `state_refused` is true when the plugin rejected the stored state.
- `editor` is whether the plugin has its own native window (`true`/`false`; `null` when it could not be checked). Built-ins use SoundCraft's generic parameter editor and report `false`.
- `instance` is true when a temporary instance was created for this call. Only hosted plugins get one: it is private to the call, never the instance running on the audio thread, and it is dropped before the command returns. Creating it can take a moment for large plugins; it runs on the control (main) thread, as the plugin formats expect.
- A slot whose plugin is not installed on this machine is still described: `available: false`, `params: []`, `editor: null`, the id and format from the session.

Errors: no `track`/`slot`; a slot out of range, a bad letter, or both `slot` and `instrument: true`; an empty insert slot; `instrument` on a track without an instrument; an unknown track.

```json
{"command": "plugin.describe", "params": {"track": "Lead", "slot": "a"}}
{"command": "plugin.describe", "params": {"track": "Keys", "instrument": true}}
```

### `plugin.param_text`

`{track, slot | instrument: true, param, values: [number], normalized?: bool}`: the plugin's own display text for chosen values of one parameter, without changing the session (calibration: what does 0.37 of this knob mean?). A query, like `plugin.describe`. `slot`, `instrument` and `track` are as above; `param` is a parameter id from `plugin.describe`.

- `values` holds 1 to 64 finite numbers, in the parameter's units (`min`..`max`). With `normalized: true` they are control travel `0..1` (honouring the parameter's taper), which is what a calibration sweep wants: `[0, 0.1, … 1]`.
- Each value is clamped into the parameter's range first (choices and toggles round to a whole index); `clamped` says it was changed and `used` is the value that was formatted.
- The text is the plugin's own value-to-text where the plugin has one (VST3 `getParamStringByValue`, CLAP `value_to_text`; Audio Units use SoundCraft's format), else SoundCraft's format for the unit (`"2.40 kHz"`, `"-6.0 dB"`). `source` says which (`"plugin"`/`"soundcraft"`). Hosted plugins are asked through a private temporary instance with the session's stored state loaded, which is dropped before the command returns.

Result: `{track, slot, param, name, unit, normalized, texts: ["100.0 Hz", "2.40 kHz"], results: [{value, used, clamped, text, source}]}`; `texts` is `results[].text`.

Errors: everything `plugin.describe` rejects; a missing or unknown `param` (the message lists valid ids, up to 24); `values` not an array, empty, longer than 64, or holding anything but finite numbers; a plugin that is not installed.

```json
{"command": "plugin.param_text", "params": {"track": "Mix", "slot": 0, "param": "mid_freq", "values": [0.0, 0.37, 1.0], "normalized": true}}
```

### `audio.analyze`

`{clip}` or `{track, start, end}`, optional `sensitivity`: measures the audio of one clip, or of a track's clips over a range, so an agent can check what it made (level, loudness, tempo, key) without bouncing or leaving the session. A query: it does not change the session, is not journaled and leaves no undo step; nothing is stored.

- `clip` is a clip id (an audio clip on the active playlist). The analysed span is the clip's own range, with the clip's gain, fades, offset and clip effects applied; other clips of the track that overlap it are left out.
- `track` (id or name) with `start` and `end` (samples, `{"seconds": x}` or a time string; `length` instead of `end` works too) analyses what the track's clips play over that range, again before the mixer (no fader, pan, inserts or sends). The range is cut to the track's material plus one second, so `end` may be far past the last clip. Without `start`/`end` the edit selection is used, if there is one. `clip` and `track` together (or `clip` with `start`/`end`) are refused rather than guessed.
- `sensitivity` 0..1 (default 0.5; more finds more) is the onset detector's, the same one as Strip Silence / Tab to Transient.
- At most 10 minutes (and 2^27 samples over all channels) per call; longer is refused. The analysis runs on the calling thread and takes a moment for long spans (seconds for ten minutes), so keep ranges to what matters.

Result:

```json
{
  "track": 3, "clip": 12, "start": 0, "end": 144000,
  "duration_s": 3.0, "sample_rate": 48000, "channels": 1,
  "peak_dbfs": -6.02, "true_peak_dbfs": -6.02, "rms_dbfs": -9.03, "lufs": -9.7,
  "onsets_s": [0.1, 0.6, 1.1], "onset_count": 3,
  "bpm": 120.0, "bpm_confidence": 0.93,
  "key": "A minor", "key_tonic": 9, "key_mode": "minor", "key_confidence": 0.82,
  "centroid_hz": 440.0, "clipped_samples": 0, "dc_offset": 0.0
}
```

- `clip` is `null` for the range form; `start`/`end` are the samples actually analysed (after the cut above) and `duration_s` their length. `channels` is the track's channel count.
- Levels are dBFS, floored at -144 (silence): `peak_dbfs` is the sample peak over all channels, `true_peak_dbfs` the 4x-oversampled peak, `rms_dbfs` the RMS of all samples of all channels. `lufs` is integrated loudness (ITU-R BS.1770 with the -70 LUFS and -10 LU gates), from the same meter the bounce reports, -144 when nothing passes the gate (silence, or less than 400 ms of audio).
- `onsets_s` are onset times in seconds from the start of the analysed span (at most 5000 listed; `onset_count` is the real number).
- `bpm` (60..200 searched, reported folded by octaves into 80..160, so a 70 BPM ballad reads 140) and `key` (`"C major"`, `"F# minor"`: sharps; `key_tonic` 0 = C .. 11 = B, `key_mode` `"major"`/`"minor"`) are estimates with a 0..1 confidence; both are `null` (confidence 0) when they cannot be estimated: silence, fewer than 3 onsets or under 3 s of audio for the tempo, a flat chroma for the key. Treat a confidence below about 0.3 as "no idea" (noise, a single note, drums for the key). Tempo needs rhythmic material; key needs tonal material, and a bass-only or percussive clip says little.
- `centroid_hz` is the magnitude-weighted mean frequency (0 for silence), `clipped_samples` the number of samples (all channels) at or beyond +-0.999, and `dc_offset` the mean sample value of the channel with the largest absolute mean.

Errors: no `clip`/`track`; an unknown clip or track; a clip that is MIDI, video, muted or whose audio is not loaded in memory; a track that is not an audio track; an empty range or none given; a span longer than 10 minutes; both `clip` and `track`/`start`/`end`. A range with no material in it is not an error: it reads as silence (-144 dB, `bpm` and `key` null).

```json
{"command": "audio.analyze", "params": {"clip": 12}}
{"command": "audio.analyze", "params": {"track": "Loop", "start": 0, "end": {"seconds": 16}}}
```

### `session.inspect` / `track.inspect`: plugin format

Every insert in `inserts[]` also carries `format` (`builtin`, `clap`, `vst3` or `au`), and a track with an instrument carries `instrument_format` beside `instrument` (the plugin id), so an agent knows which slots to describe.
