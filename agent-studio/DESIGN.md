# Agent Studio: implementation design

This is the implementation contract for the parts of [the spec](../docs/agent-studio/SPEC.md)
that live outside the Rust workspace: Forge, Forge MCP, the Plugin Atlas, the Studio client and
the playbooks. The Rust additions to SoundCraft (S1–S8, sidechain) are documented with the
commands themselves in `docs/control-protocol.md`; their wire shapes are repeated in §9 here
because the Python side depends on them.

## 1. Layout

One Python distribution, `agent-studio`, in this folder. It installs on the DGX Spark with the
`forge` extra and on the laptop with the `mcp` extra; both can be installed together.

| Path | Package | Runs on | Purpose |
|---|---|---|---|
| `src/audiokit/` | `audiokit` | both | Pure numpy/scipy audio utilities: WAV I/O, resampling, loudness, trimming, fades, normalisation, analysis (onsets, BPM, key, centroid, loop seam), monophonic pitch tracking, MIDI file writing, SFZ / DecentSampler writers |
| `src/forge/` | `forge` | Spark | Gateway (FastAPI), job queue, asset store, worker manager, post-processing, job-type registry, Gary-compatible shim |
| `src/forge/workers/` | `forge.workers.*` | Spark | One worker module per model family, each runnable as `python -m forge.workers.<family>` in its own environment |
| `src/forge_mcp/` | `forge_mcp` | laptop | MCP server (stdio) that talks to Forge over HTTP and downloads into the inbox |
| `src/studio_client/` | `studio_client` | laptop | Client for SoundCraft: control channel (TCP JSON lines) and headless `soundcraft-cli mcp` (stdio) |
| `src/atlas/` | `atlas` | laptop | Profile schema, loader/validator, profiler, built-in profile generator, MCP server |
| `atlas/profiles/` | data | laptop | Committed plugin profiles (YAML) |
| `playbooks/` | data | laptop | Skills, CLAUDE.md template, MCP config template, project bootstrap script |
| `deploy/` | data | Spark | Dockerfiles per worker family, compose file, systemd units, Forge config example |
| `tests/` | — | CI/dev | pytest suites; audio is synthesised in tests, never committed (SoundCraft's `cargo xtask assets` rejects unattributed audio files) |

Python ≥ 3.10. Core dependencies: numpy, scipy, pyyaml, jsonschema, httpx, pydantic ≥ 2.
Extra `forge`: fastapi, uvicorn, python-multipart. Extra `mcp`: mcp ≥ 1.2. Extra `dev`: pytest,
pytest-asyncio. Model packages (torch, stable-audio-tools, ACE-Step, basic-pitch) are **never**
dependencies of the distribution: each worker family's own environment installs them
(`deploy/`), and a worker whose packages are missing reports its models as unavailable instead
of failing.

Coding rules: type hints everywhere; no bare `except:`; every input from the network or a file
is hostile (size caps, duration caps, path confinement, finite numbers); no global mutable
state outside the app object; logging via `logging`, never `print` except in CLIs.

## 2. audiokit

All functions take and return `numpy.ndarray` of `float32` shaped `(channels, frames)` plus an
integer sample rate, unless stated. Pure functions, no I/O except `wav`.

| Module | Functions |
|---|---|
| `wav` | `read(path) -> (audio, sr)` for PCM 8/16/24/32-bit int and 32/64-bit float, mono or multichannel, `WAVE_FORMAT_EXTENSIBLE` included; rejects files over a frame cap. `write(path, audio, sr, bit_depth: "16"\|"24"\|"32f")`, atomic (temp file + rename). |
| `dsp` | `resample(audio, sr_in, sr_out)` (polyphase), `to_channels(audio, n)`, `peak_dbfs`, `rms_dbfs`, `true_peak_dbfs` (4× oversampled), `lufs_integrated` (ITU-R BS.1770-4: K-weighting, 400 ms blocks, 75 % overlap, −70 LUFS absolute and −10 LU relative gates), `trim_silence(audio, sr, threshold_db=-60, pad_ms=5)`, `fade(audio, sr, in_ms, out_ms)` (equal-power), `normalize_peak(audio, target_dbfs)`, `normalize_lufs(audio, sr, target_lufs, ceiling_dbfs=-1)`, `loop_crossfade(audio, sr, ms)`, `dc_offset`, `clipped_samples(threshold=0.999)` |
| `analysis` | `onsets(audio, sr) -> seconds[]` (spectral flux, adaptive threshold), `bpm(audio, sr, lo=60, hi=200) -> (bpm, confidence 0..1)` (autocorrelation of the onset envelope, octave-error correction toward 80–160), `key(audio, sr) -> (name like "F minor", confidence)` (chroma from STFT + Krumhansl–Schmuckler correlation over 24 keys), `centroid_hz`, `bandwidth_hz`, `envelope(audio, sr, ms=10)`, `decay_time_s(audio, sr, drop_db=-40)`, `loop_seam_score(audio, sr) -> 0..1` (spectral + level continuity across the end→start boundary), `summary(audio, sr) -> dict` (everything in the `analyze.audio` result, §5) |
| `pitch` | `track(audio, sr) -> [(time_s, f0_hz or None, confidence)]` (YIN, 2048 window, 256 hop, 40–1500 Hz), `notes(frames, min_note_ms=60, merge_glide_cents=80) -> [Note(pitch, start_s, end_s, velocity, confidence)]` |
| `midi` | `write_smf(path, notes, bpm=120, ppq=480)` (type 0, one track; notes in seconds converted to ticks at `bpm`), `read_smf(path)` (notes only; for tests) |
| `instruments` | `write_sfz(path, zones, sample_dir)` and `write_dspreset(path, zones, sample_dir)`, where a zone is `{sample, root_note, key_low, key_high, vel_low, vel_high, tune_cents, gain_db, loop_mode, loop_start, loop_end}`; `plan_keybed(low, high, step) -> [(root, key_low, key_high)]` covering every key in range exactly once |

## 3. Forge

### 3.1 Configuration

`forge.yaml` (path from `--config` or `FORGE_CONFIG`), every field optional with defaults:

| Field | Default | Meaning |
|---|---|---|
| `listen.host` / `listen.port` | `127.0.0.1` / `7700` | Bind address. Set to the Tailscale/WireGuard address on the Spark; never `0.0.0.0` unless behind a private network. |
| `token_env` | `FORGE_TOKEN` | Environment variable holding the bearer token. Startup fails if it is empty, unless `insecure_no_auth: true` (tests only). |
| `data_dir` | `~/.local/share/forge` | Assets, uploads, scratch, SQLite index. |
| `retention_days` | `30` | Unrated, unused assets older than this are purged by `forge purge` and a daily task. |
| `uploads.max_bytes` / `uploads.max_seconds` | `200 MB` / `600` | Upload caps. |
| `memory_budget_gb` | `96` | Sum of loaded models' `vram_gb` may not exceed this; least-recently-used models are unloaded first. |
| `families.<name>.python` | the gateway's own interpreter | Interpreter of that family's environment. |
| `families.<name>.env` | `{}` | Extra environment for the worker. |
| `families.<name>.enabled` | `true` for `testtone`, `yin`, `analysis`; `false` otherwise | |
| `families.<name>.lane` | `gpu` (`cpu` for `yin`, `basicpitch`, `stems`, `render`) | Which queue lane runs it. |
| `families.<name>.options` | `{}` | Passed to the worker on `hello` (checkpoint paths, binary paths, device). |
| `gary_shim.enabled` / `gary_shim.ports` | `false` / `{foundation: 8015, sa3: 8006}` | §3.9. |

### 3.2 Worker protocol

Each family is a subprocess speaking JSON lines on stdin/stdout (stderr is forwarded to the log).
Requests carry an integer `id`; every reply echoes it.

| Request | Reply |
|---|---|
| `{"id", "op": "hello", "options": {...}}` | `{"id", "ok": true, "result": {"protocol": 1, "family", "models": [ModelInfo]}}` |
| `{"id", "op": "load", "model"}` | `{"id", "ok": true}` after the model is in memory |
| `{"id", "op": "unload", "model"}` | `{"id", "ok": true}` |
| `{"id", "op": "run", "job": RunJob}` | zero or more `{"id", "event": "progress", "progress": 0..1, "message"}`, then `{"id", "ok": true, "result": {"outputs": [Output]}}` or `{"id", "ok": false, "error"}` |
| `{"id", "op": "cancel", "job_id"}` | `{"id", "ok": true}`; the running job then ends with `{"ok": false, "error": "cancelled"}` |
| `{"id", "op": "shutdown"}` | process exits |

`ModelInfo`: `{id, family, job_types: [...], available: bool, reason?, licence, checkpoint, checkpoint_hash?, vram_gb, max_seconds?, params_schema?}`.
`RunJob`: `{job_id, type, model, params, inputs: {name: absolute path}, out_dir, seed}`.
`Output`: `{path, kind: "audio"|"midi"|"instrument"|"json", candidate: int, extra: {}}` where
`instrument` is a directory (§3.6). Workers only write inside `out_dir`.

`forge/workers/runtime.py` implements the loop once; a family module subclasses `Family` with
`models()`, `load(model)`, `unload(model)`, `run(job, progress, cancelled) -> list[Output]`.
A worker never crashes the gateway: protocol errors and worker death mark the running job
failed, and the manager restarts the worker on next use (with backoff).

### 3.3 Families and models

| Family | Models (ids) | Job types | Notes |
|---|---|---|---|
| `testtone` | `testtone` | every `generate.*`, `transform.audio`, `extend.audio`, `inpaint.audio`, `layer.add` | Deterministic synthesis from params and seed (pitched tones, noise bursts, clicks at tempo). Used by tests and the smoke check; not a "fake" model: it is labelled as a test generator in `/v1/models`. |
| `yin` | `yin` | `transcribe.melody` | CPU, audiokit.pitch; always available fallback. |
| `basicpitch` | `basic-pitch` | `transcribe.melody` | `basic_pitch.inference.predict`. |
| `foundation` | `foundation-1`, `foundation-1.2-samples`, `foundation-1.2-keybeds` | `generate.loop`, `generate.sample`, `generate.instrument`, `transform.audio` | RC Stable Audio Tools; keybed = one generation per root note with a shared seed, then sliced and mapped. Tag prompt builder mirrors Foundation's vocabulary (family, subfamily, descriptors, effects, behaviour tags, bars, BPM, key). |
| `sa3` | `sa3-small-music`, `sa3-small-sfx`, `sa3-medium` | `generate.sfx`, `generate.loop`, `generate.sample`, `transform.audio`, `extend.audio`, `inpaint.audio` | Stable Audio 3. |
| `acestep` | `ace-step-1.5-base`, `ace-step-1.5-xl-base` | `layer.add`, `stems.split` (extract), `generate.loop` | ACE-Step 1.5. |
| `stems` | `stems-cpp` | `stems.split` | Calls the stems.cpp CLI (`options.binary`, `options.model`). |
| `render` | `soundcraft` | `render.session` | Calls headless `soundcraft-cli` (`options.binary`): open session, run commands, bounce. |

Adapters for real models import their packages lazily inside `load`/`hello`; on `ImportError`
the model is reported `available: false` with the reason. Model licences come from a table in
the adapter, surfaced in `/v1/models` and copied into every sidecar.

### 3.4 Job types and parameters

Validated with pydantic models in `forge/jobs.py`. Unknown fields are errors.

Common: `prompt: str = ""` (≤ 2000 chars), `tags: dict[str, str|list[str]] = {}`, `seed: int|None`,
`candidates: 1..8 = 1`, `sample_rate: 22050..192000 = 48000`, `channels: 1|2 = 2`,
`post: {trim_silence: bool = false, fade_in_ms: 0..1000 = 0, fade_out_ms: 0..5000 = 0,
normalize: "none"|"peak"|"lufs" = "peak", target: float = -1.0, loop_crossfade_ms: 0..500 = 0,
bit_depth: "16"|"24"|"32f" = "24"}`.

| Type | Specific params | Inputs | Output kind |
|---|---|---|---|
| `generate.loop` | `bars: 1..16 = 4`, `bpm: 40..240`, `key: str?` ("F minor") | — | audio |
| `generate.sample` | `note: 0..127 = 60`, `length_s: 0.05..20 = 2` | — | audio |
| `generate.instrument` | `note_low = 36`, `note_high = 84`, `step: 1..12 = 3`, `velocity_layers: 1..3 = 1`, `length_s = 3`, `name: str` | — | instrument |
| `generate.sfx` | `duration_s: 0.1..47 = 4` | — | audio |
| `transform.audio` | `strength: 0..1 = 0.5` | `audio` | audio |
| `extend.audio` | `seconds: 1..60 = 8` | `audio` | audio |
| `inpaint.audio` | `start_s`, `end_s` | `audio` | audio |
| `layer.add` | `description: str` | `audio` | audio |
| `stems.split` | `stems: list[str] = []` (empty = model default) | `audio` | audio (one per stem; `extra.stem`) |
| `transcribe.melody` | `min_note_ms = 60`, `bpm = 120` | `audio` | midi (+ `extra.notes`) |
| `analyze.audio` | — | `audio` | none (result in job `result.analysis`) |
| `render.session` | `commands: list[{command, params}] ≤ 1000`, `start`, `end` | `session` (zip of a session folder) | audio |

Inputs are referenced by asset id (`"inputs": {"audio": "<asset_id>"}`); uploads become assets
too. `analyze.audio` runs in-process on the CPU lane (no worker).

### 3.5 Queue

Two lanes, `gpu` and `cpu`, each running one job at a time. Within a lane: priority (`high`,
`normal`, `low`), then FIFO. Job states: `queued → running → succeeded|failed|cancelled`.
Each job records `queue_position` (computed), `progress`, `eta_s` (from the model's recent
durations), `error`, `result: {asset_ids: [...], analysis?}`. Jobs and their states persist in
SQLite; on restart, `running` and `queued` jobs are marked `failed: "forge restarted"`.
`idempotency_key` returns the existing job for the same key within 24 h.

### 3.6 Post-processing, assets, sidecars

Every audio output: read → resample/channel-convert to the requested format → optional trim →
fades → optional loop crossfade → normalise → measure (audiokit `summary`) → write WAV at
`post.bit_depth` → store. Instruments: each zone WAV is processed the same way (no trimming of
the attack, fades only at the end), then the bundle is `samples/*.wav`, `<name>.sfz`,
`<name>.dspreset`, `manifest.json` (zones), zipped. MIDI: stored as-is with the note list.

Asset id: `a_` + first 24 hex chars of the SHA-256 of the stored file. Layout
`data_dir/assets/<yyyy>/<mm>/<asset_id>.<ext>` plus `<asset_id>.json` (the sidecar), indexed in
SQLite for search.

Sidecar: `asset_id`, `job_id`, `type`, `kind`, `model`, `family`, `checkpoint`, `checkpoint_hash`,
`licence`, `prompt`, `tags`, `seed`, `params`, `requested: {bpm, key, note}`, `measured: {bpm,
bpm_confidence, key, key_confidence, onsets, peak_dbfs, true_peak_dbfs, lufs, rms_dbfs,
centroid_hz, duration_s, clipped_samples, dc_offset, loop_seam}`, `sample_rate`, `channels`,
`bit_depth`, `parents: [asset_id]`, `candidate`, `created_at` (UTC ISO 8601), `rating`, `note`,
`files` (for instruments: zone list). Requested and measured values are both kept; a mismatch
is flagged in `warnings: [...]` (e.g. `"bpm requested 130, measured 65 (0.71)"`).

### 3.7 HTTP API

All routes under `/v1`, JSON, `Authorization: Bearer <token>` (constant-time compare), except
`GET /v1/health` which needs no token but reveals nothing beyond liveness and GPU memory.

| Route | Behaviour |
|---|---|
| `GET /v1/health` | `{status, version, uptime_s, lanes: {gpu: {running, queued}, cpu: {...}}, memory: {budget_gb, used_gb}, gpu: {available, name?, memory_used_gb?, memory_total_gb?}}` |
| `GET /v1/models` | `[ModelInfo + {loaded, warm, last_used, recent_seconds}]` |
| `POST /v1/models/{id}/load`, `/unload` | 409 if the budget can't fit it after evicting idle models |
| `POST /v1/uploads` | multipart `file`; validates it decodes as WAV (or, for `session`, a zip with one `.scraft`); returns `{asset_id, kind}` |
| `POST /v1/jobs` | `{type, model?, params, inputs?, priority?, idempotency_key?}` → `202 {job_id, state}`; 422 on validation errors with field paths |
| `GET /v1/jobs/{id}` | job record |
| `GET /v1/jobs?state=&limit=` | recent jobs |
| `DELETE /v1/jobs/{id}` | cancel (queued: removed; running: worker cancel) |
| `GET /v1/events` | Server-sent events: `job` events (`{job_id, state, progress, message}`) |
| `GET /v1/assets/{id}` | the file (`audio/wav`, `audio/midi`, `application/zip`), streamed, with `ETag` |
| `GET /v1/assets/{id}/meta` | sidecar |
| `GET /v1/assets` | search: `tags`, `text` (prompt substring), `type`, `model`, `job_id`, `rated` (bool), `min_rating`, `since`, `limit ≤ 200` |
| `POST /v1/assets/{id}/rate` | `{rating: 0..5, note?}` |

Errors: `{"error": {"code", "message", "details"?}}` with matching HTTP status.

### 3.8 CLI

`forge serve [--config]`, `forge check` (spawns every enabled family, prints models and
availability, runs one tiny job per available model), `forge purge`, `forge models`.

### 3.9 Gary-compatible shim

Optional FastAPI apps on Gary4juce's local ports (Foundation 8015, SA3 8006), mapping Gary's
`POST /generate` (Foundation fields: `seed`, `bars`, `host_bpm`, `key_root`, `key_mode`,
`family`, `subfamily`, `descriptor_knob_a..c`, `descriptors_extra`, effect toggles,
`behavior_tags`, `custom_prompt_override`, `guidance_scale`, `steps`; `/audio2audio` adds
`audio_data` base64 and `init_noise_level`) to Forge jobs, and `GET /poll_status/{session_id}`
to `{success, generation_in_progress, progress (0–100), queue_status: {position,
estimated_seconds}, status, audio_data (base64 WAV when done), error?}`. Written from the
observed wire contract only; no Gary4juce code (AGPL) is copied.

## 4. Forge MCP

stdio MCP server (`forge-mcp`). Environment: `FORGE_URL`, `FORGE_TOKEN`,
`AGENT_STUDIO_PROJECT` (project folder; required). Inbox: `<project>/Forge Inbox/<YYYY-MM-DD>/`.

| Tool | Input | Output |
|---|---|---|
| `forge_models` | — | models with availability, licence, loaded state |
| `forge_submit` | `type, params, model?, inputs? (asset ids), candidates?, priority?, wait_s? (0–600), fetch? (bool)` | `job_id`, state; if finished within `wait_s`: asset ids, sidecar summaries, and with `fetch` the local paths |
| `forge_job` | `job_id` | state, progress, ETA, assets |
| `forge_cancel` | `job_id` | state |
| `forge_fetch` | `asset_id` | local absolute path(s) (instrument zips are extracted to a folder; the `.sfz` path is returned), sidecar summary |
| `forge_upload` | `path` (must resolve inside the project folder) | `asset_id` |
| `forge_search` | `tags?, text?, type?, rated?, limit?` | assets |
| `forge_rate` | `asset_id, rating, note?` | ok |
| `forge_health` | — | Forge health, or a clear "Forge is offline" result (not an exception) |

Downloads: streamed to a temp file in the inbox, size-checked, hash-checked against the asset id
prefix, renamed; resumable with `Range`. HTTP retries with exponential backoff on connection
errors and 5xx (not on 4xx). Sidecars are written next to the files.

## 5. Studio client

`studio_client.Studio` with two transports behind one interface:
`ControlChannel(host="127.0.0.1", port)` (the desktop app's JSON-lines protocol) and
`HeadlessMcp(binary="soundcraft-cli", args=[...])` (spawns `soundcraft-cli mcp`, uses the
`execute`, `batch`, `inspect_session`, `bounce_mix` tools). Methods: `execute(command, params)`,
`batch(calls, keep_going=False)`, `inspect(detail)`, `commands(filter)`, `screenshot(path)`
(control channel only), `close()`. Errors raise `StudioError(message)`.

## 6. Plugin Atlas

### 6.1 Profile format

YAML, validated by `atlas/schema/profile.schema.json`. Path:
`atlas/profiles/<format>/<vendor-slug>/<name-slug>@<major>.yaml`. Fields as in spec §8.1:
`identity {name, vendor, format, plugin_id, version, os[], profiled}`, `summary`, `signal_flow`,
`modes[]`, `params[] {id, host_name, name, section, role, unit, min, max, default, taper,
choices[], calibration [{value, text}], does, musical_use, interactions[], safe_range [lo, hi],
automatable}`, `ui_only[] {feature, procedure | "manual only"}`, `presets[]`, `known_issues[]`.
Every leaf that carries knowledge (`does`, `role`, `musical_use`, `interactions`, `safe_range`,
`calibration`, `unit`) has a sibling `provenance: {source: host|manual|source|measured|inferred,
ref?, confidence: 0..1}` (`source` = read from the plugin's own source code, which only applies
to SoundCraft's built-ins) (one `provenance` object per param entry holding a map field →
provenance). Validation rejects an `inferred` field with confidence ≥ 0.9.

### 6.2 Profiler

`atlas profile --plugin <id> [--studio port | --headless binary] [--out path]`:
1. Scratch session: new instrument or audio track; `mix.instrument` / `mix.insert`.
2. `plugin.describe` → identity and raw parameter table (provenance `host`).
3. Calibration: `plugin.param_text` at 11 points (21 for continuous params with units) →
   `calibration`, inferred unit and taper (log if text is in Hz/ms and ratios are constant).
4. Measurement (offline only): for effects, import synthesised test signals (pink noise,
   log sine sweep, a click/drum pattern) written to a temp folder; for instruments, a MIDI clip
   with a sustained C major chord and a staccato note. For each parameter at min/mid/max (other
   parameters at default): set, bounce the range, analyse with audiokit (level, centroid,
   bandwidth, decay) → a `measured` effect summary, and `does: "no measurable effect in default
   mode"` with low confidence when nothing changes beyond tolerance.
5. Draft written with provenance; `atlas merge <profile> <patch.yaml>` applies agent- or
   human-written knowledge (e.g. from the manual) with provenance `manual`.
6. `atlas check --plugin <id> --version <v>` reports when a profile's version differs.

`atlas builtins --binary soundcraft-cli` regenerates `atlas/profiles/builtin/soundcraft/*.yaml`
from `engine.plugins` (host fields) and merges the curated knowledge in
`atlas/curated/builtin/*.yaml`.

### 6.3 Atlas MCP

stdio server `atlas-mcp`: `atlas_list {query?}`, `atlas_get {plugin_id, version?}` (the profile
plus a version-match flag), `atlas_validate {path}`, `atlas_profile {plugin_id, port?}` (runs
the profiler against the running Studio on a scratch session it creates and closes; returns the
draft path), `atlas_merge {profile_path, patch}`.

## 7. Playbooks

`playbooks/skills/<name>/SKILL.md` (YAML front matter `name`, `description`):
`studio-verified-edit` (§9.1 of the spec), `studio-sample-composition` (§9.2),
`studio-playable-instrument` (§9.3), `studio-sing-to-midi` (§9.4), `studio-section-ops` (§9.5),
`studio-plugin-profiling` (§8 + manual ingestion), `studio-session-conventions` (§9.6).
`playbooks/templates/CLAUDE.md` and `playbooks/templates/mcp.json`.
`playbooks/new_project.py <folder> [--studio-port N] [--forge-url URL]` creates a music project:
copies skills into `.claude/skills/`, writes `CLAUDE.md`, `.mcp.json` (soundcraft via
`soundcraft-cli mcp --connect`, forge-mcp, atlas-mcp), `Forge Inbox/`, and a template session
(created headlessly: role-named tracks, section markers, a limiter on the master).

## 8. Deploy (Spark)

`deploy/forge.example.yaml`, `deploy/compose.yaml` (gateway + one service per GPU family, each
from its own Dockerfile on an ARM64 CUDA PyTorch base), `deploy/docker/<family>.Dockerfile`,
`deploy/systemd/forge.service`, `deploy/README.md` (Tailscale binding, token, first run,
`forge check`, model downloads, porting notes).

## 9. Studio command contracts (Rust side, used by the Python side)

| Command | Params | Result |
|---|---|---|
| `plugin.describe` | `{track, slot?: 0..9\|"a".."j", instrument?: bool}` | `{track, slot, instrument, plugin: {id, format, name, vendor, version, category, is_instrument}, params: [{id, name, min, max, default, unit, taper, choices, value, text}], state_bytes, editor}` |
| `plugin.param_text` | `{track, slot?\|instrument?, param, values: [number] (≤ 64)}` | `{param, texts: [string\|null]}` |
| `midi.from_audio` | `{track, start?, end?, to_track?, min_note_ms?: 60, quantize?: 0..1, key?: "F minor"}` | `{track, clip, notes: [{pitch, start_ticks, length_ticks, velocity, confidence}]}` |
| `mix.sampler_zones_set` | `{track, zones: [{path, root_note, key_low, key_high, vel_low?, vel_high?, tune_cents?, gain_db?, loop_mode?, loop_start?, loop_end?}], params?}` | `{track, zones}` |
| `file.import_sfz` / `file.import_dspreset` | `{path, track?}` | `{track, zones, skipped: [reason]}` |
| `audio.analyze` | `{clip}` or `{track, start, end}` | `{duration_s, sample_rate, channels, peak_dbfs, rms_dbfs, lufs, onsets_s, bpm, bpm_confidence, key, key_confidence, centroid_hz, clipped_samples, dc_offset}` |
| `view.reveal` | `{track?, start, end}` | `{samples_per_px, scroll}` |
| `session.checkpoint` / `session.checkpoints` | `{label?}` / `{}` | `{path}` / `[{path, label, created}]` |
| `mix.insert_sidechain` | `{track, slot, source: track\|bus\|null}` | `{track, slot, source}` |

The Rust implementation is authoritative; `docs/control-protocol.md` documents the final
shapes.
