# Curated knowledge for SoundCraft's built-in plugins

One YAML file per built-in plugin, named after its plugin id (`compressor.yaml`,
`subtractive_synth.yaml`, ...). All 27 plugins in the registry (`crates/dsp/src/plugins/mod.rs`)
are covered: effects, utilities and the three instruments (subtractive synth, drum synth,
sampler).

## What the files are

Each file is a **patch** in the format accepted by `atlas merge` (see `src/atlas/merge.py` and
DESIGN.md sections 6.1 and 6.2). It is not a complete profile: the host fields (`identity`,
parameter ids, host names, min/max/default, taper, choices) come from `engine.plugins` /
`plugin.describe`, and `atlas builtins --binary soundcraft-cli` merges these patches over them
to write `atlas/profiles/builtin/soundcraft/*.yaml`.

A patch contains:

- `default_provenance` - `{source: source, confidence: 0.95}` for anything without its own entry.
- `summary`, `signal_flow`, optional `modes` (`{name, description}`) and, for the Sampler,
  `ui_only`; with a matching entry under the top-level `provenance`.
- `params` - a list keyed by the host **param id**, each with `name`, `section`, `role`, `unit`,
  `does` (one line), `musical_use`, `interactions`, `safe_range` (`[lo, hi]`, always inside the
  host min/max), `automatable`, and a `provenance` map with one `{source, ref, confidence}` per
  field.

Plugins with no `modes` (most of them) simply omit the key. `invert` has no parameters, so its
patch has only `summary` and `signal_flow`.

## Provenance

Every value was read from the DSP code, not guessed. `ref` is `crates/dsp/src/plugins/<file>.rs:<line>`:

- for a parameter, the line that declares it in the plugin's parameter table (the Room and
  Plate reverbs share the `reverb_params!` macro, so both point at its lines);
- for `summary`, the plugin's `PluginInfo`;
- for `signal_flow` and `modes`, the `process` function (or the function that implements the
  mode, e.g. `design_eq1`, `pass_filter`, `fn trigger`).

Confidence is 0.95-0.97 for facts read straight from the code (`name`, `section`, `role`,
`unit`, `does`, `automatable`), 0.92 for `interactions`, and 0.9 for `musical_use` and
`safe_range`. Those last two are editorial guidance (typical settings, and a range that is safe
for an agent to move without surprising the listener) anchored to the code's real ranges and
behaviour; they are not literal code facts, which is why they sit at the bottom of the allowed
band. Nothing here is `inferred`.

Line numbers refer to the source at the time of writing; if the Rust files change a lot, re-check
the refs (the ids and the text stay valid, only the `:<line>` may drift).

## Conventions

- `unit` uses the host vocabulary: `dB`, `Hz`, `ms`, `percent`, `ratio`, `semitones`, `cents`,
  `seconds`, `bool`, `choice`, plus `none` for plain numbers (Q), `bits`, `samples`,
  `MIDI note`, `MIDI velocity`.
- `safe_range` for `bool` and `choice` params is `[0, 1]` or `[0, n-1]` (all options).
- `automatable: false` marks parameters that restructure the processing or are latched at
  note-on (filter-slope and EQ-type selectors, phaser stages, play/loop modes, key and
  velocity ranges, root note, dither bit depth). Set them; do not ramp them. Everything else is
  smoothed in the plugin and safe to automate. The drum synth's per-instrument parameters are
  read at note-on, so automation changes the next hit, not the ringing one (noted in each
  param's `interactions`).
- Instruments produce mono output copied to all channels (subtractive synth, drum synth); the
  sampler keeps stereo samples stereo.

## Checking

The patches were checked by merging each into a profile built from the Rust parameter tables
(same ids, ranges, defaults and units) with `atlas.merge.merge_patch` and validating the result
with `atlas.validate.validate_profile`: all 27 merge cleanly and validate, every parameter id
in the Rust tables is covered exactly once, and every unit matches the host unit. To re-check
after a plugin changes, run `atlas builtins` and then `atlas validate` on the output.
