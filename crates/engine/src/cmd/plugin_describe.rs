//! Plugin introspection for agents: `plugin.describe` (identity, every parameter with its range,
//! unit and current value text, state size, editor) and `plugin.param_text` (the plugin's own
//! display text for chosen values). Both are queries: they read the session, never change it,
//! and leave no undo step.
//!
//! Built-in plugins need no instance (their text is [`ParamInfo::format`]). A hosted plugin
//! (CLAP, VST3, Audio Units) gets a private temporary instance for the call, created and dropped
//! on the calling (control/UI) thread, so its own value-to-text, state size and editor can be
//! read. The instance is never the live one on the audio thread.

use super::*;
use crate::cmd;
use crate::inspect::plugin_format;
use serde_json::json;
use soundcraft_dsp::{ParamInfo, Plugin, PluginInfo};
use soundcraft_model::{INSERT_SLOTS, Insert, TrackId};

/// Most values one `plugin.param_text` call evaluates.
const MAX_VALUES: usize = 64;
/// Most parameter ids an "unknown parameter" error lists.
const MAX_HINT: usize = 24;

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            query "plugin.describe",
            "Describe Plugin",
            [],
            None,
            "{track, slot: 0..9 | \"a\"..\"j\" | \"instrument\" | instrument?: true} → {track, slot, instrument, available, plugin: {id, format, name, short_name, vendor, version, category, is_instrument}, bypass, preset, params: [{id, name, min, max, default, unit, taper, choices, value, text, text_source}], state_bytes, stored_state_bytes, editor, instance}. Read-only; hosted plugins get a temporary private instance",
            has_tracks,
            describe
        ),
        cmd!(
            query "plugin.param_text",
            "Plugin Parameter Text",
            [],
            None,
            "{track, slot | instrument?: true, param, values: [number] (1..64), normalized?: bool} → {track, slot, param, name, unit, normalized, texts: [string], results: [{value, used, clamped, text, source}]}. The plugin's own display text per value (parameter units, or 0..1 travel with `normalized`); read-only",
            has_tracks,
            param_text
        ),
    ]
}

/// `None` = the instrument.
fn parse_slot(cmd: &str, v: &Value) -> Result<Option<usize>> {
    let range = || bad(cmd, format!("slot out of range (0..{} or a..{})", INSERT_SLOTS - 1, char::from(b'a' + (INSERT_SLOTS as u8) - 1)));
    if let Some(n) = v.as_u64() {
        return usize::try_from(n).ok().filter(|&n| n < INSERT_SLOTS).map(Some).ok_or_else(range);
    }
    let s = v.as_str().map(|s| s.trim().to_ascii_lowercase()).ok_or_else(|| bad(cmd, "`slot` must be a number, a letter or \"instrument\""))?;
    if s == "instrument" {
        return Ok(None);
    }
    match s.as_bytes() {
        [c @ b'a'..=b'z'] if usize::from(c - b'a') < INSERT_SLOTS => Ok(Some(usize::from(c - b'a'))),
        _ => s.parse::<usize>().ok().filter(|&n| n < INSERT_SLOTS).map(Some).ok_or_else(range),
    }
}

/// The slot's name in results: its letter.
fn slot_name(slot: usize) -> String {
    u8::try_from(slot).ok().filter(|&s| usize::from(s) < INSERT_SLOTS).map_or_else(|| slot.to_string(), |s| char::from(b'a' + s).to_string())
}

struct Target<'a> {
    track: TrackId,
    /// `None` = the instrument.
    slot: Option<usize>,
    ins: &'a Insert,
}

/// The insert (or instrument) the params name. `track` is required.
fn locate<'a>(e: &'a Engine, cmd: &str, p: &Value) -> Result<Target<'a>> {
    let track = track_param(e, cmd, p, "track")?.ok_or_else(|| bad(cmd, "`track` required"))?;
    let want_instrument = match p.get("instrument") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad(cmd, "`instrument` must be a boolean")),
    };
    let given = p.get("slot").filter(|v| !v.is_null());
    let slot = match (want_instrument, given) {
        (true, None) => None,
        (true, Some(v)) => match parse_slot(cmd, v)? {
            None => None,
            Some(_) => return Err(bad(cmd, "pass `slot` or `instrument: true`, not both")),
        },
        (false, Some(v)) => parse_slot(cmd, v)?,
        (false, None) => return Err(bad(cmd, "`slot` (0..9 or a..j) or `instrument: true` required")),
    };
    let tr = e.session().track(track).ok_or_else(|| bad(cmd, "no track"))?;
    let ins = match slot {
        Some(i) => tr.mixer.inserts.get(i).and_then(Option::as_ref).ok_or_else(|| bad(cmd, format!("insert slot {} is empty", slot_name(i))))?,
        None => tr.instrument.as_ref().ok_or_else(|| bad(cmd, "the track has no instrument"))?,
    };
    Ok(Target { track, slot, ins })
}

/// A temporary private instance of a hosted plugin, with the stored state restored. `None` for
/// built-ins (they need none) and when the plugin cannot be created. Never the live instance.
fn open(e: &Engine, ins: &Insert) -> Option<soundcraft_mix::NewInstance> {
    if !soundcraft_mix::is_third_party(&ins.plugin) {
        return None;
    }
    let sr = e.session().sample_rate.hz() as f32;
    soundcraft_mix::create_instance(&ins.plugin, ins.state_bytes().as_deref(), sr, 1024, 2)
}

/// A finite `f32` as a JSON number that prints as the `f32` does (`0.1`, not `0.10000000149`).
fn num(v: f32) -> Value {
    if !v.is_finite() {
        return Value::Null;
    }
    v.to_string().parse::<f64>().map_or(Value::Null, |x| json!(x))
}

/// The plugin's own text for `v` when it has one, else SoundCraft's; and which it was.
fn text_of(plugin: Option<&mut Box<dyn Plugin>>, pi: &ParamInfo, v: f32) -> (String, &'static str) {
    if let Some(t) = plugin.and_then(|pl| pl.param_text(pi.id, v)) {
        return (t, "plugin");
    }
    (pi.format(v), "soundcraft")
}

/// (vendor, version) from the format's scan list; built-ins are SoundCraft's own.
fn identity(id: &str) -> (Option<String>, Option<String>) {
    fn non_empty(s: String) -> Option<String> {
        let s = s.trim().to_string();
        (!s.is_empty()).then_some(s)
    }
    match plugin_format(id) {
        "builtin" => (Some("SoundCraft".into()), Some(env!("CARGO_PKG_VERSION").into())),
        "clap" => soundcraft_clap_host::scan().into_iter().find(|d| d.id == id).map_or((None, None), |d| (non_empty(d.vendor), non_empty(d.version))),
        "vst3" => soundcraft_vst3_host::scan().into_iter().find(|d| d.id == id).map_or((None, None), |d| (non_empty(d.vendor), non_empty(d.version))),
        _ => soundcraft_au_host::scan().into_iter().find(|d| d.id == id).map_or((None, None), |d| (non_empty(d.vendor), non_empty(d.version))),
    }
}

fn describe(e: &mut Engine, p: &Value) -> Result<Value> {
    const CMD: &str = "plugin.describe";
    let e: &Engine = e;
    let Target { track, slot, ins } = locate(e, CMD, p)?;
    let info: Option<&'static PluginInfo> = super::clap::plugin_info(&ins.plugin);
    let (vendor, version) = if info.is_some() { identity(&ins.plugin) } else { (None, None) };
    let mut inst = info.and_then(|_| open(e, ins));
    let hosted = soundcraft_mix::is_third_party(&ins.plugin);

    let mut params = Vec::new();
    for pi in info.map_or(&[][..], |i| i.params) {
        let value = match ins.params.get(pi.id) {
            Some(v) => pi.clamp(*v),
            None => inst.as_ref().and_then(|i| i.plugin.param(pi.id)).map_or(pi.default, |v| pi.clamp(v)),
        };
        let (text, source) = text_of(inst.as_mut().map(|i| &mut i.plugin), pi, value);
        params.push(json!({
            "id": pi.id, "name": pi.name, "min": num(pi.min), "max": num(pi.max), "default": num(pi.default),
            "unit": pi.unit, "taper": pi.taper, "choices": pi.choices,
            "value": num(value), "text": text, "text_source": source,
        }));
    }

    let stored_state_bytes = ins.state_bytes().map_or(0, |b| b.len());
    let live_state_bytes = inst.as_mut().and_then(|i| i.plugin.save_state()).map(|b| b.len());
    // Whether the plugin has its own native window (built-ins use SoundCraft's generic editor).
    let editor = match inst.as_mut() {
        Some(i) => json!(i.plugin.editor().is_some()),
        None if hosted || info.is_none() => Value::Null,
        None => json!(false),
    };
    let plugin = match info {
        Some(i) => json!({
            "id": i.id, "format": plugin_format(i.id), "name": i.name, "short_name": i.short_name, "vendor": vendor, "version": version,
            "category": i.category, "is_instrument": i.is_instrument,
        }),
        None => json!({
            "id": ins.plugin, "format": plugin_format(&ins.plugin), "name": ins.plugin, "short_name": null, "vendor": null, "version": null,
            "category": null, "is_instrument": slot.is_none(),
        }),
    };
    Ok(json!({
        "track": track,
        "slot": slot,
        "instrument": slot.is_none(),
        "available": info.is_some(),
        "plugin": plugin,
        "bypass": ins.bypass,
        "preset": ins.preset,
        "params": params,
        "state_bytes": live_state_bytes.unwrap_or(stored_state_bytes),
        "stored_state_bytes": stored_state_bytes,
        "state_refused": inst.as_ref().is_some_and(|i| i.refused_state),
        "editor": editor,
        "instance": inst.is_some(),
    }))
}

fn param_text(e: &mut Engine, p: &Value) -> Result<Value> {
    const CMD: &str = "plugin.param_text";
    let e: &Engine = e;
    let Target { track, slot, ins } = locate(e, CMD, p)?;
    let info = super::clap::plugin_info(&ins.plugin).ok_or_else(|| bad(CMD, format!("plugin `{}` is not available", ins.plugin)))?;
    let param = str_param(p, "param").ok_or_else(|| bad(CMD, "`param` required"))?;
    let pi = info.param(param).ok_or_else(|| {
        let ids: Vec<&str> = info.params.iter().take(MAX_HINT).map(|x| x.id).collect();
        let more = info.params.len().saturating_sub(MAX_HINT);
        let tail = if more > 0 { format!(", … (+{more} more; see plugin.describe)") } else { String::new() };
        bad(CMD, format!("`{}` has no parameter `{param}` (parameters: {}{tail})", info.id, ids.join(", ")))
    })?;
    let normalized = match p.get("normalized") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad(CMD, "`normalized` must be a boolean")),
    };
    let list = p.get("values").and_then(Value::as_array).ok_or_else(|| bad(CMD, "`values` must be an array of numbers"))?;
    if list.is_empty() || list.len() > MAX_VALUES {
        return Err(bad(CMD, format!("`values` needs 1..{MAX_VALUES} numbers, got {}", list.len())));
    }
    let mut values = Vec::with_capacity(list.len());
    for v in list {
        values.push(v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| bad(CMD, format!("`values` must be finite numbers, got {v}")))?);
    }

    let mut inst = open(e, ins);
    let mut results = Vec::with_capacity(values.len());
    let mut texts = Vec::with_capacity(values.len());
    for v in values {
        let vf = v as f32;
        let (used, clamped) = if normalized {
            (pi.from_normalized(vf), !(0.0..=1.0).contains(&v))
        } else {
            let used = pi.clamp(vf);
            (used, used != vf)
        };
        let (text, source) = text_of(inst.as_mut().map(|i| &mut i.plugin), pi, used);
        results.push(json!({"value": v, "used": num(used), "clamped": clamped, "text": text, "source": source}));
        texts.push(text);
    }
    Ok(json!({
        "track": track, "slot": slot, "param": pi.id, "name": pi.name, "unit": pi.unit, "normalized": normalized,
        "texts": texts, "results": results,
    }))
}

#[cfg(test)]
mod tests {
    use crate::Engine;
    use serde_json::{Value, json};

    fn rig() -> (Engine, u64) {
        let mut e = Engine::default();
        e.execute("track.new", &json!({"name": "T"})).unwrap();
        let t = e.session().tracks[0].id.0;
        (e, t)
    }

    fn param<'a>(r: &'a Value, id: &str) -> &'a Value {
        r["params"].as_array().unwrap().iter().find(|p| p["id"] == id).unwrap_or_else(|| panic!("no param {id}"))
    }

    #[test]
    fn describes_a_built_in_eq_with_every_parameter() {
        let (mut e, t) = rig();
        e.execute("mix.insert", &json!({"track": t, "plugin": "eq_7band", "slot": 1})).unwrap();
        e.execute("mix.insert_param", &json!({"track": t, "slot": 1, "param": "mid_freq", "value": 2400})).unwrap();
        let r = e.execute("plugin.describe", &json!({"track": t, "slot": "b"})).unwrap();
        let info = soundcraft_dsp::plugin_info("eq_7band").unwrap();
        assert_eq!(r["slot"], 1);
        assert_eq!(r["instrument"], false);
        assert_eq!(r["available"], true);
        assert_eq!(r["instance"], false, "a built-in needs no temporary instance");
        assert_eq!(r["plugin"]["id"], "eq_7band");
        assert_eq!(r["plugin"]["format"], "builtin");
        assert_eq!(r["plugin"]["name"], info.name);
        assert_eq!(r["plugin"]["vendor"], "SoundCraft");
        assert_eq!(r["plugin"]["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(r["plugin"]["category"], "Eq");
        assert_eq!(r["plugin"]["is_instrument"], false);
        assert_eq!(r["editor"], false);
        assert_eq!(r["state_bytes"], 0);
        let params = r["params"].as_array().unwrap();
        assert_eq!(params.len(), info.params.len());
        for (j, pi) in params.iter().zip(info.params) {
            assert_eq!(j["id"], pi.id);
            assert_eq!(j["name"], pi.name);
            assert_eq!(j["unit"], json!(pi.unit));
            assert_eq!(j["taper"], json!(pi.taper));
            assert_eq!(j["choices"], json!(pi.choices));
            assert_eq!(j["text_source"], "soundcraft");
        }
        let mid = param(&r, "mid_freq");
        assert_eq!(mid["unit"], "Hz");
        assert_eq!(mid["taper"], "Log");
        assert_eq!(mid["min"], 100.0);
        assert_eq!(mid["max"], 8000.0);
        assert_eq!(mid["default"], 1000.0);
        assert_eq!(mid["value"], 2400.0);
        assert_eq!(mid["text"], "2.40 kHz");
        let hf = param(&r, "high_shelf_q");
        assert_eq!(hf["default"], 0.707, "f32 values print as the f32 does");
        assert_eq!(hf["value"], 0.707, "an untouched parameter reads its default");
        let slope = param(&r, "lpf_slope");
        assert_eq!(slope["unit"], "Choice");
        assert_eq!(slope["text"], soundcraft_dsp::plugin_info("eq_7band").unwrap().param("lpf_slope").unwrap().format(1.0));
        assert_eq!(param(&r, "lpf_on")["text"], "Off");
    }

    #[test]
    fn describes_a_compressor_by_track_name_and_slot_number() {
        let (mut e, _) = rig();
        let t = e.session().tracks[0].id.0;
        e.execute("mix.insert", &json!({"track": t, "plugin": "compressor", "slot": 0, "params": {"threshold": -18}})).unwrap();
        let r = e.execute("plugin.describe", &json!({"track": "T", "slot": "0"})).unwrap();
        assert_eq!(r["plugin"]["category"], "Dynamics");
        let th = param(&r, "threshold");
        assert_eq!(th["value"], -18.0);
        assert_eq!(th["text"], "-18.0 dB");
        assert_eq!(th["unit"], "Db");
        assert_eq!(param(&r, "ratio")["text"], "4.0:1");
    }

    #[test]
    fn describes_the_sampler_as_an_instrument() {
        let (mut e, _) = rig();
        e.execute("track.new", &json!({"kind": "instrument", "name": "Keys"})).unwrap();
        let t = e.session().tracks.last().unwrap().id.0;
        e.execute("mix.instrument", &json!({"track": t, "plugin": "sampler", "params": {"level": -6}})).unwrap();
        for p in [
            json!({"track": t, "instrument": true}),
            json!({"track": t, "slot": "instrument"}),
            json!({"track": "Keys", "instrument": true, "slot": "instrument"}),
        ] {
            let r = e.execute("plugin.describe", &p).unwrap();
            assert_eq!(r["instrument"], true, "{p}");
            assert_eq!(r["slot"], Value::Null);
            assert_eq!(r["plugin"]["id"], "sampler");
            assert_eq!(r["plugin"]["is_instrument"], true);
            assert_eq!(r["plugin"]["category"], "Instrument");
            assert_eq!(r["plugin"]["format"], "builtin");
            assert_eq!(param(&r, "level")["text"], "-6.0 dB");
            assert_eq!(param(&r, "play_mode")["text"], "Gate");
            assert_eq!(param(&r, "attack")["unit"], "Ms");
        }
        // The default instrument of a fresh instrument track works too.
        e.execute("track.new", &json!({"kind": "instrument"})).unwrap();
        let t2 = e.session().tracks.last().unwrap().id.0;
        let r = e.execute("plugin.describe", &json!({"track": t2, "instrument": true})).unwrap();
        assert_eq!(r["plugin"]["id"], "subtractive_synth");
    }

    #[test]
    fn a_missing_plugin_is_described_not_an_error() {
        let (mut e, t) = rig();
        e.execute("mix.insert", &json!({"track": t, "plugin": "gain", "slot": 0})).unwrap();
        e.session_mut().track_mut(soundcraft_model::TrackId(t)).unwrap().mixer.inserts[0].as_mut().unwrap().plugin =
            "vst3:00000000000000000000000000000000".into();
        let r = e.execute("plugin.describe", &json!({"track": t, "slot": 0})).unwrap();
        assert_eq!(r["available"], false);
        assert_eq!(r["plugin"]["format"], "vst3");
        assert_eq!(r["params"], json!([]));
        assert_eq!(r["editor"], Value::Null);
        assert_eq!(r["instance"], false);
        let err = e.execute("plugin.param_text", &json!({"track": t, "slot": 0, "param": "0", "values": [0.5]})).unwrap_err();
        assert!(err.to_string().contains("not available"), "{err}");
    }

    #[test]
    fn param_text_formats_hz_and_db_values() {
        let (mut e, t) = rig();
        e.execute("mix.insert", &json!({"track": t, "plugin": "eq_7band", "slot": 0})).unwrap();
        let r = e.execute("plugin.param_text", &json!({"track": t, "slot": 0, "param": "mid_freq", "values": [100, 2400, 8000]})).unwrap();
        assert_eq!(r["param"], "mid_freq");
        assert_eq!(r["unit"], "Hz");
        assert_eq!(r["texts"], json!(["100.0 Hz", "2.40 kHz", "8.00 kHz"]));
        assert_eq!(r["results"][1]["value"], 2400.0);
        assert_eq!(r["results"][1]["used"], 2400.0);
        assert_eq!(r["results"][1]["clamped"], false);
        assert_eq!(r["results"][1]["source"], "soundcraft");
        let r = e.execute("plugin.param_text", &json!({"track": t, "slot": "a", "param": "mid_gain", "values": [-6, 3.5, 99, -99]})).unwrap();
        assert_eq!(r["texts"], json!(["-6.0 dB", "3.5 dB", "24.0 dB", "-24.0 dB"]));
        assert_eq!(r["results"][2]["clamped"], true, "values clamp into range");
        assert_eq!(r["results"][2]["used"], 24.0);
        assert_eq!(r["results"][3]["clamped"], true);
        // The spec's calibration example: 0.37 of the travel, not a frequency.
        let r = e
            .execute("plugin.param_text", &json!({"track": t, "slot": 0, "param": "mid_freq", "values": [0.0, 0.37, 1.0, 2.0], "normalized": true}))
            .unwrap();
        assert_eq!(r["normalized"], true);
        let info = soundcraft_dsp::plugin_info("eq_7band").unwrap().param("mid_freq").unwrap();
        assert_eq!(r["results"][1]["text"], info.format(info.from_normalized(0.37)));
        assert_eq!(r["results"][0]["text"], "100.0 Hz");
        assert_eq!(r["results"][2]["text"], "8.00 kHz");
        assert_eq!(r["results"][3]["clamped"], true);
        assert_eq!(r["results"][3]["text"], "8.00 kHz");
        // Huge and tiny finite numbers clamp.
        let r = e.execute("plugin.param_text", &json!({"track": t, "slot": 0, "param": "mid_freq", "values": [1e300, -1e300]})).unwrap();
        assert_eq!(r["texts"], json!(["8.00 kHz", "100.0 Hz"]));
    }

    #[test]
    fn param_text_on_an_instrument_and_choices() {
        let (mut e, _) = rig();
        e.execute("track.new", &json!({"kind": "instrument"})).unwrap();
        let t = e.session().tracks.last().unwrap().id.0;
        e.execute("mix.instrument", &json!({"track": t, "plugin": "sampler"})).unwrap();
        let r = e.execute("plugin.param_text", &json!({"track": t, "instrument": true, "param": "loop_mode", "values": [0, 1, 2, 1.4]})).unwrap();
        assert_eq!(r["texts"], json!(["Off", "Forward", "Ping-Pong", "Forward"]));
        assert_eq!(r["results"][3]["clamped"], true, "choices round to a whole index");
        assert_eq!(r["slot"], Value::Null);
    }

    #[test]
    fn hostile_params_are_errors() {
        let (mut e, t) = rig();
        e.execute("mix.insert", &json!({"track": t, "plugin": "eq_7band", "slot": 1})).unwrap();
        let many: Vec<f64> = vec![0.5; 65];
        let ok_vals = vec![1000.0; 64];
        assert!(
            e.execute("plugin.param_text", &json!({"track": t, "slot": 1, "param": "mid_freq", "values": ok_vals})).is_ok(),
            "64 values is the cap"
        );
        for p in [
            json!({"track": t, "slot": 0}),   // empty slot
            json!({"track": t, "slot": 10}),  // out of range
            json!({"track": t, "slot": "k"}), // bad letter
            json!({"track": t, "slot": "zz"}),
            json!({"track": t, "slot": -1}),
            json!({"track": t, "slot": 1.5}),
            json!({"track": t, "slot": [1]}),
            json!({"track": t}), // no slot
            json!({"track": t, "slot": null}),
            json!({"track": t, "instrument": true}), // not an instrument track
            json!({"track": t, "instrument": "yes"}),
            json!({"track": t, "instrument": true, "slot": 1}),
            json!({"track": 999_999, "slot": 1}),
            json!({"track": "nope", "slot": 1}),
            json!({"track": -3, "slot": 1}),
            json!({"slot": 1}),
        ] {
            assert!(e.execute("plugin.describe", &p).is_err(), "describe {p}");
            let mut q = p.clone();
            q["param"] = json!("mid_freq");
            q["values"] = json!([1]);
            assert!(e.execute("plugin.param_text", &q).is_err(), "param_text {q}");
        }
        let base = |extra: Value| {
            let mut v = json!({"track": t, "slot": 1, "param": "mid_freq", "values": [1000]});
            for (k, x) in extra.as_object().unwrap() {
                v[k] = x.clone();
            }
            v
        };
        for extra in [
            json!({"param": "nope"}),
            json!({"param": 5}),
            json!({"param": null}),
            json!({"values": []}),
            json!({"values": many}),
            json!({"values": [null]}),
            json!({"values": ["1"]}),
            json!({"values": [[1]]}),
            json!({"values": 5}),
            json!({"values": null}),
            json!({"values": [1, "NaN"]}),
            json!({"normalized": "yes"}),
        ] {
            assert!(e.execute("plugin.param_text", &base(extra.clone())).is_err(), "{extra}");
        }
        let msg = e.execute("plugin.param_text", &base(json!({"param": "nope"}))).unwrap_err().to_string();
        assert!(msg.contains("mid_freq"), "an unknown parameter lists the valid ids: {msg}");
    }

    #[test]
    fn queries_change_nothing_and_leave_no_undo_step() {
        let (mut e, t) = rig();
        e.execute("mix.insert", &json!({"track": t, "plugin": "eq_7band", "slot": 1})).unwrap();
        e.execute("mix.insert_param", &json!({"track": t, "slot": 1, "param": "mid_freq", "value": 2400})).unwrap();
        e.execute("edit.undo", &json!({})).unwrap();
        e.execute("edit.redo", &json!({})).unwrap();
        let before = e.execute("session.inspect", &json!({})).unwrap();
        let (undo, redo, dirty, label) = (e.can_undo(), e.can_redo(), e.is_dirty(), e.undo_label().map(str::to_string));
        let journal = e.journal.len();
        e.execute("plugin.describe", &json!({"track": t, "slot": 1})).unwrap();
        e.execute("plugin.param_text", &json!({"track": t, "slot": 1, "param": "mid_freq", "values": [200, 5000]})).unwrap();
        assert!(e.execute("plugin.describe", &json!({"track": t, "slot": 5})).is_err());
        assert_eq!((e.can_undo(), e.can_redo(), e.is_dirty(), e.undo_label().map(str::to_string)), (undo, redo, dirty, label));
        assert_eq!(e.journal.len(), journal, "queries are not journaled");
        assert_eq!(e.execute("session.inspect", &json!({})).unwrap(), before, "the session is unchanged");
        // One undo still undoes the parameter edit, not a query.
        e.execute("edit.undo", &json!({})).unwrap();
        let r = e.execute("plugin.describe", &json!({"track": t, "slot": 1})).unwrap();
        assert_eq!(param(&r, "mid_freq")["value"], 1000.0);
    }

    #[test]
    fn the_description_survives_a_save_and_reopen() {
        let (mut e, t) = rig();
        e.execute("mix.insert", &json!({"track": t, "plugin": "compressor", "slot": 3, "params": {"ratio": 8}})).unwrap();
        let dir = std::env::temp_dir().join(format!("soundcraft-plugin-describe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.scraft");
        e.execute("session.save", &json!({"path": path.to_string_lossy()})).unwrap();
        let mut f = Engine::default();
        f.execute("session.open", &json!({"path": path.to_string_lossy()})).unwrap();
        let a = e.execute("plugin.describe", &json!({"track": t, "slot": 3})).unwrap();
        let b = f.execute("plugin.describe", &json!({"track": "T", "slot": 3})).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(param(&b, "ratio")["value"], 8.0);
        assert_eq!(a["params"], b["params"]);
    }

    #[test]
    fn inspect_reports_each_plugins_format() {
        let (mut e, t) = rig();
        e.execute("mix.insert", &json!({"track": t, "plugin": "gain", "slot": 0})).unwrap();
        e.execute("track.new", &json!({"kind": "instrument"})).unwrap();
        let i = e.session().tracks.last().unwrap().id.0;
        let r = e.execute("track.inspect", &json!({"track": t})).unwrap();
        assert_eq!(r["inserts"][0]["format"], "builtin");
        let r = e.execute("track.inspect", &json!({"track": i})).unwrap();
        assert_eq!(r["instrument"], "subtractive_synth", "the instrument id stays a string");
        assert_eq!(r["instrument_format"], "builtin");
        for (id, f) in [("clap:a", "clap"), ("vst3:b", "vst3"), ("au:aumu:x:y", "au"), ("gain", "builtin"), ("", "builtin")] {
            assert_eq!(crate::inspect::plugin_format(id), f);
        }
    }

    #[test]
    fn commands_are_registered_as_queries() {
        for id in ["plugin.describe", "plugin.param_text"] {
            let c = crate::cmd::find_command(id).unwrap();
            assert!(!c.journal && !c.undoable, "{id}");
        }
    }
}
