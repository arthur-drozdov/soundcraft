//! `audio.analyze`: measure the audio of one clip or of a track range without touching the session
//! (Agent Studio S5). A query: no undo step, no journal entry, nothing stored. The measurements
//! are [`soundcraft_dsp::analysis`]: peak, true peak, RMS, integrated LUFS, onsets, tempo, key,
//! spectral centroid, clipped samples and DC offset.

use super::*;
use crate::cmd;
use serde_json::json;
use soundcraft_dsp::analysis;
use soundcraft_model::{ClipContent, TrackKind};

/// Most samples (frames times channels) one call renders and analyses: 2^27 f32 = 512 MiB.
const MAX_SAMPLES: usize = 1 << 27;

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        query "audio.analyze",
        "Analyze Audio",
        [],
        None,
        "{clip} | {track, start, end}; sensitivity?: 0..1 (onsets, default 0.5) → {track, clip, start, end, duration_s, sample_rate, channels, peak_dbfs, true_peak_dbfs, rms_dbfs, lufs, onsets_s: [seconds], onset_count, bpm|null, bpm_confidence, key|null, key_tonic|null, key_mode|null, key_confidence, centroid_hz, clipped_samples, dc_offset}. Up to 10 minutes; read-only",
        always,
        analyze
    )]
}

/// `v` rounded to `places` decimals as JSON; a non-finite value reads as 0 (JSON has no NaN).
fn round(v: f64, places: i32) -> Value {
    if !v.is_finite() {
        return json!(0.0);
    }
    let k = 10f64.powi(places);
    json!((v * k).round() / k)
}

fn analyze(e: &mut Engine, p: &Value) -> Result<Value> {
    let id = "audio.analyze";
    let has = |k: &str| p.get(k).is_some_and(|v| !v.is_null());
    if has("clip") && (has("track") || has("start") || has("end") || has("length")) {
        return Err(bad(id, "give either `clip` or `track` with `start`/`end`, not both"));
    }
    let sensitivity = f32_or(p, "sensitivity", 0.5).clamp(0.0, 1.0);
    let s = e.session();

    // Which track, which range, and (for a clip) which clip.
    let (t, r, clip) = if has("clip") {
        let cid = clip_id(p, id)?;
        let (t, c) = s.find_clip(cid).ok_or_else(|| bad(id, format!("no clip with id {}", cid.0)))?;
        match c.content {
            ClipContent::Audio { .. } => {}
            ClipContent::Midi { .. } => return Err(bad(id, format!("clip {} is a MIDI clip, not audio", cid.0))),
            _ => return Err(bad(id, format!("clip {} is not an audio clip", cid.0))),
        }
        if c.muted {
            return Err(bad(id, format!("clip {} is muted; unmute it to analyse it", cid.0)));
        }
        (t, c.range(), Some(cid))
    } else {
        let t = track_param(e, id, p, "track")?.ok_or_else(|| bad(id, "give `clip`, or `track` with `start` and `end`"))?;
        let r = range_param(e, id, p)?;
        if r.is_empty() {
            return Err(bad(id, "give `start` and `end` (or `length`) for a non-empty range"));
        }
        let tr = s.track(t).ok_or_else(|| bad(id, "no such track"))?;
        if tr.kind != TrackKind::Audio {
            return Err(bad(id, format!("track `{}` is not an audio track", tr.name)));
        }
        (t, crate::io::bounded_to_track(s, t, r), None)
    };
    let tr = s.track(t).ok_or_else(|| bad(id, "no such track"))?;
    if r.is_empty() {
        return Err(bad(id, "there is no audio in that range"));
    }
    let sr = s.sample_rate.as_f64() as f32;
    let channels = tr.channels().max(1);
    let frames = usize::try_from(r.len()).unwrap_or(usize::MAX);
    let limit = analysis::max_frames(sr);
    if frames > limit {
        return Err(bad(
            id,
            format!(
                "{:.1} s is too long to analyse (limit {} s); pass a shorter `start`/`end` range",
                frames as f64 / f64::from(sr),
                analysis::MAX_ANALYZE_SECS
            ),
        ));
    }
    if frames.saturating_mul(channels) > MAX_SAMPLES {
        return Err(bad(id, format!("{frames} frames of {channels} channels is too much audio to analyse; pass a shorter range")));
    }

    // A clip whose audio is not in memory (missing or offline media) would read as silence.
    let unloaded = tr
        .clips()
        .iter()
        .filter(|c| !c.muted && c.range().overlaps(&r) && clip.is_none_or(|cid| c.id == cid))
        .find(|c| c.source().is_some_and(|src| !s.pool.contains(src)));
    if let Some(c) = unloaded {
        return Err(bad(id, format!("the audio of clip {} (`{}`) is not loaded, so it cannot be analysed", c.id.0, c.name)));
    }

    // Render the track's clips (clip gain, fades, offsets and clip effects, no mixer). For a clip,
    // other clips of the track that overlap it are left out of the measurement.
    let overlapped = clip.is_some_and(|cid| tr.clips().iter().any(|c| c.id != cid && !c.muted && c.range().overlaps(&r)));
    let audio = if let (true, Some(cid)) = (overlapped, clip) {
        let mut solo = s.clone();
        if let Some(pl) = solo.track_mut(t).and_then(|t| t.playlist_mut()) {
            pl.clips.iter_mut().filter(|c| c.id != cid).for_each(|c| c.muted = true);
        }
        soundcraft_mix::render_clips(&solo, t, r)
    } else {
        soundcraft_mix::render_clips(s, t, r)
    };
    let a = analysis::analyze(&audio, sr, sensitivity);
    if a.frames == 0 {
        return Err(bad(id, "there is no audio in that range"));
    }

    let key = a.key;
    Ok(json!({
        "track": t.0,
        "clip": clip.map(|c| c.0),
        "start": r.start,
        "end": r.end,
        "duration_s": round(a.duration_s, 4),
        "sample_rate": s.sample_rate.as_f64().round() as u32,
        "channels": a.channels,
        "peak_dbfs": round(f64::from(a.peak_dbfs), 2),
        "true_peak_dbfs": round(f64::from(a.true_peak_dbfs), 2),
        "rms_dbfs": round(f64::from(a.rms_dbfs), 2),
        "lufs": round(f64::from(a.lufs), 2),
        "onsets_s": a.onsets_s.iter().map(|&t| round(t, 4)).collect::<Vec<_>>(),
        "onset_count": a.onset_count,
        "bpm": a.bpm.map(|b| round(f64::from(b), 2)),
        "bpm_confidence": round(f64::from(a.bpm_confidence), 3),
        "key": key.map(|k| k.name()),
        "key_tonic": key.map(|k| k.tonic),
        "key_mode": key.map(|k| k.mode()),
        "key_confidence": round(f64::from(a.key_confidence), 3),
        "centroid_hz": round(f64::from(a.centroid_hz), 2),
        "clipped_samples": a.clipped_samples,
        "dc_offset": round(f64::from(a.dc_offset), 6),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundcraft_audio_io::{AudioBuffer, FileFormat};
    use soundcraft_midi::Sequence;
    use soundcraft_model::{ChannelFormat, Clip};

    const SR: usize = 48_000;

    fn sine(freq: f64, amp: f64, secs: f64) -> Vec<f32> {
        (0..(secs * SR as f64) as usize).map(|i| (amp * (std::f64::consts::TAU * freq * i as f64 / SR as f64).sin()) as f32).collect()
    }

    /// 180 Hz decaying bursts on every beat.
    fn clicks(bpm: f64, secs: f64) -> Vec<f32> {
        let mut x = vec![0.0f32; (secs * SR as f64) as usize];
        let mut t = 0.1;
        while t < secs - 0.2 {
            let start = (t * SR as f64) as usize;
            for i in 0..SR / 8 {
                let s = i as f64 / SR as f64;
                if let Some(o) = x.get_mut(start + i) {
                    *o += ((-s / 0.025).exp() * 0.7 * (std::f64::consts::TAU * 180.0 * s).sin()) as f32;
                }
            }
            t += 60.0 / bpm;
        }
        x
    }

    /// An engine with one mono audio track holding `audio` at `at`; returns (engine, track, clip).
    fn rig(audio: Vec<Vec<f32>>, at: i64) -> (Engine, u64, u64) {
        let mut e = Engine::default();
        let frames = audio.first().map_or(0, Vec::len) as i64;
        let n = audio.len();
        let s = e.session_mut();
        let src = crate::io::add_source(s, "x", AudioBuffer { sample_rate: 48_000, channels: audio }, None, FileFormat::Wav);
        let t = s.add_track(TrackKind::Audio, if n == 2 { ChannelFormat::Stereo } else { ChannelFormat::Mono }, None);
        let cid = s.new_clip_id();
        crate::edit::place_clip(s, t, Clip::audio(cid, "x", src, 0, at, frames));
        (e, t.0, cid.0)
    }

    fn num(r: &Value, k: &str) -> f64 {
        r[k].as_f64().unwrap_or_else(|| panic!("{k} in {r}"))
    }

    #[test]
    fn analyses_a_clip_and_the_same_range_alike() {
        let (mut e, t, c) = rig(vec![sine(440.0, 0.5, 3.0)], 0);
        let a = e.execute("audio.analyze", &json!({"clip": c})).unwrap();
        let b = e.execute("audio.analyze", &json!({"track": t, "start": 0, "end": 144_000})).unwrap();
        assert_eq!(a["clip"], c);
        assert_eq!(b["clip"], Value::Null);
        assert_eq!(a["track"], t);
        assert_eq!((a["start"].as_i64(), a["end"].as_i64()), (Some(0), Some(144_000)));
        assert_eq!(a["sample_rate"], 48_000);
        assert_eq!(a["channels"], 1);
        assert!((num(&a, "duration_s") - 3.0).abs() < 1e-3, "{a}");
        assert!((num(&a, "peak_dbfs") + 6.02).abs() < 0.05, "{a}");
        assert!((num(&a, "rms_dbfs") + 9.03).abs() < 0.05, "{a}");
        assert!((num(&a, "centroid_hz") - 440.0).abs() < 10.0, "{a}");
        assert_eq!(a["clipped_samples"], 0);
        assert!(num(&a, "dc_offset").abs() < 1e-3);
        assert!(num(&a, "true_peak_dbfs") >= num(&a, "peak_dbfs") - 0.01);
        assert!(num(&a, "lufs") < -6.0 && num(&a, "lufs") > -12.0, "{a}");
        assert!(a["onsets_s"].is_array() && a["onset_count"].is_u64());
        for k in ["peak_dbfs", "rms_dbfs", "lufs", "centroid_hz", "duration_s", "dc_offset", "true_peak_dbfs"] {
            assert_eq!(a[k], b[k], "{k}: clip form and range form must agree");
        }
        assert_eq!(a["bpm"], b["bpm"]);
        assert_eq!(a["key"], b["key"]);
    }

    #[test]
    fn a_range_form_measures_only_the_range_and_follows_the_timeline() {
        // 1 s of silence, then 2 s of a loud tone starting at 1 s on the timeline.
        let (mut e, t, c) = rig(vec![sine(1000.0, 0.9, 2.0)], 48_000);
        let clip = e.execute("audio.analyze", &json!({"clip": c})).unwrap();
        assert_eq!((clip["start"].as_i64(), clip["end"].as_i64()), (Some(48_000), Some(144_000)));
        assert!((num(&clip, "duration_s") - 2.0).abs() < 1e-3);
        let with_gap = e.execute("audio.analyze", &json!({"track": t, "start": 0, "end": 144_000})).unwrap();
        assert!((num(&with_gap, "duration_s") - 3.0).abs() < 1e-3);
        assert!((num(&with_gap, "peak_dbfs") - num(&clip, "peak_dbfs")).abs() < 1e-6);
        assert!(num(&with_gap, "rms_dbfs") < num(&clip, "rms_dbfs") - 1.5, "leading silence lowers the RMS: {with_gap} vs {clip}");
        let gap_only = e.execute("audio.analyze", &json!({"track": t, "start": 0, "end": 40_000})).unwrap();
        assert_eq!(gap_only["peak_dbfs"], -144.0);
        assert_eq!(gap_only["lufs"], -144.0);
        assert_eq!(gap_only["bpm"], Value::Null);
        assert_eq!(gap_only["key"], Value::Null);
        assert_eq!(gap_only["key_tonic"], Value::Null);
        assert_eq!(gap_only["centroid_hz"], 0.0);
        // Positions accept seconds and length.
        let half = e.execute("audio.analyze", &json!({"track": t, "start": {"seconds": 1.0}, "length": {"seconds": 1.0}})).unwrap();
        assert!((num(&half, "duration_s") - 1.0).abs() < 1e-3, "{half}");
        // A range running far past the material is cut to it (plus a second).
        let long = e.execute("audio.analyze", &json!({"track": t, "start": 0, "end": 4_000_000_000i64})).unwrap();
        assert!((num(&long, "duration_s") - 4.0).abs() < 1e-3, "{long}");
    }

    #[test]
    fn estimates_tempo_and_key_through_the_command() {
        let (mut e, _, c) = rig(vec![clicks(120.0, 16.0)], 0);
        let r = e.execute("audio.analyze", &json!({"clip": c})).unwrap();
        assert!((num(&r, "bpm") - 120.0).abs() <= 1.0, "{r}");
        assert!(num(&r, "bpm_confidence") > 0.3, "{r}");
        assert!(r["onset_count"].as_u64().unwrap() >= 20, "{r}");
        assert!(r["onsets_s"][0].as_f64().is_some_and(|t| (t - 0.1).abs() < 0.03), "{r}");

        // An A minor chord: key and its parts.
        let chord: Vec<f32> = (0..SR * 3)
            .map(|i| {
                [220.0, 261.63, 329.63]
                    .iter()
                    .flat_map(|f| [(1.0, 0.3), (2.0, 0.15)].map(|(h, a)| a * (std::f64::consts::TAU * f * h * i as f64 / SR as f64).sin()))
                    .sum::<f64>() as f32
            })
            .collect();
        let (mut e, _, c) = rig(vec![chord], 0);
        let r = e.execute("audio.analyze", &json!({"clip": c})).unwrap();
        assert_eq!(r["key"], "A minor", "{r}");
        assert_eq!((r["key_tonic"].clone(), r["key_mode"].clone()), (json!(9), json!("minor")));
        assert!(num(&r, "key_confidence") > 0.3);
    }

    #[test]
    fn stereo_clipping_and_dc() {
        let mut l = sine(300.0, 0.4, 2.0);
        l.iter_mut().take(20).for_each(|v| *v = 1.0);
        let r: Vec<f32> = sine(300.0, 0.4, 2.0).iter().map(|v| v + 0.15).collect();
        let (mut e, _, c) = rig(vec![l, r], 0);
        let a = e.execute("audio.analyze", &json!({"clip": c})).unwrap();
        assert_eq!(a["channels"], 2);
        assert_eq!(a["clipped_samples"], 20);
        assert!((num(&a, "dc_offset") - 0.15).abs() < 1e-3, "{a}");
        assert!(num(&a, "peak_dbfs").abs() < 1e-6);
    }

    #[test]
    fn only_the_named_clip_is_measured_when_clips_overlap() {
        let (mut e, t, c) = rig(vec![sine(500.0, 0.25, 2.0)], 0);
        let quiet = e.execute("audio.analyze", &json!({"clip": c})).unwrap();
        {
            let s = e.session_mut();
            let src =
                crate::io::add_source(s, "loud", AudioBuffer { sample_rate: 48_000, channels: vec![sine(700.0, 0.8, 2.0)] }, None, FileFormat::Wav);
            let cid = s.new_clip_id();
            // Pushed directly: place_clip would trim the first clip away.
            if let Some(pl) = s.track_mut(soundcraft_model::TrackId(t)).and_then(|t| t.playlist_mut()) {
                pl.clips.push(Clip::audio(cid, "loud", src, 0, 24_000, 96_000));
            }
        }
        let again = e.execute("audio.analyze", &json!({"clip": c})).unwrap();
        assert_eq!(quiet["peak_dbfs"], again["peak_dbfs"], "{quiet} vs {again}");
        assert_eq!(quiet["lufs"], again["lufs"]);
        // The range form measures the track, so it sees both.
        let both = e.execute("audio.analyze", &json!({"track": t, "start": 0, "end": 96_000})).unwrap();
        assert!(num(&both, "peak_dbfs") > num(&quiet, "peak_dbfs") + 3.0, "{both}");
    }

    #[test]
    fn rejects_bad_requests_with_bad_params() {
        let (mut e, t, c) = rig(vec![sine(440.0, 0.5, 1.0)], 0);
        let midi = {
            let s = e.session_mut();
            let m = s.add_track(TrackKind::Midi, ChannelFormat::Mono, None);
            let cid = s.new_clip_id();
            crate::edit::place_clip(s, m, Clip::midi(cid, "m", 0, 48_000, Sequence::default()));
            (m.0, cid.0)
        };
        let muted = {
            let s = e.session_mut();
            let src =
                crate::io::add_source(s, "mu", AudioBuffer { sample_rate: 48_000, channels: vec![sine(440.0, 0.5, 1.0)] }, None, FileFormat::Wav);
            let a = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
            let cid = s.new_clip_id();
            crate::edit::place_clip(s, a, Clip::audio(cid, "mu", src, 0, 0, 48_000));
            if let Some(c) = s.find_clip_mut(cid) {
                c.muted = true;
            }
            (a.0, cid.0)
        };
        let cases = [
            (json!({}), "no params"),
            (json!({"clip": 99_999}), "unknown clip"),
            (json!({"clip": "x"}), "clip not a number"),
            (json!({"clip": midi.1}), "midi clip"),
            (json!({"clip": muted.1}), "muted clip"),
            (json!({"track": midi.0, "start": 0, "end": 48_000}), "midi track"),
            (json!({"track": 99_999, "start": 0, "end": 48_000}), "unknown track"),
            (json!({"track": "nope", "start": 0, "end": 48_000}), "unknown track name"),
            (json!({"track": t}), "no range"),
            (json!({"track": t, "start": 5, "end": 5}), "empty range"),
            (json!({"track": t, "start": 0}), "point range"),
            (json!({"track": t, "start": 10_000_000_000i64, "end": 10_000_048_000i64}), "range past the material"),
            (json!({"track": t, "start": -48_000, "end": -1}), "range before the start"),
            (json!({"clip": c, "track": t}), "clip and track"),
            (json!({"clip": c, "start": 0, "end": 100}), "clip and range"),
            (json!({"track": {"a": 1}, "start": 0, "end": 100}), "track not an id"),
            (json!({"track": t, "start": "garbage", "end": 100}), "unparseable position"),
        ];
        for (p, what) in cases {
            match e.execute("audio.analyze", &p) {
                Err(EngineError::BadParams(id, msg)) => assert_eq!(id, "audio.analyze", "{what}: {msg}"),
                other => panic!("{what} ({p}): expected BadParams, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_clip_before_zero_and_a_clip_without_loaded_audio() {
        let (mut e, t, c) = rig(vec![sine(440.0, 0.5, 2.0)], -24_000);
        let r = e.execute("audio.analyze", &json!({"clip": c})).unwrap();
        assert_eq!(r["start"], -24_000);
        assert!((num(&r, "duration_s") - 2.0).abs() < 1e-3, "{r}");
        assert!(num(&r, "peak_dbfs") > -7.0, "{r}");
        // The same clip through the range form is cut at zero.
        let r = e.execute("audio.analyze", &json!({"track": t, "start": -48_000, "end": 96_000})).unwrap();
        assert_eq!(r["start"], 0);
        // Drop the decoded audio: an error, not silence.
        let src = e.session().find_clip(soundcraft_model::ClipId(c)).and_then(|(_, c)| c.source()).unwrap();
        e.session_mut().pool.remove(src);
        for p in [json!({"clip": c}), json!({"track": t, "start": 0, "end": 48_000})] {
            match e.execute("audio.analyze", &p) {
                Err(EngineError::BadParams(_, msg)) => assert!(msg.contains("not loaded"), "{msg}"),
                other => panic!("{p}: {other:?}"),
            }
        }
    }

    #[test]
    fn refuses_more_than_ten_minutes() {
        // The clip is long (601 s of timeline) but its source is only a second: the length alone is refused,
        // before anything is rendered.
        let (mut e, t, _) = rig(vec![sine(440.0, 0.5, 1.0)], 0);
        let long = {
            let s = e.session_mut();
            let src =
                crate::io::add_source(s, "s", AudioBuffer { sample_rate: 48_000, channels: vec![sine(440.0, 0.5, 1.0)] }, None, FileFormat::Wav);
            let a = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
            let cid = s.new_clip_id();
            crate::edit::place_clip(s, a, Clip::audio(cid, "long", src, 0, 0, 601 * 48_000));
            (a.0, cid.0)
        };
        for p in [
            json!({"clip": long.1}),
            json!({"track": long.0, "start": 0, "end": 601 * 48_000}),
            json!({"track": long.0, "start": 0, "end": i64::MAX}),
        ] {
            match e.execute("audio.analyze", &p) {
                Err(EngineError::BadParams(_, msg)) => assert!(msg.contains("too long"), "{msg}"),
                other => panic!("{p}: {other:?}"),
            }
        }
        // Exactly ten minutes is fine to ask for (here mostly silence, so it is quick to render).
        let ok = e.execute("audio.analyze", &json!({"track": long.0, "start": 0, "end": 600 * 48_000}));
        assert!(ok.is_ok(), "{ok:?}");
        // The short track is unaffected.
        assert!(e.execute("audio.analyze", &json!({"track": t, "start": 0, "end": 48_000})).is_ok());
    }

    #[test]
    fn is_a_pure_query() {
        let (mut e, t, c) = rig(vec![sine(440.0, 0.5, 2.0)], 0);
        let before = e.execute("engine.history", &json!({})).unwrap();
        let session = serde_json::to_string(e.session()).unwrap_or_default();
        e.execute("audio.analyze", &json!({"clip": c})).unwrap();
        e.execute("audio.analyze", &json!({"track": t, "start": 0, "end": 48_000})).unwrap();
        assert!(e.execute("audio.analyze", &json!({"clip": 424_242})).is_err());
        assert_eq!(before, e.execute("engine.history", &json!({})).unwrap(), "no undo step");
        assert!(!e.can_redo());
        assert_eq!(session, serde_json::to_string(e.session()).unwrap_or_default(), "the session is untouched");
        // Undo after analysing still undoes the last real edit, not the query.
        e.execute("track.new", &json!({"name": "After"})).unwrap();
        e.execute("audio.analyze", &json!({"clip": c})).unwrap();
        e.execute("edit.undo", &json!({})).unwrap();
        assert!(e.session().track_by_name("After").is_none());
    }

    #[test]
    fn is_listed_as_a_query_and_survives_empty_params() {
        let spec = find_command("audio.analyze").unwrap();
        assert!(!spec.undoable && !spec.journal);
        let mut e = Engine::default();
        assert!(matches!(e.execute("audio.analyze", &json!({})), Err(EngineError::BadParams(..))));
    }
}
