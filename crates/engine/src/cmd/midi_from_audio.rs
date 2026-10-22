//! `midi.from_audio`: pitch-track a range of an audio track and write the notes straight into a
//! new MIDI clip (Agent Studio S3). The tracker is the one behind Edit › Copy Special › Audio as
//! MIDI ([`super::audio_midi::detect_range`]); this command adds per-note confidence, an optional
//! key snap and quantize, a minimum note length, and creates the clip itself, on an existing
//! MIDI/instrument track or on a new instrument track. One undo step, whatever it creates.

use super::audio_midi::{MAX_SECS, detect_range};
use super::view::grid_param;
use super::*;
use crate::cmd;
use serde_json::json;
use soundcraft_midi::{Note, Sequence, note_name};
use soundcraft_model::{Clip, Insert, Session, TrackKind};
use soundcraft_time::{GridValue, to_samples};

/// Most notes the result lists (`count` is always the real number).
const MAX_LISTED: usize = 2000;
/// Instrument a created track gets, as `track.new` gives instrument tracks.
const DEFAULT_INSTRUMENT: &str = "subtractive_synth";

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        "midi.from_audio",
        "Audio to MIDI Clip",
        [],
        None,
        "{track, start?, end?, to_track?, min_note_ms?: 60, quantize?: 0..1, grid?, key?: 'F minor'} — monophonic pitch tracking of audio into a new MIDI clip → {track, clip, source_track, created_track, start, end, count, notes: [{index, pitch, name, start_ticks, length_ticks, velocity, confidence}]}",
        has_tracks,
        from_audio
    )]
}

/// A tracked note on the session timeline, in samples.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cand {
    start: i64,
    end: i64,
    pitch: u8,
    velocity: u8,
    confidence: f32,
}

impl Cand {
    fn len(&self) -> i64 {
        self.end.saturating_sub(self.start)
    }
}

/// A key to snap pitches to: the pitch classes of `scale` (semitones above `root`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Key {
    root: u8,
    scale: &'static [u8],
}

/// Scale names (lower case) and their semitone offsets from the tonic.
const SCALES: &[(&[&str], &[u8])] = &[
    (&["", "major", "maj", "ionian"], &[0, 2, 4, 5, 7, 9, 11]),
    (&["minor", "min", "m", "aeolian", "natural minor"], &[0, 2, 3, 5, 7, 8, 10]),
    (&["dorian"], &[0, 2, 3, 5, 7, 9, 10]),
    (&["phrygian"], &[0, 1, 3, 5, 7, 8, 10]),
    (&["lydian"], &[0, 2, 4, 6, 7, 9, 11]),
    (&["mixolydian"], &[0, 2, 4, 5, 7, 9, 10]),
    (&["locrian"], &[0, 1, 3, 5, 6, 8, 10]),
    (&["harmonic minor"], &[0, 2, 3, 5, 7, 8, 11]),
    (&["pentatonic", "major pentatonic"], &[0, 2, 4, 7, 9]),
    (&["minor pentatonic"], &[0, 3, 5, 7, 10]),
];

/// `"F minor"`, `"C#"`, `"Bb major"`, `"Am"`, `"D dorian"` … (the tonic is a letter with an optional
/// `#`/`b`; no scale name means major). `None` for anything else.
fn parse_key(text: &str) -> Option<Key> {
    let t = text.trim();
    let mut chars = t.chars();
    let base: i32 = match chars.next()?.to_ascii_uppercase() {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    let rest = chars.as_str();
    let (accidental, rest) = if let Some(r) = rest.strip_prefix(['#', '♯']) {
        (1, r)
    } else if let Some(r) = rest.strip_prefix(['b', '♭']) {
        (-1, r)
    } else {
        (0, rest)
    };
    let name: String = rest.chars().map(|c| if matches!(c, '-' | '_') { ' ' } else { c.to_ascii_lowercase() }).collect();
    let name = name.split_whitespace().collect::<Vec<_>>().join(" ");
    let (_, scale) = SCALES.iter().find(|(names, _)| names.contains(&name.as_str()))?;
    Some(Key { root: (base + accidental).rem_euclid(12) as u8, scale })
}

/// The nearest pitch in the key (the pitch itself when it is in it); a tie goes down.
fn snap_pitch(pitch: u8, key: &Key) -> u8 {
    let in_key = |p: i32| key.scale.contains(&((p - i32::from(key.root)).rem_euclid(12) as u8));
    let p = i32::from(pitch);
    for d in 0..=6 {
        for cand in [p - d, p + d] {
            if (0..=127).contains(&cand) && in_key(cand) {
                return cand as u8;
            }
        }
    }
    pitch
}

/// Join neighbours that touch and now share a pitch (vibrato that a key snap pulled onto one
/// note): the confidence is the mean weighted by length, the velocity the louder one's.
fn merge_touching(notes: Vec<Cand>) -> Vec<Cand> {
    let mut out: Vec<Cand> = Vec::with_capacity(notes.len());
    for n in notes {
        if let Some(prev) = out.last_mut()
            && prev.pitch == n.pitch
            && n.start <= prev.end
        {
            let (a, b) = (prev.len() as f64, n.len() as f64);
            prev.confidence = ((f64::from(prev.confidence) * a + f64::from(n.confidence) * b) / (a + b).max(1.0)) as f32;
            prev.velocity = prev.velocity.max(n.velocity);
            prev.end = prev.end.max(n.end);
            continue;
        }
        out.push(n);
    }
    out
}

/// Move note starts and ends `strength` (0..1) of the way to the nearest line of `grid`, inside
/// `bounds`. A note squeezed to nothing lasts one grid step; notes stay monophonic (an end never
/// passes the next start) and a note that has no room left is dropped.
fn quantize(notes: &mut Vec<Cand>, s: &Session, grid: GridValue, strength: f64, bounds: Range) {
    let (sr, tempo, rate) = (s.sample_rate, &s.tempo, s.frame_rate);
    let toward = |x: i64| -> i64 {
        let target = grid.snap(x, sr, tempo, rate);
        to_samples(x as f64 + target.saturating_sub(x) as f64 * strength).clamp(bounds.start, bounds.end)
    };
    for n in notes.iter_mut() {
        let (a, b) = (toward(n.start), toward(n.end));
        n.start = a;
        n.end = if b > a { b } else { a.saturating_add(grid.step_samples(a, sr, tempo, rate)).min(bounds.end) };
    }
    let mut next_start: Option<i64> = None;
    for n in notes.iter_mut().rev() {
        if let Some(next) = next_start {
            n.end = n.end.min(next);
        }
        if n.end > n.start {
            next_start = Some(n.start);
        }
    }
    notes.retain(|n| n.end > n.start);
}

/// A number parameter that may be absent or null, but is an error when it is anything else.
fn opt_number(p: &Value, id: &str, key: &str) -> Result<Option<f64>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().filter(|x| x.is_finite()).map(Some).ok_or_else(|| bad(id, format!("`{key}` must be a number"))),
    }
}

/// The span of audio to track: `start`/`end` as given (`end` is cut to the end of the track's
/// audio); either one missing is taken from the track's audio; neither means the edit selection,
/// else all of the audio.
fn source_range(e: &Engine, id: &str, p: &Value, t: TrackId) -> Result<Range> {
    let tr = e.session().track(t).ok_or_else(|| bad(id, "no such track"))?;
    let audio = tr.clips().iter().filter(|c| !c.muted);
    let first = audio.clone().map(|c| c.start).min();
    let last = audio.map(Clip::end).max();
    let (Some(first), Some(last)) = (first, last) else {
        return Err(bad(id, format!("track `{}` has no audio (unmute or record something first)", tr.name)));
    };
    let (st, en) = (position_param(e, id, p, "start")?, position_param(e, id, p, "end")?);
    let selection = e.session().edit.selection;
    let (a, b) = match (st, en) {
        (None, None) if !selection.is_empty() => (selection.start, selection.end),
        (None, None) => (first, last),
        (Some(a), None) => (a, last),
        (None, Some(b)) => (first, b),
        (Some(a), Some(b)) => {
            if b <= a {
                return Err(bad(id, "`end` must be after `start`"));
            }
            (a, b)
        }
    };
    let (a, b) = (a.max(0), b.min(last));
    if b <= a {
        return Err(bad(id, "there is no audio in that range"));
    }
    Ok(Range::new(a, b))
}

fn from_audio(e: &mut Engine, p: &Value) -> Result<Value> {
    let id = "midi.from_audio";
    let min_note_ms = opt_number(p, id, "min_note_ms")?.unwrap_or(60.0).clamp(0.0, 5000.0);
    let strength = opt_number(p, id, "quantize")?.unwrap_or(0.0).clamp(0.0, 1.0);
    let key = match p.get("key") {
        None | Some(Value::Null) => None,
        Some(Value::String(k)) => {
            Some(parse_key(k).ok_or_else(|| bad(id, format!("unknown key `{k}`: use a tonic and scale like `F minor`, `Bb major` or `D dorian`")))?)
        }
        Some(_) => return Err(bad(id, "`key` must be text like `F minor`")),
    };
    let grid = match p.get("grid") {
        None | Some(Value::Null) => e.session().edit.grid,
        Some(_) => grid_param(p, "grid").ok_or_else(|| bad(id, "unknown `grid` value (try `1/16`, `1/8t` or {seconds})"))?,
    };

    // Where from and where to, before anything changes.
    let src = tracks_required(e, id, p)?.first().copied().ok_or_else(|| bad(id, "no track"))?;
    let s = e.session();
    let src_track = s.track(src).ok_or_else(|| bad(id, "no such track"))?;
    if src_track.kind != TrackKind::Audio {
        return Err(bad(id, format!("track `{}` is not an audio track", src_track.name)));
    }
    let to = match p.get("to_track") {
        None | Some(Value::Null) => None,
        Some(_) => track_param(e, id, p, "to_track")?,
    };
    if let Some(to) = to {
        let tr = s.track(to).ok_or_else(|| bad(id, "no such `to_track`"))?;
        if !tr.kind.is_midi() {
            return Err(bad(id, format!("`to_track` `{}` is not a MIDI or instrument track", tr.name)));
        }
    }
    let r = source_range(e, id, p, src)?;
    let s = e.session();
    let sr = s.sample_rate.as_f64();
    if r.len() as f64 > MAX_SECS * sr {
        return Err(bad(
            id,
            format!("{:.0} s is too long to track (limit {MAX_SECS:.0} s); pass a shorter `start`/`end` range", r.len() as f64 / sr),
        ));
    }
    let src_track = s.track(src).ok_or_else(|| bad(id, "no such track"))?;
    // Audio that is not in memory would read as silence and be reported as "no notes".
    if let Some(c) = src_track.clips().iter().find(|c| !c.muted && c.range().overlaps(&r) && c.source().is_some_and(|x| !s.pool.contains(x))) {
        return Err(bad(id, format!("the audio of clip {} (`{}`) is not loaded, so it cannot be tracked", c.id.0, c.name)));
    }
    let source_name = src_track.name.clone();

    // Track, then shape the notes.
    let detected = detect_range(s, src, r);
    let found = detected.len();
    let mut notes: Vec<Cand> = detected
        .iter()
        .map(|d| {
            let start = r.start.saturating_add(i64::try_from(d.start).unwrap_or(i64::MAX)).clamp(r.start, r.end);
            let end = r.start.saturating_add(i64::try_from(d.end).unwrap_or(i64::MAX)).clamp(start, r.end);
            let confidence = if d.confidence.is_finite() { d.confidence.clamp(0.0, 1.0) } else { 0.0 };
            Cand { start, end, pitch: d.pitch.min(127), velocity: d.velocity.clamp(1, 127), confidence }
        })
        .collect();
    if let Some(key) = &key {
        for n in &mut notes {
            n.pitch = snap_pitch(n.pitch, key);
        }
        notes = merge_touching(notes);
    }
    let min_len = to_samples(min_note_ms / 1000.0 * sr);
    let before = notes.len();
    notes.retain(|n| n.len() >= min_len && n.len() > 0);
    let too_short = before - notes.len();
    if strength > 0.0 {
        quantize(&mut notes, s, grid, strength, r);
    }
    if notes.is_empty() {
        let why = if found == 0 { "found no notes" } else { "kept no notes (all were shorter than `min_note_ms` or collapsed)" };
        return Err(bad(
            id,
            format!("the tracker {why} in that audio: it follows one clear pitch at a time, so silence, noise and chords give none"),
        ));
    }

    // Ticks are relative to the clip start, like Copy Special › Audio as MIDI.
    let base = s.tempo.samples_to_ticks(r.start, s.sample_rate);
    let mut built: Vec<(Note, f32)> = notes
        .iter()
        .map(|n| {
            let a = s.tempo.samples_to_ticks(n.start, s.sample_rate).saturating_sub(base).max(0);
            let b = s.tempo.samples_to_ticks(n.end, s.sample_rate).saturating_sub(base);
            (
                Note { pitch: n.pitch, velocity: n.velocity, release_velocity: 64, channel: 0, start: a, length: b.saturating_sub(a).max(1) },
                n.confidence,
            )
        })
        .collect();
    built.sort_by_key(|(a, _)| (a.start, a.pitch, a.channel, a.length, a.velocity));
    let seq = Sequence { notes: built.iter().map(|(n, _)| *n).collect(), ..Sequence::default() };

    // Everything below happens inside this one command, so it is one undo step.
    let clip_name = format!("{source_name} MIDI");
    let s = e.session_mut();
    let (target, created) = match to {
        Some(t) => (t, false),
        None => {
            let t = s.add_track(TrackKind::Instrument, soundcraft_model::ChannelFormat::Stereo, Some(&clip_name));
            if let Some(tr) = s.track_mut(t) {
                tr.instrument = Some(Insert::new(DEFAULT_INSTRUMENT));
            }
            (t, true)
        }
    };
    let overwrote = s.track(target).map_or(0, |tr| tr.clips().iter().filter(|c| c.range().overlaps(&r)).count());
    let clip = s.new_clip_id();
    crate::edit::place_clip(s, target, Clip::midi(clip, clip_name, r.start, r.len(), seq));
    if s.find_clip(clip).is_none() {
        return Err(bad(id, "the target track cannot hold clips"));
    }

    let count = built.len();
    let listed: Vec<Value> = built
        .iter()
        .take(MAX_LISTED)
        .enumerate()
        .map(|(index, (n, confidence))| {
            json!({
                "index": index,
                "pitch": n.pitch,
                "name": note_name(n.pitch),
                "start_ticks": n.start,
                "length_ticks": n.length,
                "velocity": n.velocity,
                "confidence": (f64::from(*confidence) * 1000.0).round() / 1000.0,
            })
        })
        .collect();
    Ok(json!({
        "track": target.0,
        "clip": clip.0,
        "source_track": src.0,
        "created_track": created,
        "start": r.start,
        "end": r.end,
        "count": count,
        "detected": found,
        "dropped_short": too_short,
        "overwrote_clips": overwrote,
        "notes_truncated": count > MAX_LISTED,
        "notes": listed,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_parse_in_the_forms_an_agent_writes_them() {
        let major = &[0, 2, 4, 5, 7, 9, 11][..];
        let minor = &[0, 2, 3, 5, 7, 8, 10][..];
        assert_eq!(parse_key("C"), Some(Key { root: 0, scale: major }));
        assert_eq!(parse_key("  f minor "), Some(Key { root: 5, scale: minor }));
        assert_eq!(parse_key("F# Minor"), Some(Key { root: 6, scale: minor }));
        assert_eq!(parse_key("Bb major"), Some(Key { root: 10, scale: major }));
        assert_eq!(parse_key("Ebm"), Some(Key { root: 3, scale: minor }));
        assert_eq!(parse_key("Am"), Some(Key { root: 9, scale: minor }));
        assert_eq!(parse_key("bm"), Some(Key { root: 11, scale: minor }));
        assert_eq!(parse_key("Cb major"), Some(Key { root: 11, scale: major }));
        assert_eq!(parse_key("D dorian").map(|k| k.root), Some(2));
        assert_eq!(parse_key("A harmonic-minor").map(|k| k.scale.len()), Some(7));
        assert_eq!(parse_key("E minor pentatonic").map(|k| k.scale.len()), Some(5));
        for bad in ["", " ", "H major", "C lunar", "major", "C#b", "C minor extra", "\u{0}", "Ça", "C\u{301}"] {
            assert_eq!(parse_key(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn pitches_snap_to_the_nearest_scale_note_with_ties_going_down() {
        let c_major = parse_key("C major").unwrap();
        assert_eq!(snap_pitch(60, &c_major), 60);
        // C# sits between C and D: the tie goes down. D# likewise goes to D, F# to F.
        assert_eq!(snap_pitch(61, &c_major), 60);
        assert_eq!(snap_pitch(63, &c_major), 62);
        assert_eq!(snap_pitch(66, &c_major), 65);
        assert_eq!(snap_pitch(70, &c_major), 69);
        let a_minor = parse_key("A minor").unwrap();
        assert_eq!(snap_pitch(71, &a_minor), 71);
        assert_eq!(snap_pitch(73, &a_minor), 72, "C# in A minor");
        // Every pitch lands in the key, and never moves more than a tone in a 7-note scale.
        for key in ["C major", "F# minor", "Bb dorian", "D minor pentatonic", "E harmonic minor"] {
            let k = parse_key(key).unwrap();
            for p in 0..=127u8 {
                let q = snap_pitch(p, &k);
                assert!(k.scale.contains(&((i32::from(q) - i32::from(k.root)).rem_euclid(12) as u8)), "{key}: {p} -> {q}");
                assert!((i32::from(p) - i32::from(q)).abs() <= 3, "{key}: {p} -> {q}");
            }
        }
        // The extremes of the keyboard stay on it.
        assert_eq!(snap_pitch(0, &parse_key("D major").unwrap()), 1);
        assert_eq!(snap_pitch(127, &parse_key("C major").unwrap()), 127);
        assert_eq!(snap_pitch(126, &parse_key("C major").unwrap()), 125);
    }

    fn cand(start: i64, end: i64, pitch: u8, confidence: f32) -> Cand {
        Cand { start, end, pitch, velocity: 80, confidence }
    }

    #[test]
    fn touching_notes_of_one_pitch_merge_with_a_length_weighted_confidence() {
        let merged = merge_touching(vec![cand(0, 100, 60, 1.0), cand(100, 400, 60, 0.0), cand(400, 500, 62, 0.5), cand(600, 700, 62, 0.5)]);
        assert_eq!(merged.len(), 3, "{merged:?}");
        assert_eq!((merged[0].start, merged[0].end), (0, 400));
        assert!((merged[0].confidence - 0.25).abs() < 1e-6, "{}", merged[0].confidence);
        // A gap keeps notes apart.
        assert_eq!(merged[2].start, 600);
        assert!(merge_touching(Vec::new()).is_empty());
    }

    #[test]
    fn quantize_pulls_toward_the_grid_by_strength_and_keeps_notes_apart() {
        let s = Session::new("q", soundcraft_time::SampleRate::HZ_48000);
        let grid = GridValue::Samples(1000);
        let bounds = Range::new(0, 10_000);
        let mut full = vec![cand(1040, 2480, 60, 0.9), cand(2480, 3900, 62, 0.9)];
        quantize(&mut full, &s, grid, 1.0, bounds);
        assert_eq!((full[0].start, full[0].end, full[1].start, full[1].end), (1000, 2000, 2000, 4000), "{full:?}");
        let mut half = vec![cand(1040, 2480, 60, 0.9)];
        quantize(&mut half, &s, grid, 0.5, bounds);
        assert_eq!((half[0].start, half[0].end), (1020, 2240));
        let mut none = vec![cand(1040, 2480, 60, 0.9)];
        quantize(&mut none, &s, grid, 0.0, bounds);
        assert_eq!((none[0].start, none[0].end), (1040, 2480));
        // A note shorter than the grid collapses onto one line and then lasts one grid step.
        let mut tiny = vec![cand(1100, 1300, 60, 0.9)];
        quantize(&mut tiny, &s, grid, 1.0, bounds);
        assert_eq!((tiny[0].start, tiny[0].end), (1000, 2000));
        // Two notes landing on the same line leave one; the later one wins the line.
        let mut clash = vec![cand(1100, 1200, 60, 0.9), cand(1300, 2600, 62, 0.9)];
        quantize(&mut clash, &s, grid, 1.0, bounds);
        assert_eq!(clash.len(), 1, "{clash:?}");
        assert_eq!((clash[0].pitch, clash[0].start), (62, 1000));
        // Nothing leaves the bounds, whatever the strength.
        let mut edge = vec![cand(9_900, 10_000, 60, 0.9)];
        quantize(&mut edge, &s, grid, 1.0, Range::new(9_900, 10_000));
        assert!(edge.iter().all(|n| n.start >= 9_900 && n.end <= 10_000), "{edge:?}");
    }

    use soundcraft_audio_io::{AudioBuffer, FileFormat};
    use soundcraft_model::{ChannelFormat, ClipContent};

    const SR: f32 = 48_000.0;

    fn tone(f: f32, secs: f32) -> Vec<f32> {
        (0..(secs * SR) as usize).map(|i| (i as f32 * f * std::f32::consts::TAU / SR).sin() * 0.5).collect()
    }

    /// An engine with one mono audio track holding `audio` from sample 0; returns the track.
    fn engine_with(audio: Vec<f32>) -> (Engine, TrackId, i64) {
        let mut e = Engine::default();
        let frames = audio.len() as i64;
        let s = e.session_mut();
        let src = crate::io::add_source(s, "Hum", AudioBuffer { sample_rate: 48_000, channels: vec![audio] }, None, FileFormat::Wav);
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, Some("Hum"));
        let cid = s.new_clip_id();
        crate::edit::place_clip(s, t, Clip::audio(cid, "Hum", src, 0, 0, frames));
        (e, t, frames)
    }

    fn melody() -> Vec<f32> {
        let mut a = tone(220.0, 0.4);
        a.extend(tone(330.0, 0.4));
        a
    }

    fn pitches(v: &Value) -> Vec<u64> {
        v["notes"].as_array().map(|n| n.iter().filter_map(|n| n["pitch"].as_u64()).collect()).unwrap_or_default()
    }

    fn midi_notes(e: &Engine, clip: u64) -> Vec<Note> {
        match e.session().find_clip(soundcraft_model::ClipId(clip)).map(|(_, c)| &c.content) {
            Some(ClipContent::Midi { sequence }) => sequence.notes.clone(),
            other => panic!("expected a MIDI clip, got {other:?}"),
        }
    }

    #[test]
    fn a_melody_becomes_a_clip_on_a_new_instrument_track_in_one_undo_step() {
        let (mut e, t, frames) = engine_with(melody());
        let (steps, tracks) = (e.undo_history().len(), e.session().tracks.len());
        let r = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames})).unwrap();
        assert_eq!(pitches(&r), vec![57, 64], "{r}");
        assert_eq!(r["count"], 2);
        assert_eq!(r["created_track"], true);
        assert_eq!(r["source_track"], t.0);
        let notes = r["notes"].as_array().unwrap();
        for (i, n) in notes.iter().enumerate() {
            assert_eq!(n["index"], i);
            let c = n["confidence"].as_f64().unwrap();
            assert!((0.5..=1.0).contains(&c), "confidence {c}: {r}");
            assert!(n["length_ticks"].as_i64().unwrap() > 0 && n["start_ticks"].as_i64().unwrap() >= 0);
            assert!((1..=127).contains(&n["velocity"].as_i64().unwrap()));
        }
        assert_eq!(notes[0]["name"], "A2");

        // The new track is an instrument track named after the source, holding the clip.
        let target = TrackId(r["track"].as_u64().unwrap());
        let track = e.session().track(target).unwrap();
        assert_eq!(track.kind, TrackKind::Instrument);
        assert_eq!(track.name, "Hum MIDI");
        assert!(track.instrument.is_some());
        assert_eq!(track.clips().len(), 1);
        let clip = r["clip"].as_u64().unwrap();
        assert_eq!(track.clips()[0].id.0, clip);
        assert_eq!((track.clips()[0].start, track.clips()[0].end()), (0, frames));
        let stored = midi_notes(&e, clip);
        assert_eq!(stored.iter().map(|n| u64::from(n.pitch)).collect::<Vec<_>>(), vec![57, 64]);
        assert_eq!(stored[0].start, notes[0]["start_ticks"].as_i64().unwrap());
        assert_eq!(stored[1].length, notes[1]["length_ticks"].as_i64().unwrap());
        assert_eq!(e.session().tracks.len(), tracks + 1);

        // One step: undo removes the track and the clip together, redo brings both back.
        assert_eq!(e.undo_history().len(), steps + 1, "{:?}", e.undo_history());
        assert_eq!(e.undo_history().last().map(String::as_str), Some("Audio to MIDI Clip"));
        assert!(e.undo());
        assert_eq!(e.session().tracks.len(), tracks);
        assert!(e.session().track(target).is_none() && e.session().find_clip(soundcraft_model::ClipId(clip)).is_none());
        assert!(e.redo());
        assert_eq!(midi_notes(&e, clip).len(), 2);
        assert_eq!(e.session().track(target).map(|t| t.name.as_str()), Some("Hum MIDI"));
    }

    #[test]
    fn the_clip_goes_to_a_given_midi_track_and_replaces_what_it_covers() {
        let (mut e, t, frames) = engine_with(melody());
        e.execute("track.new", &json!({"kind": "midi", "name": "Lead"})).unwrap();
        let lead = e.session().track_by_name("Lead").unwrap().id;
        e.execute("midi.new_clip", &json!({"track": lead.0, "start": 0, "end": 4800})).unwrap();
        let (steps, tracks) = (e.undo_history().len(), e.session().tracks.len());
        let r = e.execute("midi.from_audio", &json!({"track": "Hum", "to_track": "Lead", "start": 0, "end": frames})).unwrap();
        assert_eq!((r["track"].as_u64(), r["created_track"].as_bool()), (Some(lead.0), Some(false)), "{r}");
        assert_eq!(r["overwrote_clips"], 1);
        assert_eq!(e.session().tracks.len(), tracks, "no track was created");
        let clips = e.session().track(lead).unwrap().clips();
        assert_eq!(clips.len(), 1, "the empty clip it covered is gone");
        assert_eq!(clips[0].id.0, r["clip"].as_u64().unwrap());
        assert_eq!(e.undo_history().len(), steps + 1);
        assert!(e.undo());
        assert_eq!(e.session().track(lead).unwrap().clips().len(), 1);
        assert_ne!(e.session().track(lead).unwrap().clips()[0].id.0, r["clip"].as_u64().unwrap());
        // An instrument track works too, and the source can come from the selection.
        e.execute("track.new", &json!({"kind": "instrument", "name": "Keys"})).unwrap();
        e.execute("edit.select", &json!({"tracks": [t.0], "start": 0, "end": frames})).unwrap();
        let r = e.execute("midi.from_audio", &json!({"to_track": "Keys"})).unwrap();
        assert_eq!(pitches(&r), vec![57, 64], "{r}");
        assert_eq!((r["start"].as_i64(), r["end"].as_i64()), (Some(0), Some(frames)));
    }

    #[test]
    fn refusals_leave_the_session_exactly_as_it_was() {
        let (mut e, t, frames) = engine_with(melody());
        e.execute("track.new", &json!({"kind": "audio", "name": "Other"})).unwrap();
        e.execute("track.new", &json!({"kind": "midi", "name": "Notes"})).unwrap();
        let steps = e.undo_history().len();
        let before = e.session().clone();
        let range = json!({"start": 0, "end": frames});
        let with = |extra: Value| -> Value {
            let mut m = range.as_object().cloned().unwrap_or_default();
            m.insert("track".into(), json!(t.0));
            m.extend(extra.as_object().cloned().unwrap_or_default());
            Value::Object(m)
        };
        for (what, params) in [
            ("an audio track as the target", with(json!({"to_track": "Other"}))),
            ("a missing target", with(json!({"to_track": "Nope"}))),
            ("a target that is not a name or id", with(json!({"to_track": true}))),
            ("a MIDI track as the source", json!({"track": "Notes", "start": 0, "end": frames})),
            ("a track without audio", json!({"track": "Other", "start": 0, "end": frames})),
            ("a missing source", json!({"track": 987_654, "start": 0, "end": frames})),
            ("end before start", json!({"track": t.0, "start": frames, "end": 0})),
            ("an empty range", json!({"track": t.0, "start": 100, "end": 100})),
            ("a range after the audio", json!({"track": t.0, "start": frames + 10, "end": frames + 5000})),
            ("a start after the audio and no end", json!({"track": t.0, "start": frames + 10})),
            ("an end before the audio and no start", json!({"track": t.0, "end": -50})),
            ("a quantize that is text", with(json!({"quantize": "tight"}))),
            ("a quantize that is a flag", with(json!({"quantize": true}))),
            ("a min_note_ms that is text", with(json!({"min_note_ms": "60"}))),
            ("an unknown key", with(json!({"key": "H lunar"}))),
            ("a key that is a number", with(json!({"key": 5}))),
            ("an unknown grid", with(json!({"quantize": 1, "grid": "1/13"}))),
            ("every note shorter than the minimum", with(json!({"min_note_ms": 1e300}))),
        ] {
            let r = e.execute("midi.from_audio", &params);
            assert!(matches!(r, Err(EngineError::BadParams(..))), "{what}: {r:?}");
            assert_eq!(e.undo_history().len(), steps, "{what}: left an undo step");
            assert!(*e.session() == before, "{what}: changed the session");
        }
    }

    #[test]
    fn a_range_outside_the_audio_says_so_instead_of_tracking_the_gap() {
        let (mut e, t, frames) = engine_with(melody());
        for p in [
            json!({"start": frames + 10, "end": frames + 5000}),
            json!({"start": frames + 10}),
            json!({"start": frames, "end": frames + 1}),
            json!({"end": -1}),
        ] {
            let mut p = p;
            p["track"] = json!(t.0);
            let msg = e.execute("midi.from_audio", &p).unwrap_err().to_string();
            assert!(msg.contains("no audio in that range"), "{p}: {msg}");
        }
    }

    #[test]
    fn the_clip_survives_a_save_and_reopen() {
        let (mut e, t, frames) = engine_with(melody());
        let r = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames, "key": "A minor", "quantize": 0.5})).unwrap();
        let clip = r["clip"].as_u64().unwrap();
        let before = midi_notes(&e, clip);
        assert!(!before.is_empty());
        let text = e.session().to_json().unwrap();
        let mut reopened = Engine::default();
        reopened.replace_session(Session::from_json(&text).unwrap());
        assert_eq!(midi_notes(&reopened, clip), before);
        let track = reopened.session().track(TrackId(r["track"].as_u64().unwrap())).unwrap();
        assert_eq!((track.kind, track.name.as_str()), (TrackKind::Instrument, "Hum MIDI"));
        assert!(track.instrument.is_some());
    }

    #[test]
    fn silence_and_noise_find_no_notes_and_change_nothing() {
        for audio in [vec![0.0f32; 48_000], (0..48_000u32).map(|i| ((i.wrapping_mul(2_654_435_761) >> 16) as f32 / 32_768.0 - 1.0) * 0.5).collect()] {
            let (mut e, t, frames) = engine_with(audio);
            let steps = e.undo_history().len();
            let tracks = e.session().tracks.len();
            let r = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames}));
            match r {
                Err(EngineError::BadParams(_, msg)) => assert!(msg.contains("no notes"), "{msg}"),
                // Noise may happen to track a few notes; then they are real notes in a real clip.
                Ok(v) => assert!(v["count"].as_u64().unwrap_or(0) > 0, "{v}"),
                other => panic!("{other:?}"),
            }
            if e.session().tracks.len() == tracks {
                assert_eq!(e.undo_history().len(), steps);
            }
        }
        let (mut e, t, frames) = engine_with(vec![0.0; 48_000]);
        let msg = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames})).unwrap_err().to_string();
        assert!(msg.contains("no notes"), "{msg}");
        assert_eq!(e.session().tracks.len(), 1);
    }

    #[test]
    fn a_key_snaps_pitches_and_a_minimum_length_drops_short_notes() {
        // C#4 (277.18 Hz) is not in C major: the tie between C and D goes down.
        let mut audio = tone(277.18, 0.4);
        audio.extend(tone(329.63, 0.4));
        let (mut e, t, frames) = engine_with(audio);
        let plain = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames})).unwrap();
        assert_eq!(pitches(&plain), vec![61, 64], "{plain}");
        let snapped = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames, "key": "C major"})).unwrap();
        assert_eq!(pitches(&snapped), vec![60, 64], "{snapped}");
        let minor = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames, "key": "C# minor"})).unwrap();
        assert_eq!(pitches(&minor), vec![61, 64], "E is in C# minor and so is C#: {minor}");

        // A 0.12 s blip between two longer notes goes under a 200 ms minimum, and stays at 60 ms.
        let mut audio = tone(220.0, 0.4);
        audio.extend(tone(262.0, 0.12));
        audio.extend(tone(330.0, 0.4));
        let (mut e, t, frames) = engine_with(audio);
        let all = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames})).unwrap();
        assert_eq!(all["count"], 3, "{all}");
        let long = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames, "min_note_ms": 200})).unwrap();
        assert_eq!(pitches(&long), vec![57, 64], "{long}");
        assert_eq!(long["dropped_short"], 1);
        assert_eq!(long["detected"], 3);
    }

    #[test]
    fn quantize_moves_notes_onto_the_session_grid_by_strength() {
        // 120 BPM: a 1/16 is 125 ms = 240 ticks. The tone starts 50 ms in, off the grid.
        let mut audio = vec![0.0f32; 2400];
        audio.extend(tone(220.0, 0.4));
        audio.extend(tone(330.0, 0.4));
        let (mut e, t, frames) = engine_with(audio);
        e.execute("view.grid", &json!({"value": "1/16"})).unwrap();
        let raw = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames})).unwrap();
        let raw_start = raw["notes"][0]["start_ticks"].as_i64().unwrap();
        assert!(raw_start % 240 != 0, "the test needs an off-grid start: {raw}");
        let full = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames, "quantize": 1})).unwrap();
        let starts: Vec<i64> = full["notes"].as_array().unwrap().iter().map(|n| n["start_ticks"].as_i64().unwrap()).collect();
        assert!(starts.iter().all(|s| s % 240 == 0), "{starts:?}");
        assert_eq!(full["notes"][0]["length_ticks"].as_i64().unwrap() % 240, 0, "{full}");
        let half = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames, "quantize": 0.5})).unwrap();
        let moved = half["notes"][0]["start_ticks"].as_i64().unwrap();
        let target = starts[0];
        assert!((moved - (raw_start + target) / 2).abs() <= 1, "halfway between {raw_start} and {target}: {moved}");
        // A coarser grid per call; the session grid is untouched.
        let bar = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames, "quantize": 1, "grid": "1/4"})).unwrap();
        assert!(bar["notes"].as_array().unwrap().iter().all(|n| n["start_ticks"].as_i64().unwrap() % 960 == 0), "{bar}");
        assert_eq!(e.session().edit.grid.label(), "1/16 note");
        // Strengths outside 0..1 are clamped, not errors.
        for q in [-3.0, 7.5] {
            e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames, "quantize": q})).unwrap();
        }
    }

    #[test]
    fn a_range_into_the_audio_places_the_clip_there() {
        // Start at 0.4 s: only the second tone (E4) is in the range.
        let (mut e, t, frames) = engine_with(melody());
        let at = (0.4 * SR) as i64;
        let r = e.execute("midi.from_audio", &json!({"track": t.0, "start": at - 480, "end": frames + 1_000_000_000})).unwrap();
        assert_eq!((r["start"].as_i64(), r["end"].as_i64()), (Some(at - 480), Some(frames)), "end is cut to the audio: {r}");
        assert!(pitches(&r).ends_with(&[64]), "{r}");
        let clip = r["clip"].as_u64().unwrap();
        let first = midi_notes(&e, clip)[0];
        // Ticks count from the clip's own start, which is 480 samples before the note.
        assert!(first.start >= 0 && first.start < 960, "{first:?}");
        // Seconds and open ends work like the other range commands.
        let r = e.execute("midi.from_audio", &json!({"track": "Hum", "start": {"seconds": 0.4}})).unwrap();
        assert_eq!(pitches(&r), vec![64], "{r}");
        let r = e.execute("midi.from_audio", &json!({"track": "Hum", "end": {"seconds": 0.4}})).unwrap();
        assert_eq!(pitches(&r), vec![57], "{r}");
        let r = e.execute("midi.from_audio", &json!({"track": "Hum"})).unwrap();
        assert_eq!(r["count"], 2, "no range at all means all of the audio: {r}");
    }

    #[test]
    fn hostile_numbers_and_enormous_ranges_are_handled() {
        let (mut e, t, frames) = engine_with(melody());
        for (start, end) in
            [(json!(i64::MIN), json!(i64::MAX)), (json!(-5), json!(f64::MAX)), (json!(0), json!(1e300)), (json!(frames - 1), json!(i64::MAX))]
        {
            let r = e.execute("midi.from_audio", &json!({"track": t.0, "start": start, "end": end}));
            assert!(r.is_ok() || matches!(r, Err(EngineError::BadParams(..))), "{start} {end}: {r:?}");
        }
        for extra in
            [json!({"quantize": 1e308}), json!({"min_note_ms": -1e308}), json!({"quantize": -0.0, "min_note_ms": 0}), json!({"key": "C\u{0}"})]
        {
            let mut p = json!({"track": t.0});
            if let Some(m) = p.as_object_mut() {
                m.extend(extra.as_object().cloned().unwrap_or_default());
            }
            let r = e.execute("midi.from_audio", &p);
            assert!(!matches!(r, Err(EngineError::Internal(..))), "{extra}: {r:?}");
        }
        // Past the limit is refused before any audio is rendered.
        let secs = (MAX_SECS as i64 + 20) * 48_000;
        let s = e.session_mut();
        let long = s.add_track(TrackKind::Audio, ChannelFormat::Mono, Some("Long"));
        let src = s.sources.first().map(|x| x.id).unwrap();
        let cid = s.new_clip_id();
        crate::edit::place_clip(s, long, Clip::audio(cid, "Long", src, 0, 0, secs));
        let msg = e.execute("midi.from_audio", &json!({"track": "Long"})).unwrap_err().to_string();
        assert!(msg.contains("too long"), "{msg}");
        let r = e.execute("midi.from_audio", &json!({"track": "Long", "start": 0, "end": 48_000}));
        assert!(r.is_ok() || matches!(r, Err(EngineError::BadParams(..))), "{r:?}");
    }

    #[test]
    fn audio_that_is_not_loaded_is_reported_not_called_silent() {
        let (mut e, t, frames) = engine_with(melody());
        let src = e.session().sources[0].id;
        e.session_mut().pool.remove(src);
        let msg = e.execute("midi.from_audio", &json!({"track": t.0, "start": 0, "end": frames})).unwrap_err().to_string();
        assert!(msg.contains("not loaded"), "{msg}");
    }

    #[test]
    fn copy_special_still_puts_the_same_notes_on_the_clipboard() {
        let (mut e, t, frames) = engine_with(melody());
        let r = e.execute("edit.copy_audio_as_midi", &json!({"track": t.0, "start": 0, "end": frames})).unwrap();
        assert_eq!(r["notes"], 2, "{r}");
    }

    #[test]
    fn it_is_listed_with_its_undo_label_and_reachable_without_a_selection() {
        let spec = find_command("midi.from_audio").unwrap();
        assert!(spec.journal && spec.undoable && !spec.view);
        assert_eq!(spec.label, "Audio to MIDI Clip");
        // On an empty session it is greyed out and, with no targets named, refused without a panic.
        let mut e = Engine::default();
        assert!(matches!(e.execute("midi.from_audio", &json!({})), Err(EngineError::Disabled(..))));
    }

    #[test]
    fn numbers_that_are_not_numbers_are_refused() {
        let p = serde_json::json!({"a": "x", "b": null, "c": 2.5});
        assert!(opt_number(&p, "t", "a").is_err());
        assert_eq!(opt_number(&p, "t", "b").unwrap(), None);
        assert_eq!(opt_number(&p, "t", "c").unwrap(), Some(2.5));
        assert_eq!(opt_number(&p, "t", "missing").unwrap(), None);
    }
}
