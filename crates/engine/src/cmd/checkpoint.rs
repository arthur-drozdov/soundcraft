//! `session.checkpoint` and `session.checkpoints` (Agent Studio S7): named restore points kept
//! beside the session file, so an agent can say "before the chorus layer" and a person can open
//! that exact state later. A checkpoint is a copy of the session (`session.save_copy`), written to
//! `Checkpoints/<YYYYMMDD-HHMMSS>-<label>.scraft` next to the session file (UTC time; the label
//! reduced to a safe file name). The open session, its path and its unsaved flag are untouched.

use super::*;
use crate::cmd;
use serde_json::json;
use std::path::{Path, PathBuf};

/// Longest label slug in a file name, in characters.
const MAX_SLUG: usize = 40;
/// Label used when none is given, or none of it can go in a file name.
const DEFAULT_LABEL: &str = "checkpoint";
/// 9999-12-31T23:59:59Z: the last time four year digits can show.
const MAX_SECS: u64 = 253_402_300_799;
/// How many seconds forward a name may be moved to find a free one.
const MAX_BUMPS: u32 = 1000;

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            noundo "session.checkpoint",
            "Save Checkpoint",
            [],
            None,
            "{label?} — copy the session to Checkpoints/<YYYYMMDD-HHMMSS>-<label>.scraft beside the session file (it must have been saved) → {path, label, created, audio_files_written}",
            |e| if e.path.is_some() { Ok(()) } else { Err("the session has not been saved".into()) },
            checkpoint
        ),
        cmd!(
            query "session.checkpoints",
            "List Checkpoints",
            [],
            None,
            "{} — the checkpoints beside the session file, newest first → [{path, label, created}]",
            always,
            checkpoints
        ),
    ]
}

/// `label` as the part of a file name after the time: lower case letters and digits, any other
/// run of characters one `-`, at most [`MAX_SLUG`] characters. Nothing in it can leave the folder
/// (`../../x` is `x`).
fn slug(label: &str) -> String {
    let mut out = String::new();
    for c in label.chars().take(4 * MAX_SLUG) {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let out: String = out.trim_matches('-').chars().take(MAX_SLUG).collect();
    let out = out.trim_matches('-');
    if out.is_empty() { DEFAULT_LABEL.into() } else { out.into() }
}

/// Days since 1970-01-01 → (year, month, day) in the proleptic Gregorian calendar.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

/// The inverse of [`civil_from_days`].
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `(year, month, day, hour, minute, second)` of Unix time `secs`.
fn fields(secs: u64) -> (i64, i64, i64, u64, u64, u64) {
    let secs = secs.min(MAX_SECS);
    let (y, m, d) = civil_from_days(i64::try_from(secs / 86_400).unwrap_or(0));
    let rem = secs % 86_400;
    (y, m, d, rem / 3600, rem % 3600 / 60, rem % 60)
}

/// `20261022-091503`.
fn stamp(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = fields(secs);
    format!("{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}")
}

/// `2026-10-22T09:15:03Z`.
fn iso(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = fields(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// The time and label slug of a checkpoint file name, `<stamp>-<slug>.scraft`; `None` for any
/// other file.
fn parse_name(name: &str) -> Option<(u64, &str)> {
    let stem = name.strip_suffix(&format!(".{}", soundcraft_model::SESSION_EXTENSION))?;
    let (time, label) = (stem.get(..15)?, stem.get(16..)?);
    let b = time.as_bytes();
    if b.get(8) != Some(&b'-') || stem.as_bytes().get(15) != Some(&b'-') || label.is_empty() {
        return None;
    }
    let digits = |r: std::ops::Range<usize>| -> Option<i64> {
        let part = time.get(r)?;
        if part.bytes().all(|c| c.is_ascii_digit()) { part.parse().ok() } else { None }
    };
    let (y, mo, d) = (digits(0..4)?, digits(4..6)?, digits(6..8)?);
    let (h, mi, s) = (digits(9..11)?, digits(11..13)?, digits(13..15)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || s > 59 {
        return None;
    }
    let secs = u64::try_from(days_from_civil(y, mo, d).checked_mul(86_400)?.checked_add(h * 3600 + mi * 60 + s)?).ok()?;
    // 30 February and the like do not come back as the same text.
    (stamp(secs) == time).then_some((secs, label))
}

/// Seconds since the Unix epoch (0 for a clock set before it).
fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// `path` as an absolute path where that is possible without touching the file system.
fn absolute(path: PathBuf) -> PathBuf {
    std::path::absolute(&path).unwrap_or(path)
}

fn checkpoint(e: &mut Engine, p: &Value) -> Result<Value> {
    let id = "session.checkpoint";
    if cfg!(target_arch = "wasm32") {
        // The clock and the folders are not there; the browser saves bundles instead.
        return Err(bad(id, "checkpoints are not available in the browser: use Save As"));
    }
    let label = match p.get("label") {
        None | Some(Value::Null) => "",
        Some(Value::String(s)) => s,
        Some(_) => return Err(bad(id, "`label` must be text")),
    };
    let session = e.path.clone().ok_or_else(|| bad(id, "save the session first: checkpoints are kept beside the session file"))?;
    let dir = crate::io::checkpoint_dir(Path::new(&session));
    let slug = slug(label);

    // Every checkpoint is later than the ones before it, so the names sort in the order they were
    // made even within a second (or after the clock was set back).
    // A folder that may be written but not read (a write-only automation root) is not listed.
    let latest = crate::io::list_session_files(&dir).unwrap_or_default().iter().filter_map(|n| parse_name(n)).map(|(t, _)| t).max();
    let now = now_secs();
    let mut secs = match latest {
        Some(t) if t >= now => t.saturating_add(1),
        _ => now,
    };
    let mut bumps = 0;
    let file = loop {
        let file = dir.join(format!("{}-{slug}.{}", stamp(secs), soundcraft_model::SESSION_EXTENSION));
        if !crate::io::exists_for_write(&file) {
            break file;
        }
        bumps += 1;
        if bumps > MAX_BUMPS || secs >= MAX_SECS {
            return Err(bad(id, "there are too many checkpoints with that label already"));
        }
        secs += 1;
    };
    let written = crate::io::save_checkpoint(e, &file)?;
    Ok(json!({"path": absolute(file), "label": slug, "created": iso(secs), "audio_files_written": written}))
}

fn checkpoints(e: &mut Engine, _: &Value) -> Result<Value> {
    // A session that was never saved has no folder to look in, and so no checkpoints.
    let Some(session) = e.path.as_deref() else { return Ok(json!([])) };
    let dir = crate::io::checkpoint_dir(Path::new(session));
    let mut found: Vec<(u64, String, String)> = crate::io::list_session_files(&dir)?
        .into_iter()
        .filter_map(|name| {
            let (t, label) = parse_name(&name)?;
            let label = label.to_string();
            Some((t, label, name))
        })
        .collect();
    // Newest first; names are unique, so this is a total order.
    found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.2.cmp(&a.2)));
    Ok(Value::Array(found.into_iter().map(|(t, label, name)| json!({"path": absolute(dir.join(name)), "label": label, "created": iso(t)})).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_become_safe_file_names() {
        assert_eq!(slug("Before the chorus"), "before-the-chorus");
        assert_eq!(slug("  v2 / final!! "), "v2-final");
        assert_eq!(slug(""), "checkpoint");
        assert_eq!(slug("   "), "checkpoint");
        assert_eq!(slug("---"), "checkpoint");
        assert_eq!(slug("Mix_v3"), "mix-v3");
        // Nothing that moves a path survives.
        for hostile in [
            "../../x",
            "..\\..\\x",
            "/etc/passwd",
            "C:\\Windows\\System32",
            "a/../../b",
            "..",
            ".",
            "con",
            "nul\u{0}x",
            "x\ny",
            "~/.ssh/id_rsa",
            "\u{202e}gpj.exe",
            "name.scraft/../..",
            "$(rm -rf /)",
            "a:b*c?d\"e<f>g|h",
        ] {
            let s = slug(hostile);
            assert!(!s.is_empty() && s.len() <= 4 * MAX_SLUG, "{hostile:?}: {s:?}");
            assert!(s.chars().all(|c| c.is_alphanumeric() || c == '-'), "{hostile:?}: {s:?}");
            assert!(!s.starts_with('-') && !s.ends_with('-') && !s.contains("--"), "{hostile:?}: {s:?}");
        }
        assert_eq!(slug("../../x"), "x");
        assert_eq!(slug("a/../../b"), "a-b");
        assert_eq!(slug("..\\..\\x"), "x");
        assert_eq!(slug("/etc/passwd"), "etc-passwd");
        // Long labels are cut, without cutting a character in two, and never end in a dash.
        let long = slug(&"abcdefghij ".repeat(100));
        assert!(long.chars().count() <= MAX_SLUG && !long.ends_with('-'), "{long}");
        let wide = slug(&"é".repeat(1000));
        assert_eq!(wide.chars().count(), MAX_SLUG);
        assert_eq!(slug("Façade ÉTÉ 日本語"), "façade-été-日本語");
        assert_eq!(slug(&"-".repeat(100_000)), "checkpoint");
    }

    #[test]
    fn times_format_and_parse_back() {
        assert_eq!(stamp(0), "19700101-000000");
        assert_eq!(stamp(1_000_000_000), "20010909-014640");
        assert_eq!(iso(1_000_000_000), "2001-09-09T01:46:40Z");
        // A leap day, the end of a century that is not a leap year, and the far end.
        assert_eq!(stamp(1_709_164_800), "20240229-000000");
        assert_eq!(stamp(4_107_542_400), "21000301-000000");
        assert_eq!(stamp(u64::MAX), "99991231-235959");
        for secs in [0, 59, 86_399, 86_400, 951_782_400, 1_000_000_000, 1_709_164_800, 1_792_000_000, 4_107_542_399, MAX_SECS] {
            let name = format!("{}-x.scraft", stamp(secs));
            assert_eq!(parse_name(&name), Some((secs, "x")), "{secs}");
        }
        // Every day for a while round-trips (civil conversions agree in both directions).
        for day in (0..60_000i64).step_by(7) {
            let (y, m, d) = civil_from_days(day);
            assert_eq!(days_from_civil(y, m, d), day);
        }
    }

    #[test]
    fn only_checkpoint_names_parse() {
        assert_eq!(parse_name("20261022-091503-before-chorus.scraft"), Some((1_792_660_503, "before-chorus")));
        assert_eq!(parse_name("20261022-091503-a-b-2.scraft"), Some((1_792_660_503, "a-b-2")));
        for name in [
            "Session.scraft",
            "20261022-091503.scraft",
            "20261022-091503-.scraft",
            "20261022_091503-x.scraft",
            "20261022-091503-x.wav",
            "20261022-091503-x.scraft.tmp",
            "20261322-091503-x.scraft",
            "20260230-091503-x.scraft",
            "20261022-246003-x.scraft",
            "2026102a-091503-x.scraft",
            "+0261022-091503-x.scraft",
            "20261022-091503x.scraft",
            "é0261022-091503-x.scraft",
            "日本語日本語日本語日本語日本語日本語.scraft",
            "",
            ".scraft",
        ] {
            assert_eq!(parse_name(name), None, "{name:?}");
        }
    }

    use soundcraft_audio_io::AudioBuffer;
    use soundcraft_model::{ChannelFormat, Clip, TrackKind};

    /// A session with a second of audio on one track, saved as `<base>/s/Song.scraft`; returns the
    /// engine and `<base>/s`, so that `<base>` holds nothing but the session folder.
    fn saved_engine(name: &str) -> (Engine, PathBuf) {
        let base = std::env::temp_dir().join(format!("sc-checkpoint-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let dir = base.join("s");
        std::fs::create_dir_all(&dir).unwrap();
        let mut e = Engine::default();
        let tone: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let s = e.session_mut();
        let src =
            crate::io::add_source(s, "tone", AudioBuffer { sample_rate: 48_000, channels: vec![tone] }, None, soundcraft_audio_io::FileFormat::Wav);
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, Some("Tone"));
        let cid = s.new_clip_id();
        crate::edit::place_clip(s, t, Clip::audio(cid, "tone", src, 0, 0, 48_000));
        e.execute("session.save_as", &json!({"path": dir.join("Song.scraft").to_string_lossy()})).unwrap();
        (e, dir)
    }

    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir.parent().unwrap_or(dir));
    }

    fn listing(e: &mut Engine) -> Vec<(String, String, String)> {
        let v = e.execute("session.checkpoints", &json!({})).unwrap();
        v.as_array()
            .unwrap()
            .iter()
            .map(|c| (c["path"].as_str().unwrap().to_string(), c["label"].as_str().unwrap().to_string(), c["created"].as_str().unwrap().to_string()))
            .collect()
    }

    fn files(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> =
            std::fs::read_dir(dir).map(|d| d.flatten().filter_map(|f| f.file_name().into_string().ok()).collect()).unwrap_or_default();
        v.sort();
        v
    }

    #[test]
    fn a_session_that_was_never_saved_cannot_be_checkpointed() {
        let mut e = Engine::default();
        let r = e.execute("session.checkpoint", &json!({"label": "x"}));
        assert!(matches!(r, Err(EngineError::Disabled(..))), "{r:?}");
        assert_eq!(e.execute("session.checkpoints", &json!({})).unwrap(), json!([]));
        // Called past the enablement check, it still says what to do.
        let err = checkpoint(&mut e, &json!({})).unwrap_err().to_string();
        assert!(err.contains("save the session first"), "{err}");
    }

    #[test]
    fn checkpoints_are_copies_beside_the_session_listed_newest_first() {
        let (mut e, dir) = saved_engine("copies");
        let (saved_path, tracks) = (e.path.clone(), e.session().tracks.len());
        e.execute("track.new", &json!({"kind": "midi", "name": "Added after save"})).unwrap();
        assert!(e.is_dirty());
        let undo = e.undo_history().len();

        let first = e.execute("session.checkpoint", &json!({"label": "v1"})).unwrap();
        e.execute("track.new", &json!({"kind": "midi", "name": "Later"})).unwrap();
        let second = e.execute("session.checkpoint", &json!({"label": "v1"})).unwrap();
        let third = e.execute("session.checkpoint", &json!({"label": "Before the chorus!"})).unwrap();
        let fourth = e.execute("session.checkpoint", &json!({})).unwrap();

        // The open session did not change: same file, still unsaved, and nothing new to undo.
        assert_eq!(e.path, saved_path);
        assert!(e.is_dirty());
        assert_eq!(e.undo_history().len(), undo + 1, "only the track.new made after the first checkpoint");

        // Copies sit in Checkpoints/ next to Song.scraft, with audio beside them.
        let cp = dir.join("Checkpoints");
        let names = files(&cp);
        assert_eq!(names.iter().filter(|n| n.ends_with(".scraft")).count(), 4, "{names:?}");
        assert!(names.iter().any(|n| n == "Audio Files"), "{names:?}");
        assert!(first["audio_files_written"].as_u64().unwrap() >= 1, "{first}");
        assert_eq!(second["audio_files_written"], 0, "audio is shared, not copied again: {second}");
        assert!(dir.join("Song.scraft").is_file());
        assert_eq!(files(&dir).iter().filter(|n| n.ends_with(".scraft")).count(), 1, "no copy beside the session itself");
        for r in [&first, &second, &third, &fourth] {
            let path = PathBuf::from(r["path"].as_str().unwrap());
            assert!(path.is_absolute() && path.starts_with(&cp), "{path:?}");
            assert!(path.is_file(), "{path:?}");
            assert_eq!(r["created"].as_str().unwrap().len(), 20);
        }
        assert_eq!(
            (first["label"].as_str(), third["label"].as_str(), fourth["label"].as_str()),
            (Some("v1"), Some("before-the-chorus"), Some("checkpoint"))
        );

        // Newest first, even for checkpoints made within the same second.
        let list = listing(&mut e);
        let paths: Vec<&str> = list.iter().map(|c| c.0.as_str()).collect();
        let made: Vec<&str> = [&fourth, &third, &second, &first].iter().map(|r| r["path"].as_str().unwrap()).collect();
        assert_eq!(paths, made);
        assert_eq!(list.iter().map(|c| c.1.as_str()).collect::<Vec<_>>(), vec!["checkpoint", "before-the-chorus", "v1", "v1"]);
        let created: Vec<&str> = list.iter().map(|c| c.2.as_str()).collect();
        assert!(created.windows(2).all(|w| w[0] > w[1]), "strictly newer each time: {created:?}");

        // Each opens as the session was when it was made.
        let open = |path: &str| {
            let mut o = Engine::default();
            o.execute("session.open", &json!({"path": path})).unwrap();
            o
        };
        let names = |o: &Engine| o.session().tracks.iter().map(|t| t.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(&open(first["path"].as_str().unwrap())), vec!["Tone", "Added after save"]);
        assert_eq!(names(&open(second["path"].as_str().unwrap())).len(), tracks + 2);
        let o = open(fourth["path"].as_str().unwrap());
        assert!(o.session().sources.iter().all(|s| o.session().pool.contains(s.id)), "the audio came with it");
        cleanup(&dir);
    }

    #[test]
    fn hostile_labels_stay_inside_the_checkpoints_folder() {
        let (mut e, dir) = saved_engine("hostile");
        let base = dir.parent().unwrap().to_path_buf();
        for label in ["../../x", "..\\..\\evil", "/tmp/abs", "C:\\temp\\x", "a\u{0}b", "\n", &"long ".repeat(500), "ß/../é"] {
            let r = e.execute("session.checkpoint", &json!({"label": label})).unwrap();
            let path = PathBuf::from(r["path"].as_str().unwrap());
            assert_eq!(path.parent(), Some(dir.join("Checkpoints").as_path()), "{label:?} -> {path:?}");
            assert!(path.is_file());
            assert!(path.file_name().unwrap().len() < 100, "{path:?}");
        }
        assert_eq!(files(&base), vec!["s"], "nothing was written outside the session folder");
        assert_eq!(
            files(&dir).iter().filter(|n| n.as_str() != "Checkpoints" && n.as_str() != "Audio Files" && n.as_str() != "Song.scraft").count(),
            0,
            "{:?}",
            files(&dir)
        );
        for bad in [json!({"label": 5}), json!({"label": ["x"]}), json!({"label": {"a": 1}}), json!({"label": true})] {
            assert!(matches!(e.execute("session.checkpoint", &bad), Err(EngineError::BadParams(..))), "{bad}");
        }
        let list = listing(&mut e);
        assert_eq!(list.len(), 8);
        assert_eq!(list[7].1, "x");
        cleanup(&dir);
    }

    #[test]
    fn the_listing_ignores_files_that_are_not_checkpoints_and_survives_a_clock_set_back() {
        let (mut e, dir) = saved_engine("listing");
        let cp = dir.join("Checkpoints");
        assert_eq!(listing(&mut e), Vec::<(String, String, String)>::new(), "no folder yet");
        std::fs::create_dir_all(cp.join("20250101-000000-folder.scraft")).unwrap();
        for stray in ["notes.txt", "Song.scraft", "20250101-000000-x.wav", "20251301-000000-bad-month.scraft", "20250101-000000-x.scraft.tmp"] {
            std::fs::write(cp.join(stray), "x").unwrap();
        }
        // A checkpoint from the future (the clock was set back since): the next one still sorts first.
        std::fs::write(cp.join("20991231-120000-from-the-future.scraft"), "{}").unwrap();
        std::fs::write(cp.join("20240229-000000-old.scraft"), "{}").unwrap();
        let list = listing(&mut e);
        assert_eq!(list.iter().map(|c| c.1.as_str()).collect::<Vec<_>>(), vec!["from-the-future", "old"]);
        assert_eq!(list[0].2, "2099-12-31T12:00:00Z");
        let r = e.execute("session.checkpoint", &json!({"label": "now"})).unwrap();
        assert_eq!(r["created"], "2099-12-31T12:00:01Z", "{r}");
        let list = listing(&mut e);
        assert_eq!(list.iter().map(|c| c.1.as_str()).collect::<Vec<_>>(), vec!["now", "from-the-future", "old"]);
        // The unreadable fake checkpoints are listed, not opened: listing never reads their contents.
        assert!(e.execute("session.open", &json!({"path": cp.join("20240229-000000-old.scraft").to_string_lossy()})).is_err());
        cleanup(&dir);
    }

    #[test]
    fn a_session_in_the_current_folder_checkpoints_there() {
        // A relative session path has no folder part: Checkpoints/ is relative too.
        assert_eq!(crate::io::checkpoint_dir(Path::new("Song.scraft")), PathBuf::from("Checkpoints"));
        assert_eq!(crate::io::checkpoint_dir(Path::new("/music/Song.scraft")), PathBuf::from("/music/Checkpoints"));
        assert_eq!(crate::io::checkpoint_dir(Path::new("")), PathBuf::from("Checkpoints"));
    }

    #[test]
    fn automation_roots_confine_checkpoints_like_every_other_write() {
        use crate::file_access::{AutomationRoots, confine};
        let (mut e, dir) = saved_engine("roots");
        let other = std::env::temp_dir().join(format!("sc-checkpoint-roots-other-{}", std::process::id()));
        std::fs::create_dir_all(&other).unwrap();
        let session_root = AutomationRoots::new(Some(&dir), Some(&dir)).unwrap();
        let elsewhere = AutomationRoots::new(Some(&other), Some(&other)).unwrap();
        let read_only = AutomationRoots::new(Some(&dir), None).unwrap();

        // The session's folder is the write root: allowed.
        let ok = confine(session_root.as_ref(), || e.execute("session.checkpoint", &json!({"label": "inside"})));
        assert!(ok.is_ok(), "{ok:?}");
        assert_eq!(confine(session_root.as_ref(), || listing(&mut e)).len(), 1);
        // Another folder, or read access only: refused, and nothing is created.
        let before = files(&dir.join("Checkpoints"));
        for roots in [&elsewhere, &read_only] {
            let r = confine(roots.as_ref(), || e.execute("session.checkpoint", &json!({"label": "outside"})));
            assert!(matches!(&r, Err(EngineError::Io(m)) if m.contains("automation")), "{r:?}");
            assert_eq!(files(&dir.join("Checkpoints")), before);
        }
        // A write-only root is enough to checkpoint, and names stay unique without reading the folder.
        let write_only = AutomationRoots::new(None, Some(&dir)).unwrap();
        let (a, b) = confine(write_only.as_ref(), || {
            (e.execute("session.checkpoint", &json!({"label": "w"})).unwrap(), e.execute("session.checkpoint", &json!({"label": "w"})).unwrap())
        });
        assert_ne!(a["path"], b["path"]);
        assert!(Path::new(a["path"].as_str().unwrap()).is_file() && Path::new(b["path"].as_str().unwrap()).is_file());
        // Listing needs the read root.
        let r = confine(elsewhere.as_ref(), || e.execute("session.checkpoints", &json!({})));
        assert!(matches!(&r, Err(EngineError::Io(m)) if m.contains("automation")), "{r:?}");
        let r = confine(AutomationRoots::new(None, Some(&dir)).unwrap().as_ref(), || e.execute("session.checkpoints", &json!({})));
        assert!(matches!(&r, Err(EngineError::Io(m)) if m.contains("automation")), "{r:?}");
        cleanup(&dir);
        let _ = std::fs::remove_dir_all(&other);
    }

    #[test]
    fn the_commands_are_registered_as_documented() {
        let c = find_command("session.checkpoint").unwrap();
        assert!(c.journal && !c.undoable && !c.view);
        let l = find_command("session.checkpoints").unwrap();
        assert!(!l.journal && !l.undoable);
    }
}
