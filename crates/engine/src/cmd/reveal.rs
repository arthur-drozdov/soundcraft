//! `view.reveal`: zoom and scroll the timeline so a region is in view (Agent Studio S6), for an
//! agent that edits somewhere the person is not looking. A view command: it touches only the
//! saved zoom state, so it leaves no undo step and no journal entry.

use super::view::{MAX_SPP, MIN_SPP};
use super::*;
use crate::cmd;
use serde_json::json;
use soundcraft_time::to_samples;

/// A region shorter than this (seconds) is shown as if it were this long, so a click-sized region
/// does not zoom to the sample level.
const MIN_SPAN_SECS: f64 = 0.1;

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        view "view.reveal",
        "Reveal Region",
        [],
        None,
        "{start, end?, track?, width_px?: 1200, margin?: 0.1} — zoom and scroll so start..end is in view with `margin` (a fraction of its length) free on each side; no `end` only scrolls, keeping the zoom. `track` is checked and returned (the app also scrolls its row into view) → {samples_per_px, scroll, start, end, track}",
        always,
        reveal
    )]
}

fn reveal(e: &mut Engine, p: &Value) -> Result<Value> {
    let id = "view.reveal";
    let start = position_param(e, id, p, "start")?.ok_or_else(|| bad(id, "`start` required"))?;
    let end = position_param(e, id, p, "end")?;
    if end.is_some_and(|end| end < start) {
        return Err(bad(id, "`end` must not be before `start`"));
    }
    let track = match p.get("track") {
        None | Some(Value::Null) => None,
        Some(_) => track_param(e, id, p, "track")?,
    };
    let width = f64_or(p, "width_px", 1200.0).clamp(100.0, 100_000.0);
    let margin = f64_or(p, "margin", 0.1).clamp(0.0, 0.45);
    let sr = e.session().sample_rate.as_f64();

    // The timeline starts at 0: a region before it is shown from there.
    let start = start.max(0);
    let end = end.map(|end| end.max(start));
    let len = end.map_or(0.0, |end| end.saturating_sub(start) as f64);
    let z = e.zoom_mut();
    if end.is_some() && len > 0.0 {
        let shown = len.max(MIN_SPAN_SECS * sr);
        z.samples_per_px = (shown * (1.0 + 2.0 * margin) / width).clamp(MIN_SPP, MAX_SPP);
    }
    // Centre the region: the same margin on both sides, and at the left edge when it starts near 0.
    let visible = z.samples_per_px * width;
    z.scroll = to_samples(start as f64 + len / 2.0 - visible / 2.0).clamp(0, MAX_POSITION);
    Ok(json!({"samples_per_px": z.samples_per_px, "scroll": z.scroll, "start": start, "end": end.unwrap_or(start), "track": track.map(|t| t.0)}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> Engine {
        let mut e = Engine::default();
        e.execute("track.new", &json!({"count": 3})).unwrap();
        e.execute("view.zoom_set", &json!({"samples_per_px": 500.0})).unwrap();
        e.execute("view.scroll", &json!({"to": 1000})).unwrap();
        e
    }

    fn view(e: &Engine) -> (f64, Samples) {
        (e.session().edit.zoom.samples_per_px, e.session().edit.zoom.scroll)
    }

    #[test]
    fn the_region_fills_the_window_with_a_margin_on_each_side() {
        let mut e = engine();
        let steps = e.undo_history().len();
        let r = e.execute("view.reveal", &json!({"start": 480_000, "end": 960_000, "width_px": 1000})).unwrap();
        // 480_000 samples plus 10% each side over 1000 px.
        assert_eq!(r["samples_per_px"], 576.0, "{r}");
        assert_eq!(r["scroll"], 480_000 - 48_000);
        assert_eq!(view(&e), (576.0, 432_000));
        assert_eq!(r["track"], Value::Null);
        // The region is inside the window with room on both sides.
        let (spp, scroll) = view(&e);
        assert!(scroll < 480_000 && scroll as f64 + spp * 1000.0 > 960_000.0);
        assert_eq!(e.undo_history().len(), steps, "a view setting leaves no undo step");
        // Seconds and a smaller margin.
        let r = e.execute("view.reveal", &json!({"start": {"seconds": 10.0}, "end": {"seconds": 12.0}, "width_px": 960, "margin": 0.0})).unwrap();
        assert_eq!(r["samples_per_px"], 100.0, "{r}");
        assert_eq!(r["scroll"], 480_000);
    }

    #[test]
    fn near_the_start_it_stops_at_the_left_edge_and_a_point_only_scrolls() {
        let mut e = engine();
        let r = e.execute("view.reveal", &json!({"start": 0, "end": 4800, "width_px": 1000})).unwrap();
        assert_eq!(r["scroll"], 0, "{r}");
        // 4800 samples is under 0.1 s, so it is shown as 0.1 s (4800 here too): never a sample-level zoom.
        let r = e.execute("view.reveal", &json!({"start": 2400, "end": 2410, "width_px": 1000})).unwrap();
        assert_eq!(r["samples_per_px"], 4800.0 * 1.2 / 1000.0, "{r}");
        let before = view(&e).0;
        let r = e.execute("view.reveal", &json!({"start": 1_000_000, "width_px": 1000})).unwrap();
        assert_eq!(view(&e).0, before, "no `end` keeps the zoom");
        assert_eq!(r["scroll"].as_i64(), Some(1_000_000 - (before * 500.0) as i64));
        assert_eq!(r["end"], 1_000_000);
    }

    #[test]
    fn a_track_is_checked_and_a_bad_call_changes_nothing() {
        let mut e = engine();
        let before = view(&e);
        let steps = e.undo_history().len();
        let ok = e.execute("view.reveal", &json!({"track": "Audio 2", "start": 5000, "end": 90_000})).unwrap();
        assert_eq!(ok["track"].as_u64(), e.session().track_by_name("Audio 2").map(|t| t.id.0));
        for bad in [
            json!({"track": "Nope", "start": 5000, "end": 90_000}),
            json!({"track": 424242, "start": 5000, "end": 90_000}),
            json!({"track": true, "start": 5000, "end": 90_000}),
            json!({"end": 90_000}),
            json!({"start": "tomorrow", "end": 5}),
            json!({"start": 100, "end": 50}),
        ] {
            e.execute("view.zoom_set", &json!({"samples_per_px": 500.0})).unwrap();
            e.execute("view.scroll", &json!({"to": 1000})).unwrap();
            assert!(matches!(e.execute("view.reveal", &bad), Err(EngineError::BadParams(..))), "{bad}");
            assert_eq!(view(&e), before, "{bad} moved the view");
        }
        assert_eq!(e.undo_history().len(), steps);
    }

    #[test]
    fn hostile_numbers_keep_the_view_in_range() {
        let mut e = engine();
        for p in [
            json!({"start": i64::MIN, "end": i64::MAX}),
            json!({"start": -5, "end": -1}),
            json!({"start": i64::MAX, "end": i64::MAX}),
            json!({"start": 0, "end": 1e300, "width_px": -4, "margin": 1e300}),
            json!({"start": 10, "end": 20, "width_px": "wide", "margin": "big"}),
            json!({"start": 10, "end": 20, "width_px": 1e300, "margin": -1e300}),
            json!({"start": 7, "end": 7}),
        ] {
            let r = e.execute("view.reveal", &p);
            assert!(r.is_ok(), "{p}: {r:?}");
            let (spp, scroll) = view(&e);
            assert!((MIN_SPP..=MAX_SPP).contains(&spp) && spp.is_finite(), "{p}: {spp}");
            assert!((0..=MAX_POSITION).contains(&scroll), "{p}: {scroll}");
        }
    }

    #[test]
    fn it_is_a_view_command() {
        let spec = find_command("view.reveal").unwrap();
        assert!(spec.view && !spec.undoable);
    }
}
