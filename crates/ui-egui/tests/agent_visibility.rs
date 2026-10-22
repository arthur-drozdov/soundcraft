//! What a person watching sees while an agent works over the control channel: the status bar
//! names the last command (`Agent: …`) for a few seconds, and `view.reveal` brings a region and
//! its track row into view.
// Integration tests: unwrapping and panicking on failure is fine here, unlike in shipped code (AGENTS.md).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use egui::{Pos2, RawInput, Rect, pos2, vec2};
use serde_json::{Value, json};
use soundcraft_ui_egui::{AgentStatus, ControlRequest, Services, SoundApp};
use std::sync::mpsc::{Sender, channel};

struct Rig {
    ctx: egui::Context,
    app: SoundApp,
    tx: Sender<ControlRequest>,
    time: f64,
}

impl Rig {
    fn new() -> Rig {
        let (tx, rx) = channel();
        let app = SoundApp::new(soundcraft_engine::demo::demo_engine(), None, Services::default()).with_control(rx);
        let mut r = Rig { ctx: egui::Context::default(), app, tx, time: 1.0 };
        r.frames(6);
        r
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.time += 1.0 / 60.0;
            let mut raw =
                RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1400.0, 900.0))), time: Some(self.time), ..Default::default() };
            self.app.raw_input_hook(&mut raw);
            self.ctx
                .run_ui(raw, |ui| {
                    self.app.logic(ui.ctx());
                    self.app.ui(ui);
                })
                .textures_delta
                .clear();
        }
    }

    /// One control request; returns the whole reply.
    fn request(&mut self, method: &str, params: Value) -> Value {
        let (req, reply) = ControlRequest::new(method, params);
        self.tx.send(req).unwrap();
        self.frames(40);
        reply.try_recv().unwrap()
    }

    fn execute(&mut self, command: &str, params: Value) -> Value {
        self.request("engine.execute", json!({"command": command, "params": params}))
    }

    fn agent(&self) -> Option<String> {
        self.app.agent_status.as_ref().map(AgentStatus::text)
    }

    fn timeline(&self) -> Rect {
        let [x0, y0, x1, y1] = self.app.edit_layout.timeline;
        Rect::from_min_max(pos2(x0, y0), pos2(x1, y1))
    }

    fn row(&self, track: u64) -> Rect {
        let r = self.app.edit_layout.rows.iter().find(|(id, _)| *id == track).unwrap().1;
        Rect::from_min_max(pos2(r[0], r[1]), pos2(r[2], r[3]))
    }
}

#[test]
fn the_status_bar_names_the_last_command_an_agent_ran() {
    let mut r = Rig::new();
    assert_eq!(r.agent(), None);
    assert_eq!(r.execute("track.new", json!({"name": "Pad", "kind": "instrument"}))["ok"], true);
    assert_eq!(r.agent().as_deref(), Some("Agent: New"));
    let ctx = r.ctx.clone();
    let shown = r.app.inspect(&ctx)["agent_status"].clone();
    assert_eq!(shown["text"], "Agent: New", "{shown}");
    assert_eq!(shown["failed"], false);
    assert_eq!(shown["command"], "track.new");

    // Reads leave it alone, even when the agent polls.
    for q in ["session.inspect", "audio.analyze", "midi.notes", "engine.commands"] {
        r.execute(q, json!({}));
        assert_eq!(r.agent().as_deref(), Some("Agent: New"), "{q}");
    }
    r.request("session.inspect", json!({}));
    r.request("ui.inspect", json!({}));
    assert_eq!(r.agent().as_deref(), Some("Agent: New"));

    // A newer command replaces it, and a failed one says so.
    assert_eq!(r.execute("edit.select", json!({"tracks": ["Pad"]}))["ok"], true);
    let selected = r.agent().unwrap();
    assert!(selected.starts_with("Agent: ") && !selected.contains("failed"), "{selected}");
    assert_eq!(r.execute("track.new", json!({"kind": "no-such-kind"}))["ok"], false);
    assert_eq!(r.agent().as_deref(), Some("Agent: New (failed)"));
    assert_eq!(r.app.inspect(&ctx)["agent_status"]["failed"], true);

    // A person at the keyboard is not an agent.
    r.app.run("track.new", json!({"name": "By hand"})).unwrap();
    assert_eq!(r.agent().as_deref(), Some("Agent: New (failed)"));

    // Commands that are not engine commands (windows, dialogs) and unknown ones are not announced.
    r.app.agent_status = None;
    r.execute("window.about", json!({}));
    r.execute("no.such.command", json!({}));
    assert_eq!(r.agent(), None);
}

#[test]
fn the_status_clears_itself_after_ten_seconds() {
    let mut r = Rig::new();
    r.execute("edit.select", json!({"tracks": ["Kick"]}));
    let at = r.app.agent_status.as_ref().unwrap().at;
    r.frames(60);
    assert!(r.agent().is_some(), "still showing after about a second");
    r.time = at + AgentStatus::SHOW_SECS - 0.5;
    r.frames(2);
    assert!(r.agent().is_some());
    r.time = at + AgentStatus::SHOW_SECS + 0.5;
    r.frames(2);
    assert_eq!(r.agent(), None, "gone once it has been shown");
    let ctx = r.ctx.clone();
    assert_eq!(r.app.inspect(&ctx)["agent_status"], Value::Null);
    // The expiry arithmetic.
    let a = AgentStatus { command: "x.y".into(), label: "X".into(), failed: false, at: 5.0 };
    assert_eq!((a.remaining(5.0), a.remaining(12.5), a.remaining(15.0), a.remaining(99.0)), (10.0, 2.5, 0.0, -84.0));
    assert_eq!(a.remaining(1.0), 10.0, "a clock that went back does not extend it");
}

#[test]
fn labels_lose_their_ellipsis() {
    let mut r = Rig::new();
    r.app.note_agent_command("file.import_audio", true, 0.0);
    assert_eq!(r.agent().as_deref(), Some("Agent: Audio"));
}

#[test]
fn reveal_zooms_to_the_region_and_scrolls_to_the_track() {
    let mut r = Rig::new();
    r.execute("track.new", json!({"count": 40, "name": "Extra"}));
    r.frames(10);
    let tl = r.timeline();
    assert!(r.app.edit_layout.content_h > tl.height() * 2.0, "needs a long track list");
    let last = r.app.engine.session().tracks.iter().rev().find(|t| t.name.starts_with("Extra")).unwrap().id.0;
    assert!(r.row(last).min.y >= tl.max.y, "the last track starts off screen");

    let reply = r.execute("view.reveal", json!({"track": last, "start": 96_000, "end": 192_000}));
    assert_eq!(reply["ok"], true, "{reply}");
    let z = r.app.engine.session().edit.zoom;
    // 96_000 samples plus 10% each side across the timeline as drawn.
    assert!((z.samples_per_px * f64::from(tl.width()) - 96_000.0 * 1.2).abs() < 1.0, "{} over {} px", z.samples_per_px, tl.width());
    assert_eq!(z.scroll, 96_000 - 9_600);
    assert_eq!(reply["result"]["scroll"], z.scroll);
    assert!(r.app.edit_layout.scroll_y > 0.0);
    let row = r.row(last);
    assert!(row.min.y >= tl.min.y - 0.5 && row.max.y <= tl.max.y + 0.5, "{row:?} should be inside {tl:?}");
    assert_eq!(r.app.edit_layout.reveal_track, None, "done once scrolled");
    assert_eq!(r.agent().as_deref(), Some("Agent: Reveal Region"));

    // A row that is already in view does not move, and the way back up works too.
    let (first, scroll) = (r.app.engine.session().tracks[0].id.0, r.app.edit_layout.scroll_y);
    let visible = r.app.edit_layout.rows.iter().find(|(_, rect)| rect[1] >= tl.min.y && rect[3] <= tl.max.y).unwrap().0;
    r.execute("view.reveal", json!({"track": visible, "start": 0, "end": 48_000}));
    assert_eq!(r.app.edit_layout.scroll_y, scroll);
    r.execute("view.reveal", json!({"track": first, "start": 0, "end": 48_000}));
    assert_eq!(r.app.edit_layout.scroll_y, 0.0);
    assert_eq!(r.app.engine.session().edit.zoom.scroll, 0);

    // The view is not a document change: no undo step, and a bad track changes nothing.
    let (steps, view) = (r.app.engine.undo_history().len(), r.app.engine.session().edit.zoom);
    assert_eq!(r.execute("view.reveal", json!({"track": 987_654, "start": 5, "end": 50}))["ok"], false);
    assert_eq!(r.app.engine.session().edit.zoom, view);
    assert_eq!(r.app.engine.undo_history().len(), steps);
    assert_eq!(r.agent().as_deref(), Some("Agent: Reveal Region (failed)"));
    // A caller who says how wide the view is gets that width used.
    r.execute("view.reveal", json!({"start": 0, "end": 120_000, "width_px": 1000, "margin": 0.0}));
    assert_eq!(r.app.engine.session().edit.zoom.samples_per_px, 120.0);
}
