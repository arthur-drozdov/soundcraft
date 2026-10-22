use proptest::prelude::*;
use soundcraft_dsp::analysis::{self, Key, MAX_ANALYZE_SECS};
use std::f32::consts::TAU;

const SR: f32 = 48_000.0;

fn sine(freq: f32, amp: f32, secs: f32, sr: f32) -> Vec<f32> {
    (0..(secs * sr) as usize).map(|i| (f64::from(amp) * (std::f64::consts::TAU * f64::from(freq) * i as f64 / f64::from(sr)).sin()) as f32).collect()
}

/// A decaying 180 Hz burst (with a noisy attack) every `60 / bpm` seconds.
fn clicks(bpm: f32, secs: f32) -> Vec<f32> {
    let mut x = vec![0.0f32; (secs * SR) as usize];
    let mut seed = 12345u32;
    let beat = 60.0 / bpm;
    let mut t = 0.1f32;
    while t < secs - 0.2 {
        let start = (t * SR).round() as usize;
        for i in 0..(0.12 * SR) as usize {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let n = seed as f32 / u32::MAX as f32 * 2.0 - 1.0;
            let env = (-(i as f32) / SR / 0.025).exp();
            let v = env * (0.6 * (TAU * 180.0 * i as f32 / SR).sin() + 0.2 * n);
            if let Some(o) = x.get_mut(start + i) {
                *o += v;
            }
        }
        t += beat;
    }
    x
}

/// Sum of sines at the given MIDI notes, each with three harmonics.
fn chord(notes: &[f32], secs: f32) -> Vec<f32> {
    let mut x = vec![0.0f32; (secs * SR) as usize];
    for &n in notes {
        let f = 440.0 * 2f32.powf((n - 69.0) / 12.0);
        for (h, a) in [(1.0, 0.3), (2.0, 0.15), (3.0, 0.08)] {
            for (i, o) in x.iter_mut().enumerate() {
                *o += (a * (std::f64::consts::TAU * f64::from(f * h) * i as f64 / f64::from(SR)).sin()) as f32;
            }
        }
    }
    x
}

/// Several chords one after the other.
fn progression(chords: &[&[f32]], each: f32) -> Vec<f32> {
    chords.iter().flat_map(|c| chord(c, each)).collect()
}

#[test]
fn full_scale_997hz_sine_is_minus_3_01_lufs() {
    let a = analysis::analyze(&[sine(997.0, 1.0, 10.0, SR)], SR, 0.5);
    assert!((a.lufs + 3.01).abs() < 0.1, "lufs {}", a.lufs);
    assert!(a.peak_dbfs.abs() < 0.01, "peak {}", a.peak_dbfs);
    assert!((a.rms_dbfs + 3.01).abs() < 0.05, "rms {}", a.rms_dbfs);
    assert!(a.true_peak_dbfs >= a.peak_dbfs - 1e-3);
    assert!(a.clipped_samples > 0, "a full-scale sine touches the clip threshold");
    assert!(a.dc_offset.abs() < 1e-3);
    assert!(!a.truncated);
    assert!((a.duration_s - 10.0).abs() < 1e-6);
}

#[test]
fn levels_of_a_half_scale_sine() {
    let a = analysis::analyze(&[sine(1000.0, 0.5, 4.0, SR)], SR, 0.5);
    assert!((a.peak_dbfs + 6.02).abs() < 0.05, "{}", a.peak_dbfs);
    assert!((a.rms_dbfs + 9.03).abs() < 0.05, "{}", a.rms_dbfs);
    assert_eq!(a.clipped_samples, 0);
}

#[test]
fn bounce_meter_agrees_on_stereo() {
    let l = sine(440.0, 0.5, 6.0, SR);
    let r = sine(660.0, 0.25, 6.0, SR);
    let a = analysis::analyze(&[l.clone(), r.clone()], SR, 0.5);
    let mut m = soundcraft_dsp::meter::LoudnessMeter::new(SR, 2);
    m.process(&[l, r], 6 * 48_000);
    assert_eq!(a.lufs, m.integrated_lufs());
    assert_eq!(a.channels, 2);
}

#[test]
fn tempo_of_click_tracks() {
    for bpm in [90.0f32, 100.0, 120.0, 140.0] {
        let a = analysis::analyze(&[clicks(bpm, 20.0)], SR, 0.5);
        let est = a.bpm.unwrap_or(0.0);
        assert!((est - bpm).abs() <= 1.0, "{bpm} BPM estimated as {est} (confidence {})", a.bpm_confidence);
        assert!(a.bpm_confidence > 0.3, "{bpm}: confidence {}", a.bpm_confidence);
        assert!(a.onset_count >= 10, "{bpm}: {} onsets", a.onset_count);
    }
}

#[test]
fn tempo_is_folded_into_80_to_160() {
    // 60 BPM clicks read 120; 180 BPM clicks read 90.
    let a = analysis::analyze(&[clicks(60.0, 24.0)], SR, 0.5);
    assert!((a.bpm.unwrap_or(0.0) - 120.0).abs() <= 1.5, "{:?}", a.bpm);
    let a = analysis::analyze(&[clicks(180.0, 20.0)], SR, 0.5);
    assert!((a.bpm.unwrap_or(0.0) - 90.0).abs() <= 1.5, "{:?}", a.bpm);
}

#[test]
fn onsets_of_a_click_track_are_on_the_beats() {
    let a = analysis::analyze(&[clicks(120.0, 10.0)], SR, 0.5);
    assert!(a.onsets_s.len() >= 18, "{} onsets", a.onsets_s.len());
    for t in &a.onsets_s {
        let off = (t - 0.1).rem_euclid(0.5);
        assert!(off.min(0.5 - off) < 0.03, "onset at {t} is off the beat grid");
    }
}

#[test]
fn a_steady_tone_has_no_tempo() {
    let a = analysis::analyze(&[sine(220.0, 0.5, 10.0, SR)], SR, 0.5);
    assert_eq!(a.bpm, None);
    assert_eq!(a.bpm_confidence, 0.0);
}

#[test]
fn noise_has_a_low_tempo_confidence() {
    let mut seed = 99u32;
    let n: Vec<f32> = (0..(12.0 * SR) as usize)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32 * 2.0 - 1.0) * 0.3
        })
        .collect();
    let a = analysis::analyze(&[n], SR, 0.5);
    assert!(a.bpm_confidence < 0.3, "noise confidence {}", a.bpm_confidence);
    assert!(a.key_confidence < 0.3, "noise key confidence {}", a.key_confidence);
}

#[test]
fn a_minor_triads_are_a_minor() {
    // A3 C4 E4 held, then Dm (D4 F4 A4), Am again, E (E4 G#4 B4), Am.
    let am: &[f32] = &[57.0, 60.0, 64.0];
    let dm: &[f32] = &[62.0, 65.0, 69.0];
    let e: &[f32] = &[64.0, 68.0, 71.0];
    let x = progression(&[am, am, dm, am, e, am, am], 1.0);
    let a = analysis::analyze(&[x], SR, 0.5);
    assert_eq!(a.key.map(|k| k.name()).as_deref(), Some("A minor"), "confidence {}", a.key_confidence);
    assert!(a.key_confidence > 0.3, "confidence {}", a.key_confidence);
    // A single chord alone is already enough.
    let a = analysis::analyze(&[chord(am, 3.0)], SR, 0.5);
    assert_eq!(a.key.map(|k| k.name()).as_deref(), Some("A minor"));
}

#[test]
fn c_major_and_other_keys_are_found() {
    let c: &[f32] = &[60.0, 64.0, 67.0];
    let f: &[f32] = &[65.0, 69.0, 72.0];
    let g: &[f32] = &[67.0, 71.0, 74.0];
    let a = analysis::analyze(&[progression(&[c, f, g, c, c], 1.0)], SR, 0.5);
    assert_eq!(a.key.map(|k| k.name()).as_deref(), Some("C major"));
    // Transposed up a tone: D major (D F# A, G B D, A C# E).
    let d: &[f32] = &[62.0, 66.0, 69.0];
    let g2: &[f32] = &[67.0, 71.0, 74.0];
    let a2: &[f32] = &[69.0, 73.0, 76.0];
    let a = analysis::analyze(&[progression(&[d, g2, a2, d, d], 1.0)], SR, 0.5);
    assert_eq!(a.key, Some(Key { tonic: 2, minor: false }));
    assert_eq!(Key { tonic: 6, minor: true }.name(), "F# minor");
}

#[test]
fn centroid_of_a_sine_is_its_frequency() {
    for f in [220.0f32, 1000.0, 4000.0] {
        let a = analysis::analyze(&[sine(f, 0.5, 3.0, SR)], SR, 0.5);
        assert!((a.centroid_hz - f).abs() < f * 0.02, "{f} Hz read {}", a.centroid_hz);
    }
}

#[test]
fn clipping_and_dc_are_counted() {
    let mut x = sine(100.0, 0.3, 2.0, SR);
    for v in x.iter_mut().take(10) {
        *v = 1.0;
    }
    for v in x.iter_mut().skip(100).take(5) {
        *v = -1.0;
    }
    let y: Vec<f32> = sine(100.0, 0.3, 2.0, SR).iter().map(|v| v + 0.1).collect();
    let a = analysis::analyze(&[x, y], SR, 0.5);
    assert_eq!(a.clipped_samples, 15);
    assert!((a.dc_offset - 0.1).abs() < 1e-3, "dc {}", a.dc_offset);
    let neg: Vec<f32> = sine(100.0, 0.3, 2.0, SR).iter().map(|v| v - 0.2).collect();
    let a = analysis::analyze(&[neg], SR, 0.5);
    assert!((a.dc_offset + 0.2).abs() < 1e-3, "dc {}", a.dc_offset);
}

#[test]
fn silence_empty_and_tiny_input_do_not_panic() {
    for x in [vec![], vec![0.0f32], vec![0.5f32], vec![0.0; 100], vec![0.0; 48_000 * 5]] {
        let a = analysis::analyze(std::slice::from_ref(&x), SR, 0.5);
        assert_eq!(a.bpm, None);
        assert_eq!(a.key, None);
        assert_eq!(a.centroid_hz, 0.0);
        assert!(a.peak_dbfs.is_finite() && a.lufs.is_finite() && a.rms_dbfs.is_finite());
        assert_eq!(a.bpm_confidence, 0.0);
        assert_eq!(a.key_confidence, 0.0);
    }
    let a = analysis::analyze(&[], SR, 0.5);
    assert_eq!(a.frames, 0);
    assert_eq!(a.peak_dbfs, -144.0);
    let a = analysis::analyze(&[Vec::new(), Vec::new()], SR, 0.5);
    assert_eq!(a.frames, 0);
}

#[test]
fn hostile_rates_and_values_are_sanitised() {
    let x = sine(440.0, 0.5, 4.0, SR);
    for sr in [0.0, -1.0, f32::NAN, f32::INFINITY, 1.0, 1e12] {
        let a = analysis::analyze(std::slice::from_ref(&x), sr, f32::NAN);
        assert!(a.centroid_hz.is_finite());
    }
    let mut bad = x.clone();
    for (i, v) in bad.iter_mut().enumerate().step_by(97) {
        *v = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][i % 3];
    }
    let a = analysis::analyze(&[bad], SR, 2.0);
    assert!(a.peak_dbfs.is_finite() && a.peak_dbfs <= 0.0, "{}", a.peak_dbfs);
    assert!(a.rms_dbfs.is_finite() && a.lufs.is_finite() && a.dc_offset.is_finite() && a.centroid_hz.is_finite());
    // Ragged channels and a huge channel count.
    let a = analysis::analyze(&[x.clone(), vec![0.1; 10]], SR, 0.5);
    assert!(a.peak_dbfs.is_finite());
    let many: Vec<Vec<f32>> = (0..200).map(|_| vec![0.1; 4096]).collect();
    let a = analysis::analyze(&many, SR, 0.5);
    assert_eq!(a.channels, 64);
}

#[test]
fn input_beyond_the_cap_is_truncated() {
    // At 1 kHz the cap is 600 000 frames.
    let x = vec![0.1f32; 650_000];
    assert_eq!(analysis::max_frames(1000.0), 600_000);
    let a = analysis::analyze(&[x], 1000.0, 0.5);
    assert!(a.truncated);
    assert_eq!(a.frames, 600_000);
    assert!((a.duration_s - f64::from(MAX_ANALYZE_SECS)).abs() < 1e-9);
}

#[test]
fn building_blocks_are_safe_on_nonsense() {
    assert_eq!(analysis::estimate_bpm(&[], 200.0), (None, 0.0));
    assert_eq!(analysis::estimate_bpm(&[1.0; 10], 200.0), (None, 0.0));
    assert_eq!(analysis::estimate_bpm(&[1.0; 5000], 200.0), (None, 0.0));
    assert_eq!(analysis::estimate_bpm(&[f32::NAN; 5000], 200.0), (None, 0.0));
    assert_eq!(analysis::estimate_bpm(&[1.0; 5000], f32::NAN), (None, 0.0));
    assert_eq!(analysis::estimate_bpm(&[1.0; 5000], 0.0), (None, 0.0));
    assert_eq!(analysis::estimate_bpm(&[1.0; 5000], 1e9), (None, 0.0), "an absurd envelope rate must not make a huge search");
    assert_eq!(analysis::estimate_key(&[0.0; 12]), (None, 0.0));
    assert_eq!(analysis::estimate_key(&[1.0; 12]), (None, 0.0));
    assert_eq!(analysis::estimate_key(&[f32::NAN; 12]), (None, 0.0));
    assert_eq!(analysis::rms_db(&[]), -144.0);
    assert_eq!(analysis::rms_db(&[vec![f32::NAN; 4]]), -144.0);
    assert!(analysis::mono_mix(&[]).is_empty());
    assert_eq!(analysis::mono_mix(&[vec![1.0, 3.0], vec![3.0]]), vec![2.0, 1.5]);
    assert_eq!(analysis::onset_envelope(&[0.0; 10], 48_000.0).0.len(), 0);
    assert_eq!(analysis::spectral_centroid(&[], 48_000.0), 0.0);
    assert_eq!(analysis::chroma(&[], 48_000.0), [0.0; 12]);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn random_buffers_never_panic_and_stay_finite(
        data in proptest::collection::vec(prop_oneof![
            8 => -2.0f32..2.0,
            1 => Just(f32::NAN),
            1 => Just(f32::INFINITY),
            1 => Just(f32::NEG_INFINITY),
            1 => Just(0.0f32),
        ], 0..20_000),
        second in proptest::collection::vec(-1.0f32..1.0, 0..5_000),
        sr in prop_oneof![Just(8000.0f32), Just(44_100.0), Just(48_000.0), Just(f32::NAN), Just(-5.0), Just(2_000_000.0)],
        sens in prop_oneof![Just(0.0f32), Just(0.5), Just(1.0), Just(f32::NAN), Just(-3.0)],
    ) {
        let a = analysis::analyze(&[data, second], sr, sens);
        prop_assert!(a.peak_dbfs.is_finite() && a.true_peak_dbfs.is_finite());
        prop_assert!(a.rms_dbfs.is_finite() && a.lufs.is_finite());
        prop_assert!(a.centroid_hz.is_finite() && a.centroid_hz >= 0.0);
        prop_assert!(a.dc_offset.is_finite());
        prop_assert!((0.0..=1.0).contains(&a.bpm_confidence));
        prop_assert!((0.0..=1.0).contains(&a.key_confidence));
        if let Some(b) = a.bpm {
            prop_assert!(b.is_finite() && (79.0..=161.0).contains(&b));
        }
        prop_assert!(a.onsets_s.iter().all(|t| t.is_finite() && *t >= 0.0));
    }
}

/// Kick on every beat, a quieter noisy hat on the off-beats, at `bpm`.
fn drum_pattern(bpm: f32, secs: f32) -> Vec<f32> {
    let mut x = vec![0.0f32; (secs * SR) as usize];
    let mut seed = 777u32;
    let beat = 60.0 / bpm;
    let mut k = 0u32;
    loop {
        let t = 0.05 + k as f32 * beat / 2.0;
        if t > secs - 0.3 {
            break;
        }
        let start = (t * SR).round() as usize;
        let kick = k.is_multiple_of(2);
        for i in 0..(0.15 * SR) as usize {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let n = seed as f32 / u32::MAX as f32 * 2.0 - 1.0;
            let s = i as f32 / SR;
            let v = if kick { (-s / 0.05).exp() * (TAU * (60.0 + 90.0 * (-s / 0.02).exp()) * s).sin() * 0.8 } else { (-s / 0.01).exp() * n * 0.25 };
            if let Some(o) = x.get_mut(start + i) {
                *o += v;
            }
        }
        k += 1;
    }
    x
}

#[test]
fn tempo_of_a_kick_and_hat_pattern() {
    for bpm in [100.0f32, 128.0] {
        let a = analysis::analyze(&[drum_pattern(bpm, 24.0)], SR, 0.5);
        let est = a.bpm.unwrap_or(0.0);
        assert!((est - bpm).abs() <= 1.5, "{bpm} BPM estimated as {est} (confidence {})", a.bpm_confidence);
    }
}
