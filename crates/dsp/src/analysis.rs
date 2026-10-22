//! Whole-clip audio analysis for agents: levels, loudness, onsets, tempo, key and brightness.
//!
//! Everything here is an original implementation of textbook methods and works on planar `f32`
//! buffers (`ch[channel][frame]`). It allocates (it is for control-thread use, never the audio
//! thread) and never panics: sample rates are sanitised, non-finite samples count as silence,
//! the input is capped at [`MAX_ANALYZE_SECS`] and every estimate that cannot be made is `None`
//! (with confidence 0) instead of a guess.
//!
//! - Levels: sample peak, true peak, RMS over all channels, clipped-sample count, DC offset.
//! - Loudness: integrated LUFS from the same ITU-R BS.1770 meter the bounce reports.
//! - Onsets: the vibrato-suppressed spectral-flux detector of [`crate::offline::detect_transients`].
//! - Tempo: autocorrelation of a spectral-flux envelope, picked over 60..200 BPM with a mild
//!   prior and reported folded into 80..160 BPM (see [`estimate_bpm`]).
//! - Key: peak-picked chroma correlated with the Krumhansl-Schmuckler profiles in all 24 keys.
//! - Spectral centroid: magnitude-weighted mean frequency over the whole span.

use std::f64::consts::PI;

use rustfft::{FftPlanner, num_complex::Complex};

use crate::meter::LoudnessMeter;
use crate::util::{MAX_CHANNELS, sane_sr};
use crate::{MIN_DB, gain_to_db};

/// Longest span analysed, in seconds. [`analyze`] looks at no more than this and reports
/// `truncated`; callers that prefer an error check [`max_frames`] first.
pub const MAX_ANALYZE_SECS: f32 = 600.0;
/// Most onsets listed (the real number is `onset_count`).
pub const MAX_ONSETS: usize = 5000;
/// Magnitude at or above which a sample counts as clipped.
pub const CLIP_THRESHOLD: f32 = 0.999;
/// Tempo search range, BPM.
pub const BPM_MIN: f32 = 60.0;
pub const BPM_MAX: f32 = 200.0;
/// Reported tempo is folded into this range, BPM.
pub const BPM_FOLD_LO: f32 = 80.0;
pub const BPM_FOLD_HI: f32 = 160.0;
/// Shortest span a tempo is estimated for, seconds.
const MIN_BPM_SECS: f32 = 3.0;
/// Fewest onsets [`analyze`] wants before it reports a tempo.
const MIN_BPM_ONSETS: usize = 3;

const NOTE_NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
/// Krumhansl-Schmuckler key profiles (probe-tone ratings), tonic first.
const KS_MAJOR: [f64; 12] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KS_MINOR: [f64; 12] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

/// A musical key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Key {
    /// Pitch class of the tonic, 0 = C .. 11 = B.
    pub tonic: u8,
    pub minor: bool,
}

impl Key {
    /// `"A minor"`, `"F# major"` (sharps, like the Python audiokit).
    pub fn name(&self) -> String {
        format!("{} {}", self.tonic_name(), self.mode())
    }

    pub fn tonic_name(&self) -> &'static str {
        NOTE_NAMES.get(usize::from(self.tonic)).copied().unwrap_or("C")
    }

    pub fn mode(&self) -> &'static str {
        if self.minor { "minor" } else { "major" }
    }
}

/// Result of [`analyze`]. Levels are in dBFS (floored at -144), loudness in LUFS.
#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    /// Frames analysed (after the cap) and the channels used.
    pub frames: usize,
    pub channels: usize,
    pub duration_s: f64,
    /// The input was longer than [`MAX_ANALYZE_SECS`]; only the start was analysed.
    pub truncated: bool,
    pub peak_dbfs: f32,
    pub true_peak_dbfs: f32,
    pub rms_dbfs: f32,
    pub lufs: f32,
    /// Onset times in seconds (at most [`MAX_ONSETS`]) and the real count.
    pub onsets_s: Vec<f64>,
    pub onset_count: usize,
    pub bpm: Option<f32>,
    pub bpm_confidence: f32,
    pub key: Option<Key>,
    pub key_confidence: f32,
    pub centroid_hz: f32,
    /// Samples (all channels) with magnitude of at least [`CLIP_THRESHOLD`].
    pub clipped_samples: usize,
    /// Mean sample value of the channel with the largest absolute mean.
    pub dc_offset: f32,
}

impl Analysis {
    fn empty(channels: usize, truncated: bool) -> Self {
        Self {
            frames: 0,
            channels,
            duration_s: 0.0,
            truncated,
            peak_dbfs: MIN_DB,
            true_peak_dbfs: MIN_DB,
            rms_dbfs: MIN_DB,
            lufs: MIN_DB,
            onsets_s: Vec::new(),
            onset_count: 0,
            bpm: None,
            bpm_confidence: 0.0,
            key: None,
            key_confidence: 0.0,
            centroid_hz: 0.0,
            clipped_samples: 0,
            dc_offset: 0.0,
        }
    }
}

fn max_len(ch: &[Vec<f32>]) -> usize {
    ch.iter().take(MAX_CHANNELS).map(Vec::len).max().unwrap_or(0)
}

/// Most frames [`analyze`] looks at for a sample rate (10 minutes).
pub fn max_frames(sample_rate: f32) -> usize {
    (f64::from(MAX_ANALYZE_SECS) * f64::from(sane_sr(sample_rate))) as usize
}

fn finite(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Mean of all channels (at most [`MAX_CHANNELS`]) over the first `frames` frames; non-finite
/// samples count as 0 and a short channel reads as silence past its end.
fn mono_n(ch: &[Vec<f32>], frames: usize) -> Vec<f32> {
    let ch = ch.get(..ch.len().min(MAX_CHANNELS)).unwrap_or(&[]);
    if ch.is_empty() {
        return Vec::new();
    }
    let k = ch.len() as f32;
    (0..frames).map(|i| ch.iter().map(|c| finite(c.get(i).copied().unwrap_or(0.0))).sum::<f32>() / k).collect()
}

/// Channel average of the whole buffer (at most [`MAX_CHANNELS`] channels).
pub fn mono_mix(ch: &[Vec<f32>]) -> Vec<f32> {
    mono_n(ch, max_len(ch))
}

/// RMS of every sample of every channel, dBFS (-144 for silence or no audio). Non-finite samples
/// count as silence.
pub fn rms_db(ch: &[Vec<f32>]) -> f32 {
    let (mut sum, mut n) = (0.0f64, 0u64);
    for c in ch.iter().take(MAX_CHANNELS) {
        for &v in c {
            let v = f64::from(finite(v));
            sum += v * v;
        }
        n += c.len() as u64;
    }
    if n == 0 { MIN_DB } else { gain_to_db((sum / n as f64).sqrt() as f32) }
}

/// Slice of a one-channel buffer, empty when there is none.
fn slice_of(v: &[Vec<f32>]) -> &[f32] {
    v.first().map_or(&[], Vec::as_slice)
}

/// Hann window of length `n`.
fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()) as f32).collect()
}

/// Number of `n`-sample frames at `hop` over `len` samples; one zero-padded frame when shorter
/// than `n`, none for no audio.
fn frame_count(len: usize, n: usize, hop: usize) -> usize {
    if len == 0 {
        0
    } else if len < n {
        1
    } else {
        (len - n) / hop.max(1) + 1
    }
}

/// Fills `buf` with the windowed frame starting at `start`, zero-padded past the end.
fn load_frame(buf: &mut [Complex<f32>], mono: &[f32], start: usize, win: &[f32]) {
    for (i, (b, w)) in buf.iter_mut().zip(win).enumerate() {
        *b = Complex::new(finite(mono.get(start.saturating_add(i)).copied().unwrap_or(0.0)) * w, 0.0);
    }
}

/// Full analysis of planar audio. `sensitivity` 0..1 (default 0.5) is passed to the onset
/// detector. Looks at no more than [`MAX_ANALYZE_SECS`] and at most [`MAX_CHANNELS`] channels.
pub fn analyze(ch: &[Vec<f32>], sample_rate: f32, sensitivity: f32) -> Analysis {
    let sr = sane_sr(sample_rate);
    let nch = ch.len().min(MAX_CHANNELS);
    let full = max_len(ch);
    let frames = full.min(max_frames(sr));
    let truncated = full > frames;
    if frames == 0 || nch == 0 {
        return Analysis::empty(nch, truncated);
    }
    let ch = ch.get(..nch).unwrap_or(&[]);

    // Levels in one pass: peak, RMS energy, clipping, DC.
    let (mut peak, mut energy, mut clipped) = (0.0f32, 0.0f64, 0usize);
    let mut dc = 0.0f64;
    for c in ch {
        let mut sum = 0.0f64;
        for &v in c.iter().take(frames) {
            let v = finite(v);
            let a = v.abs();
            peak = peak.max(a);
            if a >= CLIP_THRESHOLD {
                clipped += 1;
            }
            let v = f64::from(v);
            energy += v * v;
            sum += v;
        }
        let mean = sum / frames as f64;
        if mean.abs() > dc.abs() {
            dc = mean;
        }
    }
    let rms = gain_to_db((energy / (frames as f64 * nch as f64)).sqrt() as f32);

    // Loudness and true peak from the BS.1770 meter.
    let mut meter = LoudnessMeter::new(sr, nch);
    meter.process(ch, frames);

    let mono = vec![mono_n(ch, frames)];
    let m = slice_of(&mono);

    let transients = crate::offline::detect_transients(&mono, sr, if sensitivity.is_finite() { sensitivity } else { 0.5 });
    let onset_count = transients.len();
    let onsets_s: Vec<f64> = transients.iter().take(MAX_ONSETS).map(|&i| i as f64 / f64::from(sr)).collect();

    let (mut bpm, mut bpm_confidence) = (None, 0.0);
    if onset_count >= MIN_BPM_ONSETS {
        let (env, rate) = onset_envelope(m, sr);
        (bpm, bpm_confidence) = estimate_bpm(&env, rate);
    }
    let (key, key_confidence) = estimate_key(&chroma(m, sr));

    Analysis {
        frames,
        channels: nch,
        duration_s: frames as f64 / f64::from(sr),
        truncated,
        peak_dbfs: gain_to_db(peak),
        true_peak_dbfs: meter.true_peak_db().max(gain_to_db(peak)),
        rms_dbfs: rms,
        lufs: meter.integrated_lufs(),
        onsets_s,
        onset_count,
        bpm,
        bpm_confidence,
        key,
        key_confidence,
        centroid_hz: spectral_centroid(m, sr),
        clipped_samples: clipped,
        dc_offset: dc as f32,
    }
}

/// Spectral-flux onset envelope of a mono signal for tempo estimation: log-compressed magnitudes
/// up to 5 kHz, positive differences summed per frame, minus a 0.5 s moving mean and floored at
/// 0. Returns `(envelope, envelope rate in Hz)`; the rate is about 200 Hz. Empty for input
/// shorter than one analysis frame.
pub fn onset_envelope(mono: &[f32], sample_rate: f32) -> (Vec<f32>, f32) {
    let sr = sane_sr(sample_rate);
    let n = ((sr * 0.02) as usize).next_power_of_two().clamp(256, 4096);
    let hop = ((sr / 200.0).round() as usize).clamp(1, n);
    let rate = sr / hop as f32;
    if mono.len() < n {
        return (Vec::new(), rate);
    }
    let frames = frame_count(mono.len(), n, hop);
    let kmax = (((5000.0f32.min(sr * 0.45)) / (sr / n as f32)) as usize).clamp(2, n / 2);
    let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
    let win = hann(n);
    let mut buf = vec![Complex::new(0.0f32, 0.0); n];
    let mut prev = vec![0.0f32; kmax];
    // A full-scale sine peaks at n/4 in a Hann-windowed FFT.
    let norm = 4.0 / n as f32;
    let mut flux = Vec::with_capacity(frames);
    for m in 0..frames {
        load_frame(&mut buf, mono, m * hop, &win);
        fft.process(&mut buf);
        let mut f = 0.0f32;
        for (k, p) in prev.iter_mut().enumerate() {
            let mag = buf.get(k + 1).map_or(0.0, |c| c.norm()) * norm;
            let cur = (1.0 + 1000.0 * mag).ln();
            if m > 0 {
                f += (cur - *p).max(0.0);
            }
            *p = cur;
        }
        flux.push(f);
    }
    // Subtract a moving mean (prefix sums) so only local peaks stay.
    let half = ((0.25 * rate) as usize).max(1);
    let mut prefix = Vec::with_capacity(flux.len() + 1);
    let mut acc = 0.0f64;
    prefix.push(0.0f64);
    for &v in &flux {
        acc += f64::from(v);
        prefix.push(acc);
    }
    let env = flux
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let a = i.saturating_sub(half);
            let b = (i + half + 1).min(flux.len());
            let sum = prefix.get(b).copied().unwrap_or(0.0) - prefix.get(a).copied().unwrap_or(0.0);
            (f64::from(v) - sum / (b - a).max(1) as f64).max(0.0) as f32
        })
        .collect();
    (env, rate)
}

/// Tempo from an onset envelope (`rate` in Hz): the autocorrelation is searched at lags for
/// 60..200 BPM, each candidate scored by its own peak plus half of the one at twice the lag and
/// weighted by a log-normal prior around 120 BPM against octave errors; the best is refined by
/// parabolic interpolation and finally folded by octaves into 80..160 BPM (so a 70 BPM ballad
/// reads 140 and a 190 BPM track 95). Confidence 0..1 is how far the winning autocorrelation peak
/// stands above the median of the search range. `None` (confidence 0) for less than 3 s of
/// envelope or a flat one.
pub fn estimate_bpm(env: &[f32], rate: f32) -> (Option<f32>, f32) {
    let none = (None, 0.0);
    // Envelope rates are about 200-400 Hz; a larger one would only make the lag search huge.
    if !rate.is_finite() || rate <= 0.0 || rate > 2000.0 || (env.len() as f32) < MIN_BPM_SECS * rate {
        return none;
    }
    let n = env.len();
    let mean = env.iter().map(|&v| f64::from(finite(v))).sum::<f64>() / n as f64;
    let x: Vec<f64> = env.iter().map(|&v| f64::from(finite(v)) - mean).collect();
    let ac0 = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
    if ac0.is_nan() || ac0 <= 1e-12 {
        return none;
    }
    let lag_of = |bpm: f32| f64::from(rate) * 60.0 / f64::from(bpm);
    let lo = (lag_of(BPM_MAX).floor() as usize).max(2);
    let hi = lag_of(BPM_MIN).ceil() as usize;
    // r(lag) for 0..=2*hi+1, needed for the doubled-lag support; capped to what the data holds.
    let top = (2 * hi + 1).min(n.saturating_sub(1));
    if hi + 1 > top || lo >= hi {
        return none;
    }
    let r: Vec<f64> = (0..=top)
        .map(|lag| {
            if lag == 0 {
                return 1.0;
            }
            match (x.get(..n - lag), x.get(lag..)) {
                (Some(a), Some(b)) => a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>() / (n - lag) as f64 / ac0,
                _ => 0.0,
            }
        })
        .collect();
    let at = |lag: usize| r.get(lag).copied().unwrap_or(0.0);
    let prior = |lag: usize| {
        let bpm = f64::from(rate) * 60.0 / lag as f64;
        let z = (bpm / 120.0).ln() / 0.7;
        (-0.5 * z * z).exp()
    };
    let mut best: Option<(usize, f64)> = None;
    for lag in lo..=hi {
        let score = (at(lag) + 0.5 * at(2 * lag).max(0.0)) * prior(lag);
        if best.is_none_or(|(_, s)| score > s) {
            best = Some((lag, score));
        }
    }
    let Some((lag, _)) = best else { return none };
    let (a, b, c) = (at(lag - 1), at(lag), at(lag + 1));
    let denom = a - 2.0 * b + c;
    let delta = if b >= a && b >= c && denom < -1e-12 { (0.5 * (a - c) / denom).clamp(-0.5, 0.5) } else { 0.0 };
    let mut bpm = (f64::from(rate) * 60.0 / (lag as f64 + delta)) as f32;
    if !bpm.is_finite() || bpm <= 0.0 {
        return none;
    }
    if bpm < BPM_FOLD_LO {
        bpm *= 2.0;
    } else if bpm > BPM_FOLD_HI {
        bpm /= 2.0;
    }
    let mut spread: Vec<f64> = (lo..=hi).map(at).collect();
    spread.sort_by(|p, q| p.total_cmp(q));
    let median = spread.get(spread.len() / 2).copied().unwrap_or(0.0);
    let conf = (b - median).clamp(0.0, 1.0) as f32;
    (Some(bpm), finite(conf))
}

/// Pitch-class energy of a mono signal, summed over frames. Each Hann-windowed frame (about
/// 0.17 s, 50 % overlap) contributes its spectral peaks between 55 Hz and 2 kHz at their
/// interpolated frequency, with square-root magnitude, normalised to sum 1 so loud passages
/// don't dominate; frames more than 60 dB below the loudest are skipped. Index 0 is C.
pub fn chroma(mono: &[f32], sample_rate: f32) -> [f32; 12] {
    let sr = sane_sr(sample_rate);
    let n = ((sr * 0.17) as usize).next_power_of_two().clamp(2048, 32768);
    let hop = n / 2;
    let frames = frame_count(mono.len(), n, hop);
    let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
    let win = hann(n);
    let mut buf = vec![Complex::new(0.0f32, 0.0); n];
    let bin_hz = sr / n as f32;
    let (lo, hi) = (55.0f32, 2000.0f32.min(sr * 0.45));
    let mut per_frame: Vec<([f32; 12], f32)> = Vec::new();
    let mut mags = vec![0.0f32; n / 2 + 1];
    for m in 0..frames {
        load_frame(&mut buf, mono, m * hop, &win);
        fft.process(&mut buf);
        for (o, c) in mags.iter_mut().zip(&buf) {
            *o = c.norm();
        }
        let top = mags.iter().copied().fold(0.0f32, f32::max);
        if top.is_nan() || top <= 1e-9 {
            continue;
        }
        let mut v = [0.0f32; 12];
        for k in 2..mags.len().saturating_sub(1) {
            let (a, b, c) = (mags.get(k - 1).copied().unwrap_or(0.0), mags.get(k).copied().unwrap_or(0.0), mags.get(k + 1).copied().unwrap_or(0.0));
            if b <= a || b < c || b < top * 0.01 {
                continue;
            }
            // Parabolic peak position on the log magnitudes.
            let (la, lb, lc) = (a.max(1e-12).ln(), b.ln(), c.max(1e-12).ln());
            let denom = la - 2.0 * lb + lc;
            let delta = if denom < -1e-9 { (0.5 * (la - lc) / denom).clamp(-0.5, 0.5) } else { 0.0 };
            let f = (k as f32 + delta) * bin_hz;
            if !(lo..=hi).contains(&f) {
                continue;
            }
            let midi = 69.0 + 12.0 * (f / 440.0).log2();
            let pc = (midi.round() as i64).rem_euclid(12) as usize;
            if let Some(slot) = v.get_mut(pc) {
                *slot += b.sqrt();
            }
        }
        let total: f32 = v.iter().sum();
        if total > 0.0 {
            v.iter_mut().for_each(|x| *x /= total);
            per_frame.push((v, top));
        }
    }
    let loudest = per_frame.iter().map(|f| f.1).fold(0.0f32, f32::max);
    let mut out = [0.0f32; 12];
    for (v, top) in &per_frame {
        if *top >= loudest * 0.001 {
            out.iter_mut().zip(v).for_each(|(o, x)| *o += x);
        }
    }
    out
}

/// Best-matching key of a chroma vector by Pearson correlation with the Krumhansl-Schmuckler
/// profiles rotated to all 12 tonics, major and minor. Confidence 0..1 combines the correlation
/// (0 at 0.35 and below, 1 at 0.85) and its lead over the runner-up key (the relative
/// major/minor included; 1 from a lead of 0.1). `None` (confidence
/// 0) for a flat or silent chroma.
pub fn estimate_key(chroma: &[f32; 12]) -> (Option<Key>, f32) {
    let c: Vec<f64> = chroma.iter().map(|&v| f64::from(finite(v))).collect();
    let mean = c.iter().sum::<f64>() / 12.0;
    let var: f64 = c.iter().map(|v| (v - mean) * (v - mean)).sum();
    if var.is_nan() || var <= 1e-12 {
        return (None, 0.0);
    }
    let mut scores: Vec<(f64, Key)> = Vec::with_capacity(24);
    for minor in [false, true] {
        let prof = if minor { &KS_MINOR } else { &KS_MAJOR };
        let pm = prof.iter().sum::<f64>() / 12.0;
        let pv: f64 = prof.iter().map(|v| (v - pm) * (v - pm)).sum();
        for tonic in 0..12usize {
            let cov: f64 = (0..12usize)
                .map(|pc| {
                    let p = prof.get((pc + 12 - tonic) % 12).copied().unwrap_or(0.0);
                    (c.get(pc).copied().unwrap_or(0.0) - mean) * (p - pm)
                })
                .sum();
            scores.push((cov / (var * pv).sqrt(), Key { tonic: tonic as u8, minor }));
        }
    }
    scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (Some(&(best, key)), Some(&(second, _))) = (scores.first(), scores.get(1)) else { return (None, 0.0) };
    if !best.is_finite() {
        return (None, 0.0);
    }
    // Random chroma correlates with some profile at about 0.3-0.4, so only what lies above 0.35
    // counts; a margin of 0.1 over the runner-up is a clear call.
    let strength = ((best - 0.35) / 0.5).clamp(0.0, 1.0);
    let margin = ((best - second) / 0.1).clamp(0.0, 1.0);
    (Some(key), (strength * margin) as f32)
}

/// Magnitude-weighted mean frequency in Hz over the whole signal (about 0.17 s frames, 50 %
/// overlap, DC excluded); 0 for silence.
pub fn spectral_centroid(mono: &[f32], sample_rate: f32) -> f32 {
    let sr = sane_sr(sample_rate);
    let n = ((sr * 0.17) as usize).next_power_of_two().clamp(2048, 32768);
    let hop = n / 2;
    let frames = frame_count(mono.len(), n, hop);
    let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
    let win = hann(n);
    let mut buf = vec![Complex::new(0.0f32, 0.0); n];
    let bin_hz = f64::from(sr) / n as f64;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for m in 0..frames {
        load_frame(&mut buf, mono, m * hop, &win);
        fft.process(&mut buf);
        for (k, c) in buf.iter().enumerate().take(n / 2 + 1).skip(1) {
            let mag = f64::from(c.norm());
            num += mag * k as f64 * bin_hz;
            den += mag;
        }
    }
    if den > 1e-9 { finite((num / den) as f32) } else { 0.0 }
}
