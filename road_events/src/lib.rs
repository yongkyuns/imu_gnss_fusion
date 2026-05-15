#![no_std]
//! Streaming road event detectors for embedded IMU/GNSS fusion outputs.
//!
//! The speed-bump detector is intentionally small-state and embedded-friendly.
//! It uses gravity-compensated vehicle-frame vertical acceleration as the
//! primary trigger, because a bump is first a vertical road impulse at the
//! front/rear axles. Vehicle pitch is used as a corroborating signal so harsh
//! non-bump vertical motion is less likely to trigger.

/// One detected road speed-bump event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeedBumpEvent {
    pub t_s: f32,
    /// UI-facing confidence normalized after the internal score clears the
    /// trigger threshold. This is not a calibrated probability.
    pub confidence: f32,
    pub duration_s: f32,
    pub peak_abs_pitch_deg: f32,
}

/// Per-sample detector diagnostics useful for plotting and tuning.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpeedBumpDiagnostic {
    pub t_s: f32,
    pub pitch_hpf_deg: f32,
    pub pitch_noise_deg: f32,
    pub vertical_accel_hpf_mps2: f32,
    pub vertical_accel_noise_mps2: f32,
}

/// Vehicle motion sample consumed by [`SpeedBumpDetector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeedBumpSample {
    pub t_s: f32,
    pub speed_mps: f32,
    pub pitch_deg: f32,
    /// Gravity-compensated vehicle-frame vertical acceleration.
    pub vertical_accel_mps2: f32,
}

/// Configuration for the small-state speed-bump detector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeedBumpConfig {
    /// Pitch high-pass cutoff used to reject road grade and slow maneuvers.
    pub pitch_hpf_cutoff_hz: f32,
    /// Vertical acceleration high-pass cutoff used to reject slow body motion.
    pub vertical_accel_hpf_cutoff_hz: f32,
    /// EMA time constant for adaptive noise-floor estimation.
    pub noise_tau_s: f32,
    /// Minimum speed for speed-adaptive spacing logic.
    pub min_speed_mps: f32,
    /// Lower plausible wheelbase used to convert speed into peak spacing.
    pub wheelbase_min_m: f32,
    /// Upper plausible wheelbase used to convert speed into peak spacing.
    pub wheelbase_max_m: f32,
    /// Absolute minimum accepted event duration.
    pub min_event_duration_s: f32,
    /// Absolute maximum accepted event duration.
    pub max_event_duration_s: f32,
    /// Adaptive vertical-acceleration peak threshold multiplier.
    pub vertical_accel_noise_peak_scale: f32,
    /// Minimum gravity-compensated vertical acceleration peak for a bump candidate.
    pub min_vertical_accel_peak_mps2: f32,
    /// Minimum fraction of the accel-pattern duration with vertical accel above the physical floor.
    pub min_vertical_accel_active_fraction: f32,
    /// Minimum total time within the pattern with vertical accel above the physical floor.
    pub min_vertical_accel_active_duration_s: f32,
    /// Adaptive pitch corroboration threshold multiplier.
    pub pitch_noise_peak_scale: f32,
    /// Confidence required before an event is emitted.
    pub trigger_confidence: f32,
    /// Event refractory period after a trigger.
    pub refractory_s: f32,
}

impl Default for SpeedBumpConfig {
    fn default() -> Self {
        Self {
            pitch_hpf_cutoff_hz: 0.45,
            vertical_accel_hpf_cutoff_hz: 0.70,
            noise_tau_s: 6.0,
            min_speed_mps: 1.5,
            wheelbase_min_m: 1.8,
            wheelbase_max_m: 3.6,
            min_event_duration_s: 0.18,
            max_event_duration_s: 1.8,
            vertical_accel_noise_peak_scale: 3.5,
            min_vertical_accel_peak_mps2: 1.5,
            min_vertical_accel_active_fraction: 0.25,
            min_vertical_accel_active_duration_s: 0.25,
            pitch_noise_peak_scale: 3.0,
            trigger_confidence: 0.12,
            refractory_s: 4.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct AccelExtremum {
    t_s: f32,
    accel_hpf_mps2: f32,
    pitch_peak_deg: f32,
    speed_mps: f32,
    active_time_s: f32,
}

struct PatternCandidate {
    duration_s: f32,
    speeds: [f32; 3],
    accel_peaks: [f32; 3],
    pitch_peaks: [f32; 3],
    active_duration_s: f32,
    active_fraction: f32,
    balance: f32,
}

/// Streaming detector for the front/rear-axle vertical acceleration pattern of
/// a speed bump.
#[derive(Clone, Debug)]
pub struct SpeedBumpDetector {
    cfg: SpeedBumpConfig,
    last_t_s: Option<f32>,
    last_pitch_deg: f32,
    last_vertical_accel_mps2: f32,
    last_pitch_hpf_deg: f32,
    last_accel_hpf_mps2: f32,
    prev_t_s: f32,
    prev_accel_hpf_mps2: f32,
    prev_accel_slope: f32,
    pitch_noise_deg: f32,
    accel_noise_mps2: f32,
    accel_active_time_s: f32,
    pitch_peak_since_extremum_deg: f32,
    extrema: [Option<AccelExtremum>; 4],
    extrema_len: usize,
    last_event_t_s: f32,
}

impl SpeedBumpDetector {
    pub fn new(cfg: SpeedBumpConfig) -> Self {
        Self {
            cfg,
            last_t_s: None,
            last_pitch_deg: 0.0,
            last_vertical_accel_mps2: 0.0,
            last_pitch_hpf_deg: 0.0,
            last_accel_hpf_mps2: 0.0,
            prev_t_s: 0.0,
            prev_accel_hpf_mps2: 0.0,
            prev_accel_slope: 0.0,
            pitch_noise_deg: 0.03,
            accel_noise_mps2: 0.10,
            accel_active_time_s: 0.0,
            pitch_peak_since_extremum_deg: 0.0,
            extrema: [None; 4],
            extrema_len: 0,
            last_event_t_s: -1.0e9,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }

    pub fn update(
        &mut self,
        sample: SpeedBumpSample,
    ) -> (SpeedBumpDiagnostic, Option<SpeedBumpEvent>) {
        if !sample.t_s.is_finite()
            || !sample.speed_mps.is_finite()
            || !sample.pitch_deg.is_finite()
            || !sample.vertical_accel_mps2.is_finite()
        {
            return (SpeedBumpDiagnostic::default(), None);
        }

        let Some(last_t_s) = self.last_t_s else {
            self.last_t_s = Some(sample.t_s);
            self.last_pitch_deg = sample.pitch_deg;
            self.last_vertical_accel_mps2 = sample.vertical_accel_mps2;
            self.prev_t_s = sample.t_s;
            return (
                SpeedBumpDiagnostic {
                    t_s: sample.t_s,
                    ..SpeedBumpDiagnostic::default()
                },
                None,
            );
        };

        let dt = (sample.t_s - last_t_s).clamp(1.0e-3, 0.2);
        let pitch_hpf = self.high_pass_pitch(sample.pitch_deg, dt);
        let accel_hpf = self.high_pass_vertical_accel(sample.vertical_accel_mps2, dt);
        self.update_noise(pitch_hpf, accel_hpf, dt);
        if accel_hpf.abs() >= self.cfg.min_vertical_accel_peak_mps2 {
            self.accel_active_time_s += dt;
        }
        self.pitch_peak_since_extremum_deg =
            self.pitch_peak_since_extremum_deg.max(pitch_hpf.abs());

        let accel_slope = (accel_hpf - self.last_accel_hpf_mps2) / dt;
        let mut event = None;
        if self.prev_accel_slope != 0.0
            && accel_slope != 0.0
            && self.prev_accel_slope.signum() != accel_slope.signum()
            && self.prev_accel_hpf_mps2.abs() >= self.accel_threshold()
        {
            self.push_extremum(AccelExtremum {
                t_s: self.prev_t_s,
                accel_hpf_mps2: self.prev_accel_hpf_mps2,
                pitch_peak_deg: self.pitch_peak_since_extremum_deg,
                speed_mps: sample.speed_mps.max(0.0),
                active_time_s: self.accel_active_time_s,
            });
            self.pitch_peak_since_extremum_deg = 0.0;
            event = self.evaluate_latest_pattern();
        }

        self.last_t_s = Some(sample.t_s);
        self.last_pitch_deg = sample.pitch_deg;
        self.last_vertical_accel_mps2 = sample.vertical_accel_mps2;
        self.last_pitch_hpf_deg = pitch_hpf;
        self.last_accel_hpf_mps2 = accel_hpf;
        self.prev_t_s = sample.t_s;
        self.prev_accel_hpf_mps2 = accel_hpf;
        self.prev_accel_slope = accel_slope;

        (
            SpeedBumpDiagnostic {
                t_s: sample.t_s,
                pitch_hpf_deg: pitch_hpf,
                pitch_noise_deg: self.pitch_noise_deg,
                vertical_accel_hpf_mps2: accel_hpf,
                vertical_accel_noise_mps2: self.accel_noise_mps2,
            },
            event,
        )
    }

    fn high_pass_pitch(&self, pitch_deg: f32, dt: f32) -> f32 {
        let rc = 1.0 / (core::f32::consts::TAU * self.cfg.pitch_hpf_cutoff_hz.max(1.0e-3));
        let alpha = rc / (rc + dt);
        alpha * (self.last_pitch_hpf_deg + pitch_deg - self.last_pitch_deg)
    }

    fn high_pass_vertical_accel(&self, accel_mps2: f32, dt: f32) -> f32 {
        let rc = 1.0 / (core::f32::consts::TAU * self.cfg.vertical_accel_hpf_cutoff_hz.max(1.0e-3));
        let alpha = rc / (rc + dt);
        alpha * (self.last_accel_hpf_mps2 + accel_mps2 - self.last_vertical_accel_mps2)
    }

    fn update_noise(&mut self, pitch_hpf_deg: f32, accel_hpf_mps2: f32, dt: f32) {
        let alpha = dt / (self.cfg.noise_tau_s.max(dt) + dt);
        self.pitch_noise_deg = (1.0 - alpha) * self.pitch_noise_deg + alpha * pitch_hpf_deg.abs();
        self.accel_noise_mps2 =
            (1.0 - alpha) * self.accel_noise_mps2 + alpha * accel_hpf_mps2.abs();
    }

    fn accel_threshold(&self) -> f32 {
        (self.cfg.vertical_accel_noise_peak_scale * self.accel_noise_mps2)
            .max(self.cfg.min_vertical_accel_peak_mps2)
    }

    fn pitch_threshold(&self) -> f32 {
        (self.cfg.pitch_noise_peak_scale * self.pitch_noise_deg).max(1.0e-3)
    }

    fn push_extremum(&mut self, extremum: AccelExtremum) {
        if self.extrema_len > 0 {
            let last_index = self.extrema_len.min(self.extrema.len()) - 1;
            if let Some(last) = &mut self.extrema[last_index] {
                let same_polarity =
                    last.accel_hpf_mps2.signum() == extremum.accel_hpf_mps2.signum();
                let same_impulse = extremum.t_s - last.t_s < self.cfg.min_event_duration_s;
                if same_polarity && same_impulse {
                    if extremum.accel_hpf_mps2.abs() > last.accel_hpf_mps2.abs() {
                        *last = extremum;
                    } else {
                        last.pitch_peak_deg = last.pitch_peak_deg.max(extremum.pitch_peak_deg);
                    }
                    return;
                }
            }
        }
        if self.extrema_len < self.extrema.len() {
            self.extrema[self.extrema_len] = Some(extremum);
            self.extrema_len += 1;
            return;
        }
        self.extrema.rotate_left(1);
        *self.extrema.last_mut().expect("nonempty extrema ring") = Some(extremum);
    }

    fn evaluate_latest_pattern(&mut self) -> Option<SpeedBumpEvent> {
        let mut best = None;
        if let [Some(a), Some(b), Some(c)] = self.latest_three() {
            best = Some(self.score_pattern([a, b, c]));
        }

        let (t_s, pattern_score, duration_s, peak_abs_pitch_deg) = best?;
        if t_s - self.last_event_t_s < self.cfg.refractory_s {
            return None;
        }
        if pattern_score < self.cfg.trigger_confidence {
            return None;
        }

        self.last_event_t_s = t_s;
        Some(SpeedBumpEvent {
            t_s,
            confidence: self.display_confidence(pattern_score),
            duration_s,
            peak_abs_pitch_deg,
        })
    }

    fn latest_three(&self) -> [Option<AccelExtremum>; 3] {
        if self.extrema_len < 3 {
            return [None, None, None];
        }
        let start = self.extrema_len - 3;
        [
            self.extrema[start],
            self.extrema[start + 1],
            self.extrema[start + 2],
        ]
    }

    fn score_pattern(&self, extrema: [AccelExtremum; 3]) -> (f32, f32, f32, f32) {
        let [a, b, c] = extrema;
        if !alternating(a.accel_hpf_mps2, b.accel_hpf_mps2, c.accel_hpf_mps2) {
            return (b.t_s, 0.0, c.t_s - a.t_s, 0.0);
        }
        let duration_s = c.t_s - a.t_s;
        let score = self.pattern_score(PatternCandidate {
            duration_s,
            speeds: [a.speed_mps, b.speed_mps, c.speed_mps],
            accel_peaks: [
                a.accel_hpf_mps2.abs(),
                b.accel_hpf_mps2.abs(),
                c.accel_hpf_mps2.abs(),
            ],
            pitch_peaks: [a.pitch_peak_deg, b.pitch_peak_deg, c.pitch_peak_deg],
            active_duration_s: c.active_time_s - a.active_time_s,
            active_fraction: (c.active_time_s - a.active_time_s) / duration_s.max(1.0e-3),
            balance: a.accel_hpf_mps2.abs().min(c.accel_hpf_mps2.abs())
                / a.accel_hpf_mps2
                    .abs()
                    .max(c.accel_hpf_mps2.abs())
                    .max(1.0e-3),
        });
        (b.t_s, score, duration_s, pitch_peak([a, b, c]))
    }

    fn pattern_score(&self, candidate: PatternCandidate) -> f32 {
        if candidate.duration_s <= 0.0 {
            return 0.0;
        }
        let count = if candidate.speeds[2] > 0.0 { 3.0 } else { 2.0 };
        let speed_mps = (candidate.speeds[0] + candidate.speeds[1] + candidate.speeds[2]) / count;
        if speed_mps < self.cfg.min_speed_mps {
            return 0.0;
        }

        let speed_min_s = 0.35 * self.cfg.wheelbase_min_m / speed_mps;
        let speed_max_s = 2.5 * self.cfg.wheelbase_max_m / speed_mps;
        let min_s = self.cfg.min_event_duration_s.max(speed_min_s);
        let max_s = self.cfg.max_event_duration_s.min(speed_max_s.max(min_s));
        if candidate.duration_s < min_s || candidate.duration_s > max_s {
            return 0.0;
        }
        if candidate.active_fraction < self.cfg.min_vertical_accel_active_fraction {
            return 0.0;
        }
        if candidate.active_duration_s < self.cfg.min_vertical_accel_active_duration_s {
            return 0.0;
        }

        let accel_peak = candidate.accel_peaks[0]
            .max(candidate.accel_peaks[1])
            .max(candidate.accel_peaks[2]);
        let pitch_peak = candidate.pitch_peaks[0]
            .max(candidate.pitch_peaks[1])
            .max(candidate.pitch_peaks[2]);
        let accel_score = ((accel_peak / self.accel_threshold()) - 1.0).clamp(0.0, 1.0);
        let pitch_score = ((pitch_peak / self.pitch_threshold()) - 1.0).clamp(0.0, 1.0);
        let center = 0.5 * (min_s + max_s);
        let half_width = 0.5 * (max_s - min_s).max(1.0e-3);
        let spacing_score =
            (1.0 - ((candidate.duration_s - center).abs() / half_width)).clamp(0.0, 1.0);
        let shape_score = (0.45 * accel_score
            + 0.25 * spacing_score
            + 0.15 * candidate.balance.clamp(0.0, 1.0)
            + 0.15 * pitch_score)
            .clamp(0.0, 1.0);
        (shape_score * pitch_score).clamp(0.0, 1.0)
    }

    fn display_confidence(&self, pattern_score: f32) -> f32 {
        let margin = ((pattern_score - self.cfg.trigger_confidence)
            / (1.0 - self.cfg.trigger_confidence).max(1.0e-3))
        .clamp(0.0, 1.0);
        0.90 + 0.08 * margin
    }
}

fn alternating(a: f32, b: f32, c: f32) -> bool {
    a.signum() == c.signum() && a.signum() != b.signum()
}

fn pitch_peak(extrema: [AccelExtremum; 3]) -> f32 {
    extrema[0]
        .pitch_peak_deg
        .max(extrema[1].pitch_peak_deg)
        .max(extrema[2].pitch_peak_deg)
}

#[cfg(test)]
mod tests {
    use super::{SpeedBumpConfig, SpeedBumpDetector, SpeedBumpSample};

    #[test]
    fn detects_accel_double_peak_with_pitch_confirmation() {
        let mut detector = SpeedBumpDetector::new(SpeedBumpConfig {
            trigger_confidence: 0.12,
            ..SpeedBumpConfig::default()
        });
        let mut events = 0;
        for i in 0..600 {
            let t = i as f32 * 0.01;
            let accel =
                gaussian(t, 2.00, 0.12) - 1.2 * gaussian(t, 2.30, 0.12) + gaussian(t, 2.60, 0.12);
            let pitch = 0.90 * gaussian(t, 2.10, 0.14) - 1.10 * gaussian(t, 2.42, 0.14);
            let (_, event) = detector.update(SpeedBumpSample {
                t_s: t,
                speed_mps: 4.0,
                pitch_deg: pitch,
                vertical_accel_mps2: 4.0 * accel,
            });
            if event.is_some() {
                events += 1;
            }
        }
        assert_eq!(events, 1);
    }

    #[test]
    fn ignores_slow_pitch_drift() {
        let mut detector = SpeedBumpDetector::new(SpeedBumpConfig::default());
        let mut events = 0;
        for i in 0..700 {
            let t = i as f32 * 0.01;
            let (_, event) = detector.update(SpeedBumpSample {
                t_s: t,
                speed_mps: 5.0,
                pitch_deg: 0.8 * (0.2 * t).sin(),
                vertical_accel_mps2: 0.2 * (0.3 * t).sin(),
            });
            if event.is_some() {
                events += 1;
            }
        }
        assert_eq!(events, 0);
    }

    #[test]
    fn ignores_vertical_impulse_without_pitch_confirmation() {
        let mut detector = SpeedBumpDetector::new(SpeedBumpConfig {
            trigger_confidence: 0.25,
            ..SpeedBumpConfig::default()
        });
        let mut events = 0;
        for i in 0..600 {
            let t = i as f32 * 0.01;
            let accel =
                gaussian(t, 2.00, 0.12) - 1.2 * gaussian(t, 2.30, 0.12) + gaussian(t, 2.60, 0.12);
            let (_, event) = detector.update(SpeedBumpSample {
                t_s: t,
                speed_mps: 4.0,
                pitch_deg: 0.02 * (13.0 * t).sin(),
                vertical_accel_mps2: 4.0 * accel,
            });
            if event.is_some() {
                events += 1;
            }
        }
        assert_eq!(events, 0);
    }

    fn gaussian(t: f32, center: f32, sigma: f32) -> f32 {
        let z = (t - center) / sigma;
        (-0.5 * z * z).exp()
    }
}
