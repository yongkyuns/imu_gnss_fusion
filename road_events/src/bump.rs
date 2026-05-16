use crate::common::{high_pass, update_abs_ema};
use crate::{SpeedBumpConfig, SpeedBumpDiagnostic, SpeedBumpEvent, SpeedBumpSample};

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
        let pitch_hpf_deg = self.high_pass_pitch(sample.pitch_deg, dt);
        let accel_hpf_mps2 = self.high_pass_vertical_accel(sample.vertical_accel_mps2, dt);
        self.update_noise(pitch_hpf_deg, accel_hpf_mps2, dt);

        if accel_hpf_mps2.abs() >= self.min_vertical_accel_peak() {
            self.accel_active_time_s += dt;
        }
        self.pitch_peak_since_extremum_deg =
            self.pitch_peak_since_extremum_deg.max(pitch_hpf_deg.abs());

        let event = self.detect_extremum(sample, accel_hpf_mps2);

        self.last_t_s = Some(sample.t_s);
        self.last_pitch_deg = sample.pitch_deg;
        self.last_vertical_accel_mps2 = sample.vertical_accel_mps2;
        self.last_pitch_hpf_deg = pitch_hpf_deg;
        self.last_accel_hpf_mps2 = accel_hpf_mps2;
        self.prev_accel_hpf_mps2 = accel_hpf_mps2;
        self.prev_t_s = sample.t_s;

        (
            SpeedBumpDiagnostic {
                t_s: sample.t_s,
                pitch_hpf_deg,
                pitch_noise_deg: self.pitch_noise_deg,
                vertical_accel_hpf_mps2: accel_hpf_mps2,
                vertical_accel_noise_mps2: self.accel_noise_mps2,
            },
            event,
        )
    }

    fn high_pass_pitch(&self, pitch_deg: f32, dt: f32) -> f32 {
        high_pass(
            pitch_deg,
            self.last_pitch_deg,
            self.last_pitch_hpf_deg,
            self.cfg.pitch_hpf_cutoff_hz,
            dt,
        )
    }

    fn high_pass_vertical_accel(&self, accel_mps2: f32, dt: f32) -> f32 {
        high_pass(
            accel_mps2,
            self.last_vertical_accel_mps2,
            self.last_accel_hpf_mps2,
            self.cfg.vertical_accel_hpf_cutoff_hz,
            dt,
        )
    }

    fn update_noise(&mut self, pitch_hpf_deg: f32, accel_hpf_mps2: f32, dt: f32) {
        self.pitch_noise_deg = update_abs_ema(
            self.pitch_noise_deg,
            pitch_hpf_deg,
            self.cfg.noise_tau_s,
            dt,
        );
        self.accel_noise_mps2 = update_abs_ema(
            self.accel_noise_mps2,
            accel_hpf_mps2,
            self.cfg.noise_tau_s,
            dt,
        );
    }

    fn accel_threshold(&self) -> f32 {
        (self.cfg.vertical_accel_noise_peak_scale * self.accel_noise_mps2)
            .max(self.min_vertical_accel_peak())
    }

    fn min_vertical_accel_peak(&self) -> f32 {
        self.cfg.min_vertical_accel_peak_mps2.max(0.1)
    }

    fn pitch_threshold(&self) -> f32 {
        (self.cfg.pitch_noise_peak_scale * self.pitch_noise_deg).max(0.25)
    }

    fn detect_extremum(
        &mut self,
        sample: SpeedBumpSample,
        accel_hpf_mps2: f32,
    ) -> Option<SpeedBumpEvent> {
        let dt = (sample.t_s - self.prev_t_s).max(1.0e-3);
        let slope = (accel_hpf_mps2 - self.prev_accel_hpf_mps2) / dt;
        let crossed_peak = self.prev_accel_slope > 0.0 && slope <= 0.0;
        let crossed_valley = self.prev_accel_slope < 0.0 && slope >= 0.0;
        self.prev_accel_slope = slope;

        if !(crossed_peak || crossed_valley) {
            return None;
        }
        if self.prev_accel_hpf_mps2.abs() < self.min_vertical_accel_peak() {
            return None;
        }

        let extremum = AccelExtremum {
            t_s: self.prev_t_s,
            accel_hpf_mps2: self.prev_accel_hpf_mps2,
            pitch_peak_deg: self.pitch_peak_since_extremum_deg,
            speed_mps: sample.speed_mps,
            active_time_s: self.accel_active_time_s,
        };
        self.pitch_peak_since_extremum_deg = 0.0;
        self.push_extremum(extremum);
        self.evaluate_latest_pattern()
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
