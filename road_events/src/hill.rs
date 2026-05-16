use crate::common::elapsed_since_last;
use crate::{HillConfig, HillEvent, HillKind, HillSample};

#[derive(Clone, Copy, Debug)]
struct ActiveHill {
    kind: HillKind,
    start_t_s: f32,
    last_t_s: f32,
    pitch_time_sum_deg_s: f32,
    speed_time_sum_m: f32,
    duration_s: f32,
    peak_abs_pitch_deg: f32,
    emitted: bool,
}

impl ActiveHill {
    fn new(kind: HillKind, sample: HillSample) -> Self {
        Self {
            kind,
            start_t_s: sample.t_s,
            last_t_s: sample.t_s,
            pitch_time_sum_deg_s: 0.0,
            speed_time_sum_m: 0.0,
            duration_s: 0.0,
            peak_abs_pitch_deg: sample.pitch_deg.abs(),
            emitted: false,
        }
    }

    fn add_sample(&mut self, sample: HillSample, dt: f32) {
        self.last_t_s = sample.t_s;
        self.pitch_time_sum_deg_s += sample.pitch_deg * dt;
        self.speed_time_sum_m += sample.speed_mps.max(0.0) * dt;
        self.duration_s += dt;
        self.peak_abs_pitch_deg = self.peak_abs_pitch_deg.max(sample.pitch_deg.abs());
    }

    fn event(self) -> HillEvent {
        let duration_s = self.duration_s.max(0.0);
        HillEvent {
            kind: self.kind,
            start_t_s: self.start_t_s,
            end_t_s: self.last_t_s,
            duration_s,
            mean_pitch_deg: if duration_s > 0.0 {
                self.pitch_time_sum_deg_s / duration_s
            } else {
                0.0
            },
            peak_abs_pitch_deg: self.peak_abs_pitch_deg,
            mean_speed_mps: if duration_s > 0.0 {
                self.speed_time_sum_m / duration_s
            } else {
                0.0
            },
        }
    }
}

/// Streaming detector for sustained uphill/downhill vehicle pitch intervals.
#[derive(Clone, Debug)]
pub struct HillDetector {
    cfg: HillConfig,
    last_t_s: Option<f32>,
    active: Option<ActiveHill>,
}

impl HillDetector {
    pub fn new(cfg: HillConfig) -> Self {
        Self {
            cfg,
            last_t_s: None,
            active: None,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }

    pub fn update(&mut self, sample: HillSample) -> Option<HillEvent> {
        if !sample.t_s.is_finite() || !sample.speed_mps.is_finite() || !sample.pitch_deg.is_finite()
        {
            return None;
        }
        let dt = elapsed_since_last(&mut self.last_t_s, sample.t_s);
        let sample_kind = hill_kind(sample.pitch_deg, self.cfg.pitch_threshold_deg);
        let mut event = None;
        match (self.active.take(), sample_kind) {
            (Some(mut active), Some(kind)) if active.kind == kind => {
                active.add_sample(sample, dt);
                if active.duration_s >= self.cfg.min_duration_s {
                    active.emitted = true;
                }
                self.active = Some(active);
            }
            (Some(active), Some(kind)) => {
                event = self.finish_active(active);
                self.active = Some(ActiveHill::new(kind, sample));
            }
            (Some(active), None) => {
                event = self.finish_active(active);
            }
            (None, Some(kind)) => {
                self.active = Some(ActiveHill::new(kind, sample));
            }
            (None, None) => {}
        }
        event
    }

    pub fn finish(&mut self) -> Option<HillEvent> {
        self.active
            .take()
            .and_then(|active| self.finish_active(active))
    }

    fn finish_active(&self, active: ActiveHill) -> Option<HillEvent> {
        (active.emitted && active.duration_s >= self.cfg.min_duration_s).then(|| active.event())
    }
}

fn hill_kind(pitch_deg: f32, threshold_deg: f32) -> Option<HillKind> {
    if pitch_deg >= threshold_deg {
        Some(HillKind::Uphill)
    } else if pitch_deg <= -threshold_deg {
        Some(HillKind::Downhill)
    } else {
        None
    }
}
