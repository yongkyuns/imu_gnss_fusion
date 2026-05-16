use crate::common::elapsed_since_last;
use crate::{ReverseConfig, ReverseEvent, ReverseSample};

#[derive(Clone, Copy, Debug)]
struct ActiveReverse {
    start_t_s: f32,
    last_t_s: f32,
    reverse_speed_time_sum_m: f32,
    duration_s: f32,
    peak_reverse_speed_mps: f32,
}

impl ActiveReverse {
    fn new(sample: ReverseSample) -> Self {
        let reverse_speed_mps = (-sample.forward_velocity_mps).max(0.0);
        Self {
            start_t_s: sample.t_s,
            last_t_s: sample.t_s,
            reverse_speed_time_sum_m: 0.0,
            duration_s: 0.0,
            peak_reverse_speed_mps: reverse_speed_mps,
        }
    }

    fn add_sample(&mut self, sample: ReverseSample, dt: f32) {
        let reverse_speed_mps = (-sample.forward_velocity_mps).max(0.0);
        self.last_t_s = sample.t_s;
        self.reverse_speed_time_sum_m += reverse_speed_mps * dt;
        self.duration_s += dt;
        self.peak_reverse_speed_mps = self.peak_reverse_speed_mps.max(reverse_speed_mps);
    }

    fn event(self) -> ReverseEvent {
        let duration_s = self.duration_s.max(0.0);
        ReverseEvent {
            start_t_s: self.start_t_s,
            end_t_s: self.last_t_s,
            duration_s,
            mean_reverse_speed_mps: if duration_s > 0.0 {
                self.reverse_speed_time_sum_m / duration_s
            } else {
                0.0
            },
            peak_reverse_speed_mps: self.peak_reverse_speed_mps,
        }
    }
}

/// Streaming detector for reverse-driving intervals.
#[derive(Clone, Debug)]
pub struct ReverseDetector {
    cfg: ReverseConfig,
    last_t_s: Option<f32>,
    active: Option<ActiveReverse>,
    enter_duration_s: f32,
    exit_duration_s: f32,
    confirmed: bool,
}

impl ReverseDetector {
    pub fn new(cfg: ReverseConfig) -> Self {
        Self {
            cfg,
            last_t_s: None,
            active: None,
            enter_duration_s: 0.0,
            exit_duration_s: 0.0,
            confirmed: false,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }

    pub fn update(&mut self, sample: ReverseSample) -> Option<ReverseEvent> {
        if !sample.t_s.is_finite() || !sample.forward_velocity_mps.is_finite() {
            return None;
        }
        let dt = elapsed_since_last(&mut self.last_t_s, sample.t_s);
        let entering = sample.forward_velocity_mps < self.cfg.enter_forward_velocity_mps;
        let staying = sample.forward_velocity_mps < self.cfg.exit_forward_velocity_mps;

        let Some(mut active) = self.active.take() else {
            if entering {
                self.active = Some(ActiveReverse::new(sample));
                self.enter_duration_s = 0.0;
                self.exit_duration_s = 0.0;
                self.confirmed = false;
            }
            return None;
        };

        if self.confirmed {
            active.add_sample(sample, dt);
            if staying {
                self.exit_duration_s = 0.0;
                self.active = Some(active);
                return None;
            }
            self.exit_duration_s += dt;
            if self.exit_duration_s >= self.cfg.exit_debounce_s {
                self.enter_duration_s = 0.0;
                self.exit_duration_s = 0.0;
                self.confirmed = false;
                return self.finish_active(active);
            }
            self.active = Some(active);
            return None;
        }

        if entering {
            active.add_sample(sample, dt);
            self.enter_duration_s += dt;
            if self.enter_duration_s >= self.cfg.enter_debounce_s {
                self.confirmed = true;
            }
            self.active = Some(active);
        } else {
            self.enter_duration_s = 0.0;
            self.exit_duration_s = 0.0;
            self.confirmed = false;
        }
        None
    }

    pub fn finish(&mut self) -> Option<ReverseEvent> {
        let active = self.active.take()?;
        let confirmed = self.confirmed;
        self.enter_duration_s = 0.0;
        self.exit_duration_s = 0.0;
        self.confirmed = false;
        confirmed
            .then_some(active)
            .and_then(|active| self.finish_active(active))
    }

    fn finish_active(&self, active: ActiveReverse) -> Option<ReverseEvent> {
        (active.duration_s >= self.cfg.min_duration_s).then(|| active.event())
    }
}
