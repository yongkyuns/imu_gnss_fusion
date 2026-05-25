use crate::common::{high_pass, low_pass, sqrt_f32};
use crate::{
    RoadRoughnessConfig, RoadRoughnessEstimate, RoadRoughnessEvent, RoadRoughnessLevel,
    RoadRoughnessSample, RoadRoughnessUpdate, RoadShockEvent,
};

/// Streaming road-surface roughness analyzer.
///
/// Roughness is formulated as distance-normalized, band-limited vertical
/// acceleration energy:
///
/// `roughness = sqrt(EMA_distance(robust_limit(BPF(a_z))^2))`
///
/// Short, isolated high-energy impacts are routed to shock events and
/// downweighted before the roughness energy update. The analyzer keeps only
/// filter state and a few accumulators. It does not retain a rolling sample
/// buffer, so memory use is constant and suitable for embedded targets.
#[derive(Clone, Copy, Debug)]
pub struct RoadRoughnessAnalyzer {
    cfg: RoadRoughnessConfig,
    last_t_s: Option<f32>,
    hp_last_input: f32,
    hp_last_output: f32,
    lp_output: f32,
    filter_initialized: bool,
    energy_mps2_sq: f32,
    energy_initialized: bool,
    baseline_abs_mps2: f32,
    baseline_initialized: bool,
    distance_m: f32,
    rough_active: Option<ActiveRoughness>,
    last_rough_event_t_s: f32,
    shock_active: Option<ActiveShock>,
    last_shock_event_t_s: f32,
    last_estimate: RoadRoughnessEstimate,
}

#[derive(Clone, Copy, Debug)]
struct ActiveRoughness {
    start_t_s: f32,
    last_t_s: f32,
    duration_s: f32,
    roughness_time_sum: f32,
    peak_roughness_rms_mps2: f32,
    speed_time_sum_m: f32,
    distance_m: f32,
    emitted: bool,
}

#[derive(Clone, Copy, Debug)]
struct ActiveShock {
    start_t_s: f32,
    last_t_s: f32,
    duration_s: f32,
    peak_abs_vertical_accel_mps2: f32,
    speed_time_sum_m: f32,
}

impl RoadRoughnessAnalyzer {
    pub fn new(cfg: RoadRoughnessConfig) -> Self {
        Self {
            cfg,
            last_t_s: None,
            hp_last_input: 0.0,
            hp_last_output: 0.0,
            lp_output: 0.0,
            filter_initialized: false,
            energy_mps2_sq: 0.0,
            energy_initialized: false,
            baseline_abs_mps2: 0.0,
            baseline_initialized: false,
            distance_m: 0.0,
            rough_active: None,
            last_rough_event_t_s: -1.0e9,
            shock_active: None,
            last_shock_event_t_s: -1.0e9,
            last_estimate: RoadRoughnessEstimate {
                t_s: 0.0,
                roughness_rms_mps2: 0.0,
                level: RoadRoughnessLevel::VerySmooth,
                vertical_accel_bandpass_mps2: 0.0,
                vertical_accel_clipped_mps2: 0.0,
                distance_m: 0.0,
                updated: false,
            },
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }

    pub fn update(&mut self, sample: RoadRoughnessSample) -> Option<RoadRoughnessEstimate> {
        self.update_with_events(sample)
            .map(|update| update.estimate)
    }

    pub fn update_with_events(
        &mut self,
        sample: RoadRoughnessSample,
    ) -> Option<RoadRoughnessUpdate> {
        if !valid_sample(sample) {
            return None;
        }

        let raw_dt = self
            .last_t_s
            .map(|last_t_s| (sample.t_s - last_t_s).max(0.0))
            .unwrap_or(0.0);
        self.last_t_s = Some(sample.t_s);
        let dt = raw_dt.min(self.cfg.max_dt_s.max(0.0));

        if !self.filter_initialized {
            self.filter_initialized = true;
            self.hp_last_input = sample.vertical_accel_mps2;
            self.hp_last_output = 0.0;
            self.lp_output = 0.0;
            return Some(RoadRoughnessUpdate {
                estimate: self.make_estimate(sample.t_s, 0.0, 0.0, false),
                roughness_event: None,
                completed_roughness_event: None,
                shock_event: None,
            });
        }

        let bandpass = if dt > 0.0 {
            let hp = high_pass(
                sample.vertical_accel_mps2,
                self.hp_last_input,
                self.hp_last_output,
                self.cfg.high_pass_cutoff_hz,
                dt,
            );
            self.hp_last_input = sample.vertical_accel_mps2;
            self.hp_last_output = hp;
            self.lp_output = low_pass(self.lp_output, hp, self.cfg.low_pass_cutoff_hz, dt);
            self.lp_output
        } else {
            self.lp_output
        };

        let speed_mps = sample.speed_mps.max(0.0);
        let ds = speed_mps * dt;
        let updated = speed_mps >= self.cfg.min_speed_mps && ds > 0.0;
        let abs_bandpass = bandpass.abs();
        let robust = if updated {
            let cap = self.robust_cap();
            robust_limit(bandpass, cap)
        } else {
            0.0
        };
        let shock_event = if updated {
            self.update_shock(sample.t_s, dt, abs_bandpass, speed_mps)
        } else {
            self.finish_active_shock(false)
        };
        if updated {
            self.distance_m += ds;
            let robust_abs = robust.abs();
            self.update_baseline(robust_abs, ds);
            let energy_sample = robust_abs * robust_abs;
            if self.energy_initialized {
                let tau_m = self.cfg.distance_tau_m.max(ds);
                let alpha = ds / (tau_m + ds);
                self.energy_mps2_sq = (1.0 - alpha) * self.energy_mps2_sq + alpha * energy_sample;
            } else {
                self.energy_initialized = true;
                self.energy_mps2_sq = energy_sample;
            }
        }

        let estimate = self.make_estimate(sample.t_s, bandpass, robust, updated);
        let (roughness_event, completed_roughness_event) = if updated {
            self.update_roughness_event(estimate, dt, speed_mps, ds)
        } else {
            (None, self.finish_active_roughness(false))
        };

        Some(RoadRoughnessUpdate {
            estimate,
            roughness_event,
            completed_roughness_event,
            shock_event,
        })
    }

    pub fn estimate(&self) -> RoadRoughnessEstimate {
        self.last_estimate
    }

    pub fn finish(&mut self) -> Option<RoadRoughnessEvent> {
        self.finish_active_roughness(true)
    }

    fn make_estimate(
        &mut self,
        t_s: f32,
        bandpass: f32,
        clipped: f32,
        updated: bool,
    ) -> RoadRoughnessEstimate {
        let roughness_rms_mps2 = sqrt_f32(self.energy_mps2_sq.max(0.0));
        let estimate = RoadRoughnessEstimate {
            t_s,
            roughness_rms_mps2,
            level: level_for_rms(self.cfg, roughness_rms_mps2),
            vertical_accel_bandpass_mps2: bandpass,
            vertical_accel_clipped_mps2: clipped,
            distance_m: self.distance_m,
            updated,
        };
        self.last_estimate = estimate;
        estimate
    }

    fn robust_cap(&self) -> f32 {
        let baseline_cap = if self.baseline_initialized {
            self.cfg.robust_cap_scale.max(0.0) * self.baseline_abs_mps2
        } else {
            self.cfg.robust_min_cap_mps2
        };
        baseline_cap
            .max(self.cfg.robust_min_cap_mps2)
            .min(self.cfg.clip_mps2)
            .max(1.0e-3)
    }

    fn update_baseline(&mut self, robust_abs_mps2: f32, ds: f32) {
        if self.baseline_initialized {
            self.baseline_abs_mps2 = distance_ema(
                self.baseline_abs_mps2,
                robust_abs_mps2,
                self.cfg.robust_baseline_tau_m,
                ds,
            );
        } else {
            self.baseline_initialized = true;
            self.baseline_abs_mps2 = robust_abs_mps2;
        }
    }

    fn update_shock(
        &mut self,
        t_s: f32,
        dt: f32,
        abs_bandpass_mps2: f32,
        speed_mps: f32,
    ) -> Option<RoadShockEvent> {
        let baseline_threshold = if self.baseline_initialized {
            self.cfg.shock_baseline_scale.max(0.0) * self.baseline_abs_mps2.max(0.0)
        } else {
            0.0
        };
        let enter_threshold = self.cfg.shock_min_peak_mps2.max(baseline_threshold);
        let exit_threshold = enter_threshold * self.cfg.shock_exit_fraction.clamp(0.05, 0.95);
        let above_enter = abs_bandpass_mps2 >= enter_threshold;
        let above_exit = abs_bandpass_mps2 >= exit_threshold;

        match self.shock_active.take() {
            Some(mut active) if above_exit => {
                active.add_sample(t_s, dt, abs_bandpass_mps2, speed_mps);
                self.shock_active = Some(active);
                None
            }
            Some(active) => self.finish_shock(active, true),
            None if above_enter
                && t_s - self.last_shock_event_t_s >= self.cfg.shock_refractory_s =>
            {
                self.shock_active = Some(ActiveShock::new(t_s, abs_bandpass_mps2, speed_mps));
                None
            }
            None => None,
        }
    }

    fn finish_active_shock(&mut self, emit: bool) -> Option<RoadShockEvent> {
        self.shock_active
            .take()
            .and_then(|active| self.finish_shock(active, emit))
    }

    fn finish_shock(&mut self, active: ActiveShock, emit: bool) -> Option<RoadShockEvent> {
        if !emit
            || active.duration_s < self.cfg.shock_min_duration_s
            || active.duration_s > self.cfg.shock_max_duration_s
        {
            return None;
        }
        self.last_shock_event_t_s = active.last_t_s;
        Some(active.event())
    }

    fn update_roughness_event(
        &mut self,
        estimate: RoadRoughnessEstimate,
        dt: f32,
        speed_mps: f32,
        ds: f32,
    ) -> (Option<RoadRoughnessEvent>, Option<RoadRoughnessEvent>) {
        let above_enter = estimate.roughness_rms_mps2 >= self.cfg.rough_event_enter_mps2;
        let above_exit = estimate.roughness_rms_mps2 >= self.cfg.rough_event_exit_mps2;
        match self.rough_active.take() {
            Some(mut active) if above_exit => {
                active.add_sample(estimate.t_s, dt, estimate.roughness_rms_mps2, speed_mps, ds);
                let event = if !active.emitted
                    && active.duration_s >= self.cfg.rough_event_min_duration_s
                {
                    active.emitted = true;
                    self.last_rough_event_t_s = active.last_t_s;
                    Some(active.event())
                } else {
                    None
                };
                self.rough_active = Some(active);
                (event, None)
            }
            Some(active) => (None, self.finish_roughness(active, true)),
            None if above_enter
                && estimate.t_s - self.last_rough_event_t_s
                    >= self.cfg.rough_event_refractory_s =>
            {
                self.rough_active = Some(ActiveRoughness::new(
                    estimate.t_s,
                    estimate.roughness_rms_mps2,
                    speed_mps,
                ));
                (None, None)
            }
            None => (None, None),
        }
    }

    fn finish_active_roughness(&mut self, emit: bool) -> Option<RoadRoughnessEvent> {
        self.rough_active
            .take()
            .and_then(|active| self.finish_roughness(active, emit))
    }

    fn finish_roughness(
        &mut self,
        active: ActiveRoughness,
        emit: bool,
    ) -> Option<RoadRoughnessEvent> {
        if !emit || active.duration_s < self.cfg.rough_event_min_duration_s {
            return None;
        }
        self.last_rough_event_t_s = active.last_t_s;
        Some(active.event())
    }
}

impl Default for RoadRoughnessAnalyzer {
    fn default() -> Self {
        Self::new(RoadRoughnessConfig::default())
    }
}

fn level_for_rms(cfg: RoadRoughnessConfig, rms_mps2: f32) -> RoadRoughnessLevel {
    if rms_mps2 < cfg.very_smooth_threshold_mps2 {
        RoadRoughnessLevel::VerySmooth
    } else if rms_mps2 < cfg.smooth_threshold_mps2 {
        RoadRoughnessLevel::Smooth
    } else if rms_mps2 < cfg.light_texture_threshold_mps2 {
        RoadRoughnessLevel::LightTexture
    } else if rms_mps2 < cfg.moderate_threshold_mps2 {
        RoadRoughnessLevel::Moderate
    } else if rms_mps2 < cfg.rough_threshold_mps2 {
        RoadRoughnessLevel::Rough
    } else if rms_mps2 < cfg.very_rough_threshold_mps2 {
        RoadRoughnessLevel::VeryRough
    } else {
        RoadRoughnessLevel::Severe
    }
}

fn valid_sample(sample: RoadRoughnessSample) -> bool {
    sample.t_s.is_finite() && sample.speed_mps.is_finite() && sample.vertical_accel_mps2.is_finite()
}

impl ActiveRoughness {
    fn new(t_s: f32, roughness_rms_mps2: f32, _speed_mps: f32) -> Self {
        Self {
            start_t_s: t_s,
            last_t_s: t_s,
            duration_s: 0.0,
            roughness_time_sum: 0.0,
            peak_roughness_rms_mps2: roughness_rms_mps2,
            speed_time_sum_m: 0.0,
            distance_m: 0.0,
            emitted: false,
        }
    }

    fn add_sample(&mut self, t_s: f32, dt: f32, roughness_rms_mps2: f32, speed_mps: f32, ds: f32) {
        self.last_t_s = t_s;
        self.duration_s += dt;
        self.roughness_time_sum += roughness_rms_mps2 * dt;
        self.peak_roughness_rms_mps2 = self.peak_roughness_rms_mps2.max(roughness_rms_mps2);
        self.speed_time_sum_m += speed_mps * dt;
        self.distance_m += ds;
    }

    fn event(self) -> RoadRoughnessEvent {
        RoadRoughnessEvent {
            start_t_s: self.start_t_s,
            end_t_s: self.last_t_s,
            duration_s: self.duration_s,
            mean_roughness_rms_mps2: ratio_or_zero(self.roughness_time_sum, self.duration_s),
            peak_roughness_rms_mps2: self.peak_roughness_rms_mps2,
            mean_speed_mps: ratio_or_zero(self.speed_time_sum_m, self.duration_s),
            distance_m: self.distance_m,
        }
    }
}

impl ActiveShock {
    fn new(t_s: f32, abs_vertical_accel_mps2: f32, _speed_mps: f32) -> Self {
        Self {
            start_t_s: t_s,
            last_t_s: t_s,
            duration_s: 0.0,
            peak_abs_vertical_accel_mps2: abs_vertical_accel_mps2,
            speed_time_sum_m: 0.0,
        }
    }

    fn add_sample(&mut self, t_s: f32, dt: f32, abs_vertical_accel_mps2: f32, speed_mps: f32) {
        self.last_t_s = t_s;
        self.duration_s += dt;
        self.peak_abs_vertical_accel_mps2 = self
            .peak_abs_vertical_accel_mps2
            .max(abs_vertical_accel_mps2);
        self.speed_time_sum_m += speed_mps * dt;
    }

    fn event(self) -> RoadShockEvent {
        RoadShockEvent {
            start_t_s: self.start_t_s,
            end_t_s: self.last_t_s,
            duration_s: self.duration_s,
            peak_abs_vertical_accel_mps2: self.peak_abs_vertical_accel_mps2,
            mean_speed_mps: ratio_or_zero(self.speed_time_sum_m, self.duration_s),
        }
    }
}

fn robust_limit(value: f32, cap: f32) -> f32 {
    let cap = cap.max(1.0e-3);
    value.clamp(-cap, cap)
}

fn distance_ema(previous: f32, value: f32, tau_m: f32, ds: f32) -> f32 {
    let alpha = ds / (tau_m.max(ds) + ds);
    (1.0 - alpha) * previous + alpha * value
}

fn ratio_or_zero(num: f32, den: f32) -> f32 {
    if den > 0.0 { num / den } else { 0.0 }
}
