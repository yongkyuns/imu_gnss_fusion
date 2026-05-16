use crate::common::{elapsed_since_last, update_ema};
use crate::{
    HarshAccelConfig, HarshBrakeConfig, HarshCornerConfig, HarshCornerEvent, HarshCornerSample,
    HarshLongitudinalEvent, HarshLongitudinalSample,
};

#[derive(Clone, Copy, Debug)]
struct LongitudinalDerivativeEma {
    last_t_s: Option<f32>,
    last_forward_velocity_mps: f32,
    accel_ema_mps2: f32,
    initialized: bool,
}

#[derive(Clone, Copy, Debug)]
struct SmoothedLongitudinalAccel {
    accel_mps2: f32,
    speed_mps: f32,
}

#[derive(Clone, Copy, Debug)]
struct ActiveMetric {
    start_t_s: f32,
    last_t_s: f32,
    duration_s: f32,
    metric_time_sum: f32,
    peak_metric: f32,
    speed_time_sum_m: f32,
    peak_speed_mps: f32,
    start_velocity_mps: f32,
    end_velocity_mps: f32,
}

#[derive(Clone, Debug)]
struct MetricIntervalTracker {
    active: Option<ActiveMetric>,
    last_t_s: Option<f32>,
    last_event_t_s: f32,
}

#[derive(Clone, Copy)]
struct MetricIntervalConfig {
    enter_threshold: f32,
    exit_threshold: f32,
    min_duration_s: f32,
    refractory_s: f32,
}

impl LongitudinalDerivativeEma {
    fn new() -> Self {
        Self {
            last_t_s: None,
            last_forward_velocity_mps: 0.0,
            accel_ema_mps2: 0.0,
            initialized: false,
        }
    }

    fn update(
        &mut self,
        sample: HarshLongitudinalSample,
        tau_s: f32,
        max_raw_accel_mps2: f32,
    ) -> Option<SmoothedLongitudinalAccel> {
        let Some(last_t_s) = self.last_t_s else {
            self.last_t_s = Some(sample.t_s);
            self.last_forward_velocity_mps = sample.forward_velocity_mps;
            return None;
        };
        let dt = (sample.t_s - last_t_s).clamp(0.0, 0.2);
        self.last_t_s = Some(sample.t_s);
        let speed_mps = sample
            .forward_velocity_mps
            .abs()
            .max(self.last_forward_velocity_mps.abs());
        if dt <= 1.0e-4 {
            self.last_forward_velocity_mps = sample.forward_velocity_mps;
            return None;
        }
        let raw_accel_mps2 = ((sample.forward_velocity_mps - self.last_forward_velocity_mps) / dt)
            .clamp(-max_raw_accel_mps2, max_raw_accel_mps2);
        self.last_forward_velocity_mps = sample.forward_velocity_mps;
        self.accel_ema_mps2 = if self.initialized {
            update_ema(self.accel_ema_mps2, raw_accel_mps2, tau_s, dt)
        } else {
            self.initialized = true;
            raw_accel_mps2
        };
        Some(SmoothedLongitudinalAccel {
            accel_mps2: self.accel_ema_mps2,
            speed_mps,
        })
    }
}

impl ActiveMetric {
    fn new(t_s: f32, metric: f32, speed_mps: f32, velocity_mps: f32) -> Self {
        Self {
            start_t_s: t_s,
            last_t_s: t_s,
            duration_s: 0.0,
            metric_time_sum: 0.0,
            peak_metric: metric,
            speed_time_sum_m: 0.0,
            peak_speed_mps: speed_mps,
            start_velocity_mps: velocity_mps,
            end_velocity_mps: velocity_mps,
        }
    }

    fn add_sample(&mut self, t_s: f32, dt: f32, metric: f32, speed_mps: f32, velocity_mps: f32) {
        self.last_t_s = t_s;
        self.duration_s += dt;
        self.metric_time_sum += metric * dt;
        self.peak_metric = self.peak_metric.max(metric);
        self.speed_time_sum_m += speed_mps * dt;
        self.peak_speed_mps = self.peak_speed_mps.max(speed_mps);
        self.end_velocity_mps = velocity_mps;
    }

    fn mean_metric(&self) -> f32 {
        if self.duration_s > 0.0 {
            self.metric_time_sum / self.duration_s
        } else {
            0.0
        }
    }

    fn mean_speed_mps(&self) -> f32 {
        if self.duration_s > 0.0 {
            self.speed_time_sum_m / self.duration_s
        } else {
            0.0
        }
    }
}

impl MetricIntervalTracker {
    fn new() -> Self {
        Self {
            active: None,
            last_t_s: None,
            last_event_t_s: -1.0e9,
        }
    }

    fn update(
        &mut self,
        t_s: f32,
        metric: f32,
        speed_mps: f32,
        velocity_mps: f32,
        cfg: MetricIntervalConfig,
    ) -> Option<ActiveMetric> {
        let dt = elapsed_since_last(&mut self.last_t_s, t_s);
        let above_enter = metric >= cfg.enter_threshold;
        let above_exit = metric >= cfg.exit_threshold;
        match self.active.take() {
            Some(mut active) if above_exit => {
                active.add_sample(t_s, dt, metric, speed_mps, velocity_mps);
                self.active = Some(active);
                None
            }
            Some(active) => self.finish_active(active, cfg),
            None if above_enter && t_s - self.last_event_t_s >= cfg.refractory_s => {
                self.active = Some(ActiveMetric::new(t_s, metric, speed_mps, velocity_mps));
                None
            }
            None => None,
        }
    }

    fn finish(&mut self, cfg: MetricIntervalConfig) -> Option<ActiveMetric> {
        self.active
            .take()
            .and_then(|active| self.finish_active(active, cfg))
    }

    fn finish_active(
        &mut self,
        active: ActiveMetric,
        cfg: MetricIntervalConfig,
    ) -> Option<ActiveMetric> {
        if active.duration_s >= cfg.min_duration_s {
            self.last_event_t_s = active.last_t_s;
            Some(active)
        } else {
            None
        }
    }
}

/// Streaming detector for harsh positive longitudinal acceleration.
#[derive(Clone, Debug)]
pub struct HarshAccelDetector {
    cfg: HarshAccelConfig,
    accel: LongitudinalDerivativeEma,
    tracker: MetricIntervalTracker,
}

impl HarshAccelDetector {
    pub fn new(cfg: HarshAccelConfig) -> Self {
        Self {
            cfg,
            accel: LongitudinalDerivativeEma::new(),
            tracker: MetricIntervalTracker::new(),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }

    pub fn update(&mut self, sample: HarshLongitudinalSample) -> Option<HarshLongitudinalEvent> {
        if !valid_longitudinal_sample(sample) {
            return None;
        }
        let accel = self
            .accel
            .update(sample, self.cfg.accel_tau_s, self.cfg.max_raw_accel_mps2)?;
        if accel.speed_mps < self.cfg.min_speed_mps {
            return self
                .tracker
                .update(
                    sample.t_s,
                    0.0,
                    accel.speed_mps,
                    sample.forward_velocity_mps,
                    self.metric_cfg(),
                )
                .map(longitudinal_event);
        }
        self.tracker
            .update(
                sample.t_s,
                accel.accel_mps2.max(0.0),
                accel.speed_mps,
                sample.forward_velocity_mps,
                self.metric_cfg(),
            )
            .map(longitudinal_event)
    }

    pub fn finish(&mut self) -> Option<HarshLongitudinalEvent> {
        self.tracker
            .finish(self.metric_cfg())
            .map(longitudinal_event)
    }

    fn metric_cfg(&self) -> MetricIntervalConfig {
        MetricIntervalConfig {
            enter_threshold: self.cfg.accel_threshold_mps2,
            exit_threshold: self.cfg.exit_accel_threshold_mps2,
            min_duration_s: self.cfg.min_duration_s,
            refractory_s: self.cfg.refractory_s,
        }
    }
}

/// Streaming detector for harsh braking based on an EMA-smoothed velocity derivative.
#[derive(Clone, Debug)]
pub struct HarshBrakeDetector {
    cfg: HarshBrakeConfig,
    accel: LongitudinalDerivativeEma,
    tracker: MetricIntervalTracker,
}

impl HarshBrakeDetector {
    pub fn new(cfg: HarshBrakeConfig) -> Self {
        Self {
            cfg,
            accel: LongitudinalDerivativeEma::new(),
            tracker: MetricIntervalTracker::new(),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }

    pub fn update(&mut self, sample: HarshLongitudinalSample) -> Option<HarshLongitudinalEvent> {
        if !valid_longitudinal_sample(sample) {
            return None;
        }
        let accel = self
            .accel
            .update(sample, self.cfg.accel_tau_s, self.cfg.max_raw_accel_mps2)?;
        if accel.speed_mps < self.cfg.min_speed_mps {
            return self
                .tracker
                .update(
                    sample.t_s,
                    0.0,
                    accel.speed_mps,
                    sample.forward_velocity_mps,
                    self.metric_cfg(),
                )
                .map(longitudinal_event);
        }
        self.tracker
            .update(
                sample.t_s,
                (-accel.accel_mps2).max(0.0),
                accel.speed_mps,
                sample.forward_velocity_mps,
                self.metric_cfg(),
            )
            .map(longitudinal_event)
    }

    pub fn finish(&mut self) -> Option<HarshLongitudinalEvent> {
        self.tracker
            .finish(self.metric_cfg())
            .map(longitudinal_event)
    }

    fn metric_cfg(&self) -> MetricIntervalConfig {
        MetricIntervalConfig {
            enter_threshold: self.cfg.decel_threshold_mps2,
            exit_threshold: self.cfg.exit_decel_threshold_mps2,
            min_duration_s: self.cfg.min_duration_s,
            refractory_s: self.cfg.refractory_s,
        }
    }
}

/// Streaming detector for harsh cornering from `abs(yaw_rate * speed)`.
#[derive(Clone, Debug)]
pub struct HarshCornerDetector {
    cfg: HarshCornerConfig,
    tracker: MetricIntervalTracker,
}

impl HarshCornerDetector {
    pub fn new(cfg: HarshCornerConfig) -> Self {
        Self {
            cfg,
            tracker: MetricIntervalTracker::new(),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }

    pub fn update(&mut self, sample: HarshCornerSample) -> Option<HarshCornerEvent> {
        if !sample.t_s.is_finite()
            || !sample.speed_mps.is_finite()
            || !sample.yaw_rate_radps.is_finite()
        {
            return None;
        }
        let speed_mps = sample.speed_mps.abs();
        let lateral_accel_mps2 = if speed_mps >= self.cfg.min_speed_mps {
            (sample.yaw_rate_radps * speed_mps).abs()
        } else {
            0.0
        };
        self.tracker
            .update(
                sample.t_s,
                lateral_accel_mps2,
                speed_mps,
                speed_mps,
                self.metric_cfg(),
            )
            .map(corner_event)
    }

    pub fn finish(&mut self) -> Option<HarshCornerEvent> {
        self.tracker.finish(self.metric_cfg()).map(corner_event)
    }

    fn metric_cfg(&self) -> MetricIntervalConfig {
        MetricIntervalConfig {
            enter_threshold: self.cfg.lateral_accel_threshold_mps2,
            exit_threshold: self.cfg.exit_lateral_accel_threshold_mps2,
            min_duration_s: self.cfg.min_duration_s,
            refractory_s: self.cfg.refractory_s,
        }
    }
}

fn valid_longitudinal_sample(sample: HarshLongitudinalSample) -> bool {
    sample.t_s.is_finite() && sample.forward_velocity_mps.is_finite()
}

fn longitudinal_event(active: ActiveMetric) -> HarshLongitudinalEvent {
    HarshLongitudinalEvent {
        start_t_s: active.start_t_s,
        end_t_s: active.last_t_s,
        duration_s: active.duration_s,
        delta_velocity_mps: active.end_velocity_mps - active.start_velocity_mps,
        mean_accel_mps2: active.mean_metric(),
        peak_accel_mps2: active.peak_metric,
        mean_speed_mps: active.mean_speed_mps(),
        peak_speed_mps: active.peak_speed_mps,
    }
}

fn corner_event(active: ActiveMetric) -> HarshCornerEvent {
    HarshCornerEvent {
        start_t_s: active.start_t_s,
        end_t_s: active.last_t_s,
        duration_s: active.duration_s,
        mean_lateral_accel_mps2: active.mean_metric(),
        peak_lateral_accel_mps2: active.peak_metric,
        mean_speed_mps: active.mean_speed_mps(),
        peak_speed_mps: active.peak_speed_mps,
    }
}
