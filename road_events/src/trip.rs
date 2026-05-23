use crate::common::{update_abs_ema, update_ema};
use crate::{TripConfig, TripEventCounts, TripEventKind, TripSample, TripSummary};

/// Constant-memory streaming trip-statistics accumulator.
///
/// The accumulator keeps only sums, extrema, event counters, and EMA-style
/// rolling motion estimates. It is intended to run beside the detectors on an
/// embedded target without retaining a sample or event buffer.
#[derive(Clone, Copy, Debug)]
pub struct TripStats {
    cfg: TripConfig,
    last_sample: Option<TripSample>,
    sample_count: u32,
    invalid_sample_count: u32,
    data_gap_count: u32,
    max_sample_gap_s: f32,
    total_gap_duration_s: f32,
    duration_s: f32,
    moving_duration_s: f32,
    distance_m: f32,
    reverse_duration_s: f32,
    reverse_distance_m: f32,
    elevation_gain_m: f32,
    elevation_loss_m: f32,
    elevation_valid: bool,
    speed_time_sum_m: f32,
    moving_speed_time_sum_m: f32,
    peak_speed_mps: f32,
    peak_accel_mps2: f32,
    peak_decel_mps2: f32,
    peak_lateral_accel_mps2: f32,
    rolling_speed_mps: f32,
    rolling_abs_longitudinal_accel_mps2: f32,
    rolling_abs_lateral_accel_mps2: f32,
    rolling_initialized: bool,
    events: TripEventCounts,
}

impl TripStats {
    pub fn new(cfg: TripConfig) -> Self {
        Self {
            cfg,
            last_sample: None,
            sample_count: 0,
            invalid_sample_count: 0,
            data_gap_count: 0,
            max_sample_gap_s: 0.0,
            total_gap_duration_s: 0.0,
            duration_s: 0.0,
            moving_duration_s: 0.0,
            distance_m: 0.0,
            reverse_duration_s: 0.0,
            reverse_distance_m: 0.0,
            elevation_gain_m: 0.0,
            elevation_loss_m: 0.0,
            elevation_valid: false,
            speed_time_sum_m: 0.0,
            moving_speed_time_sum_m: 0.0,
            peak_speed_mps: 0.0,
            peak_accel_mps2: 0.0,
            peak_decel_mps2: 0.0,
            peak_lateral_accel_mps2: 0.0,
            rolling_speed_mps: 0.0,
            rolling_abs_longitudinal_accel_mps2: 0.0,
            rolling_abs_lateral_accel_mps2: 0.0,
            rolling_initialized: false,
            events: TripEventCounts::default(),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }

    pub fn update_motion(&mut self, sample: TripSample) {
        if !valid_sample(sample) {
            self.invalid_sample_count = self.invalid_sample_count.saturating_add(1);
            return;
        }

        self.sample_count = self.sample_count.saturating_add(1);
        self.update_accel_extrema(sample);
        let Some(prev) = self.last_sample else {
            self.last_sample = Some(sample);
            self.peak_speed_mps = self.peak_speed_mps.max(sample.speed_mps.max(0.0));
            return;
        };

        let raw_dt = (sample.t_s - prev.t_s).max(0.0);
        let max_dt = self.cfg.max_integrated_dt_s.max(0.0);
        if raw_dt > max_dt {
            self.data_gap_count = self.data_gap_count.saturating_add(1);
            self.total_gap_duration_s += raw_dt - max_dt;
        }
        self.max_sample_gap_s = self.max_sample_gap_s.max(raw_dt);
        self.last_sample = Some(sample);
        let dt = raw_dt.min(max_dt);

        let speed_mps = 0.5 * (prev.speed_mps.max(0.0) + sample.speed_mps.max(0.0));
        let forward_velocity_mps = 0.5 * (prev.forward_velocity_mps + sample.forward_velocity_mps);
        let moving = speed_mps >= self.cfg.moving_speed_threshold_mps;

        self.duration_s += dt;
        self.distance_m += speed_mps * dt;
        self.speed_time_sum_m += speed_mps * dt;
        self.peak_speed_mps = self.peak_speed_mps.max(speed_mps);

        if moving {
            self.moving_duration_s += dt;
            self.moving_speed_time_sum_m += speed_mps * dt;
        }

        let reverse_speed_mps = (-forward_velocity_mps).max(0.0);
        if reverse_speed_mps >= self.cfg.reverse_speed_threshold_mps {
            self.reverse_duration_s += dt;
        }
        self.reverse_distance_m += reverse_speed_mps * dt;
        if raw_dt <= max_dt {
            self.integrate_vertical_position(
                prev.height_m,
                prev.height_frame_id,
                sample.height_m,
                sample.height_frame_id,
            );
        }
        self.update_rolling(sample, speed_mps, dt);
    }

    pub fn record_event(&mut self, kind: TripEventKind) {
        match kind {
            TripEventKind::SpeedBump => {
                self.events.speed_bumps = self.events.speed_bumps.saturating_add(1)
            }
            TripEventKind::Uphill => self.events.uphill = self.events.uphill.saturating_add(1),
            TripEventKind::Downhill => {
                self.events.downhill = self.events.downhill.saturating_add(1)
            }
            TripEventKind::Reverse => self.events.reverse = self.events.reverse.saturating_add(1),
            TripEventKind::HarshAcceleration => {
                self.events.harsh_acceleration = self.events.harsh_acceleration.saturating_add(1)
            }
            TripEventKind::HarshBraking => {
                self.events.harsh_braking = self.events.harsh_braking.saturating_add(1)
            }
            TripEventKind::HarshCornering => {
                self.events.harsh_cornering = self.events.harsh_cornering.saturating_add(1)
            }
        }
    }

    pub fn summary(&self) -> TripSummary {
        let distance_km = self.distance_m / 1000.0;
        let harsh_count = self.events.harsh_acceleration
            + self.events.harsh_braking
            + self.events.harsh_cornering;
        TripSummary {
            sample_count: self.sample_count,
            invalid_sample_count: self.invalid_sample_count,
            data_gap_count: self.data_gap_count,
            max_sample_gap_s: self.max_sample_gap_s,
            total_gap_duration_s: self.total_gap_duration_s,
            duration_s: self.duration_s,
            moving_duration_s: self.moving_duration_s,
            stationary_duration_s: (self.duration_s - self.moving_duration_s).max(0.0),
            distance_m: self.distance_m,
            reverse_duration_s: self.reverse_duration_s,
            reverse_distance_m: self.reverse_distance_m,
            elevation_gain_m: self.elevation_gain_m,
            elevation_loss_m: self.elevation_loss_m,
            elevation_valid: self.elevation_valid,
            mean_speed_mps: ratio_or_zero(self.speed_time_sum_m, self.duration_s),
            moving_mean_speed_mps: ratio_or_zero(
                self.moving_speed_time_sum_m,
                self.moving_duration_s,
            ),
            peak_speed_mps: self.peak_speed_mps,
            peak_accel_mps2: self.peak_accel_mps2,
            peak_decel_mps2: self.peak_decel_mps2,
            peak_lateral_accel_mps2: self.peak_lateral_accel_mps2,
            rolling_speed_mps: self.rolling_speed_mps,
            rolling_abs_longitudinal_accel_mps2: self.rolling_abs_longitudinal_accel_mps2,
            rolling_abs_lateral_accel_mps2: self.rolling_abs_lateral_accel_mps2,
            events: self.events,
            speed_bumps_per_km: per_km(self.events.speed_bumps, distance_km),
            harsh_events_per_km: per_km(harsh_count, distance_km),
            reverse_seconds_per_km: ratio_or_zero(self.reverse_duration_s, distance_km),
        }
    }

    fn integrate_vertical_position(
        &mut self,
        prev_height_m: Option<f32>,
        prev_height_frame_id: u32,
        height_m: Option<f32>,
        height_frame_id: u32,
    ) {
        if prev_height_frame_id != height_frame_id {
            return;
        }
        let (Some(prev_height_m), Some(height_m)) = (prev_height_m, height_m) else {
            return;
        };
        if !prev_height_m.is_finite() || !height_m.is_finite() {
            return;
        }
        let vertical_delta_m = height_m - prev_height_m;
        self.elevation_valid = true;
        if vertical_delta_m >= 0.0 {
            self.elevation_gain_m += vertical_delta_m;
        } else {
            self.elevation_loss_m += -vertical_delta_m;
        }
    }

    fn update_accel_extrema(&mut self, sample: TripSample) {
        self.peak_accel_mps2 = self
            .peak_accel_mps2
            .max(sample.longitudinal_accel_mps2.max(0.0));
        self.peak_decel_mps2 = self
            .peak_decel_mps2
            .max((-sample.longitudinal_accel_mps2).max(0.0));
        self.peak_lateral_accel_mps2 = self
            .peak_lateral_accel_mps2
            .max(sample.lateral_accel_mps2.abs());
    }

    fn update_rolling(&mut self, sample: TripSample, speed_mps: f32, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        if !self.rolling_initialized {
            self.rolling_initialized = true;
            self.rolling_speed_mps = speed_mps;
            self.rolling_abs_longitudinal_accel_mps2 = sample.longitudinal_accel_mps2.abs();
            self.rolling_abs_lateral_accel_mps2 = sample.lateral_accel_mps2.abs();
            return;
        }
        self.rolling_speed_mps = update_ema(
            self.rolling_speed_mps,
            speed_mps,
            self.cfg.rolling_tau_s,
            dt,
        );
        self.rolling_abs_longitudinal_accel_mps2 = update_abs_ema(
            self.rolling_abs_longitudinal_accel_mps2,
            sample.longitudinal_accel_mps2,
            self.cfg.rolling_tau_s,
            dt,
        );
        self.rolling_abs_lateral_accel_mps2 = update_abs_ema(
            self.rolling_abs_lateral_accel_mps2,
            sample.lateral_accel_mps2,
            self.cfg.rolling_tau_s,
            dt,
        );
    }
}

impl Default for TripStats {
    fn default() -> Self {
        Self::new(TripConfig::default())
    }
}

fn valid_sample(sample: TripSample) -> bool {
    sample.t_s.is_finite()
        && sample.speed_mps.is_finite()
        && sample.forward_velocity_mps.is_finite()
        && match sample.height_m {
            Some(height_m) => height_m.is_finite(),
            None => true,
        }
        && sample.longitudinal_accel_mps2.is_finite()
        && sample.lateral_accel_mps2.is_finite()
}

fn ratio_or_zero(num: f32, den: f32) -> f32 {
    if den > 0.0 { num / den } else { 0.0 }
}

fn per_km(count: u32, distance_km: f32) -> f32 {
    if distance_km > 0.0 {
        count as f32 / distance_km
    } else {
        0.0
    }
}
