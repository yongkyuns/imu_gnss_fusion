//! Runtime health diagnostics for the high-level sensor-fusion facade.
//!
//! The diagnostics intentionally live below app/UI code. They answer two
//! operational questions from filter state and input history:
//!
//! - is the fusion state usable now?
//! - has it converged enough that mount/bias priors are worth persisting?

use crate::ekf::State;
use crate::math::sqrt_f32;

const TAIL_CAP: usize = 180;
const SAMPLE_PERIOD_S: f32 = 1.0;
const GNSS_STALE_S: f32 = 5.0;
const MIN_STABLE_POST_INIT_S: f32 = 180.0;
const MIN_STABLE_DISTANCE_M: f32 = 750.0;
const MIN_STABLE_TAIL_S: f32 = 90.0;
const MIN_STABLE_SAMPLES: usize = 30;
const MAX_MOUNT_DRIFT_DEG: f32 = 0.50;
const MAX_MOUNT_STD_DEG: f32 = 0.35;
const MAX_GYRO_BIAS_DRIFT_RADPS: f32 = 0.00035;
const MAX_GYRO_BIAS_STD_RADPS: f32 = 0.00020;
const MAX_ACCEL_BIAS_DRIFT_MPS2: f32 = 0.05;
const MAX_ACCEL_BIAS_STD_MPS2: f32 = 0.035;
const MAX_MOUNT_SIGMA_DEG: f32 = 2.0;
const MAX_ATTITUDE_SIGMA_DEG: f32 = 6.0;
const MAX_RECENT_GNSS_ISSUES: u32 = 12;
const DEG_PER_RAD: f32 = 180.0 / core::f32::consts::PI;

pub const FUSION_HEALTH_REASON_NOT_INITIALIZED: u32 = 1 << 0;
pub const FUSION_HEALTH_REASON_MOUNT_NOT_READY: u32 = 1 << 1;
pub const FUSION_HEALTH_REASON_GNSS_STALE: u32 = 1 << 2;
pub const FUSION_HEALTH_REASON_INSUFFICIENT_TIME: u32 = 1 << 3;
pub const FUSION_HEALTH_REASON_INSUFFICIENT_MOTION: u32 = 1 << 4;
pub const FUSION_HEALTH_REASON_TAIL_TOO_SHORT: u32 = 1 << 5;
pub const FUSION_HEALTH_REASON_MOUNT_UNSTABLE: u32 = 1 << 6;
pub const FUSION_HEALTH_REASON_BIAS_UNSTABLE: u32 = 1 << 7;
pub const FUSION_HEALTH_REASON_COVARIANCE_HIGH: u32 = 1 << 8;
pub const FUSION_HEALTH_REASON_GNSS_REJECTING: u32 = 1 << 9;
pub const FUSION_HEALTH_REASON_NUMERIC_INVALID: u32 = 1 << 10;
pub const FUSION_HEALTH_REASON_SLEEP_GAP: u32 = 1 << 11;
pub const FUSION_HEALTH_REASON_NAV_UNUSABLE: u32 = 1 << 12;

/// Single public lifecycle state for [`crate::SensorFusion`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FusionState {
    /// No usable filter state exists yet.
    #[default]
    NotReady = 0,
    /// Inputs are arriving and alignment/initialization is still in progress.
    Initializing = 1,
    /// Navigation is usable, but not yet stable enough to persist priors.
    Running = 2,
    /// Navigation is usable and invariant states are stable enough to persist.
    Stable = 3,
    /// Navigation is usable, but current input/state diagnostics are unhealthy.
    Degraded = 4,
    /// Navigation is usable but degraded while dead-reckoning after a gap.
    DegradedDeadReckoning = 5,
    /// Calibration priors are retained, but navigation must be reseeded by GNSS.
    AwaitingGnssReseed = 6,
}

/// Scalar diagnostics attached to [`FusionHealth`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FusionHealthMetrics {
    pub post_init_time_s: f32,
    pub distance_m: f32,
    pub mean_speed_mps: f32,
    pub tail_duration_s: f32,
    pub tail_samples: u32,
    pub mount_tail_drift_deg: f32,
    pub mount_tail_std_deg: f32,
    pub gyro_bias_tail_drift_radps: f32,
    pub gyro_bias_tail_std_radps: f32,
    pub accel_bias_tail_drift_mps2: f32,
    pub accel_bias_tail_std_mps2: f32,
    pub mount_sigma_max_deg: f32,
    pub attitude_sigma_max_deg: f32,
    pub recent_gnss_issue_count: u32,
}

/// Current runtime health and convergence verdict.
#[derive(Clone, Copy, Debug, Default)]
pub struct FusionHealth {
    pub state: FusionState,
    pub running: bool,
    pub stable: bool,
    pub degraded: bool,
    pub navigation_usable: bool,
    pub reason_mask: u32,
    pub metrics: FusionHealthMetrics,
}

#[derive(Clone, Copy, Debug, Default)]
struct HealthSample {
    t_s: f32,
    q_bv: [f32; 4],
    gyro_bias: [f32; 3],
    accel_bias: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FusionHealthMonitor {
    last_imu_t_s: Option<f32>,
    last_gnss_t_s: Option<f32>,
    last_gnss_distance_t_s: Option<f32>,
    init_t_s: Option<f32>,
    last_state_sample_t_s: Option<f32>,
    distance_m: f32,
    speed_sum_mps: f32,
    speed_count: u32,
    recent_gnss_issue_count: u32,
    last_gnss_issue_t_s: Option<f32>,
    state: FusionState,
    sleep_gap_active: bool,
    samples: [HealthSample; TAIL_CAP],
    sample_len: usize,
    sample_next: usize,
}

impl Default for FusionHealthMonitor {
    fn default() -> Self {
        Self {
            last_imu_t_s: None,
            last_gnss_t_s: None,
            last_gnss_distance_t_s: None,
            init_t_s: None,
            last_state_sample_t_s: None,
            distance_m: 0.0,
            speed_sum_mps: 0.0,
            speed_count: 0,
            recent_gnss_issue_count: 0,
            last_gnss_issue_t_s: None,
            state: FusionState::NotReady,
            sleep_gap_active: false,
            samples: [HealthSample::default(); TAIL_CAP],
            sample_len: 0,
            sample_next: 0,
        }
    }
}

impl FusionHealthMonitor {
    pub(crate) fn record_imu(&mut self, t_s: f32) {
        if t_s.is_finite() {
            self.last_imu_t_s = Some(t_s);
            self.decay_gnss_issue_count(t_s);
        }
    }

    pub(crate) fn record_gnss(&mut self, t_s: f32, vel_ned_mps: [f32; 3]) {
        if !t_s.is_finite() {
            return;
        }
        let speed_h = sqrt_f32(vel_ned_mps[0] * vel_ned_mps[0] + vel_ned_mps[1] * vel_ned_mps[1]);
        if speed_h.is_finite() {
            if let Some(last_t) = self.last_gnss_distance_t_s {
                let dt = t_s - last_t;
                if (0.0..=5.0).contains(&dt) {
                    self.distance_m += speed_h.max(0.0) * dt;
                    self.speed_sum_mps += speed_h.max(0.0);
                    self.speed_count = self.speed_count.saturating_add(1);
                }
            }
            self.last_gnss_distance_t_s = Some(t_s);
        }
        self.last_gnss_t_s = Some(t_s);
        self.decay_gnss_issue_count(t_s);
    }

    pub(crate) fn mark_initialized(&mut self, t_s: f32) {
        if t_s.is_finite() && self.init_t_s.is_none() {
            self.init_t_s = Some(t_s);
        }
        self.state = FusionState::Running;
        self.sleep_gap_active = false;
    }

    pub(crate) fn mark_initializing(&mut self) {
        self.state = FusionState::Initializing;
    }

    pub(crate) fn mark_degraded_dead_reckoning(&mut self) {
        self.state = FusionState::DegradedDeadReckoning;
        self.sleep_gap_active = true;
    }

    pub(crate) fn mark_awaiting_gnss_reseed(&mut self) {
        self.state = FusionState::AwaitingGnssReseed;
        self.sleep_gap_active = true;
    }

    pub(crate) fn mark_running(&mut self) {
        self.state = FusionState::Running;
        self.sleep_gap_active = false;
    }

    pub(crate) fn state(&self) -> FusionState {
        self.state
    }

    pub(crate) fn record_gnss_event_mask(&mut self, t_s: f32, mask: u32) {
        if mask == 0 || !t_s.is_finite() {
            return;
        }
        self.decay_gnss_issue_count(t_s);
        self.recent_gnss_issue_count = self
            .recent_gnss_issue_count
            .saturating_add(mask.count_ones());
        self.last_gnss_issue_t_s = Some(t_s);
    }

    pub(crate) fn record_state(&mut self, t_s: f32, state: &State) {
        if !t_s.is_finite() {
            return;
        }
        if let Some(last_t) = self.last_state_sample_t_s
            && t_s - last_t < SAMPLE_PERIOD_S
        {
            return;
        }
        self.last_state_sample_t_s = Some(t_s);
        self.samples[self.sample_next] = HealthSample {
            t_s,
            q_bv: [
                state.nominal.q_bv0,
                state.nominal.q_bv1,
                state.nominal.q_bv2,
                state.nominal.q_bv3,
            ],
            gyro_bias: [state.nominal.bgx, state.nominal.bgy, state.nominal.bgz],
            accel_bias: [state.nominal.bax, state.nominal.bay, state.nominal.baz],
        };
        self.sample_next = (self.sample_next + 1) % TAIL_CAP;
        self.sample_len = (self.sample_len + 1).min(TAIL_CAP);
    }

    pub(crate) fn health(
        &self,
        mount_ready: bool,
        ekf_initialized: bool,
        state: Option<&State>,
    ) -> FusionHealth {
        let now_t_s = self.last_imu_t_s.or(self.last_gnss_t_s).unwrap_or(0.0);
        let mut reason_mask = 0;
        if !ekf_initialized {
            reason_mask |= FUSION_HEALTH_REASON_NOT_INITIALIZED;
        }
        if !mount_ready {
            reason_mask |= FUSION_HEALTH_REASON_MOUNT_NOT_READY;
        }
        if let Some(last_gnss_t_s) = self.last_gnss_t_s {
            if now_t_s - last_gnss_t_s > GNSS_STALE_S {
                reason_mask |= FUSION_HEALTH_REASON_GNSS_STALE;
            }
        } else {
            reason_mask |= FUSION_HEALTH_REASON_GNSS_STALE;
        }

        let metrics = self.metrics(now_t_s, state);
        if !metrics_finite(metrics) || state.is_some_and(|s| !state_finite(s)) {
            reason_mask |= FUSION_HEALTH_REASON_NUMERIC_INVALID;
        }
        if metrics.recent_gnss_issue_count > MAX_RECENT_GNSS_ISSUES {
            reason_mask |= FUSION_HEALTH_REASON_GNSS_REJECTING;
        }
        if self.sleep_gap_active {
            reason_mask |= FUSION_HEALTH_REASON_SLEEP_GAP;
        }
        if matches!(self.state, FusionState::AwaitingGnssReseed) {
            reason_mask |= FUSION_HEALTH_REASON_NAV_UNUSABLE;
        }
        if metrics.attitude_sigma_max_deg > MAX_ATTITUDE_SIGMA_DEG
            || metrics.mount_sigma_max_deg > MAX_MOUNT_SIGMA_DEG
        {
            reason_mask |= FUSION_HEALTH_REASON_COVARIANCE_HIGH;
        }

        let navigation_usable = ekf_initialized
            && mount_ready
            && !matches!(
                self.state,
                FusionState::NotReady | FusionState::Initializing | FusionState::AwaitingGnssReseed
            )
            && reason_mask & FUSION_HEALTH_REASON_NUMERIC_INVALID == 0;
        let running = ekf_initialized
            && mount_ready
            && matches!(self.state, FusionState::Running | FusionState::Stable)
            && reason_mask
                & (FUSION_HEALTH_REASON_NOT_INITIALIZED
                    | FUSION_HEALTH_REASON_MOUNT_NOT_READY
                    | FUSION_HEALTH_REASON_NUMERIC_INVALID)
                == 0;

        if metrics.post_init_time_s < MIN_STABLE_POST_INIT_S {
            reason_mask |= FUSION_HEALTH_REASON_INSUFFICIENT_TIME;
        }
        if metrics.distance_m < MIN_STABLE_DISTANCE_M {
            reason_mask |= FUSION_HEALTH_REASON_INSUFFICIENT_MOTION;
        }
        if metrics.tail_duration_s < MIN_STABLE_TAIL_S
            || metrics.tail_samples < MIN_STABLE_SAMPLES as u32
        {
            reason_mask |= FUSION_HEALTH_REASON_TAIL_TOO_SHORT;
        }
        if metrics.mount_tail_drift_deg > MAX_MOUNT_DRIFT_DEG
            || metrics.mount_tail_std_deg > MAX_MOUNT_STD_DEG
        {
            reason_mask |= FUSION_HEALTH_REASON_MOUNT_UNSTABLE;
        }
        if metrics.gyro_bias_tail_drift_radps > MAX_GYRO_BIAS_DRIFT_RADPS
            || metrics.gyro_bias_tail_std_radps > MAX_GYRO_BIAS_STD_RADPS
            || metrics.accel_bias_tail_drift_mps2 > MAX_ACCEL_BIAS_DRIFT_MPS2
            || metrics.accel_bias_tail_std_mps2 > MAX_ACCEL_BIAS_STD_MPS2
        {
            reason_mask |= FUSION_HEALTH_REASON_BIAS_UNSTABLE;
        }

        let stable_blockers = FUSION_HEALTH_REASON_NOT_INITIALIZED
            | FUSION_HEALTH_REASON_MOUNT_NOT_READY
            | FUSION_HEALTH_REASON_GNSS_STALE
            | FUSION_HEALTH_REASON_INSUFFICIENT_TIME
            | FUSION_HEALTH_REASON_INSUFFICIENT_MOTION
            | FUSION_HEALTH_REASON_TAIL_TOO_SHORT
            | FUSION_HEALTH_REASON_MOUNT_UNSTABLE
            | FUSION_HEALTH_REASON_BIAS_UNSTABLE
            | FUSION_HEALTH_REASON_COVARIANCE_HIGH
            | FUSION_HEALTH_REASON_GNSS_REJECTING
            | FUSION_HEALTH_REASON_NUMERIC_INVALID
            | FUSION_HEALTH_REASON_NAV_UNUSABLE;
        let stable = running && (reason_mask & stable_blockers) == 0;
        let degraded = ekf_initialized
            && (matches!(
                self.state,
                FusionState::DegradedDeadReckoning | FusionState::AwaitingGnssReseed
            ) || reason_mask
                & (FUSION_HEALTH_REASON_GNSS_STALE
                    | FUSION_HEALTH_REASON_GNSS_REJECTING
                    | FUSION_HEALTH_REASON_NUMERIC_INVALID
                    | FUSION_HEALTH_REASON_NAV_UNUSABLE)
                != 0);
        let public_state = if stable {
            FusionState::Stable
        } else if matches!(self.state, FusionState::AwaitingGnssReseed) {
            FusionState::AwaitingGnssReseed
        } else if matches!(self.state, FusionState::DegradedDeadReckoning) {
            FusionState::DegradedDeadReckoning
        } else if running {
            FusionState::Running
        } else if degraded {
            FusionState::Degraded
        } else if ekf_initialized && self.state == FusionState::NotReady {
            FusionState::Running
        } else if self.last_imu_t_s.is_some() || self.last_gnss_t_s.is_some() {
            FusionState::Initializing
        } else {
            FusionState::NotReady
        };

        FusionHealth {
            state: public_state,
            running: matches!(public_state, FusionState::Running | FusionState::Stable),
            stable,
            degraded,
            navigation_usable,
            reason_mask,
            metrics,
        }
    }

    fn metrics(&self, now_t_s: f32, state: Option<&State>) -> FusionHealthMetrics {
        let tail = self.tail_stats();
        let (mount_sigma_max_deg, attitude_sigma_max_deg) =
            state.map(covariance_sigmas_deg).unwrap_or((0.0, 0.0));
        FusionHealthMetrics {
            post_init_time_s: self
                .init_t_s
                .map(|init_t_s| (now_t_s - init_t_s).max(0.0))
                .unwrap_or(0.0),
            distance_m: self.distance_m,
            mean_speed_mps: if self.speed_count > 0 {
                self.speed_sum_mps / self.speed_count as f32
            } else {
                0.0
            },
            tail_duration_s: tail.duration_s,
            tail_samples: tail.samples as u32,
            mount_tail_drift_deg: tail.mount_drift_deg,
            mount_tail_std_deg: tail.mount_std_deg,
            gyro_bias_tail_drift_radps: tail.gyro_bias_drift,
            gyro_bias_tail_std_radps: tail.gyro_bias_std,
            accel_bias_tail_drift_mps2: tail.accel_bias_drift,
            accel_bias_tail_std_mps2: tail.accel_bias_std,
            mount_sigma_max_deg,
            attitude_sigma_max_deg,
            recent_gnss_issue_count: self.recent_gnss_issue_count,
        }
    }

    fn tail_stats(&self) -> TailStats {
        if self.sample_len < 2 {
            return TailStats::default();
        }
        let first = self.sample_at_age(self.sample_len - 1);
        let latest = self.sample_at_age(0);
        let Some(first) = first else {
            return TailStats::default();
        };
        let Some(latest) = latest else {
            return TailStats::default();
        };
        let duration_s = (latest.t_s - first.t_s).max(0.0);
        let mount_drift_deg = quat_angle_deg(first.q_bv, latest.q_bv);
        let gyro_bias_drift = vec_distance(first.gyro_bias, latest.gyro_bias);
        let accel_bias_drift = vec_distance(first.accel_bias, latest.accel_bias);

        let mut mount_sum = 0.0;
        let mut gyro_sum = 0.0;
        let mut accel_sum = 0.0;
        for age in 0..self.sample_len {
            if let Some(sample) = self.sample_at_age(age) {
                mount_sum += quat_angle_deg(latest.q_bv, sample.q_bv);
                gyro_sum += vec_distance(latest.gyro_bias, sample.gyro_bias);
                accel_sum += vec_distance(latest.accel_bias, sample.accel_bias);
            }
        }
        let n = self.sample_len as f32;
        let mount_mean = mount_sum / n;
        let gyro_mean = gyro_sum / n;
        let accel_mean = accel_sum / n;
        let mut mount_var = 0.0;
        let mut gyro_var = 0.0;
        let mut accel_var = 0.0;
        for age in 0..self.sample_len {
            if let Some(sample) = self.sample_at_age(age) {
                let dm = quat_angle_deg(latest.q_bv, sample.q_bv) - mount_mean;
                let dg = vec_distance(latest.gyro_bias, sample.gyro_bias) - gyro_mean;
                let da = vec_distance(latest.accel_bias, sample.accel_bias) - accel_mean;
                mount_var += dm * dm;
                gyro_var += dg * dg;
                accel_var += da * da;
            }
        }

        TailStats {
            duration_s,
            samples: self.sample_len,
            mount_drift_deg,
            mount_std_deg: sqrt_f32(mount_var / n),
            gyro_bias_drift,
            gyro_bias_std: sqrt_f32(gyro_var / n),
            accel_bias_drift,
            accel_bias_std: sqrt_f32(accel_var / n),
        }
    }

    fn sample_at_age(&self, age: usize) -> Option<HealthSample> {
        if age >= self.sample_len {
            return None;
        }
        let newest = if self.sample_next == 0 {
            TAIL_CAP - 1
        } else {
            self.sample_next - 1
        };
        let index = (newest + TAIL_CAP - (age % TAIL_CAP)) % TAIL_CAP;
        Some(self.samples[index])
    }

    fn decay_gnss_issue_count(&mut self, t_s: f32) {
        if let Some(last_t_s) = self.last_gnss_issue_t_s
            && t_s - last_t_s > 10.0
        {
            self.recent_gnss_issue_count = 0;
            self.last_gnss_issue_t_s = None;
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct TailStats {
    duration_s: f32,
    samples: usize,
    mount_drift_deg: f32,
    mount_std_deg: f32,
    gyro_bias_drift: f32,
    gyro_bias_std: f32,
    accel_bias_drift: f32,
    accel_bias_std: f32,
}

fn covariance_sigmas_deg(state: &State) -> (f32, f32) {
    let mount_sigma = sqrt_f32(
        state.p[15][15]
            .max(0.0)
            .max(state.p[16][16].max(0.0))
            .max(state.p[17][17].max(0.0)),
    ) * DEG_PER_RAD;
    let attitude_sigma = sqrt_f32(
        state.p[0][0]
            .max(0.0)
            .max(state.p[1][1].max(0.0))
            .max(state.p[2][2].max(0.0)),
    ) * DEG_PER_RAD;
    (mount_sigma, attitude_sigma)
}

fn state_finite(state: &State) -> bool {
    let n = &state.nominal;
    [
        n.q0, n.q1, n.q2, n.q3, n.vn, n.ve, n.vd, n.pn, n.pe, n.pd, n.bgx, n.bgy, n.bgz, n.bax,
        n.bay, n.baz, n.q_bv0, n.q_bv1, n.q_bv2, n.q_bv3,
    ]
    .iter()
    .all(|v| v.is_finite())
}

fn metrics_finite(metrics: FusionHealthMetrics) -> bool {
    [
        metrics.post_init_time_s,
        metrics.distance_m,
        metrics.mean_speed_mps,
        metrics.tail_duration_s,
        metrics.mount_tail_drift_deg,
        metrics.mount_tail_std_deg,
        metrics.gyro_bias_tail_drift_radps,
        metrics.gyro_bias_tail_std_radps,
        metrics.accel_bias_tail_drift_mps2,
        metrics.accel_bias_tail_std_mps2,
        metrics.mount_sigma_max_deg,
        metrics.attitude_sigma_max_deg,
    ]
    .iter()
    .all(|v| v.is_finite())
}

fn quat_angle_deg(a: [f32; 4], b: [f32; 4]) -> f32 {
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
        .abs()
        .clamp(0.0, 1.0);
    // Small-angle, acos-free approximation: for unit quaternions,
    // angle ~= 2 * sqrt(2 * (1 - |dot|)).
    2.0 * sqrt_f32(2.0 * (1.0 - dot)) * DEG_PER_RAD
}

fn vec_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    sqrt_f32(
        (a[0] - b[0]) * (a[0] - b[0])
            + (a[1] - b[1]) * (a[1] - b[1])
            + (a[2] - b[2]) * (a[2] - b[2]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProcessNoise;

    fn state_with_covariance(mount_sigma_deg: f32, attitude_sigma_deg: f32) -> State {
        let mut state = State::default();
        state.nominal.q_bv0 = 1.0;
        for i in 0..18 {
            state.p[i][i] = 0.0;
        }
        let mount_var = (mount_sigma_deg / DEG_PER_RAD) * (mount_sigma_deg / DEG_PER_RAD);
        let attitude_var = (attitude_sigma_deg / DEG_PER_RAD) * (attitude_sigma_deg / DEG_PER_RAD);
        for i in 15..18 {
            state.p[i][i] = mount_var;
        }
        for i in 0..3 {
            state.p[i][i] = attitude_var;
        }
        state.noise = ProcessNoise::lsm6dso_104hz();
        state
    }

    #[test]
    fn starts_not_ready() {
        let monitor = FusionHealthMonitor::default();
        let health = monitor.health(false, false, None);
        assert_eq!(health.state, FusionState::NotReady);
        assert!(!health.running);
        assert!(!health.stable);
        assert!(!health.navigation_usable);
    }

    #[test]
    fn stable_requires_tail_motion_and_low_covariance() {
        let mut monitor = FusionHealthMonitor::default();
        let state = state_with_covariance(0.5, 1.0);
        monitor.record_imu(0.0);
        monitor.record_gnss(0.0, [10.0, 0.0, 0.0]);
        monitor.mark_initialized(0.0);
        for t in 1..=181 {
            let t_s = t as f32;
            monitor.record_imu(t_s);
            monitor.record_gnss(t_s, [10.0, 0.0, 0.0]);
            monitor.record_state(t_s, &state);
        }
        let health = monitor.health(true, true, Some(&state));
        assert_eq!(health.state, FusionState::Stable);
        assert!(health.running);
        assert!(health.stable);
        assert!(health.navigation_usable);
    }

    #[test]
    fn covariance_can_keep_initialized_filter_running_not_stable() {
        let mut monitor = FusionHealthMonitor::default();
        let state = state_with_covariance(5.0, 1.0);
        monitor.record_imu(0.0);
        monitor.record_gnss(0.0, [10.0, 0.0, 0.0]);
        monitor.mark_initialized(0.0);
        monitor.record_state(1.0, &state);
        let health = monitor.health(true, true, Some(&state));
        assert_eq!(health.state, FusionState::Running);
        assert!(health.reason_mask & FUSION_HEALTH_REASON_COVARIANCE_HIGH != 0);
        assert!(!health.stable);
    }

    #[test]
    fn degraded_dead_reckoning_keeps_navigation_usable() {
        let mut monitor = FusionHealthMonitor::default();
        let state = state_with_covariance(0.5, 1.0);
        monitor.record_imu(0.0);
        monitor.record_gnss(0.0, [10.0, 0.0, 0.0]);
        monitor.mark_initialized(0.0);
        monitor.mark_degraded_dead_reckoning();
        let health = monitor.health(true, true, Some(&state));
        assert_eq!(health.state, FusionState::DegradedDeadReckoning);
        assert!(health.degraded);
        assert!(health.navigation_usable);
    }

    #[test]
    fn awaiting_reseed_marks_navigation_unusable() {
        let mut monitor = FusionHealthMonitor::default();
        let state = state_with_covariance(0.5, 1.0);
        monitor.record_imu(0.0);
        monitor.record_gnss(0.0, [10.0, 0.0, 0.0]);
        monitor.mark_initialized(0.0);
        monitor.mark_awaiting_gnss_reseed();
        let health = monitor.health(true, true, Some(&state));
        assert_eq!(health.state, FusionState::AwaitingGnssReseed);
        assert!(!health.navigation_usable);
        assert!(health.reason_mask & FUSION_HEALTH_REASON_NAV_UNUSABLE != 0);
    }
}
