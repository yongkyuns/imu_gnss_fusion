//! High-level IMU/GNSS fusion facade.
//!
//! `SensorFusion` owns mount alignment, local WGS84 anchoring, EKF
//! initialization, and runtime updates. Feed samples in timestamp order through
//! `process_imu`, `process_gnss`, and optional vehicle-speed updates; query the
//! latest initialized state and mount estimates from the accessors.
//!
//! The facade bridges the PDF formulations in `docs/`: align estimates the
//! physical vehicle-to-body mount `q_bv`, with `R(q_bv) = C_bv` and
//! `x_b = C_bv x_v`, and the runtime filters use that mount inside inertial
//! propagation so mount errors affect attitude and velocity dynamics directly.
//! Raw IMU samples stay in the IMU body frame at the public API.

use crate::ProcessNoise;
use crate::align::{Align, AlignConfig, AlignUpdateTrace, AlignWindowSummary, GRAVITY_MPS2};
use crate::fusion_types::RuntimeConfig;
pub use crate::fusion_types::{
    AlignDebug, Config, GnssSample, ImuSample, MountMode, Update, VehicleSpeedDirection,
    VehicleSpeedSample,
};
use crate::math::{
    add3_f32, atan2_f32, cos_f32, cross3_f32, dcm_to_quat_f32, mat_mul3_f32, mat_vec3_f32,
    normalize_quat_f32, quat_from_yaw_f32, quat_multiply_f32, quat_to_dcm_f32, scale3_f32, sin_f32,
    sq_f32, sqrt_f32, sub3_f32, transpose3_f32, vec_norm3_f32,
};
use crate::nav::{
    ecef_to_lla_f64, ecef_to_ned_matrix_f32, lla_to_ecef_f64, navigation_rates_ned_f32,
    normal_gravity_mss_f64,
};

const REANCHOR_DISTANCE_M: f32 = 5000.0;
const RUNTIME_ZERO_SPEED_MPS: f32 = 0.80;
const RUNTIME_NHC_MIN_SPEED_MPS: f32 = 0.05;
const RUNTIME_NHC_MAX_GYRO_NORM_RADPS: f32 = 0.2;
const RUNTIME_NHC_MAX_ACCEL_NORM_ERR_MPS2: f32 = 1.0;
const CAN_SPEED_ZERO_MPS: f32 = 0.15;
const CAN_SPEED_SIGN_INFER_MIN_MPS: f32 = 1.0;
const GNSS_POS_MIN_STD_M: f32 = 0.1;
const GNSS_VEL_MIN_STD_MPS: f32 = 0.01;
const GNSS_VERTICAL_POS_STD_SCALE: f32 = 2.5;

#[derive(Clone, Copy, Debug, Default)]
struct Anchor {
    valid: bool,
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    ecef_m: [f64; 3],
    c_ne: [[f32; 3]; 3],
    gravity_mss: f32,
}

/// Streaming fusion state for vehicle IMU, GNSS, and optional vehicle-speed inputs.
#[derive(Debug)]
pub struct SensorFusion {
    cfg: RuntimeConfig,
    ekf: crate::ekf::Filter,
    ekf_initialized: bool,
    internal_align_enabled: bool,
    align_initialized: bool,
    align: Align,
    mount_ready: bool,
    mount_q_bv: Option<[f32; 4]>,
    ekf_mount_q_bv: Option<[f32; 4]>,
    align_ready_since_t_s: Option<f32>,
    ekf_mount_handoff_t_s: Option<f32>,
    mount_settle_released: bool,
    last_imu_t_s: Option<f32>,
    last_imu_sample: Option<ImuSample>,
    last_gnss: Option<crate::ekf::GnssSample>,
    pending_ekf_gnss: Option<crate::ekf::GnssSample>,
    last_ekf_gnss_fuse_t_s: Option<f32>,
    last_ekf_nhc_t_s: Option<f32>,
    bootstrap_prev_ekf_gnss: Option<crate::ekf::GnssSample>,
    gnss_velocity_mount_gain_scale: f32,
    anchor: Anchor,
    interval_imu_sum_gyro: [f32; 3],
    interval_imu_sum_accel: [f32; 3],
    interval_imu_count: u32,
    bootstrap_speed_ema: Ema,
    bootstrap_speed_rate_ema: Ema,
    bootstrap_course_rate_ema: Ema,
    bootstrap_gyro_ema: Ema,
    bootstrap_accel_err_ema: Ema,
    bootstrap_stationary_accel_sum: [f32; 3],
    bootstrap_stationary_count: u32,
    last_align_window: Option<AlignWindowSummary>,
    last_align_trace: Option<AlignUpdateTrace>,
    reanchor_count: u32,
    last_reanchor: Option<(f32, f32)>,
}

impl SensorFusion {
    /// Creates a fusion pipeline with internal mount alignment enabled.
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    /// Creates a fusion pipeline with a fixed manual vehicle-to-body mount quaternion.
    ///
    /// The quaternion must satisfy `R(q_bv) = C_bv`, `x_b = C_bv x_v`.
    pub fn with_mount(q_bv: [f32; 4]) -> Self {
        Self::with_config(Config {
            mount_mode: MountMode::Manual(q_bv),
            ..Config::default()
        })
    }

    /// Creates a fusion pipeline using the requested mount mode.
    pub fn with_mount_mode(mode: MountMode) -> Self {
        Self::with_config(Config {
            mount_mode: mode,
            ..Config::default()
        })
    }

    /// Creates a fusion pipeline from explicit public configuration.
    pub fn with_config(config: Config) -> Self {
        let cfg = RuntimeConfig::default();
        let mut out = Self {
            cfg,
            ekf: crate::ekf::Filter::new(cfg.noise.ekf),
            ekf_initialized: false,
            internal_align_enabled: matches!(config.mount_mode, MountMode::Auto),
            align_initialized: false,
            align: Align::new(cfg.align),
            mount_ready: false,
            mount_q_bv: None,
            ekf_mount_q_bv: None,
            align_ready_since_t_s: None,
            ekf_mount_handoff_t_s: None,
            mount_settle_released: false,
            last_imu_t_s: None,
            last_imu_sample: None,
            last_gnss: None,
            pending_ekf_gnss: None,
            last_ekf_gnss_fuse_t_s: None,
            last_ekf_nhc_t_s: None,
            bootstrap_prev_ekf_gnss: None,
            gnss_velocity_mount_gain_scale: 1.0,
            anchor: Anchor::default(),
            interval_imu_sum_gyro: [0.0; 3],
            interval_imu_sum_accel: [0.0; 3],
            interval_imu_count: 0,
            bootstrap_speed_ema: Ema::default(),
            bootstrap_speed_rate_ema: Ema::default(),
            bootstrap_course_rate_ema: Ema::default(),
            bootstrap_gyro_ema: Ema::default(),
            bootstrap_accel_err_ema: Ema::default(),
            bootstrap_stationary_accel_sum: [0.0; 3],
            bootstrap_stationary_count: 0,
            last_align_window: None,
            last_align_trace: None,
            reanchor_count: 0,
            last_reanchor: None,
        };
        if let MountMode::Manual(q_bv) = config.mount_mode {
            out.set_misalignment(q_bv);
        }
        out
    }

    /// Replaces the current mount with a fixed manual vehicle-to-body quaternion.
    ///
    /// The quaternion must satisfy `R(q_bv) = C_bv`, `x_b = C_bv x_v`.
    pub fn set_misalignment(&mut self, q_bv: [f32; 4]) {
        self.internal_align_enabled = false;
        self.mount_ready = true;
        self.mount_q_bv = Some(q_bv);
        self.ekf_mount_q_bv = Some(q_bv);
        self.align_ready_since_t_s = None;
        self.ekf_mount_handoff_t_s = None;
        self.mount_settle_released = false;
        self.last_ekf_gnss_fuse_t_s = None;
        self.last_ekf_nhc_t_s = None;
        let n = &mut self.ekf.raw_mut().nominal;
        n.q_bv0 = q_bv[0];
        n.q_bv1 = q_bv[1];
        n.q_bv2 = q_bv[2];
        n.q_bv3 = q_bv[3];
        self.ekf.set_freeze_misalignment_states(true);
    }

    /// Replaces the EKF prediction-noise configuration.
    pub fn set_ekf_noise(&mut self, noise: ProcessNoise) {
        self.cfg.noise.ekf = noise;
        self.ekf.raw_mut().noise = noise;
        self.ekf
            .analysis_set_gnss_velocity_mount_gain_scale(self.gnss_velocity_mount_gain_scale);
    }

    /// Replaces the mount-alignment filter configuration.
    ///
    /// This resets the internal alignment filter and bootstrap state, so callers
    /// should configure it before streaming samples for a replay.
    pub fn set_align_config(&mut self, align: AlignConfig) {
        self.cfg.align = align;
        self.cfg.bootstrap.max_gyro_radps = align.max_stationary_gyro_radps;
        self.cfg.bootstrap.max_accel_norm_err_mps2 = align.max_stationary_accel_norm_err_mps2;
        self.align = Align::new(align);
        self.align_initialized = false;
        self.align_ready_since_t_s = None;
        if self.internal_align_enabled {
            self.mount_ready = false;
            self.mount_q_bv = None;
        }
        self.bootstrap_speed_ema = Ema::default();
        self.bootstrap_speed_rate_ema = Ema::default();
        self.bootstrap_course_rate_ema = Ema::default();
        self.bootstrap_gyro_ema = Ema::default();
        self.bootstrap_accel_err_ema = Ema::default();
        self.bootstrap_stationary_accel_sum = [0.0; 3];
        self.bootstrap_stationary_count = 0;
        self.last_ekf_gnss_fuse_t_s = None;
        self.last_ekf_nhc_t_s = None;
    }

    /// Sets the nonholonomic vehicle-frame velocity observation variance.
    pub fn set_r_body_vel(&mut self, r_body_vel: f32) {
        if r_body_vel.is_finite() && r_body_vel >= 0.0 {
            self.cfg.r_body_vel_y = r_body_vel;
            self.cfg.r_body_vel_z = r_body_vel;
        }
    }

    /// Sets lateral and vertical nonholonomic vehicle-frame velocity
    /// measurement variances for the vehicle Y and Z axes.
    pub fn set_r_body_vel_yz(&mut self, r_body_vel_y: f32, r_body_vel_z: f32) {
        if r_body_vel_y.is_finite() && r_body_vel_y >= 0.0 {
            self.cfg.r_body_vel_y = r_body_vel_y;
        }
        if r_body_vel_z.is_finite() && r_body_vel_z >= 0.0 {
            self.cfg.r_body_vel_z = r_body_vel_z;
        }
    }

    /// Sets the minimum period between NHC updates, in seconds.
    ///
    /// `0` applies NHC at every eligible IMU epoch. Positive values decimate
    /// NHC while scaling the observation variance by the elapsed NHC interval,
    /// preserving approximately the same information rate for the continuous
    /// nonholonomic constraint. The runtime default is 10 Hz.
    pub fn set_nhc_update_period_s(&mut self, period_s: f32) {
        if period_s.is_finite() && period_s >= 0.0 {
            self.cfg.nhc_update_period_s = period_s;
            self.last_ekf_nhc_t_s = None;
        }
    }

    /// Sets initial gyro-bias one-sigma uncertainty, in radians per second.
    pub fn set_gyro_bias_init_sigma_radps(&mut self, gyro_bias_init_sigma_radps: f32) {
        if gyro_bias_init_sigma_radps.is_finite() && gyro_bias_init_sigma_radps >= 0.0 {
            self.cfg.gyro_bias_init_sigma_radps = gyro_bias_init_sigma_radps;
        }
    }

    /// Sets initial yaw one-sigma uncertainty, in radians.
    pub fn set_yaw_init_sigma_rad(&mut self, yaw_init_sigma_rad: f32) {
        if yaw_init_sigma_rad.is_finite() && yaw_init_sigma_rad >= 0.0 {
            self.cfg.yaw_init_sigma_rad = yaw_init_sigma_rad;
        }
    }

    /// Sets initial roll/pitch one-sigma uncertainty, in radians.
    pub fn set_attitude_roll_pitch_init_sigma_rad(&mut self, sigma_rad: f32) {
        if sigma_rad.is_finite() && sigma_rad >= 0.0 {
            self.cfg.attitude_roll_pitch_init_sigma_rad = sigma_rad;
        }
    }

    /// Sets initial accelerometer-bias one-sigma uncertainty, in meters per second squared.
    pub fn set_accel_bias_init_sigma_mps2(&mut self, accel_bias_init_sigma_mps2: f32) {
        if accel_bias_init_sigma_mps2.is_finite() && accel_bias_init_sigma_mps2 >= 0.0 {
            self.cfg.accel_bias_init_sigma_mps2 = accel_bias_init_sigma_mps2;
        }
    }

    /// Sets initial residual-mount yaw one-sigma uncertainty, in radians.
    pub fn set_mount_init_sigma_rad(&mut self, mount_init_sigma_rad: f32) {
        if mount_init_sigma_rad.is_finite() && mount_init_sigma_rad >= 0.0 {
            self.cfg.mount_init_sigma_rad = mount_init_sigma_rad;
        }
    }

    /// Sets initial residual-mount roll/pitch one-sigma uncertainty, in radians.
    pub fn set_mount_roll_pitch_init_sigma_rad(&mut self, mount_init_sigma_rad: f32) {
        if mount_init_sigma_rad.is_finite() && mount_init_sigma_rad >= 0.0 {
            self.cfg.mount_roll_pitch_init_sigma_rad = mount_init_sigma_rad;
            self.cfg.mount_roll_init_sigma_rad = mount_init_sigma_rad;
            self.cfg.mount_pitch_init_sigma_rad = mount_init_sigma_rad;
        }
    }

    /// Sets initial residual-mount roll one-sigma uncertainty, in radians.
    pub fn set_mount_roll_init_sigma_rad(&mut self, sigma_rad: f32) {
        if sigma_rad.is_finite() && sigma_rad >= 0.0 {
            self.cfg.mount_roll_init_sigma_rad = sigma_rad;
        }
    }

    /// Sets initial residual-mount pitch one-sigma uncertainty, in radians.
    pub fn set_mount_pitch_init_sigma_rad(&mut self, sigma_rad: f32) {
        if sigma_rad.is_finite() && sigma_rad >= 0.0 {
            self.cfg.mount_pitch_init_sigma_rad = sigma_rad;
        }
    }

    /// Sets accelerometer-bias random-walk process variance.
    pub fn set_accel_bias_rw_var(&mut self, accel_bias_rw_var: f32) {
        if accel_bias_rw_var.is_finite() && accel_bias_rw_var >= 0.0 {
            self.cfg.noise.ekf.accel_bias_rw_var = accel_bias_rw_var;
            self.ekf.raw_mut().noise.accel_bias_rw_var = accel_bias_rw_var;
        }
    }

    /// Sets residual-mount random-walk process variance.
    pub fn set_mount_align_rw_var(&mut self, mount_align_rw_var: f32) {
        if mount_align_rw_var.is_finite() && mount_align_rw_var >= 0.0 {
            self.cfg.noise.ekf.mount_align_rw_var = mount_align_rw_var;
            self.ekf.raw_mut().noise.mount_align_rw_var = mount_align_rw_var;
        }
    }

    /// Sets how long coarse alignment must remain ready before handoff, in seconds.
    pub fn set_align_handoff_delay_s(&mut self, align_handoff_delay_s: f32) {
        if align_handoff_delay_s.is_finite() && align_handoff_delay_s >= 0.0 {
            self.cfg.align_handoff_delay_s = align_handoff_delay_s;
            self.align_ready_since_t_s = None;
        }
    }

    /// Enables or disables mount-state freezing in the EKF.
    pub fn set_freeze_misalignment_states(&mut self, freeze: bool) {
        self.cfg.freeze_misalignment_states = freeze;
        self.ekf
            .set_freeze_misalignment_states(self.effective_freeze_misalignment_states());
    }

    /// Sets a post-initialization delay before mount states are released, in seconds.
    pub fn set_mount_settle_time_s(&mut self, mount_settle_time_s: f32) {
        if mount_settle_time_s.is_finite() && mount_settle_time_s >= 0.0 {
            self.cfg.mount_settle_time_s = mount_settle_time_s;
            self.mount_settle_released = false;
        }
    }

    /// Sets the mount uncertainty threshold for releasing mount-settle freezing.
    pub fn set_mount_settle_release_sigma_rad(&mut self, sigma_rad: f32) {
        if sigma_rad.is_finite() && sigma_rad >= 0.0 {
            self.cfg.mount_settle_release_sigma_rad = sigma_rad;
        }
    }

    /// Sets whether mount-settle release clears mount cross-covariance terms.
    pub fn set_mount_settle_zero_cross_covariance(&mut self, zero_cross: bool) {
        self.cfg.mount_settle_zero_cross_covariance = zero_cross;
    }

    /// Sets the vehicle-speed observation variance.
    pub fn set_r_vehicle_speed(&mut self, r_vehicle_speed: f32) {
        if r_vehicle_speed.is_finite() && r_vehicle_speed >= 0.0 {
            self.cfg.r_vehicle_speed = r_vehicle_speed;
        }
    }

    /// Sets the zero-velocity observation variance.
    pub fn set_r_zero_vel(&mut self, r_zero_vel: f32) {
        if r_zero_vel.is_finite() && r_zero_vel >= 0.0 {
            self.cfg.r_zero_vel = r_zero_vel;
        }
    }

    /// Sets the stationary-gravity observation variance.
    pub fn set_r_stationary_accel(&mut self, r_stationary_accel: f32) {
        if r_stationary_accel.is_finite() && r_stationary_accel >= 0.0 {
            self.cfg.r_stationary_accel = r_stationary_accel;
        }
    }

    /// Processes one IMU sample and returns updated runtime status.
    pub fn process_imu(&mut self, sample: ImuSample) -> Update {
        let prev_sample = self.last_imu_sample.replace(sample);
        let Some(last_t) = self.last_imu_t_s.replace(sample.t_s) else {
            self.try_bootstrap_align(sample.accel_mps2, sample.gyro_radps);
            return self.update(false, false);
        };
        let dt = sample.t_s - last_t;
        let prev_sample = prev_sample.unwrap_or(sample);

        self.interval_imu_sum_gyro[0] += sample.gyro_radps[0];
        self.interval_imu_sum_gyro[1] += sample.gyro_radps[1];
        self.interval_imu_sum_gyro[2] += sample.gyro_radps[2];
        self.interval_imu_sum_accel[0] += sample.accel_mps2[0];
        self.interval_imu_sum_accel[1] += sample.accel_mps2[1];
        self.interval_imu_sum_accel[2] += sample.accel_mps2[2];
        self.interval_imu_count += 1;

        self.try_bootstrap_align(sample.accel_mps2, sample.gyro_radps);

        if !self.ekf_initialized || !self.mount_ready {
            return self.update(false, false);
        }
        self.refresh_mount_settle_state(sample.t_s);
        if !(0.001..=0.05).contains(&dt) {
            return self.update(false, false);
        }

        let gyro_vehicle = self.current_vehicle_vector_from_body(sample.gyro_radps);
        let accel_vehicle = self.current_vehicle_vector_from_body(sample.accel_mps2);
        let (gyro_predict, coriolis_delta_v_n) =
            self.ekf_navigation_rate_corrections(sample.gyro_radps, dt);
        self.ekf.predict(crate::ekf::ImuDelta {
            dax: gyro_predict[0] * dt,
            day: gyro_predict[1] * dt,
            daz: gyro_predict[2] * dt,
            dvx: 0.5 * (prev_sample.accel_mps2[0] + sample.accel_mps2[0]) * dt,
            dvy: 0.5 * (prev_sample.accel_mps2[1] + sample.accel_mps2[1]) * dt,
            dvz: 0.5 * (prev_sample.accel_mps2[2] + sample.accel_mps2[2]) * dt,
            dt,
        });
        {
            let nominal = &mut self.ekf.raw_mut().nominal;
            nominal.vn += coriolis_delta_v_n[0];
            nominal.ve += coriolis_delta_v_n[1];
            nominal.vd += coriolis_delta_v_n[2];
        }
        self.clamp_ekf_biases();

        if self.cfg.r_body_vel_y > 0.0 || self.cfg.r_body_vel_z > 0.0 {
            if self.runtime_zero_velocity_active(sample.accel_mps2, sample.gyro_radps) {
                self.fuse_pending_gnss_at_imu(sample.t_s, None);
                if self.cfg.r_zero_vel > 0.0 {
                    self.ekf.fuse_zero_vel(self.cfg.r_zero_vel);
                }
                if self.cfg.r_stationary_accel > 0.0 {
                    self.ekf
                        .fuse_stationary_gravity(accel_vehicle, self.cfg.r_stationary_accel);
                }
            } else {
                let nhc = self.imu_epoch_nhc_variances(sample.t_s, dt, accel_vehicle, gyro_vehicle);
                let used_nhc_with_gnss = self.fuse_pending_gnss_at_imu(sample.t_s, nhc);
                if let Some(r) = nhc
                    && !used_nhc_with_gnss
                {
                    self.ekf.fuse_body_vel_yz(r[0], r[1]);
                }
            }
            self.clamp_ekf_biases();
        } else {
            self.fuse_pending_gnss_at_imu(sample.t_s, None);
        }

        self.update(false, false)
    }

    /// Processes one GNSS sample and returns updated runtime status.
    pub fn process_gnss(&mut self, gnss: GnssSample) -> Update {
        let Some(local) = self.prepare_local_gnss_sample(gnss) else {
            return self.update(false, false);
        };
        self.last_gnss = Some(local);

        let prev_mount_ready = self.mount_ready;
        self.bootstrap_update_gnss_hints(local);

        if self.internal_align_enabled {
            if self.align_initialized
                && let Some(prev) = self.bootstrap_prev_ekf_gnss
                && let Some(summary) = self.take_interval_summary(prev, local)
            {
                let (_score, trace) = self.align.update_window_with_trace(&summary);
                self.last_align_window = Some(summary);
                self.last_align_trace = Some(trace);
                self.mount_q_bv = Some(self.align.q_bv);
                self.mount_ready =
                    self.align_handoff_ready(trace.coarse_alignment_ready, local.t_s);
            }
            self.bootstrap_prev_ekf_gnss = Some(local);
        } else {
            self.reset_interval_summary();
        }

        if !self.mount_ready {
            return self.update(prev_mount_ready != self.mount_ready, false);
        }

        let ekf_initialized_now = if !self.ekf_initialized {
            self.ekf_mount_q_bv = self.mount_q_bv;
            self.initialize_ekf_from_gnss(local);
            self.ekf_initialized = true;
            self.ekf_mount_handoff_t_s = Some(local.t_s);
            self.mount_settle_released = false;
            self.refresh_mount_settle_state(local.t_s);
            true
        } else {
            self.refresh_mount_settle_state(local.t_s);
            self.pending_ekf_gnss = Some(local);
            false
        };

        self.update(prev_mount_ready != self.mount_ready, ekf_initialized_now)
    }

    /// Processes one vehicle-speed sample and returns updated runtime status.
    pub fn process_vehicle_speed(&mut self, speed: VehicleSpeedSample) -> Update {
        if !self.ekf_initialized || !self.mount_ready {
            return self.update(false, false);
        }
        self.refresh_mount_settle_state(speed.t_s);
        if speed.speed_mps < 0.0 || !speed.speed_mps.is_finite() {
            return self.update(false, false);
        }

        match speed.direction {
            VehicleSpeedDirection::Forward => {
                self.fuse_signed_body_speed(speed.speed_mps);
            }
            VehicleSpeedDirection::Reverse => {
                self.fuse_signed_body_speed(-speed.speed_mps);
            }
            VehicleSpeedDirection::Unknown => {
                if speed.speed_mps <= CAN_SPEED_ZERO_MPS {
                    self.ekf.fuse_zero_vel(self.cfg.r_vehicle_speed);
                } else {
                    let predicted = body_speed_x_estimate(self.ekf.raw());
                    if predicted.abs() >= CAN_SPEED_SIGN_INFER_MIN_MPS {
                        self.fuse_signed_body_speed(speed.speed_mps.copysign(predicted));
                    }
                }
            }
        }
        self.update(false, false)
    }

    /// Returns the current EKF state after GNSS initialization.
    pub fn ekf(&self) -> Option<&crate::ekf::State> {
        self.ekf_initialized.then_some(self.ekf.raw())
    }

    /// Returns the latest physical vehicle-to-body mount quaternion, if ready.
    ///
    /// The quaternion satisfies `R(q_bv) = C_bv`, `x_b = C_bv x_v`.
    pub fn mount_q_bv(&self) -> Option<[f32; 4]> {
        self.mount_q_bv
    }

    /// Returns the mount quaternion currently used by the EKF propagation model.
    ///
    /// The quaternion satisfies `R(q_bv) = C_bv`, `x_b = C_bv x_v`.
    pub fn ekf_mount_q_bv(&self) -> Option<[f32; 4]> {
        self.ekf_mount_q_bv
    }

    /// Returns the current local-origin anchor `[lat_deg, lon_deg, height_m]` for diagnostics.
    pub fn anchor_lla_debug(&self) -> Option<[f32; 3]> {
        self.anchor.valid.then_some([
            self.anchor.lat_deg as f32,
            self.anchor.lon_deg as f32,
            self.anchor.height_m as f32,
        ])
    }

    /// Returns how many times the local navigation anchor has been reset.
    pub fn reanchor_count(&self) -> u32 {
        self.reanchor_count
    }

    /// Returns the last reanchor distance and timestamp, when available.
    pub fn last_reanchor_info(&self) -> Option<(f32, f32)> {
        self.last_reanchor
    }

    /// Reports whether a mount estimate is ready.
    pub fn mount_ready(&self) -> bool {
        self.mount_ready
    }

    /// Returns the current EKF position as `[lat_deg, lon_deg, height_m]`.
    pub fn position_lla(&self) -> Option<[f32; 3]> {
        self.position_lla_f64()
            .map(|lla| [lla[0] as f32, lla[1] as f32, lla[2] as f32])
    }

    /// Returns the current EKF position as `[lat_deg, lon_deg, height_m]` using
    /// double precision for the final ECEF-to-geodetic conversion.
    pub fn position_lla_f64(&self) -> Option<[f64; 3]> {
        let ekf = self.ekf()?;
        if !self.anchor.valid {
            return None;
        }
        let c_en = transpose3_f32(self.anchor.c_ne);
        let p = [
            ekf.nominal.pn as f64,
            ekf.nominal.pe as f64,
            ekf.nominal.pd as f64,
        ];
        let ecef = [
            self.anchor.ecef_m[0]
                + c_en[0][0] as f64 * p[0]
                + c_en[0][1] as f64 * p[1]
                + c_en[0][2] as f64 * p[2],
            self.anchor.ecef_m[1]
                + c_en[1][0] as f64 * p[0]
                + c_en[1][1] as f64 * p[1]
                + c_en[1][2] as f64 * p[2],
            self.anchor.ecef_m[2]
                + c_en[2][0] as f64 * p[0]
                + c_en[2][1] as f64 * p[1]
                + c_en[2][2] as f64 * p[2],
        ];
        Some(ecef_to_lla_f64(ecef))
    }

    /// Returns the internal align filter after stationary bootstrap.
    pub fn align(&self) -> Option<&Align> {
        self.align_initialized.then_some(&self.align)
    }

    /// Returns the latest internal alignment debug information.
    pub fn align_debug(&self) -> Option<AlignDebug> {
        Some(AlignDebug {
            window: self.last_align_window?,
            trace: self.last_align_trace?,
        })
    }

    /// Diagnostic hook that directly sets the EKF mount quaternion.
    pub fn analysis_set_ekf_mount_quat(&mut self, q_bv: [f32; 4]) {
        if !self.ekf_initialized {
            return;
        }
        let mut q = q_bv;
        normalize_quat_f32(&mut q);
        let n = &mut self.ekf.raw_mut().nominal;
        n.q_bv0 = q[0];
        n.q_bv1 = q[1];
        n.q_bv2 = q[2];
        n.q_bv3 = q[3];
    }

    /// Diagnostic hook that directly sets residual-mount covariance.
    pub fn analysis_set_ekf_mount_covariance(&mut self, sigma_rad: f32, zero_cross: bool) {
        if !self.ekf_initialized || !sigma_rad.is_finite() || sigma_rad < 0.0 {
            return;
        }
        let var = sigma_rad * sigma_rad;
        for i in 15..18 {
            if zero_cross {
                for j in 0..18 {
                    self.ekf.raw_mut().p[i][j] = 0.0;
                    self.ekf.raw_mut().p[j][i] = 0.0;
                }
            }
            self.ekf.raw_mut().p[i][i] = var;
        }
        if self.effective_freeze_misalignment_states() {
            self.ekf.set_freeze_misalignment_states(true);
        }
    }

    /// Diagnostic hook that directly sets residual-mount covariance per axis.
    ///
    /// This is intended for offline sensitivity analysis. Runtime configuration
    /// should use the regular public tuning setters.
    pub fn analysis_set_mount_covariance_axes(&mut self, sigma_rad: [f32; 3], zero_cross: bool) {
        if sigma_rad
            .iter()
            .any(|sigma| !sigma.is_finite() || *sigma < 0.0)
        {
            return;
        }
        if !self.ekf_initialized {
            return;
        }

        let variances = [
            sigma_rad[0] * sigma_rad[0],
            sigma_rad[1] * sigma_rad[1],
            sigma_rad[2] * sigma_rad[2],
        ];
        let raw = self.ekf.raw_mut();
        set_covariance_axis_block(&mut raw.p, 15, variances, zero_cross);
        if self.effective_freeze_misalignment_states() {
            self.ekf.set_freeze_misalignment_states(true);
        }
    }

    /// Diagnostic hook that directly sets residual attitude roll/pitch covariance.
    pub fn analysis_set_ekf_attitude_roll_pitch_covariance(&mut self, sigma_rad: f32) {
        if !self.ekf_initialized || !sigma_rad.is_finite() || sigma_rad < 0.0 {
            return;
        }
        let var = sigma_rad * sigma_rad;
        let raw = self.ekf.raw_mut();
        raw.p[0][0] = var;
        raw.p[1][1] = var;
    }

    /// Diagnostic hook that scales direct mount correction from GNSS velocity rows.
    ///
    /// `1.0` is the normal filter behavior. Values below one are intended for
    /// offline residual-allocation experiments.
    pub fn analysis_set_gnss_velocity_mount_gain_scale(&mut self, scale: f32) {
        if !scale.is_finite() || scale < 0.0 {
            return;
        }
        self.gnss_velocity_mount_gain_scale = scale;
        self.ekf.analysis_set_gnss_velocity_mount_gain_scale(scale);
    }

    fn effective_freeze_misalignment_states(&self) -> bool {
        self.cfg.freeze_misalignment_states || self.manual_mount_mode()
    }

    fn manual_mount_mode(&self) -> bool {
        !self.internal_align_enabled
    }

    fn update(&self, mount_ready_changed: bool, filter_initialized_now: bool) -> Update {
        Update {
            mount_ready: self.mount_ready,
            mount_ready_changed,
            ekf_initialized: self.ekf_initialized,
            ekf_initialized_now: filter_initialized_now,
            mount_q_bv: self.mount_q_bv,
        }
    }

    fn initialize_ekf_from_gnss(&mut self, gnss: crate::ekf::GnssSample) {
        self.ekf = crate::ekf::Filter::new(self.cfg.noise.ekf);
        self.ekf
            .analysis_set_gnss_velocity_mount_gain_scale(self.gnss_velocity_mount_gain_scale);
        if self.anchor.valid {
            self.ekf.set_gravity_mss(self.anchor.gravity_mss);
        }
        self.ekf
            .set_freeze_misalignment_states(self.effective_freeze_misalignment_states());
        let speed_h = sqrt_f32(
            gnss.vel_ned_mps[0] * gnss.vel_ned_mps[0] + gnss.vel_ned_mps[1] * gnss.vel_ned_mps[1],
        );
        let yaw = gnss.heading_rad.unwrap_or_else(|| {
            if speed_h >= self.cfg.yaw_init_speed_mps.max(1.0) {
                atan2_f32(gnss.vel_ned_mps[1], gnss.vel_ned_mps[0])
            } else {
                0.0
            }
        });
        self.ekf
            .init_nominal_from_gnss(quat_from_yaw_f32(yaw), gnss);
        let raw = self.ekf.raw_mut();
        if let Some(q_bv) = self.ekf_mount_q_bv.or(self.mount_q_bv) {
            raw.nominal.q_bv0 = q_bv[0];
            raw.nominal.q_bv1 = q_bv[1];
            raw.nominal.q_bv2 = q_bv[2];
            raw.nominal.q_bv3 = q_bv[3];
        }
        raw.p[9][9] = sq_f32(self.cfg.gyro_bias_init_sigma_radps);
        raw.p[10][10] = raw.p[9][9];
        raw.p[11][11] = raw.p[9][9];
        raw.p[12][12] = sq_f32(self.cfg.accel_bias_init_sigma_mps2);
        raw.p[13][13] = raw.p[12][12];
        raw.p[14][14] = raw.p[12][12];
        raw.p[0][0] = sq_f32(self.cfg.attitude_roll_pitch_init_sigma_rad);
        raw.p[1][1] = raw.p[0][0];
        raw.p[2][2] = sq_f32(self.cfg.yaw_init_sigma_rad);
        let mount_var = sq_f32(self.cfg.mount_init_sigma_rad);
        raw.p[15][15] = sq_f32(self.cfg.mount_roll_init_sigma_rad);
        raw.p[16][16] = sq_f32(self.cfg.mount_pitch_init_sigma_rad);
        raw.p[17][17] = mount_var;
        if self.effective_freeze_misalignment_states() {
            self.ekf.set_freeze_misalignment_states(true);
        }
        self.last_ekf_gnss_fuse_t_s = Some(gnss.t_s);
        self.last_ekf_nhc_t_s = None;
    }

    fn align_handoff_ready(&mut self, coarse_ready: bool, t_s: f32) -> bool {
        if !coarse_ready {
            self.align_ready_since_t_s = None;
            return false;
        }
        let ready_since = *self.align_ready_since_t_s.get_or_insert(t_s);
        t_s - ready_since >= self.cfg.align_handoff_delay_s
    }

    fn refresh_mount_settle_state(&mut self, t_s: f32) {
        if self.effective_freeze_misalignment_states() {
            self.ekf.set_freeze_misalignment_states(true);
            return;
        }
        if self.cfg.mount_settle_time_s <= 0.0 {
            self.ekf.set_freeze_misalignment_states(false);
            return;
        }
        let Some(handoff_t_s) = self.ekf_mount_handoff_t_s else {
            return;
        };
        if t_s - handoff_t_s < self.cfg.mount_settle_time_s {
            self.ekf.set_freeze_misalignment_states(true);
            return;
        }
        if !self.mount_settle_released {
            self.ekf.set_freeze_misalignment_states(false);
            self.reset_mount_covariance_after_settle();
            self.mount_settle_released = true;
        } else {
            self.ekf.set_freeze_misalignment_states(false);
        }
    }

    fn reset_mount_covariance_after_settle(&mut self) {
        let sigma = self.cfg.mount_settle_release_sigma_rad;
        if !sigma.is_finite() || sigma < 0.0 {
            return;
        }
        let var = sigma * sigma;
        let raw = self.ekf.raw_mut();
        for i in 15..18 {
            if self.cfg.mount_settle_zero_cross_covariance {
                for j in 0..18 {
                    raw.p[i][j] = 0.0;
                    raw.p[j][i] = 0.0;
                }
            }
            raw.p[i][i] = var;
        }
    }

    fn fuse_signed_body_speed(&mut self, signed_speed_mps: f32) {
        self.ekf
            .fuse_body_speed_x(signed_speed_mps, self.cfg.r_vehicle_speed);
    }

    fn current_vehicle_vector_from_body(&self, vector_b: [f32; 3]) -> [f32; 3] {
        let n = &self.ekf.raw().nominal;
        let c_bv = quat_to_dcm_f32([n.q_bv0, n.q_bv1, n.q_bv2, n.q_bv3]);
        mat_vec3_f32(transpose3_f32(c_bv), vector_b)
    }

    fn try_bootstrap_align(&mut self, accel_b: [f32; 3], gyro_radps: [f32; 3]) {
        if !self.internal_align_enabled || self.align_initialized {
            return;
        }
        if self.bootstrap_update(accel_b, gyro_radps) {
            let mean = [
                self.bootstrap_stationary_accel_sum[0] / self.bootstrap_stationary_count as f32,
                self.bootstrap_stationary_accel_sum[1] / self.bootstrap_stationary_count as f32,
                self.bootstrap_stationary_accel_sum[2] / self.bootstrap_stationary_count as f32,
            ];
            if self.align.initialize_from_stationary(&[mean], 0.0).is_ok() {
                self.align_initialized = true;
                self.mount_q_bv = Some(self.align.q_bv);
            }
        }
    }

    fn prepare_local_gnss_sample(&mut self, sample: GnssSample) -> Option<crate::ekf::GnssSample> {
        if !self.anchor.valid {
            self.anchor = anchor_from_lla(sample.lat_deg, sample.lon_deg, sample.height_m);
            self.ekf.set_gravity_mss(self.anchor.gravity_mss);
        }
        let ecef = lla_to_ecef_f64(sample.lat_deg, sample.lon_deg, sample.height_m);
        let mut pos_ned = ecef_to_anchor_ned(&self.anchor, ecef);
        let horiz_dist = sqrt_f32(pos_ned[0] * pos_ned[0] + pos_ned[1] * pos_ned[1]);
        if horiz_dist > REANCHOR_DISTANCE_M {
            let new_anchor = anchor_from_lla(sample.lat_deg, sample.lon_deg, sample.height_m);
            self.rotate_nav_state_to_new_anchor(&new_anchor);
            self.anchor = new_anchor;
            self.ekf.set_gravity_mss(self.anchor.gravity_mss);
            self.reanchor_count += 1;
            self.last_reanchor = Some((sample.t_s, horiz_dist));
            self.bootstrap_prev_ekf_gnss = None;
            self.reset_interval_summary();
            pos_ned = [0.0; 3];
        }
        let (pos_std_m, vel_std_mps) = ekf_gnss_sigmas(sample.pos_std_m, sample.vel_std_mps);
        Some(crate::ekf::GnssSample {
            t_s: sample.t_s,
            pos_ned_m: pos_ned,
            vel_ned_mps: velocity_local_ned_to_anchor_ned(
                &self.anchor,
                sample.lat_deg,
                sample.lon_deg,
                sample.vel_ned_mps,
            ),
            pos_std_m,
            vel_std_mps,
            heading_rad: sample
                .heading_rad
                .map(|h| heading_local_to_anchor(&self.anchor, sample.lat_deg, sample.lon_deg, h)),
        })
    }

    fn rotate_nav_state_to_new_anchor(&mut self, new_anchor: &Anchor) {
        if !self.anchor.valid || !new_anchor.valid {
            return;
        }
        let c_en_old = transpose3_f32(self.anchor.c_ne);
        let r_n1_n0 = mat_mul3_f32(new_anchor.c_ne, c_en_old);

        if self.ekf_initialized {
            let raw = self.ekf.raw_mut();
            let p_n0 = [raw.nominal.pn, raw.nominal.pe, raw.nominal.pd];
            let v_n0 = [raw.nominal.vn, raw.nominal.ve, raw.nominal.vd];
            let p_ecef = [
                self.anchor.ecef_m[0]
                    + c_en_old[0][0] as f64 * p_n0[0] as f64
                    + c_en_old[0][1] as f64 * p_n0[1] as f64
                    + c_en_old[0][2] as f64 * p_n0[2] as f64,
                self.anchor.ecef_m[1]
                    + c_en_old[1][0] as f64 * p_n0[0] as f64
                    + c_en_old[1][1] as f64 * p_n0[1] as f64
                    + c_en_old[1][2] as f64 * p_n0[2] as f64,
                self.anchor.ecef_m[2]
                    + c_en_old[2][0] as f64 * p_n0[0] as f64
                    + c_en_old[2][1] as f64 * p_n0[1] as f64
                    + c_en_old[2][2] as f64 * p_n0[2] as f64,
            ];
            let p_n1 = ecef_to_anchor_ned(new_anchor, p_ecef);
            let v_n1 = mat_vec3_f32(r_n1_n0, v_n0);
            raw.nominal.pn = p_n1[0];
            raw.nominal.pe = p_n1[1];
            raw.nominal.pd = p_n1[2];
            raw.nominal.vn = v_n1[0];
            raw.nominal.ve = v_n1[1];
            raw.nominal.vd = v_n1[2];

            let q_n1_n0 = dcm_to_quat_f32(r_n1_n0);
            let q_old = [
                raw.nominal.q0,
                raw.nominal.q1,
                raw.nominal.q2,
                raw.nominal.q3,
            ];
            let mut q_new = quat_multiply_f32(q_n1_n0, q_old);
            normalize_quat_f32(&mut q_new);
            raw.nominal.q0 = q_new[0];
            raw.nominal.q1 = q_new[1];
            raw.nominal.q2 = q_new[2];
            raw.nominal.q3 = q_new[3];
            rotate_ekf_covariance_nav_blocks(&mut raw.p, r_n1_n0);
        }

        if let Some(mut last) = self.last_gnss {
            let pos_old = last.pos_ned_m;
            let v_n0 = last.vel_ned_mps;
            let last_ecef = [
                self.anchor.ecef_m[0]
                    + c_en_old[0][0] as f64 * pos_old[0] as f64
                    + c_en_old[0][1] as f64 * pos_old[1] as f64
                    + c_en_old[0][2] as f64 * pos_old[2] as f64,
                self.anchor.ecef_m[1]
                    + c_en_old[1][0] as f64 * pos_old[0] as f64
                    + c_en_old[1][1] as f64 * pos_old[1] as f64
                    + c_en_old[1][2] as f64 * pos_old[2] as f64,
                self.anchor.ecef_m[2]
                    + c_en_old[2][0] as f64 * pos_old[0] as f64
                    + c_en_old[2][1] as f64 * pos_old[1] as f64
                    + c_en_old[2][2] as f64 * pos_old[2] as f64,
            ];
            last.pos_ned_m = ecef_to_anchor_ned(new_anchor, last_ecef);
            last.vel_ned_mps = mat_vec3_f32(r_n1_n0, v_n0);
            last.heading_rad = None;
            self.last_gnss = Some(last);
        }

        if let Some(mut prev) = self.bootstrap_prev_ekf_gnss {
            prev.vel_ned_mps = mat_vec3_f32(r_n1_n0, prev.vel_ned_mps);
            self.bootstrap_prev_ekf_gnss = Some(prev);
        }
    }

    fn bootstrap_update_gnss_hints(&mut self, sample: crate::ekf::GnssSample) {
        let speed = horiz_speed(sample.vel_ned_mps);
        self.bootstrap_speed_ema
            .update(speed, self.cfg.bootstrap.ema_alpha);
        let Some(prev) = self.bootstrap_prev_ekf_gnss else {
            return;
        };
        let dt = sample.t_s - prev.t_s;
        if dt <= 1.0e-3 {
            return;
        }
        let prev_speed = horiz_speed(prev.vel_ned_mps);
        let speed_rate = (speed - prev_speed) / dt;
        let course_prev = atan2_f32(prev.vel_ned_mps[1], prev.vel_ned_mps[0]);
        let course_curr = atan2_f32(sample.vel_ned_mps[1], sample.vel_ned_mps[0]);
        let course_rate = wrap_pi(course_curr - course_prev) / dt;
        self.bootstrap_speed_rate_ema
            .update(speed_rate.abs(), self.cfg.bootstrap.ema_alpha);
        self.bootstrap_course_rate_ema
            .update(course_rate.abs(), self.cfg.bootstrap.ema_alpha);
    }

    fn bootstrap_update(&mut self, accel_b: [f32; 3], gyro_radps: [f32; 3]) -> bool {
        let gyro_norm = vec_norm3_f32(gyro_radps);
        let accel_err = (vec_norm3_f32(accel_b) - GRAVITY_MPS2).abs();
        let gyro_ema = self
            .bootstrap_gyro_ema
            .update(gyro_norm, self.cfg.bootstrap.ema_alpha);
        let accel_ema = self
            .bootstrap_accel_err_ema
            .update(accel_err, self.cfg.bootstrap.ema_alpha);
        let low_dynamic = gyro_ema <= self.cfg.bootstrap.max_gyro_radps
            && accel_ema <= self.cfg.bootstrap.max_accel_norm_err_mps2;
        let low_speed = !self.bootstrap_speed_ema.valid
            || self.bootstrap_speed_ema.value <= self.cfg.bootstrap.max_speed_mps;
        let steady_motion = self.bootstrap_speed_rate_ema.valid
            && self.bootstrap_course_rate_ema.valid
            && self.bootstrap_speed_rate_ema.value <= self.cfg.bootstrap.max_speed_rate_mps2
            && self.bootstrap_course_rate_ema.value <= self.cfg.bootstrap.max_course_rate_radps;
        if low_dynamic && (low_speed || steady_motion) {
            if self.bootstrap_stationary_count < 400 {
                for (sum, sample) in self.bootstrap_stationary_accel_sum.iter_mut().zip(accel_b) {
                    *sum += sample;
                }
                self.bootstrap_stationary_count += 1;
            }
        } else {
            self.bootstrap_stationary_count = 0;
            self.bootstrap_stationary_accel_sum = [0.0; 3];
        }
        self.bootstrap_stationary_count >= self.cfg.bootstrap.stationary_samples
    }

    fn take_interval_summary(
        &mut self,
        prev_gnss: crate::ekf::GnssSample,
        curr_gnss: crate::ekf::GnssSample,
    ) -> Option<AlignWindowSummary> {
        if self.interval_imu_count == 0 {
            return None;
        }
        let inv = 1.0 / self.interval_imu_count as f32;
        let summary = AlignWindowSummary {
            dt: (curr_gnss.t_s - prev_gnss.t_s).max(1.0e-3),
            mean_gyro_b: [
                self.interval_imu_sum_gyro[0] * inv,
                self.interval_imu_sum_gyro[1] * inv,
                self.interval_imu_sum_gyro[2] * inv,
            ],
            mean_accel_b: [
                self.interval_imu_sum_accel[0] * inv,
                self.interval_imu_sum_accel[1] * inv,
                self.interval_imu_sum_accel[2] * inv,
            ],
            gnss_vel_prev_n: prev_gnss.vel_ned_mps,
            gnss_vel_curr_n: curr_gnss.vel_ned_mps,
        };
        self.reset_interval_summary();
        Some(summary)
    }

    fn reset_interval_summary(&mut self) {
        self.interval_imu_sum_gyro = [0.0; 3];
        self.interval_imu_sum_accel = [0.0; 3];
        self.interval_imu_count = 0;
    }

    fn runtime_zero_velocity_active(&mut self, accel_b: [f32; 3], gyro_radps: [f32; 3]) -> bool {
        let gyro_ema = self
            .bootstrap_gyro_ema
            .update(vec_norm3_f32(gyro_radps), self.cfg.bootstrap.ema_alpha);
        let accel_ema = self.bootstrap_accel_err_ema.update(
            (vec_norm3_f32(accel_b) - GRAVITY_MPS2).abs(),
            self.cfg.bootstrap.ema_alpha,
        );
        let low_dynamic = gyro_ema <= self.cfg.bootstrap.max_gyro_radps
            && accel_ema <= self.cfg.bootstrap.max_accel_norm_err_mps2;
        let low_speed = if self.ekf_initialized {
            self.ekf_speed_estimate_mps() <= RUNTIME_ZERO_SPEED_MPS
        } else {
            false
        };
        low_dynamic && low_speed
    }

    fn imu_epoch_nhc_variances(
        &mut self,
        t_s: f32,
        dt: f32,
        accel_vehicle: [f32; 3],
        gyro_vehicle: [f32; 3],
    ) -> Option<[f32; 2]> {
        let speed_allows_nhc = nhc_speed_allows_update(self.ekf_speed_estimate_mps());
        let nhc_active = runtime_nhc_active(accel_vehicle, gyro_vehicle) && speed_allows_nhc;
        let (obs_dt, last_t_s) =
            self.nhc_observation_interval(self.last_ekf_nhc_t_s, t_s, dt, nhc_active);
        self.last_ekf_nhc_t_s = last_t_s;
        let obs_dt = obs_dt?;
        let r_scale = nhc_observation_r_scale(obs_dt);
        Some([
            self.cfg.r_body_vel_y * r_scale,
            self.cfg.r_body_vel_z * r_scale,
        ])
    }

    fn nhc_observation_interval(
        &self,
        last_t_s: Option<f32>,
        t_s: f32,
        fallback_dt_s: f32,
        active: bool,
    ) -> (Option<f32>, Option<f32>) {
        if !active {
            return (None, Some(t_s));
        }
        let fallback_dt_s = fallback_dt_s.max(1.0e-3);
        let period_s = self.cfg.nhc_update_period_s;
        if period_s <= 0.0 {
            return (Some(fallback_dt_s), Some(t_s));
        }
        let Some(last_t_s) = last_t_s else {
            return (Some(fallback_dt_s), Some(t_s));
        };
        let elapsed_s = t_s - last_t_s;
        if elapsed_s + 1.0e-4 < period_s {
            return (None, Some(last_t_s));
        }
        (Some(elapsed_s.max(fallback_dt_s)), Some(t_s))
    }

    fn fuse_pending_gnss_at_imu(&mut self, imu_t_s: f32, nhc: Option<[f32; 2]>) -> bool {
        let Some(gnss) = self.pending_ekf_gnss else {
            return false;
        };
        let age_s = imu_t_s - gnss.t_s;
        if age_s < -1.0e-6 {
            return false;
        }
        self.pending_ekf_gnss = None;
        let use_nhc = (0.0..=0.05).contains(&age_s).then_some(nhc).flatten();
        let gnss = self.rate_normalized_ekf_gnss(gnss);
        self.last_ekf_gnss_fuse_t_s = Some(gnss.t_s);
        self.ekf
            .fuse_gps_nhc_batch(gnss, use_nhc.map(|r| r[0]), use_nhc.map(|r| r[1]));
        use_nhc.is_some()
    }

    fn rate_normalized_ekf_gnss(&self, mut gnss: crate::ekf::GnssSample) -> crate::ekf::GnssSample {
        let Some(prev_t_s) = self.last_ekf_gnss_fuse_t_s else {
            return gnss;
        };
        let dt_s = gnss.t_s - prev_t_s;
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return gnss;
        }

        // Normalize position information rate so higher GNSS rates do not make
        // position residuals disproportionately stiff.
        let r_scale = 1.0 / dt_s.clamp(1.0e-3, 1.0);
        let std_scale = sqrt_f32(r_scale);
        for std_m in &mut gnss.pos_std_m {
            *std_m *= std_scale;
        }
        gnss
    }

    fn clamp_ekf_biases(&mut self) {
        let n = &mut self.ekf.raw_mut().nominal;
        let max_gyro = 1.5_f32.to_radians();
        let max_accel = 1.5;
        n.bgx = n.bgx.clamp(-max_gyro, max_gyro);
        n.bgy = n.bgy.clamp(-max_gyro, max_gyro);
        n.bgz = n.bgz.clamp(-max_gyro, max_gyro);
        n.bax = n.bax.clamp(-max_accel, max_accel);
        n.bay = n.bay.clamp(-max_accel, max_accel);
        n.baz = n.baz.clamp(-max_accel, max_accel);
    }

    fn ekf_speed_estimate_mps(&self) -> f32 {
        let nominal = &self.ekf.raw().nominal;
        vec_norm3_f32([nominal.vn, nominal.ve, nominal.vd])
    }

    fn ekf_navigation_rate_corrections(
        &self,
        gyro_body: [f32; 3],
        dt: f32,
    ) -> ([f32; 3], [f32; 3]) {
        if !self.anchor.valid || dt <= 0.0 {
            return (gyro_body, [0.0; 3]);
        }
        let Some(ekf) = self.ekf() else {
            return (gyro_body, [0.0; 3]);
        };
        let nominal = &ekf.nominal;
        let lla = self.position_lla_f64().unwrap_or([
            self.anchor.lat_deg,
            self.anchor.lon_deg,
            self.anchor.height_m,
        ]);
        let (omega_ie_n, omega_en_n) = navigation_rates_ned_f32(
            lla[0] as f32,
            lla[2] as f32,
            [nominal.vn, nominal.ve, nominal.vd],
        );
        // The generated EKF propagation is a local-level NED model. The IMU
        // gyro measures body rate relative to inertial space, so subtract the
        // navigation frame's inertial rate before applying the flat local
        // quaternion update, and apply the matching Coriolis/transport velocity
        // term in NED. Propagation now consumes raw body-frame IMU, so the
        // navigation rate is converted back through the current mount.
        let omega_in_n = add3_f32(omega_ie_n, omega_en_n);
        let c_nv = quat_to_dcm_f32([nominal.q0, nominal.q1, nominal.q2, nominal.q3]);
        let omega_in_v = mat_vec3_f32(transpose3_f32(c_nv), omega_in_n);
        let c_bv = quat_to_dcm_f32([nominal.q_bv0, nominal.q_bv1, nominal.q_bv2, nominal.q_bv3]);
        let omega_in_b = mat_vec3_f32(c_bv, omega_in_v);
        let gyro_predict = sub3_f32(gyro_body, omega_in_b);
        let coriolis_rate = cross3_f32(
            add3_f32(scale3_f32(omega_ie_n, 2.0), omega_en_n),
            [nominal.vn, nominal.ve, nominal.vd],
        );
        let coriolis_delta_v_n = scale3_f32(coriolis_rate, -dt);
        (gyro_predict, coriolis_delta_v_n)
    }
}

impl Default for SensorFusion {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Ema {
    valid: bool,
    value: f32,
}

impl Ema {
    fn update(&mut self, sample: f32, alpha: f32) -> f32 {
        let alpha = alpha.clamp(1.0e-4, 1.0);
        if self.valid {
            self.value = (1.0 - alpha) * self.value + alpha * sample;
        } else {
            self.value = sample;
            self.valid = true;
        }
        self.value
    }
}

fn nhc_speed_allows_update(speed: f32) -> bool {
    speed > RUNTIME_NHC_MIN_SPEED_MPS
}

fn nhc_observation_r_scale(dt_s: f32) -> f32 {
    let dt_obs = if dt_s > 0.0 && dt_s.is_finite() {
        dt_s.min(1.0)
    } else {
        1.0
    };
    1.0 / dt_obs
}

fn runtime_nhc_active(accel_v: [f32; 3], gyro_v: [f32; 3]) -> bool {
    vec_norm3_f32(gyro_v) < RUNTIME_NHC_MAX_GYRO_NORM_RADPS
        && (vec_norm3_f32(accel_v) - GRAVITY_MPS2).abs() < RUNTIME_NHC_MAX_ACCEL_NORM_ERR_MPS2
}

fn body_speed_x_estimate(ekf: &crate::ekf::State) -> f32 {
    let n = &ekf.nominal;
    (1.0 - 2.0 * n.q2 * n.q2 - 2.0 * n.q3 * n.q3) * n.vn
        + 2.0 * (n.q1 * n.q2 + n.q0 * n.q3) * n.ve
        + 2.0 * (n.q1 * n.q3 - n.q0 * n.q2) * n.vd
}

fn anchor_from_lla(lat_deg: f64, lon_deg: f64, height_m: f64) -> Anchor {
    Anchor {
        valid: true,
        lat_deg,
        lon_deg,
        height_m,
        ecef_m: lla_to_ecef_f64(lat_deg, lon_deg, height_m),
        c_ne: ecef_to_ned_matrix_f32(lat_deg, lon_deg),
        gravity_mss: normal_gravity_mss_f64(lat_deg, height_m),
    }
}

fn ecef_to_anchor_ned(anchor: &Anchor, ecef_m: [f64; 3]) -> [f32; 3] {
    let diff = [
        (ecef_m[0] - anchor.ecef_m[0]) as f32,
        (ecef_m[1] - anchor.ecef_m[1]) as f32,
        (ecef_m[2] - anchor.ecef_m[2]) as f32,
    ];
    mat_vec3_f32(anchor.c_ne, diff)
}

fn velocity_local_ned_to_anchor_ned(
    anchor: &Anchor,
    lat_deg: f64,
    lon_deg: f64,
    vel_local_ned_mps: [f32; 3],
) -> [f32; 3] {
    let c_ne_local = ecef_to_ned_matrix_f32(lat_deg, lon_deg);
    let c_en_local = transpose3_f32(c_ne_local);
    let vel_ecef = mat_vec3_f32(c_en_local, vel_local_ned_mps);
    mat_vec3_f32(anchor.c_ne, vel_ecef)
}

fn heading_local_to_anchor(
    anchor: &Anchor,
    lat_deg: f64,
    lon_deg: f64,
    heading_local_rad: f32,
) -> f32 {
    let forward_local = [cos_f32(heading_local_rad), sin_f32(heading_local_rad), 0.0];
    let c_ne_local = ecef_to_ned_matrix_f32(lat_deg, lon_deg);
    let c_en_local = transpose3_f32(c_ne_local);
    let forward_ecef = mat_vec3_f32(c_en_local, forward_local);
    let forward_anchor = mat_vec3_f32(anchor.c_ne, forward_ecef);
    atan2_f32(forward_anchor[1], forward_anchor[0])
}

fn horiz_speed(vel_ned_mps: [f32; 3]) -> f32 {
    sqrt_f32(vel_ned_mps[0] * vel_ned_mps[0] + vel_ned_mps[1] * vel_ned_mps[1])
}

#[allow(clippy::needless_range_loop)]
fn rotate_ekf_covariance_nav_blocks(p: &mut [[f32; 18]; 18], r_n1_n0: [[f32; 3]; 3]) {
    let mut t = [[0.0; 18]; 18];
    for i in 0..18 {
        t[i][i] = 1.0;
    }
    for block in 0..3 {
        let base = block * 3;
        for i in 0..3 {
            for j in 0..3 {
                t[base + i][base + j] = r_n1_n0[i][j];
            }
        }
    }

    let mut tmp = [[0.0; 18]; 18];
    for i in 0..18 {
        for j in 0..18 {
            for k in 0..18 {
                tmp[i][j] += t[i][k] * p[k][j];
            }
        }
    }
    let mut out = [[0.0; 18]; 18];
    for i in 0..18 {
        for j in 0..18 {
            for k in 0..18 {
                out[i][j] += tmp[i][k] * t[j][k];
            }
        }
    }
    *p = out;
}

fn ekf_gnss_sigmas(pos_std_m: [f32; 3], vel_std_mps: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let pos_avg = ((pos_std_m[0] + pos_std_m[1] + pos_std_m[2]) / 3.0).max(GNSS_POS_MIN_STD_M);
    let pos_std_m = [pos_avg, pos_avg, GNSS_VERTICAL_POS_STD_SCALE * pos_avg];
    let vel_std_mps = vel_std_mps.map(|v| v.max(GNSS_VEL_MIN_STD_MPS));
    (pos_std_m, vel_std_mps)
}

fn set_covariance_axis_block<const N: usize>(
    p: &mut [[f32; N]; N],
    base: usize,
    variances: [f32; 3],
    zero_cross: bool,
) {
    for axis in 0..3 {
        let i = base + axis;
        if zero_cross {
            for j in 0..N {
                p[i][j] = 0.0;
                p[j][i] = 0.0;
            }
        }
        p[i][i] = variances[axis];
    }
}

fn wrap_pi(mut rad: f32) -> f32 {
    while rad <= -core::f32::consts::PI {
        rad += 2.0 * core::f32::consts::PI;
    }
    while rad > core::f32::consts::PI {
        rad -= 2.0 * core::f32::consts::PI;
    }
    rad
}
