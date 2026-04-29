//! High-level IMU/GNSS fusion facade.
//!
//! `SensorFusion` owns mount alignment, local WGS84 anchoring, ESKF
//! initialization, and runtime updates. Feed samples in timestamp order through
//! `process_imu`, `process_gnss`, and optional vehicle-speed updates; query the
//! latest initialized ESKF state and mount estimates from the accessors.
//!
//! The facade bridges the PDF formulations in `docs/`: align estimates the
//! physical vehicle-to-body seed `q_vb`, IMU deltas are pre-rotated into the
//! ESKF seeded frame, and the runtime ESKF may then estimate residual mount
//! `q_cs` through vehicle-speed and NHC measurements.

use crate::align::{Align, AlignConfig, AlignUpdateTrace, AlignWindowSummary, GRAVITY_MPS2};
use crate::ekf::PredictNoise;
use crate::eskf_types::{EskfGnssSample, EskfImuDelta, EskfState};
use crate::rust_eskf::RustEskf;

const WGS84_A_M: f64 = 6378137.0;
const WGS84_E2: f64 = 6.69437999014e-3;
const REANCHOR_DISTANCE_M: f32 = 5000.0;
const RUNTIME_ZERO_SPEED_MPS: f32 = 0.80;
const RUNTIME_NHC_MAX_ROLL_PITCH_GYRO_RADPS: f32 = 0.03;
const RUNTIME_NHC_MAX_ACCEL_NORM_ERR_MPS2: f32 = 0.2;
const CAN_SPEED_ZERO_MPS: f32 = 0.15;
const CAN_SPEED_SIGN_INFER_MIN_MPS: f32 = 1.0;

/// One timestamped IMU sample in the sensor/body frame.
#[derive(Clone, Copy, Debug)]
pub struct FusionImuSample {
    /// Sample timestamp, in seconds.
    pub t_s: f32,
    /// Angular rate in the body frame, in radians per second.
    pub gyro_radps: [f32; 3],
    /// Specific force in the body frame, in meters per second squared.
    pub accel_mps2: [f32; 3],
}

/// One timestamped GNSS sample in geodetic coordinates with NED velocity.
#[derive(Clone, Copy, Debug)]
pub struct FusionGnssSample {
    /// Sample timestamp, in seconds.
    pub t_s: f32,
    /// Geodetic latitude, in degrees.
    pub lat_deg: f32,
    /// Geodetic longitude, in degrees.
    pub lon_deg: f32,
    /// Ellipsoidal height, in meters.
    pub height_m: f32,
    /// Velocity in local NED coordinates, in meters per second.
    pub vel_ned_mps: [f32; 3],
    /// One-sigma position standard deviation for NED axes, in meters.
    pub pos_std_m: [f32; 3],
    /// One-sigma velocity standard deviation for NED axes, in meters per second.
    pub vel_std_mps: [f32; 3],
    /// Optional heading observation, in radians.
    pub heading_rad: Option<f32>,
}

/// Timestamped vehicle speed observation, typically from CAN or wheel speed.
#[derive(Clone, Copy, Debug)]
pub struct FusionVehicleSpeedSample {
    /// Sample timestamp, in seconds.
    pub t_s: f32,
    /// Nonnegative speed magnitude, in meters per second.
    pub speed_mps: f32,
    /// Direction associated with `speed_mps`.
    pub direction: FusionVehicleSpeedDirection,
}

/// Direction qualifier for a vehicle-speed sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FusionVehicleSpeedDirection {
    /// Direction is unknown and may be inferred from the current filter state.
    #[default]
    Unknown,
    /// Speed is forward along the vehicle X axis.
    Forward,
    /// Speed is reverse along the vehicle X axis.
    Reverse,
}

/// Status returned after each fusion input sample.
#[derive(Clone, Copy, Debug, Default)]
pub struct FusionUpdate {
    /// Whether a mount estimate is ready for ESKF initialization or propagation.
    pub mount_ready: bool,
    /// Whether `mount_ready` changed during this input sample.
    pub mount_ready_changed: bool,
    /// Whether the ESKF has been initialized from GNSS.
    pub ekf_initialized: bool,
    /// Whether ESKF initialization happened during this input sample.
    pub ekf_initialized_now: bool,
    /// Current vehicle-to-body mount quaternion, when available.
    pub mount_q_vb: Option<[f32; 4]>,
}

/// Last alignment window and update trace captured by [`SensorFusion`].
#[derive(Clone, Copy, Debug)]
pub struct FusionAlignDebug {
    /// Window statistics passed to the align filter.
    pub window: AlignWindowSummary,
    /// Detailed alignment update trace for the same window.
    pub trace: AlignUpdateTrace,
}

/// Selects how the runtime obtains the IMU-to-vehicle mount estimate.
#[derive(Clone, Copy, Debug)]
pub enum MisalignmentMode {
    /// Estimate mount internally with [`Align`] before ESKF initialization.
    InternalAlign,
    /// Use the supplied vehicle-to-body mount quaternion and disable internal alignment.
    External([f32; 4]),
}

/// Selects how ESKF propagation receives the mount after alignment handoff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EskfMountSource {
    /// Latch the alignment seed at ESKF initialization and let residual ESKF mount states evolve.
    LatchedSeed,
    /// Keep following the current align mount and freeze residual ESKF mount states.
    FollowAlign,
}

#[derive(Clone, Copy, Debug)]
struct BootstrapConfig {
    ema_alpha: f32,
    max_speed_mps: f32,
    max_speed_rate_mps2: f32,
    max_course_rate_radps: f32,
    stationary_samples: u32,
    max_gyro_radps: f32,
    max_accel_norm_err_mps2: f32,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        let align = AlignConfig::default();
        Self {
            ema_alpha: 0.05,
            max_speed_mps: 0.35,
            max_speed_rate_mps2: 0.15,
            max_course_rate_radps: 1.0_f32.to_radians(),
            stationary_samples: 100,
            max_gyro_radps: align.max_stationary_gyro_radps,
            max_accel_norm_err_mps2: align.max_stationary_accel_norm_err_mps2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FusionConfig {
    align: AlignConfig,
    bootstrap: BootstrapConfig,
    predict_noise: PredictNoise,
    yaw_init_sigma_rad: f32,
    gyro_bias_init_sigma_radps: f32,
    accel_bias_init_sigma_mps2: f32,
    mount_init_sigma_rad: f32,
    r_body_vel: f32,
    gnss_pos_mount_scale: f32,
    gnss_vel_mount_scale: f32,
    gnss_vel_xy_update_min_scale: f32,
    gnss_vel_update_ramp_time_s: f32,
    mount_update_min_scale: f32,
    mount_update_ramp_time_s: f32,
    mount_update_innovation_gate_mps: f32,
    mount_update_yaw_rate_gate_radps: f32,
    align_handoff_delay_s: f32,
    freeze_misalignment_states: bool,
    eskf_mount_source: EskfMountSource,
    mount_settle_time_s: f32,
    mount_settle_release_sigma_rad: f32,
    mount_settle_zero_cross_covariance: bool,
    r_vehicle_speed: f32,
    r_zero_vel: f32,
    r_stationary_accel: f32,
    yaw_init_speed_mps: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            align: AlignConfig::default(),
            bootstrap: BootstrapConfig::default(),
            predict_noise: PredictNoise {
                gyro_var: 2.287_311_3e-7,
                accel_var: 2.450_421_4e-5,
                gyro_bias_rw_var: 0.0002e-9,
                accel_bias_rw_var: 0.002e-9,
                mount_align_rw_var: 1.0e-7,
            },
            yaw_init_sigma_rad: 2.0_f32.to_radians(),
            gyro_bias_init_sigma_radps: 0.125_f32.to_radians(),
            accel_bias_init_sigma_mps2: 0.20,
            mount_init_sigma_rad: 2.5_f32.to_radians(),
            r_body_vel: 0.001,
            gnss_pos_mount_scale: 0.0,
            gnss_vel_mount_scale: 0.0,
            gnss_vel_xy_update_min_scale: 0.25,
            gnss_vel_update_ramp_time_s: 20.0,
            mount_update_min_scale: 0.008,
            mount_update_ramp_time_s: 120.0,
            mount_update_innovation_gate_mps: 0.10,
            mount_update_yaw_rate_gate_radps: 0.0,
            align_handoff_delay_s: 0.0,
            freeze_misalignment_states: false,
            eskf_mount_source: EskfMountSource::LatchedSeed,
            mount_settle_time_s: 0.0,
            mount_settle_release_sigma_rad: 7.5_f32.to_radians(),
            mount_settle_zero_cross_covariance: true,
            r_vehicle_speed: 0.04,
            r_zero_vel: 0.0,
            r_stationary_accel: 0.0,
            yaw_init_speed_mps: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Anchor {
    valid: bool,
    lat_deg: f32,
    lon_deg: f32,
    height_m: f32,
    ecef_m: [f64; 3],
    c_ne: [[f32; 3]; 3],
}

/// Streaming fusion state for vehicle IMU, GNSS, and optional vehicle-speed inputs.
#[derive(Debug)]
pub struct SensorFusion {
    cfg: FusionConfig,
    eskf: RustEskf,
    ekf_initialized: bool,
    internal_align_enabled: bool,
    align_initialized: bool,
    align: Align,
    mount_ready: bool,
    mount_q_vb: Option<[f32; 4]>,
    eskf_mount_q_vb: Option<[f32; 4]>,
    align_ready_since_t_s: Option<f32>,
    ekf_mount_handoff_t_s: Option<f32>,
    mount_settle_released: bool,
    last_imu_t_s: Option<f32>,
    last_gnss: Option<EskfGnssSample>,
    bootstrap_prev_gnss: Option<EskfGnssSample>,
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
        Self::with_misalignment_mode(MisalignmentMode::InternalAlign)
    }

    /// Creates a fusion pipeline with a fixed external vehicle-to-body mount quaternion.
    pub fn with_misalignment(q_vb: [f32; 4]) -> Self {
        Self::with_misalignment_mode(MisalignmentMode::External(q_vb))
    }

    /// Creates a fusion pipeline using the requested mount-source mode.
    pub fn with_misalignment_mode(mode: MisalignmentMode) -> Self {
        let cfg = FusionConfig::default();
        let mut out = Self {
            cfg,
            eskf: RustEskf::new(cfg.predict_noise),
            ekf_initialized: false,
            internal_align_enabled: matches!(mode, MisalignmentMode::InternalAlign),
            align_initialized: false,
            align: Align::new(cfg.align),
            mount_ready: false,
            mount_q_vb: None,
            eskf_mount_q_vb: None,
            align_ready_since_t_s: None,
            ekf_mount_handoff_t_s: None,
            mount_settle_released: false,
            last_imu_t_s: None,
            last_gnss: None,
            bootstrap_prev_gnss: None,
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
        if let MisalignmentMode::External(q_vb) = mode {
            out.set_misalignment(q_vb);
        }
        out
    }

    /// Replaces the current mount with a fixed external vehicle-to-body quaternion.
    pub fn set_misalignment(&mut self, q_vb: [f32; 4]) {
        self.internal_align_enabled = false;
        self.mount_ready = true;
        self.mount_q_vb = Some(q_vb);
        self.eskf_mount_q_vb = Some(q_vb);
        self.align_ready_since_t_s = None;
        self.ekf_mount_handoff_t_s = None;
        self.mount_settle_released = false;
        let n = &mut self.eskf.raw_mut().nominal;
        n.qcs0 = 1.0;
        n.qcs1 = 0.0;
        n.qcs2 = 0.0;
        n.qcs3 = 0.0;
    }

    /// Sets the nonholonomic body-velocity observation variance.
    pub fn set_r_body_vel(&mut self, r_body_vel: f32) {
        if r_body_vel.is_finite() && r_body_vel >= 0.0 {
            self.cfg.r_body_vel = r_body_vel;
        }
    }

    /// Sets how strongly GNSS position rows may update residual mount states.
    pub fn set_gnss_pos_mount_scale(&mut self, gnss_pos_mount_scale: f32) {
        if gnss_pos_mount_scale.is_finite() && (0.0..=1.0).contains(&gnss_pos_mount_scale) {
            self.cfg.gnss_pos_mount_scale = gnss_pos_mount_scale;
        }
    }

    /// Sets how strongly GNSS velocity rows may update residual mount states.
    pub fn set_gnss_vel_mount_scale(&mut self, gnss_vel_mount_scale: f32) {
        if gnss_vel_mount_scale.is_finite() && (0.0..=1.0).contains(&gnss_vel_mount_scale) {
            self.cfg.gnss_vel_mount_scale = gnss_vel_mount_scale;
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

    /// Sets initial accelerometer-bias one-sigma uncertainty, in meters per second squared.
    pub fn set_accel_bias_init_sigma_mps2(&mut self, accel_bias_init_sigma_mps2: f32) {
        if accel_bias_init_sigma_mps2.is_finite() && accel_bias_init_sigma_mps2 >= 0.0 {
            self.cfg.accel_bias_init_sigma_mps2 = accel_bias_init_sigma_mps2;
        }
    }

    /// Sets initial residual-mount one-sigma uncertainty, in radians.
    pub fn set_mount_init_sigma_rad(&mut self, mount_init_sigma_rad: f32) {
        if mount_init_sigma_rad.is_finite() && mount_init_sigma_rad >= 0.0 {
            self.cfg.mount_init_sigma_rad = mount_init_sigma_rad;
        }
    }

    /// Sets accelerometer-bias random-walk process variance.
    pub fn set_accel_bias_rw_var(&mut self, accel_bias_rw_var: f32) {
        if accel_bias_rw_var.is_finite() && accel_bias_rw_var >= 0.0 {
            self.cfg.predict_noise.accel_bias_rw_var = accel_bias_rw_var;
            self.eskf.raw_mut().noise.accel_bias_rw_var = accel_bias_rw_var;
        }
    }

    /// Sets residual-mount random-walk process variance.
    pub fn set_mount_align_rw_var(&mut self, mount_align_rw_var: f32) {
        if mount_align_rw_var.is_finite() && mount_align_rw_var >= 0.0 {
            self.cfg.predict_noise.mount_align_rw_var = mount_align_rw_var;
            self.eskf.raw_mut().noise.mount_align_rw_var = mount_align_rw_var;
        }
    }

    /// Sets the minimum residual-mount update scale during the runtime ramp.
    pub fn set_mount_update_min_scale(&mut self, mount_update_min_scale: f32) {
        if mount_update_min_scale.is_finite() && (0.0..=1.0).contains(&mount_update_min_scale) {
            self.cfg.mount_update_min_scale = mount_update_min_scale;
        }
    }

    /// Sets the residual-mount update ramp duration, in seconds.
    pub fn set_mount_update_ramp_time_s(&mut self, mount_update_ramp_time_s: f32) {
        if mount_update_ramp_time_s.is_finite() && mount_update_ramp_time_s >= 0.0 {
            self.cfg.mount_update_ramp_time_s = mount_update_ramp_time_s;
        }
    }

    /// Sets the innovation magnitude at which residual-mount updates begin to be attenuated.
    pub fn set_mount_update_innovation_gate_mps(&mut self, mount_update_innovation_gate_mps: f32) {
        if mount_update_innovation_gate_mps.is_finite() && mount_update_innovation_gate_mps >= 0.0 {
            self.cfg.mount_update_innovation_gate_mps = mount_update_innovation_gate_mps;
        }
    }

    /// Sets the yaw-rate gate used by residual-mount updates, in radians per second.
    pub fn set_mount_update_yaw_rate_gate_radps(&mut self, mount_update_yaw_rate_gate_radps: f32) {
        if mount_update_yaw_rate_gate_radps.is_finite() && mount_update_yaw_rate_gate_radps >= 0.0 {
            self.cfg.mount_update_yaw_rate_gate_radps = mount_update_yaw_rate_gate_radps;
        }
    }

    /// Sets how long coarse alignment must remain ready before handoff, in seconds.
    pub fn set_align_handoff_delay_s(&mut self, align_handoff_delay_s: f32) {
        if align_handoff_delay_s.is_finite() && align_handoff_delay_s >= 0.0 {
            self.cfg.align_handoff_delay_s = align_handoff_delay_s;
            self.align_ready_since_t_s = None;
        }
    }

    /// Enables or disables residual-mount state freezing in the ESKF.
    pub fn set_freeze_misalignment_states(&mut self, freeze: bool) {
        self.cfg.freeze_misalignment_states = freeze;
        self.eskf
            .set_freeze_misalignment_states(self.effective_freeze_misalignment_states());
    }

    /// Selects whether the ESKF uses a latched mount seed or follows the align filter.
    pub fn set_eskf_mount_source(&mut self, source: EskfMountSource) {
        self.cfg.eskf_mount_source = source;
        self.refresh_follow_align_mount_source();
        self.eskf
            .set_freeze_misalignment_states(self.effective_freeze_misalignment_states());
    }

    /// Sets a post-initialization delay before residual-mount states are released, in seconds.
    pub fn set_mount_settle_time_s(&mut self, mount_settle_time_s: f32) {
        if mount_settle_time_s.is_finite() && mount_settle_time_s >= 0.0 {
            self.cfg.mount_settle_time_s = mount_settle_time_s;
            self.mount_settle_released = false;
        }
    }

    /// Sets the residual-mount uncertainty threshold for releasing mount-settle freezing.
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
    pub fn process_imu(&mut self, sample: FusionImuSample) -> FusionUpdate {
        let Some(last_t) = self.last_imu_t_s.replace(sample.t_s) else {
            self.try_bootstrap_align(sample.accel_mps2, sample.gyro_radps);
            return self.update(false, false);
        };
        let dt = sample.t_s - last_t;

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

        let mount_q = self
            .eskf_mount_q_vb
            .or(self.mount_q_vb)
            .unwrap_or([1.0, 0.0, 0.0, 0.0]);
        let c_bv = quat_to_rotmat(mount_q);
        let c_vb = transpose3(c_bv);
        let gyro_vehicle = mat3_vec(c_vb, sample.gyro_radps);
        let accel_vehicle = mat3_vec(c_vb, sample.accel_mps2);
        self.eskf.predict(EskfImuDelta {
            dax: gyro_vehicle[0] * dt,
            day: gyro_vehicle[1] * dt,
            daz: gyro_vehicle[2] * dt,
            dvx: accel_vehicle[0] * dt,
            dvy: accel_vehicle[1] * dt,
            dvz: accel_vehicle[2] * dt,
            dt,
        });
        self.clamp_eskf_biases();

        if self.cfg.r_body_vel > 0.0 {
            if self.runtime_zero_velocity_active(sample.accel_mps2, sample.gyro_radps) {
                if self.cfg.r_zero_vel > 0.0 {
                    self.eskf.fuse_zero_vel(self.cfg.r_zero_vel);
                }
                if self.cfg.r_stationary_accel > 0.0 {
                    self.eskf
                        .fuse_stationary_gravity(accel_vehicle, self.cfg.r_stationary_accel);
                }
            } else if runtime_nhc_active(accel_vehicle, gyro_vehicle) {
                let body_update_scale = self.mount_update_scale(sample.t_s);
                let nhc_speed_scale = self.runtime_nhc_speed_scale();
                if nhc_speed_scale > 0.0 {
                    let fused_body_update_scale = body_update_scale * nhc_speed_scale;
                    let mount_update_scale =
                        fused_body_update_scale * self.mount_yaw_observability_scale(gyro_vehicle);
                    let effective_r = self.cfg.r_body_vel / fused_body_update_scale.max(1.0e-3);
                    self.eskf.fuse_body_vel_scaled(
                        effective_r,
                        mount_update_scale,
                        self.cfg.mount_update_innovation_gate_mps,
                    );
                }
            }
            self.clamp_eskf_biases();
        }

        self.update(false, false)
    }

    /// Processes one GNSS sample and returns updated runtime status.
    pub fn process_gnss(&mut self, gnss: FusionGnssSample) -> FusionUpdate {
        let Some(mut local) = self.prepare_local_gnss_sample(gnss) else {
            return self.update(false, false);
        };
        let vel_scale_xy = self.gnss_vel_update_scale_xy(local.t_s);
        let vel_std_scale_xy = vel_scale_xy.sqrt();
        local.vel_std_mps[0] *= vel_std_scale_xy;
        local.vel_std_mps[1] *= vel_std_scale_xy;
        self.last_gnss = Some(local);

        let prev_mount_ready = self.mount_ready;
        self.bootstrap_update_gnss_hints(local);

        if self.internal_align_enabled {
            if self.align_initialized
                && let Some(prev) = self.bootstrap_prev_gnss
                && let Some(summary) = self.take_interval_summary(prev, local)
            {
                let (_score, trace) = self.align.update_window_with_trace(&summary);
                self.last_align_window = Some(summary);
                self.last_align_trace = Some(trace);
                self.mount_q_vb = Some(self.align.q_vb);
                self.mount_ready =
                    self.align_handoff_ready(trace.coarse_alignment_ready, local.t_s);
                self.refresh_follow_align_mount_source();
            }
            self.bootstrap_prev_gnss = Some(local);
        } else {
            self.reset_interval_summary();
        }

        if !self.mount_ready {
            return self.update(prev_mount_ready != self.mount_ready, false);
        }

        let ekf_initialized_now = if !self.ekf_initialized {
            self.eskf_mount_q_vb = self.mount_q_vb;
            self.initialize_eskf_from_gnss(local);
            self.ekf_initialized = true;
            self.ekf_mount_handoff_t_s = Some(local.t_s);
            self.mount_settle_released = false;
            self.refresh_follow_align_mount_source();
            self.refresh_mount_settle_state(local.t_s);
            true
        } else {
            self.refresh_follow_align_mount_source();
            self.refresh_mount_settle_state(local.t_s);
            self.eskf.fuse_gps_scaled(
                local,
                self.cfg.gnss_pos_mount_scale,
                self.cfg.gnss_vel_mount_scale,
            );
            false
        };

        self.update(prev_mount_ready != self.mount_ready, ekf_initialized_now)
    }

    /// Processes one vehicle-speed sample and returns updated runtime status.
    pub fn process_vehicle_speed(&mut self, speed: FusionVehicleSpeedSample) -> FusionUpdate {
        if !self.ekf_initialized || !self.mount_ready {
            return self.update(false, false);
        }
        self.refresh_mount_settle_state(speed.t_s);
        if speed.speed_mps < 0.0 || !speed.speed_mps.is_finite() {
            return self.update(false, false);
        }

        match speed.direction {
            FusionVehicleSpeedDirection::Forward => {
                self.fuse_signed_body_speed(speed.t_s, speed.speed_mps);
            }
            FusionVehicleSpeedDirection::Reverse => {
                self.fuse_signed_body_speed(speed.t_s, -speed.speed_mps);
            }
            FusionVehicleSpeedDirection::Unknown => {
                if speed.speed_mps <= CAN_SPEED_ZERO_MPS {
                    self.eskf.fuse_zero_vel(self.cfg.r_vehicle_speed);
                } else {
                    let predicted = body_speed_x_estimate(self.eskf.raw());
                    if predicted.abs() >= CAN_SPEED_SIGN_INFER_MIN_MPS {
                        self.fuse_signed_body_speed(speed.t_s, speed.speed_mps.copysign(predicted));
                    }
                }
            }
        }
        self.update(false, false)
    }

    /// Returns the current ESKF state after GNSS initialization.
    pub fn eskf(&self) -> Option<&EskfState> {
        self.ekf_initialized.then_some(self.eskf.raw())
    }

    /// Returns the latest physical vehicle-to-body mount quaternion, if ready.
    pub fn mount_q_vb(&self) -> Option<[f32; 4]> {
        self.mount_q_vb
    }

    /// Returns the mount quaternion currently used to pre-rotate IMU into the ESKF frame.
    pub fn eskf_mount_q_vb(&self) -> Option<[f32; 4]> {
        self.eskf_mount_q_vb
    }

    /// Returns the current local-origin anchor `[lat_deg, lon_deg, height_m]` for diagnostics.
    pub fn anchor_lla_debug(&self) -> Option<[f32; 3]> {
        self.anchor.valid.then_some([
            self.anchor.lat_deg,
            self.anchor.lon_deg,
            self.anchor.height_m,
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

    /// Returns the current ESKF position as `[lat_deg, lon_deg, height_m]`.
    pub fn position_lla(&self) -> Option<[f32; 3]> {
        self.position_lla_f64()
            .map(|lla| [lla[0] as f32, lla[1] as f32, lla[2] as f32])
    }

    /// Returns the current ESKF position as `[lat_deg, lon_deg, height_m]` using
    /// double precision for the final ECEF-to-geodetic conversion.
    pub fn position_lla_f64(&self) -> Option<[f64; 3]> {
        let eskf = self.eskf()?;
        if !self.anchor.valid {
            return None;
        }
        let c_en = transpose3(self.anchor.c_ne);
        let p = [
            eskf.nominal.pn as f64,
            eskf.nominal.pe as f64,
            eskf.nominal.pd as f64,
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
    pub fn align_debug(&self) -> Option<FusionAlignDebug> {
        Some(FusionAlignDebug {
            window: self.last_align_window?,
            trace: self.last_align_trace?,
        })
    }

    /// Diagnostic hook that directly sets the ESKF residual mount quaternion.
    pub fn analysis_set_eskf_mount_quat(&mut self, q_cs: [f32; 4]) {
        if !self.ekf_initialized {
            return;
        }
        if self.cfg.eskf_mount_source == EskfMountSource::FollowAlign {
            return;
        }
        let mut q = q_cs;
        quat_normalize(&mut q);
        let n = &mut self.eskf.raw_mut().nominal;
        n.qcs0 = q[0];
        n.qcs1 = q[1];
        n.qcs2 = q[2];
        n.qcs3 = q[3];
    }

    /// Diagnostic hook that directly sets residual-mount covariance.
    pub fn analysis_set_eskf_mount_covariance(&mut self, sigma_rad: f32, zero_cross: bool) {
        if !self.ekf_initialized || !sigma_rad.is_finite() || sigma_rad < 0.0 {
            return;
        }
        let var = sigma_rad * sigma_rad;
        for i in 15..18 {
            if zero_cross {
                for j in 0..18 {
                    self.eskf.raw_mut().p[i][j] = 0.0;
                    self.eskf.raw_mut().p[j][i] = 0.0;
                }
            }
            self.eskf.raw_mut().p[i][i] = var;
        }
        if self.effective_freeze_misalignment_states() {
            self.eskf.set_freeze_misalignment_states(true);
        }
    }

    fn effective_freeze_misalignment_states(&self) -> bool {
        self.cfg.freeze_misalignment_states
            || self.cfg.eskf_mount_source == EskfMountSource::FollowAlign
    }

    fn refresh_follow_align_mount_source(&mut self) {
        if self.cfg.eskf_mount_source != EskfMountSource::FollowAlign {
            return;
        }
        if let Some(q_vb) = self.mount_q_vb {
            self.eskf_mount_q_vb = Some(q_vb);
        }
        self.reset_eskf_mount_residual();
    }

    fn reset_eskf_mount_residual(&mut self) {
        let n = &mut self.eskf.raw_mut().nominal;
        n.qcs0 = 1.0;
        n.qcs1 = 0.0;
        n.qcs2 = 0.0;
        n.qcs3 = 0.0;
    }

    fn update(&self, mount_ready_changed: bool, ekf_initialized_now: bool) -> FusionUpdate {
        FusionUpdate {
            mount_ready: self.mount_ready,
            mount_ready_changed,
            ekf_initialized: self.ekf_initialized,
            ekf_initialized_now,
            mount_q_vb: self.mount_q_vb,
        }
    }

    fn initialize_eskf_from_gnss(&mut self, gnss: EskfGnssSample) {
        self.eskf = RustEskf::new(self.cfg.predict_noise);
        self.eskf
            .set_freeze_misalignment_states(self.effective_freeze_misalignment_states());
        let speed_h = (gnss.vel_ned_mps[0] * gnss.vel_ned_mps[0]
            + gnss.vel_ned_mps[1] * gnss.vel_ned_mps[1])
            .sqrt();
        let yaw = gnss.heading_rad.unwrap_or_else(|| {
            if speed_h >= self.cfg.yaw_init_speed_mps.max(1.0) {
                gnss.vel_ned_mps[1].atan2(gnss.vel_ned_mps[0])
            } else {
                0.0
            }
        });
        self.eskf.init_nominal_from_gnss(quat_from_yaw(yaw), gnss);
        let raw = self.eskf.raw_mut();
        raw.p[9][9] = self.cfg.gyro_bias_init_sigma_radps.powi(2);
        raw.p[10][10] = raw.p[9][9];
        raw.p[11][11] = raw.p[9][9];
        raw.p[12][12] = self.cfg.accel_bias_init_sigma_mps2.powi(2);
        raw.p[13][13] = raw.p[12][12];
        raw.p[14][14] = raw.p[12][12];
        raw.p[2][2] = self.cfg.yaw_init_sigma_rad.powi(2);
        let mount_var = self.cfg.mount_init_sigma_rad.powi(2);
        raw.p[15][15] = 0.0;
        raw.p[16][16] = mount_var;
        raw.p[17][17] = mount_var;
        if self.effective_freeze_misalignment_states() {
            self.eskf.set_freeze_misalignment_states(true);
        }
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
            self.eskf.set_freeze_misalignment_states(true);
            return;
        }
        if self.cfg.mount_settle_time_s <= 0.0 {
            self.eskf.set_freeze_misalignment_states(false);
            return;
        }
        let Some(handoff_t_s) = self.ekf_mount_handoff_t_s else {
            return;
        };
        if t_s - handoff_t_s < self.cfg.mount_settle_time_s {
            self.eskf.set_freeze_misalignment_states(true);
            return;
        }
        if !self.mount_settle_released {
            self.eskf.set_freeze_misalignment_states(false);
            self.reset_mount_covariance_after_settle();
            self.mount_settle_released = true;
        } else {
            self.eskf.set_freeze_misalignment_states(false);
        }
    }

    fn reset_mount_covariance_after_settle(&mut self) {
        let sigma = self.cfg.mount_settle_release_sigma_rad;
        if !sigma.is_finite() || sigma < 0.0 {
            return;
        }
        let var = sigma * sigma;
        let raw = self.eskf.raw_mut();
        for i in 15..18 {
            if self.cfg.mount_settle_zero_cross_covariance {
                for j in 0..18 {
                    raw.p[i][j] = 0.0;
                    raw.p[j][i] = 0.0;
                }
            }
            raw.p[i][i] = var;
        }
        raw.p[15][15] = 0.0;
    }

    fn fuse_signed_body_speed(&mut self, t_s: f32, signed_speed_mps: f32) {
        let scale = self.mount_update_scale(t_s);
        let effective_r = self.cfg.r_vehicle_speed / scale.max(1.0e-3);
        self.eskf.fuse_body_speed_x_scaled(
            signed_speed_mps,
            effective_r,
            scale,
            self.cfg.mount_update_innovation_gate_mps,
        );
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
                self.mount_q_vb = Some(self.align.q_vb);
            }
        }
    }

    fn prepare_local_gnss_sample(&mut self, sample: FusionGnssSample) -> Option<EskfGnssSample> {
        if !self.anchor.valid {
            self.anchor = anchor_from_lla(sample.lat_deg, sample.lon_deg, sample.height_m);
        }
        let ecef = lla_to_ecef(
            sample.lat_deg as f64,
            sample.lon_deg as f64,
            sample.height_m as f64,
        );
        let mut pos_ned = ecef_to_anchor_ned(&self.anchor, ecef);
        let horiz_dist = (pos_ned[0] * pos_ned[0] + pos_ned[1] * pos_ned[1]).sqrt();
        if horiz_dist > REANCHOR_DISTANCE_M {
            let new_anchor = anchor_from_lla(sample.lat_deg, sample.lon_deg, sample.height_m);
            self.rotate_nav_state_to_new_anchor(&new_anchor);
            self.anchor = new_anchor;
            self.reanchor_count += 1;
            self.last_reanchor = Some((sample.t_s, horiz_dist));
            self.bootstrap_prev_gnss = None;
            self.reset_interval_summary();
            pos_ned = [0.0; 3];
        }
        Some(EskfGnssSample {
            t_s: sample.t_s,
            pos_ned_m: pos_ned,
            vel_ned_mps: velocity_local_ned_to_anchor_ned(
                &self.anchor,
                sample.lat_deg,
                sample.lon_deg,
                sample.vel_ned_mps,
            ),
            pos_std_m: sample.pos_std_m,
            vel_std_mps: sample.vel_std_mps,
            heading_rad: sample
                .heading_rad
                .map(|h| heading_local_to_anchor(&self.anchor, sample.lat_deg, sample.lon_deg, h)),
        })
    }

    fn rotate_nav_state_to_new_anchor(&mut self, new_anchor: &Anchor) {
        if !self.anchor.valid || !new_anchor.valid {
            return;
        }
        let c_en_old = transpose3(self.anchor.c_ne);
        let r_n1_n0 = mat3_mul(new_anchor.c_ne, c_en_old);

        if self.ekf_initialized {
            let raw = self.eskf.raw_mut();
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
            let v_n1 = mat3_vec(r_n1_n0, v_n0);
            raw.nominal.pn = p_n1[0];
            raw.nominal.pe = p_n1[1];
            raw.nominal.pd = p_n1[2];
            raw.nominal.vn = v_n1[0];
            raw.nominal.ve = v_n1[1];
            raw.nominal.vd = v_n1[2];

            let q_n1_n0 = rotmat_to_quat(r_n1_n0);
            let q_old = [
                raw.nominal.q0,
                raw.nominal.q1,
                raw.nominal.q2,
                raw.nominal.q3,
            ];
            let mut q_new = quat_mul(q_n1_n0, q_old);
            quat_normalize(&mut q_new);
            raw.nominal.q0 = q_new[0];
            raw.nominal.q1 = q_new[1];
            raw.nominal.q2 = q_new[2];
            raw.nominal.q3 = q_new[3];
            rotate_eskf_covariance_nav_blocks(&mut raw.p, r_n1_n0);
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
            last.vel_ned_mps = mat3_vec(r_n1_n0, v_n0);
            last.heading_rad = None;
            self.last_gnss = Some(last);
        }

        if let Some(mut prev) = self.bootstrap_prev_gnss {
            prev.vel_ned_mps = mat3_vec(r_n1_n0, prev.vel_ned_mps);
            self.bootstrap_prev_gnss = Some(prev);
        }
    }

    fn bootstrap_update_gnss_hints(&mut self, sample: EskfGnssSample) {
        let speed = horiz_speed(sample.vel_ned_mps);
        self.bootstrap_speed_ema
            .update(speed, self.cfg.bootstrap.ema_alpha);
        let Some(prev) = self.bootstrap_prev_gnss else {
            return;
        };
        let dt = sample.t_s - prev.t_s;
        if dt <= 1.0e-3 {
            return;
        }
        let prev_speed = horiz_speed(prev.vel_ned_mps);
        let speed_rate = (speed - prev_speed) / dt;
        let course_prev = prev.vel_ned_mps[1].atan2(prev.vel_ned_mps[0]);
        let course_curr = sample.vel_ned_mps[1].atan2(sample.vel_ned_mps[0]);
        let course_rate = wrap_pi(course_curr - course_prev) / dt;
        self.bootstrap_speed_rate_ema
            .update(speed_rate.abs(), self.cfg.bootstrap.ema_alpha);
        self.bootstrap_course_rate_ema
            .update(course_rate.abs(), self.cfg.bootstrap.ema_alpha);
    }

    fn bootstrap_update(&mut self, accel_b: [f32; 3], gyro_radps: [f32; 3]) -> bool {
        let gyro_norm = norm3(gyro_radps);
        let accel_err = (norm3(accel_b) - GRAVITY_MPS2).abs();
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
        prev_gnss: EskfGnssSample,
        curr_gnss: EskfGnssSample,
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
            .update(norm3(gyro_radps), self.cfg.bootstrap.ema_alpha);
        let accel_ema = self.bootstrap_accel_err_ema.update(
            (norm3(accel_b) - GRAVITY_MPS2).abs(),
            self.cfg.bootstrap.ema_alpha,
        );
        let low_dynamic = gyro_ema <= self.cfg.bootstrap.max_gyro_radps
            && accel_ema <= self.cfg.bootstrap.max_accel_norm_err_mps2;
        let low_speed = self
            .last_gnss
            .is_some_and(|g| horiz_speed(g.vel_ned_mps) <= RUNTIME_ZERO_SPEED_MPS);
        low_dynamic && low_speed
    }

    fn runtime_nhc_speed_scale(&self) -> f32 {
        let Some(last) = self.last_gnss else {
            return 0.0;
        };
        let speed = horiz_speed(last.vel_ned_mps);
        let full = 15.0 / 3.6;
        if speed <= 0.0 {
            0.0
        } else if speed >= full {
            1.0
        } else {
            speed / full
        }
    }

    fn gnss_vel_update_scale_xy(&self, t_s: f32) -> f32 {
        ramp_scale(
            self.cfg.gnss_vel_xy_update_min_scale,
            self.cfg.gnss_vel_update_ramp_time_s,
            self.ekf_mount_handoff_t_s,
            t_s,
            1.0,
        )
    }

    fn mount_update_scale(&self, t_s: f32) -> f32 {
        ramp_scale(
            self.cfg.mount_update_min_scale,
            self.cfg.mount_update_ramp_time_s,
            self.ekf_mount_handoff_t_s,
            t_s,
            1.0,
        )
    }

    fn mount_yaw_observability_scale(&self, gyro_vehicle: [f32; 3]) -> f32 {
        let gate = self.cfg.mount_update_yaw_rate_gate_radps;
        if gate <= 0.0 {
            return 1.0;
        }
        (gyro_vehicle[2].abs() / gate).clamp(0.0, 1.0)
    }

    fn clamp_eskf_biases(&mut self) {
        let n = &mut self.eskf.raw_mut().nominal;
        let max_gyro = 1.5_f32.to_radians();
        let max_accel = 1.5;
        n.bgx = n.bgx.clamp(-max_gyro, max_gyro);
        n.bgy = n.bgy.clamp(-max_gyro, max_gyro);
        n.bgz = n.bgz.clamp(-max_gyro, max_gyro);
        n.bax = n.bax.clamp(-max_accel, max_accel);
        n.bay = n.bay.clamp(-max_accel, max_accel);
        n.baz = n.baz.clamp(-max_accel, max_accel);
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

fn ramp_scale(
    min_scale: f32,
    ramp_time_s: f32,
    handoff_t_s: Option<f32>,
    t_s: f32,
    fallback: f32,
) -> f32 {
    let mut min_scale = if min_scale.is_finite() {
        min_scale
    } else {
        fallback
    };
    min_scale = min_scale.clamp(0.0, 1.0);
    let Some(handoff) = handoff_t_s else {
        return fallback;
    };
    if ramp_time_s <= 0.0 || !ramp_time_s.is_finite() {
        return fallback;
    }
    let age = t_s - handoff;
    if age <= 0.0 {
        return min_scale;
    }
    let ramp = (age / ramp_time_s).clamp(0.0, 1.0);
    min_scale + (1.0 - min_scale) * ramp
}

fn runtime_nhc_active(accel_v: [f32; 3], gyro_v: [f32; 3]) -> bool {
    let roll_pitch_rate = (gyro_v[0] * gyro_v[0] + gyro_v[1] * gyro_v[1]).sqrt();
    roll_pitch_rate < RUNTIME_NHC_MAX_ROLL_PITCH_GYRO_RADPS
        && (norm3(accel_v) - GRAVITY_MPS2).abs() < RUNTIME_NHC_MAX_ACCEL_NORM_ERR_MPS2
}

fn body_speed_x_estimate(eskf: &EskfState) -> f32 {
    let n = &eskf.nominal;
    let vs0 = (1.0 - 2.0 * n.q2 * n.q2 - 2.0 * n.q3 * n.q3) * n.vn
        + 2.0 * (n.q1 * n.q2 + n.q0 * n.q3) * n.ve
        + 2.0 * (n.q1 * n.q3 - n.q0 * n.q2) * n.vd;
    let vs1 = 2.0 * (n.q1 * n.q2 - n.q0 * n.q3) * n.vn
        + (1.0 - 2.0 * n.q1 * n.q1 - 2.0 * n.q3 * n.q3) * n.ve
        + 2.0 * (n.q2 * n.q3 + n.q0 * n.q1) * n.vd;
    let vs2 = 2.0 * (n.q1 * n.q3 + n.q0 * n.q2) * n.vn
        + 2.0 * (n.q2 * n.q3 - n.q0 * n.q1) * n.ve
        + (1.0 - 2.0 * n.q1 * n.q1 - 2.0 * n.q2 * n.q2) * n.vd;
    (1.0 - 2.0 * n.qcs2 * n.qcs2 - 2.0 * n.qcs3 * n.qcs3) * vs0
        + 2.0 * (n.qcs1 * n.qcs2 - n.qcs0 * n.qcs3) * vs1
        + 2.0 * (n.qcs1 * n.qcs3 + n.qcs0 * n.qcs2) * vs2
}

fn anchor_from_lla(lat_deg: f32, lon_deg: f32, height_m: f32) -> Anchor {
    Anchor {
        valid: true,
        lat_deg,
        lon_deg,
        height_m,
        ecef_m: lla_to_ecef(lat_deg as f64, lon_deg as f64, height_m as f64),
        c_ne: ecef_to_ned_matrix(lat_deg, lon_deg),
    }
}

fn lla_to_ecef(lat_deg: f64, lon_deg: f64, height_m: f64) -> [f64; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    let n = WGS84_A_M / (1.0 - WGS84_E2 * slat * slat).sqrt();
    [
        (n + height_m) * clat * clon,
        (n + height_m) * clat * slon,
        (n * (1.0 - WGS84_E2) + height_m) * slat,
    ]
}

fn ecef_to_lla_f64(ecef_m: [f64; 3]) -> [f64; 3] {
    let x = ecef_m[0];
    let y = ecef_m[1];
    let z = ecef_m[2];
    let b = WGS84_A_M * (1.0 - WGS84_E2).sqrt();
    let ep2 = (WGS84_A_M * WGS84_A_M - b * b) / (b * b);
    let p = (x * x + y * y).sqrt();
    let th = (WGS84_A_M * z).atan2(b * p);
    let lon = y.atan2(x);
    let lat = (z + ep2 * b * th.sin().powi(3)).atan2(p - WGS84_E2 * WGS84_A_M * th.cos().powi(3));
    let n = WGS84_A_M / (1.0 - WGS84_E2 * lat.sin().powi(2)).sqrt();
    let h = p / lat.cos() - n;
    [lat.to_degrees(), lon.to_degrees(), h]
}

fn ecef_to_anchor_ned(anchor: &Anchor, ecef_m: [f64; 3]) -> [f32; 3] {
    let diff = [
        (ecef_m[0] - anchor.ecef_m[0]) as f32,
        (ecef_m[1] - anchor.ecef_m[1]) as f32,
        (ecef_m[2] - anchor.ecef_m[2]) as f32,
    ];
    mat3_vec(anchor.c_ne, diff)
}

fn velocity_local_ned_to_anchor_ned(
    anchor: &Anchor,
    lat_deg: f32,
    lon_deg: f32,
    vel_local_ned_mps: [f32; 3],
) -> [f32; 3] {
    let c_ne_local = ecef_to_ned_matrix(lat_deg, lon_deg);
    let c_en_local = transpose3(c_ne_local);
    let vel_ecef = mat3_vec(c_en_local, vel_local_ned_mps);
    mat3_vec(anchor.c_ne, vel_ecef)
}

fn heading_local_to_anchor(
    anchor: &Anchor,
    lat_deg: f32,
    lon_deg: f32,
    heading_local_rad: f32,
) -> f32 {
    let forward_local = [heading_local_rad.cos(), heading_local_rad.sin(), 0.0];
    let c_ne_local = ecef_to_ned_matrix(lat_deg, lon_deg);
    let c_en_local = transpose3(c_ne_local);
    let forward_ecef = mat3_vec(c_en_local, forward_local);
    let forward_anchor = mat3_vec(anchor.c_ne, forward_ecef);
    forward_anchor[1].atan2(forward_anchor[0])
}

fn ecef_to_ned_matrix(lat_deg: f32, lon_deg: f32) -> [[f32; 3]; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    [
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [-clat * clon, -clat * slon, -slat],
    ]
}

fn horiz_speed(vel_ned_mps: [f32; 3]) -> f32 {
    (vel_ned_mps[0] * vel_ned_mps[0] + vel_ned_mps[1] * vel_ned_mps[1]).sqrt()
}

fn quat_to_rotmat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    let inv = if n2 > 1.0e-9 { 1.0 / n2.sqrt() } else { 1.0 };
    let q0 = q[0] * inv;
    let q1 = q[1] * inv;
    let q2 = q[2] * inv;
    let q3 = q[3] * inv;
    [
        [
            1.0 - 2.0 * (q2 * q2 + q3 * q3),
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * (q1 * q1 + q3 * q3),
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * (q1 * q1 + q2 * q2),
        ],
    ]
}

fn quat_from_yaw(yaw_rad: f32) -> [f32; 4] {
    let half = 0.5 * yaw_rad;
    [half.cos(), 0.0, 0.0, half.sin()]
}

fn rotmat_to_quat(r: [[f32; 3]; 3]) -> [f32; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let mut q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        [
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        ]
    };
    quat_normalize(&mut q);
    q
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_normalize(q: &mut [f32; 4]) {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2 <= 1.0e-12 {
        *q = [1.0, 0.0, 0.0, 0.0];
        return;
    }
    let inv = 1.0 / n2.sqrt();
    for v in q {
        *v *= inv;
    }
}

fn transpose3(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ],
        [
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ],
    ]
}

fn mat3_vec(a: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

#[allow(clippy::needless_range_loop)]
fn rotate_eskf_covariance_nav_blocks(p: &mut [[f32; 18]; 18], r_n1_n0: [[f32; 3]; 3]) {
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

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
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
