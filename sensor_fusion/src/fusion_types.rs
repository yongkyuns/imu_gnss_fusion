//! Public facade types and internal configuration for [`crate::SensorFusion`].

use crate::align::{AlignConfig, AlignUpdateTrace, AlignWindowSummary};
use crate::{ProcessNoise, full};

/// One timestamped IMU sample in the sensor/body frame.
#[derive(Clone, Copy, Debug)]
pub struct ImuSample {
    /// Sample timestamp, in seconds.
    pub t_s: f32,
    /// Angular rate in the body frame, in radians per second.
    pub gyro_radps: [f32; 3],
    /// Specific force in the body frame, in meters per second squared.
    pub accel_mps2: [f32; 3],
}

/// One timestamped GNSS sample in geodetic coordinates with NED velocity.
#[derive(Clone, Copy, Debug)]
pub struct GnssSample {
    /// Sample timestamp, in seconds.
    pub t_s: f32,
    /// Geodetic latitude, in degrees.
    pub lat_deg: f64,
    /// Geodetic longitude, in degrees.
    pub lon_deg: f64,
    /// Ellipsoidal height, in meters.
    pub height_m: f64,
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
pub struct VehicleSpeedSample {
    /// Sample timestamp, in seconds.
    pub t_s: f32,
    /// Nonnegative speed magnitude, in meters per second.
    pub speed_mps: f32,
    /// Direction associated with `speed_mps`.
    pub direction: VehicleSpeedDirection,
}

/// Direction qualifier for a vehicle-speed sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VehicleSpeedDirection {
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
pub struct Update {
    /// Whether a mount estimate is ready for Reduced initialization or propagation.
    pub mount_ready: bool,
    /// Whether `mount_ready` changed during this input sample.
    pub mount_ready_changed: bool,
    /// Whether the Reduced has been initialized from GNSS.
    pub reduced_initialized: bool,
    /// Whether Reduced initialization happened during this input sample.
    pub reduced_initialized_now: bool,
    /// Whether the configured runtime filter has been initialized from GNSS.
    pub filter_initialized: bool,
    /// Whether the configured runtime filter initialized during this input sample.
    pub filter_initialized_now: bool,
    /// Current vehicle-to-body mount quaternion, when available.
    pub mount_q_vb: Option<[f32; 4]>,
}

/// Runtime filter selected by [`Config`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Filter {
    /// Reduced-state local-NED EKF.
    #[default]
    Reduced,
    /// Full-state ECEF EKF.
    Full,
}

/// Last alignment window and update trace captured by [`crate::SensorFusion`].
#[derive(Clone, Copy, Debug)]
pub struct AlignDebug {
    /// Window statistics passed to the align filter.
    pub window: AlignWindowSummary,
    /// Detailed alignment update trace for the same window.
    pub trace: AlignUpdateTrace,
}

/// Selects how the runtime obtains the IMU-to-vehicle mount estimate.
#[derive(Clone, Copy, Debug)]
pub enum MountMode {
    /// Estimate mount internally with [`crate::align::Align`] before Reduced initialization.
    InternalAlign,
    /// Use the supplied vehicle-to-body mount quaternion and disable internal alignment.
    External([f32; 4]),
}

/// Selects how Reduced propagation receives the mount after alignment handoff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MountSource {
    /// Latch the alignment seed at Reduced initialization and let residual Reduced mount states evolve.
    LatchedSeed,
    /// Keep following the current align mount and freeze residual Reduced mount states.
    FollowAlign,
}

/// Public construction configuration for [`crate::SensorFusion`].
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Runtime filter family selected by the caller.
    pub filter: Filter,
    /// Source used for the initial physical vehicle-to-body mount estimate.
    pub mount_mode: MountMode,
    /// Source used by the Reduced filter after alignment handoff.
    pub mount_source: MountSource,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            filter: Filter::Reduced,
            mount_mode: MountMode::InternalAlign,
            mount_source: MountSource::LatchedSeed,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BootstrapConfig {
    pub(crate) ema_alpha: f32,
    pub(crate) max_speed_mps: f32,
    pub(crate) max_speed_rate_mps2: f32,
    pub(crate) max_course_rate_radps: f32,
    pub(crate) stationary_samples: u32,
    pub(crate) max_gyro_radps: f32,
    pub(crate) max_accel_norm_err_mps2: f32,
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
pub(crate) struct NoiseConfig {
    pub(crate) reduced: ProcessNoise,
    pub(crate) full: ProcessNoise,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            reduced: ProcessNoise {
                gyro_var: 2.287_311_3e-7 * 10.0_f32,
                accel_var: 2.450_421_4e-5 * 15.0_f32,
                gyro_bias_rw_var: 0.0002e-9,
                accel_bias_rw_var: 0.002e-9,
                gyro_scale_rw_var: 0.0,
                accel_scale_rw_var: 0.0,
                mount_align_rw_var: 0.0,
            },
            full: ProcessNoise::lsm6dso_104hz(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RuntimeConfig {
    pub(crate) filter: Filter,
    pub(crate) align: AlignConfig,
    pub(crate) bootstrap: BootstrapConfig,
    pub(crate) noise: NoiseConfig,
    pub(crate) full_init: full::InitConfig,
    pub(crate) attitude_roll_pitch_init_sigma_rad: f32,
    pub(crate) yaw_init_sigma_rad: f32,
    pub(crate) gyro_bias_init_sigma_radps: f32,
    pub(crate) accel_bias_init_sigma_mps2: f32,
    pub(crate) mount_roll_pitch_init_sigma_rad: f32,
    pub(crate) mount_init_sigma_rad: f32,
    pub(crate) r_body_vel_y: f32,
    pub(crate) r_body_vel_z: f32,
    pub(crate) align_handoff_delay_s: f32,
    pub(crate) freeze_misalignment_states: bool,
    pub(crate) mount_source: MountSource,
    pub(crate) mount_settle_time_s: f32,
    pub(crate) mount_settle_release_sigma_rad: f32,
    pub(crate) mount_settle_zero_cross_covariance: bool,
    pub(crate) r_vehicle_speed: f32,
    pub(crate) r_zero_vel: f32,
    pub(crate) r_stationary_accel: f32,
    pub(crate) yaw_init_speed_mps: f32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            filter: Filter::Reduced,
            align: AlignConfig::default(),
            bootstrap: BootstrapConfig::default(),
            noise: NoiseConfig::default(),
            full_init: full::InitConfig::default(),
            attitude_roll_pitch_init_sigma_rad: 2.0_f32.to_radians(),
            yaw_init_sigma_rad: 6.0_f32.to_radians(),
            gyro_bias_init_sigma_radps: 0.125_f32.to_radians(),
            accel_bias_init_sigma_mps2: 0.15,
            mount_roll_pitch_init_sigma_rad: 1.2_f32.to_radians(),
            mount_init_sigma_rad: 6.0_f32.to_radians(),
            r_body_vel_y: 0.5,
            r_body_vel_z: 0.05,
            align_handoff_delay_s: 0.0,
            freeze_misalignment_states: false,
            mount_source: MountSource::LatchedSeed,
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
