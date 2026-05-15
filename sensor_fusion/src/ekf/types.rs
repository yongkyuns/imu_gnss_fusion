//! Public EKF-filter state, configuration, and diagnostic data layouts.

use super::generated::ERROR_STATES;
use crate::ProcessNoise;

pub const UPDATE_DIAG_TYPES: usize = 11;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NominalState {
    /// NED/navigation-frame attitude with respect to the vehicle frame.
    ///
    /// Scalar-first `q_nv`, with `R(q_nv) = C_nv` and `x_n = C_nv x_v`.
    pub q0: f32,
    pub q1: f32,
    pub q2: f32,
    pub q3: f32,
    /// Local-NED velocity `[north, east, down]`, meters per second.
    pub vn: f32,
    pub ve: f32,
    pub vd: f32,
    /// Local-NED position `[north, east, down]`, meters from the current anchor.
    pub pn: f32,
    pub pe: f32,
    pub pd: f32,
    /// Gyro additive correction in the raw IMU body frame, radians per second.
    pub bgx: f32,
    pub bgy: f32,
    pub bgz: f32,
    /// Accelerometer additive correction in the raw IMU body frame, meters per second squared.
    pub bax: f32,
    pub bay: f32,
    pub baz: f32,
    /// Physical vehicle-to-body mount quaternion `q_bv`.
    ///
    /// `R(q_bv) = C_bv`, `x_b = C_bv x_v`; propagation uses
    /// `C_vb = C_bv^T` to rotate raw IMU increments into the vehicle frame.
    pub q_bv0: f32,
    pub q_bv1: f32,
    pub q_bv2: f32,
    pub q_bv3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StationaryDiag {
    pub innovation_x: f32,
    pub innovation_y: f32,
    pub k_theta_x_from_x: f32,
    pub k_theta_y_from_x: f32,
    pub k_bax_from_x: f32,
    pub k_bay_from_x: f32,
    pub k_theta_x_from_y: f32,
    pub k_theta_y_from_y: f32,
    pub k_bax_from_y: f32,
    pub k_bay_from_y: f32,
    pub p_theta_x: f32,
    pub p_theta_y: f32,
    pub p_bax: f32,
    pub p_bay: f32,
    pub p_theta_x_bax: f32,
    pub p_theta_y_bay: f32,
    pub updates: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct UpdateDiag {
    pub total_updates: u32,
    pub type_counts: [u32; UPDATE_DIAG_TYPES],
    pub sum_dx_att_roll: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_att_roll: [f32; UPDATE_DIAG_TYPES],
    pub sum_dx_yaw: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_yaw: [f32; UPDATE_DIAG_TYPES],
    pub sum_dx_pitch: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_pitch: [f32; UPDATE_DIAG_TYPES],
    pub sum_dx_vel_n: [f32; UPDATE_DIAG_TYPES],
    pub sum_dx_vel_e: [f32; UPDATE_DIAG_TYPES],
    pub sum_dx_vel_d: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_vel_n: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_vel_e: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_vel_d: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_vel_h: [f32; UPDATE_DIAG_TYPES],
    pub sum_dx_gyro_bias_z: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_gyro_bias_z: [f32; UPDATE_DIAG_TYPES],
    pub sum_dx_gyro_bias: [[f32; 3]; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_gyro_bias: [[f32; 3]; UPDATE_DIAG_TYPES],
    pub sum_dx_accel_bias: [[f32; 3]; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_accel_bias: [[f32; 3]; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_mount_norm: [f32; UPDATE_DIAG_TYPES],
    pub sum_dx_mount_roll: [f32; UPDATE_DIAG_TYPES],
    pub sum_dx_mount_pitch: [f32; UPDATE_DIAG_TYPES],
    pub sum_dx_mount_yaw: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_mount_roll: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_mount_pitch: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_dx_mount_yaw: [f32; UPDATE_DIAG_TYPES],
    pub sum_innovation: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_innovation: [f32; UPDATE_DIAG_TYPES],
    pub sum_nis: [f32; UPDATE_DIAG_TYPES],
    pub max_nis: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_h_yaw: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_h_gyro_bias_z: [f32; UPDATE_DIAG_TYPES],
    pub sum_h_mount_norm: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_k_yaw: [f32; UPDATE_DIAG_TYPES],
    pub sum_k_mount_norm: [f32; UPDATE_DIAG_TYPES],
    pub sum_abs_corr_yaw_mount_yaw: [f32; UPDATE_DIAG_TYPES],
    pub last_dx_mount_roll: f32,
    pub last_dx_mount_pitch: f32,
    pub last_dx_mount_yaw: f32,
    pub last_dx_mount_roll_by_type: [f32; UPDATE_DIAG_TYPES],
    pub last_dx_mount_pitch_by_type: [f32; UPDATE_DIAG_TYPES],
    pub last_dx_mount_yaw_by_type: [f32; UPDATE_DIAG_TYPES],
    pub last_k_mount_yaw: f32,
    pub last_innovation: f32,
    pub last_innovation_by_type: [f32; UPDATE_DIAG_TYPES],
    pub last_innovation_var: f32,
    pub last_nis: f32,
    pub last_nis_by_type: [f32; UPDATE_DIAG_TYPES],
    pub last_h_mount_norm: f32,
    pub last_h_mount_norm_by_type: [f32; UPDATE_DIAG_TYPES],
    pub last_corr_yaw_mount_yaw: f32,
    pub last_type: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct State {
    pub nominal: NominalState,
    pub p: [[f32; ERROR_STATES]; ERROR_STATES],
    pub noise: ProcessNoise,
    pub stationary_diag: StationaryDiag,
    pub update_diag: UpdateDiag,
    pub last_dx: [f32; ERROR_STATES],
    pub last_h_by_obs: [[f32; ERROR_STATES]; 8],
    pub last_dx_by_obs: [[f32; ERROR_STATES]; 8],
    pub last_k_by_obs: [[f32; ERROR_STATES]; 8],
    pub last_obs_count: i32,
    pub last_obs_types: [i32; 8],
    pub last_residuals: [f32; 8],
    pub last_effective_residuals: [f32; 8],
    pub last_innovation_vars: [f32; 8],
}

impl Default for State {
    fn default() -> Self {
        Self {
            nominal: NominalState {
                q_bv0: 1.0,
                ..NominalState::default()
            },
            p: [[0.0; ERROR_STATES]; ERROR_STATES],
            noise: ProcessNoise::lsm6dso_104hz(),
            stationary_diag: StationaryDiag::default(),
            update_diag: UpdateDiag::default(),
            last_dx: [0.0; ERROR_STATES],
            last_h_by_obs: [[0.0; ERROR_STATES]; 8],
            last_dx_by_obs: [[0.0; ERROR_STATES]; 8],
            last_k_by_obs: [[0.0; ERROR_STATES]; 8],
            last_obs_count: 0,
            last_obs_types: [0; 8],
            last_residuals: [0.0; 8],
            last_effective_residuals: [0.0; 8],
            last_innovation_vars: [0.0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ImuDelta {
    /// Body-frame delta angle `[x, y, z]`, radians.
    pub dax: f32,
    pub day: f32,
    pub daz: f32,
    /// Body-frame delta velocity `[x, y, z]`, meters per second.
    pub dvx: f32,
    pub dvy: f32,
    pub dvz: f32,
    /// Integration interval, seconds.
    pub dt: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct GnssSample {
    /// Timestamp, seconds.
    pub t_s: f32,
    /// Local-NED position `[north, east, down]`, meters from the current anchor.
    pub pos_ned_m: [f32; 3],
    /// Local-NED velocity `[north, east, down]`, meters per second.
    pub vel_ned_mps: [f32; 3],
    /// One-sigma local-NED position standard deviations, meters.
    pub pos_std_m: [f32; 3],
    /// One-sigma local-NED velocity standard deviations, meters per second.
    pub vel_std_mps: [f32; 3],
    /// Optional local-NED heading, radians clockwise from north toward east.
    pub heading_rad: Option<f32>,
}
