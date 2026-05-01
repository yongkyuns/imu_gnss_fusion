//! Public state and sample structs for the runtime augmented ESKF.
//!
//! Mathematical details are maintained in `docs/eskf_mount_formulation.pdf`.
//! `EskfNominalState` stores `q_ns, v_n, p_n, b_g, b_a, q_cs`, and the
//! covariance in `EskfState` is ordered as:
//!
//! ```text
//! dtheta_s, dv_n, dp_n, dbg, dba, dpsi_cs
//! ```

use crate::ekf::PredictNoise;

pub const ESKF_UPDATE_DIAG_TYPES: usize = 11;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct EskfNominalState {
    pub q0: f32,
    pub q1: f32,
    pub q2: f32,
    pub q3: f32,
    pub vn: f32,
    pub ve: f32,
    pub vd: f32,
    pub pn: f32,
    pub pe: f32,
    pub pd: f32,
    pub bgx: f32,
    pub bgy: f32,
    pub bgz: f32,
    pub bax: f32,
    pub bay: f32,
    pub baz: f32,
    pub qcs0: f32,
    pub qcs1: f32,
    pub qcs2: f32,
    pub qcs3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct EskfStationaryDiag {
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
pub struct EskfUpdateDiag {
    pub total_updates: u32,
    pub type_counts: [u32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_dx_yaw: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_dx_yaw: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_dx_pitch: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_dx_pitch: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_dx_vel_h: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_dx_gyro_bias_z: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_dx_gyro_bias_z: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_dx_mount_norm: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_dx_mount_yaw: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_dx_mount_yaw: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_innovation: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_innovation: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_nis: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub max_nis: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_h_yaw: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_h_gyro_bias_z: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_h_mount_norm: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_k_yaw: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_k_mount_norm: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_corr_yaw_mount_yaw: [f32; ESKF_UPDATE_DIAG_TYPES],
    pub last_dx_mount_roll: f32,
    pub last_dx_mount_pitch: f32,
    pub last_dx_mount_yaw: f32,
    pub last_k_mount_yaw: f32,
    pub last_innovation: f32,
    pub last_innovation_var: f32,
    pub last_nis: f32,
    pub last_h_mount_norm: f32,
    pub last_corr_yaw_mount_yaw: f32,
    pub last_type: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EskfState {
    pub nominal: EskfNominalState,
    pub p: [[f32; 18]; 18],
    pub noise: PredictNoise,
    pub stationary_diag: EskfStationaryDiag,
    pub update_diag: EskfUpdateDiag,
}

impl Default for EskfState {
    fn default() -> Self {
        Self {
            nominal: EskfNominalState {
                qcs0: 1.0,
                ..EskfNominalState::default()
            },
            p: [[0.0; 18]; 18],
            noise: PredictNoise::lsm6dso_typical_104hz(),
            stationary_diag: EskfStationaryDiag::default(),
            update_diag: EskfUpdateDiag::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct EskfImuDelta {
    pub dax: f32,
    pub day: f32,
    pub daz: f32,
    pub dvx: f32,
    pub dvy: f32,
    pub dvz: f32,
    pub dt: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct EskfGnssSample {
    pub t_s: f32,
    pub pos_ned_m: [f32; 3],
    pub vel_ned_mps: [f32; 3],
    pub pos_std_m: [f32; 3],
    pub vel_std_mps: [f32; 3],
    pub heading_rad: Option<f32>,
}
