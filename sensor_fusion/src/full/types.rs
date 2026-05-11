//! Public Full-filter state, configuration, and diagnostic data layouts.

use crate::{ProcessNoise, math::sq_f32};

/// Number of full-filter error-state components.
pub const ERROR_STATES: usize = 24;
/// Number of full-filter process-noise components.
pub const NOISE_STATES: usize = 21;

/// Initial covariance settings used when seeding [`crate::full::Filter`]
/// through the high-level `SensorFusion` facade.
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase", default))]
#[derive(Clone, Copy, Debug)]
pub struct InitConfig {
    /// Minimum one-sigma position uncertainty, in meters.
    pub pos_min_sigma_m: f32,
    /// Minimum one-sigma velocity uncertainty, in meters per second.
    pub vel_min_sigma_mps: f32,
    /// Initial attitude one-sigma uncertainty, in degrees.
    pub attitude_sigma_deg: f32,
    /// Initial gyro-bias one-sigma uncertainty, in degrees per second.
    pub gyro_bias_sigma_dps: f32,
    /// Initial accelerometer-bias one-sigma uncertainty, in meters per second squared.
    pub accel_bias_sigma_mps2: f32,
    /// Initial gyro-scale one-sigma uncertainty.
    pub gyro_scale_sigma: f32,
    /// Initial accelerometer-scale one-sigma uncertainty.
    pub accel_scale_sigma: f32,
    /// Initial mount roll/pitch one-sigma uncertainty, in degrees.
    pub mount_sigma_deg: f32,
    /// Initial mount yaw one-sigma uncertainty, in degrees.
    pub mount_yaw_sigma_deg: f32,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self {
            pos_min_sigma_m: 0.5,
            vel_min_sigma_mps: 0.2,
            attitude_sigma_deg: 20.0,
            gyro_bias_sigma_dps: 0.125,
            accel_bias_sigma_mps2: 0.15,
            gyro_scale_sigma: 0.02,
            accel_scale_sigma: 0.0,
            mount_sigma_deg: 2.0,
            mount_yaw_sigma_deg: 6.0,
        }
    }
}

pub(crate) fn default_full_p_diag(
    pos_std_m: [f32; 3],
    vel_std_mps: [f32; 3],
    init: InitConfig,
) -> [f32; ERROR_STATES] {
    let mut p = [1.0_f32; ERROR_STATES];

    let pos_n_sigma = pos_std_m[0].max(init.pos_min_sigma_m);
    let pos_e_sigma = pos_std_m[1].max(init.pos_min_sigma_m);
    let pos_d_sigma = pos_std_m[2].max(init.pos_min_sigma_m);
    p[0] = pos_n_sigma * pos_n_sigma;
    p[1] = pos_e_sigma * pos_e_sigma;
    p[2] = pos_d_sigma * pos_d_sigma;

    let vel_sigma = vel_std_mps
        .into_iter()
        .fold(0.0_f32, f32::max)
        .max(init.vel_min_sigma_mps);
    let vel_var = vel_sigma * vel_sigma;
    p[3] = vel_var;
    p[4] = vel_var;
    p[5] = vel_var;

    let attitude_var = sq_f32(init.attitude_sigma_deg.to_radians());
    p[6] = attitude_var;
    p[7] = attitude_var;
    p[8] = attitude_var;

    let gyro_bias_sigma = init.gyro_bias_sigma_dps.to_radians();
    p[9] = init.accel_bias_sigma_mps2 * init.accel_bias_sigma_mps2;
    p[10] = p[9];
    p[11] = p[9];
    p[12] = gyro_bias_sigma * gyro_bias_sigma;
    p[13] = p[12];
    p[14] = p[12];

    p[15] = init.accel_scale_sigma * init.accel_scale_sigma;
    p[16] = p[15];
    p[17] = p[15];
    p[18] = init.gyro_scale_sigma * init.gyro_scale_sigma;
    p[19] = p[18];
    p[20] = p[18];

    let mount_var = sq_f32(init.mount_sigma_deg.to_radians());
    p[21] = mount_var;
    p[22] = mount_var;
    p[23] = sq_f32(init.mount_yaw_sigma_deg.to_radians());
    p
}

/// Full-filter nominal state.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NominalState {
    /// ECEF-frame attitude with respect to the vehicle frame.
    ///
    /// Scalar-first `q_ev`, with `R(q_ev) = C_ev` and `x_e = C_ev x_v`.
    pub q0: f32,
    pub q1: f32,
    pub q2: f32,
    pub q3: f32,
    /// ECEF velocity `[x, y, z]`, meters per second.
    pub vn: f32,
    pub ve: f32,
    pub vd: f32,
    /// ECEF position `[x, y, z]`, meters.
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
    /// Gyro scale correction in the raw IMU body frame.
    pub sgx: f32,
    pub sgy: f32,
    pub sgz: f32,
    /// Accelerometer scale correction in the raw IMU body frame.
    pub sax: f32,
    pub say: f32,
    pub saz: f32,
    /// Physical vehicle-to-body mount quaternion `q_bv`.
    ///
    /// `R(q_bv) = C_bv`, `x_b = C_bv x_v`; propagation uses
    /// `C_vb = C_bv^T` to rotate raw IMU samples into the vehicle frame.
    pub q_bv0: f32,
    pub q_bv1: f32,
    pub q_bv2: f32,
    pub q_bv3: f32,
}

/// Two-sample IMU increment used by the full ECEF propagation model.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ImuDelta {
    /// First body-frame delta angle sample, radians.
    pub dax_1: f32,
    pub day_1: f32,
    pub daz_1: f32,
    /// First body-frame delta velocity sample, meters per second.
    pub dvx_1: f32,
    pub dvy_1: f32,
    pub dvz_1: f32,
    /// Second body-frame delta angle sample, radians.
    pub dax_2: f32,
    pub day_2: f32,
    pub daz_2: f32,
    /// Second body-frame delta velocity sample, meters per second.
    pub dvx_2: f32,
    pub dvy_2: f32,
    pub dvz_2: f32,
    /// Integration interval, seconds.
    pub dt: f32,
}

/// Diagnostic snapshot for the most recent full GNSS position gate attempt.
#[derive(Clone, Copy, Debug, Default)]
pub struct GnssPositionGateDiag {
    pub attempted: bool,
    pub accepted: bool,
    pub whitened_residual: [f32; 3],
    pub innovation_sigma: [f32; 3],
    pub normalized_residual: [f32; 3],
}

/// Complete full-filter state and diagnostics.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct State {
    pub nominal: NominalState,
    pub p: [[f32; ERROR_STATES]; ERROR_STATES],
    pub noise: ProcessNoise,
    pub pos_e64: [f64; 3],
    pub q_bv64: [f64; 4],
    pub last_dx: [f32; ERROR_STATES],
    pub last_dx_by_obs: [[f32; ERROR_STATES]; 8],
    pub last_obs_count: i32,
    pub last_obs_types: [i32; 8],
    pub last_residuals: [f32; 8],
    pub last_effective_residuals: [f32; 8],
    pub last_innovation_vars: [f32; 8],
    pub last_gnss_pos_gate: GnssPositionGateDiag,
}

impl Default for State {
    fn default() -> Self {
        let mut p = [[0.0; ERROR_STATES]; ERROR_STATES];
        for i in 0..ERROR_STATES {
            p[i][i] = 1.0;
        }
        Self {
            nominal: NominalState {
                q0: 1.0,
                q_bv0: 1.0,
                ..NominalState::default()
            },
            p,
            noise: ProcessNoise::reference_nsr_demo(),
            pos_e64: [0.0; 3],
            q_bv64: [1.0, 0.0, 0.0, 0.0],
            last_dx: [0.0; ERROR_STATES],
            last_dx_by_obs: [[0.0; ERROR_STATES]; 8],
            last_obs_count: 0,
            last_obs_types: [0; 8],
            last_residuals: [0.0; 8],
            last_effective_residuals: [0.0; 8],
            last_innovation_vars: [0.0; 8],
            last_gnss_pos_gate: GnssPositionGateDiag::default(),
        }
    }
}
