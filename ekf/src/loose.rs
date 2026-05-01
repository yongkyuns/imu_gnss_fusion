//! Loose INS/GNSS reference filter.
//!
//! This module implements a loose-coupled ECEF reference filter used for
//! diagnostics and comparison against the runtime ESKF. It keeps a single
//! precision public state plus f64 shadow position and mount quaternion fields
//! used internally for numerically sensitive propagation.
//!
//! See `docs/loose_formulation.pdf` for the PDF-first derivation. The nominal
//! state is:
//!
//! ```text
//! q_es, v_e, p_e, b_g, b_a, s_g, s_a, q_cs
//! ```
//!
//! `q_es` rotates the seeded IMU frame into ECEF, and `q_cs` rotates the seeded
//! frame into the corrected vehicle frame used by NHC. The 24-state error order
//! in generated matrices and injection is:
//!
//! ```text
//! dp_e, dv_e, dtheta_s, dba, dbg, dsa, dsg, dpsi_cs
//! ```
//!
//! GNSS reference rows observe ECEF position and velocity, optionally whitening
//! NED standard deviations into ECEF. NHC rows predict
//! `v_c = C_cs C_es^T v_e` and constrain its lateral and vertical components.

#![allow(non_snake_case)]
#![allow(clippy::excessive_precision)]

use crate::generated_loose;

/// Number of loose-filter error-state components.
pub const LOOSE_ERROR_STATES: usize = 24;
/// Number of loose-filter process-noise components.
pub const LOOSE_NOISE_STATES: usize = 21;

const WGS84_A: f32 = 6_378_137.0;
const WGS84_B: f32 = 6_356_752.314_245_18;
const WGS84_E2: f32 = 6.694_379_990_141_32e-3;
const WGS84_OMEGA_IE: f32 = 7.292_115e-5;
const WGS84_GM: f32 = 3.986_004_418e14;
const WGS84_J2: f32 = 1.082_629_821_368_57e-3;

const GPS_REF_SUPPORT_ROW0: &[usize] = &[0];
const GPS_REF_SUPPORT_ROW1: &[usize] = &[0, 1];
const GPS_REF_SUPPORT_ROW2: &[usize] = &[0, 1, 2];
const VEL_REF_SUPPORT_ROW0: &[usize] = &[3];
const VEL_REF_SUPPORT_ROW1: &[usize] = &[3, 4];
const VEL_REF_SUPPORT_ROW2: &[usize] = &[3, 4, 5];
const VEL_AXIS_SUPPORT_ROW0: &[usize] = &[3];
const VEL_AXIS_SUPPORT_ROW1: &[usize] = &[4];
const VEL_AXIS_SUPPORT_ROW2: &[usize] = &[5];
const MIN_NHC_UPDATE_SPEED_MPS: f32 = 0.5;

/// Process-noise variances used by [`LooseFilter::predict`].
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
#[derive(Debug, Clone, Copy)]
pub struct LoosePredictNoise {
    /// Gyro white-noise variance.
    pub gyro_var: f32,
    /// Accelerometer white-noise variance.
    pub accel_var: f32,
    /// Gyro-bias random-walk variance.
    pub gyro_bias_rw_var: f32,
    /// Accelerometer-bias random-walk variance.
    pub accel_bias_rw_var: f32,
    /// Gyro-scale random-walk variance.
    pub gyro_scale_rw_var: f32,
    /// Accelerometer-scale random-walk variance.
    pub accel_scale_rw_var: f32,
    /// Residual mount-alignment random-walk variance.
    pub mount_align_rw_var: f32,
}

impl Default for LoosePredictNoise {
    fn default() -> Self {
        Self::lsm6dso_loose_104hz()
    }
}

impl LoosePredictNoise {
    /// Reference noise profile used by original NSR-style loose-filter demos.
    pub const fn reference_nsr_demo() -> Self {
        Self {
            gyro_var: 2.5e-5,
            accel_var: 9.0e-4,
            gyro_bias_rw_var: 1.0e-12,
            accel_bias_rw_var: 1.0e-10,
            gyro_scale_rw_var: 1.0e-10,
            accel_scale_rw_var: 1.0e-10,
            mount_align_rw_var: 1.0e-8,
        }
    }

    /// LSM6DSO-oriented loose-filter noise profile for 104 Hz IMU data.
    pub const fn lsm6dso_loose_104hz() -> Self {
        Self {
            gyro_var: 2.287_311_3e-7 * 10.0_f32,
            accel_var: 2.450_421_4e-5 * 15.0_f32,
            gyro_bias_rw_var: 0.0002e-9,
            accel_bias_rw_var: 0.002e-9,
            gyro_scale_rw_var: 1.0e-10,
            accel_scale_rw_var: 1.0e-10,
            mount_align_rw_var: 1.0e-8,
        }
    }
}

/// Loose-filter nominal state.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LooseNominalState {
    /// ECEF/seed-frame attitude quaternion scalar component.
    pub q0: f32,
    /// ECEF/seed-frame attitude quaternion X component.
    pub q1: f32,
    /// ECEF/seed-frame attitude quaternion Y component.
    pub q2: f32,
    /// ECEF/seed-frame attitude quaternion Z component.
    pub q3: f32,
    /// ECEF-frame velocity X component, in meters per second.
    pub vn: f32,
    /// ECEF-frame velocity Y component, in meters per second.
    pub ve: f32,
    /// ECEF-frame velocity Z component, in meters per second.
    pub vd: f32,
    /// ECEF-frame position X component, in meters.
    pub pn: f32,
    /// ECEF-frame position Y component, in meters.
    pub pe: f32,
    /// ECEF-frame position Z component, in meters.
    pub pd: f32,
    /// Gyro X additive correction state, in radians per second.
    pub bgx: f32,
    /// Gyro Y additive correction state, in radians per second.
    pub bgy: f32,
    /// Gyro Z additive correction state, in radians per second.
    pub bgz: f32,
    /// Accelerometer X additive correction state, in meters per second squared.
    pub bax: f32,
    /// Accelerometer Y additive correction state, in meters per second squared.
    pub bay: f32,
    /// Accelerometer Z additive correction state, in meters per second squared.
    pub baz: f32,
    /// Gyro X scale correction.
    pub sgx: f32,
    /// Gyro Y scale correction.
    pub sgy: f32,
    /// Gyro Z scale correction.
    pub sgz: f32,
    /// Accelerometer X scale correction.
    pub sax: f32,
    /// Accelerometer Y scale correction.
    pub say: f32,
    /// Accelerometer Z scale correction.
    pub saz: f32,
    /// Residual seed-to-vehicle quaternion scalar component.
    pub qcs0: f32,
    /// Residual seed-to-vehicle quaternion X component.
    pub qcs1: f32,
    /// Residual seed-to-vehicle quaternion Y component.
    pub qcs2: f32,
    /// Residual seed-to-vehicle quaternion Z component.
    pub qcs3: f32,
}

/// Two-sample IMU increment used by the loose ECEF propagation model.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LooseImuDelta {
    /// First gyro X delta angle, in radians.
    pub dax_1: f32,
    /// First gyro Y delta angle, in radians.
    pub day_1: f32,
    /// First gyro Z delta angle, in radians.
    pub daz_1: f32,
    /// First accelerometer X delta velocity, in meters per second.
    pub dvx_1: f32,
    /// First accelerometer Y delta velocity, in meters per second.
    pub dvy_1: f32,
    /// First accelerometer Z delta velocity, in meters per second.
    pub dvz_1: f32,
    /// Second gyro X delta angle, in radians.
    pub dax_2: f32,
    /// Second gyro Y delta angle, in radians.
    pub day_2: f32,
    /// Second gyro Z delta angle, in radians.
    pub daz_2: f32,
    /// Second accelerometer X delta velocity, in meters per second.
    pub dvx_2: f32,
    /// Second accelerometer Y delta velocity, in meters per second.
    pub dvy_2: f32,
    /// Second accelerometer Z delta velocity, in meters per second.
    pub dvz_2: f32,
    /// Delta duration, in seconds.
    pub dt: f32,
}

/// Complete loose-filter state and diagnostics.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LooseState {
    /// Nominal navigation, sensor-error, and residual-mount state.
    pub nominal: LooseNominalState,
    /// Public f32 covariance matrix.
    pub p: [[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
    /// Process-noise configuration.
    pub noise: LoosePredictNoise,
    /// f64 ECEF position shadow used by propagation.
    pub pos_e64: [f64; 3],
    /// f64 residual mount quaternion shadow.
    pub qcs64: [f64; 4],
    /// Last injected error-state correction.
    pub last_dx: [f32; LOOSE_ERROR_STATES],
    /// Number of observation rows used by the last batch update.
    pub last_obs_count: i32,
    /// Observation type identifiers used by the last batch update.
    pub last_obs_types: [i32; 8],
}

impl Default for LooseState {
    fn default() -> Self {
        let mut p = [[0.0; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES];
        for i in 0..LOOSE_ERROR_STATES {
            p[i][i] = 1.0;
        }
        Self {
            nominal: LooseNominalState {
                q0: 1.0,
                qcs0: 1.0,
                ..LooseNominalState::default()
            },
            p,
            noise: LoosePredictNoise::reference_nsr_demo(),
            pos_e64: [0.0; 3],
            qcs64: [1.0, 0.0, 0.0, 0.0],
            last_dx: [0.0; LOOSE_ERROR_STATES],
            last_obs_count: 0,
            last_obs_types: [0; 8],
        }
    }
}

/// Loose-coupled ECEF INS/GNSS filter used for reference comparisons.
#[derive(Clone, Debug)]
pub struct LooseFilter {
    raw: LooseState,
}

impl LooseFilter {
    /// Creates a loose filter with identity attitude/mount and the supplied process noise.
    pub fn new(noise: LoosePredictNoise) -> Self {
        let mut raw = LooseState {
            noise,
            ..LooseState::default()
        };
        raw.nominal.q0 = 1.0;
        raw.nominal.qcs0 = 1.0;
        raw.qcs64 = [1.0, 0.0, 0.0, 0.0];
        Self { raw }
    }

    /// Returns the full loose-filter state.
    pub fn raw(&self) -> &LooseState {
        &self.raw
    }

    /// Returns the nominal loose-filter state.
    pub fn nominal(&self) -> &LooseNominalState {
        &self.raw.nominal
    }

    /// Returns the public f32 covariance matrix.
    pub fn covariance(&self) -> &[[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES] {
        &self.raw.p
    }

    /// Returns the f64 ECEF position shadow used internally.
    pub fn shadow_pos_ecef(&self) -> [f64; 3] {
        self.raw.pos_e64
    }

    /// Returns the observation type identifiers from the last batch update.
    pub fn last_obs_types(&self) -> &[i32] {
        let count = self
            .raw
            .last_obs_count
            .clamp(0, self.raw.last_obs_types.len() as i32) as usize;
        &self.raw.last_obs_types[..count]
    }

    /// Returns the last injected error-state correction.
    pub fn last_dx(&self) -> &[f32; LOOSE_ERROR_STATES] {
        &self.raw.last_dx
    }

    /// Replaces covariance.
    pub fn set_covariance(&mut self, p: [[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES]) {
        self.raw.p = p;
    }

    /// Sets the residual seed-to-vehicle mount quaternion.
    pub fn set_mount_quat(&mut self, q_cs: [f32; 4]) {
        self.raw.nominal.qcs0 = q_cs[0];
        self.raw.nominal.qcs1 = q_cs[1];
        self.raw.nominal.qcs2 = q_cs[2];
        self.raw.nominal.qcs3 = q_cs[3];
        self.raw.qcs64 = [
            q_cs[0] as f64,
            q_cs[1] as f64,
            q_cs[2] as f64,
            q_cs[3] as f64,
        ];
    }

    /// Zeros residual-mount cross-covariances and sets mount variance from a degree sigma.
    pub fn tighten_mount_covariance_deg(&mut self, sigma_deg: f32) {
        let mut p = self.raw.p;
        let var = (sigma_deg as f64).to_radians().powi(2) as f32;
        for i in 21..24 {
            for j in 0..LOOSE_ERROR_STATES {
                p[i][j] = 0.0;
                p[j][i] = 0.0;
            }
            p[i][i] = var;
        }
        self.set_covariance(p);
    }

    #[allow(clippy::too_many_arguments)]
    /// Initializes the loose filter from a reference-frame state.
    pub fn init_from_reference_state(
        &mut self,
        q_bn: [f32; 4],
        pos_ned_m: [f32; 3],
        vel_ned_mps: [f32; 3],
        gyro_bias_radps: [f32; 3],
        accel_bias_mps2: [f32; 3],
        gyro_scale: [f32; 3],
        accel_scale: [f32; 3],
        q_cs: [f32; 4],
        p_diag: Option<[f32; LOOSE_ERROR_STATES]>,
    ) {
        self.raw.nominal.q0 = q_bn[0];
        self.raw.nominal.q1 = q_bn[1];
        self.raw.nominal.q2 = q_bn[2];
        self.raw.nominal.q3 = q_bn[3];
        self.raw.nominal.vn = vel_ned_mps[0];
        self.raw.nominal.ve = vel_ned_mps[1];
        self.raw.nominal.vd = vel_ned_mps[2];
        self.raw.nominal.pn = pos_ned_m[0];
        self.raw.nominal.pe = pos_ned_m[1];
        self.raw.nominal.pd = pos_ned_m[2];
        self.raw.pos_e64 = [
            pos_ned_m[0] as f64,
            pos_ned_m[1] as f64,
            pos_ned_m[2] as f64,
        ];
        self.raw.nominal.bgx = gyro_bias_radps[0];
        self.raw.nominal.bgy = gyro_bias_radps[1];
        self.raw.nominal.bgz = gyro_bias_radps[2];
        self.raw.nominal.bax = accel_bias_mps2[0];
        self.raw.nominal.bay = accel_bias_mps2[1];
        self.raw.nominal.baz = accel_bias_mps2[2];
        self.raw.nominal.sgx = gyro_scale[0];
        self.raw.nominal.sgy = gyro_scale[1];
        self.raw.nominal.sgz = gyro_scale[2];
        self.raw.nominal.sax = accel_scale[0];
        self.raw.nominal.say = accel_scale[1];
        self.raw.nominal.saz = accel_scale[2];
        self.set_mount_quat(q_cs);
        if let Some(p_diag) = p_diag {
            for i in 0..LOOSE_ERROR_STATES {
                for j in 0..LOOSE_ERROR_STATES {
                    self.raw.p[i][j] = 0.0;
                }
            }
            for (i, value) in p_diag.into_iter().enumerate() {
                self.raw.p[i][i] = value;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Initializes the loose filter from an ECEF reference-frame state.
    pub fn init_from_reference_ecef_state(
        &mut self,
        q_es: [f32; 4],
        pos_ecef_m: [f64; 3],
        vel_ecef_mps: [f32; 3],
        gyro_bias_radps: [f32; 3],
        accel_bias_mps2: [f32; 3],
        gyro_scale: [f32; 3],
        accel_scale: [f32; 3],
        q_cs: [f32; 4],
        p_diag: Option<[f32; LOOSE_ERROR_STATES]>,
    ) {
        self.init_from_reference_state(
            q_es,
            [
                pos_ecef_m[0] as f32,
                pos_ecef_m[1] as f32,
                pos_ecef_m[2] as f32,
            ],
            vel_ecef_mps,
            gyro_bias_radps,
            accel_bias_mps2,
            gyro_scale,
            accel_scale,
            q_cs,
            p_diag,
        );
        self.raw.pos_e64 = pos_ecef_m;
    }

    /// Initializes from ECEF navigation state with a yaw-seeded vehicle frame split.
    #[allow(clippy::too_many_arguments)]
    pub fn init_seeded_vehicle_from_nav_ecef_state(
        &mut self,
        yaw_rad: f32,
        lat_deg: f64,
        lon_deg: f64,
        pos_ecef_m: [f64; 3],
        vel_ecef_mps: [f32; 3],
        p_diag: Option<[f32; LOOSE_ERROR_STATES]>,
        residual_mount_sigma_deg: Option<f32>,
    ) {
        let (q_es, q_cs) = loose_seeded_vehicle_ecef_split(yaw_rad, lat_deg, lon_deg);
        self.init_from_reference_ecef_state(
            q_es,
            pos_ecef_m,
            vel_ecef_mps,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            q_cs,
            p_diag,
        );
        if let Some(sigma_deg) = residual_mount_sigma_deg {
            self.tighten_mount_covariance_deg(sigma_deg);
        }
    }

    /// Predicts nominal state and covariance from one two-sample IMU delta.
    pub fn predict(&mut self, imu: LooseImuDelta) {
        if imu.dt <= 0.0 {
            return;
        }
        self.predict_nominal(imu);
        let (f, g) = self.compute_error_transition(imu);
        let dt = imu.dt;
        let mut q = [0.0; LOOSE_NOISE_STATES];
        q[0] = self.raw.noise.accel_var * dt;
        q[1] = q[0];
        q[2] = q[0];
        q[3] = self.raw.noise.gyro_var * dt;
        q[4] = q[3];
        q[5] = q[3];
        q[6] = self.raw.noise.accel_bias_rw_var * dt;
        q[7] = q[6];
        q[8] = q[6];
        q[9] = self.raw.noise.gyro_bias_rw_var * dt;
        q[10] = q[9];
        q[11] = q[9];
        q[12] = self.raw.noise.accel_scale_rw_var * dt;
        q[13] = q[12];
        q[14] = q[12];
        q[15] = self.raw.noise.gyro_scale_rw_var * dt;
        q[16] = q[15];
        q[17] = q[15];
        q[18] = self.raw.noise.mount_align_rw_var * dt;
        q[19] = q[18];
        q[20] = q[18];

        self.raw.p = predict_covariance_sparse(&f, &g, &self.raw.p, &q);
    }

    /// Predicts only the loose nominal state from one two-sample IMU delta.
    pub fn predict_nominal(&mut self, imu: LooseImuDelta) {
        let dt = imu.dt;
        if dt <= 0.0 {
            return;
        }

        let q_es = [
            self.raw.nominal.q0,
            self.raw.nominal.q1,
            self.raw.nominal.q2,
            self.raw.nominal.q3,
        ];
        let v_e = [
            self.raw.nominal.vn,
            self.raw.nominal.ve,
            self.raw.nominal.vd,
        ];
        let x_e = self.raw.pos_e64;
        let b_w = [
            self.raw.nominal.bgx,
            self.raw.nominal.bgy,
            self.raw.nominal.bgz,
        ];
        let b_f = [
            self.raw.nominal.bax,
            self.raw.nominal.bay,
            self.raw.nominal.baz,
        ];
        let s_w = [
            self.raw.nominal.sgx,
            self.raw.nominal.sgy,
            self.raw.nominal.sgz,
        ];
        let s_f = [
            self.raw.nominal.sax,
            self.raw.nominal.say,
            self.raw.nominal.saz,
        ];
        let omega1 = [
            s_w[0] * (imu.dax_1 / dt) + b_w[0],
            s_w[1] * (imu.day_1 / dt) + b_w[1],
            s_w[2] * (imu.daz_1 / dt) + b_w[2],
        ];
        let f1 = [
            s_f[0] * (imu.dvx_1 / dt) + b_f[0],
            s_f[1] * (imu.dvy_1 / dt) + b_f[1],
            s_f[2] * (imu.dvz_1 / dt) + b_f[2],
        ];
        let omega2 = [
            s_w[0] * (imu.dax_2 / dt) + b_w[0],
            s_w[1] * (imu.day_2 / dt) + b_w[1],
            s_w[2] * (imu.daz_2 / dt) + b_w[2],
        ];
        let f2 = [
            s_f[0] * (imu.dvx_2 / dt) + b_f[0],
            s_f[1] * (imu.dvy_2 / dt) + b_f[1],
            s_f[2] * (imu.dvz_2 / dt) + b_f[2],
        ];

        let x_e_f = [x_e[0] as f32, x_e[1] as f32, x_e[2] as f32];
        let c_es = quat_to_dcm(q_es);
        let g_e1 = gravity_ecef_j2(x_e_f);
        let f1_e = mat_vec3(c_es, f1);
        let vdot1 = [
            g_e1[0] + f1_e[0] + 2.0 * WGS84_OMEGA_IE * v_e[1],
            g_e1[1] + f1_e[1] - 2.0 * WGS84_OMEGA_IE * v_e[0],
            g_e1[2] + f1_e[2],
        ];
        let qdot1_raw = quat_multiply(q_es, [0.0, omega1[0], omega1[1], omega1[2]]);
        let qdot1 = [
            0.5 * (qdot1_raw[0] + WGS84_OMEGA_IE * q_es[3]),
            0.5 * (qdot1_raw[1] + WGS84_OMEGA_IE * q_es[2]),
            0.5 * (qdot1_raw[2] - WGS84_OMEGA_IE * q_es[1]),
            0.5 * (qdot1_raw[3] - WGS84_OMEGA_IE * q_es[0]),
        ];

        let x_tmp = [
            x_e[0] + dt as f64 * v_e[0] as f64,
            x_e[1] + dt as f64 * v_e[1] as f64,
            x_e[2] + dt as f64 * v_e[2] as f64,
        ];
        let v_tmp = [
            v_e[0] + dt * vdot1[0],
            v_e[1] + dt * vdot1[1],
            v_e[2] + dt * vdot1[2],
        ];
        let mut q_tmp = [
            q_es[0] + dt * qdot1[0],
            q_es[1] + dt * qdot1[1],
            q_es[2] + dt * qdot1[2],
            q_es[3] + dt * qdot1[3],
        ];
        normalize_quat(&mut q_tmp);

        let c_es_tmp = quat_to_dcm(q_tmp);
        let x_tmp_f = [x_tmp[0] as f32, x_tmp[1] as f32, x_tmp[2] as f32];
        let g_e2 = gravity_ecef_j2(x_tmp_f);
        let f2_e = mat_vec3(c_es_tmp, f2);
        let vdot2 = [
            g_e2[0] + f2_e[0] + 2.0 * WGS84_OMEGA_IE * v_tmp[1],
            g_e2[1] + f2_e[1] - 2.0 * WGS84_OMEGA_IE * v_tmp[0],
            g_e2[2] + f2_e[2],
        ];
        let qdot2_raw = quat_multiply(q_tmp, [0.0, omega2[0], omega2[1], omega2[2]]);
        let qdot2 = [
            0.5 * (qdot2_raw[0] + WGS84_OMEGA_IE * q_tmp[3]),
            0.5 * (qdot2_raw[1] + WGS84_OMEGA_IE * q_tmp[2]),
            0.5 * (qdot2_raw[2] - WGS84_OMEGA_IE * q_tmp[1]),
            0.5 * (qdot2_raw[3] - WGS84_OMEGA_IE * q_tmp[0]),
        ];

        self.raw.pos_e64[0] = x_e[0] + 0.5 * dt as f64 * (v_e[0] + v_tmp[0]) as f64;
        self.raw.pos_e64[1] = x_e[1] + 0.5 * dt as f64 * (v_e[1] + v_tmp[1]) as f64;
        self.raw.pos_e64[2] = x_e[2] + 0.5 * dt as f64 * (v_e[2] + v_tmp[2]) as f64;
        self.sync_nominal_position_from_shadow();
        self.raw.nominal.vn = v_e[0] + 0.5 * dt * (vdot1[0] + vdot2[0]);
        self.raw.nominal.ve = v_e[1] + 0.5 * dt * (vdot1[1] + vdot2[1]);
        self.raw.nominal.vd = v_e[2] + 0.5 * dt * (vdot1[2] + vdot2[2]);
        self.raw.nominal.q0 = q_es[0] + 0.5 * dt * (qdot1[0] + qdot2[0]);
        self.raw.nominal.q1 = q_es[1] + 0.5 * dt * (qdot1[1] + qdot2[1]);
        self.raw.nominal.q2 = q_es[2] + 0.5 * dt * (qdot1[2] + qdot2[2]);
        self.raw.nominal.q3 = q_es[3] + 0.5 * dt * (qdot1[3] + qdot2[3]);
        self.normalize_nominal_quat();
    }

    /// Computes generated loose-filter transition and noise-input matrices.
    pub fn compute_error_transition(
        &self,
        imu: LooseImuDelta,
    ) -> (
        [[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
        [[f32; LOOSE_NOISE_STATES]; LOOSE_ERROR_STATES],
    ) {
        if imu.dt <= 0.0 {
            return (
                [[0.0; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
                [[0.0; LOOSE_NOISE_STATES]; LOOSE_ERROR_STATES],
            );
        }
        generated_loose::error_transition(&self.raw.nominal, imu)
    }

    /// Fuses ECEF GNSS position and optional velocity as reference observations.
    pub fn fuse_gps_reference(
        &mut self,
        pos_ecef_m: [f64; 3],
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        speed_acc_mps: f32,
        dt_since_last_gnss_s: f32,
    ) {
        let mut h_rows = [[0.0; LOOSE_ERROR_STATES]; 8];
        let mut h_supports: [Option<&'static [usize]>; 8] = [None; 8];
        let mut residuals = [0.0; 8];
        let mut variances = [0.0; 8];
        let mut obs_types = [0; 8];
        let obs_count = self.append_reference_gps_observations(
            Some(pos_ecef_m),
            vel_ecef_mps,
            h_acc_m,
            speed_acc_mps,
            None,
            dt_since_last_gnss_s,
            &mut h_rows,
            &mut h_supports,
            &mut residuals,
            &mut variances,
            Some(&mut obs_types),
            0,
        );
        if obs_count > 0 {
            self.batch_update_joseph(obs_count, &h_rows, &h_supports, &residuals, &variances);
        }
    }

    /// Fuses ECEF GNSS position and optional velocity with per-axis velocity standard deviations.
    pub fn fuse_gps_reference_full(
        &mut self,
        pos_ecef_m: [f64; 3],
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        vel_std_ned_mps: Option<[f32; 3]>,
        dt_since_last_gnss_s: f32,
    ) {
        let mut h_rows = [[0.0; LOOSE_ERROR_STATES]; 8];
        let mut h_supports: [Option<&'static [usize]>; 8] = [None; 8];
        let mut residuals = [0.0; 8];
        let mut variances = [0.0; 8];
        let mut obs_types = [0; 8];
        let obs_count = self.append_reference_gps_observations(
            Some(pos_ecef_m),
            vel_ecef_mps,
            h_acc_m,
            0.0,
            vel_std_ned_mps,
            dt_since_last_gnss_s,
            &mut h_rows,
            &mut h_supports,
            &mut residuals,
            &mut variances,
            Some(&mut obs_types),
            0,
        );
        if obs_count > 0 {
            self.batch_update_joseph(obs_count, &h_rows, &h_supports, &residuals, &variances);
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Fuses position, velocity, and nonholonomic reference observations in one batch.
    pub fn fuse_reference_batch(
        &mut self,
        pos_ecef_m: Option<[f64; 3]>,
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        speed_acc_mps: f32,
        dt_since_last_gnss_s: f32,
        gyro_radps: [f32; 3],
        accel_mps2: [f32; 3],
        dt_s: f32,
    ) {
        self.fuse_reference_batch_impl(
            pos_ecef_m,
            vel_ecef_mps,
            h_acc_m,
            speed_acc_mps,
            None,
            dt_since_last_gnss_s,
            gyro_radps,
            accel_mps2,
            dt_s,
        );
    }

    #[allow(clippy::too_many_arguments)]
    /// Fuses a full reference batch with optional per-axis velocity standard deviations.
    pub fn fuse_reference_batch_full(
        &mut self,
        pos_ecef_m: Option<[f64; 3]>,
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        vel_std_ned_mps: Option<[f32; 3]>,
        dt_since_last_gnss_s: f32,
        gyro_radps: [f32; 3],
        accel_mps2: [f32; 3],
        dt_s: f32,
    ) {
        self.fuse_reference_batch_impl(
            pos_ecef_m,
            vel_ecef_mps,
            h_acc_m,
            0.0,
            vel_std_ned_mps,
            dt_since_last_gnss_s,
            gyro_radps,
            accel_mps2,
            dt_s,
        );
    }

    /// Fuses loose-filter nonholonomic lateral and vertical velocity constraints.
    pub fn fuse_nhc_reference(&mut self, gyro_radps: [f32; 3], accel_mps2: [f32; 3], dt_s: f32) {
        if dt_s <= 0.0
            || !self.nhc_gate_allows(gyro_radps, accel_mps2)
            || self.vehicle_speed_mps() < MIN_NHC_UPDATE_SPEED_MPS
        {
            return;
        }
        let (vc_y_est, h_y) = generated_loose::nhc_y(&self.raw.nominal);
        let (vc_z_est, h_z) = generated_loose::nhc_z(&self.raw.nominal);
        let gate_var_y = 0.1_f32 * 0.1_f32;
        let gate_var_z = 0.05_f32 * 0.05_f32;
        let mut dt_obs = dt_s;
        if dt_obs <= 0.0 || dt_obs >= 1.0 {
            dt_obs = 1.0;
        }
        let var_y = 0.01 * (gate_var_y / dt_obs);
        let var_z = 0.01 * (gate_var_z / dt_obs);
        let mut h_rows = [[0.0; LOOSE_ERROR_STATES]; 8];
        let mut h_supports: [Option<&'static [usize]>; 8] = [None; 8];
        let mut residuals = [0.0; 8];
        let mut variances = [0.0; 8];
        let mut obs_count = 0;
        if !test_chi2_scalar(
            -vc_y_est,
            &self.raw.p,
            &h_y,
            gate_var_y,
            &generated_loose::NHC_Y_SUPPORT,
        ) {
            h_rows[obs_count] = h_y;
            h_supports[obs_count] = Some(&generated_loose::NHC_Y_SUPPORT);
            residuals[obs_count] = -vc_y_est;
            variances[obs_count] = var_y;
            obs_count += 1;
        }
        if !test_chi2_scalar(
            -vc_z_est,
            &self.raw.p,
            &h_z,
            gate_var_z,
            &generated_loose::NHC_Z_SUPPORT,
        ) {
            h_rows[obs_count] = h_z;
            h_supports[obs_count] = Some(&generated_loose::NHC_Z_SUPPORT);
            residuals[obs_count] = -vc_z_est;
            variances[obs_count] = var_z;
            obs_count += 1;
        }
        if obs_count > 0 {
            self.batch_update_joseph(obs_count, &h_rows, &h_supports, &residuals, &variances);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fuse_reference_batch_impl(
        &mut self,
        pos_ecef_m: Option<[f64; 3]>,
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        speed_acc_mps: f32,
        vel_std_ned_mps: Option<[f32; 3]>,
        dt_since_last_gnss_s: f32,
        gyro_radps: [f32; 3],
        accel_mps2: [f32; 3],
        dt_s: f32,
    ) {
        if dt_s <= 0.0 {
            return;
        }
        self.raw.last_obs_count = 0;
        self.raw.last_obs_types = [0; 8];

        let mut h_rows = [[0.0; LOOSE_ERROR_STATES]; 8];
        let mut h_supports: [Option<&'static [usize]>; 8] = [None; 8];
        let mut residuals = [0.0; 8];
        let mut variances = [0.0; 8];
        let mut obs_types = [0; 8];
        let mut obs_count = self.append_reference_gps_observations(
            pos_ecef_m,
            vel_ecef_mps,
            h_acc_m,
            speed_acc_mps,
            vel_std_ned_mps,
            dt_since_last_gnss_s,
            &mut h_rows,
            &mut h_supports,
            &mut residuals,
            &mut variances,
            Some(&mut obs_types),
            0,
        );

        if self.nhc_gate_allows(gyro_radps, accel_mps2)
            && self.vehicle_speed_mps() >= MIN_NHC_UPDATE_SPEED_MPS
        {
            let (vc_y_est, h_y) = generated_loose::nhc_y(&self.raw.nominal);
            let (vc_z_est, h_z) = generated_loose::nhc_z(&self.raw.nominal);
            let gate_var_y = 0.1_f32 * 0.1_f32;
            let gate_var_z = 0.05_f32 * 0.05_f32;
            let dt_obs = 0.02;
            let var_y = gate_var_y / dt_obs;
            let var_z = gate_var_z / dt_obs;
            if !test_chi2_scalar(
                -vc_y_est,
                &self.raw.p,
                &h_y,
                gate_var_y,
                &generated_loose::NHC_Y_SUPPORT,
            ) {
                h_rows[obs_count] = h_y;
                h_supports[obs_count] = Some(&generated_loose::NHC_Y_SUPPORT);
                residuals[obs_count] = -vc_y_est;
                variances[obs_count] = var_y;
                obs_types[obs_count] = 7;
                obs_count += 1;
            }
            if !test_chi2_scalar(
                -vc_z_est,
                &self.raw.p,
                &h_z,
                gate_var_z,
                &generated_loose::NHC_Z_SUPPORT,
            ) {
                h_rows[obs_count] = h_z;
                h_supports[obs_count] = Some(&generated_loose::NHC_Z_SUPPORT);
                residuals[obs_count] = -vc_z_est;
                variances[obs_count] = var_z;
                obs_types[obs_count] = 8;
                obs_count += 1;
            }
        }

        self.raw.last_obs_count = obs_count as i32;
        self.raw.last_obs_types = obs_types;
        if obs_count > 0 {
            self.batch_update_joseph(obs_count, &h_rows, &h_supports, &residuals, &variances);
        }
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn append_reference_gps_observations(
        &self,
        pos_ecef_m: Option<[f64; 3]>,
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        speed_acc_mps: f32,
        vel_std_ned_mps: Option<[f32; 3]>,
        dt_since_last_gnss_s: f32,
        h_rows: &mut [[f32; LOOSE_ERROR_STATES]; 8],
        h_supports: &mut [Option<&'static [usize]>; 8],
        residuals: &mut [f32; 8],
        variances: &mut [f32; 8],
        mut obs_types: Option<&mut [i32; 8]>,
        mut obs_count: usize,
    ) -> usize {
        if let Some(pos_ecef_m) = pos_ecef_m.filter(|_| h_acc_m > 0.0) {
            let (lat_rad, lon_rad, _) = ecef_to_llh([
                self.raw.pos_e64[0] as f32,
                self.raw.pos_e64[1] as f32,
                self.raw.pos_e64[2] as f32,
            ]);
            let c_en = dcm_ecef_to_ned(lat_rad, lon_rad);
            let r_n_diag = [
                h_acc_m * h_acc_m,
                h_acc_m * h_acc_m,
                (2.5 * h_acc_m) * (2.5 * h_acc_m),
            ];
            let mut r_e = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    for (k, r_n_k) in r_n_diag.iter().enumerate() {
                        r_e[i][j] += c_en[i][k] * *r_n_k * c_en[j][k];
                    }
                }
            }
            let u11 = libm::sqrtf(libm::fmaxf(r_e[0][0], 1.0e-9));
            let u12 = r_e[0][1] / u11;
            let u13 = r_e[0][2] / u11;
            let u22 = libm::sqrtf(libm::fmaxf(r_e[1][1] - u12 * u12, 1.0e-9));
            let u23 = (r_e[1][2] - u12 * u13) / u22;
            let u33 = libm::sqrtf(libm::fmaxf(r_e[2][2] - u13 * u13 - u23 * u23, 1.0e-9));
            let t = [
                [1.0 / u11, 0.0, 0.0],
                [-u12 / (u11 * u22), 1.0 / u22, 0.0],
                [
                    (u12 * u23 - u13 * u22) / (u11 * u22 * u33),
                    -u23 / (u22 * u33),
                    1.0 / u33,
                ],
            ];
            let mut x_meas = [0.0_f64; 3];
            let x_est = self.raw.pos_e64;
            let mut h_tmp = [[0.0; LOOSE_ERROR_STATES]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    x_meas[i] += t[i][j] as f64 * pos_ecef_m[j];
                    h_tmp[i][j] = t[i][j];
                }
            }
            let residual = [
                (x_meas[0]
                    - (t[0][0] as f64 * x_est[0]
                        + t[0][1] as f64 * x_est[1]
                        + t[0][2] as f64 * x_est[2])) as f32,
                (x_meas[1]
                    - (t[1][0] as f64 * x_est[0]
                        + t[1][1] as f64 * x_est[1]
                        + t[1][2] as f64 * x_est[2])) as f32,
                (x_meas[2]
                    - (t[2][0] as f64 * x_est[0]
                        + t[2][1] as f64 * x_est[1]
                        + t[2][2] as f64 * x_est[2])) as f32,
            ];
            let gps_supports = [
                GPS_REF_SUPPORT_ROW0,
                GPS_REF_SUPPORT_ROW1,
                GPS_REF_SUPPORT_ROW2,
            ];
            if !test_chi2_vec3(
                residual,
                &self.raw.p,
                &h_tmp,
                [1.0, 1.0, 1.0],
                &gps_supports,
            ) {
                let meas_var = 1.0 / libm::fminf(libm::fmaxf(dt_since_last_gnss_s, 1.0e-3), 1.0);
                for row in 0..3 {
                    h_rows[obs_count] = h_tmp[row];
                    h_supports[obs_count] = Some(gps_supports[row]);
                    residuals[obs_count] = residual[row];
                    variances[obs_count] = meas_var;
                    if let Some(types) = obs_types.as_deref_mut() {
                        types[obs_count] = row as i32 + 1;
                    }
                    obs_count += 1;
                }
            }
        }

        if let Some(vel_ecef_mps) = vel_ecef_mps {
            let mut vel_rows = [[0.0; LOOSE_ERROR_STATES]; 3];
            let mut vel_residual = [0.0; 3];
            let mut vel_r_diag = [1.0; 3];
            let mut vel_update_var = [1.0; 3];
            let mut vel_supports = [
                VEL_REF_SUPPORT_ROW0,
                VEL_REF_SUPPORT_ROW1,
                VEL_REF_SUPPORT_ROW2,
            ];
            if let Some(vel_std_ned_mps) = vel_std_ned_mps {
                let pos_for_llh = pos_ecef_m.unwrap_or(self.raw.pos_e64);
                let (lat_meas, lon_meas, _) = ecef_to_llh([
                    pos_for_llh[0] as f32,
                    pos_for_llh[1] as f32,
                    pos_for_llh[2] as f32,
                ]);
                let c_en_meas = dcm_ecef_to_ned(lat_meas, lon_meas);
                let mut vel_cov_n_diag = [0.0; 3];
                for i in 0..3 {
                    let std_i = libm::fmaxf(vel_std_ned_mps[i], 1.0e-3);
                    vel_cov_n_diag[i] = std_i * std_i;
                }
                let mut vel_cov_e = [[0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        for (k, cov_n_k) in vel_cov_n_diag.iter().enumerate() {
                            vel_cov_e[i][j] += c_en_meas[j][k] * *cov_n_k * c_en_meas[i][k];
                        }
                    }
                }
                let l11 = libm::sqrtf(libm::fmaxf(vel_cov_e[0][0], 1.0e-9));
                let l21 = vel_cov_e[1][0] / l11;
                let l31 = vel_cov_e[2][0] / l11;
                let l22 = libm::sqrtf(libm::fmaxf(vel_cov_e[1][1] - l21 * l21, 1.0e-9));
                let l32 = (vel_cov_e[2][1] - l31 * l21) / l22;
                let l33 = libm::sqrtf(libm::fmaxf(vel_cov_e[2][2] - l31 * l31 - l32 * l32, 1.0e-9));
                let t_vel = [
                    [1.0 / l11, 0.0, 0.0],
                    [-l21 / (l11 * l22), 1.0 / l22, 0.0],
                    [
                        (l21 * l32 - l31 * l22) / (l11 * l22 * l33),
                        -l32 / (l22 * l33),
                        1.0 / l33,
                    ],
                ];
                for row in 0..3 {
                    vel_rows[row][3] = t_vel[row][0];
                    vel_rows[row][4] = t_vel[row][1];
                    vel_rows[row][5] = t_vel[row][2];
                    vel_residual[row] = t_vel[row][0] * (vel_ecef_mps[0] - self.raw.nominal.vn)
                        + t_vel[row][1] * (vel_ecef_mps[1] - self.raw.nominal.ve)
                        + t_vel[row][2] * (vel_ecef_mps[2] - self.raw.nominal.vd);
                    vel_update_var[row] = 1.0;
                }
            } else if speed_acc_mps > 0.0 {
                let vel_var = libm::fmaxf(speed_acc_mps * speed_acc_mps, 1.0e-4);
                vel_r_diag = [vel_var; 3];
                vel_update_var = [vel_var; 3];
                vel_rows[0][3] = 1.0;
                vel_rows[1][4] = 1.0;
                vel_rows[2][5] = 1.0;
                vel_residual[0] = vel_ecef_mps[0] - self.raw.nominal.vn;
                vel_residual[1] = vel_ecef_mps[1] - self.raw.nominal.ve;
                vel_residual[2] = vel_ecef_mps[2] - self.raw.nominal.vd;
                vel_supports = [
                    VEL_AXIS_SUPPORT_ROW0,
                    VEL_AXIS_SUPPORT_ROW1,
                    VEL_AXIS_SUPPORT_ROW2,
                ];
            }
            if !test_chi2_vec3(
                vel_residual,
                &self.raw.p,
                &vel_rows,
                vel_r_diag,
                &vel_supports,
            ) {
                for row in 0..3 {
                    h_rows[obs_count] = vel_rows[row];
                    h_supports[obs_count] = Some(vel_supports[row]);
                    residuals[obs_count] = vel_residual[row];
                    variances[obs_count] = vel_update_var[row];
                    if let Some(types) = obs_types.as_deref_mut() {
                        types[obs_count] = row as i32 + 4;
                    }
                    obs_count += 1;
                }
            }
        }
        obs_count
    }

    fn nhc_gate_allows(&self, gyro_radps: [f32; 3], accel_mps2: [f32; 3]) -> bool {
        let omega_is = [
            self.raw.nominal.sgx * gyro_radps[0] + self.raw.nominal.bgx,
            self.raw.nominal.sgy * gyro_radps[1] + self.raw.nominal.bgy,
            self.raw.nominal.sgz * gyro_radps[2] + self.raw.nominal.bgz,
        ];
        let f_s = [
            self.raw.nominal.sax * accel_mps2[0] + self.raw.nominal.bax,
            self.raw.nominal.say * accel_mps2[1] + self.raw.nominal.bay,
            self.raw.nominal.saz * accel_mps2[2] + self.raw.nominal.baz,
        ];
        vec_norm3(omega_is) < 0.03 && libm::fabsf(vec_norm3(f_s) - 9.81) < 0.2
    }

    fn vehicle_speed_mps(&self) -> f32 {
        vec_norm3([
            self.raw.nominal.vn,
            self.raw.nominal.ve,
            self.raw.nominal.vd,
        ])
    }

    fn batch_update_joseph(
        &mut self,
        obs_count: usize,
        h: &[[f32; LOOSE_ERROR_STATES]; 8],
        h_supports: &[Option<&'static [usize]>; 8],
        residuals: &[f32; 8],
        variances: &[f32; 8],
    ) {
        let mut dx = [0.0; LOOSE_ERROR_STATES];
        {
            let p = &mut self.raw.p;
            let mut dense_support = [0usize; LOOSE_ERROR_STATES];
            for obs in 0..obs_count {
                let h_obs = &h[obs];
                let support = match h_supports[obs] {
                    Some(support) => support,
                    None => {
                        let len = extract_support_from_row(h_obs, &mut dense_support);
                        &dense_support[..len]
                    }
                };

                let mut ph = [0.0; LOOSE_ERROR_STATES];
                let mut s = variances[obs];
                for i in 0..LOOSE_ERROR_STATES {
                    for &state in support {
                        ph[i] += p[i][state] * h_obs[state];
                    }
                }
                for &state in support {
                    s += h_obs[state] * ph[state];
                }
                if s <= 0.0 {
                    continue;
                }
                let mut hd = 0.0;
                for &state in support {
                    hd += h_obs[state] * dx[state];
                }
                for i in 0..LOOSE_ERROR_STATES {
                    dx[i] += (ph[i] / s) * (residuals[obs] - hd);
                }
                for i in 0..LOOSE_ERROR_STATES {
                    for j in i..LOOSE_ERROR_STATES {
                        let updated = p[i][j] - (ph[i] * ph[j]) / s;
                        p[i][j] = updated;
                        p[j][i] = updated;
                    }
                }
            }
        }

        for i in 0..LOOSE_ERROR_STATES {
            self.raw.last_dx[i] = dx[i];
        }
        self.inject_error_state(dx);
    }

    fn inject_error_state(&mut self, dx: [f32; LOOSE_ERROR_STATES]) {
        let dq = euler_to_quat(dx[6], dx[7], dx[8]);
        let q_old = [
            self.raw.nominal.q0,
            self.raw.nominal.q1,
            self.raw.nominal.q2,
            self.raw.nominal.q3,
        ];
        let q_new = quat_multiply(dq, q_old);
        self.raw.nominal.q0 = q_new[0];
        self.raw.nominal.q1 = q_new[1];
        self.raw.nominal.q2 = q_new[2];
        self.raw.nominal.q3 = q_new[3];
        self.normalize_nominal_quat();

        self.raw.pos_e64[0] += dx[0] as f64;
        self.raw.pos_e64[1] += dx[1] as f64;
        self.raw.pos_e64[2] += dx[2] as f64;
        self.sync_nominal_position_from_shadow();
        self.raw.nominal.vn += dx[3];
        self.raw.nominal.ve += dx[4];
        self.raw.nominal.vd += dx[5];
        self.raw.nominal.bax += dx[9];
        self.raw.nominal.bay += dx[10];
        self.raw.nominal.baz += dx[11];
        self.raw.nominal.bgx += dx[12];
        self.raw.nominal.bgy += dx[13];
        self.raw.nominal.bgz += dx[14];
        self.raw.nominal.sax += dx[15];
        self.raw.nominal.say += dx[16];
        self.raw.nominal.saz += dx[17];
        self.raw.nominal.sgx += dx[18];
        self.raw.nominal.sgy += dx[19];
        self.raw.nominal.sgz += dx[20];

        let dqcs = euler_to_quat(dx[21], dx[22], dx[23]);
        let dqcs0 = dqcs[0] as f64;
        let dqcs1 = dqcs[1] as f64;
        let dqcs2 = dqcs[2] as f64;
        let dqcs3 = dqcs[3] as f64;
        let qcs_old0 = self.raw.qcs64[0];
        let qcs_old1 = self.raw.qcs64[1];
        let qcs_old2 = self.raw.qcs64[2];
        let qcs_old3 = self.raw.qcs64[3];
        let mut qcs_new0 =
            dqcs0 * qcs_old0 - dqcs1 * qcs_old1 - dqcs2 * qcs_old2 - dqcs3 * qcs_old3;
        let mut qcs_new1 =
            dqcs0 * qcs_old1 + dqcs1 * qcs_old0 + dqcs2 * qcs_old3 - dqcs3 * qcs_old2;
        let mut qcs_new2 =
            dqcs0 * qcs_old2 - dqcs1 * qcs_old3 + dqcs2 * qcs_old0 + dqcs3 * qcs_old1;
        let mut qcs_new3 =
            dqcs0 * qcs_old3 + dqcs1 * qcs_old2 - dqcs2 * qcs_old1 + dqcs3 * qcs_old0;
        let qcs_norm =
            (qcs_new0 * qcs_new0 + qcs_new1 * qcs_new1 + qcs_new2 * qcs_new2 + qcs_new3 * qcs_new3)
                .sqrt();
        if qcs_norm > 0.0 {
            qcs_new0 /= qcs_norm;
            qcs_new1 /= qcs_norm;
            qcs_new2 /= qcs_norm;
            qcs_new3 /= qcs_norm;
        } else {
            qcs_new0 = 1.0;
            qcs_new1 = 0.0;
            qcs_new2 = 0.0;
            qcs_new3 = 0.0;
        }
        self.raw.qcs64 = [qcs_new0, qcs_new1, qcs_new2, qcs_new3];
        self.sync_nominal_mount_from_shadow();
    }

    fn normalize_nominal_quat(&mut self) {
        let mut q = [
            self.raw.nominal.q0,
            self.raw.nominal.q1,
            self.raw.nominal.q2,
            self.raw.nominal.q3,
        ];
        normalize_quat(&mut q);
        self.raw.nominal.q0 = q[0];
        self.raw.nominal.q1 = q[1];
        self.raw.nominal.q2 = q[2];
        self.raw.nominal.q3 = q[3];
    }

    fn sync_nominal_position_from_shadow(&mut self) {
        self.raw.nominal.pn = self.raw.pos_e64[0] as f32;
        self.raw.nominal.pe = self.raw.pos_e64[1] as f32;
        self.raw.nominal.pd = self.raw.pos_e64[2] as f32;
    }

    fn sync_nominal_mount_from_shadow(&mut self) {
        self.raw.nominal.qcs0 = self.raw.qcs64[0] as f32;
        self.raw.nominal.qcs1 = self.raw.qcs64[1] as f32;
        self.raw.nominal.qcs2 = self.raw.qcs64[2] as f32;
        self.raw.nominal.qcs3 = self.raw.qcs64[3] as f32;
    }
}

pub type LooseWrapper = LooseFilter;

fn predict_covariance_sparse(
    f: &[[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
    g: &[[f32; LOOSE_NOISE_STATES]; LOOSE_ERROR_STATES],
    p: &[[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
    q: &[f32; LOOSE_NOISE_STATES],
) -> [[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES] {
    let mut next_p = [[0.0; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES];
    for i in 0..LOOSE_ERROR_STATES {
        for j in i..LOOSE_ERROR_STATES {
            let mut accum = 0.0;
            for ia in 0..generated_loose::F_ROW_COUNTS[i] {
                let a = generated_loose::F_ROW_COLS[i][ia];
                let fia = f[i][a];
                for jb in 0..generated_loose::F_ROW_COUNTS[j] {
                    let b = generated_loose::F_ROW_COLS[j][jb];
                    accum += fia * p[a][b] * f[j][b];
                }
            }
            for ia in 0..generated_loose::G_ROW_COUNTS[i] {
                let a = generated_loose::G_ROW_COLS[i][ia];
                let gia = g[i][a];
                if q[a] == 0.0 {
                    continue;
                }
                for jb in 0..generated_loose::G_ROW_COUNTS[j] {
                    let b = generated_loose::G_ROW_COLS[j][jb];
                    if a == b {
                        accum += gia * q[a] * g[j][b];
                    }
                }
            }
            next_p[i][j] = accum;
            next_p[j][i] = accum;
        }
    }
    next_p
}

fn normalize_quat(q: &mut [f32; 4]) {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2 <= 1.0e-12 {
        *q = [1.0, 0.0, 0.0, 0.0];
        return;
    }
    let inv_n = 1.0 / libm::sqrtf(n2);
    for item in q.iter_mut() {
        *item *= inv_n;
    }
}

fn quat_multiply(p: [f32; 4], q: [f32; 4]) -> [f32; 4] {
    [
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    ]
}

fn euler_to_quat(roll: f32, pitch: f32, yaw: f32) -> [f32; 4] {
    let cr = libm::cosf(0.5 * roll);
    let sr = libm::sinf(0.5 * roll);
    let cp = libm::cosf(0.5 * pitch);
    let sp = libm::sinf(0.5 * pitch);
    let cy = libm::cosf(0.5 * yaw);
    let sy = libm::sinf(0.5 * yaw);
    [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]
}

fn quat_to_dcm(q_in: [f32; 4]) -> [[f32; 3]; 3] {
    let mut q = q_in;
    normalize_quat(&mut q);
    let q1_2 = q[1] * q[1];
    let q2_2 = q[2] * q[2];
    let q3_2 = q[3] * q[3];
    [
        [
            1.0 - 2.0 * (q2_2 + q3_2),
            2.0 * (q[1] * q[2] - q[0] * q[3]),
            2.0 * (q[1] * q[3] + q[0] * q[2]),
        ],
        [
            2.0 * (q[1] * q[2] + q[0] * q[3]),
            1.0 - 2.0 * (q1_2 + q3_2),
            2.0 * (q[2] * q[3] - q[0] * q[1]),
        ],
        [
            2.0 * (q[1] * q[3] - q[0] * q[2]),
            2.0 * (q[2] * q[3] + q[0] * q[1]),
            1.0 - 2.0 * (q1_2 + q2_2),
        ],
    ]
}

fn mat_vec3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn dcm_ecef_to_ned(lat_rad: f32, lon_rad: f32) -> [[f32; 3]; 3] {
    let sin_lat = libm::sinf(lat_rad);
    let cos_lat = libm::cosf(lat_rad);
    let sin_lon = libm::sinf(lon_rad);
    let cos_lon = libm::cosf(lon_rad);
    [
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [-sin_lon, cos_lon, 0.0],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
    ]
}

fn ecef_to_llh(x_e: [f32; 3]) -> (f32, f32, f32) {
    let a2 = WGS84_A * WGS84_A;
    let b2 = WGS84_B * WGS84_B;
    let z2 = x_e[2] * x_e[2];
    let r2 = x_e[0] * x_e[0] + x_e[1] * x_e[1];
    let r = libm::sqrtf(r2);
    let f = 54.0 * b2 * z2;
    let g = r2 + (1.0 - WGS84_E2) * z2 - WGS84_E2 * (a2 - b2);
    let c = WGS84_E2 * WGS84_E2 * f * r2 / (g * g * g);
    let s = libm::cbrtf(1.0 + c + libm::sqrtf(c * c + 2.0 * c));
    let p = f / (3.0 * (s + 1.0 / s + 1.0) * (s + 1.0 / s + 1.0) * g * g);
    let q = libm::sqrtf(1.0 + 2.0 * WGS84_E2 * WGS84_E2 * p);
    let r0 = -p * WGS84_E2 * r / (1.0 + q)
        + libm::sqrtf(
            0.5 * a2 * (1.0 + 1.0 / q) - p * (1.0 - WGS84_E2) * z2 / (q * (1.0 + q)) - 0.5 * p * r2,
        );
    let tmp = (r - WGS84_E2 * r0) * (r - WGS84_E2 * r0);
    let u = libm::sqrtf(tmp + z2);
    let v = libm::sqrtf(tmp + (1.0 - WGS84_E2) * z2);
    let inv_av = 1.0 / (WGS84_A * v);
    let z0 = b2 * x_e[2] * inv_av;
    let height_m = u * (1.0 - b2 * inv_av);
    let lat_rad = libm::atan2f(x_e[2] + (a2 / b2 - 1.0) * z0, r);
    let lon_rad = libm::atan2f(x_e[1], x_e[0]);
    (lat_rad, lon_rad, height_m)
}

fn gravity_ecef_j2(x_e: [f32; 3]) -> [f32; 3] {
    let r = libm::sqrtf(x_e[0] * x_e[0] + x_e[1] * x_e[1] + x_e[2] * x_e[2]);
    if r <= 0.0 {
        return [0.0; 3];
    }
    let r2 = r * r;
    let r3 = r * r2;
    let tmp1 = WGS84_GM / r3;
    let tmp2 = 1.5 * (WGS84_A * (WGS84_A * WGS84_J2)) / r2;
    let tmp3 = 5.0 * x_e[2] * x_e[2] / r2;
    [
        tmp1 * (-x_e[0] - tmp2 * (x_e[0] - tmp3 * x_e[0]))
            + WGS84_OMEGA_IE * WGS84_OMEGA_IE * x_e[0],
        tmp1 * (-x_e[1] - tmp2 * (x_e[1] - tmp3 * x_e[1]))
            + WGS84_OMEGA_IE * WGS84_OMEGA_IE * x_e[1],
        tmp1 * (-x_e[2] - tmp2 * (3.0 * x_e[2] - tmp3 * x_e[2])),
    ]
}

fn vec_norm3(v: [f32; 3]) -> f32 {
    libm::sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

fn test_chi2_scalar(
    residual: f32,
    p: &[[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
    h: &[f32; LOOSE_ERROR_STATES],
    r: f32,
    support: &[usize],
) -> bool {
    let mut s = r;
    for &i in support {
        for &j in support {
            s += h[i] * p[i][j] * h[j];
        }
    }
    libm::fabsf(residual) > 3.0 * libm::sqrtf(libm::fmaxf(s, 0.0))
}

fn test_chi2_vec3(
    residual: [f32; 3],
    p: &[[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
    h: &[[f32; LOOSE_ERROR_STATES]; 3],
    r_diag: [f32; 3],
    supports: &[&'static [usize]; 3],
) -> bool {
    for row in 0..3 {
        if test_chi2_scalar(residual[row], p, &h[row], r_diag[row], supports[row]) {
            return true;
        }
    }
    false
}

fn extract_support_from_row(
    h: &[f32; LOOSE_ERROR_STATES],
    support: &mut [usize; LOOSE_ERROR_STATES],
) -> usize {
    let mut len = 0;
    for (i, value) in h.iter().enumerate() {
        if *value != 0.0 {
            support[len] = i;
            len += 1;
        }
    }
    len
}

/// Splits a yaw-seeded NED vehicle attitude into ECEF seed attitude and identity residual mount.
pub fn loose_seeded_vehicle_ecef_split(
    yaw_rad: f32,
    lat_deg: f64,
    lon_deg: f64,
) -> ([f32; 4], [f32; 4]) {
    let half_yaw = 0.5 * yaw_rad as f64;
    let q_ns = [half_yaw.cos(), 0.0, 0.0, half_yaw.sin()];
    let q_es = quat_mul_f64(quat_conj_f64(quat_ecef_to_ned_f64(lat_deg, lon_deg)), q_ns);
    (
        [
            q_es[0] as f32,
            q_es[1] as f32,
            q_es[2] as f32,
            q_es[3] as f32,
        ],
        [1.0, 0.0, 0.0, 0.0],
    )
}

fn quat_ecef_to_ned_f64(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let c_en = [
        [-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()],
        [-lon.sin(), lon.cos(), 0.0],
        [-lat.cos() * lon.cos(), -lat.cos() * lon.sin(), -lat.sin()],
    ];
    dcm_to_quat_f64(c_en)
}

fn dcm_to_quat_f64(c: [[f64; 3]; 3]) -> [f64; 4] {
    let trace = c[0][0] + c[1][1] + c[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (c[2][1] - c[1][2]) / s,
            (c[0][2] - c[2][0]) / s,
            (c[1][0] - c[0][1]) / s,
        ]
    } else if c[0][0] > c[1][1] && c[0][0] > c[2][2] {
        let s = (1.0 + c[0][0] - c[1][1] - c[2][2]).sqrt() * 2.0;
        [
            (c[2][1] - c[1][2]) / s,
            0.25 * s,
            (c[0][1] + c[1][0]) / s,
            (c[0][2] + c[2][0]) / s,
        ]
    } else if c[1][1] > c[2][2] {
        let s = (1.0 + c[1][1] - c[0][0] - c[2][2]).sqrt() * 2.0;
        [
            (c[0][2] - c[2][0]) / s,
            (c[0][1] + c[1][0]) / s,
            0.25 * s,
            (c[1][2] + c[2][1]) / s,
        ]
    } else {
        let s = (1.0 + c[2][2] - c[0][0] - c[1][1]).sqrt() * 2.0;
        [
            (c[1][0] - c[0][1]) / s,
            (c[0][2] + c[2][0]) / s,
            (c[1][2] + c[2][1]) / s,
            0.25 * s,
        ]
    };
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn quat_conj_f64(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn quat_mul_f64(p: [f64; 4], q: [f64; 4]) -> [f64; 4] {
    [
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    ]
}
