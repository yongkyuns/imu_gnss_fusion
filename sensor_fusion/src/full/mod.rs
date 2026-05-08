//! Full INS/GNSS filter.
//!
//! This module implements a full-coupled ECEF filter used for
//! diagnostics and comparison against the runtime Reduced. It keeps a single
//! precision public state plus f64 shadow position and mount quaternion fields
//! used internally for numerically sensitive propagation.
//!
//! See `docs/full_formulation.pdf` for the PDF-first derivation. The nominal
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

use crate::covariance::{self, SparseCovariancePolicy};
#[doc(hidden)]
pub mod generated;
mod types;

use crate::ProcessNoise;
use crate::math::{
    cos_f64, euler_to_quat_f32, mat_vec3_f32, normalize_quat_f32, quat_conj_f64, quat_mul_f64,
    quat_multiply_f32, quat_to_dcm_f32, sin_f64, sq_f64, sqrt_f64, transpose3_f32, vec_norm3_f32,
};
use crate::nav::{
    WGS84_OMEGA_IE, dcm_ecef_to_ned_f32, ecef_to_llh_f32, gravity_ecef_j2_f32, quat_ecef_to_ned_f64,
};
pub(crate) use types::default_full_p_diag;
pub use types::{
    ERROR_STATES, GnssPositionGateDiag, ImuDelta, InitConfig, NOISE_STATES, NominalState, State,
};

const GPS_REF_SUPPORT_ROW0: &[usize] = &[0];
const GPS_REF_SUPPORT_ROW1: &[usize] = &[0, 1];
const GPS_REF_SUPPORT_ROW2: &[usize] = &[0, 1, 2];
const VEL_REF_SUPPORT_ROW0: &[usize] = &[3];
const VEL_REF_SUPPORT_ROW1: &[usize] = &[3, 4];
const VEL_REF_SUPPORT_ROW2: &[usize] = &[3, 4, 5];
const VEL_AXIS_SUPPORT_ROW0: &[usize] = &[3];
const VEL_AXIS_SUPPORT_ROW1: &[usize] = &[4];
const VEL_AXIS_SUPPORT_ROW2: &[usize] = &[5];
const MIN_NHC_UPDATE_SPEED_MPS: f32 = 0.05;
const MAX_NHC_GYRO_NORM_RADPS: f32 = 0.2;
const MAX_NHC_ACCEL_NORM_ERR_MPS2: f32 = 1.0;
const NHC_REFERENCE_MAX_DT_S: f32 = 1.0;
const DEFAULT_NHC_R_Y: f32 = 0.1_f32 * 0.1_f32;
const DEFAULT_NHC_R_Z: f32 = 0.05_f32 * 0.05_f32;

/// Full-coupled ECEF INS/GNSS filter used for reference comparisons.
#[derive(Clone, Debug)]
pub struct Filter {
    raw: State,
}

impl Filter {
    /// Creates a full filter with identity attitude/mount and the supplied process noise.
    pub fn new(noise: ProcessNoise) -> Self {
        let mut raw = State {
            noise,
            ..State::default()
        };
        raw.nominal.q0 = 1.0;
        raw.nominal.qcs0 = 1.0;
        raw.qcs64 = [1.0, 0.0, 0.0, 0.0];
        Self { raw }
    }

    /// Returns the full full-filter state.
    pub fn raw(&self) -> &State {
        &self.raw
    }

    /// Returns the nominal full-filter state.
    pub fn nominal(&self) -> &NominalState {
        &self.raw.nominal
    }

    /// Returns the public f32 covariance matrix.
    pub fn covariance(&self) -> &[[f32; ERROR_STATES]; ERROR_STATES] {
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
    pub fn last_dx(&self) -> &[f32; ERROR_STATES] {
        &self.raw.last_dx
    }

    /// Returns per-observation-row contributions from the last batch update.
    pub fn last_dx_by_obs(&self) -> &[[f32; ERROR_STATES]; 8] {
        &self.raw.last_dx_by_obs
    }

    /// Returns raw residuals from the last batch update.
    pub fn last_residuals(&self) -> &[f32] {
        let count = self
            .raw
            .last_obs_count
            .clamp(0, self.raw.last_residuals.len() as i32) as usize;
        &self.raw.last_residuals[..count]
    }

    /// Returns effective residuals after preceding rows in the last sequential batch update.
    pub fn last_effective_residuals(&self) -> &[f32] {
        let count = self
            .raw
            .last_obs_count
            .clamp(0, self.raw.last_effective_residuals.len() as i32) as usize;
        &self.raw.last_effective_residuals[..count]
    }

    /// Returns scalar innovation variances from the last batch update.
    pub fn last_innovation_vars(&self) -> &[f32] {
        let count = self
            .raw
            .last_obs_count
            .clamp(0, self.raw.last_innovation_vars.len() as i32) as usize;
        &self.raw.last_innovation_vars[..count]
    }

    /// Returns diagnostics for the most recent GNSS position gate attempt.
    pub fn last_gnss_position_gate(&self) -> GnssPositionGateDiag {
        self.raw.last_gnss_pos_gate
    }

    /// Replaces covariance.
    pub fn set_covariance(&mut self, p: [[f32; ERROR_STATES]; ERROR_STATES]) {
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
        let var = sq_f64((sigma_deg as f64).to_radians()) as f32;
        for i in 21..24 {
            for j in 0..ERROR_STATES {
                p[i][j] = 0.0;
                p[j][i] = 0.0;
            }
            p[i][i] = var;
        }
        self.set_covariance(p);
    }

    #[allow(clippy::too_many_arguments)]
    /// Initializes the full filter from a reference-frame state.
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
        p_diag: Option<[f32; ERROR_STATES]>,
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
            for i in 0..ERROR_STATES {
                for j in 0..ERROR_STATES {
                    self.raw.p[i][j] = 0.0;
                }
            }
            for (i, value) in p_diag.into_iter().enumerate() {
                self.raw.p[i][i] = value;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Initializes the full filter from an ECEF reference-frame state.
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
        p_diag: Option<[f32; ERROR_STATES]>,
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
        p_diag: Option<[f32; ERROR_STATES]>,
        residual_mount_sigma_deg: Option<f32>,
    ) {
        let (q_es, q_cs) = full_seeded_vehicle_ecef_split(yaw_rad, lat_deg, lon_deg);
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
    pub fn predict(&mut self, imu: ImuDelta) {
        if imu.dt <= 0.0 {
            return;
        }
        self.predict_nominal(imu);
        let (f, g) = self.compute_error_transition(imu);
        let dt = imu.dt;
        let mut q = [0.0; NOISE_STATES];
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

    /// Predicts only the full nominal state from one two-sample IMU delta.
    pub fn predict_nominal(&mut self, imu: ImuDelta) {
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
        let c_bv = quat_to_dcm_f32([
            self.raw.nominal.qcs0,
            self.raw.nominal.qcs1,
            self.raw.nominal.qcs2,
            self.raw.nominal.qcs3,
        ]);
        let c_vb = transpose3_f32(c_bv);
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
        let omega1_v = mat_vec3_f32(c_vb, omega1);
        let f1_v = mat_vec3_f32(c_vb, f1);
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
        let omega2_v = mat_vec3_f32(c_vb, omega2);
        let f2_v = mat_vec3_f32(c_vb, f2);

        let x_e_f = [x_e[0] as f32, x_e[1] as f32, x_e[2] as f32];
        let c_es = quat_to_dcm_f32(q_es);
        let g_e1 = gravity_ecef_j2_f32(x_e_f);
        let f1_e = mat_vec3_f32(c_es, f1_v);
        let vdot1 = [
            g_e1[0] + f1_e[0] + 2.0 * WGS84_OMEGA_IE * v_e[1],
            g_e1[1] + f1_e[1] - 2.0 * WGS84_OMEGA_IE * v_e[0],
            g_e1[2] + f1_e[2],
        ];
        let qdot1_raw = quat_multiply_f32(q_es, [0.0, omega1_v[0], omega1_v[1], omega1_v[2]]);
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
        normalize_quat_f32(&mut q_tmp);

        let c_es_tmp = quat_to_dcm_f32(q_tmp);
        let x_tmp_f = [x_tmp[0] as f32, x_tmp[1] as f32, x_tmp[2] as f32];
        let g_e2 = gravity_ecef_j2_f32(x_tmp_f);
        let f2_e = mat_vec3_f32(c_es_tmp, f2_v);
        let vdot2 = [
            g_e2[0] + f2_e[0] + 2.0 * WGS84_OMEGA_IE * v_tmp[1],
            g_e2[1] + f2_e[1] - 2.0 * WGS84_OMEGA_IE * v_tmp[0],
            g_e2[2] + f2_e[2],
        ];
        let qdot2_raw = quat_multiply_f32(q_tmp, [0.0, omega2_v[0], omega2_v[1], omega2_v[2]]);
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

    /// Computes generated full-filter transition and noise-input matrices.
    pub fn compute_error_transition(
        &self,
        imu: ImuDelta,
    ) -> (
        [[f32; ERROR_STATES]; ERROR_STATES],
        [[f32; NOISE_STATES]; ERROR_STATES],
    ) {
        if imu.dt <= 0.0 {
            return (
                [[0.0; ERROR_STATES]; ERROR_STATES],
                [[0.0; NOISE_STATES]; ERROR_STATES],
            );
        }
        generated::error_transition(&self.raw.nominal, imu)
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
        let mut h_rows = [[0.0; ERROR_STATES]; 8];
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
        let mut h_rows = [[0.0; ERROR_STATES]; 8];
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
            reference_speed_mps(vel_ecef_mps),
            DEFAULT_NHC_R_Y,
            DEFAULT_NHC_R_Z,
            gyro_radps,
            accel_mps2,
            dt_s,
        );
    }

    #[allow(clippy::too_many_arguments)]
    /// Fuses a Full batch with optional per-axis velocity standard deviations.
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
        self.fuse_reference_batch_full_with_nhc_speed(
            pos_ecef_m,
            vel_ecef_mps,
            h_acc_m,
            vel_std_ned_mps,
            dt_since_last_gnss_s,
            reference_speed_mps(vel_ecef_mps),
            gyro_radps,
            accel_mps2,
            dt_s,
        );
    }

    /// Fuses a Full batch with optional per-axis velocity standard deviations and
    /// caller-provided nonholonomic lateral/vertical observation variances.
    #[allow(clippy::too_many_arguments)]
    pub fn fuse_reference_batch_full_with_nhc_speed_and_r(
        &mut self,
        pos_ecef_m: Option<[f64; 3]>,
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        vel_std_ned_mps: Option<[f32; 3]>,
        dt_since_last_gnss_s: f32,
        nhc_gate_speed_mps: Option<f32>,
        r_nhc_y: f32,
        r_nhc_z: f32,
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
            nhc_gate_speed_mps,
            r_nhc_y,
            r_nhc_z,
            gyro_radps,
            accel_mps2,
            dt_s,
        );
    }

    pub fn fuse_reference_batch_full_with_nhc_speed(
        &mut self,
        pos_ecef_m: Option<[f64; 3]>,
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        vel_std_ned_mps: Option<[f32; 3]>,
        dt_since_last_gnss_s: f32,
        nhc_gate_speed_mps: Option<f32>,
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
            nhc_gate_speed_mps,
            DEFAULT_NHC_R_Y,
            DEFAULT_NHC_R_Z,
            gyro_radps,
            accel_mps2,
            dt_s,
        );
    }

    /// Fuses full-filter nonholonomic lateral and vertical velocity constraints.
    pub fn fuse_nhc_reference(&mut self, gyro_radps: [f32; 3], accel_mps2: [f32; 3], dt_s: f32) {
        self.fuse_nhc_reference_with_speed(None, gyro_radps, accel_mps2, dt_s);
    }

    /// Fuses full-filter nonholonomic constraints using an external speed gate.
    pub fn fuse_nhc_reference_with_speed(
        &mut self,
        nhc_gate_speed_mps: Option<f32>,
        gyro_radps: [f32; 3],
        accel_mps2: [f32; 3],
        dt_s: f32,
    ) {
        self.fuse_nhc_reference_with_speed_and_r(
            nhc_gate_speed_mps,
            DEFAULT_NHC_R_Y,
            DEFAULT_NHC_R_Z,
            gyro_radps,
            accel_mps2,
            dt_s,
        );
    }

    /// Fuses full-filter nonholonomic constraints using caller-provided lateral/vertical
    /// observation variances.
    #[allow(clippy::too_many_arguments)]
    pub fn fuse_nhc_reference_with_speed_and_r(
        &mut self,
        nhc_gate_speed_mps: Option<f32>,
        r_nhc_y: f32,
        r_nhc_z: f32,
        gyro_radps: [f32; 3],
        accel_mps2: [f32; 3],
        dt_s: f32,
    ) {
        if dt_s <= 0.0
            || !self.nhc_gate_allows(gyro_radps, accel_mps2)
            || !nhc_speed_gate_allows(nhc_gate_speed_mps)
        {
            return;
        }
        let (vc_y_est, h_y) = generated::nhc_y(&self.raw.nominal);
        let (vc_z_est, h_z) = generated::nhc_z(&self.raw.nominal);
        let dt_obs = nhc_observation_dt(dt_s);
        let var_y = sanitize_nhc_variance(r_nhc_y) / dt_obs;
        let var_z = sanitize_nhc_variance(r_nhc_z) / dt_obs;
        let mut h_rows = [[0.0; ERROR_STATES]; 8];
        let mut h_supports: [Option<&'static [usize]>; 8] = [None; 8];
        let mut residuals = [0.0; 8];
        let mut variances = [0.0; 8];
        let mut obs_count = 0;
        if var_y > 0.0 {
            h_rows[obs_count] = h_y;
            h_supports[obs_count] = Some(&generated::NHC_Y_SUPPORT);
            residuals[obs_count] = -vc_y_est;
            variances[obs_count] = var_y;
            obs_count += 1;
        }
        if var_z > 0.0 {
            h_rows[obs_count] = h_z;
            h_supports[obs_count] = Some(&generated::NHC_Z_SUPPORT);
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
        nhc_gate_speed_mps: Option<f32>,
        r_nhc_y: f32,
        r_nhc_z: f32,
        gyro_radps: [f32; 3],
        accel_mps2: [f32; 3],
        dt_s: f32,
    ) {
        if dt_s <= 0.0 {
            return;
        }
        self.raw.last_obs_count = 0;
        self.raw.last_obs_types = [0; 8];
        self.raw.last_dx_by_obs = [[0.0; ERROR_STATES]; 8];
        self.raw.last_residuals = [0.0; 8];
        self.raw.last_effective_residuals = [0.0; 8];
        self.raw.last_innovation_vars = [0.0; 8];
        self.raw.last_gnss_pos_gate = GnssPositionGateDiag::default();

        let mut h_rows = [[0.0; ERROR_STATES]; 8];
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

        if self.nhc_gate_allows(gyro_radps, accel_mps2) && nhc_speed_gate_allows(nhc_gate_speed_mps)
        {
            let (vc_y_est, h_y) = generated::nhc_y(&self.raw.nominal);
            let (vc_z_est, h_z) = generated::nhc_z(&self.raw.nominal);
            let dt_obs = nhc_observation_dt(dt_s);
            let var_y = sanitize_nhc_variance(r_nhc_y) / dt_obs;
            let var_z = sanitize_nhc_variance(r_nhc_z) / dt_obs;
            if var_y > 0.0 {
                h_rows[obs_count] = h_y;
                h_supports[obs_count] = Some(&generated::NHC_Y_SUPPORT);
                residuals[obs_count] = -vc_y_est;
                variances[obs_count] = var_y;
                obs_types[obs_count] = 7;
                obs_count += 1;
            }
            if var_z > 0.0 {
                h_rows[obs_count] = h_z;
                h_supports[obs_count] = Some(&generated::NHC_Z_SUPPORT);
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
        &mut self,
        pos_ecef_m: Option<[f64; 3]>,
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        speed_acc_mps: f32,
        vel_std_ned_mps: Option<[f32; 3]>,
        dt_since_last_gnss_s: f32,
        h_rows: &mut [[f32; ERROR_STATES]; 8],
        h_supports: &mut [Option<&'static [usize]>; 8],
        residuals: &mut [f32; 8],
        variances: &mut [f32; 8],
        mut obs_types: Option<&mut [i32; 8]>,
        mut obs_count: usize,
    ) -> usize {
        if let Some(pos_ecef_m) = pos_ecef_m.filter(|_| h_acc_m > 0.0) {
            let (lat_rad, lon_rad, _) = ecef_to_llh_f32([
                self.raw.pos_e64[0] as f32,
                self.raw.pos_e64[1] as f32,
                self.raw.pos_e64[2] as f32,
            ]);
            let c_en = dcm_ecef_to_ned_f32(lat_rad, lon_rad);
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
            let mut h_tmp = [[0.0; ERROR_STATES]; 3];
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
            let gate = gnss_position_gate_diag(
                residual,
                &self.raw.p,
                &h_tmp,
                [1.0, 1.0, 1.0],
                &gps_supports,
            );
            self.raw.last_gnss_pos_gate = gate;
            if gate.accepted {
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
            let mut vel_rows = [[0.0; ERROR_STATES]; 3];
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
                let (lat_meas, lon_meas, _) = ecef_to_llh_f32([
                    pos_for_llh[0] as f32,
                    pos_for_llh[1] as f32,
                    pos_for_llh[2] as f32,
                ]);
                let c_en_meas = dcm_ecef_to_ned_f32(lat_meas, lon_meas);
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
        vec_norm3_f32(omega_is) < MAX_NHC_GYRO_NORM_RADPS
            && libm::fabsf(vec_norm3_f32(f_s) - 9.81) < MAX_NHC_ACCEL_NORM_ERR_MPS2
    }

    fn batch_update_joseph(
        &mut self,
        obs_count: usize,
        h: &[[f32; ERROR_STATES]; 8],
        h_supports: &[Option<&'static [usize]>; 8],
        residuals: &[f32; 8],
        variances: &[f32; 8],
    ) {
        let mut dx = [0.0; ERROR_STATES];
        let mut dx_by_obs = [[0.0; ERROR_STATES]; 8];
        let mut residual_diag = [0.0; 8];
        let mut effective_residual_diag = [0.0; 8];
        let mut innovation_var_diag = [0.0; 8];
        {
            let p = &mut self.raw.p;
            let mut dense_support = [0usize; ERROR_STATES];
            for obs in 0..obs_count {
                let h_obs = &h[obs];
                let support = match h_supports[obs] {
                    Some(support) => support,
                    None => {
                        let len = extract_support_from_row(h_obs, &mut dense_support);
                        &dense_support[..len]
                    }
                };

                let mut ph = [0.0; ERROR_STATES];
                let mut s = variances[obs];
                for i in 0..ERROR_STATES {
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
                let effective_residual = residuals[obs] - hd;
                residual_diag[obs] = residuals[obs];
                effective_residual_diag[obs] = effective_residual;
                innovation_var_diag[obs] = s;
                let alpha = effective_residual / s;
                for i in 0..ERROR_STATES {
                    let row_dx = ph[i] * alpha;
                    dx_by_obs[obs][i] = row_dx;
                    dx[i] += row_dx;
                }
                for i in 0..ERROR_STATES {
                    for j in i..ERROR_STATES {
                        let updated = p[i][j] - (ph[i] * ph[j]) / s;
                        p[i][j] = updated;
                        p[j][i] = updated;
                    }
                }
            }
        }

        for i in 0..ERROR_STATES {
            self.raw.last_dx[i] = dx[i];
        }
        self.raw.last_dx_by_obs = dx_by_obs;
        self.raw.last_residuals = residual_diag;
        self.raw.last_effective_residuals = effective_residual_diag;
        self.raw.last_innovation_vars = innovation_var_diag;
        self.inject_error_state(dx);
        apply_reset(&mut self.raw.p, &dx);
    }

    fn inject_error_state(&mut self, dx: [f32; ERROR_STATES]) {
        let dq = euler_to_quat_f32(dx[6], dx[7], dx[8]);
        let q_old = [
            self.raw.nominal.q0,
            self.raw.nominal.q1,
            self.raw.nominal.q2,
            self.raw.nominal.q3,
        ];
        let q_new = quat_multiply_f32(dq, q_old);
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

        let dqcs = euler_to_quat_f32(dx[21], dx[22], dx[23]);
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
        let qcs_norm = sqrt_f64(
            qcs_new0 * qcs_new0 + qcs_new1 * qcs_new1 + qcs_new2 * qcs_new2 + qcs_new3 * qcs_new3,
        );
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
        normalize_quat_f32(&mut q);
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

pub type Wrapper = Filter;

fn predict_covariance_sparse(
    f: &[[f32; ERROR_STATES]; ERROR_STATES],
    g: &[[f32; NOISE_STATES]; ERROR_STATES],
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    q: &[f32; NOISE_STATES],
) -> [[f32; ERROR_STATES]; ERROR_STATES] {
    covariance::predict_sparse(
        f,
        g,
        p,
        q,
        &generated::F_ROW_COUNTS,
        &generated::F_ROW_COLS,
        &generated::G_ROW_COUNTS,
        &generated::G_ROW_COLS,
        SparseCovariancePolicy::FULL,
    )
}

fn apply_reset(p: &mut [[f32; ERROR_STATES]; ERROR_STATES], dx: &[f32; ERROR_STATES]) {
    apply_reset_block(p, 6, [dx[6], dx[7], dx[8]]);
    apply_reset_block(p, 21, [dx[21], dx[22], dx[23]]);
    covariance::symmetrize(p);
}

#[allow(clippy::needless_range_loop)]
fn apply_reset_block(p: &mut [[f32; ERROR_STATES]; ERROR_STATES], offset: usize, dtheta: [f32; 3]) {
    let g_reset_theta = generated::reset_jacobian(dtheta);
    let mut p_aa = [[0.0; 3]; 3];
    let mut p_ab = [[0.0; ERROR_STATES - 3]; 3];
    let mut next_aa = [[0.0; 3]; 3];

    for i in 0..3 {
        for j in 0..3 {
            p_aa[i][j] = p[offset + i][offset + j];
        }
        for j in 0..ERROR_STATES {
            if j >= offset && j < offset + 3 {
                continue;
            }
            p_ab[i][if j < offset { j } else { j - 3 }] = p[offset + i][j];
        }
    }

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                next_aa[i][j] += g_reset_theta[i][k] * p_aa[k][j];
            }
        }
    }

    for i in 0..3 {
        for j in 0..3 {
            let mut accum = 0.0;
            for k in 0..3 {
                accum += next_aa[i][k] * g_reset_theta[j][k];
            }
            p[offset + i][offset + j] = accum;
        }
        for j in 0..ERROR_STATES {
            if j >= offset && j < offset + 3 {
                continue;
            }
            let mut accum = 0.0;
            for k in 0..3 {
                accum += g_reset_theta[i][k] * p_ab[k][if j < offset { j } else { j - 3 }];
            }
            p[offset + i][j] = accum;
            p[j][offset + i] = accum;
        }
    }
}

fn nhc_observation_dt(dt_s: f32) -> f32 {
    if dt_s > 0.0 && dt_s.is_finite() {
        dt_s.min(NHC_REFERENCE_MAX_DT_S)
    } else {
        NHC_REFERENCE_MAX_DT_S
    }
}

fn sanitize_nhc_variance(r: f32) -> f32 {
    if r > 0.0 && r.is_finite() { r } else { 0.0 }
}

fn reference_speed_mps(vel_ecef_mps: Option<[f32; 3]>) -> Option<f32> {
    vel_ecef_mps.map(vec_norm3_f32)
}

fn nhc_speed_gate_allows(external_speed_mps: Option<f32>) -> bool {
    let Some(speed_mps) = external_speed_mps else {
        return false;
    };
    speed_mps.is_finite() && speed_mps >= MIN_NHC_UPDATE_SPEED_MPS
}

fn test_chi2_scalar(
    residual: f32,
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    h: &[f32; ERROR_STATES],
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

fn gnss_position_gate_diag(
    residual: [f32; 3],
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    h: &[[f32; ERROR_STATES]; 3],
    r_diag: [f32; 3],
    supports: &[&'static [usize]; 3],
) -> GnssPositionGateDiag {
    let mut innovation_sigma = [0.0; 3];
    let mut normalized_residual = [0.0; 3];
    let mut accepted = true;
    for row in 0..3 {
        let mut s = r_diag[row];
        for &i in supports[row] {
            for &j in supports[row] {
                s += h[row][i] * p[i][j] * h[row][j];
            }
        }
        let sigma = libm::sqrtf(libm::fmaxf(s, 0.0));
        innovation_sigma[row] = sigma;
        normalized_residual[row] = if sigma > 0.0 {
            residual[row] / sigma
        } else {
            f32::INFINITY.copysign(residual[row])
        };
        if libm::fabsf(normalized_residual[row]) > 3.0 {
            accepted = false;
        }
    }
    GnssPositionGateDiag {
        attempted: true,
        accepted,
        whitened_residual: residual,
        innovation_sigma,
        normalized_residual,
    }
}

fn test_chi2_vec3(
    residual: [f32; 3],
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    h: &[[f32; ERROR_STATES]; 3],
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

fn extract_support_from_row(h: &[f32; ERROR_STATES], support: &mut [usize; ERROR_STATES]) -> usize {
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
pub fn full_seeded_vehicle_ecef_split(
    yaw_rad: f32,
    lat_deg: f64,
    lon_deg: f64,
) -> ([f32; 4], [f32; 4]) {
    let half_yaw = 0.5 * yaw_rad as f64;
    let q_ns = [cos_f64(half_yaw), 0.0, 0.0, sin_f64(half_yaw)];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_jacobian_matches_first_order_quaternion_reset() {
        let dtheta = [0.2, -0.1, 0.05];
        let reset = generated::reset_jacobian(dtheta);
        let expected = [
            [1.0, 0.5 * dtheta[2], -0.5 * dtheta[1]],
            [-0.5 * dtheta[2], 1.0, 0.5 * dtheta[0]],
            [0.5 * dtheta[1], -0.5 * dtheta[0], 1.0],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!((reset[i][j] - expected[i][j]).abs() < 1.0e-7);
            }
        }
    }

    #[test]
    fn apply_reset_preserves_covariance_symmetry() {
        let mut p = [[0.0_f32; ERROR_STATES]; ERROR_STATES];
        for i in 0..ERROR_STATES {
            for j in i..ERROR_STATES {
                let value = 1.0e-4 * ((i + 1) as f32) + 1.0e-6 * ((j + 1) as f32);
                p[i][j] = value;
                p[j][i] = value;
            }
            p[i][i] += 1.0;
        }
        let dx = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -0.015, 0.005, 0.01,
        ];
        apply_reset(&mut p, &dx);
        for i in 0..ERROR_STATES {
            for j in 0..ERROR_STATES {
                assert!(p[i][j].is_finite());
                assert!((p[i][j] - p[j][i]).abs() < 1.0e-6);
            }
        }
    }
}
