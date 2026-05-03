//! Runtime ESKF implementation backed by generated symbolic model functions.
//!
//! `RustEskf` owns the nominal state, covariance, update diagnostics, and
//! residual mount-state controls. The generated model wrapper supplies
//! prediction Jacobians and scalar observation rows; this module applies
//! covariance propagation, measurement fusion, reset injection, and diagnostic
//! accounting.
//!
//! See `docs/eskf_mount_formulation.pdf` for the maintained derivation. The
//! nominal state is:
//!
//! ```text
//! q_ns, v_n, p_n, b_g, b_a, q_cs
//! ```
//!
//! where `q_ns` rotates the seeded IMU frame into local NED and `q_cs` rotates
//! seeded-frame velocity into the corrected vehicle frame. The 18-state error
//! vector is:
//!
//! ```text
//! dtheta_s, dv_n, dp_n, dbg, dba, dpsi_cs
//! ```
//!
//! Propagation uses generated `F` and `G` matrices with
//! `P+ = F P F^T + G Q G^T`. GNSS and zero-velocity rows observe NED position
//! or velocity directly. Vehicle speed and NHC rows use
//! `v_c = C_cs C_ns^T v_n`, which gives the residual-mount Jacobian block
//! `-e_i^T [v_c]x`.

use libm::{fabsf, sqrtf};

use crate::ekf::PredictNoise;
use crate::eskf_types::{
    ESKF_UPDATE_DIAG_TYPES, EskfGnssSample, EskfImuDelta, EskfNominalState, EskfState,
    EskfStationaryDiag, EskfUpdateDiag,
};
use crate::generated_eskf::{self, ERROR_STATES, NOISE_STATES};

const RUNTIME_ZERO_VEL_R_DIAG: f32 = 0.01;

const DIAG_GPS_POS: usize = 0;
const DIAG_GPS_VEL: usize = 1;
const DIAG_ZERO_VEL: usize = 2;
const DIAG_BODY_SPEED_X: usize = 3;
const DIAG_BODY_VEL_Y: usize = 4;
const DIAG_BODY_VEL_Z: usize = 5;
const DIAG_STATIONARY_X: usize = 6;
const DIAG_STATIONARY_Y: usize = 7;
const DIAG_GPS_POS_D: usize = 8;
const DIAG_GPS_VEL_D: usize = 9;
const DIAG_ZERO_VEL_D: usize = 10;
const BODY_VEL_Y_SUPPORT: [usize; 8] = [0, 1, 2, 3, 4, 5, 15, 17];
const BODY_VEL_Z_SUPPORT: [usize; 8] = [0, 1, 2, 3, 4, 5, 15, 16];
const MAX_BATCH_OBS: usize = 8;

/// Rust ESKF state machine with covariance and residual mount-control policy.
#[derive(Debug, Clone)]
pub struct RustEskf {
    raw: EskfState,
    freeze_misalignment_states: bool,
}

impl RustEskf {
    /// Creates a filter with identity attitude/mount and diagonal covariance.
    pub fn new(noise: PredictNoise) -> Self {
        let mut raw = EskfState {
            nominal: EskfNominalState {
                q0: 1.0,
                qcs0: 1.0,
                ..EskfNominalState::default()
            },
            p: [[0.0; ERROR_STATES]; ERROR_STATES],
            noise,
            stationary_diag: EskfStationaryDiag::default(),
            update_diag: EskfUpdateDiag::default(),
        };
        for i in 0..ERROR_STATES {
            raw.p[i][i] = 1.0;
        }
        Self {
            raw,
            freeze_misalignment_states: false,
        }
    }

    /// Returns the full raw ESKF state.
    pub fn raw(&self) -> &EskfState {
        &self.raw
    }

    /// Returns mutable access to the full raw ESKF state for integration code and diagnostics.
    pub fn raw_mut(&mut self) -> &mut EskfState {
        &mut self.raw
    }

    /// Returns the nominal navigation state.
    pub fn nominal(&self) -> &EskfNominalState {
        &self.raw.nominal
    }

    /// Returns the error-state covariance matrix.
    pub fn covariance(&self) -> &[[f32; ERROR_STATES]; ERROR_STATES] {
        &self.raw.p
    }

    /// Enables or disables residual-mount state freezing.
    pub fn set_freeze_misalignment_states(&mut self, freeze: bool) {
        self.freeze_misalignment_states = freeze;
        if freeze {
            freeze_mount_covariance(&mut self.raw.p);
        }
    }

    /// Reports whether residual-mount states are currently frozen.
    pub fn freeze_misalignment_states(&self) -> bool {
        self.freeze_misalignment_states
    }

    /// Initializes nominal attitude, velocity, and position from a GNSS sample.
    pub fn init_nominal_from_gnss(&mut self, q_bn: [f32; 4], gnss: EskfGnssSample) {
        const DEFAULT_GYRO_BIAS_SIGMA_DPS: f32 = 0.125;
        const DEFAULT_ACCEL_BIAS_SIGMA_MPS2: f32 = 0.15;

        self.raw.nominal.q0 = q_bn[0];
        self.raw.nominal.q1 = q_bn[1];
        self.raw.nominal.q2 = q_bn[2];
        self.raw.nominal.q3 = q_bn[3];
        self.raw.nominal.vn = gnss.vel_ned_mps[0];
        self.raw.nominal.ve = gnss.vel_ned_mps[1];
        self.raw.nominal.vd = gnss.vel_ned_mps[2];
        self.raw.nominal.pn = gnss.pos_ned_m[0];
        self.raw.nominal.pe = gnss.pos_ned_m[1];
        self.raw.nominal.pd = gnss.pos_ned_m[2];
        self.raw.nominal.qcs0 = 1.0;
        self.raw.nominal.qcs1 = 0.0;
        self.raw.nominal.qcs2 = 0.0;
        self.raw.nominal.qcs3 = 0.0;

        self.raw.p = [[0.0; ERROR_STATES]; ERROR_STATES];
        let att_sigma_rad = 2.0 * core::f32::consts::PI / 180.0;
        let att_var = att_sigma_rad * att_sigma_rad;
        self.raw.p[0][0] = att_var;
        self.raw.p[1][1] = att_var;
        self.raw.p[2][2] = att_var;

        let vel_std = gnss.vel_std_mps[0]
            .max(gnss.vel_std_mps[1])
            .max(gnss.vel_std_mps[2])
            .max(0.2);
        let vel_var = vel_std * vel_std;
        self.raw.p[3][3] = vel_var;
        self.raw.p[4][4] = vel_var;
        self.raw.p[5][5] = vel_var;

        let pos_n = gnss.pos_std_m[0].max(0.5);
        let pos_e = gnss.pos_std_m[1].max(0.5);
        let pos_d = gnss.pos_std_m[2].max(0.5);
        self.raw.p[6][6] = pos_n * pos_n;
        self.raw.p[7][7] = pos_e * pos_e;
        self.raw.p[8][8] = pos_d * pos_d;

        let gyro_bias_sigma_radps = DEFAULT_GYRO_BIAS_SIGMA_DPS * core::f32::consts::PI / 180.0;
        let accel_bias_sigma_mps2 = DEFAULT_ACCEL_BIAS_SIGMA_MPS2;
        self.raw.p[9][9] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
        self.raw.p[10][10] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
        self.raw.p[11][11] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
        self.raw.p[12][12] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
        self.raw.p[13][13] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
        self.raw.p[14][14] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
        if self.freeze_misalignment_states {
            freeze_mount_covariance(&mut self.raw.p);
        } else {
            let mount_residual_sigma_rad = 10.0 * core::f32::consts::PI / 180.0;
            self.raw.p[15][15] = mount_residual_sigma_rad * mount_residual_sigma_rad;
            self.raw.p[16][16] = mount_residual_sigma_rad * mount_residual_sigma_rad;
            self.raw.p[17][17] = mount_residual_sigma_rad * mount_residual_sigma_rad;
        }
    }

    /// Predicts nominal state and covariance from one IMU delta.
    pub fn predict(&mut self, imu: EskfImuDelta) {
        let (f, g) = self.compute_error_transition_with_noise(imu);
        generated_eskf::predict_nominal(&mut self.raw.nominal, imu);
        normalize_nominal_quat(&mut self.raw.nominal);

        let dt = imu.dt;
        let mut q = [0.0; NOISE_STATES];
        // `gyro_var` and `accel_var` are continuous white-noise densities.
        // The generated G matrix maps those noises directly into attitude and
        // velocity error, so the discrete covariance is density * dt.
        q[0] = self.raw.noise.gyro_var * dt;
        q[1] = q[0];
        q[2] = q[0];
        q[3] = self.raw.noise.accel_var * dt;
        q[4] = q[3];
        q[5] = q[3];
        q[6] = self.raw.noise.gyro_bias_rw_var * dt;
        q[7] = q[6];
        q[8] = q[6];
        q[9] = self.raw.noise.accel_bias_rw_var * dt;
        q[10] = q[9];
        q[11] = q[9];
        q[12] = if self.freeze_misalignment_states {
            0.0
        } else {
            self.raw.noise.mount_align_rw_var * dt
        };
        q[13] = q[12];
        q[14] = q[12];

        self.raw.p = predict_covariance_sparse(&f, &g, &self.raw.p, &q);
        symmetrize_p(&mut self.raw.p);
        if self.freeze_misalignment_states {
            freeze_mount_covariance(&mut self.raw.p);
        }
    }

    /// Computes the generated error-state transition matrix for an IMU delta.
    pub fn compute_error_transition(
        &self,
        imu: EskfImuDelta,
    ) -> [[f32; ERROR_STATES]; ERROR_STATES] {
        self.compute_error_transition_with_noise(imu).0
    }

    /// Computes generated error-state transition and noise-input matrices.
    pub fn compute_error_transition_with_noise(
        &self,
        imu: EskfImuDelta,
    ) -> (
        [[f32; ERROR_STATES]; ERROR_STATES],
        [[f32; NOISE_STATES]; ERROR_STATES],
    ) {
        generated_eskf::error_transition(&self.raw.nominal, imu)
    }

    /// Fuses GNSS position and velocity.
    pub fn fuse_gps(&mut self, sample: EskfGnssSample) {
        self.fuse_gps_pos_n(
            sample.pos_ned_m[0],
            sample.pos_std_m[0] * sample.pos_std_m[0],
        );
        self.fuse_gps_pos_e(
            sample.pos_ned_m[1],
            sample.pos_std_m[1] * sample.pos_std_m[1],
        );
        self.fuse_gps_pos_d(
            sample.pos_ned_m[2],
            sample.pos_std_m[2] * sample.pos_std_m[2],
        );
        self.fuse_gps_vel_n(
            sample.vel_ned_mps[0],
            sample.vel_std_mps[0] * sample.vel_std_mps[0],
        );
        self.fuse_gps_vel_e(
            sample.vel_ned_mps[1],
            sample.vel_std_mps[1] * sample.vel_std_mps[1],
        );
        self.fuse_gps_vel_d(
            sample.vel_ned_mps[2],
            sample.vel_std_mps[2] * sample.vel_std_mps[2],
        );
    }

    /// Fuses GNSS position/velocity and optional lateral/vertical NHC rows as one batch.
    ///
    /// This mirrors the loose filter's coupled reference update: all rows are
    /// linearized at the same nominal state, one error state is solved, and the
    /// nominal is injected once. Running this at GNSS rate keeps the cost bounded
    /// while preserving the important GNSS/NHC covariance allocation.
    pub fn fuse_gps_nhc_batch(
        &mut self,
        sample: EskfGnssSample,
        r_body_vel_y: Option<f32>,
        r_body_vel_z: Option<f32>,
    ) {
        let mut h_rows = [[0.0; ERROR_STATES]; MAX_BATCH_OBS];
        let mut residuals = [0.0; MAX_BATCH_OBS];
        let mut variances = [0.0; MAX_BATCH_OBS];
        let mut diag_types = [0usize; MAX_BATCH_OBS];
        let mut obs_count = 0usize;

        push_batch_row(
            &mut h_rows,
            &mut residuals,
            &mut variances,
            &mut diag_types,
            &mut obs_count,
            gps_axis_h(6),
            sample.pos_ned_m[0] - self.raw.nominal.pn,
            sample.pos_std_m[0] * sample.pos_std_m[0],
            DIAG_GPS_POS,
        );
        push_batch_row(
            &mut h_rows,
            &mut residuals,
            &mut variances,
            &mut diag_types,
            &mut obs_count,
            gps_axis_h(7),
            sample.pos_ned_m[1] - self.raw.nominal.pe,
            sample.pos_std_m[1] * sample.pos_std_m[1],
            DIAG_GPS_POS,
        );
        push_batch_row(
            &mut h_rows,
            &mut residuals,
            &mut variances,
            &mut diag_types,
            &mut obs_count,
            gps_axis_h(8),
            sample.pos_ned_m[2] - self.raw.nominal.pd,
            sample.pos_std_m[2] * sample.pos_std_m[2],
            DIAG_GPS_POS_D,
        );
        push_batch_row(
            &mut h_rows,
            &mut residuals,
            &mut variances,
            &mut diag_types,
            &mut obs_count,
            gps_axis_h(3),
            sample.vel_ned_mps[0] - self.raw.nominal.vn,
            sample.vel_std_mps[0] * sample.vel_std_mps[0],
            DIAG_GPS_VEL,
        );
        push_batch_row(
            &mut h_rows,
            &mut residuals,
            &mut variances,
            &mut diag_types,
            &mut obs_count,
            gps_axis_h(4),
            sample.vel_ned_mps[1] - self.raw.nominal.ve,
            sample.vel_std_mps[1] * sample.vel_std_mps[1],
            DIAG_GPS_VEL,
        );
        push_batch_row(
            &mut h_rows,
            &mut residuals,
            &mut variances,
            &mut diag_types,
            &mut obs_count,
            gps_axis_h(5),
            sample.vel_ned_mps[2] - self.raw.nominal.vd,
            sample.vel_std_mps[2] * sample.vel_std_mps[2],
            DIAG_GPS_VEL_D,
        );

        let v_vehicle = nominal_vehicle_velocity(&self.raw.nominal);
        if let Some(r) = r_body_vel_y.filter(|r| *r > 0.0 && r.is_finite()) {
            let obs = generated_eskf::body_vel_y_observation(&self.raw.nominal, &self.raw.p, r);
            push_batch_row(
                &mut h_rows,
                &mut residuals,
                &mut variances,
                &mut diag_types,
                &mut obs_count,
                obs.h,
                -v_vehicle[1],
                r,
                DIAG_BODY_VEL_Y,
            );
        }
        if let Some(r) = r_body_vel_z.filter(|r| *r > 0.0 && r.is_finite()) {
            let obs = generated_eskf::body_vel_z_observation(&self.raw.nominal, &self.raw.p, r);
            push_batch_row(
                &mut h_rows,
                &mut residuals,
                &mut variances,
                &mut diag_types,
                &mut obs_count,
                obs.h,
                -v_vehicle[2],
                r,
                DIAG_BODY_VEL_Z,
            );
        }

        self.fuse_batch(obs_count, &h_rows, &residuals, &variances, &diag_types);
    }

    /// Applies a zero-velocity pseudo-measurement on all NED velocity axes.
    pub fn fuse_zero_vel(&mut self, r_zero_vel: f32) {
        self.fuse_gps_vel_n_impl(0.0, r_zero_vel, true);
        self.fuse_gps_vel_e_impl(0.0, r_zero_vel, true);
        self.fuse_gps_vel_d_impl(0.0, r_zero_vel, true);
    }

    /// Applies stationary gravity pseudo-measurements from body-frame acceleration.
    pub fn fuse_stationary_gravity(&mut self, accel_body_mps2: [f32; 3], r_stationary_accel: f32) {
        self.fuse_stationary_gravity_x(accel_body_mps2[0], r_stationary_accel);
        self.fuse_stationary_gravity_y(accel_body_mps2[1], r_stationary_accel);
    }

    /// Fuses forward vehicle speed as a body-frame X velocity observation.
    pub fn fuse_body_speed_x(&mut self, speed_mps: f32, r_speed: f32) {
        let obs = generated_eskf::body_vel_x_observation(&self.raw.nominal, &self.raw.p, r_speed);
        let v_vehicle = nominal_vehicle_velocity(&self.raw.nominal);
        let innovation = speed_mps - v_vehicle[0];
        let mut k = obs.k;
        let mut dx = gain_dx(k, innovation);
        self.freeze_mount_update_if_needed(&mut k, &mut dx);
        self.record_update_diag(DIAG_BODY_SPEED_X, innovation, obs.s, &obs.h, &k, &dx);
        self.fuse_measurement(obs.s, &obs.h, &k, &dx);
    }

    /// Fuses lateral and vertical nonholonomic body-velocity constraints.
    pub fn fuse_body_vel(&mut self, r_body_vel: f32) {
        self.fuse_body_vel_yz(r_body_vel, r_body_vel);
    }

    /// Fuses lateral and vertical nonholonomic body-velocity constraints with
    /// separate measurement variances for the vehicle Y and Z axes.
    pub fn fuse_body_vel_yz(&mut self, r_body_vel_y: f32, r_body_vel_z: f32) {
        self.fuse_body_vel_yz_batch(r_body_vel_y, r_body_vel_z);
    }

    fn fuse_gps_pos_n(&mut self, pos_n: f32, r_pos_n: f32) {
        let obs = generated_eskf::gps_pos_n_observation(&self.raw.p, r_pos_n);
        let innovation = pos_n - self.raw.nominal.pn;
        let mut k = obs.k;
        let mut dx = gain_dx(k, innovation);
        self.freeze_mount_update_if_needed(&mut k, &mut dx);
        self.record_update_diag(DIAG_GPS_POS, innovation, obs.s, &obs.h, &k, &dx);
        self.fuse_measurement(obs.s, &obs.h, &k, &dx);
    }

    fn fuse_gps_pos_e(&mut self, pos_e: f32, r_pos_e: f32) {
        let obs = generated_eskf::gps_pos_e_observation(&self.raw.p, r_pos_e);
        let innovation = pos_e - self.raw.nominal.pe;
        let mut k = obs.k;
        let mut dx = gain_dx(k, innovation);
        self.freeze_mount_update_if_needed(&mut k, &mut dx);
        self.record_update_diag(DIAG_GPS_POS, innovation, obs.s, &obs.h, &k, &dx);
        self.fuse_measurement(obs.s, &obs.h, &k, &dx);
    }

    fn fuse_gps_pos_d(&mut self, pos_d: f32, r_pos_d: f32) {
        let obs = generated_eskf::gps_pos_d_observation(&self.raw.p, r_pos_d);
        let innovation = pos_d - self.raw.nominal.pd;
        let mut k = obs.k;
        let mut dx = gain_dx(k, innovation);
        self.freeze_mount_update_if_needed(&mut k, &mut dx);
        self.record_update_diag(DIAG_GPS_POS_D, innovation, obs.s, &obs.h, &k, &dx);
        self.fuse_measurement(obs.s, &obs.h, &k, &dx);
    }

    fn fuse_gps_vel_n(&mut self, vel_n: f32, r_vel_n: f32) {
        self.fuse_gps_vel_n_impl(vel_n, r_vel_n, false);
    }

    fn fuse_gps_vel_n_impl(&mut self, vel_n: f32, r_vel_n: f32, block_mount: bool) {
        let innovation = vel_n - self.raw.nominal.vn;
        let obs = generated_eskf::gps_vel_n_observation(&self.raw.p, r_vel_n);
        let mut k = obs.k;
        let mut dx = gain_dx(k, innovation);
        if block_mount {
            block_mount_injection(&mut k);
            block_mount_injection(&mut dx);
        }
        self.freeze_mount_update_if_needed(&mut k, &mut dx);
        let diag = if r_vel_n == RUNTIME_ZERO_VEL_R_DIAG {
            DIAG_ZERO_VEL
        } else {
            DIAG_GPS_VEL
        };
        self.record_update_diag(diag, innovation, obs.s, &obs.h, &k, &dx);
        self.fuse_measurement(obs.s, &obs.h, &k, &dx);
    }

    fn fuse_gps_vel_e(&mut self, vel_e: f32, r_vel_e: f32) {
        self.fuse_gps_vel_e_impl(vel_e, r_vel_e, false);
    }

    fn fuse_gps_vel_e_impl(&mut self, vel_e: f32, r_vel_e: f32, block_mount: bool) {
        let innovation = vel_e - self.raw.nominal.ve;
        let obs = generated_eskf::gps_vel_e_observation(&self.raw.p, r_vel_e);
        let mut k = obs.k;
        let mut dx = gain_dx(k, innovation);
        if block_mount {
            block_mount_injection(&mut k);
            block_mount_injection(&mut dx);
        }
        self.freeze_mount_update_if_needed(&mut k, &mut dx);
        let diag = if r_vel_e == RUNTIME_ZERO_VEL_R_DIAG {
            DIAG_ZERO_VEL
        } else {
            DIAG_GPS_VEL
        };
        self.record_update_diag(diag, innovation, obs.s, &obs.h, &k, &dx);
        self.fuse_measurement(obs.s, &obs.h, &k, &dx);
    }

    fn fuse_gps_vel_d(&mut self, vel_d: f32, r_vel_d: f32) {
        self.fuse_gps_vel_d_impl(vel_d, r_vel_d, false);
    }

    fn fuse_gps_vel_d_impl(&mut self, vel_d: f32, r_vel_d: f32, block_mount: bool) {
        let innovation = vel_d - self.raw.nominal.vd;
        let obs = generated_eskf::gps_vel_d_observation(&self.raw.p, r_vel_d);
        let mut k = obs.k;
        let mut dx = gain_dx(k, innovation);
        if block_mount {
            block_mount_injection(&mut k);
            block_mount_injection(&mut dx);
        }
        self.freeze_mount_update_if_needed(&mut k, &mut dx);
        let diag = if r_vel_d == RUNTIME_ZERO_VEL_R_DIAG {
            DIAG_ZERO_VEL_D
        } else {
            DIAG_GPS_VEL_D
        };
        self.record_update_diag(diag, innovation, obs.s, &obs.h, &k, &dx);
        self.fuse_measurement(obs.s, &obs.h, &k, &dx);
    }

    fn fuse_stationary_gravity_x(&mut self, accel_x: f32, r_stationary_accel: f32) {
        floor_attitude_covariance(&mut self.raw, 0.10 * core::f32::consts::PI / 180.0);
        let obs = generated_eskf::stationary_accel_x_observation(
            &self.raw.nominal,
            &self.raw.p,
            r_stationary_accel,
        );
        let q0 = self.raw.nominal.q0;
        let q1 = self.raw.nominal.q1;
        let q2 = self.raw.nominal.q2;
        let q3 = self.raw.nominal.q3;
        let gravity_x = 2.0 * (q1 * q3 - q0 * q2) * generated_eskf::GRAVITY_MSS;
        let innovation = (accel_x - self.raw.nominal.bax) - (-gravity_x);
        let mut k = obs.k;
        let mut dx = gain_dx(k, innovation);
        self.freeze_mount_update_if_needed(&mut k, &mut dx);
        self.raw.stationary_diag.innovation_x = innovation;
        self.raw.stationary_diag.k_theta_x_from_x = k[0];
        self.raw.stationary_diag.k_theta_y_from_x = k[1];
        self.raw.stationary_diag.k_bax_from_x = k[12];
        self.raw.stationary_diag.k_bay_from_x = k[13];
        self.copy_stationary_p_diag();
        self.record_update_diag(DIAG_STATIONARY_X, innovation, obs.s, &obs.h, &k, &dx);
        self.fuse_measurement(obs.s, &obs.h, &k, &dx);
    }

    fn fuse_stationary_gravity_y(&mut self, accel_y: f32, r_stationary_accel: f32) {
        floor_attitude_covariance(&mut self.raw, 0.10 * core::f32::consts::PI / 180.0);
        let obs = generated_eskf::stationary_accel_y_observation(
            &self.raw.nominal,
            &self.raw.p,
            r_stationary_accel,
        );
        let q0 = self.raw.nominal.q0;
        let q1 = self.raw.nominal.q1;
        let q2 = self.raw.nominal.q2;
        let q3 = self.raw.nominal.q3;
        let gravity_y = 2.0 * (q2 * q3 + q0 * q1) * generated_eskf::GRAVITY_MSS;
        let innovation = (accel_y - self.raw.nominal.bay) - (-gravity_y);
        let mut k = obs.k;
        let mut dx = gain_dx(k, innovation);
        self.freeze_mount_update_if_needed(&mut k, &mut dx);
        self.raw.stationary_diag.innovation_y = innovation;
        self.raw.stationary_diag.k_theta_x_from_y = k[0];
        self.raw.stationary_diag.k_theta_y_from_y = k[1];
        self.raw.stationary_diag.k_bax_from_y = k[12];
        self.raw.stationary_diag.k_bay_from_y = k[13];
        self.copy_stationary_p_diag();
        self.raw.stationary_diag.updates += 1;
        self.record_update_diag(DIAG_STATIONARY_Y, innovation, obs.s, &obs.h, &k, &dx);
        self.fuse_measurement(obs.s, &obs.h, &k, &dx);
    }

    #[allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]
    fn fuse_body_vel_yz_batch(&mut self, r_body_vel_y: f32, r_body_vel_z: f32) {
        let obs_y =
            generated_eskf::body_vel_y_observation(&self.raw.nominal, &self.raw.p, r_body_vel_y);
        let obs_z =
            generated_eskf::body_vel_z_observation(&self.raw.nominal, &self.raw.p, r_body_vel_z);
        let v_vehicle = nominal_vehicle_velocity(&self.raw.nominal);
        let observations = [
            (obs_y, -v_vehicle[1], r_body_vel_y, DIAG_BODY_VEL_Y),
            (obs_z, -v_vehicle[2], r_body_vel_z, DIAG_BODY_VEL_Z),
        ];
        let mut dx = [0.0; ERROR_STATES];

        for (obs_index, (obs, residual, r_body_vel, diag_type)) in
            observations.into_iter().enumerate()
        {
            if !(r_body_vel > 0.0) || !r_body_vel.is_finite() {
                continue;
            }
            let support = if obs_index == 0 {
                &BODY_VEL_Y_SUPPORT
            } else {
                &BODY_VEL_Z_SUPPORT
            };
            let mut ph = [0.0; ERROR_STATES];
            let mut s = r_body_vel;
            let mut hd = 0.0;
            for i in 0..ERROR_STATES {
                for &state in support {
                    ph[i] += self.raw.p[i][state] * obs.h[state];
                }
            }
            for &state in support {
                s += obs.h[state] * ph[state];
                hd += obs.h[state] * dx[state];
            }
            if !(s > 0.0) || !s.is_finite() {
                continue;
            }
            let effective_residual = residual - hd;
            let alpha = effective_residual / s;
            let mut diag_k = [0.0; ERROR_STATES];
            let mut diag_dx = [0.0; ERROR_STATES];
            for i in 0..ERROR_STATES {
                diag_k[i] = ph[i] / s;
                diag_dx[i] = ph[i] * alpha;
            }
            self.freeze_mount_update_if_needed(&mut diag_k, &mut diag_dx);
            for i in 0..ERROR_STATES {
                dx[i] += diag_dx[i];
            }
            self.record_update_diag(diag_type, effective_residual, s, &obs.h, &diag_k, &diag_dx);
            for i in 0..ERROR_STATES {
                for j in i..ERROR_STATES {
                    let updated = self.raw.p[i][j] - diag_k[i] * ph[j] - ph[i] * diag_k[j]
                        + s * diag_k[i] * diag_k[j];
                    self.raw.p[i][j] = updated;
                    self.raw.p[j][i] = updated;
                }
            }
        }

        inject_error_state(&mut self.raw.nominal, &dx);
        apply_reset(&mut self.raw.p, &dx);
        if self.freeze_misalignment_states {
            freeze_mount_covariance(&mut self.raw.p);
        }
    }

    fn fuse_measurement(
        &mut self,
        innovation_var: f32,
        h: &[f32; ERROR_STATES],
        k: &[f32; ERROR_STATES],
        dx: &[f32; ERROR_STATES],
    ) {
        update_covariance_joseph_scalar(&mut self.raw.p, innovation_var, h, k);
        inject_error_state(&mut self.raw.nominal, dx);
        apply_reset(&mut self.raw.p, dx);
        if self.freeze_misalignment_states {
            freeze_mount_covariance(&mut self.raw.p);
        }
    }

    fn fuse_batch(
        &mut self,
        obs_count: usize,
        h_rows: &[[f32; ERROR_STATES]; MAX_BATCH_OBS],
        residuals: &[f32; MAX_BATCH_OBS],
        variances: &[f32; MAX_BATCH_OBS],
        diag_types: &[usize; MAX_BATCH_OBS],
    ) {
        if obs_count == 0 || obs_count > MAX_BATCH_OBS {
            return;
        }
        let mut ph_t = [[0.0; MAX_BATCH_OBS]; ERROR_STATES];
        for i in 0..ERROR_STATES {
            for row in 0..obs_count {
                let mut value = 0.0;
                for state in 0..ERROR_STATES {
                    value += self.raw.p[i][state] * h_rows[row][state];
                }
                ph_t[i][row] = value;
            }
        }

        let mut s = [[0.0; MAX_BATCH_OBS]; MAX_BATCH_OBS];
        for row in 0..obs_count {
            for col in 0..obs_count {
                let mut value = if row == col { variances[row] } else { 0.0 };
                for state in 0..ERROR_STATES {
                    value += h_rows[row][state] * ph_t[state][col];
                }
                s[row][col] = value;
            }
        }

        let Some(s_inv) = invert_batch_matrix(s, obs_count) else {
            return;
        };

        let mut k = [[0.0; MAX_BATCH_OBS]; ERROR_STATES];
        for i in 0..ERROR_STATES {
            for row in 0..obs_count {
                let mut value = 0.0;
                for col in 0..obs_count {
                    value += ph_t[i][col] * s_inv[col][row];
                }
                k[i][row] = value;
            }
        }

        let mut dx = [0.0; ERROR_STATES];
        for i in 0..ERROR_STATES {
            for row in 0..obs_count {
                dx[i] += k[i][row] * residuals[row];
            }
        }
        if self.freeze_misalignment_states {
            block_mount_injection(&mut dx);
            for row in 0..obs_count {
                for i in 15..18 {
                    k[i][row] = 0.0;
                }
            }
        }

        for row in 0..obs_count {
            let mut row_k = [0.0; ERROR_STATES];
            let mut row_dx = [0.0; ERROR_STATES];
            for i in 0..ERROR_STATES {
                row_k[i] = k[i][row];
                row_dx[i] = k[i][row] * residuals[row];
            }
            self.record_update_diag(
                diag_types[row],
                residuals[row],
                s[row][row],
                &h_rows[row],
                &row_k,
                &row_dx,
            );
        }

        let mut updated_p = self.raw.p;
        for i in 0..ERROR_STATES {
            for j in i..ERROR_STATES {
                let mut correction = 0.0;
                for row in 0..obs_count {
                    for col in 0..obs_count {
                        correction += k[i][row] * s[row][col] * k[j][col];
                    }
                }
                let value = self.raw.p[i][j] - correction;
                updated_p[i][j] = value;
                updated_p[j][i] = value;
            }
        }
        self.raw.p = updated_p;
        inject_error_state(&mut self.raw.nominal, &dx);
        apply_reset(&mut self.raw.p, &dx);
        if self.freeze_misalignment_states {
            freeze_mount_covariance(&mut self.raw.p);
        }
    }

    fn freeze_mount_update_if_needed(
        &self,
        k: &mut [f32; ERROR_STATES],
        dx: &mut [f32; ERROR_STATES],
    ) {
        if self.freeze_misalignment_states {
            block_mount_injection(k);
            block_mount_injection(dx);
        }
    }

    fn record_update_diag(
        &mut self,
        diag_type: usize,
        innovation: f32,
        innovation_var: f32,
        h: &[f32; ERROR_STATES],
        k: &[f32; ERROR_STATES],
        dx: &[f32; ERROR_STATES],
    ) {
        if diag_type >= ESKF_UPDATE_DIAG_TYPES {
            return;
        }
        let diag = &mut self.raw.update_diag;
        diag.total_updates += 1;
        diag.type_counts[diag_type] += 1;
        diag.sum_dx_yaw[diag_type] += dx[2];
        diag.sum_abs_dx_yaw[diag_type] += fabsf(dx[2]);
        diag.sum_dx_pitch[diag_type] += dx[1];
        diag.sum_abs_dx_pitch[diag_type] += fabsf(dx[1]);
        diag.sum_abs_dx_vel_h[diag_type] += sqrtf(dx[3] * dx[3] + dx[4] * dx[4]);
        diag.sum_dx_gyro_bias_z[diag_type] += dx[11];
        diag.sum_abs_dx_gyro_bias_z[diag_type] += fabsf(dx[11]);
        diag.sum_abs_dx_mount_norm[diag_type] +=
            sqrtf(dx[15] * dx[15] + dx[16] * dx[16] + dx[17] * dx[17]);
        diag.sum_dx_mount_yaw[diag_type] += dx[17];
        diag.sum_abs_dx_mount_yaw[diag_type] += fabsf(dx[17]);
        diag.sum_innovation[diag_type] += innovation;
        diag.sum_abs_innovation[diag_type] += fabsf(innovation);
        let nis = if innovation_var > 0.0 && innovation_var.is_finite() {
            innovation * innovation / innovation_var
        } else {
            0.0
        };
        let h_mount_norm = sqrtf(h[15] * h[15] + h[16] * h[16] + h[17] * h[17]);
        let k_mount_norm = sqrtf(k[15] * k[15] + k[16] * k[16] + k[17] * k[17]);
        let corr_yaw_mount_yaw = corr_from_cov(&self.raw.p, 2, 17);
        diag.sum_nis[diag_type] += nis;
        diag.max_nis[diag_type] = diag.max_nis[diag_type].max(nis);
        diag.sum_abs_h_yaw[diag_type] += fabsf(h[2]);
        diag.sum_abs_h_gyro_bias_z[diag_type] += fabsf(h[11]);
        diag.sum_h_mount_norm[diag_type] += h_mount_norm;
        diag.sum_abs_k_yaw[diag_type] += fabsf(k[2]);
        diag.sum_k_mount_norm[diag_type] += k_mount_norm;
        diag.sum_abs_corr_yaw_mount_yaw[diag_type] += fabsf(corr_yaw_mount_yaw);
        diag.last_dx_mount_roll = dx[15];
        diag.last_dx_mount_pitch = dx[16];
        diag.last_dx_mount_yaw = dx[17];
        diag.last_dx_mount_roll_by_type[diag_type] = dx[15];
        diag.last_dx_mount_pitch_by_type[diag_type] = dx[16];
        diag.last_dx_mount_yaw_by_type[diag_type] = dx[17];
        diag.last_k_mount_yaw = k[17];
        diag.last_innovation = innovation;
        diag.last_innovation_by_type[diag_type] = innovation;
        diag.last_innovation_var = innovation_var;
        diag.last_nis = nis;
        diag.last_nis_by_type[diag_type] = nis;
        diag.last_h_mount_norm = h_mount_norm;
        diag.last_h_mount_norm_by_type[diag_type] = h_mount_norm;
        diag.last_corr_yaw_mount_yaw = corr_yaw_mount_yaw;
        diag.last_type = diag_type as u32;
    }

    fn copy_stationary_p_diag(&mut self) {
        self.raw.stationary_diag.p_theta_x = self.raw.p[0][0];
        self.raw.stationary_diag.p_theta_y = self.raw.p[1][1];
        self.raw.stationary_diag.p_bax = self.raw.p[12][12];
        self.raw.stationary_diag.p_bay = self.raw.p[13][13];
        self.raw.stationary_diag.p_theta_x_bax = self.raw.p[0][12];
        self.raw.stationary_diag.p_theta_y_bay = self.raw.p[1][13];
    }
}

fn gain_dx(k: [f32; ERROR_STATES], innovation: f32) -> [f32; ERROR_STATES] {
    let mut dx = [0.0; ERROR_STATES];
    for i in 0..ERROR_STATES {
        dx[i] = k[i] * innovation;
    }
    dx
}

fn gps_axis_h(axis: usize) -> [f32; ERROR_STATES] {
    let mut h = [0.0; ERROR_STATES];
    if axis < ERROR_STATES {
        h[axis] = 1.0;
    }
    h
}

#[allow(clippy::too_many_arguments)]
fn push_batch_row(
    h_rows: &mut [[f32; ERROR_STATES]; MAX_BATCH_OBS],
    residuals: &mut [f32; MAX_BATCH_OBS],
    variances: &mut [f32; MAX_BATCH_OBS],
    diag_types: &mut [usize; MAX_BATCH_OBS],
    obs_count: &mut usize,
    h: [f32; ERROR_STATES],
    residual: f32,
    variance: f32,
    diag_type: usize,
) {
    if *obs_count >= MAX_BATCH_OBS || !(variance > 0.0) || !variance.is_finite() {
        return;
    }
    h_rows[*obs_count] = h;
    residuals[*obs_count] = residual;
    variances[*obs_count] = variance;
    diag_types[*obs_count] = diag_type;
    *obs_count += 1;
}

fn invert_batch_matrix(
    input: [[f32; MAX_BATCH_OBS]; MAX_BATCH_OBS],
    n: usize,
) -> Option<[[f32; MAX_BATCH_OBS]; MAX_BATCH_OBS]> {
    let mut a = input;
    let mut inv = [[0.0; MAX_BATCH_OBS]; MAX_BATCH_OBS];
    for i in 0..n {
        inv[i][i] = 1.0;
    }

    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = fabsf(a[col][col]);
        for row in (col + 1)..n {
            let candidate = fabsf(a[row][col]);
            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if pivot_abs < 1.0e-12 || !pivot_abs.is_finite() {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            inv.swap(col, pivot);
        }

        let scale = a[col][col];
        for j in 0..n {
            a[col][j] /= scale;
            inv[col][j] /= scale;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..n {
                a[row][j] -= factor * a[col][j];
                inv[row][j] -= factor * inv[col][j];
            }
        }
    }
    Some(inv)
}

fn block_mount_injection(dx: &mut [f32; ERROR_STATES]) {
    dx[15] = 0.0;
    dx[16] = 0.0;
    dx[17] = 0.0;
}

fn freeze_mount_covariance(p: &mut [[f32; ERROR_STATES]; ERROR_STATES]) {
    for i in 15..18 {
        for j in 0..ERROR_STATES {
            p[i][j] = 0.0;
            p[j][i] = 0.0;
        }
    }
}

fn corr_from_cov(p: &[[f32; ERROR_STATES]; ERROR_STATES], a: usize, b: usize) -> f32 {
    let va = p[a][a];
    let vb = p[b][b];
    if va > 0.0 && vb > 0.0 && va.is_finite() && vb.is_finite() {
        (p[a][b] / sqrtf(va * vb)).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[allow(clippy::needless_range_loop)]
fn update_covariance_joseph_scalar(
    p: &mut [[f32; ERROR_STATES]; ERROR_STATES],
    innovation_var: f32,
    h: &[f32; ERROR_STATES],
    k: &[f32; ERROR_STATES],
) {
    let mut ph = [0.0; ERROR_STATES];
    for i in 0..ERROR_STATES {
        for a in 0..ERROR_STATES {
            ph[i] += p[i][a] * h[a];
        }
    }
    for i in 0..ERROR_STATES {
        for j in i..ERROR_STATES {
            let updated = p[i][j] - k[i] * ph[j] - ph[i] * k[j] + innovation_var * k[i] * k[j];
            p[i][j] = updated;
            p[j][i] = updated;
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn predict_covariance_sparse(
    f: &[[f32; ERROR_STATES]; ERROR_STATES],
    g: &[[f32; NOISE_STATES]; ERROR_STATES],
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    q: &[f32; NOISE_STATES],
) -> [[f32; ERROR_STATES]; ERROR_STATES] {
    let mut next = [[0.0; ERROR_STATES]; ERROR_STATES];
    for i in 0..ERROR_STATES {
        for j in i..ERROR_STATES {
            let mut accum = 0.0;
            for ia in 0..generated_eskf::F_ROW_COUNTS[i] {
                let a = generated_eskf::F_ROW_COLS[i][ia];
                let fi = f[i][a];
                if fi == 0.0 {
                    continue;
                }
                for jb in 0..generated_eskf::F_ROW_COUNTS[j] {
                    let b = generated_eskf::F_ROW_COLS[j][jb];
                    let fj = f[j][b];
                    if fj != 0.0 {
                        accum += fi * p[a][b] * fj;
                    }
                }
            }
            for ia in 0..generated_eskf::G_ROW_COUNTS[i] {
                let a = generated_eskf::G_ROW_COLS[i][ia];
                let gi = g[i][a];
                if gi == 0.0 {
                    continue;
                }
                for jb in 0..generated_eskf::G_ROW_COUNTS[j] {
                    let b = generated_eskf::G_ROW_COLS[j][jb];
                    if a == b {
                        let gj = g[j][b];
                        if gj != 0.0 {
                            accum += gi * q[a] * gj;
                        }
                    }
                }
            }
            next[i][j] = accum;
            next[j][i] = accum;
        }
    }
    next
}

fn normalize_nominal_quat(nominal: &mut EskfNominalState) {
    let mut q = [nominal.q0, nominal.q1, nominal.q2, nominal.q3];
    normalize_quat(&mut q);
    nominal.q0 = q[0];
    nominal.q1 = q[1];
    nominal.q2 = q[2];
    nominal.q3 = q[3];
}

fn normalize_nominal_mount_quat(nominal: &mut EskfNominalState) {
    let mut q = [nominal.qcs0, nominal.qcs1, nominal.qcs2, nominal.qcs3];
    normalize_quat(&mut q);
    nominal.qcs0 = q[0];
    nominal.qcs1 = q[1];
    nominal.qcs2 = q[2];
    nominal.qcs3 = q[3];
}

fn normalize_quat(q: &mut [f32; 4]) {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2 <= 1.0e-12 {
        *q = [1.0, 0.0, 0.0, 0.0];
        return;
    }
    let inv_n = 1.0 / sqrtf(n2);
    q[0] *= inv_n;
    q[1] *= inv_n;
    q[2] *= inv_n;
    q[3] *= inv_n;
}

fn quat_multiply(p: [f32; 4], q: [f32; 4]) -> [f32; 4] {
    [
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    ]
}

fn inject_error_state(nominal: &mut EskfNominalState, dx: &[f32; ERROR_STATES]) {
    let dq = [1.0, 0.5 * dx[0], 0.5 * dx[1], 0.5 * dx[2]];
    let q_old = [nominal.q0, nominal.q1, nominal.q2, nominal.q3];
    let q_new = quat_multiply(q_old, dq);
    nominal.q0 = q_new[0];
    nominal.q1 = q_new[1];
    nominal.q2 = q_new[2];
    nominal.q3 = q_new[3];
    normalize_nominal_quat(nominal);

    nominal.vn += dx[3];
    nominal.ve += dx[4];
    nominal.vd += dx[5];
    nominal.pn += dx[6];
    nominal.pe += dx[7];
    nominal.pd += dx[8];
    nominal.bgx += dx[9];
    nominal.bgy += dx[10];
    nominal.bgz += dx[11];
    nominal.bax += dx[12];
    nominal.bay += dx[13];
    nominal.baz += dx[14];

    let dqcs = [1.0, 0.5 * dx[15], 0.5 * dx[16], 0.5 * dx[17]];
    let qcs_old = [nominal.qcs0, nominal.qcs1, nominal.qcs2, nominal.qcs3];
    let qcs_new = quat_multiply(dqcs, qcs_old);
    nominal.qcs0 = qcs_new[0];
    nominal.qcs1 = qcs_new[1];
    nominal.qcs2 = qcs_new[2];
    nominal.qcs3 = qcs_new[3];
    normalize_nominal_mount_quat(nominal);
}

fn apply_reset(p: &mut [[f32; ERROR_STATES]; ERROR_STATES], dx: &[f32; ERROR_STATES]) {
    apply_reset_block(p, 0, [dx[0], dx[1], dx[2]]);
    apply_reset_block(p, 15, [dx[15], dx[16], dx[17]]);
    symmetrize_p(p);
}

#[allow(clippy::needless_range_loop)]
fn apply_reset_block(p: &mut [[f32; ERROR_STATES]; ERROR_STATES], offset: usize, dtheta: [f32; 3]) {
    let g_reset_theta = generated_eskf::attitude_reset_jacobian(dtheta);
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

fn floor_attitude_covariance(eskf: &mut EskfState, sigma_rad: f32) {
    let var_floor = sigma_rad * sigma_rad;
    if eskf.p[0][0] < var_floor {
        eskf.p[0][0] = var_floor;
    }
    if eskf.p[1][1] < var_floor {
        eskf.p[1][1] = var_floor;
    }
    symmetrize_p(&mut eskf.p);
}

fn symmetrize_p(p: &mut [[f32; ERROR_STATES]; ERROR_STATES]) {
    for i in 0..ERROR_STATES {
        for j in (i + 1)..ERROR_STATES {
            let sym = 0.5 * (p[i][j] + p[j][i]);
            p[i][j] = sym;
            p[j][i] = sym;
        }
    }
}

fn nominal_vehicle_velocity(nominal: &EskfNominalState) -> [f32; 3] {
    let q0 = nominal.q0;
    let q1 = nominal.q1;
    let q2 = nominal.q2;
    let q3 = nominal.q3;
    let vn = nominal.vn;
    let ve = nominal.ve;
    let vd = nominal.vd;
    [
        (1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3) * vn
            + 2.0 * (q1 * q2 + q0 * q3) * ve
            + 2.0 * (q1 * q3 - q0 * q2) * vd,
        2.0 * (q1 * q2 - q0 * q3) * vn
            + (1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3) * ve
            + 2.0 * (q2 * q3 + q0 * q1) * vd,
        2.0 * (q1 * q3 + q0 * q2) * vn
            + 2.0 * (q2 * q3 - q0 * q1) * ve
            + (1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2) * vd,
    ]
}
