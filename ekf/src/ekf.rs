#![allow(non_snake_case)]

pub const N_STATES: usize = 16;
pub const GRAVITY_MSS: f32 = 9.80665;
const DEFAULT_P_INIT: f32 = 1.0;
const DEFAULT_BIAS_DT_S: f32 = 0.01;
const DEFAULT_GYRO_BIAS_SIGMA_DPS: f32 = 0.06;
const DEFAULT_ACCEL_BIAS_SIGMA_MPS2: f32 = 0.03;
const PI_F32: f32 = 3.141_592_7;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EkfState {
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
    pub dax_b: f32,
    pub day_b: f32,
    pub daz_b: f32,
    pub dvx_b: f32,
    pub dvy_b: f32,
    pub dvz_b: f32,
}

impl EkfState {
    #[inline]
    fn add_scaled_k(&mut self, k: &[f32; N_STATES], innovation: f32) {
        self.q0 += k[0] * innovation;
        self.q1 += k[1] * innovation;
        self.q2 += k[2] * innovation;
        self.q3 += k[3] * innovation;
        self.vn += k[4] * innovation;
        self.ve += k[5] * innovation;
        self.vd += k[6] * innovation;
        self.pn += k[7] * innovation;
        self.pe += k[8] * innovation;
        self.pd += k[9] * innovation;
        self.dax_b += k[10] * innovation;
        self.day_b += k[11] * innovation;
        self.daz_b += k[12] * innovation;
        self.dvx_b += k[13] * innovation;
        self.dvy_b += k[14] * innovation;
        self.dvz_b += k[15] * innovation;
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ImuSample {
    pub dax: f32,
    pub day: f32,
    pub daz: f32,
    pub dvx: f32,
    pub dvy: f32,
    pub dvz: f32,
    pub dt: f32,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Ekf {
    pub state: EkfState,
    pub p: [[f32; N_STATES]; N_STATES],
    pub noise: PredictNoise,
}

impl Default for Ekf {
    fn default() -> Self {
        let mut state = EkfState::default();
        state.q0 = 1.0;
        let mut p = [[0.0; N_STATES]; N_STATES];
        let p_diag = default_p_diag();
        for i in 0..N_STATES {
            p[i][i] = p_diag[i];
        }
        Self {
            state,
            p,
            noise: PredictNoise::default(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpsData {
    pub pos_n: f32,
    pub pos_e: f32,
    pub pos_d: f32,
    pub vel_n: f32,
    pub vel_e: f32,
    pub vel_d: f32,
    pub R_POS_N: f32,
    pub R_POS_E: f32,
    pub R_POS_D: f32,
    pub R_VEL_N: f32,
    pub R_VEL_E: f32,
    pub R_VEL_D: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EkfDebug {
    pub dvb_x: f32,
    pub dvb_y: f32,
    pub dvb_z: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct PredictNoise {
    // Continuous-time sensor variance inputs in intuitive physical units.
    pub gyro_var: f32,          // [rad^2/s^2]
    pub accel_var: f32,         // [(m/s^2)^2]
    pub gyro_bias_rw_var: f32,  // [rad^2/s^2], applied to bias process model
    pub accel_bias_rw_var: f32, // [(m/s^2)^2], applied to bias process model
}

impl Default for PredictNoise {
    fn default() -> Self {
        Self {
            // Defaults aligned with visualize_pygpsdata_log's prior tuning.
            gyro_var: 0.01,  // [rad^2/s^2]
            accel_var: 12.0, // [(m/s^2)^2]
            gyro_bias_rw_var: 0.1e-9,
            accel_bias_rw_var: 1.0e-8,
        }
    }
}

fn default_p_diag() -> [f32; N_STATES] {
    let mut p = [DEFAULT_P_INIT; N_STATES];
    let dt = DEFAULT_BIAS_DT_S;
    let gyro_sigma_da = (DEFAULT_GYRO_BIAS_SIGMA_DPS * PI_F32 / 180.0) * dt;
    let accel_sigma_dv = DEFAULT_ACCEL_BIAS_SIGMA_MPS2 * dt;
    let var_gyro = gyro_sigma_da * gyro_sigma_da;
    let var_accel = accel_sigma_dv * accel_sigma_dv;
    p[10] = var_gyro;
    p[11] = var_gyro;
    p[12] = var_gyro;
    p[13] = var_accel;
    p[14] = var_accel;
    p[15] = var_accel;
    p
}

#[inline]
fn normalize_quat(q: &mut [f32; 4]) {
    let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if norm_sq > 1e-6 {
        let inv = 1.0 / libm::sqrtf(norm_sq);
        q[0] *= inv;
        q[1] *= inv;
        q[2] *= inv;
        q[3] *= inv;
    } else {
        q[0] = 1.0;
        q[1] = 0.0;
        q[2] = 0.0;
        q[3] = 0.0;
    }
}

#[inline]
fn normalize_state_quat(state: &mut EkfState) {
    let mut q = [state.q0, state.q1, state.q2, state.q3];
    normalize_quat(&mut q);
    state.q0 = q[0];
    state.q1 = q[1];
    state.q2 = q[2];
    state.q3 = q[3];
}

#[inline]
fn quat2rot(q: &[f32; 4]) -> [[f32; 3]; 3] {
    let q0 = q[0];
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];
    let q1q1 = q1 * q1;
    let q2q2 = q2 * q2;
    let q3q3 = q3 * q3;
    [
        [
            1.0 - 2.0 * (q2q2 + q3q3),
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * (q1q1 + q3q3),
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * (q1q1 + q2q2),
        ],
    ]
}

pub fn ekf_init(ekf: &mut Ekf, p_diag: [f32; N_STATES], noise: PredictNoise) {
    *ekf = Ekf::default();
    ekf.noise = noise;
    for i in 0..N_STATES {
        ekf.p[i][i] = p_diag[i];
    }
}

pub fn ekf_set_predict_noise(ekf: &mut Ekf, noise: PredictNoise) {
    ekf.noise = noise;
}

pub fn ekf_predict(ekf: &mut Ekf, imu: &ImuSample, debug_out: Option<&mut EkfDebug>) {
    let noise = ekf.noise;
    let dt = imu.dt;
    let dt2 = dt * dt;
    // Internal prediction equations are formulated in delta-angle / delta-velocity.
    // Convert rate-domain variances to increment-domain variances each step.
    let dAngVar = noise.gyro_var * dt2;
    let dVelVar = noise.accel_var * dt2;
    let gyro_bias_rw_var = noise.gyro_bias_rw_var;
    let accel_bias_rw_var = noise.accel_bias_rw_var;

    let q0 = ekf.state.q0;
    let q1 = ekf.state.q1;
    let q2 = ekf.state.q2;
    let q3 = ekf.state.q3;
    let vn = ekf.state.vn;
    let ve = ekf.state.ve;
    let vd = ekf.state.vd;
    let pn = ekf.state.pn;
    let pe = ekf.state.pe;
    let pd = ekf.state.pd;
    let dax_b = ekf.state.dax_b;
    let day_b = ekf.state.day_b;
    let daz_b = ekf.state.daz_b;
    let dvx_b = ekf.state.dvx_b;
    let dvy_b = ekf.state.dvy_b;
    let dvz_b = ekf.state.dvz_b;

    let dax = imu.dax;
    let day = imu.day;
    let daz = imu.daz;
    let dvx = imu.dvx;
    let dvy = imu.dvy;
    let dvz = imu.dvz;
    let g = GRAVITY_MSS;

    if let Some(debug) = debug_out {
        let R = quat2rot(&[ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3]);
        debug.dvb_x = dvx - dvx_b + R[2][0] * g * dt;
        debug.dvb_y = dvy - dvy_b + R[2][1] * g * dt;
        debug.dvb_z = dvz - dvz_b + R[2][2] * g * dt;
    }

    include!("ekf_generated/prediction_generated.rs");

    normalize_state_quat(&mut ekf.state);

    let mut next_p = [[0.0_f32; N_STATES]; N_STATES];
    let P = &ekf.p;
    let nextP = &mut next_p;

    include!("ekf_generated/covariance_generated.rs");

    for i in 0..N_STATES {
        for j in (i + 1)..N_STATES {
            next_p[j][i] = next_p[i][j];
        }
    }
    ekf.p = next_p;
}

pub fn ekf_fuse_gps(ekf: &mut Ekf, gps: &GpsData) {
    ekf_fuse_gps_pos_n(ekf, gps.pos_n, gps.R_POS_N);
    ekf_fuse_gps_pos_e(ekf, gps.pos_e, gps.R_POS_E);
    ekf_fuse_gps_pos_d(ekf, gps.pos_d, gps.R_POS_D);
    ekf_fuse_gps_vel_n(ekf, gps.vel_n, gps.R_VEL_N);
    ekf_fuse_gps_vel_e(ekf, gps.vel_e, gps.R_VEL_E);
    ekf_fuse_gps_vel_d(ekf, gps.vel_d, gps.R_VEL_D);
}

pub fn ekf_fuse_body_vel(ekf: &mut Ekf, R_body_vel: f32) {
    ekf_fuse_body_vel_y(ekf, R_body_vel);
    ekf_fuse_body_vel_z(ekf, R_body_vel);
}

pub fn ekf_fuse_vehicle_vel(ekf: &mut Ekf, q_vb: [f32; 4], r_vehicle_vel: f32) {
    ekf_fuse_vehicle_vel_axis(ekf, q_vb, r_vehicle_vel, 1);
    ekf_fuse_vehicle_vel_axis(ekf, q_vb, r_vehicle_vel, 2);
}

fn fuse_measurement(ekf: &mut Ekf, innovation: f32, H: &[f32; N_STATES], K: &[f32; N_STATES]) {
    let p_old = ekf.p;

    ekf.state.add_scaled_k(K, innovation);

    normalize_state_quat(&mut ekf.state);

    let mut HP = [0.0_f32; N_STATES];
    for j in 0..N_STATES {
        for k in 0..N_STATES {
            HP[j] += H[k] * p_old[k][j];
        }
    }

    // Scalar covariance update:
    // P <- P - K (H P), where K and H are computed from the same prior P.
    for i in 0..N_STATES {
        for j in 0..N_STATES {
            ekf.p[i][j] = p_old[i][j] - K[i] * HP[j];
        }
    }

    for i in 0..N_STATES {
        for j in i..N_STATES {
            let temp = (ekf.p[i][j] + ekf.p[j][i]) * 0.5;
            ekf.p[i][j] = temp;
            ekf.p[j][i] = temp;
        }
    }
}

fn ekf_fuse_gps_pos_n(ekf: &mut Ekf, pos_n: f32, R_POS_N: f32) {
    let pn = ekf.state.pn;
    let P = &ekf.p;
    let innovation = pos_n - pn;
    let mut H = [0.0_f32; N_STATES];
    let mut K = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_pos_n_generated.rs");
    fuse_measurement(ekf, innovation, &H, &K);
}

fn ekf_fuse_gps_pos_e(ekf: &mut Ekf, pos_e: f32, R_POS_E: f32) {
    let pe = ekf.state.pe;
    let P = &ekf.p;
    let innovation = pos_e - pe;
    let mut H = [0.0_f32; N_STATES];
    let mut K = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_pos_e_generated.rs");
    fuse_measurement(ekf, innovation, &H, &K);
}

fn ekf_fuse_gps_pos_d(ekf: &mut Ekf, pos_d: f32, R_POS_D: f32) {
    let pd = ekf.state.pd;
    let P = &ekf.p;
    let innovation = pos_d - pd;
    let mut H = [0.0_f32; N_STATES];
    let mut K = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_pos_d_generated.rs");
    fuse_measurement(ekf, innovation, &H, &K);
}

fn ekf_fuse_gps_vel_n(ekf: &mut Ekf, vel_n: f32, R_VEL_N: f32) {
    let vn = ekf.state.vn;
    let P = &ekf.p;
    let innovation = vel_n - vn;
    let mut H = [0.0_f32; N_STATES];
    let mut K = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_vel_n_generated.rs");
    fuse_measurement(ekf, innovation, &H, &K);
}

fn ekf_fuse_gps_vel_e(ekf: &mut Ekf, vel_e: f32, R_VEL_E: f32) {
    let ve = ekf.state.ve;
    let P = &ekf.p;
    let innovation = vel_e - ve;
    let mut H = [0.0_f32; N_STATES];
    let mut K = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_vel_e_generated.rs");
    fuse_measurement(ekf, innovation, &H, &K);
}

fn ekf_fuse_gps_vel_d(ekf: &mut Ekf, vel_d: f32, R_VEL_D: f32) {
    let vd = ekf.state.vd;
    let P = &ekf.p;
    let innovation = vel_d - vd;
    let mut H = [0.0_f32; N_STATES];
    let mut K = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_vel_d_generated.rs");
    fuse_measurement(ekf, innovation, &H, &K);
}

fn ekf_fuse_body_vel_y(ekf: &mut Ekf, R_BODY_VEL: f32) {
    let q0 = ekf.state.q0;
    let q1 = ekf.state.q1;
    let q2 = ekf.state.q2;
    let q3 = ekf.state.q3;
    let vn = ekf.state.vn;
    let ve = ekf.state.ve;
    let vd = ekf.state.vd;
    let P = &ekf.p;

    let R_T_10 = 2.0 * (q1 * q2 - q0 * q3);
    let R_T_11 = 1.0 - 2.0 * (q1 * q1 + q3 * q3);
    let R_T_12 = 2.0 * (q2 * q3 + q0 * q1);
    let v_body_y = R_T_10 * vn + R_T_11 * ve + R_T_12 * vd;
    let innovation = -v_body_y;

    let mut H = [0.0_f32; N_STATES];
    let mut K = [0.0_f32; N_STATES];

    include!("ekf_generated/body_vel_y_generated.rs");
    fuse_measurement(ekf, innovation, &H, &K);
}

fn ekf_fuse_body_vel_z(ekf: &mut Ekf, R_BODY_VEL: f32) {
    let q0 = ekf.state.q0;
    let q1 = ekf.state.q1;
    let q2 = ekf.state.q2;
    let q3 = ekf.state.q3;
    let vn = ekf.state.vn;
    let ve = ekf.state.ve;
    let vd = ekf.state.vd;
    let P = &ekf.p;

    let R_T_20 = 2.0 * (q1 * q3 + q0 * q2);
    let R_T_21 = 2.0 * (q2 * q3 - q0 * q1);
    let R_T_22 = 1.0 - 2.0 * (q1 * q1 + q2 * q2);
    let v_body_z = R_T_20 * vn + R_T_21 * ve + R_T_22 * vd;
    let innovation = -v_body_z;

    let mut H = [0.0_f32; N_STATES];
    let mut K = [0.0_f32; N_STATES];

    include!("ekf_generated/body_vel_z_generated.rs");
    fuse_measurement(ekf, innovation, &H, &K);
}

fn ekf_fuse_vehicle_vel_axis(ekf: &mut Ekf, q_vb: [f32; 4], r_vehicle_vel: f32, axis: usize) {
    let pred = vehicle_velocity_prediction(&ekf.state, q_vb);
    let innovation = -pred[axis - 1];
    let h = vehicle_vel_jacobian_row(&ekf.state, q_vb, axis);

    let p_old = ekf.p;
    let mut ph = [0.0_f32; N_STATES];
    for i in 0..N_STATES {
        for (j, hj) in h.iter().enumerate() {
            ph[i] += p_old[i][j] * *hj;
        }
    }

    let mut s = r_vehicle_vel.max(1.0e-9);
    for j in 0..N_STATES {
        s += h[j] * ph[j];
    }
    let inv_s = if s > 1.0e-9 { 1.0 / s } else { 0.0 };

    let mut k = [0.0_f32; N_STATES];
    for i in 0..N_STATES {
        k[i] = ph[i] * inv_s;
    }

    fuse_measurement(ekf, innovation, &h, &k);
}

fn vehicle_velocity_prediction(state: &EkfState, q_vb: [f32; 4]) -> [f32; 2] {
    let c_n_b = quat2rot(&[state.q0, state.q1, state.q2, state.q3]);
    let c_b_n = transpose3(c_n_b);
    let v_n = [state.vn, state.ve, state.vd];
    let v_b = mat3_vec(c_b_n, v_n);
    let v_v = mat3_vec(quat2rot(&q_vb), v_b);
    [v_v[1], v_v[2]]
}

fn vehicle_vel_jacobian_row(state: &EkfState, q_vb: [f32; 4], axis: usize) -> [f32; N_STATES] {
    let base = vehicle_velocity_prediction(state, q_vb)[axis - 1];
    let mut h = [0.0_f32; N_STATES];
    let eps = 1.0e-5_f32;
    for idx in 0..7 {
        let mut perturbed = *state;
        match idx {
            0 => perturbed.q0 += eps,
            1 => perturbed.q1 += eps,
            2 => perturbed.q2 += eps,
            3 => perturbed.q3 += eps,
            4 => perturbed.vn += eps,
            5 => perturbed.ve += eps,
            6 => perturbed.vd += eps,
            _ => unreachable!(),
        }
        normalize_state_quat(&mut perturbed);
        let shifted = vehicle_velocity_prediction(&perturbed, q_vb)[axis - 1];
        h[idx] = (shifted - base) / eps;
    }
    h
}

#[inline]
fn transpose3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[inline]
fn mat3_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}
