#![allow(non_snake_case)]

use std::f32::consts::PI;

pub const N_STATES: usize = 16;
pub const GRAVITY_MSS: f32 = 9.80665;

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
}

impl Default for Ekf {
    fn default() -> Self {
        Self {
            state: EkfState::default(),
            p: [[0.0; N_STATES]; N_STATES],
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
    pub heading_rad: f32,
    pub R_POS_N: f32,
    pub R_POS_E: f32,
    pub R_POS_D: f32,
    pub R_VEL_N: f32,
    pub R_VEL_E: f32,
    pub R_VEL_D: f32,
    pub R_YAW: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EkfDebug {
    pub dvb_x: f32,
    pub dvb_y: f32,
    pub dvb_z: f32,
}

#[inline]
fn powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}

#[inline]
fn state_as_array_mut(state: &mut EkfState) -> &mut [f32; N_STATES] {
    // Safe for #[repr(C)] struct with 16 contiguous f32 fields.
    unsafe { &mut *(state as *mut EkfState as *mut [f32; N_STATES]) }
}

#[inline]
fn normalize_quat(q: &mut [f32; 4]) {
    let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if norm_sq > 1e-6 {
        let inv = 1.0 / norm_sq.sqrt();
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
        [1.0 - 2.0 * (q2q2 + q3q3), 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
        [2.0 * (q1 * q2 + q0 * q3), 1.0 - 2.0 * (q1q1 + q3q3), 2.0 * (q2 * q3 - q0 * q1)],
        [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * (q1q1 + q2q2)],
    ]
}

pub fn ekf_init(ekf: &mut Ekf, p_init_val: f32) {
    *ekf = Ekf::default();
    for i in 0..N_STATES {
        ekf.p[i][i] = p_init_val;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ekf_predict(
    ekf: &mut Ekf,
    imu: &ImuSample,
    daVar: f32,
    dvVar: f32,
    dgb_p_noise_var: f32,
    dvb_x_p_noise_var: f32,
    dvb_y_p_noise_var: f32,
    dvb_z_p_noise_var: f32,
    debug_out: Option<&mut EkfDebug>,
) {
    let daxVar = daVar;
    let dayVar = daVar;
    let dazVar = daVar;

    let dvxVar = dvVar;
    let dvyVar = dvVar;
    let dvzVar = dvVar;

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
    let dt = imu.dt;
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
    ekf_fuse_gps_heading(ekf, gps.heading_rad, gps.R_YAW);
}

pub fn ekf_fuse_body_vel(ekf: &mut Ekf, R_body_vel: f32) {
    ekf_fuse_body_vel_y(ekf, R_body_vel);
    ekf_fuse_body_vel_z(ekf, R_body_vel);
}

fn fuse_measurement(ekf: &mut Ekf, innovation: f32, Hfusion: &[f32; N_STATES], Kfusion: &[f32; N_STATES]) {
    let p_old = ekf.p;

    let state_array = state_as_array_mut(&mut ekf.state);
    for i in 0..N_STATES {
        state_array[i] += Kfusion[i] * innovation;
    }

    normalize_state_quat(&mut ekf.state);

    let mut HP = [0.0_f32; N_STATES];
    for j in 0..N_STATES {
        for k in 0..N_STATES {
            HP[j] += Hfusion[k] * p_old[k][j];
        }
    }

    // Scalar covariance update:
    // P <- P - K (H P), where K and H are computed from the same prior P.
    for i in 0..N_STATES {
        for j in 0..N_STATES {
            ekf.p[i][j] = p_old[i][j] - Kfusion[i] * HP[j];
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
    let mut Hfusion = [0.0_f32; N_STATES];
    let mut Kfusion = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_pos_n_generated.rs");
    fuse_measurement(ekf, innovation, &Hfusion, &Kfusion);
}

fn ekf_fuse_gps_pos_e(ekf: &mut Ekf, pos_e: f32, R_POS_E: f32) {
    let pe = ekf.state.pe;
    let P = &ekf.p;
    let innovation = pos_e - pe;
    let mut Hfusion = [0.0_f32; N_STATES];
    let mut Kfusion = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_pos_e_generated.rs");
    fuse_measurement(ekf, innovation, &Hfusion, &Kfusion);
}

fn ekf_fuse_gps_pos_d(ekf: &mut Ekf, pos_d: f32, R_POS_D: f32) {
    let pd = ekf.state.pd;
    let P = &ekf.p;
    let innovation = pos_d - pd;
    let mut Hfusion = [0.0_f32; N_STATES];
    let mut Kfusion = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_pos_d_generated.rs");
    fuse_measurement(ekf, innovation, &Hfusion, &Kfusion);
}

fn ekf_fuse_gps_vel_n(ekf: &mut Ekf, vel_n: f32, R_VEL_N: f32) {
    let vn = ekf.state.vn;
    let P = &ekf.p;
    let innovation = vel_n - vn;
    let mut Hfusion = [0.0_f32; N_STATES];
    let mut Kfusion = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_vel_n_generated.rs");
    fuse_measurement(ekf, innovation, &Hfusion, &Kfusion);
}

fn ekf_fuse_gps_vel_e(ekf: &mut Ekf, vel_e: f32, R_VEL_E: f32) {
    let ve = ekf.state.ve;
    let P = &ekf.p;
    let innovation = vel_e - ve;
    let mut Hfusion = [0.0_f32; N_STATES];
    let mut Kfusion = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_vel_e_generated.rs");
    fuse_measurement(ekf, innovation, &Hfusion, &Kfusion);
}

fn ekf_fuse_gps_vel_d(ekf: &mut Ekf, vel_d: f32, R_VEL_D: f32) {
    let vd = ekf.state.vd;
    let P = &ekf.p;
    let innovation = vel_d - vd;
    let mut Hfusion = [0.0_f32; N_STATES];
    let mut Kfusion = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_vel_d_generated.rs");
    fuse_measurement(ekf, innovation, &Hfusion, &Kfusion);
}

fn ekf_fuse_gps_heading(ekf: &mut Ekf, heading: f32, R_YAW: f32) {
    let q0 = ekf.state.q0;
    let q1 = ekf.state.q1;
    let q2 = ekf.state.q2;
    let q3 = ekf.state.q3;
    let P = &ekf.p;

    let R_10 = 2.0 * (q1 * q2 + q0 * q3);
    let R_00 = 1.0 - 2.0 * (q2 * q2 + q3 * q3);
    let predicted_yaw = R_10.atan2(R_00);

    let mut innovation = heading - predicted_yaw;
    if innovation > PI {
        innovation -= 2.0 * PI;
    } else if innovation < -PI {
        innovation += 2.0 * PI;
    }

    let mut Hfusion = [0.0_f32; N_STATES];
    let mut Kfusion = [0.0_f32; N_STATES];

    include!("ekf_generated/gps_heading_generated.rs");
    fuse_measurement(ekf, innovation, &Hfusion, &Kfusion);
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

    let mut Hfusion = [0.0_f32; N_STATES];
    let mut Kfusion = [0.0_f32; N_STATES];

    include!("ekf_generated/body_vel_y_generated.rs");
    fuse_measurement(ekf, innovation, &Hfusion, &Kfusion);
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

    let mut Hfusion = [0.0_f32; N_STATES];
    let mut Kfusion = [0.0_f32; N_STATES];

    include!("ekf_generated/body_vel_z_generated.rs");
    fuse_measurement(ekf, innovation, &Hfusion, &Kfusion);
}
