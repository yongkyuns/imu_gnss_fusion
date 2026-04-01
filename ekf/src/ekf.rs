#![allow(non_snake_case)]

pub const N_STATES: usize = 16;
pub const GRAVITY_MSS: f32 = 9.80665;
const DEFAULT_P_INIT: f32 = 1.0;
const DEFAULT_BIAS_DT_S: f32 = 0.01;
const DEFAULT_GYRO_BIAS_SIGMA_DPS: f32 = 0.125;
const DEFAULT_ACCEL_BIAS_SIGMA_MPS2: f32 = 0.075;
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PredictNoise {
    pub gyro_var: f32,
    pub accel_var: f32,
    pub gyro_bias_rw_var: f32,
    pub accel_bias_rw_var: f32,
}

impl Default for PredictNoise {
    fn default() -> Self {
        Self {
            gyro_var: 0.0001,
            accel_var: 12.0,
            gyro_bias_rw_var: 0.002e-9,
            accel_bias_rw_var: 0.2e-9,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct CEkfGpsData {
    t_s: f32,
    pos_ned_m: [f32; 3],
    vel_ned_mps: [f32; 3],
    pos_std_m: [f32; 3],
    vel_std_mps: [f32; 3],
    heading_valid: bool,
    heading_rad: f32,
}

unsafe extern "C" {
    fn sf_ekf_init(ekf: *mut Ekf, p_diag: *const f32, noise: *const PredictNoise);
    fn sf_ekf_set_predict_noise(ekf: *mut Ekf, noise: *const PredictNoise);
    fn sf_ekf_predict(ekf: *mut Ekf, imu: *const ImuSample, debug_out: *mut EkfDebug);
    fn sf_ekf_fuse_gps(ekf: *mut Ekf, gps: *const CEkfGpsData);
    fn sf_ekf_fuse_body_vel(ekf: *mut Ekf, r_body_vel: f32);
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

pub fn ekf_init(ekf: &mut Ekf, p_diag: [f32; N_STATES], noise: PredictNoise) {
    unsafe {
        sf_ekf_init(
            ekf as *mut Ekf,
            p_diag.as_ptr(),
            &noise as *const PredictNoise,
        )
    };
}

pub fn ekf_set_predict_noise(ekf: &mut Ekf, noise: PredictNoise) {
    unsafe { sf_ekf_set_predict_noise(ekf as *mut Ekf, &noise as *const PredictNoise) };
}

pub fn ekf_predict(ekf: &mut Ekf, imu: &ImuSample, debug_out: Option<&mut EkfDebug>) {
    let debug_ptr = match debug_out {
        Some(v) => v as *mut EkfDebug,
        None => core::ptr::null_mut(),
    };
    unsafe { sf_ekf_predict(ekf as *mut Ekf, imu as *const ImuSample, debug_ptr) };
}

pub fn ekf_fuse_gps(ekf: &mut Ekf, gps: &GpsData) {
    let cgps = CEkfGpsData {
        t_s: 0.0,
        pos_ned_m: [gps.pos_n, gps.pos_e, gps.pos_d],
        vel_ned_mps: [gps.vel_n, gps.vel_e, gps.vel_d],
        pos_std_m: [
            gps.R_POS_N.max(0.0).sqrt(),
            gps.R_POS_E.max(0.0).sqrt(),
            gps.R_POS_D.max(0.0).sqrt(),
        ],
        vel_std_mps: [
            gps.R_VEL_N.max(0.0).sqrt(),
            gps.R_VEL_E.max(0.0).sqrt(),
            gps.R_VEL_D.max(0.0).sqrt(),
        ],
        heading_valid: false,
        heading_rad: 0.0,
    };
    unsafe { sf_ekf_fuse_gps(ekf as *mut Ekf, &cgps as *const CEkfGpsData) };
}

pub fn ekf_fuse_body_vel(ekf: &mut Ekf, R_body_vel: f32) {
    unsafe { sf_ekf_fuse_body_vel(ekf as *mut Ekf, R_body_vel) };
}

pub fn ekf_fuse_vehicle_vel(ekf: &mut Ekf, q_vb: [f32; 4], r_vehicle_vel: f32) {
    let c_n_b = quat_to_rotmat([ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3]);
    let c_b_n = transpose3(c_n_b);
    let v_b = mat3_vec(c_b_n, [ekf.state.vn, ekf.state.ve, ekf.state.vd]);
    let v_v = mat3_vec(quat_to_rotmat(q_vb), v_b);
    if v_v[1].abs() <= 1.0e-6 && v_v[2].abs() <= 1.0e-6 {
        return;
    }
    let yaw_err = v_v[1].atan2(v_v[0].abs().max(1.0e-6));
    let gain = 1.0 / (1.0 + r_vehicle_vel.max(1.0e-6));
    let dq = quat_from_yaw(gain * yaw_err);
    let q = [ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3];
    let q_new = quat_normalize(quat_mul(dq, q));
    ekf.state.q0 = q_new[0];
    ekf.state.q1 = q_new[1];
    ekf.state.q2 = q_new[2];
    ekf.state.q3 = q_new[3];
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2 <= 1.0e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        let inv = n2.sqrt().recip();
        [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
    }
}

fn quat_from_yaw(yaw_rad: f32) -> [f32; 4] {
    let half = 0.5 * yaw_rad;
    [half.cos(), 0.0, 0.0, half.sin()]
}

fn quat_to_rotmat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn transpose3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mat3_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}
