#![allow(non_snake_case)]

pub const N_STATES: usize = 3;
pub const GRAVITY_MPS2: f32 = 9.80665;
const YAW_PROCESS_NOISE_SCALE: f32 = 15.0;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MisalignNoise {
    pub q_theta_rw_var: f32, // [rad^2/s]
}

impl Default for MisalignNoise {
    fn default() -> Self {
        Self {
            q_theta_rw_var: 1.0e-6,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MisalignImuSample {
    pub dt: f32,
    pub f_sx: f32,
    pub f_sy: f32,
    pub f_sz: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MisalignAttitudeSample {
    pub q_nb0: f32,
    pub q_nb1: f32,
    pub q_nb2: f32,
    pub q_nb3: f32,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Vma {
    // Body->Sensor quaternion.
    pub q_sb0: f32,
    pub q_sb1: f32,
    pub q_sb2: f32,
    pub q_sb3: f32,
    // 3x3 covariance for small-angle perturbation.
    pub P: [[f32; N_STATES]; N_STATES],
    pub noise: MisalignNoise,
    pub last_residual_n: [f32; 3],

    time_s: f32,
    prev_gnss_time_s: Option<f32>,
    prev_heading_rad: Option<f32>,
    prev_speed_mps: Option<f32>,

    gyro_sum: [f32; 3],
    gyro_count: u32,

    init_done: bool,
    init_acc_sum: [f32; 3],
    init_acc_count: u32,
}

impl Default for Vma {
    fn default() -> Self {
        Self {
            q_sb0: 1.0,
            q_sb1: 0.0,
            q_sb2: 0.0,
            q_sb3: 0.0,
            P: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            noise: MisalignNoise::default(),
            last_residual_n: [0.0; 3],
            time_s: 0.0,
            prev_gnss_time_s: None,
            prev_heading_rad: None,
            prev_speed_mps: None,
            gyro_sum: [0.0; 3],
            gyro_count: 0,
            init_done: false,
            init_acc_sum: [0.0; 3],
            init_acc_count: 0,
        }
    }
}

pub fn vma_init(vma: &mut Vma, p_diag: [f32; N_STATES], noise: MisalignNoise) {
    *vma = Vma::default();
    vma.P[0][0] = p_diag[0];
    vma.P[1][1] = p_diag[1];
    vma.P[2][2] = p_diag[2];
    vma.noise = noise;
}

pub fn vma_set_noise(vma: &mut Vma, noise: MisalignNoise) {
    vma.noise = noise;
}

pub fn vma_set_q_sb(vma: &mut Vma, q_sb: [f32; 4]) {
    let q = quat_normalize(q_sb);
    vma.q_sb0 = q[0];
    vma.q_sb1 = q[1];
    vma.q_sb2 = q[2];
    vma.q_sb3 = q[3];
    vma.init_done = true;
}

pub fn vma_q_sb(vma: &Vma) -> [f32; 4] {
    [vma.q_sb0, vma.q_sb1, vma.q_sb2, vma.q_sb3]
}

pub fn vma_reset_window(vma: &mut Vma) {
    vma.prev_gnss_time_s = None;
    vma.prev_heading_rad = None;
    vma.prev_speed_mps = None;
    vma.gyro_sum = [0.0; 3];
    vma.gyro_count = 0;
    vma.last_residual_n = [0.0; 3];
}

// Compatibility path. If gyro is unavailable, pass zeros.
pub fn vma_predict(vma: &mut Vma, imu: &MisalignImuSample, _att: &MisalignAttitudeSample) {
    vma_predict_gyro(vma, imu, 0.0, 0.0, 0.0);
}

pub fn vma_predict_gyro(vma: &mut Vma, imu: &MisalignImuSample, wx: f32, wy: f32, wz: f32) {
    if !imu.dt.is_finite() || imu.dt <= 0.0 {
        return;
    }

    vma.time_s += imu.dt;

    let qvar = (vma.noise.q_theta_rw_var * imu.dt).max(0.0);
    vma.P[0][0] += qvar;
    vma.P[1][1] += qvar;
    vma.P[2][2] += qvar * YAW_PROCESS_NOISE_SCALE;

    vma.gyro_sum[0] += wx;
    vma.gyro_sum[1] += wy;
    vma.gyro_sum[2] += wz;
    vma.gyro_count = vma.gyro_count.saturating_add(1);

    if !vma.init_done {
        let gyro_norm = (wx * wx + wy * wy + wz * wz).sqrt();
        let acc_norm = (imu.f_sx * imu.f_sx + imu.f_sy * imu.f_sy + imu.f_sz * imu.f_sz).sqrt();
        let gyro_stat = gyro_norm <= (1.0_f32.to_radians());
        let acc_stat = (acc_norm - GRAVITY_MPS2).abs() <= 0.35;
        if gyro_stat && acc_stat {
            vma.init_acc_sum[0] += imu.f_sx;
            vma.init_acc_sum[1] += imu.f_sy;
            vma.init_acc_sum[2] += imu.f_sz;
            vma.init_acc_count = vma.init_acc_count.saturating_add(1);
            if vma.init_acc_count >= 100 {
                let inv = 1.0 / (vma.init_acc_count as f32);
                let ax = vma.init_acc_sum[0] * inv;
                let ay = vma.init_acc_sum[1] * inv;
                let az = vma.init_acc_sum[2] * inv;
                seed_roll_pitch_from_stationary_acc(vma, ax, ay, az);
                vma.init_done = true;
            }
        } else {
            vma.init_acc_sum = [0.0; 3];
            vma.init_acc_count = 0;
        }
    }
}

// GNSS velocity update is used to infer body yaw-rate and constrain q_sb with gyro measurements.
pub fn vma_fuse_velocity(vma: &mut Vma, vel_ned: [f32; 3], r_gyro_diag: [f32; 3]) {
    let speed_h = (vel_ned[0] * vel_ned[0] + vel_ned[1] * vel_ned[1]).sqrt();
    let heading = vel_ned[1].atan2(vel_ned[0]);

    let Some(prev_t) = vma.prev_gnss_time_s else {
        vma.prev_gnss_time_s = Some(vma.time_s);
        vma.prev_heading_rad = Some(heading);
        vma.prev_speed_mps = Some(speed_h);
        vma.gyro_sum = [0.0; 3];
        vma.gyro_count = 0;
        return;
    };
    let Some(prev_heading) = vma.prev_heading_rad else {
        vma.prev_gnss_time_s = Some(vma.time_s);
        vma.prev_heading_rad = Some(heading);
        vma.prev_speed_mps = Some(speed_h);
        return;
    };
    let prev_speed = vma.prev_speed_mps.unwrap_or(speed_h);
    let dt = (vma.time_s - prev_t).max(0.0);

    vma.prev_gnss_time_s = Some(vma.time_s);
    vma.prev_heading_rad = Some(heading);
    vma.prev_speed_mps = Some(speed_h);

    if !vma.init_done || dt < 1.0e-3 || speed_h < 3.0 || prev_speed < 3.0 || vma.gyro_count == 0 {
        vma.gyro_sum = [0.0; 3];
        vma.gyro_count = 0;
        return;
    }

    let dpsi = wrap_pi(heading - prev_heading);
    let yaw_rate_b = dpsi / dt;
    if !yaw_rate_b.is_finite() || yaw_rate_b.abs() < 1.0e-4 {
        vma.gyro_sum = [0.0; 3];
        vma.gyro_count = 0;
        return;
    }

    let inv_n = 1.0 / (vma.gyro_count as f32);
    let gyro_meas = [
        vma.gyro_sum[0] * inv_n,
        vma.gyro_sum[1] * inv_n,
        vma.gyro_sum[2] * inv_n,
    ];
    vma.gyro_sum = [0.0; 3];
    vma.gyro_count = 0;

    let q_sb0 = vma.q_sb0;
    let q_sb1 = vma.q_sb1;
    let q_sb2 = vma.q_sb2;
    let q_sb3 = vma.q_sb3;

    let mut gyro_pred = [0.0_f32; 3];
    include!("vma_generated/gyro_rate_pred_generated.rs");

    let mut H_gyro = [[0.0_f32; N_STATES]; 3];
    include!("vma_generated/gyro_rate_obs_jacobian_generated.rs");

    let residual = [
        gyro_meas[0] - gyro_pred[0],
        gyro_meas[1] - gyro_pred[1],
        gyro_meas[2] - gyro_pred[2],
    ];
    vma.last_residual_n = residual;

    let p = vma.P;
    let hp = mat3_mul(H_gyro, p);
    let mut s = mat3_mul(hp, mat3_transpose(H_gyro));
    s[0][0] += r_gyro_diag[0].max(1.0e-8);
    s[1][1] += r_gyro_diag[1].max(1.0e-8);
    s[2][2] += r_gyro_diag[2].max(1.0e-8);
    let Some(s_inv) = mat3_inv(s) else {
        return;
    };

    let ph_t = mat3_mul(p, mat3_transpose(H_gyro));
    let k = mat3_mul(ph_t, s_inv);
    let delta = mat3_vec(k, residual);

    let dq = quat_from_small_angle(delta);
    let q_new = quat_normalize(quat_mul(dq, [vma.q_sb0, vma.q_sb1, vma.q_sb2, vma.q_sb3]));
    vma.q_sb0 = q_new[0];
    vma.q_sb1 = q_new[1];
    vma.q_sb2 = q_new[2];
    vma.q_sb3 = q_new[3];

    let kh = mat3_mul(k, H_gyro);
    let i_kh = mat3_sub(mat3_identity(), kh);
    let mut p_new = mat3_mul(i_kh, p);
    p_new = mat3_symmetrize(p_new);
    p_new[0][0] = p_new[0][0].max(1.0e-10);
    p_new[1][1] = p_new[1][1].max(1.0e-10);
    p_new[2][2] = p_new[2][2].max(1.0e-10);
    vma.P = p_new;
}

fn seed_roll_pitch_from_stationary_acc(vma: &mut Vma, ax: f32, ay: f32, az: f32) {
    let n = (ax * ax + ay * ay + az * az).sqrt();
    if n < 1.0e-6 {
        return;
    }
    let gx = ax / n;
    let gy = ay / n;
    let gz = az / n;

    // body->sensor with yaw fixed to 0, inferred from gravity direction only.
    let roll = (-gy).clamp(-1.0, 1.0).asin();
    let pitch = gx.atan2(gz);
    let q = quat_from_rpy_zyx(roll, pitch, 0.0);
    vma.q_sb0 = q[0];
    vma.q_sb1 = q[1];
    vma.q_sb2 = q[2];
    vma.q_sb3 = q[3];
}

fn wrap_pi(mut a: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    while a > std::f32::consts::PI {
        a -= two_pi;
    }
    while a < -std::f32::consts::PI {
        a += two_pi;
    }
    a
}

fn quat_from_rpy_zyx(roll: f32, pitch: f32, yaw: f32) -> [f32; 4] {
    let (sr, cr) = (0.5 * roll).sin_cos();
    let (sp, cp) = (0.5 * pitch).sin_cos();
    let (sy, cy) = (0.5 * yaw).sin_cos();
    [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]
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
        return [1.0, 0.0, 0.0, 0.0];
    }
    let inv = n2.sqrt().recip();
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

fn quat_from_small_angle(dtheta: [f32; 3]) -> [f32; 4] {
    let half = [0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]];
    quat_normalize([1.0, half[0], half[1], half[2]])
}

fn mat3_identity() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn mat3_transpose(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn mat3_vec(a: [[f32; 3]; 3], x: [f32; 3]) -> [f32; 3] {
    [
        a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
        a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
        a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
    ]
}

fn mat3_sub(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [a[0][0] - b[0][0], a[0][1] - b[0][1], a[0][2] - b[0][2]],
        [a[1][0] - b[1][0], a[1][1] - b[1][1], a[1][2] - b[1][2]],
        [a[2][0] - b[2][0], a[2][1] - b[2][1], a[2][2] - b[2][2]],
    ]
}

fn mat3_symmetrize(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = a;
    for i in 0..3 {
        for j in i..3 {
            let v = 0.5 * (out[i][j] + out[j][i]);
            out[i][j] = v;
            out[j][i] = v;
        }
    }
    out
}

fn mat3_inv(a: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1.0e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det,
        ],
    ])
}
