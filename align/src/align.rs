#![allow(non_snake_case)]

pub const N_STATES: usize = 3;
pub const GRAVITY_MPS2: f32 = 9.80665;
const YAW_PROCESS_NOISE_SCALE: f32 = 5.0;
const FORWARD_ACCEL_MIN_MPS2: f32 = 0.15;
const FORWARD_DYNAMIC_ACCEL_MIN_MPS2: f32 = 0.2;
const FORWARD_LATERAL_DOMINANCE_RATIO: f32 = 1.5;

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
pub struct Align {
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
    accel_sum: [f32; 3],
    accel_count: u32,

    init_done: bool,
    init_acc_sum: [f32; 3],
    init_acc_count: u32,
    gravity_ref_s: [f32; 3],
    gravity_ref_valid: bool,
}

impl Default for Align {
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
            accel_sum: [0.0; 3],
            accel_count: 0,
            init_done: false,
            init_acc_sum: [0.0; 3],
            init_acc_count: 0,
            gravity_ref_s: [0.0; 3],
            gravity_ref_valid: false,
        }
    }
}

pub fn align_init(filter: &mut Align, p_diag: [f32; N_STATES], noise: MisalignNoise) {
    *filter = Align::default();
    filter.P[0][0] = p_diag[0];
    filter.P[1][1] = p_diag[1];
    filter.P[2][2] = p_diag[2];
    filter.noise = noise;
}

pub fn align_set_noise(filter: &mut Align, noise: MisalignNoise) {
    filter.noise = noise;
}

pub fn align_set_q_sb(filter: &mut Align, q_sb: [f32; 4]) {
    let q = quat_normalize(q_sb);
    filter.q_sb0 = q[0];
    filter.q_sb1 = q[1];
    filter.q_sb2 = q[2];
    filter.q_sb3 = q[3];
    filter.init_done = true;
}

pub fn align_q_sb(filter: &Align) -> [f32; 4] {
    [filter.q_sb0, filter.q_sb1, filter.q_sb2, filter.q_sb3]
}

pub fn align_reset_window(filter: &mut Align) {
    filter.prev_gnss_time_s = None;
    filter.prev_heading_rad = None;
    filter.prev_speed_mps = None;
    filter.gyro_sum = [0.0; 3];
    filter.gyro_count = 0;
    filter.accel_sum = [0.0; 3];
    filter.accel_count = 0;
    filter.last_residual_n = [0.0; 3];
}

// Compatibility path. If gyro is unavailable, pass zeros.
pub fn align_predict(filter: &mut Align, imu: &MisalignImuSample, _att: &MisalignAttitudeSample) {
    align_predict_gyro(filter, imu, 0.0, 0.0, 0.0);
}

pub fn align_predict_gyro(filter: &mut Align, imu: &MisalignImuSample, wx: f32, wy: f32, wz: f32) {
    if !imu.dt.is_finite() || imu.dt <= 0.0 {
        return;
    }

    filter.time_s += imu.dt;

    let qvar = (filter.noise.q_theta_rw_var * imu.dt).max(0.0);
    filter.P[0][0] += qvar;
    filter.P[1][1] += qvar;
    filter.P[2][2] += qvar * YAW_PROCESS_NOISE_SCALE;

    filter.gyro_sum[0] += wx;
    filter.gyro_sum[1] += wy;
    filter.gyro_sum[2] += wz;
    filter.gyro_count = filter.gyro_count.saturating_add(1);
    filter.accel_sum[0] += imu.f_sx;
    filter.accel_sum[1] += imu.f_sy;
    filter.accel_sum[2] += imu.f_sz;
    filter.accel_count = filter.accel_count.saturating_add(1);

    if !filter.init_done {
        let gyro_norm = (wx * wx + wy * wy + wz * wz).sqrt();
        let acc_norm = (imu.f_sx * imu.f_sx + imu.f_sy * imu.f_sy + imu.f_sz * imu.f_sz).sqrt();
        let gyro_stat = gyro_norm <= (1.0_f32.to_radians());
        let acc_stat = (acc_norm - GRAVITY_MPS2).abs() <= 0.35;
        if gyro_stat && acc_stat {
            filter.init_acc_sum[0] += imu.f_sx;
            filter.init_acc_sum[1] += imu.f_sy;
            filter.init_acc_sum[2] += imu.f_sz;
            filter.init_acc_count = filter.init_acc_count.saturating_add(1);
            if filter.init_acc_count >= 100 {
                let inv = 1.0 / (filter.init_acc_count as f32);
                let ax = filter.init_acc_sum[0] * inv;
                let ay = filter.init_acc_sum[1] * inv;
                let az = filter.init_acc_sum[2] * inv;
                seed_roll_pitch_from_stationary_acc(filter, ax, ay, az);
                filter.gravity_ref_s = [ax, ay, az];
                filter.gravity_ref_valid = true;
                filter.init_done = true;
            }
        } else {
            filter.init_acc_sum = [0.0; 3];
            filter.init_acc_count = 0;
        }
    }
}

// GNSS velocity update is used to infer body yaw-rate and constrain q_sb with gyro measurements.
pub fn align_fuse_velocity(filter: &mut Align, vel_ned: [f32; 3], r_gyro_diag: [f32; 3]) {
    align_fuse_motion(filter, vel_ned, Some(r_gyro_diag), None);
}

// Combined GNSS motion update:
// - gyro/course-rate update constrains yaw-axis consistency
// - signed forward-accel update breaks the 180 deg yaw ambiguity
pub fn align_fuse_velocity_forward(
    filter: &mut Align,
    vel_ned: [f32; 3],
    r_gyro_diag: [f32; 3],
    r_forward_diag: [f32; 3],
) {
    align_fuse_motion(filter, vel_ned, Some(r_gyro_diag), Some(r_forward_diag));
}

fn align_fuse_motion(
    filter: &mut Align,
    vel_ned: [f32; 3],
    r_gyro_diag: Option<[f32; 3]>,
    r_forward_diag: Option<[f32; 3]>,
) {
    let speed_h = (vel_ned[0] * vel_ned[0] + vel_ned[1] * vel_ned[1]).sqrt();
    let heading = vel_ned[1].atan2(vel_ned[0]);

    let Some(prev_t) = filter.prev_gnss_time_s else {
        filter.prev_gnss_time_s = Some(filter.time_s);
        filter.prev_heading_rad = Some(heading);
        filter.prev_speed_mps = Some(speed_h);
        clear_motion_window(filter);
        return;
    };
    let Some(prev_heading) = filter.prev_heading_rad else {
        filter.prev_gnss_time_s = Some(filter.time_s);
        filter.prev_heading_rad = Some(heading);
        filter.prev_speed_mps = Some(speed_h);
        clear_motion_window(filter);
        return;
    };
    let prev_speed = filter.prev_speed_mps.unwrap_or(speed_h);
    let dt = (filter.time_s - prev_t).max(0.0);

    filter.prev_gnss_time_s = Some(filter.time_s);
    filter.prev_heading_rad = Some(heading);
    filter.prev_speed_mps = Some(speed_h);

    if !filter.init_done || dt < 1.0e-3 || speed_h < 3.0 || prev_speed < 3.0 {
        clear_motion_window(filter);
        return;
    }

    let dpsi = wrap_pi(heading - prev_heading);
    let yaw_rate_b = dpsi / dt;
    let gyro_meas = if filter.gyro_count > 0 {
        let inv_n = 1.0 / (filter.gyro_count as f32);
        Some([
            filter.gyro_sum[0] * inv_n,
            filter.gyro_sum[1] * inv_n,
            filter.gyro_sum[2] * inv_n,
        ])
    } else {
        None
    };
    let accel_meas = if filter.accel_count > 0 {
        let inv_n = 1.0 / (filter.accel_count as f32);
        Some([
            filter.accel_sum[0] * inv_n,
            filter.accel_sum[1] * inv_n,
            filter.accel_sum[2] * inv_n,
        ])
    } else {
        None
    };
    clear_motion_window(filter);

    if let (Some(r_gyro_diag), Some(gyro_meas)) = (r_gyro_diag, gyro_meas) {
        if yaw_rate_b.is_finite() && yaw_rate_b.abs() >= 1.0e-4 {
            let q_sb0 = filter.q_sb0;
            let q_sb1 = filter.q_sb1;
            let q_sb2 = filter.q_sb2;
            let q_sb3 = filter.q_sb3;

            let mut gyro_pred = [0.0_f32; 3];
            include!("align_generated/gyro_rate_pred_generated.rs");

            let mut H_gyro = [[0.0_f32; N_STATES]; 3];
            include!("align_generated/gyro_rate_obs_jacobian_generated.rs");

            let residual = [
                gyro_meas[0] - gyro_pred[0],
                gyro_meas[1] - gyro_pred[1],
                gyro_meas[2] - gyro_pred[2],
            ];
            filter.last_residual_n = residual;
            apply_measurement_update(filter, residual, H_gyro, r_gyro_diag);
        }
    }

    if let (Some(r_forward_diag), Some(accel_meas)) = (r_forward_diag, accel_meas) {
        if let Some(forward_meas_s) =
            build_forward_axis_measurement(filter, accel_meas, speed_h, prev_speed, yaw_rate_b, dt)
        {
            let mut forward_pred_s = body_forward_in_sensor(filter);
            if vec3_dot(forward_meas_s, forward_pred_s) < -0.25 {
                flip_yaw_branch(filter);
                forward_pred_s = body_forward_in_sensor(filter);
            }
            let residual = vec3_sub(forward_meas_s, forward_pred_s);
            let h_forward = neg_skew(forward_pred_s);
            filter.last_residual_n = residual;
            apply_measurement_update(filter, residual, h_forward, r_forward_diag);
        }
    }
}

fn seed_roll_pitch_from_stationary_acc(filter: &mut Align, ax: f32, ay: f32, az: f32) {
    let n = (ax * ax + ay * ay + az * az).sqrt();
    if n < 1.0e-6 {
        return;
    }
    let gx = ax / n;
    let gy = ay / n;
    let gz = az / n;

    // body->sensor with yaw fixed to 0, inferred from gravity direction only.
    // For g_s = R_sb * [0,0,1], the exact ZYX relations are:
    //   gx = sin(pitch), gy = -sin(roll)*cos(pitch), gz = cos(roll)*cos(pitch).
    let roll = (-gy).atan2(gz);
    let pitch = gx.clamp(-1.0, 1.0).asin();
    let q = quat_from_alg_rpy(roll, pitch, 0.0);
    filter.q_sb0 = q[0];
    filter.q_sb1 = q[1];
    filter.q_sb2 = q[2];
    filter.q_sb3 = q[3];
}

fn clear_motion_window(filter: &mut Align) {
    filter.gyro_sum = [0.0; 3];
    filter.gyro_count = 0;
    filter.accel_sum = [0.0; 3];
    filter.accel_count = 0;
}

fn build_forward_axis_measurement(
    filter: &Align,
    accel_meas: [f32; 3],
    speed_h: f32,
    prev_speed: f32,
    yaw_rate_b: f32,
    dt: f32,
) -> Option<[f32; 3]> {
    if !dt.is_finite() || dt <= 1.0e-3 {
        return None;
    }
    let long_acc = (speed_h - prev_speed) / dt;
    if !long_acc.is_finite() || long_acc.abs() < FORWARD_ACCEL_MIN_MPS2 {
        return None;
    }
    let lateral_acc = if yaw_rate_b.is_finite() {
        0.5 * (speed_h + prev_speed) * yaw_rate_b.abs()
    } else {
        0.0
    };
    if lateral_acc > long_acc.abs() / FORWARD_LATERAL_DOMINANCE_RATIO {
        return None;
    }

    let gravity_ref_s = if filter.gravity_ref_valid {
        filter.gravity_ref_s
    } else {
        let r_sb = quat_to_rotmat([filter.q_sb0, filter.q_sb1, filter.q_sb2, filter.q_sb3]);
        [
            r_sb[0][2] * GRAVITY_MPS2,
            r_sb[1][2] * GRAVITY_MPS2,
            r_sb[2][2] * GRAVITY_MPS2,
        ]
    };
    let z_s = vec3_normalize(gravity_ref_s)?;
    let accel_dyn = vec3_sub(accel_meas, gravity_ref_s);
    let horiz = vec3_sub(accel_dyn, vec3_scale(z_s, vec3_dot(accel_dyn, z_s)));
    let horiz_norm = vec3_norm(horiz);
    if !horiz_norm.is_finite() || horiz_norm < FORWARD_DYNAMIC_ACCEL_MIN_MPS2 {
        return None;
    }

    let mut forward_meas_s = vec3_scale(horiz, horiz_norm.recip());
    if long_acc < 0.0 {
        forward_meas_s = vec3_scale(forward_meas_s, -1.0);
    }
    Some(forward_meas_s)
}

fn apply_measurement_update(
    filter: &mut Align,
    residual: [f32; 3],
    h: [[f32; 3]; 3],
    r_diag: [f32; 3],
) {
    let p = filter.P;
    let hp = mat3_mul(h, p);
    let mut s = mat3_mul(hp, mat3_transpose(h));
    s[0][0] += r_diag[0].max(1.0e-8);
    s[1][1] += r_diag[1].max(1.0e-8);
    s[2][2] += r_diag[2].max(1.0e-8);
    let Some(s_inv) = mat3_inv(s) else {
        return;
    };
    let ph_t = mat3_mul(p, mat3_transpose(h));
    let k = mat3_mul(ph_t, s_inv);
    let delta = mat3_vec(k, residual);
    let dq = quat_from_small_angle(delta);
    let q_new = quat_normalize(quat_mul(
        dq,
        [filter.q_sb0, filter.q_sb1, filter.q_sb2, filter.q_sb3],
    ));
    filter.q_sb0 = q_new[0];
    filter.q_sb1 = q_new[1];
    filter.q_sb2 = q_new[2];
    filter.q_sb3 = q_new[3];

    let kh = mat3_mul(k, h);
    let i_kh = mat3_sub(mat3_identity(), kh);
    let mut p_new = mat3_mul(i_kh, p);
    p_new = mat3_symmetrize(p_new);
    p_new[0][0] = p_new[0][0].max(1.0e-10);
    p_new[1][1] = p_new[1][1].max(1.0e-10);
    p_new[2][2] = p_new[2][2].max(1.0e-10);
    filter.P = p_new;
}

fn body_forward_in_sensor(filter: &Align) -> [f32; 3] {
    let r_sb = quat_to_rotmat([filter.q_sb0, filter.q_sb1, filter.q_sb2, filter.q_sb3]);
    [r_sb[0][0], r_sb[1][0], r_sb[2][0]]
}

fn flip_yaw_branch(filter: &mut Align) {
    let qz_pi = [0.0, 0.0, 0.0, 1.0];
    let q_new = quat_normalize(quat_mul(
        [filter.q_sb0, filter.q_sb1, filter.q_sb2, filter.q_sb3],
        qz_pi,
    ));
    filter.q_sb0 = q_new[0];
    filter.q_sb1 = q_new[1];
    filter.q_sb2 = q_new[2];
    filter.q_sb3 = q_new[3];
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

fn quat_from_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let h = 0.5 * angle;
    let s = h.sin();
    quat_normalize([h.cos(), axis[0] * s, axis[1] * s, axis[2] * s])
}

fn quat_to_rotmat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let q = quat_normalize(q);
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
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

// ESF-ALG convention in this codebase: Rx * Ry * Rz composition.
fn quat_from_alg_rpy(roll: f32, pitch: f32, yaw: f32) -> [f32; 4] {
    let qx = quat_from_axis_angle([1.0, 0.0, 0.0], roll);
    let qy = quat_from_axis_angle([0.0, 1.0, 0.0], pitch);
    let qz = quat_from_axis_angle([0.0, 0.0, 1.0], yaw);
    quat_normalize(quat_mul(quat_mul(qx, qy), qz))
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

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_norm(v: [f32; 3]) -> f32 {
    vec3_dot(v, v).sqrt()
}

fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let n = vec3_norm(v);
    if !n.is_finite() || n < 1.0e-6 {
        return None;
    }
    Some(vec3_scale(v, n.recip()))
}

fn neg_skew(v: [f32; 3]) -> [[f32; 3]; 3] {
    [[0.0, v[2], -v[1]], [-v[2], 0.0, v[0]], [v[1], -v[0], 0.0]]
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
