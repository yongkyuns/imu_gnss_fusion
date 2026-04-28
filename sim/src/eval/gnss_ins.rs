use clap::ValueEnum;

use crate::datasets::gnss_ins_sim::GnssSample;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum SignalSource {
    Ref,
    Meas,
}

impl SignalSource {
    pub fn use_ref_signals(self) -> bool {
        matches!(self, Self::Ref)
    }
}

pub fn quat_from_rpy_deg(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> [f64; 4] {
    let (sr, cr) = (0.5 * roll_deg.to_radians()).sin_cos();
    let (sp, cp) = (0.5 * pitch_deg.to_radians()).sin_cos();
    let (sy, cy) = (0.5 * yaw_deg.to_radians()).sin_cos();
    quat_normalize([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
}

pub fn quat_from_rpy_alg_deg(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> [f64; 4] {
    quat_from_rpy_deg(roll_deg, pitch_deg, yaw_deg)
}

pub fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n <= 1.0e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

pub fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

pub fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    quat_normalize([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])
}

pub fn quat_rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let r = quat_to_rotmat(q);
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

pub fn quat_angle_deg(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
        .abs()
        .clamp(0.0, 1.0);
    2.0 * dot.acos().to_degrees()
}

pub fn axis_angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if na <= 1.0e-12 || nb <= 1.0e-12 {
        return f64::NAN;
    }
    let dot = ((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)).clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}

pub fn quat_axis_angle_deg(q_est: [f64; 4], q_ref: [f64; 4], axis: [f64; 3]) -> f64 {
    axis_angle_deg(quat_rotate(q_est, axis), quat_rotate(q_ref, axis))
}

pub fn wrap_deg180(mut deg: f64) -> f64 {
    while deg > 180.0 {
        deg -= 360.0;
    }
    while deg <= -180.0 {
        deg += 360.0;
    }
    deg
}

pub fn wrap_rad_pi(mut rad: f64) -> f64 {
    while rad > std::f64::consts::PI {
        rad -= 2.0 * std::f64::consts::PI;
    }
    while rad <= -std::f64::consts::PI {
        rad += 2.0 * std::f64::consts::PI;
    }
    rad
}

pub fn horiz_speed(v_ned_mps: [f64; 3]) -> f64 {
    (v_ned_mps[0] * v_ned_mps[0] + v_ned_mps[1] * v_ned_mps[1]).sqrt()
}

pub fn course_rate_deg(prev: GnssSample, curr: GnssSample) -> f64 {
    let dt = (curr.t_s - prev.t_s).max(1.0e-6);
    let course_prev = prev.vel_ned_mps[1].atan2(prev.vel_ned_mps[0]);
    let course_curr = curr.vel_ned_mps[1].atan2(curr.vel_ned_mps[0]);
    wrap_deg180((course_curr - course_prev).to_degrees()) / dt
}

pub fn as_q64(q: [f32; 4]) -> [f64; 4] {
    quat_normalize([q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64])
}

fn quat_to_rotmat(q: [f64; 4]) -> [[f64; 3]; 3] {
    let q = quat_normalize(q);
    let q0 = q[0];
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];
    [
        [
            1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2,
        ],
    ]
}
