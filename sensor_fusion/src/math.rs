//! Shared small math helpers for filter implementations.
//!
//! These functions intentionally stay lightweight and allocation-free so the
//! runtime filters can reuse them without pulling in a linear-algebra crate.

pub(crate) fn sqrt_f32(x: f32) -> f32 {
    libm::sqrtf(x)
}

pub(crate) fn sin_f32(x: f32) -> f32 {
    libm::sinf(x)
}

pub(crate) fn cos_f32(x: f32) -> f32 {
    libm::cosf(x)
}

pub(crate) fn atan2_f32(y: f32, x: f32) -> f32 {
    libm::atan2f(y, x)
}

pub(crate) fn asin_f32(x: f32) -> f32 {
    libm::asinf(x)
}

pub(crate) fn ceil_f32(x: f32) -> f32 {
    libm::ceilf(x)
}

pub(crate) fn sq_f32(x: f32) -> f32 {
    x * x
}

pub(crate) fn sqrt_f64(x: f64) -> f64 {
    libm::sqrt(x)
}

pub(crate) fn sin_f64(x: f64) -> f64 {
    libm::sin(x)
}

pub(crate) fn cos_f64(x: f64) -> f64 {
    libm::cos(x)
}

pub(crate) fn sin_cos_f64(x: f64) -> (f64, f64) {
    (libm::sin(x), libm::cos(x))
}

pub(crate) fn atan2_f64(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

pub(crate) fn sq_f64(x: f64) -> f64 {
    x * x
}

pub(crate) fn normalize_quat_f32(q: &mut [f32; 4]) {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2 <= 1.0e-12 {
        *q = [1.0, 0.0, 0.0, 0.0];
        return;
    }
    let inv_n = 1.0 / libm::sqrtf(n2);
    q[0] *= inv_n;
    q[1] *= inv_n;
    q[2] *= inv_n;
    q[3] *= inv_n;
}

pub(crate) fn normalized_quat_f32(mut q: [f32; 4]) -> [f32; 4] {
    normalize_quat_f32(&mut q);
    q
}

pub(crate) fn quat_multiply_f32(p: [f32; 4], q: [f32; 4]) -> [f32; 4] {
    [
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    ]
}

#[cfg(test)]
pub(crate) fn quat_conj_f32(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

pub(crate) fn quat_from_yaw_f32(yaw_rad: f32) -> [f32; 4] {
    let half = 0.5 * yaw_rad;
    [cos_f32(half), 0.0, 0.0, sin_f32(half)]
}

pub(crate) fn quat_to_dcm_f32(q_in: [f32; 4]) -> [[f32; 3]; 3] {
    let mut q = q_in;
    normalize_quat_f32(&mut q);
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

pub(crate) fn dcm_to_quat_f32(r: [[f32; 3]; 3]) -> [f32; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let mut q = if tr > 0.0 {
        let s = sqrt_f32(tr + 1.0) * 2.0;
        [
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = sqrt_f32(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0;
        [
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if r[1][1] > r[2][2] {
        let s = sqrt_f32(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0;
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = sqrt_f32(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0;
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        ]
    };
    normalize_quat_f32(&mut q);
    q
}

pub(crate) fn mat_vec3_f32(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub(crate) fn mat_mul3_f32(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ],
        [
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ],
    ]
}

pub(crate) fn transpose3_f32(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

pub(crate) fn skew3_f32(v: [f32; 3]) -> [[f32; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

pub(crate) fn vec_norm3_f32(v: [f32; 3]) -> f32 {
    libm::sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

pub(crate) fn add3_f32(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub3_f32(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn scale3_f32(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub(crate) fn cross3_f32(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn euler_to_quat_f32(roll: f32, pitch: f32, yaw: f32) -> [f32; 4] {
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

pub(crate) fn dcm_to_quat_f64(c: [[f64; 3]; 3]) -> [f64; 4] {
    let trace = c[0][0] + c[1][1] + c[2][2];
    let q = if trace > 0.0 {
        let s = sqrt_f64(trace + 1.0) * 2.0;
        [
            0.25 * s,
            (c[2][1] - c[1][2]) / s,
            (c[0][2] - c[2][0]) / s,
            (c[1][0] - c[0][1]) / s,
        ]
    } else if c[0][0] > c[1][1] && c[0][0] > c[2][2] {
        let s = sqrt_f64(1.0 + c[0][0] - c[1][1] - c[2][2]) * 2.0;
        [
            (c[2][1] - c[1][2]) / s,
            0.25 * s,
            (c[0][1] + c[1][0]) / s,
            (c[0][2] + c[2][0]) / s,
        ]
    } else if c[1][1] > c[2][2] {
        let s = sqrt_f64(1.0 + c[1][1] - c[0][0] - c[2][2]) * 2.0;
        [
            (c[0][2] - c[2][0]) / s,
            (c[0][1] + c[1][0]) / s,
            0.25 * s,
            (c[1][2] + c[2][1]) / s,
        ]
    } else {
        let s = sqrt_f64(1.0 + c[2][2] - c[0][0] - c[1][1]) * 2.0;
        [
            (c[1][0] - c[0][1]) / s,
            (c[0][2] + c[2][0]) / s,
            (c[1][2] + c[2][1]) / s,
            0.25 * s,
        ]
    };
    let n = sqrt_f64(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

pub(crate) fn quat_conj_f64(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

pub(crate) fn quat_mul_f64(p: [f64; 4], q: [f64; 4]) -> [f64; 4] {
    [
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    ]
}
