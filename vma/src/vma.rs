#![allow(non_snake_case)]

use std::vec::Vec;

pub const N_STATES: usize = 3;
pub const GRAVITY_NED_MPS2: [f32; 3] = [0.0, 0.0, 9.80665];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MisalignNoise {
    pub q_theta_rw_var: f32, // [rad^2/s]
}

impl Default for MisalignNoise {
    fn default() -> Self {
        Self {
            q_theta_rw_var: 1.0e-8,
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
    // Body->Nav quaternion (w,x,y,z).
    pub q_nb0: f32,
    pub q_nb1: f32,
    pub q_nb2: f32,
    pub q_nb3: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct HistSample {
    imu: MisalignImuSample,
    att: MisalignAttitudeSample,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Vma {
    // Body->Sensor quaternion.
    pub q_sb0: f32,
    pub q_sb1: f32,
    pub q_sb2: f32,
    pub q_sb3: f32,
    pub P: [[f32; N_STATES]; N_STATES],
    pub noise: MisalignNoise,
    pub last_residual_n: [f32; 3],
    anchor_vel_n: Option<[f32; 3]>,
    hist: Vec<HistSample>,
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
            last_residual_n: [0.0, 0.0, 0.0],
            anchor_vel_n: None,
            hist: Vec::new(),
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
}

pub fn vma_q_sb(vma: &Vma) -> [f32; 4] {
    [vma.q_sb0, vma.q_sb1, vma.q_sb2, vma.q_sb3]
}

pub fn vma_reset_window(vma: &mut Vma) {
    vma.anchor_vel_n = None;
    vma.hist.clear();
    vma.last_residual_n = [0.0, 0.0, 0.0];
}

// Buffer one IMU sample and associated body attitude.
pub fn vma_predict(vma: &mut Vma, imu: &MisalignImuSample, att: &MisalignAttitudeSample) {
    if imu.dt <= 0.0 || !imu.dt.is_finite() {
        return;
    }
    vma.hist.push(HistSample {
        imu: *imu,
        att: *att,
    });

    let qvar = (vma.noise.q_theta_rw_var * imu.dt).max(0.0);
    vma.P[0][0] += qvar;
    vma.P[1][1] += qvar;
    vma.P[2][2] += qvar;
}

// GNSS velocity update. z is NED velocity [m/s], R is variance diag [m^2/s^2].
pub fn vma_fuse_velocity(vma: &mut Vma, vel_ned: [f32; 3], r_vel_diag: [f32; 3]) {
    let Some(anchor) = vma.anchor_vel_n else {
        vma.anchor_vel_n = Some(vel_ned);
        vma.hist.clear();
        return;
    };
    if vma.hist.is_empty() {
        vma.anchor_vel_n = Some(vel_ned);
        return;
    }

    let q_sb = [vma.q_sb0, vma.q_sb1, vma.q_sb2, vma.q_sb3];
    let v_pred = predict_velocity_n(anchor, q_sb, &vma.hist);
    let residual = [
        v_pred[0] - vel_ned[0],
        v_pred[1] - vel_ned[1],
        v_pred[2] - vel_ned[2],
    ];
    vma.last_residual_n = residual;

    let eps = 1.0e-4_f32;
    let mut H = [[0.0_f32; N_STATES]; 3];
    for i in 0..3 {
        let mut d = [0.0_f32; 3];
        d[i] = eps;
        let dq = quat_from_small_angle(d);
        let q_plus = quat_normalize(quat_mul(q_sb, dq));
        let v_plus = predict_velocity_n(anchor, q_plus, &vma.hist);
        H[0][i] = (v_plus[0] - v_pred[0]) / eps;
        H[1][i] = (v_plus[1] - v_pred[1]) / eps;
        H[2][i] = (v_plus[2] - v_pred[2]) / eps;
    }

    let p = vma.P;
    let hp = mat3_mul(H, p);
    let mut s = mat3_mul(hp, mat3_transpose(H));
    s[0][0] += r_vel_diag[0].max(1.0e-6);
    s[1][1] += r_vel_diag[1].max(1.0e-6);
    s[2][2] += r_vel_diag[2].max(1.0e-6);
    let Some(s_inv) = mat3_inv(s) else {
        vma.anchor_vel_n = Some(vel_ned);
        vma.hist.clear();
        return;
    };

    let ph_t = mat3_mul(p, mat3_transpose(H));
    let k = mat3_mul(ph_t, s_inv);

    let innov = [-residual[0], -residual[1], -residual[2]];
    let delta = mat3_vec(k, innov);

    let dq = quat_from_small_angle(delta);
    let q_new = quat_normalize(quat_mul(q_sb, dq));
    vma.q_sb0 = q_new[0];
    vma.q_sb1 = q_new[1];
    vma.q_sb2 = q_new[2];
    vma.q_sb3 = q_new[3];

    let kh = mat3_mul(k, H);
    let i_kh = mat3_sub(mat3_identity(), kh);
    let mut p_new = mat3_mul(i_kh, p);
    p_new = mat3_symmetrize(p_new);
    p_new[0][0] = p_new[0][0].max(1.0e-10);
    p_new[1][1] = p_new[1][1].max(1.0e-10);
    p_new[2][2] = p_new[2][2].max(1.0e-10);
    vma.P = p_new;

    vma.anchor_vel_n = Some(vel_ned);
    vma.hist.clear();
}

fn predict_velocity_n(anchor: [f32; 3], q_sb: [f32; 4], hist: &[HistSample]) -> [f32; 3] {
    let r_bs = quat_to_rotmat(quat_conj(q_sb));
    let mut v = anchor;
    for s in hist {
        let dt = s.imu.dt;
        let f_s = [s.imu.f_sx, s.imu.f_sy, s.imu.f_sz];
        let f_b = mat3_vec(r_bs, f_s);
        let q_nb = [s.att.q_nb0, s.att.q_nb1, s.att.q_nb2, s.att.q_nb3];
        let r_nb = quat_to_rotmat(quat_normalize(q_nb));
        let f_n = mat3_vec(r_nb, f_b);
        v[0] += (f_n[0] + GRAVITY_NED_MPS2[0]) * dt;
        v[1] += (f_n[1] + GRAVITY_NED_MPS2[1]) * dt;
        v[2] += (f_n[2] + GRAVITY_NED_MPS2[2]) * dt;
    }
    v
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
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
    let a = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
    if a < 1.0e-8 {
        quat_normalize([1.0, half[0], half[1], half[2]])
    } else {
        let c = a.cos();
        let s = a.sin() / a;
        [c, s * half[0], s * half[1], s * half[2]]
    }
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
    let mut out = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn mat3_vec(a: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn mat3_sub(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][j] - b[i][j];
        }
    }
    out
}

fn mat3_symmetrize(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = a;
    for i in 0..3 {
        for j in (i + 1)..3 {
            let s = 0.5 * (out[i][j] + out[j][i]);
            out[i][j] = s;
            out[j][i] = s;
        }
    }
    out
}

fn mat3_inv(a: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let m00 = a[0][0];
    let m01 = a[0][1];
    let m02 = a[0][2];
    let m10 = a[1][0];
    let m11 = a[1][1];
    let m12 = a[1][2];
    let m20 = a[2][0];
    let m21 = a[2][1];
    let m22 = a[2][2];

    let c00 = m11 * m22 - m12 * m21;
    let c01 = -(m10 * m22 - m12 * m20);
    let c02 = m10 * m21 - m11 * m20;
    let c10 = -(m01 * m22 - m02 * m21);
    let c11 = m00 * m22 - m02 * m20;
    let c12 = -(m00 * m21 - m01 * m20);
    let c20 = m01 * m12 - m02 * m11;
    let c21 = -(m00 * m12 - m02 * m10);
    let c22 = m00 * m11 - m01 * m10;

    let det = m00 * c00 + m01 * c01 + m02 * c02;
    if det.abs() < 1.0e-12 || !det.is_finite() {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ])
}
