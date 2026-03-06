#![allow(non_snake_case)]

use nalgebra::{SMatrix, SVector};

pub const COARSE_N_STATES: usize = 3;
pub const GRAVITY_MPS2: f32 = 9.80665;

#[derive(Debug, Clone, Copy)]
pub struct CoarseAlignConfig {
    pub q_mount_std_rad: [f32; COARSE_N_STATES],
    pub r_gravity_std_mps2: f32,
    pub r_turn_gyro_std_radps: f32,
    pub r_course_rate_std_radps: f32,
    pub r_lat_std_mps2: f32,
    pub r_long_std_mps2: f32,
    pub gravity_lpf_alpha: f32,
    pub min_speed_mps: f32,
    pub min_turn_rate_radps: f32,
    pub min_lat_acc_mps2: f32,
    pub min_long_acc_mps2: f32,
    pub max_stationary_gyro_radps: f32,
    pub max_stationary_accel_norm_err_mps2: f32,
    pub use_gravity: bool,
    pub use_turn_gyro: bool,
    pub use_course_rate: bool,
    pub use_lateral_accel: bool,
    pub use_longitudinal_accel: bool,
}

impl Default for CoarseAlignConfig {
    fn default() -> Self {
        Self {
            q_mount_std_rad: [
                0.01_f32.to_radians(),
                0.01_f32.to_radians(),
                0.02_f32.to_radians(),
            ],
            r_gravity_std_mps2: 0.08,
            r_turn_gyro_std_radps: 0.2_f32.to_radians(),
            r_course_rate_std_radps: 0.35_f32.to_radians(),
            r_lat_std_mps2: 0.10,
            r_long_std_mps2: 0.10,
            gravity_lpf_alpha: 0.08,
            min_speed_mps: 4.0,
            min_turn_rate_radps: 3.0_f32.to_radians(),
            min_lat_acc_mps2: 0.35,
            min_long_acc_mps2: 0.25,
            max_stationary_gyro_radps: 0.8_f32.to_radians(),
            max_stationary_accel_norm_err_mps2: 0.2,
            use_gravity: true,
            use_turn_gyro: true,
            use_course_rate: true,
            use_lateral_accel: true,
            use_longitudinal_accel: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CoarseWindowSummary {
    pub dt: f32,
    pub mean_gyro_b: [f32; 3],
    pub mean_accel_b: [f32; 3],
    pub gnss_vel_prev_n: [f32; 3],
    pub gnss_vel_curr_n: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct CoarseAlignMEKF {
    pub q_vb: [f32; 4],
    pub P: [[f32; COARSE_N_STATES]; COARSE_N_STATES],
    pub gravity_lp_b: [f32; 3],
    pub cfg: CoarseAlignConfig,
}

impl Default for CoarseAlignMEKF {
    fn default() -> Self {
        Self::new(CoarseAlignConfig::default())
    }
}

impl CoarseAlignMEKF {
    pub fn new(cfg: CoarseAlignConfig) -> Self {
        Self {
            q_vb: [1.0, 0.0, 0.0, 0.0],
            P: diag3([
                20.0_f32.to_radians().powi(2),
                20.0_f32.to_radians().powi(2),
                60.0_f32.to_radians().powi(2),
            ]),
            gravity_lp_b: [0.0, 0.0, -GRAVITY_MPS2],
            cfg,
        }
    }

    pub fn initialize_from_stationary(
        &mut self,
        accel_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
    ) -> Result<(), &'static str> {
        if accel_samples_b.is_empty() {
            return Err("stationary initialization requires samples");
        }
        let mut f_mean_b = [0.0_f32; 3];
        for sample in accel_samples_b {
            f_mean_b = vec3_add(f_mean_b, *sample);
        }
        let inv_n = 1.0 / (accel_samples_b.len() as f32);
        f_mean_b = vec3_scale(f_mean_b, inv_n);
        let n = vec3_norm(f_mean_b);
        if n < 1.0e-6 {
            return Err("stationary initialization requires nonzero accel mean");
        }

        let z_v_in_b = vec3_scale(f_mean_b, -1.0 / n);
        let mut x_ref = [1.0, 0.0, 0.0];
        let mut x_v_in_b =
            vec3_sub(x_ref, vec3_scale(z_v_in_b, vec3_dot(z_v_in_b, x_ref)));
        if vec3_norm(x_v_in_b) < 1.0e-6 {
            x_ref = [0.0, 1.0, 0.0];
            x_v_in_b = vec3_sub(x_ref, vec3_scale(z_v_in_b, vec3_dot(z_v_in_b, x_ref)));
        }
        x_v_in_b = vec3_normalize(x_v_in_b).ok_or("failed to initialize x axis")?;
        let mut y_v_in_b = vec3_cross(z_v_in_b, x_v_in_b);
        y_v_in_b = vec3_normalize(y_v_in_b).ok_or("failed to initialize y axis")?;
        x_v_in_b = vec3_cross(y_v_in_b, z_v_in_b);
        let C_v_b = [
            [x_v_in_b[0], y_v_in_b[0], z_v_in_b[0]],
            [x_v_in_b[1], y_v_in_b[1], z_v_in_b[1]],
            [x_v_in_b[2], y_v_in_b[2], z_v_in_b[2]],
        ];
        let mut rpy = rot_to_euler_zyx(C_v_b);
        rpy[2] = wrap_angle_rad(yaw_seed_rad);
        self.q_vb = quat_from_euler_zyx(rpy[0], rpy[1], rpy[2]);
        self.P = diag3([
            6.0_f32.to_radians().powi(2),
            6.0_f32.to_radians().powi(2),
            20.0_f32.to_radians().powi(2),
        ]);
        self.gravity_lp_b = f_mean_b;
        Ok(())
    }

    pub fn predict(&mut self, dt: f32) {
        let dt = dt.max(1.0e-3);
        self.P[0][0] += self.cfg.q_mount_std_rad[0].powi(2) * dt;
        self.P[1][1] += self.cfg.q_mount_std_rad[1].powi(2) * dt;
        self.P[2][2] += self.cfg.q_mount_std_rad[2].powi(2) * dt;
    }

    pub fn update_window(&mut self, window: &CoarseWindowSummary) -> f32 {
        self.predict(window.dt);
        let mut score = 0.0_f32;

        let alpha = self.cfg.gravity_lpf_alpha;
        self.gravity_lp_b = vec3_add(
            vec3_scale(self.gravity_lp_b, 1.0 - alpha),
            vec3_scale(window.mean_accel_b, alpha),
        );

        let v_prev = window.gnss_vel_prev_n;
        let v_curr = window.gnss_vel_curr_n;
        let speed_prev = vec2_norm([v_prev[0], v_prev[1]]);
        let speed_curr = vec2_norm([v_curr[0], v_curr[1]]);
        let speed_mid = 0.5 * (speed_prev + speed_curr);

        let course_prev = v_prev[1].atan2(v_prev[0]);
        let course_curr = v_curr[1].atan2(v_curr[0]);
        let course_rate = wrap_angle_rad(course_curr - course_prev) / window.dt.max(1.0e-3);

        let a_n = vec3_scale(vec3_sub(v_curr, v_prev), 1.0 / window.dt.max(1.0e-3));
        let v_mid_h = [0.5 * (v_prev[0] + v_curr[0]), 0.5 * (v_prev[1] + v_curr[1])];
        let t_hat = vec2_normalize(v_mid_h);
        let lat_hat = t_hat.map(|t| [-t[1], t[0]]);
        let a_long = t_hat.map(|t| t[0] * a_n[0] + t[1] * a_n[1]).unwrap_or(0.0);
        let a_lat = lat_hat
            .map(|l| l[0] * a_n[0] + l[1] * a_n[1])
            .unwrap_or(0.0);

        let gyro_norm = vec3_norm(window.mean_gyro_b);
        let accel_norm = vec3_norm(window.mean_accel_b);
        let stationary = gyro_norm <= self.cfg.max_stationary_gyro_radps
            && (accel_norm - GRAVITY_MPS2).abs() <= self.cfg.max_stationary_accel_norm_err_mps2
            && speed_mid < 0.5;
        let turn_valid = speed_mid > self.cfg.min_speed_mps
            && course_rate.abs() > self.cfg.min_turn_rate_radps
            && a_lat.abs() > self.cfg.min_lat_acc_mps2;
        let long_valid = speed_mid > self.cfg.min_speed_mps
            && a_long.abs() > self.cfg.min_long_acc_mps2
            && a_lat.abs() < (0.5_f32).max(0.6 * a_long.abs());

        if self.cfg.use_gravity && stationary {
            score += self.apply_update2(
                [0.0, 0.0],
                [3, 4],
                self.gravity_lp_b,
                window.mean_gyro_b,
                [self.cfg.r_gravity_std_mps2.powi(2); 2],
            );
        }

        if turn_valid {
            if self.cfg.use_turn_gyro {
                score += self.apply_update2(
                    [0.0, 0.0],
                    [0, 1],
                    window.mean_accel_b,
                    window.mean_gyro_b,
                    [self.cfg.r_turn_gyro_std_radps.powi(2); 2],
                );
            }
            if self.cfg.use_course_rate {
                score += self.apply_update1(
                    course_rate,
                    2,
                    window.mean_accel_b,
                    window.mean_gyro_b,
                    self.cfg.r_course_rate_std_radps.powi(2),
                );
            }
            if self.cfg.use_lateral_accel {
                score += self.apply_update1(
                    a_lat,
                    4,
                    window.mean_accel_b,
                    window.mean_gyro_b,
                    self.cfg.r_lat_std_mps2.powi(2),
                );
            }
        }

        if long_valid && self.cfg.use_longitudinal_accel {
            score += self.apply_update1(
                a_long,
                3,
                window.mean_accel_b,
                window.mean_gyro_b,
                self.cfg.r_long_std_mps2.powi(2),
            );
        }

        score
    }

    pub fn mount_angles_rad(&self) -> [f32; 3] {
        rot_to_euler_zyx(quat_to_rotmat(self.q_vb))
    }

    pub fn mount_angles_deg(&self) -> [f32; 3] {
        let r = self.mount_angles_rad();
        [r[0].to_degrees(), r[1].to_degrees(), r[2].to_degrees()]
    }

    pub fn sigma_deg(&self) -> [f32; 3] {
        [
            self.P[0][0].max(0.0).sqrt().to_degrees(),
            self.P[1][1].max(0.0).sqrt().to_degrees(),
            self.P[2][2].max(0.0).sqrt().to_degrees(),
        ]
    }

    fn apply_update1(
        &mut self,
        z: f32,
        obs_idx: usize,
        accel_b: [f32; 3],
        gyro_b: [f32; 3],
        r_var: f32,
    ) -> f32 {
        let obs = coarse_obs(self.q_vb, gyro_b, accel_b);
        let H_full = coarse_obs_jacobian(self.q_vb, gyro_b, accel_b);
        let H = SMatrix::<f32, 1, 3>::from_row_slice(&[
            H_full[obs_idx][0],
            H_full[obs_idx][1],
            H_full[obs_idx][2],
        ]);
        let y = SVector::<f32, 1>::from_row_slice(&[z - obs[obs_idx]]);
        let P = mat3_to_smatrix(self.P);
        let S = H * P * H.transpose() + SMatrix::<f32, 1, 1>::from_diagonal_element(r_var);
        let S_inv = S.try_inverse().unwrap_or_else(SMatrix::<f32, 1, 1>::identity);
        let K = P * H.transpose() * S_inv;
        let dtheta = K * y;
        self.inject_small_angle([dtheta[0], dtheta[1], dtheta[2]]);
        let I = SMatrix::<f32, 3, 3>::identity();
        let P_new = (I - K * H) * P;
        self.P = smatrix_to_mat3(symmetrize3(P_new));
        (y.transpose() * S_inv * y)[0]
    }

    fn apply_update2(
        &mut self,
        z: [f32; 2],
        obs_idx: [usize; 2],
        accel_b: [f32; 3],
        gyro_b: [f32; 3],
        r_var: [f32; 2],
    ) -> f32 {
        let obs = coarse_obs(self.q_vb, gyro_b, accel_b);
        let H_full = coarse_obs_jacobian(self.q_vb, gyro_b, accel_b);
        let H = SMatrix::<f32, 2, 3>::from_row_slice(&[
            H_full[obs_idx[0]][0],
            H_full[obs_idx[0]][1],
            H_full[obs_idx[0]][2],
            H_full[obs_idx[1]][0],
            H_full[obs_idx[1]][1],
            H_full[obs_idx[1]][2],
        ]);
        let y = SVector::<f32, 2>::from_row_slice(&[
            z[0] - obs[obs_idx[0]],
            z[1] - obs[obs_idx[1]],
        ]);
        let P = mat3_to_smatrix(self.P);
        let R = SMatrix::<f32, 2, 2>::from_diagonal(&SVector::<f32, 2>::from_row_slice(&r_var));
        let S = H * P * H.transpose() + R;
        let S_inv = S.try_inverse().unwrap_or_else(SMatrix::<f32, 2, 2>::identity);
        let K = P * H.transpose() * S_inv;
        let dtheta = K * y;
        self.inject_small_angle([dtheta[0], dtheta[1], dtheta[2]]);
        let I = SMatrix::<f32, 3, 3>::identity();
        let P_new = (I - K * H) * P;
        self.P = smatrix_to_mat3(symmetrize3(P_new));
        (y.transpose() * S_inv * y)[0]
    }

    fn inject_small_angle(&mut self, dtheta: [f32; 3]) {
        self.q_vb = quat_normalize(quat_mul(quat_from_small_angle(dtheta), self.q_vb));
    }
}

fn coarse_obs(q_vb: [f32; 4], gyro_b: [f32; 3], accel_b: [f32; 3]) -> [f32; 6] {
    let (q_vb0, q_vb1, q_vb2, q_vb3) = (q_vb[0], q_vb[1], q_vb[2], q_vb[3]);
    let (gyro_bx, gyro_by, gyro_bz) = (gyro_b[0], gyro_b[1], gyro_b[2]);
    let (accel_bx, accel_by, accel_bz) = (accel_b[0], accel_b[1], accel_b[2]);
    let mut coarse_obs = [0.0_f32; 6];
    include!("align_generated/coarse_mekf_obs_generated.rs");
    coarse_obs
}

fn coarse_obs_jacobian(q_vb: [f32; 4], gyro_b: [f32; 3], accel_b: [f32; 3]) -> [[f32; 3]; 6] {
    let (q_vb0, q_vb1, q_vb2, q_vb3) = (q_vb[0], q_vb[1], q_vb[2], q_vb[3]);
    let (gyro_bx, gyro_by, gyro_bz) = (gyro_b[0], gyro_b[1], gyro_b[2]);
    let (accel_bx, accel_by, accel_bz) = (accel_b[0], accel_b[1], accel_b[2]);
    let mut H_coarse = [[0.0_f32; 3]; 6];
    include!("align_generated/coarse_mekf_obs_jacobian_generated.rs");
    H_coarse
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
    quat_normalize([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]])
}

fn quat_from_euler_zyx(roll: f32, pitch: f32, yaw: f32) -> [f32; 4] {
    let (sr, cr) = (0.5 * roll).sin_cos();
    let (sp, cp) = (0.5 * pitch).sin_cos();
    let (sy, cy) = (0.5 * yaw).sin_cos();
    quat_normalize([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
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

fn rot_to_euler_zyx(C: [[f32; 3]; 3]) -> [f32; 3] {
    let pitch = (-C[2][0]).clamp(-1.0, 1.0).asin();
    let roll = C[2][1].atan2(C[2][2]);
    let yaw = wrap_angle_rad(C[1][0].atan2(C[0][0]));
    [roll, pitch, yaw]
}

fn wrap_angle_rad(x: f32) -> f32 {
    let two_pi = 2.0 * core::f32::consts::PI;
    (x + core::f32::consts::PI).rem_euclid(two_pi) - core::f32::consts::PI
}

fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_norm(v: [f32; 3]) -> f32 {
    vec3_dot(v, v).sqrt()
}

fn vec3_normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let n = vec3_norm(v);
    if !n.is_finite() || n <= 1.0e-8 {
        return None;
    }
    Some(vec3_scale(v, 1.0 / n))
}

fn vec2_norm(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn vec2_normalize(v: [f32; 2]) -> Option<[f32; 2]> {
    let n = vec2_norm(v);
    if !n.is_finite() || n <= 1.0e-8 {
        return None;
    }
    Some([v[0] / n, v[1] / n])
}

fn diag3(d: [f32; 3]) -> [[f32; 3]; 3] {
    [[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]]
}

fn mat3_to_smatrix(a: [[f32; 3]; 3]) -> SMatrix<f32, 3, 3> {
    SMatrix::<f32, 3, 3>::from_row_slice(&[
        a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
    ])
}

fn smatrix_to_mat3(a: SMatrix<f32, 3, 3>) -> [[f32; 3]; 3] {
    [
        [a[(0, 0)], a[(0, 1)], a[(0, 2)]],
        [a[(1, 0)], a[(1, 1)], a[(1, 2)]],
        [a[(2, 0)], a[(2, 1)], a[(2, 2)]],
    ]
}

fn symmetrize3(a: SMatrix<f32, 3, 3>) -> SMatrix<f32, 3, 3> {
    0.5 * (a + a.transpose())
}
