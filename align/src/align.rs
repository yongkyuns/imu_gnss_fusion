#![allow(non_snake_case)]

use crate::horizontal_heading::{
    HorizontalHeadingCueConfig, HorizontalHeadingCueFilter, HorizontalHeadingCueSample,
    HorizontalHeadingTrace,
};
use crate::yaw_pca::{YawPcaConfig, YawPcaInitializer, YawPcaSample};
use nalgebra::{SMatrix, SVector};

pub const ALIGN_N_STATES: usize = 3;
pub const GRAVITY_MPS2: f32 = 9.80665;

#[derive(Debug, Clone, Copy)]
pub struct AlignConfig {
    pub q_mount_std_rad: [f32; ALIGN_N_STATES],
    pub r_gravity_std_mps2: f32,
    pub r_turn_gyro_std_radps: f32,
    pub r_course_rate_std_radps: f32,
    pub r_lat_std_mps2: f32,
    pub r_long_std_mps2: f32,
    pub gravity_lpf_alpha: f32,
    pub long_lpf_alpha: f32,
    pub min_speed_mps: f32,
    pub min_turn_rate_radps: f32,
    pub min_lat_acc_mps2: f32,
    pub min_long_acc_mps2: f32,
    pub min_long_sign_stable_windows: usize,
    pub use_pca_yaw_seed: bool,
    pub pca_min_speed_mps: f32,
    pub pca_min_horiz_acc_mps2: f32,
    pub pca_min_windows: usize,
    pub pca_max_windows: usize,
    pub pca_min_anisotropy_ratio: f32,
    pub max_stationary_gyro_radps: f32,
    pub max_stationary_accel_norm_err_mps2: f32,
    pub use_gravity: bool,
    pub use_turn_gyro: bool,
    pub use_course_rate: bool,
    pub use_lateral_accel: bool,
    pub use_longitudinal_accel: bool,
}

impl Default for AlignConfig {
    fn default() -> Self {
        Self {
            q_mount_std_rad: [
                0.0003_f32.to_radians(),
                0.0003_f32.to_radians(),
                0.0003_f32.to_radians(),
            ],
            r_gravity_std_mps2: 1.28,
            r_turn_gyro_std_radps: 0.1_f32.to_radians(),
            r_course_rate_std_radps: 1.10_f32.to_radians(),
            r_lat_std_mps2: 0.02,
            r_long_std_mps2: 0.3,
            gravity_lpf_alpha: 0.08,
            long_lpf_alpha: 0.05,
            min_speed_mps: 3.0 / 3.6,
            min_turn_rate_radps: 2.0_f32.to_radians(),
            min_lat_acc_mps2: 0.10,
            min_long_acc_mps2: 0.18,
            min_long_sign_stable_windows: 2,
            use_pca_yaw_seed: true,
            pca_min_speed_mps: 5.0 / 3.6,
            pca_min_horiz_acc_mps2: 0.15,
            pca_min_windows: 4,
            pca_max_windows: 12,
            pca_min_anisotropy_ratio: 1.3,
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
pub struct AlignWindowSummary {
    pub dt: f32,
    pub mean_gyro_b: [f32; 3],
    pub mean_accel_b: [f32; 3],
    pub gnss_vel_prev_n: [f32; 3],
    pub gnss_vel_curr_n: [f32; 3],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AlignUpdateTrace {
    pub q_start: [f32; 4],
    pub after_pca_yaw_seed: Option<[f32; 4]>,
    pub after_gravity: Option<[f32; 4]>,
    pub after_turn_gyro: Option<[f32; 4]>,
    pub after_course_rate: Option<[f32; 4]>,
    pub after_lateral_accel: Option<[f32; 4]>,
    pub after_longitudinal_accel: Option<[f32; 4]>,
    pub longitudinal_trace: Option<HorizontalHeadingTrace>,
}

#[derive(Debug, Clone)]
pub struct Align {
    pub q_vb: [f32; 4],
    pub P: [[f32; ALIGN_N_STATES]; ALIGN_N_STATES],
    pub gravity_lp_b: [f32; 3],
    long_filter: HorizontalHeadingCueFilter,
    yaw_pca: YawPcaInitializer,
    pub cfg: AlignConfig,
}

impl Default for Align {
    fn default() -> Self {
        Self::new(AlignConfig::default())
    }
}

impl Align {
    pub fn new(cfg: AlignConfig) -> Self {
        Self {
            q_vb: [1.0, 0.0, 0.0, 0.0],
            P: diag3([
                20.0_f32.to_radians().powi(2),
                20.0_f32.to_radians().powi(2),
                60.0_f32.to_radians().powi(2),
            ]),
            gravity_lp_b: [0.0, 0.0, -GRAVITY_MPS2],
            long_filter: HorizontalHeadingCueFilter::new(),
            yaw_pca: YawPcaInitializer::new(),
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
        let mut x_v_in_b = vec3_sub(x_ref, vec3_scale(z_v_in_b, vec3_dot(z_v_in_b, x_ref)));
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
        let rpy = rot_to_euler_zyx(C_v_b);
        let dyaw = wrap_angle_rad(yaw_seed_rad - rpy[2]);
        let (s, c) = dyaw.sin_cos();
        let c_delta = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        // Apply the yaw seed as a vehicle-frame rotation on the right so the measured
        // gravity/down axis (third column of C_v_b) is preserved exactly.
        self.q_vb = quat_from_rotmat(mat3_mul(C_v_b, c_delta));
        self.P = diag3([
            6.0_f32.to_radians().powi(2),
            6.0_f32.to_radians().powi(2),
            20.0_f32.to_radians().powi(2),
        ]);
        self.gravity_lp_b = f_mean_b;
        self.long_filter.reset();
        self.yaw_pca.reset(self.cfg.use_pca_yaw_seed);
        Ok(())
    }

    pub fn predict(&mut self, dt: f32) {
        let dt = dt.max(1.0e-3);
        self.P[0][0] += self.cfg.q_mount_std_rad[0].powi(2) * dt;
        self.P[1][1] += self.cfg.q_mount_std_rad[1].powi(2) * dt;
        self.P[2][2] += self.cfg.q_mount_std_rad[2].powi(2) * dt;
    }

    pub fn update_window(&mut self, window: &AlignWindowSummary) -> f32 {
        self.update_window_with_trace(window).0
    }

    pub fn update_window_with_trace(
        &mut self,
        window: &AlignWindowSummary,
    ) -> (f32, AlignUpdateTrace) {
        self.predict(window.dt);
        let mut score = 0.0_f32;
        let mut trace = AlignUpdateTrace {
            q_start: self.q_vb,
            ..AlignUpdateTrace::default()
        };

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
        let horiz_accel_b = if let Some(g_hat_b) = vec3_normalize(self.gravity_lp_b) {
            vec3_sub(
                window.mean_accel_b,
                vec3_scale(g_hat_b, vec3_dot(window.mean_accel_b, g_hat_b)),
            )
        } else {
            window.mean_accel_b
        };
        let horiz_obs = align_obs(self.q_vb, window.mean_gyro_b, horiz_accel_b);
        let stationary = gyro_norm <= self.cfg.max_stationary_gyro_radps
            && (accel_norm - GRAVITY_MPS2).abs() <= self.cfg.max_stationary_accel_norm_err_mps2
            && speed_mid < 0.5;
        let turn_valid = speed_mid > self.cfg.min_speed_mps
            && course_rate.abs() > self.cfg.min_turn_rate_radps
            && a_lat.abs() > self.cfg.min_lat_acc_mps2;
        let horiz_gnss_norm = (a_long * a_long + a_lat * a_lat).sqrt();
        let long_valid = speed_mid > self.cfg.min_speed_mps
            && a_long.abs() > self.cfg.min_long_acc_mps2
            // Use the horizontal-vector angle only when motion is still predominantly
            // longitudinal. In lateral-dominant motion, the instantaneous accel-vector
            // heading is not the forward cue we want from this update.
            && a_lat.abs() < (0.5_f32).max(0.6 * a_long.abs())
            && horiz_gnss_norm > self.cfg.min_long_acc_mps2;
        let heading_updates_enabled = if self.yaw_pca.is_active() {
            let pca_cfg = YawPcaConfig {
                enabled: self.cfg.use_pca_yaw_seed,
                min_speed_mps: self.cfg.pca_min_speed_mps,
                min_horiz_acc_mps2: self.cfg.pca_min_horiz_acc_mps2,
                min_windows: self.cfg.pca_min_windows,
                max_windows: self.cfg.pca_max_windows,
                min_anisotropy_ratio: self.cfg.pca_min_anisotropy_ratio,
            };
            if let Some(dpsi) = self.yaw_pca.update(
                pca_cfg,
                YawPcaSample {
                    speed_mps: speed_mid,
                    horiz_accel_v: [horiz_obs[3], horiz_obs[4]],
                    gnss_long_mps2: a_long,
                },
            ) {
                self.inject_vehicle_yaw(dpsi);
                self.long_filter.reset();
                trace.after_pca_yaw_seed = Some(self.q_vb);
            }
            !self.yaw_pca.is_active()
        } else {
            true
        };

        if stationary {
            let alpha = self.cfg.gravity_lpf_alpha;
            self.gravity_lp_b = vec3_add(
                vec3_scale(self.gravity_lp_b, 1.0 - alpha),
                vec3_scale(window.mean_accel_b, alpha),
            );
        }

        if self.cfg.use_gravity && stationary {
            score += self.apply_update2(
                [0.0, 0.0],
                [3, 4],
                self.gravity_lp_b,
                window.mean_gyro_b,
                [self.cfg.r_gravity_std_mps2.powi(2); 2],
            );
            score += self.apply_update1(
                -vec3_norm(self.gravity_lp_b),
                5,
                self.gravity_lp_b,
                window.mean_gyro_b,
                self.cfg.r_gravity_std_mps2.powi(2),
            );
            trace.after_gravity = Some(self.q_vb);
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
                trace.after_turn_gyro = Some(self.q_vb);
            }
            if heading_updates_enabled && self.cfg.use_course_rate {
                score += self.apply_update1(
                    course_rate,
                    2,
                    window.mean_accel_b,
                    window.mean_gyro_b,
                    self.cfg.r_course_rate_std_radps.powi(2),
                );
                trace.after_course_rate = Some(self.q_vb);
            }
            if heading_updates_enabled && self.cfg.use_lateral_accel {
                score += self.apply_vehicle_yaw_scalar(
                    a_lat,
                    horiz_obs[4],
                    -horiz_obs[3],
                    self.cfg.r_lat_std_mps2.powi(2),
                );
                trace.after_lateral_accel = Some(self.q_vb);
            }
        }

        if heading_updates_enabled && self.cfg.use_longitudinal_accel {
            let (cue, long_trace) = self.long_filter.update_with_trace(
                HorizontalHeadingCueConfig {
                    alpha: self.cfg.long_lpf_alpha,
                    min_abs_horiz_mps2: self.cfg.min_long_acc_mps2,
                    min_stable_windows: self.cfg.min_long_sign_stable_windows,
                    max_lat_to_long_ratio: 0.8,
                    min_abs_lat_guard_mps2: 0.35,
                },
                HorizontalHeadingCueSample {
                    gnss_horiz_mps2: [a_long, a_lat],
                    imu_horiz_mps2: [horiz_obs[3], horiz_obs[4]],
                    base_valid: long_valid,
                },
            );
            trace.longitudinal_trace = Some(long_trace);
            if let Some(cue) = cue {
                score += self
                    .apply_vehicle_yaw_angle(cue.angle_err_rad, self.cfg.r_long_std_mps2.powi(2));
                trace.after_longitudinal_accel = Some(self.q_vb);
            }
        }

        (score, trace)
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
        self.apply_update1_masked(z, obs_idx, accel_b, gyro_b, r_var, [true, true, true])
    }

    fn apply_update1_masked(
        &mut self,
        z: f32,
        obs_idx: usize,
        accel_b: [f32; 3],
        gyro_b: [f32; 3],
        r_var: f32,
        state_mask: [bool; 3],
    ) -> f32 {
        let obs = align_obs(self.q_vb, gyro_b, accel_b);
        let H_full = align_obs_jacobian(self.q_vb, gyro_b, accel_b);
        let H = SMatrix::<f32, 1, 3>::from_row_slice(&[
            if state_mask[0] {
                H_full[obs_idx][0]
            } else {
                0.0
            },
            if state_mask[1] {
                H_full[obs_idx][1]
            } else {
                0.0
            },
            if state_mask[2] {
                H_full[obs_idx][2]
            } else {
                0.0
            },
        ]);
        let y = SVector::<f32, 1>::from_row_slice(&[z - obs[obs_idx]]);
        let P = mat3_to_smatrix(self.P);
        let S = H * P * H.transpose() + SMatrix::<f32, 1, 1>::from_diagonal_element(r_var);
        let S_inv = S
            .try_inverse()
            .unwrap_or_else(SMatrix::<f32, 1, 1>::identity);
        let K = P * H.transpose() * S_inv;
        let dtheta = K * y;
        self.inject_small_angle([dtheta[0], dtheta[1], dtheta[2]]);
        let I = SMatrix::<f32, 3, 3>::identity();
        let P_new = (I - K * H) * P;
        self.P = smatrix_to_mat3(symmetrize3(P_new));
        (y.transpose() * S_inv * y)[0]
    }

    fn apply_vehicle_yaw_scalar(&mut self, z: f32, h: f32, h_yaw: f32, r_var: f32) -> f32 {
        let y = z - h;
        let pzz = self.P[2][2].max(0.0);
        let s = h_yaw * h_yaw * pzz + r_var.max(1.0e-9);
        let k = if s > 1.0e-9 { pzz * h_yaw / s } else { 0.0 };
        let dpsi = k * y;
        self.inject_vehicle_yaw(dpsi);
        self.P[2][2] = ((1.0 - k * h_yaw) * pzz).max(0.0);
        self.P[0][2] = 0.0;
        self.P[2][0] = 0.0;
        self.P[1][2] = 0.0;
        self.P[2][1] = 0.0;
        y * y / s
    }

    fn apply_vehicle_yaw_angle(&mut self, angle_err_rad: f32, r_var: f32) -> f32 {
        let pzz = self.P[2][2].max(0.0);
        let s = pzz + r_var.max(1.0e-9);
        let k = if s > 1.0e-9 { pzz / s } else { 0.0 };
        let dpsi = -k * angle_err_rad;
        self.inject_vehicle_yaw(dpsi);
        self.P[2][2] = ((1.0 - k) * pzz).max(0.0);
        self.P[0][2] = 0.0;
        self.P[2][0] = 0.0;
        self.P[1][2] = 0.0;
        self.P[2][1] = 0.0;
        angle_err_rad * angle_err_rad / s
    }

    fn apply_update2(
        &mut self,
        z: [f32; 2],
        obs_idx: [usize; 2],
        accel_b: [f32; 3],
        gyro_b: [f32; 3],
        r_var: [f32; 2],
    ) -> f32 {
        let obs = align_obs(self.q_vb, gyro_b, accel_b);
        let H_full = align_obs_jacobian(self.q_vb, gyro_b, accel_b);
        let H = SMatrix::<f32, 2, 3>::from_row_slice(&[
            H_full[obs_idx[0]][0],
            H_full[obs_idx[0]][1],
            H_full[obs_idx[0]][2],
            H_full[obs_idx[1]][0],
            H_full[obs_idx[1]][1],
            H_full[obs_idx[1]][2],
        ]);
        let y =
            SVector::<f32, 2>::from_row_slice(&[z[0] - obs[obs_idx[0]], z[1] - obs[obs_idx[1]]]);
        let P = mat3_to_smatrix(self.P);
        let R = SMatrix::<f32, 2, 2>::from_diagonal(&SVector::<f32, 2>::from_row_slice(&r_var));
        let S = H * P * H.transpose() + R;
        let S_inv = S
            .try_inverse()
            .unwrap_or_else(SMatrix::<f32, 2, 2>::identity);
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

    fn inject_vehicle_yaw(&mut self, dpsi: f32) {
        self.q_vb = quat_normalize(quat_mul(self.q_vb, quat_from_small_angle([0.0, 0.0, dpsi])));
    }
}

fn align_obs(q_vb: [f32; 4], gyro_b: [f32; 3], accel_b: [f32; 3]) -> [f32; 6] {
    let c_bv = transpose3x3(quat_to_rotmat(q_vb));
    let gyro_v = mat3_vec(c_bv, gyro_b);
    let accel_v = mat3_vec(c_bv, accel_b);
    [
        gyro_v[0], gyro_v[1], gyro_v[2], accel_v[0], accel_v[1], accel_v[2],
    ]
}

fn align_obs_jacobian(q_vb: [f32; 4], gyro_b: [f32; 3], accel_b: [f32; 3]) -> [[f32; 3]; 6] {
    let c_bv = transpose3x3(quat_to_rotmat(q_vb));
    let h_gyro = mat3_mul(c_bv, skew3(gyro_b));
    let h_accel = mat3_mul(c_bv, skew3(accel_b));
    [
        h_gyro[0], h_gyro[1], h_gyro[2], h_accel[0], h_accel[1], h_accel[2],
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
    quat_normalize([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]])
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

fn quat_from_rotmat(c: [[f32; 3]; 3]) -> [f32; 4] {
    let trace = c[0][0] + c[1][1] + c[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (c[2][1] - c[1][2]) / s,
            (c[0][2] - c[2][0]) / s,
            (c[1][0] - c[0][1]) / s,
        ]
    } else if c[0][0] > c[1][1] && c[0][0] > c[2][2] {
        let s = (1.0 + c[0][0] - c[1][1] - c[2][2]).sqrt() * 2.0;
        [
            (c[2][1] - c[1][2]) / s,
            0.25 * s,
            (c[0][1] + c[1][0]) / s,
            (c[0][2] + c[2][0]) / s,
        ]
    } else if c[1][1] > c[2][2] {
        let s = (1.0 + c[1][1] - c[0][0] - c[2][2]).sqrt() * 2.0;
        [
            (c[0][2] - c[2][0]) / s,
            (c[0][1] + c[1][0]) / s,
            0.25 * s,
            (c[1][2] + c[2][1]) / s,
        ]
    } else {
        let s = (1.0 + c[2][2] - c[0][0] - c[1][1]).sqrt() * 2.0;
        [
            (c[1][0] - c[0][1]) / s,
            (c[0][2] + c[2][0]) / s,
            (c[1][2] + c[2][1]) / s,
            0.25 * s,
        ]
    };
    quat_normalize(q)
}

fn transpose3x3(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
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

fn skew3(v: [f32; 3]) -> [[f32; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0_f32; 3]; 3];
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
