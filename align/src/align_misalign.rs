#![allow(non_snake_case)]

pub const ALIGN_MISALIGN_STATES: usize = 3;

#[derive(Debug, Clone, Copy)]
pub struct AlignMisalignConfig {
    pub q_mount_std_rad: [f32; ALIGN_MISALIGN_STATES],
    pub r_nhc_std_mps: f32,
    pub r_planar_gyro_std_radps: f32,
    pub r_yaw_cue_std_rad: f32,
    pub r_yaw_prior_std_rad: f32,
    pub nhc_gain_scale: f32,
    pub planar_gyro_gain_scale: f32,
    pub yaw_cue_gain_scale: f32,
    pub yaw_prior_gain_scale: f32,
    pub max_mount_rate_radps: [f32; ALIGN_MISALIGN_STATES],
    pub yaw_cue_min_speed_mps: f32,
    pub yaw_cue_min_horiz_acc_mps2: f32,
    pub yaw_cue_min_windows: usize,
    pub yaw_cue_max_windows: usize,
    pub yaw_cue_min_anisotropy_ratio: f32,
    pub min_nhc_speed_mps: f32,
    pub min_planar_speed_mps: f32,
    pub min_planar_yaw_rate_radps: f32,
    pub max_planar_transverse_ratio: f32,
}

impl Default for AlignMisalignConfig {
    fn default() -> Self {
        Self {
            q_mount_std_rad: [
                0.003_f32.to_radians(),
                0.003_f32.to_radians(),
                0.003_f32.to_radians(),
            ],
            r_nhc_std_mps: 0.15,
            r_planar_gyro_std_radps: 0.3_f32.to_radians(),
            r_yaw_cue_std_rad: 5.0_f32.to_radians(),
            r_yaw_prior_std_rad: 8.0_f32.to_radians(),
            nhc_gain_scale: 0.05,
            planar_gyro_gain_scale: 0.1,
            yaw_cue_gain_scale: 0.03,
            yaw_prior_gain_scale: 0.01,
            max_mount_rate_radps: [
                0.05_f32.to_radians(),
                0.05_f32.to_radians(),
                0.2_f32.to_radians(),
            ],
            yaw_cue_min_speed_mps: 5.0 / 3.6,
            yaw_cue_min_horiz_acc_mps2: 0.15,
            yaw_cue_min_windows: 12,
            yaw_cue_max_windows: 40,
            yaw_cue_min_anisotropy_ratio: 1.3,
            min_nhc_speed_mps: 3.0 / 3.6,
            min_planar_speed_mps: 3.0 / 3.6,
            min_planar_yaw_rate_radps: 2.0_f32.to_radians(),
            max_planar_transverse_ratio: 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AlignMisalignTrace {
    pub nhc_valid: bool,
    pub planar_gyro_valid: bool,
    pub yaw_cue_valid: bool,
    pub speed_h_mps: f32,
    pub omega_v_yaw_abs_radps: f32,
    pub omega_v_transverse_radps: f32,
    pub omega_v_transverse_ratio: f32,
    pub nhc_residual_vy_mps: f32,
    pub nhc_residual_vz_mps: f32,
    pub planar_gyro_residual_x_radps: f32,
    pub planar_gyro_residual_y_radps: f32,
    pub yaw_cue_residual_rad: f32,
    pub yaw_prior_residual_rad: f32,
    pub yaw_cue_anisotropy: f32,
}

#[derive(Debug, Clone)]
pub struct AlignMisalign {
    pub q_vb: [f32; 4],
    pub P: [[f32; ALIGN_MISALIGN_STATES]; ALIGN_MISALIGN_STATES],
    pub cfg: AlignMisalignConfig,
    coarse_yaw_rad: Option<f32>,
    prev_v_n: Option<[f32; 3]>,
    yaw_cue_samples: Vec<YawCueSample>,
}

impl Default for AlignMisalign {
    fn default() -> Self {
        Self::new(AlignMisalignConfig::default())
    }
}

impl AlignMisalign {
    pub fn new(cfg: AlignMisalignConfig) -> Self {
        Self {
            q_vb: [1.0, 0.0, 0.0, 0.0],
            P: diag3([
                10.0_f32.to_radians().powi(2),
                10.0_f32.to_radians().powi(2),
                20.0_f32.to_radians().powi(2),
            ]),
            cfg,
            coarse_yaw_rad: None,
            prev_v_n: None,
            yaw_cue_samples: Vec::new(),
        }
    }

    pub fn initialize_from_stationary(
        &mut self,
        accel_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
    ) -> Result<(), &'static str> {
        self.initialize_from_stationary_with_x_ref(accel_samples_b, yaw_seed_rad, [1.0, 0.0, 0.0])
    }

    pub fn stationary_mount_from_accel_with_x_ref(
        accel_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
        mut x_ref: [f32; 3],
    ) -> Result<[f32; 4], &'static str> {
        if accel_samples_b.is_empty() {
            return Err("stationary initialization requires accel samples");
        }
        let mut f_mean_b = [0.0_f32; 3];
        for sample in accel_samples_b {
            f_mean_b = vec3_add(f_mean_b, *sample);
        }
        f_mean_b = vec3_scale(f_mean_b, 1.0 / accel_samples_b.len() as f32);
        let n = vec3_norm(f_mean_b);
        if n < 1.0e-6 {
            return Err("stationary initialization requires nonzero accel mean");
        }

        let z_v_in_b = vec3_scale(f_mean_b, -1.0 / n);
        let mut x_v_in_b = vec3_sub(x_ref, vec3_scale(z_v_in_b, vec3_dot(z_v_in_b, x_ref)));
        if vec3_norm(x_v_in_b) < 1.0e-6 {
            x_ref = [0.0, 1.0, 0.0];
            x_v_in_b = vec3_sub(x_ref, vec3_scale(z_v_in_b, vec3_dot(z_v_in_b, x_ref)));
        }
        x_v_in_b = vec3_normalize(x_v_in_b).ok_or("failed to initialize x axis")?;
        let mut y_v_in_b =
            vec3_normalize(vec3_cross(z_v_in_b, x_v_in_b)).ok_or("failed to initialize y axis")?;
        x_v_in_b = vec3_cross(y_v_in_b, z_v_in_b);
        y_v_in_b = vec3_normalize(y_v_in_b).ok_or("failed to orthonormalize y axis")?;
        x_v_in_b = vec3_normalize(x_v_in_b).ok_or("failed to orthonormalize x axis")?;

        let c_v_b = [x_v_in_b, y_v_in_b, z_v_in_b];
        let rpy = rot_to_euler_zyx(c_v_b);
        let dyaw = wrap_angle_rad(yaw_seed_rad - rpy[2]);
        let (s, c) = dyaw.sin_cos();
        let c_delta = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        Ok(quat_from_rotmat(mat3_mul(c_v_b, c_delta)))
    }

    pub fn initialize_from_stationary_with_x_ref(
        &mut self,
        accel_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
        x_ref: [f32; 3],
    ) -> Result<(), &'static str> {
        self.q_vb =
            Self::stationary_mount_from_accel_with_x_ref(accel_samples_b, yaw_seed_rad, x_ref)?;
        self.P = diag3([
            0.05_f32.to_radians().powi(2),
            0.05_f32.to_radians().powi(2),
            10.0_f32.to_radians().powi(2),
        ]);
        self.prev_v_n = None;
        self.yaw_cue_samples.clear();
        Ok(())
    }

    pub fn seed_mount_yaw_from_nav_course(
        &mut self,
        q_nb: [f32; 4],
        course_rad: f32,
        yaw_std_rad: f32,
    ) {
        let c_n_b = quat_to_rotmat(q_nb);
        let (s, c) = course_rad.sin_cos();
        let c_n_v = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let c_v_n = transpose3x3(c_n_v);
        let c_v_b_target = mat3_mul(c_v_n, c_n_b);
        let target_yaw = c_v_b_target[1][0].atan2(c_v_b_target[0][0]);
        self.set_mount_yaw(target_yaw, yaw_std_rad);
    }

    pub fn seed_mount_from_nav_course_full(
        &mut self,
        q_nb: [f32; 4],
        course_rad: f32,
        tilt_std_rad: f32,
        yaw_std_rad: f32,
    ) {
        let c_n_b = quat_to_rotmat(q_nb);
        let (s, c) = course_rad.sin_cos();
        let c_n_v = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let c_v_n = transpose3x3(c_n_v);
        self.q_vb = quat_from_rotmat(mat3_mul(c_v_n, c_n_b));
        self.P = diag3([
            tilt_std_rad.powi(2),
            tilt_std_rad.powi(2),
            yaw_std_rad.powi(2),
        ]);
        let rpy = rot_to_euler_zyx(quat_to_rotmat(self.q_vb));
        self.coarse_yaw_rad = Some(rpy[2]);
        self.prev_v_n = None;
        self.yaw_cue_samples.clear();
    }

    pub fn set_mount_yaw(&mut self, yaw_rad: f32, yaw_std_rad: f32) {
        let c_v_b_current = quat_to_rotmat(self.q_vb);
        let current_rpy = rot_to_euler_zyx(c_v_b_current);
        self.q_vb = quat_from_rotmat(rot_from_euler_zyx(current_rpy[0], current_rpy[1], yaw_rad));
        self.P[2][2] = yaw_std_rad.powi(2);
        self.P[0][2] = 0.0;
        self.P[1][2] = 0.0;
        self.P[2][0] = 0.0;
        self.P[2][1] = 0.0;
        self.coarse_yaw_rad = Some(yaw_rad);
        self.prev_v_n = None;
        self.yaw_cue_samples.clear();
    }

    pub fn predict(&mut self, dt: f32) {
        let dt = dt.max(1.0e-3);
        for i in 0..ALIGN_MISALIGN_STATES {
            self.P[i][i] += self.cfg.q_mount_std_rad[i].powi(2) * dt;
        }
    }

    pub fn update_all(
        &mut self,
        dt: f32,
        q_nb: [f32; 4],
        v_n: [f32; 3],
        omega_b_corr: [f32; 3],
        accel_b_corr: [f32; 3],
    ) -> (f32, AlignMisalignTrace) {
        self.predict(dt);
        let mut score = 0.0_f32;
        let mut trace = AlignMisalignTrace::default();

        let speed_h = self.horizontal_speed_mps(v_n);
        trace.speed_h_mps = speed_h;
        let omega_v = self.omega_v(omega_b_corr);
        trace.omega_v_yaw_abs_radps = omega_v[2].abs();
        trace.omega_v_transverse_radps = (omega_v[0] * omega_v[0] + omega_v[1] * omega_v[1]).sqrt();
        trace.omega_v_transverse_ratio = if trace.omega_v_yaw_abs_radps > 1.0e-6 {
            trace.omega_v_transverse_radps / trace.omega_v_yaw_abs_radps
        } else {
            f32::INFINITY
        };

        let nhc_valid = self.nhc_gate(v_n);
        trace.nhc_valid = nhc_valid;
        if nhc_valid {
            let pred = self.nhc_prediction(q_nb, v_n);
            trace.nhc_residual_vy_mps = -pred[0];
            trace.nhc_residual_vz_mps = -pred[1];
            score += self.update_nhc(dt, q_nb, v_n);
        }

        let planar_valid = self.planar_gyro_gate(v_n, omega_b_corr);
        trace.planar_gyro_valid = planar_valid;
        if planar_valid {
            let pred = self.planar_gyro_prediction(omega_b_corr);
            trace.planar_gyro_residual_x_radps = -pred[0];
            trace.planar_gyro_residual_y_radps = -pred[1];
            score += self.update_planar_gyro(dt, omega_b_corr);
        }

        let (yaw_cue_valid, yaw_cue_residual, yaw_cue_anisotropy, yaw_cue_score) =
            self.update_yaw_cue(dt, v_n, accel_b_corr);
        trace.yaw_cue_valid = yaw_cue_valid;
        trace.yaw_cue_residual_rad = yaw_cue_residual;
        trace.yaw_cue_anisotropy = yaw_cue_anisotropy;
        score += yaw_cue_score;

        if let Some(coarse_yaw_rad) = self.coarse_yaw_rad {
            let rpy = rot_to_euler_zyx(quat_to_rotmat(self.q_vb));
            let innovation = wrap_pi(coarse_yaw_rad - rpy[2]);
            trace.yaw_prior_residual_rad = innovation;
            score += self.apply_measurement1(
                dt,
                innovation,
                &[0.0, 0.0, 1.0],
                self.cfg.r_yaw_prior_std_rad.powi(2),
                self.cfg.yaw_prior_gain_scale,
            );
        }

        (score, trace)
    }

    pub fn nhc_gate(&self, v_n: [f32; 3]) -> bool {
        self.horizontal_speed_mps(v_n) >= self.cfg.min_nhc_speed_mps
    }

    pub fn planar_gyro_gate(&self, v_n: [f32; 3], omega_b_corr: [f32; 3]) -> bool {
        let speed_h = self.horizontal_speed_mps(v_n);
        if speed_h < self.cfg.min_planar_speed_mps {
            return false;
        }
        let omega_v = self.omega_v(omega_b_corr);
        let yaw_abs = omega_v[2].abs();
        if yaw_abs < self.cfg.min_planar_yaw_rate_radps {
            return false;
        }
        let transverse = (omega_v[0] * omega_v[0] + omega_v[1] * omega_v[1]).sqrt();
        transverse <= self.cfg.max_planar_transverse_ratio * yaw_abs
    }

    pub fn horizontal_speed_mps(&self, v_n: [f32; 3]) -> f32 {
        (v_n[0] * v_n[0] + v_n[1] * v_n[1]).sqrt()
    }

    pub fn omega_v(&self, omega_b_corr: [f32; 3]) -> [f32; 3] {
        mat3_vec(quat_to_rotmat(self.q_vb), omega_b_corr)
    }

    pub fn nhc_prediction(&self, q_nb: [f32; 4], v_n: [f32; 3]) -> [f32; 2] {
        let c_b_n = transpose3x3(quat_to_rotmat(q_nb));
        let v_b = mat3_vec(c_b_n, v_n);
        let v_v = mat3_vec(quat_to_rotmat(self.q_vb), v_b);
        [v_v[1], v_v[2]]
    }

    pub fn planar_gyro_prediction(&self, omega_b_corr: [f32; 3]) -> [f32; 2] {
        let omega_v = mat3_vec(quat_to_rotmat(self.q_vb), omega_b_corr);
        [omega_v[0], omega_v[1]]
    }

    pub fn nhc_mount_jacobian(&self, q_nb: [f32; 4], v_n: [f32; 3]) -> [[f32; 3]; 2] {
        let c_n_b = quat_to_rotmat(q_nb);
        let c_b_n = transpose3x3(c_n_b);
        let c_v_b = quat_to_rotmat(self.q_vb);
        let v_b = mat3_vec(c_b_n, v_n);
        let s_yz = [[0.0_f32, 1.0, 0.0], [0.0_f32, 0.0, 1.0]];
        mat2x3_mul3x3(s_yz, mat3_mul(c_v_b, negate3(skew3(v_b))))
    }

    pub fn planar_gyro_mount_jacobian(&self, omega_b_corr: [f32; 3]) -> [[f32; 3]; 2] {
        let c_v_b = quat_to_rotmat(self.q_vb);
        let s_xy = [[1.0_f32, 0.0, 0.0], [0.0_f32, 1.0, 0.0]];
        mat2x3_mul3x3(s_xy, mat3_mul(c_v_b, negate3(skew3(omega_b_corr))))
    }

    fn update_nhc(&mut self, dt: f32, q_nb: [f32; 4], v_n: [f32; 3]) -> f32 {
        let pred = self.nhc_prediction(q_nb, v_n);
        let h_theta_vb = self.nhc_mount_jacobian(q_nb, v_n);
        let mut score = 0.0_f32;
        for axis in 0..2 {
            let innovation = -pred[axis];
            score += self.apply_measurement1(
                dt,
                innovation,
                &h_theta_vb[axis],
                self.cfg.r_nhc_std_mps.powi(2),
                self.cfg.nhc_gain_scale,
            );
        }
        score
    }

    fn update_planar_gyro(&mut self, dt: f32, omega_b_corr: [f32; 3]) -> f32 {
        let pred = self.planar_gyro_prediction(omega_b_corr);
        let h_theta_vb = self.planar_gyro_mount_jacobian(omega_b_corr);
        let mut score = 0.0_f32;
        for axis in 0..2 {
            let innovation = -pred[axis];
            score += self.apply_measurement1(
                dt,
                innovation,
                &h_theta_vb[axis],
                self.cfg.r_planar_gyro_std_radps.powi(2),
                self.cfg.planar_gyro_gain_scale,
            );
        }
        score
    }

    fn update_yaw_cue(
        &mut self,
        dt: f32,
        v_n: [f32; 3],
        accel_b_corr: [f32; 3],
    ) -> (bool, f32, f32, f32) {
        let speed_mps = self.horizontal_speed_mps(v_n);
        let rpy = rot_to_euler_zyx(quat_to_rotmat(self.q_vb));
        let yaw_rad = rpy[2];
        let tilt_rot = { rot_from_euler_zyx(rpy[0], rpy[1], 0.0) };
        let accel_level = mat3_vec(tilt_rot, accel_b_corr);
        let horiz_xy = [accel_level[0], accel_level[1]];

        if let Some(prev_v_n) = self.prev_v_n {
            let dt_safe = dt.max(1.0e-3);
            let a_n = [
                (v_n[0] - prev_v_n[0]) / dt_safe,
                (v_n[1] - prev_v_n[1]) / dt_safe,
                (v_n[2] - prev_v_n[2]) / dt_safe,
            ];
            let course = if speed_mps > 1.0e-6 {
                [v_n[0] / speed_mps, v_n[1] / speed_mps]
            } else {
                [0.0, 0.0]
            };
            let gnss_long_mps2 = a_n[0] * course[0] + a_n[1] * course[1];
            if speed_mps >= self.cfg.yaw_cue_min_speed_mps
                && vec2_norm(horiz_xy) >= self.cfg.yaw_cue_min_horiz_acc_mps2
            {
                self.yaw_cue_samples.push(YawCueSample {
                    horiz_xy,
                    gnss_long_mps2,
                });
                if self.yaw_cue_samples.len() > self.cfg.yaw_cue_max_windows {
                    let drop_n = self.yaw_cue_samples.len() - self.cfg.yaw_cue_max_windows;
                    self.yaw_cue_samples.drain(0..drop_n);
                }
            }
        }
        self.prev_v_n = Some(v_n);

        if self.yaw_cue_samples.len() < self.cfg.yaw_cue_min_windows {
            return (false, 0.0, 0.0, 0.0);
        }
        let Some((mut theta_rad, anisotropy)) = principal_axis_angle(&self.yaw_cue_samples) else {
            return (false, 0.0, 0.0, 0.0);
        };
        if anisotropy < self.cfg.yaw_cue_min_anisotropy_ratio {
            return (false, 0.0, anisotropy, 0.0);
        }
        let axis = [theta_rad.cos(), theta_rad.sin()];
        let corr = self
            .yaw_cue_samples
            .iter()
            .map(|s| s.gnss_long_mps2 * dot2(s.horiz_xy, axis))
            .sum::<f32>();
        if corr < 0.0 {
            theta_rad = wrap_pi(theta_rad + std::f32::consts::PI);
        }
        let cue_yaw_rad = wrap_pi(-theta_rad);
        let innovation = wrap_pi(cue_yaw_rad - yaw_rad);
        let score = self.apply_measurement1(
            dt,
            innovation,
            &[0.0, 0.0, 1.0],
            self.cfg.r_yaw_cue_std_rad.powi(2),
            self.cfg.yaw_cue_gain_scale,
        );
        (true, innovation, anisotropy, score)
    }

    fn apply_measurement1(
        &mut self,
        dt: f32,
        innovation: f32,
        h: &[f32; ALIGN_MISALIGN_STATES],
        r_var: f32,
        gain_scale: f32,
    ) -> f32 {
        let p_old = self.P;
        let mut ph = [0.0_f32; ALIGN_MISALIGN_STATES];
        for i in 0..ALIGN_MISALIGN_STATES {
            for (j, hj) in h.iter().enumerate() {
                ph[i] += p_old[i][j] * *hj;
            }
        }
        let mut s = r_var.max(1.0e-9);
        for j in 0..ALIGN_MISALIGN_STATES {
            s += h[j] * ph[j];
        }
        let inv_s = if s > 1.0e-9 { 1.0 / s } else { 0.0 };
        let mut k = [0.0_f32; ALIGN_MISALIGN_STATES];
        for i in 0..ALIGN_MISALIGN_STATES {
            k[i] = ph[i] * inv_s;
        }
        let mut dx = [k[0] * innovation, k[1] * innovation, k[2] * innovation];
        let alpha = gain_scale.clamp(0.0, 1.0);
        for i in 0..ALIGN_MISALIGN_STATES {
            dx[i] *= alpha;
            let max_step = self.cfg.max_mount_rate_radps[i] * dt.max(1.0e-3);
            dx[i] = dx[i].clamp(-max_step, max_step);
        }
        self.inject_error(dx);

        let mut hp = [0.0_f32; ALIGN_MISALIGN_STATES];
        for j in 0..ALIGN_MISALIGN_STATES {
            for kk in 0..ALIGN_MISALIGN_STATES {
                hp[j] += h[kk] * p_old[kk][j];
            }
        }
        for i in 0..ALIGN_MISALIGN_STATES {
            for j in 0..ALIGN_MISALIGN_STATES {
                self.P[i][j] = p_old[i][j] - alpha * k[i] * hp[j];
            }
        }
        self.P = symmetrize_mat(self.P);
        innovation * innovation * inv_s
    }

    fn inject_error(&mut self, dtheta: [f32; 3]) {
        self.q_vb = quat_normalize(quat_mul(self.q_vb, quat_from_small_angle(dtheta)));
    }
}

fn symmetrize_mat(
    mut m: [[f32; ALIGN_MISALIGN_STATES]; ALIGN_MISALIGN_STATES],
) -> [[f32; ALIGN_MISALIGN_STATES]; ALIGN_MISALIGN_STATES] {
    for i in 0..ALIGN_MISALIGN_STATES {
        for j in i..ALIGN_MISALIGN_STATES {
            let temp = 0.5 * (m[i][j] + m[j][i]);
            m[i][j] = temp;
            m[j][i] = temp;
        }
    }
    m
}

#[derive(Debug, Clone, Copy)]
struct YawCueSample {
    horiz_xy: [f32; 2],
    gnss_long_mps2: f32,
}

fn diag3(d: [f32; 3]) -> [[f32; 3]; 3] {
    [[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]]
}

fn quat_to_rotmat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = quat_normalize(q);
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

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_from_small_angle(dtheta: [f32; 3]) -> [f32; 4] {
    let half = [0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]];
    quat_normalize([1.0, half[0], half[1], half[2]])
}

fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 1.0e-9 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

fn transpose3x3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0_f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

fn mat3_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec2_norm(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn dot2(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn principal_axis_angle(samples: &[YawCueSample]) -> Option<(f32, f32)> {
    if samples.is_empty() {
        return None;
    }
    let mut sxx = 0.0_f32;
    let mut sxy = 0.0_f32;
    let mut syy = 0.0_f32;
    for s in samples {
        let x = s.horiz_xy[0];
        let y = s.horiz_xy[1];
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }
    let trace = sxx + syy;
    let disc = ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt();
    let lambda_max = 0.5 * (trace + disc);
    let lambda_min = 0.5 * (trace - disc).max(0.0);
    if lambda_max <= 1.0e-6 {
        return None;
    }
    Some((
        0.5 * (2.0 * sxy).atan2(sxx - syy),
        lambda_max / lambda_min.max(1.0e-6),
    ))
}

fn wrap_pi(x: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    (x + std::f32::consts::PI).rem_euclid(two_pi) - std::f32::consts::PI
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
    if n > 1.0e-9 {
        Some(vec3_scale(v, 1.0 / n))
    } else {
        None
    }
}

fn skew3(v: [f32; 3]) -> [[f32; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

fn negate3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [-m[0][0], -m[0][1], -m[0][2]],
        [-m[1][0], -m[1][1], -m[1][2]],
        [-m[2][0], -m[2][1], -m[2][2]],
    ]
}

fn mat2x3_mul3x3(a: [[f32; 3]; 2], b: [[f32; 3]; 3]) -> [[f32; 3]; 2] {
    let mut out = [[0.0_f32; 3]; 2];
    for r in 0..2 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

fn rot_to_euler_zyx(c: [[f32; 3]; 3]) -> [f32; 3] {
    let pitch = (-c[2][0]).asin();
    let roll = c[2][1].atan2(c[2][2]);
    let yaw = c[1][0].atan2(c[0][0]);
    [roll, pitch, yaw]
}

fn rot_from_euler_zyx(roll: f32, pitch: f32, yaw: f32) -> [[f32; 3]; 3] {
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

fn wrap_angle_rad(mut x: f32) -> f32 {
    while x > std::f32::consts::PI {
        x -= 2.0 * std::f32::consts::PI;
    }
    while x < -std::f32::consts::PI {
        x += 2.0 * std::f32::consts::PI;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nhc_prediction_zero_for_forward_velocity() {
        let f = AlignMisalign::default();
        let pred = f.nhc_prediction([1.0, 0.0, 0.0, 0.0], [12.0, 0.0, 0.0]);
        assert!(pred[0].abs() < 1.0e-6);
        assert!(pred[1].abs() < 1.0e-6);
    }

    #[test]
    fn seed_mount_yaw_from_nav_course_matches_expected_mount() {
        let mut f = AlignMisalign::default();
        let q_before = quat_from_small_angle([8.0_f32.to_radians(), -6.0_f32.to_radians(), 0.0]);
        f.q_vb = q_before;
        let q_nb = quat_from_rotmat([
            [0.5, -0.8660254, 0.0],
            [0.8660254, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        f.seed_mount_yaw_from_nav_course(q_nb, 35.0_f32.to_radians(), 5.0_f32.to_radians());
        assert!((f.P[2][2].sqrt().to_degrees() - 5.0).abs() < 1.0e-3);
        let before_rpy = rot_to_euler_zyx(quat_to_rotmat(q_before));
        let after_rpy = rot_to_euler_zyx(quat_to_rotmat(f.q_vb));
        assert!((before_rpy[0] - after_rpy[0]).abs() < 1.0e-4);
        assert!((before_rpy[1] - after_rpy[1]).abs() < 1.0e-4);
    }

    #[test]
    fn nhc_update_reduces_mount_error_with_true_nav_inputs() {
        let q_true = quat_from_small_angle([
            4.0_f32.to_radians(),
            -3.0_f32.to_radians(),
            12.0_f32.to_radians(),
        ]);
        let mut f = AlignMisalign::default();
        f.q_vb = quat_from_small_angle([0.0, 0.0, 25.0_f32.to_radians()]);
        let q_nb = [1.0, 0.0, 0.0, 0.0];
        let v_b = mat3_vec(transpose3x3(quat_to_rotmat(q_true)), [15.0, 0.0, 0.0]);
        let v_n = v_b;
        let err0 = axis_angle_deg(f.q_vb, q_true);
        for _ in 0..40 {
            let _ = f.update_all(
                0.1,
                q_nb,
                v_n,
                [0.0, 0.0, 8.0_f32.to_radians()],
                [0.5, 0.0, 0.0],
            );
        }
        let err1 = axis_angle_deg(f.q_vb, q_true);
        assert!(err1 < err0);
    }

    #[test]
    fn nhc_jacobian_matches_finite_difference() {
        let f = AlignMisalign::default();
        let q_nb = quat_from_small_angle([
            3.0_f32.to_radians(),
            -2.0_f32.to_radians(),
            10.0_f32.to_radians(),
        ]);
        let v_n = [12.0, 4.0, -0.7];
        let h = f.nhc_mount_jacobian(q_nb, v_n);
        let base = f.nhc_prediction(q_nb, v_n);
        let eps = 1.0e-5_f32;
        for j in 0..3 {
            let mut fp = f.clone();
            let mut d = [0.0_f32; 3];
            d[j] = eps;
            fp.inject_error(d);
            let pred = fp.nhc_prediction(q_nb, v_n);
            for row in 0..2 {
                let fd = (pred[row] - base[row]) / eps;
                assert!(
                    (fd - h[row][j]).abs() < 1.0e-2,
                    "row={row} col={j} fd={fd} analytic={}",
                    h[row][j]
                );
            }
        }
    }

    #[test]
    fn planar_gyro_jacobian_matches_finite_difference() {
        let f = AlignMisalign::default();
        let omega_b = [0.3, -0.2, 1.2];
        let h = f.planar_gyro_mount_jacobian(omega_b);
        let base = f.planar_gyro_prediction(omega_b);
        let eps = 1.0e-5_f32;
        for j in 0..3 {
            let mut fp = f.clone();
            let mut d = [0.0_f32; 3];
            d[j] = eps;
            fp.inject_error(d);
            let pred = fp.planar_gyro_prediction(omega_b);
            for row in 0..2 {
                let fd = (pred[row] - base[row]) / eps;
                assert!(
                    (fd - h[row][j]).abs() < 1.0e-2,
                    "row={row} col={j} fd={fd} analytic={}",
                    h[row][j]
                );
            }
        }
    }

    fn axis_angle_deg(a: [f32; 4], b: [f32; 4]) -> f32 {
        let dq = quat_mul(a, [b[0], -b[1], -b[2], -b[3]]);
        let w = dq[0].clamp(-1.0, 1.0).abs();
        (2.0 * w.acos()).to_degrees()
    }
}
