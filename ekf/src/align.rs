#![allow(non_snake_case)]

pub const ALIGN_N_STATES: usize = 3;
pub const GRAVITY_MPS2: f32 = 9.80665;

#[derive(Debug, Clone, Copy)]
pub struct AlignConfig {
    pub q_mount_std_rad: [f32; ALIGN_N_STATES],
    pub r_gravity_std_mps2: f32,
    pub r_horiz_heading_std_rad: f32,
    pub r_turn_gyro_std_radps: f32,
    pub turn_gyro_yaw_scale: f32,
    pub r_turn_heading_std_rad: f32,
    pub gravity_lpf_alpha: f32,
    pub min_speed_mps: f32,
    pub min_turn_rate_radps: f32,
    pub min_lat_acc_mps2: f32,
    pub min_long_acc_mps2: f32,
    pub turn_consistency_min_windows: usize,
    pub turn_consistency_min_fraction: f32,
    pub turn_consistency_max_abs_lat_err_mps2: f32,
    pub turn_consistency_max_rel_lat_err: f32,
    pub max_stationary_gyro_radps: f32,
    pub max_stationary_accel_norm_err_mps2: f32,
    pub use_gravity: bool,
    pub use_turn_gyro: bool,
}

impl Default for AlignConfig {
    fn default() -> Self {
        Self {
            q_mount_std_rad: [
                0.0005_f32.to_radians(),
                0.0005_f32.to_radians(),
                0.00005_f32.to_radians(),
            ],
            r_gravity_std_mps2: 0.56,
            r_horiz_heading_std_rad: 2.0_f32.to_radians(),
            r_turn_heading_std_rad: 0.2_f32.to_radians(),
            r_turn_gyro_std_radps: 0.02_f32.to_radians(),
            turn_gyro_yaw_scale: 0.0,
            gravity_lpf_alpha: 0.04,
            min_speed_mps: 3.0 / 3.6,
            min_turn_rate_radps: 2.0_f32.to_radians(),
            min_lat_acc_mps2: 0.10,
            min_long_acc_mps2: 0.18,
            turn_consistency_min_windows: 5,
            turn_consistency_min_fraction: 0.8,
            turn_consistency_max_abs_lat_err_mps2: 0.35,
            turn_consistency_max_rel_lat_err: 0.6,
            max_stationary_gyro_radps: 0.8_f32.to_radians(),
            max_stationary_accel_norm_err_mps2: 0.2,
            use_gravity: true,
            use_turn_gyro: true,
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
    pub coarse_alignment_ready: bool,
    pub after_gravity: Option<[f32; 4]>,
    pub after_gravity_quasi_static: bool,
    pub after_horiz_accel: Option<[f32; 4]>,
    pub horiz_angle_err_rad: Option<f32>,
    pub horiz_effective_std_rad: Option<f32>,
    pub horiz_gnss_norm_mps2: Option<f32>,
    pub horiz_imu_norm_mps2: Option<f32>,
    pub horiz_obs_accel_vx: Option<f32>,
    pub horiz_obs_accel_vy: Option<f32>,
    pub horiz_accel_bx: Option<f32>,
    pub horiz_accel_by: Option<f32>,
    pub horiz_speed_q: Option<f32>,
    pub horiz_accel_q: Option<f32>,
    pub horiz_straight_q: Option<f32>,
    pub horiz_turn_q: Option<f32>,
    pub horiz_dominance_q: Option<f32>,
    pub horiz_turn_core_valid: bool,
    pub horiz_straight_core_valid: bool,
    pub after_turn_gyro: Option<[f32; 4]>,
}

#[derive(Debug, Clone, Copy, Default)]
struct TurnConsistencySample {
    speed_mps: f32,
    course_rate_radps: f32,
    a_lat_mps2: f32,
}

#[derive(Debug, Clone)]
pub struct Align {
    pub q_vb: [f32; 4],
    pub P: [[f32; ALIGN_N_STATES]; ALIGN_N_STATES],
    pub gravity_lp_b: [f32; 3],
    coarse_aligned: bool,
    yaw_observed: bool,
    turn_samples: [TurnConsistencySample; 16],
    turn_count: usize,
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
            coarse_aligned: false,
            yaw_observed: false,
            turn_samples: [TurnConsistencySample::default(); 16],
            turn_count: 0,
            cfg,
        }
    }

    pub fn initialize_from_stationary(
        &mut self,
        accel_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
    ) -> Result<(), &'static str> {
        let mean = mean_accel(accel_samples_b).ok_or("stationary bootstrap failed")?;
        let c_bv =
            stationary_mount_rotmat(mean, yaw_seed_rad).ok_or("stationary bootstrap failed")?;
        self.q_vb = rotmat_to_quat(c_bv);
        self.P = diag3([
            0.2_f32.to_radians().powi(2),
            0.2_f32.to_radians().powi(2),
            60.0_f32.to_radians().powi(2),
        ]);
        self.gravity_lp_b = mean;
        self.turn_consistency_reset();
        self.yaw_observed = false;
        self.coarse_aligned = false;
        Ok(())
    }

    pub fn predict(&mut self, dt: f32) {
        let dt = dt.max(1.0e-3);
        for i in 0..3 {
            self.P[i][i] += self.cfg.q_mount_std_rad[i] * self.cfg.q_mount_std_rad[i] * dt;
        }
    }

    pub fn update_window(&mut self, window: &AlignWindowSummary) -> f32 {
        self.update_window_with_trace(window).0
    }

    pub fn update_window_with_trace(
        &mut self,
        window: &AlignWindowSummary,
    ) -> (f32, AlignUpdateTrace) {
        let mut trace = AlignUpdateTrace {
            q_start: self.q_vb,
            ..AlignUpdateTrace::default()
        };
        self.predict(window.dt);

        let v_prev = window.gnss_vel_prev_n;
        let v_curr = window.gnss_vel_curr_n;
        let speed_prev = norm2([window.gnss_vel_prev_n[0], window.gnss_vel_prev_n[1]]);
        let speed_curr = norm2([window.gnss_vel_curr_n[0], window.gnss_vel_curr_n[1]]);
        let speed_mid = 0.5 * (speed_prev + speed_curr);
        let dt = window.dt.max(1.0e-3);

        let course_prev = v_prev[1].atan2(v_prev[0]);
        let course_curr = v_curr[1].atan2(v_curr[0]);
        let course_rate = wrap_pi_mod(course_curr - course_prev) / dt;

        let a_n = [
            (v_curr[0] - v_prev[0]) / dt,
            (v_curr[1] - v_prev[1]) / dt,
            (v_curr[2] - v_prev[2]) / dt,
        ];

        let v_mid_h = [0.5 * (v_prev[0] + v_curr[0]), 0.5 * (v_prev[1] + v_curr[1])];
        let mut a_long = 0.0;
        let mut a_lat = 0.0;
        if let Some(t_hat) = vec2_normalize(v_mid_h) {
            let lat_hat = [-t_hat[1], t_hat[0]];
            a_long = t_hat[0] * a_n[0] + t_hat[1] * a_n[1];
            a_lat = lat_hat[0] * a_n[0] + lat_hat[1] * a_n[1];
        }

        let gyro_norm = vec3_norm(window.mean_gyro_b);
        let accel_norm = vec3_norm(window.mean_accel_b);
        let horiz_accel_b = remove_gravity_axis(self.q_vb, window.mean_accel_b);
        let horiz_obs = align_obs(self.q_vb, window.mean_gyro_b, horiz_accel_b);

        let stationary = gyro_norm <= self.cfg.max_stationary_gyro_radps
            && (accel_norm - GRAVITY_MPS2).abs() <= self.cfg.max_stationary_accel_norm_err_mps2
            && speed_mid < 0.5;
        let turn_valid = speed_mid > self.cfg.min_speed_mps
            && course_rate.abs() > self.cfg.min_turn_rate_radps
            && a_lat.abs() > self.cfg.min_lat_acc_mps2;
        let turn_heading_valid =
            self.turn_consistency_update(turn_valid, speed_mid, course_rate, a_lat);
        let horiz_gnss_norm = (a_long * a_long + a_lat * a_lat).sqrt();
        let long_valid = speed_mid > self.cfg.min_speed_mps
            && a_long.abs() > self.cfg.min_long_acc_mps2
            && a_lat.abs() < 0.5_f32.max(0.6 * a_long.abs())
            && horiz_gnss_norm > self.cfg.min_long_acc_mps2;

        let mut score = 0.0;
        if stationary {
            for (dst, src) in self.gravity_lp_b.iter_mut().zip(window.mean_accel_b) {
                *dst = (1.0 - self.cfg.gravity_lpf_alpha) * *dst + self.cfg.gravity_lpf_alpha * src;
            }
        }
        if self.cfg.use_gravity && stationary {
            let gravity_state_mask = [true, true, false];
            let r_gravity = self.cfg.r_gravity_std_mps2 * self.cfg.r_gravity_std_mps2;
            score += apply_update2_scaled_masked(
                &mut self.q_vb,
                &mut self.P,
                [0.0, 0.0],
                [3, 4],
                self.gravity_lp_b,
                window.mean_gyro_b,
                [r_gravity, r_gravity],
                gravity_state_mask,
                [1.0, 1.0, 1.0],
            );
            score += apply_update1_masked(
                &mut self.q_vb,
                &mut self.P,
                -vec3_norm(self.gravity_lp_b),
                5,
                self.gravity_lp_b,
                window.mean_gyro_b,
                r_gravity,
                gravity_state_mask,
            );
            trace.after_gravity = Some(self.q_vb);
            trace.after_gravity_quasi_static = false;
        }

        let horiz_imu_norm = (horiz_obs[3] * horiz_obs[3] + horiz_obs[4] * horiz_obs[4]).sqrt();
        let straight_core_valid = long_valid;
        let turn_core_valid = turn_heading_valid
            && speed_mid > (10.0 / 3.6)
            && a_lat.abs() > self.cfg.min_lat_acc_mps2.max(0.7)
            && a_lat.abs() > 1.5 * a_long.abs().max(0.2);
        let horiz_vector_valid = speed_mid > self.cfg.min_speed_mps
            && horiz_gnss_norm > self.cfg.min_long_acc_mps2
            && horiz_imu_norm > self.cfg.min_long_acc_mps2
            && (straight_core_valid || turn_core_valid);
        trace.horiz_straight_core_valid = straight_core_valid;
        trace.horiz_turn_core_valid = turn_core_valid;

        if horiz_vector_valid {
            let speed_q = ((speed_mid - (10.0 / 3.6)) / (20.0 / 3.6 - 10.0 / 3.6)).clamp(0.0, 1.0);
            let accel_q = ((horiz_gnss_norm.min(horiz_imu_norm) - 0.5) / 1.0).clamp(0.0, 1.0);
            trace.horiz_gnss_norm_mps2 = Some(horiz_gnss_norm);
            trace.horiz_imu_norm_mps2 = Some(horiz_imu_norm);
            trace.horiz_obs_accel_vx = Some(horiz_obs[3]);
            trace.horiz_obs_accel_vy = Some(horiz_obs[4]);
            trace.horiz_accel_bx = Some(horiz_accel_b[0]);
            trace.horiz_accel_by = Some(horiz_accel_b[1]);
            trace.horiz_speed_q = Some(speed_q);
            trace.horiz_accel_q = Some(accel_q);

            let effective_std = if turn_core_valid {
                let dominance = ((a_lat.abs() / (a_long.abs() + 0.2)) - 1.5) / 1.5;
                let lat_q =
                    ((a_lat.abs() - self.cfg.min_lat_acc_mps2.max(0.7)) / 1.0).clamp(0.0, 1.0);
                let turn_q = (0.35
                    + 0.65 * (speed_q * accel_q * lat_q * dominance.clamp(0.0, 1.0)))
                .clamp(0.35, 1.0);
                trace.horiz_dominance_q = Some(dominance.clamp(0.0, 1.0));
                trace.horiz_turn_q = Some(turn_q);
                self.cfg.r_turn_heading_std_rad / turn_q
            } else {
                let lat_ratio = a_lat.abs() / (0.5 + 0.6 * a_long.abs());
                let long_q = ((a_long.abs() - self.cfg.min_long_acc_mps2) / 0.8).clamp(0.0, 1.0);
                let straight_q = (speed_q * accel_q * long_q * (1.0 - lat_ratio.clamp(0.0, 1.0)))
                    .clamp(0.2, 1.0);
                trace.horiz_straight_q = Some(straight_q);
                self.cfg.r_horiz_heading_std_rad / straight_q
            };

            let cross = horiz_obs[3] * a_lat - horiz_obs[4] * a_long;
            let dot = horiz_obs[3] * a_long + horiz_obs[4] * a_lat;
            let angle_err = cross.atan2(dot);
            trace.horiz_angle_err_rad = Some(angle_err);
            trace.horiz_effective_std_rad = Some(effective_std);
            score += apply_vehicle_yaw_angle(
                &mut self.q_vb,
                &mut self.P,
                angle_err,
                effective_std * effective_std,
            );
            self.yaw_observed = true;
            trace.after_horiz_accel = Some(self.q_vb);
        }

        if turn_valid && self.cfg.use_turn_gyro {
            let turn_gyro_yaw_scale = if turn_heading_valid {
                self.cfg.turn_gyro_yaw_scale
            } else {
                0.0
            };
            let state_mask = [true, true, turn_gyro_yaw_scale > 0.0];
            let state_scale = [1.0, 1.0, turn_gyro_yaw_scale];
            let r_turn_gyro = self.cfg.r_turn_gyro_std_radps * self.cfg.r_turn_gyro_std_radps;
            score += apply_update2_scaled_masked(
                &mut self.q_vb,
                &mut self.P,
                [0.0, 0.0],
                [0, 1],
                window.mean_accel_b,
                window.mean_gyro_b,
                [r_turn_gyro, r_turn_gyro],
                state_mask,
                state_scale,
            );
            if state_mask[2] {
                self.yaw_observed = true;
            }
            trace.after_turn_gyro = Some(self.q_vb);
        }

        self.coarse_aligned = self.compute_coarse_alignment_ready();
        trace.coarse_alignment_ready = self.coarse_aligned;
        (score, trace)
    }

    pub fn mount_angles_rad(&self) -> [f32; 3] {
        rot_to_euler_zyx(quat_to_rotmat(self.q_vb))
    }

    pub fn mount_angles_deg(&self) -> [f32; 3] {
        let r = self.mount_angles_rad();
        [r[0].to_degrees(), r[1].to_degrees(), r[2].to_degrees()]
    }

    pub fn coarse_alignment_ready(&self) -> bool {
        self.coarse_aligned
    }

    pub fn sigma_deg(&self) -> [f32; 3] {
        [
            self.P[0][0].max(0.0).sqrt().to_degrees(),
            self.P[1][1].max(0.0).sqrt().to_degrees(),
            self.P[2][2].max(0.0).sqrt().to_degrees(),
        ]
    }

    fn compute_coarse_alignment_ready(&self) -> bool {
        self.yaw_observed
            && self.P[0][0].max(0.0).sqrt().to_degrees() <= 0.15
            && self.P[1][1].max(0.0).sqrt().to_degrees() <= 0.15
            && self.P[2][2].max(0.0).sqrt().to_degrees() <= 0.15
    }

    fn turn_consistency_update(
        &mut self,
        turn_valid: bool,
        speed_mps: f32,
        course_rate_radps: f32,
        a_lat_mps2: f32,
    ) -> bool {
        if !turn_valid {
            self.turn_consistency_reset();
            return false;
        }

        let sample = TurnConsistencySample {
            speed_mps,
            course_rate_radps,
            a_lat_mps2,
        };
        if self.turn_count < self.turn_samples.len() {
            self.turn_samples[self.turn_count] = sample;
            self.turn_count += 1;
        } else {
            self.turn_samples.copy_within(1.., 0);
            let last = self.turn_samples.len() - 1;
            self.turn_samples[last] = sample;
        }

        let min_windows = self.cfg.turn_consistency_min_windows.max(1);
        if self.turn_count < min_windows {
            return false;
        }

        let mut sign_ok = 0usize;
        let mut model_ok = 0usize;
        for sample in self.turn_samples.iter().take(self.turn_count) {
            let a_lat_pred = sample.speed_mps * sample.course_rate_radps;
            let tol = self.cfg.turn_consistency_max_abs_lat_err_mps2.max(
                self.cfg.turn_consistency_max_rel_lat_err
                    * a_lat_pred.abs().max(sample.a_lat_mps2.abs()),
            );
            if a_lat_pred * sample.a_lat_mps2 > 0.0 {
                sign_ok += 1;
            }
            if (sample.a_lat_mps2 - a_lat_pred).abs() <= tol {
                model_ok += 1;
            }
        }

        let fraction = self.cfg.turn_consistency_min_fraction.clamp(0.0, 1.0);
        let min_ok = (fraction * self.turn_count as f32).ceil() as usize;
        sign_ok >= min_ok && model_ok >= min_ok
    }

    fn turn_consistency_reset(&mut self) {
        self.turn_count = 0;
    }
}

pub fn leveled_horiz_accel_xy(gravity_b: [f32; 3], horiz_accel_b: [f32; 3]) -> Option<[f32; 2]> {
    let (x_in_b, y_in_b) = leveled_xy_axes(gravity_b)?;
    Some([
        vec3_dot(horiz_accel_b, x_in_b),
        vec3_dot(horiz_accel_b, y_in_b),
    ])
}

fn mean_accel(samples: &[[f32; 3]]) -> Option<[f32; 3]> {
    if samples.is_empty() {
        return None;
    }
    let mut sum = [0.0; 3];
    for sample in samples {
        sum[0] += sample[0];
        sum[1] += sample[1];
        sum[2] += sample[2];
    }
    let inv = 1.0 / samples.len() as f32;
    Some([sum[0] * inv, sum[1] * inv, sum[2] * inv])
}

fn stationary_mount_rotmat(accel_b: [f32; 3], yaw_seed_rad: f32) -> Option<[[f32; 3]; 3]> {
    let z_v_in_b = vec3_normalize(vec3_scale(accel_b, -1.0))?;
    let mut x_ref = [1.0, 0.0, 0.0];
    let mut x_v_in_b = vec3_sub(x_ref, vec3_scale(z_v_in_b, vec3_dot(z_v_in_b, x_ref)));
    if vec3_norm(x_v_in_b) < 1.0e-6 {
        x_ref = if x_ref[0].abs() > x_ref[1].abs() {
            [0.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        x_v_in_b = vec3_sub(x_ref, vec3_scale(z_v_in_b, vec3_dot(z_v_in_b, x_ref)));
    }
    let mut x_v_in_b = vec3_normalize(x_v_in_b)?;
    let y_v_in_b = vec3_normalize(vec3_cross(z_v_in_b, x_v_in_b))?;
    x_v_in_b = vec3_normalize(vec3_cross(y_v_in_b, z_v_in_b))?;
    let c_b_v_tilt = [
        [x_v_in_b[0], y_v_in_b[0], z_v_in_b[0]],
        [x_v_in_b[1], y_v_in_b[1], z_v_in_b[1]],
        [x_v_in_b[2], y_v_in_b[2], z_v_in_b[2]],
    ];
    let rpy = rot_to_euler_zyx(c_b_v_tilt);
    let dyaw = (yaw_seed_rad - rpy[2])
        .sin()
        .atan2((yaw_seed_rad - rpy[2]).cos());
    let s = dyaw.sin();
    let c = dyaw.cos();
    let c_delta = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
    Some(mat3_mul(c_b_v_tilt, c_delta))
}

fn remove_gravity_axis(q_vb: [f32; 4], accel_b: [f32; 3]) -> [f32; 3] {
    let r = quat_to_rotmat(q_vb);
    let g_hat_b = [-r[0][2], -r[1][2], -r[2][2]];
    let proj = vec3_scale(g_hat_b, vec3_dot(accel_b, g_hat_b));
    vec3_sub(accel_b, proj)
}

fn leveled_xy_axes(gravity_b: [f32; 3]) -> Option<([f32; 3], [f32; 3])> {
    let z_in_b = vec3_scale(vec3_normalize(gravity_b)?, -1.0);
    let mut x_ref = [1.0, 0.0, 0.0];
    let mut x_in_b = vec3_sub(x_ref, vec3_scale(z_in_b, vec3_dot(z_in_b, x_ref)));
    if vec3_norm(x_in_b) < 1.0e-6 {
        x_ref = [0.0, 1.0, 0.0];
        x_in_b = vec3_sub(x_ref, vec3_scale(z_in_b, vec3_dot(z_in_b, x_ref)));
    }
    let x_in_b = vec3_normalize(x_in_b)?;
    let y_in_b = vec3_normalize(vec3_cross(z_in_b, x_in_b))?;
    Some((x_in_b, y_in_b))
}

fn quat_to_rotmat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    let inv = if n2 > 1.0e-9 { 1.0 / n2.sqrt() } else { 1.0 };
    let q0 = q[0] * inv;
    let q1 = q[1] * inv;
    let q2 = q[2] * inv;
    let q3 = q[3] * inv;
    [
        [
            1.0 - 2.0 * (q2 * q2 + q3 * q3),
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * (q1 * q1 + q3 * q3),
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * (q1 * q1 + q2 * q2),
        ],
    ]
}

fn rotmat_to_quat(r: [[f32; 3]; 3]) -> [f32; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let mut q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        [
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        ]
    };
    quat_normalize(&mut q);
    q
}

fn rot_to_euler_zyx(r: [[f32; 3]; 3]) -> [f32; 3] {
    let pitch = (-r[2][0]).clamp(-1.0, 1.0).asin();
    let roll = r[2][1].atan2(r[2][2]);
    let yaw = r[1][0].atan2(r[0][0]);
    [roll, pitch, yaw]
}

fn quat_from_yaw(yaw_rad: f32) -> [f32; 4] {
    let half = 0.5 * yaw_rad;
    [half.cos(), 0.0, 0.0, half.sin()]
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
    let mut q = [1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]];
    quat_normalize(&mut q);
    q
}

fn quat_normalize(q: &mut [f32; 4]) {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2 <= 1.0e-12 {
        *q = [1.0, 0.0, 0.0, 0.0];
        return;
    }
    let inv = 1.0 / n2.sqrt();
    for v in q {
        *v *= inv;
    }
}

fn diag3(v: [f32; 3]) -> [[f32; 3]; 3] {
    [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]]
}

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
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

fn mat3_vec(a: [[f32; 3]; 3], x: [f32; 3]) -> [f32; 3] {
    [
        a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
        a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
        a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
    ]
}

fn transpose3(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn skew3(v: [f32; 3]) -> [[f32; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

fn symmetrize3(a: &mut [[f32; 3]; 3]) {
    for i in 0..3 {
        for j in (i + 1)..3 {
            let avg = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = avg;
            a[j][i] = avg;
        }
    }
}

fn align_obs(q_vb: [f32; 4], gyro_b: [f32; 3], accel_b: [f32; 3]) -> [f32; 6] {
    let c_vb = quat_to_rotmat(q_vb);
    let c_bv = transpose3(c_vb);
    let gyro_v = mat3_vec(c_bv, gyro_b);
    let accel_v = mat3_vec(c_bv, accel_b);
    [
        gyro_v[0], gyro_v[1], gyro_v[2], accel_v[0], accel_v[1], accel_v[2],
    ]
}

fn align_obs_jacobian(q_vb: [f32; 4], gyro_b: [f32; 3], accel_b: [f32; 3]) -> [[f32; 3]; 6] {
    let c_vb = quat_to_rotmat(q_vb);
    let c_bv = transpose3(c_vb);
    let h_gyro = mat3_mul(c_bv, skew3(gyro_b));
    let h_accel = mat3_mul(c_bv, skew3(accel_b));
    [
        h_gyro[0], h_gyro[1], h_gyro[2], h_accel[0], h_accel[1], h_accel[2],
    ]
}

fn inject_small_angle(q_vb: &mut [f32; 4], dtheta: [f32; 3]) {
    *q_vb = quat_mul(quat_from_small_angle(dtheta), *q_vb);
    quat_normalize(q_vb);
}

fn inject_vehicle_yaw(q_vb: &mut [f32; 4], dpsi: f32) {
    *q_vb = quat_mul(*q_vb, quat_from_yaw(dpsi));
    quat_normalize(q_vb);
}

fn apply_update1_masked(
    q_vb: &mut [f32; 4],
    p: &mut [[f32; 3]; 3],
    z: f32,
    obs_idx: usize,
    accel_b: [f32; 3],
    gyro_b: [f32; 3],
    r_var: f32,
    state_mask: [bool; 3],
) -> f32 {
    let obs = align_obs(*q_vb, gyro_b, accel_b);
    let h_full = align_obs_jacobian(*q_vb, gyro_b, accel_b);
    let h = [
        if state_mask[0] {
            h_full[obs_idx][0]
        } else {
            0.0
        },
        if state_mask[1] {
            h_full[obs_idx][1]
        } else {
            0.0
        },
        if state_mask[2] {
            h_full[obs_idx][2]
        } else {
            0.0
        },
    ];
    let y = z - obs[obs_idx];
    let ph = mat3_vec(*p, h);
    let s = vec3_dot(h, ph) + r_var;
    let s_inv = if s.abs() > 1.0e-20 { 1.0 / s } else { 1.0 };
    let k = [ph[0] * s_inv, ph[1] * s_inv, ph[2] * s_inv];
    inject_small_angle(q_vb, [k[0] * y, k[1] * y, k[2] * y]);

    let mut i_minus_kh = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            i_minus_kh[i][j] = -k[i] * h[j];
        }
        i_minus_kh[i][i] += 1.0;
    }
    let mut p_new = mat3_mul(i_minus_kh, *p);
    symmetrize3(&mut p_new);
    *p = p_new;
    y * y * s_inv
}

fn apply_update2_scaled_masked(
    q_vb: &mut [f32; 4],
    p: &mut [[f32; 3]; 3],
    z: [f32; 2],
    obs_idx: [usize; 2],
    accel_b: [f32; 3],
    gyro_b: [f32; 3],
    r_var: [f32; 2],
    state_mask: [bool; 3],
    state_scale: [f32; 3],
) -> f32 {
    let obs = align_obs(*q_vb, gyro_b, accel_b);
    let h_full = align_obs_jacobian(*q_vb, gyro_b, accel_b);
    let mut h0 = [0.0; 3];
    let mut h1 = [0.0; 3];
    for i in 0..3 {
        if state_mask[i] {
            h0[i] = state_scale[i] * h_full[obs_idx[0]][i];
            h1[i] = state_scale[i] * h_full[obs_idx[1]][i];
        }
    }
    let y = [z[0] - obs[obs_idx[0]], z[1] - obs[obs_idx[1]]];
    let ph0 = mat3_vec(*p, h0);
    let ph1 = mat3_vec(*p, h1);
    let s00 = vec3_dot(h0, ph0) + r_var[0];
    let s01 = vec3_dot(h0, ph1);
    let s10 = vec3_dot(h1, ph0);
    let s11 = vec3_dot(h1, ph1) + r_var[1];
    let det = s00 * s11 - s01 * s10;
    let (s_inv00, s_inv01, s_inv10, s_inv11) = if det.abs() > 1.0e-20 {
        let inv_det = 1.0 / det;
        (s11 * inv_det, -s01 * inv_det, -s10 * inv_det, s00 * inv_det)
    } else {
        (1.0, 0.0, 0.0, 1.0)
    };

    let mut k = [[0.0; 2]; 3];
    let mut dtheta = [0.0; 3];
    for i in 0..3 {
        k[i][0] = ph0[i] * s_inv00 + ph1[i] * s_inv10;
        k[i][1] = ph0[i] * s_inv01 + ph1[i] * s_inv11;
        dtheta[i] = k[i][0] * y[0] + k[i][1] * y[1];
    }
    inject_small_angle(q_vb, dtheta);

    let mut i_minus_kh = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            i_minus_kh[i][j] = -(k[i][0] * h0[j] + k[i][1] * h1[j]);
        }
        i_minus_kh[i][i] += 1.0;
    }
    let mut p_new = mat3_mul(i_minus_kh, *p);
    symmetrize3(&mut p_new);
    *p = p_new;

    y[0] * (s_inv00 * y[0] + s_inv01 * y[1]) + y[1] * (s_inv10 * y[0] + s_inv11 * y[1])
}

fn apply_vehicle_yaw_angle(
    q_vb: &mut [f32; 4],
    p: &mut [[f32; 3]; 3],
    angle_err_rad: f32,
    r_var: f32,
) -> f32 {
    let pzz = p[2][2].max(0.0);
    let s = pzz + r_var.max(1.0e-9);
    let k = if s > 1.0e-9 { pzz / s } else { 0.0 };
    inject_vehicle_yaw(q_vb, -k * angle_err_rad);
    p[2][2] = ((1.0 - k) * pzz).max(0.0);
    p[0][2] = 0.0;
    p[2][0] = 0.0;
    p[1][2] = 0.0;
    p[2][1] = 0.0;
    angle_err_rad * angle_err_rad / s
}

fn norm2(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn vec2_normalize(v: [f32; 2]) -> Option<[f32; 2]> {
    let n = norm2(v);
    if !n.is_finite() || n <= 1.0e-8 {
        None
    } else {
        Some([v[0] / n, v[1] / n])
    }
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_norm(v: [f32; 3]) -> f32 {
    vec3_dot(v, v).sqrt()
}

fn vec3_normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let n = vec3_norm(v);
    if !n.is_finite() || n <= 1.0e-6 {
        None
    } else {
        Some([v[0] / n, v[1] / n, v[2] / n])
    }
}

fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn wrap_pi_mod(rad: f32) -> f32 {
    let two_pi = 2.0 * core::f32::consts::PI;
    let mut y = (rad + core::f32::consts::PI) % two_pi;
    if y < 0.0 {
        y += two_pi;
    }
    y - core::f32::consts::PI
}
