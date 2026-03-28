#![allow(non_snake_case)]

use crate::stationary_mount::{
    StationaryMountBootstrap, bootstrap_vehicle_to_body_from_stationary,
    bootstrap_vehicle_to_body_from_stationary_with_x_ref,
};
use nalgebra::SMatrix;

pub const GRAVITY_MPS2: f32 = 9.80665;
pub const ALIGN_NHC_ERR_STATES: usize = 18;
const IDX_ATT: usize = 0;
const IDX_VEL: usize = 3;
const IDX_MOUNT: usize = 6;
const IDX_BG: usize = 9;
const IDX_BA: usize = 12;
const IDX_POS: usize = 15;

#[derive(Debug, Clone, Copy)]
pub struct AlignNhcConfig {
    pub q_att_std_rad: f32,
    pub q_vel_std_mps: f32,
    pub q_mount_std_rad: f32,
    pub q_bg_std_radps: f32,
    pub q_ba_std_mps2: f32,
    pub r_gnss_vel_std_mps: f32,
    pub r_nhc_std_mps: f32,
    pub r_planar_gyro_std_radps: f32,
    pub min_nhc_speed_mps: f32,
    pub min_planar_speed_mps: f32,
    pub min_planar_yaw_rate_radps: f32,
    pub max_planar_transverse_ratio: f32,
    pub nhc_straight_mount_yaw_scale: f32,
}

impl Default for AlignNhcConfig {
    fn default() -> Self {
        Self {
            q_att_std_rad: 0.01_f32.to_radians(),
            q_vel_std_mps: 0.25,
            q_mount_std_rad: 0.001_f32.to_radians(),
            q_bg_std_radps: 0.00005_f32.to_radians(),
            q_ba_std_mps2: 0.002,
            r_gnss_vel_std_mps: 0.3,
            r_nhc_std_mps: 0.15,
            r_planar_gyro_std_radps: 0.3_f32.to_radians(),
            min_nhc_speed_mps: 3.0 / 3.6,
            min_planar_speed_mps: 3.0 / 3.6,
            min_planar_yaw_rate_radps: 2.0_f32.to_radians(),
            max_planar_transverse_ratio: 0.25,
            nhc_straight_mount_yaw_scale: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AlignNhcTrace {
    pub after_gnss_pos: Option<AlignNhcSnapshot>,
    pub after_gnss_vel: Option<AlignNhcSnapshot>,
    pub after_nhc: Option<AlignNhcSnapshot>,
    pub after_planar_gyro: Option<AlignNhcSnapshot>,
    pub nhc_valid: bool,
    pub planar_gyro_valid: bool,
    pub nhc_residual_vy_mps: f32,
    pub nhc_residual_vz_mps: f32,
    pub planar_gyro_residual_x_radps: f32,
    pub planar_gyro_residual_y_radps: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AlignNhcSnapshot {
    pub q_nb: [f32; 4],
    pub v_n: [f32; 3],
    pub p_n: [f32; 3],
    pub q_vb: [f32; 4],
    pub b_g: [f32; 3],
    pub b_a: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct AlignNhc {
    pub q_nb: [f32; 4],
    pub v_n: [f32; 3],
    pub p_n: [f32; 3],
    pub q_vb: [f32; 4],
    pub b_g: [f32; 3],
    pub b_a: [f32; 3],
    pub P: [[f32; ALIGN_NHC_ERR_STATES]; ALIGN_NHC_ERR_STATES],
    pub cfg: AlignNhcConfig,
}

impl Default for AlignNhc {
    fn default() -> Self {
        Self::new(AlignNhcConfig::default())
    }
}

impl AlignNhc {
    pub fn new(cfg: AlignNhcConfig) -> Self {
        let mut P = [[0.0_f32; ALIGN_NHC_ERR_STATES]; ALIGN_NHC_ERR_STATES];
        for i in 0..3 {
            P[IDX_ATT + i][IDX_ATT + i] = 5.0_f32.to_radians().powi(2);
            P[IDX_VEL + i][IDX_VEL + i] = 4.0_f32.powi(2);
            P[IDX_MOUNT + i][IDX_MOUNT + i] = 5.0_f32.to_radians().powi(2);
            P[IDX_BG + i][IDX_BG + i] = 0.5_f32.to_radians().powi(2);
            P[IDX_BA + i][IDX_BA + i] = 0.5_f32.powi(2);
            P[IDX_POS + i][IDX_POS + i] = 10.0_f32.powi(2);
        }
        Self {
            q_nb: [1.0, 0.0, 0.0, 0.0],
            v_n: [0.0, 0.0, 0.0],
            p_n: [0.0, 0.0, 0.0],
            q_vb: [1.0, 0.0, 0.0, 0.0],
            b_g: [0.0, 0.0, 0.0],
            b_a: [0.0, 0.0, 0.0],
            P,
            cfg,
        }
    }

    pub fn snapshot(&self) -> AlignNhcSnapshot {
        AlignNhcSnapshot {
            q_nb: self.q_nb,
            v_n: self.v_n,
            p_n: self.p_n,
            q_vb: self.q_vb,
            b_g: self.b_g,
            b_a: self.b_a,
        }
    }

    pub fn initialize_from_stationary(
        &mut self,
        accel_samples_b: &[[f32; 3]],
        gyro_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
        q_vb_seed: [f32; 4],
    ) -> Result<(), &'static str> {
        let c_b_v_seed = transpose3x3(quat_to_rotmat(quat_normalize(q_vb_seed)));
        let mount_yaw_seed_rad = c_b_v_seed[1][0].atan2(c_b_v_seed[0][0]);
        let mount = bootstrap_vehicle_to_body_from_stationary(accel_samples_b, mount_yaw_seed_rad)?;
        self.initialize_from_stationary_with_mount(gyro_samples_b, yaw_seed_rad, mount)
    }

    pub fn initialize_from_stationary_with_mount_seed(
        &mut self,
        accel_samples_b: &[[f32; 3]],
        gyro_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
        q_vb_seed: [f32; 4],
    ) -> Result<(), &'static str> {
        self.initialize_from_stationary_with_mount_seed_and_sigma(
            accel_samples_b,
            gyro_samples_b,
            yaw_seed_rad,
            q_vb_seed,
            [10.0_f32.to_radians(); 3],
        )
    }

    pub fn initialize_from_stationary_with_mount_seed_and_sigma(
        &mut self,
        accel_samples_b: &[[f32; 3]],
        gyro_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
        q_vb_seed: [f32; 4],
        mount_sigma_rad: [f32; 3],
    ) -> Result<(), &'static str> {
        if accel_samples_b.is_empty() || gyro_samples_b.is_empty() {
            return Err("stationary initialization requires accel and gyro samples");
        }
        let f_mean = mean3(accel_samples_b);
        let w_mean = mean3(gyro_samples_b);
        let c_b_v = quat_to_rotmat(quat_normalize(q_vb_seed));
        let c_v_b = transpose3x3(c_b_v);

        let (s, c) = yaw_seed_rad.sin_cos();
        let c_n_v = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let c_n_b = mat3_mul(c_n_v, c_v_b);

        self.q_nb = quat_from_rotmat(c_n_b);
        self.v_n = [0.0, 0.0, 0.0];
        self.p_n = [0.0, 0.0, 0.0];
        self.q_vb = quat_from_rotmat(c_v_b);
        self.b_g = w_mean;
        let g_b = mat3_vec(transpose3x3(c_n_b), [0.0, 0.0, GRAVITY_MPS2]);
        self.b_a = vec3_add(f_mean, g_b);

        self.P = [[0.0_f32; ALIGN_NHC_ERR_STATES]; ALIGN_NHC_ERR_STATES];
        for i in 0..3 {
            self.P[IDX_ATT + i][IDX_ATT + i] = 1.0_f32.to_radians().powi(2);
            self.P[IDX_VEL + i][IDX_VEL + i] = 1.0_f32.powi(2);
            self.P[IDX_MOUNT + i][IDX_MOUNT + i] = mount_sigma_rad[i].powi(2);
            self.P[IDX_BG + i][IDX_BG + i] = 0.02_f32.to_radians().powi(2);
            self.P[IDX_BA + i][IDX_BA + i] = 0.05_f32.powi(2);
            self.P[IDX_POS + i][IDX_POS + i] = 10.0_f32.powi(2);
        }
        Ok(())
    }

    pub fn initialize_from_stationary_with_x_ref(
        &mut self,
        accel_samples_b: &[[f32; 3]],
        gyro_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
        q_vb_seed: [f32; 4],
        x_ref: [f32; 3],
    ) -> Result<(), &'static str> {
        let c_b_v_seed = transpose3x3(quat_to_rotmat(quat_normalize(q_vb_seed)));
        let mount_yaw_seed_rad = c_b_v_seed[1][0].atan2(c_b_v_seed[0][0]);
        let mount = bootstrap_vehicle_to_body_from_stationary_with_x_ref(
            accel_samples_b,
            mount_yaw_seed_rad,
            x_ref,
        )?;
        self.initialize_from_stationary_with_mount(gyro_samples_b, yaw_seed_rad, mount)
    }

    fn initialize_from_stationary_with_mount(
        &mut self,
        gyro_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
        mount: StationaryMountBootstrap,
    ) -> Result<(), &'static str> {
        if gyro_samples_b.is_empty() {
            return Err("stationary initialization requires accel and gyro samples");
        }
        let f_mean = mount.mean_accel_b;
        let w_mean = mean3(gyro_samples_b);
        let c_v_b = transpose3x3(mount.c_b_v);
        // Assume the stationary vehicle is locally level; the navigation-to-vehicle attitude is
        // therefore yaw-only at bootstrap, and the navigation-to-body attitude is its product
        // with the mount rotation.
        let (s, c) = yaw_seed_rad.sin_cos();
        let c_n_v = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let c_n_b = mat3_mul(c_n_v, c_v_b);

        self.q_nb = quat_from_rotmat(c_n_b);
        self.v_n = [0.0, 0.0, 0.0];
        self.p_n = [0.0, 0.0, 0.0];
        self.q_vb = quat_from_rotmat(c_v_b);
        self.b_g = w_mean;
        let g_b = mat3_vec(transpose3x3(c_n_b), [0.0, 0.0, GRAVITY_MPS2]);
        self.b_a = vec3_add(f_mean, g_b);

        self.P = [[0.0_f32; ALIGN_NHC_ERR_STATES]; ALIGN_NHC_ERR_STATES];
        for i in 0..3 {
            self.P[IDX_ATT + i][IDX_ATT + i] = 1.0_f32.to_radians().powi(2);
            self.P[IDX_VEL + i][IDX_VEL + i] = 1.0_f32.powi(2);
            self.P[IDX_MOUNT + i][IDX_MOUNT + i] = 10.0_f32.to_radians().powi(2);
            self.P[IDX_BG + i][IDX_BG + i] = 0.02_f32.to_radians().powi(2);
            self.P[IDX_BA + i][IDX_BA + i] = 0.05_f32.powi(2);
            self.P[IDX_POS + i][IDX_POS + i] = 10.0_f32.powi(2);
        }
        Ok(())
    }

    pub fn predict_imu(&mut self, dt: f32, f_m_b: [f32; 3], omega_m_b: [f32; 3]) {
        let dt = dt.max(1.0e-3);
        let omega_b = vec3_sub(omega_m_b, self.b_g);
        self.q_nb = quat_normalize(quat_mul(
            self.q_nb,
            quat_from_small_angle(vec3_scale(omega_b, dt)),
        ));

        let f_b = vec3_sub(f_m_b, self.b_a);
        let c_n_b = quat_to_rotmat(self.q_nb);
        let a_n = vec3_add(mat3_vec(c_n_b, f_b), [0.0, 0.0, GRAVITY_MPS2]);
        self.p_n = vec3_add(
            self.p_n,
            vec3_add(vec3_scale(self.v_n, dt), vec3_scale(a_n, 0.5 * dt * dt)),
        );
        self.v_n = vec3_add(self.v_n, vec3_scale(a_n, dt));

        let mut f = SMatrix::<f32, ALIGN_NHC_ERR_STATES, ALIGN_NHC_ERR_STATES>::zeros();
        let omega_x = skew_smatrix3(omega_b);
        let a_n_x = skew_smatrix3(a_n);
        let c_n_b_m = mat3_to_smatrix(c_n_b);

        f.fixed_view_mut::<3, 3>(IDX_ATT, IDX_ATT).copy_from(&(-omega_x));
        f.fixed_view_mut::<3, 3>(IDX_ATT, IDX_BG)
            .copy_from(&(-SMatrix::<f32, 3, 3>::identity()));
        f.fixed_view_mut::<3, 3>(IDX_VEL, IDX_ATT).copy_from(&(-a_n_x));
        f.fixed_view_mut::<3, 3>(IDX_VEL, IDX_BA).copy_from(&(-c_n_b_m));
        f.fixed_view_mut::<3, 3>(IDX_POS, IDX_VEL)
            .copy_from(&SMatrix::<f32, 3, 3>::identity());

        let phi = SMatrix::<f32, ALIGN_NHC_ERR_STATES, ALIGN_NHC_ERR_STATES>::identity() + f * dt;

        let mut g = SMatrix::<f32, ALIGN_NHC_ERR_STATES, ALIGN_NHC_ERR_STATES>::zeros();
        g.fixed_view_mut::<3, 3>(IDX_ATT, 0)
            .copy_from(&(-SMatrix::<f32, 3, 3>::identity()));
        g.fixed_view_mut::<3, 3>(IDX_VEL, 3).copy_from(&(-c_n_b_m));
        g.fixed_view_mut::<3, 3>(IDX_MOUNT, 6)
            .copy_from(&SMatrix::<f32, 3, 3>::identity());
        g.fixed_view_mut::<3, 3>(IDX_BG, 9)
            .copy_from(&SMatrix::<f32, 3, 3>::identity());
        g.fixed_view_mut::<3, 3>(IDX_BA, 12)
            .copy_from(&SMatrix::<f32, 3, 3>::identity());

        let mut qc = SMatrix::<f32, ALIGN_NHC_ERR_STATES, ALIGN_NHC_ERR_STATES>::zeros();
        for i in 0..3 {
            qc[(IDX_ATT + i, IDX_ATT + i)] = self.cfg.q_att_std_rad.powi(2);
            qc[(IDX_VEL + i, IDX_VEL + i)] = self.cfg.q_vel_std_mps.powi(2);
            qc[(IDX_MOUNT + i, IDX_MOUNT + i)] = self.cfg.q_mount_std_rad.powi(2);
            qc[(IDX_BG + i, IDX_BG + i)] = self.cfg.q_bg_std_radps.powi(2);
            qc[(IDX_BA + i, IDX_BA + i)] = self.cfg.q_ba_std_mps2.powi(2);
        }
        let qd = g * qc * g.transpose() * dt;

        let p = mat_to_smatrix::<ALIGN_NHC_ERR_STATES>(self.P);
        let p_new = phi * p * phi.transpose() + qd;
        self.P = smatrix_to_mat::<ALIGN_NHC_ERR_STATES>(symmetrize::<ALIGN_NHC_ERR_STATES>(p_new));
    }

    pub fn seed_nav_yaw_from_course(&mut self, course_rad: f32, yaw_std_rad: f32) {
        let c_v_b = quat_to_rotmat(self.q_vb);
        let (s, c) = course_rad.sin_cos();
        let c_n_v = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let c_n_b = mat3_mul(c_n_v, c_v_b);
        self.q_nb = quat_from_rotmat(c_n_b);
        let nav_sigma = [
            self.P[IDX_ATT][IDX_ATT].sqrt().max(0.5_f32.to_radians()),
            self.P[IDX_ATT + 1][IDX_ATT + 1]
                .sqrt()
                .max(0.5_f32.to_radians()),
            yaw_std_rad,
        ];
        self.reset_cov_block(IDX_ATT, nav_sigma);
    }

    pub fn seed_mount_yaw_from_course(&mut self, course_rad: f32, yaw_std_rad: f32) {
        let c_n_b = quat_to_rotmat(self.q_nb);
        let (s, c) = course_rad.sin_cos();
        let c_n_v = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let c_v_n = transpose3x3(c_n_v);
        let c_v_b_target = mat3_mul(c_v_n, c_n_b);
        self.q_vb = quat_from_rotmat(c_v_b_target);
        let mount_sigma = [
            self.P[IDX_MOUNT][IDX_MOUNT].sqrt().max(0.5_f32.to_radians()),
            self.P[IDX_MOUNT + 1][IDX_MOUNT + 1]
                .sqrt()
                .max(0.5_f32.to_radians()),
            yaw_std_rad,
        ];
        self.reset_cov_block(IDX_MOUNT, mount_sigma);
    }

    pub fn seed_mount_from_body_to_vehicle(&mut self, q_b_v: [f32; 4], mount_sigma_rad: [f32; 3]) {
        let c_b_v = quat_to_rotmat(quat_normalize(q_b_v));
        let c_v_b = transpose3x3(c_b_v);
        self.q_vb = quat_from_rotmat(c_v_b);
        self.reset_cov_block(IDX_MOUNT, mount_sigma_rad);
    }

    fn reset_cov_block(&mut self, start: usize, sigma_rad: [f32; 3]) {
        for i in start..start + 3 {
            for j in 0..ALIGN_NHC_ERR_STATES {
                self.P[i][j] = 0.0;
                self.P[j][i] = 0.0;
            }
        }
        for i in 0..3 {
            self.P[start + i][start + i] = sigma_rad[i].powi(2);
        }
    }

    pub fn update_gnss_position(&mut self, z_p_n: [f32; 3], r_p_var_n: [f32; 3]) -> f32 {
        let mut score = 0.0_f32;
        for axis in 0..3 {
            let innovation = z_p_n[axis] - self.gnss_position_prediction()[axis];
            let h = self.gnss_position_jacobian_row(axis);
            score += self.apply_measurement1(innovation, &h, r_p_var_n[axis].max(1.0e-4));
        }
        score
    }

    pub fn update_gnss_velocity(&mut self, z_v_n: [f32; 3]) -> f32 {
        let mut score = 0.0_f32;
        for axis in 0..3 {
            let innovation = z_v_n[axis] - self.gnss_velocity_prediction()[axis];
            let h = self.gnss_velocity_jacobian_row(axis);
            score += self.apply_measurement1(innovation, &h, self.cfg.r_gnss_vel_std_mps.powi(2));
        }
        score
    }

    pub fn update_gnss(
        &mut self,
        z_p_n: [f32; 3],
        r_p_var_n: [f32; 3],
        z_v_n: [f32; 3],
    ) -> (f32, AlignNhcTrace) {
        let mut score = 0.0_f32;
        let mut trace = AlignNhcTrace::default();
        score += self.update_gnss_position(z_p_n, r_p_var_n);
        trace.after_gnss_pos = Some(self.snapshot());
        score += self.update_gnss_velocity(z_v_n);
        trace.after_gnss_vel = Some(self.snapshot());
        (score, trace)
    }

    pub fn update_nhc(&mut self, allow_mount_yaw: bool) -> f32 {
        let mut score = 0.0_f32;
        for axis in 0..2 {
            let innovation = -self.nhc_prediction()[axis];
            let h = self.nhc_jacobian_row(axis, allow_mount_yaw);
            score += self.apply_measurement1(innovation, &h, self.cfg.r_nhc_std_mps.powi(2));
        }
        score
    }

    pub fn update_planar_gyro(&mut self, omega_m_b: [f32; 3]) -> f32 {
        let mut score = 0.0_f32;
        for axis in 0..2 {
            let innovation = -self.planar_gyro_prediction(omega_m_b)[axis];
            let h = self.planar_gyro_jacobian_row(axis, omega_m_b);
            score +=
                self.apply_measurement1(innovation, &h, self.cfg.r_planar_gyro_std_radps.powi(2));
        }
        score
    }

    pub fn update_pseudo_measurements(
        &mut self,
        z_v_n: [f32; 3],
        omega_m_b: [f32; 3],
        use_nhc: bool,
        use_planar_gyro: bool,
    ) -> (f32, AlignNhcTrace) {
        let mut score = 0.0_f32;
        let mut trace = AlignNhcTrace::default();

        let nhc_valid = use_nhc && self.nhc_gate(z_v_n);
        let planar_gyro_valid = use_planar_gyro && self.planar_gyro_gate(z_v_n, omega_m_b);
        trace.nhc_valid = nhc_valid;
        if nhc_valid {
            let h_pred = self.nhc_prediction();
            trace.nhc_residual_vy_mps = -h_pred[0];
            trace.nhc_residual_vz_mps = -h_pred[1];
            score += self.update_nhc(planar_gyro_valid);
            trace.after_nhc = Some(self.snapshot());
        }
        trace.planar_gyro_valid = planar_gyro_valid;
        if planar_gyro_valid {
            let h_pred = self.planar_gyro_prediction(omega_m_b);
            trace.planar_gyro_residual_x_radps = -h_pred[0];
            trace.planar_gyro_residual_y_radps = -h_pred[1];
            score += self.update_planar_gyro(omega_m_b);
            trace.after_planar_gyro = Some(self.snapshot());
        }
        (score, trace)
    }

    pub fn update_all(
        &mut self,
        z_p_n: [f32; 3],
        r_p_var_n: [f32; 3],
        z_v_n: [f32; 3],
        omega_m_b: [f32; 3],
        use_nhc: bool,
        use_planar_gyro: bool,
    ) -> (f32, AlignNhcTrace) {
        let (mut score, mut trace) = self.update_gnss(z_p_n, r_p_var_n, z_v_n);
        let (motion_score, motion_trace) =
            self.update_pseudo_measurements(z_v_n, omega_m_b, use_nhc, use_planar_gyro);
        score += motion_score;
        trace.nhc_valid = motion_trace.nhc_valid;
        trace.planar_gyro_valid = motion_trace.planar_gyro_valid;
        trace.nhc_residual_vy_mps = motion_trace.nhc_residual_vy_mps;
        trace.nhc_residual_vz_mps = motion_trace.nhc_residual_vz_mps;
        trace.planar_gyro_residual_x_radps = motion_trace.planar_gyro_residual_x_radps;
        trace.planar_gyro_residual_y_radps = motion_trace.planar_gyro_residual_y_radps;
        trace.after_nhc = motion_trace.after_nhc;
        trace.after_planar_gyro = motion_trace.after_planar_gyro;
        (score, trace)
    }

    pub fn gnss_velocity_prediction(&self) -> [f32; 3] {
        self.v_n
    }

    pub fn gnss_position_prediction(&self) -> [f32; 3] {
        self.p_n
    }

    pub fn nhc_prediction(&self) -> [f32; 2] {
        let c_b_n = transpose3x3(quat_to_rotmat(self.q_nb));
        let v_b = mat3_vec(c_b_n, self.v_n);
        let c_v_b = quat_to_rotmat(self.q_vb);
        let v_v = mat3_vec(c_v_b, v_b);
        [v_v[1], v_v[2]]
    }

    pub fn planar_gyro_prediction(&self, omega_m_b: [f32; 3]) -> [f32; 2] {
        let omega_b = vec3_sub(omega_m_b, self.b_g);
        let c_v_b = quat_to_rotmat(self.q_vb);
        let omega_v = mat3_vec(c_v_b, omega_b);
        [omega_v[0], omega_v[1]]
    }

    pub fn nhc_gate(&self, z_v_n: [f32; 3]) -> bool {
        let speed_h = (z_v_n[0] * z_v_n[0] + z_v_n[1] * z_v_n[1]).sqrt();
        speed_h >= self.cfg.min_nhc_speed_mps
    }

    pub fn planar_gyro_gate(&self, z_v_n: [f32; 3], omega_m_b: [f32; 3]) -> bool {
        let speed_h = (z_v_n[0] * z_v_n[0] + z_v_n[1] * z_v_n[1]).sqrt();
        if speed_h < self.cfg.min_planar_speed_mps {
            return false;
        }
        let omega_b = vec3_sub(omega_m_b, self.b_g);
        let c_v_b = quat_to_rotmat(self.q_vb);
        let omega_v = mat3_vec(c_v_b, omega_b);
        let yaw_abs = omega_v[2].abs();
        if yaw_abs < self.cfg.min_planar_yaw_rate_radps {
            return false;
        }
        let transverse = (omega_v[0] * omega_v[0] + omega_v[1] * omega_v[1]).sqrt();
        transverse <= self.cfg.max_planar_transverse_ratio * yaw_abs
    }

    fn gnss_velocity_jacobian_row(&self, axis: usize) -> [f32; ALIGN_NHC_ERR_STATES] {
        let mut h = [0.0_f32; ALIGN_NHC_ERR_STATES];
        h[IDX_VEL + axis] = 1.0;
        h
    }

    fn gnss_position_jacobian_row(&self, axis: usize) -> [f32; ALIGN_NHC_ERR_STATES] {
        let mut h = [0.0_f32; ALIGN_NHC_ERR_STATES];
        h[IDX_POS + axis] = 1.0;
        h
    }

    fn nhc_jacobian_row(&self, axis: usize, allow_mount_yaw: bool) -> [f32; ALIGN_NHC_ERR_STATES] {
        let c_n_b = quat_to_rotmat(self.q_nb);
        let c_b_n = transpose3x3(c_n_b);
        let c_v_b = quat_to_rotmat(self.q_vb);
        let v_b = mat3_vec(c_b_n, self.v_n);
        let s_yz = [[0.0_f32, 1.0, 0.0], [0.0_f32, 0.0, 1.0]];
        let h_theta_nb = mat2x3_mul3x3(s_yz, mat3_mul(c_v_b, skew3(v_b)));
        let h_vn = mat2x3_mul3x3(s_yz, mat3_mul(c_v_b, c_b_n));
        let h_theta_vb = mat2x3_mul3x3(s_yz, mat3_mul(c_v_b, negate3(skew3(v_b))));
        let mut h = [0.0_f32; ALIGN_NHC_ERR_STATES];
        for c in 0..3 {
            h[IDX_ATT + c] = h_theta_nb[axis][c];
            h[IDX_VEL + c] = h_vn[axis][c];
            h[IDX_MOUNT + c] = h_theta_vb[axis][c];
        }
        if !allow_mount_yaw {
            h[IDX_MOUNT + 2] *= self.cfg.nhc_straight_mount_yaw_scale;
        }
        h
    }

    fn planar_gyro_jacobian_row(
        &self,
        axis: usize,
        omega_m_b: [f32; 3],
    ) -> [f32; ALIGN_NHC_ERR_STATES] {
        let omega_b = vec3_sub(omega_m_b, self.b_g);
        let c_v_b = quat_to_rotmat(self.q_vb);
        let s_xy = [[1.0_f32, 0.0, 0.0], [0.0_f32, 1.0, 0.0]];
        let h_bg = mat2x3_mul3x3(s_xy, negate3(c_v_b));
        let h_theta_vb = mat2x3_mul3x3(s_xy, mat3_mul(c_v_b, negate3(skew3(omega_b))));
        let mut h = [0.0_f32; ALIGN_NHC_ERR_STATES];
        for c in 0..3 {
            h[IDX_MOUNT + c] = h_theta_vb[axis][c];
            h[IDX_BG + c] = h_bg[axis][c];
        }
        h
    }

    fn apply_measurement1(
        &mut self,
        innovation: f32,
        h: &[f32; ALIGN_NHC_ERR_STATES],
        r_var: f32,
    ) -> f32 {
        let p_old = self.P;

        let mut ph = [0.0_f32; ALIGN_NHC_ERR_STATES];
        for i in 0..ALIGN_NHC_ERR_STATES {
            for (j, hj) in h.iter().enumerate() {
                ph[i] += p_old[i][j] * *hj;
            }
        }

        let mut s = r_var.max(1.0e-9);
        for j in 0..ALIGN_NHC_ERR_STATES {
            s += h[j] * ph[j];
        }
        let inv_s = if s > 1.0e-9 { 1.0 / s } else { 0.0 };

        let mut k = [0.0_f32; ALIGN_NHC_ERR_STATES];
        for i in 0..ALIGN_NHC_ERR_STATES {
            k[i] = ph[i] * inv_s;
        }

        let mut dx = [0.0_f32; ALIGN_NHC_ERR_STATES];
        for i in 0..ALIGN_NHC_ERR_STATES {
            dx[i] = k[i] * innovation;
        }
        self.inject_error(dx);

        let mut hp = [0.0_f32; ALIGN_NHC_ERR_STATES];
        for j in 0..ALIGN_NHC_ERR_STATES {
            for kk in 0..ALIGN_NHC_ERR_STATES {
                hp[j] += h[kk] * p_old[kk][j];
            }
        }

        for i in 0..ALIGN_NHC_ERR_STATES {
            for j in 0..ALIGN_NHC_ERR_STATES {
                self.P[i][j] = p_old[i][j] - k[i] * hp[j];
            }
        }
        self.P = symmetrize_mat::<ALIGN_NHC_ERR_STATES>(self.P);
        innovation * innovation * inv_s
    }

    fn inject_error(&mut self, dx: [f32; ALIGN_NHC_ERR_STATES]) {
        self.q_nb = quat_normalize(quat_mul(
            self.q_nb,
            quat_from_small_angle([dx[IDX_ATT], dx[IDX_ATT + 1], dx[IDX_ATT + 2]]),
        ));
        self.v_n = vec3_add(self.v_n, [dx[IDX_VEL], dx[IDX_VEL + 1], dx[IDX_VEL + 2]]);
        self.q_vb = quat_normalize(quat_mul(
            self.q_vb,
            quat_from_small_angle([
                dx[IDX_MOUNT],
                dx[IDX_MOUNT + 1],
                dx[IDX_MOUNT + 2],
            ]),
        ));
        self.b_g = vec3_add(self.b_g, [dx[IDX_BG], dx[IDX_BG + 1], dx[IDX_BG + 2]]);
        self.b_a = vec3_add(self.b_a, [dx[IDX_BA], dx[IDX_BA + 1], dx[IDX_BA + 2]]);
        self.p_n = vec3_add(self.p_n, [dx[IDX_POS], dx[IDX_POS + 1], dx[IDX_POS + 2]]);
    }
}

fn mean3(samples: &[[f32; 3]]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for s in samples {
        out = vec3_add(out, *s);
    }
    vec3_scale(out, 1.0 / samples.len() as f32)
}

fn mat_to_smatrix<const N: usize>(m: [[f32; N]; N]) -> SMatrix<f32, N, N> {
    let mut out = SMatrix::<f32, N, N>::zeros();
    for r in 0..N {
        for c in 0..N {
            out[(r, c)] = m[r][c];
        }
    }
    out
}

fn smatrix_to_mat<const N: usize>(m: SMatrix<f32, N, N>) -> [[f32; N]; N] {
    let mut out = [[0.0_f32; N]; N];
    for r in 0..N {
        for c in 0..N {
            out[r][c] = m[(r, c)];
        }
    }
    out
}

fn symmetrize_mat<const N: usize>(mut m: [[f32; N]; N]) -> [[f32; N]; N] {
    for i in 0..N {
        for j in i..N {
            let temp = 0.5 * (m[i][j] + m[j][i]);
            m[i][j] = temp;
            m[j][i] = temp;
        }
    }
    m
}

fn symmetrize<const N: usize>(m: SMatrix<f32, N, N>) -> SMatrix<f32, N, N> {
    0.5 * (m + m.transpose())
}

fn skew3(v: [f32; 3]) -> [[f32; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

fn negate3(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0_f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = -a[r][c];
        }
    }
    out
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

fn mat3_to_smatrix(a: [[f32; 3]; 3]) -> SMatrix<f32, 3, 3> {
    SMatrix::<f32, 3, 3>::from_row_slice(&[
        a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
    ])
}

fn skew_smatrix3(v: [f32; 3]) -> SMatrix<f32, 3, 3> {
    SMatrix::<f32, 3, 3>::from_row_slice(&[0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0])
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

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0_f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
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

fn transpose3x3(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
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

#[cfg(test)]
fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
fn vec3_norm(v: [f32; 3]) -> f32 {
    (vec3_dot(v, v)).sqrt()
}

#[cfg(test)]
fn vec3_normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let n = vec3_norm(v);
    if n < 1.0e-6 {
        None
    } else {
        Some(vec3_scale(v, 1.0 / n))
    }
}

#[cfg(test)]
mod tests {
    use crate::align::Align;

    use super::*;

    fn finite_difference_jacobian<const M: usize, F>(
        filter: &AlignNhc,
        f: F,
    ) -> SMatrix<f32, M, ALIGN_NHC_ERR_STATES>
    where
        F: Fn(&AlignNhc) -> [f32; M],
    {
        let eps = 1.0e-4_f32;
        let base = f(filter);
        let mut h = SMatrix::<f32, M, ALIGN_NHC_ERR_STATES>::zeros();
        for i in 0..ALIGN_NHC_ERR_STATES {
            let mut perturbed = filter.clone();
            let mut dx = [0.0_f32; ALIGN_NHC_ERR_STATES];
            dx[i] = eps;
            perturbed.inject_error(dx);
            let val = f(&perturbed);
            for r in 0..M {
                h[(r, i)] = (val[r] - base[r]) / eps;
            }
        }
        h
    }

    fn mount_angle_deg(q_est: [f32; 4], q_true: [f32; 4]) -> f32 {
        let dq = quat_mul(q_est, quat_conj(q_true));
        let w = dq[0].clamp(-1.0, 1.0).abs();
        (2.0 * w.acos()).to_degrees()
    }

    fn quat_conj(q: [f32; 4]) -> [f32; 4] {
        [q[0], -q[1], -q[2], -q[3]]
    }

    fn euler_zyx_to_rot(roll: f32, pitch: f32, yaw: f32) -> [[f32; 3]; 3] {
        let (sr, cr) = roll.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let (sy, cy) = yaw.sin_cos();
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    }

    fn euler_from_rot(c: [[f32; 3]; 3]) -> [f32; 3] {
        let pitch = (-c[2][0]).asin();
        let roll = c[2][1].atan2(c[2][2]);
        let yaw = c[1][0].atan2(c[0][0]);
        [roll, pitch, yaw]
    }

    #[derive(Clone, Copy)]
    struct SimSeg {
        duration_s: f32,
        a_long_mps2: f32,
        yaw_rate_radps: f32,
    }

    fn simulate_truth(
        q_vb_true: [f32; 4],
    ) -> (
        Vec<[f32; 3]>,
        Vec<[f32; 3]>,
        Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 4])>,
    ) {
        let dt = 0.01_f32;
        let base = [
            SimSeg {
                duration_s: 10.0,
                a_long_mps2: 0.7,
                yaw_rate_radps: 0.0,
            },
            SimSeg {
                duration_s: 8.0,
                a_long_mps2: 0.0,
                yaw_rate_radps: 10.0_f32.to_radians(),
            },
            SimSeg {
                duration_s: 8.0,
                a_long_mps2: 0.0,
                yaw_rate_radps: -11.0_f32.to_radians(),
            },
            SimSeg {
                duration_s: 8.0,
                a_long_mps2: 0.4,
                yaw_rate_radps: 8.0_f32.to_radians(),
            },
            SimSeg {
                duration_s: 8.0,
                a_long_mps2: -0.4,
                yaw_rate_radps: -8.0_f32.to_radians(),
            },
        ];

        let c_v_b = quat_to_rotmat(q_vb_true);
        let c_b_v = transpose3x3(c_v_b);

        let mut stationary_accel = Vec::new();
        let mut stationary_gyro = Vec::new();
        let mut out = Vec::new();
        let mut yaw = 0.0_f32;
        let mut speed = 0.0_f32;

        let stationary = SimSeg {
            duration_s: 8.0,
            a_long_mps2: 0.0,
            yaw_rate_radps: 0.0,
        };
        let steps = (stationary.duration_s / dt).round() as usize;
        for _ in 0..steps {
            let c_n_v = euler_zyx_to_rot(0.0, 0.0, yaw);
            let f_b = mat3_vec(c_b_v, [0.0, 0.0, -GRAVITY_MPS2]);
            let omega_b = [0.0, 0.0, 0.0];
            let v_n = mat3_vec(c_n_v, [0.0, 0.0, 0.0]);
            let q_nb_true = quat_from_rotmat(mat3_mul(c_n_v, c_v_b));
            stationary_accel.push(f_b);
            stationary_gyro.push(omega_b);
            out.push((f_b, omega_b, v_n, q_nb_true));
        }

        for _ in 0..4 {
            for seg in base {
                let steps = (seg.duration_s / dt).round() as usize;
                for _ in 0..steps {
                    speed = (speed + seg.a_long_mps2 * dt).max(0.0);
                    yaw += seg.yaw_rate_radps * dt;

                    let c_n_v = euler_zyx_to_rot(0.0, 0.0, yaw);
                    let v_v = [speed, 0.0, 0.0];
                    let v_n = mat3_vec(c_n_v, v_v);
                    let a_v = [seg.a_long_mps2, speed * seg.yaw_rate_radps, 0.0];
                    let f_v = [a_v[0], a_v[1], -GRAVITY_MPS2];
                    let omega_v = [0.0, 0.0, seg.yaw_rate_radps];
                    let f_b = mat3_vec(c_b_v, f_v);
                    let omega_b = mat3_vec(c_b_v, omega_v);
                    let q_nb_true = quat_from_rotmat(mat3_mul(c_n_v, c_v_b));
                    out.push((f_b, omega_b, v_n, q_nb_true));
                }
            }
        }
        (stationary_accel, stationary_gyro, out)
    }

    #[test]
    fn nhc_residual_is_zero_for_forward_velocity() {
        let mut f = AlignNhc::default();
        f.q_nb = [1.0, 0.0, 0.0, 0.0];
        f.q_vb = [1.0, 0.0, 0.0, 0.0];
        f.v_n = [12.0, 0.0, 0.0];
        let h = f.nhc_prediction();
        assert!(h[0].abs() < 1.0e-6);
        assert!(h[1].abs() < 1.0e-6);
    }

    #[test]
    fn planar_gyro_residual_is_zero_for_pure_yaw() {
        let mut f = AlignNhc::default();
        f.q_vb = [1.0, 0.0, 0.0, 0.0];
        let h = f.planar_gyro_prediction([0.0, 0.0, 0.4]);
        assert!(h[0].abs() < 1.0e-6);
        assert!(h[1].abs() < 1.0e-6);
    }

    #[test]
    fn gnss_velocity_update_reduces_velocity_error() {
        let mut f = AlignNhc::default();
        f.v_n = [0.0, 0.0, 0.0];
        let before = (f.v_n[0] - 5.0).abs();
        f.update_gnss_velocity([5.0, 0.0, 0.0]);
        let after = (f.v_n[0] - 5.0).abs();
        assert!(after < before);
    }

    #[test]
    fn predict_imu_creates_attitude_velocity_coupling() {
        let mut f = AlignNhc::default();
        f.q_nb = [1.0, 0.0, 0.0, 0.0];
        f.b_a = [0.0, 0.0, 0.0];
        f.b_g = [0.0, 0.0, 0.0];
        f.predict_imu(0.1, [1.5, 0.3, -GRAVITY_MPS2], [0.0, 0.0, 0.2]);
        let theta_v_coupling = f.P[3][1].abs() + f.P[4][0].abs() + f.P[3][2].abs();
        assert!(
            theta_v_coupling > 1.0e-8,
            "expected nonzero attitude/velocity cross covariance, got {}",
            theta_v_coupling
        );
    }

    #[test]
    fn gnss_velocity_update_can_correct_attitude_via_cross_covariance() {
        let mut f = AlignNhc::default();
        f.q_nb = quat_from_rotmat(euler_zyx_to_rot(0.0, 0.0, 15.0_f32.to_radians()));
        f.b_a = [0.0, 0.0, 0.0];
        f.b_g = [0.0, 0.0, 0.0];
        f.v_n = [8.0, 0.0, 0.0];
        for _ in 0..20 {
            f.predict_imu(0.05, [0.5, 0.0, -GRAVITY_MPS2], [0.0, 0.0, 0.0]);
        }
        let yaw_before = euler_from_rot(quat_to_rotmat(f.q_nb))[2];
        let _ = f.update_gnss_velocity([8.0, 2.0, 0.0]);
        let yaw_after = euler_from_rot(quat_to_rotmat(f.q_nb))[2];
        assert!(
            (yaw_after - yaw_before).abs() > 1.0e-6,
            "expected GNSS velocity update to move nav attitude through cross-covariance"
        );
    }

    #[test]
    fn nhc_gate_requires_speed() {
        let f = AlignNhc::default();
        assert!(!f.nhc_gate([0.1, 0.0, 0.0]));
        assert!(f.nhc_gate([2.0, 0.0, 0.0]));
    }

    #[test]
    fn planar_gyro_gate_requires_yaw_dominance() {
        let f = AlignNhc::default();
        assert!(!f.planar_gyro_gate([2.0, 0.0, 0.0], [0.0, 0.0, 0.01]));
        assert!(f.planar_gyro_gate([2.0, 0.0, 0.0], [0.0, 0.0, 0.2]));

        let mut g = AlignNhc::default();
        g.q_vb = quat_from_small_angle([20.0_f32.to_radians(), 0.0, 0.0]);
        assert!(!g.planar_gyro_gate([2.0, 0.0, 0.0], [0.0, 0.0, 0.2]));
    }

    #[test]
    fn nhc_jacobian_matches_finite_difference() {
        let mut f = AlignNhc::default();
        f.q_nb = quat_from_small_angle([
            3.0_f32.to_radians(),
            -2.0_f32.to_radians(),
            15.0_f32.to_radians(),
        ]);
        f.q_vb = quat_from_small_angle([
            2.0_f32.to_radians(),
            -1.0_f32.to_radians(),
            8.0_f32.to_radians(),
        ]);
        f.v_n = [12.0, 1.5, 0.3];
        let h_analytic = SMatrix::<f32, 2, ALIGN_NHC_ERR_STATES>::from_row_slice(
            &[
                f.nhc_jacobian_row(0, true).as_slice(),
                f.nhc_jacobian_row(1, true).as_slice(),
            ]
            .concat(),
        );
        let h_numeric = finite_difference_jacobian(&f, AlignNhc::nhc_prediction);
        let diff = h_analytic - h_numeric;
        assert!(
            diff.amax() < 5.0e-3,
            "amax={} analytic={:?} numeric={:?} diff={:?}",
            diff.amax(),
            h_analytic,
            h_numeric,
            diff
        );
    }

    #[test]
    fn planar_gyro_jacobian_matches_finite_difference() {
        let mut f = AlignNhc::default();
        f.q_vb = quat_from_small_angle([
            2.0_f32.to_radians(),
            -3.0_f32.to_radians(),
            7.0_f32.to_radians(),
        ]);
        f.b_g = [0.01, -0.02, 0.005];
        let omega_m = [0.03, -0.04, 0.35];
        let h_analytic = SMatrix::<f32, 2, ALIGN_NHC_ERR_STATES>::from_row_slice(
            &[
                f.planar_gyro_jacobian_row(0, omega_m).as_slice(),
                f.planar_gyro_jacobian_row(1, omega_m).as_slice(),
            ]
            .concat(),
        );
        let h_numeric = finite_difference_jacobian(&f, |s| s.planar_gyro_prediction(omega_m));
        assert!((h_analytic - h_numeric).amax() < 2.0e-3);
    }

    #[test]
    fn converges_mount_on_simulated_planar_drive() {
        let q_vb_true = quat_from_small_angle([
            2.0_f32.to_radians(),
            -3.0_f32.to_radians(),
            10.0_f32.to_radians(),
        ]);
        let (stationary_accel, stationary_gyro, data) = simulate_truth(q_vb_true);
        let mut f = AlignNhc::default();
        f.initialize_from_stationary(
            &stationary_accel,
            &stationary_gyro,
            0.0,
            [1.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        f.v_n = [0.0, 0.0, 0.0];

        let err0 = mount_angle_deg(f.q_vb, q_vb_true);
        for (_f_b, omega_b, v_n, q_nb_true) in data {
            f.q_nb = q_nb_true;
            f.v_n = v_n;
            f.b_g = [0.0, 0.0, 0.0];
            let _ = f.update_gnss_velocity(v_n);
            let _ = f.update_nhc(true);
            let _ = f.update_planar_gyro(omega_b);
        }
        let err1 = mount_angle_deg(f.q_vb, q_vb_true);
        assert!(
            err1 < err0 - 2.0,
            "mount error did not drop enough: {} -> {}",
            err0,
            err1
        );
        assert!(err1 < 4.0, "final mount error too large: {}", err1);
    }

    #[test]
    fn stationary_init_recovers_mount_down_axis() {
        let q_vb_true = quat_from_small_angle([8.0_f32.to_radians(), -6.0_f32.to_radians(), 0.0]);
        let (stationary_accel, stationary_gyro, _data) = simulate_truth(q_vb_true);
        let mut f = AlignNhc::default();
        f.initialize_from_stationary(
            &stationary_accel,
            &stationary_gyro,
            0.0,
            [1.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        let down_est = mat3_vec(quat_to_rotmat(f.q_vb), [0.0, 0.0, 1.0]);
        let down_true = mat3_vec(quat_to_rotmat(q_vb_true), [0.0, 0.0, 1.0]);
        let cosang = vec3_dot(
            vec3_normalize(down_est).unwrap(),
            vec3_normalize(down_true).unwrap(),
        )
        .clamp(-1.0, 1.0);
        let err_deg = cosang.acos().to_degrees();
        assert!(
            err_deg < 0.5,
            "down-axis bootstrap error too large: {}",
            err_deg
        );
    }

    #[test]
    fn stationary_init_matches_align_shared_mount_bootstrap() {
        let q_vb_true = quat_from_small_angle([8.0_f32.to_radians(), -6.0_f32.to_radians(), 0.0]);
        let (stationary_accel, stationary_gyro, _data) = simulate_truth(q_vb_true);

        let mut align = Align::default();
        align
            .initialize_from_stationary(&stationary_accel, 0.0)
            .unwrap();

        let mut nhc = AlignNhc::default();
        nhc.initialize_from_stationary(
            &stationary_accel,
            &stationary_gyro,
            0.0,
            [1.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        let q_b_v_from_nhc = quat_from_rotmat(transpose3x3(quat_to_rotmat(nhc.q_vb)));
        assert!(mount_angle_deg(q_b_v_from_nhc, align.q_vb) < 1.0e-3);
    }

    #[test]
    fn seed_nav_yaw_from_course_preserves_mount_and_sets_course() {
        let mut f = AlignNhc::default();
        let q_vb_true = quat_from_small_angle([
            8.0_f32.to_radians(),
            -6.0_f32.to_radians(),
            5.0_f32.to_radians(),
        ]);
        let (stationary_accel, stationary_gyro, _data) = simulate_truth(q_vb_true);
        f.initialize_from_stationary(
            &stationary_accel,
            &stationary_gyro,
            0.0,
            [1.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        let q_vb_before = f.q_vb;
        f.seed_nav_yaw_from_course(35.0_f32.to_radians(), 5.0_f32.to_radians());

        let c_n_b = quat_to_rotmat(f.q_nb);
        let c_b_v = transpose3x3(quat_to_rotmat(f.q_vb));
        let yaw = euler_from_rot(mat3_mul(c_n_b, c_b_v))[2];
        let mut yaw_err = yaw - 35.0_f32.to_radians();
        while yaw_err > std::f32::consts::PI {
            yaw_err -= 2.0 * std::f32::consts::PI;
        }
        while yaw_err < -std::f32::consts::PI {
            yaw_err += 2.0 * std::f32::consts::PI;
        }
        assert!(yaw_err.abs() < 1.0e-3);
        assert!(mount_angle_deg(f.q_vb, q_vb_before) < 1.0e-4);
        assert!((f.P[2][2].sqrt().to_degrees() - 5.0).abs() < 1.0e-3);
    }

    #[test]
    fn seed_mount_yaw_from_course_sets_mount_yaw_without_tilt_jump() {
        let mut f = AlignNhc::default();
        let q_vb_true = quat_from_small_angle([
            8.0_f32.to_radians(),
            -6.0_f32.to_radians(),
            35.0_f32.to_radians(),
        ]);
        let (stationary_accel, stationary_gyro, data) = simulate_truth(q_vb_true);
        f.initialize_from_stationary(
            &stationary_accel,
            &stationary_gyro,
            0.0,
            [1.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        let dt = 0.01_f32;
        for (f_m_b, omega_m_b, _, _) in data.iter().take(20) {
            f.predict_imu(dt, *f_m_b, *omega_m_b);
        }

        let c_n_b = quat_to_rotmat(f.q_nb);
        let (s, c) = 35.0_f32.to_radians().sin_cos();
        let c_n_v = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let q_vb_target = quat_from_rotmat(mat3_mul(transpose3x3(c_n_v), c_n_b));
        f.seed_mount_yaw_from_course(35.0_f32.to_radians(), 5.0_f32.to_radians());
        assert!(mount_angle_deg(f.q_vb, q_vb_target) < 1.0e-3);
        assert!((f.P[8][8].sqrt().to_degrees() - 5.0).abs() < 1.0e-3);
    }

    #[test]
    fn nhc_can_reduce_direct_mount_yaw_sensitivity() {
        let mut f = AlignNhc::default();
        f.q_nb = quat_from_small_angle([
            1.0_f32.to_radians(),
            -2.0_f32.to_radians(),
            15.0_f32.to_radians(),
        ]);
        f.q_vb = quat_from_small_angle([
            2.0_f32.to_radians(),
            -1.0_f32.to_radians(),
            8.0_f32.to_radians(),
        ]);
        f.v_n = [12.0, 1.5, 0.3];
        let h_full_0 = f.nhc_jacobian_row(0, true);
        let h_full_1 = f.nhc_jacobian_row(1, true);
        let h_reduced_0 = f.nhc_jacobian_row(0, false);
        let h_reduced_1 = f.nhc_jacobian_row(1, false);
        assert!((h_reduced_0[8] - h_full_0[8] * f.cfg.nhc_straight_mount_yaw_scale).abs() < 1.0e-6);
        assert!((h_reduced_1[8] - h_full_1[8] * f.cfg.nhc_straight_mount_yaw_scale).abs() < 1.0e-6);
    }
}
