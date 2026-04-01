#![allow(non_snake_case)]

use crate::c_api::{CAlign, CAlignState, CAlignUpdateTrace};

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
                0.001_f32.to_radians(),
                0.001_f32.to_radians(),
                0.0001_f32.to_radians(),
            ],
            r_gravity_std_mps2: 0.28,
            r_horiz_heading_std_rad: 1.0_f32.to_radians(),
            r_turn_heading_std_rad: 0.1_f32.to_radians(),
            r_turn_gyro_std_radps: 0.01_f32.to_radians(),
            turn_gyro_yaw_scale: 0.0,
            gravity_lpf_alpha: 0.08,
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
    pub after_horiz_accel: Option<[f32; 4]>,
    pub horiz_angle_err_rad: Option<f32>,
    pub horiz_effective_std_rad: Option<f32>,
    pub horiz_gnss_norm_mps2: Option<f32>,
    pub horiz_imu_norm_mps2: Option<f32>,
    pub horiz_speed_q: Option<f32>,
    pub horiz_accel_q: Option<f32>,
    pub horiz_straight_q: Option<f32>,
    pub horiz_turn_q: Option<f32>,
    pub horiz_dominance_q: Option<f32>,
    pub horiz_turn_core_valid: bool,
    pub horiz_straight_core_valid: bool,
    pub after_turn_gyro: Option<[f32; 4]>,
}

#[derive(Debug, Clone)]
pub struct Align {
    pub q_vb: [f32; 4],
    pub P: [[f32; ALIGN_N_STATES]; ALIGN_N_STATES],
    pub gravity_lp_b: [f32; 3],
    coarse_aligned: bool,
    pub cfg: AlignConfig,
    raw: CAlign,
}

impl Default for Align {
    fn default() -> Self {
        Self::new(AlignConfig::default())
    }
}

impl Align {
    pub fn new(cfg: AlignConfig) -> Self {
        let raw = CAlign::new(cfg);
        let mut out = Self {
            q_vb: [1.0, 0.0, 0.0, 0.0],
            P: diag3([
                20.0_f32.to_radians().powi(2),
                20.0_f32.to_radians().powi(2),
                60.0_f32.to_radians().powi(2),
            ]),
            gravity_lp_b: [0.0, 0.0, -GRAVITY_MPS2],
            coarse_aligned: false,
            cfg,
            raw,
        };
        out.sync_from_c();
        out
    }

    pub fn initialize_from_stationary(
        &mut self,
        accel_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
    ) -> Result<(), &'static str> {
        if self
            .raw
            .initialize_from_stationary(accel_samples_b, yaw_seed_rad)
        {
            self.sync_from_c();
            Ok(())
        } else {
            Err("stationary bootstrap failed")
        }
    }

    pub fn predict(&mut self, _dt: f32) {}

    pub fn update_window(&mut self, window: &AlignWindowSummary) -> f32 {
        self.update_window_with_trace(window).0
    }

    pub fn update_window_with_trace(
        &mut self,
        window: &AlignWindowSummary,
    ) -> (f32, AlignUpdateTrace) {
        let (score, trace) = self.raw.update_window_with_trace(window);
        self.sync_from_c();
        (score, convert_trace(trace))
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

    fn sync_from_c(&mut self) {
        let s = *self.raw.state();
        self.q_vb = s.q_vb;
        self.P = s.p;
        self.gravity_lp_b = s.gravity_lp_b;
        self.coarse_aligned = s.coarse_alignment_ready;
    }

    pub(crate) fn from_c_state(cfg: AlignConfig, state: CAlignState) -> Self {
        let mut out = Self::new(cfg);
        out.q_vb = state.q_vb;
        out.P = state.p;
        out.gravity_lp_b = state.gravity_lp_b;
        out.coarse_aligned = state.coarse_alignment_ready;
        out
    }
}

fn convert_trace(trace: CAlignUpdateTrace) -> AlignUpdateTrace {
    AlignUpdateTrace {
        q_start: trace.q_start,
        coarse_alignment_ready: trace.coarse_alignment_ready,
        after_gravity: trace.after_gravity_valid.then_some(trace.after_gravity),
        after_horiz_accel: trace
            .after_horiz_accel_valid
            .then_some(trace.after_horiz_accel),
        horiz_angle_err_rad: trace
            .horiz_angle_err_rad_valid
            .then_some(trace.horiz_angle_err_rad),
        horiz_effective_std_rad: trace
            .horiz_effective_std_rad_valid
            .then_some(trace.horiz_effective_std_rad),
        horiz_gnss_norm_mps2: trace
            .horiz_gnss_norm_mps2_valid
            .then_some(trace.horiz_gnss_norm_mps2),
        horiz_imu_norm_mps2: trace
            .horiz_imu_norm_mps2_valid
            .then_some(trace.horiz_imu_norm_mps2),
        horiz_speed_q: trace.horiz_speed_q_valid.then_some(trace.horiz_speed_q),
        horiz_accel_q: trace.horiz_accel_q_valid.then_some(trace.horiz_accel_q),
        horiz_straight_q: trace
            .horiz_straight_q_valid
            .then_some(trace.horiz_straight_q),
        horiz_turn_q: trace.horiz_turn_q_valid.then_some(trace.horiz_turn_q),
        horiz_dominance_q: trace
            .horiz_dominance_q_valid
            .then_some(trace.horiz_dominance_q),
        horiz_turn_core_valid: trace.horiz_turn_core_valid,
        horiz_straight_core_valid: trace.horiz_straight_core_valid,
        after_turn_gyro: trace.after_turn_gyro_valid.then_some(trace.after_turn_gyro),
    }
}

pub fn leveled_horiz_accel_xy(gravity_b: [f32; 3], horiz_accel_b: [f32; 3]) -> Option<[f32; 2]> {
    let (x_in_b, y_in_b) = leveled_xy_axes(gravity_b)?;
    Some([
        vec3_dot(horiz_accel_b, x_in_b),
        vec3_dot(horiz_accel_b, y_in_b),
    ])
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

fn rot_to_euler_zyx(r: [[f32; 3]; 3]) -> [f32; 3] {
    let pitch = (-r[2][0]).clamp(-1.0, 1.0).asin();
    let roll = r[2][1].atan2(r[2][2]);
    let yaw = r[1][0].atan2(r[0][0]);
    [roll, pitch, yaw]
}

fn diag3(v: [f32; 3]) -> [[f32; 3]; 3] {
    [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_norm(v: [f32; 3]) -> f32 {
    vec3_dot(v, v).sqrt()
}

fn vec3_normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let n = vec3_norm(v);
    if n <= 1.0e-6 {
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
