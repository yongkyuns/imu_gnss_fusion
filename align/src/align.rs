#![allow(non_snake_case)]

use std::collections::VecDeque;

use crate::horizontal_heading::{
    HorizontalHeadingCueConfig, HorizontalHeadingCueFilter, HorizontalHeadingCueSample,
    HorizontalHeadingTrace,
};
use crate::stationary_mount::bootstrap_vehicle_to_body_from_stationary;
use crate::yaw_startup::{
    YawStartupConfig, YawStartupInitializer, YawStartupSample, YawStartupTrace,
};
use nalgebra::{SMatrix, SVector};

pub const ALIGN_N_STATES: usize = 3;
pub const GRAVITY_MPS2: f32 = 9.80665;

#[derive(Debug, Clone, Copy)]
pub struct AlignConfig {
    pub q_mount_std_rad: [f32; ALIGN_N_STATES],
    pub r_gravity_std_mps2: f32,
    pub r_horiz_heading_std_rad: f32,
    pub r_turn_gyro_std_radps: f32,
    pub turn_gyro_yaw_scale: f32,
    pub r_course_rate_std_radps: f32,
    pub r_lat_std_mps2: f32,
    pub r_long_std_mps2: f32,
    pub long_yaw_scale: f32,
    pub gravity_lpf_alpha: f32,
    pub long_lpf_alpha: f32,
    pub min_speed_mps: f32,
    pub min_turn_rate_radps: f32,
    pub min_lat_acc_mps2: f32,
    pub min_long_acc_mps2: f32,
    pub yaw_dual_resolve_min_speed_mps: f32,
    pub yaw_dual_resolve_min_long_acc_mps2: f32,
    pub yaw_dual_resolve_max_lat_to_long_ratio: f32,
    pub yaw_dual_resolve_min_lat_acc_mps2: f32,
    pub yaw_dual_resolve_max_long_to_lat_ratio: f32,
    pub yaw_dual_resolve_min_windows: usize,
    pub turn_consistency_min_windows: usize,
    pub turn_consistency_min_fraction: f32,
    pub turn_consistency_max_abs_lat_err_mps2: f32,
    pub turn_consistency_max_rel_lat_err: f32,
    pub min_long_sign_stable_windows: usize,
    pub use_unified_yaw_startup: bool,
    pub startup_min_speed_mps: f32,
    pub startup_min_horiz_acc_mps2: f32,
    pub startup_min_long_mps2: f32,
    pub startup_max_lat_to_long_ratio: f32,
    pub startup_min_abs_lat_guard_mps2: f32,
    pub startup_max_course_rate_radps: f32,
    pub startup_min_stable_windows: usize,
    pub startup_min_windows: usize,
    pub startup_max_windows: usize,
    pub startup_min_vector_concentration: f32,
    pub startup_min_sign_agreement: f32,
    pub yaw_seed_std_rad: f32,
    pub yaw_branch_only_std_rad: f32,
    pub max_stationary_gyro_radps: f32,
    pub max_stationary_accel_norm_err_mps2: f32,
    pub use_gravity: bool,
    pub use_horiz_accel_vector_update: bool,
    pub use_turn_gyro: bool,
    pub use_course_rate: bool,
    pub use_lateral_accel: bool,
    pub use_longitudinal_accel: bool,
}

impl Default for AlignConfig {
    fn default() -> Self {
        Self {
            q_mount_std_rad: [
                0.001_f32.to_radians(),
                0.003_f32.to_radians(),
                0.003_f32.to_radians(),
            ],
            r_gravity_std_mps2: 1.28,
            r_horiz_heading_std_rad: 101.0_f32.to_radians(),
            r_turn_gyro_std_radps: 0.1_f32.to_radians(),
            turn_gyro_yaw_scale: 0.0,
            r_course_rate_std_radps: 1.10_f32.to_radians(),
            r_lat_std_mps2: 0.01,
            r_long_std_mps2: 0.3,
            long_yaw_scale: 0.0,
            gravity_lpf_alpha: 0.08,
            long_lpf_alpha: 0.05,
            min_speed_mps: 3.0 / 3.6,
            min_turn_rate_radps: 2.0_f32.to_radians(),
            min_lat_acc_mps2: 0.10,
            min_long_acc_mps2: 0.18,
            yaw_dual_resolve_min_speed_mps: 10.0 / 3.6,
            yaw_dual_resolve_min_long_acc_mps2: 0.3,
            yaw_dual_resolve_max_lat_to_long_ratio: 0.35,
            yaw_dual_resolve_min_lat_acc_mps2: 0.5,
            yaw_dual_resolve_max_long_to_lat_ratio: 0.8,
            yaw_dual_resolve_min_windows: 3,
            turn_consistency_min_windows: 5,
            turn_consistency_min_fraction: 0.8,
            turn_consistency_max_abs_lat_err_mps2: 0.35,
            turn_consistency_max_rel_lat_err: 0.6,
            min_long_sign_stable_windows: 2,
            use_unified_yaw_startup: true,
            startup_min_speed_mps: 10.0 / 3.6,
            startup_min_horiz_acc_mps2: 0.15,
            startup_min_long_mps2: 0.0,
            startup_max_lat_to_long_ratio: 1.0e6,
            startup_min_abs_lat_guard_mps2: 1.0e6,
            startup_max_course_rate_radps: 180.0_f32.to_radians(),
            startup_min_stable_windows: 2,
            startup_min_windows: 6,
            startup_max_windows: 48,
            startup_min_vector_concentration: 0.8,
            startup_min_sign_agreement: 0.8,
            yaw_seed_std_rad: 0.5_f32.to_radians(),
            yaw_branch_only_std_rad: 20.0_f32.to_radians(),
            max_stationary_gyro_radps: 0.8_f32.to_radians(),
            max_stationary_accel_norm_err_mps2: 0.2,
            use_gravity: true,
            use_horiz_accel_vector_update: true,
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
    pub startup_input_xy: Option<[f32; 2]>,
    pub startup_trace: Option<YawStartupTrace>,
    pub after_yaw_seed: Option<[f32; 4]>,
    pub after_branch_resolve: Option<[f32; 4]>,
    pub after_gravity: Option<[f32; 4]>,
    pub after_horiz_accel: Option<[f32; 4]>,
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
    turn_filter: HorizontalHeadingCueFilter,
    yaw_dual: Option<YawDualHypothesis>,
    yaw_startup: YawStartupInitializer,
    turn_consistency: TurnConsistencyGate,
    pub cfg: AlignConfig,
}

#[derive(Debug, Clone)]
struct YawDualHypothesis {
    q_vb: [f32; 4],
    p: [[f32; ALIGN_N_STATES]; ALIGN_N_STATES],
    long_filter: HorizontalHeadingCueFilter,
    turn_filter: HorizontalHeadingCueFilter,
    seeded_from_startup: bool,
    primary_forward_score: f32,
    alt_forward_score: f32,
    forward_windows: usize,
    primary_turn_score: f32,
    alt_turn_score: f32,
    turn_windows: usize,
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
            turn_filter: HorizontalHeadingCueFilter::new(),
            yaw_dual: None,
            yaw_startup: YawStartupInitializer::new(),
            turn_consistency: TurnConsistencyGate::new(),
            cfg,
        }
    }

    pub fn initialize_from_stationary(
        &mut self,
        accel_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
    ) -> Result<(), &'static str> {
        let init = bootstrap_vehicle_to_body_from_stationary(accel_samples_b, yaw_seed_rad)?;
        self.q_vb = quat_from_rotmat(init.c_b_v);
        self.P = diag3([
            0.2_f32.to_radians().powi(2),
            0.2_f32.to_radians().powi(2),
            0.5_f32.to_radians().powi(2),
        ]);
        self.gravity_lp_b = init.mean_accel_b;
        self.long_filter.reset();
        self.turn_filter.reset();
        self.yaw_dual = Some(YawDualHypothesis {
            q_vb: quat_normalize(quat_mul(self.q_vb, quat_yaw_pi())),
            p: self.P,
            long_filter: HorizontalHeadingCueFilter::new(),
            turn_filter: HorizontalHeadingCueFilter::new(),
            seeded_from_startup: false,
            primary_forward_score: 0.0,
            alt_forward_score: 0.0,
            forward_windows: 0,
            primary_turn_score: 0.0,
            alt_turn_score: 0.0,
            turn_windows: 0,
        });
        self.yaw_startup.reset(self.cfg.use_unified_yaw_startup);
        self.turn_consistency.reset();
        Ok(())
    }

    pub fn predict(&mut self, dt: f32) {
        let dt = dt.max(1.0e-3);
        Self::predict_covariance(&mut self.P, &self.cfg, dt);
        if let Some(alt) = &mut self.yaw_dual {
            Self::predict_covariance(&mut alt.p, &self.cfg, dt);
        }
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
        let turn_heading_valid = self.turn_consistency.update(
            &self.cfg,
            turn_valid,
            TurnConsistencySample {
                speed_mps: speed_mid,
                course_rate_radps: course_rate,
                a_lat_mps2: a_lat,
            },
        );
        let horiz_gnss_norm = (a_long * a_long + a_lat * a_lat).sqrt();
        let long_valid = speed_mid > self.cfg.min_speed_mps
            && a_long.abs() > self.cfg.min_long_acc_mps2
            // Use the horizontal-vector angle only when motion is still predominantly
            // longitudinal. In lateral-dominant motion, the instantaneous accel-vector
            // heading is not the forward cue we want from this update.
            && a_lat.abs() < (0.5_f32).max(0.6 * a_long.abs())
            && horiz_gnss_norm > self.cfg.min_long_acc_mps2;
        let startup_cfg = YawStartupConfig {
            enabled: self.cfg.use_unified_yaw_startup,
            alpha: self.cfg.long_lpf_alpha,
            min_speed_mps: self.cfg.startup_min_speed_mps,
            min_horiz_acc_mps2: self.cfg.startup_min_horiz_acc_mps2,
            min_long_mps2: self.cfg.startup_min_long_mps2,
            max_lat_to_long_ratio: self.cfg.startup_max_lat_to_long_ratio,
            min_abs_lat_guard_mps2: self.cfg.startup_min_abs_lat_guard_mps2,
            max_course_rate_radps: self.cfg.startup_max_course_rate_radps,
            min_stable_windows: self.cfg.startup_min_stable_windows,
            min_windows: self.cfg.startup_min_windows,
            max_windows: self.cfg.startup_max_windows,
            min_alignment_score: self.cfg.startup_min_vector_concentration,
        };
        let heading_updates_enabled = !self.yaw_startup.is_active() && self.yaw_dual.is_none();

        if stationary {
            let alpha = self.cfg.gravity_lpf_alpha;
            self.gravity_lp_b = vec3_add(
                vec3_scale(self.gravity_lp_b, 1.0 - alpha),
                vec3_scale(window.mean_accel_b, alpha),
            );
        }

        let gravity_state_mask = if self.yaw_dual.is_some() {
            [true, true, false]
        } else {
            [true, true, true]
        };

        if self.cfg.use_gravity && stationary {
            score += self.apply_update2_masked(
                [0.0, 0.0],
                [3, 4],
                self.gravity_lp_b,
                window.mean_gyro_b,
                [self.cfg.r_gravity_std_mps2.powi(2); 2],
                gravity_state_mask,
            );
            score += self.apply_update1_masked(
                -vec3_norm(self.gravity_lp_b),
                5,
                self.gravity_lp_b,
                window.mean_gyro_b,
                self.cfg.r_gravity_std_mps2.powi(2),
                gravity_state_mask,
            );
            if let Some(alt) = &mut self.yaw_dual {
                score += Self::apply_update2_masked_state(
                    &mut alt.q_vb,
                    &mut alt.p,
                    [0.0, 0.0],
                    [3, 4],
                    self.gravity_lp_b,
                    window.mean_gyro_b,
                    [self.cfg.r_gravity_std_mps2.powi(2); 2],
                    gravity_state_mask,
                );
                score += Self::apply_update1_masked_state(
                    &mut alt.q_vb,
                    &mut alt.p,
                    -vec3_norm(self.gravity_lp_b),
                    5,
                    self.gravity_lp_b,
                    window.mean_gyro_b,
                    self.cfg.r_gravity_std_mps2.powi(2),
                    gravity_state_mask,
                );
            }
            trace.after_gravity = Some(self.q_vb);
        }

        let horiz_imu_norm = (horiz_obs[3] * horiz_obs[3] + horiz_obs[4] * horiz_obs[4]).sqrt();
        let straight_core_valid = long_valid;
        let turn_core_valid = turn_heading_valid
            && speed_mid > (20.0 / 3.6)
            && a_lat.abs() > self.cfg.min_lat_acc_mps2.max(0.7)
            && a_lat.abs() > 1.5 * a_long.abs().max(0.2);
        let horiz_vector_valid = heading_updates_enabled
            && self.cfg.use_horiz_accel_vector_update
            && speed_mid > self.cfg.min_speed_mps
            && horiz_gnss_norm > self.cfg.min_long_acc_mps2
            && horiz_imu_norm > self.cfg.min_long_acc_mps2
            && (straight_core_valid || turn_core_valid);
        let mut horiz_turn_heading_applied = false;
        if horiz_vector_valid {
            let speed_q = ((speed_mid - (10.0 / 3.6)) / (20.0 / 3.6 - 10.0 / 3.6)).clamp(0.0, 1.0);
            let accel_q = ((horiz_gnss_norm.min(horiz_imu_norm) - 0.5) / 1.0).clamp(0.0, 1.0);
            let mode_q = if turn_core_valid {
                let dominance = ((a_lat.abs() / (a_long.abs() + 0.2)) - 1.5) / 1.5;
                let lat_q =
                    ((a_lat.abs() - self.cfg.min_lat_acc_mps2.max(0.7)) / 1.0).clamp(0.0, 1.0);
                lat_q * dominance.clamp(0.0, 1.0)
            } else {
                let lat_ratio = a_lat.abs() / (0.5 + 0.6 * a_long.abs());
                let long_q = ((a_long.abs() - self.cfg.min_long_acc_mps2) / 0.8).clamp(0.0, 1.0);
                long_q * (1.0 - lat_ratio.clamp(0.0, 1.0))
            };
            let quality = (speed_q * accel_q * mode_q).clamp(0.2, 1.0);
            let effective_std = self.cfg.r_horiz_heading_std_rad / quality;
            let cross = horiz_obs[3] * a_lat - horiz_obs[4] * a_long;
            let dot = horiz_obs[3] * a_long + horiz_obs[4] * a_lat;
            let angle_err = cross.atan2(dot);
            score += self.apply_vehicle_yaw_angle(angle_err, effective_std.powi(2));
            trace.after_horiz_accel = Some(self.q_vb);
            horiz_turn_heading_applied = turn_core_valid;
        }

        if turn_valid {
            if self.cfg.use_turn_gyro {
                let turn_gyro_yaw_scale = if heading_updates_enabled && turn_heading_valid {
                    self.cfg.turn_gyro_yaw_scale
                } else {
                    0.0
                };
                score += self.apply_update2_scaled_masked(
                    [0.0, 0.0],
                    [0, 1],
                    window.mean_accel_b,
                    window.mean_gyro_b,
                    [self.cfg.r_turn_gyro_std_radps.powi(2); 2],
                    [true, true, turn_gyro_yaw_scale > 0.0],
                    [1.0, 1.0, turn_gyro_yaw_scale],
                );
                if let Some(alt) = &mut self.yaw_dual {
                    score += Self::apply_update2_scaled_masked_state(
                        &mut alt.q_vb,
                        &mut alt.p,
                        [0.0, 0.0],
                        [0, 1],
                        window.mean_accel_b,
                        window.mean_gyro_b,
                        [self.cfg.r_turn_gyro_std_radps.powi(2); 2],
                        [true, true, turn_gyro_yaw_scale > 0.0],
                        [1.0, 1.0, turn_gyro_yaw_scale],
                    );
                }
                trace.after_turn_gyro = Some(self.q_vb);
            }
            if heading_updates_enabled && turn_heading_valid && self.cfg.use_course_rate {
                score += self.apply_update1(
                    course_rate,
                    2,
                    window.mean_accel_b,
                    window.mean_gyro_b,
                    self.cfg.r_course_rate_std_radps.powi(2),
                );
                trace.after_course_rate = Some(self.q_vb);
            }
            if heading_updates_enabled
                && turn_heading_valid
                && self.cfg.use_lateral_accel
                && !horiz_turn_heading_applied
            {
                score += self.apply_vehicle_yaw_scalar(
                    a_lat,
                    horiz_obs[4],
                    -horiz_obs[3],
                    self.cfg.r_lat_std_mps2.powi(2),
                );
                trace.after_lateral_accel = Some(self.q_vb);
            }
        }

        if self.yaw_startup.is_active() {
            let startup_horiz_xy = leveled_horiz_accel_xy(self.gravity_lp_b, horiz_accel_b)
                .unwrap_or([horiz_accel_b[0], horiz_accel_b[1]]);
            trace.startup_input_xy = Some(startup_horiz_xy);
            let (startup_theta, startup_trace) = self.yaw_startup.update_with_trace(
                startup_cfg,
                YawStartupSample {
                    speed_mps: speed_mid,
                    course_rate_radps: course_rate,
                    horiz_accel_xy: startup_horiz_xy,
                    gnss_long_mps2: a_long,
                    gnss_lat_mps2: a_lat,
                },
            );
            trace.startup_trace = Some(startup_trace);
            if let Some(dpsi) = startup_theta {
                self.seed_yaw_dual_from_startup(dpsi);
                self.turn_consistency.reset();
                trace.after_yaw_seed = Some(self.q_vb);
            } else if self.yaw_startup.timed_out(startup_cfg) {
                self.yaw_startup.reset(false);
            }
        }

        if self.yaw_dual.is_some()
            && !self.yaw_startup.is_active()
            && self.cfg.use_longitudinal_accel
        {
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
            if let Some(alt) = &mut self.yaw_dual {
                let alt_horiz_obs = align_obs(alt.q_vb, window.mean_gyro_b, horiz_accel_b);
                let (cue_alt, alt_long_trace) = alt.long_filter.update_with_trace(
                    HorizontalHeadingCueConfig {
                        alpha: self.cfg.long_lpf_alpha,
                        min_abs_horiz_mps2: self.cfg.min_long_acc_mps2,
                        min_stable_windows: self.cfg.min_long_sign_stable_windows,
                        max_lat_to_long_ratio: 0.8,
                        min_abs_lat_guard_mps2: 0.35,
                    },
                    HorizontalHeadingCueSample {
                        gnss_horiz_mps2: [a_long, a_lat],
                        imu_horiz_mps2: [alt_horiz_obs[3], alt_horiz_obs[4]],
                        base_valid: long_valid,
                    },
                );
                if let (Some(cue_primary), Some(cue_alt)) = (cue, cue_alt) {
                    if let Some(choose_alt) = Self::update_yaw_dual_forward_resolver(
                        &self.cfg,
                        alt,
                        speed_mid,
                        a_long,
                        a_lat,
                        &long_trace,
                        &alt_long_trace,
                    ) {
                        self.resolve_yaw_dual(choose_alt, self.cfg.yaw_seed_std_rad);
                        score += self.apply_vehicle_yaw_angle(
                            self.cfg.long_yaw_scale
                                * if choose_alt {
                                    cue_alt.angle_err_rad
                                } else {
                                    cue_primary.angle_err_rad
                                },
                            self.cfg.r_long_std_mps2.powi(2),
                        );
                        trace.after_longitudinal_accel = Some(self.q_vb);
                        trace.after_branch_resolve = Some(self.q_vb);
                    }
                }
            }
            if self.yaw_dual.is_some() {
                let (turn_cue, turn_trace) = self.turn_filter.update_with_trace(
                    HorizontalHeadingCueConfig {
                        alpha: self.cfg.long_lpf_alpha,
                        min_abs_horiz_mps2: self.cfg.yaw_dual_resolve_min_lat_acc_mps2,
                        min_stable_windows: self.cfg.min_long_sign_stable_windows,
                        max_lat_to_long_ratio: self.cfg.yaw_dual_resolve_max_long_to_lat_ratio,
                        min_abs_lat_guard_mps2: self.cfg.yaw_dual_resolve_min_long_acc_mps2,
                    },
                    HorizontalHeadingCueSample {
                        gnss_horiz_mps2: [a_lat, a_long],
                        imu_horiz_mps2: [horiz_obs[4], horiz_obs[3]],
                        base_valid: turn_valid,
                    },
                );
                if let Some(alt) = &mut self.yaw_dual {
                    let alt_horiz_obs = align_obs(alt.q_vb, window.mean_gyro_b, horiz_accel_b);
                    let (turn_cue_alt, alt_turn_trace) = alt.turn_filter.update_with_trace(
                        HorizontalHeadingCueConfig {
                            alpha: self.cfg.long_lpf_alpha,
                            min_abs_horiz_mps2: self.cfg.yaw_dual_resolve_min_lat_acc_mps2,
                            min_stable_windows: self.cfg.min_long_sign_stable_windows,
                            max_lat_to_long_ratio: self.cfg.yaw_dual_resolve_max_long_to_lat_ratio,
                            min_abs_lat_guard_mps2: self.cfg.yaw_dual_resolve_min_long_acc_mps2,
                        },
                        HorizontalHeadingCueSample {
                            gnss_horiz_mps2: [a_lat, a_long],
                            imu_horiz_mps2: [alt_horiz_obs[4], alt_horiz_obs[3]],
                            base_valid: turn_valid,
                        },
                    );
                    if let (Some(cue_primary), Some(cue_alt)) = (turn_cue, turn_cue_alt) {
                        if let Some(choose_alt) = Self::update_yaw_dual_turn_resolver(
                            &self.cfg,
                            alt,
                            speed_mid,
                            a_long,
                            a_lat,
                            &turn_trace,
                            &alt_turn_trace,
                        ) {
                            let yaw_sigma = if alt.seeded_from_startup {
                                self.cfg.yaw_seed_std_rad
                            } else {
                                self.cfg.yaw_branch_only_std_rad
                            };
                            self.resolve_yaw_dual(choose_alt, yaw_sigma);
                            score += self.apply_vehicle_yaw_angle(
                                if choose_alt {
                                    cue_alt.angle_err_rad
                                } else {
                                    cue_primary.angle_err_rad
                                },
                                self.cfg.r_lat_std_mps2.powi(2),
                            );
                            trace.after_branch_resolve = Some(self.q_vb);
                        }
                    }
                }
            }
        } else if heading_updates_enabled && self.cfg.use_longitudinal_accel {
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
                score += self.apply_vehicle_yaw_angle(
                    self.cfg.long_yaw_scale * cue.angle_err_rad,
                    self.cfg.r_long_std_mps2.powi(2),
                );
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
        Self::apply_update1_masked_state(
            &mut self.q_vb,
            &mut self.P,
            z,
            obs_idx,
            accel_b,
            gyro_b,
            r_var,
            state_mask,
        )
    }

    fn apply_update1_masked_state(
        q_vb: &mut [f32; 4],
        p: &mut [[f32; ALIGN_N_STATES]; ALIGN_N_STATES],
        z: f32,
        obs_idx: usize,
        accel_b: [f32; 3],
        gyro_b: [f32; 3],
        r_var: f32,
        state_mask: [bool; 3],
    ) -> f32 {
        let obs = align_obs(*q_vb, gyro_b, accel_b);
        let H_full = align_obs_jacobian(*q_vb, gyro_b, accel_b);
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
        let P = mat3_to_smatrix(*p);
        let S = H * P * H.transpose() + SMatrix::<f32, 1, 1>::from_diagonal_element(r_var);
        let S_inv = S
            .try_inverse()
            .unwrap_or_else(SMatrix::<f32, 1, 1>::identity);
        let K = P * H.transpose() * S_inv;
        let dtheta = K * y;
        Self::inject_small_angle_state(q_vb, [dtheta[0], dtheta[1], dtheta[2]]);
        let I = SMatrix::<f32, 3, 3>::identity();
        let P_new = (I - K * H) * P;
        *p = smatrix_to_mat3(symmetrize3(P_new));
        (y.transpose() * S_inv * y)[0]
    }

    fn apply_vehicle_yaw_scalar(&mut self, z: f32, h: f32, h_yaw: f32, r_var: f32) -> f32 {
        Self::apply_vehicle_yaw_scalar_state(&mut self.q_vb, &mut self.P, z, h, h_yaw, r_var)
    }

    fn apply_vehicle_yaw_scalar_state(
        q_vb: &mut [f32; 4],
        p: &mut [[f32; ALIGN_N_STATES]; ALIGN_N_STATES],
        z: f32,
        h: f32,
        h_yaw: f32,
        r_var: f32,
    ) -> f32 {
        let y = z - h;
        let pzz = p[2][2].max(0.0);
        let s = h_yaw * h_yaw * pzz + r_var.max(1.0e-9);
        let k = if s > 1.0e-9 { pzz * h_yaw / s } else { 0.0 };
        let dpsi = k * y;
        Self::inject_vehicle_yaw_state(q_vb, dpsi);
        p[2][2] = ((1.0 - k * h_yaw) * pzz).max(0.0);
        p[0][2] = 0.0;
        p[2][0] = 0.0;
        p[1][2] = 0.0;
        p[2][1] = 0.0;
        y * y / s
    }

    fn apply_vehicle_yaw_angle(&mut self, angle_err_rad: f32, r_var: f32) -> f32 {
        Self::apply_vehicle_yaw_angle_state(&mut self.q_vb, &mut self.P, angle_err_rad, r_var)
    }

    fn apply_vehicle_yaw_angle_state(
        q_vb: &mut [f32; 4],
        p: &mut [[f32; ALIGN_N_STATES]; ALIGN_N_STATES],
        angle_err_rad: f32,
        r_var: f32,
    ) -> f32 {
        let pzz = p[2][2].max(0.0);
        let s = pzz + r_var.max(1.0e-9);
        let k = if s > 1.0e-9 { pzz / s } else { 0.0 };
        let dpsi = -k * angle_err_rad;
        Self::inject_vehicle_yaw_state(q_vb, dpsi);
        p[2][2] = ((1.0 - k) * pzz).max(0.0);
        p[0][2] = 0.0;
        p[2][0] = 0.0;
        p[1][2] = 0.0;
        p[2][1] = 0.0;
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
        self.apply_update2_masked(z, obs_idx, accel_b, gyro_b, r_var, [true, true, true])
    }

    fn apply_update2_masked(
        &mut self,
        z: [f32; 2],
        obs_idx: [usize; 2],
        accel_b: [f32; 3],
        gyro_b: [f32; 3],
        r_var: [f32; 2],
        state_mask: [bool; 3],
    ) -> f32 {
        Self::apply_update2_masked_state(
            &mut self.q_vb,
            &mut self.P,
            z,
            obs_idx,
            accel_b,
            gyro_b,
            r_var,
            state_mask,
        )
    }

    fn apply_update2_scaled_masked(
        &mut self,
        z: [f32; 2],
        obs_idx: [usize; 2],
        accel_b: [f32; 3],
        gyro_b: [f32; 3],
        r_var: [f32; 2],
        state_mask: [bool; 3],
        state_scale: [f32; 3],
    ) -> f32 {
        Self::apply_update2_scaled_masked_state(
            &mut self.q_vb,
            &mut self.P,
            z,
            obs_idx,
            accel_b,
            gyro_b,
            r_var,
            state_mask,
            state_scale,
        )
    }

    fn apply_update2_masked_state(
        q_vb: &mut [f32; 4],
        p: &mut [[f32; ALIGN_N_STATES]; ALIGN_N_STATES],
        z: [f32; 2],
        obs_idx: [usize; 2],
        accel_b: [f32; 3],
        gyro_b: [f32; 3],
        r_var: [f32; 2],
        state_mask: [bool; 3],
    ) -> f32 {
        Self::apply_update2_scaled_masked_state(
            q_vb,
            p,
            z,
            obs_idx,
            accel_b,
            gyro_b,
            r_var,
            state_mask,
            [1.0, 1.0, 1.0],
        )
    }

    fn apply_update2_scaled_masked_state(
        q_vb: &mut [f32; 4],
        p: &mut [[f32; ALIGN_N_STATES]; ALIGN_N_STATES],
        z: [f32; 2],
        obs_idx: [usize; 2],
        accel_b: [f32; 3],
        gyro_b: [f32; 3],
        r_var: [f32; 2],
        state_mask: [bool; 3],
        state_scale: [f32; 3],
    ) -> f32 {
        let obs = align_obs(*q_vb, gyro_b, accel_b);
        let H_full = align_obs_jacobian(*q_vb, gyro_b, accel_b);
        let H = SMatrix::<f32, 2, 3>::from_row_slice(&[
            if state_mask[0] {
                state_scale[0] * H_full[obs_idx[0]][0]
            } else {
                0.0
            },
            if state_mask[1] {
                state_scale[1] * H_full[obs_idx[0]][1]
            } else {
                0.0
            },
            if state_mask[2] {
                state_scale[2] * H_full[obs_idx[0]][2]
            } else {
                0.0
            },
            if state_mask[0] {
                state_scale[0] * H_full[obs_idx[1]][0]
            } else {
                0.0
            },
            if state_mask[1] {
                state_scale[1] * H_full[obs_idx[1]][1]
            } else {
                0.0
            },
            if state_mask[2] {
                state_scale[2] * H_full[obs_idx[1]][2]
            } else {
                0.0
            },
        ]);
        let y =
            SVector::<f32, 2>::from_row_slice(&[z[0] - obs[obs_idx[0]], z[1] - obs[obs_idx[1]]]);
        let P = mat3_to_smatrix(*p);
        let R = SMatrix::<f32, 2, 2>::from_diagonal(&SVector::<f32, 2>::from_row_slice(&r_var));
        let S = H * P * H.transpose() + R;
        let S_inv = S
            .try_inverse()
            .unwrap_or_else(SMatrix::<f32, 2, 2>::identity);
        let K = P * H.transpose() * S_inv;
        let dtheta = K * y;
        Self::inject_small_angle_state(q_vb, [dtheta[0], dtheta[1], dtheta[2]]);
        let I = SMatrix::<f32, 3, 3>::identity();
        let P_new = (I - K * H) * P;
        *p = smatrix_to_mat3(symmetrize3(P_new));
        (y.transpose() * S_inv * y)[0]
    }

    fn inject_small_angle(&mut self, dtheta: [f32; 3]) {
        Self::inject_small_angle_state(&mut self.q_vb, dtheta);
    }

    fn inject_small_angle_state(q_vb: &mut [f32; 4], dtheta: [f32; 3]) {
        *q_vb = quat_normalize(quat_mul(quat_from_small_angle(dtheta), *q_vb));
    }

    fn inject_vehicle_yaw(&mut self, dpsi: f32) {
        Self::inject_vehicle_yaw_state(&mut self.q_vb, dpsi);
    }

    fn set_vehicle_heading_in_level_frame(
        q_vb: &mut [f32; 4],
        gravity_b: [f32; 3],
        heading_target: f32,
    ) {
        let current_heading = leveled_forward_heading_xy(*q_vb, gravity_b)
            .unwrap_or_else(|| rot_to_euler_zyx(quat_to_rotmat(*q_vb))[2]);
        let dpsi = wrap_angle_rad(heading_target - current_heading);
        Self::inject_vehicle_yaw_state(q_vb, dpsi);
    }

    fn inject_vehicle_yaw_state(q_vb: &mut [f32; 4], dpsi: f32) {
        *q_vb = quat_normalize(quat_mul(*q_vb, quat_from_yaw(dpsi)));
    }

    fn predict_covariance(
        p: &mut [[f32; ALIGN_N_STATES]; ALIGN_N_STATES],
        cfg: &AlignConfig,
        dt: f32,
    ) {
        p[0][0] += cfg.q_mount_std_rad[0].powi(2) * dt;
        p[1][1] += cfg.q_mount_std_rad[1].powi(2) * dt;
        p[2][2] += cfg.q_mount_std_rad[2].powi(2) * dt;
    }

    fn resolve_yaw_dual(&mut self, choose_alt: bool, yaw_sigma_rad: f32) {
        if let Some(alt) = self.yaw_dual.take() {
            if choose_alt {
                self.q_vb = alt.q_vb;
                self.P = alt.p;
                self.long_filter = alt.long_filter;
                self.turn_filter = alt.turn_filter;
            }
            self.P[2][2] = yaw_sigma_rad.powi(2);
            self.P[0][2] = 0.0;
            self.P[2][0] = 0.0;
            self.P[1][2] = 0.0;
            self.P[2][1] = 0.0;
            self.turn_consistency.reset();
        }
    }

    fn seed_yaw_dual_from_startup(&mut self, heading_target: f32) {
        let mut primary_q = self.q_vb;
        let mut alt_q = self.q_vb;
        Self::set_vehicle_heading_in_level_frame(&mut primary_q, self.gravity_lp_b, heading_target);
        Self::set_vehicle_heading_in_level_frame(
            &mut alt_q,
            self.gravity_lp_b,
            wrap_angle_rad(heading_target + core::f32::consts::PI),
        );

        self.q_vb = primary_q;
        self.P[2][2] = self.cfg.yaw_seed_std_rad.powi(2);
        self.P[0][2] = 0.0;
        self.P[2][0] = 0.0;
        self.P[1][2] = 0.0;
        self.P[2][1] = 0.0;
        self.long_filter.reset();
        self.turn_filter.reset();

        self.yaw_dual = Some(YawDualHypothesis {
            q_vb: alt_q,
            p: self.P,
            long_filter: HorizontalHeadingCueFilter::new(),
            turn_filter: HorizontalHeadingCueFilter::new(),
            seeded_from_startup: true,
            primary_forward_score: 0.0,
            alt_forward_score: 0.0,
            forward_windows: 0,
            primary_turn_score: 0.0,
            alt_turn_score: 0.0,
            turn_windows: 0,
        });
        self.yaw_startup.reset(false);
    }

    fn update_yaw_dual_forward_resolver(
        cfg: &AlignConfig,
        dual: &mut YawDualHypothesis,
        speed_mid: f32,
        a_long: f32,
        a_lat: f32,
        primary_trace: &HorizontalHeadingTrace,
        alt_trace: &HorizontalHeadingTrace,
    ) -> Option<bool> {
        if speed_mid < cfg.yaw_dual_resolve_min_speed_mps
            || a_long.abs() < cfg.yaw_dual_resolve_min_long_acc_mps2
            || a_lat.abs() > cfg.yaw_dual_resolve_max_lat_to_long_ratio * a_long.abs()
        {
            return None;
        }

        let gnss_long = primary_trace.gnss_long_lp_mps2;
        let primary_long = primary_trace.imu_long_lp_mps2;
        let alt_long = alt_trace.imu_long_lp_mps2;
        if gnss_long.abs() < cfg.yaw_dual_resolve_min_long_acc_mps2
            || primary_long.abs() < cfg.yaw_dual_resolve_min_long_acc_mps2
            || alt_long.abs() < cfg.yaw_dual_resolve_min_long_acc_mps2
        {
            return None;
        }

        dual.primary_forward_score += (gnss_long * primary_long).signum();
        dual.alt_forward_score += (gnss_long * alt_long).signum();
        dual.forward_windows += 1;

        if dual.forward_windows < cfg.yaw_dual_resolve_min_windows {
            return None;
        }

        let score_gap = dual.alt_forward_score - dual.primary_forward_score;
        if score_gap.abs() < 1.0 {
            return None;
        }

        Some(score_gap > 0.0)
    }

    fn update_yaw_dual_turn_resolver(
        cfg: &AlignConfig,
        dual: &mut YawDualHypothesis,
        speed_mid: f32,
        a_long: f32,
        a_lat: f32,
        primary_trace: &HorizontalHeadingTrace,
        alt_trace: &HorizontalHeadingTrace,
    ) -> Option<bool> {
        if speed_mid < cfg.yaw_dual_resolve_min_speed_mps
            || a_lat.abs() < cfg.yaw_dual_resolve_min_lat_acc_mps2
            || a_long.abs() > cfg.yaw_dual_resolve_max_long_to_lat_ratio * a_lat.abs()
        {
            return None;
        }

        let gnss_lat = primary_trace.gnss_long_lp_mps2;
        let primary_lat = primary_trace.imu_long_lp_mps2;
        let alt_lat = alt_trace.imu_long_lp_mps2;
        if gnss_lat.abs() < cfg.yaw_dual_resolve_min_lat_acc_mps2
            || primary_lat.abs() < cfg.yaw_dual_resolve_min_lat_acc_mps2
            || alt_lat.abs() < cfg.yaw_dual_resolve_min_lat_acc_mps2
        {
            return None;
        }

        dual.primary_turn_score += (gnss_lat * primary_lat).signum();
        dual.alt_turn_score += (gnss_lat * alt_lat).signum();
        dual.turn_windows += 1;

        if dual.turn_windows < cfg.yaw_dual_resolve_min_windows {
            return None;
        }

        let score_gap = dual.alt_turn_score - dual.primary_turn_score;
        if score_gap.abs() < 1.0 {
            return None;
        }

        Some(score_gap > 0.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct TurnConsistencySample {
    speed_mps: f32,
    course_rate_radps: f32,
    a_lat_mps2: f32,
}

#[derive(Debug, Clone)]
struct TurnConsistencyGate {
    samples: VecDeque<TurnConsistencySample>,
}

impl TurnConsistencyGate {
    fn new() -> Self {
        Self {
            samples: VecDeque::new(),
        }
    }

    fn reset(&mut self) {
        self.samples.clear();
    }

    fn update(
        &mut self,
        cfg: &AlignConfig,
        turn_valid: bool,
        sample: TurnConsistencySample,
    ) -> bool {
        if !turn_valid {
            self.samples.clear();
            return false;
        }

        self.samples.push_back(sample);
        while self.samples.len() > cfg.turn_consistency_min_windows.max(1) {
            self.samples.pop_front();
        }
        if self.samples.len() < cfg.turn_consistency_min_windows.max(1) {
            return false;
        }

        let mut sign_ok = 0usize;
        let mut model_ok = 0usize;
        for s in &self.samples {
            let a_lat_pred = s.speed_mps * s.course_rate_radps;
            if a_lat_pred * s.a_lat_mps2 > 0.0 {
                sign_ok += 1;
            }
            let tol = cfg.turn_consistency_max_abs_lat_err_mps2.max(
                cfg.turn_consistency_max_rel_lat_err * a_lat_pred.abs().max(s.a_lat_mps2.abs()),
            );
            if (s.a_lat_mps2 - a_lat_pred).abs() <= tol {
                model_ok += 1;
            }
        }

        let min_ok = (cfg.turn_consistency_min_fraction.clamp(0.0, 1.0) * self.samples.len() as f32)
            .ceil() as usize;
        sign_ok >= min_ok && model_ok >= min_ok
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

pub fn leveled_horiz_accel_xy(gravity_b: [f32; 3], horiz_accel_b: [f32; 3]) -> Option<[f32; 2]> {
    let (x_in_b, y_in_b) = leveled_xy_axes(gravity_b)?;
    Some([
        vec3_dot(horiz_accel_b, x_in_b),
        vec3_dot(horiz_accel_b, y_in_b),
    ])
}

fn leveled_forward_heading_xy(q_vb: [f32; 4], gravity_b: [f32; 3]) -> Option<f32> {
    let (x_in_b, y_in_b) = leveled_xy_axes(gravity_b)?;
    let c_b_v = quat_to_rotmat(q_vb);
    let x_v_in_b = [c_b_v[0][0], c_b_v[1][0], c_b_v[2][0]];
    Some(vec3_dot(x_v_in_b, y_in_b).atan2(vec3_dot(x_v_in_b, x_in_b)))
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

fn quat_from_yaw(yaw_rad: f32) -> [f32; 4] {
    let half = 0.5 * yaw_rad;
    [half.cos(), 0.0, 0.0, half.sin()]
}

fn quat_yaw_pi() -> [f32; 4] {
    quat_from_yaw(core::f32::consts::PI)
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
