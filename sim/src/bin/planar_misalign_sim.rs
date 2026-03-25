use align_rs::{
    AlignMisalign, AlignMisalignConfig, MisalignCoarseConfig, MisalignCoarseSample,
    estimate_mount_yaw_from_tilt,
};
use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use ekf_rs::ekf::{Ekf, GpsData, ImuSample, ekf_fuse_gps, ekf_fuse_vehicle_vel, ekf_predict};
use rust_robotics_algo::control::vehicle::{BicycleModel, TireModel, VehicleParams, VehicleState};
use sim::visualizer::math::{clamp_ekf_biases, quat_rpy_deg};

#[derive(Parser, Debug)]
#[command(name = "planar_misalign_sim")]
struct Args {
    #[arg(long, default_value_t = 60.0)]
    duration_s: f64,
    #[arg(long, default_value_t = 0.01)]
    imu_dt_s: f64,
    #[arg(long, default_value_t = 0.05)]
    gps_dt_s: f64,
    #[arg(long, default_value_t = 5.0)]
    mount_roll_deg: f64,
    #[arg(long, default_value_t = -3.0)]
    mount_pitch_deg: f64,
    #[arg(long, default_value_t = 20.0)]
    mount_yaw_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    gps_pos_std_m: f64,
    #[arg(long, default_value_t = 0.0)]
    gps_vel_std_mps: f64,
    #[arg(long, default_value_t = 0.0)]
    imu_gyro_std_dps: f64,
    #[arg(long, default_value_t = 0.0)]
    imu_accel_std_mps2: f64,
    #[arg(long, default_value_t = 1)]
    noise_seed: u64,
    #[arg(long, default_value_t = 1)]
    mc_runs: usize,
    #[arg(long, default_value_t = 1)]
    mc_seed_start: u64,
    #[arg(long, default_value_t = 0.003)]
    mount_q_tilt_std_deg: f64,
    #[arg(long, default_value_t = 0.003)]
    mount_q_yaw_std_deg: f64,
    #[arg(long, default_value_t = 0.15)]
    nhc_std_mps: f64,
    #[arg(long, default_value_t = 0.3)]
    planar_gyro_std_dps: f64,
    #[arg(long, default_value_t = 0.05)]
    nhc_gain_scale: f64,
    #[arg(long, default_value_t = 0.1)]
    planar_gyro_gain_scale: f64,
    #[arg(long, default_value_t = 0.05)]
    mount_max_tilt_rate_dps: f64,
    #[arg(long, default_value_t = 0.2)]
    mount_max_yaw_rate_dps: f64,
    #[arg(long, default_value_t = 3.0)]
    min_nhc_speed_kph: f64,
    #[arg(long, default_value_t = 3.0)]
    min_planar_speed_kph: f64,
    #[arg(long, default_value_t = 2.0)]
    min_planar_yaw_rate_dps: f64,
    #[arg(long, default_value_t = 0.25)]
    max_planar_transverse_ratio: f64,
    #[arg(long, value_enum, default_value_t = ScenarioKind::Default)]
    scenario: ScenarioKind,
    #[arg(long, default_value_t = 1.0)]
    steer_scale: f64,
    #[arg(long, default_value_t = 1.0)]
    accel_scale: f64,
    #[arg(long, default_value_t = 1.0)]
    grade_scale: f64,
    #[arg(long, default_value_t = 1.0)]
    bank_scale: f64,
    #[arg(long, default_value_t = false)]
    use_true_mount_for_nav: bool,
    #[arg(long, default_value_t = false)]
    disable_misalign_updates: bool,
    #[arg(long, default_value_t = false)]
    allow_ambiguous_startup: bool,
    #[arg(long, value_enum, default_value_t = StartupMode::Heuristic)]
    startup_mode: StartupMode,
    #[arg(long, default_value_t = 5.0)]
    startup_min_rel_gap_pct: f64,
    #[arg(long, default_value_t = false)]
    startup_require_pca_consistency: bool,
    #[arg(long, default_value_t = 8.0)]
    startup_window_s: f64,
    #[arg(long, default_value_t = 30)]
    startup_min_gps: usize,
    #[arg(long, default_value_t = 5.0)]
    startup_coarse_step_deg: f64,
    #[arg(long, default_value_t = 10.0)]
    startup_refine_span_deg: f64,
    #[arg(long, default_value_t = 0.5)]
    startup_refine_step_deg: f64,
    #[arg(long, default_value_t = 1.5)]
    startup_min_pca_anisotropy: f64,
    #[arg(long, default_value_t = 20.0)]
    startup_max_pca_axis_err_deg: f64,
}

#[derive(Clone, Debug)]
struct StartupSeed {
    ekf: Ekf,
    misalign: AlignMisalign,
    course_rad: f32,
    start_t_s: f64,
}

#[derive(Clone, Copy, Debug)]
struct StartupReplayStep {
    t_s: f64,
    imu: ImuSample,
    gyro_b_radps: [f32; 3],
    gps: Option<GpsData>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ScenarioKind {
    Default,
    Aggressive,
    Extended,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum StartupMode {
    Heuristic,
    FixedLag,
    Pca,
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    t_s: f64,
    pos_err_m: f64,
    vel_err_mps: f64,
    body_att_err_deg: f64,
    body_yaw_err_deg: f64,
    veh_att_err_deg: f64,
    veh_yaw_err_deg: f64,
    mount_err_deg: f64,
    mount_yaw_err_deg: f64,
    fwd_err_deg: f64,
    down_err_deg: f64,
}

#[derive(Clone, Copy, Debug)]
struct StartupResolutionDiag {
    best_cost: f32,
    second_cost: f32,
    best_yaw_deg: f32,
    second_yaw_deg: f32,
    mount_axis_deg: f32,
    pca_axis_deg: f32,
    pca_anisotropy: f32,
    pca_axis_err_deg: f32,
    ready: bool,
}

#[derive(Clone, Debug)]
struct RunSummary {
    final_m: Option<Metrics>,
    convergence_t_s: Option<f64>,
    checkpoints: Vec<Metrics>,
    startup_diag: Option<StartupResolutionDiag>,
}

#[derive(Debug, Clone)]
struct NoiseGen {
    state: u64,
    spare: Option<f64>,
}

impl NoiseGen {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
            spare: None,
        }
    }

    fn uniform01(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = self.state >> 11;
        (bits as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn normal(&mut self, std: f64) -> f64 {
        if std <= 0.0 {
            return 0.0;
        }
        if let Some(spare) = self.spare.take() {
            return spare * std;
        }
        let u1 = self.uniform01().clamp(1.0e-12, 1.0 - 1.0e-12);
        let u2 = self.uniform01();
        let mag = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        let z0 = mag * theta.cos();
        let z1 = mag * theta.sin();
        self.spare = Some(z1);
        z0 * std
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.mc_runs <= 1 {
        let summary = run_once(&args, args.noise_seed)?;
        print_run_summary(&args, &summary);
        return Ok(());
    }

    let mut summaries = Vec::with_capacity(args.mc_runs);
    for i in 0..args.mc_runs {
        summaries.push(run_once(&args, args.mc_seed_start + i as u64)?);
    }
    print_mc_summary(&args, &summaries);
    Ok(())
}

fn run_once(args: &Args, noise_seed: u64) -> Result<RunSummary> {
    let q_vb_true = quat_from_rpy(
        args.mount_roll_deg.to_radians(),
        args.mount_pitch_deg.to_radians(),
        args.mount_yaw_deg.to_radians(),
    );
    let c_v_b_true = quat_to_rotmat(q_vb_true);
    let c_b_v_true = transpose3(c_v_b_true);

    let mut model =
        BicycleModel::new(VehicleParams::default(), 0.5).with_tire_model(TireModel::Linear);
    let mut truth = VehicleState::new();
    truth.vx = 0.5;
    let mut noise = NoiseGen::new(noise_seed);
    let mut truth_pos_n = [0.0_f64; 3];
    let mut track_s = 0.0_f64;

    let mut ekf = Ekf::default();
    let mut misalign_cfg = AlignMisalignConfig::default();
    misalign_cfg.q_mount_std_rad = [
        args.mount_q_tilt_std_deg.to_radians() as f32,
        args.mount_q_tilt_std_deg.to_radians() as f32,
        args.mount_q_yaw_std_deg.to_radians() as f32,
    ];
    misalign_cfg.r_nhc_std_mps = args.nhc_std_mps as f32;
    misalign_cfg.r_planar_gyro_std_radps = args.planar_gyro_std_dps.to_radians() as f32;
    misalign_cfg.nhc_gain_scale = args.nhc_gain_scale as f32;
    misalign_cfg.planar_gyro_gain_scale = args.planar_gyro_gain_scale as f32;
    misalign_cfg.max_mount_rate_radps = [
        args.mount_max_tilt_rate_dps.to_radians() as f32,
        args.mount_max_tilt_rate_dps.to_radians() as f32,
        args.mount_max_yaw_rate_dps.to_radians() as f32,
    ];
    misalign_cfg.min_nhc_speed_mps = (args.min_nhc_speed_kph / 3.6) as f32;
    misalign_cfg.min_planar_speed_mps = (args.min_planar_speed_kph / 3.6) as f32;
    misalign_cfg.min_planar_yaw_rate_radps = args.min_planar_yaw_rate_dps.to_radians() as f32;
    misalign_cfg.max_planar_transverse_ratio = args.max_planar_transverse_ratio as f32;
    let mut misalign = AlignMisalign::new(misalign_cfg);
    let mut stationary_accel = Vec::<[f32; 3]>::new();
    let mut initialized = false;
    let mut seed_locked = false;
    let mut startup_seed: Option<StartupSeed> = None;
    let mut startup_steps = Vec::<StartupReplayStep>::new();
    let mut metrics = Vec::<Metrics>::new();
    let mut startup_diag = None;
    let mut startup_rejected = false;

    let n_steps = (args.duration_s / args.imu_dt_s).round() as usize;
    let gps_every = (args.gps_dt_s / args.imu_dt_s).round().max(1.0) as usize;
    for k in 0..n_steps {
        let t_s = k as f64 * args.imu_dt_s;
        let (steer_rad, accel_req) =
            scenario_input(t_s, args.scenario, args.steer_scale, args.accel_scale);
        let prev_truth = truth;
        truth.step(
            &mut model,
            steer_rad as f32,
            accel_req as f32,
            args.imu_dt_s as f32,
        );

        let ds = ((truth.x - prev_truth.x) as f64).hypot((truth.y - prev_truth.y) as f64);
        let s_prev = track_s;
        track_s += ds;
        let s_mid = 0.5 * (s_prev + track_s);

        let (roll_prev, pitch_prev) =
            road_profile_rp(s_prev, args.scenario, args.grade_scale, args.bank_scale);
        let (roll_mid, pitch_mid) =
            road_profile_rp(s_mid, args.scenario, args.grade_scale, args.bank_scale);
        let (roll_curr, pitch_curr) =
            road_profile_rp(track_s, args.scenario, args.grade_scale, args.bank_scale);

        let yaw_prev = prev_truth.yaw as f64;
        let yaw_curr = truth.yaw as f64;
        let yaw_mid = wrap_pi_f64(0.5 * (yaw_prev + yaw_curr));

        let q_nv_prev = quat_from_rpy(roll_prev, pitch_prev, yaw_prev);
        let q_nv_curr = quat_from_rpy(roll_curr, pitch_curr, yaw_curr);
        let q_nv_mid = quat_from_rpy(roll_mid, pitch_mid, yaw_mid);

        let c_n_v_prev = quat_to_rotmat(q_nv_prev);
        let c_n_v_curr = quat_to_rotmat(q_nv_curr);
        let c_n_v_mid = quat_to_rotmat(q_nv_mid);

        let v_v_prev = [prev_truth.vx as f64, prev_truth.vy as f64, 0.0];
        let v_v_curr = [truth.vx as f64, truth.vy as f64, 0.0];
        let v_n_prev = mat3_vec(c_n_v_prev, v_v_prev);
        let v_n_curr = mat3_vec(c_n_v_curr, v_v_curr);
        let v_n_mid = [
            0.5 * (v_n_prev[0] + v_n_curr[0]),
            0.5 * (v_n_prev[1] + v_n_curr[1]),
            0.5 * (v_n_prev[2] + v_n_curr[2]),
        ];
        truth_pos_n = [
            truth_pos_n[0] + v_n_mid[0] * args.imu_dt_s,
            truth_pos_n[1] + v_n_mid[1] * args.imu_dt_s,
            truth_pos_n[2] + v_n_mid[2] * args.imu_dt_s,
        ];

        let a_n = [
            (v_n_curr[0] - v_n_prev[0]) / args.imu_dt_s,
            (v_n_curr[1] - v_n_prev[1]) / args.imu_dt_s,
            (v_n_curr[2] - v_n_prev[2]) / args.imu_dt_s,
        ];
        let g_n = [0.0_f64, 0.0, 9.80665_f64];
        let f_v = mat3_vec(
            transpose3(c_n_v_mid),
            [a_n[0] - g_n[0], a_n[1] - g_n[1], a_n[2] - g_n[2]],
        );
        let roll_dot = (roll_curr - roll_prev) / args.imu_dt_s;
        let pitch_dot = (pitch_curr - pitch_prev) / args.imu_dt_s;
        let yaw_dot = wrap_pi_f64(yaw_curr - yaw_prev) / args.imu_dt_s;
        let omega_v =
            euler_zyx_rates_to_body_rates(roll_mid, pitch_mid, roll_dot, pitch_dot, yaw_dot);
        let accel_b = mat3_vec(c_b_v_true, f_v);
        let gyro_b = mat3_vec(c_b_v_true, omega_v);

        let gyro_b_meas = [
            gyro_b[0] + noise.normal(args.imu_gyro_std_dps.to_radians()),
            gyro_b[1] + noise.normal(args.imu_gyro_std_dps.to_radians()),
            gyro_b[2] + noise.normal(args.imu_gyro_std_dps.to_radians()),
        ];
        let accel_b_meas = [
            accel_b[0] + noise.normal(args.imu_accel_std_mps2),
            accel_b[1] + noise.normal(args.imu_accel_std_mps2),
            accel_b[2] + noise.normal(args.imu_accel_std_mps2),
        ];
        if !initialized && t_s <= 3.0 {
            stationary_accel.push([
                accel_b_meas[0] as f32,
                accel_b_meas[1] as f32,
                accel_b_meas[2] as f32,
            ]);
        }
        if !initialized && t_s > 3.0 {
            misalign
                .initialize_from_stationary_with_x_ref(&stationary_accel, 0.0, [1.0, 0.0, 0.0])
                .map_err(|e| anyhow!(e))?;
            initialized = true;
        }
        let imu = ImuSample {
            dax: (gyro_b_meas[0] * args.imu_dt_s) as f32,
            day: (gyro_b_meas[1] * args.imu_dt_s) as f32,
            daz: (gyro_b_meas[2] * args.imu_dt_s) as f32,
            dvx: (accel_b_meas[0] * args.imu_dt_s) as f32,
            dvy: (accel_b_meas[1] * args.imu_dt_s) as f32,
            dvz: (accel_b_meas[2] * args.imu_dt_s) as f32,
            dt: args.imu_dt_s as f32,
        };
        ekf_predict(&mut ekf, &imu, None);
        clamp_ekf_biases(&mut ekf, args.imu_dt_s);
        if initialized && (args.use_true_mount_for_nav || seed_locked) {
            let q_vb_fuse = if args.use_true_mount_for_nav {
                f64_to_f32_quat(q_vb_true)
            } else {
                misalign.q_vb
            };
            ekf_fuse_vehicle_vel(&mut ekf, q_vb_fuse, 100.0);
            clamp_ekf_biases(&mut ekf, args.imu_dt_s);
        }

        let mut gps_opt = None;
        let vel_n = v_n_curr;
        let pos_n = truth_pos_n;
        if k % gps_every == 0 {
            let gps = GpsData {
                pos_n: (pos_n[0] + noise.normal(args.gps_pos_std_m)) as f32,
                pos_e: (pos_n[1] + noise.normal(args.gps_pos_std_m)) as f32,
                pos_d: (pos_n[2] + noise.normal(args.gps_pos_std_m)) as f32,
                vel_n: (vel_n[0] + noise.normal(args.gps_vel_std_mps)) as f32,
                vel_e: (vel_n[1] + noise.normal(args.gps_vel_std_mps)) as f32,
                vel_d: (vel_n[2] + noise.normal(args.gps_vel_std_mps)) as f32,
                R_POS_N: args.gps_pos_std_m.max(1.0e-3).powi(2) as f32,
                R_POS_E: args.gps_pos_std_m.max(1.0e-3).powi(2) as f32,
                R_POS_D: args.gps_pos_std_m.max(1.0e-3).powi(2) as f32,
                R_VEL_N: args.gps_vel_std_mps.max(1.0e-3).powi(2) as f32,
                R_VEL_E: args.gps_vel_std_mps.max(1.0e-3).powi(2) as f32,
                R_VEL_D: args.gps_vel_std_mps.max(1.0e-3).powi(2) as f32,
            };
            ekf_fuse_gps(&mut ekf, &gps);
            clamp_ekf_biases(&mut ekf, args.imu_dt_s);
            gps_opt = Some(gps);
        }

        if startup_seed.is_some() && !seed_locked {
            startup_steps.push(StartupReplayStep {
                t_s,
                imu,
                gyro_b_radps: [
                    gyro_b_meas[0] as f32,
                    gyro_b_meas[1] as f32,
                    gyro_b_meas[2] as f32,
                ],
                gps: gps_opt,
            });
        }

        if initialized {
            let speed_h = vel_n[0].hypot(vel_n[1]);
            if args.use_true_mount_for_nav && !seed_locked && speed_h >= 5.0 / 3.6 {
                let q_vb_seed = if args.use_true_mount_for_nav {
                    f64_to_f32_quat(q_vb_true)
                } else {
                    misalign.q_vb
                };
                seed_ekf_from_vehicle_course_and_mount(
                    &mut ekf,
                    q_vb_seed,
                    vel_n[1].atan2(vel_n[0]) as f32,
                );
                let course_rad = vel_n[1].atan2(vel_n[0]) as f32;
                let q_nb_seed = [ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3];
                let mut nav_seed = misalign.clone();
                nav_seed.seed_mount_from_nav_course_full(
                    q_nb_seed,
                    course_rad,
                    2.0_f32.to_radians(),
                    10.0_f32.to_radians(),
                );
                let down_curr = quat_rotate_f64(f32_to_f64_quat(misalign.q_vb), [0.0, 0.0, 1.0]);
                let down_seed = quat_rotate_f64(f32_to_f64_quat(nav_seed.q_vb), [0.0, 0.0, 1.0]);
                if angle_between_unit_deg(down_curr, down_seed) <= 10.0 {
                    misalign = nav_seed;
                }
                seed_locked = true;
            }

            if !args.use_true_mount_for_nav
                && !seed_locked
                && !startup_rejected
                && startup_seed.is_none()
                && speed_h >= 5.0 / 3.6
            {
                startup_seed = Some(StartupSeed {
                    ekf: ekf.clone(),
                    misalign: misalign.clone(),
                    course_rad: vel_n[1].atan2(vel_n[0]) as f32,
                    start_t_s: t_s,
                });
                startup_steps.clear();
            }

            if !args.use_true_mount_for_nav && !seed_locked {
                if let Some(seed) = startup_seed.clone() {
                    let elapsed_s = t_s - seed.start_t_s;
                    let gps_count = startup_steps.iter().filter(|s| s.gps.is_some()).count();
                    let startup_ready = match args.startup_mode {
                        StartupMode::Heuristic => elapsed_s >= 3.0 && gps_count >= 10,
                        StartupMode::FixedLag => {
                            elapsed_s >= args.startup_window_s && gps_count >= args.startup_min_gps
                        }
                        StartupMode::Pca => elapsed_s >= 3.0 && gps_count >= 10,
                    };
                    if startup_ready {
                        let solved = match args.startup_mode {
                            StartupMode::Heuristic => resolve_joint_startup_state_pca_local(
                                &seed.ekf,
                                &seed.misalign,
                                seed.course_rad,
                                &startup_steps,
                                args.startup_refine_span_deg as f32,
                                args.startup_refine_step_deg as f32,
                                args.gps_dt_s as f32,
                            ),
                            StartupMode::FixedLag => resolve_joint_startup_state_fixed_lag(
                                &seed.ekf,
                                &seed.misalign,
                                seed.course_rad,
                                &startup_steps,
                                args.startup_coarse_step_deg as f32,
                                args.startup_refine_span_deg as f32,
                                args.startup_refine_step_deg as f32,
                                args.gps_dt_s as f32,
                            ),
                            StartupMode::Pca => resolve_joint_startup_state_pca(
                                &seed.ekf,
                                &seed.misalign,
                                seed.course_rad,
                                &startup_steps,
                                args.gps_dt_s as f32,
                            ),
                        };
                        if let Some((ekf_resolved, misalign_resolved, diag)) = solved {
                            let rel_gap = (diag.second_cost - diag.best_cost) as f64
                                / diag.best_cost.max(1.0e-6) as f64;
                            let pca_ready = diag.pca_anisotropy as f64
                                >= args.startup_min_pca_anisotropy
                                && diag.pca_axis_err_deg as f64
                                    <= args.startup_max_pca_axis_err_deg;
                            let ready = match args.startup_mode {
                                StartupMode::Pca => {
                                    args.allow_ambiguous_startup
                                        || (pca_ready
                                            && rel_gap >= args.startup_min_rel_gap_pct * 0.01)
                                }
                                _ => {
                                    let pca_consistency_ok =
                                        !args.startup_require_pca_consistency || pca_ready;
                                    args.allow_ambiguous_startup
                                        || (rel_gap >= args.startup_min_rel_gap_pct * 0.01
                                            && pca_consistency_ok)
                                }
                            };
                            startup_diag = Some(StartupResolutionDiag { ready, ..diag });
                            if ready {
                                ekf = ekf_resolved;
                                misalign = misalign_resolved;
                                seed_locked = true;
                            } else {
                                startup_rejected = true;
                            }
                        }
                        startup_seed = None;
                        startup_steps.clear();
                    }
                }
            }

            if seed_locked && gps_opt.is_some() {
                let q_nb_est = [ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3];
                let v_n_est = [ekf.state.vn, ekf.state.ve, ekf.state.vd];
                let dt_safe = args.imu_dt_s.max(1.0e-3) as f32;
                let omega_b_corr = [
                    gyro_b_meas[0] as f32 - ekf.state.dax_b / dt_safe,
                    gyro_b_meas[1] as f32 - ekf.state.day_b / dt_safe,
                    gyro_b_meas[2] as f32 - ekf.state.daz_b / dt_safe,
                ];
                let accel_b_corr = [
                    accel_b_meas[0] as f32 - ekf.state.dvx_b / dt_safe,
                    accel_b_meas[1] as f32 - ekf.state.dvy_b / dt_safe,
                    accel_b_meas[2] as f32 - ekf.state.dvz_b / dt_safe,
                ];
                if !args.disable_misalign_updates {
                    let _ = misalign.update_all(
                        args.gps_dt_s as f32,
                        q_nb_est,
                        v_n_est,
                        omega_b_corr,
                        accel_b_corr,
                    );
                }

                let q_nb_true = quat_mul(q_nv_curr, q_vb_true);
                let q_nv_est = quat_mul(
                    f32_to_f64_quat(q_nb_est),
                    quat_conj(f32_to_f64_quat(misalign.q_vb)),
                );
                let q_nv_true = q_nv_curr;
                let pos_err = ((ekf.state.pn as f64 - pos_n[0]).powi(2)
                    + (ekf.state.pe as f64 - pos_n[1]).powi(2)
                    + (ekf.state.pd as f64 - pos_n[2]).powi(2))
                .sqrt();
                let vel_err = ((ekf.state.vn as f64 - vel_n[0]).powi(2)
                    + (ekf.state.ve as f64 - vel_n[1]).powi(2)
                    + (ekf.state.vd as f64 - vel_n[2]).powi(2))
                .sqrt();
                let (_, _, body_yaw_est_deg) =
                    quat_rpy_deg(q_nb_est[0], q_nb_est[1], q_nb_est[2], q_nb_est[3]);
                let (_, _, body_yaw_true_deg) = quat_rpy_deg(
                    q_nb_true[0] as f32,
                    q_nb_true[1] as f32,
                    q_nb_true[2] as f32,
                    q_nb_true[3] as f32,
                );
                let (_, _, veh_yaw_est_deg) = quat_rpy_deg(
                    q_nv_est[0] as f32,
                    q_nv_est[1] as f32,
                    q_nv_est[2] as f32,
                    q_nv_est[3] as f32,
                );
                let (_, _, veh_yaw_true_deg) = quat_rpy_deg(
                    q_nv_true[0] as f32,
                    q_nv_true[1] as f32,
                    q_nv_true[2] as f32,
                    q_nv_true[3] as f32,
                );
                let (_, _, mount_yaw_est_deg) = quat_rpy_deg(
                    misalign.q_vb[0],
                    misalign.q_vb[1],
                    misalign.q_vb[2],
                    misalign.q_vb[3],
                );
                let (_, _, mount_yaw_true_deg) = quat_rpy_deg(
                    q_vb_true[0] as f32,
                    q_vb_true[1] as f32,
                    q_vb_true[2] as f32,
                    q_vb_true[3] as f32,
                );
                metrics.push(Metrics {
                    t_s,
                    pos_err_m: pos_err,
                    vel_err_mps: vel_err,
                    body_att_err_deg: quat_angle_deg(f32_to_f64_quat(q_nb_est), q_nb_true),
                    body_yaw_err_deg: wrap_deg_180((body_yaw_est_deg - body_yaw_true_deg) as f32)
                        .abs() as f64,
                    veh_att_err_deg: quat_angle_deg(q_nv_est, q_nv_true),
                    veh_yaw_err_deg: wrap_deg_180((veh_yaw_est_deg - veh_yaw_true_deg) as f32).abs()
                        as f64,
                    mount_err_deg: quat_angle_deg(f32_to_f64_quat(misalign.q_vb), q_vb_true),
                    mount_yaw_err_deg: wrap_deg_180((mount_yaw_est_deg - mount_yaw_true_deg) as f32)
                        .abs() as f64,
                    fwd_err_deg: axis_err_deg(
                        f32_to_f64_quat(misalign.q_vb),
                        q_vb_true,
                        [1.0, 0.0, 0.0],
                    ),
                    down_err_deg: axis_err_deg(
                        f32_to_f64_quat(misalign.q_vb),
                        q_vb_true,
                        [0.0, 0.0, 1.0],
                    ),
                });
            }
        }
    }

    let final_m = metrics.last().copied();
    let convergence_t_s = if metrics.is_empty() {
        None
    } else {
        convergence_time(&metrics, 1.0, 1.0, 0.5, 0.5)
    };
    let checkpoints = [5.0_f64, 10.0, 20.0, 40.0, 60.0]
        .iter()
        .filter_map(|t_checkpoint| {
            metrics
                .iter()
                .min_by(|a, b| {
                    (a.t_s - t_checkpoint)
                        .abs()
                        .total_cmp(&(b.t_s - t_checkpoint).abs())
                })
                .copied()
        })
        .collect();
    Ok(RunSummary {
        final_m,
        convergence_t_s,
        checkpoints,
        startup_diag,
    })
}

fn scenario_input(
    t_s: f64,
    scenario: ScenarioKind,
    steer_scale: f64,
    accel_scale: f64,
) -> (f64, f64) {
    let (steer_rad, accel_req) = match scenario {
        ScenarioKind::Default => {
            if t_s < 3.0 {
                (0.0, 0.0)
            } else if t_s < 10.0 {
                (0.0, 1.5)
            } else if t_s < 22.0 {
                (6.0_f64.to_radians() * ((t_s - 10.0) * 0.5).sin(), 0.3)
            } else if t_s < 32.0 {
                (-5.0_f64.to_radians() * ((t_s - 22.0) * 0.7).sin(), 0.0)
            } else if t_s < 45.0 {
                (4.0_f64.to_radians() * ((t_s - 32.0) * 0.8).sin(), -0.2)
            } else {
                (0.0, -0.8)
            }
        }
        ScenarioKind::Aggressive => {
            if t_s < 3.0 {
                (0.0, 0.0)
            } else if t_s < 9.0 {
                (0.0, 2.0)
            } else if t_s < 20.0 {
                (10.0_f64.to_radians() * ((t_s - 9.0) * 0.8).sin(), 0.8)
            } else if t_s < 32.0 {
                (-9.0_f64.to_radians() * ((t_s - 20.0) * 1.0).sin(), -0.4)
            } else if t_s < 48.0 {
                (
                    8.0_f64.to_radians() * ((t_s - 32.0) * 1.1).sin(),
                    0.6 * ((t_s - 32.0) * 0.5).sin(),
                )
            } else {
                (0.0, -1.2)
            }
        }
        ScenarioKind::Extended => {
            if t_s < 3.0 {
                (0.0, 0.0)
            } else {
                let tau = t_s - 3.0;
                let phase = tau.rem_euclid(48.0);
                if phase < 8.0 {
                    (0.0, 1.8)
                } else if phase < 20.0 {
                    (
                        9.0_f64.to_radians() * ((phase - 8.0) * 0.7).sin(),
                        0.6 + 0.5 * ((phase - 8.0) * 0.45).sin(),
                    )
                } else if phase < 30.0 {
                    (
                        -8.0_f64.to_radians() * ((phase - 20.0) * 0.9).sin(),
                        -0.4 + 0.4 * ((phase - 20.0) * 0.5).sin(),
                    )
                } else if phase < 40.0 {
                    (
                        6.0_f64.to_radians() * ((phase - 30.0) * 1.1).sin(),
                        0.7 * ((phase - 30.0) * 0.8).sin(),
                    )
                } else {
                    (0.0, -0.8)
                }
            }
        }
    };
    (steer_rad * steer_scale, accel_req * accel_scale)
}

fn road_profile_rp(
    s_m: f64,
    scenario: ScenarioKind,
    grade_scale: f64,
    bank_scale: f64,
) -> (f64, f64) {
    let (bank_deg, grade_deg) = match scenario {
        ScenarioKind::Default => (
            2.0 * (2.0 * std::f64::consts::PI * s_m / 140.0).sin()
                + 0.8 * (2.0 * std::f64::consts::PI * s_m / 65.0).sin(),
            3.0 * (2.0 * std::f64::consts::PI * s_m / 180.0).sin()
                + 1.2 * (2.0 * std::f64::consts::PI * s_m / 90.0).sin(),
        ),
        ScenarioKind::Aggressive => (
            4.0 * (2.0 * std::f64::consts::PI * s_m / 90.0).sin()
                + 1.5 * (2.0 * std::f64::consts::PI * s_m / 45.0).sin(),
            6.0 * (2.0 * std::f64::consts::PI * s_m / 120.0).sin()
                + 2.0 * (2.0 * std::f64::consts::PI * s_m / 55.0).sin(),
        ),
        ScenarioKind::Extended => (
            3.5 * (2.0 * std::f64::consts::PI * s_m / 110.0).sin()
                + 1.2 * (2.0 * std::f64::consts::PI * s_m / 42.0).sin(),
            5.0 * (2.0 * std::f64::consts::PI * s_m / 150.0).sin()
                + 1.5 * (2.0 * std::f64::consts::PI * s_m / 70.0).sin(),
        ),
    };
    (
        (bank_deg * bank_scale).to_radians(),
        (grade_deg * grade_scale).to_radians(),
    )
}

fn euler_zyx_rates_to_body_rates(
    roll_rad: f64,
    pitch_rad: f64,
    roll_dot: f64,
    pitch_dot: f64,
    yaw_dot: f64,
) -> [f64; 3] {
    let (sr, cr) = roll_rad.sin_cos();
    let (sp, cp) = pitch_rad.sin_cos();
    [
        roll_dot - yaw_dot * sp,
        pitch_dot * cr + yaw_dot * sr * cp,
        -pitch_dot * sr + yaw_dot * cr * cp,
    ]
}

fn resolve_joint_startup_state(
    base_ekf: &Ekf,
    base_misalign: &AlignMisalign,
    course_rad: f32,
    replay_steps: &[StartupReplayStep],
    center_yaw_deg: Option<f32>,
    span_deg: f32,
    yaw_step_deg: f32,
    gps_dt_s: f32,
) -> Option<(Ekf, AlignMisalign, StartupResolutionDiag)> {
    let center_yaw_deg = center_yaw_deg.unwrap_or_else(|| {
        let (_, _, yaw_deg) = quat_rpy_deg(
            base_misalign.q_vb[0],
            base_misalign.q_vb[1],
            base_misalign.q_vb[2],
            base_misalign.q_vb[3],
        );
        yaw_deg as f32
    });
    let mut best = None;
    let mut best_cost = f32::INFINITY;
    let mut second = None;
    let mut second_cost = f32::INFINITY;
    let span = span_deg.max(yaw_step_deg);
    let n_steps = ((2.0 * span) / yaw_step_deg).round() as i32;
    for k in 0..=n_steps {
        let yaw_deg = center_yaw_deg - span + k as f32 * yaw_step_deg;
        if let Some((ekf, misalign, mean_cost)) = evaluate_startup_candidate(
            base_ekf,
            base_misalign,
            course_rad,
            replay_steps,
            yaw_deg,
            gps_dt_s,
        ) {
            if mean_cost < best_cost {
                if let Some((_, _, prev_best_yaw_deg)) = best.as_ref() {
                    second = Some(*prev_best_yaw_deg);
                    second_cost = best_cost;
                }
                best_cost = mean_cost;
                best = Some((ekf, misalign, yaw_deg));
            } else if mean_cost < second_cost {
                second_cost = mean_cost;
                second = Some(yaw_deg);
            }
        }
    }
    let (ekf, misalign, best_yaw_deg) = best?;
    finalize_startup_diag(
        ekf,
        misalign,
        best_cost,
        second_cost,
        best_yaw_deg,
        second,
        base_misalign,
        replay_steps,
    )
}

fn resolve_startup_family_local(
    base_ekf: &Ekf,
    base_misalign: &AlignMisalign,
    course_rad: f32,
    replay_steps: &[StartupReplayStep],
    center_yaw_deg: f32,
    span_deg: f32,
    step_deg: f32,
    gps_dt_s: f32,
) -> Option<(Ekf, AlignMisalign, f32, f32, f32, f32)> {
    let mut best = None;
    let mut best_cost = f32::INFINITY;
    let mut second_yaw_deg = f32::NAN;
    let mut second_cost = f32::INFINITY;
    let span = span_deg.max(step_deg);
    let n_steps = ((2.0 * span) / step_deg).round() as i32;
    for k in 0..=n_steps {
        let yaw_deg = center_yaw_deg - span + k as f32 * step_deg;
        if let Some((ekf, misalign, mean_cost)) = evaluate_startup_candidate(
            base_ekf,
            base_misalign,
            course_rad,
            replay_steps,
            yaw_deg,
            gps_dt_s,
        ) {
            if mean_cost < best_cost {
                second_cost = best_cost;
                second_yaw_deg = best.map(|(_, _, y)| y).unwrap_or(f32::NAN);
                best_cost = mean_cost;
                best = Some((ekf, misalign, yaw_deg));
            } else if mean_cost < second_cost {
                second_cost = mean_cost;
                second_yaw_deg = yaw_deg;
            }
        }
    }
    let (ekf, misalign, best_yaw_deg) = best?;
    Some((
        ekf,
        misalign,
        best_yaw_deg,
        best_cost,
        second_yaw_deg,
        second_cost,
    ))
}

fn resolve_joint_startup_state_pca_local(
    base_ekf: &Ekf,
    base_misalign: &AlignMisalign,
    course_rad: f32,
    replay_steps: &[StartupReplayStep],
    local_span_deg: f32,
    local_step_deg: f32,
    gps_dt_s: f32,
) -> Option<(Ekf, AlignMisalign, StartupResolutionDiag)> {
    let (pca_axis_deg, _) = startup_pca_diag(base_misalign, replay_steps)?;
    let primary_center_yaw_deg = wrap_deg_180(-pca_axis_deg);
    let alternate_center_yaw_deg = wrap_deg_180(primary_center_yaw_deg + 180.0);

    let primary = resolve_startup_family_local(
        base_ekf,
        base_misalign,
        course_rad,
        replay_steps,
        primary_center_yaw_deg,
        local_span_deg,
        local_step_deg,
        gps_dt_s,
    )?;
    let alternate = resolve_startup_family_local(
        base_ekf,
        base_misalign,
        course_rad,
        replay_steps,
        alternate_center_yaw_deg,
        local_span_deg,
        local_step_deg,
        gps_dt_s,
    );

    let (ekf, misalign, best_yaw_deg, best_cost, second_yaw_deg, second_cost) = match alternate {
        Some((
            alt_ekf,
            alt_misalign,
            alt_best_yaw_deg,
            alt_best_cost,
            alt_second_yaw_deg,
            alt_second_cost,
        )) if alt_best_cost < primary.3 => {
            let _ = (alt_second_yaw_deg, alt_second_cost);
            (
                alt_ekf,
                alt_misalign,
                alt_best_yaw_deg,
                alt_best_cost,
                primary.2,
                primary.3,
            )
        }
        Some((_, _, alt_best_yaw_deg, alt_best_cost, _, _)) => (
            primary.0,
            primary.1,
            primary.2,
            primary.3,
            alt_best_yaw_deg,
            alt_best_cost,
        ),
        None => (
            primary.0, primary.1, primary.2, primary.3, primary.4, primary.5,
        ),
    };

    finalize_startup_diag(
        ekf,
        misalign,
        best_cost,
        second_cost,
        best_yaw_deg,
        Some(second_yaw_deg),
        base_misalign,
        replay_steps,
    )
}

fn resolve_joint_startup_state_fixed_lag(
    base_ekf: &Ekf,
    base_misalign: &AlignMisalign,
    course_rad: f32,
    replay_steps: &[StartupReplayStep],
    coarse_step_deg: f32,
    refine_span_deg: f32,
    refine_step_deg: f32,
    gps_dt_s: f32,
) -> Option<(Ekf, AlignMisalign, StartupResolutionDiag)> {
    let (_, _, center_yaw_deg) = quat_rpy_deg(
        base_misalign.q_vb[0],
        base_misalign.q_vb[1],
        base_misalign.q_vb[2],
        base_misalign.q_vb[3],
    );
    let center_yaw_deg = center_yaw_deg as f32;
    let mut coarse_best = None;
    let mut coarse_best_cost = f32::INFINITY;
    let mut coarse_second = None;
    let mut coarse_second_cost = f32::INFINITY;
    let span = 120.0_f32.max(coarse_step_deg);
    let n_steps = ((2.0 * span) / coarse_step_deg).round() as i32;
    for k in 0..=n_steps {
        let yaw_deg = center_yaw_deg - span + k as f32 * coarse_step_deg;
        if let Some((_, _, mean_cost)) = evaluate_startup_candidate(
            base_ekf,
            base_misalign,
            course_rad,
            replay_steps,
            yaw_deg,
            gps_dt_s,
        ) {
            if mean_cost < coarse_best_cost {
                coarse_second = coarse_best;
                coarse_second_cost = coarse_best_cost;
                coarse_best = Some(yaw_deg);
                coarse_best_cost = mean_cost;
            } else if mean_cost < coarse_second_cost {
                coarse_second = Some(yaw_deg);
                coarse_second_cost = mean_cost;
            }
        }
    }
    let coarse_best = coarse_best?;
    let mut best = None;
    let mut best_cost = f32::INFINITY;
    let mut second = coarse_second;
    let mut second_cost = coarse_second_cost;
    let refine_span_deg = refine_span_deg.max(refine_step_deg);
    let refine_steps = ((2.0 * refine_span_deg) / refine_step_deg).round() as i32;
    for k in 0..=refine_steps {
        let yaw_deg = coarse_best - refine_span_deg + k as f32 * refine_step_deg;
        if let Some((ekf, misalign, mean_cost)) = evaluate_startup_candidate(
            base_ekf,
            base_misalign,
            course_rad,
            replay_steps,
            yaw_deg,
            gps_dt_s,
        ) {
            if mean_cost < best_cost {
                if let Some((_, _, prev_best_yaw_deg)) = best.as_ref() {
                    second = Some(*prev_best_yaw_deg);
                    second_cost = best_cost;
                }
                best_cost = mean_cost;
                best = Some((ekf, misalign, yaw_deg));
            } else if mean_cost < second_cost {
                second_cost = mean_cost;
                second = Some(yaw_deg);
            }
        }
    }
    let (ekf, misalign, best_yaw_deg) = best?;
    finalize_startup_diag(
        ekf,
        misalign,
        best_cost,
        second_cost,
        best_yaw_deg,
        second,
        base_misalign,
        replay_steps,
    )
}

fn resolve_joint_startup_state_pca(
    base_ekf: &Ekf,
    base_misalign: &AlignMisalign,
    course_rad: f32,
    replay_steps: &[StartupReplayStep],
    gps_dt_s: f32,
) -> Option<(Ekf, AlignMisalign, StartupResolutionDiag)> {
    let (pca_axis_deg, pca_anisotropy) = startup_pca_diag(base_misalign, replay_steps)?;
    let best_yaw_deg = wrap_deg_180(-pca_axis_deg);
    let alt_yaw_deg = wrap_deg_180(best_yaw_deg + 180.0);
    let (ekf, misalign, best_cost) = evaluate_startup_candidate(
        base_ekf,
        base_misalign,
        course_rad,
        replay_steps,
        best_yaw_deg,
        gps_dt_s,
    )?;
    let second_cost = evaluate_startup_candidate(
        base_ekf,
        base_misalign,
        course_rad,
        replay_steps,
        alt_yaw_deg,
        gps_dt_s,
    )
    .map(|(_, _, c)| c)
    .unwrap_or(f32::INFINITY);
    let mount_axis_deg = wrap_deg_180(-best_yaw_deg);
    Some((
        ekf,
        misalign,
        StartupResolutionDiag {
            best_cost,
            second_cost,
            best_yaw_deg,
            second_yaw_deg: alt_yaw_deg,
            mount_axis_deg,
            pca_axis_deg,
            pca_anisotropy,
            pca_axis_err_deg: axis_err_mod_180_deg(mount_axis_deg, pca_axis_deg),
            ready: false,
        },
    ))
}

fn evaluate_startup_candidate(
    base_ekf: &Ekf,
    base_misalign: &AlignMisalign,
    course_rad: f32,
    replay_steps: &[StartupReplayStep],
    yaw_deg: f32,
    gps_dt_s: f32,
) -> Option<(Ekf, AlignMisalign, f32)> {
    let mut ekf = base_ekf.clone();
    let mut misalign = base_misalign.clone();
    misalign.set_mount_yaw(wrap_pi(yaw_deg.to_radians()), 5.0_f32.to_radians());
    seed_ekf_from_vehicle_course_and_mount(&mut ekf, misalign.q_vb, course_rad);
    let mut cost = 0.0_f32;
    let mut n = 0usize;
    for step in replay_steps {
        ekf_predict(&mut ekf, &step.imu, None);
        clamp_ekf_biases(&mut ekf, step.imu.dt as f64);
        ekf_fuse_vehicle_vel(&mut ekf, misalign.q_vb, 100.0);
        clamp_ekf_biases(&mut ekf, step.imu.dt as f64);
        if let Some(gps) = step.gps {
            ekf_fuse_gps(&mut ekf, &gps);
            clamp_ekf_biases(&mut ekf, step.imu.dt as f64);
            let dt_safe = step.imu.dt.max(1.0e-3);
            let q_nb_est = [ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3];
            let v_n_est = [ekf.state.vn, ekf.state.ve, ekf.state.vd];
            let omega_b_corr = [
                step.gyro_b_radps[0] - ekf.state.dax_b / dt_safe,
                step.gyro_b_radps[1] - ekf.state.day_b / dt_safe,
                step.gyro_b_radps[2] - ekf.state.daz_b / dt_safe,
            ];
            let accel_b_corr = [
                step.imu.dvx / dt_safe - ekf.state.dvx_b / dt_safe,
                step.imu.dvy / dt_safe - ekf.state.dvy_b / dt_safe,
                step.imu.dvz / dt_safe - ekf.state.dvz_b / dt_safe,
            ];
            let (score, _) =
                misalign.update_all(gps_dt_s, q_nb_est, v_n_est, omega_b_corr, accel_b_corr);
            let pred = misalign.nhc_prediction(q_nb_est, v_n_est);
            let sigma = misalign.cfg.r_nhc_std_mps.max(1.0e-6);
            cost += score + (pred[0] / sigma).powi(2) + (pred[1] / sigma).powi(2);
            n += 4;
        }
    }
    if n == 0 {
        return None;
    }
    Some((ekf, misalign, cost / n as f32))
}

fn finalize_startup_diag(
    ekf: Ekf,
    misalign: AlignMisalign,
    best_cost: f32,
    second_cost: f32,
    best_yaw_deg: f32,
    second: Option<f32>,
    base_misalign: &AlignMisalign,
    replay_steps: &[StartupReplayStep],
) -> Option<(Ekf, AlignMisalign, StartupResolutionDiag)> {
    let (pca_axis_deg, pca_anisotropy) =
        startup_pca_diag(base_misalign, replay_steps).unwrap_or((f32::NAN, 0.0));
    let mount_axis_deg = wrap_deg_180(-best_yaw_deg);
    Some((
        ekf,
        misalign,
        StartupResolutionDiag {
            best_cost,
            second_cost,
            best_yaw_deg,
            second_yaw_deg: second.unwrap_or(f32::NAN),
            mount_axis_deg,
            pca_axis_deg,
            pca_anisotropy,
            pca_axis_err_deg: axis_err_mod_180_deg(mount_axis_deg, pca_axis_deg),
            ready: false,
        },
    ))
}

fn startup_pca_diag(
    base_misalign: &AlignMisalign,
    replay_steps: &[StartupReplayStep],
) -> Option<(f32, f32)> {
    let mut prev_gps: Option<GpsData> = None;
    let mut prev_t_s = 0.0_f64;
    let mut samples = Vec::with_capacity(replay_steps.len());
    for step in replay_steps {
        let gps = match step.gps {
            Some(gps) => gps,
            None => continue,
        };
        if let Some(prev) = prev_gps {
            let speed_h = ((gps.vel_n as f64).powi(2) + (gps.vel_e as f64).powi(2)).sqrt();
            let dt_gps = (step.t_s - prev_t_s).max(1.0e-3);
            let dv_h = [
                gps.vel_n as f64 - prev.vel_n as f64,
                gps.vel_e as f64 - prev.vel_e as f64,
            ];
            let acc_h = [dv_h[0] / dt_gps, dv_h[1] / dt_gps];
            let course = [
                (gps.vel_n as f64) / speed_h.max(1.0e-6),
                (gps.vel_e as f64) / speed_h.max(1.0e-6),
            ];
            let gnss_long = acc_h[0] * course[0] + acc_h[1] * course[1];
            let dt_imu = step.imu.dt.max(1.0e-3) as f64;
            samples.push(MisalignCoarseSample {
                speed_mps: speed_h as f32,
                accel_b_mps2: [
                    (step.imu.dvx as f64 / dt_imu) as f32,
                    (step.imu.dvy as f64 / dt_imu) as f32,
                    (step.imu.dvz as f64 / dt_imu) as f32,
                ],
                gnss_long_mps2: gnss_long as f32,
            });
        }
        prev_gps = Some(gps);
        prev_t_s = step.t_s;
    }
    let out = estimate_mount_yaw_from_tilt(
        MisalignCoarseConfig {
            min_speed_mps: 2.0,
            min_horiz_acc_mps2: 0.1,
            min_windows: 8,
            min_anisotropy_ratio: 1.0,
        },
        base_misalign.q_vb,
        &samples,
    )?;
    Some((
        wrap_deg_180(out.pca_axis_rad.to_degrees()),
        out.anisotropy_ratio,
    ))
}

fn convergence_time(
    metrics: &[Metrics],
    fwd_thresh_deg: f64,
    down_thresh_deg: f64,
    pos_thresh_m: f64,
    vel_thresh_mps: f64,
) -> Option<f64> {
    for i in 0..metrics.len() {
        if metrics[i..].iter().all(|m| {
            m.fwd_err_deg <= fwd_thresh_deg
                && m.down_err_deg <= down_thresh_deg
                && m.pos_err_m <= pos_thresh_m
                && m.vel_err_mps <= vel_thresh_mps
        }) {
            return Some(metrics[i].t_s);
        }
    }
    None
}

fn print_run_summary(args: &Args, summary: &RunSummary) {
    let final_m = summary.final_m;
    println!(
        "scenario: duration={:.1}s imu_dt={:.3}s gps_dt={:.3}s mount_true=[{:.2},{:.2},{:.2}] deg",
        args.duration_s,
        args.imu_dt_s,
        args.gps_dt_s,
        args.mount_roll_deg,
        args.mount_pitch_deg,
        args.mount_yaw_deg
    );
    if let Some(final_m) = final_m {
        println!(
            "final nav: pos_err={:.3} m vel_err={:.3} m/s body_att_err={:.3} deg veh_att_err={:.3} deg",
            final_m.pos_err_m,
            final_m.vel_err_mps,
            final_m.body_att_err_deg,
            final_m.veh_att_err_deg
        );
        println!(
            "final mount: rot_err={:.3} deg fwd_err={:.3} deg down_err={:.3} deg",
            final_m.mount_err_deg, final_m.fwd_err_deg, final_m.down_err_deg
        );
        if let Some(t) = summary.convergence_t_s {
            println!(
                "convergence: all thresholds met by t={:.3}s and remain satisfied",
                t
            );
        } else {
            println!("convergence: thresholds not jointly satisfied to end of run");
        }
    } else {
        println!("status: not ready");
    }
    if let Some(diag) = summary.startup_diag {
        let gap = diag.second_cost - diag.best_cost;
        let rel_gap = gap / diag.best_cost.max(1.0e-6);
        println!(
            "startup: ready={} best_cost={:.4} second_cost={:.4} gap={:.4} rel_gap={:.2}% best_yaw={:.1} deg second_yaw={:.1} deg mount_axis={:.1} deg pca_axis={:.1} deg pca_aniso={:.2} pca_axis_err={:.1} deg",
            diag.ready,
            diag.best_cost,
            diag.second_cost,
            gap,
            100.0 * rel_gap,
            diag.best_yaw_deg,
            diag.second_yaw_deg,
            diag.mount_axis_deg,
            diag.pca_axis_deg,
            diag.pca_anisotropy,
            diag.pca_axis_err_deg
        );
    }
    for m in &summary.checkpoints {
        println!(
            "t={:.1}s body_att={:.3} body_yaw={:.3} veh_att={:.3} veh_yaw={:.3} mount={:.3} mount_yaw={:.3} fwd={:.3} down={:.3}",
            m.t_s,
            m.body_att_err_deg,
            m.body_yaw_err_deg,
            m.veh_att_err_deg,
            m.veh_yaw_err_deg,
            m.mount_err_deg,
            m.mount_yaw_err_deg,
            m.fwd_err_deg,
            m.down_err_deg
        );
    }
}

fn print_mc_summary(args: &Args, summaries: &[RunSummary]) {
    let ready_summaries: Vec<_> = summaries.iter().filter(|s| s.final_m.is_some()).collect();
    let mut fwd: Vec<_> = ready_summaries
        .iter()
        .map(|s| s.final_m.expect("ready summary").fwd_err_deg)
        .collect();
    let mut down: Vec<_> = ready_summaries
        .iter()
        .map(|s| s.final_m.expect("ready summary").down_err_deg)
        .collect();
    let mut veh: Vec<_> = ready_summaries
        .iter()
        .map(|s| s.final_m.expect("ready summary").veh_att_err_deg)
        .collect();
    let mut conv: Vec<_> = summaries.iter().filter_map(|s| s.convergence_t_s).collect();
    let mut startup_gap: Vec<_> = summaries
        .iter()
        .filter_map(|s| s.startup_diag.map(|d| (d.second_cost - d.best_cost) as f64))
        .collect();
    let mut startup_rel_gap: Vec<_> = summaries
        .iter()
        .filter_map(|s| {
            s.startup_diag
                .map(|d| ((d.second_cost - d.best_cost) / d.best_cost.max(1.0e-6)) as f64)
        })
        .collect();
    fwd.sort_by(|a, b| a.total_cmp(b));
    down.sort_by(|a, b| a.total_cmp(b));
    veh.sort_by(|a, b| a.total_cmp(b));
    conv.sort_by(|a, b| a.total_cmp(b));
    startup_gap.sort_by(|a, b| a.total_cmp(b));
    startup_rel_gap.sort_by(|a, b| a.total_cmp(b));

    let ready = ready_summaries.len();
    let not_ready = summaries.len().saturating_sub(ready);
    let converged = summaries
        .iter()
        .filter(|s| s.convergence_t_s.is_some())
        .count();
    let conv_gap: Vec<_> = summaries
        .iter()
        .filter(|s| s.convergence_t_s.is_some())
        .filter_map(|s| s.startup_diag.map(|d| (d.second_cost - d.best_cost) as f64))
        .collect();
    let fail_gap: Vec<_> = summaries
        .iter()
        .filter(|s| s.convergence_t_s.is_none())
        .filter_map(|s| s.startup_diag.map(|d| (d.second_cost - d.best_cost) as f64))
        .collect();
    println!(
        "mc: runs={} seeds={}..{} scenario={:?} mount_true=[{:.2},{:.2},{:.2}] deg",
        summaries.len(),
        args.mc_seed_start,
        args.mc_seed_start + summaries.len() as u64 - 1,
        args.scenario,
        args.mount_roll_deg,
        args.mount_pitch_deg,
        args.mount_yaw_deg
    );
    println!("mc readiness: ready={} not_ready={}", ready, not_ready);
    if !fwd.is_empty() {
        println!(
            "mc final fwd_deg: mean={:.3} p50={:.3} p95={:.3} max={:.3}",
            mean(&fwd),
            percentile_sorted(&fwd, 0.50),
            percentile_sorted(&fwd, 0.95),
            *fwd.last().unwrap_or(&f64::NAN)
        );
        println!(
            "mc final down_deg: mean={:.3} p50={:.3} p95={:.3} max={:.3}",
            mean(&down),
            percentile_sorted(&down, 0.50),
            percentile_sorted(&down, 0.95),
            *down.last().unwrap_or(&f64::NAN)
        );
        println!(
            "mc final veh_att_deg: mean={:.3} p50={:.3} p95={:.3} max={:.3}",
            mean(&veh),
            percentile_sorted(&veh, 0.50),
            percentile_sorted(&veh, 0.95),
            *veh.last().unwrap_or(&f64::NAN)
        );
    }
    let ready_rate = if ready > 0 {
        100.0 * converged as f64 / ready as f64
    } else {
        f64::NAN
    };
    let total_rate = if !summaries.is_empty() {
        100.0 * converged as f64 / summaries.len() as f64
    } else {
        f64::NAN
    };
    println!(
        "mc convergence: {}/{} ready runs ({:.1}%), {}/{} total ({:.1}%)",
        converged,
        ready,
        ready_rate,
        converged,
        summaries.len(),
        total_rate
    );
    if !conv.is_empty() {
        println!(
            "mc convergence_t_s: mean={:.3} p50={:.3} p95={:.3} max={:.3}",
            mean(&conv),
            percentile_sorted(&conv, 0.50),
            percentile_sorted(&conv, 0.95),
            *conv.last().unwrap_or(&f64::NAN)
        );
    }
    if !startup_gap.is_empty() {
        println!(
            "mc startup_gap: mean={:.4} p50={:.4} p95={:.4} min={:.4}",
            mean(&startup_gap),
            percentile_sorted(&startup_gap, 0.50),
            percentile_sorted(&startup_gap, 0.95),
            startup_gap[0]
        );
    }
    if !startup_rel_gap.is_empty() {
        println!(
            "mc startup_rel_gap: mean={:.2}% p50={:.2}% p95={:.2}% min={:.2}%",
            100.0 * mean(&startup_rel_gap),
            100.0 * percentile_sorted(&startup_rel_gap, 0.50),
            100.0 * percentile_sorted(&startup_rel_gap, 0.95),
            100.0 * startup_rel_gap[0]
        );
    }
    if !conv_gap.is_empty() {
        println!("mc startup_gap converged_mean={:.4}", mean(&conv_gap));
    }
    if !fail_gap.is_empty() {
        println!("mc startup_gap failed_mean={:.4}", mean(&fail_gap));
    }
    let rel_gap_thresholds = [0.005_f64, 0.01, 0.02, 0.05, 0.10];
    for thr in rel_gap_thresholds {
        let accepted: Vec<_> = summaries
            .iter()
            .filter(|s| {
                s.startup_diag
                    .map(|d| {
                        ((d.second_cost - d.best_cost) / d.best_cost.max(1.0e-6)) as f64 >= thr
                    })
                    .unwrap_or(false)
            })
            .collect();
        if accepted.is_empty() {
            println!(
                "mc accept rel_gap>={:.1}%: 0/{} runs",
                100.0 * thr,
                summaries.len()
            );
            continue;
        }
        let accepted_ready: Vec<_> = accepted.iter().filter_map(|s| s.final_m).collect();
        let accepted_converged = accepted
            .iter()
            .filter(|s| s.convergence_t_s.is_some())
            .count();
        let accepted_fwd: Vec<_> = accepted_ready.iter().map(|m| m.fwd_err_deg).collect();
        let accepted_down: Vec<_> = accepted_ready.iter().map(|m| m.down_err_deg).collect();
        println!(
            "mc accept rel_gap>={:.1}%: accepted={}/{} ready={}/{} converged={}/{} mean_fwd={:.3} max_fwd={:.3} mean_down={:.3} max_down={:.3}",
            100.0 * thr,
            accepted.len(),
            summaries.len(),
            accepted_ready.len(),
            accepted.len(),
            accepted_converged,
            accepted.len(),
            mean(&accepted_fwd),
            accepted_fwd
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
            mean(&accepted_down),
            accepted_down
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        );
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn percentile_sorted(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let idx = ((xs.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    xs[idx]
}

fn axis_err_mod_180_deg(a_deg: f32, b_deg: f32) -> f32 {
    if !a_deg.is_finite() || !b_deg.is_finite() {
        return f32::INFINITY;
    }
    let diff = wrap_deg_180(a_deg - b_deg).abs();
    diff.min(180.0 - diff)
}

fn wrap_deg_180(mut a: f32) -> f32 {
    while a > 180.0 {
        a -= 360.0;
    }
    while a <= -180.0 {
        a += 360.0;
    }
    a
}

fn angle_between_unit_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn axis_err_deg(q_est: [f64; 4], q_true: [f64; 4], axis: [f64; 3]) -> f64 {
    let est = quat_rotate_f64(q_est, axis);
    let tru = quat_rotate_f64(q_true, axis);
    angle_between_unit_deg(est, tru)
}

fn quat_angle_deg(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dq = quat_mul(quat_conj(a), b);
    2.0 * dq[0].abs().clamp(-1.0, 1.0).acos().to_degrees()
}

fn seed_ekf_from_vehicle_course_and_mount(ekf: &mut Ekf, q_vb: [f32; 4], course_rad: f32) {
    let q_nv = quat_from_rpy(0.0, 0.0, course_rad as f64);
    let q_nb = quat_mul(q_nv, f32_to_f64_quat(q_vb));
    ekf.state.q0 = q_nb[0] as f32;
    ekf.state.q1 = q_nb[1] as f32;
    ekf.state.q2 = q_nb[2] as f32;
    ekf.state.q3 = q_nb[3] as f32;
}

fn f32_to_f64_quat(q: [f32; 4]) -> [f64; 4] {
    [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64]
}

fn f64_to_f32_quat(q: [f64; 4]) -> [f32; 4] {
    [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32]
}

fn quat_rotate_f64(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    mat3_vec(quat_to_rotmat(q), v)
}

fn quat_from_rpy(roll: f64, pitch: f64, yaw: f64) -> [f64; 4] {
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

fn quat_to_rotmat(q: [f64; 4]) -> [[f64; 3]; 3] {
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

fn quat_from_rotmat(r: [[f64; 3]; 3]) -> [f64; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let q = if tr > 0.0 {
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
    quat_normalize(q)
}

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn rot_yaw(yaw: f64) -> [[f64; 3]; 3] {
    let (s, c) = yaw.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn mat3_vec(a: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn transpose3(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn wrap_pi(mut a: f32) -> f32 {
    let pi = std::f32::consts::PI;
    while a > pi {
        a -= 2.0 * pi;
    }
    while a < -pi {
        a += 2.0 * pi;
    }
    a
}

fn wrap_pi_f64(mut a: f64) -> f64 {
    let pi = std::f64::consts::PI;
    while a > pi {
        a -= 2.0 * pi;
    }
    while a < -pi {
        a += 2.0 * pi;
    }
    a
}
