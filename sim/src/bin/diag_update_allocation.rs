use anyhow::Result;
use clap::Parser;
use sensor_fusion::eskf_types::{ESKF_UPDATE_DIAG_TYPES, EskfUpdateDiag};
use sensor_fusion::fusion::{EskfMountSource, SensorFusion};
use sensor_fusion::loose::{LOOSE_ERROR_STATES, LooseFilter, LooseImuDelta, LoosePredictNoise};
use sim::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, fusion_gnss_sample, fusion_imu_sample,
};
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::visualizer::math::lla_to_ecef;
use sim::visualizer::pipeline::EkfCompareConfig;
use sim::visualizer::pipeline::synthetic::{
    SyntheticNoiseMode, SyntheticVisualizerConfig, build_synthetic_replay_input,
};
use std::path::PathBuf;

const LOOSE_OBS_TYPES: usize = 9;
const ESKF_LABELS: [&str; ESKF_UPDATE_DIAG_TYPES] = [
    "gps_pos",
    "gps_vel",
    "zero_vel",
    "body_speed_x",
    "body_vel_y",
    "body_vel_z",
    "stationary_x",
    "stationary_y",
    "gps_pos_d",
    "gps_vel_d",
    "zero_vel_d",
];
const LOOSE_LABELS: [&str; LOOSE_OBS_TYPES] = [
    "none", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "nhc_y", "nhc_z",
];

#[derive(Parser, Debug)]
#[command(name = "diag_update_allocation")]
struct Args {
    #[arg(long, value_name = "SCENARIO")]
    motion_def: PathBuf,
    #[arg(long, default_value_t = 100.0)]
    imu_hz: f64,
    #[arg(long, default_value_t = 2.0)]
    gnss_hz: f64,
    #[arg(long, default_value_t = 600.0)]
    window_start_s: f64,
    #[arg(long, default_value_t = 1200.0)]
    window_end_s: f64,
    #[arg(long, default_value_t = 5.0)]
    mount_roll_deg: f64,
    #[arg(long, default_value_t = -5.0)]
    mount_pitch_deg: f64,
    #[arg(long, default_value_t = 5.0)]
    mount_yaw_deg: f64,
    #[arg(long)]
    r_body_vel: Option<f32>,
    #[arg(long)]
    r_body_vel_z: Option<f32>,
    #[arg(long)]
    disable_gnss_pos: bool,
    #[arg(long)]
    disable_gnss_vel: bool,
}

#[derive(Clone, Copy, Debug)]
struct Allocation<const N: usize> {
    count: [u32; N],
    residual_sum: [f64; N],
    residual_abs_sum: [f64; N],
    accel_sensor_bias: [[f64; 3]; N],
    gyro_sensor_bias_dps: [[f64; 3]; N],
    mount_deg: [[f64; 3]; N],
}

#[derive(Clone, Copy, Debug, Default)]
struct ResidualSummary {
    vel_count: u32,
    vel_sum: [f64; 3],
}

impl<const N: usize> Default for Allocation<N> {
    fn default() -> Self {
        Self {
            count: [0; N],
            residual_sum: [0.0; N],
            residual_abs_sum: [0.0; N],
            accel_sensor_bias: [[0.0; 3]; N],
            gyro_sensor_bias_dps: [[0.0; 3]; N],
            mount_deg: [[0.0; 3]; N],
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = EkfCompareConfig {
        r_body_vel: args
            .r_body_vel
            .unwrap_or(EkfCompareConfig::default().r_body_vel),
        r_body_vel_z: args
            .r_body_vel_z
            .unwrap_or(EkfCompareConfig::default().r_body_vel_z),
        ..EkfCompareConfig::default()
    };
    let synth_cfg = SyntheticVisualizerConfig {
        motion_def: Some(args.motion_def.clone()),
        motion_label: args.motion_def.display().to_string(),
        motion_text: None,
        noise_mode: SyntheticNoiseMode::Truth,
        disable_imu_noise: false,
        disable_gnss_noise: false,
        seed: 1,
        mount_rpy_deg: [
            args.mount_roll_deg,
            args.mount_pitch_deg,
            args.mount_yaw_deg,
        ],
        imu_hz: args.imu_hz,
        gnss_hz: args.gnss_hz,
        gnss_time_shift_ms: 0.0,
        early_vel_bias_ned_mps: [0.0; 3],
        early_fault_window_s: None,
    };
    let (replay, q_mount) = build_synthetic_replay_input(&synth_cfg)?;
    let (eskf, eskf_residuals) = run_eskf_allocation(&replay, q_mount, cfg, &args);
    let (loose, loose_residuals) = run_loose_allocation(&replay, q_mount, cfg, &args);

    println!(
        "update allocation scenario={} imu_hz={:.1} window=[{:.1},{:.1}]s",
        args.motion_def.display(),
        args.imu_hz,
        args.window_start_s,
        args.window_end_s
    );
    print_allocation("ESKF", &ESKF_LABELS, &eskf);
    print_residual_summary("ESKF", eskf_residuals);
    print_allocation("Loose", &LOOSE_LABELS, &loose);
    print_residual_summary("Loose", loose_residuals);
    Ok(())
}

fn run_eskf_allocation(
    replay: &sim::visualizer::pipeline::generic::GenericReplayInput,
    q_mount: [f32; 4],
    cfg: EkfCompareConfig,
    args: &Args,
) -> (Allocation<ESKF_UPDATE_DIAG_TYPES>, ResidualSummary) {
    let mut fusion = SensorFusion::new();
    apply_fusion_config(&mut fusion, cfg);
    fusion.set_misalignment(q_mount);

    let mut prev_diag = EskfUpdateDiag::default();
    let mut out = Allocation::default();
    let mut residuals = ResidualSummary::default();
    for_each_event(&replay.imu, &replay.gnss, |event| {
        let t_s = match event {
            ReplayEvent::Imu(_, sample) => {
                let _ = fusion.process_imu(fusion_imu_sample(*sample));
                sample.t_s
            }
            ReplayEvent::Gnss(_, sample) => {
                if (args.window_start_s..=args.window_end_s).contains(&sample.t_s)
                    && let Some(eskf) = fusion.eskf()
                {
                    residuals.vel_count += 1;
                    residuals.vel_sum[0] += sample.vel_ned_mps[0] - eskf.nominal.vn as f64;
                    residuals.vel_sum[1] += sample.vel_ned_mps[1] - eskf.nominal.ve as f64;
                    residuals.vel_sum[2] += sample.vel_ned_mps[2] - eskf.nominal.vd as f64;
                }
                let mut gnss = *sample;
                if args.disable_gnss_pos {
                    gnss.pos_std_m = [1.0e6; 3];
                }
                if args.disable_gnss_vel {
                    gnss.vel_std_mps = [1.0e6; 3];
                }
                let _ = fusion.process_gnss(fusion_gnss_sample(gnss));
                sample.t_s
            }
        };
        let Some(eskf) = fusion.eskf() else {
            return;
        };
        let diag = eskf.update_diag;
        if (args.window_start_s..=args.window_end_s).contains(&t_s) {
            accumulate_eskf_diag_delta(&mut out, &prev_diag, &diag);
        }
        prev_diag = diag;
    });
    (out, residuals)
}

fn run_loose_allocation(
    replay: &sim::visualizer::pipeline::generic::GenericReplayInput,
    q_mount: [f32; 4],
    cfg: EkfCompareConfig,
    args: &Args,
) -> (Allocation<LOOSE_OBS_TYPES>, ResidualSummary) {
    let mut loose = LooseFilter::new(
        cfg.loose_predict_noise
            .unwrap_or_else(LoosePredictNoise::lsm6dso_loose_104hz),
    );
    let mut out = Allocation::default();
    let mut residuals = ResidualSummary::default();
    let mut ready = false;
    let mut last_imu: Option<GenericImuSample> = None;
    let mut latest_gnss: Option<GenericGnssSample> = None;
    let mut gnss_cursor = 0usize;
    let mut last_gnss_used_t_s = f64::NEG_INFINITY;

    for_each_event(&replay.imu, &replay.gnss, |event| match event {
        ReplayEvent::Imu(_, sample) => {
            let Some(prev) = last_imu.replace(*sample) else {
                return;
            };
            if !ready {
                return;
            }
            let dt = (sample.t_s - prev.t_s).max(0.0);
            if dt <= 0.0 || dt > 1.0 {
                return;
            }
            loose.predict(loose_imu_delta(prev, *sample, dt));
            while gnss_cursor < replay.gnss.len()
                && replay.gnss[gnss_cursor].t_s <= sample.t_s + 1.0e-9
            {
                latest_gnss = Some(replay.gnss[gnss_cursor]);
                gnss_cursor += 1;
            }

            let mut gps_pos = None;
            let mut gps_vel = None;
            let mut gps_pos_std = 0.0_f32;
            let mut gps_vel_std = None;
            let mut dt_since_gnss = 1.0_f32;
            if let Some(gnss) = latest_gnss
                && (0.0..=0.05).contains(&(sample.t_s - gnss.t_s))
                && gnss.t_s != last_gnss_used_t_s
            {
                if !args.disable_gnss_pos {
                    gps_pos = Some(lla_to_ecef(gnss.lat_deg, gnss.lon_deg, gnss.height_m));
                    gps_pos_std = ((gnss.pos_std_m[0] + gnss.pos_std_m[1] + gnss.pos_std_m[2])
                        / 3.0)
                        .max(0.1) as f32;
                }
                if !args.disable_gnss_vel {
                    gps_vel = Some(ned_vector_to_ecef(
                        gnss.lat_deg,
                        gnss.lon_deg,
                        gnss.vel_ned_mps,
                    ));
                    gps_vel_std = Some(gnss.vel_std_mps.map(|v| v.max(0.01) as f32));
                }
                dt_since_gnss = if last_gnss_used_t_s.is_finite() {
                    (gnss.t_s - last_gnss_used_t_s).clamp(1.0e-3, 1.0) as f32
                } else {
                    1.0
                };
                if (args.window_start_s..=args.window_end_s).contains(&sample.t_s) {
                    let n = loose.nominal();
                    let vel_ned =
                        ecef_vector_to_ned(gnss.lat_deg, gnss.lon_deg, [n.vn, n.ve, n.vd]);
                    residuals.vel_count += 1;
                    residuals.vel_sum[0] += gnss.vel_ned_mps[0] - vel_ned[0];
                    residuals.vel_sum[1] += gnss.vel_ned_mps[1] - vel_ned[1];
                    residuals.vel_sum[2] += gnss.vel_ned_mps[2] - vel_ned[2];
                }
                last_gnss_used_t_s = gnss.t_s;
            }
            let nhc_gate_speed_mps = latest_gnss.and_then(|gnss| {
                let age_s = sample.t_s - gnss.t_s;
                (0.0..=1.0)
                    .contains(&age_s)
                    .then(|| gnss.vel_ned_mps[0].hypot(gnss.vel_ned_mps[1]) as f32)
            });
            loose.fuse_reference_batch_full_with_nhc_speed_and_r(
                gps_pos,
                gps_vel,
                gps_pos_std,
                gps_vel_std,
                dt_since_gnss,
                nhc_gate_speed_mps,
                cfg.r_body_vel,
                cfg.r_body_vel_z,
                sample.gyro_radps.map(|v| v as f32),
                sample.accel_mps2.map(|v| v as f32),
                dt as f32,
            );
            if (args.window_start_s..=args.window_end_s).contains(&sample.t_s) {
                accumulate_loose_last_update(&mut out, &loose);
            }
        }
        ReplayEvent::Gnss(index, sample) => {
            latest_gnss = Some(*sample);
            if ready {
                return;
            }
            let speed = sample.vel_ned_mps[0].hypot(sample.vel_ned_mps[1]);
            if speed < 0.5 {
                return;
            }
            let yaw_rad = sample.vel_ned_mps[1].atan2(sample.vel_ned_mps[0]) as f32;
            let pos_ecef = lla_to_ecef(sample.lat_deg, sample.lon_deg, sample.height_m);
            let vel_ecef = ned_vector_to_ecef(sample.lat_deg, sample.lon_deg, sample.vel_ned_mps);
            loose.init_seeded_vehicle_from_nav_ecef_state(
                yaw_rad,
                sample.lat_deg,
                sample.lon_deg,
                pos_ecef,
                vel_ecef,
                Some(default_loose_p_diag(*sample, cfg)),
                None,
            );
            loose.set_mount_quat(q_mount);
            ready = true;
            gnss_cursor = index + 1;
            last_gnss_used_t_s = sample.t_s;
        }
    });
    (out, residuals)
}

fn accumulate_eskf_diag_delta(
    out: &mut Allocation<ESKF_UPDATE_DIAG_TYPES>,
    prev: &EskfUpdateDiag,
    curr: &EskfUpdateDiag,
) {
    for ty in 0..ESKF_UPDATE_DIAG_TYPES {
        let count_delta = curr.type_counts[ty].saturating_sub(prev.type_counts[ty]);
        out.count[ty] += count_delta;
        out.residual_sum[ty] += (curr.sum_innovation[ty] - prev.sum_innovation[ty]) as f64;
        out.residual_abs_sum[ty] +=
            (curr.sum_abs_innovation[ty] - prev.sum_abs_innovation[ty]) as f64;
        for axis in 0..3 {
            out.gyro_sensor_bias_dps[ty][axis] +=
                (curr.sum_dx_gyro_bias[ty][axis] - prev.sum_dx_gyro_bias[ty][axis]) as f64 * 180.0
                    / core::f64::consts::PI;
            out.accel_sensor_bias[ty][axis] +=
                (curr.sum_dx_accel_bias[ty][axis] - prev.sum_dx_accel_bias[ty][axis]) as f64;
            out.mount_deg[ty][axis] += match axis {
                0 => curr.sum_dx_mount_roll[ty] - prev.sum_dx_mount_roll[ty],
                1 => curr.sum_dx_mount_pitch[ty] - prev.sum_dx_mount_pitch[ty],
                _ => curr.sum_dx_mount_yaw[ty] - prev.sum_dx_mount_yaw[ty],
            } as f64
                * 180.0
                / core::f64::consts::PI;
        }
    }
}

fn accumulate_loose_last_update(out: &mut Allocation<LOOSE_OBS_TYPES>, loose: &LooseFilter) {
    let obs_types = loose.last_obs_types();
    let dx_by_obs = loose.last_dx_by_obs();
    let effective_residuals = loose.last_effective_residuals();
    for (obs_index, &obs_type) in obs_types.iter().enumerate() {
        let Ok(ty) = usize::try_from(obs_type) else {
            continue;
        };
        if ty >= LOOSE_OBS_TYPES {
            continue;
        }
        out.count[ty] += 1;
        let residual = effective_residuals.get(obs_index).copied().unwrap_or(0.0) as f64;
        out.residual_sum[ty] += residual;
        out.residual_abs_sum[ty] += residual.abs();
        let dx = &dx_by_obs[obs_index];
        for axis in 0..3 {
            out.accel_sensor_bias[ty][axis] += -(dx[9 + axis] as f64);
            out.gyro_sensor_bias_dps[ty][axis] +=
                -(dx[12 + axis] as f64) * 180.0 / core::f64::consts::PI;
            out.mount_deg[ty][axis] += (dx[21 + axis] as f64) * 180.0 / core::f64::consts::PI;
        }
    }
}

fn print_allocation<const N: usize>(filter: &str, labels: &[&str; N], alloc: &Allocation<N>) {
    println!();
    println!("{filter}");
    let mut total = ([0.0; 3], [0.0; 3], [0.0; 3]);
    for ty in 0..N {
        if alloc.count[ty] == 0 {
            continue;
        }
        for axis in 0..3 {
            total.0[axis] += alloc.accel_sensor_bias[ty][axis];
            total.1[axis] += alloc.gyro_sensor_bias_dps[ty][axis];
            total.2[axis] += alloc.mount_deg[ty][axis];
        }
        let mean_residual = alloc.residual_sum[ty] / alloc.count[ty] as f64;
        let mean_abs_residual = alloc.residual_abs_sum[ty] / alloc.count[ty] as f64;
        println!(
            "  {:>12} count={:<6} residual_mean={:+.6} residual_abs_mean={:.6} accel_bias=[{:+.6},{:+.6},{:+.6}]m/s^2 gyro_bias=[{:+.6},{:+.6},{:+.6}]deg/s mount=[{:+.6},{:+.6},{:+.6}]deg",
            labels[ty],
            alloc.count[ty],
            mean_residual,
            mean_abs_residual,
            alloc.accel_sensor_bias[ty][0],
            alloc.accel_sensor_bias[ty][1],
            alloc.accel_sensor_bias[ty][2],
            alloc.gyro_sensor_bias_dps[ty][0],
            alloc.gyro_sensor_bias_dps[ty][1],
            alloc.gyro_sensor_bias_dps[ty][2],
            alloc.mount_deg[ty][0],
            alloc.mount_deg[ty][1],
            alloc.mount_deg[ty][2],
        );
    }
    println!(
        "  {:>12}          accel_bias=[{:+.6},{:+.6},{:+.6}]m/s^2 gyro_bias=[{:+.6},{:+.6},{:+.6}]deg/s mount=[{:+.6},{:+.6},{:+.6}]deg",
        "TOTAL",
        total.0[0],
        total.0[1],
        total.0[2],
        total.1[0],
        total.1[1],
        total.1[2],
        total.2[0],
        total.2[1],
        total.2[2],
    );
}

fn print_residual_summary(filter: &str, residuals: ResidualSummary) {
    if residuals.vel_count == 0 {
        return;
    }
    let inv = 1.0 / residuals.vel_count as f64;
    println!(
        "  {filter:>12} mean_gnss_vel_residual=[{:+.6},{:+.6},{:+.6}]m/s count={}",
        residuals.vel_sum[0] * inv,
        residuals.vel_sum[1] * inv,
        residuals.vel_sum[2] * inv,
        residuals.vel_count
    );
}

fn apply_fusion_config(fusion: &mut SensorFusion, cfg: EkfCompareConfig) {
    fusion.set_align_config(cfg.align);
    if let Some(noise) = cfg.predict_noise {
        fusion.set_predict_noise(noise);
    }
    fusion.set_r_body_vel_yz(cfg.r_body_vel, cfg.r_body_vel_z);
    fusion.set_attitude_roll_pitch_init_sigma_rad(
        cfg.attitude_roll_pitch_init_sigma_deg.to_radians(),
    );
    fusion.set_yaw_init_sigma_rad(cfg.yaw_init_sigma_deg.to_radians());
    fusion.set_gyro_bias_init_sigma_radps(cfg.gyro_bias_init_sigma_dps.to_radians());
    fusion.set_accel_bias_init_sigma_mps2(cfg.accel_bias_init_sigma_mps2);
    fusion.set_mount_roll_pitch_init_sigma_rad(cfg.mount_roll_pitch_init_sigma_deg.to_radians());
    fusion.set_mount_init_sigma_rad(cfg.mount_init_sigma_deg.to_radians());
    fusion.set_r_vehicle_speed(cfg.r_vehicle_speed);
    fusion.set_r_zero_vel(cfg.r_zero_vel);
    fusion.set_r_stationary_accel(cfg.r_stationary_accel);
    fusion.set_mount_align_rw_var(cfg.mount_align_rw_var);
    fusion.set_align_handoff_delay_s(cfg.align_handoff_delay_s);
    fusion.set_freeze_misalignment_states(cfg.freeze_misalignment_states);
    fusion.set_eskf_mount_source(EskfMountSource::LatchedSeed);
    fusion.set_mount_settle_time_s(cfg.mount_settle_time_s);
    fusion.set_mount_settle_release_sigma_rad(cfg.mount_settle_release_sigma_deg.to_radians());
    fusion.set_mount_settle_zero_cross_covariance(cfg.mount_settle_zero_cross_covariance);
}

fn default_loose_p_diag(
    gnss: GenericGnssSample,
    cfg: EkfCompareConfig,
) -> [f32; LOOSE_ERROR_STATES] {
    let mut p = [1.0_f32; LOOSE_ERROR_STATES];
    let init = cfg.loose_init;
    let pos_n_sigma = (gnss.pos_std_m[0] as f32).max(init.pos_min_sigma_m);
    let pos_e_sigma = (gnss.pos_std_m[1] as f32).max(init.pos_min_sigma_m);
    let pos_d_sigma = (gnss.pos_std_m[2] as f32).max(init.pos_min_sigma_m);
    p[0] = pos_n_sigma * pos_n_sigma;
    p[1] = pos_e_sigma * pos_e_sigma;
    p[2] = pos_d_sigma * pos_d_sigma;
    let vel_sigma = gnss
        .vel_std_mps
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(init.vel_min_sigma_mps as f64) as f32;
    let vel_var = vel_sigma * vel_sigma;
    p[3] = vel_var;
    p[4] = vel_var;
    p[5] = vel_var;
    let attitude_var = init.attitude_sigma_deg.to_radians().powi(2);
    p[6] = attitude_var;
    p[7] = attitude_var;
    p[8] = attitude_var;
    let gyro_bias_sigma = init.gyro_bias_sigma_dps.to_radians();
    p[9] = init.accel_bias_sigma_mps2 * init.accel_bias_sigma_mps2;
    p[10] = p[9];
    p[11] = p[9];
    p[12] = gyro_bias_sigma * gyro_bias_sigma;
    p[13] = p[12];
    p[14] = p[12];
    p[15] = init.accel_scale_sigma * init.accel_scale_sigma;
    p[16] = p[15];
    p[17] = p[15];
    p[18] = init.gyro_scale_sigma * init.gyro_scale_sigma;
    p[19] = p[18];
    p[20] = p[18];
    p[21] = init.mount_sigma_deg.to_radians().powi(2);
    p[22] = p[21];
    p[23] = init.mount_yaw_sigma_deg.to_radians().powi(2);
    p
}

fn loose_imu_delta(prev: GenericImuSample, curr: GenericImuSample, dt: f64) -> LooseImuDelta {
    LooseImuDelta {
        dax_1: (prev.gyro_radps[0] * dt) as f32,
        day_1: (prev.gyro_radps[1] * dt) as f32,
        daz_1: (prev.gyro_radps[2] * dt) as f32,
        dvx_1: (prev.accel_mps2[0] * dt) as f32,
        dvy_1: (prev.accel_mps2[1] * dt) as f32,
        dvz_1: (prev.accel_mps2[2] * dt) as f32,
        dax_2: (curr.gyro_radps[0] * dt) as f32,
        day_2: (curr.gyro_radps[1] * dt) as f32,
        daz_2: (curr.gyro_radps[2] * dt) as f32,
        dvx_2: (curr.accel_mps2[0] * dt) as f32,
        dvy_2: (curr.accel_mps2[1] * dt) as f32,
        dvz_2: (curr.accel_mps2[2] * dt) as f32,
        dt: dt as f32,
    }
}

fn ned_vector_to_ecef(lat_deg: f64, lon_deg: f64, v_ned: [f64; 3]) -> [f32; 3] {
    let c_ne = ecef_to_ned_matrix(lat_deg, lon_deg);
    [
        (c_ne[0][0] * v_ned[0] + c_ne[1][0] * v_ned[1] + c_ne[2][0] * v_ned[2]) as f32,
        (c_ne[0][1] * v_ned[0] + c_ne[1][1] * v_ned[1] + c_ne[2][1] * v_ned[2]) as f32,
        (c_ne[0][2] * v_ned[0] + c_ne[1][2] * v_ned[1] + c_ne[2][2] * v_ned[2]) as f32,
    ]
}

fn ecef_vector_to_ned(lat_deg: f64, lon_deg: f64, v_ecef: [f32; 3]) -> [f64; 3] {
    let c_ne = ecef_to_ned_matrix(lat_deg, lon_deg);
    let v = [v_ecef[0] as f64, v_ecef[1] as f64, v_ecef[2] as f64];
    [
        c_ne[0][0] * v[0] + c_ne[0][1] * v[1] + c_ne[0][2] * v[2],
        c_ne[1][0] * v[0] + c_ne[1][1] * v[1] + c_ne[1][2] * v[2],
        c_ne[2][0] * v[0] + c_ne[2][1] * v[1] + c_ne[2][2] * v[2],
    ]
}

fn ecef_to_ned_matrix(lat_deg: f64, lon_deg: f64) -> [[f64; 3]; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    [
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [-clat * clon, -clat * slon, -slat],
    ]
}
