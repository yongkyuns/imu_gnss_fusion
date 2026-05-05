use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use sensor_fusion::eskf_types::{ESKF_UPDATE_DIAG_TYPES, EskfState};
use sensor_fusion::fusion::SensorFusion;
use sensor_fusion::generated_loose;
use sensor_fusion::loose::{LOOSE_ERROR_STATES, LooseFilter, LooseImuDelta, LoosePredictNoise};
use sim::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, GenericReferenceRpySample, fusion_gnss_sample,
    fusion_imu_sample, load_gnss_samples, load_imu_samples, load_reference_attitude_samples,
    load_reference_mount_samples,
};
use sim::eval::gnss_ins::{as_q64, quat_angle_deg, quat_mul};
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::visualizer::math::{ecef_to_ned, lla_to_ecef};
use sim::visualizer::model::EkfImuSource;
use sim::visualizer::pipeline::EkfCompareConfig;
use sim::visualizer::pipeline::generic::reference_mount_rpy_to_q_vb;
use sim::visualizer::pipeline::synthetic::{
    SyntheticNoiseMode, SyntheticVisualizerConfig, build_synthetic_replay_input,
};

const DIAG_BODY_VEL_Y: usize = 4;
const DIAG_BODY_VEL_Z: usize = 5;
const LOOSE_NHC_GNSS_SPEED_MAX_AGE_S: f64 = 1.0;

#[derive(Parser, Debug)]
struct Args {
    /// Directory containing imu.csv, gnss.csv, and optional reference CSVs.
    #[arg(long)]
    generic_replay_dir: Option<PathBuf>,

    /// Synthetic motion DSL/CSV path to generate a replay on the fly.
    #[arg(long, alias = "synthetic-scenario")]
    synthetic_motion_def: Option<PathBuf>,

    /// Synthetic sensor noise level.
    #[arg(long, value_enum, default_value_t = SyntheticNoiseArg::Truth)]
    synthetic_noise: SyntheticNoiseArg,

    /// Synthetic noise seed.
    #[arg(long, default_value_t = 1)]
    synthetic_seed: u64,

    #[arg(long, default_value_t = 5.0)]
    synthetic_mount_roll_deg: f64,
    #[arg(long, default_value_t = -5.0)]
    synthetic_mount_pitch_deg: f64,
    #[arg(long, default_value_t = 5.0)]
    synthetic_mount_yaw_deg: f64,

    #[arg(long, default_value_t = 100.0)]
    synthetic_imu_hz: f64,
    #[arg(long, default_value_t = 2.0)]
    synthetic_gnss_hz: f64,

    /// Snapshot times in replay-relative seconds. Defaults cover Galbi startup.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "86.736,100,116,120,160,240"
    )]
    times: Vec<f64>,

    /// Optional replay-relative event-trace window, formatted as start,end seconds.
    #[arg(long, value_delimiter = ',')]
    trace_window: Option<Vec<f64>>,

    /// Optional replay-relative interval summary window, formatted as start,end seconds.
    #[arg(long, value_delimiter = ',')]
    summary_window: Option<Vec<f64>>,

    /// Minimum spacing for periodic trace lines. Update events are printed regardless.
    #[arg(long, default_value_t = 0.10)]
    trace_interval_s: f64,

    /// Mount source for the ESKF diagnostic path: internal, external, or ref.
    #[arg(
        long,
        default_value = "internal",
        value_parser = EkfImuSource::from_cli_value
    )]
    misalignment: EkfImuSource,

    /// Freeze residual mount states in the ESKF diagnostic path.
    #[arg(long)]
    freeze_misalignment_states: bool,

    /// Override ESKF/loose lateral no-motion measurement standard deviation.
    #[arg(long)]
    r_body_vel: Option<f32>,

    /// Override ESKF/loose vertical no-motion measurement standard deviation.
    #[arg(long)]
    r_body_vel_z: Option<f32>,

    /// Override ESKF residual mount roll/pitch initial sigma, in degrees.
    #[arg(long)]
    mount_roll_pitch_init_sigma_deg: Option<f32>,

    /// Override ESKF residual mount yaw initial sigma, in degrees.
    #[arg(long)]
    mount_init_sigma_deg: Option<f32>,

    /// Override ESKF initial yaw sigma, in degrees.
    #[arg(long)]
    yaw_init_sigma_deg: Option<f32>,

    /// Override ESKF initial gyro-bias sigma, in deg/s.
    #[arg(long)]
    gyro_bias_init_sigma_dps: Option<f32>,

    /// Override ESKF initial accelerometer-bias sigma, in m/s^2.
    #[arg(long)]
    accel_bias_init_sigma_mps2: Option<f32>,

    /// Diagnostic override for ESKF runtime zero-velocity measurement variance.
    #[arg(long)]
    r_zero_vel: Option<f32>,

    /// Diagnostic-only scale applied to ESKF accelerometer white-noise variance.
    #[arg(long, default_value_t = 1.0)]
    eskf_accel_var_scale: f32,

    /// Diagnostic-only scale applied to ESKF gyro white-noise variance.
    #[arg(long, default_value_t = 1.0)]
    eskf_gyro_var_scale: f32,

    /// Diagnostic-only override for loose gyro-scale initial sigma.
    #[arg(long)]
    loose_gyro_scale_sigma: Option<f32>,

    /// Diagnostic-only override for loose accel-scale initial sigma.
    #[arg(long)]
    loose_accel_scale_sigma: Option<f32>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SyntheticNoiseArg {
    Truth,
    Low,
    Mid,
    High,
}

impl From<SyntheticNoiseArg> for SyntheticNoiseMode {
    fn from(value: SyntheticNoiseArg) -> Self {
        match value {
            SyntheticNoiseArg::Truth => Self::Truth,
            SyntheticNoiseArg::Low => Self::Low,
            SyntheticNoiseArg::Mid => Self::Mid,
            SyntheticNoiseArg::High => Self::High,
        }
    }
}

#[derive(Clone)]
struct Replay {
    imu: Vec<GenericImuSample>,
    gnss: Vec<GenericGnssSample>,
    reference_attitude: Vec<GenericReferenceRpySample>,
    reference_mount: Vec<GenericReferenceRpySample>,
}

#[derive(Clone)]
struct Snapshot {
    target_rel_s: f64,
    t_s: f64,
    eskf: Option<EskfState>,
    loose: Option<LooseSnapshot>,
    reference_mount_q_vb: Option<[f64; 4]>,
    reference_att_q: Option<[f64; 4]>,
    eskf_type_counts: [u32; ESKF_UPDATE_DIAG_TYPES],
    loose_obs_counts: [u32; 9],
    loose_mount_dx_sum: [f32; 3],
    loose_mount_dx_abs_sum: [f32; 3],
    loose_att_dx_sum: [f32; 3],
    loose_att_dx_abs_sum: [f32; 3],
    loose_vel_dx_sum: [f32; 3],
    loose_vel_dx_abs_sum: [f32; 3],
    loose_mount_dx_sum_by_type: [[f32; 3]; 9],
    loose_mount_dx_abs_sum_by_type: [[f32; 3]; 9],
}

#[derive(Clone)]
struct LooseSnapshot {
    nominal: sensor_fusion::loose::LooseNominalState,
    p: [[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
    pos_ecef: [f64; 3],
    last_dx: [f32; LOOSE_ERROR_STATES],
    last_obs_types: Vec<i32>,
}

#[derive(Clone, Default)]
struct TraceState {
    initialized: bool,
    last_trace_t_s: Option<f64>,
    prev_eskf_counts: [u32; ESKF_UPDATE_DIAG_TYPES],
    prev_loose_counts: [u32; 9],
    prev_eskf_sum_dx_mount_roll: [f32; ESKF_UPDATE_DIAG_TYPES],
    prev_eskf_sum_dx_mount_pitch: [f32; ESKF_UPDATE_DIAG_TYPES],
    prev_eskf_sum_dx_mount_yaw: [f32; ESKF_UPDATE_DIAG_TYPES],
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(window) = args.trace_window.as_ref()
        && window.len() != 2
    {
        anyhow::bail!("--trace-window expects exactly two values: start,end");
    }
    if let Some(window) = args.summary_window.as_ref()
        && window.len() != 2
    {
        anyhow::bail!("--summary-window expects exactly two values: start,end");
    }
    let mut replay = load_replay_from_args(&args)?;
    sort_replay(&mut replay);
    let Some(t0) = replay_start_t_s(&replay) else {
        anyhow::bail!("empty replay");
    };
    let mut targets_abs: Vec<f64> = args.times.iter().map(|t| t0 + *t).collect();
    if let Some(window) = args.summary_window.as_ref() {
        targets_abs.push(t0 + window[0]);
        targets_abs.push(t0 + window[1]);
    }
    targets_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    targets_abs.dedup_by(|a, b| (*a - *b).abs() < 1.0e-9);
    let trace_window_abs = args.trace_window.as_ref().map(|w| [t0 + w[0], t0 + w[1]]);
    let snapshots = run_diagnostics(&replay, &args, &targets_abs, trace_window_abs)?;
    print_snapshots(&snapshots, &replay);
    if let Some(window) = args.summary_window.as_ref() {
        print_interval_summary(&snapshots, &replay, [window[0], window[1]]);
    }
    Ok(())
}

fn load_replay_from_args(args: &Args) -> Result<Replay> {
    match (&args.generic_replay_dir, &args.synthetic_motion_def) {
        (Some(_), Some(_)) => {
            anyhow::bail!("choose either --generic-replay-dir or --synthetic-motion-def")
        }
        (Some(dir), None) => load_replay(dir),
        (None, Some(motion_def)) => load_synthetic_replay(args, motion_def),
        (None, None) => {
            anyhow::bail!("provide either --generic-replay-dir or --synthetic-motion-def")
        }
    }
}

fn load_replay(dir: &PathBuf) -> Result<Replay> {
    Ok(Replay {
        imu: load_imu_samples(dir)
            .with_context(|| format!("failed to load {}", dir.join("imu.csv").display()))?,
        gnss: load_gnss_samples(dir)
            .with_context(|| format!("failed to load {}", dir.join("gnss.csv").display()))?,
        reference_attitude: load_reference_attitude_samples(dir)?,
        reference_mount: load_reference_mount_samples(dir)?,
    })
}

fn load_synthetic_replay(args: &Args, motion_def: &PathBuf) -> Result<Replay> {
    let synth_cfg = SyntheticVisualizerConfig {
        motion_def: Some(motion_def.clone()),
        motion_label: motion_def.display().to_string(),
        motion_text: None,
        noise_mode: args.synthetic_noise.into(),
        disable_imu_noise: false,
        disable_gnss_noise: false,
        seed: args.synthetic_seed,
        mount_rpy_deg: [
            args.synthetic_mount_roll_deg,
            args.synthetic_mount_pitch_deg,
            args.synthetic_mount_yaw_deg,
        ],
        imu_hz: args.synthetic_imu_hz,
        gnss_hz: args.synthetic_gnss_hz,
        gnss_time_shift_ms: 0.0,
        early_vel_bias_ned_mps: [0.0; 3],
        early_fault_window_s: None,
    };
    let (replay, _) = build_synthetic_replay_input(&synth_cfg)
        .with_context(|| format!("failed to generate {}", motion_def.display()))?;
    Ok(Replay {
        imu: replay.imu,
        gnss: replay.gnss,
        reference_attitude: replay.reference_attitude,
        reference_mount: replay.reference_mount,
    })
}

fn sort_replay(replay: &mut Replay) {
    replay.imu.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    replay.gnss.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    replay.reference_attitude.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    replay.reference_mount.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn replay_start_t_s(replay: &Replay) -> Option<f64> {
    replay
        .imu
        .first()
        .map(|s| s.t_s)
        .into_iter()
        .chain(replay.gnss.first().map(|s| s.t_s))
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn run_diagnostics(
    replay: &Replay,
    args: &Args,
    targets_abs: &[f64],
    trace_window_abs: Option<[f64; 2]>,
) -> Result<Vec<Snapshot>> {
    let mut cfg = EkfCompareConfig {
        freeze_misalignment_states: args.freeze_misalignment_states,
        ..EkfCompareConfig::default()
    };
    if let Some(r) = args.r_body_vel {
        cfg.r_body_vel = r;
    }
    if let Some(r) = args.r_body_vel_z {
        cfg.r_body_vel_z = r;
    }
    if let Some(sigma) = args.mount_roll_pitch_init_sigma_deg {
        cfg.mount_roll_pitch_init_sigma_deg = sigma;
    }
    if let Some(sigma) = args.mount_init_sigma_deg {
        cfg.mount_init_sigma_deg = sigma;
    }
    if let Some(sigma) = args.yaw_init_sigma_deg {
        cfg.yaw_init_sigma_deg = sigma;
    }
    if let Some(sigma) = args.gyro_bias_init_sigma_dps {
        cfg.gyro_bias_init_sigma_dps = sigma;
    }
    if let Some(sigma) = args.accel_bias_init_sigma_mps2 {
        cfg.accel_bias_init_sigma_mps2 = sigma;
    }
    if let Some(r) = args.r_zero_vel.filter(|v| v.is_finite() && *v >= 0.0) {
        cfg.r_zero_vel = r;
    }
    if args.eskf_accel_var_scale.is_finite() && args.eskf_accel_var_scale > 0.0 {
        if let Some(noise) = cfg.predict_noise.as_mut() {
            noise.accel_var *= args.eskf_accel_var_scale;
        }
    }
    if args.eskf_gyro_var_scale.is_finite() && args.eskf_gyro_var_scale > 0.0 {
        if let Some(noise) = cfg.predict_noise.as_mut() {
            noise.gyro_var *= args.eskf_gyro_var_scale;
        }
    }
    if let Some(sigma) = args
        .loose_gyro_scale_sigma
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        cfg.loose_init.gyro_scale_sigma = sigma;
    }
    if let Some(sigma) = args
        .loose_accel_scale_sigma
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        cfg.loose_init.accel_scale_sigma = sigma;
    }
    let Some(ref_gnss) = replay.gnss.first().copied() else {
        anyhow::bail!("gnss replay is empty");
    };

    let mut fusion = SensorFusion::new();
    apply_fusion_config(&mut fusion, cfg, args.misalignment);
    if let Some(seed_q_vb) = reference_mount_seed_q_vb(replay, args.misalignment) {
        fusion.set_misalignment(seed_q_vb);
    }

    let mut align_fusion = SensorFusion::new();
    apply_fusion_config(&mut align_fusion, cfg, EkfImuSource::Internal);

    let mut loose = LooseFilter::new(
        cfg.loose_predict_noise
            .unwrap_or_else(LoosePredictNoise::lsm6dso_loose_104hz),
    );
    let mut loose_ready = false;
    let mut last_imu: Option<GenericImuSample> = None;
    let mut latest_gnss: Option<GenericGnssSample> = None;
    let mut loose_gnss_cursor = 0usize;
    let mut last_gnss_used_t_s = f64::NEG_INFINITY;
    let mut loose_obs_counts = [0u32; 9];
    let mut loose_mount_dx_sum = [0.0f32; 3];
    let mut loose_mount_dx_abs_sum = [0.0f32; 3];
    let mut loose_att_dx_sum = [0.0f32; 3];
    let mut loose_att_dx_abs_sum = [0.0f32; 3];
    let mut loose_vel_dx_sum = [0.0f32; 3];
    let mut loose_vel_dx_abs_sum = [0.0f32; 3];
    let mut loose_mount_dx_sum_by_type = [[0.0f32; 3]; 9];
    let mut loose_mount_dx_abs_sum_by_type = [[0.0f32; 3]; 9];
    let mut trace_state = TraceState::default();

    let mut snapshots = Vec::new();
    let mut target_idx = 0usize;

    for_each_event(&replay.imu, &replay.gnss, |event| match event {
        ReplayEvent::Imu(_, sample) => {
            let _ = fusion.process_imu(fusion_imu_sample(*sample));
            let _ = align_fusion.process_imu(fusion_imu_sample(*sample));
            let Some(prev) = last_imu.replace(*sample) else {
                capture_due_snapshots(
                    replay,
                    &fusion,
                    &loose,
                    loose_ready,
                    loose_obs_counts,
                    loose_mount_dx_sum,
                    loose_mount_dx_abs_sum,
                    loose_att_dx_sum,
                    loose_att_dx_abs_sum,
                    loose_vel_dx_sum,
                    loose_vel_dx_abs_sum,
                    loose_mount_dx_sum_by_type,
                    loose_mount_dx_abs_sum_by_type,
                    targets_abs,
                    &mut target_idx,
                    sample.t_s,
                    &mut snapshots,
                );
                return;
            };
            if loose_ready {
                let dt = (sample.t_s - prev.t_s).max(0.0);
                if dt > 0.0 && dt <= 1.0 {
                    let imu = loose_imu_delta_from_vehicle(
                        prev.gyro_radps,
                        prev.accel_mps2,
                        sample.gyro_radps,
                        sample.accel_mps2,
                        dt,
                    );
                    loose.predict(imu);
                    while loose_gnss_cursor < replay.gnss.len()
                        && replay.gnss[loose_gnss_cursor].t_s <= sample.t_s + 1.0e-9
                    {
                        latest_gnss = Some(replay.gnss[loose_gnss_cursor]);
                        loose_gnss_cursor += 1;
                    }
                    let mut gps_pos = None;
                    let mut gps_vel = None;
                    let mut gps_pos_std = 0.0f32;
                    let mut gps_vel_std = None;
                    let mut dt_since_gnss = 1.0f32;
                    if let Some(gnss) = latest_gnss
                        && (0.0..=0.05).contains(&(sample.t_s - gnss.t_s))
                        && gnss.t_s != last_gnss_used_t_s
                    {
                        gps_pos = Some(lla_to_ecef(gnss.lat_deg, gnss.lon_deg, gnss.height_m));
                        gps_vel = Some(
                            ned_vector_to_ecef(gnss.lat_deg, gnss.lon_deg, gnss.vel_ned_mps)
                                .map(|v| v as f32),
                        );
                        gps_pos_std = ((gnss.pos_std_m[0] + gnss.pos_std_m[1] + gnss.pos_std_m[2])
                            / 3.0)
                            .max(0.1) as f32;
                        gps_vel_std = Some(gnss.vel_std_mps.map(|v| v.max(0.01) as f32));
                        dt_since_gnss = if last_gnss_used_t_s.is_finite() {
                            (gnss.t_s - last_gnss_used_t_s).clamp(1.0e-3, 1.0) as f32
                        } else {
                            1.0
                        };
                        last_gnss_used_t_s = gnss.t_s;
                    }
                    let nhc_gate_speed_mps = latest_gnss.and_then(|gnss| {
                        let age_s = sample.t_s - gnss.t_s;
                        (0.0..=LOOSE_NHC_GNSS_SPEED_MAX_AGE_S)
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
                    if !loose.last_obs_types().is_empty() {
                        let dx = loose.last_dx();
                        let loose_snap = LooseSnapshot {
                            nominal: *loose.nominal(),
                            p: *loose.covariance(),
                            pos_ecef: loose.shadow_pos_ecef(),
                            last_dx: *dx,
                            last_obs_types: loose.last_obs_types().to_vec(),
                        };
                        let dx_as_eskf = transform_loose_dx_to_eskf(&loose_snap, dx, ref_gnss);
                        for axis in 0..3 {
                            let value = dx[21 + axis];
                            loose_mount_dx_sum[axis] += value;
                            loose_mount_dx_abs_sum[axis] += value.abs();
                            let att_value = dx_as_eskf[axis];
                            loose_att_dx_sum[axis] += att_value;
                            loose_att_dx_abs_sum[axis] += att_value.abs();
                            let vel_value = dx_as_eskf[3 + axis];
                            loose_vel_dx_sum[axis] += vel_value;
                            loose_vel_dx_abs_sum[axis] += vel_value.abs();
                        }
                        for (obs_idx, ty) in loose.last_obs_types().iter().copied().enumerate() {
                            let Some(sum) = loose_mount_dx_sum_by_type.get_mut(ty as usize) else {
                                continue;
                            };
                            let Some(abs_sum) = loose_mount_dx_abs_sum_by_type.get_mut(ty as usize)
                            else {
                                continue;
                            };
                            let row_dx = &loose.last_dx_by_obs()[obs_idx];
                            for axis in 0..3 {
                                let value = row_dx[21 + axis];
                                sum[axis] += value;
                                abs_sum[axis] += value.abs();
                            }
                        }
                    }
                    for ty in loose.last_obs_types() {
                        if let Some(count) = loose_obs_counts.get_mut(*ty as usize) {
                            *count += 1;
                        }
                    }
                }
            }
            capture_due_snapshots(
                replay,
                &fusion,
                &loose,
                loose_ready,
                loose_obs_counts,
                loose_mount_dx_sum,
                loose_mount_dx_abs_sum,
                loose_att_dx_sum,
                loose_att_dx_abs_sum,
                loose_vel_dx_sum,
                loose_vel_dx_abs_sum,
                loose_mount_dx_sum_by_type,
                loose_mount_dx_abs_sum_by_type,
                targets_abs,
                &mut target_idx,
                sample.t_s,
                &mut snapshots,
            );
            maybe_print_trace(
                replay,
                ref_gnss,
                trace_window_abs,
                args.trace_interval_s,
                &mut trace_state,
                "imu",
                sample.t_s,
                &fusion,
                &loose,
                loose_ready,
                loose_obs_counts,
            );
        }
        ReplayEvent::Gnss(index, sample) => {
            let _ = fusion.process_gnss(fusion_gnss_sample(*sample));
            let _ = align_fusion.process_gnss(fusion_gnss_sample(*sample));
            latest_gnss = Some(*sample);
            if !loose_ready && align_fusion.mount_ready() {
                let speed = sample.vel_ned_mps[0].hypot(sample.vel_ned_mps[1]);
                if speed >= 0.5 {
                    let yaw_rad = sample.vel_ned_mps[1].atan2(sample.vel_ned_mps[0]) as f32;
                    let pos_ecef = lla_to_ecef(sample.lat_deg, sample.lon_deg, sample.height_m);
                    let vel_ecef =
                        ned_vector_to_ecef(sample.lat_deg, sample.lon_deg, sample.vel_ned_mps)
                            .map(|v| v as f32);
                    loose.init_seeded_vehicle_from_nav_ecef_state(
                        yaw_rad,
                        sample.lat_deg,
                        sample.lon_deg,
                        pos_ecef,
                        vel_ecef,
                        Some(default_loose_p_diag(*sample, cfg)),
                        None,
                    );
                    if let Some(seed_q) = align_fusion.mount_q_vb() {
                        loose.set_mount_quat(seed_q);
                    }
                    loose_ready = true;
                    loose_gnss_cursor = index + 1;
                    last_gnss_used_t_s = sample.t_s;
                }
            }
            capture_due_snapshots(
                replay,
                &fusion,
                &loose,
                loose_ready,
                loose_obs_counts,
                loose_mount_dx_sum,
                loose_mount_dx_abs_sum,
                loose_att_dx_sum,
                loose_att_dx_abs_sum,
                loose_vel_dx_sum,
                loose_vel_dx_abs_sum,
                loose_mount_dx_sum_by_type,
                loose_mount_dx_abs_sum_by_type,
                targets_abs,
                &mut target_idx,
                sample.t_s,
                &mut snapshots,
            );
            maybe_print_trace(
                replay,
                ref_gnss,
                trace_window_abs,
                args.trace_interval_s,
                &mut trace_state,
                "gnss",
                sample.t_s,
                &fusion,
                &loose,
                loose_ready,
                loose_obs_counts,
            );
        }
    });

    let final_t = replay
        .imu
        .last()
        .map(|s| s.t_s)
        .unwrap_or(ref_gnss.t_s)
        .max(replay.gnss.last().map(|s| s.t_s).unwrap_or(ref_gnss.t_s));
    capture_due_snapshots(
        replay,
        &fusion,
        &loose,
        loose_ready,
        loose_obs_counts,
        loose_mount_dx_sum,
        loose_mount_dx_abs_sum,
        loose_att_dx_sum,
        loose_att_dx_abs_sum,
        loose_vel_dx_sum,
        loose_vel_dx_abs_sum,
        loose_mount_dx_sum_by_type,
        loose_mount_dx_abs_sum_by_type,
        targets_abs,
        &mut target_idx,
        final_t,
        &mut snapshots,
    );
    Ok(snapshots)
}

fn apply_fusion_config(fusion: &mut SensorFusion, cfg: EkfCompareConfig, mode: EkfImuSource) {
    fusion.set_align_config(cfg.align);
    if let Some(noise) = cfg.predict_noise {
        fusion.set_predict_noise(noise);
    }
    fusion.set_r_body_vel_yz(cfg.r_body_vel, cfg.r_body_vel_z);
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
    fusion.set_eskf_mount_source(mode.eskf_mount_source());
    fusion.set_mount_settle_time_s(cfg.mount_settle_time_s);
    fusion.set_mount_settle_release_sigma_rad(cfg.mount_settle_release_sigma_deg.to_radians());
    fusion.set_mount_settle_zero_cross_covariance(cfg.mount_settle_zero_cross_covariance);
}

fn capture_due_snapshots(
    replay: &Replay,
    fusion: &SensorFusion,
    loose: &LooseFilter,
    loose_ready: bool,
    loose_obs_counts: [u32; 9],
    loose_mount_dx_sum: [f32; 3],
    loose_mount_dx_abs_sum: [f32; 3],
    loose_att_dx_sum: [f32; 3],
    loose_att_dx_abs_sum: [f32; 3],
    loose_vel_dx_sum: [f32; 3],
    loose_vel_dx_abs_sum: [f32; 3],
    loose_mount_dx_sum_by_type: [[f32; 3]; 9],
    loose_mount_dx_abs_sum_by_type: [[f32; 3]; 9],
    targets_abs: &[f64],
    target_idx: &mut usize,
    t_s: f64,
    snapshots: &mut Vec<Snapshot>,
) {
    while let Some(&target_t_s) = targets_abs.get(*target_idx) {
        if t_s < target_t_s {
            return;
        }
        snapshots.push(Snapshot {
            target_rel_s: target_t_s - replay_start_t_s(replay).unwrap_or(0.0),
            t_s,
            eskf: fusion.eskf().copied(),
            loose: loose_ready.then(|| LooseSnapshot {
                nominal: *loose.nominal(),
                p: *loose.covariance(),
                pos_ecef: loose.shadow_pos_ecef(),
                last_dx: *loose.last_dx(),
                last_obs_types: loose.last_obs_types().to_vec(),
            }),
            reference_mount_q_vb: reference_mount_at(&replay.reference_mount, t_s),
            reference_att_q: reference_attitude_at(&replay.reference_attitude, t_s),
            eskf_type_counts: fusion
                .eskf()
                .map(|e| e.update_diag.type_counts)
                .unwrap_or([0; ESKF_UPDATE_DIAG_TYPES]),
            loose_obs_counts,
            loose_mount_dx_sum,
            loose_mount_dx_abs_sum,
            loose_att_dx_sum,
            loose_att_dx_abs_sum,
            loose_vel_dx_sum,
            loose_vel_dx_abs_sum,
            loose_mount_dx_sum_by_type,
            loose_mount_dx_abs_sum_by_type,
        });
        *target_idx += 1;
    }
}

fn print_snapshots(snapshots: &[Snapshot], replay: &Replay) {
    let Some(ref_gnss) = replay.gnss.first().copied() else {
        return;
    };
    for snapshot in snapshots {
        println!(
            "[covhist] snapshot target_rel_s={:.3} sample_t_s={:.6}",
            snapshot.target_rel_s, snapshot.t_s
        );
        print_state_summary(snapshot, ref_gnss);
        if let (Some(eskf), Some(loose)) = (&snapshot.eskf, &snapshot.loose) {
            let p_loose_as_eskf = transform_loose_cov_to_eskf(loose, ref_gnss);
            print_covariance_summary(eskf, &p_loose_as_eskf);
            print_mount_correlations("ESKF", &eskf.p);
            print_mount_correlations("LooseAsESKF", &p_loose_as_eskf);
            print_update_summary(snapshot, loose);
        }
    }
}

fn print_interval_summary(snapshots: &[Snapshot], replay: &Replay, window_rel_s: [f64; 2]) {
    let Some(ref_gnss) = replay.gnss.first().copied() else {
        return;
    };
    let Some(start) = nearest_snapshot(snapshots, window_rel_s[0]) else {
        println!(
            "[covhist-interval] missing start snapshot rel_s={:.3}",
            window_rel_s[0]
        );
        return;
    };
    let Some(end) = nearest_snapshot(snapshots, window_rel_s[1]) else {
        println!(
            "[covhist-interval] missing end snapshot rel_s={:.3}",
            window_rel_s[1]
        );
        return;
    };

    println!(
        "[covhist-interval] window_rel_s=[{:.3},{:.3}] sample_rel_s=[{:.3},{:.3}]",
        window_rel_s[0], window_rel_s[1], start.target_rel_s, end.target_rel_s
    );
    print_interval_state_summary(start, end, ref_gnss);
    print_interval_update_summary(start, end);
    if let (Some(start_eskf), Some(end_eskf), Some(start_loose), Some(end_loose)) =
        (&start.eskf, &end.eskf, &start.loose, &end.loose)
    {
        let start_loose_p = transform_loose_cov_to_eskf(start_loose, ref_gnss);
        let end_loose_p = transform_loose_cov_to_eskf(end_loose, ref_gnss);
        print_interval_sigma_summary(start_eskf, end_eskf, &start_loose_p, &end_loose_p);
        print_interval_correlation_summary(
            &start_eskf.p,
            &end_eskf.p,
            &start_loose_p,
            &end_loose_p,
        );
    }
}

fn nearest_snapshot(snapshots: &[Snapshot], rel_s: f64) -> Option<&Snapshot> {
    snapshots.iter().min_by(|a, b| {
        (a.target_rel_s - rel_s)
            .abs()
            .partial_cmp(&(b.target_rel_s - rel_s).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn print_interval_state_summary(start: &Snapshot, end: &Snapshot, ref_gnss: GenericGnssSample) {
    let start_metrics = state_metrics(start, ref_gnss);
    let end_metrics = state_metrics(end, ref_gnss);
    println!(
        "[covhist-interval] qerr_mount_deg eskf={:.6}->{:.6} d={:.6} loose={:.6}->{:.6} d={:.6}",
        start_metrics.eskf_mount_qerr,
        end_metrics.eskf_mount_qerr,
        end_metrics.eskf_mount_qerr - start_metrics.eskf_mount_qerr,
        start_metrics.loose_mount_qerr,
        end_metrics.loose_mount_qerr,
        end_metrics.loose_mount_qerr - start_metrics.loose_mount_qerr,
    );
    println!(
        "[covhist-interval] qerr_att_deg eskf={:.6}->{:.6} d={:.6} loose={:.6}->{:.6} d={:.6}",
        start_metrics.eskf_att_qerr,
        end_metrics.eskf_att_qerr,
        end_metrics.eskf_att_qerr - start_metrics.eskf_att_qerr,
        start_metrics.loose_att_qerr,
        end_metrics.loose_att_qerr,
        end_metrics.loose_att_qerr - start_metrics.loose_att_qerr,
    );
    println!(
        "[covhist-interval] nhc_residual_end_mps eskf_yz=[{:.6},{:.6}] loose_yz=[{:.6},{:.6}]",
        end_metrics.eskf_nhc[0],
        end_metrics.eskf_nhc[1],
        end_metrics.loose_nhc[0],
        end_metrics.loose_nhc[1],
    );
}

#[derive(Clone, Copy)]
struct StateMetrics {
    eskf_mount_qerr: f64,
    loose_mount_qerr: f64,
    eskf_att_qerr: f64,
    loose_att_qerr: f64,
    eskf_nhc: [f64; 2],
    loose_nhc: [f64; 2],
}

fn state_metrics(snapshot: &Snapshot, ref_gnss: GenericGnssSample) -> StateMetrics {
    let eskf_mount_q = snapshot.eskf.as_ref().map(|e| {
        as_q64([
            e.nominal.qcs0,
            e.nominal.qcs1,
            e.nominal.qcs2,
            e.nominal.qcs3,
        ])
    });
    let loose_mount_q = snapshot.loose.as_ref().map(|l| {
        as_q64([
            l.nominal.qcs0,
            l.nominal.qcs1,
            l.nominal.qcs2,
            l.nominal.qcs3,
        ])
    });
    StateMetrics {
        eskf_mount_qerr: eskf_mount_q
            .zip(snapshot.reference_mount_q_vb)
            .map(|(a, b)| quat_angle_deg(a, b))
            .unwrap_or(f64::NAN),
        loose_mount_qerr: loose_mount_q
            .zip(snapshot.reference_mount_q_vb)
            .map(|(a, b)| quat_angle_deg(a, b))
            .unwrap_or(f64::NAN),
        eskf_att_qerr: snapshot
            .eskf
            .as_ref()
            .and_then(|e| snapshot.reference_att_q.map(|r| (e, r)))
            .map(|(e, r)| quat_angle_deg(eskf_att_q(e), r))
            .unwrap_or(f64::NAN),
        loose_att_qerr: snapshot
            .loose
            .as_ref()
            .and_then(|l| snapshot.reference_att_q.map(|r| (l, r)))
            .map(|(l, r)| quat_angle_deg(loose_att_q_ned(l, ref_gnss), r))
            .unwrap_or(f64::NAN),
        eskf_nhc: snapshot
            .eskf
            .as_ref()
            .map(eskf_nhc_residual_yz)
            .unwrap_or([f64::NAN; 2]),
        loose_nhc: snapshot
            .loose
            .as_ref()
            .map(loose_nhc_residual_yz)
            .unwrap_or([f64::NAN; 2]),
    }
}

fn print_interval_update_summary(start: &Snapshot, end: &Snapshot) {
    let eskf_delta = count_delta(&end.eskf_type_counts, &start.eskf_type_counts);
    let loose_delta = count_delta(&end.loose_obs_counts, &start.loose_obs_counts);
    println!(
        "[covhist-interval] eskf_update_delta pos_xy={} pos_d={} vel_xy={} vel_d={} zero_xy={} zero_d={} nhc_y={} nhc_z={}",
        eskf_delta[0],
        eskf_delta[8],
        eskf_delta[1],
        eskf_delta[9],
        eskf_delta[2],
        eskf_delta[10],
        eskf_delta[DIAG_BODY_VEL_Y],
        eskf_delta[DIAG_BODY_VEL_Z]
    );
    println!(
        "[covhist-interval] loose_obs_delta pos=[{},{},{}] vel=[{},{},{}] nhc_y={} nhc_z={}",
        loose_delta[1],
        loose_delta[2],
        loose_delta[3],
        loose_delta[4],
        loose_delta[5],
        loose_delta[6],
        loose_delta[7],
        loose_delta[8]
    );
    if let (Some(start_eskf), Some(end_eskf)) = (&start.eskf, &end.eskf) {
        for (label, idx) in selected_eskf_diag_types() {
            println!(
                "[covhist-interval] eskf_mount_dx type={} count_delta={} net_deg=[{:.6},{:.6},{:.6}] abs_deg=[{:.6},{:.6},{:.6}]",
                label,
                eskf_delta[idx],
                rad_f32_to_deg(
                    end_eskf.update_diag.sum_dx_mount_roll[idx]
                        - start_eskf.update_diag.sum_dx_mount_roll[idx]
                ),
                rad_f32_to_deg(
                    end_eskf.update_diag.sum_dx_mount_pitch[idx]
                        - start_eskf.update_diag.sum_dx_mount_pitch[idx]
                ),
                rad_f32_to_deg(
                    end_eskf.update_diag.sum_dx_mount_yaw[idx]
                        - start_eskf.update_diag.sum_dx_mount_yaw[idx]
                ),
                rad_f32_to_deg(
                    end_eskf.update_diag.sum_abs_dx_mount_roll[idx]
                        - start_eskf.update_diag.sum_abs_dx_mount_roll[idx]
                ),
                rad_f32_to_deg(
                    end_eskf.update_diag.sum_abs_dx_mount_pitch[idx]
                        - start_eskf.update_diag.sum_abs_dx_mount_pitch[idx]
                ),
                rad_f32_to_deg(
                    end_eskf.update_diag.sum_abs_dx_mount_yaw[idx]
                        - start_eskf.update_diag.sum_abs_dx_mount_yaw[idx]
                ),
            );
            println!(
                "[covhist-interval] eskf_att_dx type={} count_delta={} net_deg=[{:.6},{:.6},{:.6}] abs_deg=[{:.6},{:.6},{:.6}]",
                label,
                eskf_delta[idx],
                rad_f32_to_deg(
                    end_eskf.update_diag.sum_dx_att_roll[idx]
                        - start_eskf.update_diag.sum_dx_att_roll[idx]
                ),
                rad_f32_to_deg(
                    end_eskf.update_diag.sum_dx_pitch[idx]
                        - start_eskf.update_diag.sum_dx_pitch[idx]
                ),
                rad_f32_to_deg(
                    end_eskf.update_diag.sum_dx_yaw[idx] - start_eskf.update_diag.sum_dx_yaw[idx]
                ),
                rad_f32_to_deg(
                    end_eskf.update_diag.sum_abs_dx_att_roll[idx]
                        - start_eskf.update_diag.sum_abs_dx_att_roll[idx]
                ),
                rad_f32_to_deg(
                    end_eskf.update_diag.sum_abs_dx_pitch[idx]
                        - start_eskf.update_diag.sum_abs_dx_pitch[idx]
                ),
                rad_f32_to_deg(
                    end_eskf.update_diag.sum_abs_dx_yaw[idx]
                        - start_eskf.update_diag.sum_abs_dx_yaw[idx]
                ),
            );
            println!(
                "[covhist-interval] eskf_vel_dx type={} count_delta={} net_mps=[{:.6},{:.6},{:.6}] abs_mps=[{:.6},{:.6},{:.6}]",
                label,
                eskf_delta[idx],
                end_eskf.update_diag.sum_dx_vel_n[idx] - start_eskf.update_diag.sum_dx_vel_n[idx],
                end_eskf.update_diag.sum_dx_vel_e[idx] - start_eskf.update_diag.sum_dx_vel_e[idx],
                end_eskf.update_diag.sum_dx_vel_d[idx] - start_eskf.update_diag.sum_dx_vel_d[idx],
                end_eskf.update_diag.sum_abs_dx_vel_n[idx]
                    - start_eskf.update_diag.sum_abs_dx_vel_n[idx],
                end_eskf.update_diag.sum_abs_dx_vel_e[idx]
                    - start_eskf.update_diag.sum_abs_dx_vel_e[idx],
                end_eskf.update_diag.sum_abs_dx_vel_d[idx]
                    - start_eskf.update_diag.sum_abs_dx_vel_d[idx],
            );
            println!(
                "[covhist-interval] eskf_innov type={} count_delta={} net={:.6} abs={:.6} nis_sum={:.6} nis_max={:.6}",
                label,
                eskf_delta[idx],
                end_eskf.update_diag.sum_innovation[idx]
                    - start_eskf.update_diag.sum_innovation[idx],
                end_eskf.update_diag.sum_abs_innovation[idx]
                    - start_eskf.update_diag.sum_abs_innovation[idx],
                end_eskf.update_diag.sum_nis[idx] - start_eskf.update_diag.sum_nis[idx],
                end_eskf.update_diag.max_nis[idx],
            );
        }
    }
    println!(
        "[covhist-interval] loose_mount_dx total_net_deg=[{:.6},{:.6},{:.6}] total_abs_deg=[{:.6},{:.6},{:.6}]",
        rad_f32_to_deg(end.loose_mount_dx_sum[0] - start.loose_mount_dx_sum[0]),
        rad_f32_to_deg(end.loose_mount_dx_sum[1] - start.loose_mount_dx_sum[1]),
        rad_f32_to_deg(end.loose_mount_dx_sum[2] - start.loose_mount_dx_sum[2]),
        rad_f32_to_deg(end.loose_mount_dx_abs_sum[0] - start.loose_mount_dx_abs_sum[0]),
        rad_f32_to_deg(end.loose_mount_dx_abs_sum[1] - start.loose_mount_dx_abs_sum[1]),
        rad_f32_to_deg(end.loose_mount_dx_abs_sum[2] - start.loose_mount_dx_abs_sum[2]),
    );
    for (ty, label) in [
        (1usize, "pos_x"),
        (2, "pos_y"),
        (3, "pos_z"),
        (4, "vel_x"),
        (5, "vel_y"),
        (6, "vel_z"),
        (7, "nhc_y"),
        (8, "nhc_z"),
    ] {
        let net = [
            end.loose_mount_dx_sum_by_type[ty][0] - start.loose_mount_dx_sum_by_type[ty][0],
            end.loose_mount_dx_sum_by_type[ty][1] - start.loose_mount_dx_sum_by_type[ty][1],
            end.loose_mount_dx_sum_by_type[ty][2] - start.loose_mount_dx_sum_by_type[ty][2],
        ];
        let abs = [
            end.loose_mount_dx_abs_sum_by_type[ty][0] - start.loose_mount_dx_abs_sum_by_type[ty][0],
            end.loose_mount_dx_abs_sum_by_type[ty][1] - start.loose_mount_dx_abs_sum_by_type[ty][1],
            end.loose_mount_dx_abs_sum_by_type[ty][2] - start.loose_mount_dx_abs_sum_by_type[ty][2],
        ];
        println!(
            "[covhist-interval] loose_mount_dx type={} net_deg=[{:.6},{:.6},{:.6}] abs_deg=[{:.6},{:.6},{:.6}]",
            label,
            rad_f32_to_deg(net[0]),
            rad_f32_to_deg(net[1]),
            rad_f32_to_deg(net[2]),
            rad_f32_to_deg(abs[0]),
            rad_f32_to_deg(abs[1]),
            rad_f32_to_deg(abs[2]),
        );
    }
    println!(
        "[covhist-interval] loose_att_dx_as_eskf total_net_deg=[{:.6},{:.6},{:.6}] total_abs_deg=[{:.6},{:.6},{:.6}]",
        rad_f32_to_deg(end.loose_att_dx_sum[0] - start.loose_att_dx_sum[0]),
        rad_f32_to_deg(end.loose_att_dx_sum[1] - start.loose_att_dx_sum[1]),
        rad_f32_to_deg(end.loose_att_dx_sum[2] - start.loose_att_dx_sum[2]),
        rad_f32_to_deg(end.loose_att_dx_abs_sum[0] - start.loose_att_dx_abs_sum[0]),
        rad_f32_to_deg(end.loose_att_dx_abs_sum[1] - start.loose_att_dx_abs_sum[1]),
        rad_f32_to_deg(end.loose_att_dx_abs_sum[2] - start.loose_att_dx_abs_sum[2]),
    );
    println!(
        "[covhist-interval] loose_vel_dx_as_eskf total_net_mps=[{:.6},{:.6},{:.6}] total_abs_mps=[{:.6},{:.6},{:.6}]",
        end.loose_vel_dx_sum[0] - start.loose_vel_dx_sum[0],
        end.loose_vel_dx_sum[1] - start.loose_vel_dx_sum[1],
        end.loose_vel_dx_sum[2] - start.loose_vel_dx_sum[2],
        end.loose_vel_dx_abs_sum[0] - start.loose_vel_dx_abs_sum[0],
        end.loose_vel_dx_abs_sum[1] - start.loose_vel_dx_abs_sum[1],
        end.loose_vel_dx_abs_sum[2] - start.loose_vel_dx_abs_sum[2],
    );
}

fn rad_f32_to_deg(value: f32) -> f64 {
    (value as f64).to_degrees()
}

fn print_interval_sigma_summary(
    start_eskf: &EskfState,
    end_eskf: &EskfState,
    start_loose_p: &[[f32; 18]; 18],
    end_loose_p: &[[f32; 18]; 18],
) {
    for i in [0usize, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17] {
        println!(
            "[covhist-interval] sigma_deg state={} eskf={:.6}->{:.6} loose={:.6}->{:.6}",
            STATE_NAMES[i],
            sigma_deg(&start_eskf.p, i),
            sigma_deg(&end_eskf.p, i),
            sigma_deg(start_loose_p, i),
            sigma_deg(end_loose_p, i),
        );
    }
}

fn print_interval_correlation_summary(
    start_eskf_p: &[[f32; 18]; 18],
    end_eskf_p: &[[f32; 18]; 18],
    start_loose_p: &[[f32; 18]; 18],
    end_loose_p: &[[f32; 18]; 18],
) {
    for (label, i, j) in selected_corr_pairs() {
        println!(
            "[covhist-interval] corr {} eskf={:.3}->{:.3} loose={:.3}->{:.3}",
            label,
            corr_from_cov(start_eskf_p, i, j),
            corr_from_cov(end_eskf_p, i, j),
            corr_from_cov(start_loose_p, i, j),
            corr_from_cov(end_loose_p, i, j),
        );
    }
}

fn selected_corr_pairs() -> [(&'static str, usize, usize); 8] {
    [
        ("att_x:mount_roll", 0, 15),
        ("att_x:mount_pitch", 0, 16),
        ("att_y:mount_pitch", 1, 16),
        ("att_y:mount_yaw", 1, 17),
        ("att_z:mount_yaw", 2, 17),
        ("bgx:mount_roll", 9, 15),
        ("bax:mount_yaw", 12, 17),
        ("bay:mount_yaw", 13, 17),
    ]
}

fn print_state_summary(snapshot: &Snapshot, ref_gnss: GenericGnssSample) {
    let eskf_mount_q = snapshot.eskf.as_ref().map(|e| {
        as_q64([
            e.nominal.qcs0,
            e.nominal.qcs1,
            e.nominal.qcs2,
            e.nominal.qcs3,
        ])
    });
    let loose_mount_q = snapshot.loose.as_ref().map(|l| {
        as_q64([
            l.nominal.qcs0,
            l.nominal.qcs1,
            l.nominal.qcs2,
            l.nominal.qcs3,
        ])
    });
    let ref_mount_q = snapshot.reference_mount_q_vb;
    let eskf_mount_qerr = eskf_mount_q
        .zip(ref_mount_q)
        .map(|(a, b)| quat_angle_deg(a, b))
        .unwrap_or(f64::NAN);
    let loose_mount_qerr = loose_mount_q
        .zip(ref_mount_q)
        .map(|(a, b)| quat_angle_deg(a, b))
        .unwrap_or(f64::NAN);
    let eskf_att_qerr = snapshot
        .eskf
        .as_ref()
        .and_then(|e| snapshot.reference_att_q.map(|r| (e, r)))
        .map(|(e, r)| quat_angle_deg(eskf_att_q(e), r))
        .unwrap_or(f64::NAN);
    let loose_att_qerr = snapshot
        .loose
        .as_ref()
        .and_then(|l| snapshot.reference_att_q.map(|r| (l, r)))
        .map(|(l, r)| quat_angle_deg(loose_att_q_ned(l, ref_gnss), r))
        .unwrap_or(f64::NAN);
    let eskf_nhc = snapshot
        .eskf
        .as_ref()
        .map(eskf_nhc_residual_yz)
        .unwrap_or([f64::NAN; 2]);
    let loose_nhc = snapshot
        .loose
        .as_ref()
        .map(loose_nhc_residual_yz)
        .unwrap_or([f64::NAN; 2]);

    println!(
        "[covhist] state mount_qerr_deg eskf={:.6} loose={:.6} att_qerr_deg eskf={:.6} loose={:.6}",
        eskf_mount_qerr, loose_mount_qerr, eskf_att_qerr, loose_att_qerr
    );
    println!(
        "[covhist] nhc_residual_mps eskf_y={:.6} eskf_z={:.6} loose_y={:.6} loose_z={:.6}",
        eskf_nhc[0], eskf_nhc[1], loose_nhc[0], loose_nhc[1]
    );
}

fn print_covariance_summary(eskf: &EskfState, loose_as_eskf: &[[f32; 18]; 18]) {
    for (label, rows, cols) in [
        ("att", &[0usize, 1, 2][..], &[0usize, 1, 2][..]),
        ("vel", &[3usize, 4, 5][..], &[3usize, 4, 5][..]),
        ("pos", &[6usize, 7, 8][..], &[6usize, 7, 8][..]),
        ("gyro_bias", &[9usize, 10, 11][..], &[9usize, 10, 11][..]),
        ("accel_bias", &[12usize, 13, 14][..], &[12usize, 13, 14][..]),
        ("mount", &[15usize, 16, 17][..], &[15usize, 16, 17][..]),
        ("att_mount", &[0usize, 1, 2][..], &[15usize, 16, 17][..]),
        ("vel_mount", &[3usize, 4, 5][..], &[15usize, 16, 17][..]),
        (
            "gyro_bias_mount",
            &[9usize, 10, 11][..],
            &[15usize, 16, 17][..],
        ),
        (
            "accel_bias_mount",
            &[12usize, 13, 14][..],
            &[15usize, 16, 17][..],
        ),
    ] {
        println!(
            "[covhist] cov_block label={} rms_abs_diff={:.6e} rms_rel_diff={:.6e}",
            label,
            block_rms_abs_diff(&eskf.p, loose_as_eskf, rows, cols),
            block_rms_rel_diff(&eskf.p, loose_as_eskf, rows, cols)
        );
    }
    for i in [
        0usize, 1, 2, // attitude
        3, 4, 5, // velocity
        9, 10, 11, // gyro bias
        12, 13, 14, // accel bias
        15, 16, 17, // mount
    ] {
        println!(
            "[covhist] sigma state={} eskf={:.6e} loose_as_eskf={:.6e}",
            STATE_NAMES[i],
            eskf.p[i][i].max(0.0).sqrt(),
            loose_as_eskf[i][i].max(0.0).sqrt()
        );
    }
}

fn print_mount_correlations(label: &str, p: &[[f32; 18]; 18]) {
    let mut entries = Vec::new();
    for i in 0..15 {
        for j in 15..18 {
            entries.push((corr_from_cov(p, i, j).abs(), i, j, corr_from_cov(p, i, j)));
        }
    }
    entries.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, i, j, corr) in entries.into_iter().take(8) {
        println!(
            "[covhist] top_mount_corr system={} state={} mount={} corr={:.6}",
            label, STATE_NAMES[i], STATE_NAMES[j], corr
        );
    }
}

fn print_update_summary(snapshot: &Snapshot, loose: &LooseSnapshot) {
    let dy = DIAG_BODY_VEL_Y;
    let dz = DIAG_BODY_VEL_Z;
    println!(
        "[covhist] eskf_update_counts pos_xy={} pos_d={} vel_xy={} vel_d={} zero_xy={} zero_d={} nhc_y={} nhc_z={}",
        snapshot.eskf_type_counts[0],
        snapshot.eskf_type_counts[8],
        snapshot.eskf_type_counts[1],
        snapshot.eskf_type_counts[9],
        snapshot.eskf_type_counts[2],
        snapshot.eskf_type_counts[10],
        snapshot.eskf_type_counts[dy],
        snapshot.eskf_type_counts[dz]
    );
    println!(
        "[covhist] loose_obs_counts pos=[{},{},{}] vel=[{},{},{}] nhc_y={} nhc_z={}",
        snapshot.loose_obs_counts[1],
        snapshot.loose_obs_counts[2],
        snapshot.loose_obs_counts[3],
        snapshot.loose_obs_counts[4],
        snapshot.loose_obs_counts[5],
        snapshot.loose_obs_counts[6],
        snapshot.loose_obs_counts[7],
        snapshot.loose_obs_counts[8]
    );
    println!(
        "[covhist] loose_last_obs types={:?} mount_dx_deg=[{:.6},{:.6},{:.6}]",
        loose.last_obs_types,
        (loose.last_dx[21] as f64).to_degrees(),
        (loose.last_dx[22] as f64).to_degrees(),
        (loose.last_dx[23] as f64).to_degrees()
    );
    if let Some(eskf) = &snapshot.eskf {
        print_eskf_mount_dx_by_type("sum", eskf);
    }
}

fn print_eskf_mount_dx_by_type(prefix: &str, eskf: &EskfState) {
    for (label, idx) in selected_eskf_diag_types() {
        println!(
            "[covhist] eskf_mount_dx_{} type={} count={} sum_deg=[{:.6},{:.6},{:.6}] abs_sum_deg=[{:.6},{:.6},{:.6}]",
            prefix,
            label,
            eskf.update_diag.type_counts[idx],
            (eskf.update_diag.sum_dx_mount_roll[idx] as f64).to_degrees(),
            (eskf.update_diag.sum_dx_mount_pitch[idx] as f64).to_degrees(),
            (eskf.update_diag.sum_dx_mount_yaw[idx] as f64).to_degrees(),
            (eskf.update_diag.sum_abs_dx_mount_roll[idx] as f64).to_degrees(),
            (eskf.update_diag.sum_abs_dx_mount_pitch[idx] as f64).to_degrees(),
            (eskf.update_diag.sum_abs_dx_mount_yaw[idx] as f64).to_degrees(),
        );
    }
}

fn selected_eskf_diag_types() -> [(&'static str, usize); 8] {
    [
        ("pos_xy", 0),
        ("pos_d", 8),
        ("vel_xy", 1),
        ("vel_d", 9),
        ("zero_xy", 2),
        ("zero_d", 10),
        ("nhc_y", DIAG_BODY_VEL_Y),
        ("nhc_z", DIAG_BODY_VEL_Z),
    ]
}

#[allow(clippy::too_many_arguments)]
fn maybe_print_trace(
    replay: &Replay,
    ref_gnss: GenericGnssSample,
    trace_window_abs: Option<[f64; 2]>,
    trace_interval_s: f64,
    trace: &mut TraceState,
    event: &str,
    t_s: f64,
    fusion: &SensorFusion,
    loose: &LooseFilter,
    loose_ready: bool,
    loose_obs_counts: [u32; 9],
) {
    let Some([start_t_s, end_t_s]) = trace_window_abs else {
        return;
    };
    if t_s < start_t_s || t_s > end_t_s || !loose_ready {
        return;
    }
    let Some(eskf) = fusion.eskf() else {
        return;
    };
    let eskf_counts = eskf.update_diag.type_counts;
    if !trace.initialized {
        trace.prev_eskf_counts = eskf_counts;
        trace.prev_loose_counts = loose_obs_counts;
        trace.prev_eskf_sum_dx_mount_roll = eskf.update_diag.sum_dx_mount_roll;
        trace.prev_eskf_sum_dx_mount_pitch = eskf.update_diag.sum_dx_mount_pitch;
        trace.prev_eskf_sum_dx_mount_yaw = eskf.update_diag.sum_dx_mount_yaw;
        trace.initialized = true;
    }

    let eskf_delta = count_delta(&eskf_counts, &trace.prev_eskf_counts);
    let loose_delta = count_delta(&loose_obs_counts, &trace.prev_loose_counts);
    let eskf_mount_roll_delta = f32_delta(
        &eskf.update_diag.sum_dx_mount_roll,
        &trace.prev_eskf_sum_dx_mount_roll,
    );
    let eskf_mount_pitch_delta = f32_delta(
        &eskf.update_diag.sum_dx_mount_pitch,
        &trace.prev_eskf_sum_dx_mount_pitch,
    );
    let eskf_mount_yaw_delta = f32_delta(
        &eskf.update_diag.sum_dx_mount_yaw,
        &trace.prev_eskf_sum_dx_mount_yaw,
    );
    let update_event = eskf_delta.iter().any(|v| *v > 0) || loose_delta.iter().any(|v| *v > 0);
    let periodic = trace
        .last_trace_t_s
        .is_none_or(|last| t_s - last >= trace_interval_s.max(0.0));
    if !update_event && !periodic && event != "gnss" {
        return;
    }

    let loose_snap = LooseSnapshot {
        nominal: *loose.nominal(),
        p: *loose.covariance(),
        pos_ecef: loose.shadow_pos_ecef(),
        last_dx: *loose.last_dx(),
        last_obs_types: loose.last_obs_types().to_vec(),
    };
    let p_loose_as_eskf = transform_loose_cov_to_eskf(&loose_snap, ref_gnss);
    let rel_s = t_s - replay_start_t_s(replay).unwrap_or(0.0);
    let ref_mount = reference_mount_at(&replay.reference_mount, t_s);
    let ref_att = reference_attitude_at(&replay.reference_attitude, t_s);
    let eskf_mount_qerr = ref_mount
        .map(|r| {
            quat_angle_deg(
                as_q64([
                    eskf.nominal.qcs0,
                    eskf.nominal.qcs1,
                    eskf.nominal.qcs2,
                    eskf.nominal.qcs3,
                ]),
                r,
            )
        })
        .unwrap_or(f64::NAN);
    let loose_mount_qerr = ref_mount
        .map(|r| {
            quat_angle_deg(
                as_q64([
                    loose_snap.nominal.qcs0,
                    loose_snap.nominal.qcs1,
                    loose_snap.nominal.qcs2,
                    loose_snap.nominal.qcs3,
                ]),
                r,
            )
        })
        .unwrap_or(f64::NAN);
    let eskf_att_qerr = ref_att
        .map(|r| quat_angle_deg(eskf_att_q(eskf), r))
        .unwrap_or(f64::NAN);
    let loose_att_qerr = ref_att
        .map(|r| quat_angle_deg(loose_att_q_ned(&loose_snap, ref_gnss), r))
        .unwrap_or(f64::NAN);
    let eskf_nhc = eskf_nhc_residual_yz(eskf);
    let loose_nhc = loose_nhc_residual_yz(&loose_snap);

    println!(
        "[covhist-trace] rel_s={:.3} event={} update={} d_eskf(pos_xy,pos_d,vel_xy,vel_d,nhc_y,nhc_z)=[{},{},{},{},{},{}] d_loose(pos,vel,nhc_y,nhc_z)=[{},{},{},{}] qerr_mount_deg=[{:.3},{:.3}] qerr_att_deg=[{:.3},{:.3}] nhc_yz_eskf=[{:.3},{:.3}] nhc_yz_loose=[{:.3},{:.3}]",
        rel_s,
        event,
        update_event,
        eskf_delta[0],
        eskf_delta[8],
        eskf_delta[1],
        eskf_delta[9],
        eskf_delta[DIAG_BODY_VEL_Y],
        eskf_delta[DIAG_BODY_VEL_Z],
        loose_delta[1] + loose_delta[2] + loose_delta[3],
        loose_delta[4] + loose_delta[5] + loose_delta[6],
        loose_delta[7],
        loose_delta[8],
        eskf_mount_qerr,
        loose_mount_qerr,
        eskf_att_qerr,
        loose_att_qerr,
        eskf_nhc[0],
        eskf_nhc[1],
        loose_nhc[0],
        loose_nhc[1],
    );
    println!(
        "[covhist-trace] rel_s={:.3} sig_mount_deg eskf=[{:.3},{:.3},{:.3}] loose=[{:.3},{:.3},{:.3}] sig_att_deg eskf=[{:.3},{:.3},{:.3}] loose=[{:.3},{:.3},{:.3}]",
        rel_s,
        sigma_deg(&eskf.p, 15),
        sigma_deg(&eskf.p, 16),
        sigma_deg(&eskf.p, 17),
        sigma_deg(&p_loose_as_eskf, 15),
        sigma_deg(&p_loose_as_eskf, 16),
        sigma_deg(&p_loose_as_eskf, 17),
        sigma_deg(&eskf.p, 0),
        sigma_deg(&eskf.p, 1),
        sigma_deg(&eskf.p, 2),
        sigma_deg(&p_loose_as_eskf, 0),
        sigma_deg(&p_loose_as_eskf, 1),
        sigma_deg(&p_loose_as_eskf, 2),
    );
    println!(
        "[covhist-trace] rel_s={:.3} d_eskf_mount_dx_deg type_order=[pos_xy,pos_d,vel_xy,vel_d,nhc_y,nhc_z] roll=[{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}] pitch=[{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}] yaw=[{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}]",
        rel_s,
        (eskf_mount_roll_delta[0] as f64).to_degrees(),
        (eskf_mount_roll_delta[8] as f64).to_degrees(),
        (eskf_mount_roll_delta[1] as f64).to_degrees(),
        (eskf_mount_roll_delta[9] as f64).to_degrees(),
        (eskf_mount_roll_delta[DIAG_BODY_VEL_Y] as f64).to_degrees(),
        (eskf_mount_roll_delta[DIAG_BODY_VEL_Z] as f64).to_degrees(),
        (eskf_mount_pitch_delta[0] as f64).to_degrees(),
        (eskf_mount_pitch_delta[8] as f64).to_degrees(),
        (eskf_mount_pitch_delta[1] as f64).to_degrees(),
        (eskf_mount_pitch_delta[9] as f64).to_degrees(),
        (eskf_mount_pitch_delta[DIAG_BODY_VEL_Y] as f64).to_degrees(),
        (eskf_mount_pitch_delta[DIAG_BODY_VEL_Z] as f64).to_degrees(),
        (eskf_mount_yaw_delta[0] as f64).to_degrees(),
        (eskf_mount_yaw_delta[8] as f64).to_degrees(),
        (eskf_mount_yaw_delta[1] as f64).to_degrees(),
        (eskf_mount_yaw_delta[9] as f64).to_degrees(),
        (eskf_mount_yaw_delta[DIAG_BODY_VEL_Y] as f64).to_degrees(),
        (eskf_mount_yaw_delta[DIAG_BODY_VEL_Z] as f64).to_degrees(),
    );
    println!(
        "[covhist-trace] rel_s={:.3} corr bax_yaw=[{:.3},{:.3}] bay_yaw=[{:.3},{:.3}] attx_mountpitch=[{:.3},{:.3}] atty_mountyaw=[{:.3},{:.3}] bgx_mountroll=[{:.3},{:.3}]",
        rel_s,
        corr_from_cov(&eskf.p, 12, 17),
        corr_from_cov(&p_loose_as_eskf, 12, 17),
        corr_from_cov(&eskf.p, 13, 17),
        corr_from_cov(&p_loose_as_eskf, 13, 17),
        corr_from_cov(&eskf.p, 0, 16),
        corr_from_cov(&p_loose_as_eskf, 0, 16),
        corr_from_cov(&eskf.p, 1, 17),
        corr_from_cov(&p_loose_as_eskf, 1, 17),
        corr_from_cov(&eskf.p, 9, 15),
        corr_from_cov(&p_loose_as_eskf, 9, 15),
    );

    trace.last_trace_t_s = Some(t_s);
    trace.prev_eskf_counts = eskf_counts;
    trace.prev_loose_counts = loose_obs_counts;
    trace.prev_eskf_sum_dx_mount_roll = eskf.update_diag.sum_dx_mount_roll;
    trace.prev_eskf_sum_dx_mount_pitch = eskf.update_diag.sum_dx_mount_pitch;
    trace.prev_eskf_sum_dx_mount_yaw = eskf.update_diag.sum_dx_mount_yaw;
}

fn count_delta<const N: usize>(current: &[u32; N], previous: &[u32; N]) -> [u32; N] {
    let mut out = [0u32; N];
    for i in 0..N {
        out[i] = current[i].saturating_sub(previous[i]);
    }
    out
}

fn f32_delta<const N: usize>(current: &[f32; N], previous: &[f32; N]) -> [f32; N] {
    let mut out = [0.0f32; N];
    for i in 0..N {
        out[i] = current[i] - previous[i];
    }
    out
}

fn sigma_deg(p: &[[f32; 18]; 18], idx: usize) -> f64 {
    (p[idx][idx].max(0.0).sqrt() as f64).to_degrees()
}

fn block_rms_abs_diff(
    a: &[[f32; 18]; 18],
    b: &[[f32; 18]; 18],
    rows: &[usize],
    cols: &[usize],
) -> f64 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for &i in rows {
        for &j in cols {
            let d = a[i][j] as f64 - b[i][j] as f64;
            sum += d * d;
            n += 1.0;
        }
    }
    (sum / n).sqrt()
}

fn block_rms_rel_diff(
    a: &[[f32; 18]; 18],
    b: &[[f32; 18]; 18],
    rows: &[usize],
    cols: &[usize],
) -> f64 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for &i in rows {
        for &j in cols {
            let av = a[i][j] as f64;
            let bv = b[i][j] as f64;
            let scale = av.abs().max(bv.abs()).max(1.0e-12);
            let d = (av - bv) / scale;
            sum += d * d;
            n += 1.0;
        }
    }
    (sum / n).sqrt()
}

fn transform_loose_cov_to_eskf(
    loose: &LooseSnapshot,
    ref_gnss: GenericGnssSample,
) -> [[f32; 18]; 18] {
    let mut t = [[0.0f32; LOOSE_ERROR_STATES]; 18];
    let q_es = as_q64([
        loose.nominal.q0,
        loose.nominal.q1,
        loose.nominal.q2,
        loose.nominal.q3,
    ]);
    let c_es = quat_to_rot(q_es);
    let pos_ned = ecef_to_ned(
        loose.pos_ecef,
        lla_to_ecef(ref_gnss.lat_deg, ref_gnss.lon_deg, ref_gnss.height_m),
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
    );
    let (lat, lon, _) = sim::visualizer::math::ned_to_lla_exact(
        pos_ned[0],
        pos_ned[1],
        pos_ned[2],
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
        ref_gnss.height_m,
    );
    let c_ne = ecef_to_ned_matrix(lat, lon);

    for eskf_i in 0..3 {
        for loose_i in 0..3 {
            t[eskf_i][6 + loose_i] = c_es[loose_i][eskf_i] as f32;
        }
    }
    for r in 0..3 {
        for c in 0..3 {
            t[3 + r][3 + c] = c_ne[r][c] as f32;
            t[6 + r][c] = c_ne[r][c] as f32;
        }
    }
    for i in 0..3 {
        t[9 + i][12 + i] = -1.0;
        t[12 + i][9 + i] = -1.0;
        t[15 + i][21 + i] = 1.0;
    }

    let mut out = [[0.0f32; 18]; 18];
    for i in 0..18 {
        for j in 0..18 {
            let mut v = 0.0f32;
            for a in 0..LOOSE_ERROR_STATES {
                for b in 0..LOOSE_ERROR_STATES {
                    v += t[i][a] * loose.p[a][b] * t[j][b];
                }
            }
            out[i][j] = v;
        }
    }
    out
}

fn transform_loose_dx_to_eskf(
    loose: &LooseSnapshot,
    dx: &[f32; LOOSE_ERROR_STATES],
    ref_gnss: GenericGnssSample,
) -> [f32; 18] {
    let q_es = as_q64([
        loose.nominal.q0,
        loose.nominal.q1,
        loose.nominal.q2,
        loose.nominal.q3,
    ]);
    let c_es = quat_to_rot(q_es);
    let pos_ned = ecef_to_ned(
        loose.pos_ecef,
        lla_to_ecef(ref_gnss.lat_deg, ref_gnss.lon_deg, ref_gnss.height_m),
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
    );
    let (lat, lon, _) = sim::visualizer::math::ned_to_lla_exact(
        pos_ned[0],
        pos_ned[1],
        pos_ned[2],
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
        ref_gnss.height_m,
    );
    let c_ne = ecef_to_ned_matrix(lat, lon);

    let mut out = [0.0f32; 18];
    for eskf_i in 0..3 {
        for loose_i in 0..3 {
            out[eskf_i] += c_es[loose_i][eskf_i] as f32 * dx[6 + loose_i];
        }
    }
    for r in 0..3 {
        for c in 0..3 {
            out[3 + r] += c_ne[r][c] as f32 * dx[3 + c];
            out[6 + r] += c_ne[r][c] as f32 * dx[c];
        }
    }
    for i in 0..3 {
        out[9 + i] = -dx[12 + i];
        out[12 + i] = -dx[9 + i];
        out[15 + i] = dx[21 + i];
    }
    out
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
    let mount_var = init.mount_sigma_deg.to_radians().powi(2);
    p[21] = mount_var;
    p[22] = mount_var;
    p[23] = init.mount_yaw_sigma_deg.to_radians().powi(2);
    p
}

fn loose_imu_delta_from_vehicle(
    prev_gyro_radps: [f64; 3],
    prev_accel_mps2: [f64; 3],
    curr_gyro_radps: [f64; 3],
    curr_accel_mps2: [f64; 3],
    dt: f64,
) -> LooseImuDelta {
    LooseImuDelta {
        dax_1: (prev_gyro_radps[0] * dt) as f32,
        day_1: (prev_gyro_radps[1] * dt) as f32,
        daz_1: (prev_gyro_radps[2] * dt) as f32,
        dvx_1: (prev_accel_mps2[0] * dt) as f32,
        dvy_1: (prev_accel_mps2[1] * dt) as f32,
        dvz_1: (prev_accel_mps2[2] * dt) as f32,
        dax_2: (curr_gyro_radps[0] * dt) as f32,
        day_2: (curr_gyro_radps[1] * dt) as f32,
        daz_2: (curr_gyro_radps[2] * dt) as f32,
        dvx_2: (curr_accel_mps2[0] * dt) as f32,
        dvy_2: (curr_accel_mps2[1] * dt) as f32,
        dvz_2: (curr_accel_mps2[2] * dt) as f32,
        dt: dt as f32,
    }
}

fn eskf_att_q(eskf: &EskfState) -> [f64; 4] {
    as_q64([
        eskf.nominal.q0,
        eskf.nominal.q1,
        eskf.nominal.q2,
        eskf.nominal.q3,
    ])
}

fn loose_att_q_ned(loose: &LooseSnapshot, ref_gnss: GenericGnssSample) -> [f64; 4] {
    let pos_ned = ecef_to_ned(
        loose.pos_ecef,
        lla_to_ecef(ref_gnss.lat_deg, ref_gnss.lon_deg, ref_gnss.height_m),
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
    );
    let (lat, lon, _) = sim::visualizer::math::ned_to_lla_exact(
        pos_ned[0],
        pos_ned[1],
        pos_ned[2],
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
        ref_gnss.height_m,
    );
    quat_mul(
        quat_ecef_to_ned(lat, lon),
        as_q64([
            loose.nominal.q0,
            loose.nominal.q1,
            loose.nominal.q2,
            loose.nominal.q3,
        ]),
    )
}

fn reference_mount_at(samples: &[GenericReferenceRpySample], t_s: f64) -> Option<[f64; 4]> {
    nearest_rpy(samples, t_s)
        .map(|s| reference_mount_rpy_to_q_vb([s.roll_deg, s.pitch_deg, s.yaw_deg]))
}

fn reference_mount_seed_q_vb(replay: &Replay, mode: EkfImuSource) -> Option<[f32; 4]> {
    if !mode.uses_ref_mount() {
        return None;
    }
    replay
        .reference_mount
        .iter()
        .rev()
        .find(|sample| {
            sample.roll_deg.is_finite()
                && sample.pitch_deg.is_finite()
                && sample.yaw_deg.is_finite()
        })
        .map(|sample| {
            let q =
                reference_mount_rpy_to_q_vb([sample.roll_deg, sample.pitch_deg, sample.yaw_deg]);
            [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32]
        })
}

fn reference_attitude_at(samples: &[GenericReferenceRpySample], t_s: f64) -> Option<[f64; 4]> {
    nearest_rpy(samples, t_s)
        .map(|s| sim::eval::gnss_ins::quat_from_rpy_alg_deg(s.roll_deg, s.pitch_deg, s.yaw_deg))
}

fn nearest_rpy(
    samples: &[GenericReferenceRpySample],
    t_s: f64,
) -> Option<GenericReferenceRpySample> {
    if samples.is_empty() {
        return None;
    }
    let idx = samples.partition_point(|s| s.t_s < t_s);
    match (idx.checked_sub(1), samples.get(idx).copied()) {
        (Some(prev), Some(next)) => {
            if (t_s - samples[prev].t_s).abs() <= (next.t_s - t_s).abs() {
                Some(samples[prev])
            } else {
                Some(next)
            }
        }
        (Some(prev), None) => Some(samples[prev]),
        (None, Some(next)) => Some(next),
        (None, None) => None,
    }
}

fn eskf_nhc_residual_yz(eskf: &EskfState) -> [f64; 2] {
    let v = vehicle_velocity_from_q(
        as_q64([
            eskf.nominal.q0,
            eskf.nominal.q1,
            eskf.nominal.q2,
            eskf.nominal.q3,
        ]),
        [
            eskf.nominal.vn as f64,
            eskf.nominal.ve as f64,
            eskf.nominal.vd as f64,
        ],
    );
    [-v[1], -v[2]]
}

fn loose_nhc_residual_yz(loose: &LooseSnapshot) -> [f64; 2] {
    let (y, _) = generated_loose::nhc_y(&loose.nominal);
    let (z, _) = generated_loose::nhc_z(&loose.nominal);
    [-(y as f64), -(z as f64)]
}

fn vehicle_velocity_from_q(q: [f64; 4], v_n: [f64; 3]) -> [f64; 3] {
    let c = quat_to_rot(q);
    [
        c[0][0] * v_n[0] + c[1][0] * v_n[1] + c[2][0] * v_n[2],
        c[0][1] * v_n[0] + c[1][1] * v_n[1] + c[2][1] * v_n[2],
        c[0][2] * v_n[0] + c[1][2] * v_n[1] + c[2][2] * v_n[2],
    ]
}

fn corr_from_cov(p: &[[f32; 18]; 18], i: usize, j: usize) -> f64 {
    let denom = ((p[i][i].max(0.0) as f64) * (p[j][j].max(0.0) as f64)).sqrt();
    if denom <= 1.0e-12 {
        0.0
    } else {
        (p[i][j] as f64 / denom).clamp(-1.0, 1.0)
    }
}

fn ned_vector_to_ecef(lat_deg: f64, lon_deg: f64, v_ned: [f64; 3]) -> [f64; 3] {
    let c_ne = ecef_to_ned_matrix(lat_deg, lon_deg);
    [
        c_ne[0][0] * v_ned[0] + c_ne[1][0] * v_ned[1] + c_ne[2][0] * v_ned[2],
        c_ne[0][1] * v_ned[0] + c_ne[1][1] * v_ned[1] + c_ne[2][1] * v_ned[2],
        c_ne[0][2] * v_ned[0] + c_ne[1][2] * v_ned[1] + c_ne[2][2] * v_ned[2],
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

fn quat_ecef_to_ned(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    dcm_to_quat(ecef_to_ned_matrix(lat_deg, lon_deg))
}

fn dcm_to_quat(c: [[f64; 3]; 3]) -> [f64; 4] {
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
    sim::eval::gnss_ins::quat_normalize(q)
}

fn quat_to_rot(q: [f64; 4]) -> [[f64; 3]; 3] {
    let q = sim::eval::gnss_ins::quat_normalize(q);
    let [q0, q1, q2, q3] = q;
    [
        [
            1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2,
        ],
    ]
}

const STATE_NAMES: [&str; 18] = [
    "att_x",
    "att_y",
    "att_z",
    "vel_n",
    "vel_e",
    "vel_d",
    "pos_n",
    "pos_e",
    "pos_d",
    "bgx",
    "bgy",
    "bgz",
    "bax",
    "bay",
    "baz",
    "mount_roll",
    "mount_pitch",
    "mount_yaw",
];
