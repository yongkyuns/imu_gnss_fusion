use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use sensor_fusion::fusion::SensorFusion;
use sim::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, fusion_gnss_sample, fusion_imu_sample,
};
use sim::eval::gnss_ins::{
    as_q64, quat_angle_deg, quat_conj, quat_from_rpy_alg_deg, quat_mul, quat_rotate, wrap_deg180,
};
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::synthetic::gnss_ins_path::{
    GpsNoiseModel, ImuAccuracy, MeasurementNoiseConfig, MotionProfile, PathGenConfig,
    generate_with_noise,
};
use sim::visualizer::math::{ecef_to_ned, lla_to_ecef, quat_rpy_deg};

#[derive(Parser, Debug)]
#[command(name = "synthetic_bad_basin_sweep")]
struct Args {
    #[arg(
        long,
        default_value = "sim/motion_profiles/real_early_bad_basin.scenario"
    )]
    scenario: PathBuf,
    #[arg(long, value_enum, default_value_t = NoiseMode::Mid)]
    noise: NoiseMode,
    #[arg(long, default_value_t = 20260426)]
    seed: u64,
    #[arg(long, default_value_t = 5.0)]
    mount_roll_deg: f64,
    #[arg(long, default_value_t = -5.0)]
    mount_pitch_deg: f64,
    #[arg(long, default_value_t = 5.0)]
    mount_yaw_deg: f64,
    #[arg(long, default_value_t = 100.0)]
    imu_hz: f64,
    #[arg(long, default_value_t = 2.0)]
    gnss_hz: f64,
    #[arg(long, value_enum, default_value_t = SeedMode::InternalAlign)]
    seed_mode: SeedMode,
    #[arg(long, default_value_t = 0.0)]
    seed_error_roll_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    seed_error_pitch_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    seed_error_yaw_deg: f64,
    #[arg(long, value_delimiter = ',', default_value = "0")]
    gnss_time_shifts_ms: Vec<f64>,
    #[arg(long, value_delimiter = ',', default_value = "0")]
    early_vel_bias_n_mps: Vec<f64>,
    #[arg(long, value_delimiter = ',', default_value = "0")]
    early_vel_bias_e_mps: Vec<f64>,
    #[arg(long, value_delimiter = ',', default_value = "0")]
    early_vel_bias_d_mps: Vec<f64>,
    #[arg(long, default_value_t = 100.0)]
    early_fault_start_s: f64,
    #[arg(long, default_value_t = 200.0)]
    early_fault_end_s: f64,
    #[arg(long, default_value_t = 0.1)]
    gnss_pos_r_scale: f64,
    #[arg(long, default_value_t = 1.0)]
    gnss_vel_r_scale: f64,
    #[arg(long, default_value_t = 0.001)]
    r_body_vel: f32,
    #[arg(long, default_value_t = 0.0)]
    gnss_pos_mount_scale: f32,
    #[arg(long, default_value_t = 0.0)]
    gnss_vel_mount_scale: f32,
    #[arg(long, default_value_t = 2.0)]
    yaw_init_sigma_deg: f32,
    #[arg(long, default_value_t = 0.125)]
    gyro_bias_init_sigma_dps: f32,
    #[arg(long, default_value_t = 0.04)]
    r_vehicle_speed: f32,
    #[arg(long, default_value_t = 0.0)]
    r_zero_vel: f32,
    #[arg(long, default_value_t = 0.0)]
    r_stationary_accel: f32,
    #[arg(long, default_value_t = 1.0e-7)]
    mount_align_rw_var: f32,
    #[arg(long, default_value_t = 0.008)]
    mount_update_min_scale: f32,
    #[arg(long, default_value_t = 800.0)]
    mount_update_ramp_time_s: f32,
    #[arg(long, default_value_t = 0.02)]
    mount_update_innovation_gate_mps: f32,
    #[arg(long, default_value_t = 0.0)]
    early_window_start_s: f64,
    #[arg(long, default_value_t = 220.0)]
    early_window_end_s: f64,
    #[arg(long, default_value_t = 60.0)]
    tail_s: f64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum NoiseMode {
    Truth,
    Low,
    Mid,
    High,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SeedMode {
    InternalAlign,
    Truth,
    PerturbedTruth,
}

#[derive(Clone, Copy, Debug)]
struct SweepResult {
    shift_ms: f64,
    bias_n: f64,
    bias_e: f64,
    bias_d: f64,
    mount_ready_t_s: Option<f64>,
    ekf_init_t_s: Option<f64>,
    early_mount_qerr_max_deg: f64,
    early_mount_qerr_mean_deg: f64,
    early_att_qerr_mean_deg: f64,
    early_yaw_err_mean_deg: f64,
    final_mount_qerr_deg: f64,
    final_att_qerr_deg: f64,
    final_yaw_err_deg: f64,
    final_vel_err_mps: f64,
    final_pos_err_m: f64,
    tail_mount_qerr_mean_deg: f64,
    tail_att_qerr_mean_deg: f64,
    body_vel_y_innov_abs: f64,
    body_vel_y_yaw_dx_abs_deg: f64,
}

#[derive(Clone, Copy, Debug)]
struct Snapshot {
    t_s: f64,
    mount_qerr_deg: f64,
    att_qerr_deg: f64,
    yaw_err_deg: f64,
    vel_err_mps: f64,
    pos_err_m: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.gnss_time_shifts_ms.is_empty()
        || args.early_vel_bias_n_mps.is_empty()
        || args.early_vel_bias_e_mps.is_empty()
        || args.early_vel_bias_d_mps.is_empty()
    {
        bail!("sweep lists must not be empty");
    }

    let profile = MotionProfile::from_path(&args.scenario)
        .with_context(|| format!("failed to load scenario {}", args.scenario.display()))?;
    let noise = measurement_noise(args.noise);
    let gps_noise = noise.gps.unwrap_or(GpsNoiseModel {
        pos_std_m: [0.5, 0.5, 0.5],
        vel_std_mps: [0.2, 0.2, 0.2],
    });
    let generated = generate_with_noise(
        &profile,
        PathGenConfig {
            imu_hz: args.imu_hz,
            gnss_hz: args.gnss_hz,
            ..PathGenConfig::default()
        },
        noise,
        args.seed,
    )?;
    let q_truth_mount = quat_from_rpy_alg_deg(
        args.mount_roll_deg,
        args.mount_pitch_deg,
        args.mount_yaw_deg,
    );
    let imu = generated
        .imu
        .iter()
        .map(|s| GenericImuSample {
            t_s: s.t_s,
            gyro_radps: quat_rotate(q_truth_mount, s.gyro_vehicle_radps),
            accel_mps2: quat_rotate(q_truth_mount, s.accel_vehicle_mps2),
        })
        .collect::<Vec<_>>();
    let gnss_base = generated
        .gnss
        .iter()
        .map(|s| GenericGnssSample {
            t_s: s.t_s,
            lat_deg: s.lat_deg,
            lon_deg: s.lon_deg,
            height_m: s.height_m,
            vel_ned_mps: s.vel_ned_mps,
            pos_std_m: [
                gps_noise.pos_std_m[0] * args.gnss_pos_r_scale,
                gps_noise.pos_std_m[1] * args.gnss_pos_r_scale,
                gps_noise.pos_std_m[2] * args.gnss_pos_r_scale,
            ],
            vel_std_mps: [
                gps_noise.vel_std_mps[0] * args.gnss_vel_r_scale,
                gps_noise.vel_std_mps[1] * args.gnss_vel_r_scale,
                gps_noise.vel_std_mps[2] * args.gnss_vel_r_scale,
            ],
            heading_rad: None,
        })
        .collect::<Vec<_>>();

    println!(
        "scenario={} imu_samples={} gnss_samples={} duration_s={:.3} seed_mode={:?}",
        args.scenario.display(),
        imu.len(),
        gnss_base.len(),
        generated
            .reference
            .truth
            .last()
            .map(|s| s.t_s)
            .unwrap_or(0.0),
        args.seed_mode
    );
    println!(
        "shift_ms,bias_n_mps,bias_e_mps,bias_d_mps,mount_ready_t_s,ekf_init_t_s,early_mount_qerr_max_deg,early_mount_qerr_mean_deg,early_att_qerr_mean_deg,early_yaw_err_mean_deg,final_mount_qerr_deg,final_att_qerr_deg,final_yaw_err_deg,final_vel_err_mps,final_pos_err_m,tail_mount_qerr_mean_deg,tail_att_qerr_mean_deg,body_vel_y_innov_abs,body_vel_y_yaw_dx_abs_deg"
    );

    let mut results = Vec::new();
    for shift_ms in &args.gnss_time_shifts_ms {
        for bias_n in &args.early_vel_bias_n_mps {
            for bias_e in &args.early_vel_bias_e_mps {
                for bias_d in &args.early_vel_bias_d_mps {
                    let gnss = perturb_gnss(
                        &gnss_base,
                        *shift_ms,
                        [*bias_n, *bias_e, *bias_d],
                        args.early_fault_start_s,
                        args.early_fault_end_s,
                    );
                    let result = run_case(
                        &args,
                        &generated.reference.truth,
                        &imu,
                        &gnss,
                        *shift_ms,
                        [*bias_n, *bias_e, *bias_d],
                    )?;
                    print_result(&result);
                    results.push(result);
                }
            }
        }
    }

    if let Some(worst_early) = results.iter().max_by(|a, b| {
        a.early_mount_qerr_mean_deg
            .total_cmp(&b.early_mount_qerr_mean_deg)
    }) {
        println!(
            "worst_early_mount: shift_ms={:.1} bias=[{:.3},{:.3},{:.3}] early_mount_mean={:.6} final_mount={:.6}",
            worst_early.shift_ms,
            worst_early.bias_n,
            worst_early.bias_e,
            worst_early.bias_d,
            worst_early.early_mount_qerr_mean_deg,
            worst_early.final_mount_qerr_deg
        );
    }
    if let Some(worst_final) = results
        .iter()
        .max_by(|a, b| a.final_mount_qerr_deg.total_cmp(&b.final_mount_qerr_deg))
    {
        println!(
            "worst_final_mount: shift_ms={:.1} bias=[{:.3},{:.3},{:.3}] early_mount_mean={:.6} final_mount={:.6}",
            worst_final.shift_ms,
            worst_final.bias_n,
            worst_final.bias_e,
            worst_final.bias_d,
            worst_final.early_mount_qerr_mean_deg,
            worst_final.final_mount_qerr_deg
        );
    }

    Ok(())
}

fn measurement_noise(mode: NoiseMode) -> MeasurementNoiseConfig {
    match mode {
        NoiseMode::Truth => MeasurementNoiseConfig::zero(),
        NoiseMode::Low => MeasurementNoiseConfig::accuracy(ImuAccuracy::Low),
        NoiseMode::Mid => MeasurementNoiseConfig::accuracy(ImuAccuracy::Mid),
        NoiseMode::High => MeasurementNoiseConfig::accuracy(ImuAccuracy::High),
    }
}

fn perturb_gnss(
    gnss: &[GenericGnssSample],
    shift_ms: f64,
    early_vel_bias_ned_mps: [f64; 3],
    early_fault_start_s: f64,
    early_fault_end_s: f64,
) -> Vec<GenericGnssSample> {
    let dt_s = shift_ms * 1.0e-3;
    let mut out = Vec::with_capacity(gnss.len());
    for sample in gnss {
        let mut sample = *sample;
        sample.t_s += dt_s;
        if !sample.t_s.is_finite() || sample.t_s < 0.0 {
            continue;
        }
        if (early_fault_start_s..=early_fault_end_s).contains(&sample.t_s) {
            for (v, bias) in sample.vel_ned_mps.iter_mut().zip(early_vel_bias_ned_mps) {
                *v += bias;
            }
        }
        out.push(sample);
    }
    out.sort_by(|a, b| a.t_s.total_cmp(&b.t_s));
    out
}

fn run_case(
    args: &Args,
    truth: &[sim::datasets::gnss_ins_sim::TruthSample],
    imu: &[GenericImuSample],
    gnss: &[GenericGnssSample],
    shift_ms: f64,
    early_vel_bias_ned_mps: [f64; 3],
) -> Result<SweepResult> {
    let q_truth_mount = quat_from_rpy_alg_deg(
        args.mount_roll_deg,
        args.mount_pitch_deg,
        args.mount_yaw_deg,
    );
    let q_seed = match args.seed_mode {
        SeedMode::InternalAlign => None,
        SeedMode::Truth => Some(q_truth_mount),
        SeedMode::PerturbedTruth => {
            let dq = quat_from_rpy_alg_deg(
                args.seed_error_roll_deg,
                args.seed_error_pitch_deg,
                args.seed_error_yaw_deg,
            );
            Some(quat_mul(q_truth_mount, dq))
        }
    };
    let mut fusion = if let Some(q_seed) = q_seed {
        SensorFusion::with_misalignment([
            q_seed[0] as f32,
            q_seed[1] as f32,
            q_seed[2] as f32,
            q_seed[3] as f32,
        ])
    } else {
        SensorFusion::new()
    };
    apply_fusion_config(&mut fusion, args);

    let ref_ecef = lla_to_ecef(truth[0].lat_deg, truth[0].lon_deg, truth[0].height_m);
    let mut snapshots = Vec::<Snapshot>::new();
    let mut mount_ready_t_s = None;
    let mut ekf_init_t_s = None;

    for_each_event(imu, gnss, |event| match event {
        ReplayEvent::Imu(idx, sample) => {
            let update = fusion.process_imu(fusion_imu_sample(*sample));
            capture_update_times(sample.t_s, update, &mut mount_ready_t_s, &mut ekf_init_t_s);
            if let Some(snapshot) = snapshot_state(
                &fusion,
                sample.t_s,
                truth.get(idx),
                ref_ecef,
                truth[0].lat_deg,
                truth[0].lon_deg,
                q_truth_mount,
            ) {
                snapshots.push(snapshot);
            }
        }
        ReplayEvent::Gnss(_, sample) => {
            let update = fusion.process_gnss(fusion_gnss_sample(*sample));
            capture_update_times(sample.t_s, update, &mut mount_ready_t_s, &mut ekf_init_t_s);
        }
    });

    let Some(final_snapshot) = snapshots.last().copied() else {
        bail!("ESKF produced no snapshots");
    };
    let early = snapshots
        .iter()
        .copied()
        .filter(|s| s.t_s >= args.early_window_start_s && s.t_s <= args.early_window_end_s)
        .collect::<Vec<_>>();
    let tail_start = final_snapshot.t_s - args.tail_s;
    let tail = snapshots
        .iter()
        .copied()
        .filter(|s| s.t_s >= tail_start)
        .collect::<Vec<_>>();
    let (body_vel_y_innov_abs, body_vel_y_yaw_dx_abs_deg) = fusion
        .eskf()
        .map(|eskf| {
            (
                eskf.update_diag.sum_abs_innovation[4] as f64,
                eskf.update_diag.sum_abs_dx_mount_yaw[4] as f64 * 180.0 / std::f64::consts::PI,
            )
        })
        .unwrap_or((f64::NAN, f64::NAN));

    Ok(SweepResult {
        shift_ms,
        bias_n: early_vel_bias_ned_mps[0],
        bias_e: early_vel_bias_ned_mps[1],
        bias_d: early_vel_bias_ned_mps[2],
        mount_ready_t_s,
        ekf_init_t_s,
        early_mount_qerr_max_deg: max_or_nan(early.iter().map(|s| s.mount_qerr_deg)),
        early_mount_qerr_mean_deg: mean_or_nan(early.iter().map(|s| s.mount_qerr_deg)),
        early_att_qerr_mean_deg: mean_or_nan(early.iter().map(|s| s.att_qerr_deg)),
        early_yaw_err_mean_deg: mean_abs_or_nan(early.iter().map(|s| s.yaw_err_deg)),
        final_mount_qerr_deg: final_snapshot.mount_qerr_deg,
        final_att_qerr_deg: final_snapshot.att_qerr_deg,
        final_yaw_err_deg: final_snapshot.yaw_err_deg,
        final_vel_err_mps: final_snapshot.vel_err_mps,
        final_pos_err_m: final_snapshot.pos_err_m,
        tail_mount_qerr_mean_deg: mean_or_nan(tail.iter().map(|s| s.mount_qerr_deg)),
        tail_att_qerr_mean_deg: mean_or_nan(tail.iter().map(|s| s.att_qerr_deg)),
        body_vel_y_innov_abs,
        body_vel_y_yaw_dx_abs_deg,
    })
}

fn capture_update_times(
    t_s: f64,
    update: sensor_fusion::fusion::FusionUpdate,
    mount_ready_t_s: &mut Option<f64>,
    ekf_init_t_s: &mut Option<f64>,
) {
    if update.mount_ready_changed && update.mount_ready && mount_ready_t_s.is_none() {
        *mount_ready_t_s = Some(t_s);
    }
    if update.ekf_initialized_now && ekf_init_t_s.is_none() {
        *ekf_init_t_s = Some(t_s);
    }
}

fn snapshot_state(
    fusion: &SensorFusion,
    t_s: f64,
    truth: Option<&sim::datasets::gnss_ins_sim::TruthSample>,
    fallback_ref_ecef: [f64; 3],
    fallback_ref_lat_deg: f64,
    fallback_ref_lon_deg: f64,
    q_truth_mount: [f64; 4],
) -> Option<Snapshot> {
    let eskf = fusion.eskf()?;
    let truth = truth?;
    let q_seed = fusion
        .eskf_mount_q_vb()
        .or_else(|| fusion.mount_q_vb())
        .map(as_q64)
        .unwrap_or(q_truth_mount);
    let q_cs = as_q64([
        eskf.nominal.qcs0,
        eskf.nominal.qcs1,
        eskf.nominal.qcs2,
        eskf.nominal.qcs3,
    ]);
    let q_full_mount = quat_mul(q_seed, quat_conj(q_cs));
    let q_vehicle = quat_mul(
        as_q64([
            eskf.nominal.q0,
            eskf.nominal.q1,
            eskf.nominal.q2,
            eskf.nominal.q3,
        ]),
        quat_conj(q_cs),
    );
    let (ref_ecef, ref_lat_deg, ref_lon_deg) = fusion
        .anchor_lla_debug()
        .map(|anchor| {
            (
                lla_to_ecef(anchor[0] as f64, anchor[1] as f64, anchor[2] as f64),
                anchor[0] as f64,
                anchor[1] as f64,
            )
        })
        .unwrap_or((
            fallback_ref_ecef,
            fallback_ref_lat_deg,
            fallback_ref_lon_deg,
        ));
    let truth_ecef = lla_to_ecef(truth.lat_deg, truth.lon_deg, truth.height_m);
    let truth_pos_ned = ecef_to_ned(truth_ecef, ref_ecef, ref_lat_deg, ref_lon_deg);
    let (_, _, yaw) = quat_rpy_deg(
        q_vehicle[0] as f32,
        q_vehicle[1] as f32,
        q_vehicle[2] as f32,
        q_vehicle[3] as f32,
    );
    let (_, _, truth_yaw) = quat_rpy_deg(
        truth.q_bn[0] as f32,
        truth.q_bn[1] as f32,
        truth.q_bn[2] as f32,
        truth.q_bn[3] as f32,
    );
    Some(Snapshot {
        t_s,
        mount_qerr_deg: quat_angle_deg(q_full_mount, q_truth_mount),
        att_qerr_deg: quat_angle_deg(q_vehicle, truth.q_bn),
        yaw_err_deg: wrap_deg180(yaw - truth_yaw),
        vel_err_mps: norm3([
            eskf.nominal.vn as f64 - truth.vel_ned_mps[0],
            eskf.nominal.ve as f64 - truth.vel_ned_mps[1],
            eskf.nominal.vd as f64 - truth.vel_ned_mps[2],
        ]),
        pos_err_m: norm3([
            eskf.nominal.pn as f64 - truth_pos_ned[0],
            eskf.nominal.pe as f64 - truth_pos_ned[1],
            eskf.nominal.pd as f64 - truth_pos_ned[2],
        ]),
    })
}

fn apply_fusion_config(fusion: &mut SensorFusion, args: &Args) {
    fusion.set_r_body_vel(args.r_body_vel);
    fusion.set_gnss_pos_mount_scale(args.gnss_pos_mount_scale);
    fusion.set_gnss_vel_mount_scale(args.gnss_vel_mount_scale);
    fusion.set_yaw_init_sigma_rad(args.yaw_init_sigma_deg.to_radians());
    fusion.set_gyro_bias_init_sigma_radps(args.gyro_bias_init_sigma_dps.to_radians());
    fusion.set_r_vehicle_speed(args.r_vehicle_speed);
    fusion.set_r_zero_vel(args.r_zero_vel);
    fusion.set_r_stationary_accel(args.r_stationary_accel);
    fusion.set_mount_align_rw_var(args.mount_align_rw_var);
    fusion.set_mount_update_min_scale(args.mount_update_min_scale);
    fusion.set_mount_update_ramp_time_s(args.mount_update_ramp_time_s);
    fusion.set_mount_update_innovation_gate_mps(args.mount_update_innovation_gate_mps);
}

fn print_result(result: &SweepResult) {
    println!(
        "{:.1},{:.6},{:.6},{:.6},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        result.shift_ms,
        result.bias_n,
        result.bias_e,
        result.bias_d,
        fmt_opt(result.mount_ready_t_s),
        fmt_opt(result.ekf_init_t_s),
        result.early_mount_qerr_max_deg,
        result.early_mount_qerr_mean_deg,
        result.early_att_qerr_mean_deg,
        result.early_yaw_err_mean_deg,
        result.final_mount_qerr_deg,
        result.final_att_qerr_deg,
        result.final_yaw_err_deg,
        result.final_vel_err_mps,
        result.final_pos_err_m,
        result.tail_mount_qerr_mean_deg,
        result.tail_att_qerr_mean_deg,
        result.body_vel_y_innov_abs,
        result.body_vel_y_yaw_dx_abs_deg,
    );
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "nan".to_string())
}

fn mean_or_nan(values: impl Iterator<Item = f64>) -> f64 {
    let mut n = 0usize;
    let mut sum = 0.0;
    for value in values {
        if value.is_finite() {
            n += 1;
            sum += value;
        }
    }
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

fn mean_abs_or_nan(values: impl Iterator<Item = f64>) -> f64 {
    mean_or_nan(values.map(f64::abs))
}

fn max_or_nan(values: impl Iterator<Item = f64>) -> f64 {
    values
        .filter(|v| v.is_finite())
        .reduce(f64::max)
        .unwrap_or(f64::NAN)
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
