use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use sensor_fusion::fusion::SensorFusion;
use sim::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, fusion_gnss_sample as to_fusion_gnss,
    fusion_imu_sample as to_fusion_imu, write_samples as write_generic_samples,
};
use sim::datasets::gnss_ins_sim::{
    GnssSample as DatasetGnssSample, load_gnss_samples as load_dataset_gnss_samples,
    load_imu_samples as load_dataset_imu_samples, load_truth_samples as load_dataset_truth_samples,
};
use sim::eval::gnss_ins::{
    SignalSource, as_q64, course_rate_deg, horiz_speed, quat_angle_deg, quat_axis_angle_deg,
    quat_conj, quat_from_rpy_alg_deg, quat_mul, quat_rotate,
};
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::eval::state_summary::{
    SummaryMode, print_summary_table, summarize_trace_pair, write_summary_csv,
};
use sim::visualizer::math::{ecef_to_ned, lla_to_ecef, quat_rpy_deg};
use sim::visualizer::model::Trace;
use sim::visualizer::pipeline::align_replay::{
    frd_mount_quat_to_esf_alg_flu_quat, quat_rpy_alg_deg,
};

#[derive(Parser, Debug)]
#[command(name = "eskf_eval_gnss_ins_sim")]
struct Args {
    #[arg(value_name = "DATA_DIR")]
    data_dir: PathBuf,

    #[arg(long, value_enum, default_value_t = SignalSource::Meas)]
    signal_source: SignalSource,

    #[arg(long, default_value_t = 0)]
    data_key: usize,

    #[arg(long, value_enum, default_value_t = SeedSource::InternalAlign)]
    seed_source: SeedSource,

    #[arg(long)]
    r_body_vel: Option<f32>,
    #[arg(long)]
    gnss_pos_mount_scale: Option<f32>,
    #[arg(long)]
    gnss_vel_mount_scale: Option<f32>,
    #[arg(long)]
    gyro_bias_init_sigma_dps: Option<f32>,

    #[arg(long, default_value_t = 0.0)]
    mount_roll_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    mount_pitch_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    mount_yaw_deg: f64,

    #[arg(long, default_value_t = 0.5)]
    gnss_pos_std_m: f32,
    #[arg(long, default_value_t = 0.2)]
    gnss_vel_std_mps: f32,
    #[arg(long, default_value_t = 0.0)]
    gps_vd_bias_mps: f64,
    #[arg(long, default_value_t = 0.0)]
    gps_vd_bias_duration_s: f64,

    #[arg(long)]
    residual_csv: Option<PathBuf>,
    #[arg(long)]
    summary_csv: Option<PathBuf>,
    #[arg(long)]
    generic_out_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SeedSource {
    InternalAlign,
    ExternalTruth,
}

type GnssSample = DatasetGnssSample;

#[derive(Clone, Copy, Debug)]
struct ResidualSample {
    t_s: f64,
    truth_q0: f64,
    truth_q1: f64,
    truth_q2: f64,
    truth_q3: f64,
    seed_q0: f64,
    seed_q1: f64,
    seed_q2: f64,
    seed_q3: f64,
    align_q0: f64,
    align_q1: f64,
    align_q2: f64,
    align_q3: f64,
    qcs_q0: f64,
    qcs_q1: f64,
    qcs_q2: f64,
    qcs_q3: f64,
    full_a_q0: f64,
    full_a_q1: f64,
    full_a_q2: f64,
    full_a_q3: f64,
    full_b_q0: f64,
    full_b_q1: f64,
    full_b_q2: f64,
    full_b_q3: f64,
    seed_err_deg: f64,
    align_err_deg: f64,
    full_a_err_deg: f64,
    full_b_err_deg: f64,
    qcs_angle_deg: f64,
    speed_mps: f64,
    course_rate_dps: f64,
}

#[derive(Clone, Copy, Debug)]
struct StateSample {
    t_s: f64,
    pos_n_m: f64,
    pos_e_m: f64,
    pos_d_m: f64,
    truth_pos_n_m: f64,
    truth_pos_e_m: f64,
    truth_pos_d_m: f64,
    vel_n_mps: f64,
    vel_e_mps: f64,
    vel_d_mps: f64,
    truth_vel_n_mps: f64,
    truth_vel_e_mps: f64,
    truth_vel_d_mps: f64,
    att_roll_deg: f64,
    att_pitch_deg: f64,
    att_yaw_deg: f64,
    truth_att_roll_deg: f64,
    truth_att_pitch_deg: f64,
    truth_att_yaw_deg: f64,
    att_quat_err_deg: f64,
    att_fwd_err_deg: f64,
    att_down_err_deg: f64,
    mount_roll_deg: f64,
    mount_pitch_deg: f64,
    mount_yaw_deg: f64,
    truth_mount_roll_deg: f64,
    truth_mount_pitch_deg: f64,
    truth_mount_yaw_deg: f64,
    mount_quat_err_deg: f64,
    mount_fwd_err_deg: f64,
    mount_down_err_deg: f64,
    full_mount_err_deg: f64,
    gyro_bias_x_dps: f64,
    gyro_bias_y_dps: f64,
    gyro_bias_z_dps: f64,
    accel_bias_x_mps2: f64,
    accel_bias_y_mps2: f64,
    accel_bias_z_mps2: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let imu = load_dataset_imu_samples(
        &args.data_dir,
        args.signal_source.use_ref_signals(),
        args.data_key,
    )?;
    let truth = load_dataset_truth_samples(&args.data_dir)?;
    let gnss = load_dataset_gnss_samples(
        &args.data_dir,
        args.signal_source.use_ref_signals(),
        args.data_key,
    )?;
    if imu.is_empty() || gnss.is_empty() || truth.is_empty() {
        bail!("need IMU, GNSS, and truth samples");
    }
    if imu.len() != truth.len() {
        bail!("IMU and truth files have inconsistent lengths");
    }

    let q_truth = quat_from_rpy_alg_deg(
        args.mount_roll_deg,
        args.mount_pitch_deg,
        args.mount_yaw_deg,
    );
    let generic_imu = build_generic_imu_samples(&imu, q_truth);
    let generic_gnss =
        build_generic_gnss_samples(&gnss, args.gnss_pos_std_m, args.gnss_vel_std_mps);
    if let Some(dir) = &args.generic_out_dir {
        write_generic_samples(dir, &generic_imu, &generic_gnss)?;
        println!("generic_replay_dir={}", dir.display());
    }
    let mut fusion = match args.seed_source {
        SeedSource::InternalAlign => SensorFusion::new(),
        SeedSource::ExternalTruth => SensorFusion::with_misalignment([
            q_truth[0] as f32,
            q_truth[1] as f32,
            q_truth[2] as f32,
            q_truth[3] as f32,
        ]),
    };
    if let Some(r_body_vel) = args.r_body_vel {
        fusion.set_r_body_vel(r_body_vel);
    }
    if let Some(gnss_pos_mount_scale) = args.gnss_pos_mount_scale {
        fusion.set_gnss_pos_mount_scale(gnss_pos_mount_scale);
    }
    if let Some(gnss_vel_mount_scale) = args.gnss_vel_mount_scale {
        fusion.set_gnss_vel_mount_scale(gnss_vel_mount_scale);
    }
    if let Some(gyro_bias_init_sigma_dps) = args.gyro_bias_init_sigma_dps {
        fusion.set_gyro_bias_init_sigma_radps(gyro_bias_init_sigma_dps.to_radians());
    }

    let mut residuals = Vec::<ResidualSample>::new();
    let mut state_samples = Vec::<StateSample>::new();
    let mut mount_ready_s = None::<f64>;
    let mut ekf_init_s = None::<f64>;
    let mut seed_q_vb = None::<[f64; 4]>;
    let mut prev_gnss = None::<GnssSample>;
    let ref_ecef = lla_to_ecef(truth[0].lat_deg, truth[0].lon_deg, truth[0].height_m);

    for_each_event(&generic_imu, &generic_gnss, |event| match event {
        ReplayEvent::Imu(imu_idx, sample) => {
            let s = imu[imu_idx];
            let truth_s = truth[imu_idx];
            if (s.t_s - truth_s.t_s).abs() > 1.0e-6 {
                return;
            }
            let _ = fusion.process_imu(to_fusion_imu(*sample));
            if ekf_init_s.is_some()
                && let Some(eskf) = fusion.eskf()
            {
                let truth_ecef = lla_to_ecef(truth_s.lat_deg, truth_s.lon_deg, truth_s.height_m);
                let truth_pos_ned =
                    ecef_to_ned(truth_ecef, ref_ecef, truth[0].lat_deg, truth[0].lon_deg);
                let (att_roll_deg, att_pitch_deg, att_yaw_deg) = quat_rpy_deg(
                    eskf.nominal.q0,
                    eskf.nominal.q1,
                    eskf.nominal.q2,
                    eskf.nominal.q3,
                );
                let (truth_att_roll_deg, truth_att_pitch_deg, truth_att_yaw_deg) = quat_rpy_deg(
                    truth_s.q_bn[0] as f32,
                    truth_s.q_bn[1] as f32,
                    truth_s.q_bn[2] as f32,
                    truth_s.q_bn[3] as f32,
                );
                let q_predict_seed = fusion
                    .eskf_mount_q_vb()
                    .or_else(|| fusion.mount_q_vb())
                    .map(as_q64)
                    .or(seed_q_vb)
                    .unwrap_or(q_truth);
                let q_est_att = as_q64([
                    eskf.nominal.q0,
                    eskf.nominal.q1,
                    eskf.nominal.q2,
                    eskf.nominal.q3,
                ]);
                let q_cs = as_q64([
                    eskf.nominal.qcs0,
                    eskf.nominal.qcs1,
                    eskf.nominal.qcs2,
                    eskf.nominal.qcs3,
                ]);
                let q_full_mount = quat_mul(q_predict_seed, quat_conj(q_cs));
                let q_full_mount_flu = frd_mount_quat_to_esf_alg_flu_quat(q_full_mount);
                let (mount_roll_deg, mount_pitch_deg, mount_yaw_deg) = quat_rpy_alg_deg(
                    q_full_mount_flu[0],
                    q_full_mount_flu[1],
                    q_full_mount_flu[2],
                    q_full_mount_flu[3],
                );
                state_samples.push(StateSample {
                    t_s: sample.t_s,
                    pos_n_m: eskf.nominal.pn as f64,
                    pos_e_m: eskf.nominal.pe as f64,
                    pos_d_m: eskf.nominal.pd as f64,
                    truth_pos_n_m: truth_pos_ned[0],
                    truth_pos_e_m: truth_pos_ned[1],
                    truth_pos_d_m: truth_pos_ned[2],
                    vel_n_mps: eskf.nominal.vn as f64,
                    vel_e_mps: eskf.nominal.ve as f64,
                    vel_d_mps: eskf.nominal.vd as f64,
                    truth_vel_n_mps: truth_s.vel_ned_mps[0],
                    truth_vel_e_mps: truth_s.vel_ned_mps[1],
                    truth_vel_d_mps: truth_s.vel_ned_mps[2],
                    att_roll_deg,
                    att_pitch_deg,
                    att_yaw_deg,
                    truth_att_roll_deg,
                    truth_att_pitch_deg,
                    truth_att_yaw_deg,
                    att_quat_err_deg: quat_angle_deg(q_est_att, truth_s.q_bn),
                    att_fwd_err_deg: quat_axis_angle_deg(q_est_att, truth_s.q_bn, [1.0, 0.0, 0.0]),
                    att_down_err_deg: quat_axis_angle_deg(q_est_att, truth_s.q_bn, [0.0, 0.0, 1.0]),
                    mount_roll_deg,
                    mount_pitch_deg,
                    mount_yaw_deg,
                    truth_mount_roll_deg: args.mount_roll_deg,
                    truth_mount_pitch_deg: args.mount_pitch_deg,
                    truth_mount_yaw_deg: args.mount_yaw_deg,
                    mount_quat_err_deg: quat_angle_deg(q_full_mount, q_truth),
                    mount_fwd_err_deg: quat_axis_angle_deg(q_full_mount, q_truth, [1.0, 0.0, 0.0]),
                    mount_down_err_deg: quat_axis_angle_deg(q_full_mount, q_truth, [0.0, 0.0, 1.0]),
                    full_mount_err_deg: quat_angle_deg(q_full_mount, q_truth),
                    gyro_bias_x_dps: eskf.nominal.bgx as f64 * 180.0 / std::f64::consts::PI,
                    gyro_bias_y_dps: eskf.nominal.bgy as f64 * 180.0 / std::f64::consts::PI,
                    gyro_bias_z_dps: eskf.nominal.bgz as f64 * 180.0 / std::f64::consts::PI,
                    accel_bias_x_mps2: eskf.nominal.bax as f64,
                    accel_bias_y_mps2: eskf.nominal.bay as f64,
                    accel_bias_z_mps2: eskf.nominal.baz as f64,
                });
            }
        }
        ReplayEvent::Gnss(gnss_idx, sample) => {
            let s = gnss[gnss_idx];
            let rel_gnss_s = ekf_init_s.map(|t0| s.t_s - t0).unwrap_or(0.0);
            let mut vel_ned_mps = sample.vel_ned_mps;
            if args.gps_vd_bias_duration_s > 0.0
                && (0.0..=args.gps_vd_bias_duration_s).contains(&rel_gnss_s)
            {
                vel_ned_mps[2] += args.gps_vd_bias_mps;
            }
            let update = fusion.process_gnss(to_fusion_gnss(GenericGnssSample {
                vel_ned_mps,
                ..*sample
            }));
            if update.mount_ready_changed && update.mount_ready && mount_ready_s.is_none() {
                mount_ready_s = Some(sample.t_s);
            }
            if update.ekf_initialized_now && ekf_init_s.is_none() {
                ekf_init_s = Some(sample.t_s);
                seed_q_vb = fusion
                    .eskf_mount_q_vb()
                    .or_else(|| fusion.mount_q_vb())
                    .map(as_q64);
            }
            let align_q_opt = fusion
                .align()
                .map(|align| as_q64(align.q_vb))
                .or_else(|| fusion.mount_q_vb().map(as_q64));
            if let (Some(seed_q), Some(eskf), Some(q_align)) =
                (seed_q_vb, fusion.eskf(), align_q_opt)
            {
                let q_cs = as_q64([
                    eskf.nominal.qcs0,
                    eskf.nominal.qcs1,
                    eskf.nominal.qcs2,
                    eskf.nominal.qcs3,
                ]);
                // ESKF pre-rotates IMU samples by the frozen seed mount before
                // prediction. q_cs maps that seeded frame back to vehicle, so the
                // physical vehicle-to-body mount is seed * inv(q_cs).
                let q_full_a_legacy = quat_mul(q_cs, seed_q);
                let q_full = quat_mul(seed_q, quat_conj(q_cs));
                let speed_mps = horiz_speed(s.vel_ned_mps);
                let course_rate_dps = prev_gnss
                    .map(|prev| course_rate_deg(prev, s))
                    .unwrap_or(0.0);
                residuals.push(ResidualSample {
                    t_s: sample.t_s,
                    truth_q0: q_truth[0],
                    truth_q1: q_truth[1],
                    truth_q2: q_truth[2],
                    truth_q3: q_truth[3],
                    seed_q0: seed_q[0],
                    seed_q1: seed_q[1],
                    seed_q2: seed_q[2],
                    seed_q3: seed_q[3],
                    align_q0: q_align[0],
                    align_q1: q_align[1],
                    align_q2: q_align[2],
                    align_q3: q_align[3],
                    qcs_q0: q_cs[0],
                    qcs_q1: q_cs[1],
                    qcs_q2: q_cs[2],
                    qcs_q3: q_cs[3],
                    full_a_q0: q_full_a_legacy[0],
                    full_a_q1: q_full_a_legacy[1],
                    full_a_q2: q_full_a_legacy[2],
                    full_a_q3: q_full_a_legacy[3],
                    full_b_q0: q_full[0],
                    full_b_q1: q_full[1],
                    full_b_q2: q_full[2],
                    full_b_q3: q_full[3],
                    seed_err_deg: quat_angle_deg(seed_q, q_truth),
                    align_err_deg: quat_angle_deg(q_align, q_truth),
                    full_a_err_deg: quat_angle_deg(q_full_a_legacy, q_truth),
                    full_b_err_deg: quat_angle_deg(q_full, q_truth),
                    qcs_angle_deg: quat_angle_deg(q_cs, [1.0, 0.0, 0.0, 0.0]),
                    speed_mps,
                    course_rate_dps,
                });
            }
            prev_gnss = Some(s);
        }
    });

    if residuals.is_empty() || state_samples.is_empty() {
        bail!("no ESKF samples produced");
    }

    let last = residuals.last().unwrap();
    let tail = tail_window(&residuals, 60.0);
    println!("input={}", args.data_dir.display());
    println!(
        "source={:?} seed={:?} r_body_vel={} gnss_pos_mount_scale={} gnss_vel_mount_scale={} gyro_bias_init_sigma_dps={} key={} truth_mount_deg=({:.3},{:.3},{:.3})",
        args.signal_source,
        args.seed_source,
        args.r_body_vel
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "default".to_string()),
        args.gnss_pos_mount_scale
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "default".to_string()),
        args.gnss_vel_mount_scale
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "default".to_string()),
        args.gyro_bias_init_sigma_dps
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "default".to_string()),
        args.data_key,
        args.mount_roll_deg,
        args.mount_pitch_deg,
        args.mount_yaw_deg
    );
    println!(
        "gps_vd_bias_mps={:.3} gps_vd_bias_duration_s={:.3} gnss_vel_std_mps={:.3}",
        args.gps_vd_bias_mps, args.gps_vd_bias_duration_s, args.gnss_vel_std_mps
    );
    println!(
        "mount_ready={} ekf_init={}",
        mount_ready_s
            .map(|t| format!("{:.3}s", t))
            .unwrap_or_else(|| "none".to_string()),
        ekf_init_s
            .map(|t| format!("{:.3}s", t))
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "final_quat_err_deg seed={:.3} align={:.3} full(seed*inv_qcs)={:.3} legacy(qcs*seed)={:.3}",
        last.seed_err_deg, last.align_err_deg, last.full_b_err_deg, last.full_a_err_deg
    );
    println!(
        "tail60_mean_quat_err_deg seed={:.3} align={:.3} full(seed*inv_qcs)={:.3} legacy(qcs*seed)={:.3}",
        mean_of(tail.iter().map(|s| s.seed_err_deg)),
        mean_of(tail.iter().map(|s| s.align_err_deg)),
        mean_of(tail.iter().map(|s| s.full_b_err_deg)),
        mean_of(tail.iter().map(|s| s.full_a_err_deg)),
    );
    println!(
        "tail60_max_quat_err_deg seed={:.3} align={:.3} full(seed*inv_qcs)={:.3} legacy(qcs*seed)={:.3}",
        max_of(tail.iter().map(|s| s.seed_err_deg)),
        max_of(tail.iter().map(|s| s.align_err_deg)),
        max_of(tail.iter().map(|s| s.full_b_err_deg)),
        max_of(tail.iter().map(|s| s.full_a_err_deg)),
    );

    let summaries = build_state_summaries(&residuals, &state_samples);
    print_summary_table(&summaries);

    if let Some(path) = &args.residual_csv {
        write_residual_csv(path, &residuals)?;
        println!("wrote residual CSV: {}", path.display());
    }
    if let Some(path) = &args.summary_csv {
        write_summary_csv(path, &summaries)?;
        println!("state_summary_csv={}", path.display());
    }

    Ok(())
}

fn build_generic_imu_samples(
    imu_samples: &[sim::datasets::gnss_ins_sim::ImuSample],
    q_truth: [f64; 4],
) -> Vec<GenericImuSample> {
    imu_samples
        .iter()
        .map(|s| GenericImuSample {
            t_s: s.t_s,
            gyro_radps: quat_rotate(q_truth, s.gyro_vehicle_radps),
            accel_mps2: quat_rotate(q_truth, s.accel_vehicle_mps2),
        })
        .collect()
}

fn build_generic_gnss_samples(
    gnss_samples: &[GnssSample],
    gnss_pos_std_m: f32,
    gnss_vel_std_mps: f32,
) -> Vec<GenericGnssSample> {
    gnss_samples
        .iter()
        .map(|s| GenericGnssSample {
            t_s: s.t_s,
            lat_deg: s.lat_deg,
            lon_deg: s.lon_deg,
            height_m: s.height_m,
            vel_ned_mps: s.vel_ned_mps,
            pos_std_m: [
                gnss_pos_std_m as f64,
                gnss_pos_std_m as f64,
                gnss_pos_std_m as f64,
            ],
            vel_std_mps: [
                gnss_vel_std_mps as f64,
                gnss_vel_std_mps as f64,
                gnss_vel_std_mps as f64,
            ],
            heading_rad: None,
        })
        .collect()
}

fn tail_window(samples: &[ResidualSample], window_s: f64) -> &[ResidualSample] {
    let end_t = samples.last().map(|s| s.t_s).unwrap_or(0.0);
    let start_t = end_t - window_s;
    let start_idx = samples.iter().position(|s| s.t_s >= start_t).unwrap_or(0);
    &samples[start_idx..]
}

fn mean_of<I>(iter: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut n = 0usize;
    for v in iter {
        sum += v;
        n += 1;
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn max_of<I>(iter: I) -> f64
where
    I: Iterator<Item = f64>,
{
    iter.fold(0.0, f64::max)
}

fn write_residual_csv(path: &Path, samples: &[ResidualSample]) -> Result<()> {
    let file =
        fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "t_s,truth_q0,truth_q1,truth_q2,truth_q3,seed_q0,seed_q1,seed_q2,seed_q3,align_q0,align_q1,align_q2,align_q3,qcs_q0,qcs_q1,qcs_q2,qcs_q3,legacy_full_a_q0,legacy_full_a_q1,legacy_full_a_q2,legacy_full_a_q3,full_seed_inv_qcs_q0,full_seed_inv_qcs_q1,full_seed_inv_qcs_q2,full_seed_inv_qcs_q3,seed_err_deg,align_err_deg,legacy_full_a_err_deg,full_seed_inv_qcs_err_deg,qcs_angle_deg,speed_mps,course_rate_dps"
    )?;
    for s in samples {
        writeln!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            s.t_s,
            s.truth_q0,
            s.truth_q1,
            s.truth_q2,
            s.truth_q3,
            s.seed_q0,
            s.seed_q1,
            s.seed_q2,
            s.seed_q3,
            s.align_q0,
            s.align_q1,
            s.align_q2,
            s.align_q3,
            s.qcs_q0,
            s.qcs_q1,
            s.qcs_q2,
            s.qcs_q3,
            s.full_a_q0,
            s.full_a_q1,
            s.full_a_q2,
            s.full_a_q3,
            s.full_b_q0,
            s.full_b_q1,
            s.full_b_q2,
            s.full_b_q3,
            s.seed_err_deg,
            s.align_err_deg,
            s.full_a_err_deg,
            s.full_b_err_deg,
            s.qcs_angle_deg,
            s.speed_mps,
            s.course_rate_dps,
        )?;
    }
    Ok(())
}

fn build_state_summaries(
    residuals: &[ResidualSample],
    states: &[StateSample],
) -> Vec<sim::eval::state_summary::StateSummary> {
    let trace = |name: &str, values: Vec<[f64; 2]>| Trace {
        name: name.to_string(),
        points: values,
    };
    let zero_trace = |name: &str, times: &[f64]| Trace {
        name: name.to_string(),
        points: times.iter().map(|t_s| [*t_s, 0.0]).collect(),
    };

    struct Spec<'a> {
        state: &'a str,
        trace: Trace,
        reference: Option<Trace>,
        mode: SummaryMode,
        settle_threshold: Option<f64>,
    }

    let state_times: Vec<f64> = states.iter().map(|sample| sample.t_s).collect();
    let residual_times: Vec<f64> = residuals.iter().map(|sample| sample.t_s).collect();
    let specs = vec![
        Spec {
            state: "pos_n_m",
            trace: trace(
                "pos_n [m]",
                states.iter().map(|s| [s.t_s, s.pos_n_m]).collect(),
            ),
            reference: Some(trace(
                "truth pos_n [m]",
                states.iter().map(|s| [s.t_s, s.truth_pos_n_m]).collect(),
            )),
            mode: SummaryMode::Linear,
            settle_threshold: Some(2.0),
        },
        Spec {
            state: "pos_e_m",
            trace: trace(
                "pos_e [m]",
                states.iter().map(|s| [s.t_s, s.pos_e_m]).collect(),
            ),
            reference: Some(trace(
                "truth pos_e [m]",
                states.iter().map(|s| [s.t_s, s.truth_pos_e_m]).collect(),
            )),
            mode: SummaryMode::Linear,
            settle_threshold: Some(2.0),
        },
        Spec {
            state: "pos_d_m",
            trace: trace(
                "pos_d [m]",
                states.iter().map(|s| [s.t_s, s.pos_d_m]).collect(),
            ),
            reference: Some(trace(
                "truth pos_d [m]",
                states.iter().map(|s| [s.t_s, s.truth_pos_d_m]).collect(),
            )),
            mode: SummaryMode::Linear,
            settle_threshold: Some(2.0),
        },
        Spec {
            state: "vel_n_mps",
            trace: trace(
                "vel_n [m/s]",
                states.iter().map(|s| [s.t_s, s.vel_n_mps]).collect(),
            ),
            reference: Some(trace(
                "truth vel_n [m/s]",
                states.iter().map(|s| [s.t_s, s.truth_vel_n_mps]).collect(),
            )),
            mode: SummaryMode::Linear,
            settle_threshold: Some(0.5),
        },
        Spec {
            state: "vel_e_mps",
            trace: trace(
                "vel_e [m/s]",
                states.iter().map(|s| [s.t_s, s.vel_e_mps]).collect(),
            ),
            reference: Some(trace(
                "truth vel_e [m/s]",
                states.iter().map(|s| [s.t_s, s.truth_vel_e_mps]).collect(),
            )),
            mode: SummaryMode::Linear,
            settle_threshold: Some(0.5),
        },
        Spec {
            state: "vel_d_mps",
            trace: trace(
                "vel_d [m/s]",
                states.iter().map(|s| [s.t_s, s.vel_d_mps]).collect(),
            ),
            reference: Some(trace(
                "truth vel_d [m/s]",
                states.iter().map(|s| [s.t_s, s.truth_vel_d_mps]).collect(),
            )),
            mode: SummaryMode::Linear,
            settle_threshold: Some(0.5),
        },
        Spec {
            state: "att_roll_deg",
            trace: trace(
                "att_roll [deg]",
                states.iter().map(|s| [s.t_s, s.att_roll_deg]).collect(),
            ),
            reference: Some(trace(
                "truth att_roll [deg]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.truth_att_roll_deg])
                    .collect(),
            )),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(3.0),
        },
        Spec {
            state: "att_pitch_deg",
            trace: trace(
                "att_pitch [deg]",
                states.iter().map(|s| [s.t_s, s.att_pitch_deg]).collect(),
            ),
            reference: Some(trace(
                "truth att_pitch [deg]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.truth_att_pitch_deg])
                    .collect(),
            )),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(3.0),
        },
        Spec {
            state: "att_yaw_deg",
            trace: trace(
                "att_yaw [deg]",
                states.iter().map(|s| [s.t_s, s.att_yaw_deg]).collect(),
            ),
            reference: Some(trace(
                "truth att_yaw [deg]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.truth_att_yaw_deg])
                    .collect(),
            )),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "att_quat_err_deg",
            trace: trace(
                "att quat err [deg]",
                states.iter().map(|s| [s.t_s, s.att_quat_err_deg]).collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "att_fwd_err_deg",
            trace: trace(
                "att fwd err [deg]",
                states.iter().map(|s| [s.t_s, s.att_fwd_err_deg]).collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "att_down_err_deg",
            trace: trace(
                "att down err [deg]",
                states.iter().map(|s| [s.t_s, s.att_down_err_deg]).collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "mount_roll_deg",
            trace: trace(
                "mount_roll [deg]",
                states.iter().map(|s| [s.t_s, s.mount_roll_deg]).collect(),
            ),
            reference: Some(trace(
                "truth mount_roll [deg]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.truth_mount_roll_deg])
                    .collect(),
            )),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "mount_pitch_deg",
            trace: trace(
                "mount_pitch [deg]",
                states.iter().map(|s| [s.t_s, s.mount_pitch_deg]).collect(),
            ),
            reference: Some(trace(
                "truth mount_pitch [deg]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.truth_mount_pitch_deg])
                    .collect(),
            )),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "mount_yaw_deg",
            trace: trace(
                "mount_yaw [deg]",
                states.iter().map(|s| [s.t_s, s.mount_yaw_deg]).collect(),
            ),
            reference: Some(trace(
                "truth mount_yaw [deg]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.truth_mount_yaw_deg])
                    .collect(),
            )),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "mount_quat_err_deg",
            trace: trace(
                "mount quat err [deg]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.mount_quat_err_deg])
                    .collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "mount_fwd_err_deg",
            trace: trace(
                "mount fwd err [deg]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.mount_fwd_err_deg])
                    .collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "mount_down_err_deg",
            trace: trace(
                "mount down err [deg]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.mount_down_err_deg])
                    .collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "full_mount_err_deg",
            trace: trace(
                "full mount err [deg]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.full_mount_err_deg])
                    .collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "gyro_bias_x_dps",
            trace: trace(
                "gyro bias x [deg/s]",
                states.iter().map(|s| [s.t_s, s.gyro_bias_x_dps]).collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: None,
        },
        Spec {
            state: "gyro_bias_y_dps",
            trace: trace(
                "gyro bias y [deg/s]",
                states.iter().map(|s| [s.t_s, s.gyro_bias_y_dps]).collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: None,
        },
        Spec {
            state: "gyro_bias_z_dps",
            trace: trace(
                "gyro bias z [deg/s]",
                states.iter().map(|s| [s.t_s, s.gyro_bias_z_dps]).collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: None,
        },
        Spec {
            state: "accel_bias_x_mps2",
            trace: trace(
                "accel bias x [m/s^2]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.accel_bias_x_mps2])
                    .collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: None,
        },
        Spec {
            state: "accel_bias_y_mps2",
            trace: trace(
                "accel bias y [m/s^2]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.accel_bias_y_mps2])
                    .collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: None,
        },
        Spec {
            state: "accel_bias_z_mps2",
            trace: trace(
                "accel bias z [m/s^2]",
                states
                    .iter()
                    .map(|s| [s.t_s, s.accel_bias_z_mps2])
                    .collect(),
            ),
            reference: Some(zero_trace("zero", &state_times)),
            mode: SummaryMode::Linear,
            settle_threshold: None,
        },
        Spec {
            state: "seed_mount_err_deg",
            trace: trace(
                "seed mount err [deg]",
                residuals
                    .iter()
                    .map(|sample| [sample.t_s, sample.seed_err_deg])
                    .collect(),
            ),
            reference: Some(zero_trace("zero", &residual_times)),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "align_mount_err_deg",
            trace: trace(
                "align mount err [deg]",
                residuals
                    .iter()
                    .map(|sample| [sample.t_s, sample.align_err_deg])
                    .collect(),
            ),
            reference: Some(zero_trace("zero", &residual_times)),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "legacy_mount_err_deg",
            trace: trace(
                "legacy mount err [deg]",
                residuals
                    .iter()
                    .map(|sample| [sample.t_s, sample.full_a_err_deg])
                    .collect(),
            ),
            reference: Some(zero_trace("zero", &residual_times)),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
        },
        Spec {
            state: "qcs_angle_deg",
            trace: trace(
                "qcs angle [deg]",
                residuals
                    .iter()
                    .map(|sample| [sample.t_s, sample.qcs_angle_deg])
                    .collect(),
            ),
            reference: Some(zero_trace("zero", &residual_times)),
            mode: SummaryMode::Linear,
            settle_threshold: None,
        },
    ];

    specs
        .into_iter()
        .filter_map(|spec| {
            summarize_trace_pair(
                "eskf_gnss_ins",
                spec.state,
                &spec.trace,
                spec.reference.as_ref(),
                spec.mode,
                spec.settle_threshold,
            )
        })
        .collect()
}
