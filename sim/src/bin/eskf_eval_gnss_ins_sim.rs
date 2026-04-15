use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use sensor_fusion::fusion::{FusionGnssSample, FusionImuSample, SensorFusion};
use sim::datasets::gnss_ins_sim::{
    GnssSample as DatasetGnssSample, ImuSample as DatasetImuSample,
    load_gnss_samples as load_dataset_gnss_samples, load_imu_samples as load_dataset_imu_samples,
};
use sim::eval::gnss_ins::{
    as_q64, course_rate_deg, horiz_speed, quat_angle_deg, quat_conj, quat_from_rpy_alg_deg,
    quat_mul, quat_rotate,
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

    #[arg(long)]
    residual_csv: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SignalSource {
    Ref,
    Meas,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SeedSource {
    InternalAlign,
    ExternalTruth,
}

type ImuSample = DatasetImuSample;
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

fn main() -> Result<()> {
    let args = Args::parse();
    let imu = load_imu_samples(&args)?;
    let gnss = load_gnss_samples(&args)?;
    if imu.is_empty() || gnss.is_empty() {
        bail!("need both IMU and GNSS samples");
    }

    let q_truth = quat_from_rpy_alg_deg(
        args.mount_roll_deg,
        args.mount_pitch_deg,
        args.mount_yaw_deg,
    );
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

    let mut imu_idx = 0usize;
    let mut gnss_idx = 0usize;
    let mut residuals = Vec::<ResidualSample>::new();
    let mut mount_ready_s = None::<f64>;
    let mut ekf_init_s = None::<f64>;
    let mut seed_q_vb = None::<[f64; 4]>;
    let mut prev_gnss = None::<GnssSample>;

    while imu_idx < imu.len() || gnss_idx < gnss.len() {
        let next_imu_t = imu.get(imu_idx).map(|s| s.t_s);
        let next_gnss_t = gnss.get(gnss_idx).map(|s| s.t_s);
        let take_imu = match (next_imu_t, next_gnss_t) {
            (Some(ti), Some(tg)) => ti <= tg,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };

        if take_imu {
            let s = imu[imu_idx];
            let gyro_body = quat_rotate(q_truth, s.gyro_vehicle_radps);
            let accel_body = quat_rotate(q_truth, s.accel_vehicle_mps2);
            let _ = fusion.process_imu(FusionImuSample {
                t_s: s.t_s as f32,
                gyro_radps: [
                    gyro_body[0] as f32,
                    gyro_body[1] as f32,
                    gyro_body[2] as f32,
                ],
                accel_mps2: [
                    accel_body[0] as f32,
                    accel_body[1] as f32,
                    accel_body[2] as f32,
                ],
            });
            imu_idx += 1;
        } else {
            let s = gnss[gnss_idx];
            let update = fusion.process_gnss(FusionGnssSample {
                t_s: s.t_s as f32,
                lat_deg: s.lat_deg as f32,
                lon_deg: s.lon_deg as f32,
                height_m: s.height_m as f32,
                vel_ned_mps: [
                    s.vel_ned_mps[0] as f32,
                    s.vel_ned_mps[1] as f32,
                    s.vel_ned_mps[2] as f32,
                ],
                pos_std_m: [
                    args.gnss_pos_std_m,
                    args.gnss_pos_std_m,
                    args.gnss_pos_std_m,
                ],
                vel_std_mps: [
                    args.gnss_vel_std_mps,
                    args.gnss_vel_std_mps,
                    args.gnss_vel_std_mps,
                ],
                heading_rad: None,
            });
            if update.mount_ready_changed && update.mount_ready && mount_ready_s.is_none() {
                mount_ready_s = Some(s.t_s);
            }
            if update.ekf_initialized_now && ekf_init_s.is_none() {
                ekf_init_s = Some(s.t_s);
                seed_q_vb = fusion.mount_q_vb().map(as_q64);
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
                    t_s: s.t_s,
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
            gnss_idx += 1;
        }
    }

    if residuals.is_empty() {
        bail!("no ESKF samples produced");
    }

    let last = residuals.last().unwrap();
    let tail = tail_window(&residuals, 60.0);
    println!("input={}", args.data_dir.display());
    println!(
        "source={:?} seed={:?} r_body_vel={} key={} truth_mount_deg=({:.3},{:.3},{:.3})",
        args.signal_source,
        args.seed_source,
        args.r_body_vel
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "default".to_string()),
        args.data_key,
        args.mount_roll_deg,
        args.mount_pitch_deg,
        args.mount_yaw_deg
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

    if let Some(path) = &args.residual_csv {
        write_residual_csv(path, &residuals)?;
        println!("wrote residual CSV: {}", path.display());
    }

    Ok(())
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

fn load_imu_samples(args: &Args) -> Result<Vec<ImuSample>> {
    load_dataset_imu_samples(
        &args.data_dir,
        matches!(args.signal_source, SignalSource::Ref),
        args.data_key,
    )
}

fn load_gnss_samples(args: &Args) -> Result<Vec<GnssSample>> {
    load_dataset_gnss_samples(
        &args.data_dir,
        matches!(args.signal_source, SignalSource::Ref),
        args.data_key,
    )
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
