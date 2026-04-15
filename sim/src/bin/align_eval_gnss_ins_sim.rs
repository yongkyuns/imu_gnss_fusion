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
    quat_angle_deg, quat_from_rpy_alg_deg, quat_rotate, wrap_deg180, wrap_rad_pi,
};
use sim::visualizer::pipeline::align_replay::{
    axis_angle_deg, quat_rpy_alg_deg, signed_projected_axis_angle_deg,
};

#[derive(Parser, Debug)]
#[command(name = "align_eval_gnss_ins_sim")]
struct Args {
    #[arg(value_name = "DATA_DIR")]
    data_dir: PathBuf,

    #[arg(long, value_enum, default_value_t = SignalSource::Ref)]
    signal_source: SignalSource,

    #[arg(long, default_value_t = 0)]
    data_key: usize,

    #[arg(long, default_value_t = 0.0)]
    mount_roll_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    mount_pitch_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    mount_yaw_deg: f64,

    #[arg(long, default_value_t = 0.0)]
    accel_bias_x_mps2: f64,
    #[arg(long, default_value_t = 0.0)]
    accel_bias_y_mps2: f64,
    #[arg(long, default_value_t = 0.0)]
    accel_bias_z_mps2: f64,

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

type ImuSample = DatasetImuSample;
type GnssSample = DatasetGnssSample;

#[derive(Clone, Copy, Debug)]
struct ResidualSample {
    t_s: f64,
    align_roll_deg: f64,
    align_pitch_deg: f64,
    align_yaw_deg: f64,
    yaw_start_deg: f64,
    yaw_after_gravity_deg: f64,
    yaw_after_horiz_deg: f64,
    yaw_after_turn_gyro_deg: f64,
    yaw_delta_gravity_deg: f64,
    yaw_delta_horiz_deg: f64,
    yaw_delta_turn_gyro_deg: f64,
    truth_roll_deg: f64,
    truth_pitch_deg: f64,
    truth_yaw_deg: f64,
    err_roll_deg: f64,
    err_pitch_deg: f64,
    err_yaw_deg: f64,
    sigma_roll_deg: f64,
    sigma_pitch_deg: f64,
    sigma_yaw_deg: f64,
    speed_mps: f64,
    course_rate_dps: f64,
    a_lat_mps2: f64,
    a_long_mps2: f64,
    gravity_lpb_x: f64,
    gravity_lpb_y: f64,
    gravity_lpb_z: f64,
    horiz_obs_accel_vx: f64,
    horiz_obs_accel_vy: f64,
    horiz_accel_bx: f64,
    horiz_accel_by: f64,
    rot_err_deg: f64,
    fwd_err_deg: f64,
    down_err_deg: f64,
    fwd_err_signed_deg: f64,
    down_err_signed_deg: f64,
    horiz_applied: bool,
    gravity_applied: bool,
    turn_core_valid: bool,
    straight_core_valid: bool,
    coarse_alignment_ready: bool,
    horiz_angle_err_deg: f64,
    horiz_effective_std_deg: f64,
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
    let truth_rpy = quat_rpy_alg_deg(q_truth[0], q_truth[1], q_truth[2], q_truth[3]);

    let mut fusion = SensorFusion::new();
    let mut imu_idx = 0usize;
    let mut gnss_idx = 0usize;
    let mut residuals = Vec::<ResidualSample>::new();
    let mut mount_ready_s = None::<f64>;

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
            let accel_vehicle_biased = [
                s.accel_vehicle_mps2[0] + args.accel_bias_x_mps2,
                s.accel_vehicle_mps2[1] + args.accel_bias_y_mps2,
                s.accel_vehicle_mps2[2] + args.accel_bias_z_mps2,
            ];
            let accel_body = quat_rotate(q_truth, accel_vehicle_biased);
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

            if let Some(align) = fusion.align() {
                let q_align = [
                    align.q_vb[0] as f64,
                    align.q_vb[1] as f64,
                    align.q_vb[2] as f64,
                    align.q_vb[3] as f64,
                ];
                let align_rpy = quat_rpy_alg_deg(q_align[0], q_align[1], q_align[2], q_align[3]);
                let debug = fusion.align_debug();
                let (
                    speed_mps,
                    course_rate_dps,
                    a_lat_mps2,
                    a_long_mps2,
                    gravity_applied,
                    horiz_applied,
                    turn_core_valid,
                    straight_core_valid,
                ) = if let Some(debug) = debug {
                    let dt = debug.window.dt as f64;
                    let v_prev = [
                        debug.window.gnss_vel_prev_n[0] as f64,
                        debug.window.gnss_vel_prev_n[1] as f64,
                    ];
                    let v_curr = [
                        debug.window.gnss_vel_curr_n[0] as f64,
                        debug.window.gnss_vel_curr_n[1] as f64,
                    ];
                    let course_prev = v_prev[1].atan2(v_prev[0]);
                    let course_curr = v_curr[1].atan2(v_curr[0]);
                    let course_rate_dps = if dt > 1.0e-9 {
                        wrap_rad_pi(course_curr - course_prev).to_degrees() / dt
                    } else {
                        0.0
                    };
                    let a_n = [
                        (v_curr[0] - v_prev[0]) / dt.max(1.0e-3),
                        (v_curr[1] - v_prev[1]) / dt.max(1.0e-3),
                    ];
                    let v_mid = [0.5 * (v_prev[0] + v_curr[0]), 0.5 * (v_prev[1] + v_curr[1])];
                    let speed = (v_mid[0] * v_mid[0] + v_mid[1] * v_mid[1]).sqrt();
                    let (a_long, a_lat) = if speed > 1.0e-9 {
                        let t_hat = [v_mid[0] / speed, v_mid[1] / speed];
                        let lat_hat = [-t_hat[1], t_hat[0]];
                        (
                            t_hat[0] * a_n[0] + t_hat[1] * a_n[1],
                            lat_hat[0] * a_n[0] + lat_hat[1] * a_n[1],
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    (
                        speed,
                        course_rate_dps,
                        a_lat,
                        a_long,
                        debug.trace.after_gravity.is_some(),
                        debug.trace.after_horiz_accel.is_some(),
                        debug.trace.horiz_turn_core_valid,
                        debug.trace.horiz_straight_core_valid,
                    )
                } else {
                    (0.0, 0.0, 0.0, 0.0, false, false, false, false)
                };
                let (
                    yaw_start_deg,
                    yaw_after_gravity_deg,
                    yaw_after_horiz_deg,
                    yaw_after_turn_gyro_deg,
                    yaw_delta_gravity_deg,
                    yaw_delta_horiz_deg,
                    yaw_delta_turn_gyro_deg,
                    coarse_alignment_ready,
                    horiz_angle_err_deg,
                    horiz_effective_std_deg,
                    horiz_obs_accel_vx,
                    horiz_obs_accel_vy,
                    horiz_accel_bx,
                    horiz_accel_by,
                ) = if let Some(debug) = debug {
                    let yaw_of = |q: [f32; 4]| {
                        let (_, _, y) =
                            quat_rpy_alg_deg(q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64);
                        y
                    };
                    let yaw_start = yaw_of(debug.trace.q_start);
                    let yaw_after_gravity =
                        debug.trace.after_gravity.map(yaw_of).unwrap_or(f64::NAN);
                    let yaw_after_horiz = debug
                        .trace
                        .after_horiz_accel
                        .map(yaw_of)
                        .unwrap_or(f64::NAN);
                    let yaw_after_turn_gyro =
                        debug.trace.after_turn_gyro.map(yaw_of).unwrap_or(f64::NAN);
                    let yaw_delta_gravity = debug
                        .trace
                        .after_gravity
                        .map(|q| wrap_deg180(yaw_of(q) - yaw_start))
                        .unwrap_or(f64::NAN);
                    let yaw_delta_horiz = debug
                        .trace
                        .after_horiz_accel
                        .map(|q| {
                            let prev = debug.trace.after_gravity.map(yaw_of).unwrap_or(yaw_start);
                            wrap_deg180(yaw_of(q) - prev)
                        })
                        .unwrap_or(f64::NAN);
                    let yaw_delta_turn_gyro = debug
                        .trace
                        .after_turn_gyro
                        .map(|q| {
                            let prev = debug
                                .trace
                                .after_horiz_accel
                                .map(yaw_of)
                                .or_else(|| debug.trace.after_gravity.map(yaw_of))
                                .unwrap_or(yaw_start);
                            wrap_deg180(yaw_of(q) - prev)
                        })
                        .unwrap_or(f64::NAN);
                    (
                        yaw_start,
                        yaw_after_gravity,
                        yaw_after_horiz,
                        yaw_after_turn_gyro,
                        yaw_delta_gravity,
                        yaw_delta_horiz,
                        yaw_delta_turn_gyro,
                        debug.trace.coarse_alignment_ready,
                        debug
                            .trace
                            .horiz_angle_err_rad
                            .map(|x| (x as f64).to_degrees())
                            .unwrap_or(f64::NAN),
                        debug
                            .trace
                            .horiz_effective_std_rad
                            .map(|x| (x as f64).to_degrees())
                            .unwrap_or(f64::NAN),
                        debug
                            .trace
                            .horiz_obs_accel_vx
                            .map(|x| x as f64)
                            .unwrap_or(f64::NAN),
                        debug
                            .trace
                            .horiz_obs_accel_vy
                            .map(|x| x as f64)
                            .unwrap_or(f64::NAN),
                        debug
                            .trace
                            .horiz_accel_bx
                            .map(|x| x as f64)
                            .unwrap_or(f64::NAN),
                        debug
                            .trace
                            .horiz_accel_by
                            .map(|x| x as f64)
                            .unwrap_or(f64::NAN),
                    )
                } else {
                    (
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        false,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                    )
                };

                residuals.push(ResidualSample {
                    t_s: s.t_s,
                    align_roll_deg: align_rpy.0,
                    align_pitch_deg: align_rpy.1,
                    align_yaw_deg: align_rpy.2,
                    yaw_start_deg,
                    yaw_after_gravity_deg,
                    yaw_after_horiz_deg,
                    yaw_after_turn_gyro_deg,
                    yaw_delta_gravity_deg,
                    yaw_delta_horiz_deg,
                    yaw_delta_turn_gyro_deg,
                    truth_roll_deg: truth_rpy.0,
                    truth_pitch_deg: truth_rpy.1,
                    truth_yaw_deg: truth_rpy.2,
                    err_roll_deg: wrap_deg180(align_rpy.0 - truth_rpy.0),
                    err_pitch_deg: align_rpy.1 - truth_rpy.1,
                    err_yaw_deg: wrap_deg180(align_rpy.2 - truth_rpy.2),
                    sigma_roll_deg: (align.P[0][0] as f64).max(0.0).sqrt().to_degrees(),
                    sigma_pitch_deg: (align.P[1][1] as f64).max(0.0).sqrt().to_degrees(),
                    sigma_yaw_deg: (align.P[2][2] as f64).max(0.0).sqrt().to_degrees(),
                    speed_mps,
                    course_rate_dps,
                    a_lat_mps2,
                    a_long_mps2,
                    gravity_lpb_x: align.gravity_lp_b[0] as f64,
                    gravity_lpb_y: align.gravity_lp_b[1] as f64,
                    gravity_lpb_z: align.gravity_lp_b[2] as f64,
                    horiz_obs_accel_vx,
                    horiz_obs_accel_vy,
                    horiz_accel_bx,
                    horiz_accel_by,
                    rot_err_deg: quat_angle_deg(q_align, q_truth),
                    fwd_err_deg: axis_angle_deg(
                        quat_rotate(q_align, [1.0, 0.0, 0.0]),
                        quat_rotate(q_truth, [1.0, 0.0, 0.0]),
                    ),
                    down_err_deg: axis_angle_deg(
                        quat_rotate(q_align, [0.0, 0.0, 1.0]),
                        quat_rotate(q_truth, [0.0, 0.0, 1.0]),
                    ),
                    fwd_err_signed_deg: signed_projected_axis_angle_deg(
                        quat_rotate(q_align, [1.0, 0.0, 0.0]),
                        quat_rotate(q_truth, [1.0, 0.0, 0.0]),
                        quat_rotate(q_truth, [0.0, 0.0, 1.0]),
                    ),
                    down_err_signed_deg: signed_projected_axis_angle_deg(
                        quat_rotate(q_align, [0.0, 0.0, 1.0]),
                        quat_rotate(q_truth, [0.0, 0.0, 1.0]),
                        quat_rotate(q_truth, [0.0, 1.0, 0.0]),
                    ),
                    horiz_applied,
                    gravity_applied,
                    turn_core_valid,
                    straight_core_valid,
                    coarse_alignment_ready,
                    horiz_angle_err_deg,
                    horiz_effective_std_deg,
                });
            }
            gnss_idx += 1;
        }
    }

    if residuals.is_empty() {
        bail!("no align samples produced");
    }

    let first_t = residuals.first().map(|s| s.t_s).unwrap_or(0.0);
    let last = residuals.last().unwrap();
    let n = residuals.len() as f64;
    let mean_down = residuals.iter().map(|s| s.down_err_deg).sum::<f64>() / n;
    let mean_fwd = residuals.iter().map(|s| s.fwd_err_deg).sum::<f64>() / n;
    let mean_rot = residuals.iter().map(|s| s.rot_err_deg).sum::<f64>() / n;
    let mean_abs_roll = residuals.iter().map(|s| s.err_roll_deg.abs()).sum::<f64>() / n;
    let mean_abs_pitch = residuals.iter().map(|s| s.err_pitch_deg.abs()).sum::<f64>() / n;
    let turn_mean_down_signed = mean_of(
        residuals
            .iter()
            .filter(|s| s.turn_core_valid)
            .map(|s| s.down_err_signed_deg.abs()),
    );
    let straight_mean_down_signed = mean_of(
        residuals
            .iter()
            .filter(|s| s.straight_core_valid)
            .map(|s| s.down_err_signed_deg.abs()),
    );

    println!("input={}", args.data_dir.display());
    println!(
        "source={:?} key={} truth_mount_deg=({:.3},{:.3},{:.3})",
        args.signal_source,
        args.data_key,
        args.mount_roll_deg,
        args.mount_pitch_deg,
        args.mount_yaw_deg
    );
    println!(
        "n={} align_init={:.3}s mount_ready={}",
        residuals.len(),
        first_t,
        mount_ready_s
            .map(|t| format!("{:.3}s", t))
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "mean_abs_err_deg roll={:.3} pitch={:.3} | final_err_deg roll={:.3} pitch={:.3} yaw={:.3}",
        mean_abs_roll, mean_abs_pitch, last.err_roll_deg, last.err_pitch_deg, last.err_yaw_deg
    );
    println!(
        "rot/fwd/down mean_deg {:.3} / {:.3} / {:.3} | final_deg {:.3} / {:.3} / {:.3}",
        mean_rot, mean_fwd, mean_down, last.rot_err_deg, last.fwd_err_deg, last.down_err_deg
    );
    println!(
        "pitch_like_down mean turn_windows={} straight_windows={} final={:.3}",
        fmt_opt(turn_mean_down_signed),
        fmt_opt(straight_mean_down_signed),
        last.down_err_signed_deg
    );

    if let Some(path) = &args.residual_csv {
        write_residual_csv(path, &residuals)?;
        println!("wrote residual CSV: {}", path.display());
    }

    Ok(())
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

fn mean_of<I>(iter: I) -> Option<f64>
where
    I: Iterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut n = 0usize;
    for v in iter {
        sum += v;
        n += 1;
    }
    (n > 0).then_some(sum / n as f64)
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{:.3}", x))
        .unwrap_or_else(|| "none".to_string())
}

fn write_residual_csv(path: &Path, samples: &[ResidualSample]) -> Result<()> {
    let file =
        fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "t_s,align_roll_deg,align_pitch_deg,align_yaw_deg,yaw_start_deg,yaw_after_gravity_deg,yaw_after_horiz_deg,yaw_after_turn_gyro_deg,yaw_delta_gravity_deg,yaw_delta_horiz_deg,yaw_delta_turn_gyro_deg,truth_roll_deg,truth_pitch_deg,truth_yaw_deg,err_roll_deg,err_pitch_deg,err_yaw_deg,sigma_roll_deg,sigma_pitch_deg,sigma_yaw_deg,speed_mps,course_rate_dps,a_lat_mps2,a_long_mps2,gravity_lpb_x,gravity_lpb_y,gravity_lpb_z,horiz_obs_accel_vx,horiz_obs_accel_vy,horiz_accel_bx,horiz_accel_by,rot_err_deg,fwd_err_deg,down_err_deg,fwd_err_signed_deg,down_err_signed_deg,horiz_applied,gravity_applied,turn_core_valid,straight_core_valid,coarse_alignment_ready,horiz_angle_err_deg,horiz_effective_std_deg"
    )?;
    for s in samples {
        writeln!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            s.t_s,
            s.align_roll_deg,
            s.align_pitch_deg,
            s.align_yaw_deg,
            s.yaw_start_deg,
            s.yaw_after_gravity_deg,
            s.yaw_after_horiz_deg,
            s.yaw_after_turn_gyro_deg,
            s.yaw_delta_gravity_deg,
            s.yaw_delta_horiz_deg,
            s.yaw_delta_turn_gyro_deg,
            s.truth_roll_deg,
            s.truth_pitch_deg,
            s.truth_yaw_deg,
            s.err_roll_deg,
            s.err_pitch_deg,
            s.err_yaw_deg,
            s.sigma_roll_deg,
            s.sigma_pitch_deg,
            s.sigma_yaw_deg,
            s.speed_mps,
            s.course_rate_dps,
            s.a_lat_mps2,
            s.a_long_mps2,
            s.gravity_lpb_x,
            s.gravity_lpb_y,
            s.gravity_lpb_z,
            s.horiz_obs_accel_vx,
            s.horiz_obs_accel_vy,
            s.horiz_accel_bx,
            s.horiz_accel_by,
            s.rot_err_deg,
            s.fwd_err_deg,
            s.down_err_deg,
            s.fwd_err_signed_deg,
            s.down_err_signed_deg,
            s.horiz_applied as u8,
            s.gravity_applied as u8,
            s.turn_core_valid as u8,
            s.straight_core_valid as u8,
            s.coarse_alignment_ready as u8,
            s.horiz_angle_err_deg,
            s.horiz_effective_std_deg,
        )?;
    }
    Ok(())
}
