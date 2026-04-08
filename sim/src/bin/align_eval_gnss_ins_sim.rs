use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use sensor_fusion::fusion::{FusionGnssSample, FusionImuSample, SensorFusion};
use sim::visualizer::pipeline::align_replay::{
    axis_angle_deg, quat_rotate, quat_rpy_alg_deg, signed_projected_axis_angle_deg,
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

#[derive(Clone, Copy, Debug)]
struct ImuSample {
    t_s: f64,
    gyro_vehicle_radps: [f64; 3],
    accel_vehicle_mps2: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
struct GnssSample {
    t_s: f64,
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    vel_ned_mps: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
struct ResidualSample {
    t_s: f64,
    align_roll_deg: f64,
    align_pitch_deg: f64,
    align_yaw_deg: f64,
    truth_roll_deg: f64,
    truth_pitch_deg: f64,
    truth_yaw_deg: f64,
    err_roll_deg: f64,
    err_pitch_deg: f64,
    err_yaw_deg: f64,
    sigma_roll_deg: f64,
    sigma_pitch_deg: f64,
    sigma_yaw_deg: f64,
    course_rate_dps: f64,
    a_lat_mps2: f64,
    a_long_mps2: f64,
    rot_err_deg: f64,
    fwd_err_deg: f64,
    down_err_deg: f64,
    fwd_err_signed_deg: f64,
    down_err_signed_deg: f64,
    horiz_applied: bool,
    gravity_applied: bool,
    turn_core_valid: bool,
    straight_core_valid: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let imu = load_imu_samples(&args)?;
    let gnss = load_gnss_samples(&args)?;
    if imu.is_empty() || gnss.is_empty() {
        bail!("need both IMU and GNSS samples");
    }

    let q_truth = quat_from_rpy_alg_deg(args.mount_roll_deg, args.mount_pitch_deg, args.mount_yaw_deg);
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
            let accel_body = quat_rotate(q_truth, s.accel_vehicle_mps2);
            let _ = fusion.process_imu(FusionImuSample {
                t_s: s.t_s as f32,
                gyro_radps: [gyro_body[0] as f32, gyro_body[1] as f32, gyro_body[2] as f32],
                accel_mps2: [accel_body[0] as f32, accel_body[1] as f32, accel_body[2] as f32],
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
                pos_std_m: [args.gnss_pos_std_m, args.gnss_pos_std_m, args.gnss_pos_std_m],
                vel_std_mps: [args.gnss_vel_std_mps, args.gnss_vel_std_mps, args.gnss_vel_std_mps],
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
                let (course_rate_dps, a_lat_mps2, a_long_mps2, gravity_applied, horiz_applied, turn_core_valid, straight_core_valid) =
                    if let Some(debug) = debug {
                        let dt = debug.window.dt as f64;
                        let v_prev = [debug.window.gnss_vel_prev_n[0] as f64, debug.window.gnss_vel_prev_n[1] as f64];
                        let v_curr = [debug.window.gnss_vel_curr_n[0] as f64, debug.window.gnss_vel_curr_n[1] as f64];
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
                            course_rate_dps,
                            a_lat,
                            a_long,
                            debug.trace.after_gravity.is_some(),
                            debug.trace.after_horiz_accel.is_some(),
                            debug.trace.horiz_turn_core_valid,
                            debug.trace.horiz_straight_core_valid,
                        )
                    } else {
                        (0.0, 0.0, 0.0, false, false, false, false)
                    };

                residuals.push(ResidualSample {
                    t_s: s.t_s,
                    align_roll_deg: align_rpy.0,
                    align_pitch_deg: align_rpy.1,
                    align_yaw_deg: align_rpy.2,
                    truth_roll_deg: truth_rpy.0,
                    truth_pitch_deg: truth_rpy.1,
                    truth_yaw_deg: truth_rpy.2,
                    err_roll_deg: wrap_deg180(align_rpy.0 - truth_rpy.0),
                    err_pitch_deg: align_rpy.1 - truth_rpy.1,
                    err_yaw_deg: wrap_deg180(align_rpy.2 - truth_rpy.2),
                    sigma_roll_deg: (align.P[0][0] as f64).max(0.0).sqrt().to_degrees(),
                    sigma_pitch_deg: (align.P[1][1] as f64).max(0.0).sqrt().to_degrees(),
                    sigma_yaw_deg: (align.P[2][2] as f64).max(0.0).sqrt().to_degrees(),
                    course_rate_dps,
                    a_lat_mps2,
                    a_long_mps2,
                    rot_err_deg: quat_angle_deg(q_align, q_truth),
                    fwd_err_deg: axis_angle_deg(quat_rotate(q_align, [1.0, 0.0, 0.0]), quat_rotate(q_truth, [1.0, 0.0, 0.0])),
                    down_err_deg: axis_angle_deg(quat_rotate(q_align, [0.0, 0.0, 1.0]), quat_rotate(q_truth, [0.0, 0.0, 1.0])),
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
    let turn_mean_down_signed = mean_of(residuals.iter().filter(|s| s.turn_core_valid).map(|s| s.down_err_signed_deg.abs()));
    let straight_mean_down_signed =
        mean_of(residuals.iter().filter(|s| s.straight_core_valid).map(|s| s.down_err_signed_deg.abs()));

    println!("input={}", args.data_dir.display());
    println!(
        "source={:?} key={} truth_mount_deg=({:.3},{:.3},{:.3})",
        args.signal_source, args.data_key, args.mount_roll_deg, args.mount_pitch_deg, args.mount_yaw_deg
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
    let time = read_time_csv(&args.data_dir.join("time.csv"))?;
    let gyro_name = match args.signal_source {
        SignalSource::Ref => "ref_gyro.csv".to_string(),
        SignalSource::Meas => format!("gyro-{}.csv", args.data_key),
    };
    let accel_name = match args.signal_source {
        SignalSource::Ref => "ref_accel.csv".to_string(),
        SignalSource::Meas => format!("accel-{}.csv", args.data_key),
    };
    let gyro = read_matrix3_csv(&args.data_dir.join(&gyro_name))
        .with_context(|| format!("failed to load {}", gyro_name))?;
    let accel = read_matrix3_csv(&args.data_dir.join(&accel_name))
        .with_context(|| format!("failed to load {}", accel_name))?;
    if time.len() != gyro.len() || time.len() != accel.len() {
        bail!("IMU files have inconsistent lengths");
    }
    let gyro_is_deg = !matches!(args.signal_source, SignalSource::Ref);
    let mut out = Vec::with_capacity(time.len());
    for i in 0..time.len() {
        let gyro_vehicle_radps = if gyro_is_deg {
            [
                gyro[i][0].to_radians(),
                gyro[i][1].to_radians(),
                gyro[i][2].to_radians(),
            ]
        } else {
            gyro[i]
        };
        out.push(ImuSample {
            t_s: time[i],
            gyro_vehicle_radps,
            accel_vehicle_mps2: accel[i],
        });
    }
    Ok(out)
}

fn load_gnss_samples(args: &Args) -> Result<Vec<GnssSample>> {
    let gps_time = read_time_csv(&args.data_dir.join("gps_time.csv"))?;
    let gps_name = match args.signal_source {
        SignalSource::Ref => "ref_gps.csv".to_string(),
        SignalSource::Meas => format!("gps-{}.csv", args.data_key),
    };
    let gps = read_matrix_csv(&args.data_dir.join(&gps_name), 6)
        .with_context(|| format!("failed to load {}", gps_name))?;
    if gps_time.len() != gps.len() {
        bail!("GNSS files have inconsistent lengths");
    }
    let mut out = Vec::with_capacity(gps.len());
    for i in 0..gps.len() {
        out.push(GnssSample {
            t_s: gps_time[i],
            lat_deg: gps[i][0],
            lon_deg: gps[i][1],
            height_m: gps[i][2],
            vel_ned_mps: [gps[i][3], gps[i][4], gps[i][5]],
        });
    }
    Ok(out)
}

fn read_time_csv(path: &Path) -> Result<Vec<f64>> {
    let rows = read_csv_rows(path)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() != 1 {
            bail!("{} expected 1 column per row", path.display());
        }
        out.push(row[0]);
    }
    Ok(out)
}

fn read_matrix3_csv(path: &Path) -> Result<Vec<[f64; 3]>> {
    let rows = read_matrix_csv(path, 3)?;
    Ok(rows.into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn read_matrix_csv(path: &Path, cols: usize) -> Result<Vec<Vec<f64>>> {
    let rows = read_csv_rows(path)?;
    for row in &rows {
        if row.len() != cols {
            bail!("{} expected {} columns per row", path.display(), cols);
        }
    }
    Ok(rows)
}

fn read_csv_rows(path: &Path) -> Result<Vec<Vec<f64>>> {
    let text = fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let mut row = Vec::new();
        for part in line.split(',') {
            row.push(part.trim().parse::<f64>().with_context(|| {
                format!("failed to parse numeric field in {}: {}", path.display(), line)
            })?);
        }
        out.push(row);
    }
    Ok(out)
}

fn quat_from_rpy_alg_deg(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> [f64; 4] {
    let (sr, cr) = (0.5 * roll_deg.to_radians()).sin_cos();
    let (sp, cp) = (0.5 * pitch_deg.to_radians()).sin_cos();
    let (sy, cy) = (0.5 * yaw_deg.to_radians()).sin_cos();
    quat_normalize([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
}

fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n <= 1.0e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

fn quat_angle_deg(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]).abs().clamp(0.0, 1.0);
    2.0 * dot.acos().to_degrees()
}

fn wrap_deg180(mut deg: f64) -> f64 {
    while deg > 180.0 {
        deg -= 360.0;
    }
    while deg <= -180.0 {
        deg += 360.0;
    }
    deg
}

fn wrap_rad_pi(mut rad: f64) -> f64 {
    while rad > std::f64::consts::PI {
        rad -= 2.0 * std::f64::consts::PI;
    }
    while rad <= -std::f64::consts::PI {
        rad += 2.0 * std::f64::consts::PI;
    }
    rad
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
    v.map(|x| format!("{:.3}", x)).unwrap_or_else(|| "none".to_string())
}

fn write_residual_csv(path: &Path, samples: &[ResidualSample]) -> Result<()> {
    let file = fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "t_s,align_roll_deg,align_pitch_deg,align_yaw_deg,truth_roll_deg,truth_pitch_deg,truth_yaw_deg,err_roll_deg,err_pitch_deg,err_yaw_deg,sigma_roll_deg,sigma_pitch_deg,sigma_yaw_deg,course_rate_dps,a_lat_mps2,a_long_mps2,rot_err_deg,fwd_err_deg,down_err_deg,fwd_err_signed_deg,down_err_signed_deg,horiz_applied,gravity_applied,turn_core_valid,straight_core_valid"
    )?;
    for s in samples {
        writeln!(
            w,
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{}",
            s.t_s,
            s.align_roll_deg,
            s.align_pitch_deg,
            s.align_yaw_deg,
            s.truth_roll_deg,
            s.truth_pitch_deg,
            s.truth_yaw_deg,
            s.err_roll_deg,
            s.err_pitch_deg,
            s.err_yaw_deg,
            s.sigma_roll_deg,
            s.sigma_pitch_deg,
            s.sigma_yaw_deg,
            s.course_rate_dps,
            s.a_lat_mps2,
            s.a_long_mps2,
            s.rot_err_deg,
            s.fwd_err_deg,
            s.down_err_deg,
            s.fwd_err_signed_deg,
            s.down_err_signed_deg,
            s.horiz_applied as u8,
            s.gravity_applied as u8,
            s.turn_core_valid as u8,
            s.straight_core_valid as u8,
        )?;
    }
    Ok(())
}
