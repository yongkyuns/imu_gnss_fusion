use std::{fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::eval::gnss_ins::{quat_angle_deg, quat_from_rpy_alg_deg};
use sim::visualizer::model::{EkfImuSource, Trace};
use sim::visualizer::pipeline::build_plot_data;
use sim::visualizer::pipeline::ekf_compare::{EkfCompareConfig, GnssOutageConfig};

#[derive(Parser, Debug)]
#[command(name = "analyze_mount_recovery")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long, default_value = "internal", value_parser = parse_misalignment)]
    misalignment: EkfImuSource,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long, default_value_t = 1)]
    ekf_predict_imu_decimation: usize,
    #[arg(long)]
    ekf_predict_imu_lpf_cutoff_hz: Option<f64>,
    #[arg(long, default_value_t = 0)]
    gnss_outage_count: usize,
    #[arg(long, default_value_t = 0.0)]
    gnss_outage_duration_s: f64,
    #[arg(long, default_value_t = 1)]
    gnss_outage_seed: u64,
    #[arg(long, default_value_t = 30.0)]
    window_s: f64,
    #[arg(long, default_value_t = 5)]
    top_k: usize,
    #[arg(long, default_value_t = 0.0)]
    analysis_start_s: f64,
}

fn parse_misalignment(s: &str) -> Result<EkfImuSource, String> {
    EkfImuSource::from_cli_value(s)
}

fn trace_by_name<'a>(traces: &'a [Trace], name: &str) -> Result<&'a Trace> {
    traces
        .iter()
        .find(|t| t.name == name)
        .with_context(|| format!("missing trace `{name}`"))
}

fn sample_trace(trace: &Trace, t_s: f64) -> Option<f64> {
    if trace.points.is_empty() {
        return None;
    }
    let idx = trace.points.partition_point(|point| point[0] < t_s);
    let left = trace
        .points
        .get(idx.saturating_sub(1))
        .map(|point| ((point[0] - t_s).abs(), point[1]));
    let right = trace
        .points
        .get(idx)
        .map(|point| ((point[0] - t_s).abs(), point[1]));
    match (left, right) {
        (Some((left_dt, left_value)), Some((right_dt, right_value))) => {
            if right_dt < left_dt {
                Some(right_value)
            } else {
                Some(left_value)
            }
        }
        (Some((_, value)), None) | (None, Some((_, value))) => Some(value),
        (None, None) => None,
    }
}

fn build_mount_quat_err_trace(estimate: [&Trace; 3], reference: [&Trace; 3]) -> Trace {
    let mut points = Vec::with_capacity(estimate[0].points.len());
    for [t_s, roll_deg] in &estimate[0].points {
        let Some(pitch_deg) = sample_trace(estimate[1], *t_s) else {
            continue;
        };
        let Some(yaw_deg) = sample_trace(estimate[2], *t_s) else {
            continue;
        };
        let Some(ref_roll_deg) = sample_trace(reference[0], *t_s) else {
            continue;
        };
        let Some(ref_pitch_deg) = sample_trace(reference[1], *t_s) else {
            continue;
        };
        let Some(ref_yaw_deg) = sample_trace(reference[2], *t_s) else {
            continue;
        };
        let q_est = quat_from_rpy_alg_deg(*roll_deg, pitch_deg, yaw_deg);
        let q_ref = quat_from_rpy_alg_deg(ref_roll_deg, ref_pitch_deg, ref_yaw_deg);
        points.push([*t_s, quat_angle_deg(q_est, q_ref)]);
    }
    Trace {
        name: "mount quat err [deg]".to_string(),
        points,
    }
}

fn mean_abs_in_window(trace: &Trace, t0: f64, t1: f64) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for [t_s, value] in &trace.points {
        if *t_s >= t0 && *t_s <= t1 && value.is_finite() {
            sum += value.abs();
            n += 1;
        }
    }
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

fn max_abs_in_window(trace: &Trace, t0: f64, t1: f64) -> f64 {
    let mut best = f64::NAN;
    for [t_s, value] in &trace.points {
        if *t_s >= t0 && *t_s <= t1 && value.is_finite() {
            best = if best.is_nan() {
                value.abs()
            } else {
                best.max(value.abs())
            };
        }
    }
    best
}

#[derive(Debug)]
struct WindowRow {
    t0_s: f64,
    t1_s: f64,
    err_start_deg: f64,
    err_end_deg: f64,
    err_delta_deg: f64,
    mean_abs_course_dps: f64,
    max_abs_course_dps: f64,
    mean_abs_lat_mps2: f64,
    max_abs_lat_mps2: f64,
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return f64::NAN;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 1e-12 || var_y <= 1e-12 {
        f64::NAN
    } else {
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut bytes = Vec::new();
    File::open(&args.logfile)
        .with_context(|| format!("failed to open {}", args.logfile.display()))?
        .read_to_end(&mut bytes)
        .context("failed to read log")?;

    let ekf_cfg = EkfCompareConfig {
        predict_imu_decimation: args.ekf_predict_imu_decimation.max(1),
        predict_imu_lpf_cutoff_hz: args.ekf_predict_imu_lpf_cutoff_hz,
        ..EkfCompareConfig::default()
    };
    let (data, _has_itow) = build_plot_data(
        &bytes,
        args.max_records,
        args.misalignment,
        ekf_cfg,
        GnssOutageConfig {
            count: args.gnss_outage_count,
            duration_s: args.gnss_outage_duration_s,
            seed: args.gnss_outage_seed,
        },
    );

    let mount_err = build_mount_quat_err_trace(
        [
            trace_by_name(&data.eskf_misalignment, "ESKF full mount roll [deg]")?,
            trace_by_name(&data.eskf_misalignment, "ESKF full mount pitch [deg]")?,
            trace_by_name(&data.eskf_misalignment, "ESKF full mount yaw [deg]")?,
        ],
        [
            trace_by_name(&data.eskf_misalignment, "ESF-ALG mount roll [deg]")?,
            trace_by_name(&data.eskf_misalignment, "ESF-ALG mount pitch [deg]")?,
            trace_by_name(&data.eskf_misalignment, "ESF-ALG mount yaw [deg]")?,
        ],
    );
    let course = trace_by_name(&data.align_res_vel, "course rate [deg/s]")?;
    let lat = trace_by_name(&data.align_res_vel, "a_lat [m/s^2]")?;

    let mut rows = Vec::<WindowRow>::new();
    for [t0_s, err_start_deg] in &mount_err.points {
        if *t0_s < args.analysis_start_s {
            continue;
        }
        let t1_s = *t0_s + args.window_s;
        let Some(err_end_deg) = sample_trace(&mount_err, t1_s) else {
            continue;
        };
        let row = WindowRow {
            t0_s: *t0_s,
            t1_s,
            err_start_deg: *err_start_deg,
            err_end_deg,
            err_delta_deg: err_end_deg - *err_start_deg,
            mean_abs_course_dps: mean_abs_in_window(course, *t0_s, t1_s),
            max_abs_course_dps: max_abs_in_window(course, *t0_s, t1_s),
            mean_abs_lat_mps2: mean_abs_in_window(lat, *t0_s, t1_s),
            max_abs_lat_mps2: max_abs_in_window(lat, *t0_s, t1_s),
        };
        if row.err_start_deg.is_finite()
            && row.err_end_deg.is_finite()
            && row.err_delta_deg.is_finite()
            && row.mean_abs_course_dps.is_finite()
            && row.max_abs_course_dps.is_finite()
            && row.mean_abs_lat_mps2.is_finite()
            && row.max_abs_lat_mps2.is_finite()
        {
            rows.push(row);
        }
    }

    let improve: Vec<f64> = rows.iter().map(|row| -row.err_delta_deg).collect();
    let mean_course: Vec<f64> = rows.iter().map(|row| row.mean_abs_course_dps).collect();
    let max_course: Vec<f64> = rows.iter().map(|row| row.max_abs_course_dps).collect();
    let mean_lat: Vec<f64> = rows.iter().map(|row| row.mean_abs_lat_mps2).collect();
    let max_lat: Vec<f64> = rows.iter().map(|row| row.max_abs_lat_mps2).collect();

    println!("window_s={:.1}", args.window_s);
    println!(
        "corr(improvement, mean_abs_course_dps)={:.3}",
        pearson(&improve, &mean_course)
    );
    println!(
        "corr(improvement, max_abs_course_dps)={:.3}",
        pearson(&improve, &max_course)
    );
    println!(
        "corr(improvement, mean_abs_lat_mps2)={:.3}",
        pearson(&improve, &mean_lat)
    );
    println!(
        "corr(improvement, max_abs_lat_mps2)={:.3}",
        pearson(&improve, &max_lat)
    );

    let mut by_improve = rows.iter().collect::<Vec<_>>();
    by_improve.sort_by(|a, b| {
        a.err_delta_deg
            .partial_cmp(&b.err_delta_deg)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("top_recovery_windows:");
    for row in by_improve.iter().take(args.top_k) {
        println!(
            "  t=[{:.1},{:.1}] err_start={:.2} err_end={:.2} improve={:.2} mean|course|={:.2} max|course|={:.2} mean|a_lat|={:.2} max|a_lat|={:.2}",
            row.t0_s,
            row.t1_s,
            row.err_start_deg,
            row.err_end_deg,
            -row.err_delta_deg,
            row.mean_abs_course_dps,
            row.max_abs_course_dps,
            row.mean_abs_lat_mps2,
            row.max_abs_lat_mps2
        );
    }

    let mut by_worsen = rows.iter().collect::<Vec<_>>();
    by_worsen.sort_by(|a, b| {
        b.err_delta_deg
            .partial_cmp(&a.err_delta_deg)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("top_worsening_windows:");
    for row in by_worsen.iter().take(args.top_k) {
        println!(
            "  t=[{:.1},{:.1}] err_start={:.2} err_end={:.2} improve={:.2} mean|course|={:.2} max|course|={:.2} mean|a_lat|={:.2} max|a_lat|={:.2}",
            row.t0_s,
            row.t1_s,
            row.err_start_deg,
            row.err_end_deg,
            -row.err_delta_deg,
            row.mean_abs_course_dps,
            row.max_abs_course_dps,
            row.mean_abs_lat_mps2,
            row.max_abs_lat_mps2
        );
    }

    let mut by_course = rows.iter().collect::<Vec<_>>();
    by_course.sort_by(|a, b| {
        b.max_abs_course_dps
            .partial_cmp(&a.max_abs_course_dps)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("top_course_windows:");
    for row in by_course.iter().take(args.top_k) {
        println!(
            "  t=[{:.1},{:.1}] err_start={:.2} err_end={:.2} improve={:.2} mean|course|={:.2} max|course|={:.2} mean|a_lat|={:.2} max|a_lat|={:.2}",
            row.t0_s,
            row.t1_s,
            row.err_start_deg,
            row.err_end_deg,
            -row.err_delta_deg,
            row.mean_abs_course_dps,
            row.max_abs_course_dps,
            row.mean_abs_lat_mps2,
            row.max_abs_lat_mps2
        );
    }

    let mut by_lat = rows.iter().collect::<Vec<_>>();
    by_lat.sort_by(|a, b| {
        b.max_abs_lat_mps2
            .partial_cmp(&a.max_abs_lat_mps2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("top_lat_windows:");
    for row in by_lat.iter().take(args.top_k) {
        println!(
            "  t=[{:.1},{:.1}] err_start={:.2} err_end={:.2} improve={:.2} mean|course|={:.2} max|course|={:.2} mean|a_lat|={:.2} max|a_lat|={:.2}",
            row.t0_s,
            row.t1_s,
            row.err_start_deg,
            row.err_end_deg,
            -row.err_delta_deg,
            row.mean_abs_course_dps,
            row.max_abs_course_dps,
            row.mean_abs_lat_mps2,
            row.max_abs_lat_mps2
        );
    }

    Ok(())
}
