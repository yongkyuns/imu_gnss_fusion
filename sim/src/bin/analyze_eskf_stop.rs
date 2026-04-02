use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::ubxlog::parse_ubx_frames;
use sim::visualizer::model::EkfImuSource;
use sim::visualizer::pipeline::build_plot_data;
use sim::visualizer::pipeline::ekf_compare::{EkfCompareConfig, GnssOutageConfig};

#[derive(Parser, Debug)]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,

    #[arg(long, default_value_t = 940.0)]
    window_start_s: f64,

    #[arg(long, default_value_t = 960.0)]
    window_end_s: f64,

    #[arg(long, default_value_t = 3)]
    gnss_outage_count: usize,

    #[arg(long, default_value_t = 30.0)]
    gnss_outage_duration_s: f64,

    #[arg(long, default_value_t = 44)]
    gnss_outage_seed: u64,

    #[arg(long, default_value = "esf-alg")]
    ekf_imu_source: String,

    #[arg(long, default_value_t = 1)]
    ekf_predict_imu_decimation: usize,

    #[arg(long)]
    ekf_predict_imu_lpf_cutoff_hz: Option<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let data = fs::read(&args.logfile)
        .with_context(|| format!("failed to read {}", args.logfile.display()))?;
    let ekf_imu_source = match args.ekf_imu_source.as_str() {
        "align" => EkfImuSource::Align,
        "esf-alg" => EkfImuSource::EsfAlg,
        other => anyhow::bail!("unsupported --ekf-imu-source: {other}"),
    };
    let _ = parse_ubx_frames(&data, None);
    let (plot, _) = build_plot_data(
        &data,
        None,
        ekf_imu_source,
        EkfCompareConfig {
            predict_imu_decimation: args.ekf_predict_imu_decimation,
            predict_imu_lpf_cutoff_hz: args.ekf_predict_imu_lpf_cutoff_hz,
            ..EkfCompareConfig::default()
        },
        GnssOutageConfig {
            count: args.gnss_outage_count,
            duration_s: args.gnss_outage_duration_s,
            seed: args.gnss_outage_seed,
        },
    );

    let w0 = args.window_start_s;
    let w1 = args.window_end_s;

    print_trace_stats("ESKF velN [m/s]", find_trace(&plot.eskf_cmp_vel, "ESKF velN [m/s]"), w0, w1);
    print_trace_stats("ESKF velE [m/s]", find_trace(&plot.eskf_cmp_vel, "ESKF velE [m/s]"), w0, w1);
    print_trace_stats("UBX velN [m/s]", find_trace(&plot.eskf_cmp_vel, "UBX velN [m/s]"), w0, w1);
    print_trace_stats("UBX velE [m/s]", find_trace(&plot.eskf_cmp_vel, "UBX velE [m/s]"), w0, w1);

    print_trace_stats(
        "ESKF pitch [deg]",
        find_trace(&plot.eskf_cmp_att, "ESKF pitch [deg]"),
        w0,
        w1,
    );
    print_trace_stats(
        "ESKF roll [deg]",
        find_trace(&plot.eskf_cmp_att, "ESKF roll [deg]"),
        w0,
        w1,
    );
    print_trace_stats(
        "NAV-ATT pitch [deg]",
        find_trace(&plot.eskf_cmp_att, "NAV-ATT pitch [deg]"),
        w0,
        w1,
    );
    print_trace_stats(
        "NAV-ATT roll [deg]",
        find_trace(&plot.eskf_cmp_att, "NAV-ATT roll [deg]"),
        w0,
        w1,
    );
    print_diff_stats(
        "pitch err [deg]",
        find_trace(&plot.eskf_cmp_att, "ESKF pitch [deg]"),
        find_trace(&plot.eskf_cmp_att, "NAV-ATT pitch [deg]"),
        w0,
        w1,
    );
    print_diff_stats(
        "roll err [deg]",
        find_trace(&plot.eskf_cmp_att, "ESKF roll [deg]"),
        find_trace(&plot.eskf_cmp_att, "NAV-ATT roll [deg]"),
        w0,
        w1,
    );

    print_trace_stats(
        "ESKF vehicle accel x [m/s^2]",
        find_trace(&plot.eskf_meas_accel, "ESKF vehicle accel x [m/s^2]"),
        w0,
        w1,
    );
    print_trace_stats(
        "ESKF vehicle accel y [m/s^2]",
        find_trace(&plot.eskf_meas_accel, "ESKF vehicle accel y [m/s^2]"),
        w0,
        w1,
    );
    print_trace_stats(
        "ESKF gyro bias z [deg/s]",
        find_trace(&plot.eskf_bias_gyro, "ESKF gyro bias z [deg/s]"),
        w0,
        w1,
    );
    print_trace_stats(
        "ESKF accel bias x [m/s^2]",
        find_trace(&plot.eskf_bias_accel, "ESKF accel bias x [m/s^2]"),
        w0,
        w1,
    );
    print_trace_stats(
        "ESKF accel bias y [m/s^2]",
        find_trace(&plot.eskf_bias_accel, "ESKF accel bias y [m/s^2]"),
        w0,
        w1,
    );
    Ok(())
}

fn find_trace<'a>(traces: &'a [sim::visualizer::model::Trace], name: &str) -> &'a sim::visualizer::model::Trace {
    traces.iter().find(|t| t.name == name).unwrap()
}

fn print_trace_stats(label: &str, trace: &sim::visualizer::model::Trace, t0: f64, t1: f64) {
    let pts: Vec<f64> = trace
        .points
        .iter()
        .filter(|p| p[0] >= t0 && p[0] <= t1)
        .map(|p| p[1])
        .collect();
    if pts.is_empty() {
        println!("{label}: no samples");
        return;
    }
    let mean = pts.iter().sum::<f64>() / pts.len() as f64;
    let rms = (pts.iter().map(|v| v * v).sum::<f64>() / pts.len() as f64).sqrt();
    let min = pts.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = pts.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let first = pts[0];
    let last = *pts.last().unwrap();
    println!(
        "{label}: n={} mean={:.6} rms={:.6} min={:.6} max={:.6} first={:.6} last={:.6}",
        pts.len(), mean, rms, min, max, first, last
    );
}

fn interp(trace: &sim::visualizer::model::Trace, t: f64) -> Option<f64> {
    let pts = &trace.points;
    if pts.is_empty() {
        return None;
    }
    let idx = pts.partition_point(|p| p[0] < t);
    if idx == 0 {
        return Some(pts[0][1]);
    }
    if idx >= pts.len() {
        return Some(pts.last()?[1]);
    }
    let p0 = pts[idx - 1];
    let p1 = pts[idx];
    let dt = p1[0] - p0[0];
    if dt.abs() < 1.0e-9 {
        return Some(p1[1]);
    }
    let a = (t - p0[0]) / dt;
    Some(p0[1] * (1.0 - a) + p1[1] * a)
}

fn print_diff_stats(
    label: &str,
    a: &sim::visualizer::model::Trace,
    b: &sim::visualizer::model::Trace,
    t0: f64,
    t1: f64,
) {
    let mut diffs = Vec::new();
    for p in &a.points {
        if p[0] < t0 || p[0] > t1 {
            continue;
        }
        if let Some(vb) = interp(b, p[0]) {
            diffs.push(p[1] - vb);
        }
    }
    if diffs.is_empty() {
        println!("{label}: no samples");
        return;
    }
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let rms = (diffs.iter().map(|v| v * v).sum::<f64>() / diffs.len() as f64).sqrt();
    let min = diffs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = diffs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    println!(
        "{label}: n={} mean={:.6} rms={:.6} min={:.6} max={:.6}",
        diffs.len(), mean, rms, min, max
    );
}
