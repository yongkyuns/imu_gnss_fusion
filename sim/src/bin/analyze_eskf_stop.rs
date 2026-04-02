use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sensor_fusion::ekf::PredictNoise;
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

    #[arg(long)]
    r_body_vel: Option<f32>,

    #[arg(long)]
    gnss_pos_r_scale: Option<f64>,

    #[arg(long)]
    gnss_vel_r_scale: Option<f64>,

    #[arg(long)]
    vehicle_meas_lpf_cutoff_hz: Option<f64>,

    #[arg(long)]
    gyro_var: Option<f32>,

    #[arg(long)]
    accel_var: Option<f32>,

    #[arg(long)]
    gyro_bias_rw_var: Option<f32>,

    #[arg(long)]
    accel_bias_rw_var: Option<f32>,
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
    let predict_noise = if args.gyro_var.is_some()
        || args.accel_var.is_some()
        || args.gyro_bias_rw_var.is_some()
        || args.accel_bias_rw_var.is_some()
    {
        let mut noise = PredictNoise::lsm6dso_typical_104hz();
        if let Some(v) = args.gyro_var {
            noise.gyro_var = v;
        }
        if let Some(v) = args.accel_var {
            noise.accel_var = v;
        }
        if let Some(v) = args.gyro_bias_rw_var {
            noise.gyro_bias_rw_var = v;
        }
        if let Some(v) = args.accel_bias_rw_var {
            noise.accel_bias_rw_var = v;
        }
        Some(noise)
    } else {
        None
    };
    let (plot, _) = build_plot_data(
        &data,
        None,
        ekf_imu_source,
        EkfCompareConfig {
            r_body_vel: args.r_body_vel.unwrap_or(EkfCompareConfig::default().r_body_vel),
            vehicle_meas_lpf_cutoff_hz: args
                .vehicle_meas_lpf_cutoff_hz
                .unwrap_or(EkfCompareConfig::default().vehicle_meas_lpf_cutoff_hz),
            predict_imu_decimation: args.ekf_predict_imu_decimation,
            predict_imu_lpf_cutoff_hz: args.ekf_predict_imu_lpf_cutoff_hz,
            gnss_pos_r_scale: args
                .gnss_pos_r_scale
                .unwrap_or(EkfCompareConfig::default().gnss_pos_r_scale),
            gnss_vel_r_scale: args
                .gnss_vel_r_scale
                .unwrap_or(EkfCompareConfig::default().gnss_vel_r_scale),
            predict_noise,
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
    print_trace_stats(
        "ESKF yaw [deg]",
        find_trace(&plot.eskf_cmp_att, "ESKF yaw [deg]"),
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
    print_body_velocity_stats(
        "ESKF body vel x [m/s]",
        find_trace(&plot.eskf_cmp_vel, "ESKF velN [m/s]"),
        find_trace(&plot.eskf_cmp_vel, "ESKF velE [m/s]"),
        find_trace(&plot.eskf_cmp_att, "ESKF yaw [deg]"),
        w0,
        w1,
        true,
    );
    print_body_velocity_stats(
        "ESKF body vel y [m/s]",
        find_trace(&plot.eskf_cmp_vel, "ESKF velN [m/s]"),
        find_trace(&plot.eskf_cmp_vel, "ESKF velE [m/s]"),
        find_trace(&plot.eskf_cmp_att, "ESKF yaw [deg]"),
        w0,
        w1,
        false,
    );
    print_trace_stats(
        "stationary innov x",
        find_trace(&plot.eskf_stationary_diag, "stationary innov x"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary innov y",
        find_trace(&plot.eskf_stationary_diag, "stationary innov y"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary K theta_x from x",
        find_trace(&plot.eskf_stationary_diag, "stationary K theta_x from x"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary K theta_y from x",
        find_trace(&plot.eskf_stationary_diag, "stationary K theta_y from x"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary K theta_x from y",
        find_trace(&plot.eskf_stationary_diag, "stationary K theta_x from y"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary K theta_y from y",
        find_trace(&plot.eskf_stationary_diag, "stationary K theta_y from y"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary K bax from x",
        find_trace(&plot.eskf_stationary_diag, "stationary K bax from x"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary K bay from y",
        find_trace(&plot.eskf_stationary_diag, "stationary K bay from y"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary P theta_x",
        find_trace(&plot.eskf_stationary_diag, "stationary P theta_x"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary P theta_y",
        find_trace(&plot.eskf_stationary_diag, "stationary P theta_y"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary P bax",
        find_trace(&plot.eskf_stationary_diag, "stationary P bax"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary P bay",
        find_trace(&plot.eskf_stationary_diag, "stationary P bay"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary P theta_x_bax",
        find_trace(&plot.eskf_stationary_diag, "stationary P theta_x_bax"),
        w0,
        w1,
    );
    print_trace_stats(
        "stationary P theta_y_bay",
        find_trace(&plot.eskf_stationary_diag, "stationary P theta_y_bay"),
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

fn print_body_velocity_stats(
    label: &str,
    vn: &sim::visualizer::model::Trace,
    ve: &sim::visualizer::model::Trace,
    yaw_deg: &sim::visualizer::model::Trace,
    t0: f64,
    t1: f64,
    forward: bool,
) {
    let mut vals = Vec::new();
    for p in &vn.points {
        if p[0] < t0 || p[0] > t1 {
            continue;
        }
        let Some(vev) = interp(ve, p[0]) else {
            continue;
        };
        let Some(yaw_deg_v) = interp(yaw_deg, p[0]) else {
            continue;
        };
        let yaw = yaw_deg_v.to_radians();
        let vx = yaw.cos() * p[1] + yaw.sin() * vev;
        let vy = -yaw.sin() * p[1] + yaw.cos() * vev;
        vals.push(if forward { vx } else { vy });
    }
    if vals.is_empty() {
        println!("{label}: no samples");
        return;
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let rms = (vals.iter().map(|v| v * v).sum::<f64>() / vals.len() as f64).sqrt();
    let min = vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let first = vals[0];
    let last = *vals.last().unwrap();
    println!(
        "{label}: n={} mean={:.6} rms={:.6} min={:.6} max={:.6} first={:.6} last={:.6}",
        vals.len(), mean, rms, min, max, first, last
    );
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
