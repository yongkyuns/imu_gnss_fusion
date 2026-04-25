use std::{fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::visualizer::model::{EkfImuSource, Trace};
use sim::visualizer::pipeline::build_plot_data;
use sim::visualizer::pipeline::ekf_compare::{EkfCompareConfig, GnssOutageConfig};

#[derive(Parser, Debug)]
#[command(name = "analyze_interval_drift")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long, default_value = "align", value_parser = parse_misalignment)]
    misalignment: EkfImuSource,
    #[arg(long, default_value_t = 353.0)]
    start_s: f64,
    #[arg(long, default_value_t = 534.0)]
    end_s: f64,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long, default_value_t = 0.3)]
    gnss_pos_r_scale: f64,
    #[arg(long, default_value_t = 3.0)]
    gnss_vel_r_scale: f64,
    #[arg(long, default_value_t = 0.0)]
    gnss_pos_mount_scale: f32,
    #[arg(long, default_value_t = 0.0)]
    gnss_vel_mount_scale: f32,
    #[arg(long, default_value_t = 2.0)]
    r_body_vel: f32,
    #[arg(long, default_value_t = 0.125)]
    gyro_bias_init_sigma_dps: f32,
    #[arg(long, default_value_t = 0)]
    gnss_outage_count: usize,
    #[arg(long, default_value_t = 0.0)]
    gnss_outage_duration_s: f64,
    #[arg(long, default_value_t = 1)]
    gnss_outage_seed: u64,
    #[arg(long, default_value_t = 1)]
    ekf_predict_imu_decimation: usize,
    #[arg(long)]
    ekf_predict_imu_lpf_cutoff_hz: Option<f64>,
}

fn parse_misalignment(s: &str) -> Result<EkfImuSource, String> {
    match s.to_ascii_lowercase().as_str() {
        "align" | "auto" => Ok(EkfImuSource::Align),
        "esf-alg" | "esf_alg" | "esfalg" => Ok(EkfImuSource::EsfAlg),
        other => Err(format!("invalid misalignment source: {other}")),
    }
}

fn trace_by_name<'a>(traces: &'a [Trace], name: &str) -> Result<&'a Trace> {
    traces
        .iter()
        .find(|t| t.name == name)
        .with_context(|| format!("missing trace `{name}`"))
}

fn angle_wrap_deg(mut deg: f64) -> f64 {
    while deg > 180.0 {
        deg -= 360.0;
    }
    while deg <= -180.0 {
        deg += 360.0;
    }
    deg
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

fn mean_in_window(trace: &Trace, start_s: f64, end_s: f64, is_angle: bool) -> Option<f64> {
    let mut values = Vec::new();
    for point in &trace.points {
        if point[0] >= start_s && point[0] <= end_s {
            values.push(point[1]);
        }
    }
    if values.is_empty() {
        return None;
    }
    if !is_angle {
        return Some(values.iter().sum::<f64>() / values.len() as f64);
    }
    let ref_v = values[0];
    let mut accum = 0.0;
    for value in values {
        accum += angle_wrap_deg(value - ref_v);
    }
    Some(
        ref_v
            + accum
                / trace
                    .points
                    .iter()
                    .filter(|p| p[0] >= start_s && p[0] <= end_s)
                    .count() as f64,
    )
}

fn drift_in_window(trace: &Trace, start_s: f64, end_s: f64, is_angle: bool) -> Option<f64> {
    let start_v = sample_trace(trace, start_s)?;
    let end_v = sample_trace(trace, end_s)?;
    Some(if is_angle {
        angle_wrap_deg(end_v - start_v)
    } else {
        end_v - start_v
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut bytes = Vec::new();
    File::open(&args.logfile)
        .with_context(|| format!("failed to open {}", args.logfile.display()))?
        .read_to_end(&mut bytes)
        .context("failed to read log")?;

    let ekf_cfg = EkfCompareConfig {
        r_body_vel: args.r_body_vel,
        gnss_pos_mount_scale: args.gnss_pos_mount_scale,
        gnss_vel_mount_scale: args.gnss_vel_mount_scale,
        gyro_bias_init_sigma_dps: args.gyro_bias_init_sigma_dps,
        gnss_pos_r_scale: args.gnss_pos_r_scale,
        gnss_vel_r_scale: args.gnss_vel_r_scale,
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

    let eskf_yaw = trace_by_name(&data.eskf_cmp_att, "ESKF yaw [deg]")?;
    let nav_yaw = trace_by_name(&data.eskf_cmp_att, "NAV-ATT heading [deg]")?;
    let eskf_lat = trace_by_name(&data.eskf_cmp_vel, "ESKF lateral vel [m/s]")?;
    let ubx_lat = trace_by_name(&data.eskf_cmp_vel, "u-blox lateral vel [m/s]")?;
    let bgx = trace_by_name(&data.eskf_bias_gyro, "ESKF gyro bias x [deg/s]")?;
    let bgy = trace_by_name(&data.eskf_bias_gyro, "ESKF gyro bias y [deg/s]")?;
    let bgz = trace_by_name(&data.eskf_bias_gyro, "ESKF gyro bias z [deg/s]")?;

    let yaw_err = Trace {
        name: "yaw_err".to_string(),
        points: eskf_yaw
            .points
            .iter()
            .filter_map(|p| {
                sample_trace(nav_yaw, p[0]).map(|ref_yaw| [p[0], angle_wrap_deg(p[1] - ref_yaw)])
            })
            .collect(),
    };
    let lat_err = Trace {
        name: "lat_err".to_string(),
        points: eskf_lat
            .points
            .iter()
            .filter_map(|p| sample_trace(ubx_lat, p[0]).map(|ref_lat| [p[0], p[1] - ref_lat]))
            .collect(),
    };

    println!(
        "window=[{:.1}, {:.1}] config: gnss_pos_r_scale={:.3} gnss_vel_r_scale={:.3} gnss_pos_mount_scale={:.3} gnss_vel_mount_scale={:.3} r_body_vel={:.1} gyro_bias_init_sigma_dps={:.3}",
        args.start_s,
        args.end_s,
        args.gnss_pos_r_scale,
        args.gnss_vel_r_scale,
        args.gnss_pos_mount_scale,
        args.gnss_vel_mount_scale,
        args.r_body_vel,
        args.gyro_bias_init_sigma_dps,
    );

    for (name, trace, is_angle) in [
        ("yaw_err_deg", &yaw_err, true),
        ("lat_vel_err_mps", &lat_err, false),
        ("gyro_bias_x_dps", bgx, false),
        ("gyro_bias_y_dps", bgy, false),
        ("gyro_bias_z_dps", bgz, false),
    ] {
        let start_v = sample_trace(trace, args.start_s).unwrap_or(f64::NAN);
        let end_v = sample_trace(trace, args.end_s).unwrap_or(f64::NAN);
        let mean_v = mean_in_window(trace, args.start_s, args.end_s, is_angle).unwrap_or(f64::NAN);
        let drift_v =
            drift_in_window(trace, args.start_s, args.end_s, is_angle).unwrap_or(f64::NAN);
        println!(
            "{}: start={:.6} end={:.6} drift={:.6} mean={:.6}",
            name, start_v, end_v, drift_v, mean_v
        );
    }

    Ok(())
}
