use std::time::Instant;
use std::{fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::visualizer::model::EkfImuSource;
use sim::visualizer::pipeline::build_plot_data;
use sim::visualizer::stats::{
    group_stats, max_gap_sec, max_gap_trace, max_step_abs, trace_stats, trace_time_bounds,
    trace_value_bounds,
};
use sim::visualizer::ui::run_visualizer;

#[derive(Parser, Debug)]
#[command(name = "visualizer")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long)]
    profile_only: bool,
    #[arg(long, default_value = "align", value_parser = parse_ekf_imu_source)]
    ekf_imu_source: EkfImuSource,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    let mut bytes = Vec::new();
    File::open(&args.logfile)
        .with_context(|| format!("failed to open {}", args.logfile.display()))?
        .read_to_end(&mut bytes)
        .context("failed to read log")?;
    let t_read = Instant::now();

    let (data, has_itow) = build_plot_data(&bytes, args.max_records, args.ekf_imu_source);
    let t_build = Instant::now();
    let (n_traces, n_points) = trace_stats(&data);
    let (tmin, tmax) = trace_time_bounds(&data).unwrap_or((f64::NAN, f64::NAN));
    eprintln!(
        "[profile] bytes={} read={:.3}s build={:.3}s total_pre_ui={:.3}s traces={} points={} t_range=[{:.3}, {:.3}]s",
        bytes.len(),
        (t_read - t0).as_secs_f64(),
        (t_build - t_read).as_secs_f64(),
        (t_build - t0).as_secs_f64(),
        n_traces,
        n_points,
        tmin,
        tmax
    );
    for (name, nt, np) in [
        group_stats("speed", &data.speed),
        group_stats("sat_cn0", &data.sat_cn0),
        group_stats("imu_raw_gyro", &data.imu_raw_gyro),
        group_stats("imu_raw_accel", &data.imu_raw_accel),
        group_stats("imu_cal_gyro", &data.imu_cal_gyro),
        group_stats("imu_cal_accel", &data.imu_cal_accel),
        group_stats("esf_ins_gyro", &data.esf_ins_gyro),
        group_stats("esf_ins_accel", &data.esf_ins_accel),
        group_stats("orientation", &data.orientation),
        group_stats("other", &data.other),
        group_stats("ekf_cmp_pos", &data.ekf_cmp_pos),
        group_stats("ekf_cmp_vel", &data.ekf_cmp_vel),
        group_stats("ekf_cmp_att", &data.ekf_cmp_att),
        group_stats("ekf_bias_gyro", &data.ekf_bias_gyro),
        group_stats("ekf_bias_accel", &data.ekf_bias_accel),
        group_stats("ekf_cov_bias", &data.ekf_cov_bias),
        group_stats("ekf_cov_nonbias", &data.ekf_cov_nonbias),
        group_stats("ekf_map", &data.ekf_map),
        group_stats("align_cmp_att", &data.align_cmp_att),
        group_stats("align_res_vel", &data.align_res_vel),
        group_stats("align_axis_err", &data.align_axis_err),
        group_stats("align_motion", &data.align_motion),
        group_stats("align_roll_contrib", &data.align_roll_contrib),
        group_stats("align_pitch_contrib", &data.align_pitch_contrib),
        group_stats("align_yaw_contrib", &data.align_yaw_contrib),
        group_stats("align_cov", &data.align_cov),
    ] {
        eprintln!("[profile] group={} traces={} points={}", name, nt, np);
    }
    eprintln!(
        "[profile] max_gap_s raw_gyro={:.3} raw_accel={:.3} cal_gyro={:.3} cal_accel={:.3}",
        max_gap_sec(&data.imu_raw_gyro),
        max_gap_sec(&data.imu_raw_accel),
        max_gap_sec(&data.imu_cal_gyro),
        max_gap_sec(&data.imu_cal_accel),
    );
    for (group, traces) in [
        ("imu_raw_gyro", &data.imu_raw_gyro),
        ("imu_raw_accel", &data.imu_raw_accel),
        ("imu_cal_gyro", &data.imu_cal_gyro),
        ("imu_cal_accel", &data.imu_cal_accel),
        ("align_cmp_att", &data.align_cmp_att),
        ("align_res_vel", &data.align_res_vel),
        ("align_axis_err", &data.align_axis_err),
        ("align_motion", &data.align_motion),
        ("align_roll_contrib", &data.align_roll_contrib),
        ("align_pitch_contrib", &data.align_pitch_contrib),
        ("align_yaw_contrib", &data.align_yaw_contrib),
        ("align_cov", &data.align_cov),
    ] {
        if let Some((name, gap)) = max_gap_trace(traces) {
            eprintln!(
                "[profile] max_gap_trace group={} signal={} gap_s={:.3}",
                group, name, gap
            );
        }
        if let Some((vmin, vmax)) = trace_value_bounds(traces) {
            eprintln!(
                "[profile] value_range group={} min={:.6} max={:.6}",
                group, vmin, vmax
            );
        }
        if let Some(step) = max_step_abs(traces) {
            eprintln!("[profile] max_step_abs group={} value={:.6}", group, step);
        }
    }
    if args.profile_only {
        return Ok(());
    }

    run_visualizer(data, has_itow)
}

fn parse_ekf_imu_source(s: &str) -> Result<EkfImuSource, String> {
    match s.to_ascii_lowercase().as_str() {
        "align" => Ok(EkfImuSource::Align),
        "esf-alg" | "esf_alg" | "alg" => Ok(EkfImuSource::EsfAlg),
        _ => Err(format!(
            "invalid ekf IMU source '{s}', expected 'align' or 'esf-alg'"
        )),
    }
}
