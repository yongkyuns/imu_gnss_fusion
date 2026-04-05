use std::time::Instant;
use std::{fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::visualizer::model::EkfImuSource;
use sim::visualizer::pipeline::build_plot_data;
use sim::visualizer::pipeline::ekf_compare::{EkfCompareConfig, GnssOutageConfig};
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
    #[arg(long)]
    dump_align_axis_time_s: Option<f64>,
    #[arg(long, default_value_t = 3.0)]
    dump_window_s: f64,
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

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    let mut bytes = Vec::new();
    File::open(&args.logfile)
        .with_context(|| format!("failed to open {}", args.logfile.display()))?
        .read_to_end(&mut bytes)
        .context("failed to read log")?;
    let t_read = Instant::now();

    let ekf_cfg = EkfCompareConfig {
        predict_imu_decimation: args.ekf_predict_imu_decimation.max(1),
        predict_imu_lpf_cutoff_hz: args.ekf_predict_imu_lpf_cutoff_hz,
        ..EkfCompareConfig::default()
    };

    let (data, has_itow) = build_plot_data(
        &bytes,
        args.max_records,
        args.ekf_imu_source,
        ekf_cfg,
        GnssOutageConfig {
            count: args.gnss_outage_count,
            duration_s: args.gnss_outage_duration_s,
            seed: args.gnss_outage_seed,
        },
    );
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
    eprintln!(
        "[profile] ekf-only predict_imu_decimation={} ekf-only predict_imu_lpf_cutoff_hz={}",
        ekf_cfg.predict_imu_decimation,
        ekf_cfg
            .predict_imu_lpf_cutoff_hz
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "off".to_string())
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
        group_stats("eskf_cmp_pos", &data.eskf_cmp_pos),
        group_stats("eskf_cmp_vel", &data.eskf_cmp_vel),
        group_stats("eskf_cmp_att", &data.eskf_cmp_att),
        group_stats("eskf_meas_gyro", &data.eskf_meas_gyro),
        group_stats("eskf_meas_accel", &data.eskf_meas_accel),
        group_stats("eskf_bias_gyro", &data.eskf_bias_gyro),
        group_stats("eskf_bias_accel", &data.eskf_bias_accel),
        group_stats("eskf_cov_bias", &data.eskf_cov_bias),
        group_stats("eskf_cov_nonbias", &data.eskf_cov_nonbias),
        group_stats("eskf_stationary_diag", &data.eskf_stationary_diag),
        group_stats("eskf_bump_pitch_speed", &data.eskf_bump_pitch_speed),
        group_stats("eskf_bump_diag", &data.eskf_bump_diag),
        group_stats("eskf_map", &data.eskf_map),
        group_stats("loose_cmp_pos", &data.loose_cmp_pos),
        group_stats("loose_cmp_vel", &data.loose_cmp_vel),
        group_stats("loose_cmp_att", &data.loose_cmp_att),
        group_stats("loose_meas_gyro", &data.loose_meas_gyro),
        group_stats("loose_meas_accel", &data.loose_meas_accel),
        group_stats("loose_bias_gyro", &data.loose_bias_gyro),
        group_stats("loose_bias_accel", &data.loose_bias_accel),
        group_stats("loose_scale_gyro", &data.loose_scale_gyro),
        group_stats("loose_scale_accel", &data.loose_scale_accel),
        group_stats("loose_cov_bias", &data.loose_cov_bias),
        group_stats("loose_cov_nonbias", &data.loose_cov_nonbias),
        group_stats("loose_map", &data.loose_map),
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
        ("eskf_meas_gyro", &data.eskf_meas_gyro),
        ("eskf_meas_accel", &data.eskf_meas_accel),
        ("eskf_bump_pitch_speed", &data.eskf_bump_pitch_speed),
        ("eskf_bump_diag", &data.eskf_bump_diag),
        ("loose_meas_gyro", &data.loose_meas_gyro),
        ("loose_meas_accel", &data.loose_meas_accel),
        ("loose_scale_gyro", &data.loose_scale_gyro),
        ("loose_scale_accel", &data.loose_scale_accel),
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
    if let Some(t_s) = args.dump_align_axis_time_s {
        dump_traces_near_time(
            "align_cmp_att",
            &data.align_cmp_att,
            t_s,
            args.dump_window_s,
        );
        dump_traces_near_time(
            "align_axis_err",
            &data.align_axis_err,
            t_s,
            args.dump_window_s,
        );
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

fn dump_traces_near_time(
    group: &str,
    traces: &[sim::visualizer::model::Trace],
    t_s: f64,
    window_s: f64,
) {
    let half = 0.5 * window_s.abs();
    eprintln!(
        "[dump] group={} center_t_s={:.3} window_s={:.3}",
        group, t_s, window_s
    );
    for trace in traces {
        let mut any = false;
        for p in &trace.points {
            if (p[0] - t_s).abs() <= half {
                if !any {
                    eprintln!("[dump] trace={}", trace.name);
                    any = true;
                }
                eprintln!("[dump]   t_s={:.3} value={:.6}", p[0], p[1]);
            }
        }
        if !any {
            eprintln!("[dump] trace={} no points in window", trace.name);
        }
    }
}
