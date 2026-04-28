use std::{fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::visualizer::{
    model::{EkfImuSource, Trace},
    pipeline::{
        build_plot_data,
        ekf_compare::{EkfCompareConfig, GnssOutageConfig},
    },
};

#[derive(Parser, Debug)]
#[command(name = "analyze_loose_bias_drift")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long, default_value = "external", value_parser = parse_misalignment)]
    misalignment: EkfImuSource,
    #[arg(long)]
    gnss_pos_r_scale: Option<f64>,
    #[arg(long)]
    gnss_vel_r_scale: Option<f64>,
    #[arg(long)]
    ekf_predict_imu_lpf_cutoff_hz: Option<f64>,
    #[arg(long, default_value_t = 1)]
    ekf_predict_imu_decimation: usize,
    #[arg(long)]
    r_body_vel: Option<f32>,
    #[arg(long)]
    mount_align_rw_var: Option<f32>,
    #[arg(long)]
    mount_update_min_scale: Option<f32>,
    #[arg(long)]
    mount_update_ramp_time_s: Option<f32>,
    #[arg(long)]
    mount_update_innovation_gate_mps: Option<f32>,
    #[arg(long, default_value = "76.236,117.236,180,225.31,320,450,534")]
    sample_times_s: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut bytes = Vec::new();
    File::open(&args.logfile)
        .with_context(|| format!("failed to open {}", args.logfile.display()))?
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {}", args.logfile.display()))?;

    let defaults = EkfCompareConfig::default();
    let cfg = EkfCompareConfig {
        gnss_pos_r_scale: args.gnss_pos_r_scale.unwrap_or(defaults.gnss_pos_r_scale),
        gnss_vel_r_scale: args.gnss_vel_r_scale.unwrap_or(defaults.gnss_vel_r_scale),
        predict_imu_lpf_cutoff_hz: args.ekf_predict_imu_lpf_cutoff_hz,
        predict_imu_decimation: args.ekf_predict_imu_decimation.max(1),
        r_body_vel: args.r_body_vel.unwrap_or(defaults.r_body_vel),
        mount_align_rw_var: args
            .mount_align_rw_var
            .unwrap_or(defaults.mount_align_rw_var),
        mount_update_min_scale: args
            .mount_update_min_scale
            .unwrap_or(defaults.mount_update_min_scale),
        mount_update_ramp_time_s: args
            .mount_update_ramp_time_s
            .unwrap_or(defaults.mount_update_ramp_time_s),
        mount_update_innovation_gate_mps: args
            .mount_update_innovation_gate_mps
            .unwrap_or(defaults.mount_update_innovation_gate_mps),
        ..defaults
    };
    let data = build_plot_data(
        &bytes,
        None,
        args.misalignment,
        cfg,
        GnssOutageConfig::default(),
    )
    .0;
    let sample_times = parse_sample_times(&args.sample_times_s);

    println!(
        "config: mode={:?} gnss_pos_r_scale={:.3} gnss_vel_r_scale={:.3} predict_lpf={} predict_decimation={} r_body_vel={:.6} mount_align_rw_var={:.6e} mount_update_min_scale={:.6} mount_update_ramp_time_s={:.3} mount_update_innovation_gate_mps={:.3}",
        args.misalignment,
        cfg.gnss_pos_r_scale,
        cfg.gnss_vel_r_scale,
        cfg.predict_imu_lpf_cutoff_hz
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "off".to_string()),
        cfg.predict_imu_decimation,
        cfg.r_body_vel,
        cfg.mount_align_rw_var,
        cfg.mount_update_min_scale,
        cfg.mount_update_ramp_time_s,
        cfg.mount_update_innovation_gate_mps,
    );

    print_group("loose_bias_accel", &data.loose_bias_accel, &sample_times);
    print_group("loose_bias_gyro", &data.loose_bias_gyro, &sample_times);
    print_group("loose_scale_accel", &data.loose_scale_accel, &sample_times);
    print_group("loose_meas_accel", &data.loose_meas_accel, &sample_times);
    print_group("loose_cmp_vel", &data.loose_cmp_vel, &sample_times);
    print_group("loose_cmp_att", &data.loose_cmp_att, &sample_times);
    print_group("loose_nominal_att", &data.loose_nominal_att, &sample_times);
    print_group(
        "loose_residual_mount",
        &data.loose_residual_mount,
        &sample_times,
    );
    print_group("loose_cov_nonbias", &data.loose_cov_nonbias, &sample_times);
    print_group("loose_cmp_pos", &data.loose_cmp_pos, &sample_times);
    print_group(
        "loose_misalignment",
        &data.loose_misalignment,
        &sample_times,
    );
    print_group("eskf_misalignment", &data.eskf_misalignment, &sample_times);

    Ok(())
}

fn parse_misalignment(s: &str) -> Result<EkfImuSource, String> {
    EkfImuSource::from_cli_value(s)
}

fn parse_sample_times(s: &str) -> Vec<f64> {
    s.split(',')
        .filter_map(|part| part.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .collect()
}

fn print_group(name: &str, traces: &[Trace], sample_times: &[f64]) {
    println!("group={name} traces={}", traces.len());
    for trace in traces {
        print_summary(name, trace);
        for target_t in sample_times {
            if let Some((t, v)) = nearest(trace, *target_t) {
                println!(
                    "sample\tgroup={name}\ttrace={}\ttarget_s={:.3}\tt_s={:.3}\tvalue={:.9}",
                    trace.name, target_t, t, v
                );
            }
        }
    }
}

fn print_summary(group: &str, trace: &Trace) {
    if trace.points.is_empty() {
        println!("summary\tgroup={group}\ttrace={}\tn=0", trace.name);
        return;
    }
    let first = trace.points[0];
    let last = trace.points[trace.points.len() - 1];
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for point in &trace.points {
        let v = point[1];
        sum += v;
        sum_sq += v * v;
        min = min.min(v);
        max = max.max(v);
    }
    let n = trace.points.len() as f64;
    println!(
        "summary\tgroup={group}\ttrace={}\tn={}\tstart_t={:.3}\tstart={:.9}\tend_t={:.3}\tend={:.9}\tdrift={:.9}\tmean={:.9}\trms={:.9}\tmin={:.9}\tmax={:.9}",
        trace.name,
        trace.points.len(),
        first[0],
        first[1],
        last[0],
        last[1],
        last[1] - first[1],
        sum / n,
        (sum_sq / n).sqrt(),
        min,
        max
    );
}

fn nearest(trace: &Trace, target_t: f64) -> Option<(f64, f64)> {
    trace
        .points
        .iter()
        .min_by(|a, b| (a[0] - target_t).abs().total_cmp(&(b[0] - target_t).abs()))
        .map(|p| (p[0], p[1]))
}
