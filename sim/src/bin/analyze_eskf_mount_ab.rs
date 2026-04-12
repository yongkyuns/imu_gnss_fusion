use std::{fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::visualizer::model::{EkfImuSource, Trace};
use sim::visualizer::pipeline::build_plot_data;
use sim::visualizer::pipeline::ekf_compare::{EkfCompareConfig, GnssOutageConfig};

#[derive(Parser, Debug)]
#[command(name = "analyze_eskf_mount_ab")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long, default_value = "align", value_parser = parse_misalignment)]
    misalignment: EkfImuSource,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long)]
    gnss_vel_r_scale: Option<f64>,
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

fn trace_last(traces: &[Trace], name: &str) -> Result<f64> {
    let trace = traces
        .iter()
        .find(|t| t.name == name)
        .with_context(|| format!("missing trace `{name}`"))?;
    let point = trace
        .points
        .last()
        .with_context(|| format!("trace `{name}` has no points"))?;
    Ok(point[1])
}

fn trace_by_name<'a>(traces: &'a [Trace], name: &str) -> Result<&'a Trace> {
    traces
        .iter()
        .find(|t| t.name == name)
        .with_context(|| format!("missing trace `{name}`"))
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

fn print_trace_summary(group: &str, trace: &Trace) {
    let Some(first) = trace.points.first() else {
        println!("{group},\"{}\",empty", trace.name);
        return;
    };
    let mut min_v = first[1];
    let mut max_v = first[1];
    let mut max_raw_step = 0.0f64;
    let mut max_wrap_step = 0.0f64;
    let mut unwrapped = first[1];
    let mut min_unwrapped = unwrapped;
    let mut max_unwrapped = unwrapped;
    let mut prev = first[1];
    for point in trace.points.iter().skip(1) {
        let v = point[1];
        min_v = min_v.min(v);
        max_v = max_v.max(v);
        max_raw_step = max_raw_step.max((v - prev).abs());
        let wrapped_step = wrap_deg180(v - prev);
        max_wrap_step = max_wrap_step.max(wrapped_step.abs());
        unwrapped += wrapped_step;
        min_unwrapped = min_unwrapped.min(unwrapped);
        max_unwrapped = max_unwrapped.max(unwrapped);
        prev = v;
    }
    let last = trace.points.last().unwrap();
    println!(
        "{group},\"{}\",n={},t0={:.3},t1={:.3},last={:.6},min={:.6},max={:.6},unwrapped_span={:.6},max_raw_step={:.6},max_wrapped_step={:.6}",
        trace.name,
        trace.points.len(),
        first[0],
        last[0],
        last[1],
        min_v,
        max_v,
        max_unwrapped - min_unwrapped,
        max_raw_step,
        max_wrap_step,
    );
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
        gnss_vel_r_scale: args
            .gnss_vel_r_scale
            .unwrap_or(EkfCompareConfig::default().gnss_vel_r_scale),
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

    let eskf = &data.eskf_misalignment;
    let diag = &data.eskf_stationary_diag;

    println!(
        "config: misalignment={:?} decimation={} lpf_hz={} gnss_vel_r_scale={:.3} outage_count={} outage_duration_s={:.3} outage_seed={}",
        args.misalignment,
        ekf_cfg.predict_imu_decimation,
        ekf_cfg
            .predict_imu_lpf_cutoff_hz
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "off".to_string()),
        ekf_cfg.gnss_vel_r_scale,
        args.gnss_outage_count,
        args.gnss_outage_duration_s,
        args.gnss_outage_seed,
    );

    println!("final_mount_yaw_deg:");
    println!(
        "  full_mount={:.6}",
        trace_last(eskf, "ESKF full mount yaw [deg]")?
    );
    println!(
        "  esf_alg_ref={:.6}",
        trace_last(eskf, "ESF-ALG mount yaw [deg]")?
    );

    for cue in ["body_vel_y", "body_vel_z"] {
        println!("cue={cue}");
        println!(
            "  yaw_dx_sum_deg={:.6}",
            trace_last(diag, &format!("mount yaw dx sum {cue} [deg]"))?
        );
        println!(
            "  yaw_dx_abs_deg={:.6}",
            trace_last(diag, &format!("mount yaw dx abs {cue} [deg]"))?
        );
        println!(
            "  innov_sum={:.6}",
            trace_last(diag, &format!("innovation sum {cue}"))?
        );
        println!(
            "  innov_abs={:.6}",
            trace_last(diag, &format!("innovation abs {cue}"))?
        );
    }

    println!("trace_summary:");
    for name in [
        "ESKF full mount roll [deg]",
        "ESKF full mount pitch [deg]",
        "ESKF full mount yaw [deg]",
        "ESF-ALG mount roll [deg]",
        "ESF-ALG mount pitch [deg]",
        "ESF-ALG mount yaw [deg]",
    ] {
        print_trace_summary("eskf_misalignment", trace_by_name(eskf, name)?);
    }
    for name in [
        "Loose full mount roll [deg]",
        "Loose full mount pitch [deg]",
        "Loose full mount yaw [deg]",
        "ESF-ALG mount roll [deg]",
        "ESF-ALG mount pitch [deg]",
        "ESF-ALG mount yaw [deg]",
    ] {
        print_trace_summary(
            "loose_misalignment",
            trace_by_name(&data.loose_misalignment, name)?,
        );
    }
    for name in [
        "ESKF roll [deg]",
        "ESKF pitch [deg]",
        "ESKF yaw [deg]",
        "NAV-ATT roll [deg]",
        "NAV-ATT pitch [deg]",
        "NAV-ATT heading [deg]",
    ] {
        print_trace_summary("eskf_cmp_att", trace_by_name(&data.eskf_cmp_att, name)?);
    }

    Ok(())
}
