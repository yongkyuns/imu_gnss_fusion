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

fn main() -> Result<()> {
    let args = Args::parse();
    let mut bytes = Vec::new();
    File::open(&args.logfile)
        .with_context(|| format!("failed to open {}", args.logfile.display()))?
        .read_to_end(&mut bytes)
        .context("failed to read log")?;

    let ekf_cfg = EkfCompareConfig {
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
        GnssOutageConfig::default(),
    );

    let eskf = &data.eskf_misalignment;
    let diag = &data.eskf_stationary_diag;

    println!("final_mount_yaw_deg:");
    println!(
        "  comp_A={:.6}",
        trace_last(eskf, "ESKF full mount A yaw [deg]")?
    );
    println!(
        "  comp_B={:.6}",
        trace_last(eskf, "ESKF full mount B yaw [deg]")?
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

    Ok(())
}
