#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use fusion_tools::datasets::generic_replay::{
    load_gnss_samples, load_imu_samples, load_reference_attitude_samples,
    load_reference_motion_samples, load_reference_mount_samples, load_reference_position_samples,
};
use fusion_tools::eval::gnss_ins::wrap_deg180;
use fusion_tools::eval::trace::sample_nearest_value;
use fusion_tools::visualizer::model::{PlotData, Trace, VisualizerMountMode};
use fusion_tools::visualizer::pipeline::FusionTuningConfig;
use fusion_tools::visualizer::pipeline::generic::GenericReplayInput;
use fusion_tools::visualizer::replay_job::{GenericReplayJobConfig, run_generic_replay_job};

#[derive(Parser, Debug)]
#[command(name = "compare_c_rust_replay")]
struct Args {
    #[arg(long, value_name = "DIR")]
    generic_replay_dir: PathBuf,
    #[arg(long, default_value = "auto", value_parser = parse_misalignment)]
    misalignment: VisualizerMountMode,
    #[arg(long, value_enum, default_value_t = TraceSet::Core)]
    trace_set: TraceSet,
    #[arg(long, default_value_t = 0.05)]
    warn_abs: f64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TraceSet {
    Core,
    Ekf,
}

struct TraceDiff {
    name: &'static str,
    samples: usize,
    final_abs: f64,
    rms: f64,
    max_abs: f64,
    first_warn: Option<(f64, f64, f64, f64)>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let replay = load_replay(&args.generic_replay_dir)?;
    let cfg = GenericReplayJobConfig::complete(
        args.misalignment,
        FusionTuningConfig::default(),
        Default::default(),
    );
    let rust = run_generic_replay_job(
        &replay,
        GenericReplayJobConfig {
            backend: fusion_tools::visualizer::model::VisualizerFusionBackend::Rust,
            ..cfg
        },
    );
    let c = run_generic_replay_job(
        &replay,
        GenericReplayJobConfig {
            backend: fusion_tools::visualizer::model::VisualizerFusionBackend::C,
            ..cfg
        },
    );

    let names = match args.trace_set {
        TraceSet::Core => CORE_TRACE_NAMES,
        TraceSet::Ekf => EKF_TRACE_NAMES,
    };

    println!(
        "trace,samples,final_abs,rms,max_abs,first_warn_t_s,first_warn_rust,first_warn_c,first_warn_abs"
    );
    for name in names {
        let Some(diff) = compare_named_trace(&rust, &c, name, args.warn_abs) else {
            println!("{name},missing,NaN,NaN,NaN,NaN,NaN,NaN,NaN");
            continue;
        };
        let (first_t, first_rust, first_c, first_abs) =
            diff.first_warn
                .unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN));
        println!(
            "{},{},{:.9},{:.9},{:.9},{:.6},{:.9},{:.9},{:.9}",
            diff.name,
            diff.samples,
            diff.final_abs,
            diff.rms,
            diff.max_abs,
            first_t,
            first_rust,
            first_c,
            first_abs
        );
    }

    Ok(())
}

fn load_replay(dir: &PathBuf) -> Result<GenericReplayInput> {
    Ok(GenericReplayInput {
        imu: load_imu_samples(dir)
            .with_context(|| format!("loading imu from {}", dir.display()))?,
        gnss: load_gnss_samples(dir)
            .with_context(|| format!("loading gnss from {}", dir.display()))?,
        reference_attitude: load_reference_attitude_samples(dir)?,
        reference_mount: load_reference_mount_samples(dir)?,
        reference_position: load_reference_position_samples(dir)?,
        reference_motion: load_reference_motion_samples(dir)?,
    })
}

fn compare_named_trace(
    rust: &PlotData,
    c: &PlotData,
    name: &'static str,
    warn_abs: f64,
) -> Option<TraceDiff> {
    let rust_trace = rust.trace_by_name(name)?;
    let c_trace = c.trace_by_name(name)?;
    Some(compare_trace(rust_trace, c_trace, name, warn_abs))
}

fn compare_trace(rust: &Trace, c: &Trace, name: &'static str, warn_abs: f64) -> TraceDiff {
    let mut samples = 0usize;
    let mut sum_sq = 0.0;
    let mut max_abs = 0.0_f64;
    let mut final_abs = f64::NAN;
    let mut first_warn = None;
    for &[t_s, rust_value] in &rust.points {
        if !t_s.is_finite() || !rust_value.is_finite() {
            continue;
        }
        let Some(c_value) = sample_nearest_value(c, t_s).filter(|v| v.is_finite()) else {
            continue;
        };
        let diff = trace_abs_diff(name, rust_value, c_value);
        samples += 1;
        sum_sq += diff * diff;
        max_abs = max_abs.max(diff);
        final_abs = diff;
        if first_warn.is_none() && diff > warn_abs {
            first_warn = Some((t_s, rust_value, c_value, diff));
        }
    }
    TraceDiff {
        name,
        samples,
        final_abs,
        rms: if samples > 0 {
            (sum_sq / samples as f64).sqrt()
        } else {
            f64::NAN
        },
        max_abs,
        first_warn,
    }
}

fn trace_abs_diff(name: &str, rust_value: f64, c_value: f64) -> f64 {
    if name.to_ascii_lowercase().contains("yaw") {
        wrap_deg180(rust_value - c_value).abs()
    } else {
        (rust_value - c_value).abs()
    }
}

fn parse_misalignment(value: &str) -> Result<VisualizerMountMode, String> {
    VisualizerMountMode::from_cli_value(value)
}

const CORE_TRACE_NAMES: &[&str] = &[
    "EKF posN [m]",
    "EKF posE [m]",
    "EKF velN [m/s]",
    "EKF velE [m/s]",
    "EKF roll [deg]",
    "EKF pitch [deg]",
    "EKF yaw [deg]",
    "EKF mount roll [deg]",
    "EKF mount pitch [deg]",
    "EKF mount yaw [deg]",
];

const EKF_TRACE_NAMES: &[&str] = &[
    "EKF posN [m]",
    "EKF posE [m]",
    "EKF posD [m]",
    "EKF velN [m/s]",
    "EKF velE [m/s]",
    "EKF velD [m/s]",
    "EKF roll [deg]",
    "EKF pitch [deg]",
    "EKF yaw [deg]",
    "EKF mount roll [deg]",
    "EKF mount pitch [deg]",
    "EKF mount yaw [deg]",
    "EKF gyro bias X [deg/s]",
    "EKF gyro bias Y [deg/s]",
    "EKF gyro bias Z [deg/s]",
    "EKF accel bias X [m/s^2]",
    "EKF accel bias Y [m/s^2]",
    "EKF accel bias Z [m/s^2]",
    "EKF attitude roll sigma [deg]",
    "EKF attitude pitch sigma [deg]",
    "EKF attitude yaw sigma [deg]",
    "EKF mount roll sigma [deg]",
    "EKF mount pitch sigma [deg]",
    "EKF mount yaw sigma [deg]",
];
