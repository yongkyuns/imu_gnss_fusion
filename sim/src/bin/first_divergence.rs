use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use sim::eval::first_divergence::{Options, Report, run_generic_replay};

#[derive(Debug, Parser)]
#[command(
    about = "Replay generic CSV data through public SensorFusion APIs and summarize first divergence"
)]
struct Args {
    /// Generic replay directory containing imu.csv and gnss.csv.
    #[arg(long)]
    generic_replay_dir: PathBuf,
    /// Mount quaternion error threshold in degrees.
    #[arg(long, default_value_t = 2.0)]
    mount_threshold_deg: f64,
    /// Vehicle attitude quaternion error threshold in degrees.
    #[arg(long, default_value_t = 2.0)]
    attitude_threshold_deg: f64,
    /// Ignore threshold crossings before this timestamp.
    #[arg(long, default_value_t = 0.0)]
    start_after_s: f64,
    /// Half-width of the allocation summary window around first crossing.
    #[arg(long, default_value_t = 10.0)]
    window_s: f64,
    /// Snapshot period for error checks.
    #[arg(long, default_value_t = 0.5)]
    sample_period_s: f64,
    /// Stop replay after this timestamp.
    #[arg(long)]
    max_time_s: Option<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let report = run_generic_replay(
        &args.generic_replay_dir,
        Options {
            mount_threshold_deg: args.mount_threshold_deg,
            attitude_threshold_deg: args.attitude_threshold_deg,
            start_after_s: args.start_after_s,
            window_s: args.window_s,
            sample_period_s: args.sample_period_s,
            max_time_s: args.max_time_s,
        },
    )?;
    print_report(&report);
    Ok(())
}

fn print_report(report: &Report) {
    println!("input: {}", report.input);
    println!(
        "samples: imu={} gnss={} ref_att={} ref_mount={} ref_pos={}",
        report.samples.imu,
        report.samples.gnss,
        report.samples.reference_attitude,
        report.samples.reference_mount,
        report.samples.reference_position
    );
    println!(
        "ready: align={} reduced={} full={}",
        fmt_time(report.align_ready_t_s),
        fmt_time(report.reduced_init_t_s),
        fmt_time(report.full_init_t_s)
    );
    match &report.first_crossing {
        Some(crossing) => println!(
            "first crossing: t={:.3}s source={} metric={} value={:.3}deg threshold={:.3}deg",
            crossing.t_s,
            crossing.source,
            crossing.metric,
            crossing.value_deg,
            crossing.threshold_deg
        ),
        None => println!("first crossing: none"),
    }
    if !report.first_crossings.is_empty() {
        println!("first crossings by source/metric:");
        for crossing in &report.first_crossings {
            println!(
                "  {:7} {:14} t={:.3}s value={:.3}deg threshold={:.3}deg",
                crossing.source,
                crossing.metric,
                crossing.t_s,
                crossing.value_deg,
                crossing.threshold_deg
            );
        }
    }
    println!("final errors:");
    for snapshot in &report.final_errors {
        let mount_axis = snapshot
            .mount_axis_err_deg
            .map(|v| format!("[{:.3}, {:.3}, {:.3}]", v[0], v[1], v[2]))
            .unwrap_or_else(|| "n/a".to_string());
        let mount_sigma = snapshot
            .mount_sigma_deg
            .map(|v| format!("[{:.3}, {:.3}, {:.3}]", v[0], v[1], v[2]))
            .unwrap_or_else(|| "n/a".to_string());
        let att_sigma = snapshot
            .attitude_sigma_deg
            .map(|v| format!("[{:.3}, {:.3}, {:.3}]", v[0], v[1], v[2]))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "  {:7} t={:.3}s mount_qerr={}deg mount_axis={}deg att_qerr={}deg mount_sigma={}deg att_sigma={}deg",
            snapshot.source,
            snapshot.t_s,
            fmt_opt(snapshot.mount_qerr_deg),
            mount_axis,
            fmt_opt(snapshot.attitude_qerr_deg),
            mount_sigma,
            att_sigma
        );
    }
    if report.window_summaries.is_empty() {
        return;
    }
    println!("allocation window around first crossing:");
    for summary in &report.window_summaries {
        println!(
            "  {:7} {:12} n={:<4} mount_sum=[{:+.4},{:+.4},{:+.4}]deg mount_abs=[{:.4},{:.4},{:.4}]deg att_abs=[{:.4},{:.4},{:.4}]deg accel_bias_abs=[{:.3e},{:.3e},{:.3e}] gyro_bias_abs=[{:.3e},{:.3e},{:.3e}] residual_abs={:.3} mean_nis={:.3} max_nis={:.3}",
            summary.source,
            summary.update,
            summary.count,
            summary.sum_mount_dx_deg[0],
            summary.sum_mount_dx_deg[1],
            summary.sum_mount_dx_deg[2],
            summary.sum_abs_mount_dx_deg[0],
            summary.sum_abs_mount_dx_deg[1],
            summary.sum_abs_mount_dx_deg[2],
            summary.sum_abs_att_dx_deg[0],
            summary.sum_abs_att_dx_deg[1],
            summary.sum_abs_att_dx_deg[2],
            summary.sum_abs_accel_bias_dx[0],
            summary.sum_abs_accel_bias_dx[1],
            summary.sum_abs_accel_bias_dx[2],
            summary.sum_abs_gyro_bias_dx[0],
            summary.sum_abs_gyro_bias_dx[1],
            summary.sum_abs_gyro_bias_dx[2],
            summary.sum_abs_residual,
            summary.mean_nis(),
            summary.max_nis
        );
    }
}

fn fmt_time(t_s: Option<f64>) -> String {
    t_s.map(|t| format!("{t:.3}s"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "n/a".to_string())
}
