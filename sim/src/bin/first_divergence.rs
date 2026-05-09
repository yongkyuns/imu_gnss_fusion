use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use sim::eval::first_divergence::{BehaviorSample, Options, Report, run_generic_replay};

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
    /// Optional CSV path for time-aligned reference/filter behavior diagnostics.
    #[arg(long)]
    behavior_csv: Option<PathBuf>,
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
    if let Some(path) = args.behavior_csv {
        write_behavior_csv(&path, &report.behavior_samples)
            .with_context(|| format!("writing {}", path.display()))?;
        println!(
            "behavior csv: {} rows={}",
            path.display(),
            report.behavior_samples.len()
        );
    }
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

fn write_behavior_csv(path: &PathBuf, samples: &[BehaviorSample]) -> Result<()> {
    let mut out = String::new();
    out.push_str(
        "t_s,interval_s,motion_regime,gnss_speed_mps,gnss_course_rate_dps,gnss_speed_rate_mps2,imu_gyro_norm_dps,imu_gyro_z_dps,imu_accel_norm_err_mps2,\
ref_mount_roll_deg,ref_mount_pitch_deg,ref_mount_yaw_deg,ref_mount_delta_roll_deg,ref_mount_delta_pitch_deg,ref_mount_delta_yaw_deg,ref_mount_delta_q_deg,ref_mount_delta_qx_deg,ref_mount_delta_qy_deg,ref_mount_delta_qz_deg,\
align_mount_roll_deg,align_mount_pitch_deg,align_mount_yaw_deg,align_mount_delta_roll_deg,align_mount_delta_pitch_deg,align_mount_delta_yaw_deg,align_mount_delta_q_deg,align_mount_delta_qx_deg,align_mount_delta_qy_deg,align_mount_delta_qz_deg,align_mount_sigma_roll_deg,align_mount_sigma_pitch_deg,align_mount_sigma_yaw_deg,\
align_horiz_count,align_turn_gyro_count,align_horiz_delta_q_deg,align_horiz_delta_qx_deg,align_horiz_delta_qy_deg,align_horiz_delta_qz_deg,align_turn_gyro_delta_q_deg,align_turn_gyro_delta_qx_deg,align_turn_gyro_delta_qy_deg,align_turn_gyro_delta_qz_deg,\
align_horiz_angle_err_deg,align_horiz_effective_std_deg,align_horiz_speed_q,align_horiz_accel_q,align_horiz_turn_q,align_horiz_straight_q,align_horiz_turn_core_valid,align_horiz_straight_core_valid,align_horiz_obs_accel_vx,align_horiz_obs_accel_vy,align_horiz_gnss_norm_mps2,align_horiz_imu_norm_mps2,\
reduced_mount_roll_deg,reduced_mount_pitch_deg,reduced_mount_yaw_deg,reduced_mount_delta_roll_deg,reduced_mount_delta_pitch_deg,reduced_mount_delta_yaw_deg,reduced_mount_delta_q_deg,reduced_mount_delta_qx_deg,reduced_mount_delta_qy_deg,reduced_mount_delta_qz_deg,reduced_mount_error_roll_deg,reduced_mount_error_pitch_deg,reduced_mount_error_yaw_deg,reduced_mount_sigma_roll_deg,reduced_mount_sigma_pitch_deg,reduced_mount_sigma_yaw_deg,reduced_attitude_qerr_deg,\
full_mount_roll_deg,full_mount_pitch_deg,full_mount_yaw_deg,full_mount_delta_roll_deg,full_mount_delta_pitch_deg,full_mount_delta_yaw_deg,full_mount_delta_q_deg,full_mount_delta_qx_deg,full_mount_delta_qy_deg,full_mount_delta_qz_deg,full_mount_error_roll_deg,full_mount_error_pitch_deg,full_mount_error_yaw_deg,full_mount_sigma_roll_deg,full_mount_sigma_pitch_deg,full_mount_sigma_yaw_deg,full_attitude_qerr_deg,\
reduced_gnss_residual_abs,reduced_nhc_y_residual_abs,reduced_nhc_z_residual_abs,full_gnss_residual_abs,full_nhc_y_residual_abs,full_nhc_z_residual_abs,\
reduced_gnss_mount_dx_roll_deg,reduced_gnss_mount_dx_pitch_deg,reduced_gnss_mount_dx_yaw_deg,reduced_nhc_mount_dx_roll_deg,reduced_nhc_mount_dx_pitch_deg,reduced_nhc_mount_dx_yaw_deg,\
full_gnss_mount_dx_roll_deg,full_gnss_mount_dx_pitch_deg,full_gnss_mount_dx_yaw_deg,full_nhc_mount_dx_roll_deg,full_nhc_mount_dx_pitch_deg,full_nhc_mount_dx_yaw_deg\n",
    );
    for sample in samples {
        push_behavior_row(&mut out, sample);
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn push_behavior_row(out: &mut String, sample: &BehaviorSample) {
    csv_val(out, sample.t_s);
    csv_val(out, sample.interval_s);
    out.push_str(sample.motion_regime);
    out.push(',');
    csv_val(out, sample.gnss_speed_mps);
    csv_val(out, sample.gnss_course_rate_dps);
    csv_val(out, sample.gnss_speed_rate_mps2);
    csv_val(out, sample.imu_gyro_norm_dps);
    csv_val(out, sample.imu_gyro_z_dps);
    csv_val(out, sample.imu_accel_norm_err_mps2);
    csv_vec3(out, sample.reference_mount_rpy_deg);
    csv_vec3(out, sample.reference_mount_delta_deg);
    csv_opt(out, sample.reference_mount_delta_q_deg);
    csv_vec3(out, sample.reference_mount_delta_vec_deg);
    csv_vec3(out, sample.align_mount_rpy_deg);
    csv_vec3(out, sample.align_mount_delta_deg);
    csv_opt(out, sample.align_mount_delta_q_deg);
    csv_vec3(out, sample.align_mount_delta_vec_deg);
    csv_vec3(out, sample.align_mount_sigma_deg);
    csv_val(out, sample.align_horiz_count as f64);
    csv_val(out, sample.align_turn_gyro_count as f64);
    csv_opt(out, sample.align_horiz_delta_q_deg);
    csv_vec3(out, sample.align_horiz_delta_vec_deg);
    csv_opt(out, sample.align_turn_gyro_delta_q_deg);
    csv_vec3(out, sample.align_turn_gyro_delta_vec_deg);
    csv_opt(out, sample.align_horiz_angle_err_deg);
    csv_opt(out, sample.align_horiz_effective_std_deg);
    csv_opt(out, sample.align_horiz_speed_q);
    csv_opt(out, sample.align_horiz_accel_q);
    csv_opt(out, sample.align_horiz_turn_q);
    csv_opt(out, sample.align_horiz_straight_q);
    csv_bool(out, sample.align_horiz_turn_core_valid);
    csv_bool(out, sample.align_horiz_straight_core_valid);
    csv_opt(out, sample.align_horiz_obs_accel_vx);
    csv_opt(out, sample.align_horiz_obs_accel_vy);
    csv_opt(out, sample.align_horiz_gnss_norm_mps2);
    csv_opt(out, sample.align_horiz_imu_norm_mps2);
    csv_vec3(out, sample.reduced_mount_rpy_deg);
    csv_vec3(out, sample.reduced_mount_delta_deg);
    csv_opt(out, sample.reduced_mount_delta_q_deg);
    csv_vec3(out, sample.reduced_mount_delta_vec_deg);
    csv_vec3(out, sample.reduced_mount_error_deg);
    csv_vec3(out, sample.reduced_mount_sigma_deg);
    csv_opt(out, sample.reduced_attitude_qerr_deg);
    csv_vec3(out, sample.full_mount_rpy_deg);
    csv_vec3(out, sample.full_mount_delta_deg);
    csv_opt(out, sample.full_mount_delta_q_deg);
    csv_vec3(out, sample.full_mount_delta_vec_deg);
    csv_vec3(out, sample.full_mount_error_deg);
    csv_vec3(out, sample.full_mount_sigma_deg);
    csv_opt(out, sample.full_attitude_qerr_deg);
    csv_val(out, sample.reduced_gnss_residual_abs);
    csv_val(out, sample.reduced_nhc_y_residual_abs);
    csv_val(out, sample.reduced_nhc_z_residual_abs);
    csv_val(out, sample.full_gnss_residual_abs);
    csv_val(out, sample.full_nhc_y_residual_abs);
    csv_val(out, sample.full_nhc_z_residual_abs);
    csv_arr3(out, sample.reduced_gnss_mount_dx_deg);
    csv_arr3(out, sample.reduced_nhc_mount_dx_deg);
    csv_arr3(out, sample.full_gnss_mount_dx_deg);
    csv_arr3(out, sample.full_nhc_mount_dx_deg);
    out.pop();
    out.push('\n');
}

fn csv_vec3(out: &mut String, value: Option<[f64; 3]>) {
    match value {
        Some(v) => csv_arr3(out, v),
        None => out.push_str(",,,"),
    }
}

fn csv_arr3(out: &mut String, value: [f64; 3]) {
    csv_val(out, value[0]);
    csv_val(out, value[1]);
    csv_val(out, value[2]);
}

fn csv_opt(out: &mut String, value: Option<f64>) {
    match value {
        Some(v) => csv_val(out, v),
        None => out.push(','),
    }
}

fn csv_bool(out: &mut String, value: bool) {
    out.push_str(if value { "true" } else { "false" });
    out.push(',');
}

fn csv_val(out: &mut String, value: f64) {
    if value.is_finite() {
        out.push_str(&format!("{value:.9}"));
    }
    out.push(',');
}
