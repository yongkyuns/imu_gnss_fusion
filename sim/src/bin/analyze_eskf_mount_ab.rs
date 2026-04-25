use std::{fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::eval::gnss_ins::{
    quat_angle_deg, quat_axis_angle_deg, quat_from_rpy_alg_deg, quat_from_rpy_deg,
};
use sim::eval::state_summary::{
    StateSummary, SummaryMode, print_summary_table, summarize_trace_pair, write_summary_csv,
};
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
    gnss_pos_r_scale: Option<f64>,
    #[arg(long)]
    gnss_vel_r_scale: Option<f64>,
    #[arg(long)]
    r_body_vel: Option<f32>,
    #[arg(long)]
    gnss_pos_mount_scale: Option<f32>,
    #[arg(long)]
    gnss_vel_mount_scale: Option<f32>,
    #[arg(long)]
    gyro_bias_init_sigma_dps: Option<f32>,
    #[arg(long)]
    r_vehicle_speed: Option<f32>,
    #[arg(long)]
    r_zero_vel: Option<f32>,
    #[arg(long)]
    r_stationary_accel: Option<f32>,
    #[arg(long)]
    mount_align_rw_var: Option<f32>,
    #[arg(long)]
    mount_update_min_scale: Option<f32>,
    #[arg(long)]
    mount_update_ramp_time_s: Option<f32>,
    #[arg(long)]
    mount_update_innovation_gate_mps: Option<f32>,
    #[arg(long)]
    mount_update_yaw_rate_gate_dps: Option<f32>,
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
    #[arg(long)]
    summary_csv: Option<PathBuf>,
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

fn trace_by_name_opt<'a>(traces: &'a [Trace], name: &str) -> Option<&'a Trace> {
    traces.iter().find(|t| t.name == name)
}

fn sample_trace_value(trace: &Trace, t_s: f64) -> Option<f64> {
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

#[derive(Clone, Copy)]
enum EulerConvention {
    Standard,
    AlgMount,
}

fn quat_from_euler_deg(
    convention: EulerConvention,
    roll_deg: f64,
    pitch_deg: f64,
    yaw_deg: f64,
) -> [f64; 4] {
    match convention {
        EulerConvention::Standard => quat_from_rpy_deg(roll_deg, pitch_deg, yaw_deg),
        EulerConvention::AlgMount => quat_from_rpy_alg_deg(roll_deg, pitch_deg, yaw_deg),
    }
}

fn build_quat_error_trace(
    name: &str,
    estimate_rpy: [&Trace; 3],
    reference_rpy: [&Trace; 3],
    convention: EulerConvention,
) -> Trace {
    let mut points = Vec::with_capacity(estimate_rpy[0].points.len());
    for [t_s, roll_deg] in &estimate_rpy[0].points {
        let Some(pitch_deg) = sample_trace_value(estimate_rpy[1], *t_s) else {
            continue;
        };
        let Some(yaw_deg) = sample_trace_value(estimate_rpy[2], *t_s) else {
            continue;
        };
        let Some(ref_roll_deg) = sample_trace_value(reference_rpy[0], *t_s) else {
            continue;
        };
        let Some(ref_pitch_deg) = sample_trace_value(reference_rpy[1], *t_s) else {
            continue;
        };
        let Some(ref_yaw_deg) = sample_trace_value(reference_rpy[2], *t_s) else {
            continue;
        };
        let q_est = quat_from_euler_deg(convention, *roll_deg, pitch_deg, yaw_deg);
        let q_ref = quat_from_euler_deg(convention, ref_roll_deg, ref_pitch_deg, ref_yaw_deg);
        points.push([*t_s, quat_angle_deg(q_est, q_ref)]);
    }
    Trace {
        name: name.to_string(),
        points,
    }
}

fn build_axis_error_trace(
    name: &str,
    estimate_rpy: [&Trace; 3],
    reference_rpy: [&Trace; 3],
    convention: EulerConvention,
    axis: [f64; 3],
) -> Trace {
    let mut points = Vec::with_capacity(estimate_rpy[0].points.len());
    for [t_s, roll_deg] in &estimate_rpy[0].points {
        let Some(pitch_deg) = sample_trace_value(estimate_rpy[1], *t_s) else {
            continue;
        };
        let Some(yaw_deg) = sample_trace_value(estimate_rpy[2], *t_s) else {
            continue;
        };
        let Some(ref_roll_deg) = sample_trace_value(reference_rpy[0], *t_s) else {
            continue;
        };
        let Some(ref_pitch_deg) = sample_trace_value(reference_rpy[1], *t_s) else {
            continue;
        };
        let Some(ref_yaw_deg) = sample_trace_value(reference_rpy[2], *t_s) else {
            continue;
        };
        let q_est = quat_from_euler_deg(convention, *roll_deg, pitch_deg, yaw_deg);
        let q_ref = quat_from_euler_deg(convention, ref_roll_deg, ref_pitch_deg, ref_yaw_deg);
        points.push([*t_s, quat_axis_angle_deg(q_est, q_ref, axis)]);
    }
    Trace {
        name: name.to_string(),
        points,
    }
}

fn zero_trace(name: &str, trace: &Trace) -> Trace {
    Trace {
        name: name.to_string(),
        points: trace.points.iter().map(|point| [point[0], 0.0]).collect(),
    }
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
        r_body_vel: args
            .r_body_vel
            .unwrap_or(EkfCompareConfig::default().r_body_vel),
        gnss_pos_mount_scale: args
            .gnss_pos_mount_scale
            .unwrap_or(EkfCompareConfig::default().gnss_pos_mount_scale),
        gnss_vel_mount_scale: args
            .gnss_vel_mount_scale
            .unwrap_or(EkfCompareConfig::default().gnss_vel_mount_scale),
        gyro_bias_init_sigma_dps: args
            .gyro_bias_init_sigma_dps
            .unwrap_or(EkfCompareConfig::default().gyro_bias_init_sigma_dps),
        r_vehicle_speed: args
            .r_vehicle_speed
            .unwrap_or(EkfCompareConfig::default().r_vehicle_speed),
        r_zero_vel: args
            .r_zero_vel
            .unwrap_or(EkfCompareConfig::default().r_zero_vel),
        r_stationary_accel: args
            .r_stationary_accel
            .unwrap_or(EkfCompareConfig::default().r_stationary_accel),
        mount_align_rw_var: args
            .mount_align_rw_var
            .unwrap_or(EkfCompareConfig::default().mount_align_rw_var),
        mount_update_min_scale: args
            .mount_update_min_scale
            .unwrap_or(EkfCompareConfig::default().mount_update_min_scale),
        mount_update_ramp_time_s: args
            .mount_update_ramp_time_s
            .unwrap_or(EkfCompareConfig::default().mount_update_ramp_time_s),
        mount_update_innovation_gate_mps: args
            .mount_update_innovation_gate_mps
            .unwrap_or(EkfCompareConfig::default().mount_update_innovation_gate_mps),
        mount_update_yaw_rate_gate_dps: args
            .mount_update_yaw_rate_gate_dps
            .unwrap_or(EkfCompareConfig::default().mount_update_yaw_rate_gate_dps),
        gnss_pos_r_scale: args
            .gnss_pos_r_scale
            .unwrap_or(EkfCompareConfig::default().gnss_pos_r_scale),
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
    let summaries = build_state_summaries(&data);

    println!(
        "config: misalignment={:?} decimation={} lpf_hz={} gnss_pos_r_scale={:.3} gnss_vel_r_scale={:.3} r_body_vel={:.3} gnss_pos_mount_scale={:.3} gnss_vel_mount_scale={:.3} gyro_bias_init_sigma_dps={:.3} r_vehicle_speed={:.3} r_zero_vel={:.3} r_stationary_accel={:.3} mount_align_rw_var={:.6e} mount_update_min_scale={:.3} mount_update_ramp_time_s={:.3} mount_update_innovation_gate_mps={:.3} mount_update_yaw_rate_gate_dps={:.3} outage_count={} outage_duration_s={:.3} outage_seed={}",
        args.misalignment,
        ekf_cfg.predict_imu_decimation,
        ekf_cfg
            .predict_imu_lpf_cutoff_hz
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "off".to_string()),
        ekf_cfg.gnss_pos_r_scale,
        ekf_cfg.gnss_vel_r_scale,
        ekf_cfg.r_body_vel,
        ekf_cfg.gnss_pos_mount_scale,
        ekf_cfg.gnss_vel_mount_scale,
        ekf_cfg.gyro_bias_init_sigma_dps,
        ekf_cfg.r_vehicle_speed,
        ekf_cfg.r_zero_vel,
        ekf_cfg.r_stationary_accel,
        ekf_cfg.mount_align_rw_var,
        ekf_cfg.mount_update_min_scale,
        ekf_cfg.mount_update_ramp_time_s,
        ekf_cfg.mount_update_innovation_gate_mps,
        ekf_cfg.mount_update_yaw_rate_gate_dps,
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

    print_summary_table(&summaries);
    if let Some(path) = &args.summary_csv {
        write_summary_csv(path, &summaries)?;
        println!("state_summary_csv={}", path.display());
    }

    Ok(())
}

fn build_state_summaries(data: &sim::visualizer::model::PlotData) -> Vec<StateSummary> {
    struct TraceSpec<'a> {
        system: &'a str,
        state: &'a str,
        trace_name: &'a str,
        reference_name: Option<&'a str>,
        mode: SummaryMode,
        settle_threshold: Option<f64>,
        traces: &'a [Trace],
        references: &'a [Trace],
    }

    let specs = [
        TraceSpec {
            system: "eskf",
            state: "pos_n_m",
            trace_name: "ESKF posN [m]",
            reference_name: Some("UBX posN [m]"),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
            traces: &data.eskf_cmp_pos,
            references: &data.eskf_cmp_pos,
        },
        TraceSpec {
            system: "eskf",
            state: "pos_e_m",
            trace_name: "ESKF posE [m]",
            reference_name: Some("UBX posE [m]"),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
            traces: &data.eskf_cmp_pos,
            references: &data.eskf_cmp_pos,
        },
        TraceSpec {
            system: "eskf",
            state: "pos_d_m",
            trace_name: "ESKF posD [m]",
            reference_name: Some("UBX posD [m]"),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
            traces: &data.eskf_cmp_pos,
            references: &data.eskf_cmp_pos,
        },
        TraceSpec {
            system: "eskf",
            state: "vel_forward_mps",
            trace_name: "ESKF forward vel [m/s]",
            reference_name: Some("u-blox forward vel [m/s]"),
            mode: SummaryMode::Linear,
            settle_threshold: Some(0.5),
            traces: &data.eskf_cmp_vel,
            references: &data.eskf_cmp_vel,
        },
        TraceSpec {
            system: "eskf",
            state: "vel_lateral_mps",
            trace_name: "ESKF lateral vel [m/s]",
            reference_name: Some("u-blox lateral vel [m/s]"),
            mode: SummaryMode::Linear,
            settle_threshold: Some(0.5),
            traces: &data.eskf_cmp_vel,
            references: &data.eskf_cmp_vel,
        },
        TraceSpec {
            system: "eskf",
            state: "vel_vertical_mps",
            trace_name: "ESKF vertical vel [m/s]",
            reference_name: Some("u-blox vertical vel [m/s]"),
            mode: SummaryMode::Linear,
            settle_threshold: Some(0.5),
            traces: &data.eskf_cmp_vel,
            references: &data.eskf_cmp_vel,
        },
        TraceSpec {
            system: "eskf",
            state: "att_roll_deg",
            trace_name: "ESKF roll [deg]",
            reference_name: Some("NAV-ATT roll [deg]"),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(3.0),
            traces: &data.eskf_cmp_att,
            references: &data.eskf_cmp_att,
        },
        TraceSpec {
            system: "eskf",
            state: "att_pitch_deg",
            trace_name: "ESKF pitch [deg]",
            reference_name: Some("NAV-ATT pitch [deg]"),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(3.0),
            traces: &data.eskf_cmp_att,
            references: &data.eskf_cmp_att,
        },
        TraceSpec {
            system: "eskf",
            state: "att_yaw_deg",
            trace_name: "ESKF yaw [deg]",
            reference_name: Some("NAV-ATT heading [deg]"),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(5.0),
            traces: &data.eskf_cmp_att,
            references: &data.eskf_cmp_att,
        },
        TraceSpec {
            system: "eskf",
            state: "mount_roll_deg",
            trace_name: "ESKF full mount roll [deg]",
            reference_name: Some("ESF-ALG mount roll [deg]"),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(5.0),
            traces: &data.eskf_misalignment,
            references: &data.eskf_misalignment,
        },
        TraceSpec {
            system: "eskf",
            state: "mount_pitch_deg",
            trace_name: "ESKF full mount pitch [deg]",
            reference_name: Some("ESF-ALG mount pitch [deg]"),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(5.0),
            traces: &data.eskf_misalignment,
            references: &data.eskf_misalignment,
        },
        TraceSpec {
            system: "eskf",
            state: "mount_yaw_deg",
            trace_name: "ESKF full mount yaw [deg]",
            reference_name: Some("ESF-ALG mount yaw [deg]"),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(5.0),
            traces: &data.eskf_misalignment,
            references: &data.eskf_misalignment,
        },
        TraceSpec {
            system: "eskf",
            state: "gyro_bias_x_dps",
            trace_name: "ESKF gyro bias x [deg/s]",
            reference_name: None,
            mode: SummaryMode::Linear,
            settle_threshold: None,
            traces: &data.eskf_bias_gyro,
            references: &[],
        },
        TraceSpec {
            system: "eskf",
            state: "gyro_bias_y_dps",
            trace_name: "ESKF gyro bias y [deg/s]",
            reference_name: None,
            mode: SummaryMode::Linear,
            settle_threshold: None,
            traces: &data.eskf_bias_gyro,
            references: &[],
        },
        TraceSpec {
            system: "eskf",
            state: "gyro_bias_z_dps",
            trace_name: "ESKF gyro bias z [deg/s]",
            reference_name: None,
            mode: SummaryMode::Linear,
            settle_threshold: None,
            traces: &data.eskf_bias_gyro,
            references: &[],
        },
        TraceSpec {
            system: "eskf",
            state: "accel_bias_x_mps2",
            trace_name: "ESKF accel bias x [m/s^2]",
            reference_name: None,
            mode: SummaryMode::Linear,
            settle_threshold: None,
            traces: &data.eskf_bias_accel,
            references: &[],
        },
        TraceSpec {
            system: "eskf",
            state: "accel_bias_y_mps2",
            trace_name: "ESKF accel bias y [m/s^2]",
            reference_name: None,
            mode: SummaryMode::Linear,
            settle_threshold: None,
            traces: &data.eskf_bias_accel,
            references: &[],
        },
        TraceSpec {
            system: "eskf",
            state: "accel_bias_z_mps2",
            trace_name: "ESKF accel bias z [m/s^2]",
            reference_name: None,
            mode: SummaryMode::Linear,
            settle_threshold: None,
            traces: &data.eskf_bias_accel,
            references: &[],
        },
        TraceSpec {
            system: "loose",
            state: "pos_n_m",
            trace_name: "Loose posN [m]",
            reference_name: Some("UBX posN [m]"),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
            traces: &data.loose_cmp_pos,
            references: &data.loose_cmp_pos,
        },
        TraceSpec {
            system: "loose",
            state: "pos_e_m",
            trace_name: "Loose posE [m]",
            reference_name: Some("UBX posE [m]"),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
            traces: &data.loose_cmp_pos,
            references: &data.loose_cmp_pos,
        },
        TraceSpec {
            system: "loose",
            state: "pos_d_m",
            trace_name: "Loose posD [m]",
            reference_name: Some("UBX posD [m]"),
            mode: SummaryMode::Linear,
            settle_threshold: Some(5.0),
            traces: &data.loose_cmp_pos,
            references: &data.loose_cmp_pos,
        },
        TraceSpec {
            system: "loose",
            state: "vel_forward_mps",
            trace_name: "Loose forward vel [m/s]",
            reference_name: Some("u-blox forward vel [m/s]"),
            mode: SummaryMode::Linear,
            settle_threshold: Some(0.5),
            traces: &data.loose_cmp_vel,
            references: &data.loose_cmp_vel,
        },
        TraceSpec {
            system: "loose",
            state: "vel_lateral_mps",
            trace_name: "Loose lateral vel [m/s]",
            reference_name: Some("u-blox lateral vel [m/s]"),
            mode: SummaryMode::Linear,
            settle_threshold: Some(0.5),
            traces: &data.loose_cmp_vel,
            references: &data.loose_cmp_vel,
        },
        TraceSpec {
            system: "loose",
            state: "vel_vertical_mps",
            trace_name: "Loose vertical vel [m/s]",
            reference_name: Some("u-blox vertical vel [m/s]"),
            mode: SummaryMode::Linear,
            settle_threshold: Some(0.5),
            traces: &data.loose_cmp_vel,
            references: &data.loose_cmp_vel,
        },
        TraceSpec {
            system: "loose",
            state: "att_roll_deg",
            trace_name: "Loose roll [deg]",
            reference_name: Some("NAV-ATT roll [deg]"),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(3.0),
            traces: &data.loose_cmp_att,
            references: &data.loose_cmp_att,
        },
        TraceSpec {
            system: "loose",
            state: "att_pitch_deg",
            trace_name: "Loose pitch [deg]",
            reference_name: Some("NAV-ATT pitch [deg]"),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(3.0),
            traces: &data.loose_cmp_att,
            references: &data.loose_cmp_att,
        },
        TraceSpec {
            system: "loose",
            state: "att_yaw_deg",
            trace_name: "Loose yaw [deg]",
            reference_name: Some("NAV-ATT heading [deg]"),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(5.0),
            traces: &data.loose_cmp_att,
            references: &data.loose_cmp_att,
        },
        TraceSpec {
            system: "loose",
            state: "mount_roll_deg",
            trace_name: "Loose full mount roll [deg]",
            reference_name: Some("ESF-ALG mount roll [deg]"),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(5.0),
            traces: &data.loose_misalignment,
            references: &data.loose_misalignment,
        },
        TraceSpec {
            system: "loose",
            state: "mount_pitch_deg",
            trace_name: "Loose full mount pitch [deg]",
            reference_name: Some("ESF-ALG mount pitch [deg]"),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(5.0),
            traces: &data.loose_misalignment,
            references: &data.loose_misalignment,
        },
        TraceSpec {
            system: "loose",
            state: "mount_yaw_deg",
            trace_name: "Loose full mount yaw [deg]",
            reference_name: Some("ESF-ALG mount yaw [deg]"),
            mode: SummaryMode::AngleDeg,
            settle_threshold: Some(5.0),
            traces: &data.loose_misalignment,
            references: &data.loose_misalignment,
        },
    ];

    let mut summaries: Vec<StateSummary> = specs
        .iter()
        .filter_map(|spec| {
            let trace = trace_by_name_opt(spec.traces, spec.trace_name)?;
            let reference = spec
                .reference_name
                .and_then(|name| trace_by_name_opt(spec.references, name));
            summarize_trace_pair(
                spec.system,
                spec.state,
                trace,
                reference,
                spec.mode,
                spec.settle_threshold,
            )
        })
        .collect();

    for (system, att_traces, mount_traces, mount_ref_name_prefix) in [
        ("eskf", &data.eskf_cmp_att, &data.eskf_misalignment, "ESKF"),
        (
            "loose",
            &data.loose_cmp_att,
            &data.loose_misalignment,
            "Loose",
        ),
    ] {
        if let (
            Some(est_roll),
            Some(est_pitch),
            Some(est_yaw),
            Some(ref_roll),
            Some(ref_pitch),
            Some(ref_yaw),
        ) = (
            trace_by_name_opt(att_traces, &format!("{mount_ref_name_prefix} roll [deg]")),
            trace_by_name_opt(att_traces, &format!("{mount_ref_name_prefix} pitch [deg]")),
            trace_by_name_opt(att_traces, &format!("{mount_ref_name_prefix} yaw [deg]")),
            trace_by_name_opt(att_traces, "NAV-ATT roll [deg]"),
            trace_by_name_opt(att_traces, "NAV-ATT pitch [deg]"),
            trace_by_name_opt(att_traces, "NAV-ATT heading [deg]"),
        ) {
            let att_error_trace = build_quat_error_trace(
                &format!("{system} attitude quat err [deg]"),
                [est_roll, est_pitch, est_yaw],
                [ref_roll, ref_pitch, ref_yaw],
                EulerConvention::Standard,
            );
            let att_zero = zero_trace("zero", &att_error_trace);
            if let Some(summary) = summarize_trace_pair(
                system,
                "att_quat_err_deg",
                &att_error_trace,
                Some(&att_zero),
                SummaryMode::Linear,
                Some(5.0),
            ) {
                summaries.push(summary);
            }

            for (state, name, axis) in [
                ("att_fwd_err_deg", "attitude forward err", [1.0, 0.0, 0.0]),
                ("att_down_err_deg", "attitude down err", [0.0, 0.0, 1.0]),
            ] {
                let axis_trace = build_axis_error_trace(
                    &format!("{system} {name} [deg]"),
                    [est_roll, est_pitch, est_yaw],
                    [ref_roll, ref_pitch, ref_yaw],
                    EulerConvention::Standard,
                    axis,
                );
                let axis_zero = zero_trace("zero", &axis_trace);
                if let Some(summary) = summarize_trace_pair(
                    system,
                    state,
                    &axis_trace,
                    Some(&axis_zero),
                    SummaryMode::Linear,
                    Some(5.0),
                ) {
                    summaries.push(summary);
                }
            }
        }

        if let (
            Some(est_roll),
            Some(est_pitch),
            Some(est_yaw),
            Some(ref_roll),
            Some(ref_pitch),
            Some(ref_yaw),
        ) = (
            trace_by_name_opt(
                mount_traces,
                &format!("{mount_ref_name_prefix} full mount roll [deg]"),
            ),
            trace_by_name_opt(
                mount_traces,
                &format!("{mount_ref_name_prefix} full mount pitch [deg]"),
            ),
            trace_by_name_opt(
                mount_traces,
                &format!("{mount_ref_name_prefix} full mount yaw [deg]"),
            ),
            trace_by_name_opt(mount_traces, "ESF-ALG mount roll [deg]"),
            trace_by_name_opt(mount_traces, "ESF-ALG mount pitch [deg]"),
            trace_by_name_opt(mount_traces, "ESF-ALG mount yaw [deg]"),
        ) {
            let mount_error_trace = build_quat_error_trace(
                &format!("{system} mount quat err [deg]"),
                [est_roll, est_pitch, est_yaw],
                [ref_roll, ref_pitch, ref_yaw],
                EulerConvention::AlgMount,
            );
            let mount_zero = zero_trace("zero", &mount_error_trace);
            if let Some(summary) = summarize_trace_pair(
                system,
                "mount_quat_err_deg",
                &mount_error_trace,
                Some(&mount_zero),
                SummaryMode::Linear,
                Some(5.0),
            ) {
                summaries.push(summary);
            }

            for (state, name, axis) in [
                ("mount_fwd_err_deg", "mount forward err", [1.0, 0.0, 0.0]),
                ("mount_down_err_deg", "mount down err", [0.0, 0.0, 1.0]),
            ] {
                let axis_trace = build_axis_error_trace(
                    &format!("{system} {name} [deg]"),
                    [est_roll, est_pitch, est_yaw],
                    [ref_roll, ref_pitch, ref_yaw],
                    EulerConvention::AlgMount,
                    axis,
                );
                let axis_zero = zero_trace("zero", &axis_trace);
                if let Some(summary) = summarize_trace_pair(
                    system,
                    state,
                    &axis_trace,
                    Some(&axis_zero),
                    SummaryMode::Linear,
                    Some(5.0),
                ) {
                    summaries.push(summary);
                }
            }
        }
    }

    summaries
}
