use std::{fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::eval::gnss_ins::{quat_angle_deg, quat_from_rpy_alg_deg, quat_rotate};
use sim::visualizer::model::{EkfImuSource, PlotData, Trace};
use sim::visualizer::pipeline::build_plot_data;
use sim::visualizer::pipeline::ekf_compare::{EkfCompareConfig, GnssOutageConfig};

const CUES: [&str; 11] = [
    "gps_pos_ne",
    "gps_vel_ne",
    "zero_vel_ne",
    "body_speed_x",
    "body_vel_y",
    "body_vel_z",
    "stationary_x",
    "stationary_y",
    "gps_pos_d",
    "gps_vel_d",
    "zero_vel_d",
];

#[derive(Parser, Debug)]
#[command(name = "analyze_mount_update_direction")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long, default_value = "align", value_parser = parse_misalignment)]
    misalignment: EkfImuSource,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long)]
    r_body_vel: Option<f32>,
    #[arg(long)]
    gnss_pos_mount_scale: Option<f32>,
    #[arg(long)]
    gnss_vel_mount_scale: Option<f32>,
    #[arg(long)]
    mount_update_yaw_rate_gate_dps: Option<f32>,
    #[arg(long)]
    mount_update_innovation_gate_mps: Option<f32>,
    #[arg(long, default_value_t = 12)]
    top_k: usize,
}

#[derive(Clone, Debug)]
struct Row {
    cue: &'static str,
    t_s: f64,
    dx_yaw_deg: f64,
    dx_yaw_abs_deg: f64,
    innovation: f64,
    innovation_abs: f64,
    pre_err_deg: f64,
    post_err_deg: f64,
    delta_err_deg: f64,
    pre_fwd_err_deg: f64,
    post_fwd_err_deg: f64,
    delta_fwd_err_deg: f64,
    pre_down_err_deg: f64,
    post_down_err_deg: f64,
    delta_down_err_deg: f64,
    course_dps: f64,
    a_lat_mps2: f64,
    a_long_mps2: f64,
    nav_roll_deg: f64,
    nav_pitch_deg: f64,
    speed_mps: f64,
    mount_roll_deg: f64,
    mount_pitch_deg: f64,
    mount_yaw_deg: f64,
}

#[derive(Default)]
struct CueStats {
    count: usize,
    helpful: usize,
    harmful: usize,
    neutral: usize,
    dx_abs_sum: f64,
    innov_abs_sum: f64,
    delta_err_sum: f64,
    dx_weighted_delta_sum: f64,
    innov_weighted_delta_sum: f64,
}

fn parse_misalignment(s: &str) -> Result<EkfImuSource, String> {
    match s.to_ascii_lowercase().as_str() {
        "align" | "auto" => Ok(EkfImuSource::Align),
        "esf-alg" | "esf_alg" | "esfalg" => Ok(EkfImuSource::EsfAlg),
        other => Err(format!("invalid misalignment source: {other}")),
    }
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
        mount_update_yaw_rate_gate_dps: args
            .mount_update_yaw_rate_gate_dps
            .unwrap_or(EkfCompareConfig::default().mount_update_yaw_rate_gate_dps),
        mount_update_innovation_gate_mps: args
            .mount_update_innovation_gate_mps
            .unwrap_or(EkfCompareConfig::default().mount_update_innovation_gate_mps),
        ..EkfCompareConfig::default()
    };
    let (data, _has_itow) = build_plot_data(
        &bytes,
        args.max_records,
        args.misalignment,
        ekf_cfg,
        GnssOutageConfig::default(),
    );

    let rows = collect_rows(&data)?;
    print_summary(&args, ekf_cfg, &rows);
    print_top_rows("top_harmful_updates", &rows, args.top_k, |row| {
        row.delta_err_deg
    });
    print_top_rows("top_helpful_updates", &rows, args.top_k, |row| {
        -row.delta_err_deg
    });

    Ok(())
}

fn collect_rows(data: &PlotData) -> Result<Vec<Row>> {
    let mount_roll = trace_by_name(&data.eskf_misalignment, "ESKF full mount roll [deg]")?;
    let mount_pitch = trace_by_name(&data.eskf_misalignment, "ESKF full mount pitch [deg]")?;
    let mount_yaw = trace_by_name(&data.eskf_misalignment, "ESKF full mount yaw [deg]")?;
    let truth_roll = trace_by_name(&data.eskf_misalignment, "ESF-ALG mount roll [deg]")?;
    let truth_pitch = trace_by_name(&data.eskf_misalignment, "ESF-ALG mount pitch [deg]")?;
    let truth_yaw = trace_by_name(&data.eskf_misalignment, "ESF-ALG mount yaw [deg]")?;
    let course = trace_by_name(&data.align_res_vel, "course rate [deg/s]")?;
    let a_lat = trace_by_name(&data.align_res_vel, "a_lat [m/s^2]")?;
    let a_long = trace_by_name(&data.align_res_vel, "a_long [m/s^2]")?;
    let nav_roll = trace_by_name(&data.eskf_cmp_att, "NAV-ATT roll [deg]")?;
    let nav_pitch = trace_by_name(&data.eskf_cmp_att, "NAV-ATT pitch [deg]")?;
    let speed = trace_by_name(&data.speed, "speed [m/s]")
        .or_else(|_| trace_by_name(&data.eskf_cmp_vel, "u-blox forward vel [m/s]"))?;

    let mut rows = Vec::new();
    for cue in CUES {
        let dx_sum = trace_by_name(
            &data.eskf_stationary_diag,
            &format!("mount yaw dx sum {cue} [deg]"),
        )?;
        let dx_abs = trace_by_name(
            &data.eskf_stationary_diag,
            &format!("mount yaw dx abs {cue} [deg]"),
        )?;
        let innov_sum =
            trace_by_name(&data.eskf_stationary_diag, &format!("innovation sum {cue}"))?;
        let innov_abs =
            trace_by_name(&data.eskf_stationary_diag, &format!("innovation abs {cue}"))?;

        let mut prev_dx_sum = None;
        let mut prev_dx_abs = None;
        let mut prev_innov_sum = None;
        let mut prev_innov_abs = None;
        let mut prev_t = None;

        for [t_s, curr_dx_sum] in &dx_sum.points {
            let Some(curr_dx_abs) = sample_trace(dx_abs, *t_s) else {
                continue;
            };
            let Some(curr_innov_sum) = sample_trace(innov_sum, *t_s) else {
                continue;
            };
            let Some(curr_innov_abs) = sample_trace(innov_abs, *t_s) else {
                continue;
            };

            let (
                Some(prev_dx_sum_v),
                Some(prev_dx_abs_v),
                Some(prev_innov_sum_v),
                Some(prev_innov_abs_v),
                Some(pre_t),
            ) = (
                prev_dx_sum,
                prev_dx_abs,
                prev_innov_sum,
                prev_innov_abs,
                prev_t,
            )
            else {
                prev_dx_sum = Some(*curr_dx_sum);
                prev_dx_abs = Some(curr_dx_abs);
                prev_innov_sum = Some(curr_innov_sum);
                prev_innov_abs = Some(curr_innov_abs);
                prev_t = Some(*t_s);
                continue;
            };

            let d_dx = *curr_dx_sum - prev_dx_sum_v;
            let d_dx_abs = (curr_dx_abs - prev_dx_abs_v).max(0.0);
            let d_innov = curr_innov_sum - prev_innov_sum_v;
            let d_innov_abs = (curr_innov_abs - prev_innov_abs_v).max(0.0);

            prev_dx_sum = Some(*curr_dx_sum);
            prev_dx_abs = Some(curr_dx_abs);
            prev_innov_sum = Some(curr_innov_sum);
            prev_innov_abs = Some(curr_innov_abs);
            prev_t = Some(*t_s);

            if d_dx_abs <= 1.0e-9 && d_innov_abs <= 1.0e-9 {
                continue;
            }

            let Some(pre_q) = sample_mount_quat(mount_roll, mount_pitch, mount_yaw, pre_t) else {
                continue;
            };
            let Some(post_q) = sample_mount_quat(mount_roll, mount_pitch, mount_yaw, *t_s) else {
                continue;
            };
            let Some(truth_q) = sample_mount_quat(truth_roll, truth_pitch, truth_yaw, *t_s) else {
                continue;
            };
            let Some((mr, mp, my)) = sample_mount_rpy(mount_roll, mount_pitch, mount_yaw, *t_s)
            else {
                continue;
            };

            let pre_err = quat_angle_deg(pre_q, truth_q);
            let post_err = quat_angle_deg(post_q, truth_q);
            let pre_fwd_err = axis_err_deg(pre_q, truth_q, [1.0, 0.0, 0.0]);
            let post_fwd_err = axis_err_deg(post_q, truth_q, [1.0, 0.0, 0.0]);
            let pre_down_err = axis_err_deg(pre_q, truth_q, [0.0, 0.0, 1.0]);
            let post_down_err = axis_err_deg(post_q, truth_q, [0.0, 0.0, 1.0]);

            rows.push(Row {
                cue,
                t_s: *t_s,
                dx_yaw_deg: d_dx,
                dx_yaw_abs_deg: d_dx_abs,
                innovation: d_innov,
                innovation_abs: d_innov_abs,
                pre_err_deg: pre_err,
                post_err_deg: post_err,
                delta_err_deg: post_err - pre_err,
                pre_fwd_err_deg: pre_fwd_err,
                post_fwd_err_deg: post_fwd_err,
                delta_fwd_err_deg: post_fwd_err - pre_fwd_err,
                pre_down_err_deg: pre_down_err,
                post_down_err_deg: post_down_err,
                delta_down_err_deg: post_down_err - pre_down_err,
                course_dps: sample_trace(course, *t_s).unwrap_or(f64::NAN),
                a_lat_mps2: sample_trace(a_lat, *t_s).unwrap_or(f64::NAN),
                a_long_mps2: sample_trace(a_long, *t_s).unwrap_or(f64::NAN),
                nav_roll_deg: sample_trace(nav_roll, *t_s).unwrap_or(f64::NAN),
                nav_pitch_deg: sample_trace(nav_pitch, *t_s).unwrap_or(f64::NAN),
                speed_mps: sample_trace(speed, *t_s).unwrap_or(f64::NAN),
                mount_roll_deg: mr,
                mount_pitch_deg: mp,
                mount_yaw_deg: my,
            });
        }
    }
    Ok(rows)
}

fn print_summary(args: &Args, cfg: EkfCompareConfig, rows: &[Row]) {
    println!(
        "config: misalignment={:?} r_body_vel={:.6} gnss_pos_mount_scale={:.3} gnss_vel_mount_scale={:.3} mount_update_innovation_gate_mps={:.3} mount_update_yaw_rate_gate_dps={:.3}",
        args.misalignment,
        cfg.r_body_vel,
        cfg.gnss_pos_mount_scale,
        cfg.gnss_vel_mount_scale,
        cfg.mount_update_innovation_gate_mps,
        cfg.mount_update_yaw_rate_gate_dps,
    );
    println!("note: direction is scored at sample level; simultaneous batch cues share the same before/after mount state.");
    println!("cue_summary:");
    println!(
        "cue,count,helpful,harmful,neutral,helpful_frac,dx_abs_sum_deg,innov_abs_sum,mean_delta_err_deg,dx_weighted_delta_err_deg,innov_weighted_delta_err_deg"
    );
    for cue in CUES {
        let mut stats = CueStats::default();
        for row in rows.iter().filter(|row| row.cue == cue) {
            stats.count += 1;
            stats.dx_abs_sum += row.dx_yaw_abs_deg;
            stats.innov_abs_sum += row.innovation_abs;
            stats.delta_err_sum += row.delta_err_deg;
            stats.dx_weighted_delta_sum += row.delta_err_deg * row.dx_yaw_abs_deg;
            stats.innov_weighted_delta_sum += row.delta_err_deg * row.innovation_abs;
            if row.delta_err_deg < -1.0e-6 {
                stats.helpful += 1;
            } else if row.delta_err_deg > 1.0e-6 {
                stats.harmful += 1;
            } else {
                stats.neutral += 1;
            }
        }
        if stats.count == 0 {
            continue;
        }
        let helpful_frac = stats.helpful as f64 / stats.count as f64;
        let dx_weighted = if stats.dx_abs_sum > 0.0 {
            stats.dx_weighted_delta_sum / stats.dx_abs_sum
        } else {
            f64::NAN
        };
        let innov_weighted = if stats.innov_abs_sum > 0.0 {
            stats.innov_weighted_delta_sum / stats.innov_abs_sum
        } else {
            f64::NAN
        };
        println!(
            "{},{},{},{},{},{:.3},{:.6},{:.3},{:.9},{:.9},{:.9}",
            cue,
            stats.count,
            stats.helpful,
            stats.harmful,
            stats.neutral,
            helpful_frac,
            stats.dx_abs_sum,
            stats.innov_abs_sum,
            stats.delta_err_sum / stats.count as f64,
            dx_weighted,
            innov_weighted,
        );
    }
}

fn print_top_rows(
    title: &str,
    rows: &[Row],
    top_k: usize,
    mut key: impl FnMut(&Row) -> f64,
) {
    let mut selected: Vec<_> = rows
        .iter()
        .filter(|row| row.delta_err_deg.is_finite())
        .collect();
    selected.sort_by(|a, b| key(a).total_cmp(&key(b)));
    println!("{title}:");
    for row in selected.into_iter().rev().take(top_k) {
        println!(
            "  t={:.3}s cue={} d_err={:+.6} qerr={:.3}->{:.3} fwd={:.3}->{:.3}({:+.6}) down={:.3}->{:.3}({:+.6}) dx_yaw={:+.6} innov={:+.3} speed={:.2} course={:+.2} a_lat={:+.2} a_long={:+.2} nav_rp=[{:+.2},{:+.2}] mount=[{:+.2},{:+.2},{:+.2}]",
            row.t_s,
            row.cue,
            row.delta_err_deg,
            row.pre_err_deg,
            row.post_err_deg,
            row.pre_fwd_err_deg,
            row.post_fwd_err_deg,
            row.delta_fwd_err_deg,
            row.pre_down_err_deg,
            row.post_down_err_deg,
            row.delta_down_err_deg,
            row.dx_yaw_deg,
            row.innovation,
            row.speed_mps,
            row.course_dps,
            row.a_lat_mps2,
            row.a_long_mps2,
            row.nav_roll_deg,
            row.nav_pitch_deg,
            row.mount_roll_deg,
            row.mount_pitch_deg,
            row.mount_yaw_deg,
        );
    }
}

fn trace_by_name<'a>(traces: &'a [Trace], name: &str) -> Result<&'a Trace> {
    traces
        .iter()
        .find(|trace| trace.name == name)
        .with_context(|| format!("missing trace `{name}`"))
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

fn sample_mount_rpy(
    roll: &Trace,
    pitch: &Trace,
    yaw: &Trace,
    t_s: f64,
) -> Option<(f64, f64, f64)> {
    Some((
        sample_trace(roll, t_s)?,
        sample_trace(pitch, t_s)?,
        sample_trace(yaw, t_s)?,
    ))
}

fn sample_mount_quat(roll: &Trace, pitch: &Trace, yaw: &Trace, t_s: f64) -> Option<[f64; 4]> {
    let (roll_deg, pitch_deg, yaw_deg) = sample_mount_rpy(roll, pitch, yaw, t_s)?;
    Some(quat_from_rpy_alg_deg(roll_deg, pitch_deg, yaw_deg))
}

fn axis_err_deg(q_est: [f64; 4], q_ref: [f64; 4], axis: [f64; 3]) -> f64 {
    let est = quat_rotate(q_est, axis);
    let reference = quat_rotate(q_ref, axis);
    let dot = (est[0] * reference[0] + est[1] * reference[1] + est[2] * reference[2])
        .clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}
