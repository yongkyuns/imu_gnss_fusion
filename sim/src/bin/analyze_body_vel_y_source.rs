use std::{fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::visualizer::model::{EkfImuSource, Trace};
use sim::visualizer::pipeline::build_plot_data;
use sim::visualizer::pipeline::ekf_compare::{EkfCompareConfig, GnssOutageConfig};

#[derive(Parser, Debug)]
#[command(name = "analyze_body_vel_y_source")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long, default_value = "align", value_parser = parse_misalignment)]
    misalignment: EkfImuSource,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long, default_value_t = 1)]
    ekf_predict_imu_decimation: usize,
    #[arg(long)]
    ekf_predict_imu_lpf_cutoff_hz: Option<f64>,
    #[arg(long, default_value_t = 0)]
    gnss_outage_count: usize,
    #[arg(long, default_value_t = 0.0)]
    gnss_outage_duration_s: f64,
    #[arg(long, default_value_t = 1)]
    gnss_outage_seed: u64,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
}

#[derive(Clone, Debug)]
struct Row {
    t_s: f64,
    innov: f64,
    innov_abs: f64,
    yaw_dx_deg: f64,
    yaw_dx_abs_deg: f64,
    course_dps: f64,
    a_lat_mps2: f64,
    a_long_mps2: f64,
    nav_roll_deg: f64,
    nav_pitch_deg: f64,
    mount_roll_deg: f64,
    mount_pitch_deg: f64,
    mount_yaw_deg: f64,
}

fn parse_misalignment(s: &str) -> Result<EkfImuSource, String> {
    match s.to_ascii_lowercase().as_str() {
        "align" | "auto" => Ok(EkfImuSource::Align),
        "esf-alg" | "esf_alg" | "esfalg" => Ok(EkfImuSource::EsfAlg),
        other => Err(format!("invalid misalignment source: {other}")),
    }
}

fn trace_by_name<'a>(traces: &'a [Trace], name: &str) -> Result<&'a Trace> {
    traces
        .iter()
        .find(|t| t.name == name)
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

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return f64::NAN;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 1e-12 || var_y <= 1e-12 {
        f64::NAN
    } else {
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    let mut w_sum = 0.0;
    let mut vw_sum = 0.0;
    for (&value, &weight) in values.iter().zip(weights.iter()) {
        if value.is_finite() && weight.is_finite() && weight > 0.0 {
            w_sum += weight;
            vw_sum += value * weight;
        }
    }
    if w_sum <= 0.0 {
        f64::NAN
    } else {
        vw_sum / w_sum
    }
}

fn weighted_fraction(weights: &[f64], values: &[f64], pred: impl Fn(f64) -> bool) -> f64 {
    let mut w_sum = 0.0;
    let mut selected = 0.0;
    for (&weight, &value) in weights.iter().zip(values.iter()) {
        if weight.is_finite() && weight > 0.0 && value.is_finite() {
            w_sum += weight;
            if pred(value) {
                selected += weight;
            }
        }
    }
    if w_sum <= 0.0 {
        f64::NAN
    } else {
        selected / w_sum
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
        predict_imu_decimation: args.ekf_predict_imu_decimation.max(1),
        predict_imu_lpf_cutoff_hz: args.ekf_predict_imu_lpf_cutoff_hz,
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

    let innov_sum = trace_by_name(&data.eskf_stationary_diag, "innovation sum body_vel_y")?;
    let innov_abs = trace_by_name(&data.eskf_stationary_diag, "innovation abs body_vel_y")?;
    let yaw_dx_sum =
        trace_by_name(&data.eskf_stationary_diag, "mount yaw dx sum body_vel_y [deg]")?;
    let yaw_dx_abs =
        trace_by_name(&data.eskf_stationary_diag, "mount yaw dx abs body_vel_y [deg]")?;

    let course = trace_by_name(&data.align_res_vel, "course rate [deg/s]")?;
    let a_lat = trace_by_name(&data.align_res_vel, "a_lat [m/s^2]")?;
    let a_long = trace_by_name(&data.align_res_vel, "a_long [m/s^2]")?;
    let nav_roll = trace_by_name(&data.eskf_cmp_att, "NAV-ATT roll [deg]")?;
    let nav_pitch = trace_by_name(&data.eskf_cmp_att, "NAV-ATT pitch [deg]")?;
    let mount_roll = trace_by_name(&data.eskf_misalignment, "ESKF full mount roll [deg]")?;
    let mount_pitch = trace_by_name(&data.eskf_misalignment, "ESKF full mount pitch [deg]")?;
    let mount_yaw = trace_by_name(&data.eskf_misalignment, "ESKF full mount yaw [deg]")?;

    let mut rows = Vec::<Row>::new();
    let mut prev_innov_sum = None;
    let mut prev_innov_abs = None;
    let mut prev_yaw_dx_sum = None;
    let mut prev_yaw_dx_abs = None;

    for [t_s, curr_innov_sum] in &innov_sum.points {
        let Some(curr_innov_abs) = sample_trace(innov_abs, *t_s) else {
            continue;
        };
        let Some(curr_yaw_dx_sum) = sample_trace(yaw_dx_sum, *t_s) else {
            continue;
        };
        let Some(curr_yaw_dx_abs) = sample_trace(yaw_dx_abs, *t_s) else {
            continue;
        };
        let (Some(prev_i_sum), Some(prev_i_abs), Some(prev_dx_sum), Some(prev_dx_abs)) =
            (prev_innov_sum, prev_innov_abs, prev_yaw_dx_sum, prev_yaw_dx_abs)
        else {
            prev_innov_sum = Some(*curr_innov_sum);
            prev_innov_abs = Some(curr_innov_abs);
            prev_yaw_dx_sum = Some(curr_yaw_dx_sum);
            prev_yaw_dx_abs = Some(curr_yaw_dx_abs);
            continue;
        };

        let d_innov = *curr_innov_sum - prev_i_sum;
        let d_innov_abs = curr_innov_abs - prev_i_abs;
        let d_yaw_dx = curr_yaw_dx_sum - prev_dx_sum;
        let d_yaw_dx_abs = curr_yaw_dx_abs - prev_dx_abs;

        prev_innov_sum = Some(*curr_innov_sum);
        prev_innov_abs = Some(curr_innov_abs);
        prev_yaw_dx_sum = Some(curr_yaw_dx_sum);
        prev_yaw_dx_abs = Some(curr_yaw_dx_abs);

        if d_innov_abs <= 1.0e-6 && d_yaw_dx_abs <= 1.0e-6 {
            continue;
        }

        rows.push(Row {
            t_s: *t_s,
            innov: d_innov,
            innov_abs: d_innov_abs.max(0.0),
            yaw_dx_deg: d_yaw_dx,
            yaw_dx_abs_deg: d_yaw_dx_abs.max(0.0),
            course_dps: sample_trace(course, *t_s).unwrap_or(f64::NAN),
            a_lat_mps2: sample_trace(a_lat, *t_s).unwrap_or(f64::NAN),
            a_long_mps2: sample_trace(a_long, *t_s).unwrap_or(f64::NAN),
            nav_roll_deg: sample_trace(nav_roll, *t_s).unwrap_or(f64::NAN),
            nav_pitch_deg: sample_trace(nav_pitch, *t_s).unwrap_or(f64::NAN),
            mount_roll_deg: sample_trace(mount_roll, *t_s).unwrap_or(f64::NAN),
            mount_pitch_deg: sample_trace(mount_pitch, *t_s).unwrap_or(f64::NAN),
            mount_yaw_deg: sample_trace(mount_yaw, *t_s).unwrap_or(f64::NAN),
        });
    }

    let weights_innov: Vec<f64> = rows.iter().map(|row| row.innov_abs).collect();
    let weights_dx: Vec<f64> = rows.iter().map(|row| row.yaw_dx_abs_deg).collect();
    let course_abs: Vec<f64> = rows.iter().map(|row| row.course_dps.abs()).collect();
    let lat_abs: Vec<f64> = rows.iter().map(|row| row.a_lat_mps2.abs()).collect();
    let long_abs: Vec<f64> = rows.iter().map(|row| row.a_long_mps2.abs()).collect();
    let roll_abs: Vec<f64> = rows.iter().map(|row| row.nav_roll_deg.abs()).collect();
    let pitch_abs: Vec<f64> = rows.iter().map(|row| row.nav_pitch_deg.abs()).collect();

    println!("rows={}", rows.len());
    println!(
        "total_body_vel_y_innov_abs={:.3} total_body_vel_y_yaw_dx_abs_deg={:.3}",
        weights_innov.iter().sum::<f64>(),
        weights_dx.iter().sum::<f64>()
    );
    println!(
        "corr(innov_abs, |course|)={:.3} corr(innov_abs, |a_lat|)={:.3} corr(innov_abs, |a_long|)={:.3}",
        pearson(&weights_innov, &course_abs),
        pearson(&weights_innov, &lat_abs),
        pearson(&weights_innov, &long_abs),
    );
    println!(
        "corr(innov_abs, |nav_roll|)={:.3} corr(innov_abs, |nav_pitch|)={:.3}",
        pearson(&weights_innov, &roll_abs),
        pearson(&weights_innov, &pitch_abs),
    );
    println!(
        "weighted_mean(|course|, innov_abs)={:.3} weighted_mean(|a_lat|, innov_abs)={:.3} weighted_mean(|nav_roll|, innov_abs)={:.3} weighted_mean(|nav_pitch|, innov_abs)={:.3}",
        weighted_mean(&course_abs, &weights_innov),
        weighted_mean(&lat_abs, &weights_innov),
        weighted_mean(&roll_abs, &weights_innov),
        weighted_mean(&pitch_abs, &weights_innov),
    );
    println!(
        "fraction_innov_abs(|course|<1 dps)={:.3} fraction_innov_abs(|course|<2 dps)={:.3} fraction_innov_abs(|a_lat|<0.2)={:.3} fraction_innov_abs(|nav_roll|>2 deg)={:.3} fraction_innov_abs(|nav_roll|>3 deg)={:.3}",
        weighted_fraction(&weights_innov, &course_abs, |v| v < 1.0),
        weighted_fraction(&weights_innov, &course_abs, |v| v < 2.0),
        weighted_fraction(&weights_innov, &lat_abs, |v| v < 0.2),
        weighted_fraction(&weights_innov, &roll_abs, |v| v > 2.0),
        weighted_fraction(&weights_innov, &roll_abs, |v| v > 3.0),
    );

    rows.sort_by(|a, b| b.innov_abs.total_cmp(&a.innov_abs));
    println!("top_body_vel_y_updates:");
    for row in rows.iter().take(args.top_k) {
        println!(
            "  t={:.3}s innov={:.3} innov_abs={:.3} yaw_dx_deg={:.3} course={:.3} a_lat={:.3} a_long={:.3} nav_roll={:.3} nav_pitch={:.3} mount=[{:.3},{:.3},{:.3}]",
            row.t_s,
            row.innov,
            row.innov_abs,
            row.yaw_dx_deg,
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

    Ok(())
}
