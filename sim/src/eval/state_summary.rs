use crate::visualizer::model::Trace;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SummaryMode {
    Linear,
    AngleDeg,
}

#[derive(Clone, Debug)]
pub struct StateSummary {
    pub system: String,
    pub state: String,
    pub reference: Option<String>,
    pub mode: SummaryMode,
    pub sample_count: usize,
    pub t_start_s: f64,
    pub t_end_s: f64,
    pub duration_s: f64,
    pub early_window_s: f64,
    pub tail_window_s: f64,
    pub initial_value: f64,
    pub final_value: f64,
    pub mean_value: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub span_value: f64,
    pub early_stddev_value: f64,
    pub tail_stddev_value: f64,
    pub tail_span_value: f64,
    pub tail_drift_value: f64,
    pub settle_threshold: Option<f64>,
    pub initial_reference: Option<f64>,
    pub final_reference: Option<f64>,
    pub initial_error: Option<f64>,
    pub final_error: Option<f64>,
    pub mean_abs_error: Option<f64>,
    pub rmse_error: Option<f64>,
    pub max_abs_error: Option<f64>,
    pub p95_abs_error: Option<f64>,
    pub early_stddev_error: Option<f64>,
    pub tail_stddev_error: Option<f64>,
    pub tail_span_error: Option<f64>,
    pub settle_time_s: Option<f64>,
}

pub fn summarize_trace_pair(
    system: &str,
    state: &str,
    trace: &Trace,
    reference: Option<&Trace>,
    mode: SummaryMode,
    settle_threshold: Option<f64>,
) -> Option<StateSummary> {
    if trace.points.is_empty() {
        return None;
    }
    let values: Vec<f64> = trace
        .points
        .iter()
        .map(|p| p[1])
        .filter(|v| v.is_finite())
        .collect();
    if values.is_empty() {
        return None;
    }

    let t_start_s = trace.points.first().map(|p| p[0])?;
    let t_end_s = trace.points.last().map(|p| p[0])?;
    let duration_s = (t_end_s - t_start_s).max(0.0);
    let early_window_s = choose_window(duration_s, 10.0);
    let tail_window_s = choose_window(duration_s, 30.0);

    let initial_value = trace.points.first().map(|p| p[1])?;
    let final_value = trace.points.last().map(|p| p[1])?;
    let mean_value = mean(values.iter().copied());
    let min_value = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span_value = span(&values, mode);

    let early_values = values_in_window(&trace.points, t_start_s, t_start_s + early_window_s);
    let tail_values = values_in_window(&trace.points, t_end_s - tail_window_s, t_end_s);
    let early_stddev_value = stddev(&early_values, mode);
    let tail_stddev_value = stddev(&tail_values, mode);
    let tail_span_value = span(&tail_values, mode);
    let tail_drift_value = drift(&tail_values, mode);

    let mut initial_reference = None;
    let mut final_reference = None;
    let mut initial_error = None;
    let mut final_error = None;
    let mut mean_abs_error = None;
    let mut rmse_error = None;
    let mut max_abs_error = None;
    let mut p95_abs_error = None;
    let mut early_stddev_error = None;
    let mut tail_stddev_error = None;
    let mut tail_span_error = None;
    let mut settle_time_s = None;

    if let Some(reference) = reference {
        let mut errors = Vec::<f64>::new();
        let mut abs_errors = Vec::<f64>::new();
        let mut timed_errors = Vec::<(f64, f64)>::new();
        for point in &trace.points {
            let Some(reference_value) = sample_reference(reference, point[0]) else {
                continue;
            };
            let error = diff(point[1], reference_value, mode);
            errors.push(error);
            abs_errors.push(error.abs());
            timed_errors.push((point[0], error));
        }
        if !timed_errors.is_empty() {
            initial_reference = sample_reference(reference, t_start_s);
            final_reference = sample_reference(reference, t_end_s);
            initial_error = initial_reference.map(|rv| diff(initial_value, rv, mode));
            final_error = final_reference.map(|rv| diff(final_value, rv, mode));
            mean_abs_error = Some(mean(abs_errors.iter().copied()));
            rmse_error = Some(rmse(errors.iter().copied()));
            max_abs_error = Some(abs_errors.iter().copied().fold(0.0, f64::max));
            p95_abs_error = percentile(abs_errors, 0.95);

            let early_errors: Vec<f64> = timed_errors
                .iter()
                .filter(|(t_s, _)| *t_s <= t_start_s + early_window_s)
                .map(|(_, error)| *error)
                .collect();
            let tail_errors: Vec<f64> = timed_errors
                .iter()
                .filter(|(t_s, _)| *t_s >= t_end_s - tail_window_s)
                .map(|(_, error)| *error)
                .collect();
            early_stddev_error = Some(stddev(&early_errors, mode));
            tail_stddev_error = Some(stddev(&tail_errors, mode));
            tail_span_error = Some(span(&tail_errors, mode));
            settle_time_s = settle_threshold
                .and_then(|threshold| settling_time(&timed_errors, threshold, t_start_s, mode));
        }
    }

    Some(StateSummary {
        system: system.to_string(),
        state: state.to_string(),
        reference: reference.map(|trace| trace.name.clone()),
        mode,
        sample_count: trace.points.len(),
        t_start_s,
        t_end_s,
        duration_s,
        early_window_s,
        tail_window_s,
        initial_value,
        final_value,
        mean_value,
        min_value,
        max_value,
        span_value,
        early_stddev_value,
        tail_stddev_value,
        tail_span_value,
        tail_drift_value,
        settle_threshold,
        initial_reference,
        final_reference,
        initial_error,
        final_error,
        mean_abs_error,
        rmse_error,
        max_abs_error,
        p95_abs_error,
        early_stddev_error,
        tail_stddev_error,
        tail_span_error,
        settle_time_s,
    })
}

pub fn print_summary_table(summaries: &[StateSummary]) {
    println!("state_summary:");
    println!(
        "system,state,reference,samples,duration_s,initial,final,final_err,mae,rmse,max_abs_err,p95_abs_err,settle_s,tail_stddev,tail_span,tail_drift"
    );
    for summary in summaries {
        println!(
            "{},{},{},{},{:.3},{:.6},{:.6},{},{},{},{},{},{},{:.6},{:.6},{:.6}",
            summary.system,
            summary.state,
            summary.reference.as_deref().unwrap_or(""),
            summary.sample_count,
            summary.duration_s,
            summary.initial_value,
            summary.final_value,
            fmt_opt(summary.final_error),
            fmt_opt(summary.mean_abs_error),
            fmt_opt(summary.rmse_error),
            fmt_opt(summary.max_abs_error),
            fmt_opt(summary.p95_abs_error),
            fmt_opt(summary.settle_time_s),
            summary.tail_stddev_value,
            summary.tail_span_value,
            summary.tail_drift_value,
        );
    }
}

pub fn write_summary_csv(path: &std::path::Path, summaries: &[StateSummary]) -> anyhow::Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "system,state,reference,mode,sample_count,t_start_s,t_end_s,duration_s,early_window_s,tail_window_s,initial_value,final_value,mean_value,min_value,max_value,span_value,early_stddev_value,tail_stddev_value,tail_span_value,tail_drift_value,settle_threshold,initial_reference,final_reference,initial_error,final_error,mean_abs_error,rmse_error,max_abs_error,p95_abs_error,early_stddev_error,tail_stddev_error,tail_span_error,settle_time_s"
    )?;
    for summary in summaries {
        writeln!(
            writer,
            "{},{},{},{:?},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            summary.system,
            summary.state,
            summary.reference.as_deref().unwrap_or(""),
            summary.mode,
            summary.sample_count,
            summary.t_start_s,
            summary.t_end_s,
            summary.duration_s,
            summary.early_window_s,
            summary.tail_window_s,
            summary.initial_value,
            summary.final_value,
            summary.mean_value,
            summary.min_value,
            summary.max_value,
            summary.span_value,
            summary.early_stddev_value,
            summary.tail_stddev_value,
            summary.tail_span_value,
            summary.tail_drift_value,
            fmt_opt(summary.settle_threshold),
            fmt_opt(summary.initial_reference),
            fmt_opt(summary.final_reference),
            fmt_opt(summary.initial_error),
            fmt_opt(summary.final_error),
            fmt_opt(summary.mean_abs_error),
            fmt_opt(summary.rmse_error),
            fmt_opt(summary.max_abs_error),
            fmt_opt(summary.p95_abs_error),
            fmt_opt(summary.early_stddev_error),
            fmt_opt(summary.tail_stddev_error),
            fmt_opt(summary.tail_span_error),
            fmt_opt(summary.settle_time_s),
        )?;
    }
    Ok(())
}

fn choose_window(duration_s: f64, max_window_s: f64) -> f64 {
    if duration_s <= 0.0 {
        0.0
    } else {
        duration_s
            .min(max_window_s)
            .max((duration_s * 0.1).min(max_window_s))
    }
}

fn values_in_window(points: &[[f64; 2]], t_start_s: f64, t_end_s: f64) -> Vec<f64> {
    points
        .iter()
        .filter(|p| p[0] >= t_start_s && p[0] <= t_end_s && p[1].is_finite())
        .map(|p| p[1])
        .collect()
}

fn sample_reference(trace: &Trace, t_s: f64) -> Option<f64> {
    if trace.points.is_empty() {
        return None;
    }
    let idx = trace.points.partition_point(|point| point[0] < t_s);
    let left = trace
        .points
        .get(idx.saturating_sub(1))
        .filter(|point| point[0].is_finite() && point[1].is_finite())
        .map(|point| ((point[0] - t_s).abs(), point[1]));
    let right = trace
        .points
        .get(idx)
        .filter(|point| point[0].is_finite() && point[1].is_finite())
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

fn diff(a: f64, b: f64, mode: SummaryMode) -> f64 {
    match mode {
        SummaryMode::Linear => a - b,
        SummaryMode::AngleDeg => wrap_deg180(a - b),
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

fn mean<I>(iter: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let (sum, n) = iter.fold((0.0, 0usize), |(sum, n), value| (sum + value, n + 1));
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn stddev(values: &[f64], mode: SummaryMode) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let residuals = match mode {
        SummaryMode::Linear => {
            let mean = mean(values.iter().copied());
            values.iter().map(|value| value - mean).collect::<Vec<_>>()
        }
        SummaryMode::AngleDeg => {
            let mean = circular_mean_deg(values);
            values
                .iter()
                .map(|value| wrap_deg180(*value - mean))
                .collect::<Vec<_>>()
        }
    };
    let variance = residuals.iter().map(|value| value * value).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

fn span(values: &[f64], mode: SummaryMode) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    match mode {
        SummaryMode::Linear => {
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            max - min
        }
        SummaryMode::AngleDeg => {
            let mean = circular_mean_deg(values);
            let residuals = values
                .iter()
                .map(|value| wrap_deg180(*value - mean))
                .collect::<Vec<_>>();
            let min = residuals.iter().copied().fold(f64::INFINITY, f64::min);
            let max = residuals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            max - min
        }
    }
}

fn drift(values: &[f64], mode: SummaryMode) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    diff(*values.last().unwrap(), values[0], mode)
}

fn rmse<I>(iter: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let (sum_sq, n) = iter.fold((0.0, 0usize), |(sum_sq, n), value| {
        (sum_sq + value * value, n + 1)
    });
    if n == 0 {
        0.0
    } else {
        (sum_sq / n as f64).sqrt()
    }
}

fn percentile(mut values: Vec<f64>, fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize;
    values.get(index).copied()
}

fn circular_mean_deg(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sin_sum = values
        .iter()
        .map(|value| value.to_radians().sin())
        .sum::<f64>();
    let cos_sum = values
        .iter()
        .map(|value| value.to_radians().cos())
        .sum::<f64>();
    sin_sum.atan2(cos_sum).to_degrees()
}

fn settling_time(
    timed_errors: &[(f64, f64)],
    threshold: f64,
    t_start_s: f64,
    _mode: SummaryMode,
) -> Option<f64> {
    if timed_errors.is_empty() {
        return None;
    }
    let mut suffix_max = vec![0.0_f64; timed_errors.len()];
    for i in (0..timed_errors.len()).rev() {
        let value = timed_errors[i].1.abs();
        suffix_max[i] = if i + 1 < timed_errors.len() {
            suffix_max[i + 1].max(value)
        } else {
            value
        };
    }
    timed_errors
        .iter()
        .enumerate()
        .find(|(i, _)| suffix_max[*i] <= threshold)
        .map(|(_, (t_s, _))| *t_s - t_start_s)
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_default()
}
