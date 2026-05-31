use anyhow::{Result, bail};

use crate::visualizer::model::Trace;

pub fn find_trace<'a>(traces: &'a [Trace], name: &str) -> Option<&'a Trace> {
    traces.iter().find(|trace| trace.name == name)
}

pub fn require_trace<'a>(group: &str, traces: &'a [Trace], name: &str) -> Result<&'a Trace> {
    find_trace(traces, name).ok_or_else(|| {
        anyhow::anyhow!(
            "missing trace '{name}' in {group}; available traces: {}",
            trace_names(traces).join(", ")
        )
    })
}

pub fn require_trace_schema<'a>(
    group: &str,
    traces: &'a [Trace],
    required_names: &[&str],
) -> Result<Vec<&'a Trace>> {
    if traces.is_empty() {
        bail!("{group} produced no traces");
    }

    let mut required = Vec::with_capacity(required_names.len());
    for name in required_names {
        let trace = require_trace(group, traces, name)?;
        required.push(trace);
    }
    Ok(required)
}

pub fn require_trace_points(group: &str, trace: &Trace) -> Result<()> {
    require_finite_points(group, trace)
}

pub fn sample_nearest_point(trace: &Trace, t_s: f64) -> Option<[f64; 2]> {
    if trace.points.is_empty() {
        return None;
    }

    let idx = trace.points.partition_point(|point| point[0] < t_s);
    let left = trace
        .points
        .get(idx.saturating_sub(1))
        .copied()
        .filter(finite_point)
        .map(|point| ((point[0] - t_s).abs(), point));
    let right = trace
        .points
        .get(idx)
        .copied()
        .filter(finite_point)
        .map(|point| ((point[0] - t_s).abs(), point));

    match (left, right) {
        (Some((left_dt, left_point)), Some((right_dt, right_point))) => {
            if right_dt < left_dt {
                Some(right_point)
            } else {
                Some(left_point)
            }
        }
        (Some((_, point)), None) | (None, Some((_, point))) => Some(point),
        (None, None) => None,
    }
}

pub fn sample_nearest_value(trace: &Trace, t_s: f64) -> Option<f64> {
    sample_nearest_point(trace, t_s).map(|point| point[1])
}

fn require_finite_points(group: &str, trace: &Trace) -> Result<()> {
    if trace.points.is_empty() {
        bail!("trace '{}' in {group} has no points", trace.name);
    }
    if let Some(point) = trace.points.iter().find(|point| !finite_point(point)) {
        bail!(
            "trace '{}' in {group} has non-finite point [{}, {}]",
            trace.name,
            point[0],
            point[1]
        );
    }
    Ok(())
}

fn finite_point(point: &[f64; 2]) -> bool {
    point[0].is_finite() && point[1].is_finite()
}

fn trace_names(traces: &[Trace]) -> Vec<&str> {
    traces.iter().map(|trace| trace.name.as_str()).collect()
}
