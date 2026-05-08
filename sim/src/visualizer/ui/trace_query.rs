//! Pure trace lookup, filtering, interpolation, and UI-derived trace helpers.

use crate::visualizer::model::{PlotData, Trace};

pub(super) fn sample_trace_at(trace: &Trace, t_s: f64) -> Option<f64> {
    if !t_s.is_finite() || trace.points.is_empty() {
        return None;
    }
    let points = &trace.points;
    let idx = points.partition_point(|p| p[0] < t_s);
    if idx == 0 {
        return points.first().map(|p| p[1]);
    }
    if idx >= points.len() {
        return points.last().map(|p| p[1]);
    }
    let a = points[idx - 1];
    let b = points[idx];
    let dt = b[0] - a[0];
    if dt.abs() <= f64::EPSILON {
        return Some(b[1]);
    }
    let alpha = ((t_s - a[0]) / dt).clamp(0.0, 1.0);
    Some(a[1] + alpha * (b[1] - a[1]))
}

pub(super) fn find_trace_exact<'a>(traces: &'a [Trace], name: &str) -> Option<&'a Trace> {
    traces.iter().find(|trace| trace.name == name)
}

pub(super) fn trace_refs(traces: &[Trace]) -> Vec<&Trace> {
    traces.iter().filter(|t| !t.points.is_empty()).collect()
}

pub(super) fn concat_trace_refs<const N: usize>(groups: [&[Trace]; N]) -> Vec<&Trace> {
    let mut out = Vec::new();
    for group in groups {
        for trace in group {
            if trace.points.is_empty() || out.iter().any(|t: &&Trace| t.name == trace.name) {
                continue;
            }
            out.push(trace);
        }
    }
    out
}

pub(super) fn concat_trace_refs_matching<'a, const N: usize>(
    groups: [&'a [Trace]; N],
    tokens: &[&str],
) -> Vec<&'a Trace> {
    let mut out = Vec::new();
    for group in groups {
        for trace in group {
            if trace.points.is_empty()
                || !tokens.iter().any(|token| trace.name.contains(token))
                || out.iter().any(|t: &&Trace| t.name == trace.name)
            {
                continue;
            }
            out.push(trace);
        }
    }
    out
}

pub(super) fn trace_time_range<'a>(
    traces: impl IntoIterator<Item = &'a Trace>,
) -> Option<(f64, f64)> {
    traces
        .into_iter()
        .filter_map(|trace| {
            let start = trace
                .points
                .iter()
                .find_map(|p| p[0].is_finite().then_some(p[0]))?;
            let end = trace
                .points
                .iter()
                .rev()
                .find_map(|p| p[0].is_finite().then_some(p[0]))?;
            Some((start.min(end), start.max(end)))
        })
        .fold(None, |range, (start, end)| match range {
            Some((min_t, max_t)) => Some((f64::min(min_t, start), f64::max(max_t, end))),
            None => Some((start, end)),
        })
}

pub(super) fn mount_estimate_reference_traces<'a>(
    data: &'a PlotData,
    filter: &str,
    axis: &str,
) -> Option<(&'a Trace, &'a Trace)> {
    let estimate = match filter {
        "Reduced" => find_trace_exact(
            &data.reduced_misalignment,
            &format!("Reduced mount {axis} [deg]"),
        ),
        "Full" => find_trace_exact(
            &data.full_misalignment,
            &format!("Full residual mount {axis} [deg]"),
        ),
        _ => None,
    }?;
    let reference_name = format!("Reference mount {axis} [deg]");
    let reference = find_trace_exact(&data.reduced_misalignment, &reference_name)
        .or_else(|| find_trace_exact(&data.full_misalignment, &reference_name))
        .or_else(|| find_trace_exact(&data.align_cmp_att, &reference_name))?;
    Some((estimate, reference))
}

pub(super) fn vehicle_body_velocity_traces(data: &PlotData) -> Vec<Trace> {
    let velocity_groups = [
        data.reduced_cmp_vel.as_slice(),
        data.full_cmp_vel.as_slice(),
    ];
    let attitude_groups = [
        data.reduced_cmp_att.as_slice(),
        data.full_cmp_att.as_slice(),
        data.orientation.as_slice(),
    ];
    [
        BodyVelocityTraceSpec {
            label: "Reference",
            vel_n_names: &["Reference velN [m/s]", "Synthetic truth vN [m/s]"],
            vel_e_names: &["Reference velE [m/s]", "Synthetic truth vE [m/s]"],
            vel_d_names: &["Reference velD [m/s]", "Synthetic truth vD [m/s]"],
            roll_names: &["Reference roll [deg]", "Synthetic truth roll [deg]"],
            pitch_names: &["Reference pitch [deg]", "Synthetic truth pitch [deg]"],
            yaw_names: &["Reference yaw [deg]", "Synthetic truth yaw [deg]"],
        },
        BodyVelocityTraceSpec {
            label: "Reduced",
            vel_n_names: &["Reduced velN [m/s]", "Reduced vN [m/s]"],
            vel_e_names: &["Reduced velE [m/s]", "Reduced vE [m/s]"],
            vel_d_names: &["Reduced velD [m/s]", "Reduced vD [m/s]"],
            roll_names: &["Reduced roll [deg]"],
            pitch_names: &["Reduced pitch [deg]"],
            yaw_names: &["Reduced yaw [deg]"],
        },
        BodyVelocityTraceSpec {
            label: "Full",
            vel_n_names: &["Full velN [m/s]"],
            vel_e_names: &["Full velE [m/s]"],
            vel_d_names: &["Full velD [m/s]"],
            roll_names: &["Full roll [deg]"],
            pitch_names: &["Full pitch [deg]"],
            yaw_names: &["Full yaw [deg]"],
        },
    ]
    .into_iter()
    .flat_map(|spec| body_velocity_trace_set(spec, &velocity_groups, &attitude_groups))
    .collect()
}

struct BodyVelocityTraceSpec<'a> {
    label: &'static str,
    vel_n_names: &'a [&'static str],
    vel_e_names: &'a [&'static str],
    vel_d_names: &'a [&'static str],
    roll_names: &'a [&'static str],
    pitch_names: &'a [&'static str],
    yaw_names: &'a [&'static str],
}

fn body_velocity_trace_set(
    spec: BodyVelocityTraceSpec<'_>,
    velocity_groups: &[&[Trace]],
    attitude_groups: &[&[Trace]],
) -> Vec<Trace> {
    let Some(vel_n) = find_trace_by_names(velocity_groups, spec.vel_n_names) else {
        return Vec::new();
    };
    let Some(vel_e) = find_trace_by_names(velocity_groups, spec.vel_e_names) else {
        return Vec::new();
    };
    let Some(vel_d) = find_trace_by_names(velocity_groups, spec.vel_d_names) else {
        return Vec::new();
    };
    let Some(roll) = find_trace_by_names(attitude_groups, spec.roll_names) else {
        return Vec::new();
    };
    let Some(pitch) = find_trace_by_names(attitude_groups, spec.pitch_names) else {
        return Vec::new();
    };
    let Some(yaw) = find_trace_by_names(attitude_groups, spec.yaw_names) else {
        return Vec::new();
    };

    let mut longitudinal = Vec::new();
    let mut lateral = Vec::new();
    let mut vertical = Vec::new();
    for p in &vel_n.points {
        let t_s = p[0];
        let Some(ve) = sample_trace_at(vel_e, t_s) else {
            continue;
        };
        let Some(vd) = sample_trace_at(vel_d, t_s) else {
            continue;
        };
        let Some(roll_deg) = sample_trace_at(roll, t_s) else {
            continue;
        };
        let Some(pitch_deg) = sample_trace_at(pitch, t_s) else {
            continue;
        };
        let Some(yaw_deg) = sample_trace_at(yaw, t_s) else {
            continue;
        };
        let body = ned_velocity_to_vehicle_body([p[1], ve, vd], roll_deg, pitch_deg, yaw_deg);
        longitudinal.push([t_s, body[0]]);
        lateral.push([t_s, body[1]]);
        vertical.push([t_s, body[2]]);
    }
    [
        Trace {
            name: format!("{} longitudinal [m/s]", spec.label),
            points: longitudinal,
        },
        Trace {
            name: format!("{} lateral [m/s]", spec.label),
            points: lateral,
        },
        Trace {
            name: format!("{} vertical [m/s]", spec.label),
            points: vertical,
        },
    ]
    .into_iter()
    .filter(|trace| !trace.points.is_empty())
    .collect()
}

fn find_trace_by_names<'a>(groups: &[&'a [Trace]], names: &[&str]) -> Option<&'a Trace> {
    groups
        .iter()
        .find_map(|group| names.iter().find_map(|name| find_trace_exact(group, name)))
}

fn ned_velocity_to_vehicle_body(
    vel_ned_mps: [f64; 3],
    roll_deg: f64,
    pitch_deg: f64,
    yaw_deg: f64,
) -> [f64; 3] {
    let (sr, cr) = roll_deg.to_radians().sin_cos();
    let (sp, cp) = pitch_deg.to_radians().sin_cos();
    let (sy, cy) = yaw_deg.to_radians().sin_cos();

    let c_nb = [
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ];
    [
        c_nb[0][0] * vel_ned_mps[0] + c_nb[1][0] * vel_ned_mps[1] + c_nb[2][0] * vel_ned_mps[2],
        c_nb[0][1] * vel_ned_mps[0] + c_nb[1][1] * vel_ned_mps[1] + c_nb[2][1] * vel_ned_mps[2],
        c_nb[0][2] * vel_ned_mps[0] + c_nb[1][2] * vel_ned_mps[1] + c_nb[2][2] * vel_ned_mps[2],
    ]
}

pub(super) fn attitude_error_traces(data: &PlotData, axis: &str) -> Vec<Trace> {
    let Some(reference) = reference_attitude_trace(data, axis) else {
        return Vec::new();
    };
    [
        (
            "Reduced",
            find_trace_exact(&data.reduced_cmp_att, &format!("Reduced {axis} [deg]")),
        ),
        (
            "Full",
            find_trace_exact(&data.full_cmp_att, &format!("Full {axis} [deg]")),
        ),
    ]
    .into_iter()
    .filter_map(|(system, estimate)| {
        estimate.and_then(|estimate| attitude_error_trace(system, axis, estimate, reference))
    })
    .collect()
}

fn reference_attitude_trace<'a>(data: &'a PlotData, axis: &str) -> Option<&'a Trace> {
    let reference_name = format!("Reference {axis} [deg]");
    let synthetic_name = format!("Synthetic truth {axis} [deg]");
    [
        data.reduced_cmp_att.as_slice(),
        data.full_cmp_att.as_slice(),
        data.orientation.as_slice(),
    ]
    .into_iter()
    .find_map(|group| {
        find_trace_exact(group, &reference_name)
            .or_else(|| find_trace_exact(group, &synthetic_name))
    })
}

fn attitude_error_trace(
    system: &str,
    axis: &str,
    estimate: &Trace,
    reference: &Trace,
) -> Option<Trace> {
    let points: Vec<[f64; 2]> = estimate
        .points
        .iter()
        .filter_map(|p| {
            let [t_s, estimate_deg] = *p;
            if !t_s.is_finite() || !estimate_deg.is_finite() {
                return None;
            }
            sample_trace_at(reference, t_s)
                .filter(|reference_deg| reference_deg.is_finite())
                .map(|reference_deg| [t_s, wrap_degrees(estimate_deg - reference_deg)])
        })
        .collect();
    (points.len() >= 2).then(|| Trace {
        name: format!("{system} {axis} error [deg]"),
        points,
    })
}

pub(super) fn wrap_degrees(value: f64) -> f64 {
    (value + 180.0).rem_euclid(360.0) - 180.0
}
