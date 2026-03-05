use walkers::{Position, lon_lat};

use super::model::{PlotData, Trace};

pub fn trace_stats(data: &PlotData) -> (usize, usize) {
    let groups = [
        &data.speed,
        &data.sat_cn0,
        &data.imu_raw_gyro,
        &data.imu_raw_accel,
        &data.imu_cal_gyro,
        &data.imu_cal_accel,
        &data.esf_ins_gyro,
        &data.esf_ins_accel,
        &data.orientation,
        &data.other,
        &data.ekf_cmp_pos,
        &data.ekf_cmp_vel,
        &data.ekf_cmp_att,
        &data.ekf_bias_gyro,
        &data.ekf_bias_accel,
        &data.ekf_cov_bias,
        &data.ekf_cov_nonbias,
        &data.vma_cmp_att,
        &data.vma_res_vel,
        &data.vma_state_q,
        &data.vma_cov,
    ];
    let mut traces = 0usize;
    let mut points = 0usize;
    for g in groups {
        traces += g.len();
        points += g.iter().map(|t| t.points.len()).sum::<usize>();
    }
    (traces, points)
}

pub fn trace_time_bounds(data: &PlotData) -> Option<(f64, f64)> {
    let groups = [
        &data.speed,
        &data.sat_cn0,
        &data.imu_raw_gyro,
        &data.imu_raw_accel,
        &data.imu_cal_gyro,
        &data.imu_cal_accel,
        &data.esf_ins_gyro,
        &data.esf_ins_accel,
        &data.orientation,
        &data.other,
        &data.ekf_cmp_pos,
        &data.ekf_cmp_vel,
        &data.ekf_cmp_att,
        &data.ekf_bias_gyro,
        &data.ekf_bias_accel,
        &data.ekf_cov_bias,
        &data.ekf_cov_nonbias,
        &data.vma_cmp_att,
        &data.vma_res_vel,
        &data.vma_state_q,
        &data.vma_cov,
        &data.ekf_map,
    ];
    let mut min_t = f64::INFINITY;
    let mut max_t = f64::NEG_INFINITY;
    for g in groups {
        for tr in g {
            for p in &tr.points {
                min_t = min_t.min(p[0]);
                max_t = max_t.max(p[0]);
            }
        }
    }
    if min_t.is_finite() && max_t.is_finite() {
        Some((min_t, max_t))
    } else {
        None
    }
}

pub fn group_stats(name: &str, traces: &[Trace]) -> (String, usize, usize) {
    let n_traces = traces.len();
    let n_points = traces.iter().map(|t| t.points.len()).sum::<usize>();
    (name.to_string(), n_traces, n_points)
}

pub fn max_gap_sec(traces: &[Trace]) -> f64 {
    let mut worst = 0.0_f64;
    for tr in traces {
        let mut prev: Option<f64> = None;
        for p in &tr.points {
            if let Some(pt) = prev {
                let dt = p[0] - pt;
                if dt.is_finite() && dt > worst {
                    worst = dt;
                }
            }
            prev = Some(p[0]);
        }
    }
    worst
}

pub fn max_gap_trace(traces: &[Trace]) -> Option<(String, f64)> {
    let mut best: Option<(String, f64)> = None;
    for tr in traces {
        let mut local = 0.0_f64;
        let mut prev: Option<f64> = None;
        for p in &tr.points {
            if let Some(pt) = prev {
                let dt = p[0] - pt;
                if dt.is_finite() && dt > local {
                    local = dt;
                }
            }
            prev = Some(p[0]);
        }
        match &best {
            Some((_, b)) if *b >= local => {}
            _ => best = Some((tr.name.clone(), local)),
        }
    }
    best
}

pub fn trace_value_bounds(traces: &[Trace]) -> Option<(f64, f64)> {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for tr in traces {
        for p in &tr.points {
            let v = p[1];
            if v.is_finite() {
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
        }
    }
    if min_v.is_finite() && max_v.is_finite() {
        Some((min_v, max_v))
    } else {
        None
    }
}

pub fn max_step_abs(traces: &[Trace]) -> Option<f64> {
    let mut best = 0.0_f64;
    let mut any = false;
    for tr in traces {
        let mut prev: Option<f64> = None;
        for p in &tr.points {
            let v = p[1];
            if !v.is_finite() {
                continue;
            }
            if let Some(pv) = prev {
                best = best.max((v - pv).abs());
                any = true;
            }
            prev = Some(v);
        }
    }
    if any { Some(best) } else { None }
}

pub fn map_center_from_traces(traces: &[Trace]) -> Position {
    let mut n = 0usize;
    let mut lon = 0.0_f64;
    let mut lat = 0.0_f64;
    for tr in traces {
        for p in &tr.points {
            if p[0].is_finite() && p[1].is_finite() {
                lon += p[0];
                lat += p[1];
                n += 1;
            }
        }
    }
    if n == 0 {
        lon_lat(0.0, 0.0)
    } else {
        lon_lat(lon / n as f64, lat / n as f64)
    }
}
