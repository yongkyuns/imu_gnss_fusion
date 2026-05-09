use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use sensor_fusion::reduced::{UPDATE_DIAG_TYPES, UpdateDiag};
use sensor_fusion::{Config, Filter, SensorFusion};

use crate::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, GenericReferencePositionSample, GenericReferenceRpySample,
    fusion_gnss_sample, fusion_imu_sample, load_gnss_samples, load_imu_samples,
    load_reference_attitude_samples, load_reference_mount_samples, load_reference_position_samples,
};
use crate::eval::gnss_ins::{as_q64, quat_angle_deg, quat_from_rpy_alg_deg, quat_mul, wrap_deg180};
use crate::eval::replay::{ReplayEvent, for_each_event};
use crate::visualizer::pipeline::reference::{
    q_bv_to_reference_mount_rpy, reference_mount_rpy_to_q_bv,
};

const REDUCED_UPDATE_NAMES: [&str; UPDATE_DIAG_TYPES] = [
    "gnss_pos",
    "gnss_vel",
    "zero_vel",
    "body_speed_x",
    "nhc_y",
    "nhc_z",
    "stationary_x",
    "stationary_y",
    "gnss_pos_d",
    "gnss_vel_d",
    "zero_vel_d",
];

const FULL_OBS_NAMES: [&str; 9] = [
    "none",
    "gnss_pos_x",
    "gnss_pos_y",
    "gnss_pos_z",
    "gnss_vel_x",
    "gnss_vel_y",
    "gnss_vel_z",
    "nhc_y",
    "nhc_z",
];

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub mount_threshold_deg: f64,
    pub attitude_threshold_deg: f64,
    pub start_after_s: f64,
    pub window_s: f64,
    pub sample_period_s: f64,
    pub max_time_s: Option<f64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            mount_threshold_deg: 2.0,
            attitude_threshold_deg: 2.0,
            start_after_s: 0.0,
            window_s: 10.0,
            sample_period_s: 0.5,
            max_time_s: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Report {
    pub input: String,
    pub samples: SampleCounts,
    pub reduced_init_t_s: Option<f64>,
    pub full_init_t_s: Option<f64>,
    pub align_ready_t_s: Option<f64>,
    pub first_crossing: Option<Crossing>,
    pub first_crossings: Vec<Crossing>,
    pub final_errors: Vec<ErrorSnapshot>,
    pub window_summaries: Vec<WindowSummary>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SampleCounts {
    pub imu: usize,
    pub gnss: usize,
    pub reference_attitude: usize,
    pub reference_mount: usize,
    pub reference_position: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Crossing {
    pub t_s: f64,
    pub source: &'static str,
    pub metric: &'static str,
    pub value_deg: f64,
    pub threshold_deg: f64,
}

#[derive(Clone, Debug)]
pub struct ErrorSnapshot {
    pub t_s: f64,
    pub source: &'static str,
    pub mount_qerr_deg: Option<f64>,
    pub mount_axis_err_deg: Option<[f64; 3]>,
    pub attitude_qerr_deg: Option<f64>,
    pub mount_sigma_deg: Option<[f64; 3]>,
    pub attitude_sigma_deg: Option<[f64; 3]>,
}

#[derive(Clone, Debug, Default)]
pub struct WindowSummary {
    pub source: &'static str,
    pub update: String,
    pub count: u32,
    pub sum_mount_dx_deg: [f64; 3],
    pub sum_abs_mount_dx_deg: [f64; 3],
    pub sum_att_dx_deg: [f64; 3],
    pub sum_abs_att_dx_deg: [f64; 3],
    pub sum_abs_accel_bias_dx: [f64; 3],
    pub sum_abs_gyro_bias_dx: [f64; 3],
    pub sum_abs_residual: f64,
    pub sum_nis: f64,
    pub max_nis: f64,
}

impl WindowSummary {
    pub fn mean_nis(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_nis / self.count as f64
        }
    }
}

#[derive(Clone, Debug)]
struct Replay {
    imu: Vec<GenericImuSample>,
    gnss: Vec<GenericGnssSample>,
    reference_attitude: Vec<GenericReferenceRpySample>,
    reference_mount: Vec<GenericReferenceRpySample>,
    reference_position: Vec<GenericReferencePositionSample>,
}

#[derive(Clone, Debug)]
struct AllocationEvent {
    t_s: f64,
    source: &'static str,
    update: &'static str,
    mount_dx_deg: [f64; 3],
    att_dx_deg: [f64; 3],
    accel_bias_dx: [f64; 3],
    gyro_bias_dx: [f64; 3],
    residual: Option<f64>,
    nis: Option<f64>,
}

pub fn run_generic_replay(dir: &Path, options: Options) -> Result<Report> {
    let replay = load_replay(dir)?;
    let samples = SampleCounts {
        imu: replay.imu.len(),
        gnss: replay.gnss.len(),
        reference_attitude: replay.reference_attitude.len(),
        reference_mount: replay.reference_mount.len(),
        reference_position: replay.reference_position.len(),
    };
    let final_ref_mount_rpy = final_reference_rpy(&replay.reference_mount);
    let final_ref_mount_q = final_ref_mount_rpy.map(reference_mount_rpy_to_q_bv);

    let mut reduced = SensorFusion::with_config(Config {
        filter: Filter::Reduced,
        ..Config::default()
    });
    let mut full = SensorFusion::with_config(Config {
        filter: Filter::Full,
        ..Config::default()
    });
    let mut prev_reduced_diag = UpdateDiag::default();
    let mut allocations = Vec::new();
    let mut snapshots = Vec::new();
    let mut reduced_init_t_s = None;
    let mut full_init_t_s = None;
    let mut align_ready_t_s = None;
    let mut next_snapshot_t_s = f64::NEG_INFINITY;

    for_each_event(&replay.imu, &replay.gnss, |event| match event {
        ReplayEvent::Imu(_, sample) => {
            if options.max_time_s.is_some_and(|max_t| sample.t_s > max_t) {
                return;
            }
            let _ = reduced.process_imu(fusion_imu_sample(*sample));
            let _ = full.process_imu(fusion_imu_sample(*sample));
            collect_reduced_allocations(
                sample.t_s,
                &reduced,
                &mut prev_reduced_diag,
                &mut allocations,
            );
            collect_full_allocations(sample.t_s, &full, &mut allocations);
            if sample.t_s >= next_snapshot_t_s {
                collect_snapshots(
                    sample.t_s,
                    &reduced,
                    &full,
                    &replay,
                    final_ref_mount_q,
                    final_ref_mount_rpy,
                    &mut snapshots,
                );
                next_snapshot_t_s = sample.t_s + options.sample_period_s.max(1.0e-3);
            }
        }
        ReplayEvent::Gnss(_, sample) => {
            if options.max_time_s.is_some_and(|max_t| sample.t_s > max_t) {
                return;
            }
            let reduced_update = reduced.process_gnss(fusion_gnss_sample(*sample));
            let full_update = full.process_gnss(fusion_gnss_sample(*sample));
            if reduced_update.mount_ready_changed && reduced_update.mount_ready {
                align_ready_t_s.get_or_insert(sample.t_s);
            }
            if reduced_update.filter_initialized_now {
                reduced_init_t_s.get_or_insert(sample.t_s);
            }
            if full_update.filter_initialized_now {
                full_init_t_s.get_or_insert(sample.t_s);
            }
        }
    });

    let first_crossings = first_crossings(&snapshots, &options);
    let first_crossing = first_crossings.first().cloned();
    let final_errors = last_snapshots_by_source(&snapshots);
    let allocation_anchor = first_crossings
        .iter()
        .find(|crossing| crossing.source != "Align")
        .or(first_crossing.as_ref());
    let window_summaries = if let Some(anchor) = allocation_anchor {
        summarize_window(
            &allocations,
            anchor.t_s - options.window_s,
            anchor.t_s + options.window_s,
        )
    } else {
        Vec::new()
    };

    Ok(Report {
        input: dir.display().to_string(),
        samples,
        reduced_init_t_s,
        full_init_t_s,
        align_ready_t_s,
        first_crossing,
        first_crossings,
        final_errors,
        window_summaries,
    })
}

fn load_replay(dir: &Path) -> Result<Replay> {
    Ok(Replay {
        imu: load_imu_samples(dir).with_context(|| format!("loading {}", dir.display()))?,
        gnss: load_gnss_samples(dir).with_context(|| format!("loading {}", dir.display()))?,
        reference_attitude: load_reference_attitude_samples(dir)
            .with_context(|| format!("loading {}", dir.display()))?,
        reference_mount: load_reference_mount_samples(dir)
            .with_context(|| format!("loading {}", dir.display()))?,
        reference_position: load_reference_position_samples(dir)
            .with_context(|| format!("loading {}", dir.display()))?,
    })
}

fn collect_reduced_allocations(
    t_s: f64,
    fusion: &SensorFusion,
    prev: &mut UpdateDiag,
    out: &mut Vec<AllocationEvent>,
) {
    let Some(state) = fusion.reduced() else {
        return;
    };
    let diag = state.update_diag;
    for i in 0..UPDATE_DIAG_TYPES {
        let count_delta = diag.type_counts[i].saturating_sub(prev.type_counts[i]);
        if count_delta == 0 {
            continue;
        }
        out.push(AllocationEvent {
            t_s,
            source: "Reduced",
            update: REDUCED_UPDATE_NAMES[i],
            mount_dx_deg: [
                delta(diag.sum_dx_mount_roll[i], prev.sum_dx_mount_roll[i]).to_degrees() as f64,
                delta(diag.sum_dx_mount_pitch[i], prev.sum_dx_mount_pitch[i]).to_degrees() as f64,
                delta(diag.sum_dx_mount_yaw[i], prev.sum_dx_mount_yaw[i]).to_degrees() as f64,
            ],
            att_dx_deg: [
                delta(diag.sum_dx_att_roll[i], prev.sum_dx_att_roll[i]).to_degrees() as f64,
                delta(diag.sum_dx_pitch[i], prev.sum_dx_pitch[i]).to_degrees() as f64,
                delta(diag.sum_dx_yaw[i], prev.sum_dx_yaw[i]).to_degrees() as f64,
            ],
            accel_bias_dx: [
                delta(diag.sum_dx_accel_bias[i][0], prev.sum_dx_accel_bias[i][0]) as f64,
                delta(diag.sum_dx_accel_bias[i][1], prev.sum_dx_accel_bias[i][1]) as f64,
                delta(diag.sum_dx_accel_bias[i][2], prev.sum_dx_accel_bias[i][2]) as f64,
            ],
            gyro_bias_dx: [
                delta(diag.sum_dx_gyro_bias[i][0], prev.sum_dx_gyro_bias[i][0]) as f64,
                delta(diag.sum_dx_gyro_bias[i][1], prev.sum_dx_gyro_bias[i][1]) as f64,
                delta(diag.sum_dx_gyro_bias[i][2], prev.sum_dx_gyro_bias[i][2]) as f64,
            ],
            residual: Some(delta(diag.sum_abs_innovation[i], prev.sum_abs_innovation[i]) as f64),
            nis: Some(delta(diag.sum_nis[i], prev.sum_nis[i]) as f64),
        });
    }
    *prev = diag;
}

fn collect_full_allocations(t_s: f64, fusion: &SensorFusion, out: &mut Vec<AllocationEvent>) {
    let Some(state) = fusion.full() else {
        return;
    };
    let count = state
        .last_obs_count
        .clamp(0, state.last_obs_types.len() as i32) as usize;
    for row in 0..count {
        let obs_type = state.last_obs_types[row].clamp(0, (FULL_OBS_NAMES.len() - 1) as i32);
        let dx = state.last_dx_by_obs[row];
        let residual = state.last_residuals[row] as f64;
        let innovation_var = state.last_innovation_vars[row].max(1.0e-12) as f64;
        out.push(AllocationEvent {
            t_s,
            source: "Full",
            update: FULL_OBS_NAMES[obs_type as usize],
            mount_dx_deg: [
                (dx[21] as f64).to_degrees(),
                (dx[22] as f64).to_degrees(),
                (dx[23] as f64).to_degrees(),
            ],
            att_dx_deg: [
                (dx[6] as f64).to_degrees(),
                (dx[7] as f64).to_degrees(),
                (dx[8] as f64).to_degrees(),
            ],
            accel_bias_dx: [dx[9] as f64, dx[10] as f64, dx[11] as f64],
            gyro_bias_dx: [dx[12] as f64, dx[13] as f64, dx[14] as f64],
            residual: Some(residual.abs()),
            nis: Some(residual * residual / innovation_var),
        });
    }
}

fn collect_snapshots(
    t_s: f64,
    reduced: &SensorFusion,
    full: &SensorFusion,
    replay: &Replay,
    final_ref_mount_q: Option<[f64; 4]>,
    final_ref_mount_rpy: Option<[f64; 3]>,
    out: &mut Vec<ErrorSnapshot>,
) {
    if let Some(align) = reduced.align() {
        let q = as_q64(align.q_bv);
        out.push(mount_snapshot(
            t_s,
            "Align",
            q,
            final_ref_mount_q,
            final_ref_mount_rpy,
            None,
        ));
    }
    if let Some(state) = reduced.reduced() {
        let q_mount = as_q64([
            state.nominal.qcs0,
            state.nominal.qcs1,
            state.nominal.qcs2,
            state.nominal.qcs3,
        ]);
        let q_att = as_q64([
            state.nominal.q0,
            state.nominal.q1,
            state.nominal.q2,
            state.nominal.q3,
        ]);
        let mut snapshot = mount_snapshot(
            t_s,
            "Reduced",
            q_mount,
            final_ref_mount_q,
            final_ref_mount_rpy,
            Some([
                (state.p[15][15].max(0.0) as f64).sqrt().to_degrees(),
                (state.p[16][16].max(0.0) as f64).sqrt().to_degrees(),
                (state.p[17][17].max(0.0) as f64).sqrt().to_degrees(),
            ]),
        );
        snapshot.attitude_qerr_deg =
            reference_attitude_q(replay, t_s).map(|q_ref| quat_angle_deg(q_att, q_ref));
        snapshot.attitude_sigma_deg = Some([
            (state.p[0][0].max(0.0) as f64).sqrt().to_degrees(),
            (state.p[1][1].max(0.0) as f64).sqrt().to_degrees(),
            (state.p[2][2].max(0.0) as f64).sqrt().to_degrees(),
        ]);
        out.push(snapshot);
    }
    if let Some(state) = full.full() {
        let q_mount = as_q64([
            state.nominal.qcs0,
            state.nominal.qcs1,
            state.nominal.qcs2,
            state.nominal.qcs3,
        ]);
        let mut snapshot = mount_snapshot(
            t_s,
            "Full",
            q_mount,
            final_ref_mount_q,
            final_ref_mount_rpy,
            Some([
                (state.p[21][21].max(0.0) as f64).sqrt().to_degrees(),
                (state.p[22][22].max(0.0) as f64).sqrt().to_degrees(),
                (state.p[23][23].max(0.0) as f64).sqrt().to_degrees(),
            ]),
        );
        if let (Some(q_ref), Some(q_ne)) = (
            reference_attitude_q(replay, t_s),
            reference_ecef_to_ned_q(replay, t_s),
        ) {
            let q_ev = as_q64([
                state.nominal.q0,
                state.nominal.q1,
                state.nominal.q2,
                state.nominal.q3,
            ]);
            snapshot.attitude_qerr_deg = Some(quat_angle_deg(quat_mul(q_ne, q_ev), q_ref));
        }
        snapshot.attitude_sigma_deg = Some([
            (state.p[6][6].max(0.0) as f64).sqrt().to_degrees(),
            (state.p[7][7].max(0.0) as f64).sqrt().to_degrees(),
            (state.p[8][8].max(0.0) as f64).sqrt().to_degrees(),
        ]);
        out.push(snapshot);
    }
}

fn mount_snapshot(
    t_s: f64,
    source: &'static str,
    q_mount: [f64; 4],
    final_ref_mount_q: Option<[f64; 4]>,
    final_ref_mount_rpy: Option<[f64; 3]>,
    mount_sigma_deg: Option<[f64; 3]>,
) -> ErrorSnapshot {
    let mount_rpy = q_bv_to_reference_mount_rpy(q_mount);
    ErrorSnapshot {
        t_s,
        source,
        mount_qerr_deg: final_ref_mount_q.map(|q_ref| quat_angle_deg(q_mount, q_ref)),
        mount_axis_err_deg: final_ref_mount_rpy.map(|rpy| {
            [
                wrap_deg180(mount_rpy.0 - rpy[0]),
                wrap_deg180(mount_rpy.1 - rpy[1]),
                wrap_deg180(mount_rpy.2 - rpy[2]),
            ]
        }),
        attitude_qerr_deg: None,
        mount_sigma_deg,
        attitude_sigma_deg: None,
    }
}

#[cfg(test)]
fn first_crossing(snapshots: &[ErrorSnapshot], options: &Options) -> Option<Crossing> {
    snapshots
        .iter()
        .filter(|snapshot| snapshot.t_s >= options.start_after_s)
        .find_map(|snapshot| {
            if let Some(value) = snapshot.mount_qerr_deg
                && value >= options.mount_threshold_deg
            {
                return Some(Crossing {
                    t_s: snapshot.t_s,
                    source: snapshot.source,
                    metric: "mount_qerr",
                    value_deg: value,
                    threshold_deg: options.mount_threshold_deg,
                });
            }
            if let Some(value) = snapshot.attitude_qerr_deg
                && value >= options.attitude_threshold_deg
            {
                return Some(Crossing {
                    t_s: snapshot.t_s,
                    source: snapshot.source,
                    metric: "attitude_qerr",
                    value_deg: value,
                    threshold_deg: options.attitude_threshold_deg,
                });
            }
            None
        })
}

fn first_crossings(snapshots: &[ErrorSnapshot], options: &Options) -> Vec<Crossing> {
    let mut out = Vec::new();
    for source in ["Align", "Reduced", "Full"] {
        if let Some(crossing) = snapshots
            .iter()
            .filter(|snapshot| snapshot.source == source && snapshot.t_s >= options.start_after_s)
            .find_map(|snapshot| {
                snapshot.mount_qerr_deg.and_then(|value| {
                    (value >= options.mount_threshold_deg).then_some(Crossing {
                        t_s: snapshot.t_s,
                        source: snapshot.source,
                        metric: "mount_qerr",
                        value_deg: value,
                        threshold_deg: options.mount_threshold_deg,
                    })
                })
            })
        {
            out.push(crossing);
        }
        if let Some(crossing) = snapshots
            .iter()
            .filter(|snapshot| snapshot.source == source && snapshot.t_s >= options.start_after_s)
            .find_map(|snapshot| {
                snapshot.attitude_qerr_deg.and_then(|value| {
                    (value >= options.attitude_threshold_deg).then_some(Crossing {
                        t_s: snapshot.t_s,
                        source: snapshot.source,
                        metric: "attitude_qerr",
                        value_deg: value,
                        threshold_deg: options.attitude_threshold_deg,
                    })
                })
            })
        {
            out.push(crossing);
        }
    }
    out.sort_by(|a, b| {
        a.t_s
            .total_cmp(&b.t_s)
            .then_with(|| a.source.cmp(b.source))
            .then_with(|| a.metric.cmp(b.metric))
    });
    out
}

fn last_snapshots_by_source(snapshots: &[ErrorSnapshot]) -> Vec<ErrorSnapshot> {
    let mut by_source = BTreeMap::<&'static str, ErrorSnapshot>::new();
    for snapshot in snapshots {
        by_source.insert(snapshot.source, snapshot.clone());
    }
    by_source.into_values().collect()
}

fn summarize_window(
    events: &[AllocationEvent],
    start_t_s: f64,
    end_t_s: f64,
) -> Vec<WindowSummary> {
    let mut summaries = BTreeMap::<(&'static str, &'static str), WindowSummary>::new();
    for event in events
        .iter()
        .filter(|event| event.t_s >= start_t_s && event.t_s <= end_t_s)
    {
        let entry = summaries
            .entry((event.source, event.update))
            .or_insert_with(|| WindowSummary {
                source: event.source,
                update: event.update.to_string(),
                ..WindowSummary::default()
            });
        entry.count += 1;
        add3(&mut entry.sum_mount_dx_deg, event.mount_dx_deg);
        add_abs3(&mut entry.sum_abs_mount_dx_deg, event.mount_dx_deg);
        add3(&mut entry.sum_att_dx_deg, event.att_dx_deg);
        add_abs3(&mut entry.sum_abs_att_dx_deg, event.att_dx_deg);
        add_abs3(&mut entry.sum_abs_accel_bias_dx, event.accel_bias_dx);
        add_abs3(&mut entry.sum_abs_gyro_bias_dx, event.gyro_bias_dx);
        entry.sum_abs_residual += event.residual.unwrap_or(0.0);
        let nis = event.nis.unwrap_or(0.0);
        entry.sum_nis += nis;
        entry.max_nis = entry.max_nis.max(nis);
    }
    summaries.into_values().collect()
}

fn reference_attitude_q(replay: &Replay, t_s: f64) -> Option<[f64; 4]> {
    nearest_rpy(&replay.reference_attitude, t_s)
        .map(|rpy| quat_from_rpy_alg_deg(rpy[0], rpy[1], rpy[2]))
}

fn reference_ecef_to_ned_q(replay: &Replay, t_s: f64) -> Option<[f64; 4]> {
    nearest_position(&replay.reference_position, t_s)
        .map(|sample| quat_ecef_to_ned(sample.lat_deg, sample.lon_deg))
        .or_else(|| {
            nearest_gnss(&replay.gnss, t_s)
                .map(|sample| quat_ecef_to_ned(sample.lat_deg, sample.lon_deg))
        })
}

fn final_reference_rpy(samples: &[GenericReferenceRpySample]) -> Option<[f64; 3]> {
    samples
        .iter()
        .rev()
        .find(|sample| {
            sample.roll_deg.is_finite()
                && sample.pitch_deg.is_finite()
                && sample.yaw_deg.is_finite()
        })
        .map(|sample| [sample.roll_deg, sample.pitch_deg, sample.yaw_deg])
}

fn nearest_rpy(samples: &[GenericReferenceRpySample], t_s: f64) -> Option<[f64; 3]> {
    nearest_by_time(samples, t_s, |sample| sample.t_s)
        .map(|sample| [sample.roll_deg, sample.pitch_deg, sample.yaw_deg])
}

fn nearest_position(
    samples: &[GenericReferencePositionSample],
    t_s: f64,
) -> Option<GenericReferencePositionSample> {
    nearest_by_time(samples, t_s, |sample| sample.t_s).copied()
}

fn nearest_gnss(samples: &[GenericGnssSample], t_s: f64) -> Option<GenericGnssSample> {
    nearest_by_time(samples, t_s, |sample| sample.t_s).copied()
}

fn nearest_by_time<T>(samples: &[T], t_s: f64, time: impl Fn(&T) -> f64) -> Option<&T> {
    if samples.is_empty() {
        return None;
    }
    let idx = samples.partition_point(|sample| time(sample) < t_s);
    match (idx.checked_sub(1), samples.get(idx)) {
        (Some(prev_idx), Some(next)) => {
            let prev = &samples[prev_idx];
            if (t_s - time(prev)).abs() <= (time(next) - t_s).abs() {
                Some(prev)
            } else {
                Some(next)
            }
        }
        (Some(prev_idx), None) => Some(&samples[prev_idx]),
        (None, Some(next)) => Some(next),
        (None, None) => None,
    }
}

fn quat_ecef_to_ned(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    dcm_to_quat(ecef_to_ned_matrix(lat_deg, lon_deg))
}

fn ecef_to_ned_matrix(lat_deg: f64, lon_deg: f64) -> [[f64; 3]; 3] {
    let (slat, clat) = lat_deg.to_radians().sin_cos();
    let (slon, clon) = lon_deg.to_radians().sin_cos();
    [
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [-clat * clon, -clat * slon, -slat],
    ]
}

fn dcm_to_quat(c: [[f64; 3]; 3]) -> [f64; 4] {
    let trace = c[0][0] + c[1][1] + c[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (c[2][1] - c[1][2]) / s,
            (c[0][2] - c[2][0]) / s,
            (c[1][0] - c[0][1]) / s,
        ]
    } else if c[0][0] > c[1][1] && c[0][0] > c[2][2] {
        let s = (1.0 + c[0][0] - c[1][1] - c[2][2]).sqrt() * 2.0;
        [
            (c[2][1] - c[1][2]) / s,
            0.25 * s,
            (c[0][1] + c[1][0]) / s,
            (c[0][2] + c[2][0]) / s,
        ]
    } else if c[1][1] > c[2][2] {
        let s = (1.0 + c[1][1] - c[0][0] - c[2][2]).sqrt() * 2.0;
        [
            (c[0][2] - c[2][0]) / s,
            (c[0][1] + c[1][0]) / s,
            0.25 * s,
            (c[1][2] + c[2][1]) / s,
        ]
    } else {
        let s = (1.0 + c[2][2] - c[0][0] - c[1][1]).sqrt() * 2.0;
        [
            (c[1][0] - c[0][1]) / s,
            (c[0][2] + c[2][0]) / s,
            (c[1][2] + c[2][1]) / s,
            0.25 * s,
        ]
    };
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 0.0 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

fn add3(dst: &mut [f64; 3], src: [f64; 3]) {
    for i in 0..3 {
        dst[i] += src[i];
    }
}

fn add_abs3(dst: &mut [f64; 3], src: [f64; 3]) {
    for i in 0..3 {
        dst[i] += src[i].abs();
    }
}

fn delta(new: f32, old: f32) -> f32 {
    new - old
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_crossing_respects_start_time_and_metric_order() {
        let snapshots = vec![
            ErrorSnapshot {
                t_s: 1.0,
                source: "Reduced",
                mount_qerr_deg: Some(5.0),
                mount_axis_err_deg: None,
                attitude_qerr_deg: Some(0.0),
                mount_sigma_deg: None,
                attitude_sigma_deg: None,
            },
            ErrorSnapshot {
                t_s: 3.0,
                source: "Full",
                mount_qerr_deg: Some(0.5),
                mount_axis_err_deg: None,
                attitude_qerr_deg: Some(4.0),
                mount_sigma_deg: None,
                attitude_sigma_deg: None,
            },
        ];
        let crossing = first_crossing(
            &snapshots,
            &Options {
                start_after_s: 2.0,
                mount_threshold_deg: 2.0,
                attitude_threshold_deg: 2.0,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(
            crossing,
            Crossing {
                t_s: 3.0,
                source: "Full",
                metric: "attitude_qerr",
                value_deg: 4.0,
                threshold_deg: 2.0,
            }
        );
    }

    #[test]
    fn summarize_window_groups_by_filter_and_update() {
        let events = vec![
            AllocationEvent {
                t_s: 10.0,
                source: "Reduced",
                update: "nhc_y",
                mount_dx_deg: [1.0, -2.0, 3.0],
                att_dx_deg: [0.1, 0.2, -0.3],
                accel_bias_dx: [0.0, 0.0, 0.1],
                gyro_bias_dx: [0.0, 0.2, 0.0],
                residual: Some(4.0),
                nis: Some(9.0),
            },
            AllocationEvent {
                t_s: 11.0,
                source: "Reduced",
                update: "nhc_y",
                mount_dx_deg: [-0.5, 0.5, 0.0],
                att_dx_deg: [0.0, 0.0, 0.1],
                accel_bias_dx: [0.2, 0.0, 0.0],
                gyro_bias_dx: [0.0, 0.0, 0.0],
                residual: Some(1.0),
                nis: Some(1.0),
            },
            AllocationEvent {
                t_s: 30.0,
                source: "Full",
                update: "gnss_vel_x",
                mount_dx_deg: [10.0, 0.0, 0.0],
                att_dx_deg: [0.0, 0.0, 0.0],
                accel_bias_dx: [0.0, 0.0, 0.0],
                gyro_bias_dx: [0.0, 0.0, 0.0],
                residual: Some(1.0),
                nis: Some(1.0),
            },
        ];
        let summaries = summarize_window(&events, 9.0, 12.0);
        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];
        assert_eq!(summary.source, "Reduced");
        assert_eq!(summary.update, "nhc_y");
        assert_eq!(summary.count, 2);
        assert_eq!(summary.sum_mount_dx_deg, [0.5, -1.5, 3.0]);
        assert_eq!(summary.sum_abs_mount_dx_deg, [1.5, 2.5, 3.0]);
        assert_eq!(summary.sum_abs_residual, 5.0);
        assert_eq!(summary.mean_nis(), 5.0);
        assert_eq!(summary.max_nis, 9.0);
    }
}
