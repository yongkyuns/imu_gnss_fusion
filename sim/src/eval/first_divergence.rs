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
use crate::eval::gnss_ins::{
    as_q64, quat_angle_deg, quat_conj, quat_from_rpy_alg_deg, quat_mul, wrap_deg180,
};
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
    pub behavior_samples: Vec<BehaviorSample>,
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
pub struct BehaviorSample {
    pub t_s: f64,
    pub interval_s: f64,
    pub motion_regime: &'static str,
    pub gnss_speed_mps: f64,
    pub gnss_course_rate_dps: f64,
    pub gnss_speed_rate_mps2: f64,
    pub imu_gyro_norm_dps: f64,
    pub imu_gyro_z_dps: f64,
    pub imu_accel_norm_err_mps2: f64,
    pub reference_mount_rpy_deg: Option<[f64; 3]>,
    pub reference_mount_delta_deg: Option<[f64; 3]>,
    pub reference_mount_delta_q_deg: Option<f64>,
    pub reference_mount_delta_vec_deg: Option<[f64; 3]>,
    pub align_mount_rpy_deg: Option<[f64; 3]>,
    pub align_mount_delta_deg: Option<[f64; 3]>,
    pub align_mount_delta_q_deg: Option<f64>,
    pub align_mount_delta_vec_deg: Option<[f64; 3]>,
    pub align_mount_sigma_deg: Option<[f64; 3]>,
    pub align_horiz_count: u32,
    pub align_turn_gyro_count: u32,
    pub align_horiz_delta_q_deg: Option<f64>,
    pub align_horiz_delta_vec_deg: Option<[f64; 3]>,
    pub align_turn_gyro_delta_q_deg: Option<f64>,
    pub align_turn_gyro_delta_vec_deg: Option<[f64; 3]>,
    pub align_horiz_angle_err_deg: Option<f64>,
    pub align_horiz_effective_std_deg: Option<f64>,
    pub align_horiz_speed_q: Option<f64>,
    pub align_horiz_accel_q: Option<f64>,
    pub align_horiz_turn_q: Option<f64>,
    pub align_horiz_straight_q: Option<f64>,
    pub align_horiz_turn_core_valid: bool,
    pub align_horiz_straight_core_valid: bool,
    pub align_horiz_obs_accel_vx: Option<f64>,
    pub align_horiz_obs_accel_vy: Option<f64>,
    pub align_horiz_gnss_norm_mps2: Option<f64>,
    pub align_horiz_imu_norm_mps2: Option<f64>,
    pub reduced_mount_rpy_deg: Option<[f64; 3]>,
    pub reduced_mount_delta_deg: Option<[f64; 3]>,
    pub reduced_mount_delta_q_deg: Option<f64>,
    pub reduced_mount_delta_vec_deg: Option<[f64; 3]>,
    pub reduced_mount_error_deg: Option<[f64; 3]>,
    pub reduced_mount_sigma_deg: Option<[f64; 3]>,
    pub reduced_attitude_qerr_deg: Option<f64>,
    pub full_mount_rpy_deg: Option<[f64; 3]>,
    pub full_mount_delta_deg: Option<[f64; 3]>,
    pub full_mount_delta_q_deg: Option<f64>,
    pub full_mount_delta_vec_deg: Option<[f64; 3]>,
    pub full_mount_error_deg: Option<[f64; 3]>,
    pub full_mount_sigma_deg: Option<[f64; 3]>,
    pub full_attitude_qerr_deg: Option<f64>,
    pub reduced_gnss_residual_abs: f64,
    pub reduced_nhc_y_residual_abs: f64,
    pub reduced_nhc_z_residual_abs: f64,
    pub full_gnss_residual_abs: f64,
    pub full_nhc_y_residual_abs: f64,
    pub full_nhc_z_residual_abs: f64,
    pub reduced_gnss_mount_dx_deg: [f64; 3],
    pub reduced_nhc_mount_dx_deg: [f64; 3],
    pub full_gnss_mount_dx_deg: [f64; 3],
    pub full_nhc_mount_dx_deg: [f64; 3],
}

#[derive(Clone, Copy, Debug, Default)]
struct PreviousBehavior {
    t_s: Option<f64>,
    gnss_t_s: Option<f64>,
    gnss_speed_mps: Option<f64>,
    gnss_course_rad: Option<f64>,
    reference_mount_rpy_deg: Option<[f64; 3]>,
    reference_mount_q_bv: Option<[f64; 4]>,
    align_mount_rpy_deg: Option<[f64; 3]>,
    align_mount_q_bv: Option<[f64; 4]>,
    reduced_mount_rpy_deg: Option<[f64; 3]>,
    reduced_mount_q_bv: Option<[f64; 4]>,
    full_mount_rpy_deg: Option<[f64; 3]>,
    full_mount_q_bv: Option<[f64; 4]>,
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

#[derive(Clone, Debug)]
struct AlignEvent {
    horiz_delta_q_deg: Option<f64>,
    horiz_delta_vec_deg: Option<[f64; 3]>,
    turn_gyro_delta_q_deg: Option<f64>,
    turn_gyro_delta_vec_deg: Option<[f64; 3]>,
    horiz_angle_err_deg: Option<f64>,
    horiz_effective_std_deg: Option<f64>,
    horiz_speed_q: Option<f64>,
    horiz_accel_q: Option<f64>,
    horiz_turn_q: Option<f64>,
    horiz_straight_q: Option<f64>,
    horiz_turn_core_valid: bool,
    horiz_straight_core_valid: bool,
    horiz_obs_accel_vx: Option<f64>,
    horiz_obs_accel_vy: Option<f64>,
    horiz_gnss_norm_mps2: Option<f64>,
    horiz_imu_norm_mps2: Option<f64>,
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
    let mut behavior_samples = Vec::new();
    let mut reduced_init_t_s = None;
    let mut full_init_t_s = None;
    let mut align_ready_t_s = None;
    let mut align_events = Vec::new();
    let mut next_snapshot_t_s = f64::NEG_INFINITY;
    let mut prev_behavior = PreviousBehavior::default();
    let mut behavior_allocation_start = 0usize;
    let mut behavior_align_start = 0usize;

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
                behavior_samples.push(collect_behavior_sample(
                    sample.t_s,
                    sample,
                    &reduced,
                    &full,
                    &replay,
                    &allocations[behavior_allocation_start..],
                    &align_events[behavior_align_start..],
                    &mut prev_behavior,
                ));
                behavior_allocation_start = allocations.len();
                behavior_align_start = align_events.len();
                next_snapshot_t_s = sample.t_s + options.sample_period_s.max(1.0e-3);
            }
        }
        ReplayEvent::Gnss(_, sample) => {
            if options.max_time_s.is_some_and(|max_t| sample.t_s > max_t) {
                return;
            }
            let reduced_update = reduced.process_gnss(fusion_gnss_sample(*sample));
            let full_update = full.process_gnss(fusion_gnss_sample(*sample));
            collect_align_event(&reduced, &mut align_events);
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
        behavior_samples,
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

fn collect_align_event(fusion: &SensorFusion, out: &mut Vec<AlignEvent>) {
    let Some(debug) = fusion.align_debug() else {
        return;
    };
    let q_start = Some(as_q64(debug.trace.q_start));
    let after_gravity = debug.trace.after_gravity.map(as_q64);
    let after_horiz = debug.trace.after_horiz_accel.map(as_q64);
    let after_turn = debug.trace.after_turn_gyro.map(as_q64);
    let horiz_prev = after_gravity.or(q_start);
    let turn_prev = after_horiz.or(after_gravity).or(q_start);
    out.push(AlignEvent {
        horiz_delta_q_deg: delta_quat_angle_deg(after_horiz, horiz_prev),
        horiz_delta_vec_deg: delta_quat_vec_deg(after_horiz, horiz_prev),
        turn_gyro_delta_q_deg: delta_quat_angle_deg(after_turn, turn_prev),
        turn_gyro_delta_vec_deg: delta_quat_vec_deg(after_turn, turn_prev),
        horiz_angle_err_deg: debug
            .trace
            .horiz_angle_err_rad
            .map(|v| v.to_degrees() as f64),
        horiz_effective_std_deg: debug
            .trace
            .horiz_effective_std_rad
            .map(|v| v.to_degrees() as f64),
        horiz_speed_q: debug.trace.horiz_speed_q.map(f64::from),
        horiz_accel_q: debug.trace.horiz_accel_q.map(f64::from),
        horiz_turn_q: debug.trace.horiz_turn_q.map(f64::from),
        horiz_straight_q: debug.trace.horiz_straight_q.map(f64::from),
        horiz_turn_core_valid: debug.trace.horiz_turn_core_valid,
        horiz_straight_core_valid: debug.trace.horiz_straight_core_valid,
        horiz_obs_accel_vx: debug.trace.horiz_obs_accel_vx.map(f64::from),
        horiz_obs_accel_vy: debug.trace.horiz_obs_accel_vy.map(f64::from),
        horiz_gnss_norm_mps2: debug.trace.horiz_gnss_norm_mps2.map(f64::from),
        horiz_imu_norm_mps2: debug.trace.horiz_imu_norm_mps2.map(f64::from),
    });
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
            state.nominal.q_bv0,
            state.nominal.q_bv1,
            state.nominal.q_bv2,
            state.nominal.q_bv3,
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
            state.nominal.q_bv0,
            state.nominal.q_bv1,
            state.nominal.q_bv2,
            state.nominal.q_bv3,
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

fn collect_behavior_sample(
    t_s: f64,
    imu: &GenericImuSample,
    reduced: &SensorFusion,
    full: &SensorFusion,
    replay: &Replay,
    interval_events: &[AllocationEvent],
    interval_align_events: &[AlignEvent],
    prev: &mut PreviousBehavior,
) -> BehaviorSample {
    let reference_mount_rpy = nearest_rpy(&replay.reference_mount, t_s);
    let reference_mount_q = reference_mount_rpy.map(reference_mount_rpy_to_q_bv);
    let align_mount_rpy = reduced.align().map(|align| {
        let (r, p, y) = q_bv_to_reference_mount_rpy(as_q64(align.q_bv));
        [r, p, y]
    });
    let align_mount_q = reduced.align().map(|align| as_q64(align.q_bv));
    let align_mount_sigma = reduced.align().map(|align| {
        [
            (align.P[0][0].max(0.0) as f64).sqrt().to_degrees(),
            (align.P[1][1].max(0.0) as f64).sqrt().to_degrees(),
            (align.P[2][2].max(0.0) as f64).sqrt().to_degrees(),
        ]
    });
    let reduced_mount_q = reduced.reduced().map(|state| {
        as_q64([
            state.nominal.q_bv0,
            state.nominal.q_bv1,
            state.nominal.q_bv2,
            state.nominal.q_bv3,
        ])
    });
    let reduced_mount_rpy = reduced_mount_q.map(|q| {
        let (r, p, y) = q_bv_to_reference_mount_rpy(q);
        [r, p, y]
    });
    let reduced_mount_sigma = reduced.reduced().map(|state| {
        [
            (state.p[15][15].max(0.0) as f64).sqrt().to_degrees(),
            (state.p[16][16].max(0.0) as f64).sqrt().to_degrees(),
            (state.p[17][17].max(0.0) as f64).sqrt().to_degrees(),
        ]
    });
    let reduced_attitude_qerr = reduced.reduced().and_then(|state| {
        reference_attitude_q(replay, t_s).map(|q_ref| {
            quat_angle_deg(
                as_q64([
                    state.nominal.q0,
                    state.nominal.q1,
                    state.nominal.q2,
                    state.nominal.q3,
                ]),
                q_ref,
            )
        })
    });
    let full_mount_q = full.full().map(|state| {
        as_q64([
            state.nominal.q_bv0,
            state.nominal.q_bv1,
            state.nominal.q_bv2,
            state.nominal.q_bv3,
        ])
    });
    let full_mount_rpy = full_mount_q.map(|q| {
        let (r, p, y) = q_bv_to_reference_mount_rpy(q);
        [r, p, y]
    });
    let full_mount_sigma = full.full().map(|state| {
        [
            (state.p[21][21].max(0.0) as f64).sqrt().to_degrees(),
            (state.p[22][22].max(0.0) as f64).sqrt().to_degrees(),
            (state.p[23][23].max(0.0) as f64).sqrt().to_degrees(),
        ]
    });
    let full_attitude_qerr = full.full().and_then(|state| {
        let q_ref = reference_attitude_q(replay, t_s)?;
        let q_ne = reference_ecef_to_ned_q(replay, t_s)?;
        Some(quat_angle_deg(
            quat_mul(
                q_ne,
                as_q64([
                    state.nominal.q0,
                    state.nominal.q1,
                    state.nominal.q2,
                    state.nominal.q3,
                ]),
            ),
            q_ref,
        ))
    });

    let gnss = nearest_gnss(&replay.gnss, t_s);
    let gnss_speed_mps = gnss.map_or(f64::NAN, |s| horizontal_speed(s.vel_ned_mps));
    let gnss_course_rad = gnss.map(|s| s.vel_ned_mps[1].atan2(s.vel_ned_mps[0]));
    let (gnss_course_rate_dps, gnss_speed_rate_mps2) =
        gnss_motion_rates(gnss, gnss_course_rad, prev);
    let imu_gyro_norm_dps = norm3(imu.gyro_radps).to_degrees();
    let imu_gyro_z_dps = imu.gyro_radps[2].to_degrees();
    let imu_accel_norm_err_mps2 = (norm3(imu.accel_mps2) - 9.80665).abs();
    let interval_s = prev.t_s.map_or(0.0, |prev_t| (t_s - prev_t).max(0.0));
    let interval_summary = BehaviorAllocationSummary::from_events(interval_events);
    let align_summary = AlignBehaviorSummary::from_events(interval_align_events);
    let motion_regime = classify_motion(
        gnss_speed_mps,
        gnss_course_rate_dps,
        gnss_speed_rate_mps2,
        imu_accel_norm_err_mps2,
    );

    let sample = BehaviorSample {
        t_s,
        interval_s,
        motion_regime,
        gnss_speed_mps,
        gnss_course_rate_dps,
        gnss_speed_rate_mps2,
        imu_gyro_norm_dps,
        imu_gyro_z_dps,
        imu_accel_norm_err_mps2,
        reference_mount_rpy_deg: reference_mount_rpy,
        reference_mount_delta_deg: delta_rpy(reference_mount_rpy, prev.reference_mount_rpy_deg),
        reference_mount_delta_q_deg: delta_quat_angle_deg(
            reference_mount_q,
            prev.reference_mount_q_bv,
        ),
        reference_mount_delta_vec_deg: delta_quat_vec_deg(
            reference_mount_q,
            prev.reference_mount_q_bv,
        ),
        align_mount_rpy_deg: align_mount_rpy,
        align_mount_delta_deg: delta_rpy(align_mount_rpy, prev.align_mount_rpy_deg),
        align_mount_delta_q_deg: delta_quat_angle_deg(align_mount_q, prev.align_mount_q_bv),
        align_mount_delta_vec_deg: delta_quat_vec_deg(align_mount_q, prev.align_mount_q_bv),
        align_mount_sigma_deg: align_mount_sigma,
        align_horiz_count: align_summary.horiz_count,
        align_turn_gyro_count: align_summary.turn_gyro_count,
        align_horiz_delta_q_deg: align_summary.horiz_delta_q_deg(),
        align_horiz_delta_vec_deg: align_summary.horiz_delta_vec_deg(),
        align_turn_gyro_delta_q_deg: align_summary.turn_gyro_delta_q_deg(),
        align_turn_gyro_delta_vec_deg: align_summary.turn_gyro_delta_vec_deg(),
        align_horiz_angle_err_deg: align_summary.mean_horiz_angle_err_deg(),
        align_horiz_effective_std_deg: align_summary.mean_horiz_effective_std_deg(),
        align_horiz_speed_q: align_summary.mean_horiz_speed_q(),
        align_horiz_accel_q: align_summary.mean_horiz_accel_q(),
        align_horiz_turn_q: align_summary.mean_horiz_turn_q(),
        align_horiz_straight_q: align_summary.mean_horiz_straight_q(),
        align_horiz_turn_core_valid: align_summary.turn_core_valid_count > 0,
        align_horiz_straight_core_valid: align_summary.straight_core_valid_count > 0,
        align_horiz_obs_accel_vx: align_summary.mean_horiz_obs_accel_vx(),
        align_horiz_obs_accel_vy: align_summary.mean_horiz_obs_accel_vy(),
        align_horiz_gnss_norm_mps2: align_summary.mean_horiz_gnss_norm_mps2(),
        align_horiz_imu_norm_mps2: align_summary.mean_horiz_imu_norm_mps2(),
        reduced_mount_rpy_deg: reduced_mount_rpy,
        reduced_mount_delta_deg: delta_rpy(reduced_mount_rpy, prev.reduced_mount_rpy_deg),
        reduced_mount_delta_q_deg: delta_quat_angle_deg(reduced_mount_q, prev.reduced_mount_q_bv),
        reduced_mount_delta_vec_deg: delta_quat_vec_deg(reduced_mount_q, prev.reduced_mount_q_bv),
        reduced_mount_error_deg: error_rpy(reduced_mount_rpy, reference_mount_rpy),
        reduced_mount_sigma_deg: reduced_mount_sigma,
        reduced_attitude_qerr_deg: reduced_attitude_qerr,
        full_mount_rpy_deg: full_mount_rpy,
        full_mount_delta_deg: delta_rpy(full_mount_rpy, prev.full_mount_rpy_deg),
        full_mount_delta_q_deg: delta_quat_angle_deg(full_mount_q, prev.full_mount_q_bv),
        full_mount_delta_vec_deg: delta_quat_vec_deg(full_mount_q, prev.full_mount_q_bv),
        full_mount_error_deg: error_rpy(full_mount_rpy, reference_mount_rpy),
        full_mount_sigma_deg: full_mount_sigma,
        full_attitude_qerr_deg: full_attitude_qerr,
        reduced_gnss_residual_abs: interval_summary.reduced_gnss_residual_abs,
        reduced_nhc_y_residual_abs: interval_summary.reduced_nhc_y_residual_abs,
        reduced_nhc_z_residual_abs: interval_summary.reduced_nhc_z_residual_abs,
        full_gnss_residual_abs: interval_summary.full_gnss_residual_abs,
        full_nhc_y_residual_abs: interval_summary.full_nhc_y_residual_abs,
        full_nhc_z_residual_abs: interval_summary.full_nhc_z_residual_abs,
        reduced_gnss_mount_dx_deg: interval_summary.reduced_gnss_mount_dx_deg,
        reduced_nhc_mount_dx_deg: interval_summary.reduced_nhc_mount_dx_deg,
        full_gnss_mount_dx_deg: interval_summary.full_gnss_mount_dx_deg,
        full_nhc_mount_dx_deg: interval_summary.full_nhc_mount_dx_deg,
    };

    prev.t_s = Some(t_s);
    if let Some(gnss) = gnss {
        prev.gnss_t_s = Some(gnss.t_s);
        prev.gnss_speed_mps = Some(gnss_speed_mps);
        prev.gnss_course_rad = gnss_course_rad;
    }
    prev.reference_mount_rpy_deg = reference_mount_rpy;
    prev.reference_mount_q_bv = reference_mount_q;
    prev.align_mount_rpy_deg = align_mount_rpy;
    prev.align_mount_q_bv = align_mount_q;
    prev.reduced_mount_rpy_deg = reduced_mount_rpy;
    prev.reduced_mount_q_bv = reduced_mount_q;
    prev.full_mount_rpy_deg = full_mount_rpy;
    prev.full_mount_q_bv = full_mount_q;
    sample
}

#[derive(Clone, Copy, Debug, Default)]
struct OptionalStats {
    count: u32,
    sum: f64,
}

impl OptionalStats {
    fn push(&mut self, value: Option<f64>) {
        if let Some(value) = value {
            self.count += 1;
            self.sum += value;
        }
    }

    fn mean(self) -> Option<f64> {
        (self.count > 0).then_some(self.sum / self.count as f64)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct AlignBehaviorSummary {
    horiz_count: u32,
    turn_gyro_count: u32,
    horiz_delta_vec_deg: [f64; 3],
    horiz_delta_q_deg: f64,
    turn_gyro_delta_vec_deg: [f64; 3],
    turn_gyro_delta_q_deg: f64,
    horiz_angle_err_deg: OptionalStats,
    horiz_effective_std_deg: OptionalStats,
    horiz_speed_q: OptionalStats,
    horiz_accel_q: OptionalStats,
    horiz_turn_q: OptionalStats,
    horiz_straight_q: OptionalStats,
    turn_core_valid_count: u32,
    straight_core_valid_count: u32,
    horiz_obs_accel_vx: OptionalStats,
    horiz_obs_accel_vy: OptionalStats,
    horiz_gnss_norm_mps2: OptionalStats,
    horiz_imu_norm_mps2: OptionalStats,
}

impl AlignBehaviorSummary {
    fn from_events(events: &[AlignEvent]) -> Self {
        let mut summary = Self::default();
        for event in events {
            if let Some(delta) = event.horiz_delta_q_deg {
                summary.horiz_count += 1;
                summary.horiz_delta_q_deg += delta;
            }
            if let Some(delta) = event.horiz_delta_vec_deg {
                add3(&mut summary.horiz_delta_vec_deg, delta);
            }
            if let Some(delta) = event.turn_gyro_delta_q_deg {
                summary.turn_gyro_count += 1;
                summary.turn_gyro_delta_q_deg += delta;
            }
            if let Some(delta) = event.turn_gyro_delta_vec_deg {
                add3(&mut summary.turn_gyro_delta_vec_deg, delta);
            }
            summary.horiz_angle_err_deg.push(event.horiz_angle_err_deg);
            summary
                .horiz_effective_std_deg
                .push(event.horiz_effective_std_deg);
            summary.horiz_speed_q.push(event.horiz_speed_q);
            summary.horiz_accel_q.push(event.horiz_accel_q);
            summary.horiz_turn_q.push(event.horiz_turn_q);
            summary.horiz_straight_q.push(event.horiz_straight_q);
            summary.horiz_obs_accel_vx.push(event.horiz_obs_accel_vx);
            summary.horiz_obs_accel_vy.push(event.horiz_obs_accel_vy);
            summary
                .horiz_gnss_norm_mps2
                .push(event.horiz_gnss_norm_mps2);
            summary.horiz_imu_norm_mps2.push(event.horiz_imu_norm_mps2);
            summary.turn_core_valid_count += u32::from(event.horiz_turn_core_valid);
            summary.straight_core_valid_count += u32::from(event.horiz_straight_core_valid);
        }
        summary
    }

    fn horiz_delta_q_deg(&self) -> Option<f64> {
        (self.horiz_count > 0).then_some(self.horiz_delta_q_deg)
    }

    fn horiz_delta_vec_deg(&self) -> Option<[f64; 3]> {
        (self.horiz_count > 0).then_some(self.horiz_delta_vec_deg)
    }

    fn turn_gyro_delta_q_deg(&self) -> Option<f64> {
        (self.turn_gyro_count > 0).then_some(self.turn_gyro_delta_q_deg)
    }

    fn turn_gyro_delta_vec_deg(&self) -> Option<[f64; 3]> {
        (self.turn_gyro_count > 0).then_some(self.turn_gyro_delta_vec_deg)
    }

    fn mean_horiz_angle_err_deg(&self) -> Option<f64> {
        self.horiz_angle_err_deg.mean()
    }

    fn mean_horiz_effective_std_deg(&self) -> Option<f64> {
        self.horiz_effective_std_deg.mean()
    }

    fn mean_horiz_speed_q(&self) -> Option<f64> {
        self.horiz_speed_q.mean()
    }

    fn mean_horiz_accel_q(&self) -> Option<f64> {
        self.horiz_accel_q.mean()
    }

    fn mean_horiz_turn_q(&self) -> Option<f64> {
        self.horiz_turn_q.mean()
    }

    fn mean_horiz_straight_q(&self) -> Option<f64> {
        self.horiz_straight_q.mean()
    }

    fn mean_horiz_obs_accel_vx(&self) -> Option<f64> {
        self.horiz_obs_accel_vx.mean()
    }

    fn mean_horiz_obs_accel_vy(&self) -> Option<f64> {
        self.horiz_obs_accel_vy.mean()
    }

    fn mean_horiz_gnss_norm_mps2(&self) -> Option<f64> {
        self.horiz_gnss_norm_mps2.mean()
    }

    fn mean_horiz_imu_norm_mps2(&self) -> Option<f64> {
        self.horiz_imu_norm_mps2.mean()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BehaviorAllocationSummary {
    reduced_gnss_residual_abs: f64,
    reduced_nhc_y_residual_abs: f64,
    reduced_nhc_z_residual_abs: f64,
    full_gnss_residual_abs: f64,
    full_nhc_y_residual_abs: f64,
    full_nhc_z_residual_abs: f64,
    reduced_gnss_mount_dx_deg: [f64; 3],
    reduced_nhc_mount_dx_deg: [f64; 3],
    full_gnss_mount_dx_deg: [f64; 3],
    full_nhc_mount_dx_deg: [f64; 3],
}

impl BehaviorAllocationSummary {
    fn from_events(events: &[AllocationEvent]) -> Self {
        let mut summary = Self::default();
        for event in events {
            let residual = event.residual.unwrap_or(0.0);
            match (event.source, event.update) {
                ("Reduced", "nhc_y") => {
                    summary.reduced_nhc_y_residual_abs += residual;
                    add3(&mut summary.reduced_nhc_mount_dx_deg, event.mount_dx_deg);
                }
                ("Reduced", "nhc_z") => {
                    summary.reduced_nhc_z_residual_abs += residual;
                    add3(&mut summary.reduced_nhc_mount_dx_deg, event.mount_dx_deg);
                }
                ("Reduced", update) if update.starts_with("gnss_") => {
                    summary.reduced_gnss_residual_abs += residual;
                    add3(&mut summary.reduced_gnss_mount_dx_deg, event.mount_dx_deg);
                }
                ("Full", "nhc_y") => {
                    summary.full_nhc_y_residual_abs += residual;
                    add3(&mut summary.full_nhc_mount_dx_deg, event.mount_dx_deg);
                }
                ("Full", "nhc_z") => {
                    summary.full_nhc_z_residual_abs += residual;
                    add3(&mut summary.full_nhc_mount_dx_deg, event.mount_dx_deg);
                }
                ("Full", update) if update.starts_with("gnss_") => {
                    summary.full_gnss_residual_abs += residual;
                    add3(&mut summary.full_gnss_mount_dx_deg, event.mount_dx_deg);
                }
                _ => {}
            }
        }
        summary
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

fn gnss_motion_rates(
    gnss: Option<GenericGnssSample>,
    course_rad: Option<f64>,
    prev: &PreviousBehavior,
) -> (f64, f64) {
    let Some(gnss) = gnss else {
        return (f64::NAN, f64::NAN);
    };
    let Some(prev_t) = prev.gnss_t_s else {
        return (0.0, 0.0);
    };
    if (gnss.t_s - prev_t).abs() <= 1.0e-9 {
        return (0.0, 0.0);
    }
    let dt = (gnss.t_s - prev_t).max(1.0e-6);
    let speed = horizontal_speed(gnss.vel_ned_mps);
    let prev_speed = prev.gnss_speed_mps;
    let course_rate_dps = match (course_rad, prev.gnss_course_rad) {
        (Some(_), Some(_)) if speed < 0.5 || prev_speed.is_some_and(|v| v < 0.5) => 0.0,
        (Some(course), Some(prev_course)) => wrap_rad_pi(course - prev_course).to_degrees() / dt,
        _ => f64::NAN,
    };
    let speed_rate = match prev_speed {
        Some(prev_speed) => (speed - prev_speed) / dt,
        None => f64::NAN,
    };
    (course_rate_dps, speed_rate)
}

fn classify_motion(
    speed_mps: f64,
    course_rate_dps: f64,
    speed_rate_mps2: f64,
    accel_norm_err_mps2: f64,
) -> &'static str {
    if speed_mps.is_finite() && speed_mps < 0.5 {
        "stationary"
    } else if course_rate_dps.is_finite() && course_rate_dps.abs() >= 8.0 {
        "turning"
    } else if speed_rate_mps2.is_finite() && speed_rate_mps2.abs() >= 0.5 {
        "accel_brake"
    } else if accel_norm_err_mps2.is_finite() && accel_norm_err_mps2 >= 1.0 {
        "vertical_excitation"
    } else if speed_mps.is_finite() {
        "steady"
    } else {
        "unknown"
    }
}

fn horizontal_speed(vel_ned_mps: [f64; 3]) -> f64 {
    (vel_ned_mps[0] * vel_ned_mps[0] + vel_ned_mps[1] * vel_ned_mps[1]).sqrt()
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn delta_rpy(current: Option<[f64; 3]>, previous: Option<[f64; 3]>) -> Option<[f64; 3]> {
    error_rpy(current, previous)
}

fn delta_quat_angle_deg(current: Option<[f64; 4]>, previous: Option<[f64; 4]>) -> Option<f64> {
    Some(quat_angle_deg(current?, previous?))
}

fn delta_quat_vec_deg(current: Option<[f64; 4]>, previous: Option<[f64; 4]>) -> Option<[f64; 3]> {
    let current = current?;
    let previous = previous?;
    let mut dq = quat_mul(current, quat_conj(previous));
    if dq[0] < 0.0 {
        dq = [-dq[0], -dq[1], -dq[2], -dq[3]];
    }
    let v_norm = (dq[1] * dq[1] + dq[2] * dq[2] + dq[3] * dq[3]).sqrt();
    if v_norm <= 1.0e-12 {
        return Some([0.0, 0.0, 0.0]);
    }
    let angle = 2.0 * v_norm.atan2(dq[0]).to_degrees();
    Some([
        angle * dq[1] / v_norm,
        angle * dq[2] / v_norm,
        angle * dq[3] / v_norm,
    ])
}

fn error_rpy(current: Option<[f64; 3]>, reference: Option<[f64; 3]>) -> Option<[f64; 3]> {
    let current = current?;
    let reference = reference?;
    Some([
        wrap_deg180(current[0] - reference[0]),
        wrap_deg180(current[1] - reference[1]),
        wrap_deg180(current[2] - reference[2]),
    ])
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

fn wrap_rad_pi(mut rad: f64) -> f64 {
    while rad > core::f64::consts::PI {
        rad -= 2.0 * core::f64::consts::PI;
    }
    while rad <= -core::f64::consts::PI {
        rad += 2.0 * core::f64::consts::PI;
    }
    rad
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

    #[test]
    fn behavior_allocation_summary_separates_gnss_and_nhc() {
        let events = vec![
            AllocationEvent {
                t_s: 1.0,
                source: "Reduced",
                update: "gnss_vel",
                mount_dx_deg: [1.0, 2.0, 3.0],
                att_dx_deg: [0.0; 3],
                accel_bias_dx: [0.0; 3],
                gyro_bias_dx: [0.0; 3],
                residual: Some(4.0),
                nis: None,
            },
            AllocationEvent {
                t_s: 1.1,
                source: "Reduced",
                update: "nhc_y",
                mount_dx_deg: [-0.5, 0.0, 0.25],
                att_dx_deg: [0.0; 3],
                accel_bias_dx: [0.0; 3],
                gyro_bias_dx: [0.0; 3],
                residual: Some(2.0),
                nis: None,
            },
            AllocationEvent {
                t_s: 1.2,
                source: "Full",
                update: "nhc_z",
                mount_dx_deg: [0.0, -1.0, 0.5],
                att_dx_deg: [0.0; 3],
                accel_bias_dx: [0.0; 3],
                gyro_bias_dx: [0.0; 3],
                residual: Some(3.0),
                nis: None,
            },
        ];
        let summary = BehaviorAllocationSummary::from_events(&events);
        assert_eq!(summary.reduced_gnss_residual_abs, 4.0);
        assert_eq!(summary.reduced_nhc_y_residual_abs, 2.0);
        assert_eq!(summary.full_nhc_z_residual_abs, 3.0);
        assert_eq!(summary.reduced_gnss_mount_dx_deg, [1.0, 2.0, 3.0]);
        assert_eq!(summary.reduced_nhc_mount_dx_deg, [-0.5, 0.0, 0.25]);
        assert_eq!(summary.full_nhc_mount_dx_deg, [0.0, -1.0, 0.5]);
    }

    #[test]
    fn align_behavior_summary_keeps_stage_corrections_separate() {
        let events = vec![
            AlignEvent {
                horiz_delta_q_deg: Some(1.0),
                horiz_delta_vec_deg: Some([1.0, 0.0, -0.5]),
                turn_gyro_delta_q_deg: None,
                turn_gyro_delta_vec_deg: None,
                horiz_angle_err_deg: Some(4.0),
                horiz_effective_std_deg: Some(2.0),
                horiz_speed_q: Some(0.5),
                horiz_accel_q: Some(0.25),
                horiz_turn_q: Some(0.75),
                horiz_straight_q: None,
                horiz_turn_core_valid: true,
                horiz_straight_core_valid: false,
                horiz_obs_accel_vx: Some(1.0),
                horiz_obs_accel_vy: Some(2.0),
                horiz_gnss_norm_mps2: Some(3.0),
                horiz_imu_norm_mps2: Some(4.0),
            },
            AlignEvent {
                horiz_delta_q_deg: None,
                horiz_delta_vec_deg: None,
                turn_gyro_delta_q_deg: Some(2.0),
                turn_gyro_delta_vec_deg: Some([0.25, -0.5, 0.0]),
                horiz_angle_err_deg: None,
                horiz_effective_std_deg: None,
                horiz_speed_q: None,
                horiz_accel_q: None,
                horiz_turn_q: None,
                horiz_straight_q: Some(0.4),
                horiz_turn_core_valid: false,
                horiz_straight_core_valid: true,
                horiz_obs_accel_vx: None,
                horiz_obs_accel_vy: None,
                horiz_gnss_norm_mps2: None,
                horiz_imu_norm_mps2: None,
            },
        ];
        let summary = AlignBehaviorSummary::from_events(&events);
        assert_eq!(summary.horiz_count, 1);
        assert_eq!(summary.turn_gyro_count, 1);
        assert_eq!(summary.horiz_delta_q_deg(), Some(1.0));
        assert_eq!(summary.horiz_delta_vec_deg(), Some([1.0, 0.0, -0.5]));
        assert_eq!(summary.turn_gyro_delta_q_deg(), Some(2.0));
        assert_eq!(summary.turn_gyro_delta_vec_deg(), Some([0.25, -0.5, 0.0]));
        assert_eq!(summary.mean_horiz_angle_err_deg(), Some(4.0));
        assert_eq!(summary.mean_horiz_effective_std_deg(), Some(2.0));
        assert_eq!(summary.mean_horiz_turn_q(), Some(0.75));
        assert_eq!(summary.mean_horiz_straight_q(), Some(0.4));
        assert_eq!(summary.turn_core_valid_count, 1);
        assert_eq!(summary.straight_core_valid_count, 1);
    }

    #[test]
    fn quaternion_delta_uses_physical_rotation_not_euler_wrap() {
        let prev = reference_mount_rpy_to_q_bv([179.0, 0.0, 0.0]);
        let curr = reference_mount_rpy_to_q_bv([-179.0, 0.0, 0.0]);
        let delta = delta_quat_angle_deg(Some(curr), Some(prev)).unwrap();
        assert!(delta < 2.1, "delta={delta}");
    }
}
