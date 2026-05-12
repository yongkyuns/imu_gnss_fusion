use anyhow::{Context, Result, bail};
use sensor_fusion::full::{ERROR_STATES, NominalState, State};
use sensor_fusion::reduced::UPDATE_DIAG_TYPES;
use sensor_fusion::{Config, Filter, SensorFusion};

use crate::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, GenericReferenceMotionSample,
    GenericReferencePositionSample, GenericReferenceRpySample, fusion_gnss_sample,
    fusion_imu_sample,
};
use crate::eval::gnss_ins::{as_q64, quat_angle_deg, quat_conj, quat_mul, quat_rotate};
use crate::eval::replay::{ReplayEvent, for_each_event};
use crate::visualizer::math::{
    ecef_to_lla, ecef_to_ned, lla_to_ecef, ned_to_lla_exact, quat_rpy_deg,
};
use crate::visualizer::model::{
    HeadingSample, MapCursorSample, PlotData, StateContribution, StateCorrelation, Trace,
    UpdateInspectorSample, VisualizerMountMode,
};
use crate::visualizer::pipeline::reference::{
    final_reference_mount_rpy, reference_mount_seed_q_bv, reference_rpy_at, rpy_series_from_samples,
};
use crate::visualizer::pipeline::{FilterCompareConfig, GnssOutageConfig};

pub use crate::visualizer::pipeline::reference::{
    q_bv_to_reference_mount_rpy, reference_mount_rpy_to_q_bv,
};

const DIAG_BODY_VEL_Y: usize = 4;
const DIAG_BODY_VEL_Z: usize = 5;
const NHC_DIAG_TYPES: [(usize, &str); 2] = [(DIAG_BODY_VEL_Y, "NHC Y"), (DIAG_BODY_VEL_Z, "NHC Z")];
const STANDARD_GRAVITY_MPS2: f64 = 9.80665;
#[cfg(target_arch = "wasm32")]
const WEB_BUILD_MAX_POINTS_PER_TRACE: usize =
    crate::visualizer::replay_job::WEB_TRANSPORT_MAX_POINTS_PER_TRACE;

pub struct GenericReplayInput {
    pub imu: Vec<GenericImuSample>,
    pub gnss: Vec<GenericGnssSample>,
    pub reference_attitude: Vec<GenericReferenceRpySample>,
    pub reference_mount: Vec<GenericReferenceRpySample>,
    pub reference_position: Vec<GenericReferencePositionSample>,
    pub reference_motion: Vec<GenericReferenceMotionSample>,
}

#[derive(Clone, Copy, Debug)]
pub struct GenericReplayProgress {
    pub current_t_s: f64,
    pub final_t_s: f64,
    pub fraction: f64,
}

struct GenericProgressReporter<'a> {
    callback: Option<&'a mut dyn FnMut(GenericReplayProgress)>,
    start_t_s: f64,
    final_t_s: f64,
    last_fraction: f64,
}

impl<'a> GenericProgressReporter<'a> {
    fn new(
        replay: &GenericReplayInput,
        callback: Option<&'a mut dyn FnMut(GenericReplayProgress)>,
    ) -> Self {
        let (start_t_s, final_t_s) = replay_time_range(replay).unwrap_or((0.0, 1.0));
        Self {
            callback,
            start_t_s,
            final_t_s,
            last_fraction: -1.0,
        }
    }

    fn report_stage(&mut self, stage_start: f64, stage_span: f64, current_t_s: f64) {
        let Some(callback) = self.callback.as_deref_mut() else {
            return;
        };
        let denom = (self.final_t_s - self.start_t_s).abs().max(f64::EPSILON);
        let time_fraction = ((current_t_s - self.start_t_s) / denom).clamp(0.0, 1.0);
        let fraction = (stage_start + stage_span * time_fraction).clamp(0.0, 1.0);
        if fraction < 1.0 && fraction - self.last_fraction < 0.005 {
            return;
        }
        self.last_fraction = fraction;
        callback(GenericReplayProgress {
            current_t_s,
            final_t_s: self.final_t_s,
            fraction,
        });
    }

    fn complete(&mut self) {
        let Some(callback) = self.callback.as_deref_mut() else {
            return;
        };
        self.last_fraction = 1.0;
        callback(GenericReplayProgress {
            current_t_s: self.final_t_s,
            final_t_s: self.final_t_s,
            fraction: 1.0,
        });
    }
}

impl GenericReplayInput {
    pub fn new(imu: Vec<GenericImuSample>, gnss: Vec<GenericGnssSample>) -> Self {
        Self {
            imu,
            gnss,
            reference_attitude: Vec::new(),
            reference_mount: Vec::new(),
            reference_position: Vec::new(),
            reference_motion: Vec::new(),
        }
    }
}

struct GenericReplayRunContext<'a> {
    replay: &'a GenericReplayInput,
    filter_cfg: FilterCompareConfig,
    gnss_outages: GnssOutageConfig,
    reference_mount_seed_q_bv: Option<[f32; 4]>,
}

impl<'a> GenericReplayRunContext<'a> {
    fn new(
        replay: &'a GenericReplayInput,
        filter_cfg: FilterCompareConfig,
        mount_mode: VisualizerMountMode,
        gnss_outages: GnssOutageConfig,
    ) -> Self {
        Self {
            replay,
            filter_cfg,
            gnss_outages,
            reference_mount_seed_q_bv: reference_mount_seed_q_bv(replay, mount_mode),
        }
    }

    fn configure_fusion(&self, fusion: &mut SensorFusion) {
        apply_fusion_config(fusion, self.filter_cfg);
    }

    fn reference_mount_seed_q_bv(&self) -> Option<[f32; 4]> {
        self.reference_mount_seed_q_bv
    }
}

#[derive(Clone, Copy, Debug)]
struct ReplayOutputSampling {
    imu_stride: usize,
}

impl ReplayOutputSampling {
    fn for_replay(replay: &GenericReplayInput) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            let imu_stride = replay
                .imu
                .len()
                .div_ceil(WEB_BUILD_MAX_POINTS_PER_TRACE)
                .max(1);
            Self { imu_stride }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = replay;
            Self { imu_stride: 1 }
        }
    }

    fn keep_imu(self, index: usize, total: usize) -> bool {
        self.imu_stride <= 1
            || index == 0
            || index + 1 == total
            || index.is_multiple_of(self.imu_stride)
    }
}

fn replay_time_range(replay: &GenericReplayInput) -> Option<(f64, f64)> {
    let mut start = f64::INFINITY;
    let mut end = f64::NEG_INFINITY;
    for t_s in replay
        .imu
        .iter()
        .map(|sample| sample.t_s)
        .chain(replay.gnss.iter().map(|sample| sample.t_s))
    {
        if t_s.is_finite() {
            start = start.min(t_s);
            end = end.max(t_s);
        }
    }
    start.is_finite().then_some((start, end.max(start)))
}

pub fn parse_generic_replay_csvs(imu_csv: &str, gnss_csv: &str) -> Result<GenericReplayInput> {
    parse_generic_replay_csvs_with_refs(imu_csv, gnss_csv, None, None, None)
}

pub fn parse_generic_replay_csvs_with_refs(
    imu_csv: &str,
    gnss_csv: &str,
    reference_attitude_csv: Option<&str>,
    reference_mount_csv: Option<&str>,
    reference_position_csv: Option<&str>,
) -> Result<GenericReplayInput> {
    parse_generic_replay_csvs_with_optional_motion(
        imu_csv,
        gnss_csv,
        reference_attitude_csv,
        reference_mount_csv,
        reference_position_csv,
        None,
    )
}

pub fn parse_generic_replay_csvs_with_optional_motion(
    imu_csv: &str,
    gnss_csv: &str,
    reference_attitude_csv: Option<&str>,
    reference_mount_csv: Option<&str>,
    reference_position_csv: Option<&str>,
    reference_motion_csv: Option<&str>,
) -> Result<GenericReplayInput> {
    let mut imu = parse_imu_csv(imu_csv)?;
    let mut gnss = parse_gnss_csv(gnss_csv)?;
    let mut reference_attitude = reference_attitude_csv
        .map(parse_reference_rpy_csv)
        .transpose()?
        .unwrap_or_default();
    let mut reference_mount = reference_mount_csv
        .map(parse_reference_rpy_csv)
        .transpose()?
        .unwrap_or_default();
    let mut reference_position = reference_position_csv
        .map(parse_reference_position_csv)
        .transpose()?
        .unwrap_or_default();
    let mut reference_motion = reference_motion_csv
        .map(parse_reference_motion_csv)
        .transpose()?
        .unwrap_or_default();
    imu.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    gnss.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    reference_attitude.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    reference_mount.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    reference_position.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    reference_motion.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if imu.is_empty() {
        bail!("imu.csv contained no samples");
    }
    if gnss.is_empty() {
        bail!("gnss.csv contained no samples");
    }
    Ok(GenericReplayInput {
        imu,
        gnss,
        reference_attitude,
        reference_mount,
        reference_position,
        reference_motion,
    })
}

pub fn build_generic_replay_plot_data(
    replay: &GenericReplayInput,
    mount_mode: VisualizerMountMode,
    filter_cfg: FilterCompareConfig,
    gnss_outages: GnssOutageConfig,
) -> PlotData {
    build_generic_replay_plot_data_impl(replay, mount_mode, filter_cfg, gnss_outages, None, None)
}

pub fn build_generic_replay_plot_data_with_reduced_mount_seed(
    replay: &GenericReplayInput,
    mount_mode: VisualizerMountMode,
    filter_cfg: FilterCompareConfig,
    gnss_outages: GnssOutageConfig,
    reduced_mount_seed_q_bv: Option<[f32; 4]>,
) -> PlotData {
    build_generic_replay_plot_data_impl(
        replay,
        mount_mode,
        filter_cfg,
        gnss_outages,
        None,
        reduced_mount_seed_q_bv,
    )
}

pub fn build_generic_replay_plot_data_with_progress(
    replay: &GenericReplayInput,
    mount_mode: VisualizerMountMode,
    filter_cfg: FilterCompareConfig,
    gnss_outages: GnssOutageConfig,
    progress: &mut dyn FnMut(GenericReplayProgress),
) -> PlotData {
    build_generic_replay_plot_data_impl(
        replay,
        mount_mode,
        filter_cfg,
        gnss_outages,
        Some(progress),
        None,
    )
}

pub fn build_generic_replay_plot_data_with_progress_and_reduced_mount_seed(
    replay: &GenericReplayInput,
    mount_mode: VisualizerMountMode,
    filter_cfg: FilterCompareConfig,
    gnss_outages: GnssOutageConfig,
    progress: &mut dyn FnMut(GenericReplayProgress),
    reduced_mount_seed_q_bv: Option<[f32; 4]>,
) -> PlotData {
    build_generic_replay_plot_data_impl(
        replay,
        mount_mode,
        filter_cfg,
        gnss_outages,
        Some(progress),
        reduced_mount_seed_q_bv,
    )
}

fn build_generic_replay_plot_data_impl(
    replay: &GenericReplayInput,
    mount_mode: VisualizerMountMode,
    filter_cfg: FilterCompareConfig,
    gnss_outages: GnssOutageConfig,
    progress: Option<&mut dyn FnMut(GenericReplayProgress)>,
    reduced_mount_seed_q_bv: Option<[f32; 4]>,
) -> PlotData {
    let mut progress = GenericProgressReporter::new(replay, progress);
    let ctx = GenericReplayRunContext::new(replay, filter_cfg, mount_mode, gnss_outages);
    let mut fusion = SensorFusion::new();
    ctx.configure_fusion(&mut fusion);
    if let Some(seed_q_bv) = reduced_mount_seed_q_bv.or_else(|| ctx.reference_mount_seed_q_bv()) {
        fusion.set_misalignment(seed_q_bv);
    }

    let ref_gnss = replay.gnss.first().copied();
    let ref_ecef = ref_gnss.map(|s| lla_to_ecef(s.lat_deg, s.lon_deg, s.height_m));
    let outage_windows = sample_outage_windows(&replay.gnss, ctx.gnss_outages);
    let output_sampling = ReplayOutputSampling::for_replay(replay);
    let imu_total = replay.imu.len();

    let mut raw_gyro_x = Vec::new();
    let mut raw_gyro_y = Vec::new();
    let mut raw_gyro_z = Vec::new();
    let mut raw_accel_x = Vec::new();
    let mut raw_accel_y = Vec::new();
    let mut raw_accel_z = Vec::new();
    let mut gnss_speed = Vec::new();
    let mut gnss_pos_n = Vec::new();
    let mut gnss_pos_e = Vec::new();
    let mut gnss_pos_d = Vec::new();
    let mut gnss_vel_n = Vec::new();
    let mut gnss_vel_e = Vec::new();
    let mut gnss_vel_d = Vec::new();
    let mut reference_vel_n = Vec::new();
    let mut reference_vel_e = Vec::new();
    let mut reference_vel_d = Vec::new();
    let mut gnss_map = Vec::new();
    let mut reference_position_map = Vec::new();
    let mut map_cursor = Vec::new();

    for sample in &replay.gnss {
        gnss_speed.push([
            sample.t_s,
            sample.vel_ned_mps[0].hypot(sample.vel_ned_mps[1]),
        ]);
        gnss_vel_n.push([sample.t_s, sample.vel_ned_mps[0]]);
        gnss_vel_e.push([sample.t_s, sample.vel_ned_mps[1]]);
        gnss_vel_d.push([sample.t_s, sample.vel_ned_mps[2]]);
        gnss_map.push([sample.lon_deg, sample.lat_deg]);
        map_cursor.push(MapCursorSample {
            trace_name: "GNSS-only path (lon,lat)".to_string(),
            t_s: sample.t_s,
            lon_deg: sample.lon_deg,
            lat_deg: sample.lat_deg,
            yaw_deg: heading_deg_from_sample(sample.heading_rad, sample.vel_ned_mps),
        });
        if let (Some(ref_sample), Some(ref_ecef)) = (ref_gnss, ref_ecef) {
            let ecef = lla_to_ecef(sample.lat_deg, sample.lon_deg, sample.height_m);
            let ned = ecef_to_ned(ecef, ref_ecef, ref_sample.lat_deg, ref_sample.lon_deg);
            gnss_pos_n.push([sample.t_s, ned[0]]);
            gnss_pos_e.push([sample.t_s, ned[1]]);
            gnss_pos_d.push([sample.t_s, ned[2]]);
        }
    }
    for sample in &replay.reference_position {
        reference_vel_n.push([sample.t_s, sample.vel_ned_mps[0]]);
        reference_vel_e.push([sample.t_s, sample.vel_ned_mps[1]]);
        reference_vel_d.push([sample.t_s, sample.vel_ned_mps[2]]);
        reference_position_map.push([sample.lon_deg, sample.lat_deg]);
        map_cursor.push(MapCursorSample {
            trace_name: "Reference fused path (lon,lat)".to_string(),
            t_s: sample.t_s,
            lon_deg: sample.lon_deg,
            lat_deg: sample.lat_deg,
            yaw_deg: heading_deg_from_sample(sample.heading_rad, sample.vel_ned_mps),
        });
    }

    let mut reduced_pos_n = Vec::new();
    let mut reduced_pos_e = Vec::new();
    let mut reduced_pos_d = Vec::new();
    let mut reduced_vel_n = Vec::new();
    let mut reduced_vel_e = Vec::new();
    let mut reduced_vel_d = Vec::new();
    let mut reduced_roll = Vec::new();
    let mut reduced_pitch = Vec::new();
    let mut reduced_yaw = Vec::new();
    let mut reduced_mount_roll = Vec::new();
    let mut reduced_mount_pitch = Vec::new();
    let mut reduced_mount_yaw = Vec::new();
    let mut reduced_bgx = Vec::new();
    let mut reduced_bgy = Vec::new();
    let mut reduced_bgz = Vec::new();
    let mut reduced_bax = Vec::new();
    let mut reduced_bay = Vec::new();
    let mut reduced_baz = Vec::new();
    let mut reduced_motion_gyro: [Vec<[f64; 2]>; 3] = std::array::from_fn(|_| Vec::new());
    let mut reduced_motion_accel: [Vec<[f64; 2]>; 3] = std::array::from_fn(|_| Vec::new());
    let mut reduced_cov: [Vec<[f64; 2]>; 18] = std::array::from_fn(|_| Vec::new());
    let mut reduced_mount_dx: [Vec<[f64; 2]>; 3] = std::array::from_fn(|_| Vec::new());
    let mut reduced_nhc_mount_dx: [Vec<[f64; 2]>; 6] = std::array::from_fn(|_| Vec::new());
    let mut reduced_nhc_innovation: [Vec<[f64; 2]>; 2] = std::array::from_fn(|_| Vec::new());
    let mut reduced_nhc_nis: [Vec<[f64; 2]>; 2] = std::array::from_fn(|_| Vec::new());
    let mut reduced_nhc_h_mount_norm: [Vec<[f64; 2]>; 2] = std::array::from_fn(|_| Vec::new());
    let mut update_inspector = Vec::new();
    let mut last_reduced_update_count = 0u32;
    let mut last_reduced_type_counts = [0u32; UPDATE_DIAG_TYPES];
    let mut reduced_map = Vec::new();
    let mut reduced_outage_map = Vec::new();
    let mut reduced_heading = Vec::new();
    let mut mount_ready_marker = Vec::new();
    let mut reduced_init_marker = Vec::new();

    for_each_event(&replay.imu, &replay.gnss, |event| match event {
        ReplayEvent::Imu(index, sample) => {
            progress.report_stage(0.0, 0.55, sample.t_s);
            let _ = fusion.process_imu(fusion_imu_sample(*sample));
            if !output_sampling.keep_imu(index, imu_total) {
                return;
            }
            raw_gyro_x.push([sample.t_s, sample.gyro_radps[0].to_degrees()]);
            raw_gyro_y.push([sample.t_s, sample.gyro_radps[1].to_degrees()]);
            raw_gyro_z.push([sample.t_s, sample.gyro_radps[2].to_degrees()]);
            raw_accel_x.push([sample.t_s, sample.accel_mps2[0]]);
            raw_accel_y.push([sample.t_s, sample.accel_mps2[1]]);
            raw_accel_z.push([sample.t_s, sample.accel_mps2[2]]);
            append_reduced_motion_sample(
                sample.t_s,
                sample,
                &fusion,
                &mut reduced_motion_gyro,
                &mut reduced_motion_accel,
            );
            append_reduced_sample(
                sample.t_s,
                &fusion,
                ref_gnss,
                ref_ecef,
                &mut reduced_pos_n,
                &mut reduced_pos_e,
                &mut reduced_pos_d,
                &mut reduced_vel_n,
                &mut reduced_vel_e,
                &mut reduced_vel_d,
                &mut reduced_roll,
                &mut reduced_pitch,
                &mut reduced_yaw,
                &mut reduced_mount_roll,
                &mut reduced_mount_pitch,
                &mut reduced_mount_yaw,
                &mut reduced_bgx,
                &mut reduced_bgy,
                &mut reduced_bgz,
                &mut reduced_bax,
                &mut reduced_bay,
                &mut reduced_baz,
                &mut reduced_cov,
                &mut reduced_mount_dx,
                &mut last_reduced_update_count,
                &mut reduced_nhc_mount_dx,
                &mut reduced_nhc_innovation,
                &mut reduced_nhc_nis,
                &mut reduced_nhc_h_mount_norm,
                &mut update_inspector,
                &mut last_reduced_type_counts,
                &mut reduced_map,
                &mut reduced_outage_map,
                &mut reduced_heading,
                &mut map_cursor,
                in_outage(sample.t_s, &outage_windows),
            );
        }
        ReplayEvent::Gnss(_, sample) => {
            progress.report_stage(0.0, 0.55, sample.t_s);
            if !in_outage(sample.t_s, &outage_windows) {
                let update = fusion.process_gnss(fusion_gnss_sample(*sample));
                if update.mount_ready_changed && update.mount_ready {
                    mount_ready_marker.push([sample.t_s, 0.0]);
                }
                if update.reduced_initialized_now && update.reduced_initialized {
                    reduced_init_marker.push([sample.t_s, 0.0]);
                }
            }
        }
    });

    let mut data = PlotData {
        speed: vec![Trace {
            name: "GNSS speed [m/s]".to_string(),
            points: gnss_speed,
        }],
        imu_raw_gyro: vec![
            Trace {
                name: "Raw IMU gyro X [deg/s]".to_string(),
                points: raw_gyro_x.clone(),
            },
            Trace {
                name: "Raw IMU gyro Y [deg/s]".to_string(),
                points: raw_gyro_y.clone(),
            },
            Trace {
                name: "Raw IMU gyro Z [deg/s]".to_string(),
                points: raw_gyro_z.clone(),
            },
        ],
        imu_raw_accel: vec![
            Trace {
                name: "Raw IMU accel X [m/s^2]".to_string(),
                points: raw_accel_x.clone(),
            },
            Trace {
                name: "Raw IMU accel Y [m/s^2]".to_string(),
                points: raw_accel_y.clone(),
            },
            Trace {
                name: "Raw IMU accel Z [m/s^2]".to_string(),
                points: raw_accel_z.clone(),
            },
        ],
        reduced_cmp_pos: vec![
            Trace {
                name: "GNSS posN [m]".to_string(),
                points: gnss_pos_n,
            },
            Trace {
                name: "Reduced posN [m]".to_string(),
                points: reduced_pos_n,
            },
            Trace {
                name: "GNSS posE [m]".to_string(),
                points: gnss_pos_e,
            },
            Trace {
                name: "Reduced posE [m]".to_string(),
                points: reduced_pos_e,
            },
            Trace {
                name: "GNSS posD [m]".to_string(),
                points: gnss_pos_d,
            },
            Trace {
                name: "Reduced posD [m]".to_string(),
                points: reduced_pos_d,
            },
        ],
        reduced_cmp_vel: vec![
            Trace {
                name: "GNSS velN [m/s]".to_string(),
                points: gnss_vel_n,
            },
            Trace {
                name: "Reference velN [m/s]".to_string(),
                points: reference_vel_n.clone(),
            },
            Trace {
                name: "Reduced velN [m/s]".to_string(),
                points: reduced_vel_n,
            },
            Trace {
                name: "GNSS velE [m/s]".to_string(),
                points: gnss_vel_e,
            },
            Trace {
                name: "Reference velE [m/s]".to_string(),
                points: reference_vel_e.clone(),
            },
            Trace {
                name: "Reduced velE [m/s]".to_string(),
                points: reduced_vel_e,
            },
            Trace {
                name: "GNSS velD [m/s]".to_string(),
                points: gnss_vel_d,
            },
            Trace {
                name: "Reference velD [m/s]".to_string(),
                points: reference_vel_d.clone(),
            },
            Trace {
                name: "Reduced velD [m/s]".to_string(),
                points: reduced_vel_d,
            },
        ],
        reduced_cmp_att: vec![
            Trace {
                name: "Reduced roll [deg]".to_string(),
                points: reduced_roll,
            },
            Trace {
                name: "Reduced pitch [deg]".to_string(),
                points: reduced_pitch,
            },
            Trace {
                name: "Reduced yaw [deg]".to_string(),
                points: reduced_yaw,
            },
            Trace {
                name: "mount ready".to_string(),
                points: mount_ready_marker,
            },
            Trace {
                name: "reduced initialized".to_string(),
                points: reduced_init_marker,
            },
        ],
        vehicle_motion_gyro: axis_traces_deg_per_s("Reduced angular velocity", reduced_motion_gyro),
        vehicle_motion_accel: axis_traces_mps2("Reduced linear acceleration", reduced_motion_accel),
        reduced_bias_gyro: vec![
            Trace {
                name: "Reduced gyro bias X [deg/s]".to_string(),
                points: reduced_bgx,
            },
            Trace {
                name: "Reduced gyro bias Y [deg/s]".to_string(),
                points: reduced_bgy,
            },
            Trace {
                name: "Reduced gyro bias Z [deg/s]".to_string(),
                points: reduced_bgz,
            },
        ],
        reduced_bias_accel: vec![
            Trace {
                name: "Reduced accel bias X [m/s^2]".to_string(),
                points: reduced_bax,
            },
            Trace {
                name: "Reduced accel bias Y [m/s^2]".to_string(),
                points: reduced_bay,
            },
            Trace {
                name: "Reduced accel bias Z [m/s^2]".to_string(),
                points: reduced_baz,
            },
        ],
        reduced_meas_gyro: vec![
            Trace {
                name: "Reduced raw IMU gyro X [deg/s]".to_string(),
                points: raw_gyro_x.clone(),
            },
            Trace {
                name: "Reduced raw IMU gyro Y [deg/s]".to_string(),
                points: raw_gyro_y.clone(),
            },
            Trace {
                name: "Reduced raw IMU gyro Z [deg/s]".to_string(),
                points: raw_gyro_z.clone(),
            },
        ],
        reduced_meas_accel: vec![
            Trace {
                name: "Reduced raw IMU accel X [m/s^2]".to_string(),
                points: raw_accel_x.clone(),
            },
            Trace {
                name: "Reduced raw IMU accel Y [m/s^2]".to_string(),
                points: raw_accel_y.clone(),
            },
            Trace {
                name: "Reduced raw IMU accel Z [m/s^2]".to_string(),
                points: raw_accel_z.clone(),
            },
        ],
        reduced_cov_bias: vec![
            Trace {
                name: "Reduced accel bias sigma X [m/s^2]".to_string(),
                points: reduced_cov[12].clone(),
            },
            Trace {
                name: "Reduced accel bias sigma Y [m/s^2]".to_string(),
                points: reduced_cov[13].clone(),
            },
            Trace {
                name: "Reduced accel bias sigma Z [m/s^2]".to_string(),
                points: reduced_cov[14].clone(),
            },
            Trace {
                name: "Reduced gyro bias sigma X [deg/s]".to_string(),
                points: reduced_cov[9].clone(),
            },
            Trace {
                name: "Reduced gyro bias sigma Y [deg/s]".to_string(),
                points: reduced_cov[10].clone(),
            },
            Trace {
                name: "Reduced gyro bias sigma Z [deg/s]".to_string(),
                points: reduced_cov[11].clone(),
            },
        ],
        reduced_cov_nonbias: (0..9)
            .map(|i| Trace {
                name: format!("Reduced state_{i}"),
                points: reduced_cov[i].clone(),
            })
            .collect(),
        reduced_mount_sigma: vec![
            Trace {
                name: "Reduced mount roll sigma [deg]".to_string(),
                points: sigma_rad_points_to_deg(&reduced_cov[15]),
            },
            Trace {
                name: "Reduced mount pitch sigma [deg]".to_string(),
                points: sigma_rad_points_to_deg(&reduced_cov[16]),
            },
            Trace {
                name: "Reduced mount yaw sigma [deg]".to_string(),
                points: sigma_rad_points_to_deg(&reduced_cov[17]),
            },
        ],
        reduced_mount_dx: ["roll", "pitch", "yaw"]
            .into_iter()
            .zip(reduced_mount_dx)
            .map(|(axis, points)| Trace {
                name: format!("Reduced mount {axis} correction [deg/update]"),
                points,
            })
            .collect(),
        reduced_nhc_mount_dx: reduced_nhc_mount_dx_traces(&reduced_nhc_mount_dx),
        reduced_nhc_innovation: NHC_DIAG_TYPES
            .into_iter()
            .enumerate()
            .map(|(i, (_, label))| Trace {
                name: format!("Reduced {label} innovation [m/s]"),
                points: reduced_nhc_innovation[i].clone(),
            })
            .collect(),
        reduced_nhc_nis: NHC_DIAG_TYPES
            .into_iter()
            .enumerate()
            .map(|(i, (_, label))| Trace {
                name: format!("Reduced {label} NIS"),
                points: reduced_nhc_nis[i].clone(),
            })
            .collect(),
        reduced_nhc_h_mount_norm: NHC_DIAG_TYPES
            .into_iter()
            .enumerate()
            .map(|(i, (_, label))| Trace {
                name: format!("Reduced {label} mount H norm"),
                points: reduced_nhc_h_mount_norm[i].clone(),
            })
            .collect(),
        reduced_misalignment: vec![
            Trace {
                name: "Reduced mount roll [deg]".to_string(),
                points: reduced_mount_roll,
            },
            Trace {
                name: "Reduced mount pitch [deg]".to_string(),
                points: reduced_mount_pitch,
            },
            Trace {
                name: "Reduced mount yaw [deg]".to_string(),
                points: reduced_mount_yaw,
            },
        ],
        reduced_map: vec![
            Trace {
                name: "GNSS-only path (lon,lat)".to_string(),
                points: gnss_map,
            },
            Trace {
                name: "Reference fused path (lon,lat)".to_string(),
                points: reference_position_map,
            },
            Trace {
                name: "Reduced path (lon,lat)".to_string(),
                points: reduced_map,
            },
            Trace {
                name: "Reduced path during GNSS outage (lon,lat)".to_string(),
                points: reduced_outage_map,
            },
        ],
        map_cursor,
        reduced_map_heading: reduced_heading,
        update_inspector,
        ..PlotData::default()
    };
    add_auxiliary_generic_traces_impl(&mut data, &ctx, None, None, &mut progress);
    progress.complete();
    data
}

pub fn add_auxiliary_generic_traces(
    data: &mut PlotData,
    replay: &GenericReplayInput,
    filter_cfg: FilterCompareConfig,
    mount_mode: VisualizerMountMode,
    reference_mount_rpy_deg: Option<[f64; 3]>,
    reference_attitude_rpy: Option<[Vec<[f64; 2]>; 3]>,
) {
    let mut progress = GenericProgressReporter::new(replay, None);
    let ctx =
        GenericReplayRunContext::new(replay, filter_cfg, mount_mode, GnssOutageConfig::default());
    add_auxiliary_generic_traces_impl(
        data,
        &ctx,
        reference_mount_rpy_deg,
        reference_attitude_rpy,
        &mut progress,
    );
}

fn add_auxiliary_generic_traces_impl(
    data: &mut PlotData,
    ctx: &GenericReplayRunContext<'_>,
    reference_mount_rpy_deg: Option<[f64; 3]>,
    reference_attitude_rpy: Option<[Vec<[f64; 2]>; 3]>,
    progress: &mut GenericProgressReporter<'_>,
) {
    let replay = ctx.replay;
    let reference_mount_series = rpy_series_from_samples(&replay.reference_mount);
    let reference_mount_final_rpy_deg =
        reference_mount_rpy_deg.or_else(|| final_reference_mount_rpy(&replay.reference_mount));
    let reference_attitude_series =
        reference_attitude_rpy.or_else(|| rpy_series_from_samples(&replay.reference_attitude));
    populate_align_traces(
        data,
        ctx,
        reference_mount_rpy_deg,
        replay.reference_mount.as_slice(),
        progress,
    );
    populate_full_traces(data, ctx, progress);
    populate_reduced_bump_traces(data);
    append_reference_motion_traces(data, &replay.reference_motion);
    if let Some(truth) = reference_attitude_series {
        let traces = [
            Trace {
                name: "Reference roll [deg]".to_string(),
                points: truth[0].clone(),
            },
            Trace {
                name: "Reference pitch [deg]".to_string(),
                points: truth[1].clone(),
            },
            Trace {
                name: "Reference yaw [deg]".to_string(),
                points: truth[2].clone(),
            },
        ];
        data.reduced_cmp_att.extend(traces.clone());
        data.full_cmp_att.extend(traces);
    }
    if let Some(reference) = reference_mount_series {
        push_mount_quaternion_error_trace(
            &mut data.reduced_misalignment,
            "Reduced",
            "Reduced mount",
            reference_mount_final_rpy_deg,
        );
        push_mount_quaternion_error_trace(
            &mut data.full_misalignment,
            "Full",
            "Full mount",
            reference_mount_final_rpy_deg,
        );
        push_mount_quaternion_error_trace(
            &mut data.align_cmp_att,
            "Align",
            "Align",
            reference_mount_final_rpy_deg,
        );
        let traces = [
            Trace {
                name: "Reference mount roll [deg]".to_string(),
                points: reference[0].clone(),
            },
            Trace {
                name: "Reference mount pitch [deg]".to_string(),
                points: reference[1].clone(),
            },
            Trace {
                name: "Reference mount yaw [deg]".to_string(),
                points: reference[2].clone(),
            },
        ];
        data.reduced_misalignment.extend(traces.clone());
        data.full_misalignment.extend(traces);
    }
}

fn push_mount_quaternion_error_trace(
    traces: &mut Vec<Trace>,
    system_label: &str,
    trace_prefix: &str,
    reference_mount_final_rpy_deg: Option<[f64; 3]>,
) {
    if traces
        .iter()
        .any(|trace| trace.name == format!("{system_label} mount quaternion error [deg]"))
    {
        return;
    }
    let Some(roll) = trace_by_name(traces, &format!("{trace_prefix} roll [deg]")) else {
        return;
    };
    let Some(pitch) = trace_by_name(traces, &format!("{trace_prefix} pitch [deg]")) else {
        return;
    };
    let Some(yaw) = trace_by_name(traces, &format!("{trace_prefix} yaw [deg]")) else {
        return;
    };
    let Some(reference) = reference_mount_final_rpy_deg else {
        return;
    };
    let points = roll
        .points
        .iter()
        .filter_map(|sample| {
            let t_s = sample[0];
            let pitch_deg = sample_trace_at(pitch, t_s)?;
            let yaw_deg = sample_trace_at(yaw, t_s)?;
            let q_est = reference_mount_rpy_to_q_bv([sample[1], pitch_deg, yaw_deg]);
            let q_ref = reference_mount_rpy_to_q_bv(reference);
            Some([t_s, quat_angle_deg(q_est, q_ref)])
        })
        .collect::<Vec<_>>();
    if !points.is_empty() {
        traces.push(Trace {
            name: format!("{system_label} mount quaternion error [deg]"),
            points,
        });
    }
}

fn trace_by_name<'a>(traces: &'a [Trace], name: &str) -> Option<&'a Trace> {
    traces.iter().find(|trace| trace.name == name)
}

fn sample_trace_at(trace: &Trace, t_s: f64) -> Option<f64> {
    let idx = trace.points.partition_point(|point| point[0] < t_s);
    let point = match (idx.checked_sub(1), trace.points.get(idx)) {
        (Some(prev_idx), Some(next)) => {
            let prev = trace.points[prev_idx];
            if (t_s - prev[0]).abs() <= (next[0] - t_s).abs() {
                prev
            } else {
                *next
            }
        }
        (Some(prev_idx), None) => trace.points[prev_idx],
        (None, Some(next)) => *next,
        (None, None) => return None,
    };
    point[1].is_finite().then_some(point[1])
}

fn populate_align_traces(
    data: &mut PlotData,
    ctx: &GenericReplayRunContext<'_>,
    reference_mount_rpy_deg: Option<[f64; 3]>,
    reference_mount_series: &[GenericReferenceRpySample],
    progress: &mut GenericProgressReporter<'_>,
) {
    let replay = ctx.replay;
    let mut fusion = SensorFusion::new();
    ctx.configure_fusion(&mut fusion);

    let mut align_roll = Vec::new();
    let mut align_pitch = Vec::new();
    let mut align_yaw = Vec::new();
    let mut ref_roll = Vec::new();
    let mut ref_pitch = Vec::new();
    let mut ref_yaw = Vec::new();
    let mut axis_roll_err = Vec::new();
    let mut axis_pitch_err = Vec::new();
    let mut axis_yaw_err = Vec::new();
    let mut speed_mid = Vec::new();
    let mut gyro_norm = Vec::new();
    let mut accel_norm = Vec::new();
    let mut horiz_angle = Vec::new();
    let mut gnss_accel = Vec::new();
    let mut imu_accel = Vec::new();
    let mut straight_valid = Vec::new();
    let mut turn_valid = Vec::new();
    let mut coarse_ready = Vec::new();
    let mut gravity_roll = Vec::new();
    let mut gravity_pitch = Vec::new();
    let mut gravity_yaw = Vec::new();
    let mut horiz_roll = Vec::new();
    let mut horiz_pitch = Vec::new();
    let mut horiz_yaw = Vec::new();
    let mut turn_roll = Vec::new();
    let mut turn_pitch = Vec::new();
    let mut turn_yaw = Vec::new();
    let mut cov_roll = Vec::new();
    let mut cov_pitch = Vec::new();
    let mut cov_yaw = Vec::new();

    for_each_event(&replay.imu, &replay.gnss, |event| match event {
        ReplayEvent::Imu(_, sample) => {
            progress.report_stage(0.55, 0.17, sample.t_s);
            let _ = fusion.process_imu(fusion_imu_sample(*sample));
        }
        ReplayEvent::Gnss(_, sample) => {
            progress.report_stage(0.55, 0.17, sample.t_s);
            let _ = fusion.process_gnss(fusion_gnss_sample(*sample));
            let Some(debug) = fusion.align_debug() else {
                return;
            };
            let t = sample.t_s;
            if align_roll.last().is_some_and(|p: &[f64; 2]| p[0] == t) {
                return;
            }
            let Some(align) = fusion.align() else {
                return;
            };
            let (r, p, y) = q_bv_to_reference_mount_rpy([
                align.q_bv[0] as f64,
                align.q_bv[1] as f64,
                align.q_bv[2] as f64,
                align.q_bv[3] as f64,
            ]);
            align_roll.push([t, r]);
            align_pitch.push([t, p]);
            align_yaw.push([t, y]);
            let reference =
                reference_mount_rpy_deg.or_else(|| reference_rpy_at(reference_mount_series, t));
            if let Some(reference) = reference {
                ref_roll.push([t, reference[0]]);
                ref_pitch.push([t, reference[1]]);
                ref_yaw.push([t, reference[2]]);
                axis_roll_err.push([t, wrap_deg(r - reference[0])]);
                axis_pitch_err.push([t, wrap_deg(p - reference[1])]);
                axis_yaw_err.push([t, wrap_deg(y - reference[2])]);
            }

            let window = debug.window;
            let trace = debug.trace;
            let speed = 0.5
                * (window.gnss_vel_prev_n[0].hypot(window.gnss_vel_prev_n[1])
                    + window.gnss_vel_curr_n[0].hypot(window.gnss_vel_curr_n[1]))
                    as f64;
            speed_mid.push([t, speed]);
            gyro_norm.push([t, vec3_norm_f32(window.mean_gyro_b) as f64]);
            accel_norm.push([t, vec3_norm_f32(window.mean_accel_b) as f64]);
            if let Some(v) = trace.horiz_angle_err_rad {
                horiz_angle.push([t, (v as f64).to_degrees()]);
            }
            if let Some(v) = trace.horiz_gnss_norm_mps2 {
                gnss_accel.push([t, v as f64]);
            }
            if let Some(v) = trace.horiz_imu_norm_mps2 {
                imu_accel.push([t, v as f64]);
            }
            straight_valid.push([t, f64::from(trace.horiz_straight_core_valid)]);
            turn_valid.push([t, f64::from(trace.horiz_turn_core_valid)]);
            coarse_ready.push([t, f64::from(trace.coarse_alignment_ready)]);
            push_update_contrib(
                t,
                trace.q_start,
                trace.after_gravity,
                &mut gravity_roll,
                &mut gravity_pitch,
                &mut gravity_yaw,
            );
            push_update_contrib(
                t,
                trace.q_start,
                trace.after_horiz_accel,
                &mut horiz_roll,
                &mut horiz_pitch,
                &mut horiz_yaw,
            );
            push_update_contrib(
                t,
                trace.q_start,
                trace.after_turn_gyro,
                &mut turn_roll,
                &mut turn_pitch,
                &mut turn_yaw,
            );
            cov_roll.push([t, (align.P[0][0].max(0.0).sqrt() as f64).to_degrees()]);
            cov_pitch.push([t, (align.P[1][1].max(0.0).sqrt() as f64).to_degrees()]);
            cov_yaw.push([t, (align.P[2][2].max(0.0).sqrt() as f64).to_degrees()]);
        }
    });

    data.align_cmp_att = vec![
        Trace {
            name: "Align roll [deg]".to_string(),
            points: align_roll,
        },
        Trace {
            name: "Align pitch [deg]".to_string(),
            points: align_pitch,
        },
        Trace {
            name: "Align yaw [deg]".to_string(),
            points: align_yaw,
        },
    ];
    if !ref_roll.is_empty() {
        data.align_cmp_att.extend([
            Trace {
                name: "Reference mount roll [deg]".to_string(),
                points: ref_roll,
            },
            Trace {
                name: "Reference mount pitch [deg]".to_string(),
                points: ref_pitch,
            },
            Trace {
                name: "Reference mount yaw [deg]".to_string(),
                points: ref_yaw,
            },
        ]);
        data.align_axis_err = vec![
            Trace {
                name: "Align roll error [deg]".to_string(),
                points: axis_roll_err,
            },
            Trace {
                name: "Align pitch error [deg]".to_string(),
                points: axis_pitch_err,
            },
            Trace {
                name: "Align yaw error [deg]".to_string(),
                points: axis_yaw_err,
            },
        ];
    }
    data.align_res_vel = vec![
        Trace {
            name: "Window speed quality proxy [m/s]".to_string(),
            points: speed_mid,
        },
        Trace {
            name: "Mean gyro norm [rad/s]".to_string(),
            points: gyro_norm,
        },
        Trace {
            name: "Mean accel norm [m/s^2]".to_string(),
            points: accel_norm,
        },
        Trace {
            name: "Horizontal heading innovation [deg]".to_string(),
            points: horiz_angle,
        },
        Trace {
            name: "GNSS horizontal accel norm [m/s^2]".to_string(),
            points: gnss_accel,
        },
        Trace {
            name: "IMU horizontal accel norm [m/s^2]".to_string(),
            points: imu_accel,
        },
    ];
    data.align_flags = vec![
        Trace {
            name: "straight window accepted".to_string(),
            points: straight_valid,
        },
        Trace {
            name: "turn window accepted".to_string(),
            points: turn_valid,
        },
        Trace {
            name: "coarse alignment ready".to_string(),
            points: coarse_ready,
        },
    ];
    data.align_roll_contrib = vec![
        Trace {
            name: "gravity roll update [deg]".to_string(),
            points: gravity_roll,
        },
        Trace {
            name: "horizontal accel roll update [deg]".to_string(),
            points: horiz_roll,
        },
        Trace {
            name: "turn gyro roll update [deg]".to_string(),
            points: turn_roll,
        },
    ];
    data.align_pitch_contrib = vec![
        Trace {
            name: "gravity pitch update [deg]".to_string(),
            points: gravity_pitch,
        },
        Trace {
            name: "horizontal accel pitch update [deg]".to_string(),
            points: horiz_pitch,
        },
        Trace {
            name: "turn gyro pitch update [deg]".to_string(),
            points: turn_pitch,
        },
    ];
    data.align_yaw_contrib = vec![
        Trace {
            name: "gravity yaw update [deg]".to_string(),
            points: gravity_yaw,
        },
        Trace {
            name: "horizontal accel yaw update [deg]".to_string(),
            points: horiz_yaw,
        },
        Trace {
            name: "turn gyro yaw update [deg]".to_string(),
            points: turn_yaw,
        },
    ];
    data.align_cov = vec![
        Trace {
            name: "Align roll sigma [deg]".to_string(),
            points: cov_roll,
        },
        Trace {
            name: "Align pitch sigma [deg]".to_string(),
            points: cov_pitch,
        },
        Trace {
            name: "Align yaw sigma [deg]".to_string(),
            points: cov_yaw,
        },
    ];
}

fn populate_full_traces(
    data: &mut PlotData,
    ctx: &GenericReplayRunContext<'_>,
    progress: &mut GenericProgressReporter<'_>,
) {
    let replay = ctx.replay;
    let filter_cfg = ctx.filter_cfg;
    let Some(ref_gnss) = replay.gnss.first().copied() else {
        return;
    };
    let ref_ecef = lla_to_ecef(ref_gnss.lat_deg, ref_gnss.lon_deg, ref_gnss.height_m);
    let mut full_fusion = SensorFusion::with_config(Config {
        filter: Filter::Full,
        ..Config::default()
    });
    apply_fusion_config(&mut full_fusion, filter_cfg);
    if let Some(seed_q_bv) = ctx.reference_mount_seed_q_bv() {
        full_fusion.set_misalignment(seed_q_bv);
    }

    let mut pos_n = Vec::new();
    let mut pos_e = Vec::new();
    let mut pos_d = Vec::new();
    let mut vel_n = Vec::new();
    let mut vel_e = Vec::new();
    let mut vel_d = Vec::new();
    let mut roll = Vec::new();
    let mut pitch = Vec::new();
    let mut yaw = Vec::new();
    let mut mount_roll = Vec::new();
    let mut mount_pitch = Vec::new();
    let mut mount_yaw = Vec::new();
    let mut gyro_x = Vec::new();
    let mut gyro_y = Vec::new();
    let mut gyro_z = Vec::new();
    let mut accel_x = Vec::new();
    let mut accel_y = Vec::new();
    let mut accel_z = Vec::new();
    let mut bgx = Vec::new();
    let mut bgy = Vec::new();
    let mut bgz = Vec::new();
    let mut bax = Vec::new();
    let mut bay = Vec::new();
    let mut accel_bias_z = Vec::new();
    let mut sgx = Vec::new();
    let mut sgy = Vec::new();
    let mut sgz = Vec::new();
    let mut sax = Vec::new();
    let mut say = Vec::new();
    let mut saz = Vec::new();
    let mut motion_gyro: [Vec<[f64; 2]>; 3] = std::array::from_fn(|_| Vec::new());
    let mut motion_accel: [Vec<[f64; 2]>; 3] = std::array::from_fn(|_| Vec::new());
    let mut cov_bias: [Vec<[f64; 2]>; 12] = std::array::from_fn(|_| Vec::new());
    let mut cov_nonbias: [Vec<[f64; 2]>; 12] = std::array::from_fn(|_| Vec::new());
    let mut cov_mount: [Vec<[f64; 2]>; 3] = std::array::from_fn(|_| Vec::new());
    let mut dx_mount: [Vec<[f64; 2]>; 3] = std::array::from_fn(|_| Vec::new());
    let mut nhc_innovation: [Vec<[f64; 2]>; 2] = std::array::from_fn(|_| Vec::new());
    let mut gnss_pos_gate_norm: [Vec<[f64; 2]>; 3] = std::array::from_fn(|_| Vec::new());
    let mut gnss_pos_gate_status = Vec::new();
    let mut map = Vec::new();
    let mut headings = Vec::new();
    let output_sampling = ReplayOutputSampling::for_replay(replay);
    let imu_total = replay.imu.len();

    for_each_event(&replay.imu, &replay.gnss, |event| match event {
        ReplayEvent::Imu(index, sample) => {
            progress.report_stage(0.72, 0.26, sample.t_s);
            let _ = full_fusion.process_imu(fusion_imu_sample(*sample));
            if !output_sampling.keep_imu(index, imu_total) {
                return;
            }
            append_full_motion_sample(
                sample.t_s,
                sample,
                full_fusion.full(),
                &mut motion_gyro,
                &mut motion_accel,
            );
            if let Some(full) = full_fusion.full() {
                append_full_sample(
                    sample.t_s,
                    full,
                    ref_gnss,
                    ref_ecef,
                    &mut pos_n,
                    &mut pos_e,
                    &mut pos_d,
                    &mut vel_n,
                    &mut vel_e,
                    &mut vel_d,
                    &mut roll,
                    &mut pitch,
                    &mut yaw,
                    &mut mount_roll,
                    &mut mount_pitch,
                    &mut mount_yaw,
                    &mut bgx,
                    &mut bgy,
                    &mut bgz,
                    &mut bax,
                    &mut bay,
                    &mut accel_bias_z,
                    &mut sgx,
                    &mut sgy,
                    &mut sgz,
                    &mut sax,
                    &mut say,
                    &mut saz,
                    &mut cov_bias,
                    &mut cov_nonbias,
                    &mut cov_mount,
                    &mut dx_mount,
                    &mut nhc_innovation,
                    &mut gnss_pos_gate_norm,
                    &mut gnss_pos_gate_status,
                    &mut data.update_inspector,
                    &mut map,
                    &mut headings,
                    &mut data.map_cursor,
                );
            }
            gyro_x.push([sample.t_s, sample.gyro_radps[0].to_degrees()]);
            gyro_y.push([sample.t_s, sample.gyro_radps[1].to_degrees()]);
            gyro_z.push([sample.t_s, sample.gyro_radps[2].to_degrees()]);
            accel_x.push([sample.t_s, sample.accel_mps2[0]]);
            accel_y.push([sample.t_s, sample.accel_mps2[1]]);
            accel_z.push([sample.t_s, sample.accel_mps2[2]]);
        }
        ReplayEvent::Gnss(index, sample) => {
            progress.report_stage(0.72, 0.26, sample.t_s);
            let _ = index;
            let _ = full_fusion.process_gnss(fusion_gnss_sample(*sample));
        }
    });

    data.full_cmp_pos = vec![
        Trace {
            name: "Full posN [m]".to_string(),
            points: pos_n,
        },
        Trace {
            name: "Full posE [m]".to_string(),
            points: pos_e,
        },
        Trace {
            name: "Full posD [m]".to_string(),
            points: pos_d,
        },
    ];
    data.full_cmp_vel = vec![
        Trace {
            name: "Full velN [m/s]".to_string(),
            points: vel_n,
        },
        Trace {
            name: "Full velE [m/s]".to_string(),
            points: vel_e,
        },
        Trace {
            name: "Full velD [m/s]".to_string(),
            points: vel_d,
        },
    ];
    data.full_cmp_att = vec![
        Trace {
            name: "Full roll [deg]".to_string(),
            points: roll,
        },
        Trace {
            name: "Full pitch [deg]".to_string(),
            points: pitch,
        },
        Trace {
            name: "Full yaw [deg]".to_string(),
            points: yaw,
        },
    ];
    data.full_misalignment = vec![
        Trace {
            name: "Full mount roll [deg]".to_string(),
            points: mount_roll,
        },
        Trace {
            name: "Full mount pitch [deg]".to_string(),
            points: mount_pitch,
        },
        Trace {
            name: "Full mount yaw [deg]".to_string(),
            points: mount_yaw,
        },
    ];
    data.full_meas_gyro = vec![
        Trace {
            name: "Full gyro x [deg/s]".to_string(),
            points: gyro_x,
        },
        Trace {
            name: "Full gyro y [deg/s]".to_string(),
            points: gyro_y,
        },
        Trace {
            name: "Full gyro z [deg/s]".to_string(),
            points: gyro_z,
        },
    ];
    data.full_meas_accel = vec![
        Trace {
            name: "Full accel x [m/s^2]".to_string(),
            points: accel_x,
        },
        Trace {
            name: "Full accel y [m/s^2]".to_string(),
            points: accel_y,
        },
        Trace {
            name: "Full accel z [m/s^2]".to_string(),
            points: accel_z,
        },
    ];
    data.vehicle_motion_gyro
        .extend(axis_traces_deg_per_s("Full angular velocity", motion_gyro));
    data.vehicle_motion_accel
        .extend(axis_traces_mps2("Full linear acceleration", motion_accel));
    data.full_bias_gyro = vec![
        Trace {
            name: "Full gyro sensor bias X [deg/s]".to_string(),
            points: bgx,
        },
        Trace {
            name: "Full gyro sensor bias Y [deg/s]".to_string(),
            points: bgy,
        },
        Trace {
            name: "Full gyro sensor bias Z [deg/s]".to_string(),
            points: bgz,
        },
    ];
    data.full_bias_accel = vec![
        Trace {
            name: "Full accel sensor bias X [m/s^2]".to_string(),
            points: bax,
        },
        Trace {
            name: "Full accel sensor bias Y [m/s^2]".to_string(),
            points: bay,
        },
        Trace {
            name: "Full accel sensor bias Z [m/s^2]".to_string(),
            points: accel_bias_z,
        },
    ];
    data.full_scale_gyro = vec![
        Trace {
            name: "Full sgx".to_string(),
            points: sgx,
        },
        Trace {
            name: "Full sgy".to_string(),
            points: sgy,
        },
        Trace {
            name: "Full sgz".to_string(),
            points: sgz,
        },
    ];
    data.full_scale_accel = vec![
        Trace {
            name: "Full sax".to_string(),
            points: sax,
        },
        Trace {
            name: "Full say".to_string(),
            points: say,
        },
        Trace {
            name: "Full saz".to_string(),
            points: saz,
        },
    ];
    data.full_cov_bias = cov_bias
        .into_iter()
        .enumerate()
        .map(|(i, points)| Trace {
            name: format!("Full sigma bias/scale {i}"),
            points,
        })
        .collect();
    data.full_cov_nonbias = cov_nonbias
        .into_iter()
        .enumerate()
        .map(|(i, points)| Trace {
            name: format!("Full sigma state {i}"),
            points,
        })
        .collect();
    data.full_mount_sigma = ["roll", "pitch", "yaw"]
        .into_iter()
        .zip(cov_mount)
        .map(|(axis, points)| Trace {
            name: format!("Full mount {axis} sigma [deg]"),
            points,
        })
        .collect();
    data.full_mount_dx = ["roll", "pitch", "yaw"]
        .into_iter()
        .zip(dx_mount)
        .map(|(axis, points)| Trace {
            name: format!("Full mount {axis} correction [deg/update]"),
            points,
        })
        .collect();
    data.full_nhc_innovation = ["Y", "Z"]
        .into_iter()
        .zip(nhc_innovation)
        .map(|(axis, points)| Trace {
            name: format!("Full NHC {axis} innovation [m/s]"),
            points,
        })
        .collect();
    data.full_gnss_pos_gate = ["row 0", "row 1", "row 2"]
        .into_iter()
        .zip(gnss_pos_gate_norm)
        .map(|(row, points)| Trace {
            name: format!("Full GNSS position gate normalized residual {row}"),
            points,
        })
        .chain(std::iter::once(Trace {
            name: "Full GNSS position accepted".to_string(),
            points: gnss_pos_gate_status,
        }))
        .collect();
    data.full_map = vec![Trace {
        name: "Full path (lon,lat)".to_string(),
        points: map,
    }];
    data.full_map_heading = headings;
}

fn populate_reduced_bump_traces(data: &mut PlotData) {
    let pitch = data
        .reduced_cmp_att
        .iter()
        .find(|t| t.name.to_ascii_lowercase().contains("pitch"))
        .map(|t| t.points.clone())
        .unwrap_or_default();
    let speed = data
        .speed
        .first()
        .map(|t| t.points.clone())
        .unwrap_or_default();
    data.reduced_bump_pitch_speed = vec![
        Trace {
            name: "Reduced pitch [deg]".to_string(),
            points: pitch.clone(),
        },
        Trace {
            name: "Reduced vehicle speed [m/s]".to_string(),
            points: speed,
        },
    ];
    let mut hpf = Vec::new();
    let mut abs_ema = Vec::new();
    let mut ema = 0.0;
    let mut rms_ema = 0.0;
    let alpha = 0.02;
    for [t, v] in pitch {
        ema = (1.0 - alpha) * ema + alpha * v;
        let hp = v - ema;
        rms_ema = (1.0 - alpha) * rms_ema + alpha * hp * hp;
        hpf.push([t, hp]);
        abs_ema.push([t, rms_ema.sqrt()]);
    }
    data.reduced_bump_diag = vec![
        Trace {
            name: "Reduced pitch HPF [deg]".to_string(),
            points: hpf,
        },
        Trace {
            name: "Reduced pitch RMS EMA [deg]".to_string(),
            points: abs_ema,
        },
    ];
}

fn reduced_nhc_mount_dx_traces(points: &[Vec<[f64; 2]>; 6]) -> Vec<Trace> {
    let mut traces = Vec::with_capacity(6);
    for (diag_i, (_, label)) in NHC_DIAG_TYPES.iter().copied().enumerate() {
        for (axis_i, axis) in ["roll", "pitch", "yaw"].into_iter().enumerate() {
            traces.push(Trace {
                name: format!("Reduced {label} mount {axis} correction [deg/update]"),
                points: points[diag_i * 3 + axis_i].clone(),
            });
        }
    }
    traces
}

fn append_reference_motion_traces(data: &mut PlotData, samples: &[GenericReferenceMotionSample]) {
    if samples.is_empty() {
        return;
    }
    let mut gyro: [Vec<[f64; 2]>; 3] = std::array::from_fn(|_| Vec::new());
    let mut accel: [Vec<[f64; 2]>; 3] = std::array::from_fn(|_| Vec::new());
    for sample in samples {
        push_motion_triplet(
            sample.t_s,
            sample.gyro_vehicle_radps.map(f64::to_degrees),
            &mut gyro,
        );
        push_motion_triplet(sample.t_s, sample.accel_vehicle_mps2, &mut accel);
    }
    let mut gyro_traces = axis_traces_deg_per_s("Reference angular velocity", gyro);
    gyro_traces.extend(std::mem::take(&mut data.vehicle_motion_gyro));
    data.vehicle_motion_gyro = gyro_traces;

    let mut accel_traces = axis_traces_mps2("Reference linear acceleration", accel);
    accel_traces.extend(std::mem::take(&mut data.vehicle_motion_accel));
    data.vehicle_motion_accel = accel_traces;
}

fn append_reduced_motion_sample(
    t_s: f64,
    sample: &GenericImuSample,
    fusion: &SensorFusion,
    gyro: &mut [Vec<[f64; 2]>; 3],
    accel: &mut [Vec<[f64; 2]>; 3],
) {
    let Some(reduced) = fusion.reduced() else {
        push_motion_triplet(t_s, [0.0; 3], gyro);
        push_motion_triplet(t_s, [0.0; 3], accel);
        return;
    };
    let n = &reduced.nominal;
    let q_bv = [
        n.q_bv0 as f64,
        n.q_bv1 as f64,
        n.q_bv2 as f64,
        n.q_bv3 as f64,
    ];
    let q_vehicle_to_ned = reduced_vehicle_attitude_q(reduced);
    let gyro_body = [
        sample.gyro_radps[0] - n.bgx as f64,
        sample.gyro_radps[1] - n.bgy as f64,
        sample.gyro_radps[2] - n.bgz as f64,
    ];
    let accel_body = [
        sample.accel_mps2[0] - n.bax as f64,
        sample.accel_mps2[1] - n.bay as f64,
        sample.accel_mps2[2] - n.baz as f64,
    ];
    let gyro_vehicle = rotate_body_to_vehicle(q_bv, gyro_body);
    let accel_vehicle = gravity_compensate_vehicle_accel(
        q_vehicle_to_ned,
        rotate_body_to_vehicle(q_bv, accel_body),
    );
    push_motion_triplet(t_s, gyro_vehicle.map(f64::to_degrees), gyro);
    push_motion_triplet(t_s, accel_vehicle, accel);
}

fn append_full_motion_sample(
    t_s: f64,
    sample: &GenericImuSample,
    full: Option<&State>,
    gyro: &mut [Vec<[f64; 2]>; 3],
    accel: &mut [Vec<[f64; 2]>; 3],
) {
    let Some(full) = full else {
        push_motion_triplet(t_s, [0.0; 3], gyro);
        push_motion_triplet(t_s, [0.0; 3], accel);
        return;
    };
    let n = &full.nominal;
    let q_bv = [
        n.q_bv0 as f64,
        n.q_bv1 as f64,
        n.q_bv2 as f64,
        n.q_bv3 as f64,
    ];
    let (lat_deg, lon_deg, _) = ecef_to_lla(full.pos_e64);
    let q_vehicle_to_ned = quat_mul(
        quat_ecef_to_ned(lat_deg, lon_deg),
        [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64],
    );
    let gyro_body = [
        n.sgx as f64 * sample.gyro_radps[0] + n.bgx as f64,
        n.sgy as f64 * sample.gyro_radps[1] + n.bgy as f64,
        n.sgz as f64 * sample.gyro_radps[2] + n.bgz as f64,
    ];
    let accel_body = [
        n.sax as f64 * sample.accel_mps2[0] + n.bax as f64,
        n.say as f64 * sample.accel_mps2[1] + n.bay as f64,
        n.saz as f64 * sample.accel_mps2[2] + n.baz as f64,
    ];
    let gyro_vehicle = rotate_body_to_vehicle(q_bv, gyro_body);
    let accel_vehicle = gravity_compensate_vehicle_accel(
        q_vehicle_to_ned,
        rotate_body_to_vehicle(q_bv, accel_body),
    );
    push_motion_triplet(t_s, gyro_vehicle.map(f64::to_degrees), gyro);
    push_motion_triplet(t_s, accel_vehicle, accel);
}

fn rotate_body_to_vehicle(q_bv: [f64; 4], value_body: [f64; 3]) -> [f64; 3] {
    quat_rotate(quat_conj(q_bv), value_body)
}

fn gravity_compensate_vehicle_accel(
    q_vehicle_to_ned: [f64; 4],
    specific_force_vehicle: [f64; 3],
) -> [f64; 3] {
    let gravity_vehicle = quat_rotate(
        quat_conj(q_vehicle_to_ned),
        [0.0, 0.0, STANDARD_GRAVITY_MPS2],
    );
    [
        specific_force_vehicle[0] + gravity_vehicle[0],
        specific_force_vehicle[1] + gravity_vehicle[1],
        specific_force_vehicle[2] + gravity_vehicle[2],
    ]
}

fn push_motion_triplet(t_s: f64, values: [f64; 3], traces: &mut [Vec<[f64; 2]>; 3]) {
    for (trace, value) in traces.iter_mut().zip(values) {
        trace.push([t_s, value]);
    }
}

fn axis_traces_deg_per_s(prefix: &str, points: [Vec<[f64; 2]>; 3]) -> Vec<Trace> {
    axis_traces(prefix, "[deg/s]", points)
}

fn axis_traces_mps2(prefix: &str, points: [Vec<[f64; 2]>; 3]) -> Vec<Trace> {
    axis_traces(prefix, "[m/s^2]", points)
}

fn axis_traces(prefix: &str, unit: &str, points: [Vec<[f64; 2]>; 3]) -> Vec<Trace> {
    ["X", "Y", "Z"]
        .into_iter()
        .zip(points)
        .map(|(axis, points)| Trace {
            name: format!("{prefix} {axis} {unit}"),
            points,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn append_reduced_sample(
    t_s: f64,
    fusion: &SensorFusion,
    ref_gnss: Option<GenericGnssSample>,
    ref_ecef: Option<[f64; 3]>,
    pos_n: &mut Vec<[f64; 2]>,
    pos_e: &mut Vec<[f64; 2]>,
    pos_d: &mut Vec<[f64; 2]>,
    vel_n: &mut Vec<[f64; 2]>,
    vel_e: &mut Vec<[f64; 2]>,
    vel_d: &mut Vec<[f64; 2]>,
    roll: &mut Vec<[f64; 2]>,
    pitch: &mut Vec<[f64; 2]>,
    yaw: &mut Vec<[f64; 2]>,
    mount_roll: &mut Vec<[f64; 2]>,
    mount_pitch: &mut Vec<[f64; 2]>,
    mount_yaw: &mut Vec<[f64; 2]>,
    bgx: &mut Vec<[f64; 2]>,
    bgy: &mut Vec<[f64; 2]>,
    bgz: &mut Vec<[f64; 2]>,
    accel_bias_x: &mut Vec<[f64; 2]>,
    accel_bias_y: &mut Vec<[f64; 2]>,
    accel_bias_z: &mut Vec<[f64; 2]>,
    cov: &mut [Vec<[f64; 2]>; 18],
    mount_dx: &mut [Vec<[f64; 2]>; 3],
    last_update_count: &mut u32,
    nhc_mount_dx: &mut [Vec<[f64; 2]>; 6],
    nhc_innovation: &mut [Vec<[f64; 2]>; 2],
    nhc_nis: &mut [Vec<[f64; 2]>; 2],
    nhc_h_mount_norm: &mut [Vec<[f64; 2]>; 2],
    update_inspector: &mut Vec<UpdateInspectorSample>,
    last_type_counts: &mut [u32; UPDATE_DIAG_TYPES],
    map: &mut Vec<[f64; 2]>,
    outage_map: &mut Vec<[f64; 2]>,
    headings: &mut Vec<HeadingSample>,
    map_cursor: &mut Vec<MapCursorSample>,
    outage_active: bool,
) {
    let Some(reduced) = fusion.reduced() else {
        return;
    };
    let display_pos = reduced_display_position_ned(fusion, reduced, ref_gnss, ref_ecef);
    let display_vel = reduced_display_velocity_ned(fusion, reduced, ref_gnss);
    pos_n.push([t_s, display_pos[0]]);
    pos_e.push([t_s, display_pos[1]]);
    pos_d.push([t_s, display_pos[2]]);
    vel_n.push([t_s, display_vel[0]]);
    vel_e.push([t_s, display_vel[1]]);
    vel_d.push([t_s, display_vel[2]]);

    let q_vehicle = reduced_vehicle_attitude_q(reduced);
    let (r, p, y) = quat_rpy_deg(
        q_vehicle[0] as f32,
        q_vehicle[1] as f32,
        q_vehicle[2] as f32,
        q_vehicle[3] as f32,
    );
    roll.push([t_s, r]);
    pitch.push([t_s, p]);
    yaw.push([t_s, y]);

    let q_bv = [
        reduced.nominal.q_bv0 as f64,
        reduced.nominal.q_bv1 as f64,
        reduced.nominal.q_bv2 as f64,
        reduced.nominal.q_bv3 as f64,
    ];
    let (mr, mp, my) = q_bv_to_reference_mount_rpy(q_bv);
    mount_roll.push([t_s, mr]);
    mount_pitch.push([t_s, mp]);
    mount_yaw.push([t_s, my]);

    bgx.push([t_s, (reduced.nominal.bgx as f64).to_degrees()]);
    bgy.push([t_s, (reduced.nominal.bgy as f64).to_degrees()]);
    bgz.push([t_s, (reduced.nominal.bgz as f64).to_degrees()]);
    accel_bias_x.push([t_s, reduced.nominal.bax as f64]);
    accel_bias_y.push([t_s, reduced.nominal.bay as f64]);
    accel_bias_z.push([t_s, reduced.nominal.baz as f64]);
    for (i, trace) in cov.iter_mut().enumerate() {
        trace.push([t_s, reduced.p[i][i].max(0.0).sqrt() as f64]);
    }
    let diag = reduced.update_diag;
    if diag.total_updates != *last_update_count {
        mount_dx[0].push([t_s, (diag.last_dx_mount_roll as f64).to_degrees()]);
        mount_dx[1].push([t_s, (diag.last_dx_mount_pitch as f64).to_degrees()]);
        mount_dx[2].push([t_s, (diag.last_dx_mount_yaw as f64).to_degrees()]);
        *last_update_count = diag.total_updates;
    }
    for (diag_i, (diag_type, label)) in NHC_DIAG_TYPES.iter().copied().enumerate() {
        if diag.type_counts[diag_type] != last_type_counts[diag_type] {
            nhc_mount_dx[diag_i * 3].push([
                t_s,
                (diag.last_dx_mount_roll_by_type[diag_type] as f64).to_degrees(),
            ]);
            nhc_mount_dx[diag_i * 3 + 1].push([
                t_s,
                (diag.last_dx_mount_pitch_by_type[diag_type] as f64).to_degrees(),
            ]);
            nhc_mount_dx[diag_i * 3 + 2].push([
                t_s,
                (diag.last_dx_mount_yaw_by_type[diag_type] as f64).to_degrees(),
            ]);
            nhc_innovation[diag_i].push([t_s, diag.last_innovation_by_type[diag_type] as f64]);
            nhc_nis[diag_i].push([t_s, diag.last_nis_by_type[diag_type] as f64]);
            nhc_h_mount_norm[diag_i].push([t_s, diag.last_h_mount_norm_by_type[diag_type] as f64]);
            update_inspector.push(reduced_nhc_inspector_sample(
                t_s,
                label,
                diag.last_innovation_by_type[diag_type],
                diag.last_nis_by_type[diag_type],
                diag.last_dx_mount_roll_by_type[diag_type],
                diag.last_dx_mount_pitch_by_type[diag_type],
                diag.last_dx_mount_yaw_by_type[diag_type],
                &reduced.p,
            ));
            last_type_counts[diag_type] = diag.type_counts[diag_type];
        }
    }

    if let Some([lat, lon, _]) = fusion.position_lla_f64() {
        map.push([lon, lat]);
        if outage_active {
            outage_map.push([lon, lat]);
        } else if outage_map
            .last()
            .map(|p| p[0].is_finite() || p[1].is_finite())
            .unwrap_or(false)
        {
            outage_map.push([f64::NAN, f64::NAN]);
        }
        headings.push(HeadingSample {
            t_s,
            lon_deg: lon,
            lat_deg: lat,
            yaw_deg: y,
        });
        map_cursor.push(MapCursorSample {
            trace_name: "Reduced path (lon,lat)".to_string(),
            t_s,
            lon_deg: lon,
            lat_deg: lat,
            yaw_deg: Some(y),
        });
        if outage_active {
            map_cursor.push(MapCursorSample {
                trace_name: "Reduced path during GNSS outage (lon,lat)".to_string(),
                t_s,
                lon_deg: lon,
                lat_deg: lat,
                yaw_deg: Some(y),
            });
        }
    }
}

fn reduced_display_position_ned(
    fusion: &SensorFusion,
    reduced: &sensor_fusion::reduced::State,
    ref_gnss: Option<GenericGnssSample>,
    ref_ecef: Option<[f64; 3]>,
) -> [f64; 3] {
    if let (Some([lat, lon, h]), Some(ref_sample), Some(ref_ecef)) =
        (fusion.position_lla_f64(), ref_gnss, ref_ecef)
    {
        let ecef = lla_to_ecef(lat, lon, h);
        return ecef_to_ned(ecef, ref_ecef, ref_sample.lat_deg, ref_sample.lon_deg);
    }
    [
        reduced.nominal.pn as f64,
        reduced.nominal.pe as f64,
        reduced.nominal.pd as f64,
    ]
}

fn reduced_display_velocity_ned(
    fusion: &SensorFusion,
    reduced: &sensor_fusion::reduced::State,
    ref_gnss: Option<GenericGnssSample>,
) -> [f64; 3] {
    if let (Some(anchor), Some(ref_sample)) = (fusion.anchor_lla_debug(), ref_gnss) {
        let vel_ecef = ned_vector_to_ecef(
            anchor[0] as f64,
            anchor[1] as f64,
            [
                reduced.nominal.vn as f64,
                reduced.nominal.ve as f64,
                reduced.nominal.vd as f64,
            ],
        );
        return ecef_vector_to_ned(ref_sample.lat_deg, ref_sample.lon_deg, vel_ecef);
    }
    [
        reduced.nominal.vn as f64,
        reduced.nominal.ve as f64,
        reduced.nominal.vd as f64,
    ]
}

fn reduced_vehicle_attitude_q(reduced: &sensor_fusion::reduced::State) -> [f64; 4] {
    as_q64([
        reduced.nominal.q0,
        reduced.nominal.q1,
        reduced.nominal.q2,
        reduced.nominal.q3,
    ])
}

#[allow(clippy::too_many_arguments)]
fn append_full_sample(
    t_s: f64,
    full: &State,
    ref_gnss: GenericGnssSample,
    ref_ecef: [f64; 3],
    pos_n: &mut Vec<[f64; 2]>,
    pos_e: &mut Vec<[f64; 2]>,
    pos_d: &mut Vec<[f64; 2]>,
    vel_n: &mut Vec<[f64; 2]>,
    vel_e: &mut Vec<[f64; 2]>,
    vel_d: &mut Vec<[f64; 2]>,
    roll: &mut Vec<[f64; 2]>,
    pitch: &mut Vec<[f64; 2]>,
    yaw: &mut Vec<[f64; 2]>,
    mount_roll: &mut Vec<[f64; 2]>,
    mount_pitch: &mut Vec<[f64; 2]>,
    mount_yaw: &mut Vec<[f64; 2]>,
    bgx: &mut Vec<[f64; 2]>,
    bgy: &mut Vec<[f64; 2]>,
    bgz: &mut Vec<[f64; 2]>,
    bax: &mut Vec<[f64; 2]>,
    bay: &mut Vec<[f64; 2]>,
    accel_bias_z: &mut Vec<[f64; 2]>,
    sgx: &mut Vec<[f64; 2]>,
    sgy: &mut Vec<[f64; 2]>,
    sgz: &mut Vec<[f64; 2]>,
    sax: &mut Vec<[f64; 2]>,
    say: &mut Vec<[f64; 2]>,
    saz: &mut Vec<[f64; 2]>,
    cov_bias: &mut [Vec<[f64; 2]>; 12],
    cov_nonbias: &mut [Vec<[f64; 2]>; 12],
    cov_mount: &mut [Vec<[f64; 2]>; 3],
    dx_mount: &mut [Vec<[f64; 2]>; 3],
    nhc_innovation: &mut [Vec<[f64; 2]>; 2],
    gnss_pos_gate_norm: &mut [Vec<[f64; 2]>; 3],
    gnss_pos_gate_status: &mut Vec<[f64; 2]>,
    update_inspector: &mut Vec<UpdateInspectorSample>,
    map: &mut Vec<[f64; 2]>,
    headings: &mut Vec<HeadingSample>,
    map_cursor: &mut Vec<MapCursorSample>,
) {
    let n = &full.nominal;
    let pos_ecef = full.pos_e64;
    let vel_ecef = [n.vn as f64, n.ve as f64, n.vd as f64];
    let pos = ecef_to_ned(pos_ecef, ref_ecef, ref_gnss.lat_deg, ref_gnss.lon_deg);
    let vel = ecef_vector_to_ned(ref_gnss.lat_deg, ref_gnss.lon_deg, vel_ecef);
    pos_n.push([t_s, pos[0]]);
    pos_e.push([t_s, pos[1]]);
    pos_d.push([t_s, pos[2]]);
    vel_n.push([t_s, vel[0]]);
    vel_e.push([t_s, vel[1]]);
    vel_d.push([t_s, vel[2]]);

    let (lat, lon, _) = ned_to_lla_exact(
        pos[0],
        pos[1],
        pos[2],
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
        ref_gnss.height_m,
    );
    map.push([lon, lat]);
    let q_ne = quat_ecef_to_ned(lat, lon);
    let q_ev = [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64];
    let q_bv = [
        n.q_bv0 as f64,
        n.q_bv1 as f64,
        n.q_bv2 as f64,
        n.q_bv3 as f64,
    ];
    let q_ns = quat_mul(q_ne, q_ev);
    let q_vehicle = q_ns;
    let (r, p, y) = quat_rpy_deg(
        q_vehicle[0] as f32,
        q_vehicle[1] as f32,
        q_vehicle[2] as f32,
        q_vehicle[3] as f32,
    );
    roll.push([t_s, r]);
    pitch.push([t_s, p]);
    yaw.push([t_s, y]);
    headings.push(HeadingSample {
        t_s,
        lon_deg: lon,
        lat_deg: lat,
        yaw_deg: y,
    });
    map_cursor.push(MapCursorSample {
        trace_name: "Full path (lon,lat)".to_string(),
        t_s,
        lon_deg: lon,
        lat_deg: lat,
        yaw_deg: Some(y),
    });

    let (mr, mp, my) = q_bv_to_reference_mount_rpy(q_bv);
    mount_roll.push([t_s, mr]);
    mount_pitch.push([t_s, mp]);
    mount_yaw.push([t_s, my]);

    let gyro_sensor_bias_dps = full_gyro_sensor_bias_dps(n);
    let accel_sensor_bias_mps2 = full_accel_sensor_bias_mps2(n);
    bgx.push([t_s, gyro_sensor_bias_dps[0]]);
    bgy.push([t_s, gyro_sensor_bias_dps[1]]);
    bgz.push([t_s, gyro_sensor_bias_dps[2]]);
    bax.push([t_s, accel_sensor_bias_mps2[0]]);
    bay.push([t_s, accel_sensor_bias_mps2[1]]);
    accel_bias_z.push([t_s, accel_sensor_bias_mps2[2]]);
    sgx.push([t_s, n.sgx as f64]);
    sgy.push([t_s, n.sgy as f64]);
    sgz.push([t_s, n.sgz as f64]);
    sax.push([t_s, n.sax as f64]);
    say.push([t_s, n.say as f64]);
    saz.push([t_s, n.saz as f64]);
    let pmat = &full.p;
    for (dst, idx) in cov_bias
        .iter_mut()
        .zip([12usize, 13, 14, 9, 10, 11, 15, 16, 17, 18, 19, 20])
    {
        dst.push([t_s, pmat[idx][idx].max(0.0).sqrt() as f64]);
    }
    for (idx, dst) in cov_nonbias.iter_mut().enumerate() {
        dst.push([t_s, pmat[idx][idx].max(0.0).sqrt() as f64]);
    }
    for (dst, idx) in cov_mount.iter_mut().zip([21usize, 22, 23]) {
        dst.push([t_s, (pmat[idx][idx].max(0.0).sqrt() as f64).to_degrees()]);
    }
    let dx = &full.last_dx;
    let obs_count = full
        .last_obs_count
        .clamp(0, full.last_obs_types.len() as i32) as usize;
    let obs_types = &full.last_obs_types[..obs_count];
    let has_update = obs_count > 0;
    if has_update {
        for (dst, idx) in dx_mount.iter_mut().zip([21usize, 22, 23]) {
            dst.push([t_s, (dx[idx] as f64).to_degrees()]);
        }
        update_inspector.push(full_inspector_sample(t_s, full));
        for (&obs_type, &residual) in obs_types.iter().zip(full.last_residuals.iter()) {
            if obs_type == 7 {
                nhc_innovation[0].push([t_s, residual as f64]);
            } else if obs_type == 8 {
                nhc_innovation[1].push([t_s, residual as f64]);
            }
        }
    }
    let gnss_gate = full.last_gnss_pos_gate;
    if gnss_gate.attempted {
        for (dst, value) in gnss_pos_gate_norm
            .iter_mut()
            .zip(gnss_gate.normalized_residual)
        {
            dst.push([t_s, value as f64]);
        }
        gnss_pos_gate_status.push([t_s, if gnss_gate.accepted { 1.0 } else { 0.0 }]);
    }
}

#[allow(clippy::too_many_arguments)]
fn reduced_nhc_inspector_sample(
    t_s: f64,
    label: &str,
    innovation: f32,
    nis: f32,
    mount_roll_dx_rad: f32,
    mount_pitch_dx_rad: f32,
    mount_yaw_dx_rad: f32,
    p: &[[f32; 18]; 18],
) -> UpdateInspectorSample {
    let contributions = [
        ("mount roll", mount_roll_dx_rad),
        ("mount pitch", mount_pitch_dx_rad),
        ("mount yaw", mount_yaw_dx_rad),
    ]
    .into_iter()
    .map(|(state, value)| StateContribution {
        state: state.to_string(),
        group: "mount".to_string(),
        unit: "deg".to_string(),
        value: (value as f64).to_degrees(),
    })
    .collect();
    UpdateInspectorSample {
        t_s,
        filter: "Reduced".to_string(),
        update: label.to_string(),
        residual: Some(innovation as f64),
        nis: Some(nis as f64),
        contributions,
        correlations: reduced_mount_correlations(p),
    }
}

fn full_inspector_sample(t_s: f64, full: &State) -> UpdateInspectorSample {
    let obs_count = full
        .last_obs_count
        .clamp(0, full.last_obs_types.len() as i32) as usize;
    let obs_types = &full.last_obs_types[..obs_count];
    let update = if obs_types.contains(&7) || obs_types.contains(&8) {
        "batch + NHC"
    } else {
        "GNSS batch"
    };
    UpdateInspectorSample {
        t_s,
        filter: "Full".to_string(),
        update: update.to_string(),
        residual: None,
        nis: None,
        contributions: full_state_contributions(&full.last_dx),
        correlations: full_mount_correlations(&full.p),
    }
}

fn reduced_mount_correlations(p: &[[f32; 18]; 18]) -> Vec<StateCorrelation> {
    const STATES: [(&str, &str, usize, f64); 18] = [
        ("att roll", "attitude", 0, 1.0),
        ("att pitch", "attitude", 1, 1.0),
        ("att yaw", "attitude", 2, 1.0),
        ("vel N", "velocity", 3, 1.0),
        ("vel E", "velocity", 4, 1.0),
        ("vel D", "velocity", 5, 1.0),
        ("pos N", "position", 6, 1.0),
        ("pos E", "position", 7, 1.0),
        ("pos D", "position", 8, 1.0),
        ("gyro bias X", "gyro bias", 9, 1.0),
        ("gyro bias Y", "gyro bias", 10, 1.0),
        ("gyro bias Z", "gyro bias", 11, 1.0),
        ("accel bias X", "accel bias", 12, 1.0),
        ("accel bias Y", "accel bias", 13, 1.0),
        ("accel bias Z", "accel bias", 14, 1.0),
        ("mount roll", "mount", 15, 1.0),
        ("mount pitch", "mount", 16, 1.0),
        ("mount yaw", "mount", 17, 1.0),
    ];
    covariance_mount_correlations(
        p,
        &STATES,
        &[(15, "roll", 1.0), (16, "pitch", 1.0), (17, "yaw", 1.0)],
    )
}

fn full_mount_correlations(p: &[[f32; ERROR_STATES]; ERROR_STATES]) -> Vec<StateCorrelation> {
    const STATES: [(&str, &str, usize, f64); ERROR_STATES] = [
        ("pos X", "position", 0, 1.0),
        ("pos Y", "position", 1, 1.0),
        ("pos Z", "position", 2, 1.0),
        ("vel X", "velocity", 3, 1.0),
        ("vel Y", "velocity", 4, 1.0),
        ("vel Z", "velocity", 5, 1.0),
        ("att roll", "attitude", 6, 1.0),
        ("att pitch", "attitude", 7, 1.0),
        ("att yaw", "attitude", 8, 1.0),
        ("accel sensor bias X", "accel sensor bias", 9, -1.0),
        ("accel sensor bias Y", "accel sensor bias", 10, -1.0),
        ("accel sensor bias Z", "accel sensor bias", 11, -1.0),
        ("gyro sensor bias X", "gyro sensor bias", 12, -1.0),
        ("gyro sensor bias Y", "gyro sensor bias", 13, -1.0),
        ("gyro sensor bias Z", "gyro sensor bias", 14, -1.0),
        ("accel scale X", "accel scale", 15, 1.0),
        ("accel scale Y", "accel scale", 16, 1.0),
        ("accel scale Z", "accel scale", 17, 1.0),
        ("gyro scale X", "gyro scale", 18, 1.0),
        ("gyro scale Y", "gyro scale", 19, 1.0),
        ("gyro scale Z", "gyro scale", 20, 1.0),
        ("mount roll", "mount", 21, 1.0),
        ("mount pitch", "mount", 22, 1.0),
        ("mount yaw", "mount", 23, 1.0),
    ];
    covariance_mount_correlations(
        p,
        &STATES,
        &[(21, "roll", 1.0), (22, "pitch", 1.0), (23, "yaw", 1.0)],
    )
}

fn covariance_mount_correlations<const N: usize>(
    p: &[[f32; N]; N],
    states: &[(&str, &str, usize, f64)],
    mount_axes: &[(usize, &str, f64)],
) -> Vec<StateCorrelation> {
    let mut correlations = Vec::new();
    for &(mount_idx, mount_axis, mount_sign) in mount_axes {
        for &(state, group, idx, state_sign) in states {
            if idx == mount_idx {
                continue;
            }
            let value = state_sign * mount_sign * covariance_correlation(p, idx, mount_idx);
            if !value.is_finite() {
                continue;
            }
            correlations.push(StateCorrelation {
                state: state.to_string(),
                group: group.to_string(),
                mount_axis: mount_axis.to_string(),
                value,
            });
        }
    }
    correlations.sort_by(|a, b| {
        b.value
            .abs()
            .total_cmp(&a.value.abs())
            .then_with(|| a.mount_axis.cmp(&b.mount_axis))
            .then_with(|| a.state.cmp(&b.state))
    });
    correlations.truncate(6);
    correlations
}

fn covariance_correlation<const N: usize>(p: &[[f32; N]; N], i: usize, j: usize) -> f64 {
    let pii = p[i][i].max(0.0) as f64;
    let pjj = p[j][j].max(0.0) as f64;
    let denom = (pii * pjj).sqrt();
    if denom <= 0.0 {
        return f64::NAN;
    }
    (p[i][j] as f64 / denom).clamp(-1.0, 1.0)
}

fn full_state_contributions(dx: &[f32; ERROR_STATES]) -> Vec<StateContribution> {
    const STATES: [(&str, &str, &str, usize, f64); 24] = [
        ("pos X", "position", "m", 0, 1.0),
        ("pos Y", "position", "m", 1, 1.0),
        ("pos Z", "position", "m", 2, 1.0),
        ("vel X", "velocity", "m/s", 3, 1.0),
        ("vel Y", "velocity", "m/s", 4, 1.0),
        ("vel Z", "velocity", "m/s", 5, 1.0),
        (
            "att roll",
            "attitude",
            "deg",
            6,
            180.0 / core::f64::consts::PI,
        ),
        (
            "att pitch",
            "attitude",
            "deg",
            7,
            180.0 / core::f64::consts::PI,
        ),
        (
            "att yaw",
            "attitude",
            "deg",
            8,
            180.0 / core::f64::consts::PI,
        ),
        ("accel sensor bias X", "accel sensor bias", "m/s^2", 9, -1.0),
        (
            "accel sensor bias Y",
            "accel sensor bias",
            "m/s^2",
            10,
            -1.0,
        ),
        (
            "accel sensor bias Z",
            "accel sensor bias",
            "m/s^2",
            11,
            -1.0,
        ),
        (
            "gyro sensor bias X",
            "gyro sensor bias",
            "deg/s",
            12,
            -180.0 / core::f64::consts::PI,
        ),
        (
            "gyro sensor bias Y",
            "gyro sensor bias",
            "deg/s",
            13,
            -180.0 / core::f64::consts::PI,
        ),
        (
            "gyro sensor bias Z",
            "gyro sensor bias",
            "deg/s",
            14,
            -180.0 / core::f64::consts::PI,
        ),
        ("accel scale X", "accel scale", "", 15, 1.0),
        ("accel scale Y", "accel scale", "", 16, 1.0),
        ("accel scale Z", "accel scale", "", 17, 1.0),
        ("gyro scale X", "gyro scale", "", 18, 1.0),
        ("gyro scale Y", "gyro scale", "", 19, 1.0),
        ("gyro scale Z", "gyro scale", "", 20, 1.0),
        (
            "mount roll",
            "mount",
            "deg",
            21,
            180.0 / core::f64::consts::PI,
        ),
        (
            "mount pitch",
            "mount",
            "deg",
            22,
            180.0 / core::f64::consts::PI,
        ),
        (
            "mount yaw",
            "mount",
            "deg",
            23,
            180.0 / core::f64::consts::PI,
        ),
    ];
    STATES
        .into_iter()
        .map(|(state, group, unit, idx, scale)| StateContribution {
            state: state.to_string(),
            group: group.to_string(),
            unit: unit.to_string(),
            value: dx[idx] as f64 * scale,
        })
        .collect()
}

fn full_gyro_sensor_bias_dps(n: &NominalState) -> [f64; 3] {
    [
        -(n.bgx as f64).to_degrees(),
        -(n.bgy as f64).to_degrees(),
        -(n.bgz as f64).to_degrees(),
    ]
}

fn sigma_rad_points_to_deg(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    points.iter().map(|p| [p[0], p[1].to_degrees()]).collect()
}

fn full_accel_sensor_bias_mps2(n: &NominalState) -> [f64; 3] {
    [-(n.bax as f64), -(n.bay as f64), -(n.baz as f64)]
}

fn push_update_contrib(
    t: f64,
    q_start: [f32; 4],
    q_after: Option<[f32; 4]>,
    roll: &mut Vec<[f64; 2]>,
    pitch: &mut Vec<[f64; 2]>,
    yaw: &mut Vec<[f64; 2]>,
) {
    let Some(q_after) = q_after else {
        roll.push([t, 0.0]);
        pitch.push([t, 0.0]);
        yaw.push([t, 0.0]);
        return;
    };
    let before = quat_rpy_deg(q_start[0], q_start[1], q_start[2], q_start[3]);
    let after = quat_rpy_deg(q_after[0], q_after[1], q_after[2], q_after[3]);
    roll.push([t, wrap_deg(after.0 - before.0)]);
    pitch.push([t, wrap_deg(after.1 - before.1)]);
    yaw.push([t, wrap_deg(after.2 - before.2)]);
}

fn vec3_norm_f32(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn wrap_deg(mut x: f64) -> f64 {
    while x > 180.0 {
        x -= 360.0;
    }
    while x < -180.0 {
        x += 360.0;
    }
    x
}

fn ned_vector_to_ecef(lat_deg: f64, lon_deg: f64, v_ned: [f64; 3]) -> [f64; 3] {
    let c_ne = ecef_to_ned_matrix(lat_deg, lon_deg);
    [
        c_ne[0][0] * v_ned[0] + c_ne[1][0] * v_ned[1] + c_ne[2][0] * v_ned[2],
        c_ne[0][1] * v_ned[0] + c_ne[1][1] * v_ned[1] + c_ne[2][1] * v_ned[2],
        c_ne[0][2] * v_ned[0] + c_ne[1][2] * v_ned[1] + c_ne[2][2] * v_ned[2],
    ]
}

fn ecef_vector_to_ned(lat_deg: f64, lon_deg: f64, v_ecef: [f64; 3]) -> [f64; 3] {
    let c_ne = ecef_to_ned_matrix(lat_deg, lon_deg);
    [
        c_ne[0][0] * v_ecef[0] + c_ne[0][1] * v_ecef[1] + c_ne[0][2] * v_ecef[2],
        c_ne[1][0] * v_ecef[0] + c_ne[1][1] * v_ecef[1] + c_ne[1][2] * v_ecef[2],
        c_ne[2][0] * v_ecef[0] + c_ne[2][1] * v_ecef[1] + c_ne[2][2] * v_ecef[2],
    ]
}

fn ecef_to_ned_matrix(lat_deg: f64, lon_deg: f64) -> [[f64; 3]; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    [
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [-clat * clon, -clat * slon, -slat],
    ]
}

fn quat_ecef_to_ned(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    dcm_to_quat(ecef_to_ned_matrix(lat_deg, lon_deg))
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

fn apply_fusion_config(fusion: &mut SensorFusion, cfg: FilterCompareConfig) {
    crate::visualizer::pipeline::apply_filter_compare_config(fusion, cfg);
}

fn parse_imu_csv(text: &str) -> Result<Vec<GenericImuSample>> {
    let rows = parse_numeric_rows(text, 7, "imu.csv")?;
    Ok(rows
        .into_iter()
        .map(|row| GenericImuSample {
            t_s: row[0],
            gyro_radps: [row[1], row[2], row[3]],
            accel_mps2: [row[4], row[5], row[6]],
        })
        .collect())
}

fn parse_gnss_csv(text: &str) -> Result<Vec<GenericGnssSample>> {
    let rows = parse_numeric_rows_range(text, 13..=14, "gnss.csv")?;
    Ok(rows
        .into_iter()
        .map(|row| GenericGnssSample {
            t_s: row[0],
            lat_deg: row[1],
            lon_deg: row[2],
            height_m: row[3],
            vel_ned_mps: [row[4], row[5], row[6]],
            pos_std_m: [row[7], row[8], row[9]],
            vel_std_mps: [row[10], row[11], row[12]],
            heading_rad: row.get(13).copied().filter(|v| v.is_finite()),
        })
        .collect())
}

fn parse_reference_rpy_csv(text: &str) -> Result<Vec<GenericReferenceRpySample>> {
    let rows = parse_numeric_rows(text, 4, "reference_rpy.csv")?;
    Ok(rows
        .into_iter()
        .map(|row| GenericReferenceRpySample {
            t_s: row[0],
            roll_deg: row[1],
            pitch_deg: row[2],
            yaw_deg: row[3],
        })
        .collect())
}

fn parse_reference_position_csv(text: &str) -> Result<Vec<GenericReferencePositionSample>> {
    let rows = parse_numeric_rows_range(text, 7..=8, "reference_position.csv")?;
    Ok(rows
        .into_iter()
        .map(|row| GenericReferencePositionSample {
            t_s: row[0],
            lat_deg: row[1],
            lon_deg: row[2],
            height_m: row[3],
            vel_ned_mps: [row[4], row[5], row[6]],
            heading_rad: row.get(7).copied().filter(|v| v.is_finite()),
        })
        .collect())
}

fn parse_reference_motion_csv(text: &str) -> Result<Vec<GenericReferenceMotionSample>> {
    let rows = parse_numeric_rows(text, 7, "reference_motion.csv")?;
    Ok(rows
        .into_iter()
        .map(|row| GenericReferenceMotionSample {
            t_s: row[0],
            gyro_vehicle_radps: [row[1], row[2], row[3]],
            accel_vehicle_mps2: [row[4], row[5], row[6]],
        })
        .collect())
}

fn heading_deg_from_sample(heading_rad: Option<f64>, vel_ned_mps: [f64; 3]) -> Option<f64> {
    heading_rad
        .map(f64::to_degrees)
        .or_else(|| {
            let speed = vel_ned_mps[0].hypot(vel_ned_mps[1]);
            (speed > 0.2).then(|| vel_ned_mps[1].atan2(vel_ned_mps[0]).to_degrees())
        })
        .filter(|heading| heading.is_finite())
}

fn parse_numeric_rows(text: &str, cols: usize, label: &str) -> Result<Vec<Vec<f64>>> {
    parse_numeric_rows_range(text, cols..=cols, label)
}

fn parse_numeric_rows_range(
    text: &str,
    cols: std::ops::RangeInclusive<usize>,
    label: &str,
) -> Result<Vec<Vec<f64>>> {
    let mut out = Vec::new();
    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parsed = trimmed
            .split(',')
            .map(|part| {
                let value = part.trim();
                if value.eq_ignore_ascii_case("nan") {
                    Ok(f64::NAN)
                } else {
                    value
                        .parse::<f64>()
                        .with_context(|| format!("{label}: bad numeric field '{value}'"))
                }
            })
            .collect::<Result<Vec<_>>>();
        let row = match parsed {
            Ok(row) => row,
            Err(err) if out.is_empty() => {
                let _ = err;
                continue;
            }
            Err(err) => return Err(err),
        };
        if !cols.contains(&row.len()) {
            bail!(
                "{label}: line {} expected {} columns, got {}",
                line_idx + 1,
                if cols.start() == cols.end() {
                    cols.start().to_string()
                } else {
                    format!("{}..={}", cols.start(), cols.end())
                },
                row.len()
            );
        }
        out.push(row);
    }
    Ok(out)
}

fn sample_outage_windows(gnss: &[GenericGnssSample], cfg: GnssOutageConfig) -> Vec<(f64, f64)> {
    if cfg.count == 0 || cfg.duration_s <= 0.0 || gnss.len() < 2 {
        return Vec::new();
    }
    let t_min = gnss.first().map(|s| s.t_s).unwrap_or(0.0);
    let t_max = gnss.last().map(|s| s.t_s).unwrap_or(t_min);
    if t_max - t_min <= cfg.duration_s {
        return Vec::new();
    }
    let mut rng = Lcg64::new(cfg.seed);
    let mut windows = Vec::new();
    let max_attempts = cfg.count.saturating_mul(200).max(200);
    for _ in 0..max_attempts {
        if windows.len() >= cfg.count {
            break;
        }
        let start = t_min + rng.next_unit_f64() * (t_max - t_min - cfg.duration_s);
        let end = start + cfg.duration_s;
        if windows.iter().any(|(a, b)| start < *b && end > *a) {
            continue;
        }
        windows.push((start, end));
    }
    windows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    windows
}

fn in_outage(t_s: f64, windows: &[(f64, f64)]) -> bool {
    windows.iter().any(|(a, b)| t_s >= *a && t_s <= *b)
}

struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_unit_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = self.state >> 11;
        (v as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_bias_display_converts_correction_state_to_sensor_bias() {
        let nominal = NominalState {
            bgx: 1.0_f32.to_radians(),
            bgy: -2.0_f32.to_radians(),
            bgz: 0.5_f32.to_radians(),
            bax: 0.1,
            bay: -0.2,
            baz: 0.3,
            ..NominalState::default()
        };

        let gyro_bias = full_gyro_sensor_bias_dps(&nominal);
        let accel_bias = full_accel_sensor_bias_mps2(&nominal);

        assert!((gyro_bias[0] + 1.0).abs() < 1.0e-6);
        assert!((gyro_bias[1] - 2.0).abs() < 1.0e-6);
        assert!((gyro_bias[2] + 0.5).abs() < 1.0e-6);
        assert!((accel_bias[0] + 0.1).abs() < 1.0e-6);
        assert!((accel_bias[1] - 0.2).abs() < 1.0e-6);
        assert!((accel_bias[2] + 0.3).abs() < 1.0e-6);
    }

    #[test]
    fn reference_mount_rpy_round_trips_to_seed_quaternion() {
        let rpy = [4.5, -2.25, 7.75];
        let q_bv = reference_mount_rpy_to_q_bv(rpy);
        let round_trip = q_bv_to_reference_mount_rpy(q_bv);

        assert!((wrap_deg(round_trip.0 - rpy[0])).abs() < 1.0e-9);
        assert!((wrap_deg(round_trip.1 - rpy[1])).abs() < 1.0e-9);
        assert!((wrap_deg(round_trip.2 - rpy[2])).abs() < 1.0e-9);
    }

    #[test]
    fn reference_mount_seed_comes_from_generic_reference_mount_csv_samples() {
        let replay = GenericReplayInput {
            imu: Vec::new(),
            gnss: Vec::new(),
            reference_attitude: Vec::new(),
            reference_mount: vec![
                GenericReferenceRpySample {
                    t_s: 10.0,
                    roll_deg: 1.0,
                    pitch_deg: -3.0,
                    yaw_deg: 5.0,
                },
                GenericReferenceRpySample {
                    t_s: 20.0,
                    roll_deg: 2.0,
                    pitch_deg: -4.0,
                    yaw_deg: 6.0,
                },
            ],
            reference_position: Vec::new(),
            reference_motion: Vec::new(),
        };

        let seed = reference_mount_seed_q_bv(&replay, VisualizerMountMode::Manual).unwrap();
        let round_trip = q_bv_to_reference_mount_rpy([
            seed[0] as f64,
            seed[1] as f64,
            seed[2] as f64,
            seed[3] as f64,
        ]);

        assert!((wrap_deg(round_trip.0 - 2.0)).abs() < 1.0e-6);
        assert!((wrap_deg(round_trip.1 + 4.0)).abs() < 1.0e-6);
        assert!((wrap_deg(round_trip.2 - 6.0)).abs() < 1.0e-6);
        assert!(reference_mount_seed_q_bv(&replay, VisualizerMountMode::Auto).is_none());
    }

    #[test]
    fn run_context_captures_reference_mount_seed_once() {
        let replay = GenericReplayInput {
            imu: Vec::new(),
            gnss: Vec::new(),
            reference_attitude: Vec::new(),
            reference_mount: vec![GenericReferenceRpySample {
                t_s: 20.0,
                roll_deg: 2.0,
                pitch_deg: -4.0,
                yaw_deg: 6.0,
            }],
            reference_position: Vec::new(),
            reference_motion: Vec::new(),
        };
        let ref_ctx = GenericReplayRunContext::new(
            &replay,
            FilterCompareConfig::default(),
            VisualizerMountMode::Manual,
            GnssOutageConfig::default(),
        );
        let internal_ctx = GenericReplayRunContext::new(
            &replay,
            FilterCompareConfig::default(),
            VisualizerMountMode::Auto,
            GnssOutageConfig::default(),
        );
        assert!(ref_ctx.reference_mount_seed_q_bv().is_some());
        assert!(internal_ctx.reference_mount_seed_q_bv().is_none());
    }

    #[test]
    fn mount_quaternion_error_uses_final_reference_mount() {
        let final_ref = [2.0, -4.0, 6.0];
        let reference_mount = vec![
            GenericReferenceRpySample {
                t_s: 10.0,
                roll_deg: 0.0,
                pitch_deg: 0.0,
                yaw_deg: 0.0,
            },
            GenericReferenceRpySample {
                t_s: 20.0,
                roll_deg: final_ref[0],
                pitch_deg: final_ref[1],
                yaw_deg: final_ref[2],
            },
        ];
        let mut traces = vec![
            Trace {
                name: "Reduced mount roll [deg]".to_string(),
                points: vec![[10.0, final_ref[0]], [20.0, final_ref[0]]],
            },
            Trace {
                name: "Reduced mount pitch [deg]".to_string(),
                points: vec![[10.0, final_ref[1]], [20.0, final_ref[1]]],
            },
            Trace {
                name: "Reduced mount yaw [deg]".to_string(),
                points: vec![[10.0, final_ref[2]], [20.0, final_ref[2]]],
            },
        ];

        push_mount_quaternion_error_trace(
            &mut traces,
            "Reduced",
            "Reduced mount",
            final_reference_mount_rpy(&reference_mount),
        );

        let qerr = trace_by_name(&traces, "Reduced mount quaternion error [deg]").unwrap();
        assert_eq!(qerr.points.len(), 2);
        assert!(qerr.points.iter().all(|sample| sample[1].abs() < 1.0e-9));
    }
}
