use anyhow::{Context, Result, bail};
use sensor_fusion::fusion::SensorFusion;
use sensor_fusion::loose::{
    LOOSE_ERROR_STATES, LooseFilter, LooseImuDelta, LooseNominalState, LoosePredictNoise,
};

use crate::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, GenericReferenceRpySample, fusion_gnss_sample,
    fusion_imu_sample,
};
use crate::eval::gnss_ins::{as_q64, quat_conj, quat_mul};
use crate::eval::replay::{ReplayEvent, for_each_event};
use crate::visualizer::math::{ecef_to_ned, lla_to_ecef, ned_to_lla_exact, quat_rpy_deg};
use crate::visualizer::model::{EkfImuSource, HeadingSample, PlotData, Trace};
use crate::visualizer::pipeline::{EkfCompareConfig, GnssOutageConfig};

pub struct GenericReplayInput {
    pub imu: Vec<GenericImuSample>,
    pub gnss: Vec<GenericGnssSample>,
    pub reference_attitude: Vec<GenericReferenceRpySample>,
    pub reference_mount: Vec<GenericReferenceRpySample>,
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
        }
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
    parse_generic_replay_csvs_with_refs(imu_csv, gnss_csv, None, None)
}

pub fn parse_generic_replay_csvs_with_refs(
    imu_csv: &str,
    gnss_csv: &str,
    reference_attitude_csv: Option<&str>,
    reference_mount_csv: Option<&str>,
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
    })
}

pub fn build_generic_replay_plot_data(
    replay: &GenericReplayInput,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
) -> PlotData {
    build_generic_replay_plot_data_impl(replay, ekf_imu_source, ekf_cfg, gnss_outages, None, None)
}

pub fn build_generic_replay_plot_data_with_eskf_mount_seed(
    replay: &GenericReplayInput,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
    eskf_mount_seed_q_vb: Option<[f32; 4]>,
) -> PlotData {
    build_generic_replay_plot_data_impl(
        replay,
        ekf_imu_source,
        ekf_cfg,
        gnss_outages,
        None,
        eskf_mount_seed_q_vb,
    )
}

pub fn build_generic_replay_plot_data_with_progress(
    replay: &GenericReplayInput,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
    progress: &mut dyn FnMut(GenericReplayProgress),
) -> PlotData {
    build_generic_replay_plot_data_impl(
        replay,
        ekf_imu_source,
        ekf_cfg,
        gnss_outages,
        Some(progress),
        None,
    )
}

pub fn build_generic_replay_plot_data_with_progress_and_eskf_mount_seed(
    replay: &GenericReplayInput,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
    progress: &mut dyn FnMut(GenericReplayProgress),
    eskf_mount_seed_q_vb: Option<[f32; 4]>,
) -> PlotData {
    build_generic_replay_plot_data_impl(
        replay,
        ekf_imu_source,
        ekf_cfg,
        gnss_outages,
        Some(progress),
        eskf_mount_seed_q_vb,
    )
}

fn build_generic_replay_plot_data_impl(
    replay: &GenericReplayInput,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
    progress: Option<&mut dyn FnMut(GenericReplayProgress)>,
    eskf_mount_seed_q_vb: Option<[f32; 4]>,
) -> PlotData {
    let mut progress = GenericProgressReporter::new(replay, progress);
    let mut fusion = SensorFusion::new();
    apply_fusion_config(&mut fusion, ekf_cfg, ekf_imu_source);
    if let Some(seed_q_vb) =
        eskf_mount_seed_q_vb.or_else(|| reference_mount_seed_q_vb(replay, ekf_imu_source))
    {
        fusion.set_misalignment(seed_q_vb);
    }

    let ref_gnss = replay.gnss.first().copied();
    let ref_ecef = ref_gnss.map(|s| lla_to_ecef(s.lat_deg, s.lon_deg, s.height_m));
    let outage_windows = sample_outage_windows(&replay.gnss, gnss_outages);

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
    let mut gnss_map = Vec::new();

    for sample in &replay.gnss {
        gnss_speed.push([
            sample.t_s,
            sample.vel_ned_mps[0].hypot(sample.vel_ned_mps[1]),
        ]);
        gnss_vel_n.push([sample.t_s, sample.vel_ned_mps[0]]);
        gnss_vel_e.push([sample.t_s, sample.vel_ned_mps[1]]);
        gnss_vel_d.push([sample.t_s, sample.vel_ned_mps[2]]);
        gnss_map.push([sample.lon_deg, sample.lat_deg]);
        if let (Some(ref_sample), Some(ref_ecef)) = (ref_gnss, ref_ecef) {
            let ecef = lla_to_ecef(sample.lat_deg, sample.lon_deg, sample.height_m);
            let ned = ecef_to_ned(ecef, ref_ecef, ref_sample.lat_deg, ref_sample.lon_deg);
            gnss_pos_n.push([sample.t_s, ned[0]]);
            gnss_pos_e.push([sample.t_s, ned[1]]);
            gnss_pos_d.push([sample.t_s, ned[2]]);
        }
    }

    let mut eskf_pos_n = Vec::new();
    let mut eskf_pos_e = Vec::new();
    let mut eskf_pos_d = Vec::new();
    let mut eskf_vel_n = Vec::new();
    let mut eskf_vel_e = Vec::new();
    let mut eskf_vel_d = Vec::new();
    let mut eskf_roll = Vec::new();
    let mut eskf_pitch = Vec::new();
    let mut eskf_yaw = Vec::new();
    let mut eskf_mount_roll = Vec::new();
    let mut eskf_mount_pitch = Vec::new();
    let mut eskf_mount_yaw = Vec::new();
    let mut eskf_bgx = Vec::new();
    let mut eskf_bgy = Vec::new();
    let mut eskf_bgz = Vec::new();
    let mut eskf_bax = Vec::new();
    let mut eskf_bay = Vec::new();
    let mut eskf_baz = Vec::new();
    let mut eskf_cov: [Vec<[f64; 2]>; 18] = std::array::from_fn(|_| Vec::new());
    let mut eskf_map = Vec::new();
    let mut eskf_outage_map = Vec::new();
    let mut eskf_heading = Vec::new();
    let mut mount_ready_marker = Vec::new();
    let mut ekf_init_marker = Vec::new();

    for_each_event(&replay.imu, &replay.gnss, |event| match event {
        ReplayEvent::Imu(_, sample) => {
            progress.report_stage(0.0, 0.55, sample.t_s);
            let _ = fusion.process_imu(fusion_imu_sample(*sample));
            raw_gyro_x.push([sample.t_s, sample.gyro_radps[0].to_degrees()]);
            raw_gyro_y.push([sample.t_s, sample.gyro_radps[1].to_degrees()]);
            raw_gyro_z.push([sample.t_s, sample.gyro_radps[2].to_degrees()]);
            raw_accel_x.push([sample.t_s, sample.accel_mps2[0]]);
            raw_accel_y.push([sample.t_s, sample.accel_mps2[1]]);
            raw_accel_z.push([sample.t_s, sample.accel_mps2[2]]);
            append_eskf_sample(
                sample.t_s,
                &fusion,
                ref_gnss,
                ref_ecef,
                &mut eskf_pos_n,
                &mut eskf_pos_e,
                &mut eskf_pos_d,
                &mut eskf_vel_n,
                &mut eskf_vel_e,
                &mut eskf_vel_d,
                &mut eskf_roll,
                &mut eskf_pitch,
                &mut eskf_yaw,
                &mut eskf_mount_roll,
                &mut eskf_mount_pitch,
                &mut eskf_mount_yaw,
                &mut eskf_bgx,
                &mut eskf_bgy,
                &mut eskf_bgz,
                &mut eskf_bax,
                &mut eskf_bay,
                &mut eskf_baz,
                &mut eskf_cov,
                &mut eskf_map,
                &mut eskf_outage_map,
                &mut eskf_heading,
                in_outage(sample.t_s, &outage_windows),
            );
        }
        ReplayEvent::Gnss(_, sample) => {
            progress.report_stage(0.0, 0.55, sample.t_s);
            if !in_outage(sample.t_s, &outage_windows) {
                let update = fusion.process_gnss(scaled_fusion_gnss_sample(*sample, ekf_cfg));
                if update.mount_ready_changed && update.mount_ready {
                    mount_ready_marker.push([sample.t_s, 0.0]);
                }
                if update.ekf_initialized_now && update.ekf_initialized {
                    ekf_init_marker.push([sample.t_s, 0.0]);
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
        eskf_cmp_pos: vec![
            Trace {
                name: "GNSS posN [m]".to_string(),
                points: gnss_pos_n,
            },
            Trace {
                name: "ESKF posN [m]".to_string(),
                points: eskf_pos_n,
            },
            Trace {
                name: "GNSS posE [m]".to_string(),
                points: gnss_pos_e,
            },
            Trace {
                name: "ESKF posE [m]".to_string(),
                points: eskf_pos_e,
            },
            Trace {
                name: "GNSS posD [m]".to_string(),
                points: gnss_pos_d,
            },
            Trace {
                name: "ESKF posD [m]".to_string(),
                points: eskf_pos_d,
            },
        ],
        eskf_cmp_vel: vec![
            Trace {
                name: "GNSS velN [m/s]".to_string(),
                points: gnss_vel_n,
            },
            Trace {
                name: "ESKF velN [m/s]".to_string(),
                points: eskf_vel_n,
            },
            Trace {
                name: "GNSS velE [m/s]".to_string(),
                points: gnss_vel_e,
            },
            Trace {
                name: "ESKF velE [m/s]".to_string(),
                points: eskf_vel_e,
            },
            Trace {
                name: "GNSS velD [m/s]".to_string(),
                points: gnss_vel_d,
            },
            Trace {
                name: "ESKF velD [m/s]".to_string(),
                points: eskf_vel_d,
            },
        ],
        eskf_cmp_att: vec![
            Trace {
                name: "ESKF roll [deg]".to_string(),
                points: eskf_roll,
            },
            Trace {
                name: "ESKF pitch [deg]".to_string(),
                points: eskf_pitch,
            },
            Trace {
                name: "ESKF yaw [deg]".to_string(),
                points: eskf_yaw,
            },
            Trace {
                name: "mount ready".to_string(),
                points: mount_ready_marker,
            },
            Trace {
                name: "ekf initialized".to_string(),
                points: ekf_init_marker,
            },
        ],
        eskf_bias_gyro: vec![
            Trace {
                name: "ESKF gyro bias X [deg/s]".to_string(),
                points: eskf_bgx,
            },
            Trace {
                name: "ESKF gyro bias Y [deg/s]".to_string(),
                points: eskf_bgy,
            },
            Trace {
                name: "ESKF gyro bias Z [deg/s]".to_string(),
                points: eskf_bgz,
            },
        ],
        eskf_bias_accel: vec![
            Trace {
                name: "ESKF accel bias X [m/s^2]".to_string(),
                points: eskf_bax,
            },
            Trace {
                name: "ESKF accel bias Y [m/s^2]".to_string(),
                points: eskf_bay,
            },
            Trace {
                name: "ESKF accel bias Z [m/s^2]".to_string(),
                points: eskf_baz,
            },
        ],
        eskf_meas_gyro: vec![
            Trace {
                name: "ESKF raw IMU gyro X [deg/s]".to_string(),
                points: raw_gyro_x.clone(),
            },
            Trace {
                name: "ESKF raw IMU gyro Y [deg/s]".to_string(),
                points: raw_gyro_y.clone(),
            },
            Trace {
                name: "ESKF raw IMU gyro Z [deg/s]".to_string(),
                points: raw_gyro_z.clone(),
            },
        ],
        eskf_meas_accel: vec![
            Trace {
                name: "ESKF raw IMU accel X [m/s^2]".to_string(),
                points: raw_accel_x.clone(),
            },
            Trace {
                name: "ESKF raw IMU accel Y [m/s^2]".to_string(),
                points: raw_accel_y.clone(),
            },
            Trace {
                name: "ESKF raw IMU accel Z [m/s^2]".to_string(),
                points: raw_accel_z.clone(),
            },
        ],
        eskf_cov_bias: vec![
            Trace {
                name: "accel bias sigma X [m/s^2]".to_string(),
                points: eskf_cov[12].clone(),
            },
            Trace {
                name: "accel bias sigma Y [m/s^2]".to_string(),
                points: eskf_cov[13].clone(),
            },
            Trace {
                name: "accel bias sigma Z [m/s^2]".to_string(),
                points: eskf_cov[14].clone(),
            },
            Trace {
                name: "gyro bias sigma X [deg/s]".to_string(),
                points: eskf_cov[9].clone(),
            },
            Trace {
                name: "gyro bias sigma Y [deg/s]".to_string(),
                points: eskf_cov[10].clone(),
            },
            Trace {
                name: "gyro bias sigma Z [deg/s]".to_string(),
                points: eskf_cov[11].clone(),
            },
        ],
        eskf_cov_nonbias: (0..9)
            .map(|i| Trace {
                name: format!("state_{i}"),
                points: eskf_cov[i].clone(),
            })
            .collect(),
        eskf_misalignment: vec![
            Trace {
                name: "ESKF mount roll [deg]".to_string(),
                points: eskf_mount_roll,
            },
            Trace {
                name: "ESKF mount pitch [deg]".to_string(),
                points: eskf_mount_pitch,
            },
            Trace {
                name: "ESKF mount yaw [deg]".to_string(),
                points: eskf_mount_yaw,
            },
        ],
        eskf_map: vec![
            Trace {
                name: "GNSS path (lon,lat)".to_string(),
                points: gnss_map,
            },
            Trace {
                name: "ESKF path (lon,lat)".to_string(),
                points: eskf_map,
            },
            Trace {
                name: "ESKF path during GNSS outage (lon,lat)".to_string(),
                points: eskf_outage_map,
            },
        ],
        eskf_map_heading: eskf_heading,
        ..PlotData::default()
    };
    add_auxiliary_generic_traces_impl(
        &mut data,
        replay,
        ekf_cfg,
        ekf_imu_source,
        None,
        None,
        &mut progress,
    );
    progress.complete();
    data
}

pub fn add_auxiliary_generic_traces(
    data: &mut PlotData,
    replay: &GenericReplayInput,
    ekf_cfg: EkfCompareConfig,
    ekf_imu_source: EkfImuSource,
    reference_mount_rpy_deg: Option<[f64; 3]>,
    reference_attitude_rpy: Option<[Vec<[f64; 2]>; 3]>,
) {
    let mut progress = GenericProgressReporter::new(replay, None);
    add_auxiliary_generic_traces_impl(
        data,
        replay,
        ekf_cfg,
        ekf_imu_source,
        reference_mount_rpy_deg,
        reference_attitude_rpy,
        &mut progress,
    );
}

fn add_auxiliary_generic_traces_impl(
    data: &mut PlotData,
    replay: &GenericReplayInput,
    ekf_cfg: EkfCompareConfig,
    ekf_imu_source: EkfImuSource,
    reference_mount_rpy_deg: Option<[f64; 3]>,
    reference_attitude_rpy: Option<[Vec<[f64; 2]>; 3]>,
    progress: &mut GenericProgressReporter<'_>,
) {
    let reference_mount_series = rpy_series_from_samples(&replay.reference_mount);
    let reference_attitude_series =
        reference_attitude_rpy.or_else(|| rpy_series_from_samples(&replay.reference_attitude));
    populate_align_traces(
        data,
        replay,
        ekf_cfg,
        ekf_imu_source,
        reference_mount_rpy_deg,
        replay.reference_mount.as_slice(),
        progress,
    );
    populate_loose_traces(data, replay, ekf_cfg, progress);
    populate_eskf_bump_traces(data);
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
        data.eskf_cmp_att.extend(traces.clone());
        data.loose_cmp_att.extend(traces);
    }
    if let Some(reference) = reference_mount_series {
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
        data.eskf_misalignment.extend(traces.clone());
        data.loose_misalignment.extend(traces);
    }
}

fn populate_align_traces(
    data: &mut PlotData,
    replay: &GenericReplayInput,
    ekf_cfg: EkfCompareConfig,
    ekf_imu_source: EkfImuSource,
    reference_mount_rpy_deg: Option<[f64; 3]>,
    reference_mount_series: &[GenericReferenceRpySample],
    progress: &mut GenericProgressReporter<'_>,
) {
    let mut fusion = SensorFusion::new();
    apply_fusion_config(&mut fusion, ekf_cfg, ekf_imu_source);

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
            let (r, p, y) = q_vb_to_reference_mount_rpy([
                align.q_vb[0] as f64,
                align.q_vb[1] as f64,
                align.q_vb[2] as f64,
                align.q_vb[3] as f64,
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
            name: "roll sigma [deg]".to_string(),
            points: cov_roll,
        },
        Trace {
            name: "pitch sigma [deg]".to_string(),
            points: cov_pitch,
        },
        Trace {
            name: "yaw sigma [deg]".to_string(),
            points: cov_yaw,
        },
    ];
}

fn populate_loose_traces(
    data: &mut PlotData,
    replay: &GenericReplayInput,
    ekf_cfg: EkfCompareConfig,
    progress: &mut GenericProgressReporter<'_>,
) {
    let Some(ref_gnss) = replay.gnss.first().copied() else {
        return;
    };
    let ref_ecef = lla_to_ecef(ref_gnss.lat_deg, ref_gnss.lon_deg, ref_gnss.height_m);
    let mut align_fusion = SensorFusion::new();
    apply_fusion_config(&mut align_fusion, ekf_cfg, EkfImuSource::Internal);
    let mut loose = LooseFilter::new(
        ekf_cfg
            .loose_predict_noise
            .unwrap_or_else(LoosePredictNoise::lsm6dso_loose_104hz),
    );
    let mut loose_ready = false;
    let mut last_imu: Option<GenericImuSample> = None;
    let mut latest_gnss: Option<GenericGnssSample> = None;
    let mut last_gnss_used_t_s = f64::NEG_INFINITY;
    let mut seed_mount_q_vb: Option<[f32; 4]> = None;

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
    let mut cov_bias: [Vec<[f64; 2]>; 12] = std::array::from_fn(|_| Vec::new());
    let mut cov_nonbias: [Vec<[f64; 2]>; 12] = std::array::from_fn(|_| Vec::new());
    let mut map = Vec::new();
    let mut headings = Vec::new();

    for_each_event(&replay.imu, &replay.gnss, |event| match event {
        ReplayEvent::Imu(_, sample) => {
            progress.report_stage(0.72, 0.26, sample.t_s);
            let _ = align_fusion.process_imu(fusion_imu_sample(*sample));
            let Some(prev) = last_imu.replace(*sample) else {
                return;
            };
            if !loose_ready {
                return;
            }
            let dt = (sample.t_s - prev.t_s).max(0.0);
            if dt <= 0.0 || dt > 1.0 {
                return;
            }
            let (gyro_vehicle_radps, accel_vehicle_mps2) = vehicle_measurements_from_mount(
                seed_mount_q_vb,
                sample.gyro_radps,
                sample.accel_mps2,
            );
            let (prev_gyro_vehicle_radps, prev_accel_vehicle_mps2) =
                vehicle_measurements_from_mount(seed_mount_q_vb, prev.gyro_radps, prev.accel_mps2);
            let imu = loose_imu_delta_from_vehicle(
                prev_gyro_vehicle_radps,
                prev_accel_vehicle_mps2,
                gyro_vehicle_radps,
                accel_vehicle_mps2,
                dt,
            );
            loose.predict(imu);
            let mut gps_pos = None;
            let mut gps_vel = None;
            let mut gps_pos_std = 0.0f32;
            let mut gps_vel_std = None;
            let mut dt_since_gnss = 1.0f32;
            if let Some(gnss) = latest_gnss
                && (0.0..=0.05).contains(&(sample.t_s - gnss.t_s))
                && gnss.t_s != last_gnss_used_t_s
            {
                gps_pos = Some(lla_to_ecef(gnss.lat_deg, gnss.lon_deg, gnss.height_m));
                gps_vel = Some(
                    ned_vector_to_ecef(gnss.lat_deg, gnss.lon_deg, gnss.vel_ned_mps)
                        .map(|v| v as f32),
                );
                gps_pos_std = ((gnss.pos_std_m[0] + gnss.pos_std_m[1] + gnss.pos_std_m[2]) / 3.0)
                    .max(0.1) as f32;
                gps_vel_std = Some(gnss.vel_std_mps.map(|v| v.max(0.01) as f32));
                dt_since_gnss = if last_gnss_used_t_s.is_finite() {
                    (gnss.t_s - last_gnss_used_t_s).clamp(1.0e-3, 1.0) as f32
                } else {
                    1.0
                };
                last_gnss_used_t_s = gnss.t_s;
            }
            loose.fuse_reference_batch_full(
                gps_pos,
                gps_vel,
                gps_pos_std,
                gps_vel_std,
                dt_since_gnss,
                gyro_vehicle_radps.map(|v| v as f32),
                accel_vehicle_mps2.map(|v| v as f32),
                dt as f32,
            );
            append_loose_sample(
                sample.t_s,
                &loose,
                ref_gnss,
                ref_ecef,
                seed_mount_q_vb,
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
                &mut map,
                &mut headings,
            );
            gyro_x.push([sample.t_s, gyro_vehicle_radps[0].to_degrees()]);
            gyro_y.push([sample.t_s, gyro_vehicle_radps[1].to_degrees()]);
            gyro_z.push([sample.t_s, gyro_vehicle_radps[2].to_degrees()]);
            accel_x.push([sample.t_s, accel_vehicle_mps2[0]]);
            accel_y.push([sample.t_s, accel_vehicle_mps2[1]]);
            accel_z.push([sample.t_s, accel_vehicle_mps2[2]]);
        }
        ReplayEvent::Gnss(_, sample) => {
            progress.report_stage(0.72, 0.26, sample.t_s);
            let _ = align_fusion.process_gnss(fusion_gnss_sample(*sample));
            latest_gnss = Some(*sample);
            if loose_ready || !align_fusion.mount_ready() {
                return;
            }
            let speed = sample.vel_ned_mps[0].hypot(sample.vel_ned_mps[1]);
            if speed < 0.5 {
                return;
            }
            let yaw_rad = sample.vel_ned_mps[1].atan2(sample.vel_ned_mps[0]) as f32;
            let pos_ecef = lla_to_ecef(sample.lat_deg, sample.lon_deg, sample.height_m);
            let vel_ecef = ned_vector_to_ecef(sample.lat_deg, sample.lon_deg, sample.vel_ned_mps)
                .map(|v| v as f32);
            seed_mount_q_vb = align_fusion.mount_q_vb();
            loose.init_seeded_vehicle_from_nav_ecef_state(
                yaw_rad,
                sample.lat_deg,
                sample.lon_deg,
                pos_ecef,
                vel_ecef,
                Some(default_loose_p_diag(*sample, ekf_cfg)),
                None,
            );
            loose_ready = true;
        }
    });

    data.loose_cmp_pos = vec![
        Trace {
            name: "Loose posN [m]".to_string(),
            points: pos_n,
        },
        Trace {
            name: "Loose posE [m]".to_string(),
            points: pos_e,
        },
        Trace {
            name: "Loose posD [m]".to_string(),
            points: pos_d,
        },
    ];
    data.loose_cmp_vel = vec![
        Trace {
            name: "Loose velN [m/s]".to_string(),
            points: vel_n,
        },
        Trace {
            name: "Loose velE [m/s]".to_string(),
            points: vel_e,
        },
        Trace {
            name: "Loose velD [m/s]".to_string(),
            points: vel_d,
        },
    ];
    data.loose_cmp_att = vec![
        Trace {
            name: "Loose roll [deg]".to_string(),
            points: roll,
        },
        Trace {
            name: "Loose pitch [deg]".to_string(),
            points: pitch,
        },
        Trace {
            name: "Loose yaw [deg]".to_string(),
            points: yaw,
        },
    ];
    data.loose_misalignment = vec![
        Trace {
            name: "Loose residual mount roll [deg]".to_string(),
            points: mount_roll,
        },
        Trace {
            name: "Loose residual mount pitch [deg]".to_string(),
            points: mount_pitch,
        },
        Trace {
            name: "Loose residual mount yaw [deg]".to_string(),
            points: mount_yaw,
        },
    ];
    data.loose_meas_gyro = vec![
        Trace {
            name: "Loose gyro x [deg/s]".to_string(),
            points: gyro_x,
        },
        Trace {
            name: "Loose gyro y [deg/s]".to_string(),
            points: gyro_y,
        },
        Trace {
            name: "Loose gyro z [deg/s]".to_string(),
            points: gyro_z,
        },
    ];
    data.loose_meas_accel = vec![
        Trace {
            name: "Loose accel x [m/s^2]".to_string(),
            points: accel_x,
        },
        Trace {
            name: "Loose accel y [m/s^2]".to_string(),
            points: accel_y,
        },
        Trace {
            name: "Loose accel z [m/s^2]".to_string(),
            points: accel_z,
        },
    ];
    data.loose_bias_gyro = vec![
        Trace {
            name: "Loose gyro sensor bias X [deg/s]".to_string(),
            points: bgx,
        },
        Trace {
            name: "Loose gyro sensor bias Y [deg/s]".to_string(),
            points: bgy,
        },
        Trace {
            name: "Loose gyro sensor bias Z [deg/s]".to_string(),
            points: bgz,
        },
    ];
    data.loose_bias_accel = vec![
        Trace {
            name: "Loose accel sensor bias X [m/s^2]".to_string(),
            points: bax,
        },
        Trace {
            name: "Loose accel sensor bias Y [m/s^2]".to_string(),
            points: bay,
        },
        Trace {
            name: "Loose accel sensor bias Z [m/s^2]".to_string(),
            points: accel_bias_z,
        },
    ];
    data.loose_scale_gyro = vec![
        Trace {
            name: "Loose sgx".to_string(),
            points: sgx,
        },
        Trace {
            name: "Loose sgy".to_string(),
            points: sgy,
        },
        Trace {
            name: "Loose sgz".to_string(),
            points: sgz,
        },
    ];
    data.loose_scale_accel = vec![
        Trace {
            name: "Loose sax".to_string(),
            points: sax,
        },
        Trace {
            name: "Loose say".to_string(),
            points: say,
        },
        Trace {
            name: "Loose saz".to_string(),
            points: saz,
        },
    ];
    data.loose_cov_bias = cov_bias
        .into_iter()
        .enumerate()
        .map(|(i, points)| Trace {
            name: format!("Loose sigma bias/scale {i}"),
            points,
        })
        .collect();
    data.loose_cov_nonbias = cov_nonbias
        .into_iter()
        .enumerate()
        .map(|(i, points)| Trace {
            name: format!("Loose sigma state {i}"),
            points,
        })
        .collect();
    data.loose_map = vec![Trace {
        name: "Loose path (lon,lat)".to_string(),
        points: map,
    }];
    data.loose_map_heading = headings;
}

fn populate_eskf_bump_traces(data: &mut PlotData) {
    let pitch = data
        .eskf_cmp_att
        .iter()
        .find(|t| t.name.to_ascii_lowercase().contains("pitch"))
        .map(|t| t.points.clone())
        .unwrap_or_default();
    let speed = data
        .speed
        .first()
        .map(|t| t.points.clone())
        .unwrap_or_default();
    data.eskf_bump_pitch_speed = vec![
        Trace {
            name: "ESKF pitch [deg]".to_string(),
            points: pitch.clone(),
        },
        Trace {
            name: "vehicle speed [m/s]".to_string(),
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
    data.eskf_bump_diag = vec![
        Trace {
            name: "Pitch HPF [deg]".to_string(),
            points: hpf,
        },
        Trace {
            name: "Pitch RMS EMA [deg]".to_string(),
            points: abs_ema,
        },
    ];
}

#[allow(clippy::too_many_arguments)]
fn append_eskf_sample(
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
    map: &mut Vec<[f64; 2]>,
    outage_map: &mut Vec<[f64; 2]>,
    headings: &mut Vec<HeadingSample>,
    outage_active: bool,
) {
    let Some(eskf) = fusion.eskf() else {
        return;
    };
    let display_pos = eskf_display_position_ned(fusion, eskf, ref_gnss, ref_ecef);
    let display_vel = eskf_display_velocity_ned(fusion, eskf, ref_gnss);
    pos_n.push([t_s, display_pos[0]]);
    pos_e.push([t_s, display_pos[1]]);
    pos_d.push([t_s, display_pos[2]]);
    vel_n.push([t_s, display_vel[0]]);
    vel_e.push([t_s, display_vel[1]]);
    vel_d.push([t_s, display_vel[2]]);

    let q_vehicle = eskf_vehicle_attitude_q(eskf);
    let (r, p, y) = quat_rpy_deg(
        q_vehicle[0] as f32,
        q_vehicle[1] as f32,
        q_vehicle[2] as f32,
        q_vehicle[3] as f32,
    );
    roll.push([t_s, r]);
    pitch.push([t_s, p]);
    yaw.push([t_s, y]);

    if let Some(seed_q) = fusion.eskf_mount_q_vb().or_else(|| fusion.mount_q_vb()) {
        let q_cs = [
            eskf.nominal.qcs0 as f64,
            eskf.nominal.qcs1 as f64,
            eskf.nominal.qcs2 as f64,
            eskf.nominal.qcs3 as f64,
        ];
        let q_total_vb = quat_mul(as_q64(seed_q), quat_conj(q_cs));
        let (mr, mp, my) = q_vb_to_reference_mount_rpy(q_total_vb);
        mount_roll.push([t_s, mr]);
        mount_pitch.push([t_s, mp]);
        mount_yaw.push([t_s, my]);
    }

    bgx.push([t_s, (eskf.nominal.bgx as f64).to_degrees()]);
    bgy.push([t_s, (eskf.nominal.bgy as f64).to_degrees()]);
    bgz.push([t_s, (eskf.nominal.bgz as f64).to_degrees()]);
    accel_bias_x.push([t_s, eskf.nominal.bax as f64]);
    accel_bias_y.push([t_s, eskf.nominal.bay as f64]);
    accel_bias_z.push([t_s, eskf.nominal.baz as f64]);
    for (i, trace) in cov.iter_mut().enumerate() {
        trace.push([t_s, eskf.p[i][i].max(0.0).sqrt() as f64]);
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
    }
}

fn eskf_display_position_ned(
    fusion: &SensorFusion,
    eskf: &sensor_fusion::eskf_types::EskfState,
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
        eskf.nominal.pn as f64,
        eskf.nominal.pe as f64,
        eskf.nominal.pd as f64,
    ]
}

fn eskf_display_velocity_ned(
    fusion: &SensorFusion,
    eskf: &sensor_fusion::eskf_types::EskfState,
    ref_gnss: Option<GenericGnssSample>,
) -> [f64; 3] {
    if let (Some(anchor), Some(ref_sample)) = (fusion.anchor_lla_debug(), ref_gnss) {
        let vel_ecef = ned_vector_to_ecef(
            anchor[0] as f64,
            anchor[1] as f64,
            [
                eskf.nominal.vn as f64,
                eskf.nominal.ve as f64,
                eskf.nominal.vd as f64,
            ],
        );
        return ecef_vector_to_ned(ref_sample.lat_deg, ref_sample.lon_deg, vel_ecef);
    }
    [
        eskf.nominal.vn as f64,
        eskf.nominal.ve as f64,
        eskf.nominal.vd as f64,
    ]
}

fn eskf_vehicle_attitude_q(eskf: &sensor_fusion::eskf_types::EskfState) -> [f64; 4] {
    let q_seed_frame = as_q64([
        eskf.nominal.q0,
        eskf.nominal.q1,
        eskf.nominal.q2,
        eskf.nominal.q3,
    ]);
    let q_cs = as_q64([
        eskf.nominal.qcs0,
        eskf.nominal.qcs1,
        eskf.nominal.qcs2,
        eskf.nominal.qcs3,
    ]);
    quat_mul(q_seed_frame, quat_conj(q_cs))
}

fn loose_imu_delta_from_vehicle(
    prev_gyro_radps: [f64; 3],
    prev_accel_mps2: [f64; 3],
    curr_gyro_radps: [f64; 3],
    curr_accel_mps2: [f64; 3],
    dt: f64,
) -> LooseImuDelta {
    LooseImuDelta {
        dax_1: (prev_gyro_radps[0] * dt) as f32,
        day_1: (prev_gyro_radps[1] * dt) as f32,
        daz_1: (prev_gyro_radps[2] * dt) as f32,
        dvx_1: (prev_accel_mps2[0] * dt) as f32,
        dvy_1: (prev_accel_mps2[1] * dt) as f32,
        dvz_1: (prev_accel_mps2[2] * dt) as f32,
        dax_2: (curr_gyro_radps[0] * dt) as f32,
        day_2: (curr_gyro_radps[1] * dt) as f32,
        daz_2: (curr_gyro_radps[2] * dt) as f32,
        dvx_2: (curr_accel_mps2[0] * dt) as f32,
        dvy_2: (curr_accel_mps2[1] * dt) as f32,
        dvz_2: (curr_accel_mps2[2] * dt) as f32,
        dt: dt as f32,
    }
}

#[allow(clippy::too_many_arguments)]
fn append_loose_sample(
    t_s: f64,
    loose: &LooseFilter,
    ref_gnss: GenericGnssSample,
    ref_ecef: [f64; 3],
    seed_mount_q_vb: Option<[f32; 4]>,
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
    map: &mut Vec<[f64; 2]>,
    headings: &mut Vec<HeadingSample>,
) {
    let n = loose.nominal();
    let pos_ecef = loose.shadow_pos_ecef();
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
    let q_es = [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64];
    let q_cs = [n.qcs0 as f64, n.qcs1 as f64, n.qcs2 as f64, n.qcs3 as f64];
    let q_ns = quat_mul(q_ne, q_es);
    let q_vehicle = quat_mul(q_ns, quat_conj(q_cs));
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

    let q_seed = seed_mount_q_vb.map(as_q64).unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let q_total_vb = quat_mul(q_seed, quat_conj(q_cs));
    let (mr, mp, my) = q_vb_to_reference_mount_rpy(q_total_vb);
    mount_roll.push([t_s, mr]);
    mount_pitch.push([t_s, mp]);
    mount_yaw.push([t_s, my]);

    let gyro_sensor_bias_dps = loose_gyro_sensor_bias_dps(n);
    let accel_sensor_bias_mps2 = loose_accel_sensor_bias_mps2(n);
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
    let pmat = loose.covariance();
    for (dst, idx) in cov_bias
        .iter_mut()
        .zip([12usize, 13, 14, 9, 10, 11, 15, 16, 17, 18, 19, 20])
    {
        dst.push([t_s, pmat[idx][idx].max(0.0).sqrt() as f64]);
    }
    for (idx, dst) in cov_nonbias.iter_mut().enumerate() {
        dst.push([t_s, pmat[idx][idx].max(0.0).sqrt() as f64]);
    }
}

fn loose_gyro_sensor_bias_dps(n: &LooseNominalState) -> [f64; 3] {
    [
        -(n.bgx as f64).to_degrees(),
        -(n.bgy as f64).to_degrees(),
        -(n.bgz as f64).to_degrees(),
    ]
}

fn loose_accel_sensor_bias_mps2(n: &LooseNominalState) -> [f64; 3] {
    [-(n.bax as f64), -(n.bay as f64), -(n.baz as f64)]
}

fn default_loose_p_diag(
    gnss: GenericGnssSample,
    cfg: EkfCompareConfig,
) -> [f32; LOOSE_ERROR_STATES] {
    const MIN_LOOSE_MOUNT_YAW_SIGMA_DEG: f32 = 12.0;

    let mut p = [1.0_f32; LOOSE_ERROR_STATES];
    let init = cfg.loose_init;

    let pos_n_sigma = (gnss.pos_std_m[0] as f32).max(init.pos_min_sigma_m);
    let pos_e_sigma = (gnss.pos_std_m[1] as f32).max(init.pos_min_sigma_m);
    let pos_d_sigma = (gnss.pos_std_m[2] as f32).max(init.pos_min_sigma_m);
    p[0] = pos_n_sigma * pos_n_sigma;
    p[1] = pos_e_sigma * pos_e_sigma;
    p[2] = pos_d_sigma * pos_d_sigma;

    let vel_sigma = gnss
        .vel_std_mps
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(init.vel_min_sigma_mps as f64) as f32;
    let vel_var = vel_sigma * vel_sigma;
    p[3] = vel_var;
    p[4] = vel_var;
    p[5] = vel_var;

    let attitude_var = init.attitude_sigma_deg.to_radians().powi(2);
    p[6] = attitude_var;
    p[7] = attitude_var;
    p[8] = attitude_var;

    let gyro_bias_sigma = init.gyro_bias_sigma_dps.to_radians();
    p[9] = init.accel_bias_sigma_mps2 * init.accel_bias_sigma_mps2;
    p[10] = init.accel_bias_sigma_mps2 * init.accel_bias_sigma_mps2;
    p[11] = init.accel_bias_sigma_mps2 * init.accel_bias_sigma_mps2;
    p[12] = gyro_bias_sigma * gyro_bias_sigma;
    p[13] = gyro_bias_sigma * gyro_bias_sigma;
    p[14] = gyro_bias_sigma * gyro_bias_sigma;

    p[15] = init.accel_scale_sigma * init.accel_scale_sigma;
    p[16] = init.accel_scale_sigma * init.accel_scale_sigma;
    p[17] = init.accel_scale_sigma * init.accel_scale_sigma;
    p[18] = init.gyro_scale_sigma * init.gyro_scale_sigma;
    p[19] = init.gyro_scale_sigma * init.gyro_scale_sigma;
    p[20] = init.gyro_scale_sigma * init.gyro_scale_sigma;

    let mount_var = init.mount_sigma_deg.to_radians().powi(2);
    p[21] = mount_var;
    p[22] = mount_var;
    p[23] = init
        .mount_sigma_deg
        .max(MIN_LOOSE_MOUNT_YAW_SIGMA_DEG)
        .to_radians()
        .powi(2);
    p
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

fn q_vb_to_reference_mount_rpy(q_vb: [f64; 4]) -> (f64, f64, f64) {
    let q_x_180 = [0.0, 1.0, 0.0, 0.0];
    let q_flu = quat_mul(q_x_180, quat_conj(q_vb));
    quat_rpy_alg_deg(q_flu)
}

fn reference_mount_rpy_to_q_vb(rpy_deg: [f64; 3]) -> [f64; 4] {
    let q_x_180 = [0.0, 1.0, 0.0, 0.0];
    let q_flu = crate::eval::gnss_ins::quat_from_rpy_alg_deg(rpy_deg[0], rpy_deg[1], rpy_deg[2]);
    quat_mul(quat_conj(q_flu), q_x_180)
}

fn reference_mount_seed_q_vb(
    replay: &GenericReplayInput,
    ekf_imu_source: EkfImuSource,
) -> Option<[f32; 4]> {
    if !ekf_imu_source.uses_ref_mount() {
        return None;
    }
    replay
        .reference_mount
        .iter()
        .rev()
        .find(|sample| {
            sample.roll_deg.is_finite()
                && sample.pitch_deg.is_finite()
                && sample.yaw_deg.is_finite()
        })
        .map(|sample| {
            let q =
                reference_mount_rpy_to_q_vb([sample.roll_deg, sample.pitch_deg, sample.yaw_deg]);
            [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32]
        })
}

fn quat_rpy_alg_deg(q: [f64; 4]) -> (f64, f64, f64) {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (w, x, y, z) = if n > 1.0e-12 {
        (q[0] / n, q[1] / n, q[2] / n, q[3] / n)
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r10 = 2.0 * (x * y + w * z);
    let r20 = 2.0 * (x * z - w * y);
    let r21 = 2.0 * (y * z + w * x);
    let r22 = 1.0 - 2.0 * (x * x + y * y);
    let pitch = (-r20).clamp(-1.0, 1.0).asin();
    let roll = r21.atan2(r22);
    let yaw = r10.atan2(r00);
    (
        roll.to_degrees(),
        pitch.to_degrees(),
        crate::visualizer::math::normalize_heading_deg(yaw.to_degrees()),
    )
}

fn vehicle_measurements_from_mount(
    q_vb: Option<[f32; 4]>,
    raw_gyro_radps: [f64; 3],
    raw_accel_mps2: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let Some(q_vb) = q_vb else {
        return (raw_gyro_radps, raw_accel_mps2);
    };
    let c_bv = transpose3(quat_to_rotmat_f64(as_q64(q_vb)));
    (
        mat_vec3(c_bv, raw_gyro_radps),
        mat_vec3(c_bv, raw_accel_mps2),
    )
}

fn quat_to_rotmat_f64(q: [f64; 4]) -> [[f64; 3]; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (w, x, y, z) = if n > 1.0e-12 {
        (q[0] / n, q[1] / n, q[2] / n, q[3] / n)
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn transpose3(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn mat_vec3(r: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
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

fn apply_fusion_config(fusion: &mut SensorFusion, cfg: EkfCompareConfig, mode: EkfImuSource) {
    fusion.set_align_config(cfg.align);
    if let Some(noise) = cfg.predict_noise {
        fusion.set_predict_noise(noise);
    }
    fusion.set_r_body_vel(cfg.r_body_vel);
    fusion.set_gnss_pos_mount_scale(cfg.gnss_pos_mount_scale);
    fusion.set_gnss_vel_mount_scale(cfg.gnss_vel_mount_scale);
    fusion.set_yaw_init_sigma_rad(cfg.yaw_init_sigma_deg.to_radians());
    fusion.set_gyro_bias_init_sigma_radps(cfg.gyro_bias_init_sigma_dps.to_radians());
    fusion.set_accel_bias_init_sigma_mps2(cfg.accel_bias_init_sigma_mps2);
    fusion.set_mount_init_sigma_rad(cfg.mount_init_sigma_deg.to_radians());
    fusion.set_r_vehicle_speed(cfg.r_vehicle_speed);
    fusion.set_r_zero_vel(cfg.r_zero_vel);
    fusion.set_r_stationary_accel(cfg.r_stationary_accel);
    fusion.set_mount_align_rw_var(cfg.mount_align_rw_var);
    fusion.set_mount_update_min_scale(cfg.mount_update_min_scale);
    fusion.set_mount_update_ramp_time_s(cfg.mount_update_ramp_time_s);
    fusion.set_mount_update_innovation_gate_mps(cfg.mount_update_innovation_gate_mps);
    fusion.set_align_handoff_delay_s(cfg.align_handoff_delay_s);
    fusion.set_freeze_misalignment_states(cfg.freeze_misalignment_states);
    fusion.set_eskf_mount_source(mode.eskf_mount_source());
    fusion.set_mount_settle_time_s(cfg.mount_settle_time_s);
    fusion.set_mount_settle_release_sigma_rad(cfg.mount_settle_release_sigma_deg.to_radians());
    fusion.set_mount_settle_zero_cross_covariance(cfg.mount_settle_zero_cross_covariance);
}

fn scaled_fusion_gnss_sample(
    sample: GenericGnssSample,
    cfg: EkfCompareConfig,
) -> sensor_fusion::fusion::FusionGnssSample {
    let mut sample = fusion_gnss_sample(sample);
    let pos_std_scale = sanitized_r_scale(cfg.gnss_pos_r_scale).sqrt() as f32;
    let vel_std_scale = sanitized_r_scale(cfg.gnss_vel_r_scale).sqrt() as f32;
    for std in &mut sample.pos_std_m {
        *std *= pos_std_scale;
    }
    for std in &mut sample.vel_std_mps {
        *std *= vel_std_scale;
    }
    sample
}

fn sanitized_r_scale(scale: f64) -> f64 {
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
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

fn rpy_series_from_samples(samples: &[GenericReferenceRpySample]) -> Option<[Vec<[f64; 2]>; 3]> {
    if samples.is_empty() {
        return None;
    }
    let mut roll = Vec::with_capacity(samples.len());
    let mut pitch = Vec::with_capacity(samples.len());
    let mut yaw = Vec::with_capacity(samples.len());
    for sample in samples {
        roll.push([sample.t_s, sample.roll_deg]);
        pitch.push([sample.t_s, sample.pitch_deg]);
        yaw.push([sample.t_s, sample.yaw_deg]);
    }
    Some([roll, pitch, yaw])
}

fn reference_rpy_at(samples: &[GenericReferenceRpySample], t_s: f64) -> Option<[f64; 3]> {
    if samples.is_empty() {
        return None;
    }
    let idx = samples.partition_point(|sample| sample.t_s < t_s);
    let nearest = match (idx.checked_sub(1), samples.get(idx)) {
        (Some(prev_idx), Some(next)) => {
            let prev = samples[prev_idx];
            if (t_s - prev.t_s).abs() <= (next.t_s - t_s).abs() {
                prev
            } else {
                *next
            }
        }
        (Some(prev_idx), None) => samples[prev_idx],
        (None, Some(next)) => *next,
        (None, None) => return None,
    };
    Some([nearest.roll_deg, nearest.pitch_deg, nearest.yaw_deg])
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
    fn loose_bias_display_converts_correction_state_to_sensor_bias() {
        let nominal = LooseNominalState {
            bgx: 1.0_f32.to_radians(),
            bgy: -2.0_f32.to_radians(),
            bgz: 0.5_f32.to_radians(),
            bax: 0.1,
            bay: -0.2,
            baz: 0.3,
            ..LooseNominalState::default()
        };

        let gyro_bias = loose_gyro_sensor_bias_dps(&nominal);
        let accel_bias = loose_accel_sensor_bias_mps2(&nominal);

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
        let q_vb = reference_mount_rpy_to_q_vb(rpy);
        let round_trip = q_vb_to_reference_mount_rpy(q_vb);

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
        };

        let seed = reference_mount_seed_q_vb(&replay, EkfImuSource::Ref).unwrap();
        let round_trip = q_vb_to_reference_mount_rpy([
            seed[0] as f64,
            seed[1] as f64,
            seed[2] as f64,
            seed[3] as f64,
        ]);

        assert!((wrap_deg(round_trip.0 - 2.0)).abs() < 1.0e-6);
        assert!((wrap_deg(round_trip.1 + 4.0)).abs() < 1.0e-6);
        assert!((wrap_deg(round_trip.2 - 6.0)).abs() < 1.0e-6);
        assert!(reference_mount_seed_q_vb(&replay, EkfImuSource::Internal).is_none());
        assert!(reference_mount_seed_q_vb(&replay, EkfImuSource::External).is_none());
    }

    #[test]
    fn scaled_fusion_gnss_sample_applies_variance_scales_to_standard_deviations() {
        let cfg = EkfCompareConfig {
            gnss_pos_r_scale: 0.25,
            gnss_vel_r_scale: 4.0,
            ..Default::default()
        };
        let sample = GenericGnssSample {
            t_s: 1.0,
            lat_deg: 37.0,
            lon_deg: -122.0,
            height_m: 10.0,
            vel_ned_mps: [1.0, 2.0, 3.0],
            pos_std_m: [2.0, 4.0, 6.0],
            vel_std_mps: [0.1, 0.2, 0.3],
            heading_rad: None,
        };

        let scaled = scaled_fusion_gnss_sample(sample, cfg);

        assert_eq!(scaled.pos_std_m, [1.0, 2.0, 3.0]);
        assert_eq!(scaled.vel_std_mps, [0.2, 0.4, 0.6]);
    }
}
