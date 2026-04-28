use anyhow::Result;
use sensor_fusion::fusion::SensorFusion;

use crate::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, fusion_gnss_sample, fusion_imu_sample,
};
use crate::eval::gnss_ins::{
    as_q64, quat_angle_deg, quat_conj, quat_from_rpy_alg_deg, quat_mul, quat_rotate,
};
use crate::eval::replay::{ReplayEvent, for_each_event};
use crate::synthetic::gnss_ins_path::{
    GpsNoiseModel, ImuAccuracy, MeasurementNoiseConfig, MotionProfile, PathGenConfig,
    generate_with_noise,
};
use crate::visualizer::math::{ecef_to_ned, lla_to_ecef, quat_rpy_deg};
use crate::visualizer::model::{EkfImuSource, HeadingSample, PlotData, Trace};
use crate::visualizer::pipeline::generic::{GenericReplayInput, add_auxiliary_generic_traces};
use crate::visualizer::pipeline::{EkfCompareConfig, GnssOutageConfig};

#[derive(Clone, Copy, Debug)]
pub enum SyntheticNoiseMode {
    Truth,
    Low,
    Mid,
    High,
}

#[derive(Clone, Debug)]
pub struct SyntheticVisualizerConfig {
    pub motion_def: Option<std::path::PathBuf>,
    pub motion_label: String,
    pub motion_text: Option<String>,
    pub noise_mode: SyntheticNoiseMode,
    pub seed: u64,
    pub mount_rpy_deg: [f64; 3],
    pub imu_hz: f64,
    pub gnss_hz: f64,
    pub gnss_time_shift_ms: f64,
    pub early_vel_bias_ned_mps: [f64; 3],
    pub early_fault_window_s: Option<(f64, f64)>,
}

pub fn build_synthetic_plot_data(
    synth_cfg: &SyntheticVisualizerConfig,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
) -> Result<PlotData> {
    let profile = match (&synth_cfg.motion_text, &synth_cfg.motion_def) {
        (Some(text), _) if synth_cfg.motion_label.ends_with(".csv") => {
            MotionProfile::from_csv_str(text)?
        }
        (Some(text), _) => MotionProfile::from_dsl_str(text)?,
        (None, Some(path)) => MotionProfile::from_path(path)?,
        (None, None) => anyhow::bail!("synthetic scenario has no path or inline text"),
    };
    let path_cfg = PathGenConfig {
        imu_hz: synth_cfg.imu_hz,
        gnss_hz: synth_cfg.gnss_hz,
        ..PathGenConfig::default()
    };
    let noise = match synth_cfg.noise_mode {
        SyntheticNoiseMode::Truth => MeasurementNoiseConfig::zero(),
        SyntheticNoiseMode::Low => MeasurementNoiseConfig::accuracy(ImuAccuracy::Low),
        SyntheticNoiseMode::Mid => MeasurementNoiseConfig::accuracy(ImuAccuracy::Mid),
        SyntheticNoiseMode::High => MeasurementNoiseConfig::accuracy(ImuAccuracy::High),
    };
    let measured = generate_with_noise(&profile, path_cfg, noise, synth_cfg.seed)?;
    let q_truth_mount = quat_from_rpy_alg_deg(
        synth_cfg.mount_rpy_deg[0],
        synth_cfg.mount_rpy_deg[1],
        synth_cfg.mount_rpy_deg[2],
    );
    let gps_noise = noise.gps.unwrap_or(GpsNoiseModel {
        pos_std_m: [0.5, 0.5, 0.5],
        vel_std_mps: [0.2, 0.2, 0.2],
    });
    let imu = measured
        .imu
        .iter()
        .map(|s| GenericImuSample {
            t_s: s.t_s,
            gyro_radps: quat_rotate(q_truth_mount, s.gyro_vehicle_radps),
            accel_mps2: quat_rotate(q_truth_mount, s.accel_vehicle_mps2),
        })
        .collect::<Vec<_>>();
    let gnss = measured
        .gnss
        .iter()
        .filter_map(|s| {
            let t_s = s.t_s + synth_cfg.gnss_time_shift_ms * 1.0e-3;
            if !t_s.is_finite() || t_s < 0.0 {
                return None;
            }
            let mut vel_ned_mps = s.vel_ned_mps;
            if let Some((start_s, end_s)) = synth_cfg.early_fault_window_s
                && (start_s..=end_s).contains(&t_s)
            {
                for (v, bias) in vel_ned_mps.iter_mut().zip(synth_cfg.early_vel_bias_ned_mps) {
                    *v += bias;
                }
            }
            Some(GenericGnssSample {
                t_s,
                lat_deg: s.lat_deg,
                lon_deg: s.lon_deg,
                height_m: s.height_m,
                vel_ned_mps,
                pos_std_m: gps_noise.pos_std_m,
                vel_std_mps: gps_noise.vel_std_mps,
                heading_rad: None,
            })
        })
        .collect::<Vec<_>>();

    let ref_truth = &measured.reference.truth;
    let ref_ecef = lla_to_ecef(
        ref_truth[0].lat_deg,
        ref_truth[0].lon_deg,
        ref_truth[0].height_m,
    );
    let mut truth_pos_n = Vec::new();
    let mut truth_pos_e = Vec::new();
    let mut truth_pos_d = Vec::new();
    let mut truth_vel_n = Vec::new();
    let mut truth_vel_e = Vec::new();
    let mut truth_vel_d = Vec::new();
    let mut truth_roll = Vec::new();
    let mut truth_pitch = Vec::new();
    let mut truth_yaw = Vec::new();
    let mut truth_map = Vec::new();
    let mut truth_speed = Vec::new();
    for truth in ref_truth {
        let ecef = lla_to_ecef(truth.lat_deg, truth.lon_deg, truth.height_m);
        let ned = ecef_to_ned(ecef, ref_ecef, ref_truth[0].lat_deg, ref_truth[0].lon_deg);
        let (roll, pitch, yaw) = quat_rpy_deg(
            truth.q_bn[0] as f32,
            truth.q_bn[1] as f32,
            truth.q_bn[2] as f32,
            truth.q_bn[3] as f32,
        );
        truth_pos_n.push([truth.t_s, ned[0]]);
        truth_pos_e.push([truth.t_s, ned[1]]);
        truth_pos_d.push([truth.t_s, ned[2]]);
        truth_vel_n.push([truth.t_s, truth.vel_ned_mps[0]]);
        truth_vel_e.push([truth.t_s, truth.vel_ned_mps[1]]);
        truth_vel_d.push([truth.t_s, truth.vel_ned_mps[2]]);
        truth_roll.push([truth.t_s, roll]);
        truth_pitch.push([truth.t_s, pitch]);
        truth_yaw.push([truth.t_s, yaw]);
        truth_map.push([truth.lon_deg, truth.lat_deg]);
        truth_speed.push([truth.t_s, truth.vel_ned_mps[0].hypot(truth.vel_ned_mps[1])]);
    }

    let mut gnss_map = Vec::new();
    let mut gnss_speed = Vec::new();
    for sample in &gnss {
        gnss_map.push([sample.lon_deg, sample.lat_deg]);
        gnss_speed.push([
            sample.t_s,
            sample.vel_ned_mps[0].hypot(sample.vel_ned_mps[1]),
        ]);
    }

    let outage_windows = sample_outage_windows(&gnss, gnss_outages);
    let mut fusion = match ekf_imu_source {
        EkfImuSource::Internal | EkfImuSource::External => SensorFusion::new(),
        EkfImuSource::Ref => SensorFusion::with_misalignment([
            q_truth_mount[0] as f32,
            q_truth_mount[1] as f32,
            q_truth_mount[2] as f32,
            q_truth_mount[3] as f32,
        ]),
    };
    apply_fusion_config(&mut fusion, ekf_cfg, ekf_imu_source);

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
    let mut eskf_mount_err = Vec::new();
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
    let mut raw_gyro_x = Vec::new();
    let mut raw_gyro_y = Vec::new();
    let mut raw_gyro_z = Vec::new();
    let mut raw_accel_x = Vec::new();
    let mut raw_accel_y = Vec::new();
    let mut raw_accel_z = Vec::new();

    for_each_event(&imu, &gnss, |event| match event {
        ReplayEvent::Imu(_, sample) => {
            let _ = fusion.process_imu(fusion_imu_sample(*sample));
            raw_gyro_x.push([sample.t_s, sample.gyro_radps[0].to_degrees()]);
            raw_gyro_y.push([sample.t_s, sample.gyro_radps[1].to_degrees()]);
            raw_gyro_z.push([sample.t_s, sample.gyro_radps[2].to_degrees()]);
            raw_accel_x.push([sample.t_s, sample.accel_mps2[0]]);
            raw_accel_y.push([sample.t_s, sample.accel_mps2[1]]);
            raw_accel_z.push([sample.t_s, sample.accel_mps2[2]]);
            if let Some(eskf) = fusion.eskf() {
                append_eskf_synthetic_sample(
                    sample.t_s,
                    eskf,
                    fusion
                        .eskf_mount_q_vb()
                        .or_else(|| fusion.mount_q_vb())
                        .map(as_q64)
                        .unwrap_or(q_truth_mount),
                    q_truth_mount,
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
                    &mut eskf_mount_err,
                    &mut eskf_bgx,
                    &mut eskf_bgy,
                    &mut eskf_bgz,
                    &mut eskf_bax,
                    &mut eskf_bay,
                    &mut eskf_baz,
                    &mut eskf_cov,
                );
                if let Some([lat, lon, _h]) = fusion.position_lla() {
                    eskf_map.push([lon as f64, lat as f64]);
                    let q_vehicle = eskf_vehicle_attitude_q(eskf);
                    let (_, _, yaw) = quat_rpy_deg(
                        q_vehicle[0] as f32,
                        q_vehicle[1] as f32,
                        q_vehicle[2] as f32,
                        q_vehicle[3] as f32,
                    );
                    eskf_heading.push(HeadingSample {
                        t_s: sample.t_s,
                        lon_deg: lon as f64,
                        lat_deg: lat as f64,
                        yaw_deg: yaw,
                    });
                    if in_outage(sample.t_s, &outage_windows) {
                        eskf_outage_map.push([lon as f64, lat as f64]);
                    }
                }
            }
        }
        ReplayEvent::Gnss(_, sample) => {
            if in_outage(sample.t_s, &outage_windows) {
                return;
            }
            let update = fusion.process_gnss(fusion_gnss_sample(*sample));
            if update.mount_ready_changed && update.mount_ready {
                mount_ready_marker.push([sample.t_s, 0.0]);
            }
            if update.ekf_initialized_now {
                ekf_init_marker.push([sample.t_s, 0.0]);
            }
        }
    });

    let mut data = PlotData {
        speed: vec![
            Trace {
                name: "Synthetic truth horizontal speed [m/s]".to_string(),
                points: truth_speed,
            },
            Trace {
                name: "Synthetic GNSS horizontal speed [m/s]".to_string(),
                points: gnss_speed,
            },
        ],
        imu_raw_gyro: vec![
            Trace {
                name: "Synthetic body gyro x [deg/s]".to_string(),
                points: raw_gyro_x.clone(),
            },
            Trace {
                name: "Synthetic body gyro y [deg/s]".to_string(),
                points: raw_gyro_y.clone(),
            },
            Trace {
                name: "Synthetic body gyro z [deg/s]".to_string(),
                points: raw_gyro_z.clone(),
            },
        ],
        imu_raw_accel: vec![
            Trace {
                name: "Synthetic body accel x [m/s^2]".to_string(),
                points: raw_accel_x.clone(),
            },
            Trace {
                name: "Synthetic body accel y [m/s^2]".to_string(),
                points: raw_accel_y.clone(),
            },
            Trace {
                name: "Synthetic body accel z [m/s^2]".to_string(),
                points: raw_accel_z.clone(),
            },
        ],
        orientation: vec![
            Trace {
                name: "Synthetic truth roll [deg]".to_string(),
                points: truth_roll.clone(),
            },
            Trace {
                name: "Synthetic truth pitch [deg]".to_string(),
                points: truth_pitch.clone(),
            },
            Trace {
                name: "Synthetic truth yaw [deg]".to_string(),
                points: truth_yaw.clone(),
            },
        ],
        eskf_cmp_pos: vec![
            Trace {
                name: "ESKF posN [m]".to_string(),
                points: eskf_pos_n,
            },
            Trace {
                name: "Synthetic truth posN [m]".to_string(),
                points: truth_pos_n,
            },
            Trace {
                name: "ESKF posE [m]".to_string(),
                points: eskf_pos_e,
            },
            Trace {
                name: "Synthetic truth posE [m]".to_string(),
                points: truth_pos_e,
            },
            Trace {
                name: "ESKF posD [m]".to_string(),
                points: eskf_pos_d,
            },
            Trace {
                name: "Synthetic truth posD [m]".to_string(),
                points: truth_pos_d,
            },
        ],
        eskf_cmp_vel: vec![
            Trace {
                name: "ESKF vN [m/s]".to_string(),
                points: eskf_vel_n,
            },
            Trace {
                name: "Synthetic truth vN [m/s]".to_string(),
                points: truth_vel_n,
            },
            Trace {
                name: "ESKF vE [m/s]".to_string(),
                points: eskf_vel_e,
            },
            Trace {
                name: "Synthetic truth vE [m/s]".to_string(),
                points: truth_vel_e,
            },
            Trace {
                name: "ESKF vD [m/s]".to_string(),
                points: eskf_vel_d,
            },
            Trace {
                name: "Synthetic truth vD [m/s]".to_string(),
                points: truth_vel_d,
            },
        ],
        eskf_cmp_att: vec![
            Trace {
                name: "ESKF roll [deg]".to_string(),
                points: eskf_roll,
            },
            Trace {
                name: "Synthetic truth roll [deg]".to_string(),
                points: truth_roll.clone(),
            },
            Trace {
                name: "ESKF pitch [deg]".to_string(),
                points: eskf_pitch,
            },
            Trace {
                name: "Synthetic truth pitch [deg]".to_string(),
                points: truth_pitch.clone(),
            },
            Trace {
                name: "ESKF yaw [deg]".to_string(),
                points: eskf_yaw,
            },
            Trace {
                name: "Synthetic truth yaw [deg]".to_string(),
                points: truth_yaw.clone(),
            },
            Trace {
                name: "mount ready".to_string(),
                points: mount_ready_marker,
            },
            Trace {
                name: "EKF initialized".to_string(),
                points: ekf_init_marker,
            },
        ],
        eskf_meas_gyro: vec![
            Trace {
                name: "ESKF body gyro x [deg/s]".to_string(),
                points: raw_gyro_x,
            },
            Trace {
                name: "ESKF body gyro y [deg/s]".to_string(),
                points: raw_gyro_y,
            },
            Trace {
                name: "ESKF body gyro z [deg/s]".to_string(),
                points: raw_gyro_z,
            },
        ],
        eskf_meas_accel: vec![
            Trace {
                name: "ESKF body accel x [m/s^2]".to_string(),
                points: raw_accel_x,
            },
            Trace {
                name: "ESKF body accel y [m/s^2]".to_string(),
                points: raw_accel_y,
            },
            Trace {
                name: "ESKF body accel z [m/s^2]".to_string(),
                points: raw_accel_z,
            },
        ],
        eskf_bias_gyro: vec![
            Trace {
                name: "ESKF gyro bias x [deg/s]".to_string(),
                points: eskf_bgx,
            },
            Trace {
                name: "ESKF gyro bias y [deg/s]".to_string(),
                points: eskf_bgy,
            },
            Trace {
                name: "ESKF gyro bias z [deg/s]".to_string(),
                points: eskf_bgz,
            },
        ],
        eskf_bias_accel: vec![
            Trace {
                name: "ESKF accel bias x [m/s^2]".to_string(),
                points: eskf_bax,
            },
            Trace {
                name: "ESKF accel bias y [m/s^2]".to_string(),
                points: eskf_bay,
            },
            Trace {
                name: "ESKF accel bias z [m/s^2]".to_string(),
                points: eskf_baz,
            },
        ],
        eskf_cov_bias: vec![
            Trace {
                name: "acc_x".to_string(),
                points: eskf_cov[12].clone(),
            },
            Trace {
                name: "acc_y".to_string(),
                points: eskf_cov[13].clone(),
            },
            Trace {
                name: "acc_z".to_string(),
                points: eskf_cov[14].clone(),
            },
            Trace {
                name: "gyro_x".to_string(),
                points: eskf_cov[9].clone(),
            },
            Trace {
                name: "gyro_y".to_string(),
                points: eskf_cov[10].clone(),
            },
            Trace {
                name: "gyro_z".to_string(),
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
                name: "ESKF full mount roll [deg]".to_string(),
                points: eskf_mount_roll,
            },
            Trace {
                name: "ESKF full mount pitch [deg]".to_string(),
                points: eskf_mount_pitch,
            },
            Trace {
                name: "ESKF full mount yaw [deg]".to_string(),
                points: eskf_mount_yaw,
            },
            Trace {
                name: "ESKF mount quaternion error [deg]".to_string(),
                points: eskf_mount_err,
            },
            Trace {
                name: "Synthetic truth mount roll [deg]".to_string(),
                points: vec![
                    [0.0, synth_cfg.mount_rpy_deg[0]],
                    [
                        ref_truth.last().map(|s| s.t_s).unwrap_or(0.0),
                        synth_cfg.mount_rpy_deg[0],
                    ],
                ],
            },
            Trace {
                name: "Synthetic truth mount pitch [deg]".to_string(),
                points: vec![
                    [0.0, synth_cfg.mount_rpy_deg[1]],
                    [
                        ref_truth.last().map(|s| s.t_s).unwrap_or(0.0),
                        synth_cfg.mount_rpy_deg[1],
                    ],
                ],
            },
            Trace {
                name: "Synthetic truth mount yaw [deg]".to_string(),
                points: vec![
                    [0.0, synth_cfg.mount_rpy_deg[2]],
                    [
                        ref_truth.last().map(|s| s.t_s).unwrap_or(0.0),
                        synth_cfg.mount_rpy_deg[2],
                    ],
                ],
            },
        ],
        eskf_map: vec![
            Trace {
                name: "Synthetic truth path (lon,lat)".to_string(),
                points: truth_map,
            },
            Trace {
                name: "Synthetic GNSS path (lon,lat)".to_string(),
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
    add_auxiliary_generic_traces(
        &mut data,
        &GenericReplayInput::new(imu, gnss),
        ekf_cfg,
        ekf_imu_source,
        Some(synth_cfg.mount_rpy_deg),
        Some([truth_roll, truth_pitch, truth_yaw]),
    );
    Ok(data)
}

fn apply_fusion_config(fusion: &mut SensorFusion, cfg: EkfCompareConfig, mode: EkfImuSource) {
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
    fusion.set_mount_update_yaw_rate_gate_radps(cfg.mount_update_yaw_rate_gate_dps.to_radians());
    fusion.set_align_handoff_delay_s(cfg.align_handoff_delay_s);
    fusion.set_freeze_misalignment_states(cfg.freeze_misalignment_states);
    fusion.set_eskf_mount_source(mode.eskf_mount_source());
    fusion.set_mount_settle_time_s(cfg.mount_settle_time_s);
    fusion.set_mount_settle_release_sigma_rad(cfg.mount_settle_release_sigma_deg.to_radians());
    fusion.set_mount_settle_zero_cross_covariance(cfg.mount_settle_zero_cross_covariance);
}

#[allow(clippy::too_many_arguments)]
fn append_eskf_synthetic_sample(
    t_s: f64,
    eskf: &sensor_fusion::eskf_types::EskfState,
    q_seed: [f64; 4],
    q_truth_mount: [f64; 4],
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
    mount_err: &mut Vec<[f64; 2]>,
    bgx: &mut Vec<[f64; 2]>,
    bgy: &mut Vec<[f64; 2]>,
    bgz: &mut Vec<[f64; 2]>,
    bax: &mut Vec<[f64; 2]>,
    bay: &mut Vec<[f64; 2]>,
    accel_bias_z: &mut Vec<[f64; 2]>,
    cov: &mut [Vec<[f64; 2]>; 18],
) {
    let q_vehicle = eskf_vehicle_attitude_q(eskf);
    pos_n.push([t_s, eskf.nominal.pn as f64]);
    pos_e.push([t_s, eskf.nominal.pe as f64]);
    pos_d.push([t_s, eskf.nominal.pd as f64]);
    vel_n.push([t_s, eskf.nominal.vn as f64]);
    vel_e.push([t_s, eskf.nominal.ve as f64]);
    vel_d.push([t_s, eskf.nominal.vd as f64]);
    let (r, p, y) = quat_rpy_deg(
        q_vehicle[0] as f32,
        q_vehicle[1] as f32,
        q_vehicle[2] as f32,
        q_vehicle[3] as f32,
    );
    roll.push([t_s, r]);
    pitch.push([t_s, p]);
    yaw.push([t_s, y]);
    let q_cs = as_q64([
        eskf.nominal.qcs0,
        eskf.nominal.qcs1,
        eskf.nominal.qcs2,
        eskf.nominal.qcs3,
    ]);
    let q_full_mount = quat_mul(q_seed, quat_conj(q_cs));
    let (mr, mp, my) = quat_rpy_deg(
        q_full_mount[0] as f32,
        q_full_mount[1] as f32,
        q_full_mount[2] as f32,
        q_full_mount[3] as f32,
    );
    mount_roll.push([t_s, mr]);
    mount_pitch.push([t_s, mp]);
    mount_yaw.push([t_s, my]);
    mount_err.push([t_s, quat_angle_deg(q_full_mount, q_truth_mount)]);
    bgx.push([t_s, (eskf.nominal.bgx as f64).to_degrees()]);
    bgy.push([t_s, (eskf.nominal.bgy as f64).to_degrees()]);
    bgz.push([t_s, (eskf.nominal.bgz as f64).to_degrees()]);
    bax.push([t_s, eskf.nominal.bax as f64]);
    bay.push([t_s, eskf.nominal.bay as f64]);
    accel_bias_z.push([t_s, eskf.nominal.baz as f64]);
    for (i, trace) in cov.iter_mut().enumerate() {
        trace.push([t_s, eskf.p[i][i].max(0.0).sqrt() as f64]);
    }
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
