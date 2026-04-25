use std::collections::VecDeque;

use sensor_fusion::align::GRAVITY_MPS2;
use sensor_fusion::ekf::PredictNoise;
use sensor_fusion::eskf_types::EskfState;
use sensor_fusion::fusion::SensorFusion;
use sensor_fusion::loose::{LooseFilter, LooseImuDelta, LoosePredictNoise};

use crate::datasets::generic_replay::{
    fusion_gnss_sample as to_fusion_gnss, fusion_imu_sample as to_fusion_imu,
};
use crate::datasets::ubx_replay::{UbxReplayConfig, build_generic_replay_from_frames};
use crate::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_alg_status, extract_nav_att,
    extract_nav_pvt_obs, extract_nav2_pvt_obs,
};

use super::super::math::{
    deg2rad, ecef_to_ned, lla_to_ecef, mat_vec, ned_to_lla_exact, normalize_heading_deg,
    quat_rpy_deg, rad2deg, rot_zyx,
};
use super::super::model::{AlgEvent, EkfImuSource, HeadingSample, NavAttEvent, Trace};
use super::align_replay::{
    BootstrapConfig as AlignBootstrapConfig, ImuReplayConfig, build_align_replay,
    build_fusion_align_replay, esf_alg_flu_to_frd_mount_quat, frd_mount_quat_to_esf_alg_flu_quat,
    quat_rpy_alg_deg,
};
use super::timebase::MasterTimeline;

pub struct EkfCompareData {
    pub eskf_cmp_pos: Vec<Trace>,
    pub eskf_cmp_vel: Vec<Trace>,
    pub eskf_cmp_att: Vec<Trace>,
    pub eskf_meas_gyro: Vec<Trace>,
    pub eskf_meas_accel: Vec<Trace>,
    pub eskf_bias_gyro: Vec<Trace>,
    pub eskf_bias_accel: Vec<Trace>,
    pub eskf_cov_bias: Vec<Trace>,
    pub eskf_cov_nonbias: Vec<Trace>,
    pub eskf_misalignment: Vec<Trace>,
    pub eskf_stationary_diag: Vec<Trace>,
    pub eskf_bump_pitch_speed: Vec<Trace>,
    pub eskf_bump_diag: Vec<Trace>,
    pub eskf_map: Vec<Trace>,
    pub eskf_map_heading: Vec<HeadingSample>,
    pub loose_cmp_pos: Vec<Trace>,
    pub loose_cmp_vel: Vec<Trace>,
    pub loose_cmp_att: Vec<Trace>,
    pub loose_misalignment: Vec<Trace>,
    pub loose_meas_gyro: Vec<Trace>,
    pub loose_meas_accel: Vec<Trace>,
    pub loose_bias_gyro: Vec<Trace>,
    pub loose_bias_accel: Vec<Trace>,
    pub loose_scale_gyro: Vec<Trace>,
    pub loose_scale_accel: Vec<Trace>,
    pub loose_cov_bias: Vec<Trace>,
    pub loose_cov_nonbias: Vec<Trace>,
    pub loose_map: Vec<Trace>,
    pub loose_map_heading: Vec<HeadingSample>,
}

#[derive(Clone, Copy, Debug)]
pub struct EkfCompareConfig {
    pub r_body_vel: f32,
    pub gnss_pos_mount_scale: f32,
    pub gnss_vel_mount_scale: f32,
    pub gyro_bias_init_sigma_dps: f32,
    pub r_vehicle_speed: f32,
    pub mount_align_rw_var: f32,
    pub mount_update_min_scale: f32,
    pub mount_update_ramp_time_s: f32,
    pub mount_update_innovation_gate_mps: f32,
    pub mount_update_yaw_rate_gate_dps: f32,
    pub freeze_misalignment_states: bool,
    pub r_zero_vel: f32,
    pub r_stationary_accel: f32,
    pub vehicle_meas_lpf_cutoff_hz: f64,
    pub predict_imu_lpf_cutoff_hz: Option<f64>,
    pub predict_imu_decimation: usize,
    pub yaw_init_speed_mps: f64,
    pub gnss_pos_r_scale: f64,
    pub gnss_vel_r_scale: f64,
    pub predict_noise: Option<PredictNoise>,
    pub loose_predict_noise: Option<LoosePredictNoise>,
}

impl Default for EkfCompareConfig {
    fn default() -> Self {
        Self {
            r_body_vel: 0.001,
            gnss_pos_mount_scale: 0.0,
            gnss_vel_mount_scale: 0.0,
            gyro_bias_init_sigma_dps: 0.125,
            r_vehicle_speed: 0.04,
            mount_align_rw_var: 1.0e-7,
            mount_update_min_scale: 0.008,
            mount_update_ramp_time_s: 800.0,
            mount_update_innovation_gate_mps: 0.02,
            mount_update_yaw_rate_gate_dps: 0.0,
            freeze_misalignment_states: false,
            r_zero_vel: 0.0,
            r_stationary_accel: 0.0,
            vehicle_meas_lpf_cutoff_hz: 35.0,
            predict_imu_lpf_cutoff_hz: None,
            predict_imu_decimation: 1,
            yaw_init_speed_mps: 0.0 / 3.6,
            gnss_pos_r_scale: 0.1,
            gnss_vel_r_scale: 1.0,
            predict_noise: None,
            loose_predict_noise: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GnssOutageConfig {
    pub count: usize,
    pub duration_s: f64,
    pub seed: u64,
}

const ESKF_YAW_CUE_NAMES: [&str; 11] = [
    "gps_pos_ne",
    "gps_vel_ne",
    "zero_vel_ne",
    "body_speed_x",
    "body_vel_y",
    "body_vel_z",
    "stationary_x",
    "stationary_y",
    "gps_pos_d",
    "gps_vel_d",
    "zero_vel_d",
];

fn apply_fusion_config(fusion: &mut SensorFusion, cfg: EkfCompareConfig) {
    fusion.set_r_body_vel(cfg.r_body_vel);
    fusion.set_gnss_pos_mount_scale(cfg.gnss_pos_mount_scale);
    fusion.set_gnss_vel_mount_scale(cfg.gnss_vel_mount_scale);
    fusion.set_gyro_bias_init_sigma_radps(cfg.gyro_bias_init_sigma_dps.to_radians());
    fusion.set_r_vehicle_speed(cfg.r_vehicle_speed);
    fusion.set_r_zero_vel(cfg.r_zero_vel);
    fusion.set_r_stationary_accel(cfg.r_stationary_accel);
    fusion.set_mount_align_rw_var(cfg.mount_align_rw_var);
    fusion.set_mount_update_min_scale(cfg.mount_update_min_scale);
    fusion.set_mount_update_ramp_time_s(cfg.mount_update_ramp_time_s);
    fusion.set_mount_update_innovation_gate_mps(cfg.mount_update_innovation_gate_mps);
    fusion.set_mount_update_yaw_rate_gate_radps(cfg.mount_update_yaw_rate_gate_dps.to_radians());
    fusion.set_freeze_misalignment_states(cfg.freeze_misalignment_states);
}

pub fn build_ekf_compare_traces(
    frames: &[UbxFrame],
    tl: &MasterTimeline,
    ekf_imu_source: EkfImuSource,
    cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
) -> EkfCompareData {
    if tl.masters.is_empty() {
        return EkfCompareData {
            eskf_cmp_pos: Vec::new(),
            eskf_cmp_vel: Vec::new(),
            eskf_cmp_att: Vec::new(),
            eskf_meas_gyro: Vec::new(),
            eskf_meas_accel: Vec::new(),
            eskf_bias_gyro: Vec::new(),
            eskf_bias_accel: Vec::new(),
            eskf_cov_bias: Vec::new(),
            eskf_cov_nonbias: Vec::new(),
            eskf_misalignment: Vec::new(),
            eskf_stationary_diag: Vec::new(),
            eskf_bump_pitch_speed: Vec::new(),
            eskf_bump_diag: Vec::new(),
            eskf_map: Vec::new(),
            eskf_map_heading: Vec::new(),
            loose_cmp_pos: Vec::new(),
            loose_cmp_vel: Vec::new(),
            loose_cmp_att: Vec::new(),
            loose_misalignment: Vec::new(),
            loose_meas_gyro: Vec::new(),
            loose_meas_accel: Vec::new(),
            loose_bias_gyro: Vec::new(),
            loose_bias_accel: Vec::new(),
            loose_scale_gyro: Vec::new(),
            loose_scale_accel: Vec::new(),
            loose_cov_bias: Vec::new(),
            loose_cov_nonbias: Vec::new(),
            loose_map: Vec::new(),
            loose_map_heading: Vec::new(),
        };
    }

    let rel_s = |master_ms: f64| (master_ms - tl.t0_master_ms) * 1e-3;

    let mut alg_events = Vec::<AlgEvent>::new();
    let mut alg_status_events = Vec::<(f64, u8)>::new();
    let mut nav_att_events = Vec::<NavAttEvent>::new();
    let mut nav_events_pvt = Vec::<(f64, NavPvtObs)>::new();
    let mut nav_events_nav2 = Vec::<(f64, NavPvtObs)>::new();
    for f in frames {
        if let Some((_, roll, pitch, yaw)) = extract_esf_alg(f)
            && let Some(t_ms) = super::super::math::nearest_master_ms(f.seq, &tl.masters)
        {
            alg_events.push(AlgEvent {
                t_ms,
                roll_deg: roll,
                pitch_deg: pitch,
                yaw_deg: yaw,
            });
        }
        if ekf_imu_source == EkfImuSource::EsfAlg
            && let Some((_, status_code, _is_fine)) = extract_esf_alg_status(f)
            && let Some(t_ms) = super::super::math::nearest_master_ms(f.seq, &tl.masters)
        {
            alg_status_events.push((t_ms, status_code as u8));
        }
        if let Some((_itow, roll, pitch, heading)) = extract_nav_att(f) {
            if let Some(t_ms) = super::super::math::nearest_master_ms(f.seq, &tl.masters) {
                nav_att_events.push(NavAttEvent {
                    t_ms,
                    roll_deg: roll,
                    pitch_deg: pitch,
                    heading_deg: normalize_heading_deg(heading),
                });
            }
        }
        if let Some(t_ms) = super::super::math::nearest_master_ms(f.seq, &tl.masters) {
            if let Some(obs) = extract_nav2_pvt_obs(f) {
                if obs.fix_ok && !obs.invalid_llh {
                    nav_events_nav2.push((t_ms, obs));
                }
            } else if let Some(obs) = extract_nav_pvt_obs(f) {
                if obs.fix_ok && !obs.invalid_llh {
                    nav_events_pvt.push((t_ms, obs));
                }
            }
        }
    }
    alg_events.sort_by(|a, b| {
        a.t_ms
            .partial_cmp(&b.t_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    alg_status_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    nav_att_events.sort_by(|a, b| {
        a.t_ms
            .partial_cmp(&b.t_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    nav_events_nav2.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    nav_events_pvt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let nav_pvt_events_for_map = nav_events_pvt.clone();
    let nav2_events_for_map = nav_events_nav2.clone();
    let final_alg_q = alg_events
        .last()
        .map(|alg| esf_alg_flu_to_frd_mount_quat(alg.roll_deg, alg.pitch_deg, alg.yaw_deg));
    let align_events = match ekf_imu_source {
        EkfImuSource::Align | EkfImuSource::EsfAlg => {
            build_align_mount_events(frames, tl, ImuReplayConfig::default())
        }
    };
    let align_handoff_t_ms = if ekf_imu_source == EkfImuSource::EsfAlg {
        build_fusion_align_replay(frames, tl, EkfImuSource::Align, ImuReplayConfig::default())
            .ekf_initialized_times_s
            .first()
            .copied()
            .map(|t_s| tl.t0_master_ms + t_s * 1000.0)
    } else {
        align_events.first().map(|(t_ms, _)| *t_ms)
    };
    let replay = build_generic_replay_from_frames(
        frames,
        tl,
        UbxReplayConfig {
            gnss_pos_r_scale: cfg.gnss_pos_r_scale,
            gnss_vel_r_scale: cfg.gnss_vel_r_scale,
            ..UbxReplayConfig::default()
        },
    )
    .expect("failed to build generic replay from UBX frames");
    let nav_events = replay.nav_events.clone();
    let gnss_samples = replay.gnss_samples.clone();
    let use_nav2_for_ekf = replay.used_nav2;
    if !use_nav2_for_ekf {
        eprintln!(
            "WARNING: NAV2-PVT not found; falling back to NAV-PVT downsampled to 2 Hz for EKF GNSS observations."
        );
    }
    let outage_windows_ms = sample_gnss_outage_windows(&nav_events, gnss_outages);
    let mut ubx_pos_n = Vec::<[f64; 2]>::new();
    let mut ubx_pos_e = Vec::<[f64; 2]>::new();
    let mut ubx_pos_d = Vec::<[f64; 2]>::new();

    let mut ubx_vel_forward = Vec::<[f64; 2]>::new();
    let mut ubx_vel_lateral = Vec::<[f64; 2]>::new();
    let mut ubx_vel_vertical = Vec::<[f64; 2]>::new();

    let mut ubx_att_roll = Vec::<[f64; 2]>::new();
    let mut ubx_att_pitch = Vec::<[f64; 2]>::new();
    let mut ubx_att_yaw = Vec::<[f64; 2]>::new();
    let mut map_ubx = Vec::<[f64; 2]>::new();
    let mut map_nav2 = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_pos_n = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_pos_e = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_pos_d = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_vel_forward = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_vel_lateral = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_vel_vertical = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_att_roll = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_att_pitch = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_att_yaw = Vec::<[f64; 2]>::new();
    let mut eskf_meas_gyro_x = Vec::<[f64; 2]>::new();
    let mut eskf_meas_gyro_y = Vec::<[f64; 2]>::new();
    let mut eskf_meas_gyro_z = Vec::<[f64; 2]>::new();
    let mut eskf_meas_accel_x = Vec::<[f64; 2]>::new();
    let mut eskf_meas_accel_y = Vec::<[f64; 2]>::new();
    let mut eskf_meas_accel_z = Vec::<[f64; 2]>::new();
    let mut eskf_bias_gyro_x = Vec::<[f64; 2]>::new();
    let mut eskf_bias_gyro_y = Vec::<[f64; 2]>::new();
    let mut eskf_bias_gyro_z = Vec::<[f64; 2]>::new();
    let mut eskf_bias_accel_x = Vec::<[f64; 2]>::new();
    let mut eskf_bias_accel_y = Vec::<[f64; 2]>::new();
    let mut eskf_bias_accel_z = Vec::<[f64; 2]>::new();
    let mut eskf_cov_diag: [Vec<[f64; 2]>; 18] = std::array::from_fn(|_| Vec::new());
    let mut eskf_mount_roll = Vec::<[f64; 2]>::new();
    let mut eskf_mount_pitch = Vec::<[f64; 2]>::new();
    let mut eskf_mount_yaw = Vec::<[f64; 2]>::new();
    let eskf_stationary_innov_x = Vec::<[f64; 2]>::new();
    let eskf_stationary_innov_y = Vec::<[f64; 2]>::new();
    let eskf_stationary_k_theta_x_from_x = Vec::<[f64; 2]>::new();
    let eskf_stationary_k_theta_y_from_x = Vec::<[f64; 2]>::new();
    let eskf_stationary_k_theta_x_from_y = Vec::<[f64; 2]>::new();
    let eskf_stationary_k_theta_y_from_y = Vec::<[f64; 2]>::new();
    let eskf_stationary_k_bax_from_x = Vec::<[f64; 2]>::new();
    let eskf_stationary_k_bay_from_y = Vec::<[f64; 2]>::new();
    let eskf_stationary_p_theta_x = Vec::<[f64; 2]>::new();
    let eskf_stationary_p_theta_y = Vec::<[f64; 2]>::new();
    let eskf_stationary_p_bax = Vec::<[f64; 2]>::new();
    let eskf_stationary_p_bay = Vec::<[f64; 2]>::new();
    let eskf_stationary_p_theta_x_bax = Vec::<[f64; 2]>::new();
    let eskf_stationary_p_theta_y_bay = Vec::<[f64; 2]>::new();
    let mut eskf_yaw_cue_sum: [Vec<[f64; 2]>; 11] = std::array::from_fn(|_| Vec::new());
    let mut eskf_yaw_cue_abs: [Vec<[f64; 2]>; 11] = std::array::from_fn(|_| Vec::new());
    let mut eskf_yaw_cue_innov_sum: [Vec<[f64; 2]>; 11] = std::array::from_fn(|_| Vec::new());
    let mut eskf_yaw_cue_innov_abs: [Vec<[f64; 2]>; 11] = std::array::from_fn(|_| Vec::new());
    let mut loose_cmp_pos_n = Vec::<[f64; 2]>::new();
    let mut loose_cmp_pos_e = Vec::<[f64; 2]>::new();
    let mut loose_cmp_pos_d = Vec::<[f64; 2]>::new();
    let mut loose_cmp_vel_forward = Vec::<[f64; 2]>::new();
    let mut loose_cmp_vel_lateral = Vec::<[f64; 2]>::new();
    let mut loose_cmp_vel_vertical = Vec::<[f64; 2]>::new();
    let mut loose_cmp_att_roll = Vec::<[f64; 2]>::new();
    let mut loose_cmp_att_pitch = Vec::<[f64; 2]>::new();
    let mut loose_cmp_att_yaw = Vec::<[f64; 2]>::new();
    let mut loose_mount_roll = Vec::<[f64; 2]>::new();
    let mut loose_mount_pitch = Vec::<[f64; 2]>::new();
    let mut loose_mount_yaw = Vec::<[f64; 2]>::new();
    let mut loose_meas_gyro_x = Vec::<[f64; 2]>::new();
    let mut loose_meas_gyro_y = Vec::<[f64; 2]>::new();
    let mut loose_meas_gyro_z = Vec::<[f64; 2]>::new();
    let mut loose_meas_accel_x = Vec::<[f64; 2]>::new();
    let mut loose_meas_accel_y = Vec::<[f64; 2]>::new();
    let mut loose_meas_accel_z = Vec::<[f64; 2]>::new();
    let mut loose_bias_gyro_x = Vec::<[f64; 2]>::new();
    let mut loose_bias_gyro_y = Vec::<[f64; 2]>::new();
    let mut loose_bias_gyro_z = Vec::<[f64; 2]>::new();
    let mut loose_bias_accel_x = Vec::<[f64; 2]>::new();
    let mut loose_bias_accel_y = Vec::<[f64; 2]>::new();
    let mut loose_bias_accel_z = Vec::<[f64; 2]>::new();
    let mut loose_scale_gyro_x = Vec::<[f64; 2]>::new();
    let mut loose_scale_gyro_y = Vec::<[f64; 2]>::new();
    let mut loose_scale_gyro_z = Vec::<[f64; 2]>::new();
    let mut loose_scale_accel_x = Vec::<[f64; 2]>::new();
    let mut loose_scale_accel_y = Vec::<[f64; 2]>::new();
    let mut loose_scale_accel_z = Vec::<[f64; 2]>::new();
    let mut loose_cov_diag: [Vec<[f64; 2]>; 24] = std::array::from_fn(|_| Vec::new());
    let mut map_eskf = Vec::<[f64; 2]>::new();
    let mut map_eskf_outage = Vec::<[f64; 2]>::new();
    let mut map_eskf_heading = Vec::<HeadingSample>::new();
    let mut fusion_mount_ready_marker = Vec::<[f64; 2]>::new();
    let mut fusion_ekf_init_marker = Vec::<[f64; 2]>::new();
    let mut map_loose = Vec::<[f64; 2]>::new();
    let mut map_loose_heading = Vec::<HeadingSample>::new();

    let mut fusion = match ekf_imu_source {
        EkfImuSource::Align => Some(SensorFusion::new()),
        EkfImuSource::EsfAlg => None,
    };
    if let Some(fusion_ref) = fusion.as_mut() {
        apply_fusion_config(fusion_ref, cfg);
    }
    let mut loose: Option<LooseFilter> = None;
    let base_loose_predict_noise = cfg
        .loose_predict_noise
        .unwrap_or(LoosePredictNoise::lsm6dso_loose_104hz());
    let mut prev_imu_t: Option<f64> = None;
    let mut alg_idx = 0usize;
    let mut alg_status_idx = 0usize;
    let mut align_idx = 0usize;
    let mut nav_idx = 0usize;
    let mut cur_alg: Option<AlgEvent> = None;
    let mut cur_alg_status: u8 = 0;
    let mut cur_align_q_vb: Option<[f32; 4]> = None;

    let mut origin_set = false;
    let mut loose_seed_mount_q_vb: Option<[f32; 4]> = None;
    let mut ref_lat = 0.0_f64;
    let mut ref_lon = 0.0_f64;
    let mut ref_ecef = [0.0_f64; 3];
    let mut ref_h = 0.0_f64;
    let mut filt_eskf_meas_gyro: Option<[f64; 3]> = None;
    let mut filt_eskf_meas_accel: Option<[f64; 3]> = None;
    let mut filt_predict_gyro: Option<[f64; 3]> = None;
    let mut filt_predict_accel: Option<[f64; 3]> = None;
    let mut predict_gyro_sum = [0.0_f64; 3];
    let mut predict_accel_sum = [0.0_f64; 3];
    let mut predict_dt_accum = 0.0_f64;
    let mut predict_decim_count = 0usize;
    let mut prev_outage_active = false;

    if let Some((_, first_nav)) = nav_events.first().copied() {
        ref_lat = first_nav.lat_deg;
        ref_lon = first_nav.lon_deg;
        ref_h = first_nav.height_m;
        ref_ecef = lla_to_ecef(ref_lat, ref_lon, ref_h);
        origin_set = true;

        let mut nav_att_idx = 0usize;
        let mut current_nav_att: Option<NavAttEvent> = None;
        for (t_ms, nav) in &nav_events {
            while nav_att_idx < nav_att_events.len() && nav_att_events[nav_att_idx].t_ms <= *t_ms {
                current_nav_att = Some(nav_att_events[nav_att_idx]);
                nav_att_idx += 1;
            }
            let ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
            let ned = ecef_to_ned(ecef, ref_ecef, ref_lat, ref_lon);
            let vel_vehicle = ubx_vehicle_velocity(*nav, current_nav_att);
            let t = rel_s(*t_ms);
            ubx_pos_n.push([t, ned[0]]);
            ubx_pos_e.push([t, ned[1]]);
            ubx_pos_d.push([t, ned[2]]);
            ubx_vel_forward.push([t, vel_vehicle[0]]);
            ubx_vel_lateral.push([t, vel_vehicle[1]]);
            ubx_vel_vertical.push([t, vel_vehicle[2]]);
        }
    }
    for (_t_ms, nav_pvt) in &nav_pvt_events_for_map {
        map_ubx.push([nav_pvt.lon_deg, nav_pvt.lat_deg]);
    }
    for att in &nav_att_events {
        let t = rel_s(att.t_ms);
        ubx_att_roll.push([t, att.roll_deg]);
        ubx_att_pitch.push([t, att.pitch_deg]);
        ubx_att_yaw.push([t, att.heading_deg]);
    }
    let esf_alg_mount_ref = vec![
        Trace {
            name: "ESF-ALG mount roll [deg]".to_string(),
            points: alg_events
                .iter()
                .map(|alg| [rel_s(alg.t_ms), alg.roll_deg])
                .collect(),
        },
        Trace {
            name: "ESF-ALG mount pitch [deg]".to_string(),
            points: alg_events
                .iter()
                .map(|alg| [rel_s(alg.t_ms), alg.pitch_deg])
                .collect(),
        },
        Trace {
            name: "ESF-ALG mount yaw [deg]".to_string(),
            points: alg_events
                .iter()
                .map(|alg| [rel_s(alg.t_ms), normalize_heading_deg(alg.yaw_deg)])
                .collect(),
        },
    ];
    for (_t_ms, nav2) in &nav2_events_for_map {
        map_nav2.push([nav2.lon_deg, nav2.lat_deg]);
    }

    let mut loose_last_gps_update_ms: Option<f64> = None;
    for imu_sample in &replay.imu_samples {
        let pkt_t_ms = tl.t0_master_ms + imu_sample.t_s * 1.0e3;
        while alg_idx < alg_events.len() && alg_events[alg_idx].t_ms <= pkt_t_ms {
            cur_alg = Some(alg_events[alg_idx]);
            alg_idx += 1;
        }
        while alg_status_idx < alg_status_events.len()
            && alg_status_events[alg_status_idx].0 <= pkt_t_ms
        {
            cur_alg_status = alg_status_events[alg_status_idx].1;
            alg_status_idx += 1;
        }
        while align_idx < align_events.len() && align_events[align_idx].0 <= pkt_t_ms {
            cur_align_q_vb = Some(align_events[align_idx].1);
            align_idx += 1;
        }
        let dt = match prev_imu_t {
            Some(prev) => imu_sample.t_s - prev,
            None => {
                prev_imu_t = Some(imu_sample.t_s);
                continue;
            }
        };
        prev_imu_t = Some(imu_sample.t_s);
        if !(0.001..=0.05).contains(&dt) {
            continue;
        }
        if ekf_imu_source == EkfImuSource::EsfAlg
            && fusion.is_none()
            && align_handoff_t_ms.is_some_and(|t_handoff_ms| pkt_t_ms >= t_handoff_ms)
            && let Some(q_vb) = final_alg_q
        {
            let mut created = SensorFusion::with_misalignment([
                q_vb[0] as f32,
                q_vb[1] as f32,
                q_vb[2] as f32,
                q_vb[3] as f32,
            ]);
            apply_fusion_config(&mut created, cfg);
            fusion = Some(created);
        }
        let Some(fusion_ref) = fusion.as_mut() else {
            while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt_t_ms {
                nav_idx += 1;
            }
            continue;
        };
        let outage_active = in_gnss_outage(pkt_t_ms, &outage_windows_ms);

        let raw_gyro_radps = [
            imu_sample.gyro_radps[0] as f32,
            imu_sample.gyro_radps[1] as f32,
            imu_sample.gyro_radps[2] as f32,
        ];
        let raw_accel_mps2 = [
            imu_sample.accel_mps2[0] as f32,
            imu_sample.accel_mps2[1] as f32,
            imu_sample.accel_mps2[2] as f32,
        ];
        fusion_ref.process_imu(to_fusion_imu(*imu_sample));
        let raw_gyro_deg = [
            imu_sample.gyro_radps[0].to_degrees(),
            imu_sample.gyro_radps[1].to_degrees(),
            imu_sample.gyro_radps[2].to_degrees(),
        ];

        let eskf_predict_mount_q_vb = fusion_ref
            .eskf_mount_q_vb()
            .or_else(|| fusion_ref.mount_q_vb());
        let (gyro, accel) = vehicle_measurements_from_mount(
            eskf_predict_mount_q_vb,
            [
                raw_gyro_radps[0] as f64,
                raw_gyro_radps[1] as f64,
                raw_gyro_radps[2] as f64,
            ],
            [
                raw_accel_mps2[0] as f64,
                raw_accel_mps2[1] as f64,
                raw_accel_mps2[2] as f64,
            ],
        );
        let (loose_gyro_deg, loose_accel) = match ekf_imu_source {
            EkfImuSource::Align => {
                if let Some(q_vb) = loose_seed_mount_q_vb {
                    vehicle_measurements_from_mount(
                        Some(q_vb),
                        [
                            raw_gyro_radps[0] as f64,
                            raw_gyro_radps[1] as f64,
                            raw_gyro_radps[2] as f64,
                        ],
                        [
                            raw_accel_mps2[0] as f64,
                            raw_accel_mps2[1] as f64,
                            raw_accel_mps2[2] as f64,
                        ],
                    )
                } else {
                    (raw_gyro_deg, imu_sample.accel_mps2)
                }
            }
            EkfImuSource::EsfAlg => {
                if let Some(alg) = cur_alg {
                    let r_sb = rot_zyx(
                        deg2rad(alg.yaw_deg),
                        deg2rad(alg.pitch_deg),
                        deg2rad(alg.roll_deg),
                    );
                    let mut loose_gyro = mat_vec(r_sb, raw_gyro_deg);
                    let mut loose_accel = mat_vec(r_sb, imu_sample.accel_mps2);
                    loose_gyro[1] = -loose_gyro[1];
                    loose_gyro[2] = -loose_gyro[2];
                    loose_accel[1] = -loose_accel[1];
                    loose_accel[2] = -loose_accel[2];
                    (loose_gyro, loose_accel)
                } else {
                    (raw_gyro_deg, imu_sample.accel_mps2)
                }
            }
        };

        let predict_gyro = if let Some(cutoff_hz) = cfg.predict_imu_lpf_cutoff_hz {
            let alpha_pred = lpf_alpha(dt, cutoff_hz);
            lpf_vec3(&mut filt_predict_gyro, loose_gyro_deg, alpha_pred)
        } else {
            loose_gyro_deg
        };
        let predict_accel = if let Some(cutoff_hz) = cfg.predict_imu_lpf_cutoff_hz {
            let alpha_pred = lpf_alpha(dt, cutoff_hz);
            lpf_vec3(&mut filt_predict_accel, loose_accel, alpha_pred)
        } else {
            loose_accel
        };
        predict_dt_accum += dt;
        predict_decim_count += 1;
        predict_gyro_sum[0] += predict_gyro[0];
        predict_gyro_sum[1] += predict_gyro[1];
        predict_gyro_sum[2] += predict_gyro[2];
        predict_accel_sum[0] += predict_accel[0];
        predict_accel_sum[1] += predict_accel[1];
        predict_accel_sum[2] += predict_accel[2];
        let predict_decimation = cfg.predict_imu_decimation.max(1);
        let mut eskf_initialized_this_pkt = false;
        if predict_decim_count >= predict_decimation {
            let pred_dt = predict_dt_accum.max(1.0e-6);
            let inv_block_len = 1.0 / (predict_decim_count.max(1) as f64);
            let avg_predict_gyro = [
                predict_gyro_sum[0] * inv_block_len,
                predict_gyro_sum[1] * inv_block_len,
                predict_gyro_sum[2] * inv_block_len,
            ];
            let avg_predict_accel = [
                predict_accel_sum[0] * inv_block_len,
                predict_accel_sum[1] * inv_block_len,
                predict_accel_sum[2] * inv_block_len,
            ];
            let loose_imu = LooseImuDelta {
                dax_1: (deg2rad(avg_predict_gyro[0]) * pred_dt) as f32,
                day_1: (deg2rad(avg_predict_gyro[1]) * pred_dt) as f32,
                daz_1: (deg2rad(avg_predict_gyro[2]) * pred_dt) as f32,
                dvx_1: (avg_predict_accel[0] * pred_dt) as f32,
                dvy_1: (avg_predict_accel[1] * pred_dt) as f32,
                dvz_1: (avg_predict_accel[2] * pred_dt) as f32,
                dax_2: (deg2rad(avg_predict_gyro[0]) * pred_dt) as f32,
                day_2: (deg2rad(avg_predict_gyro[1]) * pred_dt) as f32,
                daz_2: (deg2rad(avg_predict_gyro[2]) * pred_dt) as f32,
                dvx_2: (avg_predict_accel[0] * pred_dt) as f32,
                dvy_2: (avg_predict_accel[1] * pred_dt) as f32,
                dvz_2: (avg_predict_accel[2] * pred_dt) as f32,
                dt: pred_dt as f32,
            };
            if let Some(loose_ref) = loose.as_mut() {
                loose_ref.predict(loose_imu);
            }
            let loose_gyro_radps = [
                deg2rad(avg_predict_gyro[0]) as f32,
                deg2rad(avg_predict_gyro[1]) as f32,
                deg2rad(avg_predict_gyro[2]) as f32,
            ];
            let loose_accel_mps2 = [
                avg_predict_accel[0] as f32,
                avg_predict_accel[1] as f32,
                avg_predict_accel[2] as f32,
            ];
            let mut loose_batch_applied = false;

            while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt_t_ms {
                let (t_ms, nav) = nav_events[nav_idx];
                let gnss_sample = gnss_samples[nav_idx];
                nav_idx += 1;
                if in_gnss_outage(t_ms, &outage_windows_ms) {
                    continue;
                }

                if !origin_set {
                    ref_lat = nav.lat_deg;
                    ref_lon = nav.lon_deg;
                    ref_h = nav.height_m;
                    ref_ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
                    origin_set = true;
                }
                let ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
                let t = rel_s(t_ms);
                let update = fusion_ref.process_gnss(to_fusion_gnss(gnss_sample));
                if update.mount_ready_changed && update.mount_ready {
                    fusion_mount_ready_marker.push([t, nav.heading_vehicle_deg]);
                }
                if update.ekf_initialized_now {
                    fusion_ekf_init_marker.push([t, nav.heading_vehicle_deg]);
                }
                if loose.is_none() {
                    match ekf_imu_source {
                        EkfImuSource::Align => {
                            if update.mount_ready {
                                if let Some(q_vb) =
                                    cur_align_q_vb.or_else(|| fusion_ref.mount_q_vb())
                                {
                                    loose_seed_mount_q_vb = Some(q_vb);
                                    let loose_init = initialize_loose_from_nav(
                                        nav,
                                        gnss_sample,
                                        base_loose_predict_noise,
                                    );
                                    loose = Some(loose_init);
                                    loose_last_gps_update_ms = Some(t_ms);
                                }
                            }
                        }
                        EkfImuSource::EsfAlg => {
                            if cur_alg_status >= 3 && cur_alg.is_some() {
                                loose_seed_mount_q_vb = cur_alg.map(|alg| {
                                    let q = esf_alg_flu_to_frd_mount_quat(
                                        alg.roll_deg,
                                        alg.pitch_deg,
                                        alg.yaw_deg,
                                    );
                                    [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32]
                                });
                                let loose_init = initialize_loose_from_nav(
                                    nav,
                                    gnss_sample,
                                    base_loose_predict_noise,
                                );
                                loose = Some(loose_init);
                                loose_last_gps_update_ms = Some(t_ms);
                            }
                        }
                    }
                }
                if let Some(loose_ref) = loose.as_mut() {
                    let dt_since_last_gnss_s = loose_last_gps_update_ms
                        .map(|last_t_ms| ((t_ms - last_t_ms) * 1.0e-3) as f32)
                        .unwrap_or(1.0)
                        .clamp(1.0e-3, 1.0);
                    let vel_ecef = mat_vec(
                        transpose3(ecef_to_ned_matrix(nav.lat_deg, nav.lon_deg)),
                        [nav.vel_n_mps, nav.vel_e_mps, nav.vel_d_mps],
                    );
                    let vel_std_ned = [
                        (nav.s_acc_mps * cfg.gnss_vel_r_scale.sqrt()) as f32,
                        (nav.s_acc_mps * cfg.gnss_vel_r_scale.sqrt()) as f32,
                        (nav.s_acc_mps * cfg.gnss_vel_r_scale.sqrt()) as f32,
                    ];
                    loose_ref.fuse_reference_batch_full(
                        Some(ecef),
                        Some([vel_ecef[0] as f32, vel_ecef[1] as f32, vel_ecef[2] as f32]),
                        (nav.h_acc_m * cfg.gnss_pos_r_scale.sqrt()) as f32,
                        Some(vel_std_ned),
                        dt_since_last_gnss_s,
                        loose_gyro_radps,
                        loose_accel_mps2,
                        loose_imu.dt,
                    );
                    loose_last_gps_update_ms = Some(t_ms);
                    loose_batch_applied = true;
                }

                if let Some(eskf_ref) = fusion_ref.eskf() {
                    let (_, _, eskf_yaw) = quat_rpy_deg(
                        eskf_ref.nominal.q0,
                        eskf_ref.nominal.q1,
                        eskf_ref.nominal.q2,
                        eskf_ref.nominal.q3,
                    );
                    let (eskf_lat, eskf_lon, _eskf_h) = eskf_display_lla(&fusion_ref).unwrap_or((
                        nav.lat_deg,
                        nav.lon_deg,
                        nav.height_m,
                    ));
                    map_eskf_heading.push(HeadingSample {
                        t_s: t,
                        lon_deg: eskf_lon,
                        lat_deg: eskf_lat,
                        yaw_deg: eskf_yaw,
                    });
                }
                if update.ekf_initialized_now {
                    eskf_initialized_this_pkt = true;
                    if let Some(eskf_ref) = fusion_ref.eskf() {
                        append_eskf_sample(
                            eskf_ref,
                            t,
                            gyro,
                            accel,
                            dt,
                            fusion_ref.eskf_mount_q_vb().or(eskf_predict_mount_q_vb),
                            cfg.vehicle_meas_lpf_cutoff_hz,
                            &mut filt_eskf_meas_gyro,
                            &mut filt_eskf_meas_accel,
                            &mut eskf_cmp_pos_n,
                            &mut eskf_cmp_pos_e,
                            &mut eskf_cmp_pos_d,
                            &mut eskf_cmp_vel_forward,
                            &mut eskf_cmp_vel_lateral,
                            &mut eskf_cmp_vel_vertical,
                            &mut eskf_cmp_att_roll,
                            &mut eskf_cmp_att_pitch,
                            &mut eskf_cmp_att_yaw,
                            &mut eskf_mount_roll,
                            &mut eskf_mount_pitch,
                            &mut eskf_mount_yaw,
                            &mut eskf_meas_gyro_x,
                            &mut eskf_meas_gyro_y,
                            &mut eskf_meas_gyro_z,
                            &mut eskf_meas_accel_x,
                            &mut eskf_meas_accel_y,
                            &mut eskf_meas_accel_z,
                            &mut eskf_bias_gyro_x,
                            &mut eskf_bias_gyro_y,
                            &mut eskf_bias_gyro_z,
                            &mut eskf_bias_accel_x,
                            &mut eskf_bias_accel_y,
                            &mut eskf_bias_accel_z,
                            &mut eskf_cov_diag,
                            &mut eskf_yaw_cue_sum,
                            &mut eskf_yaw_cue_abs,
                            &mut eskf_yaw_cue_innov_sum,
                            &mut eskf_yaw_cue_innov_abs,
                        );
                    }
                    if let Some(loose_ref) = loose.as_ref() {
                        append_loose_sample(
                            loose_ref,
                            t,
                            gyro,
                            accel,
                            dt,
                            loose_seed_mount_q_vb,
                            ref_ecef,
                            ref_lat,
                            ref_lon,
                            cfg.vehicle_meas_lpf_cutoff_hz,
                            &mut loose_cmp_pos_n,
                            &mut loose_cmp_pos_e,
                            &mut loose_cmp_pos_d,
                            &mut loose_cmp_vel_forward,
                            &mut loose_cmp_vel_lateral,
                            &mut loose_cmp_vel_vertical,
                            &mut loose_cmp_att_roll,
                            &mut loose_cmp_att_pitch,
                            &mut loose_cmp_att_yaw,
                            &mut loose_mount_roll,
                            &mut loose_mount_pitch,
                            &mut loose_mount_yaw,
                            &mut loose_meas_gyro_x,
                            &mut loose_meas_gyro_y,
                            &mut loose_meas_gyro_z,
                            &mut loose_meas_accel_x,
                            &mut loose_meas_accel_y,
                            &mut loose_meas_accel_z,
                            &mut loose_bias_gyro_x,
                            &mut loose_bias_gyro_y,
                            &mut loose_bias_gyro_z,
                            &mut loose_bias_accel_x,
                            &mut loose_bias_accel_y,
                            &mut loose_bias_accel_z,
                            &mut loose_scale_gyro_x,
                            &mut loose_scale_gyro_y,
                            &mut loose_scale_gyro_z,
                            &mut loose_scale_accel_x,
                            &mut loose_scale_accel_y,
                            &mut loose_scale_accel_z,
                            &mut loose_cov_diag,
                        );
                    }
                }
                if let Some(loose_ref) = loose.as_ref() {
                    let (loose_pos_ned, _, loose_q_ns) =
                        loose_display_state(loose_ref, ref_ecef, ref_lat, ref_lon);
                    let (_, _, loose_yaw) = quat_rpy_deg(
                        loose_q_ns[0] as f32,
                        loose_q_ns[1] as f32,
                        loose_q_ns[2] as f32,
                        loose_q_ns[3] as f32,
                    );
                    let (loose_lat, loose_lon, _loose_h) = ned_to_lla_exact(
                        loose_pos_ned[0],
                        loose_pos_ned[1],
                        loose_pos_ned[2],
                        ref_lat,
                        ref_lon,
                        ref_h,
                    );
                    map_loose_heading.push(HeadingSample {
                        t_s: t,
                        lon_deg: loose_lon,
                        lat_deg: loose_lat,
                        yaw_deg: loose_yaw,
                    });
                }
            }
            if let Some(loose_ref) = loose.as_mut() {
                if !loose_batch_applied {
                    loose_ref.fuse_reference_batch_full(
                        None,
                        None,
                        0.0,
                        None,
                        1.0,
                        loose_gyro_radps,
                        loose_accel_mps2,
                        loose_imu.dt,
                    );
                }
            }

            predict_gyro_sum = [0.0; 3];
            predict_accel_sum = [0.0; 3];
            predict_dt_accum = 0.0;
            predict_decim_count = 0;
        }

        if origin_set {
            if let Some(_eskf_ref) = fusion_ref.eskf() {
                let (eskf_lat, eskf_lon, _eskf_h) =
                    eskf_display_lla(&fusion_ref).unwrap_or((ref_lat, ref_lon, ref_h));
                map_eskf.push([eskf_lon, eskf_lat]);
                if outage_active {
                    if !prev_outage_active && !map_eskf_outage.is_empty() {
                        map_eskf_outage.push([f64::NAN, f64::NAN]);
                    }
                    map_eskf_outage.push([eskf_lon, eskf_lat]);
                }
            }
            if let Some(loose_ref) = loose.as_ref() {
                let (loose_pos_ned, _, _) =
                    loose_display_state(loose_ref, ref_ecef, ref_lat, ref_lon);
                let (loose_lat, loose_lon, _loose_h) = ned_to_lla_exact(
                    loose_pos_ned[0],
                    loose_pos_ned[1],
                    loose_pos_ned[2],
                    ref_lat,
                    ref_lon,
                    ref_h,
                );
                map_loose.push([loose_lon, loose_lat]);
            }
        }
        prev_outage_active = outage_active;

        let t_imu = imu_sample.t_s;
        let dt_safe = dt.max(1.0e-6);
        if let Some(eskf_ref) = fusion_ref.eskf()
            && !eskf_initialized_this_pkt
        {
            append_eskf_sample(
                eskf_ref,
                t_imu,
                gyro,
                accel,
                dt_safe,
                fusion_ref.eskf_mount_q_vb().or(eskf_predict_mount_q_vb),
                cfg.vehicle_meas_lpf_cutoff_hz,
                &mut filt_eskf_meas_gyro,
                &mut filt_eskf_meas_accel,
                &mut eskf_cmp_pos_n,
                &mut eskf_cmp_pos_e,
                &mut eskf_cmp_pos_d,
                &mut eskf_cmp_vel_forward,
                &mut eskf_cmp_vel_lateral,
                &mut eskf_cmp_vel_vertical,
                &mut eskf_cmp_att_roll,
                &mut eskf_cmp_att_pitch,
                &mut eskf_cmp_att_yaw,
                &mut eskf_mount_roll,
                &mut eskf_mount_pitch,
                &mut eskf_mount_yaw,
                &mut eskf_meas_gyro_x,
                &mut eskf_meas_gyro_y,
                &mut eskf_meas_gyro_z,
                &mut eskf_meas_accel_x,
                &mut eskf_meas_accel_y,
                &mut eskf_meas_accel_z,
                &mut eskf_bias_gyro_x,
                &mut eskf_bias_gyro_y,
                &mut eskf_bias_gyro_z,
                &mut eskf_bias_accel_x,
                &mut eskf_bias_accel_y,
                &mut eskf_bias_accel_z,
                &mut eskf_cov_diag,
                &mut eskf_yaw_cue_sum,
                &mut eskf_yaw_cue_abs,
                &mut eskf_yaw_cue_innov_sum,
                &mut eskf_yaw_cue_innov_abs,
            );
        }
        if let Some(loose_ref) = loose.as_ref() {
            append_loose_sample(
                loose_ref,
                t_imu,
                gyro,
                accel,
                dt_safe,
                loose_seed_mount_q_vb,
                ref_ecef,
                ref_lat,
                ref_lon,
                cfg.vehicle_meas_lpf_cutoff_hz,
                &mut loose_cmp_pos_n,
                &mut loose_cmp_pos_e,
                &mut loose_cmp_pos_d,
                &mut loose_cmp_vel_forward,
                &mut loose_cmp_vel_lateral,
                &mut loose_cmp_vel_vertical,
                &mut loose_cmp_att_roll,
                &mut loose_cmp_att_pitch,
                &mut loose_cmp_att_yaw,
                &mut loose_mount_roll,
                &mut loose_mount_pitch,
                &mut loose_mount_yaw,
                &mut loose_meas_gyro_x,
                &mut loose_meas_gyro_y,
                &mut loose_meas_gyro_z,
                &mut loose_meas_accel_x,
                &mut loose_meas_accel_y,
                &mut loose_meas_accel_z,
                &mut loose_bias_gyro_x,
                &mut loose_bias_gyro_y,
                &mut loose_bias_gyro_z,
                &mut loose_bias_accel_x,
                &mut loose_bias_accel_y,
                &mut loose_bias_accel_z,
                &mut loose_scale_gyro_x,
                &mut loose_scale_gyro_y,
                &mut loose_scale_gyro_z,
                &mut loose_scale_accel_x,
                &mut loose_scale_accel_y,
                &mut loose_scale_accel_z,
                &mut loose_cov_diag,
            );
        }
    }

    let (eskf_bump_pitch_speed, eskf_bump_diag) = build_bump_diagnostic_traces(
        &eskf_cmp_att_pitch,
        &eskf_cmp_vel_forward,
        &eskf_cmp_vel_lateral,
        &ubx_vel_forward,
        &ubx_vel_lateral,
    );

    let eskf_cmp_pos = vec![
        Trace {
            name: "ESKF posN [m]".to_string(),
            points: eskf_cmp_pos_n,
        },
        Trace {
            name: "UBX posN [m]".to_string(),
            points: ubx_pos_n.clone(),
        },
        Trace {
            name: "ESKF posE [m]".to_string(),
            points: eskf_cmp_pos_e,
        },
        Trace {
            name: "UBX posE [m]".to_string(),
            points: ubx_pos_e.clone(),
        },
        Trace {
            name: "ESKF posD [m]".to_string(),
            points: eskf_cmp_pos_d,
        },
        Trace {
            name: "UBX posD [m]".to_string(),
            points: ubx_pos_d.clone(),
        },
    ];
    let eskf_cmp_vel = vec![
        Trace {
            name: "ESKF forward vel [m/s]".to_string(),
            points: eskf_cmp_vel_forward,
        },
        Trace {
            name: "u-blox forward vel [m/s]".to_string(),
            points: ubx_vel_forward.clone(),
        },
        Trace {
            name: "ESKF lateral vel [m/s]".to_string(),
            points: eskf_cmp_vel_lateral,
        },
        Trace {
            name: "u-blox lateral vel [m/s]".to_string(),
            points: ubx_vel_lateral.clone(),
        },
        Trace {
            name: "ESKF vertical vel [m/s]".to_string(),
            points: eskf_cmp_vel_vertical,
        },
        Trace {
            name: "u-blox vertical vel [m/s]".to_string(),
            points: ubx_vel_vertical.clone(),
        },
    ];
    let eskf_cmp_att = vec![
        Trace {
            name: "ESKF roll [deg]".to_string(),
            points: eskf_cmp_att_roll,
        },
        Trace {
            name: "NAV-ATT roll [deg]".to_string(),
            points: ubx_att_roll.clone(),
        },
        Trace {
            name: "ESKF pitch [deg]".to_string(),
            points: eskf_cmp_att_pitch,
        },
        Trace {
            name: "NAV-ATT pitch [deg]".to_string(),
            points: ubx_att_pitch.clone(),
        },
        Trace {
            name: "ESKF yaw [deg]".to_string(),
            points: eskf_cmp_att_yaw,
        },
        Trace {
            name: "NAV-ATT heading [deg]".to_string(),
            points: ubx_att_yaw.clone(),
        },
        Trace {
            name: "mount ready".to_string(),
            points: fusion_mount_ready_marker,
        },
        Trace {
            name: "EKF initialized".to_string(),
            points: fusion_ekf_init_marker,
        },
    ];
    let eskf_meas_gyro = vec![
        Trace {
            name: "ESKF vehicle gyro x [deg/s]".to_string(),
            points: eskf_meas_gyro_x,
        },
        Trace {
            name: "ESKF vehicle gyro y [deg/s]".to_string(),
            points: eskf_meas_gyro_y,
        },
        Trace {
            name: "ESKF vehicle gyro z [deg/s]".to_string(),
            points: eskf_meas_gyro_z,
        },
    ];
    let eskf_meas_accel = vec![
        Trace {
            name: "ESKF vehicle accel x [m/s^2]".to_string(),
            points: eskf_meas_accel_x,
        },
        Trace {
            name: "ESKF vehicle accel y [m/s^2]".to_string(),
            points: eskf_meas_accel_y,
        },
        Trace {
            name: "ESKF vehicle accel z [m/s^2]".to_string(),
            points: eskf_meas_accel_z,
        },
    ];
    let eskf_bias_gyro = vec![
        Trace {
            name: "ESKF gyro bias x [deg/s]".to_string(),
            points: eskf_bias_gyro_x,
        },
        Trace {
            name: "ESKF gyro bias y [deg/s]".to_string(),
            points: eskf_bias_gyro_y,
        },
        Trace {
            name: "ESKF gyro bias z [deg/s]".to_string(),
            points: eskf_bias_gyro_z,
        },
    ];
    let eskf_bias_accel = vec![
        Trace {
            name: "ESKF accel bias x [m/s^2]".to_string(),
            points: eskf_bias_accel_x,
        },
        Trace {
            name: "ESKF accel bias y [m/s^2]".to_string(),
            points: eskf_bias_accel_y,
        },
        Trace {
            name: "ESKF accel bias z [m/s^2]".to_string(),
            points: eskf_bias_accel_z,
        },
    ];
    let eskf_cov_bias = vec![
        Trace {
            name: "acc_x".to_string(),
            points: eskf_cov_diag[12].clone(),
        },
        Trace {
            name: "acc_y".to_string(),
            points: eskf_cov_diag[13].clone(),
        },
        Trace {
            name: "acc_z".to_string(),
            points: eskf_cov_diag[14].clone(),
        },
        Trace {
            name: "gyro_x".to_string(),
            points: eskf_cov_diag[9].clone(),
        },
        Trace {
            name: "gyro_y".to_string(),
            points: eskf_cov_diag[10].clone(),
        },
        Trace {
            name: "gyro_z".to_string(),
            points: eskf_cov_diag[11].clone(),
        },
    ];
    let eskf_cov_nonbias = vec![
        Trace {
            name: "p_n".to_string(),
            points: eskf_cov_diag[6].clone(),
        },
        Trace {
            name: "p_e".to_string(),
            points: eskf_cov_diag[7].clone(),
        },
        Trace {
            name: "p_d".to_string(),
            points: eskf_cov_diag[8].clone(),
        },
        Trace {
            name: "v_n".to_string(),
            points: eskf_cov_diag[3].clone(),
        },
        Trace {
            name: "v_e".to_string(),
            points: eskf_cov_diag[4].clone(),
        },
        Trace {
            name: "v_d".to_string(),
            points: eskf_cov_diag[5].clone(),
        },
        Trace {
            name: "theta_x".to_string(),
            points: eskf_cov_diag[0].clone(),
        },
        Trace {
            name: "theta_y".to_string(),
            points: eskf_cov_diag[1].clone(),
        },
        Trace {
            name: "theta_z".to_string(),
            points: eskf_cov_diag[2].clone(),
        },
        Trace {
            name: "mount_x".to_string(),
            points: eskf_cov_diag[15].clone(),
        },
        Trace {
            name: "mount_y".to_string(),
            points: eskf_cov_diag[16].clone(),
        },
        Trace {
            name: "mount_z".to_string(),
            points: eskf_cov_diag[17].clone(),
        },
    ];
    let mut eskf_stationary_diag = vec![
        Trace {
            name: "stationary innov x".to_string(),
            points: eskf_stationary_innov_x,
        },
        Trace {
            name: "stationary innov y".to_string(),
            points: eskf_stationary_innov_y,
        },
        Trace {
            name: "stationary K theta_x from x".to_string(),
            points: eskf_stationary_k_theta_x_from_x,
        },
        Trace {
            name: "stationary K theta_y from x".to_string(),
            points: eskf_stationary_k_theta_y_from_x,
        },
        Trace {
            name: "stationary K theta_x from y".to_string(),
            points: eskf_stationary_k_theta_x_from_y,
        },
        Trace {
            name: "stationary K theta_y from y".to_string(),
            points: eskf_stationary_k_theta_y_from_y,
        },
        Trace {
            name: "stationary K bax from x".to_string(),
            points: eskf_stationary_k_bax_from_x,
        },
        Trace {
            name: "stationary K bay from y".to_string(),
            points: eskf_stationary_k_bay_from_y,
        },
        Trace {
            name: "stationary P theta_x".to_string(),
            points: eskf_stationary_p_theta_x,
        },
        Trace {
            name: "stationary P theta_y".to_string(),
            points: eskf_stationary_p_theta_y,
        },
        Trace {
            name: "stationary P bax".to_string(),
            points: eskf_stationary_p_bax,
        },
        Trace {
            name: "stationary P bay".to_string(),
            points: eskf_stationary_p_bay,
        },
        Trace {
            name: "stationary P theta_x_bax".to_string(),
            points: eskf_stationary_p_theta_x_bax,
        },
        Trace {
            name: "stationary P theta_y_bay".to_string(),
            points: eskf_stationary_p_theta_y_bay,
        },
    ];
    for (idx, name) in ESKF_YAW_CUE_NAMES.iter().enumerate() {
        eskf_stationary_diag.push(Trace {
            name: format!("mount yaw dx sum {name} [deg]"),
            points: eskf_yaw_cue_sum[idx].clone(),
        });
        eskf_stationary_diag.push(Trace {
            name: format!("mount yaw dx abs {name} [deg]"),
            points: eskf_yaw_cue_abs[idx].clone(),
        });
        eskf_stationary_diag.push(Trace {
            name: format!("innovation sum {name}"),
            points: eskf_yaw_cue_innov_sum[idx].clone(),
        });
        eskf_stationary_diag.push(Trace {
            name: format!("innovation abs {name}"),
            points: eskf_yaw_cue_innov_abs[idx].clone(),
        });
    }
    let eskf_map = vec![
        Trace {
            name: "u-blox path (lon,lat)".to_string(),
            points: map_ubx,
        },
        Trace {
            name: "NAV2-PVT path (GNSS-only, lon,lat)".to_string(),
            points: map_nav2,
        },
        Trace {
            name: "ESKF path (lon,lat)".to_string(),
            points: map_eskf,
        },
        Trace {
            name: "ESKF path during GNSS outage (lon,lat)".to_string(),
            points: map_eskf_outage,
        },
    ];
    let loose_cmp_pos = vec![
        Trace {
            name: "Loose posN [m]".to_string(),
            points: loose_cmp_pos_n,
        },
        Trace {
            name: "UBX posN [m]".to_string(),
            points: ubx_pos_n.clone(),
        },
        Trace {
            name: "Loose posE [m]".to_string(),
            points: loose_cmp_pos_e,
        },
        Trace {
            name: "UBX posE [m]".to_string(),
            points: ubx_pos_e.clone(),
        },
        Trace {
            name: "Loose posD [m]".to_string(),
            points: loose_cmp_pos_d,
        },
        Trace {
            name: "UBX posD [m]".to_string(),
            points: ubx_pos_d.clone(),
        },
    ];
    let loose_cmp_vel = vec![
        Trace {
            name: "Loose forward vel [m/s]".to_string(),
            points: loose_cmp_vel_forward,
        },
        Trace {
            name: "u-blox forward vel [m/s]".to_string(),
            points: ubx_vel_forward.clone(),
        },
        Trace {
            name: "Loose lateral vel [m/s]".to_string(),
            points: loose_cmp_vel_lateral,
        },
        Trace {
            name: "u-blox lateral vel [m/s]".to_string(),
            points: ubx_vel_lateral.clone(),
        },
        Trace {
            name: "Loose vertical vel [m/s]".to_string(),
            points: loose_cmp_vel_vertical,
        },
        Trace {
            name: "u-blox vertical vel [m/s]".to_string(),
            points: ubx_vel_vertical.clone(),
        },
    ];
    let loose_cmp_att = vec![
        Trace {
            name: "Loose roll [deg]".to_string(),
            points: loose_cmp_att_roll,
        },
        Trace {
            name: "NAV-ATT roll [deg]".to_string(),
            points: ubx_att_roll.clone(),
        },
        Trace {
            name: "Loose pitch [deg]".to_string(),
            points: loose_cmp_att_pitch,
        },
        Trace {
            name: "NAV-ATT pitch [deg]".to_string(),
            points: ubx_att_pitch.clone(),
        },
        Trace {
            name: "Loose yaw [deg]".to_string(),
            points: loose_cmp_att_yaw,
        },
        Trace {
            name: "NAV-ATT heading [deg]".to_string(),
            points: ubx_att_yaw.clone(),
        },
    ];
    let mut eskf_misalignment = vec![
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
    ];
    eskf_misalignment.extend(esf_alg_mount_ref.clone());
    let mut loose_misalignment = vec![
        Trace {
            name: "Loose full mount roll [deg]".to_string(),
            points: loose_mount_roll,
        },
        Trace {
            name: "Loose full mount pitch [deg]".to_string(),
            points: loose_mount_pitch,
        },
        Trace {
            name: "Loose full mount yaw [deg]".to_string(),
            points: loose_mount_yaw,
        },
    ];
    loose_misalignment.extend(esf_alg_mount_ref);
    let loose_meas_gyro = vec![
        Trace {
            name: "Loose vehicle gyro x [deg/s]".to_string(),
            points: loose_meas_gyro_x,
        },
        Trace {
            name: "Loose vehicle gyro y [deg/s]".to_string(),
            points: loose_meas_gyro_y,
        },
        Trace {
            name: "Loose vehicle gyro z [deg/s]".to_string(),
            points: loose_meas_gyro_z,
        },
    ];
    let loose_meas_accel = vec![
        Trace {
            name: "Loose vehicle accel x [m/s^2]".to_string(),
            points: loose_meas_accel_x,
        },
        Trace {
            name: "Loose vehicle accel y [m/s^2]".to_string(),
            points: loose_meas_accel_y,
        },
        Trace {
            name: "Loose vehicle accel z [m/s^2]".to_string(),
            points: loose_meas_accel_z,
        },
    ];
    let loose_bias_gyro = vec![
        Trace {
            name: "Loose gyro bias x [deg/s]".to_string(),
            points: loose_bias_gyro_x,
        },
        Trace {
            name: "Loose gyro bias y [deg/s]".to_string(),
            points: loose_bias_gyro_y,
        },
        Trace {
            name: "Loose gyro bias z [deg/s]".to_string(),
            points: loose_bias_gyro_z,
        },
    ];
    let loose_bias_accel = vec![
        Trace {
            name: "Loose accel bias x [m/s^2]".to_string(),
            points: loose_bias_accel_x,
        },
        Trace {
            name: "Loose accel bias y [m/s^2]".to_string(),
            points: loose_bias_accel_y,
        },
        Trace {
            name: "Loose accel bias z [m/s^2]".to_string(),
            points: loose_bias_accel_z,
        },
    ];
    let loose_scale_gyro = vec![
        Trace {
            name: "Loose gyro scale x".to_string(),
            points: loose_scale_gyro_x,
        },
        Trace {
            name: "Loose gyro scale y".to_string(),
            points: loose_scale_gyro_y,
        },
        Trace {
            name: "Loose gyro scale z".to_string(),
            points: loose_scale_gyro_z,
        },
    ];
    let loose_scale_accel = vec![
        Trace {
            name: "Loose accel scale x".to_string(),
            points: loose_scale_accel_x,
        },
        Trace {
            name: "Loose accel scale y".to_string(),
            points: loose_scale_accel_y,
        },
        Trace {
            name: "Loose accel scale z".to_string(),
            points: loose_scale_accel_z,
        },
    ];
    let loose_cov_bias = vec![
        Trace {
            name: "acc_x".to_string(),
            points: loose_cov_diag[9].clone(),
        },
        Trace {
            name: "acc_y".to_string(),
            points: loose_cov_diag[10].clone(),
        },
        Trace {
            name: "acc_z".to_string(),
            points: loose_cov_diag[11].clone(),
        },
        Trace {
            name: "gyro_x".to_string(),
            points: loose_cov_diag[12].clone(),
        },
        Trace {
            name: "gyro_y".to_string(),
            points: loose_cov_diag[13].clone(),
        },
        Trace {
            name: "gyro_z".to_string(),
            points: loose_cov_diag[14].clone(),
        },
        Trace {
            name: "gyro_scale_x".to_string(),
            points: loose_cov_diag[18].clone(),
        },
        Trace {
            name: "gyro_scale_y".to_string(),
            points: loose_cov_diag[19].clone(),
        },
        Trace {
            name: "gyro_scale_z".to_string(),
            points: loose_cov_diag[20].clone(),
        },
        Trace {
            name: "acc_scale_x".to_string(),
            points: loose_cov_diag[15].clone(),
        },
        Trace {
            name: "acc_scale_y".to_string(),
            points: loose_cov_diag[16].clone(),
        },
        Trace {
            name: "acc_scale_z".to_string(),
            points: loose_cov_diag[17].clone(),
        },
    ];
    let loose_cov_nonbias = vec![
        Trace {
            name: "p_n".to_string(),
            points: loose_cov_diag[0].clone(),
        },
        Trace {
            name: "p_e".to_string(),
            points: loose_cov_diag[1].clone(),
        },
        Trace {
            name: "p_d".to_string(),
            points: loose_cov_diag[2].clone(),
        },
        Trace {
            name: "v_n".to_string(),
            points: loose_cov_diag[3].clone(),
        },
        Trace {
            name: "v_e".to_string(),
            points: loose_cov_diag[4].clone(),
        },
        Trace {
            name: "v_d".to_string(),
            points: loose_cov_diag[5].clone(),
        },
        Trace {
            name: "theta_x".to_string(),
            points: loose_cov_diag[6].clone(),
        },
        Trace {
            name: "theta_y".to_string(),
            points: loose_cov_diag[7].clone(),
        },
        Trace {
            name: "theta_z".to_string(),
            points: loose_cov_diag[8].clone(),
        },
        Trace {
            name: "psi_cc_x".to_string(),
            points: loose_cov_diag[21].clone(),
        },
        Trace {
            name: "psi_cc_y".to_string(),
            points: loose_cov_diag[22].clone(),
        },
        Trace {
            name: "psi_cc_z".to_string(),
            points: loose_cov_diag[23].clone(),
        },
    ];
    let loose_map = vec![Trace {
        name: "Loose path (lon,lat)".to_string(),
        points: map_loose,
    }];

    EkfCompareData {
        eskf_cmp_pos,
        eskf_cmp_vel,
        eskf_cmp_att,
        eskf_meas_gyro,
        eskf_meas_accel,
        eskf_bias_gyro,
        eskf_bias_accel,
        eskf_cov_bias,
        eskf_cov_nonbias,
        eskf_misalignment,
        eskf_stationary_diag,
        eskf_bump_pitch_speed,
        eskf_bump_diag,
        eskf_map,
        eskf_map_heading: map_eskf_heading,
        loose_cmp_pos,
        loose_cmp_vel,
        loose_cmp_att,
        loose_misalignment,
        loose_meas_gyro,
        loose_meas_accel,
        loose_bias_gyro,
        loose_bias_accel,
        loose_scale_gyro,
        loose_scale_accel,
        loose_cov_bias,
        loose_cov_nonbias,
        loose_map,
        loose_map_heading: map_loose_heading,
    }
}

fn build_bump_diagnostic_traces(
    eskf_pitch: &[[f64; 2]],
    eskf_vel_n: &[[f64; 2]],
    eskf_vel_e: &[[f64; 2]],
    ubx_vel_n: &[[f64; 2]],
    ubx_vel_e: &[[f64; 2]],
) -> (Vec<Trace>, Vec<Trace>) {
    const WINDOW_S: f64 = 3.0;
    const FFT_STRIDE: usize = 5;
    const FFT_MAX_HZ: f64 = 12.0;

    let eskf_speed: Vec<[f64; 2]> = eskf_vel_n
        .iter()
        .zip(eskf_vel_e.iter())
        .map(|(vn, ve)| [vn[0], vn[1].hypot(ve[1])])
        .collect();
    let ubx_speed: Vec<[f64; 2]> = ubx_vel_n
        .iter()
        .zip(ubx_vel_e.iter())
        .map(|(vn, ve)| [vn[0], vn[1].hypot(ve[1])])
        .collect();

    let mut pitch_hpf = Vec::<[f64; 2]>::with_capacity(eskf_pitch.len());
    let mut pitch_rms = Vec::<[f64; 2]>::with_capacity(eskf_pitch.len());
    let mut pitch_abs_ema = Vec::<[f64; 2]>::with_capacity(eskf_pitch.len());
    let mut fft_dom_mag = Vec::<[f64; 2]>::new();
    let mut fft_dom_freq = Vec::<[f64; 2]>::new();

    let mut lp_pitch = eskf_pitch.first().map(|p| p[1]).unwrap_or(0.0);
    let mut abs_ema = 0.0_f64;
    let mut prev_t: Option<f64> = None;
    let mut hp_window = VecDeque::<[f64; 2]>::new();
    let mut sum_sq = 0.0_f64;

    for (idx, p) in eskf_pitch.iter().enumerate() {
        let t = p[0];
        let pitch_deg = p[1];
        let dt = prev_t
            .map(|prev| (t - prev).clamp(1.0e-3, 0.1))
            .unwrap_or(0.01);
        prev_t = Some(t);

        let alpha = 1.0 - (-dt / WINDOW_S).exp();
        lp_pitch += alpha * (pitch_deg - lp_pitch);
        let hp = pitch_deg - lp_pitch;
        let hp_abs = hp.abs();
        abs_ema += alpha * (hp_abs - abs_ema);

        hp_window.push_back([t, hp]);
        sum_sq += hp * hp;
        while let Some(front) = hp_window.front().copied() {
            if t - front[0] > WINDOW_S {
                sum_sq -= front[1] * front[1];
                hp_window.pop_front();
            } else {
                break;
            }
        }
        let n = hp_window.len().max(1) as f64;
        pitch_hpf.push([t, hp]);
        pitch_rms.push([t, (sum_sq / n).sqrt()]);
        pitch_abs_ema.push([t, abs_ema]);

        if idx % FFT_STRIDE == 0
            && hp_window.len() >= 16
            && let Some((freq_hz, mag_deg)) = dominant_fft_metric(&hp_window, FFT_MAX_HZ)
        {
            fft_dom_freq.push([t, freq_hz]);
            fft_dom_mag.push([t, mag_deg]);
        }
    }

    (
        vec![
            Trace {
                name: "ESKF pitch [deg]".to_string(),
                points: eskf_pitch.to_vec(),
            },
            Trace {
                name: "ESKF horiz speed [m/s]".to_string(),
                points: eskf_speed,
            },
            Trace {
                name: "UBX horiz speed [m/s]".to_string(),
                points: ubx_speed,
            },
        ],
        vec![
            Trace {
                name: "Pitch HPF [deg]".to_string(),
                points: pitch_hpf,
            },
            Trace {
                name: "Pitch HPF RMS 3.0s [deg]".to_string(),
                points: pitch_rms,
            },
            Trace {
                name: "Pitch |HPF| EMA 3.0s [deg]".to_string(),
                points: pitch_abs_ema,
            },
            Trace {
                name: "Pitch FFT dom mag 3.0s [deg]".to_string(),
                points: fft_dom_mag,
            },
            Trace {
                name: "Pitch FFT dom freq 3.0s [Hz]".to_string(),
                points: fft_dom_freq,
            },
        ],
    )
}

fn dominant_fft_metric(window: &VecDeque<[f64; 2]>, max_hz: f64) -> Option<(f64, f64)> {
    let n = window.len();
    if n < 16 {
        return None;
    }
    let t0 = window.front().map(|p| p[0])?;
    let t1 = window.back().map(|p| p[0])?;
    let duration = (t1 - t0).max(1.0e-6);
    let dt_avg = duration / ((n - 1) as f64);
    let nyquist = 0.5 / dt_avg;
    if !nyquist.is_finite() || nyquist <= 0.0 {
        return None;
    }

    let max_bin = ((max_hz * n as f64 * dt_avg).floor() as usize).min(n / 2);
    if max_bin < 1 {
        return None;
    }

    let mean = window.iter().map(|p| p[1]).sum::<f64>() / n as f64;
    let denom = (n.saturating_sub(1)) as f64;
    let mut best_freq = 0.0_f64;
    let mut best_mag = 0.0_f64;

    for k in 1..=max_bin {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let mut win_sum = 0.0_f64;
        for (i, sample) in window.iter().enumerate() {
            let hann = if denom > 0.0 {
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos()
            } else {
                1.0
            };
            let x = (sample[1] - mean) * hann;
            win_sum += hann;
            let phase = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            re += x * phase.cos();
            im -= x * phase.sin();
        }
        let mag = if win_sum > 0.0 {
            2.0 * re.hypot(im) / win_sum
        } else {
            0.0
        };
        if mag > best_mag {
            best_mag = mag;
            best_freq = k as f64 / (n as f64 * dt_avg);
        }
    }

    Some((best_freq, best_mag))
}

fn sample_gnss_outage_windows(
    nav_events: &[(f64, NavPvtObs)],
    cfg: GnssOutageConfig,
) -> Vec<(f64, f64)> {
    if cfg.count == 0 || cfg.duration_s <= 0.0 || nav_events.len() < 2 {
        return Vec::new();
    }
    let duration_ms = cfg.duration_s * 1.0e3;
    let t_min = nav_events.first().map(|(t_ms, _)| *t_ms).unwrap_or(0.0);
    let t_max = nav_events.last().map(|(t_ms, _)| *t_ms).unwrap_or(t_min);
    if t_max - t_min <= duration_ms {
        return Vec::new();
    }

    let mut rng = Lcg64::new(cfg.seed);
    let mut windows = Vec::<(f64, f64)>::new();
    let start_min = t_min;
    let start_max = t_max - duration_ms;
    let mut attempts = 0usize;
    let max_attempts = cfg.count.saturating_mul(200).max(200);
    while windows.len() < cfg.count && attempts < max_attempts {
        attempts += 1;
        let start_ms = start_min + rng.next_unit_f64() * (start_max - start_min);
        let end_ms = start_ms + duration_ms;
        if windows.iter().any(|(a, b)| start_ms < *b && end_ms > *a) {
            continue;
        }
        windows.push((start_ms, end_ms));
    }
    windows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    windows
}

fn in_gnss_outage(t_ms: f64, windows_ms: &[(f64, f64)]) -> bool {
    windows_ms
        .iter()
        .any(|(start_ms, end_ms)| t_ms >= *start_ms && t_ms <= *end_ms)
}

struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_unit_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

fn initialize_loose_from_nav(
    nav: NavPvtObs,
    gnss: crate::datasets::generic_replay::GenericGnssSample,
    noise: LoosePredictNoise,
) -> LooseFilter {
    let mut loose = LooseFilter::new(noise);
    let p_diag = default_loose_reference_p_diag(to_fusion_gnss(gnss));
    let vel_ecef = mat_vec(
        transpose3(ecef_to_ned_matrix(nav.lat_deg, nav.lon_deg)),
        [nav.vel_n_mps, nav.vel_e_mps, nav.vel_d_mps],
    );
    loose.init_seeded_vehicle_from_nav_ecef_state(
        gnss.heading_rad.unwrap_or(0.0) as f32,
        nav.lat_deg,
        nav.lon_deg,
        lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m),
        [vel_ecef[0] as f32, vel_ecef[1] as f32, vel_ecef[2] as f32],
        Some(p_diag),
        None,
    );
    loose
}

fn build_align_mount_events(
    frames: &[UbxFrame],
    tl: &MasterTimeline,
    imu_cfg: ImuReplayConfig,
) -> Vec<(f64, [f32; 4])> {
    if tl.masters.is_empty() {
        return Vec::new();
    }

    let cfg = sensor_fusion::align::AlignConfig::default();
    let bootstrap_cfg = AlignBootstrapConfig {
        ema_alpha: 0.05,
        max_speed_mps: 0.35,
        stationary_samples: 100,
        max_gyro_radps: cfg.max_stationary_gyro_radps,
        max_accel_norm_err_mps2: cfg.max_stationary_accel_norm_err_mps2,
        max_speed_rate_mps2: 0.15,
        max_course_rate_radps: 1.0_f32.to_radians(),
    };
    let replay = build_align_replay(frames, tl, cfg, bootstrap_cfg, imu_cfg);
    let mut out = Vec::<(f64, [f32; 4])>::new();
    let mut ready = false;
    for sample in replay.samples {
        if sample.yaw_initialized {
            ready = true;
        }
        if ready {
            out.push((
                sample.t_ms,
                [
                    sample.q_align[0] as f32,
                    sample.q_align[1] as f32,
                    sample.q_align[2] as f32,
                    sample.q_align[3] as f32,
                ],
            ));
        }
    }
    out
}

fn default_loose_reference_p_diag(gnss: sensor_fusion::fusion::FusionGnssSample) -> [f32; 24] {
    const DEFAULT_GYRO_BIAS_SIGMA_DPS: f32 = 0.125;
    const DEFAULT_ACCEL_BIAS_SIGMA_MPS2: f32 = 0.075;
    const DEFAULT_GYRO_SCALE_SIGMA: f32 = 0.02;
    const DEFAULT_ACCEL_SCALE_SIGMA: f32 = 0.02;

    let att_sigma_rad = 2.0f32 * core::f32::consts::PI / 180.0;
    let att_var = att_sigma_rad * att_sigma_rad;
    let mut vel_std = gnss.vel_std_mps[0]
        .max(gnss.vel_std_mps[1])
        .max(gnss.vel_std_mps[2]);
    if vel_std < 0.2 {
        vel_std = 0.2;
    }
    let vel_var = vel_std * vel_std;
    let pos_n = gnss.pos_std_m[0].max(0.5);
    let pos_e = gnss.pos_std_m[1].max(0.5);
    let pos_d = gnss.pos_std_m[2].max(0.5);
    let gyro_bias_sigma_radps = DEFAULT_GYRO_BIAS_SIGMA_DPS * core::f32::consts::PI / 180.0;
    let accel_bias_sigma_mps2 = DEFAULT_ACCEL_BIAS_SIGMA_MPS2;

    let mut p_diag = [0.0_f32; 24];
    p_diag[0] = pos_n * pos_n;
    p_diag[1] = pos_e * pos_e;
    p_diag[2] = pos_d * pos_d;
    p_diag[3] = vel_var;
    p_diag[4] = vel_var;
    p_diag[5] = vel_var;
    p_diag[6] = att_var;
    p_diag[7] = att_var;
    p_diag[8] = att_var;
    p_diag[9] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
    p_diag[10] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
    p_diag[11] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
    p_diag[12] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
    p_diag[13] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
    p_diag[14] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
    p_diag[15] = DEFAULT_ACCEL_SCALE_SIGMA * DEFAULT_ACCEL_SCALE_SIGMA;
    p_diag[16] = DEFAULT_ACCEL_SCALE_SIGMA * DEFAULT_ACCEL_SCALE_SIGMA;
    p_diag[17] = DEFAULT_ACCEL_SCALE_SIGMA * DEFAULT_ACCEL_SCALE_SIGMA;
    p_diag[18] = DEFAULT_GYRO_SCALE_SIGMA * DEFAULT_GYRO_SCALE_SIGMA;
    p_diag[19] = DEFAULT_GYRO_SCALE_SIGMA * DEFAULT_GYRO_SCALE_SIGMA;
    p_diag[20] = DEFAULT_GYRO_SCALE_SIGMA * DEFAULT_GYRO_SCALE_SIGMA;
    p_diag[21] = att_var;
    p_diag[22] = att_var;
    p_diag[23] = att_var;
    p_diag
}

fn vehicle_measurements_from_mount(
    q_vb: Option<[f32; 4]>,
    raw_gyro_radps: [f64; 3],
    raw_accel_mps2: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let Some(q_vb) = q_vb else {
        return (
            [
                raw_gyro_radps[0].to_degrees(),
                raw_gyro_radps[1].to_degrees(),
                raw_gyro_radps[2].to_degrees(),
            ],
            raw_accel_mps2,
        );
    };
    let c_bv = transpose3(quat_to_rotmat_f64([
        q_vb[0] as f64,
        q_vb[1] as f64,
        q_vb[2] as f64,
        q_vb[3] as f64,
    ]));
    let gyro_vehicle_radps = mat_vec(c_bv, raw_gyro_radps);
    let accel_vehicle_mps2 = mat_vec(c_bv, raw_accel_mps2);
    (
        [
            gyro_vehicle_radps[0].to_degrees(),
            gyro_vehicle_radps[1].to_degrees(),
            gyro_vehicle_radps[2].to_degrees(),
        ],
        accel_vehicle_mps2,
    )
}

#[allow(clippy::too_many_arguments)]
fn append_eskf_sample(
    eskf: &EskfState,
    t_imu: f64,
    gyro: [f64; 3],
    accel: [f64; 3],
    dt_safe: f64,
    predict_mount_q_vb: Option<[f32; 4]>,
    vehicle_meas_lpf_cutoff_hz: f64,
    filt_meas_gyro: &mut Option<[f64; 3]>,
    filt_meas_accel: &mut Option<[f64; 3]>,
    cmp_pos_n: &mut Vec<[f64; 2]>,
    cmp_pos_e: &mut Vec<[f64; 2]>,
    cmp_pos_d: &mut Vec<[f64; 2]>,
    cmp_vel_forward: &mut Vec<[f64; 2]>,
    cmp_vel_lateral: &mut Vec<[f64; 2]>,
    cmp_vel_vertical: &mut Vec<[f64; 2]>,
    cmp_att_roll: &mut Vec<[f64; 2]>,
    cmp_att_pitch: &mut Vec<[f64; 2]>,
    cmp_att_yaw: &mut Vec<[f64; 2]>,
    mount_roll: &mut Vec<[f64; 2]>,
    mount_pitch: &mut Vec<[f64; 2]>,
    mount_yaw: &mut Vec<[f64; 2]>,
    meas_gyro_x: &mut Vec<[f64; 2]>,
    meas_gyro_y: &mut Vec<[f64; 2]>,
    meas_gyro_z: &mut Vec<[f64; 2]>,
    meas_accel_x: &mut Vec<[f64; 2]>,
    meas_accel_y: &mut Vec<[f64; 2]>,
    meas_accel_z: &mut Vec<[f64; 2]>,
    bias_gyro_x: &mut Vec<[f64; 2]>,
    bias_gyro_y: &mut Vec<[f64; 2]>,
    bias_gyro_z: &mut Vec<[f64; 2]>,
    bias_accel_x: &mut Vec<[f64; 2]>,
    bias_accel_y: &mut Vec<[f64; 2]>,
    bias_accel_z: &mut Vec<[f64; 2]>,
    cov_diag: &mut [Vec<[f64; 2]>; 18],
    yaw_cue_sum: &mut [Vec<[f64; 2]>; 11],
    yaw_cue_abs: &mut [Vec<[f64; 2]>; 11],
    yaw_cue_innov_sum: &mut [Vec<[f64; 2]>; 11],
    yaw_cue_innov_abs: &mut [Vec<[f64; 2]>; 11],
) {
    let n = &eskf.nominal;
    cmp_pos_n.push([t_imu, n.pn as f64]);
    cmp_pos_e.push([t_imu, n.pe as f64]);
    cmp_pos_d.push([t_imu, n.pd as f64]);
    let q_n_s = [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64];
    let q_cs = [n.qcs0 as f64, n.qcs1 as f64, n.qcs2 as f64, n.qcs3 as f64];
    let q_n_c = quat_mul(q_n_s, quat_conj(q_cs));
    let c_n_b = quat_to_rotmat_f64(q_n_s);
    let c_n_c = quat_to_rotmat_f64(q_n_c);
    let vel_vehicle = mat_vec(transpose3(c_n_c), [n.vn as f64, n.ve as f64, n.vd as f64]);
    cmp_vel_forward.push([t_imu, vel_vehicle[0]]);
    cmp_vel_lateral.push([t_imu, vel_vehicle[1]]);
    cmp_vel_vertical.push([t_imu, vel_vehicle[2]]);
    let (roll, pitch, yaw) = quat_rpy_deg(
        q_n_c[0] as f32,
        q_n_c[1] as f32,
        q_n_c[2] as f32,
        q_n_c[3] as f32,
    );
    cmp_att_roll.push([t_imu, roll]);
    cmp_att_pitch.push([t_imu, pitch]);
    cmp_att_yaw.push([t_imu, yaw]);

    let q_seed = predict_mount_q_vb
        .map(|q| [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64])
        .unwrap_or([1.0, 0.0, 0.0, 0.0]);
    // Runtime prediction pre-rotates raw IMU by q_seed before the ESKF sees it.
    // q_cs maps that seed frame back to vehicle for NHC, so the physical
    // vehicle-to-body mount is q_seed * inv(q_cs).
    let q_total_vb = quat_mul(q_seed, quat_conj(q_cs));
    let q_total_flu = frd_mount_quat_to_esf_alg_flu_quat(q_total_vb);
    let (mount_r, mount_p, mount_y) = quat_rpy_alg_deg(
        q_total_flu[0],
        q_total_flu[1],
        q_total_flu[2],
        q_total_flu[3],
    );
    let mount_r_plot = if eskf.p[15][15].abs() <= 1.0e-12 {
        let q_seed_flu = frd_mount_quat_to_esf_alg_flu_quat(q_seed);
        quat_rpy_alg_deg(q_seed_flu[0], q_seed_flu[1], q_seed_flu[2], q_seed_flu[3]).0
    } else {
        mount_r
    };
    mount_roll.push([t_imu, mount_r_plot]);
    mount_pitch.push([t_imu, mount_p]);
    mount_yaw.push([t_imu, mount_y]);

    let gravity_b = [
        c_n_b[2][0] * GRAVITY_MPS2 as f64,
        c_n_b[2][1] * GRAVITY_MPS2 as f64,
        c_n_b[2][2] * GRAVITY_MPS2 as f64,
    ];
    let raw_meas_gyro = [
        gyro[0] - rad2deg(n.bgx as f64),
        gyro[1] - rad2deg(n.bgy as f64),
        gyro[2] - rad2deg(n.bgz as f64),
    ];
    let raw_meas_accel = [
        accel[0] - n.bax as f64 + gravity_b[0],
        accel[1] - n.bay as f64 + gravity_b[1],
        accel[2] - n.baz as f64 + gravity_b[2],
    ];
    let alpha_meas = lpf_alpha(dt_safe, vehicle_meas_lpf_cutoff_hz);
    let filt_gyro = lpf_vec3(filt_meas_gyro, raw_meas_gyro, alpha_meas);
    let filt_accel = lpf_vec3(filt_meas_accel, raw_meas_accel, alpha_meas);
    meas_gyro_x.push([t_imu, filt_gyro[0]]);
    meas_gyro_y.push([t_imu, filt_gyro[1]]);
    meas_gyro_z.push([t_imu, filt_gyro[2]]);
    meas_accel_x.push([t_imu, filt_accel[0]]);
    meas_accel_y.push([t_imu, filt_accel[1]]);
    meas_accel_z.push([t_imu, filt_accel[2]]);
    // Loose stores additive correction terms (`corrected = scale * raw + bias`),
    // while the ESKF plots show subtractive sensor-bias estimates. Negate here so
    // both panels represent the same physical bias convention.
    bias_gyro_x.push([t_imu, -rad2deg(n.bgx as f64)]);
    bias_gyro_y.push([t_imu, -rad2deg(n.bgy as f64)]);
    bias_gyro_z.push([t_imu, -rad2deg(n.bgz as f64)]);
    bias_accel_x.push([t_imu, -(n.bax as f64)]);
    bias_accel_y.push([t_imu, -(n.bay as f64)]);
    bias_accel_z.push([t_imu, -(n.baz as f64)]);
    let p = &eskf.p;
    for (i, tr) in cov_diag.iter_mut().enumerate() {
        tr.push([t_imu, p[i][i] as f64]);
    }
    for i in 0..11 {
        yaw_cue_sum[i].push([t_imu, rad2deg(eskf.update_diag.sum_dx_mount_yaw[i] as f64)]);
        yaw_cue_abs[i].push([
            t_imu,
            rad2deg(eskf.update_diag.sum_abs_dx_mount_yaw[i] as f64),
        ]);
        yaw_cue_innov_sum[i].push([t_imu, eskf.update_diag.sum_innovation[i] as f64]);
        yaw_cue_innov_abs[i].push([t_imu, eskf.update_diag.sum_abs_innovation[i] as f64]);
    }
}

#[allow(clippy::too_many_arguments)]
fn append_loose_sample(
    loose: &LooseFilter,
    t_imu: f64,
    gyro: [f64; 3],
    accel: [f64; 3],
    dt_safe: f64,
    seed_mount_q_vb: Option<[f32; 4]>,
    ref_ecef: [f64; 3],
    ref_lat: f64,
    ref_lon: f64,
    vehicle_meas_lpf_cutoff_hz: f64,
    cmp_pos_n: &mut Vec<[f64; 2]>,
    cmp_pos_e: &mut Vec<[f64; 2]>,
    cmp_pos_d: &mut Vec<[f64; 2]>,
    cmp_vel_forward: &mut Vec<[f64; 2]>,
    cmp_vel_lateral: &mut Vec<[f64; 2]>,
    cmp_vel_vertical: &mut Vec<[f64; 2]>,
    cmp_att_roll: &mut Vec<[f64; 2]>,
    cmp_att_pitch: &mut Vec<[f64; 2]>,
    cmp_att_yaw: &mut Vec<[f64; 2]>,
    mount_roll: &mut Vec<[f64; 2]>,
    mount_pitch: &mut Vec<[f64; 2]>,
    mount_yaw: &mut Vec<[f64; 2]>,
    meas_gyro_x: &mut Vec<[f64; 2]>,
    meas_gyro_y: &mut Vec<[f64; 2]>,
    meas_gyro_z: &mut Vec<[f64; 2]>,
    meas_accel_x: &mut Vec<[f64; 2]>,
    meas_accel_y: &mut Vec<[f64; 2]>,
    meas_accel_z: &mut Vec<[f64; 2]>,
    bias_gyro_x: &mut Vec<[f64; 2]>,
    bias_gyro_y: &mut Vec<[f64; 2]>,
    bias_gyro_z: &mut Vec<[f64; 2]>,
    bias_accel_x: &mut Vec<[f64; 2]>,
    bias_accel_y: &mut Vec<[f64; 2]>,
    bias_accel_z: &mut Vec<[f64; 2]>,
    scale_gyro_x: &mut Vec<[f64; 2]>,
    scale_gyro_y: &mut Vec<[f64; 2]>,
    scale_gyro_z: &mut Vec<[f64; 2]>,
    scale_accel_x: &mut Vec<[f64; 2]>,
    scale_accel_y: &mut Vec<[f64; 2]>,
    scale_accel_z: &mut Vec<[f64; 2]>,
    cov_diag: &mut [Vec<[f64; 2]>; 24],
) {
    let n = loose.nominal();
    let (pos_ned, vel_ned, q_ns) = loose_display_state(loose, ref_ecef, ref_lat, ref_lon);
    cmp_pos_n.push([t_imu, pos_ned[0]]);
    cmp_pos_e.push([t_imu, pos_ned[1]]);
    cmp_pos_d.push([t_imu, pos_ned[2]]);
    let q_cs = [n.qcs0 as f64, n.qcs1 as f64, n.qcs2 as f64, n.qcs3 as f64];
    let q_nc = quat_mul(q_ns, quat_conj(q_cs));
    let c_n_c = quat_to_rotmat_f64(q_nc);
    let vel_vehicle = mat_vec(transpose3(c_n_c), vel_ned);
    cmp_vel_forward.push([t_imu, vel_vehicle[0]]);
    cmp_vel_lateral.push([t_imu, vel_vehicle[1]]);
    cmp_vel_vertical.push([t_imu, vel_vehicle[2]]);
    let (roll, pitch, yaw) = quat_rpy_deg(
        q_nc[0] as f32,
        q_nc[1] as f32,
        q_nc[2] as f32,
        q_nc[3] as f32,
    );
    cmp_att_roll.push([t_imu, roll]);
    cmp_att_pitch.push([t_imu, pitch]);
    cmp_att_yaw.push([t_imu, yaw]);
    let q_seed = seed_mount_q_vb
        .map(|q| [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64])
        .unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let q_total_vb = quat_mul(q_cs, q_seed);
    let q_total_flu = frd_mount_quat_to_esf_alg_flu_quat(q_total_vb);
    let (mount_r, mount_p, mount_y) = quat_rpy_alg_deg(
        q_total_flu[0],
        q_total_flu[1],
        q_total_flu[2],
        q_total_flu[3],
    );
    mount_roll.push([t_imu, mount_r]);
    mount_pitch.push([t_imu, mount_p]);
    mount_yaw.push([t_imu, mount_y]);

    let gravity_v = [
        c_n_c[2][0] * GRAVITY_MPS2 as f64,
        c_n_c[2][1] * GRAVITY_MPS2 as f64,
        c_n_c[2][2] * GRAVITY_MPS2 as f64,
    ];
    // The visualizer feeds loose with IMU samples already converted into the vehicle frame.
    // Mirror the runtime correction convention here: corrected = scale * raw + bias.
    let raw_meas_gyro = [
        rad2deg(n.sgx as f64 * deg2rad(gyro[0]) + n.bgx as f64),
        rad2deg(n.sgy as f64 * deg2rad(gyro[1]) + n.bgy as f64),
        rad2deg(n.sgz as f64 * deg2rad(gyro[2]) + n.bgz as f64),
    ];
    let raw_meas_accel = [
        n.sax as f64 * accel[0] + n.bax as f64 + gravity_v[0],
        n.say as f64 * accel[1] + n.bay as f64 + gravity_v[1],
        n.saz as f64 * accel[2] + n.baz as f64 + gravity_v[2],
    ];
    let alpha_meas = lpf_alpha(dt_safe, vehicle_meas_lpf_cutoff_hz);
    let filt_gyro = [
        raw_meas_gyro[0] * alpha_meas
            + meas_gyro_x.last().map(|p| p[1]).unwrap_or(raw_meas_gyro[0]) * (1.0 - alpha_meas),
        raw_meas_gyro[1] * alpha_meas
            + meas_gyro_y.last().map(|p| p[1]).unwrap_or(raw_meas_gyro[1]) * (1.0 - alpha_meas),
        raw_meas_gyro[2] * alpha_meas
            + meas_gyro_z.last().map(|p| p[1]).unwrap_or(raw_meas_gyro[2]) * (1.0 - alpha_meas),
    ];
    let filt_accel = [
        raw_meas_accel[0] * alpha_meas
            + meas_accel_x
                .last()
                .map(|p| p[1])
                .unwrap_or(raw_meas_accel[0])
                * (1.0 - alpha_meas),
        raw_meas_accel[1] * alpha_meas
            + meas_accel_y
                .last()
                .map(|p| p[1])
                .unwrap_or(raw_meas_accel[1])
                * (1.0 - alpha_meas),
        raw_meas_accel[2] * alpha_meas
            + meas_accel_z
                .last()
                .map(|p| p[1])
                .unwrap_or(raw_meas_accel[2])
                * (1.0 - alpha_meas),
    ];
    meas_gyro_x.push([t_imu, filt_gyro[0]]);
    meas_gyro_y.push([t_imu, filt_gyro[1]]);
    meas_gyro_z.push([t_imu, filt_gyro[2]]);
    meas_accel_x.push([t_imu, filt_accel[0]]);
    meas_accel_y.push([t_imu, filt_accel[1]]);
    meas_accel_z.push([t_imu, filt_accel[2]]);
    bias_gyro_x.push([t_imu, rad2deg(n.bgx as f64)]);
    bias_gyro_y.push([t_imu, rad2deg(n.bgy as f64)]);
    bias_gyro_z.push([t_imu, rad2deg(n.bgz as f64)]);
    bias_accel_x.push([t_imu, n.bax as f64]);
    bias_accel_y.push([t_imu, n.bay as f64]);
    bias_accel_z.push([t_imu, n.baz as f64]);
    scale_gyro_x.push([t_imu, n.sgx as f64]);
    scale_gyro_y.push([t_imu, n.sgy as f64]);
    scale_gyro_z.push([t_imu, n.sgz as f64]);
    scale_accel_x.push([t_imu, n.sax as f64]);
    scale_accel_y.push([t_imu, n.say as f64]);
    scale_accel_z.push([t_imu, n.saz as f64]);
    let p = loose.covariance();
    for (i, tr) in cov_diag.iter_mut().enumerate() {
        tr.push([t_imu, p[i][i] as f64]);
    }
}

fn loose_display_state(
    loose: &LooseFilter,
    ref_ecef: [f64; 3],
    ref_lat: f64,
    ref_lon: f64,
) -> ([f64; 3], [f64; 3], [f64; 4]) {
    let n = loose.nominal();
    let pos_ecef = loose.shadow_pos_ecef();
    let vel_ecef = [n.vn as f64, n.ve as f64, n.vd as f64];
    let q_ne = quat_ecef_to_ned(ref_lat, ref_lon);
    (
        ecef_to_ned(pos_ecef, ref_ecef, ref_lat, ref_lon),
        mat_vec(ecef_to_ned_matrix(ref_lat, ref_lon), vel_ecef),
        quat_mul(q_ne, [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64]),
    )
}

fn eskf_display_lla(fusion: &SensorFusion) -> Option<(f64, f64, f64)> {
    let eskf = fusion.eskf()?;
    let anchor = fusion.anchor_lla_debug()?;
    Some(ned_to_lla_exact(
        eskf.nominal.pn as f64,
        eskf.nominal.pe as f64,
        eskf.nominal.pd as f64,
        anchor[0] as f64,
        anchor[1] as f64,
        anchor[2] as f64,
    ))
}

fn ecef_to_ned_matrix(lat_deg: f64, lon_deg: f64) -> [[f64; 3]; 3] {
    let lat = deg2rad(lat_deg);
    let lon = deg2rad(lon_deg);
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    [
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [-clat * clon, -clat * slon, -slat],
    ]
}

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn quat_ecef_to_ned(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    let lon = deg2rad(lon_deg);
    let lat = deg2rad(lat_deg);
    let half_lon = 0.5 * lon;
    let q_lon = [half_lon.cos(), 0.0, 0.0, -half_lon.sin()];
    let half_lat = 0.5 * (lat + 0.5 * std::f64::consts::PI);
    let q_lat = [half_lat.cos(), 0.0, half_lat.sin(), 0.0];
    quat_mul(q_lat, q_lon)
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

fn ubx_vehicle_velocity(nav: NavPvtObs, att: Option<NavAttEvent>) -> [f64; 3] {
    let vel_ned = [nav.vel_n_mps, nav.vel_e_mps, nav.vel_d_mps];
    let c_n_v = match att {
        Some(att) => transpose3(rot_zyx(
            deg2rad(att.heading_deg),
            deg2rad(att.pitch_deg),
            deg2rad(att.roll_deg),
        )),
        None => {
            let yaw_deg = if nav.head_veh_valid {
                nav.heading_vehicle_deg
            } else {
                normalize_heading_deg(rad2deg(nav.vel_e_mps.atan2(nav.vel_n_mps)))
            };
            transpose3(rot_zyx(deg2rad(yaw_deg), 0.0, 0.0))
        }
    };
    mat_vec(c_n_v, vel_ned)
}

fn lpf_alpha(dt_s: f64, cutoff_hz: f64) -> f64 {
    let dt_s = dt_s.max(1.0e-6);
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz.max(1.0e-6));
    (dt_s / (tau + dt_s)).clamp(0.0, 1.0)
}

fn lpf_vec3(state: &mut Option<[f64; 3]>, sample: [f64; 3], alpha: f64) -> [f64; 3] {
    let next = match *state {
        Some(prev) => [
            prev[0] + alpha * (sample[0] - prev[0]),
            prev[1] + alpha * (sample[1] - prev[1]),
            prev[2] + alpha * (sample[2] - prev[2]),
        ],
        None => sample,
    };
    *state = Some(next);
    next
}
