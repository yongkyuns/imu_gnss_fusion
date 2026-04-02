use sensor_fusion::align::GRAVITY_MPS2;
use sensor_fusion::c_api::{CEskfImuDelta, CEskfWrapper};
use sensor_fusion::ekf::{
    Ekf, GpsData, ImuSample, PredictNoise, ekf_fuse_body_vel, ekf_fuse_gps, ekf_predict,
    ekf_set_predict_noise,
};

use crate::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_alg_status, extract_esf_raw_samples,
    extract_nav_att, extract_nav_pvt_obs, extract_nav2_pvt_obs, sensor_meta,
};

use super::super::math::{
    clamp_ekf_biases, deg2rad, ecef_to_ned, lla_to_ecef, mat_vec, ned_to_lla_exact,
    normalize_heading_deg, quat_rpy_deg, rad2deg, rot_zyx, set_quat_yaw_only,
};
use super::super::model::{AlgEvent, EkfImuSource, HeadingSample, ImuPacket, NavAttEvent, Trace};
use super::align_replay::{
    BootstrapConfig as AlignBootstrapConfig, ImuReplayConfig, build_align_replay,
};
use super::tag_time::fit_tag_ms_map;
use super::timebase::MasterTimeline;

pub struct EkfCompareData {
    pub cmp_pos: Vec<Trace>,
    pub cmp_vel: Vec<Trace>,
    pub cmp_att: Vec<Trace>,
    pub meas_gyro: Vec<Trace>,
    pub meas_accel: Vec<Trace>,
    pub bias_gyro: Vec<Trace>,
    pub bias_accel: Vec<Trace>,
    pub cov_bias: Vec<Trace>,
    pub cov_nonbias: Vec<Trace>,
    pub map: Vec<Trace>,
    pub map_heading: Vec<HeadingSample>,
    pub eskf_cmp_pos: Vec<Trace>,
    pub eskf_cmp_vel: Vec<Trace>,
    pub eskf_cmp_att: Vec<Trace>,
    pub eskf_meas_gyro: Vec<Trace>,
    pub eskf_meas_accel: Vec<Trace>,
    pub eskf_bias_gyro: Vec<Trace>,
    pub eskf_bias_accel: Vec<Trace>,
    pub eskf_cov_bias: Vec<Trace>,
    pub eskf_cov_nonbias: Vec<Trace>,
    pub eskf_map: Vec<Trace>,
    pub eskf_map_heading: Vec<HeadingSample>,
}

#[derive(Clone, Copy, Debug)]
pub struct EkfCompareConfig {
    pub r_body_vel: f32,
    pub vehicle_meas_lpf_cutoff_hz: f64,
    pub predict_imu_lpf_cutoff_hz: Option<f64>,
    pub predict_imu_decimation: usize,
    pub yaw_init_speed_mps: f64,
    pub gnss_pos_r_scale: f64,
    pub gnss_vel_r_scale: f64,
    pub predict_noise: Option<PredictNoise>,
}

impl Default for EkfCompareConfig {
    fn default() -> Self {
        Self {
            r_body_vel: 1.0,
            vehicle_meas_lpf_cutoff_hz: 5.0,
            predict_imu_lpf_cutoff_hz: None,
            predict_imu_decimation: 1,
            yaw_init_speed_mps: 0.0 / 3.6,
            gnss_pos_r_scale: 1.0,
            gnss_vel_r_scale: 1.0,
            predict_noise: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GnssOutageConfig {
    pub count: usize,
    pub duration_s: f64,
    pub seed: u64,
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
            cmp_pos: Vec::new(),
            cmp_vel: Vec::new(),
            cmp_att: Vec::new(),
            meas_gyro: Vec::new(),
            meas_accel: Vec::new(),
            bias_gyro: Vec::new(),
            bias_accel: Vec::new(),
            cov_bias: Vec::new(),
            cov_nonbias: Vec::new(),
            map: Vec::new(),
            map_heading: Vec::new(),
            eskf_cmp_pos: Vec::new(),
            eskf_cmp_vel: Vec::new(),
            eskf_cmp_att: Vec::new(),
            eskf_meas_gyro: Vec::new(),
            eskf_meas_accel: Vec::new(),
            eskf_bias_gyro: Vec::new(),
            eskf_bias_accel: Vec::new(),
            eskf_cov_bias: Vec::new(),
            eskf_cov_nonbias: Vec::new(),
            eskf_map: Vec::new(),
            eskf_map_heading: Vec::new(),
        };
    }

    let rel_s = |master_ms: f64| (master_ms - tl.t0_master_ms) * 1e-3;

    let mut alg_events = Vec::<AlgEvent>::new();
    let mut alg_status_events = Vec::<(f64, u8)>::new();
    let mut nav_att_events = Vec::<NavAttEvent>::new();
    let mut nav_events_pvt = Vec::<(f64, NavPvtObs)>::new();
    let mut nav_events_nav2 = Vec::<(f64, NavPvtObs)>::new();
    for f in frames {
        if ekf_imu_source == EkfImuSource::EsfAlg {
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
            if let Some((_, status_code, _is_fine)) = extract_esf_alg_status(f)
                && let Some(t_ms) = super::super::math::nearest_master_ms(f.seq, &tl.masters)
            {
                alg_status_events.push((t_ms, status_code as u8));
            }
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
    let (nav_events, use_nav2_for_ekf) = if !nav_events_nav2.is_empty() {
        (nav_events_nav2, true)
    } else {
        eprintln!(
            "WARNING: NAV2-PVT not found; falling back to NAV-PVT downsampled to 2 Hz for EKF GNSS observations."
        );
        (nav_events_pvt, false)
    };
    let outage_windows_ms = sample_gnss_outage_windows(&nav_events, gnss_outages);

    let mut raw_seq = Vec::<u64>::new();
    let mut raw_tag = Vec::<u64>::new();
    let mut raw_dtype = Vec::<u8>::new();
    let mut raw_val = Vec::<f64>::new();
    for f in frames {
        for (tag, sw) in extract_esf_raw_samples(f) {
            let (_name, _unit, scale) = sensor_meta(sw.dtype);
            raw_seq.push(f.seq);
            raw_tag.push(tag);
            raw_dtype.push(sw.dtype);
            raw_val.push(sw.value_i24 as f64 * scale);
        }
    }
    let (raw_tag_u, a_raw, b_raw) = fit_tag_ms_map(&raw_seq, &raw_tag, &tl.masters, Some(1 << 16));

    let mut imu_packets = Vec::<ImuPacket>::new();
    let mut current_tag: Option<u64> = None;
    let mut t_ms = 0.0_f64;
    let mut gx: Option<f64> = None;
    let mut gy: Option<f64> = None;
    let mut gz: Option<f64> = None;
    let mut ax: Option<f64> = None;
    let mut ay: Option<f64> = None;
    let mut az: Option<f64> = None;
    for (((seq, tag_u), dtype), val) in raw_seq
        .iter()
        .zip(raw_tag_u.iter())
        .zip(raw_dtype.iter())
        .zip(raw_val.iter())
    {
        if current_tag != Some(*tag_u) {
            if let (Some(gxv), Some(gyv), Some(gzv), Some(axv), Some(ayv), Some(azv)) =
                (gx, gy, gz, ax, ay, az)
            {
                imu_packets.push(ImuPacket {
                    t_ms,
                    gx_dps: gxv,
                    gy_dps: gyv,
                    gz_dps: gzv,
                    ax_mps2: axv,
                    ay_mps2: ayv,
                    az_mps2: azv,
                });
            }
            gx = None;
            gy = None;
            gz = None;
            ax = None;
            ay = None;
            az = None;
            current_tag = Some(*tag_u);
            if let Some(mapped_ms) = tl.map_tag_ms(a_raw, b_raw, *tag_u as f64, *seq) {
                t_ms = mapped_ms;
            }
        }
        match *dtype {
            14 => gx = Some(*val),
            13 => gy = Some(*val),
            5 => gz = Some(*val),
            16 => ax = Some(*val),
            17 => ay = Some(*val),
            18 => az = Some(*val),
            _ => {}
        }
    }
    if let (Some(gxv), Some(gyv), Some(gzv), Some(axv), Some(ayv), Some(azv)) =
        (gx, gy, gz, ax, ay, az)
    {
        imu_packets.push(ImuPacket {
            t_ms,
            gx_dps: gxv,
            gy_dps: gyv,
            gz_dps: gzv,
            ax_mps2: axv,
            ay_mps2: ayv,
            az_mps2: azv,
        });
    }
    imu_packets.sort_by(|a, b| {
        a.t_ms
            .partial_cmp(&b.t_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let align_events = if ekf_imu_source == EkfImuSource::Align {
        build_align_mount_events(frames, tl, ImuReplayConfig::default())
    } else {
        Vec::new()
    };

    let mut cmp_pos_n = Vec::<[f64; 2]>::new();
    let mut cmp_pos_e = Vec::<[f64; 2]>::new();
    let mut cmp_pos_d = Vec::<[f64; 2]>::new();
    let mut ubx_pos_n = Vec::<[f64; 2]>::new();
    let mut ubx_pos_e = Vec::<[f64; 2]>::new();
    let mut ubx_pos_d = Vec::<[f64; 2]>::new();

    let mut cmp_vel_n = Vec::<[f64; 2]>::new();
    let mut cmp_vel_e = Vec::<[f64; 2]>::new();
    let mut cmp_vel_d = Vec::<[f64; 2]>::new();
    let mut ubx_vel_n = Vec::<[f64; 2]>::new();
    let mut ubx_vel_e = Vec::<[f64; 2]>::new();
    let mut ubx_vel_d = Vec::<[f64; 2]>::new();

    let mut cmp_att_roll = Vec::<[f64; 2]>::new();
    let mut cmp_att_pitch = Vec::<[f64; 2]>::new();
    let mut cmp_att_yaw = Vec::<[f64; 2]>::new();
    let mut meas_gyro_x = Vec::<[f64; 2]>::new();
    let mut meas_gyro_y = Vec::<[f64; 2]>::new();
    let mut meas_gyro_z = Vec::<[f64; 2]>::new();
    let mut meas_accel_x = Vec::<[f64; 2]>::new();
    let mut meas_accel_y = Vec::<[f64; 2]>::new();
    let mut meas_accel_z = Vec::<[f64; 2]>::new();
    let mut ubx_att_roll = Vec::<[f64; 2]>::new();
    let mut ubx_att_pitch = Vec::<[f64; 2]>::new();
    let mut ubx_att_yaw = Vec::<[f64; 2]>::new();
    let mut bias_gyro_x = Vec::<[f64; 2]>::new();
    let mut bias_gyro_y = Vec::<[f64; 2]>::new();
    let mut bias_gyro_z = Vec::<[f64; 2]>::new();
    let mut bias_accel_x = Vec::<[f64; 2]>::new();
    let mut bias_accel_y = Vec::<[f64; 2]>::new();
    let mut bias_accel_z = Vec::<[f64; 2]>::new();
    let mut cov_diag: [Vec<[f64; 2]>; 16] = std::array::from_fn(|_| Vec::new());
    let mut map_ubx = Vec::<[f64; 2]>::new();
    let mut map_nav2 = Vec::<[f64; 2]>::new();
    let mut map_ekf = Vec::<[f64; 2]>::new();
    let mut map_ekf_outage = Vec::<[f64; 2]>::new();
    let mut map_heading = Vec::<HeadingSample>::new();
    let mut eskf_cmp_pos_n = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_pos_e = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_pos_d = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_vel_n = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_vel_e = Vec::<[f64; 2]>::new();
    let mut eskf_cmp_vel_d = Vec::<[f64; 2]>::new();
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
    let mut eskf_cov_diag: [Vec<[f64; 2]>; 15] = std::array::from_fn(|_| Vec::new());
    let mut map_eskf = Vec::<[f64; 2]>::new();
    let mut map_eskf_outage = Vec::<[f64; 2]>::new();
    let mut map_eskf_heading = Vec::<HeadingSample>::new();

    let mut ekf = Ekf::default();
    let mut eskf: Option<CEskfWrapper> = None;
    let base_predict_noise = cfg.predict_noise.unwrap_or(ekf.noise);
    ekf_set_predict_noise(&mut ekf, base_predict_noise);
    let mut prev_imu_t: Option<f64> = None;
    let mut alg_idx = 0usize;
    let mut alg_status_idx = 0usize;
    let mut align_idx = 0usize;
    let mut nav_idx = 0usize;
    let mut cur_alg: Option<AlgEvent> = None;
    let mut cur_alg_status: u8 = 0;
    let mut cur_align_q_vb: Option<[f32; 4]> = None;

    let mut origin_set = false;
    let mut ref_lat = 0.0_f64;
    let mut ref_lon = 0.0_f64;
    let mut ref_ecef = [0.0_f64; 3];
    let mut ref_h = 0.0_f64;
    let mut filt_meas_gyro: Option<[f64; 3]> = None;
    let mut filt_meas_accel: Option<[f64; 3]> = None;
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

        for (t_ms, nav) in &nav_events {
            let ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
            let ned = ecef_to_ned(ecef, ref_ecef, ref_lat, ref_lon);
            let t = rel_s(*t_ms);
            ubx_pos_n.push([t, ned[0]]);
            ubx_pos_e.push([t, ned[1]]);
            ubx_pos_d.push([t, ned[2]]);
            ubx_vel_n.push([t, nav.vel_n_mps]);
            ubx_vel_e.push([t, nav.vel_e_mps]);
            ubx_vel_d.push([t, nav.vel_d_mps]);
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
    for (_t_ms, nav2) in &nav2_events_for_map {
        map_nav2.push([nav2.lon_deg, nav2.lat_deg]);
    }

    let mut next_gps_update_ms = f64::NEG_INFINITY;
    let gps_period_ms = 500.0_f64;
    let mut yaw_initialized_from_vel = false;
    let mut ekf_initialized = false;
    for pkt in &imu_packets {
        while alg_idx < alg_events.len() && alg_events[alg_idx].t_ms <= pkt.t_ms {
            cur_alg = Some(alg_events[alg_idx]);
            alg_idx += 1;
        }
        while alg_status_idx < alg_status_events.len()
            && alg_status_events[alg_status_idx].0 <= pkt.t_ms
        {
            cur_alg_status = alg_status_events[alg_status_idx].1;
            alg_status_idx += 1;
        }
        while align_idx < align_events.len() && align_events[align_idx].0 <= pkt.t_ms {
            cur_align_q_vb = Some(align_events[align_idx].1);
            align_idx += 1;
        }
        let dt = match prev_imu_t {
            Some(prev) => (pkt.t_ms - prev) * 1e-3,
            None => {
                prev_imu_t = Some(pkt.t_ms);
                continue;
            }
        };
        prev_imu_t = Some(pkt.t_ms);
        if !(0.001..=0.05).contains(&dt) {
            continue;
        }
        let imu_ready = match ekf_imu_source {
            EkfImuSource::Align => cur_align_q_vb.is_some(),
            EkfImuSource::EsfAlg => cur_alg_status >= 3 && cur_alg.is_some(),
        };
        if !imu_ready {
            while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
                nav_idx += 1;
            }
            continue;
        }
        let outage_active = in_gnss_outage(pkt.t_ms, &outage_windows_ms);

        let mut gyro = [pkt.gx_dps, pkt.gy_dps, pkt.gz_dps];
        let mut accel = [pkt.ax_mps2, pkt.ay_mps2, pkt.az_mps2];
        match ekf_imu_source {
            EkfImuSource::Align => {
                if let Some(q_vb) = cur_align_q_vb {
                    let c_bv = transpose3(quat_to_rotmat_f64([
                        q_vb[0] as f64,
                        q_vb[1] as f64,
                        q_vb[2] as f64,
                        q_vb[3] as f64,
                    ]));
                    gyro = mat_vec(c_bv, gyro);
                    accel = mat_vec(c_bv, accel);
                }
            }
            EkfImuSource::EsfAlg => {
                if let Some(alg) = cur_alg {
                    let r_sb = rot_zyx(
                        deg2rad(alg.yaw_deg),
                        deg2rad(alg.pitch_deg),
                        deg2rad(alg.roll_deg),
                    );
                    gyro = mat_vec(r_sb, gyro);
                    accel = mat_vec(r_sb, accel);
                    gyro[1] = -gyro[1];
                    gyro[2] = -gyro[2];
                    accel[1] = -accel[1];
                    accel[2] = -accel[2];
                }
            }
        }
        if !ekf_initialized {
            let mut initialized_this_pkt = false;
            while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
                let (t_ms, nav) = nav_events[nav_idx];
                nav_idx += 1;
                if in_gnss_outage(t_ms, &outage_windows_ms) {
                    continue;
                }
                if !use_nav2_for_ekf {
                    if !next_gps_update_ms.is_finite() {
                        next_gps_update_ms = t_ms;
                    }
                    if t_ms + 1e-6 < next_gps_update_ms {
                        continue;
                    }
                    next_gps_update_ms += gps_period_ms;
                }

                if !origin_set {
                    ref_lat = nav.lat_deg;
                    ref_lon = nav.lon_deg;
                    ref_h = nav.height_m;
                    ref_ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
                    origin_set = true;
                }
                let ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
                let ned = ecef_to_ned(ecef, ref_ecef, ref_lat, ref_lon);
                initialize_ekf_from_gnss(&mut ekf, nav, ned);
                let eskf_predict_noise = cfg
                    .predict_noise
                    .unwrap_or(PredictNoise::lsm6dso_typical_104hz());
                eskf = Some(initialize_eskf_from_nav(nav, ned, eskf_predict_noise, cfg));
                ekf_initialized = true;
                yaw_initialized_from_vel = true;
                append_ekf_sample(
                    &ekf,
                    t_ms,
                    rel_s,
                    ref_lat,
                    ref_lon,
                    ref_h,
                    &mut cmp_pos_n,
                    &mut cmp_pos_e,
                    &mut cmp_pos_d,
                    &mut cmp_vel_n,
                    &mut cmp_vel_e,
                    &mut cmp_vel_d,
                    &mut cmp_att_roll,
                    &mut cmp_att_pitch,
                    &mut cmp_att_yaw,
                    &mut cov_diag,
                    &mut map_ekf,
                    &mut map_heading,
                );
                initialized_this_pkt = true;
                break;
            }
            if initialized_this_pkt {
                continue;
            }
        }

        if !ekf_initialized {
            while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
                nav_idx += 1;
            }
            continue;
        }

        let predict_gyro = if let Some(cutoff_hz) = cfg.predict_imu_lpf_cutoff_hz {
            let alpha_pred = lpf_alpha(dt, cutoff_hz);
            lpf_vec3(&mut filt_predict_gyro, gyro, alpha_pred)
        } else {
            gyro
        };
        let predict_accel = if let Some(cutoff_hz) = cfg.predict_imu_lpf_cutoff_hz {
            let alpha_pred = lpf_alpha(dt, cutoff_hz);
            lpf_vec3(&mut filt_predict_accel, accel, alpha_pred)
        } else {
            accel
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
        if predict_decim_count >= predict_decimation {
            let pred_dt = predict_dt_accum.max(1.0e-6);
            let block_len = predict_decim_count.max(1) as f32;
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
            let block_len_sq = block_len * block_len;
            let scaled_predict_noise = PredictNoise {
                gyro_var: base_predict_noise.gyro_var / block_len_sq,
                accel_var: base_predict_noise.accel_var / block_len_sq,
                gyro_bias_rw_var: base_predict_noise.gyro_bias_rw_var / block_len_sq,
                accel_bias_rw_var: base_predict_noise.accel_bias_rw_var / block_len_sq,
            };
            ekf_set_predict_noise(&mut ekf, scaled_predict_noise);
            let imu = ImuSample {
                dax: (deg2rad(avg_predict_gyro[0]) * pred_dt) as f32,
                day: (deg2rad(avg_predict_gyro[1]) * pred_dt) as f32,
                daz: (deg2rad(avg_predict_gyro[2]) * pred_dt) as f32,
                dvx: (avg_predict_accel[0] * pred_dt) as f32,
                dvy: (avg_predict_accel[1] * pred_dt) as f32,
                dvz: (avg_predict_accel[2] * pred_dt) as f32,
                dt: pred_dt as f32,
            };
            ekf_predict(&mut ekf, &imu, None);
            clamp_ekf_biases(&mut ekf, pred_dt);
            if let Some(eskf_ref) = eskf.as_mut() {
                eskf_ref.predict(CEskfImuDelta {
                    dax: imu.dax,
                    day: imu.day,
                    daz: imu.daz,
                    dvx: imu.dvx,
                    dvy: imu.dvy,
                    dvz: imu.dvz,
                    dt: imu.dt,
                });
            }

            ekf_fuse_body_vel(&mut ekf, cfg.r_body_vel / block_len_sq.max(1.0));
            clamp_ekf_biases(&mut ekf, pred_dt);
            if let Some(eskf_ref) = eskf.as_mut() {
                eskf_ref.fuse_body_vel(cfg.r_body_vel / block_len_sq.max(1.0));
                let n = eskf_ref.nominal();
                let c_n_b = quat_to_rotmat_f64([n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64]);
                let gravity_b = [
                    c_n_b[2][0] * GRAVITY_MPS2 as f64,
                    c_n_b[2][1] * GRAVITY_MPS2 as f64,
                    c_n_b[2][2] * GRAVITY_MPS2 as f64,
                ];
                let accel_resid = [
                    avg_predict_accel[0] - n.bax as f64 + gravity_b[0],
                    avg_predict_accel[1] - n.bay as f64 + gravity_b[1],
                    avg_predict_accel[2] - n.baz as f64 + gravity_b[2],
                ];
                let accel_h =
                    (accel_resid[0] * accel_resid[0] + accel_resid[1] * accel_resid[1]).sqrt();
                let gyro_resid = [
                    deg2rad(avg_predict_gyro[0]) - n.bgx as f64,
                    deg2rad(avg_predict_gyro[1]) - n.bgy as f64,
                    deg2rad(avg_predict_gyro[2]) - n.bgz as f64,
                ];
                let gyro_norm = (gyro_resid[0] * gyro_resid[0]
                    + gyro_resid[1] * gyro_resid[1]
                    + gyro_resid[2] * gyro_resid[2])
                    .sqrt();
                let speed_h = (n.vn as f64).hypot(n.ve as f64);
                if speed_h < 0.35 && gyro_norm < deg2rad(0.25) && accel_h < 0.08 {
                    eskf_ref.fuse_zero_vel(0.01);
                }
            }
            ekf_set_predict_noise(&mut ekf, base_predict_noise);

            while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
                let (t_ms, nav) = nav_events[nav_idx];
                nav_idx += 1;
                if in_gnss_outage(t_ms, &outage_windows_ms) {
                    continue;
                }
                if !use_nav2_for_ekf {
                    if !next_gps_update_ms.is_finite() {
                        next_gps_update_ms = t_ms;
                    }
                    if t_ms + 1e-6 < next_gps_update_ms {
                        continue;
                    }
                    next_gps_update_ms += gps_period_ms;
                }

                if !origin_set {
                    ref_lat = nav.lat_deg;
                    ref_lon = nav.lon_deg;
                    ref_h = nav.height_m;
                    ref_ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
                    origin_set = true;
                }
                let ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
                let ned = ecef_to_ned(ecef, ref_ecef, ref_lat, ref_lon);
                let speed_h = nav.vel_n_mps.hypot(nav.vel_e_mps);
                if !yaw_initialized_from_vel && speed_h >= cfg.yaw_init_speed_mps {
                    let yaw_from_vel = nav.vel_e_mps.atan2(nav.vel_n_mps);
                    set_quat_yaw_only(&mut ekf.state, yaw_from_vel);
                    yaw_initialized_from_vel = true;
                }
                let h_acc2 = (nav.h_acc_m * nav.h_acc_m).max(0.05) * cfg.gnss_pos_r_scale;
                let v_acc2 = (nav.v_acc_m * nav.v_acc_m).max(0.05) * cfg.gnss_pos_r_scale;
                let s_acc2 = (nav.s_acc_mps * nav.s_acc_mps).max(0.02) * cfg.gnss_vel_r_scale;
                let gps = GpsData {
                    pos_n: ned[0] as f32,
                    pos_e: ned[1] as f32,
                    pos_d: ned[2] as f32,
                    vel_n: nav.vel_n_mps as f32,
                    vel_e: nav.vel_e_mps as f32,
                    vel_d: nav.vel_d_mps as f32,
                    R_POS_N: h_acc2 as f32,
                    R_POS_E: h_acc2 as f32,
                    R_POS_D: v_acc2 as f32,
                    R_VEL_N: s_acc2 as f32,
                    R_VEL_E: s_acc2 as f32,
                    R_VEL_D: s_acc2 as f32,
                };
                ekf_fuse_gps(&mut ekf, &gps);
                clamp_ekf_biases(&mut ekf, pred_dt);
                if let Some(eskf_ref) = eskf.as_mut() {
                    eskf_ref.fuse_gps(fusion_gnss_sample(nav, ned, cfg));
                }

                let t = rel_s(t_ms);
                let (_, _, ekf_yaw) =
                    quat_rpy_deg(ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3);
                let (ekf_lat, ekf_lon, _ekf_h) = ned_to_lla_exact(
                    ekf.state.pn as f64,
                    ekf.state.pe as f64,
                    ekf.state.pd as f64,
                    ref_lat,
                    ref_lon,
                    ref_h,
                );
                map_heading.push(HeadingSample {
                    t_s: t,
                    lon_deg: ekf_lon,
                    lat_deg: ekf_lat,
                    yaw_deg: ekf_yaw,
                });
                if let Some(eskf_ref) = eskf.as_ref() {
                    let n = eskf_ref.nominal();
                    let (_, _, eskf_yaw) = quat_rpy_deg(n.q0, n.q1, n.q2, n.q3);
                    let (eskf_lat, eskf_lon, _eskf_h) = ned_to_lla_exact(
                        n.pn as f64,
                        n.pe as f64,
                        n.pd as f64,
                        ref_lat,
                        ref_lon,
                        ref_h,
                    );
                    map_eskf_heading.push(HeadingSample {
                        t_s: t,
                        lon_deg: eskf_lon,
                        lat_deg: eskf_lat,
                        yaw_deg: eskf_yaw,
                    });
                }
            }

            predict_gyro_sum = [0.0; 3];
            predict_accel_sum = [0.0; 3];
            predict_dt_accum = 0.0;
            predict_decim_count = 0;
        }

        if origin_set {
            let (ekf_lat, ekf_lon, _ekf_h) = ned_to_lla_exact(
                ekf.state.pn as f64,
                ekf.state.pe as f64,
                ekf.state.pd as f64,
                ref_lat,
                ref_lon,
                ref_h,
            );
            map_ekf.push([ekf_lon, ekf_lat]);
            if outage_active {
                if !prev_outage_active && !map_ekf_outage.is_empty() {
                    map_ekf_outage.push([f64::NAN, f64::NAN]);
                }
                map_ekf_outage.push([ekf_lon, ekf_lat]);
            }
            if let Some(eskf_ref) = eskf.as_ref() {
                let n = eskf_ref.nominal();
                let (eskf_lat, eskf_lon, _eskf_h) = ned_to_lla_exact(
                    n.pn as f64,
                    n.pe as f64,
                    n.pd as f64,
                    ref_lat,
                    ref_lon,
                    ref_h,
                );
                map_eskf.push([eskf_lon, eskf_lat]);
                if outage_active {
                    if !prev_outage_active && !map_eskf_outage.is_empty() {
                        map_eskf_outage.push([f64::NAN, f64::NAN]);
                    }
                    map_eskf_outage.push([eskf_lon, eskf_lat]);
                }
            }
        }
        prev_outage_active = outage_active;

        let t_imu = rel_s(pkt.t_ms);
        cmp_pos_n.push([t_imu, ekf.state.pn as f64]);
        cmp_pos_e.push([t_imu, ekf.state.pe as f64]);
        cmp_pos_d.push([t_imu, ekf.state.pd as f64]);
        cmp_vel_n.push([t_imu, ekf.state.vn as f64]);
        cmp_vel_e.push([t_imu, ekf.state.ve as f64]);
        cmp_vel_d.push([t_imu, ekf.state.vd as f64]);
        let dt_safe = dt.max(1.0e-6);
        let c_n_b = quat_to_rotmat_f64([
            ekf.state.q0 as f64,
            ekf.state.q1 as f64,
            ekf.state.q2 as f64,
            ekf.state.q3 as f64,
        ]);
        let gravity_b = [
            c_n_b[2][0] * GRAVITY_MPS2 as f64,
            c_n_b[2][1] * GRAVITY_MPS2 as f64,
            c_n_b[2][2] * GRAVITY_MPS2 as f64,
        ];
        let raw_meas_gyro = [
            gyro[0] - rad2deg((ekf.state.dax_b as f64) / dt_safe),
            gyro[1] - rad2deg((ekf.state.day_b as f64) / dt_safe),
            gyro[2] - rad2deg((ekf.state.daz_b as f64) / dt_safe),
        ];
        let raw_meas_accel = [
            accel[0] - (ekf.state.dvx_b as f64) / dt_safe + gravity_b[0],
            accel[1] - (ekf.state.dvy_b as f64) / dt_safe + gravity_b[1],
            accel[2] - (ekf.state.dvz_b as f64) / dt_safe + gravity_b[2],
        ];
        let alpha_meas = lpf_alpha(dt, cfg.vehicle_meas_lpf_cutoff_hz);
        let filt_gyro = lpf_vec3(&mut filt_meas_gyro, raw_meas_gyro, alpha_meas);
        let filt_accel = lpf_vec3(&mut filt_meas_accel, raw_meas_accel, alpha_meas);
        meas_gyro_x.push([t_imu, filt_gyro[0]]);
        meas_gyro_y.push([t_imu, filt_gyro[1]]);
        meas_gyro_z.push([t_imu, filt_gyro[2]]);
        meas_accel_x.push([t_imu, filt_accel[0]]);
        meas_accel_y.push([t_imu, filt_accel[1]]);
        meas_accel_z.push([t_imu, filt_accel[2]]);
        bias_gyro_x.push([t_imu, rad2deg((ekf.state.dax_b as f64) / dt_safe)]);
        bias_gyro_y.push([t_imu, rad2deg((ekf.state.day_b as f64) / dt_safe)]);
        bias_gyro_z.push([t_imu, rad2deg((ekf.state.daz_b as f64) / dt_safe)]);
        bias_accel_x.push([t_imu, (ekf.state.dvx_b as f64) / dt_safe]);
        bias_accel_y.push([t_imu, (ekf.state.dvy_b as f64) / dt_safe]);
        bias_accel_z.push([t_imu, (ekf.state.dvz_b as f64) / dt_safe]);
        for (i, tr) in cov_diag.iter_mut().enumerate() {
            tr.push([t_imu, ekf.p[i][i] as f64]);
        }
        let (ekf_roll, ekf_pitch, ekf_yaw) =
            quat_rpy_deg(ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3);
        cmp_att_roll.push([t_imu, ekf_roll]);
        cmp_att_pitch.push([t_imu, ekf_pitch]);
        cmp_att_yaw.push([t_imu, ekf_yaw]);
        if let Some(eskf_ref) = eskf.as_ref() {
            append_eskf_sample(
                eskf_ref,
                t_imu,
                gyro,
                accel,
                dt_safe,
                cfg.vehicle_meas_lpf_cutoff_hz,
                &mut filt_eskf_meas_gyro,
                &mut filt_eskf_meas_accel,
                &mut eskf_cmp_pos_n,
                &mut eskf_cmp_pos_e,
                &mut eskf_cmp_pos_d,
                &mut eskf_cmp_vel_n,
                &mut eskf_cmp_vel_e,
                &mut eskf_cmp_vel_d,
                &mut eskf_cmp_att_roll,
                &mut eskf_cmp_att_pitch,
                &mut eskf_cmp_att_yaw,
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
            );
        }
    }

    let cmp_pos = vec![
        Trace {
            name: "EKF posN [m]".to_string(),
            points: cmp_pos_n,
        },
        Trace {
            name: "UBX posN [m]".to_string(),
            points: ubx_pos_n.clone(),
        },
        Trace {
            name: "EKF posE [m]".to_string(),
            points: cmp_pos_e,
        },
        Trace {
            name: "UBX posE [m]".to_string(),
            points: ubx_pos_e.clone(),
        },
        Trace {
            name: "EKF posD [m]".to_string(),
            points: cmp_pos_d,
        },
        Trace {
            name: "UBX posD [m]".to_string(),
            points: ubx_pos_d.clone(),
        },
    ];
    let cmp_vel = vec![
        Trace {
            name: "EKF velN [m/s]".to_string(),
            points: cmp_vel_n,
        },
        Trace {
            name: "UBX velN [m/s]".to_string(),
            points: ubx_vel_n.clone(),
        },
        Trace {
            name: "EKF velE [m/s]".to_string(),
            points: cmp_vel_e,
        },
        Trace {
            name: "UBX velE [m/s]".to_string(),
            points: ubx_vel_e.clone(),
        },
        Trace {
            name: "EKF velD [m/s]".to_string(),
            points: cmp_vel_d,
        },
        Trace {
            name: "UBX velD [m/s]".to_string(),
            points: ubx_vel_d.clone(),
        },
    ];
    let cmp_att = vec![
        Trace {
            name: "EKF roll [deg]".to_string(),
            points: cmp_att_roll,
        },
        Trace {
            name: "NAV-ATT roll [deg]".to_string(),
            points: ubx_att_roll.clone(),
        },
        Trace {
            name: "EKF pitch [deg]".to_string(),
            points: cmp_att_pitch,
        },
        Trace {
            name: "NAV-ATT pitch [deg]".to_string(),
            points: ubx_att_pitch.clone(),
        },
        Trace {
            name: "EKF yaw [deg]".to_string(),
            points: cmp_att_yaw,
        },
        Trace {
            name: "NAV-ATT heading [deg]".to_string(),
            points: ubx_att_yaw.clone(),
        },
    ];
    let meas_gyro = vec![
        Trace {
            name: "EKF vehicle gyro x [deg/s]".to_string(),
            points: meas_gyro_x,
        },
        Trace {
            name: "EKF vehicle gyro y [deg/s]".to_string(),
            points: meas_gyro_y,
        },
        Trace {
            name: "EKF vehicle gyro z [deg/s]".to_string(),
            points: meas_gyro_z,
        },
    ];
    let meas_accel = vec![
        Trace {
            name: "EKF vehicle accel x [m/s^2]".to_string(),
            points: meas_accel_x,
        },
        Trace {
            name: "EKF vehicle accel y [m/s^2]".to_string(),
            points: meas_accel_y,
        },
        Trace {
            name: "EKF vehicle accel z [m/s^2]".to_string(),
            points: meas_accel_z,
        },
    ];
    let bias_gyro = vec![
        Trace {
            name: "EKF gyro bias x [deg/s]".to_string(),
            points: bias_gyro_x,
        },
        Trace {
            name: "EKF gyro bias y [deg/s]".to_string(),
            points: bias_gyro_y,
        },
        Trace {
            name: "EKF gyro bias z [deg/s]".to_string(),
            points: bias_gyro_z,
        },
    ];
    let bias_accel = vec![
        Trace {
            name: "EKF accel bias x [m/s^2]".to_string(),
            points: bias_accel_x,
        },
        Trace {
            name: "EKF accel bias y [m/s^2]".to_string(),
            points: bias_accel_y,
        },
        Trace {
            name: "EKF accel bias z [m/s^2]".to_string(),
            points: bias_accel_z,
        },
    ];
    let cov_bias = vec![
        Trace {
            name: "acc_x".to_string(),
            points: cov_diag[13].clone(),
        },
        Trace {
            name: "acc_y".to_string(),
            points: cov_diag[14].clone(),
        },
        Trace {
            name: "acc_z".to_string(),
            points: cov_diag[15].clone(),
        },
        Trace {
            name: "gyro_x".to_string(),
            points: cov_diag[10].clone(),
        },
        Trace {
            name: "gyro_y".to_string(),
            points: cov_diag[11].clone(),
        },
        Trace {
            name: "gyro_z".to_string(),
            points: cov_diag[12].clone(),
        },
    ];
    let cov_nonbias = vec![
        Trace {
            name: "p_n".to_string(),
            points: cov_diag[7].clone(),
        },
        Trace {
            name: "p_e".to_string(),
            points: cov_diag[8].clone(),
        },
        Trace {
            name: "p_d".to_string(),
            points: cov_diag[9].clone(),
        },
        Trace {
            name: "v_n".to_string(),
            points: cov_diag[4].clone(),
        },
        Trace {
            name: "v_e".to_string(),
            points: cov_diag[5].clone(),
        },
        Trace {
            name: "v_d".to_string(),
            points: cov_diag[6].clone(),
        },
        Trace {
            name: "q1".to_string(),
            points: cov_diag[1].clone(),
        },
        Trace {
            name: "q2".to_string(),
            points: cov_diag[2].clone(),
        },
        Trace {
            name: "q3".to_string(),
            points: cov_diag[3].clone(),
        },
        Trace {
            name: "q0".to_string(),
            points: cov_diag[0].clone(),
        },
    ];
    let map = vec![
        Trace {
            name: "u-blox path (lon,lat)".to_string(),
            points: map_ubx,
        },
        Trace {
            name: "NAV2-PVT path (GNSS-only, lon,lat)".to_string(),
            points: map_nav2,
        },
        Trace {
            name: "EKF path (lon,lat)".to_string(),
            points: map_ekf,
        },
        Trace {
            name: "EKF path during GNSS outage (lon,lat)".to_string(),
            points: map_ekf_outage,
        },
    ];
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
            name: "ESKF velN [m/s]".to_string(),
            points: eskf_cmp_vel_n,
        },
        Trace {
            name: "UBX velN [m/s]".to_string(),
            points: ubx_vel_n.clone(),
        },
        Trace {
            name: "ESKF velE [m/s]".to_string(),
            points: eskf_cmp_vel_e,
        },
        Trace {
            name: "UBX velE [m/s]".to_string(),
            points: ubx_vel_e.clone(),
        },
        Trace {
            name: "ESKF velD [m/s]".to_string(),
            points: eskf_cmp_vel_d,
        },
        Trace {
            name: "UBX velD [m/s]".to_string(),
            points: ubx_vel_d.clone(),
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
    ];
    let eskf_map = vec![
        Trace {
            name: "ESKF path (lon,lat)".to_string(),
            points: map_eskf,
        },
        Trace {
            name: "ESKF path during GNSS outage (lon,lat)".to_string(),
            points: map_eskf_outage,
        },
    ];

    EkfCompareData {
        cmp_pos,
        cmp_vel,
        cmp_att,
        meas_gyro,
        meas_accel,
        bias_gyro,
        bias_accel,
        cov_bias,
        cov_nonbias,
        map,
        map_heading,
        eskf_cmp_pos,
        eskf_cmp_vel,
        eskf_cmp_att,
        eskf_meas_gyro,
        eskf_meas_accel,
        eskf_bias_gyro,
        eskf_bias_accel,
        eskf_cov_bias,
        eskf_cov_nonbias,
        eskf_map,
        eskf_map_heading: map_eskf_heading,
    }
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

fn initialize_ekf_from_gnss(ekf: &mut Ekf, nav: NavPvtObs, ned: [f64; 3]) {
    *ekf = Ekf::default();
    ekf.state.pn = ned[0] as f32;
    ekf.state.pe = ned[1] as f32;
    ekf.state.pd = ned[2] as f32;
    ekf.state.vn = nav.vel_n_mps as f32;
    ekf.state.ve = nav.vel_e_mps as f32;
    ekf.state.vd = nav.vel_d_mps as f32;

    let speed_h = nav.vel_n_mps.hypot(nav.vel_e_mps);
    let yaw_from_gnss = if nav.head_veh_valid {
        deg2rad(nav.heading_vehicle_deg)
    } else if speed_h >= 1.0 {
        nav.vel_e_mps.atan2(nav.vel_n_mps)
    } else {
        deg2rad(nav.heading_motion_deg)
    };
    set_quat_yaw_only(&mut ekf.state, yaw_from_gnss);

    // Startup should trust the seeded attitude enough that short-term NHC updates
    // cannot drag the filter into the opposite heading basin before the next GNSS fix.
    let att_sigma_rad = 2.0_f32.to_radians();
    let quat_var = 0.25 * att_sigma_rad * att_sigma_rad;
    for i in 0..4 {
        ekf.p[i][i] = quat_var;
    }

    let vel_var = (nav.s_acc_mps.max(0.2) as f32).powi(2);
    ekf.p[4][4] = vel_var;
    ekf.p[5][5] = vel_var;
    ekf.p[6][6] = vel_var;

    let pos_h_var = (nav.h_acc_m.max(0.5) as f32).powi(2);
    let pos_d_var = (nav.v_acc_m.max(0.5) as f32).powi(2);
    ekf.p[7][7] = pos_h_var;
    ekf.p[8][8] = pos_h_var;
    ekf.p[9][9] = pos_d_var;
}

fn initialize_eskf_from_nav(
    nav: NavPvtObs,
    ned: [f64; 3],
    noise: PredictNoise,
    cfg: EkfCompareConfig,
) -> CEskfWrapper {
    let mut eskf = CEskfWrapper::new(noise);
    let q_bn = yaw_quat_f32(initial_yaw_from_nav(nav));
    eskf.init_nominal_from_gnss(q_bn, fusion_gnss_sample(nav, ned, cfg));
    eskf
}

fn initial_yaw_from_nav(nav: NavPvtObs) -> f32 {
    let speed_h = nav.vel_n_mps.hypot(nav.vel_e_mps);
    if nav.head_veh_valid {
        deg2rad(nav.heading_vehicle_deg) as f32
    } else if speed_h >= 1.0 {
        nav.vel_e_mps.atan2(nav.vel_n_mps) as f32
    } else {
        deg2rad(nav.heading_motion_deg) as f32
    }
}

fn yaw_quat_f32(yaw_rad: f32) -> [f32; 4] {
    let half = 0.5 * yaw_rad;
    [half.cos(), 0.0, 0.0, half.sin()]
}

fn fusion_gnss_sample(
    nav: NavPvtObs,
    ned: [f64; 3],
    cfg: EkfCompareConfig,
) -> sensor_fusion::fusion::FusionGnssSample {
    let speed_h = nav.vel_n_mps.hypot(nav.vel_e_mps);
    let heading_rad = if nav.head_veh_valid {
        Some(deg2rad(nav.heading_vehicle_deg) as f32)
    } else if speed_h >= cfg.yaw_init_speed_mps.max(1.0) {
        Some(nav.vel_e_mps.atan2(nav.vel_n_mps) as f32)
    } else {
        Some(deg2rad(nav.heading_motion_deg) as f32)
    };
    sensor_fusion::fusion::FusionGnssSample {
        t_s: 0.0,
        pos_ned_m: [ned[0] as f32, ned[1] as f32, ned[2] as f32],
        vel_ned_mps: [
            nav.vel_n_mps as f32,
            nav.vel_e_mps as f32,
            nav.vel_d_mps as f32,
        ],
        pos_std_m: [
            (nav.h_acc_m * cfg.gnss_pos_r_scale.sqrt()) as f32,
            (nav.h_acc_m * cfg.gnss_pos_r_scale.sqrt()) as f32,
            (nav.v_acc_m * cfg.gnss_pos_r_scale.sqrt()) as f32,
        ],
        vel_std_mps: [
            (nav.s_acc_mps * cfg.gnss_vel_r_scale.sqrt()) as f32,
            (nav.s_acc_mps * cfg.gnss_vel_r_scale.sqrt()) as f32,
            (nav.s_acc_mps * cfg.gnss_vel_r_scale.sqrt()) as f32,
        ],
        heading_rad,
    }
}

#[allow(clippy::too_many_arguments)]
fn append_eskf_sample(
    eskf: &CEskfWrapper,
    t_imu: f64,
    gyro: [f64; 3],
    accel: [f64; 3],
    dt_safe: f64,
    vehicle_meas_lpf_cutoff_hz: f64,
    filt_meas_gyro: &mut Option<[f64; 3]>,
    filt_meas_accel: &mut Option<[f64; 3]>,
    cmp_pos_n: &mut Vec<[f64; 2]>,
    cmp_pos_e: &mut Vec<[f64; 2]>,
    cmp_pos_d: &mut Vec<[f64; 2]>,
    cmp_vel_n: &mut Vec<[f64; 2]>,
    cmp_vel_e: &mut Vec<[f64; 2]>,
    cmp_vel_d: &mut Vec<[f64; 2]>,
    cmp_att_roll: &mut Vec<[f64; 2]>,
    cmp_att_pitch: &mut Vec<[f64; 2]>,
    cmp_att_yaw: &mut Vec<[f64; 2]>,
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
    cov_diag: &mut [Vec<[f64; 2]>; 15],
) {
    let n = eskf.nominal();
    cmp_pos_n.push([t_imu, n.pn as f64]);
    cmp_pos_e.push([t_imu, n.pe as f64]);
    cmp_pos_d.push([t_imu, n.pd as f64]);
    cmp_vel_n.push([t_imu, n.vn as f64]);
    cmp_vel_e.push([t_imu, n.ve as f64]);
    cmp_vel_d.push([t_imu, n.vd as f64]);
    let (roll, pitch, yaw) = quat_rpy_deg(n.q0, n.q1, n.q2, n.q3);
    cmp_att_roll.push([t_imu, roll]);
    cmp_att_pitch.push([t_imu, pitch]);
    cmp_att_yaw.push([t_imu, yaw]);

    let c_n_b = quat_to_rotmat_f64([n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64]);
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
    bias_gyro_x.push([t_imu, rad2deg(n.bgx as f64)]);
    bias_gyro_y.push([t_imu, rad2deg(n.bgy as f64)]);
    bias_gyro_z.push([t_imu, rad2deg(n.bgz as f64)]);
    bias_accel_x.push([t_imu, n.bax as f64]);
    bias_accel_y.push([t_imu, n.bay as f64]);
    bias_accel_z.push([t_imu, n.baz as f64]);
    let p = eskf.covariance();
    for (i, tr) in cov_diag.iter_mut().enumerate() {
        tr.push([t_imu, p[i][i] as f64]);
    }
}

#[allow(clippy::too_many_arguments)]
fn append_ekf_sample(
    ekf: &Ekf,
    t_ms: f64,
    rel_s: impl Fn(f64) -> f64,
    ref_lat: f64,
    ref_lon: f64,
    ref_h: f64,
    cmp_pos_n: &mut Vec<[f64; 2]>,
    cmp_pos_e: &mut Vec<[f64; 2]>,
    cmp_pos_d: &mut Vec<[f64; 2]>,
    cmp_vel_n: &mut Vec<[f64; 2]>,
    cmp_vel_e: &mut Vec<[f64; 2]>,
    cmp_vel_d: &mut Vec<[f64; 2]>,
    cmp_att_roll: &mut Vec<[f64; 2]>,
    cmp_att_pitch: &mut Vec<[f64; 2]>,
    cmp_att_yaw: &mut Vec<[f64; 2]>,
    cov_diag: &mut [Vec<[f64; 2]>; 16],
    map_ekf: &mut Vec<[f64; 2]>,
    map_heading: &mut Vec<HeadingSample>,
) {
    let t = rel_s(t_ms);
    cmp_pos_n.push([t, ekf.state.pn as f64]);
    cmp_pos_e.push([t, ekf.state.pe as f64]);
    cmp_pos_d.push([t, ekf.state.pd as f64]);
    cmp_vel_n.push([t, ekf.state.vn as f64]);
    cmp_vel_e.push([t, ekf.state.ve as f64]);
    cmp_vel_d.push([t, ekf.state.vd as f64]);
    for (i, tr) in cov_diag.iter_mut().enumerate() {
        tr.push([t, ekf.p[i][i] as f64]);
    }
    let (ekf_roll, ekf_pitch, ekf_yaw) =
        quat_rpy_deg(ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3);
    cmp_att_roll.push([t, ekf_roll]);
    cmp_att_pitch.push([t, ekf_pitch]);
    cmp_att_yaw.push([t, ekf_yaw]);
    let (ekf_lat, ekf_lon, _ekf_h) = ned_to_lla_exact(
        ekf.state.pn as f64,
        ekf.state.pe as f64,
        ekf.state.pd as f64,
        ref_lat,
        ref_lon,
        ref_h,
    );
    map_ekf.push([ekf_lon, ekf_lat]);
    map_heading.push(HeadingSample {
        t_s: t,
        lon_deg: ekf_lon,
        lat_deg: ekf_lat,
        yaw_deg: ekf_yaw,
    });
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
