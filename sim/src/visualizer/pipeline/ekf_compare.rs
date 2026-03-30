use align_rs::align::{Align, AlignConfig, AlignWindowSummary, GRAVITY_MPS2};
use ekf_rs::ekf::{Ekf, GpsData, ImuSample, ekf_fuse_body_vel, ekf_fuse_gps, ekf_predict};

use crate::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_alg_status, extract_esf_raw_samples,
    extract_nav_att, extract_nav_pvt_obs, extract_nav2_pvt_obs, sensor_meta,
};

use super::super::math::{
    clamp_ekf_biases, deg2rad, ecef_to_ned, lla_to_ecef, mat_vec, ned_to_lla_approx,
    normalize_heading_deg, quat_rpy_deg, rad2deg, rot_zyx, set_quat_yaw_only,
};
use super::super::model::{AlgEvent, EkfImuSource, HeadingSample, ImuPacket, NavAttEvent, Trace};
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
}

#[derive(Clone, Copy)]
struct BootstrapConfig {
    ema_alpha: f32,
    max_speed_mps: f32,
    stationary_samples: usize,
    max_gyro_radps: f32,
    max_accel_norm_err_mps2: f32,
}

struct BootstrapDetector {
    cfg: BootstrapConfig,
    gyro_ema: Option<f32>,
    accel_err_ema: Option<f32>,
    speed_ema: Option<f32>,
    stationary_accel: Vec<[f32; 3]>,
}

pub fn build_ekf_compare_traces(
    frames: &[UbxFrame],
    tl: &MasterTimeline,
    ekf_imu_source: EkfImuSource,
) -> EkfCompareData {
    const R_BODY_VEL: f32 = 5.0;
    const YAW_INIT_SPEED_MPS: f64 = 0.0 / 3.6;
    const GNSS_POS_R_SCALE: f64 = 1.0;
    const GNSS_VEL_R_SCALE: f64 = 1.0;

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
        build_align_mount_events(&imu_packets, &nav_events)
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
    let mut map_heading = Vec::<HeadingSample>::new();

    let mut ekf = Ekf::default();
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
    let mut ekf_initialized = ekf_imu_source == EkfImuSource::EsfAlg;
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
        if ekf_imu_source == EkfImuSource::Align && !ekf_initialized {
            let mut initialized_this_pkt = false;
            while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
                let (t_ms, nav) = nav_events[nav_idx];
                nav_idx += 1;
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
                ekf_initialized = true;
                yaw_initialized_from_vel = true;
                initialized_this_pkt = true;
                break;
            }
            if initialized_this_pkt {
                continue;
            }
        }

        let imu = ImuSample {
            dax: (deg2rad(gyro[0]) * dt) as f32,
            day: (deg2rad(gyro[1]) * dt) as f32,
            daz: (deg2rad(gyro[2]) * dt) as f32,
            dvx: (accel[0] * dt) as f32,
            dvy: (accel[1] * dt) as f32,
            dvz: (accel[2] * dt) as f32,
            dt: dt as f32,
        };
        if !ekf_initialized {
            while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
                nav_idx += 1;
            }
            continue;
        }

        ekf_predict(&mut ekf, &imu, None);
        clamp_ekf_biases(&mut ekf, dt);

        ekf_fuse_body_vel(&mut ekf, R_BODY_VEL);
        clamp_ekf_biases(&mut ekf, dt);

        while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
            let (t_ms, nav) = nav_events[nav_idx];
            nav_idx += 1;
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
            if !yaw_initialized_from_vel && speed_h >= YAW_INIT_SPEED_MPS {
                let yaw_from_vel = nav.vel_e_mps.atan2(nav.vel_n_mps);
                set_quat_yaw_only(&mut ekf.state, yaw_from_vel);
                yaw_initialized_from_vel = true;
            }
            let h_acc2 = (nav.h_acc_m * nav.h_acc_m).max(0.05) * GNSS_POS_R_SCALE;
            let v_acc2 = (nav.v_acc_m * nav.v_acc_m).max(0.05) * GNSS_POS_R_SCALE;
            let s_acc2 = (nav.s_acc_mps * nav.s_acc_mps).max(0.02) * GNSS_VEL_R_SCALE;
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
            clamp_ekf_biases(&mut ekf, dt);

            let t = rel_s(t_ms);
            let (_, _, ekf_yaw) =
                quat_rpy_deg(ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3);
            let (ekf_lat, ekf_lon, _ekf_h) = ned_to_lla_approx(
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
        meas_gyro_x.push([t_imu, gyro[0] - rad2deg((ekf.state.dax_b as f64) / dt_safe)]);
        meas_gyro_y.push([t_imu, gyro[1] - rad2deg((ekf.state.day_b as f64) / dt_safe)]);
        meas_gyro_z.push([t_imu, gyro[2] - rad2deg((ekf.state.daz_b as f64) / dt_safe)]);
        meas_accel_x.push([
            t_imu,
            accel[0] - (ekf.state.dvx_b as f64) / dt_safe + gravity_b[0],
        ]);
        meas_accel_y.push([
            t_imu,
            accel[1] - (ekf.state.dvy_b as f64) / dt_safe + gravity_b[1],
        ]);
        meas_accel_z.push([
            t_imu,
            accel[2] - (ekf.state.dvz_b as f64) / dt_safe + gravity_b[2],
        ]);
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
    }

    let cmp_pos = vec![
        Trace {
            name: "EKF posN [m]".to_string(),
            points: cmp_pos_n,
        },
        Trace {
            name: "UBX posN [m]".to_string(),
            points: ubx_pos_n,
        },
        Trace {
            name: "EKF posE [m]".to_string(),
            points: cmp_pos_e,
        },
        Trace {
            name: "UBX posE [m]".to_string(),
            points: ubx_pos_e,
        },
        Trace {
            name: "EKF posD [m]".to_string(),
            points: cmp_pos_d,
        },
        Trace {
            name: "UBX posD [m]".to_string(),
            points: ubx_pos_d,
        },
    ];
    let cmp_vel = vec![
        Trace {
            name: "EKF velN [m/s]".to_string(),
            points: cmp_vel_n,
        },
        Trace {
            name: "UBX velN [m/s]".to_string(),
            points: ubx_vel_n,
        },
        Trace {
            name: "EKF velE [m/s]".to_string(),
            points: cmp_vel_e,
        },
        Trace {
            name: "UBX velE [m/s]".to_string(),
            points: ubx_vel_e,
        },
        Trace {
            name: "EKF velD [m/s]".to_string(),
            points: cmp_vel_d,
        },
        Trace {
            name: "UBX velD [m/s]".to_string(),
            points: ubx_vel_d,
        },
    ];
    let cmp_att = vec![
        Trace {
            name: "EKF roll [deg]".to_string(),
            points: cmp_att_roll,
        },
        Trace {
            name: "NAV-ATT roll [deg]".to_string(),
            points: ubx_att_roll,
        },
        Trace {
            name: "EKF pitch [deg]".to_string(),
            points: cmp_att_pitch,
        },
        Trace {
            name: "NAV-ATT pitch [deg]".to_string(),
            points: ubx_att_pitch,
        },
        Trace {
            name: "EKF yaw [deg]".to_string(),
            points: cmp_att_yaw,
        },
        Trace {
            name: "NAV-ATT heading [deg]".to_string(),
            points: ubx_att_yaw,
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
    }
}

impl BootstrapDetector {
    fn new(cfg: BootstrapConfig) -> Self {
        Self {
            cfg,
            gyro_ema: None,
            accel_err_ema: None,
            speed_ema: None,
            stationary_accel: Vec::new(),
        }
    }

    fn update(&mut self, accel_b: [f32; 3], gyro_radps: [f32; 3], speed_mps: f32) -> bool {
        let gyro_norm = norm3(gyro_radps);
        let accel_err = (norm3(accel_b) - GRAVITY_MPS2).abs();
        self.gyro_ema = Some(ema_update(self.gyro_ema, gyro_norm, self.cfg.ema_alpha));
        self.accel_err_ema = Some(ema_update(
            self.accel_err_ema,
            accel_err,
            self.cfg.ema_alpha,
        ));
        self.speed_ema = Some(ema_update(self.speed_ema, speed_mps, self.cfg.ema_alpha));

        let stationary = self.speed_ema.unwrap_or(speed_mps) <= self.cfg.max_speed_mps
            && self.gyro_ema.unwrap_or(gyro_norm) <= self.cfg.max_gyro_radps
            && self.accel_err_ema.unwrap_or(accel_err) <= self.cfg.max_accel_norm_err_mps2;

        if stationary {
            self.stationary_accel.push(accel_b);
        } else {
            self.stationary_accel.clear();
        }
        self.stationary_accel.len() >= self.cfg.stationary_samples
    }
}

fn build_align_mount_events(
    imu_packets: &[ImuPacket],
    nav_events: &[(f64, NavPvtObs)],
) -> Vec<(f64, [f32; 4])> {
    if imu_packets.is_empty() || nav_events.len() < 2 {
        return Vec::new();
    }

    let cfg = AlignConfig::default();
    let bootstrap_cfg = BootstrapConfig {
        ema_alpha: 0.05,
        max_speed_mps: 0.35,
        stationary_samples: 100,
        max_gyro_radps: cfg.max_stationary_gyro_radps,
        max_accel_norm_err_mps2: cfg.max_stationary_accel_norm_err_mps2,
    };

    let mut align = Align::new(cfg);
    let mut bootstrap = BootstrapDetector::new(bootstrap_cfg);
    let mut out = Vec::<(f64, [f32; 4])>::new();
    let mut align_initialized = false;
    let mut coarse_alignment_ready = false;
    let mut scan_idx = 0usize;
    let mut interval_start_idx = 0usize;
    let mut prev_nav: Option<(f64, NavPvtObs)> = None;

    for (tn, nav) in nav_events {
        while scan_idx < imu_packets.len() && imu_packets[scan_idx].t_ms <= *tn {
            let pkt = &imu_packets[scan_idx];
            if !align_initialized {
                let gyro_radps = [
                    pkt.gx_dps.to_radians() as f32,
                    pkt.gy_dps.to_radians() as f32,
                    pkt.gz_dps.to_radians() as f32,
                ];
                let accel_b = [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32];
                let speed_mps = speed_for_bootstrap(prev_nav, (*tn, *nav), pkt.t_ms) as f32;
                if bootstrap.update(accel_b, gyro_radps, speed_mps)
                    && align
                        .initialize_from_stationary(&bootstrap.stationary_accel, 0.0)
                        .is_ok()
                {
                    align_initialized = true;
                }
            }
            scan_idx += 1;
        }

        if let Some((t_prev, nav_prev)) = prev_nav {
            let dt = ((*tn - t_prev) * 1.0e-3) as f32;
            let interval_packets = &imu_packets[interval_start_idx..scan_idx];
            if align_initialized && dt > 0.0 && !interval_packets.is_empty() {
                let mut gyro_sum = [0.0_f32; 3];
                let mut accel_sum = [0.0_f32; 3];
                for pkt in interval_packets {
                    gyro_sum[0] += pkt.gx_dps.to_radians() as f32;
                    gyro_sum[1] += pkt.gy_dps.to_radians() as f32;
                    gyro_sum[2] += pkt.gz_dps.to_radians() as f32;
                    accel_sum[0] += pkt.ax_mps2 as f32;
                    accel_sum[1] += pkt.ay_mps2 as f32;
                    accel_sum[2] += pkt.az_mps2 as f32;
                }
                let inv_n = 1.0 / (interval_packets.len() as f32);
                let window = AlignWindowSummary {
                    dt,
                    mean_gyro_b: [
                        gyro_sum[0] * inv_n,
                        gyro_sum[1] * inv_n,
                        gyro_sum[2] * inv_n,
                    ],
                    mean_accel_b: [
                        accel_sum[0] * inv_n,
                        accel_sum[1] * inv_n,
                        accel_sum[2] * inv_n,
                    ],
                    gnss_vel_prev_n: [
                        nav_prev.vel_n_mps as f32,
                        nav_prev.vel_e_mps as f32,
                        nav_prev.vel_d_mps as f32,
                    ],
                    gnss_vel_curr_n: [
                        nav.vel_n_mps as f32,
                        nav.vel_e_mps as f32,
                        nav.vel_d_mps as f32,
                    ],
                };
                let (_, trace) = align.update_window_with_trace(&window);
                if trace.coarse_alignment_ready {
                    coarse_alignment_ready = true;
                }
                if coarse_alignment_ready {
                    out.push((*tn, align.q_vb));
                }
            }
        }

        prev_nav = Some((*tn, *nav));
        interval_start_idx = scan_idx;
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

fn ema_update(prev: Option<f32>, sample: f32, alpha: f32) -> f32 {
    let alpha = alpha.clamp(1.0e-4, 1.0);
    match prev {
        Some(prev) => (1.0 - alpha) * prev + alpha * sample,
        None => sample,
    }
}

fn speed_for_bootstrap(
    prev_nav: Option<(f64, NavPvtObs)>,
    curr_nav: (f64, NavPvtObs),
    t_ms: f64,
) -> f64 {
    let speed_curr = horizontal_speed(curr_nav.1);
    let Some((t_prev, nav_prev)) = prev_nav else {
        return speed_curr;
    };
    let speed_prev = horizontal_speed(nav_prev);
    let dt = curr_nav.0 - t_prev;
    if dt <= 1.0e-6 {
        return speed_curr;
    }
    let alpha = ((t_ms - t_prev) / dt).clamp(0.0, 1.0);
    speed_prev + alpha * (speed_curr - speed_prev)
}

fn horizontal_speed(nav: NavPvtObs) -> f64 {
    (nav.vel_n_mps * nav.vel_n_mps + nav.vel_e_mps * nav.vel_e_mps).sqrt()
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
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
