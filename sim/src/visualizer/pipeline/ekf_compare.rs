use ekf_rs::ekf::{Ekf, GpsData, ImuSample, ekf_fuse_body_vel, ekf_fuse_gps, ekf_predict};

use crate::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_alg_status, extract_esf_raw_samples,
    extract_nav_att, extract_nav_pvt_obs, extract_nav2_pvt_obs, sensor_meta,
};

use super::super::math::{
    clamp_ekf_biases, deg2rad, ecef_to_ned, lla_to_ecef, mat_vec, ned_to_lla_approx,
    normalize_heading_deg, quat_rpy_deg, rad2deg, rot_zyx, set_quat_yaw_only,
};
use super::super::model::{AlgEvent, HeadingSample, ImuPacket, NavAttEvent, Trace};
use super::tag_time::fit_tag_ms_map;
use super::timebase::MasterTimeline;

pub struct EkfCompareData {
    pub cmp_pos: Vec<Trace>,
    pub cmp_vel: Vec<Trace>,
    pub cmp_att: Vec<Trace>,
    pub bias_gyro: Vec<Trace>,
    pub bias_accel: Vec<Trace>,
    pub cov_bias: Vec<Trace>,
    pub cov_nonbias: Vec<Trace>,
    pub map: Vec<Trace>,
    pub map_heading: Vec<HeadingSample>,
}

pub fn build_ekf_compare_traces(frames: &[UbxFrame], tl: &MasterTimeline) -> EkfCompareData {
    const R_BODY_VEL: f32 = 1.0;
    const YAW_INIT_SPEED_MPS: f64 = 20.0 / 3.6;

    if tl.masters.is_empty() {
        return EkfCompareData {
            cmp_pos: Vec::new(),
            cmp_vel: Vec::new(),
            cmp_att: Vec::new(),
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
        if let Some((_, roll, pitch, yaw)) = extract_esf_alg(f) {
            if let Some(t_ms) = super::super::math::nearest_master_ms(f.seq, &tl.masters) {
                alg_events.push(AlgEvent {
                    t_ms,
                    roll_deg: roll,
                    pitch_deg: pitch,
                    yaw_deg: yaw,
                });
            }
        }
        if let Some((_, status_code, _is_fine)) = extract_esf_alg_status(f)
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
    let mut nav_idx = 0usize;
    let mut cur_alg: Option<AlgEvent> = None;
    let mut cur_alg_status: u8 = 0;

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
        if cur_alg_status < 3 {
            while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
                nav_idx += 1;
            }
            continue;
        }

        let mut gyro = [pkt.gx_dps, pkt.gy_dps, pkt.gz_dps];
        let mut accel = [pkt.ax_mps2, pkt.ay_mps2, pkt.az_mps2];
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
        let imu = ImuSample {
            dax: (deg2rad(gyro[0]) * dt) as f32,
            day: (deg2rad(gyro[1]) * dt) as f32,
            daz: (deg2rad(gyro[2]) * dt) as f32,
            dvx: (accel[0] * dt) as f32,
            dvy: (accel[1] * dt) as f32,
            dvz: (accel[2] * dt) as f32,
            dt: dt as f32,
        };
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
            let h_acc2 = (nav.h_acc_m * nav.h_acc_m).max(0.05) * 80.0;
            let v_acc2 = (nav.v_acc_m * nav.v_acc_m).max(0.05) * 80.0;
            let s_acc2 = (nav.s_acc_mps * nav.s_acc_mps).max(0.02) * 80.0;
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
        bias_gyro,
        bias_accel,
        cov_bias,
        cov_nonbias,
        map,
        map_heading,
    }
}
