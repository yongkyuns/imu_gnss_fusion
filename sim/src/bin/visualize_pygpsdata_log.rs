use std::{collections::HashMap, fs::File, io::Read, path::PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use ekf_rs::ekf::{Ekf, GpsData, ImuSample, ekf_fuse_gps, ekf_init, ekf_predict};
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use walkers::sources::{Mapbox, MapboxStyle, OpenStreetMap};
use walkers::{HttpTiles, Map, MapMemory, Plugin, Position, lon_lat};
use sim::ubxlog::{
    extract_esf_alg, extract_esf_cal_samples, extract_esf_ins, extract_esf_meas_samples, extract_esf_raw_samples,
    extract_itow_ms, extract_nav_att, extract_nav_pvt, extract_nav_pvt_obs, extract_nav_sat_cn0, fit_linear_map,
    parse_ubx_frames,
    sensor_meta, unwrap_counter,
};

#[derive(Parser, Debug)]
#[command(name = "visualize_pygpsdata_log")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long)]
    profile_only: bool,
}

#[derive(Clone, Default)]
struct Trace {
    name: String,
    points: Vec<[f64; 2]>,
}

fn normalize_heading_deg(mut deg: f64) -> f64 {
    deg %= 360.0;
    if deg < 0.0 {
        deg += 360.0;
    }
    deg
}

#[derive(Default)]
struct PlotData {
    speed: Vec<Trace>,
    sat_cn0: Vec<Trace>,
    imu_raw_gyro: Vec<Trace>,
    imu_raw_accel: Vec<Trace>,
    imu_cal_gyro: Vec<Trace>,
    imu_cal_accel: Vec<Trace>,
    esf_ins_gyro: Vec<Trace>,
    esf_ins_accel: Vec<Trace>,
    orientation: Vec<Trace>,
    other: Vec<Trace>,
    ekf_cmp_pos: Vec<Trace>,
    ekf_cmp_vel: Vec<Trace>,
    ekf_cmp_att: Vec<Trace>,
    ekf_res_pos: Vec<Trace>,
    ekf_res_vel: Vec<Trace>,
    ekf_res_att: Vec<Trace>,
    ekf_map: Vec<Trace>,
    ekf_map_heading: Vec<HeadingSample>,
}

#[derive(Clone, Copy, Default)]
struct HeadingSample {
    t_s: f64,
    lon_deg: f64,
    lat_deg: f64,
    yaw_deg: f64,
}

#[derive(Clone, Copy)]
struct AlgEvent {
    t_ms: f64,
    roll_deg: f64,
    pitch_deg: f64,
    yaw_deg: f64,
}

#[derive(Clone, Copy)]
struct NavAttEvent {
    t_ms: f64,
    roll_deg: f64,
    pitch_deg: f64,
    heading_deg: f64,
}

fn nearest_master_ms(seq: u64, masters: &[(u64, f64)]) -> Option<f64> {
    if masters.is_empty() {
        return None;
    }
    let idx = masters.partition_point(|(s, _)| *s < seq);
    if idx == 0 {
        return Some(masters[0].1);
    }
    if idx >= masters.len() {
        return Some(masters[masters.len() - 1].1);
    }
    let (sl, ml) = masters[idx - 1];
    let (sr, mr) = masters[idx];
    let dl = sl.abs_diff(seq);
    let dr = sr.abs_diff(seq);
    if dr < dl { Some(mr) } else { Some(ml) }
}

fn deg2rad(v: f64) -> f64 {
    v * std::f64::consts::PI / 180.0
}

fn rad2deg(v: f64) -> f64 {
    v * 180.0 / std::f64::consts::PI
}

fn wrap_pi(mut a: f64) -> f64 {
    while a > std::f64::consts::PI {
        a -= 2.0 * std::f64::consts::PI;
    }
    while a < -std::f64::consts::PI {
        a += 2.0 * std::f64::consts::PI;
    }
    a
}

fn wrap_deg180(mut a: f64) -> f64 {
    while a > 180.0 {
        a -= 360.0;
    }
    while a < -180.0 {
        a += 360.0;
    }
    a
}

fn rot_zyx(yaw_rad: f64, pitch_rad: f64, roll_rad: f64) -> [[f64; 3]; 3] {
    let (sy, cy) = yaw_rad.sin_cos();
    let (sp, cp) = pitch_rad.sin_cos();
    let (sr, cr) = roll_rad.sin_cos();
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

fn transpose(r: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][2], r[1][2], r[2][2]],
    ]
}

fn mat_vec(r: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

fn quat_rpy_deg(q0: f32, q1: f32, q2: f32, q3: f32) -> (f64, f64, f64) {
    let qw = q0 as f64;
    let qx = q1 as f64;
    let qy = q2 as f64;
    let qz = q3 as f64;
    let sinr_cosp = 2.0 * (qw * qx + qy * qz);
    let cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    let roll = sinr_cosp.atan2(cosr_cosp);
    let sinp = 2.0 * (qw * qy - qz * qx);
    let pitch = if sinp.abs() >= 1.0 {
        sinp.signum() * std::f64::consts::FRAC_PI_2
    } else {
        sinp.asin()
    };
    let siny_cosp = 2.0 * (qw * qz + qx * qy);
    let cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    let yaw = siny_cosp.atan2(cosy_cosp);
    (rad2deg(roll), rad2deg(pitch), normalize_heading_deg(rad2deg(yaw)))
}

fn lla_to_ecef(lat_deg: f64, lon_deg: f64, h_m: f64) -> [f64; 3] {
    let a = 6378137.0_f64;
    let e2 = 6.69437999014e-3_f64;
    let lat = deg2rad(lat_deg);
    let lon = deg2rad(lon_deg);
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    let n = a / (1.0 - e2 * slat * slat).sqrt();
    [
        (n + h_m) * clat * clon,
        (n + h_m) * clat * slon,
        (n * (1.0 - e2) + h_m) * slat,
    ]
}

fn ecef_to_ned(ecef: [f64; 3], ref_ecef: [f64; 3], ref_lat_deg: f64, ref_lon_deg: f64) -> [f64; 3] {
    let lat = deg2rad(ref_lat_deg);
    let lon = deg2rad(ref_lon_deg);
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    let dx = ecef[0] - ref_ecef[0];
    let dy = ecef[1] - ref_ecef[1];
    let dz = ecef[2] - ref_ecef[2];
    [
        -slat * clon * dx - slat * slon * dy + clat * dz,
        -slon * dx + clon * dy,
        -clat * clon * dx - clat * slon * dy - slat * dz,
    ]
}

fn ned_to_lla_approx(n: f64, e: f64, d: f64, ref_lat_deg: f64, ref_lon_deg: f64, ref_h_m: f64) -> (f64, f64, f64) {
    let a = 6378137.0_f64;
    let e2 = 6.69437999014e-3_f64;
    let lat0 = deg2rad(ref_lat_deg);
    let sin_lat = lat0.sin();
    let denom = (1.0 - e2 * sin_lat * sin_lat).sqrt();
    let rn = a / denom;
    let rm = a * (1.0 - e2) / (denom * denom * denom);
    let dlat = n / (rm + ref_h_m);
    let dlon = e / ((rn + ref_h_m) * lat0.cos().max(1e-6));
    let lat = ref_lat_deg + rad2deg(dlat);
    let lon = ref_lon_deg + rad2deg(dlon);
    let h = ref_h_m - d;
    (lat, lon, h)
}

fn heading_endpoint(lat_deg: f64, lon_deg: f64, heading_deg: f64, length_m: f64) -> (f64, f64) {
    let r = 6_378_137.0_f64;
    let h = deg2rad(heading_deg);
    let d_n = length_m * h.cos();
    let d_e = length_m * h.sin();
    let d_lat = d_n / r;
    let d_lon = d_e / (r * deg2rad(lat_deg).cos().max(1e-6));
    (lat_deg + rad2deg(d_lat), lon_deg + rad2deg(d_lon))
}

#[derive(Clone, Copy)]
struct ImuPacket {
    t_ms: f64,
    gx_dps: f64,
    gy_dps: f64,
    gz_dps: f64,
    ax_mps2: f64,
    ay_mps2: f64,
    az_mps2: f64,
}

fn build_ekf_compare_traces(
    frames: &[sim::ubxlog::UbxFrame],
    masters: &[(u64, f64)],
    t0_master_ms: f64,
) -> (Vec<Trace>, Vec<Trace>, Vec<Trace>, Vec<Trace>, Vec<Trace>, Vec<Trace>, Vec<Trace>, Vec<HeadingSample>) {
    if masters.is_empty() {
        return (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
    }
    let rel_s = |master_ms: f64| (master_ms - t0_master_ms) * 1e-3;

    let mut alg_events = Vec::<AlgEvent>::new();
    let mut nav_att_events = Vec::<NavAttEvent>::new();
    let mut nav_events = Vec::<(f64, sim::ubxlog::NavPvtObs)>::new();
    for f in frames {
        if let Some((_, roll, pitch, yaw)) = extract_esf_alg(f) {
            if let Some(t_ms) = nearest_master_ms(f.seq, masters) {
                alg_events.push(AlgEvent {
                    t_ms,
                    roll_deg: roll,
                    pitch_deg: pitch,
                    yaw_deg: yaw,
                });
            }
        }
        if let Some((_itow, roll, pitch, heading)) = extract_nav_att(f) {
            if let Some(t_ms) = nearest_master_ms(f.seq, masters) {
                nav_att_events.push(NavAttEvent {
                    t_ms,
                    roll_deg: roll,
                    pitch_deg: pitch,
                    heading_deg: normalize_heading_deg(heading),
                });
            }
        }
        if let Some(obs) = extract_nav_pvt_obs(f) {
            if let Some(t_ms) = nearest_master_ms(f.seq, masters) {
                if obs.fix_ok && !obs.invalid_llh {
                    nav_events.push((t_ms, obs));
                }
            }
        }
    }
    alg_events.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap_or(std::cmp::Ordering::Equal));
    nav_att_events.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap_or(std::cmp::Ordering::Equal));
    nav_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

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
    let raw_tag_u = unwrap_counter(&raw_tag, 1 << 16);
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for (seq, tag_u) in raw_seq.iter().zip(raw_tag_u.iter()) {
        if let Some(ms) = nearest_master_ms(*seq, masters) {
            x.push(*tag_u as f64);
            y.push(ms);
        }
    }
    let (a_raw, b_raw) = fit_linear_map(&x, &y, 1e-3);
    let master_min = masters.iter().map(|(_, ms)| *ms).fold(f64::INFINITY, f64::min);
    let master_max = masters.iter().map(|(_, ms)| *ms).fold(f64::NEG_INFINITY, f64::max);

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
            if let (Some(gxv), Some(gyv), Some(gzv), Some(axv), Some(ayv), Some(azv)) = (gx, gy, gz, ax, ay, az) {
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
            if let Some(seq_ms) = nearest_master_ms(*seq, masters) {
                let mut mapped_ms = a_raw * *tag_u as f64 + b_raw;
                if !mapped_ms.is_finite()
                    || mapped_ms < master_min - 1000.0
                    || mapped_ms > master_max + 1000.0
                    || (mapped_ms - seq_ms).abs() > 2000.0
                {
                    mapped_ms = seq_ms;
                }
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
    if let (Some(gxv), Some(gyv), Some(gzv), Some(axv), Some(ayv), Some(azv)) = (gx, gy, gz, ax, ay, az) {
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
    imu_packets.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap_or(std::cmp::Ordering::Equal));

    let mut cmp_pos_n = Vec::<[f64; 2]>::new();
    let mut cmp_pos_e = Vec::<[f64; 2]>::new();
    let mut cmp_pos_d = Vec::<[f64; 2]>::new();
    let mut ubx_pos_n = Vec::<[f64; 2]>::new();
    let mut ubx_pos_e = Vec::<[f64; 2]>::new();
    let mut ubx_pos_d = Vec::<[f64; 2]>::new();
    let mut res_pos_n = Vec::<[f64; 2]>::new();
    let mut res_pos_e = Vec::<[f64; 2]>::new();
    let mut res_pos_d = Vec::<[f64; 2]>::new();

    let mut cmp_vel_n = Vec::<[f64; 2]>::new();
    let mut cmp_vel_e = Vec::<[f64; 2]>::new();
    let mut cmp_vel_d = Vec::<[f64; 2]>::new();
    let mut ubx_vel_n = Vec::<[f64; 2]>::new();
    let mut ubx_vel_e = Vec::<[f64; 2]>::new();
    let mut ubx_vel_d = Vec::<[f64; 2]>::new();
    let mut res_vel_n = Vec::<[f64; 2]>::new();
    let mut res_vel_e = Vec::<[f64; 2]>::new();
    let mut res_vel_d = Vec::<[f64; 2]>::new();

    let mut cmp_att_roll = Vec::<[f64; 2]>::new();
    let mut cmp_att_pitch = Vec::<[f64; 2]>::new();
    let mut cmp_att_yaw = Vec::<[f64; 2]>::new();
    let mut ubx_att_roll = Vec::<[f64; 2]>::new();
    let mut ubx_att_pitch = Vec::<[f64; 2]>::new();
    let mut ubx_att_yaw = Vec::<[f64; 2]>::new();
    let mut res_att_roll = Vec::<[f64; 2]>::new();
    let mut res_att_pitch = Vec::<[f64; 2]>::new();
    let mut res_att_yaw = Vec::<[f64; 2]>::new();
    let mut map_ubx = Vec::<[f64; 2]>::new(); // [lon, lat]
    let mut map_ekf = Vec::<[f64; 2]>::new(); // [lon, lat]
    let mut map_heading = Vec::<HeadingSample>::new();

    let mut ekf = Ekf::default();
    ekf_init(&mut ekf, 1.0);
    ekf.state.q0 = 1.0;
    let mut prev_imu_t: Option<f64> = None;
    let mut alg_idx = 0usize;
    let mut nav_idx = 0usize;
    let mut nav_att_idx = 0usize;
    let mut cur_alg: Option<AlgEvent> = None;

    let mut origin_set = false;
    let mut ref_lat = 0.0_f64;
    let mut ref_lon = 0.0_f64;
    let mut ref_ecef = [0.0_f64; 3];
    let mut ref_h = 0.0_f64;

    let mut next_gps_update_ms = f64::NEG_INFINITY;
    let gps_period_ms = 500.0_f64; // 2Hz, aligned with EKF integration path.

    for pkt in &imu_packets {
        while alg_idx < alg_events.len() && alg_events[alg_idx].t_ms <= pkt.t_ms {
            cur_alg = Some(alg_events[alg_idx]);
            alg_idx += 1;
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

        let mut gyro = [pkt.gx_dps, pkt.gy_dps, pkt.gz_dps];
        let mut accel = [pkt.ax_mps2, pkt.ay_mps2, pkt.az_mps2];
        if let Some(alg) = cur_alg {
            let r_bs = rot_zyx(deg2rad(alg.yaw_deg), deg2rad(alg.pitch_deg), deg2rad(alg.roll_deg));
            let r_sb = transpose(r_bs);
            gyro = mat_vec(r_sb, gyro);
            accel = mat_vec(r_sb, accel);
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
        ekf_predict(
            &mut ekf,
            &imu,
            2.5e-4,
            1.2e-3,
            5.0e-7,
            2.0e-6,
            2.5e-6,
            3.0e-6,
            None,
        );

        while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
            let (t_ms, nav) = nav_events[nav_idx];
            nav_idx += 1;
            if !next_gps_update_ms.is_finite() {
                next_gps_update_ms = t_ms;
            }
            if t_ms + 1e-6 < next_gps_update_ms {
                continue;
            }
            next_gps_update_ms += gps_period_ms;

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
            let mut heading_rad = wrap_pi(deg2rad(nav.heading_motion_deg));
            let mut r_yaw = deg2rad(nav.head_acc_deg).powi(2).max(0.02);
            if speed_h < 1.0 || !r_yaw.is_finite() {
                let (_, _, yaw_deg) = quat_rpy_deg(ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3);
                heading_rad = deg2rad(yaw_deg);
                r_yaw = 1e6;
            }
            let h_acc2 = (nav.h_acc_m * nav.h_acc_m).max(0.05);
            let v_acc2 = (nav.v_acc_m * nav.v_acc_m).max(0.05);
            let s_acc2 = (nav.s_acc_mps * nav.s_acc_mps).max(0.02);
            let gps = GpsData {
                pos_n: ned[0] as f32,
                pos_e: ned[1] as f32,
                pos_d: ned[2] as f32,
                vel_n: nav.vel_n_mps as f32,
                vel_e: nav.vel_e_mps as f32,
                vel_d: nav.vel_d_mps as f32,
                heading_rad: heading_rad as f32,
                R_POS_N: h_acc2 as f32,
                R_POS_E: h_acc2 as f32,
                R_POS_D: v_acc2 as f32,
                R_VEL_N: s_acc2 as f32,
                R_VEL_E: s_acc2 as f32,
                R_VEL_D: s_acc2 as f32,
                R_YAW: r_yaw as f32,
            };
            ekf_fuse_gps(&mut ekf, &gps);

            let t = rel_s(t_ms);
            cmp_pos_n.push([t, ekf.state.pn as f64]);
            cmp_pos_e.push([t, ekf.state.pe as f64]);
            cmp_pos_d.push([t, ekf.state.pd as f64]);
            ubx_pos_n.push([t, ned[0]]);
            ubx_pos_e.push([t, ned[1]]);
            ubx_pos_d.push([t, ned[2]]);
            res_pos_n.push([t, ekf.state.pn as f64 - ned[0]]);
            res_pos_e.push([t, ekf.state.pe as f64 - ned[1]]);
            res_pos_d.push([t, ekf.state.pd as f64 - ned[2]]);

            cmp_vel_n.push([t, ekf.state.vn as f64]);
            cmp_vel_e.push([t, ekf.state.ve as f64]);
            cmp_vel_d.push([t, ekf.state.vd as f64]);
            ubx_vel_n.push([t, nav.vel_n_mps]);
            ubx_vel_e.push([t, nav.vel_e_mps]);
            ubx_vel_d.push([t, nav.vel_d_mps]);
            res_vel_n.push([t, ekf.state.vn as f64 - nav.vel_n_mps]);
            res_vel_e.push([t, ekf.state.ve as f64 - nav.vel_e_mps]);
            res_vel_d.push([t, ekf.state.vd as f64 - nav.vel_d_mps]);
            map_ubx.push([nav.lon_deg, nav.lat_deg]);
            let (ekf_roll, ekf_pitch, ekf_yaw) =
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

            while nav_att_idx + 1 < nav_att_events.len() && nav_att_events[nav_att_idx + 1].t_ms <= t_ms {
                nav_att_idx += 1;
            }
            if nav_att_idx < nav_att_events.len() && nav_att_events[nav_att_idx].t_ms <= t_ms {
                let att = nav_att_events[nav_att_idx];
                cmp_att_roll.push([t, ekf_roll]);
                cmp_att_pitch.push([t, ekf_pitch]);
                cmp_att_yaw.push([t, ekf_yaw]);
                ubx_att_roll.push([t, att.roll_deg]);
                ubx_att_pitch.push([t, att.pitch_deg]);
                ubx_att_yaw.push([t, att.heading_deg]);
                res_att_roll.push([t, ekf_roll - att.roll_deg]);
                res_att_pitch.push([t, ekf_pitch - att.pitch_deg]);
                res_att_yaw.push([t, wrap_deg180(ekf_yaw - att.heading_deg)]);
            }
        }
    }

    let cmp_pos = vec![
        Trace { name: "EKF posN [m]".to_string(), points: cmp_pos_n },
        Trace { name: "UBX posN [m]".to_string(), points: ubx_pos_n },
        Trace { name: "EKF posE [m]".to_string(), points: cmp_pos_e },
        Trace { name: "UBX posE [m]".to_string(), points: ubx_pos_e },
        Trace { name: "EKF posD [m]".to_string(), points: cmp_pos_d },
        Trace { name: "UBX posD [m]".to_string(), points: ubx_pos_d },
    ];
    let cmp_vel = vec![
        Trace { name: "EKF velN [m/s]".to_string(), points: cmp_vel_n },
        Trace { name: "UBX velN [m/s]".to_string(), points: ubx_vel_n },
        Trace { name: "EKF velE [m/s]".to_string(), points: cmp_vel_e },
        Trace { name: "UBX velE [m/s]".to_string(), points: ubx_vel_e },
        Trace { name: "EKF velD [m/s]".to_string(), points: cmp_vel_d },
        Trace { name: "UBX velD [m/s]".to_string(), points: ubx_vel_d },
    ];
    let cmp_att = vec![
        Trace { name: "EKF roll [deg]".to_string(), points: cmp_att_roll },
        Trace { name: "UBX roll [deg]".to_string(), points: ubx_att_roll },
        Trace { name: "EKF pitch [deg]".to_string(), points: cmp_att_pitch },
        Trace { name: "UBX pitch [deg]".to_string(), points: ubx_att_pitch },
        Trace { name: "EKF yaw [deg]".to_string(), points: cmp_att_yaw },
        Trace { name: "UBX yaw [deg]".to_string(), points: ubx_att_yaw },
    ];
    let res_pos = vec![
        Trace { name: "res posN [m]".to_string(), points: res_pos_n },
        Trace { name: "res posE [m]".to_string(), points: res_pos_e },
        Trace { name: "res posD [m]".to_string(), points: res_pos_d },
    ];
    let res_vel = vec![
        Trace { name: "res velN [m/s]".to_string(), points: res_vel_n },
        Trace { name: "res velE [m/s]".to_string(), points: res_vel_e },
        Trace { name: "res velD [m/s]".to_string(), points: res_vel_d },
    ];
    let res_att = vec![
        Trace { name: "res roll [deg]".to_string(), points: res_att_roll },
        Trace { name: "res pitch [deg]".to_string(), points: res_att_pitch },
        Trace { name: "res yaw [deg]".to_string(), points: res_att_yaw },
    ];
    let map = vec![
        Trace { name: "u-blox path (lon,lat)".to_string(), points: map_ubx },
        Trace { name: "EKF path (lon,lat)".to_string(), points: map_ekf },
    ];

    (cmp_pos, cmp_vel, cmp_att, res_pos, res_vel, res_att, map, map_heading)
}

fn unwrap_i64_counter(values: &[i64], modulus: i64) -> Vec<i64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut offset = 0i64;
    let mut prev = values[0];
    out.push(prev);
    for &v in values.iter().skip(1) {
        // Treat wrap only when crossing near the modulus boundary, not on small
        // out-of-order/noisy decreases.
        if v < prev && (prev - v) > (modulus / 2) {
            offset = offset.saturating_add(modulus);
        }
        out.push(v.saturating_add(offset));
        prev = v;
    }
    out
}

fn build_plot_data(bytes: &[u8], max_records: Option<usize>) -> (PlotData, bool) {
    let frames = parse_ubx_frames(bytes, max_records);
    let mut masters: Vec<(u64, f64)> = Vec::new();
    for f in &frames {
        if let Some(itow) = extract_itow_ms(f) {
            // Keep only true GPS iTOW-in-ms range; reject non-iTOW tags.
            if (0..604_800_000).contains(&itow) {
                masters.push((f.seq, itow as f64));
            }
        }
    }
    masters.sort_by_key(|x| x.0);
    if !masters.is_empty() {
        let raw: Vec<i64> = masters.iter().map(|(_, ms)| *ms as i64).collect();
        let unwrapped = unwrap_i64_counter(&raw, 604_800_000);
        for (m, msu) in masters.iter_mut().zip(unwrapped.into_iter()) {
            m.1 = msu as f64;
        }

        // Drop non-monotonic / implausible jumps in iTOW timeline (e.g. resets/spikes).
        // This keeps one consistent relative-time epoch for plotting.
        let mut filtered: Vec<(u64, f64)> = Vec::with_capacity(masters.len());
        let mut last_ms: Option<f64> = None;
        for (seq, ms) in masters.iter().copied() {
            match last_ms {
                None => {
                    filtered.push((seq, ms));
                    last_ms = Some(ms);
                }
                Some(prev) => {
                    let dt = ms - prev;
                    if (0.0..=10_000.0).contains(&dt) {
                        filtered.push((seq, ms));
                        last_ms = Some(ms);
                    }
                }
            }
        }
        if filtered.len() >= 10 {
            masters = filtered;
        }
    }
    let has_itow = !masters.is_empty();
    let t0_master_ms = masters
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::INFINITY, f64::min);
    let t0_master_ms = if t0_master_ms.is_finite() { t0_master_ms } else { 0.0 };
    let master_ms_to_rel_s = |master_ms: f64| -> Option<f64> {
        if !has_itow {
            return None;
        }
        if master_ms < t0_master_ms {
            return None;
        }
        Some((master_ms - t0_master_ms) * 1e-3)
    };

    let seq_to_rel_s = |seq: u64| -> Option<f64> {
        let master_ms = nearest_master_ms(seq, &masters)?;
        master_ms_to_rel_s(master_ms)
    };

    let (ekf_cmp_pos, ekf_cmp_vel, ekf_cmp_att, ekf_res_pos, ekf_res_vel, ekf_res_att, ekf_map, ekf_map_heading) =
        build_ekf_compare_traces(&frames, &masters, t0_master_ms);

    let mut speed_g = Vec::<[f64; 2]>::new();
    let mut speed_n = Vec::<[f64; 2]>::new();
    let mut speed_e = Vec::<[f64; 2]>::new();
    let mut speed_d = Vec::<[f64; 2]>::new();
    let mut sats: HashMap<String, Vec<[f64; 2]>> = HashMap::new();
    let mut orient_map: HashMap<String, Vec<[f64; 2]>> = HashMap::new();
    let mut other_map: HashMap<String, Vec<[f64; 2]>> = HashMap::new();
    let mut esf_ins_gyro_map: HashMap<String, Vec<[f64; 2]>> = HashMap::new();
    let mut esf_ins_accel_map: HashMap<String, Vec<[f64; 2]>> = HashMap::new();

    for f in &frames {
        if let Some((_itow, gs, vn, ve, vd, _lat, _lon)) = extract_nav_pvt(f) {
            if let Some(t) = seq_to_rel_s(f.seq) {
            speed_g.push([t, gs]);
            speed_n.push([t, vn]);
            speed_e.push([t, ve]);
            speed_d.push([t, vd]);
            }
        }
        if let Some((_itow, roll, pitch, yaw)) = extract_nav_att(f) {
            if let Some(t) = seq_to_rel_s(f.seq) {
            other_map
                .entry("NAV-ATT roll [deg]".to_string())
                .or_default()
                .push([t, roll]);
            other_map
                .entry("NAV-ATT pitch [deg]".to_string())
                .or_default()
                .push([t, pitch]);
            other_map
                .entry("NAV-ATT heading [deg]".to_string())
                .or_default()
                .push([t, normalize_heading_deg(yaw)]);
            }
        }
        if let Some((_itow, roll, pitch, yaw)) = extract_esf_alg(f) {
            if let Some(t) = seq_to_rel_s(f.seq) {
            orient_map
                .entry("ESF-ALG roll [deg]".to_string())
                .or_default()
                .push([t, roll]);
            orient_map
                .entry("ESF-ALG pitch [deg]".to_string())
                .or_default()
                .push([t, pitch]);
            orient_map
                .entry("ESF-ALG yaw [deg]".to_string())
                .or_default()
                .push([t, normalize_heading_deg(yaw)]);
            }
        }
        for (sat, cno) in extract_nav_sat_cn0(f) {
            if let Some(t) = seq_to_rel_s(f.seq) {
                sats.entry(sat).or_default().push([t, cno]);
            }
        }
        if let Some((_itow, wx, wy, wz, ax, ay, az)) = extract_esf_ins(f) {
            if let Some(t) = seq_to_rel_s(f.seq) {
                esf_ins_gyro_map
                    .entry("ESF-INS wx [deg/s]".to_string())
                    .or_default()
                    .push([t, wx]);
                esf_ins_gyro_map
                    .entry("ESF-INS wy [deg/s]".to_string())
                    .or_default()
                    .push([t, wy]);
                esf_ins_gyro_map
                    .entry("ESF-INS wz [deg/s]".to_string())
                    .or_default()
                    .push([t, wz]);
                esf_ins_accel_map
                    .entry("ESF-INS ax [m/s^2]".to_string())
                    .or_default()
                    .push([t, ax]);
                esf_ins_accel_map
                    .entry("ESF-INS ay [m/s^2]".to_string())
                    .or_default()
                    .push([t, ay]);
                esf_ins_accel_map
                    .entry("ESF-INS az [m/s^2]".to_string())
                    .or_default()
                    .push([t, az]);
            }
        }
    }

    let mut raw_tag = Vec::<u64>::new();
    let mut raw_seq = Vec::<u64>::new();
    let mut raw_sig = Vec::<(u8, f64)>::new();
    for f in &frames {
        for (tag, sw) in extract_esf_raw_samples(f) {
            raw_tag.push(tag);
            raw_seq.push(f.seq);
            let (_name, _unit, scale) = sensor_meta(sw.dtype);
            raw_sig.push((sw.dtype, sw.value_i24 as f64 * scale));
        }
    }
    let raw_unwrapped = unwrap_counter(&raw_tag, 1 << 16);
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for (seq, tag_u) in raw_seq.iter().zip(raw_unwrapped.iter()) {
        if let Some(ms) = nearest_master_ms(*seq, &masters) {
            x.push(*tag_u as f64);
            y.push(ms);
        }
    }
    let (a_raw, b_raw) = fit_linear_map(&x, &y, 1e-3);

    let mut cal_tag = Vec::<u64>::new();
    let mut cal_seq = Vec::<u64>::new();
    let mut cal_sig = Vec::<(u8, f64, &'static str)>::new();
    for f in &frames {
        for (tag, sw) in extract_esf_cal_samples(f) {
            cal_tag.push(tag);
            cal_seq.push(f.seq);
            let (_name, _unit, scale) = sensor_meta(sw.dtype);
            cal_sig.push((sw.dtype, sw.value_i24 as f64 * scale, "ESF-CAL"));
        }
    }
    let cal_u = unwrap_counter(&cal_tag, 1 << 16);
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for (seq, tag_u) in cal_seq.iter().zip(cal_u.iter()) {
        if let Some(ms) = nearest_master_ms(*seq, &masters) {
            x.push(*tag_u as f64);
            y.push(ms);
        }
    }
    let (a_cal, b_cal) = fit_linear_map(&x, &y, 1e-3);

    let mut meas_tag = Vec::<u64>::new();
    let mut meas_seq = Vec::<u64>::new();
    let mut meas_sig = Vec::<(u8, f64, &'static str)>::new();
    for f in &frames {
        for (tag, sw) in extract_esf_meas_samples(f) {
            meas_tag.push(tag);
            meas_seq.push(f.seq);
            let (_name, _unit, scale) = sensor_meta(sw.dtype);
            meas_sig.push((sw.dtype, sw.value_i24 as f64 * scale, "ESF-MEAS"));
        }
    }
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for (seq, tag) in meas_seq.iter().zip(meas_tag.iter()) {
        if let Some(ms) = nearest_master_ms(*seq, &masters) {
            x.push(*tag as f64);
            y.push(ms);
        }
    }
    let (a_meas, b_meas) = fit_linear_map(&x, &y, 1e-3);

    let master_min = masters
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::INFINITY, f64::min);
    let master_max = masters
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::NEG_INFINITY, f64::max);
    let master_min = if master_min.is_finite() { master_min } else { 0.0 };
    let master_max = if master_max.is_finite() { master_max } else { master_min };
    let map_tag_ms = |a: f64, b: f64, tag: f64, seq: u64| -> Option<f64> {
        let seq_ms = nearest_master_ms(seq, &masters)?;
        let mut ms = a * tag + b;
        // Reject impossible/outlier tag mapping and fall back to frame timebase.
        if !ms.is_finite()
            || ms < master_min - 1000.0
            || ms > master_max + 1000.0
            || (ms - seq_ms).abs() > 2000.0
        {
            ms = seq_ms;
        }
        Some(ms)
    };

    let mut raw_by_sig: HashMap<String, Vec<[f64; 2]>> = HashMap::new();
    for (((dtype, val), tag), seq) in raw_sig.iter().zip(raw_unwrapped.iter()).zip(raw_seq.iter()) {
        let (name, _unit, _scale) = sensor_meta(*dtype);
        let master_ms = match map_tag_ms(a_raw, b_raw, *tag as f64, *seq) {
            Some(v) => v,
            None => continue,
        };
        if let Some(t) = master_ms_to_rel_s(master_ms) {
            raw_by_sig
                .entry(format!("ESF-RAW {}", name))
                .or_default()
                .push([t, *val]);
        }
    }

    let mut cal_by_sig: HashMap<String, Vec<[f64; 2]>> = HashMap::new();
    for (((dtype, val, src), tag), seq) in cal_sig.iter().zip(cal_u.iter()).zip(cal_seq.iter()) {
        let (name, _unit, _scale) = sensor_meta(*dtype);
        let master_ms = match map_tag_ms(a_cal, b_cal, *tag as f64, *seq) {
            Some(v) => v,
            None => continue,
        };
        if let Some(t) = master_ms_to_rel_s(master_ms) {
            cal_by_sig
                .entry(format!("{} {}", src, name))
                .or_default()
                .push([t, *val]);
        }
    }
    for (((dtype, val, src), tag), seq) in meas_sig.iter().zip(meas_tag.iter()).zip(meas_seq.iter()) {
        let (name, _unit, _scale) = sensor_meta(*dtype);
        let master_ms = match map_tag_ms(a_meas, b_meas, *tag as f64, *seq) {
            Some(v) => v,
            None => continue,
        };
        if let Some(t) = master_ms_to_rel_s(master_ms) {
            cal_by_sig
                .entry(format!("{} {}", src, name))
                .or_default()
                .push([t, *val]);
        }
    }

    let mut out = PlotData::default();
    out.speed = vec![
        Trace {
            name: "gSpeed [m/s]".to_string(),
            points: speed_g,
        },
        Trace {
            name: "velN [m/s]".to_string(),
            points: speed_n,
        },
        Trace {
            name: "velE [m/s]".to_string(),
            points: speed_e,
        },
        Trace {
            name: "velD [m/s]".to_string(),
            points: speed_d,
        },
    ];
    out.sat_cn0 = sats
        .into_iter()
        .map(|(k, v)| Trace {
            name: k,
            points: v,
        })
        .collect();

    for (k, v) in raw_by_sig {
        if k.contains("gyro_") {
            out.imu_raw_gyro.push(Trace { name: k, points: v });
        } else if k.contains("accel_") {
            out.imu_raw_accel.push(Trace { name: k, points: v });
        } else {
            out.other.push(Trace { name: k, points: v });
        }
    }

    for (k, v) in cal_by_sig {
        if k.contains("gyro_") {
            out.imu_cal_gyro.push(Trace { name: k, points: v });
        } else if k.contains("accel_") {
            out.imu_cal_accel.push(Trace { name: k, points: v });
        } else {
            out.other.push(Trace { name: k, points: v });
        }
    }

    for (k, v) in other_map {
        out.other.push(Trace { name: k, points: v });
    }

    for (name, points) in orient_map {
        out.orientation.push(Trace { name, points });
    }
    out.orientation
        .sort_by(|a, b| a.name.partial_cmp(&b.name).unwrap_or(std::cmp::Ordering::Equal));

    for (name, points) in esf_ins_accel_map {
        out.esf_ins_accel.push(Trace { name, points });
    }
    for (name, points) in esf_ins_gyro_map {
        out.esf_ins_gyro.push(Trace { name, points });
    }
    out.esf_ins_gyro
        .sort_by(|a, b| a.name.partial_cmp(&b.name).unwrap_or(std::cmp::Ordering::Equal));
    out.esf_ins_accel
        .sort_by(|a, b| a.name.partial_cmp(&b.name).unwrap_or(std::cmp::Ordering::Equal));
    out.ekf_cmp_pos = ekf_cmp_pos;
    out.ekf_cmp_vel = ekf_cmp_vel;
    out.ekf_cmp_att = ekf_cmp_att;
    out.ekf_res_pos = ekf_res_pos;
    out.ekf_res_vel = ekf_res_vel;
    out.ekf_res_att = ekf_res_att;
    out.ekf_map = ekf_map;
    out.ekf_map_heading = ekf_map_heading;

    let max_rel_s = ((master_max - t0_master_ms) * 1e-3).max(0.0);
    let sanitize_trace = |trace: &mut Trace| {
        let mut cleaned = Vec::with_capacity(trace.points.len());
        let mut last_t = -1e-9_f64;
        for p in trace.points.iter().copied() {
            let t = p[0];
            let y = p[1];
            if !t.is_finite() || !y.is_finite() {
                continue;
            }
            if t < -1e-6 || t > max_rel_s + 1.0 {
                continue;
            }
            if t + 1e-9 < last_t {
                continue;
            }
            cleaned.push(p);
            last_t = t;
        }
        trace.points = cleaned;
    };

    for traces in [
        &mut out.speed,
        &mut out.sat_cn0,
        &mut out.imu_raw_gyro,
        &mut out.imu_raw_accel,
        &mut out.imu_cal_gyro,
        &mut out.imu_cal_accel,
        &mut out.esf_ins_gyro,
        &mut out.esf_ins_accel,
        &mut out.orientation,
        &mut out.other,
        &mut out.ekf_cmp_pos,
        &mut out.ekf_cmp_vel,
        &mut out.ekf_cmp_att,
        &mut out.ekf_res_pos,
        &mut out.ekf_res_vel,
        &mut out.ekf_res_att,
    ] {
        for tr in traces.iter_mut() {
            tr.points
                .sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
            sanitize_trace(tr);
        }
    }
    for tr in &mut out.ekf_map {
        tr.points.retain(|p| p[0].is_finite() && p[1].is_finite());
    }

    (out, has_itow)
}

fn trace_stats(data: &PlotData) -> (usize, usize) {
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
        &data.ekf_res_pos,
        &data.ekf_res_vel,
        &data.ekf_res_att,
    ];
    let mut traces = 0usize;
    let mut points = 0usize;
    for g in groups {
        traces += g.len();
        points += g.iter().map(|t| t.points.len()).sum::<usize>();
    }
    (traces, points)
}

fn trace_time_bounds(data: &PlotData) -> Option<(f64, f64)> {
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
        &data.ekf_res_pos,
        &data.ekf_res_vel,
        &data.ekf_res_att,
        &data.ekf_map,
    ];
    let mut min_t = f64::INFINITY;
    let mut max_t = f64::NEG_INFINITY;
    for g in groups {
        for tr in g.iter() {
            for p in tr.points.iter() {
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

fn group_stats(name: &str, traces: &[Trace]) -> (String, usize, usize) {
    let n_traces = traces.len();
    let n_points = traces.iter().map(|t| t.points.len()).sum::<usize>();
    (name.to_string(), n_traces, n_points)
}

fn max_gap_sec(traces: &[Trace]) -> f64 {
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

fn max_gap_trace(traces: &[Trace]) -> Option<(String, f64)> {
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

struct App {
    data: PlotData,
    show_egui_inspection: bool,
    show_esf_meas: bool,
    has_itow: bool,
    fps_ema: f32,
    max_points_per_trace: usize,
    page: Page,
    map_tiles: HttpTiles,
    map_memory: MapMemory,
    map_center: Position,
    show_heading: bool,
}

const MAPBOX_ACCESS_TOKEN: &str =
    "pk.eyJ1IjoieW9uZ2t5dW5zODciLCJhIjoiY21tNjB5NWt6MGJmOTJzcG02MmRvN3RnYiJ9.fu_66qb1G1cgrLzAE54E0w";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Signals,
    EkfCompare,
    MapDark,
}

#[derive(Clone)]
struct TrackOverlay {
    traces: Vec<Trace>,
    headings: Vec<HeadingSample>,
    show_heading: bool,
}

impl Plugin for TrackOverlay {
    fn run(
        self: Box<Self>,
        ui: &mut egui::Ui,
        _response: &egui::Response,
        projector: &walkers::Projector,
        _map_memory: &MapMemory,
    ) {
        let colors = [
            egui::Color32::from_rgb(0, 255, 255),
            egui::Color32::from_rgb(255, 196, 0),
            egui::Color32::from_rgb(60, 200, 120),
            egui::Color32::from_rgb(255, 100, 100),
        ];
        for (idx, tr) in self.traces.iter().enumerate() {
            if tr.points.len() < 2 {
                continue;
            }
            let color = colors[idx % colors.len()];
            let mut pts = Vec::<egui::Pos2>::with_capacity(tr.points.len());
            for p in &tr.points {
                let lon = p[0];
                let lat = p[1];
                if !lon.is_finite() || !lat.is_finite() {
                    continue;
                }
                let v = projector.project(lon_lat(lon, lat));
                pts.push(egui::pos2(v.x, v.y));
            }
            if pts.len() >= 2 {
                ui.painter()
                    .add(egui::epaint::PathShape::line(pts, egui::Stroke::new(2.2, color)));
            }
        }

        if self.show_heading {
            let mut last_tick_t = f64::NEG_INFINITY;
            for h in &self.headings {
                if h.t_s - last_tick_t < 1.0 {
                    continue;
                }
                last_tick_t = h.t_s;
                let from = projector.project(lon_lat(h.lon_deg, h.lat_deg));
                let (tip_lat, tip_lon) = heading_endpoint(h.lat_deg, h.lon_deg, h.yaw_deg, 6.0);
                let to = projector.project(lon_lat(tip_lon, tip_lat));
                ui.painter().line_segment(
                    [egui::pos2(from.x, from.y), egui::pos2(to.x, to.y)],
                    egui::Stroke::new(1.8, egui::Color32::from_rgb(255, 140, 0)),
                );
            }
        }
    }
}

fn map_center_from_traces(traces: &[Trace]) -> Position {
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

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        #[cfg(target_os = "macos")]
        if ctx.input(|i| i.viewport().close_requested()) {
            // Work around an AppKit/touchbar teardown crash path observed on window close.
            std::process::exit(0);
        }

        // egui is event-driven by default; schedule periodic repaint so the UI
        // keeps updating even when there is no pointer/keyboard input.
        ctx.request_repaint_after(Duration::from_millis(33));

        egui::TopBottomPanel::top("top_controls").show(ctx, |ui| {
            let fps = ctx.input(|i| {
                if i.stable_dt > 0.0 {
                    1.0 / i.stable_dt
                } else {
                    0.0
                }
            });
            // Adaptive per-trace budget to keep UI interactive under heavy load.
            if self.fps_ema <= 0.0 {
                self.fps_ema = fps;
            } else {
                self.fps_ema = self.fps_ema * 0.92 + fps * 0.08;
            }
            if self.fps_ema < 24.0 {
                self.max_points_per_trace = (self.max_points_per_trace as f32 * 0.85) as usize;
            } else if self.fps_ema > 50.0 {
                self.max_points_per_trace = (self.max_points_per_trace as f32 * 1.08) as usize;
            }
            self.max_points_per_trace = self.max_points_per_trace.clamp(300, 6000);
            ui.horizontal(|ui| {
                ui.heading("pygpsdata Visualization (Rust + egui)");
                ui.separator();
                ui.label(if self.has_itow {
                    "X-axis: Relative time [s], t=0 at first iTOW"
                } else {
                    "X-axis: Relative time [s] (no valid iTOW found)"
                });
            });
            egui::CollapsingHeader::new("Plot Controls")
                .default_open(false)
                .show(ui, |ui| {
                    ui.label(format!("Estimated FPS: {:.1}", fps));
                    ui.label(format!(
                        "Decimation budget: {} pts/trace (FPS EMA {:.1})",
                        self.max_points_per_trace, self.fps_ema
                    ));
                    ui.checkbox(&mut self.show_esf_meas, "Show ESF-MEAS (Accel)");
                    ui.checkbox(&mut self.show_egui_inspection, "Show egui inspection/profiler");
                });
            ui.horizontal(|ui| {
                ui.label("Page:");
                ui.selectable_value(&mut self.page, Page::Signals, "Signals");
                ui.selectable_value(&mut self.page, Page::EkfCompare, "EKF Compare");
                ui.selectable_value(&mut self.page, Page::MapDark, "Map (Dark)");
            });
        });

        let mut imu_gyro: Vec<Trace> =
            Vec::with_capacity(self.data.imu_raw_gyro.len() + self.data.imu_cal_gyro.len());
        imu_gyro.extend(self.data.imu_raw_gyro.iter().cloned());
        imu_gyro.extend(
            self.data
                .imu_cal_gyro
                .iter()
                .filter(|t| !t.name.starts_with("ESF-MEAS "))
                .cloned(),
        );

        let mut imu_accel: Vec<Trace> =
            Vec::with_capacity(self.data.imu_raw_accel.len() + self.data.imu_cal_accel.len());
        imu_accel.extend(self.data.imu_raw_accel.iter().cloned());
        imu_accel.extend(self.data.imu_cal_accel.iter().cloned());
        if !self.show_esf_meas {
            imu_accel.retain(|t| !t.name.starts_with("ESF-MEAS "));
        }

        match self.page {
            Page::Signals => {
                let half_width = (ctx.content_rect().width() * 0.5).max(260.0);
                egui::SidePanel::left("left_plots")
                    .resizable(false)
                    .exact_width(half_width)
                    .show(ctx, |ui| {
                        draw_plot(ui, "Speed", &self.data.speed, true, self.max_points_per_trace);
                        draw_plot(ui, "IMU Gyro ESF (RAW/CAL)", &imu_gyro, true, self.max_points_per_trace);
                        draw_plot(ui, "ESF-INS Gyro", &self.data.esf_ins_gyro, true, self.max_points_per_trace);
                        draw_plot(ui, "Orientation", &self.data.orientation, true, self.max_points_per_trace);
                    });

                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_plot(ui, "Signal Strength (C/N0)", &self.data.sat_cn0, false, self.max_points_per_trace);
                    draw_plot(ui, "IMU Accel ESF (RAW/CAL/MEAS)", &imu_accel, true, self.max_points_per_trace);
                    draw_plot(ui, "ESF-INS Accel", &self.data.esf_ins_accel, true, self.max_points_per_trace);
                    draw_plot(ui, "Other Signals", &self.data.other, true, self.max_points_per_trace);
                });
            }
            Page::EkfCompare => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_plot(ui, "Position: EKF vs u-blox", &self.data.ekf_cmp_pos, true, self.max_points_per_trace);
                    draw_plot(ui, "Position Residuals (EKF - u-blox)", &self.data.ekf_res_pos, true, self.max_points_per_trace);
                    draw_plot(ui, "Velocity: EKF vs u-blox", &self.data.ekf_cmp_vel, true, self.max_points_per_trace);
                    draw_plot(ui, "Velocity Residuals (EKF - u-blox)", &self.data.ekf_res_vel, true, self.max_points_per_trace);
                    draw_plot(ui, "Orientation: EKF vs u-blox", &self.data.ekf_cmp_att, true, self.max_points_per_trace);
                    draw_plot(ui, "Orientation Residuals (EKF - u-blox)", &self.data.ekf_res_att, true, self.max_points_per_trace);
                });
            }
            Page::MapDark => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Slippy map overlay: u-blox + EKF");
                        ui.checkbox(&mut self.show_heading, "show heading");
                        if ui.button("Recenter").clicked() {
                            self.map_memory.follow_my_position();
                        }
                    });
                    let track = TrackOverlay {
                        traces: self.data.ekf_map.clone(),
                        headings: self.data.ekf_map_heading.clone(),
                        show_heading: self.show_heading,
                    };
                    ui.add(
                        Map::new(Some(&mut self.map_tiles), &mut self.map_memory, self.map_center)
                            .with_plugin(track)
                            .double_click_to_zoom(true),
                    );
                });
            }
        }

        if self.show_egui_inspection {
            egui::Window::new("egui inspection/profiler")
                .vscroll(true)
                .show(ctx, |ui| {
                    ctx.inspection_ui(ui);
                });
        }
    }
}

fn draw_plot(ui: &mut egui::Ui, title: &str, traces: &[Trace], show_legend: bool, max_points_per_trace: usize) {
    fn visible_decimated(points: &[[f64; 2]], xmin: f64, xmax: f64, max_points: usize) -> Vec<[f64; 2]> {
        if points.is_empty() {
            return Vec::new();
        }
        let lo = points.partition_point(|p| p[0] < xmin);
        let hi = points.partition_point(|p| p[0] <= xmax);
        let start = lo.saturating_sub(1);
        let end = if hi < points.len() { hi + 1 } else { points.len() };
        let slice = &points[start..end];
        if slice.len() <= max_points || max_points == 0 {
            return slice.to_vec();
        }

        // Preserve extrema at low zoom: for each x-bucket, keep min/max y.
        let buckets = (max_points / 2).max(1);
        let x0 = slice.first().map(|p| p[0]).unwrap_or(xmin);
        let x1 = slice.last().map(|p| p[0]).unwrap_or(xmax);
        let span = (x1 - x0).abs();
        if span <= f64::EPSILON {
            let step = ((slice.len() as f64) / (max_points as f64)).ceil() as usize;
            return slice.iter().step_by(step.max(1)).copied().collect();
        }

        let mut min_b: Vec<Option<(usize, f64)>> = vec![None; buckets];
        let mut max_b: Vec<Option<(usize, f64)>> = vec![None; buckets];
        for (i, p) in slice.iter().enumerate() {
            let mut b = (((p[0] - x0) / span) * buckets as f64).floor() as usize;
            if b >= buckets {
                b = buckets - 1;
            }
            match min_b[b] {
                Some((_, y)) if p[1] >= y => {}
                _ => min_b[b] = Some((i, p[1])),
            }
            match max_b[b] {
                Some((_, y)) if p[1] <= y => {}
                _ => max_b[b] = Some((i, p[1])),
            }
        }

        let mut out = Vec::with_capacity(max_points);
        let mut last_idx: Option<usize> = None;
        for b in 0..buckets {
            let a = min_b[b].map(|(i, _)| i);
            let c = max_b[b].map(|(i, _)| i);
            match (a, c) {
                (Some(i0), Some(i1)) if i0 == i1 => {
                    if last_idx != Some(i0) {
                        out.push(slice[i0]);
                        last_idx = Some(i0);
                    }
                }
                (Some(i0), Some(i1)) => {
                    let (first, second) = if i0 < i1 { (i0, i1) } else { (i1, i0) };
                    if last_idx != Some(first) {
                        out.push(slice[first]);
                        last_idx = Some(first);
                    }
                    if out.len() < max_points && last_idx != Some(second) {
                        out.push(slice[second]);
                        last_idx = Some(second);
                    }
                }
                (Some(i0), None) | (None, Some(i0)) => {
                    if last_idx != Some(i0) {
                        out.push(slice[i0]);
                        last_idx = Some(i0);
                    }
                }
                (None, None) => {}
            }
            if out.len() >= max_points {
                break;
            }
        }

        if out.is_empty() {
            let step = ((slice.len() as f64) / (max_points as f64)).ceil() as usize;
            return slice.iter().step_by(step.max(1)).copied().collect();
        }
        out
    }

    ui.vertical(|ui| {
        ui.label(title);
        let mut plot = Plot::new(title)
            .height(220.0)
            .link_axis("shared_x", egui::Vec2b::new(true, false))
            .x_axis_formatter(|mark, _range| format!("{:.1}", mark.value))
            .allow_drag(true)
            .allow_zoom(true)
            .allow_scroll(true)
            .allow_boxed_zoom(true)
            .allow_axis_zoom_drag(true);
        if show_legend {
            plot = plot.legend(Legend::default());
        }
        plot.show(ui, |plot_ui| {
            let bounds = plot_ui.plot_bounds();
            let xmin = bounds.min()[0];
            let xmax = bounds.max()[0];
            for t in traces {
                if t.points.is_empty() {
                    continue;
                }
                let reduced = visible_decimated(&t.points, xmin, xmax, max_points_per_trace);
                if reduced.is_empty() {
                    continue;
                }
                let points: PlotPoints<'_> = reduced.into();
                plot_ui.line(Line::new(t.name.clone(), points));
            }
        });
    });
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    let mut bytes = Vec::new();
    File::open(&args.logfile)
        .with_context(|| format!("failed to open {}", args.logfile.display()))?
        .read_to_end(&mut bytes)
        .context("failed to read log")?;
    let t_read = Instant::now();

    let (data, has_itow) = build_plot_data(&bytes, args.max_records);
    let t_build = Instant::now();
    let (n_traces, n_points) = trace_stats(&data);
    let (tmin, tmax) = trace_time_bounds(&data).unwrap_or((f64::NAN, f64::NAN));
    eprintln!(
        "[profile] bytes={} read={:.3}s build={:.3}s total_pre_ui={:.3}s traces={} points={} t_range=[{:.3}, {:.3}]s",
        bytes.len(),
        (t_read - t0).as_secs_f64(),
        (t_build - t_read).as_secs_f64(),
        (t_build - t0).as_secs_f64(),
        n_traces,
        n_points,
        tmin,
        tmax
    );
    for (name, nt, np) in [
        group_stats("speed", &data.speed),
        group_stats("sat_cn0", &data.sat_cn0),
        group_stats("imu_raw_gyro", &data.imu_raw_gyro),
        group_stats("imu_raw_accel", &data.imu_raw_accel),
        group_stats("imu_cal_gyro", &data.imu_cal_gyro),
        group_stats("imu_cal_accel", &data.imu_cal_accel),
        group_stats("esf_ins_gyro", &data.esf_ins_gyro),
        group_stats("esf_ins_accel", &data.esf_ins_accel),
        group_stats("orientation", &data.orientation),
        group_stats("other", &data.other),
        group_stats("ekf_cmp_pos", &data.ekf_cmp_pos),
        group_stats("ekf_cmp_vel", &data.ekf_cmp_vel),
        group_stats("ekf_cmp_att", &data.ekf_cmp_att),
        group_stats("ekf_res_pos", &data.ekf_res_pos),
        group_stats("ekf_res_vel", &data.ekf_res_vel),
        group_stats("ekf_res_att", &data.ekf_res_att),
        group_stats("ekf_map", &data.ekf_map),
    ] {
        eprintln!("[profile] group={} traces={} points={}", name, nt, np);
    }
    eprintln!(
        "[profile] max_gap_s raw_gyro={:.3} raw_accel={:.3} cal_gyro={:.3} cal_accel={:.3}",
        max_gap_sec(&data.imu_raw_gyro),
        max_gap_sec(&data.imu_raw_accel),
        max_gap_sec(&data.imu_cal_gyro),
        max_gap_sec(&data.imu_cal_accel),
    );
    for (group, traces) in [
        ("imu_raw_gyro", &data.imu_raw_gyro),
        ("imu_raw_accel", &data.imu_raw_accel),
        ("imu_cal_gyro", &data.imu_cal_gyro),
        ("imu_cal_accel", &data.imu_cal_accel),
    ] {
        if let Some((name, gap)) = max_gap_trace(traces) {
            eprintln!("[profile] max_gap_trace group={} signal={} gap_s={:.3}", group, name, gap);
        }
    }
    if args.profile_only {
        return Ok(());
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_maximized(true),
        ..Default::default()
    };
    eframe::run_native(
        "visualize_pygpsdata_log",
        native_options,
        Box::new(move |cc| {
            let map_center = map_center_from_traces(&data.ekf_map);
            let map_tiles = if MAPBOX_ACCESS_TOKEN.is_empty() {
                HttpTiles::new(OpenStreetMap, cc.egui_ctx.clone())
            } else {
                HttpTiles::new(
                    Mapbox {
                        style: MapboxStyle::Dark,
                        high_resolution: true,
                        access_token: MAPBOX_ACCESS_TOKEN.to_string(),
                    },
                    cc.egui_ctx.clone(),
                )
            };
            let mut map_memory = MapMemory::default();
            let _ = map_memory.set_zoom(15.0);
            Ok(Box::new(App {
                data,
                show_egui_inspection: false,
                show_esf_meas: false,
                has_itow,
                fps_ema: 0.0,
                max_points_per_trace: 2500,
                page: Page::Signals,
                map_tiles,
                map_memory,
                map_center,
                show_heading: false,
            }))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;
    Ok(())
}
