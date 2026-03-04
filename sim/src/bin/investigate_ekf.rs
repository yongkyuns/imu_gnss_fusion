use std::{f64::consts::PI, fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use ekf_rs::ekf::{
    Ekf, GpsData, ImuSample, ekf_fuse_body_vel, ekf_fuse_gps, ekf_init, ekf_predict,
};
use sim::ubxlog::{
    NavPvtObs, extract_esf_alg, extract_esf_raw_samples, extract_itow_ms, extract_nav_att,
    extract_nav_pvt_obs, extract_nav2_pvt_obs, fit_linear_map, parse_ubx_frames, sensor_meta,
    unwrap_counter,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long, default_value_t = 365.0)]
    window_start_s: f64,
    #[arg(long, default_value_t = 380.0)]
    window_end_s: f64,
}

#[derive(Clone, Copy, Debug)]
enum TransformMode {
    None,
    RotXyz,
    RotXyzTranspose,
    RotXyzNeg,
    RotXyzNegTranspose,
    TransposeZyx,
    DirectZyx,
    TransposeNegZyx,
    DirectNegZyx,
}

#[derive(Clone, Copy, Debug)]
enum HeadingMode {
    NavPvtMotion,
    NavAtt,
    Disabled,
}

#[derive(Clone, Copy, Debug)]
enum FramePostRot {
    Identity,
    Rx180,
    Ry180,
    Rz180,
}

#[derive(Clone, Copy, Debug)]
struct Config {
    name: &'static str,
    transform: TransformMode,
    heading: HeadingMode,
    p_init: f32,
    use_body_vel: bool,
    r_body_vel: f32,
    turn_aware_nhc: bool,
    turn_aware_gain: f32,
    gyro_scale: f64,
    accel_scale: f64,
    frame_post_rot: FramePostRot,
}

#[derive(Default, Clone, Copy, Debug)]
struct Metrics {
    n: usize,
    rmse_pos_h: f64,
    rmse_vel_h: f64,
    rmse_yaw_deg: f64,
    mean_abs_yaw_deg: f64,
    rmse_yaw_fast_deg: f64,
    mean_abs_yaw_fast_deg: f64,
    n_roll: usize,
    rmse_roll_deg: f64,
    mean_abs_roll_deg: f64,
    n_roll_win: usize,
    rmse_roll_win_deg: f64,
    mean_abs_roll_win_deg: f64,
}

#[derive(Clone, Copy, Debug)]
struct AlgEvent {
    t_ms: f64,
    roll_deg: f64,
    pitch_deg: f64,
    yaw_deg: f64,
}

#[derive(Clone, Copy, Debug)]
struct NavAttEvent {
    t_ms: f64,
    roll_deg: f64,
    pitch_deg: f64,
    heading_deg: f64,
}

#[derive(Clone, Copy, Debug)]
struct ImuPacket {
    t_ms: f64,
    gx_dps: f64,
    gy_dps: f64,
    gz_dps: f64,
    ax_mps2: f64,
    ay_mps2: f64,
    az_mps2: f64,
}

fn deg2rad(v: f64) -> f64 {
    v * PI / 180.0
}

fn rad2deg(v: f64) -> f64 {
    v * 180.0 / PI
}

fn wrap_pi(mut a: f64) -> f64 {
    while a > PI {
        a -= 2.0 * PI;
    }
    while a < -PI {
        a += 2.0 * PI;
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

fn normalize_heading_deg(mut deg: f64) -> f64 {
    deg %= 360.0;
    if deg < 0.0 {
        deg += 360.0;
    }
    deg
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
        sinp.signum() * PI / 2.0
    } else {
        sinp.asin()
    };
    let siny_cosp = 2.0 * (qw * qz + qx * qy);
    let cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    let yaw = siny_cosp.atan2(cosy_cosp);
    (
        rad2deg(roll),
        rad2deg(pitch),
        normalize_heading_deg(rad2deg(yaw)),
    )
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

fn rot_xyz(roll_rad: f64, pitch_rad: f64, yaw_rad: f64) -> [[f64; 3]; 3] {
    let (sr, cr) = roll_rad.sin_cos();
    let (sp, cp) = pitch_rad.sin_cos();
    let (sy, cy) = yaw_rad.sin_cos();
    [
        [cp * cy, -cp * sy, sp],
        [cr * sy + sr * sp * cy, cr * cy - sr * sp * sy, -sr * cp],
        [sr * sy - cr * sp * cy, sr * cy + cr * sp * sy, cr * cp],
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

fn unwrap_i64_counter(values: &[i64], modulus: i64) -> Vec<i64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut offset = 0i64;
    let mut prev = values[0];
    out.push(prev);
    for &v in values.iter().skip(1) {
        if v < prev && (prev - v) > (modulus / 2) {
            offset = offset.saturating_add(modulus);
        }
        out.push(v.saturating_add(offset));
        prev = v;
    }
    out
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

fn set_initial_bias_covariance(ekf: &mut Ekf, dt_nominal_s: f64) {
    // Bias states are delta-angle / delta-velocity increments per predict step.
    // Use tight initial covariance so biases start near zero and settle smoothly.
    let dt = dt_nominal_s.max(1.0e-3);
    let gyro_sigma_dps = 0.15_f64;
    let accel_sigma_mps2 = 0.25_f64;
    let gyro_sigma_da = deg2rad(gyro_sigma_dps) * dt;
    let accel_sigma_dv = accel_sigma_mps2 * dt;
    let var_gyro = (gyro_sigma_da * gyro_sigma_da) as f32;
    let var_accel = (accel_sigma_dv * accel_sigma_dv) as f32;
    ekf.p[10][10] = var_gyro;
    ekf.p[11][11] = var_gyro;
    ekf.p[12][12] = var_gyro;
    ekf.p[13][13] = var_accel;
    ekf.p[14][14] = var_accel;
    ekf.p[15][15] = var_accel;
}

fn run_config(
    cfg: Config,
    imu_packets: &[ImuPacket],
    alg_events: &[AlgEvent],
    nav_events: &[(f64, NavPvtObs)],
    nav_source_is_nav2: bool,
    nav_att_events: &[NavAttEvent],
    window_start_s: f64,
    window_end_s: f64,
) -> Metrics {
    let mut ekf = Ekf::default();
    ekf_init(&mut ekf, cfg.p_init);
    set_initial_bias_covariance(&mut ekf, 0.01);
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
    let mut next_gps_update_ms = f64::NEG_INFINITY;
    let gps_period_ms = 500.0_f64; // Used only when falling back to NAV-PVT.

    let mut n = 0usize;
    let mut se_pos_h = 0.0_f64;
    let mut se_vel_h = 0.0_f64;
    let mut se_yaw = 0.0_f64;
    let mut sae_yaw = 0.0_f64;
    let mut n_yaw_fast = 0usize;
    let mut se_yaw_fast = 0.0_f64;
    let mut sae_yaw_fast = 0.0_f64;
    let mut n_roll = 0usize;
    let mut se_roll = 0.0_f64;
    let mut sae_roll = 0.0_f64;
    let mut n_roll_win = 0usize;
    let mut se_roll_win = 0.0_f64;
    let mut sae_roll_win = 0.0_f64;
    let mut nav_att_idx_metric = 0usize;
    let t0_ms = imu_packets.first().map(|p| p.t_ms).unwrap_or(0.0);

    for pkt in imu_packets {
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

        while nav_att_idx_metric + 1 < nav_att_events.len()
            && nav_att_events[nav_att_idx_metric + 1].t_ms <= pkt.t_ms
        {
            nav_att_idx_metric += 1;
        }
        if nav_att_idx_metric < nav_att_events.len()
            && nav_att_events[nav_att_idx_metric].t_ms <= pkt.t_ms
        {
            let att = nav_att_events[nav_att_idx_metric];
            let (ekf_roll, _, _) =
                quat_rpy_deg(ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3);
            let roll_err = wrap_deg180(ekf_roll - att.roll_deg);
            n_roll += 1;
            se_roll += roll_err * roll_err;
            sae_roll += roll_err.abs();
            let t_rel_s = (pkt.t_ms - t0_ms) * 1e-3;
            if t_rel_s >= window_start_s && t_rel_s <= window_end_s {
                n_roll_win += 1;
                se_roll_win += roll_err * roll_err;
                sae_roll_win += roll_err.abs();
            }
        }

        let mut gyro = [pkt.gx_dps, pkt.gy_dps, pkt.gz_dps];
        let mut accel = [pkt.ax_mps2, pkt.ay_mps2, pkt.az_mps2];
        if let Some(alg) = cur_alg {
            let (roll, pitch, yaw) = match cfg.transform {
                TransformMode::TransposeNegZyx
                | TransformMode::DirectNegZyx
                | TransformMode::RotXyzNeg
                | TransformMode::RotXyzNegTranspose => {
                    (-alg.roll_deg, -alg.pitch_deg, -alg.yaw_deg)
                }
                _ => (alg.roll_deg, alg.pitch_deg, alg.yaw_deg),
            };
            let r_bs = rot_zyx(deg2rad(yaw), deg2rad(pitch), deg2rad(roll));
            let r = match cfg.transform {
                TransformMode::None => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                TransformMode::RotXyz => rot_xyz(deg2rad(roll), deg2rad(pitch), deg2rad(yaw)),
                TransformMode::RotXyzTranspose => {
                    transpose(rot_xyz(deg2rad(roll), deg2rad(pitch), deg2rad(yaw)))
                }
                TransformMode::RotXyzNeg => rot_xyz(deg2rad(roll), deg2rad(pitch), deg2rad(yaw)),
                TransformMode::RotXyzNegTranspose => {
                    transpose(rot_xyz(deg2rad(roll), deg2rad(pitch), deg2rad(yaw)))
                }
                TransformMode::TransposeZyx | TransformMode::TransposeNegZyx => transpose(r_bs),
                TransformMode::DirectZyx | TransformMode::DirectNegZyx => r_bs,
            };
            gyro = mat_vec(r, gyro);
            accel = mat_vec(r, accel);
        }
        match cfg.frame_post_rot {
            FramePostRot::Identity => {}
            FramePostRot::Rx180 => {
                gyro[1] = -gyro[1];
                gyro[2] = -gyro[2];
                accel[1] = -accel[1];
                accel[2] = -accel[2];
            }
            FramePostRot::Ry180 => {
                gyro[0] = -gyro[0];
                gyro[2] = -gyro[2];
                accel[0] = -accel[0];
                accel[2] = -accel[2];
            }
            FramePostRot::Rz180 => {
                gyro[0] = -gyro[0];
                gyro[1] = -gyro[1];
                accel[0] = -accel[0];
                accel[1] = -accel[1];
            }
        }

        let imu = ImuSample {
            dax: (deg2rad(gyro[0] * cfg.gyro_scale) * dt) as f32,
            day: (deg2rad(gyro[1] * cfg.gyro_scale) * dt) as f32,
            daz: (deg2rad(gyro[2] * cfg.gyro_scale) * dt) as f32,
            dvx: (accel[0] * cfg.accel_scale * dt) as f32,
            dvy: (accel[1] * cfg.accel_scale * dt) as f32,
            dvz: (accel[2] * cfg.accel_scale * dt) as f32,
            dt: dt as f32,
        };
        ekf_predict(
            &mut ekf, &imu, 2.5e-4, 1.2e-3, 5.0e-7, 2.0e-6, 2.5e-6, 3.0e-6, None,
        );
        if cfg.use_body_vel {
            let mut r_body = cfg.r_body_vel;
            if cfg.turn_aware_nhc {
                let yaw_rate_dps = gyro[2].abs();
                if yaw_rate_dps > 3.0 {
                    let alpha = ((yaw_rate_dps - 3.0) / 7.0).clamp(0.0, 1.0) as f32;
                    r_body = cfg.r_body_vel * (1.0 + cfg.turn_aware_gain * alpha);
                }
            }
            ekf_fuse_body_vel(&mut ekf, r_body);
        }

        while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
            let (t_ms, nav) = nav_events[nav_idx];
            nav_idx += 1;
            if !nav_source_is_nav2 {
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
                ref_ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
                origin_set = true;
            }

            let ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
            let ned = ecef_to_ned(ecef, ref_ecef, ref_lat, ref_lon);

            let (heading_rad, r_yaw) = match cfg.heading {
                HeadingMode::NavPvtMotion => {
                    let speed_h = nav.vel_n_mps.hypot(nav.vel_e_mps);
                    let mut h = wrap_pi(deg2rad(nav.heading_motion_deg));
                    let mut r = deg2rad(nav.head_acc_deg).powi(2).max(0.02);
                    if speed_h < 1.0 || !r.is_finite() {
                        let (_, _, yaw_deg) =
                            quat_rpy_deg(ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3);
                        h = deg2rad(yaw_deg);
                        r = 1e6;
                    }
                    (h, r)
                }
                HeadingMode::NavAtt => {
                    while nav_att_idx + 1 < nav_att_events.len()
                        && nav_att_events[nav_att_idx + 1].t_ms <= t_ms
                    {
                        nav_att_idx += 1;
                    }
                    let h = if nav_att_idx < nav_att_events.len()
                        && nav_att_events[nav_att_idx].t_ms <= t_ms
                    {
                        deg2rad(nav_att_events[nav_att_idx].heading_deg)
                    } else {
                        deg2rad(nav.heading_motion_deg)
                    };
                    (wrap_pi(h), deg2rad(nav.head_acc_deg).powi(2).max(0.02))
                }
                HeadingMode::Disabled => {
                    let (_, _, yaw_deg) =
                        quat_rpy_deg(ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3);
                    (deg2rad(yaw_deg), 1e12)
                }
            };

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

            let pos_h_err = ((ekf.state.pn as f64 - ned[0]).powi(2)
                + (ekf.state.pe as f64 - ned[1]).powi(2))
            .sqrt();
            let vel_h_err = ((ekf.state.vn as f64 - nav.vel_n_mps).powi(2)
                + (ekf.state.ve as f64 - nav.vel_e_mps).powi(2))
            .sqrt();
            let (_, _, ekf_yaw) =
                quat_rpy_deg(ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3);
            let yaw_ref = match cfg.heading {
                HeadingMode::NavAtt => {
                    if nav_att_idx < nav_att_events.len() {
                        nav_att_events[nav_att_idx].heading_deg
                    } else {
                        nav.heading_motion_deg
                    }
                }
                HeadingMode::NavPvtMotion => nav.heading_motion_deg,
                HeadingMode::Disabled => nav.heading_motion_deg,
            };
            let yaw_err = wrap_deg180(ekf_yaw - yaw_ref);

            n += 1;
            se_pos_h += pos_h_err * pos_h_err;
            se_vel_h += vel_h_err * vel_h_err;
            se_yaw += yaw_err * yaw_err;
            sae_yaw += yaw_err.abs();
            let speed_h = nav.vel_n_mps.hypot(nav.vel_e_mps);
            if speed_h > 2.0 {
                n_yaw_fast += 1;
                se_yaw_fast += yaw_err * yaw_err;
                sae_yaw_fast += yaw_err.abs();
            }
        }
    }

    if n == 0 {
        return Metrics::default();
    }
    Metrics {
        n,
        rmse_pos_h: (se_pos_h / n as f64).sqrt(),
        rmse_vel_h: (se_vel_h / n as f64).sqrt(),
        rmse_yaw_deg: (se_yaw / n as f64).sqrt(),
        mean_abs_yaw_deg: sae_yaw / n as f64,
        rmse_yaw_fast_deg: if n_yaw_fast > 0 {
            (se_yaw_fast / n_yaw_fast as f64).sqrt()
        } else {
            0.0
        },
        mean_abs_yaw_fast_deg: if n_yaw_fast > 0 {
            sae_yaw_fast / n_yaw_fast as f64
        } else {
            0.0
        },
        n_roll,
        rmse_roll_deg: if n_roll > 0 {
            (se_roll / n_roll as f64).sqrt()
        } else {
            0.0
        },
        mean_abs_roll_deg: if n_roll > 0 {
            sae_roll / n_roll as f64
        } else {
            0.0
        },
        n_roll_win,
        rmse_roll_win_deg: if n_roll_win > 0 {
            (se_roll_win / n_roll_win as f64).sqrt()
        } else {
            0.0
        },
        mean_abs_roll_win_deg: if n_roll_win > 0 {
            sae_roll_win / n_roll_win as f64
        } else {
            0.0
        },
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut bytes = Vec::new();
    File::open(&args.logfile)
        .with_context(|| format!("failed to open {}", args.logfile.display()))?
        .read_to_end(&mut bytes)
        .context("failed to read log")?;

    let frames = parse_ubx_frames(&bytes, None);
    let mut masters: Vec<(u64, f64)> = Vec::new();
    for f in &frames {
        if let Some(itow) = extract_itow_ms(f) {
            if (0..604_800_000).contains(&itow) {
                masters.push((f.seq, itow as f64));
            }
        }
    }
    masters.sort_by_key(|x| x.0);
    let raw: Vec<i64> = masters.iter().map(|(_, ms)| *ms as i64).collect();
    let unwrapped = unwrap_i64_counter(&raw, 604_800_000);
    for (m, msu) in masters.iter_mut().zip(unwrapped.into_iter()) {
        m.1 = msu as f64;
    }
    let mut filtered = Vec::with_capacity(masters.len());
    let mut last_ms: Option<f64> = None;
    for (seq, ms) in masters {
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
    let masters = filtered;

    let mut alg_events = Vec::<AlgEvent>::new();
    let mut nav_att_events = Vec::<NavAttEvent>::new();
    let mut nav_events_pvt = Vec::<(f64, NavPvtObs)>::new();
    let mut nav_events_nav2 = Vec::<(f64, NavPvtObs)>::new();
    for f in &frames {
        if let Some((_, roll, pitch, yaw)) = extract_esf_alg(f) {
            if let Some(t_ms) = nearest_master_ms(f.seq, &masters) {
                alg_events.push(AlgEvent {
                    t_ms,
                    roll_deg: roll,
                    pitch_deg: pitch,
                    yaw_deg: yaw,
                });
            }
        }
        if let Some((_itow, roll, pitch, heading)) = extract_nav_att(f) {
            if let Some(t_ms) = nearest_master_ms(f.seq, &masters) {
                nav_att_events.push(NavAttEvent {
                    t_ms,
                    roll_deg: roll,
                    pitch_deg: pitch,
                    heading_deg: normalize_heading_deg(heading),
                });
            }
        }
        if let Some(t_ms) = nearest_master_ms(f.seq, &masters) {
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
    nav_att_events.sort_by(|a, b| {
        a.t_ms
            .partial_cmp(&b.t_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    nav_events_nav2.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    nav_events_pvt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let (nav_events, nav_source_is_nav2) = if !nav_events_nav2.is_empty() {
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
    for f in &frames {
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
        if let Some(ms) = nearest_master_ms(*seq, &masters) {
            x.push(*tag_u as f64);
            y.push(ms);
        }
    }
    let (a_raw, b_raw) = fit_linear_map(&x, &y, 1e-3);
    let master_min = masters
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::INFINITY, f64::min);
    let master_max = masters
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::NEG_INFINITY, f64::max);

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
            if let Some(seq_ms) = nearest_master_ms(*seq, &masters) {
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

    let configs = vec![
        Config {
            name: "xyz_id_r100",
            transform: TransformMode::RotXyz,
            heading: HeadingMode::Disabled,
            p_init: 1.0,
            use_body_vel: true,
            r_body_vel: 100.0,
            turn_aware_nhc: false,
            turn_aware_gain: 0.0,
            gyro_scale: 1.0,
            accel_scale: 1.0,
            frame_post_rot: FramePostRot::Identity,
        },
        Config {
            name: "xyz_rx_r100",
            transform: TransformMode::RotXyz,
            heading: HeadingMode::Disabled,
            p_init: 1.0,
            use_body_vel: true,
            r_body_vel: 100.0,
            turn_aware_nhc: false,
            turn_aware_gain: 0.0,
            gyro_scale: 1.0,
            accel_scale: 1.0,
            frame_post_rot: FramePostRot::Rx180,
        },
        Config {
            name: "xyz_ry_r100",
            transform: TransformMode::RotXyz,
            heading: HeadingMode::Disabled,
            p_init: 1.0,
            use_body_vel: true,
            r_body_vel: 100.0,
            turn_aware_nhc: false,
            turn_aware_gain: 0.0,
            gyro_scale: 1.0,
            accel_scale: 1.0,
            frame_post_rot: FramePostRot::Ry180,
        },
        Config {
            name: "xyz_rz_r100",
            transform: TransformMode::RotXyz,
            heading: HeadingMode::Disabled,
            p_init: 1.0,
            use_body_vel: true,
            r_body_vel: 100.0,
            turn_aware_nhc: false,
            turn_aware_gain: 0.0,
            gyro_scale: 1.0,
            accel_scale: 1.0,
            frame_post_rot: FramePostRot::Rz180,
        },
        Config {
            name: "xyzT_id_r100",
            transform: TransformMode::RotXyzTranspose,
            heading: HeadingMode::Disabled,
            p_init: 1.0,
            use_body_vel: true,
            r_body_vel: 100.0,
            turn_aware_nhc: false,
            turn_aware_gain: 0.0,
            gyro_scale: 1.0,
            accel_scale: 1.0,
            frame_post_rot: FramePostRot::Identity,
        },
        Config {
            name: "xyzT_rx_r100",
            transform: TransformMode::RotXyzTranspose,
            heading: HeadingMode::Disabled,
            p_init: 1.0,
            use_body_vel: true,
            r_body_vel: 100.0,
            turn_aware_nhc: false,
            turn_aware_gain: 0.0,
            gyro_scale: 1.0,
            accel_scale: 1.0,
            frame_post_rot: FramePostRot::Rx180,
        },
        Config {
            name: "dzyx_id_r100",
            transform: TransformMode::DirectZyx,
            heading: HeadingMode::Disabled,
            p_init: 1.0,
            use_body_vel: true,
            r_body_vel: 100.0,
            turn_aware_nhc: false,
            turn_aware_gain: 0.0,
            gyro_scale: 1.0,
            accel_scale: 1.0,
            frame_post_rot: FramePostRot::Identity,
        },
        Config {
            name: "dzyx_rx_r100",
            transform: TransformMode::DirectZyx,
            heading: HeadingMode::Disabled,
            p_init: 1.0,
            use_body_vel: true,
            r_body_vel: 100.0,
            turn_aware_nhc: false,
            turn_aware_gain: 0.0,
            gyro_scale: 1.0,
            accel_scale: 1.0,
            frame_post_rot: FramePostRot::Rx180,
        },
        Config {
            name: "dzyx_ry_r100",
            transform: TransformMode::DirectZyx,
            heading: HeadingMode::Disabled,
            p_init: 1.0,
            use_body_vel: true,
            r_body_vel: 100.0,
            turn_aware_nhc: false,
            turn_aware_gain: 0.0,
            gyro_scale: 1.0,
            accel_scale: 1.0,
            frame_post_rot: FramePostRot::Ry180,
        },
        Config {
            name: "dzyx_rz_r100",
            transform: TransformMode::DirectZyx,
            heading: HeadingMode::Disabled,
            p_init: 1.0,
            use_body_vel: true,
            r_body_vel: 100.0,
            turn_aware_nhc: false,
            turn_aware_gain: 0.0,
            gyro_scale: 1.0,
            accel_scale: 1.0,
            frame_post_rot: FramePostRot::Rz180,
        },
        Config {
            name: "tzyx_rx_r100",
            transform: TransformMode::TransposeZyx,
            heading: HeadingMode::Disabled,
            p_init: 1.0,
            use_body_vel: true,
            r_body_vel: 100.0,
            turn_aware_nhc: false,
            turn_aware_gain: 0.0,
            gyro_scale: 1.0,
            accel_scale: 1.0,
            frame_post_rot: FramePostRot::Rx180,
        },
    ];

    println!(
        "window: [{:.3}, {:.3}] s (relative)\n{:<22} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}",
        args.window_start_s,
        args.window_end_s,
        "config",
        "N",
        "pos_h",
        "vel_h",
        "yaw",
        "roll",
        "roll_mae",
        "roll_w",
        "rollw_mae",
    );
    let mut scored = Vec::new();
    for cfg in configs {
        let m = run_config(
            cfg,
            &imu_packets,
            &alg_events,
            &nav_events,
            nav_source_is_nav2,
            &nav_att_events,
            args.window_start_s,
            args.window_end_s,
        );
        println!(
            "{:<22} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>9.3} {:>9.3}",
            cfg.name,
            m.n,
            m.rmse_pos_h,
            m.rmse_vel_h,
            m.rmse_yaw_deg,
            m.rmse_roll_deg,
            m.mean_abs_roll_deg,
            m.rmse_roll_win_deg,
            m.mean_abs_roll_win_deg
        );
        let score = m.rmse_roll_win_deg * 4.0 + m.rmse_roll_deg * 2.0 + m.rmse_pos_h * 0.2;
        scored.push((score, cfg.name, m));
    }
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if let Some((score, name, m)) = scored.first() {
        println!(
            "\nBEST: {} score={:.3} roll_win_rmse={:.3} roll_win_mae={:.3} roll_rmse={:.3} pos_h={:.3}",
            name,
            score,
            m.rmse_roll_win_deg,
            m.mean_abs_roll_win_deg,
            m.rmse_roll_deg,
            m.rmse_pos_h
        );
    }
    Ok(())
}
