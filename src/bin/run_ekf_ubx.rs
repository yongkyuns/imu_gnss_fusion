use std::{f64::consts::PI, fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use ekf_rs::ekf::{Ekf, GpsData, ImuSample, ekf_fuse_gps, ekf_init, ekf_predict};
use pygps_rs::ubxlog::{
    NavPvtObs, extract_esf_alg, extract_esf_raw_samples, extract_itow_ms, extract_nav_pvt_obs, fit_linear_map,
    parse_ubx_frames, sensor_meta, unwrap_counter,
};

const USE_MOUNT_TRANSPOSE: bool = true;

#[derive(Parser, Debug)]
#[command(name = "run_ekf_ubx")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long, default_value_t = 2.0)]
    gps_update_hz: f64,
    #[arg(long, default_value_t = 1.0)]
    heading_min_speed_mps: f64,
    #[arg(long, default_value_t = 1.0)]
    p_init: f32,
    #[arg(long, default_value_t = 2.5e-4)]
    da_var: f32,
    #[arg(long, default_value_t = 1.2e-3)]
    dv_var: f32,
    #[arg(long, default_value_t = 5.0e-7)]
    dgb_p_noise_var: f32,
    #[arg(long, default_value_t = 2.0e-6)]
    dvb_x_p_noise_var: f32,
    #[arg(long, default_value_t = 2.5e-6)]
    dvb_y_p_noise_var: f32,
    #[arg(long, default_value_t = 3.0e-6)]
    dvb_z_p_noise_var: f32,
    #[arg(long, default_value_t = 0.05)]
    r_pos_floor_m2: f64,
    #[arg(long, default_value_t = 0.02)]
    r_vel_floor_m2s2: f64,
    #[arg(long, default_value_t = 0.02)]
    r_yaw_floor_rad2: f64,
    #[arg(long)]
    no_mount_rotation: bool,
}

#[derive(Clone, Copy, Debug)]
struct AlgEvent {
    t_ms: f64,
    roll_deg: f64,
    pitch_deg: f64,
    yaw_deg: f64,
}

#[derive(Clone, Copy, Debug)]
struct NavEvent {
    t_ms: f64,
    obs: NavPvtObs,
}

#[derive(Default)]
struct RawAccum {
    t_ms: f64,
    gx_dps: Option<f64>,
    gy_dps: Option<f64>,
    gz_dps: Option<f64>,
    ax_mps2: Option<f64>,
    ay_mps2: Option<f64>,
    az_mps2: Option<f64>,
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

#[derive(Default, Debug)]
struct InnovationStats {
    count: usize,
    sum_r_pos_h2: f64,
    sum_r_vel_h2: f64,
    sum_r_yaw2: f64,
    sum_nis_pos_h: f64,
    sum_nis_vel_h: f64,
    sum_nis_yaw: f64,
}

impl InnovationStats {
    fn push(&mut self, r_pos_n: f64, r_pos_e: f64, r_vel_n: f64, r_vel_e: f64, r_yaw: f64, r_pos_var: f64, r_vel_var: f64, r_yaw_var: f64) {
        self.count += 1;
        let pos_h2 = r_pos_n * r_pos_n + r_pos_e * r_pos_e;
        let vel_h2 = r_vel_n * r_vel_n + r_vel_e * r_vel_e;
        self.sum_r_pos_h2 += pos_h2;
        self.sum_r_vel_h2 += vel_h2;
        self.sum_r_yaw2 += r_yaw * r_yaw;
        if r_pos_var.is_finite() && r_pos_var > 0.0 {
            self.sum_nis_pos_h += pos_h2 / r_pos_var;
        }
        if r_vel_var.is_finite() && r_vel_var > 0.0 {
            self.sum_nis_vel_h += vel_h2 / r_vel_var;
        }
        if r_yaw_var.is_finite() && r_yaw_var > 0.0 {
            self.sum_nis_yaw += (r_yaw * r_yaw) / r_yaw_var;
        }
    }
}

#[derive(Default, Debug)]
struct GainStats {
    count: usize,
    sum_k_pn_from_posn: f64,
    sum_k_pe_from_pose: f64,
    sum_k_vn_from_veln: f64,
    sum_k_ve_from_vele: f64,
}

impl GainStats {
    fn push(&mut self, k_pn: f64, k_pe: f64, k_vn: f64, k_ve: f64) {
        self.count += 1;
        self.sum_k_pn_from_posn += k_pn;
        self.sum_k_pe_from_pose += k_pe;
        self.sum_k_vn_from_veln += k_vn;
        self.sum_k_ve_from_vele += k_ve;
    }
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

fn build_master_timeline(frames: &[pygps_rs::ubxlog::UbxFrame]) -> Vec<(u64, f64)> {
    let mut masters: Vec<(u64, f64)> = Vec::new();
    for f in frames {
        if let Some(itow) = extract_itow_ms(f) {
            if (0..604_800_000).contains(&itow) {
                masters.push((f.seq, itow as f64));
            }
        }
    }
    masters.sort_by_key(|x| x.0);
    if masters.is_empty() {
        return masters;
    }

    let raw: Vec<i64> = masters.iter().map(|(_, ms)| *ms as i64).collect();
    let unwrapped = unwrap_i64_counter(&raw, 604_800_000);
    for (m, msu) in masters.iter_mut().zip(unwrapped.into_iter()) {
        m.1 = msu as f64;
    }

    // Keep one monotonic epoch and drop jumps/resets.
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
    filtered
}

fn deg2rad(v: f64) -> f64 {
    v * PI / 180.0
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

fn wrap_pi(mut a: f64) -> f64 {
    while a > PI {
        a -= 2.0 * PI;
    }
    while a < -PI {
        a += 2.0 * PI;
    }
    a
}

fn quat_yaw(q0: f32, q1: f32, q2: f32, q3: f32) -> f32 {
    let r10 = 2.0_f32 * (q1 * q2 + q0 * q3);
    let r00 = 1.0_f32 - 2.0_f32 * (q2 * q2 + q3 * q3);
    r10.atan2(r00)
}

fn lla_to_ecef(lat_deg: f64, lon_deg: f64, h_m: f64) -> [f64; 3] {
    let a = 6378137.0_f64;
    let e2 = 6.69437999014e-3_f64;
    let lat = deg2rad(lat_deg);
    let lon = deg2rad(lon_deg);
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    let n = a / (1.0 - e2 * slat * slat).sqrt();
    let x = (n + h_m) * clat * clon;
    let y = (n + h_m) * clat * slon;
    let z = (n * (1.0 - e2) + h_m) * slat;
    [x, y, z]
}

fn ecef_to_ned(ecef: [f64; 3], ref_ecef: [f64; 3], ref_lat_deg: f64, ref_lon_deg: f64) -> [f64; 3] {
    let lat = deg2rad(ref_lat_deg);
    let lon = deg2rad(ref_lon_deg);
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    let dx = ecef[0] - ref_ecef[0];
    let dy = ecef[1] - ref_ecef[1];
    let dz = ecef[2] - ref_ecef[2];
    let n = -slat * clon * dx - slat * slon * dy + clat * dz;
    let e = -slon * dx + clon * dy;
    let d = -clat * clon * dx - clat * slon * dy - slat * dz;
    [n, e, d]
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut bytes = Vec::new();
    File::open(&args.logfile)
        .with_context(|| format!("failed to open {}", args.logfile.display()))?
        .read_to_end(&mut bytes)
        .context("failed to read log")?;

    let frames = parse_ubx_frames(&bytes, args.max_records);
    let masters = build_master_timeline(&frames);
    if masters.is_empty() {
        anyhow::bail!("no valid iTOW timeline found");
    }
    let t0_ms = masters
        .iter()
        .map(|(_, t)| *t)
        .fold(f64::INFINITY, f64::min);

    // ESF-ALG events for mount rotation (hold-last).
    let mut alg_events = Vec::<AlgEvent>::new();
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
    }
    alg_events.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap_or(std::cmp::Ordering::Equal));

    // NAV events (30Hz source).
    let mut nav_events = Vec::<NavEvent>::new();
    for f in &frames {
        if let Some(obs) = extract_nav_pvt_obs(f) {
            if let Some(t_ms) = nearest_master_ms(f.seq, &masters) {
                nav_events.push(NavEvent { t_ms, obs });
            }
        }
    }
    nav_events.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap_or(std::cmp::Ordering::Equal));

    // RAW samples.
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
    let master_min = masters.iter().map(|(_, ms)| *ms).fold(f64::INFINITY, f64::min);
    let master_max = masters
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut imu_packets = Vec::<ImuPacket>::new();
    let mut current_tag: Option<u64> = None;
    let mut acc = RawAccum::default();
    for (((seq, tag_u), dtype), val) in raw_seq
        .iter()
        .zip(raw_tag_u.iter())
        .zip(raw_dtype.iter())
        .zip(raw_val.iter())
    {
        if current_tag != Some(*tag_u) {
            if let (Some(gx), Some(gy), Some(gz), Some(ax), Some(ay), Some(az)) =
                (acc.gx_dps, acc.gy_dps, acc.gz_dps, acc.ax_mps2, acc.ay_mps2, acc.az_mps2)
            {
                imu_packets.push(ImuPacket {
                    t_ms: acc.t_ms,
                    gx_dps: gx,
                    gy_dps: gy,
                    gz_dps: gz,
                    ax_mps2: ax,
                    ay_mps2: ay,
                    az_mps2: az,
                });
            }
            acc = RawAccum::default();
            current_tag = Some(*tag_u);
            let seq_ms = nearest_master_ms(*seq, &masters).unwrap_or(t0_ms);
            let mut mapped_ms = a_raw * *tag_u as f64 + b_raw;
            if !mapped_ms.is_finite()
                || mapped_ms < master_min - 1000.0
                || mapped_ms > master_max + 1000.0
                || (mapped_ms - seq_ms).abs() > 2000.0
            {
                mapped_ms = seq_ms;
            }
            acc.t_ms = mapped_ms;
        }

        match *dtype {
            14 => acc.gx_dps = Some(*val),
            13 => acc.gy_dps = Some(*val),
            5 => acc.gz_dps = Some(*val),
            16 => acc.ax_mps2 = Some(*val),
            17 => acc.ay_mps2 = Some(*val),
            18 => acc.az_mps2 = Some(*val),
            _ => {}
        }
    }
    if let (Some(gx), Some(gy), Some(gz), Some(ax), Some(ay), Some(az)) =
        (acc.gx_dps, acc.gy_dps, acc.gz_dps, acc.ax_mps2, acc.ay_mps2, acc.az_mps2)
    {
        imu_packets.push(ImuPacket {
            t_ms: acc.t_ms,
            gx_dps: gx,
            gy_dps: gy,
            gz_dps: gz,
            ax_mps2: ax,
            ay_mps2: ay,
            az_mps2: az,
        });
    }
    imu_packets.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap_or(std::cmp::Ordering::Equal));

    // Downsample NAV updates to requested rate (2Hz default).
    let gps_period_ms = 1000.0 / args.gps_update_hz.max(0.1);
    let mut nav_updates = Vec::<NavEvent>::new();
    let mut next_update_ms = f64::NEG_INFINITY;
    for ev in nav_events {
        if !ev.obs.fix_ok || ev.obs.invalid_llh {
            continue;
        }
        if !next_update_ms.is_finite() {
            next_update_ms = ev.t_ms;
        }
        if ev.t_ms + 1e-6 >= next_update_ms {
            nav_updates.push(ev);
            next_update_ms += gps_period_ms;
        }
    }

    // EKF run.
    let mut ekf = Ekf::default();
    ekf_init(&mut ekf, args.p_init);
    ekf.state.q0 = 1.0;

    let mut alg_idx = 0usize;
    let mut current_alg: Option<AlgEvent> = None;
    let mut nav_idx = 0usize;
    let mut prev_imu_t_ms: Option<f64> = None;

    let mut origin_set = false;
    let mut ref_lat = 0.0_f64;
    let mut ref_lon = 0.0_f64;
    let mut ref_ecef = [0.0_f64; 3];

    let mut n_predict = 0usize;
    let mut n_gps_updates = 0usize;
    let mut innov = InnovationStats::default();
    let mut gains = GainStats::default();

    for pkt in &imu_packets {
        while alg_idx < alg_events.len() && alg_events[alg_idx].t_ms <= pkt.t_ms {
            current_alg = Some(alg_events[alg_idx]);
            alg_idx += 1;
        }

        let dt = match prev_imu_t_ms {
            Some(t_prev) => (pkt.t_ms - t_prev) * 1e-3,
            None => {
                prev_imu_t_ms = Some(pkt.t_ms);
                continue;
            }
        };
        if !(0.001..=0.05).contains(&dt) {
            prev_imu_t_ms = Some(pkt.t_ms);
            continue;
        }
        prev_imu_t_ms = Some(pkt.t_ms);

        let mut gyro = [pkt.gx_dps, pkt.gy_dps, pkt.gz_dps];
        let mut accel = [pkt.ax_mps2, pkt.ay_mps2, pkt.az_mps2];
        if !args.no_mount_rotation {
            if let Some(alg) = current_alg {
                // UBX-ESF-ALG provides intrinsic ZYX (yaw->pitch->roll) angles for
                // installation(body) -> sensor(IMU): v_s = R_bs * v_b.
                // ESF-RAW is measured in sensor frame, while this EKF expects body frame,
                // so use the inverse mapping: v_b = R_sb * v_s, with R_sb = R_bs^T.
                let r_bs = rot_zyx(deg2rad(alg.yaw_deg), deg2rad(alg.pitch_deg), deg2rad(alg.roll_deg));
                let r = if USE_MOUNT_TRANSPOSE { transpose(r_bs) } else { r_bs };
                gyro = mat_vec(r, gyro);
                accel = mat_vec(r, accel);
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

        ekf_predict(
            &mut ekf,
            &imu,
            args.da_var,
            args.dv_var,
            args.dgb_p_noise_var,
            args.dvb_x_p_noise_var,
            args.dvb_y_p_noise_var,
            args.dvb_z_p_noise_var,
            None,
        );
        n_predict += 1;

        while nav_idx < nav_updates.len() && nav_updates[nav_idx].t_ms <= pkt.t_ms {
            let nav = nav_updates[nav_idx];
            nav_idx += 1;

            if !origin_set {
                ref_lat = nav.obs.lat_deg;
                ref_lon = nav.obs.lon_deg;
                ref_ecef = lla_to_ecef(nav.obs.lat_deg, nav.obs.lon_deg, nav.obs.height_m);
                origin_set = true;
            }
            let ecef = lla_to_ecef(nav.obs.lat_deg, nav.obs.lon_deg, nav.obs.height_m);
            let ned = ecef_to_ned(ecef, ref_ecef, ref_lat, ref_lon);

            let h_acc2 = (nav.obs.h_acc_m * nav.obs.h_acc_m).max(args.r_pos_floor_m2);
            let v_acc2 = (nav.obs.v_acc_m * nav.obs.v_acc_m).max(args.r_pos_floor_m2);
            let s_acc2 = (nav.obs.s_acc_mps * nav.obs.s_acc_mps).max(args.r_vel_floor_m2s2);
            let heading_ok = nav.obs.vel_n_mps.hypot(nav.obs.vel_e_mps) >= args.heading_min_speed_mps;

            let mut heading_rad = wrap_pi(deg2rad(nav.obs.heading_motion_deg));
            let mut r_yaw = (deg2rad(nav.obs.head_acc_deg).powi(2)).max(args.r_yaw_floor_rad2);
            if !heading_ok || !r_yaw.is_finite() {
                heading_rad = quat_yaw(ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3) as f64;
                r_yaw = 1e6;
            }

            let yaw_pred = quat_yaw(ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3) as f64;
            innov.push(
                ned[0] - ekf.state.pn as f64,
                ned[1] - ekf.state.pe as f64,
                nav.obs.vel_n_mps - ekf.state.vn as f64,
                nav.obs.vel_e_mps - ekf.state.ve as f64,
                wrap_pi(heading_rad - yaw_pred),
                h_acc2,
                s_acc2,
                r_yaw,
            );

            let gps = GpsData {
                pos_n: ned[0] as f32,
                pos_e: ned[1] as f32,
                pos_d: ned[2] as f32,
                vel_n: nav.obs.vel_n_mps as f32,
                vel_e: nav.obs.vel_e_mps as f32,
                vel_d: nav.obs.vel_d_mps as f32,
                heading_rad: heading_rad as f32,
                R_POS_N: h_acc2 as f32,
                R_POS_E: h_acc2 as f32,
                R_POS_D: v_acc2 as f32,
                R_VEL_N: s_acc2 as f32,
                R_VEL_E: s_acc2 as f32,
                R_VEL_D: s_acc2 as f32,
                R_YAW: r_yaw as f32,
            };

            let p = &ekf.p;
            let k_pn = p[7][7] as f64 / (p[7][7] as f64 + h_acc2);
            let k_pe = p[8][8] as f64 / (p[8][8] as f64 + h_acc2);
            let k_vn = p[4][4] as f64 / (p[4][4] as f64 + s_acc2);
            let k_ve = p[5][5] as f64 / (p[5][5] as f64 + s_acc2);
            if k_pn.is_finite() && k_pe.is_finite() && k_vn.is_finite() && k_ve.is_finite() {
                gains.push(k_pn, k_pe, k_vn, k_ve);
            }

            ekf_fuse_gps(&mut ekf, &gps);
            n_gps_updates += 1;
        }
    }

    let duration_s = (masters.last().unwrap().1 - t0_ms) * 1e-3;
    println!("EKF run summary");
    println!("  file: {}", args.logfile.display());
    println!("  duration_s: {:.3}", duration_s);
    println!(
        "  predict_count: {} (~{:.2} Hz)",
        n_predict,
        n_predict as f64 / duration_s.max(1e-3)
    );
    println!(
        "  gps_update_count: {} (~{:.2} Hz, target {:.2} Hz)",
        n_gps_updates,
        n_gps_updates as f64 / duration_s.max(1e-3),
        args.gps_update_hz
    );
    println!(
        "  final_state: q=({:.4},{:.4},{:.4},{:.4}) v_ned=({:.3},{:.3},{:.3}) p_ned=({:.3},{:.3},{:.3})",
        ekf.state.q0,
        ekf.state.q1,
        ekf.state.q2,
        ekf.state.q3,
        ekf.state.vn,
        ekf.state.ve,
        ekf.state.vd,
        ekf.state.pn,
        ekf.state.pe,
        ekf.state.pd,
    );
    if innov.count > 0 {
        let n = innov.count as f64;
        println!(
            "  innovation_rms: pos_h={:.3} m vel_h={:.3} m/s yaw={:.3} deg",
            (innov.sum_r_pos_h2 / n).sqrt(),
            (innov.sum_r_vel_h2 / n).sqrt(),
            (innov.sum_r_yaw2 / n).sqrt() * 180.0 / PI
        );
        println!(
            "  innovation_nis_mean: pos_h={:.3} vel_h={:.3} yaw={:.3}",
            innov.sum_nis_pos_h / n,
            innov.sum_nis_vel_h / n,
            innov.sum_nis_yaw / n
        );
    }
    if gains.count > 0 {
        let n = gains.count as f64;
        println!(
            "  mean_gains: Kpn(posN)={:.4} Kpe(posE)={:.4} Kvn(velN)={:.4} Kve(velE)={:.4}",
            gains.sum_k_pn_from_posn / n,
            gains.sum_k_pe_from_pose / n,
            gains.sum_k_vn_from_veln / n,
            gains.sum_k_ve_from_vele / n
        );
    }
    println!(
        "  mount_rotation: {}",
        if USE_MOUNT_TRANSPOSE { "R_sb=transpose(R_bs)" } else { "R_sb=R_bs" }
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PI, deg2rad, mat_vec, rot_zyx, transpose};

    fn assert_close(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() <= eps, "expected {a} ~= {b}, eps={eps}");
    }

    fn assert_vec_close(a: [f64; 3], b: [f64; 3], eps: f64) {
        assert_close(a[0], b[0], eps);
        assert_close(a[1], b[1], eps);
        assert_close(a[2], b[2], eps);
    }

    #[test]
    fn esf_alg_intrinsic_zyx_yaw_90_body_to_sensor() {
        let r_bs = rot_zyx(PI / 2.0, 0.0, 0.0);
        let body_x = [1.0, 0.0, 0.0];
        let sensor = mat_vec(r_bs, body_x);
        assert_vec_close(sensor, [0.0, 1.0, 0.0], 1e-12);
    }

    #[test]
    fn esf_alg_sensor_to_body_uses_transpose_inverse() {
        let r_bs = rot_zyx(deg2rad(35.0), deg2rad(-12.0), deg2rad(7.5));
        let r_sb = transpose(r_bs);
        let sensor_v = [0.3, -1.4, 2.2];
        let roundtrip = mat_vec(r_bs, mat_vec(r_sb, sensor_v));
        assert_vec_close(roundtrip, sensor_v, 1e-12);
    }
}
