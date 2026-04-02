use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::c_api::{CEskfImuDelta, CEskfWrapper};
use sensor_fusion::ekf::PredictNoise;
use sensor_fusion::fusion::{FusionConfig, FusionGnssSample, FusionImuSample, SensorFusion};
use sim::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_raw_samples, extract_nav2_pvt_obs, extract_nav_pvt_obs,
    fit_linear_map, parse_ubx_frames, sensor_meta, unwrap_counter,
};
use sim::visualizer::math::nearest_master_ms;
use sim::visualizer::model::ImuPacket;
use sim::visualizer::pipeline::timebase::{MasterTimeline, build_master_timeline};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "test_files.txt")]
    file_list: PathBuf,

    #[arg(long, default_value = "Baseline")]
    section: String,

    #[arg(long, default_value = "logger/data")]
    data_dir: PathBuf,

    #[arg(long)]
    r_body_vel: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct DiffMetrics {
    samples: usize,
    rms_pos_m: f64,
    rms_vel_mps: f64,
    rms_att_deg: f64,
    rms_bias_g: f64,
    rms_bias_a: f64,
    final_pos_m: f64,
    final_vel_mps: f64,
    final_att_deg: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let files = parse_section_files(&args.file_list, &args.section)?;
    if files.is_empty() {
        bail!("no files found in section {}", args.section);
    }

    let mut cfg = FusionConfig::default();
    if let Some(r_body_vel) = args.r_body_vel {
        cfg.r_body_vel = r_body_vel;
    }
    println!("file,ekf_init_t_s,samples,rms_pos_m,rms_vel_mps,rms_att_deg,rms_bias_g,rms_bias_a,final_pos_m,final_vel_mps,final_att_deg");
    for file in files {
        let path = args.data_dir.join(&file);
        let data = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let frames = parse_ubx_frames(&data, None);
        let tl = build_master_timeline(&frames);
        let metrics = compare_file(&frames, &tl, cfg);
        println!(
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            file,
            fmt_opt(metrics.0),
            metrics.1.samples,
            metrics.1.rms_pos_m,
            metrics.1.rms_vel_mps,
            metrics.1.rms_att_deg,
            metrics.1.rms_bias_g,
            metrics.1.rms_bias_a,
            metrics.1.final_pos_m,
            metrics.1.final_vel_mps,
            metrics.1.final_att_deg,
        );
    }
    Ok(())
}

fn compare_file(frames: &[UbxFrame], tl: &MasterTimeline, cfg: FusionConfig) -> (Option<f64>, DiffMetrics) {
    let nav_events = build_nav_events(frames, tl);
    let imu_packets = build_imu_packets(frames, tl);
    let ref_nav = nav_events.first().map(|(_, nav)| *nav);

    let mut fusion = SensorFusion::new(cfg);
    let mut eskf: Option<CEskfWrapper> = None;
    let mut mount_q_vb: Option<[f32; 4]> = None;
    let mut scan_idx = 0usize;
    let mut init_t_s = None;
    let mut diffs = Vec::new();

    for (tn, nav) in &nav_events {
        while scan_idx < imu_packets.len() && imu_packets[scan_idx].t_ms <= *tn {
            let pkt = &imu_packets[scan_idx];
            let imu_sample = FusionImuSample {
                t_s: ((pkt.t_ms - tl.t0_master_ms) * 1.0e-3) as f32,
                gyro_radps: [
                    pkt.gx_dps.to_radians() as f32,
                    pkt.gy_dps.to_radians() as f32,
                    pkt.gz_dps.to_radians() as f32,
                ],
                accel_mps2: [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32],
            };
            let update = fusion.process_imu(imu_sample);
            if mount_q_vb.is_none() {
                mount_q_vb = update.mount_q_vb.or_else(|| fusion.mount_q_vb());
            }

            if let (Some(q_vb), Some(eskf_ref)) = (mount_q_vb, eskf.as_mut()) {
                if let Some(delta) = imu_to_vehicle_delta(q_vb, pkt, tl, scan_idx, &imu_packets) {
                    eskf_ref.predict(delta);
                    if cfg.r_body_vel > 0.0 {
                        eskf_ref.fuse_body_vel(cfg.r_body_vel);
                    }
                }
            }

            if let (Some(ekf_ref), Some(eskf_ref)) = (fusion.ekf(), eskf.as_ref()) {
                diffs.push(diff_metrics(ekf_ref, eskf_ref));
            }
            scan_idx += 1;
        }

        let t_s = (*tn - tl.t0_master_ms) * 1.0e-3;
        let gnss = FusionGnssSample {
            t_s: t_s as f32,
            pos_ned_m: nav_to_ned(*nav, ref_nav),
            vel_ned_mps: [nav.vel_n_mps as f32, nav.vel_e_mps as f32, nav.vel_d_mps as f32],
            pos_std_m: [nav.h_acc_m as f32, nav.h_acc_m as f32, nav.v_acc_m as f32],
            vel_std_mps: [nav.s_acc_mps as f32, nav.s_acc_mps as f32, nav.s_acc_mps as f32],
            heading_rad: if nav.head_veh_valid {
                Some(nav.heading_vehicle_deg.to_radians() as f32)
            } else {
                Some(nav.heading_motion_deg.to_radians() as f32)
            },
        };
        let update = fusion.process_gnss(gnss);
        if mount_q_vb.is_none() {
            mount_q_vb = update.mount_q_vb.or_else(|| fusion.mount_q_vb());
        }
        if update.ekf_initialized_now && eskf.is_none() {
            let mut e = CEskfWrapper::new(PredictNoise::lsm6dso_typical_104hz());
            let q_bn = yaw_quat(initial_yaw_rad(&gnss, cfg.yaw_init_speed_mps));
            e.init_nominal_from_gnss(q_bn, gnss);
            eskf = Some(e);
            init_t_s = Some(t_s);
        } else if let Some(eskf_ref) = eskf.as_mut() {
            eskf_ref.fuse_gps(gnss);
        }
        if let (Some(ekf_ref), Some(eskf_ref)) = (fusion.ekf(), eskf.as_ref()) {
            diffs.push(diff_metrics(ekf_ref, eskf_ref));
        }
    }

    (init_t_s, summarize_diffs(&diffs))
}

fn imu_to_vehicle_delta(
    q_vb: [f32; 4],
    pkt: &ImuPacket,
    tl: &MasterTimeline,
    idx: usize,
    packets: &[ImuPacket],
) -> Option<CEskfImuDelta> {
    if idx == 0 {
        return None;
    }
    let dt_s = ((pkt.t_ms - packets[idx - 1].t_ms) * 1.0e-3) as f32;
    if !(0.001..=0.05).contains(&dt_s) {
        return None;
    }
    let c_bv = quat_to_rotmat_f32(q_vb);
    let c_vb = transpose3(c_bv);
    let gyro_b = [
        pkt.gx_dps.to_radians() as f32,
        pkt.gy_dps.to_radians() as f32,
        pkt.gz_dps.to_radians() as f32,
    ];
    let accel_b = [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32];
    let gyro_v = mat3_vec(c_vb, gyro_b);
    let accel_v = mat3_vec(c_vb, accel_b);
    let _ = tl;
    Some(CEskfImuDelta {
        dax: gyro_v[0] * dt_s,
        day: gyro_v[1] * dt_s,
        daz: gyro_v[2] * dt_s,
        dvx: accel_v[0] * dt_s,
        dvy: accel_v[1] * dt_s,
        dvz: accel_v[2] * dt_s,
        dt: dt_s,
    })
}

fn diff_metrics(ekf: &sensor_fusion::ekf::Ekf, eskf: &CEskfWrapper) -> DiffSample {
    let n = eskf.nominal();
    DiffSample {
        pos_m: norm3([
            (ekf.state.pn - n.pn) as f64,
            (ekf.state.pe - n.pe) as f64,
            (ekf.state.pd - n.pd) as f64,
        ]),
        vel_mps: norm3([
            (ekf.state.vn - n.vn) as f64,
            (ekf.state.ve - n.ve) as f64,
            (ekf.state.vd - n.vd) as f64,
        ]),
        att_deg: quat_angle_deg(
            [ekf.state.q0 as f64, ekf.state.q1 as f64, ekf.state.q2 as f64, ekf.state.q3 as f64],
            [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64],
        ),
        bias_g: norm3([
            (ekf.state.dax_b / 0.01 - n.bgx) as f64,
            (ekf.state.day_b / 0.01 - n.bgy) as f64,
            (ekf.state.daz_b / 0.01 - n.bgz) as f64,
        ]),
        bias_a: norm3([
            (ekf.state.dvx_b / 0.01 - n.bax) as f64,
            (ekf.state.dvy_b / 0.01 - n.bay) as f64,
            (ekf.state.dvz_b / 0.01 - n.baz) as f64,
        ]),
    }
}

#[derive(Clone, Copy, Default)]
struct DiffSample {
    pos_m: f64,
    vel_mps: f64,
    att_deg: f64,
    bias_g: f64,
    bias_a: f64,
}

fn summarize_diffs(diffs: &[DiffSample]) -> DiffMetrics {
    if diffs.is_empty() {
        return DiffMetrics::default();
    }
    let mut out = DiffMetrics::default();
    out.samples = diffs.len();
    let mut s_pos = 0.0;
    let mut s_vel = 0.0;
    let mut s_att = 0.0;
    let mut s_bg = 0.0;
    let mut s_ba = 0.0;
    for d in diffs {
        s_pos += d.pos_m * d.pos_m;
        s_vel += d.vel_mps * d.vel_mps;
        s_att += d.att_deg * d.att_deg;
        s_bg += d.bias_g * d.bias_g;
        s_ba += d.bias_a * d.bias_a;
    }
    let n = diffs.len() as f64;
    out.rms_pos_m = (s_pos / n).sqrt();
    out.rms_vel_mps = (s_vel / n).sqrt();
    out.rms_att_deg = (s_att / n).sqrt();
    out.rms_bias_g = (s_bg / n).sqrt();
    out.rms_bias_a = (s_ba / n).sqrt();
    let last = diffs[diffs.len() - 1];
    out.final_pos_m = last.pos_m;
    out.final_vel_mps = last.vel_mps;
    out.final_att_deg = last.att_deg;
    out
}

fn initial_yaw_rad(gnss: &FusionGnssSample, yaw_init_speed_mps: f32) -> f32 {
    let speed_h = (gnss.vel_ned_mps[0] * gnss.vel_ned_mps[0] + gnss.vel_ned_mps[1] * gnss.vel_ned_mps[1]).sqrt();
    if let Some(heading_rad) = gnss.heading_rad {
        heading_rad
    } else if speed_h >= yaw_init_speed_mps.max(1.0) {
        gnss.vel_ned_mps[1].atan2(gnss.vel_ned_mps[0])
    } else {
        0.0
    }
}

fn yaw_quat(yaw_rad: f32) -> [f32; 4] {
    let half = 0.5 * yaw_rad;
    [half.cos(), 0.0, 0.0, half.sin()]
}

fn quat_to_rotmat_f32(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn transpose3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mat3_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn build_nav_events(frames: &[UbxFrame], tl: &MasterTimeline) -> Vec<(f64, NavPvtObs)> {
    let mut nav_events_nav2 = Vec::new();
    let mut nav_events_pvt = Vec::new();
    for f in frames {
        if let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters) {
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
    nav_events_nav2.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    nav_events_pvt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if !nav_events_nav2.is_empty() { nav_events_nav2 } else { nav_events_pvt }
}

fn build_imu_packets(frames: &[UbxFrame], tl: &MasterTimeline) -> Vec<ImuPacket> {
    let mut raw_seq = Vec::<u64>::new();
    let mut raw_tag = Vec::<u64>::new();
    let mut raw_dtype = Vec::<u8>::new();
    let mut raw_val = Vec::<f64>::new();
    for f in frames {
        for (tag, sw) in extract_esf_raw_samples(f) {
            let (_, _, scale) = sensor_meta(sw.dtype);
            raw_seq.push(f.seq);
            raw_tag.push(tag);
            raw_dtype.push(sw.dtype);
            raw_val.push(sw.value_i24 as f64 * scale);
        }
    }
    let (raw_tag_u, a_raw, b_raw) =
        fit_tag_ms_map_local(&raw_seq, &raw_tag, &tl.masters, Some(1 << 16));
    let mut imu_packets = Vec::new();
    let mut current_tag: Option<u64> = None;
    let mut t_ms = 0.0;
    let mut gx = None;
    let mut gy = None;
    let mut gz = None;
    let mut ax = None;
    let mut ay = None;
    let mut az = None;
    for (((seq, tag_u), dtype), val) in raw_seq.iter().zip(raw_tag_u.iter()).zip(raw_dtype.iter()).zip(raw_val.iter()) {
        if current_tag != Some(*tag_u) {
            if let (Some(gxv), Some(gyv), Some(gzv), Some(axv), Some(ayv), Some(azv)) = (gx, gy, gz, ax, ay, az) {
                imu_packets.push(ImuPacket { t_ms, gx_dps: gxv, gy_dps: gyv, gz_dps: gzv, ax_mps2: axv, ay_mps2: ayv, az_mps2: azv });
            }
            gx = None; gy = None; gz = None; ax = None; ay = None; az = None;
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
    if let (Some(gxv), Some(gyv), Some(gzv), Some(axv), Some(ayv), Some(azv)) = (gx, gy, gz, ax, ay, az) {
        imu_packets.push(ImuPacket { t_ms, gx_dps: gxv, gy_dps: gyv, gz_dps: gzv, ax_mps2: axv, ay_mps2: ayv, az_mps2: azv });
    }
    imu_packets.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap());
    imu_packets
}

fn fit_tag_ms_map_local(
    seqs: &[u64],
    tags: &[u64],
    masters: &[(u64, f64)],
    unwrap_modulus: Option<u64>,
) -> (Vec<u64>, f64, f64) {
    let mapped_tags = match unwrap_modulus {
        Some(m) => unwrap_counter(tags, m),
        None => tags.to_vec(),
    };
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for (seq, tag_u) in seqs.iter().zip(mapped_tags.iter()) {
        if let Some(ms) = nearest_master_ms(*seq, masters) {
            x.push(*tag_u as f64);
            y.push(ms);
        }
    }
    let (a, b) = fit_linear_map(&x, &y, 1e-3);
    (mapped_tags, a, b)
}

fn nav_to_ned(obs: NavPvtObs, ref_nav: Option<NavPvtObs>) -> [f32; 3] {
    let Some(ref_nav) = ref_nav else { return [0.0, 0.0, 0.0]; };
    let lat_scale = 111_320.0;
    let lon_scale = 111_320.0 * (ref_nav.lat_deg.to_radians().cos());
    [
        ((obs.lat_deg - ref_nav.lat_deg) * lat_scale) as f32,
        ((obs.lon_deg - ref_nav.lon_deg) * lon_scale) as f32,
        (ref_nav.height_m - obs.height_m) as f32,
    ]
}

fn parse_section_files(path: &PathBuf, section: &str) -> Result<Vec<String>> {
    let text = fs::read_to_string(path)?;
    let mut in_section = false;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.ends_with(':') {
            in_section = line.trim_end_matches(':') == section;
            continue;
        }
        if in_section {
            let name = line.split_whitespace().next().unwrap_or("").trim_end_matches(',');
            if !name.is_empty() {
                out.push(name.to_string());
            }
        }
    }
    Ok(out)
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn quat_angle_deg(a: [f64; 4], b: [f64; 4]) -> f64 {
    let an = quat_normalize(a);
    let bn = quat_normalize(b);
    let dot = (an[0] * bn[0] + an[1] * bn[1] + an[2] * bn[2] + an[3] * bn[3]).abs();
    (2.0 * dot.clamp(-1.0, 1.0).acos()).to_degrees()
}

fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.6}")).unwrap_or_else(|| "".to_string())
}
