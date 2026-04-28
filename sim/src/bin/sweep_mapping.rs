use std::{f64::consts::PI, fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::ubxlog::{
    extract_esf_alg, extract_esf_ins, extract_esf_raw_samples, extract_itow_ms, fit_linear_map,
    parse_ubx_frames, sensor_meta, unwrap_counter,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(value_name = "LOGFILE", required = true)]
    logfiles: Vec<PathBuf>,
}

#[derive(Clone, Copy)]
struct AlgEvent {
    t_ms: f64,
    roll_deg: f64,
    pitch_deg: f64,
    yaw_deg: f64,
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

#[derive(Clone, Copy)]
struct InsEvent {
    t_ms: f64,
    gx_dps: f64,
    gy_dps: f64,
    gz_dps: f64,
    ax_mps2: f64,
    ay_mps2: f64,
    az_mps2: f64,
}

#[derive(Clone, Copy)]
enum BaseRot {
    Xyz,
    XyzT,
    Zyx,
    ZyxT,
}

#[derive(Clone, Copy)]
enum PostRot {
    I,
    Rx180,
    Ry180,
    Rz180,
}

#[derive(Clone, Copy)]
struct Candidate {
    base: BaseRot,
    neg_angles: bool,
    post: PostRot,
}

fn deg2rad(v: f64) -> f64 {
    v * PI / 180.0
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

fn apply_candidate(
    c: Candidate,
    alg: AlgEvent,
    gyro: [f64; 3],
    accel: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let (r, p, y) = if c.neg_angles {
        (-alg.roll_deg, -alg.pitch_deg, -alg.yaw_deg)
    } else {
        (alg.roll_deg, alg.pitch_deg, alg.yaw_deg)
    };
    let rot = match c.base {
        BaseRot::Xyz => rot_xyz(deg2rad(r), deg2rad(p), deg2rad(y)),
        BaseRot::XyzT => transpose(rot_xyz(deg2rad(r), deg2rad(p), deg2rad(y))),
        BaseRot::Zyx => rot_zyx(deg2rad(y), deg2rad(p), deg2rad(r)),
        BaseRot::ZyxT => transpose(rot_zyx(deg2rad(y), deg2rad(p), deg2rad(r))),
    };
    let mut g = mat_vec(rot, gyro);
    let mut a = mat_vec(rot, accel);
    match c.post {
        PostRot::I => {}
        PostRot::Rx180 => {
            g[1] = -g[1];
            g[2] = -g[2];
            a[1] = -a[1];
            a[2] = -a[2];
        }
        PostRot::Ry180 => {
            g[0] = -g[0];
            g[2] = -g[2];
            a[0] = -a[0];
            a[2] = -a[2];
        }
        PostRot::Rz180 => {
            g[0] = -g[0];
            g[1] = -g[1];
            a[0] = -a[0];
            a[1] = -a[1];
        }
    }
    (g, a)
}

fn name_of(c: Candidate) -> String {
    let b = match c.base {
        BaseRot::Xyz => "xyz",
        BaseRot::XyzT => "xyzT",
        BaseRot::Zyx => "zyx",
        BaseRot::ZyxT => "zyxT",
    };
    let n = if c.neg_angles { "neg" } else { "pos" };
    let p = match c.post {
        PostRot::I => "I",
        PostRot::Rx180 => "Rx",
        PostRot::Ry180 => "Ry",
        PostRot::Rz180 => "Rz",
    };
    format!("{b}_{n}_{p}")
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut candidates = Vec::<Candidate>::new();
    for &base in &[BaseRot::Xyz, BaseRot::XyzT, BaseRot::Zyx, BaseRot::ZyxT] {
        for &neg in &[false, true] {
            for &post in &[PostRot::I, PostRot::Rx180, PostRot::Ry180, PostRot::Rz180] {
                candidates.push(Candidate {
                    base,
                    neg_angles: neg,
                    post,
                });
            }
        }
    }

    let mut total_scores = vec![0.0_f64; candidates.len()];
    let mut total_counts = vec![0usize; candidates.len()];

    for logfile in &args.logfiles {
        let mut bytes = Vec::new();
        File::open(logfile)
            .with_context(|| format!("failed to open {}", logfile.display()))?
            .read_to_end(&mut bytes)
            .context("failed to read log")?;
        let frames = parse_ubx_frames(&bytes, None);

        let mut masters: Vec<(u64, f64)> = Vec::new();
        for f in &frames {
            if let Some(itow) = extract_itow_ms(f)
                && (0..604_800_000).contains(&itow)
            {
                masters.push((f.seq, itow as f64));
            }
        }
        masters.sort_by_key(|x| x.0);
        let raw: Vec<i64> = masters.iter().map(|(_, ms)| *ms as i64).collect();
        let unwrapped = unwrap_i64_counter(&raw, 604_800_000);
        for (m, msu) in masters.iter_mut().zip(unwrapped.into_iter()) {
            m.1 = msu as f64;
        }

        let mut alg_events = Vec::<AlgEvent>::new();
        let mut ins_events = Vec::<InsEvent>::new();
        for f in &frames {
            if let Some((_, roll, pitch, yaw)) = extract_esf_alg(f)
                && let Some(t_ms) = nearest_master_ms(f.seq, &masters)
            {
                alg_events.push(AlgEvent {
                    t_ms,
                    roll_deg: roll,
                    pitch_deg: pitch,
                    yaw_deg: yaw,
                });
            }
            if let Some((_itow, gx, gy, gz, ax, ay, az)) = extract_esf_ins(f)
                && let Some(t_ms) = nearest_master_ms(f.seq, &masters)
            {
                ins_events.push(InsEvent {
                    t_ms,
                    gx_dps: gx,
                    gy_dps: gy,
                    gz_dps: gz,
                    ax_mps2: ax,
                    ay_mps2: ay,
                    az_mps2: az,
                });
            }
        }
        alg_events.sort_by(|a, b| {
            a.t_ms
                .partial_cmp(&b.t_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ins_events.sort_by(|a, b| {
            a.t_ms
                .partial_cmp(&b.t_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut raw_seq = Vec::<u64>::new();
        let mut raw_tag = Vec::<u64>::new();
        let mut raw_dtype = Vec::<u8>::new();
        let mut raw_val = Vec::<f64>::new();
        for f in &frames {
            for (tag, sw) in extract_esf_raw_samples(f) {
                let (_, _, scale) = sensor_meta(sw.dtype);
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
        imu_packets.sort_by(|a, b| {
            a.t_ms
                .partial_cmp(&b.t_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut file_scores = vec![0.0_f64; candidates.len()];
        let mut file_counts = vec![0usize; candidates.len()];

        let mut alg_i = 0usize;
        let mut cur_alg: Option<AlgEvent> = None;
        let mut ins_i = 0usize;
        for pkt in &imu_packets {
            while alg_i < alg_events.len() && alg_events[alg_i].t_ms <= pkt.t_ms {
                cur_alg = Some(alg_events[alg_i]);
                alg_i += 1;
            }
            let Some(alg) = cur_alg else { continue };
            while ins_i + 1 < ins_events.len() && ins_events[ins_i + 1].t_ms <= pkt.t_ms {
                ins_i += 1;
            }
            let mut j = ins_i;
            if ins_i + 1 < ins_events.len()
                && (ins_events[ins_i + 1].t_ms - pkt.t_ms).abs()
                    < (ins_events[ins_i].t_ms - pkt.t_ms).abs()
            {
                j = ins_i + 1;
            }
            let ins = ins_events[j];
            if (ins.t_ms - pkt.t_ms).abs() > 50.0 {
                continue;
            }
            for (k, cand) in candidates.iter().enumerate() {
                let (g, a) = apply_candidate(
                    *cand,
                    alg,
                    [pkt.gx_dps, pkt.gy_dps, pkt.gz_dps],
                    [pkt.ax_mps2, pkt.ay_mps2, pkt.az_mps2],
                );
                let eg = (g[0] - ins.gx_dps).powi(2)
                    + (g[1] - ins.gy_dps).powi(2)
                    + (g[2] - ins.gz_dps).powi(2);
                let ea = (a[0] - ins.ax_mps2).powi(2)
                    + (a[1] - ins.ay_mps2).powi(2)
                    + (a[2] - ins.az_mps2).powi(2);
                file_scores[k] += eg + 0.05 * ea;
                file_counts[k] += 1;
            }
        }

        println!("\nFILE {}", logfile.display());
        let mut ranked = Vec::<(f64, String, usize)>::new();
        for (k, cand) in candidates.iter().enumerate() {
            if file_counts[k] == 0 {
                continue;
            }
            let s = file_scores[k] / file_counts[k] as f64;
            ranked.push((s, name_of(*cand), file_counts[k]));
            total_scores[k] += file_scores[k];
            total_counts[k] += file_counts[k];
        }
        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (s, n, c) in ranked.iter().take(8) {
            println!("{:<16} score={:>10.5} n={}", n, s, c);
        }
    }

    let mut ranked_total = Vec::<(f64, String, usize)>::new();
    for (k, cand) in candidates.iter().enumerate() {
        if total_counts[k] == 0 {
            continue;
        }
        ranked_total.push((
            total_scores[k] / total_counts[k] as f64,
            name_of(*cand),
            total_counts[k],
        ));
    }
    ranked_total.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    println!("\nTOTAL (all files)");
    for (s, n, c) in ranked_total.iter().take(12) {
        println!("{:<16} score={:>10.5} n={}", n, s, c);
    }
    Ok(())
}
