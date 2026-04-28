use std::cmp::Ordering;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::align::{Align, AlignConfig, AlignWindowSummary, GRAVITY_MPS2};
use sim::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_raw_samples, extract_nav2_pvt_obs,
    fit_linear_map, parse_ubx_frames, sensor_meta, unwrap_counter,
};
use sim::visualizer::math::{nearest_master_ms, unwrap_i64_counter};

#[derive(Parser, Debug)]
#[command(name = "esf_alg_convention")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfiles: Vec<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
struct AlgEventRaw {
    t_ms: f64,
    q_raw: [f64; 4],
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

#[derive(Clone, Debug)]
struct Dataset {
    nav_events: Vec<(f64, NavPvtObs)>,
    alg_events: Vec<AlgEventRaw>,
    imu_packets: Vec<ImuPacket>,
}

#[derive(Clone, Copy, Debug)]
struct BootstrapConfig {
    ema_alpha: f32,
    max_speed_mps: f32,
    stationary_samples: usize,
    max_gyro_radps: f32,
    max_accel_norm_err_mps2: f32,
}

#[derive(Clone, Debug)]
struct BootstrapDetector {
    cfg: BootstrapConfig,
    gyro_ema: Option<f32>,
    accel_err_ema: Option<f32>,
    speed_ema: Option<f32>,
    stationary_accel: Vec<[f32; 3]>,
}

#[derive(Clone, Copy)]
struct Convention {
    name: &'static str,
    invert: bool,
    left_x: bool,
    right_x: bool,
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    n: usize,
    mean_rot_deg: f64,
    mean_fwd_deg: f64,
    mean_down_deg: f64,
    final_rot_deg: f64,
    final_fwd_deg: f64,
    final_down_deg: f64,
}

const CONVENTIONS: &[Convention] = &[
    Convention {
        name: "raw",
        invert: false,
        left_x: false,
        right_x: false,
    },
    Convention {
        name: "raw_right_x",
        invert: false,
        left_x: false,
        right_x: true,
    },
    Convention {
        name: "raw_left_x",
        invert: false,
        left_x: true,
        right_x: false,
    },
    Convention {
        name: "raw_conj_x",
        invert: false,
        left_x: true,
        right_x: true,
    },
    Convention {
        name: "inv",
        invert: true,
        left_x: false,
        right_x: false,
    },
    Convention {
        name: "inv_right_x",
        invert: true,
        left_x: false,
        right_x: true,
    },
    Convention {
        name: "inv_left_x",
        invert: true,
        left_x: true,
        right_x: false,
    },
    Convention {
        name: "inv_conj_x",
        invert: true,
        left_x: true,
        right_x: true,
    },
];

fn main() -> Result<()> {
    let args = Args::parse();
    let logfiles = if args.logfiles.is_empty() {
        vec![
            PathBuf::from("logger/data/ubx_raw_20260303_175904.bin"),
            PathBuf::from("logger/data/ubx_raw_20260307_124850.bin"),
            PathBuf::from("logger/data/ubx_raw_20260309_112501.bin"),
        ]
    } else {
        args.logfiles
    };

    for logfile in &logfiles {
        let dataset = load_dataset(logfile)?;
        let align_samples = run_align(&dataset)?;
        println!("\n== {} ==", logfile.display());
        for conv in CONVENTIONS {
            let m = evaluate_convention(&dataset.alg_events, &align_samples, *conv);
            println!(
                "{:<12} n={:<5} mean(rot/fwd/down)=({:6.2},{:6.2},{:6.2}) final=({:6.2},{:6.2},{:6.2})",
                conv.name,
                m.n,
                m.mean_rot_deg,
                m.mean_fwd_deg,
                m.mean_down_deg,
                m.final_rot_deg,
                m.final_fwd_deg,
                m.final_down_deg
            );
        }
    }

    Ok(())
}

fn load_dataset(logfile: &PathBuf) -> Result<Dataset> {
    let bytes =
        std::fs::read(logfile).with_context(|| format!("failed to read {}", logfile.display()))?;
    let frames = parse_ubx_frames(&bytes, None);
    if frames.is_empty() {
        bail!("no UBX frames parsed");
    }
    let timeline = build_master_timeline(&frames);
    if !timeline.has_itow {
        bail!("log does not contain a usable iTOW timeline");
    }

    let mut alg_events = Vec::new();
    let mut nav_events = Vec::new();
    for frame in &frames {
        if let Some(t_ms) = nearest_master_ms(frame.seq, &timeline.masters) {
            if let Some((_, roll, pitch, yaw)) = extract_esf_alg(frame) {
                alg_events.push(AlgEventRaw {
                    t_ms,
                    q_raw: quat_from_rpy_deg(roll, pitch, yaw),
                });
            }
            if let Some(obs) = extract_nav2_pvt_obs(frame)
                && obs.fix_ok
                && !obs.invalid_llh
            {
                nav_events.push((t_ms, obs));
            }
        }
    }
    alg_events.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap_or(Ordering::Equal));
    nav_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    if alg_events.is_empty() {
        bail!("no ESF-ALG");
    }
    if nav_events.len() < 2 {
        bail!("need at least two NAV2-PVT observations");
    }
    let imu_packets = build_imu_packets(&frames, &timeline)?;
    if imu_packets.is_empty() {
        bail!("no complete ESF-RAW IMU packets found");
    }
    Ok(Dataset {
        nav_events,
        alg_events,
        imu_packets,
    })
}

fn run_align(dataset: &Dataset) -> Result<Vec<(f64, [f64; 4])>> {
    let cfg = AlignConfig::default();
    let bootstrap_cfg = BootstrapConfig {
        ema_alpha: 0.05,
        max_speed_mps: 0.35,
        stationary_samples: 300,
        max_gyro_radps: cfg.max_stationary_gyro_radps,
        max_accel_norm_err_mps2: cfg.max_stationary_accel_norm_err_mps2,
    };
    let mut align = Align::new(cfg);
    let mut bootstrap = BootstrapDetector::new(bootstrap_cfg);
    let mut align_initialized = false;
    let mut scan_idx = 0usize;
    let mut interval_start_idx = 0usize;
    let mut prev_nav: Option<(f64, NavPvtObs)> = None;
    let mut out = Vec::new();

    for (tn, nav) in &dataset.nav_events {
        while scan_idx < dataset.imu_packets.len() && dataset.imu_packets[scan_idx].t_ms <= *tn {
            let pkt = &dataset.imu_packets[scan_idx];
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
            let interval_packets = &dataset.imu_packets[interval_start_idx..scan_idx];
            if align_initialized && dt > 0.0 && !interval_packets.is_empty() {
                let (mean_gyro_b, mean_accel_b) = mean_imu(interval_packets);
                let window = AlignWindowSummary {
                    dt,
                    mean_gyro_b,
                    mean_accel_b,
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
                align.update_window(&window);
                let q = align.q_vb;
                out.push((*tn, [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64]));
            }
        }
        prev_nav = Some((*tn, *nav));
        interval_start_idx = scan_idx;
    }
    Ok(out)
}

fn evaluate_convention(
    alg_events: &[AlgEventRaw],
    align_samples: &[(f64, [f64; 4])],
    conv: Convention,
) -> Metrics {
    let mut m = Metrics::default();
    let mut sum_rot = 0.0;
    let mut sum_fwd = 0.0;
    let mut sum_down = 0.0;
    let mut last = Metrics::default();
    for (t_ms, q_align) in align_samples {
        if let Some(raw) = interpolate_alg_raw(alg_events, *t_ms) {
            let q_ref = alg_quat(raw, conv);
            let rot = quat_angle_deg(*q_align, q_ref);
            let fwd = axis_angle_deg(
                quat_rotate(*q_align, [1.0, 0.0, 0.0]),
                quat_rotate(q_ref, [1.0, 0.0, 0.0]),
            );
            let down = axis_angle_deg(
                quat_rotate(*q_align, [0.0, 0.0, 1.0]),
                quat_rotate(q_ref, [0.0, 0.0, 1.0]),
            );
            sum_rot += rot;
            sum_fwd += fwd;
            sum_down += down;
            m.n += 1;
            last.final_rot_deg = rot;
            last.final_fwd_deg = fwd;
            last.final_down_deg = down;
        }
    }
    if m.n > 0 {
        let n = m.n as f64;
        m.mean_rot_deg = sum_rot / n;
        m.mean_fwd_deg = sum_fwd / n;
        m.mean_down_deg = sum_down / n;
        m.final_rot_deg = last.final_rot_deg;
        m.final_fwd_deg = last.final_fwd_deg;
        m.final_down_deg = last.final_down_deg;
    }
    m
}

fn alg_quat(q_raw: [f64; 4], conv: Convention) -> [f64; 4] {
    let mut q = q_raw;
    let qx = [0.0, 1.0, 0.0, 0.0];
    if conv.invert {
        q = quat_conj(q);
    }
    if conv.left_x {
        q = quat_normalize(quat_mul(qx, q));
    }
    if conv.right_x {
        q = quat_normalize(quat_mul(q, qx));
    }
    q
}

fn interpolate_alg_raw(events: &[AlgEventRaw], t_ms: f64) -> Option<[f64; 4]> {
    if events.is_empty() {
        return None;
    }
    let idx = match events.binary_search_by(|e| e.t_ms.partial_cmp(&t_ms).unwrap_or(Ordering::Less))
    {
        Ok(i) => i,
        Err(i) => i,
    };
    if idx == 0 {
        return Some(events[0].q_raw);
    }
    if idx >= events.len() {
        return Some(events.last()?.q_raw);
    }
    let e0 = events[idx - 1];
    let e1 = events[idx];
    let dt = (e1.t_ms - e0.t_ms).max(1.0);
    let alpha = ((t_ms - e0.t_ms) / dt).clamp(0.0, 1.0);
    Some(quat_nlerp_shortest(e0.q_raw, e1.q_raw, alpha))
}

fn build_imu_packets(frames: &[UbxFrame], timeline: &MasterTimeline) -> Result<Vec<ImuPacket>> {
    let mut raw_seq = Vec::<u64>::new();
    let mut raw_tag = Vec::<u64>::new();
    let mut raw_dtype = Vec::<u8>::new();
    let mut raw_val = Vec::<f64>::new();
    for frame in frames {
        for (tag, sw) in extract_esf_raw_samples(frame) {
            let (_name, _unit, scale) = sensor_meta(sw.dtype);
            raw_seq.push(frame.seq);
            raw_tag.push(tag);
            raw_dtype.push(sw.dtype);
            raw_val.push(sw.value_i24 as f64 * scale);
        }
    }
    if raw_seq.is_empty() {
        bail!("no ESF-RAW samples found");
    }
    let (raw_tag_u, a_raw, b_raw) = fit_tag_ms_map(&raw_seq, &raw_tag, &timeline.masters, 1 << 16);

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
            if let Some(mapped_ms) = timeline.map_tag_ms(a_raw, b_raw, *tag_u as f64, *seq) {
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
    imu_packets.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap_or(Ordering::Equal));
    Ok(imu_packets)
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

fn ema_update(prev: Option<f32>, x: f32, alpha: f32) -> f32 {
    if let Some(prev) = prev {
        (1.0 - alpha) * prev + alpha * x
    } else {
        x
    }
}

fn speed_for_bootstrap(
    prev_nav: Option<(f64, NavPvtObs)>,
    curr_nav: (f64, NavPvtObs),
    t_ms: f64,
) -> f64 {
    let (t1, nav1) = curr_nav;
    if let Some((t0, nav0)) = prev_nav {
        let dt = (t1 - t0).max(1.0);
        let alpha = ((t_ms - t0) / dt).clamp(0.0, 1.0);
        let v0 = (nav0.vel_n_mps * nav0.vel_n_mps + nav0.vel_e_mps * nav0.vel_e_mps).sqrt();
        let v1 = (nav1.vel_n_mps * nav1.vel_n_mps + nav1.vel_e_mps * nav1.vel_e_mps).sqrt();
        v0 + alpha * (v1 - v0)
    } else {
        (nav1.vel_n_mps * nav1.vel_n_mps + nav1.vel_e_mps * nav1.vel_e_mps).sqrt()
    }
}

fn mean_imu(packets: &[ImuPacket]) -> ([f32; 3], [f32; 3]) {
    let n = packets.len().max(1) as f64;
    let mut gyro = [0.0_f64; 3];
    let mut accel = [0.0_f64; 3];
    for pkt in packets {
        gyro[0] += pkt.gx_dps.to_radians();
        gyro[1] += pkt.gy_dps.to_radians();
        gyro[2] += pkt.gz_dps.to_radians();
        accel[0] += pkt.ax_mps2;
        accel[1] += pkt.ay_mps2;
        accel[2] += pkt.az_mps2;
    }
    (
        [
            (gyro[0] / n) as f32,
            (gyro[1] / n) as f32,
            (gyro[2] / n) as f32,
        ],
        [
            (accel[0] / n) as f32,
            (accel[1] / n) as f32,
            (accel[2] / n) as f32,
        ],
    )
}

#[derive(Clone, Debug)]
struct MasterTimeline {
    masters: Vec<(u64, f64)>,
    has_itow: bool,
}

impl MasterTimeline {
    fn map_tag_ms(&self, a_raw: f64, b_raw: f64, raw_tag: f64, seq: u64) -> Option<f64> {
        nearest_master_ms(seq, &self.masters).map(|t| t + (a_raw * raw_tag + b_raw - t))
    }
}

fn build_master_timeline(frames: &[UbxFrame]) -> MasterTimeline {
    let mut itow_seq = Vec::<u64>::new();
    let mut itow_raw = Vec::<i64>::new();
    for frame in frames {
        if let Some(itow) = sim::ubxlog::extract_itow_ms(frame) {
            itow_seq.push(frame.seq);
            itow_raw.push(itow);
        }
    }
    let itow_u = unwrap_i64_counter(&itow_raw, 1 << 30);
    let masters: Vec<(u64, f64)> = itow_seq
        .iter()
        .zip(itow_u.iter())
        .map(|(&s, &t)| (s, t as f64))
        .collect();
    let has_itow = !masters.is_empty();
    MasterTimeline { masters, has_itow }
}

fn fit_tag_ms_map(
    raw_seq: &[u64],
    raw_tag: &[u64],
    masters: &[(u64, f64)],
    wrap: u64,
) -> (Vec<u64>, f64, f64) {
    let _ = raw_seq;
    let raw_tag_u64 = unwrap_counter(raw_tag, wrap);
    let mut xs = Vec::<f64>::new();
    let mut ys = Vec::<f64>::new();
    for (&seq, &tagu) in raw_seq.iter().zip(raw_tag_u64.iter()) {
        if let Some(t_ms) = nearest_master_ms(seq, masters) {
            xs.push(tagu as f64);
            ys.push(t_ms);
        }
    }
    let (a, b) = fit_linear_map(&xs, &ys, 1.0);
    (raw_tag_u64, a, b)
}

fn quat_from_rpy_deg(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> [f64; 4] {
    let (sr, cr) = (0.5 * roll_deg.to_radians()).sin_cos();
    let (sp, cp) = (0.5 * pitch_deg.to_radians()).sin_cos();
    let (sy, cy) = (0.5 * yaw_deg.to_radians()).sin_cos();
    quat_normalize([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
}

fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n <= 1.0e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
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

fn quat_nlerp_shortest(a: [f64; 4], b: [f64; 4], alpha: f64) -> [f64; 4] {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let bb = if dot < 0.0 {
        [-b[0], -b[1], -b[2], -b[3]]
    } else {
        b
    };
    quat_normalize([
        a[0] + alpha * (bb[0] - a[0]),
        a[1] + alpha * (bb[1] - a[1]),
        a[2] + alpha * (bb[2] - a[2]),
        a[3] + alpha * (bb[3] - a[3]),
    ])
}

fn quat_rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let q = quat_normalize(q);
    let p = [0.0, v[0], v[1], v[2]];
    let qp = quat_mul(q, p);
    let qpq = quat_mul(qp, quat_conj(q));
    [qpq[1], qpq[2], qpq[3]]
}

fn quat_angle_deg(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dq = quat_normalize(quat_mul(quat_conj(a), b));
    let w = dq[0].abs().clamp(-1.0, 1.0);
    2.0 * w.acos().to_degrees()
}

fn axis_angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if na <= 1.0e-12 || nb <= 1.0e-12 {
        return f64::NAN;
    }
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
    dot.clamp(-1.0, 1.0).acos().to_degrees()
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
