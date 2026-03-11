use std::cmp::Ordering;

use align_rs::align::{Align, AlignConfig, AlignUpdateTrace, AlignWindowSummary, GRAVITY_MPS2};

use crate::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_raw_samples, extract_nav2_pvt_obs,
    sensor_meta,
};

use super::super::math::{nearest_master_ms, normalize_heading_deg};
use super::super::model::ImuPacket;
use super::tag_time::fit_tag_ms_map;
use super::timebase::MasterTimeline;

#[derive(Clone, Copy)]
pub struct BootstrapConfig {
    pub ema_alpha: f32,
    pub max_speed_mps: f32,
    pub stationary_samples: usize,
    pub max_gyro_radps: f32,
    pub max_accel_norm_err_mps2: f32,
}

struct BootstrapDetector {
    cfg: BootstrapConfig,
    gyro_ema: Option<f32>,
    accel_err_ema: Option<f32>,
    speed_ema: Option<f32>,
    stationary_accel: Vec<[f32; 3]>,
}

#[derive(Clone, Copy)]
pub struct AlgEvent {
    pub t_ms: f64,
    pub q_frd: [f64; 4],
}

#[derive(Clone, Copy, Default)]
pub struct AlignEulerContrib {
    pub turn_gyro: [f64; 3],
    pub course_rate: [f64; 3],
    pub lateral_accel: [f64; 3],
    pub longitudinal_accel: [f64; 3],
}

#[derive(Clone, Copy, Default)]
pub struct LongTraceSample {
    pub base_valid: bool,
    pub emitted: bool,
    pub stable_windows: usize,
    pub gnss_long_lp_mps2: f64,
    pub gnss_lat_lp_mps2: f64,
    pub imu_long_lp_mps2: f64,
    pub imu_lat_lp_mps2: f64,
    pub angle_err_deg: f64,
}

#[derive(Clone, Copy)]
pub struct AlignReplaySample {
    pub t_ms: f64,
    pub t_s: f64,
    pub q_align: [f64; 4],
    pub align_rpy_deg: [f64; 3],
    pub alg_q: Option<[f64; 4]>,
    pub alg_rpy_deg: Option<[f64; 3]>,
    pub course_rate_dps: f64,
    pub a_lat_mps2: f64,
    pub a_long_mps2: f64,
    pub stationary: bool,
    pub turn_valid: bool,
    pub long_valid: bool,
    pub upd_gravity: bool,
    pub upd_turn_gyro: bool,
    pub upd_course: bool,
    pub upd_lat: bool,
    pub upd_long: bool,
    pub contrib: AlignEulerContrib,
    pub p_diag: [f64; 3],
    pub long_trace: LongTraceSample,
}

pub struct AlignReplayData {
    pub alg_events: Vec<AlgEvent>,
    pub samples: Vec<AlignReplaySample>,
}

pub fn build_align_replay(
    frames: &[UbxFrame],
    tl: &MasterTimeline,
    cfg: AlignConfig,
    bootstrap_cfg: BootstrapConfig,
) -> AlignReplayData {
    let mut alg_events = Vec::<AlgEvent>::new();
    let mut nav_events = Vec::<(f64, NavPvtObs)>::new();
    for f in frames {
        if let Some((_, roll, pitch, yaw)) = extract_esf_alg(f)
            && let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters)
        {
            alg_events.push(AlgEvent {
                t_ms,
                q_frd: esf_alg_flu_to_frd_mount_quat(roll, pitch, yaw),
            });
        }
        if let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters)
            && let Some(obs) = extract_nav2_pvt_obs(f)
            && obs.fix_ok
            && !obs.invalid_llh
        {
            nav_events.push((t_ms, obs));
        }
    }
    alg_events.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap_or(Ordering::Equal));
    nav_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

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
    imu_packets.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap_or(Ordering::Equal));

    let mut align = Align::new(cfg);
    let mut bootstrap = BootstrapDetector::new(bootstrap_cfg);
    let mut align_initialized = false;
    let mut scan_idx = 0usize;
    let mut interval_start_idx = 0usize;
    let mut prev_nav: Option<(f64, NavPvtObs)> = None;
    let mut samples = Vec::<AlignReplaySample>::new();

    for (tn, nav) in &nav_events {
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
                let mean_gyro_b = [gyro_sum[0] * inv_n, gyro_sum[1] * inv_n, gyro_sum[2] * inv_n];
                let mean_accel_b = [accel_sum[0] * inv_n, accel_sum[1] * inv_n, accel_sum[2] * inv_n];
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
                let (_, trace) = align.update_window_with_trace(&window);

                let v_prev = [nav_prev.vel_n_mps, nav_prev.vel_e_mps];
                let v_curr = [nav.vel_n_mps, nav.vel_e_mps];
                let course_prev = v_prev[1].atan2(v_prev[0]);
                let course_curr = v_curr[1].atan2(v_curr[0]);
                let course_rate_dps =
                    wrap_rad_pi(course_curr - course_prev).to_degrees() / (dt as f64);
                let a_n = [
                    (nav.vel_n_mps - nav_prev.vel_n_mps) / (dt as f64),
                    (nav.vel_e_mps - nav_prev.vel_e_mps) / (dt as f64),
                ];
                let v_mid = [0.5 * (v_prev[0] + v_curr[0]), 0.5 * (v_prev[1] + v_curr[1])];
                let (a_long, a_lat) = if let Some(t_hat) = normalize2(v_mid) {
                    let lat_hat = [-t_hat[1], t_hat[0]];
                    (
                        t_hat[0] * a_n[0] + t_hat[1] * a_n[1],
                        lat_hat[0] * a_n[0] + lat_hat[1] * a_n[1],
                    )
                } else {
                    (0.0, 0.0)
                };

                let gyro_norm =
                    (mean_gyro_b[0] * mean_gyro_b[0] + mean_gyro_b[1] * mean_gyro_b[1] + mean_gyro_b[2] * mean_gyro_b[2]).sqrt();
                let accel_norm =
                    (mean_accel_b[0] * mean_accel_b[0] + mean_accel_b[1] * mean_accel_b[1] + mean_accel_b[2] * mean_accel_b[2]).sqrt();
                let speed_prev = (v_prev[0] * v_prev[0] + v_prev[1] * v_prev[1]).sqrt() as f32;
                let speed_curr = (v_curr[0] * v_curr[0] + v_curr[1] * v_curr[1]).sqrt() as f32;
                let speed_mid = 0.5_f32 * (speed_prev + speed_curr);
                let stationary = gyro_norm <= cfg.max_stationary_gyro_radps
                    && (accel_norm - GRAVITY_MPS2).abs() <= cfg.max_stationary_accel_norm_err_mps2
                    && speed_mid < 0.5;
                let turn_valid = speed_mid > cfg.min_speed_mps
                    && course_rate_dps.abs() > cfg.min_turn_rate_radps.to_degrees() as f64
                    && a_lat.abs() > cfg.min_lat_acc_mps2 as f64;
                let long_valid =
                    speed_mid > cfg.min_speed_mps && (a_long * a_long + a_lat * a_lat).sqrt() > cfg.min_long_acc_mps2 as f64;

                let q_align = [align.q_vb[0] as f64, align.q_vb[1] as f64, align.q_vb[2] as f64, align.q_vb[3] as f64];
                let align_rpy_deg = {
                    let (r, p, y) = quat_rpy_alg_deg(q_align[0], q_align[1], q_align[2], q_align[3]);
                    [r, p, y]
                };
                let alg_q = interpolate_alg_quat(&alg_events, *tn);
                let alg_rpy_deg = alg_q.map(|q| {
                    let (r, p, y) = quat_rpy_alg_deg(q[0], q[1], q[2], q[3]);
                    [r, p, y]
                });
                let long_trace = trace.longitudinal_trace.unwrap_or_default();
                let long_trace = LongTraceSample {
                    base_valid: long_trace.base_valid,
                    emitted: long_trace.emitted,
                    stable_windows: long_trace.stable_windows,
                    gnss_long_lp_mps2: long_trace.gnss_long_lp_mps2 as f64,
                    gnss_lat_lp_mps2: long_trace.gnss_lat_lp_mps2 as f64,
                    imu_long_lp_mps2: long_trace.imu_long_lp_mps2 as f64,
                    imu_lat_lp_mps2: long_trace.imu_lat_lp_mps2 as f64,
                    angle_err_deg: (long_trace.angle_err_rad as f64).to_degrees(),
                };

                samples.push(AlignReplaySample {
                    t_ms: *tn,
                    t_s: (*tn - tl.t0_master_ms) * 1.0e-3,
                    q_align,
                    align_rpy_deg,
                    alg_q,
                    alg_rpy_deg,
                    course_rate_dps,
                    a_lat_mps2: a_lat,
                    a_long_mps2: a_long,
                    stationary,
                    turn_valid,
                    long_valid,
                    upd_gravity: cfg.use_gravity && stationary,
                    upd_turn_gyro: cfg.use_turn_gyro && turn_valid,
                    upd_course: cfg.use_course_rate && turn_valid,
                    upd_lat: cfg.use_lateral_accel && turn_valid,
                    upd_long: cfg.use_longitudinal_accel && long_valid,
                    contrib: align_update_contrib_deg(trace),
                    p_diag: [align.P[0][0] as f64, align.P[1][1] as f64, align.P[2][2] as f64],
                    long_trace,
                });
            }
        }
        prev_nav = Some((*tn, *nav));
        interval_start_idx = scan_idx;
    }

    AlignReplayData { alg_events, samples }
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
        self.accel_err_ema = Some(ema_update(self.accel_err_ema, accel_err, self.cfg.ema_alpha));
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

fn ema_update(prev: Option<f32>, sample: f32, alpha: f32) -> f32 {
    let alpha = alpha.clamp(1.0e-4, 1.0);
    match prev {
        Some(prev) => (1.0 - alpha) * prev + alpha * sample,
        None => sample,
    }
}

fn speed_for_bootstrap(prev_nav: Option<(f64, NavPvtObs)>, curr_nav: (f64, NavPvtObs), t_ms: f64) -> f64 {
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

pub fn esf_alg_flu_to_frd_mount_quat(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> [f64; 4] {
    let q_flu = quat_from_rpy_alg_deg(roll_deg, pitch_deg, yaw_deg);
    let q_x_180 = [0.0, 1.0, 0.0, 0.0];
    quat_normalize(quat_mul(quat_conj(q_flu), q_x_180))
}

fn quat_nlerp_shortest(a: [f64; 4], b: [f64; 4], alpha: f64) -> [f64; 4] {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let bb = if dot < 0.0 { [-b[0], -b[1], -b[2], -b[3]] } else { b };
    quat_normalize([
        a[0] + alpha * (bb[0] - a[0]),
        a[1] + alpha * (bb[1] - a[1]),
        a[2] + alpha * (bb[2] - a[2]),
        a[3] + alpha * (bb[3] - a[3]),
    ])
}

pub fn interpolate_alg_quat(events: &[AlgEvent], t_ms: f64) -> Option<[f64; 4]> {
    if events.is_empty() {
        return None;
    }
    let idx = events.partition_point(|e| e.t_ms < t_ms);
    if idx == 0 {
        return Some(events[0].q_frd);
    }
    if idx >= events.len() {
        return Some(events[events.len() - 1].q_frd);
    }
    let e0 = events[idx - 1];
    let e1 = events[idx];
    let dt = e1.t_ms - e0.t_ms;
    if dt.abs() <= 1.0e-9 {
        return Some(e0.q_frd);
    }
    let alpha = ((t_ms - e0.t_ms) / dt).clamp(0.0, 1.0);
    Some(quat_nlerp_shortest(e0.q_frd, e1.q_frd, alpha))
}

pub fn quat_rpy_alg_deg(q0: f64, q1: f64, q2: f64, q3: f64) -> (f64, f64, f64) {
    let n = (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3).sqrt();
    let (w, x, y, z) = if n > 1.0e-12 {
        (q0 / n, q1 / n, q2 / n, q3 / n)
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r10 = 2.0 * (x * y + w * z);
    let r20 = 2.0 * (x * z - w * y);
    let r21 = 2.0 * (y * z + w * x);
    let r22 = 1.0 - 2.0 * (x * x + y * y);
    let pitch = (-r20).clamp(-1.0, 1.0).asin();
    let roll = r21.atan2(r22);
    let yaw = r10.atan2(r00);
    (roll.to_degrees(), pitch.to_degrees(), normalize_heading_deg(yaw.to_degrees()))
}

fn quat_from_rpy_alg_deg(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> [f64; 4] {
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
    if n <= 1.0e-12 { [1.0, 0.0, 0.0, 0.0] } else { [q[0] / n, q[1] / n, q[2] / n, q[3] / n] }
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

pub fn quat_rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let q = quat_normalize(q);
    let p = [0.0, v[0], v[1], v[2]];
    let qp = quat_mul(q, p);
    let qpq = quat_mul(qp, quat_conj(q));
    [qpq[1], qpq[2], qpq[3]]
}

pub fn axis_angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if na <= 1.0e-12 || nb <= 1.0e-12 {
        return f64::NAN;
    }
    let dot = ((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)).clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}

fn quat_to_rpy_alg_deg(q: [f32; 4]) -> [f64; 3] {
    let (r, p, y) = quat_rpy_alg_deg(q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64);
    [r, p, y]
}

fn wrap_deg180(x: f64) -> f64 {
    (x + 180.0).rem_euclid(360.0) - 180.0
}

fn rpy_delta_deg(before_q: [f32; 4], after_q: [f32; 4]) -> [f64; 3] {
    let before = quat_to_rpy_alg_deg(before_q);
    let after = quat_to_rpy_alg_deg(after_q);
    [
        wrap_deg180(after[0] - before[0]),
        wrap_deg180(after[1] - before[1]),
        wrap_deg180(after[2] - before[2]),
    ]
}

fn align_update_contrib_deg(trace: AlignUpdateTrace) -> AlignEulerContrib {
    let mut out = AlignEulerContrib::default();
    let mut prev_q = trace.q_start;
    if let Some(q) = trace.after_gravity {
        prev_q = q;
    }
    if let Some(q) = trace.after_turn_gyro {
        out.turn_gyro = rpy_delta_deg(prev_q, q);
        prev_q = q;
    }
    if let Some(q) = trace.after_course_rate {
        out.course_rate = rpy_delta_deg(prev_q, q);
        prev_q = q;
    }
    if let Some(q) = trace.after_lateral_accel {
        out.lateral_accel = rpy_delta_deg(prev_q, q);
        prev_q = q;
    }
    if let Some(q) = trace.after_longitudinal_accel {
        out.longitudinal_accel = rpy_delta_deg(prev_q, q);
    }
    out
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize2(v: [f64; 2]) -> Option<[f64; 2]> {
    let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if !n.is_finite() || n <= 1.0e-9 {
        return None;
    }
    Some([v[0] / n, v[1] / n])
}

fn wrap_rad_pi(x: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    (x + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI
}
