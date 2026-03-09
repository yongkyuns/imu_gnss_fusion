use align_rs::align::{Align, AlignConfig, AlignWindowSummary, GRAVITY_MPS2};

use crate::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_raw_samples, extract_nav2_pvt_obs,
    sensor_meta,
};

use super::super::math::{nearest_master_ms, normalize_heading_deg};
use super::super::model::{AlgEvent, ImuPacket, Trace};
use super::tag_time::fit_tag_ms_map;
use super::timebase::MasterTimeline;

pub struct AlignCompareData {
    pub cmp_att: Vec<Trace>,
    pub res_vel: Vec<Trace>,
    pub motion: Vec<Trace>,
    pub state_q: Vec<Trace>,
    pub cov: Vec<Trace>,
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

pub fn build_align_compare_traces(frames: &[UbxFrame], tl: &MasterTimeline) -> AlignCompareData {
    if tl.masters.is_empty() {
        return AlignCompareData {
            cmp_att: Vec::new(),
            res_vel: Vec::new(),
            motion: Vec::new(),
            state_q: Vec::new(),
            cov: Vec::new(),
        };
    }
    let rel_s = |master_ms: f64| (master_ms - tl.t0_master_ms) * 1e-3;

    let mut alg_events = Vec::<AlgEvent>::new();
    let mut nav_events = Vec::<(f64, NavPvtObs)>::new();
    for f in frames {
        if let Some((_, roll, pitch, yaw)) = extract_esf_alg(f) {
            if let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters) {
                let (roll_frd, pitch_frd, yaw_frd) = esf_alg_flu_to_frd_mount_deg(roll, pitch, yaw);
                alg_events.push(AlgEvent {
                    t_ms,
                    roll_deg: roll_frd,
                    pitch_deg: pitch_frd,
                    yaw_deg: normalize_heading_deg(yaw_frd),
                });
            }
        }
        if let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters) {
            if let Some(obs) = extract_nav2_pvt_obs(f) {
                if obs.fix_ok && !obs.invalid_llh {
                    nav_events.push((t_ms, obs));
                }
            }
        }
    }
    alg_events.sort_by(|a, b| {
        a.t_ms
            .partial_cmp(&b.t_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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

    let mut out_roll = Vec::<[f64; 2]>::new();
    let mut out_pitch = Vec::<[f64; 2]>::new();
    let mut out_yaw = Vec::<[f64; 2]>::new();
    let mut ref_roll = Vec::<[f64; 2]>::new();
    let mut ref_pitch = Vec::<[f64; 2]>::new();
    let mut ref_yaw = Vec::<[f64; 2]>::new();
    let mut diag_course = Vec::<[f64; 2]>::new();
    let mut diag_lat = Vec::<[f64; 2]>::new();
    let mut diag_long = Vec::<[f64; 2]>::new();
    let mut cls_stationary = Vec::<[f64; 2]>::new();
    let mut cls_turn = Vec::<[f64; 2]>::new();
    let mut cls_long = Vec::<[f64; 2]>::new();
    let mut upd_gravity = Vec::<[f64; 2]>::new();
    let mut upd_turn_gyro = Vec::<[f64; 2]>::new();
    let mut upd_course = Vec::<[f64; 2]>::new();
    let mut upd_lat = Vec::<[f64; 2]>::new();
    let mut upd_long = Vec::<[f64; 2]>::new();
    let mut q0_tr = Vec::<[f64; 2]>::new();
    let mut q1_tr = Vec::<[f64; 2]>::new();
    let mut q2_tr = Vec::<[f64; 2]>::new();
    let mut q3_tr = Vec::<[f64; 2]>::new();
    let mut p00 = Vec::<[f64; 2]>::new();
    let mut p11 = Vec::<[f64; 2]>::new();
    let mut p22 = Vec::<[f64; 2]>::new();

    for ev in &alg_events {
        let t = rel_s(ev.t_ms);
        ref_roll.push([t, ev.roll_deg]);
        ref_pitch.push([t, ev.pitch_deg]);
        ref_yaw.push([t, ev.yaw_deg]);
    }

    let cfg = AlignConfig::default();
    let mut align = Align::new(cfg);
    let bootstrap_cfg = BootstrapConfig {
        ema_alpha: 0.05,
        max_speed_mps: 0.35,
        stationary_samples: 100,
        max_gyro_radps: cfg.max_stationary_gyro_radps,
        max_accel_norm_err_mps2: cfg.max_stationary_accel_norm_err_mps2,
    };
    let mut bootstrap = BootstrapDetector::new(bootstrap_cfg);
    let mut align_initialized = false;
    let mut scan_idx = 0usize;
    let mut interval_start_idx = 0usize;
    let mut prev_nav: Option<(f64, NavPvtObs)> = None;
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
                let mean_gyro_b = [
                    gyro_sum[0] * inv_n,
                    gyro_sum[1] * inv_n,
                    gyro_sum[2] * inv_n,
                ];
                let mean_accel_b = [
                    accel_sum[0] * inv_n,
                    accel_sum[1] * inv_n,
                    accel_sum[2] * inv_n,
                ];
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

                let t = rel_s(*tn);
                let gyro_norm = (mean_gyro_b[0] * mean_gyro_b[0]
                    + mean_gyro_b[1] * mean_gyro_b[1]
                    + mean_gyro_b[2] * mean_gyro_b[2])
                    .sqrt();
                let accel_norm = (mean_accel_b[0] * mean_accel_b[0]
                    + mean_accel_b[1] * mean_accel_b[1]
                    + mean_accel_b[2] * mean_accel_b[2])
                    .sqrt();
                let speed_prev = (v_prev[0] * v_prev[0] + v_prev[1] * v_prev[1]).sqrt() as f32;
                let speed_curr = (v_curr[0] * v_curr[0] + v_curr[1] * v_curr[1]).sqrt() as f32;
                let speed_mid = 0.5_f32 * (speed_prev + speed_curr);
                let stationary = gyro_norm <= cfg.max_stationary_gyro_radps
                    && (accel_norm - GRAVITY_MPS2).abs() <= cfg.max_stationary_accel_norm_err_mps2
                    && speed_mid < 0.5;
                let turn_valid = speed_mid > cfg.min_speed_mps
                    && course_rate_dps.abs() > cfg.min_turn_rate_radps.to_degrees() as f64
                    && a_lat.abs() > cfg.min_lat_acc_mps2 as f64;
                let long_valid = speed_mid > cfg.min_speed_mps
                    && a_long.abs() > cfg.min_long_acc_mps2 as f64
                    && a_lat.abs() < (0.5_f64).max(0.6 * a_long.abs());

                diag_course.push([t, course_rate_dps]);
                diag_lat.push([t, a_lat]);
                diag_long.push([t, a_long]);
                cls_stationary.push([t, if stationary { 1.0 } else { 0.0 }]);
                cls_turn.push([t, if turn_valid { 1.0 } else { 0.0 }]);
                cls_long.push([t, if long_valid { 1.0 } else { 0.0 }]);
                upd_gravity.push([t, if cfg.use_gravity && stationary { 1.0 } else { 0.0 }]);
                upd_turn_gyro.push([t, if cfg.use_turn_gyro && turn_valid { 1.0 } else { 0.0 }]);
                upd_course.push([t, if cfg.use_course_rate && turn_valid { 1.0 } else { 0.0 }]);
                upd_lat.push([t, if cfg.use_lateral_accel && turn_valid { 1.0 } else { 0.0 }]);
                upd_long.push([
                    t,
                    if cfg.use_longitudinal_accel && long_valid {
                        1.0
                    } else {
                        0.0
                    },
                ]);

                let q = align.q_vb;
                let q_plot = [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64];
                let (r, p, y) = quat_rpy_alg_deg(q_plot[0], q_plot[1], q_plot[2], q_plot[3]);
                out_roll.push([t, r]);
                out_pitch.push([t, p]);
                out_yaw.push([t, y]);
                q0_tr.push([t, q[0] as f64]);
                q1_tr.push([t, q[1] as f64]);
                q2_tr.push([t, q[2] as f64]);
                q3_tr.push([t, q[3] as f64]);
                p00.push([t, align.P[0][0] as f64]);
                p11.push([t, align.P[1][1] as f64]);
                p22.push([t, align.P[2][2] as f64]);
            }
        }
        prev_nav = Some((*tn, *nav));
        interval_start_idx = scan_idx;
    }

    AlignCompareData {
        cmp_att: vec![
            Trace {
                name: "Align roll [deg]".to_string(),
                points: out_roll,
            },
            Trace {
                name: "Align pitch [deg]".to_string(),
                points: out_pitch,
            },
            Trace {
                name: "Align yaw [deg]".to_string(),
                points: out_yaw,
            },
            Trace {
                name: "ESF-ALG roll [deg]".to_string(),
                points: ref_roll,
            },
            Trace {
                name: "ESF-ALG pitch [deg]".to_string(),
                points: ref_pitch,
            },
            Trace {
                name: "ESF-ALG yaw [deg]".to_string(),
                points: ref_yaw,
            },
        ],
        res_vel: vec![
            Trace {
                name: "course rate [deg/s]".to_string(),
                points: diag_course,
            },
            Trace {
                name: "a_lat [m/s^2]".to_string(),
                points: diag_lat,
            },
            Trace {
                name: "a_long [m/s^2]".to_string(),
                points: diag_long,
            },
        ],
        motion: vec![
            Trace {
                name: "class stationary".to_string(),
                points: cls_stationary,
            },
            Trace {
                name: "class turn".to_string(),
                points: cls_turn,
            },
            Trace {
                name: "class longitudinal".to_string(),
                points: cls_long,
            },
            Trace {
                name: "update gravity".to_string(),
                points: upd_gravity,
            },
            Trace {
                name: "update turn gyro".to_string(),
                points: upd_turn_gyro,
            },
            Trace {
                name: "update course rate".to_string(),
                points: upd_course,
            },
            Trace {
                name: "update lateral accel".to_string(),
                points: upd_lat,
            },
            Trace {
                name: "update longitudinal accel".to_string(),
                points: upd_long,
            },
        ],
        state_q: vec![
            Trace {
                name: "q0".to_string(),
                points: q0_tr,
            },
            Trace {
                name: "q1".to_string(),
                points: q1_tr,
            },
            Trace {
                name: "q2".to_string(),
                points: q2_tr,
            },
            Trace {
                name: "q3".to_string(),
                points: q3_tr,
            },
        ],
        cov: vec![
            Trace {
                name: "P(0,0)".to_string(),
                points: p00,
            },
            Trace {
                name: "P(1,1)".to_string(),
                points: p11,
            },
            Trace {
                name: "P(2,2)".to_string(),
                points: p22,
            },
        ],
    }
}

// ESF-ALG publishes canonical mount angles in FLU. Converting them as a general quaternion
// over-rotates pitch/yaw. The cross-log-consistent FRD mapping is a roll branch remap only.
fn esf_alg_flu_to_frd_mount_deg(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> (f64, f64, f64) {
    (wrap_deg180(180.0 - roll_deg), pitch_deg, yaw_deg)
}

// FRD Euler extraction in the codebase convention: intrinsic Rx * Ry * Rz.
fn quat_rpy_alg_deg(q0: f64, q1: f64, q2: f64, q3: f64) -> (f64, f64, f64) {
    let n = (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3).sqrt();
    let (w, x, y, z) = if n > 1.0e-12 {
        (q0 / n, q1 / n, q2 / n, q3 / n)
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r01 = 2.0 * (x * y - w * z);
    let r02 = 2.0 * (x * z + w * y);
    let r12 = 2.0 * (y * z - w * x);
    let r22 = 1.0 - 2.0 * (x * x + y * y);
    let pitch = r02.clamp(-1.0, 1.0).asin();
    let roll = (-r12).atan2(r22);
    let yaw = (-r01).atan2(r00);
    (
        roll.to_degrees(),
        pitch.to_degrees(),
        normalize_heading_deg(yaw.to_degrees()),
    )
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
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

fn wrap_deg180(x: f64) -> f64 {
    (x + 180.0).rem_euclid(360.0) - 180.0
}
