use std::cmp::Ordering;

use align_rs::align::{Align, AlignConfig, AlignWindowSummary};
use align_rs::{AlignNhc, AlignNhcConfig};

use crate::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_raw_samples, extract_nav2_pvt_obs,
    sensor_meta,
};

use super::super::math::normalize_heading_deg;
use super::super::model::{ImuPacket, Trace};
use super::align_nhc_bootstrap::resolve_align_nhc_bootstrap_q_vb_seed;
use super::align_replay::{
    esf_alg_flu_to_frd_mount_quat, frd_mount_quat_to_esf_alg_flu_quat, interpolate_alg_quat,
    quat_rotate, quat_rpy_alg_deg, signed_projected_axis_angle_deg,
};
use super::tag_time::fit_tag_ms_map;
use super::timebase::MasterTimeline;

pub struct AlignNhcCompareData {
    pub cmp_att: Vec<Trace>,
    pub diag: Vec<Trace>,
    pub axis_err: Vec<Trace>,
    pub residuals: Vec<Trace>,
    pub gates: Vec<Trace>,
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
    stationary_gyro: Vec<[f32; 3]>,
}

pub fn build_align_nhc_compare_traces(
    frames: &[UbxFrame],
    tl: &MasterTimeline,
) -> AlignNhcCompareData {
    if tl.masters.is_empty() {
        return AlignNhcCompareData {
            cmp_att: Vec::new(),
            diag: Vec::new(),
            axis_err: Vec::new(),
            residuals: Vec::new(),
            gates: Vec::new(),
            cov: Vec::new(),
        };
    }

    let nhc_cfg = AlignNhcConfig::default();
    let bootstrap_cfg = BootstrapConfig {
        ema_alpha: 0.05,
        max_speed_mps: 0.35,
        stationary_samples: 300,
        max_gyro_radps: 0.8_f32.to_radians(),
        max_accel_norm_err_mps2: 0.2,
    };

    let mut alg_events = Vec::<(f64, [f64; 4])>::new();
    let mut alg_raw_events = Vec::<(f64, f64, f64, f64)>::new();
    let mut nav_events = Vec::<(f64, NavPvtObs)>::new();
    for f in frames {
        if let Some((_, roll, pitch, yaw)) = extract_esf_alg(f)
            && let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters)
        {
            alg_events.push((t_ms, esf_alg_flu_to_frd_mount_quat(roll, pitch, yaw)));
            alg_raw_events.push((t_ms, roll, pitch, normalize_heading_deg(yaw)));
        }
        if let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters)
            && let Some(obs) = extract_nav2_pvt_obs(f)
            && obs.fix_ok
            && !obs.invalid_llh
        {
            nav_events.push((t_ms, obs));
        }
    }
    alg_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    alg_raw_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    nav_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let final_alg_q = alg_events.last().map(|ev| ev.1);

    let imu_packets = build_imu_packets(frames, tl);

    let mut nhc = AlignNhc::default();
    let mut bootstrap = BootstrapDetector::new(bootstrap_cfg);
    let mut initialized = false;
    let mut nav_yaw_seeded = false;
    let mut mount_branch_committed = false;
    let mut scan_idx = 0usize;
    let mut interval_start_idx = 0usize;
    let mut prev_nav: Option<(f64, NavPvtObs)> = None;
    let align_cfg = AlignConfig::default();
    let mut align_mount = Align::new(align_cfg);
    let mut align_mount_initialized = false;

    let mut out_roll = Vec::<[f64; 2]>::new();
    let mut out_pitch = Vec::<[f64; 2]>::new();
    let mut out_yaw = Vec::<[f64; 2]>::new();
    let mut ref_roll = Vec::<[f64; 2]>::new();
    let mut ref_pitch = Vec::<[f64; 2]>::new();
    let mut ref_yaw = Vec::<[f64; 2]>::new();

    let mut course_rate = Vec::<[f64; 2]>::new();
    let mut a_lat = Vec::<[f64; 2]>::new();
    let mut a_long = Vec::<[f64; 2]>::new();

    let mut fwd_err = Vec::<[f64; 2]>::new();
    let mut down_err = Vec::<[f64; 2]>::new();

    let mut nhc_vy = Vec::<[f64; 2]>::new();
    let mut nhc_vz = Vec::<[f64; 2]>::new();
    let mut planar_wx = Vec::<[f64; 2]>::new();
    let mut planar_wy = Vec::<[f64; 2]>::new();

    let mut nhc_valid = Vec::<[f64; 2]>::new();
    let mut planar_valid = Vec::<[f64; 2]>::new();

    let mut sigma_mount_r = Vec::<[f64; 2]>::new();
    let mut sigma_mount_p = Vec::<[f64; 2]>::new();
    let mut sigma_mount_y = Vec::<[f64; 2]>::new();
    let mut sigma_nav_r = Vec::<[f64; 2]>::new();
    let mut sigma_nav_p = Vec::<[f64; 2]>::new();
    let mut sigma_nav_y = Vec::<[f64; 2]>::new();
    let mut sigma_vn = Vec::<[f64; 2]>::new();
    let mut sigma_ve = Vec::<[f64; 2]>::new();
    let mut sigma_vd = Vec::<[f64; 2]>::new();

    for (t_ms, roll, pitch, yaw) in &alg_raw_events {
        let t = (*t_ms - tl.t0_master_ms) * 1.0e-3;
        ref_roll.push([t, *roll]);
        ref_pitch.push([t, *pitch]);
        ref_yaw.push([t, *yaw]);
    }

    for (nav_idx, (tn, nav)) in nav_events.iter().enumerate() {
        while scan_idx < imu_packets.len() && imu_packets[scan_idx].t_ms <= *tn {
            let pkt = &imu_packets[scan_idx];
            let gyro_radps = [
                pkt.gx_dps.to_radians() as f32,
                pkt.gy_dps.to_radians() as f32,
                pkt.gz_dps.to_radians() as f32,
            ];
            let accel_b = [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32];
            if !initialized {
                let speed_mps = speed_for_bootstrap(prev_nav, (*tn, *nav), pkt.t_ms) as f32;
                if bootstrap.update(accel_b, gyro_radps, speed_mps) {
                    let seed = resolve_align_nhc_bootstrap_q_vb_seed(
                        &bootstrap.stationary_accel,
                        prev_nav,
                        &nav_events,
                        nav_idx,
                        interval_start_idx,
                        scan_idx + 1,
                        &imu_packets,
                        align_cfg,
                        align_cfg.startup_max_windows.saturating_mul(3),
                    );
                    if nhc
                        .initialize_from_stationary_with_mount_seed_and_sigma(
                            &bootstrap.stationary_accel,
                            &bootstrap.stationary_gyro,
                            0.0,
                            seed.q_vb,
                            seed.sigma_rad,
                        )
                        .is_ok()
                    {
                        initialized = true;
                        if align_mount
                            .initialize_from_stationary(&bootstrap.stationary_accel, 0.0)
                            .is_ok()
                        {
                            align_mount_initialized = true;
                        }
                    }
                }
            }
            scan_idx += 1;
        }

        if initialized && !nav_yaw_seeded {
            let speed_h = (nav.vel_n_mps * nav.vel_n_mps + nav.vel_e_mps * nav.vel_e_mps).sqrt();
            if speed_h >= 5.0 / 3.6 {
                nhc.seed_nav_yaw_from_course(
                    nav.vel_e_mps.atan2(nav.vel_n_mps) as f32,
                    5.0_f32.to_radians(),
                );
                nav_yaw_seeded = true;
            }
        }

        if let Some((t_prev, nav_prev)) = prev_nav {
            let interval_packets = &imu_packets[interval_start_idx..scan_idx];
            if initialized && !interval_packets.is_empty() {
                let mut gyro_sum = [0.0_f32; 3];
                let mut accel_sum = [0.0_f32; 3];
                let mut last_t = t_prev;
                for pkt in interval_packets {
                    gyro_sum[0] += pkt.gx_dps.to_radians() as f32;
                    gyro_sum[1] += pkt.gy_dps.to_radians() as f32;
                    gyro_sum[2] += pkt.gz_dps.to_radians() as f32;
                    accel_sum[0] += pkt.ax_mps2 as f32;
                    accel_sum[1] += pkt.ay_mps2 as f32;
                    accel_sum[2] += pkt.az_mps2 as f32;
                    let dt = ((pkt.t_ms - last_t) * 1.0e-3).max(1.0e-3) as f32;
                    nhc.predict_imu(
                        dt,
                        [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32],
                        [
                            pkt.gx_dps.to_radians() as f32,
                            pkt.gy_dps.to_radians() as f32,
                            pkt.gz_dps.to_radians() as f32,
                        ],
                    );
                    last_t = pkt.t_ms;
                }

                let v_prev = [nav_prev.vel_n_mps, nav_prev.vel_e_mps];
                let v_curr = [nav.vel_n_mps, nav.vel_e_mps];
                let dt_nav = ((*tn - t_prev) * 1.0e-3).max(1.0e-3);
                let course_prev = v_prev[1].atan2(v_prev[0]);
                let course_curr = v_curr[1].atan2(v_curr[0]);
                let course_rate_dps = wrap_rad_pi(course_curr - course_prev).to_degrees() / dt_nav;
                let a_n = [
                    (nav.vel_n_mps - nav_prev.vel_n_mps) / dt_nav,
                    (nav.vel_e_mps - nav_prev.vel_e_mps) / dt_nav,
                ];
                let v_mid = [0.5 * (v_prev[0] + v_curr[0]), 0.5 * (v_prev[1] + v_curr[1])];
                let (a_long_s, a_lat_s) = if let Some(t_hat) = normalize2(v_mid) {
                    let lat_hat = [-t_hat[1], t_hat[0]];
                    (
                        t_hat[0] * a_n[0] + t_hat[1] * a_n[1],
                        lat_hat[0] * a_n[0] + lat_hat[1] * a_n[1],
                    )
                } else {
                    (0.0, 0.0)
                };
                let speed_prev = (v_prev[0] * v_prev[0] + v_prev[1] * v_prev[1]).sqrt() as f32;
                let speed_curr = (v_curr[0] * v_curr[0] + v_curr[1] * v_curr[1]).sqrt() as f32;
                let speed_mid = 0.5_f32 * (speed_prev + speed_curr);
                let inv_n = 1.0 / interval_packets.len() as f32;
                if align_mount_initialized {
                    let window = AlignWindowSummary {
                        dt: dt_nav as f32,
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
                    let (_, align_trace) = align_mount.update_window_with_trace(&window);
                    if !mount_branch_committed && align_trace.after_branch_resolve.is_some() {
                        nhc.seed_mount_from_body_to_vehicle(
                            align_mount.q_vb,
                            [
                                align_mount.P[0][0].sqrt().clamp(0.5_f32.to_radians(), 3.0_f32.to_radians()),
                                align_mount.P[1][1].sqrt().clamp(0.5_f32.to_radians(), 3.0_f32.to_radians()),
                                align_mount.P[2][2].sqrt().clamp(1.0_f32.to_radians(), 5.0_f32.to_radians()),
                            ],
                        );
                        mount_branch_committed = true;
                    }
                }
                let turn_valid = speed_mid > nhc_cfg.min_planar_speed_mps
                    && course_rate_dps.abs()
                        > nhc_cfg.min_planar_yaw_rate_radps.to_degrees() as f64;

                let omega_last_b = interval_packets
                    .last()
                    .map(|pkt| {
                        [
                            pkt.gx_dps.to_radians() as f32,
                            pkt.gy_dps.to_radians() as f32,
                            pkt.gz_dps.to_radians() as f32,
                        ]
                    })
                    .unwrap_or([0.0, 0.0, 0.0]);
                let (_, tr) = nhc.update_all(
                    [
                        nav.vel_n_mps as f32,
                        nav.vel_e_mps as f32,
                        nav.vel_d_mps as f32,
                    ],
                    omega_last_b,
                    speed_mid > nhc_cfg.min_nhc_speed_mps,
                    mount_branch_committed && turn_valid,
                );

                let t = (*tn - tl.t0_master_ms) * 1.0e-3;
                let q_align_bv = [
                    nhc.q_vb[0] as f64,
                    nhc.q_vb[1] as f64,
                    nhc.q_vb[2] as f64,
                    nhc.q_vb[3] as f64,
                ];
                let q_align = quat_conj_f64(q_align_bv);
                let q_align_flu = frd_mount_quat_to_esf_alg_flu_quat(q_align);
                let (r, p, y) = quat_rpy_alg_deg(
                    q_align_flu[0],
                    q_align_flu[1],
                    q_align_flu[2],
                    q_align_flu[3],
                );
                out_roll.push([t, r]);
                out_pitch.push([t, p]);
                out_yaw.push([t, y]);
                course_rate.push([t, course_rate_dps]);
                a_lat.push([t, a_lat_s]);
                a_long.push([t, a_long_s]);
                nhc_vy.push([t, tr.nhc_residual_vy_mps as f64]);
                nhc_vz.push([t, tr.nhc_residual_vz_mps as f64]);
                planar_wx.push([t, tr.planar_gyro_residual_x_radps.to_degrees() as f64]);
                planar_wy.push([t, tr.planar_gyro_residual_y_radps.to_degrees() as f64]);
                nhc_valid.push([t, if tr.nhc_valid { 1.0 } else { 0.0 }]);
                planar_valid.push([t, if tr.planar_gyro_valid { 1.0 } else { 0.0 }]);

                sigma_nav_r.push([t, nhc.P[0][0].sqrt().to_degrees() as f64]);
                sigma_nav_p.push([t, nhc.P[1][1].sqrt().to_degrees() as f64]);
                sigma_nav_y.push([t, nhc.P[2][2].sqrt().to_degrees() as f64]);
                sigma_vn.push([t, nhc.P[3][3].sqrt() as f64]);
                sigma_ve.push([t, nhc.P[4][4].sqrt() as f64]);
                sigma_vd.push([t, nhc.P[5][5].sqrt() as f64]);
                sigma_mount_r.push([t, nhc.P[6][6].sqrt().to_degrees() as f64]);
                sigma_mount_p.push([t, nhc.P[7][7].sqrt().to_degrees() as f64]);
                sigma_mount_y.push([t, nhc.P[8][8].sqrt().to_degrees() as f64]);

                if let Some(q_alg) = interpolate_alg_quat(
                    &alg_events
                        .iter()
                        .map(|(t, q)| super::align_replay::AlgEvent {
                            t_ms: *t,
                            q_frd: *q,
                        })
                        .collect::<Vec<_>>(),
                    *tn,
                ) {
                    if let Some(q_ref_final) = final_alg_q {
                        let align_fwd = quat_rotate(q_align, [1.0, 0.0, 0.0]);
                        let align_down = quat_rotate(q_align, [0.0, 0.0, 1.0]);
                        let ref_fwd = quat_rotate(q_ref_final, [1.0, 0.0, 0.0]);
                        let ref_down = quat_rotate(q_ref_final, [0.0, 0.0, 1.0]);
                        let ref_right = quat_rotate(q_ref_final, [0.0, 1.0, 0.0]);
                        fwd_err.push([
                            t,
                            signed_projected_axis_angle_deg(align_fwd, ref_fwd, ref_down),
                        ]);
                        down_err.push([
                            t,
                            signed_projected_axis_angle_deg(align_down, ref_down, ref_right),
                        ]);
                    }
                    let _ = q_alg;
                }
            }
        }
        prev_nav = Some((*tn, *nav));
        interval_start_idx = scan_idx;
    }

    AlignNhcCompareData {
        cmp_att: vec![
            Trace {
                name: "AlignNhc (FLU) roll [deg]".to_string(),
                points: out_roll,
            },
            Trace {
                name: "AlignNhc (FLU) pitch [deg]".to_string(),
                points: out_pitch,
            },
            Trace {
                name: "AlignNhc (FLU) yaw [deg]".to_string(),
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
        diag: vec![
            Trace {
                name: "course rate [deg/s]".to_string(),
                points: course_rate,
            },
            Trace {
                name: "a_lat [m/s^2]".to_string(),
                points: a_lat,
            },
            Trace {
                name: "a_long [m/s^2]".to_string(),
                points: a_long,
            },
        ],
        axis_err: vec![
            Trace {
                name: "forward-axis error signed [deg]".to_string(),
                points: fwd_err,
            },
            Trace {
                name: "down-axis error signed [deg]".to_string(),
                points: down_err,
            },
        ],
        residuals: vec![
            Trace {
                name: "NHC vy residual [m/s]".to_string(),
                points: nhc_vy,
            },
            Trace {
                name: "NHC vz residual [m/s]".to_string(),
                points: nhc_vz,
            },
            Trace {
                name: "planar gyro wx residual [deg/s]".to_string(),
                points: planar_wx,
            },
            Trace {
                name: "planar gyro wy residual [deg/s]".to_string(),
                points: planar_wy,
            },
        ],
        gates: vec![
            Trace {
                name: "NHC valid".to_string(),
                points: nhc_valid,
            },
            Trace {
                name: "planar gyro valid".to_string(),
                points: planar_valid,
            },
        ],
        cov: vec![
            Trace {
                name: "sigma nav roll [deg]".to_string(),
                points: sigma_nav_r,
            },
            Trace {
                name: "sigma nav pitch [deg]".to_string(),
                points: sigma_nav_p,
            },
            Trace {
                name: "sigma nav yaw [deg]".to_string(),
                points: sigma_nav_y,
            },
            Trace {
                name: "sigma vel N [m/s]".to_string(),
                points: sigma_vn,
            },
            Trace {
                name: "sigma vel E [m/s]".to_string(),
                points: sigma_ve,
            },
            Trace {
                name: "sigma vel D [m/s]".to_string(),
                points: sigma_vd,
            },
            Trace {
                name: "sigma mount roll [deg]".to_string(),
                points: sigma_mount_r,
            },
            Trace {
                name: "sigma mount pitch [deg]".to_string(),
                points: sigma_mount_p,
            },
            Trace {
                name: "sigma mount yaw [deg]".to_string(),
                points: sigma_mount_y,
            },
        ],
    }
}

fn quat_conj_f64(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

impl BootstrapDetector {
    fn new(cfg: BootstrapConfig) -> Self {
        Self {
            cfg,
            gyro_ema: None,
            accel_err_ema: None,
            speed_ema: None,
            stationary_accel: Vec::new(),
            stationary_gyro: Vec::new(),
        }
    }

    fn update(&mut self, accel_b: [f32; 3], gyro_radps: [f32; 3], speed_mps: f32) -> bool {
        let gyro_norm = norm3(gyro_radps);
        let accel_err = (norm3(accel_b) - align_rs::align::GRAVITY_MPS2).abs();
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
            self.stationary_gyro.push(gyro_radps);
        } else {
            self.stationary_accel.clear();
            self.stationary_gyro.clear();
        }
        self.stationary_accel.len() >= self.cfg.stationary_samples
    }
}

fn build_imu_packets(frames: &[UbxFrame], tl: &MasterTimeline) -> Vec<ImuPacket> {
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
    imu_packets
}

fn nearest_master_ms(seq: u64, masters: &[(u64, f64)]) -> Option<f64> {
    if masters.is_empty() {
        return None;
    }
    match masters.binary_search_by_key(&seq, |x| x.0) {
        Ok(idx) => Some(masters[idx].1),
        Err(0) => Some(masters[0].1),
        Err(idx) if idx >= masters.len() => Some(masters[masters.len() - 1].1),
        Err(idx) => {
            let a = masters[idx - 1];
            let b = masters[idx];
            if (seq as i128 - a.0 as i128).abs() <= (b.0 as i128 - seq as i128).abs() {
                Some(a.1)
            } else {
                Some(b.1)
            }
        }
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

fn ema_update(prev: Option<f32>, sample: f32, alpha: f32) -> f32 {
    let alpha = alpha.clamp(1.0e-4, 1.0);
    match prev {
        Some(prev) => (1.0 - alpha) * prev + alpha * sample,
        None => sample,
    }
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize2(v: [f64; 2]) -> Option<[f64; 2]> {
    let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if n <= 1.0e-12 {
        None
    } else {
        Some([v[0] / n, v[1] / n])
    }
}

fn wrap_rad_pi(x: f64) -> f64 {
    (x + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}
