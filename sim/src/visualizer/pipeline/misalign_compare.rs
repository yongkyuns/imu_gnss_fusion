use std::cmp::Ordering;

use align_rs::{AlignMisalign, AlignMisalignConfig};
use ekf_rs::ekf::{Ekf, GpsData, ImuSample, ekf_fuse_gps, ekf_fuse_vehicle_vel, ekf_predict};

use crate::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_raw_samples, extract_nav2_pvt_obs,
    sensor_meta,
};

use super::super::math::{
    clamp_ekf_biases, ecef_to_ned, lla_to_ecef, nearest_master_ms, normalize_heading_deg,
    set_quat_yaw_only,
};
use super::super::model::{ImuPacket, Trace};
use super::align_replay::{
    AlgEvent, axis_angle_deg, frd_mount_quat_to_esf_alg_flu_quat, interpolate_alg_quat,
    quat_rotate, quat_rpy_alg_deg, signed_projected_axis_angle_deg,
};
use super::tag_time::fit_tag_ms_map;
use super::timebase::MasterTimeline;

pub struct MisalignCompareData {
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
}

#[derive(Clone, Copy, Debug)]
struct MisalignYawSeedSample {
    q_nb: [f32; 4],
    v_n: [f32; 3],
    omega_b_corr: [f32; 3],
}

pub fn build_misalign_compare_traces(
    frames: &[UbxFrame],
    tl: &MasterTimeline,
) -> MisalignCompareData {
    const MISALIGN_NAV_VEHICLE_VEL_R: f32 = 100.0;
    const MISALIGN_YAW_SEED_WINDOW_S: f64 = 1.5;
    const MISALIGN_YAW_SEED_MIN_SPEED_MPS: f32 = 5.0 / 3.6;
    const MISALIGN_YAW_SEED_MIN_SAMPLES: usize = 10;
    const MISALIGN_YAW_SEED_STEP_DEG: f32 = 5.0;

    if tl.masters.is_empty() {
        return empty_misalign_compare_data();
    }

    let bootstrap_cfg = BootstrapConfig {
        ema_alpha: 0.05,
        max_speed_mps: 0.35,
        stationary_samples: 300,
        max_gyro_radps: 0.8_f32.to_radians(),
        max_accel_norm_err_mps2: 0.2,
    };

    let mut alg_events = Vec::<AlgEvent>::new();
    let mut nav_events = Vec::<(f64, NavPvtObs)>::new();
    for frame in frames {
        if let Some((_, roll, pitch, yaw)) = extract_esf_alg(frame)
            && let Some(t_ms) = nearest_master_ms(frame.seq, &tl.masters)
        {
            alg_events.push(AlgEvent {
                t_ms,
                q_frd: super::align_replay::esf_alg_flu_to_frd_mount_quat(roll, pitch, yaw),
            });
        }
        if let Some(t_ms) = nearest_master_ms(frame.seq, &tl.masters)
            && let Some(obs) = extract_nav2_pvt_obs(frame)
            && obs.fix_ok
            && !obs.invalid_llh
        {
            nav_events.push((t_ms, obs));
        }
    }
    alg_events.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap_or(Ordering::Equal));
    nav_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    if nav_events.len() < 2 {
        return empty_misalign_compare_data();
    }

    let imu_packets = build_imu_packets(frames, tl);
    if imu_packets.is_empty() {
        return empty_misalign_compare_data();
    }

    let final_alg_q = alg_events.last().map(|ev| ev.q_frd);
    let mut misalign = AlignMisalign::new(AlignMisalignConfig::default());
    let mut bootstrap = BootstrapDetector::new(bootstrap_cfg);
    let mut initialized = false;
    let mut nav_yaw_seeded = false;
    let mut mount_yaw_seeded = false;
    let mut yaw_seed_samples = Vec::<MisalignYawSeedSample>::new();
    let mut init_time_s = f64::NAN;

    let mut ekf = Ekf::default();
    let mut origin_set = false;
    let mut ref_lat = 0.0_f64;
    let mut ref_lon = 0.0_f64;
    let mut ref_ecef = [0.0_f64; 3];
    let mut nav_idx = 0usize;
    let mut next_gps_update_ms = f64::NAN;
    let gps_period_ms = 1000.0 / 20.0;
    let mut prev_nav: Option<(f64, NavPvtObs)> = None;
    let mut prev_imu_t_ms: Option<f64> = None;

    let mut out_roll = Vec::<[f64; 2]>::new();
    let mut out_pitch = Vec::<[f64; 2]>::new();
    let mut out_yaw = Vec::<[f64; 2]>::new();
    let mut ref_roll = Vec::<[f64; 2]>::new();
    let mut ref_pitch = Vec::<[f64; 2]>::new();
    let mut ref_yaw = Vec::<[f64; 2]>::new();

    let mut course_rate = Vec::<[f64; 2]>::new();
    let mut a_lat = Vec::<[f64; 2]>::new();
    let mut a_long = Vec::<[f64; 2]>::new();
    let mut speed_h = Vec::<[f64; 2]>::new();
    let mut omega_yaw = Vec::<[f64; 2]>::new();
    let mut omega_transverse_ratio = Vec::<[f64; 2]>::new();

    let mut fwd_err = Vec::<[f64; 2]>::new();
    let mut down_err = Vec::<[f64; 2]>::new();

    let mut nhc_vy = Vec::<[f64; 2]>::new();
    let mut nhc_vz = Vec::<[f64; 2]>::new();
    let mut planar_wx = Vec::<[f64; 2]>::new();
    let mut planar_wy = Vec::<[f64; 2]>::new();
    let mut yaw_cue_res = Vec::<[f64; 2]>::new();
    let mut yaw_prior_res = Vec::<[f64; 2]>::new();
    let mut yaw_cue_aniso = Vec::<[f64; 2]>::new();

    let mut nhc_valid = Vec::<[f64; 2]>::new();
    let mut planar_valid = Vec::<[f64; 2]>::new();
    let mut yaw_cue_valid = Vec::<[f64; 2]>::new();

    let mut sigma_roll = Vec::<[f64; 2]>::new();
    let mut sigma_pitch = Vec::<[f64; 2]>::new();
    let mut sigma_yaw = Vec::<[f64; 2]>::new();

    for frame in frames {
        if let Some((_, roll, pitch, yaw)) = extract_esf_alg(frame)
            && let Some(t_ms) = nearest_master_ms(frame.seq, &tl.masters)
        {
            let t = (t_ms - tl.t0_master_ms) * 1.0e-3;
            ref_roll.push([t, roll]);
            ref_pitch.push([t, pitch]);
            ref_yaw.push([t, normalize_heading_deg(yaw)]);
        }
    }

    for pkt in &imu_packets {
        let dt = if let Some(t_prev_ms) = prev_imu_t_ms {
            ((pkt.t_ms - t_prev_ms) * 1.0e-3).max(1.0e-3)
        } else {
            0.01
        };
        prev_imu_t_ms = Some(pkt.t_ms);
        let gyro_radps = [
            pkt.gx_dps.to_radians() as f32,
            pkt.gy_dps.to_radians() as f32,
            pkt.gz_dps.to_radians() as f32,
        ];
        let accel_b = [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32];

        if !initialized {
            let speed_mps = nav_events
                .get(nav_idx)
                .copied()
                .or(prev_nav)
                .map(|next_nav| speed_for_bootstrap(prev_nav, next_nav, pkt.t_ms) as f32)
                .unwrap_or(0.0);
            if bootstrap.update(accel_b, gyro_radps, speed_mps)
                && misalign
                    .initialize_from_stationary_with_x_ref(
                        &bootstrap.stationary_accel,
                        0.0,
                        [1.0, 0.0, 0.0],
                    )
                    .is_ok()
            {
                initialized = true;
                init_time_s = (pkt.t_ms - tl.t0_master_ms) * 1.0e-3;
            }
        }

        let imu = ImuSample {
            dax: (gyro_radps[0] as f64 * dt) as f32,
            day: (gyro_radps[1] as f64 * dt) as f32,
            daz: (gyro_radps[2] as f64 * dt) as f32,
            dvx: (accel_b[0] as f64 * dt) as f32,
            dvy: (accel_b[1] as f64 * dt) as f32,
            dvz: (accel_b[2] as f64 * dt) as f32,
            dt: dt as f32,
        };
        ekf_predict(&mut ekf, &imu, None);
        clamp_ekf_biases(&mut ekf, dt);
        if initialized {
            ekf_fuse_vehicle_vel(&mut ekf, misalign.q_vb, MISALIGN_NAV_VEHICLE_VEL_R);
            clamp_ekf_biases(&mut ekf, dt);
        }

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
                ref_ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
                origin_set = true;
            }
            let ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
            let ned = ecef_to_ned(ecef, ref_ecef, ref_lat, ref_lon);
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

            if !initialized {
                prev_nav = Some((t_ms, nav));
                continue;
            }

            if !nav_yaw_seeded {
                let speed_h_mps =
                    (nav.vel_n_mps * nav.vel_n_mps + nav.vel_e_mps * nav.vel_e_mps).sqrt();
                if speed_h_mps >= 5.0 / 3.6 {
                    let course_rad = (nav.vel_e_mps as f32).atan2(nav.vel_n_mps as f32);
                    set_quat_yaw_only(&mut ekf.state, course_rad as f64);
                    let q_nb_seed = [ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3];
                    let mut nav_seed = misalign.clone();
                    nav_seed.seed_mount_from_nav_course_full(
                        q_nb_seed,
                        course_rad,
                        2.0_f32.to_radians(),
                        10.0_f32.to_radians(),
                    );
                    let down_curr = quat_rotate(
                        [
                            misalign.q_vb[0] as f64,
                            misalign.q_vb[1] as f64,
                            misalign.q_vb[2] as f64,
                            misalign.q_vb[3] as f64,
                        ],
                        [0.0, 0.0, 1.0],
                    );
                    let down_seed = quat_rotate(
                        [
                            nav_seed.q_vb[0] as f64,
                            nav_seed.q_vb[1] as f64,
                            nav_seed.q_vb[2] as f64,
                            nav_seed.q_vb[3] as f64,
                        ],
                        [0.0, 0.0, 1.0],
                    );
                    if axis_angle_deg(down_curr, down_seed) <= 10.0 {
                        misalign = nav_seed;
                    }
                    nav_yaw_seeded = true;
                }
            }

            let t_s = (t_ms - tl.t0_master_ms) * 1.0e-3;
            let q_nb = [ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3];
            let v_n = [ekf.state.vn, ekf.state.ve, ekf.state.vd];
            let dt_update = if let Some((t_prev, _)) = prev_nav {
                ((t_ms - t_prev) * 1.0e-3).max(1.0e-3) as f32
            } else {
                0.05
            };
            let dt_safe = dt.max(1.0e-3) as f32;
            let omega_b_corr = [
                gyro_radps[0] - ekf.state.dax_b / dt_safe,
                gyro_radps[1] - ekf.state.day_b / dt_safe,
                gyro_radps[2] - ekf.state.daz_b / dt_safe,
            ];
            let accel_b_corr = [
                accel_b[0] - ekf.state.dvx_b / dt_safe,
                accel_b[1] - ekf.state.dvy_b / dt_safe,
                accel_b[2] - ekf.state.dvz_b / dt_safe,
            ];

            if nav_yaw_seeded && !mount_yaw_seeded {
                if misalign.horizontal_speed_mps(v_n) >= MISALIGN_YAW_SEED_MIN_SPEED_MPS {
                    yaw_seed_samples.push(MisalignYawSeedSample {
                        q_nb,
                        v_n,
                        omega_b_corr,
                    });
                }
                if t_s - init_time_s >= MISALIGN_YAW_SEED_WINDOW_S
                    && yaw_seed_samples.len() >= MISALIGN_YAW_SEED_MIN_SAMPLES
                {
                    if let Some(yaw_seed_rad) = resolve_misalign_yaw_seed(
                        &misalign,
                        &yaw_seed_samples,
                        MISALIGN_YAW_SEED_STEP_DEG,
                    ) {
                        misalign.set_mount_yaw(yaw_seed_rad, 5.0_f32.to_radians());
                        mount_yaw_seeded = true;
                    }
                }
            }
            if !mount_yaw_seeded {
                prev_nav = Some((t_ms, nav));
                continue;
            }

            let (_, tr) = misalign.update_all(dt_update, q_nb, v_n, omega_b_corr, accel_b_corr);
            let q_align = [
                misalign.q_vb[0] as f64,
                misalign.q_vb[1] as f64,
                misalign.q_vb[2] as f64,
                misalign.q_vb[3] as f64,
            ];
            let q_align_flu = frd_mount_quat_to_esf_alg_flu_quat(q_align);
            let (r, p, y) = quat_rpy_alg_deg(
                q_align_flu[0],
                q_align_flu[1],
                q_align_flu[2],
                q_align_flu[3],
            );
            out_roll.push([t_s, r]);
            out_pitch.push([t_s, p]);
            out_yaw.push([t_s, y]);

            let (course_rate_dps, a_lat_s, a_long_s) = if let Some((t_prev, nav_prev)) = prev_nav {
                let dt_nav = ((t_ms - t_prev) * 1.0e-3).max(1.0e-3);
                let v_prev = [nav_prev.vel_n_mps, nav_prev.vel_e_mps];
                let v_curr = [nav.vel_n_mps, nav.vel_e_mps];
                let course_prev = v_prev[1].atan2(v_prev[0]);
                let course_curr = v_curr[1].atan2(v_curr[0]);
                let course_rate =
                    wrap_rad_pi(course_curr - course_prev).to_degrees() / dt_nav.max(1.0e-6);
                let a_n = [
                    (nav.vel_n_mps - nav_prev.vel_n_mps) / dt_nav,
                    (nav.vel_e_mps - nav_prev.vel_e_mps) / dt_nav,
                ];
                let v_mid = [0.5 * (v_prev[0] + v_curr[0]), 0.5 * (v_prev[1] + v_curr[1])];
                if let Some(t_hat) = normalize2(v_mid) {
                    let lat_hat = [-t_hat[1], t_hat[0]];
                    (
                        course_rate,
                        lat_hat[0] * a_n[0] + lat_hat[1] * a_n[1],
                        t_hat[0] * a_n[0] + t_hat[1] * a_n[1],
                    )
                } else {
                    (course_rate, 0.0, 0.0)
                }
            } else {
                (0.0, 0.0, 0.0)
            };
            course_rate.push([t_s, course_rate_dps]);
            a_lat.push([t_s, a_lat_s]);
            a_long.push([t_s, a_long_s]);
            speed_h.push([t_s, tr.speed_h_mps as f64]);
            omega_yaw.push([t_s, tr.omega_v_yaw_abs_radps.to_degrees() as f64]);
            omega_transverse_ratio.push([t_s, tr.omega_v_transverse_ratio as f64]);

            nhc_vy.push([t_s, tr.nhc_residual_vy_mps as f64]);
            nhc_vz.push([t_s, tr.nhc_residual_vz_mps as f64]);
            planar_wx.push([t_s, tr.planar_gyro_residual_x_radps.to_degrees() as f64]);
            planar_wy.push([t_s, tr.planar_gyro_residual_y_radps.to_degrees() as f64]);
            yaw_cue_res.push([t_s, tr.yaw_cue_residual_rad.to_degrees() as f64]);
            yaw_prior_res.push([t_s, tr.yaw_prior_residual_rad.to_degrees() as f64]);
            yaw_cue_aniso.push([t_s, tr.yaw_cue_anisotropy as f64]);

            nhc_valid.push([t_s, if tr.nhc_valid { 1.0 } else { 0.0 }]);
            planar_valid.push([t_s, if tr.planar_gyro_valid { 1.0 } else { 0.0 }]);
            yaw_cue_valid.push([t_s, if tr.yaw_cue_valid { 1.0 } else { 0.0 }]);

            sigma_roll.push([t_s, misalign.P[0][0].sqrt().to_degrees() as f64]);
            sigma_pitch.push([t_s, misalign.P[1][1].sqrt().to_degrees() as f64]);
            sigma_yaw.push([t_s, misalign.P[2][2].sqrt().to_degrees() as f64]);

            if let Some(q_alg) = interpolate_alg_quat(&alg_events, t_ms) {
                let q_ref_axis = final_alg_q.unwrap_or(q_alg);
                let align_fwd = quat_rotate(q_align, [1.0, 0.0, 0.0]);
                let align_down = quat_rotate(q_align, [0.0, 0.0, 1.0]);
                let ref_fwd = quat_rotate(q_ref_axis, [1.0, 0.0, 0.0]);
                let ref_down = quat_rotate(q_ref_axis, [0.0, 0.0, 1.0]);
                let ref_right = quat_rotate(q_ref_axis, [0.0, 1.0, 0.0]);
                fwd_err.push([
                    t_s,
                    signed_projected_axis_angle_deg(align_fwd, ref_fwd, ref_down),
                ]);
                down_err.push([
                    t_s,
                    signed_projected_axis_angle_deg(align_down, ref_down, ref_right),
                ]);
            }

            prev_nav = Some((t_ms, nav));
        }
    }

    MisalignCompareData {
        cmp_att: vec![
            Trace {
                name: "Misalign (FLU) roll [deg]".to_string(),
                points: out_roll,
            },
            Trace {
                name: "Misalign (FLU) pitch [deg]".to_string(),
                points: out_pitch,
            },
            Trace {
                name: "Misalign (FLU) yaw [deg]".to_string(),
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
            Trace {
                name: "speed_h [m/s]".to_string(),
                points: speed_h,
            },
            Trace {
                name: "omega_v yaw abs [deg/s]".to_string(),
                points: omega_yaw,
            },
            Trace {
                name: "omega_v transverse ratio".to_string(),
                points: omega_transverse_ratio,
            },
            Trace {
                name: "yaw cue anisotropy".to_string(),
                points: yaw_cue_aniso,
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
            Trace {
                name: "yaw cue residual [deg]".to_string(),
                points: yaw_cue_res,
            },
            Trace {
                name: "yaw prior residual [deg]".to_string(),
                points: yaw_prior_res,
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
            Trace {
                name: "yaw cue valid".to_string(),
                points: yaw_cue_valid,
            },
        ],
        cov: vec![
            Trace {
                name: "sigma roll [deg]".to_string(),
                points: sigma_roll,
            },
            Trace {
                name: "sigma pitch [deg]".to_string(),
                points: sigma_pitch,
            },
            Trace {
                name: "sigma yaw [deg]".to_string(),
                points: sigma_yaw,
            },
        ],
    }
}

fn empty_misalign_compare_data() -> MisalignCompareData {
    MisalignCompareData {
        cmp_att: Vec::new(),
        diag: Vec::new(),
        axis_err: Vec::new(),
        residuals: Vec::new(),
        gates: Vec::new(),
        cov: Vec::new(),
    }
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
        } else {
            self.stationary_accel.clear();
        }
        self.stationary_accel.len() >= self.cfg.stationary_samples
    }
}

fn resolve_misalign_yaw_seed(
    base: &AlignMisalign,
    samples: &[MisalignYawSeedSample],
    yaw_step_deg: f32,
) -> Option<f32> {
    if samples.is_empty() || yaw_step_deg <= 0.0 {
        return None;
    }
    let mut best_cost = f32::INFINITY;
    let mut best_yaw = None;
    let (_, _, center_yaw_deg) = quat_rpy_alg_deg(
        base.q_vb[0] as f64,
        base.q_vb[1] as f64,
        base.q_vb[2] as f64,
        base.q_vb[3] as f64,
    );
    let search_half_span_deg = 120.0_f32;
    let steps = ((2.0 * search_half_span_deg) / yaw_step_deg).round() as i32;
    for k in 0..=steps {
        let yaw_deg = center_yaw_deg as f32 - search_half_span_deg + k as f32 * yaw_step_deg;
        let mut cand = base.clone();
        cand.set_mount_yaw(wrap_rad_pi(yaw_deg.to_radians() as f64) as f32, 5.0_f32.to_radians());
        let mut cost = 0.0_f32;
        let mut n_terms = 0usize;
        for sample in samples {
            if cand.nhc_gate(sample.v_n) {
                let pred = cand.nhc_prediction(sample.q_nb, sample.v_n);
                let sigma = cand.cfg.r_nhc_std_mps.max(1.0e-6);
                cost += (pred[0] / sigma).powi(2) + (pred[1] / sigma).powi(2);
                n_terms += 2;
            }
            if cand.planar_gyro_gate(sample.v_n, sample.omega_b_corr) {
                let pred = cand.planar_gyro_prediction(sample.omega_b_corr);
                let sigma = cand.cfg.r_planar_gyro_std_radps.max(1.0e-6);
                cost += (pred[0] / sigma).powi(2) + (pred[1] / sigma).powi(2);
                n_terms += 2;
            }
        }
        if n_terms == 0 {
            continue;
        }
        let mean_cost = cost / n_terms as f32;
        if mean_cost < best_cost {
            best_cost = mean_cost;
            best_yaw = Some(yaw_deg.to_radians());
        }
    }
    best_yaw
}

fn build_imu_packets(frames: &[UbxFrame], tl: &MasterTimeline) -> Vec<ImuPacket> {
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
    if n <= 1.0e-9 {
        None
    } else {
        Some([v[0] / n, v[1] / n])
    }
}

fn wrap_rad_pi(x: f64) -> f64 {
    let mut y = x;
    while y > std::f64::consts::PI {
        y -= 2.0 * std::f64::consts::PI;
    }
    while y < -std::f64::consts::PI {
        y += 2.0 * std::f64::consts::PI;
    }
    y
}
