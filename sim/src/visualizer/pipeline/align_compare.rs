use align_rs::align::AlignConfig;
use std::collections::VecDeque;

use crate::ubxlog::{UbxFrame, extract_esf_alg};

use super::align_replay::{
    BootstrapConfig as ReplayBootstrapConfig, build_align_replay,
    frd_mount_quat_to_esf_alg_flu_quat, quat_rotate, quat_rpy_alg_deg,
    signed_projected_axis_angle_deg,
};

use super::super::math::nearest_master_ms;
use super::super::model::Trace;
use super::timebase::MasterTimeline;

pub struct AlignCompareData {
    pub cmp_att: Vec<Trace>,
    pub res_vel: Vec<Trace>,
    pub axis_err: Vec<Trace>,
    pub motion: Vec<Trace>,
    pub startup: Vec<Trace>,
    pub startup_angles: Vec<Trace>,
    pub pca_vectors: Vec<Trace>,
    pub roll_contrib: Vec<Trace>,
    pub pitch_contrib: Vec<Trace>,
    pub yaw_contrib: Vec<Trace>,
    pub cov: Vec<Trace>,
}

pub fn build_align_compare_traces(frames: &[UbxFrame], tl: &MasterTimeline) -> AlignCompareData {
    if tl.masters.is_empty() {
        return AlignCompareData {
            cmp_att: Vec::new(),
            res_vel: Vec::new(),
            axis_err: Vec::new(),
            motion: Vec::new(),
            startup: Vec::new(),
            startup_angles: Vec::new(),
            pca_vectors: Vec::new(),
            roll_contrib: Vec::new(),
            pitch_contrib: Vec::new(),
            yaw_contrib: Vec::new(),
            cov: Vec::new(),
        };
    }
    let rel_s = |master_ms: f64| (master_ms - tl.t0_master_ms) * 1e-3;
    let cfg = AlignConfig::default();
    let bootstrap_cfg = ReplayBootstrapConfig {
        ema_alpha: 0.05,
        max_speed_mps: 0.35,
        stationary_samples: 300,
        max_gyro_radps: cfg.max_stationary_gyro_radps,
        max_accel_norm_err_mps2: cfg.max_stationary_accel_norm_err_mps2,
    };
    let replay = build_align_replay(frames, tl, cfg, bootstrap_cfg);
    let final_alg_q = replay.final_alg_q;

    let mut out_roll = Vec::<[f64; 2]>::new();
    let mut out_pitch = Vec::<[f64; 2]>::new();
    let mut out_yaw = Vec::<[f64; 2]>::new();
    let mut ref_roll = Vec::<[f64; 2]>::new();
    let mut ref_pitch = Vec::<[f64; 2]>::new();
    let mut ref_yaw = Vec::<[f64; 2]>::new();
    let mut diag_course = Vec::<[f64; 2]>::new();
    let mut diag_lat = Vec::<[f64; 2]>::new();
    let mut diag_long = Vec::<[f64; 2]>::new();
    let mut fwd_err = Vec::<[f64; 2]>::new();
    let mut down_err = Vec::<[f64; 2]>::new();
    let mut yaw_init = Vec::<[f64; 2]>::new();
    let mut final_alg_heading = Vec::<[f64; 2]>::new();
    let mut instantaneous_pca_heading = Vec::<[f64; 2]>::new();
    let mut cumulative_pca_heading = Vec::<[f64; 2]>::new();
    let mut startup_gnss_long = Vec::<[f64; 2]>::new();
    let mut startup_gnss_lat = Vec::<[f64; 2]>::new();
    let mut startup_imu_long = Vec::<[f64; 2]>::new();
    let mut startup_imu_lat = Vec::<[f64; 2]>::new();
    let mut startup_gnss_ang = Vec::<[f64; 2]>::new();
    let mut startup_imu_ang = Vec::<[f64; 2]>::new();
    let mut startup_ang_err = Vec::<[f64; 2]>::new();
    let mut startup_gate = Vec::<[f64; 2]>::new();
    let mut startup_accept = Vec::<[f64; 2]>::new();
    let mut imu_pca_points = Vec::<[f64; 2]>::new();
    let mut gnss_pca_points = Vec::<[f64; 2]>::new();
    let mut roll_turn_gyro = Vec::<[f64; 2]>::new();
    let mut roll_course = Vec::<[f64; 2]>::new();
    let mut roll_lat = Vec::<[f64; 2]>::new();
    let mut roll_long = Vec::<[f64; 2]>::new();
    let mut pitch_turn_gyro = Vec::<[f64; 2]>::new();
    let mut pitch_course = Vec::<[f64; 2]>::new();
    let mut pitch_lat = Vec::<[f64; 2]>::new();
    let mut pitch_long = Vec::<[f64; 2]>::new();
    let mut yaw_turn_gyro = Vec::<[f64; 2]>::new();
    let mut yaw_course = Vec::<[f64; 2]>::new();
    let mut yaw_lat = Vec::<[f64; 2]>::new();
    let mut yaw_long = Vec::<[f64; 2]>::new();
    let mut p00 = Vec::<[f64; 2]>::new();
    let mut p11 = Vec::<[f64; 2]>::new();
    let mut p22 = Vec::<[f64; 2]>::new();
    let final_alg_heading_deg = final_alg_q.map(|q| quat_rpy_alg_deg(q[0], q[1], q[2], q[3]).2);
    let mut pca_sxx = 0.0_f64;
    let mut pca_sxy = 0.0_f64;
    let mut pca_syy = 0.0_f64;
    let mut pca_corr = 0.0_f64;
    let mut pca_count = 0usize;
    let mut cumulative_pca_flip = None::<bool>;
    let mut instantaneous_pca_flip = None::<bool>;
    let mut pca_window = VecDeque::<(f64, f64, f64, f64)>::new();

    for f in frames {
        if let Some((_, roll_deg, pitch_deg, yaw_deg)) = extract_esf_alg(f)
            && let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters)
        {
            let t = rel_s(t_ms);
            ref_roll.push([t, roll_deg]);
            ref_pitch.push([t, pitch_deg]);
            ref_yaw.push([t, yaw_deg]);
        }
    }
    for sample in &replay.samples {
        let t = sample.t_s;
        diag_course.push([t, sample.course_rate_dps]);
        diag_lat.push([t, sample.a_lat_mps2]);
        diag_long.push([t, sample.a_long_mps2]);
        if sample.startup_trace.gate_valid {
            startup_gate.push([t, 1.0]);
        }
        if sample.startup_trace.accepted {
            startup_accept.push([t, 1.0]);
            let g_long = sample.startup_trace.gnss_long_lp_mps2;
            let g_lat = sample.startup_trace.gnss_lat_lp_mps2;
            let i_long = sample.startup_trace.imu_long_lp_mps2;
            let i_lat = sample.startup_trace.imu_lat_lp_mps2;
            startup_gnss_long.push([t, g_long]);
            startup_gnss_lat.push([t, g_lat]);
            startup_imu_long.push([t, i_long]);
            startup_imu_lat.push([t, i_lat]);
            let g_ang = wrap_signed_deg(g_lat.atan2(g_long).to_degrees());
            let i_ang = wrap_signed_deg(i_lat.atan2(i_long).to_degrees());
            startup_gnss_ang.push([t, g_ang]);
            startup_imu_ang.push([t, i_ang]);
            startup_ang_err.push([t, wrap_signed_deg(i_ang - g_ang)]);
        }
        if let Some(yaw_deg) = final_alg_heading_deg {
            final_alg_heading.push([t, yaw_deg]);
        }

        let q_align_flu = frd_mount_quat_to_esf_alg_flu_quat(sample.q_align);
        let (align_roll_deg, align_pitch_deg, align_yaw_deg) = quat_rpy_alg_deg(
            q_align_flu[0],
            q_align_flu[1],
            q_align_flu[2],
            q_align_flu[3],
        );
        out_roll.push([t, align_roll_deg]);
        out_pitch.push([t, align_pitch_deg]);
        out_yaw.push([t, align_yaw_deg]);
        if sample.alg_q.is_some() {
            let align_fwd = quat_rotate(sample.q_align, [1.0, 0.0, 0.0]);
            let align_down = quat_rotate(sample.q_align, [0.0, 0.0, 1.0]);
            if let Some(q_ref_final) = final_alg_q {
                let ref_fwd = quat_rotate(q_ref_final, [1.0, 0.0, 0.0]);
                let ref_down = quat_rotate(q_ref_final, [0.0, 0.0, 1.0]);
                let ref_right = quat_rotate(q_ref_final, [0.0, 1.0, 0.0]);
                let fwd_signed = signed_projected_axis_angle_deg(align_fwd, ref_fwd, ref_down);
                fwd_err.push([t, fwd_signed]);
                down_err.push([
                    t,
                    signed_projected_axis_angle_deg(align_down, ref_down, ref_right),
                ]);
                if sample.yaw_initialized {
                    yaw_init.push([t, fwd_signed]);
                }
            }
        }
        let x = sample.pca_input_long_mps2;
        let y = sample.pca_input_lat_mps2;
        if x.is_finite() && y.is_finite() {
            imu_pca_points.push([sample.pca_input_lat_mps2, sample.pca_input_long_mps2]);
            gnss_pca_points.push([sample.a_lat_mps2, sample.a_long_mps2]);
            pca_window.push_back((x, y, sample.a_long_mps2, sample.a_lat_mps2));
            while pca_window.len() > cfg.pca_max_windows {
                pca_window.pop_front();
            }
            pca_sxx += x * x;
            pca_sxy += x * y;
            pca_syy += y * y;
            pca_count += 1;
            let theta = 0.5 * (2.0 * pca_sxy).atan2(pca_sxx - pca_syy);
            let axis = [theta.cos(), theta.sin()];
            pca_corr += sample.a_long_mps2 * axis[0] + sample.a_lat_mps2 * axis[1];
            if pca_count >= cfg.pca_min_windows {
                let trace_cov = pca_sxx + pca_syy;
                let disc =
                    ((pca_sxx - pca_syy) * (pca_sxx - pca_syy) + 4.0 * pca_sxy * pca_sxy).sqrt();
                let lambda_max = 0.5 * (trace_cov + disc);
                let lambda_min = (0.5 * (trace_cov - disc)).max(1.0e-9);
                let anisotropy = lambda_max / lambda_min;
                if anisotropy >= cfg.pca_min_anisotropy_ratio as f64 {
                    let flip = *cumulative_pca_flip.get_or_insert(pca_corr < 0.0);
                    let signed_theta_deg = if flip {
                        wrap_heading_deg(theta.to_degrees() + 180.0)
                    } else {
                        wrap_heading_deg(theta.to_degrees())
                    };
                    cumulative_pca_heading.push([
                        t,
                        wrap_heading_deg(sample.align_rpy_deg[2] + signed_theta_deg),
                    ]);
                }
            }
            if pca_window.len() >= cfg.pca_min_windows {
                let mut wsxx = 0.0_f64;
                let mut wsxy = 0.0_f64;
                let mut wsyy = 0.0_f64;
                for (wx, wy, _, _) in &pca_window {
                    wsxx += wx * wx;
                    wsxy += wx * wy;
                    wsyy += wy * wy;
                }
                let wtrace = wsxx + wsyy;
                let wdisc = ((wsxx - wsyy) * (wsxx - wsyy) + 4.0 * wsxy * wsxy).sqrt();
                let wlambda_max = 0.5 * (wtrace + wdisc);
                let wlambda_min = (0.5 * (wtrace - wdisc)).max(1.0e-9);
                let wanisotropy = wlambda_max / wlambda_min;
                if wanisotropy >= cfg.pca_min_anisotropy_ratio as f64 {
                    let wtheta = 0.5 * (2.0 * wsxy).atan2(wsxx - wsyy);
                    let waxis = [wtheta.cos(), wtheta.sin()];
                    let mut wcorr = 0.0_f64;
                    for (_, _, wlong, wlat) in &pca_window {
                        wcorr += wlong * waxis[0] + wlat * waxis[1];
                    }
                    let flip = *instantaneous_pca_flip.get_or_insert(wcorr < 0.0);
                    let wsigned_theta_deg = if flip {
                        wrap_heading_deg(wtheta.to_degrees() + 180.0)
                    } else {
                        wrap_heading_deg(wtheta.to_degrees())
                    };
                    instantaneous_pca_heading.push([
                        t,
                        wrap_heading_deg(sample.align_rpy_deg[2] + wsigned_theta_deg),
                    ]);
                }
            }
        }
        let contrib = sample.contrib;
        roll_turn_gyro.push([t, contrib.turn_gyro[0]]);
        roll_course.push([t, contrib.course_rate[0]]);
        roll_lat.push([t, contrib.lateral_accel[0]]);
        roll_long.push([t, contrib.longitudinal_accel[0]]);
        pitch_turn_gyro.push([t, contrib.turn_gyro[1]]);
        pitch_course.push([t, contrib.course_rate[1]]);
        pitch_lat.push([t, contrib.lateral_accel[1]]);
        pitch_long.push([t, contrib.longitudinal_accel[1]]);
        yaw_turn_gyro.push([t, contrib.turn_gyro[2]]);
        yaw_course.push([t, contrib.course_rate[2]]);
        yaw_lat.push([t, contrib.lateral_accel[2]]);
        yaw_long.push([t, contrib.longitudinal_accel[2]]);
        p00.push([t, sample.p_diag[0]]);
        p11.push([t, sample.p_diag[1]]);
        p22.push([t, sample.p_diag[2]]);
    }

    AlignCompareData {
        cmp_att: vec![
            Trace {
                name: "Align (FLU) roll [deg]".to_string(),
                points: out_roll,
            },
            Trace {
                name: "Align (FLU) pitch [deg]".to_string(),
                points: out_pitch,
            },
            Trace {
                name: "Align (FLU) yaw [deg]".to_string(),
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
        axis_err: vec![
            Trace {
                name: "forward-axis error signed [deg]".to_string(),
                points: fwd_err,
            },
            Trace {
                name: "down-axis error signed [deg]".to_string(),
                points: down_err,
            },
            Trace {
                name: "yaw initialized".to_string(),
                points: yaw_init,
            },
        ],
        motion: vec![
            Trace {
                name: "final ESF-ALG heading [deg]".to_string(),
                points: final_alg_heading,
            },
            Trace {
                name: "instantaneous PCA heading (GNSS-sign) [deg]".to_string(),
                points: instantaneous_pca_heading,
            },
            Trace {
                name: "cumulative PCA heading (GNSS-sign) [deg]".to_string(),
                points: cumulative_pca_heading,
            },
        ],
        startup: vec![
            Trace {
                name: "GNSS long LP [m/s^2]".to_string(),
                points: startup_gnss_long,
            },
            Trace {
                name: "GNSS lat LP [m/s^2]".to_string(),
                points: startup_gnss_lat,
            },
            Trace {
                name: "IMU long LP [m/s^2]".to_string(),
                points: startup_imu_long,
            },
            Trace {
                name: "IMU lat LP [m/s^2]".to_string(),
                points: startup_imu_lat,
            },
            Trace {
                name: "startup gate valid".to_string(),
                points: startup_gate,
            },
            Trace {
                name: "startup accepted".to_string(),
                points: startup_accept,
            },
        ],
        startup_angles: vec![
            Trace {
                name: "GNSS accel angle [deg]".to_string(),
                points: startup_gnss_ang,
            },
            Trace {
                name: "IMU accel angle [deg]".to_string(),
                points: startup_imu_ang,
            },
            Trace {
                name: "IMU-GNSS accel angle err [deg]".to_string(),
                points: startup_ang_err,
            },
        ],
        pca_vectors: pca_vector_traces(&imu_pca_points, &gnss_pca_points),
        roll_contrib: vec![
            Trace {
                name: "turn gyro".to_string(),
                points: roll_turn_gyro,
            },
            Trace {
                name: "course rate".to_string(),
                points: roll_course,
            },
            Trace {
                name: "lateral accel".to_string(),
                points: roll_lat,
            },
            Trace {
                name: "longitudinal accel".to_string(),
                points: roll_long,
            },
        ],
        pitch_contrib: vec![
            Trace {
                name: "turn gyro".to_string(),
                points: pitch_turn_gyro,
            },
            Trace {
                name: "course rate".to_string(),
                points: pitch_course,
            },
            Trace {
                name: "lateral accel".to_string(),
                points: pitch_lat,
            },
            Trace {
                name: "longitudinal accel".to_string(),
                points: pitch_long,
            },
        ],
        yaw_contrib: vec![
            Trace {
                name: "turn gyro".to_string(),
                points: yaw_turn_gyro,
            },
            Trace {
                name: "course rate".to_string(),
                points: yaw_course,
            },
            Trace {
                name: "lateral accel".to_string(),
                points: yaw_lat,
            },
            Trace {
                name: "longitudinal accel".to_string(),
                points: yaw_long,
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

fn pca_vector_traces(imu_points: &[[f64; 2]], gnss_points: &[[f64; 2]]) -> Vec<Trace> {
    let mut out = vec![
        Trace {
            name: "IMU accel points".to_string(),
            points: imu_points.to_vec(),
        },
        Trace {
            name: "GNSS accel points".to_string(),
            points: gnss_points.to_vec(),
        },
    ];
    if let Some(line) = pca_axis_line(imu_points, "IMU PCA axis") {
        out.push(line);
    }
    if let Some(line) = pca_axis_line(gnss_points, "GNSS PCA axis") {
        out.push(line);
    }
    out
}

fn pca_axis_line(points: &[[f64; 2]], name: &str) -> Option<Trace> {
    if points.len() < 2 {
        return None;
    }
    let mut mx = 0.0_f64;
    let mut my = 0.0_f64;
    let mut n = 0usize;
    for p in points {
        if p[0].is_finite() && p[1].is_finite() {
            mx += p[0];
            my += p[1];
            n += 1;
        }
    }
    if n < 2 {
        return None;
    }
    mx /= n as f64;
    my /= n as f64;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    let mut syy = 0.0_f64;
    for p in points {
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        let dx = p[0] - mx;
        let dy = p[1] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let dir = [theta.cos(), theta.sin()];
    let lambda_max = 0.5 * (sxx + syy + ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt());
    let half_len = (lambda_max.max(0.0) / n as f64).sqrt().max(0.1) * 2.0;
    Some(Trace {
        name: name.to_string(),
        points: vec![
            [mx - half_len * dir[0], my - half_len * dir[1]],
            [mx + half_len * dir[0], my + half_len * dir[1]],
        ],
    })
}

fn wrap_heading_deg(x: f64) -> f64 {
    x.rem_euclid(360.0)
}

fn wrap_signed_deg(x: f64) -> f64 {
    (x + 180.0).rem_euclid(360.0) - 180.0
}
