use align_rs::align::AlignConfig;

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
    pub startup_full_angles: Vec<Trace>,
    pub startup_esf_full_angles: Vec<Trace>,
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
            startup_full_angles: Vec::new(),
            startup_esf_full_angles: Vec::new(),
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
    let mut startup_gnss_long = Vec::<[f64; 2]>::new();
    let mut startup_gnss_lat = Vec::<[f64; 2]>::new();
    let mut startup_imu_long = Vec::<[f64; 2]>::new();
    let mut startup_imu_lat = Vec::<[f64; 2]>::new();
    let mut startup_gnss_ang = Vec::<[f64; 2]>::new();
    let mut startup_imu_ang = Vec::<[f64; 2]>::new();
    let mut startup_rot_imu_ang = Vec::<[f64; 2]>::new();
    let mut startup_rot_imu_alt_ang = Vec::<[f64; 2]>::new();
    let mut startup_full_gnss_ang = Vec::<[f64; 2]>::new();
    let mut startup_full_rot_imu_ang = Vec::<[f64; 2]>::new();
    let mut startup_full_current_align_rot_imu_ang = Vec::<[f64; 2]>::new();
    let mut startup_full_final_align_rot_imu_ang = Vec::<[f64; 2]>::new();
    let mut startup_full_speed = Vec::<[f64; 2]>::new();
    let mut startup_esf_full_gnss_ang = Vec::<[f64; 2]>::new();
    let mut startup_esf_full_rot_imu_ang = Vec::<[f64; 2]>::new();
    let mut startup_esf_full_speed = Vec::<[f64; 2]>::new();
    let mut startup_gate = Vec::<[f64; 2]>::new();
    let mut startup_accept = Vec::<[f64; 2]>::new();
    let mut startup_accepted_samples = Vec::<(f64, f64, f64, f64, f64)>::new();
    let mut roll_horiz = Vec::<[f64; 2]>::new();
    let mut roll_turn_gyro = Vec::<[f64; 2]>::new();
    let mut roll_course = Vec::<[f64; 2]>::new();
    let mut roll_lat = Vec::<[f64; 2]>::new();
    let mut roll_long = Vec::<[f64; 2]>::new();
    let mut pitch_horiz = Vec::<[f64; 2]>::new();
    let mut pitch_turn_gyro = Vec::<[f64; 2]>::new();
    let mut pitch_course = Vec::<[f64; 2]>::new();
    let mut pitch_lat = Vec::<[f64; 2]>::new();
    let mut pitch_long = Vec::<[f64; 2]>::new();
    let mut yaw_horiz = Vec::<[f64; 2]>::new();
    let mut yaw_turn_gyro = Vec::<[f64; 2]>::new();
    let mut yaw_course = Vec::<[f64; 2]>::new();
    let mut yaw_lat = Vec::<[f64; 2]>::new();
    let mut yaw_long = Vec::<[f64; 2]>::new();
    let mut p00 = Vec::<[f64; 2]>::new();
    let mut p11 = Vec::<[f64; 2]>::new();
    let mut p22 = Vec::<[f64; 2]>::new();
    let final_alg_heading_deg = final_alg_q.map(|q| quat_rpy_alg_deg(q[0], q[1], q[2], q[3]).2);
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
            startup_accepted_samples.push((t, g_long, g_lat, i_long, i_lat));
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
        let contrib = sample.contrib;
        roll_horiz.push([t, contrib.horiz_accel[0]]);
        roll_turn_gyro.push([t, contrib.turn_gyro[0]]);
        roll_course.push([t, contrib.course_rate[0]]);
        roll_lat.push([t, contrib.lateral_accel[0]]);
        roll_long.push([t, contrib.longitudinal_accel[0]]);
        pitch_horiz.push([t, contrib.horiz_accel[1]]);
        pitch_turn_gyro.push([t, contrib.turn_gyro[1]]);
        pitch_course.push([t, contrib.course_rate[1]]);
        pitch_lat.push([t, contrib.lateral_accel[1]]);
        pitch_long.push([t, contrib.longitudinal_accel[1]]);
        yaw_horiz.push([t, contrib.horiz_accel[2]]);
        yaw_turn_gyro.push([t, contrib.turn_gyro[2]]);
        yaw_course.push([t, contrib.course_rate[2]]);
        yaw_lat.push([t, contrib.lateral_accel[2]]);
        yaw_long.push([t, contrib.longitudinal_accel[2]]);
        p00.push([t, sample.p_diag[0].sqrt().to_degrees()]);
        p11.push([t, sample.p_diag[1].sqrt().to_degrees()]);
        p22.push([t, sample.p_diag[2].sqrt().to_degrees()]);
    }
    let final_align_q = replay.samples.last().map(|s| s.q_align);
    let final_esf_q = final_alg_q;
    let startup_esf_min_accel_mps2 = 0.15 * 9.80665_f64;
    let startup_theta = replay
        .samples
        .iter()
        .find_map(|s| s.startup_trace.emitted_theta_rad);
    if let Some(theta) = startup_theta {
        let theta_cos = theta.cos();
        let theta_sin = theta.sin();
        let theta_alt = theta + std::f64::consts::PI;
        let theta_alt_cos = theta_alt.cos();
        let theta_alt_sin = theta_alt.sin();
        for (t, g_long, g_lat, i_long, i_lat) in &startup_accepted_samples {
            let g_ang = wrap_signed_deg((*g_lat).atan2(*g_long).to_degrees());
            let i_ang = wrap_signed_deg((*i_lat).atan2(*i_long).to_degrees());
            let ri_long = theta_cos * *i_long - theta_sin * *i_lat;
            let ri_lat = theta_sin * *i_long + theta_cos * *i_lat;
            let ri_alt_long = theta_alt_cos * *i_long - theta_alt_sin * *i_lat;
            let ri_alt_lat = theta_alt_sin * *i_long + theta_alt_cos * *i_lat;
            startup_gnss_ang.push([*t, g_ang]);
            startup_imu_ang.push([*t, i_ang]);
            startup_rot_imu_ang.push([*t, wrap_signed_deg(ri_lat.atan2(ri_long).to_degrees())]);
            startup_rot_imu_alt_ang.push([
                *t,
                wrap_signed_deg(ri_alt_lat.atan2(ri_alt_long).to_degrees()),
            ]);
        }
    }
    for sample in &replay.samples {
        let t = sample.t_s;
        let g_long = sample.a_long_mps2;
        let g_lat = sample.a_lat_mps2;
        startup_full_speed.push([t, sample.speed_mps * 3.6]);
        if g_long.is_finite() && g_lat.is_finite() {
            startup_full_gnss_ang.push([t, wrap_signed_deg(g_lat.atan2(g_long).to_degrees())]);
        }

        let accel_v_align = quat_rotate(
            [
                sample.q_align[0],
                -sample.q_align[1],
                -sample.q_align[2],
                -sample.q_align[3],
            ],
            sample.horiz_accel_b,
        );
        startup_full_current_align_rot_imu_ang.push([
            t,
            wrap_signed_deg(accel_v_align[1].atan2(accel_v_align[0]).to_degrees()),
        ]);

        if let Some(theta) = startup_theta {
            let theta_cos = theta.cos();
            let theta_sin = theta.sin();
            let i_long = sample.startup_input_long_mps2;
            let i_lat = sample.startup_input_lat_mps2;
            if i_long.is_finite() && i_lat.is_finite() {
                let ri_long = theta_cos * i_long - theta_sin * i_lat;
                let ri_lat = theta_sin * i_long + theta_cos * i_lat;
                startup_full_rot_imu_ang.push([
                    t,
                    wrap_signed_deg(ri_lat.atan2(ri_long).to_degrees()),
                ]);
            }
            if let Some(q_final) = final_align_q {
                let accel_v_final = quat_rotate(
                    [q_final[0], -q_final[1], -q_final[2], -q_final[3]],
                    sample.horiz_accel_b,
                );
                startup_full_final_align_rot_imu_ang.push([
                    t,
                    wrap_signed_deg(accel_v_final[1].atan2(accel_v_final[0]).to_degrees()),
                ]);
            }
        }

        if let Some(q_esf_final) = final_esf_q {
            let accel_v_esf_final = quat_rotate(
                [q_esf_final[0], -q_esf_final[1], -q_esf_final[2], -q_esf_final[3]],
                sample.horiz_accel_b,
            );
            let gnss_norm = (g_long * g_long + g_lat * g_lat).sqrt();
            let imu_norm =
                (accel_v_esf_final[0] * accel_v_esf_final[0]
                    + accel_v_esf_final[1] * accel_v_esf_final[1])
                    .sqrt();
            if gnss_norm >= startup_esf_min_accel_mps2
                || imu_norm >= startup_esf_min_accel_mps2
            {
                startup_esf_full_gnss_ang.push([
                    t,
                    wrap_signed_deg(g_lat.atan2(g_long).to_degrees()),
                ]);
                startup_esf_full_rot_imu_ang.push([
                    t,
                    wrap_signed_deg(
                        accel_v_esf_final[1].atan2(accel_v_esf_final[0]).to_degrees(),
                    ),
                ]);
                startup_esf_full_speed.push([t, sample.speed_mps * 3.6]);
            } else {
                startup_esf_full_gnss_ang.push([t, f64::NAN]);
                startup_esf_full_rot_imu_ang.push([t, f64::NAN]);
                startup_esf_full_speed.push([t, f64::NAN]);
            }
        }
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
                name: "Rotated IMU accel angle [deg]".to_string(),
                points: startup_rot_imu_ang,
            },
            Trace {
                name: "Rotated IMU accel angle +180 [deg]".to_string(),
                points: startup_rot_imu_alt_ang,
            },
        ],
        startup_full_angles: vec![
            Trace {
                name: "GNSS accel angle [deg]".to_string(),
                points: startup_full_gnss_ang,
            },
            Trace {
                name: "Current Align rotated IMU accel angle [deg]".to_string(),
                points: startup_full_current_align_rot_imu_ang,
            },
            Trace {
                name: "Rotated IMU accel angle [deg]".to_string(),
                points: startup_full_rot_imu_ang,
            },
            Trace {
                name: "Final Align rotated IMU accel angle [deg]".to_string(),
                points: startup_full_final_align_rot_imu_ang,
            },
            Trace {
                name: "speed [km/h]".to_string(),
                points: startup_full_speed,
            },
        ],
        startup_esf_full_angles: vec![
            Trace {
                name: "GNSS accel angle [deg]".to_string(),
                points: startup_esf_full_gnss_ang,
            },
            Trace {
                name: "Final ESF-ALG rotated IMU accel angle [deg]".to_string(),
                points: startup_esf_full_rot_imu_ang,
            },
            Trace {
                name: "speed [km/h]".to_string(),
                points: startup_esf_full_speed,
            },
        ],
        roll_contrib: vec![
            Trace {
                name: "horiz accel".to_string(),
                points: roll_horiz,
            },
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
                name: "horiz accel".to_string(),
                points: pitch_horiz,
            },
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
                name: "horiz accel".to_string(),
                points: yaw_horiz,
            },
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
                name: "sigma roll [deg]".to_string(),
                points: p00,
            },
            Trace {
                name: "sigma pitch [deg]".to_string(),
                points: p11,
            },
            Trace {
                name: "sigma yaw [deg]".to_string(),
                points: p22,
            },
        ],
    }
}

fn wrap_signed_deg(x: f64) -> f64 {
    (x + 180.0).rem_euclid(360.0) - 180.0
}
