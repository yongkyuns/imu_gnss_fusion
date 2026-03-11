use align_rs::align::AlignConfig;

use crate::ubxlog::UbxFrame;

use super::align_replay::{
    BootstrapConfig as ReplayBootstrapConfig, build_align_replay, quat_rpy_alg_deg,
    quat_rotate, signed_projected_axis_angle_deg,
};

use super::super::model::Trace;
use super::timebase::MasterTimeline;

pub struct AlignCompareData {
    pub cmp_att: Vec<Trace>,
    pub res_vel: Vec<Trace>,
    pub axis_err: Vec<Trace>,
    pub motion: Vec<Trace>,
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
    let mut cls_stationary = Vec::<[f64; 2]>::new();
    let mut cls_turn = Vec::<[f64; 2]>::new();
    let mut cls_long = Vec::<[f64; 2]>::new();
    let mut upd_gravity = Vec::<[f64; 2]>::new();
    let mut upd_turn_gyro = Vec::<[f64; 2]>::new();
    let mut upd_course = Vec::<[f64; 2]>::new();
    let mut upd_lat = Vec::<[f64; 2]>::new();
    let mut upd_long = Vec::<[f64; 2]>::new();
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

    for ev in &replay.alg_events {
        let t = rel_s(ev.t_ms);
        let (roll_deg, pitch_deg, yaw_deg) =
            quat_rpy_alg_deg(ev.q_frd[0], ev.q_frd[1], ev.q_frd[2], ev.q_frd[3]);
        ref_roll.push([t, roll_deg]);
        ref_pitch.push([t, pitch_deg]);
        ref_yaw.push([t, yaw_deg]);
    }
    for sample in &replay.samples {
        let t = sample.t_s;
        diag_course.push([t, sample.course_rate_dps]);
        diag_lat.push([t, sample.a_lat_mps2]);
        diag_long.push([t, sample.a_long_mps2]);
        cls_stationary.push([t, if sample.stationary { 1.0 } else { 0.0 }]);
        cls_turn.push([t, if sample.turn_valid { 1.0 } else { 0.0 }]);
        cls_long.push([t, if sample.long_valid { 1.0 } else { 0.0 }]);
        upd_gravity.push([t, if sample.upd_gravity { 1.0 } else { 0.0 }]);
        upd_turn_gyro.push([t, if sample.upd_turn_gyro { 1.0 } else { 0.0 }]);
        upd_course.push([t, if sample.upd_course { 1.0 } else { 0.0 }]);
        upd_lat.push([t, if sample.upd_lat { 1.0 } else { 0.0 }]);
        upd_long.push([t, if sample.upd_long { 1.0 } else { 0.0 }]);

        out_roll.push([t, sample.align_rpy_deg[0]]);
        out_pitch.push([t, sample.align_rpy_deg[1]]);
        out_yaw.push([t, sample.align_rpy_deg[2]]);
        if let Some(q_alg) = sample.alg_q {
            let align_fwd = quat_rotate(sample.q_align, [1.0, 0.0, 0.0]);
            let ref_fwd = quat_rotate(q_alg, [1.0, 0.0, 0.0]);
            let align_down = quat_rotate(sample.q_align, [0.0, 0.0, 1.0]);
            let ref_down = quat_rotate(q_alg, [0.0, 0.0, 1.0]);
            let ref_right = quat_rotate(q_alg, [0.0, 1.0, 0.0]);
            let fwd_signed = signed_projected_axis_angle_deg(align_fwd, ref_fwd, ref_down);
            fwd_err.push([
                t,
                fwd_signed,
            ]);
            down_err.push([
                t,
                signed_projected_axis_angle_deg(align_down, ref_down, ref_right),
            ]);
            if sample.yaw_initialized {
                yaw_init.push([t, fwd_signed]);
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
