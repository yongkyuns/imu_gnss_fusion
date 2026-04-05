use sensor_fusion::align::AlignConfig;

use crate::ubxlog::{UbxFrame, extract_esf_alg};

use super::align_replay::{
    BootstrapConfig as ReplayBootstrapConfig, ImuReplayConfig, axis_angle_deg,
    build_align_replay, frd_mount_quat_to_esf_alg_flu_quat, quat_rotate, quat_rpy_alg_deg,
};

use super::super::math::nearest_master_ms;
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

pub fn build_align_compare_traces(
    frames: &[UbxFrame],
    tl: &MasterTimeline,
    imu_cfg: ImuReplayConfig,
) -> AlignCompareData {
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
    let replay = build_align_replay(frames, tl, cfg, bootstrap_cfg, imu_cfg);
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
    let mut roll_horiz = Vec::<[f64; 2]>::new();
    let mut roll_turn_gyro = Vec::<[f64; 2]>::new();
    let mut roll_lat = Vec::<[f64; 2]>::new();
    let mut pitch_horiz = Vec::<[f64; 2]>::new();
    let mut pitch_turn_gyro = Vec::<[f64; 2]>::new();
    let mut pitch_lat = Vec::<[f64; 2]>::new();
    let mut yaw_horiz = Vec::<[f64; 2]>::new();
    let mut yaw_turn_gyro = Vec::<[f64; 2]>::new();
    let mut yaw_lat = Vec::<[f64; 2]>::new();
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
                let fwd_unsigned = axis_angle_deg(align_fwd, ref_fwd);
                fwd_err.push([t, fwd_unsigned]);
                down_err.push([t, axis_angle_deg(align_down, ref_down)]);
                if sample.yaw_initialized {
                    yaw_init.push([t, fwd_unsigned]);
                }
            }
        }
        let contrib = sample.contrib;
        roll_horiz.push([t, contrib.horiz_accel[0]]);
        roll_turn_gyro.push([t, contrib.turn_gyro[0]]);
        roll_lat.push([t, contrib.lateral_accel[0]]);
        pitch_horiz.push([t, contrib.horiz_accel[1]]);
        pitch_turn_gyro.push([t, contrib.turn_gyro[1]]);
        pitch_lat.push([t, contrib.lateral_accel[1]]);
        yaw_horiz.push([t, contrib.horiz_accel[2]]);
        yaw_turn_gyro.push([t, contrib.turn_gyro[2]]);
        yaw_lat.push([t, contrib.lateral_accel[2]]);
        p00.push([t, sample.p_diag[0].sqrt().to_degrees()]);
        p11.push([t, sample.p_diag[1].sqrt().to_degrees()]);
        p22.push([t, sample.p_diag[2].sqrt().to_degrees()]);
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
        motion: vec![Trace {
            name: "final ESF-ALG heading [deg]".to_string(),
            points: final_alg_heading,
        }],
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
                name: "lateral accel".to_string(),
                points: roll_lat,
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
                name: "lateral accel".to_string(),
                points: pitch_lat,
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
                name: "lateral accel".to_string(),
                points: yaw_lat,
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
