use align_rs::align::{
    MisalignImuSample, MisalignNoise, Align, align_fuse_velocity, align_init, align_predict_gyro, align_q_sb,
};

use crate::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_raw_samples, extract_nav_pvt_obs,
    extract_nav2_pvt_obs, sensor_meta,
};

use super::super::math::{nearest_master_ms, normalize_heading_deg};
use super::super::model::{AlgEvent, ImuPacket, Trace};
use super::tag_time::fit_tag_ms_map;
use super::timebase::MasterTimeline;

pub struct AlignCompareData {
    pub cmp_att: Vec<Trace>,
    pub res_vel: Vec<Trace>,
    pub state_q: Vec<Trace>,
    pub cov: Vec<Trace>,
}

pub fn build_align_compare_traces(frames: &[UbxFrame], tl: &MasterTimeline) -> AlignCompareData {
    if tl.masters.is_empty() {
        return AlignCompareData {
            cmp_att: Vec::new(),
            res_vel: Vec::new(),
            state_q: Vec::new(),
            cov: Vec::new(),
        };
    }
    let rel_s = |master_ms: f64| (master_ms - tl.t0_master_ms) * 1e-3;

    let mut alg_events = Vec::<AlgEvent>::new();
    let mut nav_events_pvt = Vec::<(f64, NavPvtObs)>::new();
    let mut nav_events_nav2 = Vec::<(f64, NavPvtObs)>::new();
    for f in frames {
        if let Some((_, roll, pitch, yaw)) = extract_esf_alg(f)
            && let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters)
        {
            alg_events.push(AlgEvent {
                t_ms,
                roll_deg: roll,
                pitch_deg: pitch,
                yaw_deg: normalize_heading_deg(yaw),
            });
        }
        if let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters) {
            if let Some(obs) = extract_nav2_pvt_obs(f) {
                if obs.fix_ok && !obs.invalid_llh {
                    nav_events_nav2.push((t_ms, obs));
                }
            } else if let Some(obs) = extract_nav_pvt_obs(f)
                && obs.fix_ok
                && !obs.invalid_llh
            {
                nav_events_pvt.push((t_ms, obs));
            }
        }
    }
    alg_events.sort_by(|a, b| {
        a.t_ms
            .partial_cmp(&b.t_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    nav_events_nav2.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    nav_events_pvt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let nav_events = if nav_events_nav2.is_empty() {
        nav_events_pvt
    } else {
        nav_events_nav2
    };

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
    let mut res_vn = Vec::<[f64; 2]>::new();
    let mut res_ve = Vec::<[f64; 2]>::new();
    let mut res_vd = Vec::<[f64; 2]>::new();
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

    let mut align = Align::default();
    align_init(
        &mut align,
        [0.1, 0.1, 0.1],
        MisalignNoise {
            q_theta_rw_var: 5.0e-5,
        },
    );

    let mut nav_idx = 0usize;
    let mut prev_imu_t: Option<f64> = None;
    for pkt in &imu_packets {
        let dt = match prev_imu_t {
            Some(prev) => (pkt.t_ms - prev) * 1e-3,
            None => {
                prev_imu_t = Some(pkt.t_ms);
                continue;
            }
        };
        prev_imu_t = Some(pkt.t_ms);
        if !(0.001..=0.05).contains(&dt) {
            continue;
        }

        let imu = MisalignImuSample {
            dt: dt as f32,
            f_sx: pkt.ax_mps2 as f32,
            f_sy: pkt.ay_mps2 as f32,
            f_sz: pkt.az_mps2 as f32,
        };
        align_predict_gyro(
            &mut align,
            &imu,
            (pkt.gx_dps.to_radians()) as f32,
            (pkt.gy_dps.to_radians()) as f32,
            (pkt.gz_dps.to_radians()) as f32,
        );

        while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
            let (tn, nav) = nav_events[nav_idx];
            nav_idx += 1;
            let r = ((nav.s_acc_mps * nav.s_acc_mps).max(0.02) * 20.0) as f32;
            align_fuse_velocity(
                &mut align,
                [
                    nav.vel_n_mps as f32,
                    nav.vel_e_mps as f32,
                    nav.vel_d_mps as f32,
                ],
                [r, r, r],
            );
            let t = rel_s(tn);
            res_vn.push([t, align.last_residual_n[0].to_degrees() as f64]);
            res_ve.push([t, align.last_residual_n[1].to_degrees() as f64]);
            res_vd.push([t, align.last_residual_n[2].to_degrees() as f64]);
        }

        let t = rel_s(pkt.t_ms);
        let q = align_q_sb(&align);
        // ESF-ALG angles are reported in z-up convention; Align runs in z-down convention.
        // Convert in rotation space before Euler extraction via Rx(180 deg) post-composition.
        let qx180 = [0.0, 1.0, 0.0, 0.0];
        let q_sb = [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64];
        let q_cmp = quat_mul(q_sb, qx180);
        // Convert Align q_sb to the same Euler convention used by ESF-ALG.
        let (r, p, y) = quat_rpy_alg_deg(q_cmp[0], q_cmp[1], q_cmp[2], q_cmp[3]);
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
                name: "res gyro_x [deg/s]".to_string(),
                points: res_vn,
            },
            Trace {
                name: "res gyro_y [deg/s]".to_string(),
                points: res_ve,
            },
            Trace {
                name: "res gyro_z [deg/s]".to_string(),
                points: res_vd,
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

// ESF-ALG convention: intrinsic ZYX (equivalently Rx * Ry * Rz composition in this codebase).
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

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}
