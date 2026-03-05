use vma_rs::vma::{
    MisalignAttitudeSample, MisalignImuSample, MisalignNoise, Vma, vma_fuse_velocity, vma_init,
    vma_predict, vma_q_sb,
};

use crate::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_raw_samples, extract_nav_att,
    extract_nav_pvt_obs, extract_nav2_pvt_obs, sensor_meta,
};

use super::super::math::{deg2rad, nearest_master_ms, normalize_heading_deg, quat_rpy_deg};
use super::super::model::{AlgEvent, ImuPacket, NavAttEvent, Trace};
use super::tag_time::fit_tag_ms_map;
use super::timebase::MasterTimeline;

pub struct VmaCompareData {
    pub cmp_att: Vec<Trace>,
    pub res_vel: Vec<Trace>,
    pub state_q: Vec<Trace>,
    pub cov: Vec<Trace>,
}

pub fn build_vma_compare_traces(frames: &[UbxFrame], tl: &MasterTimeline) -> VmaCompareData {
    if tl.masters.is_empty() {
        return VmaCompareData {
            cmp_att: Vec::new(),
            res_vel: Vec::new(),
            state_q: Vec::new(),
            cov: Vec::new(),
        };
    }
    let rel_s = |master_ms: f64| (master_ms - tl.t0_master_ms) * 1e-3;

    let mut alg_events = Vec::<AlgEvent>::new();
    let mut nav_att_events = Vec::<NavAttEvent>::new();
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
        if let Some((_itow, roll, pitch, heading)) = extract_nav_att(f)
            && let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters)
        {
            nav_att_events.push(NavAttEvent {
                t_ms,
                roll_deg: roll,
                pitch_deg: pitch,
                heading_deg: normalize_heading_deg(heading),
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
    nav_att_events.sort_by(|a, b| {
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
        // Apply the same effective remap used for VMA output:
        // R_eff = D * R_sb, D = diag(1,-1,-1).
        let q_sb = quat_from_rpy_zyx_deg(ev.roll_deg, ev.pitch_deg, ev.yaw_deg);
        let r_sb = quat_to_rotmat_f64(q_sb);
        let d = [[1.0_f64, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        let r_eff = mat3_mul_f64(d, r_sb);
        let q_eff = rotmat_to_quat_f64(r_eff);
        let (rr, pp, yy) = quat_rpy_deg(
            q_eff[0] as f32,
            q_eff[1] as f32,
            q_eff[2] as f32,
            q_eff[3] as f32,
        );
        ref_roll.push([t, rr]);
        ref_pitch.push([t, pp]);
        ref_yaw.push([t, yy]);
    }

    if nav_att_events.is_empty() {
        return VmaCompareData {
            cmp_att: vec![
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
            res_vel: Vec::new(),
            state_q: Vec::new(),
            cov: Vec::new(),
        };
    }

    let mut vma = Vma::default();
    vma_init(
        &mut vma,
        [0.2, 0.2, 0.2],
        MisalignNoise {
            q_theta_rw_var: 1.0e-4,
        },
    );

    let mut att_idx = 0usize;
    let mut nav_idx = 0usize;
    let mut cur_att = nav_att_events[0];
    let mut prev_imu_t: Option<f64> = None;
    for pkt in &imu_packets {
        while att_idx + 1 < nav_att_events.len() && nav_att_events[att_idx + 1].t_ms <= pkt.t_ms {
            att_idx += 1;
            cur_att = nav_att_events[att_idx];
        }

        let q_nb = quat_from_rpy_zyx_deg(cur_att.roll_deg, cur_att.pitch_deg, cur_att.heading_deg);
        let att = MisalignAttitudeSample {
            q_nb0: q_nb[0] as f32,
            q_nb1: q_nb[1] as f32,
            q_nb2: q_nb[2] as f32,
            q_nb3: q_nb[3] as f32,
        };
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
        vma_predict(&mut vma, &imu, &att);

        while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
            let (tn, nav) = nav_events[nav_idx];
            nav_idx += 1;
            let r = ((nav.s_acc_mps * nav.s_acc_mps).max(0.05) * 100.0) as f32;
            vma_fuse_velocity(
                &mut vma,
                [
                    nav.vel_n_mps as f32,
                    nav.vel_e_mps as f32,
                    nav.vel_d_mps as f32,
                ],
                [r, r, r * 5.0],
            );
            let t = rel_s(tn);
            res_vn.push([t, vma.last_residual_n[0] as f64]);
            res_ve.push([t, vma.last_residual_n[1] as f64]);
            res_vd.push([t, vma.last_residual_n[2] as f64]);
        }

        let t = rel_s(pkt.t_ms);
        let q = vma_q_sb(&vma);
        // Match full EKF preprocessing effective mapping:
        // R_eff = D * R_sb, where D = diag(1, -1, -1).
        let r_sb = quat_to_rotmat_f64([q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64]);
        let d = [[1.0_f64, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        let r_eff = mat3_mul_f64(d, r_sb);
        let q_eff = rotmat_to_quat_f64(r_eff);
        let (r, p, y) = quat_rpy_deg(
            q_eff[0] as f32,
            q_eff[1] as f32,
            q_eff[2] as f32,
            q_eff[3] as f32,
        );
        out_roll.push([t, r]);
        out_pitch.push([t, p]);
        out_yaw.push([t, y]);
        q0_tr.push([t, q[0] as f64]);
        q1_tr.push([t, q[1] as f64]);
        q2_tr.push([t, q[2] as f64]);
        q3_tr.push([t, q[3] as f64]);
        p00.push([t, vma.P[0][0] as f64]);
        p11.push([t, vma.P[1][1] as f64]);
        p22.push([t, vma.P[2][2] as f64]);
    }

    VmaCompareData {
        cmp_att: vec![
            Trace {
                name: "VMA roll [deg]".to_string(),
                points: out_roll,
            },
            Trace {
                name: "VMA pitch [deg]".to_string(),
                points: out_pitch,
            },
            Trace {
                name: "VMA yaw [deg]".to_string(),
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
                name: "res velN [m/s]".to_string(),
                points: res_vn,
            },
            Trace {
                name: "res velE [m/s]".to_string(),
                points: res_ve,
            },
            Trace {
                name: "res velD [m/s]".to_string(),
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

fn quat_from_rpy_zyx_deg(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> [f64; 4] {
    let (r, p, y) = (deg2rad(roll_deg), deg2rad(pitch_deg), deg2rad(yaw_deg));
    let (sr, cr) = (0.5 * r).sin_cos();
    let (sp, cp) = (0.5 * p).sin_cos();
    let (sy, cy) = (0.5 * y).sin_cos();
    [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]
}

fn quat_to_rotmat_f64(q: [f64; 4]) -> [[f64; 3]; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let qn = if n > 1.0e-12 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    };
    let (w, x, y, z) = (qn[0], qn[1], qn[2], qn[3]);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn mat3_mul_f64(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn rotmat_to_quat_f64(r: [[f64; 3]; 3]) -> [f64; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let (w, x, y, z);
    if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        w = 0.25 * s;
        x = (r[2][1] - r[1][2]) / s;
        y = (r[0][2] - r[2][0]) / s;
        z = (r[1][0] - r[0][1]) / s;
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        w = (r[2][1] - r[1][2]) / s;
        x = 0.25 * s;
        y = (r[0][1] + r[1][0]) / s;
        z = (r[0][2] + r[2][0]) / s;
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        w = (r[0][2] - r[2][0]) / s;
        x = (r[0][1] + r[1][0]) / s;
        y = 0.25 * s;
        z = (r[1][2] + r[2][1]) / s;
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        w = (r[1][0] - r[0][1]) / s;
        x = (r[0][2] + r[2][0]) / s;
        y = (r[1][2] + r[2][1]) / s;
        z = 0.25 * s;
    }
    let n = (w * w + x * x + y * y + z * z).sqrt();
    if n > 1.0e-12 {
        [w / n, x / n, y / n, z / n]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}
