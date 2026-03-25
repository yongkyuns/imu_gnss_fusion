use std::collections::HashMap;

use crate::ubxlog::{
    UbxFrame, extract_esf_alg, extract_esf_alg_status, extract_esf_cal_samples, extract_esf_ins,
    extract_esf_meas_samples, extract_esf_raw_samples, extract_nav_att, extract_nav_pvt,
    extract_nav_sat_cn0, sensor_meta,
};

use super::super::math::normalize_heading_deg;
use super::super::model::{PlotData, Trace};
use super::align_compare::AlignCompareData;
use super::align_nhc_compare::AlignNhcCompareData;
use super::ekf_compare::EkfCompareData;
use super::misalign_compare::MisalignCompareData;
use super::tag_time::fit_tag_ms_map;
use super::timebase::MasterTimeline;

pub fn build_signal_traces(
    frames: &[UbxFrame],
    tl: &MasterTimeline,
    ekf: EkfCompareData,
    misalign_data: MisalignCompareData,
    align_data: AlignCompareData,
    align_nhc_data: AlignNhcCompareData,
) -> PlotData {
    let mut speed_g = Vec::<[f64; 2]>::new();
    let mut speed_n = Vec::<[f64; 2]>::new();
    let mut speed_e = Vec::<[f64; 2]>::new();
    let mut speed_d = Vec::<[f64; 2]>::new();
    let mut sats: HashMap<String, Vec<[f64; 2]>> = HashMap::new();
    let mut orient_map: HashMap<String, Vec<[f64; 2]>> = HashMap::new();
    let mut other_map: HashMap<String, Vec<[f64; 2]>> = HashMap::new();
    let mut esf_ins_gyro_map: HashMap<String, Vec<[f64; 2]>> = HashMap::new();
    let mut esf_ins_accel_map: HashMap<String, Vec<[f64; 2]>> = HashMap::new();

    for f in frames {
        if let Some((_itow, gs, vn, ve, vd, _lat, _lon)) = extract_nav_pvt(f)
            && let Some(t) = tl.seq_to_rel_s(f.seq)
        {
            speed_g.push([t, gs]);
            speed_n.push([t, vn]);
            speed_e.push([t, ve]);
            speed_d.push([t, vd]);
        }
        if let Some((_itow, roll, pitch, yaw)) = extract_nav_att(f)
            && let Some(t) = tl.seq_to_rel_s(f.seq)
        {
            other_map
                .entry("NAV-ATT roll [deg]".to_string())
                .or_default()
                .push([t, roll]);
            other_map
                .entry("NAV-ATT pitch [deg]".to_string())
                .or_default()
                .push([t, pitch]);
            other_map
                .entry("NAV-ATT heading [deg]".to_string())
                .or_default()
                .push([t, normalize_heading_deg(yaw)]);
        }
        if let Some((_itow, roll, pitch, yaw)) = extract_esf_alg(f)
            && let Some(t) = tl.seq_to_rel_s(f.seq)
        {
            orient_map
                .entry("ESF-ALG roll [deg]".to_string())
                .or_default()
                .push([t, roll]);
            orient_map
                .entry("ESF-ALG pitch [deg]".to_string())
                .or_default()
                .push([t, pitch]);
            orient_map
                .entry("ESF-ALG yaw [deg]".to_string())
                .or_default()
                .push([t, normalize_heading_deg(yaw)]);
        }
        if let Some((_itow, status_code, is_fine)) = extract_esf_alg_status(f)
            && let Some(t) = tl.seq_to_rel_s(f.seq)
        {
            orient_map
                .entry("ESF-ALG status_code".to_string())
                .or_default()
                .push([t, status_code]);
            orient_map
                .entry("ESF-ALG fine_aligned".to_string())
                .or_default()
                .push([t, is_fine]);
        }
        for (sat, cno) in extract_nav_sat_cn0(f) {
            if let Some(t) = tl.seq_to_rel_s(f.seq) {
                sats.entry(sat).or_default().push([t, cno]);
            }
        }
        if let Some((_itow, wx, wy, wz, ax, ay, az)) = extract_esf_ins(f)
            && let Some(t) = tl.seq_to_rel_s(f.seq)
        {
            esf_ins_gyro_map
                .entry("ESF-INS wx [deg/s]".to_string())
                .or_default()
                .push([t, wx]);
            esf_ins_gyro_map
                .entry("ESF-INS wy [deg/s]".to_string())
                .or_default()
                .push([t, wy]);
            esf_ins_gyro_map
                .entry("ESF-INS wz [deg/s]".to_string())
                .or_default()
                .push([t, wz]);
            esf_ins_accel_map
                .entry("ESF-INS ax [m/s^2]".to_string())
                .or_default()
                .push([t, ax]);
            esf_ins_accel_map
                .entry("ESF-INS ay [m/s^2]".to_string())
                .or_default()
                .push([t, ay]);
            esf_ins_accel_map
                .entry("ESF-INS az [m/s^2]".to_string())
                .or_default()
                .push([t, az]);
        }
    }

    let mut raw_tag = Vec::<u64>::new();
    let mut raw_seq = Vec::<u64>::new();
    let mut raw_sig = Vec::<(u8, f64)>::new();
    for f in frames {
        for (tag, sw) in extract_esf_raw_samples(f) {
            raw_tag.push(tag);
            raw_seq.push(f.seq);
            let (_name, _unit, scale) = sensor_meta(sw.dtype);
            raw_sig.push((sw.dtype, sw.value_i24 as f64 * scale));
        }
    }
    let (raw_unwrapped, a_raw, b_raw) =
        fit_tag_ms_map(&raw_seq, &raw_tag, &tl.masters, Some(1 << 16));

    let mut cal_tag = Vec::<u64>::new();
    let mut cal_seq = Vec::<u64>::new();
    let mut cal_sig = Vec::<(u8, f64, &'static str)>::new();
    for f in frames {
        for (tag, sw) in extract_esf_cal_samples(f) {
            cal_tag.push(tag);
            cal_seq.push(f.seq);
            let (_name, _unit, scale) = sensor_meta(sw.dtype);
            cal_sig.push((sw.dtype, sw.value_i24 as f64 * scale, "ESF-CAL"));
        }
    }
    let (cal_u, a_cal, b_cal) = fit_tag_ms_map(&cal_seq, &cal_tag, &tl.masters, Some(1 << 16));

    let mut meas_tag = Vec::<u64>::new();
    let mut meas_seq = Vec::<u64>::new();
    let mut meas_sig = Vec::<(u8, f64, &'static str)>::new();
    for f in frames {
        for (tag, sw) in extract_esf_meas_samples(f) {
            meas_tag.push(tag);
            meas_seq.push(f.seq);
            let (_name, _unit, scale) = sensor_meta(sw.dtype);
            meas_sig.push((sw.dtype, sw.value_i24 as f64 * scale, "ESF-MEAS"));
        }
    }
    let (_, a_meas, b_meas) = fit_tag_ms_map(&meas_seq, &meas_tag, &tl.masters, None);

    let mut raw_by_sig: HashMap<String, Vec<[f64; 2]>> = HashMap::new();
    for (((dtype, val), tag), seq) in raw_sig.iter().zip(raw_unwrapped.iter()).zip(raw_seq.iter()) {
        let (name, _unit, _scale) = sensor_meta(*dtype);
        let master_ms = match tl.map_tag_ms(a_raw, b_raw, *tag as f64, *seq) {
            Some(v) => v,
            None => continue,
        };
        if let Some(t) = tl.master_ms_to_rel_s(master_ms) {
            raw_by_sig
                .entry(format!("ESF-RAW {}", name))
                .or_default()
                .push([t, *val]);
        }
    }

    let mut cal_by_sig: HashMap<String, Vec<[f64; 2]>> = HashMap::new();
    for (((dtype, val, src), tag), seq) in cal_sig.iter().zip(cal_u.iter()).zip(cal_seq.iter()) {
        let (name, _unit, _scale) = sensor_meta(*dtype);
        let master_ms = match tl.map_tag_ms(a_cal, b_cal, *tag as f64, *seq) {
            Some(v) => v,
            None => continue,
        };
        if let Some(t) = tl.master_ms_to_rel_s(master_ms) {
            cal_by_sig
                .entry(format!("{} {}", src, name))
                .or_default()
                .push([t, *val]);
        }
    }
    for (((dtype, val, src), tag), seq) in meas_sig.iter().zip(meas_tag.iter()).zip(meas_seq.iter())
    {
        let (name, _unit, _scale) = sensor_meta(*dtype);
        let master_ms = match tl.map_tag_ms(a_meas, b_meas, *tag as f64, *seq) {
            Some(v) => v,
            None => continue,
        };
        if let Some(t) = tl.master_ms_to_rel_s(master_ms) {
            cal_by_sig
                .entry(format!("{} {}", src, name))
                .or_default()
                .push([t, *val]);
        }
    }

    let mut out = PlotData::default();
    out.speed = vec![
        Trace {
            name: "gSpeed [m/s]".to_string(),
            points: speed_g,
        },
        Trace {
            name: "velN [m/s]".to_string(),
            points: speed_n,
        },
        Trace {
            name: "velE [m/s]".to_string(),
            points: speed_e,
        },
        Trace {
            name: "velD [m/s]".to_string(),
            points: speed_d,
        },
    ];
    out.sat_cn0 = sats
        .into_iter()
        .map(|(k, v)| Trace { name: k, points: v })
        .collect();

    for (k, v) in raw_by_sig {
        if k.contains("gyro_") {
            out.imu_raw_gyro.push(Trace { name: k, points: v });
        } else if k.contains("accel_") {
            out.imu_raw_accel.push(Trace { name: k, points: v });
        } else {
            out.other.push(Trace { name: k, points: v });
        }
    }

    for (k, v) in cal_by_sig {
        if k.contains("gyro_") {
            out.imu_cal_gyro.push(Trace { name: k, points: v });
        } else if k.contains("accel_") {
            out.imu_cal_accel.push(Trace { name: k, points: v });
        } else {
            out.other.push(Trace { name: k, points: v });
        }
    }

    for (k, v) in other_map {
        out.other.push(Trace { name: k, points: v });
    }

    for (name, points) in orient_map {
        out.orientation.push(Trace { name, points });
    }
    out.orientation.sort_by(|a, b| {
        a.name
            .partial_cmp(&b.name)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (name, points) in esf_ins_accel_map {
        out.esf_ins_accel.push(Trace { name, points });
    }
    for (name, points) in esf_ins_gyro_map {
        out.esf_ins_gyro.push(Trace { name, points });
    }
    out.esf_ins_gyro.sort_by(|a, b| {
        a.name
            .partial_cmp(&b.name)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.esf_ins_accel.sort_by(|a, b| {
        a.name
            .partial_cmp(&b.name)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.ekf_cmp_pos = ekf.cmp_pos;
    out.ekf_cmp_vel = ekf.cmp_vel;
    out.ekf_cmp_att = ekf.cmp_att;
    out.ekf_bias_gyro = ekf.bias_gyro;
    out.ekf_bias_accel = ekf.bias_accel;
    out.ekf_cov_bias = ekf.cov_bias;
    out.ekf_cov_nonbias = ekf.cov_nonbias;
    out.ekf_map = ekf.map;
    out.ekf_map_heading = ekf.map_heading;
    out.misalign_cmp_att = misalign_data.cmp_att;
    out.misalign_diag = misalign_data.diag;
    out.misalign_axis_err = misalign_data.axis_err;
    out.misalign_residuals = misalign_data.residuals;
    out.misalign_gates = misalign_data.gates;
    out.misalign_cov = misalign_data.cov;
    out.align_cmp_att = align_data.cmp_att;
    out.align_res_vel = align_data.res_vel;
    out.align_axis_err = align_data.axis_err;
    out.align_motion = align_data.motion;
    out.align_startup = align_data.startup;
    out.align_startup_angles = align_data.startup_angles;
    out.align_pca_vectors = align_data.pca_vectors;
    out.align_nhc_cmp_att = align_nhc_data.cmp_att;
    out.align_nhc_diag = align_nhc_data.diag;
    out.align_nhc_axis_err = align_nhc_data.axis_err;
    out.align_nhc_residuals = align_nhc_data.residuals;
    out.align_nhc_gates = align_nhc_data.gates;
    out.align_nhc_cov = align_nhc_data.cov;
    out.align_roll_contrib = align_data.roll_contrib;
    out.align_pitch_contrib = align_data.pitch_contrib;
    out.align_yaw_contrib = align_data.yaw_contrib;
    out.align_cov = align_data.cov;

    let max_rel_s = ((tl.master_max - tl.t0_master_ms) * 1e-3).max(0.0);
    let sanitize_trace = |trace: &mut Trace| {
        let mut cleaned = Vec::with_capacity(trace.points.len());
        let mut last_t = -1e-9_f64;
        for p in trace.points.iter().copied() {
            let t = p[0];
            let y = p[1];
            if !t.is_finite() || !y.is_finite() {
                continue;
            }
            if t < -1e-6 || t > max_rel_s + 1.0 {
                continue;
            }
            if t + 1e-9 < last_t {
                continue;
            }
            cleaned.push(p);
            last_t = t;
        }
        trace.points = cleaned;
    };

    for traces in [
        &mut out.speed,
        &mut out.sat_cn0,
        &mut out.imu_raw_gyro,
        &mut out.imu_raw_accel,
        &mut out.imu_cal_gyro,
        &mut out.imu_cal_accel,
        &mut out.esf_ins_gyro,
        &mut out.esf_ins_accel,
        &mut out.orientation,
        &mut out.other,
        &mut out.ekf_cmp_pos,
        &mut out.ekf_cmp_vel,
        &mut out.ekf_cmp_att,
        &mut out.ekf_bias_gyro,
        &mut out.ekf_bias_accel,
        &mut out.ekf_cov_bias,
        &mut out.ekf_cov_nonbias,
        &mut out.misalign_cmp_att,
        &mut out.misalign_diag,
        &mut out.misalign_axis_err,
        &mut out.misalign_residuals,
        &mut out.misalign_gates,
        &mut out.misalign_cov,
        &mut out.align_cmp_att,
        &mut out.align_res_vel,
        &mut out.align_axis_err,
        &mut out.align_motion,
        &mut out.align_startup,
        &mut out.align_startup_angles,
        &mut out.align_nhc_cmp_att,
        &mut out.align_nhc_diag,
        &mut out.align_nhc_axis_err,
        &mut out.align_nhc_residuals,
        &mut out.align_nhc_gates,
        &mut out.align_nhc_cov,
        &mut out.align_roll_contrib,
        &mut out.align_pitch_contrib,
        &mut out.align_yaw_contrib,
        &mut out.align_cov,
    ] {
        for tr in traces.iter_mut() {
            tr.points
                .sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
            sanitize_trace(tr);
        }
    }
    for tr in &mut out.ekf_map {
        tr.points.retain(|p| p[0].is_finite() && p[1].is_finite());
    }
    for tr in &mut out.align_pca_vectors {
        tr.points.retain(|p| p[0].is_finite() && p[1].is_finite());
    }

    out
}
