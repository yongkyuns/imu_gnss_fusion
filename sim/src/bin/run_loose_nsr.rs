use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::c_api::{CLooseImuDelta, CLooseWrapper};
use sensor_fusion::loose::LoosePredictNoise;
use serde::{Deserialize, Serialize};
use sim::visualizer::math::{
    ecef_to_lla, ecef_to_ned, lla_to_ecef, ned_to_lla_exact, quat_rpy_deg,
};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    input_dir: PathBuf,
    #[arg(long)]
    init_json: PathBuf,
    #[arg(long)]
    out_json: PathBuf,
    #[arg(long)]
    mount_align_rw_var: Option<f32>,
    #[arg(long, default_value_t = false)]
    disable_nhc: bool,
    #[arg(long)]
    diag_json: Option<PathBuf>,
    #[arg(long, default_value_t = 60.0)]
    diag_until_s: f64,
    #[arg(long)]
    trace_time_s: Option<f64>,
    #[arg(long)]
    trace_json: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct GyroSample {
    ttag_us: i64,
    omega_radps: [f64; 3],
}

#[derive(Debug, Clone)]
struct AccelSample {
    ttag_us: i64,
    accel_mps2: [f64; 3],
}

#[derive(Debug, Clone)]
struct GnssSample {
    ttag_us: i64,
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    h_acc_m: f64,
}

#[derive(Debug, Clone, Copy)]
enum EventType {
    Accel,
    Gyro,
    Gnss,
}

#[derive(Debug, Clone, Copy)]
struct Event {
    ttag_us: i64,
    event_type: EventType,
    index: usize,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct RefInit {
    start_ttag_us: i64,
    ref_lat_deg: f64,
    ref_lon_deg: f64,
    ref_h_m: f64,
    q_bn: [f32; 4],
    pos_ecef_m: [f64; 3],
    vel_ecef_mps: [f32; 3],
    pos_ned_m: [f32; 3],
    vel_ned_mps: [f32; 3],
    gyro_bias_radps: [f32; 3],
    accel_bias_mps2: [f32; 3],
    gyro_scale: [f32; 3],
    accel_scale: [f32; 3],
    q_cs: [f32; 4],
    p_diag: [f32; 24],
    p_full: [[f32; 24]; 24],
}

#[derive(Debug, Serialize)]
struct Output {
    time_s: Vec<f64>,
    lat_deg: Vec<f64>,
    lon_deg: Vec<f64>,
    height_m: Vec<f64>,
    pos_ned_m: Vec<[f64; 3]>,
    vel_ned_mps: Vec<[f64; 3]>,
    euler_bn_deg: Vec<[f64; 3]>,
    euler_mis_deg: Vec<[f64; 3]>,
    pos_ecef_m: Vec<[f64; 3]>,
    vel_ecef_mps: Vec<[f64; 3]>,
    q_es: Vec<[f64; 4]>,
    q_cs: Vec<[f64; 4]>,
}

#[derive(Debug, Serialize)]
struct DiagRow {
    time_s: f64,
    ttag_us: i64,
    gps_active: bool,
    nhc_active: bool,
    applied_obs_types: Vec<i32>,
    pos_ecef_pre: [f64; 3],
    gyro_bias_pre: [f64; 3],
    accel_bias_pre: [f64; 3],
    gyro_scale_pre: [f64; 3],
    accel_scale_pre: [f64; 3],
    q_es_pre: [f64; 4],
    vel_ecef_pre: [f64; 3],
    omega_is_pre: [f64; 3],
    omega_norm_pre: f64,
    f_s_pre: [f64; 3],
    f_s_norm_pre: f64,
    v_c_pre: [f64; 3],
    nhc_h_y_pre: [f64; 24],
    nhc_h_z_pre: [f64; 24],
    p_pre_full: [[f64; 24]; 24],
    p_pos_psi_cc_pre: [[f64; 3]; 3],
    p_vel_psi_cc_pre: [[f64; 3]; 3],
    p_psiee_psi_cc_pre: [[f64; 3]; 3],
    p_psi_cc_pre: [[f64; 3]; 3],
    gps_h_rows_pre: [[f64; 24]; 3],
    gps_residual_pre: [f64; 3],
    gps_variance_pre: [f64; 3],
    q_cs_pre: [f64; 4],
    p_post_full: [[f64; 24]; 24],
    p_pos_psi_cc_post: [[f64; 3]; 3],
    p_vel_psi_cc_post: [[f64; 3]; 3],
    p_psiee_psi_cc_post: [[f64; 3]; 3],
    p_psi_cc_post: [[f64; 3]; 3],
    pos_ecef_post: [f64; 3],
    q_es_post: [f64; 4],
    q_cs_post: [f64; 4],
}

fn main() -> Result<()> {
    let args = Args::parse();
    let init: RefInit = serde_json::from_slice(&fs::read(&args.init_json)?)?;

    let accel = import_accel_data(&resolve_single_file(&args.input_dir, "_Acc.csv")?)?;
    let gyro = import_gyro_data(&resolve_single_file(&args.input_dir, "_Gyro.csv")?)?;
    let gnss = import_gnss_data(&resolve_single_file(&args.input_dir, "_GNSS.csv")?)?;

    let mut events = Vec::with_capacity(accel.len() + gyro.len() + gnss.len());
    events.extend(gyro.iter().enumerate().map(|(index, s)| Event {
        ttag_us: s.ttag_us,
        event_type: EventType::Gyro,
        index,
    }));
    events.extend(accel.iter().enumerate().map(|(index, s)| Event {
        ttag_us: s.ttag_us,
        event_type: EventType::Accel,
        index,
    }));
    events.extend(gnss.iter().enumerate().map(|(index, s)| Event {
        ttag_us: s.ttag_us,
        event_type: EventType::Gnss,
        index,
    }));
    events.sort_by(|a, b| {
        a.ttag_us
            .cmp(&b.ttag_us)
            .then_with(|| event_rank(a.event_type).cmp(&event_rank(b.event_type)))
            .then_with(|| a.index.cmp(&b.index))
    });

    let mut noise = LoosePredictNoise::reference_nsr_demo();
    if let Some(value) = args.mount_align_rw_var {
        noise.mount_align_rw_var = value;
    }
    let mut loose = CLooseWrapper::new(noise);
    loose.init_from_reference_ecef_state(
        init.q_bn,
        init.pos_ecef_m,
        init.vel_ecef_mps,
        init.gyro_bias_radps,
        init.accel_bias_mps2,
        init.gyro_scale,
        init.accel_scale,
        init.q_cs,
        Some(init.p_diag),
    );
    loose.set_covariance(init.p_full);

    let qcs0 = init.q_cs;
    let q_ne = quat_ecef_to_ned(init.ref_lat_deg, init.ref_lon_deg);
    let ref_ecef = lla_to_ecef(init.ref_lat_deg, init.ref_lon_deg, init.ref_h_m);
    let mut started = false;
    let mut last_gnss_used_ttag = i64::MIN;
    let mut latest_gnss_index: Option<usize> = None;
    let mut accel_seen_count = 0usize;

    let mut time_s = Vec::new();
    let mut lat_deg = Vec::new();
    let mut lon_deg = Vec::new();
    let mut height_m = Vec::new();
    let mut pos_ned_m = Vec::new();
    let mut vel_ned_mps = Vec::new();
    let mut euler_bn_deg = Vec::new();
    let mut euler_mis_deg = Vec::new();
    let mut pos_ecef_m = Vec::new();
    let mut vel_ecef_mps = Vec::new();
    let mut q_es_out = Vec::new();
    let mut q_cs_out = Vec::new();
    let mut diag_rows = Vec::new();
    let mut trace_row: Option<DiagRow> = None;

    {
        let n = loose.nominal();
        let pos_ecef = [n.pn as f64, n.pe as f64, n.pd as f64];
        let vel_ecef = [n.vn as f64, n.ve as f64, n.vd as f64];
        let pos = ecef_to_ned(pos_ecef, ref_ecef, init.ref_lat_deg, init.ref_lon_deg);
        let c_ne = [
            [
                -init.ref_lat_deg.to_radians().sin() * init.ref_lon_deg.to_radians().cos(),
                -init.ref_lat_deg.to_radians().sin() * init.ref_lon_deg.to_radians().sin(),
                init.ref_lat_deg.to_radians().cos(),
            ],
            [
                -init.ref_lon_deg.to_radians().sin(),
                init.ref_lon_deg.to_radians().cos(),
                0.0,
            ],
            [
                -init.ref_lat_deg.to_radians().cos() * init.ref_lon_deg.to_radians().cos(),
                -init.ref_lat_deg.to_radians().cos() * init.ref_lon_deg.to_radians().sin(),
                -init.ref_lat_deg.to_radians().sin(),
            ],
        ];
        let vel = [
            c_ne[0][0] * vel_ecef[0] + c_ne[0][1] * vel_ecef[1] + c_ne[0][2] * vel_ecef[2],
            c_ne[1][0] * vel_ecef[0] + c_ne[1][1] * vel_ecef[1] + c_ne[1][2] * vel_ecef[2],
            c_ne[2][0] * vel_ecef[0] + c_ne[2][1] * vel_ecef[1] + c_ne[2][2] * vel_ecef[2],
        ];
        let (lat, lon, h) = ned_to_lla_exact(
            pos[0],
            pos[1],
            pos[2],
            init.ref_lat_deg,
            init.ref_lon_deg,
            init.ref_h_m,
        );
        let q_ns = quat_mul(q_ne, [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64]);
        let (roll, pitch, yaw) = quat_rpy_deg(
            q_ns[0] as f32,
            q_ns[1] as f32,
            q_ns[2] as f32,
            q_ns[3] as f32,
        );
        let mis = quat_mul(
            [n.qcs0 as f64, n.qcs1 as f64, n.qcs2 as f64, n.qcs3 as f64],
            quat_conj([
                qcs0[0] as f64,
                qcs0[1] as f64,
                qcs0[2] as f64,
                qcs0[3] as f64,
            ]),
        );
        let (mroll, mpitch, myaw) =
            quat_rpy_deg(mis[0] as f32, mis[1] as f32, mis[2] as f32, mis[3] as f32);

        time_s.push(0.0);
        lat_deg.push(lat);
        lon_deg.push(lon);
        height_m.push(h);
        pos_ned_m.push(pos);
        vel_ned_mps.push(vel);
        euler_bn_deg.push([roll, pitch, yaw]);
        euler_mis_deg.push([mroll, mpitch, myaw]);
        pos_ecef_m.push(pos_ecef);
        vel_ecef_mps.push(vel_ecef);
        q_es_out.push([n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64]);
        q_cs_out.push([n.qcs0 as f64, n.qcs1 as f64, n.qcs2 as f64, n.qcs3 as f64]);
    }

    for event in events {
        match event.event_type {
            EventType::Accel => {
                accel_seen_count = accel_seen_count.max(event.index + 1);
            }
            EventType::Gnss => {
                latest_gnss_index = Some(event.index);
            }
            EventType::Gyro => {
                let curr = &gyro[event.index];
                if !started {
                    if curr.ttag_us < init.start_ttag_us || event.index == 0 {
                        continue;
                    }
                    started = true;
                }
                if event.index == 0 {
                    continue;
                }
                let prev = &gyro[event.index - 1];
                if prev.ttag_us < init.start_ttag_us {
                    continue;
                }

                let accel_seen = &accel[..accel_seen_count];
                let a1 = accel_at(prev.ttag_us, accel_seen)
                    .with_context(|| format!("missing accel at {}", prev.ttag_us))?;
                let a2 = accel_at(curr.ttag_us, accel_seen)
                    .with_context(|| format!("missing accel at {}", curr.ttag_us))?;
                let dt = (curr.ttag_us - prev.ttag_us) as f64 * 1.0e-6;
                if !(dt > 0.0) {
                    continue;
                }
                let imu = CLooseImuDelta {
                    dax_1: (prev.omega_radps[0] * dt) as f32,
                    day_1: (prev.omega_radps[1] * dt) as f32,
                    daz_1: (prev.omega_radps[2] * dt) as f32,
                    dvx_1: (a1[0] * dt) as f32,
                    dvy_1: (a1[1] * dt) as f32,
                    dvz_1: (a1[2] * dt) as f32,
                    dax_2: (curr.omega_radps[0] * dt) as f32,
                    day_2: (curr.omega_radps[1] * dt) as f32,
                    daz_2: (curr.omega_radps[2] * dt) as f32,
                    dvx_2: (a2[0] * dt) as f32,
                    dvy_2: (a2[1] * dt) as f32,
                    dvz_2: (a2[2] * dt) as f32,
                    dt: dt as f32,
                };
                loose.predict(imu);

                let mut gps_pos_ecef = None;
                let mut gps_h_acc_m = 0.0f32;
                let mut dt_since_last_gnss = 1.0f32;
                if let Some(gnss_index) = latest_gnss_index {
                    let g = &gnss[gnss_index];
                    let age_us = curr.ttag_us - g.ttag_us;
                    if age_us >= 0 && age_us < 50_000 && g.ttag_us != last_gnss_used_ttag {
                        let pos_ecef = lla_to_ecef(g.lat_deg, g.lon_deg, g.height_m);
                        gps_pos_ecef = Some(pos_ecef);
                        gps_h_acc_m = g.h_acc_m as f32;
                        dt_since_last_gnss = if last_gnss_used_ttag == i64::MIN {
                            1.0
                        } else {
                            ((curr.ttag_us - last_gnss_used_ttag) as f32 * 1.0e-6)
                                .clamp(1.0e-3, 1.0)
                        };
                        last_gnss_used_ttag = g.ttag_us;
                    }
                }

                let latest_accel = accel_seen
                    .last()
                    .map(|s| s.accel_mps2)
                    .with_context(|| format!("missing latest accel at {}", curr.ttag_us))?;

                let nhc_active =
                    !args.disable_nhc && nhc_gate(loose.nominal(), curr, &latest_accel);
                let t = (curr.ttag_us - init.start_ttag_us) as f64 * 1.0e-6;
                let trace_match = args
                    .trace_time_s
                    .is_some_and(|target| (t - target).abs() <= 5.0e-7);
                if (args.diag_json.is_some()
                    && t <= args.diag_until_s
                    && (gps_pos_ecef.is_some() || nhc_active))
                    || trace_match
                {
                    let p = loose.covariance();
                    let n = loose.nominal();
                    let q_es_pre = [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64];
                    let q_cs_pre = [n.qcs0 as f64, n.qcs1 as f64, n.qcs2 as f64, n.qcs3 as f64];
                    let pos_ecef_pre = loose.shadow_pos_ecef();
                    let vel_ecef_pre = [n.vn as f64, n.ve as f64, n.vd as f64];
                    let omega_is_pre = [
                        (n.sgx * curr.omega_radps[0] as f32 + n.bgx) as f64,
                        (n.sgy * curr.omega_radps[1] as f32 + n.bgy) as f64,
                        (n.sgz * curr.omega_radps[2] as f32 + n.bgz) as f64,
                    ];
                    let f_s_pre = [
                        (n.sax * latest_accel[0] as f32 + n.bax) as f64,
                        (n.say * latest_accel[1] as f32 + n.bay) as f64,
                        (n.saz * latest_accel[2] as f32 + n.baz) as f64,
                    ];
                    let c_es = quat_to_dcm64(q_es_pre);
                    let c_cs = quat_to_dcm64(q_cs_pre);
                    let c_ce = mat3_mul(c_cs, mat3_transpose(c_es));
                    let v_c_pre = mat3_vec_mul(c_ce, vel_ecef_pre);
                    let h_rows = nhc_h_rows(c_ce, vel_ecef_pre, v_c_pre);
                    let omega_norm_pre = norm3(omega_is_pre);
                    let f_s_norm_pre = norm3(f_s_pre);
                    let (gps_h_rows_pre, gps_residual_pre, gps_variance_pre) = gps_diag_inputs(
                        pos_ecef_pre,
                        init.ref_lat_deg,
                        init.ref_lon_deg,
                        gps_pos_ecef,
                        gps_h_acc_m,
                        dt_since_last_gnss,
                    );
                    let mut p_pos_psi_cc_pre = [[0.0; 3]; 3];
                    let mut p_vel_psi_cc_pre = [[0.0; 3]; 3];
                    let mut p_psiee_psi_cc_pre = [[0.0; 3]; 3];
                    let mut p_psi_cc_pre = [[0.0; 3]; 3];
                    let mut p_pre_full = [[0.0; 24]; 24];
                    for i in 0..3 {
                        for j in 0..3 {
                            p_pos_psi_cc_pre[i][j] = p[i][21 + j] as f64;
                            p_vel_psi_cc_pre[i][j] = p[3 + i][21 + j] as f64;
                            p_psiee_psi_cc_pre[i][j] = p[6 + i][21 + j] as f64;
                            p_psi_cc_pre[i][j] = p[21 + i][21 + j] as f64;
                        }
                    }
                    for i in 0..24 {
                        for j in 0..24 {
                            p_pre_full[i][j] = p[i][j] as f64;
                        }
                    }
                    let mut row = DiagRow {
                        time_s: t,
                        ttag_us: curr.ttag_us,
                        gps_active: gps_pos_ecef.is_some(),
                        nhc_active,
                        applied_obs_types: Vec::new(),
                        pos_ecef_pre,
                        gyro_bias_pre: [n.bgx as f64, n.bgy as f64, n.bgz as f64],
                        accel_bias_pre: [n.bax as f64, n.bay as f64, n.baz as f64],
                        gyro_scale_pre: [n.sgx as f64, n.sgy as f64, n.sgz as f64],
                        accel_scale_pre: [n.sax as f64, n.say as f64, n.saz as f64],
                        q_es_pre,
                        vel_ecef_pre,
                        omega_is_pre,
                        omega_norm_pre,
                        f_s_pre,
                        f_s_norm_pre,
                        v_c_pre,
                        nhc_h_y_pre: h_rows[1],
                        nhc_h_z_pre: h_rows[2],
                        p_pre_full,
                        p_pos_psi_cc_pre,
                        p_vel_psi_cc_pre,
                        p_psiee_psi_cc_pre,
                        p_psi_cc_pre,
                        gps_h_rows_pre,
                        gps_residual_pre,
                        gps_variance_pre,
                        q_cs_pre,
                        p_post_full: [[0.0; 24]; 24],
                        p_pos_psi_cc_post: [[0.0; 3]; 3],
                        p_vel_psi_cc_post: [[0.0; 3]; 3],
                        p_psiee_psi_cc_post: [[0.0; 3]; 3],
                        p_psi_cc_post: [[0.0; 3]; 3],
                        pos_ecef_post: [0.0; 3],
                        q_es_post: [0.0; 4],
                        q_cs_post: [0.0; 4],
                    };
                    if nhc_active || gps_pos_ecef.is_some() {
                        loose.fuse_reference_batch(
                            gps_pos_ecef,
                            None,
                            gps_h_acc_m,
                            0.0,
                            dt_since_last_gnss,
                            [
                                curr.omega_radps[0] as f32,
                                curr.omega_radps[1] as f32,
                                curr.omega_radps[2] as f32,
                            ],
                            [
                                latest_accel[0] as f32,
                                latest_accel[1] as f32,
                                latest_accel[2] as f32,
                            ],
                            dt as f32,
                        );
                        let p = loose.covariance();
                        let n = loose.nominal();
                        row.applied_obs_types = loose.last_obs_types().to_vec();
                        for i in 0..3 {
                            for j in 0..3 {
                                row.p_pos_psi_cc_post[i][j] = p[i][21 + j] as f64;
                                row.p_vel_psi_cc_post[i][j] = p[3 + i][21 + j] as f64;
                                row.p_psiee_psi_cc_post[i][j] = p[6 + i][21 + j] as f64;
                                row.p_psi_cc_post[i][j] = p[21 + i][21 + j] as f64;
                            }
                        }
                        for i in 0..24 {
                            for j in 0..24 {
                                row.p_post_full[i][j] = p[i][j] as f64;
                            }
                        }
                        row.pos_ecef_post = loose.shadow_pos_ecef();
                        row.q_es_post = [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64];
                        row.q_cs_post =
                            [n.qcs0 as f64, n.qcs1 as f64, n.qcs2 as f64, n.qcs3 as f64];
                    } else {
                        row.p_post_full = row.p_pre_full;
                        row.p_pos_psi_cc_post = row.p_pos_psi_cc_pre;
                        row.p_vel_psi_cc_post = row.p_vel_psi_cc_pre;
                        row.p_psiee_psi_cc_post = row.p_psiee_psi_cc_pre;
                        row.p_psi_cc_post = row.p_psi_cc_pre;
                        row.pos_ecef_post = row.pos_ecef_pre;
                        row.q_es_post = row.q_es_pre;
                        row.q_cs_post = row.q_cs_pre;
                    }
                    if args.diag_json.is_some()
                        && t <= args.diag_until_s
                        && (gps_pos_ecef.is_some() || nhc_active)
                    {
                        diag_rows.push(DiagRow {
                            time_s: row.time_s,
                            ttag_us: row.ttag_us,
                            gps_active: row.gps_active,
                            nhc_active: row.nhc_active,
                            applied_obs_types: row.applied_obs_types.clone(),
                            pos_ecef_pre: row.pos_ecef_pre,
                            gyro_bias_pre: row.gyro_bias_pre,
                            accel_bias_pre: row.accel_bias_pre,
                            gyro_scale_pre: row.gyro_scale_pre,
                            accel_scale_pre: row.accel_scale_pre,
                            q_es_pre: row.q_es_pre,
                            vel_ecef_pre: row.vel_ecef_pre,
                            omega_is_pre: row.omega_is_pre,
                            omega_norm_pre: row.omega_norm_pre,
                            f_s_pre: row.f_s_pre,
                            f_s_norm_pre: row.f_s_norm_pre,
                            v_c_pre: row.v_c_pre,
                            nhc_h_y_pre: row.nhc_h_y_pre,
                            nhc_h_z_pre: row.nhc_h_z_pre,
                            p_pre_full: row.p_pre_full,
                            p_pos_psi_cc_pre: row.p_pos_psi_cc_pre,
                            p_vel_psi_cc_pre: row.p_vel_psi_cc_pre,
                            p_psiee_psi_cc_pre: row.p_psiee_psi_cc_pre,
                            p_psi_cc_pre: row.p_psi_cc_pre,
                            gps_h_rows_pre: row.gps_h_rows_pre,
                            gps_residual_pre: row.gps_residual_pre,
                            gps_variance_pre: row.gps_variance_pre,
                            q_cs_pre: row.q_cs_pre,
                            p_post_full: row.p_post_full,
                            p_pos_psi_cc_post: row.p_pos_psi_cc_post,
                            p_vel_psi_cc_post: row.p_vel_psi_cc_post,
                            p_psiee_psi_cc_post: row.p_psiee_psi_cc_post,
                            p_psi_cc_post: row.p_psi_cc_post,
                            pos_ecef_post: row.pos_ecef_post,
                            q_es_post: row.q_es_post,
                            q_cs_post: row.q_cs_post,
                        });
                    }
                    if trace_match {
                        trace_row = Some(row);
                    }
                } else if nhc_active || gps_pos_ecef.is_some() {
                    loose.fuse_reference_batch(
                        gps_pos_ecef,
                        None,
                        gps_h_acc_m,
                        0.0,
                        dt_since_last_gnss,
                        [
                            curr.omega_radps[0] as f32,
                            curr.omega_radps[1] as f32,
                            curr.omega_radps[2] as f32,
                        ],
                        [
                            latest_accel[0] as f32,
                            latest_accel[1] as f32,
                            latest_accel[2] as f32,
                        ],
                        dt as f32,
                    );
                }

                let n = loose.nominal();
                let pos_ecef = [n.pn as f64, n.pe as f64, n.pd as f64];
                let vel_ecef = [n.vn as f64, n.ve as f64, n.vd as f64];
                let pos = ecef_to_ned(pos_ecef, ref_ecef, init.ref_lat_deg, init.ref_lon_deg);
                let vel = ecef_to_ned(
                    [
                        ref_ecef[0] + vel_ecef[0],
                        ref_ecef[1] + vel_ecef[1],
                        ref_ecef[2] + vel_ecef[2],
                    ],
                    ref_ecef,
                    init.ref_lat_deg,
                    init.ref_lon_deg,
                );
                let (lat, lon, h) = ned_to_lla_exact(
                    pos[0],
                    pos[1],
                    pos[2],
                    init.ref_lat_deg,
                    init.ref_lon_deg,
                    init.ref_h_m,
                );
                let q_ns = quat_mul(q_ne, [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64]);
                let (roll, pitch, yaw) = quat_rpy_deg(
                    q_ns[0] as f32,
                    q_ns[1] as f32,
                    q_ns[2] as f32,
                    q_ns[3] as f32,
                );
                let mis = quat_mul(
                    [n.qcs0 as f64, n.qcs1 as f64, n.qcs2 as f64, n.qcs3 as f64],
                    quat_conj([
                        qcs0[0] as f64,
                        qcs0[1] as f64,
                        qcs0[2] as f64,
                        qcs0[3] as f64,
                    ]),
                );
                let (mroll, mpitch, myaw) =
                    quat_rpy_deg(mis[0] as f32, mis[1] as f32, mis[2] as f32, mis[3] as f32);

                time_s.push(t);
                lat_deg.push(lat);
                lon_deg.push(lon);
                height_m.push(h);
                pos_ned_m.push(pos);
                vel_ned_mps.push(vel);
                euler_bn_deg.push([roll, pitch, yaw]);
                euler_mis_deg.push([mroll, mpitch, myaw]);
                pos_ecef_m.push(pos_ecef);
                vel_ecef_mps.push(vel_ecef);
                q_es_out.push([n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64]);
                q_cs_out.push([n.qcs0 as f64, n.qcs1 as f64, n.qcs2 as f64, n.qcs3 as f64]);
            }
        }
    }

    if time_s.is_empty() {
        bail!("local loose replay produced no running samples");
    }

    let out = Output {
        time_s,
        lat_deg,
        lon_deg,
        height_m,
        pos_ned_m,
        vel_ned_mps,
        euler_bn_deg,
        euler_mis_deg,
        pos_ecef_m,
        vel_ecef_mps,
        q_es: q_es_out,
        q_cs: q_cs_out,
    };
    fs::write(&args.out_json, serde_json::to_vec(&out)?)?;
    if let Some(diag_json) = &args.diag_json {
        fs::write(diag_json, serde_json::to_vec(&diag_rows)?)?;
    }
    if let Some(trace_json) = &args.trace_json {
        let row = trace_row.with_context(|| {
            format!("failed to find trace row at time {:?} s", args.trace_time_s)
        })?;
        fs::write(trace_json, serde_json::to_vec(&row)?)?;
    }
    Ok(())
}

fn event_rank(event_type: EventType) -> u8 {
    match event_type {
        EventType::Gyro => 1,
        EventType::Accel => 2,
        EventType::Gnss => 4,
    }
}

fn resolve_single_file(input_dir: &Path, suffix: &str) -> Result<PathBuf> {
    let mut matches = Vec::new();
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(suffix))
        {
            matches.push(path);
        }
    }
    matches.sort();
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => bail!(
            "missing file with suffix {suffix} in {}",
            input_dir.display()
        ),
        _ => bail!(
            "multiple files with suffix {suffix} in {}",
            input_dir.display()
        ),
    }
}

fn import_gyro_data(path: &Path) -> Result<Vec<GyroSample>> {
    let rows = semicolon_rows(path, 3)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        out.push(GyroSample {
            ttag_us: (parse_f64(&row[0])? / 1000.0).floor() as i64,
            omega_radps: [
                parse_f64(&row[1])?,
                parse_f64(&row[2])?,
                parse_f64(&row[3])?,
            ],
        });
    }
    Ok(out)
}

fn import_accel_data(path: &Path) -> Result<Vec<AccelSample>> {
    let rows = semicolon_rows(path, 3)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        out.push(AccelSample {
            ttag_us: (parse_f64(&row[0])? / 1000.0).floor() as i64,
            accel_mps2: [
                parse_f64(&row[1])?,
                parse_f64(&row[2])?,
                parse_f64(&row[3])?,
            ],
        });
    }
    Ok(out)
}

fn import_gnss_data(path: &Path) -> Result<Vec<GnssSample>> {
    let rows = semicolon_rows(path, 1)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        out.push(GnssSample {
            ttag_us: (parse_f64(&row[0])? / 1000.0).floor() as i64,
            lat_deg: parse_f64(&row[2])?,
            lon_deg: parse_f64(&row[3])?,
            height_m: parse_f64(&row[4])?,
            h_acc_m: parse_f64(&row[7])?,
        });
    }
    Ok(out)
}

fn semicolon_rows(path: &Path, skip_rows: usize) -> Result<Vec<Vec<String>>> {
    let text = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index < skip_rows {
            continue;
        }
        let row: Vec<String> = line
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect();
        if !row.is_empty() {
            out.push(row);
        }
    }
    Ok(out)
}

fn parse_f64(s: &str) -> Result<f64> {
    s.parse::<f64>()
        .with_context(|| format!("failed to parse float: {s}"))
}

fn accel_at(ttag_us: i64, accel: &[AccelSample]) -> Option<[f64; 3]> {
    match accel.binary_search_by(|s| s.ttag_us.cmp(&ttag_us)) {
        Ok(index) => Some(accel[index].accel_mps2),
        Err(0) | Err(_) if accel.is_empty() => None,
        Err(index) if index >= accel.len() => {
            let s = &accel[accel.len() - 1];
            ((ttag_us - s.ttag_us).abs() <= 100_000).then_some(s.accel_mps2)
        }
        Err(index) => {
            let prev = &accel[index - 1];
            let next = &accel[index];
            if ttag_us - prev.ttag_us > 100_000 || next.ttag_us - ttag_us > 100_000 {
                return None;
            }
            let span = (next.ttag_us - prev.ttag_us) as f64;
            if span <= 0.0 {
                return Some(prev.accel_mps2);
            }
            let a = (ttag_us - prev.ttag_us) as f64 / span;
            Some([
                prev.accel_mps2[0] + a * (next.accel_mps2[0] - prev.accel_mps2[0]),
                prev.accel_mps2[1] + a * (next.accel_mps2[1] - prev.accel_mps2[1]),
                prev.accel_mps2[2] + a * (next.accel_mps2[2] - prev.accel_mps2[2]),
            ])
        }
    }
}

fn nhc_gate(
    n: &sensor_fusion::c_api::CLooseNominalState,
    gyro: &GyroSample,
    accel: &[f64; 3],
) -> bool {
    let omega = [
        n.sgx as f64 * gyro.omega_radps[0] + n.bgx as f64,
        n.sgy as f64 * gyro.omega_radps[1] + n.bgy as f64,
        n.sgz as f64 * gyro.omega_radps[2] + n.bgz as f64,
    ];
    let f = [
        n.sax as f64 * accel[0] + n.bax as f64,
        n.say as f64 * accel[1] + n.bay as f64,
        n.saz as f64 * accel[2] + n.baz as f64,
    ];
    let omega_norm = norm3(omega);
    let f_norm = norm3(f);
    omega_norm < 0.03 && (f_norm - 9.81).abs() < 0.2
}

fn quat_to_dcm64(q: [f64; 4]) -> [[f64; 3]; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let q = if n > 0.0 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    };
    let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);
    [
        [
            1.0 - 2.0 * (q2 * q2 + q3 * q3),
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * (q1 * q1 + q3 * q3),
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * (q1 * q1 + q2 * q2),
        ],
    ]
}

fn mat3_transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn mat3_vec_mul(a: [[f64; 3]; 3], x: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
        a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
        a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
    ]
}

fn skew(v: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
}

fn nhc_h_rows(c_ce: [[f64; 3]; 3], vel_e: [f64; 3], v_c: [f64; 3]) -> [[f64; 24]; 3] {
    let mut rows = [[0.0; 24]; 3];
    let sv = skew(vel_e);
    let svc = skew([-v_c[0], -v_c[1], -v_c[2]]);
    for r in 0..3 {
        for c in 0..3 {
            rows[r][3 + c] = c_ce[r][c];
            rows[r][6 + c] = (0..3).map(|k| c_ce[r][k] * sv[k][c]).sum();
            rows[r][21 + c] = svc[r][c];
        }
    }
    rows
}

fn gps_diag_inputs(
    pos_ecef_pre: [f64; 3],
    _ref_lat_deg: f64,
    _ref_lon_deg: f64,
    gps_pos_ecef: Option<[f64; 3]>,
    gps_h_acc_m: f32,
    dt_since_last_gnss: f32,
) -> ([[f64; 24]; 3], [f64; 3], [f64; 3]) {
    let mut h_rows = [[0.0; 24]; 3];
    let mut residual = [0.0; 3];
    let mut variance = [0.0; 3];
    let Some(gps_pos_ecef) = gps_pos_ecef else {
        return (h_rows, residual, variance);
    };
    if gps_h_acc_m <= 0.0 {
        return (h_rows, residual, variance);
    }

    let (lat_deg, lon_deg, _) = ecef_to_lla(pos_ecef_pre);
    let lat_rad = lat_deg.to_radians();
    let lon_rad = lon_deg.to_radians();
    let c_en = [
        [
            -lat_rad.sin() * lon_rad.cos(),
            -lat_rad.sin() * lon_rad.sin(),
            lat_rad.cos(),
        ],
        [-lon_rad.sin(), lon_rad.cos(), 0.0],
        [
            -lat_rad.cos() * lon_rad.cos(),
            -lat_rad.cos() * lon_rad.sin(),
            -lat_rad.sin(),
        ],
    ];
    let r_n_diag = [
        (gps_h_acc_m as f64).powi(2),
        (gps_h_acc_m as f64).powi(2),
        (2.5 * gps_h_acc_m as f64).powi(2),
    ];
    let mut r_e = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r_e[i][j] += c_en[i][k] * r_n_diag[k] * c_en[j][k];
            }
        }
    }
    let u11 = r_e[0][0].max(1.0e-9).sqrt();
    let u12 = r_e[0][1] / u11;
    let u13 = r_e[0][2] / u11;
    let u22 = (r_e[1][1] - u12 * u12).max(1.0e-9).sqrt();
    let u23 = (r_e[1][2] - u12 * u13) / u22;
    let u33 = (r_e[2][2] - u13 * u13 - u23 * u23).max(1.0e-9).sqrt();
    let t = [
        [1.0 / u11, 0.0, 0.0],
        [-u12 / (u11 * u22), 1.0 / u22, 0.0],
        [
            (u12 * u23 - u13 * u22) / (u11 * u22 * u33),
            -u23 / (u22 * u33),
            1.0 / u33,
        ],
    ];
    let x_meas = mat3_vec_mul(t, gps_pos_ecef);
    let x_est = mat3_vec_mul(t, pos_ecef_pre);
    for i in 0..3 {
        h_rows[i][i] = t[i][i];
        residual[i] = x_meas[i] - x_est[i];
    }
    for i in 0..3 {
        for j in 0..3 {
            h_rows[i][j] = t[i][j];
        }
    }
    let meas_var = 1.0 / (dt_since_last_gnss as f64).clamp(1.0e-3, 1.0);
    variance = [meas_var; 3];
    (h_rows, residual, variance)
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_ecef_to_ned(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    let lon = lon_deg.to_radians();
    let lat = lat_deg.to_radians();
    let half_lon = 0.5 * lon;
    let q1 = [half_lon.cos(), 0.0, 0.0, -half_lon.sin()];
    let half_lat = 0.5 * (lat + 0.5 * std::f64::consts::PI);
    let q2 = [half_lat.cos(), 0.0, half_lat.sin(), 0.0];
    quat_mul(q2, q1)
}
