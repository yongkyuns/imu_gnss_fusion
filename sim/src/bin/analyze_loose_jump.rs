use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::fusion::SensorFusion;
use sensor_fusion::loose::{LooseFilter, LooseImuDelta, LoosePredictNoise};
use sim::datasets::generic_replay::{
    GenericGnssSample, fusion_gnss_sample as to_fusion_gnss, fusion_imu_sample as to_fusion_imu,
};
use sim::datasets::ubx_replay::{UbxReplayConfig, build_generic_replay_from_frames};
use sim::ubxlog::{NavPvtObs, UbxFrame, extract_esf_alg, parse_ubx_frames};
use sim::visualizer::math::{deg2rad, ecef_to_ned, lla_to_ecef, mat_vec, nearest_master_ms};
use sim::visualizer::pipeline::align_replay::{
    BootstrapConfig as AlignBootstrapConfig, ImuReplayConfig, build_align_replay,
    frd_mount_quat_to_esf_alg_flu_quat, quat_rpy_alg_deg,
};
use sim::visualizer::pipeline::ekf_compare::EkfCompareConfig;
use sim::visualizer::pipeline::timebase::{MasterTimeline, build_master_timeline};

#[derive(Parser, Debug)]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: std::path::PathBuf,
    #[arg(long, default_value_t = 225.31)]
    target_s: f64,
    #[arg(long, default_value_t = 0.25)]
    window_s: f64,
    #[arg(long, default_value_t = 150.0)]
    predict_imu_lpf_cutoff_hz: f64,
    #[arg(long, default_value_t = 1)]
    predict_imu_decimation: usize,
    #[arg(long)]
    gnss_vel_r_scale: Option<f64>,
    #[arg(long, default_value_t = false)]
    disable_nhc: bool,
    #[arg(long, default_value_t = false)]
    raw_imu_full_qcs: bool,
    #[arg(long)]
    seeded_qcs_sigma_deg: Option<f64>,
    #[arg(long, default_value_t = false)]
    freeze_qcs: bool,
    #[arg(long, default_value_t = false)]
    disable_gps_vel: bool,
    #[arg(long, default_value_t = false)]
    zero_gps_down_vel: bool,
    #[arg(long, default_value_t = false)]
    ground_speed_vel: bool,
    #[arg(long, default_value_t = false)]
    init_qes_from_fusion: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bytes = std::fs::read(&args.logfile)
        .with_context(|| format!("failed to read {}", args.logfile.display()))?;
    let frames = parse_ubx_frames(&bytes, None);
    let tl = build_master_timeline(&frames);
    if tl.masters.is_empty() {
        bail!("no master timeline");
    }

    let cfg = EkfCompareConfig {
        predict_imu_lpf_cutoff_hz: Some(args.predict_imu_lpf_cutoff_hz),
        predict_imu_decimation: args.predict_imu_decimation.max(1),
        gnss_vel_r_scale: args
            .gnss_vel_r_scale
            .unwrap_or(EkfCompareConfig::default().gnss_vel_r_scale),
        ..Default::default()
    };

    let align_replay = build_align_replay(
        &frames,
        &tl,
        sensor_fusion::align::AlignConfig::default(),
        AlignBootstrapConfig {
            ema_alpha: 0.05,
            max_speed_mps: 0.35,
            stationary_samples: 100,
            max_gyro_radps: sensor_fusion::align::AlignConfig::default().max_stationary_gyro_radps,
            max_accel_norm_err_mps2: sensor_fusion::align::AlignConfig::default()
                .max_stationary_accel_norm_err_mps2,
            max_speed_rate_mps2: 0.15,
            max_course_rate_radps: 1.0_f32.to_radians(),
        },
        ImuReplayConfig::default(),
    );
    let mut align_events: Vec<(f64, [f32; 4])> = Vec::new();
    let mut ready = false;
    let mut first_yaw_init: Option<(f64, [f64; 4])> = None;
    let mut first_mount_ready: Option<(f64, [f64; 4])> = None;
    for sample in align_replay.samples {
        if first_yaw_init.is_none() && sample.yaw_initialized {
            first_yaw_init = Some((sample.t_s, sample.q_align));
        }
        if first_mount_ready.is_none() && sample.p_diag[2].sqrt().to_degrees() < 0.05 {
            first_mount_ready = Some((sample.t_s, sample.q_align));
        }
        if sample.yaw_initialized {
            ready = true;
        }
        if ready {
            align_events.push((
                sample.t_ms,
                [
                    sample.q_align[0] as f32,
                    sample.q_align[1] as f32,
                    sample.q_align[2] as f32,
                    sample.q_align[3] as f32,
                ],
            ));
        }
    }
    if let Some((t_s, q)) = first_yaw_init {
        let q_flu = frd_mount_quat_to_esf_alg_flu_quat(q);
        let (r, p, y) = quat_rpy_alg_deg(q_flu[0], q_flu[1], q_flu[2], q_flu[3]);
        println!(
            "standalone align first yaw_init at t={:.3}s mount=[{:.2},{:.2},{:.2}]",
            t_s, r, p, y
        );
    }
    if let Some((t_s, q)) = first_mount_ready {
        let q_flu = frd_mount_quat_to_esf_alg_flu_quat(q);
        let (r, p, y) = quat_rpy_alg_deg(q_flu[0], q_flu[1], q_flu[2], q_flu[3]);
        println!(
            "standalone align first sigma_yaw<0.05deg at t={:.3}s mount=[{:.2},{:.2},{:.2}]",
            t_s, r, p, y
        );
    }

    let replay = build_generic_replay_from_frames(
        &frames,
        &tl,
        UbxReplayConfig {
            gnss_pos_r_scale: cfg.gnss_pos_r_scale,
            gnss_vel_r_scale: cfg.gnss_vel_r_scale,
            ..UbxReplayConfig::default()
        },
    )?;
    let nav_events = replay.nav_events.clone();
    let alg_events = collect_alg_events(&frames, &tl);
    if nav_events.is_empty() {
        bail!("no nav events");
    }
    if replay.imu_samples.is_empty() {
        bail!("no imu packets");
    }

    let mut fusion = SensorFusion::new();
    let mut loose: Option<LooseFilter> = None;
    let mut loose_seed_mount_q_vb: Option<[f32; 4]> = None;
    let mut loose_last_gps_update_ms: Option<f64> = None;
    let mut prev_imu_t: Option<f64> = None;
    let mut align_idx = 0usize;
    let mut nav_idx = 0usize;
    let mut cur_align_q_vb: Option<[f32; 4]> = None;
    let ref_lat = nav_events[0].1.lat_deg;
    let ref_lon = nav_events[0].1.lon_deg;
    let ref_h = nav_events[0].1.height_m;
    let ref_ecef = lla_to_ecef(ref_lat, ref_lon, ref_h);
    let mut filt_predict_gyro: Option<[f64; 3]> = None;
    let mut filt_predict_accel: Option<[f64; 3]> = None;
    let mut predict_gyro_sum = [0.0_f64; 3];
    let mut predict_accel_sum = [0.0_f64; 3];
    let mut predict_dt_accum = 0.0_f64;
    let mut predict_decim_count = 0usize;
    let mut prev_post_mount: Option<[f64; 3]> = None;
    type JumpSummary = (f64, f64, [f64; 3], [f64; 3], Vec<String>, [f32; 24]);
    let mut top_jumps: Vec<JumpSummary> = Vec::new();

    for imu_sample in &replay.imu_samples {
        let pkt_t_ms = tl.t0_master_ms + imu_sample.t_s * 1.0e3;
        while align_idx < align_events.len() && align_events[align_idx].0 <= pkt_t_ms {
            cur_align_q_vb = Some(align_events[align_idx].1);
            align_idx += 1;
        }
        let dt = match prev_imu_t {
            Some(prev) => imu_sample.t_s - prev,
            None => {
                prev_imu_t = Some(imu_sample.t_s);
                continue;
            }
        };
        prev_imu_t = Some(imu_sample.t_s);
        if !(0.001..=0.05).contains(&dt) {
            continue;
        }

        let raw_gyro_radps = [
            imu_sample.gyro_radps[0] as f32,
            imu_sample.gyro_radps[1] as f32,
            imu_sample.gyro_radps[2] as f32,
        ];
        let raw_accel_mps2 = [
            imu_sample.accel_mps2[0] as f32,
            imu_sample.accel_mps2[1] as f32,
            imu_sample.accel_mps2[2] as f32,
        ];
        fusion.process_imu(to_fusion_imu(*imu_sample));

        let (loose_gyro_deg, loose_accel) = if let Some(q_vb) = loose_seed_mount_q_vb {
            vehicle_measurements_from_mount(
                Some(q_vb),
                [
                    raw_gyro_radps[0] as f64,
                    raw_gyro_radps[1] as f64,
                    raw_gyro_radps[2] as f64,
                ],
                [
                    raw_accel_mps2[0] as f64,
                    raw_accel_mps2[1] as f64,
                    raw_accel_mps2[2] as f64,
                ],
            )
        } else {
            vehicle_measurements_from_mount(
                None,
                [
                    raw_gyro_radps[0] as f64,
                    raw_gyro_radps[1] as f64,
                    raw_gyro_radps[2] as f64,
                ],
                [
                    raw_accel_mps2[0] as f64,
                    raw_accel_mps2[1] as f64,
                    raw_accel_mps2[2] as f64,
                ],
            )
        };

        let predict_gyro = if let Some(cutoff_hz) = cfg.predict_imu_lpf_cutoff_hz {
            let alpha_pred = lpf_alpha(dt, cutoff_hz);
            lpf_vec3(&mut filt_predict_gyro, loose_gyro_deg, alpha_pred)
        } else {
            loose_gyro_deg
        };
        let predict_accel = if let Some(cutoff_hz) = cfg.predict_imu_lpf_cutoff_hz {
            let alpha_pred = lpf_alpha(dt, cutoff_hz);
            lpf_vec3(&mut filt_predict_accel, loose_accel, alpha_pred)
        } else {
            loose_accel
        };
        predict_dt_accum += dt;
        predict_decim_count += 1;
        predict_gyro_sum[0] += predict_gyro[0];
        predict_gyro_sum[1] += predict_gyro[1];
        predict_gyro_sum[2] += predict_gyro[2];
        predict_accel_sum[0] += predict_accel[0];
        predict_accel_sum[1] += predict_accel[1];
        predict_accel_sum[2] += predict_accel[2];
        if predict_decim_count < cfg.predict_imu_decimation {
            continue;
        }

        let pred_dt = predict_dt_accum.max(1.0e-6);
        let inv_block_len = 1.0 / (predict_decim_count as f64);
        let avg_predict_gyro = [
            predict_gyro_sum[0] * inv_block_len,
            predict_gyro_sum[1] * inv_block_len,
            predict_gyro_sum[2] * inv_block_len,
        ];
        let avg_predict_accel = [
            predict_accel_sum[0] * inv_block_len,
            predict_accel_sum[1] * inv_block_len,
            predict_accel_sum[2] * inv_block_len,
        ];
        predict_dt_accum = 0.0;
        predict_decim_count = 0;
        predict_gyro_sum = [0.0; 3];
        predict_accel_sum = [0.0; 3];

        let loose_imu = LooseImuDelta {
            dax_1: (deg2rad(avg_predict_gyro[0]) * pred_dt) as f32,
            day_1: (deg2rad(avg_predict_gyro[1]) * pred_dt) as f32,
            daz_1: (deg2rad(avg_predict_gyro[2]) * pred_dt) as f32,
            dvx_1: (avg_predict_accel[0] * pred_dt) as f32,
            dvy_1: (avg_predict_accel[1] * pred_dt) as f32,
            dvz_1: (avg_predict_accel[2] * pred_dt) as f32,
            dax_2: (deg2rad(avg_predict_gyro[0]) * pred_dt) as f32,
            day_2: (deg2rad(avg_predict_gyro[1]) * pred_dt) as f32,
            daz_2: (deg2rad(avg_predict_gyro[2]) * pred_dt) as f32,
            dvx_2: (avg_predict_accel[0] * pred_dt) as f32,
            dvy_2: (avg_predict_accel[1] * pred_dt) as f32,
            dvz_2: (avg_predict_accel[2] * pred_dt) as f32,
            dt: pred_dt as f32,
        };
        if let Some(loose_ref) = loose.as_mut() {
            loose_ref.predict(loose_imu);
        }

        let loose_gyro_radps = [
            deg2rad(avg_predict_gyro[0]) as f32,
            deg2rad(avg_predict_gyro[1]) as f32,
            deg2rad(avg_predict_gyro[2]) as f32,
        ];
        let loose_accel_mps2 = [
            avg_predict_accel[0] as f32,
            avg_predict_accel[1] as f32,
            avg_predict_accel[2] as f32,
        ];

        while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt_t_ms {
            let (t_ms, nav) = nav_events[nav_idx];
            let gnss_sample = replay.gnss_samples[nav_idx];
            nav_idx += 1;
            let ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
            let _ned = ecef_to_ned(ecef, ref_ecef, ref_lat, ref_lon);
            let update = fusion.process_gnss(to_fusion_gnss(gnss_sample));
            if loose.is_none() && update.mount_ready {
                if let Some(alg) = nearest_alg(&alg_events, t_ms) {
                    println!(
                        "ESF-ALG near loose init t={:.3}s mount=[{:.2},{:.2},{:.2}]",
                        rel_s(&tl, t_ms),
                        alg[0],
                        alg[1],
                        alg[2]
                    );
                }
                if let Some(q_vb) = fusion.mount_q_vb() {
                    let q_flu = frd_mount_quat_to_esf_alg_flu_quat([
                        q_vb[0] as f64,
                        q_vb[1] as f64,
                        q_vb[2] as f64,
                        q_vb[3] as f64,
                    ]);
                    let (fr, fp, fy) = quat_rpy_alg_deg(q_flu[0], q_flu[1], q_flu[2], q_flu[3]);
                    let qc = quat_conj([
                        q_vb[0] as f64,
                        q_vb[1] as f64,
                        q_vb[2] as f64,
                        q_vb[3] as f64,
                    ]);
                    let (fcr, fcp, fcy) = quat_rpy_alg_deg(qc[0], qc[1], qc[2], qc[3]);
                    let q_flu_conj = frd_mount_quat_to_esf_alg_flu_quat(qc);
                    let (frr2, fpp2, fyy2) = quat_rpy_alg_deg(
                        q_flu_conj[0],
                        q_flu_conj[1],
                        q_flu_conj[2],
                        q_flu_conj[3],
                    );
                    println!(
                        "fusion mount at loose init t={:.3}s mount_conv=[{:.2},{:.2},{:.2}] raw_conj=[{:.2},{:.2},{:.2}] conv_conj=[{:.2},{:.2},{:.2}]",
                        rel_s(&tl, t_ms),
                        fr,
                        fp,
                        fy,
                        fcr,
                        fcp,
                        fcy,
                        frr2,
                        fpp2,
                        fyy2
                    );
                }
                if let Some(q_vb) = cur_align_q_vb.or_else(|| fusion.mount_q_vb()) {
                    let init_q_cs = if args.raw_imu_full_qcs {
                        [q_vb[0], q_vb[1], q_vb[2], q_vb[3]]
                    } else {
                        [1.0, 0.0, 0.0, 0.0]
                    };
                    loose_seed_mount_q_vb = if args.raw_imu_full_qcs {
                        None
                    } else {
                        Some(q_vb)
                    };
                    let q_flu = frd_mount_quat_to_esf_alg_flu_quat([
                        q_vb[0] as f64,
                        q_vb[1] as f64,
                        q_vb[2] as f64,
                        q_vb[3] as f64,
                    ]);
                    let (sr, sp, sy) = quat_rpy_alg_deg(q_flu[0], q_flu[1], q_flu[2], q_flu[3]);
                    let (raw_r, raw_p, raw_y) = quat_rpy_alg_deg(
                        q_vb[0] as f64,
                        q_vb[1] as f64,
                        q_vb[2] as f64,
                        q_vb[3] as f64,
                    );
                    let speed_h = nav.vel_n_mps.hypot(nav.vel_e_mps);
                    let vel_course_deg = nav.vel_e_mps.atan2(nav.vel_n_mps).to_degrees();
                    let chosen_yaw_deg = initial_yaw_from_nav(nav).to_degrees() as f64;
                    println!(
                        "loose init at t={:.3}s seed_mount_conv=[{:.2},{:.2},{:.2}] seed_mount_raw=[{:.2},{:.2},{:.2}] source={}",
                        rel_s(&tl, t_ms),
                        sr,
                        sp,
                        sy,
                        raw_r,
                        raw_p,
                        raw_y,
                        if cur_align_q_vb.is_some() {
                            "align_events"
                        } else {
                            "fusion_mount_q_vb"
                        }
                    );
                    println!(
                        "loose init nav at t={:.3}s vel_ned=[{:.3},{:.3},{:.3}] speed_h={:.3} head_veh_valid={} heading_vehicle={:.2} heading_motion={:.2} vel_course={:.2} chosen_yaw={:.2}",
                        rel_s(&tl, t_ms),
                        nav.vel_n_mps,
                        nav.vel_e_mps,
                        nav.vel_d_mps,
                        speed_h,
                        nav.head_veh_valid,
                        nav.heading_vehicle_deg,
                        nav.heading_motion_deg,
                        vel_course_deg,
                        chosen_yaw_deg,
                    );
                    let q_es_init = if args.init_qes_from_fusion {
                        fusion.eskf().map(|eskf| {
                            let q_nb = [
                                eskf.nominal.q0 as f64,
                                eskf.nominal.q1 as f64,
                                eskf.nominal.q2 as f64,
                                eskf.nominal.q3 as f64,
                            ];
                            if args.raw_imu_full_qcs {
                                [
                                    q_nb[0] as f32,
                                    q_nb[1] as f32,
                                    q_nb[2] as f32,
                                    q_nb[3] as f32,
                                ]
                            } else {
                                let q_seed = [
                                    q_vb[0] as f64,
                                    q_vb[1] as f64,
                                    q_vb[2] as f64,
                                    q_vb[3] as f64,
                                ];
                                let q_nv = quat_mul(q_nb, quat_conj(q_seed));
                                let q_es = quat_mul(
                                    quat_conj(quat_ecef_to_ned(nav.lat_deg, nav.lon_deg)),
                                    q_nv,
                                );
                                [
                                    q_es[0] as f32,
                                    q_es[1] as f32,
                                    q_es[2] as f32,
                                    q_es[3] as f32,
                                ]
                            }
                        })
                    } else {
                        None
                    };
                    let mut loose_init =
                        initialize_loose_from_nav(nav, gnss_sample, init_q_cs, q_es_init);
                    if !args.raw_imu_full_qcs
                        && let Some(sigma_deg) = args.seeded_qcs_sigma_deg
                    {
                        loose_init.tighten_mount_covariance_deg(sigma_deg as f32);
                    }
                    if args.freeze_qcs {
                        let mut p = *loose_init.covariance();
                        for i in 21..24 {
                            for j in 0..24 {
                                p[i][j] = 0.0;
                                p[j][i] = 0.0;
                            }
                        }
                        loose_init.set_covariance(p);
                    }
                    loose = Some(loose_init);
                    loose_last_gps_update_ms = Some(t_ms);
                }
            }
            if let Some(loose_ref) = loose.as_mut() {
                let t_s = rel_s(&tl, t_ms);
                let vel_ned_for_loose = if args.disable_gps_vel {
                    None
                } else if args.ground_speed_vel {
                    let speed_h = nav.vel_n_mps.hypot(nav.vel_e_mps);
                    let heading_deg = if nav.head_veh_valid {
                        nav.heading_vehicle_deg
                    } else {
                        nav.heading_motion_deg
                    };
                    let heading_rad = deg2rad(heading_deg);
                    Some([
                        speed_h * heading_rad.cos(),
                        speed_h * heading_rad.sin(),
                        0.0,
                    ])
                } else {
                    Some([
                        nav.vel_n_mps,
                        nav.vel_e_mps,
                        if args.zero_gps_down_vel {
                            0.0
                        } else {
                            nav.vel_d_mps
                        },
                    ])
                };
                let vel_ecef = vel_ned_for_loose.map(|v_ned| {
                    mat_vec(
                        transpose3(ecef_to_ned_matrix(nav.lat_deg, nav.lon_deg)),
                        v_ned,
                    )
                });
                let vel_std_ned = if args.disable_gps_vel {
                    None
                } else {
                    Some([
                        (nav.s_acc_mps * cfg.gnss_vel_r_scale.sqrt()) as f32,
                        (nav.s_acc_mps * cfg.gnss_vel_r_scale.sqrt()) as f32,
                        (nav.s_acc_mps * cfg.gnss_vel_r_scale.sqrt()) as f32,
                    ])
                };
                let dt_since_last_gnss_s = loose_last_gps_update_ms
                    .map(|last_t_ms| ((t_ms - last_t_ms) * 1.0e-3) as f32)
                    .unwrap_or(1.0)
                    .clamp(1.0e-3, 1.0);
                let pre_qcs = qcs_rpy(loose_ref, loose_seed_mount_q_vb);
                let pre_nom = *loose_ref.nominal();
                let pre_vel_ned = mat_vec(
                    ecef_to_ned_matrix(ref_lat, ref_lon),
                    [pre_nom.vn as f64, pre_nom.ve as f64, pre_nom.vd as f64],
                );
                let pre_pos_ecef = [pre_nom.pn as f64, pre_nom.pe as f64, pre_nom.pd as f64];
                let pre_pos_res_ned = ecef_to_ned(pre_pos_ecef, ecef, nav.lat_deg, nav.lon_deg);
                let pre_vel_res_ned = [
                    pre_vel_ned[0] - nav.vel_n_mps,
                    pre_vel_ned[1] - nav.vel_e_mps,
                    pre_vel_ned[2] - nav.vel_d_mps,
                ];
                let pre_q_ns = quat_mul(
                    quat_ecef_to_ned(ref_lat, ref_lon),
                    [
                        pre_nom.q0 as f64,
                        pre_nom.q1 as f64,
                        pre_nom.q2 as f64,
                        pre_nom.q3 as f64,
                    ],
                );
                let pre_q_cs = [
                    pre_nom.qcs0 as f64,
                    pre_nom.qcs1 as f64,
                    pre_nom.qcs2 as f64,
                    pre_nom.qcs3 as f64,
                ];
                let pre_q_nc = quat_mul(pre_q_ns, quat_conj(pre_q_cs));
                let pre_vel_vehicle =
                    mat_vec(transpose3(quat_to_rotmat_f64(pre_q_nc)), pre_vel_ned);
                let (pre_roll, pre_pitch, pre_yaw) = quat_rpy_deg(pre_q_nc);
                if args.disable_nhc {
                    // Keep the same GPS pos+vel batch path, but force the internal
                    // NHC gate off by supplying clearly non-quasi-static IMU values.
                    loose_ref.fuse_reference_batch_full(
                        Some(ecef),
                        vel_ecef.map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]),
                        (nav.h_acc_m * cfg.gnss_pos_r_scale.sqrt()) as f32,
                        vel_std_ned,
                        dt_since_last_gnss_s,
                        [10.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        loose_imu.dt,
                    );
                } else {
                    loose_ref.fuse_reference_batch_full(
                        Some(ecef),
                        vel_ecef.map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]),
                        (nav.h_acc_m * cfg.gnss_pos_r_scale.sqrt()) as f32,
                        vel_std_ned,
                        dt_since_last_gnss_s,
                        loose_gyro_radps,
                        loose_accel_mps2,
                        loose_imu.dt,
                    );
                }
                loose_last_gps_update_ms = Some(t_ms);
                if args.freeze_qcs {
                    let q_locked = if args.raw_imu_full_qcs {
                        loose_seed_mount_q_vb.unwrap_or([1.0, 0.0, 0.0, 0.0])
                    } else {
                        [1.0, 0.0, 0.0, 0.0]
                    };
                    loose_ref.set_mount_quat(q_locked);
                    let mut p = *loose_ref.covariance();
                    for i in 21..24 {
                        for j in 0..24 {
                            p[i][j] = 0.0;
                            p[j][i] = 0.0;
                        }
                    }
                    loose_ref.set_covariance(p);
                }
                let post_qcs = qcs_rpy(loose_ref, loose_seed_mount_q_vb);
                let dx = *loose_ref.last_dx();
                if let Some(prev_mount) = prev_post_mount {
                    let jump = angle3_diff_deg(post_qcs, prev_mount);
                    if jump > 1.0 {
                        top_jumps.push((
                            t_s,
                            jump,
                            prev_mount,
                            post_qcs,
                            obs_names(loose_ref.last_obs_types())
                                .into_iter()
                                .map(|s| s.to_string())
                                .collect(),
                            dx,
                        ));
                        top_jumps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        top_jumps.truncate(10);
                    }
                }
                prev_post_mount = Some(post_qcs);
                if (t_s - args.target_s).abs() <= args.window_s {
                    println!(
                        "t={:.3}s obs={:?} veh_att_pre=[{:.2},{:.2},{:.2}] vel_vehicle_pre=[{:.3},{:.3},{:.3}] pos_res_ned_pre=[{:.3},{:.3},{:.3}] vel_res_ned_pre=[{:.3},{:.3},{:.3}] full_mount_pre=[{:.2},{:.2},{:.2}] post=[{:.2},{:.2},{:.2}] dpsi_cc=[{:.3},{:.3},{:.3}] dtheta_e=[{:.3},{:.3},{:.3}] dv_e=[{:.3},{:.3},{:.3}] dba=[{:.3},{:.3},{:.3}] v_e_pre=[{:.3},{:.3},{:.3}] accel_bias_pre=[{:.3},{:.3},{:.3}]",
                        t_s,
                        obs_names(loose_ref.last_obs_types()),
                        pre_roll,
                        pre_pitch,
                        pre_yaw,
                        pre_vel_vehicle[0],
                        pre_vel_vehicle[1],
                        pre_vel_vehicle[2],
                        pre_pos_res_ned[0],
                        pre_pos_res_ned[1],
                        pre_pos_res_ned[2],
                        pre_vel_res_ned[0],
                        pre_vel_res_ned[1],
                        pre_vel_res_ned[2],
                        pre_qcs[0],
                        pre_qcs[1],
                        pre_qcs[2],
                        post_qcs[0],
                        post_qcs[1],
                        post_qcs[2],
                        dx[21].to_degrees(),
                        dx[22].to_degrees(),
                        dx[23].to_degrees(),
                        dx[6].to_degrees(),
                        dx[7].to_degrees(),
                        dx[8].to_degrees(),
                        dx[3],
                        dx[4],
                        dx[5],
                        dx[9],
                        dx[10],
                        dx[11],
                        pre_nom.vn,
                        pre_nom.ve,
                        pre_nom.vd,
                        pre_nom.bax,
                        pre_nom.bay,
                        pre_nom.baz,
                    );
                }
            }
        }
    }

    println!("\nTop full-mount jumps:");
    for (t_s, jump, pre, post, obs, dx) in top_jumps {
        let alg = nearest_alg(&alg_events, tl.t0_master_ms + t_s * 1.0e3).unwrap_or([f64::NAN; 3]);
        let err = [
            wrap_deg(post[0] - alg[0]),
            wrap_deg(post[1] - alg[1]),
            wrap_deg(post[2] - alg[2]),
        ];
        println!(
            "t={:.3}s jump={:.2}deg obs={:?} pre=[{:.2},{:.2},{:.2}] post=[{:.2},{:.2},{:.2}] alg=[{:.2},{:.2},{:.2}] err=[{:.2},{:.2},{:.2}] dpsi_cc=[{:.2},{:.2},{:.2}] dtheta_e=[{:.2},{:.2},{:.2}] dv_e=[{:.3},{:.3},{:.3}] dba=[{:.3},{:.3},{:.3}]",
            t_s,
            jump,
            obs,
            pre[0],
            pre[1],
            pre[2],
            post[0],
            post[1],
            post[2],
            alg[0],
            alg[1],
            alg[2],
            err[0],
            err[1],
            err[2],
            dx[21].to_degrees(),
            dx[22].to_degrees(),
            dx[23].to_degrees(),
            dx[6].to_degrees(),
            dx[7].to_degrees(),
            dx[8].to_degrees(),
            dx[3],
            dx[4],
            dx[5],
            dx[9],
            dx[10],
            dx[11],
        );
    }

    if let Some(loose_ref) = loose.as_ref() {
        let n = loose_ref.nominal();
        let vel_ned = mat_vec(
            ecef_to_ned_matrix(ref_lat, ref_lon),
            [n.vn as f64, n.ve as f64, n.vd as f64],
        );
        let q_ns = quat_mul(
            quat_ecef_to_ned(ref_lat, ref_lon),
            [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64],
        );
        let q_cs = [n.qcs0 as f64, n.qcs1 as f64, n.qcs2 as f64, n.qcs3 as f64];
        let q_nc = quat_mul(q_ns, quat_conj(q_cs));
        let vel_vehicle = mat_vec(transpose3(quat_to_rotmat_f64(q_nc)), vel_ned);
        let (att_r, att_p, att_y) = quat_rpy_deg(q_nc);
        let (seed_r, seed_p, seed_y) = quat_rpy_deg(q_ns);
        let mount = qcs_rpy(loose_ref, loose_seed_mount_q_vb);
        println!(
            "\nFinal loose state: veh_att=[{:.2},{:.2},{:.2}] seeded_att=[{:.2},{:.2},{:.2}] vel_vehicle=[{:.2},{:.2},{:.2}] vel_ned=[{:.2},{:.2},{:.2}] accel_bias=[{:.3},{:.3},{:.3}] mount=[{:.2},{:.2},{:.2}]",
            att_r,
            att_p,
            att_y,
            seed_r,
            seed_p,
            seed_y,
            vel_vehicle[0],
            vel_vehicle[1],
            vel_vehicle[2],
            vel_ned[0],
            vel_ned[1],
            vel_ned[2],
            n.bax,
            n.bay,
            n.baz,
            mount[0],
            mount[1],
            mount[2],
        );
    }

    Ok(())
}

fn angle3_diff_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dr = wrap_deg(a[0] - b[0]).abs();
    let dp = wrap_deg(a[1] - b[1]).abs();
    let dy = wrap_deg(a[2] - b[2]).abs();
    dr.max(dp).max(dy)
}

fn wrap_deg(x: f64) -> f64 {
    let mut y = x;
    while y > 180.0 {
        y -= 360.0;
    }
    while y < -180.0 {
        y += 360.0;
    }
    y
}

fn obs_names(xs: &[i32]) -> Vec<&'static str> {
    xs.iter()
        .map(|x| match *x {
            1 => "GPS_POS_X",
            2 => "GPS_POS_Y",
            3 => "GPS_POS_Z",
            4 => "GPS_VEL_X",
            5 => "GPS_VEL_Y",
            6 => "GPS_VEL_Z",
            7 => "NHC_Y",
            8 => "NHC_Z",
            _ => "UNKNOWN",
        })
        .collect()
}

fn rel_s(tl: &MasterTimeline, t_ms: f64) -> f64 {
    (t_ms - tl.t0_master_ms) * 1.0e-3
}

fn collect_alg_events(frames: &[UbxFrame], tl: &MasterTimeline) -> Vec<(f64, [f64; 3])> {
    let mut out = Vec::new();
    for f in frames {
        if let Some((_, roll, pitch, yaw)) = extract_esf_alg(f)
            && let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters)
        {
            out.push((t_ms, [roll, pitch, yaw]));
        }
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out
}

fn nearest_alg(events: &[(f64, [f64; 3])], t_ms: f64) -> Option<[f64; 3]> {
    events
        .iter()
        .min_by(|a, b| (a.0 - t_ms).abs().partial_cmp(&(b.0 - t_ms).abs()).unwrap())
        .map(|x| x.1)
}

fn initial_yaw_from_nav(nav: NavPvtObs) -> f32 {
    let speed_h = nav.vel_n_mps.hypot(nav.vel_e_mps);
    if nav.head_veh_valid {
        deg2rad(nav.heading_vehicle_deg) as f32
    } else if speed_h >= 1.0 {
        nav.vel_e_mps.atan2(nav.vel_n_mps) as f32
    } else {
        deg2rad(nav.heading_motion_deg) as f32
    }
}

fn yaw_quat_f32(yaw_rad: f32) -> [f32; 4] {
    let half = 0.5 * yaw_rad;
    [half.cos(), 0.0, 0.0, half.sin()]
}

fn default_loose_reference_p_diag(gnss: sensor_fusion::fusion::FusionGnssSample) -> [f32; 24] {
    const DEFAULT_GYRO_BIAS_SIGMA_DPS: f32 = 0.125;
    const DEFAULT_ACCEL_BIAS_SIGMA_MPS2: f32 = 0.075;
    const DEFAULT_GYRO_SCALE_SIGMA: f32 = 0.02;
    const DEFAULT_ACCEL_SCALE_SIGMA: f32 = 0.02;

    let att_sigma_rad = 2.0f32 * core::f32::consts::PI / 180.0;
    let att_var = att_sigma_rad * att_sigma_rad;
    let mut vel_std = gnss.vel_std_mps[0]
        .max(gnss.vel_std_mps[1])
        .max(gnss.vel_std_mps[2]);
    if vel_std < 0.2 {
        vel_std = 0.2;
    }
    let vel_var = vel_std * vel_std;
    let pos_n = gnss.pos_std_m[0].max(0.5);
    let pos_e = gnss.pos_std_m[1].max(0.5);
    let pos_d = gnss.pos_std_m[2].max(0.5);
    let gyro_bias_sigma_radps = DEFAULT_GYRO_BIAS_SIGMA_DPS * core::f32::consts::PI / 180.0;
    let accel_bias_sigma_mps2 = DEFAULT_ACCEL_BIAS_SIGMA_MPS2;

    let mut p_diag = [0.0_f32; 24];
    p_diag[0] = pos_n * pos_n;
    p_diag[1] = pos_e * pos_e;
    p_diag[2] = pos_d * pos_d;
    p_diag[3] = vel_var;
    p_diag[4] = vel_var;
    p_diag[5] = vel_var;
    p_diag[6] = att_var;
    p_diag[7] = att_var;
    p_diag[8] = att_var;
    p_diag[9] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
    p_diag[10] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
    p_diag[11] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
    p_diag[12] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
    p_diag[13] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
    p_diag[14] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
    p_diag[15] = DEFAULT_ACCEL_SCALE_SIGMA * DEFAULT_ACCEL_SCALE_SIGMA;
    p_diag[16] = DEFAULT_ACCEL_SCALE_SIGMA * DEFAULT_ACCEL_SCALE_SIGMA;
    p_diag[17] = DEFAULT_ACCEL_SCALE_SIGMA * DEFAULT_ACCEL_SCALE_SIGMA;
    p_diag[18] = DEFAULT_GYRO_SCALE_SIGMA * DEFAULT_GYRO_SCALE_SIGMA;
    p_diag[19] = DEFAULT_GYRO_SCALE_SIGMA * DEFAULT_GYRO_SCALE_SIGMA;
    p_diag[20] = DEFAULT_GYRO_SCALE_SIGMA * DEFAULT_GYRO_SCALE_SIGMA;
    p_diag[21] = att_var;
    p_diag[22] = att_var;
    p_diag[23] = att_var;
    p_diag
}

fn initialize_loose_from_nav(
    nav: NavPvtObs,
    gnss: GenericGnssSample,
    q_cs_init: [f32; 4],
    q_es_init: Option<[f32; 4]>,
) -> LooseFilter {
    let mut loose = LooseFilter::new(LoosePredictNoise::lsm6dso_loose_104hz());
    let p_diag = default_loose_reference_p_diag(to_fusion_gnss(gnss));
    let heading_rad = gnss
        .heading_rad
        .map(|x| x as f32)
        .unwrap_or_else(|| initial_yaw_from_nav(nav));
    let vel_ecef = mat_vec(
        transpose3(ecef_to_ned_matrix(nav.lat_deg, nav.lon_deg)),
        [nav.vel_n_mps, nav.vel_e_mps, nav.vel_d_mps],
    );
    if q_es_init.is_none() && q_cs_init == [1.0, 0.0, 0.0, 0.0] {
        loose.init_seeded_vehicle_from_nav_ecef_state(
            heading_rad,
            nav.lat_deg,
            nav.lon_deg,
            lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m),
            [vel_ecef[0] as f32, vel_ecef[1] as f32, vel_ecef[2] as f32],
            Some(p_diag),
            None,
        );
    } else {
        let q_es = if let Some(q) = q_es_init {
            [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64]
        } else {
            let q_ns = yaw_quat_f32(heading_rad);
            quat_mul(
                quat_conj(quat_ecef_to_ned(nav.lat_deg, nav.lon_deg)),
                [
                    q_ns[0] as f64,
                    q_ns[1] as f64,
                    q_ns[2] as f64,
                    q_ns[3] as f64,
                ],
            )
        };
        loose.init_from_reference_ecef_state(
            [
                q_es[0] as f32,
                q_es[1] as f32,
                q_es[2] as f32,
                q_es[3] as f32,
            ],
            lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m),
            [vel_ecef[0] as f32, vel_ecef[1] as f32, vel_ecef[2] as f32],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            q_cs_init,
            Some(p_diag),
        );
    }
    loose
}

fn vehicle_measurements_from_mount(
    q_vb: Option<[f32; 4]>,
    raw_gyro_radps: [f64; 3],
    raw_accel_mps2: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let Some(q_vb) = q_vb else {
        return (
            [
                raw_gyro_radps[0].to_degrees(),
                raw_gyro_radps[1].to_degrees(),
                raw_gyro_radps[2].to_degrees(),
            ],
            raw_accel_mps2,
        );
    };
    let c_bv = transpose3(quat_to_rotmat_f64([
        q_vb[0] as f64,
        q_vb[1] as f64,
        q_vb[2] as f64,
        q_vb[3] as f64,
    ]));
    let gyro_vehicle_radps = mat_vec(c_bv, raw_gyro_radps);
    let accel_vehicle_mps2 = mat_vec(c_bv, raw_accel_mps2);
    (
        [
            gyro_vehicle_radps[0].to_degrees(),
            gyro_vehicle_radps[1].to_degrees(),
            gyro_vehicle_radps[2].to_degrees(),
        ],
        accel_vehicle_mps2,
    )
}

fn qcs_rpy(loose: &LooseFilter, seed_mount_q_vb: Option<[f32; 4]>) -> [f64; 3] {
    let n = loose.nominal();
    let q_seed = seed_mount_q_vb
        .map(|q| [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64])
        .unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let q_cs = [n.qcs0 as f64, n.qcs1 as f64, n.qcs2 as f64, n.qcs3 as f64];
    let q_total_vb = quat_mul(q_cs, q_seed);
    let q_total_flu = frd_mount_quat_to_esf_alg_flu_quat(q_total_vb);
    let (r, p, y) = quat_rpy_alg_deg(
        q_total_flu[0],
        q_total_flu[1],
        q_total_flu[2],
        q_total_flu[3],
    );
    [r, p, y]
}

fn ecef_to_ned_matrix(lat_deg: f64, lon_deg: f64) -> [[f64; 3]; 3] {
    let lat = deg2rad(lat_deg);
    let lon = deg2rad(lon_deg);
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    [
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [-clat * clon, -clat * slon, -slat],
    ]
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

fn quat_rpy_deg(q: [f64; 4]) -> (f64, f64, f64) {
    let r = quat_to_rotmat_f64(q);
    let roll = r[2][1].atan2(r[2][2]);
    let pitch = (-r[2][0]).asin();
    let yaw = r[1][0].atan2(r[0][0]);
    (roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees())
}

fn quat_ecef_to_ned(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    let lon = deg2rad(lon_deg);
    let lat = deg2rad(lat_deg);
    let half_lon = 0.5 * lon;
    let q_lon = [half_lon.cos(), 0.0, 0.0, -half_lon.sin()];
    let half_lat = 0.5 * (lat + 0.5 * std::f64::consts::PI);
    let q_lat = [half_lat.cos(), 0.0, half_lat.sin(), 0.0];
    quat_mul(q_lat, q_lon)
}

fn quat_to_rotmat_f64(q: [f64; 4]) -> [[f64; 3]; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (w, x, y, z) = if n > 1.0e-12 {
        (q[0] / n, q[1] / n, q[2] / n, q[3] / n)
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };
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

fn transpose3(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn lpf_alpha(dt_s: f64, cutoff_hz: f64) -> f64 {
    let dt_s = dt_s.max(1.0e-6);
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz.max(1.0e-6));
    (dt_s / (tau + dt_s)).clamp(0.0, 1.0)
}

fn lpf_vec3(state: &mut Option<[f64; 3]>, sample: [f64; 3], alpha: f64) -> [f64; 3] {
    let next = match *state {
        Some(prev) => [
            prev[0] + alpha * (sample[0] - prev[0]),
            prev[1] + alpha * (sample[1] - prev[1]),
            prev[2] + alpha * (sample[2] - prev[2]),
        ],
        None => sample,
    };
    *state = Some(next);
    next
}
