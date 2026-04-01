use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::align::{AlignConfig, AlignWindowSummary, GRAVITY_MPS2};
use sensor_fusion::c_api::{CAlign, CSensorFusionWrapper};
use sensor_fusion::fusion::{FusionConfig, FusionGnssSample, FusionImuSample, SensorFusion};
use sim::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_raw_samples, extract_nav2_pvt_obs,
    fit_linear_map, parse_ubx_frames, sensor_meta, unwrap_counter,
};
use sim::visualizer::model::ImuPacket;
use sim::visualizer::math::{ecef_to_ned, lla_to_ecef, nearest_master_ms};
use sim::visualizer::pipeline::align_replay::{
    BootstrapConfig, build_align_replay, esf_alg_flu_to_frd_mount_quat, quat_rotate,
    signed_projected_axis_angle_deg,
};
use sim::visualizer::pipeline::timebase::{MasterTimeline, build_master_timeline};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "test_files.txt")]
    file_list: PathBuf,

    #[arg(long, default_value = "Baseline")]
    section: String,

    #[arg(long, default_value = "logger/data")]
    data_dir: PathBuf,

    #[arg(long, default_value_t = false)]
    fail_on_mismatch: bool,
}

#[derive(Clone, Copy, Debug)]
struct AlignMetrics {
    ready_t_s: Option<f64>,
    ready_fwd_err_deg: Option<f64>,
    ready_down_err_deg: Option<f64>,
    final_fwd_err_deg: f64,
    final_down_err_deg: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct FusionParityMetrics {
    mount_ready_t_s: Option<f64>,
    ekf_init_t_s: Option<f64>,
    sample_count: usize,
    rms_pos_m: f64,
    rms_vel_mps: f64,
    rms_att_deg: f64,
    rms_cov_diag: f64,
    final_pos_m: f64,
    final_vel_mps: f64,
    final_att_deg: f64,
    final_cov_diag: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct FusionDiffSample {
    pos_m: f64,
    vel_mps: f64,
    att_deg: f64,
    cov_diag: f64,
}

#[derive(Clone, Debug)]
struct FileComparison {
    file: String,
    rust_align: AlignMetrics,
    c_align: AlignMetrics,
    fusion: FusionParityPair,
}

#[derive(Clone, Copy, Debug, Default)]
struct FusionParityPair {
    rust: FusionParityMetrics,
    c: FusionParityMetrics,
    diff: FusionParityMetrics,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let files = parse_section_files(&args.file_list, &args.section)?;
    if files.is_empty() {
        bail!("no files found in section {}", args.section);
    }

    let align_cfg = AlignConfig::default();
    let bootstrap_cfg = BootstrapConfig {
        ema_alpha: 0.05,
        max_speed_mps: 0.35,
        stationary_samples: 300,
        max_gyro_radps: align_cfg.max_stationary_gyro_radps,
        max_accel_norm_err_mps2: align_cfg.max_stationary_accel_norm_err_mps2,
    };
    let fusion_cfg = FusionConfig::default();

    println!(
        "file,rust_ready_s,c_ready_s,ready_dt_s,rust_ready_fwd_deg,c_ready_fwd_deg,rust_ready_down_deg,c_ready_down_deg,rust_final_fwd_deg,c_final_fwd_deg,rust_final_down_deg,c_final_down_deg,rust_mount_ready_s,c_mount_ready_s,rust_ekf_init_s,c_ekf_init_s,ekf_samples,rms_pos_m,rms_vel_mps,rms_att_deg,rms_cov_diag,final_pos_m,final_vel_mps,final_att_deg,final_cov_diag"
    );

    let mut results = Vec::new();
    for file in files {
        let log_path = args.data_dir.join(&file);
        let data = fs::read(&log_path)
            .with_context(|| format!("failed to read {}", log_path.display()))?;
        let frames = parse_ubx_frames(&data, None);
        let tl = build_master_timeline(&frames);

        let rust_replay = build_align_replay(&frames, &tl, align_cfg, bootstrap_cfg);
        let rust_align = align_metrics_from_replay(&rust_replay)
            .with_context(|| format!("missing ESF-ALG reference for {file}"))?;
        let c_align = run_c_align_replay(&frames, &tl, align_cfg, bootstrap_cfg)
            .with_context(|| format!("C align replay failed for {file}"))?;
        let fusion = compare_fusion_pair(&frames, &tl, fusion_cfg);

        println!(
            "{file},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            fmt_opt(rust_align.ready_t_s),
            fmt_opt(c_align.ready_t_s),
            fmt_opt(opt_abs_diff(rust_align.ready_t_s, c_align.ready_t_s)),
            fmt_opt(rust_align.ready_fwd_err_deg),
            fmt_opt(c_align.ready_fwd_err_deg),
            fmt_opt(rust_align.ready_down_err_deg),
            fmt_opt(c_align.ready_down_err_deg),
            rust_align.final_fwd_err_deg,
            c_align.final_fwd_err_deg,
            rust_align.final_down_err_deg,
            c_align.final_down_err_deg,
            fmt_opt(fusion.rust.mount_ready_t_s),
            fmt_opt(fusion.c.mount_ready_t_s),
            fmt_opt(fusion.rust.ekf_init_t_s),
            fmt_opt(fusion.c.ekf_init_t_s),
            fusion.diff.sample_count,
            fusion.diff.rms_pos_m,
            fusion.diff.rms_vel_mps,
            fusion.diff.rms_att_deg,
            fusion.diff.rms_cov_diag,
            fusion.diff.final_pos_m,
            fusion.diff.final_vel_mps,
            fusion.diff.final_att_deg,
            fusion.diff.final_cov_diag,
        );

        results.push(FileComparison {
            file,
            rust_align,
            c_align,
            fusion,
        });
    }

    print_summary(&results);
    if args.fail_on_mismatch {
        fail_on_mismatch(&results)?;
    }

    Ok(())
}

fn print_summary(results: &[FileComparison]) {
    let mut ready_time_mismatches = 0usize;
    let mut align_final_bad = 0usize;
    let mut fusion_init_mismatches = 0usize;
    let mut fusion_large_diffs = 0usize;

    for r in results {
        if opt_abs_diff(r.rust_align.ready_t_s, r.c_align.ready_t_s)
            .is_some_and(|dt| dt > 1.0e-6)
            || r.rust_align.ready_t_s.is_some() != r.c_align.ready_t_s.is_some()
        {
            ready_time_mismatches += 1;
        }
        if (r.rust_align.final_fwd_err_deg - r.c_align.final_fwd_err_deg).abs() > 1.0e-3
            || (r.rust_align.final_down_err_deg - r.c_align.final_down_err_deg).abs() > 1.0e-3
        {
            align_final_bad += 1;
        }
        if opt_abs_diff(r.fusion.rust.ekf_init_t_s, r.fusion.c.ekf_init_t_s)
            .is_some_and(|dt| dt > 1.0e-6)
            || r.fusion.rust.ekf_init_t_s.is_some() != r.fusion.c.ekf_init_t_s.is_some()
        {
            fusion_init_mismatches += 1;
        }
        if r.fusion.diff.rms_pos_m > 1.0e-3
            || r.fusion.diff.rms_vel_mps > 1.0e-4
            || r.fusion.diff.rms_att_deg > 1.0e-3
            || r.fusion.diff.rms_cov_diag > 1.0e-5
        {
            fusion_large_diffs += 1;
        }
    }

    println!();
    println!("Summary:");
    println!("  logs: {}", results.len());
    println!("  coarse-ready time mismatches: {}", ready_time_mismatches);
    println!("  align final error mismatches: {}", align_final_bad);
    println!("  EKF init time mismatches: {}", fusion_init_mismatches);
    println!("  EKF trend mismatches: {}", fusion_large_diffs);
}

fn fail_on_mismatch(results: &[FileComparison]) -> Result<()> {
    for r in results {
        if opt_abs_diff(r.rust_align.ready_t_s, r.c_align.ready_t_s)
            .is_some_and(|dt| dt > 1.0e-6)
            || r.rust_align.ready_t_s.is_some() != r.c_align.ready_t_s.is_some()
        {
            bail!("coarse-ready mismatch on {}", r.file);
        }
        if (r.rust_align.final_fwd_err_deg - r.c_align.final_fwd_err_deg).abs() > 1.0e-3
            || (r.rust_align.final_down_err_deg - r.c_align.final_down_err_deg).abs() > 1.0e-3
        {
            bail!("align final mismatch on {}", r.file);
        }
        if opt_abs_diff(r.fusion.rust.ekf_init_t_s, r.fusion.c.ekf_init_t_s)
            .is_some_and(|dt| dt > 1.0e-6)
            || r.fusion.rust.ekf_init_t_s.is_some() != r.fusion.c.ekf_init_t_s.is_some()
        {
            bail!("EKF init mismatch on {}", r.file);
        }
        if r.fusion.diff.rms_pos_m > 1.0e-3
            || r.fusion.diff.rms_vel_mps > 1.0e-4
            || r.fusion.diff.rms_att_deg > 1.0e-3
            || r.fusion.diff.rms_cov_diag > 1.0e-5
        {
            bail!("EKF trend mismatch on {}", r.file);
        }
    }
    Ok(())
}

fn align_metrics_from_replay(replay: &sim::visualizer::pipeline::align_replay::AlignReplayData) -> Result<AlignMetrics> {
    let final_alg_q = replay.final_alg_q.context("missing final ESF-ALG")?;
    let ref_fwd = quat_rotate(final_alg_q, [1.0, 0.0, 0.0]);
    let ref_down = quat_rotate(final_alg_q, [0.0, 0.0, 1.0]);
    let ref_right = quat_rotate(final_alg_q, [0.0, 1.0, 0.0]);

    let last = replay.samples.last().context("missing align samples")?;
    let final_align_fwd = quat_rotate(last.q_align, [1.0, 0.0, 0.0]);
    let final_align_down = quat_rotate(last.q_align, [0.0, 0.0, 1.0]);
    let final_fwd_err_deg =
        signed_projected_axis_angle_deg(final_align_fwd, ref_fwd, ref_down).abs();
    let final_down_err_deg =
        signed_projected_axis_angle_deg(final_align_down, ref_down, ref_right).abs();

    let ready = replay.samples.iter().find(|s| s.yaw_initialized).map(|s| {
        let ready_align_fwd = quat_rotate(s.q_align, [1.0, 0.0, 0.0]);
        let ready_align_down = quat_rotate(s.q_align, [0.0, 0.0, 1.0]);
        (
            s.t_s,
            signed_projected_axis_angle_deg(ready_align_fwd, ref_fwd, ref_down).abs(),
            signed_projected_axis_angle_deg(ready_align_down, ref_down, ref_right).abs(),
        )
    });

    Ok(AlignMetrics {
        ready_t_s: ready.map(|v| v.0),
        ready_fwd_err_deg: ready.map(|v| v.1),
        ready_down_err_deg: ready.map(|v| v.2),
        final_fwd_err_deg,
        final_down_err_deg,
    })
}

fn run_c_align_replay(
    frames: &[UbxFrame],
    tl: &MasterTimeline,
    cfg: AlignConfig,
    bootstrap_cfg: BootstrapConfig,
) -> Result<AlignMetrics> {
    let nav_events = build_nav_events(frames, tl);
    let imu_packets = build_imu_packets(frames, tl);
    let final_alg_q = build_final_alg_q(frames, tl).context("missing final ESF-ALG")?;
    let ref_fwd = quat_rotate(final_alg_q, [1.0, 0.0, 0.0]);
    let ref_down = quat_rotate(final_alg_q, [0.0, 0.0, 1.0]);
    let ref_right = quat_rotate(final_alg_q, [0.0, 1.0, 0.0]);

    let mut align = CAlign::new(cfg);
    let mut bootstrap = ReplayBootstrapDetector::new(bootstrap_cfg);
    let mut align_initialized = false;
    let mut prev_ready = false;
    let mut ready_t_s = None;
    let mut ready_fwd_err_deg = None;
    let mut ready_down_err_deg = None;
    let mut scan_idx = 0usize;
    let mut interval_start_idx = 0usize;
    let mut prev_nav: Option<(f64, NavPvtObs)> = None;
    let mut final_q = None;

    for (tn, nav) in &nav_events {
        while scan_idx < imu_packets.len() && imu_packets[scan_idx].t_ms <= *tn {
            let pkt = &imu_packets[scan_idx];
            if !align_initialized {
                let gyro_radps = [
                    pkt.gx_dps.to_radians() as f32,
                    pkt.gy_dps.to_radians() as f32,
                    pkt.gz_dps.to_radians() as f32,
                ];
                let accel_b = [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32];
                let speed_mps = speed_for_bootstrap(prev_nav, (*tn, *nav), pkt.t_ms) as f32;
                if bootstrap.update(accel_b, gyro_radps, speed_mps)
                    && align.initialize_from_stationary(&bootstrap.stationary_accel, 0.0)
                {
                    align_initialized = true;
                }
            }
            scan_idx += 1;
        }

        if let Some((t_prev, nav_prev)) = prev_nav {
            let dt = ((*tn - t_prev) * 1.0e-3) as f32;
            let interval_packets = &imu_packets[interval_start_idx..scan_idx];
            if align_initialized && dt > 0.0 && !interval_packets.is_empty() {
                let mut gyro_sum = [0.0_f32; 3];
                let mut accel_sum = [0.0_f32; 3];
                for pkt in interval_packets {
                    gyro_sum[0] += pkt.gx_dps.to_radians() as f32;
                    gyro_sum[1] += pkt.gy_dps.to_radians() as f32;
                    gyro_sum[2] += pkt.gz_dps.to_radians() as f32;
                    accel_sum[0] += pkt.ax_mps2 as f32;
                    accel_sum[1] += pkt.ay_mps2 as f32;
                    accel_sum[2] += pkt.az_mps2 as f32;
                }
                let inv_n = 1.0 / interval_packets.len() as f32;
                let window = AlignWindowSummary {
                    dt,
                    mean_gyro_b: [
                        gyro_sum[0] * inv_n,
                        gyro_sum[1] * inv_n,
                        gyro_sum[2] * inv_n,
                    ],
                    mean_accel_b: [
                        accel_sum[0] * inv_n,
                        accel_sum[1] * inv_n,
                        accel_sum[2] * inv_n,
                    ],
                    gnss_vel_prev_n: [
                        nav_prev.vel_n_mps as f32,
                        nav_prev.vel_e_mps as f32,
                        nav_prev.vel_d_mps as f32,
                    ],
                    gnss_vel_curr_n: [
                        nav.vel_n_mps as f32,
                        nav.vel_e_mps as f32,
                        nav.vel_d_mps as f32,
                    ],
                };
                let (_score, trace) = align.update_window_with_trace(&window);
                let q_align = f32_quat_to_f64(align.state().q_vb);
                final_q = Some(q_align);

                if trace.coarse_alignment_ready && !prev_ready {
                    let align_fwd = quat_rotate(q_align, [1.0, 0.0, 0.0]);
                    let align_down = quat_rotate(q_align, [0.0, 0.0, 1.0]);
                    ready_t_s = Some((*tn - tl.t0_master_ms) * 1.0e-3);
                    ready_fwd_err_deg =
                        Some(signed_projected_axis_angle_deg(align_fwd, ref_fwd, ref_down).abs());
                    ready_down_err_deg =
                        Some(signed_projected_axis_angle_deg(align_down, ref_down, ref_right).abs());
                }
                prev_ready = trace.coarse_alignment_ready;
            }
        }
        prev_nav = Some((*tn, *nav));
        interval_start_idx = scan_idx;
    }

    let final_q = final_q.context("C align produced no final quaternion")?;
    let final_align_fwd = quat_rotate(final_q, [1.0, 0.0, 0.0]);
    let final_align_down = quat_rotate(final_q, [0.0, 0.0, 1.0]);

    Ok(AlignMetrics {
        ready_t_s,
        ready_fwd_err_deg,
        ready_down_err_deg,
        final_fwd_err_deg: signed_projected_axis_angle_deg(final_align_fwd, ref_fwd, ref_down)
            .abs(),
        final_down_err_deg: signed_projected_axis_angle_deg(
            final_align_down,
            ref_down,
            ref_right,
        )
        .abs(),
    })
}

fn compare_fusion_pair(
    frames: &[UbxFrame],
    tl: &MasterTimeline,
    cfg: FusionConfig,
) -> FusionParityPair {
    let nav_events = build_nav_events(frames, tl);
    let imu_packets = build_imu_packets(frames, tl);
    let ref_nav = nav_events.first().map(|(_, nav)| *nav);

    let mut rust = SensorFusion::new(cfg);
    let mut c = CSensorFusionWrapper::new_internal(cfg);
    let mut scan_idx = 0usize;
    let mut rust_metrics = FusionParityMetrics::default();
    let mut c_metrics = FusionParityMetrics::default();
    let mut diffs = Vec::new();

    for (tn, nav) in &nav_events {
        while scan_idx < imu_packets.len() && imu_packets[scan_idx].t_ms <= *tn {
            let pkt = &imu_packets[scan_idx];
            let sample = FusionImuSample {
                t_s: (pkt.t_ms - tl.t0_master_ms) * 1.0e-3,
                gyro_radps: [
                    pkt.gx_dps.to_radians() as f32,
                    pkt.gy_dps.to_radians() as f32,
                    pkt.gz_dps.to_radians() as f32,
                ],
                accel_mps2: [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32],
            };
            let ru = rust.process_imu(sample);
            let cu = c.process_imu(sample);
            if ru.mount_ready_changed && ru.mount_ready && rust_metrics.mount_ready_t_s.is_none() {
                rust_metrics.mount_ready_t_s = Some(sample.t_s);
            }
            if cu.mount_ready_changed && cu.mount_ready && c_metrics.mount_ready_t_s.is_none() {
                c_metrics.mount_ready_t_s = Some(sample.t_s);
            }
            if ru.ekf_initialized_now && rust_metrics.ekf_init_t_s.is_none() {
                rust_metrics.ekf_init_t_s = Some(sample.t_s);
            }
            if cu.ekf_initialized_now && c_metrics.ekf_init_t_s.is_none() {
                c_metrics.ekf_init_t_s = Some(sample.t_s);
            }
            scan_idx += 1;
        }

        let t_s = (*tn - tl.t0_master_ms) * 1.0e-3;
        let heading_rad = if nav.head_veh_valid {
            Some(nav.heading_vehicle_deg.to_radians() as f32)
        } else {
            Some(nav.heading_motion_deg.to_radians() as f32)
        };
        let gnss = FusionGnssSample {
            t_s,
            pos_ned_m: nav_to_ned(*nav, ref_nav),
            vel_ned_mps: [nav.vel_n_mps as f32, nav.vel_e_mps as f32, nav.vel_d_mps as f32],
            pos_std_m: [nav.h_acc_m as f32, nav.h_acc_m as f32, nav.v_acc_m as f32],
            vel_std_mps: [nav.s_acc_mps as f32, nav.s_acc_mps as f32, nav.s_acc_mps as f32],
            heading_rad,
        };
        let ru = rust.process_gnss(gnss);
        let cu = c.process_gnss(gnss);
        if ru.mount_ready_changed && ru.mount_ready && rust_metrics.mount_ready_t_s.is_none() {
            rust_metrics.mount_ready_t_s = Some(t_s);
        }
        if cu.mount_ready_changed && cu.mount_ready && c_metrics.mount_ready_t_s.is_none() {
            c_metrics.mount_ready_t_s = Some(t_s);
        }
        if ru.ekf_initialized_now && rust_metrics.ekf_init_t_s.is_none() {
            rust_metrics.ekf_init_t_s = Some(t_s);
        }
        if cu.ekf_initialized_now && c_metrics.ekf_init_t_s.is_none() {
            c_metrics.ekf_init_t_s = Some(t_s);
        }

        if let (Some(rekf), Some(cekf)) = (rust.ekf(), c.ekf()) {
            diffs.push(diff_sample(rekf, cekf));
        }
    }

    summarize_fusion_metrics(&mut rust_metrics, &mut c_metrics, &mut diffs, &rust, &c);
    FusionParityPair {
        rust: rust_metrics,
        c: c_metrics,
        diff: summarize_diff(&diffs),
    }
}

fn summarize_fusion_metrics(
    rust_metrics: &mut FusionParityMetrics,
    c_metrics: &mut FusionParityMetrics,
    diffs: &mut [FusionDiffSample],
    rust: &SensorFusion,
    c: &CSensorFusionWrapper,
) {
    if let Some(rekf) = rust.ekf() {
        rust_metrics.sample_count = diffs.len();
        rust_metrics.final_pos_m =
            norm3_f64([rekf.state.pn as f64, rekf.state.pe as f64, rekf.state.pd as f64]);
        rust_metrics.final_vel_mps =
            norm3_f64([rekf.state.vn as f64, rekf.state.ve as f64, rekf.state.vd as f64]);
    }
    if let Some(cekf) = c.ekf() {
        c_metrics.sample_count = diffs.len();
        c_metrics.final_pos_m =
            norm3_f64([cekf.state.pn as f64, cekf.state.pe as f64, cekf.state.pd as f64]);
        c_metrics.final_vel_mps =
            norm3_f64([cekf.state.vn as f64, cekf.state.ve as f64, cekf.state.vd as f64]);
    }
    let diff_summary = summarize_diff(diffs);
    *rust_metrics = FusionParityMetrics {
        sample_count: rust_metrics.sample_count,
        ..*rust_metrics
    };
    *c_metrics = FusionParityMetrics {
        sample_count: c_metrics.sample_count,
        ..*c_metrics
    };
    if let Some(rekf) = rust.ekf() {
        rust_metrics.final_att_deg = quat_angle_deg_from_ekf(rekf, rekf);
        rust_metrics.final_cov_diag = 0.0;
    }
    if let Some(cekf) = c.ekf() {
        c_metrics.final_att_deg = quat_angle_deg_from_ekf(cekf, cekf);
        c_metrics.final_cov_diag = 0.0;
    }
    let _ = diff_summary;
}

fn summarize_diff(diffs: &[FusionDiffSample]) -> FusionParityMetrics {
    if diffs.is_empty() {
        return FusionParityMetrics::default();
    }
    let mut sum_pos2 = 0.0;
    let mut sum_vel2 = 0.0;
    let mut sum_att2 = 0.0;
    let mut sum_cov2 = 0.0;
    for d in diffs {
        sum_pos2 += d.pos_m * d.pos_m;
        sum_vel2 += d.vel_mps * d.vel_mps;
        sum_att2 += d.att_deg * d.att_deg;
        sum_cov2 += d.cov_diag * d.cov_diag;
    }
    let n = diffs.len() as f64;
    let last = *diffs.last().unwrap_or(&FusionDiffSample::default());
    FusionParityMetrics {
        sample_count: diffs.len(),
        rms_pos_m: (sum_pos2 / n).sqrt(),
        rms_vel_mps: (sum_vel2 / n).sqrt(),
        rms_att_deg: (sum_att2 / n).sqrt(),
        rms_cov_diag: (sum_cov2 / n).sqrt(),
        final_pos_m: last.pos_m,
        final_vel_mps: last.vel_mps,
        final_att_deg: last.att_deg,
        final_cov_diag: last.cov_diag,
        ..FusionParityMetrics::default()
    }
}

fn diff_sample(rust: &sensor_fusion::ekf::Ekf, c: &sensor_fusion::ekf::Ekf) -> FusionDiffSample {
    let pos_m = norm3_f64([
        (rust.state.pn - c.state.pn) as f64,
        (rust.state.pe - c.state.pe) as f64,
        (rust.state.pd - c.state.pd) as f64,
    ]);
    let vel_mps = norm3_f64([
        (rust.state.vn - c.state.vn) as f64,
        (rust.state.ve - c.state.ve) as f64,
        (rust.state.vd - c.state.vd) as f64,
    ]);
    let att_deg = quat_angle_deg_from_ekf(rust, c);
    let mut sum_sq = 0.0;
    for i in 0..sensor_fusion::ekf::N_STATES {
        let d = (rust.p[i][i] - c.p[i][i]) as f64;
        sum_sq += d * d;
    }
    FusionDiffSample {
        pos_m,
        vel_mps,
        att_deg,
        cov_diag: (sum_sq / sensor_fusion::ekf::N_STATES as f64).sqrt(),
    }
}

fn quat_angle_deg_from_ekf(a: &sensor_fusion::ekf::Ekf, b: &sensor_fusion::ekf::Ekf) -> f64 {
    quat_angle_deg(
        [
            a.state.q0 as f64,
            a.state.q1 as f64,
            a.state.q2 as f64,
            a.state.q3 as f64,
        ],
        [
            b.state.q0 as f64,
            b.state.q1 as f64,
            b.state.q2 as f64,
            b.state.q3 as f64,
        ],
    )
}

fn quat_angle_deg(a: [f64; 4], b: [f64; 4]) -> f64 {
    let an = quat_normalize(a);
    let bn = quat_normalize(b);
    let dot = (an[0] * bn[0] + an[1] * bn[1] + an[2] * bn[2] + an[3] * bn[3]).abs();
    (2.0 * dot.clamp(-1.0, 1.0).acos()).to_degrees()
}

fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n <= 1.0e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

fn nav_to_ned(nav: NavPvtObs, ref_nav: Option<NavPvtObs>) -> [f32; 3] {
    let Some(ref_nav) = ref_nav else {
        return [0.0; 3];
    };
    let ref_ecef = lla_to_ecef(ref_nav.lat_deg, ref_nav.lon_deg, ref_nav.height_m);
    let ecef = lla_to_ecef(nav.lat_deg, nav.lon_deg, nav.height_m);
    let ned = ecef_to_ned(ecef, ref_ecef, ref_nav.lat_deg, ref_nav.lon_deg);
    [ned[0] as f32, ned[1] as f32, ned[2] as f32]
}

fn build_nav_events(frames: &[UbxFrame], tl: &MasterTimeline) -> Vec<(f64, NavPvtObs)> {
    let mut nav_events = Vec::<(f64, NavPvtObs)>::new();
    for f in frames {
        if let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters)
            && let Some(obs) = extract_nav2_pvt_obs(f)
            && obs.fix_ok
            && !obs.invalid_llh
        {
            nav_events.push((t_ms, obs));
        }
    }
    nav_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    nav_events
}

fn build_imu_packets(frames: &[UbxFrame], tl: &MasterTimeline) -> Vec<ImuPacket> {
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
    let (raw_tag_u, a_raw, b_raw) =
        fit_tag_ms_map_local(&raw_seq, &raw_tag, &tl.masters, Some(1 << 16));

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

fn build_final_alg_q(frames: &[UbxFrame], tl: &MasterTimeline) -> Option<[f64; 4]> {
    let mut last = None;
    for f in frames {
        if let Some((_, roll, pitch, yaw)) = extract_esf_alg(f)
            && nearest_master_ms(f.seq, &tl.masters).is_some()
        {
            last = Some(esf_alg_flu_to_frd_mount_quat(roll, pitch, yaw));
        }
    }
    last
}

#[derive(Clone)]
struct ReplayBootstrapDetector {
    cfg: BootstrapConfig,
    gyro_ema: Option<f32>,
    accel_err_ema: Option<f32>,
    speed_ema: Option<f32>,
    stationary_accel: Vec<[f32; 3]>,
}

impl ReplayBootstrapDetector {
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
        let gyro_norm = norm3(accel_to_f64(gyro_radps)) as f32;
        let accel_err = (norm3(accel_to_f64(accel_b)) as f32 - GRAVITY_MPS2).abs();
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

fn accel_to_f64(v: [f32; 3]) -> [f64; 3] {
    [v[0] as f64, v[1] as f64, v[2] as f64]
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn norm3_f64(v: [f64; 3]) -> f64 {
    norm3(v)
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

fn f32_quat_to_f64(q: [f32; 4]) -> [f64; 4] {
    [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64]
}

fn parse_section_files(path: &Path, wanted: &str) -> Result<Vec<String>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut current = String::new();
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(section) = line.strip_suffix(':') {
            current = section.trim().to_string();
            continue;
        }
        if current == wanted {
            let entry = line
                .split_once('(')
                .map(|(name, _)| name.trim())
                .unwrap_or(line);
            out.push(entry.to_string());
        }
    }
    Ok(out)
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.6}"))
        .unwrap_or_else(|| "nan".to_string())
}

fn opt_abs_diff(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    Some((a? - b?).abs())
}

fn fit_tag_ms_map_local(
    seqs: &[u64],
    tags: &[u64],
    masters: &[(u64, f64)],
    unwrap_modulus: Option<u64>,
) -> (Vec<u64>, f64, f64) {
    let mapped_tags = match unwrap_modulus {
        Some(m) => unwrap_counter(tags, m),
        None => tags.to_vec(),
    };
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for (seq, tag_u) in seqs.iter().zip(mapped_tags.iter()) {
        if let Some(ms) = nearest_master_ms(*seq, masters) {
            x.push(*tag_u as f64);
            y.push(ms);
        }
    }
    let (a, b) = fit_linear_map(&x, &y, 1e-3);
    (mapped_tags, a, b)
}
