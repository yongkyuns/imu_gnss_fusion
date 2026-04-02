use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::align::AlignConfig;
use sensor_fusion::ekf::PredictNoise;
use sim::ubxlog::parse_ubx_frames;
use sim::visualizer::model::{EkfImuSource, Trace};
use sim::visualizer::pipeline::align_replay::{
    BootstrapConfig, ImuReplayConfig, build_align_replay,
};
use sim::visualizer::pipeline::ekf_compare::{
    EkfCompareConfig, GnssOutageConfig, build_ekf_compare_traces,
};
use sim::visualizer::pipeline::timebase::build_master_timeline;

#[derive(Parser, Debug)]
#[command(name = "score_ekf_decimation")]
struct Args {
    #[arg(long, default_value = "test_files.txt")]
    file_list: PathBuf,

    #[arg(long, default_value = "Baseline")]
    section: String,

    #[arg(long, default_value = "logger/data")]
    data_dir: PathBuf,

    #[arg(long, default_value = "align", value_parser = parse_ekf_imu_source)]
    ekf_imu_source: EkfImuSource,

    #[arg(long, default_value = "3")]
    decimation_list: String,

    #[arg(long, default_value = "15")]
    predict_lpf_list: String,

    #[arg(long, default_value = "5")]
    r_body_vel_list: String,

    #[arg(long, default_value = "5")]
    vehicle_meas_lpf_list: String,

    #[arg(long, default_value = "0")]
    yaw_init_speed_mps_list: String,

    #[arg(long, default_value = "1")]
    gnss_pos_r_scale_list: String,

    #[arg(long, default_value = "1")]
    gnss_vel_r_scale_list: String,

    #[arg(long, default_value = "0.0001")]
    gyro_var_list: String,

    #[arg(long, default_value = "12")]
    accel_var_list: String,

    #[arg(long, default_value = "0.002e-9")]
    gyro_bias_rw_var_list: String,

    #[arg(long, default_value = "0.2e-9")]
    accel_bias_rw_var_list: String,

    #[arg(long, default_value_t = false)]
    per_log: bool,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    decimation: usize,
    predict_lpf_cutoff_hz: Option<f64>,
    r_body_vel: f32,
    vehicle_meas_lpf_cutoff_hz: f64,
    yaw_init_speed_mps: f64,
    gnss_pos_r_scale: f64,
    gnss_vel_r_scale: f64,
    predict_noise: PredictNoise,
}

#[derive(Clone, Copy, Debug, Default)]
struct TraceDiffMetrics {
    rms: f64,
    p95: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ErrorMetrics {
    pos_rms_m: f64,
    vel_rms_mps: f64,
    att_rms_deg: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct LogScore {
    ready_dt_s: f64,
    ekf_init_dt_s: f64,
    state_pos: TraceDiffMetrics,
    state_vel: TraceDiffMetrics,
    state_att: TraceDiffMetrics,
    bias_gyro: TraceDiffMetrics,
    bias_accel: TraceDiffMetrics,
    cov_bias_log: TraceDiffMetrics,
    cov_other_log: TraceDiffMetrics,
    acc_x_cross_dt_s: f64,
    acc_z_cross_dt_s: f64,
    ext_pos_reg_m: f64,
    ext_vel_reg_mps: f64,
    ext_att_reg_deg: f64,
    score: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct CandidateSummary {
    score_mean: f64,
    score_worst: f64,
    ready_dt_max_s: f64,
    ekf_init_dt_max_s: f64,
    state_pos_rms_max_m: f64,
    state_vel_rms_max_mps: f64,
    state_att_rms_max_deg: f64,
    bias_accel_rms_max: f64,
    cov_bias_log_rms_max: f64,
    ext_pos_reg_max_m: f64,
    ext_att_reg_max_deg: f64,
    gate_failures: usize,
    logs: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let files = parse_section_files(&args.file_list, &args.section)?;
    if files.is_empty() {
        bail!("no files found in section {}", args.section);
    }

    let decimations = parse_usize_list(&args.decimation_list)?;
    let predict_lpfs = parse_optional_f64_list(&args.predict_lpf_list)?;
    let r_body_vels = parse_f32_list(&args.r_body_vel_list)?;
    let vehicle_meas_lpfs = parse_f64_list(&args.vehicle_meas_lpf_list)?;
    let yaw_init_speeds = parse_f64_list(&args.yaw_init_speed_mps_list)?;
    let pos_scales = parse_f64_list(&args.gnss_pos_r_scale_list)?;
    let vel_scales = parse_f64_list(&args.gnss_vel_r_scale_list)?;
    let gyro_vars = parse_f32_list(&args.gyro_var_list)?;
    let accel_vars = parse_f32_list(&args.accel_var_list)?;
    let gyro_bias_rw_vars = parse_f32_list(&args.gyro_bias_rw_var_list)?;
    let accel_bias_rw_vars = parse_f32_list(&args.accel_bias_rw_var_list)?;

    let mut candidates = Vec::<Candidate>::new();
    for &decimation in &decimations {
        for &predict_lpf_cutoff_hz in &predict_lpfs {
            for &r_body_vel in &r_body_vels {
                for &vehicle_meas_lpf_cutoff_hz in &vehicle_meas_lpfs {
                    for &yaw_init_speed_mps in &yaw_init_speeds {
                        for &gnss_pos_r_scale in &pos_scales {
                            for &gnss_vel_r_scale in &vel_scales {
                                for &gyro_var in &gyro_vars {
                                    for &accel_var in &accel_vars {
                                        for &gyro_bias_rw_var in &gyro_bias_rw_vars {
                                            for &accel_bias_rw_var in &accel_bias_rw_vars {
                                                candidates.push(Candidate {
                                                    decimation: decimation.max(1),
                                                    predict_lpf_cutoff_hz,
                                                    r_body_vel,
                                                    vehicle_meas_lpf_cutoff_hz,
                                                    yaw_init_speed_mps,
                                                    gnss_pos_r_scale,
                                                    gnss_vel_r_scale,
                                                    predict_noise: PredictNoise {
                                                        gyro_var,
                                                        accel_var,
                                                        gyro_bias_rw_var,
                                                        accel_bias_rw_var,
                                                    },
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if candidates.is_empty() {
        bail!("no candidates generated");
    }

    let align_cfg = AlignConfig::default();
    let bootstrap_cfg = BootstrapConfig {
        ema_alpha: 0.05,
        max_speed_mps: 0.35,
        stationary_samples: 300,
        max_gyro_radps: align_cfg.max_stationary_gyro_radps,
        max_accel_norm_err_mps2: align_cfg.max_stationary_accel_norm_err_mps2,
    };

    println!(
        "candidate,decimation,predict_lpf_hz,r_body_vel,vehicle_meas_lpf_hz,yaw_init_speed_mps,gnss_pos_r_scale,gnss_vel_r_scale,gyro_var,accel_var,gyro_bias_rw_var,accel_bias_rw_var,score_mean,score_worst,ready_dt_max_s,ekf_init_dt_max_s,state_pos_rms_max_m,state_vel_rms_max_mps,state_att_rms_max_deg,bias_accel_rms_max,cov_bias_log_rms_max,ext_pos_reg_max_m,ext_att_reg_max_deg,gate_failures,logs"
    );

    if args.per_log {
        eprintln!(
            "per_log_columns: candidate,file,score,ready_dt_s,ekf_init_dt_s,state_pos_rms,state_pos_p95,state_vel_rms,state_att_rms,state_att_p95,bias_gyro_rms,bias_accel_rms,cov_bias_log_rms,cov_other_log_rms,acc_x_cross_dt_s,acc_z_cross_dt_s,ext_pos_reg_m,ext_vel_reg_mps,ext_att_reg_deg"
        );
    }

    for (idx, candidate) in candidates.iter().enumerate() {
        let label = format!("cand_{idx:03}");
        let mut scores = Vec::<LogScore>::new();

        for file in &files {
            let path = args.data_dir.join(file);
            let bytes =
                fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
            let frames = parse_ubx_frames(&bytes, None);
            let tl = build_master_timeline(&frames);

            let full_imu_cfg = ImuReplayConfig::default();
            let full_align =
                build_align_replay(&frames, &tl, align_cfg, bootstrap_cfg, full_imu_cfg);
            let cand_align =
                build_align_replay(&frames, &tl, align_cfg, bootstrap_cfg, full_imu_cfg);

            let full_ekf_cfg = EkfCompareConfig::default();
            let mut cand_ekf_cfg = full_ekf_cfg;
            cand_ekf_cfg.predict_imu_decimation = candidate.decimation;
            cand_ekf_cfg.predict_imu_lpf_cutoff_hz = candidate.predict_lpf_cutoff_hz;
            cand_ekf_cfg.r_body_vel = candidate.r_body_vel;
            cand_ekf_cfg.vehicle_meas_lpf_cutoff_hz = candidate.vehicle_meas_lpf_cutoff_hz;
            cand_ekf_cfg.yaw_init_speed_mps = candidate.yaw_init_speed_mps;
            cand_ekf_cfg.gnss_pos_r_scale = candidate.gnss_pos_r_scale;
            cand_ekf_cfg.gnss_vel_r_scale = candidate.gnss_vel_r_scale;
            cand_ekf_cfg.predict_noise = Some(candidate.predict_noise);

            let full = build_ekf_compare_traces(
                &frames,
                &tl,
                args.ekf_imu_source,
                full_ekf_cfg,
                GnssOutageConfig::default(),
            );
            let cand = build_ekf_compare_traces(
                &frames,
                &tl,
                args.ekf_imu_source,
                cand_ekf_cfg,
                GnssOutageConfig::default(),
            );

            let score = score_log(&full_align, &cand_align, &full, &cand)?;
            if args.per_log {
                eprintln!(
                    "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                    label,
                    file,
                    score.score,
                    score.ready_dt_s,
                    score.ekf_init_dt_s,
                    score.state_pos.rms,
                    score.state_pos.p95,
                    score.state_vel.rms,
                    score.state_att.rms,
                    score.state_att.p95,
                    score.bias_gyro.rms,
                    score.bias_accel.rms,
                    score.cov_bias_log.rms,
                    score.cov_other_log.rms,
                    score.acc_x_cross_dt_s,
                    score.acc_z_cross_dt_s,
                    score.ext_pos_reg_m,
                    score.ext_vel_reg_mps,
                    score.ext_att_reg_deg,
                );
            }
            scores.push(score);
        }

        let summary = summarize_scores(&scores);
        println!(
            "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.9},{:.6},{:.12},{:.12},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
            label,
            candidate.decimation,
            fmt_opt_f64(candidate.predict_lpf_cutoff_hz),
            candidate.r_body_vel,
            candidate.vehicle_meas_lpf_cutoff_hz,
            candidate.yaw_init_speed_mps,
            candidate.gnss_pos_r_scale,
            candidate.gnss_vel_r_scale,
            candidate.predict_noise.gyro_var,
            candidate.predict_noise.accel_var,
            candidate.predict_noise.gyro_bias_rw_var,
            candidate.predict_noise.accel_bias_rw_var,
            summary.score_mean,
            summary.score_worst,
            summary.ready_dt_max_s,
            summary.ekf_init_dt_max_s,
            summary.state_pos_rms_max_m,
            summary.state_vel_rms_max_mps,
            summary.state_att_rms_max_deg,
            summary.bias_accel_rms_max,
            summary.cov_bias_log_rms_max,
            summary.ext_pos_reg_max_m,
            summary.ext_att_reg_max_deg,
            summary.gate_failures,
            summary.logs,
        );
    }

    Ok(())
}

fn score_log(
    full_align: &sim::visualizer::pipeline::align_replay::AlignReplayData,
    cand_align: &sim::visualizer::pipeline::align_replay::AlignReplayData,
    full: &sim::visualizer::pipeline::ekf_compare::EkfCompareData,
    cand: &sim::visualizer::pipeline::ekf_compare::EkfCompareData,
) -> Result<LogScore> {
    let full_err = compute_error_metrics(full)?;
    let cand_err = compute_error_metrics(cand)?;

    let ready_dt_s = opt_abs_diff(first_yaw_init_t(full_align), first_yaw_init_t(cand_align));
    let ekf_init_dt_s = opt_abs_diff(
        first_trace_t(find_trace(&full.cmp_pos, "EKF posN [m]")?),
        first_trace_t(find_trace(&cand.cmp_pos, "EKF posN [m]")?),
    );

    let state_pos = diff_vec3_metrics(
        find_trace(&full.cmp_pos, "EKF posN [m]")?,
        find_trace(&full.cmp_pos, "EKF posE [m]")?,
        find_trace(&full.cmp_pos, "EKF posD [m]")?,
        find_trace(&cand.cmp_pos, "EKF posN [m]")?,
        find_trace(&cand.cmp_pos, "EKF posE [m]")?,
        find_trace(&cand.cmp_pos, "EKF posD [m]")?,
    );
    let state_vel = diff_vec3_metrics(
        find_trace(&full.cmp_vel, "EKF velN [m/s]")?,
        find_trace(&full.cmp_vel, "EKF velE [m/s]")?,
        find_trace(&full.cmp_vel, "EKF velD [m/s]")?,
        find_trace(&cand.cmp_vel, "EKF velN [m/s]")?,
        find_trace(&cand.cmp_vel, "EKF velE [m/s]")?,
        find_trace(&cand.cmp_vel, "EKF velD [m/s]")?,
    );
    let state_att = diff_vec3_metrics(
        find_trace(&full.cmp_att, "EKF roll [deg]")?,
        find_trace(&full.cmp_att, "EKF pitch [deg]")?,
        find_trace(&full.cmp_att, "EKF yaw [deg]")?,
        find_trace(&cand.cmp_att, "EKF roll [deg]")?,
        find_trace(&cand.cmp_att, "EKF pitch [deg]")?,
        find_trace(&cand.cmp_att, "EKF yaw [deg]")?,
    );
    let bias_gyro = diff_vec3_metrics(
        find_trace(&full.bias_gyro, "EKF gyro bias x [deg/s]")?,
        find_trace(&full.bias_gyro, "EKF gyro bias y [deg/s]")?,
        find_trace(&full.bias_gyro, "EKF gyro bias z [deg/s]")?,
        find_trace(&cand.bias_gyro, "EKF gyro bias x [deg/s]")?,
        find_trace(&cand.bias_gyro, "EKF gyro bias y [deg/s]")?,
        find_trace(&cand.bias_gyro, "EKF gyro bias z [deg/s]")?,
    );
    let bias_accel = diff_vec3_metrics(
        find_trace(&full.bias_accel, "EKF accel bias x [m/s^2]")?,
        find_trace(&full.bias_accel, "EKF accel bias y [m/s^2]")?,
        find_trace(&full.bias_accel, "EKF accel bias z [m/s^2]")?,
        find_trace(&cand.bias_accel, "EKF accel bias x [m/s^2]")?,
        find_trace(&cand.bias_accel, "EKF accel bias y [m/s^2]")?,
        find_trace(&cand.bias_accel, "EKF accel bias z [m/s^2]")?,
    );
    let cov_bias_log = diff_log_cov_metrics(&full.cov_bias, &cand.cov_bias)?;
    let cov_other_log = diff_log_cov_metrics(&full.cov_nonbias, &cand.cov_nonbias)?;

    let acc_x_cross_dt_s = opt_abs_diff(
        first_below(find_trace(&full.cov_bias, "acc_x")?, 1.0e-7),
        first_below(find_trace(&cand.cov_bias, "acc_x")?, 1.0e-7),
    );
    let acc_z_cross_dt_s = opt_abs_diff(
        first_below(find_trace(&full.cov_bias, "acc_z")?, 1.0e-7),
        first_below(find_trace(&cand.cov_bias, "acc_z")?, 1.0e-7),
    );

    let ext_pos_reg_m = (cand_err.pos_rms_m - full_err.pos_rms_m).max(0.0);
    let ext_vel_reg_mps = (cand_err.vel_rms_mps - full_err.vel_rms_mps).max(0.0);
    let ext_att_reg_deg = (cand_err.att_rms_deg - full_err.att_rms_deg).max(0.0);

    let score = 3.0 * norm(state_att.rms, 1.0)
        + 2.0 * norm(state_att.p95, 2.0)
        + 2.0 * norm(state_pos.rms, 0.5)
        + 1.5 * norm(state_vel.rms, 0.1)
        + 1.0 * norm(bias_gyro.rms, 0.1)
        + 1.0 * norm(bias_accel.rms, 0.05)
        + 2.0 * norm(cov_bias_log.rms, 0.5)
        + 1.0 * norm(cov_other_log.rms, 0.5)
        + 2.0 * norm(ready_dt_s, 5.0)
        + 2.0 * norm(ekf_init_dt_s, 5.0)
        + 2.0 * norm(0.5 * (acc_x_cross_dt_s + acc_z_cross_dt_s), 20.0)
        + 2.0 * norm(ext_pos_reg_m, 0.5)
        + 1.0 * norm(ext_vel_reg_mps, 0.1)
        + 1.5 * norm(ext_att_reg_deg, 1.0);

    Ok(LogScore {
        ready_dt_s,
        ekf_init_dt_s,
        state_pos,
        state_vel,
        state_att,
        bias_gyro,
        bias_accel,
        cov_bias_log,
        cov_other_log,
        acc_x_cross_dt_s,
        acc_z_cross_dt_s,
        ext_pos_reg_m,
        ext_vel_reg_mps,
        ext_att_reg_deg,
        score,
    })
}

fn summarize_scores(scores: &[LogScore]) -> CandidateSummary {
    let mut out = CandidateSummary::default();
    out.logs = scores.len();
    if scores.is_empty() {
        return out;
    }
    for s in scores {
        out.score_mean += s.score;
        out.score_worst = out.score_worst.max(s.score);
        out.ready_dt_max_s = out.ready_dt_max_s.max(s.ready_dt_s);
        out.ekf_init_dt_max_s = out.ekf_init_dt_max_s.max(s.ekf_init_dt_s);
        out.state_pos_rms_max_m = out.state_pos_rms_max_m.max(s.state_pos.rms);
        out.state_vel_rms_max_mps = out.state_vel_rms_max_mps.max(s.state_vel.rms);
        out.state_att_rms_max_deg = out.state_att_rms_max_deg.max(s.state_att.rms);
        out.bias_accel_rms_max = out.bias_accel_rms_max.max(s.bias_accel.rms);
        out.cov_bias_log_rms_max = out.cov_bias_log_rms_max.max(s.cov_bias_log.rms);
        out.ext_pos_reg_max_m = out.ext_pos_reg_max_m.max(s.ext_pos_reg_m);
        out.ext_att_reg_max_deg = out.ext_att_reg_max_deg.max(s.ext_att_reg_deg);
        if s.state_att.rms > 5.0
            || s.state_pos.rms > 0.5
            || s.ready_dt_s > 5.0
            || s.ekf_init_dt_s > 5.0
            || s.acc_x_cross_dt_s > 20.0
            || s.acc_z_cross_dt_s > 20.0
            || s.ext_pos_reg_m > 0.5
        {
            out.gate_failures += 1;
        }
    }
    out.score_mean /= scores.len() as f64;
    out
}

fn compute_error_metrics(
    data: &sim::visualizer::pipeline::ekf_compare::EkfCompareData,
) -> Result<ErrorMetrics> {
    let ekf_pn = find_trace(&data.cmp_pos, "EKF posN [m]")?;
    let ekf_pe = find_trace(&data.cmp_pos, "EKF posE [m]")?;
    let ekf_pd = find_trace(&data.cmp_pos, "EKF posD [m]")?;
    let ubx_pn = find_trace(&data.cmp_pos, "UBX posN [m]")?;
    let ubx_pe = find_trace(&data.cmp_pos, "UBX posE [m]")?;
    let ubx_pd = find_trace(&data.cmp_pos, "UBX posD [m]")?;
    let ekf_vn = find_trace(&data.cmp_vel, "EKF velN [m/s]")?;
    let ekf_ve = find_trace(&data.cmp_vel, "EKF velE [m/s]")?;
    let ekf_vd = find_trace(&data.cmp_vel, "EKF velD [m/s]")?;
    let ubx_vn = find_trace(&data.cmp_vel, "UBX velN [m/s]")?;
    let ubx_ve = find_trace(&data.cmp_vel, "UBX velE [m/s]")?;
    let ubx_vd = find_trace(&data.cmp_vel, "UBX velD [m/s]")?;
    let ekf_roll = find_trace(&data.cmp_att, "EKF roll [deg]")?;
    let ekf_pitch = find_trace(&data.cmp_att, "EKF pitch [deg]")?;
    let ekf_yaw = find_trace(&data.cmp_att, "EKF yaw [deg]")?;
    let nav_roll = find_trace(&data.cmp_att, "NAV-ATT roll [deg]")?;
    let nav_pitch = find_trace(&data.cmp_att, "NAV-ATT pitch [deg]")?;
    let nav_yaw = find_trace(&data.cmp_att, "NAV-ATT heading [deg]")?;

    let mut n = 0usize;
    let mut sum_pos2 = 0.0;
    let mut sum_vel2 = 0.0;
    let mut sum_att2 = 0.0;
    for p in ekf_pn {
        let t = p[0];
        let Some(epn) = interp(ekf_pn, t) else {
            continue;
        };
        let Some(epe) = interp(ekf_pe, t) else {
            continue;
        };
        let Some(epd) = interp(ekf_pd, t) else {
            continue;
        };
        let Some(upn) = interp(ubx_pn, t) else {
            continue;
        };
        let Some(upe) = interp(ubx_pe, t) else {
            continue;
        };
        let Some(upd) = interp(ubx_pd, t) else {
            continue;
        };
        let Some(evn) = interp(ekf_vn, t) else {
            continue;
        };
        let Some(eve) = interp(ekf_ve, t) else {
            continue;
        };
        let Some(evd) = interp(ekf_vd, t) else {
            continue;
        };
        let Some(uvn) = interp(ubx_vn, t) else {
            continue;
        };
        let Some(uve) = interp(ubx_ve, t) else {
            continue;
        };
        let Some(uvd) = interp(ubx_vd, t) else {
            continue;
        };
        let Some(er) = interp(ekf_roll, t) else {
            continue;
        };
        let Some(ep) = interp(ekf_pitch, t) else {
            continue;
        };
        let Some(ey) = interp(ekf_yaw, t) else {
            continue;
        };
        let Some(nr) = interp(nav_roll, t) else {
            continue;
        };
        let Some(np) = interp(nav_pitch, t) else {
            continue;
        };
        let Some(ny) = interp(nav_yaw, t) else {
            continue;
        };
        sum_pos2 += (epn - upn).powi(2) + (epe - upe).powi(2) + (epd - upd).powi(2);
        sum_vel2 += (evn - uvn).powi(2) + (eve - uve).powi(2) + (evd - uvd).powi(2);
        sum_att2 += (er - nr).powi(2) + (ep - np).powi(2) + ang_diff_deg(ey, ny).powi(2);
        n += 1;
    }
    if n == 0 {
        bail!("no overlapping error samples");
    }
    Ok(ErrorMetrics {
        pos_rms_m: (sum_pos2 / n as f64).sqrt(),
        vel_rms_mps: (sum_vel2 / n as f64).sqrt(),
        att_rms_deg: (sum_att2 / n as f64).sqrt(),
    })
}

fn diff_vec3_metrics(
    ax: &[[f64; 2]],
    ay: &[[f64; 2]],
    az: &[[f64; 2]],
    bx: &[[f64; 2]],
    by: &[[f64; 2]],
    bz: &[[f64; 2]],
) -> TraceDiffMetrics {
    let mut errs = Vec::new();
    for p in ax {
        let t = p[0];
        let (Some(axv), Some(ayv), Some(azv), Some(bxv), Some(byv), Some(bzv)) = (
            interp(ax, t),
            interp(ay, t),
            interp(az, t),
            interp(bx, t),
            interp(by, t),
            interp(bz, t),
        ) else {
            continue;
        };
        errs.push(((axv - bxv).powi(2) + (ayv - byv).powi(2) + (azv - bzv).powi(2)).sqrt());
    }
    metrics_from_errors(&errs)
}

fn diff_log_cov_metrics(a: &[Trace], b: &[Trace]) -> Result<TraceDiffMetrics> {
    let mut errs = Vec::new();
    for ta in a {
        let tb = find_trace(b, &ta.name)?;
        for p in &ta.points {
            let t = p[0];
            let Some(av) = interp(&ta.points, t) else {
                continue;
            };
            let Some(bv) = interp(tb, t) else { continue };
            let av = av.max(1.0e-20);
            let bv = bv.max(1.0e-20);
            errs.push((av.ln() - bv.ln()).abs());
        }
    }
    Ok(metrics_from_errors(&errs))
}

fn metrics_from_errors(errs: &[f64]) -> TraceDiffMetrics {
    if errs.is_empty() {
        return TraceDiffMetrics::default();
    }
    let mut sorted = errs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rms = (errs.iter().map(|e| e * e).sum::<f64>() / errs.len() as f64).sqrt();
    let idx95 = ((sorted.len() - 1) as f64 * 0.95).round() as usize;
    TraceDiffMetrics {
        rms,
        p95: sorted[idx95],
    }
}

fn find_trace<'a>(traces: &'a [Trace], name: &str) -> Result<&'a [[f64; 2]]> {
    traces
        .iter()
        .find(|t| t.name == name)
        .map(|t| t.points.as_slice())
        .with_context(|| format!("missing trace {name}"))
}

fn interp(points: &[[f64; 2]], t: f64) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    let first = points.first()?;
    let last = points.last()?;
    if t < first[0] || t > last[0] {
        return None;
    }
    match points.binary_search_by(|p| p[0].partial_cmp(&t).unwrap_or(std::cmp::Ordering::Less)) {
        Ok(i) => Some(points[i][1]),
        Err(0) => Some(points[0][1]),
        Err(i) if i >= points.len() => Some(points[points.len() - 1][1]),
        Err(i) => {
            let p0 = points[i - 1];
            let p1 = points[i];
            let dt = p1[0] - p0[0];
            if dt.abs() <= 1.0e-12 {
                Some(p1[1])
            } else {
                let a = (t - p0[0]) / dt;
                Some(p0[1] + a * (p1[1] - p0[1]))
            }
        }
    }
}

fn first_trace_t(trace: &[[f64; 2]]) -> Option<f64> {
    trace.first().map(|p| p[0])
}

fn first_yaw_init_t(
    replay: &sim::visualizer::pipeline::align_replay::AlignReplayData,
) -> Option<f64> {
    replay
        .samples
        .iter()
        .find(|s| s.yaw_initialized)
        .map(|s| s.t_s)
}

fn first_below(trace: &[[f64; 2]], threshold: f64) -> Option<f64> {
    trace.iter().find(|p| p[1] < threshold).map(|p| p[0])
}

fn opt_abs_diff(a: Option<f64>, b: Option<f64>) -> f64 {
    match (a, b) {
        (Some(a), Some(b)) => (a - b).abs(),
        (None, None) => 0.0,
        _ => 1.0e9,
    }
}

fn norm(value: f64, scale: f64) -> f64 {
    value / scale.max(1.0e-12)
}

fn ang_diff_deg(a: f64, b: f64) -> f64 {
    let mut d = a - b;
    while d > 180.0 {
        d -= 360.0;
    }
    while d < -180.0 {
        d += 360.0;
    }
    d
}

fn parse_section_files(path: &Path, section: &str) -> Result<Vec<String>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read file list {}", path.display()))?;
    let mut out = Vec::new();
    let mut in_section = false;
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if let Some(name) = l.strip_suffix(':') {
            in_section = name.trim() == section;
            continue;
        }
        if l.starts_with('[') && l.ends_with(']') {
            in_section = &l[1..l.len() - 1] == section;
            continue;
        }
        if l.contains("cargo run ") || l.starts_with("--") {
            continue;
        }
        if in_section {
            let file = l.split_once(" (").map(|(head, _)| head).unwrap_or(l).trim();
            if file.ends_with(".bin") {
                out.push(file.to_string());
            }
        }
    }
    Ok(out)
}

fn parse_ekf_imu_source(s: &str) -> Result<EkfImuSource, String> {
    match s {
        "align" => Ok(EkfImuSource::Align),
        "esf-alg" => Ok(EkfImuSource::EsfAlg),
        other => Err(format!("invalid ekf imu source: {other}")),
    }
}

fn parse_usize_list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|v| {
            v.trim()
                .parse::<usize>()
                .with_context(|| format!("invalid usize value {v}"))
        })
        .collect()
}

fn parse_f32_list(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|v| {
            v.trim()
                .parse::<f32>()
                .with_context(|| format!("invalid f32 value {v}"))
        })
        .collect()
}

fn parse_f64_list(s: &str) -> Result<Vec<f64>> {
    s.split(',')
        .map(|v| {
            v.trim()
                .parse::<f64>()
                .with_context(|| format!("invalid f64 value {v}"))
        })
        .collect()
}

fn parse_optional_f64_list(s: &str) -> Result<Vec<Option<f64>>> {
    s.split(',')
        .map(|v| {
            let t = v.trim();
            if t.eq_ignore_ascii_case("none") {
                Ok(None)
            } else {
                t.parse::<f64>()
                    .map(Some)
                    .with_context(|| format!("invalid f64/none value {v}"))
            }
        })
        .collect()
}

fn fmt_opt_f64(v: Option<f64>) -> String {
    match v {
        Some(v) => format!("{v:.3}"),
        None => "none".to_string(),
    }
}
