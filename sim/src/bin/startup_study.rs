use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use align_rs::align::AlignConfig;
use anyhow::{Context, Result, bail};
use clap::Parser;
use sim::ubxlog::parse_ubx_frames;
use sim::visualizer::pipeline::align_replay::{BootstrapConfig, build_align_replay, quat_rotate};
use sim::visualizer::pipeline::timebase::build_master_timeline;

#[derive(Parser, Debug)]
#[command(name = "startup_study")]
struct Args {
    #[arg(long, default_value = "test_files.txt")]
    file_list: PathBuf,

    #[arg(long, default_value = "Baseline")]
    section: String,

    #[arg(long, default_value = "logger/data")]
    data_dir: PathBuf,

    #[arg(long, default_value = "/tmp/startup_study_baseline.csv")]
    out_csv: PathBuf,

    #[arg(long)]
    max_records: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct Summary {
    windows: usize,
    startup_accepted: usize,
    startup_emitted: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let files = parse_section_files(&args.file_list, &args.section)?;
    if files.is_empty() {
        bail!(
            "no files found in section '{}' of {}",
            args.section,
            args.file_list.display()
        );
    }

    if let Some(parent) = args.out_csv.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let fout = File::create(&args.out_csv)
        .with_context(|| format!("failed to create {}", args.out_csv.display()))?;
    let mut w = BufWriter::new(fout);
    writeln!(
        w,
        "logfile,t_s,speed_kmh,course_rate_dps,a_long_mps2,a_lat_mps2,gnss_accel_norm_mps2,gnss_accel_angle_deg,\
imu_leveled_angle_deg,startup_gate_valid,startup_accepted,startup_alignment_score,startup_emitted,startup_theta_deg,startup_theta_alt_deg,\
long_emitted,long_angle_err_deg,yaw_initialized,\
imu_startup_theta_angle_deg,imu_startup_theta_alt_angle_deg,imu_final_align_angle_deg,imu_final_esf_angle_deg,\
err_startup_theta_deg,err_startup_theta_alt_deg,err_final_align_deg,err_final_esf_deg,\
final_fwd_err_align_deg,final_down_err_align_deg"
    )?;

    let cfg = AlignConfig::default();
    let bootstrap_cfg = BootstrapConfig {
        ema_alpha: 0.05,
        max_speed_mps: 0.35,
        stationary_samples: 300,
        max_gyro_radps: cfg.max_stationary_gyro_radps,
        max_accel_norm_err_mps2: cfg.max_stationary_accel_norm_err_mps2,
    };

    let mut total = Summary {
        windows: 0,
        startup_accepted: 0,
        startup_emitted: 0,
    };

    for file in &files {
        let log_path = args.data_dir.join(file);
        let data = fs::read(&log_path)
            .with_context(|| format!("failed to read {}", log_path.display()))?;
        let frames = parse_ubx_frames(&data, args.max_records);
        let tl = build_master_timeline(&frames);
        let replay = build_align_replay(&frames, &tl, cfg, bootstrap_cfg);
        let final_align_q = replay.samples.last().map(|s| s.q_align);
        let final_alg_q = replay.final_alg_q;
        let startup_theta = replay
            .samples
            .iter()
            .find_map(|s| s.startup_trace.emitted_theta_rad);

        let (final_fwd_err_align_deg, final_down_err_align_deg) = replay
            .samples
            .last()
            .map(|s| final_axis_err_against(s, final_alg_q))
            .unwrap_or((f64::NAN, f64::NAN));

        let mut summary = Summary {
            windows: 0,
            startup_accepted: 0,
            startup_emitted: 0,
        };

        for sample in &replay.samples {
            summary.windows += 1;
            total.windows += 1;
            if sample.startup_trace.accepted {
                summary.startup_accepted += 1;
                total.startup_accepted += 1;
            }
            if sample.startup_trace.emitted {
                summary.startup_emitted += 1;
                total.startup_emitted += 1;
            }

            let g_long = sample.a_long_mps2;
            let g_lat = sample.a_lat_mps2;
            let gnss_angle_deg = wrap_signed_deg(g_lat.atan2(g_long).to_degrees());
            let gnss_accel_norm = (g_long * g_long + g_lat * g_lat).sqrt();
            let imu_leveled_angle_deg = wrap_signed_deg(
                sample
                    .startup_input_lat_mps2
                    .atan2(sample.startup_input_long_mps2)
                    .to_degrees(),
            );

            let imu_startup_theta_angle_deg = startup_theta
                .map(|theta| {
                    rotate_angle(
                        sample.startup_input_long_mps2,
                        sample.startup_input_lat_mps2,
                        theta,
                    )
                })
                .unwrap_or(f64::NAN);
            let imu_startup_theta_alt_angle_deg = startup_theta
                .map(|theta| {
                    rotate_angle(
                        sample.startup_input_long_mps2,
                        sample.startup_input_lat_mps2,
                        theta + std::f64::consts::PI,
                    )
                })
                .unwrap_or(f64::NAN);
            let imu_final_align_angle_deg = final_align_q
                .map(|q| rotate_body_horiz_with_mount(sample.horiz_accel_b, q))
                .unwrap_or(f64::NAN);
            let imu_final_esf_angle_deg = final_alg_q
                .map(|q| rotate_body_horiz_with_mount(sample.horiz_accel_b, q))
                .unwrap_or(f64::NAN);

            let err_startup_theta_deg = if startup_theta.is_some() {
                wrap_signed_deg(imu_startup_theta_angle_deg - gnss_angle_deg)
            } else {
                f64::NAN
            };
            let err_startup_theta_alt_deg = if startup_theta.is_some() {
                wrap_signed_deg(imu_startup_theta_alt_angle_deg - gnss_angle_deg)
            } else {
                f64::NAN
            };
            let err_final_align_deg = if final_align_q.is_some() {
                wrap_signed_deg(imu_final_align_angle_deg - gnss_angle_deg)
            } else {
                f64::NAN
            };
            let err_final_esf_deg = if final_alg_q.is_some() {
                wrap_signed_deg(imu_final_esf_angle_deg - gnss_angle_deg)
            } else {
                f64::NAN
            };

            writeln!(
                w,
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{:.6},{},{:.6},{:.6},{},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                file,
                sample.t_s,
                sample.speed_mps * 3.6,
                sample.course_rate_dps,
                sample.a_long_mps2,
                sample.a_lat_mps2,
                gnss_accel_norm,
                gnss_angle_deg,
                imu_leveled_angle_deg,
                bool01(sample.startup_trace.gate_valid),
                bool01(sample.startup_trace.accepted),
                sample.startup_trace.alignment_score,
                bool01(sample.startup_trace.emitted),
                startup_theta.map(|v| wrap_signed_deg(v.to_degrees())).unwrap_or(f64::NAN),
                startup_theta
                    .map(|v| wrap_signed_deg((v + std::f64::consts::PI).to_degrees()))
                    .unwrap_or(f64::NAN),
                bool01(sample.long_trace.emitted),
                sample.long_trace.angle_err_deg,
                bool01(sample.yaw_initialized),
                imu_startup_theta_angle_deg,
                imu_startup_theta_alt_angle_deg,
                imu_final_align_angle_deg,
                imu_final_esf_angle_deg,
                err_startup_theta_deg,
                err_startup_theta_alt_deg,
                err_final_align_deg,
                err_final_esf_deg,
                final_fwd_err_align_deg,
                final_down_err_align_deg,
            )?;
        }
        eprintln!(
            "{}: windows={} startup_accepted={} startup_emitted={}",
            file, summary.windows, summary.startup_accepted, summary.startup_emitted
        );
    }

    w.flush()?;
    eprintln!(
        "wrote {} rows to {}",
        total.windows,
        args.out_csv.display()
    );
    Ok(())
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

fn rotate_angle(x: f64, y: f64, theta: f64) -> f64 {
    let c = theta.cos();
    let s = theta.sin();
    let xr = c * x - s * y;
    let yr = s * x + c * y;
    wrap_signed_deg(yr.atan2(xr).to_degrees())
}

fn rotate_body_horiz_with_mount(horiz_accel_b: [f64; 3], q_mount: [f64; 4]) -> f64 {
    let q_conj = [q_mount[0], -q_mount[1], -q_mount[2], -q_mount[3]];
    let accel_v = quat_rotate(q_conj, horiz_accel_b);
    wrap_signed_deg(accel_v[1].atan2(accel_v[0]).to_degrees())
}

fn final_axis_err_against(
    sample: &sim::visualizer::pipeline::align_replay::AlignReplaySample,
    final_alg_q: Option<[f64; 4]>,
) -> (f64, f64) {
    let Some(q_ref) = final_alg_q else {
        return (f64::NAN, f64::NAN);
    };
    let align_fwd = quat_rotate(sample.q_align, [1.0, 0.0, 0.0]);
    let align_down = quat_rotate(sample.q_align, [0.0, 0.0, 1.0]);
    let ref_fwd = quat_rotate(q_ref, [1.0, 0.0, 0.0]);
    let ref_down = quat_rotate(q_ref, [0.0, 0.0, 1.0]);
    let ref_right = quat_rotate(q_ref, [0.0, 1.0, 0.0]);
    (
        sim::visualizer::pipeline::align_replay::signed_projected_axis_angle_deg(
            align_fwd, ref_fwd, ref_down,
        ),
        sim::visualizer::pipeline::align_replay::signed_projected_axis_angle_deg(
            align_down, ref_down, ref_right,
        ),
    )
}

fn wrap_signed_deg(x: f64) -> f64 {
    (x + 180.0).rem_euclid(360.0) - 180.0
}

fn bool01(v: bool) -> u8 {
    if v { 1 } else { 0 }
}
