use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::fusion::SensorFusion;
use sim::datasets::generic_replay::{
    fusion_gnss_sample as to_fusion_gnss, fusion_imu_sample as to_fusion_imu,
};
use sim::datasets::ubx_replay::{UbxReplayConfig, load_generic_replay};
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::eval::gnss_ins::{quat_angle_deg, quat_from_rpy_alg_deg};
use sim::ubxlog::{
    UbxFrame, extract_esf_alg, parse_ubx_frames,
};
use sim::visualizer::math::nearest_master_ms;
use sim::visualizer::pipeline::align_replay::esf_alg_flu_to_frd_mount_quat;
use sim::visualizer::pipeline::timebase::{MasterTimeline, build_master_timeline};

#[derive(Parser, Debug)]
#[command(name = "eval_real_mount_reseed")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long, default_value_t = 220.0)]
    start_time_s: f64,
    #[arg(long)]
    end_time_s: Option<f64>,
    #[arg(long, default_value_t = 1.0)]
    seed_roll_err_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    seed_pitch_err_deg: f64,
    #[arg(long, default_value_t = 1.0)]
    seed_yaw_err_deg: f64,
    #[arg(long, default_value_t = 150.0)]
    predict_imu_lpf_cutoff_hz: f64,
    #[arg(long, default_value_t = 1)]
    predict_imu_decimation: usize,
    #[arg(long, default_value_t = 0.1)]
    gnss_pos_r_scale: f64,
    #[arg(long, default_value_t = 3.0)]
    gnss_vel_r_scale: f64,
    #[arg(long, default_value_t = 30.0)]
    settle_threshold_deg: f64,
}

#[derive(Clone, Copy)]
struct AlgMountEvent {
    t_s: f64,
    q_vb: [f64; 4],
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

    let alg_events = collect_alg_mount_events(&frames, &tl);
    let (imu_samples, gnss_samples) = load_generic_replay(
        &args.logfile,
        UbxReplayConfig {
            gnss_pos_r_scale: args.gnss_pos_r_scale,
            gnss_vel_r_scale: args.gnss_vel_r_scale,
        },
    )?;
    if alg_events.is_empty() || gnss_samples.is_empty() || imu_samples.is_empty() {
        bail!("missing alg/nav/imu events");
    }

    let start_alg = sample_alg_mount(&alg_events, args.start_time_s)
        .context("no ESF-ALG mount at start time")?;
    let q_seed_err = quat_from_rpy_alg_deg(
        args.seed_roll_err_deg,
        args.seed_pitch_err_deg,
        args.seed_yaw_err_deg,
    );
    let q_seed = quat_mul(start_alg.q_vb, q_seed_err);
    let seed_err_deg = quat_angle_deg(q_seed, start_alg.q_vb);

    let mut fusion = SensorFusion::with_misalignment(q_seed.map(|v| v as f32));
    let end_time_s = args.end_time_s.unwrap_or(f64::INFINITY);
    let mut ekf_init_t_s = None::<f64>;
    let mut final_err_deg = f64::NAN;
    let mut best_err_deg = f64::INFINITY;
    let mut settle_time_s = None::<f64>;
    let mut samples = 0usize;

    for_each_event(&imu_samples, &gnss_samples, |event| match event {
        ReplayEvent::Gnss(_, sample) => {
            if sample.t_s >= args.start_time_s && sample.t_s <= end_time_s {
                let update = fusion.process_gnss(to_fusion_gnss(*sample));
                if update.ekf_initialized_now && ekf_init_t_s.is_none() {
                    ekf_init_t_s = Some(sample.t_s);
                }
                maybe_accumulate_error(
                    &fusion,
                    &alg_events,
                    sample.t_s,
                    ekf_init_t_s,
                    args.settle_threshold_deg,
                    &mut final_err_deg,
                    &mut best_err_deg,
                    &mut settle_time_s,
                    &mut samples,
                );
            }
        }
        ReplayEvent::Imu(_, sample) => {
            if sample.t_s < args.start_time_s || sample.t_s > end_time_s {
                return;
            }
            let _ = fusion.process_imu(to_fusion_imu(*sample));
            maybe_accumulate_error(
                &fusion,
                &alg_events,
                sample.t_s,
                ekf_init_t_s,
                args.settle_threshold_deg,
                &mut final_err_deg,
                &mut best_err_deg,
                &mut settle_time_s,
                &mut samples,
            );
        }
    });

    println!(
        "start_time_s={:.1} seed_err_deg={:.3} seed_rpy_err_deg=[{:.2},{:.2},{:.2}]",
        args.start_time_s,
        seed_err_deg,
        args.seed_roll_err_deg,
        args.seed_pitch_err_deg,
        args.seed_yaw_err_deg
    );
    println!(
        "ekf_init_t_s={} samples={} final_mount_quat_err_deg={:.3} best_mount_quat_err_deg={:.3} settle_time_s={}",
        ekf_init_t_s
            .map(|t| format!("{t:.3}"))
            .unwrap_or_else(|| "none".to_string()),
        samples,
        final_err_deg,
        best_err_deg,
        settle_time_s
            .map(|t| format!("{t:.3}"))
            .unwrap_or_else(|| "none".to_string())
    );

    Ok(())
}

fn maybe_accumulate_error(
    fusion: &SensorFusion,
    alg_events: &[AlgMountEvent],
    t_s: f64,
    ekf_init_t_s: Option<f64>,
    settle_threshold_deg: f64,
    final_err_deg: &mut f64,
    best_err_deg: &mut f64,
    settle_time_s: &mut Option<f64>,
    samples: &mut usize,
) {
    let Some(eskf) = fusion.eskf() else {
        return;
    };
    let Some(q_seed) = fusion
        .eskf_mount_q_vb()
        .or_else(|| fusion.mount_q_vb())
        .map(|q| q.map(|v| v as f64))
    else {
        return;
    };
    let Some(alg_q) = sample_alg_mount(alg_events, t_s) else {
        return;
    };
    let q_cs = [
        eskf.nominal.qcs0 as f64,
        eskf.nominal.qcs1 as f64,
        eskf.nominal.qcs2 as f64,
        eskf.nominal.qcs3 as f64,
    ];
    let q_full = quat_mul(q_seed, quat_conj(q_cs));
    let err_deg = quat_angle_deg(q_full, alg_q.q_vb);
    *final_err_deg = err_deg;
    *best_err_deg = best_err_deg.min(err_deg);
    *samples += 1;
    if settle_time_s.is_none()
        && err_deg <= settle_threshold_deg
        && let Some(t0) = ekf_init_t_s
    {
        *settle_time_s = Some(t_s - t0);
    }
}

fn rel_s(tl: &MasterTimeline, t_ms: f64) -> f64 {
    (t_ms - tl.masters.first().map(|(_, ms)| *ms).unwrap_or(t_ms)) * 1.0e-3
}

fn collect_alg_mount_events(frames: &[UbxFrame], tl: &MasterTimeline) -> Vec<AlgMountEvent> {
    let mut out = Vec::<AlgMountEvent>::new();
    for f in frames {
        if let Some((_, roll_deg, pitch_deg, yaw_deg)) = extract_esf_alg(f)
            && let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters)
        {
            out.push(AlgMountEvent {
                t_s: rel_s(tl, t_ms),
                q_vb: esf_alg_flu_to_frd_mount_quat(roll_deg, pitch_deg, yaw_deg),
            });
        }
    }
    out.sort_by(|a, b| a.t_s.partial_cmp(&b.t_s).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn sample_alg_mount(events: &[AlgMountEvent], t_s: f64) -> Option<AlgMountEvent> {
    if events.is_empty() {
        return None;
    }
    let idx = events.partition_point(|event| event.t_s < t_s);
    let left = events.get(idx.saturating_sub(1)).copied();
    let right = events.get(idx).copied();
    match (left, right) {
        (Some(left), Some(right)) => {
            if (right.t_s - t_s).abs() < (left.t_s - t_s).abs() {
                Some(right)
            } else {
                Some(left)
            }
        }
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let q = [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ];
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n <= 1e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}
