use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::fusion::SensorFusion;
use sim::datasets::generic_replay::{
    GenericGnssSample, fusion_gnss_sample as to_fusion_gnss, fusion_imu_sample as to_fusion_imu,
};
use sim::datasets::ubx_replay::{UbxReplayConfig, load_generic_replay};
use sim::eval::gnss_ins::quat_angle_deg;
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::ubxlog::{UbxFrame, extract_esf_alg, parse_ubx_frames};
use sim::visualizer::math::nearest_master_ms;
use sim::visualizer::pipeline::align_replay::esf_alg_flu_to_frd_mount_quat;
use sim::visualizer::pipeline::timebase::{MasterTimeline, build_master_timeline};

#[derive(Parser, Debug)]
#[command(name = "sweep_real_gnss_time_shift")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "-200,-100,-50,0,50,100,200"
    )]
    shifts_ms: Vec<f64>,
    #[arg(long, default_value_t = 0.1)]
    gnss_pos_r_scale: f64,
    #[arg(long, default_value_t = 3.0)]
    gnss_vel_r_scale: f64,
}

#[derive(Clone, Copy)]
struct AlgMountEvent {
    t_s: f64,
    q_vb: [f64; 4],
}

#[derive(Clone, Copy, Debug)]
struct SweepResult {
    shift_ms: f64,
    mount_ready_t_s: Option<f64>,
    ekf_init_t_s: Option<f64>,
    samples: usize,
    final_mount_quat_err_deg: f64,
    best_mount_quat_err_deg: f64,
    body_vel_y_innov_abs: f64,
    body_vel_y_yaw_dx_abs_deg: f64,
    body_vel_z_innov_abs: f64,
    body_vel_z_yaw_dx_abs_deg: f64,
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
            ..UbxReplayConfig::default()
        },
    )?;
    if alg_events.is_empty() || imu_samples.is_empty() || gnss_samples.is_empty() {
        bail!("missing replay inputs");
    }

    println!(
        "log={} imu_samples={} gnss_samples={} shifts_ms={:?}",
        args.logfile.display(),
        imu_samples.len(),
        gnss_samples.len(),
        args.shifts_ms
    );
    println!(
        "shift_ms,mount_ready_t_s,ekf_init_t_s,samples,final_mount_quat_err_deg,best_mount_quat_err_deg,body_vel_y_innov_abs,body_vel_y_yaw_dx_abs_deg,body_vel_z_innov_abs,body_vel_z_yaw_dx_abs_deg"
    );

    let mut results = Vec::<SweepResult>::new();
    for shift_ms in args.shifts_ms.iter().copied() {
        let shifted = shift_gnss_samples(&gnss_samples, shift_ms);
        let result = run_shift(shift_ms, &imu_samples, &shifted, &alg_events);
        println!(
            "{:.1},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            result.shift_ms,
            fmt_opt(result.mount_ready_t_s),
            fmt_opt(result.ekf_init_t_s),
            result.samples,
            result.final_mount_quat_err_deg,
            result.best_mount_quat_err_deg,
            result.body_vel_y_innov_abs,
            result.body_vel_y_yaw_dx_abs_deg,
            result.body_vel_z_innov_abs,
            result.body_vel_z_yaw_dx_abs_deg
        );
        results.push(result);
    }

    if let Some(best_final) = results.iter().min_by(|a, b| {
        a.final_mount_quat_err_deg
            .total_cmp(&b.final_mount_quat_err_deg)
    }) {
        println!(
            "best_final_mount: shift_ms={:.1} final_mount_quat_err_deg={:.6}",
            best_final.shift_ms, best_final.final_mount_quat_err_deg
        );
    }
    if let Some(best_body_y) = results
        .iter()
        .min_by(|a, b| a.body_vel_y_innov_abs.total_cmp(&b.body_vel_y_innov_abs))
    {
        println!(
            "best_body_vel_y: shift_ms={:.1} body_vel_y_innov_abs={:.6}",
            best_body_y.shift_ms, best_body_y.body_vel_y_innov_abs
        );
    }

    Ok(())
}

fn shift_gnss_samples(samples: &[GenericGnssSample], shift_ms: f64) -> Vec<GenericGnssSample> {
    let dt_s = shift_ms * 1.0e-3;
    let mut out = Vec::with_capacity(samples.len());
    for sample in samples {
        let shifted_t = sample.t_s + dt_s;
        if shifted_t.is_finite() && shifted_t >= 0.0 {
            let mut shifted = *sample;
            shifted.t_s = shifted_t;
            out.push(shifted);
        }
    }
    out.sort_by(|a, b| a.t_s.total_cmp(&b.t_s));
    out
}

fn run_shift(
    shift_ms: f64,
    imu_samples: &[sim::datasets::generic_replay::GenericImuSample],
    gnss_samples: &[GenericGnssSample],
    alg_events: &[AlgMountEvent],
) -> SweepResult {
    let mut fusion = SensorFusion::new();
    let mut mount_ready_t_s = None::<f64>;
    let mut ekf_init_t_s = None::<f64>;
    let mut final_mount_quat_err_deg = f64::NAN;
    let mut best_mount_quat_err_deg = f64::INFINITY;
    let mut samples = 0usize;

    for_each_event(imu_samples, gnss_samples, |event| match event {
        ReplayEvent::Imu(_, sample) => {
            let update = fusion.process_imu(to_fusion_imu(*sample));
            if update.mount_ready_changed && update.mount_ready && mount_ready_t_s.is_none() {
                mount_ready_t_s = Some(sample.t_s);
            }
            if update.ekf_initialized_now && ekf_init_t_s.is_none() {
                ekf_init_t_s = Some(sample.t_s);
            }
            maybe_accumulate_mount_error(
                &fusion,
                alg_events,
                sample.t_s,
                &mut final_mount_quat_err_deg,
                &mut best_mount_quat_err_deg,
                &mut samples,
            );
        }
        ReplayEvent::Gnss(_, sample) => {
            let update = fusion.process_gnss(to_fusion_gnss(*sample));
            if update.mount_ready_changed && update.mount_ready && mount_ready_t_s.is_none() {
                mount_ready_t_s = Some(sample.t_s);
            }
            if update.ekf_initialized_now && ekf_init_t_s.is_none() {
                ekf_init_t_s = Some(sample.t_s);
            }
            maybe_accumulate_mount_error(
                &fusion,
                alg_events,
                sample.t_s,
                &mut final_mount_quat_err_deg,
                &mut best_mount_quat_err_deg,
                &mut samples,
            );
        }
    });

    let (
        body_vel_y_innov_abs,
        body_vel_y_yaw_dx_abs_deg,
        body_vel_z_innov_abs,
        body_vel_z_yaw_dx_abs_deg,
    ) = fusion
        .eskf()
        .map(|eskf| {
            (
                eskf.update_diag.sum_abs_innovation[4] as f64,
                eskf.update_diag.sum_abs_dx_mount_yaw[4] as f64 * 180.0 / std::f64::consts::PI,
                eskf.update_diag.sum_abs_innovation[5] as f64,
                eskf.update_diag.sum_abs_dx_mount_yaw[5] as f64 * 180.0 / std::f64::consts::PI,
            )
        })
        .unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN));

    SweepResult {
        shift_ms,
        mount_ready_t_s,
        ekf_init_t_s,
        samples,
        final_mount_quat_err_deg,
        best_mount_quat_err_deg,
        body_vel_y_innov_abs,
        body_vel_y_yaw_dx_abs_deg,
        body_vel_z_innov_abs,
        body_vel_z_yaw_dx_abs_deg,
    }
}

fn maybe_accumulate_mount_error(
    fusion: &SensorFusion,
    alg_events: &[AlgMountEvent],
    t_s: f64,
    final_err_deg: &mut f64,
    best_err_deg: &mut f64,
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
    out.sort_by(|a, b| a.t_s.total_cmp(&b.t_s));
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

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.3}"))
        .unwrap_or_else(|| "none".to_string())
}
