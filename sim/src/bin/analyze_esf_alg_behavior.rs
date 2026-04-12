use std::cmp::Ordering;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use sim::ubxlog::{
    NavPvtObs, extract_esf_alg, extract_esf_alg_status, extract_nav2_pvt_obs, parse_ubx_frames,
};
use sim::visualizer::math::{nearest_master_ms, normalize_heading_deg};
use sim::visualizer::pipeline::timebase::build_master_timeline;

#[derive(Parser, Debug)]
#[command(name = "analyze_esf_alg_behavior")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,

    #[arg(long, default_value_t = 60.0)]
    init_lookback_s: f64,

    #[arg(long, default_value_t = 60.0)]
    settle_tail_s: f64,
}

#[derive(Clone, Copy, Debug)]
struct AlgEvent {
    t_s: f64,
    roll_deg: f64,
    pitch_deg: f64,
    yaw_deg: f64,
}

#[derive(Clone, Copy, Debug)]
struct StatusEvent {
    t_s: f64,
    status_code: u8,
    fine: bool,
}

#[derive(Clone, Copy, Debug)]
struct MotionWindow {
    t0_s: f64,
    t1_s: f64,
    tc_s: f64,
    speed_mid_mps: f64,
    course_rate_dps: f64,
    a_long_mps2: f64,
    a_lat_mps2: f64,
    turn_valid: bool,
    straight_valid: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct ReductionStats {
    samples: usize,
    total_roll_reduction_deg: f64,
    total_pitch_reduction_deg: f64,
    mean_abs_roll_step_deg: f64,
    mean_abs_pitch_step_deg: f64,
}

#[derive(Clone, Copy, Debug)]
struct ConvergenceStep {
    t_s: f64,
    dt_s: f64,
    roll_err_prev_deg: f64,
    roll_err_curr_deg: f64,
    pitch_err_prev_deg: f64,
    pitch_err_curr_deg: f64,
    roll_reduction_deg: f64,
    pitch_reduction_deg: f64,
    roll_step_deg: f64,
    pitch_step_deg: f64,
    motion: MotionWindow,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let data = std::fs::read(&args.logfile)
        .with_context(|| format!("failed to read {}", args.logfile.display()))?;
    let frames = parse_ubx_frames(&data, None);
    if frames.is_empty() {
        bail!("no UBX frames parsed");
    }
    let tl = build_master_timeline(&frames);
    if !tl.has_itow {
        bail!("log has no usable master timeline");
    }

    let mut alg_events = Vec::<AlgEvent>::new();
    let mut status_events = Vec::<StatusEvent>::new();
    let mut nav_events = Vec::<(f64, NavPvtObs)>::new();

    for f in &frames {
        let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters) else {
            continue;
        };
        let Some(t_s) = tl.master_ms_to_rel_s(t_ms) else {
            continue;
        };
        if let Some((_itow, roll, pitch, yaw)) = extract_esf_alg(f) {
            alg_events.push(AlgEvent {
                t_s,
                roll_deg: roll,
                pitch_deg: pitch,
                yaw_deg: normalize_heading_deg(yaw),
            });
        }
        if let Some((_itow, status_code, is_fine)) = extract_esf_alg_status(f) {
            status_events.push(StatusEvent {
                t_s,
                status_code: status_code as u8,
                fine: is_fine > 0.5,
            });
        }
        if let Some(obs) = extract_nav2_pvt_obs(f)
            && obs.fix_ok
            && !obs.invalid_llh
        {
            nav_events.push((t_s, obs));
        }
    }

    alg_events.sort_by(|a, b| a.t_s.partial_cmp(&b.t_s).unwrap_or(Ordering::Equal));
    status_events.sort_by(|a, b| a.t_s.partial_cmp(&b.t_s).unwrap_or(Ordering::Equal));
    nav_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    if alg_events.is_empty() || status_events.is_empty() || nav_events.len() < 2 {
        bail!("need ESF-ALG, ESF-ALG status, and NAV2-PVT data");
    }

    let motion_windows = build_motion_windows(&nav_events);
    let Some(first_coarse) = status_events.iter().find(|s| s.status_code >= 3).copied() else {
        bail!("no ESF-ALG status >= 3 found");
    };
    let first_fine = status_events.iter().find(|s| s.fine).copied();

    let ref_start_s = alg_events.last().unwrap().t_s - args.settle_tail_s;
    let settled_roll = median(
        &alg_events
            .iter()
            .filter(|e| e.t_s >= ref_start_s)
            .map(|e| e.roll_deg)
            .collect::<Vec<_>>(),
    );
    let settled_pitch = median(
        &alg_events
            .iter()
            .filter(|e| e.t_s >= ref_start_s)
            .map(|e| e.pitch_deg)
            .collect::<Vec<_>>(),
    );
    let settled_yaw = circular_median_deg(
        &alg_events
            .iter()
            .filter(|e| e.t_s >= ref_start_s)
            .map(|e| e.yaw_deg)
            .collect::<Vec<_>>(),
    );

    println!("file={}", args.logfile.display());
    println!(
        "first_coarse t={:.2}s status={} | first_fine={}",
        first_coarse.t_s,
        first_coarse.status_code,
        first_fine
            .map(|s| format!("{:.2}s", s.t_s))
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "settled_ref last_{:.0}s roll={:.3} pitch={:.3} yaw={:.3}",
        args.settle_tail_s, settled_roll, settled_pitch, settled_yaw
    );

    println!("\nstatus transitions:");
    let mut prev_status: Option<(u8, bool)> = None;
    let mut changed_events = Vec::<StatusEvent>::new();
    for s in &status_events {
        let cur = (s.status_code, s.fine);
        if prev_status != Some(cur) {
            println!(
                "  t={:8.3}s status={} fine={}",
                s.t_s, s.status_code, s.fine as u8
            );
            changed_events.push(*s);
            prev_status = Some(cur);
        }
    }

    println!("\nphase summaries:");
    let transitions: Vec<_> = changed_events
        .iter()
        .copied()
        .scan(None, |prev, curr| {
            let out = prev.map(|p| (p, curr));
            *prev = Some(curr);
            Some(out)
        })
        .flatten()
        .collect();
    for (a, b) in transitions {
        if a.status_code == b.status_code && a.fine == b.fine {
            continue;
        }
        let phase_windows: Vec<_> = motion_windows
            .iter()
            .copied()
            .filter(|w| w.tc_s >= a.t_s && w.tc_s <= b.t_s)
            .collect();
        println!(
            "  status {} fine={} -> status {} fine={} over {:.2}s",
            a.status_code,
            a.fine as u8,
            b.status_code,
            b.fine as u8,
            b.t_s - a.t_s
        );
        print_motion_summary(&phase_windows);
    }

    println!(
        "\npre-init motion summary ({:.0}s lookback to first coarse):",
        args.init_lookback_s
    );
    let pre_init_windows: Vec<_> = motion_windows
        .iter()
        .copied()
        .filter(|w| w.tc_s >= first_coarse.t_s - args.init_lookback_s && w.tc_s <= first_coarse.t_s)
        .collect();
    print_motion_summary(&pre_init_windows);
    print_top_windows("pre-init strongest turn", &pre_init_windows, 5, |w| {
        if w.turn_valid {
            w.a_lat_mps2.abs() + 0.2 * w.course_rate_dps.abs()
        } else {
            f64::NEG_INFINITY
        }
    });
    print_top_windows("pre-init strongest straight", &pre_init_windows, 5, |w| {
        if w.straight_valid {
            w.a_long_mps2.abs()
        } else {
            f64::NEG_INFINITY
        }
    });

    let steps = build_convergence_steps(
        &alg_events,
        &motion_windows,
        settled_roll,
        settled_pitch,
        first_coarse.t_s,
    );
    let overall = summarize_reduction(&steps);
    let turn = summarize_reduction(
        &steps
            .iter()
            .copied()
            .filter(|s| s.motion.turn_valid)
            .collect::<Vec<_>>(),
    );
    let straight = summarize_reduction(
        &steps
            .iter()
            .copied()
            .filter(|s| s.motion.straight_valid)
            .collect::<Vec<_>>(),
    );
    let other = summarize_reduction(
        &steps
            .iter()
            .copied()
            .filter(|s| !s.motion.turn_valid && !s.motion.straight_valid)
            .collect::<Vec<_>>(),
    );

    println!("\npost-init reduction summary:");
    print_reduction_summary("overall", overall);
    print_reduction_summary("turn", turn);
    print_reduction_summary("straight", straight);
    print_reduction_summary("other", other);

    let mut by_roll = steps.clone();
    by_roll.sort_by(|a, b| {
        b.roll_reduction_deg
            .partial_cmp(&a.roll_reduction_deg)
            .unwrap_or(Ordering::Equal)
    });
    println!("\nlargest roll-converging steps:");
    for s in by_roll.iter().take(8) {
        print_step(s);
    }

    let mut by_pitch = steps.clone();
    by_pitch.sort_by(|a, b| {
        b.pitch_reduction_deg
            .partial_cmp(&a.pitch_reduction_deg)
            .unwrap_or(Ordering::Equal)
    });
    println!("\nlargest pitch-converging steps:");
    for s in by_pitch.iter().take(8) {
        print_step(s);
    }

    Ok(())
}

fn build_motion_windows(nav_events: &[(f64, NavPvtObs)]) -> Vec<MotionWindow> {
    let mut out = Vec::new();
    for pair in nav_events.windows(2) {
        let (t0_s, prev) = pair[0];
        let (t1_s, curr) = pair[1];
        let dt = t1_s - t0_s;
        if dt <= 1.0e-6 {
            continue;
        }
        let v_prev = [prev.vel_n_mps, prev.vel_e_mps];
        let v_curr = [curr.vel_n_mps, curr.vel_e_mps];
        let speed_prev = (v_prev[0] * v_prev[0] + v_prev[1] * v_prev[1]).sqrt();
        let speed_curr = (v_curr[0] * v_curr[0] + v_curr[1] * v_curr[1]).sqrt();
        let speed_mid = 0.5 * (speed_prev + speed_curr);
        let course_prev = v_prev[1].atan2(v_prev[0]);
        let course_curr = v_curr[1].atan2(v_curr[0]);
        let course_rate_dps = wrap_rad_pi(course_curr - course_prev).to_degrees() / dt;
        let a_n = [(v_curr[0] - v_prev[0]) / dt, (v_curr[1] - v_prev[1]) / dt];
        let v_mid = [0.5 * (v_prev[0] + v_curr[0]), 0.5 * (v_prev[1] + v_curr[1])];
        let (a_long, a_lat) = if let Some(t_hat) = normalize2(v_mid) {
            let lat_hat = [-t_hat[1], t_hat[0]];
            (
                t_hat[0] * a_n[0] + t_hat[1] * a_n[1],
                lat_hat[0] * a_n[0] + lat_hat[1] * a_n[1],
            )
        } else {
            (0.0, 0.0)
        };
        let turn_valid =
            speed_mid > (0.833_f64) && course_rate_dps.abs() > 2.0 && a_lat.abs() > 0.1;
        let straight_valid = speed_mid > (0.833_f64)
            && a_long.abs() > 0.18
            && a_lat.abs() < (0.5_f64).max(0.6 * a_long.abs())
            && (a_long * a_long + a_lat * a_lat).sqrt() > 0.18;
        out.push(MotionWindow {
            t0_s,
            t1_s,
            tc_s: 0.5 * (t0_s + t1_s),
            speed_mid_mps: speed_mid,
            course_rate_dps,
            a_long_mps2: a_long,
            a_lat_mps2: a_lat,
            turn_valid,
            straight_valid,
        });
    }
    out
}

fn build_convergence_steps(
    alg_events: &[AlgEvent],
    motion_windows: &[MotionWindow],
    settled_roll: f64,
    settled_pitch: f64,
    start_t_s: f64,
) -> Vec<ConvergenceStep> {
    let mut out = Vec::new();
    for pair in alg_events.windows(2) {
        let prev = pair[0];
        let curr = pair[1];
        if curr.t_s < start_t_s {
            continue;
        }
        let Some(motion) = nearest_motion_window(motion_windows, curr.t_s) else {
            continue;
        };
        let roll_err_prev = (prev.roll_deg - settled_roll).abs();
        let roll_err_curr = (curr.roll_deg - settled_roll).abs();
        let pitch_err_prev = (prev.pitch_deg - settled_pitch).abs();
        let pitch_err_curr = (curr.pitch_deg - settled_pitch).abs();
        out.push(ConvergenceStep {
            t_s: curr.t_s,
            dt_s: curr.t_s - prev.t_s,
            roll_err_prev_deg: roll_err_prev,
            roll_err_curr_deg: roll_err_curr,
            pitch_err_prev_deg: pitch_err_prev,
            pitch_err_curr_deg: pitch_err_curr,
            roll_reduction_deg: (roll_err_prev - roll_err_curr).max(0.0),
            pitch_reduction_deg: (pitch_err_prev - pitch_err_curr).max(0.0),
            roll_step_deg: curr.roll_deg - prev.roll_deg,
            pitch_step_deg: curr.pitch_deg - prev.pitch_deg,
            motion,
        });
    }
    out
}

fn nearest_motion_window(motion_windows: &[MotionWindow], t_s: f64) -> Option<MotionWindow> {
    motion_windows.iter().copied().min_by(|a, b| {
        (a.tc_s - t_s)
            .abs()
            .partial_cmp(&(b.tc_s - t_s).abs())
            .unwrap_or(Ordering::Equal)
    })
}

fn summarize_reduction(steps: &[ConvergenceStep]) -> ReductionStats {
    if steps.is_empty() {
        return ReductionStats::default();
    }
    ReductionStats {
        samples: steps.len(),
        total_roll_reduction_deg: steps.iter().map(|s| s.roll_reduction_deg).sum(),
        total_pitch_reduction_deg: steps.iter().map(|s| s.pitch_reduction_deg).sum(),
        mean_abs_roll_step_deg: steps.iter().map(|s| s.roll_step_deg.abs()).sum::<f64>()
            / steps.len() as f64,
        mean_abs_pitch_step_deg: steps.iter().map(|s| s.pitch_step_deg.abs()).sum::<f64>()
            / steps.len() as f64,
    }
}

fn print_reduction_summary(label: &str, stats: ReductionStats) {
    println!(
        "  {:>8}: n={} total_roll_reduction={:.3} total_pitch_reduction={:.3} mean_abs_step roll={:.4} pitch={:.4}",
        label,
        stats.samples,
        stats.total_roll_reduction_deg,
        stats.total_pitch_reduction_deg,
        stats.mean_abs_roll_step_deg,
        stats.mean_abs_pitch_step_deg
    );
}

fn print_step(s: &ConvergenceStep) {
    println!(
        "  t={:8.3}s dt={:.3}s roll_err {:.3}->{:.3} pitch_err {:.3}->{:.3} | droll={:+.4} dpitch={:+.4} | speed={:.2} cr={:+.2} a_long={:+.3} a_lat={:+.3} turn={} straight={}",
        s.t_s,
        s.dt_s,
        s.roll_err_prev_deg,
        s.roll_err_curr_deg,
        s.pitch_err_prev_deg,
        s.pitch_err_curr_deg,
        s.roll_step_deg,
        s.pitch_step_deg,
        s.motion.speed_mid_mps,
        s.motion.course_rate_dps,
        s.motion.a_long_mps2,
        s.motion.a_lat_mps2,
        s.motion.turn_valid as u8,
        s.motion.straight_valid as u8
    );
}

fn print_motion_summary(windows: &[MotionWindow]) {
    if windows.is_empty() {
        println!("  no motion windows");
        return;
    }
    let n_turn = windows.iter().filter(|w| w.turn_valid).count();
    let n_straight = windows.iter().filter(|w| w.straight_valid).count();
    let max_abs_course_rate = windows
        .iter()
        .map(|w| w.course_rate_dps.abs())
        .fold(0.0, f64::max);
    let max_abs_lat = windows
        .iter()
        .map(|w| w.a_lat_mps2.abs())
        .fold(0.0, f64::max);
    let max_abs_long = windows
        .iter()
        .map(|w| w.a_long_mps2.abs())
        .fold(0.0, f64::max);
    let first_turn = windows.iter().find(|w| w.turn_valid).map(|w| w.tc_s);
    let first_straight = windows.iter().find(|w| w.straight_valid).map(|w| w.tc_s);
    println!(
        "  windows={} turn_valid={} straight_valid={} max|course_rate|={:.2} dps max|a_lat|={:.3} max|a_long|={:.3}",
        windows.len(),
        n_turn,
        n_straight,
        max_abs_course_rate,
        max_abs_lat,
        max_abs_long
    );
    println!(
        "  first turn={} first straight={}",
        first_turn
            .map(|v| format!("{:.3}s", v))
            .unwrap_or_else(|| "none".to_string()),
        first_straight
            .map(|v| format!("{:.3}s", v))
            .unwrap_or_else(|| "none".to_string())
    );
}

fn print_top_windows<F>(label: &str, windows: &[MotionWindow], k: usize, score_fn: F)
where
    F: Fn(&MotionWindow) -> f64,
{
    let mut ranked: Vec<_> = windows
        .iter()
        .copied()
        .filter_map(|w| {
            let score = score_fn(&w);
            if score.is_finite() {
                Some((score, w))
            } else {
                None
            }
        })
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    println!("  {}:", label);
    for (_score, w) in ranked.into_iter().take(k) {
        println!(
            "    t={:8.3}s speed={:.2} cr={:+.2} a_long={:+.3} a_lat={:+.3}",
            w.tc_s, w.speed_mid_mps, w.course_rate_dps, w.a_long_mps2, w.a_lat_mps2
        );
    }
}

fn normalize2(v: [f64; 2]) -> Option<[f64; 2]> {
    let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if n <= 1.0e-9 {
        None
    } else {
        Some([v[0] / n, v[1] / n])
    }
}

fn wrap_rad_pi(mut x: f64) -> f64 {
    while x > std::f64::consts::PI {
        x -= 2.0 * std::f64::consts::PI;
    }
    while x < -std::f64::consts::PI {
        x += 2.0 * std::f64::consts::PI;
    }
    x
}

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    v[v.len() / 2]
}

fn circular_median_deg(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut best = xs[0];
    let mut best_cost = f64::INFINITY;
    for &cand in xs {
        let cost = xs
            .iter()
            .map(|&x| normalize_heading_delta_deg(x - cand).abs())
            .sum::<f64>();
        if cost < best_cost {
            best_cost = cost;
            best = cand;
        }
    }
    normalize_heading_deg(best)
}

fn normalize_heading_delta_deg(mut deg: f64) -> f64 {
    while deg > 180.0 {
        deg -= 360.0;
    }
    while deg < -180.0 {
        deg += 360.0;
    }
    deg
}
