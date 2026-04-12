use std::cmp::Ordering;
use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;
use sim::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg_status, extract_nav2_pvt_obs, parse_ubx_frames,
};
use sim::visualizer::math::nearest_master_ms;
use sim::visualizer::pipeline::timebase::build_master_timeline;

#[derive(Parser, Debug)]
#[command(name = "analyze_esf_alg_transition_speeds")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
}

#[derive(Clone, Copy, Debug)]
struct StatusEvent {
    t_s: f64,
    status_code: u8,
    fine: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let data = std::fs::read(&args.logfile)?;
    let frames = parse_ubx_frames(&data, None);
    if frames.is_empty() {
        bail!("no UBX frames parsed");
    }
    let tl = build_master_timeline(&frames);
    if !tl.has_itow {
        bail!("log has no usable master timeline");
    }

    let mut status_events = Vec::<StatusEvent>::new();
    let mut nav_events = Vec::<(f64, NavPvtObs)>::new();

    for f in &frames {
        let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters) else {
            continue;
        };
        let Some(t_s) = tl.master_ms_to_rel_s(t_ms) else {
            continue;
        };
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

    status_events.sort_by(|a, b| a.t_s.partial_cmp(&b.t_s).unwrap_or(Ordering::Equal));
    nav_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    if status_events.is_empty() || nav_events.is_empty() {
        bail!("need ESF-ALG status and NAV2-PVT data");
    }

    let mut prev_status: Option<(u8, bool)> = None;
    for s in &status_events {
        let cur = (s.status_code, s.fine);
        if prev_status == Some(cur) {
            continue;
        }
        let nearest = nav_events.iter().min_by(|a, b| {
            (a.0 - s.t_s)
                .abs()
                .partial_cmp(&(b.0 - s.t_s).abs())
                .unwrap_or(Ordering::Equal)
        });
        if let Some((nav_t_s, nav)) = nearest {
            let speed_mps = nav.vel_n_mps.hypot(nav.vel_e_mps);
            println!(
                "t={:.3}s status={} fine={} nav_t={:.3}s speed={:.3} m/s ({:.1} km/h)",
                s.t_s,
                s.status_code,
                s.fine as u8,
                nav_t_s,
                speed_mps,
                speed_mps * 3.6
            );
        } else {
            println!(
                "t={:.3}s status={} fine={} nav_t=none speed=nan",
                s.t_s, s.status_code, s.fine as u8
            );
        }
        prev_status = Some(cur);
    }

    Ok(())
}
