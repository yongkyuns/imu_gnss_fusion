use anyhow::{Result, bail};
use clap::Parser;
use sensor_fusion::c_api::CSensorFusionWrapper;
use sim::datasets::generic_replay::{fusion_gnss_sample as to_fusion_gnss, fusion_imu_sample as to_fusion_imu};
use sim::datasets::ubx_replay::{UbxReplayConfig, load_generic_replay};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    input: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let data = fs::read(&args.input)?;
    let frames = sim::ubxlog::parse_ubx_frames(&data, None);
    if frames.is_empty() {
        bail!("no UBX frames found");
    }
    let (imu_samples, gnss_samples) = load_generic_replay(&args.input, UbxReplayConfig::default())?;
    if imu_samples.is_empty() || gnss_samples.is_empty() {
        bail!("need both IMU and GNSS events");
    }

    let mut fusion = CSensorFusionWrapper::new_internal();
    let mut imu_idx = 0usize;
    let mut gnss_idx = 0usize;
    let mut prev_reanchor_count = 0u32;
    let mut events = Vec::new();

    while imu_idx < imu_samples.len() || gnss_idx < gnss_samples.len() {
        let next_imu_s = imu_samples.get(imu_idx).map(|pkt| pkt.t_s);
        let next_gnss_s = gnss_samples.get(gnss_idx).map(|pkt| pkt.t_s);
        let take_imu = match (next_imu_s, next_gnss_s) {
            (Some(t_imu), Some(t_gnss)) => t_imu <= t_gnss,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };

        if take_imu {
            let pkt = imu_samples[imu_idx];
            let _ = fusion.process_imu(to_fusion_imu(pkt));
            imu_idx += 1;
        } else {
            let obs = gnss_samples[gnss_idx];
            let _ = fusion.process_gnss(to_fusion_gnss(obs));

            let reanchor_count = fusion.reanchor_count();
            if reanchor_count != prev_reanchor_count {
                let info = fusion.last_reanchor_info();
                let anchor = fusion.anchor_lla_debug();
                events.push((reanchor_count, obs.t_s as f32, info, anchor));
                prev_reanchor_count = reanchor_count;
            }
            gnss_idx += 1;
        }
    }

    println!("input: {}", args.input.display());
    println!("reanchor_count={}", fusion.reanchor_count());
    for (count, t_s, info, anchor) in events {
        if let Some((reported_t_s, distance_m)) = info {
            if let Some(anchor_lla) = anchor {
                println!(
                    "reanchor #{}: loop_t_s={:.3} debug_t_s={:.3} distance_m={:.1} anchor=({:.7},{:.7},{:.2})",
                    count,
                    t_s,
                    reported_t_s,
                    distance_m,
                    anchor_lla[0],
                    anchor_lla[1],
                    anchor_lla[2]
                );
            } else {
                println!(
                    "reanchor #{}: loop_t_s={:.3} debug_t_s={:.3} distance_m={:.1}",
                    count, t_s, reported_t_s, distance_m
                );
            }
        } else {
            println!("reanchor #{}: loop_t_s={:.3}", count, t_s);
        }
    }

    Ok(())
}
