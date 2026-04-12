use anyhow::{Result, bail};
use clap::Parser;
use sensor_fusion::c_api::CSensorFusionWrapper;
use sensor_fusion::fusion::{FusionGnssSample, FusionImuSample};
use sim::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_raw_samples, extract_nav_pvt_obs, extract_nav2_pvt_obs,
    fit_linear_map, parse_ubx_frames, sensor_meta,
};
use sim::visualizer::math::nearest_master_ms;
use sim::visualizer::model::ImuPacket;
use sim::visualizer::pipeline::timebase::{MasterTimeline, build_master_timeline};
use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    input: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let data = fs::read(&args.input)?;
    let frames = parse_ubx_frames(&data, None);
    if frames.is_empty() {
        bail!("no UBX frames found");
    }

    let timeline = build_master_timeline(&frames);
    if !timeline.has_itow {
        bail!("log does not contain a usable iTOW timeline");
    }

    let imu_packets = build_imu_packets(&frames, &timeline)?;
    let nav_events = collect_nav_events(&frames, &timeline);
    if imu_packets.is_empty() || nav_events.is_empty() {
        bail!("need both IMU and GNSS events");
    }

    let mut fusion = CSensorFusionWrapper::new_internal();
    let mut imu_idx = 0usize;
    let mut gnss_idx = 0usize;
    let mut prev_reanchor_count = 0u32;
    let mut events = Vec::new();

    while imu_idx < imu_packets.len() || gnss_idx < nav_events.len() {
        let next_imu_ms = imu_packets.get(imu_idx).map(|pkt| pkt.t_ms);
        let next_gnss_ms = nav_events.get(gnss_idx).map(|(t_ms, _)| *t_ms);
        let take_imu = match (next_imu_ms, next_gnss_ms) {
            (Some(t_imu), Some(t_gnss)) => t_imu <= t_gnss,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };

        if take_imu {
            let pkt = imu_packets[imu_idx];
            let Some(t_s) = timeline.master_ms_to_rel_s(pkt.t_ms).map(|v| v as f32) else {
                imu_idx += 1;
                continue;
            };
            let _ = fusion.process_imu(FusionImuSample {
                t_s,
                gyro_radps: [
                    pkt.gx_dps.to_radians() as f32,
                    pkt.gy_dps.to_radians() as f32,
                    pkt.gz_dps.to_radians() as f32,
                ],
                accel_mps2: [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32],
            });
            imu_idx += 1;
        } else {
            let (t_ms, obs) = nav_events[gnss_idx];
            let Some(t_s) = timeline.master_ms_to_rel_s(t_ms).map(|v| v as f32) else {
                gnss_idx += 1;
                continue;
            };
            let _ = fusion.process_gnss(FusionGnssSample {
                t_s,
                lat_deg: obs.lat_deg as f32,
                lon_deg: obs.lon_deg as f32,
                height_m: obs.height_m as f32,
                vel_ned_mps: [
                    obs.vel_n_mps as f32,
                    obs.vel_e_mps as f32,
                    obs.vel_d_mps as f32,
                ],
                pos_std_m: [obs.h_acc_m as f32, obs.h_acc_m as f32, obs.v_acc_m as f32],
                vel_std_mps: [
                    obs.s_acc_mps as f32,
                    obs.s_acc_mps as f32,
                    obs.s_acc_mps as f32,
                ],
                heading_rad: obs
                    .head_veh_valid
                    .then_some((obs.heading_vehicle_deg as f32).to_radians()),
            });

            let reanchor_count = fusion.reanchor_count();
            if reanchor_count != prev_reanchor_count {
                let info = fusion.last_reanchor_info();
                let anchor = fusion.anchor_lla_debug();
                events.push((reanchor_count, t_s, info, anchor));
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

fn build_imu_packets(frames: &[UbxFrame], timeline: &MasterTimeline) -> Result<Vec<ImuPacket>> {
    let mut raw_seq = Vec::<u64>::new();
    let mut raw_tag = Vec::<u64>::new();
    let mut raw_dtype = Vec::<u8>::new();
    let mut raw_val = Vec::<f64>::new();
    for frame in frames {
        for (tag, sw) in extract_esf_raw_samples(frame) {
            let (_, _, scale) = sensor_meta(sw.dtype);
            raw_seq.push(frame.seq);
            raw_tag.push(tag);
            raw_dtype.push(sw.dtype);
            raw_val.push(sw.value_i24 as f64 * scale);
        }
    }
    if raw_seq.is_empty() {
        bail!("no ESF-RAW samples found");
    }

    let (raw_tag_u, a_raw, b_raw) =
        fit_tag_ms_map(&raw_seq, &raw_tag, &timeline.masters, Some(1 << 16));
    let mut imu_packets = Vec::<ImuPacket>::new();
    let mut current_tag: Option<u64> = None;
    let mut t_ms = 0.0_f64;
    let mut gx = None;
    let mut gy = None;
    let mut gz = None;
    let mut ax = None;
    let mut ay = None;
    let mut az = None;

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
            if let Some(mapped_ms) = timeline.map_tag_ms(a_raw, b_raw, *tag_u as f64, *seq) {
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
    Ok(imu_packets)
}

fn collect_nav_events(frames: &[UbxFrame], timeline: &MasterTimeline) -> Vec<(f64, NavPvtObs)> {
    let mut nav_events = Vec::new();
    for frame in frames {
        if let Some(t_ms) = nearest_master_ms(frame.seq, &timeline.masters) {
            let obs = extract_nav2_pvt_obs(frame).or_else(|| extract_nav_pvt_obs(frame));
            if let Some(obs) = obs
                && obs.fix_ok
                && !obs.invalid_llh
            {
                nav_events.push((t_ms, obs));
            }
        }
    }
    nav_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    nav_events
}

fn fit_tag_ms_map(
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

fn unwrap_counter(xs: &[u64], modulus: u64) -> Vec<u64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(xs.len());
    let mut acc = xs[0];
    out.push(acc);
    let mut prev = xs[0];
    for &x in &xs[1..] {
        let mut cur = x;
        if x + modulus / 2 < prev {
            cur = x + modulus;
        }
        acc += cur - prev;
        out.push(acc);
        prev = x;
    }
    out
}
