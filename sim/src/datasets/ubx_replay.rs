use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::datasets::generic_replay::{GenericGnssSample, GenericImuSample};
use crate::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_raw_samples, extract_nav_pvt_obs, extract_nav2_pvt_obs,
    parse_ubx_frames, sensor_meta,
};
use crate::visualizer::math::{deg2rad, nearest_master_ms};
use crate::visualizer::pipeline::tag_time::fit_tag_ms_map;
use crate::visualizer::pipeline::timebase::{MasterTimeline, build_master_timeline};

#[derive(Clone, Copy, Debug)]
pub struct UbxReplayConfig {
    pub gnss_pos_r_scale: f64,
    pub gnss_vel_r_scale: f64,
    pub nav_pvt_fallback_midpoint_shift_ms: f64,
}

impl Default for UbxReplayConfig {
    fn default() -> Self {
        Self {
            gnss_pos_r_scale: 0.1,
            gnss_vel_r_scale: 3.0,
            nav_pvt_fallback_midpoint_shift_ms: 250.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct UbxGenericReplay {
    pub imu_samples: Vec<GenericImuSample>,
    pub gnss_samples: Vec<GenericGnssSample>,
    pub nav_events: Vec<(f64, NavPvtObs)>,
    pub used_nav2: bool,
}

pub fn load_generic_replay(
    logfile: &Path,
    cfg: UbxReplayConfig,
) -> Result<(Vec<GenericImuSample>, Vec<GenericGnssSample>)> {
    let replay = load_generic_replay_with_nav(logfile, cfg)?;
    Ok((replay.imu_samples, replay.gnss_samples))
}

pub fn load_generic_replay_with_nav(
    logfile: &Path,
    cfg: UbxReplayConfig,
) -> Result<UbxGenericReplay> {
    let bytes =
        std::fs::read(logfile).with_context(|| format!("failed to read {}", logfile.display()))?;
    let frames = parse_ubx_frames(&bytes, None);
    let tl = build_master_timeline(&frames);
    if tl.masters.is_empty() {
        bail!("no master timeline");
    }
    build_generic_replay_from_frames(&frames, &tl, cfg)
}

pub fn build_generic_replay_from_frames(
    frames: &[UbxFrame],
    tl: &MasterTimeline,
    cfg: UbxReplayConfig,
) -> Result<UbxGenericReplay> {
    let t0_ms = tl.masters.first().map(|(_, t_ms)| *t_ms).unwrap_or(0.0);

    let mut nav_events_nav2 = Vec::<(f64, NavPvtObs)>::new();
    let mut nav_events_pvt = Vec::<(f64, NavPvtObs)>::new();
    for f in frames {
        if let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters) {
            if let Some(obs) = extract_nav2_pvt_obs(f) {
                if obs.fix_ok && !obs.invalid_llh {
                    nav_events_nav2.push((t_ms, obs));
                }
            } else if let Some(obs) = extract_nav_pvt_obs(f)
                && obs.fix_ok
                && !obs.invalid_llh
            {
                nav_events_pvt.push((t_ms, obs));
            }
        }
    }
    nav_events_nav2.sort_by(|a, b| a.0.total_cmp(&b.0));
    nav_events_pvt.sort_by(|a, b| a.0.total_cmp(&b.0));

    let (nav_events, use_nav2) = if !nav_events_nav2.is_empty() {
        (nav_events_nav2, true)
    } else {
        (nav_events_pvt, false)
    };

    let mut gnss_samples = Vec::<GenericGnssSample>::new();
    let mut nav_events_used = Vec::<(f64, NavPvtObs)>::new();
    let mut next_gps_update_ms = f64::NEG_INFINITY;
    let gps_period_ms = 500.0_f64;
    for (t_ms, nav) in nav_events {
        if !use_nav2 {
            if !next_gps_update_ms.is_finite() {
                next_gps_update_ms = t_ms;
            }
            if t_ms + 1.0e-6 < next_gps_update_ms {
                continue;
            }
            next_gps_update_ms += gps_period_ms;
        }
        let effective_t_ms = if use_nav2 {
            t_ms
        } else {
            t_ms + cfg.nav_pvt_fallback_midpoint_shift_ms
        };
        nav_events_used.push((effective_t_ms, nav));
        let speed_h = nav.vel_n_mps.hypot(nav.vel_e_mps);
        let heading_rad = if nav.head_veh_valid {
            Some(deg2rad(nav.heading_vehicle_deg))
        } else if speed_h >= 1.0 {
            Some(nav.vel_e_mps.atan2(nav.vel_n_mps))
        } else {
            Some(deg2rad(nav.heading_motion_deg))
        };
        gnss_samples.push(GenericGnssSample {
            t_s: (effective_t_ms - t0_ms) * 1.0e-3,
            lat_deg: nav.lat_deg,
            lon_deg: nav.lon_deg,
            height_m: nav.height_m,
            vel_ned_mps: [nav.vel_n_mps, nav.vel_e_mps, nav.vel_d_mps],
            pos_std_m: [
                nav.h_acc_m * cfg.gnss_pos_r_scale.sqrt(),
                nav.h_acc_m * cfg.gnss_pos_r_scale.sqrt(),
                nav.v_acc_m * cfg.gnss_pos_r_scale.sqrt(),
            ],
            vel_std_mps: [
                nav.s_acc_mps * cfg.gnss_vel_r_scale.sqrt(),
                nav.s_acc_mps * cfg.gnss_vel_r_scale.sqrt(),
                nav.s_acc_mps * cfg.gnss_vel_r_scale.sqrt(),
            ],
            heading_rad,
        });
    }

    let mut raw_seq = Vec::<u64>::new();
    let mut raw_tag = Vec::<u64>::new();
    let mut raw_dtype = Vec::<u8>::new();
    let mut raw_val = Vec::<f64>::new();
    for f in frames {
        for (tag, sw) in extract_esf_raw_samples(f) {
            let (_, _, scale) = sensor_meta(sw.dtype);
            raw_seq.push(f.seq);
            raw_tag.push(tag);
            raw_dtype.push(sw.dtype);
            raw_val.push(sw.value_i24 as f64 * scale);
        }
    }
    let (raw_tag_u, a_raw, b_raw) = fit_tag_ms_map(&raw_seq, &raw_tag, &tl.masters, Some(1 << 16));

    let mut imu_samples = Vec::<GenericImuSample>::new();
    let mut current_tag = None::<u64>;
    let mut t_ms = 0.0_f64;
    let mut gx = None::<f64>;
    let mut gy = None::<f64>;
    let mut gz = None::<f64>;
    let mut ax = None::<f64>;
    let mut ay = None::<f64>;
    let mut az = None::<f64>;

    for (((seq, tag_u), dtype), value) in raw_seq
        .iter()
        .zip(raw_tag_u.iter())
        .zip(raw_dtype.iter())
        .zip(raw_val.iter())
    {
        if current_tag != Some(*tag_u) {
            push_imu_sample(&mut imu_samples, t_ms, gx, gy, gz, ax, ay, az, t0_ms);
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
            14 => gx = Some(*value),
            13 => gy = Some(*value),
            5 => gz = Some(*value),
            16 => ax = Some(*value),
            17 => ay = Some(*value),
            18 => az = Some(*value),
            _ => {}
        }
    }
    push_imu_sample(&mut imu_samples, t_ms, gx, gy, gz, ax, ay, az, t0_ms);
    imu_samples.sort_by(|a, b| a.t_s.total_cmp(&b.t_s));

    Ok(UbxGenericReplay {
        imu_samples,
        gnss_samples,
        nav_events: nav_events_used,
        used_nav2: use_nav2,
    })
}

#[allow(clippy::too_many_arguments)]
fn push_imu_sample(
    out: &mut Vec<GenericImuSample>,
    t_ms: f64,
    gx: Option<f64>,
    gy: Option<f64>,
    gz: Option<f64>,
    ax: Option<f64>,
    ay: Option<f64>,
    az: Option<f64>,
    t0_ms: f64,
) {
    if let (Some(gx), Some(gy), Some(gz), Some(ax), Some(ay), Some(az)) = (gx, gy, gz, ax, ay, az) {
        out.push(GenericImuSample {
            t_s: (t_ms - t0_ms) * 1.0e-3,
            gyro_radps: [gx.to_radians(), gy.to_radians(), gz.to_radians()],
            accel_mps2: [ax, ay, az],
        });
    }
}
