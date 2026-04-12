use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::fusion::{FusionGnssSample, FusionImuSample, SensorFusion};
use sim::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_raw_samples, extract_nav2_pvt_obs, parse_ubx_frames,
    sensor_meta,
};
use sim::visualizer::math::{ecef_to_ned, lla_to_ecef, nearest_master_ms};
use sim::visualizer::model::ImuPacket;
use sim::visualizer::pipeline::timebase::{MasterTimeline, build_master_timeline};

const MAGIC: u16 = 0x5346;
const VERSION: u8 = 1;

const MSG_CONFIG: u8 = 0x01;
const MSG_RESET: u8 = 0x02;
const MSG_IMU: u8 = 0x10;
const MSG_GNSS: u8 = 0x11;
const MSG_END: u8 = 0x12;
const MSG_STATUS: u8 = 0x81;

const MODE_INTERNAL_ALIGN: u8 = 0;
const MODE_EXTERNAL_QVB: u8 = 1;

const FLAG_MOUNT_READY: u32 = 1 << 0;
const FLAG_MOUNT_READY_CHANGED: u32 = 1 << 1;
const FLAG_EKF_INITIALIZED: u32 = 1 << 2;
const FLAG_EKF_INITIALIZED_NOW: u32 = 1 << 3;
const FLAG_MOUNT_Q_VALID: u32 = 1 << 4;
const FLAG_END_ACK: u32 = 1 << 5;

#[derive(Parser, Debug)]
#[command(name = "esp32_usb_replay")]
struct Args {
    #[arg(long)]
    port: String,

    #[arg(long, default_value_t = 921600)]
    baud: u32,

    #[arg(long, default_value_t = 200)]
    serial_timeout_ms: u64,

    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    external_q_vb: Option<String>,

    #[arg(long)]
    r_body_vel: Option<f32>,

    #[arg(long, default_value_t = 1000)]
    settle_ms: u64,

    #[arg(long)]
    max_time_s: Option<f32>,

    #[arg(long, default_value_t = false)]
    summary_only: bool,

    #[arg(long, default_value_t = 0)]
    tx_sleep_us: u64,

    #[arg(long)]
    replay_speedup: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
enum ReplayEvent {
    Imu {
        t_s: f32,
        gyro_radps: [f32; 3],
        accel_mps2: [f32; 3],
    },
    Gnss {
        t_s: f32,
        lat_deg: f32,
        lon_deg: f32,
        height_m: f32,
        pos_ned_m: [f32; 3],
        vel_ned_mps: [f32; 3],
        pos_std_m: [f32; 3],
        vel_std_mps: [f32; 3],
        heading_valid: bool,
        heading_rad: f32,
    },
}

#[derive(Debug, Clone, Copy)]
struct StatusFrame {
    t_s: f32,
    flags: u32,
    mount_q_vb: [f32; 4],
    ekf_q_bn: [f32; 4],
    ekf_vel_ned_mps: [f32; 3],
    ekf_pos_ned_m: [f32; 3],
    align_sigma_rad: [f32; 3],
    imu_count: u32,
    gnss_count: u32,
    imu_avg_us: f32,
    imu_max_us: f32,
    gnss_avg_us: f32,
    gnss_max_us: f32,
    imu_rotate_count: u32,
    imu_rotate_avg_us: f32,
    imu_rotate_max_us: f32,
    imu_predict_count: u32,
    imu_predict_avg_us: f32,
    imu_predict_max_us: f32,
    imu_clamp_count: u32,
    imu_clamp_avg_us: f32,
    imu_clamp_max_us: f32,
    imu_body_vel_count: u32,
    imu_body_vel_avg_us: f32,
    imu_body_vel_max_us: f32,
    gnss_align_count: u32,
    gnss_align_avg_us: f32,
    gnss_align_max_us: f32,
    gnss_init_count: u32,
    gnss_init_avg_us: f32,
    gnss_init_max_us: f32,
    gnss_fuse_count: u32,
    gnss_fuse_avg_us: f32,
    gnss_fuse_max_us: f32,
}

#[derive(Debug, Clone, Copy)]
struct HostSummary {
    mount_ready_t_s: Option<f32>,
    ekf_init_t_s: Option<f32>,
    mount_q_vb: Option<[f32; 4]>,
    ekf_q_bn: Option<[f32; 4]>,
    ekf_vel_ned_mps: Option<[f32; 3]>,
    ekf_pos_ned_m: Option<[f32; 3]>,
    align_sigma_rad: Option<[f32; 3]>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut events = build_replay_events(&args.input)?;
    if let Some(max_time_s) = args.max_time_s {
        events.retain(|ev| match *ev {
            ReplayEvent::Imu { t_s, .. } | ReplayEvent::Gnss { t_s, .. } => t_s <= max_time_s,
        });
    }
    if events.is_empty() {
        bail!(
            "no IMU/GNSS replay events parsed from {}",
            args.input.display()
        );
    }

    let mut port = serialport::new(&args.port, args.baud)
        .timeout(Duration::from_millis(args.serial_timeout_ms))
        .open()
        .with_context(|| format!("failed to open serial port {}", args.port))?;

    std::thread::sleep(Duration::from_millis(args.settle_ms));

    let (mode, q_vb) = parse_mode_and_q(&args.external_q_vb)?;
    let host_summary = run_host_reference(&events, mode, q_vb);
    write_frame(
        &mut *port,
        MSG_CONFIG,
        &build_config_payload(mode, q_vb, args.r_body_vel),
    )?;
    write_frame(&mut *port, MSG_RESET, &[])?;

    let mut rx = Vec::<u8>::new();
    let mut last_status: Option<StatusFrame> = None;
    if !args.summary_only {
        println!(
            "t_s,flags,mount_ready,ekf_initialized,mount_q0,mount_q1,mount_q2,mount_q3,ekf_q0,ekf_q1,ekf_q2,ekf_q3,pn,pe,pd,vn,ve,vd,sigma_roll,sigma_pitch,sigma_yaw,imu_count,gnss_count,imu_avg_us,imu_max_us,gnss_avg_us,gnss_max_us,imu_rotate_count,imu_rotate_avg_us,imu_rotate_max_us,imu_predict_count,imu_predict_avg_us,imu_predict_max_us,imu_clamp_count,imu_clamp_avg_us,imu_clamp_max_us,imu_body_vel_count,imu_body_vel_avg_us,imu_body_vel_max_us,gnss_align_count,gnss_align_avg_us,gnss_align_max_us,gnss_init_count,gnss_init_avg_us,gnss_init_max_us,gnss_fuse_count,gnss_fuse_avg_us,gnss_fuse_max_us"
        );
    }

    let mut prev_t_s: Option<f32> = None;
    for ev in events {
        let ev_t_s = match ev {
            ReplayEvent::Imu { t_s, .. } | ReplayEvent::Gnss { t_s, .. } => t_s,
        };
        if let (Some(speedup), Some(prev)) = (args.replay_speedup, prev_t_s) {
            if speedup.is_finite() && speedup > 0.0 {
                let dt_s = (ev_t_s - prev).max(0.0) / speedup;
                if dt_s > 0.0 {
                    std::thread::sleep(Duration::from_secs_f32(dt_s));
                }
            }
        }
        match ev {
            ReplayEvent::Imu {
                t_s,
                gyro_radps,
                accel_mps2,
            } => {
                write_frame(
                    &mut *port,
                    MSG_IMU,
                    &build_imu_payload(t_s, gyro_radps, accel_mps2),
                )?;
            }
            ReplayEvent::Gnss {
                t_s,
                lat_deg: _,
                lon_deg: _,
                height_m: _,
                pos_ned_m,
                vel_ned_mps,
                pos_std_m,
                vel_std_mps,
                heading_valid,
                heading_rad,
            } => {
                write_frame(
                    &mut *port,
                    MSG_GNSS,
                    &build_gnss_payload(
                        t_s,
                        pos_ned_m,
                        vel_ned_mps,
                        pos_std_m,
                        vel_std_mps,
                        heading_valid,
                        heading_rad,
                    ),
                )?;
            }
        }
        if args.tx_sleep_us > 0 {
            std::thread::sleep(Duration::from_micros(args.tx_sleep_us));
        }
        prev_t_s = Some(ev_t_s);
        drain_status(
            &mut *port,
            &mut rx,
            Duration::from_millis(2),
            &mut last_status,
            args.summary_only,
        )?;
    }

    write_frame(&mut *port, MSG_END, &[])?;
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if drain_status(
            &mut *port,
            &mut rx,
            Duration::from_millis(50),
            &mut last_status,
            args.summary_only,
        )? {
            break;
        }
    }

    if let Some(device_summary) = last_status {
        print_summary(&device_summary, &host_summary);
    } else {
        bail!("no STATUS frame received from device");
    }

    Ok(())
}

fn run_host_reference(events: &[ReplayEvent], mode: u8, q_vb: [f32; 4]) -> HostSummary {
    let mut fusion = if mode == MODE_EXTERNAL_QVB {
        SensorFusion::with_misalignment(q_vb)
    } else {
        SensorFusion::new()
    };
    let mut mount_ready_t_s = None;
    let mut ekf_init_t_s = None;

    for ev in events {
        match *ev {
            ReplayEvent::Imu {
                t_s,
                gyro_radps,
                accel_mps2,
            } => {
                let update = fusion.process_imu(FusionImuSample {
                    t_s,
                    gyro_radps,
                    accel_mps2,
                });
                if update.mount_ready_changed && mount_ready_t_s.is_none() {
                    mount_ready_t_s = Some(t_s);
                }
                if update.ekf_initialized_now && ekf_init_t_s.is_none() {
                    ekf_init_t_s = Some(t_s);
                }
            }
            ReplayEvent::Gnss {
                t_s,
                lat_deg,
                lon_deg,
                height_m,
                pos_ned_m,
                vel_ned_mps,
                pos_std_m,
                vel_std_mps,
                heading_valid,
                heading_rad,
            } => {
                let update = fusion.process_gnss(FusionGnssSample {
                    t_s,
                    lat_deg,
                    lon_deg,
                    height_m,
                    vel_ned_mps,
                    pos_std_m,
                    vel_std_mps,
                    heading_rad: heading_valid.then_some(heading_rad),
                });
                if update.mount_ready_changed && mount_ready_t_s.is_none() {
                    mount_ready_t_s = Some(t_s);
                }
                if update.ekf_initialized_now && ekf_init_t_s.is_none() {
                    ekf_init_t_s = Some(t_s);
                }
            }
        }
    }

    let mount_q_vb = fusion.mount_q_vb();
    let align_sigma_rad = fusion.align().map(|align| {
        [
            align.P[0][0].max(0.0).sqrt(),
            align.P[1][1].max(0.0).sqrt(),
            align.P[2][2].max(0.0).sqrt(),
        ]
    });
    let (ekf_q_bn, ekf_vel_ned_mps, ekf_pos_ned_m) = if let Some(eskf) = fusion.eskf() {
        (
            Some([
                eskf.nominal.q0,
                eskf.nominal.q1,
                eskf.nominal.q2,
                eskf.nominal.q3,
            ]),
            Some([eskf.nominal.vn, eskf.nominal.ve, eskf.nominal.vd]),
            Some([eskf.nominal.pn, eskf.nominal.pe, eskf.nominal.pd]),
        )
    } else {
        (None, None, None)
    };

    HostSummary {
        mount_ready_t_s,
        ekf_init_t_s,
        mount_q_vb,
        ekf_q_bn,
        ekf_vel_ned_mps,
        ekf_pos_ned_m,
        align_sigma_rad,
    }
}

fn build_replay_events(path: &PathBuf) -> Result<Vec<ReplayEvent>> {
    let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let frames = parse_ubx_frames(&data, None);
    let tl = build_master_timeline(&frames);
    let nav_events = build_nav_events(&frames, &tl);
    let imu_packets = build_imu_packets(&frames, &tl);
    if imu_packets.is_empty() || nav_events.is_empty() {
        bail!("need both IMU and NAV2-PVT data");
    }

    let ref_nav = nav_events.first().map(|(_, nav)| *nav);
    let t0_ms = imu_packets
        .first()
        .map(|p| p.t_ms)
        .into_iter()
        .chain(nav_events.first().map(|(t, _)| *t))
        .fold(f64::INFINITY, f64::min);

    let mut events = Vec::<(f64, ReplayEvent)>::with_capacity(imu_packets.len() + nav_events.len());
    for pkt in imu_packets {
        let t_s = ((pkt.t_ms - t0_ms) / 1000.0) as f32;
        events.push((
            pkt.t_ms,
            ReplayEvent::Imu {
                t_s,
                gyro_radps: [
                    pkt.gx_dps.to_radians() as f32,
                    pkt.gy_dps.to_radians() as f32,
                    pkt.gz_dps.to_radians() as f32,
                ],
                accel_mps2: [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32],
            },
        ));
    }
    for (t_ms, nav) in nav_events {
        let pos_ned_m = nav_to_ned(nav, ref_nav);
        let heading_valid = nav.head_veh_valid;
        let heading_rad = if heading_valid {
            nav.heading_vehicle_deg.to_radians() as f32
        } else {
            0.0
        };
        events.push((
            t_ms,
            ReplayEvent::Gnss {
                t_s: ((t_ms - t0_ms) / 1000.0) as f32,
                lat_deg: nav.lat_deg as f32,
                lon_deg: nav.lon_deg as f32,
                height_m: nav.height_m as f32,
                pos_ned_m,
                vel_ned_mps: [
                    nav.vel_n_mps as f32,
                    nav.vel_e_mps as f32,
                    nav.vel_d_mps as f32,
                ],
                pos_std_m: [nav.h_acc_m as f32, nav.h_acc_m as f32, nav.v_acc_m as f32],
                vel_std_mps: [
                    nav.s_acc_mps as f32,
                    nav.s_acc_mps as f32,
                    nav.s_acc_mps as f32,
                ],
                heading_valid,
                heading_rad,
            },
        ));
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    Ok(events.into_iter().map(|(_, ev)| ev).collect())
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
            let (_, _, scale) = sensor_meta(sw.dtype);
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
            if let Some(mapped_ms) = map_tag_ms(a_raw, b_raw, *tag_u as f64, *seq, &tl.masters) {
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

fn fit_tag_ms_map_local(
    raw_seq: &[u64],
    raw_tag: &[u64],
    masters: &[(u64, f64)],
    wrap: Option<u64>,
) -> (Vec<u64>, f64, f64) {
    let raw_tag_u = unwrap_counter(raw_tag, wrap);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (seq, tag_u) in raw_seq.iter().zip(raw_tag_u.iter()) {
        if let Some(ms) = nearest_master_ms(*seq, masters) {
            xs.push(*tag_u as f64);
            ys.push(ms);
        }
    }
    let (a, b) = fit_linear_map(&xs, &ys).unwrap_or((0.0, 1.0));
    (raw_tag_u, a, b)
}

fn unwrap_counter(values: &[u64], wrap: Option<u64>) -> Vec<u64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    let wrap = wrap.unwrap_or(0);
    let mut offset = 0u64;
    let mut prev = values[0];
    out.push(prev);
    for &v in &values[1..] {
        if wrap != 0 && v + wrap / 2 < prev {
            offset = offset.saturating_add(wrap);
        }
        prev = v;
        out.push(v.saturating_add(offset));
    }
    out
}

fn fit_linear_map(xs: &[f64], ys: &[f64]) -> Option<(f64, f64)> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1.0e-9 {
        return None;
    }
    let b = (n * sxy - sx * sy) / denom;
    let a = (sy - b * sx) / n;
    Some((a, b))
}

fn map_tag_ms(a: f64, b: f64, tag: f64, _seq: u64, _masters: &[(u64, f64)]) -> Option<f64> {
    Some(a + b * tag)
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

fn parse_mode_and_q(external_q_vb: &Option<String>) -> Result<(u8, [f32; 4])> {
    if let Some(text) = external_q_vb {
        let vals: Vec<f32> = text
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to parse --external-q-vb")?;
        if vals.len() != 4 {
            bail!("--external-q-vb expects 4 comma-separated floats");
        }
        Ok((MODE_EXTERNAL_QVB, [vals[0], vals[1], vals[2], vals[3]]))
    } else {
        Ok((MODE_INTERNAL_ALIGN, [1.0, 0.0, 0.0, 0.0]))
    }
}

fn build_header(msg_type: u8, payload_len: usize) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&MAGIC.to_le_bytes());
    out[2] = VERSION;
    out[3] = msg_type;
    out[4..6].copy_from_slice(&(payload_len as u16).to_le_bytes());
    out[6..8].copy_from_slice(&0u16.to_le_bytes());
    out
}

fn build_config_payload(mode: u8, q_vb: [f32; 4], r_body_vel: Option<f32>) -> Vec<u8> {
    let mut out = Vec::with_capacity(24);
    out.extend_from_slice(&[mode, 0, 0, 0]);
    for v in q_vb {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&r_body_vel.unwrap_or(f32::NAN).to_le_bytes());
    out
}

fn build_imu_payload(t_s: f32, gyro_radps: [f32; 3], accel_mps2: [f32; 3]) -> Vec<u8> {
    let mut out = Vec::with_capacity(28);
    out.extend_from_slice(&t_s.to_le_bytes());
    for v in gyro_radps {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in accel_mps2 {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn build_gnss_payload(
    t_s: f32,
    pos_ned_m: [f32; 3],
    vel_ned_mps: [f32; 3],
    pos_std_m: [f32; 3],
    vel_std_mps: [f32; 3],
    heading_valid: bool,
    heading_rad: f32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(60);
    out.extend_from_slice(&t_s.to_le_bytes());
    for v in pos_ned_m {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in vel_ned_mps {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in pos_std_m {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in vel_std_mps {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.push(u8::from(heading_valid));
    out.extend_from_slice(&[0, 0, 0]);
    out.extend_from_slice(&heading_rad.to_le_bytes());
    out
}

fn write_frame(port: &mut dyn serialport::SerialPort, msg_type: u8, payload: &[u8]) -> Result<()> {
    let hdr = build_header(msg_type, payload.len());
    port.write_all(&hdr)?;
    port.write_all(payload)?;
    Ok(())
}

fn drain_status(
    port: &mut dyn serialport::SerialPort,
    rx: &mut Vec<u8>,
    wait: Duration,
    last_status: &mut Option<StatusFrame>,
    summary_only: bool,
) -> Result<bool> {
    let deadline = Instant::now() + wait;
    let mut saw_end = false;
    while Instant::now() < deadline {
        let mut buf = [0u8; 256];
        let pending = port.bytes_to_read().unwrap_or(0);
        if pending > 0 {
            match port.read(&mut buf) {
                Ok(n) if n > 0 => rx.extend_from_slice(&buf[..n]),
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => return Err(e.into()),
            }
        } else if wait > Duration::from_millis(0) {
            std::thread::sleep(Duration::from_millis(1));
        }
        while let Some((status, consumed)) = try_parse_status(rx) {
            if let Some(status) = status {
                if !summary_only {
                    print_status(&status);
                }
                if (status.flags & FLAG_END_ACK) != 0 {
                    saw_end = true;
                }
                *last_status = Some(status);
            }
            rx.drain(0..consumed);
        }
        if saw_end {
            return Ok(true);
        }
    }
    Ok(false)
}

fn try_parse_status(buf: &[u8]) -> Option<(Option<StatusFrame>, usize)> {
    if buf.len() < 8 {
        return None;
    }
    let magic = u16::from_le_bytes([buf[0], buf[1]]);
    if magic != MAGIC || buf[2] != VERSION {
        return Some((None, 1));
    }
    if buf[3] != MSG_STATUS {
        return Some((None, 1));
    }
    let payload_len = u16::from_le_bytes([buf[4], buf[5]]) as usize;
    let total_len = 8 + payload_len;
    if buf.len() < total_len {
        return None;
    }
    if payload_len != 184 {
        return Some((None, 1));
    }
    let p = &buf[8..total_len];
    let mut at = 0usize;
    let read_f32 = |p: &[u8], at: &mut usize| -> f32 {
        let v = f32::from_le_bytes([p[*at], p[*at + 1], p[*at + 2], p[*at + 3]]);
        *at += 4;
        v
    };
    let t_s = read_f32(p, &mut at);
    let flags = u32::from_le_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]]);
    at += 4;
    let mut mount_q_vb = [0.0f32; 4];
    for v in &mut mount_q_vb {
        *v = read_f32(p, &mut at);
    }
    let mut ekf_q_bn = [0.0f32; 4];
    for v in &mut ekf_q_bn {
        *v = read_f32(p, &mut at);
    }
    let mut ekf_vel_ned_mps = [0.0f32; 3];
    for v in &mut ekf_vel_ned_mps {
        *v = read_f32(p, &mut at);
    }
    let mut ekf_pos_ned_m = [0.0f32; 3];
    for v in &mut ekf_pos_ned_m {
        *v = read_f32(p, &mut at);
    }
    let mut align_sigma_rad = [0.0f32; 3];
    for v in &mut align_sigma_rad {
        *v = read_f32(p, &mut at);
    }
    let imu_count = u32::from_le_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]]);
    at += 4;
    let gnss_count = u32::from_le_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]]);
    at += 4;
    let imu_avg_us = read_f32(p, &mut at);
    let imu_max_us = read_f32(p, &mut at);
    let gnss_avg_us = read_f32(p, &mut at);
    let gnss_max_us = read_f32(p, &mut at);
    let imu_rotate_count = u32::from_le_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]]);
    at += 4;
    let imu_rotate_avg_us = read_f32(p, &mut at);
    let imu_rotate_max_us = read_f32(p, &mut at);
    let imu_predict_count = u32::from_le_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]]);
    at += 4;
    let imu_predict_avg_us = read_f32(p, &mut at);
    let imu_predict_max_us = read_f32(p, &mut at);
    let imu_clamp_count = u32::from_le_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]]);
    at += 4;
    let imu_clamp_avg_us = read_f32(p, &mut at);
    let imu_clamp_max_us = read_f32(p, &mut at);
    let imu_body_vel_count = u32::from_le_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]]);
    at += 4;
    let imu_body_vel_avg_us = read_f32(p, &mut at);
    let imu_body_vel_max_us = read_f32(p, &mut at);
    let gnss_align_count = u32::from_le_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]]);
    at += 4;
    let gnss_align_avg_us = read_f32(p, &mut at);
    let gnss_align_max_us = read_f32(p, &mut at);
    let gnss_init_count = u32::from_le_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]]);
    at += 4;
    let gnss_init_avg_us = read_f32(p, &mut at);
    let gnss_init_max_us = read_f32(p, &mut at);
    let gnss_fuse_count = u32::from_le_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]]);
    at += 4;
    let gnss_fuse_avg_us = read_f32(p, &mut at);
    let gnss_fuse_max_us = read_f32(p, &mut at);
    Some((
        Some(StatusFrame {
            t_s,
            flags,
            mount_q_vb,
            ekf_q_bn,
            ekf_vel_ned_mps,
            ekf_pos_ned_m,
            align_sigma_rad,
            imu_count,
            gnss_count,
            imu_avg_us,
            imu_max_us,
            gnss_avg_us,
            gnss_max_us,
            imu_rotate_count,
            imu_rotate_avg_us,
            imu_rotate_max_us,
            imu_predict_count,
            imu_predict_avg_us,
            imu_predict_max_us,
            imu_clamp_count,
            imu_clamp_avg_us,
            imu_clamp_max_us,
            imu_body_vel_count,
            imu_body_vel_avg_us,
            imu_body_vel_max_us,
            gnss_align_count,
            gnss_align_avg_us,
            gnss_align_max_us,
            gnss_init_count,
            gnss_init_avg_us,
            gnss_init_max_us,
            gnss_fuse_count,
            gnss_fuse_avg_us,
            gnss_fuse_max_us,
        }),
        total_len,
    ))
}

fn print_status(status: &StatusFrame) {
    println!(
        "{:.3},{:#010x},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{},{},{:.3},{:.3},{:.3},{:.3},{},{:.3},{:.3},{},{:.3},{:.3},{},{:.3},{:.3},{},{:.3},{:.3},{},{:.3},{:.3},{},{:.3},{:.3},{},{:.3},{:.3}",
        status.t_s,
        status.flags,
        (status.flags & FLAG_MOUNT_READY) != 0,
        (status.flags & FLAG_EKF_INITIALIZED) != 0,
        status.mount_q_vb[0],
        status.mount_q_vb[1],
        status.mount_q_vb[2],
        status.mount_q_vb[3],
        status.ekf_q_bn[0],
        status.ekf_q_bn[1],
        status.ekf_q_bn[2],
        status.ekf_q_bn[3],
        status.ekf_pos_ned_m[0],
        status.ekf_pos_ned_m[1],
        status.ekf_pos_ned_m[2],
        status.ekf_vel_ned_mps[0],
        status.ekf_vel_ned_mps[1],
        status.ekf_vel_ned_mps[2],
        status.align_sigma_rad[0],
        status.align_sigma_rad[1],
        status.align_sigma_rad[2],
        status.imu_count,
        status.gnss_count,
        status.imu_avg_us,
        status.imu_max_us,
        status.gnss_avg_us,
        status.gnss_max_us,
        status.imu_rotate_count,
        status.imu_rotate_avg_us,
        status.imu_rotate_max_us,
        status.imu_predict_count,
        status.imu_predict_avg_us,
        status.imu_predict_max_us,
        status.imu_clamp_count,
        status.imu_clamp_avg_us,
        status.imu_clamp_max_us,
        status.imu_body_vel_count,
        status.imu_body_vel_avg_us,
        status.imu_body_vel_max_us,
        status.gnss_align_count,
        status.gnss_align_avg_us,
        status.gnss_align_max_us,
        status.gnss_init_count,
        status.gnss_init_avg_us,
        status.gnss_init_max_us,
        status.gnss_fuse_count,
        status.gnss_fuse_avg_us,
        status.gnss_fuse_max_us
    );
}

fn print_summary(device: &StatusFrame, host: &HostSummary) {
    eprintln!("device_final_t_s,{:.3}", device.t_s);
    eprintln!(
        "device_flags,mount_ready={},mount_ready_changed={},ekf_initialized={},ekf_initialized_now={},end_ack={}",
        (device.flags & FLAG_MOUNT_READY) != 0,
        (device.flags & FLAG_MOUNT_READY_CHANGED) != 0,
        (device.flags & FLAG_EKF_INITIALIZED) != 0,
        (device.flags & FLAG_EKF_INITIALIZED_NOW) != 0,
        (device.flags & FLAG_END_ACK) != 0
    );
    if let Some(t) = host.mount_ready_t_s {
        eprintln!("host_mount_ready_t_s,{:.3}", t);
    } else {
        eprintln!("host_mount_ready_t_s,none");
    }
    if let Some(t) = host.ekf_init_t_s {
        eprintln!("host_ekf_init_t_s,{:.3}", t);
    } else {
        eprintln!("host_ekf_init_t_s,none");
    }
    if let (Some(host_q), true) = (host.mount_q_vb, (device.flags & FLAG_MOUNT_Q_VALID) != 0) {
        eprintln!(
            "mount_q_vb_max_abs_diff,{:.8}",
            max_abs_diff4(device.mount_q_vb, host_q)
        );
    }
    if let Some(host_q) = host.ekf_q_bn {
        eprintln!(
            "ekf_q_bn_max_abs_diff,{:.8}",
            max_abs_diff4(device.ekf_q_bn, host_q)
        );
    }
    if let Some(host_v) = host.ekf_vel_ned_mps {
        eprintln!(
            "ekf_vel_max_abs_diff,{:.8}",
            max_abs_diff3(device.ekf_vel_ned_mps, host_v)
        );
    }
    if let Some(host_p) = host.ekf_pos_ned_m {
        eprintln!(
            "ekf_pos_max_abs_diff,{:.8}",
            max_abs_diff3(device.ekf_pos_ned_m, host_p)
        );
    }
    if let Some(host_sigma) = host.align_sigma_rad {
        eprintln!(
            "align_sigma_max_abs_diff,{:.8}",
            max_abs_diff3(device.align_sigma_rad, host_sigma)
        );
    }
    eprintln!(
        "timing_us,imu_count={},imu_avg={:.3},imu_max={:.3},gnss_count={},gnss_avg={:.3},gnss_max={:.3}",
        device.imu_count,
        device.imu_avg_us,
        device.imu_max_us,
        device.gnss_count,
        device.gnss_avg_us,
        device.gnss_max_us
    );
    eprintln!(
        "timing_breakdown_us,imu_rotate_count={},imu_rotate_avg={:.3},imu_rotate_max={:.3},imu_predict_count={},imu_predict_avg={:.3},imu_predict_max={:.3},imu_clamp_count={},imu_clamp_avg={:.3},imu_clamp_max={:.3},imu_body_vel_count={},imu_body_vel_avg={:.3},imu_body_vel_max={:.3},gnss_align_count={},gnss_align_avg={:.3},gnss_align_max={:.3},gnss_init_count={},gnss_init_avg={:.3},gnss_init_max={:.3},gnss_fuse_count={},gnss_fuse_avg={:.3},gnss_fuse_max={:.3}",
        device.imu_rotate_count,
        device.imu_rotate_avg_us,
        device.imu_rotate_max_us,
        device.imu_predict_count,
        device.imu_predict_avg_us,
        device.imu_predict_max_us,
        device.imu_clamp_count,
        device.imu_clamp_avg_us,
        device.imu_clamp_max_us,
        device.imu_body_vel_count,
        device.imu_body_vel_avg_us,
        device.imu_body_vel_max_us,
        device.gnss_align_count,
        device.gnss_align_avg_us,
        device.gnss_align_max_us,
        device.gnss_init_count,
        device.gnss_init_avg_us,
        device.gnss_init_max_us,
        device.gnss_fuse_count,
        device.gnss_fuse_avg_us,
        device.gnss_fuse_max_us
    );
}

fn max_abs_diff3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.into_iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs_diff4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a.into_iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}
