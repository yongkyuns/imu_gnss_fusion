use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::fusion::{FusionGnssSample, FusionImuSample, SensorFusion};
use sim::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_raw_samples, extract_nav_pvt_obs, parse_ubx_frames,
    sensor_meta,
};
use sim::visualizer::math::{deg2rad, nearest_master_ms};
use sim::visualizer::pipeline::timebase::{MasterTimeline, build_master_timeline};

#[derive(Parser, Debug)]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: std::path::PathBuf,
    #[arg(long, default_value_t = 76.0)]
    window_start_s: f64,
    #[arg(long, default_value_t = 84.0)]
    window_end_s: f64,
    #[arg(long, default_value_t = 150.0)]
    predict_imu_lpf_cutoff_hz: f64,
    #[arg(long, default_value_t = 1)]
    predict_imu_decimation: usize,
    #[arg(long, default_value_t = 3.0)]
    gnss_vel_r_scale: f64,
    #[arg(long, default_value_t = false)]
    disable_gps_vel_d: bool,
    #[arg(long, default_value_t = false)]
    disable_gps_vel_all: bool,
}

#[derive(Clone, Copy)]
struct ImuPacket {
    t_ms: f64,
    gx_dps: f64,
    gy_dps: f64,
    gz_dps: f64,
    ax_mps2: f64,
    ay_mps2: f64,
    az_mps2: f64,
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

    let nav_events = collect_nav_events(&frames, &tl);
    let imu_packets = build_imu_packets(&frames, &tl)?;
    if nav_events.is_empty() || imu_packets.is_empty() {
        bail!("missing nav or imu events");
    }

    let mut fusion = SensorFusion::new();
    let mut prev_imu_t: Option<f64> = None;
    let mut nav_idx = 0usize;
    let mut prev_update_count = 0u32;

    for pkt in &imu_packets {
        let t_s = rel_s(&tl, pkt.t_ms);
        while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt.t_ms {
            let nav = nav_events[nav_idx].1;
            let pre = fusion.eskf().copied();
            let update = fusion.process_gnss(fusion_gnss_sample(
                nav,
                args.gnss_vel_r_scale,
                t_s as f32,
                args.disable_gps_vel_all,
                args.disable_gps_vel_d,
            ));
            let post = fusion.eskf().copied();
            log_transition_if_needed(
                "GNSS",
                t_s,
                pre,
                post,
                &mut prev_update_count,
                args.window_start_s,
                args.window_end_s,
                update.mount_ready,
            );
            nav_idx += 1;
        }

        let dt = match prev_imu_t {
            Some(prev) => (pkt.t_ms - prev) * 1e-3,
            None => {
                prev_imu_t = Some(pkt.t_ms);
                continue;
            }
        };
        prev_imu_t = Some(pkt.t_ms);
        if !(0.001..=0.05).contains(&dt) {
            continue;
        }

        let pre = fusion.eskf().copied();
        let update = fusion.process_imu(FusionImuSample {
            t_s: t_s as f32,
            gyro_radps: [
                pkt.gx_dps.to_radians() as f32,
                pkt.gy_dps.to_radians() as f32,
                pkt.gz_dps.to_radians() as f32,
            ],
            accel_mps2: [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32],
        });
        let post = fusion.eskf().copied();
        log_transition_if_needed(
            "IMU",
            t_s,
            pre,
            post,
            &mut prev_update_count,
            args.window_start_s,
            args.window_end_s,
            update.mount_ready,
        );
    }

    Ok(())
}

fn log_transition_if_needed(
    source: &str,
    t_s: f64,
    pre: Option<sensor_fusion::c_api::CEskf>,
    post: Option<sensor_fusion::c_api::CEskf>,
    prev_update_count: &mut u32,
    window_start_s: f64,
    window_end_s: f64,
    mount_ready: bool,
) {
    let Some(post) = post else {
        return;
    };
    if post.update_diag.total_updates == *prev_update_count {
        return;
    }
    *prev_update_count = post.update_diag.total_updates;
    if t_s < window_start_s || t_s > window_end_s {
        return;
    }

    let pre = pre.unwrap_or(post);
    let pre_body = nominal_vehicle_velocity(&pre);
    let post_body = nominal_vehicle_velocity(&post);
    let (pre_r, pre_p, pre_y) = quat_rpy_deg(pre.nominal.q0, pre.nominal.q1, pre.nominal.q2, pre.nominal.q3);
    let (post_r, post_p, post_y) =
        quat_rpy_deg(post.nominal.q0, post.nominal.q1, post.nominal.q2, post.nominal.q3);
    let (pre_mr, pre_mp, pre_my) = quat_rpy_deg(
        pre.nominal.qcs0,
        pre.nominal.qcs1,
        pre.nominal.qcs2,
        pre.nominal.qcs3,
    );
    let (post_mr, post_mp, post_my) = quat_rpy_deg(
        post.nominal.qcs0,
        post.nominal.qcs1,
        post.nominal.qcs2,
        post.nominal.qcs3,
    );

    println!(
        "t={:.3}s src={} type={} mount_ready={} innov={:.3} var={:.3} \
pre_att=[{:.2},{:.2},{:.2}] post_att=[{:.2},{:.2},{:.2}] \
pre_body=[{:.3},{:.3},{:.3}] post_body=[{:.3},{:.3},{:.3}] \
pre_qcs=[{:.2},{:.2},{:.2}] post_qcs=[{:.2},{:.2},{:.2}] \
last_dyaw={:.3}",
        t_s,
        source,
        update_type_name(post.update_diag.last_type),
        mount_ready,
        post.update_diag.last_innovation,
        post.update_diag.last_innovation_var,
        pre_r,
        pre_p,
        pre_y,
        post_r,
        post_p,
        post_y,
        pre_body[0],
        pre_body[1],
        pre_body[2],
        post_body[0],
        post_body[1],
        post_body[2],
        pre_mr,
        pre_mp,
        pre_my,
        post_mr,
        post_mp,
        post_my,
        post.update_diag.last_dx_mount_yaw.to_degrees()
    );
}

fn update_type_name(t: u32) -> &'static str {
    match t {
        0 => "GPS_POS",
        1 => "GPS_VEL",
        2 => "ZERO_VEL",
        3 => "BODY_SPEED_X",
        4 => "BODY_VEL_Y",
        5 => "BODY_VEL_Z",
        6 => "STATIONARY_X",
        7 => "STATIONARY_Y",
        8 => "GPS_POS_D",
        9 => "GPS_VEL_D",
        10 => "ZERO_VEL_D",
        _ => "UNKNOWN",
    }
}

fn nominal_vehicle_velocity(eskf: &sensor_fusion::c_api::CEskf) -> [f64; 3] {
    let c_n_b = quat_to_rotmat_f64([
        eskf.nominal.q0 as f64,
        eskf.nominal.q1 as f64,
        eskf.nominal.q2 as f64,
        eskf.nominal.q3 as f64,
    ]);
    let c_b_n = transpose3(c_n_b);
    let v_seed = mat_vec(c_b_n, [
        eskf.nominal.vn as f64,
        eskf.nominal.ve as f64,
        eskf.nominal.vd as f64,
    ]);
    let c_c_s = quat_to_rotmat_f64([
        eskf.nominal.qcs0 as f64,
        eskf.nominal.qcs1 as f64,
        eskf.nominal.qcs2 as f64,
        eskf.nominal.qcs3 as f64,
    ]);
    mat_vec(c_c_s, v_seed)
}

fn quat_rpy_deg(q0: f32, q1: f32, q2: f32, q3: f32) -> (f64, f64, f64) {
    let q0 = q0 as f64;
    let q1 = q1 as f64;
    let q2 = q2 as f64;
    let q3 = q3 as f64;
    let sinr_cosp = 2.0 * (q0 * q1 + q2 * q3);
    let cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2);
    let roll = sinr_cosp.atan2(cosr_cosp).to_degrees();
    let sinp = 2.0 * (q0 * q2 - q3 * q1);
    let pitch = sinp.clamp(-1.0, 1.0).asin().to_degrees();
    let siny_cosp = 2.0 * (q0 * q3 + q1 * q2);
    let cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3);
    let yaw = siny_cosp.atan2(cosy_cosp).to_degrees();
    (roll, pitch, yaw)
}

fn mat_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn quat_to_rotmat_f64(q: [f64; 4]) -> [[f64; 3]; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (w, x, y, z) = if n > 1.0e-12 {
        (q[0] / n, q[1] / n, q[2] / n, q[3] / n)
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn transpose3(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn rel_s(tl: &MasterTimeline, t_ms: f64) -> f64 {
    (t_ms - tl.masters.first().map(|(_, ms)| *ms).unwrap_or(t_ms)) * 1.0e-3
}

fn collect_nav_events(frames: &[UbxFrame], tl: &MasterTimeline) -> Vec<(f64, NavPvtObs)> {
    let mut out = Vec::<(f64, NavPvtObs)>::new();
    for f in frames {
        if let Some(nav) = extract_nav_pvt_obs(f) {
            if !nav.fix_ok || nav.invalid_llh {
                continue;
            }
            if let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters) {
                out.push((t_ms, nav));
            }
        }
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out
}

fn build_imu_packets(frames: &[UbxFrame], tl: &MasterTimeline) -> Result<Vec<ImuPacket>> {
    let mut raw_seq = Vec::<u64>::new();
    let mut raw_tag = Vec::<u64>::new();
    let mut raw_dtype = Vec::<u8>::new();
    let mut raw_val = Vec::<f64>::new();
    for f in frames {
        for (tag, sw) in extract_esf_raw_samples(f) {
            let tag = tag as u64;
            let (_name, _unit, scale) = sensor_meta(sw.dtype);
            raw_seq.push(f.seq);
            raw_tag.push(tag);
            raw_dtype.push(sw.dtype);
            raw_val.push(sw.value_i24 as f64 * scale);
        }
    }
    let (raw_tag_u, a_raw, b_raw) = fit_tag_ms_map(&raw_seq, &raw_tag, &tl.masters, Some(1 << 16));
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
            if let Some(mapped_ms) = tl.map_tag_ms(a_raw, b_raw, *tag_u as f64, *seq) {
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
    imu_packets.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap());
    Ok(imu_packets)
}

fn fit_tag_ms_map(
    seqs: &[u64],
    tags: &[u64],
    masters: &[(u64, f64)],
    unwrap_modulus: Option<u64>,
) -> (Vec<u64>, f64, f64) {
    let mapped_tags = match unwrap_modulus {
        Some(m) => sim::ubxlog::unwrap_counter(tags, m),
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
    let (a, b) = sim::ubxlog::fit_linear_map(&x, &y, 1e-3);
    (mapped_tags, a, b)
}

fn fusion_gnss_sample(
    nav: NavPvtObs,
    gnss_vel_r_scale: f64,
    t_s: f32,
    disable_gps_vel_all: bool,
    disable_gps_vel_d: bool,
) -> FusionGnssSample {
    let speed_h = nav.vel_n_mps.hypot(nav.vel_e_mps);
    let heading_rad = if nav.head_veh_valid {
        Some(deg2rad(nav.heading_vehicle_deg) as f32)
    } else if speed_h >= 1.0 {
        Some(nav.vel_e_mps.atan2(nav.vel_n_mps) as f32)
    } else {
        Some(deg2rad(nav.heading_motion_deg) as f32)
    };
    FusionGnssSample {
        t_s,
        lat_deg: nav.lat_deg as f32,
        lon_deg: nav.lon_deg as f32,
        height_m: nav.height_m as f32,
        vel_ned_mps: [nav.vel_n_mps as f32, nav.vel_e_mps as f32, nav.vel_d_mps as f32],
        pos_std_m: [nav.h_acc_m as f32, nav.h_acc_m as f32, (nav.v_acc_m * 2.5) as f32],
        vel_std_mps: [
            if disable_gps_vel_all {
                1.0e6_f32
            } else {
                (nav.s_acc_mps * gnss_vel_r_scale.sqrt()) as f32
            },
            if disable_gps_vel_all {
                1.0e6_f32
            } else {
                (nav.s_acc_mps * gnss_vel_r_scale.sqrt()) as f32
            },
            if disable_gps_vel_all || disable_gps_vel_d {
                1.0e6_f32
            } else {
                (nav.s_acc_mps * gnss_vel_r_scale.sqrt()) as f32
            },
        ],
        heading_rad,
    }
}
