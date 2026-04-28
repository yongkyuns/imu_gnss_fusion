use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::eskf_types::EskfState;
use sensor_fusion::fusion::SensorFusion;
use sim::datasets::generic_replay::{
    fusion_gnss_sample as to_fusion_gnss, fusion_imu_sample as to_fusion_imu,
};
use sim::datasets::ubx_replay::{UbxReplayConfig, build_generic_replay_from_frames};
use sim::ubxlog::parse_ubx_frames;
use sim::visualizer::pipeline::timebase::build_master_timeline;

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

fn main() -> Result<()> {
    let args = Args::parse();
    let bytes = std::fs::read(&args.logfile)
        .with_context(|| format!("failed to read {}", args.logfile.display()))?;
    let frames = parse_ubx_frames(&bytes, None);
    let tl = build_master_timeline(&frames);
    if tl.masters.is_empty() {
        bail!("no master timeline");
    }

    let replay = build_generic_replay_from_frames(
        &frames,
        &tl,
        UbxReplayConfig {
            gnss_pos_r_scale: 1.0,
            gnss_vel_r_scale: args.gnss_vel_r_scale,
            ..UbxReplayConfig::default()
        },
    )?;
    let nav_events = replay.nav_events.clone();
    if nav_events.is_empty() || replay.imu_samples.is_empty() {
        bail!("missing nav or imu events");
    }

    let mut fusion = SensorFusion::new();
    let mut nav_idx = 0usize;
    let mut prev_update_count = 0u32;

    for imu_sample in &replay.imu_samples {
        let pkt_t_ms = tl.t0_master_ms + imu_sample.t_s * 1.0e3;
        let t_s = imu_sample.t_s;
        while nav_idx < nav_events.len() && nav_events[nav_idx].0 <= pkt_t_ms {
            let mut gnss_sample = replay.gnss_samples[nav_idx];
            let pre = fusion.eskf().copied();
            if args.disable_gps_vel_all {
                gnss_sample.vel_std_mps = [1.0e6; 3];
            } else if args.disable_gps_vel_d {
                gnss_sample.vel_std_mps[2] = 1.0e6;
            }
            let update = fusion.process_gnss(to_fusion_gnss(gnss_sample));
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

        let pre = fusion.eskf().copied();
        let update = fusion.process_imu(to_fusion_imu(*imu_sample));
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

#[allow(clippy::too_many_arguments)]
fn log_transition_if_needed(
    source: &str,
    t_s: f64,
    pre: Option<EskfState>,
    post: Option<EskfState>,
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
    let (pre_r, pre_p, pre_y) = quat_rpy_deg(
        pre.nominal.q0,
        pre.nominal.q1,
        pre.nominal.q2,
        pre.nominal.q3,
    );
    let (post_r, post_p, post_y) = quat_rpy_deg(
        post.nominal.q0,
        post.nominal.q1,
        post.nominal.q2,
        post.nominal.q3,
    );
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

fn nominal_vehicle_velocity(eskf: &EskfState) -> [f64; 3] {
    let c_n_b = quat_to_rotmat_f64([
        eskf.nominal.q0 as f64,
        eskf.nominal.q1 as f64,
        eskf.nominal.q2 as f64,
        eskf.nominal.q3 as f64,
    ]);
    let c_b_n = transpose3(c_n_b);
    let v_seed = mat_vec(
        c_b_n,
        [
            eskf.nominal.vn as f64,
            eskf.nominal.ve as f64,
            eskf.nominal.vd as f64,
        ],
    );
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
