use std::{fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::datasets::ubx_replay::{UbxReplayConfig, load_generic_replay_with_nav};
use sim::ubxlog::{extract_nav_att, parse_ubx_frames};
use sim::visualizer::math::{
    deg2rad, mat_vec, nearest_master_ms, normalize_heading_deg, rad2deg, rot_zyx,
};
use sim::visualizer::model::{EkfImuSource, NavAttEvent, Trace};
use sim::visualizer::pipeline::build_plot_data;
use sim::visualizer::pipeline::ekf_compare::{EkfCompareConfig, GnssOutageConfig};
use sim::visualizer::pipeline::timebase::build_master_timeline;

#[derive(Parser, Debug)]
#[command(name = "analyze_yaw_reference_window")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long, default_value = "internal", value_parser = parse_misalignment)]
    misalignment: EkfImuSource,
    #[arg(long, default_value_t = 353.0)]
    start_s: f64,
    #[arg(long, default_value_t = 534.0)]
    end_s: f64,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long, default_value_t = 0.3)]
    gnss_pos_r_scale: f64,
    #[arg(long, default_value_t = 3.0)]
    gnss_vel_r_scale: f64,
    #[arg(long, default_value_t = 0.0)]
    gnss_pos_mount_scale: f32,
    #[arg(long, default_value_t = 0.0)]
    gnss_vel_mount_scale: f32,
    #[arg(long, default_value_t = 2.0)]
    r_body_vel: f32,
    #[arg(long, default_value_t = 0.125)]
    gyro_bias_init_sigma_dps: f32,
    #[arg(long, default_value_t = 0)]
    gnss_outage_count: usize,
    #[arg(long, default_value_t = 0.0)]
    gnss_outage_duration_s: f64,
    #[arg(long, default_value_t = 1)]
    gnss_outage_seed: u64,
    #[arg(long, default_value_t = 1)]
    ekf_predict_imu_decimation: usize,
    #[arg(long)]
    ekf_predict_imu_lpf_cutoff_hz: Option<f64>,
}

fn parse_misalignment(s: &str) -> Result<EkfImuSource, String> {
    EkfImuSource::from_cli_value(s)
}

fn trace_by_name<'a>(traces: &'a [Trace], name: &str) -> Result<&'a Trace> {
    traces
        .iter()
        .find(|t| t.name == name)
        .with_context(|| format!("missing trace `{name}`"))
}

fn sample_trace(trace: &Trace, t_s: f64) -> Option<f64> {
    if trace.points.is_empty() {
        return None;
    }
    let idx = trace.points.partition_point(|point| point[0] < t_s);
    let left = trace
        .points
        .get(idx.saturating_sub(1))
        .map(|point| ((point[0] - t_s).abs(), point[1]));
    let right = trace
        .points
        .get(idx)
        .map(|point| ((point[0] - t_s).abs(), point[1]));
    match (left, right) {
        (Some((left_dt, left_value)), Some((right_dt, right_value))) => {
            if right_dt < left_dt {
                Some(right_value)
            } else {
                Some(left_value)
            }
        }
        (Some((_, value)), None) | (None, Some((_, value))) => Some(value),
        (None, None) => None,
    }
}

fn wrap_deg180(mut deg: f64) -> f64 {
    while deg > 180.0 {
        deg -= 360.0;
    }
    while deg <= -180.0 {
        deg += 360.0;
    }
    deg
}

fn transpose3(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn vehicle_velocity_with_yaw(
    vel_ned: [f64; 3],
    yaw_deg: f64,
    roll_deg: f64,
    pitch_deg: f64,
) -> [f64; 3] {
    let c_n_v = transpose3(rot_zyx(
        deg2rad(yaw_deg),
        deg2rad(pitch_deg),
        deg2rad(roll_deg),
    ));
    mat_vec(c_n_v, vel_ned)
}

#[derive(Default)]
struct Stats {
    n: usize,
    start: Option<f64>,
    end: Option<f64>,
    sum: f64,
    sum_abs: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn push(&mut self, v: f64) {
        if self.n == 0 {
            self.start = Some(v);
            self.min = v;
            self.max = v;
        }
        self.end = Some(v);
        self.n += 1;
        self.sum += v;
        self.sum_abs += v.abs();
        self.sum_sq += v * v;
        self.min = self.min.min(v);
        self.max = self.max.max(v);
    }

    fn print(&self, name: &str, is_angle: bool) {
        if self.n == 0 {
            println!("{name}: no samples");
            return;
        }
        let start = self.start.unwrap();
        let end = self.end.unwrap();
        let drift = if is_angle {
            wrap_deg180(end - start)
        } else {
            end - start
        };
        let mean = self.sum / self.n as f64;
        let rms = (self.sum_sq / self.n as f64).sqrt();
        println!(
            "{name}: n={} start={:.6} end={:.6} drift={:.6} mean={:.6} mean_abs={:.6} rms={:.6} min={:.6} max={:.6}",
            self.n,
            start,
            end,
            drift,
            mean,
            self.sum_abs / self.n as f64,
            rms,
            self.min,
            self.max
        );
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut bytes = Vec::new();
    File::open(&args.logfile)
        .with_context(|| format!("failed to open {}", args.logfile.display()))?
        .read_to_end(&mut bytes)
        .context("failed to read log")?;

    let ekf_cfg = EkfCompareConfig {
        r_body_vel: args.r_body_vel,
        gnss_pos_mount_scale: args.gnss_pos_mount_scale,
        gnss_vel_mount_scale: args.gnss_vel_mount_scale,
        gyro_bias_init_sigma_dps: args.gyro_bias_init_sigma_dps,
        gnss_pos_r_scale: args.gnss_pos_r_scale,
        gnss_vel_r_scale: args.gnss_vel_r_scale,
        predict_imu_decimation: args.ekf_predict_imu_decimation.max(1),
        predict_imu_lpf_cutoff_hz: args.ekf_predict_imu_lpf_cutoff_hz,
        ..EkfCompareConfig::default()
    };
    let (data, _has_itow) = build_plot_data(
        &bytes,
        args.max_records,
        args.misalignment,
        ekf_cfg,
        GnssOutageConfig {
            count: args.gnss_outage_count,
            duration_s: args.gnss_outage_duration_s,
            seed: args.gnss_outage_seed,
        },
    );
    let eskf_yaw = trace_by_name(&data.eskf_cmp_att, "ESKF yaw [deg]")?;
    let eskf_mount_yaw = trace_by_name(&data.eskf_misalignment, "ESKF full mount yaw [deg]")?;
    let eskf_roll = trace_by_name(&data.eskf_cmp_att, "ESKF roll [deg]")?;
    let eskf_pitch = trace_by_name(&data.eskf_cmp_att, "ESKF pitch [deg]")?;
    let eskf_vel_forward = trace_by_name(&data.eskf_cmp_vel, "ESKF forward vel [m/s]")?;
    let eskf_vel_lateral = trace_by_name(&data.eskf_cmp_vel, "ESKF lateral vel [m/s]")?;
    let eskf_vel_vertical = trace_by_name(&data.eskf_cmp_vel, "ESKF vertical vel [m/s]")?;
    let ubx_lat_trace = trace_by_name(&data.eskf_cmp_vel, "u-blox lateral vel [m/s]")?;
    let eskf_lat_trace = trace_by_name(&data.eskf_cmp_vel, "ESKF lateral vel [m/s]")?;

    let frames = parse_ubx_frames(&bytes, args.max_records);
    let tl = build_master_timeline(&frames);
    let t0_ms = tl.masters.first().map(|(_, t_ms)| *t_ms).unwrap_or(0.0);

    let replay = load_generic_replay_with_nav(
        &args.logfile,
        UbxReplayConfig {
            gnss_pos_r_scale: args.gnss_pos_r_scale,
            gnss_vel_r_scale: args.gnss_vel_r_scale,
            ..UbxReplayConfig::default()
        },
    )?;

    let mut nav_att_events = Vec::<NavAttEvent>::new();
    for f in &frames {
        if let Some((_itow, roll, pitch, heading)) = extract_nav_att(f) {
            if let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters) {
                nav_att_events.push(NavAttEvent {
                    t_ms,
                    roll_deg: roll,
                    pitch_deg: pitch,
                    heading_deg: normalize_heading_deg(heading),
                });
            }
        }
    }
    nav_att_events.sort_by(|a, b| a.t_ms.total_cmp(&b.t_ms));

    let mut nav_att_idx = 0usize;
    let mut current_nav_att: Option<NavAttEvent> = None;

    let mut nav_att_vs_head_veh = Stats::default();
    let mut nav_att_vs_course = Stats::default();
    let mut head_veh_vs_course = Stats::default();
    let mut eskf_vs_nav_att = Stats::default();
    let mut eskf_vs_head_veh = Stats::default();
    let mut eskf_vs_course = Stats::default();
    let mut eskf_course_vs_nav_att = Stats::default();
    let mut eskf_course_vs_course = Stats::default();
    let mut eskf_course_minus_yaw = Stats::default();
    let mut eskf_mount_yaw_vs_nav_att = Stats::default();
    let mut ubx_lat_nav_att = Stats::default();
    let mut ubx_lat_head_veh = Stats::default();
    let mut ubx_lat_course = Stats::default();
    let mut eskf_lat = Stats::default();
    let mut speed_h = Stats::default();

    let mut top_nav_att_vs_course = Vec::<(f64, f64, f64, f64, f64, f64, f64)>::new();

    for (t_ms, nav) in &replay.nav_events {
        let t_s = (*t_ms - t0_ms) * 1.0e-3;
        if t_s < args.start_s || t_s > args.end_s {
            continue;
        }
        while nav_att_idx < nav_att_events.len() && nav_att_events[nav_att_idx].t_ms <= *t_ms {
            current_nav_att = Some(nav_att_events[nav_att_idx]);
            nav_att_idx += 1;
        }
        let Some(att) = current_nav_att else {
            continue;
        };

        let course_deg = normalize_heading_deg(rad2deg(nav.vel_e_mps.atan2(nav.vel_n_mps)));
        let head_veh_deg = normalize_heading_deg(nav.heading_vehicle_deg);
        let nav_att_deg = normalize_heading_deg(att.heading_deg);
        let vel_ned = [nav.vel_n_mps, nav.vel_e_mps, nav.vel_d_mps];

        let body_nav_att =
            vehicle_velocity_with_yaw(vel_ned, nav_att_deg, att.roll_deg, att.pitch_deg);
        let body_head_veh =
            vehicle_velocity_with_yaw(vel_ned, head_veh_deg, att.roll_deg, att.pitch_deg);
        let body_course =
            vehicle_velocity_with_yaw(vel_ned, course_deg, att.roll_deg, att.pitch_deg);

        let Some(eskf_yaw_deg) = sample_trace(eskf_yaw, t_s) else {
            continue;
        };
        let Some(eskf_mount_yaw_deg) = sample_trace(eskf_mount_yaw, t_s) else {
            continue;
        };
        let Some(eskf_roll_deg) = sample_trace(eskf_roll, t_s) else {
            continue;
        };
        let Some(eskf_pitch_deg) = sample_trace(eskf_pitch, t_s) else {
            continue;
        };
        let Some(eskf_vx) = sample_trace(eskf_vel_forward, t_s) else {
            continue;
        };
        let Some(eskf_vy) = sample_trace(eskf_vel_lateral, t_s) else {
            continue;
        };
        let Some(eskf_vz) = sample_trace(eskf_vel_vertical, t_s) else {
            continue;
        };
        let Some(eskf_lat_v) = sample_trace(eskf_lat_trace, t_s) else {
            continue;
        };
        let Some(ubx_lat_v) = sample_trace(ubx_lat_trace, t_s) else {
            continue;
        };

        let c_n_v_eskf = rot_zyx(
            deg2rad(eskf_yaw_deg),
            deg2rad(eskf_pitch_deg),
            deg2rad(eskf_roll_deg),
        );
        let eskf_vel_ned = mat_vec(c_n_v_eskf, [eskf_vx, eskf_vy, eskf_vz]);
        let eskf_course_deg =
            normalize_heading_deg(rad2deg(eskf_vel_ned[1].atan2(eskf_vel_ned[0])));

        nav_att_vs_head_veh.push(wrap_deg180(nav_att_deg - head_veh_deg));
        nav_att_vs_course.push(wrap_deg180(nav_att_deg - course_deg));
        head_veh_vs_course.push(wrap_deg180(head_veh_deg - course_deg));
        eskf_vs_nav_att.push(wrap_deg180(eskf_yaw_deg - nav_att_deg));
        eskf_vs_head_veh.push(wrap_deg180(eskf_yaw_deg - head_veh_deg));
        eskf_vs_course.push(wrap_deg180(eskf_yaw_deg - course_deg));
        eskf_course_vs_nav_att.push(wrap_deg180(eskf_course_deg - nav_att_deg));
        eskf_course_vs_course.push(wrap_deg180(eskf_course_deg - course_deg));
        eskf_course_minus_yaw.push(wrap_deg180(eskf_course_deg - eskf_yaw_deg));
        eskf_mount_yaw_vs_nav_att.push(wrap_deg180(eskf_mount_yaw_deg - nav_att_deg));
        ubx_lat_nav_att.push(body_nav_att[1]);
        ubx_lat_head_veh.push(body_head_veh[1]);
        ubx_lat_course.push(body_course[1]);
        eskf_lat.push(eskf_lat_v);
        speed_h.push(nav.vel_n_mps.hypot(nav.vel_e_mps));

        let abs_nav_att_vs_course = wrap_deg180(nav_att_deg - course_deg).abs();
        top_nav_att_vs_course.push((
            abs_nav_att_vs_course,
            t_s,
            nav_att_deg,
            head_veh_deg,
            course_deg,
            body_nav_att[1],
            ubx_lat_v,
        ));
    }

    top_nav_att_vs_course.sort_by(|a, b| b.0.total_cmp(&a.0));

    println!(
        "window=[{:.1}, {:.1}] config: gnss_pos_r_scale={:.3} gnss_vel_r_scale={:.3} gnss_pos_mount_scale={:.3} gnss_vel_mount_scale={:.3} r_body_vel={:.1} gyro_bias_init_sigma_dps={:.3}",
        args.start_s,
        args.end_s,
        args.gnss_pos_r_scale,
        args.gnss_vel_r_scale,
        args.gnss_pos_mount_scale,
        args.gnss_vel_mount_scale,
        args.r_body_vel,
        args.gyro_bias_init_sigma_dps,
    );
    speed_h.print("speed_h_mps", false);
    nav_att_vs_head_veh.print("nav_att_minus_heading_vehicle_deg", true);
    nav_att_vs_course.print("nav_att_minus_course_deg", true);
    head_veh_vs_course.print("heading_vehicle_minus_course_deg", true);
    eskf_vs_nav_att.print("eskf_yaw_minus_nav_att_deg", true);
    eskf_vs_head_veh.print("eskf_yaw_minus_heading_vehicle_deg", true);
    eskf_vs_course.print("eskf_yaw_minus_course_deg", true);
    eskf_course_vs_nav_att.print("eskf_course_minus_nav_att_deg", true);
    eskf_course_vs_course.print("eskf_course_minus_course_deg", true);
    eskf_course_minus_yaw.print("eskf_course_minus_yaw_deg", true);
    eskf_mount_yaw_vs_nav_att.print("eskf_mount_yaw_minus_nav_att_deg", true);
    ubx_lat_nav_att.print("ubx_body_lateral_nav_att_mps", false);
    ubx_lat_head_veh.print("ubx_body_lateral_heading_vehicle_mps", false);
    ubx_lat_course.print("ubx_body_lateral_course_mps", false);
    eskf_lat.print("eskf_body_lateral_mps", false);

    println!("largest_nav_att_vs_course:");
    for (abs_diff, t_s, nav_att_deg, head_veh_deg, course_deg, body_nav_att_y, ubx_lat_v) in
        top_nav_att_vs_course.into_iter().take(10)
    {
        println!(
            "  t={:.3}s |nav_att-course|={:.3} nav_att={:.3} heading_vehicle={:.3} course={:.3} body_lat_nav_att={:.3} ubx_trace_lat={:.3}",
            t_s, abs_diff, nav_att_deg, head_veh_deg, course_deg, body_nav_att_y, ubx_lat_v
        );
    }

    Ok(())
}
