use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::eskf_types::{EskfImuDelta, EskfState};
use sensor_fusion::fusion::{FusionVehicleSpeedDirection, FusionVehicleSpeedSample, SensorFusion};
use sensor_fusion::generated_eskf;
use sim::datasets::generic_replay::{
    fusion_gnss_sample as to_fusion_gnss, fusion_imu_sample as to_fusion_imu,
};
use sim::datasets::ubx_replay::{UbxReplayConfig, load_generic_replay_with_nav};
use sim::eval::gnss_ins::quat_angle_deg;
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::ubxlog::{UbxFrame, extract_esf_alg, extract_nav_att, parse_ubx_frames};
use sim::visualizer::math::{nearest_master_ms, normalize_heading_deg, quat_rpy_deg, rad2deg};
use sim::visualizer::pipeline::align_replay::{
    esf_alg_flu_to_frd_mount_quat, frd_mount_quat_to_esf_alg_flu_quat, quat_rpy_alg_deg,
};
use sim::visualizer::pipeline::timebase::{MasterTimeline, build_master_timeline};

const CUE_NAMES: [&str; 11] = [
    "gps_pos_ne",
    "gps_vel_ne",
    "zero_vel_ne",
    "body_speed_x",
    "body_vel_y",
    "body_vel_z",
    "stationary_x",
    "stationary_y",
    "gps_pos_d",
    "gps_vel_d",
    "zero_vel_d",
];

#[derive(Parser, Debug)]
#[command(name = "analyze_real_drift_window")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long, default_value_t = 353.0)]
    start_s: f64,
    #[arg(long, default_value_t = 534.0)]
    end_s: f64,
    #[arg(long, default_value_t = 0.3)]
    gnss_pos_r_scale: f64,
    #[arg(long, default_value_t = 3.0)]
    gnss_vel_r_scale: f64,
    #[arg(long, default_value_t = 0.0)]
    gnss_time_shift_ms: f64,
    #[arg(long, default_value_t = 2.0)]
    r_body_vel: f32,
    #[arg(long, default_value_t = 0.0)]
    gnss_pos_mount_scale: f32,
    #[arg(long, default_value_t = 0.0)]
    gnss_vel_mount_scale: f32,
    #[arg(long, default_value_t = 0.125)]
    gyro_bias_init_sigma_dps: f32,
    #[arg(long, default_value_t = 0.20)]
    accel_bias_init_sigma_mps2: f32,
    #[arg(long, default_value_t = 0.002e-9)]
    accel_bias_rw_var: f32,
    #[arg(long, default_value_t = 0.04)]
    r_vehicle_speed: f32,
    #[arg(long, default_value_t = false)]
    fuse_gnss_speed_as_vehicle_speed: bool,
    #[arg(long, default_value_t = 0.0)]
    r_zero_vel: f32,
    #[arg(long, default_value_t = 0.0)]
    r_stationary_accel: f32,
    #[arg(long, default_value_t = 1.0e-6)]
    mount_align_rw_var: f32,
    #[arg(long, default_value_t = 0.01)]
    mount_update_min_scale: f32,
    #[arg(long, default_value_t = 300.0)]
    mount_update_ramp_time_s: f32,
    #[arg(long, default_value_t = 0.05)]
    mount_update_innovation_gate_mps: f32,
    #[arg(long, default_value_t = false)]
    analysis_zero_mount_cross_cov: bool,
    #[arg(long, default_value_t = false)]
    analysis_freeze_mount: bool,
    #[arg(long, default_value_t = 0.001)]
    analysis_mount_sigma_deg: f32,
    #[arg(long, default_value_t = false)]
    timeline: bool,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "76.236,82,100,117.236,120,180,220,320"
    )]
    timeline_events_s: Vec<f64>,
}

#[derive(Clone, Copy)]
struct NavAttEvent {
    t_s: f64,
    pitch_deg: f64,
    heading_deg: f64,
}

#[derive(Clone, Copy)]
struct AlgMountEvent {
    t_s: f64,
    q_vb: [f64; 4],
    pitch_deg: f64,
    yaw_deg: f64,
}

#[derive(Clone, Copy, Default)]
struct WindowSnapshot {
    t_s: f64,
    pitch_deg: f64,
    yaw_deg: f64,
    course_deg: f64,
    pitch_minus_nav_att_deg: f64,
    yaw_minus_nav_att_deg: f64,
    yaw_minus_course_deg: f64,
    course_minus_course_deg: f64,
    vel_vehicle_y_mps: f64,
    vel_vehicle_z_mps: f64,
    vn_err_mps: f64,
    ve_err_mps: f64,
    vd_err_mps: f64,
    speed_err_mps: f64,
    mount_pitch_err_deg: f64,
    mount_yaw_err_deg: f64,
    mount_quat_err_deg: f64,
    bgx_dps: f64,
    bgy_dps: f64,
    bgz_dps: f64,
    bax_mps2: f64,
    bay_mps2: f64,
    baz_mps2: f64,
    theta_z_var: f64,
    v_n_var: f64,
    v_e_var: f64,
    mount_z_var: f64,
    type_counts: [u32; 11],
    sum_dx_yaw_deg: [f64; 11],
    sum_abs_dx_yaw_deg: [f64; 11],
    sum_abs_dx_pitch_deg: [f64; 11],
    sum_abs_dx_vel_h_mps: [f64; 11],
    sum_dx_gyro_bias_z_dps: [f64; 11],
    sum_abs_dx_gyro_bias_z_dps: [f64; 11],
    sum_dx_mount_yaw_deg: [f64; 11],
    sum_abs_innovation: [f64; 11],
    sum_nis: [f64; 11],
    sum_abs_dx_mount_yaw_deg: [f64; 11],
}

#[derive(Clone, Copy)]
struct TimelineRow {
    target_t_s: f64,
    snap: WindowSnapshot,
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

#[derive(Default)]
struct GnssTrackPitchSplit {
    pos_along_innov_m: Stats,
    pos_cross_innov_m: Stats,
    vel_along_innov_mps: Stats,
    vel_cross_innov_mps: Stats,
    pos_along_pitch_dx_deg: Stats,
    pos_cross_pitch_dx_deg: Stats,
    vel_along_pitch_dx_deg: Stats,
    vel_cross_pitch_dx_deg: Stats,
    pos_along_bax_dx_mps2: Stats,
    pos_cross_bax_dx_mps2: Stats,
    vel_along_bax_dx_mps2: Stats,
    vel_cross_bax_dx_mps2: Stats,
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
        println!(
            "{name}: n={} start={:.6} end={:.6} drift={:.6} mean={:.6} mean_abs={:.6} rms={:.6} min={:.6} max={:.6}",
            self.n,
            start,
            end,
            drift,
            self.sum / self.n as f64,
            self.sum_abs / self.n as f64,
            (self.sum_sq / self.n as f64).sqrt(),
            self.min,
            self.max
        );
    }
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

    let nav_att_events = collect_nav_att_events(&frames, &tl);
    let alg_events = collect_alg_mount_events(&frames, &tl);
    let mut replay = load_generic_replay_with_nav(
        &args.logfile,
        UbxReplayConfig {
            gnss_pos_r_scale: args.gnss_pos_r_scale,
            gnss_vel_r_scale: args.gnss_vel_r_scale,
            ..UbxReplayConfig::default()
        },
    )?;
    if args.gnss_time_shift_ms != 0.0 {
        let dt_s = args.gnss_time_shift_ms * 1.0e-3;
        replay.gnss_samples.retain_mut(|sample| {
            let shifted_t = sample.t_s + dt_s;
            if shifted_t.is_finite() && shifted_t >= 0.0 {
                sample.t_s = shifted_t;
                true
            } else {
                false
            }
        });
        replay.gnss_samples.sort_by(|a, b| a.t_s.total_cmp(&b.t_s));
    }

    let mut fusion = SensorFusion::new();
    fusion.set_r_body_vel(args.r_body_vel);
    fusion.set_gnss_pos_mount_scale(args.gnss_pos_mount_scale);
    fusion.set_gnss_vel_mount_scale(args.gnss_vel_mount_scale);
    fusion.set_gyro_bias_init_sigma_radps(args.gyro_bias_init_sigma_dps.to_radians());
    fusion.set_accel_bias_init_sigma_mps2(args.accel_bias_init_sigma_mps2);
    fusion.set_accel_bias_rw_var(args.accel_bias_rw_var);
    fusion.set_r_vehicle_speed(args.r_vehicle_speed);
    fusion.set_r_zero_vel(args.r_zero_vel);
    fusion.set_r_stationary_accel(args.r_stationary_accel);
    fusion.set_mount_align_rw_var(args.mount_align_rw_var);
    fusion.set_mount_update_min_scale(args.mount_update_min_scale);
    fusion.set_mount_update_ramp_time_s(args.mount_update_ramp_time_s);
    fusion.set_mount_update_innovation_gate_mps(args.mount_update_innovation_gate_mps);

    let mut first_snapshot = None::<WindowSnapshot>;
    let mut last_snapshot = None::<WindowSnapshot>;

    let mut pitch_vs_nav_att = Stats::default();
    let mut yaw_vs_nav_att = Stats::default();
    let mut yaw_vs_course = Stats::default();
    let mut course_vs_course = Stats::default();
    let mut yaw_minus_course = Stats::default();
    let mut vel_vehicle_y = Stats::default();
    let mut vel_vehicle_z = Stats::default();
    let mut vn_err = Stats::default();
    let mut ve_err = Stats::default();
    let mut vd_err = Stats::default();
    let mut speed_err = Stats::default();
    let mut mount_pitch_err = Stats::default();
    let mut mount_yaw_err = Stats::default();
    let mut mount_quat_err = Stats::default();
    let mut theta_z_var = Stats::default();
    let mut mount_z_var = Stats::default();
    let mut f_vn_theta_y = Stats::default();
    let mut f_ve_theta_y = Stats::default();
    let mut f_vn_bax = Stats::default();
    let mut f_ve_bax = Stats::default();
    let mut p_theta_y = Stats::default();
    let mut p_bax = Stats::default();
    let mut p_theta_y_vn = Stats::default();
    let mut p_theta_y_ve = Stats::default();
    let mut p_bax_vn = Stats::default();
    let mut p_bax_ve = Stats::default();
    let mut gnss_track_pitch = GnssTrackPitchSplit::default();
    let mut latched_mount_qcs: Option<[f32; 4]> = None;
    let mut timeline_events_s = args.timeline_events_s.clone();
    timeline_events_s.retain(|t| t.is_finite());
    timeline_events_s.sort_by(|a, b| a.total_cmp(b));
    timeline_events_s.dedup_by(|a, b| (*a - *b).abs() < 1.0e-6);
    let mut next_timeline_event = 0usize;
    let mut timeline_rows = Vec::<TimelineRow>::new();

    let mut prev_imu_t_s: Option<f64> = None;
    for_each_event(
        &replay.imu_samples,
        &replay.gnss_samples,
        |event| match event {
            ReplayEvent::Imu(_, sample) => {
                if sample.t_s >= args.start_s
                    && sample.t_s <= args.end_s
                    && let Some(prev_t_s) = prev_imu_t_s
                    && let Some(eskf) = fusion.eskf()
                    && let Some(q_vb) = fusion.eskf_mount_q_vb().or_else(|| fusion.mount_q_vb())
                {
                    let dt_s = sample.t_s - prev_t_s;
                    if (0.001..=0.05).contains(&dt_s) {
                        let c_bv = quat_to_rotmat_f64(q_vb.map(|v| v as f64));
                        let c_vb = transpose3(c_bv);
                        let gyro_vehicle = mat_vec(c_vb, sample.gyro_radps);
                        let accel_vehicle = mat_vec(c_vb, sample.accel_mps2);
                        let (f, _) = generated_eskf::error_transition(
                            &eskf.nominal,
                            EskfImuDelta {
                                dax: (gyro_vehicle[0] * dt_s) as f32,
                                day: (gyro_vehicle[1] * dt_s) as f32,
                                daz: (gyro_vehicle[2] * dt_s) as f32,
                                dvx: (accel_vehicle[0] * dt_s) as f32,
                                dvy: (accel_vehicle[1] * dt_s) as f32,
                                dvz: (accel_vehicle[2] * dt_s) as f32,
                                dt: dt_s as f32,
                            },
                        );
                        f_vn_theta_y.push(f[3][1] as f64);
                        f_ve_theta_y.push(f[4][1] as f64);
                        f_vn_bax.push(f[3][12] as f64);
                        f_ve_bax.push(f[4][12] as f64);
                    }
                }
                let _ = fusion.process_imu(to_fusion_imu(*sample));
                apply_mount_ablation(&args, &mut fusion, &mut latched_mount_qcs);
                prev_imu_t_s = Some(sample.t_s);
            }
            ReplayEvent::Gnss(_, sample) => {
                if sample.t_s >= args.start_s
                    && sample.t_s <= args.end_s
                    && let Some(eskf) = fusion.eskf()
                    && let Some(nav_att) = sample_nearest_nav_att(&nav_att_events, sample.t_s)
                    && let Some(anchor_lla) = fusion.anchor_lla_debug()
                {
                    accumulate_gnss_track_pitch_split(
                        &mut gnss_track_pitch,
                        eskf,
                        sample,
                        nav_att.heading_deg,
                        anchor_lla.map(|v| v as f64),
                    );
                }
                let _ = fusion.process_gnss(to_fusion_gnss(*sample));
                apply_mount_ablation(&args, &mut fusion, &mut latched_mount_qcs);
                if args.fuse_gnss_speed_as_vehicle_speed {
                    let speed_mps = sample.vel_ned_mps[0].hypot(sample.vel_ned_mps[1]) as f32;
                    let _ = fusion.process_vehicle_speed(FusionVehicleSpeedSample {
                        t_s: sample.t_s as f32,
                        speed_mps,
                        direction: FusionVehicleSpeedDirection::Forward,
                    });
                    apply_mount_ablation(&args, &mut fusion, &mut latched_mount_qcs);
                }
                let t_s = sample.t_s;
                if t_s < args.start_s || t_s > args.end_s {
                    return;
                }
                let Some(eskf) = fusion.eskf() else {
                    return;
                };
                let Some(nav_att) = sample_nearest_nav_att(&nav_att_events, t_s) else {
                    return;
                };
                let Some(alg) = sample_nearest_alg_mount(&alg_events, t_s) else {
                    return;
                };
                let snap = make_snapshot(t_s, sample, nav_att, alg, &fusion, eskf);
                while next_timeline_event < timeline_events_s.len()
                    && t_s >= timeline_events_s[next_timeline_event]
                {
                    timeline_rows.push(TimelineRow {
                        target_t_s: timeline_events_s[next_timeline_event],
                        snap,
                    });
                    next_timeline_event += 1;
                }
                pitch_vs_nav_att.push(snap.pitch_minus_nav_att_deg);
                yaw_vs_nav_att.push(snap.yaw_minus_nav_att_deg);
                yaw_vs_course.push(snap.yaw_minus_course_deg);
                course_vs_course.push(snap.course_minus_course_deg);
                yaw_minus_course.push(wrap_deg180(snap.yaw_deg - snap.course_deg));
                vel_vehicle_y.push(snap.vel_vehicle_y_mps);
                vel_vehicle_z.push(snap.vel_vehicle_z_mps);
                vn_err.push(snap.vn_err_mps);
                ve_err.push(snap.ve_err_mps);
                vd_err.push(snap.vd_err_mps);
                speed_err.push(snap.speed_err_mps);
                mount_pitch_err.push(snap.mount_pitch_err_deg);
                mount_yaw_err.push(snap.mount_yaw_err_deg);
                mount_quat_err.push(snap.mount_quat_err_deg);
                theta_z_var.push(snap.theta_z_var);
                mount_z_var.push(snap.mount_z_var);
                p_theta_y.push(eskf.p[1][1] as f64);
                p_bax.push(eskf.p[12][12] as f64);
                p_theta_y_vn.push(eskf.p[1][3] as f64);
                p_theta_y_ve.push(eskf.p[1][4] as f64);
                p_bax_vn.push(eskf.p[12][3] as f64);
                p_bax_ve.push(eskf.p[12][4] as f64);
                first_snapshot.get_or_insert(snap);
                last_snapshot = Some(snap);
            }
        },
    );

    let Some(first) = first_snapshot else {
        bail!("no GNSS-aligned ESKF samples in window");
    };
    let Some(last) = last_snapshot else {
        bail!("no end snapshot in window");
    };

    if args.timeline {
        print_timeline(&timeline_rows);
        return Ok(());
    }

    println!(
        "window=[{:.3}, {:.3}] config: gnss_pos_r_scale={:.3} gnss_vel_r_scale={:.3} gnss_time_shift_ms={:.1} r_body_vel={:.3} gnss_pos_mount_scale={:.3} gnss_vel_mount_scale={:.3} gyro_bias_init_sigma_dps={:.3} accel_bias_init_sigma_mps2={:.3} accel_bias_rw_var={:.6e} r_vehicle_speed={:.3} fuse_gnss_speed_as_vehicle_speed={} r_zero_vel={:.3} r_stationary_accel={:.3} mount_align_rw_var={:.6e} mount_update_min_scale={:.3} mount_update_ramp_time_s={:.3} mount_update_innovation_gate_mps={:.3} analysis_zero_mount_cross_cov={} analysis_freeze_mount={} analysis_mount_sigma_deg={:.6}",
        args.start_s,
        args.end_s,
        args.gnss_pos_r_scale,
        args.gnss_vel_r_scale,
        args.gnss_time_shift_ms,
        args.r_body_vel,
        args.gnss_pos_mount_scale,
        args.gnss_vel_mount_scale,
        args.gyro_bias_init_sigma_dps,
        args.accel_bias_init_sigma_mps2,
        args.accel_bias_rw_var,
        args.r_vehicle_speed,
        args.fuse_gnss_speed_as_vehicle_speed,
        args.r_zero_vel,
        args.r_stationary_accel,
        args.mount_align_rw_var,
        args.mount_update_min_scale,
        args.mount_update_ramp_time_s,
        args.mount_update_innovation_gate_mps,
        args.analysis_zero_mount_cross_cov,
        args.analysis_freeze_mount,
        args.analysis_mount_sigma_deg,
    );
    println!(
        "window_endpoints: start_t={:.3} end_t={:.3}",
        first.t_s, last.t_s
    );
    pitch_vs_nav_att.print("pitch_minus_nav_att_deg", true);
    yaw_vs_nav_att.print("yaw_minus_nav_att_deg", true);
    yaw_vs_course.print("yaw_minus_gnss_course_deg", true);
    course_vs_course.print("nominal_course_minus_gnss_course_deg", true);
    yaw_minus_course.print("yaw_minus_nominal_course_deg", true);
    vel_vehicle_y.print("vehicle_lateral_vel_mps", false);
    vel_vehicle_z.print("vehicle_vertical_vel_mps", false);
    vn_err.print("vn_err_mps", false);
    ve_err.print("ve_err_mps", false);
    vd_err.print("vd_err_mps", false);
    speed_err.print("horizontal_speed_err_mps", false);
    mount_pitch_err.print("mount_pitch_err_deg", true);
    mount_yaw_err.print("mount_yaw_err_deg", true);
    mount_quat_err.print("mount_quat_err_deg", false);
    theta_z_var.print("theta_z_var", false);
    mount_z_var.print("mount_z_var", false);
    f_vn_theta_y.print("f_vn_theta_y", false);
    f_ve_theta_y.print("f_ve_theta_y", false);
    f_vn_bax.print("f_vn_bax", false);
    f_ve_bax.print("f_ve_bax", false);
    p_theta_y.print("p_theta_y", false);
    p_bax.print("p_bax", false);
    p_theta_y_vn.print("p_theta_y_vn", false);
    p_theta_y_ve.print("p_theta_y_ve", false);
    p_bax_vn.print("p_bax_vn", false);
    p_bax_ve.print("p_bax_ve", false);
    println!("gnss_track_pitch_split:");
    gnss_track_pitch
        .pos_along_innov_m
        .print("  pos_along_innov_m", false);
    gnss_track_pitch
        .pos_cross_innov_m
        .print("  pos_cross_innov_m", false);
    gnss_track_pitch
        .vel_along_innov_mps
        .print("  vel_along_innov_mps", false);
    gnss_track_pitch
        .vel_cross_innov_mps
        .print("  vel_cross_innov_mps", false);
    gnss_track_pitch
        .pos_along_pitch_dx_deg
        .print("  pos_along_pitch_dx_deg", false);
    gnss_track_pitch
        .pos_cross_pitch_dx_deg
        .print("  pos_cross_pitch_dx_deg", false);
    gnss_track_pitch
        .vel_along_pitch_dx_deg
        .print("  vel_along_pitch_dx_deg", false);
    gnss_track_pitch
        .vel_cross_pitch_dx_deg
        .print("  vel_cross_pitch_dx_deg", false);
    gnss_track_pitch
        .pos_along_bax_dx_mps2
        .print("  pos_along_bax_dx_mps2", false);
    gnss_track_pitch
        .pos_cross_bax_dx_mps2
        .print("  pos_cross_bax_dx_mps2", false);
    gnss_track_pitch
        .vel_along_bax_dx_mps2
        .print("  vel_along_bax_dx_mps2", false);
    gnss_track_pitch
        .vel_cross_bax_dx_mps2
        .print("  vel_cross_bax_dx_mps2", false);

    println!("snapshot_drift:");
    print_snapshot_delta("pitch_deg", first.pitch_deg, last.pitch_deg, true);
    print_snapshot_delta("yaw_deg", first.yaw_deg, last.yaw_deg, true);
    print_snapshot_delta("course_deg", first.course_deg, last.course_deg, true);
    print_snapshot_delta(
        "pitch_minus_nav_att_deg",
        first.pitch_minus_nav_att_deg,
        last.pitch_minus_nav_att_deg,
        true,
    );
    print_snapshot_delta(
        "yaw_minus_nav_att_deg",
        first.yaw_minus_nav_att_deg,
        last.yaw_minus_nav_att_deg,
        true,
    );
    print_snapshot_delta(
        "yaw_minus_gnss_course_deg",
        first.yaw_minus_course_deg,
        last.yaw_minus_course_deg,
        true,
    );
    print_snapshot_delta(
        "nominal_course_minus_gnss_course_deg",
        first.course_minus_course_deg,
        last.course_minus_course_deg,
        true,
    );
    print_snapshot_delta(
        "vehicle_lateral_vel_mps",
        first.vel_vehicle_y_mps,
        last.vel_vehicle_y_mps,
        false,
    );
    print_snapshot_delta(
        "vehicle_vertical_vel_mps",
        first.vel_vehicle_z_mps,
        last.vel_vehicle_z_mps,
        false,
    );
    print_snapshot_delta("vn_err_mps", first.vn_err_mps, last.vn_err_mps, false);
    print_snapshot_delta("ve_err_mps", first.ve_err_mps, last.ve_err_mps, false);
    print_snapshot_delta("vd_err_mps", first.vd_err_mps, last.vd_err_mps, false);
    print_snapshot_delta(
        "horizontal_speed_err_mps",
        first.speed_err_mps,
        last.speed_err_mps,
        false,
    );
    print_snapshot_delta(
        "mount_pitch_err_deg",
        first.mount_pitch_err_deg,
        last.mount_pitch_err_deg,
        true,
    );
    print_snapshot_delta(
        "mount_yaw_err_deg",
        first.mount_yaw_err_deg,
        last.mount_yaw_err_deg,
        true,
    );
    print_snapshot_delta(
        "mount_quat_err_deg",
        first.mount_quat_err_deg,
        last.mount_quat_err_deg,
        false,
    );
    print_snapshot_delta("bgx_dps", first.bgx_dps, last.bgx_dps, false);
    print_snapshot_delta("bgy_dps", first.bgy_dps, last.bgy_dps, false);
    print_snapshot_delta("bgz_dps", first.bgz_dps, last.bgz_dps, false);
    print_snapshot_delta("bax_mps2", first.bax_mps2, last.bax_mps2, false);
    print_snapshot_delta("bay_mps2", first.bay_mps2, last.bay_mps2, false);
    print_snapshot_delta("baz_mps2", first.baz_mps2, last.baz_mps2, false);
    print_snapshot_delta("theta_z_var", first.theta_z_var, last.theta_z_var, false);
    print_snapshot_delta("v_n_var", first.v_n_var, last.v_n_var, false);
    print_snapshot_delta("v_e_var", first.v_e_var, last.v_e_var, false);
    print_snapshot_delta("mount_z_var", first.mount_z_var, last.mount_z_var, false);

    println!("update_diag_window_delta:");
    for (i, name) in CUE_NAMES.iter().enumerate() {
        let count_delta = last.type_counts[i].saturating_sub(first.type_counts[i]);
        let innov_abs_delta = last.sum_abs_innovation[i] - first.sum_abs_innovation[i];
        let yaw_dx_delta = last.sum_dx_yaw_deg[i] - first.sum_dx_yaw_deg[i];
        let yaw_dx_abs_delta = last.sum_abs_dx_yaw_deg[i] - first.sum_abs_dx_yaw_deg[i];
        let pitch_dx_abs_delta = last.sum_abs_dx_pitch_deg[i] - first.sum_abs_dx_pitch_deg[i];
        let mount_yaw_dx_delta = last.sum_dx_mount_yaw_deg[i] - first.sum_dx_mount_yaw_deg[i];
        let mount_yaw_dx_abs_delta =
            last.sum_abs_dx_mount_yaw_deg[i] - first.sum_abs_dx_mount_yaw_deg[i];
        println!(
            "  cue={} count_delta={} innov_abs_delta={:.6} yaw_dx_delta_deg={:.6} yaw_dx_abs_delta_deg={:.6} pitch_dx_abs_delta_deg={:.6} mount_yaw_dx_delta_deg={:.6} mount_yaw_dx_abs_delta_deg={:.6}",
            name,
            count_delta,
            innov_abs_delta,
            yaw_dx_delta,
            yaw_dx_abs_delta,
            pitch_dx_abs_delta,
            mount_yaw_dx_delta,
            mount_yaw_dx_abs_delta
        );
    }

    Ok(())
}

fn make_snapshot(
    t_s: f64,
    gnss: &sim::datasets::generic_replay::GenericGnssSample,
    nav_att: NavAttEvent,
    alg: AlgMountEvent,
    fusion: &SensorFusion,
    eskf: &EskfState,
) -> WindowSnapshot {
    let q_seed_frame = [
        eskf.nominal.q0 as f64,
        eskf.nominal.q1 as f64,
        eskf.nominal.q2 as f64,
        eskf.nominal.q3 as f64,
    ];
    let q_cs = [
        eskf.nominal.qcs0 as f64,
        eskf.nominal.qcs1 as f64,
        eskf.nominal.qcs2 as f64,
        eskf.nominal.qcs3 as f64,
    ];
    let q_vehicle = quat_mul(q_seed_frame, quat_conj(q_cs));
    let (_roll_deg, pitch_deg, yaw_deg) = quat_rpy_deg(
        q_vehicle[0] as f32,
        q_vehicle[1] as f32,
        q_vehicle[2] as f32,
        q_vehicle[3] as f32,
    );
    let course_deg = normalize_heading_deg(rad2deg(
        (eskf.nominal.ve as f64).atan2(eskf.nominal.vn as f64),
    ));
    let c_n_vehicle = quat_to_rotmat_f64(q_vehicle);
    let vel_vehicle = mat_vec(
        transpose3(c_n_vehicle),
        [
            eskf.nominal.vn as f64,
            eskf.nominal.ve as f64,
            eskf.nominal.vd as f64,
        ],
    );
    let q_seed = fusion
        .eskf_mount_q_vb()
        .or_else(|| fusion.mount_q_vb())
        .map(|q| q.map(|v| v as f64))
        .unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let q_full_vb = quat_mul(q_seed, quat_conj(q_cs));
    let q_full_flu = frd_mount_quat_to_esf_alg_flu_quat(q_full_vb);
    let (_, mount_pitch_deg, mount_yaw_deg) =
        quat_rpy_alg_deg(q_full_flu[0], q_full_flu[1], q_full_flu[2], q_full_flu[3]);
    WindowSnapshot {
        t_s,
        pitch_deg,
        yaw_deg,
        course_deg,
        pitch_minus_nav_att_deg: wrap_deg180(pitch_deg - nav_att.pitch_deg),
        yaw_minus_nav_att_deg: wrap_deg180(yaw_deg - nav_att.heading_deg),
        yaw_minus_course_deg: wrap_deg180(
            yaw_deg
                - normalize_heading_deg(rad2deg(gnss.vel_ned_mps[1].atan2(gnss.vel_ned_mps[0]))),
        ),
        course_minus_course_deg: wrap_deg180(
            course_deg
                - normalize_heading_deg(rad2deg(gnss.vel_ned_mps[1].atan2(gnss.vel_ned_mps[0]))),
        ),
        vel_vehicle_y_mps: vel_vehicle[1],
        vel_vehicle_z_mps: vel_vehicle[2],
        vn_err_mps: eskf.nominal.vn as f64 - gnss.vel_ned_mps[0],
        ve_err_mps: eskf.nominal.ve as f64 - gnss.vel_ned_mps[1],
        vd_err_mps: eskf.nominal.vd as f64 - gnss.vel_ned_mps[2],
        speed_err_mps: (eskf.nominal.vn as f64).hypot(eskf.nominal.ve as f64)
            - gnss.vel_ned_mps[0].hypot(gnss.vel_ned_mps[1]),
        mount_pitch_err_deg: wrap_deg180(mount_pitch_deg - alg.pitch_deg),
        mount_yaw_err_deg: wrap_deg180(mount_yaw_deg - alg.yaw_deg),
        mount_quat_err_deg: quat_angle_deg(q_full_vb, alg.q_vb),
        bgx_dps: rad2deg(eskf.nominal.bgx as f64),
        bgy_dps: rad2deg(eskf.nominal.bgy as f64),
        bgz_dps: rad2deg(eskf.nominal.bgz as f64),
        bax_mps2: eskf.nominal.bax as f64,
        bay_mps2: eskf.nominal.bay as f64,
        baz_mps2: eskf.nominal.baz as f64,
        theta_z_var: eskf.p[2][2] as f64,
        v_n_var: eskf.p[3][3] as f64,
        v_e_var: eskf.p[4][4] as f64,
        mount_z_var: eskf.p[17][17] as f64,
        type_counts: eskf.update_diag.type_counts,
        sum_dx_yaw_deg: std::array::from_fn(|i| rad2deg(eskf.update_diag.sum_dx_yaw[i] as f64)),
        sum_abs_dx_yaw_deg: std::array::from_fn(|i| {
            rad2deg(eskf.update_diag.sum_abs_dx_yaw[i] as f64)
        }),
        sum_abs_dx_pitch_deg: std::array::from_fn(|i| {
            rad2deg(eskf.update_diag.sum_abs_dx_pitch[i] as f64)
        }),
        sum_abs_dx_vel_h_mps: std::array::from_fn(|i| eskf.update_diag.sum_abs_dx_vel_h[i] as f64),
        sum_dx_gyro_bias_z_dps: std::array::from_fn(|i| {
            rad2deg(eskf.update_diag.sum_dx_gyro_bias_z[i] as f64)
        }),
        sum_abs_dx_gyro_bias_z_dps: std::array::from_fn(|i| {
            rad2deg(eskf.update_diag.sum_abs_dx_gyro_bias_z[i] as f64)
        }),
        sum_dx_mount_yaw_deg: std::array::from_fn(|i| {
            rad2deg(eskf.update_diag.sum_dx_mount_yaw[i] as f64)
        }),
        sum_abs_innovation: std::array::from_fn(|i| eskf.update_diag.sum_abs_innovation[i] as f64),
        sum_nis: std::array::from_fn(|i| eskf.update_diag.sum_nis[i] as f64),
        sum_abs_dx_mount_yaw_deg: std::array::from_fn(|i| {
            rad2deg(eskf.update_diag.sum_abs_dx_mount_yaw[i] as f64)
        }),
    }
}

fn print_timeline(rows: &[TimelineRow]) {
    if rows.is_empty() {
        println!("timeline: no rows");
        return;
    }
    println!(
        "target_s\tsample_s\tinterval_s\tyaw_nav_deg\tyaw_gnss_course_deg\tnominal_course_gnss_deg\tyaw_nominal_course_deg\tlat_vel_mps\tmount_qerr_deg\tmount_yaw_err_deg\tbgz_dps\ttheta_z_var\tmount_z_var\tgps_vel_n\tgps_vel_innov_abs\tgps_vel_yaw_dx_deg\tgps_vel_abs_yaw_dx_deg\tgps_vel_bgz_dx_dps\tgps_vel_abs_bgz_dx_dps\tgps_vel_velh_abs_mps\tgps_vel_nis_mean\tbody_y_n\tbody_y_innov_abs\tbody_y_yaw_dx_deg\tbody_y_abs_yaw_dx_deg\tbody_y_bgz_dx_dps\tbody_y_mount_yaw_dx_deg\tbody_y_abs_mount_yaw_dx_deg\tbody_y_nis_mean\tbody_z_n\tbody_z_mount_yaw_dx_deg"
    );
    for (idx, row) in rows.iter().enumerate() {
        let prev = idx.checked_sub(1).and_then(|prev_idx| rows.get(prev_idx));
        let snap = row.snap;
        let interval_s = prev.map(|prev| snap.t_s - prev.snap.t_s).unwrap_or(0.0);
        let gps_vel = timeline_delta(prev, snap, 1);
        let body_y = timeline_delta(prev, snap, 4);
        let body_z = timeline_delta(prev, snap, 5);
        println!(
            "{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.4}\t{:.6e}\t{:.6e}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.4}\t{:.4}\t{:.3}\t{:.3}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.4}\t{:.3}\t{:.3}\t{:.3}\t{}\t{:.3}",
            row.target_t_s,
            snap.t_s,
            interval_s,
            snap.yaw_minus_nav_att_deg,
            snap.yaw_minus_course_deg,
            snap.course_minus_course_deg,
            wrap_deg180(snap.yaw_deg - snap.course_deg),
            snap.vel_vehicle_y_mps,
            snap.mount_quat_err_deg,
            snap.mount_yaw_err_deg,
            snap.bgz_dps,
            snap.theta_z_var,
            snap.mount_z_var,
            gps_vel.count,
            gps_vel.innov_abs,
            gps_vel.yaw_dx,
            gps_vel.yaw_dx_abs,
            gps_vel.bgz_dx,
            gps_vel.bgz_dx_abs,
            gps_vel.vel_h_abs,
            gps_vel.nis_mean,
            body_y.count,
            body_y.innov_abs,
            body_y.yaw_dx,
            body_y.yaw_dx_abs,
            body_y.bgz_dx,
            body_y.mount_yaw_dx,
            body_y.mount_yaw_dx_abs,
            body_y.nis_mean,
            body_z.count,
            body_z.mount_yaw_dx,
        );
    }
}

struct TimelineDelta {
    count: u32,
    innov_abs: f64,
    yaw_dx: f64,
    yaw_dx_abs: f64,
    vel_h_abs: f64,
    bgz_dx: f64,
    bgz_dx_abs: f64,
    mount_yaw_dx: f64,
    mount_yaw_dx_abs: f64,
    nis_mean: f64,
}

fn timeline_delta(prev: Option<&TimelineRow>, snap: WindowSnapshot, cue: usize) -> TimelineDelta {
    let prev_snap = prev.map(|row| row.snap);
    let prev_count = prev_snap.map(|s| s.type_counts[cue]).unwrap_or(0);
    let count = snap.type_counts[cue].saturating_sub(prev_count);
    let nis_sum = snap.sum_nis[cue] - prev_snap.map(|s| s.sum_nis[cue]).unwrap_or(0.0);
    TimelineDelta {
        count,
        innov_abs: snap.sum_abs_innovation[cue]
            - prev_snap.map(|s| s.sum_abs_innovation[cue]).unwrap_or(0.0),
        yaw_dx: snap.sum_dx_yaw_deg[cue] - prev_snap.map(|s| s.sum_dx_yaw_deg[cue]).unwrap_or(0.0),
        yaw_dx_abs: snap.sum_abs_dx_yaw_deg[cue]
            - prev_snap.map(|s| s.sum_abs_dx_yaw_deg[cue]).unwrap_or(0.0),
        vel_h_abs: snap.sum_abs_dx_vel_h_mps[cue]
            - prev_snap
                .map(|s| s.sum_abs_dx_vel_h_mps[cue])
                .unwrap_or(0.0),
        bgz_dx: snap.sum_dx_gyro_bias_z_dps[cue]
            - prev_snap
                .map(|s| s.sum_dx_gyro_bias_z_dps[cue])
                .unwrap_or(0.0),
        bgz_dx_abs: snap.sum_abs_dx_gyro_bias_z_dps[cue]
            - prev_snap
                .map(|s| s.sum_abs_dx_gyro_bias_z_dps[cue])
                .unwrap_or(0.0),
        mount_yaw_dx: snap.sum_dx_mount_yaw_deg[cue]
            - prev_snap
                .map(|s| s.sum_dx_mount_yaw_deg[cue])
                .unwrap_or(0.0),
        mount_yaw_dx_abs: snap.sum_abs_dx_mount_yaw_deg[cue]
            - prev_snap
                .map(|s| s.sum_abs_dx_mount_yaw_deg[cue])
                .unwrap_or(0.0),
        nis_mean: if count == 0 {
            0.0
        } else {
            nis_sum / count as f64
        },
    }
}

fn print_snapshot_delta(name: &str, start: f64, end: f64, is_angle: bool) {
    let drift = if is_angle {
        wrap_deg180(end - start)
    } else {
        end - start
    };
    println!(
        "  {}: start={:.6} end={:.6} drift={:.6}",
        name, start, end, drift
    );
}

fn accumulate_gnss_track_pitch_split(
    stats: &mut GnssTrackPitchSplit,
    eskf: &EskfState,
    gnss: &sim::datasets::generic_replay::GenericGnssSample,
    heading_deg: f64,
    anchor_lla: [f64; 3],
) {
    let pos_anchor_ned = lla_to_anchor_ned(anchor_lla, [gnss.lat_deg, gnss.lon_deg, gnss.height_m]);
    let vel_anchor_ned = velocity_local_ned_to_anchor_ned(
        anchor_lla,
        [gnss.lat_deg, gnss.lon_deg],
        gnss.vel_ned_mps,
    );
    let pos_innov_ne = [
        pos_anchor_ned[0] - eskf.nominal.pn as f64,
        pos_anchor_ned[1] - eskf.nominal.pe as f64,
    ];
    let vel_innov_ne = [
        vel_anchor_ned[0] - eskf.nominal.vn as f64,
        vel_anchor_ned[1] - eskf.nominal.ve as f64,
    ];
    let psi = heading_deg.to_radians();
    let along = [psi.cos(), psi.sin()];
    let cross = [-psi.sin(), psi.cos()];

    let r_pos_n = gnss.pos_std_m[0] * gnss.pos_std_m[0];
    let r_pos_e = gnss.pos_std_m[1] * gnss.pos_std_m[1];
    let r_vel_n = gnss.vel_std_mps[0] * gnss.vel_std_mps[0];
    let r_vel_e = gnss.vel_std_mps[1] * gnss.vel_std_mps[1];
    let k_pitch_pos = [
        (eskf.p[1][6] as f64) / ((eskf.p[6][6] as f64) + r_pos_n),
        (eskf.p[1][7] as f64) / ((eskf.p[7][7] as f64) + r_pos_e),
    ];
    let k_pitch_vel = [
        (eskf.p[1][3] as f64) / ((eskf.p[3][3] as f64) + r_vel_n),
        (eskf.p[1][4] as f64) / ((eskf.p[4][4] as f64) + r_vel_e),
    ];
    let k_bax_pos = [
        (eskf.p[12][6] as f64) / ((eskf.p[6][6] as f64) + r_pos_n),
        (eskf.p[12][7] as f64) / ((eskf.p[7][7] as f64) + r_pos_e),
    ];
    let k_bax_vel = [
        (eskf.p[12][3] as f64) / ((eskf.p[3][3] as f64) + r_vel_n),
        (eskf.p[12][4] as f64) / ((eskf.p[4][4] as f64) + r_vel_e),
    ];

    let pos_along = dot2(pos_innov_ne, along);
    let pos_cross = dot2(pos_innov_ne, cross);
    let vel_along = dot2(vel_innov_ne, along);
    let vel_cross = dot2(vel_innov_ne, cross);
    let k_pos_along = dot2(k_pitch_pos, along);
    let k_pos_cross = dot2(k_pitch_pos, cross);
    let k_vel_along = dot2(k_pitch_vel, along);
    let k_vel_cross = dot2(k_pitch_vel, cross);
    let k_bax_pos_along = dot2(k_bax_pos, along);
    let k_bax_pos_cross = dot2(k_bax_pos, cross);
    let k_bax_vel_along = dot2(k_bax_vel, along);
    let k_bax_vel_cross = dot2(k_bax_vel, cross);

    stats.pos_along_innov_m.push(pos_along);
    stats.pos_cross_innov_m.push(pos_cross);
    stats.vel_along_innov_mps.push(vel_along);
    stats.vel_cross_innov_mps.push(vel_cross);
    stats
        .pos_along_pitch_dx_deg
        .push(rad2deg(k_pos_along * pos_along));
    stats
        .pos_cross_pitch_dx_deg
        .push(rad2deg(k_pos_cross * pos_cross));
    stats
        .vel_along_pitch_dx_deg
        .push(rad2deg(k_vel_along * vel_along));
    stats
        .vel_cross_pitch_dx_deg
        .push(rad2deg(k_vel_cross * vel_cross));
    stats
        .pos_along_bax_dx_mps2
        .push(k_bax_pos_along * pos_along);
    stats
        .pos_cross_bax_dx_mps2
        .push(k_bax_pos_cross * pos_cross);
    stats
        .vel_along_bax_dx_mps2
        .push(k_bax_vel_along * vel_along);
    stats
        .vel_cross_bax_dx_mps2
        .push(k_bax_vel_cross * vel_cross);
}

fn collect_nav_att_events(frames: &[UbxFrame], tl: &MasterTimeline) -> Vec<NavAttEvent> {
    let mut out = Vec::new();
    let t0_ms = tl.masters.first().map(|(_, t)| *t).unwrap_or(0.0);
    for frame in frames {
        if let Some((_, _, pitch_deg, heading_deg)) = extract_nav_att(frame)
            && let Some(t_ms) = nearest_master_ms(frame.seq, &tl.masters)
        {
            out.push(NavAttEvent {
                t_s: (t_ms - t0_ms) * 1.0e-3,
                pitch_deg,
                heading_deg: normalize_heading_deg(heading_deg),
            });
        }
    }
    out.sort_by(|a, b| a.t_s.total_cmp(&b.t_s));
    out
}

fn collect_alg_mount_events(frames: &[UbxFrame], tl: &MasterTimeline) -> Vec<AlgMountEvent> {
    let mut out = Vec::new();
    let t0_ms = tl.masters.first().map(|(_, t)| *t).unwrap_or(0.0);
    for frame in frames {
        if let Some((_, roll_deg, pitch_deg, yaw_deg)) = extract_esf_alg(frame)
            && let Some(t_ms) = nearest_master_ms(frame.seq, &tl.masters)
        {
            out.push(AlgMountEvent {
                t_s: (t_ms - t0_ms) * 1.0e-3,
                q_vb: esf_alg_flu_to_frd_mount_quat(roll_deg, pitch_deg, yaw_deg),
                pitch_deg,
                yaw_deg,
            });
        }
    }
    out.sort_by(|a, b| a.t_s.total_cmp(&b.t_s));
    out
}

fn sample_nearest_nav_att(events: &[NavAttEvent], t_s: f64) -> Option<NavAttEvent> {
    sample_nearest(events, t_s, |event| event.t_s)
}

fn sample_nearest_alg_mount(events: &[AlgMountEvent], t_s: f64) -> Option<AlgMountEvent> {
    sample_nearest(events, t_s, |event| event.t_s)
}

fn apply_mount_ablation(
    args: &Args,
    fusion: &mut SensorFusion,
    latched_mount_qcs: &mut Option<[f32; 4]>,
) {
    let sigma_rad = args.analysis_mount_sigma_deg.to_radians() as f32;
    let Some(eskf) = fusion.eskf() else {
        return;
    };

    if args.analysis_freeze_mount {
        let current_qcs = [
            eskf.nominal.qcs0,
            eskf.nominal.qcs1,
            eskf.nominal.qcs2,
            eskf.nominal.qcs3,
        ];
        let qcs = *latched_mount_qcs.get_or_insert(current_qcs);
        fusion.analysis_set_eskf_mount_quat(qcs);
    }

    if args.analysis_zero_mount_cross_cov || args.analysis_freeze_mount {
        fusion.analysis_set_eskf_mount_covariance(sigma_rad, true);
    }
}

fn sample_nearest<T: Copy>(events: &[T], t_s: f64, time_of: impl Fn(&T) -> f64) -> Option<T> {
    if events.is_empty() {
        return None;
    }
    let idx = events.partition_point(|event| time_of(event) < t_s);
    let left = events.get(idx.saturating_sub(1)).copied();
    let right = events.get(idx).copied();
    match (left, right) {
        (Some(left), Some(right)) => {
            if (time_of(&right) - t_s).abs() < (time_of(&left) - t_s).abs() {
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

fn wrap_deg180(mut deg: f64) -> f64 {
    while deg > 180.0 {
        deg -= 360.0;
    }
    while deg <= -180.0 {
        deg += 360.0;
    }
    deg
}

fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn lla_to_anchor_ned(anchor_lla: [f64; 3], sample_lla: [f64; 3]) -> [f64; 3] {
    let anchor_ecef = lla_to_ecef(anchor_lla[0], anchor_lla[1], anchor_lla[2]);
    let sample_ecef = lla_to_ecef(sample_lla[0], sample_lla[1], sample_lla[2]);
    let diff = [
        sample_ecef[0] - anchor_ecef[0],
        sample_ecef[1] - anchor_ecef[1],
        sample_ecef[2] - anchor_ecef[2],
    ];
    mat_vec(ecef_to_ned_matrix(anchor_lla[0], anchor_lla[1]), diff)
}

fn velocity_local_ned_to_anchor_ned(
    anchor_lla: [f64; 3],
    sample_lat_lon: [f64; 2],
    vel_local_ned_mps: [f64; 3],
) -> [f64; 3] {
    let c_ne_local = ecef_to_ned_matrix(sample_lat_lon[0], sample_lat_lon[1]);
    let vel_ecef = mat_vec(transpose3(c_ne_local), vel_local_ned_mps);
    mat_vec(ecef_to_ned_matrix(anchor_lla[0], anchor_lla[1]), vel_ecef)
}

fn lla_to_ecef(lat_deg: f64, lon_deg: f64, height_m: f64) -> [f64; 3] {
    const A_M: f64 = 6_378_137.0;
    const E2: f64 = 6.69437999014e-3;
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let slat = lat.sin();
    let clat = lat.cos();
    let slon = lon.sin();
    let clon = lon.cos();
    let n = A_M / (1.0 - E2 * slat * slat).sqrt();
    [
        (n + height_m) * clat * clon,
        (n + height_m) * clat * slon,
        (n * (1.0 - E2) + height_m) * slat,
    ]
}

fn ecef_to_ned_matrix(lat_deg: f64, lon_deg: f64) -> [[f64; 3]; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let slat = lat.sin();
    let clat = lat.cos();
    let slon = lon.sin();
    let clon = lon.cos();
    [
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [-clat * clon, -clat * slon, -slat],
    ]
}

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let q = [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ];
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n <= 1.0e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn quat_to_rotmat_f64(q: [f64; 4]) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
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

fn mat_vec(r: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}
