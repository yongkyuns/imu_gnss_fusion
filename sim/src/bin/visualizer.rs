#![recursion_limit = "1024"]

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::Result;
#[cfg(not(target_arch = "wasm32"))]
use clap::{Parser, ValueEnum};
#[cfg(not(target_arch = "wasm32"))]
use sim::datasets::generic_replay::{load_gnss_samples, load_imu_samples};
#[cfg(not(target_arch = "wasm32"))]
use sim::visualizer::model::EkfImuSource;
#[cfg(not(target_arch = "wasm32"))]
use sim::visualizer::pipeline::generic::{GenericReplayInput, build_generic_replay_plot_data};
#[cfg(not(target_arch = "wasm32"))]
use sim::visualizer::pipeline::synthetic::{
    SyntheticNoiseMode, SyntheticVisualizerConfig, build_synthetic_plot_data,
};
#[cfg(not(target_arch = "wasm32"))]
use sim::visualizer::pipeline::{EkfCompareConfig, GnssOutageConfig};
#[cfg(not(target_arch = "wasm32"))]
use sim::visualizer::stats::{
    group_stats, max_gap_sec, max_gap_trace, max_step_abs, trace_stats, trace_time_bounds,
    trace_value_bounds,
};
#[cfg(not(target_arch = "wasm32"))]
use sim::visualizer::ui::{ReplayState, run_visualizer};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(name = "visualizer")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: Option<PathBuf>,
    #[arg(long, value_name = "DIR")]
    generic_replay_dir: Option<PathBuf>,
    #[arg(long, alias = "synthetic-scenario")]
    synthetic_motion_def: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = SyntheticNoiseArg::Truth)]
    synthetic_noise: SyntheticNoiseArg,
    #[arg(long, default_value_t = 1)]
    synthetic_seed: u64,
    #[arg(long, default_value_t = 5.0)]
    synthetic_mount_roll_deg: f64,
    #[arg(long, default_value_t = -5.0)]
    synthetic_mount_pitch_deg: f64,
    #[arg(long, default_value_t = 5.0)]
    synthetic_mount_yaw_deg: f64,
    #[arg(long, default_value_t = 100.0)]
    synthetic_imu_hz: f64,
    #[arg(long, default_value_t = 2.0)]
    synthetic_gnss_hz: f64,
    #[arg(long, default_value_t = 0.0)]
    synthetic_gnss_time_shift_ms: f64,
    #[arg(long, default_value_t = 0.0)]
    synthetic_early_vel_bias_n_mps: f64,
    #[arg(long, default_value_t = 0.0)]
    synthetic_early_vel_bias_e_mps: f64,
    #[arg(long, default_value_t = 0.0)]
    synthetic_early_vel_bias_d_mps: f64,
    #[arg(long)]
    synthetic_early_fault_start_s: Option<f64>,
    #[arg(long)]
    synthetic_early_fault_end_s: Option<f64>,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long)]
    profile_only: bool,
    #[arg(
        long = "misalignment",
        alias = "ekf-imu-source",
        default_value = "internal",
        value_parser = parse_misalignment
    )]
    misalignment: EkfImuSource,
    #[arg(long)]
    dump_align_axis_time_s: Option<f64>,
    #[arg(long)]
    dump_loose_time_s: Option<f64>,
    #[arg(long, default_value_t = 3.0)]
    dump_window_s: f64,
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
    #[arg(long)]
    gnss_pos_r_scale: Option<f64>,
    #[arg(long)]
    gnss_vel_r_scale: Option<f64>,
    #[arg(long)]
    r_body_vel: Option<f32>,
    #[arg(long)]
    gnss_pos_mount_scale: Option<f32>,
    #[arg(long)]
    gnss_vel_mount_scale: Option<f32>,
    #[arg(long)]
    yaw_init_sigma_deg: Option<f32>,
    #[arg(long)]
    gyro_bias_init_sigma_dps: Option<f32>,
    #[arg(long)]
    accel_bias_init_sigma_mps2: Option<f32>,
    #[arg(long)]
    mount_init_sigma_deg: Option<f32>,
    #[arg(long)]
    r_vehicle_speed: Option<f32>,
    #[arg(long)]
    r_zero_vel: Option<f32>,
    #[arg(long)]
    r_stationary_accel: Option<f32>,
    #[arg(long)]
    mount_align_rw_var: Option<f32>,
    #[arg(long)]
    mount_update_min_scale: Option<f32>,
    #[arg(long)]
    mount_update_ramp_time_s: Option<f32>,
    #[arg(long)]
    mount_update_innovation_gate_mps: Option<f32>,
    #[arg(long)]
    align_handoff_delay_s: Option<f32>,
    #[arg(long)]
    freeze_misalignment_states: bool,
    #[arg(long)]
    mount_settle_time_s: Option<f32>,
    #[arg(long)]
    mount_settle_release_sigma_deg: Option<f32>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    mount_settle_zero_cross_covariance: bool,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, ValueEnum)]
enum SyntheticNoiseArg {
    Truth,
    Low,
    Mid,
    High,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<SyntheticNoiseArg> for SyntheticNoiseMode {
    fn from(value: SyntheticNoiseArg) -> Self {
        match value {
            SyntheticNoiseArg::Truth => Self::Truth,
            SyntheticNoiseArg::Low => Self::Low,
            SyntheticNoiseArg::Mid => Self::Mid,
            SyntheticNoiseArg::High => Self::High,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    let ekf_cfg = EkfCompareConfig {
        r_body_vel: args
            .r_body_vel
            .unwrap_or(EkfCompareConfig::default().r_body_vel),
        gnss_pos_mount_scale: args
            .gnss_pos_mount_scale
            .unwrap_or(EkfCompareConfig::default().gnss_pos_mount_scale),
        gnss_vel_mount_scale: args
            .gnss_vel_mount_scale
            .unwrap_or(EkfCompareConfig::default().gnss_vel_mount_scale),
        yaw_init_sigma_deg: args
            .yaw_init_sigma_deg
            .unwrap_or(EkfCompareConfig::default().yaw_init_sigma_deg),
        gyro_bias_init_sigma_dps: args
            .gyro_bias_init_sigma_dps
            .unwrap_or(EkfCompareConfig::default().gyro_bias_init_sigma_dps),
        accel_bias_init_sigma_mps2: args
            .accel_bias_init_sigma_mps2
            .unwrap_or(EkfCompareConfig::default().accel_bias_init_sigma_mps2),
        mount_init_sigma_deg: args
            .mount_init_sigma_deg
            .unwrap_or(EkfCompareConfig::default().mount_init_sigma_deg),
        r_vehicle_speed: args
            .r_vehicle_speed
            .unwrap_or(EkfCompareConfig::default().r_vehicle_speed),
        r_zero_vel: args
            .r_zero_vel
            .unwrap_or(EkfCompareConfig::default().r_zero_vel),
        r_stationary_accel: args
            .r_stationary_accel
            .unwrap_or(EkfCompareConfig::default().r_stationary_accel),
        mount_align_rw_var: args
            .mount_align_rw_var
            .unwrap_or(EkfCompareConfig::default().mount_align_rw_var),
        mount_update_min_scale: args
            .mount_update_min_scale
            .unwrap_or(EkfCompareConfig::default().mount_update_min_scale),
        mount_update_ramp_time_s: args
            .mount_update_ramp_time_s
            .unwrap_or(EkfCompareConfig::default().mount_update_ramp_time_s),
        mount_update_innovation_gate_mps: args
            .mount_update_innovation_gate_mps
            .unwrap_or(EkfCompareConfig::default().mount_update_innovation_gate_mps),
        align_handoff_delay_s: args
            .align_handoff_delay_s
            .unwrap_or(EkfCompareConfig::default().align_handoff_delay_s),
        freeze_misalignment_states: args.freeze_misalignment_states,
        mount_settle_time_s: args
            .mount_settle_time_s
            .unwrap_or(EkfCompareConfig::default().mount_settle_time_s),
        mount_settle_release_sigma_deg: args
            .mount_settle_release_sigma_deg
            .unwrap_or(EkfCompareConfig::default().mount_settle_release_sigma_deg),
        mount_settle_zero_cross_covariance: args.mount_settle_zero_cross_covariance,
        gnss_pos_r_scale: args
            .gnss_pos_r_scale
            .unwrap_or(EkfCompareConfig::default().gnss_pos_r_scale),
        predict_imu_decimation: args.ekf_predict_imu_decimation.max(1),
        predict_imu_lpf_cutoff_hz: args.ekf_predict_imu_lpf_cutoff_hz,
        gnss_vel_r_scale: args
            .gnss_vel_r_scale
            .unwrap_or(EkfCompareConfig::default().gnss_vel_r_scale),
        ..EkfCompareConfig::default()
    };

    let gnss_outages = GnssOutageConfig {
        count: args.gnss_outage_count,
        duration_s: args.gnss_outage_duration_s,
        seed: args.gnss_outage_seed,
    };
    let (data, has_itow, input_label, input_records, t_read, replay_state) = if let Some(
        replay_dir,
    ) =
        args.generic_replay_dir.as_ref()
    {
        let imu = load_imu_samples(replay_dir)?;
        let gnss = load_gnss_samples(replay_dir)?;
        let input_records = imu.len() + gnss.len();
        let t_read = Instant::now();
        let replay = GenericReplayInput { imu, gnss };
        let data =
            build_generic_replay_plot_data(&replay, args.misalignment, ekf_cfg, gnss_outages);
        (
            data,
            false,
            format!("generic-csv:{}", replay_dir.display()),
            input_records,
            t_read,
            None,
        )
    } else if let Some(motion_def) = args.synthetic_motion_def.clone() {
        let synth_cfg = SyntheticVisualizerConfig {
            motion_def: Some(motion_def.clone()),
            motion_label: motion_def.display().to_string(),
            motion_text: None,
            noise_mode: args.synthetic_noise.into(),
            seed: args.synthetic_seed,
            mount_rpy_deg: [
                args.synthetic_mount_roll_deg,
                args.synthetic_mount_pitch_deg,
                args.synthetic_mount_yaw_deg,
            ],
            imu_hz: args.synthetic_imu_hz,
            gnss_hz: args.synthetic_gnss_hz,
            gnss_time_shift_ms: args.synthetic_gnss_time_shift_ms,
            early_vel_bias_ned_mps: [
                args.synthetic_early_vel_bias_n_mps,
                args.synthetic_early_vel_bias_e_mps,
                args.synthetic_early_vel_bias_d_mps,
            ],
            early_fault_window_s: args
                .synthetic_early_fault_start_s
                .zip(args.synthetic_early_fault_end_s),
        };
        let data = build_synthetic_plot_data(&synth_cfg, args.misalignment, ekf_cfg, gnss_outages)?;
        let replay = ReplayState {
            bytes: Vec::new(),
            synthetic: Some(synth_cfg),
            max_records: args.max_records,
            misalignment: args.misalignment,
            ekf_cfg,
            gnss_outages,
        };
        (
            data,
            false,
            format!("synthetic:{}", motion_def.display()),
            0usize,
            Instant::now(),
            Some(replay),
        )
    } else {
        let logfile = args
            .logfile
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<none>".to_string());
        anyhow::bail!(
            "binary replay is unavailable in this checkout (input: {logfile}); use --generic-replay-dir or --synthetic-motion-def"
        );
    };
    let t_build = Instant::now();
    let (n_traces, n_points) = trace_stats(&data);
    let (tmin, tmax) = trace_time_bounds(&data).unwrap_or((f64::NAN, f64::NAN));
    eprintln!(
        "[profile] input={} records={} read={:.3}s build={:.3}s total_pre_ui={:.3}s traces={} points={} t_range=[{:.3}, {:.3}]s",
        input_label,
        input_records,
        (t_read - t0).as_secs_f64(),
        (t_build - t_read).as_secs_f64(),
        (t_build - t0).as_secs_f64(),
        n_traces,
        n_points,
        tmin,
        tmax
    );
    eprintln!(
        "[profile] ekf-only misalignment={:?} predict_imu_decimation={} ekf-only predict_imu_lpf_cutoff_hz={} gnss_pos_r_scale={:.3} gnss_vel_r_scale={:.3} r_body_vel={:.3} gnss_pos_mount_scale={:.3} gnss_vel_mount_scale={:.3} yaw_init_sigma_deg={:.3} gyro_bias_init_sigma_dps={:.3} r_vehicle_speed={:.3} r_zero_vel={:.3} r_stationary_accel={:.3} mount_align_rw_var={:.6e} mount_update_min_scale={:.3} mount_update_ramp_time_s={:.3} mount_update_innovation_gate_mps={:.3} align_handoff_delay_s={:.3} freeze_misalignment_states={} mount_settle_time_s={:.3} mount_settle_release_sigma_deg={:.3} mount_settle_zero_cross_covariance={}",
        args.misalignment,
        ekf_cfg.predict_imu_decimation,
        ekf_cfg
            .predict_imu_lpf_cutoff_hz
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "off".to_string()),
        ekf_cfg.gnss_pos_r_scale,
        ekf_cfg.gnss_vel_r_scale,
        ekf_cfg.r_body_vel,
        ekf_cfg.gnss_pos_mount_scale,
        ekf_cfg.gnss_vel_mount_scale,
        ekf_cfg.yaw_init_sigma_deg,
        ekf_cfg.gyro_bias_init_sigma_dps,
        ekf_cfg.r_vehicle_speed,
        ekf_cfg.r_zero_vel,
        ekf_cfg.r_stationary_accel,
        ekf_cfg.mount_align_rw_var,
        ekf_cfg.mount_update_min_scale,
        ekf_cfg.mount_update_ramp_time_s,
        ekf_cfg.mount_update_innovation_gate_mps,
        ekf_cfg.align_handoff_delay_s,
        ekf_cfg.freeze_misalignment_states,
        ekf_cfg.mount_settle_time_s,
        ekf_cfg.mount_settle_release_sigma_deg,
        ekf_cfg.mount_settle_zero_cross_covariance,
    );
    for (name, nt, np) in [
        group_stats("speed", &data.speed),
        group_stats("sat_cn0", &data.sat_cn0),
        group_stats("imu_raw_gyro", &data.imu_raw_gyro),
        group_stats("imu_raw_accel", &data.imu_raw_accel),
        group_stats("imu_cal_gyro", &data.imu_cal_gyro),
        group_stats("imu_cal_accel", &data.imu_cal_accel),
        group_stats("orientation", &data.orientation),
        group_stats("other", &data.other),
        group_stats("eskf_cmp_pos", &data.eskf_cmp_pos),
        group_stats("eskf_cmp_vel", &data.eskf_cmp_vel),
        group_stats("eskf_cmp_att", &data.eskf_cmp_att),
        group_stats("eskf_meas_gyro", &data.eskf_meas_gyro),
        group_stats("eskf_meas_accel", &data.eskf_meas_accel),
        group_stats("eskf_bias_gyro", &data.eskf_bias_gyro),
        group_stats("eskf_bias_accel", &data.eskf_bias_accel),
        group_stats("eskf_cov_bias", &data.eskf_cov_bias),
        group_stats("eskf_cov_nonbias", &data.eskf_cov_nonbias),
        group_stats("eskf_misalignment", &data.eskf_misalignment),
        group_stats("eskf_stationary_diag", &data.eskf_stationary_diag),
        group_stats("eskf_bump_pitch_speed", &data.eskf_bump_pitch_speed),
        group_stats("eskf_bump_diag", &data.eskf_bump_diag),
        group_stats("eskf_map", &data.eskf_map),
        group_stats("loose_cmp_pos", &data.loose_cmp_pos),
        group_stats("loose_cmp_vel", &data.loose_cmp_vel),
        group_stats("loose_cmp_att", &data.loose_cmp_att),
        group_stats("loose_meas_gyro", &data.loose_meas_gyro),
        group_stats("loose_meas_accel", &data.loose_meas_accel),
        group_stats("loose_bias_gyro", &data.loose_bias_gyro),
        group_stats("loose_bias_accel", &data.loose_bias_accel),
        group_stats("loose_scale_gyro", &data.loose_scale_gyro),
        group_stats("loose_scale_accel", &data.loose_scale_accel),
        group_stats("loose_cov_bias", &data.loose_cov_bias),
        group_stats("loose_cov_nonbias", &data.loose_cov_nonbias),
        group_stats("loose_map", &data.loose_map),
        group_stats("align_cmp_att", &data.align_cmp_att),
        group_stats("align_res_vel", &data.align_res_vel),
        group_stats("align_axis_err", &data.align_axis_err),
        group_stats("align_motion", &data.align_motion),
        group_stats("align_roll_contrib", &data.align_roll_contrib),
        group_stats("align_pitch_contrib", &data.align_pitch_contrib),
        group_stats("align_yaw_contrib", &data.align_yaw_contrib),
        group_stats("align_cov", &data.align_cov),
    ] {
        eprintln!("[profile] group={} traces={} points={}", name, nt, np);
    }
    eprintln!(
        "[profile] max_gap_s raw_gyro={:.3} raw_accel={:.3} cal_gyro={:.3} cal_accel={:.3}",
        max_gap_sec(&data.imu_raw_gyro),
        max_gap_sec(&data.imu_raw_accel),
        max_gap_sec(&data.imu_cal_gyro),
        max_gap_sec(&data.imu_cal_accel),
    );
    for (group, traces) in [
        ("imu_raw_gyro", &data.imu_raw_gyro),
        ("imu_raw_accel", &data.imu_raw_accel),
        ("imu_cal_gyro", &data.imu_cal_gyro),
        ("imu_cal_accel", &data.imu_cal_accel),
        ("align_cmp_att", &data.align_cmp_att),
        ("align_res_vel", &data.align_res_vel),
        ("align_axis_err", &data.align_axis_err),
        ("align_motion", &data.align_motion),
        ("eskf_meas_gyro", &data.eskf_meas_gyro),
        ("eskf_meas_accel", &data.eskf_meas_accel),
        ("eskf_misalignment", &data.eskf_misalignment),
        ("eskf_bump_pitch_speed", &data.eskf_bump_pitch_speed),
        ("eskf_bump_diag", &data.eskf_bump_diag),
        ("loose_meas_gyro", &data.loose_meas_gyro),
        ("loose_meas_accel", &data.loose_meas_accel),
        ("loose_scale_gyro", &data.loose_scale_gyro),
        ("loose_scale_accel", &data.loose_scale_accel),
        ("align_roll_contrib", &data.align_roll_contrib),
        ("align_pitch_contrib", &data.align_pitch_contrib),
        ("align_yaw_contrib", &data.align_yaw_contrib),
        ("align_cov", &data.align_cov),
    ] {
        if let Some((name, gap)) = max_gap_trace(traces) {
            eprintln!(
                "[profile] max_gap_trace group={} signal={} gap_s={:.3}",
                group, name, gap
            );
        }
        if let Some((vmin, vmax)) = trace_value_bounds(traces) {
            eprintln!(
                "[profile] value_range group={} min={:.6} max={:.6}",
                group, vmin, vmax
            );
        }
        if let Some(step) = max_step_abs(traces) {
            eprintln!("[profile] max_step_abs group={} value={:.6}", group, step);
        }
    }
    if let Some(t_s) = args.dump_align_axis_time_s {
        dump_traces_near_time(
            "align_cmp_att",
            &data.align_cmp_att,
            t_s,
            args.dump_window_s,
        );
        dump_traces_near_time(
            "align_axis_err",
            &data.align_axis_err,
            t_s,
            args.dump_window_s,
        );
    }
    if let Some(t_s) = args.dump_loose_time_s {
        dump_traces_near_time(
            "loose_cmp_vel",
            &data.loose_cmp_vel,
            t_s,
            args.dump_window_s,
        );
        dump_traces_near_time(
            "loose_cmp_att",
            &data.loose_cmp_att,
            t_s,
            args.dump_window_s,
        );
        dump_traces_near_time(
            "loose_misalignment",
            &data.loose_misalignment,
            t_s,
            args.dump_window_s,
        );
        dump_traces_near_time(
            "eskf_misalignment",
            &data.eskf_misalignment,
            t_s,
            args.dump_window_s,
        );
        dump_traces_near_time(
            "eskf_stationary_diag",
            &data.eskf_stationary_diag,
            t_s,
            args.dump_window_s,
        );
        dump_traces_near_time(
            "loose_bias_accel",
            &data.loose_bias_accel,
            t_s,
            args.dump_window_s,
        );
    }
    if args.profile_only {
        return Ok(());
    }

    run_visualizer(data, has_itow, replay_state)
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn start_visualizer(canvas_id: &str) -> std::result::Result<(), JsValue> {
    let window = eframe::web_sys::window().ok_or_else(|| JsValue::from_str("missing window"))?;
    let document = window
        .document()
        .ok_or_else(|| JsValue::from_str("missing document"))?;
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| JsValue::from_str("missing visualizer canvas"))?
        .dyn_into::<eframe::web_sys::HtmlCanvasElement>()?;
    let runner = Box::leak(Box::new(eframe::WebRunner::new()));
    sim::visualizer::ui::run_visualizer_web(
        runner,
        canvas,
        sim::visualizer::model::PlotData::default(),
        false,
    )
    .await
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_misalignment(s: &str) -> Result<EkfImuSource, String> {
    EkfImuSource::from_cli_value(s)
}

#[cfg(not(target_arch = "wasm32"))]
fn dump_traces_near_time(
    group: &str,
    traces: &[sim::visualizer::model::Trace],
    t_s: f64,
    window_s: f64,
) {
    let half = 0.5 * window_s.abs();
    eprintln!(
        "[dump] group={} center_t_s={:.3} window_s={:.3}",
        group, t_s, window_s
    );
    for trace in traces {
        let mut any = false;
        for p in &trace.points {
            if (p[0] - t_s).abs() <= half {
                if !any {
                    eprintln!("[dump] trace={}", trace.name);
                    any = true;
                }
                eprintln!("[dump]   t_s={:.3} value={:.6}", p[0], p[1]);
            }
        }
        if !any {
            eprintln!("[dump] trace={} no points in window", trace.name);
        }
    }
}
