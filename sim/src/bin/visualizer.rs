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
use sim::datasets::generic_replay::{
    load_gnss_samples, load_imu_samples, load_reference_attitude_samples,
    load_reference_mount_samples,
};
#[cfg(not(target_arch = "wasm32"))]
use sim::eval::state_summary::{SummaryMode, summarize_trace_pair};
#[cfg(not(target_arch = "wasm32"))]
use sim::visualizer::model::{EkfImuSource, PlotData, Trace};
#[cfg(not(target_arch = "wasm32"))]
use sim::visualizer::pipeline::generic::GenericReplayInput;
#[cfg(not(target_arch = "wasm32"))]
use sim::visualizer::pipeline::synthetic::{
    SyntheticNoiseMode, SyntheticVisualizerConfig, build_synthetic_plot_data,
};
#[cfg(not(target_arch = "wasm32"))]
use sim::visualizer::pipeline::{EkfCompareConfig, GnssOutageConfig};
#[cfg(not(target_arch = "wasm32"))]
use sim::visualizer::replay_job::{GenericReplayJobConfig, run_generic_replay_job};
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
    #[arg(long)]
    synthetic_disable_imu_noise: bool,
    #[arg(long)]
    synthetic_disable_gnss_noise: bool,
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
    r_body_vel: Option<f32>,
    #[arg(long)]
    r_body_vel_z: Option<f32>,
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
    mount_roll_pitch_init_sigma_deg: Option<f32>,
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
        r_body_vel_z: args
            .r_body_vel_z
            .unwrap_or(EkfCompareConfig::default().r_body_vel_z),
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
        mount_roll_pitch_init_sigma_deg: args
            .mount_roll_pitch_init_sigma_deg
            .unwrap_or(EkfCompareConfig::default().mount_roll_pitch_init_sigma_deg),
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
        predict_imu_decimation: args.ekf_predict_imu_decimation.max(1),
        predict_imu_lpf_cutoff_hz: args.ekf_predict_imu_lpf_cutoff_hz,
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
        let reference_attitude = load_reference_attitude_samples(replay_dir)?;
        let reference_mount = load_reference_mount_samples(replay_dir)?;
        let input_records =
            imu.len() + gnss.len() + reference_attitude.len() + reference_mount.len();
        let t_read = Instant::now();
        let replay = GenericReplayInput {
            imu,
            gnss,
            reference_attitude,
            reference_mount,
        };
        let data = run_generic_replay_job(
            &replay,
            GenericReplayJobConfig::full(args.misalignment, ekf_cfg, gnss_outages),
        );
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
            disable_imu_noise: args.synthetic_disable_imu_noise,
            disable_gnss_noise: args.synthetic_disable_gnss_noise,
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
        "[profile] ekf-only misalignment={:?} predict_imu_decimation={} ekf-only predict_imu_lpf_cutoff_hz={} r_body_vel_y={:.3} r_body_vel_z={:.3} gnss_pos_mount_scale={:.3} gnss_vel_mount_scale={:.3} yaw_init_sigma_deg={:.3} gyro_bias_init_sigma_dps={:.3} mount_roll_pitch_init_sigma_deg={:.3} mount_yaw_init_sigma_deg={:.3} r_vehicle_speed={:.3} r_zero_vel={:.3} r_stationary_accel={:.3} mount_align_rw_var={:.6e} mount_update_min_scale={:.3} mount_update_ramp_time_s={:.3} mount_update_innovation_gate_mps={:.3} align_handoff_delay_s={:.3} freeze_misalignment_states={} mount_settle_time_s={:.3} mount_settle_release_sigma_deg={:.3} mount_settle_zero_cross_covariance={}",
        args.misalignment,
        ekf_cfg.predict_imu_decimation,
        ekf_cfg
            .predict_imu_lpf_cutoff_hz
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "off".to_string()),
        ekf_cfg.r_body_vel,
        ekf_cfg.r_body_vel_z,
        ekf_cfg.gnss_pos_mount_scale,
        ekf_cfg.gnss_vel_mount_scale,
        ekf_cfg.yaw_init_sigma_deg,
        ekf_cfg.gyro_bias_init_sigma_dps,
        ekf_cfg.mount_roll_pitch_init_sigma_deg,
        ekf_cfg.mount_init_sigma_deg,
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
        group_stats("eskf_mount_sigma", &data.eskf_mount_sigma),
        group_stats("eskf_mount_dx", &data.eskf_mount_dx),
        group_stats("eskf_nhc_mount_dx", &data.eskf_nhc_mount_dx),
        group_stats("eskf_nhc_innovation", &data.eskf_nhc_innovation),
        group_stats("eskf_nhc_nis", &data.eskf_nhc_nis),
        group_stats("eskf_nhc_h_mount_norm", &data.eskf_nhc_h_mount_norm),
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
        group_stats("loose_mount_sigma", &data.loose_mount_sigma),
        group_stats("loose_mount_dx", &data.loose_mount_dx),
        group_stats("loose_nhc_innovation", &data.loose_nhc_innovation),
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
    print_map_bounds("eskf_map", &data.eskf_map);
    print_map_bounds("loose_map", &data.loose_map);
    print_eskf_nhc_window_summaries(&data, tmin);
    print_loose_nhc_window_summaries(&data, tmin);
    print_loose_mount_window_summaries(&data, tmin);
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
    print_reference_error_summaries(&data);
    if args.profile_only {
        return Ok(());
    }

    run_visualizer(data, has_itow, replay_state)
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
use js_sys::{Function, Object, Reflect};
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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn build_generic_replay_job_json(request_json: &str) -> std::result::Result<String, JsValue> {
    let request: WebReplayJobRequest = serde_json::from_str(request_json)
        .map_err(|err| JsValue::from_str(&format!("invalid replay job request: {err}")))?;
    let response = build_web_replay_job_response(request, None);
    serde_json::to_string(&response)
        .map_err(|err| JsValue::from_str(&format!("failed to serialize replay job: {err}")))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn build_replay_plot_data_json(request_json: &str) -> std::result::Result<String, JsValue> {
    let request: WebReplayJobRequest = serde_json::from_str(request_json)
        .map_err(|err| JsValue::from_str(&format!("invalid replay job request: {err}")))?;
    let data = build_web_replay_plot_data(request, None)
        .map_err(|err| JsValue::from_str(&format!("replay failed: {err}")))?;
    serde_json::to_string(&data)
        .map_err(|err| JsValue::from_str(&format!("failed to serialize replay data: {err}")))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn build_replay_job_json_with_progress(
    request_json: &str,
    progress_callback: &Function,
) -> std::result::Result<String, JsValue> {
    let request: WebReplayJobRequest = serde_json::from_str(request_json)
        .map_err(|err| JsValue::from_str(&format!("invalid replay job request: {err}")))?;
    let job_id = request.job_id;

    let mut progress = |progress: sim::visualizer::pipeline::generic::GenericReplayProgress| {
        let message = Object::new();
        let _ = Reflect::set(
            &message,
            &JsValue::from_str("type"),
            &JsValue::from_str("progress"),
        );
        let _ = Reflect::set(
            &message,
            &JsValue::from_str("jobId"),
            &JsValue::from_f64(job_id as f64),
        );
        let _ = Reflect::set(
            &message,
            &JsValue::from_str("progress"),
            &JsValue::from_f64(progress.fraction),
        );
        let _ = Reflect::set(
            &message,
            &JsValue::from_str("currentTimeS"),
            &JsValue::from_f64(progress.current_t_s),
        );
        let _ = Reflect::set(
            &message,
            &JsValue::from_str("finalTimeS"),
            &JsValue::from_f64(progress.final_t_s),
        );
        let _ = progress_callback.call1(&JsValue::NULL, &message);
    };

    let response = build_web_replay_job_response(request, Some(&mut progress));
    serde_json::to_string(&response)
        .map_err(|err| JsValue::from_str(&format!("failed to serialize replay job: {err}")))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn build_generic_replay_job_json_with_progress(
    request_json: &str,
    progress_callback: &Function,
) -> std::result::Result<String, JsValue> {
    build_replay_job_json_with_progress(request_json, progress_callback)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn build_replay_plot_data_json_with_progress(
    request_json: &str,
    progress_callback: &Function,
) -> std::result::Result<String, JsValue> {
    let request: WebReplayJobRequest = serde_json::from_str(request_json)
        .map_err(|err| JsValue::from_str(&format!("invalid replay job request: {err}")))?;
    let job_id = request.job_id;

    let mut progress = |progress: sim::visualizer::pipeline::generic::GenericReplayProgress| {
        let message = Object::new();
        let _ = Reflect::set(
            &message,
            &JsValue::from_str("type"),
            &JsValue::from_str("progress"),
        );
        let _ = Reflect::set(
            &message,
            &JsValue::from_str("jobId"),
            &JsValue::from_f64(job_id as f64),
        );
        let _ = Reflect::set(
            &message,
            &JsValue::from_str("progress"),
            &JsValue::from_f64(progress.fraction),
        );
        let _ = Reflect::set(
            &message,
            &JsValue::from_str("currentTimeS"),
            &JsValue::from_f64(progress.current_t_s),
        );
        let _ = Reflect::set(
            &message,
            &JsValue::from_str("finalTimeS"),
            &JsValue::from_f64(progress.final_t_s),
        );
        let _ = progress_callback.call1(&JsValue::NULL, &message);
    };

    let data = build_web_replay_plot_data(request, Some(&mut progress))
        .map_err(|err| JsValue::from_str(&format!("replay failed: {err}")))?;
    serde_json::to_string(&data)
        .map_err(|err| JsValue::from_str(&format!("failed to serialize replay data: {err}")))
}

#[cfg(target_arch = "wasm32")]
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct WebReplayJobRequest {
    #[serde(default)]
    job_id: u64,
    #[serde(default)]
    misalignment: Option<String>,
    #[serde(default)]
    ekf_cfg: Option<sim::visualizer::pipeline::EkfCompareConfig>,
    #[serde(default)]
    gnss_outages: Option<sim::visualizer::pipeline::GnssOutageConfig>,
    label: Option<String>,
    imu_name: Option<String>,
    gnss_name: Option<String>,
    imu_csv: Option<String>,
    gnss_csv: Option<String>,
    reference_attitude_csv: Option<String>,
    reference_mount_csv: Option<String>,
    source: Option<WebReplayJobSource>,
}

#[cfg(target_arch = "wasm32")]
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct WebReplayJobSource {
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    imu_name: Option<String>,
    #[serde(default)]
    gnss_name: Option<String>,
    #[serde(default)]
    imu_csv: Option<String>,
    #[serde(default)]
    gnss_csv: Option<String>,
    #[serde(default)]
    reference_attitude_csv: Option<String>,
    #[serde(default)]
    reference_mount_csv: Option<String>,
    #[serde(default)]
    motion_label: Option<String>,
    #[serde(default)]
    motion_text: Option<String>,
    #[serde(default)]
    noise_mode: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    mount_rpy_deg: Option<[f64; 3]>,
    #[serde(default)]
    imu_hz: Option<f64>,
    #[serde(default)]
    gnss_hz: Option<f64>,
    #[serde(default)]
    gnss_time_shift_ms: Option<f64>,
    #[serde(default)]
    early_vel_bias_ned_mps: Option<[f64; 3]>,
    #[serde(default)]
    early_fault_window_s: Option<[f64; 2]>,
}

#[cfg(target_arch = "wasm32")]
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct WebReplayJobResponse {
    ok: bool,
    job_id: u64,
    label: String,
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    imu_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gnss_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[cfg(target_arch = "wasm32")]
struct WebReplayInput {
    source: String,
    label: String,
    imu_name: String,
    gnss_name: String,
    replay: sim::visualizer::pipeline::generic::GenericReplayInput,
}

#[cfg(target_arch = "wasm32")]
fn web_request_misalignment(request: &WebReplayJobRequest) -> sim::visualizer::model::EkfImuSource {
    request
        .misalignment
        .as_deref()
        .and_then(|value| sim::visualizer::model::EkfImuSource::from_cli_value(value).ok())
        .unwrap_or(sim::visualizer::model::EkfImuSource::Internal)
}

#[cfg(target_arch = "wasm32")]
fn web_request_gnss_outages(
    request: &WebReplayJobRequest,
) -> sim::visualizer::pipeline::GnssOutageConfig {
    request.gnss_outages.unwrap_or_default()
}

#[cfg(target_arch = "wasm32")]
fn web_request_ekf_cfg(
    request: &WebReplayJobRequest,
) -> sim::visualizer::pipeline::EkfCompareConfig {
    request.ekf_cfg.unwrap_or_default()
}

#[cfg(target_arch = "wasm32")]
fn build_web_replay_job_response(
    request: WebReplayJobRequest,
    progress: Option<&mut dyn FnMut(sim::visualizer::pipeline::generic::GenericReplayProgress)>,
) -> WebReplayJobResponse {
    let job_id = request.job_id;
    let fallback_label = request
        .label
        .clone()
        .unwrap_or_else(|| "CSV replay".to_string());
    let source_kind = request
        .source
        .as_ref()
        .and_then(|source| source.kind.as_deref())
        .unwrap_or("csv")
        .to_ascii_lowercase();
    if source_kind == "synthetic" {
        return build_web_synthetic_replay_job_response(request, progress);
    }
    let misalignment = web_request_misalignment(&request);
    let ekf_cfg = web_request_ekf_cfg(&request);
    let gnss_outages = web_request_gnss_outages(&request);
    match build_web_replay_input(&request) {
        Ok(input) => {
            let source = input.source;
            let label = input.label;
            let imu_name = input.imu_name;
            let gnss_name = input.gnss_name;
            let mut data = match progress {
                Some(progress) => {
                    sim::visualizer::pipeline::generic::build_generic_replay_plot_data_with_progress(
                        &input.replay,
                        misalignment,
                        ekf_cfg,
                        gnss_outages,
                        progress,
                    )
                }
                None => sim::visualizer::pipeline::generic::build_generic_replay_plot_data(
                    &input.replay,
                    misalignment,
                    ekf_cfg,
                    gnss_outages,
                ),
            };
            sim::visualizer::replay_job::decimate_for_transport(
                &mut data,
                sim::visualizer::replay_job::WEB_TRANSPORT_MAX_POINTS_PER_TRACE,
            );
            match serde_json::to_string(&data) {
                Ok(json) => WebReplayJobResponse {
                    ok: true,
                    job_id,
                    label,
                    source,
                    imu_name: Some(imu_name),
                    gnss_name: Some(gnss_name),
                    json: Some(json),
                    error: None,
                },
                Err(err) => WebReplayJobResponse {
                    ok: false,
                    job_id,
                    label,
                    source,
                    imu_name: None,
                    gnss_name: None,
                    json: None,
                    error: Some(format!("failed to serialize replay data: {err}")),
                },
            }
        }
        Err(err) => WebReplayJobResponse {
            ok: false,
            job_id,
            label: fallback_label,
            source: "unknown".to_string(),
            imu_name: None,
            gnss_name: None,
            json: None,
            error: Some(format!("replay failed: {err}")),
        },
    }
}

#[cfg(target_arch = "wasm32")]
fn build_web_replay_plot_data(
    request: WebReplayJobRequest,
    progress: Option<&mut dyn FnMut(sim::visualizer::pipeline::generic::GenericReplayProgress)>,
) -> anyhow::Result<sim::visualizer::model::PlotData> {
    let source_kind = request
        .source
        .as_ref()
        .and_then(|source| source.kind.as_deref())
        .unwrap_or("csv")
        .to_ascii_lowercase();
    let mut data = if source_kind == "synthetic" {
        build_web_synthetic_plot_data(&request, progress)?
    } else {
        let input = build_web_replay_input(&request)?;
        let misalignment = web_request_misalignment(&request);
        let ekf_cfg = web_request_ekf_cfg(&request);
        let gnss_outages = web_request_gnss_outages(&request);
        match progress {
            Some(progress) => {
                sim::visualizer::pipeline::generic::build_generic_replay_plot_data_with_progress(
                    &input.replay,
                    misalignment,
                    ekf_cfg,
                    gnss_outages,
                    progress,
                )
            }
            None => sim::visualizer::pipeline::generic::build_generic_replay_plot_data(
                &input.replay,
                misalignment,
                ekf_cfg,
                gnss_outages,
            ),
        }
    };
    sim::visualizer::replay_job::decimate_for_transport(
        &mut data,
        sim::visualizer::replay_job::WEB_TRANSPORT_MAX_POINTS_PER_TRACE,
    );
    Ok(data)
}

#[cfg(target_arch = "wasm32")]
fn build_web_replay_input(request: &WebReplayJobRequest) -> anyhow::Result<WebReplayInput> {
    let source = request.source.as_ref();
    let kind = source
        .and_then(|source| source.kind.as_deref())
        .unwrap_or("csv")
        .to_ascii_lowercase();
    match kind.as_str() {
        "csv" | "generic" | "real" => build_web_csv_replay_input(source, request),
        "synthetic" => anyhow::bail!("synthetic replay must be handled by the synthetic adapter"),
        other => anyhow::bail!("unsupported replay source kind '{other}'"),
    }
}

#[cfg(target_arch = "wasm32")]
fn build_web_csv_replay_input(
    source: Option<&WebReplayJobSource>,
    request: &WebReplayJobRequest,
) -> anyhow::Result<WebReplayInput> {
    let imu_csv = source
        .and_then(|source| source.imu_csv.as_deref())
        .or(request.imu_csv.as_deref())
        .unwrap_or_default();
    let gnss_csv = source
        .and_then(|source| source.gnss_csv.as_deref())
        .or(request.gnss_csv.as_deref())
        .unwrap_or_default();
    let reference_attitude_csv = source
        .and_then(|source| source.reference_attitude_csv.as_deref())
        .or(request.reference_attitude_csv.as_deref());
    let reference_mount_csv = source
        .and_then(|source| source.reference_mount_csv.as_deref())
        .or(request.reference_mount_csv.as_deref());
    let replay = sim::visualizer::pipeline::generic::parse_generic_replay_csvs_with_refs(
        imu_csv,
        gnss_csv,
        reference_attitude_csv,
        reference_mount_csv,
    )?;
    Ok(WebReplayInput {
        source: "csv".to_string(),
        label: source
            .and_then(|source| source.label.clone())
            .or_else(|| request.label.clone())
            .unwrap_or_else(|| "CSV replay".to_string()),
        imu_name: source
            .and_then(|source| source.imu_name.clone())
            .or_else(|| request.imu_name.clone())
            .unwrap_or_else(|| "imu.csv".to_string()),
        gnss_name: source
            .and_then(|source| source.gnss_name.clone())
            .or_else(|| request.gnss_name.clone())
            .unwrap_or_else(|| "gnss.csv".to_string()),
        replay,
    })
}

#[cfg(target_arch = "wasm32")]
fn build_web_synthetic_replay_job_response(
    request: WebReplayJobRequest,
    progress: Option<&mut dyn FnMut(sim::visualizer::pipeline::generic::GenericReplayProgress)>,
) -> WebReplayJobResponse {
    let job_id = request.job_id;
    let fallback_label = request
        .label
        .clone()
        .unwrap_or_else(|| "Synthetic replay".to_string());
    let Some(source) = request.source.as_ref() else {
        return WebReplayJobResponse {
            ok: false,
            job_id,
            label: fallback_label,
            source: "synthetic".to_string(),
            imu_name: None,
            gnss_name: None,
            json: None,
            error: Some("synthetic replay source is missing".to_string()),
        };
    };
    let motion_label = source
        .motion_label
        .as_deref()
        .unwrap_or("synthetic.scenario");
    let motion_text = source.motion_text.as_deref().map(str::to_string);
    let noise = match source
        .noise_mode
        .as_deref()
        .unwrap_or("truth")
        .to_ascii_lowercase()
        .as_str()
    {
        "truth" | "none" | "zero" => {
            sim::visualizer::pipeline::synthetic::SyntheticNoiseMode::Truth
        }
        "low" => sim::visualizer::pipeline::synthetic::SyntheticNoiseMode::Low,
        "mid" | "medium" => sim::visualizer::pipeline::synthetic::SyntheticNoiseMode::Mid,
        "high" => sim::visualizer::pipeline::synthetic::SyntheticNoiseMode::High,
        other => {
            return WebReplayJobResponse {
                ok: false,
                job_id,
                label: fallback_label,
                source: "synthetic".to_string(),
                imu_name: None,
                gnss_name: None,
                json: None,
                error: Some(format!("unsupported synthetic noise mode '{other}'")),
            };
        }
    };
    let synth_cfg = sim::visualizer::pipeline::synthetic::SyntheticVisualizerConfig {
        motion_def: None,
        motion_label: motion_label.to_string(),
        motion_text,
        noise_mode: noise,
        disable_imu_noise: false,
        disable_gnss_noise: false,
        seed: source.seed.unwrap_or(1),
        mount_rpy_deg: source.mount_rpy_deg.unwrap_or([5.0, -5.0, 5.0]),
        imu_hz: source.imu_hz.unwrap_or(100.0),
        gnss_hz: source.gnss_hz.unwrap_or(2.0),
        gnss_time_shift_ms: source.gnss_time_shift_ms.unwrap_or(0.0),
        early_vel_bias_ned_mps: source.early_vel_bias_ned_mps.unwrap_or([0.0, 0.0, 0.0]),
        early_fault_window_s: source
            .early_fault_window_s
            .map(|window| (window[0], window[1])),
    };
    let data_result = match progress {
        Some(progress) => {
            sim::visualizer::pipeline::synthetic::build_synthetic_plot_data_with_progress(
                &synth_cfg,
                web_request_misalignment(&request),
                web_request_ekf_cfg(&request),
                web_request_gnss_outages(&request),
                progress,
            )
        }
        None => sim::visualizer::pipeline::synthetic::build_synthetic_plot_data(
            &synth_cfg,
            web_request_misalignment(&request),
            web_request_ekf_cfg(&request),
            web_request_gnss_outages(&request),
        ),
    };
    let label = source
        .label
        .clone()
        .or_else(|| request.label.clone())
        .unwrap_or_else(|| format!("Synthetic: {motion_label}"));
    match data_result {
        Ok(mut data) => {
            sim::visualizer::replay_job::decimate_for_transport(
                &mut data,
                sim::visualizer::replay_job::WEB_TRANSPORT_MAX_POINTS_PER_TRACE,
            );
            match serde_json::to_string(&data) {
                Ok(json) => WebReplayJobResponse {
                    ok: true,
                    job_id,
                    label,
                    source: "synthetic".to_string(),
                    imu_name: Some("synthetic IMU".to_string()),
                    gnss_name: Some("synthetic GNSS".to_string()),
                    json: Some(json),
                    error: None,
                },
                Err(err) => WebReplayJobResponse {
                    ok: false,
                    job_id,
                    label,
                    source: "synthetic".to_string(),
                    imu_name: None,
                    gnss_name: None,
                    json: None,
                    error: Some(format!("failed to serialize replay data: {err}")),
                },
            }
        }
        Err(err) => WebReplayJobResponse {
            ok: false,
            job_id,
            label,
            source: "synthetic".to_string(),
            imu_name: None,
            gnss_name: None,
            json: None,
            error: Some(format!("synthetic replay failed: {err}")),
        },
    }
}

#[cfg(target_arch = "wasm32")]
fn build_web_synthetic_plot_data(
    request: &WebReplayJobRequest,
    progress: Option<&mut dyn FnMut(sim::visualizer::pipeline::generic::GenericReplayProgress)>,
) -> anyhow::Result<sim::visualizer::model::PlotData> {
    let Some(source) = request.source.as_ref() else {
        anyhow::bail!("synthetic replay source is missing");
    };
    let motion_label = source
        .motion_label
        .as_deref()
        .unwrap_or("synthetic.scenario");
    let motion_text = source.motion_text.as_deref().map(str::to_string);
    let noise = match source
        .noise_mode
        .as_deref()
        .unwrap_or("truth")
        .to_ascii_lowercase()
        .as_str()
    {
        "truth" | "none" | "zero" => {
            sim::visualizer::pipeline::synthetic::SyntheticNoiseMode::Truth
        }
        "low" => sim::visualizer::pipeline::synthetic::SyntheticNoiseMode::Low,
        "mid" | "medium" => sim::visualizer::pipeline::synthetic::SyntheticNoiseMode::Mid,
        "high" => sim::visualizer::pipeline::synthetic::SyntheticNoiseMode::High,
        other => anyhow::bail!("unsupported synthetic noise mode '{other}'"),
    };
    let synth_cfg = sim::visualizer::pipeline::synthetic::SyntheticVisualizerConfig {
        motion_def: None,
        motion_label: motion_label.to_string(),
        motion_text,
        noise_mode: noise,
        disable_imu_noise: false,
        disable_gnss_noise: false,
        seed: source.seed.unwrap_or(1),
        mount_rpy_deg: source.mount_rpy_deg.unwrap_or([5.0, -5.0, 5.0]),
        imu_hz: source.imu_hz.unwrap_or(100.0),
        gnss_hz: source.gnss_hz.unwrap_or(2.0),
        gnss_time_shift_ms: source.gnss_time_shift_ms.unwrap_or(0.0),
        early_vel_bias_ned_mps: source.early_vel_bias_ned_mps.unwrap_or([0.0, 0.0, 0.0]),
        early_fault_window_s: source
            .early_fault_window_s
            .map(|window| (window[0], window[1])),
    };
    match progress {
        Some(progress) => {
            sim::visualizer::pipeline::synthetic::build_synthetic_plot_data_with_progress(
                &synth_cfg,
                web_request_misalignment(request),
                web_request_ekf_cfg(request),
                web_request_gnss_outages(request),
                progress,
            )
        }
        None => sim::visualizer::pipeline::synthetic::build_synthetic_plot_data(
            &synth_cfg,
            web_request_misalignment(request),
            web_request_ekf_cfg(request),
            web_request_gnss_outages(request),
        ),
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn build_generic_replay_plot_data_json(
    imu_csv: &str,
    gnss_csv: &str,
    reference_attitude_csv: Option<String>,
    reference_mount_csv: Option<String>,
) -> std::result::Result<String, JsValue> {
    let data = sim::visualizer::replay_job::run_generic_csv_replay_job(
        sim::visualizer::replay_job::GenericReplayCsvJob {
            imu_csv,
            gnss_csv,
            reference_attitude_csv: reference_attitude_csv.as_deref(),
            reference_mount_csv: reference_mount_csv.as_deref(),
            config: sim::visualizer::replay_job::GenericReplayJobConfig::web_transport(),
        },
    )
    .map_err(|err| JsValue::from_str(&format!("CSV replay failed: {err}")))?;
    serde_json::to_string(&data)
        .map_err(|err| JsValue::from_str(&format!("failed to serialize replay data: {err}")))
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_misalignment(s: &str) -> Result<EkfImuSource, String> {
    EkfImuSource::from_cli_value(s)
}

#[cfg(not(target_arch = "wasm32"))]
fn dump_traces_near_time(group: &str, traces: &[Trace], t_s: f64, window_s: f64) {
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

#[cfg(not(target_arch = "wasm32"))]
fn print_reference_error_summaries(data: &PlotData) {
    for (group, system, state, trace_name, reference_name, mode) in [
        (
            "pos",
            "ESKF",
            "N",
            "ESKF posN [m]",
            "GNSS posN [m]",
            SummaryMode::Linear,
        ),
        (
            "pos",
            "ESKF",
            "E",
            "ESKF posE [m]",
            "GNSS posE [m]",
            SummaryMode::Linear,
        ),
        (
            "pos",
            "ESKF",
            "D",
            "ESKF posD [m]",
            "GNSS posD [m]",
            SummaryMode::Linear,
        ),
        (
            "pos",
            "Loose",
            "N",
            "Loose posN [m]",
            "GNSS posN [m]",
            SummaryMode::Linear,
        ),
        (
            "pos",
            "Loose",
            "E",
            "Loose posE [m]",
            "GNSS posE [m]",
            SummaryMode::Linear,
        ),
        (
            "pos",
            "Loose",
            "D",
            "Loose posD [m]",
            "GNSS posD [m]",
            SummaryMode::Linear,
        ),
        (
            "vel",
            "ESKF",
            "N",
            "ESKF velN [m/s]",
            "GNSS velN [m/s]",
            SummaryMode::Linear,
        ),
        (
            "vel",
            "ESKF",
            "E",
            "ESKF velE [m/s]",
            "GNSS velE [m/s]",
            SummaryMode::Linear,
        ),
        (
            "vel",
            "ESKF",
            "D",
            "ESKF velD [m/s]",
            "GNSS velD [m/s]",
            SummaryMode::Linear,
        ),
        (
            "vel",
            "Loose",
            "N",
            "Loose velN [m/s]",
            "GNSS velN [m/s]",
            SummaryMode::Linear,
        ),
        (
            "vel",
            "Loose",
            "E",
            "Loose velE [m/s]",
            "GNSS velE [m/s]",
            SummaryMode::Linear,
        ),
        (
            "vel",
            "Loose",
            "D",
            "Loose velD [m/s]",
            "GNSS velD [m/s]",
            SummaryMode::Linear,
        ),
        (
            "att",
            "ESKF",
            "roll",
            "ESKF roll [deg]",
            "Reference roll [deg]",
            SummaryMode::AngleDeg,
        ),
        (
            "att",
            "ESKF",
            "pitch",
            "ESKF pitch [deg]",
            "Reference pitch [deg]",
            SummaryMode::AngleDeg,
        ),
        (
            "att",
            "ESKF",
            "yaw",
            "ESKF yaw [deg]",
            "Reference yaw [deg]",
            SummaryMode::AngleDeg,
        ),
        (
            "att",
            "Loose",
            "roll",
            "Loose roll [deg]",
            "Reference roll [deg]",
            SummaryMode::AngleDeg,
        ),
        (
            "att",
            "Loose",
            "pitch",
            "Loose pitch [deg]",
            "Reference pitch [deg]",
            SummaryMode::AngleDeg,
        ),
        (
            "att",
            "Loose",
            "yaw",
            "Loose yaw [deg]",
            "Reference yaw [deg]",
            SummaryMode::AngleDeg,
        ),
        (
            "mount",
            "ESKF",
            "roll",
            "ESKF mount roll [deg]",
            "Reference mount roll [deg]",
            SummaryMode::AngleDeg,
        ),
        (
            "mount",
            "ESKF",
            "pitch",
            "ESKF mount pitch [deg]",
            "Reference mount pitch [deg]",
            SummaryMode::AngleDeg,
        ),
        (
            "mount",
            "ESKF",
            "yaw",
            "ESKF mount yaw [deg]",
            "Reference mount yaw [deg]",
            SummaryMode::AngleDeg,
        ),
        (
            "mount",
            "Loose",
            "roll",
            "Loose residual mount roll [deg]",
            "Reference mount roll [deg]",
            SummaryMode::AngleDeg,
        ),
        (
            "mount",
            "Loose",
            "pitch",
            "Loose residual mount pitch [deg]",
            "Reference mount pitch [deg]",
            SummaryMode::AngleDeg,
        ),
        (
            "mount",
            "Loose",
            "yaw",
            "Loose residual mount yaw [deg]",
            "Reference mount yaw [deg]",
            SummaryMode::AngleDeg,
        ),
    ] {
        let traces = match (group, system) {
            ("pos", "ESKF") | ("vel", "ESKF") => data.eskf_cmp_pos_or_vel(group),
            ("pos", "Loose") => data.loose_cmp_pos.as_slice(),
            ("vel", "Loose") => data.loose_cmp_vel.as_slice(),
            ("att", "ESKF") => data.eskf_cmp_att.as_slice(),
            ("att", "Loose") => data.loose_cmp_att.as_slice(),
            ("mount", "ESKF") => data.eskf_misalignment.as_slice(),
            ("mount", "Loose") => data.loose_misalignment.as_slice(),
            _ => &[],
        };
        let reference_traces = match group {
            "pos" => data.eskf_cmp_pos.as_slice(),
            "vel" => data.eskf_cmp_vel.as_slice(),
            _ => traces,
        };
        let Some(trace) = find_trace(traces, trace_name) else {
            continue;
        };
        let Some(reference) = find_trace(reference_traces, reference_name) else {
            continue;
        };
        if let Some(summary) =
            summarize_trace_pair(system, state, trace, Some(reference), mode, None)
        {
            eprintln!(
                "[profile] ref_error group={} system={} state={} samples={} final_err={:.6} mae={:.6} rmse={:.6} p95={:.6} tail_span={:.6} tail_drift={:.6}",
                group,
                system,
                state,
                summary.sample_count,
                summary.final_error.unwrap_or(f64::NAN),
                summary.mean_abs_error.unwrap_or(f64::NAN),
                summary.rmse_error.unwrap_or(f64::NAN),
                summary.p95_abs_error.unwrap_or(f64::NAN),
                summary.tail_span_error.unwrap_or(f64::NAN),
                summary.tail_drift_value,
            );
        }
    }
    for (system, traces, trace_name) in [
        (
            "ESKF",
            data.eskf_misalignment.as_slice(),
            "ESKF mount quaternion error [deg]",
        ),
        (
            "Loose",
            data.loose_misalignment.as_slice(),
            "Loose mount quaternion error [deg]",
        ),
        (
            "Align",
            data.align_cmp_att.as_slice(),
            "Align mount quaternion error [deg]",
        ),
    ] {
        let Some(trace) = find_trace(traces, trace_name) else {
            continue;
        };
        if let Some(summary) =
            summarize_trace_pair(system, "mount_qerr", trace, None, SummaryMode::Linear, None)
        {
            eprintln!(
                "[profile] qerr system={} samples={} final={:.6} mean={:.6} max={:.6} tail_span={:.6} tail_drift={:.6}",
                system,
                summary.sample_count,
                summary.final_value,
                summary.mean_value,
                summary.max_value,
                summary.tail_span_value,
                summary.tail_drift_value,
            );
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn find_trace<'a>(traces: &'a [Trace], name: &str) -> Option<&'a Trace> {
    traces.iter().find(|trace| trace.name == name)
}

#[cfg(not(target_arch = "wasm32"))]
fn print_eskf_nhc_window_summaries(data: &PlotData, t0: f64) {
    if !t0.is_finite() {
        return;
    }
    for (start_rel, end_rel) in [(0.0, 60.0), (60.0, 120.0), (120.0, 240.0), (240.0, 600.0)] {
        let start = t0 + start_rel;
        let end = t0 + end_rel;
        for channel in ["Y", "Z"] {
            let innovation = find_trace(
                &data.eskf_nhc_innovation,
                &format!("ESKF NHC {channel} innovation [m/s]"),
            );
            let nis = find_trace(&data.eskf_nhc_nis, &format!("ESKF NHC {channel} NIS"));
            let h_mount = find_trace(
                &data.eskf_nhc_h_mount_norm,
                &format!("ESKF NHC {channel} mount H norm"),
            );
            let innovation_stats = innovation.map(|trace| window_stats(trace, start, end));
            let nis_stats = nis.map(|trace| window_stats(trace, start, end));
            let h_stats = h_mount.map(|trace| window_stats(trace, start, end));
            eprintln!(
                "[profile] eskf_nhc_window channel={} window=[{:.1},{:.1}]s updates={} innov_mean={:.6} innov_rms={:.6} innov_abs_sum={:.6} nis_mean={:.6} nis_max={:.6} h_mount_mean={:.6}",
                channel,
                start_rel,
                end_rel,
                innovation_stats.map_or(0, |s| s.count),
                innovation_stats.map_or(f64::NAN, |s| s.mean),
                innovation_stats.map_or(f64::NAN, |s| s.rms),
                innovation_stats.map_or(f64::NAN, |s| s.sum_abs),
                nis_stats.map_or(f64::NAN, |s| s.mean),
                nis_stats.map_or(f64::NAN, |s| s.max),
                h_stats.map_or(f64::NAN, |s| s.mean),
            );
            for axis in ["roll", "pitch", "yaw"] {
                let correction = find_trace(
                    &data.eskf_nhc_mount_dx,
                    &format!("ESKF NHC {channel} mount {axis} correction [deg/update]"),
                );
                let correction_stats = correction.map(|trace| window_stats(trace, start, end));
                eprintln!(
                    "[profile] eskf_nhc_mount_window channel={} axis={} window=[{:.1},{:.1}]s updates={} dx_sum_deg={:.9} dx_abs_sum_deg={:.9} dx_mean_deg={:.9}",
                    channel,
                    axis,
                    start_rel,
                    end_rel,
                    correction_stats.map_or(0, |s| s.count),
                    correction_stats.map_or(f64::NAN, |s| s.sum),
                    correction_stats.map_or(f64::NAN, |s| s.sum_abs),
                    correction_stats.map_or(f64::NAN, |s| s.mean),
                );
            }
        }
        for axis in ["roll", "pitch", "yaw"] {
            let name = format!("ESKF mount {axis} [deg]");
            let drift = find_trace(&data.eskf_misalignment, &name)
                .and_then(|trace| window_first_last_delta(trace, start, end));
            if let Some((first, last, delta)) = drift {
                eprintln!(
                    "[profile] eskf_mount_state_window axis={} window=[{:.1},{:.1}]s first_deg={:.9} last_deg={:.9} delta_deg={:.9}",
                    axis, start_rel, end_rel, first, last, delta
                );
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn print_loose_mount_window_summaries(data: &PlotData, t0: f64) {
    if !t0.is_finite() {
        return;
    }
    for (start_rel, end_rel) in [(0.0, 60.0), (60.0, 120.0), (120.0, 240.0), (240.0, 600.0)] {
        let start = t0 + start_rel;
        let end = t0 + end_rel;
        for axis in ["roll", "pitch", "yaw"] {
            let correction = find_trace(
                &data.loose_mount_dx,
                &format!("Loose residual mount {axis} correction [deg/update]"),
            );
            let correction_stats = correction.map(|trace| window_stats(trace, start, end));
            eprintln!(
                "[profile] loose_mount_update_window axis={} window=[{:.1},{:.1}]s updates={} dx_sum_deg={:.9} dx_abs_sum_deg={:.9} dx_mean_deg={:.9}",
                axis,
                start_rel,
                end_rel,
                correction_stats.map_or(0, |s| s.count),
                correction_stats.map_or(f64::NAN, |s| s.sum),
                correction_stats.map_or(f64::NAN, |s| s.sum_abs),
                correction_stats.map_or(f64::NAN, |s| s.mean),
            );
            let name = format!("Loose residual mount {axis} [deg]");
            let drift = find_trace(&data.loose_misalignment, &name)
                .and_then(|trace| window_first_last_delta(trace, start, end));
            if let Some((first, last, delta)) = drift {
                eprintln!(
                    "[profile] loose_mount_state_window axis={} window=[{:.1},{:.1}]s first_deg={:.9} last_deg={:.9} delta_deg={:.9}",
                    axis, start_rel, end_rel, first, last, delta
                );
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn print_loose_nhc_window_summaries(data: &PlotData, t0: f64) {
    if !t0.is_finite() {
        return;
    }
    for (start_rel, end_rel) in [(0.0, 60.0), (60.0, 120.0), (120.0, 240.0), (240.0, 600.0)] {
        let start = t0 + start_rel;
        let end = t0 + end_rel;
        for channel in ["Y", "Z"] {
            let innovation = find_trace(
                &data.loose_nhc_innovation,
                &format!("Loose NHC {channel} innovation [m/s]"),
            );
            let innovation_stats = innovation.map(|trace| window_stats(trace, start, end));
            eprintln!(
                "[profile] loose_nhc_window channel={} window=[{:.1},{:.1}]s updates={} innov_mean={:.6} innov_rms={:.6} innov_abs_sum={:.6}",
                channel,
                start_rel,
                end_rel,
                innovation_stats.map_or(0, |s| s.count),
                innovation_stats.map_or(f64::NAN, |s| s.mean),
                innovation_stats.map_or(f64::NAN, |s| s.rms),
                innovation_stats.map_or(f64::NAN, |s| s.sum_abs),
            );
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug)]
struct WindowStats {
    count: usize,
    sum: f64,
    sum_abs: f64,
    mean: f64,
    rms: f64,
    max: f64,
}

#[cfg(not(target_arch = "wasm32"))]
fn window_stats(trace: &Trace, start: f64, end: f64) -> WindowStats {
    let mut count = 0usize;
    let mut sum = 0.0;
    let mut sum_abs = 0.0;
    let mut sum_sq = 0.0;
    let mut max = f64::NEG_INFINITY;
    for point in &trace.points {
        let t = point[0];
        let v = point[1];
        if t >= start && t < end && v.is_finite() {
            count += 1;
            sum += v;
            sum_abs += v.abs();
            sum_sq += v * v;
            max = max.max(v);
        }
    }
    let n = count as f64;
    WindowStats {
        count,
        sum,
        sum_abs,
        mean: if count > 0 { sum / n } else { f64::NAN },
        rms: if count > 0 {
            (sum_sq / n).sqrt()
        } else {
            f64::NAN
        },
        max: if count > 0 { max } else { f64::NAN },
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn window_first_last_delta(trace: &Trace, start: f64, end: f64) -> Option<(f64, f64, f64)> {
    let mut first = None;
    let mut last = None;
    for point in &trace.points {
        let t = point[0];
        let v = point[1];
        if t >= start && t < end && v.is_finite() {
            first.get_or_insert(v);
            last = Some(v);
        }
    }
    match (first, last) {
        (Some(first), Some(last)) => Some((first, last, last - first)),
        _ => None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
trait PlotDataProfileExt {
    fn eskf_cmp_pos_or_vel(&self, group: &str) -> &[Trace];
}

#[cfg(not(target_arch = "wasm32"))]
impl PlotDataProfileExt for PlotData {
    fn eskf_cmp_pos_or_vel(&self, group: &str) -> &[Trace] {
        match group {
            "pos" => &self.eskf_cmp_pos,
            "vel" => &self.eskf_cmp_vel,
            _ => &[],
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn print_map_bounds(group: &str, traces: &[Trace]) {
    for trace in traces {
        let mut lon_min = f64::INFINITY;
        let mut lon_max = f64::NEG_INFINITY;
        let mut lat_min = f64::INFINITY;
        let mut lat_max = f64::NEG_INFINITY;
        let mut n = 0usize;
        for point in &trace.points {
            let lon = point[0];
            let lat = point[1];
            if lon.is_finite() && lat.is_finite() {
                lon_min = lon_min.min(lon);
                lon_max = lon_max.max(lon);
                lat_min = lat_min.min(lat);
                lat_max = lat_max.max(lat);
                n += 1;
            }
        }
        if n > 0 {
            eprintln!(
                "[profile] map_bounds group={} trace={} points={} lon=[{:.8},{:.8}] lat=[{:.8},{:.8}]",
                group, trace.name, n, lon_min, lon_max, lat_min, lat_max
            );
        }
    }
}
