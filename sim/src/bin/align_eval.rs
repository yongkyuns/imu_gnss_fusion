use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::align::{AlignConfig, GRAVITY_MPS2};
use sim::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_raw_samples, extract_nav2_pvt_obs, fit_linear_map,
    parse_ubx_frames, sensor_meta, unwrap_counter,
};
use sim::visualizer::math::nearest_master_ms;
use sim::visualizer::model::{EkfImuSource, ImuPacket};
use sim::visualizer::pipeline::align_replay::{
    AlignReplayData, BootstrapConfig, ImuReplayConfig, axis_angle_deg, build_align_replay,
    build_fusion_align_replay, quat_rotate,
    signed_projected_axis_angle_deg,
};
use sim::visualizer::pipeline::timebase::{MasterTimeline, build_master_timeline};

#[derive(Parser, Debug)]
#[command(name = "align_eval")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,

    #[arg(long)]
    max_records: Option<usize>,

    #[arg(long)]
    residual_csv: Option<PathBuf>,

    #[arg(long)]
    bootstrap_debug_csv: Option<PathBuf>,

    #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
    horiz_accel_report: bool,

    #[arg(long)]
    horiz_accel_csv: Option<PathBuf>,

    #[arg(long, default_value_t = 0.2)]
    horiz_accel_min_norm_mps2: f64,

    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    alg_valid_only: bool,

    #[arg(long, default_value_t = 300)]
    stationary_samples: usize,

    #[arg(long, default_value_t = 0.05)]
    bootstrap_ema_alpha: f32,

    #[arg(long, default_value_t = 0.35)]
    bootstrap_max_speed_mps: f32,

    #[arg(long, default_value_t = false)]
    tune: bool,

    #[arg(long, default_value = "align")]
    filter_mode: String,

    #[arg(long, default_value_t = 2)]
    tune_passes: usize,

    #[arg(long)]
    q_roll_std_deg: Option<f32>,
    #[arg(long)]
    q_pitch_std_deg: Option<f32>,
    #[arg(long)]
    q_yaw_std_deg: Option<f32>,
    #[arg(long)]
    r_gravity_std_mps2: Option<f32>,
    #[arg(long)]
    r_horiz_heading_std_deg: Option<f32>,
    #[arg(long)]
    r_turn_gyro_std_dps: Option<f32>,
    #[arg(long)]
    turn_gyro_yaw_scale: Option<f32>,
    #[arg(long)]
    r_turn_heading_std_deg: Option<f32>,
    #[arg(long)]
    gravity_lpf_alpha: Option<f32>,
    #[arg(long)]
    min_speed_mps: Option<f32>,
    #[arg(long)]
    min_turn_rate_dps: Option<f32>,
    #[arg(long)]
    min_lat_acc_mps2: Option<f32>,
    #[arg(long)]
    min_long_acc_mps2: Option<f32>,
    #[arg(long)]
    max_stationary_gyro_dps: Option<f32>,
    #[arg(long)]
    max_stationary_accel_norm_err_mps2: Option<f32>,

    #[arg(long, action = clap::ArgAction::Set)]
    use_gravity: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set)]
    use_turn_gyro: Option<bool>,
}

#[derive(Clone)]
struct AlignDataset {
    frames: Vec<UbxFrame>,
    timeline: MasterTimeline,
    total_nav_events: usize,
}

#[derive(Clone)]
struct BootstrapDetector {
    cfg: BootstrapConfig,
    gyro_ema: Option<f32>,
    accel_err_ema: Option<f32>,
    speed_ema: Option<f32>,
    speed_rate_ema: Option<f32>,
    course_rate_ema: Option<f32>,
    stationary_accel: Vec<[f32; 3]>,
    stationary_gyro: Vec<[f32; 3]>,
}

#[derive(Clone, Copy, Debug)]
struct ResidualSample {
    t_s: f64,
    align_roll_deg: f64,
    align_pitch_deg: f64,
    align_yaw_deg: f64,
    alg_roll_deg: f64,
    alg_pitch_deg: f64,
    alg_yaw_deg: f64,
    err_roll_deg: f64,
    err_pitch_deg: f64,
    err_yaw_deg: f64,
    sigma_roll_deg: f64,
    sigma_pitch_deg: f64,
    sigma_yaw_deg: f64,
    course_rate_dps: f64,
    a_lat_mps2: f64,
    a_long_mps2: f64,
    rot_err_deg: f64,
    fwd_err_deg: f64,
    down_err_deg: f64,
    fwd_err_signed_deg: f64,
    down_err_signed_deg: f64,
    horiz_angle_err_deg: f64,
    horiz_effective_std_deg: f64,
    horiz_gnss_norm_mps2: f64,
    horiz_imu_norm_mps2: f64,
    horiz_speed_q: f64,
    horiz_accel_q: f64,
    horiz_straight_q: f64,
    horiz_turn_q: f64,
    horiz_dominance_q: f64,
    horiz_turn_core_valid: bool,
    horiz_straight_core_valid: bool,
    horiz_applied: bool,
    gravity_applied: bool,
    gravity_quasi_static: bool,
    planar_gyro_valid: bool,
    planar_gyro_residual_x_dps: f64,
    planar_gyro_residual_y_dps: f64,
}

#[derive(Clone, Copy, Debug)]
struct BootstrapDebugSample {
    t_s: f64,
    speed_mps: f64,
    speed_ema_mps: f64,
    speed_rate_mps2: f64,
    speed_rate_ema_mps2: f64,
    course_rate_dps: f64,
    course_rate_ema_dps: f64,
    gyro_norm_dps: f64,
    gyro_ema_dps: f64,
    accel_norm_err_mps2: f64,
    accel_err_ema_mps2: f64,
    stationary: bool,
    sample_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct EvalMetrics {
    n_samples: usize,
    init_time_s: f64,
    horizon_s: f64,
    coverage: f64,
    rmse_roll_deg: f64,
    rmse_pitch_deg: f64,
    rmse_yaw_deg: f64,
    mae_roll_deg: f64,
    mae_pitch_deg: f64,
    mae_yaw_deg: f64,
    mean_rot_err_deg: f64,
    max_rot_err_deg: f64,
    final_rot_err_deg: f64,
    mean_fwd_err_deg: f64,
    max_fwd_err_deg: f64,
    final_fwd_err_deg: f64,
    mean_down_err_deg: f64,
    max_down_err_deg: f64,
    final_down_err_deg: f64,
    final_err_roll_deg: f64,
    final_err_pitch_deg: f64,
    final_err_yaw_deg: f64,
    score: f64,
}

#[derive(Clone)]
struct EvalResult {
    cfg: AlignConfig,
    bootstrap_cfg: BootstrapConfig,
    metrics: EvalMetrics,
    samples: Vec<ResidualSample>,
}

#[derive(Clone, Copy, Debug)]
struct HorizAccelQualitySample {
    t_s: f64,
    gnss_long_mps2: f64,
    gnss_lat_mps2: f64,
    imu_long_mps2: f64,
    imu_lat_mps2: f64,
    gnss_norm_mps2: f64,
    imu_norm_mps2: f64,
    angle_err_deg: f64,
    long_resid_mps2: f64,
    lat_resid_mps2: f64,
    straight_core_valid: bool,
    turn_core_valid: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct HorizAccelQualityReport {
    n_total: usize,
    n_valid: usize,
    mean_abs_angle_err_deg: f64,
    p90_abs_angle_err_deg: f64,
    within_10_deg_frac: f64,
    within_20_deg_frac: f64,
    within_30_deg_frac: f64,
    long_bias_mps2: f64,
    lat_bias_mps2: f64,
    long_rmse_mps2: f64,
    lat_rmse_mps2: f64,
    long_corr_zero_lag: f64,
    lat_corr_zero_lag: f64,
    long_best_corr: f64,
    lat_best_corr: f64,
    long_best_lag_windows: isize,
    lat_best_lag_windows: isize,
    median_dt_s: f64,
}

#[derive(Clone, Copy)]
enum TuneParam {
    BootstrapEmaAlpha,
    BootstrapMaxSpeedMps,
    RGravityStdMps2,
    RTurnGyroStdDps,
    GravityLpfAlpha,
    MinSpeedMps,
    MinTurnRateDps,
    MinLatAccMps2,
    MinLongAccMps2,
    MaxStationaryGyroDps,
    MaxStationaryAccelNormErrMps2,
}

impl TuneParam {
    fn name(self) -> &'static str {
        match self {
            Self::BootstrapEmaAlpha => "bootstrap_ema_alpha",
            Self::BootstrapMaxSpeedMps => "bootstrap_max_speed_mps",
            Self::RGravityStdMps2 => "r_gravity_std_mps2",
            Self::RTurnGyroStdDps => "r_turn_gyro_std_dps",
            Self::GravityLpfAlpha => "gravity_lpf_alpha",
            Self::MinSpeedMps => "min_speed_mps",
            Self::MinTurnRateDps => "min_turn_rate_dps",
            Self::MinLatAccMps2 => "min_lat_acc_mps2",
            Self::MinLongAccMps2 => "min_long_acc_mps2",
            Self::MaxStationaryGyroDps => "max_stationary_gyro_dps",
            Self::MaxStationaryAccelNormErrMps2 => "max_stationary_accel_norm_err_mps2",
        }
    }

    fn candidates(self, _cfg: &AlignConfig) -> &'static [f32] {
        match self {
            Self::BootstrapEmaAlpha => &[0.02, 0.03, 0.05, 0.08, 0.12, 0.18],
            Self::BootstrapMaxSpeedMps => &[0.15, 0.25, 0.35, 0.50, 0.75],
            Self::RGravityStdMps2 => &[0.05, 0.08, 0.12, 0.18],
            Self::RTurnGyroStdDps => &[0.08, 0.15, 0.20, 0.30, 0.50],
            Self::GravityLpfAlpha => &[0.03, 0.05, 0.08, 0.12, 0.18],
            Self::MinSpeedMps => &[2.5, 3.5, 4.0, 4.5, 5.5],
            Self::MinTurnRateDps => &[1.5, 2.0, 3.0, 4.0, 5.0],
            Self::MinLatAccMps2 => &[0.15, 0.25, 0.35, 0.50, 0.70],
            Self::MinLongAccMps2 => &[0.10, 0.15, 0.25, 0.35, 0.50],
            Self::MaxStationaryGyroDps => &[0.4, 0.6, 0.8, 1.0, 1.2],
            Self::MaxStationaryAccelNormErrMps2 => &[0.10, 0.15, 0.20, 0.30, 0.40],
        }
    }

    fn current(self, cfg: &AlignConfig) -> f32 {
        match self {
            Self::BootstrapEmaAlpha | Self::BootstrapMaxSpeedMps => {
                unreachable!("bootstrap params are handled separately")
            }
            Self::RGravityStdMps2 => cfg.r_gravity_std_mps2,
            Self::RTurnGyroStdDps => cfg.r_turn_gyro_std_radps.to_degrees(),
            Self::GravityLpfAlpha => cfg.gravity_lpf_alpha,
            Self::MinSpeedMps => cfg.min_speed_mps,
            Self::MinTurnRateDps => cfg.min_turn_rate_radps.to_degrees(),
            Self::MinLatAccMps2 => cfg.min_lat_acc_mps2,
            Self::MinLongAccMps2 => cfg.min_long_acc_mps2,
            Self::MaxStationaryGyroDps => cfg.max_stationary_gyro_radps.to_degrees(),
            Self::MaxStationaryAccelNormErrMps2 => cfg.max_stationary_accel_norm_err_mps2,
        }
    }

    fn set(self, cfg: &mut AlignConfig, value: f32) {
        match self {
            Self::BootstrapEmaAlpha | Self::BootstrapMaxSpeedMps => {
                unreachable!("bootstrap params are handled separately")
            }
            Self::RGravityStdMps2 => cfg.r_gravity_std_mps2 = value,
            Self::RTurnGyroStdDps => cfg.r_turn_gyro_std_radps = value.to_radians(),
            Self::GravityLpfAlpha => cfg.gravity_lpf_alpha = value,
            Self::MinSpeedMps => cfg.min_speed_mps = value,
            Self::MinTurnRateDps => cfg.min_turn_rate_radps = value.to_radians(),
            Self::MinLatAccMps2 => cfg.min_lat_acc_mps2 = value,
            Self::MinLongAccMps2 => cfg.min_long_acc_mps2 = value,
            Self::MaxStationaryGyroDps => cfg.max_stationary_gyro_radps = value.to_radians(),
            Self::MaxStationaryAccelNormErrMps2 => cfg.max_stationary_accel_norm_err_mps2 = value,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let filter_mode = parse_filter_mode(&args.filter_mode)?;
    let cfg = config_from_args(&args);
    let bootstrap_cfg = bootstrap_config_from_args(&args, &cfg);
    let dataset = load_dataset(&args.logfile, args.max_records, args.alg_valid_only)?;
    if let Some(path) = &args.bootstrap_debug_csv {
        let samples = build_bootstrap_debug_trace(&dataset, &bootstrap_cfg);
        write_bootstrap_debug_csv(path, &samples)?;
        eprintln!("wrote bootstrap debug CSV: {}", path.display());
    }

    let base = evaluate_dispatch(filter_mode, &dataset, &cfg, &bootstrap_cfg, &args)?;
    print_metrics("baseline", &base.metrics);

    if args.horiz_accel_report || args.horiz_accel_csv.is_some() {
        let replay = build_replay_for_mode(filter_mode, &dataset, &cfg, &bootstrap_cfg);
        let quality_samples =
            build_horiz_accel_quality_samples(&replay, args.horiz_accel_min_norm_mps2);
        let report = summarize_horiz_accel_quality(&quality_samples);
        print_horiz_accel_quality_report("horiz-accel", &report);
        let straight_samples: Vec<_> = quality_samples
            .iter()
            .copied()
            .filter(|s| s.straight_core_valid)
            .collect();
        let turn_samples: Vec<_> = quality_samples
            .iter()
            .copied()
            .filter(|s| s.turn_core_valid)
            .collect();
        print_horiz_accel_quality_report(
            "horiz-accel-straight",
            &summarize_horiz_accel_quality(&straight_samples),
        );
        print_horiz_accel_quality_report(
            "horiz-accel-turn",
            &summarize_horiz_accel_quality(&turn_samples),
        );
        if let Some(path) = &args.horiz_accel_csv {
            write_horiz_accel_csv(path, &quality_samples)?;
            eprintln!("wrote horizontal accel CSV: {}", path.display());
        }
    }

    let mut best = base;
    if args.tune {
        if !matches!(filter_mode, FilterMode::Align) {
            bail!("tuning is only supported for --filter-mode align");
        }
        let tuned = tune_config(&dataset, &best.cfg, &bootstrap_cfg, args.tune_passes)?;
        best = tuned;
        print_metrics("best", &best.metrics);
        print_config(&best.cfg, &best.bootstrap_cfg);
    } else {
        print_config(&best.cfg, &bootstrap_cfg);
    }

    if let Some(path) = &args.residual_csv {
        write_residual_csv(path, &best.samples)?;
        eprintln!("wrote residual CSV: {}", path.display());
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FilterMode {
    Align,
    Fusion,
}

fn parse_filter_mode(s: &str) -> Result<FilterMode> {
    match s {
        "align" => Ok(FilterMode::Align),
        "fusion" => Ok(FilterMode::Fusion),
        _ => bail!("unknown --filter-mode {}, expected align or fusion", s),
    }
}

fn config_from_args(args: &Args) -> AlignConfig {
    let mut cfg = AlignConfig::default();
    if let Some(v) = args.q_roll_std_deg {
        cfg.q_mount_std_rad[0] = v.to_radians();
    }
    if let Some(v) = args.q_pitch_std_deg {
        cfg.q_mount_std_rad[1] = v.to_radians();
    }
    if let Some(v) = args.q_yaw_std_deg {
        cfg.q_mount_std_rad[2] = v.to_radians();
    }
    if let Some(v) = args.r_gravity_std_mps2 {
        cfg.r_gravity_std_mps2 = v;
    }
    if let Some(v) = args.r_horiz_heading_std_deg {
        cfg.r_horiz_heading_std_rad = v.to_radians();
    }
    if let Some(v) = args.r_turn_gyro_std_dps {
        cfg.r_turn_gyro_std_radps = v.to_radians();
    }
    if let Some(v) = args.turn_gyro_yaw_scale {
        cfg.turn_gyro_yaw_scale = v;
    }
    if let Some(v) = args.r_turn_heading_std_deg {
        cfg.r_turn_heading_std_rad = v.to_radians();
    }
    if let Some(v) = args.gravity_lpf_alpha {
        cfg.gravity_lpf_alpha = v;
    }
    if let Some(v) = args.min_speed_mps {
        cfg.min_speed_mps = v;
    }
    if let Some(v) = args.min_turn_rate_dps {
        cfg.min_turn_rate_radps = v.to_radians();
    }
    if let Some(v) = args.min_lat_acc_mps2 {
        cfg.min_lat_acc_mps2 = v;
    }
    if let Some(v) = args.min_long_acc_mps2 {
        cfg.min_long_acc_mps2 = v;
    }
    if let Some(v) = args.max_stationary_gyro_dps {
        cfg.max_stationary_gyro_radps = v.to_radians();
    }
    if let Some(v) = args.max_stationary_accel_norm_err_mps2 {
        cfg.max_stationary_accel_norm_err_mps2 = v;
    }
    if let Some(v) = args.use_gravity {
        cfg.use_gravity = v;
    }
    if let Some(v) = args.use_turn_gyro {
        cfg.use_turn_gyro = v;
    }
    cfg
}

fn bootstrap_config_from_args(args: &Args, cfg: &AlignConfig) -> BootstrapConfig {
    BootstrapConfig {
        ema_alpha: args.bootstrap_ema_alpha,
        max_speed_mps: args.bootstrap_max_speed_mps,
        stationary_samples: args.stationary_samples,
        max_gyro_radps: cfg.max_stationary_gyro_radps,
        max_accel_norm_err_mps2: cfg.max_stationary_accel_norm_err_mps2,
        max_speed_rate_mps2: 0.15,
        max_course_rate_radps: 1.0_f32.to_radians(),
    }
}

fn load_dataset(
    logfile: &PathBuf,
    max_records: Option<usize>,
    _alg_valid_only: bool,
) -> Result<AlignDataset> {
    let bytes =
        fs::read(logfile).with_context(|| format!("failed to read {}", logfile.display()))?;
    let frames = parse_ubx_frames(&bytes, max_records);
    if frames.is_empty() {
        bail!("no UBX frames parsed");
    }
    let timeline = build_master_timeline(&frames);
    if !timeline.has_itow {
        bail!("log does not contain a usable iTOW timeline");
    }

    let total_nav_events = collect_nav_events(&frames, &timeline).len();
    if total_nav_events < 2 {
        bail!("need at least two NAV2-PVT observations");
    }
    if build_imu_packets(&frames, &timeline)?.is_empty() {
        bail!("no complete ESF-RAW IMU packets found");
    }

    Ok(AlignDataset {
        frames,
        timeline,
        total_nav_events,
    })
}

fn build_imu_packets(frames: &[UbxFrame], timeline: &MasterTimeline) -> Result<Vec<ImuPacket>> {
    let mut raw_seq = Vec::<u64>::new();
    let mut raw_tag = Vec::<u64>::new();
    let mut raw_dtype = Vec::<u8>::new();
    let mut raw_val = Vec::<f64>::new();
    for frame in frames {
        for (tag, sw) in extract_esf_raw_samples(frame) {
            let (_name, _unit, scale) = sensor_meta(sw.dtype);
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
        if let Some(t_ms) = sim::visualizer::math::nearest_master_ms(frame.seq, &timeline.masters)
            && let Some(obs) = extract_nav2_pvt_obs(frame)
            && obs.fix_ok
            && !obs.invalid_llh
        {
            nav_events.push((t_ms, obs));
        }
    }
    nav_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    nav_events
}

impl BootstrapDetector {
    fn new(cfg: BootstrapConfig) -> Self {
        Self {
            cfg,
            gyro_ema: None,
            accel_err_ema: None,
            speed_ema: None,
            speed_rate_ema: None,
            course_rate_ema: None,
            stationary_accel: Vec::new(),
            stationary_gyro: Vec::new(),
        }
    }

    fn update(
        &mut self,
        accel_b: [f32; 3],
        gyro_radps: [f32; 3],
        speed_mps: f32,
        speed_rate_mps2: Option<f32>,
        course_rate_radps: Option<f32>,
    ) -> bool {
        let gyro_norm = norm3(gyro_radps);
        let accel_err = (norm3(accel_b) - GRAVITY_MPS2).abs();
        self.gyro_ema = Some(ema_update(self.gyro_ema, gyro_norm, self.cfg.ema_alpha));
        self.accel_err_ema = Some(ema_update(
            self.accel_err_ema,
            accel_err,
            self.cfg.ema_alpha,
        ));
        self.speed_ema = Some(ema_update(self.speed_ema, speed_mps, self.cfg.ema_alpha));
        if let Some(speed_rate_mps2) = speed_rate_mps2 {
            self.speed_rate_ema = Some(ema_update(
                self.speed_rate_ema,
                speed_rate_mps2.abs(),
                self.cfg.ema_alpha,
            ));
        }
        if let Some(course_rate_radps) = course_rate_radps {
            self.course_rate_ema = Some(ema_update(
                self.course_rate_ema,
                course_rate_radps.abs(),
                self.cfg.ema_alpha,
            ));
        }

        let low_dynamic = self.gyro_ema.unwrap_or(gyro_norm) <= self.cfg.max_gyro_radps
            && self.accel_err_ema.unwrap_or(accel_err) <= self.cfg.max_accel_norm_err_mps2;
        let low_speed = self.speed_ema.unwrap_or(speed_mps) <= self.cfg.max_speed_mps;
        let steady_motion = self
            .speed_rate_ema
            .or(speed_rate_mps2.map(f32::abs))
            .is_none_or(|v| v <= self.cfg.max_speed_rate_mps2)
            && self
                .course_rate_ema
                .or(course_rate_radps.map(f32::abs))
                .is_none_or(|v| v <= self.cfg.max_course_rate_radps);
        let stationary = low_dynamic && (low_speed || steady_motion);

        if stationary {
            self.stationary_accel.push(accel_b);
            self.stationary_gyro.push(gyro_radps);
        } else {
            self.stationary_accel.clear();
            self.stationary_gyro.clear();
        }
        self.stationary_accel.len() >= self.cfg.stationary_samples
    }

    fn snapshot(
        &self,
        speed_mps: f32,
        speed_rate_mps2: Option<f32>,
        course_rate_radps: Option<f32>,
        gyro_radps: [f32; 3],
        accel_b: [f32; 3],
    ) -> BootstrapDebugSample {
        let speed_rate = speed_rate_mps2.unwrap_or(f32::NAN);
        let course_rate = course_rate_radps.unwrap_or(f32::NAN);
        BootstrapDebugSample {
            t_s: 0.0,
            speed_mps: speed_mps as f64,
            speed_ema_mps: self.speed_ema.unwrap_or(speed_mps) as f64,
            speed_rate_mps2: speed_rate as f64,
            speed_rate_ema_mps2: self
                .speed_rate_ema
                .unwrap_or(speed_rate.abs())
                as f64,
            course_rate_dps: course_rate.to_degrees() as f64,
            course_rate_ema_dps: self
                .course_rate_ema
                .unwrap_or(course_rate.abs())
                .to_degrees() as f64,
            gyro_norm_dps: norm3(gyro_radps).to_degrees() as f64,
            gyro_ema_dps: self.gyro_ema.unwrap_or(norm3(gyro_radps)).to_degrees() as f64,
            accel_norm_err_mps2: (norm3(accel_b) - GRAVITY_MPS2).abs() as f64,
            accel_err_ema_mps2: self
                .accel_err_ema
                .unwrap_or((norm3(accel_b) - GRAVITY_MPS2).abs())
                as f64,
            stationary: {
                let low_dynamic =
                    self.gyro_ema.unwrap_or(norm3(gyro_radps)) <= self.cfg.max_gyro_radps
                        && self
                            .accel_err_ema
                            .unwrap_or((norm3(accel_b) - GRAVITY_MPS2).abs())
                            <= self.cfg.max_accel_norm_err_mps2;
                let low_speed = self.speed_ema.unwrap_or(speed_mps) <= self.cfg.max_speed_mps;
                let steady_motion = self
                    .speed_rate_ema
                    .or(speed_rate_mps2.map(f32::abs))
                    .is_none_or(|v| v <= self.cfg.max_speed_rate_mps2)
                    && self
                        .course_rate_ema
                        .or(course_rate_radps.map(f32::abs))
                        .is_none_or(|v| v <= self.cfg.max_course_rate_radps);
                low_dynamic && (low_speed || steady_motion)
            },
            sample_count: self.stationary_accel.len(),
        }
    }
}

fn build_bootstrap_debug_trace(
    dataset: &AlignDataset,
    bootstrap_cfg: &BootstrapConfig,
) -> Vec<BootstrapDebugSample> {
    let imu_packets = match build_imu_packets(&dataset.frames, &dataset.timeline) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let nav_events = collect_nav_events(&dataset.frames, &dataset.timeline);
    let mut bootstrap = BootstrapDetector::new(*bootstrap_cfg);
    let mut out = Vec::with_capacity(imu_packets.len());
    let mut scan_idx = 0usize;
    let mut prev_nav: Option<(f64, NavPvtObs)> = None;
    for (tn, nav) in &nav_events {
        while scan_idx < imu_packets.len() && imu_packets[scan_idx].t_ms <= *tn {
            let pkt = &imu_packets[scan_idx];
            let gyro_radps = [
                pkt.gx_dps.to_radians() as f32,
                pkt.gy_dps.to_radians() as f32,
                pkt.gz_dps.to_radians() as f32,
            ];
            let accel_b = [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32];
            let (speed_mps, speed_rate_mps2, course_rate_radps) =
                bootstrap_motion_hints(prev_nav, (*tn, *nav), pkt.t_ms);
            bootstrap.update(
                accel_b,
                gyro_radps,
                speed_mps,
                speed_rate_mps2,
                course_rate_radps,
            );
            let mut row = bootstrap.snapshot(
                speed_mps,
                speed_rate_mps2,
                course_rate_radps,
                gyro_radps,
                accel_b,
            );
            row.t_s = (pkt.t_ms - dataset.timeline.t0_master_ms) * 1.0e-3;
            out.push(row);
            scan_idx += 1;
        }
        prev_nav = Some((*tn, *nav));
    }
    out
}

fn evaluate_config(
    dataset: &AlignDataset,
    cfg: &AlignConfig,
    bootstrap_cfg: &BootstrapConfig,
) -> Result<EvalResult> {
    let mut samples = Vec::<ResidualSample>::new();
    let replay = build_align_replay(
        &dataset.frames,
        &dataset.timeline,
        *cfg,
        *bootstrap_cfg,
        ImuReplayConfig::default(),
    );
    let init_time_s = replay.samples.first().map(|s| s.t_s).unwrap_or(f64::NAN);
    let final_alg_q = replay.final_alg_q;

    for sample in &replay.samples {
        if let (Some(q_alg), Some(alg_rpy_deg)) = (sample.alg_q, sample.alg_rpy_deg) {
            let q_ref_axis = final_alg_q.unwrap_or(q_alg);
            let err_roll_deg = wrap_deg180(sample.align_rpy_deg[0] - alg_rpy_deg[0]);
            let err_pitch_deg = sample.align_rpy_deg[1] - alg_rpy_deg[1];
            let err_yaw_deg = wrap_deg180(sample.align_rpy_deg[2] - alg_rpy_deg[2]);
            samples.push(ResidualSample {
                t_s: sample.t_s,
                align_roll_deg: sample.align_rpy_deg[0],
                align_pitch_deg: sample.align_rpy_deg[1],
                align_yaw_deg: sample.align_rpy_deg[2],
                alg_roll_deg: alg_rpy_deg[0],
                alg_pitch_deg: alg_rpy_deg[1],
                alg_yaw_deg: alg_rpy_deg[2],
                err_roll_deg,
                err_pitch_deg,
                err_yaw_deg,
                sigma_roll_deg: sample.p_diag[0].sqrt().to_degrees(),
                sigma_pitch_deg: sample.p_diag[1].sqrt().to_degrees(),
                sigma_yaw_deg: sample.p_diag[2].sqrt().to_degrees(),
                course_rate_dps: sample.course_rate_dps,
                a_lat_mps2: sample.a_lat_mps2,
                a_long_mps2: sample.a_long_mps2,
                rot_err_deg: quat_angle_deg(sample.q_align, q_alg),
                fwd_err_deg: axis_angle_deg(
                    quat_rotate(sample.q_align, [1.0, 0.0, 0.0]),
                    quat_rotate(q_ref_axis, [1.0, 0.0, 0.0]),
                ),
                down_err_deg: axis_angle_deg(
                    quat_rotate(sample.q_align, [0.0, 0.0, 1.0]),
                    quat_rotate(q_ref_axis, [0.0, 0.0, 1.0]),
                ),
                fwd_err_signed_deg: signed_projected_axis_angle_deg(
                    quat_rotate(sample.q_align, [1.0, 0.0, 0.0]),
                    quat_rotate(q_ref_axis, [1.0, 0.0, 0.0]),
                    quat_rotate(q_ref_axis, [0.0, 0.0, 1.0]),
                ),
                down_err_signed_deg: signed_projected_axis_angle_deg(
                    quat_rotate(sample.q_align, [0.0, 0.0, 1.0]),
                    quat_rotate(q_ref_axis, [0.0, 0.0, 1.0]),
                    quat_rotate(q_ref_axis, [0.0, 1.0, 0.0]),
                ),
                horiz_angle_err_deg: sample.horiz_trace.angle_err_deg,
                horiz_effective_std_deg: sample.horiz_trace.effective_std_deg,
                horiz_gnss_norm_mps2: sample.horiz_trace.gnss_norm_mps2,
                horiz_imu_norm_mps2: sample.horiz_trace.imu_norm_mps2,
                horiz_speed_q: sample.horiz_trace.speed_q,
                horiz_accel_q: sample.horiz_trace.accel_q,
                horiz_straight_q: sample.horiz_trace.straight_q,
                horiz_turn_q: sample.horiz_trace.turn_q,
                horiz_dominance_q: sample.horiz_trace.dominance_q,
                horiz_turn_core_valid: sample.horiz_trace.turn_core_valid,
                horiz_straight_core_valid: sample.horiz_trace.straight_core_valid,
                horiz_applied: sample.horiz_trace.applied,
                gravity_applied: sample.upd_gravity,
                gravity_quasi_static: sample.upd_gravity_quasi_static,
                planar_gyro_valid: false,
                planar_gyro_residual_x_dps: f64::NAN,
                planar_gyro_residual_y_dps: f64::NAN,
            });
        }
    }

    let metrics = score_samples(&samples, dataset.total_nav_events, init_time_s);
    Ok(EvalResult {
        cfg: *cfg,
        bootstrap_cfg: *bootstrap_cfg,
        metrics,
        samples,
    })
}

fn evaluate_fusion_config(dataset: &AlignDataset) -> EvalResult {
    let cfg = AlignConfig::default();
    let bootstrap_cfg = BootstrapConfig {
        ema_alpha: 0.05,
        max_speed_mps: 0.35,
        stationary_samples: 300,
        max_gyro_radps: cfg.max_stationary_gyro_radps,
        max_accel_norm_err_mps2: cfg.max_stationary_accel_norm_err_mps2,
        max_speed_rate_mps2: 0.15,
        max_course_rate_radps: 1.0_f32.to_radians(),
    };
    let samples = replay_to_residuals(build_fusion_align_replay(
        &dataset.frames,
        &dataset.timeline,
        EkfImuSource::Align,
        ImuReplayConfig::default(),
    ));
    let init_time_s = samples.first().map(|s| s.t_s).unwrap_or(f64::NAN);
    let metrics = score_samples(&samples, dataset.total_nav_events, init_time_s);
    EvalResult {
        cfg,
        bootstrap_cfg,
        metrics,
        samples,
    }
}

fn replay_to_residuals(
    replay: sim::visualizer::pipeline::align_replay::AlignReplayData,
) -> Vec<ResidualSample> {
    let mut samples = Vec::<ResidualSample>::new();
    let final_alg_q = replay.final_alg_q;
    for sample in &replay.samples {
        if let (Some(q_alg), Some(alg_rpy_deg)) = (sample.alg_q, sample.alg_rpy_deg) {
            let q_ref_axis = final_alg_q.unwrap_or(q_alg);
            let err_roll_deg = wrap_deg180(sample.align_rpy_deg[0] - alg_rpy_deg[0]);
            let err_pitch_deg = sample.align_rpy_deg[1] - alg_rpy_deg[1];
            let err_yaw_deg = wrap_deg180(sample.align_rpy_deg[2] - alg_rpy_deg[2]);
            samples.push(ResidualSample {
                t_s: sample.t_s,
                align_roll_deg: sample.align_rpy_deg[0],
                align_pitch_deg: sample.align_rpy_deg[1],
                align_yaw_deg: sample.align_rpy_deg[2],
                alg_roll_deg: alg_rpy_deg[0],
                alg_pitch_deg: alg_rpy_deg[1],
                alg_yaw_deg: alg_rpy_deg[2],
                err_roll_deg,
                err_pitch_deg,
                err_yaw_deg,
                sigma_roll_deg: sample.p_diag[0].sqrt().to_degrees(),
                sigma_pitch_deg: sample.p_diag[1].sqrt().to_degrees(),
                sigma_yaw_deg: sample.p_diag[2].sqrt().to_degrees(),
                course_rate_dps: sample.course_rate_dps,
                a_lat_mps2: sample.a_lat_mps2,
                a_long_mps2: sample.a_long_mps2,
                rot_err_deg: quat_angle_deg(sample.q_align, q_alg),
                fwd_err_deg: axis_angle_deg(
                    quat_rotate(sample.q_align, [1.0, 0.0, 0.0]),
                    quat_rotate(q_ref_axis, [1.0, 0.0, 0.0]),
                ),
                down_err_deg: axis_angle_deg(
                    quat_rotate(sample.q_align, [0.0, 0.0, 1.0]),
                    quat_rotate(q_ref_axis, [0.0, 0.0, 1.0]),
                ),
                fwd_err_signed_deg: signed_projected_axis_angle_deg(
                    quat_rotate(sample.q_align, [1.0, 0.0, 0.0]),
                    quat_rotate(q_ref_axis, [1.0, 0.0, 0.0]),
                    quat_rotate(q_ref_axis, [0.0, 0.0, 1.0]),
                ),
                down_err_signed_deg: signed_projected_axis_angle_deg(
                    quat_rotate(sample.q_align, [0.0, 0.0, 1.0]),
                    quat_rotate(q_ref_axis, [0.0, 0.0, 1.0]),
                    quat_rotate(q_ref_axis, [0.0, 1.0, 0.0]),
                ),
                horiz_angle_err_deg: sample.horiz_trace.angle_err_deg,
                horiz_effective_std_deg: sample.horiz_trace.effective_std_deg,
                horiz_gnss_norm_mps2: sample.horiz_trace.gnss_norm_mps2,
                horiz_imu_norm_mps2: sample.horiz_trace.imu_norm_mps2,
                horiz_speed_q: sample.horiz_trace.speed_q,
                horiz_accel_q: sample.horiz_trace.accel_q,
                horiz_straight_q: sample.horiz_trace.straight_q,
                horiz_turn_q: sample.horiz_trace.turn_q,
                horiz_dominance_q: sample.horiz_trace.dominance_q,
                horiz_turn_core_valid: sample.horiz_trace.turn_core_valid,
                horiz_straight_core_valid: sample.horiz_trace.straight_core_valid,
                horiz_applied: sample.horiz_trace.applied,
                gravity_applied: sample.upd_gravity,
                gravity_quasi_static: sample.upd_gravity_quasi_static,
                planar_gyro_valid: false,
                planar_gyro_residual_x_dps: f64::NAN,
                planar_gyro_residual_y_dps: f64::NAN,
            });
        }
    }
    samples
}

fn evaluate_dispatch(
    mode: FilterMode,
    dataset: &AlignDataset,
    cfg: &AlignConfig,
    bootstrap_cfg: &BootstrapConfig,
    _args: &Args,
) -> Result<EvalResult> {
    match mode {
        FilterMode::Align => evaluate_config(dataset, cfg, bootstrap_cfg),
        FilterMode::Fusion => Ok(evaluate_fusion_config(dataset)),
    }
}

fn build_replay_for_mode(
    mode: FilterMode,
    dataset: &AlignDataset,
    cfg: &AlignConfig,
    bootstrap_cfg: &BootstrapConfig,
) -> AlignReplayData {
    match mode {
        FilterMode::Align => build_align_replay(
            &dataset.frames,
            &dataset.timeline,
            *cfg,
            *bootstrap_cfg,
            ImuReplayConfig::default(),
        ),
        FilterMode::Fusion => build_fusion_align_replay(
            &dataset.frames,
            &dataset.timeline,
            EkfImuSource::Align,
            ImuReplayConfig::default(),
        ),
    }
}

fn build_horiz_accel_quality_samples(
    replay: &AlignReplayData,
    min_norm_mps2: f64,
) -> Vec<HorizAccelQualitySample> {
    let mut out = Vec::new();
    for sample in &replay.samples {
        let Some(q_alg) = sample.alg_q else {
            continue;
        };
        let imu_vehicle = quat_rotate(
            q_alg,
            [
                sample.horiz_accel_b[0],
                sample.horiz_accel_b[1],
                sample.horiz_accel_b[2],
            ],
        );
        let gnss_long = sample.a_long_mps2;
        let gnss_lat = sample.a_lat_mps2;
        let imu_long = imu_vehicle[0];
        let imu_lat = imu_vehicle[1];
        let gnss_norm = (gnss_long * gnss_long + gnss_lat * gnss_lat).sqrt();
        let imu_norm = (imu_long * imu_long + imu_lat * imu_lat).sqrt();
        if !gnss_norm.is_finite()
            || !imu_norm.is_finite()
            || gnss_norm < min_norm_mps2
            || imu_norm < min_norm_mps2
        {
            continue;
        }
        let angle_err_deg = angle_err_2d_deg([imu_long, imu_lat], [gnss_long, gnss_lat]);
        out.push(HorizAccelQualitySample {
            t_s: sample.t_s,
            gnss_long_mps2: gnss_long,
            gnss_lat_mps2: gnss_lat,
            imu_long_mps2: imu_long,
            imu_lat_mps2: imu_lat,
            gnss_norm_mps2: gnss_norm,
            imu_norm_mps2: imu_norm,
            angle_err_deg,
            long_resid_mps2: imu_long - gnss_long,
            lat_resid_mps2: imu_lat - gnss_lat,
            straight_core_valid: sample.horiz_trace.straight_core_valid,
            turn_core_valid: sample.horiz_trace.turn_core_valid,
        });
    }
    out
}

fn summarize_horiz_accel_quality(samples: &[HorizAccelQualitySample]) -> HorizAccelQualityReport {
    if samples.is_empty() {
        return HorizAccelQualityReport::default();
    }
    let n = samples.len() as f64;
    let mut abs_angles: Vec<f64> = samples.iter().map(|s| s.angle_err_deg.abs()).collect();
    abs_angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let p90_idx = ((abs_angles.len() as f64) * 0.90).floor() as usize;
    let p90_abs_angle_err_deg = abs_angles[p90_idx.min(abs_angles.len() - 1)];
    let mean_abs_angle_err_deg = abs_angles.iter().sum::<f64>() / n;
    let within_10_deg_frac = abs_angles.iter().filter(|v| **v <= 10.0).count() as f64 / n;
    let within_20_deg_frac = abs_angles.iter().filter(|v| **v <= 20.0).count() as f64 / n;
    let within_30_deg_frac = abs_angles.iter().filter(|v| **v <= 30.0).count() as f64 / n;
    let long_bias_mps2 = samples.iter().map(|s| s.long_resid_mps2).sum::<f64>() / n;
    let lat_bias_mps2 = samples.iter().map(|s| s.lat_resid_mps2).sum::<f64>() / n;
    let long_rmse_mps2 = (samples
        .iter()
        .map(|s| s.long_resid_mps2 * s.long_resid_mps2)
        .sum::<f64>()
        / n)
        .sqrt();
    let lat_rmse_mps2 = (samples
        .iter()
        .map(|s| s.lat_resid_mps2 * s.lat_resid_mps2)
        .sum::<f64>()
        / n)
        .sqrt();
    let long_zero = pearson_corr(
        &samples.iter().map(|s| s.imu_long_mps2).collect::<Vec<_>>(),
        &samples.iter().map(|s| s.gnss_long_mps2).collect::<Vec<_>>(),
    );
    let lat_zero = pearson_corr(
        &samples.iter().map(|s| s.imu_lat_mps2).collect::<Vec<_>>(),
        &samples.iter().map(|s| s.gnss_lat_mps2).collect::<Vec<_>>(),
    );
    let (long_best_lag_windows, long_best_corr) = best_lag_corr(
        &samples.iter().map(|s| s.imu_long_mps2).collect::<Vec<_>>(),
        &samples.iter().map(|s| s.gnss_long_mps2).collect::<Vec<_>>(),
        10,
    );
    let (lat_best_lag_windows, lat_best_corr) = best_lag_corr(
        &samples.iter().map(|s| s.imu_lat_mps2).collect::<Vec<_>>(),
        &samples.iter().map(|s| s.gnss_lat_mps2).collect::<Vec<_>>(),
        10,
    );
    let mut dts = Vec::new();
    for w in samples.windows(2) {
        let dt = w[1].t_s - w[0].t_s;
        if dt.is_finite() && dt > 0.0 {
            dts.push(dt);
        }
    }
    dts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median_dt_s = if dts.is_empty() {
        f64::NAN
    } else {
        dts[dts.len() / 2]
    };
    HorizAccelQualityReport {
        n_total: samples.len(),
        n_valid: samples.len(),
        mean_abs_angle_err_deg,
        p90_abs_angle_err_deg,
        within_10_deg_frac,
        within_20_deg_frac,
        within_30_deg_frac,
        long_bias_mps2,
        lat_bias_mps2,
        long_rmse_mps2,
        lat_rmse_mps2,
        long_corr_zero_lag: long_zero,
        lat_corr_zero_lag: lat_zero,
        long_best_corr,
        lat_best_corr,
        long_best_lag_windows,
        lat_best_lag_windows,
        median_dt_s,
    }
}

fn print_horiz_accel_quality_report(label: &str, report: &HorizAccelQualityReport) {
    eprintln!(
        "[{}] n_total={} n_valid={} median_dt={:.3}s",
        label, report.n_total, report.n_valid, report.median_dt_s
    );
    eprintln!(
        "[{}] angle_abs_deg mean={:.3} p90={:.3} within10={:.3} within20={:.3} within30={:.3}",
        label,
        report.mean_abs_angle_err_deg,
        report.p90_abs_angle_err_deg,
        report.within_10_deg_frac,
        report.within_20_deg_frac,
        report.within_30_deg_frac
    );
    eprintln!(
        "[{}] residual_mps2 long_bias={:.3} long_rmse={:.3} lat_bias={:.3} lat_rmse={:.3}",
        label,
        report.long_bias_mps2,
        report.long_rmse_mps2,
        report.lat_bias_mps2,
        report.lat_rmse_mps2
    );
    eprintln!(
        "[{}] corr_zero long={:.3} lat={:.3} | corr_best long={:.3}@{} lat={:.3}@{}",
        label,
        report.long_corr_zero_lag,
        report.lat_corr_zero_lag,
        report.long_best_corr,
        report.long_best_lag_windows,
        report.lat_best_corr,
        report.lat_best_lag_windows
    );
}

fn write_horiz_accel_csv(path: &PathBuf, samples: &[HorizAccelQualitySample]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "t_s,gnss_long_mps2,gnss_lat_mps2,imu_long_mps2,imu_lat_mps2,gnss_norm_mps2,imu_norm_mps2,angle_err_deg,long_resid_mps2,lat_resid_mps2,straight_core_valid,turn_core_valid"
    )?;
    for s in samples {
        writeln!(
            w,
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
            s.t_s,
            s.gnss_long_mps2,
            s.gnss_lat_mps2,
            s.imu_long_mps2,
            s.imu_lat_mps2,
            s.gnss_norm_mps2,
            s.imu_norm_mps2,
            s.angle_err_deg,
            s.long_resid_mps2,
            s.lat_resid_mps2,
            s.straight_core_valid as u8,
            s.turn_core_valid as u8
        )?;
    }
    Ok(())
}

fn angle_err_2d_deg(a: [f64; 2], b: [f64; 2]) -> f64 {
    let na = (a[0] * a[0] + a[1] * a[1]).sqrt();
    let nb = (b[0] * b[0] + b[1] * b[1]).sqrt();
    if na <= 1.0e-9 || nb <= 1.0e-9 {
        return f64::NAN;
    }
    let dot = ((a[0] * b[0] + a[1] * b[1]) / (na * nb)).clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}

fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 3 {
        return f64::NAN;
    }
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (xa, xb) in a.iter().zip(b.iter()) {
        let da = *xa - mean_a;
        let db = *xb - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a <= 1.0e-12 || var_b <= 1.0e-12 {
        return f64::NAN;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

fn best_lag_corr(a: &[f64], b: &[f64], max_lag: isize) -> (isize, f64) {
    let mut best = (0isize, f64::NEG_INFINITY);
    for lag in -max_lag..=max_lag {
        let (a_slice, b_slice) = if lag >= 0 {
            let lag = lag as usize;
            if lag >= a.len() || lag >= b.len() {
                continue;
            }
            (&a[lag..], &b[..b.len() - lag])
        } else {
            let lag = (-lag) as usize;
            if lag >= a.len() || lag >= b.len() {
                continue;
            }
            (&a[..a.len() - lag], &b[lag..])
        };
        let corr = pearson_corr(a_slice, b_slice);
        if corr.is_finite() && corr > best.1 {
            best = (lag, corr);
        }
    }
    if best.1 == f64::NEG_INFINITY {
        (0, f64::NAN)
    } else {
        best
    }
}

fn tune_config(
    dataset: &AlignDataset,
    base_cfg: &AlignConfig,
    base_bootstrap_cfg: &BootstrapConfig,
    tune_passes: usize,
) -> Result<EvalResult> {
    let mut best = evaluate_config(dataset, base_cfg, base_bootstrap_cfg)?;
    let stages = [
        TuneParam::BootstrapEmaAlpha,
        TuneParam::BootstrapMaxSpeedMps,
        TuneParam::RGravityStdMps2,
        TuneParam::GravityLpfAlpha,
        TuneParam::MaxStationaryGyroDps,
        TuneParam::MaxStationaryAccelNormErrMps2,
        TuneParam::RTurnGyroStdDps,
        TuneParam::MinSpeedMps,
        TuneParam::MinTurnRateDps,
        TuneParam::MinLatAccMps2,
        TuneParam::MinLongAccMps2,
    ];

    for pass in 0..tune_passes {
        let mut improved = false;
        for param in stages {
            let mut local_best = best.clone();
            for &candidate in param.candidates(&best.cfg) {
                let mut cfg = best.cfg;
                let mut bootstrap_cfg = best.bootstrap_cfg;
                let current = match param {
                    TuneParam::BootstrapEmaAlpha => bootstrap_cfg.ema_alpha,
                    TuneParam::BootstrapMaxSpeedMps => bootstrap_cfg.max_speed_mps,
                    _ => param.current(&best.cfg),
                };
                if (candidate - current).abs() <= 1.0e-6 {
                    continue;
                }
                match param {
                    TuneParam::BootstrapEmaAlpha => bootstrap_cfg.ema_alpha = candidate,
                    TuneParam::BootstrapMaxSpeedMps => bootstrap_cfg.max_speed_mps = candidate,
                    TuneParam::MaxStationaryGyroDps => {
                        param.set(&mut cfg, candidate);
                        bootstrap_cfg.max_gyro_radps = cfg.max_stationary_gyro_radps;
                    }
                    TuneParam::MaxStationaryAccelNormErrMps2 => {
                        param.set(&mut cfg, candidate);
                        bootstrap_cfg.max_accel_norm_err_mps2 =
                            cfg.max_stationary_accel_norm_err_mps2;
                    }
                    _ => param.set(&mut cfg, candidate),
                }
                let eval = evaluate_config(dataset, &cfg, &bootstrap_cfg)?;
                if eval.metrics.score + 1.0e-6 < local_best.metrics.score {
                    local_best = eval;
                }
            }
            if local_best.metrics.score + 1.0e-6 < best.metrics.score {
                improved = true;
                eprintln!(
                    "[tune pass {}] {} -> {} (score {:.3} -> {:.3})",
                    pass + 1,
                    param.name(),
                    match param {
                        TuneParam::BootstrapEmaAlpha => local_best.bootstrap_cfg.ema_alpha,
                        TuneParam::BootstrapMaxSpeedMps => local_best.bootstrap_cfg.max_speed_mps,
                        _ => param.current(&local_best.cfg),
                    },
                    best.metrics.score,
                    local_best.metrics.score
                );
                best = local_best;
            }
        }
        if !improved {
            break;
        }
    }

    // A small discrete ablation on the weaker cues. Longitudinal accel is useful when clean,
    // but on real logs it can be grade/timing sensitive, so test it explicitly.
    let feature_sets = [best.cfg.use_turn_gyro, true, false];
    for use_turn_gyro in feature_sets {
        let mut cfg = best.cfg;
        let bootstrap_cfg = best.bootstrap_cfg;
        cfg.use_turn_gyro = use_turn_gyro;
        let eval = evaluate_config(dataset, &cfg, &bootstrap_cfg)?;
        if eval.metrics.score + 1.0e-6 < best.metrics.score {
            eprintln!(
                "[tune features] turn={} score {:.3} -> {:.3}",
                use_turn_gyro, best.metrics.score, eval.metrics.score
            );
            best = eval;
        }
    }

    Ok(best)
}

fn score_samples(samples: &[ResidualSample], total_nav: usize, init_time_s: f64) -> EvalMetrics {
    if samples.is_empty() {
        return EvalMetrics {
            n_samples: 0,
            init_time_s,
            horizon_s: 0.0,
            coverage: 0.0,
            score: f64::INFINITY,
            ..EvalMetrics::default()
        };
    }

    let mut sum_sq_roll = 0.0;
    let mut sum_sq_pitch = 0.0;
    let mut sum_sq_yaw = 0.0;
    let mut sum_abs_roll = 0.0;
    let mut sum_abs_pitch = 0.0;
    let mut sum_abs_yaw = 0.0;
    let mut sum_rot = 0.0;
    let mut max_rot = 0.0_f64;
    let mut sum_fwd = 0.0;
    let mut max_fwd = 0.0_f64;
    let mut sum_down = 0.0;
    let mut max_down = 0.0_f64;
    for s in samples {
        sum_sq_roll += s.err_roll_deg * s.err_roll_deg;
        sum_sq_pitch += s.err_pitch_deg * s.err_pitch_deg;
        sum_sq_yaw += s.err_yaw_deg * s.err_yaw_deg;
        sum_abs_roll += s.err_roll_deg.abs();
        sum_abs_pitch += s.err_pitch_deg.abs();
        sum_abs_yaw += s.err_yaw_deg.abs();
        sum_rot += s.rot_err_deg;
        max_rot = max_rot.max(s.rot_err_deg);
        sum_fwd += s.fwd_err_deg;
        max_fwd = max_fwd.max(s.fwd_err_deg);
        sum_down += s.down_err_deg;
        max_down = max_down.max(s.down_err_deg);
    }
    let n = samples.len() as f64;
    let rmse_roll_deg = (sum_sq_roll / n).sqrt();
    let rmse_pitch_deg = (sum_sq_pitch / n).sqrt();
    let rmse_yaw_deg = (sum_sq_yaw / n).sqrt();
    let mae_roll_deg = sum_abs_roll / n;
    let mae_pitch_deg = sum_abs_pitch / n;
    let mae_yaw_deg = sum_abs_yaw / n;
    let mean_rot_err_deg = sum_rot / n;
    let mean_fwd_err_deg = sum_fwd / n;
    let mean_down_err_deg = sum_down / n;
    let final_sample = samples[samples.len() - 1];
    let coverage = (samples.len() as f64 / total_nav.max(1) as f64).clamp(0.0, 1.0);
    let horizon_s = samples[samples.len() - 1].t_s - samples[0].t_s;
    let total_rmse = rmse_roll_deg + rmse_pitch_deg + rmse_yaw_deg;
    let total_mae = mae_roll_deg + mae_pitch_deg + mae_yaw_deg;
    let final_abs = final_sample.err_roll_deg.abs()
        + final_sample.err_pitch_deg.abs()
        + final_sample.err_yaw_deg.abs();
    let coverage_penalty = 1.0 / coverage.max(0.15);
    let init_penalty = if init_time_s.is_finite() {
        0.02 * init_time_s
    } else {
        25.0
    };
    let score =
        (total_rmse + 0.25 * total_mae + 0.05 * final_abs) * coverage_penalty + init_penalty;

    EvalMetrics {
        n_samples: samples.len(),
        init_time_s,
        horizon_s,
        coverage,
        rmse_roll_deg,
        rmse_pitch_deg,
        rmse_yaw_deg,
        mae_roll_deg,
        mae_pitch_deg,
        mae_yaw_deg,
        mean_rot_err_deg,
        max_rot_err_deg: max_rot,
        final_rot_err_deg: final_sample.rot_err_deg,
        mean_fwd_err_deg,
        max_fwd_err_deg: max_fwd,
        final_fwd_err_deg: final_sample.fwd_err_deg,
        mean_down_err_deg,
        max_down_err_deg: max_down,
        final_down_err_deg: final_sample.down_err_deg,
        final_err_roll_deg: final_sample.err_roll_deg,
        final_err_pitch_deg: final_sample.err_pitch_deg,
        final_err_yaw_deg: final_sample.err_yaw_deg,
        score,
    }
}

fn print_metrics(label: &str, metrics: &EvalMetrics) {
    eprintln!(
        "[{}] n={} init={:.2}s horizon={:.2}s coverage={:.3} score={:.3}",
        label,
        metrics.n_samples,
        metrics.init_time_s,
        metrics.horizon_s,
        metrics.coverage,
        metrics.score
    );
    eprintln!(
        "[{}] rmse_deg roll={:.3} pitch={:.3} yaw={:.3}",
        label, metrics.rmse_roll_deg, metrics.rmse_pitch_deg, metrics.rmse_yaw_deg
    );
    eprintln!(
        "[{}] mae_deg  roll={:.3} pitch={:.3} yaw={:.3}",
        label, metrics.mae_roll_deg, metrics.mae_pitch_deg, metrics.mae_yaw_deg
    );
    eprintln!(
        "[{}] final_err_deg roll={:.3} pitch={:.3} yaw={:.3}",
        label, metrics.final_err_roll_deg, metrics.final_err_pitch_deg, metrics.final_err_yaw_deg
    );
    eprintln!(
        "[{}] rot_err_deg mean={:.3} max={:.3} final={:.3}",
        label, metrics.mean_rot_err_deg, metrics.max_rot_err_deg, metrics.final_rot_err_deg
    );
    eprintln!(
        "[{}] axis_err_deg fwd mean={:.3} max={:.3} final={:.3} | down mean={:.3} max={:.3} final={:.3}",
        label,
        metrics.mean_fwd_err_deg,
        metrics.max_fwd_err_deg,
        metrics.final_fwd_err_deg,
        metrics.mean_down_err_deg,
        metrics.max_down_err_deg,
        metrics.final_down_err_deg
    );
}

fn print_config(cfg: &AlignConfig, bootstrap_cfg: &BootstrapConfig) {
    eprintln!(
        "[config] q_std_deg=[{:.4}, {:.4}, {:.4}] r_gravity={:.3} r_horiz_heading_deg={:.3} r_turn_dps={:.3} turn_gyro_yaw_scale={:.3} r_turn_heading_deg={:.3}",
        cfg.q_mount_std_rad[0].to_degrees(),
        cfg.q_mount_std_rad[1].to_degrees(),
        cfg.q_mount_std_rad[2].to_degrees(),
        cfg.r_gravity_std_mps2,
        cfg.r_horiz_heading_std_rad.to_degrees(),
        cfg.r_turn_gyro_std_radps.to_degrees(),
        cfg.turn_gyro_yaw_scale,
        cfg.r_turn_heading_std_rad.to_degrees()
    );
    eprintln!(
        "[config] gravity_alpha={:.3} min_speed={:.3} min_turn_dps={:.3} min_lat={:.3} min_long={:.3} max_stat_gyro_dps={:.3} max_stat_acc_err={:.3}",
        cfg.gravity_lpf_alpha,
        cfg.min_speed_mps,
        cfg.min_turn_rate_radps.to_degrees(),
        cfg.min_lat_acc_mps2,
        cfg.min_long_acc_mps2,
        cfg.max_stationary_gyro_radps.to_degrees(),
        cfg.max_stationary_accel_norm_err_mps2
    );
    eprintln!(
        "[config] bootstrap_ema_alpha={:.3} bootstrap_max_speed={:.3} stationary_samples={}",
        bootstrap_cfg.ema_alpha, bootstrap_cfg.max_speed_mps, bootstrap_cfg.stationary_samples
    );
    eprintln!(
        "[config] use_gravity={} use_turn_gyro={}",
        cfg.use_gravity, cfg.use_turn_gyro
    );
}

fn write_residual_csv(path: &PathBuf, samples: &[ResidualSample]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "t_s,align_roll_deg,align_pitch_deg,align_yaw_deg,alg_roll_deg,alg_pitch_deg,alg_yaw_deg,err_roll_deg,err_pitch_deg,err_yaw_deg,sigma_roll_deg,sigma_pitch_deg,sigma_yaw_deg,course_rate_dps,a_lat_mps2,a_long_mps2,rot_err_deg,fwd_err_deg,down_err_deg,fwd_err_signed_deg,down_err_signed_deg,horiz_angle_err_deg,horiz_effective_std_deg,horiz_gnss_norm_mps2,horiz_imu_norm_mps2,horiz_speed_q,horiz_accel_q,horiz_straight_q,horiz_turn_q,horiz_dominance_q,horiz_turn_core_valid,horiz_straight_core_valid,horiz_applied,gravity_applied,gravity_quasi_static,planar_gyro_valid,planar_gyro_residual_x_dps,planar_gyro_residual_y_dps"
    )?;
    for s in samples {
        writeln!(
            w,
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{:.6},{:.6}",
            s.t_s,
            s.align_roll_deg,
            s.align_pitch_deg,
            s.align_yaw_deg,
            s.alg_roll_deg,
            s.alg_pitch_deg,
            s.alg_yaw_deg,
            s.err_roll_deg,
            s.err_pitch_deg,
            s.err_yaw_deg,
            s.sigma_roll_deg,
            s.sigma_pitch_deg,
            s.sigma_yaw_deg,
            s.course_rate_dps,
            s.a_lat_mps2,
            s.a_long_mps2,
            s.rot_err_deg,
            s.fwd_err_deg,
            s.down_err_deg,
            s.fwd_err_signed_deg,
            s.down_err_signed_deg,
            s.horiz_angle_err_deg,
            s.horiz_effective_std_deg,
            s.horiz_gnss_norm_mps2,
            s.horiz_imu_norm_mps2,
            s.horiz_speed_q,
            s.horiz_accel_q,
            s.horiz_straight_q,
            s.horiz_turn_q,
            s.horiz_dominance_q,
            s.horiz_turn_core_valid as u8,
            s.horiz_straight_core_valid as u8,
            s.horiz_applied as u8,
            s.gravity_applied as u8,
            s.gravity_quasi_static as u8,
            s.planar_gyro_valid as u8,
            s.planar_gyro_residual_x_dps,
            s.planar_gyro_residual_y_dps,
        )?;
    }
    Ok(())
}

fn write_bootstrap_debug_csv(path: &PathBuf, samples: &[BootstrapDebugSample]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "t_s,speed_mps,speed_ema_mps,speed_rate_mps2,speed_rate_ema_mps2,course_rate_dps,course_rate_ema_dps,gyro_norm_dps,gyro_ema_dps,accel_norm_err_mps2,accel_err_ema_mps2,stationary,sample_count"
    )?;
    for s in samples {
        writeln!(
            w,
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
            s.t_s,
            s.speed_mps,
            s.speed_ema_mps,
            s.speed_rate_mps2,
            s.speed_rate_ema_mps2,
            s.course_rate_dps,
            s.course_rate_ema_dps,
            s.gyro_norm_dps,
            s.gyro_ema_dps,
            s.accel_norm_err_mps2,
            s.accel_err_ema_mps2,
            if s.stationary { 1 } else { 0 },
            s.sample_count
        )?;
    }
    Ok(())
}

fn ema_update(prev: Option<f32>, sample: f32, alpha: f32) -> f32 {
    let alpha = alpha.clamp(1.0e-4, 1.0);
    match prev {
        Some(prev) => (1.0 - alpha) * prev + alpha * sample,
        None => sample,
    }
}

fn speed_for_bootstrap(
    prev_nav: Option<(f64, NavPvtObs)>,
    curr_nav: (f64, NavPvtObs),
    t_ms: f64,
) -> f64 {
    let speed_curr = horizontal_speed(curr_nav.1);
    let Some((t_prev, nav_prev)) = prev_nav else {
        return speed_curr;
    };
    let speed_prev = horizontal_speed(nav_prev);
    let dt = curr_nav.0 - t_prev;
    if dt <= 1.0e-6 {
        return speed_curr;
    }
    let alpha = ((t_ms - t_prev) / dt).clamp(0.0, 1.0);
    speed_prev + alpha * (speed_curr - speed_prev)
}

fn bootstrap_motion_hints(
    prev_nav: Option<(f64, NavPvtObs)>,
    curr_nav: (f64, NavPvtObs),
    t_ms: f64,
) -> (f32, Option<f32>, Option<f32>) {
    let speed_mps = speed_for_bootstrap(prev_nav, curr_nav, t_ms) as f32;
    let Some((t_prev_ms, nav_prev)) = prev_nav else {
        return (speed_mps, None, None);
    };
    let dt_s = ((curr_nav.0 - t_prev_ms) * 1.0e-3) as f32;
    if dt_s <= 1.0e-6 {
        return (speed_mps, None, None);
    }
    let speed_prev = horizontal_speed(nav_prev) as f32;
    let speed_curr = horizontal_speed(curr_nav.1) as f32;
    let speed_rate_mps2 = (speed_curr - speed_prev) / dt_s;
    let course_prev = nav_prev.vel_e_mps.atan2(nav_prev.vel_n_mps) as f32;
    let course_curr = curr_nav.1.vel_e_mps.atan2(curr_nav.1.vel_n_mps) as f32;
    let course_rate_radps = wrap_rad_pi((course_curr - course_prev) as f64) as f32 / dt_s;
    (speed_mps, Some(speed_rate_mps2), Some(course_rate_radps))
}

fn horizontal_speed(nav: NavPvtObs) -> f64 {
    (nav.vel_n_mps * nav.vel_n_mps + nav.vel_e_mps * nav.vel_e_mps).sqrt()
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn wrap_rad_pi(mut rad: f64) -> f64 {
    while rad <= -std::f64::consts::PI {
        rad += 2.0 * std::f64::consts::PI;
    }
    while rad > std::f64::consts::PI {
        rad -= 2.0 * std::f64::consts::PI;
    }
    rad
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

fn norm_quat(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n <= 1.0e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

fn quat_mul_local(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_conj_local(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn quat_angle_deg(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dq = norm_quat(quat_mul_local(quat_conj_local(a), b));
    let w = dq[0].abs().clamp(-1.0, 1.0);
    2.0 * w.acos().to_degrees()
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
