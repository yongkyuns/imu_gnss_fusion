use std::cmp::Ordering;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use align_rs::align::{Align, AlignConfig, AlignWindowSummary, GRAVITY_MPS2};
use anyhow::{Context, Result, bail};
use clap::Parser;
use sim::ubxlog::{
    NavPvtObs, UbxFrame, extract_esf_alg, extract_esf_alg_valid, extract_esf_raw_samples,
    extract_nav2_pvt_obs, fit_linear_map, parse_ubx_frames, sensor_meta, unwrap_counter,
};
use sim::visualizer::math::{nearest_master_ms, normalize_heading_deg, unwrap_i64_counter};

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

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    alg_valid_only: bool,

    #[arg(long, default_value_t = 100)]
    stationary_samples: usize,

    #[arg(long, default_value_t = 0.05)]
    bootstrap_ema_alpha: f32,

    #[arg(long, default_value_t = 0.35)]
    bootstrap_max_speed_mps: f32,

    #[arg(long, default_value_t = false)]
    tune: bool,

    #[arg(long, default_value_t = 2)]
    tune_passes: usize,

    #[arg(long, default_value_t = 0.01)]
    q_roll_std_deg: f32,
    #[arg(long, default_value_t = 0.01)]
    q_pitch_std_deg: f32,
    #[arg(long, default_value_t = 0.02)]
    q_yaw_std_deg: f32,
    #[arg(long, default_value_t = 0.18)]
    r_gravity_std_mps2: f32,
    #[arg(long, default_value_t = 0.2)]
    r_turn_gyro_std_dps: f32,
    #[arg(long, default_value_t = 0.35)]
    r_course_rate_std_dps: f32,
    #[arg(long, default_value_t = 0.10)]
    r_lat_std_mps2: f32,
    #[arg(long, default_value_t = 0.10)]
    r_long_std_mps2: f32,
    #[arg(long, default_value_t = 0.08)]
    gravity_lpf_alpha: f32,
    #[arg(long, default_value_t = 4.0)]
    min_speed_mps: f32,
    #[arg(long, default_value_t = 3.0)]
    min_turn_rate_dps: f32,
    #[arg(long, default_value_t = 0.35)]
    min_lat_acc_mps2: f32,
    #[arg(long, default_value_t = 0.25)]
    min_long_acc_mps2: f32,
    #[arg(long, default_value_t = 0.8)]
    max_stationary_gyro_dps: f32,
    #[arg(long, default_value_t = 0.2)]
    max_stationary_accel_norm_err_mps2: f32,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    use_gravity: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    use_turn_gyro: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    use_course_rate: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    use_lateral_accel: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    use_longitudinal_accel: bool,
}

#[derive(Clone, Copy, Debug)]
struct AlgEvent {
    t_ms: f64,
    roll_deg: f64,
    pitch_deg: f64,
    yaw_deg: f64,
}

#[derive(Clone, Copy, Debug)]
struct ImuPacket {
    t_ms: f64,
    gx_dps: f64,
    gy_dps: f64,
    gz_dps: f64,
    ax_mps2: f64,
    ay_mps2: f64,
    az_mps2: f64,
}

#[derive(Clone, Debug)]
struct AlignDataset {
    t0_master_ms: f64,
    nav_events: Vec<(f64, NavPvtObs)>,
    alg_events: Vec<AlgEvent>,
    imu_packets: Vec<ImuPacket>,
}

#[derive(Clone, Copy, Debug)]
struct BootstrapConfig {
    ema_alpha: f32,
    max_speed_mps: f32,
    stationary_samples: usize,
    max_gyro_radps: f32,
    max_accel_norm_err_mps2: f32,
}

#[derive(Clone, Debug)]
struct BootstrapDetector {
    cfg: BootstrapConfig,
    gyro_ema: Option<f32>,
    accel_err_ema: Option<f32>,
    speed_ema: Option<f32>,
    stationary_accel: Vec<[f32; 3]>,
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
}

#[derive(Clone, Copy, Debug)]
struct BootstrapDebugSample {
    t_s: f64,
    speed_mps: f64,
    speed_ema_mps: f64,
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
    final_err_roll_deg: f64,
    final_err_pitch_deg: f64,
    final_err_yaw_deg: f64,
    score: f64,
}

#[derive(Clone, Debug)]
struct EvalResult {
    cfg: AlignConfig,
    bootstrap_cfg: BootstrapConfig,
    metrics: EvalMetrics,
    samples: Vec<ResidualSample>,
}

#[derive(Clone, Copy)]
enum TuneParam {
    BootstrapEmaAlpha,
    BootstrapMaxSpeedMps,
    RGravityStdMps2,
    RTurnGyroStdDps,
    RCourseRateStdDps,
    RLatStdMps2,
    RLongStdMps2,
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
            Self::RCourseRateStdDps => "r_course_rate_std_dps",
            Self::RLatStdMps2 => "r_lat_std_mps2",
            Self::RLongStdMps2 => "r_long_std_mps2",
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
            Self::RCourseRateStdDps => &[0.15, 0.25, 0.35, 0.50, 0.80],
            Self::RLatStdMps2 => &[0.05, 0.08, 0.10, 0.15, 0.20],
            Self::RLongStdMps2 => &[0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
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
            Self::RCourseRateStdDps => cfg.r_course_rate_std_radps.to_degrees(),
            Self::RLatStdMps2 => cfg.r_lat_std_mps2,
            Self::RLongStdMps2 => cfg.r_long_std_mps2,
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
            Self::RCourseRateStdDps => cfg.r_course_rate_std_radps = value.to_radians(),
            Self::RLatStdMps2 => cfg.r_lat_std_mps2 = value,
            Self::RLongStdMps2 => cfg.r_long_std_mps2 = value,
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
    let cfg = config_from_args(&args);
    let bootstrap_cfg = bootstrap_config_from_args(&args, &cfg);
    let dataset = load_dataset(&args.logfile, args.max_records, args.alg_valid_only)?;
    if let Some(path) = &args.bootstrap_debug_csv {
        let samples = build_bootstrap_debug_trace(&dataset, &bootstrap_cfg);
        write_bootstrap_debug_csv(path, &samples)?;
        eprintln!("wrote bootstrap debug CSV: {}", path.display());
    }

    let base = evaluate_config(&dataset, &cfg, &bootstrap_cfg)?;
    print_metrics("baseline", &base.metrics);

    let mut best = base;
    if args.tune {
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

fn config_from_args(args: &Args) -> AlignConfig {
    AlignConfig {
        q_mount_std_rad: [
            args.q_roll_std_deg.to_radians(),
            args.q_pitch_std_deg.to_radians(),
            args.q_yaw_std_deg.to_radians(),
        ],
        r_gravity_std_mps2: args.r_gravity_std_mps2,
        r_turn_gyro_std_radps: args.r_turn_gyro_std_dps.to_radians(),
        r_course_rate_std_radps: args.r_course_rate_std_dps.to_radians(),
        r_lat_std_mps2: args.r_lat_std_mps2,
        r_long_std_mps2: args.r_long_std_mps2,
        gravity_lpf_alpha: args.gravity_lpf_alpha,
        min_speed_mps: args.min_speed_mps,
        min_turn_rate_radps: args.min_turn_rate_dps.to_radians(),
        min_lat_acc_mps2: args.min_lat_acc_mps2,
        min_long_acc_mps2: args.min_long_acc_mps2,
        max_stationary_gyro_radps: args.max_stationary_gyro_dps.to_radians(),
        max_stationary_accel_norm_err_mps2: args.max_stationary_accel_norm_err_mps2,
        use_gravity: args.use_gravity,
        use_turn_gyro: args.use_turn_gyro,
        use_course_rate: args.use_course_rate,
        use_lateral_accel: args.use_lateral_accel,
        use_longitudinal_accel: args.use_longitudinal_accel,
    }
}

fn bootstrap_config_from_args(args: &Args, cfg: &AlignConfig) -> BootstrapConfig {
    BootstrapConfig {
        ema_alpha: args.bootstrap_ema_alpha,
        max_speed_mps: args.bootstrap_max_speed_mps,
        stationary_samples: args.stationary_samples,
        max_gyro_radps: cfg.max_stationary_gyro_radps,
        max_accel_norm_err_mps2: cfg.max_stationary_accel_norm_err_mps2,
    }
}

fn load_dataset(
    logfile: &PathBuf,
    max_records: Option<usize>,
    alg_valid_only: bool,
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

    let mut alg_events = Vec::new();
    let mut nav_events = Vec::new();
    for frame in &frames {
        if let Some(t_ms) = nearest_master_ms(frame.seq, &timeline.masters) {
            let alg = if alg_valid_only {
                extract_esf_alg_valid(frame)
            } else {
                extract_esf_alg(frame)
            };
            if let Some((_, roll, pitch, yaw)) = alg {
                let (roll_frd, pitch_frd, yaw_frd) = esf_alg_flu_to_frd_mount_deg(roll, pitch, yaw);
                alg_events.push(AlgEvent {
                    t_ms,
                    roll_deg: roll_frd,
                    pitch_deg: pitch_frd,
                    yaw_deg: normalize_heading_deg(yaw_frd),
                });
            }
            if let Some(obs) = extract_nav2_pvt_obs(frame) {
                if obs.fix_ok && !obs.invalid_llh {
                    nav_events.push((t_ms, obs));
                }
            }
        }
    }
    alg_events.sort_by(|a, b| a.t_ms.partial_cmp(&b.t_ms).unwrap_or(Ordering::Equal));
    nav_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    if alg_events.is_empty() {
        bail!("no ESF-ALG samples found for benchmarking");
    }
    if nav_events.len() < 2 {
        bail!("need at least two NAV2-PVT observations");
    }

    let imu_packets = build_imu_packets(&frames, &timeline)?;
    if imu_packets.is_empty() {
        bail!("no complete ESF-RAW IMU packets found");
    }

    Ok(AlignDataset {
        t0_master_ms: timeline.t0_master_ms,
        nav_events,
        alg_events,
        imu_packets,
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

impl BootstrapDetector {
    fn new(cfg: BootstrapConfig) -> Self {
        Self {
            cfg,
            gyro_ema: None,
            accel_err_ema: None,
            speed_ema: None,
            stationary_accel: Vec::new(),
        }
    }

    fn update(&mut self, accel_b: [f32; 3], gyro_radps: [f32; 3], speed_mps: f32) -> bool {
        let gyro_norm = norm3(gyro_radps);
        let accel_err = (norm3(accel_b) - GRAVITY_MPS2).abs();
        self.gyro_ema = Some(ema_update(self.gyro_ema, gyro_norm, self.cfg.ema_alpha));
        self.accel_err_ema = Some(ema_update(
            self.accel_err_ema,
            accel_err,
            self.cfg.ema_alpha,
        ));
        self.speed_ema = Some(ema_update(self.speed_ema, speed_mps, self.cfg.ema_alpha));

        let stationary = self.speed_ema.unwrap_or(speed_mps) <= self.cfg.max_speed_mps
            && self.gyro_ema.unwrap_or(gyro_norm) <= self.cfg.max_gyro_radps
            && self.accel_err_ema.unwrap_or(accel_err) <= self.cfg.max_accel_norm_err_mps2;

        if stationary {
            self.stationary_accel.push(accel_b);
        } else {
            self.stationary_accel.clear();
        }
        self.stationary_accel.len() >= self.cfg.stationary_samples
    }

    fn snapshot(
        &self,
        speed_mps: f32,
        gyro_radps: [f32; 3],
        accel_b: [f32; 3],
    ) -> BootstrapDebugSample {
        BootstrapDebugSample {
            t_s: 0.0,
            speed_mps: speed_mps as f64,
            speed_ema_mps: self.speed_ema.unwrap_or(speed_mps) as f64,
            gyro_norm_dps: norm3(gyro_radps).to_degrees() as f64,
            gyro_ema_dps: self.gyro_ema.unwrap_or(norm3(gyro_radps)).to_degrees() as f64,
            accel_norm_err_mps2: (norm3(accel_b) - GRAVITY_MPS2).abs() as f64,
            accel_err_ema_mps2: self
                .accel_err_ema
                .unwrap_or((norm3(accel_b) - GRAVITY_MPS2).abs())
                as f64,
            stationary: self.speed_ema.unwrap_or(speed_mps) <= self.cfg.max_speed_mps
                && self.gyro_ema.unwrap_or(norm3(gyro_radps)) <= self.cfg.max_gyro_radps
                && self
                    .accel_err_ema
                    .unwrap_or((norm3(accel_b) - GRAVITY_MPS2).abs())
                    <= self.cfg.max_accel_norm_err_mps2,
            sample_count: self.stationary_accel.len(),
        }
    }
}

fn build_bootstrap_debug_trace(
    dataset: &AlignDataset,
    bootstrap_cfg: &BootstrapConfig,
) -> Vec<BootstrapDebugSample> {
    let mut bootstrap = BootstrapDetector::new(*bootstrap_cfg);
    let mut out = Vec::with_capacity(dataset.imu_packets.len());
    let mut scan_idx = 0usize;
    let mut prev_nav: Option<(f64, NavPvtObs)> = None;
    for (tn, nav) in &dataset.nav_events {
        while scan_idx < dataset.imu_packets.len() && dataset.imu_packets[scan_idx].t_ms <= *tn {
            let pkt = &dataset.imu_packets[scan_idx];
            let gyro_radps = [
                pkt.gx_dps.to_radians() as f32,
                pkt.gy_dps.to_radians() as f32,
                pkt.gz_dps.to_radians() as f32,
            ];
            let accel_b = [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32];
            let speed_mps = speed_for_bootstrap(prev_nav, (*tn, *nav), pkt.t_ms) as f32;
            bootstrap.update(accel_b, gyro_radps, speed_mps);
            let mut row = bootstrap.snapshot(speed_mps, gyro_radps, accel_b);
            row.t_s = (pkt.t_ms - dataset.t0_master_ms) * 1.0e-3;
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
    let mut align = Align::new(*cfg);
    let mut bootstrap = BootstrapDetector::new(*bootstrap_cfg);
    let mut align_initialized = false;
    let mut init_time_s = f64::NAN;
    let mut scan_idx = 0usize;
    let mut interval_start_idx = 0usize;
    let mut prev_nav: Option<(f64, NavPvtObs)> = None;
    let mut samples = Vec::<ResidualSample>::new();

    for (tn, nav) in &dataset.nav_events {
        while scan_idx < dataset.imu_packets.len() && dataset.imu_packets[scan_idx].t_ms <= *tn {
            let pkt = &dataset.imu_packets[scan_idx];
            if !align_initialized {
                let gyro_radps = [
                    pkt.gx_dps.to_radians() as f32,
                    pkt.gy_dps.to_radians() as f32,
                    pkt.gz_dps.to_radians() as f32,
                ];
                let accel_b = [pkt.ax_mps2 as f32, pkt.ay_mps2 as f32, pkt.az_mps2 as f32];
                let speed_mps = speed_for_bootstrap(prev_nav, (*tn, *nav), pkt.t_ms) as f32;
                if bootstrap.update(accel_b, gyro_radps, speed_mps)
                    && align
                        .initialize_from_stationary(&bootstrap.stationary_accel, 0.0)
                        .is_ok()
                {
                    align_initialized = true;
                    init_time_s = (*tn - dataset.t0_master_ms) * 1.0e-3;
                }
            }
            scan_idx += 1;
        }
        if let Some((t_prev, nav_prev)) = prev_nav {
            let dt = ((*tn - t_prev) * 1.0e-3) as f32;
            let interval_packets = &dataset.imu_packets[interval_start_idx..scan_idx];
            if align_initialized && dt > 0.0 && !interval_packets.is_empty() {
                let (mean_gyro_b, mean_accel_b) = mean_imu(interval_packets);
                let window = AlignWindowSummary {
                    dt,
                    mean_gyro_b,
                    mean_accel_b,
                    gnss_vel_prev_n: [
                        nav_prev.vel_n_mps as f32,
                        nav_prev.vel_e_mps as f32,
                        nav_prev.vel_d_mps as f32,
                    ],
                    gnss_vel_curr_n: [
                        nav.vel_n_mps as f32,
                        nav.vel_e_mps as f32,
                        nav.vel_d_mps as f32,
                    ],
                };
                align.update_window(&window);

                let q = align.q_vb;
                let (align_roll_deg, align_pitch_deg, align_yaw_deg) =
                    quat_rpy_alg_deg(q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64);
                let q_align_cmp =
                    quat_from_rpy_alg_deg(align_roll_deg, align_pitch_deg, align_yaw_deg);
                let sigma = align.sigma_deg();
                if let Some((alg_roll_deg, alg_pitch_deg, alg_yaw_deg)) =
                    interpolate_alg(&dataset.alg_events, *tn)
                {
                    let q_alg = quat_from_rpy_alg_deg(alg_roll_deg, alg_pitch_deg, alg_yaw_deg);
                    let rot_err_deg = quat_angle_deg(q_align_cmp, q_alg);
                    let v_prev = [nav_prev.vel_n_mps, nav_prev.vel_e_mps];
                    let v_curr = [nav.vel_n_mps, nav.vel_e_mps];
                    let course_prev = v_prev[1].atan2(v_prev[0]);
                    let course_curr = v_curr[1].atan2(v_curr[0]);
                    let course_rate_dps =
                        wrap_rad_pi(course_curr - course_prev).to_degrees() / (dt as f64);
                    let a_n = [
                        (nav.vel_n_mps - nav_prev.vel_n_mps) / (dt as f64),
                        (nav.vel_e_mps - nav_prev.vel_e_mps) / (dt as f64),
                    ];
                    let v_mid = [0.5 * (v_prev[0] + v_curr[0]), 0.5 * (v_prev[1] + v_curr[1])];
                    let (a_long, a_lat) = if let Some(t_hat) = normalize2(v_mid) {
                        let lat_hat = [-t_hat[1], t_hat[0]];
                        (
                            t_hat[0] * a_n[0] + t_hat[1] * a_n[1],
                            lat_hat[0] * a_n[0] + lat_hat[1] * a_n[1],
                        )
                    } else {
                        (0.0, 0.0)
                    };

                    let err_roll_deg = wrap_deg180(align_roll_deg - alg_roll_deg);
                    let err_pitch_deg = align_pitch_deg - alg_pitch_deg;
                    let err_yaw_deg = wrap_deg180(align_yaw_deg - alg_yaw_deg);
                    samples.push(ResidualSample {
                        t_s: (*tn - dataset.t0_master_ms) * 1.0e-3,
                        align_roll_deg,
                        align_pitch_deg,
                        align_yaw_deg,
                        alg_roll_deg,
                        alg_pitch_deg,
                        alg_yaw_deg,
                        err_roll_deg,
                        err_pitch_deg,
                        err_yaw_deg,
                        sigma_roll_deg: sigma[0] as f64,
                        sigma_pitch_deg: sigma[1] as f64,
                        sigma_yaw_deg: sigma[2] as f64,
                        course_rate_dps,
                        a_lat_mps2: a_lat,
                        a_long_mps2: a_long,
                        rot_err_deg,
                    });
                }
            }
        }
        prev_nav = Some((*tn, *nav));
        interval_start_idx = scan_idx;
    }

    let metrics = score_samples(&samples, dataset.nav_events.len(), init_time_s);
    Ok(EvalResult {
        cfg: *cfg,
        bootstrap_cfg: *bootstrap_cfg,
        metrics,
        samples,
    })
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
        TuneParam::RCourseRateStdDps,
        TuneParam::RLatStdMps2,
        TuneParam::MinSpeedMps,
        TuneParam::MinTurnRateDps,
        TuneParam::MinLatAccMps2,
        TuneParam::RLongStdMps2,
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
    let feature_sets = [
        (
            best.cfg.use_turn_gyro,
            best.cfg.use_course_rate,
            best.cfg.use_lateral_accel,
            best.cfg.use_longitudinal_accel,
        ),
        (true, true, true, false),
        (true, true, false, true),
        (true, true, false, false),
        (false, true, true, false),
        (false, true, true, true),
    ];
    for (use_turn_gyro, use_course_rate, use_lateral_accel, use_longitudinal_accel) in feature_sets
    {
        let mut cfg = best.cfg;
        let bootstrap_cfg = best.bootstrap_cfg;
        cfg.use_turn_gyro = use_turn_gyro;
        cfg.use_course_rate = use_course_rate;
        cfg.use_lateral_accel = use_lateral_accel;
        cfg.use_longitudinal_accel = use_longitudinal_accel;
        let eval = evaluate_config(dataset, &cfg, &bootstrap_cfg)?;
        if eval.metrics.score + 1.0e-6 < best.metrics.score {
            eprintln!(
                "[tune features] turn={} course={} lat={} long={} score {:.3} -> {:.3}",
                use_turn_gyro,
                use_course_rate,
                use_lateral_accel,
                use_longitudinal_accel,
                best.metrics.score,
                eval.metrics.score
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
    for s in samples {
        sum_sq_roll += s.err_roll_deg * s.err_roll_deg;
        sum_sq_pitch += s.err_pitch_deg * s.err_pitch_deg;
        sum_sq_yaw += s.err_yaw_deg * s.err_yaw_deg;
        sum_abs_roll += s.err_roll_deg.abs();
        sum_abs_pitch += s.err_pitch_deg.abs();
        sum_abs_yaw += s.err_yaw_deg.abs();
        sum_rot += s.rot_err_deg;
        max_rot = max_rot.max(s.rot_err_deg);
    }
    let n = samples.len() as f64;
    let rmse_roll_deg = (sum_sq_roll / n).sqrt();
    let rmse_pitch_deg = (sum_sq_pitch / n).sqrt();
    let rmse_yaw_deg = (sum_sq_yaw / n).sqrt();
    let mae_roll_deg = sum_abs_roll / n;
    let mae_pitch_deg = sum_abs_pitch / n;
    let mae_yaw_deg = sum_abs_yaw / n;
    let mean_rot_err_deg = sum_rot / n;
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
}

fn print_config(cfg: &AlignConfig, bootstrap_cfg: &BootstrapConfig) {
    eprintln!(
        "[config] q_std_deg=[{:.4}, {:.4}, {:.4}] r_gravity={:.3} r_turn_dps={:.3} r_course_dps={:.3} r_lat={:.3} r_long={:.3}",
        cfg.q_mount_std_rad[0].to_degrees(),
        cfg.q_mount_std_rad[1].to_degrees(),
        cfg.q_mount_std_rad[2].to_degrees(),
        cfg.r_gravity_std_mps2,
        cfg.r_turn_gyro_std_radps.to_degrees(),
        cfg.r_course_rate_std_radps.to_degrees(),
        cfg.r_lat_std_mps2,
        cfg.r_long_std_mps2
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
        "[config] use_gravity={} use_turn_gyro={} use_course_rate={} use_lateral_accel={} use_longitudinal_accel={}",
        cfg.use_gravity,
        cfg.use_turn_gyro,
        cfg.use_course_rate,
        cfg.use_lateral_accel,
        cfg.use_longitudinal_accel
    );
}

fn write_residual_csv(path: &PathBuf, samples: &[ResidualSample]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "t_s,align_roll_deg,align_pitch_deg,align_yaw_deg,alg_roll_deg,alg_pitch_deg,alg_yaw_deg,err_roll_deg,err_pitch_deg,err_yaw_deg,sigma_roll_deg,sigma_pitch_deg,sigma_yaw_deg,course_rate_dps,a_lat_mps2,a_long_mps2,rot_err_deg"
    )?;
    for s in samples {
        writeln!(
            w,
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
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
            s.rot_err_deg
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
        "t_s,speed_mps,speed_ema_mps,gyro_norm_dps,gyro_ema_dps,accel_norm_err_mps2,accel_err_ema_mps2,stationary,sample_count"
    )?;
    for s in samples {
        writeln!(
            w,
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
            s.t_s,
            s.speed_mps,
            s.speed_ema_mps,
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

fn mean_imu(interval_packets: &[ImuPacket]) -> ([f32; 3], [f32; 3]) {
    let mut gyro_sum = [0.0_f32; 3];
    let mut accel_sum = [0.0_f32; 3];
    for pkt in interval_packets {
        gyro_sum[0] += pkt.gx_dps.to_radians() as f32;
        gyro_sum[1] += pkt.gy_dps.to_radians() as f32;
        gyro_sum[2] += pkt.gz_dps.to_radians() as f32;
        accel_sum[0] += pkt.ax_mps2 as f32;
        accel_sum[1] += pkt.ay_mps2 as f32;
        accel_sum[2] += pkt.az_mps2 as f32;
    }
    let inv_n = 1.0 / (interval_packets.len() as f32);
    (
        [
            gyro_sum[0] * inv_n,
            gyro_sum[1] * inv_n,
            gyro_sum[2] * inv_n,
        ],
        [
            accel_sum[0] * inv_n,
            accel_sum[1] * inv_n,
            accel_sum[2] * inv_n,
        ],
    )
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

fn horizontal_speed(nav: NavPvtObs) -> f64 {
    (nav.vel_n_mps * nav.vel_n_mps + nav.vel_e_mps * nav.vel_e_mps).sqrt()
}

fn interpolate_alg(events: &[AlgEvent], t_ms: f64) -> Option<(f64, f64, f64)> {
    if events.is_empty() {
        return None;
    }
    let idx = events.partition_point(|e| e.t_ms < t_ms);
    if idx == 0 {
        return Some((events[0].roll_deg, events[0].pitch_deg, events[0].yaw_deg));
    }
    if idx >= events.len() {
        let e = events[events.len() - 1];
        return Some((e.roll_deg, e.pitch_deg, e.yaw_deg));
    }

    let e0 = events[idx - 1];
    let e1 = events[idx];
    let dt = e1.t_ms - e0.t_ms;
    if dt.abs() <= 1.0e-9 {
        return Some((e0.roll_deg, e0.pitch_deg, e0.yaw_deg));
    }
    let alpha = ((t_ms - e0.t_ms) / dt).clamp(0.0, 1.0);
    let roll = e0.roll_deg + alpha * (e1.roll_deg - e0.roll_deg);
    let pitch = e0.pitch_deg + alpha * (e1.pitch_deg - e0.pitch_deg);
    let yaw_delta = wrap_deg180(e1.yaw_deg - e0.yaw_deg);
    let yaw = normalize_heading_deg(e0.yaw_deg + alpha * yaw_delta);
    Some((roll, pitch, yaw))
}

fn quat_rpy_alg_deg(q0: f64, q1: f64, q2: f64, q3: f64) -> (f64, f64, f64) {
    let n = (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3).sqrt();
    let (w, x, y, z) = if n > 1.0e-12 {
        (q0 / n, q1 / n, q2 / n, q3 / n)
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r01 = 2.0 * (x * y - w * z);
    let r02 = 2.0 * (x * z + w * y);
    let r12 = 2.0 * (y * z - w * x);
    let r22 = 1.0 - 2.0 * (x * x + y * y);
    let pitch = r02.clamp(-1.0, 1.0).asin();
    let roll = (-r12).atan2(r22);
    let yaw = (-r01).atan2(r00);
    (
        roll.to_degrees(),
        pitch.to_degrees(),
        normalize_heading_deg(yaw.to_degrees()),
    )
}

fn quat_from_rpy_alg_deg(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> [f64; 4] {
    let (sr, cr) = (0.5 * roll_deg.to_radians()).sin_cos();
    let (sp, cp) = (0.5 * pitch_deg.to_radians()).sin_cos();
    let (sy, cy) = (0.5 * yaw_deg.to_radians()).sin_cos();
    quat_normalize([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
}

fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n <= 1.0e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn quat_angle_deg(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dq = quat_normalize(quat_mul(quat_conj(a), b));
    let w = dq[0].abs().clamp(-1.0, 1.0);
    2.0 * w.acos().to_degrees()
}

fn esf_alg_flu_to_frd_mount_deg(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> (f64, f64, f64) {
    (wrap_deg180(180.0 - roll_deg), pitch_deg, yaw_deg)
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize2(v: [f64; 2]) -> Option<[f64; 2]> {
    let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if !n.is_finite() || n <= 1.0e-9 {
        return None;
    }
    Some([v[0] / n, v[1] / n])
}

fn wrap_rad_pi(x: f64) -> f64 {
    let two_pi = 2.0 * PI;
    (x + PI).rem_euclid(two_pi) - PI
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

#[derive(Clone)]
struct MasterTimeline {
    masters: Vec<(u64, f64)>,
    has_itow: bool,
    t0_master_ms: f64,
    master_min: f64,
    master_max: f64,
}

impl MasterTimeline {
    fn map_tag_ms(&self, a: f64, b: f64, tag: f64, seq: u64) -> Option<f64> {
        let seq_ms = nearest_master_ms(seq, &self.masters)?;
        let mut ms = a * tag + b;
        if !ms.is_finite()
            || ms < self.master_min - 1000.0
            || ms > self.master_max + 1000.0
            || (ms - seq_ms).abs() > 2000.0
        {
            ms = seq_ms;
        }
        Some(ms)
    }
}

fn build_master_timeline(frames: &[UbxFrame]) -> MasterTimeline {
    let mut masters: Vec<(u64, f64)> = Vec::new();
    for frame in frames {
        if let Some(itow) = sim::ubxlog::extract_itow_ms(frame) {
            if (0..604_800_000).contains(&itow) {
                masters.push((frame.seq, itow as f64));
            }
        }
    }
    masters.sort_by_key(|x| x.0);

    if !masters.is_empty() {
        let raw: Vec<i64> = masters.iter().map(|(_, ms)| *ms as i64).collect();
        let unwrapped = unwrap_i64_counter(&raw, 604_800_000);
        for (m, msu) in masters.iter_mut().zip(unwrapped.into_iter()) {
            m.1 = msu as f64;
        }

        let mut filtered: Vec<(u64, f64)> = Vec::with_capacity(masters.len());
        let mut last_ms: Option<f64> = None;
        for (seq, ms) in masters.iter().copied() {
            match last_ms {
                None => {
                    filtered.push((seq, ms));
                    last_ms = Some(ms);
                }
                Some(prev) => {
                    let dt = ms - prev;
                    if (0.0..=10_000.0).contains(&dt) {
                        filtered.push((seq, ms));
                        last_ms = Some(ms);
                    }
                }
            }
        }
        if filtered.len() >= 10 {
            masters = filtered;
        }
    }

    let has_itow = !masters.is_empty();
    let t0_master_ms = masters
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::INFINITY, f64::min);
    let t0_master_ms = if t0_master_ms.is_finite() {
        t0_master_ms
    } else {
        0.0
    };
    let master_min = masters
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::INFINITY, f64::min);
    let master_min = if master_min.is_finite() {
        master_min
    } else {
        0.0
    };
    let master_max = masters
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::NEG_INFINITY, f64::max);
    let master_max = if master_max.is_finite() {
        master_max
    } else {
        master_min
    };

    MasterTimeline {
        masters,
        has_itow,
        t0_master_ms,
        master_min,
        master_max,
    }
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
