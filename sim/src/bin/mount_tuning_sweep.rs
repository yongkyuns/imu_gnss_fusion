use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use rayon::prelude::*;
use sensor_fusion::{Config, Filter, SensorFusion};
use sim::datasets::generic_replay::{
    GenericReferenceRpySample, fusion_gnss_sample, fusion_imu_sample, load_gnss_samples,
    load_imu_samples, load_reference_mount_samples,
};
use sim::eval::gnss_ins::{quat_angle_deg, wrap_deg180};
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::visualizer::pipeline::generic::{
    q_bv_to_reference_mount_rpy, reference_mount_rpy_to_q_bv,
};
use sim::visualizer::pipeline::{FilterCompareConfig, apply_filter_compare_config};

#[derive(Parser, Debug)]
struct Args {
    /// One generic replay directory. Can be repeated.
    #[arg(long = "generic-replay-dir")]
    replay_dirs: Vec<PathBuf>,
    /// Root containing generic replay directories, used when no explicit replay dir is supplied.
    #[arg(long, default_value = "target/replay-analysis/field-sweep")]
    replay_root: PathBuf,
    /// Only include replay directory names containing this substring.
    #[arg(long)]
    dataset_filter: Option<String>,
    /// Comma-separated filters to evaluate: reduced,full.
    #[arg(long, default_value = "reduced,full")]
    filters: String,
    /// Comma-separated Reduced and Full mount roll/pitch initial sigmas, in degrees.
    #[arg(long, default_value = "1.2")]
    mount_roll_pitch_init_sigma_deg: String,
    /// Optional comma-separated mount roll initial sigmas, in degrees.
    #[arg(long)]
    mount_roll_init_sigma_deg: Option<String>,
    /// Optional comma-separated mount pitch initial sigmas, in degrees.
    #[arg(long)]
    mount_pitch_init_sigma_deg: Option<String>,
    /// Comma-separated Reduced and Full mount yaw initial sigmas, in degrees.
    #[arg(long, default_value = "6.0")]
    mount_yaw_init_sigma_deg: String,
    /// Comma-separated mount random-walk variances in rad^2/s.
    #[arg(long, default_value = "0.0")]
    mount_rw_var: String,
    /// Optional comma-separated mount roll random-walk variances in rad^2/s.
    #[arg(long)]
    mount_roll_rw_var: Option<String>,
    /// Optional comma-separated mount pitch random-walk variances in rad^2/s.
    #[arg(long)]
    mount_pitch_rw_var: Option<String>,
    /// Optional comma-separated mount yaw random-walk variances in rad^2/s.
    #[arg(long)]
    mount_yaw_rw_var: Option<String>,
    /// Comma-separated Reduced roll/pitch attitude initial sigmas, in degrees.
    #[arg(long, default_value = "2.0")]
    attitude_roll_pitch_init_sigma_deg: String,
    /// Comma-separated Reduced yaw initial sigmas, in degrees.
    #[arg(long, default_value = "6.0")]
    yaw_init_sigma_deg: String,
    /// Comma-separated gyro-bias initial sigmas, in degrees per second.
    #[arg(long, default_value = "0.125")]
    gyro_bias_init_sigma_dps: String,
    /// Comma-separated accel-bias initial sigmas, in meters per second squared.
    #[arg(long, default_value = "0.15")]
    accel_bias_init_sigma_mps2: String,
    /// Ignore samples before this offset from replay start.
    #[arg(long, default_value_t = 50.0)]
    start_after_s: f64,
    /// Tail window used for RMS and covariance coverage metrics.
    #[arg(long, default_value_t = 300.0)]
    tail_s: f64,
    /// Diagnostic sample period used after replay execution.
    #[arg(long, default_value_t = 1.0)]
    sample_period_s: f64,
    /// Number of worker threads. Defaults to available parallelism.
    #[arg(long)]
    jobs: Option<usize>,
    /// CSV output path. Defaults to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FilterKind {
    Reduced,
    Full,
}

#[derive(Clone, Copy, Debug)]
struct Variant {
    index: usize,
    mount_roll_sigma_deg: f32,
    mount_pitch_sigma_deg: f32,
    mount_yaw_sigma_deg: f32,
    mount_rw_var: f32,
    mount_rw_var_axes: [f32; 3],
    attitude_rp_sigma_deg: f32,
    yaw_sigma_deg: f32,
    gyro_bias_sigma_dps: f32,
    accel_bias_sigma_mps2: f32,
}

#[derive(Clone, Debug)]
struct Job {
    replay_dir: PathBuf,
    dataset: String,
    filter: FilterKind,
    variant: Variant,
    start_after_s: f64,
    tail_s: f64,
    sample_period_s: f64,
}

#[derive(Clone, Debug)]
struct SweepRow {
    dataset: String,
    filter: FilterKind,
    variant: Variant,
    initialized: bool,
    init_t_s: f64,
    final_t_s: f64,
    tail_start_t_s: f64,
    final_error_deg: [f64; 3],
    final_qerr_deg: f64,
    final_sigma_deg: [f64; 3],
    tail_count: usize,
    tail_rms_deg: [f64; 3],
    tail_qerr_rms_deg: f64,
    tail_peak_to_peak_deg: [f64; 3],
    tail_max_step_deg: [f64; 3],
    tail_max_q_step_deg: f64,
    within_1sigma: [usize; 3],
    within_2sigma: [usize; 3],
    within_3sigma: [usize; 3],
}

#[derive(Clone, Debug)]
struct MountObservation {
    t_s: f64,
    q_bv: [f64; 4],
    sigma_deg: [f64; 3],
}

#[derive(Clone, Debug)]
struct MetricAccumulator {
    final_obs: Option<MountObservation>,
    final_error_deg: [f64; 3],
    final_qerr_deg: f64,
    tail_start_t_s: f64,
    tail_count: usize,
    sum_sq_axis: [f64; 3],
    sum_sq_qerr: f64,
    tail_min_error_deg: [f64; 3],
    tail_max_error_deg: [f64; 3],
    tail_prev_error_deg: Option<[f64; 3]>,
    tail_prev_q_bv: Option<[f64; 4]>,
    tail_max_step_deg: [f64; 3],
    tail_max_q_step_deg: f64,
    within_1sigma: [usize; 3],
    within_2sigma: [usize; 3],
    within_3sigma: [usize; 3],
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.sample_period_s <= 0.0 {
        bail!("--sample-period-s must be positive");
    }

    let replay_dirs = replay_dirs(&args)?;
    let filters = parse_filters(&args.filters)?;
    let variants = variants(&args)?;
    let jobs = build_jobs(&args, replay_dirs, filters, variants);
    if jobs.is_empty() {
        bail!("no sweep jobs selected");
    }

    let thread_count = args
        .jobs
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from))
        .max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .context("failed to create sweep thread pool")?;

    let mut rows = Vec::new();
    let mut errors = Vec::new();
    let results: Vec<_> = pool.install(|| jobs.par_iter().map(run_job).collect());
    for result in results {
        match result {
            Ok(row) => rows.push(row),
            Err(err) => errors.push(err),
        }
    }
    if !errors.is_empty() {
        for err in &errors {
            eprintln!("sweep job failed: {err:#}");
        }
        bail!("{} sweep jobs failed", errors.len());
    }

    rows.sort_by(|a, b| {
        a.dataset
            .cmp(&b.dataset)
            .then_with(|| a.variant.index.cmp(&b.variant.index))
            .then_with(|| filter_name(a.filter).cmp(filter_name(b.filter)))
    });

    if let Some(path) = args.output {
        let file = fs::File::create(&path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        write_rows(BufWriter::new(file), &rows)?;
        eprintln!(
            "wrote {} rows from {} jobs to {}",
            rows.len(),
            jobs.len(),
            path.display()
        );
    } else {
        write_rows(BufWriter::new(io::stdout()), &rows)?;
    }
    Ok(())
}

fn replay_dirs(args: &Args) -> Result<Vec<PathBuf>> {
    let mut dirs = if args.replay_dirs.is_empty() {
        fs::read_dir(&args.replay_root)
            .with_context(|| format!("failed to read {}", args.replay_root.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.is_dir() && path.join("imu.csv").exists() && path.join("gnss.csv").exists()
            })
            .collect::<Vec<_>>()
    } else {
        args.replay_dirs.clone()
    };
    if let Some(filter) = &args.dataset_filter {
        dirs.retain(|path| dataset_name(path).contains(filter));
    }
    dirs.sort();
    Ok(dirs)
}

fn build_jobs(
    args: &Args,
    replay_dirs: Vec<PathBuf>,
    filters: Vec<FilterKind>,
    variants: Vec<Variant>,
) -> Vec<Job> {
    let mut jobs = Vec::new();
    for replay_dir in replay_dirs {
        let dataset = dataset_name(&replay_dir);
        for &variant in &variants {
            for &filter in &filters {
                jobs.push(Job {
                    replay_dir: replay_dir.clone(),
                    dataset: dataset.clone(),
                    filter,
                    variant,
                    start_after_s: args.start_after_s,
                    tail_s: args.tail_s,
                    sample_period_s: args.sample_period_s,
                });
            }
        }
    }
    jobs
}

fn run_job(job: &Job) -> Result<SweepRow> {
    let imu = load_imu_samples(&job.replay_dir)?;
    let gnss = load_gnss_samples(&job.replay_dir)?;
    let reference_mount = load_reference_mount_samples(&job.replay_dir)?;
    let final_ref = final_reference_mount(&reference_mount)
        .ok_or_else(|| anyhow!("{} has no reference_mount.csv rows", job.dataset))?;
    let final_ref_rpy = [final_ref.roll_deg, final_ref.pitch_deg, final_ref.yaw_deg];
    let final_ref_q = reference_mount_rpy_to_q_bv(final_ref_rpy);
    let start_t = replay_start_t(&imu, &gnss).ok_or_else(|| anyhow!("{} is empty", job.dataset))?;
    let final_t = replay_final_t(&imu, &gnss).unwrap_or(start_t);
    let tail_start_t = (final_t - job.tail_s).max(start_t + job.start_after_s);
    let mut metrics = MetricAccumulator::new(tail_start_t);
    let mut fusion = SensorFusion::with_config(Config {
        filter: match job.filter {
            FilterKind::Reduced => Filter::Reduced,
            FilterKind::Full => Filter::Full,
        },
        ..Default::default()
    });
    apply_filter_compare_config(&mut fusion, config_for_variant(job.variant));

    let mut init_t_s = f64::NAN;
    let mut axis_covariance_applied = false;
    let mut next_sample_t = start_t + job.start_after_s;
    for_each_event(&imu, &gnss, |event| {
        let t_s = match event {
            ReplayEvent::Imu(_, sample) => {
                fusion.process_imu(fusion_imu_sample(*sample));
                sample.t_s
            }
            ReplayEvent::Gnss(_, sample) => {
                fusion.process_gnss(fusion_gnss_sample(*sample));
                sample.t_s
            }
        };
        if observation(job.filter, &fusion, t_s).is_some() {
            if init_t_s.is_nan() {
                init_t_s = t_s;
            }
            if !axis_covariance_applied {
                fusion.analysis_set_mount_covariance_axes(
                    [
                        job.variant.mount_roll_sigma_deg.to_radians(),
                        job.variant.mount_pitch_sigma_deg.to_radians(),
                        job.variant.mount_yaw_sigma_deg.to_radians(),
                    ],
                    true,
                );
                axis_covariance_applied = true;
            }
        }
        while next_sample_t <= t_s {
            if let Some(obs) = observation(job.filter, &fusion, next_sample_t) {
                metrics.record(obs, final_ref_rpy, final_ref_q);
            }
            next_sample_t += job.sample_period_s;
        }
    });
    if let Some(obs) = observation(job.filter, &fusion, final_t) {
        metrics.record(obs, final_ref_rpy, final_ref_q);
    }

    let initialized = metrics.final_obs.is_some();
    Ok(metrics.into_row(
        job.dataset.clone(),
        job.filter,
        job.variant,
        initialized,
        init_t_s,
        final_t,
    ))
}

fn config_for_variant(variant: Variant) -> FilterCompareConfig {
    let mut cfg = FilterCompareConfig {
        mount_roll_pitch_init_sigma_deg: variant
            .mount_roll_sigma_deg
            .max(variant.mount_pitch_sigma_deg),
        mount_roll_init_sigma_deg: variant.mount_roll_sigma_deg,
        mount_pitch_init_sigma_deg: variant.mount_pitch_sigma_deg,
        mount_init_sigma_deg: variant.mount_yaw_sigma_deg,
        attitude_roll_pitch_init_sigma_deg: variant.attitude_rp_sigma_deg,
        yaw_init_sigma_deg: variant.yaw_sigma_deg,
        gyro_bias_init_sigma_dps: variant.gyro_bias_sigma_dps,
        accel_bias_init_sigma_mps2: variant.accel_bias_sigma_mps2,
        mount_align_rw_var: variant.mount_rw_var,
        ..FilterCompareConfig::default()
    };
    cfg.full_init.mount_sigma_deg = variant
        .mount_roll_sigma_deg
        .max(variant.mount_pitch_sigma_deg);
    cfg.full_init.mount_yaw_sigma_deg = variant.mount_yaw_sigma_deg;
    cfg.full_init.attitude_sigma_deg = variant.attitude_rp_sigma_deg.max(variant.yaw_sigma_deg);
    if let Some(noise) = cfg.noise.reduced.as_mut() {
        noise.mount_align_rw_var = variant.mount_rw_var;
        *noise = noise.with_mount_align_rw_var_axes(variant.mount_rw_var_axes);
    }
    if let Some(noise) = cfg.noise.full.as_mut() {
        noise.mount_align_rw_var = variant.mount_rw_var;
        *noise = noise.with_mount_align_rw_var_axes(variant.mount_rw_var_axes);
    }
    cfg
}

fn observation(filter: FilterKind, fusion: &SensorFusion, t_s: f64) -> Option<MountObservation> {
    match filter {
        FilterKind::Reduced => {
            let state = fusion.reduced()?;
            Some(MountObservation {
                t_s,
                q_bv: [
                    state.nominal.qcs0 as f64,
                    state.nominal.qcs1 as f64,
                    state.nominal.qcs2 as f64,
                    state.nominal.qcs3 as f64,
                ],
                sigma_deg: [
                    sigma_deg(state.p[15][15]),
                    sigma_deg(state.p[16][16]),
                    sigma_deg(state.p[17][17]),
                ],
            })
        }
        FilterKind::Full => {
            let state = fusion.full()?;
            Some(MountObservation {
                t_s,
                q_bv: state.qcs64,
                sigma_deg: [
                    sigma_deg(state.p[21][21]),
                    sigma_deg(state.p[22][22]),
                    sigma_deg(state.p[23][23]),
                ],
            })
        }
    }
}

impl MetricAccumulator {
    fn new(tail_start_t_s: f64) -> Self {
        Self {
            final_obs: None,
            final_error_deg: [f64::NAN; 3],
            final_qerr_deg: f64::NAN,
            tail_start_t_s,
            tail_count: 0,
            sum_sq_axis: [0.0; 3],
            sum_sq_qerr: 0.0,
            tail_min_error_deg: [f64::INFINITY; 3],
            tail_max_error_deg: [f64::NEG_INFINITY; 3],
            tail_prev_error_deg: None,
            tail_prev_q_bv: None,
            tail_max_step_deg: [0.0; 3],
            tail_max_q_step_deg: 0.0,
            within_1sigma: [0; 3],
            within_2sigma: [0; 3],
            within_3sigma: [0; 3],
        }
    }

    fn record(&mut self, obs: MountObservation, ref_rpy_deg: [f64; 3], ref_q_bv: [f64; 4]) {
        let error_deg = mount_axis_error_deg(obs.q_bv, ref_rpy_deg);
        let qerr_deg = quat_angle_deg(obs.q_bv, ref_q_bv);
        if obs.t_s >= self.tail_start_t_s {
            self.tail_count += 1;
            self.sum_sq_qerr += qerr_deg * qerr_deg;
            if let Some(prev_q) = self.tail_prev_q_bv {
                self.tail_max_q_step_deg = self
                    .tail_max_q_step_deg
                    .max(quat_angle_deg(obs.q_bv, prev_q));
            }
            for axis in 0..3 {
                let abs_err = error_deg[axis].abs();
                self.sum_sq_axis[axis] += abs_err * abs_err;
                self.tail_min_error_deg[axis] = self.tail_min_error_deg[axis].min(error_deg[axis]);
                self.tail_max_error_deg[axis] = self.tail_max_error_deg[axis].max(error_deg[axis]);
                if let Some(prev) = self.tail_prev_error_deg {
                    self.tail_max_step_deg[axis] = self.tail_max_step_deg[axis]
                        .max(wrap_deg180(error_deg[axis] - prev[axis]).abs());
                }
                if abs_err <= obs.sigma_deg[axis] {
                    self.within_1sigma[axis] += 1;
                }
                if abs_err <= 2.0 * obs.sigma_deg[axis] {
                    self.within_2sigma[axis] += 1;
                }
                if abs_err <= 3.0 * obs.sigma_deg[axis] {
                    self.within_3sigma[axis] += 1;
                }
            }
            self.tail_prev_error_deg = Some(error_deg);
            self.tail_prev_q_bv = Some(obs.q_bv);
        }
        self.final_error_deg = error_deg;
        self.final_qerr_deg = qerr_deg;
        self.final_obs = Some(obs);
    }

    fn into_row(
        self,
        dataset: String,
        filter: FilterKind,
        variant: Variant,
        initialized: bool,
        init_t_s: f64,
        final_t_s: f64,
    ) -> SweepRow {
        let inv = if self.tail_count > 0 {
            1.0 / self.tail_count as f64
        } else {
            f64::NAN
        };
        SweepRow {
            dataset,
            filter,
            variant,
            initialized,
            init_t_s,
            final_t_s,
            tail_start_t_s: self.tail_start_t_s,
            final_error_deg: self.final_error_deg,
            final_qerr_deg: self.final_qerr_deg,
            final_sigma_deg: self
                .final_obs
                .as_ref()
                .map(|obs| obs.sigma_deg)
                .unwrap_or([f64::NAN; 3]),
            tail_count: self.tail_count,
            tail_rms_deg: [
                (self.sum_sq_axis[0] * inv).sqrt(),
                (self.sum_sq_axis[1] * inv).sqrt(),
                (self.sum_sq_axis[2] * inv).sqrt(),
            ],
            tail_qerr_rms_deg: (self.sum_sq_qerr * inv).sqrt(),
            tail_peak_to_peak_deg: [
                self.tail_max_error_deg[0] - self.tail_min_error_deg[0],
                self.tail_max_error_deg[1] - self.tail_min_error_deg[1],
                self.tail_max_error_deg[2] - self.tail_min_error_deg[2],
            ],
            tail_max_step_deg: self.tail_max_step_deg,
            tail_max_q_step_deg: self.tail_max_q_step_deg,
            within_1sigma: self.within_1sigma,
            within_2sigma: self.within_2sigma,
            within_3sigma: self.within_3sigma,
        }
    }
}

fn variants(args: &Args) -> Result<Vec<Variant>> {
    let mount_rp = parse_f32_list(&args.mount_roll_pitch_init_sigma_deg)?;
    let mount_pairs = match (
        &args.mount_roll_init_sigma_deg,
        &args.mount_pitch_init_sigma_deg,
    ) {
        (None, None) => mount_rp.iter().map(|&sigma| (sigma, sigma)).collect(),
        (roll, pitch) => {
            let roll = match roll {
                Some(values) => parse_f32_list(values)?,
                None => mount_rp.clone(),
            };
            let pitch = match pitch {
                Some(values) => parse_f32_list(values)?,
                None => mount_rp.clone(),
            };
            let mut pairs = Vec::new();
            for &roll_sigma in &roll {
                for &pitch_sigma in &pitch {
                    pairs.push((roll_sigma, pitch_sigma));
                }
            }
            pairs
        }
    };
    let mount_yaw = parse_f32_list(&args.mount_yaw_init_sigma_deg)?;
    let mount_rw = parse_f32_list(&args.mount_rw_var)?;
    let mount_rw_axes = mount_rw_axis_sets(args, &mount_rw)?;
    let attitude_rp = parse_f32_list(&args.attitude_roll_pitch_init_sigma_deg)?;
    let yaw = parse_f32_list(&args.yaw_init_sigma_deg)?;
    let gyro_bias = parse_f32_list(&args.gyro_bias_init_sigma_dps)?;
    let accel_bias = parse_f32_list(&args.accel_bias_init_sigma_mps2)?;

    let mut variants = Vec::new();
    for &(mount_roll_sigma_deg, mount_pitch_sigma_deg) in &mount_pairs {
        for &mount_yaw_sigma_deg in &mount_yaw {
            for &(mount_rw_var, mount_rw_var_axes) in &mount_rw_axes {
                for &attitude_rp_sigma_deg in &attitude_rp {
                    for &yaw_sigma_deg in &yaw {
                        for &gyro_bias_sigma_dps in &gyro_bias {
                            for &accel_bias_sigma_mps2 in &accel_bias {
                                variants.push(Variant {
                                    index: variants.len(),
                                    mount_roll_sigma_deg,
                                    mount_pitch_sigma_deg,
                                    mount_yaw_sigma_deg,
                                    mount_rw_var,
                                    mount_rw_var_axes,
                                    attitude_rp_sigma_deg,
                                    yaw_sigma_deg,
                                    gyro_bias_sigma_dps,
                                    accel_bias_sigma_mps2,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(variants)
}

fn mount_rw_axis_sets(args: &Args, scalar: &[f32]) -> Result<Vec<(f32, [f32; 3])>> {
    if args.mount_roll_rw_var.is_none()
        && args.mount_pitch_rw_var.is_none()
        && args.mount_yaw_rw_var.is_none()
    {
        return Ok(scalar.iter().map(|&v| (v, [v, v, v])).collect());
    }

    let roll = parse_axis_or_scalar(args.mount_roll_rw_var.as_deref(), scalar)?;
    let pitch = parse_axis_or_scalar(args.mount_pitch_rw_var.as_deref(), scalar)?;
    let yaw = parse_axis_or_scalar(args.mount_yaw_rw_var.as_deref(), scalar)?;
    let mut sets = Vec::new();
    for &roll_var in &roll {
        for &pitch_var in &pitch {
            for &yaw_var in &yaw {
                sets.push((0.0, [roll_var, pitch_var, yaw_var]));
            }
        }
    }
    Ok(sets)
}

fn parse_axis_or_scalar(input: Option<&str>, scalar: &[f32]) -> Result<Vec<f32>> {
    match input {
        Some(input) => parse_f32_list(input),
        None => Ok(scalar.to_vec()),
    }
}

fn parse_filters(input: &str) -> Result<Vec<FilterKind>> {
    input
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|item| match item {
            "reduced" => Ok(FilterKind::Reduced),
            "full" => Ok(FilterKind::Full),
            other => bail!("unknown filter '{other}', expected reduced or full"),
        })
        .collect()
}

fn parse_f32_list(input: &str) -> Result<Vec<f32>> {
    let values = input
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|item| {
            item.parse::<f32>()
                .with_context(|| format!("failed to parse '{item}' as f32"))
        })
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        bail!("list must contain at least one value");
    }
    Ok(values)
}

fn write_rows(mut writer: impl Write, rows: &[SweepRow]) -> Result<()> {
    writeln!(
        writer,
        "dataset,filter,variant,mount_roll_sigma_deg,mount_pitch_sigma_deg,mount_yaw_sigma_deg,mount_rw_var,mount_roll_rw_var,mount_pitch_rw_var,mount_yaw_rw_var,attitude_rp_sigma_deg,yaw_sigma_deg,gyro_bias_sigma_dps,accel_bias_sigma_mps2,initialized,init_t_s,final_t_s,tail_start_t_s,tail_count,final_roll_err_deg,final_pitch_err_deg,final_yaw_err_deg,final_qerr_deg,final_roll_sigma_deg,final_pitch_sigma_deg,final_yaw_sigma_deg,tail_roll_rms_deg,tail_pitch_rms_deg,tail_yaw_rms_deg,tail_qerr_rms_deg,tail_roll_peak_to_peak_deg,tail_pitch_peak_to_peak_deg,tail_yaw_peak_to_peak_deg,tail_roll_max_step_deg,tail_pitch_max_step_deg,tail_yaw_max_step_deg,tail_q_max_step_deg,roll_within_1sigma,pitch_within_1sigma,yaw_within_1sigma,roll_within_2sigma,pitch_within_2sigma,yaw_within_2sigma,roll_within_3sigma,pitch_within_3sigma,yaw_within_3sigma"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{:.6},{:.6},{:.6},{:.9e},{:.9e},{:.9e},{:.9e},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{},{},{},{},{}",
            csv_escape(&row.dataset),
            filter_name(row.filter),
            csv_escape(&variant_name(row.variant)),
            row.variant.mount_roll_sigma_deg,
            row.variant.mount_pitch_sigma_deg,
            row.variant.mount_yaw_sigma_deg,
            row.variant.mount_rw_var,
            row.variant.mount_rw_var_axes[0],
            row.variant.mount_rw_var_axes[1],
            row.variant.mount_rw_var_axes[2],
            row.variant.attitude_rp_sigma_deg,
            row.variant.yaw_sigma_deg,
            row.variant.gyro_bias_sigma_dps,
            row.variant.accel_bias_sigma_mps2,
            row.initialized,
            row.init_t_s,
            row.final_t_s,
            row.tail_start_t_s,
            row.tail_count,
            row.final_error_deg[0],
            row.final_error_deg[1],
            row.final_error_deg[2],
            row.final_qerr_deg,
            row.final_sigma_deg[0],
            row.final_sigma_deg[1],
            row.final_sigma_deg[2],
            row.tail_rms_deg[0],
            row.tail_rms_deg[1],
            row.tail_rms_deg[2],
            row.tail_qerr_rms_deg,
            row.tail_peak_to_peak_deg[0],
            row.tail_peak_to_peak_deg[1],
            row.tail_peak_to_peak_deg[2],
            row.tail_max_step_deg[0],
            row.tail_max_step_deg[1],
            row.tail_max_step_deg[2],
            row.tail_max_q_step_deg,
            row.within_1sigma[0],
            row.within_1sigma[1],
            row.within_1sigma[2],
            row.within_2sigma[0],
            row.within_2sigma[1],
            row.within_2sigma[2],
            row.within_3sigma[0],
            row.within_3sigma[1],
            row.within_3sigma[2],
        )?;
    }
    Ok(())
}

fn variant_name(variant: Variant) -> String {
    let mut name = String::new();
    let _ = write!(
        name,
        "m_roll_{:.3}_m_pitch_{:.3}_m_yaw_{:.3}_m_rw_{:.1e}_{:.1e}_{:.1e}_att_rp_{:.3}_att_yaw_{:.3}_gb_{:.4}_ab_{:.3}",
        variant.mount_roll_sigma_deg,
        variant.mount_pitch_sigma_deg,
        variant.mount_yaw_sigma_deg,
        variant.mount_rw_var_axes[0],
        variant.mount_rw_var_axes[1],
        variant.mount_rw_var_axes[2],
        variant.attitude_rp_sigma_deg,
        variant.yaw_sigma_deg,
        variant.gyro_bias_sigma_dps,
        variant.accel_bias_sigma_mps2
    );
    name
}

fn mount_axis_error_deg(q_bv: [f64; 4], ref_rpy_deg: [f64; 3]) -> [f64; 3] {
    let rpy = q_bv_to_reference_mount_rpy(q_bv);
    [
        wrap_deg180(rpy.0 - ref_rpy_deg[0]),
        wrap_deg180(rpy.1 - ref_rpy_deg[1]),
        wrap_deg180(rpy.2 - ref_rpy_deg[2]),
    ]
}

fn sigma_deg(var_rad2: f32) -> f64 {
    (var_rad2.max(0.0) as f64).sqrt().to_degrees()
}

fn replay_start_t(
    imu: &[sim::datasets::generic_replay::GenericImuSample],
    gnss: &[sim::datasets::generic_replay::GenericGnssSample],
) -> Option<f64> {
    match (imu.first(), gnss.first()) {
        (Some(imu), Some(gnss)) => Some(imu.t_s.min(gnss.t_s)),
        (Some(imu), None) => Some(imu.t_s),
        (None, Some(gnss)) => Some(gnss.t_s),
        (None, None) => None,
    }
}

fn replay_final_t(
    imu: &[sim::datasets::generic_replay::GenericImuSample],
    gnss: &[sim::datasets::generic_replay::GenericGnssSample],
) -> Option<f64> {
    match (imu.last(), gnss.last()) {
        (Some(imu), Some(gnss)) => Some(imu.t_s.max(gnss.t_s)),
        (Some(imu), None) => Some(imu.t_s),
        (None, Some(gnss)) => Some(gnss.t_s),
        (None, None) => None,
    }
}

fn final_reference_mount(
    samples: &[GenericReferenceRpySample],
) -> Option<GenericReferenceRpySample> {
    samples
        .iter()
        .copied()
        .max_by(|a, b| a.t_s.total_cmp(&b.t_s))
}

fn dataset_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("dataset")
        .to_owned()
}

fn filter_name(filter: FilterKind) -> &'static str {
    match filter {
        FilterKind::Reduced => "reduced",
        FilterKind::Full => "full",
    }
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_comma_separated_f32_values() {
        assert_eq!(parse_f32_list("1.0, 2.5,3").unwrap(), vec![1.0, 2.5, 3.0]);
    }

    #[test]
    fn builds_cartesian_variant_grid() {
        let args = Args {
            replay_dirs: Vec::new(),
            replay_root: PathBuf::new(),
            dataset_filter: None,
            filters: "reduced".to_owned(),
            mount_roll_pitch_init_sigma_deg: "1,2".to_owned(),
            mount_roll_init_sigma_deg: None,
            mount_pitch_init_sigma_deg: None,
            mount_yaw_init_sigma_deg: "6".to_owned(),
            mount_rw_var: "0,1e-9".to_owned(),
            mount_roll_rw_var: None,
            mount_pitch_rw_var: None,
            mount_yaw_rw_var: None,
            attitude_roll_pitch_init_sigma_deg: "2".to_owned(),
            yaw_init_sigma_deg: "6".to_owned(),
            gyro_bias_init_sigma_dps: "0.125".to_owned(),
            accel_bias_init_sigma_mps2: "0.15".to_owned(),
            start_after_s: 50.0,
            tail_s: 300.0,
            sample_period_s: 1.0,
            jobs: None,
            output: None,
        };
        let variants = variants(&args).unwrap();
        assert_eq!(variants.len(), 4);
        assert_eq!(variants[3].index, 3);
        assert_eq!(
            variants[0].mount_roll_sigma_deg,
            variants[0].mount_pitch_sigma_deg
        );
    }
}
