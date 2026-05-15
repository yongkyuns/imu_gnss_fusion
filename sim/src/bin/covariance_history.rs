#![allow(
    clippy::collapsible_if,
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::ptr_arg,
    clippy::too_many_arguments
)]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use sensor_fusion::SensorFusion;
use sensor_fusion::full::ERROR_STATES;
use sensor_fusion::generated_full;
use sensor_fusion::generated_reduced;
use sensor_fusion::reduced::UPDATE_DIAG_TYPES;
use sensor_fusion::{ProcessNoise, full, reduced};
use sim::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, GenericReferenceRpySample, fusion_gnss_sample,
    fusion_imu_sample, load_gnss_samples, load_imu_samples, load_reference_attitude_samples,
    load_reference_mount_samples,
};
use sim::eval::gnss_ins::{as_q64, quat_angle_deg, quat_conj, quat_mul};
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::visualizer::math::{ecef_to_ned, lla_to_ecef};
use sim::visualizer::model::VisualizerMountMode;
use sim::visualizer::pipeline::FilterCompareConfig;
use sim::visualizer::pipeline::reference::reference_mount_rpy_to_q_bv;
use sim::visualizer::pipeline::synthetic::{
    SyntheticNoiseMode, SyntheticVisualizerConfig, build_synthetic_replay_input,
};

const DIAG_BODY_VEL_Y: usize = 4;
const DIAG_BODY_VEL_Z: usize = 5;
const RUNTIME_NHC_MIN_SPEED_MPS: f32 = 0.05;
const RUNTIME_NHC_MAX_GYRO_NORM_RADPS: f32 = 0.2;
const RUNTIME_NHC_MAX_ACCEL_NORM_ERR_MPS2: f32 = 1.0;

#[derive(Parser, Debug)]
struct Args {
    /// Directory containing imu.csv, gnss.csv, and optional reference CSVs.
    #[arg(long)]
    generic_replay_dir: Option<PathBuf>,

    /// Synthetic motion DSL/CSV path to generate a replay on the fly.
    #[arg(long, alias = "synthetic-scenario")]
    synthetic_motion_def: Option<PathBuf>,

    /// Synthetic sensor noise level.
    #[arg(long, value_enum, default_value_t = SyntheticNoiseArg::Truth)]
    synthetic_noise: SyntheticNoiseArg,

    /// Synthetic noise seed.
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

    /// Snapshot times in replay-relative seconds. Defaults cover the mixed-road startup interval.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "86.736,100,116,120,160,240"
    )]
    times: Vec<f64>,

    /// Optional replay-relative event-trace window, formatted as start,end seconds.
    #[arg(long, value_delimiter = ',')]
    trace_window: Option<Vec<f64>>,

    /// Optional CSV path for per-event update allocation diagnostics.
    #[arg(long)]
    allocation_csv: Option<PathBuf>,

    /// Optional CSV path for per-row Reduced/Full GNSS update parity diagnostics.
    #[arg(long)]
    gnss_parity_csv: Option<PathBuf>,

    /// Optional CSV path for per-row Reduced/Full diagnostics in a common Reduced/NED basis.
    #[arg(long)]
    row_basis_csv: Option<PathBuf>,

    /// Optional CSV path for Full P(velocity,mount) covariance accounting diagnostics.
    #[arg(long)]
    pvm_accounting_csv: Option<PathBuf>,

    /// Optional replay-relative interval summary window, formatted as start,end seconds.
    #[arg(long, value_delimiter = ',')]
    summary_window: Option<Vec<f64>>,

    /// Optional replay-relative GNSS parity CSV window, formatted as start,end seconds.
    #[arg(long, value_delimiter = ',')]
    gnss_parity_window: Option<Vec<f64>>,

    /// Optional replay-relative row-basis CSV window, formatted as start,end seconds.
    #[arg(long, value_delimiter = ',')]
    row_basis_window: Option<Vec<f64>>,

    /// Optional replay-relative Full P(velocity,mount) accounting window, formatted as start,end seconds.
    #[arg(long, value_delimiter = ',')]
    pvm_accounting_window: Option<Vec<f64>>,

    /// Optional replay-relative duration cap, in seconds.
    #[arg(long)]
    max_time_s: Option<f64>,

    /// Minimum spacing for periodic trace lines. Update events are printed regardless.
    #[arg(long, default_value_t = 0.10)]
    trace_interval_s: f64,

    /// Mount mode for the Reduced diagnostic path: auto or manual/reference.
    #[arg(
        long,
        default_value = "auto",
        value_parser = VisualizerMountMode::from_cli_value
    )]
    misalignment: VisualizerMountMode,

    /// Freeze mount states in the Reduced diagnostic path.
    #[arg(long)]
    freeze_misalignment_states: bool,

    /// Override Reduced/full lateral no-motion measurement standard deviation.
    #[arg(long)]
    r_body_vel: Option<f32>,

    /// Override Reduced/full vertical no-motion measurement standard deviation.
    #[arg(long)]
    r_body_vel_z: Option<f32>,

    /// Override Reduced mount roll/pitch initial sigma, in degrees.
    #[arg(long)]
    mount_roll_pitch_init_sigma_deg: Option<f32>,

    /// Override Reduced mount yaw initial sigma, in degrees.
    #[arg(long)]
    mount_init_sigma_deg: Option<f32>,

    /// Override Reduced initial yaw sigma, in degrees.
    #[arg(long)]
    yaw_init_sigma_deg: Option<f32>,

    /// Override Reduced initial gyro-bias sigma, in deg/s.
    #[arg(long)]
    gyro_bias_init_sigma_dps: Option<f32>,

    /// Override Reduced initial accelerometer-bias sigma, in m/s^2.
    #[arg(long)]
    accel_bias_init_sigma_mps2: Option<f32>,

    /// Diagnostic-only override for Reduced residual attitude roll/pitch covariance after initialization.
    #[arg(long)]
    reduced_attitude_roll_pitch_sigma_deg: Option<f32>,

    /// Diagnostic override for Reduced runtime zero-velocity measurement variance.
    #[arg(long)]
    r_zero_vel: Option<f32>,

    /// Diagnostic-only scale applied to Reduced accelerometer white-noise variance.
    #[arg(long, default_value_t = 1.0)]
    reduced_accel_var_scale: f32,

    /// Diagnostic-only scale applied to Reduced gyro white-noise variance.
    #[arg(long, default_value_t = 1.0)]
    reduced_gyro_var_scale: f32,

    /// Diagnostic-only override for full gyro-scale initial sigma.
    #[arg(long)]
    full_gyro_scale_sigma: Option<f32>,

    /// Diagnostic-only override for full accel-scale initial sigma.
    #[arg(long)]
    full_accel_scale_sigma: Option<f32>,

    /// Diagnostic-only override for full roll/pitch attitude initial sigma, in degrees.
    #[arg(long)]
    full_attitude_sigma_deg: Option<f32>,

    /// Diagnostic-only override for full yaw attitude initial sigma, in degrees.
    #[arg(long)]
    full_attitude_yaw_sigma_deg: Option<f32>,

    /// Diagnostic-only override for full mount roll/pitch initial sigma, in degrees.
    #[arg(long)]
    full_mount_sigma_deg: Option<f32>,

    /// Diagnostic-only override for full mount yaw initial sigma, in degrees.
    #[arg(long)]
    full_mount_yaw_sigma_deg: Option<f32>,

    /// Diagnostic-only scale applied to standalone Full GNSS velocity standard deviations.
    #[arg(long, default_value_t = 1.0)]
    full_gnss_vel_std_scale: f32,

    /// Diagnostic-only scale applied to Reduced GNSS position standard deviations.
    #[arg(long, default_value_t = 1.0)]
    reduced_gnss_pos_std_scale: f64,

    /// Diagnostic-only scale applied to Reduced GNSS down-position standard deviations.
    #[arg(long, default_value_t = 1.0)]
    reduced_gnss_pos_d_std_scale: f64,

    /// Diagnostic-only scale applied to Reduced GNSS velocity standard deviations.
    #[arg(long, default_value_t = 1.0)]
    reduced_gnss_vel_std_scale: f64,

    /// Diagnostic-only scale for direct mount correction from GNSS velocity rows.
    #[arg(long, default_value_t = 1.0)]
    gnss_vel_mount_gain_scale: f32,

    /// Sliding-window width for mount-drive conflict diagnostics.
    #[arg(long, default_value_t = 5.0)]
    mount_drive_window_s: f64,

    /// Diagnostic-only adaptive GNSS velocity standard-deviation inflation from mount-drive conflict.
    #[arg(long)]
    adaptive_gnss_velocity_r: bool,

    /// GNSS velocity standard-deviation scale used while adaptive conflict is active.
    #[arg(long, default_value_t = 3.0)]
    adaptive_gnss_velocity_std_scale: f32,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SyntheticNoiseArg {
    Truth,
    Low,
    Mid,
    High,
}

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

#[derive(Clone)]
struct Replay {
    imu: Vec<GenericImuSample>,
    gnss: Vec<GenericGnssSample>,
    reference_attitude: Vec<GenericReferenceRpySample>,
    reference_mount: Vec<GenericReferenceRpySample>,
}

#[derive(Clone)]
struct Snapshot {
    target_rel_s: f64,
    t_s: f64,
    reduced: Option<reduced::State>,
    full: Option<FullSnapshot>,
    transition: Option<TransitionSnapshot>,
    reference_mount_q_bv: Option<[f64; 4]>,
    reference_att_q: Option<[f64; 4]>,
    reduced_type_counts: [u32; UPDATE_DIAG_TYPES],
    full_obs_counts: [u32; 9],
    full_mount_dx_sum: [f32; 3],
    full_mount_dx_abs_sum: [f32; 3],
    full_att_dx_sum: [f32; 3],
    full_att_dx_abs_sum: [f32; 3],
    full_vel_dx_sum: [f32; 3],
    full_vel_dx_abs_sum: [f32; 3],
    full_mount_dx_sum_by_type: [[f32; 3]; 9],
    full_mount_dx_abs_sum_by_type: [[f32; 3]; 9],
    full_residual_sum_by_type: [f32; 9],
    full_residual_abs_sum_by_type: [f32; 9],
    full_effective_residual_sum_by_type: [f32; 9],
    full_effective_residual_abs_sum_by_type: [f32; 9],
    full_nis_sum_by_type: [f32; 9],
    full_nis_max_by_type: [f32; 9],
}

#[derive(Clone, Copy, Debug)]
struct TransitionSnapshot {
    dt_s: f32,
    reduced_mount_to_att: [[f32; 3]; 3],
    reduced_mount_to_vel: [[f32; 3]; 3],
    reduced_mount_to_pos: [[f32; 3]; 3],
    full_mount_to_att: [[f32; 3]; 3],
    full_mount_to_vel: [[f32; 3]; 3],
    full_mount_to_pos: [[f32; 3]; 3],
}

#[derive(Clone)]
struct FullSnapshot {
    nominal: full::NominalState,
    p: [[f32; ERROR_STATES]; ERROR_STATES],
    pos_ecef: [f64; 3],
    last_dx: [f32; ERROR_STATES],
    last_obs_types: Vec<i32>,
}

#[derive(Clone, Default)]
struct TraceState {
    initialized: bool,
    last_trace_t_s: Option<f64>,
    prev_reduced_counts: [u32; UPDATE_DIAG_TYPES],
    prev_full_counts: [u32; 9],
    prev_reduced_sum_dx_mount_roll: [f32; UPDATE_DIAG_TYPES],
    prev_reduced_sum_dx_mount_pitch: [f32; UPDATE_DIAG_TYPES],
    prev_reduced_sum_dx_mount_yaw: [f32; UPDATE_DIAG_TYPES],
}

struct AllocationCsv {
    writer: BufWriter<File>,
    window_abs: Option<[f64; 2]>,
    prev_reduced_counts: [u32; UPDATE_DIAG_TYPES],
    prev_sum_dx_att_roll: [f32; UPDATE_DIAG_TYPES],
    prev_sum_dx_pitch: [f32; UPDATE_DIAG_TYPES],
    prev_sum_dx_yaw: [f32; UPDATE_DIAG_TYPES],
    prev_sum_dx_vel_n: [f32; UPDATE_DIAG_TYPES],
    prev_sum_dx_vel_e: [f32; UPDATE_DIAG_TYPES],
    prev_sum_dx_vel_d: [f32; UPDATE_DIAG_TYPES],
    prev_sum_dx_mount_roll: [f32; UPDATE_DIAG_TYPES],
    prev_sum_dx_mount_pitch: [f32; UPDATE_DIAG_TYPES],
    prev_sum_dx_mount_yaw: [f32; UPDATE_DIAG_TYPES],
    prev_sum_dx_gyro_bias: [[f32; 3]; UPDATE_DIAG_TYPES],
    prev_sum_dx_accel_bias: [[f32; 3]; UPDATE_DIAG_TYPES],
    prev_sum_innovation: [f32; UPDATE_DIAG_TYPES],
    prev_sum_abs_innovation: [f32; UPDATE_DIAG_TYPES],
    prev_sum_nis: [f32; UPDATE_DIAG_TYPES],
    prev_sum_h_mount_norm: [f32; UPDATE_DIAG_TYPES],
    prev_sum_k_mount_norm: [f32; UPDATE_DIAG_TYPES],
    initialized: bool,
}

struct GnssParityCsv {
    writer: BufWriter<File>,
    window_abs: Option<[f64; 2]>,
}

struct RowBasisCsv {
    writer: BufWriter<File>,
    window_abs: Option<[f64; 2]>,
}

#[derive(Clone, Copy, Default)]
struct PvmPhaseStats {
    count: u64,
    net: [[f64; 3]; 3],
    activity_fro: f64,
    activity_abs: f64,
    mount_dx_sum_deg: [f64; 3],
    mount_dx_abs_sum_deg: [f64; 3],
    mount_score_deg2: f64,
    mount_axis_score_deg2: [f64; 3],
}

#[derive(Clone, Copy, Default)]
struct MountDrivePhaseStats {
    count: u64,
    dx_sum_deg: [f64; 3],
    dx_abs_sum_deg: [f64; 3],
    score_deg2: f64,
    axis_score_deg2: [f64; 3],
    nis_sum: f64,
    nis_weight: f64,
    nis_max: f64,
    effective_residual_abs_sum: f64,
}

impl MountDrivePhaseStats {
    fn record(&mut self, error_deg: [f64; 3], dx_deg: [f64; 3], nis: f64, effective_residual: f64) {
        self.count += 1;
        let (score, _) = mount_correction_score(error_deg, dx_deg);
        self.score_deg2 += score;
        if nis.is_finite() {
            self.nis_sum += nis;
            self.nis_weight += 1.0;
            self.nis_max = self.nis_max.max(nis);
        }
        if effective_residual.is_finite() {
            self.effective_residual_abs_sum += effective_residual.abs();
        }
        for idx in 0..3 {
            self.dx_sum_deg[idx] += dx_deg[idx];
            self.dx_abs_sum_deg[idx] += dx_deg[idx].abs();
            self.axis_score_deg2[idx] += -error_deg[idx] * dx_deg[idx];
        }
    }
}

#[derive(Clone, Copy, Default)]
struct MountDriveWindow {
    gnss_velocity: MountDrivePhaseStats,
    nhc: MountDrivePhaseStats,
}

impl MountDriveWindow {
    fn phase_mut(&mut self, phase: &str) -> Option<&mut MountDrivePhaseStats> {
        match phase {
            "gnss_velocity" => Some(&mut self.gnss_velocity),
            "nhc" => Some(&mut self.nhc),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct AdaptiveGnssVelocityStats {
    activations: u32,
    scaled_gnss_count: u32,
    last_scale: f32,
    last_conflict_deg: f64,
    last_conflict_ratio: f64,
    last_gnss_nis_mean: f64,
    last_nhc_nis_mean: f64,
}

#[derive(Clone)]
struct AdaptiveGnssVelocityScale {
    enabled: bool,
    window_s: f64,
    std_scale: f32,
    min_conflict_deg: f64,
    min_conflict_ratio: f64,
    min_gnss_nis_ratio: f64,
    current_window: Option<i64>,
    window: MountDriveWindow,
    evidence: MountDriveWindow,
    current_scale: f32,
    stats: AdaptiveGnssVelocityStats,
}

impl AdaptiveGnssVelocityScale {
    fn new(enabled: bool, window_s: f64, std_scale: f32) -> Self {
        Self {
            enabled,
            window_s: window_s.max(0.5),
            std_scale: std_scale.max(1.0),
            min_conflict_deg: 0.25,
            min_conflict_ratio: 0.45,
            min_gnss_nis_ratio: 2.0,
            current_window: None,
            window: MountDriveWindow::default(),
            evidence: MountDriveWindow::default(),
            current_scale: 1.0,
            stats: AdaptiveGnssVelocityStats::default(),
        }
    }

    fn current_scale(&self) -> f32 {
        if self.enabled {
            self.current_scale
        } else {
            1.0
        }
    }

    fn mark_gnss_used(&mut self) {
        if self.enabled && self.current_scale > 1.0 {
            self.stats.scaled_gnss_count += 1;
        }
    }

    fn record(
        &mut self,
        rel_s: f64,
        phase: &str,
        error_deg: [f64; 3],
        dx_deg: [f64; 3],
        nis: f64,
        effective_residual: f64,
    ) {
        if !self.enabled
            || !matches!(phase, "gnss_velocity" | "nhc")
            || !rel_s.is_finite()
            || !dx_deg.iter().all(|v| v.is_finite())
        {
            return;
        }
        let window_idx = (rel_s / self.window_s).floor() as i64;
        if self.current_window != Some(window_idx) {
            self.current_window = Some(window_idx);
            self.window = MountDriveWindow::default();
        }
        decay_mount_drive_window(&mut self.evidence, 0.97);
        if let Some(stats) = self.window.phase_mut(phase) {
            stats.record(error_deg, dx_deg, nis, effective_residual);
        }
        if let Some(stats) = self.evidence.phase_mut(phase) {
            stats.record(error_deg, dx_deg, nis, effective_residual);
        }
        self.update_scale();
    }

    fn update_scale(&mut self) {
        let gnss = self.evidence.gnss_velocity;
        let nhc = self.evidence.nhc;
        if gnss.count == 0 || nhc.count == 0 {
            self.current_scale = 1.0;
            return;
        }
        let gnss_norm = vec3_norm(gnss.dx_sum_deg);
        let nhc_norm = vec3_norm(nhc.dx_sum_deg);
        let denom = gnss_norm + nhc_norm;
        if denom <= 0.0 {
            self.current_scale = 1.0;
            return;
        }
        let conflict_deg = (denom - vec3_norm(vec3_add(gnss.dx_sum_deg, nhc.dx_sum_deg))).max(0.0);
        let conflict_ratio = conflict_deg / denom;
        let gnss_nis_mean = gnss.nis_sum / gnss.nis_weight.max(1.0);
        let nhc_nis_mean = nhc.nis_sum / nhc.nis_weight.max(1.0);
        let gnss_less_credible = gnss.score_deg2 < 0.0
            && gnss_nis_mean > (nhc_nis_mean + 1.0e-9) * self.min_gnss_nis_ratio;
        let scale = if conflict_deg >= self.min_conflict_deg
            && conflict_ratio >= self.min_conflict_ratio
            && gnss_less_credible
        {
            self.std_scale
        } else {
            1.0
        };
        if self.current_scale <= 1.0 && scale > 1.0 {
            self.stats.activations += 1;
        }
        self.current_scale = scale;
        self.stats.last_scale = scale;
        self.stats.last_conflict_deg = conflict_deg;
        self.stats.last_conflict_ratio = conflict_ratio;
        self.stats.last_gnss_nis_mean = gnss_nis_mean;
        self.stats.last_nhc_nis_mean = nhc_nis_mean;
    }

    fn print_summary(&self, label: &str) {
        if !self.enabled {
            return;
        }
        println!(
            "[covhist-adaptive] system={label} activations={} scaled_gnss_count={} last_scale={:.3} last_conflict_deg={:.6} last_ratio={:.3} last_nis_mean_gnss={:.3} last_nis_mean_nhc={:.3}",
            self.stats.activations,
            self.stats.scaled_gnss_count,
            self.stats.last_scale,
            self.stats.last_conflict_deg,
            self.stats.last_conflict_ratio,
            self.stats.last_gnss_nis_mean,
            self.stats.last_nhc_nis_mean,
        );
    }
}

fn decay_mount_drive_window(window: &mut MountDriveWindow, factor: f64) {
    decay_mount_drive_phase(&mut window.gnss_velocity, factor);
    decay_mount_drive_phase(&mut window.nhc, factor);
}

fn decay_mount_drive_phase(phase: &mut MountDrivePhaseStats, factor: f64) {
    phase.nis_sum *= factor;
    phase.nis_weight *= factor;
    phase.effective_residual_abs_sum *= factor;
    phase.score_deg2 *= factor;
    for idx in 0..3 {
        phase.dx_sum_deg[idx] *= factor;
        phase.dx_abs_sum_deg[idx] *= factor;
        phase.axis_score_deg2[idx] *= factor;
    }
}

struct PvmAccountingCsv {
    writer: BufWriter<File>,
    window_abs: Option<[f64; 2]>,
    stats: [PvmPhaseStats; PVM_PHASES.len()],
    mount_drive_window_s: f64,
    mount_drive_windows: BTreeMap<(u8, i64), MountDriveWindow>,
}

#[derive(Clone, Copy)]
struct MountDirectionDiag {
    error_deg: [f64; 3],
    dx_deg: Option<[f64; 3]>,
}

impl MountDirectionDiag {
    fn no_dx(error_deg: [f64; 3]) -> Self {
        Self {
            error_deg,
            dx_deg: None,
        }
    }

    fn with_dx(error_deg: [f64; 3], dx_deg: [f64; 3]) -> Self {
        Self {
            error_deg,
            dx_deg: Some(dx_deg),
        }
    }
}

const PVM_PHASES: [&str; 7] = [
    "propagation",
    "gnss_position",
    "gnss_velocity",
    "nhc",
    "zero_velocity",
    "remainder",
    "reset",
];

impl PvmAccountingCsv {
    fn create(
        path: &PathBuf,
        window_abs: Option<[f64; 2]>,
        mount_drive_window_s: f64,
    ) -> Result<Self> {
        let mut writer = BufWriter::new(
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?,
        );
        writeln!(
            writer,
            "rel_s,t_s,system,event,phase,obs_index,obs_type,\
residual,effective_residual,innovation_var,nis,\
pvm_vn_roll_before,pvm_vn_pitch_before,pvm_vn_yaw_before,\
pvm_ve_roll_before,pvm_ve_pitch_before,pvm_ve_yaw_before,\
pvm_vd_roll_before,pvm_vd_pitch_before,pvm_vd_yaw_before,\
pvm_vn_roll_delta,pvm_vn_pitch_delta,pvm_vn_yaw_delta,\
pvm_ve_roll_delta,pvm_ve_pitch_delta,pvm_ve_yaw_delta,\
pvm_vd_roll_delta,pvm_vd_pitch_delta,pvm_vd_yaw_delta,\
pvm_vn_roll_after,pvm_vn_pitch_after,pvm_vn_yaw_after,\
pvm_ve_roll_after,pvm_ve_pitch_after,pvm_ve_yaw_after,\
pvm_vd_roll_after,pvm_vd_pitch_after,pvm_vd_yaw_after,\
before_fro,delta_fro,after_fro,delta_abs_sum,\
mount_err_roll_deg,mount_err_pitch_deg,mount_err_yaw_deg,\
dx_mount_roll_deg,dx_mount_pitch_deg,dx_mount_yaw_deg,\
mount_correction_score_deg2,mount_correction_cos,\
delta_pvm_err_vn,delta_pvm_err_ve,delta_pvm_err_vd,\
after_pvm_err_vn,after_pvm_err_ve,after_pvm_err_vd"
        )?;
        Ok(Self {
            writer,
            window_abs,
            stats: [PvmPhaseStats::default(); PVM_PHASES.len()],
            mount_drive_window_s,
            mount_drive_windows: BTreeMap::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn record_delta(
        &mut self,
        replay: &Replay,
        system: &str,
        event: &str,
        t_s: f64,
        phase: &str,
        obs_index: Option<usize>,
        obs_type: &str,
        residual: f32,
        effective_residual: f32,
        innovation_var: f32,
        before: [[f64; 3]; 3],
        delta: [[f64; 3]; 3],
        mount_direction: MountDirectionDiag,
    ) -> Result<[[f64; 3]; 3]> {
        let after = matrix3_add(before, delta);
        if !self.in_window(t_s) {
            return Ok(after);
        }
        let rel_s = t_s - replay_start_t_s(replay).unwrap_or(0.0);
        let dx_deg = mount_direction.dx_deg.unwrap_or([f64::NAN; 3]);
        let (mount_score_deg2, mount_score_cos) =
            mount_correction_score(mount_direction.error_deg, dx_deg);
        let nis = scalar_nis(effective_residual, innovation_var);
        if let Some(phase_idx) = pvm_phase_idx(phase) {
            let stats = &mut self.stats[phase_idx];
            stats.count += 1;
            stats.activity_fro += matrix3_fro(delta);
            stats.activity_abs += matrix3_abs_sum(delta);
            for r in 0..3 {
                for c in 0..3 {
                    stats.net[r][c] += delta[r][c];
                }
            }
            if dx_deg.iter().all(|v| v.is_finite()) {
                stats.mount_score_deg2 += mount_score_deg2;
                for idx in 0..3 {
                    stats.mount_dx_sum_deg[idx] += dx_deg[idx];
                    stats.mount_dx_abs_sum_deg[idx] += dx_deg[idx].abs();
                    stats.mount_axis_score_deg2[idx] +=
                        -mount_direction.error_deg[idx] * dx_deg[idx];
                }
            }
        }
        self.record_mount_drive_window(
            system,
            rel_s,
            phase,
            mount_direction.error_deg,
            dx_deg,
            f64::from(nis),
            f64::from(effective_residual),
        );
        let obs_index = obs_index
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_owned());
        let error_rad = mount_direction.error_deg.map(f64::to_radians);
        let delta_pvm_err = matrix3_vec(delta, error_rad);
        let after_pvm_err = matrix3_vec(after, error_rad);
        writeln!(
            self.writer,
            "{rel_s:.6},{t_s:.6},{system},{event},{phase},{obs_index},{obs_type},\
{residual:.9},{effective_residual:.9},{innovation_var:.9},{nis:.9},\
{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},\
{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},\
{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},\
{:.12},{:.12},{:.12},{:.12},\
{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},\
{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            before[0][0],
            before[0][1],
            before[0][2],
            before[1][0],
            before[1][1],
            before[1][2],
            before[2][0],
            before[2][1],
            before[2][2],
            delta[0][0],
            delta[0][1],
            delta[0][2],
            delta[1][0],
            delta[1][1],
            delta[1][2],
            delta[2][0],
            delta[2][1],
            delta[2][2],
            after[0][0],
            after[0][1],
            after[0][2],
            after[1][0],
            after[1][1],
            after[1][2],
            after[2][0],
            after[2][1],
            after[2][2],
            matrix3_fro(before),
            matrix3_fro(delta),
            matrix3_fro(after),
            matrix3_abs_sum(delta),
            mount_direction.error_deg[0],
            mount_direction.error_deg[1],
            mount_direction.error_deg[2],
            dx_deg[0],
            dx_deg[1],
            dx_deg[2],
            mount_score_deg2,
            mount_score_cos,
            delta_pvm_err[0],
            delta_pvm_err[1],
            delta_pvm_err[2],
            after_pvm_err[0],
            after_pvm_err[1],
            after_pvm_err[2],
        )?;
        Ok(after)
    }

    fn print_summary(&self) {
        if self.stats.iter().all(|s| s.count == 0) {
            return;
        }
        println!("[covhist-pvm] P(velocity,mount) accounting summary");
        for (idx, phase) in PVM_PHASES.iter().enumerate() {
            let stats = self.stats[idx];
            if stats.count == 0 {
                continue;
            }
            println!(
                "  phase={phase:<13} count={} net_fro={:.6e} activity_fro={:.6e} activity_abs={:.6e}",
                stats.count,
                matrix3_fro(stats.net),
                stats.activity_fro,
                stats.activity_abs,
            );
            if stats
                .mount_dx_sum_deg
                .iter()
                .any(|v| v.is_finite() && v.abs() > 0.0)
            {
                println!(
                    "    mount_drive score_deg2={:.6e} axis_score_deg2=[{:.6e},{:.6e},{:.6e}] dx_sum_deg=[{:.6e},{:.6e},{:.6e}] dx_abs_sum_deg=[{:.6e},{:.6e},{:.6e}]",
                    stats.mount_score_deg2,
                    stats.mount_axis_score_deg2[0],
                    stats.mount_axis_score_deg2[1],
                    stats.mount_axis_score_deg2[2],
                    stats.mount_dx_sum_deg[0],
                    stats.mount_dx_sum_deg[1],
                    stats.mount_dx_sum_deg[2],
                    stats.mount_dx_abs_sum_deg[0],
                    stats.mount_dx_abs_sum_deg[1],
                    stats.mount_dx_abs_sum_deg[2],
                );
            }
            print_matrix3("    net", stats.net);
        }
        self.print_mount_drive_conflicts();
    }

    fn in_window(&self, t_s: f64) -> bool {
        self.window_abs
            .is_none_or(|[start, end]| (start..=end).contains(&t_s))
    }

    fn record_mount_drive_window(
        &mut self,
        system: &str,
        rel_s: f64,
        phase: &str,
        error_deg: [f64; 3],
        dx_deg: [f64; 3],
        nis: f64,
        effective_residual: f64,
    ) {
        if !matches!(phase, "gnss_velocity" | "nhc")
            || !rel_s.is_finite()
            || !dx_deg.iter().all(|v| v.is_finite())
            || self.mount_drive_window_s <= 0.0
        {
            return;
        }
        let system_id = match system {
            "reduced" => 0,
            "full" => 1,
            _ => return,
        };
        let window_idx = (rel_s / self.mount_drive_window_s).floor() as i64;
        let window = self
            .mount_drive_windows
            .entry((system_id, window_idx))
            .or_default();
        match phase {
            "gnss_velocity" => {
                window
                    .gnss_velocity
                    .record(error_deg, dx_deg, nis, effective_residual)
            }
            "nhc" => window
                .nhc
                .record(error_deg, dx_deg, nis, effective_residual),
            _ => {}
        }
    }

    fn print_mount_drive_conflicts(&self) {
        let mut rows = Vec::new();
        for (&(system_id, window_idx), window) in &self.mount_drive_windows {
            let gnss = window.gnss_velocity;
            let nhc = window.nhc;
            if gnss.count == 0 || nhc.count == 0 {
                continue;
            }
            let gnss_norm = vec3_norm(gnss.dx_sum_deg);
            let nhc_norm = vec3_norm(nhc.dx_sum_deg);
            let net = vec3_add(gnss.dx_sum_deg, nhc.dx_sum_deg);
            let conflict_deg = (gnss_norm + nhc_norm - vec3_norm(net)).max(0.0);
            let denom = gnss_norm + nhc_norm;
            if denom <= 0.0 {
                continue;
            }
            let conflict_ratio = conflict_deg / denom;
            rows.push((
                conflict_deg,
                conflict_ratio,
                system_id,
                window_idx,
                gnss,
                nhc,
            ));
        }
        if rows.is_empty() {
            return;
        }
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!(
            "[covhist-pvm] mount-drive conflict windows (width={:.3}s, top={})",
            self.mount_drive_window_s,
            rows.len().min(10)
        );
        for (conflict_deg, conflict_ratio, system_id, window_idx, gnss, nhc) in
            rows.into_iter().take(10)
        {
            let system = match system_id {
                0 => "reduced",
                1 => "full",
                _ => "unknown",
            };
            let start = window_idx as f64 * self.mount_drive_window_s;
            let end = start + self.mount_drive_window_s;
            println!(
                "  system={system:<7} window={start:.3}-{end:.3}s conflict_deg={conflict_deg:.6} ratio={conflict_ratio:.3} gnss_score={:.6} nhc_score={:.6}",
                gnss.score_deg2, nhc.score_deg2,
            );
            println!(
                "    gnss_nis_mean={:.3} gnss_nis_max={:.3} nhc_nis_mean={:.3} nhc_nis_max={:.3}",
                gnss.nis_sum / gnss.nis_weight.max(1.0),
                gnss.nis_max,
                nhc.nis_sum / nhc.nis_weight.max(1.0),
                nhc.nis_max,
            );
            println!(
                "    gnss_dx_sum_deg=[{:.6},{:.6},{:.6}] nhc_dx_sum_deg=[{:.6},{:.6},{:.6}]",
                gnss.dx_sum_deg[0],
                gnss.dx_sum_deg[1],
                gnss.dx_sum_deg[2],
                nhc.dx_sum_deg[0],
                nhc.dx_sum_deg[1],
                nhc.dx_sum_deg[2],
            );
            println!(
                "    gnss_axis_score=[{:.6},{:.6},{:.6}] nhc_axis_score=[{:.6},{:.6},{:.6}]",
                gnss.axis_score_deg2[0],
                gnss.axis_score_deg2[1],
                gnss.axis_score_deg2[2],
                nhc.axis_score_deg2[0],
                nhc.axis_score_deg2[1],
                nhc.axis_score_deg2[2],
            );
        }
    }
}

impl GnssParityCsv {
    fn create(path: &PathBuf, window_abs: Option<[f64; 2]>) -> Result<Self> {
        let mut writer = BufWriter::new(
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?,
        );
        writeln!(
            writer,
            "rel_s,t_s,system,event,obs_index,update_type,\
residual,effective_residual,innovation_var,nis,\
k_att_roll,k_att_pitch,k_att_yaw,k_mount_roll,k_mount_pitch,k_mount_yaw,\
dx_att_roll_deg,dx_att_pitch_deg,dx_att_yaw_deg,\
dx_vel_n_mps,dx_vel_e_mps,dx_vel_d_mps,\
dx_mount_roll_deg,dx_mount_pitch_deg,dx_mount_yaw_deg,\
mount_roll_sigma_deg,mount_pitch_sigma_deg,mount_yaw_sigma_deg,\
att_roll_sigma_deg,att_pitch_sigma_deg,att_yaw_sigma_deg"
        )?;
        Ok(Self { writer, window_abs })
    }

    fn record_reduced(
        &mut self,
        replay: &Replay,
        event: &str,
        t_s: f64,
        reduced: Option<&reduced::State>,
        full_ready: bool,
    ) -> Result<()> {
        let Some(reduced) = reduced else {
            return Ok(());
        };
        if !full_ready || !self.in_window(t_s) {
            return Ok(());
        }
        let count = reduced
            .last_obs_count
            .clamp(0, reduced.last_obs_types.len() as i32) as usize;
        if count == 0 {
            return Ok(());
        }
        let rel_s = t_s - replay_start_t_s(replay).unwrap_or(0.0);
        let mut pos_xy_count = 0usize;
        let mut vel_xy_count = 0usize;
        for obs_idx in 0..count {
            let ty = reduced.last_obs_types[obs_idx] as usize;
            let label = match reduced_gnss_row_label(ty, &mut pos_xy_count, &mut vel_xy_count) {
                Some(label) => label,
                None => continue,
            };
            let residual = reduced.last_residuals[obs_idx];
            let effective_residual = reduced.last_effective_residuals[obs_idx];
            let innovation_var = reduced.last_innovation_vars[obs_idx];
            let nis = scalar_nis(effective_residual, innovation_var);
            let k = &reduced.last_k_by_obs[obs_idx];
            let dx = &reduced.last_dx_by_obs[obs_idx];
            self.write_row(
                rel_s,
                t_s,
                "reduced",
                event,
                obs_idx,
                label,
                residual,
                effective_residual,
                innovation_var,
                nis,
                [k[0], k[1], k[2], k[15], k[16], k[17]],
                [
                    dx[0], dx[1], dx[2], dx[3], dx[4], dx[5], dx[15], dx[16], dx[17],
                ],
                &reduced.p,
            )?;
        }
        Ok(())
    }

    fn record_full(
        &mut self,
        replay: &Replay,
        event: &str,
        t_s: f64,
        full: &full::Filter,
        ref_gnss: GenericGnssSample,
    ) -> Result<()> {
        if !self.in_window(t_s) || full.last_obs_types().is_empty() {
            return Ok(());
        }
        let rel_s = t_s - replay_start_t_s(replay).unwrap_or(0.0);
        let full_snap = FullSnapshot {
            nominal: *full.nominal(),
            p: *full.covariance(),
            pos_ecef: full.shadow_pos_ecef(),
            last_dx: *full.last_dx(),
            last_obs_types: full.last_obs_types().to_vec(),
        };
        let p_as_reduced = transform_full_cov_to_reduced(&full_snap, ref_gnss);
        for (obs_idx, ty) in full.last_obs_types().iter().copied().enumerate() {
            let ty = ty as usize;
            if !matches!(ty, 1..=6) {
                continue;
            }
            let row_dx = &full.last_dx_by_obs()[obs_idx];
            let dx_as_reduced = transform_full_dx_to_reduced(&full_snap, row_dx, ref_gnss);
            let effective_residual = full.last_effective_residuals()[obs_idx];
            let residual = full.last_residuals()[obs_idx];
            let innovation_var = full.last_innovation_vars()[obs_idx];
            let nis = scalar_nis(effective_residual, innovation_var);
            let k_as_reduced = if effective_residual.abs() > 1.0e-9 {
                let mut k_full = [0.0f32; ERROR_STATES];
                for i in 0..ERROR_STATES {
                    k_full[i] = row_dx[i] / effective_residual;
                }
                transform_full_dx_to_reduced(&full_snap, &k_full, ref_gnss)
            } else {
                [0.0; 18]
            };
            self.write_row(
                rel_s,
                t_s,
                "full",
                event,
                obs_idx,
                full_obs_type_label(ty),
                residual,
                effective_residual,
                innovation_var,
                nis,
                [
                    k_as_reduced[0],
                    k_as_reduced[1],
                    k_as_reduced[2],
                    k_as_reduced[15],
                    k_as_reduced[16],
                    k_as_reduced[17],
                ],
                [
                    dx_as_reduced[0],
                    dx_as_reduced[1],
                    dx_as_reduced[2],
                    dx_as_reduced[3],
                    dx_as_reduced[4],
                    dx_as_reduced[5],
                    dx_as_reduced[15],
                    dx_as_reduced[16],
                    dx_as_reduced[17],
                ],
                &p_as_reduced,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_row(
        &mut self,
        rel_s: f64,
        t_s: f64,
        system: &str,
        event: &str,
        obs_idx: usize,
        label: &str,
        residual: f32,
        effective_residual: f32,
        innovation_var: f32,
        nis: f32,
        k: [f32; 6],
        dx: [f32; 9],
        p: &[[f32; 18]; 18],
    ) -> Result<()> {
        writeln!(
            self.writer,
            "{rel_s:.6},{t_s:.6},{system},{event},{obs_idx},{label},\
{residual:.9},{effective_residual:.9},{innovation_var:.9},{nis:.9},\
{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9}",
            k[0],
            k[1],
            k[2],
            k[3],
            k[4],
            k[5],
            rad_f32_to_deg(dx[0]),
            rad_f32_to_deg(dx[1]),
            rad_f32_to_deg(dx[2]),
            dx[3],
            dx[4],
            dx[5],
            rad_f32_to_deg(dx[6]),
            rad_f32_to_deg(dx[7]),
            rad_f32_to_deg(dx[8]),
            sigma_deg(p, 15),
            sigma_deg(p, 16),
            sigma_deg(p, 17),
            sigma_deg(p, 0),
            sigma_deg(p, 1),
            sigma_deg(p, 2),
        )?;
        Ok(())
    }

    fn in_window(&self, t_s: f64) -> bool {
        self.window_abs
            .is_none_or(|[start, end]| (start..=end).contains(&t_s))
    }
}

impl RowBasisCsv {
    fn create(path: &PathBuf, window_abs: Option<[f64; 2]>) -> Result<Self> {
        let mut writer = BufWriter::new(
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?,
        );
        writeln!(
            writer,
            "rel_s,t_s,system,event,obs_index,update_type,\
residual,effective_residual,innovation_var,nis,\
h_att_roll,h_att_pitch,h_att_yaw,h_vel_n,h_vel_e,h_vel_d,h_pos_n,h_pos_e,h_pos_d,\
h_gyro_bias_x,h_gyro_bias_y,h_gyro_bias_z,h_accel_bias_x,h_accel_bias_y,h_accel_bias_z,\
h_mount_roll,h_mount_pitch,h_mount_yaw,\
k_att_roll,k_att_pitch,k_att_yaw,k_vel_n,k_vel_e,k_vel_d,k_pos_n,k_pos_e,k_pos_d,\
k_gyro_bias_x,k_gyro_bias_y,k_gyro_bias_z,k_accel_bias_x,k_accel_bias_y,k_accel_bias_z,\
k_mount_roll,k_mount_pitch,k_mount_yaw,\
dx_att_roll_deg,dx_att_pitch_deg,dx_att_yaw_deg,dx_vel_n_mps,dx_vel_e_mps,dx_vel_d_mps,\
dx_pos_n_m,dx_pos_e_m,dx_pos_d_m,dx_gyro_bias_x_dps,dx_gyro_bias_y_dps,dx_gyro_bias_z_dps,\
dx_accel_bias_x_mps2,dx_accel_bias_y_mps2,dx_accel_bias_z_mps2,\
dx_mount_roll_deg,dx_mount_pitch_deg,dx_mount_yaw_deg,\
p_att_trace,p_vel_trace,p_pos_trace,p_gyro_bias_trace,p_accel_bias_trace,p_mount_trace,\
p_att_mount_fro,p_vel_mount_fro,p_pos_mount_fro,p_gyro_bias_mount_fro,p_accel_bias_mount_fro,\
corr_att_roll_mount_roll,corr_att_pitch_mount_pitch,corr_att_yaw_mount_yaw,\
mount_roll_sigma_deg,mount_pitch_sigma_deg,mount_yaw_sigma_deg,\
att_roll_sigma_deg,att_pitch_sigma_deg,att_yaw_sigma_deg"
        )?;
        Ok(Self { writer, window_abs })
    }

    fn record_reduced(
        &mut self,
        replay: &Replay,
        event: &str,
        t_s: f64,
        reduced: Option<&reduced::State>,
        full_ready: bool,
    ) -> Result<()> {
        let Some(reduced) = reduced else {
            return Ok(());
        };
        if !full_ready || !self.in_window(t_s) {
            return Ok(());
        }
        let rel_s = t_s - replay_start_t_s(replay).unwrap_or(0.0);
        let count = reduced
            .last_obs_count
            .clamp(0, reduced.last_obs_types.len() as i32) as usize;
        let mut pos_xy_count = 0usize;
        let mut vel_xy_count = 0usize;
        let mut zero_xy_count = 0usize;
        for obs_idx in 0..count {
            let ty = reduced.last_obs_types[obs_idx] as usize;
            let label =
                reduced_row_label(ty, &mut pos_xy_count, &mut vel_xy_count, &mut zero_xy_count);
            self.write_row(
                rel_s,
                t_s,
                "reduced",
                event,
                obs_idx,
                label,
                reduced.last_residuals[obs_idx],
                reduced.last_effective_residuals[obs_idx],
                reduced.last_innovation_vars[obs_idx],
                scalar_nis(
                    reduced.last_effective_residuals[obs_idx],
                    reduced.last_innovation_vars[obs_idx],
                ),
                &reduced.last_h_by_obs[obs_idx],
                &reduced.last_k_by_obs[obs_idx],
                &reduced.last_dx_by_obs[obs_idx],
                &reduced.p,
            )?;
        }
        Ok(())
    }

    fn record_full(
        &mut self,
        replay: &Replay,
        event: &str,
        t_s: f64,
        full: &full::Filter,
        ref_gnss: GenericGnssSample,
    ) -> Result<()> {
        if !self.in_window(t_s) || full.last_obs_types().is_empty() {
            return Ok(());
        }
        let rel_s = t_s - replay_start_t_s(replay).unwrap_or(0.0);
        let full_snap = FullSnapshot {
            nominal: *full.nominal(),
            p: *full.covariance(),
            pos_ecef: full.shadow_pos_ecef(),
            last_dx: *full.last_dx(),
            last_obs_types: full.last_obs_types().to_vec(),
        };
        let p_as_reduced = transform_full_cov_to_reduced(&full_snap, ref_gnss);
        for (obs_idx, ty) in full.last_obs_types().iter().copied().enumerate() {
            let h_as_reduced =
                transform_full_h_to_reduced(&full_snap, &full.last_h_by_obs()[obs_idx], ref_gnss);
            let k_as_reduced =
                transform_full_dx_to_reduced(&full_snap, &full.last_k_by_obs()[obs_idx], ref_gnss);
            let dx_as_reduced =
                transform_full_dx_to_reduced(&full_snap, &full.last_dx_by_obs()[obs_idx], ref_gnss);
            let effective_residual = full.last_effective_residuals()[obs_idx];
            let innovation_var = full.last_innovation_vars()[obs_idx];
            self.write_row(
                rel_s,
                t_s,
                "full",
                event,
                obs_idx,
                full_obs_type_label(ty as usize),
                full.last_residuals()[obs_idx],
                effective_residual,
                innovation_var,
                scalar_nis(effective_residual, innovation_var),
                &h_as_reduced,
                &k_as_reduced,
                &dx_as_reduced,
                &p_as_reduced,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_row(
        &mut self,
        rel_s: f64,
        t_s: f64,
        system: &str,
        event: &str,
        obs_idx: usize,
        label: &str,
        residual: f32,
        effective_residual: f32,
        innovation_var: f32,
        nis: f32,
        h: &[f32; 18],
        k: &[f32; 18],
        dx: &[f32; 18],
        p: &[[f32; 18]; 18],
    ) -> Result<()> {
        let mut values = Vec::with_capacity(75);
        values.extend(h.iter().map(|v| format!("{v:.9}")));
        values.extend(k.iter().map(|v| format!("{v:.9}")));
        values.extend([
            format!("{:.9}", rad_f32_to_deg(dx[0])),
            format!("{:.9}", rad_f32_to_deg(dx[1])),
            format!("{:.9}", rad_f32_to_deg(dx[2])),
            format!("{:.9}", dx[3]),
            format!("{:.9}", dx[4]),
            format!("{:.9}", dx[5]),
            format!("{:.9}", dx[6]),
            format!("{:.9}", dx[7]),
            format!("{:.9}", dx[8]),
            format!("{:.9}", rad_f32_to_dps(dx[9])),
            format!("{:.9}", rad_f32_to_dps(dx[10])),
            format!("{:.9}", rad_f32_to_dps(dx[11])),
            format!("{:.9}", dx[12]),
            format!("{:.9}", dx[13]),
            format!("{:.9}", dx[14]),
            format!("{:.9}", rad_f32_to_deg(dx[15])),
            format!("{:.9}", rad_f32_to_deg(dx[16])),
            format!("{:.9}", rad_f32_to_deg(dx[17])),
            format!("{:.9}", block_trace(p, 0)),
            format!("{:.9}", block_trace(p, 3)),
            format!("{:.9}", block_trace(p, 6)),
            format!("{:.9}", block_trace(p, 9)),
            format!("{:.9}", block_trace(p, 12)),
            format!("{:.9}", block_trace(p, 15)),
            format!("{:.9}", block_fro(p, 0, 15)),
            format!("{:.9}", block_fro(p, 3, 15)),
            format!("{:.9}", block_fro(p, 6, 15)),
            format!("{:.9}", block_fro(p, 9, 15)),
            format!("{:.9}", block_fro(p, 12, 15)),
            format!("{:.9}", corr_from_cov(p, 0, 15)),
            format!("{:.9}", corr_from_cov(p, 1, 16)),
            format!("{:.9}", corr_from_cov(p, 2, 17)),
            format!("{:.9}", sigma_deg(p, 15)),
            format!("{:.9}", sigma_deg(p, 16)),
            format!("{:.9}", sigma_deg(p, 17)),
            format!("{:.9}", sigma_deg(p, 0)),
            format!("{:.9}", sigma_deg(p, 1)),
            format!("{:.9}", sigma_deg(p, 2)),
        ]);
        writeln!(
            self.writer,
            "{rel_s:.6},{t_s:.6},{system},{event},{obs_idx},{label},\
{residual:.9},{effective_residual:.9},{innovation_var:.9},{nis:.9},{}",
            values.join(","),
        )?;
        Ok(())
    }

    fn in_window(&self, t_s: f64) -> bool {
        self.window_abs
            .is_none_or(|[start, end]| (start..=end).contains(&t_s))
    }
}

fn record_reduced_pvm_change(
    csv: &mut PvmAccountingCsv,
    replay: &Replay,
    event: &str,
    t_s: f64,
    before: &reduced::State,
    after: &reduced::State,
) -> Result<()> {
    let before_pvm = pvm_block_from_common(&before.p);
    let after_pvm = pvm_block_from_common(&after.p);
    let mount_error_deg = reduced_mount_error_deg(replay, before);
    let obs_count = after
        .last_obs_count
        .clamp(0, after.last_obs_types.len() as i32) as usize;
    if obs_count == 0 {
        return csv
            .record_delta(
                replay,
                "reduced",
                event,
                t_s,
                "propagation",
                None,
                "predict",
                f32::NAN,
                f32::NAN,
                f32::NAN,
                before_pvm,
                matrix3_sub(after_pvm, before_pvm),
                MountDirectionDiag::no_dx(mount_error_deg),
            )
            .map(|_| ());
    }

    let mut running = before_pvm;
    let mut row_sum = [[0.0f64; 3]; 3];
    let mut pos_xy_count = 0usize;
    let mut vel_xy_count = 0usize;
    let mut zero_xy_count = 0usize;
    for obs_idx in 0..obs_count {
        let ty = after.last_obs_types[obs_idx] as usize;
        let innovation_var = after.last_innovation_vars[obs_idx];
        if !innovation_var.is_finite() || innovation_var <= 0.0 {
            continue;
        }
        let native_delta =
            reduced_covariance_row_delta(&after.last_k_by_obs[obs_idx], innovation_var);
        let delta = pvm_block_from_common(&native_delta);
        row_sum = matrix3_add(row_sum, delta);
        let label = reduced_row_label(ty, &mut pos_xy_count, &mut vel_xy_count, &mut zero_xy_count);
        let phase = reduced_pvm_phase_for_obs(ty);
        running = csv.record_delta(
            replay,
            "reduced",
            event,
            t_s,
            phase,
            Some(obs_idx),
            label,
            after.last_residuals[obs_idx],
            after.last_effective_residuals[obs_idx],
            innovation_var,
            running,
            delta,
            MountDirectionDiag::with_dx(
                mount_error_deg,
                [
                    rad_f32_to_deg(after.last_dx_by_obs[obs_idx][15]),
                    rad_f32_to_deg(after.last_dx_by_obs[obs_idx][16]),
                    rad_f32_to_deg(after.last_dx_by_obs[obs_idx][17]),
                ],
            ),
        )?;
    }
    let remainder_delta = matrix3_sub(after_pvm, matrix3_add(before_pvm, row_sum));
    csv.record_delta(
        replay,
        "reduced",
        event,
        t_s,
        "remainder",
        None,
        "predict_reset",
        f32::NAN,
        f32::NAN,
        f32::NAN,
        running,
        remainder_delta,
        MountDirectionDiag::no_dx(mount_error_deg),
    )
    .map(|_| ())
}

fn record_reduced_mount_drive_adaptive(
    replay: &Replay,
    t_s: f64,
    before: &reduced::State,
    after: &reduced::State,
    adaptive: &mut AdaptiveGnssVelocityScale,
) {
    let obs_count = after
        .last_obs_count
        .clamp(0, after.last_obs_types.len() as i32) as usize;
    if obs_count == 0 {
        return;
    }
    let mount_error_deg = reduced_mount_error_deg(replay, before);
    let rel_s = t_s - replay_start_t_s(replay).unwrap_or(0.0);
    for obs_idx in 0..obs_count {
        let ty = after.last_obs_types[obs_idx] as usize;
        let phase = reduced_pvm_phase_for_obs(ty);
        if !matches!(phase, "gnss_velocity" | "nhc") {
            continue;
        }
        let innovation_var = after.last_innovation_vars[obs_idx];
        if !innovation_var.is_finite() || innovation_var <= 0.0 {
            continue;
        }
        adaptive.record(
            rel_s,
            phase,
            mount_error_deg,
            [
                rad_f32_to_deg(after.last_dx_by_obs[obs_idx][15]),
                rad_f32_to_deg(after.last_dx_by_obs[obs_idx][16]),
                rad_f32_to_deg(after.last_dx_by_obs[obs_idx][17]),
            ],
            f64::from(scalar_nis(
                after.last_effective_residuals[obs_idx],
                innovation_var,
            )),
            f64::from(after.last_effective_residuals[obs_idx]),
        );
    }
}

impl AllocationCsv {
    fn create(path: &PathBuf, window_abs: Option<[f64; 2]>) -> Result<Self> {
        let mut writer = BufWriter::new(
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?,
        );
        writeln!(
            writer,
            "rel_s,t_s,system,event,update_type,count_delta,\
innovation_sum,innovation_abs_sum,nis_sum,h_mount_norm_sum,k_mount_norm_sum,\
dx_att_roll_deg,dx_att_pitch_deg,dx_att_yaw_deg,\
dx_vel_n_mps,dx_vel_e_mps,dx_vel_d_mps,\
dx_mount_roll_deg,dx_mount_pitch_deg,dx_mount_yaw_deg,\
dx_gyro_bias_x_dps,dx_gyro_bias_y_dps,dx_gyro_bias_z_dps,\
dx_accel_bias_x_mps2,dx_accel_bias_y_mps2,dx_accel_bias_z_mps2,\
mount_roll_sigma_deg,mount_pitch_sigma_deg,mount_yaw_sigma_deg,\
att_roll_sigma_deg,att_pitch_sigma_deg,att_yaw_sigma_deg,\
mount_qerr_deg,att_qerr_deg"
        )?;
        Ok(Self {
            writer,
            window_abs,
            prev_reduced_counts: [0; UPDATE_DIAG_TYPES],
            prev_sum_dx_att_roll: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_dx_pitch: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_dx_yaw: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_dx_vel_n: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_dx_vel_e: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_dx_vel_d: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_dx_mount_roll: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_dx_mount_pitch: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_dx_mount_yaw: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_dx_gyro_bias: [[0.0; 3]; UPDATE_DIAG_TYPES],
            prev_sum_dx_accel_bias: [[0.0; 3]; UPDATE_DIAG_TYPES],
            prev_sum_innovation: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_abs_innovation: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_nis: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_h_mount_norm: [0.0; UPDATE_DIAG_TYPES],
            prev_sum_k_mount_norm: [0.0; UPDATE_DIAG_TYPES],
            initialized: false,
        })
    }

    fn record_reduced(
        &mut self,
        replay: &Replay,
        event: &str,
        t_s: f64,
        reduced: Option<&reduced::State>,
        full_ready: bool,
        ref_gnss: GenericGnssSample,
    ) -> Result<()> {
        let Some(reduced) = reduced else {
            return Ok(());
        };
        if !self.initialized {
            self.capture_reduced_baseline(reduced);
            self.initialized = true;
            return Ok(());
        }

        let in_window = self.in_window(t_s) && full_ready;
        for (label, idx) in selected_reduced_diag_types() {
            let count_delta = reduced.update_diag.type_counts[idx] - self.prev_reduced_counts[idx];
            if count_delta == 0 {
                continue;
            }
            if in_window {
                let rel_s = t_s - replay_start_t_s(replay).unwrap_or(0.0);
                let mount_qerr = reference_mount_at(&replay.reference_mount, t_s)
                    .map(|r| {
                        quat_angle_deg(
                            as_q64([
                                reduced.nominal.q_bv0,
                                reduced.nominal.q_bv1,
                                reduced.nominal.q_bv2,
                                reduced.nominal.q_bv3,
                            ]),
                            r,
                        )
                    })
                    .unwrap_or(f64::NAN);
                let att_qerr = reference_attitude_at(&replay.reference_attitude, t_s)
                    .map(|r| quat_angle_deg(reduced_att_q(reduced), r))
                    .unwrap_or(f64::NAN);
                let dg = delta3(
                    reduced.update_diag.sum_dx_gyro_bias[idx],
                    self.prev_sum_dx_gyro_bias[idx],
                );
                let da = delta3(
                    reduced.update_diag.sum_dx_accel_bias[idx],
                    self.prev_sum_dx_accel_bias[idx],
                );
                writeln!(
                    self.writer,
                    "{rel_s:.6},{t_s:.6},reduced,{event},{label},{count_delta},\
{:.9},{:.9},{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{mount_qerr:.9},{att_qerr:.9}",
                    reduced.update_diag.sum_innovation[idx] - self.prev_sum_innovation[idx],
                    reduced.update_diag.sum_abs_innovation[idx] - self.prev_sum_abs_innovation[idx],
                    reduced.update_diag.sum_nis[idx] - self.prev_sum_nis[idx],
                    reduced.update_diag.sum_h_mount_norm[idx] - self.prev_sum_h_mount_norm[idx],
                    reduced.update_diag.sum_k_mount_norm[idx] - self.prev_sum_k_mount_norm[idx],
                    rad_f32_to_deg(
                        reduced.update_diag.sum_dx_att_roll[idx] - self.prev_sum_dx_att_roll[idx]
                    ),
                    rad_f32_to_deg(
                        reduced.update_diag.sum_dx_pitch[idx] - self.prev_sum_dx_pitch[idx]
                    ),
                    rad_f32_to_deg(reduced.update_diag.sum_dx_yaw[idx] - self.prev_sum_dx_yaw[idx]),
                    reduced.update_diag.sum_dx_vel_n[idx] - self.prev_sum_dx_vel_n[idx],
                    reduced.update_diag.sum_dx_vel_e[idx] - self.prev_sum_dx_vel_e[idx],
                    reduced.update_diag.sum_dx_vel_d[idx] - self.prev_sum_dx_vel_d[idx],
                    rad_f32_to_deg(
                        reduced.update_diag.sum_dx_mount_roll[idx]
                            - self.prev_sum_dx_mount_roll[idx]
                    ),
                    rad_f32_to_deg(
                        reduced.update_diag.sum_dx_mount_pitch[idx]
                            - self.prev_sum_dx_mount_pitch[idx]
                    ),
                    rad_f32_to_deg(
                        reduced.update_diag.sum_dx_mount_yaw[idx] - self.prev_sum_dx_mount_yaw[idx]
                    ),
                    rad_f32_to_dps(dg[0]),
                    rad_f32_to_dps(dg[1]),
                    rad_f32_to_dps(dg[2]),
                    da[0],
                    da[1],
                    da[2],
                    sigma_deg(&reduced.p, 15),
                    sigma_deg(&reduced.p, 16),
                    sigma_deg(&reduced.p, 17),
                    sigma_deg(&reduced.p, 0),
                    sigma_deg(&reduced.p, 1),
                    sigma_deg(&reduced.p, 2),
                )?;
            }
        }
        self.capture_reduced_baseline(reduced);
        let _ = ref_gnss;
        Ok(())
    }

    fn record_full(
        &mut self,
        replay: &Replay,
        t_s: f64,
        full: &full::Filter,
        ref_gnss: GenericGnssSample,
    ) -> Result<()> {
        if !self.in_window(t_s) || full.last_obs_types().is_empty() {
            return Ok(());
        }
        let rel_s = t_s - replay_start_t_s(replay).unwrap_or(0.0);
        let full_snap = FullSnapshot {
            nominal: *full.nominal(),
            p: *full.covariance(),
            pos_ecef: full.shadow_pos_ecef(),
            last_dx: *full.last_dx(),
            last_obs_types: full.last_obs_types().to_vec(),
        };
        let p_as_reduced = transform_full_cov_to_reduced(&full_snap, ref_gnss);
        let mount_qerr = reference_mount_at(&replay.reference_mount, t_s)
            .map(|r| {
                quat_angle_deg(
                    as_q64([
                        full_snap.nominal.q_bv0,
                        full_snap.nominal.q_bv1,
                        full_snap.nominal.q_bv2,
                        full_snap.nominal.q_bv3,
                    ]),
                    r,
                )
            })
            .unwrap_or(f64::NAN);
        let att_qerr = reference_attitude_at(&replay.reference_attitude, t_s)
            .map(|r| quat_angle_deg(full_att_q_ned(&full_snap, ref_gnss), r))
            .unwrap_or(f64::NAN);

        for (obs_idx, ty) in full.last_obs_types().iter().copied().enumerate() {
            let ty = ty as usize;
            let row_dx = &full.last_dx_by_obs()[obs_idx];
            let dx_as_reduced = transform_full_dx_to_reduced(&full_snap, row_dx, ref_gnss);
            let effective_residual = full.last_effective_residuals()[obs_idx];
            let innovation_var = full.last_innovation_vars()[obs_idx];
            let nis = if innovation_var > 0.0 {
                effective_residual * effective_residual / innovation_var
            } else {
                f32::NAN
            };
            let k_mount_norm = if effective_residual.abs() > 1.0e-9 {
                let kx = row_dx[21] / effective_residual;
                let ky = row_dx[22] / effective_residual;
                let kz = row_dx[23] / effective_residual;
                (kx * kx + ky * ky + kz * kz).sqrt()
            } else {
                0.0
            };
            writeln!(
                self.writer,
                "{rel_s:.6},{t_s:.6},full,imu,{},1,\
	{:.9},{:.9},{:.9},0.000000000,{:.9},\
	{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{:.9},{:.9},{:.9},\
{mount_qerr:.9},{att_qerr:.9}",
                full_obs_type_label(ty),
                effective_residual,
                effective_residual.abs(),
                nis,
                k_mount_norm,
                rad_f32_to_deg(dx_as_reduced[0]),
                rad_f32_to_deg(dx_as_reduced[1]),
                rad_f32_to_deg(dx_as_reduced[2]),
                dx_as_reduced[3],
                dx_as_reduced[4],
                dx_as_reduced[5],
                rad_f32_to_deg(row_dx[21]),
                rad_f32_to_deg(row_dx[22]),
                rad_f32_to_deg(row_dx[23]),
                -rad_f32_to_dps(row_dx[12]),
                -rad_f32_to_dps(row_dx[13]),
                -rad_f32_to_dps(row_dx[14]),
                -row_dx[9],
                -row_dx[10],
                -row_dx[11],
                sigma_deg(&p_as_reduced, 15),
                sigma_deg(&p_as_reduced, 16),
                sigma_deg(&p_as_reduced, 17),
                sigma_deg(&p_as_reduced, 0),
                sigma_deg(&p_as_reduced, 1),
                sigma_deg(&p_as_reduced, 2),
            )?;
        }
        Ok(())
    }

    fn in_window(&self, t_s: f64) -> bool {
        self.window_abs
            .is_none_or(|[start, end]| (start..=end).contains(&t_s))
    }

    fn capture_reduced_baseline(&mut self, reduced: &reduced::State) {
        self.prev_reduced_counts = reduced.update_diag.type_counts;
        self.prev_sum_dx_att_roll = reduced.update_diag.sum_dx_att_roll;
        self.prev_sum_dx_pitch = reduced.update_diag.sum_dx_pitch;
        self.prev_sum_dx_yaw = reduced.update_diag.sum_dx_yaw;
        self.prev_sum_dx_vel_n = reduced.update_diag.sum_dx_vel_n;
        self.prev_sum_dx_vel_e = reduced.update_diag.sum_dx_vel_e;
        self.prev_sum_dx_vel_d = reduced.update_diag.sum_dx_vel_d;
        self.prev_sum_dx_mount_roll = reduced.update_diag.sum_dx_mount_roll;
        self.prev_sum_dx_mount_pitch = reduced.update_diag.sum_dx_mount_pitch;
        self.prev_sum_dx_mount_yaw = reduced.update_diag.sum_dx_mount_yaw;
        self.prev_sum_dx_gyro_bias = reduced.update_diag.sum_dx_gyro_bias;
        self.prev_sum_dx_accel_bias = reduced.update_diag.sum_dx_accel_bias;
        self.prev_sum_innovation = reduced.update_diag.sum_innovation;
        self.prev_sum_abs_innovation = reduced.update_diag.sum_abs_innovation;
        self.prev_sum_nis = reduced.update_diag.sum_nis;
        self.prev_sum_h_mount_norm = reduced.update_diag.sum_h_mount_norm;
        self.prev_sum_k_mount_norm = reduced.update_diag.sum_k_mount_norm;
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(window) = args.trace_window.as_ref()
        && window.len() != 2
    {
        anyhow::bail!("--trace-window expects exactly two values: start,end");
    }
    if let Some(window) = args.summary_window.as_ref()
        && window.len() != 2
    {
        anyhow::bail!("--summary-window expects exactly two values: start,end");
    }
    if let Some(window) = args.gnss_parity_window.as_ref()
        && window.len() != 2
    {
        anyhow::bail!("--gnss-parity-window expects exactly two values: start,end");
    }
    if let Some(window) = args.row_basis_window.as_ref()
        && window.len() != 2
    {
        anyhow::bail!("--row-basis-window expects exactly two values: start,end");
    }
    if let Some(window) = args.pvm_accounting_window.as_ref()
        && window.len() != 2
    {
        anyhow::bail!("--pvm-accounting-window expects exactly two values: start,end");
    }
    let mut replay = load_replay_from_args(&args)?;
    sort_replay(&mut replay);
    let Some(t0) = replay_start_t_s(&replay) else {
        anyhow::bail!("empty replay");
    };
    if let Some(max_time_s) = args.max_time_s {
        truncate_replay_after(&mut replay, t0 + max_time_s);
    }
    let mut targets_abs: Vec<f64> = args.times.iter().map(|t| t0 + *t).collect();
    if let Some(window) = args.summary_window.as_ref() {
        targets_abs.push(t0 + window[0]);
        targets_abs.push(t0 + window[1]);
    }
    targets_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    targets_abs.dedup_by(|a, b| (*a - *b).abs() < 1.0e-9);
    let trace_window_abs = args.trace_window.as_ref().map(|w| [t0 + w[0], t0 + w[1]]);
    let allocation_window_abs = args.summary_window.as_ref().map(|w| [t0 + w[0], t0 + w[1]]);
    let gnss_parity_window_abs = args
        .gnss_parity_window
        .as_ref()
        .or(args.summary_window.as_ref())
        .map(|w| [t0 + w[0], t0 + w[1]]);
    let row_basis_window_abs = args
        .row_basis_window
        .as_ref()
        .or(args.summary_window.as_ref())
        .map(|w| [t0 + w[0], t0 + w[1]]);
    let pvm_accounting_window_abs = args
        .pvm_accounting_window
        .as_ref()
        .or(args.summary_window.as_ref())
        .map(|w| [t0 + w[0], t0 + w[1]]);
    let mut allocation_csv = match &args.allocation_csv {
        Some(path) => Some(AllocationCsv::create(path, allocation_window_abs)?),
        None => None,
    };
    let mut gnss_parity_csv = match &args.gnss_parity_csv {
        Some(path) => Some(GnssParityCsv::create(path, gnss_parity_window_abs)?),
        None => None,
    };
    let mut row_basis_csv = match &args.row_basis_csv {
        Some(path) => Some(RowBasisCsv::create(path, row_basis_window_abs)?),
        None => None,
    };
    let mut pvm_accounting_csv = match &args.pvm_accounting_csv {
        Some(path) => Some(PvmAccountingCsv::create(
            path,
            pvm_accounting_window_abs,
            args.mount_drive_window_s,
        )?),
        None => None,
    };
    let snapshots = run_diagnostics(
        &replay,
        &args,
        &targets_abs,
        trace_window_abs,
        allocation_csv.as_mut(),
        gnss_parity_csv.as_mut(),
        row_basis_csv.as_mut(),
        pvm_accounting_csv.as_mut(),
    )?;
    if let Some(csv) = pvm_accounting_csv.as_ref() {
        csv.print_summary();
    }
    print_snapshots(&snapshots, &replay);
    if let Some(window) = args.summary_window.as_ref() {
        print_interval_summary(&snapshots, &replay, [window[0], window[1]]);
    }
    Ok(())
}

fn load_replay_from_args(args: &Args) -> Result<Replay> {
    match (&args.generic_replay_dir, &args.synthetic_motion_def) {
        (Some(_), Some(_)) => {
            anyhow::bail!("choose either --generic-replay-dir or --synthetic-motion-def")
        }
        (Some(dir), None) => load_replay(dir),
        (None, Some(motion_def)) => load_synthetic_replay(args, motion_def),
        (None, None) => {
            anyhow::bail!("provide either --generic-replay-dir or --synthetic-motion-def")
        }
    }
}

fn load_replay(dir: &PathBuf) -> Result<Replay> {
    Ok(Replay {
        imu: load_imu_samples(dir)
            .with_context(|| format!("failed to load {}", dir.join("imu.csv").display()))?,
        gnss: load_gnss_samples(dir)
            .with_context(|| format!("failed to load {}", dir.join("gnss.csv").display()))?,
        reference_attitude: load_reference_attitude_samples(dir)?,
        reference_mount: load_reference_mount_samples(dir)?,
    })
}

fn load_synthetic_replay(args: &Args, motion_def: &PathBuf) -> Result<Replay> {
    let synth_cfg = SyntheticVisualizerConfig {
        motion_def: Some(motion_def.clone()),
        motion_label: motion_def.display().to_string(),
        motion_text: None,
        noise_mode: args.synthetic_noise.into(),
        disable_imu_noise: false,
        disable_gnss_noise: false,
        seed: args.synthetic_seed,
        mount_rpy_deg: [
            args.synthetic_mount_roll_deg,
            args.synthetic_mount_pitch_deg,
            args.synthetic_mount_yaw_deg,
        ],
        imu_hz: args.synthetic_imu_hz,
        gnss_hz: args.synthetic_gnss_hz,
        gnss_time_shift_ms: 0.0,
        early_vel_bias_ned_mps: [0.0; 3],
        early_fault_window_s: None,
    };
    let (replay, _) = build_synthetic_replay_input(&synth_cfg)
        .with_context(|| format!("failed to generate {}", motion_def.display()))?;
    Ok(Replay {
        imu: replay.imu,
        gnss: replay.gnss,
        reference_attitude: replay.reference_attitude,
        reference_mount: replay.reference_mount,
    })
}

fn scaled_fusion_gnss_sample(
    sample: GenericGnssSample,
    pos_std_scale: f64,
    pos_d_std_scale: f64,
    vel_std_scale: f64,
) -> sensor_fusion::GnssSample {
    let mut out = fusion_gnss_sample(sample);
    if pos_std_scale.is_finite() && pos_std_scale > 0.0 {
        let scale = pos_std_scale as f32;
        for axis in 0..3 {
            out.pos_std_m[axis] *= scale;
        }
    }
    if pos_d_std_scale.is_finite() && pos_d_std_scale > 0.0 {
        out.pos_std_m[2] *= pos_d_std_scale as f32;
    }
    if vel_std_scale.is_finite() && vel_std_scale > 0.0 {
        let scale = vel_std_scale as f32;
        for axis in 0..3 {
            out.vel_std_mps[axis] *= scale;
        }
    }
    out
}

fn sort_replay(replay: &mut Replay) {
    replay.imu.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    replay.gnss.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    replay.reference_attitude.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    replay.reference_mount.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn truncate_replay_after(replay: &mut Replay, end_t_s: f64) {
    replay.imu.retain(|sample| sample.t_s <= end_t_s);
    replay.gnss.retain(|sample| sample.t_s <= end_t_s);
    replay
        .reference_attitude
        .retain(|sample| sample.t_s <= end_t_s);
    replay
        .reference_mount
        .retain(|sample| sample.t_s <= end_t_s);
}

fn replay_start_t_s(replay: &Replay) -> Option<f64> {
    replay
        .imu
        .first()
        .map(|s| s.t_s)
        .into_iter()
        .chain(replay.gnss.first().map(|s| s.t_s))
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn run_diagnostics(
    replay: &Replay,
    args: &Args,
    targets_abs: &[f64],
    trace_window_abs: Option<[f64; 2]>,
    mut allocation_csv: Option<&mut AllocationCsv>,
    mut gnss_parity_csv: Option<&mut GnssParityCsv>,
    mut row_basis_csv: Option<&mut RowBasisCsv>,
    mut pvm_accounting_csv: Option<&mut PvmAccountingCsv>,
) -> Result<Vec<Snapshot>> {
    let mut cfg = FilterCompareConfig {
        freeze_misalignment_states: args.freeze_misalignment_states,
        ..FilterCompareConfig::default()
    };
    if let Some(r) = args.r_body_vel {
        cfg.r_body_vel = r;
    }
    if let Some(r) = args.r_body_vel_z {
        cfg.r_body_vel_z = r;
    }
    if let Some(sigma) = args.mount_roll_pitch_init_sigma_deg {
        cfg.mount_roll_pitch_init_sigma_deg = sigma;
    }
    if let Some(sigma) = args.mount_init_sigma_deg {
        cfg.mount_init_sigma_deg = sigma;
    }
    if let Some(sigma) = args.yaw_init_sigma_deg {
        cfg.yaw_init_sigma_deg = sigma;
    }
    if let Some(sigma) = args.gyro_bias_init_sigma_dps {
        cfg.gyro_bias_init_sigma_dps = sigma;
    }
    if let Some(sigma) = args.accel_bias_init_sigma_mps2 {
        cfg.accel_bias_init_sigma_mps2 = sigma;
    }
    if let Some(r) = args.r_zero_vel.filter(|v| v.is_finite() && *v >= 0.0) {
        cfg.r_zero_vel = r;
    }
    if args.reduced_accel_var_scale.is_finite() && args.reduced_accel_var_scale > 0.0 {
        if let Some(noise) = cfg.noise.reduced.as_mut() {
            noise.accel_var *= args.reduced_accel_var_scale;
        }
    }
    if args.reduced_gyro_var_scale.is_finite() && args.reduced_gyro_var_scale > 0.0 {
        if let Some(noise) = cfg.noise.reduced.as_mut() {
            noise.gyro_var *= args.reduced_gyro_var_scale;
        }
    }
    if let Some(sigma) = args
        .full_gyro_scale_sigma
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        cfg.full_init.gyro_scale_sigma = sigma;
    }
    if let Some(sigma) = args
        .full_accel_scale_sigma
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        cfg.full_init.accel_scale_sigma = sigma;
    }
    if let Some(sigma) = args
        .full_attitude_sigma_deg
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        cfg.full_init.attitude_sigma_deg = sigma;
    }
    if let Some(sigma) = args
        .full_attitude_yaw_sigma_deg
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        cfg.full_init.attitude_yaw_sigma_deg = sigma;
    }
    if let Some(sigma) = args
        .full_mount_sigma_deg
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        cfg.full_init.mount_sigma_deg = sigma;
    }
    if let Some(sigma) = args
        .full_mount_yaw_sigma_deg
        .filter(|v| v.is_finite() && *v >= 0.0)
    {
        cfg.full_init.mount_yaw_sigma_deg = sigma;
    }
    let Some(ref_gnss) = replay.gnss.first().copied() else {
        anyhow::bail!("gnss replay is empty");
    };

    let mut fusion = SensorFusion::new();
    apply_fusion_config(&mut fusion, cfg);
    fusion.analysis_set_gnss_velocity_mount_gain_scale(args.gnss_vel_mount_gain_scale);
    if let Some(seed_q_bv) = reference_mount_seed_q_bv(replay, args.misalignment) {
        fusion.set_misalignment(seed_q_bv);
    }

    let mut align_fusion = SensorFusion::new();
    apply_fusion_config(&mut align_fusion, cfg);

    let mut full = full::Filter::new(cfg.noise.full.unwrap_or_else(ProcessNoise::lsm6dso_104hz));
    full.analysis_set_gnss_velocity_mount_gain_scale(args.gnss_vel_mount_gain_scale);
    let mut full_ready = false;
    let mut last_imu: Option<GenericImuSample> = None;
    let mut latest_gnss: Option<GenericGnssSample> = None;
    let mut full_gnss_cursor = 0usize;
    let mut last_gnss_used_t_s = f64::NEG_INFINITY;
    let mut last_full_nhc_t_s: Option<f64> = None;
    let mut full_obs_counts = [0u32; 9];
    let mut full_mount_dx_sum = [0.0f32; 3];
    let mut full_mount_dx_abs_sum = [0.0f32; 3];
    let mut full_att_dx_sum = [0.0f32; 3];
    let mut full_att_dx_abs_sum = [0.0f32; 3];
    let mut full_vel_dx_sum = [0.0f32; 3];
    let mut full_vel_dx_abs_sum = [0.0f32; 3];
    let mut full_mount_dx_sum_by_type = [[0.0f32; 3]; 9];
    let mut full_mount_dx_abs_sum_by_type = [[0.0f32; 3]; 9];
    let mut full_residual_sum_by_type = [0.0f32; 9];
    let mut full_residual_abs_sum_by_type = [0.0f32; 9];
    let mut full_effective_residual_sum_by_type = [0.0f32; 9];
    let mut full_effective_residual_abs_sum_by_type = [0.0f32; 9];
    let mut full_nis_sum_by_type = [0.0f32; 9];
    let mut full_nis_max_by_type = [0.0f32; 9];
    let mut reduced_adaptive_gnss = AdaptiveGnssVelocityScale::new(
        args.adaptive_gnss_velocity_r,
        args.mount_drive_window_s,
        args.adaptive_gnss_velocity_std_scale,
    );
    let mut full_adaptive_gnss = reduced_adaptive_gnss.clone();
    let mut trace_state = TraceState::default();
    let mut latest_transition: Option<TransitionSnapshot> = None;
    let mut reduced_attitude_covariance_override_applied = false;

    let mut snapshots = Vec::new();
    let mut target_idx = 0usize;

    for_each_event(&replay.imu, &replay.gnss, |event| match event {
        ReplayEvent::Imu(_, sample) => {
            let reduced_before_predict = (pvm_accounting_csv.is_some()
                || reduced_adaptive_gnss.enabled)
                .then(|| fusion.reduced().cloned())
                .flatten();
            let _ = fusion.process_imu(fusion_imu_sample(*sample));
            if let (Some(csv), Some(before), Some(after)) = (
                pvm_accounting_csv.as_deref_mut(),
                reduced_before_predict.as_ref(),
                fusion.reduced(),
            ) && full_ready
            {
                let _ = record_reduced_pvm_change(csv, replay, "imu", sample.t_s, before, after);
            }
            if let (Some(before), Some(after)) = (reduced_before_predict.as_ref(), fusion.reduced())
                && full_ready
            {
                record_reduced_mount_drive_adaptive(
                    replay,
                    sample.t_s,
                    before,
                    after,
                    &mut reduced_adaptive_gnss,
                );
            }
            let _ = align_fusion.process_imu(fusion_imu_sample(*sample));
            let Some(prev) = last_imu.replace(*sample) else {
                capture_due_snapshots(
                    replay,
                    &fusion,
                    &full,
                    full_ready,
                    full_obs_counts,
                    full_mount_dx_sum,
                    full_mount_dx_abs_sum,
                    full_att_dx_sum,
                    full_att_dx_abs_sum,
                    full_vel_dx_sum,
                    full_vel_dx_abs_sum,
                    full_mount_dx_sum_by_type,
                    full_mount_dx_abs_sum_by_type,
                    full_residual_sum_by_type,
                    full_residual_abs_sum_by_type,
                    full_effective_residual_sum_by_type,
                    full_effective_residual_abs_sum_by_type,
                    full_nis_sum_by_type,
                    full_nis_max_by_type,
                    latest_transition,
                    targets_abs,
                    &mut target_idx,
                    sample.t_s,
                    &mut snapshots,
                );
                return;
            };
            if full_ready {
                let dt = (sample.t_s - prev.t_s).max(0.0);
                if dt > 0.0 && dt <= 1.0 {
                    let imu = full_imu_delta_from_vehicle(
                        prev.gyro_radps,
                        prev.accel_mps2,
                        sample.gyro_radps,
                        sample.accel_mps2,
                        dt,
                    );
                    let pvm_before_predict = pvm_accounting_csv.as_ref().map(|_| FullSnapshot {
                        nominal: *full.nominal(),
                        p: *full.covariance(),
                        pos_ecef: full.shadow_pos_ecef(),
                        last_dx: *full.last_dx(),
                        last_obs_types: full.last_obs_types().to_vec(),
                    });
                    full.predict(imu);
                    if let (Some(csv), Some(before_snap)) =
                        (pvm_accounting_csv.as_deref_mut(), pvm_before_predict)
                    {
                        let after_snap = FullSnapshot {
                            nominal: *full.nominal(),
                            p: *full.covariance(),
                            pos_ecef: full.shadow_pos_ecef(),
                            last_dx: *full.last_dx(),
                            last_obs_types: full.last_obs_types().to_vec(),
                        };
                        let before = pvm_block_from_common(&transform_full_cov_to_reduced(
                            &before_snap,
                            ref_gnss,
                        ));
                        let after = pvm_block_from_common(&transform_full_cov_to_reduced(
                            &after_snap,
                            ref_gnss,
                        ));
                        let delta = matrix3_sub(after, before);
                        let _ = csv.record_delta(
                            replay,
                            "full",
                            "imu",
                            sample.t_s,
                            "propagation",
                            None,
                            "predict",
                            f32::NAN,
                            f32::NAN,
                            f32::NAN,
                            before,
                            delta,
                            MountDirectionDiag::no_dx(full_mount_error_deg(replay, &before_snap)),
                        );
                    }
                    while full_gnss_cursor < replay.gnss.len()
                        && replay.gnss[full_gnss_cursor].t_s <= sample.t_s + 1.0e-9
                    {
                        latest_gnss = Some(replay.gnss[full_gnss_cursor]);
                        full_gnss_cursor += 1;
                    }
                    let mut gps_pos = None;
                    let mut gps_vel = None;
                    let mut gps_pos_std = 0.0f32;
                    let mut gps_vel_std = None;
                    let mut dt_since_gnss = 1.0f32;
                    if let Some(gnss) = latest_gnss
                        && (0.0..=0.05).contains(&(sample.t_s - gnss.t_s))
                        && gnss.t_s != last_gnss_used_t_s
                    {
                        gps_pos = Some(lla_to_ecef(gnss.lat_deg, gnss.lon_deg, gnss.height_m));
                        gps_vel = Some(
                            ned_vector_to_ecef(gnss.lat_deg, gnss.lon_deg, gnss.vel_ned_mps)
                                .map(|v| v as f32),
                        );
                        gps_pos_std = ((gnss.pos_std_m[0] + gnss.pos_std_m[1] + gnss.pos_std_m[2])
                            / 3.0)
                            .max(0.1) as f32;
                        let vel_scale = if args.full_gnss_vel_std_scale.is_finite()
                            && args.full_gnss_vel_std_scale > 0.0
                        {
                            args.full_gnss_vel_std_scale
                        } else {
                            1.0
                        } * full_adaptive_gnss.current_scale();
                        gps_vel_std =
                            Some(gnss.vel_std_mps.map(|v| v.max(0.01) as f32 * vel_scale));
                        full_adaptive_gnss.mark_gnss_used();
                        dt_since_gnss = if last_gnss_used_t_s.is_finite() {
                            (gnss.t_s - last_gnss_used_t_s).clamp(1.0e-3, 1.0) as f32
                        } else {
                            1.0
                        };
                        last_gnss_used_t_s = gnss.t_s;
                    }
                    let estimated_speed_mps = full_speed_estimate_mps(&full);
                    let (nhc_gate_speed_mps, full_nhc_dt) = if cfg.nhc_update_period_s > 0.0 {
                        let active = runtime_nhc_active(
                            sample.accel_mps2.map(|v| v as f32),
                            sample.gyro_radps.map(|v| v as f32),
                        ) && estimated_speed_mps > RUNTIME_NHC_MIN_SPEED_MPS;
                        let interval = nhc_observation_interval(
                            last_full_nhc_t_s,
                            sample.t_s,
                            dt,
                            cfg.nhc_update_period_s as f64,
                            active,
                        );
                        last_full_nhc_t_s = interval.1;
                        (interval.0.map(|_| estimated_speed_mps), interval.0)
                    } else {
                        (
                            (estimated_speed_mps > RUNTIME_NHC_MIN_SPEED_MPS)
                                .then_some(estimated_speed_mps),
                            Some(dt.max(1.0e-3) as f32),
                        )
                    };
                    let pvm_before_batch = (pvm_accounting_csv.is_some()
                        || full_adaptive_gnss.enabled)
                        .then(|| FullSnapshot {
                            nominal: *full.nominal(),
                            p: *full.covariance(),
                            pos_ecef: full.shadow_pos_ecef(),
                            last_dx: *full.last_dx(),
                            last_obs_types: full.last_obs_types().to_vec(),
                        });
                    full.fuse_reference_batch_full_with_nhc_speed_and_r(
                        gps_pos,
                        gps_vel,
                        gps_pos_std,
                        gps_vel_std,
                        dt_since_gnss,
                        nhc_gate_speed_mps,
                        cfg.r_body_vel,
                        cfg.r_body_vel_z,
                        sample.gyro_radps.map(|v| v as f32),
                        sample.accel_mps2.map(|v| v as f32),
                        full_nhc_dt.unwrap_or(dt.max(1.0e-3) as f32),
                    );
                    if !full.last_obs_types().is_empty() {
                        if let Some(before_snap) = pvm_before_batch.as_ref() {
                            for (obs_idx, ty) in full.last_obs_types().iter().copied().enumerate() {
                                let phase = full_pvm_phase_for_obs(ty as usize);
                                if !matches!(phase, "gnss_velocity" | "nhc") {
                                    continue;
                                }
                                let innovation_var = full.last_innovation_vars()[obs_idx];
                                if !innovation_var.is_finite() || innovation_var <= 0.0 {
                                    continue;
                                }
                                full_adaptive_gnss.record(
                                    sample.t_s - replay_start_t_s(replay).unwrap_or(0.0),
                                    phase,
                                    full_mount_error_deg(replay, before_snap),
                                    [
                                        rad_f32_to_deg(full.last_dx_by_obs()[obs_idx][21]),
                                        rad_f32_to_deg(full.last_dx_by_obs()[obs_idx][22]),
                                        rad_f32_to_deg(full.last_dx_by_obs()[obs_idx][23]),
                                    ],
                                    f64::from(scalar_nis(
                                        full.last_effective_residuals()[obs_idx],
                                        innovation_var,
                                    )),
                                    f64::from(full.last_effective_residuals()[obs_idx]),
                                );
                            }
                        }
                        if let (Some(csv), Some(before_snap)) =
                            (pvm_accounting_csv.as_deref_mut(), pvm_before_batch.as_ref())
                        {
                            let after_snap = FullSnapshot {
                                nominal: *full.nominal(),
                                p: *full.covariance(),
                                pos_ecef: full.shadow_pos_ecef(),
                                last_dx: *full.last_dx(),
                                last_obs_types: full.last_obs_types().to_vec(),
                            };
                            let before_common =
                                transform_full_cov_to_reduced(before_snap, ref_gnss);
                            let after_common = transform_full_cov_to_reduced(&after_snap, ref_gnss);
                            let mut running = pvm_block_from_common(&before_common);
                            let mut row_sum = [[0.0f64; 3]; 3];
                            for (obs_idx, ty) in full.last_obs_types().iter().copied().enumerate() {
                                let innovation_var = full.last_innovation_vars()[obs_idx];
                                if !innovation_var.is_finite() || innovation_var <= 0.0 {
                                    continue;
                                }
                                let native_delta = full_covariance_row_delta(
                                    &full.last_k_by_obs()[obs_idx],
                                    innovation_var,
                                );
                                let delta_snap = FullSnapshot {
                                    nominal: before_snap.nominal,
                                    p: native_delta,
                                    pos_ecef: before_snap.pos_ecef,
                                    last_dx: [0.0; ERROR_STATES],
                                    last_obs_types: Vec::new(),
                                };
                                let delta_common = pvm_block_from_common(
                                    &transform_full_cov_to_reduced(&delta_snap, ref_gnss),
                                );
                                row_sum = matrix3_add(row_sum, delta_common);
                                let phase = full_pvm_phase_for_obs(ty as usize);
                                if let Ok(next_running) = csv.record_delta(
                                    replay,
                                    "full",
                                    "imu",
                                    sample.t_s,
                                    phase,
                                    Some(obs_idx),
                                    full_obs_type_label(ty as usize),
                                    full.last_residuals()[obs_idx],
                                    full.last_effective_residuals()[obs_idx],
                                    innovation_var,
                                    running,
                                    delta_common,
                                    MountDirectionDiag::with_dx(
                                        full_mount_error_deg(replay, &before_snap),
                                        [
                                            rad_f32_to_deg(full.last_dx_by_obs()[obs_idx][21]),
                                            rad_f32_to_deg(full.last_dx_by_obs()[obs_idx][22]),
                                            rad_f32_to_deg(full.last_dx_by_obs()[obs_idx][23]),
                                        ],
                                    ),
                                ) {
                                    running = next_running;
                                }
                            }
                            let reset_delta = matrix3_sub(
                                pvm_block_from_common(&after_common),
                                matrix3_add(pvm_block_from_common(&before_common), row_sum),
                            );
                            let _ = csv.record_delta(
                                replay,
                                "full",
                                "imu",
                                sample.t_s,
                                "reset",
                                None,
                                "reset",
                                f32::NAN,
                                f32::NAN,
                                f32::NAN,
                                running,
                                reset_delta,
                                MountDirectionDiag::no_dx(full_mount_error_deg(
                                    replay,
                                    &before_snap,
                                )),
                            );
                        }
                        let dx = full.last_dx();
                        let full_snap = FullSnapshot {
                            nominal: *full.nominal(),
                            p: *full.covariance(),
                            pos_ecef: full.shadow_pos_ecef(),
                            last_dx: *dx,
                            last_obs_types: full.last_obs_types().to_vec(),
                        };
                        let dx_as_reduced = transform_full_dx_to_reduced(&full_snap, dx, ref_gnss);
                        for axis in 0..3 {
                            let value = dx[21 + axis];
                            full_mount_dx_sum[axis] += value;
                            full_mount_dx_abs_sum[axis] += value.abs();
                            let att_value = dx_as_reduced[axis];
                            full_att_dx_sum[axis] += att_value;
                            full_att_dx_abs_sum[axis] += att_value.abs();
                            let vel_value = dx_as_reduced[3 + axis];
                            full_vel_dx_sum[axis] += vel_value;
                            full_vel_dx_abs_sum[axis] += vel_value.abs();
                        }
                        for (obs_idx, ty) in full.last_obs_types().iter().copied().enumerate() {
                            let Some(sum) = full_mount_dx_sum_by_type.get_mut(ty as usize) else {
                                continue;
                            };
                            let Some(abs_sum) = full_mount_dx_abs_sum_by_type.get_mut(ty as usize)
                            else {
                                continue;
                            };
                            let row_dx = &full.last_dx_by_obs()[obs_idx];
                            for axis in 0..3 {
                                let value = row_dx[21 + axis];
                                sum[axis] += value;
                                abs_sum[axis] += value.abs();
                            }
                            let ty = ty as usize;
                            let residual = full.last_residuals()[obs_idx];
                            let effective_residual = full.last_effective_residuals()[obs_idx];
                            let innovation_var = full.last_innovation_vars()[obs_idx];
                            full_residual_sum_by_type[ty] += residual;
                            full_residual_abs_sum_by_type[ty] += residual.abs();
                            full_effective_residual_sum_by_type[ty] += effective_residual;
                            full_effective_residual_abs_sum_by_type[ty] += effective_residual.abs();
                            if innovation_var > 0.0 {
                                let nis = effective_residual * effective_residual / innovation_var;
                                full_nis_sum_by_type[ty] += nis;
                                full_nis_max_by_type[ty] = full_nis_max_by_type[ty].max(nis);
                            }
                        }
                    }
                    for ty in full.last_obs_types() {
                        if let Some(count) = full_obs_counts.get_mut(*ty as usize) {
                            *count += 1;
                        }
                    }
                    latest_transition =
                        transition_snapshot(&fusion, &full, *sample, prev, dt, ref_gnss);
                }
            }
            capture_due_snapshots(
                replay,
                &fusion,
                &full,
                full_ready,
                full_obs_counts,
                full_mount_dx_sum,
                full_mount_dx_abs_sum,
                full_att_dx_sum,
                full_att_dx_abs_sum,
                full_vel_dx_sum,
                full_vel_dx_abs_sum,
                full_mount_dx_sum_by_type,
                full_mount_dx_abs_sum_by_type,
                full_residual_sum_by_type,
                full_residual_abs_sum_by_type,
                full_effective_residual_sum_by_type,
                full_effective_residual_abs_sum_by_type,
                full_nis_sum_by_type,
                full_nis_max_by_type,
                latest_transition,
                targets_abs,
                &mut target_idx,
                sample.t_s,
                &mut snapshots,
            );
            maybe_print_trace(
                replay,
                ref_gnss,
                trace_window_abs,
                args.trace_interval_s,
                &mut trace_state,
                "imu",
                sample.t_s,
                &fusion,
                &full,
                full_ready,
                full_obs_counts,
            );
            if let Some(csv) = allocation_csv.as_deref_mut() {
                let _ = csv.record_reduced(
                    replay,
                    "imu",
                    sample.t_s,
                    fusion.reduced(),
                    full_ready,
                    ref_gnss,
                );
                if full_ready {
                    let _ = csv.record_full(replay, sample.t_s, &full, ref_gnss);
                }
            }
            if let Some(csv) = gnss_parity_csv.as_deref_mut() {
                let _ = csv.record_reduced(replay, "imu", sample.t_s, fusion.reduced(), full_ready);
                if full_ready {
                    let _ = csv.record_full(replay, "imu", sample.t_s, &full, ref_gnss);
                }
            }
            if let Some(csv) = row_basis_csv.as_deref_mut() {
                let _ = csv.record_reduced(replay, "imu", sample.t_s, fusion.reduced(), full_ready);
                if full_ready {
                    let _ = csv.record_full(replay, "imu", sample.t_s, &full, ref_gnss);
                }
            }
        }
        ReplayEvent::Gnss(index, sample) => {
            let _ = fusion.process_gnss(scaled_fusion_gnss_sample(
                *sample,
                args.reduced_gnss_pos_std_scale,
                args.reduced_gnss_pos_d_std_scale,
                args.reduced_gnss_vel_std_scale * f64::from(reduced_adaptive_gnss.current_scale()),
            ));
            reduced_adaptive_gnss.mark_gnss_used();
            if !reduced_attitude_covariance_override_applied
                && let Some(sigma_deg) = args
                    .reduced_attitude_roll_pitch_sigma_deg
                    .filter(|v| v.is_finite() && *v >= 0.0)
                && fusion.reduced().is_some()
            {
                fusion.analysis_set_reduced_attitude_roll_pitch_covariance(sigma_deg.to_radians());
                reduced_attitude_covariance_override_applied = true;
            }
            let _ = align_fusion.process_gnss(fusion_gnss_sample(*sample));
            latest_gnss = Some(*sample);
            if !full_ready && align_fusion.mount_ready() {
                let speed = sample.vel_ned_mps[0].hypot(sample.vel_ned_mps[1]);
                if speed >= 0.5 {
                    let yaw_rad = sample.vel_ned_mps[1].atan2(sample.vel_ned_mps[0]) as f32;
                    let pos_ecef = lla_to_ecef(sample.lat_deg, sample.lon_deg, sample.height_m);
                    let vel_ecef =
                        ned_vector_to_ecef(sample.lat_deg, sample.lon_deg, sample.vel_ned_mps)
                            .map(|v| v as f32);
                    full.init_vehicle_from_nav_ecef_state(
                        yaw_rad,
                        sample.lat_deg,
                        sample.lon_deg,
                        pos_ecef,
                        vel_ecef,
                        None,
                        None,
                    );
                    full.set_covariance(full::default_full_nav_covariance(
                        yaw_rad,
                        sample.lat_deg,
                        sample.lon_deg,
                        sample.pos_std_m.map(|v| v as f32),
                        sample.vel_std_mps.map(|v| v as f32),
                        cfg.full_init,
                    ));
                    if let Some(seed_q) = align_fusion.mount_q_bv() {
                        full.set_mount_quat(seed_q);
                    }
                    full_ready = true;
                    full_gnss_cursor = index + 1;
                    last_gnss_used_t_s = sample.t_s;
                }
            }
            capture_due_snapshots(
                replay,
                &fusion,
                &full,
                full_ready,
                full_obs_counts,
                full_mount_dx_sum,
                full_mount_dx_abs_sum,
                full_att_dx_sum,
                full_att_dx_abs_sum,
                full_vel_dx_sum,
                full_vel_dx_abs_sum,
                full_mount_dx_sum_by_type,
                full_mount_dx_abs_sum_by_type,
                full_residual_sum_by_type,
                full_residual_abs_sum_by_type,
                full_effective_residual_sum_by_type,
                full_effective_residual_abs_sum_by_type,
                full_nis_sum_by_type,
                full_nis_max_by_type,
                latest_transition,
                targets_abs,
                &mut target_idx,
                sample.t_s,
                &mut snapshots,
            );
            maybe_print_trace(
                replay,
                ref_gnss,
                trace_window_abs,
                args.trace_interval_s,
                &mut trace_state,
                "gnss",
                sample.t_s,
                &fusion,
                &full,
                full_ready,
                full_obs_counts,
            );
            if let Some(csv) = allocation_csv.as_deref_mut() {
                let _ = csv.record_reduced(
                    replay,
                    "gnss",
                    sample.t_s,
                    fusion.reduced(),
                    full_ready,
                    ref_gnss,
                );
            }
        }
    });

    let final_t = replay
        .imu
        .last()
        .map(|s| s.t_s)
        .unwrap_or(ref_gnss.t_s)
        .max(replay.gnss.last().map(|s| s.t_s).unwrap_or(ref_gnss.t_s));
    capture_due_snapshots(
        replay,
        &fusion,
        &full,
        full_ready,
        full_obs_counts,
        full_mount_dx_sum,
        full_mount_dx_abs_sum,
        full_att_dx_sum,
        full_att_dx_abs_sum,
        full_vel_dx_sum,
        full_vel_dx_abs_sum,
        full_mount_dx_sum_by_type,
        full_mount_dx_abs_sum_by_type,
        full_residual_sum_by_type,
        full_residual_abs_sum_by_type,
        full_effective_residual_sum_by_type,
        full_effective_residual_abs_sum_by_type,
        full_nis_sum_by_type,
        full_nis_max_by_type,
        latest_transition,
        targets_abs,
        &mut target_idx,
        final_t,
        &mut snapshots,
    );
    reduced_adaptive_gnss.print_summary("reduced");
    full_adaptive_gnss.print_summary("full");
    Ok(snapshots)
}

fn apply_fusion_config(fusion: &mut SensorFusion, cfg: FilterCompareConfig) {
    fusion.set_align_config(cfg.align);
    if let Some(noise) = cfg.noise.reduced {
        fusion.set_reduced_noise(noise);
    }
    fusion.set_r_body_vel_yz(cfg.r_body_vel, cfg.r_body_vel_z);
    fusion.set_attitude_roll_pitch_init_sigma_rad(
        cfg.attitude_roll_pitch_init_sigma_deg.to_radians(),
    );
    fusion.set_yaw_init_sigma_rad(cfg.yaw_init_sigma_deg.to_radians());
    fusion.set_gyro_bias_init_sigma_radps(cfg.gyro_bias_init_sigma_dps.to_radians());
    fusion.set_accel_bias_init_sigma_mps2(cfg.accel_bias_init_sigma_mps2);
    fusion.set_mount_roll_pitch_init_sigma_rad(cfg.mount_roll_pitch_init_sigma_deg.to_radians());
    fusion.set_mount_init_sigma_rad(cfg.mount_init_sigma_deg.to_radians());
    fusion.set_r_vehicle_speed(cfg.r_vehicle_speed);
    fusion.set_r_zero_vel(cfg.r_zero_vel);
    fusion.set_r_stationary_accel(cfg.r_stationary_accel);
    fusion.set_mount_align_rw_var(cfg.mount_align_rw_var);
    fusion.set_align_handoff_delay_s(cfg.align_handoff_delay_s);
    fusion.set_freeze_misalignment_states(cfg.freeze_misalignment_states);
    fusion.set_mount_settle_time_s(cfg.mount_settle_time_s);
    fusion.set_mount_settle_release_sigma_rad(cfg.mount_settle_release_sigma_deg.to_radians());
    fusion.set_mount_settle_zero_cross_covariance(cfg.mount_settle_zero_cross_covariance);
}

fn full_speed_estimate_mps(full: &full::Filter) -> f32 {
    let n = full.nominal();
    (n.vn * n.vn + n.ve * n.ve + n.vd * n.vd).sqrt()
}

fn runtime_nhc_active(accel_mps2: [f32; 3], gyro_radps: [f32; 3]) -> bool {
    let gyro_norm = (gyro_radps[0] * gyro_radps[0]
        + gyro_radps[1] * gyro_radps[1]
        + gyro_radps[2] * gyro_radps[2])
        .sqrt();
    let accel_norm = (accel_mps2[0] * accel_mps2[0]
        + accel_mps2[1] * accel_mps2[1]
        + accel_mps2[2] * accel_mps2[2])
        .sqrt();
    gyro_norm < RUNTIME_NHC_MAX_GYRO_NORM_RADPS
        && (accel_norm - 9.81).abs() < RUNTIME_NHC_MAX_ACCEL_NORM_ERR_MPS2
}

fn nhc_observation_interval(
    last_t_s: Option<f64>,
    t_s: f64,
    fallback_dt_s: f64,
    period_s: f64,
    active: bool,
) -> (Option<f32>, Option<f64>) {
    if !active {
        return (None, Some(t_s));
    }
    let fallback_dt_s = fallback_dt_s.max(1.0e-3);
    if period_s <= 0.0 {
        return (Some(fallback_dt_s as f32), Some(t_s));
    }
    let Some(last_t_s) = last_t_s else {
        return (Some(fallback_dt_s as f32), Some(t_s));
    };
    let elapsed_s = t_s - last_t_s;
    if elapsed_s + 1.0e-4 < period_s {
        return (None, Some(last_t_s));
    }
    (Some(elapsed_s.max(fallback_dt_s) as f32), Some(t_s))
}

fn capture_due_snapshots(
    replay: &Replay,
    fusion: &SensorFusion,
    full: &full::Filter,
    full_ready: bool,
    full_obs_counts: [u32; 9],
    full_mount_dx_sum: [f32; 3],
    full_mount_dx_abs_sum: [f32; 3],
    full_att_dx_sum: [f32; 3],
    full_att_dx_abs_sum: [f32; 3],
    full_vel_dx_sum: [f32; 3],
    full_vel_dx_abs_sum: [f32; 3],
    full_mount_dx_sum_by_type: [[f32; 3]; 9],
    full_mount_dx_abs_sum_by_type: [[f32; 3]; 9],
    full_residual_sum_by_type: [f32; 9],
    full_residual_abs_sum_by_type: [f32; 9],
    full_effective_residual_sum_by_type: [f32; 9],
    full_effective_residual_abs_sum_by_type: [f32; 9],
    full_nis_sum_by_type: [f32; 9],
    full_nis_max_by_type: [f32; 9],
    transition: Option<TransitionSnapshot>,
    targets_abs: &[f64],
    target_idx: &mut usize,
    t_s: f64,
    snapshots: &mut Vec<Snapshot>,
) {
    while let Some(&target_t_s) = targets_abs.get(*target_idx) {
        if t_s < target_t_s {
            return;
        }
        snapshots.push(Snapshot {
            target_rel_s: target_t_s - replay_start_t_s(replay).unwrap_or(0.0),
            t_s,
            reduced: fusion.reduced().copied(),
            full: full_ready.then(|| FullSnapshot {
                nominal: *full.nominal(),
                p: *full.covariance(),
                pos_ecef: full.shadow_pos_ecef(),
                last_dx: *full.last_dx(),
                last_obs_types: full.last_obs_types().to_vec(),
            }),
            transition,
            reference_mount_q_bv: reference_mount_at(&replay.reference_mount, t_s),
            reference_att_q: reference_attitude_at(&replay.reference_attitude, t_s),
            reduced_type_counts: fusion
                .reduced()
                .map(|e| e.update_diag.type_counts)
                .unwrap_or([0; UPDATE_DIAG_TYPES]),
            full_obs_counts,
            full_mount_dx_sum,
            full_mount_dx_abs_sum,
            full_att_dx_sum,
            full_att_dx_abs_sum,
            full_vel_dx_sum,
            full_vel_dx_abs_sum,
            full_mount_dx_sum_by_type,
            full_mount_dx_abs_sum_by_type,
            full_residual_sum_by_type,
            full_residual_abs_sum_by_type,
            full_effective_residual_sum_by_type,
            full_effective_residual_abs_sum_by_type,
            full_nis_sum_by_type,
            full_nis_max_by_type,
        });
        *target_idx += 1;
    }
}

fn print_snapshots(snapshots: &[Snapshot], replay: &Replay) {
    let Some(ref_gnss) = replay.gnss.first().copied() else {
        return;
    };
    for snapshot in snapshots {
        println!(
            "[covhist] snapshot target_rel_s={:.3} sample_t_s={:.6}",
            snapshot.target_rel_s, snapshot.t_s
        );
        print_state_summary(snapshot, ref_gnss);
        if let (Some(reduced), Some(full)) = (&snapshot.reduced, &snapshot.full) {
            let p_full_as_reduced = transform_full_cov_to_reduced(full, ref_gnss);
            print_covariance_summary(reduced, &p_full_as_reduced);
            print_mount_correlations("Reduced", &reduced.p);
            print_mount_correlations("FullAsREDUCED", &p_full_as_reduced);
            print_common_gnss_gain_summary(snapshot, replay, &reduced.p, &p_full_as_reduced);
            if let Some(transition) = snapshot.transition {
                print_transition_summary(transition);
            }
            print_update_summary(snapshot, full);
        }
    }
}

fn print_interval_summary(snapshots: &[Snapshot], replay: &Replay, window_rel_s: [f64; 2]) {
    let Some(ref_gnss) = replay.gnss.first().copied() else {
        return;
    };
    let Some(start) = nearest_snapshot(snapshots, window_rel_s[0]) else {
        println!(
            "[covhist-interval] missing start snapshot rel_s={:.3}",
            window_rel_s[0]
        );
        return;
    };
    let Some(end) = nearest_snapshot(snapshots, window_rel_s[1]) else {
        println!(
            "[covhist-interval] missing end snapshot rel_s={:.3}",
            window_rel_s[1]
        );
        return;
    };

    println!(
        "[covhist-interval] window_rel_s=[{:.3},{:.3}] sample_rel_s=[{:.3},{:.3}]",
        window_rel_s[0], window_rel_s[1], start.target_rel_s, end.target_rel_s
    );
    print_interval_state_summary(start, end, ref_gnss);
    print_interval_update_summary(start, end);
    if let (Some(start_reduced), Some(end_reduced), Some(start_full), Some(end_full)) =
        (&start.reduced, &end.reduced, &start.full, &end.full)
    {
        let start_full_p = transform_full_cov_to_reduced(start_full, ref_gnss);
        let end_full_p = transform_full_cov_to_reduced(end_full, ref_gnss);
        print_interval_sigma_summary(start_reduced, end_reduced, &start_full_p, &end_full_p);
        print_interval_correlation_summary(
            &start_reduced.p,
            &end_reduced.p,
            &start_full_p,
            &end_full_p,
        );
    }
}

fn nearest_snapshot(snapshots: &[Snapshot], rel_s: f64) -> Option<&Snapshot> {
    snapshots.iter().min_by(|a, b| {
        (a.target_rel_s - rel_s)
            .abs()
            .partial_cmp(&(b.target_rel_s - rel_s).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn print_interval_state_summary(start: &Snapshot, end: &Snapshot, ref_gnss: GenericGnssSample) {
    let start_metrics = state_metrics(start, ref_gnss);
    let end_metrics = state_metrics(end, ref_gnss);
    println!(
        "[covhist-interval] qerr_mount_deg reduced={:.6}->{:.6} d={:.6} full={:.6}->{:.6} d={:.6}",
        start_metrics.reduced_mount_qerr,
        end_metrics.reduced_mount_qerr,
        end_metrics.reduced_mount_qerr - start_metrics.reduced_mount_qerr,
        start_metrics.full_mount_qerr,
        end_metrics.full_mount_qerr,
        end_metrics.full_mount_qerr - start_metrics.full_mount_qerr,
    );
    println!(
        "[covhist-interval] qerr_att_deg reduced={:.6}->{:.6} d={:.6} full={:.6}->{:.6} d={:.6}",
        start_metrics.reduced_att_qerr,
        end_metrics.reduced_att_qerr,
        end_metrics.reduced_att_qerr - start_metrics.reduced_att_qerr,
        start_metrics.full_att_qerr,
        end_metrics.full_att_qerr,
        end_metrics.full_att_qerr - start_metrics.full_att_qerr,
    );
    println!(
        "[covhist-interval] nhc_residual_end_mps reduced_yz=[{:.6},{:.6}] full_yz=[{:.6},{:.6}]",
        end_metrics.reduced_nhc[0],
        end_metrics.reduced_nhc[1],
        end_metrics.full_nhc[0],
        end_metrics.full_nhc[1],
    );
}

#[derive(Clone, Copy)]
struct StateMetrics {
    reduced_mount_qerr: f64,
    full_mount_qerr: f64,
    reduced_att_qerr: f64,
    full_att_qerr: f64,
    reduced_nhc: [f64; 2],
    full_nhc: [f64; 2],
}

fn state_metrics(snapshot: &Snapshot, ref_gnss: GenericGnssSample) -> StateMetrics {
    let reduced_mount_q = snapshot.reduced.as_ref().map(|e| {
        as_q64([
            e.nominal.q_bv0,
            e.nominal.q_bv1,
            e.nominal.q_bv2,
            e.nominal.q_bv3,
        ])
    });
    let full_mount_q = snapshot.full.as_ref().map(|l| {
        as_q64([
            l.nominal.q_bv0,
            l.nominal.q_bv1,
            l.nominal.q_bv2,
            l.nominal.q_bv3,
        ])
    });
    StateMetrics {
        reduced_mount_qerr: reduced_mount_q
            .zip(snapshot.reference_mount_q_bv)
            .map(|(a, b)| quat_angle_deg(a, b))
            .unwrap_or(f64::NAN),
        full_mount_qerr: full_mount_q
            .zip(snapshot.reference_mount_q_bv)
            .map(|(a, b)| quat_angle_deg(a, b))
            .unwrap_or(f64::NAN),
        reduced_att_qerr: snapshot
            .reduced
            .as_ref()
            .and_then(|e| snapshot.reference_att_q.map(|r| (e, r)))
            .map(|(e, r)| quat_angle_deg(reduced_att_q(e), r))
            .unwrap_or(f64::NAN),
        full_att_qerr: snapshot
            .full
            .as_ref()
            .and_then(|l| snapshot.reference_att_q.map(|r| (l, r)))
            .map(|(l, r)| quat_angle_deg(full_att_q_ned(l, ref_gnss), r))
            .unwrap_or(f64::NAN),
        reduced_nhc: snapshot
            .reduced
            .as_ref()
            .map(reduced_nhc_residual_yz)
            .unwrap_or([f64::NAN; 2]),
        full_nhc: snapshot
            .full
            .as_ref()
            .map(full_nhc_residual_yz)
            .unwrap_or([f64::NAN; 2]),
    }
}

fn print_interval_update_summary(start: &Snapshot, end: &Snapshot) {
    let reduced_delta = count_delta(&end.reduced_type_counts, &start.reduced_type_counts);
    let full_delta = count_delta(&end.full_obs_counts, &start.full_obs_counts);
    println!(
        "[covhist-interval] reduced_update_delta pos_xy={} pos_d={} vel_xy={} vel_d={} zero_xy={} zero_d={} nhc_y={} nhc_z={}",
        reduced_delta[0],
        reduced_delta[8],
        reduced_delta[1],
        reduced_delta[9],
        reduced_delta[2],
        reduced_delta[10],
        reduced_delta[DIAG_BODY_VEL_Y],
        reduced_delta[DIAG_BODY_VEL_Z]
    );
    println!(
        "[covhist-interval] full_obs_delta pos=[{},{},{}] vel=[{},{},{}] nhc_y={} nhc_z={}",
        full_delta[1],
        full_delta[2],
        full_delta[3],
        full_delta[4],
        full_delta[5],
        full_delta[6],
        full_delta[7],
        full_delta[8]
    );
    if let (Some(start_reduced), Some(end_reduced)) = (&start.reduced, &end.reduced) {
        for (label, idx) in selected_reduced_diag_types() {
            println!(
                "[covhist-interval] reduced_mount_dx type={} count_delta={} net_deg=[{:.6},{:.6},{:.6}] abs_deg=[{:.6},{:.6},{:.6}]",
                label,
                reduced_delta[idx],
                rad_f32_to_deg(
                    end_reduced.update_diag.sum_dx_mount_roll[idx]
                        - start_reduced.update_diag.sum_dx_mount_roll[idx]
                ),
                rad_f32_to_deg(
                    end_reduced.update_diag.sum_dx_mount_pitch[idx]
                        - start_reduced.update_diag.sum_dx_mount_pitch[idx]
                ),
                rad_f32_to_deg(
                    end_reduced.update_diag.sum_dx_mount_yaw[idx]
                        - start_reduced.update_diag.sum_dx_mount_yaw[idx]
                ),
                rad_f32_to_deg(
                    end_reduced.update_diag.sum_abs_dx_mount_roll[idx]
                        - start_reduced.update_diag.sum_abs_dx_mount_roll[idx]
                ),
                rad_f32_to_deg(
                    end_reduced.update_diag.sum_abs_dx_mount_pitch[idx]
                        - start_reduced.update_diag.sum_abs_dx_mount_pitch[idx]
                ),
                rad_f32_to_deg(
                    end_reduced.update_diag.sum_abs_dx_mount_yaw[idx]
                        - start_reduced.update_diag.sum_abs_dx_mount_yaw[idx]
                ),
            );
            println!(
                "[covhist-interval] reduced_att_dx type={} count_delta={} net_deg=[{:.6},{:.6},{:.6}] abs_deg=[{:.6},{:.6},{:.6}]",
                label,
                reduced_delta[idx],
                rad_f32_to_deg(
                    end_reduced.update_diag.sum_dx_att_roll[idx]
                        - start_reduced.update_diag.sum_dx_att_roll[idx]
                ),
                rad_f32_to_deg(
                    end_reduced.update_diag.sum_dx_pitch[idx]
                        - start_reduced.update_diag.sum_dx_pitch[idx]
                ),
                rad_f32_to_deg(
                    end_reduced.update_diag.sum_dx_yaw[idx]
                        - start_reduced.update_diag.sum_dx_yaw[idx]
                ),
                rad_f32_to_deg(
                    end_reduced.update_diag.sum_abs_dx_att_roll[idx]
                        - start_reduced.update_diag.sum_abs_dx_att_roll[idx]
                ),
                rad_f32_to_deg(
                    end_reduced.update_diag.sum_abs_dx_pitch[idx]
                        - start_reduced.update_diag.sum_abs_dx_pitch[idx]
                ),
                rad_f32_to_deg(
                    end_reduced.update_diag.sum_abs_dx_yaw[idx]
                        - start_reduced.update_diag.sum_abs_dx_yaw[idx]
                ),
            );
            println!(
                "[covhist-interval] reduced_vel_dx type={} count_delta={} net_mps=[{:.6},{:.6},{:.6}] abs_mps=[{:.6},{:.6},{:.6}]",
                label,
                reduced_delta[idx],
                end_reduced.update_diag.sum_dx_vel_n[idx]
                    - start_reduced.update_diag.sum_dx_vel_n[idx],
                end_reduced.update_diag.sum_dx_vel_e[idx]
                    - start_reduced.update_diag.sum_dx_vel_e[idx],
                end_reduced.update_diag.sum_dx_vel_d[idx]
                    - start_reduced.update_diag.sum_dx_vel_d[idx],
                end_reduced.update_diag.sum_abs_dx_vel_n[idx]
                    - start_reduced.update_diag.sum_abs_dx_vel_n[idx],
                end_reduced.update_diag.sum_abs_dx_vel_e[idx]
                    - start_reduced.update_diag.sum_abs_dx_vel_e[idx],
                end_reduced.update_diag.sum_abs_dx_vel_d[idx]
                    - start_reduced.update_diag.sum_abs_dx_vel_d[idx],
            );
            println!(
                "[covhist-interval] reduced_innov type={} count_delta={} net={:.6} abs={:.6} nis_sum={:.6} nis_max={:.6}",
                label,
                reduced_delta[idx],
                end_reduced.update_diag.sum_innovation[idx]
                    - start_reduced.update_diag.sum_innovation[idx],
                end_reduced.update_diag.sum_abs_innovation[idx]
                    - start_reduced.update_diag.sum_abs_innovation[idx],
                end_reduced.update_diag.sum_nis[idx] - start_reduced.update_diag.sum_nis[idx],
                end_reduced.update_diag.max_nis[idx],
            );
        }
    }
    println!(
        "[covhist-interval] full_mount_dx total_net_deg=[{:.6},{:.6},{:.6}] total_abs_deg=[{:.6},{:.6},{:.6}]",
        rad_f32_to_deg(end.full_mount_dx_sum[0] - start.full_mount_dx_sum[0]),
        rad_f32_to_deg(end.full_mount_dx_sum[1] - start.full_mount_dx_sum[1]),
        rad_f32_to_deg(end.full_mount_dx_sum[2] - start.full_mount_dx_sum[2]),
        rad_f32_to_deg(end.full_mount_dx_abs_sum[0] - start.full_mount_dx_abs_sum[0]),
        rad_f32_to_deg(end.full_mount_dx_abs_sum[1] - start.full_mount_dx_abs_sum[1]),
        rad_f32_to_deg(end.full_mount_dx_abs_sum[2] - start.full_mount_dx_abs_sum[2]),
    );
    for (ty, label) in [
        (1usize, "pos_x"),
        (2, "pos_y"),
        (3, "pos_z"),
        (4, "vel_x"),
        (5, "vel_y"),
        (6, "vel_z"),
        (7, "nhc_y"),
        (8, "nhc_z"),
    ] {
        let net = [
            end.full_mount_dx_sum_by_type[ty][0] - start.full_mount_dx_sum_by_type[ty][0],
            end.full_mount_dx_sum_by_type[ty][1] - start.full_mount_dx_sum_by_type[ty][1],
            end.full_mount_dx_sum_by_type[ty][2] - start.full_mount_dx_sum_by_type[ty][2],
        ];
        let abs = [
            end.full_mount_dx_abs_sum_by_type[ty][0] - start.full_mount_dx_abs_sum_by_type[ty][0],
            end.full_mount_dx_abs_sum_by_type[ty][1] - start.full_mount_dx_abs_sum_by_type[ty][1],
            end.full_mount_dx_abs_sum_by_type[ty][2] - start.full_mount_dx_abs_sum_by_type[ty][2],
        ];
        println!(
            "[covhist-interval] full_mount_dx type={} net_deg=[{:.6},{:.6},{:.6}] abs_deg=[{:.6},{:.6},{:.6}]",
            label,
            rad_f32_to_deg(net[0]),
            rad_f32_to_deg(net[1]),
            rad_f32_to_deg(net[2]),
            rad_f32_to_deg(abs[0]),
            rad_f32_to_deg(abs[1]),
            rad_f32_to_deg(abs[2]),
        );
        println!(
            "[covhist-interval] full_residual type={} net={:.6} abs={:.6} eff_net={:.6} eff_abs={:.6} nis_sum={:.6} nis_max={:.6}",
            label,
            end.full_residual_sum_by_type[ty] - start.full_residual_sum_by_type[ty],
            end.full_residual_abs_sum_by_type[ty] - start.full_residual_abs_sum_by_type[ty],
            end.full_effective_residual_sum_by_type[ty]
                - start.full_effective_residual_sum_by_type[ty],
            end.full_effective_residual_abs_sum_by_type[ty]
                - start.full_effective_residual_abs_sum_by_type[ty],
            end.full_nis_sum_by_type[ty] - start.full_nis_sum_by_type[ty],
            end.full_nis_max_by_type[ty],
        );
    }
    println!(
        "[covhist-interval] full_att_dx_as_reduced total_net_deg=[{:.6},{:.6},{:.6}] total_abs_deg=[{:.6},{:.6},{:.6}]",
        rad_f32_to_deg(end.full_att_dx_sum[0] - start.full_att_dx_sum[0]),
        rad_f32_to_deg(end.full_att_dx_sum[1] - start.full_att_dx_sum[1]),
        rad_f32_to_deg(end.full_att_dx_sum[2] - start.full_att_dx_sum[2]),
        rad_f32_to_deg(end.full_att_dx_abs_sum[0] - start.full_att_dx_abs_sum[0]),
        rad_f32_to_deg(end.full_att_dx_abs_sum[1] - start.full_att_dx_abs_sum[1]),
        rad_f32_to_deg(end.full_att_dx_abs_sum[2] - start.full_att_dx_abs_sum[2]),
    );
    println!(
        "[covhist-interval] full_vel_dx_as_reduced total_net_mps=[{:.6},{:.6},{:.6}] total_abs_mps=[{:.6},{:.6},{:.6}]",
        end.full_vel_dx_sum[0] - start.full_vel_dx_sum[0],
        end.full_vel_dx_sum[1] - start.full_vel_dx_sum[1],
        end.full_vel_dx_sum[2] - start.full_vel_dx_sum[2],
        end.full_vel_dx_abs_sum[0] - start.full_vel_dx_abs_sum[0],
        end.full_vel_dx_abs_sum[1] - start.full_vel_dx_abs_sum[1],
        end.full_vel_dx_abs_sum[2] - start.full_vel_dx_abs_sum[2],
    );
}

fn rad_f32_to_deg(value: f32) -> f64 {
    (value as f64).to_degrees()
}

fn rad_f32_to_dps(value: f32) -> f64 {
    (value as f64).to_degrees()
}

fn delta3(current: [f32; 3], previous: [f32; 3]) -> [f32; 3] {
    [
        current[0] - previous[0],
        current[1] - previous[1],
        current[2] - previous[2],
    ]
}

fn full_obs_type_label(ty: usize) -> &'static str {
    match ty {
        1 => "pos_x",
        2 => "pos_y",
        3 => "pos_z",
        4 => "vel_x",
        5 => "vel_y",
        6 => "vel_z",
        7 => "nhc_y",
        8 => "nhc_z",
        _ => "unknown",
    }
}

fn reduced_gnss_row_label(
    ty: usize,
    pos_xy_count: &mut usize,
    vel_xy_count: &mut usize,
) -> Option<&'static str> {
    match ty {
        0 => {
            let label = match *pos_xy_count {
                0 => "pos_n",
                1 => "pos_e",
                _ => "pos_xy_extra",
            };
            *pos_xy_count += 1;
            Some(label)
        }
        8 => Some("pos_d"),
        1 => {
            let label = match *vel_xy_count {
                0 => "vel_n",
                1 => "vel_e",
                _ => "vel_xy_extra",
            };
            *vel_xy_count += 1;
            Some(label)
        }
        9 => Some("vel_d"),
        _ => None,
    }
}

fn reduced_row_label(
    ty: usize,
    pos_xy_count: &mut usize,
    vel_xy_count: &mut usize,
    zero_xy_count: &mut usize,
) -> &'static str {
    if let Some(label) = reduced_gnss_row_label(ty, pos_xy_count, vel_xy_count) {
        return label;
    }
    match ty {
        2 => {
            let label = match *zero_xy_count {
                0 => "zero_vel_n",
                1 => "zero_vel_e",
                _ => "zero_vel_xy_extra",
            };
            *zero_xy_count += 1;
            label
        }
        4 => "nhc_y",
        5 => "nhc_z",
        10 => "zero_vel_d",
        _ => "unknown",
    }
}

fn scalar_nis(effective_residual: f32, innovation_var: f32) -> f32 {
    if innovation_var > 0.0 {
        effective_residual * effective_residual / innovation_var
    } else {
        f32::NAN
    }
}

fn print_interval_sigma_summary(
    start_reduced: &reduced::State,
    end_reduced: &reduced::State,
    start_full_p: &[[f32; 18]; 18],
    end_full_p: &[[f32; 18]; 18],
) {
    for i in [0usize, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17] {
        println!(
            "[covhist-interval] sigma_deg state={} reduced={:.6}->{:.6} full={:.6}->{:.6}",
            STATE_NAMES[i],
            sigma_deg(&start_reduced.p, i),
            sigma_deg(&end_reduced.p, i),
            sigma_deg(start_full_p, i),
            sigma_deg(end_full_p, i),
        );
    }
}

fn print_interval_correlation_summary(
    start_reduced_p: &[[f32; 18]; 18],
    end_reduced_p: &[[f32; 18]; 18],
    start_full_p: &[[f32; 18]; 18],
    end_full_p: &[[f32; 18]; 18],
) {
    for (label, i, j) in selected_corr_pairs() {
        println!(
            "[covhist-interval] corr {} reduced={:.3}->{:.3} full={:.3}->{:.3}",
            label,
            corr_from_cov(start_reduced_p, i, j),
            corr_from_cov(end_reduced_p, i, j),
            corr_from_cov(start_full_p, i, j),
            corr_from_cov(end_full_p, i, j),
        );
    }
}

fn selected_corr_pairs() -> [(&'static str, usize, usize); 8] {
    [
        ("att_x:mount_roll", 0, 15),
        ("att_x:mount_pitch", 0, 16),
        ("att_y:mount_pitch", 1, 16),
        ("att_y:mount_yaw", 1, 17),
        ("att_z:mount_yaw", 2, 17),
        ("bgx:mount_roll", 9, 15),
        ("bax:mount_yaw", 12, 17),
        ("bay:mount_yaw", 13, 17),
    ]
}

fn print_state_summary(snapshot: &Snapshot, ref_gnss: GenericGnssSample) {
    let reduced_mount_q = snapshot.reduced.as_ref().map(|e| {
        as_q64([
            e.nominal.q_bv0,
            e.nominal.q_bv1,
            e.nominal.q_bv2,
            e.nominal.q_bv3,
        ])
    });
    let full_mount_q = snapshot.full.as_ref().map(|l| {
        as_q64([
            l.nominal.q_bv0,
            l.nominal.q_bv1,
            l.nominal.q_bv2,
            l.nominal.q_bv3,
        ])
    });
    let ref_mount_q = snapshot.reference_mount_q_bv;
    let reduced_mount_qerr = reduced_mount_q
        .zip(ref_mount_q)
        .map(|(a, b)| quat_angle_deg(a, b))
        .unwrap_or(f64::NAN);
    let full_mount_qerr = full_mount_q
        .zip(ref_mount_q)
        .map(|(a, b)| quat_angle_deg(a, b))
        .unwrap_or(f64::NAN);
    let reduced_att_qerr = snapshot
        .reduced
        .as_ref()
        .and_then(|e| snapshot.reference_att_q.map(|r| (e, r)))
        .map(|(e, r)| quat_angle_deg(reduced_att_q(e), r))
        .unwrap_or(f64::NAN);
    let full_att_qerr = snapshot
        .full
        .as_ref()
        .and_then(|l| snapshot.reference_att_q.map(|r| (l, r)))
        .map(|(l, r)| quat_angle_deg(full_att_q_ned(l, ref_gnss), r))
        .unwrap_or(f64::NAN);
    let reduced_nhc = snapshot
        .reduced
        .as_ref()
        .map(reduced_nhc_residual_yz)
        .unwrap_or([f64::NAN; 2]);
    let full_nhc = snapshot
        .full
        .as_ref()
        .map(full_nhc_residual_yz)
        .unwrap_or([f64::NAN; 2]);

    println!(
        "[covhist] state mount_qerr_deg reduced={:.6} full={:.6} att_qerr_deg reduced={:.6} full={:.6}",
        reduced_mount_qerr, full_mount_qerr, reduced_att_qerr, full_att_qerr
    );
    println!(
        "[covhist] nhc_residual_mps reduced_y={:.6} reduced_z={:.6} full_y={:.6} full_z={:.6}",
        reduced_nhc[0], reduced_nhc[1], full_nhc[0], full_nhc[1]
    );
}

fn print_covariance_summary(reduced: &reduced::State, full_as_reduced: &[[f32; 18]; 18]) {
    for (label, rows, cols) in [
        ("att", &[0usize, 1, 2][..], &[0usize, 1, 2][..]),
        ("vel", &[3usize, 4, 5][..], &[3usize, 4, 5][..]),
        ("pos", &[6usize, 7, 8][..], &[6usize, 7, 8][..]),
        ("gyro_bias", &[9usize, 10, 11][..], &[9usize, 10, 11][..]),
        ("accel_bias", &[12usize, 13, 14][..], &[12usize, 13, 14][..]),
        ("mount", &[15usize, 16, 17][..], &[15usize, 16, 17][..]),
        ("att_mount", &[0usize, 1, 2][..], &[15usize, 16, 17][..]),
        ("vel_mount", &[3usize, 4, 5][..], &[15usize, 16, 17][..]),
        (
            "gyro_bias_mount",
            &[9usize, 10, 11][..],
            &[15usize, 16, 17][..],
        ),
        (
            "accel_bias_mount",
            &[12usize, 13, 14][..],
            &[15usize, 16, 17][..],
        ),
    ] {
        println!(
            "[covhist] cov_block label={} rms_abs_diff={:.6e} rms_rel_diff={:.6e}",
            label,
            block_rms_abs_diff(&reduced.p, full_as_reduced, rows, cols),
            block_rms_rel_diff(&reduced.p, full_as_reduced, rows, cols)
        );
    }
    for i in [
        0usize, 1, 2, // attitude
        3, 4, 5, // velocity
        9, 10, 11, // gyro bias
        12, 13, 14, // accel bias
        15, 16, 17, // mount
    ] {
        println!(
            "[covhist] sigma state={} reduced={:.6e} full_as_reduced={:.6e}",
            STATE_NAMES[i],
            reduced.p[i][i].max(0.0).sqrt(),
            full_as_reduced[i][i].max(0.0).sqrt()
        );
    }
}

fn print_mount_correlations(label: &str, p: &[[f32; 18]; 18]) {
    let mut entries = Vec::new();
    for i in 0..15 {
        for j in 15..18 {
            entries.push((corr_from_cov(p, i, j).abs(), i, j, corr_from_cov(p, i, j)));
        }
    }
    entries.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, i, j, corr) in entries.into_iter().take(8) {
        println!(
            "[covhist] top_mount_corr system={} state={} mount={} corr={:.6}",
            label, STATE_NAMES[i], STATE_NAMES[j], corr
        );
    }
}

fn print_update_summary(snapshot: &Snapshot, full: &FullSnapshot) {
    let dy = DIAG_BODY_VEL_Y;
    let dz = DIAG_BODY_VEL_Z;
    println!(
        "[covhist] reduced_update_counts pos_xy={} pos_d={} vel_xy={} vel_d={} zero_xy={} zero_d={} nhc_y={} nhc_z={}",
        snapshot.reduced_type_counts[0],
        snapshot.reduced_type_counts[8],
        snapshot.reduced_type_counts[1],
        snapshot.reduced_type_counts[9],
        snapshot.reduced_type_counts[2],
        snapshot.reduced_type_counts[10],
        snapshot.reduced_type_counts[dy],
        snapshot.reduced_type_counts[dz]
    );
    println!(
        "[covhist] full_obs_counts pos=[{},{},{}] vel=[{},{},{}] nhc_y={} nhc_z={}",
        snapshot.full_obs_counts[1],
        snapshot.full_obs_counts[2],
        snapshot.full_obs_counts[3],
        snapshot.full_obs_counts[4],
        snapshot.full_obs_counts[5],
        snapshot.full_obs_counts[6],
        snapshot.full_obs_counts[7],
        snapshot.full_obs_counts[8]
    );
    println!(
        "[covhist] full_last_obs types={:?} mount_dx_deg=[{:.6},{:.6},{:.6}]",
        full.last_obs_types,
        (full.last_dx[21] as f64).to_degrees(),
        (full.last_dx[22] as f64).to_degrees(),
        (full.last_dx[23] as f64).to_degrees()
    );
    if let Some(reduced) = &snapshot.reduced {
        print_reduced_mount_dx_by_type("sum", reduced);
    }
}

fn print_reduced_mount_dx_by_type(prefix: &str, reduced: &reduced::State) {
    for (label, idx) in selected_reduced_diag_types() {
        println!(
            "[covhist] reduced_mount_dx_{} type={} count={} sum_deg=[{:.6},{:.6},{:.6}] abs_sum_deg=[{:.6},{:.6},{:.6}]",
            prefix,
            label,
            reduced.update_diag.type_counts[idx],
            (reduced.update_diag.sum_dx_mount_roll[idx] as f64).to_degrees(),
            (reduced.update_diag.sum_dx_mount_pitch[idx] as f64).to_degrees(),
            (reduced.update_diag.sum_dx_mount_yaw[idx] as f64).to_degrees(),
            (reduced.update_diag.sum_abs_dx_mount_roll[idx] as f64).to_degrees(),
            (reduced.update_diag.sum_abs_dx_mount_pitch[idx] as f64).to_degrees(),
            (reduced.update_diag.sum_abs_dx_mount_yaw[idx] as f64).to_degrees(),
        );
    }
}

fn selected_reduced_diag_types() -> [(&'static str, usize); 8] {
    [
        ("pos_xy", 0),
        ("pos_d", 8),
        ("vel_xy", 1),
        ("vel_d", 9),
        ("zero_xy", 2),
        ("zero_d", 10),
        ("nhc_y", DIAG_BODY_VEL_Y),
        ("nhc_z", DIAG_BODY_VEL_Z),
    ]
}

#[allow(clippy::too_many_arguments)]
fn maybe_print_trace(
    replay: &Replay,
    ref_gnss: GenericGnssSample,
    trace_window_abs: Option<[f64; 2]>,
    trace_interval_s: f64,
    trace: &mut TraceState,
    event: &str,
    t_s: f64,
    fusion: &SensorFusion,
    full: &full::Filter,
    full_ready: bool,
    full_obs_counts: [u32; 9],
) {
    let Some([start_t_s, end_t_s]) = trace_window_abs else {
        return;
    };
    if t_s < start_t_s || t_s > end_t_s || !full_ready {
        return;
    }
    let Some(reduced) = fusion.reduced() else {
        return;
    };
    let reduced_counts = reduced.update_diag.type_counts;
    if !trace.initialized {
        trace.prev_reduced_counts = reduced_counts;
        trace.prev_full_counts = full_obs_counts;
        trace.prev_reduced_sum_dx_mount_roll = reduced.update_diag.sum_dx_mount_roll;
        trace.prev_reduced_sum_dx_mount_pitch = reduced.update_diag.sum_dx_mount_pitch;
        trace.prev_reduced_sum_dx_mount_yaw = reduced.update_diag.sum_dx_mount_yaw;
        trace.initialized = true;
    }

    let reduced_delta = count_delta(&reduced_counts, &trace.prev_reduced_counts);
    let full_delta = count_delta(&full_obs_counts, &trace.prev_full_counts);
    let reduced_mount_roll_delta = f32_delta(
        &reduced.update_diag.sum_dx_mount_roll,
        &trace.prev_reduced_sum_dx_mount_roll,
    );
    let reduced_mount_pitch_delta = f32_delta(
        &reduced.update_diag.sum_dx_mount_pitch,
        &trace.prev_reduced_sum_dx_mount_pitch,
    );
    let reduced_mount_yaw_delta = f32_delta(
        &reduced.update_diag.sum_dx_mount_yaw,
        &trace.prev_reduced_sum_dx_mount_yaw,
    );
    let update_event = reduced_delta.iter().any(|v| *v > 0) || full_delta.iter().any(|v| *v > 0);
    let periodic = trace
        .last_trace_t_s
        .is_none_or(|last| t_s - last >= trace_interval_s.max(0.0));
    if !update_event && !periodic && event != "gnss" {
        return;
    }

    let full_snap = FullSnapshot {
        nominal: *full.nominal(),
        p: *full.covariance(),
        pos_ecef: full.shadow_pos_ecef(),
        last_dx: *full.last_dx(),
        last_obs_types: full.last_obs_types().to_vec(),
    };
    let p_full_as_reduced = transform_full_cov_to_reduced(&full_snap, ref_gnss);
    let rel_s = t_s - replay_start_t_s(replay).unwrap_or(0.0);
    let ref_mount = reference_mount_at(&replay.reference_mount, t_s);
    let ref_att = reference_attitude_at(&replay.reference_attitude, t_s);
    let reduced_mount_qerr = ref_mount
        .map(|r| {
            quat_angle_deg(
                as_q64([
                    reduced.nominal.q_bv0,
                    reduced.nominal.q_bv1,
                    reduced.nominal.q_bv2,
                    reduced.nominal.q_bv3,
                ]),
                r,
            )
        })
        .unwrap_or(f64::NAN);
    let full_mount_qerr = ref_mount
        .map(|r| {
            quat_angle_deg(
                as_q64([
                    full_snap.nominal.q_bv0,
                    full_snap.nominal.q_bv1,
                    full_snap.nominal.q_bv2,
                    full_snap.nominal.q_bv3,
                ]),
                r,
            )
        })
        .unwrap_or(f64::NAN);
    let reduced_att_qerr = ref_att
        .map(|r| quat_angle_deg(reduced_att_q(reduced), r))
        .unwrap_or(f64::NAN);
    let full_att_qerr = ref_att
        .map(|r| quat_angle_deg(full_att_q_ned(&full_snap, ref_gnss), r))
        .unwrap_or(f64::NAN);
    let reduced_nhc = reduced_nhc_residual_yz(reduced);
    let full_nhc = full_nhc_residual_yz(&full_snap);

    println!(
        "[covhist-trace] rel_s={:.3} event={} update={} d_reduced(pos_xy,pos_d,vel_xy,vel_d,nhc_y,nhc_z)=[{},{},{},{},{},{}] d_full(pos,vel,nhc_y,nhc_z)=[{},{},{},{}] qerr_mount_deg=[{:.3},{:.3}] qerr_att_deg=[{:.3},{:.3}] nhc_yz_reduced=[{:.3},{:.3}] nhc_yz_full=[{:.3},{:.3}]",
        rel_s,
        event,
        update_event,
        reduced_delta[0],
        reduced_delta[8],
        reduced_delta[1],
        reduced_delta[9],
        reduced_delta[DIAG_BODY_VEL_Y],
        reduced_delta[DIAG_BODY_VEL_Z],
        full_delta[1] + full_delta[2] + full_delta[3],
        full_delta[4] + full_delta[5] + full_delta[6],
        full_delta[7],
        full_delta[8],
        reduced_mount_qerr,
        full_mount_qerr,
        reduced_att_qerr,
        full_att_qerr,
        reduced_nhc[0],
        reduced_nhc[1],
        full_nhc[0],
        full_nhc[1],
    );
    println!(
        "[covhist-trace] rel_s={:.3} sig_mount_deg reduced=[{:.3},{:.3},{:.3}] full=[{:.3},{:.3},{:.3}] sig_att_deg reduced=[{:.3},{:.3},{:.3}] full=[{:.3},{:.3},{:.3}]",
        rel_s,
        sigma_deg(&reduced.p, 15),
        sigma_deg(&reduced.p, 16),
        sigma_deg(&reduced.p, 17),
        sigma_deg(&p_full_as_reduced, 15),
        sigma_deg(&p_full_as_reduced, 16),
        sigma_deg(&p_full_as_reduced, 17),
        sigma_deg(&reduced.p, 0),
        sigma_deg(&reduced.p, 1),
        sigma_deg(&reduced.p, 2),
        sigma_deg(&p_full_as_reduced, 0),
        sigma_deg(&p_full_as_reduced, 1),
        sigma_deg(&p_full_as_reduced, 2),
    );
    println!(
        "[covhist-trace] rel_s={:.3} d_reduced_mount_dx_deg type_order=[pos_xy,pos_d,vel_xy,vel_d,nhc_y,nhc_z] roll=[{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}] pitch=[{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}] yaw=[{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}]",
        rel_s,
        (reduced_mount_roll_delta[0] as f64).to_degrees(),
        (reduced_mount_roll_delta[8] as f64).to_degrees(),
        (reduced_mount_roll_delta[1] as f64).to_degrees(),
        (reduced_mount_roll_delta[9] as f64).to_degrees(),
        (reduced_mount_roll_delta[DIAG_BODY_VEL_Y] as f64).to_degrees(),
        (reduced_mount_roll_delta[DIAG_BODY_VEL_Z] as f64).to_degrees(),
        (reduced_mount_pitch_delta[0] as f64).to_degrees(),
        (reduced_mount_pitch_delta[8] as f64).to_degrees(),
        (reduced_mount_pitch_delta[1] as f64).to_degrees(),
        (reduced_mount_pitch_delta[9] as f64).to_degrees(),
        (reduced_mount_pitch_delta[DIAG_BODY_VEL_Y] as f64).to_degrees(),
        (reduced_mount_pitch_delta[DIAG_BODY_VEL_Z] as f64).to_degrees(),
        (reduced_mount_yaw_delta[0] as f64).to_degrees(),
        (reduced_mount_yaw_delta[8] as f64).to_degrees(),
        (reduced_mount_yaw_delta[1] as f64).to_degrees(),
        (reduced_mount_yaw_delta[9] as f64).to_degrees(),
        (reduced_mount_yaw_delta[DIAG_BODY_VEL_Y] as f64).to_degrees(),
        (reduced_mount_yaw_delta[DIAG_BODY_VEL_Z] as f64).to_degrees(),
    );
    println!(
        "[covhist-trace] rel_s={:.3} corr bax_yaw=[{:.3},{:.3}] bay_yaw=[{:.3},{:.3}] attx_mountpitch=[{:.3},{:.3}] atty_mountyaw=[{:.3},{:.3}] bgx_mountroll=[{:.3},{:.3}]",
        rel_s,
        corr_from_cov(&reduced.p, 12, 17),
        corr_from_cov(&p_full_as_reduced, 12, 17),
        corr_from_cov(&reduced.p, 13, 17),
        corr_from_cov(&p_full_as_reduced, 13, 17),
        corr_from_cov(&reduced.p, 0, 16),
        corr_from_cov(&p_full_as_reduced, 0, 16),
        corr_from_cov(&reduced.p, 1, 17),
        corr_from_cov(&p_full_as_reduced, 1, 17),
        corr_from_cov(&reduced.p, 9, 15),
        corr_from_cov(&p_full_as_reduced, 9, 15),
    );

    trace.last_trace_t_s = Some(t_s);
    trace.prev_reduced_counts = reduced_counts;
    trace.prev_full_counts = full_obs_counts;
    trace.prev_reduced_sum_dx_mount_roll = reduced.update_diag.sum_dx_mount_roll;
    trace.prev_reduced_sum_dx_mount_pitch = reduced.update_diag.sum_dx_mount_pitch;
    trace.prev_reduced_sum_dx_mount_yaw = reduced.update_diag.sum_dx_mount_yaw;
}

fn count_delta<const N: usize>(current: &[u32; N], previous: &[u32; N]) -> [u32; N] {
    let mut out = [0u32; N];
    for i in 0..N {
        out[i] = current[i].saturating_sub(previous[i]);
    }
    out
}

fn f32_delta<const N: usize>(current: &[f32; N], previous: &[f32; N]) -> [f32; N] {
    let mut out = [0.0f32; N];
    for i in 0..N {
        out[i] = current[i] - previous[i];
    }
    out
}

fn sigma_deg(p: &[[f32; 18]; 18], idx: usize) -> f64 {
    (p[idx][idx].max(0.0).sqrt() as f64).to_degrees()
}

fn block_trace(p: &[[f32; 18]; 18], start: usize) -> f64 {
    (0..3).map(|i| p[start + i][start + i] as f64).sum()
}

fn block_fro(p: &[[f32; 18]; 18], row_start: usize, col_start: usize) -> f64 {
    let mut sum = 0.0;
    for r in 0..3 {
        for c in 0..3 {
            let v = p[row_start + r][col_start + c] as f64;
            sum += v * v;
        }
    }
    sum.sqrt()
}

fn pvm_block_from_common(p: &[[f32; 18]; 18]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = p[3 + r][15 + c] as f64;
        }
    }
    out
}

fn reduced_mount_error_deg(replay: &Replay, state: &reduced::State) -> [f64; 3] {
    mount_error_deg_from_q_bv(
        replay,
        [
            state.nominal.q_bv0 as f64,
            state.nominal.q_bv1 as f64,
            state.nominal.q_bv2 as f64,
            state.nominal.q_bv3 as f64,
        ],
    )
}

fn full_mount_error_deg(replay: &Replay, state: &FullSnapshot) -> [f64; 3] {
    mount_error_deg_from_q_bv(
        replay,
        [
            state.nominal.q_bv0 as f64,
            state.nominal.q_bv1 as f64,
            state.nominal.q_bv2 as f64,
            state.nominal.q_bv3 as f64,
        ],
    )
}

fn mount_error_deg_from_q_bv(replay: &Replay, q_bv: [f64; 4]) -> [f64; 3] {
    let Some(reference) = final_reference_mount_rpy(replay) else {
        return [f64::NAN; 3];
    };
    let q_ref = reference_mount_rpy_to_q_bv(reference);
    let q_correction = quat_mul(q_ref, quat_conj(q_bv));
    let correction = quat_log_small_angle(q_correction);
    [
        -correction[0].to_degrees(),
        -correction[1].to_degrees(),
        -correction[2].to_degrees(),
    ]
}

fn final_reference_mount_rpy(replay: &Replay) -> Option<[f64; 3]> {
    replay
        .reference_mount
        .iter()
        .rev()
        .find(|sample| {
            sample.roll_deg.is_finite()
                && sample.pitch_deg.is_finite()
                && sample.yaw_deg.is_finite()
        })
        .map(|sample| [sample.roll_deg, sample.pitch_deg, sample.yaw_deg])
}

fn quat_log_small_angle(q: [f64; 4]) -> [f64; 3] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm <= 1.0e-12 {
        return [0.0; 3];
    }
    let mut w = q[0] / norm;
    let mut v = [q[1] / norm, q[2] / norm, q[3] / norm];
    if w < 0.0 {
        w = -w;
        v = [-v[0], -v[1], -v[2]];
    }
    let v_norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if v_norm <= 1.0e-12 {
        return [2.0 * v[0], 2.0 * v[1], 2.0 * v[2]];
    }
    let angle = 2.0 * v_norm.atan2(w);
    [
        angle * v[0] / v_norm,
        angle * v[1] / v_norm,
        angle * v[2] / v_norm,
    ]
}

fn matrix3_add(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][c] + b[r][c];
        }
    }
    out
}

fn matrix3_sub(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][c] - b[r][c];
        }
    }
    out
}

fn matrix3_vec(a: [[f64; 3]; 3], x: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
        a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
        a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
    ]
}

fn matrix3_fro(a: [[f64; 3]; 3]) -> f64 {
    let mut sum = 0.0;
    for row in a {
        for value in row {
            sum += value * value;
        }
    }
    sum.sqrt()
}

fn matrix3_abs_sum(a: [[f64; 3]; 3]) -> f64 {
    let mut sum = 0.0;
    for row in a {
        for value in row {
            sum += value.abs();
        }
    }
    sum
}

fn vec3_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn mount_correction_score(error_deg: [f64; 3], dx_deg: [f64; 3]) -> (f64, f64) {
    if !error_deg.iter().chain(dx_deg.iter()).all(|v| v.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    let dot = error_deg[0] * dx_deg[0] + error_deg[1] * dx_deg[1] + error_deg[2] * dx_deg[2];
    let err_norm =
        (error_deg[0] * error_deg[0] + error_deg[1] * error_deg[1] + error_deg[2] * error_deg[2])
            .sqrt();
    let dx_norm = (dx_deg[0] * dx_deg[0] + dx_deg[1] * dx_deg[1] + dx_deg[2] * dx_deg[2]).sqrt();
    let score = -dot;
    let cos = if err_norm > 1.0e-12 && dx_norm > 1.0e-12 {
        score / (err_norm * dx_norm)
    } else {
        f64::NAN
    };
    (score, cos)
}

fn print_matrix3(label: &str, a: [[f64; 3]; 3]) {
    println!(
        "{label}=[[{:.6e},{:.6e},{:.6e}],[{:.6e},{:.6e},{:.6e}],[{:.6e},{:.6e},{:.6e}]]",
        a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2],
    );
}

fn pvm_phase_idx(phase: &str) -> Option<usize> {
    PVM_PHASES.iter().position(|candidate| *candidate == phase)
}

fn full_pvm_phase_for_obs(obs_type: usize) -> &'static str {
    match obs_type {
        1..=3 => "gnss_position",
        4..=6 => "gnss_velocity",
        7 | 8 => "nhc",
        _ => "reset",
    }
}

fn reduced_pvm_phase_for_obs(obs_type: usize) -> &'static str {
    match obs_type {
        0 | 8 => "gnss_position",
        1 | 9 => "gnss_velocity",
        4 | 5 => "nhc",
        2 | 10 => "zero_velocity",
        _ => "reset",
    }
}

fn full_covariance_row_delta(
    k: &[f32; ERROR_STATES],
    innovation_var: f32,
) -> [[f32; ERROR_STATES]; ERROR_STATES] {
    let mut out = [[0.0; ERROR_STATES]; ERROR_STATES];
    if !innovation_var.is_finite() || innovation_var <= 0.0 {
        return out;
    }
    for i in 0..ERROR_STATES {
        for j in 0..ERROR_STATES {
            out[i][j] = -innovation_var * k[i] * k[j];
        }
    }
    out
}

fn reduced_covariance_row_delta(k: &[f32; 18], innovation_var: f32) -> [[f32; 18]; 18] {
    let mut out = [[0.0; 18]; 18];
    if !innovation_var.is_finite() || innovation_var <= 0.0 {
        return out;
    }
    for i in 0..18 {
        for j in 0..18 {
            out[i][j] = -innovation_var * k[i] * k[j];
        }
    }
    out
}

fn block_rms_abs_diff(
    a: &[[f32; 18]; 18],
    b: &[[f32; 18]; 18],
    rows: &[usize],
    cols: &[usize],
) -> f64 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for &i in rows {
        for &j in cols {
            let d = a[i][j] as f64 - b[i][j] as f64;
            sum += d * d;
            n += 1.0;
        }
    }
    (sum / n).sqrt()
}

fn block_rms_rel_diff(
    a: &[[f32; 18]; 18],
    b: &[[f32; 18]; 18],
    rows: &[usize],
    cols: &[usize],
) -> f64 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for &i in rows {
        for &j in cols {
            let av = a[i][j] as f64;
            let bv = b[i][j] as f64;
            let scale = av.abs().max(bv.abs()).max(1.0e-12);
            let d = (av - bv) / scale;
            sum += d * d;
            n += 1.0;
        }
    }
    (sum / n).sqrt()
}

fn transform_full_cov_to_reduced(
    full: &FullSnapshot,
    ref_gnss: GenericGnssSample,
) -> [[f32; 18]; 18] {
    let mut t = [[0.0f32; ERROR_STATES]; 18];
    let q_ev = as_q64([
        full.nominal.q0,
        full.nominal.q1,
        full.nominal.q2,
        full.nominal.q3,
    ]);
    let c_ev = quat_to_rot(q_ev);
    let pos_ned = ecef_to_ned(
        full.pos_ecef,
        lla_to_ecef(ref_gnss.lat_deg, ref_gnss.lon_deg, ref_gnss.height_m),
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
    );
    let (lat, lon, _) = sim::visualizer::math::ned_to_lla_exact(
        pos_ned[0],
        pos_ned[1],
        pos_ned[2],
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
        ref_gnss.height_m,
    );
    let c_ne = ecef_to_ned_matrix(lat, lon);

    for reduced_i in 0..3 {
        for full_i in 0..3 {
            t[reduced_i][6 + full_i] = c_ev[full_i][reduced_i] as f32;
        }
    }
    for r in 0..3 {
        for c in 0..3 {
            t[3 + r][3 + c] = c_ne[r][c] as f32;
            t[6 + r][c] = c_ne[r][c] as f32;
        }
    }
    for i in 0..3 {
        t[9 + i][12 + i] = -1.0;
        t[12 + i][9 + i] = -1.0;
        t[15 + i][21 + i] = 1.0;
    }

    let mut out = [[0.0f32; 18]; 18];
    for i in 0..18 {
        for j in 0..18 {
            let mut v = 0.0f32;
            for a in 0..ERROR_STATES {
                for b in 0..ERROR_STATES {
                    v += t[i][a] * full.p[a][b] * t[j][b];
                }
            }
            out[i][j] = v;
        }
    }
    out
}

fn transform_full_dx_to_reduced(
    full: &FullSnapshot,
    dx: &[f32; ERROR_STATES],
    ref_gnss: GenericGnssSample,
) -> [f32; 18] {
    let q_ev = as_q64([
        full.nominal.q0,
        full.nominal.q1,
        full.nominal.q2,
        full.nominal.q3,
    ]);
    let c_ev = quat_to_rot(q_ev);
    let pos_ned = ecef_to_ned(
        full.pos_ecef,
        lla_to_ecef(ref_gnss.lat_deg, ref_gnss.lon_deg, ref_gnss.height_m),
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
    );
    let (lat, lon, _) = sim::visualizer::math::ned_to_lla_exact(
        pos_ned[0],
        pos_ned[1],
        pos_ned[2],
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
        ref_gnss.height_m,
    );
    let c_ne = ecef_to_ned_matrix(lat, lon);

    let mut out = [0.0f32; 18];
    for reduced_i in 0..3 {
        for full_i in 0..3 {
            out[reduced_i] += c_ev[full_i][reduced_i] as f32 * dx[6 + full_i];
        }
    }
    for r in 0..3 {
        for c in 0..3 {
            out[3 + r] += c_ne[r][c] as f32 * dx[3 + c];
            out[6 + r] += c_ne[r][c] as f32 * dx[c];
        }
    }
    for i in 0..3 {
        out[9 + i] = -dx[12 + i];
        out[12 + i] = -dx[9 + i];
        out[15 + i] = dx[21 + i];
    }
    out
}

fn transform_full_h_to_reduced(
    full: &FullSnapshot,
    h: &[f32; ERROR_STATES],
    ref_gnss: GenericGnssSample,
) -> [f32; 18] {
    let q_ev = as_q64([
        full.nominal.q0,
        full.nominal.q1,
        full.nominal.q2,
        full.nominal.q3,
    ]);
    let c_ev = quat_to_rot(q_ev);
    let pos_ned = ecef_to_ned(
        full.pos_ecef,
        lla_to_ecef(ref_gnss.lat_deg, ref_gnss.lon_deg, ref_gnss.height_m),
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
    );
    let (lat, lon, _) = sim::visualizer::math::ned_to_lla_exact(
        pos_ned[0],
        pos_ned[1],
        pos_ned[2],
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
        ref_gnss.height_m,
    );
    let c_ne = ecef_to_ned_matrix(lat, lon);

    let mut out = [0.0f32; 18];
    for reduced_i in 0..3 {
        for full_i in 0..3 {
            out[reduced_i] += h[6 + full_i] * c_ev[full_i][reduced_i] as f32;
        }
    }
    for r in 0..3 {
        for c in 0..3 {
            out[3 + r] += h[3 + c] * c_ne[r][c] as f32;
            out[6 + r] += h[c] * c_ne[r][c] as f32;
        }
    }
    for i in 0..3 {
        out[9 + i] = -h[12 + i];
        out[12 + i] = -h[9 + i];
        out[15 + i] = h[21 + i];
    }
    out
}

fn transition_snapshot(
    fusion: &SensorFusion,
    full: &full::Filter,
    curr: GenericImuSample,
    prev: GenericImuSample,
    dt: f64,
    ref_gnss: GenericGnssSample,
) -> Option<TransitionSnapshot> {
    let reduced = fusion.reduced()?;
    if dt <= 0.0 || dt > 1.0 || !dt.is_finite() {
        return None;
    }

    let reduced_imu = reduced::ImuDelta {
        dax: (curr.gyro_radps[0] * dt) as f32,
        day: (curr.gyro_radps[1] * dt) as f32,
        daz: (curr.gyro_radps[2] * dt) as f32,
        dvx: (curr.accel_mps2[0] * dt) as f32,
        dvy: (curr.accel_mps2[1] * dt) as f32,
        dvz: (curr.accel_mps2[2] * dt) as f32,
        dt: dt as f32,
    };
    let f_reduced = generated_reduced::error_transition_with_gravity(
        &reduced.nominal,
        reduced_imu,
        generated_reduced::GRAVITY_MSS,
    )
    .0;
    let full_imu = full_imu_delta_from_vehicle(
        prev.gyro_radps,
        prev.accel_mps2,
        curr.gyro_radps,
        curr.accel_mps2,
        dt,
    );
    let (f_full, _) = generated_full::error_transition(full.nominal(), full_imu);
    let full_snap = FullSnapshot {
        nominal: *full.nominal(),
        p: *full.covariance(),
        pos_ecef: full.shadow_pos_ecef(),
        last_dx: *full.last_dx(),
        last_obs_types: full.last_obs_types().to_vec(),
    };

    let mut out = TransitionSnapshot {
        dt_s: dt as f32,
        reduced_mount_to_att: [[0.0; 3]; 3],
        reduced_mount_to_vel: [[0.0; 3]; 3],
        reduced_mount_to_pos: [[0.0; 3]; 3],
        full_mount_to_att: [[0.0; 3]; 3],
        full_mount_to_vel: [[0.0; 3]; 3],
        full_mount_to_pos: [[0.0; 3]; 3],
    };

    for mount_axis in 0..3 {
        for row in 0..3 {
            out.reduced_mount_to_att[row][mount_axis] = f_reduced[row][15 + mount_axis];
            out.reduced_mount_to_vel[row][mount_axis] = f_reduced[3 + row][15 + mount_axis];
            out.reduced_mount_to_pos[row][mount_axis] = f_reduced[6 + row][15 + mount_axis];
        }

        let mut full_col = [0.0f32; ERROR_STATES];
        for row in 0..ERROR_STATES {
            full_col[row] = f_full[row][21 + mount_axis];
        }
        let full_as_reduced = transform_full_dx_to_reduced(&full_snap, &full_col, ref_gnss);
        for row in 0..3 {
            out.full_mount_to_att[row][mount_axis] = full_as_reduced[row];
            out.full_mount_to_vel[row][mount_axis] = full_as_reduced[3 + row];
            out.full_mount_to_pos[row][mount_axis] = full_as_reduced[6 + row];
        }
    }
    Some(out)
}

fn print_transition_summary(transition: TransitionSnapshot) {
    for (label, reduced, full) in [
        (
            "att",
            transition.reduced_mount_to_att,
            transition.full_mount_to_att,
        ),
        (
            "vel",
            transition.reduced_mount_to_vel,
            transition.full_mount_to_vel,
        ),
        (
            "pos",
            transition.reduced_mount_to_pos,
            transition.full_mount_to_pos,
        ),
    ] {
        println!(
            "[covhist] transition_mount_block block={} dt={:.6} rms_abs_diff={:.6e} reduced_col_norms=[{:.6e},{:.6e},{:.6e}] full_col_norms=[{:.6e},{:.6e},{:.6e}]",
            label,
            transition.dt_s,
            mat3_rms_abs_diff(&reduced, &full),
            col_norm3(&reduced, 0),
            col_norm3(&reduced, 1),
            col_norm3(&reduced, 2),
            col_norm3(&full, 0),
            col_norm3(&full, 1),
            col_norm3(&full, 2),
        );
    }
    for mount_axis in 0..3 {
        println!(
            "[covhist] transition_mount_col axis={} reduced_vel=[{:.6e},{:.6e},{:.6e}] full_vel=[{:.6e},{:.6e},{:.6e}] reduced_pos=[{:.6e},{:.6e},{:.6e}] full_pos=[{:.6e},{:.6e},{:.6e}]",
            ["roll", "pitch", "yaw"][mount_axis],
            transition.reduced_mount_to_vel[0][mount_axis],
            transition.reduced_mount_to_vel[1][mount_axis],
            transition.reduced_mount_to_vel[2][mount_axis],
            transition.full_mount_to_vel[0][mount_axis],
            transition.full_mount_to_vel[1][mount_axis],
            transition.full_mount_to_vel[2][mount_axis],
            transition.reduced_mount_to_pos[0][mount_axis],
            transition.reduced_mount_to_pos[1][mount_axis],
            transition.reduced_mount_to_pos[2][mount_axis],
            transition.full_mount_to_pos[0][mount_axis],
            transition.full_mount_to_pos[1][mount_axis],
            transition.full_mount_to_pos[2][mount_axis],
        );
    }
}

fn print_common_gnss_gain_summary(
    snapshot: &Snapshot,
    replay: &Replay,
    p_reduced: &[[f32; 18]; 18],
    p_full: &[[f32; 18]; 18],
) {
    let Some((gnss, dt_since_gnss)) = nearest_gnss_with_period(&replay.gnss, snapshot.t_s) else {
        return;
    };
    let reduced_vel = snapshot.reduced.map(|state| {
        [
            state.nominal.vn as f64,
            state.nominal.ve as f64,
            state.nominal.vd as f64,
        ]
    });
    let full_vel = snapshot.full.as_ref().map(|full| {
        let pos_ned = ecef_to_ned(
            full.pos_ecef,
            lla_to_ecef(gnss.lat_deg, gnss.lon_deg, gnss.height_m),
            gnss.lat_deg,
            gnss.lon_deg,
        );
        let (lat, lon, _) = sim::visualizer::math::ned_to_lla_exact(
            pos_ned[0],
            pos_ned[1],
            pos_ned[2],
            gnss.lat_deg,
            gnss.lon_deg,
            gnss.height_m,
        );
        let c_ne = ecef_to_ned_matrix(lat, lon);
        mat_vec3_f64(
            c_ne,
            [
                full.nominal.vn as f64,
                full.nominal.ve as f64,
                full.nominal.vd as f64,
            ],
        )
    });
    let pos_r_scale = 1.0 / (dt_since_gnss as f32).clamp(1.0e-3, 1.0);
    let rows = [
        (
            "pos_n",
            6usize,
            gnss.pos_std_m[0] as f32 * gnss.pos_std_m[0] as f32 * pos_r_scale,
            None,
        ),
        (
            "pos_e",
            7usize,
            gnss.pos_std_m[1] as f32 * gnss.pos_std_m[1] as f32 * pos_r_scale,
            None,
        ),
        (
            "vel_n",
            3usize,
            gnss.vel_std_mps[0] as f32 * gnss.vel_std_mps[0] as f32,
            Some(0usize),
        ),
        (
            "vel_e",
            4usize,
            gnss.vel_std_mps[1] as f32 * gnss.vel_std_mps[1] as f32,
            Some(1usize),
        ),
        (
            "vel_d",
            5usize,
            gnss.vel_std_mps[2] as f32 * gnss.vel_std_mps[2] as f32,
            Some(2usize),
        ),
    ];
    for (label, state, r, vel_axis) in rows {
        let reduced_gain = scalar_mount_gain(p_reduced, state, r);
        let full_gain = scalar_mount_gain(p_full, state, r);
        let (reduced_residual, full_residual) = vel_axis
            .and_then(|axis| {
                Some((
                    gnss.vel_ned_mps[axis] - reduced_vel?[axis],
                    gnss.vel_ned_mps[axis] - full_vel?[axis],
                ))
            })
            .unwrap_or((f64::NAN, f64::NAN));
        let reduced_s = p_reduced[state][state] as f64 + r as f64;
        let full_s = p_full[state][state] as f64 + r as f64;
        let full_with_reduced_vel_mount_gain = if (3..=5).contains(&state) && full_s > 0.0 {
            [
                p_reduced[15][state] as f64 / full_s,
                p_reduced[16][state] as f64 / full_s,
                p_reduced[17][state] as f64 / full_s,
            ]
        } else {
            full_gain
        };
        let full_with_reduced_s_gain = if (3..=5).contains(&state) && reduced_s > 0.0 {
            [
                p_full[15][state] as f64 / reduced_s,
                p_full[16][state] as f64 / reduced_s,
                p_full[17][state] as f64 / reduced_s,
            ]
        } else {
            full_gain
        };
        println!(
            "[covhist] common_gnss_mount_gain row={} dt_gnss={:.3} r={:.6e} residual_reduced={:.6e} residual_full={:.6e} s_reduced={:.6e} s_full={:.6e} mount_yaw_dx_deg_reduced={:.6e} mount_yaw_dx_deg_full={:.6e} full_dx_with_reduced_vel_mount={:.6e} full_dx_with_reduced_s={:.6e} reduced=[{:.6e},{:.6e},{:.6e}] full=[{:.6e},{:.6e},{:.6e}] norm_ratio={:.3}",
            label,
            dt_since_gnss,
            r,
            reduced_residual,
            full_residual,
            reduced_s,
            full_s,
            (reduced_gain[2] * reduced_residual).to_degrees(),
            (full_gain[2] * full_residual).to_degrees(),
            (full_with_reduced_vel_mount_gain[2] * full_residual).to_degrees(),
            (full_with_reduced_s_gain[2] * full_residual).to_degrees(),
            reduced_gain[0],
            reduced_gain[1],
            reduced_gain[2],
            full_gain[0],
            full_gain[1],
            full_gain[2],
            norm3(reduced_gain) / norm3(full_gain).max(1.0e-12),
        );
    }
}

fn nearest_gnss_with_period(
    samples: &[GenericGnssSample],
    t_s: f64,
) -> Option<(GenericGnssSample, f64)> {
    if samples.is_empty() {
        return None;
    }
    let idx = samples.partition_point(|s| s.t_s <= t_s);
    let current_idx = idx.saturating_sub(1);
    let current = *samples.get(current_idx)?;
    let period = current_idx
        .checked_sub(1)
        .and_then(|prev| samples.get(prev).map(|sample| current.t_s - sample.t_s))
        .filter(|dt| dt.is_finite() && *dt > 0.0)
        .unwrap_or(1.0);
    Some((current, period))
}

fn scalar_mount_gain(p: &[[f32; 18]; 18], state: usize, r: f32) -> [f64; 3] {
    let s = p[state][state] + r;
    if !(s > 0.0) || !s.is_finite() {
        return [0.0; 3];
    }
    [
        p[15][state] as f64 / s as f64,
        p[16][state] as f64 / s as f64,
        p[17][state] as f64 / s as f64,
    ]
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn mat_vec3_f64(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat3_rms_abs_diff(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> f64 {
    let mut sum = 0.0;
    for row in 0..3 {
        for col in 0..3 {
            let d = a[row][col] as f64 - b[row][col] as f64;
            sum += d * d;
        }
    }
    (sum / 9.0).sqrt()
}

fn col_norm3(a: &[[f32; 3]; 3], col: usize) -> f64 {
    let mut sum = 0.0;
    for row in 0..3 {
        let v = a[row][col] as f64;
        sum += v * v;
    }
    sum.sqrt()
}

fn full_imu_delta_from_vehicle(
    prev_gyro_radps: [f64; 3],
    prev_accel_mps2: [f64; 3],
    curr_gyro_radps: [f64; 3],
    curr_accel_mps2: [f64; 3],
    dt: f64,
) -> full::ImuDelta {
    full::ImuDelta {
        dax_1: (prev_gyro_radps[0] * dt) as f32,
        day_1: (prev_gyro_radps[1] * dt) as f32,
        daz_1: (prev_gyro_radps[2] * dt) as f32,
        dvx_1: (prev_accel_mps2[0] * dt) as f32,
        dvy_1: (prev_accel_mps2[1] * dt) as f32,
        dvz_1: (prev_accel_mps2[2] * dt) as f32,
        dax_2: (curr_gyro_radps[0] * dt) as f32,
        day_2: (curr_gyro_radps[1] * dt) as f32,
        daz_2: (curr_gyro_radps[2] * dt) as f32,
        dvx_2: (curr_accel_mps2[0] * dt) as f32,
        dvy_2: (curr_accel_mps2[1] * dt) as f32,
        dvz_2: (curr_accel_mps2[2] * dt) as f32,
        dt: dt as f32,
    }
}

fn reduced_att_q(reduced: &reduced::State) -> [f64; 4] {
    as_q64([
        reduced.nominal.q0,
        reduced.nominal.q1,
        reduced.nominal.q2,
        reduced.nominal.q3,
    ])
}

fn full_att_q_ned(full: &FullSnapshot, ref_gnss: GenericGnssSample) -> [f64; 4] {
    let pos_ned = ecef_to_ned(
        full.pos_ecef,
        lla_to_ecef(ref_gnss.lat_deg, ref_gnss.lon_deg, ref_gnss.height_m),
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
    );
    let (lat, lon, _) = sim::visualizer::math::ned_to_lla_exact(
        pos_ned[0],
        pos_ned[1],
        pos_ned[2],
        ref_gnss.lat_deg,
        ref_gnss.lon_deg,
        ref_gnss.height_m,
    );
    quat_mul(
        quat_ecef_to_ned(lat, lon),
        as_q64([
            full.nominal.q0,
            full.nominal.q1,
            full.nominal.q2,
            full.nominal.q3,
        ]),
    )
}

fn reference_mount_at(samples: &[GenericReferenceRpySample], t_s: f64) -> Option<[f64; 4]> {
    nearest_rpy(samples, t_s)
        .map(|s| reference_mount_rpy_to_q_bv([s.roll_deg, s.pitch_deg, s.yaw_deg]))
}

fn reference_mount_seed_q_bv(replay: &Replay, mode: VisualizerMountMode) -> Option<[f32; 4]> {
    if !mode.uses_ref_mount() {
        return None;
    }
    replay
        .reference_mount
        .iter()
        .rev()
        .find(|sample| {
            sample.roll_deg.is_finite()
                && sample.pitch_deg.is_finite()
                && sample.yaw_deg.is_finite()
        })
        .map(|sample| {
            let q =
                reference_mount_rpy_to_q_bv([sample.roll_deg, sample.pitch_deg, sample.yaw_deg]);
            [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32]
        })
}

fn reference_attitude_at(samples: &[GenericReferenceRpySample], t_s: f64) -> Option<[f64; 4]> {
    nearest_rpy(samples, t_s)
        .map(|s| sim::eval::gnss_ins::quat_from_rpy_alg_deg(s.roll_deg, s.pitch_deg, s.yaw_deg))
}

fn nearest_rpy(
    samples: &[GenericReferenceRpySample],
    t_s: f64,
) -> Option<GenericReferenceRpySample> {
    if samples.is_empty() {
        return None;
    }
    let idx = samples.partition_point(|s| s.t_s < t_s);
    match (idx.checked_sub(1), samples.get(idx).copied()) {
        (Some(prev), Some(next)) => {
            if (t_s - samples[prev].t_s).abs() <= (next.t_s - t_s).abs() {
                Some(samples[prev])
            } else {
                Some(next)
            }
        }
        (Some(prev), None) => Some(samples[prev]),
        (None, Some(next)) => Some(next),
        (None, None) => None,
    }
}

fn reduced_nhc_residual_yz(reduced: &reduced::State) -> [f64; 2] {
    let v = vehicle_velocity_from_q(
        as_q64([
            reduced.nominal.q0,
            reduced.nominal.q1,
            reduced.nominal.q2,
            reduced.nominal.q3,
        ]),
        [
            reduced.nominal.vn as f64,
            reduced.nominal.ve as f64,
            reduced.nominal.vd as f64,
        ],
    );
    [-v[1], -v[2]]
}

fn full_nhc_residual_yz(full: &FullSnapshot) -> [f64; 2] {
    let (y, _) = generated_full::nhc_y(&full.nominal);
    let (z, _) = generated_full::nhc_z(&full.nominal);
    [-(y as f64), -(z as f64)]
}

fn vehicle_velocity_from_q(q: [f64; 4], v_n: [f64; 3]) -> [f64; 3] {
    let c = quat_to_rot(q);
    [
        c[0][0] * v_n[0] + c[1][0] * v_n[1] + c[2][0] * v_n[2],
        c[0][1] * v_n[0] + c[1][1] * v_n[1] + c[2][1] * v_n[2],
        c[0][2] * v_n[0] + c[1][2] * v_n[1] + c[2][2] * v_n[2],
    ]
}

fn corr_from_cov(p: &[[f32; 18]; 18], i: usize, j: usize) -> f64 {
    let denom = ((p[i][i].max(0.0) as f64) * (p[j][j].max(0.0) as f64)).sqrt();
    if denom <= 1.0e-12 {
        0.0
    } else {
        (p[i][j] as f64 / denom).clamp(-1.0, 1.0)
    }
}

fn ned_vector_to_ecef(lat_deg: f64, lon_deg: f64, v_ned: [f64; 3]) -> [f64; 3] {
    let c_ne = ecef_to_ned_matrix(lat_deg, lon_deg);
    [
        c_ne[0][0] * v_ned[0] + c_ne[1][0] * v_ned[1] + c_ne[2][0] * v_ned[2],
        c_ne[0][1] * v_ned[0] + c_ne[1][1] * v_ned[1] + c_ne[2][1] * v_ned[2],
        c_ne[0][2] * v_ned[0] + c_ne[1][2] * v_ned[1] + c_ne[2][2] * v_ned[2],
    ]
}

fn ecef_to_ned_matrix(lat_deg: f64, lon_deg: f64) -> [[f64; 3]; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    [
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [-clat * clon, -clat * slon, -slat],
    ]
}

fn quat_ecef_to_ned(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    dcm_to_quat(ecef_to_ned_matrix(lat_deg, lon_deg))
}

fn dcm_to_quat(c: [[f64; 3]; 3]) -> [f64; 4] {
    let trace = c[0][0] + c[1][1] + c[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (c[2][1] - c[1][2]) / s,
            (c[0][2] - c[2][0]) / s,
            (c[1][0] - c[0][1]) / s,
        ]
    } else if c[0][0] > c[1][1] && c[0][0] > c[2][2] {
        let s = (1.0 + c[0][0] - c[1][1] - c[2][2]).sqrt() * 2.0;
        [
            (c[2][1] - c[1][2]) / s,
            0.25 * s,
            (c[0][1] + c[1][0]) / s,
            (c[0][2] + c[2][0]) / s,
        ]
    } else if c[1][1] > c[2][2] {
        let s = (1.0 + c[1][1] - c[0][0] - c[2][2]).sqrt() * 2.0;
        [
            (c[0][2] - c[2][0]) / s,
            (c[0][1] + c[1][0]) / s,
            0.25 * s,
            (c[1][2] + c[2][1]) / s,
        ]
    } else {
        let s = (1.0 + c[2][2] - c[0][0] - c[1][1]).sqrt() * 2.0;
        [
            (c[1][0] - c[0][1]) / s,
            (c[0][2] + c[2][0]) / s,
            (c[1][2] + c[2][1]) / s,
            0.25 * s,
        ]
    };
    sim::eval::gnss_ins::quat_normalize(q)
}

fn quat_to_rot(q: [f64; 4]) -> [[f64; 3]; 3] {
    let q = sim::eval::gnss_ins::quat_normalize(q);
    let [q0, q1, q2, q3] = q;
    [
        [
            1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2,
        ],
    ]
}

const STATE_NAMES: [&str; 18] = [
    "att_x",
    "att_y",
    "att_z",
    "vel_n",
    "vel_e",
    "vel_d",
    "pos_n",
    "pos_e",
    "pos_d",
    "bgx",
    "bgy",
    "bgz",
    "bax",
    "bay",
    "baz",
    "mount_roll",
    "mount_pitch",
    "mount_yaw",
];
