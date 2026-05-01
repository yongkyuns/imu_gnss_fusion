use anyhow::Result;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;

use super::model::{EkfImuSource, PlotData, Trace};
use super::pipeline::generic::{
    GenericReplayInput, GenericReplayProgress, build_generic_replay_plot_data,
    build_generic_replay_plot_data_with_progress, parse_generic_replay_csvs_with_refs,
};
use super::pipeline::{EkfCompareConfig, GnssOutageConfig};

pub const WEB_TRANSPORT_MAX_POINTS_PER_TRACE: usize = 6000;

#[derive(Clone, Debug)]
pub struct GenericReplayCsvInputs {
    pub imu_csv: String,
    pub gnss_csv: String,
    pub reference_attitude_csv: Option<String>,
    pub reference_mount_csv: Option<String>,
}

impl GenericReplayCsvInputs {
    pub fn new(imu_csv: impl Into<String>, gnss_csv: impl Into<String>) -> Self {
        Self {
            imu_csv: imu_csv.into(),
            gnss_csv: gnss_csv.into(),
            reference_attitude_csv: None,
            reference_mount_csv: None,
        }
    }

    pub fn with_reference_attitude_csv(mut self, csv: impl Into<String>) -> Self {
        self.reference_attitude_csv = Some(csv.into());
        self
    }

    pub fn with_reference_mount_csv(mut self, csv: impl Into<String>) -> Self {
        self.reference_mount_csv = Some(csv.into());
        self
    }
}

#[derive(Clone, Debug)]
pub struct GenericReplayLabels {
    pub label: String,
    pub imu_name: String,
    pub gnss_name: String,
}

impl Default for GenericReplayLabels {
    fn default() -> Self {
        Self {
            label: "CSV replay".to_string(),
            imu_name: "imu.csv".to_string(),
            gnss_name: "gnss.csv".to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReplayOutputPolicy {
    #[default]
    Full,
    WebTransport {
        max_points_per_trace: usize,
    },
}

impl ReplayOutputPolicy {
    pub fn web_transport() -> Self {
        Self::WebTransport {
            max_points_per_trace: WEB_TRANSPORT_MAX_POINTS_PER_TRACE,
        }
    }

    fn apply(self, data: &mut PlotData) {
        match self {
            Self::Full => {}
            Self::WebTransport {
                max_points_per_trace,
            } => decimate_for_transport(data, max_points_per_trace),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenericReplayJobRequest {
    pub inputs: GenericReplayCsvInputs,
    pub labels: GenericReplayLabels,
    pub ekf_imu_source: EkfImuSource,
    pub ekf_cfg: EkfCompareConfig,
    pub gnss_outages: GnssOutageConfig,
    pub output_policy: ReplayOutputPolicy,
}

#[derive(Clone, Copy, Debug)]
pub struct GenericReplayJobConfig {
    pub misalignment: EkfImuSource,
    pub ekf_cfg: EkfCompareConfig,
    pub gnss_outages: GnssOutageConfig,
    pub output_policy: ReplayOutputPolicy,
}

impl GenericReplayJobConfig {
    pub fn full(
        misalignment: EkfImuSource,
        ekf_cfg: EkfCompareConfig,
        gnss_outages: GnssOutageConfig,
    ) -> Self {
        Self {
            misalignment,
            ekf_cfg,
            gnss_outages,
            output_policy: ReplayOutputPolicy::Full,
        }
    }

    pub fn web_transport() -> Self {
        Self {
            misalignment: EkfImuSource::Internal,
            ekf_cfg: EkfCompareConfig::default(),
            gnss_outages: GnssOutageConfig::default(),
            output_policy: ReplayOutputPolicy::web_transport(),
        }
    }
}

pub struct GenericReplayCsvJob<'a> {
    pub imu_csv: &'a str,
    pub gnss_csv: &'a str,
    pub reference_attitude_csv: Option<&'a str>,
    pub reference_mount_csv: Option<&'a str>,
    pub config: GenericReplayJobConfig,
}

impl GenericReplayJobRequest {
    pub fn new(inputs: GenericReplayCsvInputs) -> Self {
        Self {
            inputs,
            labels: GenericReplayLabels::default(),
            ekf_imu_source: EkfImuSource::Internal,
            ekf_cfg: EkfCompareConfig::default(),
            gnss_outages: GnssOutageConfig::default(),
            output_policy: ReplayOutputPolicy::Full,
        }
    }
}

pub struct GenericReplayJobResult {
    pub labels: GenericReplayLabels,
    pub replay: GenericReplayInput,
    pub plot_data: PlotData,
}

pub fn run_generic_replay_request(
    request: GenericReplayJobRequest,
) -> Result<GenericReplayJobResult> {
    let replay = parse_generic_replay_csvs_with_refs(
        &request.inputs.imu_csv,
        &request.inputs.gnss_csv,
        request.inputs.reference_attitude_csv.as_deref(),
        request.inputs.reference_mount_csv.as_deref(),
    )?;
    let plot_data = build_generic_replay_plot_data_from_input(
        &replay,
        request.ekf_imu_source,
        request.ekf_cfg,
        request.gnss_outages,
        request.output_policy,
    );
    Ok(GenericReplayJobResult {
        labels: request.labels,
        replay,
        plot_data,
    })
}

pub fn run_generic_replay_job(
    replay: &GenericReplayInput,
    config: GenericReplayJobConfig,
) -> PlotData {
    build_generic_replay_plot_data_from_input(
        replay,
        config.misalignment,
        config.ekf_cfg,
        config.gnss_outages,
        config.output_policy,
    )
}

pub fn run_generic_csv_replay_job(job: GenericReplayCsvJob<'_>) -> Result<PlotData> {
    let replay = parse_generic_replay_csvs_with_refs(
        job.imu_csv,
        job.gnss_csv,
        job.reference_attitude_csv,
        job.reference_mount_csv,
    )?;
    Ok(run_generic_replay_job(&replay, job.config))
}

pub fn run_generic_csv_replay_job_with_progress(
    job: GenericReplayCsvJob<'_>,
    progress: &mut dyn FnMut(GenericReplayProgress),
) -> Result<PlotData> {
    let replay = parse_generic_replay_csvs_with_refs(
        job.imu_csv,
        job.gnss_csv,
        job.reference_attitude_csv,
        job.reference_mount_csv,
    )?;
    let mut plot_data = build_generic_replay_plot_data_with_progress(
        &replay,
        job.config.misalignment,
        job.config.ekf_cfg,
        job.config.gnss_outages,
        progress,
    );
    job.config.output_policy.apply(&mut plot_data);
    Ok(plot_data)
}

#[cfg(not(target_arch = "wasm32"))]
pub struct GenericReplayThread {
    job_id: u64,
    receiver: mpsc::Receiver<PlotData>,
}

#[cfg(not(target_arch = "wasm32"))]
pub enum GenericReplayThreadStatus {
    Pending,
    Complete {
        job_id: u64,
        plot_data: Box<PlotData>,
    },
    Disconnected {
        job_id: u64,
    },
}

#[cfg(not(target_arch = "wasm32"))]
impl GenericReplayThread {
    pub fn spawn(job_id: u64, replay: GenericReplayInput, config: GenericReplayJobConfig) -> Self {
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let plot_data = run_generic_replay_job(&replay, config);
            let _ = sender.send(plot_data);
        });
        Self { job_id, receiver }
    }

    pub fn job_id(&self) -> u64 {
        self.job_id
    }

    pub fn poll(&self) -> GenericReplayThreadStatus {
        match self.receiver.try_recv() {
            Ok(plot_data) => GenericReplayThreadStatus::Complete {
                job_id: self.job_id,
                plot_data: Box::new(plot_data),
            },
            Err(mpsc::TryRecvError::Empty) => GenericReplayThreadStatus::Pending,
            Err(mpsc::TryRecvError::Disconnected) => GenericReplayThreadStatus::Disconnected {
                job_id: self.job_id,
            },
        }
    }
}

pub fn parse_and_build_generic_replay_plot_data(
    inputs: &GenericReplayCsvInputs,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
    output_policy: ReplayOutputPolicy,
) -> Result<PlotData> {
    let replay = parse_generic_replay_csvs_with_refs(
        &inputs.imu_csv,
        &inputs.gnss_csv,
        inputs.reference_attitude_csv.as_deref(),
        inputs.reference_mount_csv.as_deref(),
    )?;
    Ok(build_generic_replay_plot_data_from_input(
        &replay,
        ekf_imu_source,
        ekf_cfg,
        gnss_outages,
        output_policy,
    ))
}

pub fn build_generic_replay_plot_data_from_input(
    replay: &GenericReplayInput,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
    output_policy: ReplayOutputPolicy,
) -> PlotData {
    let mut plot_data =
        build_generic_replay_plot_data(replay, ekf_imu_source, ekf_cfg, gnss_outages);
    output_policy.apply(&mut plot_data);
    plot_data
}

pub fn decimate_for_transport(data: &mut PlotData, max_points_per_trace: usize) {
    if max_points_per_trace < 2 {
        return;
    }

    decimate_group(&mut data.speed, max_points_per_trace);
    decimate_group(&mut data.sat_cn0, max_points_per_trace);
    decimate_group(&mut data.imu_raw_gyro, max_points_per_trace);
    decimate_group(&mut data.imu_raw_accel, max_points_per_trace);
    decimate_group(&mut data.imu_cal_gyro, max_points_per_trace);
    decimate_group(&mut data.imu_cal_accel, max_points_per_trace);
    decimate_group(&mut data.orientation, max_points_per_trace);
    decimate_group(&mut data.other, max_points_per_trace);
    decimate_group(&mut data.eskf_cmp_pos, max_points_per_trace);
    decimate_group(&mut data.eskf_cmp_vel, max_points_per_trace);
    decimate_group(&mut data.eskf_cmp_att, max_points_per_trace);
    decimate_group(&mut data.eskf_meas_gyro, max_points_per_trace);
    decimate_group(&mut data.eskf_meas_accel, max_points_per_trace);
    decimate_group(&mut data.eskf_bias_gyro, max_points_per_trace);
    decimate_group(&mut data.eskf_bias_accel, max_points_per_trace);
    decimate_group(&mut data.eskf_cov_bias, max_points_per_trace);
    decimate_group(&mut data.eskf_cov_nonbias, max_points_per_trace);
    decimate_group(&mut data.eskf_mount_sigma, max_points_per_trace);
    decimate_group(&mut data.eskf_mount_dx, max_points_per_trace);
    decimate_group(&mut data.eskf_nhc_mount_dx, max_points_per_trace);
    decimate_group(&mut data.eskf_nhc_innovation, max_points_per_trace);
    decimate_group(&mut data.eskf_nhc_nis, max_points_per_trace);
    decimate_group(&mut data.eskf_nhc_h_mount_norm, max_points_per_trace);
    decimate_group(&mut data.eskf_misalignment, max_points_per_trace);
    decimate_group(&mut data.eskf_stationary_diag, max_points_per_trace);
    decimate_group(&mut data.eskf_bump_pitch_speed, max_points_per_trace);
    decimate_group(&mut data.eskf_bump_diag, max_points_per_trace);
    decimate_group(&mut data.loose_cmp_pos, max_points_per_trace);
    decimate_group(&mut data.loose_cmp_vel, max_points_per_trace);
    decimate_group(&mut data.loose_cmp_att, max_points_per_trace);
    decimate_group(&mut data.loose_nominal_att, max_points_per_trace);
    decimate_group(&mut data.loose_residual_mount, max_points_per_trace);
    decimate_group(&mut data.loose_misalignment, max_points_per_trace);
    decimate_group(&mut data.loose_meas_gyro, max_points_per_trace);
    decimate_group(&mut data.loose_meas_accel, max_points_per_trace);
    decimate_group(&mut data.loose_bias_gyro, max_points_per_trace);
    decimate_group(&mut data.loose_bias_accel, max_points_per_trace);
    decimate_group(&mut data.loose_scale_gyro, max_points_per_trace);
    decimate_group(&mut data.loose_scale_accel, max_points_per_trace);
    decimate_group(&mut data.loose_cov_bias, max_points_per_trace);
    decimate_group(&mut data.loose_cov_nonbias, max_points_per_trace);
    decimate_group(&mut data.loose_mount_sigma, max_points_per_trace);
    decimate_group(&mut data.loose_mount_dx, max_points_per_trace);
    decimate_group(&mut data.align_cmp_att, max_points_per_trace);
    decimate_group(&mut data.align_res_vel, max_points_per_trace);
    decimate_group(&mut data.align_axis_err, max_points_per_trace);
    decimate_group(&mut data.align_motion, max_points_per_trace);
    decimate_group(&mut data.align_flags, max_points_per_trace);
    decimate_group(&mut data.align_roll_contrib, max_points_per_trace);
    decimate_group(&mut data.align_pitch_contrib, max_points_per_trace);
    decimate_group(&mut data.align_yaw_contrib, max_points_per_trace);
    decimate_group(&mut data.align_cov, max_points_per_trace);
}

fn decimate_group(group: &mut [Trace], max_points: usize) {
    for trace in group {
        decimate_trace(trace, max_points);
    }
}

fn decimate_trace(trace: &mut Trace, max_points: usize) {
    if trace.points.len() <= max_points {
        return;
    }

    let step = ((trace.points.len() - 1) as f64 / (max_points - 1) as f64).ceil() as usize;
    let mut points: Vec<[f64; 2]> = trace.points.iter().step_by(step.max(1)).copied().collect();
    if let Some(last) = trace.points.last().copied()
        && points.last().copied() != Some(last)
    {
        points.push(last);
    }
    trace.points = points;
}
