#![allow(clippy::items_after_test_module)]

use anyhow::Result;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;

use super::model::{PlotData, Trace, VisualizerMountMode};
use super::pipeline::generic::{
    GenericReplayInput, GenericReplayProgress, build_generic_replay_plot_data,
    build_generic_replay_plot_data_with_progress, parse_generic_replay_csvs_with_optional_motion,
};
use super::pipeline::{FusionTuningConfig, GnssOutageConfig};

pub const WEB_TRANSPORT_MAX_POINTS_PER_TRACE: usize = 6000;
const WEB_TRANSPORT_DETAIL_POINTS_PER_TRACE: usize = 30000;

#[derive(Clone, Debug)]
pub struct GenericReplayCsvInputs {
    pub imu_csv: String,
    pub gnss_csv: String,
    pub reference_attitude_csv: Option<String>,
    pub reference_mount_csv: Option<String>,
    pub reference_position_csv: Option<String>,
    pub reference_motion_csv: Option<String>,
}

impl GenericReplayCsvInputs {
    pub fn new(imu_csv: impl Into<String>, gnss_csv: impl Into<String>) -> Self {
        Self {
            imu_csv: imu_csv.into(),
            gnss_csv: gnss_csv.into(),
            reference_attitude_csv: None,
            reference_mount_csv: None,
            reference_position_csv: None,
            reference_motion_csv: None,
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

    pub fn with_reference_position_csv(mut self, csv: impl Into<String>) -> Self {
        self.reference_position_csv = Some(csv.into());
        self
    }

    pub fn with_reference_motion_csv(mut self, csv: impl Into<String>) -> Self {
        self.reference_motion_csv = Some(csv.into());
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
    Complete,
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
            Self::Complete => {}
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
    pub mount_mode: VisualizerMountMode,
    pub filter_cfg: FusionTuningConfig,
    pub gnss_outages: GnssOutageConfig,
    pub output_policy: ReplayOutputPolicy,
}

#[derive(Clone, Copy, Debug)]
pub struct GenericReplayJobConfig {
    pub misalignment: VisualizerMountMode,
    pub filter_cfg: FusionTuningConfig,
    pub gnss_outages: GnssOutageConfig,
    pub output_policy: ReplayOutputPolicy,
}

impl GenericReplayJobConfig {
    pub fn complete(
        misalignment: VisualizerMountMode,
        filter_cfg: FusionTuningConfig,
        gnss_outages: GnssOutageConfig,
    ) -> Self {
        Self {
            misalignment,
            filter_cfg,
            gnss_outages,
            output_policy: ReplayOutputPolicy::Complete,
        }
    }

    pub fn web_transport() -> Self {
        Self {
            misalignment: VisualizerMountMode::Auto,
            filter_cfg: FusionTuningConfig::default(),
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
    pub reference_position_csv: Option<&'a str>,
    pub reference_motion_csv: Option<&'a str>,
    pub config: GenericReplayJobConfig,
}

impl GenericReplayJobRequest {
    pub fn new(inputs: GenericReplayCsvInputs) -> Self {
        Self {
            inputs,
            labels: GenericReplayLabels::default(),
            mount_mode: VisualizerMountMode::Auto,
            filter_cfg: FusionTuningConfig::default(),
            gnss_outages: GnssOutageConfig::default(),
            output_policy: ReplayOutputPolicy::Complete,
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
    let replay = parse_generic_replay_csvs_with_optional_motion(
        &request.inputs.imu_csv,
        &request.inputs.gnss_csv,
        request.inputs.reference_attitude_csv.as_deref(),
        request.inputs.reference_mount_csv.as_deref(),
        request.inputs.reference_position_csv.as_deref(),
        request.inputs.reference_motion_csv.as_deref(),
    )?;
    let plot_data = build_generic_replay_plot_data_from_input(
        &replay,
        request.mount_mode,
        request.filter_cfg,
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
        config.filter_cfg,
        config.gnss_outages,
        config.output_policy,
    )
}

pub fn run_generic_csv_replay_job(job: GenericReplayCsvJob<'_>) -> Result<PlotData> {
    let replay = parse_generic_replay_csvs_with_optional_motion(
        job.imu_csv,
        job.gnss_csv,
        job.reference_attitude_csv,
        job.reference_mount_csv,
        job.reference_position_csv,
        job.reference_motion_csv,
    )?;
    Ok(run_generic_replay_job(&replay, job.config))
}

pub fn run_generic_csv_replay_job_with_progress(
    job: GenericReplayCsvJob<'_>,
    progress: &mut dyn FnMut(GenericReplayProgress),
) -> Result<PlotData> {
    let replay = parse_generic_replay_csvs_with_optional_motion(
        job.imu_csv,
        job.gnss_csv,
        job.reference_attitude_csv,
        job.reference_mount_csv,
        job.reference_position_csv,
        job.reference_motion_csv,
    )?;
    let mut plot_data = build_generic_replay_plot_data_with_progress(
        &replay,
        job.config.misalignment,
        job.config.filter_cfg,
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
    mount_mode: VisualizerMountMode,
    filter_cfg: FusionTuningConfig,
    gnss_outages: GnssOutageConfig,
    output_policy: ReplayOutputPolicy,
) -> Result<PlotData> {
    let replay = parse_generic_replay_csvs_with_optional_motion(
        &inputs.imu_csv,
        &inputs.gnss_csv,
        inputs.reference_attitude_csv.as_deref(),
        inputs.reference_mount_csv.as_deref(),
        inputs.reference_position_csv.as_deref(),
        inputs.reference_motion_csv.as_deref(),
    )?;
    Ok(build_generic_replay_plot_data_from_input(
        &replay,
        mount_mode,
        filter_cfg,
        gnss_outages,
        output_policy,
    ))
}

pub fn build_generic_replay_plot_data_from_input(
    replay: &GenericReplayInput,
    mount_mode: VisualizerMountMode,
    filter_cfg: FusionTuningConfig,
    gnss_outages: GnssOutageConfig,
    output_policy: ReplayOutputPolicy,
) -> PlotData {
    let mut plot_data =
        build_generic_replay_plot_data(replay, mount_mode, filter_cfg, gnss_outages);
    output_policy.apply(&mut plot_data);
    plot_data
}

pub fn decimate_for_transport(data: &mut PlotData, max_points_per_trace: usize) {
    if max_points_per_trace < 2 {
        return;
    }

    decimate_group(&mut data.speed, max_points_per_trace);
    decimate_group(&mut data.vehicle_motion_gyro, max_points_per_trace);
    decimate_group(&mut data.vehicle_motion_accel, max_points_per_trace);
    decimate_group(&mut data.sat_cn0, max_points_per_trace);
    decimate_group(&mut data.imu_raw_gyro, max_points_per_trace);
    decimate_group(&mut data.imu_raw_accel, max_points_per_trace);
    decimate_group(&mut data.imu_cal_gyro, max_points_per_trace);
    decimate_group(&mut data.imu_cal_accel, max_points_per_trace);
    decimate_group(&mut data.orientation, max_points_per_trace);
    decimate_group(&mut data.other, max_points_per_trace);
    let detail_max_points = max_points_per_trace.max(WEB_TRANSPORT_DETAIL_POINTS_PER_TRACE);

    decimate_group(&mut data.ekf_cmp_pos, detail_max_points);
    decimate_group(&mut data.ekf_cmp_vel, detail_max_points);
    decimate_group(&mut data.ekf_cmp_att, detail_max_points);
    decimate_group(&mut data.ekf_meas_gyro, max_points_per_trace);
    decimate_group(&mut data.ekf_meas_accel, max_points_per_trace);
    decimate_group(&mut data.ekf_bias_gyro, max_points_per_trace);
    decimate_group(&mut data.ekf_bias_accel, max_points_per_trace);
    decimate_group(&mut data.ekf_cov_bias, max_points_per_trace);
    decimate_group(&mut data.ekf_cov_nonbias, max_points_per_trace);
    decimate_group(&mut data.ekf_mount_sigma, max_points_per_trace);
    decimate_group(&mut data.ekf_mount_dx, max_points_per_trace);
    decimate_group(&mut data.ekf_nhc_mount_dx, max_points_per_trace);
    decimate_group(&mut data.ekf_nhc_innovation, max_points_per_trace);
    decimate_group(&mut data.ekf_nhc_nis, max_points_per_trace);
    decimate_group(&mut data.ekf_nhc_h_mount_norm, max_points_per_trace);
    decimate_group(&mut data.ekf_misalignment, detail_max_points);
    decimate_group(&mut data.ekf_stationary_diag, max_points_per_trace);
    decimate_group(&mut data.ekf_bump_pitch_speed, max_points_per_trace);
    decimate_group(&mut data.ekf_bump_diag, max_points_per_trace);
    decimate_group(&mut data.align_cmp_att, max_points_per_trace);
    decimate_group(&mut data.align_res_vel, max_points_per_trace);
    decimate_group(&mut data.align_axis_err, max_points_per_trace);
    decimate_group(&mut data.align_motion, max_points_per_trace);
    decimate_group(&mut data.align_flags, max_points_per_trace);
    decimate_group(&mut data.align_roll_contrib, max_points_per_trace);
    decimate_group(&mut data.align_pitch_contrib, max_points_per_trace);
    decimate_group(&mut data.align_yaw_contrib, max_points_per_trace);
    decimate_group(&mut data.align_cov, max_points_per_trace);
    decimate_update_inspector(&mut data.update_inspector, max_points_per_trace.min(2000));
}

fn decimate_group(group: &mut [Trace], max_points: usize) {
    for trace in group {
        decimate_trace(trace, max_points);
    }
}

fn decimate_trace(trace: &mut Trace, max_points: usize) {
    if trace.points.len() <= max_points || max_points < 2 {
        return;
    }

    if trace
        .points
        .iter()
        .all(|point| point[0].is_finite() && point[1].is_finite())
    {
        trace.points = decimate_finite_trace_segment(&trace.points, max_points);
        return;
    }

    let finite_total = trace
        .points
        .iter()
        .filter(|point| point[0].is_finite() && point[1].is_finite())
        .count();
    if finite_total == 0 {
        trace.points.truncate(max_points);
        return;
    }

    let mut decimated = Vec::with_capacity(max_points);
    let mut finite_remaining = finite_total;
    let mut budget_remaining = max_points;
    let mut segment_start = 0usize;
    while segment_start < trace.points.len() {
        while segment_start < trace.points.len()
            && (!trace.points[segment_start][0].is_finite()
                || !trace.points[segment_start][1].is_finite())
        {
            if decimated.len() < max_points {
                push_unique_point(&mut decimated, trace.points[segment_start]);
                budget_remaining = max_points.saturating_sub(decimated.len());
            }
            segment_start += 1;
        }
        if segment_start >= trace.points.len() {
            break;
        }

        let mut segment_end = segment_start;
        while segment_end < trace.points.len()
            && trace.points[segment_end][0].is_finite()
            && trace.points[segment_end][1].is_finite()
        {
            segment_end += 1;
        }

        let segment_len = segment_end - segment_start;
        if budget_remaining < 2 {
            break;
        }
        let segment_budget = if finite_remaining == segment_len {
            budget_remaining
        } else {
            ((segment_len as f64 / finite_remaining as f64) * budget_remaining as f64)
                .round()
                .clamp(2.0, budget_remaining as f64) as usize
        };
        let segment = decimate_finite_trace_segment(
            &trace.points[segment_start..segment_end],
            segment_budget,
        );
        decimated.extend(segment);
        finite_remaining = finite_remaining.saturating_sub(segment_len);
        budget_remaining = max_points.saturating_sub(decimated.len());

        segment_start = segment_end;
    }

    if let Some(last) = trace.points.last().copied() {
        if decimated.len() < max_points {
            push_unique_point(&mut decimated, last);
        } else if decimated.last().copied() != Some(last) {
            decimated[max_points - 1] = last;
        }
    }
    trace.points = decimated;
}

fn decimate_finite_trace_segment(points: &[[f64; 2]], max_points: usize) -> Vec<[f64; 2]> {
    if points.len() <= max_points {
        return points.to_vec();
    }
    if max_points < 4 {
        let step = ((points.len() - 1) as f64 / (max_points - 1) as f64).ceil() as usize;
        let mut out: Vec<_> = points.iter().step_by(step.max(1)).copied().collect();
        if let Some(last) = points.last().copied() {
            if out.len() < max_points {
                push_unique_point(&mut out, last);
            } else if out.last().copied() != Some(last) {
                out[max_points - 1] = last;
            }
        }
        return out;
    }

    let x0 = points.first().map(|point| point[0]).unwrap_or(0.0);
    let x1 = points.last().map(|point| point[0]).unwrap_or(0.0);
    let span = x1 - x0;
    if span.abs() <= f64::EPSILON {
        return decimate_finite_trace_segment_by_stride(points, max_points);
    }

    let bucket_count = (max_points / 6).max(1);
    let mut buckets = vec![TraceBucket::default(); bucket_count];
    for (idx, point) in points.iter().enumerate() {
        let mut bucket = (((point[0] - x0) / span) * bucket_count as f64).floor() as usize;
        if bucket >= bucket_count {
            bucket = bucket_count - 1;
        }
        buckets[bucket].visit(idx, point[1]);
    }

    let mut out = Vec::with_capacity(max_points);
    for bucket in buckets {
        let mut indices = bucket.indices();
        indices.sort_unstable();
        indices.dedup();
        for idx in indices {
            if out.len() >= max_points {
                break;
            }
            push_unique_point(&mut out, points[idx]);
        }
        if out.len() >= max_points {
            break;
        }
    }
    if let Some(last) = points.last().copied() {
        if out.len() < max_points {
            push_unique_point(&mut out, last);
        } else if out.last().copied() != Some(last) {
            out[max_points - 1] = last;
        }
    }
    out
}

fn decimate_finite_trace_segment_by_stride(
    points: &[[f64; 2]],
    max_points: usize,
) -> Vec<[f64; 2]> {
    let step = ((points.len() - 1) as f64 / (max_points - 1) as f64).ceil() as usize;
    let mut out: Vec<_> = points.iter().step_by(step.max(1)).copied().collect();
    if let Some(last) = points.last().copied() {
        if out.len() < max_points {
            push_unique_point(&mut out, last);
        } else if out.last().copied() != Some(last) {
            out[max_points - 1] = last;
        }
    }
    out
}

fn push_unique_point(points: &mut Vec<[f64; 2]>, point: [f64; 2]) {
    if points.last().copied() != Some(point) {
        points.push(point);
    }
}

#[derive(Clone, Copy, Default)]
struct TraceBucket {
    first_idx: Option<usize>,
    last_idx: Option<usize>,
    min_y: Option<(usize, f64)>,
    max_y: Option<(usize, f64)>,
}

impl TraceBucket {
    fn visit(&mut self, idx: usize, y: f64) {
        self.first_idx.get_or_insert(idx);
        self.last_idx = Some(idx);
        match self.min_y {
            Some((_, min_y)) if y >= min_y => {}
            _ => self.min_y = Some((idx, y)),
        }
        match self.max_y {
            Some((_, max_y)) if y <= max_y => {}
            _ => self.max_y = Some((idx, y)),
        }
    }

    fn indices(self) -> Vec<usize> {
        [
            self.first_idx,
            self.first_idx
                .zip(self.last_idx)
                .map(|(first, last)| first + (last - first) / 3),
            self.min_y.map(|(idx, _)| idx),
            self.max_y.map(|(idx, _)| idx),
            self.first_idx
                .zip(self.last_idx)
                .map(|(first, last)| first + 2 * (last - first) / 3),
            self.last_idx,
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_decimation_preserves_local_extrema_and_boundaries() {
        let mut trace = Trace {
            name: "test".to_string(),
            points: (0..1000)
                .map(|i| {
                    let x = i as f64 * 0.01;
                    let y = if (4.0..=6.0).contains(&x) {
                        (x * 18.0).sin() * 20.0 + x
                    } else {
                        x
                    };
                    [x, y]
                })
                .collect(),
        };

        decimate_trace(&mut trace, 120);

        assert!(trace.points.len() <= 120);
        assert_eq!(trace.points.first().copied(), Some([0.0, 0.0]));
        let last = trace.points.last().copied().expect("trace is not empty");
        assert!((last[0] - 9.99).abs() < 1.0e-12);
        assert!((last[1] - 9.99).abs() < 1.0e-12);
        assert!(
            trace
                .points
                .iter()
                .any(|point| (4.0..=6.0).contains(&point[0]) && point[1] > 20.0),
            "positive transient extrema should survive transport decimation"
        );
        assert!(
            trace
                .points
                .iter()
                .any(|point| (4.0..=6.0).contains(&point[0]) && point[1] < -10.0),
            "negative transient extrema should survive transport decimation"
        );
    }

    #[test]
    fn transport_decimation_keeps_nan_breaks_between_finite_segments() {
        let mut trace = Trace {
            name: "test".to_string(),
            points: (0..300)
                .map(|i| [i as f64 * 0.01, i as f64])
                .chain([[f64::NAN, f64::NAN]])
                .chain((300..600).map(|i| [i as f64 * 0.01, i as f64]))
                .collect(),
        };

        decimate_trace(&mut trace, 80);

        assert!(trace.points.iter().any(|point| !point[0].is_finite()));
        let nan_idx = trace
            .points
            .iter()
            .position(|point| !point[0].is_finite())
            .expect("decimated trace should retain explicit gap marker");
        assert!(nan_idx > 0);
        assert!(nan_idx + 1 < trace.points.len());
        assert!(trace.points[nan_idx - 1][0] < 3.0);
        assert!(trace.points[nan_idx + 1][0] >= 3.0);
    }

    #[test]
    fn web_transport_keeps_higher_resolution_for_comparison_traces() {
        let dense_trace = || Trace {
            name: "dense".to_string(),
            points: (0..100_000)
                .map(|i| [i as f64 * 0.01, (i as f64 * 0.001).sin()])
                .collect(),
        };
        let mut data = PlotData {
            ekf_cmp_att: vec![dense_trace()],
            ekf_meas_gyro: vec![dense_trace()],
            ..PlotData::default()
        };

        decimate_for_transport(&mut data, WEB_TRANSPORT_MAX_POINTS_PER_TRACE);

        assert!(data.ekf_cmp_att[0].points.len() > WEB_TRANSPORT_MAX_POINTS_PER_TRACE);
        assert!(data.ekf_cmp_att[0].points.len() <= WEB_TRANSPORT_DETAIL_POINTS_PER_TRACE);
        assert!(data.ekf_meas_gyro[0].points.len() <= WEB_TRANSPORT_MAX_POINTS_PER_TRACE);
    }
}

fn decimate_update_inspector(
    samples: &mut Vec<super::model::UpdateInspectorSample>,
    max_samples: usize,
) {
    if samples.len() <= max_samples || max_samples < 2 {
        return;
    }

    let step = ((samples.len() - 1) as f64 / (max_samples - 1) as f64).ceil() as usize;
    let mut decimated: Vec<_> = samples.iter().step_by(step.max(1)).cloned().collect();
    if let Some(last) = samples.last().cloned()
        && decimated
            .last()
            .map(|sample| sample.t_s != last.t_s || sample.filter != last.filter)
            .unwrap_or(true)
    {
        decimated.push(last);
    }
    *samples = decimated;
}
