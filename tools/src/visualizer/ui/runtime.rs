//! App construction, runtime lifecycle, replay refresh, and frame update orchestration.

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};
#[cfg(not(target_arch = "wasm32"))]
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use eframe::egui;
#[cfg(not(target_arch = "wasm32"))]
use flate2::read::GzDecoder;
use walkers::MapMemory;

#[cfg(not(target_arch = "wasm32"))]
use crate::datasets::generic_replay::{
    load_gnss_samples, load_imu_samples, load_reference_attitude_samples,
    load_reference_motion_samples, load_reference_mount_samples, load_reference_position_samples,
};
use crate::visualizer::model::{Page, PlotData, VisualizerFusionBackend, VisualizerMountMode};
#[cfg(not(target_arch = "wasm32"))]
use crate::visualizer::pipeline::generic::GenericReplayInput;
#[cfg(not(target_arch = "wasm32"))]
use crate::visualizer::pipeline::synthetic::SyntheticVisualizerConfig;
use crate::visualizer::pipeline::synthetic::build_synthetic_plot_data_with_backend;
use crate::visualizer::stats::map_center_from_traces;
use crate::visualizer::theme::{UiDensity, UiTheme};

#[cfg(not(target_arch = "wasm32"))]
use super::MAPBOX_ACCESS_TOKEN_ENV;
#[cfg(not(target_arch = "wasm32"))]
use super::input::{
    DATASET_MANIFEST_PATH, HostedDatasetEntry, HostedDatasetManifest, NativeRealDataSource,
};
use super::input::{InputMode, SyntheticNoise, SyntheticScenario};
use super::maps::map_tiles_from_token;
use super::state::{DataOrigin, TraceVisibility};
#[cfg(target_arch = "wasm32")]
use super::web::{
    WEB_MAX_POINTS_PER_TRACE, WebDatasetState, WebPerf, WebRealDataSource,
    web_initial_mapbox_token, web_initial_ui_theme, web_query_flag, web_query_synthetic_noise,
    web_query_synthetic_scenario, web_remember_ui_theme,
};
use super::{App, ReplayState};
#[cfg(not(target_arch = "wasm32"))]
use super::{NativeReplayTask, NativeReplayTaskResult};

#[cfg(not(target_arch = "wasm32"))]
fn initial_ui_theme() -> UiTheme {
    std::env::var("IMU_GNSS_FUSION_THEME")
        .ok()
        .and_then(|value| UiTheme::from_value(&value))
        .unwrap_or_default()
}

#[cfg(target_arch = "wasm32")]
fn initial_ui_theme() -> UiTheme {
    web_initial_ui_theme()
}

fn current_ui_density() -> UiDensity {
    #[cfg(target_arch = "wasm32")]
    {
        UiDensity::Compact
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        UiDensity::Comfortable
    }
}

pub(super) fn create_app(
    cc: &eframe::CreationContext<'_>,
    data: PlotData,
    has_itow: bool,
    replay: Option<ReplayState>,
) -> App {
    let map_center = map_center_from_traces(&data.ekf_map);
    #[cfg(not(target_arch = "wasm32"))]
    let mapbox_access_token = std::env::var(MAPBOX_ACCESS_TOKEN_ENV).unwrap_or_default();
    #[cfg(target_arch = "wasm32")]
    let mapbox_access_token = web_initial_mapbox_token();
    let ui_theme = initial_ui_theme();
    let data_origin = if replay
        .as_ref()
        .and_then(|replay| replay.synthetic.as_ref())
        .is_some()
    {
        DataOrigin::Synthetic
    } else {
        DataOrigin::Real
    };
    let map_tiles = map_tiles_from_token(&mapbox_access_token, ui_theme, cc.egui_ctx.clone());
    let mut map_memory = MapMemory::default();
    let _ = map_memory.set_zoom(15.0);
    #[cfg(target_arch = "wasm32")]
    let initial_max_points_per_trace = WEB_MAX_POINTS_PER_TRACE;
    #[cfg(not(target_arch = "wasm32"))]
    let initial_max_points_per_trace = 2500;

    #[cfg_attr(not(target_arch = "wasm32"), allow(unused_mut))]
    let tuning_cfg = replay
        .as_ref()
        .map(|replay| replay.filter_cfg)
        .unwrap_or_default();
    let tuning_gnss_outages = replay
        .as_ref()
        .map(|replay| replay.gnss_outages)
        .unwrap_or_default();
    let tuning_misalignment = replay
        .as_ref()
        .map(|replay| replay.misalignment)
        .unwrap_or(VisualizerMountMode::Auto);
    let tuning_backend = replay
        .as_ref()
        .map(|replay| replay.backend)
        .unwrap_or(VisualizerFusionBackend::Rust);
    #[cfg(not(target_arch = "wasm32"))]
    let native_input_mode = if replay
        .as_ref()
        .and_then(|replay| replay.synthetic.as_ref())
        .is_some()
    {
        InputMode::Synthetic
    } else {
        InputMode::RealData
    };
    #[cfg(not(target_arch = "wasm32"))]
    let native_generic_replay_dir = replay
        .as_ref()
        .and_then(|replay| replay.generic_replay_dir.as_ref())
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "target/replay-parity/parking-lot-figure8-nominal-001".to_string());
    #[cfg(not(target_arch = "wasm32"))]
    let native_datasets = load_native_dataset_manifest().unwrap_or_default();
    #[cfg(not(target_arch = "wasm32"))]
    let native_real_data_source = if native_datasets.is_empty() {
        NativeRealDataSource::CustomDirectory
    } else {
        NativeRealDataSource::ManifestDataset
    };
    #[cfg(not(target_arch = "wasm32"))]
    let native_selected_dataset = replay
        .as_ref()
        .and_then(|replay| replay.generic_replay_dir.as_ref())
        .and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .and_then(|name| {
            native_datasets
                .iter()
                .position(|dataset| dataset.id.as_deref() == Some(name))
        })
        .unwrap_or(0);
    #[cfg(not(target_arch = "wasm32"))]
    let native_scenario = SyntheticScenario::CityBlocks;
    #[cfg(not(target_arch = "wasm32"))]
    let native_synthetic_noise = replay
        .as_ref()
        .and_then(|replay| replay.synthetic.as_ref())
        .map(|synthetic| synthetic.noise_mode)
        .map(SyntheticNoise::from)
        .unwrap_or(SyntheticNoise::Truth);

    #[cfg_attr(not(target_arch = "wasm32"), allow(unused_mut))]
    let mut app = App {
        data,
        ghost_data: None,
        current_run_key: replay.as_ref().map(replay_run_key),
        #[cfg(target_arch = "wasm32")]
        pending_run_key: None,
        has_itow,
        fps_ema: 0.0,
        last_frame_time_s: 0.0,
        max_points_per_trace: initial_max_points_per_trace,
        ui_theme,
        data_origin,
        page: Page::Overview,
        map_tiles,
        map_memory,
        map_center,
        show_reference: true,
        show_align: true,
        show_heading: false,
        show_gnss_map: true,
        show_ekf: true,
        show_events: true,
        map_color_source: super::maps::MapColorSource::None,
        event_visibility: super::state::EventVisibility::default(),
        shared_cursor_t_s: None,
        update_inspector_cursor_t_s: None,
        show_update_inspector: false,
        tuning_cfg,
        tuning_gnss_outages,
        tuning_misalignment,
        tuning_backend,
        tuning_panel: None,
        replay,
        replay_status: None,
        #[cfg(not(target_arch = "wasm32"))]
        native_input_mode,
        #[cfg(not(target_arch = "wasm32"))]
        native_generic_replay_dir,
        #[cfg(not(target_arch = "wasm32"))]
        native_real_data_source,
        #[cfg(not(target_arch = "wasm32"))]
        native_datasets,
        #[cfg(not(target_arch = "wasm32"))]
        native_selected_dataset,
        #[cfg(not(target_arch = "wasm32"))]
        native_scenario,
        #[cfg(not(target_arch = "wasm32"))]
        native_synthetic_noise,
        #[cfg(not(target_arch = "wasm32"))]
        native_replay_task: None,
        #[cfg(not(target_arch = "wasm32"))]
        native_replay_job_id: 0,
        #[cfg(not(target_arch = "wasm32"))]
        native_run_progress: 0.0,
        #[cfg(not(target_arch = "wasm32"))]
        native_run_started_time_s: 0.0,
        #[cfg(not(target_arch = "wasm32"))]
        native_run_estimated_duration_s: 1.0,
        #[cfg(target_arch = "wasm32")]
        web_imu_csv: None,
        #[cfg(target_arch = "wasm32")]
        web_gnss_csv: None,
        #[cfg(target_arch = "wasm32")]
        web_reference_attitude_csv: None,
        #[cfg(target_arch = "wasm32")]
        web_reference_mount_csv: None,
        #[cfg(target_arch = "wasm32")]
        web_reference_position_csv: None,
        #[cfg(target_arch = "wasm32")]
        web_reference_motion_csv: None,
        #[cfg(target_arch = "wasm32")]
        web_mapbox_token: mapbox_access_token.clone(),
        #[cfg(target_arch = "wasm32")]
        web_mapbox_token_applied: mapbox_access_token,
        #[cfg(target_arch = "wasm32")]
        show_mapbox_token_window: false,
        #[cfg(target_arch = "wasm32")]
        web_scenario: SyntheticScenario::CityBlocks,
        #[cfg(target_arch = "wasm32")]
        web_synthetic_noise: web_query_synthetic_noise().unwrap_or(SyntheticNoise::Truth),
        #[cfg(target_arch = "wasm32")]
        web_input_mode: InputMode::Synthetic,
        #[cfg(target_arch = "wasm32")]
        web_real_data_source: WebRealDataSource::DroppedCsv,
        #[cfg(target_arch = "wasm32")]
        web_datasets: WebDatasetState::new(),
        #[cfg(target_arch = "wasm32")]
        web_run_progress: 0.0,
        #[cfg(target_arch = "wasm32")]
        web_run_started_time_s: 0.0,
        #[cfg(target_arch = "wasm32")]
        web_run_estimated_duration_s: 1.0,
        #[cfg(target_arch = "wasm32")]
        web_status: "Drag imu.csv and gnss.csv onto the app, or run a built-in synthetic scenario."
            .to_string(),
        #[cfg(target_arch = "wasm32")]
        web_perf: WebPerf {
            enabled: web_query_flag("bench"),
            ..WebPerf::default()
        },
    };
    #[cfg(target_arch = "wasm32")]
    let auto_load_dataset = app.web_datasets.auto_load_id.is_some();
    #[cfg(target_arch = "wasm32")]
    if auto_load_dataset {
        app.web_input_mode = InputMode::RealData;
        app.web_real_data_source = WebRealDataSource::ManifestDataset;
    }
    #[cfg(target_arch = "wasm32")]
    if !auto_load_dataset {
        if let Some(scenario) = web_query_synthetic_scenario() {
            app.web_scenario = scenario;
        }
        app.refresh_from_web_synthetic(&cc.egui_ctx);
    }
    #[cfg(target_arch = "wasm32")]
    app.start_web_manifest_load(&cc.egui_ctx);
    app
}

impl App {
    pub(super) fn replace_plot_data(&mut self, data: PlotData, run_key: Option<String>) {
        self.ghost_data = match (&self.current_run_key, &run_key) {
            (Some(current), Some(next)) if current == next && self.data.has_trace_points() => {
                Some(self.data.clone())
            }
            _ => None,
        };
        self.current_run_key = run_key;
        self.data = data;
    }

    pub(super) fn trace_visibility(&self) -> TraceVisibility {
        TraceVisibility {
            show_reference: self.show_reference,
            show_align: self.show_align,
            show_ekf: self.show_ekf,
        }
    }

    pub(super) fn set_ui_theme(&mut self, theme: UiTheme, ctx: &egui::Context) {
        if self.ui_theme == theme {
            return;
        }
        self.ui_theme = theme;
        super::super::theme::apply(ctx, current_ui_density(), self.ui_theme);
        self.refresh_map_tiles(ctx);
        #[cfg(target_arch = "wasm32")]
        web_remember_ui_theme(self.ui_theme);
    }

    pub(super) fn refresh_map_tiles(&mut self, ctx: &egui::Context) {
        #[cfg(target_arch = "wasm32")]
        let token = self.web_mapbox_token.clone();
        #[cfg(not(target_arch = "wasm32"))]
        let token = std::env::var(MAPBOX_ACCESS_TOKEN_ENV).unwrap_or_default();
        self.map_tiles = map_tiles_from_token(&token, self.ui_theme, ctx.clone());
    }

    pub(super) fn refresh_from_replay(&mut self, ctx: &egui::Context) {
        #[cfg(target_arch = "wasm32")]
        let _ = ctx;
        let Some(replay) = self.replay.as_ref() else {
            return;
        };
        let misalignment = self.tuning_misalignment;
        let backend = self.tuning_backend;
        let filter_cfg = self.tuning_cfg;
        let gnss_outages = self.tuning_gnss_outages;
        if let Some(synthetic) = &replay.synthetic {
            let mut next_replay = replay.clone();
            next_replay.backend = backend;
            next_replay.misalignment = misalignment;
            next_replay.filter_cfg = filter_cfg;
            next_replay.gnss_outages = gnss_outages;
            match build_synthetic_plot_data_with_backend(
                synthetic,
                backend,
                misalignment,
                filter_cfg,
                gnss_outages,
            ) {
                Ok(mut data) => {
                    crate::visualizer::replay_job::decimate_for_transport(
                        &mut data,
                        crate::visualizer::replay_job::WEB_TRANSPORT_MAX_POINTS_PER_TRACE,
                    );
                    self.replace_plot_data(data, Some(replay_run_key(&next_replay)));
                    self.map_center = map_center_from_traces(&self.data.ekf_map);
                    self.has_itow = false;
                    self.data_origin = DataOrigin::Synthetic;
                    self.replay = Some(next_replay);
                    self.replay_status = Some(format!(
                        "Synthetic replay refreshed with {} backend",
                        backend.label()
                    ));
                }
                Err(err) => {
                    self.replay_status = Some(format!("Synthetic replay failed: {err}"));
                }
            }
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(dataset_id) = replay.hosted_dataset_id.clone() {
                if let Some(idx) = self
                    .native_datasets
                    .iter()
                    .position(|dataset| dataset.id.as_deref() == Some(dataset_id.as_str()))
                {
                    self.native_real_data_source = NativeRealDataSource::ManifestDataset;
                    self.native_selected_dataset = idx;
                    self.start_native_hosted_dataset(ctx);
                    return;
                }
                self.replay_status = Some(format!(
                    "Hosted dataset '{dataset_id}' is not in the manifest."
                ));
                return;
            }
            if let Some(generic_replay_dir) = replay.generic_replay_dir.clone() {
                self.start_native_generic_replay(generic_replay_dir, ctx);
                return;
            }
        }

        self.replay_status = Some("No replay source is available to rerun.".to_string());
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn run_native_selected_input(&mut self, ctx: &egui::Context) {
        match self.native_input_mode {
            InputMode::Synthetic => {
                self.start_native_synthetic(ctx);
            }
            InputMode::RealData => match self.native_real_data_source {
                NativeRealDataSource::CustomDirectory => {
                    let path = PathBuf::from(self.native_generic_replay_dir.trim());
                    self.start_native_generic_replay(path, ctx);
                }
                NativeRealDataSource::ManifestDataset => {
                    self.start_native_hosted_dataset(ctx);
                }
            },
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_native_synthetic(&mut self, ctx: &egui::Context) {
        let scenario = self.native_scenario;
        let (motion_label, motion_text) = scenario.scenario_text();
        let (early_vel_bias_ned_mps, early_fault_window_s) = scenario.early_fault();
        let synth_cfg = SyntheticVisualizerConfig {
            motion_def: None,
            motion_label: motion_label.to_string(),
            motion_text: Some(motion_text.to_string()),
            noise_mode: self.native_synthetic_noise.into(),
            disable_imu_noise: false,
            disable_gnss_noise: false,
            seed: 1,
            mount_rpy_deg: scenario.mount_rpy_deg(),
            imu_hz: 100.0,
            gnss_hz: 2.0,
            gnss_time_shift_ms: 0.0,
            early_vel_bias_ned_mps,
            early_fault_window_s: early_fault_window_s.map(|[start_s, end_s]| (start_s, end_s)),
        };
        let backend = self.tuning_backend;
        let misalignment = self.tuning_misalignment;
        let filter_cfg = self.tuning_cfg;
        let gnss_outages = self.tuning_gnss_outages;
        let label = scenario.display_label().to_string();
        self.start_native_replay_task(ctx, format!("Running replay: {label}"), 4.0, move || {
            match build_synthetic_plot_data_with_backend(
                &synth_cfg,
                backend,
                misalignment,
                filter_cfg,
                gnss_outages,
            ) {
                Ok(mut data) => {
                    crate::visualizer::replay_job::decimate_for_transport(
                        &mut data,
                        crate::visualizer::replay_job::WEB_TRANSPORT_MAX_POINTS_PER_TRACE,
                    );
                    let replay = ReplayState {
                        bytes: Vec::new(),
                        generic_replay_dir: None,
                        hosted_dataset_id: None,
                        synthetic: Some(synth_cfg),
                        max_records: None,
                        misalignment,
                        backend,
                        filter_cfg,
                        gnss_outages,
                    };
                    Ok((
                        data,
                        replay,
                        DataOrigin::Synthetic,
                        format!(
                            "Synthetic scenario loaded with {} backend: {label}",
                            backend.label()
                        ),
                    ))
                }
                Err(err) => Err(format!("Synthetic replay failed for {label}: {err}")),
            }
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_native_generic_replay(&mut self, replay_dir: PathBuf, ctx: &egui::Context) {
        if replay_dir.as_os_str().is_empty() {
            self.replay_status = Some("Generic replay directory is empty.".to_string());
            return;
        }
        let backend = self.tuning_backend;
        let misalignment = self.tuning_misalignment;
        let filter_cfg = self.tuning_cfg;
        let gnss_outages = self.tuning_gnss_outages;
        let label = replay_dir.display().to_string();
        self.start_native_replay_task(ctx, format!("Running replay: {label}"), 6.0, move || {
            let replay_input = load_generic_replay_dir(&replay_dir)
                .map_err(|err| format!("Generic replay failed for {label}: {err}"))?;
            let data = crate::visualizer::replay_job::run_generic_replay_job(
                &replay_input,
                crate::visualizer::replay_job::GenericReplayJobConfig {
                    backend,
                    output_policy: crate::visualizer::replay_job::ReplayOutputPolicy::web_transport(
                    ),
                    ..crate::visualizer::replay_job::GenericReplayJobConfig::complete(
                        misalignment,
                        filter_cfg,
                        gnss_outages,
                    )
                },
            );
            let replay = ReplayState {
                bytes: replay_dir.display().to_string().into_bytes(),
                generic_replay_dir: Some(replay_dir.clone()),
                hosted_dataset_id: None,
                synthetic: None,
                max_records: None,
                misalignment,
                backend,
                filter_cfg,
                gnss_outages,
            };
            Ok((
                data,
                replay,
                DataOrigin::Real,
                format!(
                    "Generic replay loaded with {} backend: {label}",
                    backend.label()
                ),
            ))
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_native_hosted_dataset(&mut self, ctx: &egui::Context) {
        if self.native_datasets.is_empty() {
            self.replay_status = Some(format!("No datasets in {DATASET_MANIFEST_PATH}."));
            return;
        }
        let selected = self
            .native_selected_dataset
            .min(self.native_datasets.len().saturating_sub(1));
        self.native_selected_dataset = selected;
        let dataset = self.native_datasets[selected].clone();
        let label = dataset.display_label();
        let backend = self.tuning_backend;
        let misalignment = self.tuning_misalignment;
        let filter_cfg = self.tuning_cfg;
        let gnss_outages = self.tuning_gnss_outages;
        self.start_native_replay_task(ctx, format!("Running replay: {label}"), 6.0, move || {
            let csvs = load_native_hosted_dataset_csvs(&dataset)
                .map_err(|err| format!("Dataset load failed for {label}: {err}"))?;
            let data = crate::visualizer::replay_job::run_generic_csv_replay_job(
                crate::visualizer::replay_job::GenericReplayCsvJob {
                    imu_csv: &csvs.imu,
                    gnss_csv: &csvs.gnss,
                    reference_attitude_csv: csvs.reference_attitude.as_deref(),
                    reference_mount_csv: csvs.reference_mount.as_deref(),
                    reference_position_csv: csvs.reference_position.as_deref(),
                    reference_motion_csv: csvs.reference_motion.as_deref(),
                    config: crate::visualizer::replay_job::GenericReplayJobConfig {
                        backend,
                        output_policy:
                            crate::visualizer::replay_job::ReplayOutputPolicy::web_transport(),
                        ..crate::visualizer::replay_job::GenericReplayJobConfig::complete(
                            misalignment,
                            filter_cfg,
                            gnss_outages,
                        )
                    },
                },
            )
            .map_err(|err| format!("Dataset replay failed for {label}: {err}"))?;
            let dataset_id = dataset.id.as_deref().unwrap_or(label.as_str());
            let replay = ReplayState {
                bytes: dataset_id.as_bytes().to_vec(),
                generic_replay_dir: None,
                hosted_dataset_id: Some(dataset_id.to_string()),
                synthetic: None,
                max_records: None,
                misalignment,
                backend,
                filter_cfg,
                gnss_outages,
            };
            Ok((
                data,
                replay,
                DataOrigin::Real,
                format!("Dataset loaded with {} backend: {label}", backend.label()),
            ))
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn start_native_replay_task<F>(
        &mut self,
        ctx: &egui::Context,
        status: String,
        estimated_duration_s: f64,
        build: F,
    ) where
        F: FnOnce() -> Result<(PlotData, ReplayState, DataOrigin, String), String> + Send + 'static,
    {
        let (sender, receiver) = std::sync::mpsc::channel();
        self.native_replay_job_id = self.native_replay_job_id.wrapping_add(1);
        let job_id = self.native_replay_job_id;
        self.native_replay_task = Some(NativeReplayTask { job_id, receiver });
        self.native_run_progress = 0.02;
        self.native_run_started_time_s = ctx.input(|i| i.time);
        self.native_run_estimated_duration_s = estimated_duration_s.max(0.5);
        self.replay_status = Some(status);
        thread::spawn(move || {
            let result = match build() {
                Ok((plot_data, replay, origin, status)) => NativeReplayTaskResult::Complete {
                    plot_data: Box::new(plot_data),
                    replay,
                    origin,
                    status,
                },
                Err(status) => NativeReplayTaskResult::Failed { status },
            };
            let _ = sender.send(result);
        });
        ctx.request_repaint();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn advance_native_run_progress(&mut self, ctx: &egui::Context) {
        if self.native_replay_task.is_some() {
            let elapsed_s = (ctx.input(|i| i.time) - self.native_run_started_time_s).max(0.0);
            let estimated = (elapsed_s / self.native_run_estimated_duration_s.max(0.5)) as f32;
            self.native_run_progress = self.native_run_progress.max(estimated.min(0.95));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll_native_replay_task(&mut self, ctx: &egui::Context) {
        let Some(task) = self.native_replay_task.as_ref() else {
            return;
        };
        match task.receiver.try_recv() {
            Ok(NativeReplayTaskResult::Complete {
                plot_data,
                replay,
                origin,
                status,
            }) => {
                self.native_run_progress = 1.0;
                self.replace_plot_data(*plot_data, Some(replay_run_key(&replay)));
                self.map_center = map_center_from_traces(&self.data.ekf_map);
                self.has_itow = false;
                self.data_origin = origin;
                self.replay = Some(replay);
                self.replay_status = Some(status);
                self.native_replay_task = None;
            }
            Ok(NativeReplayTaskResult::Failed { status }) => {
                self.replay_status = Some(status);
                self.native_replay_task = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                ctx.request_repaint();
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.replay_status = Some(format!(
                    "Replay worker disconnected before completing job {}.",
                    task.job_id
                ));
                self.native_replay_task = None;
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn draw_native_bulk_loading_page(&self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered_justified(|ui| {
                ui.add_space((ui.available_height() * 0.35).max(24.0));
                ui.heading("Building replay");
                ui.add_space(8.0);
                ui.add(
                    egui::ProgressBar::new(self.native_run_progress.clamp(0.0, 1.0))
                        .desired_width((ui.available_width() * 0.45).clamp(220.0, 520.0))
                        .text(format!(
                            "{:.0}%",
                            100.0 * self.native_run_progress.clamp(0.0, 1.0)
                        )),
                );
                ui.add_space(6.0);
                if let Some(status) = &self.replay_status {
                    ui.label(status);
                }
            });
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeHostedDatasetCsvs {
    imu: String,
    gnss: String,
    reference_attitude: Option<String>,
    reference_mount: Option<String>,
    reference_position: Option<String>,
    reference_motion: Option<String>,
}

#[cfg(not(target_arch = "wasm32"))]
fn load_native_dataset_manifest() -> anyhow::Result<Vec<HostedDatasetEntry>> {
    let text = fs::read_to_string(native_dataset_manifest_path())?;
    let manifest: HostedDatasetManifest = serde_json::from_str(&text)?;
    Ok(manifest.datasets)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_native_hosted_dataset_csvs(
    entry: &HostedDatasetEntry,
) -> anyhow::Result<NativeHostedDatasetCsvs> {
    Ok(NativeHostedDatasetCsvs {
        imu: load_native_dataset_csv(entry, "imu", true)?
            .ok_or_else(|| anyhow::anyhow!("missing imu.csv or imu.csv.gz"))?,
        gnss: load_native_dataset_csv(entry, "gnss", true)?
            .ok_or_else(|| anyhow::anyhow!("missing gnss.csv or gnss.csv.gz"))?,
        reference_attitude: load_native_dataset_csv(entry, "reference_attitude", false)?,
        reference_mount: load_native_dataset_csv(entry, "reference_mount", false)?,
        reference_position: load_native_dataset_csv(entry, "reference_position", false)?,
        reference_motion: load_native_dataset_csv(entry, "reference_motion", false)?,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn load_native_dataset_csv(
    entry: &HostedDatasetEntry,
    kind: &str,
    required: bool,
) -> anyhow::Result<Option<String>> {
    let (plain, gz) = match kind {
        "imu" => (entry.imu.as_deref(), entry.imu_gz.as_deref()),
        "gnss" => (entry.gnss.as_deref(), entry.gnss_gz.as_deref()),
        "reference_attitude" => (
            entry.reference_attitude.as_deref(),
            entry.reference_attitude_gz.as_deref(),
        ),
        "reference_mount" => (
            entry.reference_mount.as_deref(),
            entry.reference_mount_gz.as_deref(),
        ),
        "reference_position" => (
            entry.reference_position.as_deref(),
            entry.reference_position_gz.as_deref(),
        ),
        "reference_motion" => (
            entry.reference_motion.as_deref(),
            entry.reference_motion_gz.as_deref(),
        ),
        _ => anyhow::bail!("unsupported dataset file kind: {kind}"),
    };

    if let Some(path) = gz {
        return read_native_dataset_file(entry, path, true).map(Some);
    }
    if let Some(path) = plain {
        return read_native_dataset_file(entry, path, path.ends_with(".gz")).map(Some);
    }
    if !required {
        return Ok(None);
    }

    let gz_fallback = format!("{kind}.csv.gz");
    match read_native_dataset_file(entry, &gz_fallback, true) {
        Ok(text) => Ok(Some(text)),
        Err(gz_err) => {
            let csv_fallback = format!("{kind}.csv");
            read_native_dataset_file(entry, &csv_fallback, false)
                .map(Some)
                .map_err(|csv_err| anyhow::anyhow!("{gz_err}; fallback {csv_err}"))
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn read_native_dataset_file(
    entry: &HostedDatasetEntry,
    relative_path: &str,
    gzipped: bool,
) -> anyhow::Result<String> {
    if relative_path.starts_with("http://") || relative_path.starts_with("https://") {
        anyhow::bail!("native visualizer expects local hosted datasets, got {relative_path}");
    }
    let manifest_path = native_dataset_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let path = manifest_dir
        .join(entry.base_url.as_deref().unwrap_or_default())
        .join(relative_path.trim_start_matches('/'));
    let bytes = fs::read(&path)?;
    if gzipped {
        let mut decoder = GzDecoder::new(bytes.as_slice());
        let mut text = String::new();
        decoder.read_to_string(&mut text)?;
        Ok(text)
    } else {
        Ok(String::from_utf8(bytes)?)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_dataset_manifest_path() -> PathBuf {
    let cwd_relative = PathBuf::from(DATASET_MANIFEST_PATH);
    if cwd_relative.exists() {
        return cwd_relative;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(DATASET_MANIFEST_PATH)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_generic_replay_dir(dir: &Path) -> anyhow::Result<GenericReplayInput> {
    Ok(GenericReplayInput {
        imu: load_imu_samples(dir)?,
        gnss: load_gnss_samples(dir)?,
        reference_attitude: load_reference_attitude_samples(dir)?,
        reference_mount: load_reference_mount_samples(dir)?,
        reference_position: load_reference_position_samples(dir)?,
        reference_motion: load_reference_motion_samples(dir)?,
    })
}

pub(super) fn replay_run_key(replay: &ReplayState) -> String {
    if let Some(synthetic) = &replay.synthetic {
        return format!(
            "synthetic:{synthetic:?}:max={:?}:backend={:?}",
            replay.max_records, replay.backend
        );
    }
    let mut hasher = DefaultHasher::new();
    replay.bytes.hash(&mut hasher);
    replay.max_records.hash(&mut hasher);
    replay.backend.hash(&mut hasher);
    format!("csv:{:016x}", hasher.finish())
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        super::super::theme::apply(ctx, current_ui_density(), self.ui_theme);

        #[cfg(target_arch = "wasm32")]
        self.consume_dropped_files(ctx);

        #[cfg(target_arch = "wasm32")]
        self.poll_web_dataset_tasks(ctx);
        #[cfg(target_arch = "wasm32")]
        self.advance_web_run_progress();
        #[cfg(target_arch = "wasm32")]
        self.poll_web_replay_tasks();

        #[cfg(target_arch = "wasm32")]
        self.publish_web_perf(ctx);
        #[cfg(not(target_arch = "wasm32"))]
        self.advance_native_run_progress(ctx);
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_native_replay_task(ctx);

        #[cfg(target_os = "macos")]
        if ctx.input(|i| i.viewport().close_requested()) {
            std::process::exit(0);
        }

        #[cfg(target_arch = "wasm32")]
        if self.web_datasets.loading_manifest
            || self.web_datasets.loading_dataset
            || self.web_datasets.loading_replay
        {
            ctx.request_repaint();
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.native_replay_task.is_some() {
            ctx.request_repaint();
        }
        #[cfg(not(target_arch = "wasm32"))]
        if matches!(self.page, Page::Overview) {
            ctx.request_repaint_after(Duration::from_millis(16));
        }

        self.draw_top_controls(ctx);
        self.draw_tuning_window(ctx);
        self.draw_update_inspector_window(ctx);
        #[cfg(target_arch = "wasm32")]
        self.draw_mapbox_token_window(ctx);

        #[cfg(target_arch = "wasm32")]
        if self.web_datasets.loading_dataset || self.web_datasets.loading_replay {
            self.draw_web_bulk_loading_page(ctx);
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.native_replay_task.is_some() {
            self.draw_native_bulk_loading_page(ctx);
            return;
        }

        self.draw_current_page(ctx);
    }
}
