use anyhow::Result;
#[cfg(not(target_arch = "wasm32"))]
use eframe::egui;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
use walkers::{HttpTiles, MapMemory};

use super::model::{Page, PlotData, VisualizerFusionBackend, VisualizerMountMode};
use super::pipeline::synthetic::SyntheticVisualizerConfig;
use super::pipeline::{FusionTuningConfig, GnssOutageConfig};
use super::theme::UiTheme;

mod colors;
mod controls;
mod input;
mod inspector;
mod maps;
mod orthogonal;
mod pages;
mod plots;
mod runtime;
mod state;
mod trace_query;
mod tuning;
#[cfg(target_arch = "wasm32")]
mod web;
mod windows;

#[cfg(not(target_arch = "wasm32"))]
use input::{HostedDatasetEntry, NativeRealDataSource};
use input::{InputMode, SyntheticNoise, SyntheticScenario};
use runtime::create_app;
use state::{DataOrigin, TuningPanel};
#[cfg(target_arch = "wasm32")]
use web::{NamedText, WebDatasetState, WebPerf, WebRealDataSource};

#[cfg(not(target_arch = "wasm32"))]
pub(super) const MAPBOX_ACCESS_TOKEN_ENV: &str = "MAPBOX_ACCESS_TOKEN";
const SYNTHETIC_TRAJECTORY_MAX_POINTS: usize = 2_000;
const LOG_Y_FLOOR: f64 = 1.0e-6;
pub struct App {
    data: PlotData,
    ghost_data: Option<PlotData>,
    current_run_key: Option<String>,
    #[cfg(target_arch = "wasm32")]
    pending_run_key: Option<String>,
    has_itow: bool,
    fps_ema: f32,
    last_frame_time_s: f64,
    max_points_per_trace: usize,
    ui_theme: UiTheme,
    data_origin: DataOrigin,
    page: Page,
    map_tiles: HttpTiles,
    map_memory: MapMemory,
    map_center: walkers::Position,
    show_reference: bool,
    show_align: bool,
    show_heading: bool,
    show_gnss_map: bool,
    show_ekf: bool,
    show_events: bool,
    map_color_source: maps::MapColorSource,
    event_visibility: state::EventVisibility,
    shared_cursor_t_s: Option<f64>,
    update_inspector_cursor_t_s: Option<f64>,
    show_update_inspector: bool,
    tuning_cfg: FusionTuningConfig,
    tuning_gnss_outages: GnssOutageConfig,
    tuning_misalignment: VisualizerMountMode,
    tuning_backend: VisualizerFusionBackend,
    tuning_panel: Option<TuningPanel>,
    replay: Option<ReplayState>,
    replay_status: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    native_input_mode: InputMode,
    #[cfg(not(target_arch = "wasm32"))]
    native_generic_replay_dir: String,
    #[cfg(not(target_arch = "wasm32"))]
    native_real_data_source: NativeRealDataSource,
    #[cfg(not(target_arch = "wasm32"))]
    native_datasets: Vec<HostedDatasetEntry>,
    #[cfg(not(target_arch = "wasm32"))]
    native_selected_dataset: usize,
    #[cfg(not(target_arch = "wasm32"))]
    native_scenario: SyntheticScenario,
    #[cfg(not(target_arch = "wasm32"))]
    native_synthetic_noise: SyntheticNoise,
    #[cfg(not(target_arch = "wasm32"))]
    native_replay_task: Option<NativeReplayTask>,
    #[cfg(not(target_arch = "wasm32"))]
    native_replay_job_id: u64,
    #[cfg(not(target_arch = "wasm32"))]
    native_run_progress: f32,
    #[cfg(not(target_arch = "wasm32"))]
    native_run_started_time_s: f64,
    #[cfg(not(target_arch = "wasm32"))]
    native_run_estimated_duration_s: f64,
    #[cfg(target_arch = "wasm32")]
    web_imu_csv: Option<NamedText>,
    #[cfg(target_arch = "wasm32")]
    web_gnss_csv: Option<NamedText>,
    #[cfg(target_arch = "wasm32")]
    web_reference_attitude_csv: Option<NamedText>,
    #[cfg(target_arch = "wasm32")]
    web_reference_mount_csv: Option<NamedText>,
    #[cfg(target_arch = "wasm32")]
    web_reference_position_csv: Option<NamedText>,
    #[cfg(target_arch = "wasm32")]
    web_reference_motion_csv: Option<NamedText>,
    #[cfg(target_arch = "wasm32")]
    web_mapbox_token: String,
    #[cfg(target_arch = "wasm32")]
    web_mapbox_token_applied: String,
    #[cfg(target_arch = "wasm32")]
    show_mapbox_token_window: bool,
    #[cfg(target_arch = "wasm32")]
    web_scenario: SyntheticScenario,
    #[cfg(target_arch = "wasm32")]
    web_synthetic_noise: SyntheticNoise,
    #[cfg(target_arch = "wasm32")]
    web_input_mode: InputMode,
    #[cfg(target_arch = "wasm32")]
    web_real_data_source: WebRealDataSource,
    #[cfg(target_arch = "wasm32")]
    web_datasets: WebDatasetState,
    #[cfg(target_arch = "wasm32")]
    web_run_progress: f32,
    #[cfg(target_arch = "wasm32")]
    web_run_started_time_s: f64,
    #[cfg(target_arch = "wasm32")]
    web_run_estimated_duration_s: f64,
    #[cfg(target_arch = "wasm32")]
    web_status: String,
    #[cfg(target_arch = "wasm32")]
    web_perf: WebPerf,
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeReplayTask {
    job_id: u64,
    receiver: mpsc::Receiver<NativeReplayTaskResult>,
}

#[cfg(not(target_arch = "wasm32"))]
enum NativeReplayTaskResult {
    Complete {
        plot_data: Box<PlotData>,
        replay: ReplayState,
        origin: DataOrigin,
        status: String,
    },
    Failed {
        status: String,
    },
}

#[derive(Clone)]
pub struct ReplayState {
    pub bytes: Vec<u8>,
    #[cfg(not(target_arch = "wasm32"))]
    pub generic_replay_dir: Option<PathBuf>,
    #[cfg(not(target_arch = "wasm32"))]
    pub hosted_dataset_id: Option<String>,
    pub synthetic: Option<SyntheticVisualizerConfig>,
    pub max_records: Option<usize>,
    pub misalignment: VisualizerMountMode,
    pub backend: VisualizerFusionBackend,
    pub filter_cfg: FusionTuningConfig,
    pub gnss_outages: GnssOutageConfig,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_visualizer(data: PlotData, has_itow: bool, replay: Option<ReplayState>) -> Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 950.0]),
        renderer: eframe::Renderer::Glow,
        run_and_return: false,
        ..Default::default()
    };
    eframe::run_native(
        "IMU/GNSS Filter Evaluation",
        native_options,
        Box::new(move |cc| Ok(Box::new(create_app(cc, data, has_itow, replay)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub async fn run_visualizer_web(
    runner: &eframe::WebRunner,
    canvas: eframe::web_sys::HtmlCanvasElement,
    data: PlotData,
    has_itow: bool,
) -> std::result::Result<(), eframe::wasm_bindgen::JsValue> {
    runner
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(move |cc| Ok(Box::new(create_app(cc, data, has_itow, None)))),
        )
        .await
}

#[cfg(target_arch = "wasm32")]
pub fn run_visualizer(
    _data: PlotData,
    _has_itow: bool,
    _replay: Option<ReplayState>,
) -> Result<()> {
    anyhow::bail!("run_visualizer is native-only on wasm; use run_visualizer_web instead")
}
