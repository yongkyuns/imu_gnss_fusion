use anyhow::Result;
#[cfg(not(target_arch = "wasm32"))]
use eframe::egui;
use walkers::{HttpTiles, MapMemory};

use super::model::{Page, PlotData, VisualizerMountMode};
use super::pipeline::synthetic::SyntheticVisualizerConfig;
use super::pipeline::{FilterCompareConfig, GnssOutageConfig};
use super::theme::UiTheme;

mod colors;
mod controls;
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

use runtime::create_app;
use state::{DataOrigin, TuningPanel};
#[cfg(target_arch = "wasm32")]
use web::{
    NamedText, WebDatasetState, WebInputMode, WebPerf, WebRealDataSource, WebSyntheticNoise,
    WebSyntheticScenario,
};

#[cfg(not(target_arch = "wasm32"))]
pub(super) const MAPBOX_ACCESS_TOKEN_ENV: &str = "MAPBOX_ACCESS_TOKEN";
const SYNTHETIC_TRAJECTORY_MAX_POINTS: usize = 2_000;
const LOG_Y_FLOOR: f64 = 1.0e-6;
pub struct App {
    data: PlotData,
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
    show_reduced: bool,
    show_full: bool,
    shared_cursor_t_s: Option<f64>,
    update_inspector_cursor_t_s: Option<f64>,
    show_update_inspector: bool,
    tuning_cfg: FilterCompareConfig,
    tuning_gnss_outages: GnssOutageConfig,
    tuning_misalignment: VisualizerMountMode,
    tuning_panel: Option<TuningPanel>,
    replay: Option<ReplayState>,
    replay_status: Option<String>,
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
    web_scenario: WebSyntheticScenario,
    #[cfg(target_arch = "wasm32")]
    web_synthetic_noise: WebSyntheticNoise,
    #[cfg(target_arch = "wasm32")]
    web_input_mode: WebInputMode,
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

#[derive(Clone)]
pub struct ReplayState {
    pub bytes: Vec<u8>,
    pub synthetic: Option<SyntheticVisualizerConfig>,
    pub max_records: Option<usize>,
    pub misalignment: VisualizerMountMode,
    pub filter_cfg: FilterCompareConfig,
    pub gnss_outages: GnssOutageConfig,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_visualizer(data: PlotData, has_itow: bool, replay: Option<ReplayState>) -> Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_maximized(true),
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
