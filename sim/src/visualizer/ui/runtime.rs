//! App construction, runtime lifecycle, replay refresh, and frame update orchestration.

#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use eframe::egui;
use walkers::MapMemory;

use crate::visualizer::model::{Page, PlotData, VisualizerMountMode};
use crate::visualizer::pipeline::synthetic::build_synthetic_plot_data;
use crate::visualizer::stats::map_center_from_traces;
use crate::visualizer::theme::{UiDensity, UiTheme};

#[cfg(not(target_arch = "wasm32"))]
use super::MAPBOX_ACCESS_TOKEN_ENV;
use super::maps::map_tiles_from_token;
use super::state::{DataOrigin, TraceVisibility};
#[cfg(target_arch = "wasm32")]
use super::web::{
    WEB_MAX_POINTS_PER_TRACE, WebDatasetState, WebInputMode, WebPerf, WebRealDataSource,
    WebSyntheticNoise, WebSyntheticScenario, web_initial_mapbox_token, web_initial_ui_theme,
    web_query_flag, web_query_synthetic_noise, web_query_synthetic_scenario, web_remember_ui_theme,
};
use super::{App, ReplayState};

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
        shared_cursor_t_s: None,
        update_inspector_cursor_t_s: None,
        show_update_inspector: false,
        tuning_cfg,
        tuning_gnss_outages,
        tuning_misalignment,
        tuning_panel: None,
        replay,
        replay_status: None,
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
        web_scenario: WebSyntheticScenario::CityBlocks,
        #[cfg(target_arch = "wasm32")]
        web_synthetic_noise: web_query_synthetic_noise().unwrap_or(WebSyntheticNoise::Truth),
        #[cfg(target_arch = "wasm32")]
        web_input_mode: WebInputMode::Synthetic,
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
        app.web_input_mode = WebInputMode::RealData;
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

    pub(super) fn refresh_from_replay(&mut self) {
        let Some(replay) = self.replay.as_ref() else {
            return;
        };
        let misalignment = self.tuning_misalignment;
        let filter_cfg = self.tuning_cfg;
        let gnss_outages = self.tuning_gnss_outages;
        if let Some(synthetic) = &replay.synthetic {
            match build_synthetic_plot_data(synthetic, misalignment, filter_cfg, gnss_outages) {
                Ok(data) => {
                    self.replace_plot_data(data, Some(replay_run_key(replay)));
                    self.map_center = map_center_from_traces(&self.data.ekf_map);
                    self.has_itow = false;
                    self.data_origin = DataOrigin::Synthetic;
                    self.replay_status = Some("Synthetic replay refreshed".to_string());
                }
                Err(err) => {
                    self.replay_status = Some(format!("Synthetic replay failed: {err}"));
                }
            }
        } else {
            self.replay_status =
                Some("Generic CSV replay can be loaded from the browser UI".to_string());
        }
    }
}

pub(super) fn replay_run_key(replay: &ReplayState) -> String {
    if let Some(synthetic) = &replay.synthetic {
        return format!("synthetic:{synthetic:?}:max={:?}", replay.max_records);
    }
    let mut hasher = DefaultHasher::new();
    replay.bytes.hash(&mut hasher);
    replay.max_records.hash(&mut hasher);
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

        self.draw_current_page(ctx);
    }
}
