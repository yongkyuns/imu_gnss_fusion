#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, io::Read, rc::Rc};

use anyhow::Result;
use eframe::egui;
use egui_plot::{
    GridInput, GridMark, Legend, Line, LineStyle, Plot, PlotPoints, Points, VLine, log_grid_spacer,
};
#[cfg(target_arch = "wasm32")]
use flate2::read::GzDecoder;
#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Date, Function, Object, Reflect, Uint8Array};
#[cfg(target_arch = "wasm32")]
use serde::Deserialize;
use walkers::sources::{Attribution, Mapbox, MapboxStyle, TileSource};
use walkers::{HttpTiles, Map, MapMemory, Plugin, TileId, lon_lat};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::{JsFuture, spawn_local};
#[cfg(target_arch = "wasm32")]
use web_sys::{ErrorEvent, MessageEvent, Worker, WorkerOptions, WorkerType};

use super::math::{ecef_to_ned, heading_endpoint, lla_to_ecef};
use super::model::{EkfImuSource, HeadingSample, Page, PlotData, Trace};
use super::pipeline::synthetic::{SyntheticVisualizerConfig, build_synthetic_plot_data};
use super::pipeline::{EkfCompareConfig, GnssOutageConfig};
use super::stats::map_center_from_traces;
use super::theme::{UiDensity, UiTheme};

#[cfg(not(target_arch = "wasm32"))]
const MAPBOX_ACCESS_TOKEN_ENV: &str = "MAPBOX_ACCESS_TOKEN";
const TIME_SERIES_PLOT_HEIGHT: f32 = 200.0;
const SYNTHETIC_TRAJECTORY_MAX_POINTS: usize = 2_000;
const PLOT_GRID_MIN_SPACING_PX: f32 = 14.0;
const PLOT_GRID_MAX_SPACING_PX: f32 = 360.0;
const PLOT_GRID_STRENGTH_SCALE: f64 = 0.45;
#[cfg(target_arch = "wasm32")]
const TIME_SERIES_PLOT_FRAME_PADDING: f32 = 26.0;
#[cfg(target_arch = "wasm32")]
const WEB_DECIMATION_SCAN_MULTIPLIER: usize = 8;
#[cfg(target_arch = "wasm32")]
const WEB_MAX_POINTS_PER_TRACE: usize = 80;
#[cfg(target_arch = "wasm32")]
const WEB_DATASET_MANIFEST_URL: &str = "datasets/manifest.json";

struct TrackOverlay<'a> {
    traces: Vec<&'a Trace>,
    headings: Vec<&'a HeadingSample>,
    show_heading: bool,
    cursor_t_s: Option<f64>,
}

impl Plugin for TrackOverlay<'_> {
    fn run(
        self: Box<Self>,
        ui: &mut egui::Ui,
        response: &egui::Response,
        projector: &walkers::Projector,
        map_memory: &MapMemory,
    ) {
        let map_rect = response.rect.intersect(ui.clip_rect());
        let painter = ui.painter().with_clip_rect(map_rect);
        let visuals = ui.visuals();
        let view = map_view_bounds(projector, map_rect, 0.15);
        let point_stride = map_trace_point_stride(map_memory.zoom());
        let min_step = map_trace_min_pixel_step(map_memory.zoom());
        let min_step_sq = min_step * min_step;
        for tr in &self.traces {
            if tr.points.len() < 2 {
                continue;
            }
            let color = map_trace_color(tr.name.as_str(), visuals);
            let mut segment = Vec::<egui::Pos2>::with_capacity(tr.points.len().min(8192));
            let mut last_drawn: Option<egui::Pos2> = None;
            let mut pending: Option<egui::Pos2> = None;
            for p in tr.points.iter().step_by(point_stride) {
                let lon = p[0];
                let lat = p[1];
                if !lon.is_finite() || !lat.is_finite() || !view.contains(lon, lat) {
                    if segment.len() >= 2 {
                        painter.add(egui::epaint::PathShape::line(
                            segment,
                            egui::Stroke::new(2.2, color),
                        ));
                    }
                    segment = Vec::new();
                    last_drawn = None;
                    pending = None;
                    continue;
                }
                let v = projector.project(lon_lat(lon, lat));
                let pos = egui::pos2(v.x, v.y);
                match last_drawn {
                    None => {
                        segment.push(pos);
                        last_drawn = Some(pos);
                    }
                    Some(last) if last.distance_sq(pos) >= min_step_sq => {
                        if let Some(pending) = pending.take()
                            && last.distance_sq(pending) >= min_step_sq
                            && pending.distance_sq(pos) >= min_step_sq
                        {
                            segment.push(pending);
                        }
                        segment.push(pos);
                        last_drawn = Some(pos);
                    }
                    Some(_) => {
                        pending = Some(pos);
                    }
                }
            }
            if let (Some(last), Some(pending)) = (last_drawn, pending)
                && last.distance_sq(pending) > 0.0
            {
                segment.push(pending);
            }
            if segment.len() >= 2 {
                painter.add(egui::epaint::PathShape::line(
                    segment,
                    egui::Stroke::new(2.2, color),
                ));
            }
        }

        if self.show_heading {
            let mut last_tick_t = f64::NEG_INFINITY;
            for h in &self.headings {
                if h.t_s - last_tick_t < 1.0 {
                    continue;
                }
                last_tick_t = h.t_s;
                let from = projector.project(lon_lat(h.lon_deg, h.lat_deg));
                let (tip_lat, tip_lon) = heading_endpoint(h.lat_deg, h.lon_deg, h.yaw_deg, 6.0);
                let to = projector.project(lon_lat(tip_lon, tip_lat));
                painter.line_segment(
                    [egui::pos2(from.x, from.y), egui::pos2(to.x, to.y)],
                    egui::Stroke::new(1.8, map_heading_color(visuals)),
                );
            }
        }

        if let Some(t_s) = self.cursor_t_s {
            draw_map_cursor_marker(&painter, projector, view, &self.headings, t_s, visuals);
        }

        if self.show_heading
            && let Some(mouse_pos) = ui.input(|i| i.pointer.hover_pos())
            && map_rect.contains(mouse_pos)
        {
            let mut best: Option<(f32, &HeadingSample, egui::Pos2)> = None;
            let step = (self.headings.len() / 200).max(1);
            for h in self.headings.iter().step_by(step) {
                let v = projector.project(lon_lat(h.lon_deg, h.lat_deg));
                let p = egui::pos2(v.x, v.y);
                let d2 = p.distance_sq(mouse_pos);
                match best {
                    Some((bd2, _, _)) if d2 >= bd2 => {}
                    _ => best = Some((d2, h, p)),
                }
            }
            if let Some((d2, h, p)) = best
                && d2 <= 12.0_f32 * 12.0_f32
            {
                painter.circle_filled(p, 3.0, cursor_marker_color(visuals));
                let label = format!("t={:.2}s", h.t_s);
                let bg_min = p + egui::vec2(8.0, -24.0);
                let bg_rect = egui::Rect::from_min_size(bg_min, egui::vec2(78.0, 18.0));
                painter.rect_filled(bg_rect, 4.0, tooltip_fill(visuals));
                painter.text(
                    bg_min + egui::vec2(6.0, 2.0),
                    egui::Align2::LEFT_TOP,
                    label,
                    egui::FontId::monospace(12.0),
                    tooltip_text_color(visuals),
                );
            }
        }
    }
}

fn draw_map_cursor_marker(
    painter: &egui::Painter,
    projector: &walkers::Projector,
    view: MapViewBounds,
    headings: &[&HeadingSample],
    t_s: f64,
    visuals: &egui::Visuals,
) {
    let Some(sample) = sample_heading_at(headings, t_s) else {
        return;
    };
    if !view.contains(sample.lon_deg, sample.lat_deg) {
        return;
    }

    let projected = projector.project(lon_lat(sample.lon_deg, sample.lat_deg));
    let origin = egui::pos2(projected.x, projected.y);
    let yaw = (sample.yaw_deg as f32).to_radians();
    let dir = egui::vec2(yaw.sin(), -yaw.cos());
    let tip = origin + dir * 24.0;
    let side = egui::vec2(-dir.y, dir.x);
    let color = cursor_marker_color(visuals);
    painter.circle_filled(origin, 4.0, color);
    painter.line_segment([origin, tip], egui::Stroke::new(2.2, color));
    painter.add(egui::Shape::convex_polygon(
        vec![
            tip,
            tip - dir * 8.0 + side * 4.5,
            tip - dir * 8.0 - side * 4.5,
        ],
        color,
        egui::Stroke::NONE,
    ));

    let label = format!("{t_s:.2}s");
    let bg_min = origin + egui::vec2(8.0, 8.0);
    let bg_rect = egui::Rect::from_min_size(bg_min, egui::vec2(62.0, 18.0));
    painter.rect_filled(bg_rect, 4.0, tooltip_fill(visuals));
    painter.text(
        bg_min + egui::vec2(6.0, 2.0),
        egui::Align2::LEFT_TOP,
        label,
        egui::FontId::monospace(12.0),
        tooltip_text_color(visuals),
    );
}

fn sample_heading_at(headings: &[&HeadingSample], t_s: f64) -> Option<HeadingSample> {
    if !t_s.is_finite() || headings.is_empty() {
        return None;
    }
    headings
        .iter()
        .min_by(|a, b| {
            let da = (a.t_s - t_s).abs();
            let db = (b.t_s - t_s).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|h| **h)
}

fn map_trace_color(name: &str, visuals: &egui::Visuals) -> egui::Color32 {
    if name.contains("GNSS") {
        if visuals.dark_mode {
            egui::Color32::from_rgb(0, 255, 255)
        } else {
            egui::Color32::from_rgb(0, 118, 152)
        }
    } else if name == "ESKF path (lon,lat)" {
        if visuals.dark_mode {
            egui::Color32::from_rgb(120, 170, 255)
        } else {
            egui::Color32::from_rgb(35, 105, 200)
        }
    } else if name == "ESKF path during GNSS outage (lon,lat)" {
        if visuals.dark_mode {
            egui::Color32::from_rgb(255, 140, 220)
        } else {
            egui::Color32::from_rgb(184, 55, 144)
        }
    } else if name == "Loose path (lon,lat)" {
        if visuals.dark_mode {
            egui::Color32::from_rgb(120, 255, 170)
        } else {
            egui::Color32::from_rgb(26, 138, 78)
        }
    } else {
        visuals.text_color()
    }
}

fn map_heading_color(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_rgb(245, 248, 252)
    } else {
        egui::Color32::from_rgb(42, 49, 59)
    }
}

fn cursor_marker_color(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_rgb(255, 220, 70)
    } else {
        egui::Color32::from_rgb(194, 119, 0)
    }
}

fn tooltip_fill(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_black_alpha(190)
    } else {
        egui::Color32::from_white_alpha(230)
    }
}

fn tooltip_text_color(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::WHITE
    } else {
        egui::Color32::from_rgb(32, 38, 48)
    }
}

fn map_trace_min_pixel_step(zoom: f64) -> f32 {
    if zoom >= 18.0 {
        0.25
    } else if zoom >= 17.0 {
        0.75
    } else if zoom >= 16.0 {
        1.5
    } else if zoom >= 15.0 {
        3.0
    } else {
        5.0
    }
}

fn map_trace_point_stride(zoom: f64) -> usize {
    if zoom >= 17.0 {
        1
    } else if zoom >= 16.0 {
        3
    } else if zoom >= 15.0 {
        8
    } else {
        16
    }
}

#[derive(Clone, Copy)]
struct MapViewBounds {
    lon_min: f64,
    lon_max: f64,
    lat_min: f64,
    lat_max: f64,
}

impl MapViewBounds {
    fn contains(self, lon: f64, lat: f64) -> bool {
        lon >= self.lon_min && lon <= self.lon_max && lat >= self.lat_min && lat <= self.lat_max
    }
}

fn map_view_bounds(
    projector: &walkers::Projector,
    rect: egui::Rect,
    margin_fraction: f32,
) -> MapViewBounds {
    let margin = egui::vec2(
        rect.width() * margin_fraction,
        rect.height() * margin_fraction,
    );
    let rect = rect.expand2(margin);
    let corners = [
        rect.left_top(),
        rect.right_top(),
        rect.left_bottom(),
        rect.right_bottom(),
    ];
    let mut lon_min = f64::INFINITY;
    let mut lon_max = f64::NEG_INFINITY;
    let mut lat_min = f64::INFINITY;
    let mut lat_max = f64::NEG_INFINITY;
    for corner in corners {
        let pos = projector.unproject(corner.to_vec2());
        lon_min = lon_min.min(pos.x());
        lon_max = lon_max.max(pos.x());
        lat_min = lat_min.min(pos.y());
        lat_max = lat_max.max(pos.y());
    }
    MapViewBounds {
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    }
}

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
    show_heading: bool,
    show_gnss_map: bool,
    show_eskf: bool,
    show_loose: bool,
    overview_cursor_t_s: Option<f64>,
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
    web_mapbox_token: String,
    #[cfg(target_arch = "wasm32")]
    web_mapbox_token_applied: String,
    #[cfg(target_arch = "wasm32")]
    web_scenario: WebSyntheticScenario,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DataOrigin {
    Real,
    Synthetic,
}

#[derive(Clone)]
pub struct ReplayState {
    pub bytes: Vec<u8>,
    pub synthetic: Option<SyntheticVisualizerConfig>,
    pub max_records: Option<usize>,
    pub misalignment: EkfImuSource,
    pub ekf_cfg: EkfCompareConfig,
    pub gnss_outages: GnssOutageConfig,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
struct NamedText {
    name: String,
    text: String,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Deserialize)]
struct WebDatasetEntry {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default, alias = "baseUrl")]
    base_url: Option<String>,
    #[serde(default, alias = "imu_csv")]
    imu: Option<String>,
    #[serde(default, alias = "gnss_csv")]
    gnss: Option<String>,
    #[serde(default, alias = "imu_csv_gz")]
    imu_gz: Option<String>,
    #[serde(default, alias = "gnss_csv_gz")]
    gnss_gz: Option<String>,
    #[serde(default, alias = "reference_attitude_csv")]
    reference_attitude: Option<String>,
    #[serde(default, alias = "reference_attitude_csv_gz")]
    reference_attitude_gz: Option<String>,
    #[serde(default, alias = "reference_mount_csv")]
    reference_mount: Option<String>,
    #[serde(default, alias = "reference_mount_csv_gz")]
    reference_mount_gz: Option<String>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Deserialize)]
struct WebDatasetManifest {
    #[serde(default)]
    datasets: Vec<WebDatasetEntry>,
}

#[cfg(target_arch = "wasm32")]
struct WebDatasetState {
    manifest_url: String,
    datasets: Vec<WebDatasetEntry>,
    selected: usize,
    auto_load_id: Option<String>,
    auto_load_attempted: bool,
    loading_manifest: bool,
    loading_dataset: bool,
    loading_replay: bool,
    replay_job_id: u64,
    pending: Rc<RefCell<Option<WebDatasetTaskResult>>>,
    pending_replay: Rc<RefCell<Option<WebReplayTaskResult>>>,
    pending_progress: Rc<RefCell<Option<WebReplayProgress>>>,
    replay_worker: Option<Worker>,
    replay_onmessage: Option<Closure<dyn FnMut(MessageEvent)>>,
    replay_onerror: Option<Closure<dyn FnMut(ErrorEvent)>>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct WebReplayProgress {
    job_id: u64,
    progress: f32,
}

#[cfg(target_arch = "wasm32")]
enum WebDatasetTaskResult {
    Manifest(std::result::Result<Vec<WebDatasetEntry>, String>),
    Dataset(std::result::Result<WebDatasetFiles, String>),
}

#[cfg(target_arch = "wasm32")]
enum WebReplayTaskResult {
    Complete {
        job_id: u64,
        result: std::result::Result<WebReplayWorkerOutput, String>,
    },
}

#[cfg(target_arch = "wasm32")]
struct WebReplayWorkerOutput {
    source: String,
    label: String,
    imu_name: String,
    gnss_name: String,
    json: String,
}

#[cfg(target_arch = "wasm32")]
enum WebReplayWorkerJob {
    Csv {
        label: String,
        imu_name: String,
        gnss_name: String,
        imu_csv: String,
        gnss_csv: String,
        reference_attitude_csv: Option<String>,
        reference_mount_csv: Option<String>,
    },
    Synthetic {
        label: String,
        motion_label: String,
        motion_text: String,
    },
}

#[cfg(target_arch = "wasm32")]
struct WebDatasetFiles {
    label: String,
    imu: NamedText,
    gnss: NamedText,
    reference_attitude: Option<NamedText>,
    reference_mount: Option<NamedText>,
}

#[cfg(target_arch = "wasm32")]
impl WebDatasetState {
    fn new() -> Self {
        Self {
            manifest_url: WEB_DATASET_MANIFEST_URL.to_string(),
            datasets: Vec::new(),
            selected: 0,
            auto_load_id: web_query_value("dataset"),
            auto_load_attempted: false,
            loading_manifest: false,
            loading_dataset: false,
            loading_replay: false,
            replay_job_id: 0,
            pending: Rc::new(RefCell::new(None)),
            pending_replay: Rc::new(RefCell::new(None)),
            pending_progress: Rc::new(RefCell::new(None)),
            replay_worker: None,
            replay_onmessage: None,
            replay_onerror: None,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl WebReplayWorkerJob {
    fn label(&self) -> &str {
        match self {
            Self::Csv { label, .. } | Self::Synthetic { label, .. } => label,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl WebDatasetEntry {
    fn display_label(&self) -> String {
        self.label
            .as_deref()
            .or(self.id.as_deref())
            .unwrap_or("unnamed dataset")
            .to_string()
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum WebInputMode {
    Synthetic,
    RealData,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum WebRealDataSource {
    DroppedCsv,
    ManifestDataset,
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum WebSyntheticScenario {
    CityBlocks,
    FigureEight,
    StraightAccelBrake,
}

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct WebPerf {
    enabled: bool,
    frame_count: u32,
    start_time_s: f64,
    last_time_s: f64,
    fps_ema: f64,
}

#[cfg(target_arch = "wasm32")]
fn web_query_value_raw(key: &str) -> Option<String> {
    let search = eframe::web_sys::window()?.location().search().ok()?;
    for pair in search.trim_start_matches('?').split('&') {
        if pair.is_empty() {
            continue;
        }
        let (name, value) = pair.split_once('=').unwrap_or((pair, "1"));
        if name == key {
            return Some(value.to_string());
        }
    }
    None
}

#[cfg(target_arch = "wasm32")]
fn web_query_value(key: &str) -> Option<String> {
    web_query_value_raw(key).map(|value| value.to_ascii_lowercase())
}

#[cfg(target_arch = "wasm32")]
fn web_query_flag(key: &str) -> bool {
    matches!(
        web_query_value(key).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

#[cfg(target_arch = "wasm32")]
fn web_now_s() -> f64 {
    Date::now() * 0.001
}

#[cfg(target_arch = "wasm32")]
fn estimate_web_replay_duration_s(imu_csv: &str, gnss_csv: &str) -> f64 {
    let span_s = csv_time_span_s(imu_csv).max(csv_time_span_s(gnss_csv));
    if span_s.is_finite() && span_s > 0.0 {
        (span_s / 140.0).clamp(3.0, 18.0)
    } else {
        6.0
    }
}

#[cfg(target_arch = "wasm32")]
fn csv_time_span_s(csv: &str) -> f64 {
    let mut first = None::<f64>;
    let mut last = None::<f64>;
    for line in csv.lines() {
        let Some(value) = line
            .split(',')
            .next()
            .and_then(|cell| cell.trim().parse().ok())
        else {
            continue;
        };
        if first.is_none() {
            first = Some(value);
        }
        last = Some(value);
    }
    match (first, last) {
        (Some(first), Some(last)) => (last - first).abs(),
        _ => 0.0,
    }
}

#[cfg(target_arch = "wasm32")]
fn web_replay_worker_request(job_id: u64, job: &WebReplayWorkerJob) -> Object {
    let request = Object::new();
    let _ = Reflect::set(
        &request,
        &JsValue::from_str("jobId"),
        &JsValue::from_f64(job_id as f64),
    );
    let _ = Reflect::set(
        &request,
        &JsValue::from_str("label"),
        &JsValue::from_str(job.label()),
    );
    let source = Object::new();
    match job {
        WebReplayWorkerJob::Csv {
            label,
            imu_name,
            gnss_name,
            imu_csv,
            gnss_csv,
            reference_attitude_csv,
            reference_mount_csv,
        } => {
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("kind"),
                &JsValue::from_str("csv"),
            );
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("label"),
                &JsValue::from_str(label),
            );
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("imuName"),
                &JsValue::from_str(imu_name),
            );
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("gnssName"),
                &JsValue::from_str(gnss_name),
            );
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("imuCsv"),
                &JsValue::from_str(imu_csv),
            );
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("gnssCsv"),
                &JsValue::from_str(gnss_csv),
            );
            if let Some(reference) = reference_attitude_csv {
                let _ = Reflect::set(
                    &source,
                    &JsValue::from_str("referenceAttitudeCsv"),
                    &JsValue::from_str(reference),
                );
            }
            if let Some(reference) = reference_mount_csv {
                let _ = Reflect::set(
                    &source,
                    &JsValue::from_str("referenceMountCsv"),
                    &JsValue::from_str(reference),
                );
            }
        }
        WebReplayWorkerJob::Synthetic {
            label,
            motion_label,
            motion_text,
        } => {
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("kind"),
                &JsValue::from_str("synthetic"),
            );
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("label"),
                &JsValue::from_str(label),
            );
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("motionLabel"),
                &JsValue::from_str(motion_label),
            );
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("motionText"),
                &JsValue::from_str(motion_text),
            );
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("noiseMode"),
                &JsValue::from_str("low"),
            );
            let _ = Reflect::set(&source, &JsValue::from_str("seed"), &JsValue::from_f64(1.0));
            let mount_rpy_deg = js_number_array([5.0, -5.0, 5.0]);
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("mountRpyDeg"),
                mount_rpy_deg.as_ref(),
            );
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("imuHz"),
                &JsValue::from_f64(100.0),
            );
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("gnssHz"),
                &JsValue::from_f64(2.0),
            );
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("gnssTimeShiftMs"),
                &JsValue::from_f64(0.0),
            );
            let early_vel_bias_ned_mps = js_number_array([0.0, 0.0, 0.0]);
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("earlyVelBiasNedMps"),
                early_vel_bias_ned_mps.as_ref(),
            );
        }
    }
    let _ = Reflect::set(&request, &JsValue::from_str("source"), source.as_ref());
    request
}

#[cfg(target_arch = "wasm32")]
fn js_number_array(values: [f64; 3]) -> Array {
    let array = Array::new();
    for value in values {
        array.push(&JsValue::from_f64(value));
    }
    array
}

#[cfg(target_arch = "wasm32")]
fn web_query_synthetic_scenario() -> Option<WebSyntheticScenario> {
    match web_query_value("scenario").as_deref() {
        Some("city" | "city_blocks" | "city-blocks") => Some(WebSyntheticScenario::CityBlocks),
        Some("figure8" | "figure_eight" | "figure-eight") => {
            Some(WebSyntheticScenario::FigureEight)
        }
        Some("straight" | "straight_accel_brake" | "straight-accel-brake") => {
            Some(WebSyntheticScenario::StraightAccelBrake)
        }
        _ => None,
    }
}

#[cfg(target_arch = "wasm32")]
async fn web_fetch_bytes(url: &str) -> std::result::Result<Vec<u8>, String> {
    let window = eframe::web_sys::window().ok_or_else(|| "missing window".to_string())?;
    let response_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(js_error_string)?;
    let response = response_value
        .dyn_into::<eframe::web_sys::Response>()
        .map_err(|_| format!("{url}: fetch did not return a Response"))?;
    if !response.ok() {
        return Err(format!("{url}: HTTP {}", response.status()));
    }
    let buffer = JsFuture::from(response.array_buffer().map_err(js_error_string)?)
        .await
        .map_err(js_error_string)?;
    let bytes = Uint8Array::new(&buffer);
    let mut out = vec![0; bytes.length() as usize];
    bytes.copy_to(&mut out);
    Ok(out)
}

#[cfg(target_arch = "wasm32")]
async fn web_fetch_text(url: &str) -> std::result::Result<String, String> {
    String::from_utf8(web_fetch_bytes(url).await?)
        .map_err(|err| format!("{url}: response was not UTF-8: {err}"))
}

#[cfg(target_arch = "wasm32")]
async fn web_fetch_manifest(url: String) -> std::result::Result<Vec<WebDatasetEntry>, String> {
    let text = web_fetch_text(&url).await?;
    let manifest: WebDatasetManifest =
        serde_json::from_str(&text).map_err(|err| format!("{url}: bad manifest JSON: {err}"))?;
    Ok(manifest.datasets)
}

#[cfg(target_arch = "wasm32")]
async fn web_fetch_dataset(
    manifest_url: String,
    entry: WebDatasetEntry,
) -> std::result::Result<WebDatasetFiles, String> {
    let label = entry.display_label();
    let imu = web_fetch_dataset_csv(&manifest_url, &entry, "imu")
        .await
        .map(|(name, text)| NamedText { name, text })?;
    let gnss = web_fetch_dataset_csv(&manifest_url, &entry, "gnss")
        .await
        .map(|(name, text)| NamedText { name, text })?;
    let reference_attitude =
        web_fetch_optional_dataset_csv(&manifest_url, &entry, "reference_attitude")
            .await?
            .map(|(name, text)| NamedText { name, text });
    let reference_mount = web_fetch_optional_dataset_csv(&manifest_url, &entry, "reference_mount")
        .await?
        .map(|(name, text)| NamedText { name, text });
    Ok(WebDatasetFiles {
        label,
        imu,
        gnss,
        reference_attitude,
        reference_mount,
    })
}

#[cfg(target_arch = "wasm32")]
async fn web_fetch_dataset_csv(
    manifest_url: &str,
    entry: &WebDatasetEntry,
    kind: &str,
) -> std::result::Result<(String, String), String> {
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
        _ => return Err(format!("unsupported dataset file kind: {kind}")),
    };
    if let Some(path) = gz {
        let url = web_dataset_url(manifest_url, entry.base_url.as_deref(), path);
        let bytes = web_fetch_bytes(&url).await?;
        return Ok((web_dataset_file_name(&url), decode_gzip_csv(&url, &bytes)?));
    }
    if let Some(path) = plain {
        let url = web_dataset_url(manifest_url, entry.base_url.as_deref(), path);
        let bytes = web_fetch_bytes(&url).await?;
        let text = if url.to_ascii_lowercase().ends_with(".gz") {
            decode_gzip_csv(&url, &bytes)?
        } else {
            decode_plain_csv(&url, bytes)?
        };
        return Ok((web_dataset_file_name(&url), text));
    }

    let gz_url = web_dataset_url(
        manifest_url,
        entry.base_url.as_deref(),
        &format!("{kind}.csv.gz"),
    );
    match web_fetch_bytes(&gz_url).await {
        Ok(bytes) => {
            return Ok((
                web_dataset_file_name(&gz_url),
                decode_gzip_csv(&gz_url, &bytes)?,
            ));
        }
        Err(gz_err) => {
            let csv_url = web_dataset_url(
                manifest_url,
                entry.base_url.as_deref(),
                &format!("{kind}.csv"),
            );
            let bytes = web_fetch_bytes(&csv_url)
                .await
                .map_err(|csv_err| format!("{gz_err}; fallback {csv_err}"))?;
            Ok((
                web_dataset_file_name(&csv_url),
                decode_plain_csv(&csv_url, bytes)?,
            ))
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn web_fetch_optional_dataset_csv(
    manifest_url: &str,
    entry: &WebDatasetEntry,
    kind: &str,
) -> std::result::Result<Option<(String, String)>, String> {
    let has_explicit_file = match kind {
        "reference_attitude" => {
            entry.reference_attitude.is_some() || entry.reference_attitude_gz.is_some()
        }
        "reference_mount" => entry.reference_mount.is_some() || entry.reference_mount_gz.is_some(),
        _ => false,
    };
    if !has_explicit_file {
        return Ok(None);
    }
    web_fetch_dataset_csv(manifest_url, entry, kind)
        .await
        .map(Some)
}

#[cfg(target_arch = "wasm32")]
fn web_dataset_url(manifest_url: &str, base_url: Option<&str>, path: &str) -> String {
    if is_absolute_web_url(path) {
        return path.to_string();
    }
    let base = base_url.unwrap_or("");
    let joined = if base.is_empty() {
        path.to_string()
    } else {
        format!(
            "{}/{}",
            base.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    };
    if is_absolute_web_url(&joined) || joined.starts_with('/') {
        joined
    } else {
        let manifest_dir = manifest_url
            .rsplit_once('/')
            .map(|(dir, _)| format!("{dir}/"))
            .unwrap_or_default();
        format!("{manifest_dir}{joined}")
    }
}

#[cfg(target_arch = "wasm32")]
fn is_absolute_web_url(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://") || url.starts_with('/')
}

#[cfg(target_arch = "wasm32")]
fn web_dataset_file_name(url: &str) -> String {
    url.rsplit('/').next().unwrap_or(url).to_string()
}

#[cfg(target_arch = "wasm32")]
fn decode_plain_csv(url: &str, bytes: Vec<u8>) -> std::result::Result<String, String> {
    String::from_utf8(bytes).map_err(|err| format!("{url}: CSV was not UTF-8: {err}"))
}

#[cfg(target_arch = "wasm32")]
fn decode_gzip_csv(url: &str, bytes: &[u8]) -> std::result::Result<String, String> {
    let mut decoder = GzDecoder::new(bytes);
    let mut text = String::new();
    decoder
        .read_to_string(&mut text)
        .map_err(|err| format!("{url}: gzip decode failed: {err}"))?;
    Ok(text)
}

#[cfg(target_arch = "wasm32")]
fn js_error_string(value: JsValue) -> String {
    value
        .as_string()
        .unwrap_or_else(|| "JavaScript error".to_string())
}

#[cfg(target_arch = "wasm32")]
fn web_initial_mapbox_token() -> String {
    let Some(window) = eframe::web_sys::window() else {
        return web_query_value_raw("mapbox_token").unwrap_or_default();
    };
    let Ok(value) = Reflect::get(
        window.as_ref(),
        &JsValue::from_str("__imuGnssFusionInitialMapboxToken"),
    ) else {
        return web_query_value_raw("mapbox_token").unwrap_or_default();
    };
    let Some(function) = value.dyn_ref::<Function>() else {
        return web_query_value_raw("mapbox_token").unwrap_or_default();
    };
    function
        .call0(&JsValue::NULL)
        .ok()
        .and_then(|value| value.as_string())
        .unwrap_or_else(|| web_query_value_raw("mapbox_token").unwrap_or_default())
}

#[cfg(target_arch = "wasm32")]
fn web_remember_mapbox_token(token: &str) {
    let Some(window) = eframe::web_sys::window() else {
        return;
    };
    let Ok(value) = Reflect::get(
        window.as_ref(),
        &JsValue::from_str("__imuGnssFusionRememberMapboxToken"),
    ) else {
        return;
    };
    if let Some(function) = value.dyn_ref::<Function>() {
        let _ = function.call1(&JsValue::NULL, &JsValue::from_str(token));
    }
}

#[cfg(target_arch = "wasm32")]
fn web_initial_ui_theme() -> UiTheme {
    if let Some(theme) = web_query_value_raw("theme").and_then(|value| UiTheme::from_value(&value))
    {
        return theme;
    }
    let Some(window) = eframe::web_sys::window() else {
        return UiTheme::default();
    };
    let Ok(value) = Reflect::get(
        window.as_ref(),
        &JsValue::from_str("__imuGnssFusionInitialTheme"),
    ) else {
        return UiTheme::default();
    };
    let Some(function) = value.dyn_ref::<Function>() else {
        return UiTheme::default();
    };
    function
        .call0(&JsValue::NULL)
        .ok()
        .and_then(|value| value.as_string())
        .and_then(|value| UiTheme::from_value(&value))
        .unwrap_or_default()
}

#[cfg(target_arch = "wasm32")]
fn web_remember_ui_theme(theme: UiTheme) {
    let Some(window) = eframe::web_sys::window() else {
        return;
    };
    let Ok(value) = Reflect::get(
        window.as_ref(),
        &JsValue::from_str("__imuGnssFusionRememberTheme"),
    ) else {
        return;
    };
    if let Some(function) = value.dyn_ref::<Function>() {
        let _ = function.call1(&JsValue::NULL, &JsValue::from_str(theme.storage_value()));
    }
}

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

#[derive(Clone, Copy)]
enum CartoRasterStyle {
    Positron,
    DarkMatter,
}

struct CartoRasterTiles {
    style: CartoRasterStyle,
}

impl CartoRasterTiles {
    fn for_theme(theme: UiTheme) -> Self {
        let style = match theme {
            UiTheme::Light => CartoRasterStyle::Positron,
            UiTheme::Dark => CartoRasterStyle::DarkMatter,
        };
        Self { style }
    }
}

impl TileSource for CartoRasterTiles {
    fn tile_url(&self, tile_id: TileId) -> String {
        let style = match self.style {
            CartoRasterStyle::Positron => "light_all",
            CartoRasterStyle::DarkMatter => "dark_all",
        };
        let subdomain =
            ["a", "b", "c", "d"][((tile_id.x + tile_id.y + tile_id.zoom as u32) % 4) as usize];
        format!(
            "https://{subdomain}.basemaps.cartocdn.com/{style}/{}/{}/{}.png",
            tile_id.zoom, tile_id.x, tile_id.y
        )
    }

    fn attribution(&self) -> Attribution {
        Attribution {
            text: "(C) OpenStreetMap contributors, (C) CARTO",
            url: "https://carto.com/attributions",
            logo_light: None,
            logo_dark: None,
        }
    }
}

fn map_tiles_from_token(token: &str, theme: UiTheme, egui_ctx: egui::Context) -> HttpTiles {
    if token.is_empty() {
        HttpTiles::new(CartoRasterTiles::for_theme(theme), egui_ctx)
    } else {
        let style = match theme {
            UiTheme::Light => MapboxStyle::Light,
            UiTheme::Dark => MapboxStyle::Dark,
        };
        HttpTiles::new(
            Mapbox {
                style,
                high_resolution: true,
                access_token: token.to_string(),
            },
            egui_ctx,
        )
    }
}

fn create_app(
    cc: &eframe::CreationContext<'_>,
    data: PlotData,
    has_itow: bool,
    replay: Option<ReplayState>,
) -> App {
    let map_center = map_center_from_traces(&data.eskf_map);
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
    let mut app = App {
        data,
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
        show_heading: false,
        show_gnss_map: true,
        show_eskf: true,
        show_loose: true,
        overview_cursor_t_s: None,
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
        web_mapbox_token: mapbox_access_token.clone(),
        #[cfg(target_arch = "wasm32")]
        web_mapbox_token_applied: mapbox_access_token,
        #[cfg(target_arch = "wasm32")]
        web_scenario: WebSyntheticScenario::CityBlocks,
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
    if matches!(
        web_query_value("page").as_deref(),
        Some("map" | "mapdark" | "map-dark")
    ) {
        app.page = Page::Map;
    }
    #[cfg(target_arch = "wasm32")]
    if !auto_load_dataset {
        if let Some(scenario) = web_query_synthetic_scenario() {
            app.web_scenario = scenario;
        }
        app.refresh_from_web_synthetic();
    }
    #[cfg(target_arch = "wasm32")]
    app.start_web_manifest_load();
    app
}

impl App {
    fn set_ui_theme(&mut self, theme: UiTheme, ctx: &egui::Context) {
        if self.ui_theme == theme {
            return;
        }
        self.ui_theme = theme;
        super::theme::apply(ctx, current_ui_density(), self.ui_theme);
        self.refresh_map_tiles(ctx);
        #[cfg(target_arch = "wasm32")]
        web_remember_ui_theme(self.ui_theme);
    }

    fn refresh_map_tiles(&mut self, ctx: &egui::Context) {
        #[cfg(target_arch = "wasm32")]
        let token = self.web_mapbox_token.clone();
        #[cfg(not(target_arch = "wasm32"))]
        let token = std::env::var(MAPBOX_ACCESS_TOKEN_ENV).unwrap_or_default();
        self.map_tiles = map_tiles_from_token(&token, self.ui_theme, ctx.clone());
    }

    #[cfg(target_arch = "wasm32")]
    fn start_web_manifest_load(&mut self) {
        if self.web_datasets.loading_manifest {
            return;
        }
        self.web_datasets.loading_manifest = true;
        self.web_status = format!(
            "Loading dataset manifest: {}",
            self.web_datasets.manifest_url
        );
        let manifest_url = self.web_datasets.manifest_url.clone();
        let pending = Rc::clone(&self.web_datasets.pending);
        spawn_local(async move {
            let result = web_fetch_manifest(manifest_url).await;
            *pending.borrow_mut() = Some(WebDatasetTaskResult::Manifest(result));
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn start_web_dataset_load(&mut self) {
        if self.web_datasets.loading_dataset
            || self.web_datasets.loading_replay
            || self.web_datasets.datasets.is_empty()
        {
            return;
        }
        let selected = self
            .web_datasets
            .selected
            .min(self.web_datasets.datasets.len().saturating_sub(1));
        let entry = self.web_datasets.datasets[selected].clone();
        let label = entry.display_label();
        self.web_datasets.selected = selected;
        self.web_input_mode = WebInputMode::RealData;
        self.web_real_data_source = WebRealDataSource::ManifestDataset;
        self.web_datasets.loading_dataset = true;
        self.web_run_progress = 0.0;
        self.web_run_started_time_s = web_now_s();
        self.web_run_estimated_duration_s = 2.0;
        self.web_status = format!("Loading dataset: {label}");
        let manifest_url = self.web_datasets.manifest_url.clone();
        let pending = Rc::clone(&self.web_datasets.pending);
        spawn_local(async move {
            let result = web_fetch_dataset(manifest_url, entry).await;
            *pending.borrow_mut() = Some(WebDatasetTaskResult::Dataset(result));
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn poll_web_dataset_tasks(&mut self) {
        let Some(result) = self.web_datasets.pending.borrow_mut().take() else {
            return;
        };
        match result {
            WebDatasetTaskResult::Manifest(result) => {
                self.web_datasets.loading_manifest = false;
                match result {
                    Ok(datasets) => {
                        let count = datasets.len();
                        self.web_datasets.datasets = datasets;
                        self.web_datasets.selected = self
                            .web_datasets
                            .selected
                            .min(self.web_datasets.datasets.len().saturating_sub(1));
                        self.web_status = if count == 0 {
                            "Dataset manifest loaded with no entries.".to_string()
                        } else {
                            format!("Dataset manifest loaded: {count} entries")
                        };
                        if let Some(auto_id) = self.web_datasets.auto_load_id.clone()
                            && !self.web_datasets.auto_load_attempted
                        {
                            self.web_datasets.auto_load_attempted = true;
                            if let Some(idx) = self.web_datasets.datasets.iter().position(|d| {
                                d.id.as_deref() == Some(auto_id.as_str())
                                    || d.display_label().to_ascii_lowercase() == auto_id
                            }) {
                                self.web_datasets.selected = idx;
                                self.start_web_dataset_load();
                            } else {
                                self.web_status = format!(
                                    "Dataset manifest loaded, but '{auto_id}' was not found"
                                );
                            }
                        }
                    }
                    Err(err) => {
                        self.web_datasets.datasets.clear();
                        self.web_datasets.selected = 0;
                        self.web_status = format!("Dataset manifest failed: {err}");
                    }
                }
            }
            WebDatasetTaskResult::Dataset(result) => {
                self.web_datasets.loading_dataset = false;
                match result {
                    Ok(files) => {
                        let label = files.label.clone();
                        let imu_name = files.imu.name.clone();
                        let gnss_name = files.gnss.name.clone();
                        self.web_input_mode = WebInputMode::RealData;
                        self.web_real_data_source = WebRealDataSource::ManifestDataset;
                        self.web_imu_csv = Some(files.imu);
                        self.web_gnss_csv = Some(files.gnss);
                        self.web_reference_attitude_csv = files.reference_attitude;
                        self.web_reference_mount_csv = files.reference_mount;
                        self.web_run_progress = self.web_run_progress.max(0.02);
                        self.start_web_replay_build(label, imu_name, gnss_name);
                    }
                    Err(err) => {
                        self.web_status = format!("Dataset load failed: {err}");
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn poll_web_replay_progress(&mut self) {
        if self.web_datasets.loading_dataset || self.web_datasets.loading_replay {
            let elapsed_s = (web_now_s() - self.web_run_started_time_s).max(0.0);
            let estimated = (elapsed_s / self.web_run_estimated_duration_s.max(0.5)) as f32;
            self.web_run_progress = self.web_run_progress.max(estimated.min(0.95));
        }
        let Some(progress) = self.web_datasets.pending_progress.borrow_mut().take() else {
            return;
        };
        if progress.job_id == self.web_datasets.replay_job_id {
            self.web_run_progress = self.web_run_progress.max(progress.progress);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn poll_web_replay_tasks(&mut self) {
        let Some(result) = self.web_datasets.pending_replay.borrow_mut().take() else {
            return;
        };
        let WebReplayTaskResult::Complete { job_id, result } = result;
        if job_id != self.web_datasets.replay_job_id {
            return;
        }
        self.finish_web_replay_worker();
        self.web_datasets.loading_replay = false;
        self.web_run_progress = 1.0;
        match result {
            Ok(output) => match serde_json::from_str::<PlotData>(&output.json) {
                Ok(data) => {
                    let is_synthetic = output.source == "synthetic";
                    self.data = data;
                    self.map_center = map_center_from_traces(&self.data.eskf_map);
                    self.has_itow = false;
                    self.data_origin = if is_synthetic {
                        DataOrigin::Synthetic
                    } else {
                        DataOrigin::Real
                    };
                    self.web_input_mode = if is_synthetic {
                        WebInputMode::Synthetic
                    } else {
                        WebInputMode::RealData
                    };
                    if self.page != Page::Map {
                        self.page = Page::Overview;
                    }
                    self.web_status = if is_synthetic {
                        format!("Synthetic scenario loaded: {}", output.label)
                    } else {
                        format!(
                            "Dataset loaded: {} ({} / {})",
                            output.label, output.imu_name, output.gnss_name
                        )
                    };
                }
                Err(err) => {
                    self.web_status = format!("Replay result decode failed: {err}");
                }
            },
            Err(err) => {
                self.web_status = format!("CSV replay failed: {err}");
            }
        }
    }

    fn refresh_from_replay(&mut self) {
        let Some(replay) = self.replay.as_ref() else {
            return;
        };
        if let Some(synthetic) = &replay.synthetic {
            match build_synthetic_plot_data(
                synthetic,
                replay.misalignment,
                replay.ekf_cfg,
                replay.gnss_outages,
            ) {
                Ok(data) => {
                    self.data = data;
                    self.map_center = map_center_from_traces(&self.data.eskf_map);
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

    #[cfg(target_arch = "wasm32")]
    fn refresh_from_generic_csv(&mut self) -> bool {
        let (Some(imu), Some(gnss)) = (&self.web_imu_csv, &self.web_gnss_csv) else {
            self.web_status =
                "Load both imu.csv and gnss.csv before running CSV replay.".to_string();
            return false;
        };
        self.web_input_mode = WebInputMode::RealData;
        self.web_real_data_source = WebRealDataSource::DroppedCsv;
        self.data_origin = DataOrigin::Real;
        self.start_web_replay_build(
            "CSV replay".to_string(),
            imu.name.clone(),
            gnss.name.clone(),
        );
        true
    }

    #[cfg(target_arch = "wasm32")]
    fn start_web_replay_build(&mut self, label: String, imu_name: String, gnss_name: String) {
        let (Some(imu), Some(gnss)) = (&self.web_imu_csv, &self.web_gnss_csv) else {
            self.web_status =
                "Load both imu.csv and gnss.csv before running CSV replay.".to_string();
            return;
        };
        let imu_text = imu.text.clone();
        let gnss_text = gnss.text.clone();
        let reference_attitude_text = self
            .web_reference_attitude_csv
            .as_ref()
            .map(|reference| reference.text.clone());
        let reference_mount_text = self
            .web_reference_mount_csv
            .as_ref()
            .map(|reference| reference.text.clone());
        let estimated_duration_s = estimate_web_replay_duration_s(&imu_text, &gnss_text);
        self.start_web_replay_worker(
            WebReplayWorkerJob::Csv {
                label,
                imu_name,
                gnss_name,
                imu_csv: imu_text,
                gnss_csv: gnss_text,
                reference_attitude_csv: reference_attitude_text,
                reference_mount_csv: reference_mount_text,
            },
            estimated_duration_s,
        );
    }

    #[cfg(target_arch = "wasm32")]
    fn start_web_replay_worker(&mut self, job: WebReplayWorkerJob, estimated_duration_s: f64) {
        let label = job.label().to_string();
        self.web_run_started_time_s = web_now_s();
        self.web_run_estimated_duration_s = estimated_duration_s;
        self.finish_web_replay_worker();
        *self.web_datasets.pending_replay.borrow_mut() = None;
        *self.web_datasets.pending_progress.borrow_mut() = None;
        self.web_datasets.replay_job_id = self.web_datasets.replay_job_id.wrapping_add(1);
        let job_id = self.web_datasets.replay_job_id;
        self.web_run_progress = 0.02;

        let worker_options = WorkerOptions::new();
        worker_options.set_type(WorkerType::Module);
        let worker = match Worker::new_with_options("replay_worker.js", &worker_options) {
            Ok(worker) => worker,
            Err(err) => {
                self.web_status =
                    format!("Failed to start replay worker: {}", js_error_string(err));
                return;
            }
        };

        let pending = Rc::clone(&self.web_datasets.pending_replay);
        let pending_progress = Rc::clone(&self.web_datasets.pending_progress);
        let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
            let value = event.data();
            let output_job_id = Reflect::get(&value, &JsValue::from_str("jobId"))
                .ok()
                .and_then(|v| v.as_f64())
                .map(|v| v as u64)
                .unwrap_or(job_id);
            let message_type = Reflect::get(&value, &JsValue::from_str("type"))
                .ok()
                .and_then(|v| v.as_string());
            if message_type.as_deref() == Some("progress") {
                let progress = Reflect::get(&value, &JsValue::from_str("progress"))
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0) as f32;
                *pending_progress.borrow_mut() = Some(WebReplayProgress {
                    job_id: output_job_id,
                    progress,
                });
                return;
            }
            let ok = Reflect::get(&value, &JsValue::from_str("ok"))
                .ok()
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let label = Reflect::get(&value, &JsValue::from_str("label"))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_else(|| "CSV replay".to_string());
            let source = Reflect::get(&value, &JsValue::from_str("source"))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_else(|| "csv".to_string());
            if ok {
                let output = WebReplayWorkerOutput {
                    source,
                    label,
                    imu_name: Reflect::get(&value, &JsValue::from_str("imuName"))
                        .ok()
                        .and_then(|v| v.as_string())
                        .unwrap_or_else(|| "imu.csv".to_string()),
                    gnss_name: Reflect::get(&value, &JsValue::from_str("gnssName"))
                        .ok()
                        .and_then(|v| v.as_string())
                        .unwrap_or_else(|| "gnss.csv".to_string()),
                    json: Reflect::get(&value, &JsValue::from_str("json"))
                        .ok()
                        .and_then(|v| v.as_string())
                        .unwrap_or_default(),
                };
                *pending.borrow_mut() = Some(WebReplayTaskResult::Complete {
                    job_id: output_job_id,
                    result: Ok(output),
                });
            } else {
                let err = Reflect::get(&value, &JsValue::from_str("error"))
                    .ok()
                    .and_then(|v| v.as_string())
                    .unwrap_or_else(|| "replay worker failed".to_string());
                *pending.borrow_mut() = Some(WebReplayTaskResult::Complete {
                    job_id: output_job_id,
                    result: Err(format!("{label}: {err}")),
                });
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        let pending = Rc::clone(&self.web_datasets.pending_replay);
        let onerror = Closure::wrap(Box::new(move |event: ErrorEvent| {
            *pending.borrow_mut() = Some(WebReplayTaskResult::Complete {
                job_id,
                result: Err(format!("replay worker error: {}", event.message())),
            });
        }) as Box<dyn FnMut(ErrorEvent)>);

        worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));

        let request = web_replay_worker_request(job_id, &job);
        let _ = Reflect::set(
            &request,
            &JsValue::from_str("label"),
            &JsValue::from_str(&label),
        );

        if let Err(err) = worker.post_message(&request) {
            self.web_status = format!("Failed to start replay: {}", js_error_string(err));
            worker.terminate();
            return;
        }

        self.web_datasets.loading_replay = true;
        self.web_status = format!("Running replay: {label}");
        self.web_datasets.replay_worker = Some(worker);
        self.web_datasets.replay_onmessage = Some(onmessage);
        self.web_datasets.replay_onerror = Some(onerror);
    }

    #[cfg(target_arch = "wasm32")]
    fn finish_web_replay_worker(&mut self) {
        if let Some(worker) = self.web_datasets.replay_worker.take() {
            worker.terminate();
        }
        *self.web_datasets.pending_progress.borrow_mut() = None;
        self.web_datasets.replay_onmessage = None;
        self.web_datasets.replay_onerror = None;
    }

    #[cfg(target_arch = "wasm32")]
    fn refresh_from_web_synthetic(&mut self) {
        let (label, text) = self.web_scenario.scenario_text();
        self.web_input_mode = WebInputMode::Synthetic;
        self.start_web_replay_worker(
            WebReplayWorkerJob::Synthetic {
                label: self.web_scenario.display_label().to_string(),
                motion_label: label.to_string(),
                motion_text: text.to_string(),
            },
            4.0,
        );
    }

    fn draw_map_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = ctx;
        ui.horizontal_wrapped(|ui| {
            ui.label("Map traces");
            ui.checkbox(&mut self.show_heading, "show heading");
            ui.checkbox(&mut self.show_gnss_map, "show GNSS");
            ui.checkbox(&mut self.show_eskf, "show ESKF");
            ui.checkbox(&mut self.show_loose, "show Loose");
            if ui.button("Recenter").clicked() {
                self.map_memory.follow_my_position();
            }
        });
        #[cfg(target_arch = "wasm32")]
        ui.horizontal_wrapped(|ui| {
            ui.label("Mapbox token");
            let token_width = ui.available_width().clamp(120.0, 260.0);
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.web_mapbox_token)
                    .desired_width(token_width)
                    .password(true),
            );
            if response.changed() {
                web_remember_mapbox_token(&self.web_mapbox_token);
                self.refresh_map_tiles(ctx);
                self.web_mapbox_token_applied = self.web_mapbox_token.clone();
            } else if self.web_mapbox_token != self.web_mapbox_token_applied {
                self.refresh_map_tiles(ctx);
                self.web_mapbox_token_applied = self.web_mapbox_token.clone();
            }
        });
        ui.horizontal_wrapped(|ui| {
            let visuals = ui.visuals().clone();
            ui.colored_label(map_trace_color("GNSS", &visuals), "GNSS");
            ui.colored_label(map_trace_color("ESKF path (lon,lat)", &visuals), "ESKF");
            ui.colored_label(map_trace_color("Loose path (lon,lat)", &visuals), "Loose");
            ui.colored_label(
                map_trace_color("ESKF path during GNSS outage (lon,lat)", &visuals),
                "ESKF during GNSS outage",
            );
            ui.colored_label(map_heading_color(&visuals), "ESKF heading");
        });
    }

    fn draw_map_body(&mut self, ui: &mut egui::Ui, size: egui::Vec2, cursor_t_s: Option<f64>) {
        if self.data_origin == DataOrigin::Synthetic {
            self.draw_synthetic_trajectory_body(ui, size, cursor_t_s);
            return;
        }

        let mut map_traces: Vec<&Trace> = self.data.eskf_map.iter().collect();
        if !self.show_gnss_map {
            map_traces.retain(|t| {
                !t.name.contains("GNSS")
                    && !t.name.contains("GNSS reference")
                    && !t.name.contains("NAV")
                    && !t.name.contains("truth")
            });
        }
        if !self.show_eskf {
            map_traces.retain(|t| !t.name.contains("ESKF"));
        }
        if self.show_loose {
            map_traces.extend(self.data.loose_map.iter());
        }
        let mut headings: Vec<&HeadingSample> = self.data.eskf_map_heading.iter().collect();
        if self.show_loose {
            headings.extend(self.data.loose_map_heading.iter());
        }
        let track = TrackOverlay {
            traces: map_traces,
            headings,
            show_heading: self.show_heading,
            cursor_t_s,
        };
        ui.add_sized(
            size,
            Map::new(
                Some(&mut self.map_tiles),
                &mut self.map_memory,
                self.map_center,
            )
            .with_plugin(track)
            .double_click_to_zoom(true),
        );
    }

    fn draw_synthetic_trajectory_body(
        &self,
        ui: &mut egui::Ui,
        size: egui::Vec2,
        cursor_t_s: Option<f64>,
    ) {
        let traces = synthetic_trajectory_traces(
            &self.data,
            ui.visuals(),
            self.show_gnss_map,
            self.show_eskf,
            self.show_loose,
        );
        if traces.is_empty() {
            ui.allocate_ui(size, |ui| {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("No local trajectory").weak());
                });
            });
            return;
        }

        let cursor_point = cursor_t_s.and_then(|t_s| synthetic_cursor_point(&self.data, t_s));
        let cursor_color = cursor_marker_color(ui.visuals());
        Plot::new("synthetic_local_trajectory")
            .width(size.x)
            .height(size.y)
            .data_aspect(1.0)
            .grid_spacing(subtle_plot_grid_spacing())
            .x_grid_spacer(subdued_plot_grid_marks)
            .y_grid_spacer(subdued_plot_grid_marks)
            .legend(Legend::default())
            .include_x(0.0)
            .include_y(0.0)
            .x_axis_label("East [m]")
            .y_axis_label("North [m]")
            .x_axis_formatter(|mark, _range| format!("{:.0}", mark.value))
            .y_axis_formatter(|mark, _range| format!("{:.0}", mark.value))
            .allow_drag(true)
            .allow_zoom(true)
            .allow_scroll(true)
            .allow_boxed_zoom(true)
            .allow_axis_zoom_drag(true)
            .show(ui, |plot_ui| {
                for trace in traces {
                    let points: PlotPoints<'_> =
                        decimate_trajectory_points(&trace.points, SYNTHETIC_TRAJECTORY_MAX_POINTS)
                            .into();
                    plot_ui.line(Line::new(trace.name, points).color(trace.color));
                }
                if let Some(point) = cursor_point {
                    plot_ui.points(
                        Points::new("Cursor", vec![point])
                            .radius(5.0)
                            .color(cursor_color),
                    );
                }
            });
    }

    fn draw_overview_page(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                page_header(
                    ui,
                    "Overview",
                    "Primary signals, references, and filter estimates.",
                );
                let speed: Vec<Trace> = trace_refs(&self.data.speed).into_iter().cloned().collect();
                let mount: Vec<Trace> = concat_trace_refs_matching(
                    [
                        self.data.eskf_misalignment.as_slice(),
                        self.data.loose_misalignment.as_slice(),
                        self.data.align_cmp_att.as_slice(),
                    ],
                    &[
                        "mount roll",
                        "mount pitch",
                        "mount yaw",
                        "Align roll",
                        "Align pitch",
                        "Align yaw",
                        "Reference mount roll",
                        "Reference mount pitch",
                        "Reference mount yaw",
                    ],
                )
                .into_iter()
                .cloned()
                .collect();
                let attitude: Vec<Trace> = concat_trace_refs_matching(
                    [
                        self.data.eskf_cmp_att.as_slice(),
                        self.data.loose_cmp_att.as_slice(),
                        self.data.orientation.as_slice(),
                    ],
                    &["roll", "pitch", "yaw"],
                )
                .into_iter()
                .cloned()
                .collect();
                let biases: Vec<Trace> = concat_trace_refs([
                    self.data.eskf_bias_gyro.as_slice(),
                    self.data.loose_bias_gyro.as_slice(),
                    self.data.eskf_bias_accel.as_slice(),
                    self.data.loose_bias_accel.as_slice(),
                ])
                .into_iter()
                .cloned()
                .collect();

                let tile_height = overview_tile_height(ui.available_width());
                let cursor_t_s = self.overview_cursor_t_s;
                let mut hovered_t_s = None;
                if ui.available_width() < 900.0 {
                    if let Some(t_s) = draw_plot_with_cursor_time(
                        ui,
                        "Vehicle Speed",
                        speed.iter(),
                        true,
                        self.max_points_per_trace,
                        cursor_t_s,
                    ) {
                        hovered_t_s = Some(t_s);
                    }
                    if let Some(t_s) = draw_plot_with_cursor_time(
                        ui,
                        "Mount Angles: Reference / Align / ESKF / Loose",
                        mount.iter(),
                        true,
                        self.max_points_per_trace,
                        hovered_t_s.or(cursor_t_s),
                    ) {
                        hovered_t_s = Some(t_s);
                        draw_orthogonal_views_popup(
                            ui,
                            "Mount Alignment",
                            &mount,
                            t_s,
                            OrthogonalViewKind::Mount,
                        );
                    }
                    if let Some(t_s) = draw_plot_with_cursor_time(
                        ui,
                        "Vehicle Attitude: Reference / ESKF / Loose",
                        attitude.iter(),
                        true,
                        self.max_points_per_trace,
                        hovered_t_s.or(cursor_t_s),
                    ) {
                        hovered_t_s = Some(t_s);
                        draw_orthogonal_views_popup(
                            ui,
                            "Vehicle Attitude",
                            &attitude,
                            t_s,
                            OrthogonalViewKind::Vehicle,
                        );
                    }
                    if let Some(t_s) = draw_plot_with_cursor_time(
                        ui,
                        "Biases: ESKF / Loose",
                        biases.iter(),
                        true,
                        self.max_points_per_trace,
                        hovered_t_s.or(cursor_t_s),
                    ) {
                        hovered_t_s = Some(t_s);
                    }
                    draw_map_tile(ui, "Map", tile_height, |ui, size| {
                        self.draw_map_body(ui, size, hovered_t_s.or(cursor_t_s));
                    });
                } else {
                    ui.columns(2, |cols| {
                        if let Some(t_s) = draw_plot_with_cursor_time(
                            &mut cols[0],
                            "Vehicle Speed",
                            speed.iter(),
                            true,
                            self.max_points_per_trace,
                            cursor_t_s,
                        ) {
                            hovered_t_s = Some(t_s);
                        }
                        if let Some(t_s) = draw_plot_with_cursor_time(
                            &mut cols[0],
                            "Mount Angles: Reference / Align / ESKF / Loose",
                            mount.iter(),
                            true,
                            self.max_points_per_trace,
                            hovered_t_s.or(cursor_t_s),
                        ) {
                            hovered_t_s = Some(t_s);
                            draw_orthogonal_views_popup(
                                &mut cols[0],
                                "Mount Alignment",
                                &mount,
                                t_s,
                                OrthogonalViewKind::Mount,
                            );
                        }
                        draw_map_tile(&mut cols[1], "Map", tile_height, |ui, size| {
                            self.draw_map_body(ui, size, hovered_t_s.or(cursor_t_s));
                        });
                        if let Some(t_s) = draw_plot_with_cursor_time(
                            &mut cols[0],
                            "Vehicle Attitude: Reference / ESKF / Loose",
                            attitude.iter(),
                            true,
                            self.max_points_per_trace,
                            hovered_t_s.or(cursor_t_s),
                        ) {
                            hovered_t_s = Some(t_s);
                            draw_orthogonal_views_popup(
                                &mut cols[0],
                                "Vehicle Attitude",
                                &attitude,
                                t_s,
                                OrthogonalViewKind::Vehicle,
                            );
                        }
                        if let Some(t_s) = draw_plot_with_cursor_time(
                            &mut cols[1],
                            "Biases: ESKF / Loose",
                            biases.iter(),
                            true,
                            self.max_points_per_trace,
                            hovered_t_s.or(cursor_t_s),
                        ) {
                            hovered_t_s = Some(t_s);
                        }
                    });
                }
                if self.overview_cursor_t_s != hovered_t_s {
                    self.overview_cursor_t_s = hovered_t_s;
                    ui.ctx().request_repaint();
                }
            });
    }

    #[cfg(target_arch = "wasm32")]
    fn publish_web_perf(&mut self, ctx: &egui::Context) {
        if !self.web_perf.enabled {
            return;
        }
        let now_s = ctx.input(|i| i.time);
        if self.web_perf.frame_count == 0 {
            self.web_perf.start_time_s = now_s;
            self.web_perf.last_time_s = now_s;
        }
        self.web_perf.frame_count = self.web_perf.frame_count.saturating_add(1);
        let dt = now_s - self.web_perf.last_time_s;
        if dt > 0.0 {
            let fps = 1.0 / dt;
            self.web_perf.fps_ema = if self.web_perf.fps_ema > 0.0 {
                self.web_perf.fps_ema * 0.9 + fps * 0.1
            } else {
                fps
            };
        }
        self.web_perf.last_time_s = now_s;

        let elapsed_s = (now_s - self.web_perf.start_time_s).max(0.0);
        let avg_fps = if elapsed_s > 0.0 {
            self.web_perf.frame_count as f64 / elapsed_s
        } else {
            0.0
        };
        let sample = Object::new();
        let _ = Reflect::set(
            &sample,
            &JsValue::from_str("frameCount"),
            &JsValue::from_f64(self.web_perf.frame_count as f64),
        );
        let _ = Reflect::set(
            &sample,
            &JsValue::from_str("elapsedSec"),
            &JsValue::from_f64(elapsed_s),
        );
        let _ = Reflect::set(
            &sample,
            &JsValue::from_str("avgFps"),
            &JsValue::from_f64(avg_fps),
        );
        let _ = Reflect::set(
            &sample,
            &JsValue::from_str("emaFps"),
            &JsValue::from_f64(self.web_perf.fps_ema),
        );
        let _ = Reflect::set(
            &sample,
            &JsValue::from_str("maxPointsPerTrace"),
            &JsValue::from_f64(self.max_points_per_trace as f64),
        );
        let _ = Reflect::set(
            &sample,
            &JsValue::from_str("status"),
            &JsValue::from_str(&self.web_status),
        );
        if let Some(window) = eframe::web_sys::window() {
            let _ = Reflect::set(
                window.as_ref(),
                &JsValue::from_str("__imuGnssFusionPerf"),
                sample.as_ref(),
            );
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn consume_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }
        for file in dropped {
            let Some(bytes) = file.bytes else {
                self.web_status = format!("{} has no available browser bytes", file.name);
                continue;
            };
            let text = match std::str::from_utf8(bytes.as_ref()) {
                Ok(text) => text.to_string(),
                Err(err) => {
                    self.web_status = format!("{} is not UTF-8 CSV text: {err}", file.name);
                    continue;
                }
            };
            let lower = file.name.to_ascii_lowercase();
            let named = NamedText {
                name: file.name.clone(),
                text,
            };
            if lower.contains("reference_attitude") || lower.contains("ref_att") {
                self.web_reference_attitude_csv = Some(named);
            } else if lower.contains("reference_mount") || lower.contains("ref_mount") {
                self.web_reference_mount_csv = Some(named);
            } else if lower.contains("gnss") {
                self.web_gnss_csv = Some(named);
            } else if lower.contains("imu") || lower.contains("acc") || lower.contains("gyro") {
                self.web_imu_csv = Some(named);
            } else if self.web_imu_csv.is_none() {
                self.web_imu_csv = Some(named);
            } else {
                self.web_gnss_csv = Some(named);
            }
        }
        self.web_input_mode = WebInputMode::RealData;
        self.web_real_data_source = WebRealDataSource::DroppedCsv;
        self.data_origin = DataOrigin::Real;
        self.web_status =
            "Dropped file(s) staged. Select Experimental/real data, then click Run.".to_string();
    }
}

#[cfg(target_arch = "wasm32")]
impl WebSyntheticScenario {
    fn display_label(self) -> &'static str {
        match self {
            Self::CityBlocks => "City blocks",
            Self::FigureEight => "Figure eight",
            Self::StraightAccelBrake => "Straight accel/brake",
        }
    }

    fn scenario_text(self) -> (&'static str, &'static str) {
        match self {
            Self::CityBlocks => ("city_blocks_builtin.scenario", CITY_BLOCKS_SCENARIO),
            Self::FigureEight => ("figure8_builtin.csv", FIGURE_EIGHT_CSV),
            Self::StraightAccelBrake => {
                ("straight_accel_brake_builtin.csv", STRAIGHT_ACCEL_BRAKE_CSV)
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
const CITY_BLOCKS_SCENARIO: &str = r#"
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 20s
repeat 3 {
    accelerate 1.0m/s^2 for 8s
    wait 10s
    turn left 10dps for 9s
    wait 10s
    brake 1.0m/s^2 for 8s
    wait 10s
    accelerate 1.0m/s^2 for 8s
    wait 10s
    turn right 10dps for 9s
    wait 10s
    brake 1.0m/s^2 for 8s
    wait 10s
}
"#;

#[cfg(target_arch = "wasm32")]
const FIGURE_EIGHT_CSV: &str = r#"ini lat (deg),ini lon (deg),ini alt (m),ini vx_body (m/s),ini vy_body (m/s),ini vz_body (m/s),ini yaw (deg),ini pitch (deg),ini roll (deg)
32,120,0,0,0,0,0,0,0
command type,yaw (deg),pitch (deg),roll (deg),vx_body (m/s),vy_body (m/s),vz_body (m/s),command duration (s),GNSS visibility
1,0,0,0,0,0,0,20,1
1,0,0,0,0.6,0,0,20,1
1,0,0,0,0,0,0,10,1
1,10,0,0,0,0,0,36,1
1,-10,0,0,0,0,0,36,1
1,10,0,0,0,0,0,36,1
1,-10,0,0,0,0,0,36,1
1,0,0,0,-0.6666667,0,0,18,1
"#;

#[cfg(target_arch = "wasm32")]
const STRAIGHT_ACCEL_BRAKE_CSV: &str = r#"ini lat (deg),ini lon (deg),ini alt (m),ini vx_body (m/s),ini vy_body (m/s),ini vz_body (m/s),ini yaw (deg),ini pitch (deg),ini roll (deg)
32,120,0,0,0,0,0,0,0
command type,yaw (deg),pitch (deg),roll (deg),vx_body (m/s),vy_body (m/s),vz_body (m/s),command duration (s),GNSS visibility
1,0,0,0,0,0,0,20,1
1,0,0,0,0.5,0,0,20,1
1,0,0,0,0,0,0,20,1
1,0,0,0,-0.5,0,0,20,1
1,0,0,0,0,0,0,15,1
1,0,0,0,0.5,0,0,20,1
1,0,0,0,0,0,0,20,1
1,0,0,0,-0.5,0,0,20,1
1,0,0,0,0,0,0,15,1
"#;

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        super::theme::apply(ctx, current_ui_density(), self.ui_theme);

        #[cfg(target_arch = "wasm32")]
        self.consume_dropped_files(ctx);

        #[cfg(target_arch = "wasm32")]
        self.poll_web_dataset_tasks();
        #[cfg(target_arch = "wasm32")]
        self.poll_web_replay_progress();
        #[cfg(target_arch = "wasm32")]
        self.poll_web_replay_tasks();

        #[cfg(target_arch = "wasm32")]
        self.publish_web_perf(ctx);

        #[cfg(target_os = "macos")]
        if ctx.input(|i| i.viewport().close_requested()) {
            std::process::exit(0);
        }

        #[cfg(target_arch = "wasm32")]
        ctx.request_repaint();
        #[cfg(not(target_arch = "wasm32"))]
        if matches!(self.page, Page::Overview | Page::Map) {
            ctx.request_repaint_after(Duration::from_millis(16));
        }

        egui::TopBottomPanel::top("top_controls").show(ctx, |ui| {
            let mut replay_changed = false;
            let mut apply_replay = false;
            #[cfg(target_arch = "wasm32")]
            let now_s = eframe::web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now() / 1000.0)
                .unwrap_or_else(|| ctx.input(|i| i.time));
            #[cfg(not(target_arch = "wasm32"))]
            let now_s = ctx.input(|i| i.time);
            let fps = if self.last_frame_time_s > 0.0 {
                let dt = (now_s - self.last_frame_time_s).max(0.0);
                if dt > 0.0 { (1.0 / dt) as f32 } else { 0.0 }
            } else {
                0.0
            };
            self.last_frame_time_s = now_s;
            if fps > 0.0 && self.fps_ema <= 0.0 {
                self.fps_ema = fps;
            } else if fps > 0.0 {
                self.fps_ema = self.fps_ema * 0.92 + fps * 0.08;
            }
            #[cfg(target_arch = "wasm32")]
            {
                if self.fps_ema < 55.0 {
                    self.max_points_per_trace = (self.max_points_per_trace as f32 * 0.80) as usize;
                } else if self.fps_ema > 58.0 {
                    self.max_points_per_trace = (self.max_points_per_trace as f32 * 1.03) as usize;
                }
                self.max_points_per_trace = self
                    .max_points_per_trace
                    .clamp(50, WEB_MAX_POINTS_PER_TRACE);
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                if self.fps_ema < 24.0 {
                    self.max_points_per_trace = (self.max_points_per_trace as f32 * 0.85) as usize;
                } else if self.fps_ema > 50.0 {
                    self.max_points_per_trace = (self.max_points_per_trace as f32 * 1.08) as usize;
                }
                self.max_points_per_trace = self.max_points_per_trace.clamp(300, 6000);
            }
            ui.horizontal_wrapped(|ui| {
                ui.heading("IMU/GNSS Filter Evaluation");
                ui.separator();
                ui.label(if self.has_itow {
                    "time axis: relative seconds from first GNSS epoch"
                } else {
                    "time axis: relative seconds"
                });
                ui.separator();
                ui.label(format!("FPS {:.1}", self.fps_ema.max(fps)));
                ui.separator();
                ui.label("Theme");
                let mut selected_theme = self.ui_theme;
                ui.selectable_value(&mut selected_theme, UiTheme::Light, "Light");
                ui.selectable_value(&mut selected_theme, UiTheme::Dark, "Dark");
                if selected_theme != self.ui_theme {
                    self.set_ui_theme(selected_theme, ctx);
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.page, Page::Overview, "Overview");
                ui.selectable_value(&mut self.page, Page::Map, "Map");
                ui.selectable_value(&mut self.page, Page::Motion, "Motion");
                ui.selectable_value(&mut self.page, Page::Mount, "Mount");
                ui.selectable_value(&mut self.page, Page::Calibration, "Calibration");
                ui.selectable_value(&mut self.page, Page::Sensors, "Sensors");
                ui.selectable_value(&mut self.page, Page::Diagnostics, "Diagnostics");
            });
            {
                #[cfg(target_arch = "wasm32")]
                egui::CollapsingHeader::new("Inputs")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.selectable_value(
                                &mut self.web_input_mode,
                                WebInputMode::Synthetic,
                                "Synthetic",
                            );
                            ui.selectable_value(
                                &mut self.web_input_mode,
                                WebInputMode::RealData,
                                "Experimental/real data",
                            );
                        });
                        ui.horizontal_wrapped(|ui| {
                            ui.label(match self.web_input_mode {
                                WebInputMode::Synthetic => "Scenario:",
                                WebInputMode::RealData => "Input:",
                            });
                            match self.web_input_mode {
                                WebInputMode::Synthetic => {
                                    egui::ComboBox::from_id_salt("web_synthetic_scenario_select")
                                        .selected_text(self.web_scenario.display_label())
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut self.web_scenario,
                                                WebSyntheticScenario::CityBlocks,
                                                WebSyntheticScenario::CityBlocks.display_label(),
                                            );
                                            ui.selectable_value(
                                                &mut self.web_scenario,
                                                WebSyntheticScenario::FigureEight,
                                                WebSyntheticScenario::FigureEight.display_label(),
                                            );
                                            ui.selectable_value(
                                                &mut self.web_scenario,
                                                WebSyntheticScenario::StraightAccelBrake,
                                                WebSyntheticScenario::StraightAccelBrake
                                                    .display_label(),
                                            );
                                        });
                                }
                                WebInputMode::RealData => {
                                    let selected_text = match self.web_real_data_source {
                                        WebRealDataSource::DroppedCsv => {
                                            "Dropped CSV files".to_string()
                                        }
                                        WebRealDataSource::ManifestDataset => self
                                            .web_datasets
                                            .datasets
                                            .get(self.web_datasets.selected)
                                            .map(WebDatasetEntry::display_label)
                                            .unwrap_or_else(|| "No manifest entries".to_string()),
                                    };
                                    egui::ComboBox::from_id_salt("web_real_data_select")
                                        .selected_text(selected_text)
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut self.web_real_data_source,
                                                WebRealDataSource::DroppedCsv,
                                                "Dropped CSV files",
                                            );
                                            for (idx, dataset) in
                                                self.web_datasets.datasets.iter().enumerate()
                                            {
                                                let selected = self.web_real_data_source
                                                    == WebRealDataSource::ManifestDataset
                                                    && self.web_datasets.selected == idx;
                                                if ui
                                                    .selectable_label(
                                                        selected,
                                                        dataset.display_label(),
                                                    )
                                                    .clicked()
                                                {
                                                    self.web_real_data_source =
                                                        WebRealDataSource::ManifestDataset;
                                                    self.web_datasets.selected = idx;
                                                }
                                            }
                                        });
                                }
                            }

                            let run_enabled = match self.web_input_mode {
                                WebInputMode::Synthetic => true,
                                WebInputMode::RealData => match self.web_real_data_source {
                                    WebRealDataSource::DroppedCsv => {
                                        !self.web_datasets.loading_replay
                                            && self.web_imu_csv.is_some()
                                            && self.web_gnss_csv.is_some()
                                    }
                                    WebRealDataSource::ManifestDataset => {
                                        !self.web_datasets.loading_dataset
                                            && !self.web_datasets.loading_replay
                                            && !self.web_datasets.loading_manifest
                                            && !self.web_datasets.datasets.is_empty()
                                    }
                                },
                            };
                            let run_text = match self.web_input_mode {
                                _ if self.web_datasets.loading_replay => "Running replay...",
                                WebInputMode::RealData if self.web_datasets.loading_dataset => {
                                    "Loading dataset..."
                                }
                                _ => "Run",
                            };
                            let run_busy = self.web_datasets.loading_dataset
                                || self.web_datasets.loading_replay;
                            if draw_web_run_button(
                                ui,
                                run_enabled,
                                run_busy,
                                self.web_run_progress,
                                run_text,
                            ) {
                                match self.web_input_mode {
                                    WebInputMode::Synthetic => self.refresh_from_web_synthetic(),
                                    WebInputMode::RealData => match self.web_real_data_source {
                                        WebRealDataSource::DroppedCsv => {
                                            self.refresh_from_generic_csv();
                                        }
                                        WebRealDataSource::ManifestDataset => {
                                            self.start_web_dataset_load();
                                        }
                                    },
                                }
                            }
                            if ui
                                .add_enabled(
                                    !self.web_datasets.loading_manifest
                                        && !self.web_datasets.loading_dataset
                                        && !self.web_datasets.loading_replay,
                                    egui::Button::new("Reload manifest"),
                                )
                                .clicked()
                            {
                                self.start_web_manifest_load();
                            }
                        });
                        match self.web_input_mode {
                            WebInputMode::Synthetic => {}
                            WebInputMode::RealData => {
                                let imu_name = self
                                    .web_imu_csv
                                    .as_ref()
                                    .map(|f| f.name.as_str())
                                    .unwrap_or("no imu.csv");
                                let gnss_name = self
                                    .web_gnss_csv
                                    .as_ref()
                                    .map(|f| f.name.as_str())
                                    .unwrap_or("no gnss.csv");
                                let ref_att = self
                                    .web_reference_attitude_csv
                                    .as_ref()
                                    .map(|f| f.name.as_str())
                                    .unwrap_or("no reference attitude");
                                ui.label(format!("CSV: {imu_name} / {gnss_name} / {ref_att}"));
                                if self.web_datasets.loading_manifest {
                                    ui.label("loading manifest...");
                                } else if self.web_datasets.datasets.is_empty() {
                                    ui.label("no manifest entries");
                                }
                                if let WebRealDataSource::ManifestDataset =
                                    self.web_real_data_source
                                    && let Some(dataset) =
                                        self.web_datasets.datasets.get(self.web_datasets.selected)
                                    && let Some(description) = dataset.description.as_deref()
                                {
                                    ui.label(description);
                                }
                            }
                        }
                        ui.label(&self.web_status);
                    });
                if let Some(replay) = self.replay.as_mut() {
                    egui::CollapsingHeader::new("Replay")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label("Adjust controls, then click Apply to rebuild the replay.");
                            ui.horizontal_wrapped(|ui| {
                                ui.label("Mount source:");
                                replay_changed |= ui
                                    .selectable_value(
                                        &mut replay.misalignment,
                                        EkfImuSource::Internal,
                                        "latched align",
                                    )
                                    .changed();
                                replay_changed |= ui
                                    .selectable_value(
                                        &mut replay.misalignment,
                                        EkfImuSource::External,
                                        "follow align",
                                    )
                                    .changed();
                                replay_changed |= ui
                                    .selectable_value(
                                        &mut replay.misalignment,
                                        EkfImuSource::Ref,
                                        "reference",
                                    )
                                    .changed();
                            });
                            ui.horizontal_wrapped(|ui| {
                                ui.label("GNSS pos R");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut replay.ekf_cfg.gnss_pos_r_scale)
                                            .speed(0.01)
                                            .range(0.001..=10.0),
                                    )
                                    .changed();
                                ui.label("GNSS vel R");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut replay.ekf_cfg.gnss_vel_r_scale)
                                            .speed(0.01)
                                            .range(0.001..=10.0),
                                    )
                                    .changed();
                                ui.label("NHC R");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut replay.ekf_cfg.r_body_vel)
                                            .speed(0.001)
                                            .range(0.0001..=1000.0),
                                    )
                                    .changed();
                                ui.label("Yaw init m/s");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.yaw_init_speed_mps,
                                        )
                                        .speed(0.1)
                                        .range(0.0..=20.0),
                                    )
                                    .changed();
                                ui.label("Yaw sigma deg");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.yaw_init_sigma_deg,
                                        )
                                        .speed(0.5)
                                        .range(0.0..=90.0),
                                    )
                                    .changed();
                                ui.label("Mount init deg");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.mount_init_sigma_deg,
                                        )
                                        .speed(0.5)
                                        .range(0.0..=90.0),
                                    )
                                    .changed();
                            });
                            ui.horizontal_wrapped(|ui| {
                                ui.label("Mount RW");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.mount_align_rw_var,
                                        )
                                        .speed(1.0e-8)
                                        .range(0.0..=1.0e-3),
                                    )
                                    .changed();
                                ui.label("Mount min scale");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.mount_update_min_scale,
                                        )
                                        .speed(0.001)
                                        .range(0.0..=1.0),
                                    )
                                    .changed();
                                ui.label("Mount ramp s");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.mount_update_ramp_time_s,
                                        )
                                        .speed(10.0)
                                        .range(0.0..=20000.0),
                                    )
                                    .changed();
                                ui.label("Mount gate m/s");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.mount_update_innovation_gate_mps,
                                        )
                                        .speed(0.001)
                                        .range(0.0..=10.0),
                                    )
                                    .changed();
                                ui.label("Mount yaw gate deg/s");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.mount_update_yaw_rate_gate_dps,
                                        )
                                        .speed(0.1)
                                        .range(0.0..=90.0),
                                    )
                                    .changed();
                                ui.label("Align handoff s");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.align_handoff_delay_s,
                                        )
                                        .speed(1.0)
                                        .range(0.0..=600.0),
                                    )
                                    .changed();
                                replay_changed |= ui
                                    .checkbox(
                                        &mut replay.ekf_cfg.freeze_misalignment_states,
                                        "Freeze mount states",
                                    )
                                    .changed();
                            });
                            ui.horizontal_wrapped(|ui| {
                                ui.label("Mount settle s");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.mount_settle_time_s,
                                        )
                                        .speed(10.0)
                                        .range(0.0..=1000.0),
                                    )
                                    .changed();
                                ui.label("Release sigma deg");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.mount_settle_release_sigma_deg,
                                        )
                                        .speed(0.5)
                                        .range(0.0..=90.0),
                                    )
                                    .changed();
                                replay_changed |= ui
                                    .checkbox(
                                        &mut replay.ekf_cfg.mount_settle_zero_cross_covariance,
                                        "Zero mount cross-cov",
                                    )
                                    .changed();
                            });
                            ui.horizontal_wrapped(|ui| {
                                ui.label("GNSS pos->mount");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.gnss_pos_mount_scale,
                                        )
                                        .speed(0.01)
                                        .range(0.0..=1.0),
                                    )
                                    .changed();
                                ui.label("GNSS vel->mount");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.gnss_vel_mount_scale,
                                        )
                                        .speed(0.01)
                                        .range(0.0..=1.0),
                                    )
                                    .changed();
                                ui.label("Gyro bias init dps");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.gyro_bias_init_sigma_dps,
                                        )
                                        .speed(0.01)
                                        .range(0.0..=10.0),
                                    )
                                    .changed();
                                ui.label("Accel bias init");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.accel_bias_init_sigma_mps2,
                                        )
                                        .speed(0.01)
                                        .range(0.0..=10.0),
                                    )
                                    .changed();
                                ui.label("Vehicle speed R");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut replay.ekf_cfg.r_vehicle_speed)
                                            .speed(0.01)
                                            .range(0.0..=10.0),
                                    )
                                    .changed();
                            });
                            ui.horizontal_wrapped(|ui| {
                                ui.label("Zero vel R");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut replay.ekf_cfg.r_zero_vel)
                                            .speed(0.01)
                                            .range(0.0..=10.0),
                                    )
                                    .changed();
                                ui.label("Stationary accel R");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.r_stationary_accel,
                                        )
                                        .speed(0.01)
                                        .range(0.0..=10.0),
                                    )
                                    .changed();
                                ui.label("Predict decimation");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut replay.ekf_cfg.predict_imu_decimation,
                                        )
                                        .speed(1)
                                        .range(1..=32),
                                    )
                                    .changed();
                            });
                            ui.horizontal_wrapped(|ui| {
                                let mut lpf_on = replay.ekf_cfg.predict_imu_lpf_cutoff_hz.is_some();
                                if ui.checkbox(&mut lpf_on, "Predict IMU LPF").changed() {
                                    replay.ekf_cfg.predict_imu_lpf_cutoff_hz = if lpf_on {
                                        Some(
                                            replay
                                                .ekf_cfg
                                                .predict_imu_lpf_cutoff_hz
                                                .unwrap_or(150.0),
                                        )
                                    } else {
                                        None
                                    };
                                    replay_changed = true;
                                }
                                if let Some(cutoff_hz) =
                                    replay.ekf_cfg.predict_imu_lpf_cutoff_hz.as_mut()
                                {
                                    replay_changed |= ui
                                        .add(
                                            egui::DragValue::new(cutoff_hz)
                                                .speed(1.0)
                                                .range(1.0..=500.0),
                                        )
                                        .changed();
                                }
                                ui.label("Outage count");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut replay.gnss_outages.count)
                                            .speed(1)
                                            .range(0..=20),
                                    )
                                    .changed();
                                ui.label("Outage duration s");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut replay.gnss_outages.duration_s)
                                            .speed(1.0)
                                            .range(0.0..=300.0),
                                    )
                                    .changed();
                                ui.label("Outage seed");
                                replay_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut replay.gnss_outages.seed)
                                            .speed(1)
                                            .range(0..=100000),
                                    )
                                    .changed();
                            });
                            ui.horizontal_wrapped(|ui| {
                                apply_replay = ui.button("Apply").clicked();
                                if let Some(status) = &self.replay_status {
                                    ui.label(status);
                                }
                            });
                        });
                }
            }
            if replay_changed {
                self.replay_status = Some("Pending changes".to_string());
            }
            if apply_replay {
                self.refresh_from_replay();
            }
        });

        let imu_cal_gyro: Vec<&Trace> = self
            .data
            .imu_cal_gyro
            .iter()
            .filter(|t| !t.name.starts_with("IMU measurement "))
            .collect();
        let imu_cal_accel: Vec<&Trace> = self
            .data
            .imu_cal_accel
            .iter()
            .filter(|t| !t.name.starts_with("IMU measurement "))
            .collect();

        match self.page {
            Page::Overview => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    self.draw_overview_page(ui);
                });
            }
            Page::Map => {
                egui::TopBottomPanel::top("map_controls")
                    .resizable(false)
                    .show(ctx, |ui| {
                        self.draw_map_controls(ui, ctx);
                    });
                egui::CentralPanel::default().show(ctx, |ui| {
                    self.draw_map_body(ui, ui.available_size(), None);
                });
            }
            Page::Motion => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_analysis_page(
                        ui,
                        "Motion",
                        "Position, velocity, and vehicle attitude comparisons.",
                        vec![
                            plot_spec(
                                "North Position: GNSS / ESKF / Loose",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_cmp_pos.as_slice(),
                                        self.data.loose_cmp_pos.as_slice(),
                                    ],
                                    &["posN"],
                                ),
                                true,
                            ),
                            plot_spec(
                                "East Position: GNSS / ESKF / Loose",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_cmp_pos.as_slice(),
                                        self.data.loose_cmp_pos.as_slice(),
                                    ],
                                    &["posE"],
                                ),
                                true,
                            ),
                            plot_spec(
                                "Down Position: GNSS / ESKF / Loose",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_cmp_pos.as_slice(),
                                        self.data.loose_cmp_pos.as_slice(),
                                    ],
                                    &["posD"],
                                ),
                                true,
                            ),
                            plot_spec(
                                "North Velocity: GNSS / ESKF / Loose",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_cmp_vel.as_slice(),
                                        self.data.loose_cmp_vel.as_slice(),
                                    ],
                                    &["velN", "vN "],
                                ),
                                true,
                            ),
                            plot_spec(
                                "East Velocity: GNSS / ESKF / Loose",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_cmp_vel.as_slice(),
                                        self.data.loose_cmp_vel.as_slice(),
                                    ],
                                    &["velE", "vE "],
                                ),
                                true,
                            ),
                            plot_spec(
                                "Down Velocity: GNSS / ESKF / Loose",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_cmp_vel.as_slice(),
                                        self.data.loose_cmp_vel.as_slice(),
                                    ],
                                    &["velD", "vD "],
                                ),
                                true,
                            ),
                            plot_spec(
                                "Roll: Reference / ESKF / Loose",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_cmp_att.as_slice(),
                                        self.data.loose_cmp_att.as_slice(),
                                        self.data.orientation.as_slice(),
                                    ],
                                    &["roll"],
                                ),
                                true,
                            ),
                            plot_spec(
                                "Pitch: Reference / ESKF / Loose",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_cmp_att.as_slice(),
                                        self.data.loose_cmp_att.as_slice(),
                                        self.data.orientation.as_slice(),
                                    ],
                                    &["pitch"],
                                ),
                                true,
                            ),
                            plot_spec(
                                "Yaw: Reference / ESKF / Loose",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_cmp_att.as_slice(),
                                        self.data.loose_cmp_att.as_slice(),
                                        self.data.orientation.as_slice(),
                                    ],
                                    &["yaw"],
                                ),
                                true,
                            ),
                        ],
                        self.max_points_per_trace,
                    );
                });
            }
            Page::Mount => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_analysis_page(
                        ui,
                        "Mount",
                        "Mount angle estimates and alignment diagnostics.",
                        vec![
                            plot_spec(
                                "Mount Roll: Reference / Align / ESKF / Loose",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_misalignment.as_slice(),
                                        self.data.loose_misalignment.as_slice(),
                                        self.data.align_cmp_att.as_slice(),
                                    ],
                                    &["mount roll", "Align roll", "Reference mount roll"],
                                ),
                                true,
                            ),
                            plot_spec(
                                "Mount Pitch: Reference / Align / ESKF / Loose",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_misalignment.as_slice(),
                                        self.data.loose_misalignment.as_slice(),
                                        self.data.align_cmp_att.as_slice(),
                                    ],
                                    &["mount pitch", "Align pitch", "Reference mount pitch"],
                                ),
                                true,
                            ),
                            plot_spec(
                                "Mount Yaw: Reference / Align / ESKF / Loose",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_misalignment.as_slice(),
                                        self.data.loose_misalignment.as_slice(),
                                        self.data.align_cmp_att.as_slice(),
                                    ],
                                    &["mount yaw", "Align yaw", "Reference mount yaw"],
                                ),
                                true,
                            ),
                            plot_spec(
                                "Mount Quaternion Error",
                                concat_trace_refs_matching(
                                    [
                                        self.data.eskf_misalignment.as_slice(),
                                        self.data.loose_misalignment.as_slice(),
                                        self.data.align_cmp_att.as_slice(),
                                    ],
                                    &["quaternion error"],
                                ),
                                true,
                            ),
                            plot_spec(
                                "Align Axis Error vs Reference Mount",
                                trace_refs(&self.data.align_axis_err),
                                true,
                            ),
                            plot_spec(
                                "Mount Reference vs Motion Heading",
                                trace_refs(&self.data.align_motion),
                                true,
                            ),
                            plot_spec("Align Covariance", trace_refs(&self.data.align_cov), true),
                        ],
                        self.max_points_per_trace,
                    );
                });
            }
            Page::Calibration => {
                let scale: Vec<&Trace> = self
                    .data
                    .loose_scale_gyro
                    .iter()
                    .chain(self.data.loose_scale_accel.iter())
                    .collect();
                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_analysis_page(
                        ui,
                        "Calibration",
                        "Biases, scale factors, and covariance diagonals.",
                        vec![
                            plot_spec(
                                "Gyro Bias: ESKF / Loose",
                                concat_trace_refs([
                                    self.data.eskf_bias_gyro.as_slice(),
                                    self.data.loose_bias_gyro.as_slice(),
                                ]),
                                true,
                            ),
                            plot_spec(
                                "Accel Bias: ESKF / Loose",
                                concat_trace_refs([
                                    self.data.eskf_bias_accel.as_slice(),
                                    self.data.loose_bias_accel.as_slice(),
                                ]),
                                true,
                            ),
                            plot_spec(
                                "Bias Covariance: ESKF / Loose",
                                concat_trace_refs([
                                    self.data.eskf_cov_bias.as_slice(),
                                    self.data.loose_cov_bias.as_slice(),
                                ]),
                                true,
                            ),
                            plot_spec(
                                "Non-bias Covariance: ESKF / Loose",
                                concat_trace_refs([
                                    self.data.eskf_cov_nonbias.as_slice(),
                                    self.data.loose_cov_nonbias.as_slice(),
                                ]),
                                true,
                            ),
                            plot_spec("Loose Scale Factors", scale, true),
                        ],
                        self.max_points_per_trace,
                    );
                });
            }
            Page::Sensors => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_analysis_page(
                        ui,
                        "Sensors",
                        "Raw, calibrated, and filter input sensor signals.",
                        vec![
                            plot_spec("Speed", trace_refs(&self.data.speed), true),
                            plot_spec(
                                "GNSS Signal Strength",
                                trace_refs(&self.data.sat_cn0),
                                false,
                            ),
                            plot_spec("Raw IMU Gyro", trace_refs(&self.data.imu_raw_gyro), true),
                            plot_spec("Raw IMU Accel", trace_refs(&self.data.imu_raw_accel), true),
                            plot_spec("Calibrated IMU Gyro", imu_cal_gyro, true),
                            plot_spec("Calibrated IMU Accel", imu_cal_accel, true),
                            plot_spec(
                                "ESKF Raw IMU Gyro Input",
                                trace_refs(&self.data.eskf_meas_gyro),
                                true,
                            ),
                            plot_spec(
                                "ESKF Raw IMU Accel Input",
                                trace_refs(&self.data.eskf_meas_accel),
                                true,
                            ),
                            plot_spec(
                                "Loose Vehicle-frame Gyro Input",
                                trace_refs(&self.data.loose_meas_gyro),
                                true,
                            ),
                            plot_spec(
                                "Loose Vehicle-frame Accel Input",
                                trace_refs(&self.data.loose_meas_accel),
                                true,
                            ),
                            plot_spec("Other Signals", trace_refs(&self.data.other), true),
                        ],
                        self.max_points_per_trace,
                    );
                });
            }
            Page::Diagnostics => {
                let bump_pitch: Vec<&Trace> = self
                    .data
                    .eskf_bump_pitch_speed
                    .iter()
                    .filter(|t| t.name.contains("pitch"))
                    .collect();
                let bump_speed: Vec<&Trace> = self
                    .data
                    .eskf_bump_pitch_speed
                    .iter()
                    .filter(|t| t.name.contains("speed"))
                    .collect();
                let bump_time: Vec<&Trace> = self
                    .data
                    .eskf_bump_diag
                    .iter()
                    .filter(|t| !t.name.contains("FFT dom"))
                    .collect();
                let bump_fft: Vec<&Trace> = self
                    .data
                    .eskf_bump_diag
                    .iter()
                    .filter(|t| t.name.contains("FFT dom"))
                    .collect();
                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_analysis_page(
                        ui,
                        "Diagnostics",
                        "Alignment windows, bump detector signals, and update contributions.",
                        vec![
                            plot_spec(
                                "Align Mount Angles vs Reference",
                                trace_refs(&self.data.align_cmp_att),
                                true,
                            ),
                            plot_spec(
                                "Align Window Diagnostics",
                                trace_refs(&self.data.align_res_vel),
                                true,
                            ),
                            plot_spec(
                                "Align Axis Error vs Reference",
                                trace_refs(&self.data.align_axis_err),
                                true,
                            ),
                            plot_spec(
                                "Mount Reference vs Motion Heading",
                                trace_refs(&self.data.align_motion),
                                true,
                            ),
                            plot_spec(
                                "Align Window Flags",
                                trace_refs(&self.data.align_flags),
                                true,
                            ),
                            plot_spec(
                                "Align Roll Update Contributions",
                                trace_refs(&self.data.align_roll_contrib),
                                true,
                            ),
                            plot_spec(
                                "Align Pitch Update Contributions",
                                trace_refs(&self.data.align_pitch_contrib),
                                true,
                            ),
                            plot_spec(
                                "Align Yaw Update Contributions",
                                trace_refs(&self.data.align_yaw_contrib),
                                true,
                            ),
                            plot_spec("Align Covariance", trace_refs(&self.data.align_cov), true),
                            plot_spec("ESKF Bump Pitch", bump_pitch, true),
                            plot_spec("ESKF Bump Speed", bump_speed, true),
                            plot_spec("ESKF Bump Time-domain Diagnostics", bump_time, true),
                            plot_spec("ESKF Bump FFT Diagnostics", bump_fft, true),
                            plot_spec(
                                "ESKF Stationary Diagnostics",
                                trace_refs(&self.data.eskf_stationary_diag),
                                true,
                            ),
                        ],
                        self.max_points_per_trace,
                    );
                });
            }
        }
    }
}

struct PlotSpec<'a> {
    title: &'static str,
    traces: Vec<&'a Trace>,
    show_legend: bool,
}

fn overview_tile_height(width: f32) -> f32 {
    #[cfg(not(target_arch = "wasm32"))]
    let _ = width;
    #[cfg(target_arch = "wasm32")]
    {
        responsive_plot_height(width)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        TIME_SERIES_PLOT_HEIGHT
    }
}

fn draw_map_tile(
    ui: &mut egui::Ui,
    title: &str,
    height: f32,
    add_map: impl FnOnce(&mut egui::Ui, egui::Vec2),
) {
    egui::Frame::group(ui.style())
        .fill(ui.visuals().window_fill)
        .inner_margin(egui::Margin::same(8))
        .corner_radius(egui::CornerRadius::same(8))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(title).strong());
            let size = egui::vec2(ui.available_width(), height);
            add_map(ui, size);
        });
}

#[cfg(target_arch = "wasm32")]
fn draw_web_run_button(
    ui: &mut egui::Ui,
    enabled: bool,
    busy: bool,
    progress: f32,
    text: &str,
) -> bool {
    let width = 128.0_f32.max(ui.spacing().interact_size.x * 3.2);
    let height = ui.spacing().interact_size.y;
    if busy {
        ui.add_sized(
            [width, height],
            egui::ProgressBar::new(progress.clamp(0.0, 1.0))
                .desired_width(width)
                .desired_height(height)
                .fill(ui.visuals().selection.bg_fill)
                .text(egui::RichText::new(format!(
                    "{text} {:>3.0}%",
                    100.0 * progress.clamp(0.0, 1.0)
                ))),
        );
        false
    } else {
        ui.add_enabled(
            enabled,
            egui::Button::new("Run").min_size(egui::vec2(width, height)),
        )
        .clicked()
    }
}

struct SyntheticTrajectoryTrace {
    name: String,
    points: Vec<[f64; 2]>,
    color: egui::Color32,
}

fn synthetic_trajectory_traces(
    data: &PlotData,
    visuals: &egui::Visuals,
    show_gnss: bool,
    show_eskf: bool,
    show_loose: bool,
) -> Vec<SyntheticTrajectoryTrace> {
    let mut traces = Vec::new();
    push_position_pair_trace(
        &mut traces,
        "Reference",
        &data.eskf_cmp_pos,
        "Synthetic truth posN [m]",
        "Synthetic truth posE [m]",
        SeriesColor::Reference.resolve(visuals),
    );
    if show_gnss
        && let Some(reference) = first_lonlat_trace_point(&data.eskf_map, "Synthetic truth path")
        && let Some(gnss) = data
            .eskf_map
            .iter()
            .find(|trace| trace.name.contains("Synthetic GNSS path"))
        && let Some(points) = lonlat_trace_to_local_en(gnss, reference)
    {
        traces.push(SyntheticTrajectoryTrace {
            name: "GNSS".to_string(),
            points,
            color: map_trace_color("GNSS", visuals),
        });
    }
    if show_eskf {
        push_position_pair_trace(
            &mut traces,
            "ESKF",
            &data.eskf_cmp_pos,
            "ESKF posN [m]",
            "ESKF posE [m]",
            map_trace_color("ESKF path (lon,lat)", visuals),
        );
    }
    if show_loose {
        push_position_pair_trace(
            &mut traces,
            "Loose",
            &data.loose_cmp_pos,
            "Loose posN [m]",
            "Loose posE [m]",
            map_trace_color("Loose path (lon,lat)", visuals),
        );
    }
    traces
}

fn push_position_pair_trace(
    out: &mut Vec<SyntheticTrajectoryTrace>,
    name: &str,
    traces: &[Trace],
    north_name: &str,
    east_name: &str,
    color: egui::Color32,
) {
    let (Some(north), Some(east)) = (
        traces.iter().find(|trace| trace.name == north_name),
        traces.iter().find(|trace| trace.name == east_name),
    ) else {
        return;
    };
    let points = position_pair_to_en_points(north, east);
    if points.len() >= 2 {
        out.push(SyntheticTrajectoryTrace {
            name: name.to_string(),
            points,
            color,
        });
    }
}

fn position_pair_to_en_points(north: &Trace, east: &Trace) -> Vec<[f64; 2]> {
    north
        .points
        .iter()
        .filter_map(|sample| {
            let t_s = sample[0];
            let north_m = sample[1];
            let east_m = sample_trace_at(east, t_s)?;
            (north_m.is_finite() && east_m.is_finite()).then_some([east_m, north_m])
        })
        .collect()
}

fn decimate_trajectory_points(points: &[[f64; 2]], max_points: usize) -> Vec<[f64; 2]> {
    if points.len() <= max_points || max_points < 2 {
        return points.to_vec();
    }
    let stride = points.len().div_ceil(max_points).max(1);
    let mut out: Vec<[f64; 2]> = points.iter().copied().step_by(stride).collect();
    if let Some(last) = points.last().copied()
        && out.last().copied() != Some(last)
    {
        out.push(last);
    }
    out
}

fn subtle_plot_grid_spacing() -> egui::emath::Rangef {
    egui::emath::Rangef::new(PLOT_GRID_MIN_SPACING_PX, PLOT_GRID_MAX_SPACING_PX)
}

fn subdued_plot_grid_marks(input: GridInput) -> Vec<GridMark> {
    log_grid_spacer(10)(input)
        .into_iter()
        .map(|mut mark| {
            mark.step_size *= PLOT_GRID_STRENGTH_SCALE;
            mark
        })
        .collect()
}

fn synthetic_cursor_point(data: &PlotData, t_s: f64) -> Option<[f64; 2]> {
    synthetic_cursor_point_from_traces(&data.eskf_cmp_pos, t_s, "ESKF posN [m]", "ESKF posE [m]")
        .or_else(|| {
            synthetic_cursor_point_from_traces(
                &data.eskf_cmp_pos,
                t_s,
                "Synthetic truth posN [m]",
                "Synthetic truth posE [m]",
            )
        })
        .or_else(|| {
            synthetic_cursor_point_from_traces(
                &data.loose_cmp_pos,
                t_s,
                "Loose posN [m]",
                "Loose posE [m]",
            )
        })
}

fn synthetic_cursor_point_from_traces(
    traces: &[Trace],
    t_s: f64,
    north_name: &str,
    east_name: &str,
) -> Option<[f64; 2]> {
    let north = traces.iter().find(|trace| trace.name == north_name)?;
    let east = traces.iter().find(|trace| trace.name == east_name)?;
    let north_m = sample_trace_at(north, t_s)?;
    let east_m = sample_trace_at(east, t_s)?;
    (north_m.is_finite() && east_m.is_finite()).then_some([east_m, north_m])
}

fn first_lonlat_trace_point(traces: &[Trace], name_contains: &str) -> Option<[f64; 2]> {
    traces
        .iter()
        .find(|trace| trace.name.contains(name_contains))
        .and_then(|trace| {
            trace
                .points
                .iter()
                .copied()
                .find(|point| point[0].is_finite() && point[1].is_finite())
        })
}

fn lonlat_trace_to_local_en(trace: &Trace, reference_lonlat: [f64; 2]) -> Option<Vec<[f64; 2]>> {
    let ref_lon_deg = reference_lonlat[0];
    let ref_lat_deg = reference_lonlat[1];
    if !ref_lon_deg.is_finite() || !ref_lat_deg.is_finite() {
        return None;
    }
    let ref_ecef = lla_to_ecef(ref_lat_deg, ref_lon_deg, 0.0);
    let points: Vec<[f64; 2]> = trace
        .points
        .iter()
        .filter_map(|point| {
            let lon_deg = point[0];
            let lat_deg = point[1];
            if !lon_deg.is_finite() || !lat_deg.is_finite() {
                return None;
            }
            let ecef = lla_to_ecef(lat_deg, lon_deg, 0.0);
            let ned = ecef_to_ned(ecef, ref_ecef, ref_lat_deg, ref_lon_deg);
            Some([ned[1], ned[0]])
        })
        .collect();
    (points.len() >= 2).then_some(points)
}

#[derive(Clone, Copy)]
enum AttitudeAxis {
    Roll,
    Pitch,
    Yaw,
}

struct AttitudeSeriesSample {
    label: &'static str,
    color: SeriesColor,
    angle_deg: f64,
}

#[derive(Clone, Copy)]
enum SeriesColor {
    Reference,
    Eskf,
    Loose,
    Align,
}

impl SeriesColor {
    fn resolve(self, visuals: &egui::Visuals) -> egui::Color32 {
        let dark = visuals.dark_mode;
        match self {
            Self::Reference if dark => egui::Color32::from_rgb(235, 238, 244),
            Self::Reference => egui::Color32::from_rgb(34, 43, 55),
            Self::Eskf if dark => egui::Color32::from_rgb(120, 170, 255),
            Self::Eskf => egui::Color32::from_rgb(35, 105, 200),
            Self::Loose if dark => egui::Color32::from_rgb(120, 255, 170),
            Self::Loose => egui::Color32::from_rgb(26, 138, 78),
            Self::Align if dark => egui::Color32::from_rgb(244, 190, 96),
            Self::Align => egui::Color32::from_rgb(168, 93, 22),
        }
    }
}

#[derive(Clone, Copy)]
enum OrthogonalViewKind {
    Vehicle,
    Mount,
}

fn draw_orthogonal_views_popup(
    ui: &mut egui::Ui,
    title: &str,
    traces: &[Trace],
    t_s: f64,
    kind: OrthogonalViewKind,
) {
    let Some(pointer_pos) = ui.input(|i| i.pointer.hover_pos()) else {
        return;
    };
    let screen_rect = ui.ctx().content_rect();
    let popup_size = egui::vec2(430.0, 218.0);
    let mut pos = pointer_pos + egui::vec2(14.0, 14.0);
    if pos.x + popup_size.x > screen_rect.right() - 8.0 {
        pos.x = pointer_pos.x - popup_size.x - 14.0;
    }
    if pos.y + popup_size.y > screen_rect.bottom() - 8.0 {
        pos.y = pointer_pos.y - popup_size.y - 14.0;
    }
    pos.x = pos.x.max(screen_rect.left() + 8.0);
    pos.y = pos.y.max(screen_rect.top() + 8.0);

    egui::Area::new(egui::Id::new(("orthogonal_views_popup", title)))
        .order(egui::Order::Tooltip)
        .fixed_pos(pos)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style())
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    ui.set_min_size(popup_size);
                    ui.label(egui::RichText::new(format!("{title}  t={t_s:.2}s")).strong());
                    let rect = ui
                        .allocate_exact_size(
                            egui::vec2(ui.available_width(), popup_size.y - 28.0),
                            egui::Sense::hover(),
                        )
                        .0;
                    let painter = ui.painter_at(rect);
                    let gap = 8.0;
                    let w = (rect.width() - 2.0 * gap).max(1.0) / 3.0;
                    for (idx, (axis, label)) in [
                        (AttitudeAxis::Roll, "Roll"),
                        (AttitudeAxis::Pitch, "Pitch"),
                        (AttitudeAxis::Yaw, "Yaw"),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let x0 = rect.left() + idx as f32 * (w + gap);
                        let view = egui::Rect::from_min_size(
                            egui::pos2(x0, rect.top()),
                            egui::vec2(w, rect.height()),
                        );
                        let samples = angle_samples_for_axis(traces, t_s, axis, kind);
                        draw_angle_axis_view(&painter, view, label, axis, &samples, kind);
                    }
                });
        });
}

fn angle_samples_for_axis(
    traces: &[Trace],
    t_s: f64,
    axis: AttitudeAxis,
    kind: OrthogonalViewKind,
) -> Vec<AttitudeSeriesSample> {
    let prefixes: &[(&str, &str, SeriesColor)] = match kind {
        OrthogonalViewKind::Vehicle => &[
            ("Reference", "Reference", SeriesColor::Reference),
            ("Synthetic truth", "Reference", SeriesColor::Reference),
            ("ESKF", "ESKF", SeriesColor::Eskf),
            ("Loose", "Loose", SeriesColor::Loose),
        ],
        OrthogonalViewKind::Mount => &[
            ("Reference mount", "Reference", SeriesColor::Reference),
            ("Synthetic truth mount", "Reference", SeriesColor::Reference),
            ("Align", "Align", SeriesColor::Align),
            ("ESKF mount", "ESKF", SeriesColor::Eskf),
            ("Loose residual mount", "Loose", SeriesColor::Loose),
        ],
    };
    prefixes
        .iter()
        .copied()
        .into_iter()
        .filter_map(|(prefix, label, color)| {
            find_angle_trace(traces, prefix, axis, kind).and_then(|trace| {
                sample_trace_at(trace, t_s).map(|angle_deg| AttitudeSeriesSample {
                    label,
                    color,
                    angle_deg,
                })
            })
        })
        .fold(Vec::<AttitudeSeriesSample>::new(), |mut out, sample| {
            if !out.iter().any(|existing| existing.label == sample.label) {
                out.push(sample);
            }
            out
        })
}

fn find_angle_trace<'a>(
    traces: &'a [Trace],
    prefix: &str,
    axis: AttitudeAxis,
    kind: OrthogonalViewKind,
) -> Option<&'a Trace> {
    let axis_token = match axis {
        AttitudeAxis::Roll => "roll",
        AttitudeAxis::Pitch => "pitch",
        AttitudeAxis::Yaw => "yaw",
    };
    traces.iter().find(|trace| {
        let name = trace.name.as_str();
        name.starts_with(prefix)
            && name.contains(axis_token)
            && match kind {
                OrthogonalViewKind::Vehicle => {
                    !name.contains("mount") && !name.contains("residual")
                }
                OrthogonalViewKind::Mount => name.contains("mount") || name.starts_with("Align"),
            }
    })
}

fn sample_trace_at(trace: &Trace, t_s: f64) -> Option<f64> {
    if !t_s.is_finite() || trace.points.is_empty() {
        return None;
    }
    let points = &trace.points;
    let idx = points.partition_point(|p| p[0] < t_s);
    if idx == 0 {
        return points.first().map(|p| p[1]);
    }
    if idx >= points.len() {
        return points.last().map(|p| p[1]);
    }
    let a = points[idx - 1];
    let b = points[idx];
    let dt = b[0] - a[0];
    if dt.abs() <= f64::EPSILON {
        return Some(b[1]);
    }
    let alpha = ((t_s - a[0]) / dt).clamp(0.0, 1.0);
    Some(a[1] + alpha * (b[1] - a[1]))
}

fn draw_angle_axis_view(
    painter: &egui::Painter,
    rect: egui::Rect,
    label: &str,
    axis: AttitudeAxis,
    samples: &[AttitudeSeriesSample],
    kind: OrthogonalViewKind,
) {
    let visuals = painter.ctx().style().visuals.clone();
    painter.rect_stroke(
        rect,
        egui::CornerRadius::same(5),
        egui::Stroke::new(1.0, orthogonal_panel_stroke(&visuals)),
        egui::StrokeKind::Inside,
    );
    painter.text(
        rect.center_top() + egui::vec2(0.0, 8.0),
        egui::Align2::CENTER_TOP,
        label,
        egui::FontId::proportional(12.0),
        visuals.text_color(),
    );
    if samples.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No sample",
            egui::FontId::proportional(11.0),
            visuals.weak_text_color(),
        );
        return;
    }

    let figure_rect = rect.shrink2(egui::vec2(8.0, 24.0));
    let scale = figure_rect.width().min(figure_rect.height()).max(1.0);
    let center = figure_rect.center() + egui::vec2(0.0, -4.0);
    draw_zero_angle_axis(painter, center, scale, axis, &visuals);
    for sample in samples {
        draw_rotated_outline(
            painter,
            center,
            scale,
            axis,
            sample.angle_deg,
            sample.color.resolve(&visuals),
            matches!(kind, OrthogonalViewKind::Vehicle),
        );
    }

    let mut y = rect.bottom() - 14.0 * samples.len() as f32 - 3.0;
    for sample in samples {
        painter.text(
            egui::pos2(rect.left() + 7.0, y),
            egui::Align2::LEFT_TOP,
            format!("{} {:+.1} deg", sample.label, sample.angle_deg),
            egui::FontId::proportional(10.5),
            sample.color.resolve(&visuals),
        );
        y += 14.0;
    }
}

fn orthogonal_panel_stroke(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_gray(58)
    } else {
        egui::Color32::from_rgb(185, 176, 164)
    }
}

fn draw_zero_angle_axis(
    painter: &egui::Painter,
    center: egui::Pos2,
    scale: f32,
    axis: AttitudeAxis,
    visuals: &egui::Visuals,
) {
    let half = 0.36 * scale;
    let dash = 4.0;
    let gap = 4.0;
    let color = zero_angle_axis_color(visuals);
    match axis {
        AttitudeAxis::Roll | AttitudeAxis::Pitch => {
            let mut x = center.x - half;
            while x < center.x + half {
                let x1 = (x + dash).min(center.x + half);
                painter.line_segment(
                    [egui::pos2(x, center.y), egui::pos2(x1, center.y)],
                    egui::Stroke::new(1.0, color),
                );
                x += dash + gap;
            }
        }
        AttitudeAxis::Yaw => {
            let mut y = center.y - half;
            while y < center.y + half {
                let y1 = (y + dash).min(center.y + half);
                painter.line_segment(
                    [egui::pos2(center.x, y), egui::pos2(center.x, y1)],
                    egui::Stroke::new(1.0, color),
                );
                y += dash + gap;
            }
        }
    }
}

fn zero_angle_axis_color(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_gray(95).gamma_multiply(0.58)
    } else {
        egui::Color32::from_rgb(105, 113, 126).gamma_multiply(0.58)
    }
}

fn draw_rotated_outline(
    painter: &egui::Painter,
    center: egui::Pos2,
    scale: f32,
    axis: AttitudeAxis,
    angle_deg: f64,
    color: egui::Color32,
    show_wheels: bool,
) {
    let (w, h) = match axis {
        AttitudeAxis::Roll => (0.58 * scale, 0.30 * scale),
        AttitudeAxis::Pitch => (0.68 * scale, 0.24 * scale),
        AttitudeAxis::Yaw => (0.62 * scale, 0.28 * scale),
    };
    let angle = orthogonal_view_screen_angle(axis, angle_deg);
    let corners = [
        egui::vec2(-0.5 * w, -0.5 * h),
        egui::vec2(0.5 * w, -0.5 * h),
        egui::vec2(0.5 * w, 0.5 * h),
        egui::vec2(-0.5 * w, 0.5 * h),
    ];
    let mut points: Vec<egui::Pos2> = corners
        .into_iter()
        .map(|p| rotate_point(center, p, angle))
        .collect();
    points.push(points[0]);
    painter.add(egui::Shape::line(points, egui::Stroke::new(1.8, color)));

    let nose = match axis {
        AttitudeAxis::Roll => [egui::vec2(0.0, -0.5 * h), egui::vec2(0.0, -0.82 * h)],
        _ => [egui::vec2(0.5 * w, 0.0), egui::vec2(0.78 * w, 0.0)],
    };
    painter.line_segment(
        [
            rotate_point(center, nose[0], angle),
            rotate_point(center, nose[1], angle),
        ],
        egui::Stroke::new(1.8, color),
    );
    if show_wheels {
        draw_vehicle_wheels(painter, center, w, h, angle, color, axis);
    }
}

fn orthogonal_view_screen_angle(axis: AttitudeAxis, angle_deg: f64) -> f32 {
    let angle = angle_deg as f32;
    match axis {
        // FRD roll: positive roll moves the vehicle right side down. With
        // screen +x as vehicle right and screen +y as vehicle down, this is a
        // clockwise screen rotation.
        AttitudeAxis::Roll => angle.to_radians(),
        // FRD/NED pitch traces use positive pitch as nose-up. The side view
        // has screen +x forward and screen +y down, so nose-up is
        // counter-clockwise on screen.
        AttitudeAxis::Pitch => (-angle).to_radians(),
        // Yaw traces are headings: 0 deg is north/up, positive yaw turns
        // clockwise toward east/right.
        AttitudeAxis::Yaw => (-90.0 + angle).to_radians(),
    }
}

fn draw_vehicle_wheels(
    painter: &egui::Painter,
    center: egui::Pos2,
    w: f32,
    h: f32,
    angle: f32,
    color: egui::Color32,
    axis: AttitudeAxis,
) {
    if matches!(axis, AttitudeAxis::Roll | AttitudeAxis::Yaw) {
        draw_rect_vehicle_wheels(painter, center, w, h, angle, color, axis);
        return;
    }

    let wheel_radius = (w.min(h) * 0.20).clamp(4.5, 9.0);
    let wheel_points: &[egui::Vec2] = match axis {
        AttitudeAxis::Roll => &[
            egui::vec2(-0.36 * w, 0.58 * h),
            egui::vec2(0.36 * w, 0.58 * h),
        ],
        AttitudeAxis::Pitch => &[
            egui::vec2(-0.34 * w, 0.56 * h),
            egui::vec2(0.34 * w, 0.56 * h),
        ],
        AttitudeAxis::Yaw => &[],
    };
    for wheel in wheel_points {
        let pos = rotate_point(center, *wheel, angle);
        painter.circle_filled(pos, wheel_radius, egui::Color32::from_black_alpha(180));
        painter.circle_stroke(pos, wheel_radius, egui::Stroke::new(1.0, color));
    }
}

fn draw_rect_vehicle_wheels(
    painter: &egui::Painter,
    center: egui::Pos2,
    w: f32,
    h: f32,
    angle: f32,
    color: egui::Color32,
    axis: AttitudeAxis,
) {
    let wheel_size = match axis {
        AttitudeAxis::Roll => egui::vec2(0.18 * w, 0.28 * h),
        AttitudeAxis::Yaw => egui::vec2(0.18 * w, 0.18 * h),
        AttitudeAxis::Pitch => egui::Vec2::ZERO,
    };
    let wheel_centers: &[egui::Vec2] = match axis {
        AttitudeAxis::Roll => &[
            egui::vec2(-0.36 * w, 0.62 * h),
            egui::vec2(0.36 * w, 0.62 * h),
        ],
        AttitudeAxis::Yaw => &[
            egui::vec2(-0.34 * w, -0.56 * h),
            egui::vec2(0.34 * w, -0.56 * h),
            egui::vec2(-0.34 * w, 0.56 * h),
            egui::vec2(0.34 * w, 0.56 * h),
        ],
        AttitudeAxis::Pitch => &[],
    };
    for wheel_center in wheel_centers {
        let half = 0.5 * wheel_size;
        let corners = [
            *wheel_center + egui::vec2(-half.x, -half.y),
            *wheel_center + egui::vec2(half.x, -half.y),
            *wheel_center + egui::vec2(half.x, half.y),
            *wheel_center + egui::vec2(-half.x, half.y),
        ];
        let points: Vec<egui::Pos2> = corners
            .into_iter()
            .map(|p| rotate_point(center, p, angle))
            .collect();
        painter.add(egui::Shape::convex_polygon(
            points,
            egui::Color32::from_black_alpha(185),
            egui::Stroke::new(1.0, color),
        ));
    }
}

fn rotate_point(center: egui::Pos2, point: egui::Vec2, angle: f32) -> egui::Pos2 {
    let (sin, cos) = angle.sin_cos();
    center + egui::vec2(point.x * cos - point.y * sin, point.x * sin + point.y * cos)
}

fn plot_spec<'a>(title: &'static str, traces: Vec<&'a Trace>, show_legend: bool) -> PlotSpec<'a> {
    PlotSpec {
        title,
        traces,
        show_legend,
    }
}

fn trace_refs(traces: &[Trace]) -> Vec<&Trace> {
    traces.iter().filter(|t| !t.points.is_empty()).collect()
}

fn trace_time_range<'a>(traces: impl IntoIterator<Item = &'a Trace>) -> Option<(f64, f64)> {
    traces
        .into_iter()
        .filter_map(|trace| {
            let start = trace
                .points
                .iter()
                .find_map(|p| p[0].is_finite().then_some(p[0]))?;
            let end = trace
                .points
                .iter()
                .rev()
                .find_map(|p| p[0].is_finite().then_some(p[0]))?;
            Some((start.min(end), start.max(end)))
        })
        .fold(None, |range, (start, end)| match range {
            Some((min_t, max_t)) => Some((f64::min(min_t, start), f64::max(max_t, end))),
            None => Some((start, end)),
        })
}

fn time_in_range(t_s: f64, range: Option<(f64, f64)>) -> bool {
    let Some((min_t, max_t)) = range else {
        return false;
    };
    t_s.is_finite() && t_s >= min_t && t_s <= max_t
}

fn concat_trace_refs<const N: usize>(groups: [&[Trace]; N]) -> Vec<&Trace> {
    let mut out = Vec::new();
    for group in groups {
        for trace in group {
            if trace.points.is_empty() || out.iter().any(|t: &&Trace| t.name == trace.name) {
                continue;
            }
            out.push(trace);
        }
    }
    out
}

fn concat_trace_refs_matching<'a, const N: usize>(
    groups: [&'a [Trace]; N],
    tokens: &[&str],
) -> Vec<&'a Trace> {
    let mut out = Vec::new();
    for group in groups {
        for trace in group {
            if trace.points.is_empty()
                || !tokens.iter().any(|token| trace.name.contains(token))
                || out.iter().any(|t: &&Trace| t.name == trace.name)
            {
                continue;
            }
            out.push(trace);
        }
    }
    out
}

fn page_header(ui: &mut egui::Ui, title: &str, _subtitle: &str) {
    ui.heading(title);
    ui.add_space(6.0);
}

fn draw_analysis_page(
    ui: &mut egui::Ui,
    title: &str,
    subtitle: &str,
    plots: Vec<PlotSpec<'_>>,
    max_points_per_trace: usize,
) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            page_header(ui, title, subtitle);
            let plots: Vec<PlotSpec<'_>> = plots
                .into_iter()
                .filter(|plot| plot.traces.iter().any(|trace| !trace.points.is_empty()))
                .collect();
            if plots.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("No data").weak());
                });
                return;
            }
            let min_plot_width = 560.0;
            let column_count =
                ((ui.available_width() / min_plot_width).floor() as usize).clamp(1, 2);
            ui.columns(column_count, |cols| {
                for (idx, spec) in plots.into_iter().enumerate() {
                    let column = idx % column_count;
                    draw_plot(
                        &mut cols[column],
                        spec.title,
                        spec.traces.iter().copied(),
                        spec.show_legend,
                        max_points_per_trace,
                    );
                }
            });
        });
}

#[cfg(target_arch = "wasm32")]
fn responsive_plot_height(width: f32) -> f32 {
    if width < 420.0 {
        160.0
    } else if width < 560.0 {
        180.0
    } else {
        TIME_SERIES_PLOT_HEIGHT
    }
}

fn draw_plot<'a, I>(
    ui: &mut egui::Ui,
    title: &str,
    traces: I,
    show_legend: bool,
    max_points_per_trace: usize,
) where
    I: IntoIterator<Item = &'a Trace>,
{
    let _ = draw_plot_with_cursor_time(ui, title, traces, show_legend, max_points_per_trace, None);
}

fn draw_plot_with_cursor_time<'a, I>(
    ui: &mut egui::Ui,
    title: &str,
    traces: I,
    show_legend: bool,
    max_points_per_trace: usize,
    cursor_t_s: Option<f64>,
) -> Option<f64>
where
    I: IntoIterator<Item = &'a Trace>,
{
    fn visible_decimated(
        points: &[[f64; 2]],
        xmin: f64,
        xmax: f64,
        max_points: usize,
    ) -> Vec<[f64; 2]> {
        fn decimate_finite_slice(slice: &[[f64; 2]], max_points: usize) -> Vec<[f64; 2]> {
            if slice.len() <= max_points || max_points == 0 {
                return slice.to_vec();
            }

            let buckets = (max_points / 2).max(1);
            let x0 = slice.first().map(|p| p[0]).unwrap_or(0.0);
            let x1 = slice.last().map(|p| p[0]).unwrap_or(0.0);
            let span = (x1 - x0).abs();
            if span <= f64::EPSILON {
                let step = ((slice.len() as f64) / (max_points as f64)).ceil() as usize;
                return slice.iter().step_by(step.max(1)).copied().collect();
            }

            #[cfg(target_arch = "wasm32")]
            let scan_step = {
                let scan_budget = max_points.saturating_mul(WEB_DECIMATION_SCAN_MULTIPLIER);
                (slice.len() / scan_budget.max(1)).max(1)
            };
            #[cfg(not(target_arch = "wasm32"))]
            let scan_step = 1usize;

            let mut min_b: Vec<Option<(usize, f64)>> = vec![None; buckets];
            let mut max_b: Vec<Option<(usize, f64)>> = vec![None; buckets];
            let mut visit = |i: usize, p: &[f64; 2]| {
                if !p[0].is_finite() || !p[1].is_finite() {
                    return;
                }
                let mut b = (((p[0] - x0) / span) * buckets as f64).floor() as usize;
                if b >= buckets {
                    b = buckets - 1;
                }
                match min_b[b] {
                    Some((_, y)) if p[1] >= y => {}
                    _ => min_b[b] = Some((i, p[1])),
                }
                match max_b[b] {
                    Some((_, y)) if p[1] <= y => {}
                    _ => max_b[b] = Some((i, p[1])),
                }
            };
            for (i, p) in slice.iter().enumerate().step_by(scan_step) {
                visit(i, p);
            }
            if scan_step > 1
                && let Some((i, p)) = slice.len().checked_sub(1).map(|i| (i, &slice[i]))
            {
                visit(i, p);
            }

            let mut out = Vec::with_capacity(max_points);
            let mut last_idx: Option<usize> = None;
            for b in 0..buckets {
                let a = min_b[b].map(|(i, _)| i);
                let c = max_b[b].map(|(i, _)| i);
                match (a, c) {
                    (Some(i0), Some(i1)) if i0 == i1 => {
                        if last_idx != Some(i0) {
                            out.push(slice[i0]);
                            last_idx = Some(i0);
                        }
                    }
                    (Some(i0), Some(i1)) => {
                        let (first, second) = if i0 < i1 { (i0, i1) } else { (i1, i0) };
                        if last_idx != Some(first) {
                            out.push(slice[first]);
                            last_idx = Some(first);
                        }
                        if out.len() < max_points && last_idx != Some(second) {
                            out.push(slice[second]);
                            last_idx = Some(second);
                        }
                    }
                    (Some(i0), None) | (None, Some(i0)) => {
                        if last_idx != Some(i0) {
                            out.push(slice[i0]);
                            last_idx = Some(i0);
                        }
                    }
                    (None, None) => {}
                }
                if out.len() >= max_points {
                    break;
                }
            }

            if out.is_empty() {
                let step = ((slice.len() as f64) / (max_points as f64)).ceil() as usize;
                return slice
                    .iter()
                    .step_by(step.max(1))
                    .copied()
                    .filter(|p| p[0].is_finite() && p[1].is_finite())
                    .collect();
            }
            out
        }

        if points.is_empty() {
            return Vec::new();
        }
        let lo = points.partition_point(|p| p[0] < xmin);
        let hi = points.partition_point(|p| p[0] <= xmax);
        let start = lo.saturating_sub(1);
        let end = if hi < points.len() {
            hi + 1
        } else {
            points.len()
        };
        let slice = &points[start..end];
        #[cfg(target_arch = "wasm32")]
        if slice.len()
            > max_points
                .saturating_mul(WEB_DECIMATION_SCAN_MULTIPLIER)
                .max(max_points + 1)
        {
            return decimate_finite_slice(slice, max_points);
        }

        let mut out = Vec::new();
        let mut seg_start = 0usize;
        while seg_start < slice.len() {
            while seg_start < slice.len()
                && (!slice[seg_start][0].is_finite() || !slice[seg_start][1].is_finite())
            {
                out.push(slice[seg_start]);
                seg_start += 1;
            }
            if seg_start >= slice.len() {
                break;
            }
            let mut seg_end = seg_start;
            while seg_end < slice.len()
                && slice[seg_end][0].is_finite()
                && slice[seg_end][1].is_finite()
            {
                seg_end += 1;
            }
            out.extend(decimate_finite_slice(
                &slice[seg_start..seg_end],
                max_points,
            ));
            if seg_end < slice.len() {
                out.push(slice[seg_end]);
            }
            seg_start = seg_end + 1;
        }
        out
    }

    #[cfg(target_arch = "wasm32")]
    {
        let plot_height = responsive_plot_height(ui.available_width());
        let desired_size = egui::vec2(
            ui.available_width(),
            plot_height + TIME_SERIES_PLOT_FRAME_PADDING,
        );
        let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
        if !ui.is_rect_visible(rect) {
            return None;
        }
        return ui
            .scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                draw_plot_body(
                    ui,
                    title,
                    traces,
                    show_legend,
                    max_points_per_trace,
                    plot_height,
                    cursor_t_s,
                )
            })
            .inner;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        return ui
            .vertical(|ui| {
                draw_plot_body(
                    ui,
                    title,
                    traces,
                    show_legend,
                    max_points_per_trace,
                    TIME_SERIES_PLOT_HEIGHT,
                    cursor_t_s,
                )
            })
            .inner;
    }

    fn draw_plot_body<'a, I>(
        ui: &mut egui::Ui,
        title: &str,
        traces: I,
        show_legend: bool,
        max_points_per_trace: usize,
        plot_height: f32,
        cursor_t_s: Option<f64>,
    ) -> Option<f64>
    where
        I: IntoIterator<Item = &'a Trace>,
    {
        let traces: Vec<&Trace> = traces.into_iter().collect();
        egui::Frame::group(ui.style())
            .fill(ui.visuals().window_fill)
            .inner_margin(egui::Margin::same(8))
            .corner_radius(egui::CornerRadius::same(8))
            .show(ui, |ui| {
                ui.label(egui::RichText::new(title).strong());
                if traces.is_empty() {
                    ui.allocate_ui(egui::vec2(ui.available_width(), plot_height), |ui| {
                        ui.centered_and_justified(|ui| {
                            ui.label(egui::RichText::new("No data").weak());
                        });
                    });
                    return None;
                }
                let data_time_range = trace_time_range(traces.iter().copied());

                let mut plot = Plot::new(title)
                    .height(plot_height)
                    .grid_spacing(subtle_plot_grid_spacing())
                    .x_grid_spacer(subdued_plot_grid_marks)
                    .y_grid_spacer(subdued_plot_grid_marks)
                    .link_axis("shared_x", egui::Vec2b::new(true, false))
                    .x_axis_formatter(|mark, _range| format!("{:.1}", mark.value))
                    .allow_drag(true)
                    .allow_zoom(true)
                    .allow_scroll(true)
                    .allow_boxed_zoom(true)
                    .allow_axis_zoom_drag(true);
                if show_legend {
                    plot = plot.legend(Legend::default());
                }
                let shared_cursor_color = shared_cursor_color(ui.visuals());
                plot.show(ui, |plot_ui| {
                    let bounds = plot_ui.plot_bounds();
                    let xmin = bounds.min()[0];
                    let xmax = bounds.max()[0];
                    for t in &traces {
                        if t.points.is_empty() {
                            continue;
                        }
                        let reduced =
                            visible_decimated(&t.points, xmin, xmax, max_points_per_trace);
                        if reduced.is_empty() {
                            continue;
                        }
                        let points: PlotPoints<'_> = reduced.into();
                        if t.name == "yaw initialized" {
                            plot_ui.points(Points::new(t.name.clone(), points).radius(4.0));
                        } else {
                            plot_ui.line(Line::new(t.name.clone(), points));
                        }
                    }
                    let hover_t_s = plot_ui
                        .response()
                        .hovered()
                        .then(|| plot_ui.pointer_coordinate().map(|p| p.x))
                        .flatten()
                        .filter(|t| time_in_range(*t, data_time_range));
                    if let Some(cursor_t_s) = hover_t_s
                        .or(cursor_t_s)
                        .filter(|t| time_in_range(*t, data_time_range))
                    {
                        plot_ui.vline(
                            VLine::new("Shared cursor", cursor_t_s)
                                .name("")
                                .allow_hover(false)
                                .color(shared_cursor_color)
                                .style(LineStyle::Dotted { spacing: 5.0 }),
                        );
                    }
                    hover_t_s
                })
                .inner
            })
            .inner
    }
}

fn shared_cursor_color(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_gray(210).gamma_multiply(0.55)
    } else {
        egui::Color32::from_rgb(73, 84, 100).gamma_multiply(0.72)
    }
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
