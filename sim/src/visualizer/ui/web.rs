#![cfg(target_arch = "wasm32")]
//! Browser-only dataset loading, replay worker orchestration, and web input helpers.

use std::{cell::RefCell, io::Read, rc::Rc};

use eframe::egui;
use flate2::read::GzDecoder;
use js_sys::{Array, Date, Function, Object, Reflect, Uint8Array};
use serde::Deserialize;
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{ErrorEvent, MessageEvent, Worker, WorkerOptions, WorkerType};

use crate::visualizer::model::{PlotData, VisualizerMountMode};
use crate::visualizer::pipeline::{FilterCompareConfig, GnssOutageConfig};
use crate::visualizer::stats::map_center_from_traces;
use crate::visualizer::theme::UiTheme;

use super::App;
use super::state::DataOrigin;

pub(super) const WEB_MIN_POINTS_PER_TRACE: usize = 80;
pub(super) const WEB_MAX_POINTS_PER_TRACE: usize = 2500;
const WEB_DATASET_MANIFEST_URL: &str = "datasets/manifest.json";

#[derive(Clone)]
pub(super) struct NamedText {
    pub(super) name: String,
    pub(super) text: String,
}

#[derive(Clone, Deserialize)]
pub(super) struct WebDatasetEntry {
    #[serde(default)]
    pub(super) id: Option<String>,
    #[serde(default)]
    pub(super) label: Option<String>,
    #[serde(default)]
    pub(super) description: Option<String>,
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
    #[serde(default, alias = "reference_position_csv")]
    reference_position: Option<String>,
    #[serde(default, alias = "reference_position_csv_gz")]
    reference_position_gz: Option<String>,
}

#[derive(Deserialize)]
struct WebDatasetManifest {
    #[serde(default)]
    datasets: Vec<WebDatasetEntry>,
}

pub(super) struct WebDatasetState {
    pub(super) manifest_url: String,
    pub(super) datasets: Vec<WebDatasetEntry>,
    pub(super) selected: usize,
    pub(super) auto_load_id: Option<String>,
    pub(super) auto_load_attempted: bool,
    pub(super) loading_manifest: bool,
    pub(super) loading_dataset: bool,
    pub(super) loading_replay: bool,
    pub(super) replay_job_id: u64,
    pub(super) replay_cfg: FilterCompareConfig,
    pub(super) pending: Rc<RefCell<Option<WebDatasetTaskResult>>>,
    pub(super) pending_replay: Rc<RefCell<Option<WebReplayTaskResult>>>,
    pub(super) replay_worker: Option<Worker>,
    pub(super) replay_onmessage: Option<Closure<dyn FnMut(MessageEvent)>>,
    pub(super) replay_onerror: Option<Closure<dyn FnMut(ErrorEvent)>>,
}

pub(super) enum WebDatasetTaskResult {
    Manifest(std::result::Result<Vec<WebDatasetEntry>, String>),
    Dataset(std::result::Result<WebDatasetFiles, String>),
}

pub(super) enum WebReplayTaskResult {
    Progress {
        job_id: u64,
        progress: f32,
        current_t_s: f64,
        final_t_s: f64,
    },
    Complete {
        job_id: u64,
        result: std::result::Result<WebReplayWorkerOutput, String>,
    },
}

pub(super) struct WebReplayWorkerOutput {
    pub(super) source: String,
    pub(super) label: String,
    pub(super) imu_name: String,
    pub(super) gnss_name: String,
    pub(super) json: String,
}

pub(super) enum WebReplayWorkerJob {
    Csv {
        label: String,
        imu_name: String,
        gnss_name: String,
        imu_csv: String,
        gnss_csv: String,
        reference_attitude_csv: Option<String>,
        reference_mount_csv: Option<String>,
        reference_position_csv: Option<String>,
    },
    Synthetic {
        label: String,
        motion_label: String,
        motion_text: String,
        noise: WebSyntheticNoise,
        early_vel_bias_ned_mps: [f64; 3],
        early_fault_window_s: Option<[f64; 2]>,
    },
}

pub(super) struct WebDatasetFiles {
    pub(super) label: String,
    pub(super) imu: NamedText,
    pub(super) gnss: NamedText,
    pub(super) reference_attitude: Option<NamedText>,
    pub(super) reference_mount: Option<NamedText>,
    pub(super) reference_position: Option<NamedText>,
}

impl WebDatasetState {
    pub(super) fn new() -> Self {
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
            replay_cfg: FilterCompareConfig::default(),
            pending: Rc::new(RefCell::new(None)),
            pending_replay: Rc::new(RefCell::new(None)),
            replay_worker: None,
            replay_onmessage: None,
            replay_onerror: None,
        }
    }
}

impl WebReplayWorkerJob {
    pub(super) fn label(&self) -> &str {
        match self {
            Self::Csv { label, .. } | Self::Synthetic { label, .. } => label,
        }
    }
}

impl WebDatasetEntry {
    pub(super) fn display_label(&self) -> String {
        self.label
            .as_deref()
            .or(self.id.as_deref())
            .unwrap_or("unnamed dataset")
            .to_string()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum WebInputMode {
    Synthetic,
    RealData,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum WebRealDataSource {
    DroppedCsv,
    ManifestDataset,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum WebSyntheticScenario {
    CityBlocks,
    FigureEight,
    FigureEightEarlyVelocityFault,
    FigureEightRollExcitation,
    StraightAccelBrake,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum WebSyntheticNoise {
    Truth,
    Low,
    Mid,
    High,
}

#[derive(Default)]
pub(super) struct WebPerf {
    pub(super) enabled: bool,
    pub(super) frame_count: u32,
    pub(super) start_time_s: f64,
    pub(super) last_time_s: f64,
    pub(super) fps_ema: f64,
}

pub(super) fn web_query_value_raw(key: &str) -> Option<String> {
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

fn web_query_value(key: &str) -> Option<String> {
    web_query_value_raw(key).map(|value| value.to_ascii_lowercase())
}

pub(super) fn web_query_flag(key: &str) -> bool {
    matches!(
        web_query_value(key).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub(super) fn web_now_s() -> f64 {
    Date::now() * 0.001
}

pub(super) fn estimate_web_replay_duration_s(imu_csv: &str, gnss_csv: &str) -> f64 {
    let span_s = csv_time_span_s(imu_csv).max(csv_time_span_s(gnss_csv));
    if span_s.is_finite() && span_s > 0.0 {
        (span_s / 140.0).clamp(3.0, 18.0)
    } else {
        6.0
    }
}

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

pub(super) fn web_replay_worker_request(
    job_id: u64,
    job: &WebReplayWorkerJob,
    misalignment: VisualizerMountMode,
    filter_cfg: FilterCompareConfig,
    gnss_outages: GnssOutageConfig,
) -> Object {
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
            reference_position_csv,
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
            if let Some(reference) = reference_position_csv {
                let _ = Reflect::set(
                    &source,
                    &JsValue::from_str("referencePositionCsv"),
                    &JsValue::from_str(reference),
                );
            }
        }
        WebReplayWorkerJob::Synthetic {
            label,
            motion_label,
            motion_text,
            noise,
            early_vel_bias_ned_mps,
            early_fault_window_s,
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
                &JsValue::from_str(noise.cli_value()),
            );
            let _ = Reflect::set(&source, &JsValue::from_str("seed"), &JsValue::from_f64(1.0));
            let mount_rpy_deg = js_number_array(&[5.0, -5.0, 5.0]);
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
            let early_vel_bias_ned_mps = js_number_array(early_vel_bias_ned_mps);
            let _ = Reflect::set(
                &source,
                &JsValue::from_str("earlyVelBiasNedMps"),
                early_vel_bias_ned_mps.as_ref(),
            );
            if let Some(window) = early_fault_window_s {
                let early_fault_window_s = js_number_array(window);
                let _ = Reflect::set(
                    &source,
                    &JsValue::from_str("earlyFaultWindowS"),
                    early_fault_window_s.as_ref(),
                );
            }
        }
    }
    let _ = Reflect::set(&request, &JsValue::from_str("source"), source.as_ref());
    let _ = Reflect::set(
        &request,
        &JsValue::from_str("misalignment"),
        &JsValue::from_str(misalignment.cli_value()),
    );
    let filter_cfg_value = serde_wasm_bindgen::to_value(&filter_cfg).unwrap_or(JsValue::NULL);
    let _ = Reflect::set(&request, &JsValue::from_str("filterCfg"), &filter_cfg_value);
    let _ = Reflect::set(
        &request,
        &JsValue::from_str("gnssOutages"),
        &serde_wasm_bindgen::to_value(&gnss_outages).unwrap_or(JsValue::NULL),
    );
    request
}

fn js_number_array(values: &[f64]) -> Array {
    let array = Array::new();
    for value in values {
        array.push(&JsValue::from_f64(*value));
    }
    array
}

pub(super) fn web_query_synthetic_scenario() -> Option<WebSyntheticScenario> {
    match web_query_value("scenario").as_deref() {
        Some("city" | "city_blocks" | "city-blocks") => Some(WebSyntheticScenario::CityBlocks),
        Some("figure8" | "figure_eight" | "figure-eight") => {
            Some(WebSyntheticScenario::FigureEight)
        }
        Some("figure8_fault" | "figure-eight-fault" | "fig8_fault" | "bad_basin") => {
            Some(WebSyntheticScenario::FigureEightEarlyVelocityFault)
        }
        Some(
            "figure8_roll"
            | "figure-eight-roll"
            | "figure8_roll_excitation"
            | "figure-eight-roll-excitation",
        ) => Some(WebSyntheticScenario::FigureEightRollExcitation),
        Some("straight" | "straight_accel_brake" | "straight-accel-brake") => {
            Some(WebSyntheticScenario::StraightAccelBrake)
        }
        _ => None,
    }
}

pub(super) fn web_query_synthetic_noise() -> Option<WebSyntheticNoise> {
    match web_query_value("noise").as_deref() {
        Some("truth" | "none" | "zero") => Some(WebSyntheticNoise::Truth),
        Some("low") => Some(WebSyntheticNoise::Low),
        Some("mid" | "medium") => Some(WebSyntheticNoise::Mid),
        Some("high") => Some(WebSyntheticNoise::High),
        _ => None,
    }
}

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

async fn web_fetch_text(url: &str) -> std::result::Result<String, String> {
    String::from_utf8(web_fetch_bytes(url).await?)
        .map_err(|err| format!("{url}: response was not UTF-8: {err}"))
}

pub(super) async fn web_fetch_manifest(
    url: String,
) -> std::result::Result<Vec<WebDatasetEntry>, String> {
    let text = web_fetch_text(&url).await?;
    let manifest: WebDatasetManifest =
        serde_json::from_str(&text).map_err(|err| format!("{url}: bad manifest JSON: {err}"))?;
    Ok(manifest.datasets)
}

pub(super) async fn web_fetch_dataset(
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
    let reference_position =
        web_fetch_optional_dataset_csv(&manifest_url, &entry, "reference_position")
            .await?
            .map(|(name, text)| NamedText { name, text });
    Ok(WebDatasetFiles {
        label,
        imu,
        gnss,
        reference_attitude,
        reference_mount,
        reference_position,
    })
}

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
        "reference_position" => (
            entry.reference_position.as_deref(),
            entry.reference_position_gz.as_deref(),
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
        "reference_position" => {
            entry.reference_position.is_some() || entry.reference_position_gz.is_some()
        }
        _ => false,
    };
    if !has_explicit_file {
        return Ok(None);
    }
    web_fetch_dataset_csv(manifest_url, entry, kind)
        .await
        .map(Some)
}

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

fn is_absolute_web_url(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://") || url.starts_with('/')
}

fn web_dataset_file_name(url: &str) -> String {
    url.rsplit('/').next().unwrap_or(url).to_string()
}

fn decode_plain_csv(url: &str, bytes: Vec<u8>) -> std::result::Result<String, String> {
    String::from_utf8(bytes).map_err(|err| format!("{url}: CSV was not UTF-8: {err}"))
}

fn decode_gzip_csv(url: &str, bytes: &[u8]) -> std::result::Result<String, String> {
    let mut decoder = GzDecoder::new(bytes);
    let mut text = String::new();
    decoder
        .read_to_string(&mut text)
        .map_err(|err| format!("{url}: gzip decode failed: {err}"))?;
    Ok(text)
}

pub(super) fn js_error_string(value: JsValue) -> String {
    value
        .as_string()
        .unwrap_or_else(|| "JavaScript error".to_string())
}

pub(super) fn web_initial_mapbox_token() -> String {
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

pub(super) fn web_remember_mapbox_token(token: &str) {
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

pub(super) fn web_initial_ui_theme() -> UiTheme {
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

pub(super) fn web_remember_ui_theme(theme: UiTheme) {
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

impl WebSyntheticScenario {
    pub(super) fn display_label(self) -> &'static str {
        match self {
            Self::CityBlocks => "City blocks",
            Self::FigureEight => "Figure eight",
            Self::FigureEightEarlyVelocityFault => "Figure eight early GNSS fault",
            Self::FigureEightRollExcitation => "Figure eight roll excitation + GNSS fault",
            Self::StraightAccelBrake => "Straight accel/brake",
        }
    }

    pub(super) fn scenario_text(self) -> (&'static str, &'static str) {
        match self {
            Self::CityBlocks => ("city_blocks_builtin.scenario", CITY_BLOCKS_SCENARIO),
            Self::FigureEight => ("figure8_builtin.scenario", FIGURE_EIGHT_SCENARIO),
            Self::FigureEightEarlyVelocityFault => (
                "figure8_early_gnss_fault_builtin.scenario",
                FIGURE_EIGHT_SCENARIO,
            ),
            Self::FigureEightRollExcitation => (
                "figure8_roll_excitation_builtin.scenario",
                FIGURE_EIGHT_ROLL_EXCITATION_SCENARIO,
            ),
            Self::StraightAccelBrake => (
                "straight_accel_brake_builtin.scenario",
                STRAIGHT_ACCEL_BRAKE_SCENARIO,
            ),
        }
    }

    pub(super) fn early_fault(self) -> ([f64; 3], Option<[f64; 2]>) {
        match self {
            Self::FigureEightEarlyVelocityFault | Self::FigureEightRollExcitation => {
                ([0.5, 0.0, 0.0], Some([120.0, 360.0]))
            }
            Self::CityBlocks | Self::FigureEight | Self::StraightAccelBrake => {
                ([0.0, 0.0, 0.0], None)
            }
        }
    }
}

impl WebSyntheticNoise {
    pub(super) fn display_label(self) -> &'static str {
        match self {
            Self::Truth => "None",
            Self::Low => "Low noise",
            Self::Mid => "Mid noise",
            Self::High => "High noise",
        }
    }

    pub(super) fn tooltip(self) -> &'static str {
        match self {
            Self::Truth => {
                "None\n\
                 IMU and GNSS are exact generated measurements.\n\
                 Use this to isolate filter formulation from sensor noise."
            }
            Self::Low => {
                "Low noise\n\
                 IMU: gyro ARW 0.05 deg/sqrt(hr), gyro bias drift 1 deg/hr\n\
                 IMU: accel VRW 0.015 m/s/sqrt(hr), accel bias drift 0.0002 m/s^2\n\
                 GNSS: position sigma 0.8 m horizontal, 1.2 m vertical\n\
                 GNSS: velocity sigma 0.03 m/s horizontal, 0.05 m/s vertical"
            }
            Self::Mid => {
                "Mid noise, consumer-grade reference point\n\
                 IMU: gyro ARW 0.3 deg/sqrt(hr), gyro bias drift 10 deg/hr\n\
                 IMU: accel VRW 0.05 m/s/sqrt(hr), accel bias drift 0.001 m/s^2\n\
                 GNSS: position sigma 3 m horizontal, 5 m vertical\n\
                 GNSS: velocity sigma 0.10 m/s horizontal, 0.15 m/s vertical"
            }
            Self::High => {
                "High noise\n\
                 IMU: gyro ARW 1.0 deg/sqrt(hr), gyro bias drift 30 deg/hr\n\
                 IMU: accel VRW 0.12 m/s/sqrt(hr), accel bias drift 0.005 m/s^2\n\
                 GNSS: position sigma 8 m horizontal, 12 m vertical\n\
                 GNSS: velocity sigma 0.30 m/s horizontal, 0.50 m/s vertical"
            }
        }
    }

    pub(super) fn cli_value(self) -> &'static str {
        match self {
            Self::Truth => "truth",
            Self::Low => "low",
            Self::Mid => "mid",
            Self::High => "high",
        }
    }
}

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

const FIGURE_EIGHT_SCENARIO: &str = r#"
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 60s
accelerate 0.6m/s^2 for 20s
wait 10s
repeat 11 {
    turn left 10dps for 36s
    turn right 10dps for 36s
}
brake 0.6666667m/s^2 for 18s
"#;

const FIGURE_EIGHT_ROLL_EXCITATION_SCENARIO: &str = r#"
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 60s
accelerate 0.6m/s^2 for 20s
wait 10s
repeat 11 {
    drive yaw=10 roll=0.25 for=18s
    drive yaw=10 roll=-0.25 for=18s
    drive yaw=-10 roll=-0.25 for=18s
    drive yaw=-10 roll=0.25 for=18s
}
brake 0.6666667m/s^2 for 18s
"#;

const STRAIGHT_ACCEL_BRAKE_SCENARIO: &str = r#"
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 20s
repeat 2 {
    accelerate 0.5m/s^2 for 20s
    wait 20s
    brake 0.5m/s^2 for 20s
    wait 15s
}
"#;

pub(super) fn draw_web_run_button(
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

impl App {
    pub(super) fn draw_web_bulk_loading_page(&self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered_justified(|ui| {
                ui.add_space((ui.available_height() * 0.35).max(24.0));
                ui.heading("Building replay");
                ui.add_space(8.0);
                ui.add(
                    egui::ProgressBar::new(self.web_run_progress.clamp(0.0, 0.95))
                        .desired_width((ui.available_width() * 0.45).clamp(220.0, 520.0))
                        .text(format!(
                            "{:.0}%",
                            100.0 * self.web_run_progress.clamp(0.0, 0.95)
                        )),
                );
                ui.add_space(6.0);
                ui.label(&self.web_status);
            });
        });
    }

    pub(super) fn start_web_manifest_load(&mut self, ctx: &egui::Context) {
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
        let repaint_ctx = ctx.clone();
        spawn_local(async move {
            let result = web_fetch_manifest(manifest_url).await;
            *pending.borrow_mut() = Some(WebDatasetTaskResult::Manifest(result));
            repaint_ctx.request_repaint();
        });
    }

    pub(super) fn start_web_dataset_load(&mut self, ctx: &egui::Context) {
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
        let repaint_ctx = ctx.clone();
        spawn_local(async move {
            let result = web_fetch_dataset(manifest_url, entry).await;
            *pending.borrow_mut() = Some(WebDatasetTaskResult::Dataset(result));
            repaint_ctx.request_repaint();
        });
    }

    pub(super) fn poll_web_dataset_tasks(&mut self, ctx: &egui::Context) {
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
                                self.start_web_dataset_load(ctx);
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
                        self.web_reference_position_csv = files.reference_position;
                        self.web_run_progress = self.web_run_progress.max(0.02);
                        self.start_web_replay_build(label, imu_name, gnss_name, ctx);
                    }
                    Err(err) => {
                        self.web_status = format!("Dataset load failed: {err}");
                    }
                }
            }
        }
    }

    pub(super) fn advance_web_run_progress(&mut self) {
        if self.web_datasets.loading_dataset || self.web_datasets.loading_replay {
            let elapsed_s = (web_now_s() - self.web_run_started_time_s).max(0.0);
            let estimated = (elapsed_s / self.web_run_estimated_duration_s.max(0.5)) as f32;
            self.web_run_progress = self.web_run_progress.max(estimated.min(0.95));
        }
    }

    pub(super) fn poll_web_replay_tasks(&mut self) {
        let Some(result) = self.web_datasets.pending_replay.borrow_mut().take() else {
            return;
        };
        match result {
            WebReplayTaskResult::Progress {
                job_id,
                progress,
                current_t_s,
                final_t_s,
            } => {
                if job_id != self.web_datasets.replay_job_id {
                    return;
                }
                self.web_run_progress = progress.clamp(0.0, 1.0);
                self.web_status = if final_t_s.is_finite() && final_t_s > 0.0 {
                    format!(
                        "Running replay: {:.2}s / {:.2}s ({:.1}%)",
                        current_t_s,
                        final_t_s,
                        100.0 * self.web_run_progress,
                    )
                } else {
                    format!("Running replay: {:.1}%", 100.0 * self.web_run_progress)
                }
            }
            WebReplayTaskResult::Complete { job_id, result } => {
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
                            self.map_center = map_center_from_traces(&self.data.reduced_map);
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
                            self.web_status = if is_synthetic {
                                format!("Synthetic scenario loaded: {}", output.label)
                            } else {
                                format!(
                                    "Dataset loaded: {} ({} / {})",
                                    output.label, output.imu_name, output.gnss_name,
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
        }
    }

    pub(super) fn refresh_from_generic_csv(&mut self, ctx: &egui::Context) -> bool {
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
            ctx,
        );
        true
    }

    pub(super) fn start_web_replay_build(
        &mut self,
        label: String,
        imu_name: String,
        gnss_name: String,
        ctx: &egui::Context,
    ) {
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
        let reference_position_text = self
            .web_reference_position_csv
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
                reference_position_csv: reference_position_text,
            },
            estimated_duration_s,
            ctx,
        );
    }

    pub(super) fn start_web_replay_worker(
        &mut self,
        job: WebReplayWorkerJob,
        estimated_duration_s: f64,
        ctx: &egui::Context,
    ) {
        let label = job.label().to_string();
        let run_cfg = self.tuning_cfg;
        self.web_run_started_time_s = web_now_s();
        self.web_run_estimated_duration_s = estimated_duration_s;
        self.finish_web_replay_worker();
        *self.web_datasets.pending_replay.borrow_mut() = None;
        self.web_datasets.replay_job_id = self.web_datasets.replay_job_id.wrapping_add(1);
        self.web_datasets.replay_cfg = run_cfg;
        let job_id = self.web_datasets.replay_job_id;
        self.web_run_progress = 0.02;

        let worker_options = WorkerOptions::new();
        worker_options.set_type(WorkerType::Module);
        let worker_url = format!("replay_worker.js?v={:.0}", Date::now());
        let worker = match Worker::new_with_options(&worker_url, &worker_options) {
            Ok(worker) => worker,
            Err(err) => {
                self.web_status =
                    format!("Failed to start replay worker: {}", js_error_string(err));
                return;
            }
        };

        let pending = Rc::clone(&self.web_datasets.pending_replay);
        let repaint_ctx = ctx.clone();
        let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
            let value = event.data();
            let output_job_id = Reflect::get(&value, &JsValue::from_str("jobId"))
                .ok()
                .and_then(|v| v.as_f64())
                .map(|v| v as u64)
                .unwrap_or(job_id);
            let is_progress = Reflect::get(&value, &JsValue::from_str("type"))
                .ok()
                .and_then(|v| v.as_string())
                .is_some_and(|value| value == "progress");
            if is_progress {
                let progress = Reflect::get(&value, &JsValue::from_str("progress"))
                    .ok()
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(0.0);
                let current_t_s = Reflect::get(&value, &JsValue::from_str("currentTimeS"))
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let final_t_s = Reflect::get(&value, &JsValue::from_str("finalTimeS"))
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                *pending.borrow_mut() = Some(WebReplayTaskResult::Progress {
                    job_id: output_job_id,
                    progress,
                    current_t_s,
                    final_t_s,
                });
                repaint_ctx.request_repaint();
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
            repaint_ctx.request_repaint();
        }) as Box<dyn FnMut(MessageEvent)>);

        let pending = Rc::clone(&self.web_datasets.pending_replay);
        let repaint_ctx = ctx.clone();
        let onerror = Closure::wrap(Box::new(move |event: ErrorEvent| {
            *pending.borrow_mut() = Some(WebReplayTaskResult::Complete {
                job_id,
                result: Err(format!("replay worker error: {}", event.message())),
            });
            repaint_ctx.request_repaint();
        }) as Box<dyn FnMut(ErrorEvent)>);

        worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));

        let request = web_replay_worker_request(
            job_id,
            &job,
            self.tuning_misalignment,
            run_cfg,
            self.tuning_gnss_outages,
        );
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

    pub(super) fn finish_web_replay_worker(&mut self) {
        if let Some(worker) = self.web_datasets.replay_worker.take() {
            worker.terminate();
        }
        self.web_datasets.replay_onmessage = None;
        self.web_datasets.replay_onerror = None;
    }

    pub(super) fn refresh_from_web_synthetic(&mut self, ctx: &egui::Context) {
        let (label, text) = self.web_scenario.scenario_text();
        let (early_vel_bias_ned_mps, early_fault_window_s) = self.web_scenario.early_fault();
        self.web_input_mode = WebInputMode::Synthetic;
        self.start_web_replay_worker(
            WebReplayWorkerJob::Synthetic {
                label: self.web_scenario.display_label().to_string(),
                motion_label: label.to_string(),
                motion_text: text.to_string(),
                noise: self.web_synthetic_noise,
                early_vel_bias_ned_mps,
                early_fault_window_s,
            },
            4.0,
            ctx,
        );
    }

    pub(super) fn publish_web_perf(&mut self, ctx: &egui::Context) {
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

    pub(super) fn consume_dropped_files(&mut self, ctx: &egui::Context) {
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
            } else if lower.contains("reference_position") || lower.contains("ref_pos") {
                self.web_reference_position_csv = Some(named);
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
