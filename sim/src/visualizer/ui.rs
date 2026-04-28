use std::time::Duration;

use anyhow::Result;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
#[cfg(not(target_arch = "wasm32"))]
use walkers::sources::{Mapbox, MapboxStyle, OpenStreetMap};
#[cfg(not(target_arch = "wasm32"))]
use walkers::{HttpTiles, Map, MapMemory, Plugin, lon_lat};

#[cfg(not(target_arch = "wasm32"))]
use super::math::heading_endpoint;
use super::model::{EkfImuSource, HeadingSample, Page, PlotData, Trace};
#[cfg(target_arch = "wasm32")]
use super::pipeline::generic::{
    GenericReplayInput, build_generic_replay_plot_data, parse_generic_replay_csvs,
};
use super::pipeline::synthetic::{SyntheticVisualizerConfig, build_synthetic_plot_data};
use super::pipeline::{EkfCompareConfig, GnssOutageConfig};
#[cfg(not(target_arch = "wasm32"))]
use super::stats::map_center_from_traces;

#[cfg(not(target_arch = "wasm32"))]
const MAPBOX_ACCESS_TOKEN: &str = "pk.eyJ1IjoieW9uZ2t5dW5zODciLCJhIjoiY21tNjB5NWt6MGJmOTJzcG02MmRvN3RnYiJ9.fu_66qb1G1cgrLzAE54E0w";

#[cfg(not(target_arch = "wasm32"))]
struct TrackOverlay<'a> {
    traces: Vec<&'a Trace>,
    headings: Vec<&'a HeadingSample>,
    show_heading: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl Plugin for TrackOverlay<'_> {
    fn run(
        self: Box<Self>,
        ui: &mut egui::Ui,
        _response: &egui::Response,
        projector: &walkers::Projector,
        _map_memory: &MapMemory,
    ) {
        for tr in &self.traces {
            if tr.points.len() < 2 {
                continue;
            }
            let color = if tr.name.contains("GNSS") {
                egui::Color32::from_rgb(0, 255, 255)
            } else if tr.name == "ESKF path (lon,lat)" {
                egui::Color32::from_rgb(120, 170, 255)
            } else if tr.name == "ESKF path during GNSS outage (lon,lat)" {
                egui::Color32::from_rgb(255, 140, 220)
            } else if tr.name == "Loose path (lon,lat)" {
                egui::Color32::from_rgb(120, 255, 170)
            } else {
                egui::Color32::WHITE
            };
            let mut segment = Vec::<egui::Pos2>::with_capacity(tr.points.len());
            for p in &tr.points {
                let lon = p[0];
                let lat = p[1];
                if !lon.is_finite() || !lat.is_finite() {
                    if segment.len() >= 2 {
                        ui.painter().add(egui::epaint::PathShape::line(
                            segment,
                            egui::Stroke::new(2.2, color),
                        ));
                    }
                    segment = Vec::new();
                    continue;
                }
                let v = projector.project(lon_lat(lon, lat));
                segment.push(egui::pos2(v.x, v.y));
            }
            if segment.len() >= 2 {
                ui.painter().add(egui::epaint::PathShape::line(
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
                ui.painter().line_segment(
                    [egui::pos2(from.x, from.y), egui::pos2(to.x, to.y)],
                    egui::Stroke::new(1.8, egui::Color32::from_rgb(255, 255, 255)),
                );
            }
        }

        if let Some(mouse_pos) = ui.input(|i| i.pointer.hover_pos()) {
            let mut best: Option<(f32, &HeadingSample, egui::Pos2)> = None;
            for h in &self.headings {
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
                ui.painter()
                    .circle_filled(p, 3.0, egui::Color32::from_rgb(255, 220, 0));
                let label = format!("t={:.2}s", h.t_s);
                let bg_min = p + egui::vec2(8.0, -24.0);
                let bg_rect = egui::Rect::from_min_size(bg_min, egui::vec2(78.0, 18.0));
                ui.painter()
                    .rect_filled(bg_rect, 4.0, egui::Color32::from_black_alpha(180));
                ui.painter().text(
                    bg_min + egui::vec2(6.0, 2.0),
                    egui::Align2::LEFT_TOP,
                    label,
                    egui::FontId::monospace(12.0),
                    egui::Color32::WHITE,
                );
            }
        }
    }
}

pub struct App {
    data: PlotData,
    show_egui_inspection: bool,
    show_meas_accel: bool,
    has_itow: bool,
    fps_ema: f32,
    max_points_per_trace: usize,
    page: Page,
    #[cfg(not(target_arch = "wasm32"))]
    map_tiles: HttpTiles,
    #[cfg(not(target_arch = "wasm32"))]
    map_memory: MapMemory,
    #[cfg(not(target_arch = "wasm32"))]
    map_center: walkers::Position,
    show_heading: bool,
    show_gnss_map: bool,
    show_eskf: bool,
    show_loose: bool,
    replay: Option<ReplayState>,
    replay_status: Option<String>,
    #[cfg(target_arch = "wasm32")]
    web_imu_csv: Option<NamedText>,
    #[cfg(target_arch = "wasm32")]
    web_gnss_csv: Option<NamedText>,
    #[cfg(target_arch = "wasm32")]
    web_scenario: WebSyntheticScenario,
    #[cfg(target_arch = "wasm32")]
    web_status: String,
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
#[derive(Clone, Copy, PartialEq, Eq)]
enum WebSyntheticScenario {
    CityBlocks,
    FigureEight,
    StraightAccelBrake,
}

fn create_app(
    cc: &eframe::CreationContext<'_>,
    data: PlotData,
    has_itow: bool,
    replay: Option<ReplayState>,
) -> App {
    #[cfg(target_arch = "wasm32")]
    let _ = cc;
    #[cfg(not(target_arch = "wasm32"))]
    let map_center = map_center_from_traces(&data.eskf_map);
    #[cfg(not(target_arch = "wasm32"))]
    let mapbox_access_token =
        std::env::var("MAPBOX_ACCESS_TOKEN").unwrap_or_else(|_| MAPBOX_ACCESS_TOKEN.to_string());
    #[cfg(not(target_arch = "wasm32"))]
    let map_tiles = if mapbox_access_token.is_empty() {
        HttpTiles::new(OpenStreetMap, cc.egui_ctx.clone())
    } else {
        HttpTiles::new(
            Mapbox {
                style: MapboxStyle::Dark,
                high_resolution: true,
                access_token: mapbox_access_token,
            },
            cc.egui_ctx.clone(),
        )
    };
    #[cfg(not(target_arch = "wasm32"))]
    let mut map_memory = MapMemory::default();
    #[cfg(not(target_arch = "wasm32"))]
    let _ = map_memory.set_zoom(15.0);
    App {
        data,
        show_egui_inspection: false,
        show_meas_accel: false,
        has_itow,
        fps_ema: 0.0,
        max_points_per_trace: 2500,
        page: Page::Signals,
        #[cfg(not(target_arch = "wasm32"))]
        map_tiles,
        #[cfg(not(target_arch = "wasm32"))]
        map_memory,
        #[cfg(not(target_arch = "wasm32"))]
        map_center,
        show_heading: false,
        show_gnss_map: true,
        show_eskf: true,
        show_loose: true,
        replay,
        replay_status: None,
        #[cfg(target_arch = "wasm32")]
        web_imu_csv: None,
        #[cfg(target_arch = "wasm32")]
        web_gnss_csv: None,
        #[cfg(target_arch = "wasm32")]
        web_scenario: WebSyntheticScenario::CityBlocks,
        #[cfg(target_arch = "wasm32")]
        web_status: "Drag imu.csv and gnss.csv onto the app, or run a built-in synthetic scenario."
            .to_string(),
    }
}

impl App {
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
                    self.has_itow = false;
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
    fn refresh_from_generic_csv(&mut self) {
        let (Some(imu), Some(gnss)) = (&self.web_imu_csv, &self.web_gnss_csv) else {
            self.web_status =
                "Load both imu.csv and gnss.csv before running CSV replay.".to_string();
            return;
        };
        let imu_name = imu.name.clone();
        let gnss_name = gnss.name.clone();
        let imu_text = imu.text.clone();
        let gnss_text = gnss.text.clone();
        match parse_generic_replay_csvs(&imu_text, &gnss_text) {
            Ok(replay) => {
                self.set_generic_replay(replay);
                self.web_status = format!("CSV replay loaded: {imu_name} and {gnss_name}");
            }
            Err(err) => {
                self.web_status = format!("CSV replay failed: {err}");
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn set_generic_replay(&mut self, replay: GenericReplayInput) {
        let data = build_generic_replay_plot_data(
            &replay,
            EkfImuSource::Internal,
            EkfCompareConfig::default(),
            GnssOutageConfig::default(),
        );
        self.data = data;
        self.has_itow = false;
        self.page = Page::EskfCompare;
    }

    #[cfg(target_arch = "wasm32")]
    fn refresh_from_web_synthetic(&mut self) {
        let (label, text) = self.web_scenario.scenario_text();
        let synth_cfg = SyntheticVisualizerConfig {
            motion_def: None,
            motion_label: label.to_string(),
            motion_text: Some(text.to_string()),
            noise_mode: super::pipeline::synthetic::SyntheticNoiseMode::Low,
            seed: 1,
            mount_rpy_deg: [5.0, -5.0, 5.0],
            imu_hz: 100.0,
            gnss_hz: 2.0,
            gnss_time_shift_ms: 0.0,
            early_vel_bias_ned_mps: [0.0, 0.0, 0.0],
            early_fault_window_s: None,
        };
        match build_synthetic_plot_data(
            &synth_cfg,
            EkfImuSource::Internal,
            EkfCompareConfig::default(),
            GnssOutageConfig::default(),
        ) {
            Ok(data) => {
                self.data = data;
                self.has_itow = false;
                self.page = Page::EskfCompare;
                self.web_status = format!("Synthetic scenario loaded: {label}");
            }
            Err(err) => {
                self.web_status = format!("Synthetic scenario failed: {err}");
            }
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
            if lower.contains("gnss") {
                self.web_gnss_csv = Some(named);
            } else if lower.contains("imu") || lower.contains("acc") || lower.contains("gyro") {
                self.web_imu_csv = Some(named);
            } else if self.web_imu_csv.is_none() {
                self.web_imu_csv = Some(named);
            } else {
                self.web_gnss_csv = Some(named);
            }
        }
        self.web_status = "Dropped file(s) staged. Click Run CSV replay.".to_string();
    }
}

#[cfg(target_arch = "wasm32")]
impl WebSyntheticScenario {
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
command type,yaw (deg),pitch (deg),roll (deg),vx_body (m/s),vy_body (m/s),vz_body (m/s),command duration (s),GPS visibility
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
command type,yaw (deg),pitch (deg),roll (deg),vx_body (m/s),vy_body (m/s),vz_body (m/s),command duration (s),GPS visibility
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
        #[cfg(target_arch = "wasm32")]
        self.consume_dropped_files(ctx);

        #[cfg(target_os = "macos")]
        if ctx.input(|i| i.viewport().close_requested()) {
            std::process::exit(0);
        }

        if matches!(self.page, Page::MapDark) || self.show_egui_inspection {
            ctx.request_repaint_after(Duration::from_millis(33));
        }

        egui::TopBottomPanel::top("top_controls").show(ctx, |ui| {
            let mut replay_changed = false;
            let mut apply_replay = false;
            let fps = ctx.input(|i| {
                if i.stable_dt > 0.0 {
                    1.0 / i.stable_dt
                } else {
                    0.0
                }
            });
            if self.fps_ema <= 0.0 {
                self.fps_ema = fps;
            } else {
                self.fps_ema = self.fps_ema * 0.92 + fps * 0.08;
            }
            if self.fps_ema < 24.0 {
                self.max_points_per_trace = (self.max_points_per_trace as f32 * 0.85) as usize;
            } else if self.fps_ema > 50.0 {
                self.max_points_per_trace = (self.max_points_per_trace as f32 * 1.08) as usize;
            }
            self.max_points_per_trace = self.max_points_per_trace.clamp(300, 6000);
            ui.horizontal(|ui| {
                ui.heading("pygpsdata Visualization (Rust + egui)");
                ui.separator();
                ui.label(if self.has_itow {
                    "X-axis: Relative time [s], t=0 at first iTOW"
                } else {
                    "X-axis: Relative time [s] (no valid iTOW found)"
                });
            });
            egui::CollapsingHeader::new("Plot Controls")
                .default_open(false)
                .show(ui, |ui| {
                    ui.label(format!("Estimated FPS: {:.1}", fps));
                    ui.label(format!(
                        "Decimation budget: {} pts/trace (FPS EMA {:.1})",
                        self.max_points_per_trace, self.fps_ema
                    ));
                    ui.checkbox(&mut self.show_meas_accel, "Show IMU measurement (Accel)");
                    ui.checkbox(
                        &mut self.show_egui_inspection,
                        "Show egui inspection/profiler",
                    );
                });
            #[cfg(target_arch = "wasm32")]
            egui::CollapsingHeader::new("Browser Inputs")
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Synthetic:");
                        ui.selectable_value(
                            &mut self.web_scenario,
                            WebSyntheticScenario::CityBlocks,
                            "City blocks",
                        );
                        ui.selectable_value(
                            &mut self.web_scenario,
                            WebSyntheticScenario::FigureEight,
                            "Figure eight",
                        );
                        ui.selectable_value(
                            &mut self.web_scenario,
                            WebSyntheticScenario::StraightAccelBrake,
                            "Straight accel/brake",
                        );
                        if ui.button("Run synthetic").clicked() {
                            self.refresh_from_web_synthetic();
                        }
                    });
                    ui.horizontal_wrapped(|ui| {
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
                        ui.label(format!("CSV: {imu_name} / {gnss_name}"));
                        if ui.button("Run CSV replay").clicked() {
                            self.refresh_from_generic_csv();
                        }
                    });
                    ui.label(&self.web_status);
                });
            if let Some(replay) = self.replay.as_mut() {
                egui::CollapsingHeader::new("Replay Controls")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.label("Adjust controls, then click Apply to rebuild the replay.");
                        ui.horizontal(|ui| {
                            ui.label("Misalignment:");
                            replay_changed |= ui
                                .selectable_value(
                                    &mut replay.misalignment,
                                    EkfImuSource::Internal,
                                    "internal",
                                )
                                .changed();
                            replay_changed |= ui
                                .selectable_value(
                                    &mut replay.misalignment,
                                    EkfImuSource::External,
                                    "external",
                                )
                                .changed();
                            replay_changed |= ui
                                .selectable_value(
                                    &mut replay.misalignment,
                                    EkfImuSource::Ref,
                                    "ref",
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
                                    egui::DragValue::new(&mut replay.ekf_cfg.yaw_init_speed_mps)
                                        .speed(0.1)
                                        .range(0.0..=20.0),
                                )
                                .changed();
                            ui.label("Yaw sigma deg");
                            replay_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut replay.ekf_cfg.yaw_init_sigma_deg)
                                        .speed(0.5)
                                        .range(0.0..=90.0),
                                )
                                .changed();
                            ui.label("Mount init deg");
                            replay_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut replay.ekf_cfg.mount_init_sigma_deg)
                                        .speed(0.5)
                                        .range(0.0..=90.0),
                                )
                                .changed();
                        });
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Mount RW");
                            replay_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut replay.ekf_cfg.mount_align_rw_var)
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
                                    egui::DragValue::new(&mut replay.ekf_cfg.align_handoff_delay_s)
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
                                    egui::DragValue::new(&mut replay.ekf_cfg.mount_settle_time_s)
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
                                    egui::DragValue::new(&mut replay.ekf_cfg.gnss_pos_mount_scale)
                                        .speed(0.01)
                                        .range(0.0..=1.0),
                                )
                                .changed();
                            ui.label("GNSS vel->mount");
                            replay_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut replay.ekf_cfg.gnss_vel_mount_scale)
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
                                    egui::DragValue::new(&mut replay.ekf_cfg.r_stationary_accel)
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
                                    Some(replay.ekf_cfg.predict_imu_lpf_cutoff_hz.unwrap_or(150.0))
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
                        ui.horizontal(|ui| {
                            apply_replay = ui.button("Apply").clicked();
                            if let Some(status) = &self.replay_status {
                                ui.label(status);
                            }
                        });
                    });
            }
            ui.horizontal(|ui| {
                ui.label("Page:");
                ui.selectable_value(&mut self.page, Page::Signals, "Signals");
                ui.selectable_value(&mut self.page, Page::EskfCompare, "ESKF Compare");
                ui.selectable_value(&mut self.page, Page::LooseCompare, "Loose Compare");
                ui.selectable_value(&mut self.page, Page::EskfBump, "ESKF Bump");
                ui.selectable_value(&mut self.page, Page::AlignCompare, "Align Compare");
                ui.selectable_value(&mut self.page, Page::MapDark, "Map (Dark)");
            });
            if replay_changed {
                self.replay_status = Some("Pending changes".to_string());
            }
            if apply_replay {
                self.refresh_from_replay();
            }
        });

        let mut imu_gyro: Vec<&Trace> =
            Vec::with_capacity(self.data.imu_raw_gyro.len() + self.data.imu_cal_gyro.len());
        imu_gyro.extend(self.data.imu_raw_gyro.iter());
        imu_gyro.extend(
            self.data
                .imu_cal_gyro
                .iter()
                .filter(|t| !t.name.starts_with("IMU measurement ")),
        );

        let mut imu_accel: Vec<&Trace> =
            Vec::with_capacity(self.data.imu_raw_accel.len() + self.data.imu_cal_accel.len());
        imu_accel.extend(self.data.imu_raw_accel.iter());
        imu_accel.extend(self.data.imu_cal_accel.iter());
        if !self.show_meas_accel {
            imu_accel.retain(|t| !t.name.starts_with("IMU measurement "));
        }

        match self.page {
            Page::Signals => {
                let half_width = (ctx.content_rect().width() * 0.5).max(260.0);
                egui::SidePanel::left("left_plots")
                    .resizable(false)
                    .exact_width(half_width)
                    .show(ctx, |ui| {
                        draw_plot(
                            ui,
                            "Speed",
                            self.data.speed.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "IMU Gyro Reference (RAW/CAL)",
                            imu_gyro.iter().copied(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Orientation",
                            self.data.orientation.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                    });

                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_plot(
                        ui,
                        "Signal Strength (C/N0)",
                        self.data.sat_cn0.iter(),
                        false,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "IMU Accel Reference (RAW/CAL/MEAS)",
                        imu_accel.iter().copied(),
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "Other Signals",
                        self.data.other.iter(),
                        true,
                        self.max_points_per_trace,
                    );
                });
            }
            Page::EskfCompare => {
                let half_width = (ctx.content_rect().width() * 0.5).max(260.0);
                let mut vehicle_gyro: Vec<&Trace> =
                    Vec::with_capacity(self.data.eskf_meas_gyro.len());
                vehicle_gyro.extend(self.data.eskf_meas_gyro.iter());
                let mut vehicle_accel: Vec<&Trace> =
                    Vec::with_capacity(self.data.eskf_meas_accel.len());
                vehicle_accel.extend(self.data.eskf_meas_accel.iter());
                egui::SidePanel::left("eskf_compare_left")
                    .resizable(false)
                    .exact_width(half_width)
                    .show(ctx, |ui| {
                        draw_plot(
                            ui,
                            "Vehicle Velocity: ESKF vs GNSS",
                            self.data.eskf_cmp_vel.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Euler Angles: ESKF Quaternion vs reference attitude",
                            self.data.eskf_cmp_att.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "ESKF Misalignment Estimates",
                            self.data.eskf_misalignment.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "ESKF Gyro Bias Estimates",
                            self.data.eskf_bias_gyro.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Vehicle Gyro: ESKF input",
                            vehicle_gyro.iter().copied(),
                            true,
                            self.max_points_per_trace,
                        );
                    });

                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_plot(
                        ui,
                        "ESKF Accel Bias Estimates",
                        self.data.eskf_bias_accel.iter(),
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "ESKF Bias Covariance Diagonal",
                        self.data.eskf_cov_bias.iter(),
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "ESKF Covariance Diagonal (Non-bias States)",
                        self.data.eskf_cov_nonbias.iter(),
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "Vehicle Accel: ESKF input",
                        vehicle_accel.iter().copied(),
                        true,
                        self.max_points_per_trace,
                    );
                });
            }
            Page::EskfBump => {
                let half_width = (ctx.content_rect().width() * 0.5).max(260.0);
                egui::SidePanel::left("eskf_bump_left")
                    .resizable(false)
                    .exact_width(half_width)
                    .show(ctx, |ui| {
                        draw_plot(
                            ui,
                            "ESKF Pitch Angle",
                            self.data
                                .eskf_bump_pitch_speed
                                .iter()
                                .filter(|t| t.name.contains("pitch")),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Vehicle Speed",
                            self.data
                                .eskf_bump_pitch_speed
                                .iter()
                                .filter(|t| t.name.contains("speed")),
                            true,
                            self.max_points_per_trace,
                        );
                    });

                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_plot(
                        ui,
                        "Pitch HPF / RMS / EMA (3.0 s)",
                        self.data
                            .eskf_bump_diag
                            .iter()
                            .filter(|t| !t.name.contains("FFT dom")),
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "Pitch FFT Metrics (3.0 s trailing window)",
                        self.data
                            .eskf_bump_diag
                            .iter()
                            .filter(|t| t.name.contains("FFT dom")),
                        true,
                        self.max_points_per_trace,
                    );
                });
            }
            Page::LooseCompare => {
                let half_width = (ctx.content_rect().width() * 0.5).max(260.0);
                let mut vehicle_gyro: Vec<&Trace> =
                    Vec::with_capacity(self.data.loose_meas_gyro.len());
                vehicle_gyro.extend(self.data.loose_meas_gyro.iter());
                let mut vehicle_accel: Vec<&Trace> =
                    Vec::with_capacity(self.data.loose_meas_accel.len());
                vehicle_accel.extend(self.data.loose_meas_accel.iter());
                egui::SidePanel::left("loose_compare_left")
                    .resizable(false)
                    .exact_width(half_width)
                    .show(ctx, |ui| {
                        draw_plot(
                            ui,
                            "Vehicle Velocity: Loose INS/GNSS vs GNSS",
                            self.data.loose_cmp_vel.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Euler Angles: Loose INS/GNSS vs reference attitude",
                            self.data.loose_cmp_att.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Loose Misalignment Estimates",
                            self.data.loose_misalignment.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Loose Gyro Bias Estimates",
                            self.data.loose_bias_gyro.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Vehicle Gyro: Loose INS/GNSS input",
                            vehicle_gyro.iter().copied(),
                            true,
                            self.max_points_per_trace,
                        );
                    });

                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_plot(
                        ui,
                        "Loose Accel Bias Estimates",
                        self.data.loose_bias_accel.iter(),
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "Loose Scale Factor Estimates",
                        self.data
                            .loose_scale_gyro
                            .iter()
                            .chain(self.data.loose_scale_accel.iter()),
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "Loose Covariance Diagonal (Bias/Scale States)",
                        self.data.loose_cov_bias.iter(),
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "Vehicle Accel: Loose INS/GNSS input",
                        vehicle_accel.iter().copied(),
                        true,
                        self.max_points_per_trace,
                    );
                });
            }
            Page::AlignCompare => {
                let half_width = (ctx.content_rect().width() * 0.5).max(260.0);
                egui::SidePanel::left("align_compare_left")
                    .resizable(false)
                    .exact_width(half_width)
                    .show(ctx, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            draw_plot(
                                ui,
                                "Euler Angles: Align KF vs Reference mount",
                                self.data.align_cmp_att.iter(),
                                true,
                                self.max_points_per_trace,
                            );
                            draw_plot(
                                ui,
                                "Align Window Diagnostics",
                                self.data.align_res_vel.iter(),
                                true,
                                self.max_points_per_trace,
                            );
                            draw_plot(
                                ui,
                                "Align Axis Error vs Reference mount",
                                self.data.align_axis_err.iter(),
                                true,
                                self.max_points_per_trace,
                            );
                            draw_plot(
                                ui,
                                "Final Reference mount vs PCA Heading",
                                self.data.align_motion.iter(),
                                true,
                                self.max_points_per_trace,
                            );
                            draw_plot(
                                ui,
                                "Align Window Flags",
                                self.data.align_flags.iter(),
                                true,
                                self.max_points_per_trace,
                            );
                        });
                    });

                egui::CentralPanel::default().show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        draw_plot(
                            ui,
                            "Align Roll Update Contributions",
                            self.data.align_roll_contrib.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Align Pitch Update Contributions",
                            self.data.align_pitch_contrib.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Align Yaw Update Contributions",
                            self.data.align_yaw_contrib.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Align Covariance Diagonal",
                            self.data.align_cov.iter(),
                            true,
                            self.max_points_per_trace,
                        );
                    });
                });
            }
            Page::MapDark => {
                #[cfg(not(target_arch = "wasm32"))]
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Map overlay: GNSS + ESKF");
                        ui.checkbox(&mut self.show_heading, "show heading");
                        ui.checkbox(&mut self.show_gnss_map, "show GNSS");
                        ui.checkbox(&mut self.show_eskf, "show ESKF");
                        ui.checkbox(&mut self.show_loose, "show Loose");
                        if ui.button("Recenter").clicked() {
                            self.map_memory.follow_my_position();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.colored_label(egui::Color32::from_rgb(0, 255, 255), "GNSS");
                        ui.colored_label(egui::Color32::from_rgb(120, 170, 255), "ESKF");
                        ui.colored_label(egui::Color32::from_rgb(120, 255, 170), "Loose");
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 140, 220),
                            "ESKF during GNSS outage",
                        );
                        ui.colored_label(egui::Color32::from_rgb(255, 255, 255), "ESKF heading");
                    });
                    let mut map_traces: Vec<&Trace> = self.data.eskf_map.iter().collect();
                    if !self.show_gnss_map {
                        map_traces.retain(|t| !t.name.contains("GNSS"));
                    }
                    if !self.show_eskf {
                        map_traces.retain(|t| {
                            t.name != "ESKF path (lon,lat)"
                                && t.name != "ESKF path during GNSS outage (lon,lat)"
                        });
                    }
                    if self.show_loose {
                        map_traces.extend(self.data.loose_map.iter());
                    }
                    let mut headings: Vec<&HeadingSample> =
                        self.data.eskf_map_heading.iter().collect();
                    if self.show_loose {
                        headings.extend(self.data.loose_map_heading.iter());
                    }
                    let track = TrackOverlay {
                        traces: map_traces,
                        headings,
                        show_heading: self.show_heading,
                    };
                    ui.add(
                        Map::new(
                            Some(&mut self.map_tiles),
                            &mut self.map_memory,
                            self.map_center,
                        )
                        .with_plugin(track)
                        .double_click_to_zoom(true),
                    );
                });
                #[cfg(target_arch = "wasm32")]
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Map traces");
                        ui.checkbox(&mut self.show_heading, "show heading");
                        ui.checkbox(&mut self.show_gnss_map, "show GNSS");
                        ui.checkbox(&mut self.show_eskf, "show ESKF");
                        ui.checkbox(&mut self.show_loose, "show Loose");
                    });
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
                    draw_lon_lat_plot(ui, map_traces.iter().copied(), self.max_points_per_trace);
                    if self.show_heading {
                        draw_heading_plot(ui, &self.data.eskf_map_heading);
                    }
                });
            }
        }

        if self.show_egui_inspection {
            egui::Window::new("egui inspection/profiler")
                .vscroll(true)
                .show(ctx, |ui| {
                    ctx.inspection_ui(ui);
                });
        }
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

            let mut min_b: Vec<Option<(usize, f64)>> = vec![None; buckets];
            let mut max_b: Vec<Option<(usize, f64)>> = vec![None; buckets];
            for (i, p) in slice.iter().enumerate() {
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
                return slice.iter().step_by(step.max(1)).copied().collect();
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

    ui.vertical(|ui| {
        ui.label(title);
        let mut plot = Plot::new(title)
            .height(200.0)
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
        plot.show(ui, |plot_ui| {
            let bounds = plot_ui.plot_bounds();
            let xmin = bounds.min()[0];
            let xmax = bounds.max()[0];
            for t in traces {
                if t.points.is_empty() {
                    continue;
                }
                let reduced = visible_decimated(&t.points, xmin, xmax, max_points_per_trace);
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
        });
    });
}

#[cfg(target_arch = "wasm32")]
fn draw_lon_lat_plot<'a, I>(ui: &mut egui::Ui, traces: I, max_points_per_trace: usize)
where
    I: IntoIterator<Item = &'a Trace>,
{
    ui.vertical(|ui| {
        ui.label("Longitude / Latitude");
        Plot::new("web_lon_lat_map")
            .height((ui.available_height() - 20.0).max(320.0))
            .data_aspect(1.0)
            .allow_drag(true)
            .allow_zoom(true)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                for trace in traces {
                    if trace.points.is_empty() {
                        continue;
                    }
                    let step = (trace.points.len() / max_points_per_trace.max(1)).max(1);
                    let points: PlotPoints<'_> =
                        trace.points.iter().step_by(step).copied().collect();
                    plot_ui.line(Line::new(trace.name.clone(), points));
                }
            });
    });
}

#[cfg(target_arch = "wasm32")]
fn draw_heading_plot(ui: &mut egui::Ui, headings: &[HeadingSample]) {
    if headings.is_empty() {
        return;
    }
    let points: PlotPoints<'_> = headings
        .iter()
        .step_by((headings.len() / 500).max(1))
        .map(|h| [h.t_s, h.yaw_deg])
        .collect();
    ui.label("Heading");
    Plot::new("web_heading")
        .height(160.0)
        .allow_drag(true)
        .allow_zoom(true)
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new("ESKF heading [deg]", points));
        });
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_visualizer(data: PlotData, has_itow: bool, replay: Option<ReplayState>) -> Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_maximized(true),
        ..Default::default()
    };
    eframe::run_native(
        "visualizer",
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
