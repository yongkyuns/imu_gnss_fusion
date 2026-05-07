//! Always-visible visualizer controls: global trace toggles, map controls, page tabs, and web inputs.

use eframe::egui;

use crate::visualizer::model::Page;
use crate::visualizer::theme::UiTheme;

use super::App;
use super::state::TuningPanel;
#[cfg(target_arch = "wasm32")]
use super::web::{
    WEB_MAX_POINTS_PER_TRACE, WEB_MIN_POINTS_PER_TRACE, WebDatasetEntry, WebInputMode,
    WebRealDataSource, WebSyntheticNoise, WebSyntheticScenario, draw_web_run_button,
    web_remember_mapbox_token,
};

impl App {
    pub(super) fn draw_top_controls(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_controls").show(ctx, |ui| {
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
                self.max_points_per_trace = self
                    .max_points_per_trace
                    .clamp(WEB_MIN_POINTS_PER_TRACE, WEB_MAX_POINTS_PER_TRACE);
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
                ui.label(format!(
                    "FPS {:.1} | detail {}",
                    self.fps_ema.max(fps),
                    self.max_points_per_trace
                ));
                ui.separator();
                ui.label("Theme");
                let mut selected_theme = self.ui_theme;
                ui.selectable_value(&mut selected_theme, UiTheme::Light, "Light");
                ui.selectable_value(&mut selected_theme, UiTheme::Dark, "Dark");
                if selected_theme != self.ui_theme {
                    self.set_ui_theme(selected_theme, ctx);
                }
                ui.separator();
                ui.label("Traces");
                ui.checkbox(&mut self.show_reference, "Reference");
                ui.checkbox(&mut self.show_align, "Align");
                ui.checkbox(&mut self.show_eskf, "ESKF");
                ui.checkbox(&mut self.show_loose, "Loose");
                ui.separator();
                ui.label("Map");
                ui.checkbox(&mut self.show_gnss_map, "GNSS");
                ui.checkbox(&mut self.show_heading, "Heading");
                if ui.button("Recenter").clicked() {
                    self.map_memory.follow_my_position();
                }
                #[cfg(target_arch = "wasm32")]
                {
                    ui.label("Mapbox");
                    let token_width = ui.available_width().clamp(100.0, 180.0);
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
                }
                ui.separator();
                ui.label("Tune");
                if ui.button("ESKF").clicked() {
                    self.tuning_panel = Some(TuningPanel::Eskf);
                }
                if ui.button("Align").clicked() {
                    self.tuning_panel = Some(TuningPanel::Align);
                }
                if ui.button("Loose").clicked() {
                    self.tuning_panel = Some(TuningPanel::Loose);
                }
                ui.separator();
                ui.toggle_value(&mut self.show_update_inspector, "Inspector");
            });
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.page, Page::Overview, "Overview");
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
                                                WebSyntheticScenario::FigureEightEarlyVelocityFault,
                                                WebSyntheticScenario::FigureEightEarlyVelocityFault
                                                    .display_label(),
                                            );
                                            ui.selectable_value(
                                                &mut self.web_scenario,
                                                WebSyntheticScenario::FigureEightRollExcitation,
                                                WebSyntheticScenario::FigureEightRollExcitation
                                                    .display_label(),
                                            );
                                            ui.selectable_value(
                                                &mut self.web_scenario,
                                                WebSyntheticScenario::StraightAccelBrake,
                                                WebSyntheticScenario::StraightAccelBrake
                                                    .display_label(),
                                            );
                                        });
                                    ui.label("Noise:");
                                    egui::ComboBox::from_id_salt("web_synthetic_noise_select")
                                        .selected_text(self.web_synthetic_noise.display_label())
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut self.web_synthetic_noise,
                                                WebSyntheticNoise::Truth,
                                                WebSyntheticNoise::Truth.display_label(),
                                            );
                                            ui.selectable_value(
                                                &mut self.web_synthetic_noise,
                                                WebSyntheticNoise::Low,
                                                WebSyntheticNoise::Low.display_label(),
                                            );
                                            ui.selectable_value(
                                                &mut self.web_synthetic_noise,
                                                WebSyntheticNoise::Mid,
                                                WebSyntheticNoise::Mid.display_label(),
                                            );
                                            ui.selectable_value(
                                                &mut self.web_synthetic_noise,
                                                WebSyntheticNoise::High,
                                                WebSyntheticNoise::High.display_label(),
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
                                    WebInputMode::Synthetic => self.refresh_from_web_synthetic(ctx),
                                    WebInputMode::RealData => match self.web_real_data_source {
                                        WebRealDataSource::DroppedCsv => {
                                            self.refresh_from_generic_csv(ctx);
                                        }
                                        WebRealDataSource::ManifestDataset => {
                                            self.start_web_dataset_load(ctx);
                                        }
                                    },
                                }
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
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(status) = &self.replay_status {
                    ui.label(status);
                }
            }
        });
    }
}
