//! Always-visible visualizer controls: global trace toggles, map controls, page tabs, and web inputs.

use eframe::egui;

use crate::visualizer::model::Page;
use crate::visualizer::theme::UiTheme;

use super::App;
use super::state::{FULL_FILTER_LABEL, REDUCED_FILTER_LABEL, TuningPanel};
#[cfg(target_arch = "wasm32")]
use super::web::{
    WEB_MAX_POINTS_PER_TRACE, WEB_MIN_POINTS_PER_TRACE, WebDatasetEntry, WebInputMode,
    WebRealDataSource, WebSyntheticNoise, WebSyntheticScenario, draw_web_run_button,
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
                ui.label(format!("FPS {:.1}", self.fps_ema.max(fps)));
                ui.separator();
                ui.label("Theme");
                let mut selected_theme = self.ui_theme;
                ui.selectable_value(&mut selected_theme, UiTheme::Light, "Light");
                ui.selectable_value(&mut selected_theme, UiTheme::Dark, "Dark");
                if selected_theme != self.ui_theme {
                    self.set_ui_theme(selected_theme, ctx);
                }
                ui.separator();
                help_label(
                    ui,
                    "Traces",
                    "Show or hide result groups globally across plots. Reference is truth/reference data when available, Align is the standalone mount estimator, and Reduced/Full are the two filter implementations.",
                );
                ui.checkbox(&mut self.show_reference, "Reference");
                ui.checkbox(&mut self.show_align, "Align");
                ui.checkbox(&mut self.show_reduced, REDUCED_FILTER_LABEL);
                ui.checkbox(&mut self.show_full, FULL_FILTER_LABEL);
                ui.separator();
                help_label(
                    ui,
                    "Map",
                    "Control map overlays. GNSS toggles GNSS/reference trajectory traces, Heading toggles directional arrows, and the optional Mapbox token is set from the map corner button.",
                );
                ui.checkbox(&mut self.show_gnss_map, "GNSS");
                ui.checkbox(&mut self.show_heading, "Heading");
                ui.separator();
                help_label(
                    ui,
                    "Tune",
                    "Open filter tuning panels. Adjusted values are used when the simulation is run again or replay is applied.",
                );
                if ui.button(REDUCED_FILTER_LABEL).clicked() {
                    self.tuning_panel = Some(TuningPanel::Reduced);
                }
                if ui.button("Align").clicked() {
                    self.tuning_panel = Some(TuningPanel::Align);
                }
                if ui.button(FULL_FILTER_LABEL).clicked() {
                    self.tuning_panel = Some(TuningPanel::Full);
                }
                ui.separator();
                let inspector_response =
                    ui.toggle_value(&mut self.show_update_inspector, "Inspector");
                show_immediate_help(
                    ui,
                    &inspector_response,
                    "Open the update inspector window. Hover a plot to inspect recent measurement residuals and state correlations near that timestamp.",
                );
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
                                    let noise_label = ui.add(
                                        egui::Label::new(egui::RichText::new("Noise:").underline())
                                            .sense(egui::Sense::hover()),
                                    );
                                    if noise_label.hovered() {
                                        egui::Tooltip::always_open(
                                            ui.ctx().clone(),
                                            ui.layer_id(),
                                            noise_label.id,
                                            noise_label.rect,
                                        )
                                        .width(560.0)
                                        .show(draw_synthetic_noise_help);
                                    }
                                    egui::ComboBox::from_id_salt("web_synthetic_noise_select")
                                        .selected_text(self.web_synthetic_noise.display_label())
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut self.web_synthetic_noise,
                                                WebSyntheticNoise::Truth,
                                                WebSyntheticNoise::Truth.display_label(),
                                            )
                                            .on_hover_text(WebSyntheticNoise::Truth.tooltip());
                                            ui.selectable_value(
                                                &mut self.web_synthetic_noise,
                                                WebSyntheticNoise::Low,
                                                WebSyntheticNoise::Low.display_label(),
                                            )
                                            .on_hover_text(WebSyntheticNoise::Low.tooltip());
                                            ui.selectable_value(
                                                &mut self.web_synthetic_noise,
                                                WebSyntheticNoise::Mid,
                                                WebSyntheticNoise::Mid.display_label(),
                                            )
                                            .on_hover_text(WebSyntheticNoise::Mid.tooltip());
                                            ui.selectable_value(
                                                &mut self.web_synthetic_noise,
                                                WebSyntheticNoise::High,
                                                WebSyntheticNoise::High.display_label(),
                                            )
                                            .on_hover_text(WebSyntheticNoise::High.tooltip());
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

fn help_label(ui: &mut egui::Ui, text: &'static str, help: &'static str) {
    let response =
        ui.add(egui::Label::new(egui::RichText::new(text).underline()).sense(egui::Sense::hover()));
    show_immediate_help(ui, &response, help);
}

fn show_immediate_help(ui: &mut egui::Ui, response: &egui::Response, help: &'static str) {
    if response.hovered() {
        egui::Tooltip::always_open(ui.ctx().clone(), ui.layer_id(), response.id, response.rect)
            .width(360.0)
            .show(|ui| {
                ui.label(help);
            });
    }
}

#[cfg(target_arch = "wasm32")]
fn draw_synthetic_noise_help(ui: &mut egui::Ui) {
    ui.label(egui::RichText::new("Synthetic noise presets").strong());
    ui.add_space(2.0);
    ui.label(
        "Noise is applied to generated IMU and GNSS measurements. Values are 1-sigma figures.",
    );
    ui.add_space(8.0);

    egui::Grid::new("synthetic_noise_help_grid")
        .num_columns(5)
        .spacing([16.0, 5.0])
        .striped(true)
        .show(ui, |ui| {
            ui.strong("Preset");
            ui.strong("Gyro");
            ui.strong("Accel");
            ui.strong("GNSS pos");
            ui.strong("GNSS vel");
            ui.end_row();

            noise_help_row(ui, "None", "exact", "exact", "exact", "exact");
            noise_help_row(
                ui,
                "Low noise",
                "0.05 deg/sqrt(hr)\n1 deg/hr drift",
                "0.015 m/s/sqrt(hr)\n0.0002 m/s^2 drift",
                "0.8 m horiz\n1.2 m vert",
                "0.03 m/s horiz\n0.05 m/s vert",
            );
            noise_help_row(
                ui,
                "Mid noise",
                "0.3 deg/sqrt(hr)\n10 deg/hr drift",
                "0.05 m/s/sqrt(hr)\n0.001 m/s^2 drift",
                "3 m horiz\n5 m vert",
                "0.10 m/s horiz\n0.15 m/s vert",
            );
            noise_help_row(
                ui,
                "High noise",
                "1.0 deg/sqrt(hr)\n30 deg/hr drift",
                "0.12 m/s/sqrt(hr)\n0.005 m/s^2 drift",
                "8 m horiz\n12 m vert",
                "0.30 m/s horiz\n0.50 m/s vert",
            );
        });

    ui.add_space(6.0);
    ui.label("Mid noise is intended to be close to a consumer IMU/GNSS noise level.");
}

#[cfg(target_arch = "wasm32")]
fn noise_help_row(
    ui: &mut egui::Ui,
    preset: &str,
    gyro: &str,
    accel: &str,
    gnss_pos: &str,
    gnss_vel: &str,
) {
    ui.label(egui::RichText::new(preset).strong());
    ui.label(gyro);
    ui.label(accel);
    ui.label(gnss_pos);
    ui.label(gnss_vel);
    ui.end_row();
}
