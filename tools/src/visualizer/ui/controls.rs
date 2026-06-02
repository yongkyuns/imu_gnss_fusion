//! Always-visible visualizer controls: global trace toggles, map controls, page tabs, and web inputs.

use eframe::egui;

use crate::visualizer::model::{Page, VisualizerFusionBackend};
use crate::visualizer::theme::UiTheme;

use super::App;
#[cfg(not(target_arch = "wasm32"))]
use super::input::{DATASET_MANIFEST_PATH, NativeRealDataSource};
use super::input::{
    HostedDatasetEntry, InputMode, SYNTHETIC_NOISE_PRESETS, SYNTHETIC_SCENARIOS, SyntheticNoise,
    SyntheticScenario, draw_run_button, draw_synthetic_noise_help,
};
use super::state::{EKF_FILTER_LABEL, TuningPanel};
#[cfg(target_arch = "wasm32")]
use super::web::{WEB_MAX_POINTS_PER_TRACE, WEB_MIN_POINTS_PER_TRACE, WebRealDataSource};

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
                    "Show or hide result groups globally across plots. Reference is truth/reference data when available, Align is the standalone mount estimator, and EKF is the runtime filter.",
                );
                ui.checkbox(&mut self.show_reference, "Reference");
                ui.checkbox(&mut self.show_align, "Align");
                ui.checkbox(&mut self.show_ekf, EKF_FILTER_LABEL);
                ui.separator();
                help_label(
                    ui,
                    "Map",
                    "Control map overlays. GNSS toggles GNSS/reference trajectory traces, Heading toggles directional arrows, Events toggles detected road-event markers/segments, and map-specific options live in the map corner controls.",
                );
                ui.checkbox(&mut self.show_gnss_map, "GNSS");
                ui.checkbox(&mut self.show_heading, "Heading");
                ui.checkbox(&mut self.show_events, "Events");
                ui.add_enabled_ui(self.show_events, |ui| {
                    ui.menu_button("Filter", |ui| {
                        ui.set_min_width(190.0);
                        ui.label(egui::RichText::new("Event Types").strong());
                        ui.separator();
                        ui.checkbox(&mut self.event_visibility.show_speed_bump, "Speed bumps");
                        ui.checkbox(&mut self.event_visibility.show_road_shock, "Road shocks");
                        ui.checkbox(&mut self.event_visibility.show_rough_road, "Rough road");
                        ui.checkbox(&mut self.event_visibility.show_uphill, "Uphill");
                        ui.checkbox(&mut self.event_visibility.show_downhill, "Downhill");
                        ui.checkbox(&mut self.event_visibility.show_reverse, "Reverse");
                        ui.checkbox(&mut self.event_visibility.show_harsh_accel, "Harsh accel");
                        ui.checkbox(&mut self.event_visibility.show_harsh_brake, "Harsh brake");
                        ui.checkbox(
                            &mut self.event_visibility.show_harsh_cornering,
                            "Harsh cornering",
                        );
                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button("All").clicked() {
                                self.event_visibility.set_all(true);
                            }
                            if ui.button("None").clicked() {
                                self.event_visibility.set_all(false);
                            }
                        });
                    });
                });
                ui.separator();
                help_label(
                    ui,
                    "Tune",
                    "Open filter tuning panels. Adjusted values are used when the simulation is run again or replay is applied.",
                );
                if ui.button(EKF_FILTER_LABEL).clicked() {
                    self.tuning_panel = Some(TuningPanel::Ekf);
                }
                if ui.button("Align").clicked() {
                    self.tuning_panel = Some(TuningPanel::Align);
                }
                if ui.button("Events").clicked() {
                    self.tuning_panel = Some(TuningPanel::RoadEvents);
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
                ui.selectable_value(&mut self.page, Page::Events, "Events");
                ui.selectable_value(&mut self.page, Page::Diagnostics, "Diagnostics");
            });
            {
                #[cfg(target_arch = "wasm32")]
                egui::CollapsingHeader::new("Inputs")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            draw_input_backend_selector(ui, &mut self.tuning_backend);
                        });
                        ui.horizontal_wrapped(|ui| {
                            ui.selectable_value(
                                &mut self.web_input_mode,
                                InputMode::Synthetic,
                                "Synthetic",
                            );
                            ui.selectable_value(
                                &mut self.web_input_mode,
                                InputMode::RealData,
                                "Experimental/real data",
                            );
                        });
                        ui.horizontal_wrapped(|ui| {
                            ui.label(match self.web_input_mode {
                                InputMode::Synthetic => "Scenario:",
                                InputMode::RealData => "Input:",
                            });
                            match self.web_input_mode {
                                InputMode::Synthetic => {
                                    draw_synthetic_scenario_select(
                                        ui,
                                        "web_synthetic_scenario_select",
                                        &mut self.web_scenario,
                                    );
                                    draw_synthetic_noise_select(
                                        ui,
                                        "web_synthetic_noise_select",
                                        &mut self.web_synthetic_noise,
                                    );
                                }
                                InputMode::RealData => {
                                    let selected_text = match self.web_real_data_source {
                                        WebRealDataSource::DroppedCsv => {
                                            "Dropped CSV files".to_string()
                                        }
                                        WebRealDataSource::ManifestDataset => self
                                            .web_datasets
                                            .datasets
                                            .get(self.web_datasets.selected)
                                            .map(HostedDatasetEntry::display_label)
                                            .unwrap_or_else(|| "No manifest entries".to_string()),
                                    };
                                    egui::ComboBox::from_id_salt("web_real_data_select")
                                        .selected_text(selected_text)
                                        .show_ui(ui, |ui| {
                                            ui.label(egui::RichText::new("Local files").strong());
                                            ui.selectable_value(
                                                &mut self.web_real_data_source,
                                                WebRealDataSource::DroppedCsv,
                                                "Dropped CSV files",
                                            );
                                            let groups =
                                                ["UBX/reference datasets", "iOS recordings"];
                                            for group in groups {
                                                let mut showed_group = false;
                                                for (idx, dataset) in
                                                    self.web_datasets.datasets.iter().enumerate()
                                                {
                                                    if dataset.picker_group_label() != group {
                                                        continue;
                                                    }
                                                    if !showed_group {
                                                        ui.separator();
                                                        ui.label(egui::RichText::new(group).strong());
                                                        showed_group = true;
                                                    }
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
                                            }
                                        });
                                }
                            }

                            let run_enabled = match self.web_input_mode {
                                InputMode::Synthetic => true,
                                InputMode::RealData => match self.web_real_data_source {
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
                                InputMode::RealData if self.web_datasets.loading_dataset => {
                                    "Loading dataset..."
                                }
                                _ => "Run",
                            };
                            let run_busy = self.web_datasets.loading_dataset
                                || self.web_datasets.loading_replay;
                            if draw_run_button(
                                ui,
                                run_enabled,
                                run_busy,
                                self.web_run_progress,
                                run_text,
                            ) {
                                match self.web_input_mode {
                                    InputMode::Synthetic => self.refresh_from_web_synthetic(ctx),
                                    InputMode::RealData => match self.web_real_data_source {
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
                            InputMode::Synthetic => {}
                            InputMode::RealData => {
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
                self.draw_native_inputs(ui);
            }
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn draw_native_inputs(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Inputs")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    draw_input_backend_selector(ui, &mut self.tuning_backend);
                    ui.selectable_value(
                        &mut self.native_input_mode,
                        InputMode::Synthetic,
                        "Synthetic",
                    );
                    ui.selectable_value(
                        &mut self.native_input_mode,
                        InputMode::RealData,
                        "Experimental/real data",
                    );
                });
                ui.horizontal_wrapped(|ui| match self.native_input_mode {
                    InputMode::Synthetic => {
                        ui.label("Scenario:");
                        draw_synthetic_scenario_select(
                            ui,
                            "native_synthetic_scenario_select",
                            &mut self.native_scenario,
                        );
                        draw_synthetic_noise_select(
                            ui,
                            "native_synthetic_noise_select",
                            &mut self.native_synthetic_noise,
                        );
                    }
                    InputMode::RealData => {
                        ui.label("Input:");
                        let selected_text = match self.native_real_data_source {
                            NativeRealDataSource::CustomDirectory => {
                                "Generic replay directory".to_string()
                            }
                            NativeRealDataSource::ManifestDataset => self
                                .native_datasets
                                .get(self.native_selected_dataset)
                                .map(HostedDatasetEntry::display_label)
                                .unwrap_or_else(|| "No manifest entries".to_string()),
                        };
                        egui::ComboBox::from_id_salt("native_real_data_select")
                            .selected_text(selected_text)
                            .show_ui(ui, |ui| {
                                ui.label(egui::RichText::new("Local files").strong());
                                ui.selectable_value(
                                    &mut self.native_real_data_source,
                                    NativeRealDataSource::CustomDirectory,
                                    "Generic replay directory",
                                );
                                let groups = ["UBX/reference datasets", "iOS recordings"];
                                for group in groups {
                                    let mut showed_group = false;
                                    for (idx, dataset) in self.native_datasets.iter().enumerate() {
                                        if dataset.picker_group_label() != group {
                                            continue;
                                        }
                                        if !showed_group {
                                            ui.separator();
                                            ui.label(egui::RichText::new(group).strong());
                                            showed_group = true;
                                        }
                                        let selected = self.native_real_data_source
                                            == NativeRealDataSource::ManifestDataset
                                            && self.native_selected_dataset == idx;
                                        if ui
                                            .selectable_label(selected, dataset.display_label())
                                            .clicked()
                                        {
                                            self.native_real_data_source =
                                                NativeRealDataSource::ManifestDataset;
                                            self.native_selected_dataset = idx;
                                        }
                                    }
                                }
                            });
                        if self.native_real_data_source == NativeRealDataSource::CustomDirectory {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.native_generic_replay_dir)
                                    .desired_width(520.0)
                                    .hint_text("/path/to/generic replay directory"),
                            );
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    let run_busy = self.native_replay_task.is_some();
                    let run_enabled = match self.native_input_mode {
                        InputMode::Synthetic => true,
                        InputMode::RealData => match self.native_real_data_source {
                            NativeRealDataSource::CustomDirectory => {
                                !self.native_generic_replay_dir.trim().is_empty()
                            }
                            NativeRealDataSource::ManifestDataset => {
                                !self.native_datasets.is_empty()
                            }
                        },
                    };
                    let run_text = if run_busy { "Running replay..." } else { "Run" };
                    if draw_run_button(
                        ui,
                        run_enabled && !run_busy,
                        run_busy,
                        self.native_run_progress,
                        run_text,
                    ) {
                        self.run_native_selected_input(ui.ctx());
                    }
                    if self.native_input_mode == InputMode::RealData
                        && self.native_real_data_source == NativeRealDataSource::ManifestDataset
                    {
                        if self.native_datasets.is_empty() {
                            ui.label(format!("no manifest entries in {DATASET_MANIFEST_PATH}"));
                        } else if let Some(dataset) =
                            self.native_datasets.get(self.native_selected_dataset)
                            && let Some(description) = dataset.description.as_deref()
                        {
                            ui.label(description);
                        }
                    }
                    if let Some(status) = &self.replay_status {
                        ui.label(status);
                    }
                });
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

fn draw_input_backend_selector(ui: &mut egui::Ui, backend: &mut VisualizerFusionBackend) {
    ui.label("Backend:");
    ui.selectable_value(backend, VisualizerFusionBackend::Rust, "Rust");
    #[cfg(not(target_arch = "wasm32"))]
    ui.selectable_value(backend, VisualizerFusionBackend::C, "C");
    #[cfg(target_arch = "wasm32")]
    ui.add_enabled(false, egui::Button::new("C"));
    ui.separator();
}

fn draw_synthetic_scenario_select(
    ui: &mut egui::Ui,
    id: &'static str,
    scenario: &mut SyntheticScenario,
) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(scenario.display_label())
        .show_ui(ui, |ui| {
            for candidate in SYNTHETIC_SCENARIOS {
                ui.selectable_value(scenario, *candidate, candidate.display_label());
            }
        });
}

fn draw_synthetic_noise_select(ui: &mut egui::Ui, id: &'static str, noise: &mut SyntheticNoise) {
    let noise_label = ui.add(
        egui::Label::new(egui::RichText::new("Noise:").underline()).sense(egui::Sense::hover()),
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
    egui::ComboBox::from_id_salt(id)
        .selected_text(noise.display_label())
        .show_ui(ui, |ui| {
            for candidate in SYNTHETIC_NOISE_PRESETS {
                ui.selectable_value(noise, *candidate, candidate.display_label())
                    .on_hover_text(candidate.tooltip());
            }
        });
}
