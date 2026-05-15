//! Floating tuning and update-inspector windows.

use eframe::egui;

use crate::visualizer::model::VisualizerMountMode;
use crate::visualizer::pipeline::{FusionTuningConfig, GnssOutageConfig};

use super::App;
use super::inspector::{render_update_inspector_contents, update_inspector_view_model};
use super::state::TuningPanel;
use super::tuning::{draw_align_tuning, draw_ekf_tuning};
#[cfg(target_arch = "wasm32")]
use super::web::web_remember_mapbox_token;

impl App {
    pub(super) fn draw_tuning_window(&mut self, ctx: &egui::Context) {
        let Some(panel) = self.tuning_panel else {
            return;
        };
        let mut open = true;
        let mut apply_replay = false;
        let title = match panel {
            TuningPanel::Ekf => "EKF Tuning",
            TuningPanel::Align => "Align Tuning",
        };
        egui::Window::new(title)
            .open(&mut open)
            .resizable(true)
            .default_width(620.0)
            .show(ctx, |ui| {
                match panel {
                    TuningPanel::Ekf => {
                        draw_ekf_tuning(ui, &mut self.tuning_misalignment, &mut self.tuning_cfg)
                    }
                    TuningPanel::Align => draw_align_tuning(ui, &mut self.tuning_cfg),
                }
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Reset section defaults").clicked() {
                        let defaults = FusionTuningConfig::default();
                        match panel {
                            TuningPanel::Ekf => {
                                self.tuning_misalignment = VisualizerMountMode::Auto;
                                self.tuning_gnss_outages = GnssOutageConfig::default();
                                self.tuning_cfg.r_body_vel = defaults.r_body_vel;
                                self.tuning_cfg.r_body_vel_z = defaults.r_body_vel_z;
                                self.tuning_cfg.attitude_roll_pitch_init_sigma_deg =
                                    defaults.attitude_roll_pitch_init_sigma_deg;
                                self.tuning_cfg.yaw_init_sigma_deg = defaults.yaw_init_sigma_deg;
                                self.tuning_cfg.gyro_bias_init_sigma_dps =
                                    defaults.gyro_bias_init_sigma_dps;
                                self.tuning_cfg.accel_bias_init_sigma_mps2 =
                                    defaults.accel_bias_init_sigma_mps2;
                                self.tuning_cfg.mount_roll_pitch_init_sigma_deg =
                                    defaults.mount_roll_pitch_init_sigma_deg;
                                self.tuning_cfg.mount_roll_init_sigma_deg =
                                    defaults.mount_roll_init_sigma_deg;
                                self.tuning_cfg.mount_pitch_init_sigma_deg =
                                    defaults.mount_pitch_init_sigma_deg;
                                self.tuning_cfg.mount_init_sigma_deg =
                                    defaults.mount_init_sigma_deg;
                                self.tuning_cfg.r_vehicle_speed = defaults.r_vehicle_speed;
                                self.tuning_cfg.mount_align_rw_var = defaults.mount_align_rw_var;
                                self.tuning_cfg.align_handoff_delay_s =
                                    defaults.align_handoff_delay_s;
                                self.tuning_cfg.freeze_misalignment_states =
                                    defaults.freeze_misalignment_states;
                                self.tuning_cfg.mount_settle_time_s = defaults.mount_settle_time_s;
                                self.tuning_cfg.mount_settle_release_sigma_deg =
                                    defaults.mount_settle_release_sigma_deg;
                                self.tuning_cfg.mount_settle_zero_cross_covariance =
                                    defaults.mount_settle_zero_cross_covariance;
                                self.tuning_cfg.r_zero_vel = defaults.r_zero_vel;
                                self.tuning_cfg.r_stationary_accel = defaults.r_stationary_accel;
                                self.tuning_cfg.vehicle_meas_lpf_cutoff_hz =
                                    defaults.vehicle_meas_lpf_cutoff_hz;
                                self.tuning_cfg.predict_imu_lpf_cutoff_hz =
                                    defaults.predict_imu_lpf_cutoff_hz;
                                self.tuning_cfg.predict_imu_decimation =
                                    defaults.predict_imu_decimation;
                                self.tuning_cfg.yaw_init_speed_mps = defaults.yaw_init_speed_mps;
                                self.tuning_cfg.noise.ekf = defaults.noise.ekf;
                            }
                            TuningPanel::Align => {
                                self.tuning_cfg.align = defaults.align;
                            }
                        }
                    }
                    if self.replay.is_some() && ui.button("Apply replay").clicked() {
                        apply_replay = true;
                    }
                });
            });
        if !open {
            self.tuning_panel = None;
        }
        if apply_replay {
            self.refresh_from_replay();
        }
    }

    pub(super) fn draw_update_inspector_window(&mut self, ctx: &egui::Context) {
        if !self.show_update_inspector {
            return;
        }
        let mut open = true;
        egui::Window::new("Update Inspector")
            .open(&mut open)
            .resizable(true)
            .default_size(egui::vec2(820.0, 320.0))
            .min_size(egui::vec2(420.0, 160.0))
            .vscroll(true)
            .show(ctx, |ui| {
                let Some(t_s) = self.update_inspector_cursor_t_s else {
                    ui.label(egui::RichText::new("Hover a plot to inspect recent updates.").weak());
                    return;
                };
                let Some((samples, columns, max_abs)) =
                    update_inspector_view_model(&self.data, t_s)
                else {
                    ui.label(
                        egui::RichText::new(format!(
                            "No update inspector samples near t={t_s:.2}s."
                        ))
                        .weak(),
                    );
                    return;
                };
                render_update_inspector_contents(ui, &self.data, t_s, &samples, &columns, max_abs);
            });
        if !open {
            self.show_update_inspector = false;
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn draw_mapbox_token_window(&mut self, ctx: &egui::Context) {
        if !self.show_mapbox_token_window {
            return;
        }

        let mut open = true;
        let mut changed = false;
        let mut clear = false;
        egui::Window::new("Mapbox Token")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.label("Optional Mapbox access token");
                ui.label(
                    egui::RichText::new("Leave blank to use the OpenStreetMap/CARTO fallback.")
                        .weak(),
                );
                ui.add_space(6.0);
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.web_mapbox_token)
                        .desired_width(f32::INFINITY)
                        .password(true)
                        .hint_text("pk..."),
                );
                changed |= response.changed();
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button("Clear").clicked() {
                        clear = true;
                    }
                    if ui.button("Close").clicked() {
                        self.show_mapbox_token_window = false;
                    }
                });
            });

        if clear {
            self.web_mapbox_token.clear();
            changed = true;
        }
        if changed || self.web_mapbox_token != self.web_mapbox_token_applied {
            web_remember_mapbox_token(&self.web_mapbox_token);
            self.refresh_map_tiles(ctx);
            self.web_mapbox_token_applied = self.web_mapbox_token.clone();
        }
        if !open {
            self.show_mapbox_token_window = false;
        }
    }
}
