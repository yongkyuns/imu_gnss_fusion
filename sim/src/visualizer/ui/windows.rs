//! Floating tuning and update-inspector windows.

use eframe::egui;

use crate::visualizer::model::VisualizerMountMode;
use crate::visualizer::pipeline::{FilterCompareConfig, GnssOutageConfig};

use super::App;
use super::inspector::{render_update_inspector_contents, update_inspector_view_model};
use super::state::TuningPanel;
use super::tuning::{draw_align_tuning, draw_full_tuning, draw_reduced_tuning};

impl App {
    pub(super) fn draw_tuning_window(&mut self, ctx: &egui::Context) {
        let Some(panel) = self.tuning_panel else {
            return;
        };
        let mut open = true;
        let mut apply_replay = false;
        let title = match panel {
            TuningPanel::Reduced => "Reduced Tuning",
            TuningPanel::Align => "Align Tuning",
            TuningPanel::Full => "Full Tuning",
        };
        egui::Window::new(title)
            .open(&mut open)
            .resizable(true)
            .default_width(620.0)
            .show(ctx, |ui| {
                match panel {
                    TuningPanel::Reduced => {
                        draw_reduced_tuning(ui, &mut self.tuning_misalignment, &mut self.tuning_cfg)
                    }
                    TuningPanel::Align => draw_align_tuning(ui, &mut self.tuning_cfg),
                    TuningPanel::Full => draw_full_tuning(ui, &mut self.tuning_cfg),
                }
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Reset section defaults").clicked() {
                        let defaults = FilterCompareConfig::default();
                        match panel {
                            TuningPanel::Reduced => {
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
                                self.tuning_cfg.noise.reduced = defaults.noise.reduced;
                            }
                            TuningPanel::Align => {
                                self.tuning_cfg.align = defaults.align;
                            }
                            TuningPanel::Full => {
                                self.tuning_cfg.noise.full = defaults.noise.full;
                                self.tuning_cfg.full_init = defaults.full_init;
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
}
