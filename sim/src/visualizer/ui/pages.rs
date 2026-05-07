//! Visualizer page composition for overview, motion, mount, calibration, sensors, and diagnostics.

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use walkers::Map;

use crate::visualizer::model::{HeadingSample, MapCursorSample, Page, Trace};

use super::maps::{
    TrackOverlay, decimate_trajectory_points, draw_collapsible_map_tile, synthetic_cursor_markers,
    synthetic_trajectory_traces,
};
use super::orthogonal::OrthogonalViewKind;
use super::plots::{
    PlotInteraction, draw_analysis_sections_page, draw_overview_plot_spec, overview_tile_height,
    page_header, plot_section, plot_spec, subdued_plot_grid_marks, subtle_plot_grid_spacing,
};
use super::state::{DataOrigin, is_reference_trace_name};
use super::trace_query::{
    attitude_error_traces, concat_trace_refs, concat_trace_refs_matching, trace_refs,
    vehicle_body_velocity_traces,
};
use super::{App, LOG_Y_FLOOR, SYNTHETIC_TRAJECTORY_MAX_POINTS};

impl App {
    pub(super) fn draw_current_page(&mut self, ctx: &egui::Context) {
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
            Page::Motion => {
                let roll_attitude_error = attitude_error_traces(&self.data, "roll");
                let pitch_attitude_error = attitude_error_traces(&self.data, "pitch");
                let yaw_attitude_error = attitude_error_traces(&self.data, "yaw");
                let mut hovered_t_s = None;
                egui::CentralPanel::default().show(ctx, |ui| {
                    hovered_t_s = draw_analysis_sections_page(
                        ui,
                        "Motion",
                        "Velocity, attitude errors, and raw attitude comparisons.",
                        vec![
                            plot_section(
                                "Velocity",
                                true,
                                vec![
                                    plot_spec(
                                        "North Velocity",
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
                                        "East Velocity",
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
                                        "Down Velocity",
                                        concat_trace_refs_matching(
                                            [
                                                self.data.eskf_cmp_vel.as_slice(),
                                                self.data.loose_cmp_vel.as_slice(),
                                            ],
                                            &["velD", "vD "],
                                        ),
                                        true,
                                    ),
                                ],
                            ),
                            plot_section(
                                "Attitude Error",
                                true,
                                vec![
                                    plot_spec("Roll Error", trace_refs(&roll_attitude_error), true),
                                    plot_spec(
                                        "Pitch Error",
                                        trace_refs(&pitch_attitude_error),
                                        true,
                                    ),
                                    plot_spec("Yaw Error", trace_refs(&yaw_attitude_error), true),
                                ],
                            ),
                            plot_section(
                                "Raw Attitude",
                                false,
                                vec![
                                    plot_spec(
                                        "Roll",
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
                                        "Pitch",
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
                                        "Yaw",
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
                            ),
                        ],
                        self.max_points_per_trace,
                        self.trace_visibility(),
                        self.shared_cursor_t_s,
                    );
                });
                self.shared_cursor_t_s = hovered_t_s;
                if let Some(t_s) = hovered_t_s {
                    self.update_inspector_cursor_t_s = Some(t_s);
                }
            }
            Page::Mount => {
                let mut hovered_t_s = None;
                egui::CentralPanel::default().show(ctx, |ui| {
                    hovered_t_s = draw_analysis_sections_page(
                        ui,
                        "Mount",
                        "Mount angle estimates and alignment diagnostics.",
                        vec![
                            plot_section(
                                "Mount Estimates",
                                true,
                                vec![
                                    plot_spec(
                                        "Mount Roll",
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
                                        "Mount Pitch",
                                        concat_trace_refs_matching(
                                            [
                                                self.data.eskf_misalignment.as_slice(),
                                                self.data.loose_misalignment.as_slice(),
                                                self.data.align_cmp_att.as_slice(),
                                            ],
                                            &[
                                                "mount pitch",
                                                "Align pitch",
                                                "Reference mount pitch",
                                            ],
                                        ),
                                        true,
                                    ),
                                    plot_spec(
                                        "Mount Yaw",
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
                                        "Mount Quaternion Error [deg]",
                                        concat_trace_refs_matching(
                                            [
                                                self.data.eskf_misalignment.as_slice(),
                                                self.data.loose_misalignment.as_slice(),
                                                self.data.align_cmp_att.as_slice(),
                                            ],
                                            &["quaternion error"],
                                        ),
                                        true,
                                    )
                                    .with_log_y(LOG_Y_FLOOR, Some("deg")),
                                ],
                            ),
                            plot_section(
                                "Align Diagnostics",
                                true,
                                vec![
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
                                ],
                            ),
                        ],
                        self.max_points_per_trace,
                        self.trace_visibility(),
                        self.shared_cursor_t_s,
                    );
                });
                self.shared_cursor_t_s = hovered_t_s;
                if let Some(t_s) = hovered_t_s {
                    self.update_inspector_cursor_t_s = Some(t_s);
                }
            }
            Page::Calibration => {
                let scale: Vec<&Trace> = self
                    .data
                    .loose_scale_gyro
                    .iter()
                    .chain(self.data.loose_scale_accel.iter())
                    .collect();
                let mut hovered_t_s = None;
                egui::CentralPanel::default().show(ctx, |ui| {
                    hovered_t_s = draw_analysis_sections_page(
                        ui,
                        "Calibration",
                        "Biases, scale factors, and covariance diagonals.",
                        vec![
                            plot_section(
                                "Calibration States",
                                true,
                                vec![
                                    plot_spec(
                                        "Gyro Bias",
                                        concat_trace_refs([
                                            self.data.eskf_bias_gyro.as_slice(),
                                            self.data.loose_bias_gyro.as_slice(),
                                        ]),
                                        true,
                                    ),
                                    plot_spec(
                                        "Accel Bias",
                                        concat_trace_refs([
                                            self.data.eskf_bias_accel.as_slice(),
                                            self.data.loose_bias_accel.as_slice(),
                                        ]),
                                        true,
                                    ),
                                    plot_spec(
                                        "Mount Uncertainty [deg]",
                                        concat_trace_refs([
                                            self.data.eskf_mount_sigma.as_slice(),
                                            self.data.loose_mount_sigma.as_slice(),
                                            self.data.align_cov.as_slice(),
                                        ]),
                                        true,
                                    )
                                    .with_log_y(LOG_Y_FLOOR, Some("deg")),
                                    plot_spec("Loose Scale Factors", scale, true),
                                ],
                            ),
                            plot_section(
                                "Covariance Diagonals",
                                false,
                                vec![
                                    plot_spec(
                                        "Bias Sigma",
                                        concat_trace_refs([
                                            self.data.eskf_cov_bias.as_slice(),
                                            self.data.loose_cov_bias.as_slice(),
                                        ]),
                                        true,
                                    )
                                    .with_log_y(LOG_Y_FLOOR, None),
                                    plot_spec(
                                        "Non-bias Sigma",
                                        concat_trace_refs([
                                            self.data.eskf_cov_nonbias.as_slice(),
                                            self.data.loose_cov_nonbias.as_slice(),
                                        ]),
                                        false,
                                    )
                                    .with_log_y(LOG_Y_FLOOR, None),
                                ],
                            ),
                        ],
                        self.max_points_per_trace,
                        self.trace_visibility(),
                        self.shared_cursor_t_s,
                    );
                });
                self.shared_cursor_t_s = hovered_t_s;
                if let Some(t_s) = hovered_t_s {
                    self.update_inspector_cursor_t_s = Some(t_s);
                }
            }
            Page::Sensors => {
                let mut hovered_t_s = None;
                egui::CentralPanel::default().show(ctx, |ui| {
                    hovered_t_s = draw_analysis_sections_page(
                        ui,
                        "Sensors",
                        "Raw, calibrated, and filter input sensor signals.",
                        vec![
                            plot_section(
                                "Source Sensors",
                                true,
                                vec![
                                    plot_spec(
                                        "GNSS Signal Strength",
                                        trace_refs(&self.data.sat_cn0),
                                        false,
                                    ),
                                    plot_spec(
                                        "Raw IMU Gyro",
                                        trace_refs(&self.data.imu_raw_gyro),
                                        true,
                                    ),
                                    plot_spec(
                                        "Raw IMU Accel",
                                        trace_refs(&self.data.imu_raw_accel),
                                        true,
                                    ),
                                    plot_spec("Calibrated IMU Gyro", imu_cal_gyro, true),
                                    plot_spec("Calibrated IMU Accel", imu_cal_accel, true),
                                ],
                            ),
                            plot_section(
                                "Advanced Filter Inputs",
                                false,
                                vec![
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
                            ),
                        ],
                        self.max_points_per_trace,
                        self.trace_visibility(),
                        self.shared_cursor_t_s,
                    );
                });
                self.shared_cursor_t_s = hovered_t_s;
                if let Some(t_s) = hovered_t_s {
                    self.update_inspector_cursor_t_s = Some(t_s);
                }
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
                let mut hovered_t_s = None;
                egui::CentralPanel::default().show(ctx, |ui| {
                    hovered_t_s = draw_analysis_sections_page(
                        ui,
                        "Diagnostics",
                        "Alignment windows, bump detector signals, and update contributions.",
                        vec![
                            plot_section(
                                "Align Internals",
                                true,
                                vec![
                                    plot_spec(
                                        "Align Window Diagnostics",
                                        trace_refs(&self.data.align_res_vel),
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
                                ],
                            ),
                            plot_section(
                                "Filter Update Diagnostics",
                                true,
                                vec![
                                    plot_spec(
                                        "Loose Mount Correction",
                                        trace_refs(&self.data.loose_mount_dx),
                                        true,
                                    ),
                                    plot_spec(
                                        "Loose GNSS Position Gate",
                                        trace_refs(&self.data.loose_gnss_pos_gate),
                                        true,
                                    ),
                                    plot_spec(
                                        "ESKF Mount Correction",
                                        trace_refs(&self.data.eskf_mount_dx),
                                        true,
                                    ),
                                ],
                            ),
                            plot_section(
                                "ESKF Detectors",
                                false,
                                vec![
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
                            ),
                        ],
                        self.max_points_per_trace,
                        self.trace_visibility(),
                        self.shared_cursor_t_s,
                    );
                });
                self.shared_cursor_t_s = hovered_t_s;
                if let Some(t_s) = hovered_t_s {
                    self.update_inspector_cursor_t_s = Some(t_s);
                }
            }
        }
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
                    && !t.name.contains("GNSS-only")
                    && !t.name.contains("GNSS reference")
                    && !t.name.contains("NAV")
                    && !t.name.contains("truth")
            });
        }
        if !self.show_reference {
            map_traces.retain(|t| !is_reference_trace_name(t.name.as_str()));
        }
        if !self.show_eskf {
            map_traces.retain(|t| !t.name.contains("ESKF"));
        }
        if self.show_loose {
            map_traces.extend(self.data.loose_map.iter());
        }
        let mut headings: Vec<&HeadingSample> = if self.show_eskf {
            self.data.eskf_map_heading.iter().collect()
        } else {
            Vec::new()
        };
        if self.show_loose {
            headings.extend(self.data.loose_map_heading.iter());
        }
        let cursor_samples: Vec<&MapCursorSample> = self
            .data
            .map_cursor
            .iter()
            .filter(|sample| {
                map_traces
                    .iter()
                    .any(|trace| trace.name == sample.trace_name)
            })
            .collect();
        let track = TrackOverlay {
            traces: map_traces,
            headings,
            cursor_samples,
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
            self.show_reference,
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

        let cursor_markers = cursor_t_s.map(|t_s| {
            synthetic_cursor_markers(
                &self.data,
                ui.visuals(),
                self.show_reference,
                self.show_eskf,
                self.show_loose,
                t_s,
            )
        });
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
                if let Some(markers) = cursor_markers {
                    for marker in markers {
                        plot_ui.points(
                            Points::new(format!("{} cursor", marker.name), vec![marker.point])
                                .radius(5.0)
                                .color(marker.color),
                        );
                    }
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
                let visibility = self.trace_visibility();
                let speed: Vec<Trace> = vehicle_body_velocity_traces(&self.data)
                    .into_iter()
                    .filter(|trace| visibility.allows(trace))
                    .collect();
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
                .filter(|trace| visibility.allows(trace))
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
                .filter(|trace| visibility.allows(trace))
                .cloned()
                .collect();
                let biases: Vec<Trace> = concat_trace_refs([
                    self.data.eskf_bias_gyro.as_slice(),
                    self.data.loose_bias_gyro.as_slice(),
                    self.data.eskf_bias_accel.as_slice(),
                    self.data.loose_bias_accel.as_slice(),
                ])
                .into_iter()
                .filter(|trace| visibility.allows(trace))
                .cloned()
                .collect();
                let speed_spec = plot_spec("Vehicle Speed", trace_refs(&speed), true);
                let mount_spec = plot_spec("Mount Angles", trace_refs(&mount), true)
                    .with_interaction(PlotInteraction::OrthogonalPopup {
                        title: "Mount Alignment",
                        kind: OrthogonalViewKind::Mount,
                    });
                let attitude_spec = plot_spec("Vehicle Attitude", trace_refs(&attitude), true)
                    .with_interaction(PlotInteraction::OrthogonalPopup {
                        title: "Vehicle Attitude",
                        kind: OrthogonalViewKind::Vehicle,
                    });
                let biases_spec = plot_spec("Biases", trace_refs(&biases), true);

                let tile_height = overview_tile_height(ui.available_width());
                let cursor_t_s = self.shared_cursor_t_s;
                let mut hovered_t_s = None;
                if ui.available_width() < 900.0 {
                    if let Some(t_s) = draw_overview_plot_spec(
                        ui,
                        &speed_spec,
                        self.max_points_per_trace,
                        cursor_t_s,
                    ) {
                        hovered_t_s = Some(t_s);
                    }
                    if let Some(t_s) = draw_overview_plot_spec(
                        ui,
                        &mount_spec,
                        self.max_points_per_trace,
                        hovered_t_s.or(cursor_t_s),
                    ) {
                        hovered_t_s = Some(t_s);
                    }
                    if let Some(t_s) = draw_overview_plot_spec(
                        ui,
                        &attitude_spec,
                        self.max_points_per_trace,
                        hovered_t_s.or(cursor_t_s),
                    ) {
                        hovered_t_s = Some(t_s);
                    }
                    if let Some(t_s) = draw_overview_plot_spec(
                        ui,
                        &biases_spec,
                        self.max_points_per_trace,
                        hovered_t_s.or(cursor_t_s),
                    ) {
                        hovered_t_s = Some(t_s);
                    }
                    draw_collapsible_map_tile(ui, "Map", tile_height, |ui, size| {
                        self.draw_map_body(ui, size, hovered_t_s.or(cursor_t_s));
                    });
                } else {
                    ui.columns(2, |cols| {
                        if let Some(t_s) = draw_overview_plot_spec(
                            &mut cols[0],
                            &speed_spec,
                            self.max_points_per_trace,
                            cursor_t_s,
                        ) {
                            hovered_t_s = Some(t_s);
                        }
                        if let Some(t_s) = draw_overview_plot_spec(
                            &mut cols[0],
                            &mount_spec,
                            self.max_points_per_trace,
                            hovered_t_s.or(cursor_t_s),
                        ) {
                            hovered_t_s = Some(t_s);
                        }
                        if let Some(t_s) = draw_overview_plot_spec(
                            &mut cols[0],
                            &attitude_spec,
                            self.max_points_per_trace,
                            hovered_t_s.or(cursor_t_s),
                        ) {
                            hovered_t_s = Some(t_s);
                        }
                        if let Some(t_s) = draw_overview_plot_spec(
                            &mut cols[0],
                            &biases_spec,
                            self.max_points_per_trace,
                            hovered_t_s.or(cursor_t_s),
                        ) {
                            hovered_t_s = Some(t_s);
                        }
                        let map_height = tile_height * 2.0;
                        draw_collapsible_map_tile(&mut cols[1], "Map", map_height, |ui, size| {
                            self.draw_map_body(ui, size, hovered_t_s.or(cursor_t_s));
                        });
                    });
                }
                if self.shared_cursor_t_s != hovered_t_s {
                    self.shared_cursor_t_s = hovered_t_s;
                    ui.ctx().request_repaint();
                }
                if let Some(t_s) = hovered_t_s {
                    self.update_inspector_cursor_t_s = Some(t_s);
                }
            });
    }
}
