//! Visualizer page composition for overview, motion, mount, calibration, sensors, and diagnostics.

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use walkers::Map;

use crate::visualizer::model::{HeadingSample, MapCursorSample, Page, Trace};

use super::maps::{
    MapColorSource, TrackOverlay, decimate_trajectory_points, draw_collapsible_map_tile,
    map_color_series, synthetic_cursor_markers, synthetic_trajectory_traces,
};
use super::orthogonal::OrthogonalViewKind;
use super::plots::{
    PlotEventMarker, PlotEventMarkerEdge, PlotInteraction, draw_analysis_sections_page,
    draw_overview_plot_spec, overview_tile_height, page_header, plot_section, plot_spec,
    subdued_plot_grid_marks, subtle_plot_grid_spacing,
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
                let ghost_roll_attitude_error = self
                    .ghost_data
                    .as_ref()
                    .map(|data| attitude_error_traces(data, "roll"))
                    .unwrap_or_default();
                let ghost_pitch_attitude_error = self
                    .ghost_data
                    .as_ref()
                    .map(|data| attitude_error_traces(data, "pitch"))
                    .unwrap_or_default();
                let ghost_yaw_attitude_error = self
                    .ghost_data
                    .as_ref()
                    .map(|data| attitude_error_traces(data, "yaw"))
                    .unwrap_or_default();
                let mut hovered_t_s = None;
                egui::CentralPanel::default().show(ctx, |ui| {
                    let event_markers = self.timeseries_event_markers();
                    hovered_t_s = draw_analysis_sections_page(
                        ui,
                        "Motion",
                        "Vehicle-frame motion, velocity, attitude errors, and raw attitude comparisons.",
                        vec![
                            plot_section(
                                "Vehicle Motion",
                                true,
                                vec![
                                    plot_spec(
                                        "Angular Velocity",
                                        trace_refs(&self.data.vehicle_motion_gyro),
                                        true,
                                    ),
                                    plot_spec(
                                        "Gravity-compensated Acceleration",
                                        trace_refs(&self.data.vehicle_motion_accel),
                                        true,
                                    ),
                                ],
                            ),
                            plot_section(
                                "Velocity",
                                true,
                                vec![
                                    plot_spec(
                                        "North Velocity",
                                        concat_trace_refs_matching(
                                            [self.data.ekf_cmp_vel.as_slice()],
                                            &["velN", "vN "],
                                        ),
                                        true,
                                    ),
                                    plot_spec(
                                        "East Velocity",
                                        concat_trace_refs_matching(
                                            [self.data.ekf_cmp_vel.as_slice()],
                                            &["velE", "vE "],
                                        ),
                                        true,
                                    ),
                                    plot_spec(
                                        "Down Velocity",
                                        concat_trace_refs_matching(
                                            [self.data.ekf_cmp_vel.as_slice()],
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
                                    plot_spec("Roll Error", trace_refs(&roll_attitude_error), true)
                                        .with_ghost_traces(trace_refs(&ghost_roll_attitude_error)),
                                    plot_spec(
                                        "Pitch Error",
                                        trace_refs(&pitch_attitude_error),
                                        true,
                                    )
                                    .with_ghost_traces(trace_refs(&ghost_pitch_attitude_error)),
                                    plot_spec("Yaw Error", trace_refs(&yaw_attitude_error), true)
                                        .with_ghost_traces(trace_refs(&ghost_yaw_attitude_error)),
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
                                                self.data.ekf_cmp_att.as_slice(),
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
                                                self.data.ekf_cmp_att.as_slice(),
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
                                                self.data.ekf_cmp_att.as_slice(),
                                                self.data.orientation.as_slice(),
                                            ],
                                            &["yaw"],
                                        ),
                                        true,
                                    ),
                                ],
                            ),
                            plot_section(
                                "NED Position",
                                false,
                                vec![
                                    plot_spec(
                                        "North Position",
                                        concat_trace_refs_matching(
                                            [self.data.ekf_cmp_pos.as_slice()],
                                            &["posN"],
                                        ),
                                        true,
                                    ),
                                    plot_spec(
                                        "East Position",
                                        concat_trace_refs_matching(
                                            [self.data.ekf_cmp_pos.as_slice()],
                                            &["posE"],
                                        ),
                                        true,
                                    ),
                                    plot_spec(
                                        "Down Position",
                                        concat_trace_refs_matching(
                                            [self.data.ekf_cmp_pos.as_slice()],
                                            &["posD"],
                                        ),
                                        true,
                                    ),
                                ],
                            ),
                        ],
                        self.max_points_per_trace,
                        self.trace_visibility(),
                        self.shared_cursor_t_s,
                        self.ghost_data.as_ref(),
                        &event_markers,
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
                    let event_markers = self.timeseries_event_markers();
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
                                                self.data.ekf_misalignment.as_slice(),
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
                                                self.data.ekf_misalignment.as_slice(),
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
                                                self.data.ekf_misalignment.as_slice(),
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
                                                self.data.ekf_misalignment.as_slice(),
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
                                    plot_spec(
                                        "Mount Uncertainty [deg]",
                                        concat_trace_refs([
                                            self.data.ekf_mount_sigma.as_slice(),
                                            self.data.align_cov.as_slice(),
                                        ]),
                                        true,
                                    )
                                    .with_log_y(LOG_Y_FLOOR, Some("deg")),
                                ],
                            ),
                        ],
                        self.max_points_per_trace,
                        self.trace_visibility(),
                        self.shared_cursor_t_s,
                        self.ghost_data.as_ref(),
                        &event_markers,
                    );
                });
                self.shared_cursor_t_s = hovered_t_s;
                if let Some(t_s) = hovered_t_s {
                    self.update_inspector_cursor_t_s = Some(t_s);
                }
            }
            Page::Calibration => {
                let mut hovered_t_s = None;
                egui::CentralPanel::default().show(ctx, |ui| {
                    let event_markers = self.timeseries_event_markers();
                    hovered_t_s = draw_analysis_sections_page(
                        ui,
                        "Calibration",
                        "Biases and covariance diagonals.",
                        vec![
                            plot_section(
                                "Calibration States",
                                true,
                                vec![
                                    plot_spec(
                                        "Gyro Bias",
                                        concat_trace_refs([self.data.ekf_bias_gyro.as_slice()]),
                                        true,
                                    ),
                                    plot_spec(
                                        "Accel Bias",
                                        concat_trace_refs([self.data.ekf_bias_accel.as_slice()]),
                                        true,
                                    ),
                                ],
                            ),
                            plot_section(
                                "Covariance Diagonals",
                                false,
                                vec![
                                    plot_spec(
                                        "Bias Sigma",
                                        concat_trace_refs([self.data.ekf_cov_bias.as_slice()]),
                                        true,
                                    )
                                    .with_log_y(LOG_Y_FLOOR, None),
                                    plot_spec(
                                        "Non-bias Sigma",
                                        concat_trace_refs([self.data.ekf_cov_nonbias.as_slice()]),
                                        true,
                                    )
                                    .with_log_y(LOG_Y_FLOOR, None),
                                ],
                            ),
                        ],
                        self.max_points_per_trace,
                        self.trace_visibility(),
                        self.shared_cursor_t_s,
                        self.ghost_data.as_ref(),
                        &event_markers,
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
                    let event_markers = self.timeseries_event_markers();
                    hovered_t_s = draw_analysis_sections_page(
                        ui,
                        "Sensors",
                        "Raw and calibrated sensor signals.",
                        vec![plot_section(
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
                        )],
                        self.max_points_per_trace,
                        self.trace_visibility(),
                        self.shared_cursor_t_s,
                        self.ghost_data.as_ref(),
                        &event_markers,
                    );
                });
                self.shared_cursor_t_s = hovered_t_s;
                if let Some(t_s) = hovered_t_s {
                    self.update_inspector_cursor_t_s = Some(t_s);
                }
            }
            Page::Diagnostics => {
                let mut hovered_t_s = None;
                egui::CentralPanel::default().show(ctx, |ui| {
                    let event_markers = self.timeseries_event_markers();
                    hovered_t_s = draw_analysis_sections_page(
                        ui,
                        "Diagnostics",
                        "Alignment windows and update contributions.",
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
                                vec![plot_spec(
                                    "EKF Mount Correction",
                                    trace_refs(&self.data.ekf_mount_dx),
                                    true,
                                )],
                            ),
                        ],
                        self.max_points_per_trace,
                        self.trace_visibility(),
                        self.shared_cursor_t_s,
                        self.ghost_data.as_ref(),
                        &event_markers,
                    );
                });
                self.shared_cursor_t_s = hovered_t_s;
                if let Some(t_s) = hovered_t_s {
                    self.update_inspector_cursor_t_s = Some(t_s);
                }
            }
            Page::Events => {
                self.draw_trip_summary_panel(ctx);
                let bump_pitch: Vec<&Trace> = self
                    .data
                    .ekf_bump_pitch_speed
                    .iter()
                    .filter(|t| t.name.contains("pitch"))
                    .collect();
                let bump_speed: Vec<&Trace> = self
                    .data
                    .ekf_bump_pitch_speed
                    .iter()
                    .filter(|t| t.name.contains("speed"))
                    .collect();
                let bump_time: Vec<&Trace> = self.data.ekf_bump_diag.iter().collect();
                let roughness: Vec<&Trace> = self.data.ekf_road_roughness.iter().collect();
                let mut hovered_t_s = None;
                egui::CentralPanel::default().show(ctx, |ui| {
                    let event_markers = self.timeseries_event_markers();
                    hovered_t_s = draw_analysis_sections_page(
                        ui,
                        "Events",
                        "Detector signals for bumps, stationarity, and related motion events.",
                        vec![plot_section(
                            "EKF Detectors",
                            true,
                            vec![
                                plot_spec("EKF Bump Pitch", bump_pitch, true),
                                plot_spec("EKF Bump Speed", bump_speed, true),
                                plot_spec("Speed Bump Detector", bump_time, true),
                                plot_spec("Road Roughness", roughness, true),
                                plot_spec(
                                    "EKF Stationary Diagnostics",
                                    trace_refs(&self.data.ekf_stationary_diag),
                                    true,
                                ),
                            ],
                        )],
                        self.max_points_per_trace,
                        self.trace_visibility(),
                        self.shared_cursor_t_s,
                        self.ghost_data.as_ref(),
                        &event_markers,
                    );
                });
                self.shared_cursor_t_s = hovered_t_s;
                if let Some(t_s) = hovered_t_s {
                    self.update_inspector_cursor_t_s = Some(t_s);
                }
            }
        }
    }

    fn draw_trip_summary_panel(&self, ctx: &egui::Context) {
        let summary = &self.data.trip_summary;
        if summary.sample_count == 0 {
            return;
        }
        egui::TopBottomPanel::top("events_trip_summary_panel")
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    ui.strong("Trip");
                    trip_stat_label(
                        ui,
                        "Distance",
                        format!("{:.2} km", summary.distance_m / 1000.0),
                    );
                    trip_stat_label(ui, "Duration", format_duration(summary.duration_s));
                    trip_stat_label(ui, "Moving", format_duration(summary.moving_duration_s));
                    trip_stat_label(
                        ui,
                        "Mean speed",
                        format!("{:.1} km/h", 3.6 * summary.mean_speed_mps),
                    );
                    trip_stat_label(
                        ui,
                        "Peak speed",
                        format!("{:.1} km/h", 3.6 * summary.peak_speed_mps),
                    );
                    trip_stat_label(
                        ui,
                        "Reverse",
                        format!(
                            "{:.0} m / {}",
                            summary.reverse_distance_m,
                            format_duration(summary.reverse_duration_s)
                        ),
                    );
                    trip_stat_label(
                        ui,
                        "Elevation",
                        if summary.elevation_valid {
                            format!(
                                "GNSS z +{:.0} / -{:.0} m",
                                summary.elevation_gain_m, summary.elevation_loss_m
                            )
                        } else {
                            "GNSS z n/a".to_string()
                        },
                    );
                    trip_stat_label(
                        ui,
                        "Events",
                        format!(
                            "{} bump, {} hill, {} reverse, {} harsh",
                            summary.events.speed_bumps,
                            summary.events.uphill + summary.events.downhill,
                            summary.events.reverse,
                            summary.events.harsh_acceleration
                                + summary.events.harsh_braking
                                + summary.events.harsh_cornering
                        ),
                    );
                    trip_stat_label(
                        ui,
                        "Rates",
                        format!(
                            "{:.1} bump/km, {:.1} harsh/km",
                            summary.speed_bumps_per_km, summary.harsh_events_per_km
                        ),
                    );
                    trip_stat_label(
                        ui,
                        "Roughness",
                        format!(
                            "{:.2} m/s^2 {}",
                            summary.road_roughness_rms_mps2,
                            roughness_level_label(summary.road_roughness_level)
                        ),
                    );
                });
                ui.add_space(6.0);
            });
    }

    fn timeseries_event_markers(&self) -> Vec<PlotEventMarker> {
        if !self.show_events {
            return Vec::new();
        }

        let point_markers = self
            .data
            .road_events
            .iter()
            .filter(|event| {
                event.t_s.is_finite() && self.event_visibility.allows_kind(event.kind.as_str())
            })
            .map(|event| PlotEventMarker {
                kind: event.kind.clone(),
                t_s: event.t_s,
                edge: PlotEventMarkerEdge::Point,
            });
        let segment_markers = self
            .data
            .road_segments
            .iter()
            .filter(|segment| self.event_visibility.allows_kind(segment.kind.as_str()))
            .flat_map(|segment| {
                [
                    PlotEventMarker {
                        kind: segment.kind.clone(),
                        t_s: segment.start_t_s,
                        edge: PlotEventMarkerEdge::SegmentStart,
                    },
                    PlotEventMarker {
                        kind: segment.kind.clone(),
                        t_s: segment.end_t_s,
                        edge: PlotEventMarkerEdge::SegmentEnd,
                    },
                ]
            })
            .filter(|marker| marker.t_s.is_finite());

        point_markers.chain(segment_markers).collect()
    }

    fn draw_map_body(&mut self, ui: &mut egui::Ui, size: egui::Vec2, cursor_t_s: Option<f64>) {
        if self.data_origin == DataOrigin::Synthetic {
            self.draw_synthetic_trajectory_body(ui, size, cursor_t_s);
            return;
        }

        let mut map_traces: Vec<&Trace> = self.data.ekf_map.iter().collect();
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
        if !self.show_ekf {
            map_traces.retain(|t| !t.name.contains("EKF"));
        }
        let headings: Vec<&HeadingSample> = if self.show_ekf {
            self.data.ekf_map_heading.iter().collect()
        } else {
            Vec::new()
        };
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
        let road_events: Vec<_> = if self.show_events {
            self.data
                .road_events
                .iter()
                .filter(|event| self.event_visibility.allows_kind(event.kind.as_str()))
                .collect()
        } else {
            Vec::new()
        };
        let road_segments: Vec<_> = if self.show_events {
            self.data
                .road_segments
                .iter()
                .filter(|segment| self.event_visibility.allows_kind(segment.kind.as_str()))
                .collect()
        } else {
            Vec::new()
        };
        let map_color = self
            .show_ekf
            .then(|| map_color_series(&self.data, self.map_color_source))
            .flatten();
        let track = TrackOverlay {
            traces: map_traces,
            headings,
            cursor_samples,
            road_events,
            road_segments,
            map_color,
            show_heading: self.show_heading,
            cursor_t_s,
        };
        let _map_response = ui.add_sized(
            size,
            Map::new(
                Some(&mut self.map_tiles),
                &mut self.map_memory,
                self.map_center,
            )
            .with_plugin(track)
            .double_click_to_zoom(true),
        );
        #[cfg(target_arch = "wasm32")]
        self.draw_mapbox_token_button(ui, _map_response.rect);
        self.draw_map_color_source_control(ui, _map_response.rect);
    }

    #[cfg(target_arch = "wasm32")]
    fn draw_mapbox_token_button(&mut self, ui: &mut egui::Ui, map_rect: egui::Rect) {
        let button_size = egui::vec2(78.0, 28.0);
        if map_rect.width() < button_size.x + 16.0 || map_rect.height() < button_size.y + 16.0 {
            return;
        }
        let button_rect =
            egui::Rect::from_min_size(map_rect.right_top() + egui::vec2(-86.0, 8.0), button_size);
        if ui
            .put(button_rect, egui::Button::new("Mapbox"))
            .on_hover_text("Set optional Mapbox token")
            .clicked()
        {
            self.show_mapbox_token_window = true;
        }
    }

    fn draw_map_color_source_control(&mut self, ui: &mut egui::Ui, map_rect: egui::Rect) {
        let control_size = egui::vec2(132.0, 28.0);
        if map_rect.width() < control_size.x + 16.0 || map_rect.height() < control_size.y + 16.0 {
            return;
        }
        let top_offset = if cfg!(target_arch = "wasm32") {
            44.0
        } else {
            8.0
        };
        let control_rect = egui::Rect::from_min_size(
            map_rect.right_top() + egui::vec2(-control_size.x - 8.0, top_offset),
            control_size,
        );
        ui.scope_builder(egui::UiBuilder::new().max_rect(control_rect), |ui| {
            ui.set_min_width(control_size.x);
            ui.add_enabled_ui(self.show_ekf, |ui| {
                egui::ComboBox::from_id_salt("map_color_source_overlay")
                    .width(control_size.x)
                    .selected_text(if self.show_ekf {
                        self.map_color_source.label()
                    } else {
                        "EKF hidden"
                    })
                    .show_ui(ui, |ui| {
                        for source in MapColorSource::ALL {
                            ui.selectable_value(&mut self.map_color_source, source, source.label());
                        }
                    });
            });
        });
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
            self.show_ekf,
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
                self.show_ekf,
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
                let ghost_speed: Vec<Trace> = self
                    .ghost_data
                    .as_ref()
                    .map(vehicle_body_velocity_traces)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|trace| visibility.allows(trace))
                    .collect();
                let mount: Vec<Trace> = concat_trace_refs_matching(
                    [
                        self.data.ekf_misalignment.as_slice(),
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
                        self.data.ekf_cmp_att.as_slice(),
                        self.data.orientation.as_slice(),
                    ],
                    &["roll", "pitch", "yaw"],
                )
                .into_iter()
                .filter(|trace| visibility.allows(trace))
                .cloned()
                .collect();
                let biases: Vec<Trace> = concat_trace_refs([
                    self.data.ekf_bias_gyro.as_slice(),
                    self.data.ekf_bias_accel.as_slice(),
                ])
                .into_iter()
                .filter(|trace| visibility.allows(trace))
                .cloned()
                .collect();
                let speed_spec = plot_spec("Vehicle Speed", trace_refs(&speed), true)
                    .with_ghost_traces(trace_refs(&ghost_speed));
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
                let event_markers = self.timeseries_event_markers();
                let mut hovered_t_s = None;
                if ui.available_width() < 900.0 {
                    if let Some(t_s) = draw_overview_plot_spec(
                        ui,
                        &speed_spec,
                        self.max_points_per_trace,
                        cursor_t_s,
                        self.ghost_data.as_ref(),
                        &event_markers,
                    ) {
                        hovered_t_s = Some(t_s);
                    }
                    if let Some(t_s) = draw_overview_plot_spec(
                        ui,
                        &mount_spec,
                        self.max_points_per_trace,
                        hovered_t_s.or(cursor_t_s),
                        self.ghost_data.as_ref(),
                        &event_markers,
                    ) {
                        hovered_t_s = Some(t_s);
                    }
                    if let Some(t_s) = draw_overview_plot_spec(
                        ui,
                        &attitude_spec,
                        self.max_points_per_trace,
                        hovered_t_s.or(cursor_t_s),
                        self.ghost_data.as_ref(),
                        &event_markers,
                    ) {
                        hovered_t_s = Some(t_s);
                    }
                    if let Some(t_s) = draw_overview_plot_spec(
                        ui,
                        &biases_spec,
                        self.max_points_per_trace,
                        hovered_t_s.or(cursor_t_s),
                        self.ghost_data.as_ref(),
                        &event_markers,
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
                            self.ghost_data.as_ref(),
                            &event_markers,
                        ) {
                            hovered_t_s = Some(t_s);
                        }
                        if let Some(t_s) = draw_overview_plot_spec(
                            &mut cols[0],
                            &mount_spec,
                            self.max_points_per_trace,
                            hovered_t_s.or(cursor_t_s),
                            self.ghost_data.as_ref(),
                            &event_markers,
                        ) {
                            hovered_t_s = Some(t_s);
                        }
                        if let Some(t_s) = draw_overview_plot_spec(
                            &mut cols[0],
                            &attitude_spec,
                            self.max_points_per_trace,
                            hovered_t_s.or(cursor_t_s),
                            self.ghost_data.as_ref(),
                            &event_markers,
                        ) {
                            hovered_t_s = Some(t_s);
                        }
                        if let Some(t_s) = draw_overview_plot_spec(
                            &mut cols[0],
                            &biases_spec,
                            self.max_points_per_trace,
                            hovered_t_s.or(cursor_t_s),
                            self.ghost_data.as_ref(),
                            &event_markers,
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

fn trip_stat_label(ui: &mut egui::Ui, label: &str, value: String) {
    ui.separator();
    ui.label(egui::RichText::new(label).weak());
    ui.label(value);
}

fn roughness_level_label(level: u8) -> &'static str {
    match level {
        0 => "very smooth",
        1 => "smooth",
        2 => "light texture",
        3 => "moderate",
        4 => "rough",
        5 => "very rough",
        _ => "severe",
    }
}

fn format_duration(seconds: f64) -> String {
    if seconds >= 3600.0 {
        format!("{:.1} h", seconds / 3600.0)
    } else if seconds >= 60.0 {
        format!("{:.1} min", seconds / 60.0)
    } else {
        format!("{:.1} s", seconds)
    }
}
