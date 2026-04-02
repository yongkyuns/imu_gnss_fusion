use std::time::Duration;

use anyhow::Result;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use walkers::sources::{Mapbox, MapboxStyle, OpenStreetMap};
use walkers::{HttpTiles, Map, MapMemory, Plugin, lon_lat};

use super::math::heading_endpoint;
use super::model::{HeadingSample, Page, PlotData, Trace};
use super::stats::map_center_from_traces;

const MAPBOX_ACCESS_TOKEN: &str = "pk.eyJ1IjoieW9uZ2t5dW5zODciLCJhIjoiY21tNjB5NWt6MGJmOTJzcG02MmRvN3RnYiJ9.fu_66qb1G1cgrLzAE54E0w";

#[derive(Clone)]
struct TrackOverlay {
    traces: Vec<Trace>,
    headings: Vec<HeadingSample>,
    show_heading: bool,
}

impl Plugin for TrackOverlay {
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
            let color = if tr.name == "u-blox path (lon,lat)" {
                egui::Color32::from_rgb(0, 255, 255)
            } else if tr.name == "NAV2-PVT path (GNSS-only, lon,lat)" {
                egui::Color32::from_rgb(255, 196, 0)
            } else if tr.name == "EKF path (lon,lat)" {
                egui::Color32::from_rgb(60, 200, 120)
            } else if tr.name.contains("GNSS outage") {
                egui::Color32::from_rgb(255, 80, 80)
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
            if let Some((d2, h, p)) = best {
                if d2 <= 12.0_f32 * 12.0_f32 {
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
}

pub struct App {
    data: PlotData,
    show_egui_inspection: bool,
    show_esf_meas: bool,
    has_itow: bool,
    fps_ema: f32,
    max_points_per_trace: usize,
    page: Page,
    map_tiles: HttpTiles,
    map_memory: MapMemory,
    map_center: walkers::Position,
    show_heading: bool,
    show_nav_pvt: bool,
    show_nav2_pvt: bool,
    show_ekf: bool,
    show_eskf: bool,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        #[cfg(target_os = "macos")]
        if ctx.input(|i| i.viewport().close_requested()) {
            std::process::exit(0);
        }

        ctx.request_repaint_after(Duration::from_millis(33));

        egui::TopBottomPanel::top("top_controls").show(ctx, |ui| {
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
                    ui.checkbox(&mut self.show_esf_meas, "Show ESF-MEAS (Accel)");
                    ui.checkbox(
                        &mut self.show_egui_inspection,
                        "Show egui inspection/profiler",
                    );
                });
            ui.horizontal(|ui| {
                ui.label("Page:");
                ui.selectable_value(&mut self.page, Page::Signals, "Signals");
                ui.selectable_value(&mut self.page, Page::EkfCompare, "EKF Compare");
                ui.selectable_value(&mut self.page, Page::EskfCompare, "ESKF Compare");
                ui.selectable_value(&mut self.page, Page::AlignCompare, "Align Compare");
                ui.selectable_value(&mut self.page, Page::MapDark, "Map (Dark)");
            });
        });

        let mut imu_gyro: Vec<Trace> =
            Vec::with_capacity(self.data.imu_raw_gyro.len() + self.data.imu_cal_gyro.len());
        imu_gyro.extend(self.data.imu_raw_gyro.iter().cloned());
        imu_gyro.extend(
            self.data
                .imu_cal_gyro
                .iter()
                .filter(|t| !t.name.starts_with("ESF-MEAS "))
                .cloned(),
        );

        let mut imu_accel: Vec<Trace> =
            Vec::with_capacity(self.data.imu_raw_accel.len() + self.data.imu_cal_accel.len());
        imu_accel.extend(self.data.imu_raw_accel.iter().cloned());
        imu_accel.extend(self.data.imu_cal_accel.iter().cloned());
        if !self.show_esf_meas {
            imu_accel.retain(|t| !t.name.starts_with("ESF-MEAS "));
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
                            &self.data.speed,
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "IMU Gyro ESF (RAW/CAL)",
                            &imu_gyro,
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "ESF-INS Gyro",
                            &self.data.esf_ins_gyro,
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Orientation",
                            &self.data.orientation,
                            true,
                            self.max_points_per_trace,
                        );
                    });

                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_plot(
                        ui,
                        "Signal Strength (C/N0)",
                        &self.data.sat_cn0,
                        false,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "IMU Accel ESF (RAW/CAL/MEAS)",
                        &imu_accel,
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "ESF-INS Accel",
                        &self.data.esf_ins_accel,
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "Other Signals",
                        &self.data.other,
                        true,
                        self.max_points_per_trace,
                    );
                });
            }
            Page::EkfCompare => {
                let half_width = (ctx.content_rect().width() * 0.5).max(260.0);
                let mut vehicle_gyro = self.data.esf_ins_gyro.clone();
                vehicle_gyro.extend(self.data.ekf_meas_gyro.iter().cloned());
                let mut vehicle_accel = self.data.esf_ins_accel.clone();
                vehicle_accel.extend(self.data.ekf_meas_accel.iter().cloned());
                egui::SidePanel::left("ekf_compare_left")
                    .resizable(false)
                    .exact_width(half_width)
                    .show(ctx, |ui| {
                        draw_plot(
                            ui,
                            "Velocity: EKF vs u-blox",
                            &self.data.ekf_cmp_vel,
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Euler Angles: EKF Quaternion vs NAV-ATT",
                            &self.data.ekf_cmp_att,
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "EKF Gyro Bias Estimates",
                            &self.data.ekf_bias_gyro,
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Vehicle Gyro: ESF-INS vs EKF",
                            &vehicle_gyro,
                            true,
                            self.max_points_per_trace,
                        );
                    });

                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_plot(
                        ui,
                        "EKF Accel Bias Estimates",
                        &self.data.ekf_bias_accel,
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "EKF Bias Covariance Diagonal",
                        &self.data.ekf_cov_bias,
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "EKF Covariance Diagonal (Non-bias States)",
                        &self.data.ekf_cov_nonbias,
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "Vehicle Accel: ESF-INS vs EKF",
                        &vehicle_accel,
                        true,
                        self.max_points_per_trace,
                    );
                });
            }
            Page::EskfCompare => {
                let half_width = (ctx.content_rect().width() * 0.5).max(260.0);
                let mut vehicle_gyro = self.data.esf_ins_gyro.clone();
                vehicle_gyro.extend(self.data.eskf_meas_gyro.iter().cloned());
                let mut vehicle_accel = self.data.esf_ins_accel.clone();
                vehicle_accel.extend(self.data.eskf_meas_accel.iter().cloned());
                egui::SidePanel::left("eskf_compare_left")
                    .resizable(false)
                    .exact_width(half_width)
                    .show(ctx, |ui| {
                        draw_plot(
                            ui,
                            "Velocity: ESKF vs u-blox",
                            &self.data.eskf_cmp_vel,
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Euler Angles: ESKF Quaternion vs NAV-ATT",
                            &self.data.eskf_cmp_att,
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "ESKF Gyro Bias Estimates",
                            &self.data.eskf_bias_gyro,
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Vehicle Gyro: ESF-INS vs ESKF",
                            &vehicle_gyro,
                            true,
                            self.max_points_per_trace,
                        );
                    });

                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_plot(
                        ui,
                        "ESKF Accel Bias Estimates",
                        &self.data.eskf_bias_accel,
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "ESKF Bias Covariance Diagonal",
                        &self.data.eskf_cov_bias,
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "ESKF Covariance Diagonal (Non-bias States)",
                        &self.data.eskf_cov_nonbias,
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "Vehicle Accel: ESF-INS vs ESKF",
                        &vehicle_accel,
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
                        draw_plot(
                            ui,
                            "Euler Angles: Align KF vs ESF-ALG",
                            &self.data.align_cmp_att,
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Align Window Diagnostics",
                            &self.data.align_res_vel,
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Align Axis Error vs ESF-ALG",
                            &self.data.align_axis_err,
                            true,
                            self.max_points_per_trace,
                        );
                        draw_plot(
                            ui,
                            "Final ESF-ALG vs PCA Heading",
                            &self.data.align_motion,
                            true,
                            self.max_points_per_trace,
                        );
                    });

                egui::CentralPanel::default().show(ctx, |ui| {
                    draw_plot(
                        ui,
                        "Align Roll Update Contributions",
                        &self.data.align_roll_contrib,
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "Align Pitch Update Contributions",
                        &self.data.align_pitch_contrib,
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "Align Yaw Update Contributions",
                        &self.data.align_yaw_contrib,
                        true,
                        self.max_points_per_trace,
                    );
                    draw_plot(
                        ui,
                        "Align Covariance Diagonal",
                        &self.data.align_cov,
                        true,
                        self.max_points_per_trace,
                    );
                });
            }
            Page::MapDark => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Slippy map overlay: NAV-PVT + NAV2-PVT + EKF");
                        ui.checkbox(&mut self.show_heading, "show heading");
                        ui.checkbox(&mut self.show_nav_pvt, "show NAV-PVT");
                        ui.checkbox(&mut self.show_nav2_pvt, "show NAV2-PVT");
                        ui.checkbox(&mut self.show_ekf, "show EKF");
                        ui.checkbox(&mut self.show_eskf, "show ESKF");
                        if ui.button("Recenter").clicked() {
                            self.map_memory.follow_my_position();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.colored_label(egui::Color32::from_rgb(0, 255, 255), "NAV-PVT");
                        ui.colored_label(egui::Color32::from_rgb(255, 196, 0), "NAV2-PVT");
                        ui.colored_label(egui::Color32::from_rgb(60, 200, 120), "EKF");
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 80, 80),
                            "EKF during GNSS outage",
                        );
                        ui.colored_label(egui::Color32::from_rgb(120, 170, 255), "ESKF");
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 140, 220),
                            "ESKF during GNSS outage",
                        );
                        ui.colored_label(egui::Color32::from_rgb(255, 255, 255), "EKF heading");
                    });
                    let mut map_traces = self.data.ekf_map.clone();
                    if !self.show_nav_pvt {
                        map_traces.retain(|t| t.name != "u-blox path (lon,lat)");
                    }
                    if !self.show_nav2_pvt {
                        map_traces.retain(|t| t.name != "NAV2-PVT path (GNSS-only, lon,lat)");
                    }
                    if !self.show_ekf {
                        map_traces.retain(|t| {
                            t.name != "EKF path (lon,lat)"
                                && t.name != "EKF path during GNSS outage (lon,lat)"
                        });
                    }
                    if self.show_eskf {
                        map_traces.extend(self.data.eskf_map.clone());
                    }
                    let mut headings = self.data.ekf_map_heading.clone();
                    if self.show_eskf {
                        headings.extend(self.data.eskf_map_heading.clone());
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

fn draw_plot(
    ui: &mut egui::Ui,
    title: &str,
    traces: &[Trace],
    show_legend: bool,
    max_points_per_trace: usize,
) {
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
            .height(220.0)
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

pub fn run_visualizer(data: PlotData, has_itow: bool) -> Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_maximized(true),
        ..Default::default()
    };
    eframe::run_native(
        "visualizer",
        native_options,
        Box::new(move |cc| {
            let map_center = map_center_from_traces(&data.ekf_map);
            let map_tiles = if MAPBOX_ACCESS_TOKEN.is_empty() {
                HttpTiles::new(OpenStreetMap, cc.egui_ctx.clone())
            } else {
                HttpTiles::new(
                    Mapbox {
                        style: MapboxStyle::Dark,
                        high_resolution: true,
                        access_token: MAPBOX_ACCESS_TOKEN.to_string(),
                    },
                    cc.egui_ctx.clone(),
                )
            };
            let mut map_memory = MapMemory::default();
            let _ = map_memory.set_zoom(15.0);
            Ok(Box::new(App {
                data,
                show_egui_inspection: false,
                show_esf_meas: false,
                has_itow,
                fps_ema: 0.0,
                max_points_per_trace: 2500,
                page: Page::Signals,
                map_tiles,
                map_memory,
                map_center,
                show_heading: false,
                show_nav_pvt: true,
                show_nav2_pvt: true,
                show_ekf: true,
                show_eskf: true,
            }))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;
    Ok(())
}
