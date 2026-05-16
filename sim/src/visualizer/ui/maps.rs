//! Map tile providers and trajectory overlay rendering for real and synthetic runs.

use eframe::egui;
use egui_plot::{GridInput, GridMark, Line, Plot, PlotPoints, VLine, uniform_grid_spacer};
use walkers::sources::{Attribution, Mapbox, MapboxStyle, TileSource};
use walkers::{HttpTiles, MapMemory, Plugin, TileId, lon_lat};

use crate::visualizer::math::{ecef_to_ned, heading_endpoint, lla_to_ecef};
use crate::visualizer::model::{
    HeadingSample, MapCursorSample, PlotData, RoadEventSample, RoadSegmentSample, Trace,
};
use crate::visualizer::theme::UiTheme;

use super::colors::{
    SeriesColor, cursor_marker_color, map_heading_color, map_marker_color, map_trace_color,
    marker_outline_color, tooltip_fill, tooltip_text_color,
};
use super::state::EKF_FILTER_LABEL;
use super::trace_query::sample_trace_at;

#[derive(Clone, Copy)]
enum CartoRasterStyle {
    Positron,
    DarkMatter,
}

struct CartoRasterTiles {
    style: CartoRasterStyle,
}

impl CartoRasterTiles {
    fn for_theme(theme: UiTheme) -> Self {
        let style = match theme {
            UiTheme::Light => CartoRasterStyle::Positron,
            UiTheme::Dark => CartoRasterStyle::DarkMatter,
        };
        Self { style }
    }
}

impl TileSource for CartoRasterTiles {
    fn tile_url(&self, tile_id: TileId) -> String {
        let style = match self.style {
            CartoRasterStyle::Positron => "light_all",
            CartoRasterStyle::DarkMatter => "dark_all",
        };
        let subdomain =
            ["a", "b", "c", "d"][((tile_id.x + tile_id.y + tile_id.zoom as u32) % 4) as usize];
        format!(
            "https://{subdomain}.basemaps.cartocdn.com/{style}/{}/{}/{}.png",
            tile_id.zoom, tile_id.x, tile_id.y
        )
    }

    fn attribution(&self) -> Attribution {
        Attribution {
            text: "(C) OpenStreetMap contributors, (C) CARTO",
            url: "https://carto.com/attributions",
            logo_light: None,
            logo_dark: None,
        }
    }
}

pub(super) fn map_tiles_from_token(
    token: &str,
    theme: UiTheme,
    egui_ctx: egui::Context,
) -> HttpTiles {
    if token.is_empty() {
        HttpTiles::new(CartoRasterTiles::for_theme(theme), egui_ctx)
    } else {
        let style = match theme {
            UiTheme::Light => MapboxStyle::Light,
            UiTheme::Dark => MapboxStyle::Dark,
        };
        HttpTiles::new(
            Mapbox {
                style,
                high_resolution: true,
                access_token: token.to_string(),
            },
            egui_ctx,
        )
    }
}

pub(super) fn draw_collapsible_map_tile(
    ui: &mut egui::Ui,
    title: &'static str,
    height: f32,
    add_map: impl FnOnce(&mut egui::Ui, egui::Vec2),
) {
    egui::CollapsingHeader::new(title)
        .default_open(true)
        .show(ui, |ui| {
            draw_map_tile_body(ui, title, height, false, add_map)
        });
}

fn draw_map_tile_body(
    ui: &mut egui::Ui,
    title: &str,
    height: f32,
    show_title_label: bool,
    add_map: impl FnOnce(&mut egui::Ui, egui::Vec2),
) {
    egui::Frame::group(ui.style())
        .fill(ui.visuals().window_fill)
        .inner_margin(egui::Margin::same(8))
        .corner_radius(egui::CornerRadius::same(8))
        .show(ui, |ui| {
            if show_title_label {
                ui.label(egui::RichText::new(title).strong());
            }
            let size = egui::vec2(ui.available_width(), height);
            add_map(ui, size);
        });
}

pub(super) struct TrackOverlay<'a> {
    pub(super) traces: Vec<&'a Trace>,
    pub(super) headings: Vec<&'a HeadingSample>,
    pub(super) cursor_samples: Vec<&'a MapCursorSample>,
    pub(super) road_events: Vec<&'a RoadEventSample>,
    pub(super) road_segments: Vec<&'a RoadSegmentSample>,
    pub(super) show_heading: bool,
    pub(super) cursor_t_s: Option<f64>,
}

impl Plugin for TrackOverlay<'_> {
    fn run(
        self: Box<Self>,
        ui: &mut egui::Ui,
        response: &egui::Response,
        projector: &walkers::Projector,
        map_memory: &MapMemory,
    ) {
        let map_rect = response.rect.intersect(ui.clip_rect());
        let painter = ui.painter().with_clip_rect(map_rect);
        let visuals = ui.visuals().clone();
        let projection_offset = map_rect.center() - response.rect.center();
        let projection = MapOverlayProjection::new(projector, map_rect, projection_offset);
        let point_stride = map_trace_point_stride(map_memory.zoom());
        let min_step = map_trace_min_pixel_step(map_memory.zoom());
        let min_step_sq = min_step * min_step;
        for tr in &self.traces {
            if tr.points.len() < 2 {
                continue;
            }
            let color = map_trace_color(tr.name.as_str(), &visuals);
            let mut segment = Vec::<egui::Pos2>::with_capacity(tr.points.len().min(8192));
            let mut last_drawn: Option<egui::Pos2> = None;
            let mut pending: Option<egui::Pos2> = None;
            for p in tr.points.iter().step_by(point_stride) {
                let lon = p[0];
                let lat = p[1];
                if !lon.is_finite() || !lat.is_finite() || !projection.contains(lon, lat) {
                    if segment.len() >= 2 {
                        painter.add(egui::epaint::PathShape::line(
                            segment,
                            egui::Stroke::new(2.2, color),
                        ));
                    }
                    segment = Vec::new();
                    last_drawn = None;
                    pending = None;
                    continue;
                }
                let pos = projection.project(lon, lat);
                match last_drawn {
                    None => {
                        segment.push(pos);
                        last_drawn = Some(pos);
                    }
                    Some(last) if last.distance_sq(pos) >= min_step_sq => {
                        if let Some(pending) = pending.take()
                            && last.distance_sq(pending) >= min_step_sq
                            && pending.distance_sq(pos) >= min_step_sq
                        {
                            segment.push(pending);
                        }
                        segment.push(pos);
                        last_drawn = Some(pos);
                    }
                    Some(_) => {
                        pending = Some(pos);
                    }
                }
            }
            if let (Some(last), Some(pending)) = (last_drawn, pending)
                && last.distance_sq(pending) > 0.0
            {
                segment.push(pending);
            }
            if segment.len() >= 2 {
                painter.add(egui::epaint::PathShape::line(
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
                let from = projection.project(h.lon_deg, h.lat_deg);
                let (tip_lat, tip_lon) = heading_endpoint(h.lat_deg, h.lon_deg, h.yaw_deg, 6.0);
                let to = projection.project(tip_lon, tip_lat);
                painter.line_segment(
                    [from, to],
                    egui::Stroke::new(1.8, map_heading_color(&visuals)),
                );
            }
        }

        if let Some(t_s) = self.cursor_t_s {
            draw_map_cursor_markers(
                &painter,
                projection,
                &self.traces,
                &self.cursor_samples,
                t_s,
                &visuals,
            );
        }

        draw_road_segment_overlays(
            ui,
            &painter,
            projection,
            &self.cursor_samples,
            &self.road_segments,
            &visuals,
        );
        draw_road_event_markers(&painter, projection, &self.road_events, &visuals);

        if self.show_heading
            && let Some(mouse_pos) = ui.input(|i| i.pointer.hover_pos())
            && map_rect.contains(mouse_pos)
        {
            let mut best: Option<(f32, &HeadingSample, egui::Pos2)> = None;
            let step = (self.headings.len() / 200).max(1);
            for h in self.headings.iter().step_by(step) {
                let p = projection.project(h.lon_deg, h.lat_deg);
                let d2 = p.distance_sq(mouse_pos);
                match best {
                    Some((bd2, _, _)) if d2 >= bd2 => {}
                    _ => best = Some((d2, h, p)),
                }
            }
            if let Some((d2, h, p)) = best
                && d2 <= 12.0_f32 * 12.0_f32
            {
                painter.circle_filled(p, 3.0, cursor_marker_color(&visuals));
                let label = format!("t={:.2}s", h.t_s);
                let bg_min = p + egui::vec2(8.0, -24.0);
                let bg_rect = egui::Rect::from_min_size(bg_min, egui::vec2(78.0, 18.0));
                painter.rect_filled(bg_rect, 4.0, tooltip_fill(&visuals));
                painter.text(
                    bg_min + egui::vec2(6.0, 2.0),
                    egui::Align2::LEFT_TOP,
                    label,
                    egui::FontId::monospace(12.0),
                    tooltip_text_color(&visuals),
                );
            }
        }
    }
}

fn draw_road_event_markers(
    painter: &egui::Painter,
    projection: MapOverlayProjection<'_>,
    events: &[&RoadEventSample],
    visuals: &egui::Visuals,
) {
    let color = if visuals.dark_mode {
        egui::Color32::from_rgb(255, 212, 92)
    } else {
        egui::Color32::from_rgb(190, 109, 0)
    };
    let stroke = egui::Stroke::new(1.4, marker_outline_color(visuals));
    let hover_pos = painter.ctx().input(|input| input.pointer.hover_pos());
    for event in events {
        if !projection.contains(event.lon_deg, event.lat_deg) {
            continue;
        }
        let pos = projection.project(event.lon_deg, event.lat_deg);
        painter.circle_filled(pos, 6.5, color);
        painter.circle_stroke(pos, 6.5, stroke);
        let Some(hover_pos) = hover_pos else {
            continue;
        };
        if pos.distance_sq(hover_pos) > 12.0_f32 * 12.0_f32 {
            continue;
        }
        let label = format!(
            "bump {:.0}%\nt={:.2}s\n{:.1} km/h",
            100.0 * event.confidence.clamp(0.0, 1.0),
            event.t_s,
            3.6 * event.speed_mps
        );
        let bg_min = pos + egui::vec2(8.0, -26.0);
        let bg_rect = egui::Rect::from_min_size(bg_min, egui::vec2(88.0, 54.0));
        painter.rect_filled(bg_rect, 4.0, tooltip_fill(visuals));
        painter.text(
            bg_min + egui::vec2(6.0, 2.0),
            egui::Align2::LEFT_TOP,
            label,
            egui::FontId::monospace(12.0),
            tooltip_text_color(visuals),
        );
    }
}

fn draw_road_segment_overlays(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    projection: MapOverlayProjection<'_>,
    cursor_samples: &[&MapCursorSample],
    segments: &[&RoadSegmentSample],
    visuals: &egui::Visuals,
) {
    let hover_pos = painter.ctx().input(|input| input.pointer.hover_pos());
    for segment in segments {
        let color = road_segment_color(segment.kind.as_str(), visuals);
        let mut previous = None;
        for sample in cursor_samples.iter().copied().filter(|sample| {
            sample.trace_name == "EKF path (lon,lat)"
                && sample.t_s >= segment.start_t_s
                && sample.t_s <= segment.end_t_s
        }) {
            if !projection.contains(sample.lon_deg, sample.lat_deg) {
                previous = None;
                continue;
            }
            let current = projection.project(sample.lon_deg, sample.lat_deg);
            if let Some(previous) = previous {
                painter.line_segment(
                    [previous, current],
                    egui::Stroke::new(5.0, color.linear_multiply(0.72)),
                );
            }
            previous = Some(current);
        }

        let start = sample_map_cursor_at(cursor_samples, "EKF path (lon,lat)", segment.start_t_s);
        let end = sample_map_cursor_at(cursor_samples, "EKF path (lon,lat)", segment.end_t_s);
        let start_pos = start
            .filter(|sample| projection.contains(sample.lon_deg, sample.lat_deg))
            .map(|sample| projection.project(sample.lon_deg, sample.lat_deg));
        let end_pos = end
            .filter(|sample| projection.contains(sample.lon_deg, sample.lat_deg))
            .map(|sample| projection.project(sample.lon_deg, sample.lat_deg));

        if let Some(pos) = start_pos {
            painter.circle_filled(pos, 5.5, color);
            painter.circle_stroke(
                pos,
                5.5,
                egui::Stroke::new(1.2, marker_outline_color(visuals)),
            );
        }
        if let Some(pos) = end_pos {
            painter.rect_filled(
                egui::Rect::from_center_size(pos, egui::vec2(10.0, 10.0)),
                2.0,
                color,
            );
            painter.rect_stroke(
                egui::Rect::from_center_size(pos, egui::vec2(10.0, 10.0)),
                2.0,
                egui::Stroke::new(1.2, marker_outline_color(visuals)),
                egui::StrokeKind::Outside,
            );
        }

        let Some(hover_pos) = hover_pos else {
            continue;
        };
        let hover_anchor = [start_pos, end_pos]
            .into_iter()
            .flatten()
            .find(|pos| pos.distance_sq(hover_pos) <= 13.0_f32 * 13.0_f32);
        if let Some(pos) = hover_anchor {
            let label = road_segment_tooltip_lines(segment);
            let bg_min = pos + egui::vec2(8.0, -34.0);
            show_road_segment_tooltip(ui, bg_min, &label, segment, color, visuals);
        }
    }
}

fn road_segment_tooltip_lines(segment: &RoadSegmentSample) -> Vec<String> {
    match segment.kind.as_str() {
        "reverse" => {
            return vec![
                segment.kind.clone(),
                format!("Duration: {:.1} s", segment.duration_s),
                format!(
                    "Start/end: {:.1} - {:.1} s",
                    segment.start_t_s, segment.end_t_s
                ),
                format!("Avg speed: {:.1} km/h", 3.6 * segment.mean_speed_mps),
                format!("Peak speed: {:.1} km/h", 3.6 * segment.peak_speed_mps),
            ];
        }
        "harsh acceleration" => {
            return vec![
                segment.kind.clone(),
                format!("Duration: {:.1} s", segment.duration_s),
                format!(
                    "Start/end: {:.1} - {:.1} s",
                    segment.start_t_s, segment.end_t_s
                ),
                format!("Avg accel: {:.1} m/s^2", segment.mean_accel_mps2),
                format!("Peak accel: {:.1} m/s^2", segment.peak_accel_mps2),
                format!("Delta speed: {:+.1} km/h", 3.6 * segment.delta_speed_mps),
            ];
        }
        "harsh braking" => {
            return vec![
                segment.kind.clone(),
                format!("Duration: {:.1} s", segment.duration_s),
                format!(
                    "Start/end: {:.1} - {:.1} s",
                    segment.start_t_s, segment.end_t_s
                ),
                format!("Avg decel: {:.1} m/s^2", segment.mean_accel_mps2),
                format!("Peak decel: {:.1} m/s^2", segment.peak_accel_mps2),
                format!("Delta speed: {:+.1} km/h", 3.6 * segment.delta_speed_mps),
            ];
        }
        "harsh cornering" => {
            return vec![
                segment.kind.clone(),
                format!("Duration: {:.1} s", segment.duration_s),
                format!(
                    "Start/end: {:.1} - {:.1} s",
                    segment.start_t_s, segment.end_t_s
                ),
                format!("Avg lateral: {:.1} m/s^2", segment.mean_accel_mps2),
                format!("Peak lateral: {:.1} m/s^2", segment.peak_accel_mps2),
                format!("Speed: {:.1} km/h", 3.6 * segment.mean_speed_mps),
            ];
        }
        _ => {}
    }
    vec![
        segment.kind.clone(),
        format!("Duration: {:.1} s", segment.duration_s),
        format!(
            "Start/end: {:.1} - {:.1} s",
            segment.start_t_s, segment.end_t_s
        ),
        format!("Avg inclination: {:+.1} deg", segment.mean_pitch_deg),
        format!("Peak: {:.1} deg", segment.peak_abs_pitch_deg),
        format!("Speed: {:.1} km/h", 3.6 * segment.mean_speed_mps),
    ]
}

fn show_road_segment_tooltip(
    ui: &mut egui::Ui,
    pos: egui::Pos2,
    lines: &[String],
    segment: &RoadSegmentSample,
    segment_color: egui::Color32,
    visuals: &egui::Visuals,
) {
    let id = egui::Id::new((
        "road_segment_tooltip",
        segment.kind.as_str(),
        (segment.start_t_s * 100.0).round() as i64,
        (segment.end_t_s * 100.0).round() as i64,
    ));
    egui::Area::new(id)
        .order(egui::Order::Tooltip)
        .fixed_pos(pos)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style())
                .fill(tooltip_fill(visuals))
                .inner_margin(egui::Margin::same(6))
                .corner_radius(egui::CornerRadius::same(4))
                .show(ui, |ui| {
                    ui.set_width(312.0);
                    for line in lines {
                        ui.label(
                            egui::RichText::new(line)
                                .monospace()
                                .color(tooltip_text_color(visuals)),
                        );
                    }
                    if segment
                        .trigger_traces
                        .iter()
                        .any(|trace| trace.points.iter().any(|point| point[1].is_finite()))
                    {
                        ui.add_space(4.0);
                        show_segment_trigger_plot(ui, segment, segment_color, visuals);
                    }
                });
        });
}

fn show_segment_trigger_plot(
    ui: &mut egui::Ui,
    segment: &RoadSegmentSample,
    segment_color: egui::Color32,
    visuals: &egui::Visuals,
) {
    let id = format!(
        "road_segment_trigger_plot_{}_{}",
        segment.kind,
        (segment.start_t_s * 100.0).round() as i64
    );
    let mut plot = Plot::new(id)
        .height(132.0)
        .show_x(false)
        .show_y(true)
        .grid_spacing(egui::emath::Rangef::new(10.0, 1400.0))
        .x_grid_spacer(popup_plot_grid_marks)
        .y_grid_spacer(popup_plot_grid_marks)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false);
    if let Some((x_min, x_max)) = trigger_plot_x_range(segment) {
        plot = plot.include_x(x_min).include_x(x_max);
    }
    if let Some((y_min, y_max)) = trigger_plot_y_range(segment) {
        plot = plot.include_y(y_min).include_y(y_max);
    }
    plot.show(ui, |plot_ui| {
        for (index, trace) in segment.trigger_traces.iter().enumerate() {
            let points: Vec<[f64; 2]> = trace
                .points
                .iter()
                .copied()
                .filter(|point| point[0].is_finite() && point[1].is_finite())
                .collect();
            if points.len() < 2 {
                continue;
            }
            let points: PlotPoints<'_> = points.into();
            plot_ui.line(
                Line::new(trace.name.clone(), points).color(mini_plot_trace_color(index, visuals)),
            );
        }
        plot_ui.vline(
            VLine::new("event start", segment.start_t_s)
                .name("")
                .allow_hover(false)
                .color(segment_color),
        );
        plot_ui.vline(
            VLine::new("event end", segment.end_t_s)
                .name("")
                .allow_hover(false)
                .color(segment_color),
        );
    });
    show_segment_trigger_legend(ui, segment, visuals);
}

fn popup_plot_grid_marks(input: GridInput) -> Vec<GridMark> {
    let range = (input.bounds.1 - input.bounds.0).abs();
    if !range.is_finite() || range <= f64::EPSILON {
        return Vec::new();
    }
    let major_step = nice_grid_step((range / 4.0).max(input.base_step_size));
    uniform_grid_spacer(move |_| [major_step, major_step * 0.5, major_step * 0.25])(input)
}

fn nice_grid_step(raw_step: f64) -> f64 {
    if !raw_step.is_finite() || raw_step <= f64::EPSILON {
        return 1.0;
    }
    let exponent = raw_step.log10().floor();
    let scale = 10.0_f64.powf(exponent);
    let normalized = raw_step / scale;
    let nice = if normalized <= 1.0 {
        1.0
    } else if normalized <= 2.0 {
        2.0
    } else if normalized <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice * scale
}

fn show_segment_trigger_legend(
    ui: &mut egui::Ui,
    segment: &RoadSegmentSample,
    visuals: &egui::Visuals,
) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(8.0, 2.0);
        for (index, trace) in segment.trigger_traces.iter().enumerate() {
            if !trace.points.iter().any(|point| point[1].is_finite()) {
                continue;
            }
            let color = mini_plot_trace_color(index, visuals);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 10.0), egui::Sense::hover());
            ui.painter().line_segment(
                [
                    egui::pos2(rect.left(), rect.center().y),
                    egui::pos2(rect.right(), rect.center().y),
                ],
                egui::Stroke::new(2.0, color),
            );
            ui.label(
                egui::RichText::new(trace.name.as_str())
                    .small()
                    .color(tooltip_text_color(visuals)),
            );
        }
    });
}

fn trigger_plot_x_range(segment: &RoadSegmentSample) -> Option<(f64, f64)> {
    if segment.trigger_window_start_t_s.is_finite()
        && segment.trigger_window_end_t_s.is_finite()
        && segment.trigger_window_end_t_s > segment.trigger_window_start_t_s
    {
        return Some((
            segment.trigger_window_start_t_s,
            segment.trigger_window_end_t_s,
        ));
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    for trace in &segment.trigger_traces {
        for point in &trace.points {
            if point[0].is_finite() && point[1].is_finite() {
                min_x = min_x.min(point[0]);
                max_x = max_x.max(point[0]);
            }
        }
    }
    min_x
        .is_finite()
        .then_some(expand_degenerate_range(min_x, max_x))
}

fn trigger_plot_y_range(segment: &RoadSegmentSample) -> Option<(f64, f64)> {
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for trace in &segment.trigger_traces {
        for point in &trace.points {
            if point[1].is_finite() {
                min_y = min_y.min(point[1]);
                max_y = max_y.max(point[1]);
            }
        }
    }
    min_y
        .is_finite()
        .then_some(expand_degenerate_range(min_y, max_y))
}

fn expand_degenerate_range(min: f64, max: f64) -> (f64, f64) {
    if (max - min).abs() < 1.0e-6 {
        let pad = max.abs().max(1.0) * 0.05;
        (min - pad, max + pad)
    } else {
        let pad = (max - min) * 0.08;
        (min - pad, max + pad)
    }
}

fn mini_plot_trace_color(index: usize, visuals: &egui::Visuals) -> egui::Color32 {
    match index {
        0 => SeriesColor::Ekf.resolve(visuals),
        1 => SeriesColor::Align.resolve(visuals),
        2 => SeriesColor::Reference.resolve(visuals),
        _ => map_marker_color("event", visuals),
    }
}

fn road_segment_color(kind: &str, visuals: &egui::Visuals) -> egui::Color32 {
    match kind {
        "uphill" if visuals.dark_mode => egui::Color32::from_rgb(255, 154, 68),
        "uphill" => egui::Color32::from_rgb(214, 97, 0),
        "downhill" if visuals.dark_mode => egui::Color32::from_rgb(86, 190, 255),
        "downhill" => egui::Color32::from_rgb(0, 126, 182),
        "reverse" if visuals.dark_mode => egui::Color32::from_rgb(196, 146, 255),
        "reverse" => egui::Color32::from_rgb(133, 76, 214),
        "harsh acceleration" if visuals.dark_mode => egui::Color32::from_rgb(92, 230, 128),
        "harsh acceleration" => egui::Color32::from_rgb(0, 150, 78),
        "harsh braking" if visuals.dark_mode => egui::Color32::from_rgb(255, 92, 92),
        "harsh braking" => egui::Color32::from_rgb(206, 45, 45),
        "harsh cornering" if visuals.dark_mode => egui::Color32::from_rgb(255, 116, 218),
        "harsh cornering" => egui::Color32::from_rgb(190, 54, 165),
        _ => map_marker_color(kind, visuals),
    }
}

pub(super) struct SyntheticTrajectoryTrace {
    pub(super) name: String,
    pub(super) points: Vec<[f64; 2]>,
    pub(super) color: egui::Color32,
}

pub(super) struct SyntheticCursorMarker {
    pub(super) name: String,
    pub(super) point: [f64; 2],
    pub(super) color: egui::Color32,
}

pub(super) fn synthetic_trajectory_traces(
    data: &PlotData,
    visuals: &egui::Visuals,
    show_reference: bool,
    show_gnss: bool,
    show_ekf: bool,
) -> Vec<SyntheticTrajectoryTrace> {
    let mut traces = Vec::new();
    if show_reference {
        push_position_pair_trace(
            &mut traces,
            "Reference",
            &data.ekf_cmp_pos,
            "Synthetic truth posN [m]",
            "Synthetic truth posE [m]",
            SeriesColor::Reference.resolve(visuals),
        );
    }
    if show_gnss
        && let Some(reference) = first_lonlat_trace_point(&data.ekf_map, "Synthetic truth path")
        && let Some(gnss) = data
            .ekf_map
            .iter()
            .find(|trace| trace.name.contains("Synthetic GNSS path"))
        && let Some(points) = lonlat_trace_to_local_en(gnss, reference)
    {
        traces.push(SyntheticTrajectoryTrace {
            name: "GNSS".to_string(),
            points,
            color: map_trace_color("GNSS", visuals),
        });
    }
    if show_ekf {
        push_position_pair_trace(
            &mut traces,
            EKF_FILTER_LABEL,
            &data.ekf_cmp_pos,
            "EKF posN [m]",
            "EKF posE [m]",
            map_trace_color("EKF path (lon,lat)", visuals),
        );
    }
    traces
}

pub(super) fn decimate_trajectory_points(points: &[[f64; 2]], max_points: usize) -> Vec<[f64; 2]> {
    if points.len() <= max_points || max_points < 2 {
        return points.to_vec();
    }
    let stride = points.len().div_ceil(max_points).max(1);
    let mut out: Vec<[f64; 2]> = points.iter().copied().step_by(stride).collect();
    if let Some(last) = points.last().copied()
        && out.last().copied() != Some(last)
    {
        out.push(last);
    }
    out
}

pub(super) fn synthetic_cursor_markers(
    data: &PlotData,
    visuals: &egui::Visuals,
    show_reference: bool,
    show_ekf: bool,
    t_s: f64,
) -> Vec<SyntheticCursorMarker> {
    [
        (
            show_reference,
            "Reference",
            &data.ekf_cmp_pos,
            "Synthetic truth posN [m]",
            "Synthetic truth posE [m]",
            SeriesColor::Reference.resolve(visuals),
        ),
        (
            show_ekf,
            EKF_FILTER_LABEL,
            &data.ekf_cmp_pos,
            "EKF posN [m]",
            "EKF posE [m]",
            map_marker_color("EKF path (lon,lat)", visuals),
        ),
    ]
    .into_iter()
    .filter_map(|(visible, name, traces, north_name, east_name, color)| {
        if !visible {
            return None;
        }
        synthetic_cursor_point_from_traces(traces, t_s, north_name, east_name).map(|point| {
            SyntheticCursorMarker {
                name: name.to_string(),
                point,
                color,
            }
        })
    })
    .collect()
}

fn draw_map_cursor_markers(
    painter: &egui::Painter,
    projection: MapOverlayProjection<'_>,
    traces: &[&Trace],
    samples: &[&MapCursorSample],
    t_s: f64,
    visuals: &egui::Visuals,
) {
    let mut label_origin = None;
    for trace in traces {
        let Some(sample) = sample_map_cursor_at(samples, trace.name.as_str(), t_s) else {
            continue;
        };
        if !projection.contains(sample.lon_deg, sample.lat_deg) {
            continue;
        }
        let origin = projection.project(sample.lon_deg, sample.lat_deg);
        let color = map_marker_color(trace.name.as_str(), visuals);
        let stroke = egui::Stroke::new(1.2, marker_outline_color(visuals));
        painter.circle_filled(origin, 5.0, color);
        painter.circle_stroke(origin, 5.0, stroke);
        if let Some(yaw_deg) = sample.yaw_deg {
            let yaw = (yaw_deg as f32).to_radians();
            let dir = egui::vec2(yaw.sin(), -yaw.cos());
            let tip = origin + dir * 25.0;
            let side = egui::vec2(-dir.y, dir.x);
            painter.line_segment([origin, tip], egui::Stroke::new(2.4, color));
            painter.add(egui::Shape::convex_polygon(
                vec![
                    tip,
                    tip - dir * 8.0 + side * 4.5,
                    tip - dir * 8.0 - side * 4.5,
                ],
                color,
                stroke,
            ));
        }
        label_origin.get_or_insert(origin);
    }

    if let Some(origin) = label_origin {
        let label = format!("{t_s:.2}s");
        let bg_min = origin + egui::vec2(8.0, 8.0);
        let bg_rect = egui::Rect::from_min_size(bg_min, egui::vec2(62.0, 18.0));
        painter.rect_filled(bg_rect, 4.0, tooltip_fill(visuals));
        painter.text(
            bg_min + egui::vec2(6.0, 2.0),
            egui::Align2::LEFT_TOP,
            label,
            egui::FontId::monospace(12.0),
            tooltip_text_color(visuals),
        );
    }
}

fn sample_map_cursor_at(
    samples: &[&MapCursorSample],
    trace_name: &str,
    t_s: f64,
) -> Option<MapCursorSample> {
    if !t_s.is_finite() {
        return None;
    }
    samples
        .iter()
        .filter(|sample| sample.trace_name == trace_name)
        .min_by(|a, b| {
            let da = (a.t_s - t_s).abs();
            let db = (b.t_s - t_s).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|sample| (**sample).clone())
}

fn map_trace_min_pixel_step(zoom: f64) -> f32 {
    if zoom >= 18.0 {
        0.25
    } else if zoom >= 17.0 {
        0.5
    } else if zoom >= 16.0 {
        1.0
    } else if zoom >= 15.0 {
        2.0
    } else {
        3.0
    }
}

fn map_trace_point_stride(zoom: f64) -> usize {
    if zoom >= 17.0 {
        1
    } else if zoom >= 16.0 {
        2
    } else if zoom >= 15.0 {
        4
    } else {
        8
    }
}

#[derive(Clone, Copy)]
struct MapOverlayProjection<'a> {
    projector: &'a walkers::Projector,
    view: MapViewBounds,
    offset: egui::Vec2,
}

impl<'a> MapOverlayProjection<'a> {
    fn new(
        projector: &'a walkers::Projector,
        visible_rect: egui::Rect,
        offset: egui::Vec2,
    ) -> Self {
        Self {
            projector,
            view: map_view_bounds(projector, visible_rect, offset, 0.15),
            offset,
        }
    }

    fn contains(self, lon_deg: f64, lat_deg: f64) -> bool {
        self.view.contains(lon_deg, lat_deg)
    }

    fn project(self, lon_deg: f64, lat_deg: f64) -> egui::Pos2 {
        let projected = self.projector.project(lon_lat(lon_deg, lat_deg)) + self.offset;
        egui::pos2(projected.x, projected.y)
    }
}

#[derive(Clone, Copy)]
struct MapViewBounds {
    lon_min: f64,
    lon_max: f64,
    lat_min: f64,
    lat_max: f64,
}

impl MapViewBounds {
    fn contains(self, lon: f64, lat: f64) -> bool {
        lon >= self.lon_min && lon <= self.lon_max && lat >= self.lat_min && lat <= self.lat_max
    }
}

fn map_view_bounds(
    projector: &walkers::Projector,
    rect: egui::Rect,
    projection_offset: egui::Vec2,
    margin_fraction: f32,
) -> MapViewBounds {
    let margin = egui::vec2(
        rect.width() * margin_fraction,
        rect.height() * margin_fraction,
    );
    let rect = rect.expand2(margin);
    let corners = [
        rect.left_top(),
        rect.right_top(),
        rect.left_bottom(),
        rect.right_bottom(),
    ];
    let mut lon_min = f64::INFINITY;
    let mut lon_max = f64::NEG_INFINITY;
    let mut lat_min = f64::INFINITY;
    let mut lat_max = f64::NEG_INFINITY;
    for corner in corners {
        let pos = projector.unproject((corner - projection_offset).to_vec2());
        lon_min = lon_min.min(pos.x());
        lon_max = lon_max.max(pos.x());
        lat_min = lat_min.min(pos.y());
        lat_max = lat_max.max(pos.y());
    }
    MapViewBounds {
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    }
}

fn push_position_pair_trace(
    out: &mut Vec<SyntheticTrajectoryTrace>,
    name: &str,
    traces: &[Trace],
    north_name: &str,
    east_name: &str,
    color: egui::Color32,
) {
    let (Some(north), Some(east)) = (
        traces.iter().find(|trace| trace.name == north_name),
        traces.iter().find(|trace| trace.name == east_name),
    ) else {
        return;
    };
    let points = position_pair_to_en_points(north, east);
    if points.len() >= 2 {
        out.push(SyntheticTrajectoryTrace {
            name: name.to_string(),
            points,
            color,
        });
    }
}

fn position_pair_to_en_points(north: &Trace, east: &Trace) -> Vec<[f64; 2]> {
    north
        .points
        .iter()
        .filter_map(|sample| {
            let t_s = sample[0];
            let north_m = sample[1];
            let east_m = sample_trace_at(east, t_s)?;
            (north_m.is_finite() && east_m.is_finite()).then_some([east_m, north_m])
        })
        .collect()
}

fn synthetic_cursor_point_from_traces(
    traces: &[Trace],
    t_s: f64,
    north_name: &str,
    east_name: &str,
) -> Option<[f64; 2]> {
    let north = traces.iter().find(|trace| trace.name == north_name)?;
    let east = traces.iter().find(|trace| trace.name == east_name)?;
    let north_m = sample_trace_at(north, t_s)?;
    let east_m = sample_trace_at(east, t_s)?;
    (north_m.is_finite() && east_m.is_finite()).then_some([east_m, north_m])
}

fn first_lonlat_trace_point(traces: &[Trace], name_contains: &str) -> Option<[f64; 2]> {
    traces
        .iter()
        .find(|trace| trace.name.contains(name_contains))
        .and_then(|trace| {
            trace
                .points
                .iter()
                .copied()
                .find(|point| point[0].is_finite() && point[1].is_finite())
        })
}

fn lonlat_trace_to_local_en(trace: &Trace, reference_lonlat: [f64; 2]) -> Option<Vec<[f64; 2]>> {
    let ref_lon_deg = reference_lonlat[0];
    let ref_lat_deg = reference_lonlat[1];
    if !ref_lon_deg.is_finite() || !ref_lat_deg.is_finite() {
        return None;
    }
    let ref_ecef = lla_to_ecef(ref_lat_deg, ref_lon_deg, 0.0);
    let points: Vec<[f64; 2]> = trace
        .points
        .iter()
        .filter_map(|point| {
            let lon_deg = point[0];
            let lat_deg = point[1];
            if !lon_deg.is_finite() || !lat_deg.is_finite() {
                return None;
            }
            let ecef = lla_to_ecef(lat_deg, lon_deg, 0.0);
            let ned = ecef_to_ned(ecef, ref_ecef, ref_lat_deg, ref_lon_deg);
            Some([ned[1], ned[0]])
        })
        .collect();
    (points.len() >= 2).then_some(points)
}
