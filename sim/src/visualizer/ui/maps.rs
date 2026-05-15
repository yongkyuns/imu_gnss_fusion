//! Map tile providers and trajectory overlay rendering for real and synthetic runs.

use eframe::egui;
use walkers::sources::{Attribution, Mapbox, MapboxStyle, TileSource};
use walkers::{HttpTiles, MapMemory, Plugin, TileId, lon_lat};

use crate::visualizer::math::{ecef_to_ned, heading_endpoint, lla_to_ecef};
use crate::visualizer::model::{HeadingSample, MapCursorSample, PlotData, Trace};
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
        let visuals = ui.visuals();
        let view = map_view_bounds(projector, map_rect, 0.15);
        let point_stride = map_trace_point_stride(map_memory.zoom());
        let min_step = map_trace_min_pixel_step(map_memory.zoom());
        let min_step_sq = min_step * min_step;
        for tr in &self.traces {
            if tr.points.len() < 2 {
                continue;
            }
            let color = map_trace_color(tr.name.as_str(), visuals);
            let mut segment = Vec::<egui::Pos2>::with_capacity(tr.points.len().min(8192));
            let mut last_drawn: Option<egui::Pos2> = None;
            let mut pending: Option<egui::Pos2> = None;
            for p in tr.points.iter().step_by(point_stride) {
                let lon = p[0];
                let lat = p[1];
                if !lon.is_finite() || !lat.is_finite() || !view.contains(lon, lat) {
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
                let v = projector.project(lon_lat(lon, lat));
                let pos = egui::pos2(v.x, v.y);
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
                let from = projector.project(lon_lat(h.lon_deg, h.lat_deg));
                let (tip_lat, tip_lon) = heading_endpoint(h.lat_deg, h.lon_deg, h.yaw_deg, 6.0);
                let to = projector.project(lon_lat(tip_lon, tip_lat));
                painter.line_segment(
                    [egui::pos2(from.x, from.y), egui::pos2(to.x, to.y)],
                    egui::Stroke::new(1.8, map_heading_color(visuals)),
                );
            }
        }

        if let Some(t_s) = self.cursor_t_s {
            draw_map_cursor_markers(
                &painter,
                projector,
                view,
                &self.traces,
                &self.cursor_samples,
                t_s,
                visuals,
            );
        }

        if self.show_heading
            && let Some(mouse_pos) = ui.input(|i| i.pointer.hover_pos())
            && map_rect.contains(mouse_pos)
        {
            let mut best: Option<(f32, &HeadingSample, egui::Pos2)> = None;
            let step = (self.headings.len() / 200).max(1);
            for h in self.headings.iter().step_by(step) {
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
                painter.circle_filled(p, 3.0, cursor_marker_color(visuals));
                let label = format!("t={:.2}s", h.t_s);
                let bg_min = p + egui::vec2(8.0, -24.0);
                let bg_rect = egui::Rect::from_min_size(bg_min, egui::vec2(78.0, 18.0));
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
    projector: &walkers::Projector,
    view: MapViewBounds,
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
        if !view.contains(sample.lon_deg, sample.lat_deg) {
            continue;
        }
        let projected = projector.project(lon_lat(sample.lon_deg, sample.lat_deg));
        let origin = egui::pos2(projected.x, projected.y);
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
        let pos = projector.unproject(corner.to_vec2());
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
