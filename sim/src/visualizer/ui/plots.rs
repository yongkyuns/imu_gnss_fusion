//! Reusable egui plot sections, overview tiles, decimation, shared cursor, and log axis handling.

use eframe::egui;
use egui_plot::{
    GridInput, GridMark, Legend, Line, LineStyle, Plot, PlotPoint, PlotPoints, Points, VLine,
    log_grid_spacer,
};

use crate::visualizer::model::Trace;

use super::colors::shared_cursor_color;
use super::orthogonal::{OrthogonalViewKind, draw_orthogonal_views_popup};
use super::state::{TraceVisibility, display_filter_trace_name};
use super::trace_query::trace_time_range;

const TIME_SERIES_PLOT_HEIGHT: f32 = 200.0;
const PLOT_GRID_MIN_SPACING_PX: f32 = 14.0;
const PLOT_GRID_MAX_SPACING_PX: f32 = 360.0;
const PLOT_GRID_STRENGTH_SCALE: f64 = 0.45;
#[cfg(target_arch = "wasm32")]
const TIME_SERIES_PLOT_FRAME_PADDING: f32 = 26.0;

pub(super) struct PlotSpec<'a> {
    title: &'static str,
    traces: Vec<&'a Trace>,
    show_legend: bool,
    y_axis: PlotYAxis,
    interaction: PlotInteraction,
}

pub(super) struct PlotSection<'a> {
    title: Option<&'static str>,
    default_open: bool,
    plots: Vec<PlotSpec<'a>>,
}

#[derive(Clone, Copy)]
pub(super) enum PlotInteraction {
    None,
    OrthogonalPopup {
        title: &'static str,
        kind: OrthogonalViewKind,
    },
}

#[derive(Clone, Copy)]
enum PlotYAxis {
    Linear,
    Log10 {
        floor: f64,
        unit: Option<&'static str>,
    },
}

impl<'a> PlotSpec<'a> {
    pub(super) fn with_interaction(mut self, interaction: PlotInteraction) -> Self {
        self.interaction = interaction;
        self
    }

    pub(super) fn with_log_y(mut self, floor: f64, unit: Option<&'static str>) -> Self {
        self.y_axis = PlotYAxis::Log10 { floor, unit };
        self
    }
}

pub(super) fn plot_section<'a>(
    title: &'static str,
    default_open: bool,
    plots: Vec<PlotSpec<'a>>,
) -> PlotSection<'a> {
    PlotSection {
        title: Some(title),
        default_open,
        plots,
    }
}

pub(super) fn overview_tile_height(width: f32) -> f32 {
    #[cfg(not(target_arch = "wasm32"))]
    let _ = width;
    #[cfg(target_arch = "wasm32")]
    {
        responsive_plot_height(width)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        TIME_SERIES_PLOT_HEIGHT
    }
}

pub(super) fn subtle_plot_grid_spacing() -> egui::emath::Rangef {
    egui::emath::Rangef::new(PLOT_GRID_MIN_SPACING_PX, PLOT_GRID_MAX_SPACING_PX)
}

pub(super) fn subdued_plot_grid_marks(input: GridInput) -> Vec<GridMark> {
    log_grid_spacer(10)(input)
        .into_iter()
        .map(|mut mark| {
            mark.step_size *= PLOT_GRID_STRENGTH_SCALE;
            mark
        })
        .collect()
}

pub(super) fn plot_spec<'a>(
    title: &'static str,
    traces: Vec<&'a Trace>,
    show_legend: bool,
) -> PlotSpec<'a> {
    PlotSpec {
        title,
        traces,
        show_legend,
        y_axis: PlotYAxis::Linear,
        interaction: PlotInteraction::None,
    }
}

pub(super) fn draw_plot_spec_with_cursor_time(
    ui: &mut egui::Ui,
    spec: &PlotSpec<'_>,
    max_points_per_trace: usize,
    cursor_t_s: Option<f64>,
) -> Option<f64> {
    draw_plot_spec_with_title_label(ui, spec, max_points_per_trace, cursor_t_s, true)
}

pub(super) fn draw_overview_plot_spec(
    ui: &mut egui::Ui,
    spec: &PlotSpec<'_>,
    max_points_per_trace: usize,
    cursor_t_s: Option<f64>,
) -> Option<f64> {
    let mut hovered_t_s = None;
    egui::CollapsingHeader::new(spec.title)
        .default_open(true)
        .show(ui, |ui| {
            hovered_t_s =
                draw_plot_spec_with_title_label(ui, spec, max_points_per_trace, cursor_t_s, false);
        });
    hovered_t_s
}

pub(super) fn page_header(ui: &mut egui::Ui, title: &str, _subtitle: &str) {
    ui.heading(title);
    ui.add_space(6.0);
}

pub(super) fn draw_analysis_sections_page(
    ui: &mut egui::Ui,
    title: &str,
    subtitle: &str,
    sections: Vec<PlotSection<'_>>,
    max_points_per_trace: usize,
    visibility: TraceVisibility,
    cursor_t_s: Option<f64>,
) -> Option<f64> {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let mut hovered_t_s = None;
            page_header(ui, title, subtitle);
            let sections: Vec<PlotSection<'_>> = sections
                .into_iter()
                .filter_map(|mut section| {
                    section.plots = filter_visible_plots(section.plots, visibility);
                    (!section.plots.is_empty()).then_some(section)
                })
                .collect();
            if sections.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("No data").weak());
                });
                return None;
            }

            for section in sections {
                match section.title {
                    Some(title) => {
                        egui::CollapsingHeader::new(title)
                            .default_open(section.default_open)
                            .show(ui, |ui| {
                                if let Some(t_s) = draw_plot_grid(
                                    ui,
                                    section.plots,
                                    max_points_per_trace,
                                    hovered_t_s.or(cursor_t_s),
                                ) {
                                    hovered_t_s = Some(t_s);
                                }
                            });
                    }
                    None => {
                        if let Some(t_s) = draw_plot_grid(
                            ui,
                            section.plots,
                            max_points_per_trace,
                            hovered_t_s.or(cursor_t_s),
                        ) {
                            hovered_t_s = Some(t_s);
                        }
                    }
                }
            }
            hovered_t_s
        })
        .inner
}

fn draw_plot_spec_with_title_label(
    ui: &mut egui::Ui,
    spec: &PlotSpec<'_>,
    max_points_per_trace: usize,
    cursor_t_s: Option<f64>,
    show_title_label: bool,
) -> Option<f64> {
    let hovered_t_s = draw_plot_with_cursor_time(
        ui,
        spec.title,
        spec.traces.iter().copied(),
        spec.show_legend,
        spec.y_axis,
        show_title_label,
        max_points_per_trace,
        cursor_t_s,
    );
    if let Some(t_s) = hovered_t_s {
        draw_plot_interaction(ui, spec, t_s);
    }
    hovered_t_s
}

fn draw_plot_interaction(ui: &mut egui::Ui, spec: &PlotSpec<'_>, t_s: f64) {
    match spec.interaction {
        PlotInteraction::None => {}
        PlotInteraction::OrthogonalPopup { title, kind } => {
            draw_orthogonal_views_popup(ui, title, &spec.traces, t_s, kind);
        }
    }
}

fn filter_visible_plots<'a>(
    plots: Vec<PlotSpec<'a>>,
    visibility: TraceVisibility,
) -> Vec<PlotSpec<'a>> {
    plots
        .into_iter()
        .map(|mut plot| {
            plot.traces
                .retain(|trace| visibility.allows_in_plot(plot.title, trace));
            plot
        })
        .filter(|plot| plot.traces.iter().any(|trace| !trace.points.is_empty()))
        .collect()
}

fn draw_plot_grid(
    ui: &mut egui::Ui,
    plots: Vec<PlotSpec<'_>>,
    max_points_per_trace: usize,
    cursor_t_s: Option<f64>,
) -> Option<f64> {
    let mut hovered_t_s = None;
    let min_plot_width = 560.0;
    let column_count = ((ui.available_width() / min_plot_width).floor() as usize).clamp(1, 2);
    ui.columns(column_count, |cols| {
        for (idx, spec) in plots.into_iter().enumerate() {
            let column = idx % column_count;
            if let Some(t_s) = draw_plot_spec_with_cursor_time(
                &mut cols[column],
                &spec,
                max_points_per_trace,
                hovered_t_s.or(cursor_t_s),
            ) {
                hovered_t_s = Some(t_s);
            }
        }
    });
    hovered_t_s
}

#[cfg(target_arch = "wasm32")]
fn responsive_plot_height(width: f32) -> f32 {
    if width < 420.0 {
        160.0
    } else if width < 560.0 {
        180.0
    } else {
        TIME_SERIES_PLOT_HEIGHT
    }
}

fn trace_value_range_for_axis<'a>(
    traces: impl IntoIterator<Item = &'a Trace>,
    y_axis: PlotYAxis,
) -> Option<(f64, f64)> {
    traces
        .into_iter()
        .flat_map(|trace| trace.points.iter().filter_map(|p| display_y(p[1], y_axis)))
        .fold(None, |range, value| match range {
            Some((min_y, max_y)) => Some((f64::min(min_y, value), f64::max(max_y, value))),
            None => Some((value, value)),
        })
}

fn transform_points_for_axis(points: Vec<[f64; 2]>, y_axis: PlotYAxis) -> Vec<[f64; 2]> {
    points
        .into_iter()
        .map(|p| {
            if !p[0].is_finite() {
                return p;
            }
            [p[0], display_y(p[1], y_axis).unwrap_or(f64::NAN)]
        })
        .collect()
}

fn display_y(value: f64, y_axis: PlotYAxis) -> Option<f64> {
    if !value.is_finite() {
        return None;
    }
    match y_axis {
        PlotYAxis::Linear => Some(value),
        PlotYAxis::Log10 { floor, .. } => Some(value.abs().max(floor).log10()),
    }
}

fn format_log_axis_value(log_value: f64, unit: Option<&str>) -> String {
    if !log_value.is_finite() {
        return String::new();
    }
    let value = 10.0_f64.powf(log_value);
    let number = format_physical_value(value);
    match unit {
        Some(unit) => format!("{number} {unit}"),
        None => number,
    }
}

fn format_plot_hover_label(name: &str, value: &PlotPoint, y_axis: PlotYAxis) -> String {
    let y = match y_axis {
        PlotYAxis::Linear => format_physical_value(value.y),
        PlotYAxis::Log10 { unit, .. } => format_log_axis_value(value.y, unit),
    };
    format!("{name}\nx = {:.1}\ny = {y}", value.x)
}

fn format_physical_value(value: f64) -> String {
    let abs = value.abs();
    if abs == 0.0 {
        "0".to_owned()
    } else if !(1.0e-2..1.0e4).contains(&abs) {
        format!("{value:.1e}")
    } else if abs >= 100.0 {
        format!("{value:.0}")
    } else if abs >= 10.0 {
        format!("{value:.1}")
    } else if abs >= 1.0 {
        format!("{value:.2}")
    } else {
        format!("{value:.3}")
    }
}

fn expand_degenerate_range((min, max): (f64, f64)) -> (f64, f64) {
    if min < max {
        (min, max)
    } else {
        let pad = min.abs().max(1.0) * 0.01;
        (min - pad, max + pad)
    }
}

fn time_in_range(t_s: f64, range: Option<(f64, f64)>) -> bool {
    let Some((min_t, max_t)) = range else {
        return false;
    };
    t_s.is_finite() && t_s >= min_t && t_s <= max_t
}

#[allow(clippy::too_many_arguments)]
fn draw_plot_with_cursor_time<'a, I>(
    ui: &mut egui::Ui,
    title: &str,
    traces: I,
    show_legend: bool,
    y_axis: PlotYAxis,
    show_title_label: bool,
    max_points_per_trace: usize,
    cursor_t_s: Option<f64>,
) -> Option<f64>
where
    I: IntoIterator<Item = &'a Trace>,
{
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

            let buckets = (max_points / 6).max(1);
            let x0 = slice.first().map(|p| p[0]).unwrap_or(0.0);
            let x1 = slice.last().map(|p| p[0]).unwrap_or(0.0);
            let span = (x1 - x0).abs();
            if span <= f64::EPSILON {
                let step = ((slice.len() as f64) / (max_points as f64)).ceil() as usize;
                return slice.iter().step_by(step.max(1)).copied().collect();
            }

            let scan_step = 1usize;

            let mut first_b: Vec<Option<usize>> = vec![None; buckets];
            let mut last_b: Vec<Option<usize>> = vec![None; buckets];
            let mut min_b: Vec<Option<(usize, f64)>> = vec![None; buckets];
            let mut max_b: Vec<Option<(usize, f64)>> = vec![None; buckets];
            let mut visit = |i: usize, p: &[f64; 2]| {
                if !p[0].is_finite() || !p[1].is_finite() {
                    return;
                }
                let mut b = (((p[0] - x0) / span) * buckets as f64).floor() as usize;
                if b >= buckets {
                    b = buckets - 1;
                }
                first_b[b].get_or_insert(i);
                last_b[b] = Some(i);
                match min_b[b] {
                    Some((_, y)) if p[1] >= y => {}
                    _ => min_b[b] = Some((i, p[1])),
                }
                match max_b[b] {
                    Some((_, y)) if p[1] <= y => {}
                    _ => max_b[b] = Some((i, p[1])),
                }
            };
            for (i, p) in slice.iter().enumerate().step_by(scan_step) {
                visit(i, p);
            }
            if scan_step > 1
                && let Some((i, p)) = slice.len().checked_sub(1).map(|i| (i, &slice[i]))
            {
                visit(i, p);
            }

            let mut out = Vec::with_capacity(max_points);
            let mut last_idx: Option<usize> = None;
            for b in 0..buckets {
                let mut indices = [
                    first_b[b],
                    first_b[b]
                        .zip(last_b[b])
                        .map(|(first, last)| first + (last - first) / 3),
                    min_b[b].map(|(i, _)| i),
                    max_b[b].map(|(i, _)| i),
                    first_b[b]
                        .zip(last_b[b])
                        .map(|(first, last)| first + 2 * (last - first) / 3),
                    last_b[b],
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
                indices.sort_unstable();
                indices.dedup();
                for idx in indices {
                    if last_idx != Some(idx) {
                        out.push(slice[idx]);
                        last_idx = Some(idx);
                    }
                    if out.len() >= max_points {
                        break;
                    }
                }
                if out.len() >= max_points {
                    break;
                }
            }

            if out.is_empty() {
                let step = ((slice.len() as f64) / (max_points as f64)).ceil() as usize;
                return slice
                    .iter()
                    .step_by(step.max(1))
                    .copied()
                    .filter(|p| p[0].is_finite() && p[1].is_finite())
                    .collect();
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

    fn finite_line_segments(mut points: Vec<[f64; 2]>) -> Vec<Vec<[f64; 2]>> {
        if points.is_empty() {
            return Vec::new();
        }

        let mut segments = Vec::new();
        let mut segment = Vec::new();
        for point in points.drain(..) {
            if point[0].is_finite() && point[1].is_finite() {
                segment.push(point);
            } else if !segment.is_empty() {
                segments.push(clean_finite_line_segment(segment));
                segment = Vec::new();
            }
        }
        if !segment.is_empty() {
            segments.push(clean_finite_line_segment(segment));
        }
        segments
            .into_iter()
            .filter(|segment| !segment.is_empty())
            .collect()
    }

    fn clean_finite_line_segment(mut points: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
        points.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
        let mut out: Vec<[f64; 2]> = Vec::with_capacity(points.len());
        for point in points {
            match out.last_mut() {
                Some(last) if last[0] == point[0] => {
                    *last = point;
                }
                _ => out.push(point),
            }
        }
        out
    }

    #[cfg(target_arch = "wasm32")]
    {
        let plot_height = responsive_plot_height(ui.available_width());
        let desired_size = egui::vec2(
            ui.available_width(),
            plot_height + TIME_SERIES_PLOT_FRAME_PADDING,
        );
        let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
        if !ui.is_rect_visible(rect) {
            return None;
        }
        return ui
            .scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                draw_plot_body(
                    ui,
                    title,
                    traces,
                    show_legend,
                    y_axis,
                    show_title_label,
                    max_points_per_trace,
                    plot_height,
                    cursor_t_s,
                )
            })
            .inner;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        return ui
            .vertical(|ui| {
                draw_plot_body(
                    ui,
                    title,
                    traces,
                    show_legend,
                    y_axis,
                    show_title_label,
                    max_points_per_trace,
                    TIME_SERIES_PLOT_HEIGHT,
                    cursor_t_s,
                )
            })
            .inner;
    }

    #[allow(clippy::too_many_arguments, unreachable_code)]
    fn draw_plot_body<'a, I>(
        ui: &mut egui::Ui,
        title: &str,
        traces: I,
        show_legend: bool,
        y_axis: PlotYAxis,
        show_title_label: bool,
        max_points_per_trace: usize,
        plot_height: f32,
        cursor_t_s: Option<f64>,
    ) -> Option<f64>
    where
        I: IntoIterator<Item = &'a Trace>,
    {
        let traces: Vec<&Trace> = traces.into_iter().collect();
        egui::Frame::group(ui.style())
            .fill(ui.visuals().window_fill)
            .inner_margin(egui::Margin::same(8))
            .corner_radius(egui::CornerRadius::same(8))
            .show(ui, |ui| {
                if show_title_label {
                    ui.label(egui::RichText::new(title).strong());
                }
                if traces.is_empty() {
                    ui.allocate_ui(egui::vec2(ui.available_width(), plot_height), |ui| {
                        ui.centered_and_justified(|ui| {
                            ui.label(egui::RichText::new("No data").weak());
                        });
                    });
                    return None;
                }
                let data_time_range = trace_time_range(traces.iter().copied());
                let data_value_range = trace_value_range_for_axis(traces.iter().copied(), y_axis);

                let mut plot = Plot::new(title)
                    .height(plot_height)
                    .grid_spacing(subtle_plot_grid_spacing())
                    .x_grid_spacer(subdued_plot_grid_marks)
                    .y_grid_spacer(subdued_plot_grid_marks)
                    .link_axis("shared_x", egui::Vec2b::new(true, false))
                    .show_x(false)
                    .show_y(false)
                    .x_axis_formatter(|mark, _range| format!("{:.1}", mark.value))
                    .allow_drag(true)
                    .allow_zoom(true)
                    .allow_scroll(true)
                    .allow_boxed_zoom(true)
                    .allow_axis_zoom_drag(true);
                if let PlotYAxis::Log10 { unit, .. } = y_axis {
                    plot = plot.y_axis_formatter(move |mark, _range| {
                        format_log_axis_value(mark.value, unit)
                    });
                }
                if let Some(range) = data_time_range.map(expand_degenerate_range) {
                    plot = plot.include_x(range.0).include_x(range.1);
                }
                if let Some(range) = data_value_range.map(expand_degenerate_range) {
                    plot = plot.include_y(range.0).include_y(range.1);
                }
                if show_legend {
                    plot = plot.legend(Legend::default());
                }
                let shared_cursor_color = shared_cursor_color(ui.visuals());
                plot.show(ui, |plot_ui| {
                    let bounds = plot_ui.plot_bounds();
                    let xmin = bounds.min()[0];
                    let xmax = bounds.max()[0];
                    let pointer_pos = plot_ui
                        .response()
                        .hovered()
                        .then(|| plot_ui.ctx().input(|i| i.pointer.latest_pos()))
                        .flatten();
                    let max_hover_dist_sq =
                        plot_ui.ctx().style().interaction.interact_radius.powi(2);
                    let mut nearest_hover: Option<(String, PlotPoint, f32)> = None;
                    for t in &traces {
                        if t.points.is_empty() {
                            continue;
                        }
                        let reduced =
                            visible_decimated(&t.points, xmin, xmax, max_points_per_trace);
                        let reduced = transform_points_for_axis(reduced, y_axis);
                        if reduced.is_empty() {
                            continue;
                        }
                        if let Some(pointer_pos) = pointer_pos {
                            let name = display_filter_trace_name(&t.name);
                            for p in reduced
                                .iter()
                                .filter(|p| p[0].is_finite() && p[1].is_finite())
                            {
                                let plot_point = PlotPoint::new(p[0], p[1]);
                                let screen_pos = plot_ui.screen_from_plot(plot_point);
                                let dist_sq = screen_pos.distance_sq(pointer_pos);
                                if dist_sq <= max_hover_dist_sq
                                    && nearest_hover
                                        .as_ref()
                                        .is_none_or(|(_, _, best_dist_sq)| dist_sq < *best_dist_sq)
                                {
                                    nearest_hover = Some((name.clone(), plot_point, dist_sq));
                                }
                            }
                        }
                        if t.name == "yaw initialized" {
                            let points: PlotPoints<'_> = reduced.into();
                            plot_ui.points(
                                Points::new(display_filter_trace_name(&t.name), points).radius(4.0),
                            );
                        } else {
                            for segment in finite_line_segments(reduced) {
                                let points: PlotPoints<'_> = segment.into();
                                plot_ui.line(Line::new(display_filter_trace_name(&t.name), points));
                            }
                        }
                    }
                    if let Some((name, point, _)) = nearest_hover {
                        plot_ui.points(
                            Points::new("", vec![[point.x, point.y]])
                                .radius(3.5)
                                .color(shared_cursor_color)
                                .allow_hover(false),
                        );
                        egui::Tooltip::always_open(
                            plot_ui.ctx().clone(),
                            plot_ui.response().layer_id,
                            plot_ui.response().id.with("trace-hover"),
                            egui::PopupAnchor::Pointer,
                        )
                        .gap(12.0)
                        .show(|ui| {
                            ui.label(format_plot_hover_label(&name, &point, y_axis));
                        });
                    }
                    let hover_t_s = plot_ui
                        .response()
                        .hovered()
                        .then(|| plot_ui.pointer_coordinate().map(|p| p.x))
                        .flatten()
                        .filter(|t| time_in_range(*t, data_time_range));
                    if let Some(cursor_t_s) = hover_t_s
                        .or(cursor_t_s)
                        .filter(|t| time_in_range(*t, data_time_range))
                    {
                        plot_ui.vline(
                            VLine::new("Shared cursor", cursor_t_s)
                                .name("")
                                .allow_hover(false)
                                .color(shared_cursor_color)
                                .style(LineStyle::Dotted { spacing: 5.0 }),
                        );
                    }
                    hover_t_s
                })
                .inner
            })
            .inner
    }
}
