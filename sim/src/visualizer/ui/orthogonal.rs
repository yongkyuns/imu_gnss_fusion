//! Orthogonal vehicle and sensor attitude popups shown from attitude and mount plots.

use eframe::egui;

use crate::visualizer::model::Trace;

use super::colors::SeriesColor;
use super::trace_query::sample_trace_at;

#[derive(Clone, Copy)]
enum AttitudeAxis {
    Roll,
    Pitch,
    Yaw,
}

struct AttitudeSeriesSample {
    label: &'static str,
    color: SeriesColor,
    angle_deg: f64,
}

#[derive(Clone, Copy)]
pub(super) enum OrthogonalViewKind {
    Vehicle,
    Mount,
}

pub(super) fn draw_orthogonal_views_popup(
    ui: &mut egui::Ui,
    title: &str,
    traces: &[&Trace],
    t_s: f64,
    kind: OrthogonalViewKind,
) {
    let Some(pointer_pos) = ui.input(|i| i.pointer.hover_pos()) else {
        return;
    };
    let screen_rect = ui.ctx().content_rect();
    let popup_size = egui::vec2(430.0, 218.0);
    let mut pos = pointer_pos + egui::vec2(14.0, 14.0);
    if pos.x + popup_size.x > screen_rect.right() - 8.0 {
        pos.x = pointer_pos.x - popup_size.x - 14.0;
    }
    if pos.y + popup_size.y > screen_rect.bottom() - 8.0 {
        pos.y = pointer_pos.y - popup_size.y - 14.0;
    }
    pos.x = pos.x.max(screen_rect.left() + 8.0);
    pos.y = pos.y.max(screen_rect.top() + 8.0);

    egui::Area::new(egui::Id::new(("orthogonal_views_popup", title)))
        .order(egui::Order::Tooltip)
        .fixed_pos(pos)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style())
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    ui.set_min_size(popup_size);
                    ui.label(egui::RichText::new(format!("{title}  t={t_s:.2}s")).strong());
                    let rect = ui
                        .allocate_exact_size(
                            egui::vec2(ui.available_width(), popup_size.y - 28.0),
                            egui::Sense::hover(),
                        )
                        .0;
                    let painter = ui.painter_at(rect);
                    let gap = 8.0;
                    let w = (rect.width() - 2.0 * gap).max(1.0) / 3.0;
                    for (idx, (axis, label)) in [
                        (AttitudeAxis::Roll, "Roll"),
                        (AttitudeAxis::Pitch, "Pitch"),
                        (AttitudeAxis::Yaw, "Yaw"),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let x0 = rect.left() + idx as f32 * (w + gap);
                        let view = egui::Rect::from_min_size(
                            egui::pos2(x0, rect.top()),
                            egui::vec2(w, rect.height()),
                        );
                        let samples = angle_samples_for_axis(traces, t_s, axis, kind);
                        draw_angle_axis_view(&painter, view, label, axis, &samples, kind);
                    }
                });
        });
}

fn angle_samples_for_axis(
    traces: &[&Trace],
    t_s: f64,
    axis: AttitudeAxis,
    kind: OrthogonalViewKind,
) -> Vec<AttitudeSeriesSample> {
    let prefixes: &[(&str, &str, SeriesColor)] = match kind {
        OrthogonalViewKind::Vehicle => &[
            ("Reference", "Reference", SeriesColor::Reference),
            ("Synthetic truth", "Reference", SeriesColor::Reference),
            ("EKF", "EKF", SeriesColor::Ekf),
        ],
        OrthogonalViewKind::Mount => &[
            ("Reference mount", "Reference", SeriesColor::Reference),
            ("Synthetic truth mount", "Reference", SeriesColor::Reference),
            ("Align", "Align", SeriesColor::Align),
            ("EKF mount", "EKF", SeriesColor::Ekf),
        ],
    };
    prefixes
        .iter()
        .copied()
        .filter_map(|(prefix, label, color)| {
            find_angle_trace(traces, prefix, axis, kind).and_then(|trace| {
                sample_trace_at(trace, t_s).map(|angle_deg| AttitudeSeriesSample {
                    label,
                    color,
                    angle_deg,
                })
            })
        })
        .fold(Vec::<AttitudeSeriesSample>::new(), |mut out, sample| {
            if !out.iter().any(|existing| existing.label == sample.label) {
                out.push(sample);
            }
            out
        })
}

fn find_angle_trace<'a>(
    traces: &'a [&'a Trace],
    prefix: &str,
    axis: AttitudeAxis,
    kind: OrthogonalViewKind,
) -> Option<&'a Trace> {
    let axis_token = match axis {
        AttitudeAxis::Roll => "roll",
        AttitudeAxis::Pitch => "pitch",
        AttitudeAxis::Yaw => "yaw",
    };
    traces.iter().copied().find(|trace| {
        let name = trace.name.as_str();
        name.starts_with(prefix)
            && name.contains(axis_token)
            && match kind {
                OrthogonalViewKind::Vehicle => {
                    !name.contains("mount") && !name.contains("residual")
                }
                OrthogonalViewKind::Mount => name.contains("mount") || name.starts_with("Align"),
            }
    })
}

fn draw_angle_axis_view(
    painter: &egui::Painter,
    rect: egui::Rect,
    label: &str,
    axis: AttitudeAxis,
    samples: &[AttitudeSeriesSample],
    kind: OrthogonalViewKind,
) {
    let visuals = painter.ctx().style().visuals.clone();
    painter.rect_stroke(
        rect,
        egui::CornerRadius::same(5),
        egui::Stroke::new(1.0, orthogonal_panel_stroke(&visuals)),
        egui::StrokeKind::Inside,
    );
    painter.text(
        rect.center_top() + egui::vec2(0.0, 8.0),
        egui::Align2::CENTER_TOP,
        label,
        egui::FontId::proportional(12.0),
        visuals.text_color(),
    );
    if samples.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No sample",
            egui::FontId::proportional(11.0),
            visuals.weak_text_color(),
        );
        return;
    }

    let figure_rect = rect.shrink2(egui::vec2(8.0, 24.0));
    let scale = figure_rect.width().min(figure_rect.height()).max(1.0);
    let center = figure_rect.center() + egui::vec2(0.0, -4.0);
    draw_zero_angle_axis(painter, center, scale, axis, &visuals);
    for sample in samples {
        draw_rotated_outline(
            painter,
            center,
            scale,
            axis,
            sample.angle_deg,
            sample.color.resolve(&visuals),
            matches!(kind, OrthogonalViewKind::Vehicle),
        );
    }

    let mut y = rect.bottom() - 14.0 * samples.len() as f32 - 3.0;
    for sample in samples {
        painter.text(
            egui::pos2(rect.left() + 7.0, y),
            egui::Align2::LEFT_TOP,
            format!("{} {:+.1} deg", sample.label, sample.angle_deg),
            egui::FontId::proportional(10.5),
            sample.color.resolve(&visuals),
        );
        y += 14.0;
    }
}

fn orthogonal_panel_stroke(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_gray(58)
    } else {
        egui::Color32::from_rgb(185, 176, 164)
    }
}

fn draw_zero_angle_axis(
    painter: &egui::Painter,
    center: egui::Pos2,
    scale: f32,
    axis: AttitudeAxis,
    visuals: &egui::Visuals,
) {
    let half = 0.36 * scale;
    let dash = 4.0;
    let gap = 4.0;
    let color = zero_angle_axis_color(visuals);
    match axis {
        AttitudeAxis::Roll | AttitudeAxis::Pitch => {
            let mut x = center.x - half;
            while x < center.x + half {
                let x1 = (x + dash).min(center.x + half);
                painter.line_segment(
                    [egui::pos2(x, center.y), egui::pos2(x1, center.y)],
                    egui::Stroke::new(1.0, color),
                );
                x += dash + gap;
            }
        }
        AttitudeAxis::Yaw => {
            let mut y = center.y - half;
            while y < center.y + half {
                let y1 = (y + dash).min(center.y + half);
                painter.line_segment(
                    [egui::pos2(center.x, y), egui::pos2(center.x, y1)],
                    egui::Stroke::new(1.0, color),
                );
                y += dash + gap;
            }
        }
    }
}

fn zero_angle_axis_color(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_gray(95).gamma_multiply(0.58)
    } else {
        egui::Color32::from_rgb(105, 113, 126).gamma_multiply(0.58)
    }
}

fn draw_rotated_outline(
    painter: &egui::Painter,
    center: egui::Pos2,
    scale: f32,
    axis: AttitudeAxis,
    angle_deg: f64,
    color: egui::Color32,
    show_wheels: bool,
) {
    let (w, h) = match axis {
        AttitudeAxis::Roll => (0.58 * scale, 0.30 * scale),
        AttitudeAxis::Pitch => (0.68 * scale, 0.24 * scale),
        AttitudeAxis::Yaw => (0.62 * scale, 0.28 * scale),
    };
    let angle = orthogonal_view_screen_angle(axis, angle_deg);
    let corners = [
        egui::vec2(-0.5 * w, -0.5 * h),
        egui::vec2(0.5 * w, -0.5 * h),
        egui::vec2(0.5 * w, 0.5 * h),
        egui::vec2(-0.5 * w, 0.5 * h),
    ];
    let mut points: Vec<egui::Pos2> = corners
        .into_iter()
        .map(|p| rotate_point(center, p, angle))
        .collect();
    points.push(points[0]);
    painter.add(egui::Shape::line(points, egui::Stroke::new(1.8, color)));

    let nose = match axis {
        AttitudeAxis::Roll => [egui::vec2(0.0, -0.5 * h), egui::vec2(0.0, -0.82 * h)],
        _ => [egui::vec2(0.5 * w, 0.0), egui::vec2(0.78 * w, 0.0)],
    };
    painter.line_segment(
        [
            rotate_point(center, nose[0], angle),
            rotate_point(center, nose[1], angle),
        ],
        egui::Stroke::new(1.8, color),
    );
    if show_wheels {
        draw_vehicle_wheels(painter, center, w, h, angle, color, axis);
    }
}

fn orthogonal_view_screen_angle(axis: AttitudeAxis, angle_deg: f64) -> f32 {
    let angle = angle_deg as f32;
    match axis {
        // FRD roll: positive roll moves the vehicle right side down. With
        // screen +x as vehicle right and screen +y as vehicle down, this is a
        // clockwise screen rotation.
        AttitudeAxis::Roll => angle.to_radians(),
        // FRD/NED pitch traces use positive pitch as nose-up. The side view
        // has screen +x forward and screen +y down, so nose-up is
        // counter-clockwise on screen.
        AttitudeAxis::Pitch => (-angle).to_radians(),
        // Yaw traces are headings: 0 deg is north/up, positive yaw turns
        // clockwise toward east/right.
        AttitudeAxis::Yaw => (-90.0 + angle).to_radians(),
    }
}

fn draw_vehicle_wheels(
    painter: &egui::Painter,
    center: egui::Pos2,
    w: f32,
    h: f32,
    angle: f32,
    color: egui::Color32,
    axis: AttitudeAxis,
) {
    if matches!(axis, AttitudeAxis::Roll | AttitudeAxis::Yaw) {
        draw_rect_vehicle_wheels(painter, center, w, h, angle, color, axis);
        return;
    }

    let wheel_radius = (w.min(h) * 0.20).clamp(4.5, 9.0);
    let wheel_points: &[egui::Vec2] = match axis {
        AttitudeAxis::Roll => &[
            egui::vec2(-0.36 * w, 0.58 * h),
            egui::vec2(0.36 * w, 0.58 * h),
        ],
        AttitudeAxis::Pitch => &[
            egui::vec2(-0.34 * w, 0.56 * h),
            egui::vec2(0.34 * w, 0.56 * h),
        ],
        AttitudeAxis::Yaw => &[],
    };
    for wheel in wheel_points {
        let pos = rotate_point(center, *wheel, angle);
        painter.circle_filled(pos, wheel_radius, egui::Color32::from_black_alpha(180));
        painter.circle_stroke(pos, wheel_radius, egui::Stroke::new(1.0, color));
    }
}

fn draw_rect_vehicle_wheels(
    painter: &egui::Painter,
    center: egui::Pos2,
    w: f32,
    h: f32,
    angle: f32,
    color: egui::Color32,
    axis: AttitudeAxis,
) {
    let wheel_size = match axis {
        AttitudeAxis::Roll => egui::vec2(0.18 * w, 0.28 * h),
        AttitudeAxis::Yaw => egui::vec2(0.18 * w, 0.18 * h),
        AttitudeAxis::Pitch => egui::Vec2::ZERO,
    };
    let wheel_centers: &[egui::Vec2] = match axis {
        AttitudeAxis::Roll => &[
            egui::vec2(-0.36 * w, 0.62 * h),
            egui::vec2(0.36 * w, 0.62 * h),
        ],
        AttitudeAxis::Yaw => &[
            egui::vec2(-0.34 * w, -0.56 * h),
            egui::vec2(0.34 * w, -0.56 * h),
            egui::vec2(-0.34 * w, 0.56 * h),
            egui::vec2(0.34 * w, 0.56 * h),
        ],
        AttitudeAxis::Pitch => &[],
    };
    for wheel_center in wheel_centers {
        let half = 0.5 * wheel_size;
        let corners = [
            *wheel_center + egui::vec2(-half.x, -half.y),
            *wheel_center + egui::vec2(half.x, -half.y),
            *wheel_center + egui::vec2(half.x, half.y),
            *wheel_center + egui::vec2(-half.x, half.y),
        ];
        let points: Vec<egui::Pos2> = corners
            .into_iter()
            .map(|p| rotate_point(center, p, angle))
            .collect();
        painter.add(egui::Shape::convex_polygon(
            points,
            egui::Color32::from_black_alpha(185),
            egui::Stroke::new(1.0, color),
        ));
    }
}

fn rotate_point(center: egui::Pos2, point: egui::Vec2, angle: f32) -> egui::Pos2 {
    let (sin, cos) = angle.sin_cos();
    center + egui::vec2(point.x * cos - point.y * sin, point.x * sin + point.y * cos)
}
