//! Hover-driven update inspector aggregation and rendering.

use eframe::egui;

use crate::visualizer::model::{
    PlotData, StateContribution, StateCorrelation, UpdateInspectorSample,
};

use super::state::display_filter_trace_name;
use super::trace_query::{mount_estimate_reference_traces, sample_trace_at, wrap_degrees};

const UPDATE_INSPECTOR_WINDOW_S: f64 = 5.0;

pub(super) fn update_inspector_view_model(
    data: &PlotData,
    t_s: f64,
) -> Option<(Vec<UpdateInspectorSample>, Vec<String>, f64)> {
    let samples = window_update_inspector_samples(
        &data.update_inspector,
        t_s - UPDATE_INSPECTOR_WINDOW_S,
        t_s,
    );
    if samples.is_empty() {
        return None;
    }
    let columns = top_inspector_states(&samples, 6);
    if columns.is_empty() {
        return None;
    }
    let max_abs = samples
        .iter()
        .flat_map(|sample| sample.contributions.iter())
        .filter(|contribution| columns.iter().any(|name| name == &contribution.state))
        .map(|contribution| contribution.value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0e-12);
    Some((samples, columns, max_abs))
}

pub(super) fn render_update_inspector_contents(
    ui: &mut egui::Ui,
    data: &PlotData,
    t_s: f64,
    samples: &[UpdateInspectorSample],
    columns: &[String],
    max_abs: f64,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Update Inspector").strong());
        ui.label(
            egui::RichText::new(format!(
                "last {:.1}s, {:.2}-{t_s:.2} s",
                UPDATE_INSPECTOR_WINDOW_S,
                t_s - UPDATE_INSPECTOR_WINDOW_S
            ))
            .weak(),
        );
    });
    ui.label(
        egui::RichText::new(
            "Mount error change and signed update allocation over the hover interval.",
        )
        .small()
        .weak(),
    );
    render_mount_error_ledger(ui, data, t_s, samples);
    ui.add_space(4.0);
    egui::ScrollArea::horizontal()
        .id_salt("update_inspector_heatmap")
        .max_height(154.0)
        .show(ui, |ui| {
            egui::Grid::new("update_inspector_grid")
                .striped(false)
                .spacing(egui::vec2(4.0, 4.0))
                .show(ui, |ui| {
                    ui.label("");
                    for column in columns {
                        ui.label(egui::RichText::new(column).small().strong());
                    }
                    ui.end_row();

                    for sample in samples {
                        let residual = sample
                            .residual
                            .map(|v| format!(" r={v:+.3}"))
                            .unwrap_or_default();
                        let nis = sample
                            .nis
                            .map(|v| format!(" NIS={v:.2}"))
                            .unwrap_or_default();
                        ui.label(
                            egui::RichText::new(format!(
                                "{} {}  dt={:+.2}s{}{}",
                                display_filter_trace_name(&sample.filter),
                                sample.update,
                                sample.t_s - t_s,
                                residual,
                                nis
                            ))
                            .small(),
                        );
                        for column in columns {
                            let contribution = sample
                                .contributions
                                .iter()
                                .find(|contribution| &contribution.state == column);
                            draw_inspector_cell(ui, contribution, max_abs);
                        }
                        ui.end_row();
                    }
                });
        });
    if let Some(sample) = samples.first() {
        let mut ranked = sample.contributions.clone();
        ranked.sort_by(|a, b| b.value.abs().total_cmp(&a.value.abs()));
        let summary = ranked
            .into_iter()
            .filter(|c| c.value.abs() > 0.0)
            .take(5)
            .map(|c| format!("{} {:+.4} {}", c.state, c.value, c.unit))
            .collect::<Vec<_>>()
            .join("  ");
        if !summary.is_empty() {
            ui.label(egui::RichText::new(summary).small().weak());
        }
    }
    render_update_inspector_correlations(ui, samples);
}

fn render_mount_error_ledger(
    ui: &mut egui::Ui,
    data: &PlotData,
    t_s: f64,
    samples: &[UpdateInspectorSample],
) {
    let start_t_s = t_s - UPDATE_INSPECTOR_WINDOW_S;
    let rows = mount_error_rows(data, start_t_s, t_s);
    if rows.is_empty() {
        return;
    }

    ui.add_space(6.0);
    ui.label(egui::RichText::new("Mount error ledger").small().strong());
    egui::Grid::new("mount_error_ledger_grid")
        .striped(false)
        .spacing(egui::vec2(8.0, 3.0))
        .show(ui, |ui| {
            ui.label(egui::RichText::new("filter").small().weak());
            ui.label(egui::RichText::new("axis").small().weak());
            ui.label(egui::RichText::new("error start").small().weak());
            ui.label(egui::RichText::new("error end").small().weak());
            ui.label(egui::RichText::new("delta").small().weak());
            ui.label(egui::RichText::new("net").small().weak());
            ui.end_row();

            for row in &rows {
                let net = if row.abs_delta > 0.05 {
                    "away"
                } else if row.abs_delta < -0.05 {
                    "toward"
                } else {
                    "flat"
                };
                ui.label(egui::RichText::new(display_filter_trace_name(row.filter)).small());
                ui.label(egui::RichText::new(row.axis).small());
                ui.label(egui::RichText::new(format!("{:+.2} deg", row.start_error_deg)).small());
                ui.label(egui::RichText::new(format!("{:+.2} deg", row.end_error_deg)).small());
                draw_signed_status_value(ui, row.delta_error_deg, row.abs_delta);
                ui.label(egui::RichText::new(net).small());
                ui.end_row();
            }
        });

    let pushes = mount_update_push_rows(samples, &rows);
    if pushes.is_empty() {
        return;
    }
    ui.add_space(3.0);
    ui.label(egui::RichText::new("Mount update pushes").small().strong());
    egui::Grid::new("mount_update_push_grid")
        .striped(false)
        .spacing(egui::vec2(8.0, 3.0))
        .show(ui, |ui| {
            ui.label(egui::RichText::new("update").small().weak());
            ui.label(egui::RichText::new("axis").small().weak());
            ui.label(egui::RichText::new("push").small().weak());
            ui.label(egui::RichText::new("effect").small().weak());
            ui.end_row();
            for push in pushes.into_iter().take(8) {
                ui.label(egui::RichText::new(push.update).small());
                ui.label(egui::RichText::new(push.axis).small());
                draw_signed_status_value(ui, push.dx_deg, push.away_score);
                ui.label(egui::RichText::new(push.effect).small());
                ui.end_row();
            }
        });
}

struct MountErrorLedgerRow {
    filter: &'static str,
    axis: &'static str,
    start_error_deg: f64,
    end_error_deg: f64,
    delta_error_deg: f64,
    abs_delta: f64,
}

struct MountUpdatePushRow {
    update: String,
    axis: &'static str,
    dx_deg: f64,
    away_score: f64,
    effect: &'static str,
}

fn mount_error_rows(data: &PlotData, start_t_s: f64, end_t_s: f64) -> Vec<MountErrorLedgerRow> {
    let mut rows = Vec::new();
    for filter in ["EKF"] {
        for axis in ["roll", "pitch", "yaw"] {
            let Some((estimate, reference)) = mount_estimate_reference_traces(data, filter, axis)
            else {
                continue;
            };
            let Some(start_estimate) = sample_trace_at(estimate, start_t_s) else {
                continue;
            };
            let Some(end_estimate) = sample_trace_at(estimate, end_t_s) else {
                continue;
            };
            let Some(start_reference) = sample_trace_at(reference, start_t_s) else {
                continue;
            };
            let Some(end_reference) = sample_trace_at(reference, end_t_s) else {
                continue;
            };
            let start_error_deg = wrap_degrees(start_estimate - start_reference);
            let end_error_deg = wrap_degrees(end_estimate - end_reference);
            let delta_error_deg = wrap_degrees(end_error_deg - start_error_deg);
            rows.push(MountErrorLedgerRow {
                filter,
                axis,
                start_error_deg,
                end_error_deg,
                delta_error_deg,
                abs_delta: end_error_deg.abs() - start_error_deg.abs(),
            });
        }
    }
    rows
}

fn mount_update_push_rows(
    samples: &[UpdateInspectorSample],
    error_rows: &[MountErrorLedgerRow],
) -> Vec<MountUpdatePushRow> {
    let mut pushes = Vec::new();
    for sample in samples {
        for axis in ["roll", "pitch", "yaw"] {
            let state = format!("mount {axis}");
            let Some(contribution) = sample
                .contributions
                .iter()
                .find(|contribution| contribution.state == state)
            else {
                continue;
            };
            if contribution.value.abs() < 1.0e-6 {
                continue;
            }
            let Some(error) = error_rows
                .iter()
                .find(|row| row.filter == sample.filter && row.axis == axis)
            else {
                continue;
            };
            let before = error.start_error_deg.abs();
            let after = wrap_degrees(error.start_error_deg + contribution.value).abs();
            let away_score = after - before;
            let effect = if away_score > 0.01 {
                "away"
            } else if away_score < -0.01 {
                "toward"
            } else {
                "flat"
            };
            pushes.push(MountUpdatePushRow {
                update: format!(
                    "{} {}",
                    display_filter_trace_name(&sample.filter),
                    sample.update
                ),
                axis,
                dx_deg: contribution.value,
                away_score,
                effect,
            });
        }
    }
    pushes.sort_by(|a, b| {
        b.away_score
            .abs()
            .total_cmp(&a.away_score.abs())
            .then_with(|| b.dx_deg.abs().total_cmp(&a.dx_deg.abs()))
    });
    pushes
}

fn draw_signed_status_value(ui: &mut egui::Ui, value: f64, bad_score: f64) -> egui::Response {
    let normalized = (bad_score.abs() / 1.0).clamp(0.0, 1.0) as f32;
    let alpha = (28.0 + 150.0 * normalized) as u8;
    let color = if bad_score > 0.0 {
        egui::Color32::from_rgba_premultiplied(186, 62, 70, alpha)
    } else if bad_score < 0.0 {
        egui::Color32::from_rgba_premultiplied(44, 132, 94, alpha)
    } else {
        ui.visuals().faint_bg_color
    };
    let text_color = if normalized > 0.55 {
        egui::Color32::WHITE
    } else {
        ui.visuals().text_color()
    };
    egui::Frame::new()
        .fill(color)
        .corner_radius(egui::CornerRadius::same(3))
        .inner_margin(egui::Margin::symmetric(5, 1))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(format!("{value:+.3} deg"))
                    .small()
                    .monospace()
                    .color(text_color),
            )
        })
        .inner
}

fn render_update_inspector_correlations(ui: &mut egui::Ui, samples: &[UpdateInspectorSample]) {
    let mut correlations = samples
        .iter()
        .flat_map(|sample| {
            sample
                .correlations
                .iter()
                .map(|correlation| (sample.filter.as_str(), sample.update.as_str(), correlation))
        })
        .filter(|(_, _, correlation)| {
            correlation.value.is_finite() && correlation.value.abs() >= 0.15
        })
        .collect::<Vec<_>>();
    if correlations.is_empty() {
        return;
    }
    correlations.sort_by(|a, b| {
        b.2.value
            .abs()
            .total_cmp(&a.2.value.abs())
            .then_with(|| a.0.cmp(b.0))
            .then_with(|| a.2.mount_axis.cmp(&b.2.mount_axis))
            .then_with(|| a.2.state.cmp(&b.2.state))
    });
    ui.add_space(6.0);
    ui.label(
        egui::RichText::new("Top covariance correlations to mount")
            .small()
            .strong(),
    );
    egui::ScrollArea::vertical()
        .id_salt("update_inspector_correlations")
        .max_height(96.0)
        .show(ui, |ui| {
            egui::Grid::new("update_inspector_correlations_grid")
                .striped(false)
                .spacing(egui::vec2(8.0, 3.0))
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("filter").small().weak());
                    ui.label(egui::RichText::new("mount").small().weak());
                    ui.label(egui::RichText::new("state").small().weak());
                    ui.label(egui::RichText::new("rho").small().weak());
                    ui.end_row();
                    for (filter, update, correlation) in correlations.into_iter().take(10) {
                        draw_correlation_row(ui, filter, update, correlation);
                    }
                });
        });
}

fn draw_correlation_row(
    ui: &mut egui::Ui,
    filter: &str,
    update: &str,
    correlation: &StateCorrelation,
) {
    ui.label(egui::RichText::new(display_filter_trace_name(filter)).small());
    ui.label(egui::RichText::new(&correlation.mount_axis).small());
    ui.label(
        egui::RichText::new(compact_correlation_state(&correlation.state))
            .small()
            .monospace(),
    );
    draw_correlation_value(ui, correlation.value).on_hover_text(format!(
        "{} {update}: rho({} / {}, mount {}) = {:+.4}",
        display_filter_trace_name(filter),
        correlation.group,
        correlation.state,
        correlation.mount_axis,
        correlation.value
    ));
    ui.end_row();
}

fn compact_correlation_state(state: &str) -> String {
    state
        .replace("gyro bias ", "gb")
        .replace("accel bias ", "ab")
        .replace("gyro scale ", "gs")
        .replace("accel scale ", "as")
        .replace("mount ", "m")
        .replace("vel ", "v")
        .replace("pos ", "p")
}

fn draw_correlation_value(ui: &mut egui::Ui, value: f64) -> egui::Response {
    let normalized = value.abs().clamp(0.0, 1.0) as f32;
    let alpha = (42.0 + 160.0 * normalized) as u8;
    let color = if value >= 0.0 {
        egui::Color32::from_rgba_premultiplied(186, 62, 70, alpha)
    } else {
        egui::Color32::from_rgba_premultiplied(53, 112, 190, alpha)
    };
    let text_color = if normalized > 0.55 {
        egui::Color32::WHITE
    } else {
        ui.visuals().text_color()
    };
    egui::Frame::new()
        .fill(color)
        .corner_radius(egui::CornerRadius::same(3))
        .inner_margin(egui::Margin::symmetric(5, 1))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(format!("{value:+.2}"))
                    .small()
                    .monospace()
                    .color(text_color),
            )
        })
        .inner
}

fn window_update_inspector_samples(
    samples: &[UpdateInspectorSample],
    start_t_s: f64,
    end_t_s: f64,
) -> Vec<UpdateInspectorSample> {
    let mut out = Vec::<UpdateInspectorSample>::new();
    for sample in samples
        .iter()
        .filter(|sample| sample.t_s >= start_t_s && sample.t_s <= end_t_s)
    {
        let Some(existing) = out
            .iter_mut()
            .find(|existing| existing.filter == sample.filter && existing.update == sample.update)
        else {
            let mut sample = sample.clone();
            sample.residual = sample.residual.map(f64::abs);
            sample.nis = sample.nis.map(f64::abs);
            out.push(sample);
            continue;
        };
        existing.t_s = sample.t_s;
        existing.residual = match (existing.residual, sample.residual) {
            (Some(a), Some(b)) => Some(a + b.abs()),
            (None, Some(b)) => Some(b.abs()),
            (value, None) => value,
        };
        existing.nis = match (existing.nis, sample.nis) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (None, Some(b)) => Some(b),
            (value, None) => value,
        };
        for contribution in &sample.contributions {
            if let Some(existing_contribution) = existing
                .contributions
                .iter_mut()
                .find(|existing| existing.state == contribution.state)
            {
                existing_contribution.value += contribution.value;
            } else {
                existing.contributions.push(contribution.clone());
            }
        }
        for correlation in &sample.correlations {
            if let Some(existing_correlation) = existing.correlations.iter_mut().find(|existing| {
                existing.group == correlation.group
                    && existing.state == correlation.state
                    && existing.mount_axis == correlation.mount_axis
            }) {
                if correlation.value.abs() > existing_correlation.value.abs() {
                    *existing_correlation = correlation.clone();
                }
            } else {
                existing.correlations.push(correlation.clone());
            }
        }
    }
    out.sort_by(|a, b| {
        a.filter
            .cmp(&b.filter)
            .then_with(|| a.update.cmp(&b.update))
    });
    out
}

fn top_inspector_states(samples: &[UpdateInspectorSample], max_states: usize) -> Vec<String> {
    let mut totals = Vec::<(String, f64)>::new();
    for sample in samples {
        for contribution in &sample.contributions {
            if let Some((_, total)) = totals
                .iter_mut()
                .find(|(state, _)| state == &contribution.state)
            {
                *total += contribution.value.abs();
            } else {
                totals.push((contribution.state.clone(), contribution.value.abs()));
            }
        }
    }
    totals.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    totals
        .into_iter()
        .filter(|(_, value)| *value > 0.0)
        .take(max_states)
        .map(|(state, _)| state)
        .collect()
}

fn draw_inspector_cell(ui: &mut egui::Ui, contribution: Option<&StateContribution>, max_abs: f64) {
    let Some(contribution) = contribution else {
        ui.label(egui::RichText::new(" ").small());
        return;
    };
    let normalized = (contribution.value.abs() / max_abs).clamp(0.0, 1.0) as f32;
    let alpha = (38.0 + 170.0 * normalized) as u8;
    let color = if contribution.value >= 0.0 {
        egui::Color32::from_rgba_premultiplied(186, 62, 70, alpha)
    } else {
        egui::Color32::from_rgba_premultiplied(53, 112, 190, alpha)
    };
    let text_color = if normalized > 0.55 {
        egui::Color32::WHITE
    } else {
        ui.visuals().text_color()
    };
    egui::Frame::new()
        .fill(color)
        .corner_radius(egui::CornerRadius::same(3))
        .inner_margin(egui::Margin::symmetric(5, 2))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(format!("{:+.3}", contribution.value))
                    .small()
                    .color(text_color),
            )
            .on_hover_text(format!(
                "{} / {}: {:+.6} {}",
                contribution.group, contribution.state, contribution.value, contribution.unit
            ));
        });
}
