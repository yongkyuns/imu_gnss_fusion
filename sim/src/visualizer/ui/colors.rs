//! Shared color classification for plots, map overlays, markers, and popups.

use eframe::egui;

use super::state::is_reference_trace_name;

pub(super) fn map_trace_color(name: &str, visuals: &egui::Visuals) -> egui::Color32 {
    if is_reference_trace_name(name) && name.contains("path") {
        SeriesColor::Reference.resolve(visuals)
    } else if name.contains("GNSS") || name.contains("GNSS-only") {
        if visuals.dark_mode {
            egui::Color32::from_rgb(0, 255, 255)
        } else {
            egui::Color32::from_rgb(0, 118, 152)
        }
    } else if name == "EKF path (lon,lat)" {
        if visuals.dark_mode {
            egui::Color32::from_rgb(120, 170, 255)
        } else {
            egui::Color32::from_rgb(35, 105, 200)
        }
    } else if name == "EKF path during GNSS outage (lon,lat)" {
        if visuals.dark_mode {
            egui::Color32::from_rgb(255, 140, 220)
        } else {
            egui::Color32::from_rgb(184, 55, 144)
        }
    } else {
        visuals.text_color()
    }
}

pub(super) fn map_heading_color(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_rgb(245, 248, 252)
    } else {
        egui::Color32::from_rgb(42, 49, 59)
    }
}

pub(super) fn cursor_marker_color(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_rgb(255, 220, 70)
    } else {
        egui::Color32::from_rgb(194, 119, 0)
    }
}

pub(super) fn map_marker_color(name: &str, visuals: &egui::Visuals) -> egui::Color32 {
    let base = map_trace_color(name, visuals);
    if visuals.dark_mode {
        base.gamma_multiply(1.35)
    } else {
        base.gamma_multiply(0.72)
    }
}

pub(super) fn marker_outline_color(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_black_alpha(210)
    } else {
        egui::Color32::from_white_alpha(230)
    }
}

pub(super) fn tooltip_fill(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_black_alpha(190)
    } else {
        egui::Color32::from_white_alpha(230)
    }
}

pub(super) fn tooltip_text_color(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::WHITE
    } else {
        egui::Color32::from_rgb(32, 38, 48)
    }
}

pub(super) fn shared_cursor_color(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_gray(210).gamma_multiply(0.55)
    } else {
        egui::Color32::from_rgb(73, 84, 100).gamma_multiply(0.72)
    }
}

#[derive(Clone, Copy)]
pub(super) enum SeriesColor {
    Reference,
    Ekf,
    Align,
}

impl SeriesColor {
    pub(super) fn resolve(self, visuals: &egui::Visuals) -> egui::Color32 {
        let dark = visuals.dark_mode;
        match self {
            Self::Reference if dark => egui::Color32::from_rgb(235, 238, 244),
            Self::Reference => egui::Color32::from_rgb(34, 43, 55),
            Self::Ekf if dark => egui::Color32::from_rgb(120, 170, 255),
            Self::Ekf => egui::Color32::from_rgb(35, 105, 200),
            Self::Align if dark => egui::Color32::from_rgb(244, 190, 96),
            Self::Align => egui::Color32::from_rgb(168, 93, 22),
        }
    }
}
