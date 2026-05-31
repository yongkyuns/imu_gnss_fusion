//! Shared color classification for plots, map overlays, markers, and popups.

use eframe::egui;

use crate::visualizer::model::RoadEventKind;

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

pub(super) fn event_marker_color(kind: &str, visuals: &egui::Visuals) -> egui::Color32 {
    match RoadEventKind::parse(kind) {
        Some(RoadEventKind::SpeedBump) if visuals.dark_mode => {
            egui::Color32::from_rgb(255, 212, 92)
        }
        Some(RoadEventKind::SpeedBump) => egui::Color32::from_rgb(190, 109, 0),
        Some(RoadEventKind::RoadShock) if visuals.dark_mode => {
            egui::Color32::from_rgb(255, 184, 77)
        }
        Some(RoadEventKind::RoadShock) => egui::Color32::from_rgb(185, 91, 0),
        Some(RoadEventKind::RoughRoad) if visuals.dark_mode => {
            egui::Color32::from_rgb(180, 210, 95)
        }
        Some(RoadEventKind::RoughRoad) => egui::Color32::from_rgb(92, 134, 35),
        Some(RoadEventKind::Uphill) if visuals.dark_mode => egui::Color32::from_rgb(255, 154, 68),
        Some(RoadEventKind::Uphill) => egui::Color32::from_rgb(214, 97, 0),
        Some(RoadEventKind::Downhill) if visuals.dark_mode => egui::Color32::from_rgb(86, 190, 255),
        Some(RoadEventKind::Downhill) => egui::Color32::from_rgb(0, 126, 182),
        Some(RoadEventKind::Reverse) if visuals.dark_mode => egui::Color32::from_rgb(196, 146, 255),
        Some(RoadEventKind::Reverse) => egui::Color32::from_rgb(133, 76, 214),
        Some(RoadEventKind::HarshAcceleration) if visuals.dark_mode => {
            egui::Color32::from_rgb(92, 230, 128)
        }
        Some(RoadEventKind::HarshAcceleration) => egui::Color32::from_rgb(0, 150, 78),
        Some(RoadEventKind::HarshBraking) if visuals.dark_mode => {
            egui::Color32::from_rgb(255, 92, 92)
        }
        Some(RoadEventKind::HarshBraking) => egui::Color32::from_rgb(206, 45, 45),
        Some(RoadEventKind::HarshCornering) if visuals.dark_mode => {
            egui::Color32::from_rgb(255, 116, 218)
        }
        Some(RoadEventKind::HarshCornering) => egui::Color32::from_rgb(190, 54, 165),
        _ => map_marker_color(kind, visuals),
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
