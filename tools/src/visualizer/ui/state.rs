//! Shared UI state enums and trace visibility classification.

use crate::visualizer::model::{RoadEventKind, Trace};

pub(super) const EKF_FILTER_LABEL: &str = "EKF";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DataOrigin {
    Real,
    Synthetic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TuningPanel {
    Ekf,
    Align,
    RoadEvents,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EventVisibility {
    pub show_speed_bump: bool,
    pub show_road_shock: bool,
    pub show_rough_road: bool,
    pub show_uphill: bool,
    pub show_downhill: bool,
    pub show_reverse: bool,
    pub show_harsh_accel: bool,
    pub show_harsh_brake: bool,
    pub show_harsh_cornering: bool,
}

impl Default for EventVisibility {
    fn default() -> Self {
        Self {
            show_speed_bump: true,
            show_road_shock: true,
            show_rough_road: true,
            show_uphill: true,
            show_downhill: true,
            show_reverse: true,
            show_harsh_accel: true,
            show_harsh_brake: true,
            show_harsh_cornering: true,
        }
    }
}

impl EventVisibility {
    pub(super) fn allows_kind(self, kind: &str) -> bool {
        match RoadEventKind::parse(kind) {
            Some(RoadEventKind::SpeedBump) => self.show_speed_bump,
            Some(RoadEventKind::RoadShock) => self.show_road_shock,
            Some(RoadEventKind::RoughRoad) => self.show_rough_road,
            Some(RoadEventKind::Uphill) => self.show_uphill,
            Some(RoadEventKind::Downhill) => self.show_downhill,
            Some(RoadEventKind::Reverse) => self.show_reverse,
            Some(RoadEventKind::HarshAcceleration) => self.show_harsh_accel,
            Some(RoadEventKind::HarshBraking) => self.show_harsh_brake,
            Some(RoadEventKind::HarshCornering) => self.show_harsh_cornering,
            _ => true,
        }
    }

    pub(super) fn set_all(&mut self, enabled: bool) {
        self.show_speed_bump = enabled;
        self.show_road_shock = enabled;
        self.show_rough_road = enabled;
        self.show_uphill = enabled;
        self.show_downhill = enabled;
        self.show_reverse = enabled;
        self.show_harsh_accel = enabled;
        self.show_harsh_brake = enabled;
        self.show_harsh_cornering = enabled;
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TraceVisibility {
    pub show_reference: bool,
    pub show_align: bool,
    pub show_ekf: bool,
}

impl TraceVisibility {
    pub(super) fn allows(self, trace: &Trace) -> bool {
        self.allows_in_plot("", trace)
    }

    pub(super) fn allows_in_plot(self, plot_title: &str, trace: &Trace) -> bool {
        let name = trace.name.as_str();
        if !self.show_reference && is_reference_trace_name(name) {
            return false;
        }
        if !self.show_align && (is_align_trace_name(name) || is_align_plot_title(plot_title)) {
            return false;
        }
        if !self.show_ekf && is_ekf_trace_name(name) {
            return false;
        }
        true
    }
}

pub(super) fn is_reference_trace_name(name: &str) -> bool {
    name.starts_with("Reference")
        || name.contains(" Reference")
        || name.starts_with("Synthetic truth")
        || name.contains("Synthetic truth")
        || name.contains("truth path")
}

fn is_align_trace_name(name: &str) -> bool {
    name.starts_with("Align") || name.contains(" Align")
}

fn is_align_plot_title(title: &str) -> bool {
    title.starts_with("Align ")
}

fn is_ekf_trace_name(name: &str) -> bool {
    name.starts_with("EKF")
        || name.contains(" EKF")
        || name.contains("ekf")
        || name.contains("EKF initialized")
        || name.contains("ekf initialized")
        || name.contains("mount ready")
}

pub(super) fn display_filter_trace_name(name: &str) -> String {
    name.replace("EKF", EKF_FILTER_LABEL)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(name: &str) -> Trace {
        Trace {
            name: name.to_string(),
            points: vec![[0.0, 0.0]],
        }
    }

    #[test]
    fn ekf_toggle_hides_ekf_traces() {
        let visibility = TraceVisibility {
            show_reference: true,
            show_align: true,
            show_ekf: true,
        };

        assert!(visibility.allows(&trace("Reference mount roll [deg]")));
        assert!(visibility.allows(&trace("Align mount quaternion error [deg]")));
        assert!(visibility.allows(&trace("EKF mount roll [deg]")));
        assert!(visibility.allows(&trace("mount ready")));
    }

    #[test]
    fn source_toggles_are_independent_for_mount_traces() {
        let reference_off = TraceVisibility {
            show_reference: false,
            show_align: true,
            show_ekf: true,
        };
        assert!(!reference_off.allows(&trace("Reference mount pitch [deg]")));
        assert!(reference_off.allows(&trace("Align pitch [deg]")));
        assert!(reference_off.allows(&trace("EKF mount pitch [deg]")));

        let align_off = TraceVisibility {
            show_reference: true,
            show_align: false,
            show_ekf: true,
        };
        assert!(!align_off.allows(&trace("Align yaw [deg]")));
        assert!(align_off.allows(&trace("Reference mount yaw [deg]")));
        assert!(align_off.allows(&trace("EKF mount yaw [deg]")));

        let ekf_off = TraceVisibility {
            show_reference: true,
            show_align: true,
            show_ekf: false,
        };
        assert!(!ekf_off.allows(&trace("EKF mount roll [deg]")));
        assert!(!ekf_off.allows(&trace("EKF accel bias sigma X [m/s^2]")));
        assert!(!ekf_off.allows(&trace("EKF attitude roll sigma [deg]")));
        assert!(!ekf_off.allows(&trace("EKF pitch HPF [deg]")));
        assert!(!ekf_off.allows(&trace("EKF vehicle speed [m/s]")));
        assert!(ekf_off.allows(&trace("Reference mount roll [deg]")));
        assert!(ekf_off.allows(&trace("Align roll [deg]")));
    }

    #[test]
    fn event_visibility_filters_known_event_kinds() {
        let mut visibility = EventVisibility::default();
        assert!(visibility.allows_kind("speed bump"));
        assert!(visibility.allows_kind("road shock"));
        assert!(visibility.allows_kind("rough road"));
        assert!(visibility.allows_kind("uphill"));
        assert!(visibility.allows_kind("harsh braking"));
        assert!(visibility.allows_kind("unknown future event"));

        visibility.show_speed_bump = false;
        visibility.show_road_shock = false;
        visibility.show_rough_road = false;
        visibility.show_downhill = false;
        visibility.show_harsh_cornering = false;
        assert!(!visibility.allows_kind("speed bump"));
        assert!(!visibility.allows_kind("road shock"));
        assert!(!visibility.allows_kind("rough road"));
        assert!(visibility.allows_kind("uphill"));
        assert!(!visibility.allows_kind("downhill"));
        assert!(visibility.allows_kind("harsh acceleration"));
        assert!(!visibility.allows_kind("harsh cornering"));

        visibility.set_all(false);
        assert!(!visibility.allows_kind("reverse"));
        assert!(!visibility.allows_kind("harsh braking"));
        assert!(visibility.allows_kind("unknown future event"));
    }
}
