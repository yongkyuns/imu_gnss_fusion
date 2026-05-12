//! Shared UI state enums and trace visibility classification.

use crate::visualizer::model::Trace;

pub(super) const REDUCED_FILTER_LABEL: &str = "Reduced";
pub(super) const FULL_FILTER_LABEL: &str = "Full";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DataOrigin {
    Real,
    Synthetic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TuningPanel {
    Reduced,
    Align,
    Full,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TraceVisibility {
    pub show_reference: bool,
    pub show_align: bool,
    pub show_reduced: bool,
    pub show_full: bool,
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
        if !self.show_reduced && is_reduced_trace_name(name) {
            return false;
        }
        if !self.show_full && is_full_trace_name(name) {
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

fn is_reduced_trace_name(name: &str) -> bool {
    name.starts_with("Reduced")
        || name.contains(" Reduced")
        || name.contains("reduced")
        || name.contains("Reduced initialized")
        || name.contains("reduced initialized")
        || name.contains("mount ready")
}

fn is_full_trace_name(name: &str) -> bool {
    name.starts_with("Full") || name.contains(" Full") || name.contains("full")
}

pub(super) fn display_filter_trace_name(name: &str) -> String {
    name.replace("Full", FULL_FILTER_LABEL)
        .replace("Reduced", REDUCED_FILTER_LABEL)
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
    fn full_toggle_only_hides_full_traces() {
        let visibility = TraceVisibility {
            show_reference: true,
            show_align: true,
            show_reduced: true,
            show_full: false,
        };

        assert!(visibility.allows(&trace("Reference mount roll [deg]")));
        assert!(visibility.allows(&trace("Align mount quaternion error [deg]")));
        assert!(visibility.allows(&trace("Reduced mount roll [deg]")));
        assert!(visibility.allows(&trace("mount ready")));
        assert!(!visibility.allows(&trace("Full mount roll [deg]")));
        assert!(!visibility.allows(&trace("Full sigma state 0")));
    }

    #[test]
    fn source_toggles_are_independent_for_mount_traces() {
        let reference_off = TraceVisibility {
            show_reference: false,
            show_align: true,
            show_reduced: true,
            show_full: true,
        };
        assert!(!reference_off.allows(&trace("Reference mount pitch [deg]")));
        assert!(reference_off.allows(&trace("Align pitch [deg]")));
        assert!(reference_off.allows(&trace("Reduced mount pitch [deg]")));
        assert!(reference_off.allows(&trace("Full mount pitch [deg]")));

        let align_off = TraceVisibility {
            show_reference: true,
            show_align: false,
            show_reduced: true,
            show_full: true,
        };
        assert!(!align_off.allows(&trace("Align yaw [deg]")));
        assert!(align_off.allows(&trace("Reference mount yaw [deg]")));
        assert!(align_off.allows(&trace("Reduced mount yaw [deg]")));
        assert!(align_off.allows(&trace("Full mount yaw [deg]")));

        let reduced_off = TraceVisibility {
            show_reference: true,
            show_align: true,
            show_reduced: false,
            show_full: true,
        };
        assert!(!reduced_off.allows(&trace("Reduced mount roll [deg]")));
        assert!(!reduced_off.allows(&trace("Reduced accel bias sigma X [m/s^2]")));
        assert!(!reduced_off.allows(&trace("Reduced state_0")));
        assert!(!reduced_off.allows(&trace("Reduced pitch HPF [deg]")));
        assert!(!reduced_off.allows(&trace("Reduced vehicle speed [m/s]")));
        assert!(reduced_off.allows(&trace("Reference mount roll [deg]")));
        assert!(reduced_off.allows(&trace("Align roll [deg]")));
        assert!(reduced_off.allows(&trace("Full mount roll [deg]")));
    }
}
