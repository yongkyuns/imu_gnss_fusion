//! Shared UI state enums and trace visibility classification.

use crate::visualizer::model::Trace;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DataOrigin {
    Real,
    Synthetic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TuningPanel {
    Eskf,
    Align,
    Loose,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TraceVisibility {
    pub show_reference: bool,
    pub show_align: bool,
    pub show_eskf: bool,
    pub show_loose: bool,
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
        if !self.show_eskf && is_eskf_trace_name(name) {
            return false;
        }
        if !self.show_loose && is_loose_trace_name(name) {
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

fn is_eskf_trace_name(name: &str) -> bool {
    name.starts_with("ESKF")
        || name.contains(" ESKF")
        || name.contains("eskf")
        || name.contains("EKF initialized")
        || name.contains("ekf initialized")
        || name.contains("mount ready")
}

fn is_loose_trace_name(name: &str) -> bool {
    name.starts_with("Loose")
        || name.contains(" Loose")
        || name.contains("loose")
        || name.contains("residual mount")
}
