mod ekf_compare;
mod signals;
mod tag_time;
mod timebase;
mod vma_compare;

use crate::ubxlog::parse_ubx_frames;

use super::model::PlotData;
use ekf_compare::build_ekf_compare_traces;
use signals::build_signal_traces;
use timebase::build_master_timeline;
use vma_compare::build_vma_compare_traces;

pub fn build_plot_data(bytes: &[u8], max_records: Option<usize>) -> (PlotData, bool) {
    let frames = parse_ubx_frames(bytes, max_records);
    let timeline = build_master_timeline(&frames);
    let ekf_data = build_ekf_compare_traces(&frames, &timeline);
    let vma_data = build_vma_compare_traces(&frames, &timeline);
    let out = build_signal_traces(&frames, &timeline, ekf_data, vma_data);
    (out, timeline.has_itow)
}
