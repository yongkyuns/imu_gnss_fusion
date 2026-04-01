mod align_compare;
pub mod align_replay;
pub mod ekf_compare;
mod signals;
mod tag_time;
pub mod timebase;

use crate::ubxlog::parse_ubx_frames;

use super::model::{EkfImuSource, PlotData};
use crate::visualizer::pipeline::align_replay::ImuReplayConfig;
use align_compare::build_align_compare_traces;
use ekf_compare::{EkfCompareConfig, GnssOutageConfig, build_ekf_compare_traces};
use signals::build_signal_traces;
use timebase::build_master_timeline;

pub fn build_plot_data(
    bytes: &[u8],
    max_records: Option<usize>,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
) -> (PlotData, bool) {
    let frames = parse_ubx_frames(bytes, max_records);
    let timeline = build_master_timeline(&frames);
    let ekf_data =
        build_ekf_compare_traces(&frames, &timeline, ekf_imu_source, ekf_cfg, gnss_outages);
    let align_data = build_align_compare_traces(&frames, &timeline, ImuReplayConfig::default());
    let out = build_signal_traces(&frames, &timeline, ekf_data, align_data);
    (out, timeline.has_itow)
}
