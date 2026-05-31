use crate::datasets::generic_replay::{GenericGnssSample, GenericImuSample};

pub enum ReplayEvent<'a> {
    Imu(usize, &'a GenericImuSample),
    Gnss(usize, &'a GenericGnssSample),
}

pub fn for_each_event(
    imu_samples: &[GenericImuSample],
    gnss_samples: &[GenericGnssSample],
    mut f: impl FnMut(ReplayEvent<'_>),
) {
    let mut imu_idx = 0usize;
    let mut gnss_idx = 0usize;
    while imu_idx < imu_samples.len() || gnss_idx < gnss_samples.len() {
        let next_imu_t = imu_samples.get(imu_idx).map(|s| s.t_s);
        let next_gnss_t = gnss_samples.get(gnss_idx).map(|s| s.t_s);
        let take_imu = match (next_imu_t, next_gnss_t) {
            (Some(ti), Some(tg)) => ti <= tg,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        if take_imu {
            f(ReplayEvent::Imu(imu_idx, &imu_samples[imu_idx]));
            imu_idx += 1;
        } else {
            f(ReplayEvent::Gnss(gnss_idx, &gnss_samples[gnss_idx]));
            gnss_idx += 1;
        }
    }
}
