#[derive(Debug, Clone, Copy)]
pub struct HorizontalHeadingCueConfig {
    pub alpha: f32,
    pub min_abs_horiz_mps2: f32,
    pub min_stable_windows: usize,
    pub max_lat_to_long_ratio: f32,
    pub min_abs_lat_guard_mps2: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HorizontalHeadingCueSample {
    pub gnss_horiz_mps2: [f32; 2],
    pub imu_horiz_mps2: [f32; 2],
    pub base_valid: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HorizontalHeadingCue {
    pub angle_err_rad: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HorizontalHeadingTrace {
    pub gnss_long_lp_mps2: f32,
    pub gnss_lat_lp_mps2: f32,
    pub imu_long_lp_mps2: f32,
    pub imu_lat_lp_mps2: f32,
    pub angle_err_rad: f32,
    pub base_valid: bool,
    pub stable_windows: usize,
    pub emitted: bool,
}

#[derive(Debug, Clone)]
pub struct HorizontalHeadingCueFilter {
    gnss_long_lp: Option<f32>,
    gnss_lat_lp: Option<f32>,
    imu_long_lp: Option<f32>,
    imu_lat_lp: Option<f32>,
    stable_windows: usize,
}

impl Default for HorizontalHeadingCueFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl HorizontalHeadingCueFilter {
    pub fn new() -> Self {
        Self {
            gnss_long_lp: None,
            gnss_lat_lp: None,
            imu_long_lp: None,
            imu_lat_lp: None,
            stable_windows: 0,
        }
    }

    pub fn reset(&mut self) {
        self.gnss_long_lp = None;
        self.gnss_lat_lp = None;
        self.imu_long_lp = None;
        self.imu_lat_lp = None;
        self.stable_windows = 0;
    }

    pub fn update_with_trace(
        &mut self,
        cfg: HorizontalHeadingCueConfig,
        sample: HorizontalHeadingCueSample,
    ) -> (Option<HorizontalHeadingCue>, HorizontalHeadingTrace) {
        let alpha = cfg.alpha.clamp(0.0, 1.0);
        let gnss_long_lp = lpf_step(&mut self.gnss_long_lp, sample.gnss_horiz_mps2[0], alpha);
        let gnss_lat_lp = lpf_step(&mut self.gnss_lat_lp, sample.gnss_horiz_mps2[1], alpha);
        let imu_long_lp = lpf_step(&mut self.imu_long_lp, sample.imu_horiz_mps2[0], alpha);
        let imu_lat_lp = lpf_step(&mut self.imu_lat_lp, sample.imu_horiz_mps2[1], alpha);

        let mut trace = HorizontalHeadingTrace {
            gnss_long_lp_mps2: gnss_long_lp,
            gnss_lat_lp_mps2: gnss_lat_lp,
            imu_long_lp_mps2: imu_long_lp,
            imu_lat_lp_mps2: imu_lat_lp,
            base_valid: sample.base_valid,
            ..HorizontalHeadingTrace::default()
        };

        if !sample.base_valid {
            self.stable_windows = 0;
            return (None, trace);
        }

        let gnss = [gnss_long_lp, gnss_lat_lp];
        let imu = [imu_long_lp, imu_lat_lp];
        let gnss_norm = vec2_norm(gnss);
        let imu_norm = vec2_norm(imu);
        if gnss_norm < cfg.min_abs_horiz_mps2 || imu_norm < cfg.min_abs_horiz_mps2 {
            self.stable_windows = 0;
            return (None, trace);
        }

        let gnss_long_abs = gnss_long_lp.abs();
        let gnss_lat_abs = gnss_lat_lp.abs();
        let max_lat_abs = cfg
            .min_abs_lat_guard_mps2
            .max(cfg.max_lat_to_long_ratio * gnss_long_abs);
        if gnss_long_abs < cfg.min_abs_horiz_mps2 || gnss_lat_abs > max_lat_abs {
            self.stable_windows = 0;
            return (None, trace);
        }

        self.stable_windows += 1;
        trace.stable_windows = self.stable_windows;
        if self.stable_windows < cfg.min_stable_windows.max(1) {
            return (None, trace);
        }

        let cross = imu[0] * gnss[1] - imu[1] * gnss[0];
        let dot = imu[0] * gnss[0] + imu[1] * gnss[1];
        let angle_err = cross.atan2(dot);
        trace.angle_err_rad = angle_err;
        trace.emitted = true;
        (
            Some(HorizontalHeadingCue {
                angle_err_rad: angle_err,
            }),
            trace,
        )
    }
}

fn lpf_step(state: &mut Option<f32>, x: f32, alpha: f32) -> f32 {
    match state {
        Some(s) => {
            *s = (1.0 - alpha) * *s + alpha * x;
            *s
        }
        None => {
            *state = Some(x);
            x
        }
    }
}

fn vec2_norm(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{
        HorizontalHeadingCueConfig, HorizontalHeadingCueFilter, HorizontalHeadingCueSample,
    };

    #[test]
    fn requires_stable_windows() {
        let cfg = HorizontalHeadingCueConfig {
            alpha: 1.0,
            min_abs_horiz_mps2: 0.25,
            min_stable_windows: 2,
            max_lat_to_long_ratio: 0.6,
            min_abs_lat_guard_mps2: 0.5,
        };
        let mut filter = HorizontalHeadingCueFilter::new();
        let sample = HorizontalHeadingCueSample {
            gnss_horiz_mps2: [1.0, 0.0],
            imu_horiz_mps2: [0.0, 1.0],
            base_valid: true,
        };
        assert!(filter.update_with_trace(cfg, sample).0.is_none());
        assert!(filter.update_with_trace(cfg, sample).0.is_some());
    }

    #[test]
    fn computes_signed_angle_error() {
        let cfg = HorizontalHeadingCueConfig {
            alpha: 1.0,
            min_abs_horiz_mps2: 0.25,
            min_stable_windows: 1,
            max_lat_to_long_ratio: 0.6,
            min_abs_lat_guard_mps2: 0.5,
        };
        let mut filter = HorizontalHeadingCueFilter::new();
        let cue = filter
            .update_with_trace(
                cfg,
                HorizontalHeadingCueSample {
                    gnss_horiz_mps2: [1.0, 0.0],
                    imu_horiz_mps2: [0.0, 1.0],
                    base_valid: true,
                },
            )
            .0
            .unwrap();
        assert!((cue.angle_err_rad + std::f32::consts::FRAC_PI_2).abs() < 1.0e-5);
    }

    #[test]
    fn rejects_filtered_lateral_dominant_gnss_vector() {
        let cfg = HorizontalHeadingCueConfig {
            alpha: 1.0,
            min_abs_horiz_mps2: 0.25,
            min_stable_windows: 1,
            max_lat_to_long_ratio: 0.6,
            min_abs_lat_guard_mps2: 0.5,
        };
        let mut filter = HorizontalHeadingCueFilter::new();
        let sample = HorizontalHeadingCueSample {
            gnss_horiz_mps2: [0.05, 0.40],
            imu_horiz_mps2: [0.30, 0.10],
            base_valid: true,
        };
        assert!(filter.update_with_trace(cfg, sample).0.is_none());
    }
}
