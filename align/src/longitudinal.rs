#[derive(Debug, Clone, Copy)]
pub struct LongitudinalCueConfig {
    pub alpha: f32,
    pub min_abs_long_mps2: f32,
    pub min_sign_stable_windows: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LongitudinalCueSample {
    pub gnss_long_mps2: f32,
    pub imu_long_mps2: f32,
    pub imu_lat_mps2: f32,
    pub base_valid: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LongitudinalCue {
    pub z_long_mps2: f32,
    pub h_long_mps2: f32,
    pub h_yaw_jac: f32,
}

#[derive(Debug, Clone)]
pub struct LongitudinalCueFilter {
    gnss_long_lp: Option<f32>,
    imu_long_lp: Option<f32>,
    imu_lat_lp: Option<f32>,
    last_sign: i8,
    stable_windows: usize,
}

impl Default for LongitudinalCueFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl LongitudinalCueFilter {
    pub fn new() -> Self {
        Self {
            gnss_long_lp: None,
            imu_long_lp: None,
            imu_lat_lp: None,
            last_sign: 0,
            stable_windows: 0,
        }
    }

    pub fn reset(&mut self) {
        self.gnss_long_lp = None;
        self.imu_long_lp = None;
        self.imu_lat_lp = None;
        self.last_sign = 0;
        self.stable_windows = 0;
    }

    pub fn update(
        &mut self,
        cfg: LongitudinalCueConfig,
        sample: LongitudinalCueSample,
    ) -> Option<LongitudinalCue> {
        let alpha = cfg.alpha.clamp(0.0, 1.0);
        let z_lp = lpf_step(&mut self.gnss_long_lp, sample.gnss_long_mps2, alpha);
        let h_lp = lpf_step(&mut self.imu_long_lp, sample.imu_long_mps2, alpha);
        let h_yaw_lp = lpf_step(&mut self.imu_lat_lp, sample.imu_lat_mps2, alpha);

        if !sample.base_valid {
            self.last_sign = 0;
            self.stable_windows = 0;
            return None;
        }

        let sign = signed_region(z_lp, cfg.min_abs_long_mps2);
        if sign == 0 {
            self.last_sign = 0;
            self.stable_windows = 0;
            return None;
        }

        if sign == self.last_sign {
            self.stable_windows += 1;
        } else {
            self.last_sign = sign;
            self.stable_windows = 1;
        }

        if self.stable_windows < cfg.min_sign_stable_windows.max(1) {
            return None;
        }

        Some(LongitudinalCue {
            z_long_mps2: z_lp,
            h_long_mps2: h_lp,
            h_yaw_jac: h_yaw_lp,
        })
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

fn signed_region(x: f32, deadband: f32) -> i8 {
    if x > deadband {
        1
    } else if x < -deadband {
        -1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LongitudinalCueConfig, LongitudinalCueFilter, LongitudinalCueSample,
    };

    #[test]
    fn requires_sign_persistence_before_emitting() {
        let cfg = LongitudinalCueConfig {
            alpha: 1.0,
            min_abs_long_mps2: 0.25,
            min_sign_stable_windows: 3,
        };
        let mut filter = LongitudinalCueFilter::new();
        for _ in 0..2 {
            let cue = filter.update(
                cfg,
                LongitudinalCueSample {
                    gnss_long_mps2: 0.8,
                    imu_long_mps2: 0.7,
                    imu_lat_mps2: 0.2,
                    base_valid: true,
                },
            );
            assert!(cue.is_none());
        }
        let cue = filter.update(
            cfg,
            LongitudinalCueSample {
                gnss_long_mps2: 0.8,
                imu_long_mps2: 0.7,
                imu_lat_mps2: 0.2,
                base_valid: true,
            },
        );
        assert!(cue.is_some());
    }

    #[test]
    fn suppresses_near_zero_filtered_longitudinal_signal() {
        let cfg = LongitudinalCueConfig {
            alpha: 1.0,
            min_abs_long_mps2: 0.25,
            min_sign_stable_windows: 1,
        };
        let mut filter = LongitudinalCueFilter::new();
        let cue = filter.update(
            cfg,
            LongitudinalCueSample {
                gnss_long_mps2: 0.1,
                imu_long_mps2: 0.1,
                imu_lat_mps2: 0.0,
                base_valid: true,
            },
        );
        assert!(cue.is_none());
    }

    #[test]
    fn invalid_window_resets_sign_stability() {
        let cfg = LongitudinalCueConfig {
            alpha: 1.0,
            min_abs_long_mps2: 0.25,
            min_sign_stable_windows: 2,
        };
        let mut filter = LongitudinalCueFilter::new();
        let _ = filter.update(
            cfg,
            LongitudinalCueSample {
                gnss_long_mps2: 0.8,
                imu_long_mps2: 0.7,
                imu_lat_mps2: 0.2,
                base_valid: true,
            },
        );
        let _ = filter.update(
            cfg,
            LongitudinalCueSample {
                gnss_long_mps2: 0.8,
                imu_long_mps2: 0.7,
                imu_lat_mps2: 0.2,
                base_valid: false,
            },
        );
        let cue = filter.update(
            cfg,
            LongitudinalCueSample {
                gnss_long_mps2: 0.8,
                imu_long_mps2: 0.7,
                imu_lat_mps2: 0.2,
                base_valid: true,
            },
        );
        assert!(cue.is_none());
    }
}
