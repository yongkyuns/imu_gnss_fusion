#[derive(Debug, Clone, Copy)]
pub struct YawPcaConfig {
    pub enabled: bool,
    pub min_speed_mps: f32,
    pub min_horiz_acc_mps2: f32,
    pub min_long_mps2: f32,
    pub max_lat_to_long_ratio: f32,
    pub min_abs_lat_guard_mps2: f32,
    pub min_windows: usize,
    pub max_windows: usize,
    pub min_anisotropy_ratio: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct YawPcaSample {
    pub speed_mps: f32,
    pub horiz_accel_xy: [f32; 2],
    pub gnss_long_mps2: f32,
    pub gnss_lat_mps2: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct StoredSample {
    horiz_accel_v: [f32; 2],
    gnss_long_mps2: f32,
}

#[derive(Debug, Clone)]
pub struct YawPcaInitializer {
    active: bool,
    resolved: bool,
    samples: Vec<StoredSample>,
}

impl Default for YawPcaInitializer {
    fn default() -> Self {
        Self::new()
    }
}

impl YawPcaInitializer {
    pub fn new() -> Self {
        Self {
            active: false,
            resolved: false,
            samples: Vec::new(),
        }
    }

    pub fn reset(&mut self, enabled: bool) {
        self.active = enabled;
        self.resolved = false;
        self.samples.clear();
    }

    pub fn is_active(&self) -> bool {
        self.active && !self.resolved
    }

    pub fn update(&mut self, cfg: YawPcaConfig, sample: YawPcaSample) -> Option<f32> {
        if !cfg.enabled || !self.is_active() {
            return None;
        }
        if sample.speed_mps < cfg.min_speed_mps
            || vec2_norm(sample.horiz_accel_xy) < cfg.min_horiz_acc_mps2
            || sample.gnss_long_mps2.abs() < cfg.min_long_mps2
        {
            return None;
        }
        let max_lat_abs = cfg
            .min_abs_lat_guard_mps2
            .max(cfg.max_lat_to_long_ratio * sample.gnss_long_mps2.abs());
        if sample.gnss_lat_mps2.abs() > max_lat_abs {
            return None;
        }
        self.samples.push(StoredSample {
            horiz_accel_v: sample.horiz_accel_xy,
            gnss_long_mps2: sample.gnss_long_mps2,
        });
        if self.samples.len() < cfg.min_windows {
            return None;
        }
        let (theta, anisotropy) = principal_axis_angle(&self.samples)?;
        if anisotropy < cfg.min_anisotropy_ratio && self.samples.len() < cfg.max_windows {
            return None;
        }
        let mut axis = [theta.cos(), theta.sin()];
        let corr = self
            .samples
            .iter()
            .map(|s| s.gnss_long_mps2 * dot2(s.horiz_accel_v, axis))
            .sum::<f32>();
        let mut dpsi = theta;
        if corr < 0.0 {
            dpsi = wrap_pi(dpsi + std::f32::consts::PI);
            axis = [-axis[0], -axis[1]];
            let _ = axis;
        }
        self.active = false;
        self.resolved = true;
        Some(wrap_pi(dpsi))
    }
}

fn principal_axis_angle(samples: &[StoredSample]) -> Option<(f32, f32)> {
    if samples.is_empty() {
        return None;
    }
    let mut sxx = 0.0_f32;
    let mut sxy = 0.0_f32;
    let mut syy = 0.0_f32;
    for s in samples {
        let x = s.horiz_accel_v[0];
        let y = s.horiz_accel_v[1];
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }
    let trace = sxx + syy;
    let disc = ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt();
    let lambda_max = 0.5 * (trace + disc);
    let lambda_min = 0.5 * (trace - disc).max(0.0);
    if lambda_max <= 1.0e-6 {
        return None;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let anisotropy = lambda_max / lambda_min.max(1.0e-6);
    Some((theta, anisotropy))
}

fn vec2_norm(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn dot2(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn wrap_pi(x: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    (x + std::f32::consts::PI).rem_euclid(two_pi) - std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_principal_axis_and_sign() {
        let cfg = YawPcaConfig {
            enabled: true,
            min_speed_mps: 2.0,
            min_horiz_acc_mps2: 0.1,
            min_long_mps2: 0.1,
            max_lat_to_long_ratio: 0.6,
            min_abs_lat_guard_mps2: 0.3,
            min_windows: 4,
            max_windows: 8,
            min_anisotropy_ratio: 1.2,
        };
        let theta = 60.0_f32.to_radians();
        let axis = [theta.cos(), theta.sin()];
        let mags = [1.0_f32, -0.7, 1.3, -0.9];
        let mut init = YawPcaInitializer::new();
        init.reset(true);
        let mut out = None;
        for m in mags {
            out = init.update(
                cfg,
                YawPcaSample {
                    speed_mps: 8.0,
                    horiz_accel_xy: [m * axis[0], m * axis[1]],
                    gnss_long_mps2: m,
                    gnss_lat_mps2: 0.0,
                },
            );
        }
        let dpsi = out.expect("pca should resolve");
        assert!((wrap_pi(dpsi - theta)).abs() < 5.0_f32.to_radians());
    }

    #[test]
    fn flips_axis_when_correlation_is_negative() {
        let cfg = YawPcaConfig {
            enabled: true,
            min_speed_mps: 2.0,
            min_horiz_acc_mps2: 0.1,
            min_long_mps2: 0.1,
            max_lat_to_long_ratio: 0.6,
            min_abs_lat_guard_mps2: 0.3,
            min_windows: 4,
            max_windows: 8,
            min_anisotropy_ratio: 1.2,
        };
        let theta = 35.0_f32.to_radians();
        let axis = [theta.cos(), theta.sin()];
        let mags = [1.0_f32, -0.8, 1.1, -1.2];
        let mut init = YawPcaInitializer::new();
        init.reset(true);
        let mut out = None;
        for m in mags {
            out = init.update(
                cfg,
                YawPcaSample {
                    speed_mps: 8.0,
                    horiz_accel_xy: [m * axis[0], m * axis[1]],
                    gnss_long_mps2: -m,
                    gnss_lat_mps2: 0.0,
                },
            );
        }
        let dpsi = out.expect("pca should resolve");
        let expected = wrap_pi(theta + std::f32::consts::PI);
        assert!((wrap_pi(dpsi - expected)).abs() < 5.0_f32.to_radians());
    }

    #[test]
    fn rejects_lateral_dominant_samples() {
        let cfg = YawPcaConfig {
            enabled: true,
            min_speed_mps: 2.0,
            min_horiz_acc_mps2: 0.1,
            min_long_mps2: 0.1,
            max_lat_to_long_ratio: 0.6,
            min_abs_lat_guard_mps2: 0.3,
            min_windows: 4,
            max_windows: 8,
            min_anisotropy_ratio: 1.2,
        };
        let mut init = YawPcaInitializer::new();
        init.reset(true);
        let mut out = None;
        for _ in 0..6 {
            out = init.update(
                cfg,
                YawPcaSample {
                    speed_mps: 8.0,
                    horiz_accel_xy: [0.3, 1.0],
                    gnss_long_mps2: 0.2,
                    gnss_lat_mps2: 1.0,
                },
            );
        }
        assert!(out.is_none());
        assert!(init.is_active());
    }
}
