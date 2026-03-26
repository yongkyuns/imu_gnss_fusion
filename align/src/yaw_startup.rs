#[derive(Debug, Clone, Copy)]
pub struct YawStartupConfig {
    pub enabled: bool,
    pub alpha: f32,
    pub min_speed_mps: f32,
    pub min_horiz_acc_mps2: f32,
    pub min_long_mps2: f32,
    pub max_lat_to_long_ratio: f32,
    pub min_abs_lat_guard_mps2: f32,
    pub max_course_rate_radps: f32,
    pub min_stable_windows: usize,
    pub min_windows: usize,
    pub max_windows: usize,
    pub min_alignment_score: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct YawStartupSample {
    pub speed_mps: f32,
    pub course_rate_radps: f32,
    pub horiz_accel_xy: [f32; 2],
    pub gnss_long_mps2: f32,
    pub gnss_lat_mps2: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct YawStartupTrace {
    pub gnss_long_lp_mps2: f32,
    pub gnss_lat_lp_mps2: f32,
    pub imu_long_lp_mps2: f32,
    pub imu_lat_lp_mps2: f32,
    pub gate_valid: bool,
    pub accepted: bool,
    pub stable_windows: usize,
    pub sample_count: usize,
    pub alignment_score: f32,
    pub emitted: bool,
    pub emitted_theta_rad: Option<f32>,
}

#[derive(Debug, Clone, Copy, Default)]
struct StoredSample {
    imu_horiz_xy: [f32; 2],
    gnss_horiz_xy: [f32; 2],
}

#[derive(Debug, Clone)]
pub struct YawStartupInitializer {
    active: bool,
    resolved: bool,
    samples: Vec<StoredSample>,
    total_windows: usize,
    gnss_long_lp: Option<f32>,
    gnss_lat_lp: Option<f32>,
    imu_x_lp: Option<f32>,
    imu_y_lp: Option<f32>,
    stable_windows: usize,
}

impl Default for YawStartupInitializer {
    fn default() -> Self {
        Self::new()
    }
}

impl YawStartupInitializer {
    pub fn new() -> Self {
        Self {
            active: false,
            resolved: false,
            samples: Vec::new(),
            total_windows: 0,
            gnss_long_lp: None,
            gnss_lat_lp: None,
            imu_x_lp: None,
            imu_y_lp: None,
            stable_windows: 0,
        }
    }

    pub fn reset(&mut self, enabled: bool) {
        self.active = enabled;
        self.resolved = false;
        self.samples.clear();
        self.total_windows = 0;
        self.gnss_long_lp = None;
        self.gnss_lat_lp = None;
        self.imu_x_lp = None;
        self.imu_y_lp = None;
        self.stable_windows = 0;
    }

    pub fn is_active(&self) -> bool {
        self.active && !self.resolved
    }

    pub fn update(&mut self, cfg: YawStartupConfig, sample: YawStartupSample) -> Option<f32> {
        self.update_with_trace(cfg, sample).0
    }

    pub fn update_with_trace(
        &mut self,
        cfg: YawStartupConfig,
        sample: YawStartupSample,
    ) -> (Option<f32>, YawStartupTrace) {
        let mut trace = YawStartupTrace::default();
        if !cfg.enabled || !self.is_active() {
            return (None, trace);
        }
        self.total_windows += 1;

        let alpha = cfg.alpha.clamp(0.0, 1.0);
        let gnss_long_lp = lpf_step(&mut self.gnss_long_lp, sample.gnss_long_mps2, alpha);
        let gnss_lat_lp = lpf_step(&mut self.gnss_lat_lp, sample.gnss_lat_mps2, alpha);
        let imu_x_lp = lpf_step(&mut self.imu_x_lp, sample.horiz_accel_xy[0], alpha);
        let imu_y_lp = lpf_step(&mut self.imu_y_lp, sample.horiz_accel_xy[1], alpha);
        trace.gnss_long_lp_mps2 = gnss_long_lp;
        trace.gnss_lat_lp_mps2 = gnss_lat_lp;
        trace.imu_long_lp_mps2 = imu_x_lp;
        trace.imu_lat_lp_mps2 = imu_y_lp;

        if sample.speed_mps < cfg.min_speed_mps
            || sample.course_rate_radps.abs() > cfg.max_course_rate_radps
            || vec2_norm([imu_x_lp, imu_y_lp]) < cfg.min_horiz_acc_mps2
            || vec2_norm([gnss_long_lp, gnss_lat_lp]) < cfg.min_horiz_acc_mps2
            || gnss_long_lp.abs() < cfg.min_long_mps2
        {
            self.stable_windows = 0;
            self.samples.clear();
            return (None, trace);
        }
        let max_lat_abs = cfg
            .min_abs_lat_guard_mps2
            .max(cfg.max_lat_to_long_ratio * gnss_long_lp.abs());
        if gnss_lat_lp.abs() > max_lat_abs {
            self.stable_windows = 0;
            self.samples.clear();
            return (None, trace);
        }
        trace.gate_valid = true;

        self.stable_windows += 1;
        trace.stable_windows = self.stable_windows;
        if self.stable_windows < cfg.min_stable_windows.max(1) {
            self.samples.clear();
            return (None, trace);
        }

        self.samples.push(StoredSample {
            imu_horiz_xy: [imu_x_lp, imu_y_lp],
            gnss_horiz_xy: [gnss_long_lp, gnss_lat_lp],
        });
        trace.accepted = true;
        trace.sample_count = self.samples.len();

        if self.samples.len() < cfg.min_windows {
            return (None, trace);
        }

        let Some((theta, alignment_score)) = best_rotation_angle(&self.samples) else {
            return (None, trace);
        };
        trace.alignment_score = alignment_score;
        if alignment_score < cfg.min_alignment_score && self.samples.len() < cfg.max_windows {
            return (None, trace);
        }

        self.active = false;
        self.resolved = true;
        trace.emitted = true;
        let theta = wrap_pi(theta);
        trace.emitted_theta_rad = Some(theta);
        (Some(theta), trace)
    }

    pub fn timed_out(&self, cfg: YawStartupConfig) -> bool {
        self.is_active() && self.total_windows >= cfg.max_windows
    }
}

fn best_rotation_angle(samples: &[StoredSample]) -> Option<(f32, f32)> {
    if samples.is_empty() {
        return None;
    }
    let mut cross_sum = 0.0_f32;
    let mut dot_sum = 0.0_f32;
    let mut total = 0.0_f32;
    for s in samples {
        let h = s.imu_horiz_xy;
        let g = s.gnss_horiz_xy;
        cross_sum += h[0] * g[1] - h[1] * g[0];
        dot_sum += h[0] * g[0] + h[1] * g[1];
        total += vec2_norm(h) * vec2_norm(g);
    }
    let score = (cross_sum * cross_sum + dot_sum * dot_sum).sqrt();
    if score <= 1.0e-6 || total <= 1.0e-6 {
        return None;
    }
    Some((cross_sum.atan2(dot_sum), score / total))
}

fn vec2_norm(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
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

fn wrap_pi(x: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    (x + std::f32::consts::PI).rem_euclid(two_pi) - std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_axis_and_branch_together() {
        let cfg = YawStartupConfig {
            enabled: true,
            alpha: 1.0,
            min_speed_mps: 2.0,
            min_horiz_acc_mps2: 0.1,
            min_long_mps2: 0.1,
            max_lat_to_long_ratio: 0.6,
            min_abs_lat_guard_mps2: 0.3,
            max_course_rate_radps: 10.0_f32.to_radians(),
            min_stable_windows: 1,
            min_windows: 4,
            max_windows: 8,
            min_alignment_score: 0.8,
        };
        let theta = 30.0_f32.to_radians();
        let imu_axis = [1.0_f32, 0.0];
        let mags = [1.0_f32, 0.7, 1.3, 0.9];
        let mut init = YawStartupInitializer::new();
        init.reset(true);
        let mut out = None;
        for m in mags {
            let gnss = [
                m * (theta.cos() * imu_axis[0] - theta.sin() * imu_axis[1]),
                m * (theta.sin() * imu_axis[0] + theta.cos() * imu_axis[1]),
            ];
            out = init.update(
                cfg,
                YawStartupSample {
                    speed_mps: 8.0,
                    course_rate_radps: 0.0,
                    horiz_accel_xy: [m * imu_axis[0], m * imu_axis[1]],
                    gnss_long_mps2: gnss[0],
                    gnss_lat_mps2: gnss[1],
                },
            );
        }
        let dpsi = out.expect("startup should resolve");
        assert!((wrap_pi(dpsi - theta)).abs() < 5.0_f32.to_radians());
    }

    #[test]
    fn rejects_inconsistent_alignment() {
        let cfg = YawStartupConfig {
            enabled: true,
            alpha: 1.0,
            min_speed_mps: 2.0,
            min_horiz_acc_mps2: 0.1,
            min_long_mps2: 0.1,
            max_lat_to_long_ratio: 0.6,
            min_abs_lat_guard_mps2: 0.3,
            max_course_rate_radps: 10.0_f32.to_radians(),
            min_stable_windows: 1,
            min_windows: 4,
            max_windows: 8,
            min_alignment_score: 0.8,
        };
        let mut init = YawStartupInitializer::new();
        init.reset(true);
        let mut out = None;
        for g in [[1.0_f32, 0.0_f32], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]] {
            out = init.update(
                cfg,
                YawStartupSample {
                    speed_mps: 8.0,
                    course_rate_radps: 0.0,
                    horiz_accel_xy: [1.0, 0.0],
                    gnss_long_mps2: g[0],
                    gnss_lat_mps2: g[1],
                },
            );
        }
        assert!(out.is_none());
        assert!(init.is_active());
    }
}
