#[derive(Debug, Clone, Copy)]
pub struct InitMonitorConfig {
    pub warmup_windows: usize,
    pub min_heading_valid_windows: usize,
    pub min_reinit_persistence: usize,
    pub heading_ema_alpha: f32,
    pub heading_var_alpha: f32,
    pub heading_reinit_angle_rad: f32,
    pub heading_reinit_std_rad: f32,
    pub gravity_ema_alpha: f32,
    pub gravity_reinit_leak_mps2: f32,
    pub course_ema_alpha: f32,
    pub course_reinit_residual_radps: f32,
    pub require_course_confirmation: bool,
}

impl Default for InitMonitorConfig {
    fn default() -> Self {
        Self {
            warmup_windows: 8,
            min_heading_valid_windows: 6,
            min_reinit_persistence: 4,
            heading_ema_alpha: 0.12,
            heading_var_alpha: 0.12,
            heading_reinit_angle_rad: 20.0_f32.to_radians(),
            heading_reinit_std_rad: 8.0_f32.to_radians(),
            gravity_ema_alpha: 0.12,
            gravity_reinit_leak_mps2: 0.6,
            course_ema_alpha: 0.12,
            course_reinit_residual_radps: 8.0_f32.to_radians(),
            require_course_confirmation: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct InitMonitorSample {
    pub heading_valid: bool,
    pub heading_residual_rad: f32,
    pub gravity_valid: bool,
    pub gravity_leak_mps2: f32,
    pub course_valid: bool,
    pub course_residual_radps: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct InitMonitorTrace {
    pub total_windows: usize,
    pub heading_valid_windows: usize,
    pub heading_residual_ema_rad: f32,
    pub heading_residual_std_rad: f32,
    pub gravity_leak_ema_mps2: f32,
    pub course_residual_ema_radps: f32,
    pub reinit_candidate: bool,
    pub should_reinitialize: bool,
}

#[derive(Debug, Clone)]
pub struct InitMonitor {
    total_windows: usize,
    heading_valid_windows: usize,
    heading_ema: Option<f32>,
    heading_sq_ema: Option<f32>,
    gravity_ema: Option<f32>,
    course_ema: Option<f32>,
    persistence: usize,
}

impl Default for InitMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl InitMonitor {
    pub fn new() -> Self {
        Self {
            total_windows: 0,
            heading_valid_windows: 0,
            heading_ema: None,
            heading_sq_ema: None,
            gravity_ema: None,
            course_ema: None,
            persistence: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn update(
        &mut self,
        cfg: InitMonitorConfig,
        sample: InitMonitorSample,
    ) -> InitMonitorTrace {
        self.total_windows += 1;
        if sample.heading_valid {
            self.heading_valid_windows += 1;
            let abs_res = sample.heading_residual_rad.abs();
            self.heading_ema = Some(ema_step(self.heading_ema, abs_res, cfg.heading_ema_alpha));
            self.heading_sq_ema = Some(ema_step(
                self.heading_sq_ema,
                abs_res * abs_res,
                cfg.heading_var_alpha,
            ));
        }
        if sample.gravity_valid {
            self.gravity_ema = Some(ema_step(
                self.gravity_ema,
                sample.gravity_leak_mps2.abs(),
                cfg.gravity_ema_alpha,
            ));
        }
        if sample.course_valid {
            self.course_ema = Some(ema_step(
                self.course_ema,
                sample.course_residual_radps.abs(),
                cfg.course_ema_alpha,
            ));
        }

        let heading_mean = self.heading_ema.unwrap_or(0.0);
        let heading_var = (self.heading_sq_ema.unwrap_or(0.0) - heading_mean * heading_mean).max(0.0);
        let heading_std = heading_var.sqrt();
        let gravity_leak = self.gravity_ema.unwrap_or(0.0);
        let course_res = self.course_ema.unwrap_or(0.0);

        let warm = self.total_windows >= cfg.warmup_windows;
        let enough_heading = self.heading_valid_windows >= cfg.min_heading_valid_windows;
        let heading_bad =
            heading_mean >= cfg.heading_reinit_angle_rad && heading_std <= cfg.heading_reinit_std_rad;
        let gravity_bad = gravity_leak >= cfg.gravity_reinit_leak_mps2;
        let course_bad = course_res >= cfg.course_reinit_residual_radps;

        let candidate = warm
            && enough_heading
            && heading_bad
            && (gravity_bad || !cfg.require_course_confirmation || course_bad);

        if candidate {
            self.persistence += 1;
        } else {
            self.persistence = 0;
        }

        InitMonitorTrace {
            total_windows: self.total_windows,
            heading_valid_windows: self.heading_valid_windows,
            heading_residual_ema_rad: heading_mean,
            heading_residual_std_rad: heading_std,
            gravity_leak_ema_mps2: gravity_leak,
            course_residual_ema_radps: course_res,
            reinit_candidate: candidate,
            should_reinitialize: candidate && self.persistence >= cfg.min_reinit_persistence,
        }
    }
}

fn ema_step(state: Option<f32>, x: f32, alpha: f32) -> f32 {
    match state {
        Some(s) => (1.0 - alpha) * s + alpha * x,
        None => x,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn does_not_trigger_on_short_transient() {
        let cfg = InitMonitorConfig {
            warmup_windows: 0,
            min_heading_valid_windows: 2,
            min_reinit_persistence: 3,
            ..InitMonitorConfig::default()
        };
        let mut mon = InitMonitor::new();
        let mut fired = false;
        for _ in 0..2 {
            let tr = mon.update(
                cfg,
                InitMonitorSample {
                    heading_valid: true,
                    heading_residual_rad: 30.0_f32.to_radians(),
                    gravity_valid: true,
                    gravity_leak_mps2: 1.0,
                    ..InitMonitorSample::default()
                },
            );
            fired |= tr.should_reinitialize;
        }
        assert!(!fired);
    }

    #[test]
    fn triggers_on_persistent_large_heading_and_gravity_leak() {
        let cfg = InitMonitorConfig {
            warmup_windows: 0,
            min_heading_valid_windows: 2,
            min_reinit_persistence: 3,
            ..InitMonitorConfig::default()
        };
        let mut mon = InitMonitor::new();
        let mut fired = false;
        for _ in 0..8 {
            let tr = mon.update(
                cfg,
                InitMonitorSample {
                    heading_valid: true,
                    heading_residual_rad: 28.0_f32.to_radians(),
                    gravity_valid: true,
                    gravity_leak_mps2: 1.2,
                    ..InitMonitorSample::default()
                },
            );
            fired |= tr.should_reinitialize;
        }
        assert!(fired);
    }
}
