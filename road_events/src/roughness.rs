use crate::common::{high_pass, low_pass, sqrt_f32};
use crate::{RoadRoughnessConfig, RoadRoughnessEstimate, RoadRoughnessLevel, RoadRoughnessSample};

/// Streaming road-surface roughness analyzer.
///
/// Roughness is formulated as distance-normalized, band-limited vertical
/// acceleration energy:
///
/// `roughness = sqrt(EMA_distance(clamp(BPF(a_z), +/-clip)^2))`
///
/// The analyzer keeps only filter state and one energy accumulator. It does not
/// retain a rolling sample buffer, so memory use is constant and suitable for
/// embedded targets.
#[derive(Clone, Copy, Debug)]
pub struct RoadRoughnessAnalyzer {
    cfg: RoadRoughnessConfig,
    last_t_s: Option<f32>,
    hp_last_input: f32,
    hp_last_output: f32,
    lp_output: f32,
    filter_initialized: bool,
    energy_mps2_sq: f32,
    energy_initialized: bool,
    distance_m: f32,
    last_estimate: RoadRoughnessEstimate,
}

impl RoadRoughnessAnalyzer {
    pub fn new(cfg: RoadRoughnessConfig) -> Self {
        Self {
            cfg,
            last_t_s: None,
            hp_last_input: 0.0,
            hp_last_output: 0.0,
            lp_output: 0.0,
            filter_initialized: false,
            energy_mps2_sq: 0.0,
            energy_initialized: false,
            distance_m: 0.0,
            last_estimate: RoadRoughnessEstimate {
                t_s: 0.0,
                roughness_rms_mps2: 0.0,
                level: RoadRoughnessLevel::VerySmooth,
                vertical_accel_bandpass_mps2: 0.0,
                vertical_accel_clipped_mps2: 0.0,
                distance_m: 0.0,
                updated: false,
            },
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }

    pub fn update(&mut self, sample: RoadRoughnessSample) -> Option<RoadRoughnessEstimate> {
        if !valid_sample(sample) {
            return None;
        }

        let raw_dt = self
            .last_t_s
            .map(|last_t_s| (sample.t_s - last_t_s).max(0.0))
            .unwrap_or(0.0);
        self.last_t_s = Some(sample.t_s);
        let dt = raw_dt.min(self.cfg.max_dt_s.max(0.0));

        if !self.filter_initialized {
            self.filter_initialized = true;
            self.hp_last_input = sample.vertical_accel_mps2;
            self.hp_last_output = 0.0;
            self.lp_output = 0.0;
            return Some(self.make_estimate(sample.t_s, 0.0, 0.0, false));
        }

        let bandpass = if dt > 0.0 {
            let hp = high_pass(
                sample.vertical_accel_mps2,
                self.hp_last_input,
                self.hp_last_output,
                self.cfg.high_pass_cutoff_hz,
                dt,
            );
            self.hp_last_input = sample.vertical_accel_mps2;
            self.hp_last_output = hp;
            self.lp_output = low_pass(self.lp_output, hp, self.cfg.low_pass_cutoff_hz, dt);
            self.lp_output
        } else {
            self.lp_output
        };

        let clipped = bandpass.clamp(-self.cfg.clip_mps2, self.cfg.clip_mps2);
        let speed_mps = sample.speed_mps.max(0.0);
        let ds = speed_mps * dt;
        let updated = speed_mps >= self.cfg.min_speed_mps && ds > 0.0;
        if updated {
            self.distance_m += ds;
            let energy_sample = clipped * clipped;
            if self.energy_initialized {
                let tau_m = self.cfg.distance_tau_m.max(ds);
                let alpha = ds / (tau_m + ds);
                self.energy_mps2_sq = (1.0 - alpha) * self.energy_mps2_sq + alpha * energy_sample;
            } else {
                self.energy_initialized = true;
                self.energy_mps2_sq = energy_sample;
            }
        }

        Some(self.make_estimate(sample.t_s, bandpass, clipped, updated))
    }

    pub fn estimate(&self) -> RoadRoughnessEstimate {
        self.last_estimate
    }

    fn make_estimate(
        &mut self,
        t_s: f32,
        bandpass: f32,
        clipped: f32,
        updated: bool,
    ) -> RoadRoughnessEstimate {
        let roughness_rms_mps2 = sqrt_f32(self.energy_mps2_sq.max(0.0));
        let estimate = RoadRoughnessEstimate {
            t_s,
            roughness_rms_mps2,
            level: level_for_rms(self.cfg, roughness_rms_mps2),
            vertical_accel_bandpass_mps2: bandpass,
            vertical_accel_clipped_mps2: clipped,
            distance_m: self.distance_m,
            updated,
        };
        self.last_estimate = estimate;
        estimate
    }
}

impl Default for RoadRoughnessAnalyzer {
    fn default() -> Self {
        Self::new(RoadRoughnessConfig::default())
    }
}

fn level_for_rms(cfg: RoadRoughnessConfig, rms_mps2: f32) -> RoadRoughnessLevel {
    if rms_mps2 < cfg.very_smooth_threshold_mps2 {
        RoadRoughnessLevel::VerySmooth
    } else if rms_mps2 < cfg.smooth_threshold_mps2 {
        RoadRoughnessLevel::Smooth
    } else if rms_mps2 < cfg.light_texture_threshold_mps2 {
        RoadRoughnessLevel::LightTexture
    } else if rms_mps2 < cfg.moderate_threshold_mps2 {
        RoadRoughnessLevel::Moderate
    } else if rms_mps2 < cfg.rough_threshold_mps2 {
        RoadRoughnessLevel::Rough
    } else if rms_mps2 < cfg.very_rough_threshold_mps2 {
        RoadRoughnessLevel::VeryRough
    } else {
        RoadRoughnessLevel::Severe
    }
}

fn valid_sample(sample: RoadRoughnessSample) -> bool {
    sample.t_s.is_finite() && sample.speed_mps.is_finite() && sample.vertical_accel_mps2.is_finite()
}
