/// One detected road speed-bump event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeedBumpEvent {
    pub t_s: f32,
    /// UI-facing confidence normalized after the internal score clears the
    /// trigger threshold. This is not a calibrated probability.
    pub confidence: f32,
    pub duration_s: f32,
    pub peak_abs_pitch_deg: f32,
}

/// Per-sample detector diagnostics useful for plotting and tuning.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpeedBumpDiagnostic {
    pub t_s: f32,
    pub pitch_hpf_deg: f32,
    pub pitch_noise_deg: f32,
    pub vertical_accel_hpf_mps2: f32,
    pub vertical_accel_noise_mps2: f32,
}

/// Vehicle motion sample consumed by [`crate::SpeedBumpDetector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeedBumpSample {
    pub t_s: f32,
    pub speed_mps: f32,
    pub pitch_deg: f32,
    /// Gravity-compensated vehicle-frame vertical acceleration.
    pub vertical_accel_mps2: f32,
}

/// One detected uphill or downhill interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HillEvent {
    pub kind: HillKind,
    pub start_t_s: f32,
    pub end_t_s: f32,
    pub duration_s: f32,
    pub mean_pitch_deg: f32,
    pub peak_abs_pitch_deg: f32,
    pub mean_speed_mps: f32,
}

/// Signed hill direction inferred from vehicle pitch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HillKind {
    Uphill,
    Downhill,
}

/// Vehicle pitch/speed sample consumed by [`crate::HillDetector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HillSample {
    pub t_s: f32,
    pub speed_mps: f32,
    pub pitch_deg: f32,
}

/// One detected reverse-driving interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReverseEvent {
    pub start_t_s: f32,
    pub end_t_s: f32,
    pub duration_s: f32,
    pub mean_reverse_speed_mps: f32,
    pub peak_reverse_speed_mps: f32,
}

/// Vehicle-frame longitudinal velocity sample consumed by [`crate::ReverseDetector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReverseSample {
    pub t_s: f32,
    /// Vehicle-frame forward velocity. Negative values indicate reverse motion.
    pub forward_velocity_mps: f32,
}

/// One detected harsh longitudinal acceleration or braking interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarshLongitudinalEvent {
    pub start_t_s: f32,
    pub end_t_s: f32,
    pub duration_s: f32,
    pub delta_velocity_mps: f32,
    pub mean_accel_mps2: f32,
    pub peak_accel_mps2: f32,
    pub mean_speed_mps: f32,
    pub peak_speed_mps: f32,
}

/// Vehicle-frame forward velocity sample consumed by harsh accel/brake detectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarshLongitudinalSample {
    pub t_s: f32,
    pub forward_velocity_mps: f32,
}

/// One detected harsh cornering interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarshCornerEvent {
    pub start_t_s: f32,
    pub end_t_s: f32,
    pub duration_s: f32,
    pub mean_lateral_accel_mps2: f32,
    pub peak_lateral_accel_mps2: f32,
    pub mean_speed_mps: f32,
    pub peak_speed_mps: f32,
}

/// Vehicle yaw-rate and speed sample consumed by [`crate::HarshCornerDetector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarshCornerSample {
    pub t_s: f32,
    pub speed_mps: f32,
    pub yaw_rate_radps: f32,
}

/// Configuration for the small-state speed-bump detector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeedBumpConfig {
    /// Pitch high-pass cutoff used to reject road grade and slow maneuvers.
    pub pitch_hpf_cutoff_hz: f32,
    /// Vertical acceleration high-pass cutoff used to reject slow body motion.
    pub vertical_accel_hpf_cutoff_hz: f32,
    /// EMA time constant for adaptive noise-floor estimation.
    pub noise_tau_s: f32,
    /// Minimum speed for speed-adaptive spacing logic.
    pub min_speed_mps: f32,
    /// Lower plausible wheelbase used to convert speed into peak spacing.
    pub wheelbase_min_m: f32,
    /// Upper plausible wheelbase used to convert speed into peak spacing.
    pub wheelbase_max_m: f32,
    /// Absolute minimum accepted event duration.
    pub min_event_duration_s: f32,
    /// Absolute maximum accepted event duration.
    pub max_event_duration_s: f32,
    /// Adaptive vertical-acceleration peak threshold multiplier.
    pub vertical_accel_noise_peak_scale: f32,
    /// Minimum gravity-compensated vertical acceleration peak for a bump candidate.
    pub min_vertical_accel_peak_mps2: f32,
    /// Minimum fraction of the accel-pattern duration with vertical accel above the physical floor.
    pub min_vertical_accel_active_fraction: f32,
    /// Minimum total time within the pattern with vertical accel above the physical floor.
    pub min_vertical_accel_active_duration_s: f32,
    /// Adaptive pitch corroboration threshold multiplier.
    pub pitch_noise_peak_scale: f32,
    /// Confidence required before an event is emitted.
    pub trigger_confidence: f32,
    /// Event refractory period after a trigger.
    pub refractory_s: f32,
}

/// Configuration for sustained uphill/downhill detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HillConfig {
    /// Absolute vehicle pitch needed to enter and remain in a hill candidate.
    pub pitch_threshold_deg: f32,
    /// Minimum sustained duration before the candidate becomes an emitted hill.
    pub min_duration_s: f32,
}

/// Configuration for reverse-driving interval detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReverseConfig {
    /// Candidate reverse motion starts below this vehicle-frame forward velocity.
    pub enter_forward_velocity_mps: f32,
    /// Reverse motion remains active until forward velocity rises above this value.
    pub exit_forward_velocity_mps: f32,
    /// Time below the enter threshold before the interval is confirmed.
    pub enter_debounce_s: f32,
    /// Time above the exit threshold before the interval is closed.
    pub exit_debounce_s: f32,
    /// Minimum confirmed interval duration required before an event is emitted.
    pub min_duration_s: f32,
}

/// Configuration for EMA-smoothed velocity-derivative harsh acceleration detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarshAccelConfig {
    /// Time constant for EMA smoothing of `dv / dt`, seconds.
    pub accel_tau_s: f32,
    /// Clamp applied to raw `dv / dt` before smoothing.
    pub max_raw_accel_mps2: f32,
    /// Enter threshold for smoothed longitudinal acceleration.
    pub accel_threshold_mps2: f32,
    /// Exit threshold for hysteresis.
    pub exit_accel_threshold_mps2: f32,
    /// Minimum duration above threshold before an event is emitted.
    pub min_duration_s: f32,
    /// Minimum speed for harsh acceleration detection.
    pub min_speed_mps: f32,
    /// Event refractory period after a trigger.
    pub refractory_s: f32,
}

/// Configuration for EMA-smoothed velocity-derivative harsh braking detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarshBrakeConfig {
    /// Time constant for EMA smoothing of `dv / dt`, seconds.
    pub accel_tau_s: f32,
    /// Clamp applied to raw `dv / dt` before smoothing.
    pub max_raw_accel_mps2: f32,
    /// Enter threshold for positive deceleration magnitude.
    pub decel_threshold_mps2: f32,
    /// Exit threshold for hysteresis.
    pub exit_decel_threshold_mps2: f32,
    /// Minimum duration above threshold before an event is emitted.
    pub min_duration_s: f32,
    /// Minimum speed for harsh braking detection.
    pub min_speed_mps: f32,
    /// Event refractory period after a trigger.
    pub refractory_s: f32,
}

/// Configuration for steady-turn harsh cornering detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarshCornerConfig {
    /// Enter threshold for `abs(yaw_rate * speed)`.
    pub lateral_accel_threshold_mps2: f32,
    /// Exit threshold for hysteresis.
    pub exit_lateral_accel_threshold_mps2: f32,
    /// Minimum duration above threshold before an event is emitted.
    pub min_duration_s: f32,
    /// Minimum speed for the steady-turn assumption to be meaningful.
    pub min_speed_mps: f32,
    /// Event refractory period after a trigger.
    pub refractory_s: f32,
}

impl Default for HillConfig {
    fn default() -> Self {
        Self {
            pitch_threshold_deg: 4.0,
            min_duration_s: 3.0,
        }
    }
}

impl Default for ReverseConfig {
    fn default() -> Self {
        Self {
            enter_forward_velocity_mps: -0.5,
            exit_forward_velocity_mps: -0.2,
            enter_debounce_s: 0.5,
            exit_debounce_s: 0.5,
            min_duration_s: 1.0,
        }
    }
}

impl Default for HarshAccelConfig {
    fn default() -> Self {
        Self {
            accel_tau_s: 0.6,
            max_raw_accel_mps2: 15.0,
            accel_threshold_mps2: 2.5,
            exit_accel_threshold_mps2: 2.0,
            min_duration_s: 0.4,
            min_speed_mps: 1.0,
            refractory_s: 2.0,
        }
    }
}

impl Default for HarshBrakeConfig {
    fn default() -> Self {
        Self {
            accel_tau_s: 0.6,
            max_raw_accel_mps2: 15.0,
            decel_threshold_mps2: 3.0,
            exit_decel_threshold_mps2: 2.4,
            min_duration_s: 0.4,
            min_speed_mps: 1.0,
            refractory_s: 2.0,
        }
    }
}

impl Default for HarshCornerConfig {
    fn default() -> Self {
        Self {
            lateral_accel_threshold_mps2: 3.0,
            exit_lateral_accel_threshold_mps2: 2.4,
            min_duration_s: 0.5,
            min_speed_mps: 3.0,
            refractory_s: 2.0,
        }
    }
}

impl Default for SpeedBumpConfig {
    fn default() -> Self {
        Self {
            pitch_hpf_cutoff_hz: 0.45,
            vertical_accel_hpf_cutoff_hz: 0.70,
            noise_tau_s: 6.0,
            min_speed_mps: 1.5,
            wheelbase_min_m: 1.8,
            wheelbase_max_m: 3.6,
            min_event_duration_s: 0.18,
            max_event_duration_s: 1.8,
            vertical_accel_noise_peak_scale: 3.5,
            min_vertical_accel_peak_mps2: 1.5,
            min_vertical_accel_active_fraction: 0.25,
            min_vertical_accel_active_duration_s: 0.25,
            pitch_noise_peak_scale: 3.0,
            trigger_confidence: 0.12,
            refractory_s: 4.0,
        }
    }
}
