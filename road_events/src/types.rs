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

/// Vehicle-motion sample consumed by [`crate::RoadRoughnessAnalyzer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoadRoughnessSample {
    pub t_s: f32,
    /// Nonnegative vehicle speed used to convert time into traveled distance.
    pub speed_mps: f32,
    /// Gravity-compensated vehicle-frame vertical acceleration.
    pub vertical_accel_mps2: f32,
}

/// Qualitative road-surface class derived from roughness RMS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoadRoughnessLevel {
    VerySmooth,
    Smooth,
    LightTexture,
    Moderate,
    Rough,
    VeryRough,
    Severe,
}

/// Streaming road-roughness estimate for the current effective distance window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoadRoughnessEstimate {
    pub t_s: f32,
    /// Distance-domain RMS of robustly limited, band-passed vertical acceleration.
    pub roughness_rms_mps2: f32,
    pub level: RoadRoughnessLevel,
    /// Band-passed vertical acceleration before robust limiting.
    pub vertical_accel_bandpass_mps2: f32,
    /// Band-passed vertical acceleration after robust impulse limiting.
    pub vertical_accel_clipped_mps2: f32,
    /// Integrated valid moving distance since analyzer reset.
    pub distance_m: f32,
    /// Whether this sample updated the distance-domain energy estimate.
    pub updated: bool,
}

/// One detected sustained rough-road interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoadRoughnessEvent {
    pub start_t_s: f32,
    pub end_t_s: f32,
    pub duration_s: f32,
    pub mean_roughness_rms_mps2: f32,
    pub peak_roughness_rms_mps2: f32,
    pub mean_speed_mps: f32,
    pub distance_m: f32,
}

/// One detected short vertical shock such as a pothole, bump, or sharp joint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoadShockEvent {
    pub start_t_s: f32,
    pub end_t_s: f32,
    pub duration_s: f32,
    pub peak_abs_vertical_accel_mps2: f32,
    pub mean_speed_mps: f32,
}

/// Roughness analyzer output including optional event emissions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoadRoughnessUpdate {
    pub estimate: RoadRoughnessEstimate,
    /// Live notification emitted as soon as sustained roughness is confirmed.
    pub roughness_event: Option<RoadRoughnessEvent>,
    /// Completed rough-road interval emitted when the active rough patch exits
    /// or is flushed.
    pub completed_roughness_event: Option<RoadRoughnessEvent>,
    pub shock_event: Option<RoadShockEvent>,
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

/// Vehicle lateral-motion sample consumed by [`crate::HarshCornerDetector`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarshCornerSample {
    pub t_s: f32,
    pub speed_mps: f32,
    /// Vehicle-frame lateral acceleration/specific-force sample.
    ///
    /// This represents passenger side-load, including bank effects. Callers
    /// should provide the bias-corrected accelerometer specific force rotated
    /// into vehicle frame.
    pub lateral_accel_mps2: f32,
}

/// Vehicle-motion sample consumed by [`crate::TripStats`].
///
/// All fields are expressed in the vehicle frame or scalar vehicle-motion
/// quantities. The accumulator is streaming and does not retain sample history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TripSample {
    pub t_s: f32,
    /// Nonnegative vehicle speed magnitude used for total distance.
    pub speed_mps: f32,
    /// Signed vehicle-frame longitudinal velocity. Negative means reverse.
    pub forward_velocity_mps: f32,
    /// Optional vertical position, positive up, used for elevation gain/loss.
    ///
    /// If this value comes from a local-NED down coordinate, callers should
    /// supply `-down` and change `height_frame_id` whenever the local anchor is
    /// reset. Deltas across different frame IDs are ignored.
    pub height_m: Option<f32>,
    /// Monotonic identifier for the vertical-position reference frame.
    pub height_frame_id: u32,
    /// Gravity-compensated vehicle-frame longitudinal acceleration.
    pub longitudinal_accel_mps2: f32,
    /// Vehicle-frame lateral acceleration magnitude or signed lateral acceleration.
    pub lateral_accel_mps2: f32,
}

/// Event category used by [`crate::TripStats`] counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TripEventKind {
    SpeedBump,
    RoadShock,
    RoughRoad,
    Uphill,
    Downhill,
    Reverse,
    HarshAcceleration,
    HarshBraking,
    HarshCornering,
}

/// Constant-memory event counters accumulated over a trip.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TripEventCounts {
    pub speed_bumps: u32,
    pub road_shocks: u32,
    pub rough_road: u32,
    pub uphill: u32,
    pub downhill: u32,
    pub reverse: u32,
    pub harsh_acceleration: u32,
    pub harsh_braking: u32,
    pub harsh_cornering: u32,
}

/// Configuration for streaming trip statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TripConfig {
    /// Speed above which the vehicle is counted as moving.
    pub moving_speed_threshold_mps: f32,
    /// Reverse speed magnitude above which reverse duration is accumulated.
    pub reverse_speed_threshold_mps: f32,
    /// Time constant for EMA-style rolling statistics.
    pub rolling_tau_s: f32,
    /// Maximum sample interval integrated into totals. Larger gaps are counted
    /// as data gaps and clipped to this value for distance/mean integration.
    pub max_integrated_dt_s: f32,
}

/// Snapshot of accumulated trip statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TripSummary {
    pub sample_count: u32,
    pub invalid_sample_count: u32,
    pub data_gap_count: u32,
    pub max_sample_gap_s: f32,
    pub total_gap_duration_s: f32,
    pub duration_s: f32,
    pub moving_duration_s: f32,
    pub stationary_duration_s: f32,
    pub distance_m: f32,
    pub reverse_duration_s: f32,
    pub reverse_distance_m: f32,
    /// Position-derived ascent estimate from optional height samples.
    pub elevation_gain_m: f32,
    /// Position-derived descent estimate from optional height samples.
    pub elevation_loss_m: f32,
    /// Whether vertical-position samples contributed to the elevation fields.
    pub elevation_valid: bool,
    pub mean_speed_mps: f32,
    pub moving_mean_speed_mps: f32,
    pub peak_speed_mps: f32,
    pub peak_accel_mps2: f32,
    pub peak_decel_mps2: f32,
    pub peak_lateral_accel_mps2: f32,
    pub rolling_speed_mps: f32,
    pub rolling_abs_longitudinal_accel_mps2: f32,
    pub rolling_abs_lateral_accel_mps2: f32,
    pub events: TripEventCounts,
    pub speed_bumps_per_km: f32,
    pub road_shocks_per_km: f32,
    pub rough_road_events_per_km: f32,
    pub harsh_events_per_km: f32,
    pub reverse_seconds_per_km: f32,
}

/// Sensitivity preset for harsh acceleration, braking, and cornering detectors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub enum HarshBehaviorPreset {
    /// Detect mild-but-noticeable uncomfortable driving.
    Sensitive,
    /// Everyday default detection.
    #[default]
    Balanced,
    /// Emit only clearer harsh events.
    Conservative,
}

/// Detector configuration bundle derived from a [`HarshBehaviorPreset`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarshBehaviorConfig {
    pub accel: HarshAccelConfig,
    pub brake: HarshBrakeConfig,
    pub corner: HarshCornerConfig,
}

/// Configuration for streaming road-roughness analysis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoadRoughnessConfig {
    /// High-pass cutoff for rejecting grade, body attitude drift, and slow suspension settling.
    pub high_pass_cutoff_hz: f32,
    /// Low-pass cutoff for rejecting high-frequency IMU/structure noise.
    pub low_pass_cutoff_hz: f32,
    /// Distance-domain EMA time constant. Smaller values make the roughness estimate react
    /// and dissipate over a shorter traveled distance without a rolling sample buffer.
    pub distance_tau_m: f32,
    /// Minimum speed required to update roughness energy.
    pub min_speed_mps: f32,
    /// Absolute ceiling for the adaptive robust limiter applied to band-passed acceleration.
    pub clip_mps2: f32,
    /// Distance-domain EMA used to track local ambient vertical vibration.
    pub robust_baseline_tau_m: f32,
    /// Minimum robust limiter cap for preserving small real texture on smooth roads.
    pub robust_min_cap_mps2: f32,
    /// Multiplier from ambient baseline to robust limiter cap.
    pub robust_cap_scale: f32,
    /// Minimum band-passed vertical acceleration magnitude for a shock candidate.
    pub shock_min_peak_mps2: f32,
    /// Multiplier from ambient baseline to shock candidate threshold.
    pub shock_baseline_scale: f32,
    /// Candidate remains active until magnitude falls below this fraction of the enter threshold.
    pub shock_exit_fraction: f32,
    /// Minimum shock duration before event emission.
    pub shock_min_duration_s: f32,
    /// Maximum duration still considered a discrete shock instead of sustained roughness.
    pub shock_max_duration_s: f32,
    /// Refractory period after a shock event.
    pub shock_refractory_s: f32,
    /// Rough-road event enter threshold for robust roughness RMS.
    pub rough_event_enter_mps2: f32,
    /// Rough-road event exit threshold for robust roughness RMS.
    pub rough_event_exit_mps2: f32,
    /// Minimum rough-road interval duration before event emission.
    pub rough_event_min_duration_s: f32,
    /// Refractory period after a rough-road event.
    pub rough_event_refractory_s: f32,
    /// Maximum sample interval used for filter and distance updates.
    pub max_dt_s: f32,
    /// Upper bound for VerySmooth roughness.
    pub very_smooth_threshold_mps2: f32,
    /// Upper bound for Smooth roughness.
    pub smooth_threshold_mps2: f32,
    /// Upper bound for LightTexture roughness.
    pub light_texture_threshold_mps2: f32,
    /// Upper bound for Moderate roughness.
    pub moderate_threshold_mps2: f32,
    /// Upper bound for Rough roughness.
    pub rough_threshold_mps2: f32,
    /// Upper bound for VeryRough roughness. Larger values are Severe.
    pub very_rough_threshold_mps2: f32,
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

/// Configuration for jerk-gated harsh cornering detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarshCornerConfig {
    /// Minimum smoothed lateral side-load required to start an event.
    pub lateral_accel_threshold_mps2: f32,
    /// Smoothed lateral side-load threshold below which an active event exits.
    pub exit_lateral_accel_threshold_mps2: f32,
    /// EMA time constant for denoising lateral side-load before differentiating.
    pub lateral_accel_tau_s: f32,
    /// EMA time constant for denoising absolute lateral jerk.
    pub lateral_jerk_tau_s: f32,
    /// Minimum smoothed absolute lateral jerk required to arm an event.
    pub lateral_jerk_threshold_mps3: f32,
    /// Maximum raw lateral jerk used before jerk smoothing.
    pub max_raw_lateral_jerk_mps3: f32,
    /// Time window in which a jerk trigger can start a lateral-load interval.
    pub jerk_trigger_window_s: f32,
    /// Minimum lateral-load interval duration before an event is emitted.
    pub min_duration_s: f32,
    /// Minimum speed for harsh cornering detection.
    pub min_speed_mps: f32,
    /// Event refractory period after a trigger.
    pub refractory_s: f32,
}

impl Default for HillConfig {
    fn default() -> Self {
        Self {
            pitch_threshold_deg: 4.0,
            min_duration_s: 1.0,
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
            lateral_accel_tau_s: 0.15,
            lateral_jerk_tau_s: 0.20,
            lateral_jerk_threshold_mps3: 4.0,
            max_raw_lateral_jerk_mps3: 80.0,
            jerk_trigger_window_s: 0.50,
            min_duration_s: 0.5,
            min_speed_mps: 3.0,
            refractory_s: 2.0,
        }
    }
}

impl HarshBehaviorPreset {
    pub const ALL: [Self; 3] = [Self::Sensitive, Self::Balanced, Self::Conservative];

    pub fn configs(self) -> HarshBehaviorConfig {
        let mut accel = HarshAccelConfig::default();
        let mut brake = HarshBrakeConfig::default();
        let mut corner = HarshCornerConfig::default();

        match self {
            Self::Sensitive => {
                accel.accel_threshold_mps2 = 2.0;
                accel.exit_accel_threshold_mps2 = 1.6;
                brake.decel_threshold_mps2 = 2.5;
                brake.exit_decel_threshold_mps2 = 2.0;
                corner.lateral_accel_threshold_mps2 = 2.3;
                corner.exit_lateral_accel_threshold_mps2 = 1.84;
                corner.lateral_jerk_threshold_mps3 = 3.0;
            }
            Self::Balanced => {
                corner.lateral_accel_threshold_mps2 = 3.4;
                corner.exit_lateral_accel_threshold_mps2 = 2.9;
                corner.lateral_jerk_threshold_mps3 = 5.0;
            }
            Self::Conservative => {
                accel.accel_threshold_mps2 = 3.2;
                accel.exit_accel_threshold_mps2 = 2.56;
                brake.decel_threshold_mps2 = 4.0;
                brake.exit_decel_threshold_mps2 = 3.2;
                corner.lateral_accel_threshold_mps2 = 3.8;
                corner.exit_lateral_accel_threshold_mps2 = 3.04;
                corner.lateral_jerk_threshold_mps3 = 6.0;
            }
        }

        HarshBehaviorConfig {
            accel,
            brake,
            corner,
        }
    }
}

impl Default for RoadRoughnessConfig {
    fn default() -> Self {
        Self {
            high_pass_cutoff_hz: 0.7,
            low_pass_cutoff_hz: 10.0,
            distance_tau_m: 10.0,
            min_speed_mps: 2.0,
            clip_mps2: 1.5,
            robust_baseline_tau_m: 20.0,
            robust_min_cap_mps2: 0.60,
            robust_cap_scale: 3.0,
            shock_min_peak_mps2: 2.5,
            shock_baseline_scale: 6.0,
            shock_exit_fraction: 0.45,
            shock_min_duration_s: 0.02,
            shock_max_duration_s: 0.65,
            shock_refractory_s: 0.50,
            rough_event_enter_mps2: 0.60,
            rough_event_exit_mps2: 0.42,
            rough_event_min_duration_s: 1.0,
            rough_event_refractory_s: 8.0,
            max_dt_s: 0.2,
            very_smooth_threshold_mps2: 0.15,
            smooth_threshold_mps2: 0.25,
            light_texture_threshold_mps2: 0.40,
            moderate_threshold_mps2: 0.60,
            rough_threshold_mps2: 0.90,
            very_rough_threshold_mps2: 1.20,
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

impl Default for TripConfig {
    fn default() -> Self {
        Self {
            moving_speed_threshold_mps: 0.5,
            reverse_speed_threshold_mps: 0.2,
            rolling_tau_s: 5.0,
            max_integrated_dt_s: 1.0,
        }
    }
}
