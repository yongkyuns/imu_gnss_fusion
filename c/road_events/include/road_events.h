#ifndef ROAD_EVENTS_H
#define ROAD_EVENTS_H

/**
 * @file road_events.h
 * @brief C99 streaming road-event detectors.
 *
 * This API mirrors the Rust `road_events` crate at the type and detector
 * boundary. All detector state is caller-owned. The implementation performs no
 * allocation, file I/O, or global mutable runtime access.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ROAD_EVENTS_ROUGHNESS_VERY_SMOOTH = 0,
    ROAD_EVENTS_ROUGHNESS_SMOOTH = 1,
    ROAD_EVENTS_ROUGHNESS_LIGHT_TEXTURE = 2,
    ROAD_EVENTS_ROUGHNESS_MODERATE = 3,
    ROAD_EVENTS_ROUGHNESS_ROUGH = 4,
    ROAD_EVENTS_ROUGHNESS_VERY_ROUGH = 5,
    ROAD_EVENTS_ROUGHNESS_SEVERE = 6,
} road_events_roughness_level_t;

typedef enum {
    ROAD_EVENTS_HILL_UPHILL = 0,
    ROAD_EVENTS_HILL_DOWNHILL = 1,
} road_events_hill_kind_t;

typedef enum {
    ROAD_EVENTS_HARSH_PRESET_SENSITIVE = 0,
    ROAD_EVENTS_HARSH_PRESET_BALANCED = 1,
    ROAD_EVENTS_HARSH_PRESET_CONSERVATIVE = 2,
    ROAD_EVENTS_HARSH_SENSITIVE = ROAD_EVENTS_HARSH_PRESET_SENSITIVE,
    ROAD_EVENTS_HARSH_BALANCED = ROAD_EVENTS_HARSH_PRESET_BALANCED,
    ROAD_EVENTS_HARSH_CONSERVATIVE = ROAD_EVENTS_HARSH_PRESET_CONSERVATIVE,
} road_events_harsh_preset_t;

typedef road_events_harsh_preset_t road_events_harsh_behavior_preset_t;

typedef enum {
    ROAD_EVENTS_TRIP_EVENT_SPEED_BUMP = 0,
    ROAD_EVENTS_TRIP_EVENT_ROAD_SHOCK = 1,
    ROAD_EVENTS_TRIP_EVENT_ROUGH_ROAD = 2,
    ROAD_EVENTS_TRIP_EVENT_UPHILL = 3,
    ROAD_EVENTS_TRIP_EVENT_DOWNHILL = 4,
    ROAD_EVENTS_TRIP_EVENT_REVERSE = 5,
    ROAD_EVENTS_TRIP_EVENT_HARSH_ACCELERATION = 6,
    ROAD_EVENTS_TRIP_EVENT_HARSH_BRAKING = 7,
    ROAD_EVENTS_TRIP_EVENT_HARSH_CORNERING = 8,
    ROAD_EVENTS_TRIP_SPEED_BUMP = ROAD_EVENTS_TRIP_EVENT_SPEED_BUMP,
    ROAD_EVENTS_TRIP_ROAD_SHOCK = ROAD_EVENTS_TRIP_EVENT_ROAD_SHOCK,
    ROAD_EVENTS_TRIP_ROUGH_ROAD = ROAD_EVENTS_TRIP_EVENT_ROUGH_ROAD,
    ROAD_EVENTS_TRIP_UPHILL = ROAD_EVENTS_TRIP_EVENT_UPHILL,
    ROAD_EVENTS_TRIP_DOWNHILL = ROAD_EVENTS_TRIP_EVENT_DOWNHILL,
    ROAD_EVENTS_TRIP_REVERSE = ROAD_EVENTS_TRIP_EVENT_REVERSE,
    ROAD_EVENTS_TRIP_HARSH_ACCELERATION = ROAD_EVENTS_TRIP_EVENT_HARSH_ACCELERATION,
    ROAD_EVENTS_TRIP_HARSH_BRAKING = ROAD_EVENTS_TRIP_EVENT_HARSH_BRAKING,
    ROAD_EVENTS_TRIP_HARSH_CORNERING = ROAD_EVENTS_TRIP_EVENT_HARSH_CORNERING,
} road_events_trip_event_kind_t;

typedef struct {
    float pitch_hpf_cutoff_hz;
    float vertical_accel_hpf_cutoff_hz;
    float noise_tau_s;
    float min_speed_mps;
    float wheelbase_min_m;
    float wheelbase_max_m;
    float min_event_duration_s;
    float max_event_duration_s;
    float vertical_accel_noise_peak_scale;
    float min_vertical_accel_peak_mps2;
    float min_vertical_accel_active_fraction;
    float min_vertical_accel_active_duration_s;
    float pitch_noise_peak_scale;
    float trigger_confidence;
    float refractory_s;
} road_events_speed_bump_config_t;

typedef struct {
    float t_s;
    float speed_mps;
    float pitch_deg;
    float vertical_accel_mps2;
} road_events_speed_bump_sample_t;

typedef struct {
    float t_s;
    float confidence;
    float duration_s;
    float peak_abs_pitch_deg;
} road_events_speed_bump_event_t;

typedef struct {
    float t_s;
    float pitch_hpf_deg;
    float pitch_noise_deg;
    float vertical_accel_hpf_mps2;
    float vertical_accel_noise_mps2;
} road_events_speed_bump_diagnostic_t;

typedef struct {
    float pitch_threshold_deg;
    float min_duration_s;
} road_events_hill_config_t;

typedef struct {
    float t_s;
    float speed_mps;
    float pitch_deg;
} road_events_hill_sample_t;

typedef struct {
    road_events_hill_kind_t kind;
    float start_t_s;
    float end_t_s;
    float duration_s;
    float mean_pitch_deg;
    float peak_abs_pitch_deg;
    float mean_speed_mps;
} road_events_hill_event_t;

typedef struct {
    float enter_forward_velocity_mps;
    float exit_forward_velocity_mps;
    float enter_debounce_s;
    float exit_debounce_s;
    float min_duration_s;
} road_events_reverse_config_t;

typedef struct {
    float t_s;
    float forward_velocity_mps;
} road_events_reverse_sample_t;

typedef struct {
    float start_t_s;
    float end_t_s;
    float duration_s;
    float mean_reverse_speed_mps;
    float peak_reverse_speed_mps;
} road_events_reverse_event_t;

typedef struct {
    float accel_tau_s;
    float max_raw_accel_mps2;
    float accel_threshold_mps2;
    float exit_accel_threshold_mps2;
    float min_duration_s;
    float min_speed_mps;
    float refractory_s;
} road_events_harsh_accel_config_t;

typedef struct {
    float accel_tau_s;
    float max_raw_accel_mps2;
    float decel_threshold_mps2;
    float exit_decel_threshold_mps2;
    float min_duration_s;
    float min_speed_mps;
    float refractory_s;
} road_events_harsh_brake_config_t;

typedef struct {
    float lateral_accel_threshold_mps2;
    float exit_lateral_accel_threshold_mps2;
    float lateral_accel_tau_s;
    float lateral_jerk_tau_s;
    float lateral_jerk_threshold_mps3;
    float max_raw_lateral_jerk_mps3;
    float jerk_trigger_window_s;
    float min_duration_s;
    float min_speed_mps;
    float refractory_s;
} road_events_harsh_corner_config_t;

typedef struct {
    road_events_harsh_accel_config_t accel;
    road_events_harsh_brake_config_t brake;
    road_events_harsh_corner_config_t corner;
} road_events_harsh_behavior_config_t;

typedef struct {
    float t_s;
    float forward_velocity_mps;
} road_events_harsh_longitudinal_sample_t;

typedef struct {
    float start_t_s;
    float end_t_s;
    float duration_s;
    float delta_velocity_mps;
    float mean_accel_mps2;
    float peak_accel_mps2;
    float mean_speed_mps;
    float peak_speed_mps;
} road_events_harsh_longitudinal_event_t;

typedef struct {
    float t_s;
    float speed_mps;
    float lateral_accel_mps2;
} road_events_harsh_corner_sample_t;

typedef struct {
    float start_t_s;
    float end_t_s;
    float duration_s;
    float mean_lateral_accel_mps2;
    float peak_lateral_accel_mps2;
    float mean_speed_mps;
    float peak_speed_mps;
} road_events_harsh_corner_event_t;

typedef struct {
    float high_pass_cutoff_hz;
    float low_pass_cutoff_hz;
    float distance_tau_m;
    float min_speed_mps;
    float clip_mps2;
    float robust_baseline_tau_m;
    float robust_min_cap_mps2;
    float robust_cap_scale;
    float shock_min_peak_mps2;
    float shock_baseline_scale;
    float shock_exit_fraction;
    float shock_min_duration_s;
    float shock_max_duration_s;
    float shock_refractory_s;
    float rough_event_enter_mps2;
    float rough_event_exit_mps2;
    float rough_event_min_duration_s;
    float rough_event_refractory_s;
    float max_dt_s;
    float very_smooth_threshold_mps2;
    float smooth_threshold_mps2;
    float light_texture_threshold_mps2;
    float moderate_threshold_mps2;
    float rough_threshold_mps2;
    float very_rough_threshold_mps2;
} road_events_roughness_config_t;

typedef struct {
    float t_s;
    float speed_mps;
    float vertical_accel_mps2;
} road_events_roughness_sample_t;

typedef struct {
    float t_s;
    float roughness_rms_mps2;
    road_events_roughness_level_t level;
    float vertical_accel_bandpass_mps2;
    float vertical_accel_clipped_mps2;
    float distance_m;
    bool updated;
} road_events_roughness_estimate_t;

typedef struct {
    float start_t_s;
    float end_t_s;
    float duration_s;
    float mean_roughness_rms_mps2;
    float peak_roughness_rms_mps2;
    float mean_speed_mps;
    float distance_m;
} road_events_roughness_event_t;

typedef struct {
    float start_t_s;
    float end_t_s;
    float duration_s;
    float peak_abs_vertical_accel_mps2;
    float mean_speed_mps;
} road_events_shock_event_t;

typedef struct {
    road_events_roughness_estimate_t estimate;
    bool has_roughness_event;
    road_events_roughness_event_t roughness_event;
    bool has_completed_roughness_event;
    road_events_roughness_event_t completed_roughness_event;
    bool has_shock_event;
    road_events_shock_event_t shock_event;
} road_events_roughness_update_t;

typedef struct {
    float moving_speed_threshold_mps;
    float reverse_speed_threshold_mps;
    float rolling_tau_s;
    float max_integrated_dt_s;
} road_events_trip_config_t;

typedef struct {
    float t_s;
    float speed_mps;
    float forward_velocity_mps;
    bool height_valid;
    float height_m;
    uint32_t height_frame_id;
    float longitudinal_accel_mps2;
    float lateral_accel_mps2;
} road_events_trip_sample_t;

typedef struct {
    uint32_t speed_bumps;
    uint32_t road_shocks;
    uint32_t rough_road;
    uint32_t uphill;
    uint32_t downhill;
    uint32_t reverse;
    uint32_t harsh_acceleration;
    uint32_t harsh_braking;
    uint32_t harsh_cornering;
} road_events_trip_event_counts_t;

typedef struct {
    uint32_t sample_count;
    uint32_t invalid_sample_count;
    uint32_t data_gap_count;
    float max_sample_gap_s;
    float total_gap_duration_s;
    float duration_s;
    float moving_duration_s;
    float stationary_duration_s;
    float distance_m;
    float reverse_duration_s;
    float reverse_distance_m;
    float elevation_gain_m;
    float elevation_loss_m;
    bool elevation_valid;
    float mean_speed_mps;
    float moving_mean_speed_mps;
    float peak_speed_mps;
    float peak_accel_mps2;
    float peak_decel_mps2;
    float peak_lateral_accel_mps2;
    float rolling_speed_mps;
    float rolling_abs_longitudinal_accel_mps2;
    float rolling_abs_lateral_accel_mps2;
    road_events_trip_event_counts_t events;
    float speed_bumps_per_km;
    float road_shocks_per_km;
    float rough_road_events_per_km;
    float harsh_events_per_km;
    float reverse_seconds_per_km;
} road_events_trip_summary_t;

typedef struct {
    bool valid;
    float t_s;
    float accel_hpf_mps2;
    float pitch_peak_deg;
    float speed_mps;
    float active_time_s;
} road_events_speed_bump_extremum_t;

typedef struct road_events_speed_bump_detector {
    road_events_speed_bump_config_t cfg;
    bool has_last_t_s;
    float last_t_s;
    float last_pitch_deg;
    float last_vertical_accel_mps2;
    float last_pitch_hpf_deg;
    float last_accel_hpf_mps2;
    float prev_t_s;
    float prev_accel_hpf_mps2;
    float prev_accel_slope;
    float pitch_noise_deg;
    float accel_noise_mps2;
    float accel_active_time_s;
    float pitch_peak_since_extremum_deg;
    road_events_speed_bump_extremum_t extrema[4];
    unsigned extrema_len;
    float last_event_t_s;
} road_events_speed_bump_detector_t;

typedef struct road_events_hill_detector {
    road_events_hill_config_t cfg;
    bool has_last_t_s;
    float last_t_s;
    bool has_active;
    road_events_hill_kind_t active_kind;
    float active_start_t_s;
    float active_last_t_s;
    float active_pitch_time_sum_deg_s;
    float active_speed_time_sum_m;
    float active_duration_s;
    float active_peak_abs_pitch_deg;
    bool active_emitted;
} road_events_hill_detector_t;

typedef struct road_events_reverse_detector {
    road_events_reverse_config_t cfg;
    bool has_last_t_s;
    float last_t_s;
    bool has_active;
    float active_start_t_s;
    float active_last_t_s;
    float active_reverse_speed_time_sum_m;
    float active_duration_s;
    float active_peak_reverse_speed_mps;
    float enter_duration_s;
    float exit_duration_s;
    bool confirmed;
} road_events_reverse_detector_t;

typedef struct {
    bool has_last_t_s;
    float last_t_s;
    float last_forward_velocity_mps;
    bool initialized;
    float accel_ema_mps2;
    bool tracker_has_active;
    bool tracker_has_last_t_s;
    float tracker_last_t_s;
    float tracker_last_event_t_s;
    float active_start_t_s;
    float active_last_t_s;
    float active_duration_s;
    float active_metric_time_sum;
    float active_peak_metric;
    float active_speed_time_sum_m;
    float active_peak_speed_mps;
    float active_start_velocity_mps;
    float active_end_velocity_mps;
} road_events_harsh_longitudinal_state_t;

typedef struct road_events_harsh_accel_detector {
    road_events_harsh_accel_config_t cfg;
    road_events_harsh_longitudinal_state_t state;
} road_events_harsh_accel_detector_t;

typedef struct road_events_harsh_brake_detector {
    road_events_harsh_brake_config_t cfg;
    road_events_harsh_longitudinal_state_t state;
} road_events_harsh_brake_detector_t;

typedef struct road_events_harsh_corner_detector {
    road_events_harsh_corner_config_t cfg;
    bool tracker_has_active;
    bool tracker_has_last_t_s;
    float tracker_last_t_s;
    float tracker_last_event_t_s;
    float active_start_t_s;
    float active_last_t_s;
    float active_duration_s;
    float active_metric_time_sum;
    float active_peak_metric;
    float active_speed_time_sum_m;
    float active_peak_speed_mps;
    float active_start_velocity_mps;
    float active_end_velocity_mps;
    bool lateral_has_last_t_s;
    float lateral_last_t_s;
    bool lateral_initialized;
    float lateral_accel_ema_mps2;
    bool jerk_initialized;
    float jerk_abs_ema_mps3;
    float last_jerk_trigger_t_s;
} road_events_harsh_corner_detector_t;

typedef struct road_events_roughness_analyzer {
    road_events_roughness_config_t cfg;
    bool has_last_t_s;
    float last_t_s;
    bool filter_initialized;
    float hp_last_input;
    float hp_last_output;
    float lp_output;
    bool baseline_initialized;
    float baseline_abs_mps2;
    bool energy_initialized;
    float energy_mps2_sq;
    float distance_m;
    road_events_roughness_estimate_t last_estimate;
    bool shock_has_active;
    float shock_start_t_s;
    float shock_last_t_s;
    float shock_duration_s;
    float shock_peak_abs_vertical_accel_mps2;
    float shock_speed_time_sum_m;
    float last_shock_event_t_s;
    bool rough_has_active;
    float rough_start_t_s;
    float rough_last_t_s;
    float rough_duration_s;
    float rough_roughness_time_sum;
    float rough_peak_roughness_rms_mps2;
    float rough_speed_time_sum_m;
    float rough_distance_m;
    bool rough_emitted;
    float last_rough_event_t_s;
} road_events_roughness_analyzer_t;

typedef struct road_events_trip_stats {
    road_events_trip_config_t cfg;
    road_events_trip_sample_t last_sample;
    bool has_last_sample;
    uint32_t sample_count;
    uint32_t invalid_sample_count;
    uint32_t data_gap_count;
    float max_sample_gap_s;
    float total_gap_duration_s;
    float duration_s;
    float moving_duration_s;
    float distance_m;
    float reverse_duration_s;
    float reverse_distance_m;
    float elevation_gain_m;
    float elevation_loss_m;
    bool elevation_valid;
    float speed_time_sum_m;
    float moving_speed_time_sum_m;
    float peak_speed_mps;
    float peak_accel_mps2;
    float peak_decel_mps2;
    float peak_lateral_accel_mps2;
    float rolling_speed_mps;
    float rolling_abs_longitudinal_accel_mps2;
    float rolling_abs_lateral_accel_mps2;
    road_events_trip_event_counts_t events;
    bool rolling_initialized;
} road_events_trip_stats_t;

/** Return the Rust-equivalent default speed-bump detector configuration. */
road_events_speed_bump_config_t road_events_speed_bump_config_default(void);
/** Return the Rust-equivalent default hill detector configuration. */
road_events_hill_config_t road_events_hill_config_default(void);
/** Return the Rust-equivalent default reverse detector configuration. */
road_events_reverse_config_t road_events_reverse_config_default(void);
/** Return the Rust-equivalent default harsh acceleration detector configuration. */
road_events_harsh_accel_config_t road_events_harsh_accel_config_default(void);
/** Return the Rust-equivalent default harsh braking detector configuration. */
road_events_harsh_brake_config_t road_events_harsh_brake_config_default(void);
/** Return the Rust-equivalent default harsh cornering detector configuration. */
road_events_harsh_corner_config_t road_events_harsh_corner_config_default(void);
/** Return the Rust-equivalent default road-roughness analyzer configuration. */
road_events_roughness_config_t road_events_roughness_config_default(void);
/** Return the Rust-equivalent default trip statistics configuration. */
road_events_trip_config_t road_events_trip_config_default(void);
/** Return the detector threshold bundle for a harsh behavior preset. */
road_events_harsh_behavior_config_t
road_events_harsh_behavior_preset_config(road_events_harsh_preset_t preset);

size_t road_events_speed_bump_detector_size(void);
size_t road_events_speed_bump_detector_alignment(void);
size_t road_events_hill_detector_size(void);
size_t road_events_hill_detector_alignment(void);
size_t road_events_reverse_detector_size(void);
size_t road_events_reverse_detector_alignment(void);
size_t road_events_harsh_accel_detector_size(void);
size_t road_events_harsh_accel_detector_alignment(void);
size_t road_events_harsh_brake_detector_size(void);
size_t road_events_harsh_brake_detector_alignment(void);
size_t road_events_harsh_corner_detector_size(void);
size_t road_events_harsh_corner_detector_alignment(void);
size_t road_events_roughness_analyzer_size(void);
size_t road_events_roughness_analyzer_alignment(void);
size_t road_events_trip_stats_size(void);
size_t road_events_trip_stats_alignment(void);

void road_events_speed_bump_init_default(road_events_speed_bump_detector_t *det);
void road_events_hill_init_default(road_events_hill_detector_t *det);
void road_events_reverse_init_default(road_events_reverse_detector_t *det);
void road_events_harsh_accel_init_preset(road_events_harsh_accel_detector_t *det,
                                         road_events_harsh_preset_t preset);
void road_events_harsh_brake_init_preset(road_events_harsh_brake_detector_t *det,
                                         road_events_harsh_preset_t preset);
void road_events_harsh_corner_init_preset(road_events_harsh_corner_detector_t *det,
                                          road_events_harsh_preset_t preset);
void road_events_roughness_init_default(road_events_roughness_analyzer_t *analyzer);
void road_events_trip_stats_init_default(road_events_trip_stats_t *stats);

/** Initialize a speed-bump detector with caller-owned storage. */
void road_events_speed_bump_init(road_events_speed_bump_detector_t *det,
                                 road_events_speed_bump_config_t cfg);
/** Reset a speed-bump detector while preserving its configuration. */
void road_events_speed_bump_reset(road_events_speed_bump_detector_t *det);
/** Update a speed-bump detector; returns true when @p out_event is written. */
bool road_events_speed_bump_update(road_events_speed_bump_detector_t *det,
                                   road_events_speed_bump_sample_t sample,
                                   road_events_speed_bump_diagnostic_t *out_diagnostic,
                                   road_events_speed_bump_event_t *out_event);

/** Initialize a hill detector with caller-owned storage. */
void road_events_hill_init(road_events_hill_detector_t *det,
                           road_events_hill_config_t cfg);
/** Reset a hill detector while preserving its configuration. */
void road_events_hill_reset(road_events_hill_detector_t *det);
/** Update a hill detector; returns true when @p out_event is written. */
bool road_events_hill_update(road_events_hill_detector_t *det,
                             road_events_hill_sample_t sample,
                             road_events_hill_event_t *out_event);
/** Flush a pending hill interval; returns true when @p out_event is written. */
bool road_events_hill_finish(road_events_hill_detector_t *det,
                             road_events_hill_event_t *out_event);

/** Initialize a reverse detector with caller-owned storage. */
void road_events_reverse_init(road_events_reverse_detector_t *det,
                              road_events_reverse_config_t cfg);
/** Reset a reverse detector while preserving its configuration. */
void road_events_reverse_reset(road_events_reverse_detector_t *det);
/** Update a reverse detector; returns true when @p out_event is written. */
bool road_events_reverse_update(road_events_reverse_detector_t *det,
                                road_events_reverse_sample_t sample,
                                road_events_reverse_event_t *out_event);
/** Flush a pending reverse interval; returns true when @p out_event is written. */
bool road_events_reverse_finish(road_events_reverse_detector_t *det,
                                road_events_reverse_event_t *out_event);

/** Initialize a harsh acceleration detector with caller-owned storage. */
void road_events_harsh_accel_init(road_events_harsh_accel_detector_t *det,
                                  road_events_harsh_accel_config_t cfg);
/** Reset a harsh acceleration detector while preserving its configuration. */
void road_events_harsh_accel_reset(road_events_harsh_accel_detector_t *det);
/** Update a harsh acceleration detector; returns true when @p out_event is written. */
bool road_events_harsh_accel_update(road_events_harsh_accel_detector_t *det,
                                    road_events_harsh_longitudinal_sample_t sample,
                                    road_events_harsh_longitudinal_event_t *out_event);
/** Flush a pending harsh acceleration interval; returns true when @p out_event is written. */
bool road_events_harsh_accel_finish(road_events_harsh_accel_detector_t *det,
                                    road_events_harsh_longitudinal_event_t *out_event);

/** Initialize a harsh braking detector with caller-owned storage. */
void road_events_harsh_brake_init(road_events_harsh_brake_detector_t *det,
                                  road_events_harsh_brake_config_t cfg);
/** Reset a harsh braking detector while preserving its configuration. */
void road_events_harsh_brake_reset(road_events_harsh_brake_detector_t *det);
/** Update a harsh braking detector; returns true when @p out_event is written. */
bool road_events_harsh_brake_update(road_events_harsh_brake_detector_t *det,
                                    road_events_harsh_longitudinal_sample_t sample,
                                    road_events_harsh_longitudinal_event_t *out_event);
/** Flush a pending harsh braking interval; returns true when @p out_event is written. */
bool road_events_harsh_brake_finish(road_events_harsh_brake_detector_t *det,
                                    road_events_harsh_longitudinal_event_t *out_event);

/** Initialize a harsh cornering detector with caller-owned storage. */
void road_events_harsh_corner_init(road_events_harsh_corner_detector_t *det,
                                   road_events_harsh_corner_config_t cfg);
/** Reset a harsh cornering detector while preserving its configuration. */
void road_events_harsh_corner_reset(road_events_harsh_corner_detector_t *det);
/** Update a harsh cornering detector; returns true when @p out_event is written. */
bool road_events_harsh_corner_update(road_events_harsh_corner_detector_t *det,
                                     road_events_harsh_corner_sample_t sample,
                                     road_events_harsh_corner_event_t *out_event);
/** Flush a pending harsh cornering interval; returns true when @p out_event is written. */
bool road_events_harsh_corner_finish(road_events_harsh_corner_detector_t *det,
                                     road_events_harsh_corner_event_t *out_event);

/** Initialize a road-roughness analyzer with caller-owned storage. */
void road_events_roughness_init(road_events_roughness_analyzer_t *analyzer,
                                road_events_roughness_config_t cfg);
/** Reset a road-roughness analyzer while preserving its configuration. */
void road_events_roughness_reset(road_events_roughness_analyzer_t *analyzer);
/** Update roughness estimate state; returns false only for invalid input. */
bool road_events_roughness_update(road_events_roughness_analyzer_t *analyzer,
                                  road_events_roughness_sample_t sample,
                                  road_events_roughness_estimate_t *out_estimate);
/** Update roughness and event state; returns false only for invalid input. */
bool road_events_roughness_update_with_events(road_events_roughness_analyzer_t *analyzer,
                                              road_events_roughness_sample_t sample,
                                              road_events_roughness_update_t *out_update);
/** Return the most recent roughness estimate. */
road_events_roughness_estimate_t
road_events_roughness_estimate(const road_events_roughness_analyzer_t *analyzer);
/** Flush a pending rough-road interval; returns true when @p out_event is written. */
bool road_events_roughness_finish(road_events_roughness_analyzer_t *analyzer,
                                  road_events_roughness_event_t *out_event);

/** Initialize trip statistics with caller-owned storage. */
void road_events_trip_stats_init(road_events_trip_stats_t *stats,
                                 road_events_trip_config_t cfg);
/** Reset trip statistics while preserving its configuration. */
void road_events_trip_stats_reset(road_events_trip_stats_t *stats);
/** Add one vehicle-motion sample to the trip statistics accumulator. */
void road_events_trip_stats_update_motion(road_events_trip_stats_t *stats,
                                          road_events_trip_sample_t sample);
/** Increment one trip event counter. */
void road_events_trip_stats_record_event(road_events_trip_stats_t *stats,
                                         road_events_trip_event_kind_t kind);
/** Return a snapshot of accumulated trip statistics. */
road_events_trip_summary_t
road_events_trip_stats_summary(const road_events_trip_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif
