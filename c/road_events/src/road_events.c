#include "road_events.h"

#include <math.h>

#define ROAD_EVENTS_TAU 6.28318530717958647692f

static float re_max(float a, float b) { return a > b ? a : b; }
static float re_min(float a, float b) { return a < b ? a : b; }
static float re_clamp(float x, float lo, float hi) { return re_min(re_max(x, lo), hi); }
static float re_abs(float x) { return fabsf(x); }
static bool re_finite(float x) { return isfinite(x) != 0; }
static float re_ratio_or_zero(float num, float den) { return den > 0.0f ? num / den : 0.0f; }

static float re_high_pass(float current, float last_input, float last_output, float cutoff_hz,
                          float dt)
{
    float rc = 1.0f / (ROAD_EVENTS_TAU * re_max(cutoff_hz, 1.0e-3f));
    float alpha = rc / (rc + dt);
    return alpha * (last_output + current - last_input);
}

static float re_low_pass(float previous, float current, float cutoff_hz, float dt)
{
    float rc = 1.0f / (ROAD_EVENTS_TAU * re_max(cutoff_hz, 1.0e-3f));
    float alpha = dt / (rc + dt);
    return (1.0f - alpha) * previous + alpha * current;
}

static float re_elapsed(bool *has_last_t_s, float *last_t_s, float t_s)
{
    float dt = *has_last_t_s ? re_clamp(t_s - *last_t_s, 0.0f, 0.2f) : 0.0f;
    *has_last_t_s = true;
    *last_t_s = t_s;
    return dt;
}

static float re_update_ema(float previous, float value, float tau_s, float dt)
{
    float alpha = dt / (re_max(tau_s, dt) + dt);
    return (1.0f - alpha) * previous + alpha * value;
}

static float re_update_abs_ema(float previous, float value, float tau_s, float dt)
{
    return re_update_ema(previous, re_abs(value), tau_s, dt);
}

static uint32_t re_sat_add_one(uint32_t value)
{
    return value == UINT32_MAX ? UINT32_MAX : value + 1u;
}

#define ROAD_EVENTS_CONTEXT_SIZE_FN(fn_name, type_name) \
    size_t fn_name##_size(void) { return sizeof(type_name); } \
    size_t fn_name##_alignment(void) \
    { \
        struct fn_name##_alignment_probe { \
            char c; \
            type_name value; \
        }; \
        return offsetof(struct fn_name##_alignment_probe, value); \
    }

ROAD_EVENTS_CONTEXT_SIZE_FN(road_events_speed_bump_detector,
                            road_events_speed_bump_detector_t)
ROAD_EVENTS_CONTEXT_SIZE_FN(road_events_hill_detector, road_events_hill_detector_t)
ROAD_EVENTS_CONTEXT_SIZE_FN(road_events_reverse_detector, road_events_reverse_detector_t)
ROAD_EVENTS_CONTEXT_SIZE_FN(road_events_harsh_accel_detector,
                            road_events_harsh_accel_detector_t)
ROAD_EVENTS_CONTEXT_SIZE_FN(road_events_harsh_brake_detector,
                            road_events_harsh_brake_detector_t)
ROAD_EVENTS_CONTEXT_SIZE_FN(road_events_harsh_corner_detector,
                            road_events_harsh_corner_detector_t)
ROAD_EVENTS_CONTEXT_SIZE_FN(road_events_roughness_analyzer, road_events_roughness_analyzer_t)
ROAD_EVENTS_CONTEXT_SIZE_FN(road_events_trip_stats, road_events_trip_stats_t)

road_events_speed_bump_config_t road_events_speed_bump_config_default(void)
{
    road_events_speed_bump_config_t cfg = {
        0.45f, 0.70f, 6.0f, 1.5f, 1.8f, 3.6f, 0.18f, 1.8f,
        3.5f, 1.5f, 0.25f, 0.25f, 3.0f, 0.12f, 4.0f};
    return cfg;
}

road_events_hill_config_t road_events_hill_config_default(void)
{
    road_events_hill_config_t cfg = {4.0f, 1.0f};
    return cfg;
}

road_events_reverse_config_t road_events_reverse_config_default(void)
{
    road_events_reverse_config_t cfg = {-0.5f, -0.2f, 0.5f, 0.5f, 1.0f};
    return cfg;
}

road_events_harsh_accel_config_t road_events_harsh_accel_config_default(void)
{
    road_events_harsh_accel_config_t cfg = {0.6f, 15.0f, 2.5f, 2.0f, 0.4f, 1.0f, 2.0f};
    return cfg;
}

road_events_harsh_brake_config_t road_events_harsh_brake_config_default(void)
{
    road_events_harsh_brake_config_t cfg = {0.6f, 15.0f, 3.0f, 2.4f, 0.4f, 1.0f, 2.0f};
    return cfg;
}

road_events_harsh_corner_config_t road_events_harsh_corner_config_default(void)
{
    road_events_harsh_corner_config_t cfg = {
        3.0f, 2.4f, 0.15f, 0.20f, 4.0f, 80.0f, 0.50f, 0.5f, 3.0f, 2.0f};
    return cfg;
}

road_events_roughness_config_t road_events_roughness_config_default(void)
{
    road_events_roughness_config_t cfg = {
        0.7f, 10.0f, 10.0f, 2.0f, 1.5f, 20.0f, 0.60f, 3.0f,
        2.5f, 6.0f, 0.45f, 0.02f, 0.65f, 0.50f, 0.60f, 0.42f,
        1.0f, 8.0f, 0.2f, 0.15f, 0.25f, 0.40f, 0.60f, 0.90f, 1.20f};
    return cfg;
}

road_events_trip_config_t road_events_trip_config_default(void)
{
    road_events_trip_config_t cfg = {0.5f, 0.2f, 5.0f, 1.0f};
    return cfg;
}

road_events_harsh_behavior_config_t road_events_harsh_behavior_preset_config(
    road_events_harsh_behavior_preset_t preset)
{
    road_events_harsh_behavior_config_t cfg;
    cfg.accel = road_events_harsh_accel_config_default();
    cfg.brake = road_events_harsh_brake_config_default();
    cfg.corner = road_events_harsh_corner_config_default();

    switch (preset) {
    case ROAD_EVENTS_HARSH_SENSITIVE:
        cfg.accel.accel_threshold_mps2 = 2.0f;
        cfg.accel.exit_accel_threshold_mps2 = 1.6f;
        cfg.brake.decel_threshold_mps2 = 2.5f;
        cfg.brake.exit_decel_threshold_mps2 = 2.0f;
        cfg.corner.lateral_accel_threshold_mps2 = 2.3f;
        cfg.corner.exit_lateral_accel_threshold_mps2 = 1.84f;
        cfg.corner.lateral_jerk_threshold_mps3 = 3.0f;
        break;
    case ROAD_EVENTS_HARSH_CONSERVATIVE:
        cfg.accel.accel_threshold_mps2 = 3.2f;
        cfg.accel.exit_accel_threshold_mps2 = 2.56f;
        cfg.brake.decel_threshold_mps2 = 4.0f;
        cfg.brake.exit_decel_threshold_mps2 = 3.2f;
        cfg.corner.lateral_accel_threshold_mps2 = 3.8f;
        cfg.corner.exit_lateral_accel_threshold_mps2 = 3.04f;
        cfg.corner.lateral_jerk_threshold_mps3 = 6.0f;
        break;
    case ROAD_EVENTS_HARSH_BALANCED:
    default:
        cfg.corner.lateral_accel_threshold_mps2 = 3.4f;
        cfg.corner.exit_lateral_accel_threshold_mps2 = 2.9f;
        cfg.corner.lateral_jerk_threshold_mps3 = 5.0f;
        break;
    }
    return cfg;
}

void road_events_speed_bump_init_default(road_events_speed_bump_detector_t *det)
{
    road_events_speed_bump_init(det, road_events_speed_bump_config_default());
}

void road_events_hill_init_default(road_events_hill_detector_t *det)
{
    road_events_hill_init(det, road_events_hill_config_default());
}

void road_events_reverse_init_default(road_events_reverse_detector_t *det)
{
    road_events_reverse_init(det, road_events_reverse_config_default());
}

void road_events_harsh_accel_init_preset(road_events_harsh_accel_detector_t *det,
                                         road_events_harsh_preset_t preset)
{
    road_events_harsh_accel_init(det, road_events_harsh_behavior_preset_config(preset).accel);
}

void road_events_harsh_brake_init_preset(road_events_harsh_brake_detector_t *det,
                                         road_events_harsh_preset_t preset)
{
    road_events_harsh_brake_init(det, road_events_harsh_behavior_preset_config(preset).brake);
}

void road_events_harsh_corner_init_preset(road_events_harsh_corner_detector_t *det,
                                          road_events_harsh_preset_t preset)
{
    road_events_harsh_corner_init(det, road_events_harsh_behavior_preset_config(preset).corner);
}

void road_events_roughness_init_default(road_events_roughness_analyzer_t *analyzer)
{
    road_events_roughness_init(analyzer, road_events_roughness_config_default());
}

void road_events_trip_stats_init_default(road_events_trip_stats_t *stats)
{
    road_events_trip_stats_init(stats, road_events_trip_config_default());
}

void road_events_speed_bump_init(road_events_speed_bump_detector_t *d,
                                 road_events_speed_bump_config_t cfg)
{
    *d = (road_events_speed_bump_detector_t){0};
    d->cfg = cfg;
    d->pitch_noise_deg = 0.03f;
    d->accel_noise_mps2 = 0.10f;
    d->last_event_t_s = -1.0e9f;
}

void road_events_speed_bump_reset(road_events_speed_bump_detector_t *d)
{
    road_events_speed_bump_config_t cfg = d->cfg;
    road_events_speed_bump_init(d, cfg);
}

static float re_speed_bump_min_accel_peak(const road_events_speed_bump_detector_t *d)
{
    return re_max(d->cfg.min_vertical_accel_peak_mps2, 0.1f);
}

static float re_speed_bump_accel_threshold(const road_events_speed_bump_detector_t *d)
{
    return re_max(d->cfg.vertical_accel_noise_peak_scale * d->accel_noise_mps2,
                  re_speed_bump_min_accel_peak(d));
}

static float re_speed_bump_pitch_threshold(const road_events_speed_bump_detector_t *d)
{
    return re_max(d->cfg.pitch_noise_peak_scale * d->pitch_noise_deg, 0.25f);
}

static bool re_speed_bump_alternating(float a, float b, float c)
{
    return ((a > 0.0f && c > 0.0f && b < 0.0f) || (a < 0.0f && c < 0.0f && b > 0.0f));
}

static void re_speed_bump_push_extremum(road_events_speed_bump_detector_t *d, float t_s,
                                        float accel_hpf_mps2, float pitch_peak_deg,
                                        float speed_mps, float active_time_s)
{
    if (d->extrema_len > 0u) {
        unsigned last_index = (d->extrema_len < 4u ? d->extrema_len : 4u) - 1u;
        bool same_polarity = (d->extrema[last_index].accel_hpf_mps2 > 0.0f) ==
                             (accel_hpf_mps2 > 0.0f);
        bool same_impulse = t_s - d->extrema[last_index].t_s < d->cfg.min_event_duration_s;
        if (same_polarity && same_impulse) {
            if (re_abs(accel_hpf_mps2) > re_abs(d->extrema[last_index].accel_hpf_mps2)) {
                d->extrema[last_index].valid = true;
                d->extrema[last_index].t_s = t_s;
                d->extrema[last_index].accel_hpf_mps2 = accel_hpf_mps2;
                d->extrema[last_index].pitch_peak_deg = pitch_peak_deg;
                d->extrema[last_index].speed_mps = speed_mps;
                d->extrema[last_index].active_time_s = active_time_s;
            } else {
                d->extrema[last_index].pitch_peak_deg =
                    re_max(d->extrema[last_index].pitch_peak_deg, pitch_peak_deg);
            }
            return;
        }
    }

    if (d->extrema_len < 4u) {
        d->extrema[d->extrema_len].valid = true;
        d->extrema[d->extrema_len].t_s = t_s;
        d->extrema[d->extrema_len].accel_hpf_mps2 = accel_hpf_mps2;
        d->extrema[d->extrema_len].pitch_peak_deg = pitch_peak_deg;
        d->extrema[d->extrema_len].speed_mps = speed_mps;
        d->extrema[d->extrema_len].active_time_s = active_time_s;
        d->extrema_len++;
    } else {
        unsigned i;
        for (i = 0u; i < 3u; i++) {
            d->extrema[i] = d->extrema[i + 1u];
        }
        d->extrema[3].valid = true;
        d->extrema[3].t_s = t_s;
        d->extrema[3].accel_hpf_mps2 = accel_hpf_mps2;
        d->extrema[3].pitch_peak_deg = pitch_peak_deg;
        d->extrema[3].speed_mps = speed_mps;
        d->extrema[3].active_time_s = active_time_s;
    }
}

static float re_speed_bump_pattern_score(const road_events_speed_bump_detector_t *d,
                                         float duration_s, const float speeds[3],
                                         const float accel_peaks[3],
                                         const float pitch_peaks[3], float active_duration_s,
                                         float active_fraction, float balance)
{
    float count;
    float speed_mps;
    float speed_min_s;
    float speed_max_s;
    float min_s;
    float max_s;
    float accel_peak;
    float pitch_peak;
    float accel_score;
    float pitch_score;
    float center;
    float half_width;
    float spacing_score;
    float shape_score;

    if (duration_s <= 0.0f) {
        return 0.0f;
    }
    count = speeds[2] > 0.0f ? 3.0f : 2.0f;
    speed_mps = (speeds[0] + speeds[1] + speeds[2]) / count;
    if (speed_mps < d->cfg.min_speed_mps) {
        return 0.0f;
    }
    speed_min_s = 0.35f * d->cfg.wheelbase_min_m / speed_mps;
    speed_max_s = 2.5f * d->cfg.wheelbase_max_m / speed_mps;
    min_s = re_max(d->cfg.min_event_duration_s, speed_min_s);
    max_s = re_min(d->cfg.max_event_duration_s, re_max(speed_max_s, min_s));
    if (duration_s < min_s || duration_s > max_s) {
        return 0.0f;
    }
    if (active_fraction < d->cfg.min_vertical_accel_active_fraction) {
        return 0.0f;
    }
    if (active_duration_s < d->cfg.min_vertical_accel_active_duration_s) {
        return 0.0f;
    }

    accel_peak = re_max(accel_peaks[0], re_max(accel_peaks[1], accel_peaks[2]));
    pitch_peak = re_max(pitch_peaks[0], re_max(pitch_peaks[1], pitch_peaks[2]));
    accel_score = re_clamp((accel_peak / re_speed_bump_accel_threshold(d)) - 1.0f, 0.0f, 1.0f);
    pitch_score = re_clamp((pitch_peak / re_speed_bump_pitch_threshold(d)) - 1.0f, 0.0f, 1.0f);
    center = 0.5f * (min_s + max_s);
    half_width = 0.5f * re_max(max_s - min_s, 1.0e-3f);
    spacing_score = re_clamp(1.0f - (re_abs(duration_s - center) / half_width), 0.0f, 1.0f);
    shape_score = re_clamp(0.45f * accel_score + 0.25f * spacing_score +
                               0.15f * re_clamp(balance, 0.0f, 1.0f) + 0.15f * pitch_score,
                           0.0f, 1.0f);
    return re_clamp(shape_score * pitch_score, 0.0f, 1.0f);
}

static bool re_speed_bump_evaluate(road_events_speed_bump_detector_t *d,
                                   road_events_speed_bump_event_t *event)
{
    unsigned start;
    float a_accel, b_accel, c_accel;
    float duration_s;
    float active_duration_s;
    float speeds[3];
    float accel_peaks[3];
    float pitch_peaks[3];
    float balance;
    float score;
    float t_s;
    float peak_pitch;
    float margin;

    if (d->extrema_len < 3u) {
        return false;
    }
    start = d->extrema_len - 3u;
    a_accel = d->extrema[start].accel_hpf_mps2;
    b_accel = d->extrema[start + 1u].accel_hpf_mps2;
    c_accel = d->extrema[start + 2u].accel_hpf_mps2;
    duration_s = d->extrema[start + 2u].t_s - d->extrema[start].t_s;
    t_s = d->extrema[start + 1u].t_s;
    if (!re_speed_bump_alternating(a_accel, b_accel, c_accel)) {
        return false;
    }

    active_duration_s = d->extrema[start + 2u].active_time_s - d->extrema[start].active_time_s;
    speeds[0] = d->extrema[start].speed_mps;
    speeds[1] = d->extrema[start + 1u].speed_mps;
    speeds[2] = d->extrema[start + 2u].speed_mps;
    accel_peaks[0] = re_abs(a_accel);
    accel_peaks[1] = re_abs(b_accel);
    accel_peaks[2] = re_abs(c_accel);
    pitch_peaks[0] = d->extrema[start].pitch_peak_deg;
    pitch_peaks[1] = d->extrema[start + 1u].pitch_peak_deg;
    pitch_peaks[2] = d->extrema[start + 2u].pitch_peak_deg;
    balance = re_min(re_abs(a_accel), re_abs(c_accel)) / re_max(re_max(re_abs(a_accel), re_abs(c_accel)), 1.0e-3f);
    score = re_speed_bump_pattern_score(d, duration_s, speeds, accel_peaks, pitch_peaks,
                                        active_duration_s, active_duration_s / re_max(duration_s, 1.0e-3f),
                                        balance);

    if (t_s - d->last_event_t_s < d->cfg.refractory_s || score < d->cfg.trigger_confidence) {
        return false;
    }
    d->last_event_t_s = t_s;
    peak_pitch = re_max(pitch_peaks[0], re_max(pitch_peaks[1], pitch_peaks[2]));
    margin = re_clamp((score - d->cfg.trigger_confidence) /
                          re_max(1.0f - d->cfg.trigger_confidence, 1.0e-3f),
                      0.0f, 1.0f);
    if (event) {
        event->t_s = t_s;
        event->confidence = 0.90f + 0.08f * margin;
        event->duration_s = duration_s;
        event->peak_abs_pitch_deg = peak_pitch;
    }
    return true;
}

bool road_events_speed_bump_update(road_events_speed_bump_detector_t *d,
                                   road_events_speed_bump_sample_t sample,
                                   road_events_speed_bump_diagnostic_t *diagnostic,
                                   road_events_speed_bump_event_t *event)
{
    float dt;
    float pitch_hpf_deg;
    float accel_hpf_mps2;
    float slope_dt;
    float slope;
    bool crossed_peak;
    bool crossed_valley;
    bool emitted = false;

    if (diagnostic) {
        *diagnostic = (road_events_speed_bump_diagnostic_t){0};
    }
    if (!re_finite(sample.t_s) || !re_finite(sample.speed_mps) || !re_finite(sample.pitch_deg) ||
        !re_finite(sample.vertical_accel_mps2)) {
        return false;
    }
    if (!d->has_last_t_s) {
        d->has_last_t_s = true;
        d->last_t_s = sample.t_s;
        d->last_pitch_deg = sample.pitch_deg;
        d->last_vertical_accel_mps2 = sample.vertical_accel_mps2;
        d->prev_t_s = sample.t_s;
        if (diagnostic) {
            diagnostic->t_s = sample.t_s;
        }
        return false;
    }

    dt = re_clamp(sample.t_s - d->last_t_s, 1.0e-3f, 0.2f);
    pitch_hpf_deg = re_high_pass(sample.pitch_deg, d->last_pitch_deg, d->last_pitch_hpf_deg,
                                 d->cfg.pitch_hpf_cutoff_hz, dt);
    accel_hpf_mps2 = re_high_pass(sample.vertical_accel_mps2, d->last_vertical_accel_mps2,
                                  d->last_accel_hpf_mps2,
                                  d->cfg.vertical_accel_hpf_cutoff_hz, dt);
    d->pitch_noise_deg = re_update_abs_ema(d->pitch_noise_deg, pitch_hpf_deg, d->cfg.noise_tau_s, dt);
    d->accel_noise_mps2 = re_update_abs_ema(d->accel_noise_mps2, accel_hpf_mps2,
                                            d->cfg.noise_tau_s, dt);
    if (re_abs(accel_hpf_mps2) >= re_speed_bump_min_accel_peak(d)) {
        d->accel_active_time_s += dt;
    }
    d->pitch_peak_since_extremum_deg =
        re_max(d->pitch_peak_since_extremum_deg, re_abs(pitch_hpf_deg));

    slope_dt = re_max(sample.t_s - d->prev_t_s, 1.0e-3f);
    slope = (accel_hpf_mps2 - d->prev_accel_hpf_mps2) / slope_dt;
    crossed_peak = d->prev_accel_slope > 0.0f && slope <= 0.0f;
    crossed_valley = d->prev_accel_slope < 0.0f && slope >= 0.0f;
    d->prev_accel_slope = slope;
    if ((crossed_peak || crossed_valley) &&
        re_abs(d->prev_accel_hpf_mps2) >= re_speed_bump_min_accel_peak(d)) {
        re_speed_bump_push_extremum(d, d->prev_t_s, d->prev_accel_hpf_mps2,
                                    d->pitch_peak_since_extremum_deg, sample.speed_mps,
                                    d->accel_active_time_s);
        d->pitch_peak_since_extremum_deg = 0.0f;
        emitted = re_speed_bump_evaluate(d, event);
    }

    d->last_t_s = sample.t_s;
    d->last_pitch_deg = sample.pitch_deg;
    d->last_vertical_accel_mps2 = sample.vertical_accel_mps2;
    d->last_pitch_hpf_deg = pitch_hpf_deg;
    d->last_accel_hpf_mps2 = accel_hpf_mps2;
    d->prev_accel_hpf_mps2 = accel_hpf_mps2;
    d->prev_t_s = sample.t_s;

    if (diagnostic) {
        diagnostic->t_s = sample.t_s;
        diagnostic->pitch_hpf_deg = pitch_hpf_deg;
        diagnostic->pitch_noise_deg = d->pitch_noise_deg;
        diagnostic->vertical_accel_hpf_mps2 = accel_hpf_mps2;
        diagnostic->vertical_accel_noise_mps2 = d->accel_noise_mps2;
    }
    return emitted;
}

void road_events_hill_init(road_events_hill_detector_t *d, road_events_hill_config_t cfg)
{
    *d = (road_events_hill_detector_t){0};
    d->cfg = cfg;
}

void road_events_hill_reset(road_events_hill_detector_t *d)
{
    road_events_hill_config_t cfg = d->cfg;
    road_events_hill_init(d, cfg);
}

static bool re_hill_event_from_active(const road_events_hill_detector_t *d,
                                      road_events_hill_event_t *event)
{
    float duration_s = re_max(d->active_duration_s, 0.0f);
    if (event) {
        event->kind = d->active_kind;
        event->start_t_s = d->active_start_t_s;
        event->end_t_s = d->active_last_t_s;
        event->duration_s = duration_s;
        event->mean_pitch_deg = re_ratio_or_zero(d->active_pitch_time_sum_deg_s, duration_s);
        event->peak_abs_pitch_deg = d->active_peak_abs_pitch_deg;
        event->mean_speed_mps = re_ratio_or_zero(d->active_speed_time_sum_m, duration_s);
    }
    return true;
}

static bool re_hill_finish_active(road_events_hill_detector_t *d, road_events_hill_event_t *event)
{
    bool should_emit = !d->active_emitted && d->active_duration_s >= d->cfg.min_duration_s;
    if (should_emit) {
        re_hill_event_from_active(d, event);
    }
    d->has_active = false;
    return should_emit;
}

bool road_events_hill_update(road_events_hill_detector_t *d, road_events_hill_sample_t sample,
                             road_events_hill_event_t *event)
{
    float dt;
    bool has_kind;
    road_events_hill_kind_t kind = ROAD_EVENTS_HILL_UPHILL;
    bool emitted = false;

    if (!re_finite(sample.t_s) || !re_finite(sample.speed_mps) || !re_finite(sample.pitch_deg)) {
        return false;
    }
    dt = re_elapsed(&d->has_last_t_s, &d->last_t_s, sample.t_s);
    has_kind = true;
    if (sample.pitch_deg >= d->cfg.pitch_threshold_deg) {
        kind = ROAD_EVENTS_HILL_UPHILL;
    } else if (sample.pitch_deg <= -d->cfg.pitch_threshold_deg) {
        kind = ROAD_EVENTS_HILL_DOWNHILL;
    } else {
        has_kind = false;
    }

    if (d->has_active && has_kind && d->active_kind == kind) {
        d->active_last_t_s = sample.t_s;
        d->active_pitch_time_sum_deg_s += sample.pitch_deg * dt;
        d->active_speed_time_sum_m += re_max(sample.speed_mps, 0.0f) * dt;
        d->active_duration_s += dt;
        d->active_peak_abs_pitch_deg = re_max(d->active_peak_abs_pitch_deg, re_abs(sample.pitch_deg));
        if (!d->active_emitted && d->active_duration_s >= d->cfg.min_duration_s) {
            d->active_emitted = true;
            emitted = re_hill_event_from_active(d, event);
        }
        return emitted;
    }

    if (d->has_active) {
        emitted = re_hill_finish_active(d, event);
    }
    if (has_kind) {
        d->has_active = true;
        d->active_kind = kind;
        d->active_start_t_s = sample.t_s;
        d->active_last_t_s = sample.t_s;
        d->active_pitch_time_sum_deg_s = 0.0f;
        d->active_speed_time_sum_m = 0.0f;
        d->active_duration_s = 0.0f;
        d->active_peak_abs_pitch_deg = re_abs(sample.pitch_deg);
        d->active_emitted = false;
    }
    return emitted;
}

bool road_events_hill_finish(road_events_hill_detector_t *d, road_events_hill_event_t *event)
{
    if (!d->has_active) {
        return false;
    }
    return re_hill_finish_active(d, event);
}

void road_events_reverse_init(road_events_reverse_detector_t *d, road_events_reverse_config_t cfg)
{
    *d = (road_events_reverse_detector_t){0};
    d->cfg = cfg;
}

void road_events_reverse_reset(road_events_reverse_detector_t *d)
{
    road_events_reverse_config_t cfg = d->cfg;
    road_events_reverse_init(d, cfg);
}

static void re_reverse_add_sample(road_events_reverse_detector_t *d,
                                  road_events_reverse_sample_t sample, float dt)
{
    float reverse_speed_mps = re_max(-sample.forward_velocity_mps, 0.0f);
    d->active_last_t_s = sample.t_s;
    d->active_reverse_speed_time_sum_m += reverse_speed_mps * dt;
    d->active_duration_s += dt;
    d->active_peak_reverse_speed_mps = re_max(d->active_peak_reverse_speed_mps, reverse_speed_mps);
}

static bool re_reverse_finish_active(road_events_reverse_detector_t *d,
                                     road_events_reverse_event_t *event)
{
    bool should_emit = d->active_duration_s >= d->cfg.min_duration_s;
    if (should_emit && event) {
        event->start_t_s = d->active_start_t_s;
        event->end_t_s = d->active_last_t_s;
        event->duration_s = re_max(d->active_duration_s, 0.0f);
        event->mean_reverse_speed_mps =
            re_ratio_or_zero(d->active_reverse_speed_time_sum_m, event->duration_s);
        event->peak_reverse_speed_mps = d->active_peak_reverse_speed_mps;
    }
    d->has_active = false;
    return should_emit;
}

bool road_events_reverse_update(road_events_reverse_detector_t *d,
                                road_events_reverse_sample_t sample,
                                road_events_reverse_event_t *event)
{
    float dt;
    bool entering;
    bool staying;
    float reverse_speed_mps;

    if (!re_finite(sample.t_s) || !re_finite(sample.forward_velocity_mps)) {
        return false;
    }
    dt = re_elapsed(&d->has_last_t_s, &d->last_t_s, sample.t_s);
    entering = sample.forward_velocity_mps < d->cfg.enter_forward_velocity_mps;
    staying = sample.forward_velocity_mps < d->cfg.exit_forward_velocity_mps;

    if (!d->has_active) {
        if (entering) {
            reverse_speed_mps = re_max(-sample.forward_velocity_mps, 0.0f);
            d->has_active = true;
            d->active_start_t_s = sample.t_s;
            d->active_last_t_s = sample.t_s;
            d->active_reverse_speed_time_sum_m = 0.0f;
            d->active_duration_s = 0.0f;
            d->active_peak_reverse_speed_mps = reverse_speed_mps;
            d->enter_duration_s = 0.0f;
            d->exit_duration_s = 0.0f;
            d->confirmed = false;
        }
        return false;
    }

    if (d->confirmed) {
        re_reverse_add_sample(d, sample, dt);
        if (staying) {
            d->exit_duration_s = 0.0f;
            return false;
        }
        d->exit_duration_s += dt;
        if (d->exit_duration_s >= d->cfg.exit_debounce_s) {
            d->enter_duration_s = 0.0f;
            d->exit_duration_s = 0.0f;
            d->confirmed = false;
            return re_reverse_finish_active(d, event);
        }
        return false;
    }

    if (entering) {
        re_reverse_add_sample(d, sample, dt);
        d->enter_duration_s += dt;
        if (d->enter_duration_s >= d->cfg.enter_debounce_s) {
            d->confirmed = true;
        }
    } else {
        d->has_active = false;
        d->enter_duration_s = 0.0f;
        d->exit_duration_s = 0.0f;
        d->confirmed = false;
    }
    return false;
}

bool road_events_reverse_finish(road_events_reverse_detector_t *d, road_events_reverse_event_t *event)
{
    bool confirmed;
    if (!d->has_active) {
        return false;
    }
    confirmed = d->confirmed;
    d->enter_duration_s = 0.0f;
    d->exit_duration_s = 0.0f;
    d->confirmed = false;
    if (!confirmed) {
        d->has_active = false;
        return false;
    }
    return re_reverse_finish_active(d, event);
}

static void re_harsh_long_state_init(road_events_harsh_longitudinal_state_t *s)
{
    *s = (road_events_harsh_longitudinal_state_t){0};
    s->tracker_last_event_t_s = -1.0e9f;
}

void road_events_harsh_accel_init(road_events_harsh_accel_detector_t *d,
                                  road_events_harsh_accel_config_t cfg)
{
    d->cfg = cfg;
    re_harsh_long_state_init(&d->state);
}

void road_events_harsh_accel_reset(road_events_harsh_accel_detector_t *d)
{
    road_events_harsh_accel_config_t cfg = d->cfg;
    road_events_harsh_accel_init(d, cfg);
}

void road_events_harsh_brake_init(road_events_harsh_brake_detector_t *d,
                                  road_events_harsh_brake_config_t cfg)
{
    d->cfg = cfg;
    re_harsh_long_state_init(&d->state);
}

void road_events_harsh_brake_reset(road_events_harsh_brake_detector_t *d)
{
    road_events_harsh_brake_config_t cfg = d->cfg;
    road_events_harsh_brake_init(d, cfg);
}

static bool re_longitudinal_ema_update(road_events_harsh_longitudinal_state_t *s,
                                       road_events_harsh_longitudinal_sample_t sample,
                                       float tau_s, float max_raw_accel_mps2,
                                       float *accel_mps2, float *speed_mps)
{
    float dt;
    float raw_accel;
    if (!s->has_last_t_s) {
        s->has_last_t_s = true;
        s->last_t_s = sample.t_s;
        s->last_forward_velocity_mps = sample.forward_velocity_mps;
        return false;
    }
    dt = re_clamp(sample.t_s - s->last_t_s, 0.0f, 0.2f);
    s->last_t_s = sample.t_s;
    *speed_mps = re_max(re_abs(sample.forward_velocity_mps), re_abs(s->last_forward_velocity_mps));
    if (dt <= 1.0e-4f) {
        s->last_forward_velocity_mps = sample.forward_velocity_mps;
        return false;
    }
    raw_accel = re_clamp((sample.forward_velocity_mps - s->last_forward_velocity_mps) / dt,
                         -max_raw_accel_mps2, max_raw_accel_mps2);
    s->last_forward_velocity_mps = sample.forward_velocity_mps;
    if (s->initialized) {
        s->accel_ema_mps2 = re_update_ema(s->accel_ema_mps2, raw_accel, tau_s, dt);
    } else {
        s->initialized = true;
        s->accel_ema_mps2 = raw_accel;
    }
    *accel_mps2 = s->accel_ema_mps2;
    return true;
}

static void re_metric_start(road_events_harsh_longitudinal_state_t *s, float t_s, float metric,
                            float speed_mps, float velocity_mps)
{
    s->tracker_has_active = true;
    s->active_start_t_s = t_s;
    s->active_last_t_s = t_s;
    s->active_duration_s = 0.0f;
    s->active_metric_time_sum = 0.0f;
    s->active_peak_metric = metric;
    s->active_speed_time_sum_m = 0.0f;
    s->active_peak_speed_mps = speed_mps;
    s->active_start_velocity_mps = velocity_mps;
    s->active_end_velocity_mps = velocity_mps;
}

static void re_metric_add(road_events_harsh_longitudinal_state_t *s, float t_s, float dt,
                          float metric, float speed_mps, float velocity_mps)
{
    s->active_last_t_s = t_s;
    s->active_duration_s += dt;
    s->active_metric_time_sum += metric * dt;
    s->active_peak_metric = re_max(s->active_peak_metric, metric);
    s->active_speed_time_sum_m += speed_mps * dt;
    s->active_peak_speed_mps = re_max(s->active_peak_speed_mps, speed_mps);
    s->active_end_velocity_mps = velocity_mps;
}

static bool re_metric_finish(road_events_harsh_longitudinal_state_t *s, float min_duration_s,
                             road_events_harsh_longitudinal_event_t *event)
{
    bool emit = s->active_duration_s >= min_duration_s;
    if (emit) {
        s->tracker_last_event_t_s = s->active_last_t_s;
        if (event) {
            event->start_t_s = s->active_start_t_s;
            event->end_t_s = s->active_last_t_s;
            event->duration_s = s->active_duration_s;
            event->delta_velocity_mps = s->active_end_velocity_mps - s->active_start_velocity_mps;
            event->mean_accel_mps2 = re_ratio_or_zero(s->active_metric_time_sum, s->active_duration_s);
            event->peak_accel_mps2 = s->active_peak_metric;
            event->mean_speed_mps = re_ratio_or_zero(s->active_speed_time_sum_m, s->active_duration_s);
            event->peak_speed_mps = s->active_peak_speed_mps;
        }
    }
    s->tracker_has_active = false;
    return emit;
}

static bool re_metric_update_long(road_events_harsh_longitudinal_state_t *s, float t_s,
                                  float metric, float speed_mps, float velocity_mps,
                                  float enter_threshold, float exit_threshold,
                                  float min_duration_s, float refractory_s,
                                  road_events_harsh_longitudinal_event_t *event)
{
    float dt = re_elapsed(&s->tracker_has_last_t_s, &s->tracker_last_t_s, t_s);
    bool above_enter = metric >= enter_threshold;
    bool above_exit = metric >= exit_threshold;
    if (s->tracker_has_active) {
        if (above_exit) {
            re_metric_add(s, t_s, dt, metric, speed_mps, velocity_mps);
            return false;
        }
        return re_metric_finish(s, min_duration_s, event);
    }
    if (above_enter && t_s - s->tracker_last_event_t_s >= refractory_s) {
        re_metric_start(s, t_s, metric, speed_mps, velocity_mps);
    }
    return false;
}

bool road_events_harsh_accel_update(road_events_harsh_accel_detector_t *d,
                                    road_events_harsh_longitudinal_sample_t sample,
                                    road_events_harsh_longitudinal_event_t *event)
{
    float accel;
    float speed;
    float metric;
    if (!re_finite(sample.t_s) || !re_finite(sample.forward_velocity_mps)) {
        return false;
    }
    if (!re_longitudinal_ema_update(&d->state, sample, d->cfg.accel_tau_s,
                                    d->cfg.max_raw_accel_mps2, &accel, &speed)) {
        return false;
    }
    metric = speed < d->cfg.min_speed_mps ? 0.0f : re_max(accel, 0.0f);
    return re_metric_update_long(&d->state, sample.t_s, metric, speed, sample.forward_velocity_mps,
                                 d->cfg.accel_threshold_mps2, d->cfg.exit_accel_threshold_mps2,
                                 d->cfg.min_duration_s, d->cfg.refractory_s, event);
}

bool road_events_harsh_accel_finish(road_events_harsh_accel_detector_t *d,
                                    road_events_harsh_longitudinal_event_t *event)
{
    return d->state.tracker_has_active ? re_metric_finish(&d->state, d->cfg.min_duration_s, event) : false;
}

bool road_events_harsh_brake_update(road_events_harsh_brake_detector_t *d,
                                    road_events_harsh_longitudinal_sample_t sample,
                                    road_events_harsh_longitudinal_event_t *event)
{
    float accel;
    float speed;
    float metric;
    if (!re_finite(sample.t_s) || !re_finite(sample.forward_velocity_mps)) {
        return false;
    }
    if (!re_longitudinal_ema_update(&d->state, sample, d->cfg.accel_tau_s,
                                    d->cfg.max_raw_accel_mps2, &accel, &speed)) {
        return false;
    }
    metric = speed < d->cfg.min_speed_mps ? 0.0f : re_max(-accel, 0.0f);
    return re_metric_update_long(&d->state, sample.t_s, metric, speed, sample.forward_velocity_mps,
                                 d->cfg.decel_threshold_mps2, d->cfg.exit_decel_threshold_mps2,
                                 d->cfg.min_duration_s, d->cfg.refractory_s, event);
}

bool road_events_harsh_brake_finish(road_events_harsh_brake_detector_t *d,
                                    road_events_harsh_longitudinal_event_t *event)
{
    return d->state.tracker_has_active ? re_metric_finish(&d->state, d->cfg.min_duration_s, event) : false;
}

void road_events_harsh_corner_init(road_events_harsh_corner_detector_t *d,
                                   road_events_harsh_corner_config_t cfg)
{
    *d = (road_events_harsh_corner_detector_t){0};
    d->cfg = cfg;
    d->tracker_last_event_t_s = -1.0e9f;
    d->last_jerk_trigger_t_s = -1.0e9f;
}

void road_events_harsh_corner_reset(road_events_harsh_corner_detector_t *d)
{
    road_events_harsh_corner_config_t cfg = d->cfg;
    road_events_harsh_corner_init(d, cfg);
}

static bool re_corner_metric_finish(road_events_harsh_corner_detector_t *d,
                                    road_events_harsh_corner_event_t *event)
{
    bool emit = d->active_duration_s >= d->cfg.min_duration_s;
    if (emit) {
        d->tracker_last_event_t_s = d->active_last_t_s;
        if (event) {
            event->start_t_s = d->active_start_t_s;
            event->end_t_s = d->active_last_t_s;
            event->duration_s = d->active_duration_s;
            event->mean_lateral_accel_mps2 =
                re_ratio_or_zero(d->active_metric_time_sum, d->active_duration_s);
            event->peak_lateral_accel_mps2 = d->active_peak_metric;
            event->mean_speed_mps = re_ratio_or_zero(d->active_speed_time_sum_m, d->active_duration_s);
            event->peak_speed_mps = d->active_peak_speed_mps;
        }
    }
    d->tracker_has_active = false;
    return emit;
}

static bool re_corner_metric_update(road_events_harsh_corner_detector_t *d, float t_s,
                                    float metric, float speed_mps,
                                    road_events_harsh_corner_event_t *event)
{
    float dt = re_elapsed(&d->tracker_has_last_t_s, &d->tracker_last_t_s, t_s);
    bool above_enter = metric >= d->cfg.lateral_accel_threshold_mps2;
    bool above_exit = metric >= d->cfg.exit_lateral_accel_threshold_mps2;
    if (d->tracker_has_active) {
        if (above_exit) {
            d->active_last_t_s = t_s;
            d->active_duration_s += dt;
            d->active_metric_time_sum += metric * dt;
            d->active_peak_metric = re_max(d->active_peak_metric, metric);
            d->active_speed_time_sum_m += speed_mps * dt;
            d->active_peak_speed_mps = re_max(d->active_peak_speed_mps, speed_mps);
            d->active_end_velocity_mps = speed_mps;
            return false;
        }
        return re_corner_metric_finish(d, event);
    }
    if (above_enter && t_s - d->tracker_last_event_t_s >= d->cfg.refractory_s) {
        d->tracker_has_active = true;
        d->active_start_t_s = t_s;
        d->active_last_t_s = t_s;
        d->active_duration_s = 0.0f;
        d->active_metric_time_sum = 0.0f;
        d->active_peak_metric = metric;
        d->active_speed_time_sum_m = 0.0f;
        d->active_peak_speed_mps = speed_mps;
        d->active_start_velocity_mps = speed_mps;
        d->active_end_velocity_mps = speed_mps;
    }
    return false;
}

bool road_events_harsh_corner_update(road_events_harsh_corner_detector_t *d,
                                     road_events_harsh_corner_sample_t sample,
                                     road_events_harsh_corner_event_t *event)
{
    float speed_mps;
    float dt;
    float previous_lateral;
    float raw_jerk;
    float load_mps2;
    bool speed_valid;
    bool jerk_recent;
    float metric_mps2;

    if (!re_finite(sample.t_s) || !re_finite(sample.speed_mps) ||
        !re_finite(sample.lateral_accel_mps2)) {
        return false;
    }
    speed_mps = re_abs(sample.speed_mps);
    if (!d->lateral_has_last_t_s) {
        d->lateral_has_last_t_s = true;
        d->lateral_last_t_s = sample.t_s;
        d->lateral_accel_ema_mps2 = sample.lateral_accel_mps2;
        d->lateral_initialized = true;
        return re_corner_metric_update(d, sample.t_s, 0.0f, speed_mps, event);
    }
    dt = re_clamp(sample.t_s - d->lateral_last_t_s, 0.0f, 0.2f);
    d->lateral_last_t_s = sample.t_s;
    if (dt <= 1.0e-4f) {
        return false;
    }
    if (!d->lateral_initialized) {
        d->lateral_accel_ema_mps2 = sample.lateral_accel_mps2;
        d->lateral_initialized = true;
    } else {
        previous_lateral = d->lateral_accel_ema_mps2;
        d->lateral_accel_ema_mps2 = re_update_ema(d->lateral_accel_ema_mps2,
                                                  sample.lateral_accel_mps2,
                                                  d->cfg.lateral_accel_tau_s, dt);
        raw_jerk = re_clamp((d->lateral_accel_ema_mps2 - previous_lateral) / dt,
                            -d->cfg.max_raw_lateral_jerk_mps3,
                            d->cfg.max_raw_lateral_jerk_mps3);
        if (d->jerk_initialized) {
            d->jerk_abs_ema_mps3 = re_update_ema(d->jerk_abs_ema_mps3, re_abs(raw_jerk),
                                                 d->cfg.lateral_jerk_tau_s, dt);
        } else {
            d->jerk_initialized = true;
            d->jerk_abs_ema_mps3 = re_abs(raw_jerk);
        }
    }

    load_mps2 = re_abs(d->lateral_accel_ema_mps2);
    if (speed_mps >= d->cfg.min_speed_mps &&
        d->jerk_abs_ema_mps3 >= d->cfg.lateral_jerk_threshold_mps3) {
        d->last_jerk_trigger_t_s = sample.t_s;
    }
    jerk_recent = sample.t_s - d->last_jerk_trigger_t_s <= d->cfg.jerk_trigger_window_s;
    speed_valid = speed_mps >= d->cfg.min_speed_mps;
    metric_mps2 = (speed_valid && (d->tracker_has_active ||
                                   (load_mps2 >= d->cfg.lateral_accel_threshold_mps2 &&
                                    jerk_recent)))
                      ? load_mps2
                      : 0.0f;
    return re_corner_metric_update(d, sample.t_s, metric_mps2, speed_mps, event);
}

bool road_events_harsh_corner_finish(road_events_harsh_corner_detector_t *d,
                                     road_events_harsh_corner_event_t *event)
{
    return d->tracker_has_active ? re_corner_metric_finish(d, event) : false;
}

static road_events_roughness_level_t re_level_for_rms(road_events_roughness_config_t cfg,
                                                     float rms)
{
    if (rms < cfg.very_smooth_threshold_mps2) return ROAD_EVENTS_ROUGHNESS_VERY_SMOOTH;
    if (rms < cfg.smooth_threshold_mps2) return ROAD_EVENTS_ROUGHNESS_SMOOTH;
    if (rms < cfg.light_texture_threshold_mps2) return ROAD_EVENTS_ROUGHNESS_LIGHT_TEXTURE;
    if (rms < cfg.moderate_threshold_mps2) return ROAD_EVENTS_ROUGHNESS_MODERATE;
    if (rms < cfg.rough_threshold_mps2) return ROAD_EVENTS_ROUGHNESS_ROUGH;
    if (rms < cfg.very_rough_threshold_mps2) return ROAD_EVENTS_ROUGHNESS_VERY_ROUGH;
    return ROAD_EVENTS_ROUGHNESS_SEVERE;
}

void road_events_roughness_init(road_events_roughness_analyzer_t *a,
                                road_events_roughness_config_t cfg)
{
    *a = (road_events_roughness_analyzer_t){0};
    a->cfg = cfg;
    a->last_rough_event_t_s = -1.0e9f;
    a->last_shock_event_t_s = -1.0e9f;
    a->last_estimate.level = ROAD_EVENTS_ROUGHNESS_VERY_SMOOTH;
}

void road_events_roughness_reset(road_events_roughness_analyzer_t *a)
{
    road_events_roughness_config_t cfg = a->cfg;
    road_events_roughness_init(a, cfg);
}

static float re_roughness_cap(const road_events_roughness_analyzer_t *a)
{
    float baseline_cap = a->baseline_initialized
                             ? re_max(a->cfg.robust_cap_scale, 0.0f) * re_max(a->baseline_abs_mps2, 0.0f)
                             : a->cfg.robust_min_cap_mps2;
    return re_max(re_min(re_max(baseline_cap, a->cfg.robust_min_cap_mps2), a->cfg.clip_mps2),
                  1.0e-3f);
}

static float re_distance_ema(float previous, float value, float tau_m, float ds)
{
    float alpha = ds / (re_max(tau_m, ds) + ds);
    return (1.0f - alpha) * previous + alpha * value;
}

static road_events_roughness_estimate_t re_roughness_make_estimate(
    road_events_roughness_analyzer_t *a, float t_s, float bandpass, float clipped, bool updated)
{
    float rms = sqrtf(re_max(a->energy_mps2_sq, 0.0f));
    road_events_roughness_estimate_t estimate;
    estimate.t_s = t_s;
    estimate.roughness_rms_mps2 = rms;
    estimate.level = re_level_for_rms(a->cfg, rms);
    estimate.vertical_accel_bandpass_mps2 = bandpass;
    estimate.vertical_accel_clipped_mps2 = clipped;
    estimate.distance_m = a->distance_m;
    estimate.updated = updated;
    a->last_estimate = estimate;
    return estimate;
}

static bool re_shock_event_from_active(const road_events_roughness_analyzer_t *a,
                                       road_events_shock_event_t *event)
{
    if (event) {
        event->start_t_s = a->shock_start_t_s;
        event->end_t_s = a->shock_last_t_s;
        event->duration_s = a->shock_duration_s;
        event->peak_abs_vertical_accel_mps2 = a->shock_peak_abs_vertical_accel_mps2;
        event->mean_speed_mps = re_ratio_or_zero(a->shock_speed_time_sum_m, a->shock_duration_s);
    }
    return true;
}

static bool re_finish_shock(road_events_roughness_analyzer_t *a, bool emit,
                            road_events_shock_event_t *event)
{
    bool should_emit = emit && a->shock_duration_s >= a->cfg.shock_min_duration_s &&
                       a->shock_duration_s <= a->cfg.shock_max_duration_s;
    if (should_emit) {
        a->last_shock_event_t_s = a->shock_last_t_s;
        re_shock_event_from_active(a, event);
    }
    a->shock_has_active = false;
    return should_emit;
}

static bool re_update_shock(road_events_roughness_analyzer_t *a, float t_s, float dt,
                            float abs_bandpass, float speed_mps, road_events_shock_event_t *event)
{
    float baseline_threshold = a->baseline_initialized
                                   ? re_max(a->cfg.shock_baseline_scale, 0.0f) *
                                         re_max(a->baseline_abs_mps2, 0.0f)
                                   : 0.0f;
    float enter_threshold = re_max(a->cfg.shock_min_peak_mps2, baseline_threshold);
    float exit_threshold = enter_threshold * re_clamp(a->cfg.shock_exit_fraction, 0.05f, 0.95f);
    bool above_enter = abs_bandpass >= enter_threshold;
    bool above_exit = abs_bandpass >= exit_threshold;

    if (a->shock_has_active) {
        if (above_exit) {
            a->shock_last_t_s = t_s;
            a->shock_duration_s += dt;
            a->shock_peak_abs_vertical_accel_mps2 =
                re_max(a->shock_peak_abs_vertical_accel_mps2, abs_bandpass);
            a->shock_speed_time_sum_m += speed_mps * dt;
            return false;
        }
        return re_finish_shock(a, true, event);
    }
    if (above_enter && t_s - a->last_shock_event_t_s >= a->cfg.shock_refractory_s) {
        a->shock_has_active = true;
        a->shock_start_t_s = t_s;
        a->shock_last_t_s = t_s;
        a->shock_duration_s = 0.0f;
        a->shock_peak_abs_vertical_accel_mps2 = abs_bandpass;
        a->shock_speed_time_sum_m = 0.0f;
    }
    return false;
}

static bool re_roughness_event_from_active(const road_events_roughness_analyzer_t *a,
                                           road_events_roughness_event_t *event)
{
    if (event) {
        event->start_t_s = a->rough_start_t_s;
        event->end_t_s = a->rough_last_t_s;
        event->duration_s = a->rough_duration_s;
        event->mean_roughness_rms_mps2 =
            re_ratio_or_zero(a->rough_roughness_time_sum, a->rough_duration_s);
        event->peak_roughness_rms_mps2 = a->rough_peak_roughness_rms_mps2;
        event->mean_speed_mps = re_ratio_or_zero(a->rough_speed_time_sum_m, a->rough_duration_s);
        event->distance_m = a->rough_distance_m;
    }
    return true;
}

static bool re_finish_roughness(road_events_roughness_analyzer_t *a, bool emit,
                                road_events_roughness_event_t *event)
{
    bool should_emit = emit && a->rough_duration_s >= a->cfg.rough_event_min_duration_s;
    if (should_emit) {
        a->last_rough_event_t_s = a->rough_last_t_s;
        re_roughness_event_from_active(a, event);
    }
    a->rough_has_active = false;
    return should_emit;
}

static void re_update_roughness_event(road_events_roughness_analyzer_t *a,
                                      road_events_roughness_estimate_t estimate, float dt,
                                      float speed_mps, float ds,
                                      road_events_roughness_update_t *update)
{
    bool above_enter = estimate.roughness_rms_mps2 >= a->cfg.rough_event_enter_mps2;
    bool above_exit = estimate.roughness_rms_mps2 >= a->cfg.rough_event_exit_mps2;
    if (a->rough_has_active) {
        if (above_exit) {
            a->rough_last_t_s = estimate.t_s;
            a->rough_duration_s += dt;
            a->rough_roughness_time_sum += estimate.roughness_rms_mps2 * dt;
            a->rough_peak_roughness_rms_mps2 =
                re_max(a->rough_peak_roughness_rms_mps2, estimate.roughness_rms_mps2);
            a->rough_speed_time_sum_m += speed_mps * dt;
            a->rough_distance_m += ds;
            if (!a->rough_emitted && a->rough_duration_s >= a->cfg.rough_event_min_duration_s) {
                a->rough_emitted = true;
                a->last_rough_event_t_s = a->rough_last_t_s;
                update->has_roughness_event = re_roughness_event_from_active(a, &update->roughness_event);
            }
            return;
        }
        update->has_completed_roughness_event =
            re_finish_roughness(a, true, &update->completed_roughness_event);
        return;
    }
    if (above_enter && estimate.t_s - a->last_rough_event_t_s >= a->cfg.rough_event_refractory_s) {
        a->rough_has_active = true;
        a->rough_start_t_s = estimate.t_s;
        a->rough_last_t_s = estimate.t_s;
        a->rough_duration_s = 0.0f;
        a->rough_roughness_time_sum = 0.0f;
        a->rough_peak_roughness_rms_mps2 = estimate.roughness_rms_mps2;
        a->rough_speed_time_sum_m = 0.0f;
        a->rough_distance_m = 0.0f;
        a->rough_emitted = false;
    }
}

bool road_events_roughness_update_with_events(road_events_roughness_analyzer_t *a,
                                              road_events_roughness_sample_t sample,
                                              road_events_roughness_update_t *update)
{
    float raw_dt;
    float dt;
    float bandpass;
    float speed_mps;
    float ds;
    bool updated;
    float abs_bandpass;
    float robust = 0.0f;
    float cap;
    float robust_abs;
    float energy_sample;

    if (update) {
        *update = (road_events_roughness_update_t){0};
    }
    if (!re_finite(sample.t_s) || !re_finite(sample.speed_mps) ||
        !re_finite(sample.vertical_accel_mps2)) {
        return false;
    }
    raw_dt = a->has_last_t_s ? re_max(sample.t_s - a->last_t_s, 0.0f) : 0.0f;
    a->has_last_t_s = true;
    a->last_t_s = sample.t_s;
    dt = re_min(raw_dt, re_max(a->cfg.max_dt_s, 0.0f));

    if (!a->filter_initialized) {
        a->filter_initialized = true;
        a->hp_last_input = sample.vertical_accel_mps2;
        a->hp_last_output = 0.0f;
        a->lp_output = 0.0f;
        if (update) {
            update->estimate = re_roughness_make_estimate(a, sample.t_s, 0.0f, 0.0f, false);
        }
        return true;
    }

    if (dt > 0.0f) {
        float hp = re_high_pass(sample.vertical_accel_mps2, a->hp_last_input, a->hp_last_output,
                                a->cfg.high_pass_cutoff_hz, dt);
        a->hp_last_input = sample.vertical_accel_mps2;
        a->hp_last_output = hp;
        a->lp_output = re_low_pass(a->lp_output, hp, a->cfg.low_pass_cutoff_hz, dt);
    }
    bandpass = a->lp_output;
    speed_mps = re_max(sample.speed_mps, 0.0f);
    ds = speed_mps * dt;
    updated = speed_mps >= a->cfg.min_speed_mps && ds > 0.0f;
    abs_bandpass = re_abs(bandpass);

    if (updated) {
        cap = re_roughness_cap(a);
        robust = re_clamp(bandpass, -cap, cap);
        if (update) {
            update->has_shock_event =
                re_update_shock(a, sample.t_s, dt, abs_bandpass, speed_mps, &update->shock_event);
        } else {
            (void)re_update_shock(a, sample.t_s, dt, abs_bandpass, speed_mps, 0);
        }
        a->distance_m += ds;
        robust_abs = re_abs(robust);
        if (a->baseline_initialized) {
            a->baseline_abs_mps2 =
                re_distance_ema(a->baseline_abs_mps2, robust_abs, a->cfg.robust_baseline_tau_m, ds);
        } else {
            a->baseline_initialized = true;
            a->baseline_abs_mps2 = robust_abs;
        }
        energy_sample = robust_abs * robust_abs;
        if (a->energy_initialized) {
            float tau_m = re_max(a->cfg.distance_tau_m, ds);
            float alpha = ds / (tau_m + ds);
            a->energy_mps2_sq = (1.0f - alpha) * a->energy_mps2_sq + alpha * energy_sample;
        } else {
            a->energy_initialized = true;
            a->energy_mps2_sq = energy_sample;
        }
    } else if (update) {
        update->has_shock_event = re_finish_shock(a, false, &update->shock_event);
    } else {
        (void)re_finish_shock(a, false, 0);
    }

    if (update) {
        update->estimate = re_roughness_make_estimate(a, sample.t_s, bandpass, robust, updated);
        if (updated) {
            re_update_roughness_event(a, update->estimate, dt, speed_mps, ds, update);
        } else {
            update->has_completed_roughness_event =
                re_finish_roughness(a, false, &update->completed_roughness_event);
        }
    } else {
        road_events_roughness_update_t scratch = {0};
        scratch.estimate = re_roughness_make_estimate(a, sample.t_s, bandpass, robust, updated);
        if (updated) {
            re_update_roughness_event(a, scratch.estimate, dt, speed_mps, ds, &scratch);
        } else {
            (void)re_finish_roughness(a, false, 0);
        }
    }
    return true;
}

bool road_events_roughness_update(road_events_roughness_analyzer_t *a,
                                  road_events_roughness_sample_t sample,
                                  road_events_roughness_estimate_t *estimate)
{
    road_events_roughness_update_t update;
    bool ok = road_events_roughness_update_with_events(a, sample, &update);
    if (ok && estimate) {
        *estimate = update.estimate;
    }
    return ok;
}

road_events_roughness_estimate_t road_events_roughness_estimate(
    const road_events_roughness_analyzer_t *a)
{
    return a->last_estimate;
}

bool road_events_roughness_finish(road_events_roughness_analyzer_t *a,
                                  road_events_roughness_event_t *event)
{
    return a->rough_has_active ? re_finish_roughness(a, true, event) : false;
}

void road_events_trip_stats_init(road_events_trip_stats_t *stats, road_events_trip_config_t cfg)
{
    *stats = (road_events_trip_stats_t){0};
    stats->cfg = cfg;
}

void road_events_trip_stats_reset(road_events_trip_stats_t *stats)
{
    road_events_trip_config_t cfg = stats->cfg;
    road_events_trip_stats_init(stats, cfg);
}

static bool re_valid_trip_sample(road_events_trip_sample_t sample)
{
    return re_finite(sample.t_s) && re_finite(sample.speed_mps) &&
           re_finite(sample.forward_velocity_mps) &&
           (!sample.height_valid || re_finite(sample.height_m)) &&
           re_finite(sample.longitudinal_accel_mps2) && re_finite(sample.lateral_accel_mps2);
}

static void re_trip_update_accel_extrema(road_events_trip_stats_t *s,
                                         road_events_trip_sample_t sample)
{
    s->peak_accel_mps2 = re_max(s->peak_accel_mps2, re_max(sample.longitudinal_accel_mps2, 0.0f));
    s->peak_decel_mps2 = re_max(s->peak_decel_mps2, re_max(-sample.longitudinal_accel_mps2, 0.0f));
    s->peak_lateral_accel_mps2 = re_max(s->peak_lateral_accel_mps2, re_abs(sample.lateral_accel_mps2));
}

void road_events_trip_stats_update_motion(road_events_trip_stats_t *s,
                                          road_events_trip_sample_t sample)
{
    road_events_trip_sample_t prev;
    float raw_dt;
    float max_dt;
    float dt;
    float speed_mps;
    float forward_velocity_mps;
    float reverse_speed_mps;

    if (!re_valid_trip_sample(sample)) {
        s->invalid_sample_count = re_sat_add_one(s->invalid_sample_count);
        return;
    }
    s->sample_count = re_sat_add_one(s->sample_count);
    re_trip_update_accel_extrema(s, sample);
    if (!s->has_last_sample) {
        s->has_last_sample = true;
        s->last_sample = sample;
        s->peak_speed_mps = re_max(s->peak_speed_mps, re_max(sample.speed_mps, 0.0f));
        return;
    }

    prev = s->last_sample;
    raw_dt = re_max(sample.t_s - prev.t_s, 0.0f);
    max_dt = re_max(s->cfg.max_integrated_dt_s, 0.0f);
    if (raw_dt > max_dt) {
        s->data_gap_count = re_sat_add_one(s->data_gap_count);
        s->total_gap_duration_s += raw_dt - max_dt;
    }
    s->max_sample_gap_s = re_max(s->max_sample_gap_s, raw_dt);
    s->last_sample = sample;
    dt = re_min(raw_dt, max_dt);

    speed_mps = 0.5f * (re_max(prev.speed_mps, 0.0f) + re_max(sample.speed_mps, 0.0f));
    forward_velocity_mps = 0.5f * (prev.forward_velocity_mps + sample.forward_velocity_mps);

    s->duration_s += dt;
    s->distance_m += speed_mps * dt;
    s->speed_time_sum_m += speed_mps * dt;
    s->peak_speed_mps = re_max(s->peak_speed_mps, speed_mps);
    if (speed_mps >= s->cfg.moving_speed_threshold_mps) {
        s->moving_duration_s += dt;
        s->moving_speed_time_sum_m += speed_mps * dt;
    }
    reverse_speed_mps = re_max(-forward_velocity_mps, 0.0f);
    if (reverse_speed_mps >= s->cfg.reverse_speed_threshold_mps) {
        s->reverse_duration_s += dt;
    }
    s->reverse_distance_m += reverse_speed_mps * dt;

    if (raw_dt <= max_dt && prev.height_frame_id == sample.height_frame_id &&
        prev.height_valid && sample.height_valid && re_finite(prev.height_m) &&
        re_finite(sample.height_m)) {
        float vertical_delta_m = sample.height_m - prev.height_m;
        s->elevation_valid = true;
        if (vertical_delta_m >= 0.0f) {
            s->elevation_gain_m += vertical_delta_m;
        } else {
            s->elevation_loss_m += -vertical_delta_m;
        }
    }

    if (dt > 0.0f) {
        if (!s->rolling_initialized) {
            s->rolling_initialized = true;
            s->rolling_speed_mps = speed_mps;
            s->rolling_abs_longitudinal_accel_mps2 = re_abs(sample.longitudinal_accel_mps2);
            s->rolling_abs_lateral_accel_mps2 = re_abs(sample.lateral_accel_mps2);
        } else {
            s->rolling_speed_mps = re_update_ema(s->rolling_speed_mps, speed_mps,
                                                 s->cfg.rolling_tau_s, dt);
            s->rolling_abs_longitudinal_accel_mps2 =
                re_update_abs_ema(s->rolling_abs_longitudinal_accel_mps2,
                                  sample.longitudinal_accel_mps2, s->cfg.rolling_tau_s, dt);
            s->rolling_abs_lateral_accel_mps2 =
                re_update_abs_ema(s->rolling_abs_lateral_accel_mps2, sample.lateral_accel_mps2,
                                  s->cfg.rolling_tau_s, dt);
        }
    }
}

void road_events_trip_stats_record_event(road_events_trip_stats_t *s,
                                         road_events_trip_event_kind_t kind)
{
    switch (kind) {
    case ROAD_EVENTS_TRIP_SPEED_BUMP:
        s->events.speed_bumps = re_sat_add_one(s->events.speed_bumps);
        break;
    case ROAD_EVENTS_TRIP_ROAD_SHOCK:
        s->events.road_shocks = re_sat_add_one(s->events.road_shocks);
        break;
    case ROAD_EVENTS_TRIP_ROUGH_ROAD:
        s->events.rough_road = re_sat_add_one(s->events.rough_road);
        break;
    case ROAD_EVENTS_TRIP_UPHILL:
        s->events.uphill = re_sat_add_one(s->events.uphill);
        break;
    case ROAD_EVENTS_TRIP_DOWNHILL:
        s->events.downhill = re_sat_add_one(s->events.downhill);
        break;
    case ROAD_EVENTS_TRIP_REVERSE:
        s->events.reverse = re_sat_add_one(s->events.reverse);
        break;
    case ROAD_EVENTS_TRIP_HARSH_ACCELERATION:
        s->events.harsh_acceleration = re_sat_add_one(s->events.harsh_acceleration);
        break;
    case ROAD_EVENTS_TRIP_HARSH_BRAKING:
        s->events.harsh_braking = re_sat_add_one(s->events.harsh_braking);
        break;
    case ROAD_EVENTS_TRIP_HARSH_CORNERING:
        s->events.harsh_cornering = re_sat_add_one(s->events.harsh_cornering);
        break;
    }
}

static float re_per_km(uint32_t count, float distance_km)
{
    return distance_km > 0.0f ? (float)count / distance_km : 0.0f;
}

road_events_trip_summary_t road_events_trip_stats_summary(const road_events_trip_stats_t *s)
{
    float distance_km = s->distance_m / 1000.0f;
    uint32_t harsh_count = s->events.harsh_acceleration + s->events.harsh_braking +
                           s->events.harsh_cornering;
    road_events_trip_summary_t out;
    out.sample_count = s->sample_count;
    out.invalid_sample_count = s->invalid_sample_count;
    out.data_gap_count = s->data_gap_count;
    out.max_sample_gap_s = s->max_sample_gap_s;
    out.total_gap_duration_s = s->total_gap_duration_s;
    out.duration_s = s->duration_s;
    out.moving_duration_s = s->moving_duration_s;
    out.stationary_duration_s = re_max(s->duration_s - s->moving_duration_s, 0.0f);
    out.distance_m = s->distance_m;
    out.reverse_duration_s = s->reverse_duration_s;
    out.reverse_distance_m = s->reverse_distance_m;
    out.elevation_gain_m = s->elevation_gain_m;
    out.elevation_loss_m = s->elevation_loss_m;
    out.elevation_valid = s->elevation_valid;
    out.mean_speed_mps = re_ratio_or_zero(s->speed_time_sum_m, s->duration_s);
    out.moving_mean_speed_mps = re_ratio_or_zero(s->moving_speed_time_sum_m, s->moving_duration_s);
    out.peak_speed_mps = s->peak_speed_mps;
    out.peak_accel_mps2 = s->peak_accel_mps2;
    out.peak_decel_mps2 = s->peak_decel_mps2;
    out.peak_lateral_accel_mps2 = s->peak_lateral_accel_mps2;
    out.rolling_speed_mps = s->rolling_speed_mps;
    out.rolling_abs_longitudinal_accel_mps2 = s->rolling_abs_longitudinal_accel_mps2;
    out.rolling_abs_lateral_accel_mps2 = s->rolling_abs_lateral_accel_mps2;
    out.events = s->events;
    out.speed_bumps_per_km = re_per_km(s->events.speed_bumps, distance_km);
    out.road_shocks_per_km = re_per_km(s->events.road_shocks, distance_km);
    out.rough_road_events_per_km = re_per_km(s->events.rough_road, distance_km);
    out.harsh_events_per_km = re_per_km(harsh_count, distance_km);
    out.reverse_seconds_per_km = re_ratio_or_zero(s->reverse_duration_s, distance_km);
    return out;
}
