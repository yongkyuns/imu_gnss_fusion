#include "road_events.h"
#include "unity.h"

#include <math.h>
#include <stdint.h>

#define RE_TAU 6.28318530717958647692f

static float gaussian(float t, float center, float sigma)
{
    float z = (t - center) / sigma;
    return expf(-0.5f * z * z);
}

static void assert_close(float actual, float expected, float tol)
{
    TEST_ASSERT_FLOAT_WITHIN(tol, expected, actual);
}

static unsigned corner_events_for_signed_profile(road_events_harsh_corner_config_t cfg,
                                                 float peak_lateral_accel_mps2,
                                                 float speed_mps,
                                                 float sign)
{
    road_events_harsh_corner_detector_t detector;
    road_events_harsh_corner_event_t event;
    unsigned events = 0u;
    int i;
    road_events_harsh_corner_init(&detector, cfg);
    for (i = 0; i < 220; i++) {
        float t = (float)i * 0.02f;
        float lateral = 0.0f;
        if (t >= 1.0f && t < 1.22f) {
            lateral = peak_lateral_accel_mps2 * (t - 1.0f) / 0.22f;
        } else if (t <= 2.2f && t >= 1.22f) {
            lateral = peak_lateral_accel_mps2;
        }
        if (road_events_harsh_corner_update(
                &detector,
                (road_events_harsh_corner_sample_t){t, speed_mps, sign * lateral},
                &event)) {
            events++;
        }
    }
    if (road_events_harsh_corner_finish(&detector, &event)) {
        events++;
    }
    return events;
}

static unsigned corner_events_for_profile(road_events_harsh_corner_config_t cfg,
                                          float peak_lateral_accel_mps2,
                                          float speed_mps)
{
    return corner_events_for_signed_profile(cfg, peak_lateral_accel_mps2, speed_mps, 1.0f);
}

static float roughness_for_distance(road_events_roughness_config_t cfg,
                                    float speed_mps,
                                    float distance_m,
                                    float accel_amplitude_mps2)
{
    road_events_roughness_analyzer_t analyzer;
    road_events_roughness_estimate_t estimate;
    float dt = 0.01f;
    int samples = (int)(distance_m / (speed_mps * dt));
    int i;
    road_events_roughness_init(&analyzer, cfg);
    estimate = road_events_roughness_estimate(&analyzer);
    for (i = 0; i < samples; i++) {
        float t = (float)i * dt;
        float x = speed_mps * t;
        float phase = RE_TAU * x / 8.0f;
        TEST_ASSERT_TRUE(road_events_roughness_update(
            &analyzer,
            (road_events_roughness_sample_t){t, speed_mps,
                                             accel_amplitude_mps2 * sinf(phase)},
            &estimate));
    }
    return estimate.roughness_rms_mps2;
}

static road_events_roughness_estimate_t roughness_estimate_for_spatial_sine(
    road_events_roughness_config_t cfg,
    float accel_amplitude_mps2)
{
    road_events_roughness_analyzer_t analyzer;
    road_events_roughness_estimate_t estimate;
    float speed_mps = 10.0f;
    float dt = 0.02f;
    int i;
    road_events_roughness_init(&analyzer, cfg);
    estimate = road_events_roughness_estimate(&analyzer);
    for (i = 0; i < 3000; i++) {
        float t = (float)i * dt;
        float x = speed_mps * t;
        float phase = RE_TAU * x / 8.0f;
        TEST_ASSERT_TRUE(road_events_roughness_update(
            &analyzer,
            (road_events_roughness_sample_t){t, speed_mps,
                                             accel_amplitude_mps2 * sinf(phase)},
            &estimate));
    }
    return estimate;
}

void test_detects_accel_double_peak_with_pitch_confirmation(void)
{
    road_events_speed_bump_config_t cfg = road_events_speed_bump_config_default();
    road_events_speed_bump_detector_t detector;
    road_events_speed_bump_diagnostic_t diagnostic;
    road_events_speed_bump_event_t event;
    unsigned events = 0u;
    int i;
    cfg.trigger_confidence = 0.12f;
    road_events_speed_bump_init(&detector, cfg);
    for (i = 0; i < 600; i++) {
        float t = (float)i * 0.01f;
        float accel = gaussian(t, 2.00f, 0.12f) -
                      1.2f * gaussian(t, 2.30f, 0.12f) +
                      gaussian(t, 2.60f, 0.12f);
        float pitch = 0.90f * gaussian(t, 2.10f, 0.14f) -
                      1.10f * gaussian(t, 2.42f, 0.14f);
        if (road_events_speed_bump_update(
                &detector, (road_events_speed_bump_sample_t){t, 4.0f, pitch, 4.0f * accel},
                &diagnostic, &event)) {
            events++;
        }
    }
    TEST_ASSERT_EQUAL_UINT(1u, events);
}

void test_speed_bump_ignores_slow_pitch_drift_and_uncorroborated_impulse(void)
{
    road_events_speed_bump_detector_t detector;
    road_events_speed_bump_diagnostic_t diagnostic;
    road_events_speed_bump_event_t event;
    unsigned events = 0u;
    int i;
    road_events_speed_bump_init(&detector, road_events_speed_bump_config_default());
    for (i = 0; i < 700; i++) {
        float t = (float)i * 0.01f;
        if (road_events_speed_bump_update(
                &detector,
                (road_events_speed_bump_sample_t){t, 5.0f, 0.8f * sinf(0.2f * t),
                                                  0.2f * sinf(0.3f * t)},
                &diagnostic, &event)) {
            events++;
        }
    }
    TEST_ASSERT_EQUAL_UINT(0u, events);

    road_events_speed_bump_config_t cfg = road_events_speed_bump_config_default();
    cfg.trigger_confidence = 0.25f;
    road_events_speed_bump_init(&detector, cfg);
    for (i = 0; i < 600; i++) {
        float t = (float)i * 0.01f;
        float accel = gaussian(t, 2.00f, 0.12f) -
                      1.2f * gaussian(t, 2.30f, 0.12f) +
                      gaussian(t, 2.60f, 0.12f);
        if (road_events_speed_bump_update(
                &detector,
                (road_events_speed_bump_sample_t){t, 4.0f, 0.02f * sinf(13.0f * t),
                                                  4.0f * accel},
                &diagnostic, &event)) {
            events++;
        }
    }
    TEST_ASSERT_EQUAL_UINT(0u, events);
}

void test_hill_detection_confirmation_and_short_excursion(void)
{
    road_events_hill_detector_t detector;
    road_events_hill_event_t events[4];
    unsigned count = 0u;
    int i;
    road_events_hill_init(&detector, road_events_hill_config_default());
    for (i = 0; i < 160; i++) {
        float t = (float)i * 0.1f;
        float pitch = 1.0f;
        if (t >= 2.0f && t <= 6.0f) {
            pitch = 4.5f;
        } else if (t >= 9.0f && t <= 12.4f) {
            pitch = -4.8f;
        }
        if (road_events_hill_update(&detector, (road_events_hill_sample_t){t, 5.0f, pitch},
                                    &events[count])) {
            count++;
        }
    }
    if (road_events_hill_finish(&detector, &events[count])) {
        count++;
    }
    TEST_ASSERT_EQUAL_UINT(2u, count);
    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_HILL_UPHILL, events[0].kind);
    TEST_ASSERT_TRUE(events[0].duration_s >= 1.0f);
    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_HILL_DOWNHILL, events[1].kind);
    TEST_ASSERT_TRUE(events[1].duration_s >= 1.0f);

    count = 0u;
    road_events_hill_init(&detector, road_events_hill_config_default());
    for (i = 0; i < 80; i++) {
        float t = (float)i * 0.1f;
        float pitch = (t >= 2.0f && t < 2.8f) ? 5.0f : 0.0f;
        if (road_events_hill_update(&detector, (road_events_hill_sample_t){t, 4.0f, pitch},
                                    &events[0])) {
            count++;
        }
    }
    if (road_events_hill_finish(&detector, &events[0])) {
        count++;
    }
    TEST_ASSERT_EQUAL_UINT(0u, count);
}

void test_hill_emits_once_when_confirmed(void)
{
    road_events_hill_detector_t detector;
    road_events_hill_event_t event;
    unsigned count = 0u;
    int i;
    road_events_hill_init(&detector, road_events_hill_config_default());
    for (i = 0; i < 180; i++) {
        float t = (float)i * 0.1f;
        float pitch = (t >= 2.0f && t <= 14.0f) ? 4.5f : 0.0f;
        if (road_events_hill_update(&detector, (road_events_hill_sample_t){t, 5.0f, pitch},
                                    &event)) {
            count++;
        }
    }
    if (road_events_hill_finish(&detector, &event)) {
        count++;
    }
    TEST_ASSERT_EQUAL_UINT(1u, count);
    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_HILL_UPHILL, event.kind);
    assert_close(event.start_t_s, 2.0f, 1.0e-5f);
    TEST_ASSERT_TRUE(event.end_t_s <= 3.1f);
    TEST_ASSERT_TRUE(event.duration_s >= 1.0f);
}

void test_reverse_interval_hysteresis_and_short_blip(void)
{
    road_events_reverse_detector_t detector;
    road_events_reverse_event_t event;
    unsigned count = 0u;
    int i;
    road_events_reverse_init(&detector, road_events_reverse_config_default());
    for (i = 0; i < 80; i++) {
        float t = (float)i * 0.1f;
        float v = 0.0f;
        if (t >= 1.0f && t <= 3.0f) {
            v = -0.8f;
        } else if (t >= 3.1f && t <= 3.3f) {
            v = -0.15f;
        } else if (t >= 3.4f && t <= 4.0f) {
            v = -0.3f;
        }
        if (road_events_reverse_update(&detector, (road_events_reverse_sample_t){t, v}, &event)) {
            count++;
        }
    }
    if (road_events_reverse_finish(&detector, &event)) {
        count++;
    }
    TEST_ASSERT_EQUAL_UINT(1u, count);
    TEST_ASSERT_TRUE(event.duration_s >= 1.0f);
    TEST_ASSERT_TRUE(event.mean_reverse_speed_mps > 0.4f);
    TEST_ASSERT_TRUE(event.peak_reverse_speed_mps >= 0.8f);

    count = 0u;
    road_events_reverse_init(&detector, road_events_reverse_config_default());
    for (i = 0; i < 30; i++) {
        float t = (float)i * 0.1f;
        float v = (t >= 1.0f && t < 1.3f) ? -0.9f : 0.0f;
        if (road_events_reverse_update(&detector, (road_events_reverse_sample_t){t, v}, &event)) {
            count++;
        }
    }
    if (road_events_reverse_finish(&detector, &event)) {
        count++;
    }
    TEST_ASSERT_EQUAL_UINT(0u, count);
}

void test_harsh_accel_and_brake_from_velocity_derivative_ema(void)
{
    road_events_harsh_accel_detector_t accel_detector;
    road_events_harsh_brake_detector_t brake_detector;
    road_events_harsh_longitudinal_event_t event;
    unsigned count = 0u;
    int i;
    road_events_harsh_accel_init(&accel_detector, road_events_harsh_accel_config_default());
    for (i = 0; i < 250; i++) {
        float t = (float)i * 0.02f;
        float v = t < 1.0f ? 2.0f : (t <= 3.0f ? 2.0f + 3.0f * (t - 1.0f) : 8.0f);
        if (road_events_harsh_accel_update(&accel_detector,
                                           (road_events_harsh_longitudinal_sample_t){t, v},
                                           &event)) {
            count++;
        }
    }
    if (road_events_harsh_accel_finish(&accel_detector, &event)) {
        count++;
    }
    TEST_ASSERT_EQUAL_UINT(1u, count);
    TEST_ASSERT_TRUE(event.peak_accel_mps2 >= 2.5f);
    TEST_ASSERT_TRUE(event.delta_velocity_mps > 0.0f);

    count = 0u;
    road_events_harsh_brake_init(&brake_detector, road_events_harsh_brake_config_default());
    for (i = 0; i < 250; i++) {
        float t = (float)i * 0.02f;
        float v = t < 1.0f ? 10.0f : (t <= 2.5f ? 10.0f - 4.0f * (t - 1.0f) : 4.0f);
        if (road_events_harsh_brake_update(&brake_detector,
                                           (road_events_harsh_longitudinal_sample_t){t, v},
                                           &event)) {
            count++;
        }
    }
    if (road_events_harsh_brake_finish(&brake_detector, &event)) {
        count++;
    }
    TEST_ASSERT_EQUAL_UINT(1u, count);
    TEST_ASSERT_TRUE(event.peak_accel_mps2 >= 3.0f);
    TEST_ASSERT_TRUE(event.delta_velocity_mps < 0.0f);
}

void test_harsh_cornering_profiles_presets_and_invalid_input(void)
{
    road_events_harsh_behavior_config_t sensitive =
        road_events_harsh_behavior_preset_config(ROAD_EVENTS_HARSH_SENSITIVE);
    road_events_harsh_behavior_config_t balanced =
        road_events_harsh_behavior_preset_config(ROAD_EVENTS_HARSH_BALANCED);
    road_events_harsh_behavior_config_t conservative =
        road_events_harsh_behavior_preset_config(ROAD_EVENTS_HARSH_CONSERVATIVE);
    road_events_harsh_corner_detector_t detector;
    road_events_harsh_corner_event_t event;
    unsigned events = 0u;
    int i;

    TEST_ASSERT_EQUAL_UINT(1u, corner_events_for_profile(road_events_harsh_corner_config_default(),
                                                        3.8f, 12.0f));
    TEST_ASSERT_EQUAL_UINT(1u, corner_events_for_profile(sensitive.corner, 2.7f, 12.0f));
    TEST_ASSERT_EQUAL_UINT(0u, corner_events_for_profile(balanced.corner, 2.7f, 12.0f));
    TEST_ASSERT_EQUAL_UINT(0u, corner_events_for_profile(conservative.corner, 3.5f, 12.0f));
    TEST_ASSERT_EQUAL_UINT(0u, corner_events_for_profile(road_events_harsh_corner_config_default(),
                                                        4.2f, 2.0f));
    TEST_ASSERT_EQUAL_UINT(1u, corner_events_for_signed_profile(
                                   road_events_harsh_corner_config_default(), 4.2f, 12.0f, 1.0f));
    TEST_ASSERT_EQUAL_UINT(1u, corner_events_for_signed_profile(
                                   road_events_harsh_corner_config_default(), 4.2f, 12.0f, -1.0f));

    road_events_harsh_corner_init(&detector, road_events_harsh_corner_config_default());
    for (i = 0; i < 360; i++) {
        float t = (float)i * 0.02f;
        float lateral = t < 1.0f ? 0.0f : (t < 3.5f ? 3.8f * (t - 1.0f) / 2.5f : 3.8f);
        if (road_events_harsh_corner_update(
                &detector, (road_events_harsh_corner_sample_t){t, 12.0f, lateral}, &event)) {
            events++;
        }
    }
    if (road_events_harsh_corner_finish(&detector, &event)) {
        events++;
    }
    TEST_ASSERT_EQUAL_UINT(0u, events);

    road_events_harsh_corner_init(&detector, road_events_harsh_corner_config_default());
    for (i = 0; i < 180; i++) {
        float t = (float)i * 0.02f;
        TEST_ASSERT_FALSE(road_events_harsh_corner_update(
            &detector, (road_events_harsh_corner_sample_t){t, 10.0f, NAN}, &event));
    }
    TEST_ASSERT_FALSE(road_events_harsh_corner_finish(&detector, &event));
}

void test_harsh_behavior_presets_adjust_only_thresholds(void)
{
    road_events_harsh_behavior_config_t sensitive =
        road_events_harsh_behavior_preset_config(ROAD_EVENTS_HARSH_SENSITIVE);
    road_events_harsh_behavior_config_t balanced =
        road_events_harsh_behavior_preset_config(ROAD_EVENTS_HARSH_BALANCED);
    road_events_harsh_behavior_config_t conservative =
        road_events_harsh_behavior_preset_config(ROAD_EVENTS_HARSH_CONSERVATIVE);
    assert_close(sensitive.accel.accel_tau_s, balanced.accel.accel_tau_s, 1.0e-6f);
    assert_close(conservative.accel.accel_tau_s, balanced.accel.accel_tau_s, 1.0e-6f);
    assert_close(sensitive.corner.lateral_accel_tau_s, balanced.corner.lateral_accel_tau_s,
                 1.0e-6f);
    assert_close(conservative.corner.lateral_jerk_tau_s, balanced.corner.lateral_jerk_tau_s,
                 1.0e-6f);
    assert_close(balanced.corner.lateral_accel_threshold_mps2, 3.4f, 1.0e-6f);
    assert_close(balanced.corner.exit_lateral_accel_threshold_mps2, 2.9f, 1.0e-6f);
    assert_close(balanced.corner.lateral_jerk_threshold_mps3, 5.0f, 1.0e-6f);
    TEST_ASSERT_TRUE(sensitive.accel.accel_threshold_mps2 < balanced.accel.accel_threshold_mps2);
    TEST_ASSERT_TRUE(balanced.accel.accel_threshold_mps2 < conservative.accel.accel_threshold_mps2);
    TEST_ASSERT_TRUE(sensitive.brake.decel_threshold_mps2 < balanced.brake.decel_threshold_mps2);
    TEST_ASSERT_TRUE(balanced.brake.decel_threshold_mps2 < conservative.brake.decel_threshold_mps2);
    TEST_ASSERT_TRUE(sensitive.corner.lateral_accel_threshold_mps2 <
                     balanced.corner.lateral_accel_threshold_mps2);
    TEST_ASSERT_TRUE(balanced.corner.lateral_accel_threshold_mps2 <
                     conservative.corner.lateral_accel_threshold_mps2);
}

void test_trip_stats_distance_events_gaps_and_elevation(void)
{
    road_events_trip_stats_t stats;
    road_events_trip_config_t cfg = {0.5f, 0.2f, 2.0f, 1.0f};
    road_events_trip_summary_t summary;
    int i;
    road_events_trip_stats_init(&stats, cfg);
    for (i = 0; i <= 10; i++) {
        road_events_trip_stats_update_motion(
            &stats, (road_events_trip_sample_t){(float)i, i < 3 ? 0.0f : 10.0f,
                                                i < 3 ? 0.0f : 10.0f, false, 0.0f, 0u,
                                                i == 4 ? 2.5f : 0.0f,
                                                i == 6 ? -3.2f : 0.0f});
    }
    summary = road_events_trip_stats_summary(&stats);
    TEST_ASSERT_EQUAL_UINT32(11u, summary.sample_count);
    assert_close(summary.duration_s, 10.0f, 1.0e-5f);
    assert_close(summary.moving_duration_s, 8.0f, 1.0e-5f);
    assert_close(summary.stationary_duration_s, 2.0f, 1.0e-5f);
    assert_close(summary.distance_m, 75.0f, 1.0e-5f);
    assert_close(summary.mean_speed_mps, 7.5f, 1.0e-5f);
    assert_close(summary.moving_mean_speed_mps, 9.375f, 1.0e-5f);
    assert_close(summary.peak_accel_mps2, 2.5f, 1.0e-5f);
    assert_close(summary.peak_lateral_accel_mps2, 3.2f, 1.0e-5f);

    road_events_trip_stats_init(&stats, road_events_trip_config_default());
    for (i = 0; i <= 10; i++) {
        float v = (i >= 2 && i <= 5) ? -2.0f : 4.0f;
        road_events_trip_stats_update_motion(
            &stats, (road_events_trip_sample_t){(float)i, fabsf(v), v, false, 0.0f, 0u,
                                                i == 7 ? -3.5f : 0.0f, 0.0f});
    }
    road_events_trip_stats_record_event(&stats, ROAD_EVENTS_TRIP_SPEED_BUMP);
    road_events_trip_stats_record_event(&stats, ROAD_EVENTS_TRIP_HARSH_ACCELERATION);
    road_events_trip_stats_record_event(&stats, ROAD_EVENTS_TRIP_HARSH_BRAKING);
    road_events_trip_stats_record_event(&stats, ROAD_EVENTS_TRIP_REVERSE);
    summary = road_events_trip_stats_summary(&stats);
    assert_close(summary.reverse_duration_s, 3.0f, 1.0e-5f);
    assert_close(summary.reverse_distance_m, 6.0f, 1.0e-5f);
    TEST_ASSERT_FALSE(summary.elevation_valid);
    TEST_ASSERT_EQUAL_UINT32(1u, summary.events.speed_bumps);
    TEST_ASSERT_EQUAL_UINT32(1u, summary.events.harsh_acceleration);
    TEST_ASSERT_EQUAL_UINT32(1u, summary.events.harsh_braking);
    TEST_ASSERT_EQUAL_UINT32(1u, summary.events.reverse);
    TEST_ASSERT_TRUE(summary.harsh_events_per_km > summary.speed_bumps_per_km);
    assert_close(summary.peak_decel_mps2, 3.5f, 1.0e-5f);

    cfg = road_events_trip_config_default();
    cfg.max_integrated_dt_s = 1.0f;
    road_events_trip_stats_init(&stats, cfg);
    road_events_trip_stats_update_motion(
        &stats, (road_events_trip_sample_t){0.0f, 5.0f, 5.0f, false, 0.0f, 0u, 0.0f, 0.0f});
    road_events_trip_stats_update_motion(
        &stats, (road_events_trip_sample_t){5.0f, 5.0f, 5.0f, false, 0.0f, 0u, 0.0f, 0.0f});
    road_events_trip_stats_update_motion(
        &stats, (road_events_trip_sample_t){NAN, 5.0f, 5.0f, false, 0.0f, 0u, 0.0f, 0.0f});
    summary = road_events_trip_stats_summary(&stats);
    TEST_ASSERT_EQUAL_UINT32(2u, summary.sample_count);
    TEST_ASSERT_EQUAL_UINT32(1u, summary.invalid_sample_count);
    TEST_ASSERT_EQUAL_UINT32(1u, summary.data_gap_count);
    assert_close(summary.total_gap_duration_s, 4.0f, 1.0e-5f);
    assert_close(summary.distance_m, 5.0f, 1.0e-5f);

    road_events_trip_stats_init(&stats, road_events_trip_config_default());
    road_events_trip_stats_update_motion(
        &stats, (road_events_trip_sample_t){0.0f, 10.0f, 10.0f, true, 100.0f, 0u, 0.0f, 0.0f});
    road_events_trip_stats_update_motion(
        &stats, (road_events_trip_sample_t){1.0f, 10.0f, 10.0f, true, 110.0f, 0u, 0.0f, 0.0f});
    road_events_trip_stats_update_motion(
        &stats, (road_events_trip_sample_t){2.0f, 10.0f, 10.0f, true, 50.0f, 1u, 0.0f, 0.0f});
    summary = road_events_trip_stats_summary(&stats);
    TEST_ASSERT_TRUE(summary.elevation_valid);
    assert_close(summary.elevation_gain_m, 10.0f, 1.0e-5f);
    assert_close(summary.elevation_loss_m, 0.0f, 1.0e-5f);
}

void test_roughness_distance_min_speed_clipping_and_levels(void)
{
    road_events_roughness_config_t cfg = road_events_roughness_config_default();
    road_events_roughness_analyzer_t analyzer;
    road_events_roughness_estimate_t estimate;
    float slow;
    float fast;
    float peak = 0.0f;
    int i;

    cfg.high_pass_cutoff_hz = 0.05f;
    cfg.low_pass_cutoff_hz = 20.0f;
    cfg.distance_tau_m = 70.0f;
    cfg.min_speed_mps = 0.5f;
    cfg.clip_mps2 = 3.0f;
    slow = roughness_for_distance(cfg, 5.0f, 160.0f, 1.0f);
    fast = roughness_for_distance(cfg, 20.0f, 160.0f, 1.0f);
    TEST_ASSERT_TRUE(fabsf(slow - fast) < 0.08f);

    cfg = road_events_roughness_config_default();
    cfg.min_speed_mps = 2.0f;
    road_events_roughness_init(&analyzer, cfg);
    for (i = 0; i < 200; i++) {
        float t = (float)i * 0.02f;
        TEST_ASSERT_TRUE(road_events_roughness_update(
            &analyzer,
            (road_events_roughness_sample_t){t, 0.5f, 2.0f * sinf(12.0f * t)},
            &estimate));
    }
    TEST_ASSERT_FALSE(estimate.updated);
    assert_close(estimate.distance_m, 0.0f, 1.0e-5f);
    assert_close(estimate.roughness_rms_mps2, 0.0f, 1.0e-5f);

    cfg.high_pass_cutoff_hz = 0.05f;
    cfg.low_pass_cutoff_hz = 20.0f;
    cfg.distance_tau_m = 70.0f;
    cfg.min_speed_mps = 0.5f;
    cfg.clip_mps2 = 2.0f;
    road_events_roughness_init(&analyzer, cfg);
    for (i = 0; i < 600; i++) {
        float t = (float)i * 0.02f;
        float accel = (t >= 2.0f && t < 2.06f) ? 20.0f : 0.15f;
        TEST_ASSERT_TRUE(road_events_roughness_update(
            &analyzer, (road_events_roughness_sample_t){t, 10.0f, accel}, &estimate));
        if (estimate.roughness_rms_mps2 > peak) {
            peak = estimate.roughness_rms_mps2;
        }
    }
    TEST_ASSERT_TRUE(peak <= cfg.clip_mps2 + 1.0e-4f);

    cfg = road_events_roughness_config_default();
    cfg.high_pass_cutoff_hz = 0.05f;
    cfg.low_pass_cutoff_hz = 20.0f;
    cfg.distance_tau_m = 30.0f;
    cfg.min_speed_mps = 0.5f;
    cfg.clip_mps2 = 3.0f;
    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_ROUGHNESS_VERY_SMOOTH,
                          roughness_estimate_for_spatial_sine(cfg, 0.12f).level);
    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_ROUGHNESS_SMOOTH,
                          roughness_estimate_for_spatial_sine(cfg, 0.25f).level);
    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_ROUGHNESS_LIGHT_TEXTURE,
                          roughness_estimate_for_spatial_sine(cfg, 0.50f).level);
    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_ROUGHNESS_MODERATE,
                          roughness_estimate_for_spatial_sine(cfg, 0.80f).level);
    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_ROUGHNESS_ROUGH,
                          roughness_estimate_for_spatial_sine(cfg, 1.10f).level);
    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_ROUGHNESS_VERY_ROUGH,
                          roughness_estimate_for_spatial_sine(cfg, 1.55f).level);
    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_ROUGHNESS_SEVERE,
                          roughness_estimate_for_spatial_sine(cfg, 2.20f).level);
}

void test_roughness_shock_and_rough_road_event_separation(void)
{
    road_events_roughness_config_t cfg = road_events_roughness_config_default();
    road_events_roughness_analyzer_t analyzer;
    road_events_roughness_update_t update;
    unsigned shock_events = 0u;
    unsigned rough_events = 0u;
    unsigned completed_events = 0u;
    road_events_roughness_event_t live_event = {0};
    road_events_roughness_event_t completed_event = {0};
    float peak_rms = 0.0f;
    float speed_mps = 10.0f;
    float dt = 0.02f;
    int i;

    cfg.high_pass_cutoff_hz = 0.05f;
    cfg.low_pass_cutoff_hz = 20.0f;
    cfg.distance_tau_m = 30.0f;
    cfg.min_speed_mps = 0.5f;
    cfg.shock_min_peak_mps2 = 2.0f;
    road_events_roughness_init(&analyzer, cfg);
    for (i = 0; i < 5000; i++) {
        float t = (float)i * dt;
        bool shock = fabsf(t - 8.0f) < 0.03f || fabsf(t - 11.0f) < 0.03f ||
                     fabsf(t - 12.5f) < 0.03f || fabsf(t - 14.0f) < 0.03f ||
                     fabsf(t - 25.0f) < 0.03f || fabsf(t - 31.0f) < 0.03f;
        float accel = 0.25f * sinf(8.0f * t) + (shock ? 10.0f : 0.0f);
        TEST_ASSERT_TRUE(road_events_roughness_update_with_events(
            &analyzer, (road_events_roughness_sample_t){t, speed_mps, accel}, &update));
        peak_rms = fmaxf(peak_rms, update.estimate.roughness_rms_mps2);
        shock_events += update.has_shock_event ? 1u : 0u;
        rough_events += update.has_roughness_event ? 1u : 0u;
    }
    TEST_ASSERT_TRUE(shock_events >= 3u);
    TEST_ASSERT_EQUAL_UINT(0u, rough_events);
    TEST_ASSERT_TRUE(peak_rms < cfg.rough_event_enter_mps2);

    cfg = road_events_roughness_config_default();
    cfg.high_pass_cutoff_hz = 0.05f;
    cfg.low_pass_cutoff_hz = 20.0f;
    cfg.min_speed_mps = 0.5f;
    cfg.shock_min_peak_mps2 = 2.5f;
    road_events_roughness_init(&analyzer, cfg);
    shock_events = rough_events = completed_events = 0u;
    for (i = 0; i < 5000; i++) {
        float t = (float)i * dt;
        float x = speed_mps * t;
        float phase = RE_TAU * x / 5.0f;
        float accel = (x > 20.0f && x < 90.0f) ? 1.15f * sinf(phase) : 0.08f * sinf(phase);
        TEST_ASSERT_TRUE(road_events_roughness_update_with_events(
            &analyzer, (road_events_roughness_sample_t){t, speed_mps, accel}, &update));
        if (update.has_roughness_event) {
            rough_events++;
            live_event = update.roughness_event;
        }
        if (update.has_completed_roughness_event) {
            completed_events++;
            completed_event = update.completed_roughness_event;
        }
        shock_events += update.has_shock_event ? 1u : 0u;
    }
    TEST_ASSERT_TRUE(rough_events >= 1u);
    TEST_ASSERT_EQUAL_UINT(0u, shock_events);
    TEST_ASSERT_EQUAL_UINT(1u, rough_events);
    TEST_ASSERT_EQUAL_UINT(1u, completed_events);
    assert_close(live_event.start_t_s, completed_event.start_t_s, 1.0e-5f);
    TEST_ASSERT_TRUE(completed_event.end_t_s > live_event.end_t_s);
    TEST_ASSERT_TRUE(completed_event.duration_s > live_event.duration_s);
}

void test_roughness_dissipates_after_clean_road_distance(void)
{
    road_events_roughness_config_t cfg = road_events_roughness_config_default();
    road_events_roughness_analyzer_t analyzer;
    road_events_roughness_estimate_t estimate;
    float rough_end = 0.0f;
    float clean_end = 0.0f;
    float speed_mps = 10.0f;
    float dt = 0.02f;
    int i;
    cfg.high_pass_cutoff_hz = 0.05f;
    cfg.low_pass_cutoff_hz = 20.0f;
    cfg.min_speed_mps = 0.5f;
    cfg.clip_mps2 = 3.0f;
    assert_close(cfg.distance_tau_m, 10.0f, 1.0e-5f);
    road_events_roughness_init(&analyzer, cfg);
    for (i = 0; i < 1500; i++) {
        float t = (float)i * dt;
        float x = speed_mps * t;
        bool rough_segment = x < 20.0f;
        float phase = RE_TAU * x / 5.0f;
        float accel = rough_segment ? 1.2f * sinf(phase) : 0.02f * sinf(phase);
        TEST_ASSERT_TRUE(road_events_roughness_update(
            &analyzer, (road_events_roughness_sample_t){t, speed_mps, accel}, &estimate));
        if (rough_segment) {
            rough_end = estimate.roughness_rms_mps2;
        }
        clean_end = estimate.roughness_rms_mps2;
    }
    TEST_ASSERT_TRUE(rough_end > cfg.moderate_threshold_mps2);
    TEST_ASSERT_TRUE(clean_end < cfg.smooth_threshold_mps2);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_detects_accel_double_peak_with_pitch_confirmation);
    RUN_TEST(test_speed_bump_ignores_slow_pitch_drift_and_uncorroborated_impulse);
    RUN_TEST(test_hill_detection_confirmation_and_short_excursion);
    RUN_TEST(test_hill_emits_once_when_confirmed);
    RUN_TEST(test_reverse_interval_hysteresis_and_short_blip);
    RUN_TEST(test_harsh_accel_and_brake_from_velocity_derivative_ema);
    RUN_TEST(test_harsh_cornering_profiles_presets_and_invalid_input);
    RUN_TEST(test_harsh_behavior_presets_adjust_only_thresholds);
    RUN_TEST(test_trip_stats_distance_events_gaps_and_elevation);
    RUN_TEST(test_roughness_distance_min_speed_clipping_and_levels);
    RUN_TEST(test_roughness_shock_and_rough_road_event_separation);
    RUN_TEST(test_roughness_dissipates_after_clean_road_distance);
    return UNITY_END();
}
