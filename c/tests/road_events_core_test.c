#include "road_events.h"
#include "unity.h"

#include <math.h>

static float parity_speed_at(float t)
{
    if (t < 6.0f) {
        return 6.0f;
    }
    if (t <= 7.2f) {
        return 6.0f + (t - 6.0f) * 4.5f;
    }
    if (t < 9.0f) {
        return 11.4f;
    }
    if (t <= 10.2f) {
        return 11.4f - (t - 9.0f) * 5.2f;
    }
    return 4.8f;
}

static float parity_pitch_at(float t)
{
    return (t >= 1.0f && t <= 2.5f) ? 4.8f : 0.0f;
}

static float parity_forward_velocity_at(float t)
{
    return (t >= 3.0f && t <= 5.2f) ? -0.85f : parity_speed_at(t);
}

static float parity_lateral_at(float t)
{
    if (t < 12.0f) {
        return 0.0f;
    }
    if (t <= 12.3f) {
        return (t - 12.0f) / 0.3f * 4.2f;
    }
    if (t <= 13.1f) {
        return 4.2f;
    }
    if (t <= 13.5f) {
        return 4.2f * (1.0f - (t - 13.1f) / 0.4f);
    }
    return 0.0f;
}

static float parity_gaussian(float t, float center, float sigma)
{
    float z = (t - center) / sigma;
    return expf(-0.5f * z * z);
}

static void test_hill_emits_once_when_confirmed(void)
{
    road_events_hill_detector_t detector;
    road_events_hill_event_t event;
    int events = 0;
    road_events_hill_init(&detector, road_events_hill_config_default());
    for (int i = 0; i < 180; ++i) {
        float t = (float)i * 0.1f;
        float pitch = (t >= 2.0f && t <= 14.0f) ? 4.5f : 0.0f;
        if (road_events_hill_update(&detector,
                                    (road_events_hill_sample_t){t, 5.0f, pitch},
                                    &event)) {
            events++;
        }
    }
    if (road_events_hill_finish(&detector, &event)) {
        events++;
    }
    TEST_ASSERT_EQUAL_INT(1, events);
    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_HILL_UPHILL, event.kind);
    TEST_ASSERT_FLOAT_WITHIN(0.11f, 2.0f, event.start_t_s);
}

static void test_reverse_interval_with_hysteresis(void)
{
    road_events_reverse_detector_t detector;
    road_events_reverse_event_t event;
    int events = 0;
    road_events_reverse_init(&detector, road_events_reverse_config_default());
    for (int i = 0; i < 80; ++i) {
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
            events++;
        }
    }
    if (road_events_reverse_finish(&detector, &event)) {
        events++;
    }
    TEST_ASSERT_EQUAL_INT(1, events);
    TEST_ASSERT_TRUE(event.duration_s >= 1.0f);
    TEST_ASSERT_TRUE(event.mean_reverse_speed_mps > 0.4f);
}

static void test_harsh_presets_match_balanced_corner_thresholds(void)
{
    road_events_harsh_behavior_config_t balanced =
        road_events_harsh_behavior_preset_config(ROAD_EVENTS_HARSH_PRESET_BALANCED);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 3.4f, balanced.corner.lateral_accel_threshold_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 2.9f, balanced.corner.exit_lateral_accel_threshold_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 5.0f, balanced.corner.lateral_jerk_threshold_mps3);
}

static void test_trip_stats_accumulates_distance_and_events(void)
{
    road_events_trip_stats_t stats;
    road_events_trip_stats_init(&stats, road_events_trip_config_default());
    for (int i = 0; i <= 10; ++i) {
        float t = (float)i;
        road_events_trip_stats_update_motion(
            &stats,
            (road_events_trip_sample_t){t, 10.0f, 10.0f, true, t, 1u, 0.0f, 0.0f});
    }
    road_events_trip_stats_record_event(&stats, ROAD_EVENTS_TRIP_EVENT_HARSH_ACCELERATION);
    road_events_trip_summary_t summary = road_events_trip_stats_summary(&stats);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 100.0f, summary.distance_m);
    TEST_ASSERT_EQUAL_UINT32(1u, summary.events.harsh_acceleration);
    TEST_ASSERT_TRUE(summary.elevation_valid);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 10.0f, summary.elevation_gain_m);
}

static void test_road_events_replay_matches_rust_golden_trace(void)
{
    road_events_harsh_behavior_config_t harsh =
        road_events_harsh_behavior_preset_config(ROAD_EVENTS_HARSH_PRESET_BALANCED);
    road_events_hill_detector_t hill;
    road_events_reverse_detector_t reverse;
    road_events_harsh_accel_detector_t accel;
    road_events_harsh_brake_detector_t brake;
    road_events_harsh_corner_detector_t corner;
    road_events_trip_stats_t stats;
    road_events_hill_event_t hill_event = {0};
    road_events_reverse_event_t reverse_event = {0};
    road_events_harsh_longitudinal_event_t accel_event = {0};
    road_events_harsh_longitudinal_event_t brake_event = {0};
    road_events_harsh_corner_event_t corner_event = {0};
    int hill_count = 0;
    int reverse_count = 0;
    int accel_count = 0;
    int brake_count = 0;
    int corner_count = 0;

    road_events_hill_init(&hill, road_events_hill_config_default());
    road_events_reverse_init(&reverse, road_events_reverse_config_default());
    road_events_harsh_accel_init(&accel, harsh.accel);
    road_events_harsh_brake_init(&brake, harsh.brake);
    road_events_harsh_corner_init(&corner, harsh.corner);
    road_events_trip_stats_init(&stats, road_events_trip_config_default());

    for (int i = 0; i <= 150; ++i) {
        float t = (float)i * 0.1f;
        float speed = parity_speed_at(t);
        float forward = parity_forward_velocity_at(t);
        float pitch = parity_pitch_at(t);
        float lateral = parity_lateral_at(t);
        float longitudinal_accel =
            i == 0 ? 0.0f : (parity_speed_at(t) - parity_speed_at(t - 0.1f)) / 0.1f;
        road_events_trip_stats_update_motion(
            &stats,
            (road_events_trip_sample_t){t,
                                        speed,
                                        forward,
                                        true,
                                        0.15f * t,
                                        7u,
                                        longitudinal_accel,
                                        lateral});
        if (road_events_hill_update(&hill,
                                    (road_events_hill_sample_t){t, speed, pitch},
                                    &hill_event)) {
            ++hill_count;
            road_events_trip_stats_record_event(&stats, ROAD_EVENTS_TRIP_EVENT_UPHILL);
        }
        if (road_events_reverse_update(&reverse,
                                       (road_events_reverse_sample_t){t, forward},
                                       &reverse_event)) {
            ++reverse_count;
            road_events_trip_stats_record_event(&stats, ROAD_EVENTS_TRIP_EVENT_REVERSE);
        }
        if (road_events_harsh_accel_update(&accel,
                                           (road_events_harsh_longitudinal_sample_t){t, speed},
                                           &accel_event)) {
            ++accel_count;
            road_events_trip_stats_record_event(&stats,
                                                ROAD_EVENTS_TRIP_EVENT_HARSH_ACCELERATION);
        }
        if (road_events_harsh_brake_update(&brake,
                                           (road_events_harsh_longitudinal_sample_t){t, speed},
                                           &brake_event)) {
            ++brake_count;
            road_events_trip_stats_record_event(&stats, ROAD_EVENTS_TRIP_EVENT_HARSH_BRAKING);
        }
        if (road_events_harsh_corner_update(&corner,
                                            (road_events_harsh_corner_sample_t){t, speed, lateral},
                                            &corner_event)) {
            ++corner_count;
            road_events_trip_stats_record_event(&stats,
                                                ROAD_EVENTS_TRIP_EVENT_HARSH_CORNERING);
        }
    }
    if (road_events_reverse_finish(&reverse, &reverse_event)) {
        ++reverse_count;
        road_events_trip_stats_record_event(&stats, ROAD_EVENTS_TRIP_EVENT_REVERSE);
    }
    if (road_events_harsh_accel_finish(&accel, &accel_event)) {
        ++accel_count;
        road_events_trip_stats_record_event(&stats, ROAD_EVENTS_TRIP_EVENT_HARSH_ACCELERATION);
    }
    if (road_events_harsh_brake_finish(&brake, &brake_event)) {
        ++brake_count;
        road_events_trip_stats_record_event(&stats, ROAD_EVENTS_TRIP_EVENT_HARSH_BRAKING);
    }
    if (road_events_harsh_corner_finish(&corner, &corner_event)) {
        ++corner_count;
        road_events_trip_stats_record_event(&stats, ROAD_EVENTS_TRIP_EVENT_HARSH_CORNERING);
    }

    road_events_trip_summary_t summary = road_events_trip_stats_summary(&stats);

    TEST_ASSERT_EQUAL_INT(1, hill_count);
    TEST_ASSERT_EQUAL_INT(1, reverse_count);
    TEST_ASSERT_EQUAL_INT(1, accel_count);
    TEST_ASSERT_EQUAL_INT(1, brake_count);
    TEST_ASSERT_EQUAL_INT(1, corner_count);

    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_HILL_UPHILL, hill_event.kind);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.0f, hill_event.start_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 2.0f, hill_event.end_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.0f, hill_event.duration_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 4.7999997f, hill_event.mean_pitch_deg);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 6.0f, hill_event.mean_speed_mps);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 3.0f, reverse_event.start_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 5.6f, reverse_event.end_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 2.6f, reverse_event.duration_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.6865384f, reverse_event.mean_reverse_speed_mps);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.85f, reverse_event.peak_reverse_speed_mps);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 6.6f, accel_event.start_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 7.6f, accel_event.end_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.0f, accel_event.duration_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 2.6999998f, accel_event.delta_velocity_mps);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 3.1010695f, accel_event.mean_accel_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 3.7922947f, accel_event.peak_accel_mps2);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 9.6f, brake_event.start_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 10.6f, brake_event.end_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.0f, brake_event.duration_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, -3.4799976f, brake_event.delta_velocity_mps);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 3.7049084f, brake_event.mean_accel_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 4.345013f, brake_event.peak_accel_mps2);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 12.5f, corner_event.start_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 13.3f, corner_event.end_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.8000002f, corner_event.duration_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 3.9017348f, corner_event.mean_lateral_accel_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 4.172347f, corner_event.peak_lateral_accel_mps2);

    TEST_ASSERT_EQUAL_UINT32(151u, summary.sample_count);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 15.0f, summary.duration_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 99.953994751f, summary.distance_m);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 2.099999905f, summary.reverse_duration_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.784999847f, summary.reverse_distance_m);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 2.25f, summary.elevation_gain_m);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 6.663599491f, summary.mean_speed_mps);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 11.399999619f, summary.peak_speed_mps);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 4.499998093f, summary.peak_accel_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 5.200023651f, summary.peak_decel_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 4.200002193f, summary.peak_lateral_accel_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 6.061759472f, summary.rolling_speed_mps);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.656066775f,
                             summary.rolling_abs_longitudinal_accel_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.605278075f, summary.rolling_abs_lateral_accel_mps2);
    TEST_ASSERT_EQUAL_UINT32(1u, summary.events.uphill);
    TEST_ASSERT_EQUAL_UINT32(1u, summary.events.reverse);
    TEST_ASSERT_EQUAL_UINT32(1u, summary.events.harsh_acceleration);
    TEST_ASSERT_EQUAL_UINT32(1u, summary.events.harsh_braking);
    TEST_ASSERT_EQUAL_UINT32(1u, summary.events.harsh_cornering);
}

static void test_roughness_replay_matches_rust_golden_trace(void)
{
    road_events_roughness_config_t cfg = road_events_roughness_config_default();
    road_events_roughness_analyzer_t analyzer;
    road_events_roughness_update_t update;
    road_events_roughness_estimate_t final_estimate = {0};
    road_events_shock_event_t last_shock = {0};
    road_events_roughness_event_t last_live = {0};
    road_events_roughness_event_t last_completed = {0};
    road_events_roughness_event_t finish_event = {0};
    float peak_rms = 0.0f;
    float peak_bandpass = 0.0f;
    float peak_clipped = 0.0f;
    unsigned shock_count = 0u;
    unsigned live_count = 0u;
    unsigned completed_count = 0u;
    const float speed_mps = 10.0f;
    const float dt = 0.02f;
    const float spatial_wavelength_m = 5.0f;

    cfg.high_pass_cutoff_hz = 0.05f;
    cfg.low_pass_cutoff_hz = 20.0f;
    cfg.min_speed_mps = 0.5f;
    cfg.shock_min_peak_mps2 = 2.5f;
    road_events_roughness_init(&analyzer, cfg);

    for (int i = 0; i < 5000; ++i) {
        float t = (float)i * dt;
        float x = speed_mps * t;
        float phase = 6.2831853071795864769f * x / spatial_wavelength_m;
        bool shock = fabsf(t - 8.0f) < 0.03f || fabsf(t - 11.0f) < 0.03f ||
                     fabsf(t - 12.5f) < 0.03f || fabsf(t - 14.0f) < 0.03f ||
                     fabsf(t - 96.0f) < 0.03f;
        float texture = (x > 20.0f && x < 90.0f) ? 1.15f * sinf(phase)
                                                  : 0.08f * sinf(phase);
        float accel = texture + (shock ? 6.0f : 0.0f);
        TEST_ASSERT_TRUE(road_events_roughness_update_with_events(
            &analyzer, (road_events_roughness_sample_t){t, speed_mps, accel}, &update));
        final_estimate = update.estimate;
        peak_rms = fmaxf(peak_rms, update.estimate.roughness_rms_mps2);
        peak_bandpass = fmaxf(peak_bandpass, fabsf(update.estimate.vertical_accel_bandpass_mps2));
        peak_clipped = fmaxf(peak_clipped, fabsf(update.estimate.vertical_accel_clipped_mps2));
        if (update.has_shock_event) {
            ++shock_count;
            last_shock = update.shock_event;
        }
        if (update.has_roughness_event) {
            ++live_count;
            last_live = update.roughness_event;
        }
        if (update.has_completed_roughness_event) {
            ++completed_count;
            last_completed = update.completed_roughness_event;
        }
    }

    TEST_ASSERT_FALSE(road_events_roughness_finish(&analyzer, &finish_event));
    TEST_ASSERT_EQUAL_UINT32(5u, shock_count);
    TEST_ASSERT_EQUAL_UINT32(1u, live_count);
    TEST_ASSERT_EQUAL_UINT32(1u, completed_count);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.891504467f, peak_rms);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 5.960410118f, peak_bandpass);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.5f, peak_clipped);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 99.979995728f, final_estimate.t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.079187021f, final_estimate.roughness_rms_mps2);
    TEST_ASSERT_EQUAL_INT(ROAD_EVENTS_ROUGHNESS_VERY_SMOOTH, final_estimate.level);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, -0.057348553f,
                             final_estimate.vertical_accel_bandpass_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, -0.057348553f,
                             final_estimate.vertical_accel_clipped_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, 999.820190430f, final_estimate.distance_m);
    TEST_ASSERT_TRUE(final_estimate.updated);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 95.979995728f, last_shock.start_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 96.040000916f, last_shock.end_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.060005188f, last_shock.duration_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 5.778273582f,
                             last_shock.peak_abs_vertical_accel_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 10.0f, last_shock.mean_speed_mps);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 3.879999876f, last_live.start_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 4.880000114f, last_live.end_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.000000238f, last_live.duration_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.682472467f,
                             last_live.mean_roughness_rms_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.736603260f,
                             last_live.peak_roughness_rms_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 10.0f, last_live.mean_speed_mps);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 10.000001907f, last_live.distance_m);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 3.879999876f, last_completed.start_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 10.420000076f, last_completed.end_t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 6.539999962f, last_completed.duration_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.739212573f,
                             last_completed.mean_roughness_rms_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.891504467f,
                             last_completed.peak_roughness_rms_mps2);
    TEST_ASSERT_FLOAT_WITHIN(5.0e-5f, 10.000034332f, last_completed.mean_speed_mps);
    TEST_ASSERT_FLOAT_WITHIN(2.0e-4f, 65.400222778f, last_completed.distance_m);
}

static void test_speed_bump_replay_matches_rust_golden_trace(void)
{
    road_events_speed_bump_config_t cfg = road_events_speed_bump_config_default();
    road_events_speed_bump_detector_t detector;
    road_events_speed_bump_diagnostic_t diagnostic = {0};
    road_events_speed_bump_event_t event = {0};
    road_events_speed_bump_event_t last_event = {0};
    float peak_abs_pitch_hpf = 0.0f;
    float peak_abs_accel_hpf = 0.0f;
    unsigned events = 0u;

    cfg.trigger_confidence = 0.12f;
    road_events_speed_bump_init(&detector, cfg);
    for (int i = 0; i < 600; ++i) {
        float t = (float)i * 0.01f;
        float accel = parity_gaussian(t, 2.00f, 0.12f) -
                      1.2f * parity_gaussian(t, 2.30f, 0.12f) +
                      parity_gaussian(t, 2.60f, 0.12f);
        float pitch = 0.90f * parity_gaussian(t, 2.10f, 0.14f) -
                      1.10f * parity_gaussian(t, 2.42f, 0.14f);
        if (road_events_speed_bump_update(
                &detector,
                (road_events_speed_bump_sample_t){t, 4.0f, pitch, 4.0f * accel},
                &diagnostic,
                &event)) {
            ++events;
            last_event = event;
        }
        peak_abs_pitch_hpf = fmaxf(peak_abs_pitch_hpf, fabsf(diagnostic.pitch_hpf_deg));
        peak_abs_accel_hpf =
            fmaxf(peak_abs_accel_hpf, fabsf(diagnostic.vertical_accel_hpf_mps2));
    }

    TEST_ASSERT_EQUAL_UINT32(1u, events);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.014764309f, peak_abs_pitch_hpf);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 4.174280643f, peak_abs_accel_hpf);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 5.989999771f, diagnostic.t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.000036202f, diagnostic.pitch_hpf_deg);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.059578892f, diagnostic.pitch_noise_deg);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, -0.000001988f,
                             diagnostic.vertical_accel_hpf_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.263562053f,
                             diagnostic.vertical_accel_noise_mps2);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 2.259999990f, last_event.t_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.964721441f, last_event.confidence);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.620000005f, last_event.duration_s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.014764309f, last_event.peak_abs_pitch_deg);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_hill_emits_once_when_confirmed);
    RUN_TEST(test_reverse_interval_with_hysteresis);
    RUN_TEST(test_harsh_presets_match_balanced_corner_thresholds);
    RUN_TEST(test_trip_stats_accumulates_distance_and_events);
    RUN_TEST(test_road_events_replay_matches_rust_golden_trace);
    RUN_TEST(test_roughness_replay_matches_rust_golden_trace);
    RUN_TEST(test_speed_bump_replay_matches_rust_golden_trace);
    return UNITY_END();
}
