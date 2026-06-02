#include "sensor_fusion.h"
#include "unity.h"

#include <math.h>

#define TEST_EARTH_RADIUS_M 6378137.0
#define TEST_PI 3.14159265358979323846

static sensor_fusion_gnss_sample_t nominal_gnss(void)
{
    sensor_fusion_gnss_sample_t sample = {0};
    sample.t_s = 0.02f;
    sample.lat_deg = 37.0;
    sample.lon_deg = -122.0;
    sample.height_m = 10.0;
    sample.vel_ned_mps[0] = 6.0f;
    sample.vel_ned_mps[1] = 0.0f;
    sample.vel_ned_mps[2] = 0.0f;
    sample.pos_std_m[0] = 1.0f;
    sample.pos_std_m[1] = 1.0f;
    sample.pos_std_m[2] = 2.5f;
    sample.vel_std_mps[0] = 0.1f;
    sample.vel_std_mps[1] = 0.1f;
    sample.vel_std_mps[2] = 0.2f;
    sample.has_heading_rad = true;
    sample.heading_rad = 0.0f;
    return sample;
}

static sensor_fusion_gnss_sample_t lateral_gnss(void)
{
    sensor_fusion_gnss_sample_t sample = nominal_gnss();
    sample.vel_ned_mps[1] = 2.0f;
    return sample;
}

static sensor_fusion_gnss_sample_t low_speed_no_heading_gnss(void)
{
    sensor_fusion_gnss_sample_t sample = nominal_gnss();
    sample.vel_ned_mps[0] = 0.2f;
    sample.vel_ned_mps[1] = 0.0f;
    sample.has_heading_rad = false;
    sample.heading_rad = 0.0f;
    return sample;
}

static float expected_north_offset_m(double anchor_lat_deg, double lat_deg)
{
    return (float)((lat_deg - anchor_lat_deg) * (TEST_PI / 180.0) * TEST_EARTH_RADIUS_M);
}

static float expected_east_offset_m(double anchor_lat_deg,
                                    double anchor_lon_deg,
                                    double lon_deg)
{
    double lat0_rad = anchor_lat_deg * (TEST_PI / 180.0);
    return (float)((lon_deg - anchor_lon_deg) * (TEST_PI / 180.0) *
                   cos(lat0_rad) * TEST_EARTH_RADIUS_M);
}

static void test_default_config_initializes_auto_not_ready(void)
{
    sensor_fusion_t ctx;
    sensor_fusion_config_t config = sensor_fusion_config_default();
    sensor_fusion_init(&ctx, config);

    sensor_fusion_health_t health = sensor_fusion_health(&ctx);
    TEST_ASSERT_FALSE(health.usable);
    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_NOT_READY, health.state);
    TEST_ASSERT_TRUE((health.reason_mask & SENSOR_FUSION_HEALTH_REASON_MOUNT_NOT_READY) != 0u);
}

static void test_manual_mount_is_normalized_and_reported(void)
{
    sensor_fusion_t ctx;
    float q_bv[4] = {2.0f, 0.0f, 0.0f, 0.0f};
    float reported[4] = {0};

    sensor_fusion_init_with_mount(&ctx, q_bv);

    TEST_ASSERT_TRUE(sensor_fusion_mount_q_bv(&ctx, reported));
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, reported[0]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, reported[1]);
}

static void test_auto_gnss_does_not_seed_identity_mount(void)
{
    sensor_fusion_t ctx;
    sensor_fusion_update_t update;
    sensor_fusion_health_t health;
    float q_bv[4];

    sensor_fusion_init_auto(&ctx);
    update = sensor_fusion_process_gnss(&ctx, nominal_gnss());
    health = sensor_fusion_health(&ctx);

    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_NOT_READY, update.state);
    TEST_ASSERT_FALSE(update.mount_ready);
    TEST_ASSERT_FALSE(update.navigation_started);
    TEST_ASSERT_FALSE(update.navigation_usable);
    TEST_ASSERT_FALSE(sensor_fusion_mount_q_bv(&ctx, q_bv));
    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_NOT_READY, health.state);
    TEST_ASSERT_TRUE((health.reason_mask & SENSOR_FUSION_HEALTH_REASON_MOUNT_NOT_READY) != 0u);
}

static void test_auto_stationary_tilt_init_reports_progress_without_mount_ready(void)
{
    sensor_fusion_t ctx;
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_align_progress_t progress;
    float q_bv[4];

    sensor_fusion_init_auto(&ctx);
    imu.accel_mps2[2] = -9.80665f;
    for (int i = 0; i < 100; ++i) {
        imu.t_s = 0.01f * (float)i;
        (void)sensor_fusion_process_imu(&ctx, imu);
    }
    progress = sensor_fusion_align_progress(&ctx);

    TEST_ASSERT_TRUE(progress.valid);
    TEST_ASSERT_FALSE(progress.coarse_ready);
    TEST_ASSERT_FALSE(ctx.mount_ready);
    TEST_ASSERT_FALSE(sensor_fusion_mount_q_bv(&ctx, q_bv));
    TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, 10.0f, progress.roll_sigma_deg);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, 10.0f, progress.pitch_sigma_deg);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, 60.0f, progress.yaw_sigma_deg);
}

static void test_auto_align_handoff_starts_navigation_after_motion_windows(void)
{
    sensor_fusion_t ctx;
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_gnss_sample_t gnss = nominal_gnss();
    sensor_fusion_update_t update = {0};
    sensor_fusion_align_progress_t progress;
    sensor_fusion_ekf_state_t state;

    sensor_fusion_init_auto(&ctx);
    imu.accel_mps2[2] = -9.80665f;
    for (int i = 0; i < 100; ++i) {
        imu.t_s = 0.01f * (float)i;
        (void)sensor_fusion_process_imu(&ctx, imu);
    }

    gnss.t_s = 1.0f;
    gnss.has_heading_rad = false;
    gnss.vel_ned_mps[0] = 0.0f;
    gnss.vel_ned_mps[1] = 0.0f;
    update = sensor_fusion_process_gnss(&ctx, gnss);
    TEST_ASSERT_FALSE(update.mount_ready);
    TEST_ASSERT_FALSE(update.navigation_started);

    for (int window = 0; window < 20; ++window) {
        for (int k = 1; k <= 100; ++k) {
            imu.t_s = 1.0f + (float)window + 0.01f * (float)k;
            imu.accel_mps2[0] = 0.0f;
            imu.accel_mps2[1] = 0.0f;
            imu.accel_mps2[2] = -9.80665f;
            (void)sensor_fusion_process_imu(&ctx, imu);
        }
        gnss.t_s = 2.0f + (float)window;
        gnss.vel_ned_mps[0] = 0.0f;
        update = sensor_fusion_process_gnss(&ctx, gnss);
        TEST_ASSERT_FALSE(update.navigation_started);
    }

    for (int window = 0; window < 80 && !update.navigation_started; ++window) {
        for (int k = 1; k <= 100; ++k) {
            imu.t_s = 21.0f + (float)window + 0.01f * (float)k;
            imu.accel_mps2[0] = 0.0f;
            imu.accel_mps2[1] = 5.0f;
            imu.accel_mps2[2] = -9.80665f;
            (void)sensor_fusion_process_imu(&ctx, imu);
        }
        gnss.t_s = 22.0f + (float)window;
        gnss.vel_ned_mps[0] = 10.0f + 5.0f * (float)(window + 1);
        update = sensor_fusion_process_gnss(&ctx, gnss);
    }

    progress = sensor_fusion_align_progress(&ctx);
    TEST_ASSERT_TRUE(update.mount_ready);
    TEST_ASSERT_TRUE(update.mount_ready_changed);
    TEST_ASSERT_TRUE(update.navigation_started);
    TEST_ASSERT_TRUE(update.navigation_usable);
    TEST_ASSERT_TRUE(progress.valid);
    TEST_ASSERT_TRUE(progress.coarse_ready);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, progress.progress);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &state));
    TEST_ASSERT_TRUE(state.covariance[17][17] < (10.0f * TEST_PI / 180.0f) *
                                                 (10.0f * TEST_PI / 180.0f));
}

static void seed_auto_running_context(sensor_fusion_t *ctx)
{
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_gnss_sample_t gnss = nominal_gnss();
    sensor_fusion_update_t update = {0};

    sensor_fusion_init_auto(ctx);
    imu.accel_mps2[2] = -9.80665f;
    for (int i = 0; i < 100; ++i) {
        imu.t_s = 0.01f * (float)i;
        (void)sensor_fusion_process_imu(ctx, imu);
    }
    gnss.t_s = 1.0f;
    gnss.has_heading_rad = false;
    gnss.vel_ned_mps[0] = 0.0f;
    (void)sensor_fusion_process_gnss(ctx, gnss);
    for (int window = 0; window < 20; ++window) {
        for (int k = 1; k <= 100; ++k) {
            imu.t_s = 1.0f + (float)window + 0.01f * (float)k;
            imu.accel_mps2[0] = 0.0f;
            imu.accel_mps2[1] = 0.0f;
            imu.accel_mps2[2] = -9.80665f;
            (void)sensor_fusion_process_imu(ctx, imu);
        }
        gnss.t_s = 2.0f + (float)window;
        gnss.vel_ned_mps[0] = 0.0f;
        (void)sensor_fusion_process_gnss(ctx, gnss);
    }
    for (int window = 0; window < 80 && !update.navigation_started; ++window) {
        for (int k = 1; k <= 100; ++k) {
            imu.t_s = 21.0f + (float)window + 0.01f * (float)k;
            imu.accel_mps2[0] = 0.0f;
            imu.accel_mps2[1] = 5.0f;
            imu.accel_mps2[2] = -9.80665f;
            (void)sensor_fusion_process_imu(ctx, imu);
        }
        gnss.t_s = 22.0f + (float)window;
        gnss.vel_ned_mps[0] = 10.0f + 5.0f * (float)(window + 1);
        update = sensor_fusion_process_gnss(ctx, gnss);
    }
    TEST_ASSERT_TRUE(update.navigation_started);
}

static void test_auto_expected_long_sleep_reseed_uses_velocity_yaw_without_heading(void)
{
    sensor_fusion_t ctx;
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_gnss_sample_t gnss = nominal_gnss();
    sensor_fusion_update_t update;
    sensor_fusion_ekf_state_t state;

    seed_auto_running_context(&ctx);
    (void)sensor_fusion_end_trip(&ctx);
    imu.t_s = 5000.0f;
    imu.accel_mps2[2] = -9.80665f;
    update = sensor_fusion_process_imu(&ctx, imu);
    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED, update.state);

    gnss.t_s = 5001.0f;
    gnss.has_heading_rad = false;
    gnss.vel_ned_mps[0] = 12.0f;
    gnss.vel_ned_mps[1] = 0.0f;
    gnss.lat_deg += 0.0001;
    update = sensor_fusion_process_gnss(&ctx, gnss);

    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_RUNNING, update.state);
    TEST_ASSERT_TRUE(update.navigation_started);
    TEST_ASSERT_TRUE(update.navigation_usable);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &state));
}

static void test_auto_align_pre_ekf_imu_gap_clears_pending_window(void)
{
    sensor_fusion_t ctx;
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_gnss_sample_t gnss = nominal_gnss();
    sensor_fusion_update_t update;
    sensor_fusion_align_progress_t progress;

    sensor_fusion_init_auto(&ctx);
    imu.accel_mps2[2] = -9.80665f;
    for (int i = 0; i < 100; ++i) {
        imu.t_s = 0.01f * (float)i;
        (void)sensor_fusion_process_imu(&ctx, imu);
    }
    gnss.t_s = 1.0f;
    gnss.has_heading_rad = false;
    gnss.vel_ned_mps[0] = 10.0f;
    update = sensor_fusion_process_gnss(&ctx, gnss);
    TEST_ASSERT_FALSE(update.navigation_started);

    for (int k = 1; k <= 10; ++k) {
        imu.t_s = 1.0f + 0.01f * (float)k;
        imu.accel_mps2[1] = 5.0f;
        imu.accel_mps2[2] = -9.80665f;
        (void)sensor_fusion_process_imu(&ctx, imu);
    }
    imu.t_s = 2.0f;
    (void)sensor_fusion_process_imu(&ctx, imu);

    gnss.t_s = 2.1f;
    gnss.vel_ned_mps[0] = 15.0f;
    update = sensor_fusion_process_gnss(&ctx, gnss);
    progress = sensor_fusion_align_progress(&ctx);

    TEST_ASSERT_FALSE(update.mount_ready);
    TEST_ASSERT_FALSE(update.navigation_started);
    TEST_ASSERT_TRUE(progress.valid);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, 60.0f, progress.yaw_sigma_deg);
}

static void test_manual_gnss_starts_navigation_snapshot(void)
{
    sensor_fusion_t ctx;
    float q_bv[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    sensor_fusion_ekf_state_t nav;
    double lla[3];

    sensor_fusion_init_with_mount(&ctx, q_bv);
    sensor_fusion_update_t update = sensor_fusion_process_gnss(&ctx, nominal_gnss());

    TEST_ASSERT_TRUE(update.navigation_started);
    TEST_ASSERT_TRUE(update.navigation_usable);
    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_RUNNING, update.state);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &nav));
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 6.0f, nav.vel_ned_mps[0]);
    TEST_ASSERT_TRUE(sensor_fusion_position_lla(&ctx, lla));
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 10.0f, (float)lla[2]);
}

static void test_manual_low_speed_no_heading_gnss_does_not_seed_yaw(void)
{
    sensor_fusion_t ctx;
    float q_bv[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    sensor_fusion_update_t update;
    sensor_fusion_ekf_state_t state;

    sensor_fusion_init_with_mount(&ctx, q_bv);
    update = sensor_fusion_process_gnss(&ctx, low_speed_no_heading_gnss());

    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_INITIALIZING, update.state);
    TEST_ASSERT_TRUE(update.mount_ready);
    TEST_ASSERT_FALSE(update.navigation_started);
    TEST_ASSERT_FALSE(update.navigation_usable);
    TEST_ASSERT_FALSE(sensor_fusion_ekf_state(&ctx, &state));
}

static void test_delayed_manual_seed_uses_current_gnss_position_not_anchor(void)
{
    sensor_fusion_t ctx;
    float q_bv[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    sensor_fusion_gnss_sample_t first = low_speed_no_heading_gnss();
    sensor_fusion_gnss_sample_t second = nominal_gnss();
    sensor_fusion_update_t update;
    sensor_fusion_ekf_state_t state;
    double lla[3];

    second.t_s = 0.50f;
    second.lat_deg = first.lat_deg + 0.00003;
    second.lon_deg = first.lon_deg - 0.00004;
    second.height_m = first.height_m + 2.0;

    sensor_fusion_init_with_mount(&ctx, q_bv);
    update = sensor_fusion_process_gnss(&ctx, first);
    TEST_ASSERT_FALSE(update.navigation_started);

    update = sensor_fusion_process_gnss(&ctx, second);
    TEST_ASSERT_TRUE(update.navigation_started);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &state));
    TEST_ASSERT_FLOAT_WITHIN(0.02f,
                             expected_north_offset_m(first.lat_deg, second.lat_deg),
                             state.pos_ned_m[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.02f,
                             expected_east_offset_m(first.lat_deg, first.lon_deg, second.lon_deg),
                             state.pos_ned_m[1]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, -2.0f, state.pos_ned_m[2]);
    TEST_ASSERT_TRUE(sensor_fusion_position_lla(&ctx, lla));
    TEST_ASSERT_FLOAT_WITHIN(1.0e-7f, (float)second.lat_deg, (float)lla[0]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-7f, (float)second.lon_deg, (float)lla[1]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, (float)second.height_m, (float)lla[2]);
}

static void test_manual_imu_prediction_uses_ekf_runtime(void)
{
    sensor_fusion_t ctx;
    float q_bv[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_ekf_state_t nav;
    double lla[3];

    sensor_fusion_init_with_mount(&ctx, q_bv);
    sensor_fusion_set_r_vehicle_speed(&ctx, 0.04f);
    (void)sensor_fusion_process_gnss(&ctx, nominal_gnss());
    imu.t_s = 0.02f;
    imu.accel_mps2[2] = -9.80665f;
    (void)sensor_fusion_process_imu(&ctx, imu);
    for (int step = 1; step <= 20; ++step) {
        imu.t_s = 0.02f + 0.025f * (float)step;
        imu.accel_mps2[2] = -9.80665f;
        (void)sensor_fusion_process_imu(&ctx, imu);
    }

    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &nav));
    TEST_ASSERT_TRUE(nav.pos_ned_m[0] > 2.5f);
    TEST_ASSERT_TRUE(nav.pos_ned_m[0] < 3.5f);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, 6.0f, nav.vel_ned_mps[0]);
    TEST_ASSERT_TRUE(sensor_fusion_position_lla(&ctx, lla));
    TEST_ASSERT_TRUE(lla[0] > 37.0);
    TEST_ASSERT_FLOAT_WITHIN(3.0e-7f,
                             37.0 + (double)nav.pos_ned_m[0] / 6378137.0 *
                                        (180.0 / 3.14159265358979323846),
                             lla[0]);
}

static void test_manual_gnss_update_reports_runtime_gate_events(void)
{
    sensor_fusion_t ctx;
    float q_bv[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_gnss_sample_t jump = nominal_gnss();
    sensor_fusion_update_t update;

    sensor_fusion_init_with_mount(&ctx, q_bv);
    (void)sensor_fusion_process_gnss(&ctx, nominal_gnss());
    imu.t_s = 0.02f;
    imu.accel_mps2[2] = -9.80665f;
    (void)sensor_fusion_process_imu(&ctx, imu);
    jump.t_s = 0.04f;
    jump.lat_deg += 0.01;
    update = sensor_fusion_process_gnss(&ctx, jump);
    TEST_ASSERT_EQUAL(0u, update.gnss_event_mask);
    imu.t_s = 0.05f;
    update = sensor_fusion_process_imu(&ctx, imu);

    TEST_ASSERT_TRUE((update.gnss_event_mask & SENSOR_FUSION_GNSS_EVENT_POSITION_REJECTED) != 0u);
    TEST_ASSERT_FALSE((update.gnss_event_mask & SENSOR_FUSION_GNSS_EVENT_POSITION_GAP_BYPASS) != 0u);
}

static void test_manual_vehicle_speed_update_uses_runtime_body_speed_row(void)
{
    sensor_fusion_t ctx;
    float q_bv[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    sensor_fusion_vehicle_speed_sample_t speed = {0};
    sensor_fusion_ekf_state_t before;
    sensor_fusion_ekf_state_t after;

    sensor_fusion_init_with_mount(&ctx, q_bv);
    (void)sensor_fusion_process_gnss(&ctx, nominal_gnss());
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &before));
    speed.t_s = 0.10f;
    speed.speed_mps = 3.0f;
    speed.direction = SENSOR_FUSION_VEHICLE_SPEED_FORWARD;
    (void)sensor_fusion_process_vehicle_speed(&ctx, speed);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &after));

    TEST_ASSERT_TRUE(after.vel_ned_mps[0] < before.vel_ned_mps[0]);
    TEST_ASSERT_TRUE(after.vel_ned_mps[0] > speed.speed_mps);
}

static void test_unknown_tiny_vehicle_speed_uses_zero_velocity(void)
{
    sensor_fusion_t ctx;
    float q_bv[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    sensor_fusion_vehicle_speed_sample_t speed = {0};
    sensor_fusion_ekf_state_t before;
    sensor_fusion_ekf_state_t after;

    sensor_fusion_init_with_mount(&ctx, q_bv);
    (void)sensor_fusion_process_gnss(&ctx, lateral_gnss());
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &before));
    speed.t_s = 0.10f;
    speed.speed_mps = 0.1f;
    speed.direction = SENSOR_FUSION_VEHICLE_SPEED_UNKNOWN;
    (void)sensor_fusion_process_vehicle_speed(&ctx, speed);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &after));

    TEST_ASSERT_TRUE(fabsf(after.vel_ned_mps[0]) < fabsf(before.vel_ned_mps[0]));
    TEST_ASSERT_TRUE(fabsf(after.vel_ned_mps[1]) < fabsf(before.vel_ned_mps[1]));
}

static void test_manual_imu_applies_decimated_nhc_from_facade_config(void)
{
    sensor_fusion_t ctx;
    float q_bv[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_ekf_state_t before;
    sensor_fusion_ekf_state_t after;

    sensor_fusion_init_with_mount(&ctx, q_bv);
    sensor_fusion_set_r_vehicle_roll_prior(&ctx, 0.0f);
    (void)sensor_fusion_process_gnss(&ctx, lateral_gnss());
    imu.t_s = 0.02f;
    imu.accel_mps2[2] = -9.80665f;
    (void)sensor_fusion_process_imu(&ctx, imu);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &before));
    imu.t_s = 0.07f;
    (void)sensor_fusion_process_imu(&ctx, imu);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &after));

    TEST_ASSERT_TRUE(fabsf(after.vel_ned_mps[1]) < fabsf(before.vel_ned_mps[1]));
}

static void seed_manual_running_context(sensor_fusion_t *ctx)
{
    float q_bv[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_init_with_mount(ctx, q_bv);
    sensor_fusion_set_r_vehicle_roll_prior(ctx, 0.0f);
    (void)sensor_fusion_process_gnss(ctx, nominal_gnss());
    imu.t_s = 0.02f;
    imu.accel_mps2[2] = -9.80665f;
    (void)sensor_fusion_process_imu(ctx, imu);
}

static void test_expected_short_sleep_ages_covariance_but_stays_usable(void)
{
    sensor_fusion_t ctx;
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_ekf_state_t before;
    sensor_fusion_ekf_state_t after;
    sensor_fusion_update_t update;
    seed_manual_running_context(&ctx);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &before));

    (void)sensor_fusion_end_trip(&ctx);
    imu.t_s = 60.02f;
    imu.accel_mps2[2] = -9.80665f;
    update = sensor_fusion_process_imu(&ctx, imu);

    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_RUNNING, update.state);
    TEST_ASSERT_TRUE(update.navigation_usable);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &after));
    TEST_ASSERT_TRUE(after.covariance[6][6] > before.covariance[6][6]);
    TEST_ASSERT_TRUE(after.covariance[3][3] > before.covariance[3][3]);
}

static void test_unexpected_large_gap_awaits_gnss_reseed(void)
{
    sensor_fusion_t ctx;
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_update_t update;
    sensor_fusion_health_t health;
    sensor_fusion_ekf_state_t state;
    double lla[3];
    seed_manual_running_context(&ctx);

    imu.t_s = 2.02f;
    imu.accel_mps2[2] = -9.80665f;
    update = sensor_fusion_process_imu(&ctx, imu);
    health = sensor_fusion_health(&ctx);

    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED, update.state);
    TEST_ASSERT_FALSE(update.navigation_usable);
    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED, health.state);
    TEST_ASSERT_TRUE((health.reason_mask & SENSOR_FUSION_HEALTH_REASON_NAV_UNUSABLE) != 0u);
    TEST_ASSERT_FALSE(sensor_fusion_ekf_state(&ctx, &state));
    TEST_ASSERT_FALSE(sensor_fusion_position_lla(&ctx, lla));
}

static void test_unexpected_gap_reseed_preserves_attitude_without_yaw_seed(void)
{
    sensor_fusion_t ctx;
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_gnss_sample_t sample = low_speed_no_heading_gnss();
    sensor_fusion_ekf_state_t before_gap;
    sensor_fusion_ekf_state_t after_reseed;
    sensor_fusion_update_t update;
    seed_manual_running_context(&ctx);

    imu.t_s = 0.045f;
    imu.gyro_radps[2] = 1.0f;
    imu.accel_mps2[2] = -9.80665f;
    (void)sensor_fusion_process_imu(&ctx, imu);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &before_gap));
    TEST_ASSERT_TRUE(fabsf(before_gap.q_nv[3]) > 1.0e-3f);

    imu.t_s = 2.045f;
    update = sensor_fusion_process_imu(&ctx, imu);
    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED, update.state);

    sample.t_s = 2.10f;
    sample.lat_deg += 0.00002;
    update = sensor_fusion_process_gnss(&ctx, sample);
    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_RUNNING, update.state);
    TEST_ASSERT_TRUE(update.navigation_started);
    TEST_ASSERT_TRUE(update.navigation_usable);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &after_reseed));
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, before_gap.q_nv[0], after_reseed.q_nv[0]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, before_gap.q_nv[1], after_reseed.q_nv[1]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, before_gap.q_nv[2], after_reseed.q_nv[2]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, before_gap.q_nv[3], after_reseed.q_nv[3]);
    TEST_ASSERT_TRUE(after_reseed.covariance[2][2] >=
                     (45.0f * 3.14159265358979323846f / 180.0f) *
                         (45.0f * 3.14159265358979323846f / 180.0f));
    TEST_ASSERT_TRUE(after_reseed.pos_ned_m[0] > 2.0f);
}

static void test_expected_long_sleep_awaits_gnss_reseed_and_gnss_restores_running(void)
{
    sensor_fusion_t ctx;
    sensor_fusion_imu_sample_t imu = {0};
    sensor_fusion_gnss_sample_t sample = nominal_gnss();
    sensor_fusion_update_t update;
    sensor_fusion_ekf_state_t state;
    double anchor_lat_deg = sample.lat_deg;
    double anchor_lon_deg = sample.lon_deg;
    seed_manual_running_context(&ctx);

    (void)sensor_fusion_end_trip(&ctx);
    imu.t_s = 4000.02f;
    imu.accel_mps2[2] = -9.80665f;
    update = sensor_fusion_process_imu(&ctx, imu);
    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED, update.state);
    TEST_ASSERT_FALSE(update.navigation_usable);
    TEST_ASSERT_FALSE(sensor_fusion_ekf_state(&ctx, &state));

    sample.t_s = 4001.0f;
    sample.lat_deg += 0.00001;
    sample.lon_deg += 0.00002;
    update = sensor_fusion_process_gnss(&ctx, sample);
    TEST_ASSERT_EQUAL(SENSOR_FUSION_STATE_RUNNING, update.state);
    TEST_ASSERT_TRUE(update.navigation_usable);
    TEST_ASSERT_TRUE(update.navigation_started);
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&ctx, &state));
    TEST_ASSERT_FLOAT_WITHIN(0.02f,
                             expected_north_offset_m(anchor_lat_deg, sample.lat_deg),
                             state.pos_ned_m[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.02f,
                             expected_east_offset_m(anchor_lat_deg, anchor_lon_deg, sample.lon_deg),
                             state.pos_ned_m[1]);
}

static void test_align_progress_reports_manual_mount_ready(void)
{
    sensor_fusion_t ctx;
    float q_bv[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    sensor_fusion_init_with_mount(&ctx, q_bv);
    sensor_fusion_align_progress_t progress = sensor_fusion_align_progress(&ctx);
    TEST_ASSERT_TRUE(progress.valid);
    TEST_ASSERT_TRUE(progress.coarse_ready);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, progress.progress);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_default_config_initializes_auto_not_ready);
    RUN_TEST(test_manual_mount_is_normalized_and_reported);
    RUN_TEST(test_auto_gnss_does_not_seed_identity_mount);
    RUN_TEST(test_auto_stationary_tilt_init_reports_progress_without_mount_ready);
    RUN_TEST(test_auto_align_handoff_starts_navigation_after_motion_windows);
    RUN_TEST(test_auto_expected_long_sleep_reseed_uses_velocity_yaw_without_heading);
    RUN_TEST(test_auto_align_pre_ekf_imu_gap_clears_pending_window);
    RUN_TEST(test_manual_gnss_starts_navigation_snapshot);
    RUN_TEST(test_manual_low_speed_no_heading_gnss_does_not_seed_yaw);
    RUN_TEST(test_delayed_manual_seed_uses_current_gnss_position_not_anchor);
    RUN_TEST(test_manual_imu_prediction_uses_ekf_runtime);
    RUN_TEST(test_manual_gnss_update_reports_runtime_gate_events);
    RUN_TEST(test_manual_vehicle_speed_update_uses_runtime_body_speed_row);
    RUN_TEST(test_unknown_tiny_vehicle_speed_uses_zero_velocity);
    RUN_TEST(test_manual_imu_applies_decimated_nhc_from_facade_config);
    RUN_TEST(test_expected_short_sleep_ages_covariance_but_stays_usable);
    RUN_TEST(test_unexpected_large_gap_awaits_gnss_reseed);
    RUN_TEST(test_unexpected_gap_reseed_preserves_attitude_without_yaw_seed);
    RUN_TEST(test_expected_long_sleep_awaits_gnss_reseed_and_gnss_restores_running);
    RUN_TEST(test_align_progress_reports_manual_mount_ready);
    return UNITY_END();
}
