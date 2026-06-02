#include "align.h"
#include "unity.h"

#include <math.h>

#define TEST_PI 3.14159265358979323846f

static float deg_to_rad(float deg)
{
    return deg * (TEST_PI / 180.0f);
}

static float sq(float x)
{
    return x * x;
}

static void set_diag3(float p[3][3], float d0, float d1, float d2)
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            p[i][j] = 0.0f;
        }
    }
    p[0][0] = d0;
    p[1][1] = d1;
    p[2][2] = d2;
}

static sf_align_window_summary_t straight_yaw_window(void)
{
    sf_align_window_summary_t window = {0};
    window.dt = 1.0f;
    window.mean_accel_b[0] = 0.0f;
    window.mean_accel_b[1] = 1.0f;
    window.mean_accel_b[2] = -SF_ALIGN_GRAVITY_MPS2;
    window.gnss_vel_prev_n[0] = 10.0f;
    window.gnss_vel_curr_n[0] = 11.0f;
    window.gnss_vel_prev_std_mps[0] = 0.05f;
    window.gnss_vel_prev_std_mps[1] = 0.05f;
    window.gnss_vel_prev_std_mps[2] = 0.05f;
    window.gnss_vel_curr_std_mps[0] = 0.05f;
    window.gnss_vel_curr_std_mps[1] = 0.05f;
    window.gnss_vel_curr_std_mps[2] = 0.05f;
    window.imu_sample_count = 100u;
    return window;
}

static void test_stationary_tilt_init_sets_identity_tilt_and_conservative_sigma(void)
{
    sf_align_t align;
    float samples[16][3];
    float angles_deg[3];
    float sigma_deg[3];

    sf_align_init(&align, sf_align_config_default());
    for (size_t i = 0u; i < 16u; ++i) {
        samples[i][0] = 0.0f;
        samples[i][1] = 0.0f;
        samples[i][2] = -SF_ALIGN_GRAVITY_MPS2;
    }

    TEST_ASSERT_TRUE(sf_align_initialize_from_stationary(&align, samples, 16u));
    sf_align_mount_angles_deg(&align, angles_deg);
    sf_align_sigma_deg(&align, sigma_deg);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, 0.0f, angles_deg[0]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, 0.0f, angles_deg[1]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, 0.0f, angles_deg[2]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, 10.0f, sigma_deg[0]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, 10.0f, sigma_deg[1]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, 60.0f, sigma_deg[2]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, -SF_ALIGN_GRAVITY_MPS2, align.gravity_lp_b[2]);
    TEST_ASSERT_FALSE(sf_align_coarse_alignment_ready(&align));
}

static void test_predict_adds_mount_random_walk_variance(void)
{
    sf_align_config_t cfg = sf_align_config_default();
    sf_align_t align;
    float before[3];
    cfg.q_mount_std_rad[0] = 0.01f;
    cfg.q_mount_std_rad[1] = 0.02f;
    cfg.q_mount_std_rad[2] = 0.03f;
    sf_align_init(&align, cfg);
    before[0] = align.p[0][0];
    before[1] = align.p[1][1];
    before[2] = align.p[2][2];

    sf_align_predict(&align, 2.0f);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-7f, before[0] + 0.01f * 0.01f * 2.0f, align.p[0][0]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-7f, before[1] + 0.02f * 0.02f * 2.0f, align.p[1][1]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-7f, before[2] + 0.03f * 0.03f * 2.0f, align.p[2][2]);
}

static void test_coarse_progress_and_readiness_follow_axis_sigmas(void)
{
    sf_align_t align;
    float sigma_deg[3];
    sf_align_init(&align, sf_align_config_default());

    set_diag3(align.p,
              sq(deg_to_rad(10.0f)),
              sq(deg_to_rad(10.0f)),
              sq(deg_to_rad(60.0f)));
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, sf_align_coarse_progress(&align));

    set_diag3(align.p, sq(deg_to_rad(5.0f)), sq(deg_to_rad(5.0f)), sq(deg_to_rad(60.0f)));
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.30f, sf_align_coarse_progress(&align));

    set_diag3(align.p, sq(deg_to_rad(5.0f)), sq(deg_to_rad(5.0f)), sq(deg_to_rad(8.0f)));
    align.yaw_observed = true;
    align.coarse_aligned = true;
    TEST_ASSERT_TRUE(sf_align_coarse_alignment_ready(&align));
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, sf_align_coarse_progress(&align));

    align.p[1][1] = sq(deg_to_rad(6.0f));
    align.coarse_aligned = false;
    sf_align_sigma_deg(&align, sigma_deg);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, 6.0f, sigma_deg[1]);
    TEST_ASSERT_FALSE(sf_align_coarse_alignment_ready(&align));
}

static void test_horizontal_yaw_observation_reduces_yaw_sigma_and_reports_ready(void)
{
    sf_align_config_t cfg = sf_align_config_default();
    sf_align_t align;
    sf_align_update_trace_t trace;
    sf_align_window_summary_t window = straight_yaw_window();
    float sigma_before[3];
    float sigma_after[3];
    float angles_deg[3];
    cfg.q_mount_std_rad[0] = 0.0f;
    cfg.q_mount_std_rad[1] = 0.0f;
    cfg.q_mount_std_rad[2] = 0.0f;
    sf_align_init(&align, cfg);
    set_diag3(align.p, sq(deg_to_rad(4.0f)), sq(deg_to_rad(4.0f)), sq(deg_to_rad(60.0f)));
    sf_align_sigma_deg(&align, sigma_before);

    (void)sf_align_update_window_with_trace(&align, &window, &trace);
    sf_align_sigma_deg(&align, sigma_after);
    sf_align_mount_angles_deg(&align, angles_deg);

    TEST_ASSERT_TRUE(trace.horiz_accel_applied);
    TEST_ASSERT_TRUE(trace.horiz_straight_core_valid);
    TEST_ASSERT_TRUE(align.yaw_observed);
    TEST_ASSERT_TRUE(sigma_after[2] < sigma_before[2]);
    TEST_ASSERT_FALSE(sf_align_coarse_alignment_ready(&align));
    TEST_ASSERT_FLOAT_WITHIN(2.0e-5f, 36.33279f, sigma_after[2]);
    TEST_ASSERT_FLOAT_WITHIN(2.0e-5f, 56.99821f, angles_deg[2]);
    TEST_ASSERT_FLOAT_WITHIN(2.0e-6f, 0.618597f, sf_align_coarse_progress(&align));
    TEST_ASSERT_FLOAT_WITHIN(2.0e-5f, 45.65511f,
                             trace.horiz_effective_std_rad * 180.0f / TEST_PI);
}

static void test_turn_gyro_update_does_not_reduce_yaw_sigma(void)
{
    sf_align_config_t cfg = sf_align_config_default();
    sf_align_t align;
    sf_align_window_summary_t window = {0};
    sf_align_update_trace_t trace;
    float pzz_before;
    cfg.use_gravity = false;
    cfg.min_speed_mps = 0.1f;
    cfg.min_turn_rate_radps = 0.01f;
    cfg.min_lat_acc_mps2 = 0.01f;
    cfg.turn_consistency_min_windows = 1u;
    cfg.turn_consistency_min_fraction = 1.0f;
    cfg.turn_consistency_max_abs_lat_err_mps2 = 2.0f;
    cfg.turn_consistency_max_rel_lat_err = 2.0f;
    cfg.q_mount_std_rad[0] = 0.0f;
    cfg.q_mount_std_rad[1] = 0.0f;
    cfg.q_mount_std_rad[2] = 0.0f;
    sf_align_init(&align, cfg);
    align.p[2][2] = sq(deg_to_rad(30.0f));
    pzz_before = align.p[2][2];

    window.dt = 0.5f;
    window.mean_gyro_b[0] = 0.02f;
    window.mean_gyro_b[1] = -0.01f;
    window.mean_gyro_b[2] = 0.20f;
    window.mean_accel_b[2] = -SF_ALIGN_GRAVITY_MPS2;
    window.gnss_vel_prev_n[0] = 5.0f;
    window.gnss_vel_curr_n[0] = 4.975f;
    window.gnss_vel_curr_n[1] = 0.499f;
    window.imu_sample_count = 50u;

    (void)sf_align_update_window_with_trace(&align, &window, &trace);

    TEST_ASSERT_TRUE(trace.turn_gyro_applied);
    TEST_ASSERT_FALSE(trace.horiz_accel_applied);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-9f, pzz_before, align.p[2][2]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-9f, 0.0f, align.p[0][2]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-9f, 0.0f, align.p[1][2]);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_stationary_tilt_init_sets_identity_tilt_and_conservative_sigma);
    RUN_TEST(test_predict_adds_mount_random_walk_variance);
    RUN_TEST(test_coarse_progress_and_readiness_follow_axis_sigmas);
    RUN_TEST(test_horizontal_yaw_observation_reduces_yaw_sigma_and_reports_ready);
    RUN_TEST(test_turn_gyro_update_does_not_reduce_yaw_sigma);
    return UNITY_END();
}
