#include "sensor_fusion.h"
#include "unity.h"

#include <math.h>
#include <string.h>

typedef struct replay_expected_snapshot {
    const char *label;
    uint32_t event_mask;
    float q_nv[4];
    float vel_ned_mps[3];
    float pos_ned_m[3];
    float p_diag[4];
} replay_expected_snapshot_t;

typedef struct replay_rich_expected_snapshot {
    const char *label;
    uint32_t event_mask;
    float q_nv[4];
    float vel_ned_mps[3];
    float pos_ned_m[3];
    float q_bv[4];
    float p_diag[6];
} replay_rich_expected_snapshot_t;

static sensor_fusion_gnss_sample_t replay_gnss(float t_s,
                                               double lat_deg,
                                               double lon_deg,
                                               float vel_n,
                                               float vel_e)
{
    sensor_fusion_gnss_sample_t sample;
    memset(&sample, 0, sizeof(sample));
    sample.t_s = t_s;
    sample.lat_deg = lat_deg;
    sample.lon_deg = lon_deg;
    sample.height_m = 10.0;
    sample.vel_ned_mps[0] = vel_n;
    sample.vel_ned_mps[1] = vel_e;
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

static sensor_fusion_gnss_sample_t replay_gnss_heading(float t_s,
                                                       double lat_deg,
                                                       double lon_deg,
                                                       float vel_n,
                                                       float vel_e,
                                                       float heading_rad)
{
    sensor_fusion_gnss_sample_t sample;
    memset(&sample, 0, sizeof(sample));
    sample.t_s = t_s;
    sample.lat_deg = lat_deg;
    sample.lon_deg = lon_deg;
    sample.height_m = 10.0;
    sample.vel_ned_mps[0] = vel_n;
    sample.vel_ned_mps[1] = vel_e;
    sample.vel_ned_mps[2] = 0.0f;
    sample.pos_std_m[0] = 1.2f;
    sample.pos_std_m[1] = 1.2f;
    sample.pos_std_m[2] = 2.5f;
    sample.vel_std_mps[0] = 0.15f;
    sample.vel_std_mps[1] = 0.15f;
    sample.vel_std_mps[2] = 0.2f;
    sample.has_heading_rad = true;
    sample.heading_rad = heading_rad;
    return sample;
}

static sensor_fusion_imu_sample_t replay_imu(float t_s)
{
    sensor_fusion_imu_sample_t sample;
    memset(&sample, 0, sizeof(sample));
    sample.t_s = t_s;
    sample.accel_mps2[2] = -9.80665f;
    return sample;
}

static sensor_fusion_imu_sample_t replay_motion_imu(float t_s,
                                                    float gyro_z,
                                                    float accel_x,
                                                    float accel_y)
{
    sensor_fusion_imu_sample_t sample;
    memset(&sample, 0, sizeof(sample));
    sample.t_s = t_s;
    sample.gyro_radps[2] = gyro_z;
    sample.accel_mps2[0] = accel_x;
    sample.accel_mps2[1] = accel_y;
    sample.accel_mps2[2] = -9.80665f;
    return sample;
}

static void assert_snapshot_close(const replay_expected_snapshot_t *expected,
                                  uint32_t event_mask,
                                  const sensor_fusion_ekf_state_t *actual)
{
    TEST_ASSERT_EQUAL_UINT32(expected->event_mask, event_mask);
    for (int i = 0; i < 4; ++i) {
        (void)expected->label;
        TEST_ASSERT_FLOAT_WITHIN(2.0e-3f, expected->q_nv[i], actual->q_nv[i]);
    }
    for (int i = 0; i < 3; ++i) {
        TEST_ASSERT_FLOAT_WITHIN(2.0e-2f, expected->vel_ned_mps[i], actual->vel_ned_mps[i]);
        TEST_ASSERT_FLOAT_WITHIN(2.0e-2f, expected->pos_ned_m[i], actual->pos_ned_m[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(2.0e-2f, expected->p_diag[0], actual->covariance[3][3]);
    TEST_ASSERT_FLOAT_WITHIN(2.0e-2f, expected->p_diag[1], actual->covariance[4][4]);
    TEST_ASSERT_FLOAT_WITHIN(5.0e-2f, expected->p_diag[2], actual->covariance[6][6]);
    TEST_ASSERT_FLOAT_WITHIN(5.0e-2f, expected->p_diag[3], actual->covariance[7][7]);
}

static void assert_rich_snapshot_close(const replay_rich_expected_snapshot_t *expected,
                                       uint32_t event_mask,
                                       const sensor_fusion_ekf_state_t *actual)
{
    TEST_ASSERT_EQUAL_UINT32(expected->event_mask, event_mask);
    for (int i = 0; i < 4; ++i) {
        (void)expected->label;
        TEST_ASSERT_FLOAT_WITHIN(3.0e-3f, expected->q_nv[i], actual->q_nv[i]);
        TEST_ASSERT_FLOAT_WITHIN(3.0e-3f, expected->q_bv[i], actual->q_bv[i]);
    }
    for (int i = 0; i < 3; ++i) {
        TEST_ASSERT_FLOAT_WITHIN(3.0e-2f, expected->vel_ned_mps[i], actual->vel_ned_mps[i]);
        TEST_ASSERT_FLOAT_WITHIN(4.0e-2f, expected->pos_ned_m[i], actual->pos_ned_m[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(3.0e-3f, expected->p_diag[0], actual->covariance[0][0]);
    TEST_ASSERT_FLOAT_WITHIN(3.0e-3f, expected->p_diag[1], actual->covariance[1][1]);
    TEST_ASSERT_FLOAT_WITHIN(4.0e-3f, expected->p_diag[2], actual->covariance[2][2]);
    TEST_ASSERT_FLOAT_WITHIN(3.0e-2f, expected->p_diag[3], actual->covariance[3][3]);
    TEST_ASSERT_FLOAT_WITHIN(6.0e-2f, expected->p_diag[4], actual->covariance[6][6]);
    TEST_ASSERT_FLOAT_WITHIN(4.0e-3f, expected->p_diag[5], actual->covariance[15][15]);
}

static void test_manual_mount_replay_matches_rust_golden_trace(void)
{
    static const replay_expected_snapshot_t expected_after_nhc = {
        "after_nhc",
        0u,
        {0.999980092f, -0.000058688f, 0.000000074f, 0.006306030f},
        {5.999991417f, 1.992163777f, -0.000392393f},
        {0.300000012f, 0.099616706f, 0.000000044f},
        {0.041025449f, 0.040865295f, 2.250099897f, 2.250099421f},
    };
    static const replay_expected_snapshot_t expected_after_pending_gnss = {
        "after_pending_gnss",
        0u,
        {0.999980211f, -0.000060151f, 0.000000107f, 0.006307141f},
        {5.999984741f, 1.991995454f, -0.000785584f},
        {0.599999547f, 0.199224889f, -0.000019575f},
        {0.044064973f, 0.043900173f, 2.250402451f, 2.250400782f},
    };
    static const replay_expected_snapshot_t expected_after_speed = {
        "after_speed",
        0u,
        {0.987362623f, -0.000195402f, 0.009665838f, -0.158182174f},
        {5.273637295f, 1.974522352f, -0.001256972f},
        {0.532394111f, 0.197578102f, -0.000066622f},
        {0.028256854f, 0.043891024f, 2.250265598f, 2.250400782f},
    };
    sensor_fusion_t fusion;
    sensor_fusion_ekf_state_t state;
    sensor_fusion_update_t update;
    float q_bv[4] = {1.0f, 0.0f, 0.0f, 0.0f};

    sensor_fusion_init_with_mount(&fusion, q_bv);
    sensor_fusion_set_r_vehicle_roll_prior(&fusion, 0.0f);
    sensor_fusion_set_r_body_vel_yz(&fusion, 0.5f, 0.5f);
    sensor_fusion_set_nhc_update_period_s(&fusion, 0.1f);

    (void)sensor_fusion_process_gnss(
        &fusion, replay_gnss(0.02f, 37.0, -122.0, 6.0f, 2.0f));
    (void)sensor_fusion_process_imu(&fusion, replay_imu(0.02f));
    update = sensor_fusion_process_imu(&fusion, replay_imu(0.07f));
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&fusion, &state));
    assert_snapshot_close(&expected_after_nhc, update.gnss_event_mask, &state);

    (void)sensor_fusion_process_imu(&fusion, replay_imu(0.12f));
    (void)sensor_fusion_process_gnss(
        &fusion, replay_gnss(0.16f, 37.000010, -122.0, 5.5f, 0.5f));
    update = sensor_fusion_process_imu(&fusion, replay_imu(0.17f));
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&fusion, &state));
    assert_snapshot_close(&expected_after_pending_gnss, update.gnss_event_mask, &state);

    update = sensor_fusion_process_vehicle_speed(
        &fusion,
        (sensor_fusion_vehicle_speed_sample_t){
            .t_s = 0.22f,
            .speed_mps = 4.0f,
            .direction = SENSOR_FUSION_VEHICLE_SPEED_FORWARD,
        });
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&fusion, &state));
    assert_snapshot_close(&expected_after_speed, update.gnss_event_mask, &state);
}

static void test_manual_mount_turn_replay_matches_rust_golden_trace(void)
{
    static const replay_rich_expected_snapshot_t expected_turn_mid = {
        "turn_mid",
        0u,
        {0.999535143f, 0.001954838f, -0.001385825f, 0.030394232f},
        {8.203193665f, 0.546857834f, 0.002524295f},
        {4.025900841f, 0.105060287f, -0.000090070f},
        {0.999985099f, -0.004853053f, 0.002519514f, 0.000037425f},
        {0.001030092f, 0.000980730f, 0.006799215f, 0.024202911f, 1.921372294f,
         0.001774401f},
    };
    static const replay_rich_expected_snapshot_t expected_after_speed = {
        "after_speed",
        0u,
        {0.998754799f, 0.002092058f, -0.000110330f, 0.049844448f},
        {8.124683380f, 0.849385440f, 0.003243601f},
        {5.645681381f, 0.239861116f, 0.000066231f},
        {0.999989092f, -0.004534845f, -0.001014996f, -0.000523726f},
        {0.001033475f, 0.000929204f, 0.005769151f, 0.027026674f, 1.922850966f,
         0.001765851f},
    };
    static const replay_rich_expected_snapshot_t expected_turn_final = {
        "turn_final",
        0u,
        {0.997052372f, 0.001982818f, -0.001095855f, 0.076690197f},
        {8.102236748f, 1.314721704f, 0.005472708f},
        {8.037650108f, 0.600261807f, 0.000521457f},
        {0.999988139f, -0.004835451f, 0.000649815f, -0.000183071f},
        {0.000940727f, 0.000857182f, 0.004650806f, 0.024986014f, 1.494204283f,
         0.001266527f},
    };
    sensor_fusion_t fusion;
    sensor_fusion_ekf_state_t state;
    sensor_fusion_update_t update;
    uint32_t last_mask = 0u;
    float q_bv[4] = {1.0f, 0.0f, 0.0f, 0.0f};

    sensor_fusion_init_with_mount(&fusion, q_bv);
    sensor_fusion_set_r_body_vel_yz(&fusion, 0.5f, 0.5f);
    sensor_fusion_set_r_vehicle_roll_prior(&fusion, 0.1f);
    sensor_fusion_set_r_vehicle_speed(&fusion, 0.04f);
    sensor_fusion_set_nhc_update_period_s(&fusion, 0.1f);

    update = sensor_fusion_process_gnss(
        &fusion, replay_gnss_heading(0.02f, 37.0, -122.0, 8.0f, 0.0f, 0.0f));
    last_mask = update.gnss_event_mask;
    for (int step = 0; step <= 40; ++step) {
        float t_s = 0.02f + (float)step * 0.025f;
        float gyro_z = step >= 8 ? 0.18f : 0.0f;
        float accel_y = step >= 8 ? 1.4f : 0.0f;
        float accel_x = step <= 18 ? 0.35f : 0.0f;
        update = sensor_fusion_process_imu(&fusion,
                                           replay_motion_imu(t_s, gyro_z, accel_x, accel_y));
        last_mask = update.gnss_event_mask;
        if (step == 16) {
            update = sensor_fusion_process_gnss(
                &fusion,
                replay_gnss_heading(0.41f, 37.000030, -121.999999, 8.2f, 0.45f, 0.055f));
            last_mask = update.gnss_event_mask;
        }
        if (step == 24) {
            update = sensor_fusion_process_vehicle_speed(
                &fusion,
                (sensor_fusion_vehicle_speed_sample_t){
                    .t_s = t_s,
                    .speed_mps = 8.1f,
                    .direction = SENSOR_FUSION_VEHICLE_SPEED_FORWARD,
                });
            last_mask = update.gnss_event_mask;
        }
        if (step == 32) {
            update = sensor_fusion_process_gnss(
                &fusion,
                replay_gnss_heading(0.81f, 37.000058, -121.999994, 8.15f, 1.05f, 0.128f));
            last_mask = update.gnss_event_mask;
        }
        if (step == 20) {
            TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&fusion, &state));
            assert_rich_snapshot_close(&expected_turn_mid, last_mask, &state);
        }
        if (step == 28) {
            TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&fusion, &state));
            assert_rich_snapshot_close(&expected_after_speed, last_mask, &state);
        }
    }
    TEST_ASSERT_TRUE(sensor_fusion_ekf_state(&fusion, &state));
    assert_rich_snapshot_close(&expected_turn_final, last_mask, &state);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_manual_mount_replay_matches_rust_golden_trace);
    RUN_TEST(test_manual_mount_turn_replay_matches_rust_golden_trace);
    return UNITY_END();
}
