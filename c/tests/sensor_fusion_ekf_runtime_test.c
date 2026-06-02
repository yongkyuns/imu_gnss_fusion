#include "ekf/runtime.h"
#include "unity.h"

#include <math.h>

static float quat_norm(const float q[4])
{
    return sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
}

static void test_inject_error_state_uses_expected_quaternion_sides(void)
{
    sf_ekf_nominal_state_t nominal = {0};
    float dx[SF_EKF_ERROR_STATES] = {0};
    nominal.q0 = 1.0f;
    nominal.q_bv0 = 1.0f;

    dx[0] = 0.02f;
    dx[1] = -0.04f;
    dx[2] = 0.06f;
    dx[3] = 1.0f;
    dx[4] = -2.0f;
    dx[5] = 3.0f;
    dx[15] = -0.10f;
    dx[16] = 0.08f;
    dx[17] = -0.06f;

    sf_ekf_inject_error_state(&nominal, dx);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.0f, quat_norm(&nominal.q0));
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.0f, quat_norm(&nominal.q_bv0));
    TEST_ASSERT_TRUE(nominal.q1 > 0.0f);
    TEST_ASSERT_TRUE(nominal.q2 < 0.0f);
    TEST_ASSERT_TRUE(nominal.q3 > 0.0f);
    TEST_ASSERT_TRUE(nominal.q_bv1 < 0.0f);
    TEST_ASSERT_TRUE(nominal.q_bv2 > 0.0f);
    TEST_ASSERT_TRUE(nominal.q_bv3 < 0.0f);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, nominal.vn);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, -2.0f, nominal.ve);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 3.0f, nominal.vd);
}

static void test_generated_nominal_prediction_identity_fixture(void)
{
    sf_ekf_runtime_state_t state;
    sf_ekf_imu_delta_t imu = {0};
    sf_ekf_runtime_init(&state);
    state.nominal.vn = 2.0f;
    state.nominal.ve = -3.0f;
    state.nominal.vd = 0.5f;
    state.nominal.pn = 10.0f;
    state.nominal.pe = -20.0f;
    state.nominal.pd = 4.0f;
    imu.dvx = 1.0f;
    imu.dt = 0.1f;

    sf_ekf_runtime_predict(&state, &imu);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 3.0f, state.nominal.vn);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, -3.0f, state.nominal.ve);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.480665f, state.nominal.vd);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 10.2f, state.nominal.pn);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, -20.3f, state.nominal.pe);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 4.05f, state.nominal.pd);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.0f, quat_norm(&state.nominal.q0));
}

static void test_covariance_prediction_stays_symmetric_and_positive_diagonal(void)
{
    sf_ekf_runtime_state_t state;
    sf_ekf_imu_delta_t imu = {0};
    sf_ekf_runtime_init(&state);
    for (int i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        state.p[i][i] = 0.1f + (float)i * 0.01f;
    }
    imu.dax = 0.001f;
    imu.day = -0.002f;
    imu.daz = 0.003f;
    imu.dvx = 0.02f;
    imu.dvy = -0.03f;
    imu.dvz = 0.01f;
    imu.dt = 0.01f;

    sf_ekf_runtime_predict(&state, &imu);

    for (int i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        TEST_ASSERT_TRUE(state.p[i][i] >= 0.0f);
        for (int j = i + 1; j < SF_EKF_ERROR_STATES; ++j) {
            TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, state.p[i][j], state.p[j][i]);
        }
    }
}

static void test_scalar_joseph_update_and_injection(void)
{
    sf_ekf_runtime_state_t state;
    float h[SF_EKF_ERROR_STATES] = {0};
    float k[SF_EKF_ERROR_STATES] = {0};
    float dx[SF_EKF_ERROR_STATES] = {0};
    sf_ekf_runtime_init(&state);
    state.p[6][6] = 1.0f;
    h[6] = 1.0f;
    k[6] = 0.25f;
    dx[6] = 2.0f;

    sf_ekf_runtime_fuse_scalar(&state, 4.0f, h, k, dx);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 2.0f, state.nominal.pn);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.75f, state.p[6][6]);
}

static void test_attitude_reset_preserves_covariance_symmetry(void)
{
    sf_ekf_runtime_state_t state;
    float dx[SF_EKF_ERROR_STATES] = {0};
    sf_ekf_runtime_init(&state);
    state.p[0][0] = 0.20f;
    state.p[1][1] = 0.30f;
    state.p[2][2] = 0.40f;
    state.p[0][3] = 0.01f;
    state.p[3][0] = 0.01f;
    dx[0] = 0.03f;
    dx[1] = -0.02f;
    dx[2] = 0.01f;

    sf_ekf_apply_reset(state.p, dx);

    for (int i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        TEST_ASSERT_TRUE(state.p[i][i] >= 0.0f);
        for (int j = i + 1; j < SF_EKF_ERROR_STATES; ++j) {
            TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, state.p[i][j], state.p[j][i]);
        }
    }
}

static void test_gnss_scalar_wrappers_correct_position_and_velocity(void)
{
    sf_ekf_runtime_state_t state;
    sf_ekf_runtime_init(&state);
    state.nominal.pn = 10.0f;
    state.nominal.ve = -3.0f;
    state.p[6][6] = 1.0f;
    state.p[4][4] = 4.0f;

    sf_ekf_runtime_fuse_gps_pos_n(&state, 14.0f, 3.0f);
    sf_ekf_runtime_fuse_gps_vel_e(&state, 1.0f, 4.0f);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 11.0f, state.nominal.pn);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, -1.0f, state.nominal.ve);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.75f, state.p[6][6]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 2.0f, state.p[4][4]);
}

static void test_batch_gnss_matches_independent_scalar_rows(void)
{
    sf_ekf_runtime_state_t scalar;
    sf_ekf_runtime_state_t batch;
    sf_ekf_gnss_ned_sample_t sample = {0};
    sf_ekf_runtime_init(&scalar);
    sf_ekf_runtime_init(&batch);
    scalar.nominal.pn = batch.nominal.pn = 10.0f;
    scalar.nominal.pe = batch.nominal.pe = -5.0f;
    scalar.nominal.pd = batch.nominal.pd = 2.0f;
    scalar.nominal.vn = batch.nominal.vn = 3.0f;
    scalar.nominal.ve = batch.nominal.ve = -2.0f;
    scalar.nominal.vd = batch.nominal.vd = 1.0f;
    for (int i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        scalar.p[i][i] = batch.p[i][i] = 1.0f + (float)i * 0.1f;
    }
    sample.pos_ned_m[0] = 14.0f;
    sample.pos_ned_m[1] = -1.0f;
    sample.pos_ned_m[2] = 0.0f;
    sample.vel_ned_mps[0] = 5.0f;
    sample.vel_ned_mps[1] = -4.0f;
    sample.vel_ned_mps[2] = 2.0f;
    sample.pos_std_m[0] = sample.pos_std_m[1] = sample.pos_std_m[2] = 2.0f;
    sample.vel_std_mps[0] = sample.vel_std_mps[1] = sample.vel_std_mps[2] = 1.0f;

    sf_ekf_runtime_fuse_gps_pos_n(&scalar, sample.pos_ned_m[0], 4.0f);
    sf_ekf_runtime_fuse_gps_pos_e(&scalar, sample.pos_ned_m[1], 4.0f);
    sf_ekf_runtime_fuse_gps_pos_d(&scalar, sample.pos_ned_m[2], 4.0f);
    sf_ekf_runtime_fuse_gps_vel_n(&scalar, sample.vel_ned_mps[0], 1.0f);
    sf_ekf_runtime_fuse_gps_vel_e(&scalar, sample.vel_ned_mps[1], 1.0f);
    sf_ekf_runtime_fuse_gps_vel_d(&scalar, sample.vel_ned_mps[2], 1.0f);
    sf_ekf_runtime_fuse_gps_nhc_batch_no_gate(&batch, &sample, 0, 0.0f, 0, 0.0f);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, scalar.nominal.pn, batch.nominal.pn);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, scalar.nominal.pe, batch.nominal.pe);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, scalar.nominal.pd, batch.nominal.pd);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, scalar.nominal.vn, batch.nominal.vn);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, scalar.nominal.ve, batch.nominal.ve);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, scalar.nominal.vd, batch.nominal.vd);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, scalar.p[6][6], batch.p[6][6]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, scalar.p[4][4], batch.p[4][4]);
}

static void test_body_velocity_batch_reduces_lateral_velocity(void)
{
    sf_ekf_runtime_state_t state;
    sf_ekf_runtime_init(&state);
    state.nominal.q0 = 1.0f;
    state.nominal.q_bv0 = 1.0f;
    state.nominal.ve = 4.0f;
    for (int i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        state.p[i][i] = 1.0f;
    }

    sf_ekf_runtime_fuse_body_vel_yz(&state, 1.0f, 1.0f);

    TEST_ASSERT_TRUE(fabsf(state.nominal.ve) < 4.0f);
    for (int i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        TEST_ASSERT_TRUE(state.p[i][i] >= 0.0f);
        for (int j = i + 1; j < SF_EKF_ERROR_STATES; ++j) {
            TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, state.p[i][j], state.p[j][i]);
        }
    }
}

static void test_body_speed_x_reduces_forward_velocity_residual(void)
{
    sf_ekf_runtime_state_t state;
    sf_ekf_runtime_init(&state);
    state.nominal.q0 = 1.0f;
    state.nominal.q_bv0 = 1.0f;
    state.nominal.vn = 8.0f;
    for (int i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        state.p[i][i] = 1.0f;
    }

    sf_ekf_runtime_fuse_body_speed_x(&state, 4.0f, 1.0f);

    TEST_ASSERT_TRUE(state.nominal.vn < 8.0f);
    TEST_ASSERT_TRUE(state.nominal.vn > 4.0f);
}

static void test_vehicle_roll_prior_reduces_vehicle_roll(void)
{
    sf_ekf_runtime_state_t state;
    float roll = 5.0f * 3.14159265358979323846f / 180.0f;
    sf_ekf_runtime_init(&state);
    state.nominal.q0 = cosf(0.5f * roll);
    state.nominal.q1 = sinf(0.5f * roll);
    state.nominal.q_bv0 = 1.0f;
    for (int i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        state.p[i][i] = 0.1f;
    }

    sf_ekf_runtime_fuse_vehicle_roll_prior(&state, 0.01f);

    TEST_ASSERT_TRUE(fabsf(state.nominal.q1) < sinf(0.5f * roll));
}

static sf_ekf_gnss_ned_sample_t gate_sample(float t_s, float pos_n, float vel_n, float std)
{
    sf_ekf_gnss_ned_sample_t sample = {0};
    sample.t_s = t_s;
    sample.pos_ned_m[0] = pos_n;
    sample.vel_ned_mps[0] = vel_n;
    sample.pos_std_m[0] = sample.pos_std_m[1] = sample.pos_std_m[2] = std;
    sample.vel_std_mps[0] = sample.vel_std_mps[1] = sample.vel_std_mps[2] = std;
    return sample;
}

static void init_gated_state(sf_ekf_runtime_state_t *state)
{
    sf_ekf_runtime_init(state);
    state->nominal.pn = 0.0f;
    state->nominal.vn = 0.0f;
    for (int i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        state->p[i][i] = 1.0f;
    }
    sf_ekf_runtime_set_gnss_position_outlier_gate_sigma(state, 2.0f);
    sf_ekf_runtime_set_gnss_velocity_outlier_gate_sigma(state, 2.0f);
}

static void test_gated_gnss_batch_rejects_large_position_jump(void)
{
    sf_ekf_runtime_state_t state;
    sf_ekf_gnss_ned_sample_t sample;
    sf_ekf_gnss_update_result_t result;
    init_gated_state(&state);
    sample = gate_sample(1.0f, 20.0f, 0.0f, 1.0f);

    result = sf_ekf_runtime_fuse_gps_nhc_batch(&state, &sample, 0, 0.0f, 0, 0.0f);

    TEST_ASSERT_FALSE(result.accepted_position);
    TEST_ASSERT_TRUE(result.accepted_velocity);
    TEST_ASSERT_TRUE((result.event_mask & SF_EKF_GNSS_EVENT_POSITION_REJECTED) != 0u);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, state.nominal.pn);
}

static void test_gated_gnss_batch_marks_consecutive_rejections(void)
{
    sf_ekf_runtime_state_t state;
    sf_ekf_gnss_update_result_t result = {0};
    init_gated_state(&state);

    for (int i = 0; i < 3; ++i) {
        sf_ekf_gnss_ned_sample_t sample = gate_sample(1.0f + (float)i, 20.0f, 0.0f, 1.0f);
        result = sf_ekf_runtime_fuse_gps_nhc_batch(&state, &sample, 0, 0.0f, 0, 0.0f);
    }

    TEST_ASSERT_FALSE(result.accepted_position);
    TEST_ASSERT_TRUE((result.event_mask & SF_EKF_GNSS_EVENT_POSITION_CONSECUTIVE_REJECTED) != 0u);
}

static void test_gated_gnss_batch_uses_gap_bypass_unless_required_gate_pass(void)
{
    sf_ekf_runtime_state_t state;
    sf_ekf_gnss_ned_sample_t sample;
    sf_ekf_gnss_update_result_t result;
    init_gated_state(&state);

    sample = gate_sample(1.0f, 20.0f, 0.0f, 1.0f);
    (void)sf_ekf_runtime_fuse_gps_nhc_batch(&state, &sample, 0, 0.0f, 0, 0.0f);
    sample = gate_sample(5.1f, 20.0f, 0.0f, 1.0f);
    result = sf_ekf_runtime_fuse_gps_nhc_batch(&state, &sample, 0, 0.0f, 0, 0.0f);
    TEST_ASSERT_TRUE(result.accepted_position);
    TEST_ASSERT_TRUE((result.event_mask & SF_EKF_GNSS_EVENT_POSITION_GAP_BYPASS) != 0u);

    init_gated_state(&state);
    sample = gate_sample(1.0f, 20.0f, 0.0f, 1.0f);
    (void)sf_ekf_runtime_fuse_gps_nhc_batch(&state, &sample, 0, 0.0f, 0, 0.0f);
    sf_ekf_runtime_require_next_gnss_gate_pass(&state);
    sample = gate_sample(5.1f, 20.0f, 0.0f, 1.0f);
    result = sf_ekf_runtime_fuse_gps_nhc_batch(&state, &sample, 0, 0.0f, 0, 0.0f);
    TEST_ASSERT_FALSE(result.accepted_position);
    TEST_ASSERT_TRUE((result.event_mask & SF_EKF_GNSS_EVENT_POSITION_REJECTED) != 0u);
}

static void test_gated_gnss_batch_uses_accuracy_bypass(void)
{
    sf_ekf_runtime_state_t state;
    sf_ekf_gnss_ned_sample_t sample;
    sf_ekf_gnss_update_result_t result;
    init_gated_state(&state);

    sample = gate_sample(1.0f, 20.0f, 0.0f, 4.0f);
    (void)sf_ekf_runtime_fuse_gps_nhc_batch(&state, &sample, 0, 0.0f, 0, 0.0f);
    sample = gate_sample(2.0f, 20.0f, 0.0f, 1.0f);
    result = sf_ekf_runtime_fuse_gps_nhc_batch(&state, &sample, 0, 0.0f, 0, 0.0f);

    TEST_ASSERT_TRUE(result.accepted_position);
    TEST_ASSERT_TRUE((result.event_mask & SF_EKF_GNSS_EVENT_POSITION_ACCURACY_BYPASS) != 0u);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_inject_error_state_uses_expected_quaternion_sides);
    RUN_TEST(test_generated_nominal_prediction_identity_fixture);
    RUN_TEST(test_covariance_prediction_stays_symmetric_and_positive_diagonal);
    RUN_TEST(test_scalar_joseph_update_and_injection);
    RUN_TEST(test_attitude_reset_preserves_covariance_symmetry);
    RUN_TEST(test_gnss_scalar_wrappers_correct_position_and_velocity);
    RUN_TEST(test_batch_gnss_matches_independent_scalar_rows);
    RUN_TEST(test_body_velocity_batch_reduces_lateral_velocity);
    RUN_TEST(test_body_speed_x_reduces_forward_velocity_residual);
    RUN_TEST(test_vehicle_roll_prior_reduces_vehicle_roll);
    RUN_TEST(test_gated_gnss_batch_rejects_large_position_jump);
    RUN_TEST(test_gated_gnss_batch_marks_consecutive_rejections);
    RUN_TEST(test_gated_gnss_batch_uses_gap_bypass_unless_required_gate_pass);
    RUN_TEST(test_gated_gnss_batch_uses_accuracy_bypass);
    return UNITY_END();
}
