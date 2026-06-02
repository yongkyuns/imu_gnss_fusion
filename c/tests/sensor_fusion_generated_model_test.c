#include "ekf/generated_model.h"
#include "unity.h"

static void set_identity_covariance(float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES])
{
    for (int r = 0; r < SF_EKF_ERROR_STATES; ++r) {
        for (int c = 0; c < SF_EKF_ERROR_STATES; ++c) {
            p[r][c] = r == c ? 1.0f : 0.0f;
        }
    }
}

static void test_gnss_position_row_uses_position_error_slot(void)
{
    float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES];
    sf_ekf_scalar_observation_t obs;
    set_identity_covariance(p);

    sf_ekf_gps_pos_n_observation(p, 3.0f, &obs);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, obs.h[6]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 4.0f, obs.s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.25f, obs.k[6]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, obs.k[5]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, obs.h[3]);
}

static void test_gnss_velocity_row_uses_velocity_error_slot(void)
{
    float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES];
    sf_ekf_scalar_observation_t obs;
    set_identity_covariance(p);

    sf_ekf_gps_vel_n_observation(p, 1.0f, &obs);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, obs.h[3]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 2.0f, obs.s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.5f, obs.k[3]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, obs.k[6]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, obs.h[6]);
}

static void test_roll_prior_identity_row_is_roll_error(void)
{
    float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES];
    sf_ekf_scalar_observation_t obs;
    sf_ekf_nominal_state_t nominal = {0};
    set_identity_covariance(p);
    nominal.q0 = 1.0f;
    nominal.q_bv0 = 1.0f;

    sf_ekf_vehicle_roll_prior_observation(&nominal, p, 0.25f, &obs);

    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, obs.h[0]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.25f, obs.s);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.8f, obs.k[0]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, obs.h[1]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, obs.h[2]);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_gnss_position_row_uses_position_error_slot);
    RUN_TEST(test_gnss_velocity_row_uses_velocity_error_slot);
    RUN_TEST(test_roll_prior_identity_row_is_roll_error);
    return UNITY_END();
}
