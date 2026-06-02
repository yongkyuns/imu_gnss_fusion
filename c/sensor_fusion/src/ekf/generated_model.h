#ifndef SF_EKF_GENERATED_MODEL_H
#define SF_EKF_GENERATED_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#define SF_EKF_ERROR_STATES 18
#define SF_EKF_NOISE_STATES 15
#define SF_EKF_GRAVITY_MSS 9.80665F
#define SF_EKF_F_MAX_ROW_NONZERO 10
#define SF_EKF_G_MAX_ROW_NONZERO 3

typedef struct sf_ekf_nominal_state {
    float q0;
    float q1;
    float q2;
    float q3;
    float vn;
    float ve;
    float vd;
    float pn;
    float pe;
    float pd;
    float bgx;
    float bgy;
    float bgz;
    float bax;
    float bay;
    float baz;
    float q_bv0;
    float q_bv1;
    float q_bv2;
    float q_bv3;
} sf_ekf_nominal_state_t;

typedef struct sf_ekf_imu_delta {
    float dax;
    float day;
    float daz;
    float dvx;
    float dvy;
    float dvz;
    float dt;
} sf_ekf_imu_delta_t;

typedef struct sf_ekf_error_transition {
    float f[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES];
    float g[SF_EKF_ERROR_STATES][SF_EKF_NOISE_STATES];
} sf_ekf_error_transition_t;

typedef struct sf_ekf_scalar_observation {
    float h[SF_EKF_ERROR_STATES];
    float k[SF_EKF_ERROR_STATES];
    float s;
} sf_ekf_scalar_observation_t;

extern const unsigned int SF_EKF_F_ROW_COUNTS[SF_EKF_ERROR_STATES];
extern const unsigned int SF_EKF_F_ROW_COLS[SF_EKF_ERROR_STATES][SF_EKF_F_MAX_ROW_NONZERO];
extern const unsigned int SF_EKF_G_ROW_COUNTS[SF_EKF_ERROR_STATES];
extern const unsigned int SF_EKF_G_ROW_COLS[SF_EKF_ERROR_STATES][SF_EKF_G_MAX_ROW_NONZERO];

void sf_ekf_predict_nominal(sf_ekf_nominal_state_t *nominal, const sf_ekf_imu_delta_t *imu);
void sf_ekf_predict_nominal_with_gravity(
    sf_ekf_nominal_state_t *nominal,
    const sf_ekf_imu_delta_t *imu,
    float gravity_mss);
void sf_ekf_error_transition(
    const sf_ekf_nominal_state_t *nominal,
    const sf_ekf_imu_delta_t *imu,
    sf_ekf_error_transition_t *out);
void sf_ekf_error_transition_with_gravity(
    const sf_ekf_nominal_state_t *nominal,
    const sf_ekf_imu_delta_t *imu,
    float gravity_mss,
    sf_ekf_error_transition_t *out);
void sf_ekf_attitude_reset_jacobian(const float dtheta[3], float g_reset_theta[3][3]);
void sf_ekf_gps_pos_n_observation(
    const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    float r_pos_n,
    sf_ekf_scalar_observation_t *out);
void sf_ekf_gps_pos_e_observation(
    const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    float r_pos_e,
    sf_ekf_scalar_observation_t *out);
void sf_ekf_gps_pos_d_observation(
    const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    float r_pos_d,
    sf_ekf_scalar_observation_t *out);
void sf_ekf_gps_vel_n_observation(
    const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    float r_vel_n,
    sf_ekf_scalar_observation_t *out);
void sf_ekf_gps_vel_e_observation(
    const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    float r_vel_e,
    sf_ekf_scalar_observation_t *out);
void sf_ekf_gps_vel_d_observation(
    const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    float r_vel_d,
    sf_ekf_scalar_observation_t *out);
void sf_ekf_body_vel_x_observation(
    const sf_ekf_nominal_state_t *nominal,
    const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    float r_body_vel,
    sf_ekf_scalar_observation_t *out);
void sf_ekf_body_vel_y_observation(
    const sf_ekf_nominal_state_t *nominal,
    const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    float r_body_vel,
    sf_ekf_scalar_observation_t *out);
void sf_ekf_body_vel_z_observation(
    const sf_ekf_nominal_state_t *nominal,
    const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    float r_body_vel,
    sf_ekf_scalar_observation_t *out);
void sf_ekf_vehicle_roll_prior_observation(
    const sf_ekf_nominal_state_t *nominal,
    const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    float r_vehicle_roll,
    sf_ekf_scalar_observation_t *out);

#ifdef __cplusplus
}
#endif

#endif
