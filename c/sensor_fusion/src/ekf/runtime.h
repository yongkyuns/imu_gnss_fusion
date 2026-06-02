#ifndef SF_EKF_RUNTIME_H
#define SF_EKF_RUNTIME_H

#include "generated_model.h"

#include <stdint.h>

#define SF_EKF_MAX_BATCH_OBS 8
#define SF_EKF_GNSS_OUTLIER_GATE_SIGMA 25.0f
#define SF_EKF_GNSS_OUTLIER_GAP_BYPASS_S 3.0f
#define SF_EKF_GNSS_OUTLIER_CONSECUTIVE_REJECTION_EVENT_COUNT 3u
#define SF_EKF_GNSS_OUTLIER_ACCURACY_IMPROVEMENT_RATIO 0.5f

#define SF_EKF_GNSS_EVENT_POSITION_REJECTED (1u << 0)
#define SF_EKF_GNSS_EVENT_VELOCITY_REJECTED (1u << 1)
#define SF_EKF_GNSS_EVENT_POSITION_CONSECUTIVE_REJECTED (1u << 2)
#define SF_EKF_GNSS_EVENT_VELOCITY_CONSECUTIVE_REJECTED (1u << 3)
#define SF_EKF_GNSS_EVENT_POSITION_GAP_BYPASS (1u << 4)
#define SF_EKF_GNSS_EVENT_VELOCITY_GAP_BYPASS (1u << 5)
#define SF_EKF_GNSS_EVENT_POSITION_ACCURACY_BYPASS (1u << 6)
#define SF_EKF_GNSS_EVENT_VELOCITY_ACCURACY_BYPASS (1u << 7)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sf_ekf_process_noise {
    float gyro_var;
    float accel_var;
    float gyro_bias_rw_var;
    float accel_bias_rw_var;
    float mount_align_rw_var_axes[3];
} sf_ekf_process_noise_t;

typedef struct sf_ekf_gnss_gate_state {
    uint8_t consecutive_rejections;
    int has_last_t_s;
    float last_t_s;
    int has_last_accuracy_rms;
    float last_accuracy_rms;
    int require_next_gate_pass;
} sf_ekf_gnss_gate_state_t;

typedef struct sf_ekf_runtime_state {
    sf_ekf_nominal_state_t nominal;
    float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES];
    sf_ekf_process_noise_t noise;
    float gravity_mss;
    int freeze_misalignment_states;
    float gnss_position_outlier_gate_sigma;
    float gnss_velocity_outlier_gate_sigma;
    sf_ekf_gnss_gate_state_t gnss_position_gate_state;
    sf_ekf_gnss_gate_state_t gnss_velocity_gate_state;
} sf_ekf_runtime_state_t;

typedef struct sf_ekf_gnss_ned_sample {
    float t_s;
    float pos_ned_m[3];
    float vel_ned_mps[3];
    float pos_std_m[3];
    float vel_std_mps[3];
} sf_ekf_gnss_ned_sample_t;

typedef struct sf_ekf_gnss_update_result {
    int accepted_position;
    int accepted_velocity;
    uint32_t event_mask;
} sf_ekf_gnss_update_result_t;

sf_ekf_process_noise_t sf_ekf_process_noise_default(void);
void sf_ekf_runtime_init(sf_ekf_runtime_state_t *state);
void sf_ekf_normalize_quat(float q[4]);
void sf_ekf_quat_multiply(const float p[4], const float q[4], float out[4]);
void sf_ekf_inject_error_state(sf_ekf_nominal_state_t *nominal,
                               const float dx[SF_EKF_ERROR_STATES]);
void sf_ekf_runtime_predict(sf_ekf_runtime_state_t *state, const sf_ekf_imu_delta_t *imu);
void sf_ekf_runtime_predict_covariance(
    sf_ekf_runtime_state_t *state,
    const sf_ekf_error_transition_t *transition,
    const sf_ekf_imu_delta_t *imu);
void sf_ekf_update_covariance_joseph_scalar(
    float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    float innovation_var,
    const float h[SF_EKF_ERROR_STATES],
    const float k[SF_EKF_ERROR_STATES]);
void sf_ekf_apply_reset(float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
                        const float dx[SF_EKF_ERROR_STATES]);
void sf_ekf_runtime_fuse_scalar(sf_ekf_runtime_state_t *state,
                                float innovation_var,
                                const float h[SF_EKF_ERROR_STATES],
                                const float k[SF_EKF_ERROR_STATES],
                                const float dx[SF_EKF_ERROR_STATES]);
void sf_ekf_runtime_fuse_batch(sf_ekf_runtime_state_t *state,
                               unsigned int obs_count,
                               const float h_rows[SF_EKF_MAX_BATCH_OBS][SF_EKF_ERROR_STATES],
                               const float residuals[SF_EKF_MAX_BATCH_OBS],
                               const float variances[SF_EKF_MAX_BATCH_OBS]);
void sf_ekf_runtime_fuse_gps_pos_n(sf_ekf_runtime_state_t *state, float pos_n, float variance);
void sf_ekf_runtime_fuse_gps_pos_e(sf_ekf_runtime_state_t *state, float pos_e, float variance);
void sf_ekf_runtime_fuse_gps_pos_d(sf_ekf_runtime_state_t *state, float pos_d, float variance);
void sf_ekf_runtime_fuse_gps_vel_n(sf_ekf_runtime_state_t *state, float vel_n, float variance);
void sf_ekf_runtime_fuse_gps_vel_e(sf_ekf_runtime_state_t *state, float vel_e, float variance);
void sf_ekf_runtime_fuse_gps_vel_d(sf_ekf_runtime_state_t *state, float vel_d, float variance);
void sf_ekf_nominal_vehicle_velocity(const sf_ekf_nominal_state_t *nominal, float out_v_vehicle[3]);
void sf_ekf_runtime_fuse_body_speed_x(sf_ekf_runtime_state_t *state,
                                      float speed_mps,
                                      float r_speed);
void sf_ekf_runtime_fuse_zero_vel(sf_ekf_runtime_state_t *state, float r_zero_vel);
void sf_ekf_runtime_fuse_vehicle_roll_prior(sf_ekf_runtime_state_t *state,
                                            float r_vehicle_roll);
void sf_ekf_runtime_set_gnss_position_outlier_gate_sigma(sf_ekf_runtime_state_t *state,
                                                         float gate_sigma);
void sf_ekf_runtime_set_gnss_velocity_outlier_gate_sigma(sf_ekf_runtime_state_t *state,
                                                         float gate_sigma);
void sf_ekf_runtime_require_next_gnss_gate_pass(sf_ekf_runtime_state_t *state);
void sf_ekf_runtime_fuse_body_vel_yz(sf_ekf_runtime_state_t *state,
                                     float r_body_vel_y,
                                     float r_body_vel_z);
sf_ekf_gnss_update_result_t sf_ekf_runtime_fuse_gps_nhc_batch(
    sf_ekf_runtime_state_t *state,
    const sf_ekf_gnss_ned_sample_t *sample,
    int use_body_vel_y,
    float r_body_vel_y,
    int use_body_vel_z,
    float r_body_vel_z);
void sf_ekf_runtime_fuse_gps_nhc_batch_no_gate(sf_ekf_runtime_state_t *state,
                                               const sf_ekf_gnss_ned_sample_t *sample,
                                               int use_body_vel_y,
                                               float r_body_vel_y,
                                               int use_body_vel_z,
                                               float r_body_vel_z);
void sf_ekf_covariance_symmetrize(float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES]);

#ifdef __cplusplus
}
#endif

#endif
