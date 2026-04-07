#ifndef SF_ESKF_H
#define SF_ESKF_H

#include "sensor_fusion_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SF_ESKF_NOMINAL_STATES 16
#define SF_ESKF_ERROR_STATES 15
#define SF_ESKF_NOISE_STATES 12

typedef struct {
  float q0, q1, q2, q3;
  float vn, ve, vd;
  float pn, pe, pd;
  float bgx, bgy, bgz;
  float bax, bay, baz;
} sf_eskf_nominal_state_t;

typedef struct {
  float dtheta_x, dtheta_y, dtheta_z;
  float dv_n, dv_e, dv_d;
  float dp_n, dp_e, dp_d;
  float dbg_x, dbg_y, dbg_z;
  float dba_x, dba_y, dba_z;
} sf_eskf_error_state_t;

typedef struct {
  float innovation_x;
  float innovation_y;
  float k_theta_x_from_x;
  float k_theta_y_from_x;
  float k_bax_from_x;
  float k_bay_from_x;
  float k_theta_x_from_y;
  float k_theta_y_from_y;
  float k_bax_from_y;
  float k_bay_from_y;
  float p_theta_x;
  float p_theta_y;
  float p_bax;
  float p_bay;
  float p_theta_x_bax;
  float p_theta_y_bay;
  unsigned int updates;
} sf_eskf_stationary_diag_t;

typedef struct sf_eskf {
  sf_eskf_nominal_state_t nominal;
  float p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES];
  sf_predict_noise_t noise;
  sf_eskf_stationary_diag_t stationary_diag;
} sf_eskf_t;

typedef struct {
  float dax;
  float day;
  float daz;
  float dvx;
  float dvy;
  float dvz;
  float dt;
} sf_eskf_imu_delta_t;

void sf_predict_noise_default(sf_predict_noise_t *cfg);

void sf_eskf_init(sf_eskf_t *eskf,
                  const float p_diag[SF_ESKF_ERROR_STATES],
                  const sf_predict_noise_t *noise);

void sf_eskf_predict(sf_eskf_t *eskf, const sf_eskf_imu_delta_t *imu);

void sf_eskf_predict_nominal(sf_eskf_t *eskf, const sf_eskf_imu_delta_t *imu);
void sf_eskf_fuse_gps(sf_eskf_t *eskf, const sf_gnss_ned_sample_t *gps);
void sf_eskf_fuse_body_speed_x(sf_eskf_t *eskf, float speed_mps, float r_speed);
void sf_eskf_fuse_body_vel(sf_eskf_t *eskf, float r_body_vel);
void sf_eskf_fuse_zero_vel(sf_eskf_t *eskf, float r_zero_vel);
void sf_eskf_fuse_stationary_gravity(sf_eskf_t *eskf,
                                     const float accel_body_mps2[3],
                                     float r_stationary_accel);

void sf_eskf_compute_error_transition(float f_out[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
                                      float g_out[SF_ESKF_ERROR_STATES][SF_ESKF_NOISE_STATES],
                                      const sf_eskf_t *eskf,
                                      const sf_eskf_imu_delta_t *imu);

#ifdef __cplusplus
}
#endif

#endif
