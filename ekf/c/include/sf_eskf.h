#ifndef SF_ESKF_H
#define SF_ESKF_H

#include "sensor_fusion_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SF_ESKF_NOMINAL_STATES 20
#define SF_ESKF_ERROR_STATES 18
#define SF_ESKF_NOISE_STATES 15
#define SF_ESKF_UPDATE_DIAG_TYPES 11

typedef enum {
  SF_ESKF_UPDATE_DIAG_GPS_POS = 0,
  SF_ESKF_UPDATE_DIAG_GPS_VEL = 1,
  SF_ESKF_UPDATE_DIAG_ZERO_VEL = 2,
  SF_ESKF_UPDATE_DIAG_BODY_SPEED_X = 3,
  SF_ESKF_UPDATE_DIAG_BODY_VEL_Y = 4,
  SF_ESKF_UPDATE_DIAG_BODY_VEL_Z = 5,
  SF_ESKF_UPDATE_DIAG_STATIONARY_X = 6,
  SF_ESKF_UPDATE_DIAG_STATIONARY_Y = 7,
  SF_ESKF_UPDATE_DIAG_GPS_POS_D = 8,
  SF_ESKF_UPDATE_DIAG_GPS_VEL_D = 9,
  SF_ESKF_UPDATE_DIAG_ZERO_VEL_D = 10,
} sf_eskf_update_diag_type_t;

typedef struct {
  float q0, q1, q2, q3;
  float vn, ve, vd;
  float pn, pe, pd;
  float bgx, bgy, bgz;
  float bax, bay, baz;
  float qcs0, qcs1, qcs2, qcs3;
} sf_eskf_nominal_state_t;

typedef struct {
  float dtheta_x, dtheta_y, dtheta_z;
  float dv_n, dv_e, dv_d;
  float dp_n, dp_e, dp_d;
  float dbg_x, dbg_y, dbg_z;
  float dba_x, dba_y, dba_z;
  float dpsi_cs_x, dpsi_cs_y, dpsi_cs_z;
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

typedef struct {
  unsigned int total_updates;
  unsigned int type_counts[SF_ESKF_UPDATE_DIAG_TYPES];
  float sum_dx_pitch[SF_ESKF_UPDATE_DIAG_TYPES];
  float sum_abs_dx_pitch[SF_ESKF_UPDATE_DIAG_TYPES];
  float sum_dx_mount_yaw[SF_ESKF_UPDATE_DIAG_TYPES];
  float sum_abs_dx_mount_yaw[SF_ESKF_UPDATE_DIAG_TYPES];
  float sum_innovation[SF_ESKF_UPDATE_DIAG_TYPES];
  float sum_abs_innovation[SF_ESKF_UPDATE_DIAG_TYPES];
  float last_dx_mount_yaw;
  float last_k_mount_yaw;
  float last_innovation;
  float last_innovation_var;
  unsigned int last_type;
} sf_eskf_update_diag_t;

typedef struct sf_eskf {
  sf_eskf_nominal_state_t nominal;
  float p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES];
  sf_predict_noise_t noise;
  sf_eskf_stationary_diag_t stationary_diag;
  sf_eskf_update_diag_t update_diag;
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
void sf_eskf_fuse_gps_scaled(sf_eskf_t *eskf, const sf_gnss_ned_sample_t *gps,
                             float gnss_pos_mount_scale,
                             float gnss_vel_mount_scale);
void sf_eskf_fuse_body_speed_x(sf_eskf_t *eskf, float speed_mps, float r_speed);
void sf_eskf_fuse_body_speed_x_scaled(sf_eskf_t *eskf, float speed_mps,
                                      float r_speed, float mount_update_scale,
                                      float mount_update_innovation_gate_mps);
void sf_eskf_fuse_body_vel(sf_eskf_t *eskf, float r_body_vel);
void sf_eskf_fuse_body_vel_scaled(sf_eskf_t *eskf, float r_body_vel,
                                  float mount_update_scale,
                                  float mount_update_innovation_gate_mps);
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
