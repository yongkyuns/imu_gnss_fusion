#ifndef SENSOR_FUSION_DEFS_H
#define SENSOR_FUSION_DEFS_H

#include "sensor_fusion.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float gyro_var;
  float accel_var;
  float gyro_bias_rw_var;
  float accel_bias_rw_var;
  float mount_align_rw_var;
} sf_predict_noise_t;

typedef struct {
  float q_vb[4];
  float p[SF_ALIGN_N_STATES][SF_ALIGN_N_STATES];
  float gravity_lp_b[3];
  bool coarse_alignment_ready;
} sf_align_t;

typedef struct {
  float q_mount_std_rad[SF_ALIGN_N_STATES];
  float r_gravity_std_mps2;
  float r_horiz_heading_std_rad;
  float r_turn_gyro_std_radps;
  float turn_gyro_yaw_scale;
  float r_turn_heading_std_rad;
  float gravity_lpf_alpha;
  float min_speed_mps;
  float min_turn_rate_radps;
  float min_lat_acc_mps2;
  float min_long_acc_mps2;
  uint32_t turn_consistency_min_windows;
  float turn_consistency_min_fraction;
  float turn_consistency_max_abs_lat_err_mps2;
  float turn_consistency_max_rel_lat_err;
  float max_stationary_gyro_radps;
  float max_stationary_accel_norm_err_mps2;
  bool use_gravity;
  bool use_turn_gyro;
} sf_align_config_t;

typedef struct {
  float ema_alpha;
  float max_speed_mps;
  float max_speed_rate_mps2;
  float max_course_rate_radps;
  uint32_t stationary_samples;
  float max_gyro_radps;
  float max_accel_norm_err_mps2;
} sf_bootstrap_config_t;

typedef struct {
  sf_align_config_t align;
  sf_bootstrap_config_t bootstrap;
  sf_predict_noise_t predict_noise;
  float gyro_bias_init_sigma_radps;
  float accel_bias_init_sigma_mps2;
  float r_body_vel;
  float gnss_pos_mount_scale;
  float gnss_vel_mount_scale;
  float gnss_vel_xy_update_min_scale;
  float gnss_vel_update_ramp_time_s;
  float mount_update_min_scale;
  float mount_update_ramp_time_s;
  float mount_update_innovation_gate_mps;
  float r_vehicle_speed;
  float r_zero_vel;
  float r_stationary_accel;
  float yaw_init_speed_mps;
} sf_fusion_config_t;

typedef struct {
  float t_s;
  float pos_ned_m[3];
  float vel_ned_mps[3];
  float pos_std_m[3];
  float vel_std_mps[3];
  bool heading_valid;
  float heading_rad;
} sf_gnss_ned_sample_t;

#ifdef __cplusplus
}
#endif

#endif
