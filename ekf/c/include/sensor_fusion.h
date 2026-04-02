/**
 * @file sensor_fusion.h
 * @brief Public API for the standalone sensor_fusion C library.
 *
 * This library provides:
 * - IMU/GNSS EKF fusion
 * - optional internal IMU-to-vehicle mount alignment
 * - optional externally supplied IMU-to-vehicle misalignment
 *
 * Design constraints:
 * - single precision only
 * - no heap allocation
 * - no OS/time/syscall dependency
 * - caller-owned storage only
 * - deterministic behavior for deterministic inputs
 */
#ifndef SENSOR_FUSION_H
#define SENSOR_FUSION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SF_ALIGN_N_STATES 3
#define SF_GRAVITY_MSS 9.80665f
#define SF_SENSOR_FUSION_STORAGE_BYTES 32768u

typedef struct {
  float gyro_var;
  float accel_var;
  float gyro_bias_rw_var;
  float accel_bias_rw_var;
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
  uint32_t stationary_samples;
  float max_gyro_radps;
  float max_accel_norm_err_mps2;
} sf_bootstrap_config_t;

typedef struct {
  sf_align_config_t align;
  sf_bootstrap_config_t bootstrap;
  sf_predict_noise_t predict_noise;
  float r_body_vel;
  float yaw_init_speed_mps;
} sf_fusion_config_t;

typedef struct {
  float t_s;
  float gyro_radps[3];
  float accel_mps2[3];
} sf_imu_sample_t;

typedef struct {
  float t_s;
  float pos_ned_m[3];
  float vel_ned_mps[3];
  float pos_std_m[3];
  float vel_std_mps[3];
  bool heading_valid;
  float heading_rad;
} sf_gnss_sample_t;

typedef struct {
  bool mount_ready;
  bool mount_ready_changed;
  bool ekf_initialized;
  bool ekf_initialized_now;
  bool mount_q_vb_valid;
  float mount_q_vb[4];
} sf_update_t;

typedef union {
  max_align_t _align;
  unsigned char _storage[SF_SENSOR_FUSION_STORAGE_BYTES];
} sf_sensor_fusion_t;

typedef struct sf_eskf sf_eskf_t;

void sf_align_config_default(sf_align_config_t *cfg);
void sf_bootstrap_config_default(sf_bootstrap_config_t *cfg);
void sf_predict_noise_default(sf_predict_noise_t *cfg);
void sf_fusion_config_default(sf_fusion_config_t *cfg);

void sf_fusion_init_internal(sf_sensor_fusion_t *fusion,
                             const sf_fusion_config_t *cfg);
void sf_fusion_init_external(sf_sensor_fusion_t *fusion,
                             const sf_fusion_config_t *cfg,
                             const float q_vb[4]);
void sf_fusion_set_misalignment(sf_sensor_fusion_t *fusion, const float q_vb[4]);

sf_update_t sf_fusion_process_imu(sf_sensor_fusion_t *fusion,
                                  const sf_imu_sample_t *sample);
sf_update_t sf_fusion_process_gnss(sf_sensor_fusion_t *fusion,
                                   const sf_gnss_sample_t *sample);

const sf_eskf_t *sf_fusion_eskf(const sf_sensor_fusion_t *fusion);
const sf_align_t *sf_fusion_align(const sf_sensor_fusion_t *fusion);
bool sf_fusion_mount_ready(const sf_sensor_fusion_t *fusion);
bool sf_fusion_mount_q_vb(const sf_sensor_fusion_t *fusion, float out_q_vb[4]);

#ifdef __cplusplus
}
#endif

#endif
