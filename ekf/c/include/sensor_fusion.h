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
#define SF_ESKF_ERROR_STATES 18
#define SF_GRAVITY_MSS 9.80665f
#define SF_SENSOR_FUSION_STORAGE_BYTES 32768u

typedef enum {
  SF_ALIGN_STATE_NONE = 0,
  SF_ALIGN_STATE_COARSE = 1,
  SF_ALIGN_STATE_FINE = 2,
} sf_align_state_t;

typedef struct {
  float t_s;
  float gyro_radps[3];
  float accel_mps2[3];
} sf_imu_sample_t;

typedef struct {
  float t_s;
  float lat_deg;
  float lon_deg;
  float height_m;
  float vel_ned_mps[3];
  float pos_std_m[3];
  float vel_std_mps[3];
  bool heading_valid;
  float heading_rad;
} sf_gnss_sample_t;

typedef enum {
  SF_VEHICLE_SPEED_DIRECTION_UNKNOWN = 0,
  SF_VEHICLE_SPEED_DIRECTION_FORWARD = 1,
  SF_VEHICLE_SPEED_DIRECTION_REVERSE = 2,
} sf_vehicle_speed_direction_t;

typedef struct {
  float t_s;
  float speed_mps;
  sf_vehicle_speed_direction_t direction;
} sf_vehicle_speed_sample_t;

typedef struct {
  bool mount_ready;
  bool mount_ready_changed;
  bool sensor_fusion_state;
  bool sensor_fusion_state_changed;
} sf_update_t;

typedef struct {
  bool mount_ready;
  bool mount_q_vb_valid;
  float mount_q_vb[4];

  sf_align_state_t align_state;
  float align_q_vb[4];
  float align_sigma_rad[SF_ALIGN_N_STATES];
  float gravity_lp_b[3];

  bool sensor_fusion_state;
  float q_bn[4];
  float vel_ned_mps[3];
  float pos_ned_m[3];
  float gyro_bias_radps[3];
  float accel_bias_mps2[3];
} sf_state_t;

typedef union {
  max_align_t _align;
  unsigned char _storage[SF_SENSOR_FUSION_STORAGE_BYTES];
} sf_t;

void sf_init(sf_t *sf, const float *q_vb_or_null);
void sf_set_r_body_vel(sf_t *sf, float r_body_vel);
void sf_set_gnss_pos_mount_scale(sf_t *sf, float gnss_pos_mount_scale);
void sf_set_gnss_vel_mount_scale(sf_t *sf, float gnss_vel_mount_scale);
void sf_set_gyro_bias_init_sigma_radps(sf_t *sf, float gyro_bias_init_sigma_radps);
void sf_set_accel_bias_init_sigma_mps2(sf_t *sf, float accel_bias_init_sigma_mps2);
void sf_set_accel_bias_rw_var(sf_t *sf, float accel_bias_rw_var);
void sf_set_mount_align_rw_var(sf_t *sf, float mount_align_rw_var);
void sf_set_mount_update_min_scale(sf_t *sf, float mount_update_min_scale);
void sf_set_mount_update_ramp_time_s(sf_t *sf, float mount_update_ramp_time_s);
void sf_set_mount_update_innovation_gate_mps(
    sf_t *sf, float mount_update_innovation_gate_mps);
void sf_set_r_vehicle_speed(sf_t *sf, float r_vehicle_speed);
void sf_set_r_zero_vel(sf_t *sf, float r_zero_vel);
void sf_set_r_stationary_accel(sf_t *sf, float r_stationary_accel);
sf_update_t sf_process_imu(sf_t *sf, const sf_imu_sample_t *sample);
sf_update_t sf_process_gnss(sf_t *sf, const sf_gnss_sample_t *sample);
sf_update_t sf_process_vehicle_speed(sf_t *sf,
                                     const sf_vehicle_speed_sample_t *sample);
bool sf_get_state(const sf_t *sf, sf_state_t *out);
bool sf_get_lla(const sf_t *sf, float out_lla[3]);

#ifdef __cplusplus
}
#endif

#endif
