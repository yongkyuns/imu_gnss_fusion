#ifndef SENSOR_FUSION_INTERNAL_H
#define SENSOR_FUSION_INTERNAL_H

#include "sensor_fusion.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float t_s;
  float vel_ned_mps[3];
} sf_internal_bootstrap_gnss_state_t;

typedef struct {
  float t_s;
  float gyro_radps[3];
  float accel_mps2[3];
} sf_internal_bootstrap_imu_sample_t;

typedef struct {
  float dt_s;
  float mean_gyro_b[3];
  float mean_accel_b[3];
} sf_internal_turn_interval_summary_t;

typedef struct {
  float mean_accel_b[3];
  float c_b_v[3][3];
} sf_stationary_mount_bootstrap_t;

typedef struct {
  float dt;
  float mean_gyro_b[3];
  float mean_accel_b[3];
  float gnss_vel_prev_n[3];
  float gnss_vel_curr_n[3];
} sf_align_window_summary_t;

typedef struct {
  float q_start[4];
  bool coarse_alignment_ready;
  bool after_gravity_valid;
  float after_gravity[4];
  bool after_horiz_accel_valid;
  float after_horiz_accel[4];
  bool horiz_angle_err_rad_valid;
  float horiz_angle_err_rad;
  bool horiz_effective_std_rad_valid;
  float horiz_effective_std_rad;
  bool horiz_gnss_norm_mps2_valid;
  float horiz_gnss_norm_mps2;
  bool horiz_imu_norm_mps2_valid;
  float horiz_imu_norm_mps2;
  bool horiz_speed_q_valid;
  float horiz_speed_q;
  bool horiz_accel_q_valid;
  float horiz_accel_q;
  bool horiz_straight_q_valid;
  float horiz_straight_q;
  bool horiz_turn_q_valid;
  float horiz_turn_q;
  bool horiz_dominance_q_valid;
  float horiz_dominance_q;
  bool horiz_turn_core_valid;
  bool horiz_straight_core_valid;
  bool after_turn_gyro_valid;
  float after_turn_gyro[4];
} sf_align_update_trace_t;

typedef struct {
  float speed_mps;
  float course_rate_radps;
  float a_lat_mps2;
} sf_turn_consistency_sample_t;

#define SF_TURN_CONSISTENCY_CAPACITY 16u

typedef struct {
  sf_align_t state;
  sf_turn_consistency_sample_t samples[SF_TURN_CONSISTENCY_CAPACITY];
  uint32_t count;
} sf_align_runtime_t;

typedef struct {
  sf_fusion_config_t cfg;
  sf_ekf_t ekf;
  sf_align_runtime_t align_rt;
  bool internal_align_enabled;
  bool align_initialized;
  bool mount_ready;
  bool ekf_initialized;
  bool mount_q_vb_valid;
  float mount_q_vb[4];
  float last_imu_t_s;
  bool last_imu_t_valid;
  sf_gnss_sample_t last_gnss;
  bool last_gnss_valid;
  sf_internal_bootstrap_gnss_state_t bootstrap_prev_gnss;
  bool bootstrap_prev_gnss_valid;
  float interval_imu_sum_gyro[3];
  float interval_imu_sum_accel[3];
  uint32_t interval_imu_count;
  uint32_t bootstrap_stationary_count;
  float bootstrap_gyro_ema;
  float bootstrap_accel_err_ema;
  float bootstrap_speed_ema;
  bool bootstrap_gyro_ema_valid;
  bool bootstrap_accel_err_ema_valid;
  bool bootstrap_speed_ema_valid;
  float stationary_accel_buffer[400][3];
  sf_internal_bootstrap_imu_sample_t bootstrap_imu_buffer[512];
  uint32_t bootstrap_imu_count;
} sf_sensor_fusion_impl_t;

_Static_assert(sizeof(sf_sensor_fusion_impl_t) <= SF_SENSOR_FUSION_STORAGE_BYTES,
               "sf_sensor_fusion_t storage is too small for implementation");

static inline sf_sensor_fusion_impl_t *sf_impl(sf_sensor_fusion_t *fusion) {
  return (sf_sensor_fusion_impl_t *)(void *)fusion;
}

static inline const sf_sensor_fusion_impl_t *sf_impl_const(
    const sf_sensor_fusion_t *fusion) {
  return (const sf_sensor_fusion_impl_t *)(const void *)fusion;
}

bool sf_bootstrap_vehicle_to_body_from_stationary(
    const float (*accel_samples_b)[3],
    uint32_t sample_count,
    float yaw_seed_rad,
    sf_stationary_mount_bootstrap_t *out);

void sf_align_init(sf_align_runtime_t *align_rt, const sf_align_config_t *cfg);
bool sf_align_initialize_from_stationary(sf_align_runtime_t *align_rt,
                                         const sf_align_config_t *cfg,
                                         const float (*accel_samples_b)[3],
                                         uint32_t sample_count,
                                         float yaw_seed_rad);
float sf_align_update_window_with_trace(sf_align_runtime_t *align_rt,
                                        const sf_align_config_t *cfg,
                                        const sf_align_window_summary_t *window,
                                        sf_align_update_trace_t *trace_out);
bool sf_align_coarse_alignment_ready(const sf_align_runtime_t *align_rt);

#ifdef __cplusplus
}
#endif

#endif
