#ifndef SENSOR_FUSION_INTERNAL_H
#define SENSOR_FUSION_INTERNAL_H

#include "sensor_fusion_defs.h"
#include "sf_eskf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef sf_t sf_sensor_fusion_t;

typedef struct {
  float t_s;
  float vel_ned_mps[3];
} sf_internal_bootstrap_gnss_state_t;

typedef struct {
  bool valid;
  float lat_deg;
  float lon_deg;
  float height_m;
  double ecef_m[3];
  float c_ne[3][3];
} sf_internal_anchor_state_t;

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
  bool after_gravity_quasi_static;
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
  bool yaw_observed;
} sf_align_runtime_t;

typedef uint32_t (*sf_profile_now_us_fn)(void *ctx);

typedef struct {
  uint32_t imu_rotate_count;
  uint64_t imu_rotate_total_us;
  uint32_t imu_rotate_max_us;
  uint32_t imu_predict_count;
  uint64_t imu_predict_total_us;
  uint32_t imu_predict_max_us;
  uint32_t imu_clamp_count;
  uint64_t imu_clamp_total_us;
  uint32_t imu_clamp_max_us;
  uint32_t imu_body_vel_count;
  uint64_t imu_body_vel_total_us;
  uint32_t imu_body_vel_max_us;
  uint32_t gnss_align_count;
  uint64_t gnss_align_total_us;
  uint32_t gnss_align_max_us;
  uint32_t gnss_init_count;
  uint64_t gnss_init_total_us;
  uint32_t gnss_init_max_us;
  uint32_t gnss_fuse_count;
  uint64_t gnss_fuse_total_us;
  uint32_t gnss_fuse_max_us;
} sf_profile_counters_t;

typedef struct {
  bool align_window_valid;
  sf_align_window_summary_t align_window;
  bool align_trace_valid;
  sf_align_update_trace_t align_trace;
  bool eskf_valid;
  sf_eskf_t eskf;
  uint32_t reanchor_count;
  bool last_reanchor_valid;
  float last_reanchor_t_s;
  float last_reanchor_distance_m;
  bool anchor_valid;
  float anchor_lat_deg;
  float anchor_lon_deg;
  float anchor_height_m;
} sf_fusion_debug_t;

typedef struct {
  sf_fusion_config_t cfg;
  sf_eskf_t eskf;
  sf_align_runtime_t align_rt;
  sf_internal_anchor_state_t anchor;
  bool internal_align_enabled;
  bool align_initialized;
  bool mount_ready;
  bool ekf_initialized;
  bool mount_q_vb_valid;
  float mount_q_vb[4];
  float last_imu_t_s;
  bool last_imu_t_valid;
  sf_gnss_ned_sample_t last_gnss;
  bool last_gnss_valid;
  sf_internal_bootstrap_gnss_state_t bootstrap_prev_gnss;
  bool bootstrap_prev_gnss_valid;
  float interval_imu_sum_gyro[3];
  float interval_imu_sum_accel[3];
  uint32_t interval_imu_count;
  uint32_t bootstrap_stationary_count;
  float bootstrap_stationary_accel_sum[3];
  float bootstrap_speed_ema;
  float bootstrap_speed_rate_ema;
  float bootstrap_course_rate_ema;
  float bootstrap_gyro_ema;
  float bootstrap_accel_err_ema;
  bool bootstrap_speed_ema_valid;
  bool bootstrap_speed_rate_ema_valid;
  bool bootstrap_course_rate_ema_valid;
  bool bootstrap_gyro_ema_valid;
  bool bootstrap_accel_err_ema_valid;
  bool last_align_window_valid;
  sf_align_window_summary_t last_align_window;
  bool last_align_trace_valid;
  sf_align_update_trace_t last_align_trace;
  sf_profile_now_us_fn profile_now_us;
  void *profile_ctx;
  sf_profile_counters_t profile;
  uint32_t reanchor_count;
  bool last_reanchor_valid;
  float last_reanchor_t_s;
  float last_reanchor_distance_m;
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
sf_update_t sf_fusion_process_vehicle_speed(
    sf_sensor_fusion_t *fusion, const sf_vehicle_speed_sample_t *sample);
const sf_eskf_t *sf_fusion_eskf(const sf_sensor_fusion_t *fusion);
const sf_align_t *sf_fusion_align(const sf_sensor_fusion_t *fusion);
bool sf_fusion_mount_ready(const sf_sensor_fusion_t *fusion);
bool sf_fusion_mount_q_vb(const sf_sensor_fusion_t *fusion, float out_q_vb[4]);
bool sf_fusion_get_debug(const sf_sensor_fusion_t *fusion, sf_fusion_debug_t *out);

void sf_fusion_set_profile_now_us(sf_sensor_fusion_t *fusion,
                                  sf_profile_now_us_fn now_us,
                                  void *ctx);
const sf_profile_counters_t *sf_fusion_profile(const sf_sensor_fusion_t *fusion);

#ifdef __cplusplus
}
#endif

#endif
