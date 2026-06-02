#ifndef SF_ALIGN_H
#define SF_ALIGN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SF_ALIGN_N_STATES 3u
#define SF_ALIGN_TURN_CONSISTENCY_CAPACITY 16u

extern const float SF_ALIGN_GRAVITY_MPS2;

typedef struct {
    float q_mount_std_rad[SF_ALIGN_N_STATES];
    bool refine_after_coarse_ready;
    float refine_process_noise_scale;
    float refine_observation_std_scale;
    float r_gravity_std_mps2;
    float r_horiz_yaw_std_rad;
    float r_turn_gyro_std_radps;
    float gravity_lpf_alpha;
    float min_speed_mps;
    float min_turn_rate_radps;
    float min_lat_acc_mps2;
    float min_long_acc_mps2;
    size_t turn_consistency_min_windows;
    float turn_consistency_min_fraction;
    float turn_consistency_max_abs_lat_err_mps2;
    float turn_consistency_max_rel_lat_err;
    float max_stationary_gyro_radps;
    float max_stationary_accel_norm_err_mps2;
    bool use_gravity;
    bool use_turn_gyro;
} sf_align_config_t;

typedef struct {
    float dt;
    float mean_gyro_b[3];
    float mean_accel_b[3];
    float gnss_vel_prev_n[3];
    float gnss_vel_curr_n[3];
    float gnss_vel_prev_std_mps[3];
    float gnss_vel_curr_std_mps[3];
    uint32_t imu_sample_count;
} sf_align_window_summary_t;

typedef struct {
    float q_start[4];
    bool coarse_alignment_ready;
    bool gravity_applied;
    bool horiz_accel_applied;
    bool turn_gyro_applied;
    bool horiz_turn_core_valid;
    bool horiz_straight_core_valid;
    bool refinement_active;
    float refinement_process_noise_scale;
    float refinement_observation_std_scale;
    float horiz_angle_err_rad;
    float horiz_effective_std_rad;
    float horiz_gnss_norm_mps2;
    float horiz_imu_norm_mps2;
    float horiz_obs_accel_vx;
    float horiz_obs_accel_vy;
} sf_align_update_trace_t;

typedef struct {
    float speed_mps;
    float course_rate_radps;
    float a_lat_mps2;
} sf_align_turn_consistency_sample_t;

typedef struct {
    float q_bv[4];
    float p[SF_ALIGN_N_STATES][SF_ALIGN_N_STATES];
    float gravity_lp_b[3];
    bool coarse_aligned;
    bool yaw_observed;
    sf_align_turn_consistency_sample_t turn_samples[SF_ALIGN_TURN_CONSISTENCY_CAPACITY];
    size_t turn_count;
    sf_align_config_t cfg;
} sf_align_t;

sf_align_config_t sf_align_config_default(void);
void sf_align_init(sf_align_t *align, sf_align_config_t cfg);
bool sf_align_initialize_from_stationary(sf_align_t *align,
                                         const float (*accel_samples_b)[3],
                                         size_t sample_count);
void sf_align_predict(sf_align_t *align, float dt);
float sf_align_update_window(sf_align_t *align, const sf_align_window_summary_t *window);
float sf_align_update_window_with_trace(sf_align_t *align,
                                        const sf_align_window_summary_t *window,
                                        sf_align_update_trace_t *trace);
void sf_align_mount_angles_rad(const sf_align_t *align, float out_rad[3]);
void sf_align_mount_angles_deg(const sf_align_t *align, float out_deg[3]);
void sf_align_sigma_deg(const sf_align_t *align, float out_deg[3]);
bool sf_align_coarse_alignment_ready(const sf_align_t *align);
float sf_align_coarse_progress(const sf_align_t *align);

#ifdef __cplusplus
}
#endif

#endif
