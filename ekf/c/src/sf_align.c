#include "sensor_fusion_internal.h"

#include <math.h>
#include <string.h>

#define SF_ALIGN_SIGMA_READY_DEG 0.15f

static void sf_diag3(const float d[3], float out[3][3]);
static float sf_wrap_angle_rad(float x);
static void sf_vec3_add(const float a[3], const float b[3], float out[3]);
static void sf_vec3_sub(const float a[3], const float b[3], float out[3]);
static void sf_vec3_scale(const float v[3], float s, float out[3]);
static float sf_vec3_dot(const float a[3], const float b[3]);
static float sf_vec3_norm(const float v[3]);
static bool sf_vec3_normalize(const float v[3], float out[3]);
static float sf_vec2_norm(const float v[2]);
static bool sf_vec2_normalize(const float v[2], float out[2]);
static void sf_skew3(const float v[3], float out[3][3]);
static void sf_mat3_mul(const float a[3][3], const float b[3][3], float out[3][3]);
static void sf_mat3_vec(const float a[3][3], const float x[3], float out[3]);
static void sf_mat3_f32_to_f64(const float a[3][3], double out[3][3]);
static void sf_mat3_f64_to_f32(const double a[3][3], float out[3][3]);
static void sf_mat3_mul_f64(const double a[3][3], const double b[3][3], double out[3][3]);
static void sf_mat3_vec_f64(const double a[3][3], const double x[3], double out[3]);
static double sf_dot3_f64(const double a[3], const double b[3]);
static void sf_symmetrize3_f64(double a[3][3]);
static void sf_quat_mul(const float a[4], const float b[4], float out[4]);
static void sf_quat_normalize(float q[4]);
static void sf_quat_from_small_angle(const float dtheta[3], float out[4]);
static void sf_quat_from_yaw(float yaw_rad, float out[4]);
static void sf_quat_to_rotmat(const float q[4], float out[3][3]);
static void sf_quat_from_rotmat(const float c[3][3], float out[4]);
static void sf_transpose3x3(const float a[3][3], float out[3][3]);
static void sf_align_obs(const float q_vb[4],
                         const float gyro_b[3],
                         const float accel_b[3],
                         float out[6]);
static void sf_align_obs_jacobian(const float q_vb[4],
                                  const float gyro_b[3],
                                  const float accel_b[3],
                                  float out[6][3]);
static bool sf_turn_consistency_update(sf_align_runtime_t *align_rt,
                                       const sf_align_config_t *cfg,
                                       bool turn_valid,
                                       float speed_mps,
                                       float course_rate_radps,
                                       float a_lat_mps2);
static void sf_turn_consistency_reset(sf_align_runtime_t *align_rt);
static void sf_align_predict_covariance(float p[3][3],
                                        const sf_align_config_t *cfg,
                                        float dt);
static void sf_align_inject_small_angle(float q_vb[4], const float dtheta[3]);
static void sf_align_inject_vehicle_yaw(float q_vb[4], float dpsi);
static float sf_apply_update1_masked(float q_vb[4],
                                     float p[3][3],
                                     float z,
                                     int obs_idx,
                                     const float accel_b[3],
                                     const float gyro_b[3],
                                     float r_var,
                                     const bool state_mask[3]);
static float sf_apply_update2_scaled_masked(float q_vb[4],
                                            float p[3][3],
                                            const float z[2],
                                            const int obs_idx[2],
                                            const float accel_b[3],
                                            const float gyro_b[3],
                                            const float r_var[2],
                                            const bool state_mask[3],
                                            const float state_scale[3]);
static float sf_apply_vehicle_yaw_angle(float q_vb[4],
                                        float p[3][3],
                                        float angle_err_rad,
                                        float r_var);
static bool sf_compute_coarse_alignment_ready(const sf_align_runtime_t *align_rt);

void sf_align_init(sf_align_runtime_t *align_rt, const sf_align_config_t *cfg) {
  float d[3];
  (void)cfg;
  if (align_rt == NULL) {
    return;
  }
  memset(align_rt, 0, sizeof(*align_rt));
  align_rt->state.q_vb[0] = 1.0f;
  d[0] = (20.0f * 3.1415927f / 180.0f) * (20.0f * 3.1415927f / 180.0f);
  d[1] = d[0];
  d[2] = (60.0f * 3.1415927f / 180.0f) * (60.0f * 3.1415927f / 180.0f);
  sf_diag3(d, align_rt->state.p);
  align_rt->state.gravity_lp_b[0] = 0.0f;
  align_rt->state.gravity_lp_b[1] = 0.0f;
  align_rt->state.gravity_lp_b[2] = -SF_GRAVITY_MSS;
}

bool sf_align_initialize_from_stationary(sf_align_runtime_t *align_rt,
                                         const sf_align_config_t *cfg,
                                         const float (*accel_samples_b)[3],
                                         uint32_t sample_count,
                                         float yaw_seed_rad) {
  sf_stationary_mount_bootstrap_t init;
  float d[3];

  if (align_rt == NULL || cfg == NULL) {
    return false;
  }
  if (!sf_bootstrap_vehicle_to_body_from_stationary(
          accel_samples_b, sample_count, yaw_seed_rad, &init)) {
    return false;
  }

  sf_quat_from_rotmat(init.c_b_v, align_rt->state.q_vb);
  d[0] = (0.2f * 3.1415927f / 180.0f) * (0.2f * 3.1415927f / 180.0f);
  d[1] = d[0];
  d[2] = (0.5f * 3.1415927f / 180.0f) * (0.5f * 3.1415927f / 180.0f);
  sf_diag3(d, align_rt->state.p);
  memcpy(align_rt->state.gravity_lp_b, init.mean_accel_b, sizeof(init.mean_accel_b));
  sf_turn_consistency_reset(align_rt);
  align_rt->state.coarse_alignment_ready = sf_compute_coarse_alignment_ready(align_rt);
  return true;
}

float sf_align_update_window_with_trace(sf_align_runtime_t *align_rt,
                                        const sf_align_config_t *cfg,
                                        const sf_align_window_summary_t *window,
                                        sf_align_update_trace_t *trace_out) {
  float score = 0.0f;
  float v_prev[3];
  float v_curr[3];
  float speed_prev;
  float speed_curr;
  float speed_mid;
  float course_prev;
  float course_curr;
  float course_rate;
  float a_n[3];
  float v_mid_h[2];
  float t_hat[2];
  bool t_hat_valid;
  float lat_hat[2] = {0.0f, 0.0f};
  float a_long = 0.0f;
  float a_lat = 0.0f;
  float gyro_norm;
  float accel_norm;
  float g_hat_b[3];
  float horiz_accel_b[3];
  float horiz_obs[6];
  bool stationary;
  bool turn_valid;
  bool turn_heading_valid;
  float horiz_gnss_norm;
  bool long_valid;
  bool heading_updates_enabled = true;
  bool gravity_state_mask[3] = {true, true, true};

  if (align_rt == NULL || cfg == NULL || window == NULL) {
    return 0.0f;
  }

  memset(trace_out, 0, sizeof(*trace_out));
  memcpy(trace_out->q_start, align_rt->state.q_vb, sizeof(trace_out->q_start));

  sf_align_predict_covariance(align_rt->state.p, cfg, window->dt > 1.0e-3f ? window->dt : 1.0e-3f);

  memcpy(v_prev, window->gnss_vel_prev_n, sizeof(v_prev));
  memcpy(v_curr, window->gnss_vel_curr_n, sizeof(v_curr));
  speed_prev = sf_vec2_norm((float[2]){v_prev[0], v_prev[1]});
  speed_curr = sf_vec2_norm((float[2]){v_curr[0], v_curr[1]});
  speed_mid = 0.5f * (speed_prev + speed_curr);

  course_prev = atan2f(v_prev[1], v_prev[0]);
  course_curr = atan2f(v_curr[1], v_curr[0]);
  course_rate = sf_wrap_angle_rad(course_curr - course_prev) /
                (window->dt > 1.0e-3f ? window->dt : 1.0e-3f);

  sf_vec3_sub(v_curr, v_prev, a_n);
  sf_vec3_scale(a_n, 1.0f / (window->dt > 1.0e-3f ? window->dt : 1.0e-3f), a_n);
  v_mid_h[0] = 0.5f * (v_prev[0] + v_curr[0]);
  v_mid_h[1] = 0.5f * (v_prev[1] + v_curr[1]);
  t_hat_valid = sf_vec2_normalize(v_mid_h, t_hat);
  if (t_hat_valid) {
    lat_hat[0] = -t_hat[1];
    lat_hat[1] = t_hat[0];
    a_long = t_hat[0] * a_n[0] + t_hat[1] * a_n[1];
    a_lat = lat_hat[0] * a_n[0] + lat_hat[1] * a_n[1];
  }

  gyro_norm = sf_vec3_norm(window->mean_gyro_b);
  accel_norm = sf_vec3_norm(window->mean_accel_b);
  if (sf_vec3_normalize(align_rt->state.gravity_lp_b, g_hat_b)) {
    float proj[3];
    sf_vec3_scale(g_hat_b, sf_vec3_dot(window->mean_accel_b, g_hat_b), proj);
    sf_vec3_sub(window->mean_accel_b, proj, horiz_accel_b);
  } else {
    memcpy(horiz_accel_b, window->mean_accel_b, sizeof(horiz_accel_b));
  }
  sf_align_obs(align_rt->state.q_vb, window->mean_gyro_b, horiz_accel_b, horiz_obs);

  stationary = gyro_norm <= cfg->max_stationary_gyro_radps &&
               fabsf(accel_norm - SF_GRAVITY_MSS) <= cfg->max_stationary_accel_norm_err_mps2 &&
               speed_mid < 0.5f;
  turn_valid = speed_mid > cfg->min_speed_mps &&
               fabsf(course_rate) > cfg->min_turn_rate_radps &&
               fabsf(a_lat) > cfg->min_lat_acc_mps2;
  turn_heading_valid = sf_turn_consistency_update(
      align_rt, cfg, turn_valid, speed_mid, course_rate, a_lat);
  horiz_gnss_norm = sqrtf(a_long * a_long + a_lat * a_lat);
  long_valid = speed_mid > cfg->min_speed_mps &&
               fabsf(a_long) > cfg->min_long_acc_mps2 &&
               fabsf(a_lat) < fmaxf(0.5f, 0.6f * fabsf(a_long)) &&
               horiz_gnss_norm > cfg->min_long_acc_mps2;

  if (stationary) {
    float a[3];
    sf_vec3_scale(align_rt->state.gravity_lp_b, 1.0f - cfg->gravity_lpf_alpha, a);
    sf_vec3_scale(window->mean_accel_b, cfg->gravity_lpf_alpha, g_hat_b);
    sf_vec3_add(a, g_hat_b, align_rt->state.gravity_lp_b);
  }

  if (cfg->use_gravity && stationary) {
    score += sf_apply_update2_scaled_masked(
        align_rt->state.q_vb, align_rt->state.p, (float[2]){0.0f, 0.0f},
        (int[2]){3, 4}, align_rt->state.gravity_lp_b, window->mean_gyro_b,
        (float[2]){cfg->r_gravity_std_mps2 * cfg->r_gravity_std_mps2,
                   cfg->r_gravity_std_mps2 * cfg->r_gravity_std_mps2},
        gravity_state_mask, (float[3]){1.0f, 1.0f, 1.0f});
    score += sf_apply_update1_masked(
        align_rt->state.q_vb, align_rt->state.p, -sf_vec3_norm(align_rt->state.gravity_lp_b), 5,
        align_rt->state.gravity_lp_b, window->mean_gyro_b,
        cfg->r_gravity_std_mps2 * cfg->r_gravity_std_mps2, gravity_state_mask);
    if (trace_out != NULL) {
      trace_out->after_gravity_valid = true;
      memcpy(trace_out->after_gravity, align_rt->state.q_vb, sizeof(trace_out->after_gravity));
    }
  }

  {
    float horiz_imu_norm = sqrtf(horiz_obs[3] * horiz_obs[3] + horiz_obs[4] * horiz_obs[4]);
    bool straight_core_valid = long_valid;
    bool turn_core_valid = turn_heading_valid &&
                           speed_mid > (10.0f / 3.6f) &&
                           fabsf(a_lat) > fmaxf(cfg->min_lat_acc_mps2, 0.7f) &&
                           fabsf(a_lat) > 1.5f * fmaxf(fabsf(a_long), 0.2f);
    bool horiz_vector_valid = speed_mid > cfg->min_speed_mps &&
                              horiz_gnss_norm > cfg->min_long_acc_mps2 &&
                              horiz_imu_norm > cfg->min_long_acc_mps2 &&
                              (straight_core_valid || turn_core_valid);
    if (trace_out != NULL) {
      trace_out->horiz_straight_core_valid = straight_core_valid;
      trace_out->horiz_turn_core_valid = turn_core_valid;
    }
    if (horiz_vector_valid) {
      float speed_q = ((speed_mid - (10.0f / 3.6f)) /
                       (20.0f / 3.6f - 10.0f / 3.6f));
      float accel_q = ((fminf(horiz_gnss_norm, horiz_imu_norm) - 0.5f) / 1.0f);
      float effective_std;
      speed_q = fminf(fmaxf(speed_q, 0.0f), 1.0f);
      accel_q = fminf(fmaxf(accel_q, 0.0f), 1.0f);
      if (trace_out != NULL) {
        trace_out->horiz_gnss_norm_mps2_valid = true;
        trace_out->horiz_gnss_norm_mps2 = horiz_gnss_norm;
        trace_out->horiz_imu_norm_mps2_valid = true;
        trace_out->horiz_imu_norm_mps2 = horiz_imu_norm;
        trace_out->horiz_speed_q_valid = true;
        trace_out->horiz_speed_q = speed_q;
        trace_out->horiz_accel_q_valid = true;
        trace_out->horiz_accel_q = accel_q;
      }
      if (turn_core_valid) {
        float dominance = ((fabsf(a_lat) / (fabsf(a_long) + 0.2f)) - 1.5f) / 1.5f;
        float lat_q = ((fabsf(a_lat) - fmaxf(cfg->min_lat_acc_mps2, 0.7f)) / 1.0f);
        float turn_q;
        dominance = fminf(fmaxf(dominance, 0.0f), 1.0f);
        lat_q = fminf(fmaxf(lat_q, 0.0f), 1.0f);
        turn_q = 0.35f + 0.65f * (speed_q * accel_q * lat_q * dominance);
        turn_q = fminf(fmaxf(turn_q, 0.35f), 1.0f);
        effective_std = cfg->r_turn_heading_std_rad / turn_q;
        if (trace_out != NULL) {
          trace_out->horiz_dominance_q_valid = true;
          trace_out->horiz_dominance_q = dominance;
          trace_out->horiz_turn_q_valid = true;
          trace_out->horiz_turn_q = turn_q;
        }
      } else {
        float lat_ratio = fabsf(a_lat) / (0.5f + 0.6f * fabsf(a_long));
        float long_q = ((fabsf(a_long) - cfg->min_long_acc_mps2) / 0.8f);
        float straight_q = speed_q * accel_q * fminf(fmaxf(long_q, 0.0f), 1.0f) *
                           (1.0f - fminf(fmaxf(lat_ratio, 0.0f), 1.0f));
        straight_q = fminf(fmaxf(straight_q, 0.2f), 1.0f);
        effective_std = cfg->r_horiz_heading_std_rad / straight_q;
        if (trace_out != NULL) {
          trace_out->horiz_straight_q_valid = true;
          trace_out->horiz_straight_q = straight_q;
        }
      }
      {
        float cross = horiz_obs[3] * a_lat - horiz_obs[4] * a_long;
        float dot = horiz_obs[3] * a_long + horiz_obs[4] * a_lat;
        float angle_err = atan2f(cross, dot);
        if (trace_out != NULL) {
          trace_out->horiz_angle_err_rad_valid = true;
          trace_out->horiz_angle_err_rad = angle_err;
          trace_out->horiz_effective_std_rad_valid = true;
          trace_out->horiz_effective_std_rad = effective_std;
        }
        score += sf_apply_vehicle_yaw_angle(
            align_rt->state.q_vb, align_rt->state.p, angle_err, effective_std * effective_std);
        if (trace_out != NULL) {
          trace_out->after_horiz_accel_valid = true;
          memcpy(trace_out->after_horiz_accel,
                 align_rt->state.q_vb,
                 sizeof(trace_out->after_horiz_accel));
        }
      }
    }
  }

  if (turn_valid && cfg->use_turn_gyro) {
    float turn_gyro_yaw_scale = (heading_updates_enabled && turn_heading_valid)
                                    ? cfg->turn_gyro_yaw_scale
                                    : 0.0f;
    bool state_mask[3] = {true, true, turn_gyro_yaw_scale > 0.0f};
    float state_scale[3] = {1.0f, 1.0f, turn_gyro_yaw_scale};
    score += sf_apply_update2_scaled_masked(
        align_rt->state.q_vb, align_rt->state.p, (float[2]){0.0f, 0.0f},
        (int[2]){0, 1}, window->mean_accel_b, window->mean_gyro_b,
        (float[2]){cfg->r_turn_gyro_std_radps * cfg->r_turn_gyro_std_radps,
                   cfg->r_turn_gyro_std_radps * cfg->r_turn_gyro_std_radps},
        state_mask, state_scale);
    if (trace_out != NULL) {
      trace_out->after_turn_gyro_valid = true;
      memcpy(trace_out->after_turn_gyro, align_rt->state.q_vb, sizeof(trace_out->after_turn_gyro));
    }
  }

  align_rt->state.coarse_alignment_ready = sf_compute_coarse_alignment_ready(align_rt);
  if (trace_out != NULL) {
    trace_out->coarse_alignment_ready = align_rt->state.coarse_alignment_ready;
  }
  return score;
}

bool sf_align_coarse_alignment_ready(const sf_align_runtime_t *align_rt) {
  return align_rt != NULL && sf_compute_coarse_alignment_ready(align_rt);
}

static bool sf_turn_consistency_update(sf_align_runtime_t *align_rt,
                                       const sf_align_config_t *cfg,
                                       bool turn_valid,
                                       float speed_mps,
                                       float course_rate_radps,
                                       float a_lat_mps2) {
  uint32_t min_windows;
  uint32_t min_ok;
  uint32_t sign_ok = 0U;
  uint32_t model_ok = 0U;

  if (!turn_valid) {
    sf_turn_consistency_reset(align_rt);
    return false;
  }

  if (align_rt->count < SF_TURN_CONSISTENCY_CAPACITY) {
    align_rt->samples[align_rt->count].speed_mps = speed_mps;
    align_rt->samples[align_rt->count].course_rate_radps = course_rate_radps;
    align_rt->samples[align_rt->count].a_lat_mps2 = a_lat_mps2;
    align_rt->count++;
  } else {
    memmove(&align_rt->samples[0],
            &align_rt->samples[1],
            sizeof(align_rt->samples[0]) * (SF_TURN_CONSISTENCY_CAPACITY - 1U));
    align_rt->samples[SF_TURN_CONSISTENCY_CAPACITY - 1U].speed_mps = speed_mps;
    align_rt->samples[SF_TURN_CONSISTENCY_CAPACITY - 1U].course_rate_radps = course_rate_radps;
    align_rt->samples[SF_TURN_CONSISTENCY_CAPACITY - 1U].a_lat_mps2 = a_lat_mps2;
  }

  min_windows = cfg->turn_consistency_min_windows > 1U
                    ? (uint32_t)cfg->turn_consistency_min_windows
                    : 1U;
  if (align_rt->count < min_windows) {
    return false;
  }

  for (uint32_t i = 0; i < align_rt->count; ++i) {
    float a_lat_pred =
        align_rt->samples[i].speed_mps * align_rt->samples[i].course_rate_radps;
    float tol = fmaxf(cfg->turn_consistency_max_abs_lat_err_mps2,
                      cfg->turn_consistency_max_rel_lat_err *
                          fmaxf(fabsf(a_lat_pred), fabsf(align_rt->samples[i].a_lat_mps2)));
    if (a_lat_pred * align_rt->samples[i].a_lat_mps2 > 0.0f) {
      sign_ok++;
    }
    if (fabsf(align_rt->samples[i].a_lat_mps2 - a_lat_pred) <= tol) {
      model_ok++;
    }
  }

  min_ok = (uint32_t)ceilf(fminf(fmaxf(cfg->turn_consistency_min_fraction, 0.0f), 1.0f) *
                           (float)align_rt->count);
  return sign_ok >= min_ok && model_ok >= min_ok;
}

static void sf_turn_consistency_reset(sf_align_runtime_t *align_rt) {
  align_rt->count = 0U;
}

static bool sf_compute_coarse_alignment_ready(const sf_align_runtime_t *align_rt) {
  float sigma_roll_deg = sqrtf(fmaxf(align_rt->state.p[0][0], 0.0f)) * 180.0f / 3.1415927f;
  float sigma_pitch_deg = sqrtf(fmaxf(align_rt->state.p[1][1], 0.0f)) * 180.0f / 3.1415927f;
  float sigma_yaw_deg = sqrtf(fmaxf(align_rt->state.p[2][2], 0.0f)) * 180.0f / 3.1415927f;
  return sigma_roll_deg <= SF_ALIGN_SIGMA_READY_DEG &&
         sigma_pitch_deg <= SF_ALIGN_SIGMA_READY_DEG &&
         sigma_yaw_deg <= SF_ALIGN_SIGMA_READY_DEG;
}

static void sf_align_predict_covariance(float p[3][3],
                                        const sf_align_config_t *cfg,
                                        float dt) {
  p[0][0] += cfg->q_mount_std_rad[0] * cfg->q_mount_std_rad[0] * dt;
  p[1][1] += cfg->q_mount_std_rad[1] * cfg->q_mount_std_rad[1] * dt;
  p[2][2] += cfg->q_mount_std_rad[2] * cfg->q_mount_std_rad[2] * dt;
}

static float sf_apply_update1_masked(float q_vb[4],
                                     float p[3][3],
                                     float z,
                                     int obs_idx,
                                     const float accel_b[3],
                                     const float gyro_b[3],
                                     float r_var,
                                     const bool state_mask[3]) {
  float obs[6];
  float H_full[6][3];
  float h[3];
  float y;
  double p64[3][3];
  double h64[3];
  double ph[3];
  double s;
  double s_inv;
  double k[3];
  float dtheta[3];
  double i_minus_kh[3][3];
  double p_new[3][3];

  sf_align_obs(q_vb, gyro_b, accel_b, obs);
  sf_align_obs_jacobian(q_vb, gyro_b, accel_b, H_full);
  for (int i = 0; i < 3; ++i) {
    h[i] = state_mask[i] ? H_full[obs_idx][i] : 0.0f;
    h64[i] = (double)h[i];
  }
  y = z - obs[obs_idx];
  sf_mat3_f32_to_f64(p, p64);
  sf_mat3_vec_f64(p64, h64, ph);
  s = sf_dot3_f64(h64, ph) + (double)r_var;
  s_inv = fabs(s) > 1.0e-20 ? 1.0 / s : 1.0;
  for (int i = 0; i < 3; ++i) {
    k[i] = ph[i] * s_inv;
    dtheta[i] = (float)(k[i] * (double)y);
  }
  sf_align_inject_small_angle(q_vb, dtheta);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      i_minus_kh[i][j] = -k[i] * h64[j];
    }
    i_minus_kh[i][i] += 1.0;
  }
  sf_mat3_mul_f64(i_minus_kh, p64, p_new);
  sf_symmetrize3_f64(p_new);
  sf_mat3_f64_to_f32(p_new, p);
  return (float)(((double)y * (double)y) * s_inv);
}

static float sf_apply_update2_scaled_masked(float q_vb[4],
                                            float p[3][3],
                                            const float z[2],
                                            const int obs_idx[2],
                                            const float accel_b[3],
                                            const float gyro_b[3],
                                            const float r_var[2],
                                            const bool state_mask[3],
                                            const float state_scale[3]) {
  float obs[6];
  float H_full[6][3];
  float h0[3];
  float h1[3];
  float y[2];
  double p64[3][3];
  double h0_64[3];
  double h1_64[3];
  double ph0[3];
  double ph1[3];
  double s00;
  double s01;
  double s10;
  double s11;
  double det;
  double s_inv00;
  double s_inv01;
  double s_inv10;
  double s_inv11;
  double k[3][2];
  float dtheta[3];
  double i_minus_kh[3][3];
  double p_new[3][3];

  sf_align_obs(q_vb, gyro_b, accel_b, obs);
  sf_align_obs_jacobian(q_vb, gyro_b, accel_b, H_full);
  for (int i = 0; i < 3; ++i) {
    h0[i] = state_mask[i] ? state_scale[i] * H_full[obs_idx[0]][i] : 0.0f;
    h1[i] = state_mask[i] ? state_scale[i] * H_full[obs_idx[1]][i] : 0.0f;
    h0_64[i] = (double)h0[i];
    h1_64[i] = (double)h1[i];
  }
  y[0] = z[0] - obs[obs_idx[0]];
  y[1] = z[1] - obs[obs_idx[1]];
  sf_mat3_f32_to_f64(p, p64);
  sf_mat3_vec_f64(p64, h0_64, ph0);
  sf_mat3_vec_f64(p64, h1_64, ph1);
  s00 = sf_dot3_f64(h0_64, ph0) + (double)r_var[0];
  s01 = sf_dot3_f64(h0_64, ph1);
  s10 = sf_dot3_f64(h1_64, ph0);
  s11 = sf_dot3_f64(h1_64, ph1) + (double)r_var[1];
  det = s00 * s11 - s01 * s10;
  if (fabs(det) > 1.0e-20) {
    double inv_det = 1.0 / det;
    s_inv00 = s11 * inv_det;
    s_inv01 = -s01 * inv_det;
    s_inv10 = -s10 * inv_det;
    s_inv11 = s00 * inv_det;
  } else {
    s_inv00 = 1.0;
    s_inv01 = 0.0;
    s_inv10 = 0.0;
    s_inv11 = 1.0;
  }
  for (int i = 0; i < 3; ++i) {
    k[i][0] = ph0[i] * s_inv00 + ph1[i] * s_inv10;
    k[i][1] = ph0[i] * s_inv01 + ph1[i] * s_inv11;
    dtheta[i] = (float)(k[i][0] * (double)y[0] + k[i][1] * (double)y[1]);
  }
  sf_align_inject_small_angle(q_vb, dtheta);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      i_minus_kh[i][j] = -(k[i][0] * h0_64[j] + k[i][1] * h1_64[j]);
    }
    i_minus_kh[i][i] += 1.0;
  }
  sf_mat3_mul_f64(i_minus_kh, p64, p_new);
  sf_symmetrize3_f64(p_new);
  sf_mat3_f64_to_f32(p_new, p);

  return (float)((double)y[0] * (s_inv00 * (double)y[0] + s_inv01 * (double)y[1]) +
                 (double)y[1] * (s_inv10 * (double)y[0] + s_inv11 * (double)y[1]));
}

static float sf_apply_vehicle_yaw_angle(float q_vb[4],
                                        float p[3][3],
                                        float angle_err_rad,
                                        float r_var) {
  float pzz = fmaxf(p[2][2], 0.0f);
  float s = pzz + fmaxf(r_var, 1.0e-9f);
  float k = s > 1.0e-9f ? pzz / s : 0.0f;
  float dpsi = -k * angle_err_rad;
  sf_align_inject_vehicle_yaw(q_vb, dpsi);
  p[2][2] = fmaxf((1.0f - k) * pzz, 0.0f);
  p[0][2] = 0.0f;
  p[2][0] = 0.0f;
  p[1][2] = 0.0f;
  p[2][1] = 0.0f;
  return angle_err_rad * angle_err_rad / s;
}

static void sf_align_inject_small_angle(float q_vb[4], const float dtheta[3]) {
  float dq[4];
  float q_out[4];
  sf_quat_from_small_angle(dtheta, dq);
  sf_quat_mul(dq, q_vb, q_out);
  memcpy(q_vb, q_out, sizeof(q_out));
  sf_quat_normalize(q_vb);
}

static void sf_align_inject_vehicle_yaw(float q_vb[4], float dpsi) {
  float dq[4];
  float q_out[4];
  sf_quat_from_yaw(dpsi, dq);
  sf_quat_mul(q_vb, dq, q_out);
  memcpy(q_vb, q_out, sizeof(q_out));
  sf_quat_normalize(q_vb);
}

static void sf_align_obs(const float q_vb[4],
                         const float gyro_b[3],
                         const float accel_b[3],
                         float out[6]) {
  float c_bv[3][3];
  float c_vb[3][3];
  float gyro_v[3];
  float accel_v[3];
  sf_quat_to_rotmat(q_vb, c_vb);
  sf_transpose3x3(c_vb, c_bv);
  sf_mat3_vec(c_bv, gyro_b, gyro_v);
  sf_mat3_vec(c_bv, accel_b, accel_v);
  out[0] = gyro_v[0];
  out[1] = gyro_v[1];
  out[2] = gyro_v[2];
  out[3] = accel_v[0];
  out[4] = accel_v[1];
  out[5] = accel_v[2];
}

static void sf_align_obs_jacobian(const float q_vb[4],
                                  const float gyro_b[3],
                                  const float accel_b[3],
                                  float out[6][3]) {
  float c_bv[3][3];
  float c_vb[3][3];
  float gyro_skew[3][3];
  float accel_skew[3][3];
  float h_gyro[3][3];
  float h_accel[3][3];
  sf_quat_to_rotmat(q_vb, c_vb);
  sf_transpose3x3(c_vb, c_bv);
  sf_skew3(gyro_b, gyro_skew);
  sf_skew3(accel_b, accel_skew);
  sf_mat3_mul(c_bv, gyro_skew, h_gyro);
  sf_mat3_mul(c_bv, accel_skew, h_accel);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[i][j] = h_gyro[i][j];
      out[i + 3][j] = h_accel[i][j];
    }
  }
}

static void sf_diag3(const float d[3], float out[3][3]) {
  memset(out, 0, sizeof(float) * 9U);
  out[0][0] = d[0];
  out[1][1] = d[1];
  out[2][2] = d[2];
}

static float sf_wrap_angle_rad(float x) {
  float two_pi = 2.0f * 3.1415927f;
  float y = fmodf(x + 3.1415927f, two_pi);
  if (y < 0.0f) {
    y += two_pi;
  }
  return y - 3.1415927f;
}

static void sf_vec3_add(const float a[3], const float b[3], float out[3]) {
  out[0] = a[0] + b[0];
  out[1] = a[1] + b[1];
  out[2] = a[2] + b[2];
}

static void sf_vec3_sub(const float a[3], const float b[3], float out[3]) {
  out[0] = a[0] - b[0];
  out[1] = a[1] - b[1];
  out[2] = a[2] - b[2];
}

static void sf_vec3_scale(const float v[3], float s, float out[3]) {
  out[0] = v[0] * s;
  out[1] = v[1] * s;
  out[2] = v[2] * s;
}

static float sf_vec3_dot(const float a[3], const float b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static float sf_vec3_norm(const float v[3]) {
  return sqrtf(sf_vec3_dot(v, v));
}

static bool sf_vec3_normalize(const float v[3], float out[3]) {
  float n = sf_vec3_norm(v);
  if (!isfinite(n) || n <= 1.0e-8f) {
    return false;
  }
  sf_vec3_scale(v, 1.0f / n, out);
  return true;
}

static float sf_vec2_norm(const float v[2]) {
  return sqrtf(v[0] * v[0] + v[1] * v[1]);
}

static bool sf_vec2_normalize(const float v[2], float out[2]) {
  float n = sf_vec2_norm(v);
  if (!isfinite(n) || n <= 1.0e-8f) {
    return false;
  }
  out[0] = v[0] / n;
  out[1] = v[1] / n;
  return true;
}

static void sf_skew3(const float v[3], float out[3][3]) {
  out[0][0] = 0.0f;
  out[0][1] = -v[2];
  out[0][2] = v[1];
  out[1][0] = v[2];
  out[1][1] = 0.0f;
  out[1][2] = -v[0];
  out[2][0] = -v[1];
  out[2][1] = v[0];
  out[2][2] = 0.0f;
}

static void sf_mat3_mul(const float a[3][3], const float b[3][3], float out[3][3]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
}

static void sf_mat3_vec(const float a[3][3], const float x[3], float out[3]) {
  out[0] = a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2];
  out[1] = a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2];
  out[2] = a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2];
}

static void sf_mat3_f32_to_f64(const float a[3][3], double out[3][3]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[i][j] = (double)a[i][j];
    }
  }
}

static void sf_mat3_f64_to_f32(const double a[3][3], float out[3][3]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[i][j] = (float)a[i][j];
    }
  }
}

static void sf_mat3_mul_f64(const double a[3][3], const double b[3][3], double out[3][3]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
}

static void sf_mat3_vec_f64(const double a[3][3], const double x[3], double out[3]) {
  out[0] = a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2];
  out[1] = a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2];
  out[2] = a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2];
}

static double sf_dot3_f64(const double a[3], const double b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void sf_symmetrize3_f64(double a[3][3]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      double avg = 0.5 * (a[i][j] + a[j][i]);
      a[i][j] = avg;
      a[j][i] = avg;
    }
  }
}

static void sf_quat_mul(const float a[4], const float b[4], float out[4]) {
  out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

static void sf_quat_normalize(float q[4]) {
  float n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  if (n2 <= 1.0e-12f) {
    q[0] = 1.0f;
    q[1] = 0.0f;
    q[2] = 0.0f;
    q[3] = 0.0f;
    return;
  }
  {
    float inv = 1.0f / sqrtf(n2);
    q[0] *= inv;
    q[1] *= inv;
    q[2] *= inv;
    q[3] *= inv;
  }
}

static void sf_quat_from_small_angle(const float dtheta[3], float out[4]) {
  out[0] = 1.0f;
  out[1] = 0.5f * dtheta[0];
  out[2] = 0.5f * dtheta[1];
  out[3] = 0.5f * dtheta[2];
  sf_quat_normalize(out);
}

static void sf_quat_from_yaw(float yaw_rad, float out[4]) {
  float half = 0.5f * yaw_rad;
  out[0] = cosf(half);
  out[1] = 0.0f;
  out[2] = 0.0f;
  out[3] = sinf(half);
}

static void sf_quat_to_rotmat(const float q_in[4], float out[3][3]) {
  float q[4] = {q_in[0], q_in[1], q_in[2], q_in[3]};
  float w, x, y, z;
  sf_quat_normalize(q);
  w = q[0];
  x = q[1];
  y = q[2];
  z = q[3];
  out[0][0] = 1.0f - 2.0f * (y * y + z * z);
  out[0][1] = 2.0f * (x * y - w * z);
  out[0][2] = 2.0f * (x * z + w * y);
  out[1][0] = 2.0f * (x * y + w * z);
  out[1][1] = 1.0f - 2.0f * (x * x + z * z);
  out[1][2] = 2.0f * (y * z - w * x);
  out[2][0] = 2.0f * (x * z - w * y);
  out[2][1] = 2.0f * (y * z + w * x);
  out[2][2] = 1.0f - 2.0f * (x * x + y * y);
}

static void sf_quat_from_rotmat(const float c[3][3], float out[4]) {
  float trace = c[0][0] + c[1][1] + c[2][2];
  if (trace > 0.0f) {
    float s = sqrtf(trace + 1.0f) * 2.0f;
    out[0] = 0.25f * s;
    out[1] = (c[2][1] - c[1][2]) / s;
    out[2] = (c[0][2] - c[2][0]) / s;
    out[3] = (c[1][0] - c[0][1]) / s;
  } else if (c[0][0] > c[1][1] && c[0][0] > c[2][2]) {
    float s = sqrtf(1.0f + c[0][0] - c[1][1] - c[2][2]) * 2.0f;
    out[0] = (c[2][1] - c[1][2]) / s;
    out[1] = 0.25f * s;
    out[2] = (c[0][1] + c[1][0]) / s;
    out[3] = (c[0][2] + c[2][0]) / s;
  } else if (c[1][1] > c[2][2]) {
    float s = sqrtf(1.0f + c[1][1] - c[0][0] - c[2][2]) * 2.0f;
    out[0] = (c[0][2] - c[2][0]) / s;
    out[1] = (c[0][1] + c[1][0]) / s;
    out[2] = 0.25f * s;
    out[3] = (c[1][2] + c[2][1]) / s;
  } else {
    float s = sqrtf(1.0f + c[2][2] - c[0][0] - c[1][1]) * 2.0f;
    out[0] = (c[1][0] - c[0][1]) / s;
    out[1] = (c[0][2] + c[2][0]) / s;
    out[2] = (c[1][2] + c[2][1]) / s;
    out[3] = 0.25f * s;
  }
  sf_quat_normalize(out);
}

static void sf_transpose3x3(const float a[3][3], float out[3][3]) {
  out[0][0] = a[0][0];
  out[0][1] = a[1][0];
  out[0][2] = a[2][0];
  out[1][0] = a[0][1];
  out[1][1] = a[1][1];
  out[1][2] = a[2][1];
  out[2][0] = a[0][2];
  out[2][1] = a[1][2];
  out[2][2] = a[2][2];
}
