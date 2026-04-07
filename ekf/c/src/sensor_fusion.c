#include "sensor_fusion_internal.h"

#include <math.h>
#include <string.h>

static void sf_initialize_eskf_from_gnss(sf_eskf_t *eskf,
                                         const sf_gnss_sample_t *gnss,
                                         float yaw_init_speed_mps);
static void sf_set_nominal_yaw_only(sf_eskf_nominal_state_t *state,
                                    float yaw_rad);
static void sf_clamp_eskf_biases(sf_eskf_t *eskf);
static void sf_quat_to_rotmat(const float q[4], float r[3][3]);
static void sf_transpose3(const float in[3][3], float out[3][3]);
static void sf_mat3_vec(const float m[3][3], const float v[3], float out[3]);
static float sf_norm3(const float v[3]);
static float sf_ema_update(bool *valid, float *prev, float sample, float alpha);
static float sf_wrap_pi(float rad);
static float sf_horiz_speed(const float vel_ned_mps[3]);
static void sf_bootstrap_update_gnss_hints(sf_sensor_fusion_impl_t *impl,
                                           const sf_gnss_sample_t *sample);
static bool sf_runtime_zero_velocity_active(sf_sensor_fusion_impl_t *impl,
                                            const float accel_b[3],
                                            const float gyro_radps[3]);
static bool sf_bootstrap_update(sf_sensor_fusion_impl_t *impl,
                                const float accel_b[3],
                                const float gyro_radps[3]);
static bool sf_take_interval_summary(sf_sensor_fusion_impl_t *impl, float t0_s,
                                     float t1_s,
                                     sf_align_window_summary_t *summary_out);
static sf_update_t sf_update_from_fusion(const sf_sensor_fusion_impl_t *fusion,
                                         bool mount_ready_changed,
                                         bool ekf_initialized_now);
static sf_align_state_t
sf_align_state_from_impl(const sf_sensor_fusion_impl_t *impl);
static uint32_t sf_profile_stamp(const sf_sensor_fusion_impl_t *impl);
static void sf_profile_accumulate(uint32_t *count, uint64_t *total_us,
                                  uint32_t *max_us, uint32_t elapsed_us);

#define SF_RUNTIME_ZERO_SPEED_MPS 0.80f
#define SF_RUNTIME_R_ZERO_VEL 0.01f
#define SF_RUNTIME_R_STATIONARY_ACCEL 0.05f

void sf_init(sf_t *sf, const float *q_vb_or_null) {
  if (q_vb_or_null == NULL) {
    sf_fusion_init_internal((sf_sensor_fusion_t *)sf, NULL);
  } else {
    sf_fusion_init_external((sf_sensor_fusion_t *)sf, NULL, q_vb_or_null);
  }
}

sf_update_t sf_process_imu(sf_t *sf, const sf_imu_sample_t *sample) {
  return sf_fusion_process_imu((sf_sensor_fusion_t *)sf, sample);
}

sf_update_t sf_process_gnss(sf_t *sf, const sf_gnss_sample_t *sample) {
  return sf_fusion_process_gnss((sf_sensor_fusion_t *)sf, sample);
}

bool sf_get_state(const sf_t *sf, sf_state_t *out) {
  const sf_sensor_fusion_impl_t *impl;

  if (sf == NULL || out == NULL) {
    return false;
  }

  impl = sf_impl_const((const sf_sensor_fusion_t *)sf);
  memset(out, 0, sizeof(*out));
  out->mount_ready = impl->mount_ready;
  out->mount_q_vb_valid = impl->mount_q_vb_valid;
  if (impl->mount_q_vb_valid) {
    memcpy(out->mount_q_vb, impl->mount_q_vb, sizeof(out->mount_q_vb));
  }

  out->align_state = sf_align_state_from_impl(impl);
  if (!impl->internal_align_enabled && impl->mount_q_vb_valid) {
    memcpy(out->align_q_vb, impl->mount_q_vb, sizeof(out->align_q_vb));
  } else {
    memcpy(out->align_q_vb, impl->align_rt.state.q_vb, sizeof(out->align_q_vb));
  }
  out->align_sigma_rad[0] = sqrtf(fmaxf(impl->align_rt.state.p[0][0], 0.0f));
  out->align_sigma_rad[1] = sqrtf(fmaxf(impl->align_rt.state.p[1][1], 0.0f));
  out->align_sigma_rad[2] = sqrtf(fmaxf(impl->align_rt.state.p[2][2], 0.0f));
  memcpy(out->gravity_lp_b, impl->align_rt.state.gravity_lp_b,
         sizeof(out->gravity_lp_b));

  out->sensor_fusion_state = impl->ekf_initialized;
  if (impl->ekf_initialized) {
    out->q_bn[0] = impl->eskf.nominal.q0;
    out->q_bn[1] = impl->eskf.nominal.q1;
    out->q_bn[2] = impl->eskf.nominal.q2;
    out->q_bn[3] = impl->eskf.nominal.q3;
    out->vel_ned_mps[0] = impl->eskf.nominal.vn;
    out->vel_ned_mps[1] = impl->eskf.nominal.ve;
    out->vel_ned_mps[2] = impl->eskf.nominal.vd;
    out->pos_ned_m[0] = impl->eskf.nominal.pn;
    out->pos_ned_m[1] = impl->eskf.nominal.pe;
    out->pos_ned_m[2] = impl->eskf.nominal.pd;
    out->gyro_bias_radps[0] = impl->eskf.nominal.bgx;
    out->gyro_bias_radps[1] = impl->eskf.nominal.bgy;
    out->gyro_bias_radps[2] = impl->eskf.nominal.bgz;
    out->accel_bias_mps2[0] = impl->eskf.nominal.bax;
    out->accel_bias_mps2[1] = impl->eskf.nominal.bay;
    out->accel_bias_mps2[2] = impl->eskf.nominal.baz;
  }

  return true;
}

void sf_align_config_default(sf_align_config_t *cfg) {
  if (cfg == NULL) {
    return;
  }

  cfg->q_mount_std_rad[0] = 0.001f * 3.1415927f / 180.0f;
  cfg->q_mount_std_rad[1] = 0.001f * 3.1415927f / 180.0f;
  cfg->q_mount_std_rad[2] = 0.0001f * 3.1415927f / 180.0f;
  cfg->r_gravity_std_mps2 = 0.28f;
  cfg->r_horiz_heading_std_rad = 1.0f * 3.1415927f / 180.0f;
  cfg->r_turn_gyro_std_radps = 0.01f * 3.1415927f / 180.0f;
  cfg->turn_gyro_yaw_scale = 0.0f;
  cfg->r_turn_heading_std_rad = 0.1f * 3.1415927f / 180.0f;
  cfg->gravity_lpf_alpha = 0.08f;
  cfg->min_speed_mps = 3.0f / 3.6f;
  cfg->min_turn_rate_radps = 2.0f * 3.1415927f / 180.0f;
  cfg->min_lat_acc_mps2 = 0.10f;
  cfg->min_long_acc_mps2 = 0.18f;
  cfg->turn_consistency_min_windows = 5U;
  cfg->turn_consistency_min_fraction = 0.8f;
  cfg->turn_consistency_max_abs_lat_err_mps2 = 0.35f;
  cfg->turn_consistency_max_rel_lat_err = 0.6f;
  cfg->max_stationary_gyro_radps = 0.8f * 3.1415927f / 180.0f;
  cfg->max_stationary_accel_norm_err_mps2 = 0.2f;
  cfg->use_gravity = true;
  cfg->use_turn_gyro = true;
}

void sf_bootstrap_config_default(sf_bootstrap_config_t *cfg) {
  sf_align_config_t align_cfg;

  if (cfg == NULL) {
    return;
  }

  sf_align_config_default(&align_cfg);
  cfg->ema_alpha = 0.05f;
  cfg->max_speed_mps = 0.35f;
  cfg->max_speed_rate_mps2 = 0.15f;
  cfg->max_course_rate_radps = 1.0f * 3.1415927f / 180.0f;
  cfg->stationary_samples = 100U;
  cfg->max_gyro_radps = align_cfg.max_stationary_gyro_radps;
  cfg->max_accel_norm_err_mps2 = align_cfg.max_stationary_accel_norm_err_mps2;
}

void sf_predict_noise_default(sf_predict_noise_t *cfg) {
  if (cfg == NULL) {
    return;
  }
  cfg->gyro_var = 2.2873113e-7f * 10.0f;
  cfg->accel_var = 2.4504214e-5f * 15.0f;
  cfg->gyro_bias_rw_var = 0.0002e-9f;
  cfg->accel_bias_rw_var = 0.002e-9f;
}

void sf_fusion_config_default(sf_fusion_config_t *cfg) {
  if (cfg == NULL) {
    return;
  }
  sf_align_config_default(&cfg->align);
  sf_bootstrap_config_default(&cfg->bootstrap);
  sf_predict_noise_default(&cfg->predict_noise);
  cfg->r_body_vel = 0.2f;
  cfg->yaw_init_speed_mps = 0.0f;
}

void sf_fusion_init_internal(sf_sensor_fusion_t *fusion,
                             const sf_fusion_config_t *cfg) {
  sf_sensor_fusion_impl_t *impl;
  sf_fusion_config_t cfg_local;

  if (fusion == NULL) {
    return;
  }
  impl = sf_impl(fusion);

  sf_fusion_config_default(&cfg_local);
  memset(impl, 0, sizeof(*impl));
  impl->cfg = cfg ? *cfg : cfg_local;
  impl->internal_align_enabled = true;
  sf_align_init(&impl->align_rt, &impl->cfg.align);
  sf_eskf_init(&impl->eskf, NULL, &impl->cfg.predict_noise);
}

void sf_fusion_init_external(sf_sensor_fusion_t *fusion,
                             const sf_fusion_config_t *cfg,
                             const float q_vb[4]) {
  sf_sensor_fusion_impl_t *impl;
  sf_fusion_init_internal(fusion, cfg);
  if (fusion == NULL || q_vb == NULL) {
    return;
  }
  impl = sf_impl(fusion);
  impl->internal_align_enabled = false;
  impl->mount_ready = true;
  impl->mount_q_vb_valid = true;
  memcpy(impl->mount_q_vb, q_vb, sizeof(impl->mount_q_vb));
}

void sf_fusion_set_misalignment(sf_sensor_fusion_t *fusion,
                                const float q_vb[4]) {
  sf_sensor_fusion_impl_t *impl;
  if (fusion == NULL || q_vb == NULL) {
    return;
  }
  impl = sf_impl(fusion);
  impl->internal_align_enabled = false;
  impl->mount_ready = true;
  impl->mount_q_vb_valid = true;
  memcpy(impl->mount_q_vb, q_vb, sizeof(impl->mount_q_vb));
}

bool sf_fusion_get_debug(const sf_sensor_fusion_t *fusion,
                         sf_fusion_debug_t *out) {
  const sf_sensor_fusion_impl_t *impl;

  if (fusion == NULL || out == NULL) {
    return false;
  }

  impl = sf_impl_const(fusion);
  memset(out, 0, sizeof(*out));
  out->align_window_valid = impl->last_align_window_valid;
  if (impl->last_align_window_valid) {
    out->align_window = impl->last_align_window;
  }
  out->align_trace_valid = impl->last_align_trace_valid;
  if (impl->last_align_trace_valid) {
    out->align_trace = impl->last_align_trace;
  }
  out->eskf_valid = impl->ekf_initialized;
  if (impl->ekf_initialized) {
    out->eskf = impl->eskf;
  }
  return true;
}

void sf_fusion_set_profile_now_us(sf_sensor_fusion_t *fusion,
                                  sf_profile_now_us_fn now_us, void *ctx) {
  sf_sensor_fusion_impl_t *impl;
  if (fusion == NULL) {
    return;
  }
  impl = sf_impl(fusion);
  impl->profile_now_us = now_us;
  impl->profile_ctx = ctx;
  memset(&impl->profile, 0, sizeof(impl->profile));
}

const sf_profile_counters_t *
sf_fusion_profile(const sf_sensor_fusion_t *fusion) {
  return fusion != NULL ? &sf_impl_const(fusion)->profile : NULL;
}

sf_update_t sf_fusion_process_imu(sf_sensor_fusion_t *fusion,
                                  const sf_imu_sample_t *sample) {
  sf_sensor_fusion_impl_t *impl;
  float dt_s;
  float c_bv[3][3];
  float c_vb[3][3];
  float gyro_vehicle[3];
  float accel_vehicle[3];
  sf_eskf_imu_delta_t imu_delta;
  uint32_t t0_us;
  uint32_t elapsed_us;
  bool zero_vel_active;

  if (fusion == NULL) {
    sf_update_t empty = {0};
    return empty;
  }
  impl = sf_impl(fusion);
  if (sample == NULL) {
    return sf_update_from_fusion(impl, false, false);
  }

  if (!impl->last_imu_t_valid) {
    impl->last_imu_t_s = sample->t_s;
    impl->last_imu_t_valid = true;
    if (impl->internal_align_enabled && !impl->align_initialized &&
        sf_bootstrap_update(impl, sample->accel_mps2, sample->gyro_radps)) {
      float mean_accel_sample[1][3] = {{
          impl->bootstrap_stationary_accel_sum[0] /
              (float)impl->bootstrap_stationary_count,
          impl->bootstrap_stationary_accel_sum[1] /
              (float)impl->bootstrap_stationary_count,
          impl->bootstrap_stationary_accel_sum[2] /
              (float)impl->bootstrap_stationary_count,
      }};
      if (sf_align_initialize_from_stationary(&impl->align_rt, &impl->cfg.align,
                                              mean_accel_sample, 1U, 0.0f)) {
        impl->align_initialized = true;
        impl->mount_q_vb_valid = true;
        memcpy(impl->mount_q_vb, impl->align_rt.state.q_vb,
               sizeof(impl->mount_q_vb));
      }
    }
    return sf_update_from_fusion(impl, false, false);
  }

  dt_s = sample->t_s - impl->last_imu_t_s;
  impl->last_imu_t_s = sample->t_s;

  impl->interval_imu_sum_gyro[0] += sample->gyro_radps[0];
  impl->interval_imu_sum_gyro[1] += sample->gyro_radps[1];
  impl->interval_imu_sum_gyro[2] += sample->gyro_radps[2];
  impl->interval_imu_sum_accel[0] += sample->accel_mps2[0];
  impl->interval_imu_sum_accel[1] += sample->accel_mps2[1];
  impl->interval_imu_sum_accel[2] += sample->accel_mps2[2];
  impl->interval_imu_count++;

  if (impl->internal_align_enabled && !impl->align_initialized &&
      sf_bootstrap_update(impl, sample->accel_mps2, sample->gyro_radps)) {
    float mean_accel_sample[1][3] = {{
        impl->bootstrap_stationary_accel_sum[0] /
            (float)impl->bootstrap_stationary_count,
        impl->bootstrap_stationary_accel_sum[1] /
            (float)impl->bootstrap_stationary_count,
        impl->bootstrap_stationary_accel_sum[2] /
            (float)impl->bootstrap_stationary_count,
    }};
    if (sf_align_initialize_from_stationary(&impl->align_rt, &impl->cfg.align,
                                            mean_accel_sample, 1U, 0.0f)) {
      impl->align_initialized = true;
      impl->mount_q_vb_valid = true;
      memcpy(impl->mount_q_vb, impl->align_rt.state.q_vb,
             sizeof(impl->mount_q_vb));
    }
  }

  if (!impl->ekf_initialized || !impl->mount_ready || !impl->mount_q_vb_valid) {
    return sf_update_from_fusion(impl, false, false);
  }
  if (dt_s < 0.001f || dt_s > 0.05f) {
    return sf_update_from_fusion(impl, false, false);
  }

  t0_us = sf_profile_stamp(impl);
  sf_quat_to_rotmat(impl->mount_q_vb, c_bv);
  sf_transpose3((const float (*)[3])c_bv, c_vb);
  sf_mat3_vec((const float (*)[3])c_vb, sample->gyro_radps, gyro_vehicle);
  sf_mat3_vec((const float (*)[3])c_vb, sample->accel_mps2, accel_vehicle);
  if (impl->profile_now_us != NULL) {
    elapsed_us = sf_profile_stamp(impl) - t0_us;
    sf_profile_accumulate(&impl->profile.imu_rotate_count,
                          &impl->profile.imu_rotate_total_us,
                          &impl->profile.imu_rotate_max_us, elapsed_us);
  }

  imu_delta.dax = gyro_vehicle[0] * dt_s;
  imu_delta.day = gyro_vehicle[1] * dt_s;
  imu_delta.daz = gyro_vehicle[2] * dt_s;
  imu_delta.dvx = accel_vehicle[0] * dt_s;
  imu_delta.dvy = accel_vehicle[1] * dt_s;
  imu_delta.dvz = accel_vehicle[2] * dt_s;
  imu_delta.dt = dt_s;

  t0_us = sf_profile_stamp(impl);
  sf_eskf_predict(&impl->eskf, &imu_delta);
  if (impl->profile_now_us != NULL) {
    elapsed_us = sf_profile_stamp(impl) - t0_us;
    sf_profile_accumulate(&impl->profile.imu_predict_count,
                          &impl->profile.imu_predict_total_us,
                          &impl->profile.imu_predict_max_us, elapsed_us);
  }
  t0_us = sf_profile_stamp(impl);
  sf_clamp_eskf_biases(&impl->eskf);
  if (impl->profile_now_us != NULL) {
    elapsed_us = sf_profile_stamp(impl) - t0_us;
    sf_profile_accumulate(&impl->profile.imu_clamp_count,
                          &impl->profile.imu_clamp_total_us,
                          &impl->profile.imu_clamp_max_us, elapsed_us);
  }
  if (impl->cfg.r_body_vel > 0.0f) {
    zero_vel_active =
        sf_runtime_zero_velocity_active(impl, sample->accel_mps2,
                                        sample->gyro_radps);
    t0_us = sf_profile_stamp(impl);
    if (zero_vel_active) {
      sf_eskf_fuse_zero_vel(&impl->eskf, SF_RUNTIME_R_ZERO_VEL);
      sf_eskf_fuse_stationary_gravity(&impl->eskf, accel_vehicle,
                                      SF_RUNTIME_R_STATIONARY_ACCEL);
    } else {
      sf_eskf_fuse_body_vel(&impl->eskf, impl->cfg.r_body_vel);
    }
    if (impl->profile_now_us != NULL) {
      elapsed_us = sf_profile_stamp(impl) - t0_us;
      sf_profile_accumulate(&impl->profile.imu_body_vel_count,
                            &impl->profile.imu_body_vel_total_us,
                            &impl->profile.imu_body_vel_max_us, elapsed_us);
    }
    t0_us = sf_profile_stamp(impl);
    sf_clamp_eskf_biases(&impl->eskf);
    if (impl->profile_now_us != NULL) {
      elapsed_us = sf_profile_stamp(impl) - t0_us;
      sf_profile_accumulate(&impl->profile.imu_clamp_count,
                            &impl->profile.imu_clamp_total_us,
                            &impl->profile.imu_clamp_max_us, elapsed_us);
    }
  }
  return sf_update_from_fusion(impl, false, false);
}

sf_update_t sf_fusion_process_gnss(sf_sensor_fusion_t *fusion,
                                   const sf_gnss_sample_t *sample) {
  sf_sensor_fusion_impl_t *impl;
  bool ekf_initialized_now = false;
  bool prev_mount_ready;
  sf_align_window_summary_t summary;
  bool have_summary = false;
  sf_align_update_trace_t trace;
  uint32_t t0_us;
  uint32_t elapsed_us;

  if (fusion == NULL) {
    sf_update_t empty = {0};
    return empty;
  }
  impl = sf_impl(fusion);
  if (sample == NULL) {
    return sf_update_from_fusion(impl, false, false);
  }

  impl->last_gnss = *sample;
  impl->last_gnss_valid = true;

  prev_mount_ready = impl->mount_ready;

  sf_bootstrap_update_gnss_hints(impl, sample);

  if (impl->internal_align_enabled) {
    if (impl->align_initialized && impl->bootstrap_prev_gnss_valid) {
      have_summary = sf_take_interval_summary(
          impl, impl->bootstrap_prev_gnss.t_s, sample->t_s, &summary);
    }

    if (impl->align_initialized && have_summary) {
      t0_us = sf_profile_stamp(impl);
      sf_align_update_window_with_trace(&impl->align_rt, &impl->cfg.align,
                                        &summary, &trace);
      if (impl->profile_now_us != NULL) {
        elapsed_us = sf_profile_stamp(impl) - t0_us;
        sf_profile_accumulate(&impl->profile.gnss_align_count,
                              &impl->profile.gnss_align_total_us,
                              &impl->profile.gnss_align_max_us, elapsed_us);
      }
      impl->last_align_window = summary;
      impl->last_align_window_valid = true;
      impl->last_align_trace = trace;
      impl->last_align_trace_valid = true;
      memcpy(impl->mount_q_vb, impl->align_rt.state.q_vb,
             sizeof(impl->mount_q_vb));
      impl->mount_q_vb_valid = true;
      impl->mount_ready = trace.coarse_alignment_ready;
    }

    impl->bootstrap_prev_gnss.t_s = sample->t_s;
    memcpy(impl->bootstrap_prev_gnss.vel_ned_mps, sample->vel_ned_mps,
           sizeof(sample->vel_ned_mps));
    impl->bootstrap_prev_gnss_valid = true;
  } else {
    impl->interval_imu_sum_gyro[0] = 0.0f;
    impl->interval_imu_sum_gyro[1] = 0.0f;
    impl->interval_imu_sum_gyro[2] = 0.0f;
    impl->interval_imu_sum_accel[0] = 0.0f;
    impl->interval_imu_sum_accel[1] = 0.0f;
    impl->interval_imu_sum_accel[2] = 0.0f;
    impl->interval_imu_count = 0U;
  }

  if (!impl->mount_ready) {
    return sf_update_from_fusion(impl, prev_mount_ready != impl->mount_ready,
                                 false);
  }

  if (!impl->ekf_initialized) {
    t0_us = sf_profile_stamp(impl);
    sf_initialize_eskf_from_gnss(&impl->eskf, sample,
                                 impl->cfg.yaw_init_speed_mps);
    if (impl->profile_now_us != NULL) {
      elapsed_us = sf_profile_stamp(impl) - t0_us;
      sf_profile_accumulate(&impl->profile.gnss_init_count,
                            &impl->profile.gnss_init_total_us,
                            &impl->profile.gnss_init_max_us, elapsed_us);
    }
    impl->ekf_initialized = true;
    ekf_initialized_now = true;
  } else {
    t0_us = sf_profile_stamp(impl);
    sf_eskf_fuse_gps(&impl->eskf, sample);
    if (impl->profile_now_us != NULL) {
      elapsed_us = sf_profile_stamp(impl) - t0_us;
      sf_profile_accumulate(&impl->profile.gnss_fuse_count,
                            &impl->profile.gnss_fuse_total_us,
                            &impl->profile.gnss_fuse_max_us, elapsed_us);
    }
  }

  return sf_update_from_fusion(impl, prev_mount_ready != impl->mount_ready,
                               ekf_initialized_now);
}

const sf_eskf_t *sf_fusion_eskf(const sf_sensor_fusion_t *fusion) {
  return fusion != NULL ? &sf_impl_const(fusion)->eskf : NULL;
}

const sf_align_t *sf_fusion_align(const sf_sensor_fusion_t *fusion) {
  return fusion != NULL ? &sf_impl_const(fusion)->align_rt.state : NULL;
}

bool sf_fusion_mount_ready(const sf_sensor_fusion_t *fusion) {
  return fusion != NULL && sf_impl_const(fusion)->mount_ready;
}

bool sf_fusion_mount_q_vb(const sf_sensor_fusion_t *fusion, float out_q_vb[4]) {
  const sf_sensor_fusion_impl_t *impl;

  if (fusion == NULL || out_q_vb == NULL) {
    return false;
  }
  impl = sf_impl_const(fusion);
  if (!impl->mount_q_vb_valid) {
    return false;
  }
  memcpy(out_q_vb, impl->mount_q_vb, sizeof(impl->mount_q_vb));
  return true;
}

static sf_update_t sf_update_from_fusion(const sf_sensor_fusion_impl_t *fusion,
                                         bool mount_ready_changed,
                                         bool ekf_initialized_now) {
  sf_update_t out;
  memset(&out, 0, sizeof(out));
  if (fusion == NULL) {
    return out;
  }

  out.mount_ready = fusion->mount_ready;
  out.mount_ready_changed = mount_ready_changed;
  out.sensor_fusion_state = fusion->ekf_initialized;
  out.sensor_fusion_state_changed = ekf_initialized_now;
  return out;
}

static sf_align_state_t
sf_align_state_from_impl(const sf_sensor_fusion_impl_t *impl) {
  if (impl == NULL) {
    return SF_ALIGN_STATE_NONE;
  }
  if (!impl->internal_align_enabled && impl->mount_q_vb_valid) {
    return SF_ALIGN_STATE_FINE;
  }
  if (!impl->align_initialized) {
    return SF_ALIGN_STATE_NONE;
  }
  if (impl->mount_ready) {
    return SF_ALIGN_STATE_FINE;
  }
  return SF_ALIGN_STATE_COARSE;
}

static void sf_initialize_eskf_from_gnss(sf_eskf_t *eskf,
                                         const sf_gnss_sample_t *gnss,
                                         float yaw_init_speed_mps) {
  float vel_var;
  float speed_h;
  float yaw_rad;
  const float gyro_bias_sigma_radps = 0.125f * 3.1415927f / 180.0f;
  const float accel_bias_sigma_mps2 = 0.20f;
  sf_predict_noise_t noise = eskf->noise;

  sf_eskf_init(eskf, NULL, &noise);

  eskf->nominal.pn = gnss->pos_ned_m[0];
  eskf->nominal.pe = gnss->pos_ned_m[1];
  eskf->nominal.pd = gnss->pos_ned_m[2];
  eskf->nominal.vn = gnss->vel_ned_mps[0];
  eskf->nominal.ve = gnss->vel_ned_mps[1];
  eskf->nominal.vd = gnss->vel_ned_mps[2];

  speed_h = sqrtf(gnss->vel_ned_mps[0] * gnss->vel_ned_mps[0] +
                  gnss->vel_ned_mps[1] * gnss->vel_ned_mps[1]);
  if (gnss->heading_valid) {
    yaw_rad = gnss->heading_rad;
  } else if (speed_h >=
             (yaw_init_speed_mps > 1.0f ? yaw_init_speed_mps : 1.0f)) {
    yaw_rad = atan2f(gnss->vel_ned_mps[1], gnss->vel_ned_mps[0]);
  } else {
    yaw_rad = 0.0f;
  }
  sf_set_nominal_yaw_only(&eskf->nominal, yaw_rad);

  {
    const float att_sigma_rad = 2.0f * 3.1415927f / 180.0f;
    const float att_var = att_sigma_rad * att_sigma_rad;
    eskf->p[0][0] = att_var;
    eskf->p[1][1] = att_var;
    eskf->p[2][2] = att_var;
  }

  vel_var = gnss->vel_std_mps[0];
  if (gnss->vel_std_mps[1] > vel_var) {
    vel_var = gnss->vel_std_mps[1];
  }
  if (gnss->vel_std_mps[2] > vel_var) {
    vel_var = gnss->vel_std_mps[2];
  }
  if (vel_var < 0.2f) {
    vel_var = 0.2f;
  }
  vel_var *= vel_var;
  eskf->p[3][3] = vel_var;
  eskf->p[4][4] = vel_var;
  eskf->p[5][5] = vel_var;

  eskf->p[6][6] = (gnss->pos_std_m[0] > 0.5f ? gnss->pos_std_m[0] : 0.5f);
  eskf->p[7][7] = (gnss->pos_std_m[1] > 0.5f ? gnss->pos_std_m[1] : 0.5f);
  eskf->p[8][8] = (gnss->pos_std_m[2] > 0.5f ? gnss->pos_std_m[2] : 0.5f);
  eskf->p[6][6] *= eskf->p[6][6];
  eskf->p[7][7] *= eskf->p[7][7];
  eskf->p[8][8] *= eskf->p[8][8];
  eskf->p[9][9] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
  eskf->p[10][10] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
  eskf->p[11][11] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
  eskf->p[12][12] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
  eskf->p[13][13] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
  eskf->p[14][14] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
}

static void sf_set_nominal_yaw_only(sf_eskf_nominal_state_t *state,
                                    float yaw_rad) {
  const float half = 0.5f * yaw_rad;
  state->q0 = cosf(half);
  state->q1 = 0.0f;
  state->q2 = 0.0f;
  state->q3 = sinf(half);
}

static void sf_clamp_eskf_biases(sf_eskf_t *eskf) {
  const float max_gyro_bias_radps = 1.5f * 3.1415927f / 180.0f;
  const float max_accel_bias_mps2 = 1.5f;

  if (eskf->nominal.bgx > max_gyro_bias_radps)
    eskf->nominal.bgx = max_gyro_bias_radps;
  if (eskf->nominal.bgx < -max_gyro_bias_radps)
    eskf->nominal.bgx = -max_gyro_bias_radps;
  if (eskf->nominal.bgy > max_gyro_bias_radps)
    eskf->nominal.bgy = max_gyro_bias_radps;
  if (eskf->nominal.bgy < -max_gyro_bias_radps)
    eskf->nominal.bgy = -max_gyro_bias_radps;
  if (eskf->nominal.bgz > max_gyro_bias_radps)
    eskf->nominal.bgz = max_gyro_bias_radps;
  if (eskf->nominal.bgz < -max_gyro_bias_radps)
    eskf->nominal.bgz = -max_gyro_bias_radps;

  if (eskf->nominal.bax > max_accel_bias_mps2)
    eskf->nominal.bax = max_accel_bias_mps2;
  if (eskf->nominal.bax < -max_accel_bias_mps2)
    eskf->nominal.bax = -max_accel_bias_mps2;
  if (eskf->nominal.bay > max_accel_bias_mps2)
    eskf->nominal.bay = max_accel_bias_mps2;
  if (eskf->nominal.bay < -max_accel_bias_mps2)
    eskf->nominal.bay = -max_accel_bias_mps2;
  if (eskf->nominal.baz > max_accel_bias_mps2)
    eskf->nominal.baz = max_accel_bias_mps2;
  if (eskf->nominal.baz < -max_accel_bias_mps2)
    eskf->nominal.baz = -max_accel_bias_mps2;
}

static void sf_quat_to_rotmat(const float q[4], float r[3][3]) {
  float n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  float inv = n2 > 1.0e-9f ? 1.0f / sqrtf(n2) : 1.0f;
  float q0 = q[0] * inv;
  float q1 = q[1] * inv;
  float q2 = q[2] * inv;
  float q3 = q[3] * inv;

  r[0][0] = 1.0f - 2.0f * (q2 * q2 + q3 * q3);
  r[0][1] = 2.0f * (q1 * q2 - q0 * q3);
  r[0][2] = 2.0f * (q1 * q3 + q0 * q2);
  r[1][0] = 2.0f * (q1 * q2 + q0 * q3);
  r[1][1] = 1.0f - 2.0f * (q1 * q1 + q3 * q3);
  r[1][2] = 2.0f * (q2 * q3 - q0 * q1);
  r[2][0] = 2.0f * (q1 * q3 - q0 * q2);
  r[2][1] = 2.0f * (q2 * q3 + q0 * q1);
  r[2][2] = 1.0f - 2.0f * (q1 * q1 + q2 * q2);
}

static void sf_transpose3(const float in[3][3], float out[3][3]) {
  float tmp[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      tmp[i][j] = in[j][i];
    }
  }
  memcpy(out, tmp, sizeof(tmp));
}

static void sf_mat3_vec(const float m[3][3], const float v[3], float out[3]) {
  for (int i = 0; i < 3; ++i) {
    out[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static float sf_norm3(const float v[3]) {
  return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static uint32_t sf_profile_stamp(const sf_sensor_fusion_impl_t *impl) {
  if (impl == NULL || impl->profile_now_us == NULL) {
    return 0u;
  }
  return impl->profile_now_us(impl->profile_ctx);
}

static void sf_profile_accumulate(uint32_t *count, uint64_t *total_us,
                                  uint32_t *max_us, uint32_t elapsed_us) {
  if (count == NULL || total_us == NULL || max_us == NULL) {
    return;
  }
  *count += 1u;
  *total_us += (uint64_t)elapsed_us;
  if (elapsed_us > *max_us) {
    *max_us = elapsed_us;
  }
}

static float sf_ema_update(bool *valid, float *prev, float sample,
                           float alpha) {
  if (alpha < 1.0e-4f)
    alpha = 1.0e-4f;
  if (alpha > 1.0f)
    alpha = 1.0f;
  if (*valid) {
    *prev = (1.0f - alpha) * (*prev) + alpha * sample;
  } else {
    *prev = sample;
    *valid = true;
  }
  return *prev;
}

static float sf_wrap_pi(float rad) {
  while (rad <= -3.1415927f) {
    rad += 2.0f * 3.1415927f;
  }
  while (rad > 3.1415927f) {
    rad -= 2.0f * 3.1415927f;
  }
  return rad;
}

static float sf_horiz_speed(const float vel_ned_mps[3]) {
  return sqrtf(vel_ned_mps[0] * vel_ned_mps[0] +
               vel_ned_mps[1] * vel_ned_mps[1]);
}

static void sf_bootstrap_update_gnss_hints(sf_sensor_fusion_impl_t *impl,
                                           const sf_gnss_sample_t *sample) {
  float dt_s;
  float speed_mps;

  if (impl == NULL || sample == NULL) {
    return;
  }

  speed_mps = sf_horiz_speed(sample->vel_ned_mps);
  sf_ema_update(&impl->bootstrap_speed_ema_valid, &impl->bootstrap_speed_ema,
                speed_mps, impl->cfg.bootstrap.ema_alpha);

  if (!impl->bootstrap_prev_gnss_valid) {
    return;
  }

  dt_s = sample->t_s - impl->bootstrap_prev_gnss.t_s;
  if (dt_s <= 1.0e-3f) {
    return;
  }

  {
    float prev_speed_mps =
        sf_horiz_speed(impl->bootstrap_prev_gnss.vel_ned_mps);
    float speed_rate_mps2 = (speed_mps - prev_speed_mps) / dt_s;
    float course_prev_rad = atan2f(impl->bootstrap_prev_gnss.vel_ned_mps[1],
                                   impl->bootstrap_prev_gnss.vel_ned_mps[0]);
    float course_curr_rad =
        atan2f(sample->vel_ned_mps[1], sample->vel_ned_mps[0]);
    float course_rate_radps =
        sf_wrap_pi(course_curr_rad - course_prev_rad) / dt_s;

    sf_ema_update(&impl->bootstrap_speed_rate_ema_valid,
                  &impl->bootstrap_speed_rate_ema, fabsf(speed_rate_mps2),
                  impl->cfg.bootstrap.ema_alpha);
    sf_ema_update(&impl->bootstrap_course_rate_ema_valid,
                  &impl->bootstrap_course_rate_ema, fabsf(course_rate_radps),
                  impl->cfg.bootstrap.ema_alpha);
  }
}

static bool sf_bootstrap_update(sf_sensor_fusion_impl_t *impl,
                                const float accel_b[3],
                                const float gyro_radps[3]) {
  float gyro_norm = sf_norm3(gyro_radps);
  float accel_err = fabsf(sf_norm3(accel_b) - SF_GRAVITY_MSS);
  float gyro_ema =
      sf_ema_update(&impl->bootstrap_gyro_ema_valid, &impl->bootstrap_gyro_ema,
                    gyro_norm, impl->cfg.bootstrap.ema_alpha);
  float accel_ema = sf_ema_update(&impl->bootstrap_accel_err_ema_valid,
                                  &impl->bootstrap_accel_err_ema, accel_err,
                                  impl->cfg.bootstrap.ema_alpha);
  bool low_dynamic = gyro_ema <= impl->cfg.bootstrap.max_gyro_radps &&
                     accel_ema <= impl->cfg.bootstrap.max_accel_norm_err_mps2;
  bool low_speed =
      !impl->bootstrap_speed_ema_valid ||
      impl->bootstrap_speed_ema <= impl->cfg.bootstrap.max_speed_mps;
  bool steady_motion = impl->bootstrap_speed_rate_ema_valid &&
                       impl->bootstrap_course_rate_ema_valid &&
                       impl->bootstrap_speed_rate_ema <=
                           impl->cfg.bootstrap.max_speed_rate_mps2 &&
                       impl->bootstrap_course_rate_ema <=
                           impl->cfg.bootstrap.max_course_rate_radps;
  bool stationary = low_dynamic && (low_speed || steady_motion);

  if (stationary) {
    if (impl->bootstrap_stationary_count < 400U) {
      impl->bootstrap_stationary_accel_sum[0] += accel_b[0];
      impl->bootstrap_stationary_accel_sum[1] += accel_b[1];
      impl->bootstrap_stationary_accel_sum[2] += accel_b[2];
      impl->bootstrap_stationary_count++;
    }
  } else {
    impl->bootstrap_stationary_count = 0U;
    impl->bootstrap_stationary_accel_sum[0] = 0.0f;
    impl->bootstrap_stationary_accel_sum[1] = 0.0f;
    impl->bootstrap_stationary_accel_sum[2] = 0.0f;
  }
  return impl->bootstrap_stationary_count >=
         impl->cfg.bootstrap.stationary_samples;
}

static bool sf_runtime_zero_velocity_active(sf_sensor_fusion_impl_t *impl,
                                            const float accel_b[3],
                                            const float gyro_radps[3]) {
  float gyro_norm;
  float accel_err;
  float gyro_ema;
  float accel_ema;
  float speed_mps;
  bool low_dynamic;
  bool low_speed;

  if (impl == NULL || accel_b == NULL || gyro_radps == NULL) {
    return false;
  }

  gyro_norm = sf_norm3(gyro_radps);
  accel_err = fabsf(sf_norm3(accel_b) - SF_GRAVITY_MSS);
  gyro_ema =
      sf_ema_update(&impl->bootstrap_gyro_ema_valid, &impl->bootstrap_gyro_ema,
                    gyro_norm, impl->cfg.bootstrap.ema_alpha);
  accel_ema = sf_ema_update(&impl->bootstrap_accel_err_ema_valid,
                            &impl->bootstrap_accel_err_ema, accel_err,
                            impl->cfg.bootstrap.ema_alpha);
  low_dynamic = gyro_ema <= impl->cfg.bootstrap.max_gyro_radps &&
                accel_ema <= impl->cfg.bootstrap.max_accel_norm_err_mps2;
  speed_mps = impl->last_gnss_valid ? sf_horiz_speed(impl->last_gnss.vel_ned_mps)
                                    : 0.0f;
  low_speed = impl->last_gnss_valid && speed_mps <= SF_RUNTIME_ZERO_SPEED_MPS;
  return low_dynamic && low_speed;
}

static bool sf_take_interval_summary(sf_sensor_fusion_impl_t *impl, float t0_s,
                                     float t1_s,
                                     sf_align_window_summary_t *summary_out) {
  float inv_n;
  if (impl->interval_imu_count == 0U || summary_out == NULL) {
    return false;
  }
  memset(summary_out, 0, sizeof(*summary_out));
  summary_out->dt = t1_s - t0_s;
  if (summary_out->dt < 1.0e-3f) {
    summary_out->dt = 1.0e-3f;
  }
  inv_n = 1.0f / (float)impl->interval_imu_count;
  summary_out->mean_gyro_b[0] = impl->interval_imu_sum_gyro[0] * inv_n;
  summary_out->mean_gyro_b[1] = impl->interval_imu_sum_gyro[1] * inv_n;
  summary_out->mean_gyro_b[2] = impl->interval_imu_sum_gyro[2] * inv_n;
  summary_out->mean_accel_b[0] = impl->interval_imu_sum_accel[0] * inv_n;
  summary_out->mean_accel_b[1] = impl->interval_imu_sum_accel[1] * inv_n;
  summary_out->mean_accel_b[2] = impl->interval_imu_sum_accel[2] * inv_n;
  memcpy(summary_out->gnss_vel_prev_n, impl->bootstrap_prev_gnss.vel_ned_mps,
         sizeof(summary_out->gnss_vel_prev_n));
  memcpy(summary_out->gnss_vel_curr_n, impl->last_gnss.vel_ned_mps,
         sizeof(summary_out->gnss_vel_curr_n));
  impl->interval_imu_sum_gyro[0] = 0.0f;
  impl->interval_imu_sum_gyro[1] = 0.0f;
  impl->interval_imu_sum_gyro[2] = 0.0f;
  impl->interval_imu_sum_accel[0] = 0.0f;
  impl->interval_imu_sum_accel[1] = 0.0f;
  impl->interval_imu_sum_accel[2] = 0.0f;
  impl->interval_imu_count = 0U;
  return true;
}
