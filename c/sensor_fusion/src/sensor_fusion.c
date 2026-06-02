#include "sensor_fusion.h"

#include "align.h"
#include "ekf/runtime.h"

#include <math.h>
#include <string.h>

#define SF_EARTH_RADIUS_M 6378137.0
#define SF_GRAVITY_MSS 9.80665f
#define SF_NHC_MIN_SPEED_MPS 0.05f
#define SF_NHC_MAX_GYRO_NORM_RADPS 0.2f
#define SF_NHC_MAX_ACCEL_NORM_ERR_MPS2 1.0f
#define SF_PI 3.14159265358979323846f
#define SF_GNSS_POS_MIN_STD_M 0.1f
#define SF_GNSS_VEL_MIN_STD_MPS 0.01f
#define SF_GNSS_VERTICAL_POS_STD_SCALE 2.5f
#define SF_MANUAL_MOUNT_SEED_SIGMA_RAD (3.0f * SF_PI / 180.0f)
#define SF_MANUAL_MOUNT_YAW_SEED_MIN_SPEED_MPS (20.0f / 3.6f)
#define SF_CAN_SPEED_ZERO_MPS 0.15f
#define SF_CAN_SPEED_SIGN_INFER_MIN_MPS 1.0f
#define SF_SHORT_SLEEP_MAX_S (15.0f * 60.0f)
#define SF_MEDIUM_SLEEP_MAX_S 3600.0f
#define SF_UNEXPECTED_STREAM_GAP_RESEED_MIN_S 1.0f
#define SF_NAV_USABLE_HORIZONTAL_POS_SIGMA_M 30.0f
#define SF_NAV_USABLE_HORIZONTAL_VEL_SIGMA_MPS 2.5f
#define SF_NAV_USABLE_YAW_SIGMA_RAD (15.0f * SF_PI / 180.0f)
#define SF_NAV_USABLE_ROLL_PITCH_SIGMA_RAD (5.0f * SF_PI / 180.0f)
#define SF_TILT_INIT_EMA_ALPHA 0.05f
#define SF_TILT_INIT_MAX_SPEED_MPS 0.35f
#define SF_TILT_INIT_MAX_SPEED_RATE_MPS2 0.15f
#define SF_TILT_INIT_MAX_COURSE_RATE_RADPS (1.0f * SF_PI / 180.0f)
#define SF_TILT_INIT_STATIONARY_SAMPLES 100u
#define SF_TILT_INIT_MAX_SAMPLES 400u
#define SF_AUTO_YAW_SEED_MIN_SPEED_MPS 1.0f

typedef struct sf_ema {
    float value;
    int valid;
} sf_ema_t;

typedef struct sensor_fusion_private {
    sf_ekf_runtime_state_t ekf;
    sf_align_t align;
    float r_body_vel_y;
    float r_body_vel_z;
    float r_vehicle_roll_prior;
    float r_vehicle_speed;
    float nhc_update_period_s;
    float yaw_init_sigma_rad;
    float mount_init_sigma_rad;
    float last_nhc_t_s;
    float last_nhc_obs_dt_s;
    int has_last_nhc_t_s;
    sf_ekf_gnss_ned_sample_t pending_gnss;
    int has_pending_gnss;
    float last_gnss_fuse_t_s;
    int has_last_gnss_fuse_t_s;
    int preserve_attitude_on_reseed;
    int align_initialized;
    float align_ready_since_t_s;
    int has_align_ready_since_t_s;
    sf_ekf_gnss_ned_sample_t align_prev_gnss;
    int has_align_prev_gnss;
    float interval_imu_sum_gyro[3];
    float interval_imu_sum_accel[3];
    uint32_t interval_imu_count;
    sf_ema_t tilt_init_speed_ema;
    sf_ema_t tilt_init_speed_rate_ema;
    sf_ema_t tilt_init_course_rate_ema;
    sf_ema_t tilt_init_gyro_ema;
    sf_ema_t tilt_init_accel_err_ema;
    float tilt_init_accel_sum[3];
    uint32_t tilt_init_sample_count;
    sensor_fusion_state_t state;
} sensor_fusion_private_t;

typedef char sf_private_storage_size_check
    [(sizeof(sensor_fusion_private_t) <= SENSOR_FUSION_PRIVATE_STORAGE_SIZE) ? 1 : -1];

static sensor_fusion_private_t *sf_private(sensor_fusion_t *fusion) {
    return (sensor_fusion_private_t *)fusion->private_storage.bytes;
}

static const sensor_fusion_private_t *sf_private_const(const sensor_fusion_t *fusion) {
    return (const sensor_fusion_private_t *)fusion->private_storage.bytes;
}

static void sf_sample_to_local_ned(const sensor_fusion_t *fusion,
                                   const sensor_fusion_gnss_sample_t *sample,
                                   float out_pos_ned_m[3]);

static void sf_set_identity_mount(float q[4]) {
    q[0] = 1.0f;
    q[1] = 0.0f;
    q[2] = 0.0f;
    q[3] = 0.0f;
}

static void sf_normalize_quat(float q[4]) {
    float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!(n > 0.0f) || !isfinite(n)) {
        sf_set_identity_mount(q);
        return;
    }
    q[0] /= n;
    q[1] /= n;
    q[2] /= n;
    q[3] /= n;
}

static void sf_quat_from_yaw(float yaw_rad, float q[4]) {
    float half = 0.5f * yaw_rad;
    q[0] = cosf(half);
    q[1] = 0.0f;
    q[2] = 0.0f;
    q[3] = sinf(half);
}

static float sf_sq(float value) {
    return value * value;
}

static float sf_sleep_gap_scale(float gap_s, float max_gap_s) {
    if (!isfinite(gap_s) || !isfinite(max_gap_s) || max_gap_s <= 0.0f) {
        return 0.0f;
    }
    return sqrtf(fminf(fmaxf(gap_s, 0.0f) / max_gap_s, 1.0f));
}

static void sf_add_covariance_sigma(sf_ekf_runtime_state_t *ekf,
                                    unsigned int index,
                                    float sigma) {
    if (index < SF_EKF_ERROR_STATES && sigma > 0.0f && isfinite(sigma)) {
        ekf->p[index][index] += sigma * sigma;
    }
}

static int sf_nav_covariance_usable(const sensor_fusion_private_t *priv) {
    const float(*p)[SF_EKF_ERROR_STATES] = priv->ekf.p;
    float pos_h_sigma_m = sqrtf(fmaxf(fmaxf(p[6][6], 0.0f), fmaxf(p[7][7], 0.0f)));
    float vel_h_sigma_mps = sqrtf(fmaxf(fmaxf(p[3][3], 0.0f), fmaxf(p[4][4], 0.0f)));
    float yaw_sigma_rad = sqrtf(fmaxf(p[2][2], 0.0f));
    float roll_pitch_sigma_rad = sqrtf(fmaxf(fmaxf(p[0][0], 0.0f), fmaxf(p[1][1], 0.0f)));
    return pos_h_sigma_m <= SF_NAV_USABLE_HORIZONTAL_POS_SIGMA_M &&
           vel_h_sigma_mps <= SF_NAV_USABLE_HORIZONTAL_VEL_SIGMA_MPS &&
           yaw_sigma_rad <= SF_NAV_USABLE_YAW_SIGMA_RAD &&
           roll_pitch_sigma_rad <= SF_NAV_USABLE_ROLL_PITCH_SIGMA_RAD;
}

static void sf_age_covariance_for_short_sleep(sensor_fusion_private_t *priv, float gap_s) {
    float scale = sf_sleep_gap_scale(gap_s, SF_SHORT_SLEEP_MAX_S);
    sf_add_covariance_sigma(&priv->ekf, 6u, 2.0f * scale);
    sf_add_covariance_sigma(&priv->ekf, 7u, 2.0f * scale);
    sf_add_covariance_sigma(&priv->ekf, 8u, 1.0f * scale);
    for (unsigned int i = 3u; i < 6u; ++i) {
        sf_add_covariance_sigma(&priv->ekf, i, 0.25f * scale);
    }
    sf_add_covariance_sigma(&priv->ekf, 0u, (0.25f * SF_PI / 180.0f) * scale);
    sf_add_covariance_sigma(&priv->ekf, 1u, (0.25f * SF_PI / 180.0f) * scale);
    sf_add_covariance_sigma(&priv->ekf, 2u, (1.0f * SF_PI / 180.0f) * scale);
    for (unsigned int i = 9u; i < 12u; ++i) {
        sf_add_covariance_sigma(&priv->ekf, i, (0.002f * SF_PI / 180.0f) * scale);
    }
    for (unsigned int i = 12u; i < 15u; ++i) {
        sf_add_covariance_sigma(&priv->ekf, i, 0.01f * scale);
    }
}

static void sf_age_covariance_for_medium_sleep(sensor_fusion_private_t *priv, float gap_s) {
    float clamped_gap_s = fminf(fmaxf(gap_s, 0.0f), SF_MEDIUM_SLEEP_MAX_S);
    float scale = sf_sleep_gap_scale(fmaxf(clamped_gap_s - SF_SHORT_SLEEP_MAX_S, 0.0f),
                                     SF_MEDIUM_SLEEP_MAX_S - SF_SHORT_SLEEP_MAX_S);
    float pos_h_sigma_m = 2.0f + 6.0f * scale;
    float vel_sigma_mps = 0.25f + 0.50f * scale;
    sf_add_covariance_sigma(&priv->ekf, 6u, pos_h_sigma_m);
    sf_add_covariance_sigma(&priv->ekf, 7u, pos_h_sigma_m);
    sf_add_covariance_sigma(&priv->ekf, 8u, 1.0f + 3.0f * scale);
    for (unsigned int i = 3u; i < 6u; ++i) {
        sf_add_covariance_sigma(&priv->ekf, i, vel_sigma_mps);
    }
    sf_add_covariance_sigma(&priv->ekf, 0u, (0.25f + 0.75f * scale) * SF_PI / 180.0f);
    sf_add_covariance_sigma(&priv->ekf, 1u, (0.25f + 0.75f * scale) * SF_PI / 180.0f);
    sf_add_covariance_sigma(&priv->ekf, 2u, (1.0f + 4.0f * scale) * SF_PI / 180.0f);
    for (unsigned int i = 9u; i < 12u; ++i) {
        sf_add_covariance_sigma(&priv->ekf, i, (0.002f + 0.008f * scale) * SF_PI / 180.0f);
    }
    for (unsigned int i = 12u; i < 15u; ++i) {
        sf_add_covariance_sigma(&priv->ekf, i, 0.01f + 0.02f * scale);
    }
}

static void sf_ekf_gnss_sigmas(const float pos_std_m[3],
                               const float vel_std_mps[3],
                               float out_pos_std_m[3],
                               float out_vel_std_mps[3]) {
    float pos_avg =
        fmaxf((pos_std_m[0] + pos_std_m[1] + pos_std_m[2]) / 3.0f, SF_GNSS_POS_MIN_STD_M);
    out_pos_std_m[0] = pos_avg;
    out_pos_std_m[1] = pos_avg;
    out_pos_std_m[2] = SF_GNSS_VERTICAL_POS_STD_SCALE * pos_avg;
    for (unsigned int axis = 0; axis < 3u; ++axis) {
        out_vel_std_mps[axis] = fmaxf(vel_std_mps[axis], SF_GNSS_VEL_MIN_STD_MPS);
    }
}

static float sf_vec_norm3(const float v[3]) {
    return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static float sf_horiz_speed(const float vel_ned_mps[3]) {
    return sqrtf(vel_ned_mps[0] * vel_ned_mps[0] + vel_ned_mps[1] * vel_ned_mps[1]);
}

static float sf_wrap_pi(float rad) {
    float y = fmodf(rad + SF_PI, 2.0f * SF_PI);
    if (y < 0.0f) {
        y += 2.0f * SF_PI;
    }
    return y - SF_PI;
}

static float sf_ema_update(sf_ema_t *ema, float value, float alpha) {
    if (!ema->valid) {
        ema->value = value;
        ema->valid = 1;
    } else {
        ema->value = (1.0f - alpha) * ema->value + alpha * value;
    }
    return ema->value;
}

static int sf_gnss_can_seed_yaw(const sensor_fusion_gnss_sample_t *sample) {
    float speed_h = sqrtf(sample->vel_ned_mps[0] * sample->vel_ned_mps[0] +
                          sample->vel_ned_mps[1] * sample->vel_ned_mps[1]);
    return sample->has_heading_rad && speed_h > SF_MANUAL_MOUNT_YAW_SEED_MIN_SPEED_MPS;
}

static void sf_reset_interval_summary(sensor_fusion_private_t *priv) {
    memset(priv->interval_imu_sum_gyro, 0, sizeof(priv->interval_imu_sum_gyro));
    memset(priv->interval_imu_sum_accel, 0, sizeof(priv->interval_imu_sum_accel));
    priv->interval_imu_count = 0u;
}

static void sf_reset_align_scheduler(sensor_fusion_private_t *priv) {
    priv->has_align_prev_gnss = 0;
    priv->has_align_ready_since_t_s = 0;
    sf_reset_interval_summary(priv);
}

static void sf_accumulate_interval_imu(sensor_fusion_private_t *priv,
                                       const sensor_fusion_imu_sample_t *sample) {
    for (unsigned int axis = 0u; axis < 3u; ++axis) {
        priv->interval_imu_sum_gyro[axis] += sample->gyro_radps[axis];
        priv->interval_imu_sum_accel[axis] += sample->accel_mps2[axis];
    }
    priv->interval_imu_count += 1u;
}

static int sf_take_align_window(sensor_fusion_private_t *priv,
                                const sf_ekf_gnss_ned_sample_t *prev,
                                const sf_ekf_gnss_ned_sample_t *curr,
                                sf_align_window_summary_t *out) {
    if (priv->interval_imu_count == 0u) {
        return 0;
    }
    float inv = 1.0f / (float)priv->interval_imu_count;
    memset(out, 0, sizeof(*out));
    out->dt = fmaxf(curr->t_s - prev->t_s, 1.0e-3f);
    for (unsigned int axis = 0u; axis < 3u; ++axis) {
        out->mean_gyro_b[axis] = priv->interval_imu_sum_gyro[axis] * inv;
        out->mean_accel_b[axis] = priv->interval_imu_sum_accel[axis] * inv;
        out->gnss_vel_prev_n[axis] = prev->vel_ned_mps[axis];
        out->gnss_vel_curr_n[axis] = curr->vel_ned_mps[axis];
        out->gnss_vel_prev_std_mps[axis] = prev->vel_std_mps[axis];
        out->gnss_vel_curr_std_mps[axis] = curr->vel_std_mps[axis];
    }
    out->imu_sample_count = priv->interval_imu_count;
    sf_reset_interval_summary(priv);
    return 1;
}

static void sf_body_vector_to_vehicle(const float q_bv[4], const float vector_b[3], float out_v[3]) {
    float q0 = q_bv[0];
    float q1 = q_bv[1];
    float q2 = q_bv[2];
    float q3 = q_bv[3];
    float c00 = 1.0f - 2.0f * q2 * q2 - 2.0f * q3 * q3;
    float c01 = 2.0f * (q1 * q2 - q0 * q3);
    float c02 = 2.0f * (q1 * q3 + q0 * q2);
    float c10 = 2.0f * (q1 * q2 + q0 * q3);
    float c11 = 1.0f - 2.0f * q1 * q1 - 2.0f * q3 * q3;
    float c12 = 2.0f * (q2 * q3 - q0 * q1);
    float c20 = 2.0f * (q1 * q3 - q0 * q2);
    float c21 = 2.0f * (q2 * q3 + q0 * q1);
    float c22 = 1.0f - 2.0f * q1 * q1 - 2.0f * q2 * q2;
    out_v[0] = c00 * vector_b[0] + c10 * vector_b[1] + c20 * vector_b[2];
    out_v[1] = c01 * vector_b[0] + c11 * vector_b[1] + c21 * vector_b[2];
    out_v[2] = c02 * vector_b[0] + c12 * vector_b[1] + c22 * vector_b[2];
}

static float sf_nhc_observation_r_scale(float dt_s) {
    float dt_obs = (dt_s > 0.0f && isfinite(dt_s)) ? fminf(dt_s, 1.0f) : 1.0f;
    return 1.0f / dt_obs;
}

static int sf_nhc_interval_due(sensor_fusion_private_t *priv, float t_s, float fallback_dt_s) {
    float elapsed_s;
    if (priv->nhc_update_period_s <= 0.0f) {
        priv->last_nhc_t_s = t_s;
        priv->last_nhc_obs_dt_s = fallback_dt_s;
        priv->has_last_nhc_t_s = 1;
        return 1;
    }
    if (!priv->has_last_nhc_t_s) {
        priv->last_nhc_t_s = t_s;
        priv->last_nhc_obs_dt_s = fallback_dt_s;
        priv->has_last_nhc_t_s = 1;
        return 1;
    }
    elapsed_s = t_s - priv->last_nhc_t_s;
    if (elapsed_s + 1.0e-4f < priv->nhc_update_period_s) {
        return 0;
    }
    priv->last_nhc_t_s = t_s;
    priv->last_nhc_obs_dt_s = fmaxf(elapsed_s, fallback_dt_s);
    return 1;
}

static int sf_nhc_active(const sensor_fusion_private_t *priv,
                         const sensor_fusion_imu_sample_t *sample) {
    float gyro_v[3];
    float accel_v[3];
    float v_vehicle[3];
    sf_body_vector_to_vehicle(&priv->ekf.nominal.q_bv0, sample->gyro_radps, gyro_v);
    sf_body_vector_to_vehicle(&priv->ekf.nominal.q_bv0, sample->accel_mps2, accel_v);
    sf_ekf_nominal_vehicle_velocity(&priv->ekf.nominal, v_vehicle);
    return sf_vec_norm3(v_vehicle) > SF_NHC_MIN_SPEED_MPS &&
           sf_vec_norm3(gyro_v) < SF_NHC_MAX_GYRO_NORM_RADPS &&
           fabsf(sf_vec_norm3(accel_v) - SF_GRAVITY_MSS) < SF_NHC_MAX_ACCEL_NORM_ERR_MPS2;
}

static void sf_rate_normalize_pending_gnss(sensor_fusion_private_t *priv,
                                           sf_ekf_gnss_ned_sample_t *sample) {
    float dt_s;
    float r_scale;
    float std_scale;
    if (!priv->has_last_gnss_fuse_t_s) {
        return;
    }
    dt_s = sample->t_s - priv->last_gnss_fuse_t_s;
    if (!(dt_s > 0.0f) || !isfinite(dt_s)) {
        return;
    }
    r_scale = 1.0f / fminf(fmaxf(dt_s, 1.0e-3f), 1.0f);
    std_scale = sqrtf(r_scale);
    for (unsigned int axis = 0; axis < 3u; ++axis) {
        sample->pos_std_m[axis] *= std_scale;
    }
}

static uint32_t sf_fuse_pending_gnss_at_imu(sensor_fusion_private_t *priv,
                                            float imu_t_s,
                                            int has_nhc,
                                            float r_body_vel_y,
                                            float r_body_vel_z,
                                            int *used_nhc) {
    sf_ekf_gnss_ned_sample_t sample;
    sf_ekf_gnss_update_result_t result;
    float age_s;
    int use_nhc = 0;
    *used_nhc = 0;
    if (!priv->has_pending_gnss) {
        return 0u;
    }
    sample = priv->pending_gnss;
    age_s = imu_t_s - sample.t_s;
    if (age_s < -1.0e-6f) {
        return 0u;
    }
    priv->has_pending_gnss = 0;
    use_nhc = has_nhc && age_s >= 0.0f && age_s <= 0.05f;
    sf_rate_normalize_pending_gnss(priv, &sample);
    result = sf_ekf_runtime_fuse_gps_nhc_batch(&priv->ekf,
                                               &sample,
                                               use_nhc,
                                               r_body_vel_y,
                                               use_nhc,
                                               r_body_vel_z);
    priv->last_gnss_fuse_t_s = sample.t_s;
    priv->has_last_gnss_fuse_t_s = 1;
    *used_nhc = use_nhc;
    return result.event_mask;
}

static void sf_clear_stream_coupling_after_gap(sensor_fusion_t *fusion,
                                               sensor_fusion_private_t *priv) {
    fusion->has_last_imu = false;
    priv->has_pending_gnss = 0;
    priv->has_last_nhc_t_s = 0;
    sf_reset_align_scheduler(priv);
}

static void sf_enter_navigation_reseed_mode(sensor_fusion_t *fusion,
                                            sensor_fusion_private_t *priv) {
    priv->has_pending_gnss = 0;
    priv->has_last_nhc_t_s = 0;
    priv->preserve_attitude_on_reseed = 0;
    fusion->expected_sleep = false;
    fusion->navigation_usable = false;
    priv->state = SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED;
}

static void sf_enter_navigation_reseed_mode_preserving_attitude(sensor_fusion_t *fusion,
                                                                sensor_fusion_private_t *priv) {
    sf_enter_navigation_reseed_mode(fusion, priv);
    priv->preserve_attitude_on_reseed = 1;
}

static void sf_handle_imu_stream_gap(sensor_fusion_t *fusion,
                                     sensor_fusion_private_t *priv,
                                     float gap_s) {
    int expected_sleep = fusion->expected_sleep;
    fusion->expected_sleep = false;
    sf_clear_stream_coupling_after_gap(fusion, priv);
    if (!isfinite(gap_s) || gap_s < 0.0f) {
        sf_enter_navigation_reseed_mode(fusion, priv);
        return;
    }
    if (!fusion->ekf_initialized) {
        priv->state = SENSOR_FUSION_STATE_INITIALIZING;
        return;
    }
    if (!expected_sleep) {
        if (gap_s >= SF_UNEXPECTED_STREAM_GAP_RESEED_MIN_S) {
            sf_enter_navigation_reseed_mode_preserving_attitude(fusion, priv);
        }
        return;
    }
    sf_ekf_runtime_require_next_gnss_gate_pass(&priv->ekf);
    if (gap_s <= SF_SHORT_SLEEP_MAX_S) {
        sf_age_covariance_for_short_sleep(priv, gap_s);
        priv->state = SENSOR_FUSION_STATE_RUNNING;
        fusion->navigation_usable = 1;
        return;
    }
    if (gap_s <= SF_MEDIUM_SLEEP_MAX_S) {
        sf_age_covariance_for_medium_sleep(priv, gap_s);
        if (sf_nav_covariance_usable(priv)) {
            priv->state = SENSOR_FUSION_STATE_DEGRADED_DEAD_RECKONING;
            fusion->navigation_usable = 1;
        } else {
            sf_enter_navigation_reseed_mode(fusion, priv);
        }
        return;
    }
    sf_enter_navigation_reseed_mode(fusion, priv);
}

static void sf_copy_runtime_snapshot(sensor_fusion_t *fusion) {
    const sensor_fusion_private_t *priv = sf_private_const(fusion);
    const sf_ekf_nominal_state_t *nominal = &priv->ekf.nominal;
    fusion->ekf.q_nv[0] = nominal->q0;
    fusion->ekf.q_nv[1] = nominal->q1;
    fusion->ekf.q_nv[2] = nominal->q2;
    fusion->ekf.q_nv[3] = nominal->q3;
    fusion->ekf.vel_ned_mps[0] = nominal->vn;
    fusion->ekf.vel_ned_mps[1] = nominal->ve;
    fusion->ekf.vel_ned_mps[2] = nominal->vd;
    fusion->ekf.pos_ned_m[0] = nominal->pn;
    fusion->ekf.pos_ned_m[1] = nominal->pe;
    fusion->ekf.pos_ned_m[2] = nominal->pd;
    fusion->ekf.gyro_bias_b_radps[0] = nominal->bgx;
    fusion->ekf.gyro_bias_b_radps[1] = nominal->bgy;
    fusion->ekf.gyro_bias_b_radps[2] = nominal->bgz;
    fusion->ekf.accel_bias_b_mps2[0] = nominal->bax;
    fusion->ekf.accel_bias_b_mps2[1] = nominal->bay;
    fusion->ekf.accel_bias_b_mps2[2] = nominal->baz;
    fusion->ekf.q_bv[0] = nominal->q_bv0;
    fusion->ekf.q_bv[1] = nominal->q_bv1;
    fusion->ekf.q_bv[2] = nominal->q_bv2;
    fusion->ekf.q_bv[3] = nominal->q_bv3;
    memcpy(fusion->ekf.covariance, priv->ekf.p, sizeof(fusion->ekf.covariance));
}

static void sf_init_runtime_from_public_state(sensor_fusion_t *fusion,
                                              const sensor_fusion_gnss_sample_t *sample) {
    sensor_fusion_private_t *priv = sf_private(fusion);
    float yaw = 0.0f;
    float q_nv[4];
    float pos_std_m[3];
    float vel_std_mps[3];
    float vel_std;
    float mount_seed_var = sf_sq(SF_MANUAL_MOUNT_SEED_SIGMA_RAD);
    if (sample->has_heading_rad) {
        yaw = sample->heading_rad;
    } else {
        float horiz_speed =
            sqrtf(sample->vel_ned_mps[0] * sample->vel_ned_mps[0] +
                  sample->vel_ned_mps[1] * sample->vel_ned_mps[1]);
        if (horiz_speed >= SF_AUTO_YAW_SEED_MIN_SPEED_MPS) {
            yaw = atan2f(sample->vel_ned_mps[1], sample->vel_ned_mps[0]);
        }
    }
    sf_ekf_runtime_init(&priv->ekf);
    sf_quat_from_yaw(yaw, q_nv);
    priv->ekf.nominal.q0 = q_nv[0];
    priv->ekf.nominal.q1 = q_nv[1];
    priv->ekf.nominal.q2 = q_nv[2];
    priv->ekf.nominal.q3 = q_nv[3];
    priv->ekf.nominal.vn = sample->vel_ned_mps[0];
    priv->ekf.nominal.ve = sample->vel_ned_mps[1];
    priv->ekf.nominal.vd = sample->vel_ned_mps[2];
    sf_sample_to_local_ned(fusion, sample, &priv->ekf.nominal.pn);
    priv->ekf.nominal.q_bv0 = fusion->mount_q_bv[0];
    priv->ekf.nominal.q_bv1 = fusion->mount_q_bv[1];
    priv->ekf.nominal.q_bv2 = fusion->mount_q_bv[2];
    priv->ekf.nominal.q_bv3 = fusion->mount_q_bv[3];
    memset(priv->ekf.p, 0, sizeof(priv->ekf.p));
    sf_ekf_gnss_sigmas(sample->pos_std_m, sample->vel_std_mps, pos_std_m, vel_std_mps);
    priv->ekf.p[0][0] = sf_sq(2.0f * SF_PI / 180.0f);
    priv->ekf.p[1][1] = sf_sq(2.0f * SF_PI / 180.0f);
    priv->ekf.p[2][2] = sf_sq(priv->yaw_init_sigma_rad);
    vel_std = fmaxf(fmaxf(vel_std_mps[0], vel_std_mps[1]), fmaxf(vel_std_mps[2], 0.2f));
    priv->ekf.p[3][3] = sf_sq(vel_std);
    priv->ekf.p[4][4] = sf_sq(vel_std);
    priv->ekf.p[5][5] = sf_sq(vel_std);
    priv->ekf.p[6][6] = sf_sq(fmaxf(pos_std_m[0], 0.5f));
    priv->ekf.p[7][7] = sf_sq(fmaxf(pos_std_m[1], 0.5f));
    priv->ekf.p[8][8] = sf_sq(fmaxf(pos_std_m[2], 0.5f));
    priv->ekf.p[9][9] = sf_sq(0.125f * SF_PI / 180.0f);
    priv->ekf.p[10][10] = priv->ekf.p[9][9];
    priv->ekf.p[11][11] = priv->ekf.p[9][9];
    priv->ekf.p[12][12] = sf_sq(0.15f);
    priv->ekf.p[13][13] = priv->ekf.p[12][12];
    priv->ekf.p[14][14] = priv->ekf.p[12][12];
    if (fusion->cfg.manual_mount) {
        for (unsigned int axis = 0; axis < 3u; ++axis) {
            priv->ekf.p[15u + axis][15u + axis] = mount_seed_var;
        }
    } else if (priv->align_initialized) {
        for (unsigned int row = 0u; row < 3u; ++row) {
            for (unsigned int col = 0u; col < 3u; ++col) {
                priv->ekf.p[15u + row][15u + col] = priv->align.p[row][col];
            }
        }
    } else {
        float mount_auto_var = sf_sq(priv->mount_init_sigma_rad);
        for (unsigned int axis = 0; axis < 3u; ++axis) {
            priv->ekf.p[15u + axis][15u + axis] = mount_auto_var;
        }
    }
    priv->last_gnss_fuse_t_s = sample->t_s;
    priv->has_last_gnss_fuse_t_s = 1;
    priv->has_pending_gnss = 0;
    priv->has_last_nhc_t_s = 0;
    priv->state = SENSOR_FUSION_STATE_RUNNING;
}

static void sf_reseed_runtime_from_gnss_preserving_calibration(
    sensor_fusion_t *fusion,
    const sensor_fusion_gnss_sample_t *sample,
    int preserve_attitude_when_yaw_unobservable) {
    sensor_fusion_private_t *priv = sf_private(fusion);
    sf_ekf_nominal_state_t prev_nominal = priv->ekf.nominal;
    float prev_p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES];
    int preserve_attitude = preserve_attitude_when_yaw_unobservable &&
                            !sf_gnss_can_seed_yaw(sample);
    memcpy(prev_p, priv->ekf.p, sizeof(prev_p));
    fusion->mount_q_bv[0] = prev_nominal.q_bv0;
    fusion->mount_q_bv[1] = prev_nominal.q_bv1;
    fusion->mount_q_bv[2] = prev_nominal.q_bv2;
    fusion->mount_q_bv[3] = prev_nominal.q_bv3;
    sf_init_runtime_from_public_state(fusion, sample);
    priv->ekf.nominal.bgx = prev_nominal.bgx;
    priv->ekf.nominal.bgy = prev_nominal.bgy;
    priv->ekf.nominal.bgz = prev_nominal.bgz;
    priv->ekf.nominal.bax = prev_nominal.bax;
    priv->ekf.nominal.bay = prev_nominal.bay;
    priv->ekf.nominal.baz = prev_nominal.baz;
    if (preserve_attitude) {
        priv->ekf.nominal.q0 = prev_nominal.q0;
        priv->ekf.nominal.q1 = prev_nominal.q1;
        priv->ekf.nominal.q2 = prev_nominal.q2;
        priv->ekf.nominal.q3 = prev_nominal.q3;
        priv->ekf.p[0][0] = fmaxf(priv->ekf.p[0][0], prev_p[0][0]);
        priv->ekf.p[1][1] = fmaxf(priv->ekf.p[1][1], prev_p[1][1]);
        priv->ekf.p[2][2] =
            fmaxf(fmaxf(priv->ekf.p[2][2], prev_p[2][2]), sf_sq(45.0f * SF_PI / 180.0f));
    }
    for (unsigned int i = 9u; i < 12u; ++i) {
        priv->ekf.p[i][i] =
            fmaxf(fmaxf(priv->ekf.p[i][i], prev_p[i][i]), sf_sq(0.03f * SF_PI / 180.0f));
    }
    for (unsigned int i = 12u; i < 15u; ++i) {
        priv->ekf.p[i][i] = fmaxf(fmaxf(priv->ekf.p[i][i], prev_p[i][i]), sf_sq(0.05f));
    }
    for (unsigned int i = 15u; i < 18u; ++i) {
        priv->ekf.p[i][i] =
            fmaxf(fmaxf(priv->ekf.p[i][i], prev_p[i][i]), sf_sq(0.50f * SF_PI / 180.0f));
    }
    priv->preserve_attitude_on_reseed = 0;
    priv->state = SENSOR_FUSION_STATE_RUNNING;
}

static void sf_sample_to_local_ned(const sensor_fusion_t *fusion,
                                   const sensor_fusion_gnss_sample_t *sample,
                                   float out_pos_ned_m[3]) {
    double lat0_rad = fusion->anchor_lat_deg * (3.14159265358979323846 / 180.0);
    double d_lat_rad = (sample->lat_deg - fusion->anchor_lat_deg) *
                       (3.14159265358979323846 / 180.0);
    double d_lon_rad = (sample->lon_deg - fusion->anchor_lon_deg) *
                       (3.14159265358979323846 / 180.0);
    out_pos_ned_m[0] = (float)(d_lat_rad * SF_EARTH_RADIUS_M);
    out_pos_ned_m[1] = (float)(d_lon_rad * cos(lat0_rad) * SF_EARTH_RADIUS_M);
    out_pos_ned_m[2] = (float)(fusion->anchor_height_m - sample->height_m);
}

static void sf_gnss_to_local_sample(const sensor_fusion_t *fusion,
                                    const sensor_fusion_gnss_sample_t *sample,
                                    sf_ekf_gnss_ned_sample_t *out) {
    memset(out, 0, sizeof(*out));
    out->t_s = sample->t_s;
    sf_sample_to_local_ned(fusion, sample, out->pos_ned_m);
    memcpy(out->vel_ned_mps, sample->vel_ned_mps, sizeof(out->vel_ned_mps));
    sf_ekf_gnss_sigmas(sample->pos_std_m, sample->vel_std_mps, out->pos_std_m, out->vel_std_mps);
}

static void sf_update_tilt_init_gnss_hints(sensor_fusion_private_t *priv,
                                           const sf_ekf_gnss_ned_sample_t *sample) {
    float speed = sf_horiz_speed(sample->vel_ned_mps);
    sf_ema_update(&priv->tilt_init_speed_ema, speed, SF_TILT_INIT_EMA_ALPHA);
    if (!priv->has_align_prev_gnss) {
        return;
    }
    float dt = sample->t_s - priv->align_prev_gnss.t_s;
    if (dt <= 1.0e-3f) {
        return;
    }
    float prev_speed = sf_horiz_speed(priv->align_prev_gnss.vel_ned_mps);
    float speed_rate = (speed - prev_speed) / dt;
    float course_prev = atan2f(priv->align_prev_gnss.vel_ned_mps[1],
                               priv->align_prev_gnss.vel_ned_mps[0]);
    float course_curr = atan2f(sample->vel_ned_mps[1], sample->vel_ned_mps[0]);
    float course_rate = sf_wrap_pi(course_curr - course_prev) / dt;
    sf_ema_update(&priv->tilt_init_speed_rate_ema, fabsf(speed_rate), SF_TILT_INIT_EMA_ALPHA);
    sf_ema_update(&priv->tilt_init_course_rate_ema, fabsf(course_rate), SF_TILT_INIT_EMA_ALPHA);
}

static int sf_update_tilt_init(sensor_fusion_private_t *priv,
                               const float accel_b[3],
                               const float gyro_radps[3]) {
    float gyro_norm = sf_vec_norm3(gyro_radps);
    float accel_err = fabsf(sf_vec_norm3(accel_b) - SF_GRAVITY_MSS);
    float gyro_ema = sf_ema_update(&priv->tilt_init_gyro_ema, gyro_norm, SF_TILT_INIT_EMA_ALPHA);
    float accel_ema =
        sf_ema_update(&priv->tilt_init_accel_err_ema, accel_err, SF_TILT_INIT_EMA_ALPHA);
    int low_dynamic = gyro_ema <= priv->align.cfg.max_stationary_gyro_radps &&
                      accel_ema <= priv->align.cfg.max_stationary_accel_norm_err_mps2;
    int low_speed = !priv->tilt_init_speed_ema.valid ||
                    priv->tilt_init_speed_ema.value <= SF_TILT_INIT_MAX_SPEED_MPS;
    int steady_motion =
        priv->tilt_init_speed_rate_ema.valid && priv->tilt_init_course_rate_ema.valid &&
        priv->tilt_init_speed_rate_ema.value <= SF_TILT_INIT_MAX_SPEED_RATE_MPS2 &&
        priv->tilt_init_course_rate_ema.value <= SF_TILT_INIT_MAX_COURSE_RATE_RADPS;
    if (low_dynamic && (low_speed || steady_motion)) {
        if (priv->tilt_init_sample_count < SF_TILT_INIT_MAX_SAMPLES) {
            for (unsigned int axis = 0u; axis < 3u; ++axis) {
                priv->tilt_init_accel_sum[axis] += accel_b[axis];
            }
            priv->tilt_init_sample_count += 1u;
        }
    } else {
        memset(priv->tilt_init_accel_sum, 0, sizeof(priv->tilt_init_accel_sum));
        priv->tilt_init_sample_count = 0u;
    }
    return priv->tilt_init_sample_count >= SF_TILT_INIT_STATIONARY_SAMPLES;
}

static void sf_try_tilt_init_align(sensor_fusion_t *fusion,
                                   sensor_fusion_private_t *priv,
                                   const sensor_fusion_imu_sample_t *sample) {
    if (fusion->cfg.manual_mount || priv->align_initialized) {
        return;
    }
    if (sf_update_tilt_init(priv, sample->accel_mps2, sample->gyro_radps)) {
        float mean[1][3];
        for (unsigned int axis = 0u; axis < 3u; ++axis) {
            mean[0][axis] =
                priv->tilt_init_accel_sum[axis] / (float)priv->tilt_init_sample_count;
        }
        if (sf_align_initialize_from_stationary(&priv->align, mean, 1u)) {
            priv->align_initialized = 1;
            memcpy(fusion->mount_q_bv, priv->align.q_bv, sizeof(fusion->mount_q_bv));
        }
    }
}

static int sf_align_handoff_ready(sensor_fusion_private_t *priv, int coarse_ready, float t_s) {
    if (!coarse_ready) {
        priv->has_align_ready_since_t_s = 0;
        return 0;
    }
    if (!priv->has_align_ready_since_t_s) {
        priv->align_ready_since_t_s = t_s;
        priv->has_align_ready_since_t_s = 1;
    }
    return t_s - priv->align_ready_since_t_s >= 0.0f;
}

static sensor_fusion_update_t sf_update(const sensor_fusion_t *fusion,
                                        bool mount_ready_changed,
                                        bool nav_started,
                                        uint32_t gnss_event_mask) {
    const sensor_fusion_private_t *priv = sf_private_const(fusion);
    sensor_fusion_update_t out;
    memset(&out, 0, sizeof(out));
    out.state = priv->state;
    out.mount_ready = fusion->mount_ready;
    out.mount_ready_changed = mount_ready_changed;
    out.navigation_usable = fusion->navigation_usable;
    out.navigation_started = nav_started;
    out.has_mount_q_bv = fusion->mount_ready;
    if (out.has_mount_q_bv) {
        memcpy(out.mount_q_bv, fusion->mount_q_bv, sizeof(out.mount_q_bv));
    }
    out.gnss_event_mask = gnss_event_mask;
    return out;
}

sensor_fusion_config_t sensor_fusion_config_default(void) {
    sensor_fusion_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    sf_set_identity_mount(cfg.manual_q_bv);
    return cfg;
}

size_t sensor_fusion_context_size(void) {
    return sizeof(sensor_fusion_t);
}

size_t sensor_fusion_context_alignment(void) {
    struct sensor_fusion_alignment_probe {
        char c;
        sensor_fusion_t value;
    };
    return offsetof(struct sensor_fusion_alignment_probe, value);
}

void sensor_fusion_init(sensor_fusion_t *fusion, sensor_fusion_config_t cfg) {
    sensor_fusion_private_t *priv;
    memset(fusion, 0, sizeof(*fusion));
    priv = sf_private(fusion);
    fusion->cfg = cfg;
    sf_ekf_runtime_init(&priv->ekf);
    sf_align_init(&priv->align, sf_align_config_default());
    priv->r_body_vel_y = 0.5f;
    priv->r_body_vel_z = 0.5f;
    priv->r_vehicle_roll_prior = 0.1f;
    priv->r_vehicle_speed = 0.04f;
    priv->nhc_update_period_s = 0.1f;
    priv->yaw_init_sigma_rad = 6.0f * SF_PI / 180.0f;
    priv->mount_init_sigma_rad = 0.017453292f;
    priv->state = cfg.manual_mount ? SENSOR_FUSION_STATE_INITIALIZING
                                   : SENSOR_FUSION_STATE_NOT_READY;
    sf_set_identity_mount(fusion->mount_q_bv);
    sf_set_identity_mount(fusion->ekf.q_nv);
    sf_set_identity_mount(fusion->ekf.q_bv);
    for (unsigned i = 0; i < SENSOR_FUSION_ERROR_STATES; ++i) {
        fusion->ekf.covariance[i][i] = 1.0f;
    }
    if (cfg.manual_mount) {
        sensor_fusion_set_misalignment(fusion, cfg.manual_q_bv);
    }
}

void sensor_fusion_init_auto(sensor_fusion_t *fusion) {
    sensor_fusion_init(fusion, sensor_fusion_config_default());
}

void sensor_fusion_init_with_mount(sensor_fusion_t *fusion, const float q_bv[4]) {
    sensor_fusion_config_t cfg = sensor_fusion_config_default();
    cfg.manual_mount = true;
    memcpy(cfg.manual_q_bv, q_bv, sizeof(cfg.manual_q_bv));
    sensor_fusion_init(fusion, cfg);
}

void sensor_fusion_set_misalignment(sensor_fusion_t *fusion, const float q_bv[4]) {
    memcpy(fusion->mount_q_bv, q_bv, sizeof(fusion->mount_q_bv));
    sf_normalize_quat(fusion->mount_q_bv);
    memcpy(fusion->ekf.q_bv, fusion->mount_q_bv, sizeof(fusion->ekf.q_bv));
    fusion->mount_ready = true;
}

sensor_fusion_update_t sensor_fusion_end_trip(sensor_fusion_t *fusion) {
    sensor_fusion_private_t *priv = sf_private(fusion);
    fusion->expected_sleep = true;
    priv->has_pending_gnss = 0;
    priv->has_last_nhc_t_s = 0;
    return sf_update(fusion, false, false, 0u);
}

sensor_fusion_update_t sensor_fusion_process_imu(sensor_fusion_t *fusion,
                                                 sensor_fusion_imu_sample_t sample) {
    sensor_fusion_private_t *priv = sf_private(fusion);
    uint32_t event_mask = 0u;
    if (fusion->has_last_imu) {
        float gap = sample.t_s - fusion->last_imu_t_s;
        if ((!isfinite(gap) || gap > 0.05f || gap < 0.0f) && !fusion->ekf_initialized) {
            sf_clear_stream_coupling_after_gap(fusion, priv);
        }
    }
    if (!fusion->cfg.manual_mount && fusion->has_last_imu) {
        sf_accumulate_interval_imu(priv, &sample);
    }
    sf_try_tilt_init_align(fusion, priv, &sample);
    if (fusion->has_last_imu && fusion->ekf_initialized) {
        float dt = sample.t_s - fusion->last_imu_t_s;
        if (dt > 0.05f || dt < 0.0f || !isfinite(dt)) {
            sf_handle_imu_stream_gap(fusion, priv, dt);
            fusion->last_imu_t_s = sample.t_s;
            fusion->has_last_imu = true;
            sf_copy_runtime_snapshot(fusion);
            return sf_update(fusion, false, false, 0u);
        }
        if (dt >= 0.001f && dt <= 0.05f && isfinite(dt)) {
            sf_ekf_imu_delta_t imu = {0};
            int has_nhc = 0;
            int used_nhc = 0;
            float r_body_vel_y = 0.0f;
            float r_body_vel_z = 0.0f;
            float r_scale = 1.0f;
            imu.dax = sample.gyro_radps[0] * dt;
            imu.day = sample.gyro_radps[1] * dt;
            imu.daz = sample.gyro_radps[2] * dt;
            imu.dvx = sample.accel_mps2[0] * dt;
            imu.dvy = sample.accel_mps2[1] * dt;
            imu.dvz = sample.accel_mps2[2] * dt;
            imu.dt = dt;
            sf_ekf_runtime_predict(&priv->ekf, &imu);
            if ((priv->r_body_vel_y > 0.0f || priv->r_body_vel_z > 0.0f) &&
                sf_nhc_active(priv, &sample) && sf_nhc_interval_due(priv, sample.t_s, dt)) {
                has_nhc = 1;
                r_scale = sf_nhc_observation_r_scale(priv->last_nhc_obs_dt_s);
                r_body_vel_y = priv->r_body_vel_y * r_scale;
                r_body_vel_z = priv->r_body_vel_z * r_scale;
            }
            event_mask = sf_fuse_pending_gnss_at_imu(priv,
                                                     sample.t_s,
                                                     has_nhc,
                                                     r_body_vel_y,
                                                     r_body_vel_z,
                                                     &used_nhc);
            if (has_nhc && !used_nhc) {
                sf_ekf_runtime_fuse_body_vel_yz(&priv->ekf, r_body_vel_y, r_body_vel_z);
            }
            if (has_nhc && priv->r_vehicle_roll_prior > 0.0f &&
                isfinite(priv->r_vehicle_roll_prior)) {
                sf_ekf_runtime_fuse_vehicle_roll_prior(&priv->ekf,
                                                       priv->r_vehicle_roll_prior * r_scale);
            }
            sf_copy_runtime_snapshot(fusion);
        }
    }
    fusion->last_imu_t_s = sample.t_s;
    fusion->has_last_imu = true;
    return sf_update(fusion, false, false, event_mask);
}

sensor_fusion_update_t sensor_fusion_process_gnss(sensor_fusion_t *fusion,
                                                  sensor_fusion_gnss_sample_t sample) {
    sensor_fusion_private_t *priv = sf_private(fusion);
    bool nav_started = false;
    bool mount_ready_changed = false;
    uint32_t event_mask = 0u;
    sf_ekf_gnss_ned_sample_t local;
    if (!fusion->has_anchor) {
        fusion->anchor_lat_deg = sample.lat_deg;
        fusion->anchor_lon_deg = sample.lon_deg;
        fusion->anchor_height_m = sample.height_m;
        fusion->has_anchor = true;
    }
    sf_gnss_to_local_sample(fusion, &sample, &local);
    fusion->last_gnss_t_s = sample.t_s;
    fusion->has_last_gnss = true;
    if (!fusion->cfg.manual_mount) {
        int prev_mount_ready = fusion->mount_ready;
        sf_update_tilt_init_gnss_hints(priv, &local);
        if (priv->align_initialized && priv->has_align_prev_gnss) {
            sf_align_window_summary_t summary;
            if (sf_take_align_window(priv, &priv->align_prev_gnss, &local, &summary)) {
                sf_align_update_trace_t trace;
                (void)sf_align_update_window_with_trace(&priv->align, &summary, &trace);
                memcpy(fusion->mount_q_bv, priv->align.q_bv, sizeof(fusion->mount_q_bv));
                fusion->mount_ready =
                    sf_align_handoff_ready(priv, trace.coarse_alignment_ready, local.t_s) != 0;
            }
        }
        priv->align_prev_gnss = local;
        priv->has_align_prev_gnss = 1;
        mount_ready_changed = prev_mount_ready != fusion->mount_ready;
    }
    if (!fusion->mount_ready) {
        sf_copy_runtime_snapshot(fusion);
        return sf_update(fusion, mount_ready_changed, false, event_mask);
    }
    if (!fusion->ekf_initialized && fusion->mount_ready && !sf_gnss_can_seed_yaw(&sample)) {
        if (!fusion->cfg.manual_mount && sf_horiz_speed(sample.vel_ned_mps) >=
                                             SF_AUTO_YAW_SEED_MIN_SPEED_MPS) {
            sf_init_runtime_from_public_state(fusion, &sample);
            fusion->ekf_initialized = true;
            fusion->navigation_usable = true;
            nav_started = true;
        } else {
            /* Manual mount still needs an observable yaw seed. */
        }
    } else if (!fusion->ekf_initialized && fusion->mount_ready) {
        sf_init_runtime_from_public_state(fusion, &sample);
        fusion->ekf_initialized = true;
        fusion->navigation_usable = true;
        nav_started = true;
    } else if (priv->state == SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED &&
               fusion->mount_ready) {
        if (fusion->cfg.manual_mount && !priv->preserve_attitude_on_reseed &&
            !sf_gnss_can_seed_yaw(&sample)) {
            sf_copy_runtime_snapshot(fusion);
            return sf_update(fusion, false, false, event_mask);
        }
        sf_reseed_runtime_from_gnss_preserving_calibration(
            fusion, &sample, priv->preserve_attitude_on_reseed);
        fusion->ekf_initialized = true;
        fusion->navigation_usable = true;
        nav_started = true;
    } else if (fusion->ekf_initialized) {
        priv->pending_gnss = local;
        priv->has_pending_gnss = 1;
    }
    sf_copy_runtime_snapshot(fusion);
    return sf_update(fusion, mount_ready_changed, nav_started, event_mask);
}

sensor_fusion_update_t
sensor_fusion_process_vehicle_speed(sensor_fusion_t *fusion,
                                    sensor_fusion_vehicle_speed_sample_t sample) {
    sensor_fusion_private_t *priv = sf_private(fusion);
    float signed_speed = sample.speed_mps;
    if (fusion->ekf_initialized && fusion->navigation_usable && sample.speed_mps >= 0.0f &&
        isfinite(sample.speed_mps)) {
        if (sample.direction == SENSOR_FUSION_VEHICLE_SPEED_REVERSE) {
            signed_speed = -signed_speed;
            sf_ekf_runtime_fuse_body_speed_x(&priv->ekf, signed_speed, priv->r_vehicle_speed);
        } else if (sample.direction == SENSOR_FUSION_VEHICLE_SPEED_FORWARD) {
            sf_ekf_runtime_fuse_body_speed_x(&priv->ekf, signed_speed, priv->r_vehicle_speed);
        } else if (sample.speed_mps <= SF_CAN_SPEED_ZERO_MPS) {
            sf_ekf_runtime_fuse_zero_vel(&priv->ekf, priv->r_vehicle_speed);
        } else {
            float v_vehicle[3];
            sf_ekf_nominal_vehicle_velocity(&priv->ekf.nominal, v_vehicle);
            if (fabsf(v_vehicle[0]) >= SF_CAN_SPEED_SIGN_INFER_MIN_MPS) {
                signed_speed = copysignf(sample.speed_mps, v_vehicle[0]);
                sf_ekf_runtime_fuse_body_speed_x(&priv->ekf, signed_speed,
                                                 priv->r_vehicle_speed);
            }
        }
        sf_copy_runtime_snapshot(fusion);
    }
    return sf_update(fusion, false, false, 0u);
}

sensor_fusion_health_t sensor_fusion_health(const sensor_fusion_t *fusion) {
    const sensor_fusion_private_t *priv = sf_private_const(fusion);
    sensor_fusion_health_t out;
    memset(&out, 0, sizeof(out));
    out.state = priv->state;
    out.usable = fusion->navigation_usable;
    out.stable = priv->state == SENSOR_FUSION_STATE_STABLE;
    out.degraded = priv->state == SENSOR_FUSION_STATE_DEGRADED ||
                   priv->state == SENSOR_FUSION_STATE_DEGRADED_DEAD_RECKONING ||
                   priv->state == SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED;
    if (!fusion->ekf_initialized) out.reason_mask |= SENSOR_FUSION_HEALTH_REASON_NOT_INITIALIZED;
    if (!fusion->mount_ready) out.reason_mask |= SENSOR_FUSION_HEALTH_REASON_MOUNT_NOT_READY;
    if (priv->state == SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED) {
        out.reason_mask |= SENSOR_FUSION_HEALTH_REASON_SLEEP_GAP |
                           SENSOR_FUSION_HEALTH_REASON_NAV_UNUSABLE;
    } else if (priv->state == SENSOR_FUSION_STATE_DEGRADED_DEAD_RECKONING) {
        out.reason_mask |= SENSOR_FUSION_HEALTH_REASON_SLEEP_GAP;
    }
    return out;
}

bool sensor_fusion_mount_q_bv(const sensor_fusion_t *fusion, float out_q_bv[4]) {
    if (!fusion->mount_ready) return false;
    memcpy(out_q_bv, fusion->mount_q_bv, sizeof(fusion->mount_q_bv));
    return true;
}

bool sensor_fusion_ekf_state(const sensor_fusion_t *fusion, sensor_fusion_ekf_state_t *out_state) {
    const sensor_fusion_private_t *priv = sf_private_const(fusion);
    if (!fusion->ekf_initialized || priv->state == SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED) {
        return false;
    }
    if (out_state) *out_state = fusion->ekf;
    return true;
}

bool sensor_fusion_position_lla(const sensor_fusion_t *fusion, double out_lla[3]) {
    const sensor_fusion_private_t *priv = sf_private_const(fusion);
    double lat0_rad;
    double lat_deg;
    double lon_deg;
    if (!fusion->has_anchor || !fusion->ekf_initialized ||
        priv->state == SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED) {
        return false;
    }
    lat0_rad = fusion->anchor_lat_deg * (3.14159265358979323846 / 180.0);
    lat_deg = fusion->anchor_lat_deg +
              (double)fusion->ekf.pos_ned_m[0] / SF_EARTH_RADIUS_M *
                  (180.0 / 3.14159265358979323846);
    lon_deg = fusion->anchor_lon_deg +
              (double)fusion->ekf.pos_ned_m[1] / (cos(lat0_rad) * SF_EARTH_RADIUS_M) *
                  (180.0 / 3.14159265358979323846);
    out_lla[0] = lat_deg;
    out_lla[1] = lon_deg;
    out_lla[2] = fusion->anchor_height_m - (double)fusion->ekf.pos_ned_m[2];
    return true;
}

sensor_fusion_align_progress_t sensor_fusion_align_progress(const sensor_fusion_t *fusion) {
    const sensor_fusion_private_t *priv = sf_private_const(fusion);
    sensor_fusion_align_progress_t out;
    memset(&out, 0, sizeof(out));
    if (fusion->mount_ready) {
        out.valid = true;
        out.coarse_ready = true;
        out.progress = 1.0f;
        if (fusion->cfg.manual_mount) {
            out.roll_sigma_deg = 0.0f;
            out.pitch_sigma_deg = 0.0f;
            out.yaw_sigma_deg = 0.0f;
        } else {
            float sigma_deg[3];
            sf_align_sigma_deg(&priv->align, sigma_deg);
            out.roll_sigma_deg = sigma_deg[0];
            out.pitch_sigma_deg = sigma_deg[1];
            out.yaw_sigma_deg = sigma_deg[2];
        }
    } else if (!fusion->cfg.manual_mount && priv->align_initialized) {
        float sigma_deg[3];
        sf_align_sigma_deg(&priv->align, sigma_deg);
        out.valid = true;
        out.coarse_ready = sf_align_coarse_alignment_ready(&priv->align);
        out.roll_sigma_deg = sigma_deg[0];
        out.pitch_sigma_deg = sigma_deg[1];
        out.yaw_sigma_deg = sigma_deg[2];
        out.progress = sf_align_coarse_progress(&priv->align);
    }
    return out;
}

void sensor_fusion_set_r_body_vel_yz(sensor_fusion_t *fusion, float r_y, float r_z) {
    sensor_fusion_private_t *priv = sf_private(fusion);
    if (r_y >= 0.0f && isfinite(r_y)) priv->r_body_vel_y = r_y;
    if (r_z >= 0.0f && isfinite(r_z)) priv->r_body_vel_z = r_z;
}
void sensor_fusion_set_r_vehicle_roll_prior(sensor_fusion_t *fusion, float r) {
    sensor_fusion_private_t *priv = sf_private(fusion);
    if (r >= 0.0f && isfinite(r)) priv->r_vehicle_roll_prior = r;
}
void sensor_fusion_set_r_vehicle_speed(sensor_fusion_t *fusion, float r) {
    sensor_fusion_private_t *priv = sf_private(fusion);
    if (r >= 0.0f && isfinite(r)) priv->r_vehicle_speed = r;
}
void sensor_fusion_set_nhc_update_period_s(sensor_fusion_t *fusion, float period_s) {
    sensor_fusion_private_t *priv = sf_private(fusion);
    if (period_s >= 0.0f && isfinite(period_s)) {
        priv->nhc_update_period_s = period_s;
        priv->has_last_nhc_t_s = 0;
    }
}
void sensor_fusion_set_yaw_init_sigma_rad(sensor_fusion_t *fusion, float sigma_rad) {
    sensor_fusion_private_t *priv = sf_private(fusion);
    if (sigma_rad >= 0.0f && isfinite(sigma_rad)) priv->yaw_init_sigma_rad = sigma_rad;
}
void sensor_fusion_set_mount_init_sigma_rad(sensor_fusion_t *fusion, float sigma_rad) {
    sensor_fusion_private_t *priv = sf_private(fusion);
    if (sigma_rad >= 0.0f && isfinite(sigma_rad)) priv->mount_init_sigma_rad = sigma_rad;
}
