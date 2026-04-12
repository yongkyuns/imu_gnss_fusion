#include "sf_eskf.h"

#include <math.h>
#include <string.h>

#define SF_ESKF_RUNTIME_ZERO_VEL_R_DIAG 0.01f
#ifndef SF_ESKF_DIAG_DISABLE_BODY_VEL_Y_MOUNT
#define SF_ESKF_DIAG_DISABLE_BODY_VEL_Y_MOUNT 0
#endif
#ifndef SF_ESKF_DIAG_DISABLE_BODY_VEL_Z_MOUNT
#define SF_ESKF_DIAG_DISABLE_BODY_VEL_Z_MOUNT 0
#endif
#ifndef SF_ESKF_BODY_VEL_USE_QCS_CONJ
#define SF_ESKF_BODY_VEL_USE_QCS_CONJ 0
#endif
#ifndef SF_ESKF_ENABLE_BODY_VEL_MOUNT_UPDATE
#define SF_ESKF_ENABLE_BODY_VEL_MOUNT_UPDATE 1
#endif

static void sf_eskf_normalize_quat(float q[4]);
static void sf_eskf_normalize_nominal_quat(sf_eskf_t *eskf);
static void sf_eskf_normalize_nominal_mount_quat(sf_eskf_t *eskf);
static float sf_eskf_mount_roll_rad(const sf_eskf_t *eskf);
static void sf_eskf_project_mount_roll(sf_eskf_t *eskf, float roll_rad);
static void
sf_eskf_symmetrize_p(float p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES]);
static void sf_eskf_zero_mount_roll_covariance(sf_eskf_t *eskf);
static void sf_eskf_quat_multiply(const float p[4], const float q[4],
                                  float out[4]);
static void sf_eskf_nominal_vehicle_velocity(const sf_eskf_t *eskf,
                                             float out[3]);
static void sf_eskf_inject_error_state(sf_eskf_t *eskf,
                                       const float dx[SF_ESKF_ERROR_STATES]);
static void
sf_eskf_apply_reset(float p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
                    const float dx[SF_ESKF_ERROR_STATES]);
static void
sf_eskf_apply_reset_block(float p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
                          int offset, const float dtheta[3]);
static void sf_eskf_floor_attitude_covariance(sf_eskf_t *eskf, float sigma_rad);
static void
sf_eskf_fuse_measurement(sf_eskf_t *eskf, float innovation_var,
                         const float k[SF_ESKF_ERROR_STATES],
                         const float dx_injected[SF_ESKF_ERROR_STATES]);
static void sf_eskf_record_update_diag(
    sf_eskf_t *eskf, sf_eskf_update_diag_type_t type, float innovation,
    float innovation_var, const float k[SF_ESKF_ERROR_STATES],
    const float dx[SF_ESKF_ERROR_STATES]);
static void sf_eskf_block_mount_injection(float dx[SF_ESKF_ERROR_STATES]);
static void sf_eskf_fuse_gps_pos_n(sf_eskf_t *eskf, float pos_n, float r_pos_n);
static void sf_eskf_fuse_gps_pos_e(sf_eskf_t *eskf, float pos_e, float r_pos_e);
static void sf_eskf_fuse_gps_pos_d(sf_eskf_t *eskf, float pos_d, float r_pos_d);
static void sf_eskf_fuse_gps_vel_n(sf_eskf_t *eskf, float vel_n, float r_vel_n);
static void sf_eskf_fuse_gps_vel_e(sf_eskf_t *eskf, float vel_e, float r_vel_e);
static void sf_eskf_fuse_gps_vel_d(sf_eskf_t *eskf, float vel_d, float r_vel_d);
static void sf_eskf_fuse_stationary_gravity_x(sf_eskf_t *eskf, float accel_x,
                                              float r_stationary_accel);
static void sf_eskf_fuse_stationary_gravity_y(sf_eskf_t *eskf, float accel_y,
                                              float r_stationary_accel);
static void sf_eskf_fuse_body_speed_x_impl(sf_eskf_t *eskf, float speed_mps,
                                           float r_speed);
static void sf_eskf_body_vel_y_observation(sf_eskf_t *eskf, float r_body_vel,
                                           float h[SF_ESKF_ERROR_STATES],
                                           float k[SF_ESKF_ERROR_STATES],
                                           float *innovation, float *s);
static void sf_eskf_body_vel_z_observation(sf_eskf_t *eskf, float r_body_vel,
                                           float h[SF_ESKF_ERROR_STATES],
                                           float k[SF_ESKF_ERROR_STATES],
                                           float *innovation, float *s);
static void sf_eskf_fuse_body_vel_yz_batch(sf_eskf_t *eskf, float r_body_vel);
static void sf_eskf_predict_noise_default(sf_predict_noise_t *cfg);
static void sf_eskf_predict_covariance_dense(
    float nextP[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    const float F[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    const float G[SF_ESKF_ERROR_STATES][SF_ESKF_NOISE_STATES],
    const float P[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    const float Q[SF_ESKF_NOISE_STATES]);

static void sf_eskf_predict_noise_default(sf_predict_noise_t *cfg) {
  sf_predict_noise_default(cfg);
  cfg->gyro_var = 2.2873113e-7f;
  cfg->accel_var = 2.4504214e-5f;
  cfg->mount_align_rw_var = 1.0e-8f;
}

static void sf_eskf_normalize_nominal_quat(sf_eskf_t *eskf) {
  float q[4] = {
      eskf->nominal.q0,
      eskf->nominal.q1,
      eskf->nominal.q2,
      eskf->nominal.q3,
  };
  sf_eskf_normalize_quat(q);
  eskf->nominal.q0 = q[0];
  eskf->nominal.q1 = q[1];
  eskf->nominal.q2 = q[2];
  eskf->nominal.q3 = q[3];
}

static void sf_eskf_normalize_nominal_mount_quat(sf_eskf_t *eskf) {
  float q[4] = {
      eskf->nominal.qcs0,
      eskf->nominal.qcs1,
      eskf->nominal.qcs2,
      eskf->nominal.qcs3,
  };
  sf_eskf_normalize_quat(q);
  eskf->nominal.qcs0 = q[0];
  eskf->nominal.qcs1 = q[1];
  eskf->nominal.qcs2 = q[2];
  eskf->nominal.qcs3 = q[3];
}

static float sf_eskf_mount_roll_rad(const sf_eskf_t *eskf) {
  const float q0 = eskf->nominal.qcs0;
  const float q1 = eskf->nominal.qcs1;
  const float q2 = eskf->nominal.qcs2;
  const float q3 = eskf->nominal.qcs3;
  return atan2f(2.0f * (q0 * q1 + q2 * q3),
                1.0f - 2.0f * (q1 * q1 + q2 * q2));
}

static void sf_eskf_project_mount_roll(sf_eskf_t *eskf, float roll_rad) {
  sf_eskf_normalize_nominal_mount_quat(eskf);

  const float q0 = eskf->nominal.qcs0;
  const float q1 = eskf->nominal.qcs1;
  const float q2 = eskf->nominal.qcs2;
  const float q3 = eskf->nominal.qcs3;
  const float sin_pitch_raw = 2.0f * (q0 * q2 - q3 * q1);
  const float sin_pitch =
      fminf(1.0f, fmaxf(-1.0f, sin_pitch_raw));
  const float pitch_rad = asinf(sin_pitch);
  const float yaw_rad = atan2f(2.0f * (q0 * q3 + q1 * q2),
                               1.0f - 2.0f * (q2 * q2 + q3 * q3));
  const float cr = cosf(0.5f * roll_rad);
  const float sr = sinf(0.5f * roll_rad);
  const float cp = cosf(0.5f * pitch_rad);
  const float sp = sinf(0.5f * pitch_rad);
  const float cy = cosf(0.5f * yaw_rad);
  const float sy = sinf(0.5f * yaw_rad);

  eskf->nominal.qcs0 = cr * cp * cy + sr * sp * sy;
  eskf->nominal.qcs1 = sr * cp * cy - cr * sp * sy;
  eskf->nominal.qcs2 = cr * sp * cy + sr * cp * sy;
  eskf->nominal.qcs3 = cr * cp * sy - sr * sp * cy;
  sf_eskf_normalize_nominal_mount_quat(eskf);
}

static void sf_eskf_predict_covariance_dense(
    float nextP[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    const float F[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    const float G[SF_ESKF_ERROR_STATES][SF_ESKF_NOISE_STATES],
    const float P[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    const float Q[SF_ESKF_NOISE_STATES]) {
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    for (int j = i; j < SF_ESKF_ERROR_STATES; ++j) {
      float accum = 0.0f;

      for (int a = 0; a < SF_ESKF_ERROR_STATES; ++a) {
        const float fi = F[i][a];
        if (fi == 0.0f) {
          continue;
        }
        for (int b = 0; b < SF_ESKF_ERROR_STATES; ++b) {
          const float fj = F[j][b];
          if (fj != 0.0f) {
            accum += fi * P[a][b] * fj;
          }
        }
      }

      for (int a = 0; a < SF_ESKF_NOISE_STATES; ++a) {
        const float gi = G[i][a];
        const float gj = G[j][a];
        if (gi != 0.0f && gj != 0.0f) {
          accum += gi * Q[a] * gj;
        }
      }

      nextP[i][j] = accum;
      nextP[j][i] = accum;
    }
  }
}

void sf_eskf_init(sf_eskf_t *eskf, const float p_diag[SF_ESKF_ERROR_STATES],
                  const sf_predict_noise_t *noise) {
  sf_predict_noise_t default_noise;

  if (eskf == NULL) {
    return;
  }

  memset(eskf, 0, sizeof(*eskf));
  eskf->nominal.q0 = 1.0f;
  eskf->nominal.qcs0 = 1.0f;

  sf_eskf_predict_noise_default(&default_noise);
  eskf->noise = noise ? *noise : default_noise;

  if (p_diag != NULL) {
    for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
      eskf->p[i][i] = p_diag[i];
    }
  } else {
    for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
      eskf->p[i][i] = 1.0f;
    }
  }
  sf_eskf_zero_mount_roll_covariance(eskf);
}

void sf_eskf_predict_nominal(sf_eskf_t *eskf, const sf_eskf_imu_delta_t *imu) {
  if (eskf == NULL || imu == NULL) {
    return;
  }

  const float q0 = eskf->nominal.q0;
  const float q1 = eskf->nominal.q1;
  const float q2 = eskf->nominal.q2;
  const float q3 = eskf->nominal.q3;
  const float vn = eskf->nominal.vn;
  const float ve = eskf->nominal.ve;
  const float vd = eskf->nominal.vd;
  const float pn = eskf->nominal.pn;
  const float pe = eskf->nominal.pe;
  const float pd = eskf->nominal.pd;
  const float bgx = eskf->nominal.bgx;
  const float bgy = eskf->nominal.bgy;
  const float bgz = eskf->nominal.bgz;
  const float bax = eskf->nominal.bax;
  const float bay = eskf->nominal.bay;
  const float baz = eskf->nominal.baz;
  const float dax = imu->dax;
  const float day = imu->day;
  const float daz = imu->daz;
  const float dvx = imu->dvx;
  const float dvy = imu->dvy;
  const float dvz = imu->dvz;
  const float dt = imu->dt;
  const float g = SF_GRAVITY_MSS;

#include "../generated_eskf/nominal_prediction_generated.c"
  sf_eskf_normalize_nominal_quat(eskf);
}

void sf_eskf_predict(sf_eskf_t *eskf, const sf_eskf_imu_delta_t *imu) {
  float F[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES];
  float G[SF_ESKF_ERROR_STATES][SF_ESKF_NOISE_STATES];
  float Q[SF_ESKF_NOISE_STATES];
  float nextP[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES];
  const float dt = imu->dt;

  if (eskf == NULL || imu == NULL) {
    return;
  }

  sf_eskf_compute_error_transition(F, G, eskf, imu);
  sf_eskf_predict_nominal(eskf, imu);

  Q[0] = eskf->noise.gyro_var * dt * dt;
  Q[1] = Q[0];
  Q[2] = Q[0];
  Q[3] = eskf->noise.accel_var * dt * dt;
  Q[4] = Q[3];
  Q[5] = Q[3];
  Q[6] = eskf->noise.gyro_bias_rw_var * dt;
  Q[7] = Q[6];
  Q[8] = Q[6];
  Q[9] = eskf->noise.accel_bias_rw_var * dt;
  Q[10] = Q[9];
  Q[11] = Q[9];

  memset(nextP, 0, sizeof(nextP));
  Q[12] = 0.0f;
  Q[13] = eskf->noise.mount_align_rw_var * dt;
  Q[14] = Q[13];

  sf_eskf_predict_covariance_dense(nextP, F, G, eskf->p, Q);

  memcpy(eskf->p, nextP, sizeof(eskf->p));
  sf_eskf_symmetrize_p(eskf->p);
  sf_eskf_zero_mount_roll_covariance(eskf);
}

void sf_eskf_fuse_gps(sf_eskf_t *eskf, const sf_gnss_ned_sample_t *gps) {
  if (eskf == NULL || gps == NULL) {
    return;
  }
  sf_eskf_zero_mount_roll_covariance(eskf);

  sf_eskf_fuse_gps_pos_n(eskf, gps->pos_ned_m[0],
                         gps->pos_std_m[0] * gps->pos_std_m[0]);
  sf_eskf_fuse_gps_pos_e(eskf, gps->pos_ned_m[1],
                         gps->pos_std_m[1] * gps->pos_std_m[1]);
  sf_eskf_fuse_gps_pos_d(eskf, gps->pos_ned_m[2],
                         gps->pos_std_m[2] * gps->pos_std_m[2]);
  sf_eskf_fuse_gps_vel_n(eskf, gps->vel_ned_mps[0],
                         gps->vel_std_mps[0] * gps->vel_std_mps[0]);
  sf_eskf_fuse_gps_vel_e(eskf, gps->vel_ned_mps[1],
                         gps->vel_std_mps[1] * gps->vel_std_mps[1]);
  sf_eskf_fuse_gps_vel_d(eskf, gps->vel_ned_mps[2],
                         gps->vel_std_mps[2] * gps->vel_std_mps[2]);
}

void sf_eskf_fuse_body_speed_x(sf_eskf_t *eskf, float speed_mps,
                               float r_speed) {
  if (eskf == NULL) {
    return;
  }
  sf_eskf_zero_mount_roll_covariance(eskf);
  sf_eskf_fuse_body_speed_x_impl(eskf, speed_mps, r_speed);
}

void sf_eskf_fuse_body_vel(sf_eskf_t *eskf, float r_body_vel) {
  if (eskf == NULL) {
    return;
  }
  sf_eskf_zero_mount_roll_covariance(eskf);
  sf_eskf_fuse_body_vel_yz_batch(eskf, r_body_vel);
}

void sf_eskf_fuse_zero_vel(sf_eskf_t *eskf, float r_zero_vel) {
  if (eskf == NULL) {
    return;
  }
  sf_eskf_zero_mount_roll_covariance(eskf);
  sf_eskf_fuse_gps_vel_n(eskf, 0.0f, r_zero_vel);
  sf_eskf_fuse_gps_vel_e(eskf, 0.0f, r_zero_vel);
  sf_eskf_fuse_gps_vel_d(eskf, 0.0f, r_zero_vel);
}

void sf_eskf_fuse_stationary_gravity(sf_eskf_t *eskf,
                                     const float accel_body_mps2[3],
                                     float r_stationary_accel) {
  if (eskf == NULL || accel_body_mps2 == NULL) {
    return;
  }
  sf_eskf_zero_mount_roll_covariance(eskf);
  sf_eskf_fuse_stationary_gravity_x(eskf, accel_body_mps2[0],
                                    r_stationary_accel);
  sf_eskf_fuse_stationary_gravity_y(eskf, accel_body_mps2[1],
                                    r_stationary_accel);
}

void sf_eskf_compute_error_transition(
    float f_out[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    float g_out[SF_ESKF_ERROR_STATES][SF_ESKF_NOISE_STATES],
    const sf_eskf_t *eskf, const sf_eskf_imu_delta_t *imu) {
  if (f_out == NULL || g_out == NULL || eskf == NULL || imu == NULL) {
    return;
  }

  const float q0 = eskf->nominal.q0;
  const float q1 = eskf->nominal.q1;
  const float q2 = eskf->nominal.q2;
  const float q3 = eskf->nominal.q3;
  const float vn = eskf->nominal.vn;
  const float ve = eskf->nominal.ve;
  const float vd = eskf->nominal.vd;
  const float pn = eskf->nominal.pn;
  const float pe = eskf->nominal.pe;
  const float pd = eskf->nominal.pd;
  const float bgx = eskf->nominal.bgx;
  const float bgy = eskf->nominal.bgy;
  const float bgz = eskf->nominal.bgz;
  const float bax = eskf->nominal.bax;
  const float bay = eskf->nominal.bay;
  const float baz = eskf->nominal.baz;
  const float qcs0 = eskf->nominal.qcs0;
  const float qcs1 = eskf->nominal.qcs1;
  const float qcs2 = eskf->nominal.qcs2;
  const float qcs3 = eskf->nominal.qcs3;
  const float dax = imu->dax;
  const float day = imu->day;
  const float daz = imu->daz;
  const float dvx = imu->dvx;
  const float dvy = imu->dvy;
  const float dvz = imu->dvz;
  const float dt = imu->dt;
  const float g_scalar = SF_GRAVITY_MSS;

  (void)vn;
  (void)ve;
  (void)vd;
  (void)pn;
  (void)pe;
  (void)pd;

  memset(f_out, 0, sizeof(float) * SF_ESKF_ERROR_STATES * SF_ESKF_ERROR_STATES);
  memset(g_out, 0, sizeof(float) * SF_ESKF_ERROR_STATES * SF_ESKF_NOISE_STATES);

#define F f_out
#define G g_out
#define g g_scalar
#include "../generated_eskf/error_transition_generated.c"
#undef g

#define g g_scalar
#include "../generated_eskf/error_noise_input_generated.c"
#undef g
#undef G
#undef F
}

static void sf_eskf_normalize_quat(float q[4]) {
  const float n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  if (n2 <= 1.0e-12f) {
    q[0] = 1.0f;
    q[1] = 0.0f;
    q[2] = 0.0f;
    q[3] = 0.0f;
    return;
  }
  const float inv_n = 1.0f / sqrtf(n2);
  q[0] *= inv_n;
  q[1] *= inv_n;
  q[2] *= inv_n;
  q[3] *= inv_n;
}

static void
sf_eskf_symmetrize_p(float p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES]) {
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    for (int j = i + 1; j < SF_ESKF_ERROR_STATES; ++j) {
      const float sym = 0.5f * (p[i][j] + p[j][i]);
      p[i][j] = sym;
      p[j][i] = sym;
    }
  }
}

static void sf_eskf_zero_mount_roll_covariance(sf_eskf_t *eskf) {
  if (eskf == NULL) {
    return;
  }
  for (int j = 0; j < SF_ESKF_ERROR_STATES; ++j) {
    eskf->p[15][j] = 0.0f;
    eskf->p[j][15] = 0.0f;
  }
}

static void sf_eskf_quat_multiply(const float p[4], const float q[4],
                                  float out[4]) {
  out[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3];
  out[1] = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2];
  out[2] = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1];
  out[3] = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0];
}

static void sf_eskf_nominal_vehicle_velocity(const sf_eskf_t *eskf,
                                             float out[3]) {
  const float q0 = eskf->nominal.q0;
  const float q1 = eskf->nominal.q1;
  const float q2 = eskf->nominal.q2;
  const float q3 = eskf->nominal.q3;
  const float qcs0 = eskf->nominal.qcs0;
#if SF_ESKF_BODY_VEL_USE_QCS_CONJ
  const float qcs1 = -eskf->nominal.qcs1;
  const float qcs2 = -eskf->nominal.qcs2;
  const float qcs3 = -eskf->nominal.qcs3;
#else
  const float qcs1 = eskf->nominal.qcs1;
  const float qcs2 = eskf->nominal.qcs2;
  const float qcs3 = eskf->nominal.qcs3;
#endif
  const float vn = eskf->nominal.vn;
  const float ve = eskf->nominal.ve;
  const float vd = eskf->nominal.vd;
  const float vs0 = (1.0f - 2.0f * q2 * q2 - 2.0f * q3 * q3) * vn +
                    2.0f * (q1 * q2 + q0 * q3) * ve +
                    2.0f * (q1 * q3 - q0 * q2) * vd;
  const float vs1 = 2.0f * (q1 * q2 - q0 * q3) * vn +
                    (1.0f - 2.0f * q1 * q1 - 2.0f * q3 * q3) * ve +
                    2.0f * (q2 * q3 + q0 * q1) * vd;
  const float vs2 = 2.0f * (q1 * q3 + q0 * q2) * vn +
                    2.0f * (q2 * q3 - q0 * q1) * ve +
                    (1.0f - 2.0f * q1 * q1 - 2.0f * q2 * q2) * vd;

  out[0] = (1.0f - 2.0f * qcs2 * qcs2 - 2.0f * qcs3 * qcs3) * vs0 +
           2.0f * (qcs1 * qcs2 - qcs0 * qcs3) * vs1 +
           2.0f * (qcs1 * qcs3 + qcs0 * qcs2) * vs2;
  out[1] = 2.0f * (qcs1 * qcs2 + qcs0 * qcs3) * vs0 +
           (1.0f - 2.0f * qcs1 * qcs1 - 2.0f * qcs3 * qcs3) * vs1 +
           2.0f * (qcs2 * qcs3 - qcs0 * qcs1) * vs2;
  out[2] = 2.0f * (qcs1 * qcs3 - qcs0 * qcs2) * vs0 +
           2.0f * (qcs2 * qcs3 + qcs0 * qcs1) * vs1 +
           (1.0f - 2.0f * qcs1 * qcs1 - 2.0f * qcs2 * qcs2) * vs2;
}

static void sf_eskf_inject_error_state(sf_eskf_t *eskf,
                                       const float dx[SF_ESKF_ERROR_STATES]) {
  float dq[4] = {1.0f, 0.5f * dx[0], 0.5f * dx[1], 0.5f * dx[2]};
  float q_old[4] = {
      eskf->nominal.q0,
      eskf->nominal.q1,
      eskf->nominal.q2,
      eskf->nominal.q3,
  };
  float q_new[4];

  sf_eskf_quat_multiply(q_old, dq, q_new);
  memcpy(&eskf->nominal.q0, q_new, sizeof(q_new));
  sf_eskf_normalize_nominal_quat(eskf);

  eskf->nominal.vn += dx[3];
  eskf->nominal.ve += dx[4];
  eskf->nominal.vd += dx[5];
  eskf->nominal.pn += dx[6];
  eskf->nominal.pe += dx[7];
  eskf->nominal.pd += dx[8];
  eskf->nominal.bgx += dx[9];
  eskf->nominal.bgy += dx[10];
  eskf->nominal.bgz += dx[11];
  eskf->nominal.bax += dx[12];
  eskf->nominal.bay += dx[13];
  eskf->nominal.baz += dx[14];

  {
    const float frozen_mount_roll_rad = sf_eskf_mount_roll_rad(eskf);
    float dqcs[4] = {1.0f, 0.0f, 0.5f * dx[16], 0.5f * dx[17]};
    float qcs_old[4] = {
        eskf->nominal.qcs0,
        eskf->nominal.qcs1,
        eskf->nominal.qcs2,
        eskf->nominal.qcs3,
    };
    float qcs_new[4];
    sf_eskf_quat_multiply(dqcs, qcs_old, qcs_new);
    eskf->nominal.qcs0 = qcs_new[0];
    eskf->nominal.qcs1 = qcs_new[1];
    eskf->nominal.qcs2 = qcs_new[2];
    eskf->nominal.qcs3 = qcs_new[3];
    sf_eskf_project_mount_roll(eskf, frozen_mount_roll_rad);
  }
}

static void
sf_eskf_apply_reset(float p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
                    const float dx[SF_ESKF_ERROR_STATES]) {
  const float dx_mount[3] = {0.0f, dx[16], dx[17]};
  sf_eskf_apply_reset_block(p, 0, dx);
  sf_eskf_apply_reset_block(p, 15, dx_mount);
  sf_eskf_symmetrize_p(p);
}

static void
sf_eskf_apply_reset_block(float p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
                          int offset, const float dtheta[3]) {
  const float dtheta_x = dtheta[0];
  const float dtheta_y = dtheta[1];
  const float dtheta_z = dtheta[2];
  float G_reset_theta[3][3];
  float p_aa[3][3];
  float p_ab[3][SF_ESKF_ERROR_STATES - 3];
  float next_aa[3][3] = {{0}};

#include "../generated_eskf/attitude_reset_jacobian_generated.c"

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      p_aa[i][j] = p[offset + i][offset + j];
    }
    for (int j = 0; j < SF_ESKF_ERROR_STATES; ++j) {
      if (j >= offset && j < offset + 3) {
        continue;
      }
      p_ab[i][j < offset ? j : j - 3] = p[offset + i][j];
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        next_aa[i][j] += G_reset_theta[i][k] * p_aa[k][j];
      }
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      float accum = 0.0f;
      for (int k = 0; k < 3; ++k) {
        accum += next_aa[i][k] * G_reset_theta[j][k];
      }
      p[offset + i][offset + j] = accum;
    }
    for (int j = 0; j < SF_ESKF_ERROR_STATES; ++j) {
      if (j >= offset && j < offset + 3) {
        continue;
      }
      float accum = 0.0f;
      for (int k = 0; k < 3; ++k) {
        accum += G_reset_theta[i][k] * p_ab[k][j < offset ? j : j - 3];
      }
      p[offset + i][j] = accum;
      p[j][offset + i] = accum;
    }
  }
}

static void sf_eskf_floor_attitude_covariance(sf_eskf_t *eskf,
                                              float sigma_rad) {
  const float var_floor = sigma_rad * sigma_rad;
  if (eskf->p[0][0] < var_floor) {
    eskf->p[0][0] = var_floor;
  }
  if (eskf->p[1][1] < var_floor) {
    eskf->p[1][1] = var_floor;
  }
  sf_eskf_symmetrize_p(eskf->p);
}

static void
sf_eskf_fuse_measurement(sf_eskf_t *eskf, float innovation_var,
                         const float k[SF_ESKF_ERROR_STATES],
                         const float dx_injected[SF_ESKF_ERROR_STATES]) {
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    for (int j = i; j < SF_ESKF_ERROR_STATES; ++j) {
      const float updated = eskf->p[i][j] - innovation_var * k[i] * k[j];
      eskf->p[i][j] = updated;
      eskf->p[j][i] = updated;
    }
  }

  sf_eskf_inject_error_state(eskf, dx_injected);
  sf_eskf_apply_reset(eskf->p, dx_injected);
  sf_eskf_zero_mount_roll_covariance(eskf);
}

static void sf_eskf_record_update_diag(
    sf_eskf_t *eskf, sf_eskf_update_diag_type_t type, float innovation,
    float innovation_var, const float k[SF_ESKF_ERROR_STATES],
    const float dx[SF_ESKF_ERROR_STATES]) {
  sf_eskf_update_diag_t *diag;
  const unsigned int idx = (unsigned int)type;

  if (eskf == NULL || idx >= SF_ESKF_UPDATE_DIAG_TYPES) {
    return;
  }
  diag = &eskf->update_diag;
  diag->total_updates += 1u;
  diag->type_counts[idx] += 1u;
  diag->sum_dx_mount_yaw[idx] += dx[17];
  diag->sum_abs_dx_mount_yaw[idx] += fabsf(dx[17]);
  diag->sum_innovation[idx] += innovation;
  diag->sum_abs_innovation[idx] += fabsf(innovation);
  diag->last_dx_mount_yaw = dx[17];
  diag->last_k_mount_yaw = k[17];
  diag->last_innovation = innovation;
  diag->last_innovation_var = innovation_var;
  diag->last_type = idx;
}

static void sf_eskf_block_mount_injection(float dx[SF_ESKF_ERROR_STATES]) {
  dx[15] = 0.0f;
  dx[16] = 0.0f;
  dx[17] = 0.0f;
}

static void sf_eskf_fuse_gps_pos_n(sf_eskf_t *eskf, float pos_n,
                                   float r_pos_n) {
  const float pn = eskf->nominal.pn;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = pos_n - pn;
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float S;
  float dx[SF_ESKF_ERROR_STATES];
#define R_POS_N r_pos_n
#include "../generated_eskf/gps_pos_n_generated.c"
#undef R_POS_N
  (void)H;
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_block_mount_injection(dx);
  sf_eskf_record_update_diag(eskf, SF_ESKF_UPDATE_DIAG_GPS_POS, innovation, S,
                             K, dx);
  sf_eskf_fuse_measurement(eskf, S, K, dx);
}

static void sf_eskf_fuse_gps_pos_e(sf_eskf_t *eskf, float pos_e,
                                   float r_pos_e) {
  const float pe = eskf->nominal.pe;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = pos_e - pe;
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float S;
  float dx[SF_ESKF_ERROR_STATES];
#define R_POS_E r_pos_e
#include "../generated_eskf/gps_pos_e_generated.c"
#undef R_POS_E
  (void)H;
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_block_mount_injection(dx);
  sf_eskf_record_update_diag(eskf, SF_ESKF_UPDATE_DIAG_GPS_POS, innovation, S,
                             K, dx);
  sf_eskf_fuse_measurement(eskf, S, K, dx);
}

static void sf_eskf_fuse_gps_pos_d(sf_eskf_t *eskf, float pos_d,
                                   float r_pos_d) {
  const float pd = eskf->nominal.pd;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = pos_d - pd;
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float S;
  float dx[SF_ESKF_ERROR_STATES];
#define R_POS_D r_pos_d
#include "../generated_eskf/gps_pos_d_generated.c"
#undef R_POS_D
  (void)H;
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_block_mount_injection(dx);
  sf_eskf_record_update_diag(eskf, SF_ESKF_UPDATE_DIAG_GPS_POS_D, innovation,
                             S, K, dx);
  sf_eskf_fuse_measurement(eskf, S, K, dx);
}

static void sf_eskf_fuse_gps_vel_n(sf_eskf_t *eskf, float vel_n,
                                   float r_vel_n) {
  const float vn = eskf->nominal.vn;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = vel_n - vn;
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float S;
  float dx[SF_ESKF_ERROR_STATES];
#define R_VEL_N r_vel_n
#include "../generated_eskf/gps_vel_n_generated.c"
#undef R_VEL_N
  (void)H;
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  if (r_vel_n != SF_ESKF_RUNTIME_ZERO_VEL_R_DIAG) {
    sf_eskf_block_mount_injection(dx);
  }
  sf_eskf_record_update_diag(eskf, r_vel_n == SF_ESKF_RUNTIME_ZERO_VEL_R_DIAG
                                       ? SF_ESKF_UPDATE_DIAG_ZERO_VEL
                                       : SF_ESKF_UPDATE_DIAG_GPS_VEL,
                             innovation, S, K, dx);
  sf_eskf_fuse_measurement(eskf, S, K, dx);
}

static void sf_eskf_fuse_gps_vel_e(sf_eskf_t *eskf, float vel_e,
                                   float r_vel_e) {
  const float ve = eskf->nominal.ve;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = vel_e - ve;
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float S;
  float dx[SF_ESKF_ERROR_STATES];
#define R_VEL_E r_vel_e
#include "../generated_eskf/gps_vel_e_generated.c"
#undef R_VEL_E
  (void)H;
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  if (r_vel_e != SF_ESKF_RUNTIME_ZERO_VEL_R_DIAG) {
    sf_eskf_block_mount_injection(dx);
  }
  sf_eskf_record_update_diag(eskf, r_vel_e == SF_ESKF_RUNTIME_ZERO_VEL_R_DIAG
                                       ? SF_ESKF_UPDATE_DIAG_ZERO_VEL
                                       : SF_ESKF_UPDATE_DIAG_GPS_VEL,
                             innovation, S, K, dx);
  sf_eskf_fuse_measurement(eskf, S, K, dx);
}

static void sf_eskf_fuse_gps_vel_d(sf_eskf_t *eskf, float vel_d,
                                   float r_vel_d) {
  const float vd = eskf->nominal.vd;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = vel_d - vd;
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float S;
  float dx[SF_ESKF_ERROR_STATES];
#define R_VEL_D r_vel_d
#include "../generated_eskf/gps_vel_d_generated.c"
#undef R_VEL_D
  (void)H;
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  if (r_vel_d != SF_ESKF_RUNTIME_ZERO_VEL_R_DIAG) {
    sf_eskf_block_mount_injection(dx);
  }
  sf_eskf_record_update_diag(eskf, r_vel_d == SF_ESKF_RUNTIME_ZERO_VEL_R_DIAG
                                       ? SF_ESKF_UPDATE_DIAG_ZERO_VEL_D
                                       : SF_ESKF_UPDATE_DIAG_GPS_VEL_D,
                             innovation, S, K, dx);
  sf_eskf_fuse_measurement(eskf, S, K, dx);
}

static void sf_eskf_fuse_stationary_gravity_x(sf_eskf_t *eskf, float accel_x,
                                              float r_stationary_accel) {
  sf_eskf_floor_attitude_covariance(eskf,
                                    0.10f * (3.14159265358979323846f / 180.0f));
  const float g_scalar = SF_GRAVITY_MSS;
  const float q0 = eskf->nominal.q0;
  const float q1 = eskf->nominal.q1;
  const float q2 = eskf->nominal.q2;
  const float q3 = eskf->nominal.q3;
  const float bax = eskf->nominal.bax;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float gravity_x = 2.0f * (q1 * q3 - q0 * q2) * SF_GRAVITY_MSS;
  const float innovation = (accel_x - bax) - (-gravity_x);
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float S;
  float dx[SF_ESKF_ERROR_STATES];
#define R_STATIONARY_ACCEL r_stationary_accel
#define g g_scalar
#include "../generated_eskf/stationary_accel_x_generated.c"
#undef g
#undef R_STATIONARY_ACCEL
  (void)H;
  eskf->stationary_diag.innovation_x = innovation;
  eskf->stationary_diag.k_theta_x_from_x = K[0];
  eskf->stationary_diag.k_theta_y_from_x = K[1];
  eskf->stationary_diag.k_bax_from_x = K[12];
  eskf->stationary_diag.k_bay_from_x = K[13];
  eskf->stationary_diag.p_theta_x = P[0][0];
  eskf->stationary_diag.p_theta_y = P[1][1];
  eskf->stationary_diag.p_bax = P[12][12];
  eskf->stationary_diag.p_bay = P[13][13];
  eskf->stationary_diag.p_theta_x_bax = P[0][12];
  eskf->stationary_diag.p_theta_y_bay = P[1][13];
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_record_update_diag(eskf, SF_ESKF_UPDATE_DIAG_STATIONARY_X,
                             innovation, S, K, dx);
  sf_eskf_fuse_measurement(eskf, S, K, dx);
}

static void sf_eskf_fuse_stationary_gravity_y(sf_eskf_t *eskf, float accel_y,
                                              float r_stationary_accel) {
  sf_eskf_floor_attitude_covariance(eskf,
                                    0.10f * (3.14159265358979323846f / 180.0f));
  const float g_scalar = SF_GRAVITY_MSS;
  const float q0 = eskf->nominal.q0;
  const float q1 = eskf->nominal.q1;
  const float q2 = eskf->nominal.q2;
  const float q3 = eskf->nominal.q3;
  const float bay = eskf->nominal.bay;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float gravity_y = 2.0f * (q2 * q3 + q0 * q1) * SF_GRAVITY_MSS;
  const float innovation = (accel_y - bay) - (-gravity_y);
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float S;
  float dx[SF_ESKF_ERROR_STATES];
#define R_STATIONARY_ACCEL r_stationary_accel
#define g g_scalar
#include "../generated_eskf/stationary_accel_y_generated.c"
#undef g
#undef R_STATIONARY_ACCEL
  (void)H;
  eskf->stationary_diag.innovation_y = innovation;
  eskf->stationary_diag.k_theta_x_from_y = K[0];
  eskf->stationary_diag.k_theta_y_from_y = K[1];
  eskf->stationary_diag.k_bax_from_y = K[12];
  eskf->stationary_diag.k_bay_from_y = K[13];
  eskf->stationary_diag.p_theta_x = P[0][0];
  eskf->stationary_diag.p_theta_y = P[1][1];
  eskf->stationary_diag.p_bax = P[12][12];
  eskf->stationary_diag.p_bay = P[13][13];
  eskf->stationary_diag.p_theta_x_bax = P[0][12];
  eskf->stationary_diag.p_theta_y_bay = P[1][13];
  eskf->stationary_diag.updates += 1u;
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_record_update_diag(eskf, SF_ESKF_UPDATE_DIAG_STATIONARY_Y,
                             innovation, S, K, dx);
  sf_eskf_fuse_measurement(eskf, S, K, dx);
}

static void sf_eskf_body_vel_y_observation(sf_eskf_t *eskf, float r_body_vel,
                                           float h[SF_ESKF_ERROR_STATES],
                                           float k[SF_ESKF_ERROR_STATES],
                                           float *innovation, float *s) {
  float v_vehicle[3];
  const float q0 = eskf->nominal.q0;
  const float q1 = eskf->nominal.q1;
  const float q2 = eskf->nominal.q2;
  const float q3 = eskf->nominal.q3;
  const float qcs0 = eskf->nominal.qcs0;
#if SF_ESKF_BODY_VEL_USE_QCS_CONJ
  const float qcs1 = -eskf->nominal.qcs1;
  const float qcs2 = -eskf->nominal.qcs2;
  const float qcs3 = -eskf->nominal.qcs3;
#else
  const float qcs1 = eskf->nominal.qcs1;
  const float qcs2 = eskf->nominal.qcs2;
  const float qcs3 = eskf->nominal.qcs3;
#endif
  const float vn = eskf->nominal.vn;
  const float ve = eskf->nominal.ve;
  const float vd = eskf->nominal.vd;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  sf_eskf_nominal_vehicle_velocity(eskf, v_vehicle);
  *innovation = -v_vehicle[1];
#define H h
#define K k
#define S (*s)
#define R_BODY_VEL r_body_vel
#include "../generated_eskf/body_vel_y_generated.c"
#undef R_BODY_VEL
#undef S
#undef K
#undef H
}

static void sf_eskf_body_vel_z_observation(sf_eskf_t *eskf, float r_body_vel,
                                           float h[SF_ESKF_ERROR_STATES],
                                           float k[SF_ESKF_ERROR_STATES],
                                           float *innovation, float *s) {
  float v_vehicle[3];
  const float q0 = eskf->nominal.q0;
  const float q1 = eskf->nominal.q1;
  const float q2 = eskf->nominal.q2;
  const float q3 = eskf->nominal.q3;
  const float qcs0 = eskf->nominal.qcs0;
#if SF_ESKF_BODY_VEL_USE_QCS_CONJ
  const float qcs1 = -eskf->nominal.qcs1;
  const float qcs2 = -eskf->nominal.qcs2;
  const float qcs3 = -eskf->nominal.qcs3;
#else
  const float qcs1 = eskf->nominal.qcs1;
  const float qcs2 = eskf->nominal.qcs2;
  const float qcs3 = eskf->nominal.qcs3;
#endif
  const float vn = eskf->nominal.vn;
  const float ve = eskf->nominal.ve;
  const float vd = eskf->nominal.vd;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  sf_eskf_nominal_vehicle_velocity(eskf, v_vehicle);
  *innovation = -v_vehicle[2];
#define H h
#define K k
#define S (*s)
#define R_BODY_VEL r_body_vel
#include "../generated_eskf/body_vel_z_generated.c"
#undef R_BODY_VEL
#undef S
#undef K
#undef H
}

static void sf_eskf_fuse_body_vel_yz_batch(sf_eskf_t *eskf, float r_body_vel) {
  float h_rows[2][SF_ESKF_ERROR_STATES];
  float k_scalar[2][SF_ESKF_ERROR_STATES];
  float residuals[2];
  float scalar_s[2];
  double p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES];
  double dx[SF_ESKF_ERROR_STATES] = {0.0};

  sf_eskf_body_vel_y_observation(eskf, r_body_vel, h_rows[0], k_scalar[0],
                                 &residuals[0], &scalar_s[0]);
  sf_eskf_body_vel_z_observation(eskf, r_body_vel, h_rows[1], k_scalar[1],
                                 &residuals[1], &scalar_s[1]);

  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    for (int j = 0; j < SF_ESKF_ERROR_STATES; ++j) {
      p[i][j] = (double)eskf->p[i][j];
    }
  }

  for (int obs = 0; obs < 2; ++obs) {
    double ph[SF_ESKF_ERROR_STATES] = {0.0};
    double s = (double)r_body_vel;
    double hd = 0.0;
    double alpha;
    float diag_k[SF_ESKF_ERROR_STATES];
    float diag_dx[SF_ESKF_ERROR_STATES];
    const sf_eskf_update_diag_type_t diag_type =
        obs == 0 ? SF_ESKF_UPDATE_DIAG_BODY_VEL_Y
                 : SF_ESKF_UPDATE_DIAG_BODY_VEL_Z;

    for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
      for (int j = 0; j < SF_ESKF_ERROR_STATES; ++j) {
        ph[i] += p[i][j] * (double)h_rows[obs][j];
      }
    }
    for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
      s += (double)h_rows[obs][i] * ph[i];
      hd += (double)h_rows[obs][i] * dx[i];
    }
    if (!(s > 0.0) || !isfinite(s)) {
      continue;
    }

    alpha = ((double)residuals[obs] - hd) / s;
    for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
      const double gain = ph[i] / s;
      const double contribution = ph[i] * alpha;
      diag_k[i] = (float)gain;
      diag_dx[i] = (float)contribution;
    }

#if !SF_ESKF_ENABLE_BODY_VEL_MOUNT_UPDATE
    sf_eskf_block_mount_injection(diag_dx);
#endif
#if SF_ESKF_DIAG_DISABLE_BODY_VEL_Y_MOUNT
    if (obs == 0) {
      sf_eskf_block_mount_injection(diag_dx);
    }
#endif
#if SF_ESKF_DIAG_DISABLE_BODY_VEL_Z_MOUNT
    if (obs == 1) {
      sf_eskf_block_mount_injection(diag_dx);
    }
#endif

    for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
      dx[i] += (double)diag_dx[i];
    }

    sf_eskf_record_update_diag(eskf, diag_type, residuals[obs], (float)s,
                               diag_k, diag_dx);

    for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
      for (int j = i; j < SF_ESKF_ERROR_STATES; ++j) {
        const double updated = p[i][j] - (ph[i] * ph[j]) / s;
        p[i][j] = updated;
        p[j][i] = updated;
      }
    }
  }

  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    for (int j = 0; j < SF_ESKF_ERROR_STATES; ++j) {
      eskf->p[i][j] = (float)p[i][j];
    }
  }

  {
    float dx_f32[SF_ESKF_ERROR_STATES];
    for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
      dx_f32[i] = (float)dx[i];
    }
    sf_eskf_inject_error_state(eskf, dx_f32);
    sf_eskf_apply_reset(eskf->p, dx_f32);
    sf_eskf_zero_mount_roll_covariance(eskf);
  }

  (void)k_scalar;
  (void)scalar_s;
}

static void sf_eskf_fuse_body_speed_x_impl(sf_eskf_t *eskf, float speed_mps,
                                           float r_speed) {
  float v_vehicle[3];
  const float q0 = eskf->nominal.q0;
  const float q1 = eskf->nominal.q1;
  const float q2 = eskf->nominal.q2;
  const float q3 = eskf->nominal.q3;
  const float qcs0 = eskf->nominal.qcs0;
#if SF_ESKF_BODY_VEL_USE_QCS_CONJ
  const float qcs1 = -eskf->nominal.qcs1;
  const float qcs2 = -eskf->nominal.qcs2;
  const float qcs3 = -eskf->nominal.qcs3;
#else
  const float qcs1 = eskf->nominal.qcs1;
  const float qcs2 = eskf->nominal.qcs2;
  const float qcs3 = eskf->nominal.qcs3;
#endif
  const float vn = eskf->nominal.vn;
  const float ve = eskf->nominal.ve;
  const float vd = eskf->nominal.vd;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  sf_eskf_nominal_vehicle_velocity(eskf, v_vehicle);
  const float innovation = speed_mps - v_vehicle[0];
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float S;
  float dx[SF_ESKF_ERROR_STATES];
#define R_BODY_VEL r_speed
#include "../generated_eskf/body_vel_x_generated.c"
#undef R_BODY_VEL
  (void)H;
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_record_update_diag(eskf, SF_ESKF_UPDATE_DIAG_BODY_SPEED_X,
                             innovation, S, K, dx);
  sf_eskf_fuse_measurement(eskf, S, K, dx);
}
