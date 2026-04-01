#include "sensor_fusion.h"

#include <math.h>
#include <string.h>

#define SF_DEFAULT_P_INIT 1.0f
#define SF_DEFAULT_BIAS_DT_S 0.01f
#define SF_DEFAULT_GYRO_BIAS_SIGMA_DPS 0.125f
#define SF_DEFAULT_ACCEL_BIAS_SIGMA_MPS2 0.075f
#define SF_PI_F32 3.1415927f

static void sf_quat_to_rot(const float q[4], float r[3][3]);
static void sf_normalize_quat(float q[4]);
static void sf_default_p_diag(float out_diag[SF_N_STATES]);
static void sf_fuse_measurement(sf_ekf_t *ekf,
                                float innovation,
                                const float h[SF_N_STATES],
                                const float k[SF_N_STATES]);
static void sf_ekf_fuse_gps_pos_n(sf_ekf_t *ekf, float pos_n, float r_pos_n);
static void sf_ekf_fuse_gps_pos_e(sf_ekf_t *ekf, float pos_e, float r_pos_e);
static void sf_ekf_fuse_gps_pos_d(sf_ekf_t *ekf, float pos_d, float r_pos_d);
static void sf_ekf_fuse_gps_vel_n(sf_ekf_t *ekf, float vel_n, float r_vel_n);
static void sf_ekf_fuse_gps_vel_e(sf_ekf_t *ekf, float vel_e, float r_vel_e);
static void sf_ekf_fuse_gps_vel_d(sf_ekf_t *ekf, float vel_d, float r_vel_d);
static void sf_ekf_fuse_body_vel_y(sf_ekf_t *ekf, float r_body_vel);
static void sf_ekf_fuse_body_vel_z(sf_ekf_t *ekf, float r_body_vel);

void sf_predict_noise_default(sf_predict_noise_t *cfg) {
  if (cfg == NULL) {
    return;
  }
  cfg->gyro_var = 0.0001f;
  cfg->accel_var = 12.0f;
  cfg->gyro_bias_rw_var = 0.002e-9f;
  cfg->accel_bias_rw_var = 0.2e-9f;
}

void sf_ekf_init(sf_ekf_t *ekf,
                 const float p_diag[SF_N_STATES],
                 const sf_predict_noise_t *noise) {
  float p_diag_local[SF_N_STATES];
  sf_predict_noise_t default_noise;

  if (ekf == NULL) {
    return;
  }

  memset(ekf, 0, sizeof(*ekf));
  ekf->state.q0 = 1.0f;

  sf_predict_noise_default(&default_noise);
  ekf->noise = noise ? *noise : default_noise;

  sf_default_p_diag(p_diag_local);
  for (int i = 0; i < SF_N_STATES; ++i) {
    ekf->p[i][i] = p_diag ? p_diag[i] : p_diag_local[i];
  }
}

void sf_ekf_set_predict_noise(sf_ekf_t *ekf, const sf_predict_noise_t *noise) {
  if (ekf == NULL || noise == NULL) {
    return;
  }
  ekf->noise = *noise;
}

void sf_ekf_predict(sf_ekf_t *ekf,
                    const sf_ekf_imu_delta_t *imu,
                    sf_ekf_debug_t *debug_out) {
  if (ekf == NULL || imu == NULL) {
    return;
  }

  const float gyro_var = ekf->noise.gyro_var;
  const float accel_var = ekf->noise.accel_var;
  const float gyro_bias_rw_var = ekf->noise.gyro_bias_rw_var;
  const float accel_bias_rw_var = ekf->noise.accel_bias_rw_var;
  const float dt = imu->dt;
  const float dt2 = dt * dt;
  const float dAngVar = gyro_var * dt2;
  const float dVelVar = accel_var * dt2;

  const float q0 = ekf->state.q0;
  const float q1 = ekf->state.q1;
  const float q2 = ekf->state.q2;
  const float q3 = ekf->state.q3;
  const float vn = ekf->state.vn;
  const float ve = ekf->state.ve;
  const float vd = ekf->state.vd;
  const float pn = ekf->state.pn;
  const float pe = ekf->state.pe;
  const float pd = ekf->state.pd;
  const float dax_b = ekf->state.dax_b;
  const float day_b = ekf->state.day_b;
  const float daz_b = ekf->state.daz_b;
  const float dvx_b = ekf->state.dvx_b;
  const float dvy_b = ekf->state.dvy_b;
  const float dvz_b = ekf->state.dvz_b;

  const float dax = imu->dax;
  const float day = imu->day;
  const float daz = imu->daz;
  const float dvx = imu->dvx;
  const float dvy = imu->dvy;
  const float dvz = imu->dvz;
  const float g = SF_GRAVITY_MSS;

  if (debug_out != NULL) {
    float r[3][3];
    sf_quat_to_rot((float *)&ekf->state, r);
    debug_out->dvb_x = dvx - dvx_b + r[2][0] * g * dt;
    debug_out->dvb_y = dvy - dvy_b + r[2][1] * g * dt;
    debug_out->dvb_z = dvz - dvz_b + r[2][2] * g * dt;
  }

#include "../generated/prediction_generated.c"

  sf_normalize_quat((float *)&ekf->state);

  {
    float nextP[SF_N_STATES][SF_N_STATES] = {{0}};
    float (*P)[SF_N_STATES] = ekf->p;

#include "../generated/covariance_generated.c"

    for (int i = 0; i < SF_N_STATES; ++i) {
      for (int j = 0; j < SF_N_STATES; ++j) {
        nextP[j][i] = nextP[i][j];
      }
    }
    memcpy(ekf->p, nextP, sizeof(ekf->p));
  }
}

void sf_ekf_fuse_gps(sf_ekf_t *ekf, const sf_gnss_sample_t *gps) {
  if (ekf == NULL || gps == NULL) {
    return;
  }

  sf_ekf_fuse_gps_pos_n(ekf, gps->pos_ned_m[0], gps->pos_std_m[0] * gps->pos_std_m[0]);
  sf_ekf_fuse_gps_pos_e(ekf, gps->pos_ned_m[1], gps->pos_std_m[1] * gps->pos_std_m[1]);
  sf_ekf_fuse_gps_pos_d(ekf, gps->pos_ned_m[2], gps->pos_std_m[2] * gps->pos_std_m[2]);
  sf_ekf_fuse_gps_vel_n(ekf, gps->vel_ned_mps[0], gps->vel_std_mps[0] * gps->vel_std_mps[0]);
  sf_ekf_fuse_gps_vel_e(ekf, gps->vel_ned_mps[1], gps->vel_std_mps[1] * gps->vel_std_mps[1]);
  sf_ekf_fuse_gps_vel_d(ekf, gps->vel_ned_mps[2], gps->vel_std_mps[2] * gps->vel_std_mps[2]);
}

void sf_ekf_fuse_body_vel(sf_ekf_t *ekf, float r_body_vel) {
  if (ekf == NULL) {
    return;
  }
  sf_ekf_fuse_body_vel_y(ekf, r_body_vel);
  sf_ekf_fuse_body_vel_z(ekf, r_body_vel);
}

static void sf_fuse_measurement(sf_ekf_t *ekf,
                                float innovation,
                                const float h[SF_N_STATES],
                                const float k[SF_N_STATES]) {
  float p_old[SF_N_STATES][SF_N_STATES];
  float hp[SF_N_STATES];
  float *state_array = (float *)&ekf->state;

  memcpy(p_old, ekf->p, sizeof(p_old));
  for (int i = 0; i < SF_N_STATES; ++i) {
    state_array[i] += k[i] * innovation;
  }

  sf_normalize_quat(state_array);

  for (int j = 0; j < SF_N_STATES; ++j) {
    hp[j] = 0.0f;
    for (int k_idx = 0; k_idx < SF_N_STATES; ++k_idx) {
      hp[j] += h[k_idx] * p_old[k_idx][j];
    }
  }

  for (int i = 0; i < SF_N_STATES; ++i) {
    for (int j = 0; j < SF_N_STATES; ++j) {
      ekf->p[i][j] = p_old[i][j] - k[i] * hp[j];
    }
  }

  for (int i = 0; i < SF_N_STATES; ++i) {
    for (int j = i; j < SF_N_STATES; ++j) {
      const float temp = 0.5f * (ekf->p[i][j] + ekf->p[j][i]);
      ekf->p[i][j] = temp;
      ekf->p[j][i] = temp;
    }
  }
}

static void sf_ekf_fuse_gps_pos_n(sf_ekf_t *ekf, float pos_n, float r_pos_n) {
  const float pn = ekf->state.pn;
  float (*P)[SF_N_STATES] = ekf->p;
  const float innovation = pos_n - pn;
  float H[SF_N_STATES];
  float K[SF_N_STATES];
#define R_POS_N r_pos_n
#include "../generated/gps_pos_n_generated.c"
#undef R_POS_N
  sf_fuse_measurement(ekf, innovation, H, K);
}

static void sf_ekf_fuse_gps_pos_e(sf_ekf_t *ekf, float pos_e, float r_pos_e) {
  const float pe = ekf->state.pe;
  float (*P)[SF_N_STATES] = ekf->p;
  const float innovation = pos_e - pe;
  float H[SF_N_STATES];
  float K[SF_N_STATES];
#define R_POS_E r_pos_e
#include "../generated/gps_pos_e_generated.c"
#undef R_POS_E
  sf_fuse_measurement(ekf, innovation, H, K);
}

static void sf_ekf_fuse_gps_pos_d(sf_ekf_t *ekf, float pos_d, float r_pos_d) {
  const float pd = ekf->state.pd;
  float (*P)[SF_N_STATES] = ekf->p;
  const float innovation = pos_d - pd;
  float H[SF_N_STATES];
  float K[SF_N_STATES];
#define R_POS_D r_pos_d
#include "../generated/gps_pos_d_generated.c"
#undef R_POS_D
  sf_fuse_measurement(ekf, innovation, H, K);
}

static void sf_ekf_fuse_gps_vel_n(sf_ekf_t *ekf, float vel_n, float r_vel_n) {
  const float vn = ekf->state.vn;
  float (*P)[SF_N_STATES] = ekf->p;
  const float innovation = vel_n - vn;
  float H[SF_N_STATES];
  float K[SF_N_STATES];
#define R_VEL_N r_vel_n
#include "../generated/gps_vel_n_generated.c"
#undef R_VEL_N
  sf_fuse_measurement(ekf, innovation, H, K);
}

static void sf_ekf_fuse_gps_vel_e(sf_ekf_t *ekf, float vel_e, float r_vel_e) {
  const float ve = ekf->state.ve;
  float (*P)[SF_N_STATES] = ekf->p;
  const float innovation = vel_e - ve;
  float H[SF_N_STATES];
  float K[SF_N_STATES];
#define R_VEL_E r_vel_e
#include "../generated/gps_vel_e_generated.c"
#undef R_VEL_E
  sf_fuse_measurement(ekf, innovation, H, K);
}

static void sf_ekf_fuse_gps_vel_d(sf_ekf_t *ekf, float vel_d, float r_vel_d) {
  const float vd = ekf->state.vd;
  float (*P)[SF_N_STATES] = ekf->p;
  const float innovation = vel_d - vd;
  float H[SF_N_STATES];
  float K[SF_N_STATES];
#define R_VEL_D r_vel_d
#include "../generated/gps_vel_d_generated.c"
#undef R_VEL_D
  sf_fuse_measurement(ekf, innovation, H, K);
}

static void sf_ekf_fuse_body_vel_y(sf_ekf_t *ekf, float r_body_vel) {
  const float q0 = ekf->state.q0;
  const float q1 = ekf->state.q1;
  const float q2 = ekf->state.q2;
  const float q3 = ekf->state.q3;
  const float vn = ekf->state.vn;
  const float ve = ekf->state.ve;
  const float vd = ekf->state.vd;
  float (*P)[SF_N_STATES] = ekf->p;
  const float R_T_10 = 2.0f * (q1 * q2 - q0 * q3);
  const float R_T_11 = 1.0f - 2.0f * (q1 * q1 + q3 * q3);
  const float R_T_12 = 2.0f * (q2 * q3 + q0 * q1);
  const float v_body_y = R_T_10 * vn + R_T_11 * ve + R_T_12 * vd;
  const float innovation = -v_body_y;
  float H[SF_N_STATES];
  float K[SF_N_STATES];
#define R_BODY_VEL r_body_vel
#include "../generated/body_vel_y_generated.c"
#undef R_BODY_VEL
  sf_fuse_measurement(ekf, innovation, H, K);
}

static void sf_ekf_fuse_body_vel_z(sf_ekf_t *ekf, float r_body_vel) {
  const float q0 = ekf->state.q0;
  const float q1 = ekf->state.q1;
  const float q2 = ekf->state.q2;
  const float q3 = ekf->state.q3;
  const float vn = ekf->state.vn;
  const float ve = ekf->state.ve;
  const float vd = ekf->state.vd;
  float (*P)[SF_N_STATES] = ekf->p;
  const float R_T_20 = 2.0f * (q1 * q3 + q0 * q2);
  const float R_T_21 = 2.0f * (q2 * q3 - q0 * q1);
  const float R_T_22 = 1.0f - 2.0f * (q1 * q1 + q2 * q2);
  const float v_body_z = R_T_20 * vn + R_T_21 * ve + R_T_22 * vd;
  const float innovation = -v_body_z;
  float H[SF_N_STATES];
  float K[SF_N_STATES];
#define R_BODY_VEL r_body_vel
#include "../generated/body_vel_z_generated.c"
#undef R_BODY_VEL
  sf_fuse_measurement(ekf, innovation, H, K);
}

static void sf_quat_to_rot(const float q[4], float r[3][3]) {
  const float q0 = q[0];
  const float q1 = q[1];
  const float q2 = q[2];
  const float q3 = q[3];
  const float q1q1 = q1 * q1;
  const float q2q2 = q2 * q2;
  const float q3q3 = q3 * q3;

  r[0][0] = 1.0f - 2.0f * (q2q2 + q3q3);
  r[0][1] = 2.0f * (q1 * q2 - q0 * q3);
  r[0][2] = 2.0f * (q1 * q3 + q0 * q2);
  r[1][0] = 2.0f * (q1 * q2 + q0 * q3);
  r[1][1] = 1.0f - 2.0f * (q1q1 + q3q3);
  r[1][2] = 2.0f * (q2 * q3 - q0 * q1);
  r[2][0] = 2.0f * (q1 * q3 - q0 * q2);
  r[2][1] = 2.0f * (q2 * q3 + q0 * q1);
  r[2][2] = 1.0f - 2.0f * (q1q1 + q2q2);
}

static void sf_normalize_quat(float q[4]) {
  const float norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];

  if (norm_sq > 1.0e-6f) {
    const float norm_inv = 1.0f / sqrtf(norm_sq);
    q[0] *= norm_inv;
    q[1] *= norm_inv;
    q[2] *= norm_inv;
    q[3] *= norm_inv;
  } else {
    q[0] = 1.0f;
    q[1] = 0.0f;
    q[2] = 0.0f;
    q[3] = 0.0f;
  }
}

static void sf_default_p_diag(float out_diag[SF_N_STATES]) {
  const float dt = SF_DEFAULT_BIAS_DT_S;
  const float gyro_sigma_da =
      (SF_DEFAULT_GYRO_BIAS_SIGMA_DPS * SF_PI_F32 / 180.0f) * dt;
  const float accel_sigma_dv = SF_DEFAULT_ACCEL_BIAS_SIGMA_MPS2 * dt;
  const float var_gyro = gyro_sigma_da * gyro_sigma_da;
  const float var_accel = accel_sigma_dv * accel_sigma_dv;

  for (int i = 0; i < SF_N_STATES; ++i) {
    out_diag[i] = SF_DEFAULT_P_INIT;
  }

  out_diag[10] = var_gyro;
  out_diag[11] = var_gyro;
  out_diag[12] = var_gyro;
  out_diag[13] = var_accel;
  out_diag[14] = var_accel;
  out_diag[15] = var_accel;
}
