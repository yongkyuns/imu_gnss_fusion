#include "ekf.h"
#include <math.h>
#include <string.h>

#define DEFAULT_P_INIT 1.0f
#define DEFAULT_BIAS_DT_S 0.01f
#define DEFAULT_GYRO_BIAS_SIGMA_DPS 0.15f
#define DEFAULT_ACCEL_BIAS_SIGMA_MPS2 0.25f
#define PI_F32 3.1415927f

static void quat_mult(const float p[4], const float q[4], float r[4]);
static void quat2Rot(const float q[4], float R[3][3]);
static void normalize_quat(float q[4]);
static predict_noise_t default_predict_noise(void);
static void default_p_diag(float out_diag[N_STATES]);

static void ekf_fuse_gps_pos_n(ekf_t *ekf, float pos_n, float R_POS_N);
static void ekf_fuse_gps_pos_e(ekf_t *ekf, float pos_e, float R_POS_E);
static void ekf_fuse_gps_pos_d(ekf_t *ekf, float pos_d, float R_POS_D);
static void ekf_fuse_gps_vel_n(ekf_t *ekf, float vel_n, float R_VEL_N);
static void ekf_fuse_gps_vel_e(ekf_t *ekf, float vel_e, float R_VEL_E);
static void ekf_fuse_gps_vel_d(ekf_t *ekf, float vel_d, float R_VEL_D);
static void ekf_fuse_body_vel_y(ekf_t *ekf, const float R_BODY_VEL);
static void ekf_fuse_body_vel_z(ekf_t *ekf, const float R_BODY_VEL);

void ekf_init(ekf_t *ekf, const float P_diag[N_STATES],
              const predict_noise_t *noise) {
  float p_diag_local[N_STATES];
  memset(ekf, 0, sizeof(ekf_t));
  ekf->state.q0 = 1.0f;
  ekf->noise = noise ? *noise : default_predict_noise();
  default_p_diag(p_diag_local);
  for (int i = 0; i < N_STATES; i++) {
    ekf->P[i][i] = P_diag ? P_diag[i] : p_diag_local[i];
  }
}

void ekf_set_predict_noise(ekf_t *ekf, const predict_noise_t *noise) {
  if (noise) {
    ekf->noise = *noise;
  }
}

void ekf_predict(ekf_t *ekf, const imu_sample_t *imu, ekf_debug_t *debug_out) {
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
  const float g = GRAVITY_MSS;

  if (debug_out) {
    float R[3][3];
    quat2Rot((float *)&ekf->state, R);
    debug_out->dvb_x = dvx - dvx_b + R[2][0] * g * dt;
    debug_out->dvb_y = dvy - dvy_b + R[2][1] * g * dt;
    debug_out->dvb_z = dvz - dvz_b + R[2][2] * g * dt;
  }

#include "generated/prediction_generated.c"

  normalize_quat((float *)&ekf->state);

  float nextP[N_STATES][N_STATES] = {{0}};
  const float (*P)[N_STATES] = ekf->P;

#include "generated/covariance_generated.c"

  for (int i = 0; i < N_STATES; i++) {
    for (int j = 0; j < N_STATES; j++) {
      nextP[j][i] = nextP[i][j];
    }
  }
  memcpy(ekf->P, nextP, sizeof(ekf->P));
}

void ekf_fuse_gps(ekf_t *ekf, const gps_data_t *gps) {
  ekf_fuse_gps_pos_n(ekf, gps->pos_n, gps->R_POS_N);
  ekf_fuse_gps_pos_e(ekf, gps->pos_e, gps->R_POS_E);
  ekf_fuse_gps_pos_d(ekf, gps->pos_d, gps->R_POS_D);
  ekf_fuse_gps_vel_n(ekf, gps->vel_n, gps->R_VEL_N);
  ekf_fuse_gps_vel_e(ekf, gps->vel_e, gps->R_VEL_E);
  ekf_fuse_gps_vel_d(ekf, gps->vel_d, gps->R_VEL_D);
}

void ekf_fuse_body_vel(ekf_t *ekf, const float R_body_vel) {
  ekf_fuse_body_vel_y(ekf, R_body_vel);
  ekf_fuse_body_vel_z(ekf, R_body_vel);
}

static void fuse_measurement(ekf_t *ekf, float innovation,
                             const float H[N_STATES], const float K[N_STATES]) {
  float P_old[N_STATES][N_STATES];
  memcpy(P_old, ekf->P, sizeof(P_old));

  float *state_array = (float *)&ekf->state;

  for (int i = 0; i < N_STATES; i++) {
    state_array[i] += K[i] * innovation;
  }

  normalize_quat(state_array);

  float HP[N_STATES];
  for (int j = 0; j < N_STATES; j++) {
    HP[j] = 0;
    for (int k = 0; k < N_STATES; k++) {
      HP[j] += H[k] * P_old[k][j];
    }
  }

  for (int i = 0; i < N_STATES; i++) {
    for (int j = 0; j < N_STATES; j++) {
      ekf->P[i][j] = P_old[i][j] - K[i] * HP[j];
    }
  }

  for (int i = 0; i < N_STATES; i++) {
    for (int j = i; j < N_STATES; j++) {
      const float temp = (ekf->P[i][j] + ekf->P[j][i]) / 2.0f;
      ekf->P[i][j] = temp;
      ekf->P[j][i] = temp;
    }
  }
}

static void ekf_fuse_gps_pos_n(ekf_t *ekf, float pos_n, float R_POS_N) {
  const float pn = ekf->state.pn;
  const float (*P)[N_STATES] = ekf->P;

  const float innovation = pos_n - pn;

  float H[N_STATES];
  float K[N_STATES];

#include "generated/gps_pos_n_generated.c"
  fuse_measurement(ekf, innovation, H, K);
}

static void ekf_fuse_gps_pos_e(ekf_t *ekf, float pos_e, float R_POS_E) {
  const float pe = ekf->state.pe;
  const float (*P)[N_STATES] = ekf->P;

  const float innovation = pos_e - pe;

  float H[N_STATES];
  float K[N_STATES];

#include "generated/gps_pos_e_generated.c"
  fuse_measurement(ekf, innovation, H, K);
}

static void ekf_fuse_gps_pos_d(ekf_t *ekf, float pos_d, float R_POS_D) {
  const float pd = ekf->state.pd;
  const float (*P)[N_STATES] = ekf->P;

  const float innovation = pos_d - pd;

  float H[N_STATES];
  float K[N_STATES];

#include "generated/gps_pos_d_generated.c"
  fuse_measurement(ekf, innovation, H, K);
}

static void ekf_fuse_gps_vel_n(ekf_t *ekf, float vel_n, float R_VEL_N) {
  const float vn = ekf->state.vn;
  const float (*P)[N_STATES] = ekf->P;

  const float innovation = vel_n - vn;

  float H[N_STATES];
  float K[N_STATES];

#include "generated/gps_vel_n_generated.c"
  fuse_measurement(ekf, innovation, H, K);
}

static void ekf_fuse_gps_vel_e(ekf_t *ekf, float vel_e, float R_VEL_E) {
  const float ve = ekf->state.ve;
  const float (*P)[N_STATES] = ekf->P;

  const float innovation = vel_e - ve;

  float H[N_STATES];
  float K[N_STATES];

#include "generated/gps_vel_e_generated.c"
  fuse_measurement(ekf, innovation, H, K);
}

static void ekf_fuse_gps_vel_d(ekf_t *ekf, float vel_d, float R_VEL_D) {
  const float vd = ekf->state.vd;
  const float (*P)[N_STATES] = ekf->P;

  const float innovation = vel_d - vd;

  float H[N_STATES];
  float K[N_STATES];

#include "generated/gps_vel_d_generated.c"
  fuse_measurement(ekf, innovation, H, K);
}

static void ekf_fuse_body_vel_y(ekf_t *ekf, const float R_BODY_VEL) {
  const float q0 = ekf->state.q0, q1 = ekf->state.q1, q2 = ekf->state.q2,
              q3 = ekf->state.q3;
  const float vn = ekf->state.vn, ve = ekf->state.ve, vd = ekf->state.vd;
  const float (*P)[N_STATES] = ekf->P;

  const float R_T_10 = 2.0f * (q1 * q2 - q0 * q3);
  const float R_T_11 = 1.0f - 2.0f * (q1 * q1 + q3 * q3);
  const float R_T_12 = 2.0f * (q2 * q3 + q0 * q1);
  const float v_body_y = R_T_10 * vn + R_T_11 * ve + R_T_12 * vd;

  const float innovation = -v_body_y;

  float H[N_STATES];
  float K[N_STATES];
#include "generated/body_vel_y_generated.c"
  fuse_measurement(ekf, innovation, H, K);
}

static void ekf_fuse_body_vel_z(ekf_t *ekf, const float R_BODY_VEL) {
  const float q0 = ekf->state.q0, q1 = ekf->state.q1, q2 = ekf->state.q2,
              q3 = ekf->state.q3;
  const float vn = ekf->state.vn, ve = ekf->state.ve, vd = ekf->state.vd;
  const float (*P)[N_STATES] = ekf->P;

  const float R_T_20 = 2.0f * (q1 * q3 + q0 * q2);
  const float R_T_21 = 2.0f * (q2 * q3 - q0 * q1);
  const float R_T_22 = 1.0f - 2.0f * (q1 * q1 + q2 * q2);
  const float v_body_z = R_T_20 * vn + R_T_21 * ve + R_T_22 * vd;

  const float innovation = -v_body_z;

  float H[N_STATES];
  float K[N_STATES];
#include "generated/body_vel_z_generated.c"
  fuse_measurement(ekf, innovation, H, K);
}

static void quat_mult(const float p[4], const float q[4], float r[4]) {
  r[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3];
  r[1] = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2];
  r[2] = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1];
  r[3] = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0];
}

static void quat2Rot(const float q[4], float R[3][3]) {
  const float q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
  const float q1q1 = q1 * q1, q2q2 = q2 * q2, q3q3 = q3 * q3;
  R[0][0] = 1.0f - 2.0f * (q2q2 + q3q3);
  R[0][1] = 2.0f * (q1 * q2 - q0 * q3);
  R[0][2] = 2.0f * (q1 * q3 + q0 * q2);
  R[1][0] = 2.0f * (q1 * q2 + q0 * q3);
  R[1][1] = 1.0f - 2.0f * (q1q1 + q3q3);
  R[1][2] = 2.0f * (q2 * q3 - q0 * q1);
  R[2][0] = 2.0f * (q1 * q3 - q0 * q2);
  R[2][1] = 2.0f * (q2 * q3 + q0 * q1);
  R[2][2] = 1.0f - 2.0f * (q1q1 + q2q2);
}

static void normalize_quat(float q[4]) {
  const float norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];

  if (norm_sq > 1e-6f) {
    const float norm_inv = 1.0f / sqrtf(norm_sq);
    q[0] *= norm_inv;
    q[1] *= norm_inv;
    q[2] *= norm_inv;
    q[3] *= norm_inv;
  } else {
    // If the quaternion is too close to zero, reset to a default orientation
    q[0] = 1.0f;
    q[1] = 0.0f;
    q[2] = 0.0f;
    q[3] = 0.0f;
  }
}

static predict_noise_t default_predict_noise(void) {
  predict_noise_t noise;
  noise.gyro_var = 0.03f;
  noise.accel_var = 12.0f;
  noise.gyro_bias_rw_var = 0.1e-9f;
  noise.accel_bias_rw_var = 1.0e-8f;
  return noise;
}

static void default_p_diag(float out_diag[N_STATES]) {
  const float dt = DEFAULT_BIAS_DT_S;
  const float gyro_sigma_da = (DEFAULT_GYRO_BIAS_SIGMA_DPS * PI_F32 / 180.0f) * dt;
  const float accel_sigma_dv = DEFAULT_ACCEL_BIAS_SIGMA_MPS2 * dt;
  const float var_gyro = gyro_sigma_da * gyro_sigma_da;
  const float var_accel = accel_sigma_dv * accel_sigma_dv;
  for (int i = 0; i < N_STATES; i++) {
    out_diag[i] = DEFAULT_P_INIT;
  }
  out_diag[10] = var_gyro;
  out_diag[11] = var_gyro;
  out_diag[12] = var_gyro;
  out_diag[13] = var_accel;
  out_diag[14] = var_accel;
  out_diag[15] = var_accel;
}
