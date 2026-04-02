#include "sf_eskf.h"

#include <math.h>
#include <string.h>

static void sf_eskf_normalize_quat(float q[4]);
static void sf_eskf_symmetrize_p(float p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES]);
static void sf_eskf_quat_multiply(const float p[4], const float q[4], float out[4]);
static void sf_eskf_inject_error_state(sf_eskf_t *eskf, const float dx[SF_ESKF_ERROR_STATES]);
static void sf_eskf_apply_reset(float p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
                                const float dtheta[3]);
static void sf_eskf_floor_attitude_covariance(sf_eskf_t *eskf, float sigma_rad);
static void sf_eskf_fuse_measurement(sf_eskf_t *eskf,
                                     float innovation,
                                     const float h[SF_ESKF_ERROR_STATES],
                                     const float k[SF_ESKF_ERROR_STATES],
                                     const float dx_injected[SF_ESKF_ERROR_STATES]);
static void sf_eskf_fuse_gps_pos_n(sf_eskf_t *eskf, float pos_n, float r_pos_n);
static void sf_eskf_fuse_gps_pos_e(sf_eskf_t *eskf, float pos_e, float r_pos_e);
static void sf_eskf_fuse_gps_pos_d(sf_eskf_t *eskf, float pos_d, float r_pos_d);
static void sf_eskf_fuse_gps_vel_n(sf_eskf_t *eskf, float vel_n, float r_vel_n);
static void sf_eskf_fuse_gps_vel_e(sf_eskf_t *eskf, float vel_e, float r_vel_e);
static void sf_eskf_fuse_gps_vel_d(sf_eskf_t *eskf, float vel_d, float r_vel_d);
static void sf_eskf_fuse_stationary_gravity_x(sf_eskf_t *eskf, float accel_x, float r_stationary_accel);
static void sf_eskf_fuse_stationary_gravity_y(sf_eskf_t *eskf, float accel_y, float r_stationary_accel);
static void sf_eskf_fuse_body_vel_x(sf_eskf_t *eskf, float r_body_vel);
static void sf_eskf_fuse_body_vel_y(sf_eskf_t *eskf, float r_body_vel);
static void sf_eskf_fuse_body_vel_z(sf_eskf_t *eskf, float r_body_vel);
static void sf_eskf_predict_noise_default(sf_predict_noise_t *cfg);
static void sf_eskf_predict_covariance_sparse(
    float nextP[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    const float F[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    const float G[SF_ESKF_ERROR_STATES][SF_ESKF_NOISE_STATES],
    const float P[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    const float Q[SF_ESKF_NOISE_STATES]);

static void sf_eskf_predict_noise_default(sf_predict_noise_t *cfg) {
  sf_predict_noise_default(cfg);
  cfg->gyro_var = 2.2873113e-7f;
  cfg->accel_var = 2.4504214e-5f;
}

static void sf_eskf_predict_covariance_sparse(
    float nextP[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    const float F[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    const float G[SF_ESKF_ERROR_STATES][SF_ESKF_NOISE_STATES],
    const float P[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
    const float Q[SF_ESKF_NOISE_STATES]) {
  static const unsigned char f_row_counts[SF_ESKF_ERROR_STATES] = {
      6, 6, 6, 7, 7, 7, 2, 2, 2, 1, 1, 1, 1, 1, 1,
  };
  static const unsigned char f_row_cols[SF_ESKF_ERROR_STATES][7] = {
      {0, 1, 2, 9, 10, 11, 0},
      {0, 1, 2, 9, 10, 11, 0},
      {0, 1, 2, 9, 10, 11, 0},
      {0, 1, 2, 3, 12, 13, 14},
      {0, 1, 2, 4, 12, 13, 14},
      {0, 1, 2, 5, 12, 13, 14},
      {3, 6, 0, 0, 0, 0, 0},
      {4, 7, 0, 0, 0, 0, 0},
      {5, 8, 0, 0, 0, 0, 0},
      {9, 0, 0, 0, 0, 0, 0},
      {10, 0, 0, 0, 0, 0, 0},
      {11, 0, 0, 0, 0, 0, 0},
      {12, 0, 0, 0, 0, 0, 0},
      {13, 0, 0, 0, 0, 0, 0},
      {14, 0, 0, 0, 0, 0, 0},
  };
  static const unsigned char g_row_counts[SF_ESKF_ERROR_STATES] = {
      3, 3, 3, 3, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1,
  };
  static const unsigned char g_row_cols[SF_ESKF_ERROR_STATES][3] = {
      {0, 1, 2},
      {0, 1, 2},
      {0, 1, 2},
      {3, 4, 5},
      {3, 4, 5},
      {3, 4, 5},
      {0, 0, 0},
      {0, 0, 0},
      {0, 0, 0},
      {6, 0, 0},
      {7, 0, 0},
      {8, 0, 0},
      {9, 0, 0},
      {10, 0, 0},
      {11, 0, 0},
  };

  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    for (int j = i; j < SF_ESKF_ERROR_STATES; ++j) {
      float accum = 0.0f;

      for (int ia = 0; ia < f_row_counts[i]; ++ia) {
        const int a = f_row_cols[i][ia];
        const float fi = F[i][a];
        for (int jb = 0; jb < f_row_counts[j]; ++jb) {
          const int b = f_row_cols[j][jb];
          accum += fi * P[a][b] * F[j][b];
        }
      }

      for (int ia = 0; ia < g_row_counts[i]; ++ia) {
        const int a = g_row_cols[i][ia];
        const float gi = G[i][a];
        for (int jb = 0; jb < g_row_counts[j]; ++jb) {
          const int b = g_row_cols[j][jb];
          if (a == b) {
            accum += gi * Q[a] * G[j][b];
          }
        }
      }

      nextP[i][j] = accum;
      nextP[j][i] = accum;
    }
  }
}

void sf_eskf_init(sf_eskf_t *eskf,
                  const float p_diag[SF_ESKF_ERROR_STATES],
                  const sf_predict_noise_t *noise) {
  sf_predict_noise_t default_noise;

  if (eskf == NULL) {
    return;
  }

  memset(eskf, 0, sizeof(*eskf));
  eskf->nominal.q0 = 1.0f;

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
  sf_eskf_normalize_quat(&eskf->nominal.q0);
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
  sf_eskf_predict_covariance_sparse(nextP, F, G, eskf->p, Q);

  memcpy(eskf->p, nextP, sizeof(eskf->p));
  sf_eskf_symmetrize_p(eskf->p);
}

void sf_eskf_fuse_gps(sf_eskf_t *eskf, const sf_gnss_sample_t *gps) {
  if (eskf == NULL || gps == NULL) {
    return;
  }

  sf_eskf_fuse_gps_pos_n(eskf, gps->pos_ned_m[0], gps->pos_std_m[0] * gps->pos_std_m[0]);
  sf_eskf_fuse_gps_pos_e(eskf, gps->pos_ned_m[1], gps->pos_std_m[1] * gps->pos_std_m[1]);
  sf_eskf_fuse_gps_pos_d(eskf, gps->pos_ned_m[2], gps->pos_std_m[2] * gps->pos_std_m[2]);
  sf_eskf_fuse_gps_vel_n(eskf, gps->vel_ned_mps[0], gps->vel_std_mps[0] * gps->vel_std_mps[0]);
  sf_eskf_fuse_gps_vel_e(eskf, gps->vel_ned_mps[1], gps->vel_std_mps[1] * gps->vel_std_mps[1]);
  sf_eskf_fuse_gps_vel_d(eskf, gps->vel_ned_mps[2], gps->vel_std_mps[2] * gps->vel_std_mps[2]);
}

void sf_eskf_fuse_body_vel(sf_eskf_t *eskf, float r_body_vel) {
  if (eskf == NULL) {
    return;
  }
  sf_eskf_fuse_body_vel_y(eskf, r_body_vel);
  sf_eskf_fuse_body_vel_z(eskf, r_body_vel);
}

void sf_eskf_fuse_zero_vel(sf_eskf_t *eskf, float r_zero_vel) {
  if (eskf == NULL) {
    return;
  }
  sf_eskf_fuse_body_vel_x(eskf, r_zero_vel);
  sf_eskf_fuse_body_vel_y(eskf, r_zero_vel);
  sf_eskf_fuse_body_vel_z(eskf, r_zero_vel);
}

void sf_eskf_fuse_stationary_gravity(sf_eskf_t *eskf,
                                     const float accel_body_mps2[3],
                                     float r_stationary_accel) {
  if (eskf == NULL || accel_body_mps2 == NULL) {
    return;
  }
  sf_eskf_fuse_stationary_gravity_x(eskf, accel_body_mps2[0], r_stationary_accel);
  sf_eskf_fuse_stationary_gravity_y(eskf, accel_body_mps2[1], r_stationary_accel);
}

void sf_eskf_compute_error_transition(float f_out[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
                                      float g_out[SF_ESKF_ERROR_STATES][SF_ESKF_NOISE_STATES],
                                      const sf_eskf_t *eskf,
                                      const sf_eskf_imu_delta_t *imu) {
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

static void sf_eskf_symmetrize_p(float p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES]) {
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    for (int j = i + 1; j < SF_ESKF_ERROR_STATES; ++j) {
      const float sym = 0.5f * (p[i][j] + p[j][i]);
      p[i][j] = sym;
      p[j][i] = sym;
    }
  }
}

static void sf_eskf_quat_multiply(const float p[4], const float q[4], float out[4]) {
  out[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3];
  out[1] = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2];
  out[2] = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1];
  out[3] = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0];
}

static void sf_eskf_inject_error_state(sf_eskf_t *eskf, const float dx[SF_ESKF_ERROR_STATES]) {
  float dq[4] = {1.0f, 0.5f * dx[0], 0.5f * dx[1], 0.5f * dx[2]};
  float q_old[4] = {
      eskf->nominal.q0, eskf->nominal.q1, eskf->nominal.q2, eskf->nominal.q3,
  };
  float q_new[4];

  sf_eskf_quat_multiply(q_old, dq, q_new);
  memcpy(&eskf->nominal.q0, q_new, sizeof(q_new));
  sf_eskf_normalize_quat(&eskf->nominal.q0);

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
}

static void sf_eskf_apply_reset(float p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES],
                                const float dtheta[3]) {
  float g_reset[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES] = {{0}};
  float gp[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES] = {{0}};
  float next_p[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES] = {{0}};
  const float dtheta_x = dtheta[0];
  const float dtheta_y = dtheta[1];
  const float dtheta_z = dtheta[2];
  float G_reset_theta[3][3];

  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    g_reset[i][i] = 1.0f;
  }

#include "../generated_eskf/attitude_reset_jacobian_generated.c"

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      g_reset[i][j] = G_reset_theta[i][j];
    }
  }

  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    for (int j = 0; j < SF_ESKF_ERROR_STATES; ++j) {
      for (int k = 0; k < SF_ESKF_ERROR_STATES; ++k) {
        gp[i][j] += g_reset[i][k] * p[k][j];
      }
    }
  }

  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    for (int j = 0; j < SF_ESKF_ERROR_STATES; ++j) {
      for (int k = 0; k < SF_ESKF_ERROR_STATES; ++k) {
        next_p[i][j] += gp[i][k] * g_reset[j][k];
      }
    }
  }

  memcpy(p, next_p, sizeof(next_p));
  sf_eskf_symmetrize_p(p);
}

static void sf_eskf_floor_attitude_covariance(sf_eskf_t *eskf, float sigma_rad) {
  const float var_floor = sigma_rad * sigma_rad;
  if (eskf->p[0][0] < var_floor) {
    eskf->p[0][0] = var_floor;
  }
  if (eskf->p[1][1] < var_floor) {
    eskf->p[1][1] = var_floor;
  }
  sf_eskf_symmetrize_p(eskf->p);
}

static void sf_eskf_fuse_measurement(sf_eskf_t *eskf,
                                     float innovation,
                                     const float h[SF_ESKF_ERROR_STATES],
                                     const float k[SF_ESKF_ERROR_STATES],
                                     const float dx_injected[SF_ESKF_ERROR_STATES]) {
  float p_old[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES];
  float hp[SF_ESKF_ERROR_STATES] = {0};

  memcpy(p_old, eskf->p, sizeof(p_old));
  sf_eskf_inject_error_state(eskf, dx_injected);

  for (int j = 0; j < SF_ESKF_ERROR_STATES; ++j) {
    for (int k_idx = 0; k_idx < SF_ESKF_ERROR_STATES; ++k_idx) {
      hp[j] += h[k_idx] * p_old[k_idx][j];
    }
  }

  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    for (int j = 0; j < SF_ESKF_ERROR_STATES; ++j) {
      eskf->p[i][j] = p_old[i][j] - k[i] * hp[j];
    }
  }

  sf_eskf_symmetrize_p(eskf->p);
  sf_eskf_apply_reset(eskf->p, dx_injected);
  (void)innovation;
}

static void sf_eskf_fuse_gps_pos_n(sf_eskf_t *eskf, float pos_n, float r_pos_n) {
  const float pn = eskf->nominal.pn;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = pos_n - pn;
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float dx[SF_ESKF_ERROR_STATES];
#define R_POS_N r_pos_n
#include "../generated_eskf/gps_pos_n_generated.c"
#undef R_POS_N
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_fuse_measurement(eskf, innovation, H, K, dx);
}

static void sf_eskf_fuse_gps_pos_e(sf_eskf_t *eskf, float pos_e, float r_pos_e) {
  const float pe = eskf->nominal.pe;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = pos_e - pe;
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float dx[SF_ESKF_ERROR_STATES];
#define R_POS_E r_pos_e
#include "../generated_eskf/gps_pos_e_generated.c"
#undef R_POS_E
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_fuse_measurement(eskf, innovation, H, K, dx);
}

static void sf_eskf_fuse_gps_pos_d(sf_eskf_t *eskf, float pos_d, float r_pos_d) {
  const float pd = eskf->nominal.pd;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = pos_d - pd;
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float dx[SF_ESKF_ERROR_STATES];
#define R_POS_D r_pos_d
#include "../generated_eskf/gps_pos_d_generated.c"
#undef R_POS_D
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_fuse_measurement(eskf, innovation, H, K, dx);
}

static void sf_eskf_fuse_gps_vel_n(sf_eskf_t *eskf, float vel_n, float r_vel_n) {
  const float vn = eskf->nominal.vn;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = vel_n - vn;
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float dx[SF_ESKF_ERROR_STATES];
#define R_VEL_N r_vel_n
#include "../generated_eskf/gps_vel_n_generated.c"
#undef R_VEL_N
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_fuse_measurement(eskf, innovation, H, K, dx);
}

static void sf_eskf_fuse_gps_vel_e(sf_eskf_t *eskf, float vel_e, float r_vel_e) {
  const float ve = eskf->nominal.ve;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = vel_e - ve;
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float dx[SF_ESKF_ERROR_STATES];
#define R_VEL_E r_vel_e
#include "../generated_eskf/gps_vel_e_generated.c"
#undef R_VEL_E
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_fuse_measurement(eskf, innovation, H, K, dx);
}

static void sf_eskf_fuse_gps_vel_d(sf_eskf_t *eskf, float vel_d, float r_vel_d) {
  const float vd = eskf->nominal.vd;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = vel_d - vd;
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float dx[SF_ESKF_ERROR_STATES];
#define R_VEL_D r_vel_d
#include "../generated_eskf/gps_vel_d_generated.c"
#undef R_VEL_D
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_fuse_measurement(eskf, innovation, H, K, dx);
}

static void sf_eskf_fuse_stationary_gravity_x(sf_eskf_t *eskf,
                                              float accel_x,
                                              float r_stationary_accel) {
  sf_eskf_floor_attitude_covariance(eskf, 0.10f * (3.14159265358979323846f / 180.0f));
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
  float dx[SF_ESKF_ERROR_STATES];
#define R_STATIONARY_ACCEL r_stationary_accel
#define g g_scalar
#include "../generated_eskf/stationary_accel_x_generated.c"
#undef g
#undef R_STATIONARY_ACCEL
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
  sf_eskf_fuse_measurement(eskf, innovation, H, K, dx);
}

static void sf_eskf_fuse_stationary_gravity_y(sf_eskf_t *eskf,
                                              float accel_y,
                                              float r_stationary_accel) {
  sf_eskf_floor_attitude_covariance(eskf, 0.10f * (3.14159265358979323846f / 180.0f));
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
  float dx[SF_ESKF_ERROR_STATES];
#define R_STATIONARY_ACCEL r_stationary_accel
#define g g_scalar
#include "../generated_eskf/stationary_accel_y_generated.c"
#undef g
#undef R_STATIONARY_ACCEL
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
  sf_eskf_fuse_measurement(eskf, innovation, H, K, dx);
}

static void sf_eskf_fuse_body_vel_y(sf_eskf_t *eskf, float r_body_vel) {
  const float q0 = eskf->nominal.q0;
  const float q1 = eskf->nominal.q1;
  const float q2 = eskf->nominal.q2;
  const float q3 = eskf->nominal.q3;
  const float vn = eskf->nominal.vn;
  const float ve = eskf->nominal.ve;
  const float vd = eskf->nominal.vd;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = -(2.0f * (q1 * q2 - q0 * q3) * vn +
                             (1.0f - 2.0f * q1 * q1 - 2.0f * q3 * q3) * ve +
                             2.0f * (q2 * q3 + q0 * q1) * vd);
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float dx[SF_ESKF_ERROR_STATES];
#define R_BODY_VEL r_body_vel
#include "../generated_eskf/body_vel_y_generated.c"
#undef R_BODY_VEL
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_fuse_measurement(eskf, innovation, H, K, dx);
}

static void sf_eskf_fuse_body_vel_x(sf_eskf_t *eskf, float r_body_vel) {
  const float q0 = eskf->nominal.q0;
  const float q1 = eskf->nominal.q1;
  const float q2 = eskf->nominal.q2;
  const float q3 = eskf->nominal.q3;
  const float vn = eskf->nominal.vn;
  const float ve = eskf->nominal.ve;
  const float vd = eskf->nominal.vd;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = -((1.0f - 2.0f * q2 * q2 - 2.0f * q3 * q3) * vn +
                             2.0f * (q1 * q2 + q0 * q3) * ve +
                             2.0f * (q1 * q3 - q0 * q2) * vd);
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float dx[SF_ESKF_ERROR_STATES];
#define R_BODY_VEL r_body_vel
#include "../generated_eskf/body_vel_x_generated.c"
#undef R_BODY_VEL
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_fuse_measurement(eskf, innovation, H, K, dx);
}

static void sf_eskf_fuse_body_vel_z(sf_eskf_t *eskf, float r_body_vel) {
  const float q0 = eskf->nominal.q0;
  const float q1 = eskf->nominal.q1;
  const float q2 = eskf->nominal.q2;
  const float q3 = eskf->nominal.q3;
  const float vn = eskf->nominal.vn;
  const float ve = eskf->nominal.ve;
  const float vd = eskf->nominal.vd;
  float (*P)[SF_ESKF_ERROR_STATES] = eskf->p;
  const float innovation = -(2.0f * (q1 * q3 + q0 * q2) * vn +
                             2.0f * (q2 * q3 - q0 * q1) * ve +
                             (1.0f - 2.0f * q1 * q1 - 2.0f * q2 * q2) * vd);
  float H[SF_ESKF_ERROR_STATES];
  float K[SF_ESKF_ERROR_STATES];
  float dx[SF_ESKF_ERROR_STATES];
#define R_BODY_VEL r_body_vel
#include "../generated_eskf/body_vel_z_generated.c"
#undef R_BODY_VEL
  for (int i = 0; i < SF_ESKF_ERROR_STATES; ++i) {
    dx[i] = K[i] * innovation;
  }
  sf_eskf_fuse_measurement(eskf, innovation, H, K, dx);
}
