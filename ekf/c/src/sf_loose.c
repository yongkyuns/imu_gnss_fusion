#include "sf_loose.h"

#include <math.h>
#include <string.h>

#define SF_WGS84_A 6378137.0f
#define SF_WGS84_B 6356752.31424518f
#define SF_WGS84_E2 6.69437999014132e-3f
#define SF_WGS84_OMEGA_IE 7.292115e-5f
#define SF_WGS84_GM 3.986004418e14f
#define SF_WGS84_J2 1.08262982136857e-3f
#define SF_LOOSE_REF_GYRO_DT_S 0.02f

static const int SF_LOOSE_GPS_REF_SUPPORT_ROW0[1] = {0};
static const int SF_LOOSE_GPS_REF_SUPPORT_ROW1[2] = {0, 1};
static const int SF_LOOSE_GPS_REF_SUPPORT_ROW2[3] = {0, 1, 2};
static const int SF_LOOSE_NHC_Y_SUPPORT[8] = {3, 4, 5, 6, 7, 8, 21, 23};
static const int SF_LOOSE_NHC_Z_SUPPORT[8] = {3, 4, 5, 6, 7, 8, 21, 22};
static const unsigned char SF_LOOSE_F_ROW_COUNTS[SF_LOOSE_ERROR_STATES] = {
    2, 2, 2, 10, 10, 9, 8, 8, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};
static const unsigned char SF_LOOSE_F_ROW_COLS[SF_LOOSE_ERROR_STATES][10] = {
    {0, 3, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 4, 0, 0, 0, 0, 0, 0, 0, 0},
    {2, 5, 0, 0, 0, 0, 0, 0, 0, 0},
    {3, 4, 7, 8, 9, 10, 11, 15, 16, 17},
    {3, 4, 6, 8, 9, 10, 11, 15, 16, 17},
    {5, 6, 7, 9, 10, 11, 15, 16, 17, 0},
    {6, 7, 12, 13, 14, 18, 19, 20, 0, 0},
    {6, 7, 12, 13, 14, 18, 19, 20, 0, 0},
    {8, 12, 13, 14, 18, 19, 20, 0, 0, 0},
    {9, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {10, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {11, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {12, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {13, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {14, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {15, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {16, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {17, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {18, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {19, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {20, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {21, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {22, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {23, 0, 0, 0, 0, 0, 0, 0, 0, 0},
};
static const unsigned char SF_LOOSE_G_ROW_COUNTS[SF_LOOSE_ERROR_STATES] = {
    0, 0, 0, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};
static const unsigned char SF_LOOSE_G_ROW_COLS[SF_LOOSE_ERROR_STATES][3] = {
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0},
    {0, 1, 2},
    {0, 1, 2},
    {0, 1, 2},
    {3, 4, 5},
    {3, 4, 5},
    {3, 4, 5},
    {6, 0, 0},
    {7, 0, 0},
    {8, 0, 0},
    {9, 0, 0},
    {10, 0, 0},
    {11, 0, 0},
    {12, 0, 0},
    {13, 0, 0},
    {14, 0, 0},
    {15, 0, 0},
    {16, 0, 0},
    {17, 0, 0},
    {18, 0, 0},
    {19, 0, 0},
    {20, 0, 0},
};

#if defined(__GNUC__) || defined(__clang__)
#define SF_MAYBE_UNUSED __attribute__((unused))
#else
#define SF_MAYBE_UNUSED
#endif

static void sf_loose_normalize_quat(float q[4]);
static void sf_loose_normalize_nominal_quat(sf_loose_t *loose);
static void sf_loose_sync_nominal_position_from_shadow(sf_loose_t *loose);
static void sf_loose_sync_nominal_mount_from_shadow(sf_loose_t *loose);
static void sf_loose_sync_covariance_from_shadow(sf_loose_t *loose);
static void sf_loose_symmetrize_p(float p[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES]);
static void sf_loose_quat_multiply(const float p[4], const float q[4], float out[4]);
static void sf_loose_euler_to_quat(float roll, float pitch, float yaw, float out[4]);
static void sf_loose_inject_error_state(sf_loose_t *loose, const float dx[SF_LOOSE_ERROR_STATES]);
static SF_MAYBE_UNUSED void sf_loose_apply_reset(
    float p[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
    const float dtheta[3]);
static void sf_loose_batch_update_joseph(sf_loose_t *loose,
                                         int obs_count,
                                         const float h[][SF_LOOSE_ERROR_STATES],
                                         const int *const h_supports[],
                                         const int h_support_lens[],
                                         const float residuals[],
                                         const float variances[]);
static int sf_loose_append_reference_gps_observations(sf_loose_t *loose,
                                                      const double pos_ecef_m[3],
                                                      const float vel_ecef_mps[3],
                                                      float h_acc_m,
                                                      float speed_acc_mps,
                                                      float dt_since_last_gnss_s,
                                                      float h_rows[][SF_LOOSE_ERROR_STATES],
                                                      const int *h_supports[],
                                                      int h_support_lens[],
                                                      float residuals[],
                                                      float variances[],
                                                      int obs_types[],
                                                      int obs_count);
static void sf_loose_predict_noise_default(sf_loose_predict_noise_t *cfg);
static void sf_loose_predict_covariance_sparse(
    double nextP[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
    const float F[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
    const float G[SF_LOOSE_ERROR_STATES][SF_LOOSE_NOISE_STATES],
    const double P[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
    const float Q[SF_LOOSE_NOISE_STATES]);
static void sf_loose_quat_to_dcm(const float q[4], float c[3][3]);
static void sf_loose_dcm_ecef_to_ned(float lat_rad, float lon_rad, float c[3][3]);
static void sf_loose_ecef_to_llh(const float x_e[3], float *lat_rad, float *lon_rad, float *height_m);
static void sf_loose_gravity_ecef_j2(const float x_e[3], float g_e[3]);
static float sf_loose_vec_norm3(const float v[3]);
static int sf_loose_test_chi2_scalar(float residual,
                                     const float p[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
                                     const float h[SF_LOOSE_ERROR_STATES],
                                     float r);
static int sf_loose_test_chi2_vec3(const float residual[3],
                                   const float p[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
                                   const float h[3][SF_LOOSE_ERROR_STATES],
                                   const float r_diag[3]);
static void sf_loose_extract_support_from_row(const float h[SF_LOOSE_ERROR_STATES],
                                              int support[SF_LOOSE_ERROR_STATES],
                                              int *support_len);

static void sf_loose_predict_noise_default(sf_loose_predict_noise_t *cfg) {
  if (cfg == NULL) {
    return;
  }
  cfg->gyro_var = 2.5e-5f;
  cfg->accel_var = 9.0e-4f;
  cfg->gyro_bias_rw_var = 1.0e-12f;
  cfg->accel_bias_rw_var = 1.0e-10f;
  cfg->gyro_scale_rw_var = 1.0e-10f;
  cfg->accel_scale_rw_var = 1.0e-10f;
  cfg->mount_align_rw_var = 1.0e-8f;
}

static void sf_loose_extract_support_from_row(const float h[SF_LOOSE_ERROR_STATES],
                                              int support[SF_LOOSE_ERROR_STATES],
                                              int *support_len) {
  if (h == NULL || support == NULL || support_len == NULL) {
    return;
  }
  *support_len = 0;
  for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
    if (h[i] != 0.0f) {
      support[*support_len] = i;
      ++(*support_len);
    }
  }
}

static void sf_loose_normalize_nominal_quat(sf_loose_t *loose) {
  float q[4] = {
      loose->nominal.q0,
      loose->nominal.q1,
      loose->nominal.q2,
      loose->nominal.q3,
  };
  sf_loose_normalize_quat(q);
  loose->nominal.q0 = q[0];
  loose->nominal.q1 = q[1];
  loose->nominal.q2 = q[2];
  loose->nominal.q3 = q[3];
}

static void sf_loose_sync_nominal_position_from_shadow(sf_loose_t *loose) {
  if (loose == NULL) {
    return;
  }
  loose->nominal.pn = (float)loose->pos_e64[0];
  loose->nominal.pe = (float)loose->pos_e64[1];
  loose->nominal.pd = (float)loose->pos_e64[2];
}

static void sf_loose_sync_nominal_mount_from_shadow(sf_loose_t *loose) {
  if (loose == NULL) {
    return;
  }
  loose->nominal.qcs0 = (float)loose->qcs64[0];
  loose->nominal.qcs1 = (float)loose->qcs64[1];
  loose->nominal.qcs2 = (float)loose->qcs64[2];
  loose->nominal.qcs3 = (float)loose->qcs64[3];
}

static void sf_loose_sync_covariance_from_shadow(sf_loose_t *loose) {
  if (loose == NULL) {
    return;
  }
  for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
    for (int j = 0; j < SF_LOOSE_ERROR_STATES; ++j) {
      loose->p[i][j] = (float)loose->p64[i][j];
    }
  }
  sf_loose_symmetrize_p(loose->p);
}

static void sf_loose_predict_covariance_sparse(
    double nextP[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
    const float F[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
    const float G[SF_LOOSE_ERROR_STATES][SF_LOOSE_NOISE_STATES],
    const double P[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
    const float Q[SF_LOOSE_NOISE_STATES]) {
  for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
    for (int j = i; j < SF_LOOSE_ERROR_STATES; ++j) {
      double accum = 0.0;
      for (int ia = 0; ia < SF_LOOSE_F_ROW_COUNTS[i]; ++ia) {
        const int a = SF_LOOSE_F_ROW_COLS[i][ia];
        const double fia = (double)F[i][a];
        for (int jb = 0; jb < SF_LOOSE_F_ROW_COUNTS[j]; ++jb) {
          const int b = SF_LOOSE_F_ROW_COLS[j][jb];
          accum += fia * P[a][b] * (double)F[j][b];
        }
      }

      for (int ia = 0; ia < SF_LOOSE_G_ROW_COUNTS[i]; ++ia) {
        const int a = SF_LOOSE_G_ROW_COLS[i][ia];
        const double gia = (double)G[i][a];
        if (Q[a] == 0.0f) {
          continue;
        }
        for (int jb = 0; jb < SF_LOOSE_G_ROW_COUNTS[j]; ++jb) {
          const int b = SF_LOOSE_G_ROW_COLS[j][jb];
          if (a == b) {
            accum += gia * (double)Q[a] * (double)G[j][b];
          }
        }
      }

      nextP[i][j] = accum;
      nextP[j][i] = accum;
    }
  }
}

void sf_loose_init(sf_loose_t *loose,
                   const float p_diag[SF_LOOSE_ERROR_STATES],
                   const sf_loose_predict_noise_t *noise) {
  sf_loose_predict_noise_t default_noise;

  if (loose == NULL) {
    return;
  }

  memset(loose, 0, sizeof(*loose));
  loose->nominal.q0 = 1.0f;
  loose->nominal.qcs0 = 1.0f;
  loose->pos_e64[0] = 0.0;
  loose->pos_e64[1] = 0.0;
  loose->pos_e64[2] = 0.0;
  loose->qcs64[0] = 1.0;
  loose->qcs64[1] = 0.0;
  loose->qcs64[2] = 0.0;
  loose->qcs64[3] = 0.0;

  sf_loose_predict_noise_default(&default_noise);
  loose->noise = noise ? *noise : default_noise;

  if (p_diag != NULL) {
    for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
      loose->p[i][i] = p_diag[i];
      loose->p64[i][i] = (double)p_diag[i];
    }
  } else {
    for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
      loose->p[i][i] = 1.0f;
      loose->p64[i][i] = 1.0;
    }
  }
}

void sf_loose_predict_nominal(sf_loose_t *loose, const sf_loose_imu_delta_t *imu) {
  if (loose == NULL || imu == NULL) {
    return;
  }
  const float dt = imu->dt;
  if (dt <= 0.0f) {
    return;
  }

  const float q_es[4] = {
      loose->nominal.q0, loose->nominal.q1, loose->nominal.q2, loose->nominal.q3};
  const float v_e[3] = {loose->nominal.vn, loose->nominal.ve, loose->nominal.vd};
  const double x_e[3] = {loose->pos_e64[0], loose->pos_e64[1], loose->pos_e64[2]};
  const float b_w[3] = {loose->nominal.bgx, loose->nominal.bgy, loose->nominal.bgz};
  const float b_f[3] = {loose->nominal.bax, loose->nominal.bay, loose->nominal.baz};
  const float s_w[3] = {loose->nominal.sgx, loose->nominal.sgy, loose->nominal.sgz};
  const float s_f[3] = {loose->nominal.sax, loose->nominal.say, loose->nominal.saz};
  float omega1[3] = {
      s_w[0] * (imu->dax_1 / dt) + b_w[0],
      s_w[1] * (imu->day_1 / dt) + b_w[1],
      s_w[2] * (imu->daz_1 / dt) + b_w[2],
  };
  float f1[3] = {
      s_f[0] * (imu->dvx_1 / dt) + b_f[0],
      s_f[1] * (imu->dvy_1 / dt) + b_f[1],
      s_f[2] * (imu->dvz_1 / dt) + b_f[2],
  };
  float omega2[3] = {
      s_w[0] * (imu->dax_2 / dt) + b_w[0],
      s_w[1] * (imu->day_2 / dt) + b_w[1],
      s_w[2] * (imu->daz_2 / dt) + b_w[2],
  };
  float f2[3] = {
      s_f[0] * (imu->dvx_2 / dt) + b_f[0],
      s_f[1] * (imu->dvy_2 / dt) + b_f[1],
      s_f[2] * (imu->dvz_2 / dt) + b_f[2],
  };

  float x_e_f[3] = {(float)x_e[0], (float)x_e[1], (float)x_e[2]};
  float c_es[3][3];
  sf_loose_quat_to_dcm(q_es, c_es);
  float g_e1[3];
  sf_loose_gravity_ecef_j2(x_e_f, g_e1);
  float f1_e[3] = {
      c_es[0][0] * f1[0] + c_es[0][1] * f1[1] + c_es[0][2] * f1[2],
      c_es[1][0] * f1[0] + c_es[1][1] * f1[1] + c_es[1][2] * f1[2],
      c_es[2][0] * f1[0] + c_es[2][1] * f1[1] + c_es[2][2] * f1[2],
  };
  float vdot1[3] = {
      g_e1[0] + f1_e[0] + 2.0f * SF_WGS84_OMEGA_IE * v_e[1],
      g_e1[1] + f1_e[1] - 2.0f * SF_WGS84_OMEGA_IE * v_e[0],
      g_e1[2] + f1_e[2],
  };
  float q_omega1[4] = {0.0f, omega1[0], omega1[1], omega1[2]};
  float qdot1[4];
  sf_loose_quat_multiply(q_es, q_omega1, qdot1);
  qdot1[0] = 0.5f * (qdot1[0] + SF_WGS84_OMEGA_IE * q_es[3]);
  qdot1[1] = 0.5f * (qdot1[1] + SF_WGS84_OMEGA_IE * q_es[2]);
  qdot1[2] = 0.5f * (qdot1[2] - SF_WGS84_OMEGA_IE * q_es[1]);
  qdot1[3] = 0.5f * (qdot1[3] - SF_WGS84_OMEGA_IE * q_es[0]);

  double x_tmp[3] = {
      x_e[0] + (double)dt * (double)v_e[0],
      x_e[1] + (double)dt * (double)v_e[1],
      x_e[2] + (double)dt * (double)v_e[2],
  };
  float v_tmp[3] = {
      v_e[0] + dt * vdot1[0],
      v_e[1] + dt * vdot1[1],
      v_e[2] + dt * vdot1[2],
  };
  float q_tmp[4] = {
      q_es[0] + dt * qdot1[0],
      q_es[1] + dt * qdot1[1],
      q_es[2] + dt * qdot1[2],
      q_es[3] + dt * qdot1[3],
  };
  sf_loose_normalize_quat(q_tmp);

  float c_es_tmp[3][3];
  sf_loose_quat_to_dcm(q_tmp, c_es_tmp);
  float g_e2[3];
  float x_tmp_f[3] = {(float)x_tmp[0], (float)x_tmp[1], (float)x_tmp[2]};
  sf_loose_gravity_ecef_j2(x_tmp_f, g_e2);
  float f2_e[3] = {
      c_es_tmp[0][0] * f2[0] + c_es_tmp[0][1] * f2[1] + c_es_tmp[0][2] * f2[2],
      c_es_tmp[1][0] * f2[0] + c_es_tmp[1][1] * f2[1] + c_es_tmp[1][2] * f2[2],
      c_es_tmp[2][0] * f2[0] + c_es_tmp[2][1] * f2[1] + c_es_tmp[2][2] * f2[2],
  };
  float vdot2[3] = {
      g_e2[0] + f2_e[0] + 2.0f * SF_WGS84_OMEGA_IE * v_tmp[1],
      g_e2[1] + f2_e[1] - 2.0f * SF_WGS84_OMEGA_IE * v_tmp[0],
      g_e2[2] + f2_e[2],
  };
  float q_omega2[4] = {0.0f, omega2[0], omega2[1], omega2[2]};
  float qdot2[4];
  sf_loose_quat_multiply(q_tmp, q_omega2, qdot2);
  qdot2[0] = 0.5f * (qdot2[0] + SF_WGS84_OMEGA_IE * q_tmp[3]);
  qdot2[1] = 0.5f * (qdot2[1] + SF_WGS84_OMEGA_IE * q_tmp[2]);
  qdot2[2] = 0.5f * (qdot2[2] - SF_WGS84_OMEGA_IE * q_tmp[1]);
  qdot2[3] = 0.5f * (qdot2[3] - SF_WGS84_OMEGA_IE * q_tmp[0]);

  loose->pos_e64[0] = x_e[0] + 0.5 * (double)dt * (double)(v_e[0] + v_tmp[0]);
  loose->pos_e64[1] = x_e[1] + 0.5 * (double)dt * (double)(v_e[1] + v_tmp[1]);
  loose->pos_e64[2] = x_e[2] + 0.5 * (double)dt * (double)(v_e[2] + v_tmp[2]);
  sf_loose_sync_nominal_position_from_shadow(loose);
  loose->nominal.vn = v_e[0] + 0.5f * dt * (vdot1[0] + vdot2[0]);
  loose->nominal.ve = v_e[1] + 0.5f * dt * (vdot1[1] + vdot2[1]);
  loose->nominal.vd = v_e[2] + 0.5f * dt * (vdot1[2] + vdot2[2]);
  loose->nominal.q0 = q_es[0] + 0.5f * dt * (qdot1[0] + qdot2[0]);
  loose->nominal.q1 = q_es[1] + 0.5f * dt * (qdot1[1] + qdot2[1]);
  loose->nominal.q2 = q_es[2] + 0.5f * dt * (qdot1[2] + qdot2[2]);
  loose->nominal.q3 = q_es[3] + 0.5f * dt * (qdot1[3] + qdot2[3]);
  sf_loose_normalize_nominal_quat(loose);
}

void sf_loose_predict(sf_loose_t *loose, const sf_loose_imu_delta_t *imu) {
  float F[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES];
  float G[SF_LOOSE_ERROR_STATES][SF_LOOSE_NOISE_STATES];
  float Q[SF_LOOSE_NOISE_STATES];
  double nextP[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES];

  if (loose == NULL || imu == NULL) {
    return;
  }
  const float dt = imu->dt;
  if (dt <= 0.0f) {
    return;
  }

  sf_loose_predict_nominal(loose, imu);
  sf_loose_compute_error_transition(F, G, loose, imu);

  Q[0] = loose->noise.accel_var * dt;
  Q[1] = Q[0];
  Q[2] = Q[0];
  Q[3] = loose->noise.gyro_var * dt;
  Q[4] = Q[3];
  Q[5] = Q[3];
  Q[6] = loose->noise.accel_bias_rw_var * dt;
  Q[7] = Q[6];
  Q[8] = Q[6];
  Q[9] = loose->noise.gyro_bias_rw_var * dt;
  Q[10] = Q[9];
  Q[11] = Q[9];
  Q[12] = loose->noise.accel_scale_rw_var * dt;
  Q[13] = Q[12];
  Q[14] = Q[12];
  Q[15] = loose->noise.gyro_scale_rw_var * dt;
  Q[16] = Q[15];
  Q[17] = Q[15];
  Q[18] = loose->noise.mount_align_rw_var * dt;
  Q[19] = Q[18];
  Q[20] = Q[18];

  memset(nextP, 0, sizeof(nextP));
  sf_loose_predict_covariance_sparse(nextP, F, G, loose->p64, Q);
  memcpy(loose->p64, nextP, sizeof(loose->p64));
  sf_loose_sync_covariance_from_shadow(loose);
}

void sf_loose_fuse_gps_reference(sf_loose_t *loose,
                                 const double pos_ecef_m[3],
                                 const float vel_ecef_mps[3],
                                 float h_acc_m,
                                 float speed_acc_mps,
                                 float dt_since_last_gnss_s) {
  if (loose == NULL) {
    return;
  }
  float h_rows[3][SF_LOOSE_ERROR_STATES] = {{0}};
  const int *h_supports[3] = {0};
  int h_support_lens[3] = {0};
  float residuals[3] = {0};
  float variances[3] = {0};
  int obs_types[3] = {0};
  int obs_count = sf_loose_append_reference_gps_observations(
      loose,
      pos_ecef_m,
      vel_ecef_mps,
      h_acc_m,
      speed_acc_mps,
      dt_since_last_gnss_s,
      h_rows,
      h_supports,
      h_support_lens,
      residuals,
      variances,
      obs_types,
      0);
  if (obs_count > 0) {
    sf_loose_batch_update_joseph(
        loose, obs_count, h_rows, h_supports, h_support_lens, residuals, variances);
  }
}

void sf_loose_fuse_reference_batch(sf_loose_t *loose,
                                   const double pos_ecef_m[3],
                                   const float vel_ecef_mps[3],
                                   float h_acc_m,
                                   float speed_acc_mps,
                                   float dt_since_last_gnss_s,
                                   const float gyro_radps[3],
                                   const float accel_mps2[3],
                                   float dt_s) {
  if (loose == NULL || gyro_radps == NULL || accel_mps2 == NULL || dt_s <= 0.0f) {
    return;
  }

  loose->last_obs_count = 0;
  memset(loose->last_obs_types, 0, sizeof(loose->last_obs_types));

  float h_rows[8][SF_LOOSE_ERROR_STATES] = {{0}};
  const int *h_supports[8] = {0};
  int h_support_lens[8] = {0};
  float residuals[8] = {0};
  float variances[8] = {0};
  int obs_count = 0;
  obs_count = sf_loose_append_reference_gps_observations(
      loose,
      pos_ecef_m,
      vel_ecef_mps,
      h_acc_m,
      speed_acc_mps,
      dt_since_last_gnss_s,
      h_rows,
      h_supports,
      h_support_lens,
      residuals,
      variances,
      loose->last_obs_types,
      obs_count);

  float omega_is[3] = {
      loose->nominal.sgx * gyro_radps[0] + loose->nominal.bgx,
      loose->nominal.sgy * gyro_radps[1] + loose->nominal.bgy,
      loose->nominal.sgz * gyro_radps[2] + loose->nominal.bgz,
  };
  float f_s[3] = {
      loose->nominal.sax * accel_mps2[0] + loose->nominal.bax,
      loose->nominal.say * accel_mps2[1] + loose->nominal.bay,
      loose->nominal.saz * accel_mps2[2] + loose->nominal.baz,
  };
  if (sf_loose_vec_norm3(omega_is) < 0.03f && fabsf(sf_loose_vec_norm3(f_s) - 9.81f) < 0.2f) {
    float h_y[SF_LOOSE_ERROR_STATES] = {0};
    float h_z[SF_LOOSE_ERROR_STATES] = {0};
    const float q0 = loose->nominal.q0;
    const float q1 = loose->nominal.q1;
    const float q2 = loose->nominal.q2;
    const float q3 = loose->nominal.q3;
    const float qcs0 = loose->nominal.qcs0;
    const float qcs1 = loose->nominal.qcs1;
    const float qcs2 = loose->nominal.qcs2;
    const float qcs3 = loose->nominal.qcs3;
    const float vn = loose->nominal.vn;
    const float ve = loose->nominal.ve;
    const float vd = loose->nominal.vd;
    float vc_est = 0.0f;
    float vc_y_est = 0.0f;
    float vc_z_est = 0.0f;
    float H[SF_LOOSE_ERROR_STATES];
#include "../generated_loose/reference_nhc_y_generated.c"
    memcpy(h_y, H, sizeof(h_y));
    vc_y_est = vc_est;
#include "../generated_loose/reference_nhc_z_generated.c"
    memcpy(h_z, H, sizeof(h_z));
    vc_z_est = vc_est;
    float gate_var_y = 0.1f * 0.1f;
    float gate_var_z = 0.05f * 0.05f;
    float var_y = gate_var_y / SF_LOOSE_REF_GYRO_DT_S;
    float var_z = gate_var_z / SF_LOOSE_REF_GYRO_DT_S;
    if (!sf_loose_test_chi2_scalar(-vc_y_est, loose->p, h_y, gate_var_y)) {
      memcpy(h_rows[obs_count], h_y, sizeof(h_y));
      h_supports[obs_count] = SF_LOOSE_NHC_Y_SUPPORT;
      h_support_lens[obs_count] = 8;
      residuals[obs_count] = -vc_y_est;
      variances[obs_count] = var_y;
      loose->last_obs_types[obs_count] = 7;
      ++obs_count;
    }
    if (!sf_loose_test_chi2_scalar(-vc_z_est, loose->p, h_z, gate_var_z)) {
      memcpy(h_rows[obs_count], h_z, sizeof(h_z));
      h_supports[obs_count] = SF_LOOSE_NHC_Z_SUPPORT;
      h_support_lens[obs_count] = 8;
      residuals[obs_count] = -vc_z_est;
      variances[obs_count] = var_z;
      loose->last_obs_types[obs_count] = 8;
      ++obs_count;
    }
  }

  loose->last_obs_count = obs_count;
  if (obs_count > 0) {
    sf_loose_batch_update_joseph(
        loose, obs_count, h_rows, h_supports, h_support_lens, residuals, variances);
  }
}

static int sf_loose_append_reference_gps_observations(sf_loose_t *loose,
                                                      const double pos_ecef_m[3],
                                                      const float vel_ecef_mps[3],
                                                      float h_acc_m,
                                                      float speed_acc_mps,
                                                      float dt_since_last_gnss_s,
                                                      float h_rows[][SF_LOOSE_ERROR_STATES],
                                                      const int *h_supports[],
                                                      int h_support_lens[],
                                                      float residuals[],
                                                      float variances[],
                                                      int obs_types[],
                                                      int obs_count) {
  static const int gps_vel_supports[3][1] = {{3}, {4}, {5}};

  if (loose == NULL || h_rows == NULL ||
      h_supports == NULL || h_support_lens == NULL || residuals == NULL || variances == NULL) {
    return obs_count;
  }

  if (pos_ecef_m != NULL && h_acc_m > 0.0f) {
    float lat_rad, lon_rad, height_m;
    sf_loose_ecef_to_llh(
        (const float[3]){(float)loose->pos_e64[0], (float)loose->pos_e64[1], (float)loose->pos_e64[2]},
        &lat_rad,
        &lon_rad,
        &height_m);
    (void)height_m;

    float c_en[3][3];
    sf_loose_dcm_ecef_to_ned(lat_rad, lon_rad, c_en);
    float r_n_diag[3] = {
        h_acc_m * h_acc_m,
        h_acc_m * h_acc_m,
        (2.5f * h_acc_m) * (2.5f * h_acc_m),
    };
    float r_e[3][3] = {{0}};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          r_e[i][j] += c_en[i][k] * r_n_diag[k] * c_en[j][k];
        }
      }
    }

    float u11 = sqrtf(fmaxf(r_e[0][0], 1.0e-9f));
    float u12 = r_e[0][1] / u11;
    float u13 = r_e[0][2] / u11;
    float u22 = sqrtf(fmaxf(r_e[1][1] - u12 * u12, 1.0e-9f));
    float u23 = (r_e[1][2] - u12 * u13) / u22;
    float u33 = sqrtf(fmaxf(r_e[2][2] - u13 * u13 - u23 * u23, 1.0e-9f));

    float t[3][3] = {
        {1.0f / u11, 0.0f, 0.0f},
        {-u12 / (u11 * u22), 1.0f / u22, 0.0f},
        {(u12 * u23 - u13 * u22) / (u11 * u22 * u33), -u23 / (u22 * u33), 1.0f / u33},
    };
    double x_meas[3] = {0};
    double x_est[3] = {loose->pos_e64[0], loose->pos_e64[1], loose->pos_e64[2]};
    float h_tmp[3][SF_LOOSE_ERROR_STATES] = {{0}};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        x_meas[i] += (double)t[i][j] * pos_ecef_m[j];
        h_tmp[i][j] = t[i][j];
      }
    }

    float residual[3] = {
        (float)(x_meas[0] -
                ((double)t[0][0] * x_est[0] + (double)t[0][1] * x_est[1] + (double)t[0][2] * x_est[2])),
        (float)(x_meas[1] -
                ((double)t[1][0] * x_est[0] + (double)t[1][1] * x_est[1] + (double)t[1][2] * x_est[2])),
        (float)(x_meas[2] -
                ((double)t[2][0] * x_est[0] + (double)t[2][1] * x_est[1] + (double)t[2][2] * x_est[2])),
    };
    float r_diag[3] = {1.0f, 1.0f, 1.0f};
    if (!sf_loose_test_chi2_vec3(residual, loose->p, h_tmp, r_diag)) {
      const float meas_var = 1.0f / fminf(fmaxf(dt_since_last_gnss_s, 1.0e-3f), 1.0f);
      const int *gps_supports[3] = {
          SF_LOOSE_GPS_REF_SUPPORT_ROW0,
          SF_LOOSE_GPS_REF_SUPPORT_ROW1,
          SF_LOOSE_GPS_REF_SUPPORT_ROW2,
      };
      const int gps_support_lens[3] = {1, 2, 3};
      for (int row = 0; row < 3; ++row) {
        memcpy(h_rows[obs_count], h_tmp[row], sizeof(h_tmp[row]));
        h_supports[obs_count] = gps_supports[row];
        h_support_lens[obs_count] = gps_support_lens[row];
        residuals[obs_count] = residual[row];
        variances[obs_count] = meas_var;
        if (obs_types != NULL) {
          obs_types[obs_count] = row + 1;
        }
        ++obs_count;
      }
    }
  }

  if (vel_ecef_mps != NULL && speed_acc_mps > 0.0f) {
    float vel_residual[3] = {
        vel_ecef_mps[0] - loose->nominal.vn,
        vel_ecef_mps[1] - loose->nominal.ve,
        vel_ecef_mps[2] - loose->nominal.vd,
    };
    float vel_rows[3][SF_LOOSE_ERROR_STATES] = {{0}};
    const float vel_var = fmaxf(speed_acc_mps * speed_acc_mps, 1.0e-4f);
    float vel_r_diag[3] = {vel_var, vel_var, vel_var};
    vel_rows[0][3] = 1.0f;
    vel_rows[1][4] = 1.0f;
    vel_rows[2][5] = 1.0f;
    if (!sf_loose_test_chi2_vec3(vel_residual, loose->p, vel_rows, vel_r_diag)) {
      for (int row = 0; row < 3; ++row) {
        memcpy(h_rows[obs_count], vel_rows[row], sizeof(vel_rows[row]));
        h_supports[obs_count] = gps_vel_supports[row];
        h_support_lens[obs_count] = 1;
        residuals[obs_count] = vel_residual[row];
        variances[obs_count] = vel_var;
        if (obs_types != NULL) {
          obs_types[obs_count] = row + 4;
        }
        ++obs_count;
      }
    }
  }
  return obs_count;
}

void sf_loose_fuse_nhc_reference(sf_loose_t *loose,
                                 const float gyro_radps[3],
                                 const float accel_mps2[3],
                                 float dt_s) {
  if (loose == NULL || gyro_radps == NULL || accel_mps2 == NULL || dt_s <= 0.0f) {
    return;
  }
  float omega_is[3] = {
      loose->nominal.sgx * gyro_radps[0] + loose->nominal.bgx,
      loose->nominal.sgy * gyro_radps[1] + loose->nominal.bgy,
      loose->nominal.sgz * gyro_radps[2] + loose->nominal.bgz,
  };
  float f_s[3] = {
      loose->nominal.sax * accel_mps2[0] + loose->nominal.bax,
      loose->nominal.say * accel_mps2[1] + loose->nominal.bay,
      loose->nominal.saz * accel_mps2[2] + loose->nominal.baz,
  };
  if (sf_loose_vec_norm3(omega_is) >= 0.03f || fabsf(sf_loose_vec_norm3(f_s) - 9.81f) >= 0.2f) {
    return;
  }
  float h_y[SF_LOOSE_ERROR_STATES] = {0};
  float h_z[SF_LOOSE_ERROR_STATES] = {0};
  const float q0 = loose->nominal.q0;
  const float q1 = loose->nominal.q1;
  const float q2 = loose->nominal.q2;
  const float q3 = loose->nominal.q3;
  const float qcs0 = loose->nominal.qcs0;
  const float qcs1 = loose->nominal.qcs1;
  const float qcs2 = loose->nominal.qcs2;
  const float qcs3 = loose->nominal.qcs3;
  const float vn = loose->nominal.vn;
  const float ve = loose->nominal.ve;
  const float vd = loose->nominal.vd;
  float vc_est = 0.0f;
  float vc_y_est = 0.0f;
  float vc_z_est = 0.0f;
  float H[SF_LOOSE_ERROR_STATES];
#include "../generated_loose/reference_nhc_y_generated.c"
  memcpy(h_y, H, sizeof(h_y));
  vc_y_est = vc_est;
#include "../generated_loose/reference_nhc_z_generated.c"
  memcpy(h_z, H, sizeof(h_z));
  vc_z_est = vc_est;
  float gate_var_y = 0.1f * 0.1f;
  float gate_var_z = 0.05f * 0.05f;
  float var_y = gate_var_y / SF_LOOSE_REF_GYRO_DT_S;
  float var_z = gate_var_z / SF_LOOSE_REF_GYRO_DT_S;
  float h_rows[2][SF_LOOSE_ERROR_STATES];
  const int *h_supports[2] = {0};
  int h_support_lens[2] = {0};
  float residuals[2];
  float variances[2];
  int obs_count = 0;
  if (!sf_loose_test_chi2_scalar(-vc_y_est, loose->p, h_y, gate_var_y)) {
    memcpy(h_rows[obs_count], h_y, sizeof(h_y));
    h_supports[obs_count] = SF_LOOSE_NHC_Y_SUPPORT;
    h_support_lens[obs_count] = 8;
    residuals[obs_count] = -vc_y_est;
    variances[obs_count] = var_y;
    ++obs_count;
  }
  if (!sf_loose_test_chi2_scalar(-vc_z_est, loose->p, h_z, gate_var_z)) {
    memcpy(h_rows[obs_count], h_z, sizeof(h_z));
    h_supports[obs_count] = SF_LOOSE_NHC_Z_SUPPORT;
    h_support_lens[obs_count] = 8;
    residuals[obs_count] = -vc_z_est;
    variances[obs_count] = var_z;
    ++obs_count;
  }
  if (obs_count > 0) {
    sf_loose_batch_update_joseph(
        loose, obs_count, h_rows, h_supports, h_support_lens, residuals, variances);
  }
}

void sf_loose_compute_error_transition(
    float f_out[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
    float g_out[SF_LOOSE_ERROR_STATES][SF_LOOSE_NOISE_STATES],
    const sf_loose_t *loose,
    const sf_loose_imu_delta_t *imu) {
  if (f_out == NULL || g_out == NULL || loose == NULL || imu == NULL) {
    return;
  }

  const float dt = imu->dt;
  memset(f_out, 0, sizeof(float) * SF_LOOSE_ERROR_STATES * SF_LOOSE_ERROR_STATES);
  memset(g_out, 0, sizeof(float) * SF_LOOSE_ERROR_STATES * SF_LOOSE_NOISE_STATES);
  if (dt <= 0.0f) {
    return;
  }
  const float q0 = loose->nominal.q0;
  const float q1 = loose->nominal.q1;
  const float q2 = loose->nominal.q2;
  const float q3 = loose->nominal.q3;
  const float bax = loose->nominal.bax;
  const float bay = loose->nominal.bay;
  const float baz = loose->nominal.baz;
  const float sax = loose->nominal.sax;
  const float say = loose->nominal.say;
  const float saz = loose->nominal.saz;
  const float dax = imu->dax_2;
  const float day = imu->day_2;
  const float daz = imu->daz_2;
  const float dvx = imu->dvx_2;
  const float dvy = imu->dvy_2;
  const float dvz = imu->dvz_2;
  float (*F)[SF_LOOSE_ERROR_STATES] = f_out;
  float (*G)[SF_LOOSE_NOISE_STATES] = g_out;
#include "../generated_loose/reference_error_transition_generated.c"
#include "../generated_loose/reference_error_noise_input_generated.c"
}

static void sf_loose_normalize_quat(float q[4]) {
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

static void sf_loose_symmetrize_p(float p[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES]) {
  for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
    for (int j = i + 1; j < SF_LOOSE_ERROR_STATES; ++j) {
      const float sym = 0.5f * (p[i][j] + p[j][i]);
      p[i][j] = sym;
      p[j][i] = sym;
    }
  }
}

static void sf_loose_quat_multiply(const float p[4], const float q[4], float out[4]) {
  out[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3];
  out[1] = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2];
  out[2] = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1];
  out[3] = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0];
}

static void sf_loose_euler_to_quat(float roll, float pitch, float yaw, float out[4]) {
  const float cr = cosf(0.5f * roll);
  const float sr = sinf(0.5f * roll);
  const float cp = cosf(0.5f * pitch);
  const float sp = sinf(0.5f * pitch);
  const float cy = cosf(0.5f * yaw);
  const float sy = sinf(0.5f * yaw);
  out[0] = cr * cp * cy + sr * sp * sy;
  out[1] = sr * cp * cy - cr * sp * sy;
  out[2] = cr * sp * cy + sr * cp * sy;
  out[3] = cr * cp * sy - sr * sp * cy;
}

static void sf_loose_inject_error_state(sf_loose_t *loose, const float dx[SF_LOOSE_ERROR_STATES]) {
  float dq[4];
  sf_loose_euler_to_quat(dx[6], dx[7], dx[8], dq);
  float q_old[4] = {
      loose->nominal.q0, loose->nominal.q1, loose->nominal.q2, loose->nominal.q3,
  };
  float q_new[4];

  sf_loose_quat_multiply(dq, q_old, q_new);
  memcpy(&loose->nominal.q0, q_new, sizeof(q_new));
  sf_loose_normalize_nominal_quat(loose);

  loose->pos_e64[0] += (double)dx[0];
  loose->pos_e64[1] += (double)dx[1];
  loose->pos_e64[2] += (double)dx[2];
  sf_loose_sync_nominal_position_from_shadow(loose);
  loose->nominal.vn += dx[3];
  loose->nominal.ve += dx[4];
  loose->nominal.vd += dx[5];
  loose->nominal.bax += dx[9];
  loose->nominal.bay += dx[10];
  loose->nominal.baz += dx[11];
  loose->nominal.bgx += dx[12];
  loose->nominal.bgy += dx[13];
  loose->nominal.bgz += dx[14];
  loose->nominal.sax += dx[15];
  loose->nominal.say += dx[16];
  loose->nominal.saz += dx[17];
  loose->nominal.sgx += dx[18];
  loose->nominal.sgy += dx[19];
  loose->nominal.sgz += dx[20];
  float dqcs[4];
  sf_loose_euler_to_quat(dx[21], dx[22], dx[23], dqcs);
  const double dqcs0 = (double)dqcs[0];
  const double dqcs1 = (double)dqcs[1];
  const double dqcs2 = (double)dqcs[2];
  const double dqcs3 = (double)dqcs[3];
  const double qcs_old0 = loose->qcs64[0];
  const double qcs_old1 = loose->qcs64[1];
  const double qcs_old2 = loose->qcs64[2];
  const double qcs_old3 = loose->qcs64[3];
  double qcs_new0 = dqcs0 * qcs_old0 - dqcs1 * qcs_old1 - dqcs2 * qcs_old2 - dqcs3 * qcs_old3;
  double qcs_new1 = dqcs0 * qcs_old1 + dqcs1 * qcs_old0 + dqcs2 * qcs_old3 - dqcs3 * qcs_old2;
  double qcs_new2 = dqcs0 * qcs_old2 - dqcs1 * qcs_old3 + dqcs2 * qcs_old0 + dqcs3 * qcs_old1;
  double qcs_new3 = dqcs0 * qcs_old3 + dqcs1 * qcs_old2 - dqcs2 * qcs_old1 + dqcs3 * qcs_old0;
  double qcs_norm = sqrt(qcs_new0 * qcs_new0 + qcs_new1 * qcs_new1 + qcs_new2 * qcs_new2 +
                         qcs_new3 * qcs_new3);
  if (qcs_norm > 0.0) {
    qcs_new0 /= qcs_norm;
    qcs_new1 /= qcs_norm;
    qcs_new2 /= qcs_norm;
    qcs_new3 /= qcs_norm;
  } else {
    qcs_new0 = 1.0;
    qcs_new1 = 0.0;
    qcs_new2 = 0.0;
    qcs_new3 = 0.0;
  }
  loose->qcs64[0] = qcs_new0;
  loose->qcs64[1] = qcs_new1;
  loose->qcs64[2] = qcs_new2;
  loose->qcs64[3] = qcs_new3;
  sf_loose_sync_nominal_mount_from_shadow(loose);
}

static SF_MAYBE_UNUSED void sf_loose_apply_reset(
    float p[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
    const float dtheta[3]) {
  const float dtheta_x = dtheta[0];
  const float dtheta_y = dtheta[1];
  const float dtheta_z = dtheta[2];
  float G_reset_theta[3][3];
  float p_aa[3][3];
  float p_ab[3][SF_LOOSE_ERROR_STATES - 3];
  float next_aa[3][3] = {{0}};

#include "../generated_loose/attitude_reset_jacobian_generated.c"

  const int att0 = 6;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      p_aa[i][j] = p[att0 + i][att0 + j];
    }
    for (int j = 0; j < SF_LOOSE_ERROR_STATES; ++j) {
      if (j >= att0 && j < att0 + 3) {
        continue;
      }
      p_ab[i][j < att0 ? j : j - 3] = p[att0 + i][j];
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
      p[att0 + i][att0 + j] = accum;
    }
    for (int j = 0; j < SF_LOOSE_ERROR_STATES; ++j) {
      if (j >= att0 && j < att0 + 3) {
        continue;
      }
      float accum = 0.0f;
      for (int k = 0; k < 3; ++k) {
        accum += G_reset_theta[i][k] * p_ab[k][j < att0 ? j : j - 3];
      }
      p[att0 + i][j] = accum;
      p[j][att0 + i] = accum;
    }
  }

  sf_loose_symmetrize_p(p);
}

static void sf_loose_batch_update_joseph(sf_loose_t *loose,
                                         int obs_count,
                                         const float h[][SF_LOOSE_ERROR_STATES],
                                         const int *const h_supports[],
                                         const int h_support_lens[],
                                         const float residuals[],
                                         const float variances[]) {
  double p[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES];
  double dx[SF_LOOSE_ERROR_STATES] = {0};
  memcpy(p, loose->p64, sizeof(p));
  int dense_support[SF_LOOSE_ERROR_STATES];

  for (int obs = 0; obs < obs_count; ++obs) {
    const int *support = dense_support;
    int support_len = 0;
    if (h_supports != NULL && h_support_lens != NULL && h_supports[obs] != NULL && h_support_lens[obs] > 0) {
      support = h_supports[obs];
      support_len = h_support_lens[obs];
    } else {
      sf_loose_extract_support_from_row(h[obs], dense_support, &support_len);
    }
    double ph[SF_LOOSE_ERROR_STATES];
    double s = (double)variances[obs];
    for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
      ph[i] = 0.0;
      for (int j = 0; j < support_len; ++j) {
        const int state = support[j];
        ph[i] += p[i][state] * (double)h[obs][state];
      }
    }
    for (int i = 0; i < support_len; ++i) {
      const int state = support[i];
      s += (double)h[obs][state] * ph[state];
    }
    if (s <= 0.0) {
      continue;
    }
    double hd = 0.0;
    for (int i = 0; i < support_len; ++i) {
      const int state = support[i];
      hd += (double)h[obs][state] * dx[state];
    }
    for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
      dx[i] += (ph[i] / s) * ((double)residuals[obs] - hd);
    }
    /* Scalar Joseph update simplifies exactly to P - (P h^T)(P h^T)^T / s. */
    for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
      for (int j = i; j < SF_LOOSE_ERROR_STATES; ++j) {
        const double updated = p[i][j] - (ph[i] * ph[j]) / s;
        p[i][j] = updated;
        p[j][i] = updated;
      }
    }
  }

  memcpy(loose->p64, p, sizeof(loose->p64));
  sf_loose_sync_covariance_from_shadow(loose);
  float dx_f32[SF_LOOSE_ERROR_STATES];
  for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
    dx_f32[i] = (float)dx[i];
  }
  sf_loose_inject_error_state(loose, dx_f32);
}

static void sf_loose_quat_to_dcm(const float q_in[4], float c[3][3]) {
  float q[4] = {q_in[0], q_in[1], q_in[2], q_in[3]};
  sf_loose_normalize_quat(q);
  const float q1_2 = q[1] * q[1];
  const float q2_2 = q[2] * q[2];
  const float q3_2 = q[3] * q[3];
  c[0][0] = 1.0f - 2.0f * (q2_2 + q3_2);
  c[1][1] = 1.0f - 2.0f * (q1_2 + q3_2);
  c[2][2] = 1.0f - 2.0f * (q1_2 + q2_2);
  c[0][1] = 2.0f * (q[1] * q[2] - q[0] * q[3]);
  c[1][0] = 2.0f * (q[1] * q[2] + q[0] * q[3]);
  c[0][2] = 2.0f * (q[1] * q[3] + q[0] * q[2]);
  c[2][0] = 2.0f * (q[1] * q[3] - q[0] * q[2]);
  c[1][2] = 2.0f * (q[2] * q[3] - q[0] * q[1]);
  c[2][1] = 2.0f * (q[2] * q[3] + q[0] * q[1]);
}

static void sf_loose_dcm_ecef_to_ned(float lat_rad, float lon_rad, float c[3][3]) {
  const float sin_lat = sinf(lat_rad);
  const float cos_lat = cosf(lat_rad);
  const float sin_lon = sinf(lon_rad);
  const float cos_lon = cosf(lon_rad);
  c[0][0] = -sin_lat * cos_lon;
  c[0][1] = -sin_lat * sin_lon;
  c[0][2] = cos_lat;
  c[1][0] = -sin_lon;
  c[1][1] = cos_lon;
  c[1][2] = 0.0f;
  c[2][0] = -cos_lat * cos_lon;
  c[2][1] = -cos_lat * sin_lon;
  c[2][2] = -sin_lat;
}

static void sf_loose_ecef_to_llh(const float x_e[3], float *lat_rad, float *lon_rad, float *height_m) {
  const float a2 = SF_WGS84_A * SF_WGS84_A;
  const float b2 = SF_WGS84_B * SF_WGS84_B;
  const float z2 = x_e[2] * x_e[2];
  const float r2 = x_e[0] * x_e[0] + x_e[1] * x_e[1];
  const float r = sqrtf(r2);
  const float f = 54.0f * b2 * z2;
  const float g = r2 + (1.0f - SF_WGS84_E2) * z2 - SF_WGS84_E2 * (a2 - b2);
  const float c = SF_WGS84_E2 * SF_WGS84_E2 * f * r2 / (g * g * g);
  const float s = cbrtf(1.0f + c + sqrtf(c * c + 2.0f * c));
  const float p = f / (3.0f * (s + 1.0f / s + 1.0f) * (s + 1.0f / s + 1.0f) * g * g);
  const float q = sqrtf(1.0f + 2.0f * SF_WGS84_E2 * SF_WGS84_E2 * p);
  const float r0 = -p * SF_WGS84_E2 * r / (1.0f + q) +
                   sqrtf(0.5f * a2 * (1.0f + 1.0f / q) -
                         p * (1.0f - SF_WGS84_E2) * z2 / (q * (1.0f + q)) -
                         0.5f * p * r2);
  const float tmp = (r - SF_WGS84_E2 * r0) * (r - SF_WGS84_E2 * r0);
  const float u = sqrtf(tmp + z2);
  const float v = sqrtf(tmp + (1.0f - SF_WGS84_E2) * z2);
  const float inv_av = 1.0f / (SF_WGS84_A * v);
  const float z0 = b2 * x_e[2] * inv_av;
  *height_m = u * (1.0f - b2 * inv_av);
  *lat_rad = atan2f(x_e[2] + (a2 / b2 - 1.0f) * z0, r);
  *lon_rad = atan2f(x_e[1], x_e[0]);
}

static void sf_loose_gravity_ecef_j2(const float x_e[3], float g_e[3]) {
  const float r = sqrtf(x_e[0] * x_e[0] + x_e[1] * x_e[1] + x_e[2] * x_e[2]);
  if (r <= 0.0f) {
    g_e[0] = g_e[1] = g_e[2] = 0.0f;
    return;
  }
  const float r2 = r * r;
  const float r3 = r * r2;
  const float tmp1 = SF_WGS84_GM / r3;
  const float tmp2 = 1.5f * (SF_WGS84_A * (SF_WGS84_A * SF_WGS84_J2)) / r2;
  const float tmp3 = 5.0f * x_e[2] * x_e[2] / r2;
  g_e[0] = tmp1 * (-x_e[0] - tmp2 * (x_e[0] - tmp3 * x_e[0])) + SF_WGS84_OMEGA_IE * SF_WGS84_OMEGA_IE * x_e[0];
  g_e[1] = tmp1 * (-x_e[1] - tmp2 * (x_e[1] - tmp3 * x_e[1])) + SF_WGS84_OMEGA_IE * SF_WGS84_OMEGA_IE * x_e[1];
  g_e[2] = tmp1 * (-x_e[2] - tmp2 * (3.0f * x_e[2] - tmp3 * x_e[2]));
}

static float sf_loose_vec_norm3(const float v[3]) {
  return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static int sf_loose_test_chi2_scalar(float residual,
                                     const float p[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
                                     const float h[SF_LOOSE_ERROR_STATES],
                                     float r) {
  float s = r;
  for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
    for (int j = 0; j < SF_LOOSE_ERROR_STATES; ++j) {
      s += h[i] * p[i][j] * h[j];
    }
  }
  return fabsf(residual) > 3.0f * sqrtf(fmaxf(s, 0.0f));
}

static int sf_loose_test_chi2_vec3(const float residual[3],
                                   const float p[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
                                   const float h[3][SF_LOOSE_ERROR_STATES],
                                   const float r_diag[3]) {
  for (int row = 0; row < 3; ++row) {
    if (sf_loose_test_chi2_scalar(residual[row], p, h[row], r_diag[row])) {
      return 1;
    }
  }
  return 0;
}
