#ifndef SF_LOOSE_H
#define SF_LOOSE_H

#include "sensor_fusion.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SF_LOOSE_NOMINAL_STATES 26
#define SF_LOOSE_ERROR_STATES 24
#define SF_LOOSE_NOISE_STATES 21

typedef struct {
  float gyro_var;
  float accel_var;
  float gyro_bias_rw_var;
  float accel_bias_rw_var;
  float gyro_scale_rw_var;
  float accel_scale_rw_var;
  float mount_align_rw_var;
} sf_loose_predict_noise_t;

typedef struct {
  float q0, q1, q2, q3;
  float vn, ve, vd;
  float pn, pe, pd;
  float bgx, bgy, bgz;
  float bax, bay, baz;
  float sgx, sgy, sgz;
  float sax, say, saz;
  float qcs0, qcs1, qcs2, qcs3;
} sf_loose_nominal_state_t;

typedef struct {
  float dtheta_x, dtheta_y, dtheta_z;
  float dv_n, dv_e, dv_d;
  float dp_n, dp_e, dp_d;
  float dbg_x, dbg_y, dbg_z;
  float dba_x, dba_y, dba_z;
  float dsg_x, dsg_y, dsg_z;
  float dsa_x, dsa_y, dsa_z;
  float dpsi_cc_x, dpsi_cc_y, dpsi_cc_z;
} sf_loose_error_state_t;

typedef struct sf_loose {
  sf_loose_nominal_state_t nominal;
  float p[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES];
  sf_loose_predict_noise_t noise;
  double pos_e64[3];
  double qcs64[4];
  double p64[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES];
  int last_obs_count;
  int last_obs_types[5];
} sf_loose_t;

typedef struct {
  float dax_1;
  float day_1;
  float daz_1;
  float dvx_1;
  float dvy_1;
  float dvz_1;
  float dax_2;
  float day_2;
  float daz_2;
  float dvx_2;
  float dvy_2;
  float dvz_2;
  float dt;
} sf_loose_imu_delta_t;

void sf_loose_init(sf_loose_t *loose,
                   const float p_diag[SF_LOOSE_ERROR_STATES],
                   const sf_loose_predict_noise_t *noise);

void sf_loose_predict(sf_loose_t *loose, const sf_loose_imu_delta_t *imu);
void sf_loose_predict_nominal(sf_loose_t *loose, const sf_loose_imu_delta_t *imu);
void sf_loose_fuse_gps(sf_loose_t *loose, const sf_gnss_sample_t *gps);
void sf_loose_fuse_gps_reference(sf_loose_t *loose,
                                 const double pos_ecef_m[3],
                                 float h_acc_m,
                                 float dt_since_last_gnss_s);
void sf_loose_fuse_reference_batch(sf_loose_t *loose,
                                   const double pos_ecef_m[3],
                                   float h_acc_m,
                                   float dt_since_last_gnss_s,
                                   const float gyro_radps[3],
                                   const float accel_mps2[3],
                                   float dt_s);
void sf_loose_fuse_body_vel(sf_loose_t *loose, float r_body_vel);
void sf_loose_fuse_nhc_reference(sf_loose_t *loose,
                                 const float gyro_radps[3],
                                 const float accel_mps2[3],
                                 float dt_s);
void sf_loose_fuse_zero_vel(sf_loose_t *loose, float r_zero_vel);

void sf_loose_compute_error_transition(
    float f_out[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES],
    float g_out[SF_LOOSE_ERROR_STATES][SF_LOOSE_NOISE_STATES],
    const sf_loose_t *loose,
    const sf_loose_imu_delta_t *imu);

#ifdef __cplusplus
}
#endif

#endif
