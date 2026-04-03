#include "sensor_fusion_internal.h"
#include "sf_eskf.h"
#include "sf_loose.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static void test_scaffold_runs(void) { TEST_ASSERT_TRUE(1); }

static float quat_yaw_rad(float q0, float q1, float q2, float q3) {
  return atan2f(2.0f * (q0 * q3 + q1 * q2),
                1.0f - 2.0f * (q2 * q2 + q3 * q3));
}

static void test_predict_noise_defaults(void) {
  sf_predict_noise_t noise;
  sf_predict_noise_default(&noise);

  TEST_ASSERT_FLOAT_WITHIN(1.0e-9f, 0.0001f, noise.gyro_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 12.0f, noise.accel_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-15f, 0.002e-9f, noise.gyro_bias_rw_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-13f, 0.2e-9f, noise.accel_bias_rw_var);
}

static void test_external_misalignment_sets_mount_ready(void) {
  sf_sensor_fusion_t fusion;
  sf_fusion_config_t cfg;
  float q_vb[4] = {0.9238795f, 0.0f, 0.0f, 0.3826834f};
  float out_q_vb[4] = {0};

  sf_fusion_config_default(&cfg);
  sf_fusion_init_external(&fusion, &cfg, q_vb);

  TEST_ASSERT_TRUE(sf_fusion_mount_ready(&fusion));
  TEST_ASSERT_TRUE(sf_fusion_mount_q_vb(&fusion, out_q_vb));
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, q_vb[0], out_q_vb[0]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, q_vb[3], out_q_vb[3]);
}

static void test_sensor_fusion_external_mode_initializes_and_predicts(void) {
  sf_sensor_fusion_t fusion;
  sf_sensor_fusion_impl_t *impl = sf_impl(&fusion);
  sf_fusion_config_t cfg;
  sf_imu_sample_t imu0 = {
      .t_s = 10.00,
      .gyro_radps = {0.0f, 0.0f, 0.1f},
      .accel_mps2 = {0.0f, 0.0f, -9.80665f},
  };
  sf_imu_sample_t imu1 = {
      .t_s = 10.01,
      .gyro_radps = {0.0f, 0.0f, 0.1f},
      .accel_mps2 = {0.0f, 0.0f, -9.80665f},
  };
  sf_gnss_sample_t gnss = {
      .t_s = 10.00,
      .pos_ned_m = {100.0f, 50.0f, -2.0f},
      .vel_ned_mps = {5.0f, 0.0f, 0.0f},
      .pos_std_m = {0.5f, 0.5f, 1.0f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
      .heading_valid = true,
      .heading_rad = 0.25f,
  };
  float q_vb[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  sf_update_t upd;

  sf_fusion_config_default(&cfg);
  sf_fusion_init_external(&fusion, &cfg, q_vb);

  upd = sf_fusion_process_gnss(&fusion, &gnss);
  TEST_ASSERT_TRUE(upd.mount_ready);
  TEST_ASSERT_TRUE(upd.ekf_initialized);
  TEST_ASSERT_TRUE(upd.ekf_initialized_now);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, gnss.pos_ned_m[0], impl->eskf.nominal.pn);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, gnss.pos_ned_m[1], impl->eskf.nominal.pe);

  upd = sf_fusion_process_imu(&fusion, &imu0);
  TEST_ASSERT_TRUE(upd.ekf_initialized);
  upd = sf_fusion_process_imu(&fusion, &imu1);
  TEST_ASSERT_TRUE(upd.ekf_initialized);
  TEST_ASSERT_TRUE(impl->eskf.nominal.q3 > 0.0f);
}

static void test_stationary_bootstrap_identity_mount(void) {
  const float samples[4][3] = {
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
  };
  sf_stationary_mount_bootstrap_t boot;

  TEST_ASSERT_TRUE(
      sf_bootstrap_vehicle_to_body_from_stationary(samples, 4U, 0.0f, &boot));
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, boot.mean_accel_b[0]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, boot.mean_accel_b[1]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, -9.80665f, boot.mean_accel_b[2]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.0f, boot.c_b_v[0][0]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.0f, boot.c_b_v[1][1]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.0f, boot.c_b_v[2][2]);
}

static void test_align_initialize_from_stationary_sets_mount(void) {
  sf_align_runtime_t align_rt;
  sf_align_config_t cfg;
  const float samples[8][3] = {
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
  };

  sf_align_config_default(&cfg);
  sf_align_init(&align_rt, &cfg);
  TEST_ASSERT_TRUE(
      sf_align_initialize_from_stationary(&align_rt, &cfg, samples, 8U, 0.0f));
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.0f, align_rt.state.q_vb[0]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.0f, align_rt.state.q_vb[1]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.0f, align_rt.state.q_vb[2]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.0f, align_rt.state.q_vb[3]);
}

static void test_align_update_window_reduces_yaw_sigma_on_straight_motion(void) {
  sf_align_runtime_t align_rt;
  sf_align_config_t cfg;
  sf_align_window_summary_t window = {0};
  sf_align_update_trace_t trace;
  const float samples[8][3] = {
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
      {0.0f, 0.0f, -9.80665f},
  };
  float yaw_sigma_before;
  float yaw_sigma_after;

  sf_align_config_default(&cfg);
  sf_align_init(&align_rt, &cfg);
  TEST_ASSERT_TRUE(
      sf_align_initialize_from_stationary(&align_rt, &cfg, samples, 8U, 0.0f));

  yaw_sigma_before = sqrtf(align_rt.state.p[2][2]);
  window.dt = 1.0f;
  window.mean_gyro_b[0] = 0.0f;
  window.mean_gyro_b[1] = 0.0f;
  window.mean_gyro_b[2] = 0.0f;
  window.mean_accel_b[0] = 1.5f;
  window.mean_accel_b[1] = 0.0f;
  window.mean_accel_b[2] = -9.80665f;
  window.gnss_vel_prev_n[0] = 0.0f;
  window.gnss_vel_prev_n[1] = 0.0f;
  window.gnss_vel_prev_n[2] = 0.0f;
  window.gnss_vel_curr_n[0] = 3.0f;
  window.gnss_vel_curr_n[1] = 0.0f;
  window.gnss_vel_curr_n[2] = 0.0f;

  (void)sf_align_update_window_with_trace(&align_rt, &cfg, &window, &trace);
  yaw_sigma_after = sqrtf(align_rt.state.p[2][2]);
  TEST_ASSERT_TRUE(trace.after_horiz_accel_valid);
  TEST_ASSERT_TRUE(yaw_sigma_after < yaw_sigma_before);
}

static void test_eskf_init_and_nominal_predict_rest_case(void) {
  sf_eskf_t eskf;
  sf_eskf_imu_delta_t imu = {
      .dax = 0.0f,
      .day = 0.0f,
      .daz = 0.0f,
      .dvx = 0.0f,
      .dvy = 0.0f,
      .dvz = -SF_GRAVITY_MSS * 0.01f,
      .dt = 0.01f,
  };

  sf_eskf_init(&eskf, NULL, NULL);
  sf_eskf_predict_nominal(&eskf, &imu);

  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, eskf.nominal.q0);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, eskf.nominal.q1);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, eskf.nominal.q2);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, eskf.nominal.q3);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.0f, eskf.nominal.vn);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.0f, eskf.nominal.ve);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.0f, eskf.nominal.vd);
}

static void test_eskf_error_transition_has_expected_rest_structure(void) {
  sf_eskf_t eskf;
  sf_eskf_imu_delta_t imu = {
      .dax = 0.0f,
      .day = 0.0f,
      .daz = 0.0f,
      .dvx = 0.0f,
      .dvy = 0.0f,
      .dvz = 0.0f,
      .dt = 0.01f,
  };
  float F[SF_ESKF_ERROR_STATES][SF_ESKF_ERROR_STATES];
  float G[SF_ESKF_ERROR_STATES][SF_ESKF_NOISE_STATES];

  sf_eskf_init(&eskf, NULL, NULL);
  sf_eskf_compute_error_transition(F, G, &eskf, &imu);

  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, F[0][0]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, F[1][1]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, F[2][2]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, F[9][9]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, F[12][12]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.01f, F[6][3]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.01f, F[7][4]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.01f, F[8][5]);
  TEST_ASSERT_TRUE(G[0][0] != 0.0f);
  TEST_ASSERT_TRUE(G[3][3] != 0.0f);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.01f, G[9][6]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.01f, G[10][7]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.01f, G[11][8]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.01f, G[12][9]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.01f, G[13][10]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.01f, G[14][11]);
}

static void test_eskf_predict_updates_covariance_symmetrically(void) {
  sf_eskf_t eskf;
  sf_eskf_imu_delta_t imu = {
      .dax = 0.001f,
      .day = -0.002f,
      .daz = 0.003f,
      .dvx = 0.01f,
      .dvy = -0.02f,
      .dvz = -SF_GRAVITY_MSS * 0.01f,
      .dt = 0.01f,
  };

  sf_eskf_init(&eskf, NULL, NULL);
  sf_eskf_predict(&eskf, &imu);

  TEST_ASSERT_TRUE(eskf.p[0][0] > 0.0f);
  TEST_ASSERT_TRUE(eskf.p[3][3] > 0.0f);
  TEST_ASSERT_TRUE(eskf.p[9][9] > 0.0f);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, eskf.p[0][1], eskf.p[1][0]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, eskf.p[3][6], eskf.p[6][3]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, eskf.p[9][12], eskf.p[12][9]);
}

static void test_eskf_fuse_gps_moves_nominal_state_toward_measurement(void) {
  sf_eskf_t eskf;
  sf_gnss_sample_t gps = {
      .pos_ned_m = {10.0f, -5.0f, 2.0f},
      .vel_ned_mps = {3.0f, -1.0f, 0.5f},
      .pos_std_m = {0.5f, 0.5f, 0.5f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
  };
  float p_pos_before;
  float p_vel_before;

  sf_eskf_init(&eskf, NULL, NULL);
  eskf.nominal.pn = 0.0f;
  eskf.nominal.pe = 0.0f;
  eskf.nominal.pd = 0.0f;
  eskf.nominal.vn = 0.0f;
  eskf.nominal.ve = 0.0f;
  eskf.nominal.vd = 0.0f;
  p_pos_before = eskf.p[6][6];
  p_vel_before = eskf.p[3][3];

  sf_eskf_fuse_gps(&eskf, &gps);

  TEST_ASSERT_TRUE(eskf.nominal.pn > 0.0f);
  TEST_ASSERT_TRUE(eskf.nominal.pe < 0.0f);
  TEST_ASSERT_TRUE(eskf.nominal.pd > 0.0f);
  TEST_ASSERT_TRUE(eskf.nominal.vn > 0.0f);
  TEST_ASSERT_TRUE(eskf.nominal.ve < 0.0f);
  TEST_ASSERT_TRUE(eskf.nominal.vd > 0.0f);
  TEST_ASSERT_TRUE(eskf.p[6][6] < p_pos_before);
  TEST_ASSERT_TRUE(eskf.p[3][3] < p_vel_before);
}

static void test_eskf_fuse_body_vel_reduces_lateral_and_vertical_velocity(void) {
  sf_eskf_t eskf;
  float py_before;
  float pz_before;

  sf_eskf_init(&eskf, NULL, NULL);
  eskf.nominal.ve = 1.5f;
  eskf.nominal.vd = -0.7f;
  py_before = eskf.p[4][4];
  pz_before = eskf.p[5][5];

  sf_eskf_fuse_body_vel(&eskf, 5.0f);

  TEST_ASSERT_TRUE(fabsf(eskf.nominal.ve) < 1.5f);
  TEST_ASSERT_TRUE(fabsf(eskf.nominal.vd) < 0.7f);
  TEST_ASSERT_TRUE(eskf.p[4][4] < py_before);
  TEST_ASSERT_TRUE(eskf.p[5][5] < pz_before);
}

static void test_eskf_body_vel_y_reduces_small_yaw_error_on_forward_motion(void) {
  sf_eskf_t eskf;
  const float yaw_before = 10.0f * 3.1415927f / 180.0f;
  const float half_yaw = 0.5f * yaw_before;
  float yaw_after;

  sf_eskf_init(&eskf, NULL, NULL);
  eskf.nominal.q0 = cosf(half_yaw);
  eskf.nominal.q1 = 0.0f;
  eskf.nominal.q2 = 0.0f;
  eskf.nominal.q3 = sinf(half_yaw);
  eskf.nominal.vn = 5.0f;
  eskf.nominal.ve = 0.0f;
  eskf.nominal.vd = 0.0f;
  eskf.p[0][0] = 1.0f;
  eskf.p[1][1] = 1.0f;
  eskf.p[2][2] = 1.0f;
  eskf.p[3][3] = 1.0e-6f;
  eskf.p[4][4] = 1.0e-6f;
  eskf.p[5][5] = 1.0e-6f;

  sf_eskf_fuse_body_vel(&eskf, 5.0f);
  yaw_after = quat_yaw_rad(eskf.nominal.q0, eskf.nominal.q1, eskf.nominal.q2, eskf.nominal.q3);

  TEST_ASSERT_TRUE(fabsf(yaw_after) < fabsf(yaw_before));
}

static void test_eskf_zero_vel_reduces_forward_velocity_when_stopped(void) {
  sf_eskf_t eskf;
  float px_before;

  sf_eskf_init(&eskf, NULL, NULL);
  eskf.nominal.vn = 0.3f;
  eskf.nominal.ve = 0.0f;
  eskf.nominal.vd = 0.0f;
  px_before = eskf.p[3][3];

  sf_eskf_fuse_zero_vel(&eskf, 0.01f);

  TEST_ASSERT_TRUE(fabsf(eskf.nominal.vn) < 0.3f);
  TEST_ASSERT_TRUE(eskf.p[3][3] < px_before);
}

static void test_loose_init_copies_all_noise_fields_and_covariance_diag(void) {
  sf_loose_t loose;
  float p_diag[SF_LOOSE_ERROR_STATES];
  sf_loose_predict_noise_t noise = {
      .gyro_var = 0.11f,
      .accel_var = 0.22f,
      .gyro_bias_rw_var = 0.33f,
      .accel_bias_rw_var = 0.44f,
      .gyro_scale_rw_var = 0.55f,
      .accel_scale_rw_var = 0.66f,
      .mount_align_rw_var = 0.77f,
  };

  for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
    p_diag[i] = 0.01f * (float)(i + 1);
  }

  sf_loose_init(&loose, p_diag, &noise);

  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, loose.nominal.q0);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.11f, loose.noise.gyro_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.22f, loose.noise.accel_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.33f, loose.noise.gyro_bias_rw_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.44f, loose.noise.accel_bias_rw_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.55f, loose.noise.gyro_scale_rw_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.66f, loose.noise.accel_scale_rw_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.77f, loose.noise.mount_align_rw_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, p_diag[0], loose.p[0][0]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, p_diag[9], loose.p[9][9]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, p_diag[21], loose.p[21][21]);
}

static void test_loose_error_transition_has_expected_reference_structure(void) {
  sf_loose_t loose;
  sf_loose_imu_delta_t imu = {
      .dax_1 = 0.0f,
      .day_1 = 0.0f,
      .daz_1 = 0.0f,
      .dvx_1 = 0.0f,
      .dvy_1 = 0.0f,
      .dvz_1 = 0.0f,
      .dax_2 = 0.0f,
      .day_2 = 0.0f,
      .daz_2 = 0.0f,
      .dvx_2 = 0.0f,
      .dvy_2 = 0.0f,
      .dvz_2 = 0.0f,
      .dt = 0.01f,
  };
  float F[SF_LOOSE_ERROR_STATES][SF_LOOSE_ERROR_STATES];
  float G[SF_LOOSE_ERROR_STATES][SF_LOOSE_NOISE_STATES];

  sf_loose_init(&loose, NULL, NULL);
  loose.nominal.q0 = 1.0f;
  loose.nominal.qcs0 = 1.0f;
  loose.nominal.sax = 1.0f;
  loose.nominal.say = 1.0f;
  loose.nominal.saz = 1.0f;
  loose.nominal.sgx = 1.0f;
  loose.nominal.sgy = 1.0f;
  loose.nominal.sgz = 1.0f;

  sf_loose_compute_error_transition(F, G, &loose, &imu);

  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, F[0][0]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, F[3][3]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, F[21][21]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.01f, F[0][3]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.01f, F[1][4]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.01f, F[2][5]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-8f, 2.0f * 7.292115e-5f * 0.01f, F[3][4]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-8f, -2.0f * 7.292115e-5f * 0.01f, F[4][3]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, G[3][0]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, G[4][1]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, G[5][2]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, G[6][3]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, G[7][4]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, G[8][5]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, G[9][6]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, G[12][9]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, G[15][12]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, G[18][15]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 1.0f, G[21][18]);
}

static void test_loose_predict_with_zero_dt_is_noop(void) {
  sf_loose_t loose;
  sf_loose_imu_delta_t imu = {
      .dax_1 = 1.0f,
      .day_1 = -2.0f,
      .daz_1 = 3.0f,
      .dvx_1 = 4.0f,
      .dvy_1 = -5.0f,
      .dvz_1 = 6.0f,
      .dax_2 = 1.0f,
      .day_2 = -2.0f,
      .daz_2 = 3.0f,
      .dvx_2 = 4.0f,
      .dvy_2 = -5.0f,
      .dvz_2 = 6.0f,
      .dt = 0.0f,
  };
  float p_diag[SF_LOOSE_ERROR_STATES];
  sf_loose_nominal_state_t before;
  float p00_before;
  float p99_before;

  for (int i = 0; i < SF_LOOSE_ERROR_STATES; ++i) {
    p_diag[i] = 0.02f * (float)(i + 1);
  }

  sf_loose_init(&loose, p_diag, NULL);
  loose.nominal.q0 = 1.0f;
  loose.nominal.pn = 6378137.0f;
  loose.pos_e64[0] = 6378137.0;
  loose.qcs64[0] = 1.0;
  loose.nominal.vn = 1.0f;
  loose.nominal.ve = 2.0f;
  loose.nominal.vd = 3.0f;
  before = loose.nominal;
  p00_before = loose.p[0][0];
  p99_before = loose.p[9][9];

  sf_loose_predict(&loose, &imu);

  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, before.q0, loose.nominal.q0);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, before.pn, loose.nominal.pn);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, before.vn, loose.nominal.vn);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, before.ve, loose.nominal.ve);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, before.vd, loose.nominal.vd);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, p00_before, loose.p[0][0]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, p99_before, loose.p[9][9]);
}

static void test_loose_nominal_predict_keeps_equatorial_rest_state_near_rest(void) {
  sf_loose_t loose;
  sf_loose_imu_delta_t imu = {
      .dax_1 = 0.0f,
      .day_1 = 0.0f,
      .daz_1 = 0.0f,
      .dvx_1 = 9.7983f * 0.01f,
      .dvy_1 = 0.0f,
      .dvz_1 = 0.0f,
      .dax_2 = 0.0f,
      .day_2 = 0.0f,
      .daz_2 = 0.0f,
      .dvx_2 = 9.7983f * 0.01f,
      .dvy_2 = 0.0f,
      .dvz_2 = 0.0f,
      .dt = 0.01f,
  };

  sf_loose_init(&loose, NULL, NULL);
  loose.nominal.q0 = 1.0f;
  loose.nominal.sax = 1.0f;
  loose.nominal.say = 1.0f;
  loose.nominal.saz = 1.0f;
  loose.nominal.sgx = 1.0f;
  loose.nominal.sgy = 1.0f;
  loose.nominal.sgz = 1.0f;
  loose.nominal.qcs0 = 1.0f;
  loose.qcs64[0] = 1.0;
  loose.nominal.pn = 6378137.0f;
  loose.nominal.pe = 0.0f;
  loose.nominal.pd = 0.0f;
  loose.pos_e64[0] = 6378137.0;
  loose.pos_e64[1] = 0.0;
  loose.pos_e64[2] = 0.0;

  sf_loose_predict_nominal(&loose, &imu);

  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 1.0f, loose.nominal.q0);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.0f, loose.nominal.q1);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.0f, loose.nominal.q2);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 0.0f, loose.nominal.q3);
  TEST_ASSERT_TRUE(fabsf(loose.nominal.vn) < 5.0e-3f);
  TEST_ASSERT_TRUE(fabsf(loose.nominal.ve) < 5.0e-3f);
  TEST_ASSERT_TRUE(fabsf(loose.nominal.vd) < 5.0e-3f);
}

static void test_loose_reference_gps_update_moves_ecef_position_toward_measurement(void) {
  sf_loose_t loose;
  const float pos_meas[3] = {6378147.0f, 2.0f, -3.0f};
  float before_err;
  float after_err;
  float p_before;

  sf_loose_init(&loose, NULL, NULL);
  loose.nominal.q0 = 1.0f;
  loose.nominal.pn = 6378137.0f;
  loose.nominal.pe = 0.0f;
  loose.nominal.pd = 0.0f;
  loose.pos_e64[0] = 6378137.0;
  loose.pos_e64[1] = 0.0;
  loose.pos_e64[2] = 0.0;
  loose.qcs64[0] = 1.0;
  loose.p[0][0] = 25.0f;
  loose.p[1][1] = 25.0f;
  loose.p[2][2] = 25.0f;
  p_before = loose.p[0][0];
  before_err = fabsf(pos_meas[0] - loose.nominal.pn);

  sf_loose_fuse_gps_reference(&loose, pos_meas, 0.5f, 1.0f);

  after_err = fabsf(pos_meas[0] - loose.nominal.pn);
  TEST_ASSERT_TRUE(after_err < before_err);
  TEST_ASSERT_TRUE(loose.p[0][0] < p_before);
}

static void test_loose_reference_nhc_reduces_lateral_and_vertical_car_velocity(void) {
  sf_loose_t loose;
  const float gyro_radps[3] = {0.0f, 0.0f, 0.0f};
  const float accel_mps2[3] = {9.81f, 0.0f, 0.0f};
  float vy_before;
  float vz_before;

  sf_loose_init(&loose, NULL, NULL);
  loose.nominal.q0 = 1.0f;
  loose.nominal.qcs0 = 1.0f;
  loose.qcs64[0] = 1.0;
  loose.nominal.sax = 1.0f;
  loose.nominal.say = 1.0f;
  loose.nominal.saz = 1.0f;
  loose.nominal.sgx = 1.0f;
  loose.nominal.sgy = 1.0f;
  loose.nominal.sgz = 1.0f;
  loose.nominal.vn = 4.0f;
  loose.nominal.ve = 0.8f;
  loose.nominal.vd = -0.6f;
  loose.p[3][3] = 1.0e-3f;
  loose.p[4][4] = 1.0f;
  loose.p[5][5] = 1.0f;
  loose.p[6][6] = 1.0f;
  loose.p[7][7] = 1.0f;
  loose.p[8][8] = 1.0f;
  vy_before = loose.nominal.ve;
  vz_before = loose.nominal.vd;

  sf_loose_fuse_nhc_reference(&loose, gyro_radps, accel_mps2, 0.01f);

  TEST_ASSERT_TRUE(fabsf(loose.nominal.ve) < fabsf(vy_before));
  TEST_ASSERT_TRUE(fabsf(loose.nominal.vd) < fabsf(vz_before));
}

static void test_sensor_fusion_internal_mode_bootstraps_align_state(void) {
  sf_sensor_fusion_t fusion;
  sf_sensor_fusion_impl_t *impl = sf_impl(&fusion);
  sf_fusion_config_t cfg;
  sf_gnss_sample_t gnss0 = {
      .t_s = 0.0,
      .pos_ned_m = {0.0f, 0.0f, 0.0f},
      .vel_ned_mps = {0.0f, 0.0f, 0.0f},
      .pos_std_m = {0.5f, 0.5f, 1.0f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
      .heading_valid = true,
      .heading_rad = 0.0f,
  };
  sf_gnss_sample_t gnss1 = {
      .t_s = 1.0,
      .pos_ned_m = {0.0f, 0.0f, 0.0f},
      .vel_ned_mps = {0.0f, 0.0f, 0.0f},
      .pos_std_m = {0.5f, 0.5f, 1.0f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
      .heading_valid = true,
      .heading_rad = 0.0f,
  };
  sf_gnss_sample_t gnss2 = {
      .t_s = 2.0,
      .pos_ned_m = {0.0f, 0.0f, 0.0f},
      .vel_ned_mps = {3.0f, 0.0f, 0.0f},
      .pos_std_m = {0.5f, 0.5f, 1.0f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
      .heading_valid = true,
      .heading_rad = 0.0f,
  };
  sf_update_t upd;

  sf_fusion_config_default(&cfg);
  cfg.bootstrap.stationary_samples = 4U;
  sf_fusion_init_internal(&fusion, &cfg);

  upd = sf_fusion_process_gnss(&fusion, &gnss0);
  TEST_ASSERT_FALSE(upd.mount_ready);

  for (int i = 0; i < 4; ++i) {
    sf_imu_sample_t imu = {
        .t_s = 0.10f + 0.10f * (float)i,
        .gyro_radps = {0.0f, 0.0f, 0.0f},
        .accel_mps2 = {0.0f, 0.0f, -9.80665f},
    };
    upd = sf_fusion_process_imu(&fusion, &imu);
    TEST_ASSERT_FALSE(upd.mount_ready);
  }

  upd = sf_fusion_process_gnss(&fusion, &gnss1);
  TEST_ASSERT_TRUE(impl->align_initialized);
  TEST_ASSERT_TRUE(upd.mount_q_vb_valid);
  TEST_ASSERT_TRUE(impl->mount_q_vb_valid);
  upd = sf_fusion_process_gnss(&fusion, &gnss2);
  TEST_ASSERT_TRUE(upd.mount_q_vb_valid);
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_scaffold_runs);
  RUN_TEST(test_predict_noise_defaults);
  RUN_TEST(test_external_misalignment_sets_mount_ready);
  RUN_TEST(test_sensor_fusion_external_mode_initializes_and_predicts);
  RUN_TEST(test_stationary_bootstrap_identity_mount);
  RUN_TEST(test_align_initialize_from_stationary_sets_mount);
  RUN_TEST(test_align_update_window_reduces_yaw_sigma_on_straight_motion);
  RUN_TEST(test_eskf_init_and_nominal_predict_rest_case);
  RUN_TEST(test_eskf_error_transition_has_expected_rest_structure);
  RUN_TEST(test_eskf_predict_updates_covariance_symmetrically);
  RUN_TEST(test_eskf_fuse_gps_moves_nominal_state_toward_measurement);
  RUN_TEST(test_eskf_fuse_body_vel_reduces_lateral_and_vertical_velocity);
  RUN_TEST(test_eskf_body_vel_y_reduces_small_yaw_error_on_forward_motion);
  RUN_TEST(test_eskf_zero_vel_reduces_forward_velocity_when_stopped);
  RUN_TEST(test_loose_init_copies_all_noise_fields_and_covariance_diag);
  RUN_TEST(test_loose_error_transition_has_expected_reference_structure);
  RUN_TEST(test_loose_predict_with_zero_dt_is_noop);
  RUN_TEST(test_loose_nominal_predict_keeps_equatorial_rest_state_near_rest);
  RUN_TEST(test_loose_reference_gps_update_moves_ecef_position_toward_measurement);
  RUN_TEST(test_loose_reference_nhc_reduces_lateral_and_vertical_car_velocity);
  RUN_TEST(test_sensor_fusion_internal_mode_bootstraps_align_state);
  return UNITY_END();
}
