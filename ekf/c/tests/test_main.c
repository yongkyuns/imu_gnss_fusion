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

static void test_transpose3(const float in[3][3], float out[3][3]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[j][i] = in[i][j];
    }
  }
}

static void test_mat3_vec(const float m[3][3], const float v[3], float out[3]) {
  for (int i = 0; i < 3; ++i) {
    out[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
  }
}

static void test_mat3_mul(const float a[3][3], const float b[3][3], float out[3][3]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
}

static void test_quat_normalize(float q[4]) {
  const float norm =
      sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm > 0.0f) {
    q[0] /= norm;
    q[1] /= norm;
    q[2] /= norm;
    q[3] /= norm;
  }
}

static void test_quat_to_rotmat(const float q[4], float r[3][3]) {
  const float q0 = q[0];
  const float q1 = q[1];
  const float q2 = q[2];
  const float q3 = q[3];
  const float q00 = q0 * q0;
  const float q11 = q1 * q1;
  const float q22 = q2 * q2;
  const float q33 = q3 * q3;
  const float q01 = q0 * q1;
  const float q02 = q0 * q2;
  const float q03 = q0 * q3;
  const float q12 = q1 * q2;
  const float q13 = q1 * q3;
  const float q23 = q2 * q3;

  r[0][0] = q00 + q11 - q22 - q33;
  r[0][1] = 2.0f * (q12 - q03);
  r[0][2] = 2.0f * (q13 + q02);
  r[1][0] = 2.0f * (q12 + q03);
  r[1][1] = q00 - q11 + q22 - q33;
  r[1][2] = 2.0f * (q23 - q01);
  r[2][0] = 2.0f * (q13 - q02);
  r[2][1] = 2.0f * (q23 + q01);
  r[2][2] = q00 - q11 - q22 + q33;
}

static void test_quat_from_rpy(float roll_rad, float pitch_rad, float yaw_rad,
                               float out[4]) {
  const float cr = cosf(0.5f * roll_rad);
  const float sr = sinf(0.5f * roll_rad);
  const float cp = cosf(0.5f * pitch_rad);
  const float sp = sinf(0.5f * pitch_rad);
  const float cy = cosf(0.5f * yaw_rad);
  const float sy = sinf(0.5f * yaw_rad);

  out[0] = cr * cp * cy + sr * sp * sy;
  out[1] = sr * cp * cy - cr * sp * sy;
  out[2] = cr * sp * cy + sr * cp * sy;
  out[3] = cr * cp * sy - sr * sp * cy;
  test_quat_normalize(out);
}

static void test_lla_to_ecef(double lat_deg, double lon_deg, double height_m,
                             double out[3]) {
  const double a = 6378137.0;
  const double e2 = 6.69437999014e-3;
  const double lat = lat_deg * 3.14159265358979323846 / 180.0;
  const double lon = lon_deg * 3.14159265358979323846 / 180.0;
  const double sin_lat = sin(lat);
  const double cos_lat = cos(lat);
  const double sin_lon = sin(lon);
  const double cos_lon = cos(lon);
  const double n = a / sqrt(1.0 - e2 * sin_lat * sin_lat);

  out[0] = (n + height_m) * cos_lat * cos_lon;
  out[1] = (n + height_m) * cos_lat * sin_lon;
  out[2] = (n * (1.0 - e2) + height_m) * sin_lat;
}

static void test_ecef_to_lla(const double ecef[3], float out_lla[3]) {
  const double a = 6378137.0;
  const double e2 = 6.69437999014e-3;
  const double b = a * sqrt(1.0 - e2);
  const double ep2 = (a * a - b * b) / (b * b);
  const double x = ecef[0];
  const double y = ecef[1];
  const double z = ecef[2];
  const double p = sqrt(x * x + y * y);
  const double th = atan2(a * z, b * p);
  const double lon = atan2(y, x);
  const double lat = atan2(z + ep2 * b * pow(sin(th), 3.0),
                           p - e2 * a * pow(cos(th), 3.0));
  const double sin_lat = sin(lat);
  const double n = a / sqrt(1.0 - e2 * sin_lat * sin_lat);
  const double h = p / cos(lat) - n;
  out_lla[0] = (float)(lat * 180.0 / 3.14159265358979323846);
  out_lla[1] = (float)(lon * 180.0 / 3.14159265358979323846);
  out_lla[2] = (float)h;
}

static void test_ecef_to_ned_matrix(float lat_deg, float lon_deg, float out[3][3]) {
  const float lat = lat_deg * 3.14159265358979323846f / 180.0f;
  const float lon = lon_deg * 3.14159265358979323846f / 180.0f;
  const float s_lat = sinf(lat);
  const float c_lat = cosf(lat);
  const float s_lon = sinf(lon);
  const float c_lon = cosf(lon);

  out[0][0] = -s_lat * c_lon;
  out[0][1] = -s_lat * s_lon;
  out[0][2] = c_lat;
  out[1][0] = -s_lon;
  out[1][1] = c_lon;
  out[1][2] = 0.0f;
  out[2][0] = -c_lat * c_lon;
  out[2][1] = -c_lat * s_lon;
  out[2][2] = -s_lat;
}

static void test_ned_offset_to_lla(float lat_deg, float lon_deg, float height_m,
                                   const float ned_m[3], float out_lla[3]) {
  double anchor_ecef[3];
  float c_ne[3][3];
  float c_en[3][3];
  double ecef[3];

  test_lla_to_ecef((double)lat_deg, (double)lon_deg, (double)height_m, anchor_ecef);
  test_ecef_to_ned_matrix(lat_deg, lon_deg, c_ne);
  test_transpose3(c_ne, c_en);
  ecef[0] = anchor_ecef[0] + (double)c_en[0][0] * ned_m[0] +
            (double)c_en[0][1] * ned_m[1] + (double)c_en[0][2] * ned_m[2];
  ecef[1] = anchor_ecef[1] + (double)c_en[1][0] * ned_m[0] +
            (double)c_en[1][1] * ned_m[1] + (double)c_en[1][2] * ned_m[2];
  ecef[2] = anchor_ecef[2] + (double)c_en[2][0] * ned_m[0] +
            (double)c_en[2][1] * ned_m[1] + (double)c_en[2][2] * ned_m[2];
  test_ecef_to_lla(ecef, out_lla);
}

static void test_predict_noise_defaults(void) {
  sf_predict_noise_t noise;
  sf_predict_noise_default(&noise);

  TEST_ASSERT_FLOAT_WITHIN(1.0e-12f, 2.2873113e-7f * 10.0f, noise.gyro_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-9f, 2.4504214e-5f * 15.0f, noise.accel_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-15f, 0.0002e-9f, noise.gyro_bias_rw_var);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-15f, 0.002e-9f, noise.accel_bias_rw_var);
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
      .lat_deg = 45.0f,
      .lon_deg = -79.0f,
      .height_m = 120.0f,
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
  TEST_ASSERT_TRUE(upd.sensor_fusion_state);
  TEST_ASSERT_TRUE(upd.sensor_fusion_state_changed);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, impl->eskf.nominal.pn);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.0f, impl->eskf.nominal.pe);

  upd = sf_fusion_process_imu(&fusion, &imu0);
  TEST_ASSERT_TRUE(upd.sensor_fusion_state);
  upd = sf_fusion_process_imu(&fusion, &imu1);
  TEST_ASSERT_TRUE(upd.sensor_fusion_state);
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
  sf_gnss_ned_sample_t gps = {
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

static void test_eskf_fuse_body_speed_x_moves_forward_velocity_toward_measurement(void) {
  sf_eskf_t eskf;
  float vx_before;

  sf_eskf_init(&eskf, NULL, NULL);
  eskf.nominal.vn = 2.5f;
  eskf.nominal.ve = 0.0f;
  eskf.nominal.vd = 0.0f;
  vx_before = eskf.nominal.vn;

  sf_eskf_fuse_body_speed_x(&eskf, 4.0f, 0.04f);

  TEST_ASSERT_TRUE(eskf.nominal.vn > vx_before);
  TEST_ASSERT_TRUE(eskf.nominal.vn < 4.0f);
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
  const double pos_meas[3] = {6378147.0, 2.0, -3.0};
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
  before_err = fabsf((float)pos_meas[0] - loose.nominal.pn);

  sf_loose_fuse_gps_reference(&loose, pos_meas, 0.5f, 1.0f);

  after_err = fabsf((float)pos_meas[0] - loose.nominal.pn);
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
      .lat_deg = 45.0f,
      .lon_deg = -79.0f,
      .height_m = 120.0f,
      .vel_ned_mps = {0.0f, 0.0f, 0.0f},
      .pos_std_m = {0.5f, 0.5f, 1.0f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
      .heading_valid = true,
      .heading_rad = 0.0f,
  };
  sf_gnss_sample_t gnss1 = {
      .t_s = 1.0,
      .lat_deg = 45.0f,
      .lon_deg = -79.0f,
      .height_m = 120.0f,
      .vel_ned_mps = {0.0f, 0.0f, 0.0f},
      .pos_std_m = {0.5f, 0.5f, 1.0f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
      .heading_valid = true,
      .heading_rad = 0.0f,
  };
  sf_gnss_sample_t gnss2 = {
      .t_s = 2.0,
      .lat_deg = 45.0f,
      .lon_deg = -79.0f,
      .height_m = 120.0f,
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
  TEST_ASSERT_TRUE(impl->mount_q_vb_valid);
  TEST_ASSERT_TRUE(impl->mount_q_vb_valid);
  upd = sf_fusion_process_gnss(&fusion, &gnss2);
  TEST_ASSERT_TRUE(impl->mount_q_vb_valid);
}

static void test_sensor_fusion_vehicle_speed_updates_forward_velocity(void) {
  sf_sensor_fusion_t fusion;
  sf_sensor_fusion_impl_t *impl = sf_impl(&fusion);
  sf_fusion_config_t cfg;
  sf_gnss_sample_t gnss = {
      .t_s = 1.0f,
      .lat_deg = 43.500000f,
      .lon_deg = -79.600000f,
      .height_m = 100.0f,
      .vel_ned_mps = {2.0f, 0.0f, 0.0f},
      .pos_std_m = {0.5f, 0.5f, 1.0f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
      .heading_valid = true,
      .heading_rad = 0.0f,
  };
  sf_vehicle_speed_sample_t speed = {
      .t_s = 1.1f,
      .speed_mps = 4.0f,
      .direction = SF_VEHICLE_SPEED_DIRECTION_FORWARD,
  };
  float vn_before;

  sf_fusion_config_default(&cfg);
  sf_fusion_init_external(&fusion, &cfg, (float[4]){1.0f, 0.0f, 0.0f, 0.0f});
  (void)sf_fusion_process_gnss(&fusion, &gnss);
  vn_before = impl->eskf.nominal.vn;
  (void)sf_fusion_process_vehicle_speed(&fusion, &speed);

  TEST_ASSERT_TRUE(impl->ekf_initialized);
  TEST_ASSERT_TRUE(impl->eskf.nominal.vn > vn_before);
  TEST_ASSERT_TRUE(impl->eskf.nominal.vn < speed.speed_mps);
}

static void test_sensor_fusion_vehicle_speed_reverse_updates_negative_velocity(void) {
  sf_sensor_fusion_t fusion;
  sf_sensor_fusion_impl_t *impl = sf_impl(&fusion);
  sf_fusion_config_t cfg;
  sf_gnss_sample_t gnss = {
      .t_s = 1.0f,
      .lat_deg = 43.500000f,
      .lon_deg = -79.600000f,
      .height_m = 100.0f,
      .vel_ned_mps = {-2.0f, 0.0f, 0.0f},
      .pos_std_m = {0.5f, 0.5f, 1.0f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
      .heading_valid = true,
      .heading_rad = 0.0f,
  };
  sf_vehicle_speed_sample_t speed = {
      .t_s = 1.1f,
      .speed_mps = 4.0f,
      .direction = SF_VEHICLE_SPEED_DIRECTION_REVERSE,
  };
  float vn_before;

  sf_fusion_config_default(&cfg);
  sf_fusion_init_external(&fusion, &cfg, (float[4]){1.0f, 0.0f, 0.0f, 0.0f});
  (void)sf_fusion_process_gnss(&fusion, &gnss);
  vn_before = impl->eskf.nominal.vn;
  (void)sf_fusion_process_vehicle_speed(&fusion, &speed);

  TEST_ASSERT_TRUE(impl->eskf.nominal.vn < vn_before);
  TEST_ASSERT_TRUE(impl->eskf.nominal.vn > -speed.speed_mps);
}

static void test_sensor_fusion_vehicle_speed_unknown_direction_uses_predicted_sign(void) {
  sf_sensor_fusion_t fusion;
  sf_sensor_fusion_impl_t *impl = sf_impl(&fusion);
  sf_fusion_config_t cfg;
  sf_gnss_sample_t gnss = {
      .t_s = 1.0f,
      .lat_deg = 43.500000f,
      .lon_deg = -79.600000f,
      .height_m = 100.0f,
      .vel_ned_mps = {-3.0f, 0.0f, 0.0f},
      .pos_std_m = {0.5f, 0.5f, 1.0f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
      .heading_valid = true,
      .heading_rad = 0.0f,
  };
  sf_vehicle_speed_sample_t speed = {
      .t_s = 1.1f,
      .speed_mps = 4.0f,
      .direction = SF_VEHICLE_SPEED_DIRECTION_UNKNOWN,
  };
  float vn_before;

  sf_fusion_config_default(&cfg);
  sf_fusion_init_external(&fusion, &cfg, (float[4]){1.0f, 0.0f, 0.0f, 0.0f});
  (void)sf_fusion_process_gnss(&fusion, &gnss);
  vn_before = impl->eskf.nominal.vn;
  (void)sf_fusion_process_vehicle_speed(&fusion, &speed);

  TEST_ASSERT_TRUE(impl->eskf.nominal.vn < vn_before);
  TEST_ASSERT_TRUE(impl->eskf.nominal.vn > -speed.speed_mps);
}

static void test_sensor_fusion_get_lla_reports_current_global_position(void) {
  sf_sensor_fusion_t fusion;
  sf_sensor_fusion_impl_t *impl = sf_impl(&fusion);
  sf_fusion_config_t cfg;
  sf_gnss_sample_t gnss = {
      .t_s = 0.0f,
      .lat_deg = 43.500000f,
      .lon_deg = -79.600000f,
      .height_m = 100.0f,
      .vel_ned_mps = {1.0f, 2.0f, -0.1f},
      .pos_std_m = {0.5f, 0.5f, 1.0f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
      .heading_valid = false,
      .heading_rad = 0.0f,
  };
  float lla[3];

  sf_fusion_config_default(&cfg);
  sf_fusion_init_external(&fusion, &cfg, (float[4]){1.0f, 0.0f, 0.0f, 0.0f});
  (void)sf_fusion_process_gnss(&fusion, &gnss);

  impl->eskf.nominal.pn = 120.0f;
  impl->eskf.nominal.pe = -45.0f;
  impl->eskf.nominal.pd = 8.0f;

  TEST_ASSERT_TRUE(sf_get_lla((const sf_t *)&fusion, lla));
  TEST_ASSERT_TRUE(fabsf(lla[0] - gnss.lat_deg) > 1.0e-4f);
  TEST_ASSERT_TRUE(fabsf(lla[1] - gnss.lon_deg) > 1.0e-4f);
  TEST_ASSERT_FLOAT_WITHIN(20.0f, gnss.height_m - 8.0f, lla[2]);
}

static void test_sensor_fusion_reanchor_debug_counts_and_tracks_anchor(void) {
  sf_sensor_fusion_t fusion;
  sf_sensor_fusion_impl_t *impl = sf_impl(&fusion);
  sf_fusion_config_t cfg;
  sf_gnss_sample_t gnss0 = {
      .t_s = 0.0f,
      .lat_deg = 43.500000f,
      .lon_deg = -79.600000f,
      .height_m = 100.0f,
      .vel_ned_mps = {4.0f, 0.0f, 0.0f},
      .pos_std_m = {0.5f, 0.5f, 1.0f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
      .heading_valid = false,
      .heading_rad = 0.0f,
  };
  sf_gnss_sample_t gnss1;
  sf_gnss_sample_t gnss2;
  sf_fusion_debug_t debug;
  float lla1[3];
  float lla2[3];

  sf_fusion_config_default(&cfg);
  sf_fusion_init_external(&fusion, &cfg, (float[4]){1.0f, 0.0f, 0.0f, 0.0f});
  (void)sf_fusion_process_gnss(&fusion, &gnss0);

  test_ned_offset_to_lla(gnss0.lat_deg, gnss0.lon_deg, gnss0.height_m,
                         (float[3]){6000.0f, 0.0f, 0.0f}, lla1);
  gnss1 = gnss0;
  gnss1.t_s = 10.0f;
  gnss1.lat_deg = lla1[0];
  gnss1.lon_deg = lla1[1];
  gnss1.height_m = lla1[2];
  (void)sf_fusion_process_gnss(&fusion, &gnss1);

  TEST_ASSERT_TRUE(sf_fusion_get_debug(&fusion, &debug));
  TEST_ASSERT_EQUAL_UINT32(1u, debug.reanchor_count);
  TEST_ASSERT_TRUE(debug.last_reanchor_valid);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 10.0f, debug.last_reanchor_t_s);
  TEST_ASSERT_TRUE(debug.last_reanchor_distance_m > 5000.0f);
  TEST_ASSERT_TRUE(debug.anchor_valid);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, gnss1.lat_deg, debug.anchor_lat_deg);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, gnss1.lon_deg, debug.anchor_lon_deg);
  TEST_ASSERT_FLOAT_WITHIN(0.5f, gnss1.height_m, debug.anchor_height_m);
  TEST_ASSERT_TRUE(fabsf(impl->last_gnss.pos_ned_m[0]) < 2.0f);
  TEST_ASSERT_TRUE(fabsf(impl->last_gnss.pos_ned_m[1]) < 2.0f);

  test_ned_offset_to_lla(gnss1.lat_deg, gnss1.lon_deg, gnss1.height_m,
                         (float[3]){6000.0f, 100.0f, 0.0f}, lla2);
  gnss2 = gnss1;
  gnss2.t_s = 20.0f;
  gnss2.lat_deg = lla2[0];
  gnss2.lon_deg = lla2[1];
  gnss2.height_m = lla2[2];
  (void)sf_fusion_process_gnss(&fusion, &gnss2);

  TEST_ASSERT_TRUE(sf_fusion_get_debug(&fusion, &debug));
  TEST_ASSERT_EQUAL_UINT32(2u, debug.reanchor_count);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-5f, 20.0f, debug.last_reanchor_t_s);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, gnss2.lat_deg, debug.anchor_lat_deg);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, gnss2.lon_deg, debug.anchor_lon_deg);
}

static void test_sensor_fusion_reanchor_preserves_global_state_continuity(void) {
  sf_sensor_fusion_t fusion;
  sf_sensor_fusion_impl_t *impl = sf_impl(&fusion);
  sf_fusion_config_t cfg;
  sf_gnss_sample_t gnss0 = {
      .t_s = 0.0f,
      .lat_deg = 43.500000f,
      .lon_deg = -79.600000f,
      .height_m = 100.0f,
      .vel_ned_mps = {3.0f, -1.0f, 0.2f},
      .pos_std_m = {0.5f, 0.5f, 1.0f},
      .vel_std_mps = {0.2f, 0.2f, 0.2f},
      .heading_valid = false,
      .heading_rad = 0.0f,
  };
  sf_gnss_sample_t gnss_far;
  sf_fusion_debug_t debug;
  float lla_before[3];
  float lla_after[3];
  float q_before[4];
  float q_after[4];
  float c_bn_before[3][3];
  float c_bn_after[3][3];
  float c_ne_before[3][3];
  float c_en_before[3][3];
  float c_en_after[3][3];
  float c_be_before[3][3];
  float c_be_after[3][3];
  float vel_ecef_before[3];
  float vel_ecef_after[3];
  float far_lla[3];

  sf_fusion_config_default(&cfg);
  sf_fusion_init_external(&fusion, &cfg, (float[4]){1.0f, 0.0f, 0.0f, 0.0f});
  (void)sf_fusion_process_gnss(&fusion, &gnss0);

  impl->eskf.nominal.pn = 1250.0f;
  impl->eskf.nominal.pe = -340.0f;
  impl->eskf.nominal.pd = 12.0f;
  impl->eskf.nominal.vn = 7.5f;
  impl->eskf.nominal.ve = -2.0f;
  impl->eskf.nominal.vd = 0.3f;
  test_quat_from_rpy(-2.0f * 3.14159265358979323846f / 180.0f,
                     5.0f * 3.14159265358979323846f / 180.0f,
                     20.0f * 3.14159265358979323846f / 180.0f, q_before);
  impl->eskf.nominal.q0 = q_before[0];
  impl->eskf.nominal.q1 = q_before[1];
  impl->eskf.nominal.q2 = q_before[2];
  impl->eskf.nominal.q3 = q_before[3];

  TEST_ASSERT_TRUE(sf_get_lla((const sf_t *)&fusion, lla_before));
  test_quat_to_rotmat(q_before, c_bn_before);
  test_ecef_to_ned_matrix(impl->anchor.lat_deg, impl->anchor.lon_deg, c_ne_before);
  test_transpose3(c_ne_before, c_en_before);
  test_mat3_vec(c_en_before,
                (float[3]){impl->eskf.nominal.vn, impl->eskf.nominal.ve,
                           impl->eskf.nominal.vd},
                vel_ecef_before);
  test_mat3_mul(c_en_before, c_bn_before, c_be_before);

  test_ned_offset_to_lla(gnss0.lat_deg, gnss0.lon_deg, gnss0.height_m,
                         (float[3]){6000.0f, 0.0f, 0.0f}, far_lla);
  gnss_far = gnss0;
  gnss_far.t_s = 10.0f;
  gnss_far.lat_deg = far_lla[0];
  gnss_far.lon_deg = far_lla[1];
  gnss_far.height_m = far_lla[2];
  gnss_far.pos_std_m[0] = 1.0e6f;
  gnss_far.pos_std_m[1] = 1.0e6f;
  gnss_far.pos_std_m[2] = 1.0e6f;
  gnss_far.vel_std_mps[0] = 1.0e6f;
  gnss_far.vel_std_mps[1] = 1.0e6f;
  gnss_far.vel_std_mps[2] = 1.0e6f;
  gnss_far.heading_valid = false;
  (void)sf_fusion_process_gnss(&fusion, &gnss_far);

  TEST_ASSERT_TRUE(sf_fusion_get_debug(&fusion, &debug));
  TEST_ASSERT_EQUAL_UINT32(1u, debug.reanchor_count);
  TEST_ASSERT_TRUE(sf_get_lla((const sf_t *)&fusion, lla_after));
  TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, lla_before[0], lla_after[0]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, lla_before[1], lla_after[1]);
  TEST_ASSERT_FLOAT_WITHIN(0.5f, lla_before[2], lla_after[2]);

  q_after[0] = impl->eskf.nominal.q0;
  q_after[1] = impl->eskf.nominal.q1;
  q_after[2] = impl->eskf.nominal.q2;
  q_after[3] = impl->eskf.nominal.q3;
  test_quat_to_rotmat(q_after, c_bn_after);
  test_transpose3(impl->anchor.c_ne, c_en_after);
  test_mat3_vec(c_en_after,
                (float[3]){impl->eskf.nominal.vn, impl->eskf.nominal.ve,
                           impl->eskf.nominal.vd},
                vel_ecef_after);
  test_mat3_mul(c_en_after, c_bn_after, c_be_after);

  TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, vel_ecef_before[0], vel_ecef_after[0]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, vel_ecef_before[1], vel_ecef_after[1]);
  TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, vel_ecef_before[2], vel_ecef_after[2]);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, c_be_before[i][j], c_be_after[i][j]);
    }
  }
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
  RUN_TEST(test_eskf_fuse_body_speed_x_moves_forward_velocity_toward_measurement);
  RUN_TEST(test_loose_init_copies_all_noise_fields_and_covariance_diag);
  RUN_TEST(test_loose_error_transition_has_expected_reference_structure);
  RUN_TEST(test_loose_predict_with_zero_dt_is_noop);
  RUN_TEST(test_loose_nominal_predict_keeps_equatorial_rest_state_near_rest);
  RUN_TEST(test_loose_reference_gps_update_moves_ecef_position_toward_measurement);
  RUN_TEST(test_loose_reference_nhc_reduces_lateral_and_vertical_car_velocity);
  RUN_TEST(test_sensor_fusion_internal_mode_bootstraps_align_state);
  RUN_TEST(test_sensor_fusion_vehicle_speed_updates_forward_velocity);
  RUN_TEST(test_sensor_fusion_vehicle_speed_reverse_updates_negative_velocity);
  RUN_TEST(test_sensor_fusion_vehicle_speed_unknown_direction_uses_predicted_sign);
  RUN_TEST(test_sensor_fusion_get_lla_reports_current_global_position);
  RUN_TEST(test_sensor_fusion_reanchor_debug_counts_and_tracks_anchor);
  RUN_TEST(test_sensor_fusion_reanchor_preserves_global_state_continuity);
  return UNITY_END();
}
