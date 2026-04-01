#include "sensor_fusion_internal.h"

#include <math.h>

static float sf_wrap_angle_rad(float x);
static void sf_rot_to_euler_zyx(const float c[3][3], float rpy[3]);
static void sf_vec3_add_inplace(float a[3], const float b[3]);
static void sf_vec3_scale(const float v[3], float s, float out[3]);
static void sf_vec3_sub(const float a[3], const float b[3], float out[3]);
static float sf_vec3_dot(const float a[3], const float b[3]);
static void sf_vec3_cross(const float a[3], const float b[3], float out[3]);
static float sf_vec3_norm(const float v[3]);
static bool sf_vec3_normalize(const float v[3], float out[3]);
static void sf_mat3_mul(const float a[3][3], const float b[3][3], float out[3][3]);

bool sf_bootstrap_vehicle_to_body_from_stationary(
    const float (*accel_samples_b)[3],
    uint32_t sample_count,
    float yaw_seed_rad,
    sf_stationary_mount_bootstrap_t *out) {
  float f_mean_b[3] = {0.0f, 0.0f, 0.0f};
  float z_v_in_b[3];
  float x_ref[3] = {1.0f, 0.0f, 0.0f};
  float x_v_in_b[3];
  float y_v_in_b[3];
  float c_b_v_tilt[3][3];
  float rpy[3];
  float dyaw;
  float s;
  float c;
  float c_delta[3][3];

  if (accel_samples_b == NULL || out == NULL || sample_count == 0U) {
    return false;
  }

  for (uint32_t i = 0; i < sample_count; ++i) {
    sf_vec3_add_inplace(f_mean_b, accel_samples_b[i]);
  }
  sf_vec3_scale(f_mean_b, 1.0f / (float)sample_count, f_mean_b);

  if (sf_vec3_norm(f_mean_b) < 1.0e-6f) {
    return false;
  }

  sf_vec3_scale(f_mean_b, -1.0f / sf_vec3_norm(f_mean_b), z_v_in_b);
  {
    float proj[3];
    sf_vec3_scale(z_v_in_b, sf_vec3_dot(z_v_in_b, x_ref), proj);
    sf_vec3_sub(x_ref, proj, x_v_in_b);
  }
  if (sf_vec3_norm(x_v_in_b) < 1.0e-6f) {
    if (fabsf(x_ref[0]) > fabsf(x_ref[1])) {
      x_ref[0] = 0.0f;
      x_ref[1] = 1.0f;
      x_ref[2] = 0.0f;
    } else {
      x_ref[0] = 1.0f;
      x_ref[1] = 0.0f;
      x_ref[2] = 0.0f;
    }
    {
      float proj[3];
      sf_vec3_scale(z_v_in_b, sf_vec3_dot(z_v_in_b, x_ref), proj);
      sf_vec3_sub(x_ref, proj, x_v_in_b);
    }
  }
  if (!sf_vec3_normalize(x_v_in_b, x_v_in_b)) {
    return false;
  }
  sf_vec3_cross(z_v_in_b, x_v_in_b, y_v_in_b);
  if (!sf_vec3_normalize(y_v_in_b, y_v_in_b)) {
    return false;
  }
  sf_vec3_cross(y_v_in_b, z_v_in_b, x_v_in_b);
  if (!sf_vec3_normalize(x_v_in_b, x_v_in_b)) {
    return false;
  }

  c_b_v_tilt[0][0] = x_v_in_b[0];
  c_b_v_tilt[1][0] = x_v_in_b[1];
  c_b_v_tilt[2][0] = x_v_in_b[2];
  c_b_v_tilt[0][1] = y_v_in_b[0];
  c_b_v_tilt[1][1] = y_v_in_b[1];
  c_b_v_tilt[2][1] = y_v_in_b[2];
  c_b_v_tilt[0][2] = z_v_in_b[0];
  c_b_v_tilt[1][2] = z_v_in_b[1];
  c_b_v_tilt[2][2] = z_v_in_b[2];

  sf_rot_to_euler_zyx(c_b_v_tilt, rpy);
  dyaw = sf_wrap_angle_rad(yaw_seed_rad - rpy[2]);
  s = sinf(dyaw);
  c = cosf(dyaw);
  c_delta[0][0] = c;
  c_delta[0][1] = -s;
  c_delta[0][2] = 0.0f;
  c_delta[1][0] = s;
  c_delta[1][1] = c;
  c_delta[1][2] = 0.0f;
  c_delta[2][0] = 0.0f;
  c_delta[2][1] = 0.0f;
  c_delta[2][2] = 1.0f;

  out->mean_accel_b[0] = f_mean_b[0];
  out->mean_accel_b[1] = f_mean_b[1];
  out->mean_accel_b[2] = f_mean_b[2];
  sf_mat3_mul(c_b_v_tilt, c_delta, out->c_b_v);
  return true;
}

static float sf_wrap_angle_rad(float x) {
  return atan2f(sinf(x), cosf(x));
}

static void sf_rot_to_euler_zyx(const float c[3][3], float rpy[3]) {
  float pitch = asinf(-c[2][0]);
  float roll = atan2f(c[2][1], c[2][2]);
  float yaw = sf_wrap_angle_rad(atan2f(c[1][0], c[0][0]));
  rpy[0] = roll;
  rpy[1] = pitch;
  rpy[2] = yaw;
}

static void sf_vec3_add_inplace(float a[3], const float b[3]) {
  a[0] += b[0];
  a[1] += b[1];
  a[2] += b[2];
}

static void sf_vec3_scale(const float v[3], float s, float out[3]) {
  out[0] = v[0] * s;
  out[1] = v[1] * s;
  out[2] = v[2] * s;
}

static void sf_vec3_sub(const float a[3], const float b[3], float out[3]) {
  out[0] = a[0] - b[0];
  out[1] = a[1] - b[1];
  out[2] = a[2] - b[2];
}

static float sf_vec3_dot(const float a[3], const float b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void sf_vec3_cross(const float a[3], const float b[3], float out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

static float sf_vec3_norm(const float v[3]) {
  return sqrtf(sf_vec3_dot(v, v));
}

static bool sf_vec3_normalize(const float v[3], float out[3]) {
  float n = sf_vec3_norm(v);
  if (n < 1.0e-9f) {
    return false;
  }
  sf_vec3_scale(v, 1.0f / n, out);
  return true;
}

static void sf_mat3_mul(const float a[3][3], const float b[3][3], float out[3][3]) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
}
