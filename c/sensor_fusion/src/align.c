#include "align.h"

#include <math.h>
#include <string.h>

const float SF_ALIGN_GRAVITY_MPS2 = 9.80665f;

#define SF_ALIGN_PI 3.14159265358979323846f
#define SF_ALIGN_COARSE_READY_ROLL_DEG 5.0f
#define SF_ALIGN_COARSE_READY_PITCH_DEG 5.0f
#define SF_ALIGN_COARSE_READY_YAW_DEG 8.0f
#define SF_ALIGN_STATIONARY_INIT_ROLL_DEG 10.0f
#define SF_ALIGN_STATIONARY_INIT_PITCH_DEG 10.0f
#define SF_ALIGN_STATIONARY_INIT_YAW_DEG 60.0f
#define SF_ALIGN_TILT_PROGRESS_WEIGHT 0.30f
#define SF_ALIGN_YAW_PROGRESS_WEIGHT 0.70f
#define SF_ALIGN_TILT_JAC_EPS_RAD 1.0e-4f

static float sf_align_deg_to_rad(float deg)
{
    return deg * (SF_ALIGN_PI / 180.0f);
}

static float sf_align_rad_to_deg(float rad)
{
    return rad * (180.0f / SF_ALIGN_PI);
}

static float sf_align_sq(float x)
{
    return x * x;
}

static float sf_align_clamp(float x, float lo, float hi)
{
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

static float sf_align_max(float a, float b)
{
    return a > b ? a : b;
}

static float sf_align_vec3_dot(const float a[3], const float b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static float sf_align_vec3_norm(const float v[3])
{
    return sqrtf(sf_align_vec3_dot(v, v));
}

static float sf_align_norm2(const float v[2])
{
    return sqrtf(v[0] * v[0] + v[1] * v[1]);
}

static bool sf_align_vec2_normalize(const float v[2], float out[2])
{
    const float n = sf_align_norm2(v);
    if (!isfinite(n) || n <= 1.0e-8f) {
        return false;
    }
    out[0] = v[0] / n;
    out[1] = v[1] / n;
    return true;
}

static bool sf_align_vec3_normalize(const float v[3], float out[3])
{
    const float n = sf_align_vec3_norm(v);
    if (!isfinite(n) || n <= 1.0e-6f) {
        return false;
    }
    out[0] = v[0] / n;
    out[1] = v[1] / n;
    out[2] = v[2] / n;
    return true;
}

static void sf_align_vec3_cross(const float a[3], const float b[3], float out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static void sf_align_diag3(float p[3][3], const float diag[3])
{
    memset(p, 0, sizeof(float) * 9u);
    p[0][0] = diag[0];
    p[1][1] = diag[1];
    p[2][2] = diag[2];
}

static void sf_align_quat_normalize(float q[4])
{
    const float n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if (!isfinite(n2) || n2 <= 1.0e-12f) {
        q[0] = 1.0f;
        q[1] = q[2] = q[3] = 0.0f;
        return;
    }
    const float inv = 1.0f / sqrtf(n2);
    q[0] *= inv;
    q[1] *= inv;
    q[2] *= inv;
    q[3] *= inv;
}

static void sf_align_quat_to_rotmat(const float q_in[4], float r[3][3])
{
    float q[4] = {q_in[0], q_in[1], q_in[2], q_in[3]};
    sf_align_quat_normalize(q);
    const float q0 = q[0];
    const float q1 = q[1];
    const float q2 = q[2];
    const float q3 = q[3];

    r[0][0] = 1.0f - 2.0f * (q2 * q2 + q3 * q3);
    r[0][1] = 2.0f * (q1 * q2 - q0 * q3);
    r[0][2] = 2.0f * (q1 * q3 + q0 * q2);
    r[1][0] = 2.0f * (q1 * q2 + q0 * q3);
    r[1][1] = 1.0f - 2.0f * (q1 * q1 + q3 * q3);
    r[1][2] = 2.0f * (q2 * q3 - q0 * q1);
    r[2][0] = 2.0f * (q1 * q3 - q0 * q2);
    r[2][1] = 2.0f * (q2 * q3 + q0 * q1);
    r[2][2] = 1.0f - 2.0f * (q1 * q1 + q2 * q2);
}

static void sf_align_rotmat_to_quat(const float r[3][3], float q[4])
{
    const float tr = r[0][0] + r[1][1] + r[2][2];
    if (tr > 0.0f) {
        const float s = sqrtf(tr + 1.0f) * 2.0f;
        q[0] = 0.25f * s;
        q[1] = (r[2][1] - r[1][2]) / s;
        q[2] = (r[0][2] - r[2][0]) / s;
        q[3] = (r[1][0] - r[0][1]) / s;
    } else if (r[0][0] > r[1][1] && r[0][0] > r[2][2]) {
        const float s = sqrtf(1.0f + r[0][0] - r[1][1] - r[2][2]) * 2.0f;
        q[0] = (r[2][1] - r[1][2]) / s;
        q[1] = 0.25f * s;
        q[2] = (r[0][1] + r[1][0]) / s;
        q[3] = (r[0][2] + r[2][0]) / s;
    } else if (r[1][1] > r[2][2]) {
        const float s = sqrtf(1.0f + r[1][1] - r[0][0] - r[2][2]) * 2.0f;
        q[0] = (r[0][2] - r[2][0]) / s;
        q[1] = (r[0][1] + r[1][0]) / s;
        q[2] = 0.25f * s;
        q[3] = (r[1][2] + r[2][1]) / s;
    } else {
        const float s = sqrtf(1.0f + r[2][2] - r[0][0] - r[1][1]) * 2.0f;
        q[0] = (r[1][0] - r[0][1]) / s;
        q[1] = (r[0][2] + r[2][0]) / s;
        q[2] = (r[1][2] + r[2][1]) / s;
        q[3] = 0.25f * s;
    }
    sf_align_quat_normalize(q);
}

static void sf_align_quat_mul(const float a[4], const float b[4], float out[4])
{
    out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

static void sf_align_quat_from_small_angle(const float dtheta[3], float q[4])
{
    q[0] = 1.0f;
    q[1] = 0.5f * dtheta[0];
    q[2] = 0.5f * dtheta[1];
    q[3] = 0.5f * dtheta[2];
    sf_align_quat_normalize(q);
}

static void sf_align_inject_small_angle(float q_bv[4], const float dtheta[3])
{
    float dq[4];
    float out[4];
    sf_align_quat_from_small_angle(dtheta, dq);
    sf_align_quat_mul(dq, q_bv, out);
    memcpy(q_bv, out, sizeof(out));
    sf_align_quat_normalize(q_bv);
}

static void sf_align_inject_vehicle_yaw(float q_bv[4], float dpsi)
{
    const float half = 0.5f * dpsi;
    const float q_yaw[4] = {cosf(half), 0.0f, 0.0f, sinf(half)};
    float out[4];
    sf_align_quat_mul(q_bv, q_yaw, out);
    memcpy(q_bv, out, sizeof(out));
    sf_align_quat_normalize(q_bv);
}

static void sf_align_mat3_vec(const float a[3][3], const float x[3], float out[3])
{
    out[0] = a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2];
    out[1] = a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2];
    out[2] = a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2];
}

static void sf_align_mat3_mul(const float a[3][3], const float b[3][3], float out[3][3])
{
    for (size_t i = 0u; i < 3u; ++i) {
        for (size_t j = 0u; j < 3u; ++j) {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
}

static void sf_align_transpose3(const float a[3][3], float out[3][3])
{
    for (size_t i = 0u; i < 3u; ++i) {
        for (size_t j = 0u; j < 3u; ++j) {
            out[i][j] = a[j][i];
        }
    }
}

static void sf_align_skew3(const float v[3], float out[3][3])
{
    out[0][0] = 0.0f;
    out[0][1] = -v[2];
    out[0][2] = v[1];
    out[1][0] = v[2];
    out[1][1] = 0.0f;
    out[1][2] = -v[0];
    out[2][0] = -v[1];
    out[2][1] = v[0];
    out[2][2] = 0.0f;
}

static void sf_align_symmetrize3(float a[3][3])
{
    for (size_t i = 0u; i < 3u; ++i) {
        for (size_t j = i + 1u; j < 3u; ++j) {
            const float avg = 0.5f * (a[i][j] + a[j][i]);
            a[i][j] = avg;
            a[j][i] = avg;
        }
    }
}

static float sf_align_wrap_pi(float rad)
{
    const float two_pi = 2.0f * SF_ALIGN_PI;
    float y = fmodf(rad + SF_ALIGN_PI, two_pi);
    if (y < 0.0f) {
        y += two_pi;
    }
    return y - SF_ALIGN_PI;
}

static bool sf_align_stationary_tilt_rotmat(const float accel_b[3], float c_bv[3][3])
{
    float z_v_in_b[3] = {-accel_b[0], -accel_b[1], -accel_b[2]};
    if (!sf_align_vec3_normalize(z_v_in_b, z_v_in_b)) {
        return false;
    }

    float x_ref[3] = {1.0f, 0.0f, 0.0f};
    float x_v_in_b[3];
    float dot = sf_align_vec3_dot(z_v_in_b, x_ref);
    x_v_in_b[0] = x_ref[0] - z_v_in_b[0] * dot;
    x_v_in_b[1] = x_ref[1] - z_v_in_b[1] * dot;
    x_v_in_b[2] = x_ref[2] - z_v_in_b[2] * dot;
    if (sf_align_vec3_norm(x_v_in_b) < 1.0e-6f) {
        x_ref[0] = 0.0f;
        x_ref[1] = 1.0f;
        x_ref[2] = 0.0f;
        dot = sf_align_vec3_dot(z_v_in_b, x_ref);
        x_v_in_b[0] = x_ref[0] - z_v_in_b[0] * dot;
        x_v_in_b[1] = x_ref[1] - z_v_in_b[1] * dot;
        x_v_in_b[2] = x_ref[2] - z_v_in_b[2] * dot;
    }
    if (!sf_align_vec3_normalize(x_v_in_b, x_v_in_b)) {
        return false;
    }

    float y_v_in_b[3];
    sf_align_vec3_cross(z_v_in_b, x_v_in_b, y_v_in_b);
    if (!sf_align_vec3_normalize(y_v_in_b, y_v_in_b)) {
        return false;
    }
    sf_align_vec3_cross(y_v_in_b, z_v_in_b, x_v_in_b);
    if (!sf_align_vec3_normalize(x_v_in_b, x_v_in_b)) {
        return false;
    }

    for (int row = 0; row < 3; ++row) {
        c_bv[row][0] = x_v_in_b[row];
        c_bv[row][1] = y_v_in_b[row];
        c_bv[row][2] = z_v_in_b[row];
    }
    return true;
}

static void sf_align_remove_gravity_axis(const float q_bv[4], const float accel_b[3], float out[3])
{
    float c_bv[3][3];
    sf_align_quat_to_rotmat(q_bv, c_bv);
    const float g_hat_b[3] = {-c_bv[0][2], -c_bv[1][2], -c_bv[2][2]};
    const float proj_scale = sf_align_vec3_dot(accel_b, g_hat_b);
    out[0] = accel_b[0] - g_hat_b[0] * proj_scale;
    out[1] = accel_b[1] - g_hat_b[1] * proj_scale;
    out[2] = accel_b[2] - g_hat_b[2] * proj_scale;
}

static void sf_align_obs_accel_v(const float q_bv[4], const float accel_b[3], float accel_v[3])
{
    float c_bv[3][3];
    sf_align_quat_to_rotmat(q_bv, c_bv);
    for (int i = 0; i < 3; ++i) {
        accel_v[i] = c_bv[0][i] * accel_b[0] + c_bv[1][i] * accel_b[1] +
                     c_bv[2][i] * accel_b[2];
    }
}

static void sf_align_obs(const float q_bv[4],
                         const float gyro_b[3],
                         const float accel_b[3],
                         float obs[6])
{
    float c_bv[3][3];
    float c_vb[3][3];
    sf_align_quat_to_rotmat(q_bv, c_bv);
    sf_align_transpose3(c_bv, c_vb);
    sf_align_mat3_vec(c_vb, gyro_b, &obs[0]);
    sf_align_mat3_vec(c_vb, accel_b, &obs[3]);
}

static void sf_align_obs_jacobian(const float q_bv[4],
                                  const float gyro_b[3],
                                  const float accel_b[3],
                                  float h[6][3])
{
    float c_bv[3][3];
    float c_vb[3][3];
    float skew_gyro[3][3];
    float skew_accel[3][3];
    float h_gyro[3][3];
    float h_accel[3][3];
    sf_align_quat_to_rotmat(q_bv, c_bv);
    sf_align_transpose3(c_bv, c_vb);
    sf_align_skew3(gyro_b, skew_gyro);
    sf_align_skew3(accel_b, skew_accel);
    sf_align_mat3_mul(c_vb, skew_gyro, h_gyro);
    sf_align_mat3_mul(c_vb, skew_accel, h_accel);
    memcpy(&h[0][0], &h_gyro[0][0], sizeof(h_gyro));
    memcpy(&h[3][0], &h_accel[0][0], sizeof(h_accel));
}

static float sf_align_horizontal_accel_angle_error(const float q_bv[4],
                                                  const float accel_b[3],
                                                  const float gnss_xy[2],
                                                  float imu_xy[2])
{
    float horiz_accel_b[3];
    float accel_v[3];
    sf_align_remove_gravity_axis(q_bv, accel_b, horiz_accel_b);
    sf_align_obs_accel_v(q_bv, horiz_accel_b, accel_v);
    imu_xy[0] = accel_v[0];
    imu_xy[1] = accel_v[1];
    const float cross = imu_xy[0] * gnss_xy[1] - imu_xy[1] * gnss_xy[0];
    const float dot = imu_xy[0] * gnss_xy[0] + imu_xy[1] * gnss_xy[1];
    return atan2f(cross, dot);
}

static void sf_align_horizontal_accel_xy_for_q(const float q_bv[4],
                                               const float accel_b[3],
                                               float out_xy[2])
{
    float horiz_accel_b[3];
    float obs[6];
    const float zero_gyro[3] = {0.0f, 0.0f, 0.0f};
    sf_align_remove_gravity_axis(q_bv, accel_b, horiz_accel_b);
    sf_align_obs(q_bv, zero_gyro, horiz_accel_b, obs);
    out_xy[0] = obs[3];
    out_xy[1] = obs[4];
}

static void sf_align_mat2_mul(const float a[2][2], const float b[2][2], float out[2][2])
{
    out[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
    out[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
    out[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
    out[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
}

static void sf_align_transpose2(const float a[2][2], float out[2][2])
{
    out[0][0] = a[0][0];
    out[0][1] = a[1][0];
    out[1][0] = a[0][1];
    out[1][1] = a[1][1];
}

static void sf_align_horizontal_accel_tilt_jacobian(const float q_bv[4],
                                                    const float accel_b[3],
                                                    float j[2][2])
{
    float base[2];
    sf_align_horizontal_accel_xy_for_q(q_bv, accel_b, base);
    for (size_t axis = 0u; axis < 2u; ++axis) {
        float dq[3] = {0.0f, 0.0f, 0.0f};
        float q_pert[4] = {q_bv[0], q_bv[1], q_bv[2], q_bv[3]};
        float pert[2];
        dq[axis] = SF_ALIGN_TILT_JAC_EPS_RAD;
        sf_align_inject_small_angle(q_pert, dq);
        sf_align_horizontal_accel_xy_for_q(q_pert, accel_b, pert);
        j[0][axis] = (pert[0] - base[0]) / SF_ALIGN_TILT_JAC_EPS_RAD;
        j[1][axis] = (pert[1] - base[1]) / SF_ALIGN_TILT_JAC_EPS_RAD;
    }
}

static float sf_align_angle_variance_from_vector(const float v[2], const float cov[2][2])
{
    const float n2 = v[0] * v[0] + v[1] * v[1];
    if (n2 <= 1.0e-9f) {
        return sf_align_sq(SF_ALIGN_PI);
    }
    const float grad[2] = {-v[1] / n2, v[0] / n2};
    const float cg0 = cov[0][0] * grad[0] + cov[0][1] * grad[1];
    const float cg1 = cov[1][0] * grad[0] + cov[1][1] * grad[1];
    return sf_align_max(grad[0] * cg0 + grad[1] * cg1, 0.0f);
}

static void sf_align_projected_gnss_accel_covariance(const float tangent_hat_n[2],
                                                     const float lateral_hat_n[2],
                                                     const float vel_prev_std_mps[3],
                                                     const float vel_curr_std_mps[3],
                                                     float dt_s,
                                                     float out[2][2])
{
    const float dt2_inv = 1.0f / sf_align_max(dt_s * dt_s, 1.0e-9f);
    const float cov_ne[2] = {
        (sf_align_sq(vel_prev_std_mps[0]) + sf_align_sq(vel_curr_std_mps[0])) * dt2_inv,
        (sf_align_sq(vel_prev_std_mps[1]) + sf_align_sq(vel_curr_std_mps[1])) * dt2_inv,
    };
    out[0][0] = tangent_hat_n[0] * tangent_hat_n[0] * cov_ne[0] +
                tangent_hat_n[1] * tangent_hat_n[1] * cov_ne[1];
    out[0][1] = tangent_hat_n[0] * lateral_hat_n[0] * cov_ne[0] +
                tangent_hat_n[1] * lateral_hat_n[1] * cov_ne[1];
    out[1][0] = lateral_hat_n[0] * tangent_hat_n[0] * cov_ne[0] +
                lateral_hat_n[1] * tangent_hat_n[1] * cov_ne[1];
    out[1][1] = lateral_hat_n[0] * lateral_hat_n[0] * cov_ne[0] +
                lateral_hat_n[1] * lateral_hat_n[1] * cov_ne[1];
}

static float sf_align_horizontal_heading_variance(const sf_align_t *align,
                                                  const sf_align_window_summary_t *window,
                                                  const float gnss_xy[2],
                                                  const float gnss_cov_xy[2][2],
                                                  float model_var_rad2)
{
    float imu_xy[2];
    float imu_cov_xy[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    float tilt_j[2][2];
    float p_tilt[2][2];
    float jp[2][2];
    float tilt_j_t[2][2];
    float tilt_cov[2][2];
    const float imu_count = (float)(window->imu_sample_count > 0u ? window->imu_sample_count : 1u);
    const float imu_mean_var = sf_align_sq(align->cfg.r_gravity_std_mps2) / imu_count;
    sf_align_horizontal_accel_xy_for_q(align->q_bv, window->mean_accel_b, imu_xy);
    imu_cov_xy[0][0] = imu_mean_var;
    imu_cov_xy[1][1] = imu_mean_var;
    sf_align_horizontal_accel_tilt_jacobian(align->q_bv, window->mean_accel_b, tilt_j);
    p_tilt[0][0] = align->p[0][0];
    p_tilt[0][1] = align->p[0][1];
    p_tilt[1][0] = align->p[1][0];
    p_tilt[1][1] = align->p[1][1];
    sf_align_mat2_mul(tilt_j, p_tilt, jp);
    sf_align_transpose2(tilt_j, tilt_j_t);
    sf_align_mat2_mul(jp, tilt_j_t, tilt_cov);
    for (size_t i = 0u; i < 2u; ++i) {
        for (size_t j = 0u; j < 2u; ++j) {
            imu_cov_xy[i][j] += tilt_cov[i][j];
        }
    }
    return sf_align_max(model_var_rad2, 0.0f) +
           sf_align_angle_variance_from_vector(imu_xy, imu_cov_xy) +
           sf_align_angle_variance_from_vector(gnss_xy, gnss_cov_xy);
}

static float sf_align_axis_progress(float sigma_deg, float initial_sigma_deg, float ready_sigma_deg)
{
    if (!isfinite(sigma_deg)) {
        return 0.0f;
    }
    if (initial_sigma_deg <= ready_sigma_deg) {
        return sigma_deg <= ready_sigma_deg ? 1.0f : 0.0f;
    }
    if (sigma_deg <= ready_sigma_deg) {
        return 1.0f;
    }
    if (sigma_deg >= initial_sigma_deg) {
        return 0.0f;
    }
    return (initial_sigma_deg - sigma_deg) / (initial_sigma_deg - ready_sigma_deg);
}

static bool sf_align_compute_coarse_ready(const sf_align_t *align)
{
    float sigma_deg[3];
    sf_align_sigma_deg(align, sigma_deg);
    return align->yaw_observed && sigma_deg[0] <= SF_ALIGN_COARSE_READY_ROLL_DEG &&
           sigma_deg[1] <= SF_ALIGN_COARSE_READY_PITCH_DEG &&
           sigma_deg[2] <= SF_ALIGN_COARSE_READY_YAW_DEG;
}

static bool sf_align_refinement_active(const sf_align_t *align)
{
    return align->cfg.refine_after_coarse_ready && align->coarse_aligned;
}

static float sf_align_refinement_process_noise_scale(const sf_align_t *align)
{
    return sf_align_refinement_active(align)
               ? sf_align_max(align->cfg.refine_process_noise_scale, 0.0f)
               : 1.0f;
}

static float sf_align_refinement_observation_std_scale(const sf_align_t *align)
{
    return sf_align_refinement_active(align)
               ? sf_align_max(align->cfg.refine_observation_std_scale, 1.0e-3f)
               : 1.0f;
}

static void sf_align_turn_consistency_reset(sf_align_t *align)
{
    align->turn_count = 0u;
}

static bool sf_align_turn_consistency_update(sf_align_t *align,
                                             bool turn_valid,
                                             float speed_mps,
                                             float course_rate_radps,
                                             float a_lat_mps2)
{
    if (!turn_valid) {
        sf_align_turn_consistency_reset(align);
        return false;
    }

    sf_align_turn_consistency_sample_t sample;
    sample.speed_mps = speed_mps;
    sample.course_rate_radps = course_rate_radps;
    sample.a_lat_mps2 = a_lat_mps2;
    if (align->turn_count < SF_ALIGN_TURN_CONSISTENCY_CAPACITY) {
        align->turn_samples[align->turn_count++] = sample;
    } else {
        memmove(&align->turn_samples[0],
                &align->turn_samples[1],
                sizeof(align->turn_samples[0]) *
                    (SF_ALIGN_TURN_CONSISTENCY_CAPACITY - 1u));
        align->turn_samples[SF_ALIGN_TURN_CONSISTENCY_CAPACITY - 1u] = sample;
    }

    const size_t min_windows =
        align->cfg.turn_consistency_min_windows > 0u ? align->cfg.turn_consistency_min_windows : 1u;
    if (align->turn_count < min_windows) {
        return false;
    }

    size_t sign_ok = 0u;
    size_t model_ok = 0u;
    for (size_t i = 0; i < align->turn_count; ++i) {
        const sf_align_turn_consistency_sample_t *s = &align->turn_samples[i];
        const float a_lat_pred = s->speed_mps * s->course_rate_radps;
        const float tol =
            sf_align_max(align->cfg.turn_consistency_max_abs_lat_err_mps2,
                         align->cfg.turn_consistency_max_rel_lat_err *
                             sf_align_max(fabsf(a_lat_pred), fabsf(s->a_lat_mps2)));
        if (a_lat_pred * s->a_lat_mps2 > 0.0f) {
            ++sign_ok;
        }
        if (fabsf(s->a_lat_mps2 - a_lat_pred) <= tol) {
            ++model_ok;
        }
    }

    const float fraction = sf_align_clamp(align->cfg.turn_consistency_min_fraction, 0.0f, 1.0f);
    const size_t min_ok = (size_t)ceilf(fraction * (float)align->turn_count);
    return sign_ok >= min_ok && model_ok >= min_ok;
}

static float sf_align_apply_vehicle_yaw_angle(sf_align_t *align, float angle_err_rad, float r_var)
{
    const float pzz = sf_align_max(align->p[2][2], 0.0f);
    const float s = pzz + sf_align_max(r_var, 1.0e-9f);
    const float k = s > 1.0e-9f ? pzz / s : 0.0f;
    sf_align_inject_vehicle_yaw(align->q_bv, -k * angle_err_rad);
    align->p[2][2] = sf_align_max((1.0f - k) * pzz, 0.0f);
    align->p[0][2] = align->p[2][0] = 0.0f;
    align->p[1][2] = align->p[2][1] = 0.0f;
    return angle_err_rad * angle_err_rad / s;
}

static float sf_align_apply_update1_masked(sf_align_t *align,
                                           float z,
                                           size_t obs_idx,
                                           const float accel_b[3],
                                           const float gyro_b[3],
                                           float r_var,
                                           const bool state_mask[3])
{
    float obs[6];
    float h_all[6][3];
    float h[3];
    float ph[3];
    float k[3];
    float dtheta[3];
    float i_minus_kh[3][3];
    float p_new[3][3];
    sf_align_obs(align->q_bv, gyro_b, accel_b, obs);
    sf_align_obs_jacobian(align->q_bv, gyro_b, accel_b, h_all);
    for (size_t i = 0u; i < 3u; ++i) {
        h[i] = state_mask[i] ? h_all[obs_idx][i] : 0.0f;
    }
    const float y = z - obs[obs_idx];
    sf_align_mat3_vec((const float(*)[3])align->p, h, ph);
    const float s = sf_align_vec3_dot(h, ph) + r_var;
    const float s_inv = fabsf(s) > 1.0e-20f ? 1.0f / s : 1.0f;
    for (size_t i = 0u; i < 3u; ++i) {
        k[i] = ph[i] * s_inv;
        dtheta[i] = k[i] * y;
    }
    sf_align_inject_small_angle(align->q_bv, dtheta);
    for (size_t i = 0u; i < 3u; ++i) {
        for (size_t j = 0u; j < 3u; ++j) {
            i_minus_kh[i][j] = -k[i] * h[j];
        }
        i_minus_kh[i][i] += 1.0f;
    }
    sf_align_mat3_mul(i_minus_kh, (const float(*)[3])align->p, p_new);
    sf_align_symmetrize3(p_new);
    memcpy(align->p, p_new, sizeof(p_new));
    return y * y * s_inv;
}

static float sf_align_apply_update2_scaled_masked(sf_align_t *align,
                                                  const float z[2],
                                                  const size_t obs_idx[2],
                                                  const float accel_b[3],
                                                  const float gyro_b[3],
                                                  const float r_var[2],
                                                  const bool state_mask[3],
                                                  const float state_scale[3])
{
    float obs[6];
    float h_all[6][3];
    float h0[3] = {0.0f, 0.0f, 0.0f};
    float h1[3] = {0.0f, 0.0f, 0.0f};
    float ph0[3];
    float ph1[3];
    float k[3][2];
    float dtheta[3];
    float i_minus_kh[3][3];
    float p_new[3][3];
    sf_align_obs(align->q_bv, gyro_b, accel_b, obs);
    sf_align_obs_jacobian(align->q_bv, gyro_b, accel_b, h_all);
    for (size_t i = 0u; i < 3u; ++i) {
        if (state_mask[i]) {
            h0[i] = state_scale[i] * h_all[obs_idx[0]][i];
            h1[i] = state_scale[i] * h_all[obs_idx[1]][i];
        }
    }
    const float y[2] = {z[0] - obs[obs_idx[0]], z[1] - obs[obs_idx[1]]};
    sf_align_mat3_vec((const float(*)[3])align->p, h0, ph0);
    sf_align_mat3_vec((const float(*)[3])align->p, h1, ph1);
    const float s00 = sf_align_vec3_dot(h0, ph0) + r_var[0];
    const float s01 = sf_align_vec3_dot(h0, ph1);
    const float s10 = sf_align_vec3_dot(h1, ph0);
    const float s11 = sf_align_vec3_dot(h1, ph1) + r_var[1];
    const float det = s00 * s11 - s01 * s10;
    float s_inv00 = 1.0f;
    float s_inv01 = 0.0f;
    float s_inv10 = 0.0f;
    float s_inv11 = 1.0f;
    if (fabsf(det) > 1.0e-20f) {
        const float inv_det = 1.0f / det;
        s_inv00 = s11 * inv_det;
        s_inv01 = -s01 * inv_det;
        s_inv10 = -s10 * inv_det;
        s_inv11 = s00 * inv_det;
    }
    for (size_t i = 0u; i < 3u; ++i) {
        k[i][0] = ph0[i] * s_inv00 + ph1[i] * s_inv10;
        k[i][1] = ph0[i] * s_inv01 + ph1[i] * s_inv11;
        dtheta[i] = k[i][0] * y[0] + k[i][1] * y[1];
    }
    sf_align_inject_small_angle(align->q_bv, dtheta);
    for (size_t i = 0u; i < 3u; ++i) {
        for (size_t j = 0u; j < 3u; ++j) {
            i_minus_kh[i][j] = -(k[i][0] * h0[j] + k[i][1] * h1[j]);
        }
        i_minus_kh[i][i] += 1.0f;
    }
    sf_align_mat3_mul(i_minus_kh, (const float(*)[3])align->p, p_new);
    sf_align_symmetrize3(p_new);
    memcpy(align->p, p_new, sizeof(p_new));
    return y[0] * (s_inv00 * y[0] + s_inv01 * y[1]) +
           y[1] * (s_inv10 * y[0] + s_inv11 * y[1]);
}

sf_align_config_t sf_align_config_default(void)
{
    sf_align_config_t cfg;
    cfg.q_mount_std_rad[0] = sf_align_deg_to_rad(0.0005f);
    cfg.q_mount_std_rad[1] = sf_align_deg_to_rad(0.0005f);
    cfg.q_mount_std_rad[2] = sf_align_deg_to_rad(0.00005f);
    cfg.refine_after_coarse_ready = false;
    cfg.refine_process_noise_scale = 1.0f;
    cfg.refine_observation_std_scale = 1.0f;
    cfg.r_gravity_std_mps2 = 4.0f;
    cfg.r_horiz_yaw_std_rad = sf_align_deg_to_rad(1.0f);
    cfg.r_turn_gyro_std_radps = sf_align_deg_to_rad(2.0f);
    cfg.gravity_lpf_alpha = 0.04f;
    cfg.min_speed_mps = 3.0f / 3.6f;
    cfg.min_turn_rate_radps = sf_align_deg_to_rad(2.0f);
    cfg.min_lat_acc_mps2 = 0.10f;
    cfg.min_long_acc_mps2 = 0.18f;
    cfg.turn_consistency_min_windows = 5u;
    cfg.turn_consistency_min_fraction = 0.8f;
    cfg.turn_consistency_max_abs_lat_err_mps2 = 0.35f;
    cfg.turn_consistency_max_rel_lat_err = 0.6f;
    cfg.max_stationary_gyro_radps = sf_align_deg_to_rad(0.8f);
    cfg.max_stationary_accel_norm_err_mps2 = 0.2f;
    cfg.use_gravity = true;
    cfg.use_turn_gyro = true;
    return cfg;
}

void sf_align_init(sf_align_t *align, sf_align_config_t cfg)
{
    memset(align, 0, sizeof(*align));
    align->q_bv[0] = 1.0f;
    const float diag[3] = {
        sf_align_sq(sf_align_deg_to_rad(20.0f)),
        sf_align_sq(sf_align_deg_to_rad(20.0f)),
        sf_align_sq(sf_align_deg_to_rad(60.0f)),
    };
    sf_align_diag3(align->p, diag);
    align->gravity_lp_b[2] = -SF_ALIGN_GRAVITY_MPS2;
    align->cfg = cfg;
}

bool sf_align_initialize_from_stationary(sf_align_t *align,
                                         const float (*accel_samples_b)[3],
                                         size_t sample_count)
{
    if (sample_count == 0u || accel_samples_b == NULL) {
        return false;
    }
    float mean[3] = {0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < sample_count; ++i) {
        mean[0] += accel_samples_b[i][0];
        mean[1] += accel_samples_b[i][1];
        mean[2] += accel_samples_b[i][2];
    }
    const float inv = 1.0f / (float)sample_count;
    mean[0] *= inv;
    mean[1] *= inv;
    mean[2] *= inv;

    float c_bv[3][3];
    if (!sf_align_stationary_tilt_rotmat(mean, c_bv)) {
        return false;
    }
    sf_align_rotmat_to_quat(c_bv, align->q_bv);
    const float diag[3] = {
        sf_align_sq(sf_align_deg_to_rad(SF_ALIGN_STATIONARY_INIT_ROLL_DEG)),
        sf_align_sq(sf_align_deg_to_rad(SF_ALIGN_STATIONARY_INIT_PITCH_DEG)),
        sf_align_sq(sf_align_deg_to_rad(SF_ALIGN_STATIONARY_INIT_YAW_DEG)),
    };
    sf_align_diag3(align->p, diag);
    memcpy(align->gravity_lp_b, mean, sizeof(mean));
    sf_align_turn_consistency_reset(align);
    align->yaw_observed = false;
    align->coarse_aligned = false;
    return true;
}

void sf_align_predict(sf_align_t *align, float dt)
{
    const float bounded_dt = sf_align_max(dt, 1.0e-3f);
    const float process_scale = sf_align_refinement_process_noise_scale(align);
    for (size_t i = 0u; i < SF_ALIGN_N_STATES; ++i) {
        const float q = align->cfg.q_mount_std_rad[i] * process_scale;
        align->p[i][i] += q * q * bounded_dt;
    }
}

float sf_align_update_window(sf_align_t *align, const sf_align_window_summary_t *window)
{
    return sf_align_update_window_with_trace(align, window, NULL);
}

float sf_align_update_window_with_trace(sf_align_t *align,
                                        const sf_align_window_summary_t *window,
                                        sf_align_update_trace_t *trace)
{
    sf_align_update_trace_t local_trace;
    if (trace == NULL) {
        trace = &local_trace;
    }
    memset(trace, 0, sizeof(*trace));
    memcpy(trace->q_start, align->q_bv, sizeof(trace->q_start));
    trace->refinement_active = sf_align_refinement_active(align);
    trace->refinement_process_noise_scale = sf_align_refinement_process_noise_scale(align);
    trace->refinement_observation_std_scale = sf_align_refinement_observation_std_scale(align);

    sf_align_predict(align, window->dt);

    const float dt = sf_align_max(window->dt, 1.0e-3f);
    const float speed_prev = sf_align_norm2((const float[2]){window->gnss_vel_prev_n[0],
                                                            window->gnss_vel_prev_n[1]});
    const float speed_curr = sf_align_norm2((const float[2]){window->gnss_vel_curr_n[0],
                                                            window->gnss_vel_curr_n[1]});
    const float speed_mid = 0.5f * (speed_prev + speed_curr);
    const float course_prev = atan2f(window->gnss_vel_prev_n[1], window->gnss_vel_prev_n[0]);
    const float course_curr = atan2f(window->gnss_vel_curr_n[1], window->gnss_vel_curr_n[0]);
    const float course_rate = sf_align_wrap_pi(course_curr - course_prev) / dt;
    const float a_n[3] = {
        (window->gnss_vel_curr_n[0] - window->gnss_vel_prev_n[0]) / dt,
        (window->gnss_vel_curr_n[1] - window->gnss_vel_prev_n[1]) / dt,
        (window->gnss_vel_curr_n[2] - window->gnss_vel_prev_n[2]) / dt,
    };

    float a_long = 0.0f;
    float a_lat = 0.0f;
    const float v_mid_h[2] = {0.5f * (window->gnss_vel_prev_n[0] + window->gnss_vel_curr_n[0]),
                              0.5f * (window->gnss_vel_prev_n[1] + window->gnss_vel_curr_n[1])};
    float gnss_accel_cov_v[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    float t_hat[2];
    if (sf_align_vec2_normalize(v_mid_h, t_hat)) {
        const float lat_hat[2] = {-t_hat[1], t_hat[0]};
        a_long = t_hat[0] * a_n[0] + t_hat[1] * a_n[1];
        a_lat = lat_hat[0] * a_n[0] + lat_hat[1] * a_n[1];
        sf_align_projected_gnss_accel_covariance(t_hat,
                                                 lat_hat,
                                                 window->gnss_vel_prev_std_mps,
                                                 window->gnss_vel_curr_std_mps,
                                                 dt,
                                                 gnss_accel_cov_v);
    }

    const float gyro_norm = sf_align_vec3_norm(window->mean_gyro_b);
    const float accel_norm = sf_align_vec3_norm(window->mean_accel_b);
    const bool stationary =
        gyro_norm <= align->cfg.max_stationary_gyro_radps &&
        fabsf(accel_norm - SF_ALIGN_GRAVITY_MPS2) <= align->cfg.max_stationary_accel_norm_err_mps2 &&
        speed_mid < 0.5f;
    const bool turn_valid = speed_mid > align->cfg.min_speed_mps &&
                            fabsf(course_rate) > align->cfg.min_turn_rate_radps &&
                            fabsf(a_lat) > align->cfg.min_lat_acc_mps2;
    const bool turn_heading_valid =
        sf_align_turn_consistency_update(align, turn_valid, speed_mid, course_rate, a_lat);
    const float horiz_gnss_norm = sqrtf(a_long * a_long + a_lat * a_lat);
    const bool long_valid = speed_mid > align->cfg.min_speed_mps &&
                            fabsf(a_long) > align->cfg.min_long_acc_mps2 &&
                            fabsf(a_lat) < sf_align_max(0.5f, 0.6f * fabsf(a_long)) &&
                            horiz_gnss_norm > align->cfg.min_long_acc_mps2;

    float score = 0.0f;
    if (stationary) {
        for (size_t i = 0u; i < 3u; ++i) {
            align->gravity_lp_b[i] =
                (1.0f - align->cfg.gravity_lpf_alpha) * align->gravity_lp_b[i] +
                align->cfg.gravity_lpf_alpha * window->mean_accel_b[i];
        }
    }
    if (align->cfg.use_gravity && stationary) {
        const bool gravity_state_mask[3] = {true, true, false};
        const float gravity_std =
            align->cfg.r_gravity_std_mps2 * sf_align_refinement_observation_std_scale(align);
        const float r_gravity = gravity_std * gravity_std;
        const float z2[2] = {0.0f, 0.0f};
        const size_t obs2[2] = {3u, 4u};
        const float r2[2] = {r_gravity, r_gravity};
        const float state_scale[3] = {1.0f, 1.0f, 1.0f};
        score += sf_align_apply_update2_scaled_masked(align,
                                                      z2,
                                                      obs2,
                                                      align->gravity_lp_b,
                                                      window->mean_gyro_b,
                                                      r2,
                                                      gravity_state_mask,
                                                      state_scale);
        score += sf_align_apply_update1_masked(align,
                                               -sf_align_vec3_norm(align->gravity_lp_b),
                                               5u,
                                               align->gravity_lp_b,
                                               window->mean_gyro_b,
                                               r_gravity,
                                               gravity_state_mask);
        trace->gravity_applied = true;
    }

    float horiz_accel_b[3];
    float horiz_accel_v[3];
    sf_align_remove_gravity_axis(align->q_bv, window->mean_accel_b, horiz_accel_b);
    sf_align_obs_accel_v(align->q_bv, horiz_accel_b, horiz_accel_v);
    const float horiz_imu_norm =
        sqrtf(horiz_accel_v[0] * horiz_accel_v[0] + horiz_accel_v[1] * horiz_accel_v[1]);
    const bool straight_core_valid = long_valid;
    const bool turn_core_valid = turn_heading_valid && speed_mid > (10.0f / 3.6f) &&
                                 fabsf(a_lat) > sf_align_max(align->cfg.min_lat_acc_mps2, 0.7f) &&
                                 fabsf(a_lat) > 1.5f * sf_align_max(fabsf(a_long), 0.2f);
    const bool horiz_vector_valid = speed_mid > align->cfg.min_speed_mps &&
                                    horiz_gnss_norm > align->cfg.min_long_acc_mps2 &&
                                    horiz_imu_norm > align->cfg.min_long_acc_mps2 &&
                                    (straight_core_valid || turn_core_valid);
    trace->horiz_straight_core_valid = straight_core_valid;
    trace->horiz_turn_core_valid = turn_core_valid;
    trace->horiz_gnss_norm_mps2 = horiz_gnss_norm;
    trace->horiz_imu_norm_mps2 = horiz_imu_norm;
    trace->horiz_obs_accel_vx = horiz_accel_v[0];
    trace->horiz_obs_accel_vy = horiz_accel_v[1];

    if (horiz_vector_valid) {
        const float gnss_xy[2] = {a_long, a_lat};
        float imu_xy[2];
        const float angle_err =
            sf_align_horizontal_accel_angle_error(align->q_bv, window->mean_accel_b, gnss_xy, imu_xy);
        const float observation_scale = sf_align_refinement_observation_std_scale(align);
        const float speed_q =
            sf_align_clamp((speed_mid - (10.0f / 3.6f)) / ((20.0f / 3.6f) - (10.0f / 3.6f)),
                           0.0f,
                           1.0f);
        const float accel_q =
            sf_align_clamp((sf_align_max(0.0f, fminf(horiz_gnss_norm, horiz_imu_norm)) - 0.5f) /
                               1.0f,
                           0.0f,
                           1.0f);
        float base_effective_std;
        if (turn_core_valid) {
            const float dominance = ((fabsf(a_lat) / (fabsf(a_long) + 0.2f)) - 1.5f) / 1.5f;
            const float lat_q = sf_align_clamp(
                (fabsf(a_lat) - sf_align_max(align->cfg.min_lat_acc_mps2, 0.7f)) / 1.0f,
                0.0f,
                1.0f);
            const float turn_q =
                sf_align_clamp(0.35f +
                                   0.65f * (speed_q * accel_q * lat_q *
                                            sf_align_clamp(dominance, 0.0f, 1.0f)),
                               0.35f,
                               1.0f);
            base_effective_std = align->cfg.r_horiz_yaw_std_rad / turn_q;
        } else {
            const float lat_ratio = fabsf(a_lat) / (0.5f + 0.6f * fabsf(a_long));
            const float long_q =
                sf_align_clamp((fabsf(a_long) - align->cfg.min_long_acc_mps2) / 0.8f,
                               0.0f,
                               1.0f);
            const float straight_q =
                sf_align_clamp(speed_q * accel_q * long_q *
                                   (1.0f - sf_align_clamp(lat_ratio, 0.0f, 1.0f)),
                               0.2f,
                               1.0f);
            base_effective_std = align->cfg.r_horiz_yaw_std_rad / straight_q;
        }
        const float effective_std = observation_scale * base_effective_std;
        const float effective_var = sf_align_horizontal_heading_variance(align,
                                                                         window,
                                                                         gnss_xy,
                                                                         gnss_accel_cov_v,
                                                                         effective_std *
                                                                             effective_std);
        trace->horiz_angle_err_rad = angle_err;
        trace->horiz_effective_std_rad = sqrtf(effective_var);
        score += sf_align_apply_vehicle_yaw_angle(align, angle_err, effective_var);
        align->yaw_observed = true;
        trace->horiz_accel_applied = true;
    }

    if (turn_valid && align->cfg.use_turn_gyro) {
        const float turn_gyro_std =
            align->cfg.r_turn_gyro_std_radps * sf_align_refinement_observation_std_scale(align);
        const float r = sf_align_sq(turn_gyro_std);
        const float z2[2] = {0.0f, 0.0f};
        const size_t obs2[2] = {0u, 1u};
        const float r2[2] = {r, r};
        const bool state_mask[3] = {true, true, false};
        const float state_scale[3] = {1.0f, 1.0f, 0.0f};
        score += sf_align_apply_update2_scaled_masked(align,
                                                      z2,
                                                      obs2,
                                                      window->mean_accel_b,
                                                      window->mean_gyro_b,
                                                      r2,
                                                      state_mask,
                                                      state_scale);
        trace->turn_gyro_applied = true;
    }

    align->coarse_aligned = sf_align_compute_coarse_ready(align);
    trace->coarse_alignment_ready = align->coarse_aligned;
    return score;
}

void sf_align_mount_angles_rad(const sf_align_t *align, float out_rad[3])
{
    float r[3][3];
    sf_align_quat_to_rotmat(align->q_bv, r);
    out_rad[1] = asinf(sf_align_clamp(-r[2][0], -1.0f, 1.0f));
    out_rad[0] = atan2f(r[2][1], r[2][2]);
    out_rad[2] = atan2f(r[1][0], r[0][0]);
}

void sf_align_mount_angles_deg(const sf_align_t *align, float out_deg[3])
{
    float rad[3];
    sf_align_mount_angles_rad(align, rad);
    out_deg[0] = sf_align_rad_to_deg(rad[0]);
    out_deg[1] = sf_align_rad_to_deg(rad[1]);
    out_deg[2] = sf_align_rad_to_deg(rad[2]);
}

void sf_align_sigma_deg(const sf_align_t *align, float out_deg[3])
{
    for (size_t i = 0u; i < SF_ALIGN_N_STATES; ++i) {
        out_deg[i] = sf_align_rad_to_deg(sqrtf(sf_align_max(align->p[i][i], 0.0f)));
    }
}

bool sf_align_coarse_alignment_ready(const sf_align_t *align)
{
    return align->coarse_aligned;
}

float sf_align_coarse_progress(const sf_align_t *align)
{
    float sigma_deg[3];
    sf_align_sigma_deg(align, sigma_deg);
    const float tilt_progress =
        sf_align_axis_progress(sigma_deg[0],
                               SF_ALIGN_STATIONARY_INIT_ROLL_DEG,
                               SF_ALIGN_COARSE_READY_ROLL_DEG) <
                sf_align_axis_progress(sigma_deg[1],
                                       SF_ALIGN_STATIONARY_INIT_PITCH_DEG,
                                       SF_ALIGN_COARSE_READY_PITCH_DEG)
            ? sf_align_axis_progress(sigma_deg[0],
                                     SF_ALIGN_STATIONARY_INIT_ROLL_DEG,
                                     SF_ALIGN_COARSE_READY_ROLL_DEG)
            : sf_align_axis_progress(sigma_deg[1],
                                     SF_ALIGN_STATIONARY_INIT_PITCH_DEG,
                                     SF_ALIGN_COARSE_READY_PITCH_DEG);
    const float yaw_progress = sf_align_axis_progress(sigma_deg[2],
                                                      SF_ALIGN_STATIONARY_INIT_YAW_DEG,
                                                      SF_ALIGN_COARSE_READY_YAW_DEG);
    return sf_align_clamp(SF_ALIGN_TILT_PROGRESS_WEIGHT * tilt_progress +
                              SF_ALIGN_YAW_PROGRESS_WEIGHT * yaw_progress,
                          0.0f,
                          1.0f);
}
