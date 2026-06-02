#include "runtime.h"

#include <math.h>
#include <string.h>

static float sf_maxf(float a, float b)
{
    return a > b ? a : b;
}

sf_ekf_process_noise_t sf_ekf_process_noise_default(void)
{
    sf_ekf_process_noise_t noise;
    noise.gyro_var = 2.2873113e-7f * 10.0f;
    noise.accel_var = 2.4504214e-5f * 15.0f;
    noise.gyro_bias_rw_var = 0.0002e-9f;
    noise.accel_bias_rw_var = 0.002e-9f;
    noise.mount_align_rw_var_axes[0] = 0.0f;
    noise.mount_align_rw_var_axes[1] = 0.0f;
    noise.mount_align_rw_var_axes[2] = 0.0f;
    return noise;
}

void sf_ekf_runtime_init(sf_ekf_runtime_state_t *state)
{
    memset(state, 0, sizeof(*state));
    state->nominal.q0 = 1.0f;
    state->nominal.q_bv0 = 1.0f;
    state->noise = sf_ekf_process_noise_default();
    state->gravity_mss = SF_EKF_GRAVITY_MSS;
    state->gnss_position_outlier_gate_sigma = SF_EKF_GNSS_OUTLIER_GATE_SIGMA;
    state->gnss_velocity_outlier_gate_sigma = SF_EKF_GNSS_OUTLIER_GATE_SIGMA;
}

void sf_ekf_normalize_quat(float q[4])
{
    float n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    float inv_n;
    if (!(n2 > 1.0e-12f) || !isfinite(n2)) {
        q[0] = 1.0f;
        q[1] = 0.0f;
        q[2] = 0.0f;
        q[3] = 0.0f;
        return;
    }
    inv_n = 1.0f / sqrtf(n2);
    q[0] *= inv_n;
    q[1] *= inv_n;
    q[2] *= inv_n;
    q[3] *= inv_n;
}

void sf_ekf_quat_multiply(const float p[4], const float q[4], float out[4])
{
    out[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3];
    out[1] = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2];
    out[2] = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1];
    out[3] = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0];
}

static void sf_nominal_get_q_nv(const sf_ekf_nominal_state_t *nominal, float q[4])
{
    q[0] = nominal->q0;
    q[1] = nominal->q1;
    q[2] = nominal->q2;
    q[3] = nominal->q3;
}

static void sf_nominal_set_q_nv(sf_ekf_nominal_state_t *nominal, const float q[4])
{
    nominal->q0 = q[0];
    nominal->q1 = q[1];
    nominal->q2 = q[2];
    nominal->q3 = q[3];
}

static void sf_nominal_get_q_bv(const sf_ekf_nominal_state_t *nominal, float q[4])
{
    q[0] = nominal->q_bv0;
    q[1] = nominal->q_bv1;
    q[2] = nominal->q_bv2;
    q[3] = nominal->q_bv3;
}

static void sf_nominal_set_q_bv(sf_ekf_nominal_state_t *nominal, const float q[4])
{
    nominal->q_bv0 = q[0];
    nominal->q_bv1 = q[1];
    nominal->q_bv2 = q[2];
    nominal->q_bv3 = q[3];
}

static void sf_normalize_nominal_quats(sf_ekf_nominal_state_t *nominal)
{
    float q[4];
    sf_nominal_get_q_nv(nominal, q);
    sf_ekf_normalize_quat(q);
    sf_nominal_set_q_nv(nominal, q);
    sf_nominal_get_q_bv(nominal, q);
    sf_ekf_normalize_quat(q);
    sf_nominal_set_q_bv(nominal, q);
}

void sf_ekf_inject_error_state(sf_ekf_nominal_state_t *nominal,
                               const float dx[SF_EKF_ERROR_STATES])
{
    float dq[4];
    float q_old[4];
    float q_new[4];
    float dq_bv[4];
    float q_bv_old[4];
    float q_bv_new[4];

    dq[0] = 1.0f;
    dq[1] = 0.5f * dx[0];
    dq[2] = 0.5f * dx[1];
    dq[3] = 0.5f * dx[2];
    sf_nominal_get_q_nv(nominal, q_old);
    sf_ekf_quat_multiply(q_old, dq, q_new);
    sf_ekf_normalize_quat(q_new);
    sf_nominal_set_q_nv(nominal, q_new);

    nominal->vn += dx[3];
    nominal->ve += dx[4];
    nominal->vd += dx[5];
    nominal->pn += dx[6];
    nominal->pe += dx[7];
    nominal->pd += dx[8];
    nominal->bgx += dx[9];
    nominal->bgy += dx[10];
    nominal->bgz += dx[11];
    nominal->bax += dx[12];
    nominal->bay += dx[13];
    nominal->baz += dx[14];

    dq_bv[0] = 1.0f;
    dq_bv[1] = 0.5f * dx[15];
    dq_bv[2] = 0.5f * dx[16];
    dq_bv[3] = 0.5f * dx[17];
    sf_nominal_get_q_bv(nominal, q_bv_old);
    sf_ekf_quat_multiply(dq_bv, q_bv_old, q_bv_new);
    sf_ekf_normalize_quat(q_bv_new);
    sf_nominal_set_q_bv(nominal, q_bv_new);
}

void sf_ekf_covariance_symmetrize(float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES])
{
    unsigned int i;
    unsigned int j;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        for (j = i + 1u; j < SF_EKF_ERROR_STATES; ++j) {
            float v = 0.5f * (p[i][j] + p[j][i]);
            p[i][j] = v;
            p[j][i] = v;
        }
    }
}

void sf_ekf_update_covariance_joseph_scalar(
    float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    float innovation_var,
    const float h[SF_EKF_ERROR_STATES],
    const float k[SF_EKF_ERROR_STATES])
{
    float ph[SF_EKF_ERROR_STATES] = {0.0f};
    unsigned int i;
    unsigned int j;
    unsigned int a;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        for (a = 0; a < SF_EKF_ERROR_STATES; ++a) {
            ph[i] += p[i][a] * h[a];
        }
    }
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        for (j = i; j < SF_EKF_ERROR_STATES; ++j) {
            float updated = p[i][j] - k[i] * ph[j] - ph[i] * k[j] +
                            innovation_var * k[i] * k[j];
            p[i][j] = updated;
            p[j][i] = updated;
        }
    }
}

static void sf_ekf_apply_reset_block(float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
                                     unsigned int offset,
                                     const float dtheta[3])
{
    float g_reset_theta[3][3];
    float p_aa[3][3] = {{0.0f}};
    float p_ab[3][SF_EKF_ERROR_STATES - 3] = {{0.0f}};
    float next_aa[3][3] = {{0.0f}};
    unsigned int i;
    unsigned int j;
    unsigned int k;

    sf_ekf_attitude_reset_jacobian(dtheta, g_reset_theta);
    for (i = 0; i < 3u; ++i) {
        for (j = 0; j < 3u; ++j) {
            p_aa[i][j] = p[offset + i][offset + j];
        }
        for (j = 0; j < SF_EKF_ERROR_STATES; ++j) {
            if (j >= offset && j < offset + 3u) {
                continue;
            }
            p_ab[i][j < offset ? j : j - 3u] = p[offset + i][j];
        }
    }

    for (i = 0; i < 3u; ++i) {
        for (j = 0; j < 3u; ++j) {
            for (k = 0; k < 3u; ++k) {
                next_aa[i][j] += g_reset_theta[i][k] * p_aa[k][j];
            }
        }
    }

    for (i = 0; i < 3u; ++i) {
        for (j = 0; j < 3u; ++j) {
            float accum = 0.0f;
            for (k = 0; k < 3u; ++k) {
                accum += next_aa[i][k] * g_reset_theta[j][k];
            }
            p[offset + i][offset + j] = accum;
        }
        for (j = 0; j < SF_EKF_ERROR_STATES; ++j) {
            float accum = 0.0f;
            if (j >= offset && j < offset + 3u) {
                continue;
            }
            for (k = 0; k < 3u; ++k) {
                accum += g_reset_theta[i][k] * p_ab[k][j < offset ? j : j - 3u];
            }
            p[offset + i][j] = accum;
            p[j][offset + i] = accum;
        }
    }
}

void sf_ekf_apply_reset(float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
                        const float dx[SF_EKF_ERROR_STATES])
{
    const float dtheta[3] = {dx[0], dx[1], dx[2]};
    sf_ekf_apply_reset_block(p, 0u, dtheta);
    sf_ekf_covariance_symmetrize(p);
}

void sf_ekf_runtime_fuse_scalar(sf_ekf_runtime_state_t *state,
                                float innovation_var,
                                const float h[SF_EKF_ERROR_STATES],
                                const float k[SF_EKF_ERROR_STATES],
                                const float dx[SF_EKF_ERROR_STATES])
{
    sf_ekf_update_covariance_joseph_scalar(state->p, innovation_var, h, k);
    sf_ekf_inject_error_state(&state->nominal, dx);
    sf_ekf_apply_reset(state->p, dx);
}

void sf_ekf_runtime_fuse_batch(sf_ekf_runtime_state_t *state,
                               unsigned int obs_count,
                               const float h_rows[SF_EKF_MAX_BATCH_OBS][SF_EKF_ERROR_STATES],
                               const float residuals[SF_EKF_MAX_BATCH_OBS],
                               const float variances[SF_EKF_MAX_BATCH_OBS])
{
    float dx[SF_EKF_ERROR_STATES] = {0.0f};
    unsigned int row;
    unsigned int i;
    unsigned int j;
    unsigned int state_index;
    if (obs_count == 0u || obs_count > SF_EKF_MAX_BATCH_OBS) {
        return;
    }
    for (row = 0; row < obs_count; ++row) {
        const float *h = h_rows[row];
        float ph[SF_EKF_ERROR_STATES] = {0.0f};
        float s = variances[row];
        float hd = 0.0f;
        float effective_residual;
        float alpha;
        if (!(s > 0.0f) || !isfinite(s)) {
            continue;
        }
        for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
            for (state_index = 0; state_index < SF_EKF_ERROR_STATES; ++state_index) {
                ph[i] += state->p[i][state_index] * h[state_index];
            }
        }
        for (state_index = 0; state_index < SF_EKF_ERROR_STATES; ++state_index) {
            s += h[state_index] * ph[state_index];
        }
        if (!(s > 0.0f) || !isfinite(s)) {
            continue;
        }
        for (state_index = 0; state_index < SF_EKF_ERROR_STATES; ++state_index) {
            hd += h[state_index] * dx[state_index];
        }
        effective_residual = residuals[row] - hd;
        alpha = effective_residual / s;
        for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
            dx[i] += ph[i] * alpha;
        }
        for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
            for (j = i; j < SF_EKF_ERROR_STATES; ++j) {
                float value = state->p[i][j] - ph[i] * ph[j] / s;
                state->p[i][j] = value;
                state->p[j][i] = value;
            }
        }
    }
    sf_ekf_inject_error_state(&state->nominal, dx);
    sf_ekf_apply_reset(state->p, dx);
}

static void sf_fuse_observation_with_residual(sf_ekf_runtime_state_t *state,
                                              const sf_ekf_scalar_observation_t *obs,
                                              float residual)
{
    float dx[SF_EKF_ERROR_STATES];
    unsigned int i;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        dx[i] = obs->k[i] * residual;
    }
    sf_ekf_runtime_fuse_scalar(state, obs->s, obs->h, obs->k, dx);
}

void sf_ekf_runtime_fuse_gps_pos_n(sf_ekf_runtime_state_t *state, float pos_n, float variance)
{
    sf_ekf_scalar_observation_t obs;
    sf_ekf_gps_pos_n_observation(state->p, variance, &obs);
    sf_fuse_observation_with_residual(state, &obs, pos_n - state->nominal.pn);
}

void sf_ekf_runtime_fuse_gps_pos_e(sf_ekf_runtime_state_t *state, float pos_e, float variance)
{
    sf_ekf_scalar_observation_t obs;
    sf_ekf_gps_pos_e_observation(state->p, variance, &obs);
    sf_fuse_observation_with_residual(state, &obs, pos_e - state->nominal.pe);
}

void sf_ekf_runtime_fuse_gps_pos_d(sf_ekf_runtime_state_t *state, float pos_d, float variance)
{
    sf_ekf_scalar_observation_t obs;
    sf_ekf_gps_pos_d_observation(state->p, variance, &obs);
    sf_fuse_observation_with_residual(state, &obs, pos_d - state->nominal.pd);
}

void sf_ekf_runtime_fuse_gps_vel_n(sf_ekf_runtime_state_t *state, float vel_n, float variance)
{
    sf_ekf_scalar_observation_t obs;
    sf_ekf_gps_vel_n_observation(state->p, variance, &obs);
    sf_fuse_observation_with_residual(state, &obs, vel_n - state->nominal.vn);
}

void sf_ekf_runtime_fuse_gps_vel_e(sf_ekf_runtime_state_t *state, float vel_e, float variance)
{
    sf_ekf_scalar_observation_t obs;
    sf_ekf_gps_vel_e_observation(state->p, variance, &obs);
    sf_fuse_observation_with_residual(state, &obs, vel_e - state->nominal.ve);
}

void sf_ekf_runtime_fuse_gps_vel_d(sf_ekf_runtime_state_t *state, float vel_d, float variance)
{
    sf_ekf_scalar_observation_t obs;
    sf_ekf_gps_vel_d_observation(state->p, variance, &obs);
    sf_fuse_observation_with_residual(state, &obs, vel_d - state->nominal.vd);
}

void sf_ekf_nominal_vehicle_velocity(const sf_ekf_nominal_state_t *nominal, float out_v_vehicle[3])
{
    float q0 = nominal->q0;
    float q1 = nominal->q1;
    float q2 = nominal->q2;
    float q3 = nominal->q3;
    float vn = nominal->vn;
    float ve = nominal->ve;
    float vd = nominal->vd;
    out_v_vehicle[0] = (1.0f - 2.0f * q2 * q2 - 2.0f * q3 * q3) * vn +
                       2.0f * (q1 * q2 + q0 * q3) * ve +
                       2.0f * (q1 * q3 - q0 * q2) * vd;
    out_v_vehicle[1] = 2.0f * (q1 * q2 - q0 * q3) * vn +
                       (1.0f - 2.0f * q1 * q1 - 2.0f * q3 * q3) * ve +
                       2.0f * (q2 * q3 + q0 * q1) * vd;
    out_v_vehicle[2] = 2.0f * (q1 * q3 + q0 * q2) * vn +
                       2.0f * (q2 * q3 - q0 * q1) * ve +
                       (1.0f - 2.0f * q1 * q1 - 2.0f * q2 * q2) * vd;
}

static float sf_ekf_vehicle_roll_rad(const sf_ekf_nominal_state_t *nominal)
{
    float q0 = nominal->q0;
    float q1 = nominal->q1;
    float q2 = nominal->q2;
    float q3 = nominal->q3;
    float c21 = 2.0f * (q2 * q3 + q0 * q1);
    float c22 = 1.0f - 2.0f * q1 * q1 - 2.0f * q2 * q2;
    return atan2f(c21, c22);
}

void sf_ekf_runtime_fuse_body_speed_x(sf_ekf_runtime_state_t *state,
                                      float speed_mps,
                                      float r_speed)
{
    sf_ekf_scalar_observation_t obs;
    float v_vehicle[3];
    if (!(r_speed > 0.0f) || !isfinite(r_speed) || !isfinite(speed_mps)) {
        return;
    }
    sf_ekf_body_vel_x_observation(&state->nominal, state->p, r_speed, &obs);
    sf_ekf_nominal_vehicle_velocity(&state->nominal, v_vehicle);
    sf_fuse_observation_with_residual(state, &obs, speed_mps - v_vehicle[0]);
}

void sf_ekf_runtime_fuse_zero_vel(sf_ekf_runtime_state_t *state, float r_zero_vel)
{
    if (!(r_zero_vel > 0.0f) || !isfinite(r_zero_vel)) {
        return;
    }
    sf_ekf_runtime_fuse_gps_vel_n(state, 0.0f, r_zero_vel);
    sf_ekf_runtime_fuse_gps_vel_e(state, 0.0f, r_zero_vel);
    sf_ekf_runtime_fuse_gps_vel_d(state, 0.0f, r_zero_vel);
}

void sf_ekf_runtime_fuse_vehicle_roll_prior(sf_ekf_runtime_state_t *state,
                                            float r_vehicle_roll)
{
    sf_ekf_scalar_observation_t obs;
    float roll;
    if (!(r_vehicle_roll > 0.0f) || !isfinite(r_vehicle_roll)) {
        return;
    }
    roll = sf_ekf_vehicle_roll_rad(&state->nominal);
    if (!isfinite(roll)) {
        return;
    }
    sf_ekf_vehicle_roll_prior_observation(&state->nominal, state->p, r_vehicle_roll, &obs);
    if (!(obs.s > 0.0f) || !isfinite(obs.s)) {
        return;
    }
    sf_fuse_observation_with_residual(state, &obs, -roll);
}

void sf_ekf_runtime_set_gnss_position_outlier_gate_sigma(sf_ekf_runtime_state_t *state,
                                                         float gate_sigma)
{
    if (gate_sigma >= 0.0f) {
        state->gnss_position_outlier_gate_sigma = gate_sigma;
    }
}

void sf_ekf_runtime_set_gnss_velocity_outlier_gate_sigma(sf_ekf_runtime_state_t *state,
                                                         float gate_sigma)
{
    if (gate_sigma >= 0.0f) {
        state->gnss_velocity_outlier_gate_sigma = gate_sigma;
    }
}

void sf_ekf_runtime_require_next_gnss_gate_pass(sf_ekf_runtime_state_t *state)
{
    state->gnss_position_gate_state.require_next_gate_pass = 1;
    state->gnss_velocity_gate_state.require_next_gate_pass = 1;
}

static void sf_push_batch_row(float h_rows[SF_EKF_MAX_BATCH_OBS][SF_EKF_ERROR_STATES],
                              float residuals[SF_EKF_MAX_BATCH_OBS],
                              float variances[SF_EKF_MAX_BATCH_OBS],
                              unsigned int *obs_count,
                              const float h[SF_EKF_ERROR_STATES],
                              float residual,
                              float variance)
{
    if (*obs_count >= SF_EKF_MAX_BATCH_OBS || !(variance > 0.0f) || !isfinite(variance)) {
        return;
    }
    memcpy(h_rows[*obs_count], h, sizeof(h_rows[*obs_count]));
    residuals[*obs_count] = residual;
    variances[*obs_count] = variance;
    *obs_count += 1u;
}

void sf_ekf_runtime_fuse_body_vel_yz(sf_ekf_runtime_state_t *state,
                                     float r_body_vel_y,
                                     float r_body_vel_z)
{
    float h_rows[SF_EKF_MAX_BATCH_OBS][SF_EKF_ERROR_STATES] = {{0.0f}};
    float residuals[SF_EKF_MAX_BATCH_OBS] = {0.0f};
    float variances[SF_EKF_MAX_BATCH_OBS] = {0.0f};
    float v_vehicle[3];
    unsigned int obs_count = 0u;
    sf_ekf_scalar_observation_t obs_y;
    sf_ekf_scalar_observation_t obs_z;

    sf_ekf_nominal_vehicle_velocity(&state->nominal, v_vehicle);
    sf_ekf_body_vel_y_observation(&state->nominal, state->p, r_body_vel_y, &obs_y);
    sf_ekf_body_vel_z_observation(&state->nominal, state->p, r_body_vel_z, &obs_z);
    sf_push_batch_row(h_rows, residuals, variances, &obs_count, obs_y.h, -v_vehicle[1], r_body_vel_y);
    sf_push_batch_row(h_rows, residuals, variances, &obs_count, obs_z.h, -v_vehicle[2], r_body_vel_z);
    sf_ekf_runtime_fuse_batch(state, obs_count, h_rows, residuals, variances);
}

typedef struct sf_gnss_gate_event_bits {
    uint32_t rejected;
    uint32_t consecutive_rejected;
    uint32_t gap_bypass;
    uint32_t accuracy_bypass;
} sf_gnss_gate_event_bits_t;

static int sf_gnss_group_passes_per_axis_sigma_gate(
    const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES],
    unsigned int state_base,
    const float residuals[3],
    const float variances[3],
    float gate_sigma)
{
    float gate_nis;
    unsigned int i;
    if (isinf(gate_sigma)) {
        return 1;
    }
    if (state_base + 2u >= SF_EKF_ERROR_STATES) {
        return 1;
    }
    gate_nis = gate_sigma * gate_sigma;
    for (i = 0; i < 3u; ++i) {
        float innovation_var;
        float nis;
        if (!isfinite(residuals[i]) || !(variances[i] > 0.0f) || !isfinite(variances[i])) {
            continue;
        }
        innovation_var = p[state_base + i][state_base + i] + variances[i];
        if (!(innovation_var > 0.0f) || !isfinite(innovation_var)) {
            continue;
        }
        nis = residuals[i] * residuals[i] / innovation_var;
        if (isfinite(nis) && nis > gate_nis) {
            return 0;
        }
    }
    return 1;
}

static int sf_accuracy_rms(const float variances[3], float *out_rms)
{
    float sum = 0.0f;
    float count = 0.0f;
    unsigned int i;
    for (i = 0; i < 3u; ++i) {
        if (variances[i] > 0.0f && isfinite(variances[i])) {
            sum += variances[i];
            count += 1.0f;
        }
    }
    if (!(count > 0.0f)) {
        return 0;
    }
    *out_rms = sqrtf(sum / count);
    return 1;
}

static int sf_apply_gnss_gate_policy(sf_ekf_gnss_gate_state_t *state,
                                     float t_s,
                                     int gate_failed,
                                     int has_accuracy_rms,
                                     float accuracy_rms,
                                     sf_gnss_gate_event_bits_t bits,
                                     uint32_t *event_mask)
{
    int require_gate_pass = state->require_next_gate_pass;
    int gap_bypass = !require_gate_pass && state->has_last_t_s &&
                     (t_s - state->last_t_s > SF_EKF_GNSS_OUTLIER_GAP_BYPASS_S);
    int accuracy_bypass = 0;
    int accepted;
    uint32_t mask = 0u;
    if (!require_gate_pass && has_accuracy_rms && state->has_last_accuracy_rms &&
        state->last_accuracy_rms > 0.0f) {
        accuracy_bypass =
            accuracy_rms <= state->last_accuracy_rms * SF_EKF_GNSS_OUTLIER_ACCURACY_IMPROVEMENT_RATIO;
    }

    if (!gate_failed) {
        accepted = 1;
    } else if (gap_bypass) {
        accepted = 1;
        mask = bits.gap_bypass;
    } else if (accuracy_bypass) {
        accepted = 1;
        mask = bits.accuracy_bypass;
    } else {
        uint8_t next_rejections = (state->consecutive_rejections == UINT8_MAX)
                                      ? UINT8_MAX
                                      : (uint8_t)(state->consecutive_rejections + 1u);
        accepted = 0;
        mask = bits.rejected;
        if (next_rejections >= SF_EKF_GNSS_OUTLIER_CONSECUTIVE_REJECTION_EVENT_COUNT) {
            mask |= bits.consecutive_rejected;
        }
    }

    if (accepted) {
        state->consecutive_rejections = 0u;
    } else if (state->consecutive_rejections != UINT8_MAX) {
        state->consecutive_rejections++;
    }
    if (isfinite(t_s)) {
        state->has_last_t_s = 1;
        state->last_t_s = t_s;
    } else {
        state->has_last_t_s = 0;
    }
    if (has_accuracy_rms) {
        state->has_last_accuracy_rms = 1;
        state->last_accuracy_rms = accuracy_rms;
    }
    state->require_next_gate_pass = 0;
    *event_mask |= mask;
    return accepted;
}

static int sf_runtime_gnss_group_passes_gate(sf_ekf_runtime_state_t *state,
                                             float t_s,
                                             unsigned int state_base,
                                             const float residuals[3],
                                             const float variances[3],
                                             int position_group,
                                             uint32_t *event_mask)
{
    float gate_sigma = position_group ? state->gnss_position_outlier_gate_sigma
                                      : state->gnss_velocity_outlier_gate_sigma;
    sf_ekf_gnss_gate_state_t *gate_state = position_group ? &state->gnss_position_gate_state
                                                          : &state->gnss_velocity_gate_state;
    sf_gnss_gate_event_bits_t bits;
    float rms = 0.0f;
    int has_rms;
    int gate_failed;
    if (isinf(gate_sigma)) {
        return 1;
    }
    bits.rejected = position_group ? SF_EKF_GNSS_EVENT_POSITION_REJECTED
                                   : SF_EKF_GNSS_EVENT_VELOCITY_REJECTED;
    bits.consecutive_rejected = position_group ? SF_EKF_GNSS_EVENT_POSITION_CONSECUTIVE_REJECTED
                                               : SF_EKF_GNSS_EVENT_VELOCITY_CONSECUTIVE_REJECTED;
    bits.gap_bypass = position_group ? SF_EKF_GNSS_EVENT_POSITION_GAP_BYPASS
                                     : SF_EKF_GNSS_EVENT_VELOCITY_GAP_BYPASS;
    bits.accuracy_bypass = position_group ? SF_EKF_GNSS_EVENT_POSITION_ACCURACY_BYPASS
                                          : SF_EKF_GNSS_EVENT_VELOCITY_ACCURACY_BYPASS;
    gate_failed = !sf_gnss_group_passes_per_axis_sigma_gate(state->p, state_base, residuals,
                                                            variances, gate_sigma);
    has_rms = sf_accuracy_rms(variances, &rms);
    return sf_apply_gnss_gate_policy(gate_state, t_s, gate_failed, has_rms, rms, bits, event_mask);
}

sf_ekf_gnss_update_result_t sf_ekf_runtime_fuse_gps_nhc_batch(
    sf_ekf_runtime_state_t *state,
    const sf_ekf_gnss_ned_sample_t *sample,
    int use_body_vel_y,
    float r_body_vel_y,
    int use_body_vel_z,
    float r_body_vel_z)
{
    sf_ekf_gnss_update_result_t result;
    float h_rows[SF_EKF_MAX_BATCH_OBS][SF_EKF_ERROR_STATES] = {{0.0f}};
    float residuals[SF_EKF_MAX_BATCH_OBS] = {0.0f};
    float variances[SF_EKF_MAX_BATCH_OBS] = {0.0f};
    float pos_residuals[3];
    float pos_variances[3];
    float vel_residuals[3];
    float vel_variances[3];
    unsigned int obs_count = 0u;
    unsigned int axis;
    float v_vehicle[3];
    sf_ekf_scalar_observation_t obs;

    result.accepted_position = 0;
    result.accepted_velocity = 0;
    result.event_mask = 0u;

    pos_residuals[0] = sample->pos_ned_m[0] - state->nominal.pn;
    pos_residuals[1] = sample->pos_ned_m[1] - state->nominal.pe;
    pos_residuals[2] = sample->pos_ned_m[2] - state->nominal.pd;
    vel_residuals[0] = sample->vel_ned_mps[0] - state->nominal.vn;
    vel_residuals[1] = sample->vel_ned_mps[1] - state->nominal.ve;
    vel_residuals[2] = sample->vel_ned_mps[2] - state->nominal.vd;
    for (axis = 0; axis < 3u; ++axis) {
        pos_variances[axis] = sample->pos_std_m[axis] * sample->pos_std_m[axis];
        vel_variances[axis] = sample->vel_std_mps[axis] * sample->vel_std_mps[axis];
    }

    result.accepted_position = sf_runtime_gnss_group_passes_gate(
        state, sample->t_s, 6u, pos_residuals, pos_variances, 1, &result.event_mask);
    if (result.accepted_position) {
        for (axis = 0; axis < 3u; ++axis) {
            float h[SF_EKF_ERROR_STATES] = {0.0f};
            h[6u + axis] = 1.0f;
            sf_push_batch_row(h_rows, residuals, variances, &obs_count, h, pos_residuals[axis],
                              pos_variances[axis]);
        }
    }

    result.accepted_velocity = sf_runtime_gnss_group_passes_gate(
        state, sample->t_s, 3u, vel_residuals, vel_variances, 0, &result.event_mask);
    if (result.accepted_velocity) {
        for (axis = 0; axis < 3u; ++axis) {
            float h[SF_EKF_ERROR_STATES] = {0.0f};
            h[3u + axis] = 1.0f;
            sf_push_batch_row(h_rows, residuals, variances, &obs_count, h, vel_residuals[axis],
                              vel_variances[axis]);
        }
    }

    sf_ekf_nominal_vehicle_velocity(&state->nominal, v_vehicle);
    if (use_body_vel_y) {
        sf_ekf_body_vel_y_observation(&state->nominal, state->p, r_body_vel_y, &obs);
        sf_push_batch_row(h_rows, residuals, variances, &obs_count, obs.h, -v_vehicle[1],
                          r_body_vel_y);
    }
    if (use_body_vel_z) {
        sf_ekf_body_vel_z_observation(&state->nominal, state->p, r_body_vel_z, &obs);
        sf_push_batch_row(h_rows, residuals, variances, &obs_count, obs.h, -v_vehicle[2],
                          r_body_vel_z);
    }
    sf_ekf_runtime_fuse_batch(state, obs_count, h_rows, residuals, variances);
    return result;
}

void sf_ekf_runtime_fuse_gps_nhc_batch_no_gate(sf_ekf_runtime_state_t *state,
                                               const sf_ekf_gnss_ned_sample_t *sample,
                                               int use_body_vel_y,
                                               float r_body_vel_y,
                                               int use_body_vel_z,
                                               float r_body_vel_z)
{
    float h_rows[SF_EKF_MAX_BATCH_OBS][SF_EKF_ERROR_STATES] = {{0.0f}};
    float residuals[SF_EKF_MAX_BATCH_OBS] = {0.0f};
    float variances[SF_EKF_MAX_BATCH_OBS] = {0.0f};
    unsigned int obs_count = 0u;
    unsigned int axis;
    float v_vehicle[3];
    sf_ekf_scalar_observation_t obs;

    for (axis = 0; axis < 3u; ++axis) {
        float h[SF_EKF_ERROR_STATES] = {0.0f};
        h[6u + axis] = 1.0f;
        sf_push_batch_row(h_rows, residuals, variances, &obs_count, h,
                          sample->pos_ned_m[axis] -
                              (axis == 0u ? state->nominal.pn
                               : axis == 1u ? state->nominal.pe
                                             : state->nominal.pd),
                          sample->pos_std_m[axis] * sample->pos_std_m[axis]);
    }
    for (axis = 0; axis < 3u; ++axis) {
        float h[SF_EKF_ERROR_STATES] = {0.0f};
        h[3u + axis] = 1.0f;
        sf_push_batch_row(h_rows, residuals, variances, &obs_count, h,
                          sample->vel_ned_mps[axis] -
                              (axis == 0u ? state->nominal.vn
                               : axis == 1u ? state->nominal.ve
                                             : state->nominal.vd),
                          sample->vel_std_mps[axis] * sample->vel_std_mps[axis]);
    }

    sf_ekf_nominal_vehicle_velocity(&state->nominal, v_vehicle);
    if (use_body_vel_y) {
        sf_ekf_body_vel_y_observation(&state->nominal, state->p, r_body_vel_y, &obs);
        sf_push_batch_row(h_rows, residuals, variances, &obs_count, obs.h, -v_vehicle[1],
                          r_body_vel_y);
    }
    if (use_body_vel_z) {
        sf_ekf_body_vel_z_observation(&state->nominal, state->p, r_body_vel_z, &obs);
        sf_push_batch_row(h_rows, residuals, variances, &obs_count, obs.h, -v_vehicle[2],
                          r_body_vel_z);
    }
    sf_ekf_runtime_fuse_batch(state, obs_count, h_rows, residuals, variances);
}

static void sf_discrete_noise(const sf_ekf_runtime_state_t *state,
                              const sf_ekf_imu_delta_t *imu,
                              float q[SF_EKF_NOISE_STATES])
{
    float dt = sf_maxf(imu->dt, 1.0e-9f);
    q[0] = state->noise.gyro_var * dt;
    q[1] = q[0];
    q[2] = q[0];
    q[3] = state->noise.accel_var * dt;
    q[4] = q[3];
    q[5] = q[3];
    q[6] = state->noise.gyro_bias_rw_var / dt;
    q[7] = q[6];
    q[8] = q[6];
    q[9] = state->noise.accel_bias_rw_var / dt;
    q[10] = q[9];
    q[11] = q[9];
    q[12] = state->freeze_misalignment_states ? 0.0f : state->noise.mount_align_rw_var_axes[0] * dt;
    q[13] = state->freeze_misalignment_states ? 0.0f : state->noise.mount_align_rw_var_axes[1] * dt;
    q[14] = state->freeze_misalignment_states ? 0.0f : state->noise.mount_align_rw_var_axes[2] * dt;
}

void sf_ekf_runtime_predict_covariance(
    sf_ekf_runtime_state_t *state,
    const sf_ekf_error_transition_t *transition,
    const sf_ekf_imu_delta_t *imu)
{
    float q[SF_EKF_NOISE_STATES];
    float fp[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES];
    float next[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES];
    unsigned int i;
    unsigned int j;
    unsigned int k_index;
    unsigned int l_index;

    sf_discrete_noise(state, imu, q);
    memset(fp, 0, sizeof(fp));
    memset(next, 0, sizeof(next));

    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        for (k_index = 0; k_index < SF_EKF_F_ROW_COUNTS[i]; ++k_index) {
            unsigned int k = SF_EKF_F_ROW_COLS[i][k_index];
            float f = transition->f[i][k];
            for (j = 0; j < SF_EKF_ERROR_STATES; ++j) {
                fp[i][j] += f * state->p[k][j];
            }
        }
    }

    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        for (j = 0; j < SF_EKF_ERROR_STATES; ++j) {
            for (l_index = 0; l_index < SF_EKF_F_ROW_COUNTS[j]; ++l_index) {
                unsigned int l = SF_EKF_F_ROW_COLS[j][l_index];
                next[i][j] += fp[i][l] * transition->f[j][l];
            }
        }
    }

    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        for (j = 0; j < SF_EKF_ERROR_STATES; ++j) {
            float gqgt = 0.0f;
            for (k_index = 0; k_index < SF_EKF_G_ROW_COUNTS[i]; ++k_index) {
                unsigned int k = SF_EKF_G_ROW_COLS[i][k_index];
                float gik = transition->g[i][k];
                unsigned int has_jk = 0u;
                for (l_index = 0; l_index < SF_EKF_G_ROW_COUNTS[j]; ++l_index) {
                    if (SF_EKF_G_ROW_COLS[j][l_index] == k) {
                        has_jk = 1u;
                        break;
                    }
                }
                if (has_jk) {
                    gqgt += gik * q[k] * transition->g[j][k];
                }
            }
            next[i][j] += gqgt;
        }
    }

    memcpy(state->p, next, sizeof(state->p));
    sf_ekf_covariance_symmetrize(state->p);
}

void sf_ekf_runtime_predict(sf_ekf_runtime_state_t *state, const sf_ekf_imu_delta_t *imu)
{
    sf_ekf_error_transition_t transition;
    if (!(imu->dt > 0.0f) || !isfinite(imu->dt)) {
        return;
    }
    sf_ekf_error_transition_with_gravity(&state->nominal, imu, state->gravity_mss, &transition);
    sf_ekf_predict_nominal_with_gravity(&state->nominal, imu, state->gravity_mss);
    sf_normalize_nominal_quats(&state->nominal);
    sf_ekf_runtime_predict_covariance(state, &transition, imu);
}
