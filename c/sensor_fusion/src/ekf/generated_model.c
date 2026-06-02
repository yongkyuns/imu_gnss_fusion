#include "generated_model.h"

#include <math.h>

#define SF_EKF_UNUSED(x) ((void)(x))

#include "generated/error_transition_support_generated.inc"

void sf_ekf_predict_nominal(sf_ekf_nominal_state_t *nominal, const sf_ekf_imu_delta_t *imu)
{
    sf_ekf_predict_nominal_with_gravity(nominal, imu, SF_EKF_GRAVITY_MSS);
}

void sf_ekf_predict_nominal_with_gravity(
    sf_ekf_nominal_state_t *nominal,
    const sf_ekf_imu_delta_t *imu,
    float gravity_mss)
{
    const float q0 = nominal->q0;
    const float q1 = nominal->q1;
    const float q2 = nominal->q2;
    const float q3 = nominal->q3;
    const float vn = nominal->vn;
    const float ve = nominal->ve;
    const float vd = nominal->vd;
    const float pn = nominal->pn;
    const float pe = nominal->pe;
    const float pd = nominal->pd;
    const float bgx = nominal->bgx;
    const float bgy = nominal->bgy;
    const float bgz = nominal->bgz;
    const float bax = nominal->bax;
    const float bay = nominal->bay;
    const float baz = nominal->baz;
    const float q_bv0 = nominal->q_bv0;
    const float q_bv1 = nominal->q_bv1;
    const float q_bv2 = nominal->q_bv2;
    const float q_bv3 = nominal->q_bv3;
    const float dax = imu->dax;
    const float day = imu->day;
    const float daz = imu->daz;
    const float dvx = imu->dvx;
    const float dvy = imu->dvy;
    const float dvz = imu->dvz;
    const float dt = imu->dt;
    const float g = gravity_mss;
    SF_EKF_UNUSED(q0);
    SF_EKF_UNUSED(q1);
    SF_EKF_UNUSED(q2);
    SF_EKF_UNUSED(q3);
    SF_EKF_UNUSED(vn);
    SF_EKF_UNUSED(ve);
    SF_EKF_UNUSED(vd);
    SF_EKF_UNUSED(pn);
    SF_EKF_UNUSED(pe);
    SF_EKF_UNUSED(pd);
    SF_EKF_UNUSED(bgx);
    SF_EKF_UNUSED(bgy);
    SF_EKF_UNUSED(bgz);
    SF_EKF_UNUSED(bax);
    SF_EKF_UNUSED(bay);
    SF_EKF_UNUSED(baz);
    SF_EKF_UNUSED(q_bv0);
    SF_EKF_UNUSED(q_bv1);
    SF_EKF_UNUSED(q_bv2);
    SF_EKF_UNUSED(q_bv3);
    SF_EKF_UNUSED(dax);
    SF_EKF_UNUSED(day);
    SF_EKF_UNUSED(daz);
    SF_EKF_UNUSED(dvx);
    SF_EKF_UNUSED(dvy);
    SF_EKF_UNUSED(dvz);
    SF_EKF_UNUSED(dt);
    SF_EKF_UNUSED(g);
    #include "generated/nominal_prediction_generated.inc"
}

void sf_ekf_error_transition(
    const sf_ekf_nominal_state_t *nominal,
    const sf_ekf_imu_delta_t *imu,
    sf_ekf_error_transition_t *out)
{
    sf_ekf_error_transition_with_gravity(nominal, imu, SF_EKF_GRAVITY_MSS, out);
}

void sf_ekf_error_transition_with_gravity(
    const sf_ekf_nominal_state_t *nominal,
    const sf_ekf_imu_delta_t *imu,
    float gravity_mss,
    sf_ekf_error_transition_t *out)
{
    const float q0 = nominal->q0;
    const float q1 = nominal->q1;
    const float q2 = nominal->q2;
    const float q3 = nominal->q3;
    const float vn = nominal->vn;
    const float ve = nominal->ve;
    const float vd = nominal->vd;
    const float pn = nominal->pn;
    const float pe = nominal->pe;
    const float pd = nominal->pd;
    const float bgx = nominal->bgx;
    const float bgy = nominal->bgy;
    const float bgz = nominal->bgz;
    const float bax = nominal->bax;
    const float bay = nominal->bay;
    const float baz = nominal->baz;
    const float q_bv0 = nominal->q_bv0;
    const float q_bv1 = nominal->q_bv1;
    const float q_bv2 = nominal->q_bv2;
    const float q_bv3 = nominal->q_bv3;
    const float dax = imu->dax;
    const float day = imu->day;
    const float daz = imu->daz;
    const float dvx = imu->dvx;
    const float dvy = imu->dvy;
    const float dvz = imu->dvz;
    const float dt = imu->dt;
    const float g = gravity_mss;
    float (*F)[SF_EKF_ERROR_STATES] = out->f;
    float (*G)[SF_EKF_NOISE_STATES] = out->g;
    unsigned int i;
    unsigned int j;
    SF_EKF_UNUSED(q0);
    SF_EKF_UNUSED(q1);
    SF_EKF_UNUSED(q2);
    SF_EKF_UNUSED(q3);
    SF_EKF_UNUSED(vn);
    SF_EKF_UNUSED(ve);
    SF_EKF_UNUSED(vd);
    SF_EKF_UNUSED(pn);
    SF_EKF_UNUSED(pe);
    SF_EKF_UNUSED(pd);
    SF_EKF_UNUSED(bgx);
    SF_EKF_UNUSED(bgy);
    SF_EKF_UNUSED(bgz);
    SF_EKF_UNUSED(bax);
    SF_EKF_UNUSED(bay);
    SF_EKF_UNUSED(baz);
    SF_EKF_UNUSED(q_bv0);
    SF_EKF_UNUSED(q_bv1);
    SF_EKF_UNUSED(q_bv2);
    SF_EKF_UNUSED(q_bv3);
    SF_EKF_UNUSED(dax);
    SF_EKF_UNUSED(day);
    SF_EKF_UNUSED(daz);
    SF_EKF_UNUSED(dvx);
    SF_EKF_UNUSED(dvy);
    SF_EKF_UNUSED(dvz);
    SF_EKF_UNUSED(dt);
    SF_EKF_UNUSED(g);
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        for (j = 0; j < SF_EKF_ERROR_STATES; ++j) {
            F[i][j] = 0.0F;
        }
        for (j = 0; j < SF_EKF_NOISE_STATES; ++j) {
            G[i][j] = 0.0F;
        }
    }
    #include "generated/error_transition_generated.inc"
    #include "generated/error_noise_input_generated.inc"
}

void sf_ekf_attitude_reset_jacobian(const float dtheta[3], float g_reset_theta[3][3])
{
    float (*G_reset_theta)[3] = g_reset_theta;
    const float dtheta_x = dtheta[0];
    const float dtheta_y = dtheta[1];
    const float dtheta_z = dtheta[2];
    unsigned int i;
    unsigned int j;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            G_reset_theta[i][j] = 0.0F;
        }
    }
    #include "generated/attitude_reset_jacobian_generated.inc"
}

void sf_ekf_gps_pos_n_observation(const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES], float r_pos_n, sf_ekf_scalar_observation_t *out)
{
    const float (*P)[SF_EKF_ERROR_STATES] = p;
    const float R_POS_N = r_pos_n;
    float *H = out->h;
    float *K = out->k;
    float S = 0.0F;
    unsigned int i;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        H[i] = 0.0F;
        K[i] = 0.0F;
    }
    #include "generated/gps_pos_n_generated.inc"
    out->s = S;
}

void sf_ekf_gps_pos_e_observation(const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES], float r_pos_e, sf_ekf_scalar_observation_t *out)
{
    const float (*P)[SF_EKF_ERROR_STATES] = p;
    const float R_POS_E = r_pos_e;
    float *H = out->h;
    float *K = out->k;
    float S = 0.0F;
    unsigned int i;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        H[i] = 0.0F;
        K[i] = 0.0F;
    }
    #include "generated/gps_pos_e_generated.inc"
    out->s = S;
}

void sf_ekf_gps_pos_d_observation(const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES], float r_pos_d, sf_ekf_scalar_observation_t *out)
{
    const float (*P)[SF_EKF_ERROR_STATES] = p;
    const float R_POS_D = r_pos_d;
    float *H = out->h;
    float *K = out->k;
    float S = 0.0F;
    unsigned int i;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        H[i] = 0.0F;
        K[i] = 0.0F;
    }
    #include "generated/gps_pos_d_generated.inc"
    out->s = S;
}

void sf_ekf_gps_vel_n_observation(const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES], float r_vel_n, sf_ekf_scalar_observation_t *out)
{
    const float (*P)[SF_EKF_ERROR_STATES] = p;
    const float R_VEL_N = r_vel_n;
    float *H = out->h;
    float *K = out->k;
    float S = 0.0F;
    unsigned int i;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        H[i] = 0.0F;
        K[i] = 0.0F;
    }
    #include "generated/gps_vel_n_generated.inc"
    out->s = S;
}

void sf_ekf_gps_vel_e_observation(const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES], float r_vel_e, sf_ekf_scalar_observation_t *out)
{
    const float (*P)[SF_EKF_ERROR_STATES] = p;
    const float R_VEL_E = r_vel_e;
    float *H = out->h;
    float *K = out->k;
    float S = 0.0F;
    unsigned int i;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        H[i] = 0.0F;
        K[i] = 0.0F;
    }
    #include "generated/gps_vel_e_generated.inc"
    out->s = S;
}

void sf_ekf_gps_vel_d_observation(const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES], float r_vel_d, sf_ekf_scalar_observation_t *out)
{
    const float (*P)[SF_EKF_ERROR_STATES] = p;
    const float R_VEL_D = r_vel_d;
    float *H = out->h;
    float *K = out->k;
    float S = 0.0F;
    unsigned int i;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        H[i] = 0.0F;
        K[i] = 0.0F;
    }
    #include "generated/gps_vel_d_generated.inc"
    out->s = S;
}

void sf_ekf_body_vel_x_observation(const sf_ekf_nominal_state_t *nominal, const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES], float r_body_vel, sf_ekf_scalar_observation_t *out)
{
    const float q0 = nominal->q0;
    const float q1 = nominal->q1;
    const float q2 = nominal->q2;
    const float q3 = nominal->q3;
    const float vn = nominal->vn;
    const float ve = nominal->ve;
    const float vd = nominal->vd;
    const float pn = nominal->pn;
    const float pe = nominal->pe;
    const float pd = nominal->pd;
    const float bgx = nominal->bgx;
    const float bgy = nominal->bgy;
    const float bgz = nominal->bgz;
    const float bax = nominal->bax;
    const float bay = nominal->bay;
    const float baz = nominal->baz;
    const float q_bv0 = nominal->q_bv0;
    const float q_bv1 = nominal->q_bv1;
    const float q_bv2 = nominal->q_bv2;
    const float q_bv3 = nominal->q_bv3;
    SF_EKF_UNUSED(q0);
    SF_EKF_UNUSED(q1);
    SF_EKF_UNUSED(q2);
    SF_EKF_UNUSED(q3);
    SF_EKF_UNUSED(vn);
    SF_EKF_UNUSED(ve);
    SF_EKF_UNUSED(vd);
    SF_EKF_UNUSED(pn);
    SF_EKF_UNUSED(pe);
    SF_EKF_UNUSED(pd);
    SF_EKF_UNUSED(bgx);
    SF_EKF_UNUSED(bgy);
    SF_EKF_UNUSED(bgz);
    SF_EKF_UNUSED(bax);
    SF_EKF_UNUSED(bay);
    SF_EKF_UNUSED(baz);
    SF_EKF_UNUSED(q_bv0);
    SF_EKF_UNUSED(q_bv1);
    SF_EKF_UNUSED(q_bv2);
    SF_EKF_UNUSED(q_bv3);
    const float (*P)[SF_EKF_ERROR_STATES] = p;
    const float R_BODY_VEL = r_body_vel;
    float *H = out->h;
    float *K = out->k;
    float S = 0.0F;
    unsigned int i;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        H[i] = 0.0F;
        K[i] = 0.0F;
    }
    #include "generated/body_vel_x_generated.inc"
    out->s = S;
}

void sf_ekf_body_vel_y_observation(const sf_ekf_nominal_state_t *nominal, const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES], float r_body_vel, sf_ekf_scalar_observation_t *out)
{
    const float q0 = nominal->q0;
    const float q1 = nominal->q1;
    const float q2 = nominal->q2;
    const float q3 = nominal->q3;
    const float vn = nominal->vn;
    const float ve = nominal->ve;
    const float vd = nominal->vd;
    const float pn = nominal->pn;
    const float pe = nominal->pe;
    const float pd = nominal->pd;
    const float bgx = nominal->bgx;
    const float bgy = nominal->bgy;
    const float bgz = nominal->bgz;
    const float bax = nominal->bax;
    const float bay = nominal->bay;
    const float baz = nominal->baz;
    const float q_bv0 = nominal->q_bv0;
    const float q_bv1 = nominal->q_bv1;
    const float q_bv2 = nominal->q_bv2;
    const float q_bv3 = nominal->q_bv3;
    SF_EKF_UNUSED(q0);
    SF_EKF_UNUSED(q1);
    SF_EKF_UNUSED(q2);
    SF_EKF_UNUSED(q3);
    SF_EKF_UNUSED(vn);
    SF_EKF_UNUSED(ve);
    SF_EKF_UNUSED(vd);
    SF_EKF_UNUSED(pn);
    SF_EKF_UNUSED(pe);
    SF_EKF_UNUSED(pd);
    SF_EKF_UNUSED(bgx);
    SF_EKF_UNUSED(bgy);
    SF_EKF_UNUSED(bgz);
    SF_EKF_UNUSED(bax);
    SF_EKF_UNUSED(bay);
    SF_EKF_UNUSED(baz);
    SF_EKF_UNUSED(q_bv0);
    SF_EKF_UNUSED(q_bv1);
    SF_EKF_UNUSED(q_bv2);
    SF_EKF_UNUSED(q_bv3);
    const float (*P)[SF_EKF_ERROR_STATES] = p;
    const float R_BODY_VEL = r_body_vel;
    float *H = out->h;
    float *K = out->k;
    float S = 0.0F;
    unsigned int i;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        H[i] = 0.0F;
        K[i] = 0.0F;
    }
    #include "generated/body_vel_y_generated.inc"
    out->s = S;
}

void sf_ekf_body_vel_z_observation(const sf_ekf_nominal_state_t *nominal, const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES], float r_body_vel, sf_ekf_scalar_observation_t *out)
{
    const float q0 = nominal->q0;
    const float q1 = nominal->q1;
    const float q2 = nominal->q2;
    const float q3 = nominal->q3;
    const float vn = nominal->vn;
    const float ve = nominal->ve;
    const float vd = nominal->vd;
    const float pn = nominal->pn;
    const float pe = nominal->pe;
    const float pd = nominal->pd;
    const float bgx = nominal->bgx;
    const float bgy = nominal->bgy;
    const float bgz = nominal->bgz;
    const float bax = nominal->bax;
    const float bay = nominal->bay;
    const float baz = nominal->baz;
    const float q_bv0 = nominal->q_bv0;
    const float q_bv1 = nominal->q_bv1;
    const float q_bv2 = nominal->q_bv2;
    const float q_bv3 = nominal->q_bv3;
    SF_EKF_UNUSED(q0);
    SF_EKF_UNUSED(q1);
    SF_EKF_UNUSED(q2);
    SF_EKF_UNUSED(q3);
    SF_EKF_UNUSED(vn);
    SF_EKF_UNUSED(ve);
    SF_EKF_UNUSED(vd);
    SF_EKF_UNUSED(pn);
    SF_EKF_UNUSED(pe);
    SF_EKF_UNUSED(pd);
    SF_EKF_UNUSED(bgx);
    SF_EKF_UNUSED(bgy);
    SF_EKF_UNUSED(bgz);
    SF_EKF_UNUSED(bax);
    SF_EKF_UNUSED(bay);
    SF_EKF_UNUSED(baz);
    SF_EKF_UNUSED(q_bv0);
    SF_EKF_UNUSED(q_bv1);
    SF_EKF_UNUSED(q_bv2);
    SF_EKF_UNUSED(q_bv3);
    const float (*P)[SF_EKF_ERROR_STATES] = p;
    const float R_BODY_VEL = r_body_vel;
    float *H = out->h;
    float *K = out->k;
    float S = 0.0F;
    unsigned int i;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        H[i] = 0.0F;
        K[i] = 0.0F;
    }
    #include "generated/body_vel_z_generated.inc"
    out->s = S;
}

void sf_ekf_vehicle_roll_prior_observation(const sf_ekf_nominal_state_t *nominal, const float p[SF_EKF_ERROR_STATES][SF_EKF_ERROR_STATES], float r_vehicle_roll, sf_ekf_scalar_observation_t *out)
{
    const float q0 = nominal->q0;
    const float q1 = nominal->q1;
    const float q2 = nominal->q2;
    const float q3 = nominal->q3;
    const float vn = nominal->vn;
    const float ve = nominal->ve;
    const float vd = nominal->vd;
    const float pn = nominal->pn;
    const float pe = nominal->pe;
    const float pd = nominal->pd;
    const float bgx = nominal->bgx;
    const float bgy = nominal->bgy;
    const float bgz = nominal->bgz;
    const float bax = nominal->bax;
    const float bay = nominal->bay;
    const float baz = nominal->baz;
    const float q_bv0 = nominal->q_bv0;
    const float q_bv1 = nominal->q_bv1;
    const float q_bv2 = nominal->q_bv2;
    const float q_bv3 = nominal->q_bv3;
    SF_EKF_UNUSED(q0);
    SF_EKF_UNUSED(q1);
    SF_EKF_UNUSED(q2);
    SF_EKF_UNUSED(q3);
    SF_EKF_UNUSED(vn);
    SF_EKF_UNUSED(ve);
    SF_EKF_UNUSED(vd);
    SF_EKF_UNUSED(pn);
    SF_EKF_UNUSED(pe);
    SF_EKF_UNUSED(pd);
    SF_EKF_UNUSED(bgx);
    SF_EKF_UNUSED(bgy);
    SF_EKF_UNUSED(bgz);
    SF_EKF_UNUSED(bax);
    SF_EKF_UNUSED(bay);
    SF_EKF_UNUSED(baz);
    SF_EKF_UNUSED(q_bv0);
    SF_EKF_UNUSED(q_bv1);
    SF_EKF_UNUSED(q_bv2);
    SF_EKF_UNUSED(q_bv3);
    const float (*P)[SF_EKF_ERROR_STATES] = p;
    const float R_VEHICLE_ROLL = r_vehicle_roll;
    float *H = out->h;
    float *K = out->k;
    float S = 0.0F;
    unsigned int i;
    for (i = 0; i < SF_EKF_ERROR_STATES; ++i) {
        H[i] = 0.0F;
        K[i] = 0.0F;
    }
    #include "generated/vehicle_roll_prior_generated.inc"
    out->s = S;
}

#undef SF_EKF_UNUSED
