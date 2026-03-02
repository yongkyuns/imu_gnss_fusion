#ifndef EKF_H
#define EKF_H

#include <stdint.h>

#define N_STATES 16
#define GRAVITY_MSS 9.80665f

typedef struct{
    float q0, q1, q2, q3; // Quaternion components
    float vn, ve, vd;    // Velocity components in NED frame
    float pn, pe, pd;    // Position components in NED frame
    float dax_b, day_b, daz_b; // Accelerometer bias
    float dvx_b, dvy_b, dvz_b; // Gyroscope bias
} ekf_state_t;

typedef struct 
{
    float dax, day, daz; // Accelerometer measurements
    float dvx, dvy, dvz; // Gyroscope measurements
    float dt;            // Time step
} imu_sample_t;

typedef struct 
{
    ekf_state_t state;
    float P[N_STATES][N_STATES]; // State covariance matrix
} ekf_t;

typedef struct
{
  float pos_n; // North position measurement (m)
  float pos_e; // East position measurement (m)
  float pos_d; // Down position measurement (m)
  float vel_n; // North velocity measurement (m/s)
  float vel_e; // East velocity measurement (m/s)
  float vel_d; // Down velocity measurement (m/s)
  float heading_rad; // Heading measurement (radians)
  float R_POS_N; // Measurement noise variance for North position (m^2)
  float R_POS_E; // Measurement noise variance for East position (m^2)
  float R_POS_D; // Measurement noise variance for Down position (m^2)
  float R_VEL_N; // Measurement noise variance for North velocity (m^2/s^2) 
  float R_VEL_E; // Measurement noise variance for East velocity (m^2/s^2)
  float R_VEL_D; // Measurement noise variance for Down velocity (m^2/s^2)
  float R_YAW; // Measurement noise variance for heading (rad^2)
} gps_data_t;

typedef struct
{
    float dvb_x, dvb_y, dvb_z; // Delta velocity in body frame (m/s)
} ekf_debug_t;

/**
 * @brief Initialize EKF state and covariance matrix
 * 
 * @param ekf Pointer to EKF structure to initialize
 * @param P_init_val Initial value to set for all elements of the covariance matrix P
 */
void ekf_init(ekf_t *ekf, float P_init_val);

void ekf_predict(ekf_t * ekf, const imu_sample_t *imu, const float daVar, const float dvVar, const float dgb_p_noise_var, const float dvb_x_p_noise_var, const float dvb_y_p_noise_var, const float dvb_z_p_noise_var, ekf_debug_t * debug_out);

void ekf_fuse_gps(ekf_t *ekf, const gps_data_t *gps);

void ekf_fuse_body_vel(ekf_t *ekf, const float R_body_vel);

#endif // EKF_H