#ifndef SENSOR_FUSION_H
#define SENSOR_FUSION_H

/**
 * @file sensor_fusion.h
 * @brief C99 public facade for the IMU/GNSS ground-vehicle fusion runtime.
 *
 * This API mirrors the Rust `sensor_fusion::SensorFusion` facade. The C
 * implementation is caller-owned and allocation-free: initialize a
 * `sensor_fusion_t` object, feed timestamp-ordered IMU/GNSS/vehicle-speed
 * samples, then query update status, health, mount, position, and EKF state.
 *
 * Frame conventions match the Rust crate:
 * - body frame `b`: raw IMU sensor axes.
 * - vehicle frame `v`: forward-right-down.
 * - navigation frame `n`: local NED.
 * - `q_bv` maps vehicle coordinates to body coordinates, `x_b = C_bv x_v`.
 * - `q_nv` maps vehicle coordinates to navigation coordinates, `x_n = C_nv x_v`.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SENSOR_FUSION_ERROR_STATES 18u
#define SENSOR_FUSION_NOMINAL_STATE_SIZE 21u
#define SENSOR_FUSION_PRIVATE_STORAGE_SIZE 4096u

typedef enum {
    SENSOR_FUSION_STATE_NOT_READY = 0,
    SENSOR_FUSION_STATE_INITIALIZING = 1,
    SENSOR_FUSION_STATE_RUNNING = 2,
    SENSOR_FUSION_STATE_STABLE = 3,
    SENSOR_FUSION_STATE_DEGRADED = 4,
    SENSOR_FUSION_STATE_DEGRADED_DEAD_RECKONING = 5,
    SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED = 6,
} sensor_fusion_state_t;

enum {
    SENSOR_FUSION_HEALTH_REASON_NOT_INITIALIZED = 1u << 0,
    SENSOR_FUSION_HEALTH_REASON_MOUNT_NOT_READY = 1u << 1,
    SENSOR_FUSION_HEALTH_REASON_GNSS_STALE = 1u << 2,
    SENSOR_FUSION_HEALTH_REASON_INSUFFICIENT_TIME = 1u << 3,
    SENSOR_FUSION_HEALTH_REASON_INSUFFICIENT_MOTION = 1u << 4,
    SENSOR_FUSION_HEALTH_REASON_TAIL_TOO_SHORT = 1u << 5,
    SENSOR_FUSION_HEALTH_REASON_MOUNT_UNSTABLE = 1u << 6,
    SENSOR_FUSION_HEALTH_REASON_BIAS_UNSTABLE = 1u << 7,
    SENSOR_FUSION_HEALTH_REASON_COVARIANCE_HIGH = 1u << 8,
    SENSOR_FUSION_HEALTH_REASON_GNSS_REJECTING = 1u << 9,
    SENSOR_FUSION_HEALTH_REASON_NUMERIC_INVALID = 1u << 10,
    SENSOR_FUSION_HEALTH_REASON_SLEEP_GAP = 1u << 11,
    SENSOR_FUSION_HEALTH_REASON_NAV_UNUSABLE = 1u << 12,
};

enum {
    SENSOR_FUSION_GNSS_EVENT_POSITION_REJECTED = 1u << 0,
    SENSOR_FUSION_GNSS_EVENT_VELOCITY_REJECTED = 1u << 1,
    SENSOR_FUSION_GNSS_EVENT_POSITION_CONSECUTIVE_REJECTED = 1u << 2,
    SENSOR_FUSION_GNSS_EVENT_VELOCITY_CONSECUTIVE_REJECTED = 1u << 3,
    SENSOR_FUSION_GNSS_EVENT_POSITION_GAP_BYPASS = 1u << 4,
    SENSOR_FUSION_GNSS_EVENT_VELOCITY_GAP_BYPASS = 1u << 5,
    SENSOR_FUSION_GNSS_EVENT_POSITION_ACCURACY_BYPASS = 1u << 6,
    SENSOR_FUSION_GNSS_EVENT_VELOCITY_ACCURACY_BYPASS = 1u << 7,
};

typedef enum {
    SENSOR_FUSION_VEHICLE_SPEED_UNKNOWN = 0,
    SENSOR_FUSION_VEHICLE_SPEED_FORWARD = 1,
    SENSOR_FUSION_VEHICLE_SPEED_REVERSE = 2,
} sensor_fusion_vehicle_speed_direction_t;

typedef struct {
    float t_s;
    float gyro_radps[3];
    float accel_mps2[3];
} sensor_fusion_imu_sample_t;

typedef struct {
    float t_s;
    double lat_deg;
    double lon_deg;
    double height_m;
    float vel_ned_mps[3];
    float pos_std_m[3];
    float vel_std_mps[3];
    bool has_heading_rad;
    float heading_rad;
} sensor_fusion_gnss_sample_t;

typedef struct {
    float t_s;
    float speed_mps;
    sensor_fusion_vehicle_speed_direction_t direction;
} sensor_fusion_vehicle_speed_sample_t;

typedef struct {
    sensor_fusion_state_t state;
    bool mount_ready;
    bool mount_ready_changed;
    bool navigation_usable;
    bool navigation_started;
    bool has_mount_q_bv;
    float mount_q_bv[4];
    uint32_t gnss_event_mask;
} sensor_fusion_update_t;

typedef struct {
    sensor_fusion_state_t state;
    bool usable;
    bool stable;
    bool degraded;
    uint32_t reason_mask;
} sensor_fusion_health_t;

typedef struct {
    float q_nv[4];
    float vel_ned_mps[3];
    float pos_ned_m[3];
    float gyro_bias_b_radps[3];
    float accel_bias_b_mps2[3];
    float q_bv[4];
    float covariance[SENSOR_FUSION_ERROR_STATES][SENSOR_FUSION_ERROR_STATES];
} sensor_fusion_ekf_state_t;

typedef struct {
    bool valid;
    bool coarse_ready;
    float roll_sigma_deg;
    float pitch_sigma_deg;
    float yaw_sigma_deg;
    float progress;
} sensor_fusion_align_progress_t;

typedef struct {
    bool manual_mount;
    float manual_q_bv[4];
} sensor_fusion_config_t;

/** Opaque internal storage used by the C implementation. */
typedef union {
    double align;
    uint8_t bytes[SENSOR_FUSION_PRIVATE_STORAGE_SIZE];
} sensor_fusion_private_storage_t;

typedef struct {
    sensor_fusion_config_t cfg;
    bool mount_ready;
    bool navigation_usable;
    bool ekf_initialized;
    bool expected_sleep;
    float mount_q_bv[4];
    sensor_fusion_ekf_state_t ekf;
    double anchor_lat_deg;
    double anchor_lon_deg;
    double anchor_height_m;
    bool has_anchor;
    float last_imu_t_s;
    bool has_last_imu;
    float last_gnss_t_s;
    bool has_last_gnss;
    uint32_t reanchor_count;
    sensor_fusion_private_storage_t private_storage;
} sensor_fusion_t;

/** Return default automatic-mount configuration. */
sensor_fusion_config_t sensor_fusion_config_default(void);

/** Return `sizeof(sensor_fusion_t)` for foreign-language opaque storage. */
size_t sensor_fusion_context_size(void);

/** Return the byte alignment required by `sensor_fusion_t`. */
size_t sensor_fusion_context_alignment(void);

/** Initialize a caller-owned fusion context from an explicit configuration. */
void sensor_fusion_init(sensor_fusion_t *fusion, sensor_fusion_config_t cfg);

/** Initialize a caller-owned fusion context in automatic mount-alignment mode. */
void sensor_fusion_init_auto(sensor_fusion_t *fusion);

/** Initialize a caller-owned fusion context with a known physical mount `q_bv`. */
void sensor_fusion_init_with_mount(sensor_fusion_t *fusion, const float q_bv[4]);

/** Set or replace the physical vehicle-to-body mount quaternion `q_bv`. */
void sensor_fusion_set_misalignment(sensor_fusion_t *fusion, const float q_bv[4]);

/**
 * Declare an expected trip boundary before the host stops feeding samples.
 *
 * The next IMU timestamp gap is then interpreted as stationary sleep rather
 * than an in-trip stream dropout. Short expected gaps age the covariance and
 * keep navigation usable. Long expected gaps enter
 * `SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED` until a fresh GNSS sample reseeds
 * the navigation state. Unexpected large IMU gaps are treated conservatively
 * as stream loss and also require GNSS reseed.
 */
sensor_fusion_update_t sensor_fusion_end_trip(sensor_fusion_t *fusion);

/** Process one body-frame IMU sample. Samples must be timestamp ordered. */
sensor_fusion_update_t sensor_fusion_process_imu(sensor_fusion_t *fusion,
                                                 sensor_fusion_imu_sample_t sample);

/** Process one local-NED GNSS position/velocity sample. */
sensor_fusion_update_t sensor_fusion_process_gnss(sensor_fusion_t *fusion,
                                                  sensor_fusion_gnss_sample_t sample);

/** Process one optional signed vehicle-speed observation. */
sensor_fusion_update_t
sensor_fusion_process_vehicle_speed(sensor_fusion_t *fusion,
                                    sensor_fusion_vehicle_speed_sample_t sample);

/** Return the current consolidated fusion health state. */
sensor_fusion_health_t sensor_fusion_health(const sensor_fusion_t *fusion);

/** Copy the current physical mount quaternion `q_bv` when available. */
bool sensor_fusion_mount_q_bv(const sensor_fusion_t *fusion, float out_q_bv[4]);

/** Copy a snapshot of the EKF nominal state and covariance when initialized. */
bool sensor_fusion_ekf_state(const sensor_fusion_t *fusion, sensor_fusion_ekf_state_t *out_state);

/** Copy the current geodetic position when navigation is initialized and usable. */
bool sensor_fusion_position_lla(const sensor_fusion_t *fusion, double out_lla[3]);

/** Return mount-alignment progress fields used by host diagnostics and UI. */
sensor_fusion_align_progress_t sensor_fusion_align_progress(const sensor_fusion_t *fusion);

/** Set lateral and vertical NHC measurement variances. */
void sensor_fusion_set_r_body_vel_yz(sensor_fusion_t *fusion, float r_y, float r_z);

/** Set the flat-road vehicle-roll-prior measurement variance. */
void sensor_fusion_set_r_vehicle_roll_prior(sensor_fusion_t *fusion, float r);

/** Set the vehicle-frame X speed measurement variance. */
void sensor_fusion_set_r_vehicle_speed(sensor_fusion_t *fusion, float r);

/** Set the decimated NHC update period in seconds. */
void sensor_fusion_set_nhc_update_period_s(sensor_fusion_t *fusion, float period_s);

/** Set initial vehicle yaw standard deviation in radians. */
void sensor_fusion_set_yaw_init_sigma_rad(sensor_fusion_t *fusion, float sigma_rad);

/** Set initial mount-angle standard deviation in radians. */
void sensor_fusion_set_mount_init_sigma_rad(sensor_fusion_t *fusion, float sigma_rad);

#ifdef __cplusplus
}
#endif

#endif
