#ifndef SENSOR_FUSION_FFI_H
#define SENSOR_FUSION_FFI_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SensorFusionFfi SensorFusionFfi;

#define SENSOR_FUSION_HARSH_BEHAVIOR_SENSITIVE 1u
#define SENSOR_FUSION_HARSH_BEHAVIOR_BALANCED 2u
#define SENSOR_FUSION_HARSH_BEHAVIOR_CONSERVATIVE 3u
#define SENSOR_FUSION_GNSS_EVENT_POSITION_REJECTED (1u << 0)
#define SENSOR_FUSION_GNSS_EVENT_VELOCITY_REJECTED (1u << 1)
#define SENSOR_FUSION_GNSS_EVENT_POSITION_CONSECUTIVE_REJECTED (1u << 2)
#define SENSOR_FUSION_GNSS_EVENT_VELOCITY_CONSECUTIVE_REJECTED (1u << 3)
#define SENSOR_FUSION_GNSS_EVENT_POSITION_GAP_BYPASS (1u << 4)
#define SENSOR_FUSION_GNSS_EVENT_VELOCITY_GAP_BYPASS (1u << 5)
#define SENSOR_FUSION_GNSS_EVENT_POSITION_ACCURACY_BYPASS (1u << 6)
#define SENSOR_FUSION_GNSS_EVENT_VELOCITY_ACCURACY_BYPASS (1u << 7)
#define SENSOR_FUSION_STATE_NOT_READY 0u
#define SENSOR_FUSION_STATE_INITIALIZING 1u
#define SENSOR_FUSION_STATE_RUNNING 2u
#define SENSOR_FUSION_STATE_STABLE 3u
#define SENSOR_FUSION_STATE_DEGRADED 4u
#define SENSOR_FUSION_STATE_DEGRADED_DEAD_RECKONING 5u
#define SENSOR_FUSION_STATE_AWAITING_GNSS_RESEED 6u

typedef struct SensorFusionFfiUpdate {
    uint32_t state;
    bool mount_ready;
    bool mount_ready_changed;
    bool navigation_usable;
    bool navigation_started;
    bool mount_q_bv_valid;
    float mount_q_bv[4];
    uint32_t gnss_event_mask;
} SensorFusionFfiUpdate;

typedef struct SensorFusionFfiHealth {
    uint32_t state;
    bool running;
    bool stable;
    bool degraded;
    bool navigation_usable;
    uint32_t reason_mask;
    float post_init_time_s;
    float distance_m;
    float mean_speed_mps;
    float tail_duration_s;
    uint32_t tail_samples;
    float mount_tail_drift_deg;
    float mount_tail_std_deg;
    float gyro_bias_tail_drift_radps;
    float gyro_bias_tail_std_radps;
    float accel_bias_tail_drift_mps2;
    float accel_bias_tail_std_mps2;
    float mount_sigma_max_deg;
    float attitude_sigma_max_deg;
    uint32_t recent_gnss_issue_count;
} SensorFusionFfiHealth;

typedef struct SensorFusionFfiEkfSnapshot {
    bool mount_ready;
    bool initialized;
    float q0;
    float q1;
    float q2;
    float q3;
    float vel_n_mps;
    float vel_e_mps;
    float vel_d_mps;
    float pos_n_m;
    float pos_e_m;
    float pos_d_m;
    float gyro_bias_x_radps;
    float gyro_bias_y_radps;
    float gyro_bias_z_radps;
    float accel_bias_x_mps2;
    float accel_bias_y_mps2;
    float accel_bias_z_mps2;
    float q_bv0;
    float q_bv1;
    float q_bv2;
    float q_bv3;
    bool position_lla_valid;
    double lat_deg;
    double lon_deg;
    double height_m;
} SensorFusionFfiEkfSnapshot;

typedef struct SensorFusionFfiAlignProgress {
    bool valid;
    bool coarse_ready;
    float roll_sigma_deg;
    float pitch_sigma_deg;
    float yaw_sigma_deg;
} SensorFusionFfiAlignProgress;

typedef struct SensorFusionFfiRoadEvent {
    uint32_t kind;
    float t_s;
    float start_t_s;
    float end_t_s;
    float duration_s;
    float value;
    float confidence;
} SensorFusionFfiRoadEvent;

typedef struct SensorFusionFfiTripSummary {
    uint32_t sample_count;
    uint32_t invalid_sample_count;
    uint32_t data_gap_count;
    float max_sample_gap_s;
    float total_gap_duration_s;
    float duration_s;
    float moving_duration_s;
    float stationary_duration_s;
    float distance_m;
    float reverse_duration_s;
    float reverse_distance_m;
    float uphill_distance_m;
    float downhill_distance_m;
    float elevation_gain_m;
    float elevation_loss_m;
    float mean_speed_mps;
    float moving_mean_speed_mps;
    float peak_speed_mps;
    float peak_accel_mps2;
    float peak_decel_mps2;
    float peak_lateral_accel_mps2;
    float rolling_speed_mps;
    float rolling_abs_longitudinal_accel_mps2;
    float rolling_abs_lateral_accel_mps2;
    uint32_t speed_bumps;
    uint32_t road_shocks;
    uint32_t rough_road_events;
    uint32_t uphill_events;
    uint32_t downhill_events;
    uint32_t reverse_events;
    uint32_t harsh_acceleration_events;
    uint32_t harsh_braking_events;
    uint32_t harsh_cornering_events;
    float speed_bumps_per_km;
    float road_shocks_per_km;
    float rough_road_events_per_km;
    float harsh_events_per_km;
    float reverse_seconds_per_km;
} SensorFusionFfiTripSummary;

SensorFusionFfi *sensor_fusion_create_ekf_auto(void);
SensorFusionFfi *sensor_fusion_create_ekf_manual(float qw, float qx, float qy, float qz);

void sensor_fusion_destroy(SensorFusionFfi *handle);

void sensor_fusion_reset_ekf_auto(SensorFusionFfi *handle);
void sensor_fusion_reset_ekf_manual(SensorFusionFfi *handle, float qw, float qx, float qy, float qz);

bool sensor_fusion_set_harsh_behavior_preset(SensorFusionFfi *handle, uint32_t preset);

SensorFusionFfiUpdate sensor_fusion_snapshot_status(const SensorFusionFfi *handle);

SensorFusionFfiHealth sensor_fusion_snapshot_health(const SensorFusionFfi *handle);

SensorFusionFfiUpdate sensor_fusion_process_imu(
    SensorFusionFfi *handle,
    float t_s,
    float ax,
    float ay,
    float az,
    float gx,
    float gy,
    float gz
);

SensorFusionFfiUpdate sensor_fusion_process_gnss(
    SensorFusionFfi *handle,
    float t_s,
    double lat_deg,
    double lon_deg,
    double height_m,
    float vn,
    float ve,
    float vd,
    float pos_std_n,
    float pos_std_e,
    float pos_std_d,
    float vel_std_n,
    float vel_std_e,
    float vel_std_d,
    float heading_rad,
    bool is_heading_valid
);

bool sensor_fusion_snapshot_ekf(
    const SensorFusionFfi *handle,
    SensorFusionFfiEkfSnapshot *out
);

bool sensor_fusion_snapshot_align_progress(
    const SensorFusionFfi *handle,
    SensorFusionFfiAlignProgress *out
);

uintptr_t sensor_fusion_process_road_event_motion(
    SensorFusionFfi *handle,
    float t_s,
    float forward_velocity_mps,
    float ground_speed_mps,
    float longitudinal_accel_mps2,
    bool longitudinal_accel_valid,
    float yaw_rate_radps,
    bool yaw_rate_valid,
    float pitch_deg,
    bool pitch_valid,
    float lateral_accel_mps2,
    bool lateral_accel_valid,
    float vertical_accel_mps2,
    bool vertical_accel_valid,
    SensorFusionFfiRoadEvent *out,
    uintptr_t max_events
);

bool sensor_fusion_snapshot_trip_summary(
    const SensorFusionFfi *handle,
    SensorFusionFfiTripSummary *out
);

#ifdef __cplusplus
}
#endif

#endif
