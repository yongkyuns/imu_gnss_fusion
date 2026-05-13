#ifndef SENSOR_FUSION_FFI_H
#define SENSOR_FUSION_FFI_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SensorFusionFfi SensorFusionFfi;

typedef struct SensorFusionFfiUpdate {
    bool mount_ready;
    bool mount_ready_changed;
    bool reduced_initialized;
    bool reduced_initialized_now;
    bool filter_initialized;
    bool filter_initialized_now;
    bool mount_q_bv_valid;
    float mount_q_bv[4];
} SensorFusionFfiUpdate;

typedef struct SensorFusionFfiReducedSnapshot {
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
} SensorFusionFfiReducedSnapshot;

SensorFusionFfi *sensor_fusion_create_reduced_auto(void);
SensorFusionFfi *sensor_fusion_create_reduced_manual(float qw, float qx, float qy, float qz);

void sensor_fusion_destroy(SensorFusionFfi *handle);

void sensor_fusion_reset_reduced_auto(SensorFusionFfi *handle);
void sensor_fusion_reset_reduced_manual(SensorFusionFfi *handle, float qw, float qx, float qy, float qz);

SensorFusionFfiUpdate sensor_fusion_snapshot_status(const SensorFusionFfi *handle);

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

bool sensor_fusion_snapshot_reduced(
    const SensorFusionFfi *handle,
    SensorFusionFfiReducedSnapshot *out
);

#ifdef __cplusplus
}
#endif

#endif
