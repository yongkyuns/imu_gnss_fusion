#ifndef SF_USB_PROTOCOL_H
#define SF_USB_PROTOCOL_H

#include <stdint.h>

#define SF_USB_MAGIC 0x5346u
#define SF_USB_VERSION 1u

enum {
  SF_USB_MSG_CONFIG = 0x01,
  SF_USB_MSG_RESET = 0x02,
  SF_USB_MSG_IMU = 0x10,
  SF_USB_MSG_GNSS = 0x11,
  SF_USB_MSG_END = 0x12,
  SF_USB_MSG_STATUS = 0x81,
};

enum {
  SF_USB_MODE_INTERNAL_ALIGN = 0,
  SF_USB_MODE_EXTERNAL_QVB = 1,
};

enum {
  SF_USB_FLAG_MOUNT_READY = 1u << 0,
  SF_USB_FLAG_MOUNT_READY_CHANGED = 1u << 1,
  SF_USB_FLAG_EKF_INITIALIZED = 1u << 2,
  SF_USB_FLAG_EKF_INITIALIZED_NOW = 1u << 3,
  SF_USB_FLAG_MOUNT_Q_VALID = 1u << 4,
  SF_USB_FLAG_END_ACK = 1u << 5,
};

typedef struct __attribute__((packed)) {
  uint16_t magic_le;
  uint8_t version;
  uint8_t msg_type;
  uint16_t payload_len_le;
  uint16_t reserved_le;
} sf_usb_frame_header_t;

typedef struct __attribute__((packed)) {
  uint8_t mode;
  uint8_t reserved0;
  uint8_t reserved1;
  uint8_t reserved2;
  float q_vb[4];
} sf_usb_config_payload_t;

typedef struct __attribute__((packed)) {
  float t_s;
  float gyro_radps[3];
  float accel_mps2[3];
} sf_usb_imu_payload_t;

typedef struct __attribute__((packed)) {
  float t_s;
  float pos_ned_m[3];
  float vel_ned_mps[3];
  float pos_std_m[3];
  float vel_std_mps[3];
  uint8_t heading_valid;
  uint8_t reserved0;
  uint8_t reserved1;
  uint8_t reserved2;
  float heading_rad;
} sf_usb_gnss_payload_t;

typedef struct __attribute__((packed)) {
  float t_s;
  uint32_t flags;
  float mount_q_vb[4];
  float ekf_q_bn[4];
  float ekf_vel_ned_mps[3];
  float ekf_pos_ned_m[3];
  float align_sigma_rad[3];
  uint32_t imu_count;
  uint32_t gnss_count;
  float imu_avg_us;
  float imu_max_us;
  float gnss_avg_us;
  float gnss_max_us;
} sf_usb_status_payload_t;

#endif
