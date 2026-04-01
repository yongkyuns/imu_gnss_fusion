#include "sf_usb_protocol.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sensor_fusion.h"
#include "sensor_fusion_internal.h"
#include "driver/usb_serial_jtag.h"

static const char *TAG = "sf_usb";

typedef struct {
  uint32_t imu_count;
  uint32_t gnss_count;
  uint64_t imu_total_us;
  uint64_t gnss_total_us;
  uint32_t imu_max_us;
  uint32_t gnss_max_us;
} sf_profile_t;

typedef struct {
  sf_sensor_fusion_t fusion;
  sf_fusion_config_t cfg;
  bool external_mode;
  float q_vb[4];
  uint8_t rx_buf[512];
  size_t rx_len;
  float last_t_s;
  sf_profile_t profile;
} sf_app_t;

static sf_app_t g_app;

static void sf_app_reset(sf_app_t *app);
static void sf_usb_init(void);
static void sf_process_rx(sf_app_t *app);
static bool sf_try_process_one_frame(sf_app_t *app);
static void sf_handle_config(sf_app_t *app, const sf_usb_config_payload_t *payload);
static void sf_handle_imu(sf_app_t *app, const sf_usb_imu_payload_t *payload);
static void sf_handle_gnss(sf_app_t *app, const sf_usb_gnss_payload_t *payload);
static void sf_send_status(sf_app_t *app, float t_s, sf_update_t update, bool end_ack);
static float sf_diag_sigma(const float p[SF_ALIGN_N_STATES][SF_ALIGN_N_STATES], int idx);
static uint32_t sf_now_us(void *ctx);
static float sf_avg_us(uint64_t total_us, uint32_t count);

void app_main(void) {
  memset(&g_app, 0, sizeof(g_app));
  sf_fusion_config_default(&g_app.cfg);
  sf_app_reset(&g_app);
  sf_usb_init();

  ESP_LOGI(TAG, "USB CDC replay ready");

  while (true) {
    sf_process_rx(&g_app);
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

static void sf_app_reset(sf_app_t *app) {
  if (app == NULL) {
    return;
  }
  memset(&app->profile, 0, sizeof(app->profile));
  app->last_t_s = 0.0f;
  if (app->external_mode) {
    sf_fusion_init_external(&app->fusion, &app->cfg, app->q_vb);
  } else {
    sf_fusion_init_internal(&app->fusion, &app->cfg);
  }
  sf_fusion_set_profile_now_us(&app->fusion, sf_now_us, NULL);
}

static void sf_usb_init(void) {
  usb_serial_jtag_driver_config_t cfg = {
      .tx_buffer_size = 1024,
      .rx_buffer_size = 1024,
  };
  ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));
}

static void sf_process_rx(sf_app_t *app) {
  int n;
  while (app->rx_len < sizeof(app->rx_buf)) {
    n = usb_serial_jtag_read_bytes(&app->rx_buf[app->rx_len],
                                   sizeof(app->rx_buf) - app->rx_len,
                                   0);
    if (n <= 0) {
      break;
    }
    app->rx_len += (size_t)n;
  }

  while (sf_try_process_one_frame(app)) {
  }
}

static bool sf_try_process_one_frame(sf_app_t *app) {
  sf_usb_frame_header_t hdr;
  size_t total_len;

  if (app->rx_len < sizeof(hdr)) {
    return false;
  }

  memcpy(&hdr, app->rx_buf, sizeof(hdr));
  if (hdr.magic_le != SF_USB_MAGIC || hdr.version != SF_USB_VERSION) {
    memmove(app->rx_buf, app->rx_buf + 1, app->rx_len - 1);
    app->rx_len -= 1;
    return app->rx_len >= sizeof(hdr);
  }

  total_len = sizeof(hdr) + (size_t)hdr.payload_len_le;
  if (app->rx_len < total_len) {
    return false;
  }

  switch (hdr.msg_type) {
    case SF_USB_MSG_CONFIG: {
      sf_usb_config_payload_t payload;
      if (hdr.payload_len_le == sizeof(payload)) {
        memcpy(&payload, app->rx_buf + sizeof(hdr), sizeof(payload));
        sf_handle_config(app, &payload);
      }
      break;
    }
    case SF_USB_MSG_RESET:
      sf_app_reset(app);
      break;
    case SF_USB_MSG_IMU: {
      sf_usb_imu_payload_t payload;
      if (hdr.payload_len_le == sizeof(payload)) {
        memcpy(&payload, app->rx_buf + sizeof(hdr), sizeof(payload));
        sf_handle_imu(app, &payload);
      }
      break;
    }
    case SF_USB_MSG_GNSS: {
      sf_usb_gnss_payload_t payload;
      if (hdr.payload_len_le == sizeof(payload)) {
        memcpy(&payload, app->rx_buf + sizeof(hdr), sizeof(payload));
        sf_handle_gnss(app, &payload);
      }
      break;
    }
    case SF_USB_MSG_END: {
      sf_update_t update = {0};
      update.mount_ready = sf_fusion_mount_ready(&app->fusion);
      update.mount_q_vb_valid = sf_fusion_mount_q_vb(&app->fusion, update.mount_q_vb);
      sf_send_status(app, app->last_t_s, update, true);
      break;
    }
    default:
      break;
  }

  memmove(app->rx_buf, app->rx_buf + total_len, app->rx_len - total_len);
  app->rx_len -= total_len;
  return app->rx_len > 0;
}

static void sf_handle_config(sf_app_t *app, const sf_usb_config_payload_t *payload) {
  if (app == NULL || payload == NULL) {
    return;
  }
  sf_fusion_config_default(&app->cfg);
  app->external_mode = payload->mode == SF_USB_MODE_EXTERNAL_QVB;
  memcpy(app->q_vb, payload->q_vb, sizeof(app->q_vb));
  if (isfinite(payload->r_body_vel) && payload->r_body_vel >= 0.0f) {
    app->cfg.r_body_vel = payload->r_body_vel;
  }
  sf_app_reset(app);
}

static void sf_handle_imu(sf_app_t *app, const sf_usb_imu_payload_t *payload) {
  sf_imu_sample_t sample;
  sf_update_t update;
  int64_t t0_us;
  uint32_t elapsed_us;

  if (app == NULL || payload == NULL) {
    return;
  }

  sample.t_s = payload->t_s;
  memcpy(sample.gyro_radps, payload->gyro_radps, sizeof(sample.gyro_radps));
  memcpy(sample.accel_mps2, payload->accel_mps2, sizeof(sample.accel_mps2));
  app->last_t_s = payload->t_s;
  t0_us = esp_timer_get_time();
  update = sf_fusion_process_imu(&app->fusion, &sample);
  elapsed_us = (uint32_t)(esp_timer_get_time() - t0_us);
  app->profile.imu_count += 1u;
  app->profile.imu_total_us += (uint64_t)elapsed_us;
  if (elapsed_us > app->profile.imu_max_us) {
    app->profile.imu_max_us = elapsed_us;
  }
  if (update.mount_ready_changed || update.ekf_initialized_now) {
    sf_send_status(app, payload->t_s, update, false);
  }
}

static void sf_handle_gnss(sf_app_t *app, const sf_usb_gnss_payload_t *payload) {
  sf_gnss_sample_t sample;
  sf_update_t update;
  int64_t t0_us;
  uint32_t elapsed_us;

  if (app == NULL || payload == NULL) {
    return;
  }

  sample.t_s = payload->t_s;
  memcpy(sample.pos_ned_m, payload->pos_ned_m, sizeof(sample.pos_ned_m));
  memcpy(sample.vel_ned_mps, payload->vel_ned_mps, sizeof(sample.vel_ned_mps));
  memcpy(sample.pos_std_m, payload->pos_std_m, sizeof(sample.pos_std_m));
  memcpy(sample.vel_std_mps, payload->vel_std_mps, sizeof(sample.vel_std_mps));
  sample.heading_valid = payload->heading_valid != 0;
  sample.heading_rad = payload->heading_rad;
  app->last_t_s = payload->t_s;
  t0_us = esp_timer_get_time();
  update = sf_fusion_process_gnss(&app->fusion, &sample);
  elapsed_us = (uint32_t)(esp_timer_get_time() - t0_us);
  app->profile.gnss_count += 1u;
  app->profile.gnss_total_us += (uint64_t)elapsed_us;
  if (elapsed_us > app->profile.gnss_max_us) {
    app->profile.gnss_max_us = elapsed_us;
  }
  sf_send_status(app, payload->t_s, update, false);
}

static void sf_send_status(sf_app_t *app, float t_s, sf_update_t update, bool end_ack) {
  sf_usb_frame_header_t hdr;
  sf_usb_status_payload_t payload;
  const sf_ekf_t *ekf;
  const sf_align_t *align;
  const sf_profile_counters_t *prof;

  int written;

  if (app == NULL) {
    return;
  }

  memset(&payload, 0, sizeof(payload));
  payload.t_s = t_s;
  if (update.mount_ready) {
    payload.flags |= SF_USB_FLAG_MOUNT_READY;
  }
  if (update.mount_ready_changed) {
    payload.flags |= SF_USB_FLAG_MOUNT_READY_CHANGED;
  }
  if (update.ekf_initialized) {
    payload.flags |= SF_USB_FLAG_EKF_INITIALIZED;
  }
  if (update.ekf_initialized_now) {
    payload.flags |= SF_USB_FLAG_EKF_INITIALIZED_NOW;
  }
  if (update.mount_q_vb_valid) {
    payload.flags |= SF_USB_FLAG_MOUNT_Q_VALID;
    memcpy(payload.mount_q_vb, update.mount_q_vb, sizeof(payload.mount_q_vb));
  }
  if (end_ack) {
    payload.flags |= SF_USB_FLAG_END_ACK;
  }

  ekf = sf_fusion_ekf(&app->fusion);
  align = sf_fusion_align(&app->fusion);
  prof = sf_fusion_profile(&app->fusion);
  if (ekf != NULL) {
    payload.ekf_q_bn[0] = ekf->state.q0;
    payload.ekf_q_bn[1] = ekf->state.q1;
    payload.ekf_q_bn[2] = ekf->state.q2;
    payload.ekf_q_bn[3] = ekf->state.q3;
    payload.ekf_vel_ned_mps[0] = ekf->state.vn;
    payload.ekf_vel_ned_mps[1] = ekf->state.ve;
    payload.ekf_vel_ned_mps[2] = ekf->state.vd;
    payload.ekf_pos_ned_m[0] = ekf->state.pn;
    payload.ekf_pos_ned_m[1] = ekf->state.pe;
    payload.ekf_pos_ned_m[2] = ekf->state.pd;
  }
  if (align != NULL) {
    payload.align_sigma_rad[0] = sf_diag_sigma(align->p, 0);
    payload.align_sigma_rad[1] = sf_diag_sigma(align->p, 1);
    payload.align_sigma_rad[2] = sf_diag_sigma(align->p, 2);
  }
  payload.imu_count = app->profile.imu_count;
  payload.gnss_count = app->profile.gnss_count;
  payload.imu_avg_us = app->profile.imu_count == 0u
                           ? 0.0f
                           : (float)app->profile.imu_total_us / (float)app->profile.imu_count;
  payload.imu_max_us = (float)app->profile.imu_max_us;
  payload.gnss_avg_us = app->profile.gnss_count == 0u
                            ? 0.0f
                            : (float)app->profile.gnss_total_us / (float)app->profile.gnss_count;
  payload.gnss_max_us = (float)app->profile.gnss_max_us;
  if (prof != NULL) {
    payload.imu_rotate_count = prof->imu_rotate_count;
    payload.imu_rotate_avg_us = sf_avg_us(prof->imu_rotate_total_us, prof->imu_rotate_count);
    payload.imu_rotate_max_us = (float)prof->imu_rotate_max_us;
    payload.imu_predict_count = prof->imu_predict_count;
    payload.imu_predict_avg_us = sf_avg_us(prof->imu_predict_total_us, prof->imu_predict_count);
    payload.imu_predict_max_us = (float)prof->imu_predict_max_us;
    payload.imu_clamp_count = prof->imu_clamp_count;
    payload.imu_clamp_avg_us = sf_avg_us(prof->imu_clamp_total_us, prof->imu_clamp_count);
    payload.imu_clamp_max_us = (float)prof->imu_clamp_max_us;
    payload.imu_body_vel_count = prof->imu_body_vel_count;
    payload.imu_body_vel_avg_us = sf_avg_us(prof->imu_body_vel_total_us, prof->imu_body_vel_count);
    payload.imu_body_vel_max_us = (float)prof->imu_body_vel_max_us;
    payload.gnss_align_count = prof->gnss_align_count;
    payload.gnss_align_avg_us = sf_avg_us(prof->gnss_align_total_us, prof->gnss_align_count);
    payload.gnss_align_max_us = (float)prof->gnss_align_max_us;
    payload.gnss_init_count = prof->gnss_init_count;
    payload.gnss_init_avg_us = sf_avg_us(prof->gnss_init_total_us, prof->gnss_init_count);
    payload.gnss_init_max_us = (float)prof->gnss_init_max_us;
    payload.gnss_fuse_count = prof->gnss_fuse_count;
    payload.gnss_fuse_avg_us = sf_avg_us(prof->gnss_fuse_total_us, prof->gnss_fuse_count);
    payload.gnss_fuse_max_us = (float)prof->gnss_fuse_max_us;
  }

  hdr.magic_le = SF_USB_MAGIC;
  hdr.version = SF_USB_VERSION;
  hdr.msg_type = SF_USB_MSG_STATUS;
  hdr.payload_len_le = sizeof(payload);
  hdr.reserved_le = 0;
  written = usb_serial_jtag_write_bytes((const char *)&hdr, sizeof(hdr), pdMS_TO_TICKS(10));
  if (written == (int)sizeof(hdr)) {
    (void)usb_serial_jtag_write_bytes((const char *)&payload, sizeof(payload), pdMS_TO_TICKS(10));
  }
}

static float sf_diag_sigma(const float p[SF_ALIGN_N_STATES][SF_ALIGN_N_STATES], int idx) {
  float v;
  if (p == NULL || idx < 0 || idx >= SF_ALIGN_N_STATES) {
    return 0.0f;
  }
  v = p[idx][idx];
  return v > 0.0f ? sqrtf(v) : 0.0f;
}

static uint32_t sf_now_us(void *ctx) {
  (void)ctx;
  return (uint32_t)esp_timer_get_time();
}

static float sf_avg_us(uint64_t total_us, uint32_t count) {
  return count == 0u ? 0.0f : (float)total_us / (float)count;
}
