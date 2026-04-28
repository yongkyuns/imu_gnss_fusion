use sensor_fusion::ekf::PredictNoise;
use sensor_fusion::loose::LoosePredictNoise;

#[derive(Clone, Copy, Debug)]
pub struct EkfCompareConfig {
    pub r_body_vel: f32,
    pub gnss_pos_mount_scale: f32,
    pub gnss_vel_mount_scale: f32,
    pub yaw_init_sigma_deg: f32,
    pub gyro_bias_init_sigma_dps: f32,
    pub accel_bias_init_sigma_mps2: f32,
    pub mount_init_sigma_deg: f32,
    pub r_vehicle_speed: f32,
    pub mount_align_rw_var: f32,
    pub mount_update_min_scale: f32,
    pub mount_update_ramp_time_s: f32,
    pub mount_update_innovation_gate_mps: f32,
    pub mount_update_yaw_rate_gate_dps: f32,
    pub align_handoff_delay_s: f32,
    pub freeze_misalignment_states: bool,
    pub mount_settle_time_s: f32,
    pub mount_settle_release_sigma_deg: f32,
    pub mount_settle_zero_cross_covariance: bool,
    pub r_zero_vel: f32,
    pub r_stationary_accel: f32,
    pub vehicle_meas_lpf_cutoff_hz: f64,
    pub predict_imu_lpf_cutoff_hz: Option<f64>,
    pub predict_imu_decimation: usize,
    pub yaw_init_speed_mps: f64,
    pub gnss_pos_r_scale: f64,
    pub gnss_vel_r_scale: f64,
    pub predict_noise: Option<PredictNoise>,
    pub loose_predict_noise: Option<LoosePredictNoise>,
}

impl Default for EkfCompareConfig {
    fn default() -> Self {
        Self {
            r_body_vel: 0.001,
            gnss_pos_mount_scale: 0.0,
            gnss_vel_mount_scale: 0.0,
            yaw_init_sigma_deg: 2.0,
            gyro_bias_init_sigma_dps: 0.125,
            accel_bias_init_sigma_mps2: 0.20,
            mount_init_sigma_deg: 2.5,
            r_vehicle_speed: 0.04,
            mount_align_rw_var: 1.0e-7,
            mount_update_min_scale: 0.008,
            mount_update_ramp_time_s: 120.0,
            mount_update_innovation_gate_mps: 0.10,
            mount_update_yaw_rate_gate_dps: 0.0,
            align_handoff_delay_s: 0.0,
            freeze_misalignment_states: false,
            mount_settle_time_s: 0.0,
            mount_settle_release_sigma_deg: 7.5,
            mount_settle_zero_cross_covariance: true,
            r_zero_vel: 0.0,
            r_stationary_accel: 0.0,
            vehicle_meas_lpf_cutoff_hz: 35.0,
            predict_imu_lpf_cutoff_hz: None,
            predict_imu_decimation: 1,
            yaw_init_speed_mps: 0.0 / 3.6,
            gnss_pos_r_scale: 0.05,
            gnss_vel_r_scale: 2.5,
            predict_noise: None,
            loose_predict_noise: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GnssOutageConfig {
    pub count: usize,
    pub duration_s: f64,
    pub seed: u64,
}
