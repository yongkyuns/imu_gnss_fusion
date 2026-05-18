use sensor_fusion::align::AlignConfig;
use sensor_fusion::{ProcessNoise, SensorFusion};

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FusionTuningConfig {
    pub align: AlignConfig,
    #[serde(default = "default_r_body_vel_y")]
    pub r_body_vel: f32,
    #[serde(default = "default_r_body_vel_z")]
    pub r_body_vel_z: f32,
    #[serde(default)]
    pub nhc_update_period_s: f32,
    #[serde(default = "default_attitude_roll_pitch_init_sigma_deg")]
    pub attitude_roll_pitch_init_sigma_deg: f32,
    pub yaw_init_sigma_deg: f32,
    pub gyro_bias_init_sigma_dps: f32,
    pub accel_bias_init_sigma_mps2: f32,
    /// Initial mount roll/pitch sigma, in degrees.
    #[serde(default = "default_mount_roll_pitch_init_sigma_deg")]
    pub mount_roll_pitch_init_sigma_deg: f32,
    /// Initial mount roll sigma, in degrees.
    #[serde(default = "default_mount_roll_init_sigma_deg")]
    pub mount_roll_init_sigma_deg: f32,
    /// Initial mount pitch sigma, in degrees.
    #[serde(default = "default_mount_roll_pitch_init_sigma_deg")]
    pub mount_pitch_init_sigma_deg: f32,
    /// Initial mount yaw sigma, in degrees.
    pub mount_init_sigma_deg: f32,
    #[serde(default = "default_use_align_mount_covariance_on_seed")]
    pub use_align_mount_covariance_on_seed: bool,
    pub r_vehicle_speed: f32,
    pub mount_align_rw_var: f32,
    pub align_handoff_delay_s: f32,
    pub r_zero_vel: f32,
    pub r_stationary_accel: f32,
    pub vehicle_meas_lpf_cutoff_hz: f64,
    #[serde(default)]
    pub predict_imu_lpf_cutoff_hz: Option<f64>,
    pub predict_imu_decimation: usize,
    pub yaw_init_speed_mps: f64,
    #[serde(default)]
    pub noise: NoiseConfig,
}

impl Default for FusionTuningConfig {
    fn default() -> Self {
        Self {
            align: AlignConfig::default(),
            r_body_vel: default_r_body_vel_y(),
            r_body_vel_z: default_r_body_vel_z(),
            nhc_update_period_s: 0.1,
            attitude_roll_pitch_init_sigma_deg: default_attitude_roll_pitch_init_sigma_deg(),
            yaw_init_sigma_deg: 6.0,
            gyro_bias_init_sigma_dps: 0.125,
            accel_bias_init_sigma_mps2: 0.15,
            mount_roll_pitch_init_sigma_deg: default_mount_roll_pitch_init_sigma_deg(),
            mount_roll_init_sigma_deg: default_mount_roll_init_sigma_deg(),
            mount_pitch_init_sigma_deg: default_mount_roll_pitch_init_sigma_deg(),
            mount_init_sigma_deg: 6.0,
            use_align_mount_covariance_on_seed: default_use_align_mount_covariance_on_seed(),
            r_vehicle_speed: 0.04,
            mount_align_rw_var: 0.0,
            align_handoff_delay_s: 0.0,
            r_zero_vel: 0.0,
            r_stationary_accel: 0.0,
            vehicle_meas_lpf_cutoff_hz: 35.0,
            predict_imu_lpf_cutoff_hz: None,
            predict_imu_decimation: 1,
            yaw_init_speed_mps: 0.0 / 3.6,
            noise: NoiseConfig::default(),
        }
    }
}

fn default_r_body_vel_y() -> f32 {
    0.5
}

fn default_r_body_vel_z() -> f32 {
    0.5
}

fn default_mount_roll_pitch_init_sigma_deg() -> f32 {
    1.2
}

fn default_mount_roll_init_sigma_deg() -> f32 {
    1.7
}

fn default_attitude_roll_pitch_init_sigma_deg() -> f32 {
    2.0
}

fn default_use_align_mount_covariance_on_seed() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GnssOutageConfig {
    pub count: usize,
    pub duration_s: f64,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NoiseConfig {
    #[serde(default = "default_ekf_noise_option")]
    pub ekf: Option<ProcessNoise>,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            ekf: Some(default_ekf_noise()),
        }
    }
}

pub const fn default_ekf_noise() -> ProcessNoise {
    ProcessNoise {
        gyro_var: 2.287_311_3e-7 * 10.0_f32,
        accel_var: 2.450_421_4e-5 * 15.0_f32,
        gyro_bias_rw_var: 0.0002e-9,
        accel_bias_rw_var: 0.002e-9,
        mount_align_rw_var: 0.0,
        mount_align_rw_var_axes: [0.0; 3],
        mount_align_rw_var_axes_enabled: false,
    }
}

pub fn apply_fusion_tuning_config(fusion: &mut SensorFusion, cfg: FusionTuningConfig) {
    fusion.set_align_config(cfg.align);
    if let Some(noise) = cfg.noise.ekf {
        fusion.set_ekf_noise(noise);
    }
    fusion.set_r_body_vel_yz(cfg.r_body_vel, cfg.r_body_vel_z);
    fusion.set_nhc_update_period_s(cfg.nhc_update_period_s);
    fusion.set_attitude_roll_pitch_init_sigma_rad(
        cfg.attitude_roll_pitch_init_sigma_deg.to_radians(),
    );
    fusion.set_yaw_init_sigma_rad(cfg.yaw_init_sigma_deg.to_radians());
    fusion.set_gyro_bias_init_sigma_radps(cfg.gyro_bias_init_sigma_dps.to_radians());
    fusion.set_accel_bias_init_sigma_mps2(cfg.accel_bias_init_sigma_mps2);
    fusion.set_mount_roll_pitch_init_sigma_rad(cfg.mount_roll_pitch_init_sigma_deg.to_radians());
    fusion.set_mount_roll_init_sigma_rad(cfg.mount_roll_init_sigma_deg.to_radians());
    fusion.set_mount_pitch_init_sigma_rad(cfg.mount_pitch_init_sigma_deg.to_radians());
    fusion.set_mount_init_sigma_rad(cfg.mount_init_sigma_deg.to_radians());
    fusion.set_use_align_mount_covariance_on_seed(cfg.use_align_mount_covariance_on_seed);
    fusion.set_r_vehicle_speed(cfg.r_vehicle_speed);
    fusion.set_r_zero_vel(cfg.r_zero_vel);
    fusion.set_r_stationary_accel(cfg.r_stationary_accel);
    fusion.set_mount_align_rw_var(cfg.mount_align_rw_var);
    fusion.set_align_handoff_delay_s(cfg.align_handoff_delay_s);
}

fn default_ekf_noise_option() -> Option<ProcessNoise> {
    Some(default_ekf_noise())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_configs_round_trip_through_canonical_json() {
        let mut cfg = FusionTuningConfig {
            r_body_vel: 0.42,
            r_body_vel_z: 0.24,
            use_align_mount_covariance_on_seed: true,
            predict_imu_lpf_cutoff_hz: Some(120.0),
            ..Default::default()
        };
        cfg.align.min_speed_mps = 1.5;
        cfg.align.q_mount_std_rad = [1.0e-5, 2.0e-5, 3.0e-5];
        cfg.align.refine_after_coarse_ready = true;
        cfg.align.refine_process_noise_scale = 0.2;
        cfg.align.refine_observation_std_scale = 3.0;
        cfg.noise.ekf.as_mut().unwrap().mount_align_rw_var = 2.0e-8;

        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: FusionTuningConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.r_body_vel, cfg.r_body_vel);
        assert_eq!(decoded.r_body_vel_z, cfg.r_body_vel_z);
        assert_eq!(
            decoded.mount_roll_pitch_init_sigma_deg,
            cfg.mount_roll_pitch_init_sigma_deg
        );
        assert_eq!(
            decoded.use_align_mount_covariance_on_seed,
            cfg.use_align_mount_covariance_on_seed
        );
        assert_eq!(
            decoded.predict_imu_lpf_cutoff_hz,
            cfg.predict_imu_lpf_cutoff_hz
        );
        assert_eq!(decoded.align.min_speed_mps, cfg.align.min_speed_mps);
        assert_eq!(decoded.align.q_mount_std_rad, cfg.align.q_mount_std_rad);
        assert_eq!(
            decoded.align.refine_after_coarse_ready,
            cfg.align.refine_after_coarse_ready
        );
        assert_eq!(
            decoded.align.refine_process_noise_scale,
            cfg.align.refine_process_noise_scale
        );
        assert_eq!(
            decoded.align.refine_observation_std_scale,
            cfg.align.refine_observation_std_scale
        );
        assert_eq!(
            decoded.noise.ekf.unwrap().mount_align_rw_var,
            cfg.noise.ekf.unwrap().mount_align_rw_var
        );

        let outages = GnssOutageConfig {
            count: 2,
            duration_s: 12.5,
            seed: 9,
        };
        let json = serde_json::to_string(&outages).unwrap();
        let decoded: GnssOutageConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.count, outages.count);
        assert_eq!(decoded.duration_s, outages.duration_s);
        assert_eq!(decoded.seed, outages.seed);
    }

    #[test]
    fn replay_config_missing_optional_fields_uses_runtime_defaults() {
        let json = serde_json::json!({
            "align": AlignConfig::default(),
            "rBodyVel": 0.001,
            "yawInitSigmaDeg": 2.0,
            "gyroBiasInitSigmaDps": 0.125,
            "accelBiasInitSigmaMps2": 0.075,
            "mountInitSigmaDeg": 6.0,
            "rVehicleSpeed": 0.04,
            "mountAlignRwVar": 1.0e-7,
            "mountUpdateYawRateGateDps": 10.0,
            "alignHandoffDelayS": 0.0,
            "rZeroVel": 0.0,
            "rStationaryAccel": 0.0,
            "vehicleMeasLpfCutoffHz": 35.0,
            "predictImuDecimation": 1,
            "yawInitSpeedMps": 0.0,
        });

        let decoded: FusionTuningConfig = serde_json::from_value(json).unwrap();

        assert_eq!(decoded.predict_imu_lpf_cutoff_hz, None);
        assert_eq!(decoded.r_body_vel_z, default_r_body_vel_z());
        assert_eq!(
            decoded.mount_roll_pitch_init_sigma_deg,
            default_mount_roll_pitch_init_sigma_deg()
        );
        assert_eq!(
            decoded.mount_roll_init_sigma_deg,
            default_mount_roll_init_sigma_deg()
        );
        assert_eq!(
            decoded.mount_pitch_init_sigma_deg,
            default_mount_roll_pitch_init_sigma_deg()
        );
        assert!(decoded.use_align_mount_covariance_on_seed);
        assert!(decoded.noise.ekf.is_some());
    }
}
