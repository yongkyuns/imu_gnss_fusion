use sensor_fusion::align::AlignConfig;
use sensor_fusion::ekf::PredictNoise;
use sensor_fusion::loose::LoosePredictNoise;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EkfCompareConfig {
    pub align: AlignConfig,
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
    pub align_handoff_delay_s: f32,
    pub freeze_misalignment_states: bool,
    pub mount_settle_time_s: f32,
    pub mount_settle_release_sigma_deg: f32,
    pub mount_settle_zero_cross_covariance: bool,
    pub r_zero_vel: f32,
    pub r_stationary_accel: f32,
    pub vehicle_meas_lpf_cutoff_hz: f64,
    #[serde(default)]
    pub predict_imu_lpf_cutoff_hz: Option<f64>,
    pub predict_imu_decimation: usize,
    pub yaw_init_speed_mps: f64,
    #[serde(default = "default_eskf_predict_noise_option")]
    pub predict_noise: Option<PredictNoise>,
    #[serde(default = "default_loose_predict_noise_option")]
    pub loose_predict_noise: Option<LoosePredictNoise>,
    pub loose_init: LooseInitConfig,
}

impl Default for EkfCompareConfig {
    fn default() -> Self {
        Self {
            align: AlignConfig::default(),
            r_body_vel: 0.005,
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
            align_handoff_delay_s: 0.0,
            freeze_misalignment_states: false,
            mount_settle_time_s: 0.0,
            mount_settle_release_sigma_deg: 7.5,
            mount_settle_zero_cross_covariance: true,
            r_zero_vel: 0.01,
            r_stationary_accel: 0.0,
            vehicle_meas_lpf_cutoff_hz: 35.0,
            predict_imu_lpf_cutoff_hz: None,
            predict_imu_decimation: 1,
            yaw_init_speed_mps: 0.0 / 3.6,
            predict_noise: Some(default_eskf_predict_noise()),
            loose_predict_noise: Some(default_loose_predict_noise()),
            loose_init: LooseInitConfig::default(),
        }
    }
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
pub struct LooseInitConfig {
    pub pos_min_sigma_m: f32,
    pub vel_min_sigma_mps: f32,
    pub attitude_sigma_deg: f32,
    pub gyro_bias_sigma_dps: f32,
    pub accel_bias_sigma_mps2: f32,
    pub gyro_scale_sigma: f32,
    pub accel_scale_sigma: f32,
    /// Initial residual mount roll/pitch sigma, in degrees.
    pub mount_sigma_deg: f32,
    /// Initial residual mount yaw sigma, in degrees.
    #[serde(default = "default_loose_mount_yaw_sigma_deg")]
    pub mount_yaw_sigma_deg: f32,
}

impl Default for LooseInitConfig {
    fn default() -> Self {
        Self {
            pos_min_sigma_m: 0.5,
            vel_min_sigma_mps: 0.2,
            attitude_sigma_deg: 2.0,
            gyro_bias_sigma_dps: 0.125,
            accel_bias_sigma_mps2: 0.075,
            gyro_scale_sigma: 0.02,
            accel_scale_sigma: 0.0,
            mount_sigma_deg: 2.0,
            mount_yaw_sigma_deg: default_loose_mount_yaw_sigma_deg(),
        }
    }
}

fn default_loose_mount_yaw_sigma_deg() -> f32 {
    6.0
}

pub const fn default_eskf_predict_noise() -> PredictNoise {
    PredictNoise {
        gyro_var: 2.287_311_3e-7 * 10.0_f32,
        accel_var: 2.450_421_4e-5 * 15.0_f32,
        gyro_bias_rw_var: 0.0002e-9,
        accel_bias_rw_var: 0.002e-9,
        mount_align_rw_var: 1.0e-7,
    }
}

pub const fn default_loose_predict_noise() -> LoosePredictNoise {
    LoosePredictNoise::lsm6dso_loose_104hz()
}

fn default_eskf_predict_noise_option() -> Option<PredictNoise> {
    Some(default_eskf_predict_noise())
}

fn default_loose_predict_noise_option() -> Option<LoosePredictNoise> {
    Some(default_loose_predict_noise())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_configs_round_trip_through_canonical_json() {
        let mut cfg = EkfCompareConfig {
            r_body_vel: 0.42,
            predict_imu_lpf_cutoff_hz: Some(120.0),
            ..Default::default()
        };
        cfg.align.min_speed_mps = 1.5;
        cfg.align.q_mount_std_rad = [1.0e-5, 2.0e-5, 3.0e-5];
        cfg.loose_init.mount_sigma_deg = 4.0;
        cfg.loose_init.mount_yaw_sigma_deg = 8.0;
        cfg.loose_predict_noise.as_mut().unwrap().mount_align_rw_var = 2.0e-8;

        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: EkfCompareConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.r_body_vel, cfg.r_body_vel);
        assert_eq!(
            decoded.predict_imu_lpf_cutoff_hz,
            cfg.predict_imu_lpf_cutoff_hz
        );
        assert_eq!(decoded.align.min_speed_mps, cfg.align.min_speed_mps);
        assert_eq!(decoded.align.q_mount_std_rad, cfg.align.q_mount_std_rad);
        assert_eq!(
            decoded.loose_init.mount_sigma_deg,
            cfg.loose_init.mount_sigma_deg
        );
        assert_eq!(
            decoded.loose_init.mount_yaw_sigma_deg,
            cfg.loose_init.mount_yaw_sigma_deg
        );
        assert_eq!(
            decoded.loose_predict_noise.unwrap().mount_align_rw_var,
            cfg.loose_predict_noise.unwrap().mount_align_rw_var
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
            "gnssPosMountScale": 0.0,
            "gnssVelMountScale": 0.0,
            "yawInitSigmaDeg": 2.0,
            "gyroBiasInitSigmaDps": 0.125,
            "accelBiasInitSigmaMps2": 0.20,
            "mountInitSigmaDeg": 2.5,
            "rVehicleSpeed": 0.04,
            "mountAlignRwVar": 1.0e-7,
            "mountUpdateMinScale": 0.008,
            "mountUpdateRampTimeS": 120.0,
            "mountUpdateInnovationGateMps": 0.10,
            "mountUpdateYawRateGateDps": 10.0,
            "alignHandoffDelayS": 0.0,
            "freezeMisalignmentStates": false,
            "mountSettleTimeS": 0.0,
            "mountSettleReleaseSigmaDeg": 7.5,
            "mountSettleZeroCrossCovariance": true,
            "rZeroVel": 0.01,
            "rStationaryAccel": 0.0,
            "vehicleMeasLpfCutoffHz": 35.0,
            "predictImuDecimation": 1,
            "yawInitSpeedMps": 0.0,
            "looseInit": {
                "posMinSigmaM": 0.5,
                "velMinSigmaMps": 0.2,
                "attitudeSigmaDeg": 2.0,
                "gyroBiasSigmaDps": 0.125,
                "accelBiasSigmaMps2": 0.075,
                "gyroScaleSigma": 0.02,
                "accelScaleSigma": 0.0,
                "mountSigmaDeg": 2.0
            },
        });

        let decoded: EkfCompareConfig = serde_json::from_value(json).unwrap();

        assert_eq!(decoded.predict_imu_lpf_cutoff_hz, None);
        assert_eq!(
            decoded.loose_init.mount_yaw_sigma_deg,
            default_loose_mount_yaw_sigma_deg()
        );
        assert!(decoded.predict_noise.is_some());
        assert!(decoded.loose_predict_noise.is_some());
    }
}
