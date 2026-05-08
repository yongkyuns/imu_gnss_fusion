//! Shared process-noise configuration for runtime filters.

/// Continuous process-noise variances used by Reduced and Full prediction.
///
/// Reduced uses gyro, accelerometer, bias, and mount random walks. Full uses
/// the same terms plus gyro/accelerometer scale random walks.
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
#[derive(Debug, Clone, Copy)]
pub struct ProcessNoise {
    /// Gyro white-noise variance.
    pub gyro_var: f32,
    /// Accelerometer white-noise variance.
    pub accel_var: f32,
    /// Gyro-bias random-walk variance.
    pub gyro_bias_rw_var: f32,
    /// Accelerometer-bias random-walk variance.
    pub accel_bias_rw_var: f32,
    /// Gyro-scale random-walk variance. Ignored by Reduced.
    pub gyro_scale_rw_var: f32,
    /// Accelerometer-scale random-walk variance. Ignored by Reduced.
    pub accel_scale_rw_var: f32,
    /// Mount-alignment random-walk variance.
    pub mount_align_rw_var: f32,
}

impl Default for ProcessNoise {
    fn default() -> Self {
        Self::lsm6dso_104hz()
    }
}

impl ProcessNoise {
    /// LSM6DSO-oriented process-noise profile for 104 Hz IMU data.
    pub const fn lsm6dso_104hz() -> Self {
        Self {
            gyro_var: 2.287_311_3e-7 * 10.0_f32,
            accel_var: 2.450_421_4e-5 * 15.0_f32,
            gyro_bias_rw_var: 0.0002e-9,
            accel_bias_rw_var: 0.002e-9,
            gyro_scale_rw_var: 1.0e-10,
            accel_scale_rw_var: 1.0e-10,
            mount_align_rw_var: 0.0,
        }
    }

    /// Legacy broad covariance profile used by standalone Reduced tests.
    pub const fn reduced_debug_default() -> Self {
        Self {
            gyro_var: 0.0001,
            accel_var: 12.0,
            gyro_bias_rw_var: 0.002e-9,
            accel_bias_rw_var: 0.2e-9,
            gyro_scale_rw_var: 0.0,
            accel_scale_rw_var: 0.0,
            mount_align_rw_var: 0.0,
        }
    }

    /// Reference noise profile used by original NSR-style full-filter demos.
    pub const fn reference_nsr_demo() -> Self {
        Self {
            gyro_var: 2.5e-5,
            accel_var: 9.0e-4,
            gyro_bias_rw_var: 1.0e-12,
            accel_bias_rw_var: 1.0e-10,
            gyro_scale_rw_var: 1.0e-10,
            accel_scale_rw_var: 1.0e-10,
            mount_align_rw_var: 0.0,
        }
    }
}
