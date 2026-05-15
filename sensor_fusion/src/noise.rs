//! Shared process-noise configuration for the EKF runtime.

/// Continuous process-noise variances used by EKF prediction.
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
    /// Mount-alignment random-walk variance.
    pub mount_align_rw_var: f32,
    /// Axis-specific mount-alignment random-walk variances.
    ///
    /// These values are interpreted as `[roll, pitch, yaw]` only when
    /// [`Self::mount_align_rw_var_axes_enabled`] is true. Otherwise
    /// [`Self::mount_align_rw_var`] is used for all three axes.
    #[cfg_attr(feature = "serde", serde(default))]
    pub mount_align_rw_var_axes: [f32; 3],
    /// Enables [`Self::mount_align_rw_var_axes`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub mount_align_rw_var_axes_enabled: bool,
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
            mount_align_rw_var: 0.0,
            mount_align_rw_var_axes: [0.0; 3],
            mount_align_rw_var_axes_enabled: false,
        }
    }

    /// Legacy broad covariance profile used by standalone EKF tests.
    pub const fn ekf_debug_default() -> Self {
        Self {
            gyro_var: 0.0001,
            accel_var: 12.0,
            gyro_bias_rw_var: 0.002e-9,
            accel_bias_rw_var: 0.2e-9,
            mount_align_rw_var: 0.0,
            mount_align_rw_var_axes: [0.0; 3],
            mount_align_rw_var_axes_enabled: false,
        }
    }

    /// Broad reference noise profile used by diagnostics and examples.
    pub const fn reference_nsr_demo() -> Self {
        Self {
            gyro_var: 2.5e-5,
            accel_var: 9.0e-4,
            gyro_bias_rw_var: 1.0e-12,
            accel_bias_rw_var: 1.0e-10,
            mount_align_rw_var: 0.0,
            mount_align_rw_var_axes: [0.0; 3],
            mount_align_rw_var_axes_enabled: false,
        }
    }

    /// Returns the mount random-walk variance for one mount error axis.
    pub fn mount_align_rw_var_axis(&self, axis: usize) -> f32 {
        if self.mount_align_rw_var_axes_enabled {
            self.mount_align_rw_var_axes[axis]
        } else {
            self.mount_align_rw_var
        }
    }

    /// Returns a copy with axis-specific mount random-walk variances enabled.
    pub fn with_mount_align_rw_var_axes(mut self, axes: [f32; 3]) -> Self {
        self.mount_align_rw_var_axes = axes;
        self.mount_align_rw_var_axes_enabled = true;
        self
    }
}
