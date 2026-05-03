#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
#[derive(Debug, Clone, Copy)]
pub struct PredictNoise {
    pub gyro_var: f32,
    pub accel_var: f32,
    pub gyro_bias_rw_var: f32,
    pub accel_bias_rw_var: f32,
    pub mount_align_rw_var: f32,
}

impl Default for PredictNoise {
    fn default() -> Self {
        Self {
            gyro_var: 0.0001,
            accel_var: 12.0,
            gyro_bias_rw_var: 0.002e-9,
            accel_bias_rw_var: 0.2e-9,
            mount_align_rw_var: 0.0,
        }
    }
}

impl PredictNoise {
    pub const fn lsm6dso_typical_104hz() -> Self {
        Self {
            gyro_var: 2.287_311_3e-7 * 10.0_f32,
            accel_var: 2.450_421_4e-5 * 15.0_f32,
            gyro_bias_rw_var: 0.0002e-9,
            accel_bias_rw_var: 0.002e-9,
            mount_align_rw_var: 0.0,
        }
    }
}
