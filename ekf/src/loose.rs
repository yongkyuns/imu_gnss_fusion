#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LoosePredictNoise {
    pub gyro_var: f32,
    pub accel_var: f32,
    pub gyro_bias_rw_var: f32,
    pub accel_bias_rw_var: f32,
    pub gyro_scale_rw_var: f32,
    pub accel_scale_rw_var: f32,
    pub mount_align_rw_var: f32,
}

impl Default for LoosePredictNoise {
    fn default() -> Self {
        Self::lsm6dso_loose_104hz()
    }
}

impl LoosePredictNoise {
    pub const fn reference_nsr_demo() -> Self {
        Self {
            gyro_var: 2.5e-5,
            accel_var: 9.0e-4,
            gyro_bias_rw_var: 1.0e-12,
            accel_bias_rw_var: 1.0e-10,
            gyro_scale_rw_var: 1.0e-10,
            accel_scale_rw_var: 1.0e-10,
            mount_align_rw_var: 1.0e-8,
        }
    }

    pub const fn lsm6dso_loose_104hz() -> Self {
        Self {
            gyro_var: 2.287_311_3e-7 * 10.0_f32,
            accel_var: 2.450_421_4e-5 * 15.0_f32,
            gyro_bias_rw_var: 0.0002e-9,
            accel_bias_rw_var: 0.002e-9,
            gyro_scale_rw_var: 1.0e-10,
            accel_scale_rw_var: 1.0e-10,
            mount_align_rw_var: 1.0e-8,
        }
    }
}
