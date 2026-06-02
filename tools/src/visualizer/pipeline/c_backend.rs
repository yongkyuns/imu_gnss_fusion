//! Native-only FFI adapter for the standalone C sensor-fusion implementation.

use crate::datasets::generic_replay::{GenericGnssSample, GenericImuSample};
use crate::visualizer::pipeline::FusionTuningConfig;

const C_SENSOR_FUSION_CONTEXT_STORAGE_SIZE: usize = 8192;
const C_SENSOR_FUSION_CONTEXT_STORAGE_ALIGN: usize = 16;

#[repr(C, align(16))]
struct CContextStorage {
    bytes: [u8; C_SENSOR_FUSION_CONTEXT_STORAGE_SIZE],
}

impl Default for CContextStorage {
    fn default() -> Self {
        Self {
            bytes: [0; C_SENSOR_FUSION_CONTEXT_STORAGE_SIZE],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CImuSample {
    pub t_s: f32,
    pub gyro_radps: [f32; 3],
    pub accel_mps2: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CGnssSample {
    pub t_s: f32,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
    pub vel_ned_mps: [f32; 3],
    pub pos_std_m: [f32; 3],
    pub vel_std_mps: [f32; 3],
    pub has_heading_rad: bool,
    pub heading_rad: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CUpdate {
    pub state: i32,
    pub mount_ready: bool,
    pub mount_ready_changed: bool,
    pub navigation_usable: bool,
    pub navigation_started: bool,
    pub has_mount_q_bv: bool,
    pub mount_q_bv: [f32; 4],
    pub gnss_event_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CEkfState {
    pub q_nv: [f32; 4],
    pub vel_ned_mps: [f32; 3],
    pub pos_ned_m: [f32; 3],
    pub gyro_bias_b_radps: [f32; 3],
    pub accel_bias_b_mps2: [f32; 3],
    pub q_bv: [f32; 4],
    pub covariance: [[f32; 18]; 18],
}

impl Default for CEkfState {
    fn default() -> Self {
        Self {
            q_nv: [0.0; 4],
            vel_ned_mps: [0.0; 3],
            pos_ned_m: [0.0; 3],
            gyro_bias_b_radps: [0.0; 3],
            accel_bias_b_mps2: [0.0; 3],
            q_bv: [0.0; 4],
            covariance: [[0.0; 18]; 18],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CAlignProgress {
    pub valid: bool,
    pub coarse_ready: bool,
    pub roll_sigma_deg: f32,
    pub pitch_sigma_deg: f32,
    pub yaw_sigma_deg: f32,
    pub progress: f32,
}

pub(crate) struct CSensorFusion {
    storage: CContextStorage,
}

impl CSensorFusion {
    pub(crate) fn new_auto() -> Self {
        assert_context_layout();
        let mut fusion = Self {
            storage: CContextStorage::default(),
        };
        // SAFETY: storage is aligned and sized for sensor_fusion_t; C only uses caller-owned storage.
        unsafe {
            sensor_fusion_init_auto(fusion.as_mut_ptr());
        }
        fusion
    }

    pub(crate) fn new_with_mount(q_bv: [f32; 4]) -> Self {
        assert_context_layout();
        let mut fusion = Self {
            storage: CContextStorage::default(),
        };
        // SAFETY: storage is aligned and sized for sensor_fusion_t; q_bv points to four finite f32 values.
        unsafe {
            sensor_fusion_init_with_mount(fusion.as_mut_ptr(), q_bv.as_ptr());
        }
        fusion
    }

    pub(crate) fn configure(&mut self, cfg: FusionTuningConfig) {
        // SAFETY: setter calls operate on a live C context and copy scalar values only.
        unsafe {
            sensor_fusion_set_r_body_vel_yz(self.as_mut_ptr(), cfg.r_body_vel, cfg.r_body_vel_z);
            sensor_fusion_set_r_vehicle_roll_prior(self.as_mut_ptr(), cfg.r_vehicle_roll_prior);
            sensor_fusion_set_r_vehicle_speed(self.as_mut_ptr(), cfg.r_vehicle_speed);
            sensor_fusion_set_yaw_init_sigma_rad(
                self.as_mut_ptr(),
                cfg.yaw_init_sigma_deg.to_radians(),
            );
            sensor_fusion_set_mount_init_sigma_rad(
                self.as_mut_ptr(),
                cfg.mount_init_sigma_deg.to_radians(),
            );
        }
    }

    pub(crate) fn process_imu(&mut self, sample: GenericImuSample) -> CUpdate {
        // SAFETY: sample is passed by value and self is a live C context.
        unsafe { sensor_fusion_process_imu(self.as_mut_ptr(), CImuSample::from(sample)) }
    }

    pub(crate) fn process_gnss(&mut self, sample: GenericGnssSample) -> CUpdate {
        // SAFETY: sample is passed by value and self is a live C context.
        unsafe { sensor_fusion_process_gnss(self.as_mut_ptr(), CGnssSample::from(sample)) }
    }

    pub(crate) fn ekf_state(&self) -> Option<CEkfState> {
        let mut out = CEkfState::default();
        // SAFETY: out is valid writable storage and self is a live C context.
        let ok = unsafe { sensor_fusion_ekf_state(self.as_ptr(), &mut out) };
        ok.then_some(out)
    }

    pub(crate) fn position_lla(&self) -> Option<[f64; 3]> {
        let mut out = [0.0_f64; 3];
        // SAFETY: out is valid writable storage and self is a live C context.
        let ok = unsafe { sensor_fusion_position_lla(self.as_ptr(), out.as_mut_ptr()) };
        ok.then_some(out)
    }

    pub(crate) fn align_progress(&self) -> CAlignProgress {
        // SAFETY: self is a live C context and the return struct is POD.
        unsafe { sensor_fusion_align_progress(self.as_ptr()) }
    }

    fn as_ptr(&self) -> *const CSensorFusionOpaque {
        self.storage.bytes.as_ptr().cast()
    }

    fn as_mut_ptr(&mut self) -> *mut CSensorFusionOpaque {
        self.storage.bytes.as_mut_ptr().cast()
    }
}

impl From<GenericImuSample> for CImuSample {
    fn from(sample: GenericImuSample) -> Self {
        Self {
            t_s: sample.t_s as f32,
            gyro_radps: sample.gyro_radps.map(|v| v as f32),
            accel_mps2: sample.accel_mps2.map(|v| v as f32),
        }
    }
}

impl From<GenericGnssSample> for CGnssSample {
    fn from(sample: GenericGnssSample) -> Self {
        Self {
            t_s: sample.t_s as f32,
            lat_deg: sample.lat_deg,
            lon_deg: sample.lon_deg,
            height_m: sample.height_m,
            vel_ned_mps: sample.vel_ned_mps.map(|v| v as f32),
            pos_std_m: sample.pos_std_m.map(|v| v as f32),
            vel_std_mps: sample.vel_std_mps.map(|v| v as f32),
            has_heading_rad: sample.heading_rad.is_some(),
            heading_rad: sample.heading_rad.unwrap_or(0.0) as f32,
        }
    }
}

fn assert_context_layout() {
    // SAFETY: pure C introspection helpers return constants.
    let size = unsafe { sensor_fusion_context_size() };
    let alignment = unsafe { sensor_fusion_context_alignment() };
    assert!(
        size <= C_SENSOR_FUSION_CONTEXT_STORAGE_SIZE,
        "C sensor_fusion_t size {size} exceeds Rust FFI storage"
    );
    assert!(
        alignment <= C_SENSOR_FUSION_CONTEXT_STORAGE_ALIGN,
        "C sensor_fusion_t alignment {alignment} exceeds Rust FFI storage alignment"
    );
}

#[repr(C)]
struct CSensorFusionOpaque {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn sensor_fusion_context_size() -> usize;
    fn sensor_fusion_context_alignment() -> usize;
    fn sensor_fusion_init_auto(fusion: *mut CSensorFusionOpaque);
    fn sensor_fusion_init_with_mount(fusion: *mut CSensorFusionOpaque, q_bv: *const f32);
    fn sensor_fusion_process_imu(fusion: *mut CSensorFusionOpaque, sample: CImuSample) -> CUpdate;
    fn sensor_fusion_process_gnss(fusion: *mut CSensorFusionOpaque, sample: CGnssSample)
    -> CUpdate;
    fn sensor_fusion_ekf_state(
        fusion: *const CSensorFusionOpaque,
        out_state: *mut CEkfState,
    ) -> bool;
    fn sensor_fusion_position_lla(fusion: *const CSensorFusionOpaque, out_lla: *mut f64) -> bool;
    fn sensor_fusion_align_progress(fusion: *const CSensorFusionOpaque) -> CAlignProgress;
    fn sensor_fusion_set_r_body_vel_yz(fusion: *mut CSensorFusionOpaque, r_y: f32, r_z: f32);
    fn sensor_fusion_set_r_vehicle_roll_prior(fusion: *mut CSensorFusionOpaque, r: f32);
    fn sensor_fusion_set_r_vehicle_speed(fusion: *mut CSensorFusionOpaque, r: f32);
    fn sensor_fusion_set_yaw_init_sigma_rad(fusion: *mut CSensorFusionOpaque, sigma_rad: f32);
    fn sensor_fusion_set_mount_init_sigma_rad(fusion: *mut CSensorFusionOpaque, sigma_rad: f32);
}
