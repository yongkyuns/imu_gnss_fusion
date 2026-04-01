#![allow(non_snake_case)]

use crate::align::{AlignConfig, AlignWindowSummary};
use crate::ekf::{Ekf, PredictNoise};
use crate::fusion::{FusionConfig, FusionGnssSample, FusionImuSample, FusionUpdate};

pub const SF_SENSOR_FUSION_STORAGE_BYTES: usize = 32768;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CSensorFusion {
    pub storage: [u8; SF_SENSOR_FUSION_STORAGE_BYTES],
}

impl Default for CSensorFusion {
    fn default() -> Self {
        Self {
            storage: [0; SF_SENSOR_FUSION_STORAGE_BYTES],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CAlignState {
    pub q_vb: [f32; 4],
    pub p: [[f32; 3]; 3],
    pub gravity_lp_b: [f32; 3],
    pub coarse_alignment_ready: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CAlignConfig {
    pub q_mount_std_rad: [f32; 3],
    pub r_gravity_std_mps2: f32,
    pub r_horiz_heading_std_rad: f32,
    pub r_turn_gyro_std_radps: f32,
    pub turn_gyro_yaw_scale: f32,
    pub r_turn_heading_std_rad: f32,
    pub gravity_lpf_alpha: f32,
    pub min_speed_mps: f32,
    pub min_turn_rate_radps: f32,
    pub min_lat_acc_mps2: f32,
    pub min_long_acc_mps2: f32,
    pub turn_consistency_min_windows: u32,
    pub turn_consistency_min_fraction: f32,
    pub turn_consistency_max_abs_lat_err_mps2: f32,
    pub turn_consistency_max_rel_lat_err: f32,
    pub max_stationary_gyro_radps: f32,
    pub max_stationary_accel_norm_err_mps2: f32,
    pub use_gravity: bool,
    pub use_turn_gyro: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CBootstrapConfig {
    pub ema_alpha: f32,
    pub max_speed_mps: f32,
    pub stationary_samples: u32,
    pub max_gyro_radps: f32,
    pub max_accel_norm_err_mps2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CFusionConfig {
    pub align: CAlignConfig,
    pub bootstrap: CBootstrapConfig,
    pub predict_noise: PredictNoise,
    pub r_body_vel: f32,
    pub yaw_init_speed_mps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CImuSample {
    pub t_s: f64,
    pub gyro_radps: [f32; 3],
    pub accel_mps2: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CGnssSample {
    pub t_s: f64,
    pub pos_ned_m: [f32; 3],
    pub vel_ned_mps: [f32; 3],
    pub pos_std_m: [f32; 3],
    pub vel_std_mps: [f32; 3],
    pub heading_valid: bool,
    pub heading_rad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CUpdate {
    pub mount_ready: bool,
    pub mount_ready_changed: bool,
    pub ekf_initialized: bool,
    pub ekf_initialized_now: bool,
    pub mount_q_vb_valid: bool,
    pub mount_q_vb: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CStationaryMountBootstrap {
    pub mean_accel_b: [f32; 3],
    pub c_b_v: [[f32; 3]; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CAlignWindowSummary {
    pub dt: f32,
    pub mean_gyro_b: [f32; 3],
    pub mean_accel_b: [f32; 3],
    pub gnss_vel_prev_n: [f32; 3],
    pub gnss_vel_curr_n: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CAlignUpdateTrace {
    pub q_start: [f32; 4],
    pub coarse_alignment_ready: bool,
    pub after_gravity_valid: bool,
    pub after_gravity: [f32; 4],
    pub after_horiz_accel_valid: bool,
    pub after_horiz_accel: [f32; 4],
    pub horiz_angle_err_rad_valid: bool,
    pub horiz_angle_err_rad: f32,
    pub horiz_effective_std_rad_valid: bool,
    pub horiz_effective_std_rad: f32,
    pub horiz_gnss_norm_mps2_valid: bool,
    pub horiz_gnss_norm_mps2: f32,
    pub horiz_imu_norm_mps2_valid: bool,
    pub horiz_imu_norm_mps2: f32,
    pub horiz_speed_q_valid: bool,
    pub horiz_speed_q: f32,
    pub horiz_accel_q_valid: bool,
    pub horiz_accel_q: f32,
    pub horiz_straight_q_valid: bool,
    pub horiz_straight_q: f32,
    pub horiz_turn_q_valid: bool,
    pub horiz_turn_q: f32,
    pub horiz_dominance_q_valid: bool,
    pub horiz_dominance_q: f32,
    pub horiz_turn_core_valid: bool,
    pub horiz_straight_core_valid: bool,
    pub after_turn_gyro_valid: bool,
    pub after_turn_gyro: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CTurnConsistencySample {
    pub speed_mps: f32,
    pub course_rate_radps: f32,
    pub a_lat_mps2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CAlignRuntime {
    pub state: CAlignState,
    pub samples: [CTurnConsistencySample; 16],
    pub count: u32,
}

impl Default for CAlignRuntime {
    fn default() -> Self {
        Self {
            state: CAlignState::default(),
            samples: [CTurnConsistencySample::default(); 16],
            count: 0,
        }
    }
}

unsafe extern "C" {
    fn sf_bootstrap_vehicle_to_body_from_stationary(
        accel_samples_b: *const [f32; 3],
        sample_count: u32,
        yaw_seed_rad: f32,
        out: *mut CStationaryMountBootstrap,
    ) -> bool;

    fn sf_align_init(align_rt: *mut CAlignRuntime, cfg: *const CAlignConfig);
    fn sf_align_initialize_from_stationary(
        align_rt: *mut CAlignRuntime,
        cfg: *const CAlignConfig,
        accel_samples_b: *const [f32; 3],
        sample_count: u32,
        yaw_seed_rad: f32,
    ) -> bool;
    fn sf_align_update_window_with_trace(
        align_rt: *mut CAlignRuntime,
        cfg: *const CAlignConfig,
        window: *const CAlignWindowSummary,
        trace_out: *mut CAlignUpdateTrace,
    ) -> f32;
    fn sf_align_coarse_alignment_ready(align_rt: *const CAlignRuntime) -> bool;

    fn sf_fusion_init_internal(fusion: *mut CSensorFusion, cfg: *const CFusionConfig);
    fn sf_fusion_init_external(
        fusion: *mut CSensorFusion,
        cfg: *const CFusionConfig,
        q_vb: *const f32,
    );
    fn sf_fusion_set_misalignment(fusion: *mut CSensorFusion, q_vb: *const f32);
    fn sf_fusion_process_imu(fusion: *mut CSensorFusion, sample: *const CImuSample) -> CUpdate;
    fn sf_fusion_process_gnss(fusion: *mut CSensorFusion, sample: *const CGnssSample) -> CUpdate;
    fn sf_fusion_ekf(fusion: *const CSensorFusion) -> *const Ekf;
    fn sf_fusion_align(fusion: *const CSensorFusion) -> *const CAlignState;
    fn sf_fusion_mount_ready(fusion: *const CSensorFusion) -> bool;
    fn sf_fusion_mount_q_vb(fusion: *const CSensorFusion, out_q_vb: *mut f32) -> bool;
}

pub fn c_align_config_from_rust(cfg: AlignConfig) -> CAlignConfig {
    CAlignConfig {
        q_mount_std_rad: cfg.q_mount_std_rad,
        r_gravity_std_mps2: cfg.r_gravity_std_mps2,
        r_horiz_heading_std_rad: cfg.r_horiz_heading_std_rad,
        r_turn_gyro_std_radps: cfg.r_turn_gyro_std_radps,
        turn_gyro_yaw_scale: cfg.turn_gyro_yaw_scale,
        r_turn_heading_std_rad: cfg.r_turn_heading_std_rad,
        gravity_lpf_alpha: cfg.gravity_lpf_alpha,
        min_speed_mps: cfg.min_speed_mps,
        min_turn_rate_radps: cfg.min_turn_rate_radps,
        min_lat_acc_mps2: cfg.min_lat_acc_mps2,
        min_long_acc_mps2: cfg.min_long_acc_mps2,
        turn_consistency_min_windows: cfg.turn_consistency_min_windows as u32,
        turn_consistency_min_fraction: cfg.turn_consistency_min_fraction,
        turn_consistency_max_abs_lat_err_mps2: cfg.turn_consistency_max_abs_lat_err_mps2,
        turn_consistency_max_rel_lat_err: cfg.turn_consistency_max_rel_lat_err,
        max_stationary_gyro_radps: cfg.max_stationary_gyro_radps,
        max_stationary_accel_norm_err_mps2: cfg.max_stationary_accel_norm_err_mps2,
        use_gravity: cfg.use_gravity,
        use_turn_gyro: cfg.use_turn_gyro,
    }
}

pub fn c_fusion_config_from_rust(cfg: FusionConfig) -> CFusionConfig {
    CFusionConfig {
        align: c_align_config_from_rust(cfg.align),
        bootstrap: CBootstrapConfig {
            ema_alpha: cfg.bootstrap.ema_alpha,
            max_speed_mps: cfg.bootstrap.max_speed_mps,
            stationary_samples: cfg.bootstrap.stationary_samples as u32,
            max_gyro_radps: cfg.bootstrap.max_gyro_radps,
            max_accel_norm_err_mps2: cfg.bootstrap.max_accel_norm_err_mps2,
        },
        predict_noise: cfg.predict_noise,
        r_body_vel: cfg.r_body_vel,
        yaw_init_speed_mps: cfg.yaw_init_speed_mps,
    }
}

pub fn c_update_to_rust(update: CUpdate) -> FusionUpdate {
    FusionUpdate {
        mount_ready: update.mount_ready,
        mount_ready_changed: update.mount_ready_changed,
        ekf_initialized: update.ekf_initialized,
        ekf_initialized_now: update.ekf_initialized_now,
        mount_q_vb: update.mount_q_vb_valid.then_some(update.mount_q_vb),
    }
}

#[derive(Clone, Debug)]
pub struct CAlign {
    raw: CAlignRuntime,
    cfg: CAlignConfig,
}

impl CAlign {
    pub fn new(cfg: AlignConfig) -> Self {
        let c_cfg = c_align_config_from_rust(cfg);
        let mut raw = CAlignRuntime::default();
        // SAFETY: raw/cfg are valid stack objects and C does not retain pointers.
        unsafe { sf_align_init(&mut raw as *mut CAlignRuntime, &c_cfg as *const CAlignConfig) };
        Self { raw, cfg: c_cfg }
    }

    pub fn initialize_from_stationary(
        &mut self,
        accel_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
    ) -> bool {
        // SAFETY: slice pointer/count are valid for the duration of call.
        unsafe {
            sf_align_initialize_from_stationary(
                &mut self.raw as *mut CAlignRuntime,
                &self.cfg as *const CAlignConfig,
                accel_samples_b.as_ptr(),
                accel_samples_b.len() as u32,
                yaw_seed_rad,
            )
        }
    }

    pub fn update_window_with_trace(
        &mut self,
        window: &AlignWindowSummary,
    ) -> (f32, CAlignUpdateTrace) {
        let c_window = CAlignWindowSummary {
            dt: window.dt,
            mean_gyro_b: window.mean_gyro_b,
            mean_accel_b: window.mean_accel_b,
            gnss_vel_prev_n: window.gnss_vel_prev_n,
            gnss_vel_curr_n: window.gnss_vel_curr_n,
        };
        let mut trace = CAlignUpdateTrace::default();
        // SAFETY: pointers are valid and C does not retain them.
        let score = unsafe {
            sf_align_update_window_with_trace(
                &mut self.raw as *mut CAlignRuntime,
                &self.cfg as *const CAlignConfig,
                &c_window as *const CAlignWindowSummary,
                &mut trace as *mut CAlignUpdateTrace,
            )
        };
        (score, trace)
    }

    pub fn state(&self) -> &CAlignState {
        &self.raw.state
    }

    pub fn coarse_alignment_ready(&self) -> bool {
        // SAFETY: pointer is valid for self lifetime.
        unsafe { sf_align_coarse_alignment_ready(&self.raw as *const CAlignRuntime) }
    }

    pub fn bootstrap_from_stationary(
        accel_samples_b: &[[f32; 3]],
        yaw_seed_rad: f32,
    ) -> Option<CStationaryMountBootstrap> {
        let mut out = CStationaryMountBootstrap::default();
        // SAFETY: slice pointer/count are valid for duration of call.
        let ok = unsafe {
            sf_bootstrap_vehicle_to_body_from_stationary(
                accel_samples_b.as_ptr(),
                accel_samples_b.len() as u32,
                yaw_seed_rad,
                &mut out as *mut CStationaryMountBootstrap,
            )
        };
        ok.then_some(out)
    }
}

#[derive(Debug)]
pub struct CSensorFusionWrapper {
    raw: CSensorFusion,
}

impl CSensorFusionWrapper {
    pub fn new_internal(cfg: FusionConfig) -> Self {
        let mut raw = CSensorFusion::default();
        let c_cfg = c_fusion_config_from_rust(cfg);
        // SAFETY: valid pointers, C does not retain cfg pointer.
        unsafe { sf_fusion_init_internal(&mut raw as *mut CSensorFusion, &c_cfg as *const CFusionConfig) };
        Self { raw }
    }

    pub fn new_external(cfg: FusionConfig, q_vb: [f32; 4]) -> Self {
        let mut raw = CSensorFusion::default();
        let c_cfg = c_fusion_config_from_rust(cfg);
        // SAFETY: valid pointers, C copies inputs.
        unsafe {
            sf_fusion_init_external(
                &mut raw as *mut CSensorFusion,
                &c_cfg as *const CFusionConfig,
                q_vb.as_ptr(),
            )
        };
        Self { raw }
    }

    pub fn set_misalignment(&mut self, q_vb: [f32; 4]) {
        // SAFETY: valid pointer, C copies input.
        unsafe { sf_fusion_set_misalignment(&mut self.raw as *mut CSensorFusion, q_vb.as_ptr()) }
    }

    pub fn process_imu(&mut self, sample: FusionImuSample) -> FusionUpdate {
        let c_sample = CImuSample {
            t_s: sample.t_s,
            gyro_radps: sample.gyro_radps,
            accel_mps2: sample.accel_mps2,
        };
        // SAFETY: valid pointers, C returns by value.
        let update = unsafe {
            sf_fusion_process_imu(&mut self.raw as *mut CSensorFusion, &c_sample as *const CImuSample)
        };
        c_update_to_rust(update)
    }

    pub fn process_gnss(&mut self, sample: FusionGnssSample) -> FusionUpdate {
        let c_sample = CGnssSample {
            t_s: sample.t_s,
            pos_ned_m: sample.pos_ned_m,
            vel_ned_mps: sample.vel_ned_mps,
            pos_std_m: sample.pos_std_m,
            vel_std_mps: sample.vel_std_mps,
            heading_valid: sample.heading_rad.is_some(),
            heading_rad: sample.heading_rad.unwrap_or(0.0),
        };
        // SAFETY: valid pointers, C returns by value.
        let update = unsafe {
            sf_fusion_process_gnss(
                &mut self.raw as *mut CSensorFusion,
                &c_sample as *const CGnssSample,
            )
        };
        c_update_to_rust(update)
    }

    pub fn ekf(&self) -> Option<&Ekf> {
        // SAFETY: C returns null or pointer to internal state owned by self.
        let p = unsafe { sf_fusion_ekf(&self.raw as *const CSensorFusion) };
        if p.is_null() { None } else { Some(unsafe { &*p }) }
    }

    pub fn ekf_mut(&mut self) -> Option<&mut Ekf> {
        let p = unsafe { sf_fusion_ekf(&self.raw as *const CSensorFusion) } as *mut Ekf;
        if p.is_null() {
            None
        } else {
            Some(unsafe { &mut *p })
        }
    }

    pub fn align_state(&self) -> Option<&CAlignState> {
        // SAFETY: C returns null or pointer to internal state owned by self.
        let p = unsafe { sf_fusion_align(&self.raw as *const CSensorFusion) };
        if p.is_null() { None } else { Some(unsafe { &*p }) }
    }

    pub fn mount_ready(&self) -> bool {
        // SAFETY: valid pointer to owned storage.
        unsafe { sf_fusion_mount_ready(&self.raw as *const CSensorFusion) }
    }

    pub fn mount_q_vb(&self) -> Option<[f32; 4]> {
        let mut out = [0.0f32; 4];
        // SAFETY: valid pointers, C writes exactly 4 floats on success.
        let ok = unsafe {
            sf_fusion_mount_q_vb(&self.raw as *const CSensorFusion, out.as_mut_ptr())
        };
        ok.then_some(out)
    }
}
