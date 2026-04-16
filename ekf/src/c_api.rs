#![allow(non_snake_case)]

use crate::align::{AlignConfig, AlignWindowSummary};
use crate::ekf::PredictNoise;
use crate::fusion::{
    FusionGnssSample, FusionImuSample, FusionUpdate, FusionVehicleSpeedDirection,
    FusionVehicleSpeedSample,
};
use crate::loose::LoosePredictNoise;

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
pub struct CImuSample {
    pub t_s: f32,
    pub gyro_radps: [f32; 3],
    pub accel_mps2: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CGnssSample {
    pub t_s: f32,
    pub lat_deg: f32,
    pub lon_deg: f32,
    pub height_m: f32,
    pub vel_ned_mps: [f32; 3],
    pub pos_std_m: [f32; 3],
    pub vel_std_mps: [f32; 3],
    pub heading_valid: bool,
    pub heading_rad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CVehicleSpeedSample {
    pub t_s: f32,
    pub speed_mps: f32,
    pub direction: CVehicleSpeedDirection,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CVehicleSpeedDirection {
    #[default]
    Unknown = 0,
    Forward = 1,
    Reverse = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CGnssNedSample {
    pub t_s: f32,
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
    pub sensor_fusion_state: bool,
    pub sensor_fusion_state_changed: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CAlignStatus {
    None = 0,
    Coarse = 1,
    Fine = 2,
}

impl Default for CAlignStatus {
    fn default() -> Self {
        Self::None
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CFusionState {
    pub mount_ready: bool,
    pub mount_q_vb_valid: bool,
    pub mount_q_vb: [f32; 4],
    pub align_state: CAlignStatus,
    pub align_q_vb: [f32; 4],
    pub align_sigma_rad: [f32; 3],
    pub gravity_lp_b: [f32; 3],
    pub sensor_fusion_state: bool,
    pub q_bn: [f32; 4],
    pub vel_ned_mps: [f32; 3],
    pub pos_ned_m: [f32; 3],
    pub gyro_bias_radps: [f32; 3],
    pub accel_bias_mps2: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CFusionDebug {
    pub align_window_valid: bool,
    pub align_window: CAlignWindowSummary,
    pub align_trace_valid: bool,
    pub align_trace: CAlignUpdateTrace,
    pub eskf_valid: bool,
    pub eskf: CEskf,
    pub reanchor_count: u32,
    pub last_reanchor_valid: bool,
    pub last_reanchor_t_s: f32,
    pub last_reanchor_distance_m: f32,
    pub anchor_valid: bool,
    pub anchor_lat_deg: f32,
    pub anchor_lon_deg: f32,
    pub anchor_height_m: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CEskfNominalState {
    pub q0: f32,
    pub q1: f32,
    pub q2: f32,
    pub q3: f32,
    pub vn: f32,
    pub ve: f32,
    pub vd: f32,
    pub pn: f32,
    pub pe: f32,
    pub pd: f32,
    pub bgx: f32,
    pub bgy: f32,
    pub bgz: f32,
    pub bax: f32,
    pub bay: f32,
    pub baz: f32,
    pub qcs0: f32,
    pub qcs1: f32,
    pub qcs2: f32,
    pub qcs3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CEskfStationaryDiag {
    pub innovation_x: f32,
    pub innovation_y: f32,
    pub k_theta_x_from_x: f32,
    pub k_theta_y_from_x: f32,
    pub k_bax_from_x: f32,
    pub k_bay_from_x: f32,
    pub k_theta_x_from_y: f32,
    pub k_theta_y_from_y: f32,
    pub k_bax_from_y: f32,
    pub k_bay_from_y: f32,
    pub p_theta_x: f32,
    pub p_theta_y: f32,
    pub p_bax: f32,
    pub p_bay: f32,
    pub p_theta_x_bax: f32,
    pub p_theta_y_bay: f32,
    pub updates: u32,
}

pub const C_ESKF_UPDATE_DIAG_TYPES: usize = 11;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CEskfUpdateDiag {
    pub total_updates: u32,
    pub type_counts: [u32; C_ESKF_UPDATE_DIAG_TYPES],
    pub sum_dx_pitch: [f32; C_ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_dx_pitch: [f32; C_ESKF_UPDATE_DIAG_TYPES],
    pub sum_dx_mount_yaw: [f32; C_ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_dx_mount_yaw: [f32; C_ESKF_UPDATE_DIAG_TYPES],
    pub sum_innovation: [f32; C_ESKF_UPDATE_DIAG_TYPES],
    pub sum_abs_innovation: [f32; C_ESKF_UPDATE_DIAG_TYPES],
    pub last_dx_mount_yaw: f32,
    pub last_k_mount_yaw: f32,
    pub last_innovation: f32,
    pub last_innovation_var: f32,
    pub last_type: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CEskf {
    pub nominal: CEskfNominalState,
    pub p: [[f32; 18]; 18],
    pub noise: PredictNoise,
    pub stationary_diag: CEskfStationaryDiag,
    pub update_diag: CEskfUpdateDiag,
}

#[derive(Clone, Copy, Debug)]
pub struct EskfGnssSample {
    pub t_s: f32,
    pub pos_ned_m: [f32; 3],
    pub vel_ned_mps: [f32; 3],
    pub pos_std_m: [f32; 3],
    pub vel_std_mps: [f32; 3],
    pub heading_rad: Option<f32>,
}

impl Default for CEskf {
    fn default() -> Self {
        Self {
            nominal: CEskfNominalState {
                qcs0: 1.0,
                ..CEskfNominalState::default()
            },
            p: [[0.0; 18]; 18],
            noise: PredictNoise::lsm6dso_typical_104hz(),
            stationary_diag: CEskfStationaryDiag::default(),
            update_diag: CEskfUpdateDiag::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CEskfImuDelta {
    pub dax: f32,
    pub day: f32,
    pub daz: f32,
    pub dvx: f32,
    pub dvy: f32,
    pub dvz: f32,
    pub dt: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CLooseNominalState {
    pub q0: f32,
    pub q1: f32,
    pub q2: f32,
    pub q3: f32,
    pub vn: f32,
    pub ve: f32,
    pub vd: f32,
    pub pn: f32,
    pub pe: f32,
    pub pd: f32,
    pub bgx: f32,
    pub bgy: f32,
    pub bgz: f32,
    pub bax: f32,
    pub bay: f32,
    pub baz: f32,
    pub sgx: f32,
    pub sgy: f32,
    pub sgz: f32,
    pub sax: f32,
    pub say: f32,
    pub saz: f32,
    pub qcs0: f32,
    pub qcs1: f32,
    pub qcs2: f32,
    pub qcs3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CLoose {
    pub nominal: CLooseNominalState,
    pub p: [[f32; 24]; 24],
    pub noise: LoosePredictNoise,
    pub pos_e64: [f64; 3],
    pub qcs64: [f64; 4],
    pub p64: [[f64; 24]; 24],
    pub last_dx: [f32; 24],
    pub last_obs_count: i32,
    pub last_obs_types: [i32; 8],
}

impl Default for CLoose {
    fn default() -> Self {
        Self {
            nominal: CLooseNominalState::default(),
            p: [[0.0; 24]; 24],
            noise: LoosePredictNoise::default(),
            pos_e64: [0.0; 3],
            qcs64: [1.0, 0.0, 0.0, 0.0],
            p64: [[0.0; 24]; 24],
            last_dx: [0.0; 24],
            last_obs_count: 0,
            last_obs_types: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CLooseImuDelta {
    pub dax_1: f32,
    pub day_1: f32,
    pub daz_1: f32,
    pub dvx_1: f32,
    pub dvy_1: f32,
    pub dvz_1: f32,
    pub dax_2: f32,
    pub day_2: f32,
    pub daz_2: f32,
    pub dvx_2: f32,
    pub dvy_2: f32,
    pub dvz_2: f32,
    pub dt: f32,
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
    pub after_gravity_quasi_static: bool,
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
    pub horiz_obs_accel_vx_valid: bool,
    pub horiz_obs_accel_vx: f32,
    pub horiz_obs_accel_vy_valid: bool,
    pub horiz_obs_accel_vy: f32,
    pub horiz_accel_bx_valid: bool,
    pub horiz_accel_bx: f32,
    pub horiz_accel_by_valid: bool,
    pub horiz_accel_by: f32,
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
    pub yaw_observed: bool,
}

impl Default for CAlignRuntime {
    fn default() -> Self {
        Self {
            state: CAlignState::default(),
            samples: [CTurnConsistencySample::default(); 16],
            count: 0,
            yaw_observed: false,
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

    fn sf_init(fusion: *mut CSensorFusion, q_vb_or_null: *const f32);
    fn sf_set_r_body_vel(fusion: *mut CSensorFusion, r_body_vel: f32);
    fn sf_set_gnss_pos_mount_scale(fusion: *mut CSensorFusion, gnss_pos_mount_scale: f32);
    fn sf_set_gnss_vel_mount_scale(fusion: *mut CSensorFusion, gnss_vel_mount_scale: f32);
    fn sf_set_gyro_bias_init_sigma_radps(
        fusion: *mut CSensorFusion,
        gyro_bias_init_sigma_radps: f32,
    );
    fn sf_set_accel_bias_init_sigma_mps2(
        fusion: *mut CSensorFusion,
        accel_bias_init_sigma_mps2: f32,
    );
    fn sf_set_accel_bias_rw_var(fusion: *mut CSensorFusion, accel_bias_rw_var: f32);
    fn sf_set_mount_align_rw_var(fusion: *mut CSensorFusion, mount_align_rw_var: f32);
    fn sf_set_mount_update_min_scale(fusion: *mut CSensorFusion, mount_update_min_scale: f32);
    fn sf_set_mount_update_ramp_time_s(
        fusion: *mut CSensorFusion,
        mount_update_ramp_time_s: f32,
    );
    fn sf_set_mount_update_innovation_gate_mps(
        fusion: *mut CSensorFusion,
        mount_update_innovation_gate_mps: f32,
    );
    fn sf_set_r_vehicle_speed(fusion: *mut CSensorFusion, r_vehicle_speed: f32);
    fn sf_set_r_zero_vel(fusion: *mut CSensorFusion, r_zero_vel: f32);
    fn sf_set_r_stationary_accel(fusion: *mut CSensorFusion, r_stationary_accel: f32);
    fn sf_process_imu(fusion: *mut CSensorFusion, sample: *const CImuSample) -> CUpdate;
    fn sf_process_gnss(fusion: *mut CSensorFusion, sample: *const CGnssSample) -> CUpdate;
    fn sf_process_vehicle_speed(
        fusion: *mut CSensorFusion,
        sample: *const CVehicleSpeedSample,
    ) -> CUpdate;
    fn sf_get_state(fusion: *const CSensorFusion, out: *mut CFusionState) -> bool;
    fn sf_get_lla(fusion: *const CSensorFusion, out_lla: *mut f32) -> bool;
    fn sf_fusion_get_debug(fusion: *const CSensorFusion, out: *mut CFusionDebug) -> bool;
    fn sf_fusion_eskf_mount_q_vb(fusion: *const CSensorFusion, out_q_vb: *mut f32) -> bool;

    fn sf_fusion_set_misalignment(fusion: *mut CSensorFusion, q_vb: *const f32);

    fn sf_eskf_init(eskf: *mut CEskf, p_diag: *const f32, noise: *const PredictNoise);
    fn sf_eskf_predict(eskf: *mut CEskf, imu: *const CEskfImuDelta);
    fn sf_eskf_fuse_gps(eskf: *mut CEskf, gps: *const CGnssNedSample);
    fn sf_eskf_fuse_body_speed_x(eskf: *mut CEskf, speed_mps: f32, r_speed: f32);
    fn sf_eskf_fuse_body_vel(eskf: *mut CEskf, r_body_vel: f32);
    fn sf_eskf_fuse_zero_vel(eskf: *mut CEskf, r_zero_vel: f32);
    fn sf_eskf_fuse_stationary_gravity(
        eskf: *mut CEskf,
        accel_body_mps2: *const f32,
        r_stationary_accel: f32,
    );

    fn sf_loose_init(loose: *mut CLoose, p_diag: *const f32, noise: *const LoosePredictNoise);
    fn sf_loose_predict(loose: *mut CLoose, imu: *const CLooseImuDelta);
    fn sf_loose_fuse_gps_reference(
        loose: *mut CLoose,
        pos_ecef_m: *const f64,
        vel_ecef_mps: *const f32,
        h_acc_m: f32,
        speed_acc_mps: f32,
        dt_since_last_gnss_s: f32,
    );
    fn sf_loose_fuse_gps_reference_full(
        loose: *mut CLoose,
        pos_ecef_m: *const f64,
        vel_ecef_mps: *const f32,
        h_acc_m: f32,
        vel_std_ned_mps: *const f32,
        dt_since_last_gnss_s: f32,
    );
    fn sf_loose_fuse_reference_batch(
        loose: *mut CLoose,
        pos_ecef_m: *const f64,
        vel_ecef_mps: *const f32,
        h_acc_m: f32,
        speed_acc_mps: f32,
        dt_since_last_gnss_s: f32,
        gyro_radps: *const f32,
        accel_mps2: *const f32,
        dt_s: f32,
    );
    fn sf_loose_fuse_reference_batch_full(
        loose: *mut CLoose,
        pos_ecef_m: *const f64,
        vel_ecef_mps: *const f32,
        h_acc_m: f32,
        vel_std_ned_mps: *const f32,
        dt_since_last_gnss_s: f32,
        gyro_radps: *const f32,
        accel_mps2: *const f32,
        dt_s: f32,
    );
    fn sf_loose_fuse_nhc_reference(
        loose: *mut CLoose,
        gyro_radps: *const f32,
        accel_mps2: *const f32,
        dt_s: f32,
    );
    fn sf_loose_compute_error_transition(
        f_out: *mut [[f32; 24]; 24],
        g_out: *mut [[f32; 21]; 24],
        loose: *const CLoose,
        imu: *const CLooseImuDelta,
    );
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

pub fn c_update_to_rust(update: CUpdate, mount_q_vb: Option<[f32; 4]>) -> FusionUpdate {
    FusionUpdate {
        mount_ready: update.mount_ready,
        mount_ready_changed: update.mount_ready_changed,
        ekf_initialized: update.sensor_fusion_state,
        ekf_initialized_now: update.sensor_fusion_state_changed,
        mount_q_vb,
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
        unsafe {
            sf_align_init(
                &mut raw as *mut CAlignRuntime,
                &c_cfg as *const CAlignConfig,
            )
        };
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
    state: CFusionState,
    debug: CFusionDebug,
    cached_eskf: Option<CEskf>,
    cached_align: Option<CAlignState>,
}

#[derive(Debug, Clone)]
pub struct CEskfWrapper {
    raw: CEskf,
}

pub struct CLooseWrapper {
    raw: CLoose,
}

impl CSensorFusionWrapper {
    pub fn new_internal() -> Self {
        let mut raw = CSensorFusion::default();
        unsafe { sf_init(&mut raw as *mut CSensorFusion, core::ptr::null()) };
        let mut out = Self {
            raw,
            state: CFusionState::default(),
            debug: CFusionDebug::default(),
            cached_eskf: None,
            cached_align: None,
        };
        out.refresh_state();
        out
    }

    pub fn new_external(q_vb: [f32; 4]) -> Self {
        let mut raw = CSensorFusion::default();
        unsafe { sf_init(&mut raw as *mut CSensorFusion, q_vb.as_ptr()) };
        let mut out = Self {
            raw,
            state: CFusionState::default(),
            debug: CFusionDebug::default(),
            cached_eskf: None,
            cached_align: None,
        };
        out.refresh_state();
        out
    }

    pub fn set_misalignment(&mut self, q_vb: [f32; 4]) {
        // SAFETY: valid pointer, C copies input.
        unsafe { sf_fusion_set_misalignment(&mut self.raw as *mut CSensorFusion, q_vb.as_ptr()) }
        self.refresh_state();
    }

    pub fn set_r_body_vel(&mut self, r_body_vel: f32) {
        unsafe { sf_set_r_body_vel(&mut self.raw as *mut CSensorFusion, r_body_vel) }
        self.refresh_state();
    }

    pub fn set_gnss_pos_mount_scale(&mut self, gnss_pos_mount_scale: f32) {
        unsafe {
            sf_set_gnss_pos_mount_scale(
                &mut self.raw as *mut CSensorFusion,
                gnss_pos_mount_scale,
            )
        }
        self.refresh_state();
    }

    pub fn set_gnss_vel_mount_scale(&mut self, gnss_vel_mount_scale: f32) {
        unsafe {
            sf_set_gnss_vel_mount_scale(
                &mut self.raw as *mut CSensorFusion,
                gnss_vel_mount_scale,
            )
        }
        self.refresh_state();
    }

    pub fn set_gyro_bias_init_sigma_radps(&mut self, gyro_bias_init_sigma_radps: f32) {
        unsafe {
            sf_set_gyro_bias_init_sigma_radps(
                &mut self.raw as *mut CSensorFusion,
                gyro_bias_init_sigma_radps,
            )
        }
        self.refresh_state();
    }

    pub fn set_accel_bias_init_sigma_mps2(&mut self, accel_bias_init_sigma_mps2: f32) {
        unsafe {
            sf_set_accel_bias_init_sigma_mps2(
                &mut self.raw as *mut CSensorFusion,
                accel_bias_init_sigma_mps2,
            )
        }
        self.refresh_state();
    }

    pub fn set_accel_bias_rw_var(&mut self, accel_bias_rw_var: f32) {
        unsafe { sf_set_accel_bias_rw_var(&mut self.raw as *mut CSensorFusion, accel_bias_rw_var) }
        self.refresh_state();
    }

    pub fn set_mount_align_rw_var(&mut self, mount_align_rw_var: f32) {
        unsafe {
            sf_set_mount_align_rw_var(&mut self.raw as *mut CSensorFusion, mount_align_rw_var)
        }
        self.refresh_state();
    }

    pub fn set_mount_update_min_scale(&mut self, mount_update_min_scale: f32) {
        unsafe {
            sf_set_mount_update_min_scale(
                &mut self.raw as *mut CSensorFusion,
                mount_update_min_scale,
            )
        }
        self.refresh_state();
    }

    pub fn set_mount_update_ramp_time_s(&mut self, mount_update_ramp_time_s: f32) {
        unsafe {
            sf_set_mount_update_ramp_time_s(
                &mut self.raw as *mut CSensorFusion,
                mount_update_ramp_time_s,
            )
        }
        self.refresh_state();
    }

    pub fn set_mount_update_innovation_gate_mps(
        &mut self,
        mount_update_innovation_gate_mps: f32,
    ) {
        unsafe {
            sf_set_mount_update_innovation_gate_mps(
                &mut self.raw as *mut CSensorFusion,
                mount_update_innovation_gate_mps,
            )
        }
        self.refresh_state();
    }

    pub fn set_r_vehicle_speed(&mut self, r_vehicle_speed: f32) {
        unsafe { sf_set_r_vehicle_speed(&mut self.raw as *mut CSensorFusion, r_vehicle_speed) }
        self.refresh_state();
    }

    pub fn set_r_zero_vel(&mut self, r_zero_vel: f32) {
        unsafe { sf_set_r_zero_vel(&mut self.raw as *mut CSensorFusion, r_zero_vel) }
        self.refresh_state();
    }

    pub fn set_r_stationary_accel(&mut self, r_stationary_accel: f32) {
        unsafe {
            sf_set_r_stationary_accel(&mut self.raw as *mut CSensorFusion, r_stationary_accel)
        }
        self.refresh_state();
    }

    pub fn process_imu(&mut self, sample: FusionImuSample) -> FusionUpdate {
        let c_sample = CImuSample {
            t_s: sample.t_s,
            gyro_radps: sample.gyro_radps,
            accel_mps2: sample.accel_mps2,
        };
        // SAFETY: valid pointers, C returns by value.
        let update = unsafe {
            sf_process_imu(
                &mut self.raw as *mut CSensorFusion,
                &c_sample as *const CImuSample,
            )
        };
        self.refresh_state();
        let mount_q_vb = self.mount_q_vb();
        c_update_to_rust(update, mount_q_vb)
    }

    pub fn process_gnss(&mut self, sample: FusionGnssSample) -> FusionUpdate {
        let c_sample = CGnssSample {
            t_s: sample.t_s,
            lat_deg: sample.lat_deg,
            lon_deg: sample.lon_deg,
            height_m: sample.height_m,
            vel_ned_mps: sample.vel_ned_mps,
            pos_std_m: sample.pos_std_m,
            vel_std_mps: sample.vel_std_mps,
            heading_valid: sample.heading_rad.is_some(),
            heading_rad: sample.heading_rad.unwrap_or(0.0),
        };
        // SAFETY: valid pointers, C returns by value.
        let update = unsafe {
            sf_process_gnss(
                &mut self.raw as *mut CSensorFusion,
                &c_sample as *const CGnssSample,
            )
        };
        self.refresh_state();
        let mount_q_vb = self.mount_q_vb();
        c_update_to_rust(update, mount_q_vb)
    }

    pub fn process_vehicle_speed(&mut self, sample: FusionVehicleSpeedSample) -> FusionUpdate {
        let c_sample = CVehicleSpeedSample {
            t_s: sample.t_s,
            speed_mps: sample.speed_mps,
            direction: match sample.direction {
                FusionVehicleSpeedDirection::Unknown => CVehicleSpeedDirection::Unknown,
                FusionVehicleSpeedDirection::Forward => CVehicleSpeedDirection::Forward,
                FusionVehicleSpeedDirection::Reverse => CVehicleSpeedDirection::Reverse,
            },
        };
        let update = unsafe {
            sf_process_vehicle_speed(
                &mut self.raw as *mut CSensorFusion,
                &c_sample as *const CVehicleSpeedSample,
            )
        };
        self.refresh_state();
        let mount_q_vb = self.mount_q_vb();
        c_update_to_rust(update, mount_q_vb)
    }

    pub fn eskf(&self) -> Option<&CEskf> {
        self.cached_eskf.as_ref()
    }

    pub fn align_state(&self) -> Option<&CAlignState> {
        self.cached_align.as_ref()
    }

    pub fn mount_ready(&self) -> bool {
        self.state.mount_ready
    }

    pub fn mount_q_vb(&self) -> Option<[f32; 4]> {
        self.state.mount_q_vb_valid.then_some(self.state.mount_q_vb)
    }

    pub fn eskf_mount_q_vb(&self) -> Option<[f32; 4]> {
        let mut out = [0.0_f32; 4];
        let ok = unsafe {
            sf_fusion_eskf_mount_q_vb(&self.raw as *const CSensorFusion, out.as_mut_ptr())
        };
        ok.then_some(out)
    }

    pub fn position_lla(&self) -> Option<[f32; 3]> {
        let mut out = [0.0_f32; 3];
        let ok = unsafe { sf_get_lla(&self.raw as *const CSensorFusion, out.as_mut_ptr()) };
        ok.then_some(out)
    }

    pub fn reanchor_count(&self) -> u32 {
        self.debug.reanchor_count
    }

    pub fn last_reanchor_info(&self) -> Option<(f32, f32)> {
        self.debug.last_reanchor_valid.then_some((
            self.debug.last_reanchor_t_s,
            self.debug.last_reanchor_distance_m,
        ))
    }

    pub fn anchor_lla_debug(&self) -> Option<[f32; 3]> {
        self.debug.anchor_valid.then_some([
            self.debug.anchor_lat_deg,
            self.debug.anchor_lon_deg,
            self.debug.anchor_height_m,
        ])
    }

    pub fn align_window_debug(&self) -> Option<&CAlignWindowSummary> {
        self.debug
            .align_window_valid
            .then_some(&self.debug.align_window)
    }

    pub fn align_trace_debug(&self) -> Option<&CAlignUpdateTrace> {
        self.debug
            .align_trace_valid
            .then_some(&self.debug.align_trace)
    }

    fn refresh_state(&mut self) {
        let mut state = CFusionState::default();
        let mut debug = CFusionDebug::default();
        let ok = unsafe {
            sf_get_state(
                &self.raw as *const CSensorFusion,
                &mut state as *mut CFusionState,
            )
        };
        if !ok {
            return;
        }
        let _ = unsafe {
            sf_fusion_get_debug(
                &self.raw as *const CSensorFusion,
                &mut debug as *mut CFusionDebug,
            )
        };
        self.state = state;
        self.debug = debug;

        self.cached_align = match state.align_state {
            CAlignStatus::None => None,
            CAlignStatus::Coarse | CAlignStatus::Fine => {
                let mut p = [[0.0_f32; 3]; 3];
                p[0][0] = state.align_sigma_rad[0] * state.align_sigma_rad[0];
                p[1][1] = state.align_sigma_rad[1] * state.align_sigma_rad[1];
                p[2][2] = state.align_sigma_rad[2] * state.align_sigma_rad[2];
                Some(CAlignState {
                    q_vb: state.align_q_vb,
                    p,
                    gravity_lp_b: state.gravity_lp_b,
                    coarse_alignment_ready: state.align_state == CAlignStatus::Fine,
                })
            }
        };

        self.cached_eskf = if debug.eskf_valid {
            Some(debug.eskf)
        } else if state.sensor_fusion_state {
            let mut eskf = CEskf::default();
            eskf.nominal.q0 = state.q_bn[0];
            eskf.nominal.q1 = state.q_bn[1];
            eskf.nominal.q2 = state.q_bn[2];
            eskf.nominal.q3 = state.q_bn[3];
            eskf.nominal.vn = state.vel_ned_mps[0];
            eskf.nominal.ve = state.vel_ned_mps[1];
            eskf.nominal.vd = state.vel_ned_mps[2];
            eskf.nominal.pn = state.pos_ned_m[0];
            eskf.nominal.pe = state.pos_ned_m[1];
            eskf.nominal.pd = state.pos_ned_m[2];
            eskf.nominal.bgx = state.gyro_bias_radps[0];
            eskf.nominal.bgy = state.gyro_bias_radps[1];
            eskf.nominal.bgz = state.gyro_bias_radps[2];
            eskf.nominal.bax = state.accel_bias_mps2[0];
            eskf.nominal.bay = state.accel_bias_mps2[1];
            eskf.nominal.baz = state.accel_bias_mps2[2];
            eskf.nominal.qcs0 = 1.0;
            Some(eskf)
        } else {
            None
        };
    }
}

impl CEskfWrapper {
    pub fn new(noise: PredictNoise) -> Self {
        let mut raw = CEskf::default();
        unsafe {
            sf_eskf_init(
                &mut raw as *mut CEskf,
                core::ptr::null(),
                &noise as *const _,
            )
        };
        Self { raw }
    }

    pub fn nominal(&self) -> &CEskfNominalState {
        &self.raw.nominal
    }

    pub fn covariance(&self) -> &[[f32; 18]; 18] {
        &self.raw.p
    }

    pub fn stationary_diag(&self) -> &CEskfStationaryDiag {
        &self.raw.stationary_diag
    }

    pub fn init_nominal_from_gnss(&mut self, q_bn: [f32; 4], gnss: EskfGnssSample) {
        const DEFAULT_GYRO_BIAS_SIGMA_DPS: f32 = 0.125;
        const DEFAULT_ACCEL_BIAS_SIGMA_MPS2: f32 = 0.20;

        self.raw.nominal.q0 = q_bn[0];
        self.raw.nominal.q1 = q_bn[1];
        self.raw.nominal.q2 = q_bn[2];
        self.raw.nominal.q3 = q_bn[3];
        self.raw.nominal.vn = gnss.vel_ned_mps[0];
        self.raw.nominal.ve = gnss.vel_ned_mps[1];
        self.raw.nominal.vd = gnss.vel_ned_mps[2];
        self.raw.nominal.pn = gnss.pos_ned_m[0];
        self.raw.nominal.pe = gnss.pos_ned_m[1];
        self.raw.nominal.pd = gnss.pos_ned_m[2];
        self.raw.nominal.qcs0 = 1.0;
        self.raw.nominal.qcs1 = 0.0;
        self.raw.nominal.qcs2 = 0.0;
        self.raw.nominal.qcs3 = 0.0;
        let att_sigma_rad = 2.0f32 * core::f32::consts::PI / 180.0;
        let att_var = att_sigma_rad * att_sigma_rad;
        self.raw.p[0][0] = att_var;
        self.raw.p[1][1] = att_var;
        self.raw.p[2][2] = att_var;

        let mut vel_std = gnss.vel_std_mps[0]
            .max(gnss.vel_std_mps[1])
            .max(gnss.vel_std_mps[2]);
        if vel_std < 0.2 {
            vel_std = 0.2;
        }
        let vel_var = vel_std * vel_std;
        self.raw.p[3][3] = vel_var;
        self.raw.p[4][4] = vel_var;
        self.raw.p[5][5] = vel_var;

        let pos_n = gnss.pos_std_m[0].max(0.5);
        let pos_e = gnss.pos_std_m[1].max(0.5);
        let pos_d = gnss.pos_std_m[2].max(0.5);
        self.raw.p[6][6] = pos_n * pos_n;
        self.raw.p[7][7] = pos_e * pos_e;
        self.raw.p[8][8] = pos_d * pos_d;

        let gyro_bias_sigma_radps = DEFAULT_GYRO_BIAS_SIGMA_DPS * core::f32::consts::PI / 180.0;
        let accel_bias_sigma_mps2 = DEFAULT_ACCEL_BIAS_SIGMA_MPS2;
        self.raw.p[9][9] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
        self.raw.p[10][10] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
        self.raw.p[11][11] = gyro_bias_sigma_radps * gyro_bias_sigma_radps;
        self.raw.p[12][12] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
        self.raw.p[13][13] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
        self.raw.p[14][14] = accel_bias_sigma_mps2 * accel_bias_sigma_mps2;
        let mount_residual_sigma_rad = 10.0f32 * core::f32::consts::PI / 180.0;
        self.raw.p[15][15] = mount_residual_sigma_rad * mount_residual_sigma_rad;
        self.raw.p[16][16] = mount_residual_sigma_rad * mount_residual_sigma_rad;
        self.raw.p[17][17] = mount_residual_sigma_rad * mount_residual_sigma_rad;
    }

    pub fn predict(&mut self, imu: CEskfImuDelta) {
        unsafe { sf_eskf_predict(&mut self.raw as *mut CEskf, &imu as *const CEskfImuDelta) };
    }

    pub fn fuse_gps(&mut self, sample: EskfGnssSample) {
        let c_sample = CGnssNedSample {
            t_s: sample.t_s,
            pos_ned_m: sample.pos_ned_m,
            vel_ned_mps: sample.vel_ned_mps,
            pos_std_m: sample.pos_std_m,
            vel_std_mps: sample.vel_std_mps,
            heading_valid: sample.heading_rad.is_some(),
            heading_rad: sample.heading_rad.unwrap_or(0.0),
        };
        unsafe {
            sf_eskf_fuse_gps(
                &mut self.raw as *mut CEskf,
                &c_sample as *const CGnssNedSample,
            )
        };
    }

    pub fn fuse_body_vel(&mut self, r_body_vel: f32) {
        unsafe { sf_eskf_fuse_body_vel(&mut self.raw as *mut CEskf, r_body_vel) };
    }

    pub fn fuse_body_speed_x(&mut self, speed_mps: f32, r_speed: f32) {
        unsafe { sf_eskf_fuse_body_speed_x(&mut self.raw as *mut CEskf, speed_mps, r_speed) };
    }

    pub fn fuse_zero_vel(&mut self, r_zero_vel: f32) {
        unsafe { sf_eskf_fuse_zero_vel(&mut self.raw as *mut CEskf, r_zero_vel) };
    }

    pub fn fuse_stationary_gravity(&mut self, accel_body_mps2: [f32; 3], r_stationary_accel: f32) {
        unsafe {
            sf_eskf_fuse_stationary_gravity(
                &mut self.raw as *mut CEskf,
                accel_body_mps2.as_ptr(),
                r_stationary_accel,
            )
        };
    }
}

impl CLooseWrapper {
    pub fn new(noise: LoosePredictNoise) -> Self {
        let mut raw = CLoose::default();
        unsafe {
            sf_loose_init(
                &mut raw as *mut CLoose,
                core::ptr::null(),
                &noise as *const _,
            )
        };
        Self { raw }
    }

    pub fn nominal(&self) -> &CLooseNominalState {
        &self.raw.nominal
    }

    pub fn covariance(&self) -> &[[f32; 24]; 24] {
        &self.raw.p
    }

    pub fn shadow_pos_ecef(&self) -> [f64; 3] {
        self.raw.pos_e64
    }

    pub fn set_covariance(&mut self, p: [[f32; 24]; 24]) {
        self.raw.p = p;
        for i in 0..24 {
            for j in 0..24 {
                self.raw.p64[i][j] = self.raw.p[i][j] as f64;
            }
        }
    }

    pub fn last_obs_types(&self) -> &[i32] {
        let count = self
            .raw
            .last_obs_count
            .clamp(0, self.raw.last_obs_types.len() as i32) as usize;
        &self.raw.last_obs_types[..count]
    }

    pub fn last_dx(&self) -> &[f32; 24] {
        &self.raw.last_dx
    }

    pub fn set_mount_quat(&mut self, q_cs: [f32; 4]) {
        self.raw.nominal.qcs0 = q_cs[0];
        self.raw.nominal.qcs1 = q_cs[1];
        self.raw.nominal.qcs2 = q_cs[2];
        self.raw.nominal.qcs3 = q_cs[3];
        self.raw.qcs64 = [
            q_cs[0] as f64,
            q_cs[1] as f64,
            q_cs[2] as f64,
            q_cs[3] as f64,
        ];
    }

    pub fn tighten_mount_covariance_deg(&mut self, sigma_deg: f32) {
        let mut p = self.raw.p;
        let var = (sigma_deg as f64).to_radians().powi(2) as f32;
        for i in 21..24 {
            for j in 0..24 {
                p[i][j] = 0.0;
                p[j][i] = 0.0;
            }
            p[i][i] = var;
        }
        self.set_covariance(p);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn init_from_reference_state(
        &mut self,
        q_bn: [f32; 4],
        pos_ned_m: [f32; 3],
        vel_ned_mps: [f32; 3],
        gyro_bias_radps: [f32; 3],
        accel_bias_mps2: [f32; 3],
        gyro_scale: [f32; 3],
        accel_scale: [f32; 3],
        q_cs: [f32; 4],
        p_diag: Option<[f32; 24]>,
    ) {
        self.raw.nominal.q0 = q_bn[0];
        self.raw.nominal.q1 = q_bn[1];
        self.raw.nominal.q2 = q_bn[2];
        self.raw.nominal.q3 = q_bn[3];
        self.raw.nominal.vn = vel_ned_mps[0];
        self.raw.nominal.ve = vel_ned_mps[1];
        self.raw.nominal.vd = vel_ned_mps[2];
        self.raw.nominal.pn = pos_ned_m[0];
        self.raw.nominal.pe = pos_ned_m[1];
        self.raw.nominal.pd = pos_ned_m[2];
        self.raw.pos_e64 = [
            pos_ned_m[0] as f64,
            pos_ned_m[1] as f64,
            pos_ned_m[2] as f64,
        ];
        self.raw.nominal.bgx = gyro_bias_radps[0];
        self.raw.nominal.bgy = gyro_bias_radps[1];
        self.raw.nominal.bgz = gyro_bias_radps[2];
        self.raw.nominal.bax = accel_bias_mps2[0];
        self.raw.nominal.bay = accel_bias_mps2[1];
        self.raw.nominal.baz = accel_bias_mps2[2];
        self.raw.nominal.sgx = gyro_scale[0];
        self.raw.nominal.sgy = gyro_scale[1];
        self.raw.nominal.sgz = gyro_scale[2];
        self.raw.nominal.sax = accel_scale[0];
        self.raw.nominal.say = accel_scale[1];
        self.raw.nominal.saz = accel_scale[2];
        self.raw.nominal.qcs0 = q_cs[0];
        self.raw.nominal.qcs1 = q_cs[1];
        self.raw.nominal.qcs2 = q_cs[2];
        self.raw.nominal.qcs3 = q_cs[3];
        self.raw.qcs64 = [
            q_cs[0] as f64,
            q_cs[1] as f64,
            q_cs[2] as f64,
            q_cs[3] as f64,
        ];
        if let Some(p_diag) = p_diag {
            for (i, value) in p_diag.into_iter().enumerate() {
                self.raw.p[i][i] = value;
            }
        }
        for i in 0..24 {
            for j in 0..24 {
                self.raw.p64[i][j] = self.raw.p[i][j] as f64;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn init_from_reference_ecef_state(
        &mut self,
        q_es: [f32; 4],
        pos_ecef_m: [f64; 3],
        vel_ecef_mps: [f32; 3],
        gyro_bias_radps: [f32; 3],
        accel_bias_mps2: [f32; 3],
        gyro_scale: [f32; 3],
        accel_scale: [f32; 3],
        q_cs: [f32; 4],
        p_diag: Option<[f32; 24]>,
    ) {
        self.init_from_reference_state(
            q_es,
            [
                pos_ecef_m[0] as f32,
                pos_ecef_m[1] as f32,
                pos_ecef_m[2] as f32,
            ],
            vel_ecef_mps,
            gyro_bias_radps,
            accel_bias_mps2,
            gyro_scale,
            accel_scale,
            q_cs,
            p_diag,
        );
        self.raw.pos_e64 = pos_ecef_m;
    }

    pub fn init_seeded_vehicle_from_nav_ecef_state(
        &mut self,
        yaw_rad: f32,
        lat_deg: f64,
        lon_deg: f64,
        pos_ecef_m: [f64; 3],
        vel_ecef_mps: [f32; 3],
        p_diag: Option<[f32; 24]>,
        residual_mount_sigma_deg: Option<f32>,
    ) {
        let (q_es, q_cs) = loose_seeded_vehicle_ecef_split(yaw_rad, lat_deg, lon_deg);
        self.init_from_reference_ecef_state(
            q_es,
            pos_ecef_m,
            vel_ecef_mps,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            q_cs,
            p_diag,
        );
        if let Some(sigma_deg) = residual_mount_sigma_deg {
            self.tighten_mount_covariance_deg(sigma_deg);
        }
    }

    pub fn predict(&mut self, imu: CLooseImuDelta) {
        unsafe { sf_loose_predict(&mut self.raw as *mut CLoose, &imu as *const CLooseImuDelta) };
    }

    pub fn fuse_gps_reference(
        &mut self,
        pos_ecef_m: [f64; 3],
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        speed_acc_mps: f32,
        dt_since_last_gnss_s: f32,
    ) {
        let vel_ptr = vel_ecef_mps
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(core::ptr::null());
        unsafe {
            sf_loose_fuse_gps_reference(
                &mut self.raw as *mut CLoose,
                pos_ecef_m.as_ptr(),
                vel_ptr,
                h_acc_m,
                speed_acc_mps,
                dt_since_last_gnss_s,
            )
        };
    }

    pub fn fuse_gps_reference_full(
        &mut self,
        pos_ecef_m: [f64; 3],
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        vel_std_ned_mps: Option<[f32; 3]>,
        dt_since_last_gnss_s: f32,
    ) {
        let vel_ptr = vel_ecef_mps
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(core::ptr::null());
        let vel_std_ptr = vel_std_ned_mps
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(core::ptr::null());
        unsafe {
            sf_loose_fuse_gps_reference_full(
                &mut self.raw as *mut CLoose,
                pos_ecef_m.as_ptr(),
                vel_ptr,
                h_acc_m,
                vel_std_ptr,
                dt_since_last_gnss_s,
            )
        };
    }

    pub fn fuse_nhc_reference(&mut self, gyro_radps: [f32; 3], accel_mps2: [f32; 3], dt_s: f32) {
        unsafe {
            sf_loose_fuse_nhc_reference(
                &mut self.raw as *mut CLoose,
                gyro_radps.as_ptr(),
                accel_mps2.as_ptr(),
                dt_s,
            )
        };
    }

    pub fn fuse_reference_batch(
        &mut self,
        pos_ecef_m: Option<[f64; 3]>,
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        speed_acc_mps: f32,
        dt_since_last_gnss_s: f32,
        gyro_radps: [f32; 3],
        accel_mps2: [f32; 3],
        dt_s: f32,
    ) {
        let pos_ptr = pos_ecef_m
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(core::ptr::null());
        let vel_ptr = vel_ecef_mps
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(core::ptr::null());
        unsafe {
            sf_loose_fuse_reference_batch(
                &mut self.raw as *mut CLoose,
                pos_ptr,
                vel_ptr,
                h_acc_m,
                speed_acc_mps,
                dt_since_last_gnss_s,
                gyro_radps.as_ptr(),
                accel_mps2.as_ptr(),
                dt_s,
            )
        };
    }

    pub fn fuse_reference_batch_full(
        &mut self,
        pos_ecef_m: Option<[f64; 3]>,
        vel_ecef_mps: Option<[f32; 3]>,
        h_acc_m: f32,
        vel_std_ned_mps: Option<[f32; 3]>,
        dt_since_last_gnss_s: f32,
        gyro_radps: [f32; 3],
        accel_mps2: [f32; 3],
        dt_s: f32,
    ) {
        let pos_ptr = pos_ecef_m
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(core::ptr::null());
        let vel_ptr = vel_ecef_mps
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(core::ptr::null());
        let vel_std_ptr = vel_std_ned_mps
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(core::ptr::null());
        unsafe {
            sf_loose_fuse_reference_batch_full(
                &mut self.raw as *mut CLoose,
                pos_ptr,
                vel_ptr,
                h_acc_m,
                vel_std_ptr,
                dt_since_last_gnss_s,
                gyro_radps.as_ptr(),
                accel_mps2.as_ptr(),
                dt_s,
            )
        };
    }

    pub fn compute_error_transition(
        &self,
        imu: CLooseImuDelta,
    ) -> ([[f32; 24]; 24], [[f32; 21]; 24]) {
        let mut f = [[0.0_f32; 24]; 24];
        let mut g = [[0.0_f32; 21]; 24];
        unsafe {
            sf_loose_compute_error_transition(
                &mut f as *mut [[f32; 24]; 24],
                &mut g as *mut [[f32; 21]; 24],
                &self.raw as *const CLoose,
                &imu as *const CLooseImuDelta,
            );
        }
        (f, g)
    }
}

pub fn loose_seeded_vehicle_ecef_split(
    yaw_rad: f32,
    lat_deg: f64,
    lon_deg: f64,
) -> ([f32; 4], [f32; 4]) {
    let half_yaw = 0.5 * yaw_rad as f64;
    let q_ns = [half_yaw.cos(), 0.0, 0.0, half_yaw.sin()];
    let q_es = quat_mul_f64(quat_conj_f64(quat_ecef_to_ned_f64(lat_deg, lon_deg)), q_ns);
    (
        [
            q_es[0] as f32,
            q_es[1] as f32,
            q_es[2] as f32,
            q_es[3] as f32,
        ],
        [1.0, 0.0, 0.0, 0.0],
    )
}

fn quat_conj_f64(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn quat_mul_f64(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_ecef_to_ned_f64(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    let lon = lon_deg.to_radians();
    let lat = lat_deg.to_radians();
    let half_lon = 0.5 * lon;
    let q_lon = [half_lon.cos(), 0.0, 0.0, -half_lon.sin()];
    let half_lat = 0.5 * (lat + 0.5 * core::f64::consts::PI);
    let q_lat = [half_lat.cos(), 0.0, half_lat.sin(), 0.0];
    quat_mul_f64(q_lat, q_lon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn loose_predict_noise_ffi_layout_is_seven_f32s() {
        assert_eq!(size_of::<LoosePredictNoise>(), 7 * size_of::<f32>());
        assert_eq!(align_of::<LoosePredictNoise>(), align_of::<f32>());
    }

    #[test]
    fn eskf_predict_noise_ffi_layout_is_five_f32s() {
        assert_eq!(size_of::<PredictNoise>(), 5 * size_of::<f32>());
        assert_eq!(align_of::<PredictNoise>(), align_of::<f32>());
    }

    #[test]
    fn eskf_wrapper_passes_mount_noise_to_c_state() {
        let noise = PredictNoise {
            gyro_var: 0.11,
            accel_var: 0.22,
            gyro_bias_rw_var: 0.33,
            accel_bias_rw_var: 0.44,
            mount_align_rw_var: 0.55,
        };

        let wrapper = CEskfWrapper::new(noise);

        assert_eq!(wrapper.raw.noise.gyro_var, noise.gyro_var);
        assert_eq!(wrapper.raw.noise.accel_var, noise.accel_var);
        assert_eq!(wrapper.raw.noise.gyro_bias_rw_var, noise.gyro_bias_rw_var);
        assert_eq!(wrapper.raw.noise.accel_bias_rw_var, noise.accel_bias_rw_var);
        assert_eq!(
            wrapper.raw.noise.mount_align_rw_var,
            noise.mount_align_rw_var
        );
        assert_eq!(wrapper.raw.nominal.qcs0, 1.0);
    }

    #[test]
    fn loose_wrapper_passes_all_noise_fields_to_c_state() {
        let noise = LoosePredictNoise {
            gyro_var: 0.11,
            accel_var: 0.22,
            gyro_bias_rw_var: 0.33,
            accel_bias_rw_var: 0.44,
            gyro_scale_rw_var: 0.55,
            accel_scale_rw_var: 0.66,
            mount_align_rw_var: 0.77,
        };

        let wrapper = CLooseWrapper::new(noise);

        assert_eq!(wrapper.raw.noise.gyro_var, noise.gyro_var);
        assert_eq!(wrapper.raw.noise.accel_var, noise.accel_var);
        assert_eq!(wrapper.raw.noise.gyro_bias_rw_var, noise.gyro_bias_rw_var);
        assert_eq!(wrapper.raw.noise.accel_bias_rw_var, noise.accel_bias_rw_var);
        assert_eq!(wrapper.raw.noise.gyro_scale_rw_var, noise.gyro_scale_rw_var);
        assert_eq!(
            wrapper.raw.noise.accel_scale_rw_var,
            noise.accel_scale_rw_var
        );
        assert_eq!(
            wrapper.raw.noise.mount_align_rw_var,
            noise.mount_align_rw_var
        );
    }
}
