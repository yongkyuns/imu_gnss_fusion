use crate::align::{Align, AlignConfig, AlignUpdateTrace, AlignWindowSummary};
use crate::c_api::{CAlignUpdateTrace, CEskf, CSensorFusionWrapper};

#[derive(Clone, Copy, Debug)]
pub struct FusionImuSample {
    pub t_s: f32,
    pub gyro_radps: [f32; 3],
    pub accel_mps2: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct FusionGnssSample {
    pub t_s: f32,
    pub lat_deg: f32,
    pub lon_deg: f32,
    pub height_m: f32,
    pub vel_ned_mps: [f32; 3],
    pub pos_std_m: [f32; 3],
    pub vel_std_mps: [f32; 3],
    pub heading_rad: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct FusionVehicleSpeedSample {
    pub t_s: f32,
    pub speed_mps: f32,
    pub direction: FusionVehicleSpeedDirection,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FusionVehicleSpeedDirection {
    #[default]
    Unknown,
    Forward,
    Reverse,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FusionUpdate {
    pub mount_ready: bool,
    pub mount_ready_changed: bool,
    pub ekf_initialized: bool,
    pub ekf_initialized_now: bool,
    pub mount_q_vb: Option<[f32; 4]>,
}

#[derive(Clone, Copy, Debug)]
pub struct FusionAlignDebug {
    pub window: AlignWindowSummary,
    pub trace: AlignUpdateTrace,
}

#[derive(Clone, Copy, Debug)]
pub enum MisalignmentMode {
    InternalAlign,
    External([f32; 4]),
}

#[derive(Debug)]
pub struct SensorFusion {
    raw: CSensorFusionWrapper,
    cached_align: Option<Align>,
}

impl SensorFusion {
    pub fn new() -> Self {
        Self::with_misalignment_mode(MisalignmentMode::InternalAlign)
    }

    pub fn with_misalignment(q_vb: [f32; 4]) -> Self {
        Self::with_misalignment_mode(MisalignmentMode::External(q_vb))
    }

    pub fn with_misalignment_mode(mode: MisalignmentMode) -> Self {
        let raw = match mode {
            MisalignmentMode::InternalAlign => CSensorFusionWrapper::new_internal(),
            MisalignmentMode::External(q_vb) => CSensorFusionWrapper::new_external(q_vb),
        };
        let mut out = Self {
            raw,
            cached_align: None,
        };
        out.refresh_align_snapshot();
        out
    }

    pub fn set_misalignment(&mut self, q_vb: [f32; 4]) {
        self.raw.set_misalignment(q_vb);
        self.refresh_align_snapshot();
    }

    pub fn set_r_body_vel(&mut self, r_body_vel: f32) {
        self.raw.set_r_body_vel(r_body_vel);
        self.refresh_align_snapshot();
    }

    pub fn set_gnss_pos_mount_scale(&mut self, gnss_pos_mount_scale: f32) {
        self.raw.set_gnss_pos_mount_scale(gnss_pos_mount_scale);
        self.refresh_align_snapshot();
    }

    pub fn set_gnss_vel_mount_scale(&mut self, gnss_vel_mount_scale: f32) {
        self.raw.set_gnss_vel_mount_scale(gnss_vel_mount_scale);
        self.refresh_align_snapshot();
    }

    pub fn set_gyro_bias_init_sigma_radps(&mut self, gyro_bias_init_sigma_radps: f32) {
        self.raw
            .set_gyro_bias_init_sigma_radps(gyro_bias_init_sigma_radps);
        self.refresh_align_snapshot();
    }

    pub fn set_accel_bias_init_sigma_mps2(&mut self, accel_bias_init_sigma_mps2: f32) {
        self.raw
            .set_accel_bias_init_sigma_mps2(accel_bias_init_sigma_mps2);
        self.refresh_align_snapshot();
    }

    pub fn set_accel_bias_rw_var(&mut self, accel_bias_rw_var: f32) {
        self.raw.set_accel_bias_rw_var(accel_bias_rw_var);
        self.refresh_align_snapshot();
    }

    pub fn set_mount_align_rw_var(&mut self, mount_align_rw_var: f32) {
        self.raw.set_mount_align_rw_var(mount_align_rw_var);
        self.refresh_align_snapshot();
    }

    pub fn set_mount_update_min_scale(&mut self, mount_update_min_scale: f32) {
        self.raw.set_mount_update_min_scale(mount_update_min_scale);
        self.refresh_align_snapshot();
    }

    pub fn set_mount_update_ramp_time_s(&mut self, mount_update_ramp_time_s: f32) {
        self.raw
            .set_mount_update_ramp_time_s(mount_update_ramp_time_s);
        self.refresh_align_snapshot();
    }

    pub fn set_mount_update_innovation_gate_mps(
        &mut self,
        mount_update_innovation_gate_mps: f32,
    ) {
        self.raw
            .set_mount_update_innovation_gate_mps(mount_update_innovation_gate_mps);
        self.refresh_align_snapshot();
    }

    pub fn set_r_vehicle_speed(&mut self, r_vehicle_speed: f32) {
        self.raw.set_r_vehicle_speed(r_vehicle_speed);
        self.refresh_align_snapshot();
    }

    pub fn set_r_zero_vel(&mut self, r_zero_vel: f32) {
        self.raw.set_r_zero_vel(r_zero_vel);
        self.refresh_align_snapshot();
    }

    pub fn set_r_stationary_accel(&mut self, r_stationary_accel: f32) {
        self.raw.set_r_stationary_accel(r_stationary_accel);
        self.refresh_align_snapshot();
    }

    pub fn process_imu(&mut self, sample: FusionImuSample) -> FusionUpdate {
        let update = self.raw.process_imu(sample);
        self.refresh_align_snapshot();
        update
    }

    pub fn process_gnss(&mut self, gnss: FusionGnssSample) -> FusionUpdate {
        let update = self.raw.process_gnss(gnss);
        self.refresh_align_snapshot();
        update
    }

    pub fn process_vehicle_speed(&mut self, speed: FusionVehicleSpeedSample) -> FusionUpdate {
        let update = self.raw.process_vehicle_speed(speed);
        self.refresh_align_snapshot();
        update
    }

    pub fn eskf(&self) -> Option<&CEskf> {
        self.raw.eskf()
    }

    pub fn mount_q_vb(&self) -> Option<[f32; 4]> {
        self.raw.mount_q_vb()
    }

    pub fn eskf_mount_q_vb(&self) -> Option<[f32; 4]> {
        self.raw.eskf_mount_q_vb()
    }

    pub fn anchor_lla_debug(&self) -> Option<[f32; 3]> {
        self.raw.anchor_lla_debug()
    }

    pub fn mount_ready(&self) -> bool {
        self.raw.mount_ready()
    }

    pub fn position_lla(&self) -> Option<[f32; 3]> {
        self.raw.position_lla()
    }

    pub fn align(&self) -> Option<&Align> {
        self.cached_align.as_ref()
    }

    pub fn align_debug(&self) -> Option<FusionAlignDebug> {
        let window = self.raw.align_window_debug()?;
        let trace = self.raw.align_trace_debug()?;
        Some(FusionAlignDebug {
            window: AlignWindowSummary {
                dt: window.dt,
                mean_gyro_b: window.mean_gyro_b,
                mean_accel_b: window.mean_accel_b,
                gnss_vel_prev_n: window.gnss_vel_prev_n,
                gnss_vel_curr_n: window.gnss_vel_curr_n,
            },
            trace: convert_align_trace(*trace),
        })
    }

    fn refresh_align_snapshot(&mut self) {
        self.cached_align = self
            .raw
            .align_state()
            .map(|s| Align::from_c_state(AlignConfig::default(), *s));
    }
}

fn convert_align_trace(trace: CAlignUpdateTrace) -> AlignUpdateTrace {
    AlignUpdateTrace {
        q_start: trace.q_start,
        coarse_alignment_ready: trace.coarse_alignment_ready,
        after_gravity: trace.after_gravity_valid.then_some(trace.after_gravity),
        after_gravity_quasi_static: trace.after_gravity_quasi_static,
        after_horiz_accel: trace
            .after_horiz_accel_valid
            .then_some(trace.after_horiz_accel),
        horiz_angle_err_rad: trace
            .horiz_angle_err_rad_valid
            .then_some(trace.horiz_angle_err_rad),
        horiz_effective_std_rad: trace
            .horiz_effective_std_rad_valid
            .then_some(trace.horiz_effective_std_rad),
        horiz_gnss_norm_mps2: trace
            .horiz_gnss_norm_mps2_valid
            .then_some(trace.horiz_gnss_norm_mps2),
        horiz_imu_norm_mps2: trace
            .horiz_imu_norm_mps2_valid
            .then_some(trace.horiz_imu_norm_mps2),
        horiz_obs_accel_vx: trace
            .horiz_obs_accel_vx_valid
            .then_some(trace.horiz_obs_accel_vx),
        horiz_obs_accel_vy: trace
            .horiz_obs_accel_vy_valid
            .then_some(trace.horiz_obs_accel_vy),
        horiz_accel_bx: trace.horiz_accel_bx_valid.then_some(trace.horiz_accel_bx),
        horiz_accel_by: trace.horiz_accel_by_valid.then_some(trace.horiz_accel_by),
        horiz_speed_q: trace.horiz_speed_q_valid.then_some(trace.horiz_speed_q),
        horiz_accel_q: trace.horiz_accel_q_valid.then_some(trace.horiz_accel_q),
        horiz_straight_q: trace
            .horiz_straight_q_valid
            .then_some(trace.horiz_straight_q),
        horiz_turn_q: trace.horiz_turn_q_valid.then_some(trace.horiz_turn_q),
        horiz_dominance_q: trace
            .horiz_dominance_q_valid
            .then_some(trace.horiz_dominance_q),
        horiz_turn_core_valid: trace.horiz_turn_core_valid,
        horiz_straight_core_valid: trace.horiz_straight_core_valid,
        after_turn_gyro: trace.after_turn_gyro_valid.then_some(trace.after_turn_gyro),
    }
}
