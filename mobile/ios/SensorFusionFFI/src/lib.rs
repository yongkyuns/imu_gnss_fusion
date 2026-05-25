//! C ABI wrapper for the Rust `sensor_fusion` facade.

use core::ptr;

use road_events::{
    HarshAccelDetector, HarshBehaviorPreset, HarshBrakeDetector, HarshCornerDetector,
    HarshCornerSample, HarshLongitudinalSample, HillConfig, HillDetector, HillKind, HillSample,
    ReverseConfig, ReverseDetector, ReverseSample, SpeedBumpConfig, SpeedBumpDetector,
    SpeedBumpSample, TripEventKind, TripSample, TripStats,
};
use sensor_fusion::{GnssSample, ImuSample, SensorFusion, Update};

const ROAD_EVENT_HARSH_ACCELERATION: u32 = 1;
const ROAD_EVENT_HARSH_BRAKING: u32 = 2;
const ROAD_EVENT_HARSH_CORNERING: u32 = 3;
const ROAD_EVENT_REVERSE: u32 = 4;
const ROAD_EVENT_SPEED_BUMP: u32 = 5;
const ROAD_EVENT_UPHILL: u32 = 6;
const ROAD_EVENT_DOWNHILL: u32 = 7;
const SENSOR_FUSION_HARSH_BEHAVIOR_SENSITIVE: u32 = 1;
const SENSOR_FUSION_HARSH_BEHAVIOR_BALANCED: u32 = 2;
const SENSOR_FUSION_HARSH_BEHAVIOR_CONSERVATIVE: u32 = 3;

/// Opaque fusion handle owned by Rust and passed across the C ABI as a pointer.
pub struct SensorFusionFfi {
    inner: SensorFusion,
    last_update: SensorFusionFfiUpdate,
    road_events: RoadEventDetectors,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SensorFusionFfiUpdate {
    pub mount_ready: bool,
    pub mount_ready_changed: bool,
    pub ekf_initialized: bool,
    pub ekf_initialized_now: bool,
    pub filter_initialized: bool,
    pub filter_initialized_now: bool,
    pub mount_q_bv_valid: bool,
    pub mount_q_bv: [f32; 4],
}

impl Default for SensorFusionFfiUpdate {
    fn default() -> Self {
        Self {
            mount_ready: false,
            mount_ready_changed: false,
            ekf_initialized: false,
            ekf_initialized_now: false,
            filter_initialized: false,
            filter_initialized_now: false,
            mount_q_bv_valid: false,
            mount_q_bv: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

impl From<Update> for SensorFusionFfiUpdate {
    fn from(update: Update) -> Self {
        Self {
            mount_ready: update.mount_ready,
            mount_ready_changed: update.mount_ready_changed,
            ekf_initialized: update.ekf_initialized,
            ekf_initialized_now: update.ekf_initialized_now,
            filter_initialized: update.ekf_initialized,
            filter_initialized_now: update.ekf_initialized_now,
            mount_q_bv_valid: update.mount_q_bv.is_some(),
            mount_q_bv: update.mount_q_bv.unwrap_or([1.0, 0.0, 0.0, 0.0]),
        }
    }
}

impl SensorFusionFfiUpdate {
    fn from_fusion_state(fusion: &SensorFusion) -> Self {
        let filter_initialized = fusion.ekf().is_some();
        let mount_q_bv = fusion.mount_q_bv();

        Self {
            mount_ready: fusion.mount_ready(),
            ekf_initialized: fusion.ekf().is_some(),
            filter_initialized,
            mount_q_bv_valid: mount_q_bv.is_some(),
            mount_q_bv: mount_q_bv.unwrap_or([1.0, 0.0, 0.0, 0.0]),
            ..Self::default()
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SensorFusionFfiEkfSnapshot {
    pub mount_ready: bool,
    pub initialized: bool,
    pub q0: f32,
    pub q1: f32,
    pub q2: f32,
    pub q3: f32,
    pub vel_n_mps: f32,
    pub vel_e_mps: f32,
    pub vel_d_mps: f32,
    pub pos_n_m: f32,
    pub pos_e_m: f32,
    pub pos_d_m: f32,
    pub gyro_bias_x_radps: f32,
    pub gyro_bias_y_radps: f32,
    pub gyro_bias_z_radps: f32,
    pub accel_bias_x_mps2: f32,
    pub accel_bias_y_mps2: f32,
    pub accel_bias_z_mps2: f32,
    pub q_bv0: f32,
    pub q_bv1: f32,
    pub q_bv2: f32,
    pub q_bv3: f32,
    pub position_lla_valid: bool,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
}

impl Default for SensorFusionFfiEkfSnapshot {
    fn default() -> Self {
        Self {
            mount_ready: false,
            initialized: false,
            q0: 1.0,
            q1: 0.0,
            q2: 0.0,
            q3: 0.0,
            vel_n_mps: 0.0,
            vel_e_mps: 0.0,
            vel_d_mps: 0.0,
            pos_n_m: 0.0,
            pos_e_m: 0.0,
            pos_d_m: 0.0,
            gyro_bias_x_radps: 0.0,
            gyro_bias_y_radps: 0.0,
            gyro_bias_z_radps: 0.0,
            accel_bias_x_mps2: 0.0,
            accel_bias_y_mps2: 0.0,
            accel_bias_z_mps2: 0.0,
            q_bv0: 1.0,
            q_bv1: 0.0,
            q_bv2: 0.0,
            q_bv3: 0.0,
            position_lla_valid: false,
            lat_deg: 0.0,
            lon_deg: 0.0,
            height_m: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SensorFusionFfiAlignProgress {
    pub valid: bool,
    pub coarse_ready: bool,
    pub roll_sigma_deg: f32,
    pub pitch_sigma_deg: f32,
    pub yaw_sigma_deg: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SensorFusionFfiRoadEvent {
    pub kind: u32,
    pub t_s: f32,
    pub start_t_s: f32,
    pub end_t_s: f32,
    pub duration_s: f32,
    pub value: f32,
    pub confidence: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SensorFusionFfiTripSummary {
    pub sample_count: u32,
    pub invalid_sample_count: u32,
    pub data_gap_count: u32,
    pub max_sample_gap_s: f32,
    pub total_gap_duration_s: f32,
    pub duration_s: f32,
    pub moving_duration_s: f32,
    pub stationary_duration_s: f32,
    pub distance_m: f32,
    pub reverse_duration_s: f32,
    pub reverse_distance_m: f32,
    pub uphill_distance_m: f32,
    pub downhill_distance_m: f32,
    pub elevation_gain_m: f32,
    pub elevation_loss_m: f32,
    pub mean_speed_mps: f32,
    pub moving_mean_speed_mps: f32,
    pub peak_speed_mps: f32,
    pub peak_accel_mps2: f32,
    pub peak_decel_mps2: f32,
    pub peak_lateral_accel_mps2: f32,
    pub rolling_speed_mps: f32,
    pub rolling_abs_longitudinal_accel_mps2: f32,
    pub rolling_abs_lateral_accel_mps2: f32,
    pub speed_bumps: u32,
    pub uphill_events: u32,
    pub downhill_events: u32,
    pub reverse_events: u32,
    pub harsh_acceleration_events: u32,
    pub harsh_braking_events: u32,
    pub harsh_cornering_events: u32,
    pub speed_bumps_per_km: f32,
    pub harsh_events_per_km: f32,
    pub reverse_seconds_per_km: f32,
}

#[derive(Clone, Debug)]
struct RoadEventDetectors {
    speed_bump: SpeedBumpDetector,
    hill: HillDetector,
    reverse: ReverseDetector,
    harsh_accel: HarshAccelDetector,
    harsh_brake: HarshBrakeDetector,
    harsh_corner: HarshCornerDetector,
    trip_stats: TripStats,
    harsh_behavior_preset: HarshBehaviorPreset,
}

impl RoadEventDetectors {
    fn new() -> Self {
        Self::new_with_harsh_behavior(HarshBehaviorPreset::Balanced)
    }

    fn new_with_harsh_behavior(harsh_behavior_preset: HarshBehaviorPreset) -> Self {
        let harsh_behavior = harsh_behavior_preset.configs();
        Self {
            speed_bump: SpeedBumpDetector::new(SpeedBumpConfig::default()),
            hill: HillDetector::new(HillConfig::default()),
            reverse: ReverseDetector::new(ReverseConfig::default()),
            harsh_accel: HarshAccelDetector::new(harsh_behavior.accel),
            harsh_brake: HarshBrakeDetector::new(harsh_behavior.brake),
            harsh_corner: HarshCornerDetector::new(harsh_behavior.corner),
            trip_stats: TripStats::default(),
            harsh_behavior_preset,
        }
    }

    fn reset(&mut self) {
        *self = Self::new_with_harsh_behavior(self.harsh_behavior_preset);
    }

    fn set_harsh_behavior_preset(&mut self, preset: HarshBehaviorPreset) {
        if preset == self.harsh_behavior_preset {
            return;
        }
        let harsh_behavior = preset.configs();
        self.harsh_accel = HarshAccelDetector::new(harsh_behavior.accel);
        self.harsh_brake = HarshBrakeDetector::new(harsh_behavior.brake);
        self.harsh_corner = HarshCornerDetector::new(harsh_behavior.corner);
        self.harsh_behavior_preset = preset;
    }
}

impl SensorFusionFfi {
    fn new(inner: SensorFusion) -> Self {
        let last_update = SensorFusionFfiUpdate::from_fusion_state(&inner);
        Self {
            inner,
            last_update,
            road_events: RoadEventDetectors::new(),
        }
    }

    fn status(&self) -> SensorFusionFfiUpdate {
        let mut status = SensorFusionFfiUpdate::from_fusion_state(&self.inner);
        status.mount_ready_changed = self.last_update.mount_ready_changed;
        status.ekf_initialized_now = self.last_update.ekf_initialized_now;
        status.filter_initialized_now = self.last_update.filter_initialized_now;
        status
    }

    fn reset(&mut self, inner: SensorFusion) {
        self.inner = inner;
        self.last_update = SensorFusionFfiUpdate::from_fusion_state(&self.inner);
        self.road_events.reset();
    }

    fn store_update(&mut self, update: Update) -> SensorFusionFfiUpdate {
        self.last_update = update.into();
        self.status()
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions.
pub unsafe extern "C" fn sensor_fusion_set_harsh_behavior_preset(
    handle: *mut SensorFusionFfi,
    preset: u32,
) -> bool {
    let Some(fusion) = fusion_mut(handle) else {
        return false;
    };
    let Some(preset) = harsh_behavior_preset_from_ffi(preset) else {
        return false;
    };
    fusion.road_events.set_harsh_behavior_preset(preset);
    true
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions. When `out` is non-null, it must point to writable memory
/// for at least `max_events` `SensorFusionFfiRoadEvent` values.
pub unsafe extern "C" fn sensor_fusion_process_road_event_motion(
    handle: *mut SensorFusionFfi,
    t_s: f32,
    forward_velocity_mps: f32,
    ground_speed_mps: f32,
    longitudinal_accel_mps2: f32,
    longitudinal_accel_valid: bool,
    _yaw_rate_radps: f32,
    _yaw_rate_valid: bool,
    pitch_deg: f32,
    pitch_valid: bool,
    lateral_accel_mps2: f32,
    lateral_accel_valid: bool,
    vertical_accel_mps2: f32,
    vertical_accel_valid: bool,
    out: *mut SensorFusionFfiRoadEvent,
    max_events: usize,
) -> usize {
    let Some(fusion) = fusion_mut(handle) else {
        return 0;
    };
    if out.is_null() || max_events == 0 || !t_s.is_finite() {
        return 0;
    }

    let mut writer = RoadEventWriter {
        out,
        max_events,
        len: 0,
    };

    fusion.road_events.trip_stats.update_motion(TripSample {
        t_s,
        speed_mps: ground_speed_mps.max(0.0),
        forward_velocity_mps,
        height_m: None,
        height_frame_id: 0,
        longitudinal_accel_mps2: if longitudinal_accel_valid {
            longitudinal_accel_mps2
        } else {
            0.0
        },
        lateral_accel_mps2: if lateral_accel_valid {
            lateral_accel_mps2
        } else {
            0.0
        },
    });

    let longitudinal = HarshLongitudinalSample {
        t_s,
        forward_velocity_mps,
    };
    if let Some(event) = fusion.road_events.harsh_accel.update(longitudinal) {
        fusion
            .road_events
            .trip_stats
            .record_event(TripEventKind::HarshAcceleration);
        writer.push(SensorFusionFfiRoadEvent {
            kind: ROAD_EVENT_HARSH_ACCELERATION,
            t_s: event.end_t_s,
            start_t_s: event.start_t_s,
            end_t_s: event.end_t_s,
            duration_s: event.duration_s,
            value: event.peak_accel_mps2,
            confidence: 0.9,
        });
    }
    if let Some(event) = fusion.road_events.harsh_brake.update(longitudinal) {
        fusion
            .road_events
            .trip_stats
            .record_event(TripEventKind::HarshBraking);
        writer.push(SensorFusionFfiRoadEvent {
            kind: ROAD_EVENT_HARSH_BRAKING,
            t_s: event.end_t_s,
            start_t_s: event.start_t_s,
            end_t_s: event.end_t_s,
            duration_s: event.duration_s,
            value: event.peak_accel_mps2,
            confidence: 0.9,
        });
    }
    if let Some(event) = fusion.road_events.reverse.update(ReverseSample {
        t_s,
        forward_velocity_mps,
    }) {
        fusion
            .road_events
            .trip_stats
            .record_event(TripEventKind::Reverse);
        writer.push(SensorFusionFfiRoadEvent {
            kind: ROAD_EVENT_REVERSE,
            t_s: event.end_t_s,
            start_t_s: event.start_t_s,
            end_t_s: event.end_t_s,
            duration_s: event.duration_s,
            value: event.peak_reverse_speed_mps,
            confidence: 0.9,
        });
    }
    if lateral_accel_valid {
        if let Some(event) = fusion.road_events.harsh_corner.update(HarshCornerSample {
            t_s,
            speed_mps: ground_speed_mps,
            lateral_accel_mps2,
        }) {
            fusion
                .road_events
                .trip_stats
                .record_event(TripEventKind::HarshCornering);
            writer.push(SensorFusionFfiRoadEvent {
                kind: ROAD_EVENT_HARSH_CORNERING,
                t_s: event.end_t_s,
                start_t_s: event.start_t_s,
                end_t_s: event.end_t_s,
                duration_s: event.duration_s,
                value: event.peak_lateral_accel_mps2,
                confidence: 0.9,
            });
        }
    }
    if pitch_valid {
        if let Some(event) = fusion.road_events.hill.update(HillSample {
            t_s,
            speed_mps: ground_speed_mps,
            pitch_deg,
        }) {
            fusion
                .road_events
                .trip_stats
                .record_event(match event.kind {
                    HillKind::Uphill => TripEventKind::Uphill,
                    HillKind::Downhill => TripEventKind::Downhill,
                });
            writer.push(SensorFusionFfiRoadEvent {
                kind: match event.kind {
                    HillKind::Uphill => ROAD_EVENT_UPHILL,
                    HillKind::Downhill => ROAD_EVENT_DOWNHILL,
                },
                t_s: event.end_t_s,
                start_t_s: event.start_t_s,
                end_t_s: event.end_t_s,
                duration_s: event.duration_s,
                value: event.mean_pitch_deg,
                confidence: 0.9,
            });
        }
    }
    if pitch_valid && vertical_accel_valid {
        let (_, event) = fusion.road_events.speed_bump.update(SpeedBumpSample {
            t_s,
            speed_mps: ground_speed_mps,
            pitch_deg,
            vertical_accel_mps2,
        });
        if let Some(event) = event {
            fusion
                .road_events
                .trip_stats
                .record_event(TripEventKind::SpeedBump);
            writer.push(SensorFusionFfiRoadEvent {
                kind: ROAD_EVENT_SPEED_BUMP,
                t_s: event.t_s,
                start_t_s: event.t_s - event.duration_s * 0.5,
                end_t_s: event.t_s + event.duration_s * 0.5,
                duration_s: event.duration_s,
                value: event.peak_abs_pitch_deg,
                confidence: event.confidence,
            });
        }
    }

    writer.len
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions. `out` must be either null or point to writable memory for
/// one `SensorFusionFfiTripSummary`.
pub unsafe extern "C" fn sensor_fusion_snapshot_trip_summary(
    handle: *const SensorFusionFfi,
    out: *mut SensorFusionFfiTripSummary,
) -> bool {
    let Some(fusion) = fusion_ref(handle) else {
        return false;
    };
    if out.is_null() {
        return false;
    }
    let summary = fusion.road_events.trip_stats.summary();
    unsafe {
        ptr::write(
            out,
            SensorFusionFfiTripSummary {
                sample_count: summary.sample_count,
                invalid_sample_count: summary.invalid_sample_count,
                data_gap_count: summary.data_gap_count,
                max_sample_gap_s: summary.max_sample_gap_s,
                total_gap_duration_s: summary.total_gap_duration_s,
                duration_s: summary.duration_s,
                moving_duration_s: summary.moving_duration_s,
                stationary_duration_s: summary.stationary_duration_s,
                distance_m: summary.distance_m,
                reverse_duration_s: summary.reverse_duration_s,
                reverse_distance_m: summary.reverse_distance_m,
                uphill_distance_m: 0.0,
                downhill_distance_m: 0.0,
                elevation_gain_m: summary.elevation_gain_m,
                elevation_loss_m: summary.elevation_loss_m,
                mean_speed_mps: summary.mean_speed_mps,
                moving_mean_speed_mps: summary.moving_mean_speed_mps,
                peak_speed_mps: summary.peak_speed_mps,
                peak_accel_mps2: summary.peak_accel_mps2,
                peak_decel_mps2: summary.peak_decel_mps2,
                peak_lateral_accel_mps2: summary.peak_lateral_accel_mps2,
                rolling_speed_mps: summary.rolling_speed_mps,
                rolling_abs_longitudinal_accel_mps2: summary.rolling_abs_longitudinal_accel_mps2,
                rolling_abs_lateral_accel_mps2: summary.rolling_abs_lateral_accel_mps2,
                speed_bumps: summary.events.speed_bumps,
                uphill_events: summary.events.uphill,
                downhill_events: summary.events.downhill,
                reverse_events: summary.events.reverse,
                harsh_acceleration_events: summary.events.harsh_acceleration,
                harsh_braking_events: summary.events.harsh_braking,
                harsh_cornering_events: summary.events.harsh_cornering,
                speed_bumps_per_km: summary.speed_bumps_per_km,
                harsh_events_per_km: summary.harsh_events_per_km,
                reverse_seconds_per_km: summary.reverse_seconds_per_km,
            },
        );
    }
    true
}

struct RoadEventWriter {
    out: *mut SensorFusionFfiRoadEvent,
    max_events: usize,
    len: usize,
}

impl RoadEventWriter {
    fn push(&mut self, event: SensorFusionFfiRoadEvent) {
        if self.len >= self.max_events {
            return;
        }
        unsafe {
            ptr::write(self.out.add(self.len), event);
        }
        self.len += 1;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sensor_fusion_create_ekf_auto() -> *mut SensorFusionFfi {
    Box::into_raw(Box::new(SensorFusionFfi::new(SensorFusion::new())))
}

#[unsafe(no_mangle)]
pub extern "C" fn sensor_fusion_create_ekf_manual(
    qw: f32,
    qx: f32,
    qy: f32,
    qz: f32,
) -> *mut SensorFusionFfi {
    Box::into_raw(Box::new(SensorFusionFfi::new(SensorFusion::with_mount([
        qw, qx, qy, qz,
    ]))))
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a pointer returned by this crate's create
/// functions that has not already been destroyed.
pub unsafe extern "C" fn sensor_fusion_destroy(handle: *mut SensorFusionFfi) {
    if handle.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(handle));
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions.
pub unsafe extern "C" fn sensor_fusion_reset_ekf_auto(handle: *mut SensorFusionFfi) {
    let Some(fusion) = fusion_mut(handle) else {
        return;
    };
    fusion.reset(SensorFusion::new());
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions.
pub unsafe extern "C" fn sensor_fusion_reset_ekf_manual(
    handle: *mut SensorFusionFfi,
    qw: f32,
    qx: f32,
    qy: f32,
    qz: f32,
) {
    let Some(fusion) = fusion_mut(handle) else {
        return;
    };
    fusion.reset(SensorFusion::with_mount([qw, qx, qy, qz]));
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions.
pub unsafe extern "C" fn sensor_fusion_snapshot_status(
    handle: *const SensorFusionFfi,
) -> SensorFusionFfiUpdate {
    let Some(fusion) = fusion_ref(handle) else {
        return SensorFusionFfiUpdate::default();
    };
    fusion.status()
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions.
pub unsafe extern "C" fn sensor_fusion_process_imu(
    handle: *mut SensorFusionFfi,
    t_s: f32,
    ax: f32,
    ay: f32,
    az: f32,
    gx: f32,
    gy: f32,
    gz: f32,
) -> SensorFusionFfiUpdate {
    let Some(fusion) = fusion_mut(handle) else {
        return SensorFusionFfiUpdate::default();
    };

    let update = fusion.inner.process_imu(ImuSample {
        t_s,
        gyro_radps: [gx, gy, gz],
        accel_mps2: [ax, ay, az],
    });
    fusion.store_update(update)
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions.
pub unsafe extern "C" fn sensor_fusion_process_gnss(
    handle: *mut SensorFusionFfi,
    t_s: f32,
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    vn: f32,
    ve: f32,
    vd: f32,
    pos_std_n: f32,
    pos_std_e: f32,
    pos_std_d: f32,
    vel_std_n: f32,
    vel_std_e: f32,
    vel_std_d: f32,
    heading_rad: f32,
    is_heading_valid: bool,
) -> SensorFusionFfiUpdate {
    let Some(fusion) = fusion_mut(handle) else {
        return SensorFusionFfiUpdate::default();
    };

    let update = fusion.inner.process_gnss(GnssSample {
        t_s,
        lat_deg,
        lon_deg,
        height_m,
        vel_ned_mps: [vn, ve, vd],
        pos_std_m: [pos_std_n, pos_std_e, pos_std_d],
        vel_std_mps: [vel_std_n, vel_std_e, vel_std_d],
        heading_rad: is_heading_valid.then_some(heading_rad),
    });
    fusion.store_update(update)
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions. When non-null, `out` must point to writable memory for one
/// `SensorFusionFfiEkfSnapshot`.
pub unsafe extern "C" fn sensor_fusion_snapshot_ekf(
    handle: *const SensorFusionFfi,
    out: *mut SensorFusionFfiEkfSnapshot,
) -> bool {
    if out.is_null() {
        return false;
    }

    let snapshot = ekf_snapshot(handle);
    unsafe {
        ptr::write(out, snapshot);
    }
    snapshot.initialized
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must be either null or a valid pointer returned by this crate's
/// create functions. When non-null, `out` must point to writable memory for one
/// `SensorFusionFfiAlignProgress`.
pub unsafe extern "C" fn sensor_fusion_snapshot_align_progress(
    handle: *const SensorFusionFfi,
    out: *mut SensorFusionFfiAlignProgress,
) -> bool {
    if out.is_null() {
        return false;
    }
    let Some(fusion) = fusion_ref(handle) else {
        unsafe {
            ptr::write(out, SensorFusionFfiAlignProgress::default());
        }
        return false;
    };
    let Some(align) = fusion.inner.align() else {
        unsafe {
            ptr::write(out, SensorFusionFfiAlignProgress::default());
        }
        return false;
    };
    let sigma_deg = align.sigma_deg();
    unsafe {
        ptr::write(
            out,
            SensorFusionFfiAlignProgress {
                valid: true,
                coarse_ready: align.coarse_alignment_ready(),
                roll_sigma_deg: sigma_deg[0],
                pitch_sigma_deg: sigma_deg[1],
                yaw_sigma_deg: sigma_deg[2],
            },
        );
    }
    true
}

fn fusion_mut(handle: *mut SensorFusionFfi) -> Option<&'static mut SensorFusionFfi> {
    if handle.is_null() {
        None
    } else {
        unsafe { handle.as_mut() }
    }
}

fn fusion_ref(handle: *const SensorFusionFfi) -> Option<&'static SensorFusionFfi> {
    if handle.is_null() {
        None
    } else {
        unsafe { handle.as_ref() }
    }
}

fn harsh_behavior_preset_from_ffi(preset: u32) -> Option<HarshBehaviorPreset> {
    match preset {
        SENSOR_FUSION_HARSH_BEHAVIOR_SENSITIVE => Some(HarshBehaviorPreset::Sensitive),
        SENSOR_FUSION_HARSH_BEHAVIOR_BALANCED => Some(HarshBehaviorPreset::Balanced),
        SENSOR_FUSION_HARSH_BEHAVIOR_CONSERVATIVE => Some(HarshBehaviorPreset::Conservative),
        _ => None,
    }
}

fn ekf_snapshot(handle: *const SensorFusionFfi) -> SensorFusionFfiEkfSnapshot {
    let Some(fusion) = fusion_ref(handle) else {
        return SensorFusionFfiEkfSnapshot::default();
    };

    let mut snapshot = SensorFusionFfiEkfSnapshot {
        mount_ready: fusion.inner.mount_ready(),
        ..SensorFusionFfiEkfSnapshot::default()
    };

    let Some(ekf) = fusion.inner.ekf() else {
        return snapshot;
    };

    let nominal = &ekf.nominal;
    snapshot.initialized = true;
    snapshot.q0 = nominal.q0;
    snapshot.q1 = nominal.q1;
    snapshot.q2 = nominal.q2;
    snapshot.q3 = nominal.q3;
    snapshot.vel_n_mps = nominal.vn;
    snapshot.vel_e_mps = nominal.ve;
    snapshot.vel_d_mps = nominal.vd;
    snapshot.pos_n_m = nominal.pn;
    snapshot.pos_e_m = nominal.pe;
    snapshot.pos_d_m = nominal.pd;
    snapshot.gyro_bias_x_radps = nominal.bgx;
    snapshot.gyro_bias_y_radps = nominal.bgy;
    snapshot.gyro_bias_z_radps = nominal.bgz;
    snapshot.accel_bias_x_mps2 = nominal.bax;
    snapshot.accel_bias_y_mps2 = nominal.bay;
    snapshot.accel_bias_z_mps2 = nominal.baz;
    snapshot.q_bv0 = nominal.q_bv0;
    snapshot.q_bv1 = nominal.q_bv1;
    snapshot.q_bv2 = nominal.q_bv2;
    snapshot.q_bv3 = nominal.q_bv3;
    if let Some(lla) = fusion.inner.position_lla_f64() {
        snapshot.position_lla_valid = true;
        snapshot.lat_deg = lla[0];
        snapshot.lon_deg = lla[1];
        snapshot.height_m = lla[2];
    }
    snapshot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_handles_return_defaults() {
        let update = unsafe {
            sensor_fusion_process_imu(ptr::null_mut(), 0.0, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
        };
        assert!(!update.filter_initialized);

        let status = unsafe { sensor_fusion_snapshot_status(ptr::null()) };
        assert!(!status.mount_ready);
        assert!(!status.ekf_initialized);
        assert!(!status.filter_initialized);
        assert!(!status.mount_q_bv_valid);
        assert_eq!(status.mount_q_bv, [1.0, 0.0, 0.0, 0.0]);

        let mut snapshot = SensorFusionFfiEkfSnapshot {
            lat_deg: 42.0,
            ..SensorFusionFfiEkfSnapshot::default()
        };
        assert!(!unsafe { sensor_fusion_snapshot_ekf(ptr::null(), &mut snapshot) });
        assert_eq!(snapshot.lat_deg, 0.0);

        let mut align = SensorFusionFfiAlignProgress {
            valid: true,
            roll_sigma_deg: 42.0,
            ..SensorFusionFfiAlignProgress::default()
        };
        assert!(!unsafe { sensor_fusion_snapshot_align_progress(ptr::null(), &mut align) });
        assert!(!align.valid);
        assert_eq!(align.roll_sigma_deg, 0.0);
    }

    #[test]
    fn status_reports_pre_initialization_manual_mount_and_resets_edges() {
        let handle = sensor_fusion_create_ekf_manual(0.5, 0.5, 0.5, 0.5);
        assert!(!handle.is_null());

        let status = unsafe { sensor_fusion_snapshot_status(handle) };
        assert!(status.mount_ready);
        assert!(!status.mount_ready_changed);
        assert!(!status.ekf_initialized);
        assert!(!status.ekf_initialized_now);
        assert!(!status.filter_initialized);
        assert!(!status.filter_initialized_now);
        assert!(status.mount_q_bv_valid);
        assert_eq!(status.mount_q_bv, [0.5, 0.5, 0.5, 0.5]);

        let mut align = SensorFusionFfiAlignProgress::default();
        assert!(!unsafe { sensor_fusion_snapshot_align_progress(handle, &mut align) });
        assert!(!align.valid);

        let update = unsafe {
            sensor_fusion_process_gnss(
                handle, 1.0, 37.3318, -122.0312, 15.0, 6.0, 0.0, 0.0, 1.0, 1.0, 1.5, 0.2, 0.2, 0.2,
                0.0, true,
            )
        };
        assert!(update.ekf_initialized_now);

        let status = unsafe { sensor_fusion_snapshot_status(handle) };
        assert!(status.mount_ready);
        assert!(status.ekf_initialized);
        assert!(status.ekf_initialized_now);
        assert!(status.filter_initialized);
        assert!(status.filter_initialized_now);
        assert_eq!(status.mount_q_bv, [0.5, 0.5, 0.5, 0.5]);

        unsafe {
            sensor_fusion_reset_ekf_auto(handle);
        }
        let status = unsafe { sensor_fusion_snapshot_status(handle) };
        assert!(!status.mount_ready);
        assert!(!status.mount_ready_changed);
        assert!(!status.ekf_initialized);
        assert!(!status.ekf_initialized_now);
        assert!(!status.filter_initialized);
        assert!(!status.filter_initialized_now);
        assert!(!status.mount_q_bv_valid);
        assert_eq!(status.mount_q_bv, [1.0, 0.0, 0.0, 0.0]);

        unsafe {
            sensor_fusion_reset_ekf_manual(handle, 1.0, 0.0, 0.0, 0.0);
        }
        let status = unsafe { sensor_fusion_snapshot_status(handle) };
        assert!(status.mount_ready);
        assert!(!status.mount_ready_changed);
        assert!(!status.ekf_initialized);
        assert!(!status.ekf_initialized_now);
        assert!(!status.filter_initialized);
        assert!(!status.filter_initialized_now);
        assert!(status.mount_q_bv_valid);
        assert_eq!(status.mount_q_bv, [1.0, 0.0, 0.0, 0.0]);

        unsafe {
            sensor_fusion_destroy(handle);
        }
    }

    #[test]
    fn manual_gnss_initializes_and_snapshots_ekf_state() {
        let handle = sensor_fusion_create_ekf_manual(1.0, 0.0, 0.0, 0.0);
        assert!(!handle.is_null());

        let update = unsafe {
            sensor_fusion_process_gnss(
                handle, 1.0, 37.3318, -122.0312, 15.0, 6.0, 0.0, 0.0, 1.0, 1.0, 1.5, 0.2, 0.2, 0.2,
                0.0, true,
            )
        };
        assert!(update.mount_ready);
        assert!(update.ekf_initialized);
        assert!(update.ekf_initialized_now);

        let mut snapshot = SensorFusionFfiEkfSnapshot::default();
        assert!(unsafe { sensor_fusion_snapshot_ekf(handle, &mut snapshot) });
        assert!(snapshot.mount_ready);
        assert!(snapshot.initialized);
        assert_eq!(
            [
                snapshot.q_bv0,
                snapshot.q_bv1,
                snapshot.q_bv2,
                snapshot.q_bv3
            ],
            [1.0, 0.0, 0.0, 0.0]
        );
        assert!((snapshot.vel_n_mps - 6.0).abs() < 1.0e-6);
        assert!(snapshot.vel_e_mps.abs() < 1.0e-6);
        assert!(snapshot.vel_d_mps.abs() < 1.0e-6);
        assert!(snapshot.position_lla_valid);
        assert!((snapshot.lat_deg - 37.3318).abs() < 1.0e-6);
        assert!((snapshot.lon_deg + 122.0312).abs() < 1.0e-6);

        unsafe {
            sensor_fusion_destroy(handle);
        }
    }

    #[test]
    fn manual_mount_ffi_waits_for_yaw_seed_before_snapshot_initializes() {
        let handle = sensor_fusion_create_ekf_manual(1.0, 0.0, 0.0, 0.0);
        assert!(!handle.is_null());

        let stationary = unsafe {
            sensor_fusion_process_gnss(
                handle, 1.0, 37.3318, -122.0312, 15.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.5, 0.2, 0.2, 0.2,
                0.0, false,
            )
        };
        assert!(stationary.mount_ready);
        assert!(!stationary.ekf_initialized);
        assert!(!stationary.ekf_initialized_now);

        let mut snapshot = SensorFusionFfiEkfSnapshot::default();
        assert!(!unsafe { sensor_fusion_snapshot_ekf(handle, &mut snapshot) });
        assert!(snapshot.mount_ready);
        assert!(!snapshot.initialized);

        let moving = unsafe {
            sensor_fusion_process_gnss(
                handle, 2.0, 37.3318, -122.0312, 15.0, -6.0, 0.0, 0.0, 1.0, 1.0, 1.5, 0.2, 0.2,
                0.2, 0.0, false,
            )
        };
        assert!(!moving.ekf_initialized);
        assert!(!moving.ekf_initialized_now);

        let below_speed = unsafe {
            sensor_fusion_process_gnss(
                handle,
                3.0,
                37.3318,
                -122.0312,
                15.0,
                -5.5,
                0.0,
                0.0,
                1.0,
                1.0,
                1.5,
                0.2,
                0.2,
                0.2,
                core::f32::consts::PI,
                true,
            )
        };
        assert!(!below_speed.ekf_initialized);
        assert!(!below_speed.ekf_initialized_now);

        let moving = unsafe {
            sensor_fusion_process_gnss(
                handle,
                4.0,
                37.3318,
                -122.0312,
                15.0,
                -6.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.5,
                0.2,
                0.2,
                0.2,
                core::f32::consts::PI,
                true,
            )
        };
        assert!(moving.ekf_initialized);
        assert!(moving.ekf_initialized_now);

        assert!(unsafe { sensor_fusion_snapshot_ekf(handle, &mut snapshot) });
        assert!(snapshot.q3.abs() > 0.99);

        unsafe {
            sensor_fusion_destroy(handle);
        }
    }

    #[test]
    fn road_event_motion_ffi_emits_reverse_from_road_events_crate() {
        let handle = sensor_fusion_create_ekf_auto();
        assert!(!handle.is_null());

        let mut out = [SensorFusionFfiRoadEvent::default(); 4];
        let mut total = 0;
        for i in 0..25 {
            let count = unsafe {
                sensor_fusion_process_road_event_motion(
                    handle,
                    i as f32 * 0.1,
                    -0.8,
                    0.8,
                    0.0,
                    true,
                    0.0,
                    false,
                    0.0,
                    false,
                    0.0,
                    false,
                    0.0,
                    false,
                    out.as_mut_ptr(),
                    out.len(),
                )
            };
            total += count;
        }
        for i in 25..35 {
            let count = unsafe {
                sensor_fusion_process_road_event_motion(
                    handle,
                    i as f32 * 0.1,
                    0.0,
                    0.0,
                    0.0,
                    true,
                    0.0,
                    false,
                    0.0,
                    false,
                    0.0,
                    false,
                    0.0,
                    false,
                    out.as_mut_ptr(),
                    out.len(),
                )
            };
            total += count;
        }

        assert!(total > 0);
        assert_eq!(out[0].kind, ROAD_EVENT_REVERSE);
        assert!(out[0].duration_s >= 1.0);

        let mut trip = SensorFusionFfiTripSummary::default();
        assert!(unsafe { sensor_fusion_snapshot_trip_summary(handle, &mut trip) });
        assert!(trip.sample_count > 0);
        assert!(trip.distance_m > 0.0);
        assert!(trip.reverse_distance_m > 0.0);
        assert!(trip.reverse_events > 0);

        unsafe {
            sensor_fusion_destroy(handle);
        }
    }

    #[test]
    fn harsh_behavior_preset_ffi_rejects_unknown_values() {
        let handle = sensor_fusion_create_ekf_auto();
        assert!(!handle.is_null());

        assert!(unsafe {
            sensor_fusion_set_harsh_behavior_preset(
                handle,
                SENSOR_FUSION_HARSH_BEHAVIOR_SENSITIVE,
            )
        });
        assert!(!unsafe { sensor_fusion_set_harsh_behavior_preset(handle, 99) });

        unsafe {
            sensor_fusion_destroy(handle);
        }
    }
}
