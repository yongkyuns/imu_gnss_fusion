//! Native-only FFI adapter for the standalone C road-event detectors.

use road_events::{
    HarshBehaviorPreset, HarshCornerEvent, HarshCornerSample, HarshLongitudinalEvent,
    HarshLongitudinalSample, HillEvent, HillKind, HillSample, ReverseEvent, ReverseSample,
    RoadRoughnessEstimate, RoadRoughnessEvent, RoadRoughnessLevel, RoadRoughnessSample,
    RoadRoughnessUpdate, RoadShockEvent, SpeedBumpDiagnostic, SpeedBumpEvent, SpeedBumpSample,
    TripEventCounts, TripEventKind, TripSample, TripSummary,
};

const C_ROAD_EVENTS_CONTEXT_STORAGE_SIZE: usize = 2048;
const C_ROAD_EVENTS_CONTEXT_STORAGE_ALIGN: usize = 16;

#[repr(C, align(16))]
struct CDetectorStorage {
    bytes: [u8; C_ROAD_EVENTS_CONTEXT_STORAGE_SIZE],
}

impl Default for CDetectorStorage {
    fn default() -> Self {
        Self {
            bytes: [0; C_ROAD_EVENTS_CONTEXT_STORAGE_SIZE],
        }
    }
}

pub(crate) struct CRoadEvents {
    speed_bump: CDetectorStorage,
    hill: CDetectorStorage,
    reverse: CDetectorStorage,
    harsh_accel: CDetectorStorage,
    harsh_brake: CDetectorStorage,
    harsh_corner: CDetectorStorage,
    roughness: CDetectorStorage,
    trip_stats: CDetectorStorage,
}

impl CRoadEvents {
    pub(crate) fn new(preset: HarshBehaviorPreset) -> Self {
        assert_context_layouts();
        let mut runtime = Self {
            speed_bump: CDetectorStorage::default(),
            hill: CDetectorStorage::default(),
            reverse: CDetectorStorage::default(),
            harsh_accel: CDetectorStorage::default(),
            harsh_brake: CDetectorStorage::default(),
            harsh_corner: CDetectorStorage::default(),
            roughness: CDetectorStorage::default(),
            trip_stats: CDetectorStorage::default(),
        };
        let c_preset = harsh_preset_to_c(preset);
        // SAFETY: detector storage blocks are aligned, sized, and used only by their matching C APIs.
        unsafe {
            road_events_speed_bump_init_default(runtime.speed_bump_ptr());
            road_events_hill_init_default(runtime.hill_ptr());
            road_events_reverse_init_default(runtime.reverse_ptr());
            road_events_harsh_accel_init_preset(runtime.harsh_accel_ptr(), c_preset);
            road_events_harsh_brake_init_preset(runtime.harsh_brake_ptr(), c_preset);
            road_events_harsh_corner_init_preset(runtime.harsh_corner_ptr(), c_preset);
            road_events_roughness_init_default(runtime.roughness_ptr());
            road_events_trip_stats_init_default(runtime.trip_stats_ptr());
        }
        runtime
    }

    pub(crate) fn update_speed_bump(
        &mut self,
        sample: SpeedBumpSample,
    ) -> (SpeedBumpDiagnostic, Option<SpeedBumpEvent>) {
        let mut diag = CSpeedBumpDiagnostic::default();
        let mut event = CSpeedBumpEvent::default();
        // SAFETY: pointers reference initialized detector and output storage.
        let has_event = unsafe {
            road_events_speed_bump_update(
                self.speed_bump_ptr(),
                sample.into(),
                &mut diag,
                &mut event,
            )
        };
        (diag.into(), has_event.then_some(event.into()))
    }

    pub(crate) fn update_roughness_with_events(
        &mut self,
        sample: RoadRoughnessSample,
    ) -> Option<RoadRoughnessUpdate> {
        let mut update = CRoughnessUpdate::default();
        // SAFETY: pointers reference initialized analyzer and output storage.
        let ok = unsafe {
            road_events_roughness_update_with_events(
                self.roughness_ptr(),
                sample.into(),
                &mut update,
            )
        };
        ok.then_some(update.into())
    }

    pub(crate) fn update_hill(&mut self, sample: HillSample) -> Option<HillEvent> {
        let mut event = CHillEvent::default();
        // SAFETY: pointers reference initialized detector and output storage.
        let has_event =
            unsafe { road_events_hill_update(self.hill_ptr(), sample.into(), &mut event) };
        has_event.then_some(event.into())
    }

    pub(crate) fn finish_hill(&mut self) -> Option<HillEvent> {
        let mut event = CHillEvent::default();
        // SAFETY: pointers reference initialized detector and output storage.
        let has_event = unsafe { road_events_hill_finish(self.hill_ptr(), &mut event) };
        has_event.then_some(event.into())
    }

    pub(crate) fn update_reverse(&mut self, sample: ReverseSample) -> Option<ReverseEvent> {
        let mut event = CReverseEvent::default();
        // SAFETY: pointers reference initialized detector and output storage.
        let has_event =
            unsafe { road_events_reverse_update(self.reverse_ptr(), sample.into(), &mut event) };
        has_event.then_some(event.into())
    }

    pub(crate) fn finish_reverse(&mut self) -> Option<ReverseEvent> {
        let mut event = CReverseEvent::default();
        // SAFETY: pointers reference initialized detector and output storage.
        let has_event = unsafe { road_events_reverse_finish(self.reverse_ptr(), &mut event) };
        has_event.then_some(event.into())
    }

    pub(crate) fn update_harsh_accel(
        &mut self,
        sample: HarshLongitudinalSample,
    ) -> Option<HarshLongitudinalEvent> {
        let mut event = CHarshLongitudinalEvent::default();
        // SAFETY: pointers reference initialized detector and output storage.
        let has_event = unsafe {
            road_events_harsh_accel_update(self.harsh_accel_ptr(), sample.into(), &mut event)
        };
        has_event.then_some(event.into())
    }

    pub(crate) fn finish_harsh_accel(&mut self) -> Option<HarshLongitudinalEvent> {
        let mut event = CHarshLongitudinalEvent::default();
        // SAFETY: pointers reference initialized detector and output storage.
        let has_event =
            unsafe { road_events_harsh_accel_finish(self.harsh_accel_ptr(), &mut event) };
        has_event.then_some(event.into())
    }

    pub(crate) fn update_harsh_brake(
        &mut self,
        sample: HarshLongitudinalSample,
    ) -> Option<HarshLongitudinalEvent> {
        let mut event = CHarshLongitudinalEvent::default();
        // SAFETY: pointers reference initialized detector and output storage.
        let has_event = unsafe {
            road_events_harsh_brake_update(self.harsh_brake_ptr(), sample.into(), &mut event)
        };
        has_event.then_some(event.into())
    }

    pub(crate) fn finish_harsh_brake(&mut self) -> Option<HarshLongitudinalEvent> {
        let mut event = CHarshLongitudinalEvent::default();
        // SAFETY: pointers reference initialized detector and output storage.
        let has_event =
            unsafe { road_events_harsh_brake_finish(self.harsh_brake_ptr(), &mut event) };
        has_event.then_some(event.into())
    }

    pub(crate) fn update_harsh_corner(
        &mut self,
        sample: HarshCornerSample,
    ) -> Option<HarshCornerEvent> {
        let mut event = CHarshCornerEvent::default();
        // SAFETY: pointers reference initialized detector and output storage.
        let has_event = unsafe {
            road_events_harsh_corner_update(self.harsh_corner_ptr(), sample.into(), &mut event)
        };
        has_event.then_some(event.into())
    }

    pub(crate) fn finish_harsh_corner(&mut self) -> Option<HarshCornerEvent> {
        let mut event = CHarshCornerEvent::default();
        // SAFETY: pointers reference initialized detector and output storage.
        let has_event =
            unsafe { road_events_harsh_corner_finish(self.harsh_corner_ptr(), &mut event) };
        has_event.then_some(event.into())
    }

    pub(crate) fn finish_roughness(&mut self) -> Option<RoadRoughnessEvent> {
        let mut event = CRoughnessEvent::default();
        // SAFETY: pointers reference initialized analyzer and output storage.
        let has_event = unsafe { road_events_roughness_finish(self.roughness_ptr(), &mut event) };
        has_event.then_some(event.into())
    }

    pub(crate) fn roughness_estimate(&self) -> RoadRoughnessEstimate {
        // SAFETY: pointer references initialized analyzer and C returns a POD snapshot.
        unsafe { road_events_roughness_estimate(self.roughness_const_ptr()).into() }
    }

    pub(crate) fn update_trip_motion(&mut self, sample: TripSample) {
        // SAFETY: pointer references initialized stats state and sample is passed by value.
        unsafe {
            road_events_trip_stats_update_motion(self.trip_stats_ptr(), sample.into());
        }
    }

    pub(crate) fn record_event(&mut self, kind: TripEventKind) {
        // SAFETY: pointer references initialized stats state and enum value is in C range.
        unsafe {
            road_events_trip_stats_record_event(self.trip_stats_ptr(), trip_event_kind_to_c(kind));
        }
    }

    pub(crate) fn trip_summary(&self) -> TripSummary {
        // SAFETY: pointer references initialized stats state and C returns a POD snapshot.
        unsafe { road_events_trip_stats_summary(self.trip_stats_const_ptr()).into() }
    }

    fn speed_bump_ptr(&mut self) -> *mut CSpeedBumpDetectorOpaque {
        self.speed_bump.bytes.as_mut_ptr().cast()
    }
    fn hill_ptr(&mut self) -> *mut CHillDetectorOpaque {
        self.hill.bytes.as_mut_ptr().cast()
    }
    fn reverse_ptr(&mut self) -> *mut CReverseDetectorOpaque {
        self.reverse.bytes.as_mut_ptr().cast()
    }
    fn harsh_accel_ptr(&mut self) -> *mut CHarshAccelDetectorOpaque {
        self.harsh_accel.bytes.as_mut_ptr().cast()
    }
    fn harsh_brake_ptr(&mut self) -> *mut CHarshBrakeDetectorOpaque {
        self.harsh_brake.bytes.as_mut_ptr().cast()
    }
    fn harsh_corner_ptr(&mut self) -> *mut CHarshCornerDetectorOpaque {
        self.harsh_corner.bytes.as_mut_ptr().cast()
    }
    fn roughness_ptr(&mut self) -> *mut CRoughnessAnalyzerOpaque {
        self.roughness.bytes.as_mut_ptr().cast()
    }
    fn roughness_const_ptr(&self) -> *const CRoughnessAnalyzerOpaque {
        self.roughness.bytes.as_ptr().cast()
    }
    fn trip_stats_ptr(&mut self) -> *mut CTripStatsOpaque {
        self.trip_stats.bytes.as_mut_ptr().cast()
    }
    fn trip_stats_const_ptr(&self) -> *const CTripStatsOpaque {
        self.trip_stats.bytes.as_ptr().cast()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CSpeedBumpSample {
    t_s: f32,
    speed_mps: f32,
    pitch_deg: f32,
    vertical_accel_mps2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CSpeedBumpEvent {
    t_s: f32,
    confidence: f32,
    duration_s: f32,
    peak_abs_pitch_deg: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CSpeedBumpDiagnostic {
    t_s: f32,
    pitch_hpf_deg: f32,
    pitch_noise_deg: f32,
    vertical_accel_hpf_mps2: f32,
    vertical_accel_noise_mps2: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CHillSample {
    t_s: f32,
    speed_mps: f32,
    pitch_deg: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CHillEvent {
    kind: i32,
    start_t_s: f32,
    end_t_s: f32,
    duration_s: f32,
    mean_pitch_deg: f32,
    peak_abs_pitch_deg: f32,
    mean_speed_mps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CReverseSample {
    t_s: f32,
    forward_velocity_mps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CReverseEvent {
    start_t_s: f32,
    end_t_s: f32,
    duration_s: f32,
    mean_reverse_speed_mps: f32,
    peak_reverse_speed_mps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CHarshLongitudinalSample {
    t_s: f32,
    forward_velocity_mps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CHarshLongitudinalEvent {
    start_t_s: f32,
    end_t_s: f32,
    duration_s: f32,
    delta_velocity_mps: f32,
    mean_accel_mps2: f32,
    peak_accel_mps2: f32,
    mean_speed_mps: f32,
    peak_speed_mps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CHarshCornerSample {
    t_s: f32,
    speed_mps: f32,
    lateral_accel_mps2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CHarshCornerEvent {
    start_t_s: f32,
    end_t_s: f32,
    duration_s: f32,
    mean_lateral_accel_mps2: f32,
    peak_lateral_accel_mps2: f32,
    mean_speed_mps: f32,
    peak_speed_mps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CRoughnessSample {
    t_s: f32,
    speed_mps: f32,
    vertical_accel_mps2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CRoughnessEstimate {
    t_s: f32,
    roughness_rms_mps2: f32,
    level: i32,
    vertical_accel_bandpass_mps2: f32,
    vertical_accel_clipped_mps2: f32,
    distance_m: f32,
    updated: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CRoughnessEvent {
    start_t_s: f32,
    end_t_s: f32,
    duration_s: f32,
    mean_roughness_rms_mps2: f32,
    peak_roughness_rms_mps2: f32,
    mean_speed_mps: f32,
    distance_m: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CShockEvent {
    start_t_s: f32,
    end_t_s: f32,
    duration_s: f32,
    peak_abs_vertical_accel_mps2: f32,
    mean_speed_mps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CRoughnessUpdate {
    estimate: CRoughnessEstimate,
    has_roughness_event: bool,
    roughness_event: CRoughnessEvent,
    has_completed_roughness_event: bool,
    completed_roughness_event: CRoughnessEvent,
    has_shock_event: bool,
    shock_event: CShockEvent,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CTripSample {
    t_s: f32,
    speed_mps: f32,
    forward_velocity_mps: f32,
    height_valid: bool,
    height_m: f32,
    height_frame_id: u32,
    longitudinal_accel_mps2: f32,
    lateral_accel_mps2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CTripEventCounts {
    speed_bumps: u32,
    road_shocks: u32,
    rough_road: u32,
    uphill: u32,
    downhill: u32,
    reverse: u32,
    harsh_acceleration: u32,
    harsh_braking: u32,
    harsh_cornering: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CTripSummary {
    sample_count: u32,
    invalid_sample_count: u32,
    data_gap_count: u32,
    max_sample_gap_s: f32,
    total_gap_duration_s: f32,
    duration_s: f32,
    moving_duration_s: f32,
    stationary_duration_s: f32,
    distance_m: f32,
    reverse_duration_s: f32,
    reverse_distance_m: f32,
    elevation_gain_m: f32,
    elevation_loss_m: f32,
    elevation_valid: bool,
    mean_speed_mps: f32,
    moving_mean_speed_mps: f32,
    peak_speed_mps: f32,
    peak_accel_mps2: f32,
    peak_decel_mps2: f32,
    peak_lateral_accel_mps2: f32,
    rolling_speed_mps: f32,
    rolling_abs_longitudinal_accel_mps2: f32,
    rolling_abs_lateral_accel_mps2: f32,
    events: CTripEventCounts,
    speed_bumps_per_km: f32,
    road_shocks_per_km: f32,
    rough_road_events_per_km: f32,
    harsh_events_per_km: f32,
    reverse_seconds_per_km: f32,
}

impl From<SpeedBumpSample> for CSpeedBumpSample {
    fn from(sample: SpeedBumpSample) -> Self {
        Self {
            t_s: sample.t_s,
            speed_mps: sample.speed_mps,
            pitch_deg: sample.pitch_deg,
            vertical_accel_mps2: sample.vertical_accel_mps2,
        }
    }
}

impl From<CSpeedBumpDiagnostic> for SpeedBumpDiagnostic {
    fn from(value: CSpeedBumpDiagnostic) -> Self {
        Self {
            t_s: value.t_s,
            pitch_hpf_deg: value.pitch_hpf_deg,
            pitch_noise_deg: value.pitch_noise_deg,
            vertical_accel_hpf_mps2: value.vertical_accel_hpf_mps2,
            vertical_accel_noise_mps2: value.vertical_accel_noise_mps2,
        }
    }
}

impl From<CSpeedBumpEvent> for SpeedBumpEvent {
    fn from(value: CSpeedBumpEvent) -> Self {
        Self {
            t_s: value.t_s,
            confidence: value.confidence,
            duration_s: value.duration_s,
            peak_abs_pitch_deg: value.peak_abs_pitch_deg,
        }
    }
}

impl From<HillSample> for CHillSample {
    fn from(sample: HillSample) -> Self {
        Self {
            t_s: sample.t_s,
            speed_mps: sample.speed_mps,
            pitch_deg: sample.pitch_deg,
        }
    }
}

impl From<CHillEvent> for HillEvent {
    fn from(value: CHillEvent) -> Self {
        Self {
            kind: match value.kind {
                0 => HillKind::Uphill,
                _ => HillKind::Downhill,
            },
            start_t_s: value.start_t_s,
            end_t_s: value.end_t_s,
            duration_s: value.duration_s,
            mean_pitch_deg: value.mean_pitch_deg,
            peak_abs_pitch_deg: value.peak_abs_pitch_deg,
            mean_speed_mps: value.mean_speed_mps,
        }
    }
}

impl From<ReverseSample> for CReverseSample {
    fn from(sample: ReverseSample) -> Self {
        Self {
            t_s: sample.t_s,
            forward_velocity_mps: sample.forward_velocity_mps,
        }
    }
}

impl From<CReverseEvent> for ReverseEvent {
    fn from(value: CReverseEvent) -> Self {
        Self {
            start_t_s: value.start_t_s,
            end_t_s: value.end_t_s,
            duration_s: value.duration_s,
            mean_reverse_speed_mps: value.mean_reverse_speed_mps,
            peak_reverse_speed_mps: value.peak_reverse_speed_mps,
        }
    }
}

impl From<HarshLongitudinalSample> for CHarshLongitudinalSample {
    fn from(sample: HarshLongitudinalSample) -> Self {
        Self {
            t_s: sample.t_s,
            forward_velocity_mps: sample.forward_velocity_mps,
        }
    }
}

impl From<CHarshLongitudinalEvent> for HarshLongitudinalEvent {
    fn from(value: CHarshLongitudinalEvent) -> Self {
        Self {
            start_t_s: value.start_t_s,
            end_t_s: value.end_t_s,
            duration_s: value.duration_s,
            delta_velocity_mps: value.delta_velocity_mps,
            mean_accel_mps2: value.mean_accel_mps2,
            peak_accel_mps2: value.peak_accel_mps2,
            mean_speed_mps: value.mean_speed_mps,
            peak_speed_mps: value.peak_speed_mps,
        }
    }
}

impl From<HarshCornerSample> for CHarshCornerSample {
    fn from(sample: HarshCornerSample) -> Self {
        Self {
            t_s: sample.t_s,
            speed_mps: sample.speed_mps,
            lateral_accel_mps2: sample.lateral_accel_mps2,
        }
    }
}

impl From<CHarshCornerEvent> for HarshCornerEvent {
    fn from(value: CHarshCornerEvent) -> Self {
        Self {
            start_t_s: value.start_t_s,
            end_t_s: value.end_t_s,
            duration_s: value.duration_s,
            mean_lateral_accel_mps2: value.mean_lateral_accel_mps2,
            peak_lateral_accel_mps2: value.peak_lateral_accel_mps2,
            mean_speed_mps: value.mean_speed_mps,
            peak_speed_mps: value.peak_speed_mps,
        }
    }
}

impl From<RoadRoughnessSample> for CRoughnessSample {
    fn from(sample: RoadRoughnessSample) -> Self {
        Self {
            t_s: sample.t_s,
            speed_mps: sample.speed_mps,
            vertical_accel_mps2: sample.vertical_accel_mps2,
        }
    }
}

impl From<CRoughnessEstimate> for RoadRoughnessEstimate {
    fn from(value: CRoughnessEstimate) -> Self {
        Self {
            t_s: value.t_s,
            roughness_rms_mps2: value.roughness_rms_mps2,
            level: roughness_level_from_c(value.level),
            vertical_accel_bandpass_mps2: value.vertical_accel_bandpass_mps2,
            vertical_accel_clipped_mps2: value.vertical_accel_clipped_mps2,
            distance_m: value.distance_m,
            updated: value.updated,
        }
    }
}

impl From<CRoughnessEvent> for RoadRoughnessEvent {
    fn from(value: CRoughnessEvent) -> Self {
        Self {
            start_t_s: value.start_t_s,
            end_t_s: value.end_t_s,
            duration_s: value.duration_s,
            mean_roughness_rms_mps2: value.mean_roughness_rms_mps2,
            peak_roughness_rms_mps2: value.peak_roughness_rms_mps2,
            mean_speed_mps: value.mean_speed_mps,
            distance_m: value.distance_m,
        }
    }
}

impl From<CShockEvent> for RoadShockEvent {
    fn from(value: CShockEvent) -> Self {
        Self {
            start_t_s: value.start_t_s,
            end_t_s: value.end_t_s,
            duration_s: value.duration_s,
            peak_abs_vertical_accel_mps2: value.peak_abs_vertical_accel_mps2,
            mean_speed_mps: value.mean_speed_mps,
        }
    }
}

impl From<CRoughnessUpdate> for RoadRoughnessUpdate {
    fn from(value: CRoughnessUpdate) -> Self {
        Self {
            estimate: value.estimate.into(),
            roughness_event: value
                .has_roughness_event
                .then_some(value.roughness_event.into()),
            completed_roughness_event: value
                .has_completed_roughness_event
                .then_some(value.completed_roughness_event.into()),
            shock_event: value.has_shock_event.then_some(value.shock_event.into()),
        }
    }
}

impl From<TripSample> for CTripSample {
    fn from(sample: TripSample) -> Self {
        Self {
            t_s: sample.t_s,
            speed_mps: sample.speed_mps,
            forward_velocity_mps: sample.forward_velocity_mps,
            height_valid: sample.height_m.is_some(),
            height_m: sample.height_m.unwrap_or(0.0),
            height_frame_id: sample.height_frame_id,
            longitudinal_accel_mps2: sample.longitudinal_accel_mps2,
            lateral_accel_mps2: sample.lateral_accel_mps2,
        }
    }
}

impl From<CTripEventCounts> for TripEventCounts {
    fn from(value: CTripEventCounts) -> Self {
        Self {
            speed_bumps: value.speed_bumps,
            road_shocks: value.road_shocks,
            rough_road: value.rough_road,
            uphill: value.uphill,
            downhill: value.downhill,
            reverse: value.reverse,
            harsh_acceleration: value.harsh_acceleration,
            harsh_braking: value.harsh_braking,
            harsh_cornering: value.harsh_cornering,
        }
    }
}

impl From<CTripSummary> for TripSummary {
    fn from(value: CTripSummary) -> Self {
        Self {
            sample_count: value.sample_count,
            invalid_sample_count: value.invalid_sample_count,
            data_gap_count: value.data_gap_count,
            max_sample_gap_s: value.max_sample_gap_s,
            total_gap_duration_s: value.total_gap_duration_s,
            duration_s: value.duration_s,
            moving_duration_s: value.moving_duration_s,
            stationary_duration_s: value.stationary_duration_s,
            distance_m: value.distance_m,
            reverse_duration_s: value.reverse_duration_s,
            reverse_distance_m: value.reverse_distance_m,
            elevation_gain_m: value.elevation_gain_m,
            elevation_loss_m: value.elevation_loss_m,
            elevation_valid: value.elevation_valid,
            mean_speed_mps: value.mean_speed_mps,
            moving_mean_speed_mps: value.moving_mean_speed_mps,
            peak_speed_mps: value.peak_speed_mps,
            peak_accel_mps2: value.peak_accel_mps2,
            peak_decel_mps2: value.peak_decel_mps2,
            peak_lateral_accel_mps2: value.peak_lateral_accel_mps2,
            rolling_speed_mps: value.rolling_speed_mps,
            rolling_abs_longitudinal_accel_mps2: value.rolling_abs_longitudinal_accel_mps2,
            rolling_abs_lateral_accel_mps2: value.rolling_abs_lateral_accel_mps2,
            events: value.events.into(),
            speed_bumps_per_km: value.speed_bumps_per_km,
            road_shocks_per_km: value.road_shocks_per_km,
            rough_road_events_per_km: value.rough_road_events_per_km,
            harsh_events_per_km: value.harsh_events_per_km,
            reverse_seconds_per_km: value.reverse_seconds_per_km,
        }
    }
}

fn roughness_level_from_c(level: i32) -> RoadRoughnessLevel {
    match level {
        0 => RoadRoughnessLevel::VerySmooth,
        1 => RoadRoughnessLevel::Smooth,
        2 => RoadRoughnessLevel::LightTexture,
        3 => RoadRoughnessLevel::Moderate,
        4 => RoadRoughnessLevel::Rough,
        5 => RoadRoughnessLevel::VeryRough,
        _ => RoadRoughnessLevel::Severe,
    }
}

fn harsh_preset_to_c(preset: HarshBehaviorPreset) -> i32 {
    match preset {
        HarshBehaviorPreset::Sensitive => 0,
        HarshBehaviorPreset::Balanced => 1,
        HarshBehaviorPreset::Conservative => 2,
    }
}

fn trip_event_kind_to_c(kind: TripEventKind) -> i32 {
    match kind {
        TripEventKind::SpeedBump => 0,
        TripEventKind::RoadShock => 1,
        TripEventKind::RoughRoad => 2,
        TripEventKind::Uphill => 3,
        TripEventKind::Downhill => 4,
        TripEventKind::Reverse => 5,
        TripEventKind::HarshAcceleration => 6,
        TripEventKind::HarshBraking => 7,
        TripEventKind::HarshCornering => 8,
    }
}

fn assert_context_layouts() {
    assert_context_layout(
        "road_events_speed_bump_detector_t",
        unsafe { road_events_speed_bump_detector_size() },
        unsafe { road_events_speed_bump_detector_alignment() },
    );
    assert_context_layout(
        "road_events_hill_detector_t",
        unsafe { road_events_hill_detector_size() },
        unsafe { road_events_hill_detector_alignment() },
    );
    assert_context_layout(
        "road_events_reverse_detector_t",
        unsafe { road_events_reverse_detector_size() },
        unsafe { road_events_reverse_detector_alignment() },
    );
    assert_context_layout(
        "road_events_harsh_accel_detector_t",
        unsafe { road_events_harsh_accel_detector_size() },
        unsafe { road_events_harsh_accel_detector_alignment() },
    );
    assert_context_layout(
        "road_events_harsh_brake_detector_t",
        unsafe { road_events_harsh_brake_detector_size() },
        unsafe { road_events_harsh_brake_detector_alignment() },
    );
    assert_context_layout(
        "road_events_harsh_corner_detector_t",
        unsafe { road_events_harsh_corner_detector_size() },
        unsafe { road_events_harsh_corner_detector_alignment() },
    );
    assert_context_layout(
        "road_events_roughness_analyzer_t",
        unsafe { road_events_roughness_analyzer_size() },
        unsafe { road_events_roughness_analyzer_alignment() },
    );
    assert_context_layout(
        "road_events_trip_stats_t",
        unsafe { road_events_trip_stats_size() },
        unsafe { road_events_trip_stats_alignment() },
    );
}

fn assert_context_layout(name: &str, size: usize, alignment: usize) {
    assert!(
        size <= C_ROAD_EVENTS_CONTEXT_STORAGE_SIZE,
        "{name} size {size} exceeds Rust FFI storage"
    );
    assert!(
        alignment <= C_ROAD_EVENTS_CONTEXT_STORAGE_ALIGN,
        "{name} alignment {alignment} exceeds Rust FFI storage alignment"
    );
}

macro_rules! opaque_type {
    ($name:ident) => {
        #[repr(C)]
        struct $name {
            _private: [u8; 0],
        }
    };
}

opaque_type!(CSpeedBumpDetectorOpaque);
opaque_type!(CHillDetectorOpaque);
opaque_type!(CReverseDetectorOpaque);
opaque_type!(CHarshAccelDetectorOpaque);
opaque_type!(CHarshBrakeDetectorOpaque);
opaque_type!(CHarshCornerDetectorOpaque);
opaque_type!(CRoughnessAnalyzerOpaque);
opaque_type!(CTripStatsOpaque);

unsafe extern "C" {
    fn road_events_speed_bump_detector_size() -> usize;
    fn road_events_speed_bump_detector_alignment() -> usize;
    fn road_events_hill_detector_size() -> usize;
    fn road_events_hill_detector_alignment() -> usize;
    fn road_events_reverse_detector_size() -> usize;
    fn road_events_reverse_detector_alignment() -> usize;
    fn road_events_harsh_accel_detector_size() -> usize;
    fn road_events_harsh_accel_detector_alignment() -> usize;
    fn road_events_harsh_brake_detector_size() -> usize;
    fn road_events_harsh_brake_detector_alignment() -> usize;
    fn road_events_harsh_corner_detector_size() -> usize;
    fn road_events_harsh_corner_detector_alignment() -> usize;
    fn road_events_roughness_analyzer_size() -> usize;
    fn road_events_roughness_analyzer_alignment() -> usize;
    fn road_events_trip_stats_size() -> usize;
    fn road_events_trip_stats_alignment() -> usize;

    fn road_events_speed_bump_init_default(det: *mut CSpeedBumpDetectorOpaque);
    fn road_events_hill_init_default(det: *mut CHillDetectorOpaque);
    fn road_events_reverse_init_default(det: *mut CReverseDetectorOpaque);
    fn road_events_harsh_accel_init_preset(det: *mut CHarshAccelDetectorOpaque, preset: i32);
    fn road_events_harsh_brake_init_preset(det: *mut CHarshBrakeDetectorOpaque, preset: i32);
    fn road_events_harsh_corner_init_preset(det: *mut CHarshCornerDetectorOpaque, preset: i32);
    fn road_events_roughness_init_default(analyzer: *mut CRoughnessAnalyzerOpaque);
    fn road_events_trip_stats_init_default(stats: *mut CTripStatsOpaque);

    fn road_events_speed_bump_update(
        det: *mut CSpeedBumpDetectorOpaque,
        sample: CSpeedBumpSample,
        out_diagnostic: *mut CSpeedBumpDiagnostic,
        out_event: *mut CSpeedBumpEvent,
    ) -> bool;
    fn road_events_hill_update(
        det: *mut CHillDetectorOpaque,
        sample: CHillSample,
        out_event: *mut CHillEvent,
    ) -> bool;
    fn road_events_hill_finish(det: *mut CHillDetectorOpaque, out_event: *mut CHillEvent) -> bool;
    fn road_events_reverse_update(
        det: *mut CReverseDetectorOpaque,
        sample: CReverseSample,
        out_event: *mut CReverseEvent,
    ) -> bool;
    fn road_events_reverse_finish(
        det: *mut CReverseDetectorOpaque,
        out_event: *mut CReverseEvent,
    ) -> bool;
    fn road_events_harsh_accel_update(
        det: *mut CHarshAccelDetectorOpaque,
        sample: CHarshLongitudinalSample,
        out_event: *mut CHarshLongitudinalEvent,
    ) -> bool;
    fn road_events_harsh_accel_finish(
        det: *mut CHarshAccelDetectorOpaque,
        out_event: *mut CHarshLongitudinalEvent,
    ) -> bool;
    fn road_events_harsh_brake_update(
        det: *mut CHarshBrakeDetectorOpaque,
        sample: CHarshLongitudinalSample,
        out_event: *mut CHarshLongitudinalEvent,
    ) -> bool;
    fn road_events_harsh_brake_finish(
        det: *mut CHarshBrakeDetectorOpaque,
        out_event: *mut CHarshLongitudinalEvent,
    ) -> bool;
    fn road_events_harsh_corner_update(
        det: *mut CHarshCornerDetectorOpaque,
        sample: CHarshCornerSample,
        out_event: *mut CHarshCornerEvent,
    ) -> bool;
    fn road_events_harsh_corner_finish(
        det: *mut CHarshCornerDetectorOpaque,
        out_event: *mut CHarshCornerEvent,
    ) -> bool;
    fn road_events_roughness_update_with_events(
        analyzer: *mut CRoughnessAnalyzerOpaque,
        sample: CRoughnessSample,
        out_update: *mut CRoughnessUpdate,
    ) -> bool;
    fn road_events_roughness_estimate(
        analyzer: *const CRoughnessAnalyzerOpaque,
    ) -> CRoughnessEstimate;
    fn road_events_roughness_finish(
        analyzer: *mut CRoughnessAnalyzerOpaque,
        out_event: *mut CRoughnessEvent,
    ) -> bool;
    fn road_events_trip_stats_update_motion(stats: *mut CTripStatsOpaque, sample: CTripSample);
    fn road_events_trip_stats_record_event(stats: *mut CTripStatsOpaque, kind: i32);
    fn road_events_trip_stats_summary(stats: *const CTripStatsOpaque) -> CTripSummary;
}

#[cfg(test)]
mod tests {
    use super::*;
    use road_events::{
        HarshAccelConfig, HarshAccelDetector, HarshBrakeConfig, HarshBrakeDetector,
        HarshCornerConfig, HarshCornerDetector, HillConfig, HillDetector, ReverseConfig,
        ReverseDetector, RoadRoughnessAnalyzer, SpeedBumpConfig, SpeedBumpDetector, TripStats,
    };

    #[test]
    fn c_road_events_ffi_matches_rust_detectors_on_representative_streams() {
        let mut c = CRoadEvents::new(HarshBehaviorPreset::Balanced);
        let mut rust_bump = SpeedBumpDetector::new(SpeedBumpConfig::default());
        let mut rust_roughness = RoadRoughnessAnalyzer::default();
        let mut rust_hill = HillDetector::new(HillConfig::default());
        let mut rust_reverse = ReverseDetector::new(ReverseConfig::default());
        let mut rust_accel = HarshAccelDetector::new(HarshAccelConfig::default());
        let mut rust_brake = HarshBrakeDetector::new(HarshBrakeConfig::default());
        let mut rust_corner = HarshCornerDetector::new(HarshCornerConfig::default());
        let mut rust_trip = TripStats::default();

        let mut bump_events = (0_u32, 0_u32);
        let mut rough_events = (0_u32, 0_u32);
        let mut shock_events = (0_u32, 0_u32);
        let mut hill_events = (0_u32, 0_u32);
        let mut reverse_events = (0_u32, 0_u32);
        let mut accel_events = (0_u32, 0_u32);
        let mut brake_events = (0_u32, 0_u32);
        let mut corner_events = (0_u32, 0_u32);

        for i in 0..700 {
            let t = i as f32 * 0.02;
            let bump_wave =
                gaussian(t, 2.0, 0.12) - 1.2 * gaussian(t, 2.3, 0.12) + gaussian(t, 2.6, 0.12);
            let speed = if t < 4.0 {
                4.0
            } else if t < 7.0 {
                4.0 + 3.0 * (t - 4.0)
            } else if t < 10.0 {
                13.0 - 4.0 * (t - 7.0)
            } else if t < 12.0 {
                -1.0
            } else {
                5.0
            };
            let pitch = if (8.0..=10.0).contains(&t) {
                4.8
            } else {
                0.9 * gaussian(t, 2.1, 0.14) - 1.1 * gaussian(t, 2.42, 0.14)
            };
            let vertical_accel =
                4.0 * bump_wave + if (12.0..12.1).contains(&t) { 6.0 } else { 0.0 };
            let lateral_accel = if (5.0..5.15).contains(&t) {
                0.0
            } else if (5.15..6.2).contains(&t) {
                4.0
            } else {
                0.0
            };

            let bump_sample = SpeedBumpSample {
                t_s: t,
                speed_mps: speed.abs(),
                pitch_deg: pitch,
                vertical_accel_mps2: vertical_accel,
            };
            let (rust_diag, rust_event) = rust_bump.update(bump_sample);
            let (c_diag, c_event) = c.update_speed_bump(bump_sample);
            assert_close(c_diag.pitch_hpf_deg, rust_diag.pitch_hpf_deg, 1.0e-4);
            assert_close(
                c_diag.vertical_accel_hpf_mps2,
                rust_diag.vertical_accel_hpf_mps2,
                1.0e-4,
            );
            bump_events.0 += u32::from(rust_event.is_some());
            bump_events.1 += u32::from(c_event.is_some());

            let rough_sample = RoadRoughnessSample {
                t_s: t,
                speed_mps: speed.abs(),
                vertical_accel_mps2: vertical_accel,
            };
            let rust_rough = rust_roughness.update_with_events(rough_sample);
            let c_rough = c.update_roughness_with_events(rough_sample);
            assert_eq!(rust_rough.is_some(), c_rough.is_some());
            if let (Some(rust_rough), Some(c_rough)) = (rust_rough, c_rough) {
                assert_close(
                    c_rough.estimate.roughness_rms_mps2,
                    rust_rough.estimate.roughness_rms_mps2,
                    1.0e-4,
                );
                rough_events.0 += u32::from(rust_rough.completed_roughness_event.is_some());
                rough_events.1 += u32::from(c_rough.completed_roughness_event.is_some());
                shock_events.0 += u32::from(rust_rough.shock_event.is_some());
                shock_events.1 += u32::from(c_rough.shock_event.is_some());
            }

            let hill_sample = HillSample {
                t_s: t,
                speed_mps: speed.abs(),
                pitch_deg: pitch,
            };
            hill_events.0 += u32::from(rust_hill.update(hill_sample).is_some());
            hill_events.1 += u32::from(c.update_hill(hill_sample).is_some());

            let reverse_sample = ReverseSample {
                t_s: t,
                forward_velocity_mps: speed,
            };
            reverse_events.0 += u32::from(rust_reverse.update(reverse_sample).is_some());
            reverse_events.1 += u32::from(c.update_reverse(reverse_sample).is_some());

            let longitudinal_sample = HarshLongitudinalSample {
                t_s: t,
                forward_velocity_mps: speed,
            };
            accel_events.0 += u32::from(rust_accel.update(longitudinal_sample).is_some());
            accel_events.1 += u32::from(c.update_harsh_accel(longitudinal_sample).is_some());
            brake_events.0 += u32::from(rust_brake.update(longitudinal_sample).is_some());
            brake_events.1 += u32::from(c.update_harsh_brake(longitudinal_sample).is_some());

            let corner_sample = HarshCornerSample {
                t_s: t,
                speed_mps: speed.abs(),
                lateral_accel_mps2: lateral_accel,
            };
            corner_events.0 += u32::from(rust_corner.update(corner_sample).is_some());
            corner_events.1 += u32::from(c.update_harsh_corner(corner_sample).is_some());

            let trip_sample = TripSample {
                t_s: t,
                speed_mps: speed.abs(),
                forward_velocity_mps: speed,
                height_m: Some(100.0 + 0.1 * t),
                height_frame_id: 0,
                longitudinal_accel_mps2: if (4.5..4.7).contains(&t) { 3.0 } else { 0.0 },
                lateral_accel_mps2: lateral_accel,
            };
            rust_trip.update_motion(trip_sample);
            c.update_trip_motion(trip_sample);
        }

        rough_events.0 += u32::from(rust_roughness.finish().is_some());
        rough_events.1 += u32::from(c.finish_roughness().is_some());
        hill_events.0 += u32::from(rust_hill.finish().is_some());
        hill_events.1 += u32::from(c.finish_hill().is_some());
        reverse_events.0 += u32::from(rust_reverse.finish().is_some());
        reverse_events.1 += u32::from(c.finish_reverse().is_some());
        accel_events.0 += u32::from(rust_accel.finish().is_some());
        accel_events.1 += u32::from(c.finish_harsh_accel().is_some());
        brake_events.0 += u32::from(rust_brake.finish().is_some());
        brake_events.1 += u32::from(c.finish_harsh_brake().is_some());
        corner_events.0 += u32::from(rust_corner.finish().is_some());
        corner_events.1 += u32::from(c.finish_harsh_corner().is_some());

        assert_eq!(bump_events.0, bump_events.1);
        assert_eq!(rough_events.0, rough_events.1);
        assert_eq!(shock_events.0, shock_events.1);
        assert_eq!(hill_events.0, hill_events.1);
        assert_eq!(reverse_events.0, reverse_events.1);
        assert_eq!(accel_events.0, accel_events.1);
        assert_eq!(brake_events.0, brake_events.1);
        assert_eq!(corner_events.0, corner_events.1);

        let rust_summary = rust_trip.summary();
        let c_summary = c.trip_summary();
        assert_eq!(rust_summary.sample_count, c_summary.sample_count);
        assert_close(c_summary.distance_m, rust_summary.distance_m, 1.0e-3);
        assert_close(
            c_summary.mean_speed_mps,
            rust_summary.mean_speed_mps,
            1.0e-5,
        );
        assert_close(
            c_summary.peak_lateral_accel_mps2,
            rust_summary.peak_lateral_accel_mps2,
            1.0e-5,
        );

        let rust_rough = rust_roughness.estimate();
        let c_rough = c.roughness_estimate();
        assert_close(
            c_rough.roughness_rms_mps2,
            rust_rough.roughness_rms_mps2,
            1.0e-4,
        );
        assert_eq!(c_rough.level, rust_rough.level);
    }

    fn gaussian(t: f32, center: f32, sigma: f32) -> f32 {
        let z = (t - center) / sigma;
        (-0.5 * z * z).exp()
    }

    fn assert_close(actual: f32, expected: f32, tol: f32) {
        assert!(
            (actual - expected).abs() <= tol,
            "actual={actual} expected={expected} tol={tol}"
        );
    }
}
