#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Default)]
pub struct Trace {
    pub name: String,
    pub points: Vec<[f64; 2]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoadEventKind {
    SpeedBump,
    RoadShock,
    RoughRoad,
    Uphill,
    Downhill,
    Reverse,
    HarshAcceleration,
    HarshBraking,
    HarshCornering,
}

impl RoadEventKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SpeedBump => "speed bump",
            Self::RoadShock => "road shock",
            Self::RoughRoad => "rough road",
            Self::Uphill => "uphill",
            Self::Downhill => "downhill",
            Self::Reverse => "reverse",
            Self::HarshAcceleration => "harsh acceleration",
            Self::HarshBraking => "harsh braking",
            Self::HarshCornering => "harsh cornering",
        }
    }

    pub fn parse(kind: &str) -> Option<Self> {
        match kind {
            "speed bump" => Some(Self::SpeedBump),
            "road shock" => Some(Self::RoadShock),
            "rough road" => Some(Self::RoughRoad),
            "uphill" => Some(Self::Uphill),
            "downhill" => Some(Self::Downhill),
            "reverse" => Some(Self::Reverse),
            "harsh acceleration" => Some(Self::HarshAcceleration),
            "harsh braking" => Some(Self::HarshBraking),
            "harsh cornering" => Some(Self::HarshCornering),
            _ => None,
        }
    }
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Default)]
pub struct PlotData {
    pub speed: Vec<Trace>,
    pub vehicle_motion_gyro: Vec<Trace>,
    pub vehicle_motion_accel: Vec<Trace>,
    pub sat_cn0: Vec<Trace>,
    pub imu_raw_gyro: Vec<Trace>,
    pub imu_raw_accel: Vec<Trace>,
    pub imu_cal_gyro: Vec<Trace>,
    pub imu_cal_accel: Vec<Trace>,
    pub orientation: Vec<Trace>,
    pub other: Vec<Trace>,
    pub ekf_cmp_pos: Vec<Trace>,
    pub ekf_cmp_vel: Vec<Trace>,
    pub ekf_cmp_att: Vec<Trace>,
    pub ekf_meas_gyro: Vec<Trace>,
    pub ekf_meas_accel: Vec<Trace>,
    pub ekf_bias_gyro: Vec<Trace>,
    pub ekf_bias_accel: Vec<Trace>,
    pub ekf_cov_bias: Vec<Trace>,
    pub ekf_cov_nonbias: Vec<Trace>,
    pub ekf_mount_sigma: Vec<Trace>,
    pub ekf_mount_dx: Vec<Trace>,
    pub ekf_nhc_mount_dx: Vec<Trace>,
    pub ekf_nhc_innovation: Vec<Trace>,
    pub ekf_nhc_nis: Vec<Trace>,
    pub ekf_nhc_h_mount_norm: Vec<Trace>,
    pub ekf_misalignment: Vec<Trace>,
    pub ekf_stationary_diag: Vec<Trace>,
    pub ekf_bump_pitch_speed: Vec<Trace>,
    pub ekf_bump_diag: Vec<Trace>,
    pub ekf_road_roughness: Vec<Trace>,
    pub ekf_road_event_motion: Vec<Trace>,
    pub road_events: Vec<RoadEventSample>,
    pub road_segments: Vec<RoadSegmentSample>,
    pub trip_summary: TripSummarySample,
    pub ekf_map: Vec<Trace>,
    pub map_cursor: Vec<MapCursorSample>,
    pub ekf_map_heading: Vec<HeadingSample>,
    pub align_cmp_att: Vec<Trace>,
    pub align_res_vel: Vec<Trace>,
    pub align_axis_err: Vec<Trace>,
    pub align_motion: Vec<Trace>,
    pub align_flags: Vec<Trace>,
    pub align_roll_contrib: Vec<Trace>,
    pub align_pitch_contrib: Vec<Trace>,
    pub align_yaw_contrib: Vec<Trace>,
    pub align_cov: Vec<Trace>,
    pub update_inspector: Vec<UpdateInspectorSample>,
}

impl PlotData {
    pub fn trace_by_name(&self, name: &str) -> Option<&Trace> {
        self.trace_groups()
            .into_iter()
            .flatten()
            .find(|trace| trace.name == name && !trace.points.is_empty())
    }

    pub fn has_trace_points(&self) -> bool {
        self.trace_groups()
            .into_iter()
            .flatten()
            .any(|trace| !trace.points.is_empty())
    }

    fn trace_groups(&self) -> [&[Trace]; 41] {
        [
            &self.speed,
            &self.vehicle_motion_gyro,
            &self.vehicle_motion_accel,
            &self.sat_cn0,
            &self.imu_raw_gyro,
            &self.imu_raw_accel,
            &self.imu_cal_gyro,
            &self.imu_cal_accel,
            &self.orientation,
            &self.other,
            &self.ekf_cmp_pos,
            &self.ekf_cmp_vel,
            &self.ekf_cmp_att,
            &self.ekf_meas_gyro,
            &self.ekf_meas_accel,
            &self.ekf_bias_gyro,
            &self.ekf_bias_accel,
            &self.ekf_cov_bias,
            &self.ekf_cov_nonbias,
            &self.ekf_mount_sigma,
            &self.ekf_mount_dx,
            &self.ekf_nhc_mount_dx,
            &self.ekf_nhc_innovation,
            &self.ekf_nhc_nis,
            &self.ekf_nhc_h_mount_norm,
            &self.ekf_misalignment,
            &self.ekf_stationary_diag,
            &self.ekf_bump_pitch_speed,
            &self.ekf_bump_diag,
            &self.ekf_road_roughness,
            &self.ekf_road_event_motion,
            &self.ekf_map,
            &self.align_cmp_att,
            &self.align_res_vel,
            &self.align_axis_err,
            &self.align_motion,
            &self.align_flags,
            &self.align_roll_contrib,
            &self.align_pitch_contrib,
            &self.align_yaw_contrib,
            &self.align_cov,
        ]
    }
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Default)]
pub struct UpdateInspectorSample {
    pub t_s: f64,
    pub filter: String,
    pub update: String,
    pub residual: Option<f64>,
    pub nis: Option<f64>,
    pub contributions: Vec<StateContribution>,
    pub correlations: Vec<StateCorrelation>,
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Default)]
pub struct StateContribution {
    pub state: String,
    pub group: String,
    pub unit: String,
    pub value: f64,
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Default)]
pub struct StateCorrelation {
    pub state: String,
    pub group: String,
    pub mount_axis: String,
    pub value: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VisualizerMountMode {
    #[default]
    Auto,
    Manual,
}

impl VisualizerMountMode {
    pub fn from_cli_value(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "ref" | "reference" | "manual" => Ok(Self::Manual),
            _ => Err(format!(
                "invalid misalignment '{s}', expected 'auto' or 'manual'"
            )),
        }
    }

    pub fn uses_ref_mount(self) -> bool {
        matches!(self, Self::Manual)
    }

    pub fn uses_align_mount(self) -> bool {
        matches!(self, Self::Auto)
    }

    pub fn cli_value(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Auto => "auto",
        }
    }
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Copy, Default)]
pub struct HeadingSample {
    pub t_s: f64,
    pub lon_deg: f64,
    pub lat_deg: f64,
    pub yaw_deg: f64,
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Default)]
pub struct MapCursorSample {
    pub trace_name: String,
    pub t_s: f64,
    pub lon_deg: f64,
    pub lat_deg: f64,
    pub yaw_deg: Option<f64>,
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Default)]
pub struct RoadEventSample {
    pub kind: String,
    pub t_s: f64,
    pub lon_deg: f64,
    pub lat_deg: f64,
    pub confidence: f64,
    pub speed_mps: f64,
    pub trigger_window_start_t_s: f64,
    pub trigger_window_end_t_s: f64,
    pub trigger_traces: Vec<Trace>,
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Default)]
pub struct RoadSegmentSample {
    pub kind: String,
    pub start_t_s: f64,
    pub end_t_s: f64,
    pub duration_s: f64,
    pub mean_pitch_deg: f64,
    pub peak_abs_pitch_deg: f64,
    pub mean_speed_mps: f64,
    pub peak_speed_mps: f64,
    pub delta_speed_mps: f64,
    pub mean_accel_mps2: f64,
    pub peak_accel_mps2: f64,
    pub trigger_window_start_t_s: f64,
    pub trigger_window_end_t_s: f64,
    pub trigger_traces: Vec<Trace>,
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Copy, Default)]
pub struct TripEventCountsSample {
    pub speed_bumps: u32,
    pub road_shocks: u32,
    pub rough_road: u32,
    pub uphill: u32,
    pub downhill: u32,
    pub reverse: u32,
    pub harsh_acceleration: u32,
    pub harsh_braking: u32,
    pub harsh_cornering: u32,
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Copy, Default)]
pub struct TripSummarySample {
    pub sample_count: u32,
    pub invalid_sample_count: u32,
    pub data_gap_count: u32,
    pub max_sample_gap_s: f64,
    pub total_gap_duration_s: f64,
    pub duration_s: f64,
    pub moving_duration_s: f64,
    pub stationary_duration_s: f64,
    pub distance_m: f64,
    pub reverse_duration_s: f64,
    pub reverse_distance_m: f64,
    pub elevation_gain_m: f64,
    pub elevation_loss_m: f64,
    pub elevation_valid: bool,
    pub mean_speed_mps: f64,
    pub moving_mean_speed_mps: f64,
    pub peak_speed_mps: f64,
    pub peak_accel_mps2: f64,
    pub peak_decel_mps2: f64,
    pub peak_lateral_accel_mps2: f64,
    pub rolling_speed_mps: f64,
    pub rolling_abs_longitudinal_accel_mps2: f64,
    pub rolling_abs_lateral_accel_mps2: f64,
    pub events: TripEventCountsSample,
    pub speed_bumps_per_km: f64,
    pub road_shocks_per_km: f64,
    pub rough_road_events_per_km: f64,
    pub harsh_events_per_km: f64,
    pub reverse_seconds_per_km: f64,
    pub road_roughness_rms_mps2: f64,
    pub road_roughness_level: u8,
    pub road_roughness_distance_m: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Overview,
    Motion,
    Mount,
    Calibration,
    Sensors,
    Events,
    Diagnostics,
}

#[cfg(test)]
mod tests {
    use super::{PlotData, RoadEventSample, Trace};

    #[test]
    fn trace_by_name_ignores_empty_traces_and_finds_populated_trace() {
        let data = PlotData {
            ekf_cmp_att: vec![
                Trace {
                    name: "EKF roll [deg]".to_string(),
                    points: Vec::new(),
                },
                Trace {
                    name: "EKF pitch [deg]".to_string(),
                    points: vec![[1.0, 2.0]],
                },
            ],
            ..PlotData::default()
        };

        assert!(data.trace_by_name("EKF roll [deg]").is_none());
        assert_eq!(
            data.trace_by_name("EKF pitch [deg]")
                .map(|trace| trace.points.as_slice()),
            Some(&[[1.0, 2.0]][..])
        );
        assert!(data.has_trace_points());
    }

    #[test]
    fn event_motion_traces_are_part_of_global_trace_lookup() {
        let data = PlotData {
            ekf_road_event_motion: vec![Trace {
                name: "Event lateral accel [m/s^2]".to_string(),
                points: vec![[2.0, 3.0]],
            }],
            ..PlotData::default()
        };

        assert_eq!(
            data.trace_by_name("Event lateral accel [m/s^2]")
                .map(|trace| trace.points.as_slice()),
            Some(&[[2.0, 3.0]][..])
        );
    }

    #[test]
    fn road_event_samples_can_carry_hover_trigger_traces() {
        let event = RoadEventSample {
            kind: "speed bump".to_string(),
            t_s: 10.0,
            lon_deg: 1.0,
            lat_deg: 2.0,
            confidence: 0.9,
            speed_mps: 5.0,
            trigger_window_start_t_s: 5.0,
            trigger_window_end_t_s: 15.0,
            trigger_traces: vec![Trace {
                name: "Vertical accel raw [m/s^2]".to_string(),
                points: vec![[9.0, 1.0], [10.0, -1.0]],
            }],
        };

        assert_eq!(event.trigger_traces.len(), 1);
        assert_eq!(
            event.trigger_window_end_t_s - event.trigger_window_start_t_s,
            10.0
        );
    }
}
