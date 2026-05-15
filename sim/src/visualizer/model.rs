#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Default)]
pub struct Trace {
    pub name: String,
    pub points: Vec<[f64; 2]>,
}

#[cfg_attr(target_arch = "wasm32", derive(serde::Deserialize, serde::Serialize))]
#[derive(Default)]
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Overview,
    Motion,
    Mount,
    Calibration,
    Sensors,
    Diagnostics,
}
