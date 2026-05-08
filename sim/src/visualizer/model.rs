use sensor_fusion::MountSource;

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
    pub sat_cn0: Vec<Trace>,
    pub imu_raw_gyro: Vec<Trace>,
    pub imu_raw_accel: Vec<Trace>,
    pub imu_cal_gyro: Vec<Trace>,
    pub imu_cal_accel: Vec<Trace>,
    pub orientation: Vec<Trace>,
    pub other: Vec<Trace>,
    pub reduced_cmp_pos: Vec<Trace>,
    pub reduced_cmp_vel: Vec<Trace>,
    pub reduced_cmp_att: Vec<Trace>,
    pub reduced_meas_gyro: Vec<Trace>,
    pub reduced_meas_accel: Vec<Trace>,
    pub reduced_bias_gyro: Vec<Trace>,
    pub reduced_bias_accel: Vec<Trace>,
    pub reduced_cov_bias: Vec<Trace>,
    pub reduced_cov_nonbias: Vec<Trace>,
    pub reduced_mount_sigma: Vec<Trace>,
    pub reduced_mount_dx: Vec<Trace>,
    pub reduced_nhc_mount_dx: Vec<Trace>,
    pub reduced_nhc_innovation: Vec<Trace>,
    pub reduced_nhc_nis: Vec<Trace>,
    pub reduced_nhc_h_mount_norm: Vec<Trace>,
    pub reduced_misalignment: Vec<Trace>,
    pub reduced_stationary_diag: Vec<Trace>,
    pub reduced_bump_pitch_speed: Vec<Trace>,
    pub reduced_bump_diag: Vec<Trace>,
    pub reduced_map: Vec<Trace>,
    pub map_cursor: Vec<MapCursorSample>,
    pub reduced_map_heading: Vec<HeadingSample>,
    pub full_cmp_pos: Vec<Trace>,
    pub full_cmp_vel: Vec<Trace>,
    pub full_cmp_att: Vec<Trace>,
    pub full_nominal_att: Vec<Trace>,
    pub full_mount: Vec<Trace>,
    pub full_misalignment: Vec<Trace>,
    pub full_meas_gyro: Vec<Trace>,
    pub full_meas_accel: Vec<Trace>,
    pub full_bias_gyro: Vec<Trace>,
    pub full_bias_accel: Vec<Trace>,
    pub full_scale_gyro: Vec<Trace>,
    pub full_scale_accel: Vec<Trace>,
    pub full_cov_bias: Vec<Trace>,
    pub full_cov_nonbias: Vec<Trace>,
    pub full_mount_sigma: Vec<Trace>,
    pub full_mount_dx: Vec<Trace>,
    pub full_nhc_innovation: Vec<Trace>,
    pub full_gnss_pos_gate: Vec<Trace>,
    pub full_map: Vec<Trace>,
    pub full_map_heading: Vec<HeadingSample>,
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
pub enum MountSourceMode {
    #[default]
    Internal,
    External,
    Ref,
}

impl MountSourceMode {
    pub fn from_cli_value(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "internal" | "auto" | "align" | "align-seed" | "seed" => Ok(Self::Internal),
            "external" | "follow-align" | "continuous-align" | "align-external" => {
                Ok(Self::External)
            }
            "ref" | "reference" | "manual" => Ok(Self::Ref),
            _ => Err(format!(
                "invalid misalignment '{s}', expected 'ref', 'external', or 'internal'"
            )),
        }
    }

    pub fn uses_ref_mount(self) -> bool {
        matches!(self, Self::Ref)
    }

    pub fn uses_align_mount(self) -> bool {
        matches!(self, Self::Internal | Self::External)
    }

    pub fn mount_source(self) -> MountSource {
        match self {
            Self::External => MountSource::FollowAlign,
            Self::Internal | Self::Ref => MountSource::LatchedSeed,
        }
    }

    pub fn cli_value(self) -> &'static str {
        match self {
            Self::Ref => "ref",
            Self::External => "external",
            Self::Internal => "internal",
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
