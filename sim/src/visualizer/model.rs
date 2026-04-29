use sensor_fusion::fusion::EskfMountSource;

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
    pub eskf_cmp_pos: Vec<Trace>,
    pub eskf_cmp_vel: Vec<Trace>,
    pub eskf_cmp_att: Vec<Trace>,
    pub eskf_meas_gyro: Vec<Trace>,
    pub eskf_meas_accel: Vec<Trace>,
    pub eskf_bias_gyro: Vec<Trace>,
    pub eskf_bias_accel: Vec<Trace>,
    pub eskf_cov_bias: Vec<Trace>,
    pub eskf_cov_nonbias: Vec<Trace>,
    pub eskf_misalignment: Vec<Trace>,
    pub eskf_stationary_diag: Vec<Trace>,
    pub eskf_bump_pitch_speed: Vec<Trace>,
    pub eskf_bump_diag: Vec<Trace>,
    pub eskf_map: Vec<Trace>,
    pub eskf_map_heading: Vec<HeadingSample>,
    pub loose_cmp_pos: Vec<Trace>,
    pub loose_cmp_vel: Vec<Trace>,
    pub loose_cmp_att: Vec<Trace>,
    pub loose_nominal_att: Vec<Trace>,
    pub loose_residual_mount: Vec<Trace>,
    pub loose_misalignment: Vec<Trace>,
    pub loose_meas_gyro: Vec<Trace>,
    pub loose_meas_accel: Vec<Trace>,
    pub loose_bias_gyro: Vec<Trace>,
    pub loose_bias_accel: Vec<Trace>,
    pub loose_scale_gyro: Vec<Trace>,
    pub loose_scale_accel: Vec<Trace>,
    pub loose_cov_bias: Vec<Trace>,
    pub loose_cov_nonbias: Vec<Trace>,
    pub loose_map: Vec<Trace>,
    pub loose_map_heading: Vec<HeadingSample>,
    pub align_cmp_att: Vec<Trace>,
    pub align_res_vel: Vec<Trace>,
    pub align_axis_err: Vec<Trace>,
    pub align_motion: Vec<Trace>,
    pub align_flags: Vec<Trace>,
    pub align_roll_contrib: Vec<Trace>,
    pub align_pitch_contrib: Vec<Trace>,
    pub align_yaw_contrib: Vec<Trace>,
    pub align_cov: Vec<Trace>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EkfImuSource {
    #[default]
    Internal,
    External,
    Ref,
}

impl EkfImuSource {
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

    pub fn eskf_mount_source(self) -> EskfMountSource {
        match self {
            Self::External => EskfMountSource::FollowAlign,
            Self::Internal | Self::Ref => EskfMountSource::LatchedSeed,
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Overview,
    Map,
    Motion,
    Mount,
    Calibration,
    Sensors,
    Diagnostics,
}
