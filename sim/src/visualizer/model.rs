#[derive(Clone, Default)]
pub struct Trace {
    pub name: String,
    pub points: Vec<[f64; 2]>,
}

#[derive(Default)]
pub struct PlotData {
    pub speed: Vec<Trace>,
    pub sat_cn0: Vec<Trace>,
    pub imu_raw_gyro: Vec<Trace>,
    pub imu_raw_accel: Vec<Trace>,
    pub imu_cal_gyro: Vec<Trace>,
    pub imu_cal_accel: Vec<Trace>,
    pub esf_ins_gyro: Vec<Trace>,
    pub esf_ins_accel: Vec<Trace>,
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
    pub eskf_stationary_diag: Vec<Trace>,
    pub eskf_bump_pitch_speed: Vec<Trace>,
    pub eskf_bump_diag: Vec<Trace>,
    pub eskf_map: Vec<Trace>,
    pub eskf_map_heading: Vec<HeadingSample>,
    pub loose_cmp_pos: Vec<Trace>,
    pub loose_cmp_vel: Vec<Trace>,
    pub loose_cmp_att: Vec<Trace>,
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
    pub align_roll_contrib: Vec<Trace>,
    pub align_pitch_contrib: Vec<Trace>,
    pub align_yaw_contrib: Vec<Trace>,
    pub align_cov: Vec<Trace>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EkfImuSource {
    #[default]
    Align,
    EsfAlg,
}

#[derive(Clone, Copy, Default)]
pub struct HeadingSample {
    pub t_s: f64,
    pub lon_deg: f64,
    pub lat_deg: f64,
    pub yaw_deg: f64,
}

#[derive(Clone, Copy)]
pub struct AlgEvent {
    pub t_ms: f64,
    pub roll_deg: f64,
    pub pitch_deg: f64,
    pub yaw_deg: f64,
}

#[derive(Clone, Copy)]
pub struct NavAttEvent {
    pub t_ms: f64,
    pub roll_deg: f64,
    pub pitch_deg: f64,
    pub heading_deg: f64,
}

#[derive(Clone, Copy)]
pub struct ImuPacket {
    pub t_ms: f64,
    pub gx_dps: f64,
    pub gy_dps: f64,
    pub gz_dps: f64,
    pub ax_mps2: f64,
    pub ay_mps2: f64,
    pub az_mps2: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Signals,
    EskfCompare,
    LooseCompare,
    EskfBump,
    AlignCompare,
    MapDark,
}
