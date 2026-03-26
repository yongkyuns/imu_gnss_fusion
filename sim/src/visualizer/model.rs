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
    pub ekf_cmp_pos: Vec<Trace>,
    pub ekf_cmp_vel: Vec<Trace>,
    pub ekf_cmp_att: Vec<Trace>,
    pub ekf_bias_gyro: Vec<Trace>,
    pub ekf_bias_accel: Vec<Trace>,
    pub ekf_cov_bias: Vec<Trace>,
    pub ekf_cov_nonbias: Vec<Trace>,
    pub ekf_map: Vec<Trace>,
    pub ekf_map_heading: Vec<HeadingSample>,
    pub misalign_cmp_att: Vec<Trace>,
    pub misalign_diag: Vec<Trace>,
    pub misalign_axis_err: Vec<Trace>,
    pub misalign_residuals: Vec<Trace>,
    pub misalign_gates: Vec<Trace>,
    pub misalign_cov: Vec<Trace>,
    pub align_cmp_att: Vec<Trace>,
    pub align_res_vel: Vec<Trace>,
    pub align_axis_err: Vec<Trace>,
    pub align_motion: Vec<Trace>,
    pub align_startup: Vec<Trace>,
    pub align_startup_angles: Vec<Trace>,
    pub align_startup_full_angles: Vec<Trace>,
    pub align_startup_esf_full_angles: Vec<Trace>,
    pub align_nhc_cmp_att: Vec<Trace>,
    pub align_nhc_diag: Vec<Trace>,
    pub align_nhc_axis_err: Vec<Trace>,
    pub align_nhc_residuals: Vec<Trace>,
    pub align_nhc_gates: Vec<Trace>,
    pub align_nhc_cov: Vec<Trace>,
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
    EkfCompare,
    MisalignCompare,
    AlignCompare,
    AlignStartup,
    AlignNhcCompare,
    MapDark,
}
