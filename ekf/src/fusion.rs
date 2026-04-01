use alloc::vec::Vec;

use crate::align::{Align, AlignConfig, AlignWindowSummary};
use crate::ekf::{
    Ekf, GpsData, ImuSample, PredictNoise, ekf_fuse_body_vel, ekf_fuse_gps, ekf_predict,
    ekf_set_predict_noise,
};
#[derive(Clone, Copy, Debug)]
pub struct FusionBootstrapConfig {
    pub ema_alpha: f32,
    pub max_speed_mps: f32,
    pub stationary_samples: usize,
    pub max_gyro_radps: f32,
    pub max_accel_norm_err_mps2: f32,
}

impl Default for FusionBootstrapConfig {
    fn default() -> Self {
        let align = AlignConfig::default();
        Self {
            ema_alpha: 0.05,
            max_speed_mps: 0.35,
            stationary_samples: 100,
            max_gyro_radps: align.max_stationary_gyro_radps,
            max_accel_norm_err_mps2: align.max_stationary_accel_norm_err_mps2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FusionConfig {
    pub align: AlignConfig,
    pub bootstrap: FusionBootstrapConfig,
    pub predict_noise: PredictNoise,
    pub r_body_vel: f32,
    pub yaw_init_speed_mps: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            align: AlignConfig::default(),
            bootstrap: FusionBootstrapConfig::default(),
            predict_noise: PredictNoise::default(),
            r_body_vel: 5.0,
            yaw_init_speed_mps: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FusionImuSample {
    pub t_s: f64,
    pub gyro_radps: [f32; 3],
    pub accel_mps2: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct FusionGnssSample {
    pub t_s: f64,
    pub pos_ned_m: [f32; 3],
    pub vel_ned_mps: [f32; 3],
    pub pos_std_m: [f32; 3],
    pub vel_std_mps: [f32; 3],
    pub heading_rad: Option<f32>,
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
pub enum MisalignmentMode {
    InternalAlign,
    External([f32; 4]),
}

#[derive(Clone, Copy, Debug)]
struct TurnIntervalSummary {
    dt_s: f32,
    mean_gyro_b: [f32; 3],
    mean_accel_b: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
struct BootstrapImuSample {
    t_s: f64,
    gyro_radps: [f32; 3],
    accel_mps2: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
struct BootstrapGnssState {
    t_s: f64,
    vel_ned_mps: [f32; 3],
}

#[derive(Debug, Clone)]
struct BootstrapDetector {
    cfg: FusionBootstrapConfig,
    gyro_ema: Option<f32>,
    accel_err_ema: Option<f32>,
    speed_ema: Option<f32>,
    stationary_accel: Vec<[f32; 3]>,
}

#[derive(Debug, Clone)]
enum MountAlignment {
    Internal {
        align: Align,
        bootstrap: BootstrapDetector,
        align_initialized: bool,
    },
    External,
}

#[derive(Debug, Clone)]
pub struct SensorFusion {
    cfg: FusionConfig,
    ekf: Ekf,
    alignment: MountAlignment,
    mount_q_vb: Option<[f32; 4]>,
    mount_ready: bool,
    ekf_initialized: bool,
    last_imu_t_s: Option<f64>,
    last_gnss: Option<FusionGnssSample>,
    bootstrap_prev_gnss: Option<BootstrapGnssState>,
    interval_imu_sum_gyro: [f32; 3],
    interval_imu_sum_accel: [f32; 3],
    interval_imu_count: usize,
    interval_bootstrap_imu: Vec<BootstrapImuSample>,
}

impl SensorFusion {
    pub fn new(cfg: FusionConfig) -> Self {
        Self::with_misalignment_mode(cfg, MisalignmentMode::InternalAlign)
    }

    pub fn with_misalignment(cfg: FusionConfig, q_vb: [f32; 4]) -> Self {
        Self::with_misalignment_mode(cfg, MisalignmentMode::External(q_vb))
    }

    pub fn with_misalignment_mode(cfg: FusionConfig, mode: MisalignmentMode) -> Self {
        let mut ekf = Ekf::default();
        ekf_set_predict_noise(&mut ekf, cfg.predict_noise);
        let (alignment, mount_q_vb, mount_ready) = match mode {
            MisalignmentMode::InternalAlign => (
                MountAlignment::Internal {
                    align: Align::new(cfg.align),
                    bootstrap: BootstrapDetector::new(cfg.bootstrap),
                    align_initialized: false,
                },
                None,
                false,
            ),
            MisalignmentMode::External(q_vb) => (MountAlignment::External, Some(q_vb), true),
        };
        Self {
            cfg,
            ekf,
            alignment,
            mount_q_vb,
            mount_ready,
            ekf_initialized: false,
            last_imu_t_s: None,
            last_gnss: None,
            bootstrap_prev_gnss: None,
            interval_imu_sum_gyro: [0.0; 3],
            interval_imu_sum_accel: [0.0; 3],
            interval_imu_count: 0,
            interval_bootstrap_imu: Vec::new(),
        }
    }

    pub fn set_misalignment(&mut self, q_vb: [f32; 4]) {
        self.alignment = MountAlignment::External;
        self.mount_q_vb = Some(q_vb);
        self.mount_ready = true;
    }

    pub fn process_imu(&mut self, sample: FusionImuSample) -> FusionUpdate {
        self.interval_imu_sum_gyro = vec3_add(self.interval_imu_sum_gyro, sample.gyro_radps);
        self.interval_imu_sum_accel = vec3_add(self.interval_imu_sum_accel, sample.accel_mps2);
        self.interval_imu_count += 1;

        if matches!(self.alignment, MountAlignment::Internal { .. }) {
            self.interval_bootstrap_imu.push(BootstrapImuSample {
                t_s: sample.t_s,
                gyro_radps: sample.gyro_radps,
                accel_mps2: sample.accel_mps2,
            });
        }

        let dt_s = match self.last_imu_t_s {
            Some(prev_t_s) => (sample.t_s - prev_t_s) as f32,
            None => {
                self.last_imu_t_s = Some(sample.t_s);
                return self.status(false, false);
            }
        };
        self.last_imu_t_s = Some(sample.t_s);

        if !self.ekf_initialized || !self.mount_ready {
            return self.status(false, false);
        }
        let Some(q_vb) = self.mount_q_vb else {
            return self.status(false, false);
        };

        if !(0.001..=0.05).contains(&dt_s) {
            return self.status(false, false);
        }

        let c_bv = transpose3(quat_to_rotmat(q_vb));
        let gyro_vehicle = mat3_vec(c_bv, sample.gyro_radps);
        let accel_vehicle = mat3_vec(c_bv, sample.accel_mps2);
        let imu = ImuSample {
            dax: gyro_vehicle[0] * dt_s,
            day: gyro_vehicle[1] * dt_s,
            daz: gyro_vehicle[2] * dt_s,
            dvx: accel_vehicle[0] * dt_s,
            dvy: accel_vehicle[1] * dt_s,
            dvz: accel_vehicle[2] * dt_s,
            dt: dt_s,
        };
        ekf_predict(&mut self.ekf, &imu, None);
        clamp_ekf_biases(&mut self.ekf, dt_s as f64);
        ekf_fuse_body_vel(&mut self.ekf, self.cfg.r_body_vel);
        clamp_ekf_biases(&mut self.ekf, dt_s as f64);
        self.status(false, false)
    }

    pub fn process_gnss(&mut self, gnss: FusionGnssSample) -> FusionUpdate {
        let prev_mount_ready = self.mount_ready;
        let interval_summary = if matches!(
            self.alignment,
            MountAlignment::Internal {
                align_initialized: true,
                ..
            }
        ) {
            self.last_gnss
                .map(|prev_gnss| (prev_gnss, self.take_interval_summary(prev_gnss.t_s, gnss.t_s)))
        } else {
            None
        };

        if let MountAlignment::Internal {
            align,
            bootstrap,
            align_initialized,
        } = &mut self.alignment
        {
            if !*align_initialized {
                if let Some(prev_gnss) = self.bootstrap_prev_gnss {
                    for sample in &self.interval_bootstrap_imu {
                        let speed_mps = interp_speed(prev_gnss, gnss, sample.t_s);
                        if bootstrap.update(sample.accel_mps2, sample.gyro_radps, speed_mps)
                            && align
                                .initialize_from_stationary(&bootstrap.stationary_accel, 0.0)
                                .is_ok()
                        {
                            *align_initialized = true;
                            self.mount_q_vb = Some(align.q_vb);
                            break;
                        }
                    }
                }
                self.bootstrap_prev_gnss = Some(BootstrapGnssState {
                    t_s: gnss.t_s,
                    vel_ned_mps: gnss.vel_ned_mps,
                });
            }

            if *align_initialized {
                if let Some((prev_gnss, Some(summary))) = interval_summary
                {
                    let window = AlignWindowSummary {
                        dt: summary.dt_s,
                        mean_gyro_b: summary.mean_gyro_b,
                        mean_accel_b: summary.mean_accel_b,
                        gnss_vel_prev_n: prev_gnss.vel_ned_mps,
                        gnss_vel_curr_n: gnss.vel_ned_mps,
                    };
                    let (_, trace) = align.update_window_with_trace(&window);
                    self.mount_q_vb = Some(align.q_vb);
                    self.mount_ready = trace.coarse_alignment_ready;
                }
            }
        } else {
            self.clear_interval_summary();
        }

        let mut ekf_initialized_now = false;
        if self.mount_ready {
            if !self.ekf_initialized {
                initialize_ekf_from_gnss(&mut self.ekf, gnss, self.cfg.yaw_init_speed_mps);
                self.ekf_initialized = true;
                ekf_initialized_now = true;
            } else {
                let gps = GpsData {
                    pos_n: gnss.pos_ned_m[0],
                    pos_e: gnss.pos_ned_m[1],
                    pos_d: gnss.pos_ned_m[2],
                    vel_n: gnss.vel_ned_mps[0],
                    vel_e: gnss.vel_ned_mps[1],
                    vel_d: gnss.vel_ned_mps[2],
                    R_POS_N: gnss.pos_std_m[0].max(0.01).powi(2),
                    R_POS_E: gnss.pos_std_m[1].max(0.01).powi(2),
                    R_POS_D: gnss.pos_std_m[2].max(0.01).powi(2),
                    R_VEL_N: gnss.vel_std_mps[0].max(0.01).powi(2),
                    R_VEL_E: gnss.vel_std_mps[1].max(0.01).powi(2),
                    R_VEL_D: gnss.vel_std_mps[2].max(0.01).powi(2),
                };
                ekf_fuse_gps(&mut self.ekf, &gps);
            }
        }

        self.last_gnss = Some(gnss);
        self.interval_bootstrap_imu.clear();
        self.status(prev_mount_ready != self.mount_ready, ekf_initialized_now)
    }

    pub fn ekf(&self) -> Option<&Ekf> {
        self.ekf_initialized.then_some(&self.ekf)
    }

    pub fn ekf_mut(&mut self) -> Option<&mut Ekf> {
        self.ekf_initialized.then_some(&mut self.ekf)
    }

    pub fn mount_q_vb(&self) -> Option<[f32; 4]> {
        self.mount_q_vb
    }

    pub fn mount_ready(&self) -> bool {
        self.mount_ready
    }

    pub fn align(&self) -> Option<&Align> {
        match &self.alignment {
            MountAlignment::Internal { align, .. } => Some(align),
            MountAlignment::External => None,
        }
    }

    fn status(&self, mount_ready_changed: bool, ekf_initialized_now: bool) -> FusionUpdate {
        FusionUpdate {
            mount_ready: self.mount_ready,
            mount_ready_changed,
            ekf_initialized: self.ekf_initialized,
            ekf_initialized_now,
            mount_q_vb: self.mount_q_vb,
        }
    }

    fn take_interval_summary(&mut self, t0_s: f64, t1_s: f64) -> Option<TurnIntervalSummary> {
        if self.interval_imu_count == 0 {
            return None;
        }
        let dt_s = (t1_s - t0_s).max(1.0e-3) as f32;
        let inv_n = 1.0 / self.interval_imu_count as f32;
        let summary = TurnIntervalSummary {
            dt_s,
            mean_gyro_b: vec3_scale(self.interval_imu_sum_gyro, inv_n),
            mean_accel_b: vec3_scale(self.interval_imu_sum_accel, inv_n),
        };
        self.clear_interval_summary();
        Some(summary)
    }

    fn clear_interval_summary(&mut self) {
        self.interval_imu_sum_gyro = [0.0; 3];
        self.interval_imu_sum_accel = [0.0; 3];
        self.interval_imu_count = 0;
    }
}

impl BootstrapDetector {
    fn new(cfg: FusionBootstrapConfig) -> Self {
        Self {
            cfg,
            gyro_ema: None,
            accel_err_ema: None,
            speed_ema: None,
            stationary_accel: Vec::new(),
        }
    }

    fn update(&mut self, accel_b: [f32; 3], gyro_radps: [f32; 3], speed_mps: f32) -> bool {
        let gyro_norm = norm3(gyro_radps);
        let accel_err = (norm3(accel_b) - crate::align::GRAVITY_MPS2).abs();
        self.gyro_ema = Some(ema_update(self.gyro_ema, gyro_norm, self.cfg.ema_alpha));
        self.accel_err_ema = Some(ema_update(
            self.accel_err_ema,
            accel_err,
            self.cfg.ema_alpha,
        ));
        self.speed_ema = Some(ema_update(self.speed_ema, speed_mps, self.cfg.ema_alpha));

        let stationary = self.speed_ema.unwrap_or(speed_mps) <= self.cfg.max_speed_mps
            && self.gyro_ema.unwrap_or(gyro_norm) <= self.cfg.max_gyro_radps
            && self.accel_err_ema.unwrap_or(accel_err) <= self.cfg.max_accel_norm_err_mps2;
        if stationary {
            self.stationary_accel.push(accel_b);
        } else {
            self.stationary_accel.clear();
        }
        self.stationary_accel.len() >= self.cfg.stationary_samples
    }
}

fn initialize_ekf_from_gnss(ekf: &mut Ekf, gnss: FusionGnssSample, yaw_init_speed_mps: f32) {
    *ekf = Ekf::default();
    ekf.state.pn = gnss.pos_ned_m[0];
    ekf.state.pe = gnss.pos_ned_m[1];
    ekf.state.pd = gnss.pos_ned_m[2];
    ekf.state.vn = gnss.vel_ned_mps[0];
    ekf.state.ve = gnss.vel_ned_mps[1];
    ekf.state.vd = gnss.vel_ned_mps[2];

    let speed_h = (gnss.vel_ned_mps[0] * gnss.vel_ned_mps[0]
        + gnss.vel_ned_mps[1] * gnss.vel_ned_mps[1])
        .sqrt();
    let yaw_rad = gnss.heading_rad.unwrap_or_else(|| {
        if speed_h >= yaw_init_speed_mps.max(1.0) {
            gnss.vel_ned_mps[1].atan2(gnss.vel_ned_mps[0])
        } else {
            0.0
        }
    });
    set_state_yaw_only(&mut ekf.state, yaw_rad);

    let att_sigma_rad = 2.0_f32.to_radians();
    let quat_var = 0.25 * att_sigma_rad * att_sigma_rad;
    for i in 0..4 {
        ekf.p[i][i] = quat_var;
    }
    let vel_var = gnss
        .vel_std_mps
        .iter()
        .copied()
        .fold(0.2_f32, f32::max)
        .powi(2);
    ekf.p[4][4] = vel_var;
    ekf.p[5][5] = vel_var;
    ekf.p[6][6] = vel_var;
    ekf.p[7][7] = gnss.pos_std_m[0].max(0.5).powi(2);
    ekf.p[8][8] = gnss.pos_std_m[1].max(0.5).powi(2);
    ekf.p[9][9] = gnss.pos_std_m[2].max(0.5).powi(2);
}

fn set_state_yaw_only(state: &mut crate::ekf::EkfState, yaw_rad: f32) {
    let half = 0.5 * yaw_rad;
    state.q0 = half.cos();
    state.q1 = 0.0;
    state.q2 = 0.0;
    state.q3 = half.sin();
}

fn interp_speed(prev: BootstrapGnssState, curr: FusionGnssSample, t_s: f64) -> f32 {
    let speed_prev = horizontal_speed(prev.vel_ned_mps);
    let speed_curr = horizontal_speed(curr.vel_ned_mps);
    let dt = curr.t_s - prev.t_s;
    if dt <= 1.0e-6 {
        return speed_curr;
    }
    let alpha = ((t_s - prev.t_s) / dt).clamp(0.0, 1.0) as f32;
    speed_prev + alpha * (speed_curr - speed_prev)
}

fn horizontal_speed(v_ned_mps: [f32; 3]) -> f32 {
    (v_ned_mps[0] * v_ned_mps[0] + v_ned_mps[1] * v_ned_mps[1]).sqrt()
}

fn ema_update(prev: Option<f32>, sample: f32, alpha: f32) -> f32 {
    let alpha = alpha.clamp(1.0e-4, 1.0);
    match prev {
        Some(prev) => (1.0 - alpha) * prev + alpha * sample,
        None => sample,
    }
}

fn clamp_ekf_biases(ekf: &mut Ekf, dt_s: f64) {
    let dt = dt_s.max(1.0e-3) as f32;
    let max_gyro_bias_da = (1.5_f32.to_radians()) * dt;
    let max_accel_bias_dv = 1.5 * dt;
    ekf.state.dax_b = ekf.state.dax_b.clamp(-max_gyro_bias_da, max_gyro_bias_da);
    ekf.state.day_b = ekf.state.day_b.clamp(-max_gyro_bias_da, max_gyro_bias_da);
    ekf.state.daz_b = ekf.state.daz_b.clamp(-max_gyro_bias_da, max_gyro_bias_da);
    ekf.state.dvx_b = ekf.state.dvx_b.clamp(-max_accel_bias_dv, max_accel_bias_dv);
    ekf.state.dvy_b = ekf.state.dvy_b.clamp(-max_accel_bias_dv, max_accel_bias_dv);
    ekf.state.dvz_b = ekf.state.dvz_b.clamp(-max_accel_bias_dv, max_accel_bias_dv);
}

fn quat_to_rotmat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    let inv = if n2 > 1.0e-9 { 1.0 / n2.sqrt() } else { 1.0 };
    let q0 = q[0] * inv;
    let q1 = q[1] * inv;
    let q2 = q[2] * inv;
    let q3 = q[3] * inv;
    [
        [
            1.0 - 2.0 * (q2 * q2 + q3 * q3),
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * (q1 * q1 + q3 * q3),
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * (q1 * q1 + q2 * q2),
        ],
    ]
}

fn transpose3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mat3_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
