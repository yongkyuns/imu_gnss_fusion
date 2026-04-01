use crate::align::{Align, AlignConfig};
use crate::c_api::CSensorFusionWrapper;
use crate::ekf::{Ekf, PredictNoise};

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
    pub t_s: f32,
    pub gyro_radps: [f32; 3],
    pub accel_mps2: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct FusionGnssSample {
    pub t_s: f32,
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

#[derive(Debug)]
pub struct SensorFusion {
    cfg: FusionConfig,
    raw: CSensorFusionWrapper,
    cached_align: Option<Align>,
}

impl SensorFusion {
    pub fn new(cfg: FusionConfig) -> Self {
        Self::with_misalignment_mode(cfg, MisalignmentMode::InternalAlign)
    }

    pub fn with_misalignment(cfg: FusionConfig, q_vb: [f32; 4]) -> Self {
        Self::with_misalignment_mode(cfg, MisalignmentMode::External(q_vb))
    }

    pub fn with_misalignment_mode(cfg: FusionConfig, mode: MisalignmentMode) -> Self {
        let raw = match mode {
            MisalignmentMode::InternalAlign => CSensorFusionWrapper::new_internal(cfg),
            MisalignmentMode::External(q_vb) => CSensorFusionWrapper::new_external(cfg, q_vb),
        };
        let mut out = Self {
            cfg,
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

    pub fn ekf(&self) -> Option<&Ekf> {
        self.raw.ekf()
    }

    pub fn ekf_mut(&mut self) -> Option<&mut Ekf> {
        self.raw.ekf_mut()
    }

    pub fn mount_q_vb(&self) -> Option<[f32; 4]> {
        self.raw.mount_q_vb()
    }

    pub fn mount_ready(&self) -> bool {
        self.raw.mount_ready()
    }

    pub fn align(&self) -> Option<&Align> {
        self.cached_align.as_ref()
    }

    fn refresh_align_snapshot(&mut self) {
        self.cached_align = self
            .raw
            .align_state()
            .map(|s| Align::from_c_state(self.cfg.align, *s));
    }
}
