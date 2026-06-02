#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod c_backend;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod c_road_events;
pub mod config;
pub mod generic;
pub mod reference;
pub mod synthetic;

pub use config::{FusionTuningConfig, GnssOutageConfig, apply_fusion_tuning_config};
