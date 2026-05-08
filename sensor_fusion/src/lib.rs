//! Sensor fusion filters for IMU/GNSS experiments.
//!
//! The crate exposes [`SensorFusion`] as the high-level runtime facade, with
//! standalone [`align`], [`reduced`], and [`full`] modules for focused filter
//! work and diagnostics. Public APIs use SI units unless a field name states
//! otherwise.
//!
//! Maintained mathematical references:
//! `docs/align_nhc_formulation.pdf`, `docs/reduced_mount_formulation.pdf`, and
//! `docs/full_formulation.pdf`.

#![no_std]
#![allow(clippy::needless_range_loop)]

mod covariance;
mod fusion;
mod fusion_types;
mod math;
mod nav;
mod noise;

/// Mount-alignment filter used to estimate the IMU-to-vehicle rotation.
pub mod align;
/// Full INS/GNSS filter used for comparison and diagnostics.
pub mod full;
/// Symbolic full-filter model wrapper around generated Rust include files.
#[doc(hidden)]
pub mod generated_full {
    pub use crate::full::generated::*;
}
/// Symbolic Reduced model wrapper around generated Rust include files.
#[doc(hidden)]
pub mod generated_reduced {
    pub use crate::reduced::generated::*;
}
/// Reduced EKF runtime, public state structs, and standalone state helpers.
pub mod reduced;

pub use full::FullInitConfig;
pub use fusion::{
    AlignDebug, Config, Filter, GnssSample, ImuSample, MountMode, MountSource, SensorFusion,
    Update, VehicleSpeedDirection, VehicleSpeedSample,
};
pub use noise::ProcessNoise;
