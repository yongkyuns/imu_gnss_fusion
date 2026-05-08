//! Sensor fusion filters for IMU/GNSS experiments.
//!
//! The crate exposes [`SensorFusion`] as the high-level runtime facade, with
//! standalone [`align`], [`reduced`], and [`full`] modules for focused filter
//! work and diagnostics. Public APIs use SI units unless a field name states
//! otherwise.
//!
//! Frame and quaternion convention:
//!
//! - `b`: raw IMU body/sensor frame.
//! - `v`: vehicle frame, forward-right-down.
//! - `n`: local NED navigation frame used by [`reduced`].
//! - `e`: ECEF frame used by [`full`].
//! - Active rotations use `x_a = C_ab x_b` and quaternion products compose as
//!   `C(q1 * q2) = C(q1) C(q2)`.
//! - The mount quaternion stored in `qcs0..qcs3` by both filters is the current
//!   physical vehicle-to-body mount. Its DCM maps `x_v` into `x_b`; the filters
//!   use its transpose to rotate raw IMU vectors into the vehicle frame during
//!   propagation.
//! - Reduced attitude `q0..q3` maps vehicle frame to local NED (`q_nv`). Full
//!   attitude `q0..q3` maps vehicle frame to ECEF (`q_ev`).
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

pub use fusion::{
    AlignDebug, Config, Filter, GnssSample, ImuSample, MountMode, MountSource, SensorFusion,
    Update, VehicleSpeedDirection, VehicleSpeedSample,
};
pub use noise::ProcessNoise;
