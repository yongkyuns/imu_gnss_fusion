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
//! - Direction cosine matrix `C_ab` maps coordinates from frame `b` to frame
//!   `a`: `x_a = C_ab x_b`.
//! - Quaternion `q_ab` follows `R(q_ab) = C_ab`; products compose as
//!   `R(q1 * q2) = R(q1) R(q2)`.
//! - The mount quaternion stored in `q_bv0..q_bv3` by both filters is the current
//!   physical vehicle-to-body mount: `R(q_bv) = C_bv`, `x_b = C_bv x_v`. The
//!   filters use `C_vb = C_bv^T` to rotate raw IMU vectors into the vehicle
//!   frame during propagation.
//! - Reduced attitude `q0..q3` is `q_nv`: the NED/navigation-frame attitude
//!   with respect to the vehicle frame, with `R(q_nv) = C_nv` and
//!   `x_n = C_nv x_v`.
//! - Full attitude `q0..q3` is `q_ev`: the ECEF-frame attitude with respect to
//!   the vehicle frame, with `R(q_ev) = C_ev` and `x_e = C_ev x_v`.
//!
//! Maintained mathematical references:
//! `docs/align.pdf`, `docs/reduced.pdf`, and `docs/full.pdf`.

#![no_std]
#![allow(clippy::needless_range_loop)]

#[cfg(test)]
mod coordinate_conventions;
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
    AlignDebug, Config, Filter, GnssSample, ImuSample, MountMode, SensorFusion, Update,
    VehicleSpeedDirection, VehicleSpeedSample,
};
pub use noise::ProcessNoise;
