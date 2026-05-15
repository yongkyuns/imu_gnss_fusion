//! Sensor fusion filters for IMU/GNSS experiments.
//!
//! The crate exposes [`SensorFusion`] as the high-level runtime facade, with
//! standalone [`align`] and [`ekf`] modules for focused filter work and
//! diagnostics. Public APIs use SI units unless a field name states otherwise.
//!
//! Frame and quaternion convention:
//!
//! - `b`: raw IMU body/sensor frame.
//! - `v`: vehicle frame, forward-right-down.
//! - `n`: local NED navigation frame used by [`ekf`].
//! - Direction cosine matrix `C_ab` maps coordinates from frame `b` to frame
//!   `a`: `x_a = C_ab x_b`.
//! - Quaternion `q_ab` follows `R(q_ab) = C_ab`; products compose as
//!   `R(q1 * q2) = R(q1) R(q2)`.
//! - The mount quaternion stored in `q_bv0..q_bv3` is the current physical
//!   vehicle-to-body mount: `R(q_bv) = C_bv`, `x_b = C_bv x_v`. The filter uses
//!   `C_vb = C_bv^T` to rotate raw IMU vectors into the vehicle frame during
//!   propagation.
//! - EKF attitude `q0..q3` is `q_nv`: the NED/navigation-frame attitude
//!   with respect to the vehicle frame, with `R(q_nv) = C_nv` and
//!   `x_n = C_nv x_v`.
//!
//! Maintained mathematical references:
//! `docs/align.pdf` and `docs/ekf.pdf`.

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
/// Symbolic EKF model wrapper around generated Rust include files.
#[doc(hidden)]
pub mod generated_ekf {
    pub use crate::ekf::generated::*;
}
/// EKF runtime, public state structs, and standalone state helpers.
pub mod ekf;

pub use fusion::{
    AlignDebug, Config, GnssSample, ImuSample, MountMode, SensorFusion, Update,
    VehicleSpeedDirection, VehicleSpeedSample,
};
pub use noise::ProcessNoise;
