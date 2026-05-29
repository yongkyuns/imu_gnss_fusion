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
//! Maintained mathematical references live in the Sphinx docs under `docs/math/`.

#![no_std]
#![allow(clippy::needless_range_loop)]

#[cfg(test)]
mod coordinate_conventions;
mod covariance;
pub mod diagnostics;
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

pub use diagnostics::{
    FUSION_HEALTH_REASON_BIAS_UNSTABLE, FUSION_HEALTH_REASON_COVARIANCE_HIGH,
    FUSION_HEALTH_REASON_GNSS_REJECTING, FUSION_HEALTH_REASON_GNSS_STALE,
    FUSION_HEALTH_REASON_INSUFFICIENT_MOTION, FUSION_HEALTH_REASON_INSUFFICIENT_TIME,
    FUSION_HEALTH_REASON_MOUNT_NOT_READY, FUSION_HEALTH_REASON_MOUNT_UNSTABLE,
    FUSION_HEALTH_REASON_NAV_UNUSABLE, FUSION_HEALTH_REASON_NOT_INITIALIZED,
    FUSION_HEALTH_REASON_NUMERIC_INVALID, FUSION_HEALTH_REASON_SLEEP_GAP,
    FUSION_HEALTH_REASON_TAIL_TOO_SHORT, FusionHealth, FusionHealthMetrics, FusionState,
};
pub use fusion::{
    AlignDebug, Config, GNSS_EVENT_POSITION_ACCURACY_BYPASS,
    GNSS_EVENT_POSITION_CONSECUTIVE_REJECTED, GNSS_EVENT_POSITION_GAP_BYPASS,
    GNSS_EVENT_POSITION_REJECTED, GNSS_EVENT_VELOCITY_ACCURACY_BYPASS,
    GNSS_EVENT_VELOCITY_CONSECUTIVE_REJECTED, GNSS_EVENT_VELOCITY_GAP_BYPASS,
    GNSS_EVENT_VELOCITY_REJECTED, GnssSample, ImuSample, MountMode, SensorFusion, Update,
    VehicleSpeedDirection, VehicleSpeedSample,
};
pub use noise::ProcessNoise;
