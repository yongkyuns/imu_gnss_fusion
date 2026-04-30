//! Sensor-fusion filters for IMU/GNSS experiments.
//!
//! The crate exposes the runtime fusion facade, standalone alignment logic,
//! ESKF and loose-reference filters, and the generated symbolic model wrappers
//! used by those filters. Public APIs use SI units unless a field name states
//! otherwise.
//!
//! Maintained mathematical references:
//! `docs/align_nhc_formulation.pdf`, `docs/eskf_mount_formulation.pdf`, and
//! `docs/loose_formulation.pdf`.

#![allow(clippy::needless_range_loop)]

extern crate alloc;

/// Mount-alignment filter used to estimate the IMU-to-vehicle rotation.
pub mod align;
/// Legacy EKF support types and generated-model glue.
pub mod ekf;
/// Legacy ESKF support module.
pub mod eskf;
/// Public ESKF state, sample, and diagnostic data structures.
pub mod eskf_types;
/// High-level sensor-fusion facade that consumes timestamped IMU, GNSS, and vehicle-speed samples.
pub mod fusion;
/// Symbolic ESKF model wrapper around generated Rust include files.
pub mod generated_eskf;
/// Symbolic loose-filter model wrapper around generated Rust include files.
pub mod generated_loose;
/// Loose INS/GNSS reference filter used for comparison and diagnostics.
pub mod loose;
/// Runtime Rust ESKF implementation built on the generated model wrappers.
pub mod rust_eskf;
