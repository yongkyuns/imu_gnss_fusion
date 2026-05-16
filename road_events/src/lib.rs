#![no_std]
//! Streaming road event detectors for embedded IMU/GNSS fusion outputs.
//!
//! The crate is organized as independent, small-state detectors that can run in
//! parallel from the same vehicle-motion stream:
//!
//! - [`SpeedBumpDetector`] detects front/rear-axle vertical impulse patterns.
//! - [`HillDetector`] detects sustained uphill/downhill pitch intervals.
//! - [`ReverseDetector`] detects sustained reverse longitudinal velocity.
//! - [`HarshAccelDetector`] and [`HarshBrakeDetector`] detect EMA-smoothed
//!   velocity-derivative events.
//! - [`HarshCornerDetector`] detects steady-turn lateral acceleration from
//!   `abs(yaw_rate * speed)`.

#[cfg(test)]
extern crate std;

mod bump;
mod common;
mod harsh;
mod hill;
mod reverse;
mod types;

pub use bump::SpeedBumpDetector;
pub use harsh::{HarshAccelDetector, HarshBrakeDetector, HarshCornerDetector};
pub use hill::HillDetector;
pub use reverse::ReverseDetector;
pub use types::{
    HarshAccelConfig, HarshBrakeConfig, HarshCornerConfig, HarshCornerEvent, HarshCornerSample,
    HarshLongitudinalEvent, HarshLongitudinalSample, HillConfig, HillEvent, HillKind, HillSample,
    ReverseConfig, ReverseEvent, ReverseSample, SpeedBumpConfig, SpeedBumpDiagnostic,
    SpeedBumpEvent, SpeedBumpSample,
};

#[cfg(test)]
mod tests;
