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
//! - [`HarshCornerDetector`] detects jerk-gated lateral side-load from
//!   vehicle-frame specific force.
//! - [`RoadRoughnessAnalyzer`] estimates distance-normalized vertical vibration
//!   energy over a short effective road interval.
//! - [`TripStats`] accumulates constant-memory trip distance, speed, grade,
//!   rolling-motion, and event-count summaries.

#[cfg(test)]
extern crate std;

mod bump;
mod common;
mod harsh;
mod hill;
mod reverse;
mod roughness;
mod trip;
mod types;

pub use bump::SpeedBumpDetector;
pub use harsh::{HarshAccelDetector, HarshBrakeDetector, HarshCornerDetector};
pub use hill::HillDetector;
pub use reverse::ReverseDetector;
pub use roughness::RoadRoughnessAnalyzer;
pub use trip::TripStats;
pub use types::{
    HarshAccelConfig, HarshBehaviorConfig, HarshBehaviorPreset, HarshBrakeConfig,
    HarshCornerConfig, HarshCornerEvent, HarshCornerSample, HarshLongitudinalEvent,
    HarshLongitudinalSample, HillConfig, HillEvent, HillKind, HillSample, ReverseConfig,
    ReverseEvent, ReverseSample, RoadRoughnessConfig, RoadRoughnessEstimate, RoadRoughnessLevel,
    RoadRoughnessSample, SpeedBumpConfig, SpeedBumpDiagnostic, SpeedBumpEvent, SpeedBumpSample,
    TripConfig, TripEventCounts, TripEventKind, TripSample, TripSummary,
};

#[cfg(test)]
mod tests;
