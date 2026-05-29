# iOS App

`mobile/ios` contains the phone data-collection and replay app. It is not a
separate estimator: live samples, replay samples, fusion state, and road-event
detections all pass through the same Rust-backed path used by the rest of the
workspace.

```{figure} ../_static/diagrams/ios-app-flow.svg
:alt: iOS app flow from live sensors and recorded sessions through SensorStore, Rust FFI, Drive, Review, Settings, and Diagnostics.
:class: framed

The app has four surfaces: Drive for live use, Review for saved sessions,
Settings for operating choices, and a debug-only Diagnostics tab.
```

## What The App Is For

Use the app to collect phone IMU/GNSS data, inspect fusion and event output while
driving, replay saved sessions on the phone, and export raw recordings for the
browser visualizer. The app is intentionally thin around the estimator:

- CoreLocation supplies GNSS position, speed, course, and accuracy.
- CoreMotion supplies accelerometer, gyro, and optional barometer-like altitude
  context.
- `SensorStore` orders live or replay samples and owns app-facing state.
- `SensorFusionFFI` calls the Rust `sensor_fusion` and `road_events` crates.
- Swift views render the fused route, GNSS route, health, event annotations, and
  saved-session UI.

## Drive Surface Implementation

The Drive surface is a SwiftUI projection of `SensorStore` published state.
Route rendering is derived from store histories rather than stored as a
separate map model:

- raw GNSS route coordinates come from the GNSS route history when
  `RouteLayerSelection.showsGnssRoute` is true;
- fused route coordinates come from EKF geographic snapshots and are gated by
  `FusedMapVisibilityPolicy::shouldShowFusedOutput(initialized, mountReady)`;
- follow heading prefers EKF yaw history and falls back to CoreLocation course;
- event annotations use the recent suffix of `motionEvents`, not a separate
  detector pass in Swift.

`RouteLayerSelection` is the render contract for `none`, `fused`, `gnss`,
`both`, and `delta`. Its derived flags (`showsFusedRoute`, `showsGnssRoute`,
`showsDeltaOverlay`) are what MapKit views consume. The telemetry drawer reads
`FusionHealth`, `AlignProgressSnapshot`, EKF velocity/attitude histories,
vehicle-frame motion, `TripStatsSummary`, stream mode, and raw logging state
from `SensorStore`.

Playback mode adds a playback control panel, but it does not switch to a
different estimator. Recorded samples still enter the same replay handlers that
call the Rust FFI.

## Review Replay Pipeline

Review is the file-backed entry point to `RawSessionFileStore`. Saved sessions
are represented by `.summary.json` sidecars under `Documents/RawSessions`.
`SensorStore.replaySession` stops live sensor sources, advances the stream
generation, resets app-side runtime state and the fusion engine for the selected
source, loads the `.motionfusion` JSON in a detached task, and converts envelopes
with `RawSessionTimeline::events`.

Replay scheduling is controlled by `ReplayBatchPolicy`: playback begins at the
first event timestamp, advances virtual elapsed time by wall-clock delta times
`PlaybackSpeedPolicy`, processes up to `maximumEventsPerBatch` events per tick,
and publishes progress separately. IMU events call the EKF predict path, GNSS
events call the GNSS fuse path, and barometer events update app-side vertical
display state.

## Settings State And Policies

Settings is a binding layer over `SettingsControlModel`. `SensorStore` publishes
`SettingsControlState`; the model forwards user actions back to `SensorStore`
without owning estimator logic. The state carries authorization, stream mode,
playback speed, harsh-behavior preset, event-audio settings, mount-memory
settings, recording state, and saved-session count.

Policy types keep settings behavior testable:

- `PlaybackSpeedPolicy` clamps replay speed to the supported range and exposes
  picker labels;
- `HarshBehaviorPreset` persists detector sensitivity and is applied to the Rust
  FFI through `FusionEngine::setHarshBehaviorPreset`;
- `EventAudioSettingsDefaults` persists alert mode and silent-mode behavior;
- `MountMemoryPolicy` validates scalar-first `q_bv`, requires at least 60 s of
  initialized EKF time before saving, rate-limits periodic stores to 30 s, and
  ignores saved-mount changes below 0.25 deg.

Mount memory is a stored prior, not a forced reset path. Normal stop/start
pauses retain the existing Rust `SensorFusion` object so the library can
classify the gap as short, medium, or long sleep. Saving a mount prior should be
tied to a stable fusion state; see [](../runtime-state-and-persistence.md).

## Diagnostics Tab

Diagnostics is compiled into debug builds and hidden until enabled in Settings.
It is for implementation work, not normal driving. It exposes fusion health,
resource/profiling counters, location and IMU charts, stream timing, and
developer comparison views used to inspect estimator or UI regressions.

## Events And Alerts

Road-event decisions come from Rust through the FFI. Swift should not duplicate
the detector logic. Current event kinds include harsh acceleration, harsh
braking, harsh cornering, reverse, speed bump, uphill, downhill, road shock,
rough road, and GNSS degradation. Events can appear as map annotations, heads-up
cards, trip counters, and audio alerts depending on settings.

The harsh-behavior preset changes detector thresholds inside `road_events`.
Changing the preset affects future samples; it is not a post-processing pass over
old events.

## Fusion Continuity

The app should keep a live `SensorFusion` context across ordinary Drive
stop/start operations. Stopping the sensor stream resets app-side synchronizers,
audio notification state, and profiling counters, but it should not force a
fresh estimator. The next IMU timestamp lets the library classify the pause:

- short sleep: retain navigation with small covariance inflation;
- medium sleep: continue degraded IMU prediction until GNSS returns, unless
  confidence falls below the usable baseline;
- long sleep: keep retained calibration information but wait for GNSS reseed
  before publishing usable navigation.

Replay, source changes, explicit mount changes, and lost retained memory are
different streams and should start a fresh context.

## Recording Format

Raw sessions are saved as `.motionfusion` JSON recordings under the app's
Documents `RawSessions` area, with a small summary file beside the recording.
The log stores app/build/session metadata and timestamped envelopes for IMU,
GNSS, barometer/location context, and event-related state.

Export a recording to the generic replay CSV layout:

```bash
cd mobile/ios
python3 scripts/export_motionfusion.py ~/Downloads/session.motionfusion --output-dir /tmp/session-web
```

The exporter writes `imu.csv`, `gnss.csv`, and `summary.txt`. GNSS velocity is
exported directly when the recording has explicit NED velocity. Otherwise the
exporter derives horizontal velocity from speed and course when course accuracy
is usable, uses zero velocity for stationary samples, and skips GNSS rows that
cannot produce a usable velocity.

To publish a new recording as a hosted browser dataset, use the dataset
packaging workflow in [](../development/datasets.md).

## Code Map

- Swift app state and UI: `mobile/ios/IMUGNSSPhone/App/ContentView.swift` and
  `mobile/ios/IMUGNSSPhone/App/SensorStore.swift`.
- Rust bridge: `mobile/ios/SensorFusionFFI/src/lib.rs` and
  `mobile/ios/SensorFusionFFI/include/sensor_fusion_ffi.h`.
- iOS export: `mobile/ios/scripts/export_motionfusion.py`.
- App tests: `mobile/ios/IMUGNSSPhone/App/Analysis/Tests` and
  `mobile/ios/IMUGNSSPhone/App/UI/Tests`.

## Practical Caveats

The iOS simulator is not a realistic IMU/GNSS source. Use it for UI state and
replay behavior, not for validating live sensor quality. Background execution is
also platform-sensitive: location can continue with the right permissions, while
high-rate motion streams are constrained by iOS power and lifecycle policy.
