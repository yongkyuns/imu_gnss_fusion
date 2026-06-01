# Architecture

IMU/GNSS Fusion is a Rust workspace centered on the portable `sensor_fusion`
navigation crate. Host programs such as the visualizer, iOS app, firmware, and
batch tools link that crate and wrap it with platform-specific input and output
code. The `road_events` crate is an optional downstream demonstration consumer
fed by vehicle-motion samples derived from fusion records.

```{figure} _static/diagrams/overall-runtime-architecture.svg
:alt: Architecture diagram for replay, simulation, sensor fusion, and visualization.
:class: framed

Offline and realtime sources are normalized by lightweight host adapters into
the `sensor_fusion` typed-sample boundary. Fusion records are the primary
host-facing output. Optional road-event detectors can consume derived
vehicle-motion records, but they are not part of the navigation filter.
```

## Crates

| Path | Role |
| --- | --- |
| `sensor_fusion/` | `#![no_std]` library crate exposing the public EKF facade, alignment, generated model wrappers, and state helpers. |
| `road_events/` | `#![no_std]` streaming road-event detectors and trip statistics shared by simulator/web/iOS integration. |
| `tools/` | Replay/evaluation crate with synthetic data generation, diagnostics, and the egui visualizer. |
| `web/` | Static host for the wasm visualizer and hosted datasets. |
| `mobile/ios/` | iOS sensor collection, replay/export tooling, SwiftUI app, and Rust FFI integration. |

`sensor_fusion` is the reusable navigation runtime. `road_events` is a reusable
detector crate used by the visualizer and app as a downstream demonstration of
fusion-derived vehicle-motion outputs. `fusion_tools` owns dataset parsing,
replay ordering, visualizer trace construction, and diagnostic tools. `web/` is
not a Rust crate; it is the static browser host for the wasm build produced by
`fusion_tools`.

## Host Boundary

Data enters the system in two modes:

- Offline sources: hosted `web/datasets`, drag-and-drop generic CSV packages,
  synthetic scenarios, and iOS `.motionfusion` exports.
- Realtime streams: timestamped IMU samples, GNSS fixes, and optional vehicle
  speed observations from a platform host.

`sensor_fusion` does not depend on file formats, browser APIs, iOS APIs,
sockets, threads, clocks, or embedded HALs. Host wrappers convert
source-specific data into typed fusion samples:

| Component | Input contract | Host-wrapper responsibility |
| --- | --- | --- |
| `sensor_fusion` | `ImuSample`, `GnssSample`, optional `VehicleSpeed` | preserve sample order, units, accuracy fields, raw body-frame IMU data, and GNSS position/velocity semantics. |
| Optional `road_events` branch | speed, vehicle pitch, vehicle-frame acceleration, height, and detector-specific motion samples | construct vehicle-motion samples from fusion records and platform measurements, then feed the streaming detectors. |

`sensor_fusion` returns typed records rather than UI-specific data. The host
then decides whether those records become plots, alerts, telemetry, logs, test
reports, exported traces, or inputs to optional downstream consumers:

| Component | Output contract | Host use |
| --- | --- | --- |
| `sensor_fusion` | update lifecycle state, health, pose, velocity, attitude, mount, covariance, diagnostics | traces and maps, Drive/Review state, firmware telemetry, batch summaries. |
| Optional `road_events` branch | bump/shock/roughness events, harsh accel/brake/cornering events, hills, reverse intervals, trip statistics | demonstrative event tabs, mobile alerts, device logs, trip summaries. |

The firmware case follows the same pattern as the other hosts: firmware contains
the linked runtime crates and wraps them with device drivers, clocks, scheduling,
storage, and telemetry. It is not a downstream consumer of a separate runtime
service.

## Sensor Fusion Flow

Inside `sensor_fusion`, raw IMU and GNSS updates pass through the public facade:

```text
IMU/GNSS samples
  -> SensorFusion
  -> Align tilt initialization when automatic mount mode is enabled
  -> EKF initialization after mount and GNSS readiness
  -> EKF prediction/update loop
  -> public Update lifecycle state, health, and state accessors
```

The facade owns:

- mount mode selection,
- alignment tilt initialization,
- WGS84/local anchoring,
- sample dispatch,
- NHC scheduling,
- GNSS update staging,
- reanchoring,
- vehicle-speed updates,
- public accessors and diagnostics.

The facade also owns runtime continuity policy. Keeping the same `SensorFusion`
object preserves navigation state, covariance, mount, biases, and diagnostics
across normal stream pauses. The next IMU timestamp classifies the gap as short,
medium, or long sleep; long or unsafe gaps wait for GNSS reseed while retaining
calibration priors. See [](runtime-state-and-persistence.md) for the public
contract.

## Frames

- `b`: raw IMU body/sensor frame.
- `v`: physical vehicle frame, forward-right-down.
- `n`: local NED frame.
- `e`: ECEF frame for WGS84 conversion and global position calculations.

Raw IMU samples are not pre-rotated by callers. The runtime rotates body-frame
increments through the physical mount internally.

## Replay Data Flow

Generic replay data enters as:

```text
imu.csv
gnss.csv
reference_position.csv   # optional
reference_attitude.csv   # optional
reference_mount.csv      # optional
reference_motion.csv     # optional
```

`fusion_tools::datasets` parses the hardware-agnostic CSVs. `fusion_tools::eval::replay` merges
IMU and GNSS events by timestamp. The visualizer pipeline feeds the public
`SensorFusion` API, optionally constructs vehicle-motion samples consumed by
`road_events`, and uses optional references for plots, maps, summaries, and
manual mount seeding.

Generic dataset packaging is hardware-agnostic. The repository also includes an
iOS `.motionfusion` exporter and packaging wrapper for the bundled iOS app.

## Visualizer Flow

Native visualizer:

```text
CLI args
  -> load generic replay or synthesize scenario
  -> build replay job
  -> run SensorFusion
  -> build traces, maps, summaries, inspector data
  -> egui UI
```

Browser visualizer:

```text
static web shell
  -> load hosted manifest or dropped CSV files
  -> worker replay job
  -> compressed response payload
  -> egui UI
```

The UI owns presentation state only. Runtime estimator behavior remains in the
linked `sensor_fusion` crate; optional road-event detector behavior remains in
the linked `road_events` crate; replay, trace construction, hosted-dataset
loading, and map/event overlay construction remain in `fusion_tools` and `web/`.

## Mobile Flow

The iOS app streams Core Motion and Core Location data through the Rust FFI
wrapper. Swift owns platform concerns such as permissions, background execution,
recording, playback, MapKit presentation, settings, and alerts. The Rust FFI
wrapper owns the `SensorFusion` object and optional `road_events` detectors,
exposes typed C ABI records, and keeps estimator logic on the Rust side.

Stopping and starting app streaming should not require Swift to reinterpret EKF
continuity. The runtime health state and persistence policy remain inside
`sensor_fusion`; Swift consumes the resulting state and diagnostics.

## Diagnostics

Common diagnostic tools:

| Tool | Purpose |
| --- | --- |
| `visualizer` | Interactive replay inspection. |
| `diag_mount_observability` | Synthetic roll/pitch mount observability diagnostic. |
| `synthetic_bad_basin_sweep` | Synthetic early-convergence stress sweep. |
| `export_synthetic_replay_generic` | Synthetic replay export to the generic CSV schema. |

## Ownership Rules

- `sensor_fusion` owns estimator behavior and public runtime contracts.
- `road_events` owns event-detector behavior and trip-summary contracts.
- `fusion_tools::datasets` owns source data parsing, not estimator formulation.
- `fusion_tools::eval::replay` owns event ordering, not update behavior.
- `fusion_tools::visualizer::pipeline` owns trace construction, road-event sample construction, and reference overlays.
- `web/` owns static hosting and browser loading behavior.
- `scripts/` own packaging and validation automation.
- `mobile/ios/` owns mobile collection, permissions, UI, recording/playback, and FFI integration, not estimator or detector behavior.

Developer-only workflows such as generated EKF code regeneration, Pages
artifact assembly, and hosted dataset validation live under the Developer
Reference section.
