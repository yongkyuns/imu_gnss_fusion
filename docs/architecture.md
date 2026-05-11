# Repository Architecture

This repository is a Rust workspace for IMU/GNSS filter development. It is
organized around a small embedded-oriented filter crate and a larger simulator,
evaluation, and visualization crate. Data conversion, hosted datasets, browser
hosting, docs, and the experimental iOS app sit around those two crates.

The main ownership boundary is:

- `sensor_fusion/` owns the reusable filter runtime and generated EKF math.
- `sim/` owns replay, synthetic data, diagnostics, summaries, and visualization.
- `web/` owns the static browser host and hosted generic replay assets.
- `scripts/` owns repository automation for dataset packaging, hosted-data
  validation, web validation, and browser FPS checks.
- `docs/` owns math references, testing notes, architecture notes, and assets.
- `mobile/ios/` owns an experimental Swift sensor-display scaffold. It is not
  currently wired into the Rust filter runtime.

## Top-Level Layout

| Path | Role |
| --- | --- |
| `Cargo.toml` | Workspace root with `sensor_fusion` and `sim` members. |
| `sensor_fusion/` | `#![no_std]` library crate exposing the public filter API and standalone filter modules. |
| `sim/` | Standard-library tooling crate for replay, synthetic generation, diagnostics, native egui, and wasm visualizer builds. |
| `web/` | Static host for the wasm visualizer plus compressed generic replay datasets and dataset manifests. |
| `scripts/` | Python and Node.js tooling for packaging datasets, validating hosted data, validating GitHub Pages output, and benchmarking web FPS. |
| `.github/` | CI workflows and hosted generic dataset manifest/schema. |
| `docs/` | Documentation, math PDFs/TEX sources, frame notes, test notes, and architecture assets. |
| `mobile/ios/` | XcodeGen-based iOS scaffold using CoreLocation, CoreMotion, SwiftUI, and Charts. |
| `target/` | Build outputs and local replay/diagnostic artifacts. This is generated output, not source. |

The workspace dependency direction is one-way: `sim` depends on
`sensor_fusion`; `sensor_fusion` does not depend on `sim`, file I/O, egui,
serde JSON, web APIs, or platform UI code.

## Core Crate: `sensor_fusion`

`sensor_fusion` is the reusable filter crate. Its `Cargo.toml` keeps default
features empty and only exposes optional `serde` support for public
configuration/data types that need to cross the web/native UI boundary. The
crate is `#![no_std]` and uses `libm` plus fixed-size arrays instead of heap
allocation or a linear algebra runtime.

### Public Surface

`sensor_fusion/src/lib.rs` exports:

- `SensorFusion`: high-level streaming facade.
- Public facade types from `fusion_types`: `Config`, `Filter`, `MountMode`,
  `ImuSample`, `GnssSample`, `VehicleSpeedSample`,
  `VehicleSpeedDirection`, `Update`, and `AlignDebug`.
- `ProcessNoise`.
- Standalone modules `align`, `reduced`, and `full`.
- Hidden generated wrappers `generated_reduced` and `generated_full` used by
  tests and parity diagnostics.

Callers normally use `SensorFusion` instead of constructing `reduced::Filter`
or `full::Filter` directly. The lower-level modules remain public so tests,
diagnostic binaries, and formulation work can inspect or compare one filter at
a time.

### Module Ownership

| Module | Ownership |
| --- | --- |
| `fusion.rs` | Public runtime facade. Owns mount mode selection, alignment bootstrap, WGS84 local anchoring, Reduced/Full initialization, sample dispatch, NHC scheduling, GNSS update staging, reanchoring, vehicle-speed updates, and public accessors. |
| `fusion_types.rs` | Public facade data types plus internal runtime configuration structs. |
| `align/` | Standalone vehicle-to-body mount estimator. It owns mount covariance, stationary gravity bootstrap, GNSS-derived motion windows, turn consistency, and alignment diagnostics. |
| `reduced/` | Local-NED reduced-state EKF runtime, public state layouts, generated-model wrappers, update diagnostics, and state operation helpers. |
| `full/` | ECEF full-state EKF runtime, public state layouts, generated-model wrappers, f64 shadow state for numerically sensitive position/mount propagation, and Full-specific diagnostics. |
| `math.rs` | Small quaternion, vector, matrix, and scalar math helpers. |
| `nav.rs` | WGS84, ECEF, local NED, Earth-rate, gravity, and geodetic conversion helpers. |
| `covariance.rs` | Shared sparse covariance propagation and symmetrization helpers used by generated-model filters. |
| `noise.rs` | Shared `ProcessNoise` profiles and mount random-walk configuration. |
| `coordinate_conventions.rs` | Test-only convention checks for frames, quaternions, mounts, and Reduced/Full attitude equivalence. |

### Public Runtime Flow

The public API accepts raw body-frame IMU samples, WGS84 GNSS samples, and
optional vehicle-speed observations:

```text
caller
  -> SensorFusion::{new, with_config, with_mount}
  -> process_imu(ImuSample)
  -> process_gnss(GnssSample)
  -> process_vehicle_speed(VehicleSpeedSample)
  -> reduced(), full(), mount_q_bv(), position_lla(), align_debug()
```

`SensorFusion` stores both a Reduced and a Full instance internally, but the
configured `Filter` determines which runtime path is advanced for the selected
EKF:

- `Filter::Reduced` advances the local-NED Reduced EKF after mount and GNSS
  initialization are ready.
- `Filter::Full` advances the ECEF Full EKF after mount and GNSS
  initialization are ready.

The facade also owns an `Align` instance for automatic mount estimation. In
`MountMode::Auto`, it bootstraps roll/pitch from stationary IMU samples, builds
GNSS interval summaries, updates `Align`, and releases the resulting physical
mount once handoff gates pass. In `MountMode::Manual`, `set_misalignment` stores
the caller-supplied physical mount and freezes mount states in both EKFs.

GNSS handling does three jobs:

1. Convert WGS84 latitude/longitude/height into the local anchor used by
   Reduced, while also preserving the original GNSS sample for Full.
2. Feed alignment motion windows when automatic mount alignment is enabled.
3. Initialize or update the configured EKF. Reduced GNSS updates may be staged
   and fused at an IMU epoch so GNSS and NHC rows can be batched.

IMU handling does three jobs:

1. Maintain the current sample pair and IMU interval summaries for alignment.
2. Predict the selected EKF when the filter is initialized and the `dt` is in
   the accepted runtime range.
3. Apply eligible NHC, zero-velocity, stationary-gravity, and pending GNSS
   updates.

Vehicle-speed handling is Reduced-facing today. It fuses forward/reverse speed
or zero velocity through the Reduced body-frame speed path after the Reduced
filter is initialized.

### Frame And State Boundary

Public samples use the project frame conventions documented in
`docs/math/frames.md`:

- `b`: raw IMU body/sensor frame.
- `v`: vehicle frame, forward-right-down.
- `n`: local NED frame used by Reduced.
- `e`: ECEF frame used by Full.

Rotations are active: `C_ab` maps coordinates from frame `b` to frame `a`, and
`R(q_ab) = C_ab`.

The public mount quaternion is the physical vehicle-to-body mount `q_bv`:

```text
x_b = C_bv x_v
C_vb = C_bv^T
x_v = C_vb x_b
```

Raw IMU samples are not pre-rotated by callers. Reduced and Full rotate raw
body-frame increments through `C_vb` internally during propagation. Reduced
attitude `q0..q3` is `q_nv`; Full attitude `q0..q3` is `q_ev`. In both filters,
the `q_bv0..q_bv3` fields store the physical `q_bv` mount.

### Reduced Filter

`sensor_fusion/src/reduced/mod.rs` owns the Reduced EKF runtime. It wraps
generated nominal prediction, transition, and observation snippets in a Rust
`Filter` with explicit covariance and mount-freeze policy.

Important Reduced files:

- `reduced/types.rs`: public `NominalState`, `State`, `ImuDelta`,
  `GnssSample`, `StationaryDiag`, and `UpdateDiag` layouts.
- `reduced/generated.rs`: stable Rust function boundary around generated
  snippets.
- `reduced/generated/*.rs`: generated formulas for nominal prediction,
  transition/noise matrices, GPS scalar observations, body velocity
  observations, stationary acceleration observations, reset Jacobian, and
  sparse matrix support.
- `reduced/state_ops.rs`: focused state conversion/injection helpers used by
  tests and diagnostics.
- `reduced/formulation.py`: SymPy source for generated Reduced equations.

Reduced state is local-NED and compact:

```text
nominal: q_nv, v_n, p_n, gyro bias, accel bias, q_bv
error:   dtheta_v, dv_n, dp_n, dbg, dba, dpsi_vb
```

The runtime fuses GNSS position/velocity, optional GNSS+NHC batches, body speed,
lateral/vertical NHC, zero velocity, and stationary gravity. Update diagnostic
fields are intentionally part of the public state layout so visualizer and
diagnostic tools can attribute corrections by update type.

### Full Filter

`sensor_fusion/src/full/mod.rs` owns the Full ECEF EKF runtime. It is maintained
beside Reduced for formulation comparison and diagnostics.

Important Full files:

- `full/types.rs`: public `InitConfig`, `NominalState`, `State`, `ImuDelta`,
  `GnssPositionGateDiag`, and state-size constants.
- `full/generated.rs`: stable Rust function boundary around generated snippets.
- `full/generated/*.rs`: generated reference transition/noise, NHC rows,
  sparse matrix support, and reset Jacobian.
- `full/formulation.py`: SymPy source for generated Full equations.

Full state is ECEF and larger:

```text
nominal: q_ev, v_e, p_e, accel bias, gyro bias, accel scale, gyro scale, q_bv
error:   dp_e, dv_e, dtheta_v, dba, dbg, dsa, dsg, dpsi_bv
```

Full keeps public f32 state fields plus f64 shadow ECEF position and mount
quaternion fields. GNSS rows observe ECEF position and velocity; NHC rows
predict vehicle-frame velocity with `C_ev^T v_e` and constrain lateral/vertical
components. Full also stores last-update residuals, innovation variances,
per-observation corrections, and GNSS position-gate diagnostics for the
visualizer and diagnostic binaries.

### Generated-Code Layout

The generated Rust snippets are checked in so normal Rust builds do not require
Python, NumPy, or SymPy. The ownership rule is:

- Edit `sensor_fusion/src/reduced/formulation.py` or
  `sensor_fusion/src/full/formulation.py`.
- Use `sensor_fusion/code_gen.py` through the formulation scripts to regenerate
  Rust snippets.
- Review generated diffs, then run targeted Rust tests.
- Do not hand-edit files under `sensor_fusion/src/*/generated/`.

Generated wrappers keep formulas behind named Rust functions. Runtime modules
depend on wrapper functions, not on the include files directly. Sparse support
arrays generated beside the formulas feed `covariance::predict_sparse`.

## Tooling Crate: `sim`

`sim` is the standard-library layer around the embedded core. It depends on
`sensor_fusion` with the `serde` feature enabled, then adds CSV parsing, JSON,
clap, rayon, egui/eframe, walkers maps, and wasm-only browser dependencies.

`sim/src/lib.rs` exposes four public module families:

```text
sim::datasets
sim::eval
sim::synthetic
sim::visualizer
```

### Dataset Modules

| Module | Ownership |
| --- | --- |
| `datasets/generic_replay.rs` | Hardware-agnostic CSV schema for `imu.csv`, `gnss.csv`, and optional reference CSVs. Converts generic rows into public `sensor_fusion` samples. |
| `datasets/synthetic_replay.rs` | Loader for older/generated synthetic-export files such as `time.csv`, `ref_gyro.csv`, `gps-*.csv`, and truth samples. |
| `datasets/seeded_full.rs` | Semicolon-delimited fixture loader for seeded Full EKF diagnostics and tests. |

The generic replay schema is the repository boundary for field data. External
hardware-specific converters should emit:

- `imu.csv`
- `gnss.csv`
- optional `reference_position.csv`
- optional `reference_attitude.csv`
- optional `reference_mount.csv`
- optional `reference_motion.csv`

Reference files are for plots, summaries, manual mount seeding, and tests. They
are not normal filter inputs.

### Evaluation Modules

| Module | Ownership |
| --- | --- |
| `eval/replay.rs` | Canonical timestamp merge order. IMU events are consumed before GNSS when timestamps tie. |
| `eval/gnss_ins.rs` | Quaternion, rotation, signal-source, and GNSS/INS helper math used by replay and synthetic tooling. |
| `eval/trace.rs` | Trace lookup, schema checks, finite-point checks, and nearest-sample helpers. |
| `eval/state_summary.rs` | Final, tail, convergence, and fluctuation summaries for trace pairs. |
| `eval/first_divergence.rs` | Public-API Reduced/Full divergence analysis helpers. |
| `eval/config.rs` | Snapshot defaults for replay comparison configuration. |

Any new replay or diagnostic tool should use `eval::replay::for_each_event`
instead of inventing its own IMU/GNSS merge ordering.

### Synthetic Modules

| Module | Ownership |
| --- | --- |
| `synthetic/motion_dsl.rs` | Parser for high-level scenario files under `sim/motion_profiles/`. |
| `synthetic/gnss_ins_path.rs` | Motion profile integration, Earth model helpers, truth trajectory generation, IMU/GNSS noise models, and noisy measurement generation. |

Synthetic scenarios are converted into the same `GenericReplayInput` shape as
field CSVs before they are passed to visualization or filter replay code.

### Visualizer Modules

The visualizer is shared by native and browser builds. The binary entry point is
`sim/src/bin/visualizer.rs`; reusable code lives under `sim/src/visualizer/`.

| Module | Ownership |
| --- | --- |
| `visualizer/model.rs` | Serializable plot, map, heading, cursor, page, mount-mode, and update-inspector data structs. |
| `visualizer/pipeline/config.rs` | Serializable replay tuning config and the adapter that applies it to `SensorFusion`. |
| `visualizer/pipeline/generic.rs` | Main generic replay pipeline. Parses CSV text, feeds `SensorFusion`, records Reduced traces, adds Align and Full traces, overlays references, and populates diagnostics. |
| `visualizer/pipeline/synthetic.rs` | Generates synthetic measurements, applies synthetic mount, converts to generic replay input, then reuses the generic pipeline. |
| `visualizer/pipeline/reference.rs` | Reference mount/attitude conversions, mount seed extraction, and nearest reference RPY lookup. |
| `visualizer/replay_job.rs` | Native background jobs, web replay-job request/result structs, progress reporting, and transport decimation. |
| `visualizer/ui.rs` | Shared app state plus native/web app launch functions. |
| `visualizer/ui/runtime.rs` | App construction, startup state, theme/density setup, replay refresh, and egui frame lifecycle. |
| `visualizer/ui/web.rs` | Browser-only manifest loading, gzip/CSV loading, dropped-file handling, worker orchestration, query parameters, and web synthetic inputs. |
| `visualizer/ui/{controls,pages,plots,maps,inspector,tuning,state,trace_query,colors,orthogonal,windows}.rs` | Modular egui controls, layouts, plot rendering, map overlays, inspector windows, tuning panels, state classification, query helpers, colors, and popups. |
| `visualizer/{math,stats,theme}.rs` | Visualization-specific coordinate helpers, trace/map statistics, and egui theme setup. |

The generic visualizer pipeline intentionally uses the public `SensorFusion`
API. Its core flow is:

```text
GenericReplayInput
  -> GenericReplayRunContext
  -> SensorFusion::new() or SensorFusion::with_config(...)
  -> apply_filter_compare_config(...)
  -> optional reference mount seed through set_misalignment(...)
  -> eval::replay::for_each_event(...)
       IMU  -> process_imu(...)
       GNSS -> process_gnss(...)
  -> append Reduced, Align, Full, reference, map, and inspector traces
  -> PlotData
  -> egui UI or JSON/web transport
```

Reduced traces are produced while the main `SensorFusion::new()` instance runs
with the default Reduced runtime. Full traces are populated by a separate
`SensorFusion::with_config(Config { filter: Filter::Full, ... })` replay inside
the auxiliary trace path. This gives the visualizer side-by-side Reduced and
Full output while keeping each filter behind the same public facade.

### Binaries

`sim/src/bin/` contains operational tools:

| Binary | Role |
| --- | --- |
| `visualizer` | Native egui visualizer and wasm visualizer entry point. |
| `export_synthetic_replay_generic` | Converts synthetic-export output into generic replay CSVs. |
| `compare_filter_runtime` | Runtime benchmark harness. |
| `filter_equivalence_harness` | Reduced/Full equivalence diagnostic replay harness. |
| `filter_equivalence_summary` | Summarizes equivalence harness CSV output. |
| `first_divergence` | Finds first Reduced/Full divergence against replay/reference data. |
| `covariance_history` | Detailed covariance, correction allocation, and update-history diagnostics. |
| `mount_tuning_sweep` | Replay sweep over mount/tuning variants. |
| `diag_update_allocation` | Focused Reduced update-allocation diagnostics. |
| `diag_prediction_residual` | Prediction residual diagnostics. |
| `diag_full_accel_z_bias` | Full vertical accelerometer-bias diagnostic. |
| `run_full_nsr` | Seeded Full EKF replay from prepared NSR-style fixtures. |
| `seeded_full_stress_mc` | Monte Carlo stress replay for seeded Full fixtures. |
| `synthetic_bad_basin_sweep` | Synthetic early-convergence stress sweep. |

## Browser/WASM Architecture

The browser app is the same `visualizer` binary compiled for
`wasm32-unknown-unknown` and hosted by static files under `web/`.

Browser startup flow:

```text
web/index.html
  -> import ./pkg/visualizer.js
  -> load ./pkg/visualizer_bg.wasm
  -> wasm start_visualizer("visualizer_canvas")
  -> sim::visualizer::ui::run_visualizer_web(...)
  -> egui canvas app
```

Replay work that may block the UI is delegated to `web/replay_worker.js`:

```text
egui web UI
  -> WebReplayWorkerJob
  -> replay_worker.js
  -> same wasm module in a module Worker
  -> build_*_json_with_progress(...)
  -> JSON PlotData/result back to UI
```

Hosted datasets live under `web/datasets/`. The browser loads
`web/datasets/manifest.json`, fetches compressed CSVs such as `imu.csv.gz` and
`gnss.csv.gz`, decompresses them in Rust/WASM, and submits a replay worker job.
Each dataset directory also has its own `manifest.json` produced by
`scripts/package_dataset.py`.

The web layer owns browser integration only. Replay semantics remain in
`sim::visualizer::pipeline` and filter behavior remains in `sensor_fusion`.

## Data And Asset Ownership

### Generic Replay Data

Generic replay CSVs are the main data contract:

```text
imu.csv
gnss.csv
reference_position.csv      optional
reference_attitude.csv      optional
reference_mount.csv         optional
reference_motion.csv        optional
```

`sim::datasets::generic_replay` owns uncompressed local file parsing/writing and
conversion to `sensor_fusion` samples. `sim::visualizer::pipeline::generic`
owns parsing CSV text in native/web replay jobs. `scripts/package_dataset.py`
owns compressed static-host packaging and per-dataset manifests.

### Hosted Data

Hosted dataset metadata is split across two layers:

- `.github/datasets/generic-datasets.json` is the CI/download/checksum manifest
  for hosted generic datasets.
- `web/datasets/manifest.json` is the browser-visible dataset list.
- `web/datasets/<dataset-id>/manifest.json` describes one packaged static
  dataset and its compressed files.

`scripts/validate_generic_datasets.mjs` validates the CI manifest, downloads or
copies files into a cache, validates generic CSV schemas, and optionally runs
smoke-profile visualizer replays. `scripts/validate_pages_static.mjs` validates
the static site, wasm artifact paths, MIME behavior, and browser dataset
manifest references.

### Documentation Assets

Math references are in `docs/align.tex`, `docs/reduced.tex`, and
`docs/full.tex`, with generated PDFs checked in. Architecture images are under
`docs/assets/`, with `arch.pen` as the editable Pencil source for the existing
diagram asset.

## Test Architecture

Tests are split by ownership:

- `sensor_fusion/tests/` protects public API behavior, alignment API contracts,
  Reduced state operations, NHC Jacobian/equivalence details, Full mount
  observability, and generated-model parity.
- `sim/tests/` protects synthetic GNSS/INS generation, visualizer replay
  equivalence, evaluation scaffolding, and seeded Full fixture parity.
- `scripts/tests/` protects dataset packaging behavior.

Generated-code changes should be verified with `cargo test -p sensor_fusion`
and targeted simulator tests such as `sim --test full_parity` and
`sim --test synthetic_gnss_ins`. Browser/static changes should be checked with
the Node validators described in `docs/testing.md` and `web/README.md`.

## Mobile Scaffold

`mobile/ios/IMUGNSSPhone` is a separate Swift app scaffold generated by
XcodeGen. It owns live sensor display and charting:

- `SensorStore.swift` reads CoreLocation, CoreMotion, and barometer data,
  derives rough local NED position/velocity, and stores rolling chart samples.
- `ContentView.swift` renders status rows and Swift Charts.
- `IMUGNSSPhone-Bridging-Header.h` is currently empty.
- `SensorStore` has `runEkfPredict`, `runEkfFuseGps`, and
  `appendEkfSamplesFromState` hooks, but they are stubs.

There is no current Rust FFI bridge from the iOS app to `sensor_fusion`.

## Dependency Boundaries

The intended dependency graph is:

```text
sensor_fusion
  <- sim
       <- sim binaries
       <- web wasm build of visualizer
       <- scripts/package_dataset.py for synthetic-export conversion through a sim binary

docs, web static assets, .github manifests, and mobile/ios sit beside the Rust
workspace and do not feed back into sensor_fusion.
```

Boundary rules:

- `sensor_fusion` should remain allocation-free, `no_std`, and free of file,
  UI, web, and dataset concerns.
- `sim::datasets` owns source data parsing, not filter math.
- `sim::eval::replay` owns event ordering, not filter update behavior.
- `sim::visualizer::pipeline` owns trace construction and comparison overlays,
  not direct EKF internals except where public diagnostic state is exposed.
- `web/` owns static hosting and JavaScript worker glue, not replay semantics.
- `scripts/` own packaging/validation automation, not runtime filter behavior.
- `mobile/ios/` owns native phone sensor UI; until an FFI bridge exists, it is
  not an owner of the Rust filter runtime.

## End-To-End Control And Data Flow

### Generic Field Replay To Visualizer

```text
generic replay directory or hosted web dataset
  -> imu.csv / gnss.csv / optional references
  -> sim::datasets::generic_replay or visualizer CSV-text parsers
  -> GenericReplayInput
  -> build_generic_replay_plot_data(...)
  -> SensorFusion facade
       -> Align for automatic mount, when enabled
       -> Reduced EKF for primary replay traces
       -> Full EKF through auxiliary replay for comparison traces
  -> PlotData
  -> native egui UI or web JSON transport
  -> plots, map, mount views, calibration views, diagnostics, update inspector
```

Reference files are consumed by the visualizer pipeline for overlays, summary
comparison, cursor metadata, vehicle-frame reference motion traces, and optional
manual mount seeding. They are not silently injected into automatic filter
updates.

### Synthetic Scenario To Visualizer

```text
sim/motion_profiles/*.scenario or inline web scenario
  -> synthetic::motion_dsl
  -> synthetic::gnss_ins_path::generate_with_noise
  -> synthetic pipeline applies truth mount to vehicle-frame IMU
  -> GenericReplayInput with reference attitude/mount/motion
  -> same generic replay pipeline as field data
  -> PlotData and UI
```

This shared path is deliberate: field and synthetic inputs should exercise the
same public `SensorFusion` replay semantics after data has been normalized to
generic samples.

### Public API Runtime

```text
application code
  -> SensorFusion::with_config(Config { filter, mount_mode })
  -> optional tuning setters
  -> process_imu / process_gnss / process_vehicle_speed
  -> Update status
  -> reduced() or full() state accessors
```

Applications own timestamp ordering, input sensor availability, and persistence.
`SensorFusion` owns filter initialization, alignment handoff, anchoring,
prediction/update scheduling, and state access.

## Extension Points

Add new filter math inside `sensor_fusion` only when it is part of the reusable
runtime. Keep new generated formulas beside their formulation script and expose
them through a stable wrapper module.

Add new replay formats by converting them to the generic replay schema at the
edge. Do not add hardware-specific parsing to `sensor_fusion`.

Add new visualizer plots by extending `PlotData`, populating traces in
`visualizer/pipeline`, and rendering them in `visualizer/ui` modules. Keep the
pipeline independent from egui widgets when possible so native and web builds
continue to share replay output.

Add hosted datasets through the generic dataset packaging and validation path.
Keep raw device logs outside this repository unless they are converted to the
generic CSV contract first.
