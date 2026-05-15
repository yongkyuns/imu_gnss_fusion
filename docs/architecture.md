# Architecture

IMU/GNSS Fusion is a Rust workspace for embedded-oriented EKF development,
hardware-agnostic replay, synthetic scenario generation, diagnostics, and a
native/browser visualizer.

## Crates

| Path | Role |
| --- | --- |
| `sensor_fusion/` | `#![no_std]` library crate exposing the public EKF facade, alignment, generated model wrappers, and state helpers. |
| `sim/` | Replay/evaluation crate with synthetic data generation, diagnostics, and the egui visualizer. |
| `web/` | Static host for the wasm visualizer and hosted datasets. |
| `mobile/ios/` | Experimental iOS sensor collection app and FFI shell. |

`sensor_fusion` is the reusable runtime. `sim` owns dataset parsing, replay
ordering, visualizer trace construction, and diagnostic tools.

## Runtime Flow

```text
IMU/GNSS samples
  -> SensorFusion
  -> Align bootstrap when automatic mount mode is enabled
  -> EKF initialization after mount and GNSS readiness
  -> EKF prediction/update loop
  -> public Update status and state accessors
```

The facade owns:

- mount mode selection,
- alignment bootstrap,
- WGS84/local anchoring,
- sample dispatch,
- NHC scheduling,
- GNSS update staging,
- reanchoring,
- vehicle-speed updates,
- public accessors and diagnostics.

## Frames

- `b`: raw IMU body/sensor frame.
- `v`: physical vehicle frame, forward-right-down.
- `n`: local NED frame.
- `e`: ECEF frame for WGS84 conversion and global position math.

Raw IMU samples are not pre-rotated by callers. The runtime rotates body-frame
increments through the physical mount internally.

## Generated Model Code

Generated EKF snippets are checked in so normal Rust builds do not require
Python or SymPy. The symbolic sources are kept beside the implementation
modules and emit Rust wrappers used by the runtime.

Edit symbolic model files, regenerate, review the generated diff, and run the
focused tests from [testing.md](testing.md).

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

`sim::datasets` parses the hardware-agnostic CSVs. `sim::eval::replay` merges
IMU and GNSS events by timestamp. The visualizer pipeline feeds only the public
`SensorFusion` API and uses optional references for plots, maps, summaries, and
manual mount seeding.

Device-specific log conversion is outside the repository boundary.

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

The UI owns presentation state only. Runtime estimator behavior remains in
`sensor_fusion`; replay and trace construction remain in `sim`.

## Diagnostics

Diagnostic binaries are intentionally thin wrappers around reusable replay,
state-summary, trace, and covariance helpers. Prefer adding reusable analysis
code under `sim::eval` or the visualizer pipeline before adding logic directly
to a binary.

Common diagnostic tools:

| Tool | Purpose |
| --- | --- |
| `visualizer` | Interactive replay inspection. |
| `diag_mount_observability` | Synthetic roll/pitch mount observability diagnostic. |
| `synthetic_bad_basin_sweep` | Synthetic early-convergence stress sweep. |
| `export_synthetic_replay_generic` | Synthetic replay export to the generic CSV schema. |

## Ownership Rules

- `sensor_fusion` owns estimator behavior and public runtime contracts.
- `sim::datasets` owns source data parsing, not estimator math.
- `sim::eval::replay` owns event ordering, not update behavior.
- `sim::visualizer::pipeline` owns trace construction and reference overlays.
- `web/` owns static hosting and browser loading behavior.
- `scripts/` own packaging and validation automation.
- `mobile/ios/` owns mobile collection/FFI integration, not estimator behavior.
