# `sim` Tooling Map

This crate contains the hardware-agnostic replay, evaluation, synthetic-data, and visualization tooling around the `sensor_fusion` library.

## Canonical Binaries

- `visualizer`: interactive viewer for generic CSV replay directories and synthetic scenarios.
- `export_synthetic_replay_generic`: exporter from synthetic-export outputs into the generic replay schema.
- `run_full_nsr`: seeded Full EKF replay from prepared reference-state inputs.
- `seeded_full_stress_mc`: fast Rust Monte Carlo replay for seeded Full EKF stress.
- `compare_filter_runtime`: runtime benchmark harness.
- `diag_full_accel_z_bias`: controlled Full EKF Z-accelerometer-bias diagnostic.
- `synthetic_bad_basin_sweep`: synthetic early-convergence stress sweep.

## Visualizer Inputs

Run a generic replay directory containing `imu.csv` and `gnss.csv`:

```bash
cargo run --release -p sim --bin visualizer -- \
  --generic-replay-dir /path/to/replay-dir
```

Run a synthetic motion scenario:

```bash
cargo run --release -p sim --bin visualizer -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low
```

The same generic CSV parser is used by the native visualizer and the browser visualizer.

## Visualizer Mount Modes

The visualizer uses a single `--misalignment` option:

- `internal`: Align provides the initial mount seed, then Reduced EKF estimates mount states.
- `external`: Reduced EKF continuously follows Align and freezes its mount states.
- `ref`: Reference mount angles are used when a synthetic or converted dataset provides them.

Full EKF keeps its historical behavior: `external` is treated like `internal`, so Full uses the latched Align seed rather than continuously following Align.

## Generic Replay Format

The common replay format is a directory with:

- `imu.csv`
  - columns: `t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2`
- `gnss.csv`
  - columns: `t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,pos_std_n_m,pos_std_e_m,pos_std_d_m,vel_std_n_mps,vel_std_e_mps,vel_std_d_mps,heading_rad`
  - `heading_rad` may be `NaN` when heading is unavailable or intentionally omitted
- Optional `reference_attitude.csv` and `reference_mount.csv`
  - columns: `t_s,roll_deg,pitch_deg,yaw_deg`
  - used by visualizer comparison plots only; filter inputs still come from the public generic IMU/GNSS path
- Optional `reference_position.csv`
  - columns: `t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,heading_rad`
  - used by the map as a fused reference trajectory; `gnss.csv` remains the GNSS-only trajectory and filter input

Hardware-specific converters should live outside this repository and emit this schema.

## Hosted Dataset Packaging

Use the repository-level packaging script to convert a generic replay directory into a static-hosting layout:

```bash
python3 scripts/package_dataset.py /path/to/replay-dir /tmp/hosted-drive
```

The output directory contains `manifest.json`, `imu.csv.gz`, `gnss.csv.gz`, and any optional reference CSVs found in the source replay. The manifest records the generic replay schema, sample counts, time span, compressed file sizes, and SHA-256 hashes.

The script also accepts synthetic-export output directories by invoking `export_synthetic_replay_generic` before packaging:

```bash
python3 scripts/package_dataset.py /path/to/synthetic-export/output /tmp/hosted-drive \
  --source-format synthetic-export \
  --signal-source meas
```

Raw UBX or other `.bin` logs are not packaged here. Rebuilding that conversion path would reintroduce the device-specific UBX parser stack removed from this repository; external converters should emit the generic CSV schema first.

## Shared Code Direction

Prefer these shared modules instead of duplicating replay logic in new tools:

- `src/datasets/generic_replay.rs`: hardware-agnostic IMU/GNSS replay schema.
- `src/datasets/synthetic_replay.rs`: synthetic-export CSV parsing and sample loading.
- `src/datasets/seeded_full.rs`: semicolon-delimited seeded Full EKF dataset parsing.
- `src/eval/replay.rs`: canonical IMU/GNSS merge order for feeding `SensorFusion`.
- `src/eval/state_summary.rs`: convergence, fluctuation, and final-error summaries.
- `src/visualizer/pipeline/generic.rs`: generic replay pipeline used by native and browser visualization.
- `src/visualizer/pipeline/synthetic.rs`: synthetic replay pipeline.

## Source Module Overview

The `sim` crate is deliberately split between data ingestion, evaluation,
synthetic generation, and visualization. New tools should reuse these modules
instead of adding one-off CSV parsers or filter replay loops.

| Path | Responsibility |
| --- | --- |
| `src/datasets/generic_replay.rs` | Reads/writes hardware-agnostic `imu.csv`, `gnss.csv`, and optional reference CSVs. This is the boundary format for real and hosted data. |
| `src/datasets/synthetic_replay.rs` | Loads generated synthetic truth, IMU, and GNSS sample files. |
| `src/datasets/seeded_full.rs` | Loads prepared seeded Full EKF fixtures used by diagnostic binaries and tests. |
| `src/eval/replay.rs` | Defines the canonical timestamp merge order for IMU/GNSS events. |
| `src/eval/gnss_ins.rs` | Quaternion, rotation, signal-source, and GNSS/INS conversion helpers. |
| `src/eval/state_summary.rs` | Computes final, tail, convergence, and fluctuation summaries from traces. |
| `src/eval/trace.rs` | Trace lookup, validation, and nearest-sample helpers. |
| `src/synthetic/motion_dsl.rs` | Parses high-level synthetic scenario files such as `city_blocks_15min.scenario`. |
| `src/synthetic/gnss_ins_path.rs` | Generates synthetic trajectories, truth samples, noisy IMU/GNSS measurements, and preset noise levels. |
| `src/visualizer/model.rs` | Shared plot, map, page, heading, contribution, and inspector structs. |
| `src/visualizer/pipeline/config.rs` | Serializable Align/Reduced/Full tuning configuration used by native and web replays. |
| `src/visualizer/pipeline/generic.rs` | Generic replay pipeline used by both native and web visualizers; feeds only the public `SensorFusion` API. |
| `src/visualizer/pipeline/synthetic.rs` | Synthetic pipeline that generates data, converts it into the generic replay path, and overlays truth traces. |
| `src/visualizer/replay_job.rs` | Native background replay jobs and web replay transport shaping. |
| `src/visualizer/ui/*` | Modular egui UI implementation. See the table below. |

### Visualizer UI Modules

| Path | Responsibility |
| --- | --- |
| `ui/controls.rs` | Top-level controls, input selectors, trace visibility, map controls, and web run controls. |
| `ui/pages.rs` | Overview, motion, mount, calibration, sensors, and diagnostics layouts. |
| `ui/plots.rs` | Reusable egui plot sections, shared cursors, decimation, log axes, and hover behavior. |
| `ui/maps.rs` | Map tile source selection, map overlays, markers, arrows, and synthetic local trajectory drawing. |
| `ui/inspector.rs` | Update allocation and covariance/residual inspector aggregation/rendering. |
| `ui/tuning.rs` | Align, Reduced, and Full filter tuning panels. |
| `ui/web.rs` | Browser-only dataset manifest loading, gzip/CSV loading, worker orchestration, and synthetic web inputs. |
| `ui/runtime.rs` | App construction, startup state, replay refresh, and egui frame lifecycle. |
| `ui/state.rs` | Data-origin state, tuning-window state, trace labels, and visibility classification. |
| `ui/trace_query.rs` | Trace interpolation, filtering, derived traces, and mount-reference lookup. |
| `ui/colors.rs` | Shared color rules for plots, maps, markers, and tooltips. |
| `ui/orthogonal.rs` | Vehicle and sensor angle popups for attitude and mount hover views. |
| `ui/windows.rs` | Floating tuning and update-inspector windows. |

## Full EKF Diagnostics

Full EKF accel and gyro diagnostic plots use the same raw body-frame IMU stream
that is fed through the public `sensor_fusion` API. Reduced and Full mount plots
use the current `qcs` mount quaternion directly, where `qcs` stores the physical
vehicle-to-body mount `q_bv`.

The Full EKF accel-bias states are additive correction states:

```text
corrected_accel = accel_scale * raw_accel + accel_bias
```

So a physical sensor bias has the opposite sign of the plotted Full EKF bias state. The Full EKF Z-bias investigation showed that vertical accel bias is observable through GNSS vertical position/velocity, but estimating accel scale and accel bias together makes the Z channel underdetermined because the vehicle Z axis is dominated by gravity. The visualizer Full path therefore keeps accel scale fixed by default and lets accel bias absorb the remaining correction.
