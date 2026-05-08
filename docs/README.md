# Documentation

This directory collects project-level notes for the IMU/GNSS fusion workspace. The root [README](../README.md) is the entrypoint for setup and common workflows; these pages keep deeper math, testing, and architecture details out of the main guide.

## Start Here

- [Testing](testing.md): local test commands, targeted suites, fixtures, and expensive-data notes.
- [Frame conventions](math/frames.md): short index for navigation, ECEF, body, vehicle, seeded, and corrected frames.
- [Full EKF notes](math/full.md): concise operational links for the Full EKF.
- [Simulation tooling map](../sim/README.md): stable `sim` binaries, generic replay schema, and supported visualizer modes.
- [Browser visualizer](../web/README.md): wasm build and static hosting instructions.

## Repository Structure

The repository is organized around one reusable filter crate and one tooling
crate:

| Path | Contents |
| --- | --- |
| `sensor_fusion/` | `sensor_fusion` Rust library, generated filter formulas, SymPy model sources, and filter-level tests. |
| `sim/` | Offline replay, evaluation, synthetic data generation, diagnostics, native visualizer, and browser visualizer entrypoint. |
| `docs/` | Math PDFs/TEX, frame notes, testing notes, and architecture assets. |
| `web/` | Static host for the wasm visualizer and hosted generic replay datasets. |
| `scripts/` | Dataset packaging, static-site validation, generic-dataset validation, and browser FPS benchmarking. |
| `mobile/ios/` | Experimental iOS sensor collection app; not part of the filter runtime. |

Hardware-specific log conversion is intentionally outside this repository.
Inputs here should already be generic `imu.csv` and `gnss.csv` replay files,
with optional generic reference CSVs for visualization and evaluation.

## Rust Module Overview

### `sensor_fusion` (`sensor_fusion/src`)

| Module | Responsibility |
| --- | --- |
| `align` | Mount-alignment filter. Estimates the vehicle-to-body mount quaternion from stationary gravity, GNSS-derived acceleration, turn-rate windows, and nonholonomic cues. |
| `SensorFusion` | High-level runtime API. Accepts timestamped sensor samples, manages alignment and Reduced initialization, owns the local WGS84 anchor, and exposes current filter states. |
| `reduced` | Reduced EKF runtime, process-noise configuration, public state/sample/diagnostic structs, and focused state helpers. |
| `full` | Full-state ECEF EKF used for comparison and diagnostics. Includes Full prediction, GNSS and NHC updates, and Full-specific process-noise configuration. |
| `math`, `nav`, `covariance`, generated wrappers | Internal shared helpers for quaternion/vector math, WGS84/ECEF navigation, generated-model glue, and covariance propagation. |

Generated files under `sensor_fusion/src/generated_reduced/` and
`sensor_fusion/src/generated_full/` should not be edited by hand. Update
`sensor_fusion/reduced.py` or `sensor_fusion/ins_gnss_full.py`, regenerate, then review the Rust
diff.

### `sim` (`sim/src`)

| Module | Responsibility |
| --- | --- |
| `datasets/generic_replay` | Hardware-agnostic replay schema and CSV readers/writers for `imu.csv`, `gnss.csv`, and optional references. |
| `datasets/synthetic_replay` | Loader for generated synthetic truth/measured sample files. |
| `datasets/seeded_full` | Fixture loader for seeded Full EKF diagnostics. |
| `eval/replay` | Canonical IMU/GNSS event merge order used by tests, diagnostics, and visualization. |
| `eval/gnss_ins` | Quaternion, rotation, and signal-source helpers used by synthetic and replay analysis. |
| `eval/state_summary` and `eval/trace` | Trace lookup, sampling, convergence summaries, and final/tail error metrics. |
| `synthetic/motion_dsl` | Rust-native high-level motion syntax for synthetic scenarios. |
| `synthetic/gnss_ins_path` | Path generation, Earth model helpers, generated truth samples, IMU/GNSS noise models, and noisy measurement generation. |
| `visualizer/model` | Serializable plot, map, page, heading, contribution, and update-inspector data structures. |
| `visualizer/pipeline/generic` | Generic CSV replay pipeline used by native and browser visualization. Feeds the public `SensorFusion` API and builds all plot groups. |
| `visualizer/pipeline/synthetic` | Synthetic scenario pipeline. Generates synthetic samples, converts them to the same generic replay path, and adds truth overlays. |
| `visualizer/replay_job` | Shared replay-job orchestration and web transport decimation. |
| `visualizer/theme`, `stats`, `math` | UI theme, trace/map statistics, and coordinate/math helpers. |
| `visualizer/ui/*` | Modular egui UI implementation for controls, pages, plots, maps, inspector, tuning, web loading, runtime, colors, and popups. |

### Visualizer UI Modules

| Module | Responsibility |
| --- | --- |
| `ui/controls` | Top bar, input selectors, web run controls, map controls, global trace visibility, and page tabs. |
| `ui/pages` | Page layout for overview, motion, mount, calibration, sensors, and diagnostics. |
| `ui/plots` | Reusable plot sections, foldable overview plots, decimation, shared cursor lines, log-scale formatting, and hover popups. |
| `ui/maps` | Mapbox/CARTO tile selection, map overlay drawing, hover markers, heading arrows, and synthetic local-map rendering. |
| `ui/inspector` | Five-second hover-window update allocation, mount error ledger, covariance/residual tables, and heatmap rendering. |
| `ui/tuning` | Align, Reduced, and Full filter tuning controls. |
| `ui/web` | Browser-only manifest loading, gzip/CSV loading, dropped-file staging, worker requests, and synthetic scenario/noise inputs. |
| `ui/runtime` | App construction, startup theme/density, replay refresh, and egui update loop. |
| `ui/state` | Trace labels, trace visibility classification, data-origin state, and tuning-panel state. |
| `ui/trace_query` | Pure trace filtering, interpolation, mount-reference lookup, and derived trace helpers. |
| `ui/colors` | Shared color policy for plots, map overlays, markers, and tooltips. |
| `ui/orthogonal` | Vehicle and sensor orthogonal angle popups for attitude and mount plots. |
| `ui/windows` | Floating tuning and update-inspector windows. |

## Detailed Math Notes

- [Mount-in-propagation revision PDF](mount_in_propagation_revision.pdf) and [TEX](mount_in_propagation_revision.tex).
- [Reduced EKF mount formulation PDF](reduced_mount_formulation.pdf) and [TEX](reduced_mount_formulation.tex).
- [Align/NHC formulation PDF](align_nhc_formulation.pdf) and [TEX](align_nhc_formulation.tex).
- [Full EKF formulation PDF](full_formulation.pdf) and [TEX](full_formulation.tex).

## Documentation Conventions

- Keep README-level material short and operational.
- Prefer `sim/README.md` for crate-specific tool inventory and diagnostics.
- Put estimator conventions and equations in the formulation PDFs under `docs/`, with TEX kept as the editable source.
- Keep `docs/math/*.md` as index and operational-reference pages, not duplicate derivations.
- When generated Rust changes, update both the symbolic source notes and the testing notes if the verification path changes.
