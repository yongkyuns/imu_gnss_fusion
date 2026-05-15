# Documentation

This directory collects project-level notes for IMU/GNSS Fusion. The root
[README](../README.md) is the setup entrypoint; these pages keep deeper
architecture, math, replay, and testing details out of the main guide.

## Start Here

- [Repository architecture](architecture.md): crate boundaries, module
  ownership, generated code, replay data flow, browser flow, and documentation
  ownership.
- [API and conventions](api-and-conventions.md): public `sensor_fusion` API
  contract, frames, quaternions, units, mount modes, readiness, and integration
  pitfalls.
- [EKF algorithms](filter-algorithms.md): runtime state, propagation, update
  families, covariance behavior, tuning, generated code, and observability
  limits.
- [Data and simulation](data-and-simulation.md): generic replay CSV schemas,
  optional references, synthetic motion DSL, noise models, hosted dataset
  packaging, and external converter checklist.
- [Visualizer, tools, and testing](visualizer-tools-testing.md): native/web
  visualizer architecture, UI trace groups, worker flow, diagnostics,
  command-line tools, CI, and regression-test workflow.
- [EKF diagnostics](ekf-diagnostics.md): current diagnostic tools and what to
  record from replay investigations.
- [Testing](testing.md): local test commands, targeted suites, fixtures, and
  expensive-data notes.
- [Frame conventions](math/frames.md): short index for navigation, ECEF, raw
  body, vehicle, and mount frames.
- [EKF math notes](math/ekf.md): concise links for formulation PDFs/TEX.
- [Simulation tooling map](../sim/README.md): stable `sim` binaries, generic
  replay schema, and supported visualizer modes.
- [Browser visualizer](../web/README.md): wasm build and static hosting
  instructions.

## Repository Structure

| Path | Contents |
| --- | --- |
| `sensor_fusion/` | `sensor_fusion` Rust library, EKF runtime, generated formulas, SymPy model sources, and filter-level tests. |
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
| `align` | Mount-alignment estimator. Estimates the `q_bv` vehicle-to-body mount quaternion from stationary gravity, GNSS-derived acceleration, turn-rate windows, and nonholonomic cues. |
| `SensorFusion` | High-level runtime API. Accepts timestamped sensor samples, manages alignment and EKF initialization, owns local anchoring, and exposes current runtime state. |
| `math`, `nav`, `covariance`, generated wrappers | Internal helpers for quaternion/vector math, WGS84/ECEF navigation, generated-model glue, and covariance propagation. |

Generated files should not be edited by hand. Update the matching symbolic
formulation source, regenerate, then review the Rust diff.

### `sim` (`sim/src`)

| Module | Responsibility |
| --- | --- |
| `datasets/generic_replay` | Hardware-agnostic replay schema and CSV readers/writers for `imu.csv`, `gnss.csv`, and optional references. |
| `datasets/synthetic_replay` | Loader for generated synthetic truth/measured sample files. |
| `eval/replay` | Canonical IMU/GNSS event merge order used by tests, diagnostics, and visualization. |
| `eval/gnss_ins` | Quaternion, rotation, and signal-source helpers used by synthetic and replay analysis. |
| `eval/state_summary` and `eval/trace` | Trace lookup, sampling, convergence summaries, and final/tail error metrics. |
| `synthetic/motion_dsl` | Rust-native high-level motion syntax for synthetic scenarios. |
| `synthetic/gnss_ins_path` | Path generation, Earth model helpers, generated truth samples, IMU/GNSS noise models, and noisy measurement generation. |
| `visualizer/model` | Serializable plot, map, page, heading, contribution, and update-inspector data structures. |
| `visualizer/pipeline/generic` | Generic CSV replay pipeline used by native and browser visualization. Feeds the public `SensorFusion` API and builds plot groups. |
| `visualizer/pipeline/synthetic` | Synthetic scenario pipeline. Generates synthetic samples, converts them to the same generic replay path, and adds truth overlays. |
| `visualizer/replay_job` | Shared replay-job orchestration and web transport decimation. |
| `visualizer/theme`, `stats`, `math` | UI theme, trace/map statistics, and coordinate/math helpers. |
| `visualizer/ui/*` | Modular egui UI implementation for controls, pages, plots, maps, inspector, tuning, web loading, runtime, colors, and popups. |

## Detailed Math Notes

- [Align/NHC formulation PDF](align.pdf) and [TEX](align.tex).
- [EKF formulation PDF](ekf.pdf) and [TEX](ekf.tex).
- [Roll observability PDF](roll-observability.pdf) and [TEX](roll-observability.tex).

## Documentation Conventions

- Keep README-level material short and operational.
- Prefer `sim/README.md` for crate-specific tool inventory and diagnostics.
- Put estimator conventions and equations in the formulation PDFs under
  `docs/`, with TEX kept as the editable source.
- Keep `docs/math/*.md` as index and operational-reference pages, not duplicate
  derivations.
- When generated Rust changes, update both the symbolic source notes and the
  testing notes if the verification path changes.
