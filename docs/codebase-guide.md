# Codebase Guide

This page is the fastest way to find the code that owns a behavior.

## Main Runtime Path

| Area | Files | What to read first |
| --- | --- | --- |
| Public fusion API | `sensor_fusion/src/lib.rs`, `sensor_fusion/src/fusion.rs`, `sensor_fusion/src/fusion_types.rs` | `SensorFusion`, input sample types, `Update`, runtime configuration setters |
| Health and persistence | `sensor_fusion/src/diagnostics.rs`, `sensor_fusion/src/fusion.rs` | `FusionState`, stable/persistable verdict, short/medium/long sleep handling |
| Alignment | `sensor_fusion/src/align/mod.rs` | stationary tilt initialization, horizontal yaw update, turn gyro refinement |
| Runtime EKF | `sensor_fusion/src/ekf/mod.rs`, `sensor_fusion/src/ekf/types.rs` | prediction, measurement updates, GNSS gating, NHC, vehicle-roll prior |
| Generated EKF formulation | `sensor_fusion/src/ekf/formulation.py`, `sensor_fusion/src/ekf/generated/` | symbolic model, generated `F`, `G`, scalar observation rows |
| Frame tests | `sensor_fusion/src/coordinate_conventions.rs`, `sensor_fusion/tests/` | convention checks and API-level behavior |

The user-facing `SensorFusion` facade is the correct boundary for applications.
Code outside `sensor_fusion` should feed raw body-frame IMU, GNSS, and optional
vehicle speed samples rather than reaching into EKF internals.

## Replay And Visualization Path

| Area | Files | Role |
| --- | --- | --- |
| Generic replay parsing | `tools/src/datasets/generic_replay.rs` | loads `imu.csv`, `gnss.csv`, and optional reference streams |
| Replay job orchestration | `tools/src/visualizer/replay_job.rs` | runs generic or synthetic jobs and produces `PlotData` |
| Trace construction | `tools/src/visualizer/pipeline/generic.rs` | feeds `SensorFusion`, builds traces, maps, event samples, and diagnostics |
| Synthetic scenarios | `tools/src/visualizer/pipeline/synthetic.rs`, `tools/motion_profiles/` | creates controlled replay inputs |
| Visualizer UI | `tools/src/visualizer/ui/` | pages, maps, plots, tuning windows, web dataset loading |
| Browser shell | `web/index.html`, `web/replay_worker.js`, `web/datasets/manifest.json` | wasm host, replay worker, hosted dataset picker |

The native and browser visualizers share the same Rust visualizer model from the
`fusion_tools` crate. `web/` is only the static HTML shell, wasm package output
location, hosted dataset manifest, and web worker layer.

## Road Events Path

| Area | Files | Role |
| --- | --- | --- |
| Detector crate | `road_events/src/` | no-std streaming event detectors and trip stats |
| Visualizer integration | `tools/src/visualizer/pipeline/generic.rs` | converts fusion outputs into road-event motion samples and plot traces |
| iOS integration | `mobile/ios/SensorFusionFFI/src/lib.rs`, `mobile/ios/IMUGNSSPhone/App/Analysis/MotionEventDetector.swift` | exposes detector events and display models to Swift |

`road_events` is independent from the EKF. It consumes vehicle-motion quantities
such as speed, vehicle-frame acceleration, pitch, and vertical acceleration.

## iOS App Path

| Area | Files | Role |
| --- | --- | --- |
| App shell and tabs | `mobile/ios/IMUGNSSPhone/App/ContentView.swift` | Drive, Review, Settings, debug Diagnostics |
| Sensor orchestration | `mobile/ios/IMUGNSSPhone/App/SensorStore.swift` | CoreMotion/CoreLocation streams, recording, replay, fusion lifecycle |
| Rust bridge | `mobile/ios/SensorFusionFFI/` | C ABI wrapper around `sensor_fusion` and `road_events` |
| Display models | `mobile/ios/IMUGNSSPhone/App/Models/` | health, resource use, map policy, settings state, motion-event display |
| Raw sessions | `mobile/ios/IMUGNSSPhone/App/Replay/`, `mobile/ios/scripts/export_motionfusion.py` | `.motionfusion` recording, replay, and export |
| Alerts | `mobile/ios/IMUGNSSPhone/App/Audio/` | chime/voice event notifications |

The app should not duplicate filter or road-event logic in Swift. Swift owns
collection, UI, persistence, replay, and presentation; Rust owns fusion and
detector behavior.

## Documentation And Diagrams

| Area | Files | Role |
| --- | --- | --- |
| Sphinx pages | `docs/**/*.md` | public documentation source |
| ELK diagram specs | `docs/assets/elk/*.json` | maintainable diagram source |
| Generated diagrams | `docs/_static/diagrams/*.svg` | committed SVGs used by Sphinx |
| Diagram renderer | `scripts/render_elk_diagrams.mjs` | renders ELK specs into styled SVGs |

Generated documentation output belongs under `target/docs-html` or
`target/pages-site`; those directories are not committed.

## Where To Make Changes

- Estimator behavior: change `sensor_fusion`, then update the algorithm/reference docs.
- Road-event detection: change `road_events`, then update road-event docs and iOS/visualizer integration notes.
- Replay format: change `fusion_tools::datasets`, packaging scripts, and data docs together.
- Visualizer UI or trace plots: change `tools/src/visualizer`, then update [](tools/visualizer.md).
- iOS collection/review behavior: change `mobile/ios`, then update [](mobile/ios.md).
- CI, Pages, and validation automation: keep details under the Developer Reference section.
