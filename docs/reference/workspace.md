# Workspace Reference

| Path | Purpose |
| --- | --- |
| `sensor_fusion/` | no-std fusion library, high-level `SensorFusion` facade, alignment estimator, EKF runtime, tests |
| `road_events/` | no-std streaming road-event detectors and trip statistics |
| `sim/` | replay, simulation, diagnostics, synthetic generation, native/wasm visualizer |
| `web/` | static browser host for the wasm visualizer and hosted dataset files |
| `mobile/ios/` | iOS app, Rust FFI wrapper, recording/export tools |
| `scripts/` | dataset packaging, validation, web benchmark, and support scripts |
| `docs/` | Sphinx source, integrated algorithm notes, and documentation assets |

Generated outputs belong under ignored build paths such as `target/`, `web/pkg/`, `web/docs/`, `mobile/ios/build/`, or crate-local `target/` directories.
