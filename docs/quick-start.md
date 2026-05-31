# Quick Start

## Try The Web Visualizer

Open the hosted visualizer:

[https://yongkyuns.github.io/imu_gnss_fusion/](https://yongkyuns.github.io/imu_gnss_fusion/)

The visualizer can run checked-in hosted datasets, synthetic scenarios compiled into the wasm app, or user-provided generic replay CSVs.

## Integrate The Runtime

Create one `SensorFusion` object per live sensor stream and feed timestamped IMU/GNSS samples in order. Keep that object across intentional trip-end sleep by calling `SensorFusion::end_trip()` before stopping samples. Unmarked long sample gaps are treated as unexpected in-trip data loss and require GNSS reseed. Reset only for a new source, a changed mount, lost retained memory, or a replay switch.

Use `Update.navigation_usable` before consuming navigation output, and use `SensorFusion::health().stable` before saving priors. See [](runtime-state-and-persistence.md) for the full state model.

## Build And Test Locally

Requirements:

- Rust stable.
- Python only for scripts and generated-code workflows.

```bash
cargo build --workspace --locked
cargo test --workspace --locked
```

Run the native visualizer on a generic replay directory:

```bash
cargo run --release -p fusion_tools --bin visualizer -- \
  --generic-replay-dir /path/to/replay-dir
```

Run the native visualizer on a synthetic scenario:

```bash
cargo run --release -p fusion_tools --bin visualizer -- \
  --synthetic-motion-def tools/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low
```

For browser visualizer builds, documentation builds, and dataset publication,
see the Developer Reference section.
