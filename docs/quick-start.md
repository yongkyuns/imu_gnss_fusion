# Quick Start

## Try The Web Visualizer

Open the hosted visualizer:

[https://yongkyuns.github.io/imu_gnss_fusion/](https://yongkyuns.github.io/imu_gnss_fusion/)

The visualizer can run checked-in hosted datasets, synthetic scenarios compiled into the wasm app, or user-provided generic replay CSVs.

## Integrate The Runtime

Create one `SensorFusion` object per live sensor stream and feed timestamped IMU/GNSS samples in order. Keep that object across ordinary stream pauses so the EKF can apply its built-in short/medium/long sleep behavior. Reset only for a new source, a changed mount, lost retained memory, or a replay switch.

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
cargo run --release -p sim --bin visualizer -- \
  --generic-replay-dir /path/to/replay-dir
```

Run the native visualizer on a synthetic scenario:

```bash
cargo run --release -p sim --bin visualizer -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low
```

For browser visualizer builds, documentation builds, and dataset publication,
see the Developer Reference section.
