# imu_gnss_fusion

[![CI](https://github.com/yongkyuns/imu_gnss_fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/yongkyuns/imu_gnss_fusion/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-stable-orange.svg)](https://www.rust-lang.org/)

`imu_gnss_fusion` is a Rust workspace for experimenting with IMU, GNSS, wheel-speed, and u-blox dead-reckoning data. It contains an embedded-oriented filter crate, offline replay and visualization tools, UBX parsing support, and optional realtime logging utilities.

The project is useful for:

- replaying real u-blox UBX logs and generic IMU/GNSS CSV datasets,
- comparing align, loose INS/GNSS, and augmented ESKF behavior,
- generating synthetic trajectories for repeatable mount-angle experiments,
- validating generated Kalman-model code against focused Rust tests.

## Workspace Layout

| Path | Purpose |
| --- | --- |
| `ekf/` | `sensor-fusion` library crate. Contains align, loose, ESKF, generated model code, and filter API tests. |
| `sim/` | Replay, simulation, evaluation, diagnostics, and egui visualizer crate. |
| `ublox/` | Local UBX parser fork used by replay and logger tooling. |
| `logger/` | Optional realtime serial logger and Rerun UI. It is intentionally excluded from the root workspace and has its own lockfile. |
| `docs/` | Project documentation, math notes, architecture diagram, and generated PDFs/TEX notes. |
| `mobile/ios/` | Experimental iOS sensor collection app. |

See [docs/repo_architecture.png](docs/repo_architecture.png) for the high-level data flow.

## Architecture

The main replay path is:

1. Real device logs enter as raw UBX files, or synthetic/reference data enters as CSV.
2. `ublox` parses device packets and `sim::datasets` converts inputs into timestamped IMU/GNSS samples.
3. `sim::eval::replay` merges IMU and GNSS events in a consistent time order.
4. `sensor-fusion` runs the align, loose, and ESKF estimators.
5. `sim::visualizer` displays traces, map data, mount states, diagnostics, and summary statistics.

The runtime Rust filter code consumes generated matrix/Jacobian snippets under `ekf/src/generated_eskf/` and `ekf/src/generated_loose/`. The symbolic sources live in Python so model derivation stays reviewable while generated Rust stays fast and dependency-light.

## Quick Start

Requirements:

- Rust stable. The workspace uses Rust 2024 crates plus the local `ublox` crate, which declares Rust 1.83+.
- On Linux, install `pkg-config` and `libudev-dev` for serial/device dependencies used by `sim`.
- Python with `sympy` is only needed when regenerating Kalman-model Rust files.

Build and test the main workspace:

```bash
cargo build --workspace --locked
cargo test --workspace --locked
```

Run the visualizer on a UBX log:

```bash
cargo run --release -p sim --bin visualizer -- /path/to/ubx_raw_*.bin
```

Run the visualizer on a synthetic scenario:

```bash
cargo run --release -p sim --bin visualizer -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low
```

Run the optional realtime logger from its crate directory:

```bash
cd logger
cargo run --release --bin realtime_rerun_logger -- \
  --port /dev/tty.usbmodemXXXX \
  --baud 230400
```

By default, logger output is written under `logger/data/`.

## Filter And Replay Modes

The visualizer and primary A/B analyzers use `--misalignment` to select the mount-angle source:

| Mode | Behavior |
| --- | --- |
| `internal` | Align seeds the mount angle; ESKF then estimates residual mount states. |
| `external` | ESKF continuously follows Align and freezes its residual mount states. |
| `ref` | Uses ESF-ALG/reference mount angles when available. |

Legacy aliases are accepted in some tools. See [sim/README.md](sim/README.md) for the current tool map and alias details.

## Data Formats

The common hardware-agnostic replay directory contains two CSV files:

`imu.csv`

```text
t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2
```

`gnss.csv`

```text
t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,pos_std_n_m,pos_std_e_m,pos_std_d_m,vel_std_n_mps,vel_std_e_mps,vel_std_d_mps,heading_rad
```

`heading_rad` may be `NaN` when heading is unavailable. Producers include `export_gnss_ins_sim_generic` and `convert_ubx_to_generic_replay`.

Example conversions:

```bash
cargo run --release -p sim --bin convert_ubx_to_generic_replay -- \
  /path/to/ubx_raw.bin /tmp/replay

cargo run --release -p sim --bin export_gnss_ins_sim_generic -- \
  /path/to/gnss-ins-sim/output /tmp/replay \
  --signal-source meas
```

## Generated-Code Workflow

Generated Rust files are checked in and included by `ekf/src/generated_eskf.rs` and `ekf/src/generated_loose.rs`.

Regenerate ESKF model code after changing `ekf/eskf.py`:

```bash
python ekf/eskf.py --emit-rust
```

Regenerate loose-reference model code after changing `ekf/ins_gnss_loose.py`:

```bash
python ekf/ins_gnss_loose.py --emit-rust
```

After regeneration, review the generated diffs and run targeted tests from [docs/testing.md](docs/testing.md).

## Tests

Common local checks:

```bash
cargo test -p sensor-fusion --locked
cargo test -p sim --locked
cargo test -p ublox --locked
cargo test --workspace --locked
```

See [docs/testing.md](docs/testing.md) for focused test groups, fixtures, and expensive/local-data notes.

## Documentation

- [docs/README.md](docs/README.md): documentation index.
- [docs/testing.md](docs/testing.md): testing workflow.
- [docs/math/frames.md](docs/math/frames.md): frame and quaternion conventions.
- [docs/math/loose.md](docs/math/loose.md): loose INS/GNSS reference notes.
- [docs/eskf_mount_formulation.pdf](docs/eskf_mount_formulation.pdf): detailed ESKF mount formulation.
- [docs/align_nhc_formulation.pdf](docs/align_nhc_formulation.pdf): detailed Align/NHC formulation.

## License

MIT. See [LICENSE](LICENSE).
