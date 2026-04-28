# imu_gnss_fusion

[![CI](https://github.com/yongkyuns/imu_gnss_fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/yongkyuns/imu_gnss_fusion/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-stable-orange.svg)](https://www.rust-lang.org/)

`imu_gnss_fusion` is a Rust workspace for experimenting with IMU, GNSS, and wheel-speed sensor fusion. It contains an embedded-oriented filter crate, offline replay and visualization tools, synthetic trajectory generation, and hardware-agnostic CSV replay support.

The project is useful for:

- replaying hardware-agnostic IMU/GNSS CSV datasets,
- comparing align, loose INS/GNSS, and augmented ESKF behavior,
- generating synthetic trajectories for repeatable mount-angle experiments,
- validating generated Kalman-model code against focused Rust tests.

## Workspace Layout

| Path | Purpose |
| --- | --- |
| `ekf/` | `sensor-fusion` library crate. Contains align, loose, ESKF, generated model code, and filter API tests. |
| `sim/` | Replay, simulation, evaluation, diagnostics, and egui visualizer crate. |
| `docs/` | Project documentation, math PDFs/TEX sources, and test guidance. |
| `web/` | Static browser host for the wasm visualizer. |
| `mobile/ios/` | Experimental iOS sensor collection app. |

## Architecture

The main replay path is:

1. Data enters as hardware-agnostic `imu.csv` and `gnss.csv`, or as a synthetic scenario generated inside `sim`.
2. `sim::datasets` converts CSV rows into timestamped IMU/GNSS samples.
3. `sim::eval::replay` merges IMU and GNSS events in a consistent time order.
4. `sensor-fusion` runs the align, loose, and ESKF estimators.
5. `sim::visualizer` displays traces, map data, mount states, diagnostics, and summary statistics.

The runtime Rust filter code consumes generated matrix/Jacobian snippets under `ekf/src/generated_eskf/` and `ekf/src/generated_loose/`. The symbolic sources live in Python so model derivation stays reviewable while generated Rust stays fast and dependency-light.

## Quick Start

Requirements:

- Rust stable. The workspace uses Rust 2024 crates.
- Python with `sympy` is only needed when regenerating Kalman-model Rust files.

Build and test the main workspace:

```bash
cargo build --workspace --locked
cargo test --workspace --locked
```

Run the visualizer on a generic replay directory:

```bash
cargo run --release -p sim --bin visualizer -- \
  --generic-replay-dir /path/to/replay-dir
```

Run the visualizer on a synthetic scenario:

```bash
cargo run --release -p sim --bin visualizer -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low
```

Build the browser visualizer:

```bash
cargo build -p sim --bin visualizer --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/visualizer.wasm
python3 -m http.server --directory web 8080
```

## Filter And Replay Modes

The visualizer and primary A/B analyzers use `--misalignment` to select the mount-angle source:

| Mode | Behavior |
| --- | --- |
| `internal` | Align seeds the mount angle; ESKF then estimates residual mount states. |
| `external` | ESKF continuously follows Align and freezes its residual mount states. |
| `ref` | Uses reference mount angles when available in synthetic or converted data. |

See [sim/README.md](sim/README.md) for the current tool map.

## Data Formats

The common hardware-agnostic replay directory contains two required CSV files:

`imu.csv`

```text
t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2
```

`gnss.csv`

```text
t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,pos_std_n_m,pos_std_e_m,pos_std_d_m,vel_std_n_mps,vel_std_e_mps,vel_std_d_mps,heading_rad
```

`heading_rad` may be `NaN` when heading is unavailable. Producers include `export_gnss_ins_sim_generic`; hardware-specific converters should live outside this repository and emit this schema.

Replay directories can also include optional reference traces used only for evaluation and visualization:

```text
reference_attitude.csv
reference_mount.csv
```

Both reference CSVs use `t_s,roll_deg,pitch_deg,yaw_deg`. They are intentionally generic: a converter may derive them from any trusted reference system, but this repository does not depend on the reference device protocol.

Example conversions:

```bash
cargo run --release -p sim --bin export_gnss_ins_sim_generic -- \
  /path/to/gnss-ins-sim/output /tmp/replay \
  --signal-source meas
```

Package generic replay data for static hosting:

```bash
python3 scripts/package_dataset.py /path/to/replay-dir /tmp/hosted-drive
```

The hosted dataset layout is:

```text
manifest.json
imu.csv.gz
gnss.csv.gz
reference_attitude.csv.gz  # optional
reference_mount.csv.gz     # optional
```

`scripts/package_dataset.py` can stage an existing generic replay directory or call `export_gnss_ins_sim_generic` for `gnss-ins-sim` output. Raw `.bin` logs are intentionally not supported in this repository because the prior UBX path depended on deleted device-specific parsing code.

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
cargo test --workspace --locked
```

See [docs/testing.md](docs/testing.md) for focused test groups, fixtures, and expensive/local-data notes.

## Documentation

- [docs/README.md](docs/README.md): documentation index.
- [docs/testing.md](docs/testing.md): testing workflow.
- [docs/math/frames.md](docs/math/frames.md): frame and quaternion conventions.
- [docs/math/loose.md](docs/math/loose.md): loose INS/GNSS operational links.
- [docs/eskf_mount_formulation.pdf](docs/eskf_mount_formulation.pdf): detailed ESKF mount formulation.
- [docs/align_nhc_formulation.pdf](docs/align_nhc_formulation.pdf): detailed Align/NHC formulation.
- [docs/loose_formulation.pdf](docs/loose_formulation.pdf): detailed loose INS/GNSS formulation.

## License

MIT. See [LICENSE](LICENSE).
