# imu_gnss_fusion

`imu_gnss_fusion` is a Rust workspace for u-blox dead-reckoning workflows:

- Works with u-blox EVKs that support DR (for example, M9DR-class devices).
- Fuses IMU, GNSS, and wheel-speed signals with an EKF pipeline.
- Provides batteries-included logging, simulation, and visualization tools.
- Visualization stack is designed to be cross-platform (macOS, Linux, Windows, and web-oriented viewer flow).

## Workspace crates

- `sim`: offline replay/simulation, diagnostics, and visualization app.
- `logger`: realtime serial logger and Rerun-based live UI (`realtime_rerun_logger`).
- `ekf`: Rust EKF implementation crate, including generated model code.
- `ublox`: local fork of the UBX parsing crate used by the workspace.

See `docs/repo_architecture.png` for a high-level workspace data-flow diagram.

## Build

```bash
cd simulation/imu_gnss_fusion
cargo build --release
```

## Run simulation/visualization

```bash
cargo run --release -p sim --bin visualizer -- /path/to/ubx_raw_*.bin
```

The visualizer's `--misalignment` option selects one mount-angle source:

- `internal`: seed from Align, then let ESKF estimate residual fine alignment.
- `external`: keep ESKF mount states frozen and continuously follow Align.
- `ref`: use ESF-ALG/reference mount angles.

For example:

```bash
cargo run --release -p sim --bin visualizer -- \
  logger/data/ubx_raw_20260328_153757.bin \
  --misalignment external
```

## Run realtime logger

```bash
cargo run --release -p logger --bin realtime_rerun_logger -- \
  --port /dev/tty.usbmodemXXXX \
  --baud 230400
```

By default, raw UBX logs are written to `logger/data/`.

## License

MIT. See `LICENSE`.
