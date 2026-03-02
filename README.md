# imu_gnss_fusion

`imu_gnss_fusion` is a Rust workspace for u-blox dead-reckoning workflows:

- Works with u-blox EVKs that support DR (for example, M9DR-class devices).
- Fuses IMU, GNSS, and wheel-speed signals with an EKF pipeline.
- Provides batteries-included logging, simulation, and visualization tools.
- Visualization stack is designed to be cross-platform (macOS, Linux, Windows, and web-oriented viewer flow).

## Workspace crates

- `sim`: offline replay/simulation and visualization app (`visualize_pygpsdata_log`).
- `logger`: realtime serial logger and Rerun-based live UI (`realtime_rerun_logger`).
- `ekf`: EKF implementation crate (including generated model code and C parity assets).
- `ublox`: local fork of the UBX parsing crate used by the workspace.

## Build

```bash
cd simulation/imu_gnss_fusion
cargo build --release
```

## Run simulation/visualization

```bash
cargo run --release -p sim -- /path/to/ubx_raw_*.bin
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
