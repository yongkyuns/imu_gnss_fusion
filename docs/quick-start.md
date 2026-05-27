# Quick Start

## Try The Web Visualizer

Open the hosted visualizer:

[https://yongkyuns.github.io/imu_gnss_fusion/](https://yongkyuns.github.io/imu_gnss_fusion/)

The visualizer can run checked-in hosted datasets, synthetic scenarios compiled into the wasm app, or user-provided generic replay CSVs.

## Build And Test Locally

Requirements:

- Rust stable.
- `wasm32-unknown-unknown` target and `wasm-bindgen-cli` for the browser visualizer.
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

Build and serve the browser visualizer:

```bash
cargo build -p sim --bin visualizer --release --target wasm32-unknown-unknown --locked
wasm-bindgen --target web --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/visualizer.wasm
python3 -m http.server --directory web 8080
```

Build this documentation site:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs target/docs-html
```
