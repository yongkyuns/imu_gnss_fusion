# Browser Visualizer

This directory is a static host for the wasm visualizer. Build the wasm bundle into `web/pkg/`, then serve this directory from any static file server or GitHub Pages.

```bash
cargo build -p sim --bin visualizer --release --target wasm32-unknown-unknown
wasm-bindgen \
  --target web \
  --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/visualizer.wasm
python3 -m http.server --directory web 8080
```

Open `http://localhost:8080`. The browser visualizer can generate built-in synthetic scenarios or load a generic replay by dragging both `imu.csv` and `gnss.csv` into the app.
