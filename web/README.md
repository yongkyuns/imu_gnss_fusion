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

Open `http://localhost:8080`. The browser visualizer can generate built-in synthetic scenarios, load a generic replay by dragging `imu.csv`, `gnss.csv`, and optional reference CSVs into the app, or load an experimental generic dataset from `web/datasets/manifest.json`.

The map page is available with `?page=map`. Browser and native maps are rendered by the Rust `walkers` egui widget, so map interaction stays inside the same canvas as the rest of the visualizer. Maps use OpenStreetMap tiles by default. To use Mapbox dark tiles, enter a token in the map-page token field or pass it in the URL:

```text
http://localhost:8080/?page=map&mapbox_token=<token>
```

The browser stores a token entered in the field in local storage, so it is reused on later reloads. For local development without entering a token manually, create an ignored `web/local-config.js`:

```js
window.IMU_GNSS_FUSION_CONFIG = {
  mapboxToken: "<token>",
};
```

Native visualizer builds use the same fallback behavior: set `MAPBOX_ACCESS_TOKEN` for Mapbox tiles, or leave it unset for OpenStreetMap.

## Experimental dataset manifest

The browser loads `datasets/manifest.json` at startup. Entries are hardware-agnostic generic replay datasets:

```json
{
  "datasets": [
    {
      "id": "example",
      "label": "Example replay",
      "description": "Optional short browser UI note.",
      "base_url": "example/",
      "imu_gz": "imu.csv.gz",
      "gnss_gz": "gnss.csv.gz",
      "reference_attitude_gz": "reference_attitude.csv.gz",
      "reference_mount_gz": "reference_mount.csv.gz"
    }
  ]
}
```

If `imu_gz`/`gnss_gz` are omitted, the loader tries `imu.csv.gz` and `gnss.csv.gz` under `base_url`, then falls back to plain `imu.csv` and `gnss.csv`. Plain CSV paths can also be set explicitly with `imu` and `gnss`. Reference files are optional and only fetched when listed explicitly.

## FPS benchmark

After building `web/pkg/`, run the automated browser benchmark:

```bash
node scripts/benchmark_web_fps.mjs --scenario city_blocks --min-fps 55
```

The script has no npm package dependencies. It serves `web/`, launches Chrome/Chromium headless through the DevTools protocol, starts the wasm visualizer with `?bench=1&scenario=...`, warms up, then samples both browser `requestAnimationFrame` timing and egui frame timing while moving the mouse over the canvas. Use `--activity none` for an idle measurement, `--json` for machine-readable output, or `--min-fps <n>` to fail a CI job below a threshold.

Requirements:

- Node.js 22 or newer, for the built-in WebSocket client.
- Google Chrome, Chromium, or Microsoft Edge. If it is not in a standard location, pass `--browser /path/to/chrome`.
- A built wasm bundle at `web/pkg/visualizer.js` and `web/pkg/visualizer_bg.wasm`.

## GitHub Pages artifact validation

CI builds the wasm bundle, writes `web/pkg/`, and validates the static site before uploading a Pages artifact:

```bash
node scripts/validate_pages_static.mjs --site-dir web --require-wasm
```

The validator checks that `index.html` uses relative wasm paths, required wasm files exist, `visualizer_bg.wasm` has a wasm header, and the local static server returns Pages-compatible MIME types for HTML, JavaScript, and wasm.
