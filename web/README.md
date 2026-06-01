# Browser Visualizer

This directory is a static host for the wasm visualizer. Build the wasm bundle into `web/pkg/`, then serve this directory from any static file server or GitHub Pages.

```bash
cargo build -p fusion_tools --bin visualizer --release --target wasm32-unknown-unknown
wasm-bindgen \
  --target web \
  --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/visualizer.wasm
python3 -m http.server --directory web 8080
```

Open `http://localhost:8080`. The browser visualizer can generate built-in synthetic scenarios, load a generic replay by dragging `imu.csv`, `gnss.csv`, and optional reference CSVs into the app, or load an experimental generic dataset from `web/datasets/manifest.json`.

The wasm build enables WebGPU with a WebGL2 fallback through `wgpu`, so browsers that do not expose `navigator.gpu` can still run the visualizer when WebGL2 is available.

The overview page embeds the map beside the primary plots. Browser and native maps are rendered by the Rust `walkers` egui widget, so map interaction stays inside the same canvas as the rest of the visualizer. The app has light and dark themes; the map source follows the selected theme. Without a Mapbox token, maps use CARTO Positron for light theme and CARTO Dark Matter for dark theme, with OpenStreetMap/CARTO attribution. With a Mapbox token, maps use Mapbox Light or Mapbox Dark to match the app theme.

```text
http://localhost:8080/?theme=light&mapbox_token=<token>
```

The browser stores the selected theme and any token entered in the field in local storage, so they are reused on later reloads. For local development without entering a token manually, create an ignored `web/local-config.js`:

```js
window.IMU_GNSS_FUSION_LOCAL_CONFIG = {
  mapboxToken: "<token>",
  theme: "dark",
};
```

Native visualizer builds use the same map fallback behavior: set `MAPBOX_ACCESS_TOKEN` for Mapbox tiles, or leave it unset for CARTO Positron/Dark Matter. Set `IMU_GNSS_FUSION_THEME=light` or `IMU_GNSS_FUSION_THEME=dark` to choose the startup theme.

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
      "reference_position_gz": "reference_position.csv.gz",
      "reference_attitude_gz": "reference_attitude.csv.gz",
      "reference_mount_gz": "reference_mount.csv.gz",
      "reference_motion_gz": "reference_motion.csv.gz"
    }
  ]
}
```

If `imu_gz`/`gnss_gz` are omitted, the loader tries `imu.csv.gz` and `gnss.csv.gz` under `base_url`, then falls back to plain `imu.csv` and `gnss.csv`. Plain CSV paths can also be set explicitly with `imu` and `gnss`. Reference files are optional and only fetched when listed explicitly. `reference_position.csv` is rendered as the fused reference trajectory on the map, `reference_motion.csv` provides vehicle-frame reference angular velocity and gravity-compensated acceleration in the Motion tab, and `gnss.csv` remains the GNSS-only trajectory and filter input.

The web dataset picker groups manifest entries whose id starts with `ios-` or label starts with `iOS ` under "iOS recordings"; all other manifest entries appear under "UBX/reference datasets".

## Adding an iOS recording

iOS raw recordings are stored as `.motionfusion` JSON logs. Do not commit the raw log for normal web-visualizer use. Convert it to generic replay CSVs, package those CSVs as compressed static assets, and update the web and CI manifests:

```bash
python3 scripts/package_ios_motionfusion_dataset.py \
  target/ios-raw-sessions/<recording>.motionfusion \
  --dataset-id ios-mixed-drive-reverse-parking-001 \
  --title "iOS mixed drive with reverse parking 001" \
  --label "iOS mixed drive, reverse parking 001" \
  --description "iOS MotionFusion mixed-road drive with a low-speed reverse-parking maneuver; exported with course-accuracy velocity covariance."
```

The script writes an intermediate generic replay directory under `target/ios-web-replay/<dataset-id>/`, packages the official browser assets under `web/datasets/<dataset-id>/`, and appends or replaces entries in:

- `web/datasets/manifest.json`, used by the browser dataset picker.
- `.github/datasets/generic-datasets.json`, used by hosted dataset validation.

Pass `--force` to replace an existing dataset package and manifest entries. Pass `--no-update-ci-manifest` for a local-only browser dataset. The generated package contains `imu.csv.gz`, `gnss.csv.gz`, and a per-dataset `manifest.json`. iOS logs usually do not have `reference_position`, `reference_attitude`, `reference_mount`, or `reference_motion` files, so those overlays are absent unless generated separately.

Validate the result before committing:

```bash
python3 scripts/test_package_ios_motionfusion_dataset.py
node scripts/validate_generic_datasets.mjs \
  --manifest .github/datasets/generic-datasets.json \
  --cache-dir .cache/generic-datasets \
  --work-dir target/generic-datasets
```

Then serve `web/` and open the dataset by id:

```bash
python3 -m http.server --directory web 8080
open "http://127.0.0.1:8080/?dataset=ios-mixed-drive-reverse-parking-001&theme=dark"
```

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

CI builds the wasm bundle, builds Sphinx documentation, assembles a staged Pages artifact, and validates the static site before upload. The visualizer stays at the artifact root and generated docs are copied to `/docs/`:

```bash
node scripts/validate_pages_static.mjs \
  --site-dir target/pages-site \
  --require-wasm \
  --require-docs
```

The validator checks that `index.html` uses relative wasm paths, required wasm files exist, `visualizer_bg.wasm` has a wasm header, the Sphinx docs entry point exists, dataset URLs are safe, required static files are fetchable, and the local static server returns Pages-compatible MIME types for HTML, JavaScript, wasm, CSS, and common Sphinx assets. `local-config.js` is intentionally excluded from the Pages artifact; the browser treats it as an optional local-development override.
