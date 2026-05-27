# Visualizer And Simulator

The `sim` crate owns replay, simulation, diagnostics, and the egui visualizer. The same visualizer code is built as a native binary and as a wasm app hosted from `web/`.

```{figure} ../_static/screenshots/web-visualizer-overview.png
:alt: Web visualizer overview with plots and map.
:class: framed

A replay overview with synchronized plots and map trace. Current builds also include the Events page, event filters, and road-event map overlays.
```

## Browser Visualizer

The GitHub Pages root serves the wasm visualizer. It supports:

- hosted generic datasets from `web/datasets/manifest.json`;
- drag/drop generic replay files;
- synthetic scenarios compiled into the app;
- worker-backed replay in the browser;
- map rendering with local theme configuration and optional Mapbox token;
- road-event overlays, filters, and Events-page plots.

The browser app loads `./pkg/visualizer.js` and `./pkg/visualizer_bg.wasm` relative to `web/index.html`. For this reason the docs site is deployed under `/docs/` rather than replacing the root page.

## Native Visualizer

Run a generic replay:

```bash
cargo run --release -p sim --bin visualizer -- \
  --generic-replay-dir /path/to/replay-dir
```

Run a synthetic scenario:

```bash
cargo run --release -p sim --bin visualizer -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low
```

## Synthetic Scenarios

The repository currently includes 14 `.scenario` motion profiles under `sim/motion_profiles`. The DSL supports controlled motion profiles used to exercise mount alignment, turn loops, grade changes, GNSS outages/data gaps, early bad basins, and fault-style robustness cases.

`export_synthetic_replay_generic` converts synthetic scenarios into generic replay directories so synthetic and field-style datasets can use the same replay path.
