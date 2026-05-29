# Visualizer And Simulator

The visualizer is the main inspection surface for replays, synthetic scenarios,
mount behavior, EKF diagnostics, and road events. The native app and browser app
share the same Rust replay pipeline and the same `PlotData` model, so plots in
the browser are intended to mean the same thing as plots in the desktop build.

```{figure} ../_static/screenshots/web-visualizer-overview.png
:alt: Web visualizer overview with plots and map.
:class: framed

A replay overview with synchronized plots and map trace. Current builds also
include Events, Diagnostics, tuning controls, and road-event map overlays.
```

```{figure} ../_static/diagrams/visualizer-architecture-orthogonal.svg
:alt: Visualizer architecture from hosted datasets, drag and drop CSVs, and synthetic scenarios through replay jobs, sensor fusion, road events, PlotData, and native/browser surfaces.
:class: framed

Hosted data, drag/drop files, and synthetic scenarios all become replay jobs.
Replay jobs feed `sensor_fusion` and `road_events`, then publish `PlotData` to
the native egui shell or the browser wasm shell.
```

## Entry Points

Browser visualizer:

- the public Pages root loads `web/index.html`;
- `web/index.html` loads `pkg/visualizer.js` and `pkg/visualizer_bg.wasm`;
- browser replay work runs through `web/replay_worker.js` so the UI stays
  responsive while a dataset is parsed and replayed.

Native visualizer:

```bash
cargo run --release -p sim --bin visualizer -- \
  --generic-replay-dir /path/to/replay-dir
```

Synthetic native run:

```bash
cargo run --release -p sim --bin visualizer -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low
```

## Data Sources

Hosted datasets are listed in `web/datasets/manifest.json` and grouped in the
browser's dataset picker. The current hosted set contains 32 datasets, including
reference-style generic replays and iOS recordings converted to the generic CSV
layout.

Drag/drop replay accepts text CSV files in the same generic layout. At minimum,
drop an `imu.csv` and `gnss.csv`. Optional reference files can add attitude,
mount, position, or vehicle-motion truth traces. Browser drag/drop expects plain
text files, not compressed archives.

Synthetic scenarios are compiled into the browser app and can also be loaded by
the native binary. Built-in browser scenarios include city blocks, figure eight,
straight acceleration/braking, and several mount-observability cases. Noise
presets are `None`, `Low`, `Mid`, and `High`; use `None` to separate model
behavior from sensor-noise behavior.

## Replay Product

Every visualizer run produces a `PlotData` value. That object is the boundary
between replay computation and UI rendering. It contains:

- synchronized time-series traces for motion, attitude, mount, bias,
  covariance, residuals, and update diagnostics;
- map-ready GNSS, fused, synthetic truth, and optional reference paths;
- road-event point samples, interval segments, trip summary counters, and
  captured trigger windows;
- cursor and inspector samples used to correlate plots, map hover state, and
  update-level EKF behavior;
- metadata describing source type, mount mode, replay configuration, and whether
  reference truth was present.

This structure matters because the native and browser shells render the same
analysis product. Browser replay adds asynchronous fetch/worker plumbing and
transport decimation; it does not change estimator equations or event-detector
logic.

## Page Semantics

`Overview` is the fastest sanity check. It combines map traces, speed, attitude,
navigation health, and summary plots.

`Motion` focuses on vehicle and navigation motion: NED position/velocity,
vehicle-frame velocity, acceleration channels, and route behavior.

`Mount` shows mount roll, pitch, yaw, covariance, reference comparison when
available, and update-allocation inspection used for roll-observability work.

`Calibration` collects estimator quantities that should move slowly: gyro bias,
accelerometer bias, mount states, and related uncertainty traces.

`Sensors` shows raw IMU and GNSS streams, timing, sample quality, and source
measurements. Use it first when a replay looks physically impossible.

`Events` is the road-event inspection page. It includes trip summary counters,
event filters, event point/segment overlays, and trigger-window plots for
signals such as vertical shock, roughness energy, harsh corner load, and hill
grade.

`Diagnostics` is for estimator internals: update residuals, accepted/rejected
measurements, covariance/correlation views, and health/lifecycle state.

## Map And Event Overlays

The map can show GNSS, fused, synthetic truth, and reference traces depending on
the replay. Event overlays are derived from `road_events` output. Point events
such as speed bumps and shocks appear at event timestamps; segment events such
as rough road and hills cover intervals. Hovering or selecting an event exposes
the trigger traces when they were captured in `PlotData`.

Map coloring can be tied to data channels such as speed or event state. Browser
maps use CARTO tiles by default and can use Mapbox when a token is supplied.

## Browser Configuration

The browser shell reads optional configuration from URL query parameters, local
storage, and `web/local-config.js`.

- `?theme=light` or `?theme=dark` selects the initial visual theme.
- `?mapbox_token=...` supplies and remembers a Mapbox token.
- `web/local-config.js` is for local development and is not required for the
  hosted visualizer.

## Replay Pipeline

The replay pipeline lives in `sim/src/visualizer`:

- `pipeline/generic.rs` parses generic CSV replay inputs and creates estimator
  traces.
- `pipeline/synthetic.rs` generates synthetic measurements and truth overlays.
- `replay_job.rs` owns replay request/config/result types and transport
  decimation.
- `model.rs` defines `PlotData`, trace containers, road-event samples, pages,
  and visualizer mount mode.
- `ui/` renders pages, maps, tuning windows, inspector panels, and browser-only
  dataset controls.

The browser and native shells differ mostly in file access and worker plumbing.
The estimator, road events, replay parsing, and plot construction stay in Rust.

Implementation ownership is deliberately narrow:

- `sim::datasets` decodes CSV rows and reference streams.
- `sim::visualizer::pipeline` orders samples, feeds the public `SensorFusion`
  facade, converts fused snapshots into `road_events::VehicleMotionSample`, and
  records derived traces.
- `sensor_fusion` owns EKF/align behavior. Tuning controls only change public
  `SensorFusion` configuration before replay.
- `road_events` owns detector state and trip statistics. The visualizer displays
  events and trigger channels; it does not reclassify events in the UI layer.
- `web/replay_worker.js` isolates replay execution from browser rendering and
  returns serialized `PlotData`.

## Practical Caveats

Browser transport decimates dense traces so large datasets stay responsive.
Use native replay or targeted trace export when investigating a very high-rate
detail. iOS datasets often lack external reference truth, so missing reference
traces usually mean the source recording did not contain them rather than a
failed replay.
