# Visualizer And Simulator

The visualizer is a replay and simulation environment for the fusion runtime,
not only a plotting frontend. Each run feeds recorded or synthesized IMU/GNSS
samples through the selected fusion backend, derives road-event samples from
the estimated vehicle motion, then plots the resulting states, diagnostics, map
traces, and events. The browser version runs the wasm-compiled Rust visualizer
and fusion code.

Use synthetic runs to study theoretical properties of the filter under
controlled motion, and use experimental runs to evaluate performance on real
sensor data.

```{figure} ../_static/screenshots/web-visualizer-overview.png
:alt: Web visualizer overview with plots and map.
:class: framed

Overview page for a replay with speed, mount, attitude, bias, and map panels.
```

```{figure} ../_static/gifs/web-visualizer-workflow.gif
:alt: Browser workflow loading a hosted replay and inspecting map and plots.
:class: framed

Hosted replay workflow: load a dataset, run replay, inspect map and trace
panels, then open Events and zoom/pan the timeline around detected activity.
```

```{figure} ../_static/gifs/web-visualizer-tabs.gif
:alt: Browser visualizer tabs for synthetic replay inspection.
:class: framed

Synthetic and experimental runs share the same page structure, so controlled
filter behavior and real-data behavior can be compared with the same plots.
```

## Choosing A Run Type

Synthetic runs are for filter reasoning. They are useful when checking:

- whether a motion pattern should make mount, attitude, velocity, or bias
  observable;
- how align and EKF behave when the ground-truth trajectory is known;
- how noise level, mount angle, acceleration, turns, stops, and GNSS outages
  affect convergence;
- whether a proposed model change fixes the intended theoretical case without
  relying on dataset-specific behavior.

Experimental runs are for actual performance. They are useful when checking:

- whether the filter remains stable on real IMU/GNSS timing, noise, and outages;
- whether mount, bias, position, velocity, and attitude estimates are consistent
  across trips;
- whether road-event detectors match observed bumps, roughness, hills, reverse,
  harsh acceleration/braking, and cornering;
- whether iOS-exported recordings and hosted datasets replay correctly.

The distinction matters: synthetic data can show that a formulation is
internally plausible, but real recordings determine whether the implementation
is robust to sensor quality, mounting, road geometry, and driving behavior.

## Browser Workflow

Open the hosted visualizer:

```text
https://yongkyuns.github.io/imu_gnss_fusion/
```

For synthetic analysis:

1. Select `Synthetic`.
2. Choose a scenario and noise preset.
3. Press `Run`.
4. Inspect Overview first, then Motion, Mount, Calibration, and Diagnostics.

For experimental analysis:

1. Select `Experimental/real data`.
2. Choose a hosted dataset, or drop generic replay CSV files.
3. Press `Run`.
4. Inspect Overview and Map first, then use Sensors to check input quality,
   Mount/Calibration for estimator consistency, and Events for road-event
   detector behavior.

Browser replay accepts hosted datasets listed in `web/datasets/manifest.json`.
The current hosted set contains 36 datasets. Drag/drop replay accepts plain-text
generic CSV files; at minimum provide `imu.csv` and `gnss.csv`. Optional
reference files add attitude, mount, position, or vehicle-motion traces.

## Native Workflow

Run a generic replay directory:

```bash
cargo run --release -p fusion_tools --bin visualizer -- \
  --generic-replay-dir /path/to/replay-dir
```

Run a synthetic scenario:

```bash
cargo run --release -p fusion_tools --bin visualizer -- \
  --synthetic-motion-def tools/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low
```

Use native replay when browser decimation or browser memory limits hide a
high-rate detail.

Native replay can also select the standalone C fusion implementation:

```bash
cargo run --release -p fusion_tools --bin visualizer -- \
  --generic-replay-dir /path/to/replay-dir \
  --backend c
```

The native C backend fills the core navigation, mount, bias, covariance, map,
roughness, bump-diagnostic, road-event, and trip-statistic views. Rust-only EKF
update inspector traces, NHC correction internals, and align debug-window
traces are not exposed yet, so those diagnostic groups remain empty when `C` is
selected.

## Main Controls

| Control | Use |
| --- | --- |
| `Theme` | Switch light/dark rendering. |
| `Traces` | Show or hide Reference, Align, and EKF traces across plots. |
| `Map` | Show or hide GNSS/reference paths, heading arrows, and event overlays. |
| `Filter` | Choose which road-event kinds are visible. |
| `Tune` | Select the native runtime backend and adjust EKF, align, or road-event parameters for the next run. |
| `Inspector` | Show update-level residual allocation near the hovered plot time. |

Changing tabs, trace visibility, map overlays, event filters, map color, hover,
or inspector state does not rerun the filter. Changing input data, synthetic
scenario, noise, mount mode, GNSS-outage settings, or tuning parameters requires
pressing `Run` to create a new replay result.

## Page Guide

| Page | Use |
| --- | --- |
| `Overview` | First-pass check for speed, mount, attitude, bias, and map consistency. |
| `Motion` | Inspect velocity, acceleration, attitude error, and route motion. |
| `Mount` | Inspect mount roll/pitch/yaw estimates, reference comparison, and mount uncertainty. |
| `Calibration` | Inspect gyro bias, accelerometer bias, mount states, and covariance-like traces. |
| `Sensors` | Check raw/calibrated IMU, GNSS measurements, timing, and source quality. |
| `Events` | Inspect road-event detector signals, trip summary, event markers, and trigger traces. |
| `Diagnostics` | Inspect align internals, EKF correction traces, and update-inspector context. |

## Map And Hover

Real-data runs use a geographic map when latitude/longitude are available.
Synthetic runs without geographic coordinates use a plot-style trajectory view.
CARTO tiles are used by default; Mapbox is optional when a token is supplied.

The map can show EKF, GNSS, synthetic truth, and reference paths depending on
the dataset. The map color selector can color the EKF path by speed,
longitudinal acceleration, lateral acceleration, road roughness, or vehicle
pitch. Hovering plots and map traces synchronizes the cursor time across visible
panels.

Road events appear as point markers or route segments. Point events include
speed bumps and shocks. Segment events include rough road, hills, reverse, and
harsh maneuvers. When trigger traces are available, hover cards show the local
signals that caused the event.

## What Happens On Run

When `Run` is pressed, the visualizer builds a replay job from the selected
input and current tuning values. In the browser, that replay job is executed by
the wasm-compiled Rust visualizer/fusion code. In the native app, the job can
execute either the Rust fusion backend or the standalone C fusion backend. The
replay path feeds each recorded or synthesized sample through the selected
fusion backend and road-event sample pipeline, then returns one render product
containing time-series traces, map traces, event outputs, trip statistics, and
diagnostic samples. The UI only displays and filters that render product; it
does not change estimator state after replay finishes.

Browser replay runs this job in a web worker so the page can remain responsive.
Dense browser traces are decimated for transport and rendering. Native replay is
the better tool when every high-rate sample matters.

## Useful URLs

- `?theme=light` or `?theme=dark` selects the initial visual theme.
- `?dataset=<id>` auto-loads a hosted dataset after the manifest is fetched.
- `?scenario=<name>` and `?noise=<level>` initialize synthetic controls.
- `?mapbox_token=...` supplies and remembers a Mapbox token.
- `?bench=1` enables browser-frame timing diagnostics.
