# IMU/GNSS Fusion

```{raw} html
<style>
#furo-main-content > section > h1:first-of-type {
  display: none;
}
</style>
```

```{image} _static/titlebar.png
:alt: IMU/GNSS Fusion
:class: titlebar
```

IMU/GNSS Fusion is a Rust workspace for ground-vehicle inertial/GNSS navigation experiments. It combines an embedded-oriented `sensor_fusion` runtime, replay and simulation tools, a browser visualizer, a `road_events` detector crate, and an iOS data-collection app.

The public GitHub Pages site has two entry points:

- [interactive visualizer](https://yongkyuns.github.io/imu_gnss_fusion/): the wasm web app at the site root.
- documentation: this Sphinx/Furo site under `/docs/`.

The core runtime accepts timestamped raw IMU samples in the sensor/body frame, GNSS position and NED velocity samples, and optional vehicle-speed observations. It can run with a caller-supplied vehicle-to-body mount quaternion or estimate the mount internally before EKF initialization.

```{figure} _static/diagrams/overall-architecture-orthogonal.svg
:alt: Overall project architecture showing data inputs, replay tooling, reusable Rust runtimes, visualizers, and iOS app.
:class: framed

Data capture, replay, reusable Rust runtimes, visualizers, and mobile integration are intentionally separated so estimator behavior can be reused across every surface.
```

```{figure} _static/screenshots/web-visualizer-overview.png
:alt: Browser visualizer showing replay plots and a map trace.
:class: framed

The browser visualizer is the fastest way to inspect replay outputs, map traces, mount behavior, diagnostics, and event detector signals.
```

```{figure} _static/gifs/web-visualizer-workflow.gif
:alt: Hosted replay workflow in the browser visualizer.
:class: framed

Hosted recordings can be replayed directly in the browser. The workflow shows a
selected iOS drive, the replay result, map trace inspection, hover/cursor
interaction, and the analysis tabs used to inspect motion, mount, sensors,
events, and EKF diagnostics.
```

```{toctree}
:maxdepth: 2
:caption: Start

quick-start
architecture
codebase-guide
```

```{toctree}
:maxdepth: 2
:caption: Filter Algorithms

filter-algorithms
api-and-conventions
algorithms/frames
algorithms/ekf
algorithms/runtime-ekf
algorithms/ekf-matrices
algorithms/mount-states
algorithms/align
algorithms/observability
algorithms/roll-observability
reference/prior-work
runtime-state-and-persistence
ekf-diagnostics
```

```{toctree}
:maxdepth: 2
:caption: Visualizer

tools/visualizer
```

```{toctree}
:maxdepth: 2
:caption: Mobile

mobile/ios
```

```{toctree}
:maxdepth: 2
:caption: Road Events

algorithms/road-events
algorithms/road-events-formulation
```

```{toctree}
:maxdepth: 2
:caption: Data And Tools

data-and-simulation
data/hosted-datasets
```

```{toctree}
:maxdepth: 2
:caption: Developer Reference

testing
development/generated-models
development/ci
development/datasets
development/embedded-performance
reference/workspace
```
