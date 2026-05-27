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

```{figure} _static/diagrams/overall-architecture.png
:alt: Overall project architecture showing data inputs, replay tooling, reusable Rust runtimes, visualizers, iOS app, docs, and CI.
:class: framed

Data capture, replay, reusable Rust runtimes, visualizers, mobile integration, docs, and CI are intentionally separated so estimator behavior can be reused across every surface.
```

```{figure} _static/screenshots/web-visualizer-overview.png
:alt: Browser visualizer showing replay plots and a map trace.
:class: framed

The browser visualizer is the fastest way to inspect replay outputs, map traces, mount behavior, diagnostics, and event detector signals.
```

```{toctree}
:maxdepth: 2
:caption: Start

quick-start
architecture
api-and-conventions
```

```{toctree}
:maxdepth: 2
:caption: Algorithms

filter-algorithms
algorithms/observability
algorithms/road-events
ekf-diagnostics
```

```{toctree}
:maxdepth: 2
:caption: Data And Tools

data-and-simulation
data/hosted-datasets
tools/visualizer
visualizer-tools-testing
mobile/ios
```

```{toctree}
:maxdepth: 2
:caption: Development

testing
development/ci
development/embedded-performance
reference/workspace
```

```{toctree}
:maxdepth: 2
:caption: Math Notes

math/frames
math/ekf
math/align
math/runtime-ekf
math/roll-observability
math/road-events
math/notes
```
