# Road Events

`road_events` is a standalone `#![no_std]` crate of streaming detectors. It consumes vehicle-motion samples and emits compact events that can be used in embedded runtimes, the simulator, the browser visualizer, and the iOS FFI layer.

```{figure} ../_static/diagrams/road-events-pipeline-orthogonal.svg
:alt: Road events pipeline from vehicle motion samples through detector families to point events, interval events, trip stats, iOS, and visualizer consumers.
:class: framed

The detector crate stays streaming and constant-memory; UI layers consume point events, completed intervals, live rough-road notifications, and trip summaries.
```

## Detector Families

| Detector | Input | Output |
| --- | --- | --- |
| `SpeedBumpDetector` | speed, pitch, gravity-compensated vertical acceleration | speed-bump events and diagnostics |
| `RoadRoughnessAnalyzer` | speed, gravity-compensated vertical acceleration | roughness estimates, rough-road intervals, shock events |
| `HillDetector` | speed and pitch | uphill/downhill intervals |
| `ReverseDetector` | signed forward velocity | reverse-driving intervals |
| `HarshAccelDetector` | forward velocity | harsh acceleration intervals |
| `HarshBrakeDetector` | forward velocity | harsh braking intervals |
| `HarshCornerDetector` | speed and vehicle-frame lateral specific force | jerk-gated harsh cornering intervals |
| `TripStats` | vehicle-motion and event counters | constant-memory trip summaries |

Roughness and shock are intentionally separate. Roughness estimates ambient, distance-normalized vertical vibration after robust impulse limiting. Shock events capture short, impulse-like vertical hits such as potholes, sharp joints, or bumps so they do not dominate the ambient road-noise metric.

`RoadRoughnessAnalyzer::update_with_events` can emit both live rough-road notifications and completed rough-road intervals. The live event appears when sustained roughness is confirmed; the completed event appears when the active interval exits or is flushed. `RoadShockEvent` is separate from both.

## Harsh Behavior Presets

Harsh acceleration, braking, and cornering use preset configuration sets exposed through the Rust crate, web tooling, and iOS app settings:

- `Sensitive`
- `Balanced`
- `Conservative`

The balanced cornering thresholds currently use a corner load threshold of $3.4\,\mathrm{m/s^2}$, exit threshold of $2.9\,\mathrm{m/s^2}$, and jerk gate of $5.0\,\mathrm{m/s^3}$.

Harsh cornering is not based on $\dot{\psi}v$. It is based on jerk-gated lateral side-load from vehicle-frame specific force. The detector expects callers to feed bias-corrected vehicle-frame lateral specific force, which represents passenger side-load including bank effects.

## Hill Defaults

The default hill detector uses a $4.0^\circ$ pitch threshold and $1.0\,\mathrm{s}$ confirmation window.

## Integration

The iOS FFI layer uses `road_events` directly for road-event motion updates and trip summaries. The visualizer displays point events and segment events with event filters, map overlays, hover trigger plots, and Events-page plots.
