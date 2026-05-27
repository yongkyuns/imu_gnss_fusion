# iOS App

`mobile/ios` contains an iOS app and Rust FFI layer for collecting and reviewing vehicle-motion data on a phone.

```{figure} ../_static/diagrams/ios-app-operation.svg
:alt: iOS app operation flow from sensors and replay through SensorStore, Rust FFI, and app surfaces.
:class: framed

Live driving and replayed `.motionfusion` sessions pass through the same app-side store and Rust-backed fusion/event path before reaching Drive, Review, Settings, and Diagnostics views.
```

## App Surface

The SwiftUI app has Drive, Review, Settings, and developer Diagnostics surfaces. The Drive view includes a MapKit route display, live telemetry drawer, motion-event annotations, and heads-up event notifications. Review covers saved/completed drive sessions. Settings exposes harsh-behavior presets, event-audio behavior, mount-memory controls, and diagnostics visibility.

The app uses:

- CoreLocation for GNSS/location.
- CoreMotion device-motion updates for IMU-like motion input.
- `SensorFusionFFI` for Rust-backed fusion and `road_events` detectors.
- raw `.motionfusion` session logs for export and replay.

## Rust FFI

Build the XCFramework:

```bash
cd mobile/ios
./scripts/build_sensor_fusion_xcframework.sh
```

The FFI layer wraps `sensor_fusion` and `road_events`. It exposes fusion status, road-event detections, harsh behavior presets, and trip summaries. The Swift app layers resource and profiling models on top for the Diagnostics view.

Road-event FFI kinds are:

1. harsh acceleration
2. harsh braking
3. harsh cornering
4. reverse
5. speed bump
6. uphill
7. downhill
8. road shock
9. rough road

## Exporting Recordings

Export a `.motionfusion` recording to generic replay CSV:

```bash
cd mobile/ios
python3 scripts/export_motionfusion.py ~/Downloads/session.motionfusion --output-dir /tmp/session-web
```

Package an iOS recording for the hosted visualizer:

```bash
python3 scripts/package_ios_motionfusion_dataset.py \
  ~/Downloads/session.motionfusion \
  --dataset-id ios-new-drive \
  --output-root web/datasets \
  --web-manifest web/datasets/manifest.json
```

The exporter accepts raw accelerometer/gyro/specific-force recordings, writes `imu.csv`, `gnss.csv`, and a summary, and preserves the generic replay column conventions. Rows with explicit GNSS NED velocity are exported directly. Rows without NED velocity are derived from speed/course when possible; stationary speed can produce zero velocity; heading is cleared when course accuracy is worse than `45 deg`; GNSS rows are skipped when a usable velocity cannot be formed.
