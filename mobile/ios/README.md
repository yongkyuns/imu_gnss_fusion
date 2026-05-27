# iOS App

This directory contains the iOS vehicle-motion app, SwiftUI UI, Rust FFI wrapper, and `.motionfusion` export tools.

The app currently includes:

- Drive, Review, Settings, and developer Diagnostics tabs;
- MapKit route display with motion-event annotations;
- live fusion status, align progress, EKF snapshots, and trip summaries;
- Rust FFI integration for `sensor_fusion` and `road_events`;
- harsh behavior preset settings;
- event audio settings;
- mount-memory settings;
- raw `.motionfusion` recording, playback/export support, and resource/profiling diagnostics.

## Build The Rust FFI XCFramework

Install the iOS Rust targets:

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
```

Optional, for Intel simulator slices:

```bash
rustup target add x86_64-apple-ios
```

Build the XCFramework:

```bash
cd mobile/ios
./scripts/build_sensor_fusion_xcframework.sh
```

The script packages the Rust static library as `mobile/ios/build/SensorFusionFFI.xcframework` and copies `include/sensor_fusion_ffi.h` into `mobile/ios/build/include/`.

## Generate The Xcode Project

```bash
cd mobile/ios/IMUGNSSPhone
xcodegen generate
```

## Build / Run

1. Open `mobile/ios/IMUGNSSPhone/IMUGNSSPhone.xcodeproj` in Xcode.
2. Select a physical iPhone for real GNSS/IMU data.
3. Run the app and grant Location + Motion permissions.

The simulator can build UI code, but it does not provide realistic GNSS/IMU streams.

## Export `.motionfusion` Recordings

Use the stdlib-only exporter to inspect an iOS raw session recording and create generic replay inputs:

```bash
cd mobile/ios
python3 scripts/export_motionfusion.py ~/Downloads/session.motionfusion --output-dir /tmp/session-web
```

The output directory contains:

- `imu.csv` with `t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2`;
- `gnss.csv` with the generic replay GNSS columns;
- `summary.txt` with counts, duration, IMU/GNSS rates, missing GNSS velocity rows, and accel/gyro magnitude statistics.

Rows with explicit GNSS NED velocity are exported directly. Rows without NED velocity are derived from speed/course when possible; stationary speed can emit zero velocity; heading is cleared when course accuracy is worse than `45 deg`; GNSS rows are skipped when a usable velocity cannot be formed.

## Package A Hosted Dataset

```bash
python3 scripts/package_ios_motionfusion_dataset.py \
  target/ios-raw-sessions/<recording>.motionfusion \
  --dataset-id ios-new-drive \
  --title "iOS new drive" \
  --label "iOS new drive"
```

By default this writes the packaged dataset under `web/datasets/<dataset-id>/` and updates both:

- `web/datasets/manifest.json`;
- `.github/datasets/generic-datasets.json`.

Use `--no-update-web-manifest` or `--no-update-ci-manifest` for local-only packaging variants.
