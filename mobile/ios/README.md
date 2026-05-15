# iOS App Scaffold

This directory contains a minimal iOS app scaffold for live IMU/GNSS display.

## Build the Rust FFI XCFramework

The app expects the Rust FFI crate at `mobile/ios/SensorFusionFFI`, with its C
header at `include/sensor_fusion_ffi.h`. The build script packages the Rust
`staticlib` as `mobile/ios/build/SensorFusionFFI.xcframework` and copies the
header to `mobile/ios/build/include/sensor_fusion_ffi.h`.

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

If the Rust static library target is not named `sensor_fusion_ffi`, pass the
library name without the `lib` prefix or `.a` suffix:

```bash
SENSOR_FUSION_FFI_LIB_NAME=my_ffi_name ./scripts/build_sensor_fusion_xcframework.sh
```

## Generate the Xcode project

```bash
cd mobile/ios/IMUGNSSPhone
xcodegen generate
```

## Build / Run

1. Open `mobile/ios/IMUGNSSPhone/IMUGNSSPhone.xcodeproj` in Xcode.
2. Select an iPhone (simulator has no real GNSS/IMU).
3. Run the app and grant Location + Motion permissions.

## Export `.motionfusion` Recordings

Use the stdlib-only exporter to inspect an iOS raw session recording and create
web visualizer inputs:

```bash
cd mobile/ios
python3 scripts/export_motionfusion.py ~/Downloads/session.motionfusion --output-dir /tmp/session-web
```

The output directory contains:

- `imu.csv` with `t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2`
- `gnss.csv` with the generic web replay GNSS columns
- `summary.txt` with counts, duration, IMU/GNSS rates, missing GNSS velocity
  rows, and accel/gyro magnitude statistics

Rows with explicit GNSS NED velocity are exported directly. Rows without
explicit NED velocity are counted in the summary; when possible, the exporter
derives north/east velocity from speed/course, otherwise it skips the row so
playback does not feed a synthetic zero-velocity update.

## Notes

- `project.yml` links `../build/SensorFusionFFI.xcframework` with XcodeGen's
  `dependencies.framework` syntax and sets header search paths for
  `sensor_fusion_ffi.h`.
- GNSS uses `CoreLocation`.
- IMU uses `CoreMotion` device motion updates at 50 Hz.
- UI is SwiftUI with a simple live status list.
