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

## Notes

- `project.yml` links `../build/SensorFusionFFI.xcframework` with XcodeGen's
  `dependencies.framework` syntax and sets header search paths for
  `sensor_fusion_ffi.h`.
- GNSS uses `CoreLocation`.
- IMU uses `CoreMotion` device motion updates at 50 Hz.
- UI is SwiftUI with a simple live status list.
