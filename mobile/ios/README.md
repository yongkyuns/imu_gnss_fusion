# iOS App Scaffold

This directory contains a minimal iOS app scaffold for live IMU/GNSS display.

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

- GNSS uses `CoreLocation`.
- IMU uses `CoreMotion` device motion updates at 50 Hz.
- UI is SwiftUI with a simple live status list.
