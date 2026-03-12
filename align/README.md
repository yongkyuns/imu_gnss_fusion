# Align

This crate now contains only the align filter implemented in [align.rs](./src/align.rs).

Model:
- State: installation quaternion `q_vb` (vehicle -> body/IMU)
- Error-state: 3D small-angle perturbation
- Inputs: `ESF-RAW` gyro and accelerometer, `NAV2-PVT` velocity
- Outputs: FRD Euler angles using intrinsic `Rx * Ry * Rz`

Measurements:
- stationary gravity
- planar turn gyro structure
- GNSS course-rate
- GNSS lateral acceleration
- GNSS longitudinal acceleration

Primary API:
- `AlignConfig`
- `AlignWindowSummary`
- `Align`

Python bindings are exposed when the `python` feature is enabled.
