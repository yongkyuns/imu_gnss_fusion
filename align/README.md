# VMA (Nonlinear Align EKF)

This crate now implements a stripped-down, nonlinear align estimator inspired by velocity matching alignment:

- State: installation quaternion `q_sb` (body -> sensor)
- Error-state EKF: 3D small rotation (`delta_theta`)
- No bias states (bias handling is left to full navigation EKF)

## Model Summary

1. `align_predict(...)`
- Buffers IMU specific force in sensor frame and associated body attitude `q_nb`.
- Applies process noise to `P` for slow align drift.

2. `align_fuse_velocity(...)`
- Uses GNSS NED velocity as observation.
- Predicts velocity over buffered IMU window from anchor GNSS velocity.
- Builds Jacobian numerically w.r.t. `delta_theta`.
- EKF update on 3-state error and injects correction into `q_sb`.

## Public API

- `Align`
- `MisalignNoise`
- `MisalignImuSample`
- `MisalignAttitudeSample`
- `align_init`
- `align_set_noise`
- `align_set_q_sb`
- `align_q_sb`
- `align_predict`
- `align_fuse_velocity`
- `align_reset_window`

## Notes

- This is an alignment-only filter, not a full navigation EKF.
- Yaw observability still depends on vehicle excitation.
- Keep rotation convention consistent:
  - `q_sb`: body -> sensor
  - sensor -> body uses inverse/transpose.
