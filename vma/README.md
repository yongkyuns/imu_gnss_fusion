# VMA (Nonlinear Misalignment EKF)

This crate now implements a stripped-down, nonlinear misalignment estimator inspired by velocity matching alignment:

- State: installation quaternion `q_sb` (body -> sensor)
- Error-state EKF: 3D small rotation (`delta_theta`)
- No bias states (bias handling is left to full navigation EKF)

## Model Summary

1. `vma_predict(...)`
- Buffers IMU specific force in sensor frame and associated body attitude `q_nb`.
- Applies process noise to `P` for slow misalignment drift.

2. `vma_fuse_velocity(...)`
- Uses GNSS NED velocity as observation.
- Predicts velocity over buffered IMU window from anchor GNSS velocity.
- Builds Jacobian numerically w.r.t. `delta_theta`.
- EKF update on 3-state error and injects correction into `q_sb`.

## Public API

- `Vma`
- `MisalignNoise`
- `MisalignImuSample`
- `MisalignAttitudeSample`
- `vma_init`
- `vma_set_noise`
- `vma_set_q_sb`
- `vma_q_sb`
- `vma_predict`
- `vma_fuse_velocity`
- `vma_reset_window`

## Notes

- This is an alignment-only filter, not a full navigation EKF.
- Yaw observability still depends on vehicle excitation.
- Keep rotation convention consistent:
  - `q_sb`: body -> sensor
  - sensor -> body uses inverse/transpose.
