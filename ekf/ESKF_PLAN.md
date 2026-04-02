# ESKF Migration Plan

Goal:
- replace the current direct-state quaternion EKF with a nominal-state plus error-state ESKF
- preserve the public `SensorFusion` API shape while swapping the internal estimator architecture
- keep the same implementation flow as the legacy EKF:
  - symbolic Python derivation as source of truth
  - generated C artifacts for embedded/runtime use
  - Rust only wraps the C implementation

Target nominal state:
- quaternion `q_bn`
- velocity `v_n`
- position `p_n`
- gyro bias `b_g` in `rad/s`
- accel bias `b_a` in `m/s^2`

Target error state:
- small attitude error `dtheta`
- velocity error `dv`
- position error `dp`
- gyro-bias error `dbg`
- accel-bias error `dba`

Planned migration order:
1. Add ESKF nominal/error-state primitives and unit tests.
2. Add a parallel SymPy derivation for ESKF covariance and measurement Jacobians.
3. Emit parallel generated C artifacts from `ekf/eskf.py` into `ekf/c/generated_eskf/`.
4. Implement C-side ESKF core beside the current EKF and consume those generated files.
5. Add Rust FFI/glue only after the C ESKF path is stable.
4. Add replay parity and regression tests against baseline logs.
5. Switch `SensorFusion` runtime to the ESKF backend only after acceptance gates pass.

Acceptance gates:
- baseline convergence and alignment behavior remain acceptable on `test_files.txt`
- no regression in coarse-ready and EKF init timing unless intentionally changed
- ESP32-S3 replay remains functional and timing stays within budget
