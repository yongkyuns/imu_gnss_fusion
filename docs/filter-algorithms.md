# EKF Algorithms

This page summarizes the runtime estimator used by the `sensor_fusion` crate.
It is intentionally implementation-facing.

## Runtime Shape

`SensorFusion` owns the streaming runtime for ground-vehicle IMU/GNSS fusion:

- `Align` estimates the physical vehicle-to-body mount quaternion, `q_bv`, when
  automatic mount mode is enabled.
- The EKF propagates raw body-frame IMU samples through the current mount and
  vehicle attitude.
- GNSS position, GNSS velocity, nonholonomic constraints, optional vehicle
  speed, and stationary cues update the EKF when their gates pass.
- The facade exposes readiness through `Update::mount_ready`,
  `Update::ekf_initialized`, and `Update::ekf_initialized_now`.

The public API uses active rotations. `C_ab` maps coordinates from frame `b` to
frame `a`:

```text
x_a = C_ab x_b
R(q_ab) = C_ab
R(q1 * q2) = R(q1) R(q2)
```

Quaternions are scalar-first `[w, x, y, z]`.

| Symbol | Meaning |
| --- | --- |
| `b` | Raw IMU body/sensor frame. |
| `v` | Vehicle frame, forward-right-down. |
| `n` | Local NED navigation frame. |
| `e` | ECEF frame used for geodetic conversion and global position math. |

The mount quaternion is the physical vehicle-to-body rotation:

```text
x_b = C_bv x_v
C_vb = C_bv^T
x_v = C_vb x_b
```

Raw IMU samples are not pre-rotated by callers. The runtime applies `C_vb`
internally before propagation.

## State

The EKF tracks the vehicle navigation state, IMU calibration terms, and mount
correction states needed for ground-vehicle fusion:

| Block | Purpose |
| --- | --- |
| Attitude | Vehicle attitude relative to the navigation basis. |
| Velocity | Vehicle/platform velocity in navigation coordinates. |
| Position | Local or geodetic navigation position managed by the facade. |
| Gyro bias | Slowly varying gyroscope bias. |
| Accelerometer bias | Slowly varying accelerometer bias. |
| Mount | Residual mount correction around the physical `q_bv` seed. |

The public nominal-state fields follow the crate layout, but callers should
prefer facade accessors such as `mount_q_bv()` and `position_lla_f64()` where
available.

## Initialization

Automatic mount mode waits for enough stationary and GNSS-backed vehicle motion
to form a usable mount seed. Manual mount mode accepts a caller-supplied `q_bv`
and marks the mount ready immediately.

EKF initialization requires:

- a ready mount,
- a GNSS position sample,
- enough horizontal motion or heading information to seed yaw.

Initialization covariance is configured through the facade setters for attitude,
yaw, bias, and mount uncertainty. Keep those values broad enough to cover real
mount and heading uncertainty; overly tight priors can delay convergence.

## Prediction

Each accepted IMU interval performs:

1. Bias correction.
2. Body-to-vehicle rotation through `C_vb`.
3. Attitude, velocity, and position propagation.
4. Linearized covariance propagation using generated Jacobian code.
5. Quaternion normalization and covariance symmetrization.

The facade rejects invalid or extreme `dt` values before propagation. This
protects the EKF from startup duplicate timestamps, dropped batches, and large
replay gaps.

## Measurements

The runtime can fuse these update families:

| Update | Measurement |
| --- | --- |
| GNSS position | WGS84 latitude, longitude, height converted to the runtime basis. |
| GNSS velocity | Local NED velocity converted as needed for the runtime basis. |
| Heading/course | Optional GNSS heading or course-derived yaw during initialization. |
| NHC | Vehicle-frame lateral and vertical velocity constrained toward zero. |
| Vehicle speed | Forward/reverse speed along vehicle `+X` when available. |
| Zero velocity | Low-dynamic vehicle-frame velocity near zero. |
| Stationary gravity | Body/vehicle accelerometer channels compared to gravity. |

Measurement variances come from sample standard deviations where provided and
from facade tuning setters for NHC, vehicle speed, zero velocity, and stationary
gravity updates.

## NHC Scheduling

The default NHC cadence is 10 Hz. The facade schedules NHC updates from IMU
timestamps and may batch compatible rows with a nearby GNSS update. This keeps
the embedded compute budget predictable while preserving the key lateral and
vertical constraints for ground vehicles.

NHC is gated by speed and freshness:

- low-speed or stationary windows prefer zero-velocity/stationary updates,
- moving windows apply lateral and vertical body-velocity rows,
- outage behavior depends on the current EKF speed estimate and configured
  GNSS outage simulation.

## Mount Handling

`MountMode::Auto` uses `Align` to seed `q_bv`; the EKF can then estimate
residual mount states. `MountMode::Manual(q_bv)` uses the supplied physical
mount and freezes mount states.

Mount freezing is stronger than setting process noise to zero. It also zeros
the relevant covariance rows/columns and correction components so measurement
updates cannot silently move frozen mount states.

## Generated Model Code

Generated Rust snippets live under the EKF implementation modules and are
checked in so normal builds do not require Python or SymPy. Regenerate them only
after changing the matching symbolic formulation source:

```bash
python sensor_fusion/src/ekf/formulation.py --emit-rust
```

After regeneration, review the generated diff and run the focused tests listed
in [testing.md](testing.md).

## Tuning

Primary runtime tuning controls:

| Control | Effect |
| --- | --- |
| `set_process_noise` or EKF noise setters | Continuous IMU, bias, and mount process noise. |
| `set_r_body_vel_yz` | NHC lateral and vertical velocity variances. |
| `set_nhc_update_period_s` | NHC cadence. |
| Initialization covariance setters | Initial attitude, yaw, bias, and mount uncertainty. |
| `set_freeze_misalignment_states` | Freeze or release mount correction states. |
| `set_align_config` | Stationary, acceleration, turn-rate, and handoff behavior for mount alignment. |

Tune from physical sensor specifications first, then use replay diagnostics to
adjust measurement weights. Avoid dataset-specific gains unless the deployment
really has dataset-specific sensor behavior.

## Verification

Use focused tests for estimator changes:

```bash
cargo test -p sensor_fusion --locked
cargo test -p sensor_fusion --test fusion_api --locked
cargo test -p sensor_fusion --test align_api --locked
cargo test -p sim --test synthetic_gnss_ins --locked
```

For generated-model changes, also run the model-specific parity tests and a
visualizer smoke replay for the data path you changed.
