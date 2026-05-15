# IMU/GNSS Fusion API and Coordinate Conventions

This document describes the public `sensor_fusion` crate API and the frame,
rotation, unit, initialization, and readiness conventions that callers must
preserve. It focuses on `SensorFusion`, the public sample types, mount handling,
and EKF readiness.

For derivation details, see the formulation PDFs in `docs/`.

## Public API Shape

The crate is `#![no_std]` and exposes these main entry points:

- `SensorFusion`: high-level streaming facade for IMU, GNSS, optional vehicle
  speed, mount alignment, local anchoring, and EKF state.
- `Config` and `MountMode`: construction-time mount handling.
- `ImuSample`, `GnssSample`, `VehicleSpeedSample`, `VehicleSpeedDirection`,
  and `Update`: public sample and status types.
- `ProcessNoise`: continuous process-noise configuration.
- `align`: standalone mount-alignment estimator and diagnostics.

For most applications, use `SensorFusion`. Standalone modules and generated
model wrappers are for focused estimator work, diagnostics, and tests.

## Frames

The public convention uses these frame labels:

- `b`: raw IMU body/sensor frame. `ImuSample` gyro and accelerometer vectors
  are expressed in this frame.
- `v`: vehicle frame. Axes are forward, right, down.
- `n`: local NED navigation frame. Axes are north, east, down.
- `e`: ECEF frame used for WGS84 conversion and global position math.

Public GNSS samples arrive as latitude, longitude, ellipsoidal height, and
local-NED velocity at the sample location. The facade converts those
measurements into the runtime basis internally.

## Quaternion and DCM Naming

The crate uses active rotations:

```text
C_ab maps coordinates from frame b to frame a
x_a = C_ab x_b
R(q_ab) = C_ab
R(q1 * q2) = R(q1) R(q2)
```

Quaternions are scalar-first arrays: `[w, x, y, z]`.

Important public quaternions:

- `q_bv`: physical vehicle-to-body mount. `R(q_bv) = C_bv`, so
  `x_b = C_bv x_v`.
- `q_vb`: inverse body-to-vehicle mount. It is the conjugate of `q_bv`, and
  `C_vb = C_bv^T`.

Propagation consumes raw IMU samples in body frame `b`. The EKF uses
`C_vb = C_bv^T` to rotate raw body-frame inertial measurements into the vehicle
frame.

## Units and Sample Types

Public APIs use SI units unless the field name says otherwise.

`ImuSample`:

- `t_s`: seconds.
- `gyro_radps`: angular rate in raw body frame `b`, radians per second.
- `accel_mps2`: specific force in raw body frame `b`, meters per second
  squared.

`GnssSample`:

- `t_s`: seconds.
- `lat_deg`, `lon_deg`: WGS84 geodetic latitude and longitude, degrees.
- `height_m`: ellipsoidal height, meters.
- `vel_ned_mps`: local NED velocity `[north, east, down]`, meters per second.
- `pos_std_m`: one-sigma position standard deviations for NED axes, meters.
- `vel_std_mps`: one-sigma velocity standard deviations for NED axes, meters
  per second.
- `heading_rad`: optional yaw/course heading in local NED, radians clockwise
  from north toward east.

`VehicleSpeedSample`:

- `t_s`: seconds.
- `speed_mps`: nonnegative speed magnitude, meters per second.
- `direction`: `Forward`, `Reverse`, or `Unknown`.

Covariance tuning setters that start with `set_r_...` take variances, not
standard deviations. Setters with `sigma` in the name take one-sigma standard
deviations.

## Construction

`SensorFusion::new()` is equivalent to `SensorFusion::with_config` using
`Config::default()`: automatic mount alignment.

Use `Config` when selecting mount behavior:

```rust
use sensor_fusion::{Config, MountMode, SensorFusion};

let mut auto_mount = SensorFusion::with_config(Config {
    mount_mode: MountMode::Auto,
});

let mut manual_mount = SensorFusion::with_config(Config {
    mount_mode: MountMode::Manual([1.0, 0.0, 0.0, 0.0]),
});
```

## Mount Modes

`MountMode::Auto`:

- The facade bootstraps internal `align::Align` from low-dynamic IMU samples.
- Alignment estimates the physical `q_bv`.
- Handoff requires the align trace's `coarse_alignment_ready` flag and any
  configured handoff delay.
- After handoff, the EKF receives the mount seed and can estimate residual
  mount states unless freezing is configured.

`MountMode::Manual(q_bv)` and `SensorFusion::with_mount(q_bv)`:

- The supplied quaternion is the physical vehicle-to-body mount `q_bv`.
- `mount_ready` is true immediately.
- Internal alignment is disabled.
- Mount states are frozen.

Do not pass a body-to-vehicle quaternion to manual APIs. If the mount you have
maps body vectors into vehicle vectors, conjugate it before passing it as
`q_bv`.

## Typical Facade Call Sequence

1. Construct the facade with `new`, `with_mount`, `with_mount_mode`, or
   `with_config`.
2. Apply tuning before replay or live streaming.
3. Feed timestamped samples in time order:
   - `process_imu(ImuSample)` at IMU rate.
   - `process_gnss(GnssSample)` at GNSS rate.
   - `process_vehicle_speed(VehicleSpeedSample)` when available.
4. Inspect each returned `Update`.
5. Read state through public accessors such as `mount_q_bv()`,
   `position_lla_f64()`, and diagnostics after readiness has been reached.

Samples should be monotonic in timestamp. Invalid, duplicate, or very large
intervals are rejected before EKF propagation.

## Readiness and Accessors

Every `process_*` method returns `Update`:

- `mount_ready`: a physical mount estimate is available.
- `mount_ready_changed`: this input changed `mount_ready`.
- `ekf_initialized`: the EKF has been initialized from GNSS.
- `ekf_initialized_now`: EKF initialization happened during this input.
- `mount_q_bv`: current physical mount when available.

State accessors are readiness-gated. Do not read position, velocity, attitude,
or diagnostics until the corresponding readiness flag is true.

EKF initialization requires `mount_ready` plus a GNSS sample. It seeds yaw from
`heading_rad` when present; otherwise it uses GNSS course if horizontal speed is
high enough.

## NHC, Zero-Velocity, and Vehicle Speed Updates

The nonholonomic constraint (NHC) constrains vehicle-frame lateral and vertical
velocity (`v_y` and `v_z`) toward zero.

Facade NHC behavior:

- cadence is controlled by `set_nhc_update_period_s`,
- lateral and vertical variances are controlled by `set_r_body_vel_yz`,
- low-speed windows can prefer zero-velocity or stationary updates,
- vehicle speed samples constrain vehicle-frame forward/reverse speed when the
  EKF and mount are ready.

## Integration Pitfalls

- Passing body-to-vehicle mount quaternions to APIs that require `q_bv`.
- Pre-rotating IMU samples before calling `process_imu`.
- Treating covariance `r_*` values as standard deviations.
- Feeding non-monotonic timestamps.
- Reading EKF state before `Update::ekf_initialized`.
- Assuming optional reference CSVs are runtime inputs. They are for
  visualization, evaluation, and manual mount seeding only.
