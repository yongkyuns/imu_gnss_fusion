# Sensor Fusion API and Coordinate Conventions

This document describes the public `sensor_fusion` crate API and the frame,
rotation, unit, initialization, and readiness conventions that callers must
preserve. It is written against the crate API exposed from
`sensor_fusion/src/lib.rs`, especially the `SensorFusion` facade plus the
standalone `align`, `reduced`, and `full` modules.

For derivation details, see `docs/align.pdf`, `docs/reduced.pdf`, and
`docs/full.pdf`. This file focuses on the runtime contract that integration
code should rely on.

## Public API Shape

The crate is `#![no_std]` and exposes these main entry points:

- `SensorFusion`: high-level streaming facade for IMU, GNSS, optional vehicle
  speed, mount alignment, local anchoring, and runtime filter selection.
- `Config`, `Filter`, and `MountMode`: construction-time selection of Reduced
  versus Full and automatic versus manual mount handling.
- `ImuSample`, `GnssSample`, `VehicleSpeedSample`, `VehicleSpeedDirection`,
  and `Update`: public sample and status types for the facade.
- `ProcessNoise`: shared continuous process-noise configuration.
- `align`: standalone mount-alignment filter and diagnostics.
- `reduced`: local-NED EKF runtime, public state layout, generated-model
  wrappers, diagnostics, and state-operation helpers.
- `full`: ECEF EKF runtime, public state layout, generated-model wrappers, and
  diagnostics.

For most applications, use `SensorFusion`. The standalone `align`, `reduced`,
and `full` modules are useful for focused filter work, diagnostics, generated
model comparisons, and tests.

## Frames

The public convention uses these frame labels:

- `b`: raw IMU body/sensor frame. `ImuSample` gyro and accelerometer vectors
  are expressed in this frame.
- `v`: vehicle frame. Axes are forward, right, down: `+X` forward, `+Y` right,
  `+Z` down.
- `n`: local NED navigation frame. Axes are north, east, down. Reduced uses
  this frame for velocity and position.
- `e`: ECEF frame. Full uses this frame for velocity and position.

Reduced has a current local geodetic anchor. Public GNSS samples arrive as
latitude, longitude, ellipsoidal height, and local-NED velocity at the sample
location. The facade converts those measurements into the current anchor-NED
frame for Reduced. If the current Reduced position drifts more than 5000 m
horizontally from the anchor, the facade reanchors and rotates the Reduced
navigation state into the new local frame.

Full uses ECEF position and velocity internally. Through `SensorFusion`, GNSS
latitude/longitude/height is converted to ECEF position and `vel_ned_mps` is
converted to ECEF velocity at the GNSS sample location.

## Quaternion and DCM Naming

The crate uses active rotations:

```text
C_ab maps coordinates from frame b to frame a
x_a = C_ab x_b
R(q_ab) = C_ab
R(q1 * q2) = R(q1) R(q2)
```

Quaternions are scalar-first arrays:

```text
[w, x, y, z]
```

Important public quaternions:

- `q_bv`: physical vehicle-to-body mount. `R(q_bv) = C_bv`, so
  `x_b = C_bv x_v`.
- `q_vb`: inverse body-to-vehicle mount. It is the conjugate of `q_bv`, and
  `C_vb = C_bv^T`.
- Reduced `q0..q3`: `q_nv`, the navigation-frame attitude with respect to the
  vehicle frame. `R(q_nv) = C_nv`, so `x_n = C_nv x_v`.
- Full `q0..q3`: `q_ev`, the ECEF-frame attitude with respect to the vehicle
  frame. `R(q_ev) = C_ev`, so `x_e = C_ev x_v`.
- Reduced and Full `q_bv0..q_bv3`: physical mount quaternion components. These
  fields store `q_bv`, not its inverse.

Propagation consumes raw IMU samples in body frame `b`. Both Reduced and Full
use `C_vb = C_bv^T` to rotate raw body-frame inertial measurements into the
vehicle frame.

The Reduced and Full attitudes describe the same physical vehicle attitude in
different navigation bases. At a given latitude/longitude:

```text
C_ev = C_en C_nv
C_en = C_ne^T
```

where `C_ne` maps ECEF vectors into local NED vectors.

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
deviations. `ProcessNoise` fields are continuous process-noise variances or
random-walk variances; the filters discretize them internally using the IMU
time step.

## Construction and Filter Selection

`SensorFusion::new()` is equivalent to `SensorFusion::with_config` using
`Config::default()`: Reduced filter, automatic mount alignment.

Use `Config` when selecting the runtime filter:

```rust
use sensor_fusion::{Config, Filter, MountMode, SensorFusion};

let mut reduced = SensorFusion::with_config(Config {
    filter: Filter::Reduced,
    mount_mode: MountMode::Auto,
});

let mut full = SensorFusion::with_config(Config {
    filter: Filter::Full,
    mount_mode: MountMode::Manual([1.0, 0.0, 0.0, 0.0]),
});
```

`Filter::Reduced` selects the local-NED EKF. `Filter::Full` selects the ECEF
EKF. The selected filter controls `Update::filter_initialized`,
`Update::filter_initialized_now`, and which state accessor is expected to
return `Some`.

`SensorFusion` still owns a Reduced instance even when `Filter::Full` is
selected, but the facade initializes and reports the selected runtime filter.
For a Full facade, `system.full()` becomes available after Full initialization;
`system.reduced()` is not the selected runtime state.

## Mount Modes

`MountMode::Auto`:

- The facade bootstraps internal `align::Align` from low-dynamic IMU samples.
- Alignment estimates the physical `q_bv`.
- Handoff requires the align trace's `coarse_alignment_ready` flag and any
  configured `set_align_handoff_delay_s`.
- After handoff, the selected runtime filter receives the mount seed and can
  estimate residual mount states unless freezing is configured.

`MountMode::Manual(q_bv)` and `SensorFusion::with_mount(q_bv)`:

- The supplied quaternion is the physical vehicle-to-body mount `q_bv`.
- `mount_ready` is true immediately.
- Internal alignment is disabled.
- Reduced and Full mount states are frozen.

`SensorFusion::set_misalignment(q_bv)` switches an existing facade to a fixed
manual mount. It sets the current public mount to `q_bv`, disables internal
alignment, freezes mount states, and applies the same mount to both underlying
filters.

Do not pass a body-to-vehicle quaternion to the manual APIs. If the mount you
have maps body vectors into vehicle vectors, conjugate it before passing it as
`q_bv`.

## Typical Facade Call Sequence

1. Construct the facade with `new`, `with_mount`, `with_mount_mode`, or
   `with_config`.
2. Apply tuning before replay or live streaming. Examples:
   `set_reduced_noise`, `set_full_noise`, `set_full_init_config`,
   `set_align_config`, `set_r_body_vel_yz`, `set_nhc_update_period_s`, and
   initialization covariance setters.
3. Feed timestamped samples in time order:
   - `process_imu(ImuSample)` at IMU rate.
   - `process_gnss(GnssSample)` at GNSS rate.
   - `process_vehicle_speed(VehicleSpeedSample)` when available.
4. Inspect each returned `Update`.
5. Read state through `reduced()`, `full()`, `mount_q_bv()`,
   `position_lla_f64()`, and diagnostics after readiness has been reached.

Samples should be monotonic in timestamp. Reduced prediction only runs for IMU
intervals in roughly `[0.001, 0.05]` seconds. Full prediction accepts positive
intervals up to 1 second through the facade. The first IMU sample establishes
the previous timestamp and does not propagate a filter.

For Reduced, GNSS updates after initialization are queued by `process_gnss` and
normally fused at the next eligible IMU epoch. If the GNSS sample is within
0.05 seconds of that IMU epoch, eligible NHC rows can be batched with the GNSS
rows. For Full, recent GNSS is batched during IMU processing when the GNSS age
is in `[0, 0.05]` seconds and the sample has not already been fused.

## Readiness and Accessors

Every `process_*` method returns `Update`:

- `mount_ready`: a physical mount estimate is available.
- `mount_ready_changed`: this input changed `mount_ready`.
- `reduced_initialized`: Reduced has been initialized from GNSS.
- `reduced_initialized_now`: Reduced initialized on this input, only meaningful
  for a Reduced runtime facade.
- `filter_initialized`: the selected runtime filter has been initialized.
- `filter_initialized_now`: the selected runtime filter initialized on this
  input.
- `mount_q_bv`: current physical mount when available.

State accessors are readiness-gated:

- `reduced()` returns `Some(&reduced::State)` only after Reduced
  initialization.
- `full()` returns `Some(&full::State)` only after Full initialization.
- `position_lla()` and `position_lla_f64()` use the Reduced state and local
  anchor, so they require Reduced initialization and a valid anchor.
- `align()` returns the internal align filter after stationary bootstrap.
- `align_debug()` returns the latest alignment window and trace only after a
  window update has been captured.

Reduced initialization requires `mount_ready` plus a GNSS sample. It seeds yaw
from `heading_rad` when present; otherwise it uses GNSS course if horizontal
speed is high enough, falling back to zero yaw at low speed.

Full initialization through `SensorFusion` requires `mount_ready` plus GNSS
horizontal speed of at least 0.5 m/s. If the GNSS sample is stationary, Full
does not initialize and `filter_initialized_now` remains false.

## NHC, Zero-Velocity, and Vehicle Speed Updates

The nonholonomic constraint (NHC) constrains vehicle-frame lateral and vertical
velocity (`v_y` and `v_z`) toward zero.

Facade NHC behavior:

- `set_r_body_vel(r)` sets both lateral and vertical NHC variances.
- `set_r_body_vel_yz(r_y, r_z)` sets them separately.
- `set_nhc_update_period_s(0.0)` enables an NHC attempt at every eligible IMU
  epoch.
- Positive `set_nhc_update_period_s(period)` decimates NHC updates and scales
  the observation variance by the elapsed NHC interval, preserving an
  approximate information rate. The runtime default is 0.1 s, or 10 Hz.
- NHC is gated out near zero speed, high gyro norm, or acceleration norm far
  from gravity.

Reduced uses the estimated Reduced speed for outage-time NHC gating. Full uses
the estimated Full speed through the facade. Standalone Full batch APIs can
also take an explicit `nhc_gate_speed_mps`; `None` does not always mean "use
state speed" for every standalone method, so read the specific method contract.

Stationary behavior:

- Reduced may apply zero-velocity updates when the runtime detects low dynamics
  and low speed.
- `set_r_zero_vel` controls zero-velocity variance; zero disables the
  pseudo-measurement.
- `set_r_stationary_accel` controls stationary gravity pseudo-measurements;
  zero disables them.
- Zero-velocity updates intentionally block mount injection so stationary
  periods do not directly force mount corrections.

Vehicle speed behavior:

- `process_vehicle_speed` currently applies to the Reduced facade path. It
  returns without doing work until Reduced is initialized and mount is ready.
- `Forward` fuses positive vehicle-frame X speed.
- `Reverse` fuses negative vehicle-frame X speed.
- `Unknown` fuses zero velocity when speed is very small; otherwise it infers
  sign only when the current state speed along vehicle X is confident enough.

## Standalone Align API

`align::Align` estimates only the physical mount `q_bv` and a 3 by 3 small-angle
mount covariance over roll, pitch, and yaw.

Typical standalone use:

1. Create `Align::new(AlignConfig::default())`.
2. Call `initialize_from_stationary(&accel_samples_b, yaw_seed_rad)`.
3. Feed `AlignWindowSummary` values to `update_window` or
   `update_window_with_trace`.
4. Check `coarse_alignment_ready()`, `q_bv`, `mount_angles_rad`,
   `mount_angles_deg`, and `sigma_deg`.

`AlignWindowSummary` uses mean body-frame gyro and accelerometer over a window
plus previous/current GNSS NED velocities. Stationary gravity constrains roll
and pitch. GNSS-derived horizontal acceleration and turn windows provide yaw
and planar turn constraints when their motion gates pass.

## Standalone Reduced API

`reduced::Filter` is the local-NED EKF. Its nominal state layout is:

```text
q_nv, v_n, p_n, b_g_b, b_a_b, q_bv
```

where `v_n` and `p_n` are `[north, east, down]`, body-frame biases are in raw
IMU frame `b`, and `q_bv` is stored in `q_bv0..q_bv3`.

Core standalone methods include:

- `Filter::new(ProcessNoise)`.
- `init_nominal_from_gnss(q_nv, reduced::GnssSample)`.
- `predict(reduced::ImuDelta)`, with body-frame delta angle in radians and
  body-frame delta velocity in meters per second.
- `fuse_gps`, `fuse_gps_nhc_batch`, `fuse_zero_vel`,
  `fuse_stationary_gravity`, `fuse_body_speed_x`, and `fuse_body_vel_yz`.
- `set_gravity_mss`, `set_freeze_misalignment_states`, `nominal`,
  `covariance`, `raw`, and `raw_mut`.

The standalone Reduced API expects the caller to provide an already-local
`reduced::GnssSample`; it does not perform WGS84 anchoring. Use
`SensorFusion` if you want lat/lon/height conversion and reanchoring.

## Standalone Full API

`full::Filter` is the ECEF EKF. Its nominal state layout is:

```text
q_ev, v_e, p_e, b_g_b, b_a_b, s_g_b, s_a_b, q_bv
```

The public error-state order is:

```text
dp_e, dv_e, dtheta_v, dba, dbg, dsa, dsg, dpsi_bv
```

Core standalone methods include:

- `Filter::new(ProcessNoise)`.
- `init_from_reference_state`, `init_from_reference_ecef_state`, and
  `init_vehicle_from_nav_ecef_state`.
- `predict(full::ImuDelta)` and `predict_nominal(full::ImuDelta)`, with
  two-sample body-frame delta angles and delta velocities.
- `fuse_gps_reference`, `fuse_gps_reference_full`,
  `fuse_reference_batch_full`, `fuse_reference_batch_full_with_nhc_speed`,
  `fuse_reference_batch_full_with_nhc_speed_and_r`,
  `fuse_nhc_reference_with_speed`, and
  `fuse_nhc_reference_with_speed_and_r`.
- `set_mount_quat`, `set_freeze_mount_states`, `set_covariance`,
  `tighten_mount_covariance_deg`, `raw`, `nominal`, `covariance`,
  `shadow_pos_ecef`, and last-update diagnostics.

Full stores a public f32 ECEF position in `nominal.pn/pe/pd` and an internal
f64 shadow ECEF position exposed by `shadow_pos_ecef()`. Prefer the shadow
position for diagnostics that need ECEF precision.

## no_std Expectations

The `sensor_fusion` crate itself is `no_std`. It uses fixed-size arrays and
`libm` math helpers instead of heap allocation or a standard-library linear
algebra dependency. The optional `serde` feature derives serialization support
for selected configuration/state types without enabling `std`.

Simulation, CSV parsing, hosted datasets, web visualization, and evaluation
tools live outside the embedded core. If adding public APIs intended for
embedded use, keep them allocation-free and avoid `std` dependencies in
`sensor_fusion`.

## Common Pitfalls

- Passing `q_vb` where the API expects `q_bv`. Manual mount APIs require
  vehicle-to-body: `x_b = C_bv x_v`.
- Treating `q_bv0..q_bv3` as an inverse/body-to-vehicle mount. In the current
  public API and tests, both Reduced and Full store physical `q_bv` in those
  fields.
- Forgetting that vectors in `ImuSample` are raw body-frame values. The facade
  and filters rotate them internally using the current mount.
- Assuming Reduced and Full state field names use the same physical frame.
  Reduced `vn/ve/vd` and `pn/pe/pd` are local NED. Full `vn/ve/vd` and
  `pn/pe/pd` are ECEF.
- Reading `reduced()` or `full()` before initialization. Use `Update` readiness
  fields or check for `Option::Some`.
- Expecting Full to initialize from a stationary GNSS sample. Through the
  facade, Full requires at least 0.5 m/s horizontal GNSS speed.
- Feeding out-of-order timestamps. The facade does not sort samples and many
  updates are skipped for invalid or stale intervals.
- Passing standard deviations to `set_r_*` APIs. These setters take variances.
- Passing variances to `sigma` APIs. These setters take one-sigma values.
- Expecting `position_lla_f64()` to work for a Full-only facade. It reports the
  Reduced state through the local anchor.
- Assuming NHC is always active. It is speed and dynamics gated, and positive
  `set_nhc_update_period_s` decimates it.
- Expecting vehicle speed updates to drive Full through the facade. The current
  facade implementation applies `process_vehicle_speed` to the Reduced runtime
  path.
- Treating manual mount plus mount-settle timing as a way to release mount
  states. Manual mount mode freezes mount states through the effective freeze
  policy.
