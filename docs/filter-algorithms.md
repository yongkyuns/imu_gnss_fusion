# Filter Algorithms

The runtime has two filter layers:

- `align`: a reduced mount estimator used to produce an initial `q_bv` seed in automatic mount mode.
- `ekf`: the runtime vehicle-state filter used after GNSS and mount readiness.

```{figure} _static/diagrams/estimator-runtime.png
:alt: Estimator lifecycle from raw IMU/GNSS inputs through align, EKF initialization, prediction, updates, and public state outputs.
:class: framed

The facade owns readiness and dispatch. Align only seeds automatic mount mode; the runtime EKF owns prediction, measurement updates, vehicle constraints, and public state.
```

## Align

Align estimates the physical vehicle-to-body mount quaternion `q_bv`. It uses stationary gravity for tilt initialization/refinement, GNSS-derived horizontal acceleration for yaw, and planar turn gyro windows for roll/pitch refinement. It is not the NHC runtime filter.

Coarse alignment readiness requires yaw observation and one-sigma covariance gates approximately:

```text
roll <= 5 deg
pitch <= 5 deg
yaw <= 8 deg
```

Stationary tilt initialization seeds roll/pitch/yaw covariance at about `[10, 10, 60] deg`; yaw remains broad because gravity does not observe yaw. A fresh `Align::new` starts with about `[20, 20, 60] deg`.

## EKF Runtime

The EKF state includes vehicle attitude `q_nv`, local velocity/position, IMU biases, and residual mount states. Prediction consumes raw body-frame IMU deltas, rotates them through the current `C_vb`, and propagates the vehicle navigation state.

Runtime update families include:

- GNSS position and velocity;
- zero-velocity and stationary-gravity updates when the runtime detects stationary conditions;
- optional vehicle-speed observations;
- nonholonomic vehicle-frame lateral/vertical velocity constraints;
- optional vehicle-roll prior at eligible NHC epochs.

## NHC Scheduling

NHC applies only when runtime gates pass:

- EKF speed estimate is greater than `0.05 m/s`;
- vehicle-frame gyro norm is below `0.2 rad/s`;
- accelerometer norm error from gravity is below `1.0 m/s^2`.

The default NHC update period is `0.1 s` (10 Hz). A positive period decimates eligible updates and scales observation variance by the elapsed NHC interval so the constraint behaves like a rate-limited continuous observation.

## Vehicle-Roll Prior

`set_r_vehicle_roll_prior` configures a soft flat-road prior:

```text
vehicle_roll ~= 0
```

`0` disables the prior. Positive values are interpreted as a variance density and scaled at the same eligible epochs as NHC. The default runtime value is `0.1`.

This is not a direct mount-roll measurement. It can reduce roll ambiguity on mostly flat roads, but sustained banked roads can convert the flat-road assumption into mount error.
