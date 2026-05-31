# Filter Algorithms

The runtime has two filter layers:

- `align`: a reduced mount estimator used to produce an initial $q_{bv}$ seed in automatic mount mode.
- `ekf`: the runtime vehicle-state filter used after GNSS and mount readiness.

```{figure} _static/diagrams/estimator-runtime-orthogonal.svg
:alt: Estimator lifecycle from raw IMU/GNSS inputs through align, EKF initialization, prediction, updates, and public state outputs.
:class: framed

The facade owns readiness and dispatch. Align only seeds automatic mount mode; the runtime EKF owns prediction, measurement updates, vehicle constraints, and public state.
```

## Align

Align estimates the physical vehicle-to-body mount quaternion $q_{bv}$. It uses static accelerometer tilt samples for roll/pitch initialization/refinement, GNSS-derived horizontal acceleration for yaw, and planar turn gyro windows for roll/pitch refinement. It is not the NHC runtime filter.

See [](algorithms/align.md) for the full reduced-estimator formulation.

Coarse alignment readiness requires yaw observation and one-sigma covariance gates approximately:

$$
\begin{aligned}
\sigma_\mathrm{roll} &\le 5^\circ,\\
\sigma_\mathrm{pitch} &\le 5^\circ,\\
\sigma_\mathrm{yaw} &\le 8^\circ .
\end{aligned}
$$

Stationary tilt initialization seeds roll/pitch/yaw covariance at about $[10, 10, 60]^\circ$; yaw remains broad because gravity does not observe yaw. A fresh `Align::new` starts with about $[20, 20, 60]^\circ$.

## EKF Runtime

The EKF state includes vehicle attitude $q_{nv}$, local velocity/position, IMU biases, and residual mount states. Prediction consumes raw body-frame IMU deltas, rotates them through the current $C_{vb}$, and propagates the vehicle navigation state.

See [](algorithms/runtime-ekf.md) for the implemented EKF state ordering, generated Jacobians, scalar and sequential-batch update algebra, GNSS gating, NHC scheduling, and injection/reset behavior.

Runtime update families include:

- GNSS position and velocity;
- zero-velocity updates when the runtime detects stationary conditions;
- optional vehicle-speed observations;
- nonholonomic vehicle-frame lateral/vertical velocity constraints;
- optional vehicle-roll prior at eligible NHC epochs.

For comparisons with related smartphone telematics, automotive INS/GNSS,
OpenIMU, and PX4 EKF2 formulations, see [](reference/prior-work.md).

## Mount States And AHRS Boundaries

The runtime estimates mount because $q_{bv}$ is part of IMU mechanization. A
mount error rotates gyro and accelerometer increments into the wrong vehicle
axes before attitude, velocity, and position are propagated. Automatic mode uses
align to obtain the initial mount seed; the EKF then estimates residual mount
states.

The separation is also a local-linearity boundary. Align is a reduced estimator
with a wider practical capture range for stationary tilt, horizontal-acceleration
yaw, and planar turn cues. The EKF is a local error-state filter; after
initialization it expects small attitude and mount corrections around the
current nominal state.

Mahony-, Madgwick-, and other AHRS-style filters estimate attitude from gyro
integration plus accelerometer leveling, and sometimes magnetometer heading.
They do not estimate local velocity/position, accelerometer bias, or
sensor-to-vehicle mount as part of a GNSS-aided navigation state. During
acceleration, braking, and cornering, treating the accelerometer as a gravity
vector can bias the attitude estimate. The EKF uses the accelerometer as a
propagation input and uses GNSS, vehicle-frame constraints, and covariance to
allocate residuals among attitude, velocity, bias, and mount.

See [](algorithms/mount-states.md) for the detailed discussion, including the
mount/vehicle-roll ambiguity, bias states, and omitted scale/lever-arm terms.

## Runtime Persistence

The runtime EKF is persistent as long as the caller keeps the same `SensorFusion` object. That context includes nominal navigation state, covariance, mount, raw-body IMU biases, and diagnostics. A normal pause does not require a new alignment cycle.

```{figure} _static/diagrams/ekf-persistence-lifecycle-elk.svg
:alt: EKF persistence lifecycle across fresh initialization, running, stream pause, sleep gaps, degraded dead reckoning, and GNSS reseed.
:class: framed

Sleep gaps are treated as stationary missing-sample intervals. The runtime ages covariance in a bounded way, keeps mount unchanged, and decides whether navigation can continue or must wait for GNSS reseed.
```

The post-gap model is conservative:

- short sleep up to 15 minutes keeps navigation usable;
- medium sleep up to one hour enters degraded dead reckoning if covariance remains usable;
- long sleep, invalid gaps, or unusable covariance enter GNSS reseed mode.

See [](runtime-state-and-persistence.md) for caller responsibilities and exact covariance aging values.

## NHC Scheduling

NHC applies only when runtime gates pass:

- EKF speed estimate is greater than $0.05\,\mathrm{m/s}$;
- vehicle-frame gyro norm is below $0.2\,\mathrm{rad/s}$;
- accelerometer norm error from gravity is below $1.0\,\mathrm{m/s^2}$.

The default NHC update period is $0.1\,\mathrm{s}$ (10 Hz). A positive period decimates eligible updates and scales observation variance by the elapsed NHC interval so the constraint behaves like a rate-limited continuous observation.

See [](algorithms/observability.md) for the distinction between direct NHC sensitivity and covariance-mediated mount correction.

## Vehicle-Roll Prior

`set_r_vehicle_roll_prior` configures a soft flat-road prior:

$$
\operatorname{roll}(q_{nv}) \approx 0.
$$

$0$ disables the prior. Positive values are interpreted as a variance density and scaled at the same eligible epochs as NHC. The default runtime value is $0.1$.

The prior is scheduled inside the NHC epoch path. If both lateral and vertical
NHC variances are set to zero, the facade does not enter that path and the
vehicle-roll prior will not run even if its own variance is positive.

This is not a direct mount-roll measurement. It can reduce roll ambiguity on mostly flat roads, but sustained banked roads can convert the flat-road assumption into mount error.

Diagnostics for these update families are summarized in [](ekf-diagnostics.md).
