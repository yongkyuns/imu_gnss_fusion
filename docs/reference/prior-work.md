# Prior Work And Technical Comparisons

The following projects and papers target different sensor suites, platforms,
and operating domains. They are included for formulation comparison.

## Reference Formulation

`sensor_fusion` is a ground-vehicle IMU/GNSS filter written for reusable Rust
and embedded/mobile targets. The runtime state uses vehicle attitude $q_{nv}$,
local velocity/position, raw-body IMU biases, and a physical vehicle-to-body
mount quaternion $q_{bv}$. Automatic mount mode uses a reduced align estimator
before EKF initialization; after that, the runtime EKF owns propagation, GNSS
updates, NHC, optional vehicle-speed observations, a flat-road vehicle-roll
prior, diagnostics, and sleep/reseed behavior.

The current measurement set is raw IMU plus GNSS, with optional vehicle speed.
Magnetometer heading, wheel odometry, OBD, RTK, cameras, and vehicle-bus signals
are outside the current formulation.

## Summary Table

| Resource | Domain | Relevant overlap | Main formulation difference | Relevant section |
| --- | --- | --- | --- | --- |
| [Wahlstrom 2017, Chapter 6](https://www.diva-portal.org/smash/get/diva2:1159484/FULLTEXT02.pdf) | smartphone automotive telematics | phone-to-vehicle alignment using GNSS-aided INS, NHC, and roll pseudo-observation | single augmented EKF with smartphone attitude and phone-to-vehicle Euler states; uses GNSS position, speed, and course plus an MPF-style yaw initialization | roll observability, NHC, and why a roll pseudo-observation appears |
| [fedorbaklanov/open-aided-navigation](https://github.com/fedorbaklanov/open-aided-navigation) | educational MATLAB aided navigation demos | loosely coupled automotive INS/GNSS with mounting alignment and NHC | ECEF nominal state, sensor scale-factor states, direct NHC Jacobian columns for mount alignment, MATLAB clarity over embedded packaging | scalar update algebra, NHC gating, ECEF loose coupling |
| [PX4 EKF2](https://github.com/PX4/PX4-Autopilot/tree/main/src/modules/ekf2) | production aerial robotics autopilot | error-state EKF, generated Jacobians, aid-source health, resets, embedded constraints | no phone/vehicle mount state or ground-vehicle NHC; much richer aerial sensor suite with magnetometer, barometer, range, optical flow, airspeed, wind, and terrain states | generated-model workflow, estimator health, reset/fault design |
| [ACEINNA OpenIMU](https://www.aceinna.com/openimu) | open embedded IMU hardware/software platform | EKF/AHRS/INS implementation context and embedded navigation stack | hardware/toolchain-centered platform; typical measurement model uses accelerometer/magnetometer/GPS-derived attitude references rather than estimating arbitrary phone-to-vehicle mount | embedded IMU platform architecture and general EKF background |

## Wahlstrom 2017 Smartphone Telematics

Johan Wahlstrom's thesis, especially Chapter 6, is the closest prior formulation
to this repository's mount-estimation work. It frames smartphone automotive
navigation as a GNSS-aided INS where the smartphone IMU orientation relative to
the vehicle is unknown. The chapter augments the navigation state with the
smartphone-to-vehicle orientation and introduces pseudo-observations for:

- vehicle-frame lateral and vertical velocity, i.e. NHC;
- vehicle roll being close to zero.

The chapter states the roll issue explicitly: with GNSS measurements and NHC
alone, the initial smartphone-to-vehicle roll angle does not sufficiently affect
the error estimates, so a roll pseudo-observation is introduced to reduce the
initial roll ambiguity. This is consistent with the bank/mount-roll ambiguity
described in [](../algorithms/observability.md).

Differences:

- **State representation.** Chapter 6 uses a navigation state
  containing smartphone position, smartphone velocity, smartphone Euler attitude,
  and smartphone-to-vehicle Euler angles. This implementation uses vehicle attitude
  $q_{nv}$ as the runtime attitude and a physical mount quaternion $q_{bv}$ as a
  residual calibration state.
- **Measurement set.** Chapter 6 models GNSS position, horizontal speed, and
  planar course. This implementation consumes GNSS position and full NED
  velocity, with optional vehicle-speed observations.
- **Initialization.** Chapter 6 initializes position/velocity from
  GNSS, tilt from accelerometer data, vehicle yaw from GNSS course, and
  smartphone yaw through a marginalized particle filter. This implementation uses a
  reduced `align` estimator in automatic mount mode, with stationary tilt,
  horizontal-acceleration yaw, and turn-gyro roll/pitch cues.
- **Roll pseudo-observation.** The chapter's roll pseudo-observation is part of
  the proposed augmented EKF model. This implementation exposes the analogous
  flat-road assumption as a configurable vehicle-roll prior, enabled by default
  and not bank-safe.
- **Update scheduling.** The chapter presents dense EKF updates at the
  smartphone IMU/GNSS rates used in the study. This implementation uses scalar and
  sequential-batch updates, decimated NHC by default, and variance-density
  scaling so update strength is less sample-rate dependent.

The roll pseudo-observation is an assumption about the road/vehicle roll
channel. It should not be interpreted as bank-safe mount-roll observability from
IMU/GNSS and NHC alone.

## Open Aided Navigation

[Open Aided Navigation](https://github.com/fedorbaklanov/open-aided-navigation)
is a MATLAB project for aided navigation demonstrations. Its automotive
[`demo/insGnssLoose`](https://github.com/fedorbaklanov/open-aided-navigation/tree/master/demo/insGnssLoose)
example is a loosely coupled INS/GNSS automotive filter.

The automotive demo uses:

- a 26-element nominal state with ECEF position/velocity, sensor-to-ECEF
  quaternion, accelerometer and gyro biases, accelerometer and gyro scale
  factors, and an IMU-to-car quaternion;
- a 24-element error state with three mount-alignment error angles;
- GNSS position updates;
- NHC updates on car-frame lateral and vertical velocity;
- scalar Joseph-form updates with residual correction of the form
  $r_i - H_i\delta x$;
- measurement-variance scaling by elapsed update time so covariance is less
  sensitive to measurement frequency.

The NHC formulation is a contrast. The demo projects ECEF velocity into the car
frame using the current IMU-to-car rotation, then builds
NHC rows with direct mount-alignment sensitivity. In this implementation, the
implemented NHC rows are functions of the runtime vehicle attitude $q_{nv}$ and
local velocity, so the instantaneous NHC rows have zero direct residual-mount
columns; mount updates arrive through propagation-created covariance coupling.
These are different state-coordinate choices around similar physical
constraints.

Differences:

- **Coordinate frame.** Open Aided Navigation keeps the navigation state in ECEF.
  This implementation anchors a local NED frame through the facade before runtime
  EKF updates.
- **Mount initialization.** The MATLAB demo starts from configured
  IMU-to-car Euler constants. This implementation supports automatic and manual
  mount initialization.
- **Sensor error model.** The MATLAB demo estimates accelerometer and gyro scale
  factors as states. This implementation estimates raw-body additive gyro
  and accelerometer biases, but not scale factors.
- **NHC gating.** The MATLAB demo disables NHC under higher angular rate or
  accelerometer-norm deviation. This implementation has similar runtime gates plus
  NHC decimation and diagnostics.
- **Roll prior.** The MATLAB demo's automotive example has NHC but no explicit
  flat-road vehicle-roll prior. This implementation separates NHC from the optional
  vehicle-roll prior because they address different observability questions.

Open Aided Navigation documents loose INS/GNSS structure and scalar update
mechanics. Its direct NHC mount Jacobian should be interpreted with its state
definition; it does not by itself remove the physical road-bank/mount-roll
ambiguity.

## PX4 EKF2

[PX4 EKF2](https://github.com/PX4/PX4-Autopilot/tree/main/src/modules/ekf2) is
a production flight-control estimator. It is not a ground-vehicle phone-mount
estimator, but it is relevant for error-state EKF structure, generated algebra,
aid-source status, resets, fault handling, and embedded constraints.

Notable comparison points:

- **Generated algebra.** PX4's EKF2 includes generated state and covariance
  algebra under its `EKF/python/ekf_derivation` path and calls generated
  prediction/observation helpers from C++. This implementation uses SymPy to emit
  Rust fragments under `sensor_fusion/src/ekf/generated/`, with hand-written
  wrappers and runtime policy around them.
- **State size and purpose.** PX4's generated tangent state includes attitude,
  velocity, position, gyro bias, accelerometer bias, magnetic earth/body states,
  wind, and terrain. This implementation's runtime tangent state is smaller and
  ground-vehicle specific: attitude, velocity, position, raw-body IMU biases,
  and residual mount.
- **Aiding architecture.** PX4 has a broad aid-source matrix: GNSS, barometer,
  range finder, optical flow, external vision, magnetometer, airspeed, drag,
  wind, and terrain. This implementation keeps the primary runtime to
  IMU/GNSS/vehicle constraints and puts road-event analysis in a separate crate.
- **Health and resets.** PX4's estimator architecture has extensive innovation
  checks, reset status, aid-source status, delayed fusion horizons, output
  prediction, and multi-IMU/multi-magnetometer failover. This implementation has a
  simpler public `FusionState` lifecycle plus GNSS gates and sleep/reseed
  behavior, aimed at app and embedded callers rather than flight control.
- **Mount model.** PX4 EKF2 does not estimate an arbitrary phone-to-vehicle
  mount. Sensor extrinsics are configuration/calibration inputs, not the central
  observable state.

PX4 EKF2 is relevant for production estimator architecture and generated-code
workflows. It is not a direct comparison for ground-vehicle NHC or smartphone
mount-roll observability.

## ACEINNA OpenIMU

[OpenIMU](https://www.aceinna.com/openimu) is an open embedded IMU platform.
Its VG/AHRS/INS algorithms use standard inertial navigation components:
accelerometers level roll/pitch, magnetometers or GPS-derived heading constrain
yaw, and GPS position/velocity correct INS drift.

OpenIMU has a different scope:

- **Platform versus library.** OpenIMU is tied to calibrated ACEINNA IMU
  hardware, firmware deployment, a VS Code extension, drivers, and Navigation
  Studio tooling. This implementation is a Rust software workspace with web/iOS
  replay tools and no required hardware vendor.
- **Heading sources.** OpenIMU includes magnetometer and GPS heading source
  selection. This implementation avoids magnetometer reliance
  for vehicle mount estimation because phone magnetometers are often disturbed
  in vehicles.
- **Attitude measurement model.** OpenIMU's educational measurement model
  derives roll/pitch from accelerometer leveling and heading from magnetometer
  or GPS course. This implementation uses those ideas in the reduced align
  estimator, but the runtime EKF estimates a vehicle-frame navigation state and
  residual mount rather than treating attitude measurements as direct Euler
  observations.
- **Embedded emphasis.** OpenIMU documents embedded sampling, packets, drivers,
  and deployment. The repo's embedded notes are currently budget evidence for
  `sensor_fusion` and `road_events`, not a complete firmware distribution.

OpenIMU is a general INS/AHRS and embedded IMU platform reference. It is less
directly comparable to the phone/device-to-vehicle mount observability problem.

## Reading These Sources Against This Codebase

When comparing formulations, avoid matching only by the words "NHC" or "EKF".
The important questions are:

- Is the attitude state the sensor attitude, vehicle attitude, or both?
- Is the mount an estimated state, a configured extrinsic, or an initialization
  product?
- Does an NHC row depend directly on mount in that state coordinate choice, or
  only through propagated covariance?
- Is roll constrained by data, by a flat-road prior, by a vehicle model, or by
  an external sensor?
- Is the method an attitude-only AHRS, or does it estimate navigation velocity,
  position, accelerometer bias, and mount inside one aided navigation state?
- Are GNSS measurements position/velocity, speed/course, raw pseudorange, or
  something else?
- Are Jacobians hand-written, generated, or numerically approximated?
- Are update variances treated as per-sample values or density-like values
  scaled by update interval?

These distinctions explain why two filters can look similar on paper while
behaving differently in roll ambiguity, banked turns, GNSS outages, or embedded
compute budgets.
