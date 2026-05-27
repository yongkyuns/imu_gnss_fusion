# Alignment Estimator Formulation

`align` estimates the physical vehicle-to-body mount quaternion `q_bv` before the runtime EKF initializes in automatic mount mode. It is a reduced estimator: its main state is the mount angle, not the full vehicle navigation state.

## Scope

Align uses:

- stationary gravity to initialize roll and pitch;
- GNSS-derived horizontal acceleration to constrain mount yaw during acceleration/braking;
- planar turn gyro consistency to refine roll and pitch during turns.

It does not run NHC. NHC is an EKF runtime update.

## Frames And Quaternion Convention

The mount quaternion is:

```text
q_bv
x_b = C_bv x_v
x_v = C_bv^T x_b
```

Vehicle frame `v` is forward-right-down. Body frame `b` is the raw IMU sensor frame. Quaternion products compose active rotations:

```text
R(q1 * q2) = R(q1) R(q2)
```

## State And Covariance

The estimator tracks a local mount error:

$$
\delta \rho =
\begin{bmatrix}
\delta\rho_x & \delta\rho_y & \delta\rho_z
\end{bmatrix}^T ,
$$

where the axes correspond to local roll, pitch, and yaw perturbations of `q_bv`. A fresh `Align::new` starts with conservative one-sigma uncertainties of approximately:

```text
roll  = 20 deg
pitch = 20 deg
yaw   = 60 deg
```

Stationary initialization tightens the tilt seed to approximately:

```text
roll  = 10 deg
pitch = 10 deg
yaw   = 60 deg
```

Yaw stays broad until motion provides a horizontal acceleration cue.

## Stationary Initialization

When the phone/vehicle is stationary, the accelerometer sees gravity. This constrains the body-frame down direction relative to the vehicle down direction and gives a roll/pitch seed. Yaw remains unobservable from gravity alone.

The runtime tilt initializer uses stationary gates derived from gyro norm and accelerometer norm error. Once enough stationary samples are accumulated, align can initialize tilt. It does not seed yaw; yaw stays broad until motion supplies a horizontal acceleration cue.

## Prediction

The align covariance can be inflated between observation windows to reflect drift and unmodeled motion. This is deliberately simple; the estimator is a seed generator, not a full inertial navigation filter.

## Shared Observation Linearization

For a predicted vehicle-frame vector `x_v`, the body-frame prediction is:

$$
\hat{x}_b = C_{bv} x_v .
$$

A small mount perturbation changes this prediction approximately by a cross-product Jacobian:

$$
\delta \hat{x}_b \approx -[C_{bv}x_v]_\times \delta\rho .
$$

The same local linearization is used for gravity, horizontal acceleration, and turn-gyro consistency rows.

## Stationary Gravity Observation

The gravity observation compares the measured body-frame gravity direction against the predicted body-frame direction from `q_bv`. It strongly constrains roll and pitch, but not yaw:

$$
z_g \approx C_{bv} e_z .
$$

This is why stationary initialization can seed tilt while yaw remains broad.

## GNSS-Derived Horizontal Yaw Observation

During forward acceleration or braking, GNSS velocity changes provide an approximate vehicle-frame longitudinal acceleration direction. Comparing that direction with the body-frame accelerometer direction gives a mount yaw cue.

This cue is sensitive to GNSS velocity noise, weak acceleration, and non-longitudinal vehicle motion. Align therefore gates and weights the observation conservatively.

## Turn Consistency And Planar Gyro Observation

During approximately planar turns, yaw-rate structure gives another body-to-vehicle consistency cue for roll and pitch. It does not observe mount yaw. The estimator uses this as a refinement signal rather than as a full vehicle dynamics model.

## Readiness

Coarse alignment readiness requires yaw to have been observed and one-sigma covariance gates approximately:

```text
roll  <= 5 deg
pitch <= 5 deg
yaw   <= 8 deg
```

The `SensorFusion` facade can optionally copy align's mount covariance into the EKF residual mount block at handoff.

## Known Limitations

Because align is a reduced formulation, it cannot fully distinguish mount roll from sustained vehicle roll or road bank. It should be treated as a seed whose uncertainty and downstream EKF behavior must still be evaluated.
