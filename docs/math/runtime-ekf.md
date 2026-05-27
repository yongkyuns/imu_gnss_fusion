# Runtime EKF Formulation

The runtime EKF is an error-state vehicle navigation filter. The high-level `SensorFusion` facade owns mount-mode policy, WGS84/local anchoring, sample dispatch, update staging, and diagnostics; the EKF module owns nominal propagation, covariance propagation, measurement updates, injection, and reset.

## Frames And Mount Sources

The filter uses the project-wide active rotation convention:

```text
x_a = C_ab x_b
R(q_ab) = C_ab
```

The public mount is the physical vehicle-to-body quaternion:

```text
x_b = C_bv x_v
x_v = C_vb x_b = C_bv^T x_b
```

The vehicle attitude is:

```text
x_n = C_nv x_v
```

Raw IMU deltas are expressed in body frame `b`. Prediction rotates them through the current mount into vehicle frame, then through vehicle attitude into the navigation frame.

## Nominal State

The nominal state is:

$$
x =
\begin{bmatrix}
q_{nv} & v_n^T & p_n^T & b_g^T & b_a^T & q_{bv}
\end{bmatrix}^T .
$$

The corresponding error state contains:

$$
\delta x =
\begin{bmatrix}
\delta\theta_v &
\delta v_n &
\delta p_n &
\delta b_g &
\delta b_a &
\delta\psi_{bv}
\end{bmatrix}^T .
$$

`q_nv` is the vehicle-to-navigation attitude. `q_bv` is the physical vehicle-to-body mount. The mount error is represented locally as a small rotation and injected back into `q_bv`.

## Nominal Propagation

For one IMU interval, the runtime forms body-frame angle and velocity increments:

$$
\Delta\alpha_b = \omega_b \Delta t,\qquad
\Delta v_b = f_b \Delta t .
$$

Bias-corrected increments are rotated into vehicle and navigation coordinates:

$$
\Delta v_v = C_{vb}(\Delta v_b - b_a\Delta t),
$$

$$
\Delta v_n = C_{nv}\Delta v_v + g_n \Delta t .
$$

The nominal velocity and position update follow the standard strapdown structure:

$$
v_n^+ = v_n + \Delta v_n,
\qquad
p_n^+ = p_n + v_n\Delta t + \frac{1}{2}\Delta v_n\Delta t .
$$

The attitude update composes the current attitude with the bias-corrected vehicle-frame angular increment. After each prediction and injection, quaternions are normalized.

## Error Propagation

The covariance propagation uses the generated linearized transition:

$$
P^+ = FPF^T + GQG^T .
$$

Gyro and accelerometer white noise are continuous noise densities and contribute proportional to `dt`. Bias random-walk noise is also density-based. Mount random-walk terms are zero when mount states are frozen and otherwise use the configured per-axis or shared mount process noise.

The implementation uses sparse covariance update policies where practical so the embedded runtime does not pay for dense operations in every scalar update.

## Measurement Updates

### GNSS Position And Velocity

GNSS samples are converted from WGS84 into the current local frame. Position and NED velocity rows directly observe `p_n` and `v_n`, with caller-provided one-sigma standard deviations clamped to conservative lower bounds.

When GNSS and NHC are pending at the same IMU epoch, the runtime can batch GNSS velocity with the vehicle-frame NHC rows so the covariance update sees a consistent measurement set.

### Zero Velocity

When runtime stationary gates pass, the filter can apply a zero-velocity observation:

$$
v_n \approx 0 .
$$

The facade currently keeps this update configurable. The default runtime value disables it unless the caller sets a positive variance.

### Vehicle Speed And NHC

A vehicle-speed observation constrains the forward vehicle-frame velocity:

$$
e_x^T C_{nv}^T v_n \approx v_x .
$$

The nonholonomic constraint observes lateral and vertical vehicle-frame velocity:

$$
e_y^T C_{nv}^T v_n \approx 0,
\qquad
e_z^T C_{nv}^T v_n \approx 0 .
$$

NHC is eligible only when:

- EKF speed is greater than `0.05 m/s`;
- vehicle-frame gyro norm is below `0.2 rad/s`;
- accelerometer norm error from gravity is below `1.0 m/s^2`.

The default NHC period is `0.1 s`. Positive decimation periods scale the observation variance by the elapsed NHC interval to preserve an approximately stable information rate.

### Stationary Gravity

When stationary, the accelerometer specific force can provide a gravity-direction observation in the vehicle frame. This is separate from normal dynamic accelerometer propagation and is disabled unless configured.

### Vehicle-Roll Prior

The optional vehicle-roll prior observes:

$$
\operatorname{roll}(q_{nv}) \approx 0 .
$$

The default facade value is `r_vehicle_roll_prior = 0.1`; `0` disables it. This is a flat-road prior, not a bank-safe sensor measurement. It is applied only at eligible NHC epochs and uses the same interval scaling as NHC.

## Injection And Reset

After a measurement update, the error-state correction is injected into the nominal state:

- attitude error updates `q_nv`;
- velocity, position, and bias errors add directly;
- mount error updates `q_bv`.

The covariance is then transformed by the reset Jacobian so the post-injection error state is again centered near zero.

## Initialization And Mount Modes

Auto mode waits for align to produce a ready physical mount seed and for GNSS yaw/course readiness. Manual mode accepts a caller-supplied `q_bv` and bypasses internal align, but the facade does not freeze EKF mount states. It seeds a live residual mount prior.

Manual mode requires `heading_rad` and speed greater than `max(yaw_init_speed_mps, 20 / 3.6)` before EKF initialization. Auto mode can use `heading_rad` or GNSS course once horizontal speed is at least `max(yaw_init_speed_mps, 1.0)`.

## Modeling Boundaries

NHC lateral/vertical rows are instantaneous vehicle-frame velocity constraints. In the current formulation their direct measurement sensitivity to mount is zero; mount correction is mediated through covariance and propagation coupling. Absolute mount roll remains weakly identifiable without informative motion or an explicit prior/model for road bank and vehicle roll.
