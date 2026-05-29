# Runtime EKF Formulation

The runtime EKF is an error-state vehicle navigation filter. The high-level
`SensorFusion` facade owns mount-mode policy, WGS84/local anchoring, sample
dispatch, update staging, NHC scheduling, sleep/reseed policy, and diagnostics.
The `ekf` module owns nominal propagation, covariance propagation, measurement
updates, injection, and covariance reset.

This page documents the implemented filter. For frame and quaternion conventions,
see [](frames.md).

## State Ordering

The nominal state is:

$$
x =
\begin{bmatrix}
q_{nv} & v_n^T & p_n^T & b_g^T & b_a^T & q_{bv}
\end{bmatrix}^T .
$$

The error state is 18-dimensional:

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

The ordering in generated code is:

$$
\begin{aligned}
0{:}2 &: \delta\theta_v, &
3{:}5 &: \delta v_n, &
6{:}8 &: \delta p_n,\\
9{:}11 &: \delta b_g, &
12{:}14 &: \delta b_a, &
15{:}17 &: \delta\psi_{bv}.
\end{aligned}
$$

Gyro and accelerometer biases are additive corrections in the raw IMU body frame
`b`. The mount quaternion `q_bv` is the physical vehicle-to-body mount, with
`x_b = C_bv x_v`.

The process-noise vector used by the generated transition is:

$$
w =
\begin{bmatrix}
n_{\Delta\alpha_b} &
n_{\Delta v_b} &
n_{b_g} &
n_{b_a} &
n_{\psi_{bv}}
\end{bmatrix}^T ,
$$

with three axes per group.

## Nominal Propagation

For one IMU interval, the facade forms body-frame increments from two adjacent
raw samples:

$$
\Delta\alpha_b = \omega_b \Delta t,\qquad
\Delta v_b =
\frac{1}{2}(f_{b,k-1}+f_{b,k})\Delta t .
$$

The facade also subtracts local Earth-rate/transport-rate terms from gyro before
calling the generated local-level model, and applies the matching
Coriolis/transport velocity correction in NED after prediction.

The generated nominal prediction rotates bias-corrected body increments through
the current mount into vehicle frame:

$$
\begin{aligned}
\Delta\alpha_v &= C_{vb}(\Delta\alpha_b - b_g\Delta t),\\
\Delta v_v &= C_{vb}(\Delta v_b - b_a\Delta t).
\end{aligned}
$$

It then propagates attitude, velocity, and position as:

$$
\begin{aligned}
q_{nv}^+ &= q_{nv}\,\delta q(\Delta\alpha_v),\\
v_n^+ &= v_n + C_{nv}\Delta v_v + g_n\Delta t,\\
p_n^+ &= p_n + v_n\Delta t .
\end{aligned}
$$

Biases and the nominal mount quaternion are constant during prediction. Their
uncertainty evolves through the error-state transition and process-noise model.

## Jacobian Generation

The generated matrices are discrete-time matrices for one IMU increment, not
continuous-time matrices. The symbolic generator uses a perturb-propagate-linearize
construction:

1. Create a nominal state `x` and a perturbed true state `x \oplus \delta x`.
2. Propagate nominal and perturbed states through the same mechanization.
3. Add noise to IMU increments, bias random walks, and mount random walk.
4. Extract the next error state from the propagated true and nominal states.
5. Differentiate at zero error and zero noise.

In compact form:

$$
\delta x_{k+1} =
f_\delta(\delta x_k, w_k; x_k, u_k),
$$

$$
F_k =
\left.\frac{\partial f_\delta}{\partial \delta x}\right|_{\delta x=0,w=0},
\qquad
G_k =
\left.\frac{\partial f_\delta}{\partial w}\right|_{\delta x=0,w=0}.
$$

Covariance prediction is:

$$
P^+ = FPF^T + GQG^T .
$$

Gyro and accelerometer white-noise densities contribute `density * dt`.
Bias-random-walk columns in `G` already include a factor of `dt`, so the runtime
uses `density / dt` in the diagonal `Q` entries to produce a net
`density * dt` contribution. Mount random-walk variance is applied per axis as
`q_mount_i * dt`, or zero when residual mount states are frozen.

The runtime uses generated row-support metadata to run the same prediction with
sparse covariance multiplication.

## Scalar Observation Model

Most measurement rows are generated as scalar observations. For a predicted
scalar measurement `h(x)`, the generator differentiates the perturbed true
measurement with respect to the error state:

$$
H = \left.\frac{\partial h(x\oplus\delta x)}{\partial \delta x}\right|_{\delta x=0}.
$$

For a scalar residual `r = z - h(x)`, innovation variance and gain are:

$$
S = HPH^T + R,\qquad K = PH^T S^{-1},\qquad \delta x = Kr .
$$

Single scalar updates use Joseph-form covariance update:

$$
P^+ = P - K(HP) - (PH^T)K^T + SKK^T .
$$

After covariance update, the runtime injects the error state once and applies
the attitude reset described in [](frames.md).

## Sequential Batch Updates

When pending GNSS and NHC are fused at the same IMU epoch, the runtime builds up
to eight rows:

$$
\begin{bmatrix}
p_n & p_e & p_d & v_n & v_e & v_d & v_y^v & v_z^v
\end{bmatrix}.
$$

Rows are processed sequentially using scalar inverses. The batch keeps an
accumulated correction `\delta x_b`. For row `i`, the effective residual is:

$$
r_i^\star = r_i - H_i\delta x_b .
$$

Then:

$$
S_i = H_iPH_i^T + R_i,\qquad
\delta x_i = PH_i^T S_i^{-1} r_i^\star .
$$

The batch covariance downdate is:

$$
P^+ = P - PH_i^T S_i^{-1} H_iP .
$$

The nominal state is injected once after all accepted rows have contributed.
This avoids an expensive dense measurement inverse while keeping the GNSS and
NHC rows linearized at the same pre-update nominal state.

## Measurement Families

### GNSS Position And Velocity

GNSS samples are converted from WGS84 into the current local anchor frame before
they reach the EKF. The facade preprocesses reported standard deviations:

$$
\sigma_p =
\max\left(
\frac{\sigma_{p,n}+\sigma_{p,e}+\sigma_{p,d}}{3},
0.1\,\mathrm{m}
\right),
$$

$$
\sigma_{p,n}'=\sigma_p,\qquad
\sigma_{p,e}'=\sigma_p,\qquad
\sigma_{p,d}'=2.5\sigma_p ,
$$

and velocity sigmas are floored per axis at `0.01 m/s`.

When GNSS position is fused faster than 1 Hz after initialization, position
standard deviations are multiplied by:

$$
\sqrt{\frac{1}{\operatorname{clamp}(\Delta t_\mathrm{gnss}, 10^{-3}, 1)}} ,
$$

so higher GNSS rates do not make position residuals disproportionately stiff.
Velocity rows are not rate-normalized by this step.

GNSS rows directly observe local NED position and velocity:

$$
\begin{aligned}
r_{p_i} &= p_{i,\mathrm{gnss}} - p_i,\\
r_{v_i} &= v_{i,\mathrm{gnss}} - v_i .
\end{aligned}
$$

### GNSS Gating And Events

Position and velocity are gated independently as two three-axis groups. The
default gate is three sigma per axis:

$$
\mathrm{NIS}_i =
\frac{r_i^2}{P_{ii}+R_i}
\le 3^2 .
$$

Invalid residuals, nonpositive variances, and invalid innovation variances are
skipped for that axis. A failed group is rejected unless one of two bypasses
applies:

- the elapsed GNSS update gap is greater than `3 s`;
- reported RMS accuracy improves to at most `0.5` of the previous RMS.

Rejected groups emit public `gnss_event_mask` bits. From the third consecutive
rejection onward the runtime also emits the corresponding consecutive-rejection
bit; it still rejects the group rather than forcing a recovery update.

### Zero Velocity

When runtime stationary gates pass, the filter can apply:

$$
v_n \approx 0 .
$$

The default runtime value disables this update unless the caller sets a positive
variance.

### Vehicle Speed And NHC

A vehicle-speed observation constrains forward vehicle-frame velocity:

$$
e_x^T C_{nv}^T v_n \approx v_x .
$$

The nonholonomic constraint observes lateral and vertical vehicle-frame velocity:

$$
e_y^T C_{nv}^T v_n \approx 0,
\qquad
e_z^T C_{nv}^T v_n \approx 0 .
$$

The direct NHC row Jacobian contains attitude and velocity sensitivity, but no
direct residual-mount columns. Mount correction can still occur through
cross-covariance created by prediction and previous updates.

NHC is eligible only when:

- EKF speed is greater than `0.05 m/s`;
- vehicle-frame gyro norm is below `0.2 rad/s`;
- accelerometer norm error from gravity is below `1.0 m/s^2`.

The default NHC period is `0.1 s`. If NHC is inactive, the previous NHC time is
reset. If a positive period is configured, eligible updates are decimated until
the elapsed observation interval reaches the period. The variance scale is:

$$
R_\mathrm{eff} =
R_0 \frac{1}{\min(\Delta t_\mathrm{obs},1)} .
$$

GNSS samples are queued on `process_gnss` and fused on the next IMU epoch. If the
queued GNSS sample is no more than `0.05 s` older than that IMU epoch and NHC is
eligible, GNSS position, GNSS velocity, and NHC rows are fused in the same
sequential batch. Otherwise NHC is applied as a standalone vehicle-frame velocity
update.

### Stationary Gravity

When stationary, the runtime can compare vehicle-frame acceleration with the
gravity direction. This is separate from dynamic acceleration propagation and is
disabled unless configured.

### Vehicle-Roll Prior

The optional vehicle-roll prior observes:

$$
\operatorname{roll}(q_{nv}) \approx 0 .
$$

`SensorFusion` currently enables it by default with
`r_vehicle_roll_prior = 0.1`; `0` disables it. The configured value is a variance
density and is scaled by the same observation interval as NHC. The update is
applied only at eligible NHC epochs.

The residual is `-\operatorname{roll}(q_{nv})`. Its Jacobian is computed by
finite-differencing only the vehicle-attitude error states `0..2`; direct mount
columns are zero. The update can still move mount states through covariance
coupling.

## Injection And Reset

After measurement updates, the runtime injects:

$$
\begin{aligned}
q_{nv}^+ &= q_{nv}\,\delta q(\delta\theta_v),\\
v_n^+ &= v_n + \delta v_n,\\
p_n^+ &= p_n + \delta p_n,\\
b_g^+ &= b_g + \delta b_g,\\
b_a^+ &= b_a + \delta b_a,\\
q_{bv}^+ &= \delta q(\delta\psi_{bv})\,q_{bv}.
\end{aligned}
$$

The current covariance reset applies the generated first-order reset Jacobian to
the vehicle-attitude tangent block only:

$$
G_\theta = I - \frac{1}{2}[\delta\theta_v]_\times .
$$

The reset transforms the attitude block and its cross-covariances, then
symmetrizes `P`. The mount quaternion is normalized after injection, but there
is no separate mount tangent reset block in the current implementation.

## Initialization And Mount Modes

Auto mode waits for align to produce a ready physical mount seed. The EKF yaw
seed is selected as:

1. `heading_rad`, when present;
2. GNSS course when horizontal speed is at least `max(yaw_init_speed_mps, 1.0)`;
3. otherwise `0`.

Manual mode accepts a caller-supplied `q_bv`, bypasses internal align, and
requires `heading_rad` plus speed greater than
`max(yaw_init_speed_mps, 20 / 3.6)` before EKF initialization.

At initialization, the facade sets configured roll, pitch, yaw, gyro-bias,
accelerometer-bias, and residual-mount covariance values. In automatic mode it
can copy align's 3 by 3 mount covariance block into EKF residual mount
covariance. In manual mode it seeds a tighter mount prior around the supplied
mount.

## Sleep And Reseed Model

The public facade treats a large IMU timestamp gap as a missing-sample interval,
not as a long strapdown propagation interval. It clears stream coupling, anchors
the next IMU sample as the start of a new integration interval, and then applies
a stationary sleep model to selected covariance diagonals.

For each aged diagonal covariance entry:

$$
P_{ii}^+ = P_{ii} + \sigma_\mathrm{added}^2 .
$$

Mount covariance is not aged because sleep assumes the physical mount is
unchanged. Short sleep is bounded at 15 minutes and keeps navigation usable.
Medium sleep is bounded at one hour and enters degraded dead reckoning only if
the resulting navigation covariance passes usability gates. Longer gaps, invalid
gaps, or failed usability gates enter GNSS reseed mode.

For exact public state behavior and sigma values, see
[](../runtime-state-and-persistence.md).

## Modeling Boundaries

The runtime is an inertial/GNSS estimator with optional vehicle pseudo-
observations. NHC lateral/vertical rows are instantaneous vehicle-frame velocity
constraints. In the implemented measurement model their direct mount Jacobian is
zero; mount updates are propagation- and covariance-mediated. Absolute mount
roll remains weak or ambiguous without informative motion plus a defensible
roll/bank anchor such as a flat-road prior.
