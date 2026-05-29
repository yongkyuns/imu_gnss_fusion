# Alignment Estimator Formulation

`align` estimates the physical vehicle-to-body mount quaternion $q_{bv}$ before
the runtime EKF initializes in automatic mount mode. It is a reduced estimator:
its state is the mount angle only, not the full vehicle navigation state.

It uses:

- stationary gravity to initialize and refine roll/pitch;
- GNSS-derived horizontal acceleration to constrain mount yaw;
- planar turn gyro consistency to refine roll/pitch during turns.

It does not run NHC. NHC is an EKF runtime update.

For frame and injection conventions, see [](frames.md).

## State And Covariance

The align nominal state is the mount quaternion:

$$
q_{bv},\qquad x_b = C_{bv}x_v .
$$

The covariance is a 3 by 3 covariance over a local mount error:

$$
\delta\rho =
\begin{bmatrix}
\delta\rho_x & \delta\rho_y & \delta\rho_z
\end{bmatrix}^T ,
$$

where the axes correspond to local mount roll, pitch, and yaw perturbations.

A fresh `Align::new` starts with:

$$
\begin{aligned}
\sigma_\mathrm{roll} &= 20^\circ,\\
\sigma_\mathrm{pitch} &= 20^\circ,\\
\sigma_\mathrm{yaw} &= 60^\circ .
\end{aligned}
$$

Stationary tilt initialization tightens only the tilt seed:

$$
\begin{aligned}
\sigma_\mathrm{roll} &= 10^\circ,\\
\sigma_\mathrm{pitch} &= 10^\circ,\\
\sigma_\mathrm{yaw} &= 60^\circ .
\end{aligned}
$$

Yaw remains broad until horizontal motion supplies a yaw cue.

## Prediction

Align models the physical mount as constant between windows and inflates only
covariance:

$$
P_{ii}^+ = P_{ii} + q_i^2\Delta t .
$$

When post-coarse refinement is enabled, process noise and observation standard
deviations are scaled by the configured refinement factors.

## Generic Mount Observation Rows

Gravity and turn-gyro rows use the same local linearization. For a body-frame
vector $x_b$, the predicted vehicle-frame vector is:

$$
\hat{x}_v = C_{vb}x_b .
$$

For a left small-angle mount perturbation, align uses:

$$
H_x = C_{vb}[x_b]_\times .
$$

Scalar and two-row updates use:

$$
r = z - h(q_{bv}),\qquad
S = HPH^T + R,\qquad
K = PH^T S^{-1}.
$$

The generic correction is injected as:

$$
q_{bv}^+ = \delta q(Kr)\,q_{bv}.
$$

For these generic rows the covariance update is the simplified form:

$$
P^+ = (I-KH)P,
$$

followed by symmetrization. Masked rows zero selected state columns so, for
example, turn-gyro and stationary-gravity updates do not directly update yaw.

## Stationary Tilt Initialization

The facade accumulates stationary samples before calling align's tilt
initializer. The default gates require low dynamics and either low speed or
steady motion:

$$
\operatorname{EMA}(\|\omega_b\|) \le \omega_\mathrm{max},
\qquad
\operatorname{EMA}(|\|f_b\|-g|) \le a_\mathrm{err,max},
$$

with at least 100 accumulated samples by default. When gates fail, the
accumulated tilt-initialization samples reset.

Given the mean stationary accelerometer vector $\bar{f}_b$, the initializer
constructs the vehicle down axis in body coordinates:

$$
z_v^b = -\frac{\bar{f}_b}{\|\bar{f}_b\|}.
$$

It then projects a body-frame reference axis into the plane normal to $z_v^b$:

$$
x_v^b =
\operatorname{normalize}
\left(x_\mathrm{ref} - z_v^b(z_v^b\cdot x_\mathrm{ref})\right),
\qquad
y_v^b = \operatorname{normalize}(z_v^b \times x_v^b).
$$

The resulting $C_{bv}$ is formed from these body-coordinate vehicle axes. This is
tilt-only initialization: the horizontal reference fixes an arbitrary yaw until
motion provides a real yaw cue.

## Stationary Gravity Updates

During stationary windows, align low-pass filters the body-frame gravity vector.
It then constrains vehicle-frame horizontal acceleration components toward zero
and vertical magnitude toward gravity. Roll and pitch are active; yaw is masked:

$$
\begin{aligned}
r_x &= 0 - \hat{f}_{v,x},\\
r_y &= 0 - \hat{f}_{v,y},\\
r_z &= -\|\bar{f}_b\| - \hat{f}_{v,z}.
\end{aligned}
$$

These rows can tighten tilt, but gravity alone cannot observe yaw.

## Horizontal Acceleration Yaw Update

The horizontal-acceleration path is not the generic vector update. It is a
scalar yaw-only correction.

GNSS velocities define a horizontal acceleration in a path-aligned frame:

$$
a_n = \frac{v_{n,k}-v_{n,k-1}}{\Delta t}.
$$

Using the mid-velocity direction $\hat{t}$ and its left/right horizontal normal
$\hat{\ell}$, align forms:

$$
a_\mathrm{gnss,xy} =
\begin{bmatrix}
\hat{t}^Ta_n\\
\hat{\ell}^Ta_n
\end{bmatrix}.
$$

The IMU horizontal vector is gravity-axis removed in body frame and rotated into
vehicle coordinates:

$$
a_\mathrm{imu,xy} =
\begin{bmatrix}
e_x^T C_{vb} f_{b,\mathrm{horiz}}\\
e_y^T C_{vb} f_{b,\mathrm{horiz}}
\end{bmatrix}.
$$

The scalar yaw residual is the signed 2D angle from the IMU horizontal vector to
the GNSS horizontal vector:

$$
r_\psi =
\operatorname{atan2}
\left(
a_{\mathrm{imu},x}a_{\mathrm{gnss},y}
- a_{\mathrm{imu},y}a_{\mathrm{gnss},x},
a_{\mathrm{imu},x}a_{\mathrm{gnss},x}
+ a_{\mathrm{imu},y}a_{\mathrm{gnss},y}
\right).
$$

The effective variance combines model variance, GNSS acceleration covariance,
IMU mean variance, and tilt covariance projected into horizontal-angle variance.
The update is:

$$
K_\psi = \frac{P_{\psi\psi}}{P_{\psi\psi}+R_\mathrm{eff}},
\qquad
q_{bv}^+ = q_{bv}\,q_z(-K_\psi r_\psi),
$$

$$
P_{\psi\psi}^+ = (1-K_\psi)P_{\psi\psi}.
$$

The implementation then clears yaw cross-covariances $P_{x\psi}$ and $P_{y\psi}$.

Horizontal yaw updates are gated before the scalar update is allowed. The shared
vector gate requires:

$$
v > 0.83\,\mathrm{m/s},\qquad
\|a_\mathrm{gnss,xy}\| > 0.18\,\mathrm{m/s^2},\qquad
\|a_\mathrm{imu,xy}\| > 0.18\,\mathrm{m/s^2}.
$$

One of two motion cores must also pass. A straight-line core requires dominant
longitudinal acceleration:

$$
|a_\mathrm{long}| > 0.18,\qquad
|a_\mathrm{lat}| < \max(0.5,\;0.6|a_\mathrm{long}|).
$$

A turn core requires a retained turn-consistency check plus stronger lateral
dominance:

$$
v > 2.78,\qquad
|a_\mathrm{lat}| > 0.7,\qquad
|a_\mathrm{lat}| > 1.5\max(|a_\mathrm{long}|,0.2).
$$

The effective yaw observation standard deviation starts from $1^\circ$ and is
divided by a quality factor. Straight windows use speed quality, acceleration
quality, longitudinal-acceleration quality, and lateral-rejection quality. Turn
windows use speed quality, acceleration quality, lateral-acceleration quality,
and lateral-over-longitudinal dominance, clamped to a minimum quality of `0.35`.
The final variance then adds GNSS acceleration covariance, IMU mean variance,
and tilt-covariance projection before the scalar yaw update.

## Turn Consistency And Planar Gyro Update

For each GNSS interval:

$$
\dot{\chi} =
\frac{\operatorname{wrap}(\chi_k-\chi_{k-1})}{\Delta t},
\qquad
a_\mathrm{lat} = \hat{\ell}^T\frac{v_{n,k}-v_{n,k-1}}{\Delta t}.
$$

A turn candidate requires speed, course-rate, and lateral-acceleration gates:

$$
v > 0.83\,\mathrm{m/s},\qquad
|\dot{\chi}| > 2^\circ/\mathrm{s},\qquad
|a_\mathrm{lat}| > 0.10\,\mathrm{m/s^2}.
$$

The retained-window consistency check requires at least five windows by default.
At least 80% of retained windows must agree in sign and in the model relation:

$$
a_\mathrm{lat} \approx v\,\dot{\chi}.
$$

The model-error tolerance for each retained window is:

$$
\epsilon =
\max\left(0.35,\;0.6\max(|v\dot{\chi}|,\;|a_\mathrm{lat}|)\right).
$$

The planar gyro update constrains vehicle-frame gyro roll and pitch rates toward
zero:

$$
r_\omega =
\begin{bmatrix}
0\\0
\end{bmatrix}
-
\begin{bmatrix}
\omega_{v,x}\\
\omega_{v,y}
\end{bmatrix}.
$$

Only roll and pitch mount states are active in this row; yaw is masked. This is
a refinement cue, not a heading observation.

## Readiness And Handoff

Coarse alignment readiness requires yaw to have been observed and one-sigma
covariance gates:

$$
\begin{aligned}
\sigma_\mathrm{roll} &\le 5^\circ,\\
\sigma_\mathrm{pitch} &\le 5^\circ,\\
\sigma_\mathrm{yaw} &\le 8^\circ .
\end{aligned}
$$

The `SensorFusion` facade can optionally copy align's 3 by 3 mount covariance
into the EKF residual mount block at handoff.

## Known Limitations

Because align is a reduced formulation, it cannot fully distinguish mount roll
from sustained vehicle roll or road bank. It should be treated as a seed whose
uncertainty and downstream EKF behavior must still be evaluated.
