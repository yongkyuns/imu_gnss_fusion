# Roll And Pitch Observability

This note is the derivation appendix for [](observability.md). It
explains why pitch is usually easier to observe than absolute mount roll in a
ground-vehicle IMU/GNSS EKF, and why NHC should not be described as a direct
mount-roll sensor.

## Implemented Instantaneous NHC

Use local NED `n`, vehicle frame `v`, and raw body frame `b`:

$$
x_n = C_{nv}x_v,\qquad x_v = C_{nv}^T x_n,
$$

$$
x_b = C_{bv}x_v,\qquad x_v = C_{bv}^T x_b.
$$

The relevant nominal state is:

$$
x =
\begin{bmatrix}
q_{nv} & v_n^T & p_n^T & b_g^T & b_a^T & q_{bv}
\end{bmatrix}^T .
$$

The implemented NHC pseudo-measurements observe vehicle-frame velocity:

$$
r_y = -e_y^T C_{nv}^T v_n,
\qquad
r_z = -e_z^T C_{nv}^T v_n .
$$

Their direct measurement Jacobian has attitude and velocity columns:

$$
H_\mathrm{nhc} =
\begin{bmatrix}
H_{\theta_v} & H_{v_n} & 0 & 0 & 0 & 0
\end{bmatrix}.
$$

In particular:

$$
H_{\psi_{bv}} = 0 .
$$

Mount correction from NHC is therefore not direct measurement sensitivity. It
comes from covariance cross terms created by propagation and other updates:

$$
K_{\psi_{bv}} =
P_{\psi_{bv},\,\theta/v}H_{\theta/v}^T S^{-1}.
$$

If the cross-covariance is weak or misleading, the NHC residual can stay small
without uniquely identifying mount roll.

## Propagation-Mediated Roll Intuition

A small roll-only local model is still useful for intuition:

$$
x_r =
\begin{bmatrix}
\delta\phi_v &
\delta\rho_b &
\delta b_{ay} &
\delta b_{az} &
\delta v_y &
\delta v_z
\end{bmatrix}^T ,
$$

where $\delta\phi_v$ is vehicle roll error and $\delta\rho_b$ is mount roll error.

For small errors, lateral and vertical vehicle-frame velocity error dynamics can
have the approximate structure:

$$
\delta\dot{v}_y \approx
g\delta\phi_v - g\delta\rho_b - \delta b_{ay} + w_y,
$$

$$
\delta\dot{v}_z \approx
-a_y\delta\rho_b - \delta b_{az} + w_z .
$$

The important invariant is the structure, not the sign convention:

- lateral velocity tends to see a difference between vehicle roll and mount
  roll;
- vertical velocity can contain a mount-roll term proportional to lateral
  acceleration;
- both channels are entangled with accelerometer bias and vehicle/road roll.

Over a short interval, this intuition may be written as interval sensitivity:

$$
\delta v_y(\Delta t) \sim
(g\delta\phi_v - g\delta\rho_b - \delta b_{ay})\Delta t,
$$

$$
\delta v_z(\Delta t) \sim
(-a_y\delta\rho_b - \delta b_{az})\Delta t.
$$

This is not the implemented instantaneous NHC row Jacobian. It describes how
prediction can create covariance coupling that later lets NHC move mount states.

## Pitch Channel

A pitch-only state has the analogous form:

$$
x_p =
\begin{bmatrix}
\delta\theta_v &
\delta\pi_b &
\delta b_{ax} &
\delta b_{az} &
\delta v_x &
\delta v_z
\end{bmatrix}^T .
$$

Pitch affects longitudinal and vertical dynamics. During acceleration/braking,
GNSS velocity changes and optional vehicle speed make pitch mount errors visible
because the forward axis is excited and externally constrained. Gravity also
provides strong tilt information during stationary periods.

This is why pitch usually converges more reliably than roll in ordinary driving.

## Information Strength

Roll information depends on motion regime:

| Regime | Roll effect |
| --- | --- |
| Stationary | gravity constrains body tilt but not yaw or road bank decomposition |
| Straight constant speed | weak roll excitation |
| Acceleration/braking | strong pitch/yaw cues, limited roll cue |
| Flat turns | lateral acceleration can help if bank is known or small |
| Banked turns | mount roll and road/vehicle roll can trade |

The practical roll nullspace is a forward-axis gauge: a change in vehicle roll
can be offset by a corresponding change in mount roll while preserving many
body-frame measurements and NHC residuals.

## Bias And Covariance Effects

Accelerometer bias and covariance coupling can make roll appear to improve
without being uniquely identified. Diagnostics should separate:

- direct measurement sensitivity;
- propagation-mediated covariance coupling;
- assumptions such as flat-road roll priors;
- real excitation from lateral acceleration and changing attitude.

## Vehicle-Roll Prior

The implemented vehicle-roll prior observes:

$$
\operatorname{roll}(q_{nv}) \approx 0 .
$$

`SensorFusion` enables this by default with variance density
$r_\mathrm{vehicle\_roll\_prior}=0.1$; $0$ disables it. It is applied only at eligible
NHC epochs and scaled by the NHC observation interval.

Its direct Jacobian is over vehicle-attitude error states only. It anchors the
vehicle-roll side of the vehicle/mount roll split, and covariance coupling can
then move mount states. This is useful on mostly flat roads, but not bank-safe:
sustained bank can be converted into mount error if the prior is trusted too
strongly.

## Replay Diagnostic

A roll diagnostic should report:

- NHC innovation and NIS by channel;
- direct NHC mount Jacobian norm;
- propagation-mediated mount correction;
- vehicle-roll prior contribution;
- mount/vehicle roll covariance and correlation;
- performance on banked and flat synthetic cases.

The key failure case to keep visible is a banked maneuver where align or EKF can
absorb true road/vehicle roll into mount roll while keeping residuals plausible.
