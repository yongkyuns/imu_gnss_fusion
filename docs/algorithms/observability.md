# Mount Observability

Mount estimation is split between a reduced alignment estimator and the EKF
runtime. The important distinction is direct measurement sensitivity versus
propagation-mediated correction through covariance.

For the broader state-design discussion, including why mount is estimated in
the EKF and why AHRS-only attitude filters are insufficient during vehicle
dynamics, see [](mount-states.md).

For the detailed roll-channel derivation, see [](roll-observability.md).

## What IMU/GNSS Can See

GNSS velocity changes and raw IMU angular/specific-force signals make mount
pitch and yaw visible during ordinary acceleration, braking, and turning. Mount
roll is weaker because ground-vehicle motion can trade vehicle roll, road bank,
and mount roll while producing similar body-frame measurements.

In practical terms:

| Motion regime | Mount information |
| --- | --- |
| Stationary | gravity constrains body tilt but not yaw or road-bank decomposition |
| Straight constant speed | weak mount excitation |
| Acceleration/braking | strong pitch and yaw cues |
| Flat turns | lateral acceleration can improve roll conditioning if bank is known or small |
| Banked turns | vehicle roll/bank and mount roll can trade |

## Implemented NHC Rows

Nonholonomic constraints enforce vehicle-frame velocity consistency:

$$
\begin{aligned}
v_y^v &\approx 0,\\
v_z^v &\approx 0.
\end{aligned}
$$

The implemented instantaneous rows are:

$$
\begin{aligned}
r_y &= -e_y^T C_{nv}^T v_n,\\
r_z &= -e_z^T C_{nv}^T v_n.
\end{aligned}
$$

Their direct Jacobian has attitude and velocity sensitivity, but no residual
mount columns:

$$
H_\mathrm{mount}=0.
$$

The full row expression is listed in [](ekf-matrices.md).

Mount states can still be corrected through Kalman gain if prediction and prior
updates have created cross-covariance between mount and the observed
attitude/velocity states. That is not the same as NHC directly measuring mount
roll.

## Vehicle-Roll Prior

The runtime includes a soft flat-road prior at eligible NHC epochs:

$$
\operatorname{roll}(q_{nv}) \approx 0.
$$

`SensorFusion` currently enables it by default with variance density
$r_\mathrm{vehicle\_roll\_prior}=0.1$; setting it to $0$ disables the update. The
variance is scaled by the same observation interval as NHC.

The residual is:

$$
r_\phi = -\operatorname{roll}(q_{nv}).
$$

The SymPy-generated Jacobian has nonzero entries only over vehicle-attitude
error states $0{:}2$. Direct mount columns are zero, so any mount motion comes
through covariance coupling.

This prior reduces roll ambiguity on mostly flat roads by anchoring the
vehicle-roll side of the roll split. It is not bank-safe: sustained banked roads
can be converted into mount error if the prior is trusted too strongly.

## Why Roll Remains Ambiguous

The practical roll nullspace is a forward-axis gauge. A change in vehicle roll
can be offset by a corresponding change in mount roll while preserving many
body-frame measurements and NHC residuals.

Lateral acceleration can improve practical conditioning under a flat-road or
known-bank assumption, but it does not remove the gauge by itself. Therefore the
public claim should be narrow:

NHC and GNSS constrain the full vehicle-motion solution. They do not provide
bank-safe absolute mount-roll observability using only IMU/GNSS. Mount roll
becomes practically separable only with sufficiently informative motion plus an
external or assumed roll/bank anchor, such as the flat-road vehicle-roll prior.

This is consistent with the closest smartphone-telematics prior work: Wahlstrom
2017 augments a GNSS-aided INS with phone-to-vehicle orientation and NHC, but
also introduces a roll pseudo-observation because GNSS plus NHC alone do not
fade out the initial phone-to-vehicle roll ambiguity. See
[](../reference/prior-work.md) for the full comparison and for why some
state-coordinate choices show direct NHC mount Jacobian columns while the
implemented rows here do not.

## Alignment Versus EKF

Align is a reduced formulation with mount angles as its main states. It can seed
the EKF well when the motion excites the mount, but it cannot fully separate
mount roll from vehicle roll/bank by itself.

The EKF has a richer dynamic state and covariance coupling, but it can still
inherit a poor seed when the data does not identify the same decomposition. A
low NHC residual or smaller covariance is not by itself proof that mount roll is
uniquely observable.
