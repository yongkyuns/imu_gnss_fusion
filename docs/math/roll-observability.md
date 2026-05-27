# Roll And Pitch Observability

This note explains why pitch is usually easier to observe than absolute mount roll in a ground-vehicle IMU/GNSS EKF, and why NHC should not be described as a direct mount-roll sensor.

## Local EKF Context

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

NHC pseudo-measurements observe vehicle-frame velocity:

$$
z_y = 0 \approx e_y^T C_{nv}^T v_n,
\qquad
z_z = 0 \approx e_z^T C_{nv}^T v_n .
$$

These rows constrain the vehicle-motion solution. Their direct measurement Jacobian is with respect to velocity and vehicle attitude, not mount. Mount correction appears through propagation and covariance coupling.

## Roll Channel

Use a local roll-only error state:

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

where `delta phi_v` is vehicle roll error and `delta rho_b` is mount roll error.

For small errors, the lateral and vertical vehicle-frame velocity error dynamics have the approximate structure:

$$
\delta\dot{v}_y \approx
g\delta\phi_v - g\delta\rho_b - \delta b_{ay} + w_y,
$$

$$
\delta\dot{v}_z \approx
-a_y\delta\rho_b - \delta b_{az} + w_z .
$$

The important invariant is the structure, not the sign convention:

- lateral velocity sees the difference between vehicle roll and mount roll;
- vertical velocity contains a mount-roll term proportional to lateral acceleration.

Over a short interval `dt`, the NHC rows are approximately:

$$
H_y \approx
\begin{bmatrix}
-gdt & +gdt & +dt & 0 & -1 & 0
\end{bmatrix},
$$

$$
H_z \approx
\begin{bmatrix}
0 & +a_y dt & 0 & +dt & 0 & -1
\end{bmatrix}.
$$

The lateral row is strong but ambiguous. The vertical row can provide separating information during lateral acceleration, but in practice it is weak and easily entangled with bias, vehicle roll, and road bank.

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

Pitch affects longitudinal and vertical dynamics. During acceleration/braking, GNSS velocity changes and optional vehicle speed make pitch mount errors visible because the forward axis is excited and externally constrained. Gravity also provides strong tilt information during stationary periods.

This is why pitch usually converges more reliably than roll in ordinary driving.

## Information Strength

Roll information depends on motion regime:

| Regime | Roll effect |
| --- | --- |
| Stationary | gravity constrains body tilt but not yaw or road bank decomposition |
| Straight constant speed | weak roll excitation |
| Acceleration/braking | strong pitch/yaw cues, limited roll cue |
| Flat turns | lateral acceleration can help separate roll if bank is known or small |
| Banked turns | mount roll and road/vehicle roll can trade |

The practical roll nullspace is a forward-axis gauge: a change in vehicle roll can be offset by a corresponding change in mount roll while preserving many body-frame measurements and NHC residuals.

## Bias And Covariance Effects

Accelerometer bias and covariance coupling can make roll appear to improve without being uniquely identified. A lower residual or smaller covariance is not automatically proof that mount roll is observable. Diagnostics should separate:

- direct measurement sensitivity;
- propagation-mediated covariance coupling;
- assumptions such as flat-road roll priors;
- real excitation from lateral acceleration and changing attitude.

## Vehicle-Roll Prior

The implemented vehicle-roll prior observes:

$$
\operatorname{roll}(q_{nv}) \approx 0 .
$$

It is useful on mostly flat roads because it anchors the vehicle-roll side of the roll split. It is not bank-safe: sustained bank can be converted into mount error if the prior is trusted too strongly.

## Implications

The accurate public statement is:

NHC and GNSS constrain the full vehicle-motion solution. They do not directly observe absolute mount roll in all regimes. Mount roll becomes practically separable only with sufficiently informative motion, a defensible bank/vehicle-roll model, or an explicit prior such as the flat-road vehicle-roll prior.

## Replay Diagnostic

A roll diagnostic should report:

- NHC innovation and NIS by channel;
- direct NHC mount Jacobian norm;
- propagation-mediated mount correction;
- vehicle-roll prior contribution;
- mount/vehicle roll covariance and correlation;
- performance on banked and flat synthetic cases.

The key failure case to keep visible is a banked maneuver where align or EKF can absorb true road/vehicle roll into mount roll while keeping residuals plausible.
