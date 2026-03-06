# Nonlinear Align EKF (VMA-Inspired)

This note defines a stripped-down EKF for IMU installation-angle (align) estimation.
It keeps the idea of Velocity Matching Alignment (VMA): use GNSS velocity (and optional NHC) to observe align.
Unlike the linear small-angle VMA in Shin 4.2.3, this uses a quaternion state so it is not tied to a fixed small-angle global model.

## 1. Goal

Estimate sensor-to-body rotation while moving:

- body frame: `b` (vehicle frame used by navigation constraints)
- sensor frame: `s` (IMU measurement frame)
- navigation frame: `n` (NED)

Primary output:

- `q_sb`: quaternion rotating vector from body to sensor
- equivalent Euler angles for reporting (intrinsic ZYX convention)

Optional outputs:

- gyro bias `b_g` (sensor frame)
- accel bias `b_a` (sensor frame)

## 2. State

Use a nominal + error-state structure.

Nominal state:

```text
x_nom = [ q_sb, b_g, b_a ]
```

Error state (minimal):

```text
delta_x = [ delta_theta, delta_b_g, delta_b_a ]  in R^9
```

where `delta_theta` is a 3D small rotation in tangent space (only local linearization).

## 3. Inputs and Known Quantities

- IMU increments or rates in sensor frame:
  - gyro: `omega_s_meas`
  - accel: `f_s_meas` (specific force)
- GNSS velocity in NED: `v_gnss_n`
- Optional wheel speed / NHC signals.
- Current attitude of body in NED from mechanization/host filter:
  - `R_nb` or `q_nb` (required to map predicted body-frame quantities to NED).

Notes:

- This filter estimates mounting/align; it does not replace full navigation mechanization.
- If `q_nb` is unavailable, a coupled formulation with navigation states is required (larger filter).

## 4. Process Model

Use random-walk model for align and biases:

```text
q_sb,k+1 = q_sb,k * Exp( 0.5 * w_theta * dt )
b_g,k+1  = b_g,k + w_bg * dt
b_a,k+1  = b_a,k + w_ba * dt
```

where:

- `w_theta`, `w_bg`, `w_ba` are zero-mean Gaussian process noises
- `Exp(.)` is quaternion exponential map from so(3)

Typical assumption:

- align is nearly constant -> very small `Q_theta`
- biases are slow random walks -> small `Q_bg`, `Q_ba`

## 5. Measurement Model (Velocity Matching)

### 5.1 Core idea

Use transformed IMU data and current `q_nb` to predict navigation-frame velocity behavior, then compare to GNSS velocity.

One practical residual form at update time `k`:

```text
r_v = v_pred_n(q_sb, b_g, b_a) - v_gnss_n
```

with `v_pred_n` from short-horizon integration between GNSS updates.

IMU transform chain:

```text
omega_b = R_bs * (omega_s_meas - b_g)
f_b     = R_bs * (f_s_meas - b_a)
```

where `R_bs = R_sb^T`.

Then map to navigation frame:

```text
f_n = R_nb * f_b
```

and propagate velocity (discrete):

```text
v_pred_n,k+1 = v_pred_n,k + (f_n + g_n + coriolis_terms) * dt
```

For a simplified implementation, coriolis/transport terms may be omitted first, then added.

### 5.2 Optional NHC residuals

For wheeled vehicles at moderate speed:

```text
r_nhc = [ v_b,y, v_b,z ]^T - [0, 0]^T
```

with:

```text
v_b = R_bn * v_pred_n
```

This improves observability of roll/pitch and some bias terms.

### 5.3 Optional wheel-speed residual

```text
r_ws = ||v_b,xy|| - v_wheel
```

## 6. EKF Update on Manifold

Linearize residuals around nominal:

```text
r ~= h(x_nom) + H * delta_x
```

Then standard EKF:

```text
S = H P H^T + R
K = P H^T S^-1
delta_x_hat = K * r
P <- (I - K H) P
```

Inject correction:

```text
q_sb <- Exp(0.5 * delta_theta_hat) * q_sb
b_g  <- b_g + delta_b_g_hat
b_a  <- b_a + delta_b_a_hat
```

Reset error state to zero and continue.

## 7. Why this avoids the 4.2.3 limitation

Shin 4.2.3 uses a globally linear small-angle model for attitude error states.
Here:

- global attitude/align is represented by quaternion (`q_sb`)
- only local correction `delta_theta` is linearized each step
- relinearization happens every update

So large initial align can be handled better, provided excitation and initialization are reasonable.

## 8. Observability and Practical Constraints

- Roll/pitch are usually observable from gravity + vehicle motion.
- Yaw align is weak without horizontal excitation (turns, accelerations, non-collinear maneuvers).
- Bias and align are coupled; tune process noise to avoid bias absorbing mounting error.
- Use gated updates:
  - speed threshold for yaw-sensitive updates
  - reject near-zero-dynamics windows for heading observability

## 9. Coordinate and Rotation Convention (must be fixed)

Use one unambiguous convention throughout:

- `q_sb`: body -> sensor
- Euler reporting in intrinsic ZYX:
  - `R_sb = Rz(yaw) * Ry(pitch) * Rx(roll)` (intrinsic sequence)
- sensor -> body transform is transpose/inverse:
  - `R_bs = R_sb^T`

A convention mismatch here will dominate all tuning.

## 10. Suggested Integration Plan in This Repo

1. New crate/module: `misalign_ekf` (or inside `ekf` crate initially).
2. Inputs:
   - IMU at prediction rate
   - GNSS velocity updates
   - `q_nb` from existing navigation EKF for now
3. Outputs:
   - `q_sb` and Euler ZYX
   - bias estimates
   - covariance diagonals
4. Visualization:
   - compare estimated mounting angles vs ESF-ALG (reference only, not fused)
5. Validation:
   - synthetic truth tests (known fixed align)
   - replay logs with different mounts
   - convergence time and steady-state error metrics

## 11. Minimal Equivalence to VMA Idea

If you want the closest conceptual mapping to 4.2.3 while staying nonlinear:

- keep only `[q_sb, b_g, b_a]`
- use velocity-matching residuals as the main update
- avoid full position states in this alignment filter

This preserves the "alignment by velocity matching" principle without locking the model to a global small-angle linear form.
