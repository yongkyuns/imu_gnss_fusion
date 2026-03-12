# Align MEKF Derivation

This note documents the filter that is currently implemented in `align/src/align.rs`.

It is not the earlier two-stage coarse/fine design. The shipped implementation is a single reduced-state mount-alignment filter that estimates a constant vehicle-to-IMU rotation using:

- `ESF-RAW` gyro and accelerometer data
- `NAV2-PVT` GNSS velocity over each GNSS interval
- a stationary accelerometer bootstrap

The state is the constant mount rotation `q_vb`, represented by a nominal quaternion and a 3-state small-angle covariance.

## 1. Frames and Conventions

Frames:

- `v`: vehicle frame, FRD (forward, right, down)
- `b`: IMU/body frame, the raw `ESF-RAW` sensor frame
- `n`: local navigation frame used by GNSS velocity, with horizontal components `[v_N, v_E]`

The filter stores:

```math
q_{vb}
```

which represents the vehicle-to-body rotation.

The corresponding rotation matrix is:

```math
C_v^b = C(q_{vb})
```

and the inverse used for measurements is:

```math
C_b^v = (C_v^b)^T
```

The code extracts mount Euler angles from `C_v^b` using the repository's FRD convention:

```math
C_v^b = R_x(\phi) R_y(\theta) R_z(\psi)
```

with:

```math
\theta = \arcsin(-C_{31}), \quad
\phi = \operatorname{atan2}(C_{32}, C_{33}), \quad
\psi = \operatorname{atan2}(C_{21}, C_{11})
```

## 2. State and Error-State Model

The nominal state is only the mount quaternion:

```math
\hat q_{vb}
```

The covariance is defined on a 3-vector small-angle error state:

```math
\delta x = \delta\theta \in \mathbb{R}^3
```

So the filter covariance is:

```math
P \in \mathbb{R}^{3\times 3}
```

The mount is assumed constant, so the process model is a random walk on the error-state:

```math
\delta\theta_{k+1} = \delta\theta_k + w_k
```

with diagonal process noise from configuration:

```math
Q_c = \operatorname{diag}(\sigma_{q,roll}^2, \sigma_{q,pitch}^2, \sigma_{q,yaw}^2)
```

The implementation applies this in discrete time as:

```math
P_{k+1|k} = P_{k|k} + Q_c \, \Delta t
```

with no nominal-state propagation beyond keeping `q_vb` constant.

## 3. Stationary Initialization

Initialization uses a set of stationary accelerometer samples:

```math
f_i^b, \quad i = 1,\dots,N
```

and forms the mean:

```math
\bar f^b = \frac{1}{N} \sum_{i=1}^N f_i^b
```

At rest, specific force is approximately opposite gravity, so the vehicle down axis expressed in the body frame is seeded as:

```math
\hat z_v^b = -\frac{\bar f^b}{\|\bar f^b\|}
```

A reference horizontal axis is then constructed by projecting a fixed reference vector onto the plane orthogonal to `\hat z_v^b`.

First try body x:

```math
x_{ref} = [1,0,0]^T
```

and project:

```math
\tilde x_v^b = x_{ref} - \hat z_v^b (\hat z_v^{bT} x_{ref})
```

If degenerate, use body y instead. Normalize:

```math
\hat x_v^b = \frac{\tilde x_v^b}{\|\tilde x_v^b\|}
```

Then form:

```math
\hat y_v^b = \frac{\hat z_v^b \times \hat x_v^b}{\|\hat z_v^b \times \hat x_v^b\|}
```

and re-orthogonalize:

```math
\hat x_v^b = \hat y_v^b \times \hat z_v^b
```

This gives the initial vehicle-to-body rotation:

```math
\hat C_v^b = [\hat x_v^b\; \hat y_v^b\; \hat z_v^b]
```

where the columns are the vehicle axes expressed in the body frame.

Roll and pitch come from `\hat C_v^b`. Yaw is not observable from stationary accelerometer data, so the implementation overwrites the initialized yaw with a supplied seed `\psi_0`:

```math
\hat q_{vb,0} = q(\hat\phi, \hat\theta, \psi_0)
```

The covariance is reset to a smaller diagonal prior after initialization.

## 4. GNSS-Window Summary

The update operates once per GNSS interval using:

```math
\Delta t,
\bar\omega^b,
\bar f^b,
 v_{k-1}^n,
 v_k^n
```

stored in `AlignWindowSummary`.

The IMU part is a simple average over all `ESF-RAW` packets in the interval:

```math
\bar\omega^b = \frac{1}{M} \sum_{j=1}^M \omega_j^b,
\qquad
\bar f^b = \frac{1}{M} \sum_{j=1}^M f_j^b
```

The filter also maintains a low-pass accelerometer state used by the stationary gravity update:

```math
f_{g,lp,k}^b = (1-\alpha) f_{g,lp,k-1}^b + \alpha \, \bar f_k^b
```

## 5. GNSS-Derived Kinematics

From the two GNSS velocity samples:

```math
v_{k-1}^n = [v_{N,k-1}, v_{E,k-1}, v_{D,k-1}]^T,
\qquad
v_k^n = [v_{N,k}, v_{E,k}, v_{D,k}]^T
```

compute horizontal speeds:

```math
s_{k-1} = \sqrt{v_{N,k-1}^2 + v_{E,k-1}^2},
\qquad
s_k = \sqrt{v_{N,k}^2 + v_{E,k}^2}
```

and the mid-speed:

```math
s_{mid} = \frac{s_{k-1} + s_k}{2}
```

Course angles:

```math
\chi_{k-1} = \operatorname{atan2}(v_{E,k-1}, v_{N,k-1}),
\qquad
\chi_k = \operatorname{atan2}(v_{E,k}, v_{N,k})
```

Course rate:

```math
\dot\chi_k = \frac{\operatorname{wrap}(\chi_k - \chi_{k-1})}{\Delta t}
```

Navigation-frame acceleration estimate:

```math
a_k^n = \frac{v_k^n - v_{k-1}^n}{\Delta t}
```

Mid-interval horizontal velocity:

```math
v_{mid,h}^n = \frac{1}{2}
\begin{bmatrix}
v_{N,k-1} + v_{N,k} \\
v_{E,k-1} + v_{E,k}
\end{bmatrix}
```

If `v_{mid,h}^n` is nonzero, define tangent and lateral unit vectors:

```math
t^n = \frac{v_{mid,h}^n}{\|v_{mid,h}^n\|},
\qquad
\ell^n = \begin{bmatrix} -t_E \\ t_N \end{bmatrix}
```

Then longitudinal and lateral accelerations are:

```math
a_{long} = t^{nT} a_h^n,
\qquad
a_{lat} = \ell^{nT} a_h^n
```

where `a_h^n = [a_N, a_E]^T`.

## 6. Motion Classification and Gating

The implementation classifies the current GNSS interval into three possible update types.

### Stationary

```math
\|\bar\omega^b\| \le \omega_{stat,max}
```

```math
\left| \|\bar f^b\| - g \right| \le a_{stat,max}
```

```math
s_{mid} < 0.5 \text{ m/s}
```

### Turn-valid

```math
s_{mid} > s_{min}
```

```math
|\dot\chi_k| > \dot\chi_{min}
```

```math
|a_{lat}| > a_{lat,min}
```

### Longitudinal-valid

```math
s_{mid} > s_{min}
```

```math
|a_{long}| > a_{long,min}
```

```math
|a_{lat}| < \max(0.5, 0.6 |a_{long}|)
```

Only the enabled and valid measurements are fused in each window.

## 7. Observation Model

For a candidate mount quaternion `q_vb`, rotate averaged gyro and accelerometer into the candidate vehicle frame:

```math
\omega^v = C_b^v(q_{vb}) \, \bar\omega^b
```

```math
f^v = C_b^v(q_{vb}) \, \bar f^b
```

The full observation vector used by the implementation is:

```math
h(q_{vb}) =
\begin{bmatrix}
\omega_x^v \\
\omega_y^v \\
\omega_z^v \\
f_x^v \\
f_y^v \\
f_z^v
\end{bmatrix}
```

The filter never fuses all six rows at once. It selects subsets depending on motion class.

## 8. Measurement Equations Actually Used

### 8.1 Stationary gravity pseudo-measurement

The current implementation uses only the horizontal accelerometer components of the low-pass gravity estimate:

```math
z_g = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
```

```math
h_g(q_{vb}) = \begin{bmatrix} f_x^v \\ f_y^v \end{bmatrix}
```

with `f^b = f_{g,lp}^b`.

This enforces that gravity should lie on the vehicle z-axis, but it does not explicitly fuse `f_z^v \approx -g`.

### 8.2 Turn gyro pseudo-measurement

During turning windows:

```math
z_{turn,gyro} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
```

```math
h_{turn,gyro}(q_{vb}) = \begin{bmatrix} \omega_x^v \\ \omega_y^v \end{bmatrix}
```

This encodes the planar-vehicle assumption that body rotation is predominantly about vehicle z.

### 8.3 Course-rate measurement

```math
z_{course} = \dot\chi_k
```

```math
h_{course}(q_{vb}) = \omega_z^v
```

So the transformed gyro z component is matched to GNSS course rate.

### 8.4 Lateral-acceleration measurement

```math
z_{lat} = a_{lat}
```

```math
h_{lat}(q_{vb}) = f_y^v
```

This uses the transformed accelerometer lateral component as a proxy for GNSS-derived lateral acceleration.

### 8.5 Longitudinal-acceleration measurement

```math
z_{long} = a_{long}
```

```math
h_{long}(q_{vb}) = f_x^v
```

This uses the transformed accelerometer longitudinal component as a proxy for GNSS-derived longitudinal acceleration.

## 9. Analytical Jacobian Used by the Code

For any body-frame vector `u^b`, define:

```math
u^v = C_b^v(q_{vb}) u^b
```

The filter uses a left-multiplicative small-angle correction:

```math
q_{vb}^{+} = \delta q(\delta\theta) \otimes q_{vb}
```

with small-angle quaternion:

```math
\delta q(\delta\theta) \approx \begin{bmatrix} 1 \\ \tfrac{1}{2}\delta\theta \end{bmatrix}
```

For this convention, the implementation uses the linearization:

```math
\delta \nu^v \approx C_b^v [u^b]_{\times} \, \delta\theta
```

where `[u]_{\times}` is the skew-symmetric cross-product matrix.

So the stacked Jacobian for the 6-vector observation is:

```math
H(q_{vb}) =
\begin{bmatrix}
C_b^v [\bar\omega^b]_{\times} \\
C_b^v [\bar f^b]_{\times}
\end{bmatrix}
\in \mathbb{R}^{6\times 3}
```

The implementation forms this as:

```math
H_\omega = C_b^v [\bar\omega^b]_{\times}
```

```math
H_f = C_b^v [\bar f^b]_{\times}
```

and then selects the required rows for each scalar or 2-vector update.

## 10. EKF Update Equations

For a selected measurement subset:

```math
z = h(q_{vb}) + v,
\qquad v \sim \mathcal N(0,R)
```

with residual:

```math
y = z - h(\hat q_{vb})
```

and selected Jacobian `H`, the filter computes:

```math
S = H P H^T + R
```

```math
K = P H^T S^{-1}
```

```math
\delta\theta = K y
```

and injects the correction multiplicatively:

```math
\hat q_{vb}^{+} = \operatorname{normalize}(\delta q(\delta\theta) \otimes \hat q_{vb})
```

Covariance update:

```math
P^{+} = (I - K H) P
```

followed by explicit symmetrization in code.

The implementation uses only:

- `1 x 1` innovation inversions for scalar updates
- `2 x 2` innovation inversions for vector updates

## 11. Measurement Noise Mapping

The configuration maps directly to the measurement covariances:

### Gravity update

```math
R_g = \sigma_g^2 I_2
```

### Turn-gyro update

```math
R_{turn,gyro} = \sigma_{turn,gyro}^2 I_2
```

### Course-rate update

```math
R_{course} = \sigma_{course}^2
```

### Lateral update

```math
R_{lat} = \sigma_{lat}^2
```

### Longitudinal update

```math
R_{long} = \sigma_{long}^2
```

## 12. What the Current Filter Is and Is Not

The current implementation is:

- a single-stage reduced-state mount-alignment MEKF
- windowed on GNSS intervals
- driven by averaged IMU and GNSS-derived pseudo-measurements
- initialized by stationary accelerometer tilt plus a yaw seed

It is not:

- a full navigation EKF
- a fine-alignment filter with navigation attitude in the state
- a batch optimizer
- a two-stage coarse/fine pipeline in the current code

## 13. Important Implementation Consequence

Because the stationary update uses only:

```math
f_x^v \approx 0, \qquad f_y^v \approx 0
```

and not the full signed gravity vector, the current implementation constrains the gravity direction only through horizontality, not through an explicit `f_z^v \approx -g` residual.

That is a property of the current code, not a theoretical recommendation.

Any redesign discussion should start from this exact measurement set, because it determines the current observability and branch behavior seen in replay.
