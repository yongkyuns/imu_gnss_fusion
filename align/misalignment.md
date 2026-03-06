# Pre-Fusion IMU Mount-Alignment Estimator for Ground Vehicles

This note defines a two-stage IMU mount-alignment design for automotive use:

1. coarse alignment
2. fine alignment

The coarse stage uses only:

- `ESF-RAW` IMU gyro `\omega^b`
- `ESF-RAW` IMU accelerometer `f^b`
- `NAV2-PVT` GNSS velocity `v^n`

The fine stage starts only after a usable coarse mount estimate exists.

The intent is the bootstrap flow used by vehicle systems:

- estimate mount roll/pitch from gravity
- estimate mount yaw from turning and acceleration events
- rotate IMU data by the coarse mount estimate
- start the full GNSS/INS EKF
- optionally refine mount inside the full EKF

No `NAV-ATT` or externally supplied attitude is assumed.

---

## 1. Goal

Estimate the constant rotation between:

- vehicle frame `v`: forward-right-down
- IMU frame `b`

using only raw IMU and GNSS velocity, before the full navigation filter is trusted.

We denote the mount rotation by:

```math
C_v^b(\phi_m, \theta_m, \psi_m)
```

where:

- `\phi_m`: roll mount angle
- `\theta_m`: pitch mount angle
- `\psi_m`: yaw mount angle

The inverse rotation is:

```math
C_b^v = (C_v^b)^T
```

which rotates IMU signals into the candidate vehicle frame.

---

## 2. Two-Stage Design

### Stage A: Coarse alignment

Estimate only the three constant mount angles:

```math
x =
\begin{bmatrix}
\phi_m \\
\theta_m \\
\psi_m
\end{bmatrix}
```

This stage is intentionally reduced-state and motion-model based.

It does not estimate full vehicle attitude, position, or biases as part of the main navigation state.

### Stage B: Fine alignment

Once a sufficiently good coarse `C_b^v` exists:

- rotate IMU into the vehicle frame
- start the full GNSS/INS EKF
- let the full EKF refine navigation states and, if desired, refine mount parameters

The coarse stage is only responsible for getting close enough that the full EKF becomes reliable.

---

## 3. Coarse-Stage State

Use only the three mount angles:

```math
x_k =
\begin{bmatrix}
\phi_m \\
\theta_m \\
\psi_m
\end{bmatrix}
```

with a slow random-walk process model:

```math
x_{k+1} = x_k + w_k
```

where `w_k` is very small.

This can be implemented as:

- a recursive EKF on the three angles
- or a sliding-window nonlinear least-squares fit

Both are acceptable. The measurement structure below is the important part.

---

## 4. Available Measurements from `NAV2-PVT`

From GNSS velocity:

```math
v^n =
\begin{bmatrix}
v_N \\
v_E
\end{bmatrix}
```

compute:

### Speed

```math
s = \sqrt{v_N^2 + v_E^2}
```

### Course

```math
\chi = \operatorname{atan2}(v_E, v_N)
```

### Course rate

```math
\dot\chi_k \approx \frac{\mathrm{wrap}(\chi_k - \chi_{k-1})}{\Delta t}
```

### Horizontal acceleration

```math
a_k^n \approx \frac{v_k^n - v_{k-1}^n}{\Delta t}
```

Define unit tangent and lateral directions from GNSS velocity:

```math
t^n =
\frac{1}{s}
\begin{bmatrix}
v_N \\
v_E
\end{bmatrix}
```

```math
\ell^n =
\begin{bmatrix}
-t_E \\
t_N
\end{bmatrix}
```

Then define:

### Longitudinal acceleration

```math
a_{long} = {t^n}^T a^n
```

### Lateral acceleration

```math
a_{lat} = {\ell^n}^T a^n
```

These GNSS-derived signals are the main horizontal observability source for coarse yaw.

---

## 5. Rotate IMU into a Candidate Vehicle Frame

For a candidate mount estimate `x`, define:

```math
\omega^v = C_b^v(x)\,\omega^b
```

```math
f^v = C_b^v(x)\,f^b
```

If `x` is correct, then transformed IMU signals should look like ordinary ground-vehicle motion.

---

## 6. Ground-Vehicle Structure Used by Coarse Alignment

For normal on-road motion, use the approximate vehicle-frame structure:

```math
\omega^v \approx
\begin{bmatrix}
0 \\
0 \\
r
\end{bmatrix}
```

```math
f^v \approx
\begin{bmatrix}
a_x \\
a_y \\
g
\end{bmatrix}
```

where:

- `r` is vehicle yaw rate
- `a_x` is longitudinal acceleration
- `a_y` is lateral acceleration
- `g` is gravity in the down direction

A wrong mount estimate causes:

- yaw-rate leakage into `\omega_x^v`, `\omega_y^v`
- gravity leakage into `f_x^v`, `f_y^v`
- lateral acceleration leakage into the wrong horizontal axis
- longitudinal acceleration leakage into the wrong horizontal axis

This is the physical basis of the coarse filter.

---

## 7. Observability in the Coarse Stage

The coarse stage should be designed around what is actually observable from raw IMU and GNSS velocity.

### Roll and pitch

Roll and pitch are primarily observed from gravity:

- stationary windows
- smooth-driving low-pass accelerometer windows

### Yaw

Yaw is not robustly observable from gravity.

Yaw is only weakly constrained by:

- course-rate matching `\omega_z^v \approx \dot\chi`
- planar-turn structure `\omega_x^v \approx 0`, `\omega_y^v \approx 0`

Those terms help, but by themselves they mostly constrain the vehicle vertical axis and can leave a forward/backward ambiguity or a very weak yaw estimate.

The decisive yaw information comes from matching the transformed IMU horizontal acceleration to GNSS-derived:

- lateral acceleration `a_{lat}`
- longitudinal acceleration `a_{long}`

So the coarse stage should not rely on course-rate alone for yaw.

---

## 8. Window Types

Do not treat all samples identically. Classify short windows and only apply the measurements valid for each window.

### 8.1 Stationary windows

Use when:

- `||\omega^b||` is small
- `||\,||f^b|| - g\,||` is small

These windows are used for:

- gravity alignment
- rough roll/pitch initialization

### 8.2 Turn windows

Use when:

- speed is above threshold
- `|\dot\chi|` is above threshold
- lateral acceleration is above threshold
- GNSS velocity quality is good

These windows are used for:

- planar gyro structure
- yaw-rate matching
- lateral-acceleration matching

### 8.3 Longitudinal excitation windows

Use when:

- speed is above threshold
- `|a_{long}|` is above threshold
- reverse motion is rejected

These windows are used for:

- forward-direction disambiguation
- longitudinal-acceleration matching

---

## 9. Coarse-Stage Measurements

The coarse filter should use four pseudo-measurement blocks.

### 9.1 Gravity consistency

For low-pass accelerometer or stationary average:

```math
z_g =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
```

with:

```math
h_g(x) =
\begin{bmatrix}
f_{x,LP}^v(x) \\
f_{y,LP}^v(x)
\end{bmatrix}
```

Interpretation:

- in the correct vehicle frame, gravity should point along vehicle down
- the horizontal components should vanish

This is the main rough initializer for `\phi_m` and `\theta_m`.

### 9.2 Planar-turn gyro structure

During ordinary turning:

```math
z_\omega =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
```

with:

```math
h_\omega(x) =
\begin{bmatrix}
\omega_x^v(x) \\
\omega_y^v(x)
\end{bmatrix}
```

Interpretation:

- most vehicle angular rate should appear on vehicle `z`
- `\omega_x^v` and `\omega_y^v` should be small

This strongly helps roll/pitch and reduces bad yaw hypotheses, but it is not by itself a sufficient yaw observable.

### 9.3 GNSS course-rate vs transformed yaw-rate

During turning:

```math
z_r = \dot\chi
```

with:

```math
h_r(x) = \omega_z^v(x)
```

Interpretation:

- transformed IMU yaw-like rate should match GNSS course rate

This is a useful yaw-related bridge, but it should not be the only yaw term.

### 9.4 GNSS lateral and longitudinal acceleration matching

During turning, use:

```math
z_{lat} = a_{lat}
```

with:

```math
h_{lat}(x) = f_y^v(x)
```

During accel/brake windows, use:

```math
z_{long} = a_{long}
```

with:

```math
h_{long}(x) = f_x^v(x)
```

Interpretation:

- if the mount yaw is wrong, transformed horizontal specific force is rotated into the wrong vehicle axes
- matching `f_y^v` to GNSS lateral acceleration helps identify yaw during left/right turns
- matching `f_x^v` to GNSS longitudinal acceleration helps identify forward direction and reject the 180 degree yaw branch

These are the missing yaw observables that the coarse stage must use.

---

## 10. Recommended Batch Cost

If the coarse stage is implemented as a sliding-window nonlinear least-squares problem, use:

```math
J(\phi_m,\theta_m,\psi_m)
=
\sum_{k\in S_g} w_{g,k}\left(f_{x,LP,k}^v{}^2 + f_{y,LP,k}^v{}^2\right)
+
\sum_{k\in S_t} w_{\omega,k}\left(\omega_{x,k}^v{}^2 + \omega_{y,k}^v{}^2\right)
+
\sum_{k\in S_t} w_{r,k}\left(\omega_{z,k}^v - \dot\chi_k\right)^2
+
\sum_{k\in S_t} w_{lat,k}\left(f_{y,k}^v - a_{lat,k}\right)^2
+
\sum_{k\in S_x} w_{long,k}\left(f_{x,k}^v - a_{long,k}\right)^2
```

where:

- `S_g` are gravity/stationary windows
- `S_t` are turn windows
- `S_x` are longitudinal excitation windows

Interpretation:

- gravity term aligns vehicle down
- planar gyro term enforces turn-axis structure
- course-rate term matches GNSS turning geometry
- lateral-accel term makes yaw observable in turns
- longitudinal-accel term disambiguates forward direction

This preserves the original coarse-filter design, but makes the yaw observability story correct.

---

## 11. Recursive EKF Formulation

If a recursive filter is preferred, keep the same 3-angle state:

```math
x =
\begin{bmatrix}
\phi_m \\
\theta_m \\
\psi_m
\end{bmatrix}
```

with random-walk prediction:

```math
x_{k+1} = x_k + w_k
```

Use separate measurement updates by window class.

### 11.1 Gravity update

```math
z_g =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
,\quad
h_g(x)=
\begin{bmatrix}
f_{x,LP}^v(x) \\
f_{y,LP}^v(x)
\end{bmatrix}
```

### 11.2 Turn update

```math
z_t =
\begin{bmatrix}
0 \\
0 \\
\dot\chi \\
a_{lat}
\end{bmatrix}
```

```math
h_t(x)=
\begin{bmatrix}
\omega_x^v(x) \\
\omega_y^v(x) \\
\omega_z^v(x) \\
f_y^v(x)
\end{bmatrix}
```

### 11.3 Longitudinal excitation update

```math
z_x = a_{long}
,\quad
h_x(x)=f_x^v(x)
```

Innovation:

```math
y_k = z_k - h(x_k)
```

Jacobian:

```math
H_k = \frac{\partial h}{\partial x}\Bigg|_{x=\hat x_k}
```

For implementation, `H_k` can be computed numerically with small perturbations in:

- `\phi_m`
- `\theta_m`
- `\psi_m`

This is simple and sufficient for the coarse stage.

---

## 12. Practical Coarse-Alignment Sequence

### Step 1: Rough roll/pitch from gravity

Use stationary or low-dynamic windows:

- low-pass accelerometer
- estimate `\phi_m`, `\theta_m`
- hold `\psi_m` weakly constrained or fixed initially

### Step 2: Rough yaw from turning

Use speed-qualified turn windows:

- compute GNSS course and course rate
- compute GNSS lateral acceleration
- rotate gyro and accel with candidate mount estimate
- fit `\psi_m` so that:
  - `\omega_z^v \approx \dot\chi`
  - `\omega_x^v \approx 0`
  - `\omega_y^v \approx 0`
  - `f_y^v \approx a_{lat}`

### Step 3: Forward-direction disambiguation

Use accel/brake windows:

- compute GNSS longitudinal acceleration
- fit `\psi_m` so that `f_x^v \approx a_{long}`
- reject the 180 degree wrong-forward solution

### Step 4: Joint coarse refinement

Run the batch fit or recursive filter over all valid windows until the coarse covariance or residuals are acceptable.

---

## 13. Data Selection Rules

Use only windows that pass quality gates such as:

- speed above threshold
- GNSS velocity quality is good
- turn windows have sufficient `|\dot\chi|`
- lateral-accel windows have sufficient `|a_{lat}|`
- longitudinal windows have sufficient `|a_{long}|`
- reject reverse motion
- reject very low speed
- reject violent bumps for gravity estimation
- require both left and right turns for robust yaw convergence

The coarse stage is driven as much by window selection as by the filter equations.

---

## 14. Why Straight Driving Is Not Enough

Straight driving mostly provides gravity.

That is usually enough for:

- roll mount angle
- pitch mount angle

but not enough for robust yaw mount alignment because:

- `\dot\chi \approx 0`
- `a_{lat} \approx 0`
- horizontal excitation is weak

Repeated left and right turns help because:

- `a_{lat}` changes sign
- `\dot\chi` changes sign
- gravity remains fixed
- the estimator can separate fixed mount tilt from true vehicle dynamics

Longitudinal accel/brake events add the missing forward-direction information that removes the 180 degree yaw ambiguity.

---

## 15. Limitations

This pre-fusion estimator produces a coarse mount estimate, not a final calibration.

Its estimate can absorb errors from:

- gyro bias
- accelerometer bias
- GNSS velocity noise
- vehicle sideslip
- lever arm effects
- road bank and slope
- timing mismatch

That is acceptable at bootstrap.

The goal is only to get close enough for the full EKF to start reliably.

---

## 16. Fine Alignment Stage

After coarse alignment converges sufficiently:

1. rotate IMU measurements by `C_b^v`
2. initialize the full GNSS/INS EKF
3. let the full-state filter refine navigation states
4. optionally refine mount inside the full EKF using the better vehicle attitude and bias estimates

This is where a full quaternion `q_{sb}` or equivalent mount state belongs.

The coarse stage should remain intentionally simple and motion-model based.

---

## 17. Summary

The intended architecture is correct:

- coarse reduced-state mount estimator first
- fine full-state EKF second

For the coarse stage, the correct design is:

- use only raw IMU and GNSS velocity
- rotate IMU into a candidate vehicle frame
- enforce ground-vehicle motion structure
- estimate `[\phi_m,\theta_m,\psi_m]`

The measurements that should drive the coarse estimator are:

- gravity consistency for roll/pitch
- planar-turn gyro structure
- GNSS course-rate matching
- GNSS lateral-acceleration matching
- GNSS longitudinal-acceleration matching

The important correction is:

- course-rate and planar-turn gyro terms help yaw, but they are not enough alone
- lateral and longitudinal acceleration matching are required for robust coarse yaw identification and forward-direction disambiguation

This gives a coarse estimate suitable for handing off to the fine alignment EKF.
