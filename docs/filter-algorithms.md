# Filter Algorithms

This document summarizes the estimator implementations in `sensor_fusion`.
It complements the PDF/TEX derivations in `docs/align.tex`,
`docs/reduced.tex`, and `docs/full.tex` with source-level behavior from:

- `sensor_fusion/src/align/`
- `sensor_fusion/src/reduced/`
- `sensor_fusion/src/full/`
- `sensor_fusion/src/noise.rs`
- `sensor_fusion/src/covariance.rs`
- generated wrappers and generated fragments under `sensor_fusion/src/*/generated*`

The high-level public runtime is `SensorFusion` in
`sensor_fusion/src/fusion.rs`. It owns alignment, GNSS anchoring,
Reduced/Full initialization, manual mount handling, nonholonomic-constraint
scheduling, and optional vehicle-speed updates.

## Shared Frames And Quaternion Conventions

The project uses active rotations:

```text
x_a = C_ab x_b
R(q_ab) = C_ab
R(q1 * q2) = R(q1) R(q2)
```

Frame names:

| Frame | Meaning |
| --- | --- |
| `b` | Raw IMU body/sensor frame. Public IMU angular rates and specific force are expressed here. |
| `v` | Vehicle frame, forward-right-down. Vehicle speed and NHC are expressed here. |
| `n` | Local NED navigation frame used by Reduced and by GNSS velocity inputs to the facade. |
| `e` | ECEF frame used by Full. |

The public physical mount is always the vehicle-to-body quaternion `q_bv`:

```text
C_bv = R(q_bv)
x_b = C_bv x_v
C_vb = C_bv^T
x_v = C_vb x_b
```

The legacy Rust fields `qcs0..qcs3` in Reduced and Full store this physical
`q_bv`. The name is historical; current mainline code should treat it as
vehicle-to-body, not as a separate "car-to-sensor" convention.

Reduced attitude is `q_nv`, with `R(q_nv) = C_nv` and `x_n = C_nv x_v`.
Full attitude is `q_ev`, with `R(q_ev) = C_ev` and `x_e = C_ev x_v`.

## Runtime Pipeline

`SensorFusion` supports two runtime filter families:

| Mode | Filter |
| --- | --- |
| `Filter::Reduced` | Local-NED EKF in `sensor_fusion/src/reduced`. This is the default. |
| `Filter::Full` | ECEF full-state EKF in `sensor_fusion/src/full`. Used for comparison, diagnostics, and optional runtime selection. |

The mount source is selected separately:

| Mode | Behavior |
| --- | --- |
| `MountMode::Auto` | The internal `Align` filter estimates an initial `q_bv`. The selected runtime filter then initializes with that mount and may continue estimating mount states. |
| `MountMode::Manual(q_bv)` | The caller supplies a fixed physical `q_bv`. Reduced and Full set the mount quaternion and freeze their internal mount states. |

`SensorFusion::process_imu` and `SensorFusion::process_gnss` feed both
alignment and the selected runtime. In Reduced mode, GNSS may be staged and
fused at an IMU epoch so GNSS and NHC rows can be batched. In Full mode, GNSS
position/velocity and NHC rows are batched by the Full reference update path.

The runtime also supports optional vehicle speed observations through
`process_vehicle_speed`. Forward/reverse samples observe vehicle-frame
`v_x`. Unknown-direction low speeds may become zero-velocity updates.

## Align Filter

### Role

`Align` is a standalone mount estimator in `sensor_fusion/src/align/mod.rs`.
It estimates only `q_bv` and a 3 by 3 covariance over mount roll, pitch, and
yaw small-angle errors. It is used to produce the coarse mount seed for
Reduced or Full in auto mount mode.

It is intentionally not the same EKF as Reduced or Full. It does not integrate
navigation state and it does not estimate IMU biases.

### State

Nominal state:

```text
q_bv
```

Error state and covariance:

```text
delta x_align = [droll, dpitch, dyaw]
P_align: 3 x 3
```

Default construction starts at identity mount with covariance:

```text
roll sigma  = 20 deg
pitch sigma = 20 deg
yaw sigma   = 60 deg
```

`initialize_from_stationary` computes roll and pitch from stationary
accelerometer samples, applies a supplied yaw seed, and resets the covariance
to tight roll/pitch and wide yaw:

```text
roll sigma  = 0.2 deg
pitch sigma = 0.2 deg
yaw sigma   = 60 deg
```

### Stationary Bootstrap

The stationary bootstrap averages body-frame accelerometer samples. At rest,
specific force is treated as opposite vehicle down, so the vehicle down axis
in body coordinates is initialized from `-mean_accel / norm(mean_accel)`.
The vehicle forward axis is chosen by projecting a body reference axis onto
the plane orthogonal to vehicle down; a fallback reference axis is used if the
projection degenerates. The supplied yaw seed is then applied as a
vehicle-frame yaw adjustment.

### Prediction

Align assumes the mount is constant over a window. Prediction leaves the
nominal quaternion unchanged and only adds diagonal random-walk variance:

```text
P_ii <- P_ii + q_mount_std_rad[i]^2 * dt
```

If post-coarse refinement is active, `refine_process_noise_scale` scales the
process standard deviation before squaring.

### Measurement Families

`update_window_with_trace` consumes an `AlignWindowSummary`:

```text
dt
mean_gyro_b
mean_accel_b
gnss_vel_prev_n
gnss_vel_curr_n
```

It derives speed, course-rate, GNSS acceleration, longitudinal acceleration,
lateral acceleration, stationary status, and turn consistency.

The implemented observation families are:

| Observation | What It Constrains | Important Gates |
| --- | --- | --- |
| Stationary gravity | Roll and pitch from gravity direction in body frame. Yaw is masked. | Low gyro norm, accelerometer norm near gravity, near-zero speed. |
| Horizontal acceleration heading | Vehicle yaw from comparing GNSS horizontal acceleration direction to IMU-derived horizontal acceleration in the leveled vehicle frame. | Minimum speed, minimum acceleration, straight-line or consistent-turn gates. |
| Turn gyro | Roll/pitch rates from planar turning windows. Optional yaw effect is scaled by `turn_gyro_yaw_scale`. | Minimum speed, course rate, lateral acceleration, and turn consistency for yaw use. |

The common linearization rotates a body vector into the vehicle frame with
`C_vb = C_bv^T`. For small left perturbations of `q_bv`, the rows are of the
form:

```text
H = C_vb [body_vector]_x
```

Rows can be masked so gravity does not update yaw and so turn-gyro yaw can be
disabled or downscaled.

### Coarse Readiness

Align reports coarse alignment ready only when yaw has been observed and all
three covariance sigmas are at or below `0.15 deg`:

```text
sigma_roll  <= 0.15 deg
sigma_pitch <= 0.15 deg
sigma_yaw   <= 0.15 deg
```

`SensorFusion` can additionally impose `align_handoff_delay_s` before using
that mount.

### Important Align Tuning Parameters

`AlignConfig` fields are one-sigma standard deviations unless noted:

| Parameter | Purpose |
| --- | --- |
| `q_mount_std_rad` | Per-axis mount random-walk standard deviation. Default yaw random walk is lower than roll/pitch. |
| `r_gravity_std_mps2` | Stationary gravity observation noise. |
| `r_horiz_heading_std_rad` | Straight-line horizontal acceleration heading noise. |
| `r_turn_heading_std_rad` | Turn-derived heading noise. |
| `r_turn_gyro_std_radps` | Turn-gyro observation noise. |
| `turn_gyro_yaw_scale` | How much turn-gyro updates may affect yaw. Default is zero. |
| `gravity_lpf_alpha` | Low-pass coefficient for stationary gravity samples. |
| `min_speed_mps`, `min_turn_rate_radps`, `min_lat_acc_mps2`, `min_long_acc_mps2` | Motion gates for GNSS-derived observations. |
| `turn_consistency_*` | Windowed consistency checks comparing lateral acceleration and speed times course rate. |
| `max_stationary_gyro_radps`, `max_stationary_accel_norm_err_mps2` | Stationary detection gates. |
| `refine_after_coarse_ready`, `refine_process_noise_scale`, `refine_observation_std_scale` | Optional lower-bandwidth refinement after coarse readiness. |

### Align Limitations

Align can be wrong when the first useful motion windows are weak,
ambiguous, or inconsistent with the vehicle model. Stationary data does not
observe yaw. Straight acceleration mainly helps heading/pitch-like alignment
and is weaker for roll. Turning is the strongest generic source for yaw/roll
evidence, but bad GNSS velocity, sideslip, or poor turn consistency can still
create a bad seed. Downstream covariance should reflect seed uncertainty; a
single hard-coded mount covariance can be overconfident on bad starts.

## Reduced Filter

### Role

Reduced is the default runtime EKF. It keeps a local NED navigation state,
body-frame additive IMU corrections, and the physical vehicle-to-body mount.
It is smaller than Full and has richer update diagnostics.

### Nominal State

`sensor_fusion/src/reduced/types.rs` stores:

```text
q_nv
v_n
p_n
b_g_b
b_a_b
q_bv
```

Rust field layout:

| Fields | Meaning |
| --- | --- |
| `q0..q3` | `q_nv`, vehicle-to-NED attitude. |
| `vn, ve, vd` | NED velocity in m/s. |
| `pn, pe, pd` | Local NED position in m from the active anchor. |
| `bgx..bgz` | Additive gyro correction in raw body frame, rad/s. |
| `bax..baz` | Additive accelerometer correction in raw body frame, m/s^2. |
| `qcs0..qcs3` | Physical vehicle-to-body mount `q_bv`. |

### Error State

The generated Reduced model has 18 error states:

```text
[dtheta_v, dv_n, dp_n, dbg_b, dba_b, dpsi_bv]
```

Index order:

| Indices | Error Component |
| --- | --- |
| `0..2` | Vehicle-attitude small angle. |
| `3..5` | NED velocity error. |
| `6..8` | Local NED position error. |
| `9..11` | Gyro correction error. |
| `12..14` | Accelerometer correction error. |
| `15..17` | Mount small-angle error for `q_bv`. |

### Initialization

`Filter::init_nominal_from_gnss` sets attitude from the caller-supplied
`q_nv`, copies GNSS position and velocity, resets the mount to identity unless
the facade overwrites it, and initializes a diagonal covariance. Through the
facade, the mount is normally set from Align in auto mode or from the manual
mount in manual mode.

Important initialization defaults in the Reduced source/facade include:

| Setting | Default |
| --- | --- |
| Attitude roll/pitch sigma | `2 deg` in the facade. |
| Yaw sigma | `6 deg` in the facade. |
| Gyro-bias sigma | `0.125 deg/s`. |
| Accelerometer-bias sigma | `0.15 m/s^2`. |
| Mount roll/pitch sigma | `1.2 deg` in the facade. |
| Mount yaw sigma | `6 deg` in the facade. |

The lower-level `init_nominal_from_gnss` uses an internal `10 deg` residual
mount sigma if mount states are not frozen. The facade can replace covariance
blocks after initialization for its configured mount sigmas and settle policy.

### Propagation

Reduced consumes one body-frame IMU delta:

```text
dtheta_b, dvel_b, dt
```

The nominal prediction is generated from
`sensor_fusion/src/reduced/formulation.py` and included through
`sensor_fusion/src/reduced/generated.rs`. Conceptually:

```text
dtheta_b_corr = dtheta_b - b_g * dt
dvel_b_corr   = dvel_b   - b_a * dt
dtheta_v      = C_vb dtheta_b_corr
dvel_v        = C_vb dvel_b_corr
q_nv          <- normalize(q_nv * dq(dtheta_v))
v_n           <- v_n + C_nv dvel_v + g_n dt
p_n           <- p_n + v_n_old dt
```

`SensorFusion` adds local navigation-rate/Coriolis corrections around the
generated Reduced prediction. It also reanchors the local NED origin when the
position moves far enough from the current anchor.

The covariance prediction uses generated `F` and `G` matrices:

```text
P <- F P F^T + G Q G^T
```

The shared sparse covariance helper in `covariance.rs` evaluates this using
generated row-support metadata. Reduced uses `SparseCovariancePolicy::REDUCED`.

Runtime process-noise assembly in `reduced::Filter::predict` is source-specific:

| Noise Block | Runtime Diagonal Entry |
| --- | --- |
| Gyro white noise | `gyro_var * dt` |
| Accelerometer white noise | `accel_var * dt` |
| Gyro-bias random walk | `gyro_bias_rw_var / dt` because generated bias-RW columns already include `dt`. |
| Accelerometer-bias random walk | `accel_bias_rw_var / dt` for the same generated-column convention. |
| Mount random walk | `mount_align_rw_var_axis(i) * dt`, or zero when mount states are frozen. |

### Reduced Update Families

Reduced supports scalar updates and sequential batches. After every accepted
update it injects the accumulated error state and applies reset conventions.

| Update | Source Function | Residual |
| --- | --- | --- |
| GNSS position | `fuse_gps`, `fuse_gps_nhc_batch` | `z_p_ned - p_n` per axis. |
| GNSS velocity | `fuse_gps`, `fuse_gps_nhc_batch` | `z_v_ned - v_n` per axis. |
| Zero velocity | `fuse_zero_vel` | `0 - v_n` per NED velocity axis. |
| Vehicle speed | `fuse_body_speed_x` | `speed - e_x^T C_nv^T v_n`. |
| Lateral/vertical NHC | `fuse_body_vel_yz`, `fuse_gps_nhc_batch` | `0 - e_y^T C_nv^T v_n`, `0 - e_z^T C_nv^T v_n`. |
| Stationary gravity | `fuse_stationary_gravity` | Body/vehicle accelerometer channels compared to gravity-derived roll/pitch prediction. |

GNSS rows are generated scalar observations. NHC and vehicle-speed rows are
generated from the predicted vehicle-frame velocity:

```text
v_v = C_nv^T v_n
```

The instantaneous Reduced NHC residual has no direct mount Jacobian. Mount is
corrected through propagation-induced covariance with attitude, velocity, and
biases.

Reduced has two update styles:

- Scalar updates use Joseph-form covariance update:

  ```text
  P <- P - K H P - P H^T K^T + S K K^T
  ```

- `fuse_gps_nhc_batch` and `fuse_body_vel_yz_batch` apply sequential rows
  against one accumulated correction `dx`. Later rows see the covariance and
  predicted residual effect from earlier rows.

The batch path is deliberately used to couple GNSS and NHC at the same
linearization epoch while keeping runtime cost bounded.

### Reduced NHC Scheduling

`SensorFusion` controls when Reduced NHC is applied:

- default `nhc_update_period_s` is `0.1 s` (10 Hz),
- `0` applies at every eligible IMU epoch,
- positive periods decimate NHC and scale variance by elapsed observation time,
- near-zero speed can trigger zero-velocity/stationary pseudo-measurements
  instead of normal NHC,
- normal NHC uses vehicle-frame accel/gyro gates and a minimum speed gate.

The default NHC variances are:

```text
r_body_vel_y = 0.5
r_body_vel_z = 0.05
```

These are variances, not standard deviations. The lateral and vertical values
can be set together with `set_r_body_vel` or separately with
`set_r_body_vel_yz`.

### Reduced Mount Freezing And Settle Policy

`Filter::set_freeze_misalignment_states(true)` freezes Reduced mount states by:

- zeroing mount correction entries in update injections,
- zeroing mount covariance rows and columns,
- setting mount process noise to zero in prediction.

Manual mount mode always uses this behavior. Auto mode can also freeze mount
through `SensorFusion::set_freeze_misalignment_states`.

The facade also contains a mount-settle mechanism:

- `mount_settle_time_s` can temporarily freeze mount states after handoff,
- `mount_settle_release_sigma_rad` controls release based on mount uncertainty,
- `mount_settle_zero_cross_covariance` can clear cross-covariance on release.

This exists because early attitude/mount/bias allocation can be poorly
conditioned immediately after a coarse align seed.

## Full Filter

### Role

Full is a coupled ECEF INS/GNSS EKF in `sensor_fusion/src/full`. It carries
position, velocity, vehicle attitude, additive IMU corrections, scale
corrections, and mount. It is used for diagnostics, comparison, and optional
runtime operation.

Mainline Full keeps the same physical mount-in-propagation structure as
Reduced: raw body-frame IMU is corrected, rotated through `C_vb`, and then
used to propagate vehicle attitude and navigation state. This is important for
mount observability through propagation/GNSS cross-covariance.

### Nominal State

`sensor_fusion/src/full/types.rs` stores:

```text
q_ev
v_e
p_e
b_g_b
b_a_b
s_g_b
s_a_b
q_bv
```

Rust field layout:

| Fields | Meaning |
| --- | --- |
| `q0..q3` | `q_ev`, vehicle-to-ECEF attitude. |
| `vn, ve, vd` | ECEF velocity in m/s. |
| `pn, pe, pd` | ECEF position in m. Public f32 mirror of `pos_e64`. |
| `bgx..bgz` | Additive gyro correction in raw body frame. |
| `bax..baz` | Additive accelerometer correction in raw body frame. |
| `sgx..sgz` | Gyro scale correction. |
| `sax..saz` | Accelerometer scale correction. |
| `qcs0..qcs3` | Physical vehicle-to-body mount `q_bv`. Public f32 mirror of `qcs64`. |

Full keeps f64 shadow fields for numerically sensitive values:

| Field | Purpose |
| --- | --- |
| `pos_e64` | ECEF position propagated/injected in double precision. |
| `qcs64` | Mount quaternion used internally for injection and normalization. |

### Error State

Full has 24 error states:

```text
[dp_e, dv_e, dtheta_v, dba_b, dbg_b, dsa_b, dsg_b, dpsi_bv]
```

Index order:

| Indices | Error Component |
| --- | --- |
| `0..2` | ECEF position error. |
| `3..5` | ECEF velocity error. |
| `6..8` | Vehicle-attitude small angle. |
| `9..11` | Accelerometer correction error. |
| `12..14` | Gyro correction error. |
| `15..17` | Accelerometer-scale error. |
| `18..20` | Gyro-scale error. |
| `21..23` | Mount small-angle error for `q_bv`. |

### Initialization

Full exposes several initialization paths:

- `init_from_reference_state`
- `init_from_reference_ecef_state`
- `init_vehicle_from_nav_ecef_state`

The public facade initializes Full from GNSS, local yaw/course, and the
selected mount. `InitConfig` controls default diagonal covariance:

| Setting | Default |
| --- | --- |
| Position min sigma | `0.5 m` |
| Velocity min sigma | `0.2 m/s` |
| Attitude sigma | `20 deg` |
| Gyro-bias sigma | `0.125 deg/s` |
| Accelerometer-bias sigma | `0.15 m/s^2` |
| Gyro-scale sigma | `0.02` |
| Accelerometer-scale sigma | `0.0` |
| Mount roll/pitch sigma | `2 deg` |
| Mount yaw sigma | `6 deg` |

Manual mount mode calls `set_mount_quat` and freezes mount states.
Full facade initialization seeds both accelerometer and gyro scale nominal
states to `1.0`, because propagation applies them multiplicatively to raw IMU
rates and specific force.

### Nominal Propagation

Full consumes a two-sample IMU increment:

```text
dtheta_b_1, dvel_b_1
dtheta_b_2, dvel_b_2
dt
```

For each half-step, the code applies scale and additive corrections:

```text
omega_b = s_g * (dtheta_b / dt) + b_g
f_b     = s_a * (dvel_b   / dt) + b_a
omega_v = C_vb omega_b
f_v     = C_vb f_b
```

Then it rotates specific force into ECEF with `C_ev`, adds ECEF J2 gravity,
adds Earth-rate/Coriolis terms, and propagates position, velocity, and
quaternion with a predictor-corrector/trapezoidal structure. Nominal biases,
scales, and mount are constant during propagation.

The attitude derivative includes Earth rotation using
`WGS84_OMEGA_IE = 7.292115e-5 rad/s`.

### Error Propagation

The generated Full model in `sensor_fusion/src/full/formulation.py` emits
linearized `F` and `G` for the mainline state order. The covariance prediction
uses:

```text
P <- F P F^T + G Q G^T
```

Full uses `SparseCovariancePolicy::FULL`, which skips zero process-noise
diagonal entries. Runtime process-noise diagonal entries are:

| Noise Block | Runtime Diagonal Entry |
| --- | --- |
| Accelerometer white noise | `accel_var * dt` |
| Gyro white noise | `gyro_var * dt` |
| Accelerometer-bias random walk | `accel_bias_rw_var * dt` |
| Gyro-bias random walk | `gyro_bias_rw_var * dt` |
| Accelerometer-scale random walk | `accel_scale_rw_var * dt` |
| Gyro-scale random walk | `gyro_scale_rw_var * dt` |
| Mount random walk | `mount_align_rw_var_axis(i) * dt`, or zero when mount states are frozen. |

### Full Update Families

Full batches reference observations in `fuse_reference_batch_impl`. A batch can
contain up to eight rows:

```text
3 position rows
3 velocity rows
2 NHC rows
```

GNSS position is supplied in ECEF. The implementation builds a local NED
position covariance from horizontal accuracy:

```text
R_n = diag(h_acc^2, h_acc^2, (2.5 h_acc)^2)
```

It rotates that covariance to ECEF and applies a Cholesky whitening transform.
The fused residual is a whitened `z_p_ecef - p_e` residual.

GNSS velocity is supplied in ECEF. If per-axis NED velocity standard deviations
are available, the velocity covariance is rotated to ECEF and whitened. If only
a scalar speed accuracy is available, Full uses axis-aligned ECEF velocity rows
with that scalar variance.

Full has GNSS position and velocity gating diagnostics:

- `last_gnss_pos_gate` records position gate attempts and normalized residuals,
- velocity rows use a 3D chi-square-style rejection helper.

### Full NHC

Full NHC predicts vehicle-frame velocity from ECEF velocity:

```text
v_v = C_ev^T v_e
```

The lateral and vertical NHC residuals are:

```text
r_y = 0 - v_v_y
r_z = 0 - v_v_z
```

The generated NHC rows include velocity and vehicle-attitude derivatives. In
mainline Full they do not include direct mount columns, because `q_ev` already
represents vehicle attitude. Mount affects NHC through propagation-induced
covariance, not through the instantaneous residual.

NHC gates in `full::Filter` include:

```text
speed gate:       external/reference speed must be >= 0.05 m/s
gyro norm gate:   corrected gyro norm < 0.2 rad/s
accel norm gate:  |corrected accel norm - 9.81| < 1.0 m/s^2
```

The default Full NHC variances are:

```text
DEFAULT_NHC_R_Y = 0.1^2
DEFAULT_NHC_R_Z = 0.05^2
```

When the facade calls Full through common runtime settings, it passes the
configured `r_body_vel_y` and `r_body_vel_z` values into the Full batch path.

### Full Batch Update

Full applies batch rows sequentially. For each row:

```text
S = H P H^T + R
K = P H^T / S
effective_residual = residual - H dx_accum
dx_accum += K effective_residual
P <- P - (P H^T H P) / S
```

After all rows, Full:

- optionally blocks mount injection if mount states are frozen,
- records `last_dx`, per-observation `last_dx_by_obs`, residuals, effective
  residuals, innovation variances, and observation types,
- injects the accumulated error state,
- applies covariance reset to the attitude and mount blocks,
- freezes mount covariance again if required.

## Generated SymPy Code

Generated Rust fragments are checked in so normal Rust builds do not require
Python or SymPy.

### Reduced Generated Model

Source:

```text
sensor_fusion/src/reduced/formulation.py
```

Wrapper:

```text
sensor_fusion/src/reduced/generated.rs
```

Generated fragments:

```text
sensor_fusion/src/reduced/generated/*.rs
```

The Reduced generator emits:

- nominal prediction,
- error transition `F`,
- noise input `G`,
- sparsity supports,
- attitude reset Jacobian,
- GNSS position and velocity scalar rows,
- stationary accelerometer rows,
- body velocity X/Y/Z rows.

Regenerate with:

```bash
python sensor_fusion/src/reduced/formulation.py --emit-rust
```

### Full Generated Model

Source:

```text
sensor_fusion/src/full/formulation.py
```

Wrapper:

```text
sensor_fusion/src/full/generated.rs
```

Generated fragments:

```text
sensor_fusion/src/full/generated/*.rs
```

The Full generator emits:

- reference error transition `F`,
- noise input `G`,
- sparsity supports,
- reset Jacobian,
- lateral and vertical NHC rows.

Regenerate with:

```bash
python sensor_fusion/src/full/formulation.py --emit-rust
```

`sensor_fusion/code_gen.py` contains the shared Rust code generator. It uses a
custom C99 printer so small integer powers are expanded into multiplications,
then rewrites numeric literals for Rust `f32`.

Do not hand-edit generated fragments unless the change is explicitly a
temporary diagnostic. Change the formulation script, regenerate, and review the
diff.

## Covariance, Injection, And Reset Conventions

### Prediction

Reduced and Full both use the shared sparse helper:

```text
covariance::predict_sparse(F, G, P, Q, row_supports)
```

The helper computes the upper triangle of:

```text
F P F^T + G Q G^T
```

and mirrors it to preserve symmetry. The generated support arrays avoid dense
matrix work in embedded/runtime paths.

### Injection

Reduced injection:

```text
q_nv <- q_nv * dq(dtheta_v)
v_n  <- v_n + dv_n
p_n  <- p_n + dp_n
b_g  <- b_g + dbg
b_a  <- b_a + dba
q_bv <- dq(dpsi_bv) * q_bv
```

Full injection:

```text
p_e  <- p_e + dp_e
v_e  <- v_e + dv_e
q_ev <- dq(dtheta_v) * q_ev
b_a  <- b_a + dba
b_g  <- b_g + dbg
s_a  <- s_a + dsa
s_g  <- s_g + dsg
q_bv <- dq(dpsi_bv) * q_bv
```

Reduced uses a first-order small quaternion `[1, 0.5 dx, 0.5 dy, 0.5 dz]`.
Full builds quaternions through `euler_to_quat_f32` for attitude and mount
injection. Both filters normalize quaternions after injection.

### Reset

After injection, the covariance is reset because the error-state tangent point
has changed. The generated reset Jacobian block is:

```text
G_reset ~= I - 0.5 [dtheta]_x
```

Reduced applies this reset to the attitude block. The Reduced source currently
resets the attitude covariance block and then symmetrizes the full covariance;
mount covariance is not passed through a separate reset block in
`reduced::apply_reset`.

Full applies reset blocks to both vehicle attitude and mount:

```text
attitude block offset = 6
mount block offset    = 21
```

### Freezing Mount States

Freezing is stronger than merely setting process noise to zero. Both filters
also block mount correction entries and clear mount covariance rows/columns:

| Filter | Mount Error Indices |
| --- | --- |
| Reduced | `15..17` |
| Full | `21..23` |

Manual mount mode uses freezing to make the supplied `q_bv` fixed.

## Process Noise Profiles

`ProcessNoise` in `sensor_fusion/src/noise.rs` is shared by Reduced and Full.
It stores continuous variance-like settings:

| Field | Used By |
| --- | --- |
| `gyro_var` | Reduced and Full gyro white noise. |
| `accel_var` | Reduced and Full accelerometer white noise. |
| `gyro_bias_rw_var` | Reduced and Full gyro-bias random walk. |
| `accel_bias_rw_var` | Reduced and Full accelerometer-bias random walk. |
| `gyro_scale_rw_var` | Full only. |
| `accel_scale_rw_var` | Full only. |
| `mount_align_rw_var` | Reduced and Full mount random walk, all axes unless axis-specific mode is enabled. |
| `mount_align_rw_var_axes` | Optional per-axis mount random walk `[roll, pitch, yaw]`. |
| `mount_align_rw_var_axes_enabled` | Selects axis-specific mount random walk. |

Provided profiles:

| Function | Intent |
| --- | --- |
| `lsm6dso_104hz()` | Default LSM6DSO-oriented profile for 104 Hz IMU data. |
| `reduced_debug_default()` | Legacy broad covariance profile for standalone Reduced tests. |
| `reference_nsr_demo()` | Original NSR-style Full demo profile. |

The facade default uses a Reduced-specific `NoiseConfig` with scale random
walks set to zero for Reduced and `ProcessNoise::lsm6dso_104hz()` for Full.

## Runtime Tuning Parameters

The facade exposes tuning setters for the parameters most often used in replay
and visualizer experiments:

| Setter | Effect |
| --- | --- |
| `set_align_config` | Replaces Align settings and resets alignment/bootstrap state. |
| `set_reduced_noise`, `set_full_noise` | Replace process-noise profiles. Full replacement resets Full initialization. |
| `set_full_init_config` | Replaces Full initial covariance settings. |
| `set_r_body_vel`, `set_r_body_vel_yz` | Set lateral/vertical NHC variances. |
| `set_nhc_update_period_s` | Set NHC decimation period. Default is 0.1 s. |
| `set_gyro_bias_init_sigma_radps` | Set initial gyro-bias uncertainty. |
| `set_accel_bias_init_sigma_mps2` | Set initial accelerometer-bias uncertainty. |
| `set_yaw_init_sigma_rad` | Set initial yaw uncertainty. |
| `set_attitude_roll_pitch_init_sigma_rad` | Set initial roll/pitch attitude uncertainty. |
| `set_mount_init_sigma_rad`, `set_mount_roll_pitch_init_sigma_rad`, `set_mount_roll_init_sigma_rad`, `set_mount_pitch_init_sigma_rad` | Set initial mount uncertainty. |
| `set_accel_bias_rw_var` | Convenience setter for Reduced accelerometer-bias random walk. |
| `set_mount_align_rw_var` | Convenience setter for Reduced mount random walk. |
| `set_align_handoff_delay_s` | Require Align readiness to persist before handoff. |
| `set_freeze_misalignment_states` | Freeze or release mount states. |
| `set_mount_settle_time_s`, `set_mount_settle_release_sigma_rad`, `set_mount_settle_zero_cross_covariance` | Control mount-settle freezing after initialization. |
| `set_r_vehicle_speed` | Set vehicle speed observation variance. |
| `set_r_zero_vel` | Enable/set zero-velocity variance. |
| `set_r_stationary_accel` | Enable/set stationary gravity variance. |

Operationally important defaults from `RuntimeConfig`:

```text
r_body_vel_y = 0.5
r_body_vel_z = 0.05
nhc_update_period_s = 0.1
r_vehicle_speed = 0.04
r_zero_vel = 0.0
r_stationary_accel = 0.0
freeze_misalignment_states = false
mount_settle_time_s = 0.0
```

NHC variance values are variances. Smaller values increase constraint
strength.

## Known Observability And Robustness Limits

### Mount Is Mostly A Motion-Excited State

The physical mount becomes observable when changing vehicle motion causes a
wrong body-to-vehicle rotation to produce inconsistent inertial propagation or
vehicle-frame velocity constraints. Stationary gravity observes roll/pitch
tilt but not yaw. GNSS position/velocity rows do not directly observe mount in
either Reduced or mainline Full.

### NHC Is Powerful But Not Sufficient Alone

NHC is the main vehicle-frame constraint during motion. It is most informative
for turning and figure-eight maneuvers, where lateral/vertical vehicle-frame
velocity errors expose attitude and mount allocation. It can be weak or
misleading in:

- near-zero speed,
- long straight driving with little lateral excitation,
- sideslip or maneuvers that violate the nonholonomic assumption,
- GNSS outages or data gaps after covariance has collapsed,
- early operation with a bad align seed and poor covariance allocation.

The current practical default is 10 Hz NHC. Field Reduced behavior has shown
regressions at lower global rates such as 5 Hz and 2 Hz, while 10 Hz preserves
most behavior with lower compute cost than IMU-rate NHC.

### Attitude, Bias, And Mount Can Trade Off

The filters estimate vehicle attitude, IMU corrections, and mount together.
Several residuals can be explained by more than one state combination,
especially when motion lacks excitation. Overconfident covariance in the wrong
subspace can make later correction too weak or allocate residuals into the
wrong states.

Stationary and near-zero-speed regimes are particularly delicate. Zero
velocity and stationary gravity are useful, but if mount updates are left too
free when attitude and mount are not distinguishable, residuals can leak into
mount. This is why freezing, mount-settle, and zero-cross-covariance controls
exist.

### Bad Align Seeds Can Put Runtime Filters In A Wrong Basin

Align is a coarse seed, not a guarantee. If it hands off a bad yaw/roll/pitch
seed with overly tight covariance, Reduced or Full can converge slowly or settle
into a bad attitude/mount allocation. Better uncertainty calibration should be
based on general observability metrics rather than dataset-specific patches.

### Mainline Full Depends On Mount-In-Propagation Coupling

Mainline Full, like Reduced, rotates raw body IMU through `q_bv` before
propagating vehicle attitude and navigation state. This gives GNSS/inertial
residuals a propagation cross-covariance path to correct mount. Historical
experiments that placed mount only in vehicle-frame constraints removed this
coupling. They improved some maneuver-rich logs but regressed others,
especially outage/data-gap or weak early-observability cases. Do not add
separate mount columns to mainline NHC without changing and proving the state
semantics; in mainline Full `q_ev` already represents vehicle attitude.

### Tuning Does Not Replace Observability

Tightening lateral NHC variance can improve some maneuver-rich datasets, but
it is not a globally safe fix. Vertical NHC tightening can also regress some
cases. Treat NHC R, mount process noise, and initial covariance as controls
over residual allocation, not as proof that mount is observable in every
motion regime.

## Verification Pointers

Useful focused tests and diagnostics:

```bash
cargo test -p sensor_fusion --test align_api --locked
cargo test -p sensor_fusion --test reduced_state_ops --locked
cargo test -p sensor_fusion --test reduced_nhc_jacobian --locked
cargo test -p sensor_fusion --test nhc_reduced_full_equivalence --locked
cargo test -p sensor_fusion --test full_mount_observability --locked
cargo test -p sim --test full_parity --locked
```

Generated-model changes should generally be followed by:

```bash
python sensor_fusion/src/reduced/formulation.py --emit-rust
python sensor_fusion/src/full/formulation.py --emit-rust
cargo test -p sensor_fusion --locked
cargo test -p sim --test full_parity --locked
cargo test -p sim --test synthetic_gnss_ins --locked
```
