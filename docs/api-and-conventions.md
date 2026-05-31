# API And Conventions

Public APIs use SI units unless the field name says otherwise. Raw IMU samples stay in the IMU body frame; callers do not pre-rotate them into the vehicle frame.

For the full frame convention, including error-state injection sides, see [](algorithms/frames.md).

## Frames

| Symbol | Meaning |
| --- | --- |
| `b` | raw IMU body/sensor frame |
| `v` | vehicle frame, forward-right-down |
| `n` | local NED navigation frame |
| `e` | ECEF frame for WGS84 conversion |

The runtime uses active rotations. $C_{ab}$ maps coordinates from frame $b$
to frame $a$:

$$
\begin{aligned}
x_a &= C_{ab} x_b,\\
R(q_{ab}) &= C_{ab},\\
R(q_1 q_2) &= R(q_1)R(q_2).
\end{aligned}
$$

Quaternions are scalar-first $[q_w, q_x, q_y, q_z]$. The full
quaternion-to-matrix definition is in [](algorithms/frames.md).

The public mount quaternion is $q_{bv}$, the physical vehicle-to-body mount:

$$
\begin{aligned}
x_b &= C_{bv} x_v,\\
C_{vb} &= C_{bv}^{\top},\\
x_v &= C_{vb} x_b .
\end{aligned}
$$

The EKF attitude is $q_{nv}$:

$$
x_n = C_{nv} x_v.
$$

## Core Inputs

| Input | Required convention |
| --- | --- |
| `ImuSample::gyro_radps` | raw body-frame angular rate $[x_b, y_b, z_b]$, rad/s |
| `ImuSample::accel_mps2` | raw body-frame specific force $[x_b, y_b, z_b]$, m/s^2 |
| `GnssSample::lat_deg/lon_deg/height_m` | WGS84 latitude/longitude degrees and ellipsoidal height meters |
| `GnssSample::vel_ned_mps` | local $[\text{north}, \text{east}, \text{down}]$ velocity, m/s |
| `GnssSample::pos_std_m` | one-sigma local NED position standard deviations, meters |
| `GnssSample::vel_std_mps` | one-sigma local NED velocity standard deviations, m/s |
| `GnssSample::heading_rad` | optional vehicle yaw/course heading in NED, radians clockwise from north toward east |
| `VehicleSpeedSample` | nonnegative speed magnitude along vehicle `+X`; direction selects forward/reverse |

## Mount Modes

`SensorFusion` supports two public construction modes:

| Mode | Behavior |
| --- | --- |
| `MountMode::Auto` | internal align estimates the initial $q_{bv}$; EKF initializes after mount readiness and the first usable GNSS seed |
| `MountMode::Manual(q_bv)` | caller supplies the initial $q_{bv}$; internal align is disabled, but EKF residual mount states remain live with a prior |

Manual mode does not mean the facade freezes every EKF mount state. `with_mount` and `set_misalignment` provide the physical mount seed and bypass align. At EKF initialization, the runtime seeds mount covariance from the manual prior so residual mount correction can still occur.

## EKF Initialization

Yaw initialization is mode-specific:

- auto mode initializes once mount is ready and uses `heading_rad` when present, otherwise GNSS course once horizontal speed is at least $\max(\texttt{yaw\_init\_speed\_mps}, 1.0)$, otherwise yaw $0$;
- manual mode waits for `heading_rad` and speed above $\max(\texttt{yaw\_init\_speed\_mps}, 20 / 3.6)$.

The first GNSS sample after mount readiness initializes the EKF immediately when
the mode-specific yaw rule is satisfied. After initialization, GNSS samples are
queued by `process_gnss` and fused at the next IMU epoch. If the queued GNSS
sample is between $0$ and $0.05\,\mathrm{s}$ older than that IMU epoch, eligible NHC
rows can be fused in the same batch.

The runtime anchors WGS84 GNSS into a local navigation frame and reanchors when
local displacement exceeds $5000\,\mathrm{m}$. Reanchoring creates a new local origin,
rotates the navigation state/covariance into the new local frame, clears
align/GNSS interval coupling, and resets the current local position near zero.

## Runtime State

Each processed input returns an `Update` with a single lifecycle `state`, `navigation_usable`, `navigation_started`, mount readiness, and the current $q_{bv}$ when available. Use those public fields, or `SensorFusion::health()`, instead of inferring readiness from internal EKF existence.

The main public states are:

- `NotReady` and `Initializing`: keep feeding samples; public navigation is not usable yet.
- `Running`: navigation is usable, but convergence is not mature enough for saved priors.
- `Stable`: navigation is usable and invariant states are stable enough to persist externally.
- `Degraded` and `DegradedDeadReckoning`: navigation may still be usable, but callers should surface degraded confidence.
- `AwaitingGnssReseed`: calibration is retained, but navigation output is intentionally unavailable until GNSS reseeds it.

Normal trip-end pauses should keep the same `SensorFusion` object and call `SensorFusion::end_trip()` before samples stop. This marks the next timestamp gap as expected stationary sleep. Unmarked long gaps are treated as unexpected in-trip data loss and public navigation waits for GNSS reseed. Source switches, replay changes, physical mount changes, and lost retained memory should create a fresh context. See [](runtime-state-and-persistence.md) for the full sleep/resume contract and covariance aging behavior.

## GNSS Events

`Update.gnss_event_mask` reports GNSS rejection and bypass events emitted while queued GNSS rows are fused at an IMU epoch. Position and velocity groups are gated independently with a deliberately loose default 25-sigma per-axis test, so only very large GNSS discontinuities are rejected by default. Public bits distinguish:

- position or velocity rejected;
- repeated consecutive rejection;
- bypass after a GNSS update gap greater than $3\,\mathrm{s}$;
- bypass after reported RMS accuracy improves to at most half the previous RMS.

Rejected groups are not fused. Consecutive-rejection bits are diagnostic events, not an automatic recovery update. See [](algorithms/runtime-ekf.md) for the update equations.

## Tuning Surface

Important public setters include:

- `set_ekf_noise(ProcessNoise)`;
- `set_accel_bias_rw_var` and `set_mount_align_rw_var`;
- `set_r_body_vel` and `set_r_body_vel_yz`;
- `set_nhc_update_period_s`;
- `set_r_vehicle_roll_prior`;
- `set_r_vehicle_speed`, `set_r_zero_vel`, and `set_r_stationary_accel`;
- roll, pitch, yaw, bias, and per-axis mount initialization sigma setters;
- `set_use_align_mount_covariance_on_seed`.

The facade also exposes diagnostics such as `align_debug`, anchor/reanchor debug values, and `analysis_*` hooks. Treat `analysis_*` methods as diagnostic controls rather than normal production API.

Important default values include:

| Setting | Default |
| --- | --- |
| lateral/vertical NHC variance | `0.5`, `0.5` |
| NHC update period | `0.1 s` |
| vehicle-roll prior variance density | `0.1` |
| align covariance handoff on EKF seed | enabled |
| automatic mount seed sigma | align covariance when handoff is enabled |
| manual residual mount sigma | `0.5 deg` per axis |

## Minimal Example

```rust
use sensor_fusion::{
    Config, GnssSample, ImuSample, MountMode, SensorFusion,
    VehicleSpeedDirection, VehicleSpeedSample,
};

let mount_q = [1.0, 0.0, 0.0, 0.0]; // q_bv: x_b = R(q_bv) x_v
let mut fusion = SensorFusion::with_config(Config {
    mount_mode: MountMode::Manual(mount_q),
});

fusion.process_imu(ImuSample {
    t_s: 0.00,
    gyro_radps: [0.0, 0.0, 0.0],
    accel_mps2: [0.0, 0.0, -9.80665],
});

fusion.process_gnss(GnssSample {
    t_s: 0.01,
    lat_deg: 37.0,
    lon_deg: -122.0,
    height_m: 10.0,
    vel_ned_mps: [6.0, 0.0, 0.0],
    pos_std_m: [1.0, 1.0, 2.5],
    vel_std_mps: [0.1, 0.1, 0.2],
    heading_rad: Some(0.0),
});

fusion.process_vehicle_speed(VehicleSpeedSample {
    t_s: 0.02,
    speed_mps: 5.0,
    direction: VehicleSpeedDirection::Forward,
});

if let Some(q_bv) = fusion.mount_q_bv() {
    let _ = q_bv;
}
```
