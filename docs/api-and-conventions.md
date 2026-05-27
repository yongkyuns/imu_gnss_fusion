# API And Conventions

Public APIs use SI units unless the field name says otherwise. Raw IMU samples stay in the IMU body frame; callers do not pre-rotate them into the vehicle frame.

## Frames

| Symbol | Meaning |
| --- | --- |
| `b` | raw IMU body/sensor frame |
| `v` | vehicle frame, forward-right-down |
| `n` | local NED navigation frame |
| `e` | ECEF frame for WGS84 conversion |

The runtime uses active rotations. `C_ab` maps coordinates from frame `b` to frame `a`:

```text
x_a = C_ab x_b
R(q_ab) = C_ab
R(q1 * q2) = R(q1) R(q2)
```

Quaternions are scalar-first `[w, x, y, z]`.

The public mount quaternion is `q_bv`, the physical vehicle-to-body mount:

```text
x_b = C_bv x_v
C_vb = C_bv^T
x_v = C_vb x_b
```

The EKF attitude is `q_nv`:

```text
x_n = C_nv x_v
```

## Core Inputs

| Input | Required convention |
| --- | --- |
| `ImuSample::gyro_radps` | raw body-frame angular rate `[x_b, y_b, z_b]`, rad/s |
| `ImuSample::accel_mps2` | raw body-frame specific force `[x_b, y_b, z_b]`, m/s^2 |
| `GnssSample::lat_deg/lon_deg/height_m` | WGS84 latitude/longitude degrees and ellipsoidal height meters |
| `GnssSample::vel_ned_mps` | local `[north, east, down]` velocity, m/s |
| `GnssSample::pos_std_m` | one-sigma local NED position standard deviations, meters |
| `GnssSample::vel_std_mps` | one-sigma local NED velocity standard deviations, m/s |
| `GnssSample::heading_rad` | optional vehicle yaw/course heading in NED, radians clockwise from north toward east |
| `VehicleSpeedSample` | nonnegative speed magnitude along vehicle `+X`; direction selects forward/reverse |

## Mount Modes

`SensorFusion` supports two public construction modes:

| Mode | Behavior |
| --- | --- |
| `MountMode::Auto` | internal align estimates the initial `q_bv`; EKF initializes after mount readiness and GNSS yaw/course readiness |
| `MountMode::Manual(q_bv)` | caller supplies the initial `q_bv`; internal align is disabled, but EKF residual mount states remain live with a prior |

Manual mode does not mean the facade freezes every EKF mount state. `with_mount` and `set_misalignment` provide the physical mount seed and bypass align. At EKF initialization, the runtime seeds mount covariance from the manual prior so residual mount correction can still occur.

## EKF Initialization

Yaw initialization is mode-specific:

- auto mode can use `heading_rad`, or GNSS course once horizontal speed is at least `max(yaw_init_speed_mps, 1.0)`;
- manual mode waits for `heading_rad` and speed above `max(yaw_init_speed_mps, 20 / 3.6)`.

The runtime anchors WGS84 GNSS into a local navigation frame and reanchors when the local displacement grows large enough to keep local coordinates well-conditioned.

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
    vel_ned_mps: [5.0, 0.0, 0.0],
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
