# Frame And Quaternion Conventions

This note summarizes the frame names used by the Rust filters and simulator docs. The detailed derivations remain in the PDF/TEX notes in `docs/`; this file is a quick reference for code review and test work.

## Frames

| Symbol | Name | Convention |
| --- | --- | --- |
| `n` | Navigation | Local NED: north, east, down. Positions and velocities in filter outputs use this frame. |
| `b` | Raw body/IMU | Sensor frame of raw accelerometer and gyroscope samples. |
| `v` | Vehicle | Forward, right, down vehicle frame. |
| `s` | Seeded vehicle | Raw IMU samples after applying the frozen coarse Align mount seed. If the seed is perfect, `s == v`. |
| `c` | Corrected vehicle | Vehicle frame after the ESKF or loose residual mount correction. If the residual is perfect, `c == v`. |
| `e` | ECEF | Earth-centered frame used internally by the loose reference path and geodetic conversion helpers. |

## Rotation Convention

Project math notes use active rotations:

```text
x_a = C_ab x_b
```

`C_ab` maps coordinates of a vector from frame `b` into frame `a`. The quaternion `q_ab` denotes the same rotation. Quaternion multiplication composes rotations in matrix order:

```text
C(q1 * q2) = C(q1) C(q2)
```

For a small angle `dtheta`, the first-order quaternion is:

```text
dq(dtheta) ~= [1, 0.5*dtheta_x, 0.5*dtheta_y, 0.5*dtheta_z]
C(dq) ~= I + [dtheta]x
```

## Align Mount Seed

The Align filter exposes a mount quaternion named `q_vb` in code. In the detailed formulation this is the vehicle-to-body mount attitude; the replay path uses its inverse/transpose to rotate raw body-frame IMU increments into the seeded frame.

At ESKF initialization, the current Align estimate is frozen:

```text
q_vb0 <- q_vb_align(t_init)
```

Raw IMU increments are then pre-rotated before ESKF prediction:

```text
dtheta_s = C_sb(q_vb0) dtheta_b
dv_s     = C_sb(q_vb0) dv_b
```

The augmented ESKF keeps an additional residual mount quaternion `q_cs`, initialized to identity, to estimate the remaining seeded-to-corrected vehicle-frame misalignment.

## ESKF State Frames

The augmented ESKF nominal state is:

```text
q_ns, v_n, p_n, b_g_s, b_a_s, q_cs
```

- `q_ns`: seeded-frame attitude into NED.
- `v_n`, `p_n`: NED velocity and local NED position.
- `b_g_s`, `b_a_s`: gyro and accelerometer bias in the seeded frame.
- `q_cs`: residual mount from seeded frame to corrected vehicle frame.

The 18-dimensional ESKF error state order is:

```text
dtheta_s, dv_n, dp_n, dbg_s, dba_s, dpsi_cs
```

## Loose State Frames

The loose reference path carries a larger nominal state:

```text
q, v_n, p_n, b_g, b_a, s_g, s_a, q_cs
```

It includes gyro/accelerometer scale states as well as biases and residual mount. The Rust loose filter receives the same pre-rotated IMU stream as the ESKF replay path in the visualizer.

## Practical Checks

- NED `z` is positive down; vertical signs should be interpreted with that convention.
- Vehicle `x` is forward, `y` is right, and `z` is down.
- `heading_rad` in generic replay GNSS rows is optional; `NaN` means no heading update.
- When comparing mount plots, confirm whether the trace is showing the frozen seed, residual `q_cs`, or the composed full mount.
