# Frames And Quaternions

The project uses active rotations and scalar-first quaternions.

`C_ab` maps coordinates from frame `b` to frame `a`:

```text
x_a = C_ab x_b
R(q_ab) = C_ab
R(q1 * q2) = R(q1) R(q2)
```

Frame names:

| Symbol | Meaning |
| --- | --- |
| `b` | raw IMU body/sensor frame |
| `v` | vehicle frame, forward-right-down |
| `n` | local NED navigation frame |
| `e` | ECEF frame |

The public mount quaternion is the physical vehicle-to-body mount:

```text
q_bv
x_b = C_bv x_v
C_vb = C_bv^T
x_v = C_vb x_b
```

The EKF attitude is:

```text
q_nv
x_n = C_nv x_v
```

This convention is enforced by coordinate-convention tests in `sensor_fusion`.
