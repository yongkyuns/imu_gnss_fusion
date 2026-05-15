# Frame And Quaternion Conventions

This page is the short operational reference for the EKF frame conventions.
Detailed derivations live in the formulation PDFs and TEX sources in `docs/`.

| Symbol | Meaning |
| --- | --- |
| `b` | Raw IMU body/sensor frame. Public IMU samples are expressed here. |
| `v` | Physical vehicle frame, FRD: forward, right, down. Vehicle speed and NHC are expressed here. |
| `n` | Local NED navigation frame: north, east, down. |
| `e` | ECEF frame used for WGS84 conversion and global position math. |

Project math uses active rotations. `C_ab` maps coordinates from frame `b` to
frame `a`, and quaternion subscripts follow the same direction:

```text
x_a = C_ab x_b
R(q_ab) = C_ab
R(q1 * q2) = R(q1) R(q2)
```

Quaternions are scalar-first `[w, x, y, z]`.

The public mount is the physical vehicle-to-body quaternion `q_bv`. Its DCM is
`C_bv = R(q_bv)`:

```text
x_b = C_bv x_v
C_vb = C_bv^T
x_v = C_vb x_b
```

Do not pre-rotate IMU samples before passing them to `SensorFusion`; the EKF
rotates raw body-frame samples into the vehicle frame internally.

NHC uses vehicle-frame velocity:

```text
v_v = C_nv^T v_n
```

Do not introduce a separate "car" frame in code or docs; use `vehicle`/`v`.
