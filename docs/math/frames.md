# Frame And Quaternion Conventions

This page is a short index for code review and operations. Detailed frame
definitions, state layouts, and measurement derivations live in the PDF math
notes:

- [Align/NHC formulation](../align.pdf)
- [Augmented Reduced formulation](../reduced.pdf)
- [Full INS/GNSS formulation](../full.pdf)

Quick reference:

| Symbol | Meaning |
| --- | --- |
| `n` | Local NED navigation frame: north, east, down. |
| `e` | ECEF frame used by the Full filter. |
| `b` | Raw IMU body/sensor frame. Public IMU samples are expressed here. |
| `v` | Physical vehicle frame, FRD: forward, right, down. Vehicle speed and NHC are expressed here. |

Project math notes use active rotations. `C_ab` maps coordinates from frame `b`
to frame `a`, and quaternion subscripts follow the same direction:

```text
x_a = C_ab x_b
R(q_ab) = C_ab
R(q1 * q2) = R(q1) R(q2)
```

The align mount, the Reduced `qcs0..qcs3` fields, and the Full `qcs0..qcs3`
fields all represent the same public physical vehicle-to-body mount `q_bv`. The
generated state field name `qcs` is retained for layout compatibility, but its
DCM is `C_bv = R(q_bv)`:

```text
x_b = C_bv x_v
C_vb = C_bv^T
x_v = C_vb x_b
```

Reduced and Full both consume raw body-frame IMU samples and rotate them into
the vehicle frame during propagation. Reduced attitude `q0..q3` is `q_nv`: the
NED/navigation-frame attitude with respect to the vehicle frame, with
`R(q_nv) = C_nv` and `x_n = C_nv x_v`. Full attitude `q0..q3` is `q_ev`: the
ECEF-frame attitude with respect to the vehicle frame, with `R(q_ev) = C_ev`
and `x_e = C_ev x_v`.
NHC uses vehicle-frame velocity:

```text
Reduced: v_v = C_nv^T v_n
Full:    v_v = C_ev^T v_e
```

Do not introduce a separate "car" frame in code or docs; use `vehicle`/`v`.
