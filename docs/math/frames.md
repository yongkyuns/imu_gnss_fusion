# Frame And Quaternion Conventions

This page is a short index for code review and operations. Detailed frame
definitions, state layouts, and measurement derivations live in the PDF math
notes:

- [Align/NHC formulation](../align_nhc_formulation.pdf)
- [Augmented Reduced formulation](../reduced_mount_formulation.pdf)
- [Full INS/GNSS formulation](../full_formulation.pdf)
- [Mount-in-propagation revision](../mount_in_propagation_revision.pdf)

Quick reference:

| Symbol | Meaning |
| --- | --- |
| `n` | Local NED navigation frame: north, east, down. |
| `e` | ECEF frame used by the Full filter. |
| `b` | Raw IMU body/sensor frame. Public IMU samples are expressed here. |
| `v` | Physical vehicle frame, FRD: forward, right, down. Vehicle speed and NHC are expressed here. |

Project math notes use active rotations:

```text
x_a = C_ab x_b
C(q1 * q2) = C(q1) C(q2)
```

The align mount, the Reduced `qcs0..qcs3` fields, and the Full `qcs0..qcs3`
fields all represent the same physical vehicle-to-body mount. The generated
state field name `qcs` is retained for layout compatibility, but its DCM is
used as `C_bv`:

```text
x_b = C_bv(q_mount) x_v
x_v = C_bv(q_mount)^T x_b
```

Reduced and Full both consume raw body-frame IMU samples and rotate them into
the vehicle frame during propagation. Reduced attitude `q0..q3` is `q_nv`
(vehicle to local NED). Full attitude `q0..q3` is `q_ev` (vehicle to ECEF).
NHC uses vehicle-frame velocity:

```text
Reduced: v_v = C_nv^T v_n
Full:    v_v = C_ev^T v_e
```

Do not introduce a separate "car" frame in code or docs; use `vehicle`/`v`.
