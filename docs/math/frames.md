# Frame And Quaternion Conventions

This page is a short index for code review and operations. Detailed frame
definitions, state layouts, and measurement derivations live in the PDF math
notes:

- [Align/NHC formulation](../align_nhc_formulation.pdf)
- [Augmented ESKF formulation](../eskf_mount_formulation.pdf)
- [Loose INS/GNSS formulation](../loose_formulation.pdf)
- [Mount-in-propagation revision](../mount_in_propagation_revision.pdf)

Quick reference:

| Symbol | Meaning |
| --- | --- |
| `n` | Local NED navigation frame: north, east, down. |
| `e` | ECEF frame used by the loose reference filter. |
| `b` | Raw IMU body frame. |
| `v` | Physical vehicle frame, FRD: forward, right, down. |
| `s` | Seeded frame after raw IMU samples are pre-rotated by the coarse mount. |
| `c` | Corrected vehicle frame after residual mount `q_cs`. |

Project math notes use active rotations:

```text
x_a = C_ab x_b
C(q1 * q2) = C(q1) C(q2)
```

The align mount `q_vb` is the physical vehicle-to-body seed. ESKF and loose
consume IMU samples already rotated by that seed and carry residual mount
quaternions `q_cs` for seeded-to-corrected vehicle-frame constraints.
