# Math Notes

The math notes are maintained as normal Sphinx pages so the documentation site is self-contained.

| Topic | Page |
| --- | --- |
| Frame and quaternion conventions | [](frames.md) |
| Runtime EKF formulation | [](runtime-ekf.md) |
| Alignment estimator formulation | [](align.md) |
| Roll/pitch observability | [](roll-observability.md) |
| Road event detector math | [](road-events.md) |

## Current Interpretation

The runtime model uses active rotations. The physical mount is `q_bv`, mapping vehicle-frame vectors into the raw IMU body frame:

```text
x_b = C_bv x_v
C_vb = C_bv^T
```

The EKF attitude is `q_nv`, mapping vehicle-frame vectors into local NED:

```text
x_n = C_nv x_v
```

Nonholonomic constraints are vehicle-frame pseudo-observations of lateral and vertical velocity. They are useful for velocity and attitude consistency, but they do not by themselves identify every decomposition of vehicle attitude and sensor mount. Current roll-observability claims in this documentation distinguish true observability from flat-road priors such as the optional vehicle-roll prior.
