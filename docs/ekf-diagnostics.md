# EKF Diagnostics

Use this page as an index for current diagnostics that help explain EKF
behavior during replay and synthetic testing.

## Primary Tools

| Tool | Use |
| --- | --- |
| `visualizer` | Inspect traces, maps, mount estimates, update allocation, and summary statistics. |
| `diag_mount_observability` | Exercise roll/pitch mount observability in synthetic scenarios. |
| `synthetic_bad_basin_sweep` | Sweep early-convergence synthetic stress cases. |
| `export_synthetic_replay_generic` | Export synthetic replay outputs to the generic CSV schema. |

## What To Record

When a diagnostic result should be preserved, record:

- exact command line,
- dataset or synthetic scenario name,
- mount mode,
- noise/tuning overrides,
- time window,
- key metrics and thresholds,
- artifact paths for generated CSVs or screenshots.

Keep interpretation tied to physical quantities: position, velocity, attitude,
mount, IMU bias, residuals, NIS, and covariance. Avoid comparing field names
without first confirming the frame and units.
