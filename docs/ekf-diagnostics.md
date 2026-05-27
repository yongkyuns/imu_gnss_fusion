# EKF Diagnostics

EKF diagnostics are exposed for replay analysis, tuning, and visualizer plots. They are not separate runtime sensors.

Useful diagnostic surfaces include:

- update diagnostics for GNSS, zero-velocity, body-speed, NHC, stationary-gravity, and vehicle-roll-prior updates;
- `align_debug` for the most recent align window and update trace;
- anchor/reanchor debug accessors for local-frame behavior;
- visualizer mount, motion, calibration, sensor, diagnostics, and road-event pages;
- diagnostic binaries in `sim` for mount observability and synthetic sweeps.

When reporting mount error, use the physical public convention `q_bv`, where `x_b = C_bv x_v`.
