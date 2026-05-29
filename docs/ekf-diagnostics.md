# EKF Diagnostics

EKF diagnostics are exposed for replay analysis, tuning, and visualizer plots. They are not separate runtime sensors.

Useful diagnostic surfaces include:

- update diagnostics for GNSS, zero-velocity, body-speed, NHC, stationary-gravity, and vehicle-roll-prior updates;
- `Update.state`, `Update.navigation_usable`, and `SensorFusion::health()` for the public lifecycle and convergence verdict;
- `align_debug` for the most recent align window and update trace;
- anchor/reanchor debug accessors for local-frame behavior;
- visualizer mount, motion, calibration, sensor, diagnostics, and road-event pages;
- diagnostic binaries in `sim` for mount observability and synthetic sweeps.

When reporting mount error, use the physical public convention $q_{bv}$, where $x_b = C_{bv}x_v$.

Health is intentionally exposed as one lifecycle state plus derived flags. `Stable` is the only state suitable for saving external priors. `DegradedDeadReckoning` can still be navigation-usable after a medium sleep gap, while `AwaitingGnssReseed` means calibration is retained but public navigation must wait for GNSS. See [](runtime-state-and-persistence.md) for the full state table.
