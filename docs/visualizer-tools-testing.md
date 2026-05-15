# Visualizer, Tools, Diagnostics, and Testing

This document maps the visualizer, browser app, replay tooling, diagnostics,
and CI/test workflows. It is the operational guide for changes under
`sim/src/visualizer`, `sim/src/bin`, `web`, `scripts`, `sim/tests`,
`sensor_fusion/tests`, and the hosted generic dataset configuration.

## Source Map

- `sim/src/visualizer/model.rs`: shared `PlotData`, `Trace`, map cursor,
  update-inspector, and `Page` data model.
- `sim/src/visualizer/pipeline/`: synthetic and generic replay builders that
  convert replay inputs into `PlotData`.
- `sim/src/visualizer/replay_job.rs`: reusable replay job wrapper for native,
  tests, and wasm transport, including web decimation.
- `sim/src/visualizer/ui/`: egui UI for native and wasm builds.
- `sim/src/bin/visualizer.rs`: native CLI entrypoint and wasm exports.
- `web/`: static browser host, built wasm bundle location, web worker, and
  hosted dataset manifest.
- `scripts/`: dataset packaging, hosted dataset validation, Pages artifact
  validation, and browser FPS benchmarking.

## Visualizer Architecture

The visualizer has one data contract: `PlotData`. Every input path populates
the same trace groups, and the UI draws pages from those groups rather than
knowing whether the data came from a synthetic scenario, generic CSV directory,
dropped browser CSV, or hosted dataset.

Native flow:

1. `sim/src/bin/visualizer.rs` parses CLI args.
2. `--generic-replay-dir` loads `imu.csv`, `gnss.csv`, and optional references.
3. `--synthetic-motion-def` builds synthetic replay data.
4. The selected pipeline produces `PlotData`.
5. `visualizer --profile-only` prints diagnostics and exits; otherwise egui
   starts.

Browser flow:

1. `web/index.html` loads the wasm bundle and starts the app.
2. The app loads a hosted dataset, dropped CSV files, or a synthetic scenario.
3. Browser replay runs in `web/replay_worker.js`.
4. The worker posts progress and a JSON `PlotData` result.
5. The UI deserializes the result, updates map center, and redraws the same
   pages as native.

The worker keeps CSV parsing and replay execution off the UI thread.

## Data Loading

Generic replay CSV schema:

- `imu.csv`: `t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2`
- `gnss.csv`:
  `t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,pos_std_n_m,pos_std_e_m,pos_std_d_m,vel_std_n_mps,vel_std_e_mps,vel_std_d_mps,heading_rad`
- `reference_attitude.csv`: `t_s,roll_deg,pitch_deg,yaw_deg`
- `reference_mount.csv`: `t_s,roll_deg,pitch_deg,yaw_deg`
- `reference_position.csv`:
  `t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,heading_rad`
- `reference_motion.csv`:
  `t_s,wx_radps,wy_radps,wz_radps,ax_mps2,ay_mps2,az_mps2`

`imu.csv` and `gnss.csv` are required. Reference files are optional and are used
for plots, maps, manual/reference mount seeding, and reference-error summaries.

## UI Controls and Tabs

The top bar is defined in `sim/src/visualizer/ui/controls.rs`.

- Theme: light/dark. Native reads `IMU_GNSS_FUSION_THEME`; browser reads
  `?theme=`, `web/local-config.js`, or local storage.
- Trace toggles: reference, align, EKF, map, and diagnostic trace families.
- Map toggles: GNSS and Heading.
- Mapbox token: browser field, query parameter `?mapbox_token=`, local config,
  or local storage. Native uses `MAPBOX_ACCESS_TOKEN`.
- Tune: opens runtime and align tuning windows.
- Inspector: opens the update inspector window.
- Page tabs: Overview, Motion, Mount, Calibration, Sensors, Diagnostics.

Tabs:

- Overview: vehicle speed, mount angles, vehicle attitude, biases, and
  map/trajectory.
- Motion: vehicle-frame angular velocity and gravity-compensated acceleration,
  NED velocities, attitude error vs reference, and raw roll/pitch/yaw.
- Mount: EKF, align, and reference mount roll/pitch/yaw, plus quaternion mount
  error on a log-scaled y-axis.
- Calibration: gyro/accel biases, mount uncertainty, and covariance diagonals when
  available, and covariance diagonals.
- Sensors: GNSS signal strength when present, raw IMU, calibrated IMU, EKF IMU
  inputs, and miscellaneous signals.
- Diagnostics: align internals, update contributions, gates, residuals, bump
  detector signals, FFT diagnostics, and stationary diagnostics.

The shared hover cursor is propagated between plots and the map. Real data uses
geographic lon/lat traces; synthetic data uses a local east/north plot.

## Trace Groups

`PlotData` groups are the stable bridge between replay pipelines, UI pages,
profile diagnostics, and tests.

Primary/common groups:

- `speed`: GNSS or synthetic horizontal speed.
- `vehicle_motion_gyro`, `vehicle_motion_accel`: vehicle-frame reference or
  estimated motion signals.
- `imu_raw_gyro`, `imu_raw_accel`: source IMU in sensor/body input frame.
- `imu_cal_gyro`, `imu_cal_accel`: calibrated IMU traces when populated.
- `orientation`: reference or synthetic truth roll/pitch/yaw.
- `other`: miscellaneous debug traces.

EKF groups:

- position, velocity, attitude, and map traces,
- raw/calibrated IMU input traces,
- gyro and accelerometer bias traces,
- covariance and mount sigma traces,
- NHC residual, NIS, and correction traces,
- GNSS gate diagnostics,
- mount estimate and mount error traces.

Align groups:

- `align_cmp_att`: align roll/pitch/yaw and reference mount traces.
- `align_res_vel`, `align_flags`: align window diagnostics and accepted/ready
  flags.
- `align_axis_err`, `align_motion`: axis error vs reference mount and motion
  heading checks.
- `align_roll_contrib`, `align_pitch_contrib`, `align_yaw_contrib`: staged
  gravity, horizontal acceleration, and turn-gyro update contributions.
- `align_cov`: align covariance/sigma traces.

Inspection groups:

- `map_cursor`: per-trace cursor samples used to synchronize map markers with
  plot hover time.
- `update_inspector`: recent update samples with signed state contributions,
  residual/NIS data, and mount correlations.

Trace visibility is name-based in `ui/state.rs`; keep trace names stable unless
the UI classifier and tests are updated at the same time.

## Maps

Real data uses `walkers::Map` with either Mapbox or CARTO raster tiles. The map
draws GNSS, optional reference, EKF trajectory, heading ticks, and hover cursor
markers when those traces are present.

Synthetic data uses an egui plot instead of web tiles. It converts position
traces to local east/north relative to the first synthetic truth path point.

## Tuning and Inspector

Runtime tuning controls mount mode, NHC/vehicle/stationary measurement weights,
initialization sigmas, mount update behavior, process noise, prediction
decimation, and IMU low-pass settings.

Align tuning controls handoff delay, process/observation noise, post-coarse
refinement, motion gates, turn consistency, and which align updates are enabled.

The update inspector is hover-driven. It looks back five seconds from the
current hover time, ranks active state contribution columns, shows signed update
allocation heatmap rows, and adds a mount error ledger comparing start/end
mount error against reference.

## Command-Line Tools

Run tools with `cargo run -p sim --bin <name> -- <args>` unless noted.

- `visualizer`: main native visualizer and profile tool.
- `diag_mount_observability`: synthetic roll/pitch mount observability
  diagnostic.
- `synthetic_bad_basin_sweep`: synthetic early-convergence stress sweep.
- `export_synthetic_replay_generic`: exports synthetic replay data to the
  generic CSV schema.

Generic replay smoke:

```bash
cargo run --release -p sim --bin visualizer -- \
  --generic-replay-dir /path/to/replay \
  --profile-only
```

Synthetic replay smoke:

```bash
cargo run --release -p sim --bin visualizer -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low \
  --profile-only
```

## Browser Build and Validation

Build the wasm visualizer:

```bash
cargo build -p sim --bin visualizer --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/visualizer.wasm
```

Serve locally:

```bash
python3 -m http.server --directory web 8080
```

Validate hosted datasets and packaging with the commands in [testing.md](testing.md).
