# Visualizer, Tools, Diagnostics, and Testing

This document maps the visualizer, browser app, replay tooling, diagnostics,
and CI/test workflows. It is intended as the operational guide for changing
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
- `.github/datasets/`: CI manifest and schema for hosted replay datasets.
- `.github/workflows/ci.yml`: Rust, hosted dataset, Pages artifact, and Pages
  deploy jobs.

## Visualizer Architecture

The visualizer has one data contract: `PlotData`. Every input path populates the
same trace groups, and the UI draws pages from those groups rather than knowing
whether the data came from a synthetic scenario, generic CSV directory, dropped
browser CSV, or hosted dataset.

Native flow:

1. `sim/src/bin/visualizer.rs` parses CLI args.
2. `--generic-replay-dir` loads `imu.csv`, `gnss.csv`, and optional references
   with `sim/src/datasets/generic_replay.rs`.
3. `--synthetic-motion-def` builds synthetic replay data through
   `sim/src/visualizer/pipeline/synthetic.rs`.
4. The selected pipeline produces `PlotData`.
5. `visualizer --profile-only` prints diagnostics and exits; otherwise
   `run_visualizer` starts the egui app.

Browser flow:

1. `web/index.html` loads `web/pkg/visualizer.js` and
   `web/pkg/visualizer_bg.wasm`, then calls the wasm export
   `start_visualizer("visualizer_canvas")`.
2. The app starts with empty/default `PlotData`, auto-loads the built-in
   synthetic scenario unless `?dataset=<id>` is present, and fetches
   `datasets/manifest.json`.
3. Browser replay building runs in `web/replay_worker.js`, a module worker that
   imports the same wasm bundle and calls exported replay functions.
4. The worker posts progress messages and a final JSON-serialized `PlotData`
   result back to `sim/src/visualizer/ui/web.rs`.
5. The UI deserializes `PlotData`, updates map center, and redraws the same
   egui pages as native.

The worker exists to keep CSV parsing and replay execution off the UI thread.
`replay_job.rs` applies `ReplayOutputPolicy::WebTransport` for wasm transport:
high-volume groups are decimated to preserve extrema and finite segment
boundaries, while comparison traces keep a higher cap so attitude, mount, and
position overlays remain useful.

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
for plots, maps, seeding when manual/reference mount mode is selected, and
reference-error summaries. They are not hidden device-specific inputs.

Browser hosted datasets are listed in `web/datasets/manifest.json`. Entries may
use plain CSV or gzip fields such as `imu_gz`, `gnss_gz`,
`reference_position_gz`, `reference_attitude_gz`, `reference_mount_gz`, and
`reference_motion_gz`. If `imu_gz` or `gnss_gz` is omitted, the browser loader
tries the conventional `imu.csv.gz`/`gnss.csv.gz` under `base_url`, then plain
CSV.

Dropped browser files are staged by filename. Names containing
`reference_attitude`, `reference_mount`, `reference_position`,
`reference_motion`, `gnss`, `imu`, `acc`, or `gyro` are assigned to the matching
input slots. After dropping files, select experimental/real data and run the
replay.

## UI Controls and Tabs

The top bar is defined in `sim/src/visualizer/ui/controls.rs`.

- Theme: light/dark. Native reads `IMU_GNSS_FUSION_THEME`; browser reads
  `?theme=`, `web/local-config.js`, or local storage.
- Trace toggles: Reference, Align, Reduced, Full. These are global filters over
  trace names and plot titles.
- Map toggles: GNSS and Heading. Recenter calls the map memory follow behavior.
- Mapbox token: browser field, query parameter `?mapbox_token=`, local config,
  or local storage. Native uses `MAPBOX_ACCESS_TOKEN`. Without a token, CARTO
  Positron/Dark Matter tiles are used.
- Tune: opens Reduced, Align, or Full tuning windows.
- Inspector: opens the update inspector window.
- Page tabs: Overview, Motion, Mount, Calibration, Sensors, Diagnostics.

Tabs:

- Overview: primary operational view. Shows vehicle speed, mount angles,
  vehicle attitude, biases, and map/trajectory. Mount and attitude plots can
  open orthogonal popups.
- Motion: vehicle-frame angular velocity and gravity-compensated acceleration,
  NED velocities, attitude error vs reference, and raw roll/pitch/yaw.
- Mount: Reduced, Full, Align, and reference mount roll/pitch/yaw, plus
  quaternion mount error on a log-scaled y-axis. Also shows align axis error
  and mount reference vs motion heading diagnostics.
- Calibration: gyro/accel biases, mount uncertainty, Full scale factors, and
  covariance diagonals. Covariance and mount uncertainty plots use log y scale
  to keep small sigmas visible.
- Sensors: GNSS signal strength when present, raw IMU, calibrated IMU, Reduced
  raw IMU inputs, Full vehicle-frame inputs, and miscellaneous signals.
- Diagnostics: align window internals, align flags and update contributions,
  Full mount correction, Full GNSS position gates, Reduced mount correction,
  bump detector signals, FFT diagnostics, and stationary diagnostics.

The shared hover cursor is propagated between plots and the map. For real data,
the map uses geographic lon/lat traces and cursor markers. For synthetic data,
the map panel becomes a local east/north plot because synthetic lat/lon is not
always the most useful inspection frame.

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

Reduced groups:

- `reduced_cmp_pos`, `reduced_cmp_vel`, `reduced_cmp_att`: GNSS/reference/truth
  comparisons and Reduced state estimates.
- `reduced_meas_gyro`, `reduced_meas_accel`: Reduced filter IMU inputs.
- `reduced_bias_gyro`, `reduced_bias_accel`: bias estimates.
- `reduced_cov_bias`, `reduced_cov_nonbias`, `reduced_mount_sigma`: covariance
  diagonal summaries.
- `reduced_mount_dx`, `reduced_nhc_mount_dx`, `reduced_nhc_innovation`,
  `reduced_nhc_nis`, `reduced_nhc_h_mount_norm`: update diagnostics.
- `reduced_misalignment`: mount estimates and mount errors.
- `reduced_stationary_diag`, `reduced_bump_pitch_speed`, `reduced_bump_diag`:
  detector diagnostics.
- `reduced_map`, `reduced_map_heading`: map path and heading samples.

Full groups:

- `full_cmp_pos`, `full_cmp_vel`, `full_cmp_att`: Full state comparison traces.
- `full_nominal_att`, `full_mount`, `full_misalignment`: attitude/mount
  details.
- `full_meas_gyro`, `full_meas_accel`: vehicle-frame Full filter inputs.
- `full_bias_gyro`, `full_bias_accel`, `full_scale_gyro`,
  `full_scale_accel`: calibration states.
- `full_cov_bias`, `full_cov_nonbias`, `full_mount_sigma`: covariance summaries.
- `full_mount_dx`, `full_nhc_innovation`, `full_gnss_pos_gate`: update and gate
  diagnostics.
- `full_map`, `full_map_heading`: Full path and heading samples.

Align groups:

- `align_cmp_att`: Align roll/pitch/yaw and reference mount traces.
- `align_res_vel`, `align_flags`: align window diagnostics and accepted/ready
  flags.
- `align_axis_err`, `align_motion`: axis error vs reference mount and motion
  heading comparison.
- `align_roll_contrib`, `align_pitch_contrib`, `align_yaw_contrib`: staged
  gravity, horizontal acceleration, and turn-gyro update contributions.
- `align_cov`: align covariance/sigma traces.

Inspection groups:

- `map_cursor`: per-trace cursor samples used to synchronize map markers with
  plot hover time.
- `update_inspector`: recent update samples with signed state contributions,
  residual/NIS data, and mount correlations.

Trace visibility is name-based in `ui/state.rs`. Reference includes names that
start with `Reference` or `Synthetic truth`; Align includes names that start
with or contain `Align`; Reduced and Full are matched by filter labels and some
legacy state marker strings. Be careful when adding new trace names: names drive
visibility, color classification, map filtering, and some tab composition.

## Maps

Real data uses `walkers::Map` with either Mapbox or CARTO raster tiles. The map
draws:

- GNSS-only path and optional reference fused path from `reduced_map`.
- Reduced path and Reduced outage path from `reduced_map`.
- Full path from `full_map` when Full is visible.
- Heading ticks from `reduced_map_heading` and `full_map_heading` when Heading
  is enabled.
- Hover cursor markers from `map_cursor`.

Synthetic data uses an egui plot instead of web tiles. It converts position
traces to local east/north and can show reference, GNSS, Reduced, and Full
trajectories. GNSS synthetic lon/lat traces are converted back to local EN
relative to the first synthetic truth path point.

## Tuning and Inspector

Reduced tuning controls mount mode, NHC/vehicle/stationary measurement weights,
initialization sigmas, mount update behavior, process noise, prediction
decimation, and IMU low-pass settings.

Align tuning controls handoff delay, align process/observation noise,
post-coarse refinement, motion gates, turn consistency, and which align updates
are enabled.

Full tuning controls Full process noise and Full initialization settings.

When the native app was launched with a replay state, `Apply replay` rebuilds
the current replay with the edited config. Browser runs are rebuilt through the
worker when the user clicks `Run`.

The update inspector is hover-driven. It looks back five seconds from the
current hover time, ranks the most active state contribution columns, shows
signed update allocation heatmap rows, and adds a mount error ledger comparing
start/end mount error against reference. It is most useful when diagnosing which
updates are pushing mount estimates toward or away from the reference.

## Command-Line Tools

Run tools with `cargo run -p sim --bin <name> -- <args>` unless noted.

- `visualizer`: main native visualizer and profile tool.
  - Generic replay:
    `cargo run --release -p sim --bin visualizer -- --generic-replay-dir /path/to/replay --profile-only`
  - Synthetic replay:
    `cargo run --release -p sim --bin visualizer -- --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario --synthetic-noise low --profile-only`
  - Key knobs include `--misalignment auto|manual`, synthetic noise/seed/mount,
    GNSS outage injection, NHC weights, init sigmas, mount settle controls,
    `--dump-align-axis-time-s`, and `--dump-full-time-s`.
- `first_divergence`: replays generic CSV data through public
  `SensorFusion::process_imu/process_gnss`, reports first threshold crossings,
  and can emit a behavior CSV.
- `filter_equivalence_harness`: emits Reduced/Full paired snapshots in a common
  physical basis for synthetic or generic replay input.
- `filter_equivalence_summary`: summarizes one or more harness CSVs with
  threshold-crossing flags and optional time windows.
- `covariance_history`: prints covariance, correction allocation, residual,
  NIS, and correlation diagnostics over selected times/windows; can write an
  allocation CSV.
- `diag_update_allocation`: synthetic allocation investigation for mount and
  related update contributions.
- `diag_prediction_residual`: synthetic prediction residual checks for Reduced,
  Full, or both.
- `diag_full_accel_z_bias`: synthetic probe for Full accel Z bias behavior.
- `compare_filter_runtime`: micro-benchmark style runtime comparison for filter
  paths.
- `run_full_nsr`: fixture-oriented Full replay runner used by parity tests.
- `seeded_full_stress_mc`: Monte Carlo stress utility for seeded Full behavior.
- `synthetic_bad_basin_sweep`: broad synthetic parameter/fault sweep.
- `mount_tuning_sweep`: sweeps generic replay directories over mount/tuning
  parameters and writes summary output.
- `export_synthetic_replay_generic`: converts synthetic export output into
  generic replay CSV files.

Useful diagnostic commands:

```bash
cargo run --release -p sim --bin first_divergence -- \
  --generic-replay-dir target/replay-analysis/field-sweep/urban-short-turn-loop-nominal-002 \
  --start-after-s 50 \
  --mount-threshold-deg 2.0 \
  --attitude-threshold-deg 2.0 \
  --window-s 10 \
  --behavior-csv /tmp/reference_filter_behavior.csv

cargo run -p sim --bin filter_equivalence_harness -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise truth \
  --mount-mode ref \
  --sample-stride 2000 \
  --output /tmp/equiv_city_ref.csv

cargo run -p sim --bin filter_equivalence_summary -- /tmp/equiv_city_ref.csv

cargo run -p sim --bin covariance_history -- \
  --synthetic-motion-def sim/motion_profiles/figure8_roll_excitation_30min.scenario \
  --synthetic-noise truth \
  --misalignment auto \
  --max-time-s 220 \
  --times 100,140,200 \
  --summary-window 100,200 \
  --allocation-csv /tmp/roll_excitation_alloc.csv
```

Important mount caveat: current Reduced and Full states store the physical
vehicle-to-body mount in `q_bv0..q_bv3`. The name is legacy. Diagnostic code
should treat those fields as `q_bv`, with `R(q_bv) = C_bv` and
`x_b = C_bv x_v`, rather than as an inverse/body-to-vehicle quantity.

## Web App Workflows

Build and serve the browser app:

```bash
cargo build -p sim --bin visualizer --release --target wasm32-unknown-unknown --locked
wasm-bindgen \
  --target web \
  --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/visualizer.wasm
python3 -m http.server --directory web 8080
```

Open `http://localhost:8080`.

Browser query parameters:

- `?theme=light` or `?theme=dark`
- `?mapbox_token=<token>`
- `?bench=1`
- `?scenario=city_blocks`, `figure8`, `figure8_fault`, `figure8_roll`, or
  `straight_accel_brake`
- `?noise=truth`, `low`, `mid`, or `high`
- `?dataset=<id>` to auto-load a hosted manifest dataset by id or label

Local browser defaults can be set in ignored `web/local-config.js`:

```js
window.IMU_GNSS_FUSION_LOCAL_CONFIG = {
  mapboxToken: "<token>",
  theme: "dark",
};
```

Benchmark FPS after building `web/pkg`:

```bash
node scripts/benchmark_web_fps.mjs --scenario city_blocks --min-fps 55
node scripts/benchmark_web_fps.mjs --dataset urban-short-turn-loop-nominal-001 --sample-memory --json
```

The benchmark serves `web/`, starts Chrome/Chromium through the DevTools
protocol, opens the app with `?bench=1`, waits for startup or dataset load, then
samples browser `requestAnimationFrame` and egui-published
`window.__imuGnssFusionPerf`. Use `--activity none` for idle sampling or the
default mouse activity to exercise hover/cursor rendering.

Validate the static Pages artifact:

```bash
node scripts/validate_pages_static.mjs --site-dir web --require-wasm
```

The validator checks relative wasm references, wasm presence and magic header,
safe dataset manifest URLs for the fields it recognizes, local static-server
MIME types, and fetchability of the listed core GNSS/IMU and attitude/mount
dataset artifacts.

## Dataset Packaging and Hosted Validation

Package a generic replay directory for static hosting:

```bash
python3 scripts/package_dataset.py /path/to/generic-replay web/datasets/my-dataset-id
```

The package contains `manifest.json`, `imu.csv.gz`, `gnss.csv.gz`, and optional
reference gzip files. Raw binary logs are intentionally rejected; conversion to
generic replay CSV belongs outside this repo. The packager validates CSV headers
and numeric fields, writes deterministic gzip files, and records sample counts,
time span, byte counts, and SHA-256 checksums.

Hosted dataset CI uses `.github/datasets/generic-datasets.json` and
`.github/datasets/generic-datasets.schema.json`. Each dataset entry has:

- `id`, `version`, optional metadata, and optional `replay_dir`.
- `files`, each with safe relative `path`, HTTP(S) or safe relative `url`,
  `sha256`, and optional `bytes`.
- Optional `smoke` config: `enabled`, `max_imu_rows`, `max_gnss_rows`, and
  `misalignment` (`auto` or `manual`).

Validate locally:

```bash
node scripts/validate_generic_datasets.mjs \
  --manifest .github/datasets/generic-datasets.json \
  --cache-dir .cache/generic-datasets \
  --work-dir target/generic-datasets-local-smoke \
  --smoke-profile
```

Validation steps:

1. Validate manifest shape and safe paths.
2. Download or reuse checksum-addressed cache entries.
3. Verify SHA-256 and optional byte count.
4. Assemble each dataset in `target/generic-datasets...`.
5. Validate generic CSV headers, finite numeric rows, and monotonic timestamps.
6. For enabled smoke entries, write bounded CSV subsets and run
   `visualizer --profile-only --misalignment <mode>`.

When changing hosted data, update the dataset manifest and, if needed,
`.github/datasets/logger-data.version` so the Actions cache key changes.

## CI Workflow

`.github/workflows/ci.yml` has four jobs:

- `rust`: runs on macOS and Linux. It installs Rust with rustfmt/clippy, Linux
  system dependencies, then runs:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets --locked -- -D warnings`
  - `cargo build --workspace --locked`
  - `cargo check -p sim --bins --locked`
  - `cargo test --workspace --locked`
  - `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked`
- `hosted-generic-datasets`: Node 22 plus Rust on Linux. It caches
  `.cache/generic-datasets` using `.github/datasets/**` as the key and runs
  hosted manifest/checksum/CSV/smoke validation.
- `web-pages-artifact`: builds the wasm visualizer, runs `wasm-bindgen`, touches
  `web/.nojekyll`, validates the Pages artifact, and uploads `web/`.
- `deploy-pages`: deploys the Pages artifact on pushes to `main`.

## Test Layers

Common commands:

```bash
cargo test --workspace --locked
cargo test -p sensor_fusion --locked
cargo test -p sim --locked
cargo test -p sim --test synthetic_gnss_ins --locked
cargo test -p sim --test visualizer_replay_equivalence --locked
cargo test -p sim --test full_parity --locked
cargo test -p sensor_fusion --test fusion_api --locked
cargo test -p sensor_fusion --test nhc_reduced_full_equivalence --locked
python3 -m unittest scripts.tests.test_package_dataset
```

Simulator tests:

- `sim/tests/synthetic_gnss_ins.rs`: synthetic path generation, motion DSL,
  noise models, Reduced convergence, synthetic visualizer trace population,
  mount observability scenarios, and helper checks against named traces.
- `sim/tests/visualizer_replay_equivalence.rs`: direct `GenericReplayInput`
  replay vs CSV replay job equivalence, and shared auxiliary trace equivalence
  between synthetic and generic replay paths.
- `sim/tests/eval_scaffolding.rs`: trace helper behavior and
  `FilterCompareConfig` default snapshot.
- `sim/tests/full_parity.rs`: `run_full_nsr` fixture replay against
  `sim/tests/fixtures/full_nsr_short/golden_summary.json`.

Sensor-fusion tests:

- `fusion_api.rs`: public API initialization, filter selection, NHC behavior,
  manual/internal mount modes, freeze/settle behavior, and vehicle speed
  update behavior.
- `align_api.rs`: align bootstrap and update stability.
- `reduced_state_ops.rs`: Reduced state injection, reset Jacobian, prediction,
  and layout behavior.
- `reduced_nhc_jacobian.rs`: finite-difference checks for Reduced body-velocity
  Y/Z Jacobians.
- `reduced_update_diag.rs`: Reduced update diagnostic accounting.
- `nhc_reduced_full_equivalence.rs`: generated Reduced/Full NHC rows, update
  math, transition blocks, and covariance transforms in a common basis.
- `full_mount_observability.rs`: Full mount-state NHC observability and
  reference speed behavior.

Script tests:

- `scripts/tests/test_package_dataset.py`: package format, deterministic gzip,
  manifest output, optional reference files, and raw binary rejection.

## Adding a New Regression Test

1. Pick the lowest layer that protects the behavior.
   - Pure filter math/API: add or extend `sensor_fusion/tests/*.rs`.
   - Synthetic replay or visualizer trace population: add or extend
     `sim/tests/synthetic_gnss_ins.rs`.
   - Generic CSV replay equivalence or browser/native replay contract: add or
     extend `sim/tests/visualizer_replay_equivalence.rs`.
   - Full fixture/golden behavior: add a fixture under `sim/tests/fixtures/`
     only when a small deterministic replay is necessary.
   - Dataset packaging/validation: add to `scripts/tests/test_package_dataset.py`
     or the Node validator when the behavior is outside Rust.
2. Make the input small and deterministic. Prefer inline motion DSL or generated
   generic replay samples over large checked-in logs.
3. Assert named trace schemas before numeric behavior. Use helpers such as
   `require_trace_schema`, `require_trace_points`, `sample_nearest_value`, and
   explicit trace names from `PlotData`.
4. Compare physical quantities in the correct basis. Reduced and Full
   `q_bv0..q_bv3` fields are the physical mount `q_bv`; use the inverse only
   when explicitly rotating body-frame vectors back into the vehicle frame.
5. Use tolerances that reflect the contract. Exact equality is appropriate for
   config snapshots and generated row names; replay/filter floating-point tests
   should use small explicit tolerances and clear failure labels.
6. Run the focused test first, then at least the crate-level suite:

```bash
cargo test -p sim --test visualizer_replay_equivalence --locked
cargo test -p sim --locked
cargo test -p sensor_fusion --locked
```

For hosted replay changes, also run:

```bash
node scripts/validate_generic_datasets.mjs \
  --manifest .github/datasets/generic-datasets.json \
  --cache-dir .cache/generic-datasets \
  --work-dir target/generic-datasets-local-smoke \
  --smoke-profile
node scripts/validate_pages_static.mjs --site-dir web --require-wasm
```
