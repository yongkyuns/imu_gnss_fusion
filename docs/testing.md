# Testing

This project has three main test layers: Rust workspace tests, simulator/evaluator fixtures, and generated-model parity checks. Most contributors should start with the Rust commands below and only run data-heavy or external-reference checks when changing replay or model-generation behavior.

## Common Commands

Run all root-workspace tests:

```bash
cargo test --workspace --locked
```

Build all root-workspace crates:

```bash
cargo build --workspace --locked
```

Run focused crate tests:

```bash
cargo test -p sensor_fusion --locked
cargo test -p sim --locked
```

Run one integration test target:

```bash
cargo test -p sim --test synthetic_gnss_ins --locked
cargo test -p sim --test full_parity --locked
cargo test -p sensor_fusion --test fusion_api --locked
cargo test -p sensor_fusion --test nhc_reduced_full_equivalence --locked
```

## Test Coverage Map

| Area | Tests | What they protect |
| --- | --- | --- |
| Filter APIs | `sensor_fusion/tests/fusion_api.rs`, `align_api.rs`, `reduced_state_ops.rs` | Public API behavior, state initialization, and basic update contracts. |
| Reduced/Full NHC equivalence | `sensor_fusion/tests/nhc_reduced_full_equivalence.rs` | Generated Reduced and Full NHC rows, update math, transition blocks, and common covariance propagation after the attitude-basis transform. |
| Full observability | `sensor_fusion/tests/full_mount_observability.rs`, `sim/tests/full_parity.rs` | Full EKF mount behavior and parity with prepared seeded replay fixtures. |
| Synthetic replay | `sim/tests/synthetic_gnss_ins.rs` | Synthetic path generation, Reduced EKF convergence, visualizer trace population, and checked-in replay fixtures. |
| Generic replay | `sim/src/datasets/generic_replay.rs`, `sim/tests/synthetic_gnss_ins.rs` | Hardware-agnostic CSV schema loading and visualizer trace population. |
| Dataset packaging | `scripts/tests/test_package_dataset.py` | Hosted-data manifest generation, deterministic gzip staging, and raw binary rejection. |

## Reduced/Full Equivalence Diagnostics

Use `sensor_fusion/tests/nhc_reduced_full_equivalence.rs` as the lightweight regression suite for NHC equivalence. It runs without replay fixtures and checks the generated rows and covariance-basis transform directly.

The simulator-side filter equivalence harness is a diagnostic tool for interactive investigation, not a replacement for the focused unit-style checks above. When using it, record the exact scenario, noise mode, seeds, NHC noise settings, and acceptance thresholds with the observed deltas so the result can be reproduced before promoting anything into a golden test.

For a synthetic scenario:

```bash
cargo run -p sim --bin filter_equivalence_harness -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise truth \
  --mount-mode ref \
  --sample-stride 2000 \
  --output /tmp/equiv_city_ref.csv
```

For a generic replay directory:

```bash
cargo run -p sim --bin filter_equivalence_harness -- \
  --generic-replay-dir /path/to/replay-dir \
  --mount-mode internal \
  --sample-stride 2000 \
  --output /tmp/equiv_replay_internal.csv
```

The harness also accepts `--attitude-roll-pitch-init-sigma-deg` for diagnostic
prior sweeps. Keep this as an experiment knob: a `5 deg` roll/pitch attitude
prior improves the synthetic roll-excitation figure-eight case, but broad replay
sweeps showed material regressions on several experimental logs, so the runtime
default remains the historical `2 deg`.

Summarize one or more harness CSVs with:

```bash
cargo run -p sim --bin filter_equivalence_summary -- /tmp/equiv_city_ref.csv
```

The summary reports Reduced-minus-Full deltas in a common physical basis for position, velocity, attitude, mount, gyro bias, accel bias, latest GNSS velocity residuals, and reference-relative attitude/mount errors when reference CSVs are available. Use the threshold flags on `filter_equivalence_summary` to mark the first time each axis crosses an investigation threshold. Use `--start-time-s` and `--end-time-s` to separate startup, convergence, and settled windows.

For update-allocation diagnostics, `covariance_history` can summarize a bounded interval:

```bash
cargo run -p sim --bin covariance_history -- \
  --synthetic-motion-def sim/motion_profiles/figure8_roll_excitation_30min.scenario \
  --synthetic-noise truth \
  --misalignment internal \
  --max-time-s 220 \
  --times 100,140,200 \
  --summary-window 100,200 \
  --allocation-csv /tmp/roll_excitation_alloc.csv
```

This prints Reduced update-type deltas, Full observation deltas, mount/attitude/velocity correction allocation, NHC residuals, covariance sigmas, and selected cross-correlations for the requested interval. When `--allocation-csv` is provided, the same window is written as per-event rows with innovation, NIS, correction allocation, covariance sigma, and reference-error context.

## Fixtures And Local Data

Small checked-in fixtures live under `sim/tests/fixtures/`. The `full_nsr_short` fixture is used by the seeded Full EKF replay path and includes IMU/GNSS CSV inputs plus expected summary data.

Synthetic replay tests use checked-in fixtures and Rust-native path generation. External simulator checkouts are not required for the default local loop.

## Generated-Code Changes

When changing symbolic model files or generated Rust under `sensor_fusion/src/generated_reduced/` or `sensor_fusion/src/generated_full/`:

```bash
python sensor_fusion/reduced.py --emit-rust
python sensor_fusion/ins_gnss_full.py --emit-rust
cargo test -p sensor_fusion --locked
cargo test -p sim --test full_parity --locked
cargo test -p sim --test synthetic_gnss_ins --locked
```

Review generated diffs carefully. The generated files are intentionally checked in so runtime builds do not need Python or SymPy.

## Replay And Visualizer Smoke Tests

For a generic replay directory:

```bash
cargo run --release -p sim --bin visualizer -- \
  --generic-replay-dir /path/to/replay-dir \
  --profile-only
```

For synthetic data:

```bash
cargo run --release -p sim --bin visualizer -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low \
  --profile-only
```

Use the full visualizer without `--profile-only` when checking plots, maps, and interactive trace behavior.

## Generic Replay Data

Hardware-specific conversion is intentionally outside this repository. External converters should write `imu.csv` and `gnss.csv` using the schema documented in the root README, then this repository consumes only those generic files. Optional `reference_attitude.csv` and `reference_mount.csv` files can be included for plots and evaluation; they are not fed back into the filters.

## Hosted Generic Dataset CI

Hosted replay datasets are described by `.github/datasets/generic-datasets.json`. Each dataset entry lists URLs for `imu.csv`/`gnss.csv`, optional reference CSVs, their SHA-256 checksums, and optional byte counts. CI validates the manifest shape, downloads files into `.cache/generic-datasets`, verifies checksums, checks the generic CSV headers/numeric rows, then runs:

```bash
node scripts/validate_generic_datasets.mjs \
  --manifest .github/datasets/generic-datasets.json \
  --cache-dir .cache/generic-datasets \
  --work-dir target/generic-datasets \
  --smoke-profile
```

The smoke profile creates a bounded CSV subset before invoking `visualizer --profile-only`, so large hosted datasets can be represented without replaying the full log in every CI run. When changing hosted data, update `.github/datasets/logger-data.version` or the manifest so the Actions cache key changes.

Package hosted datasets with:

```bash
python3 -m unittest scripts.tests.test_package_dataset
python3 scripts/package_dataset.py /path/to/replay-dir /tmp/hosted-drive
```
