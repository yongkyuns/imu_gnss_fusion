# Testing

This project has three main test layers: Rust workspace tests,
simulator/evaluator fixtures, and generated-model checks. Start with the Rust
commands below and only run data-heavy or external-reference checks when
changing replay or model-generation behavior.

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

Run representative integration tests:

```bash
cargo test -p sensor_fusion --test fusion_api --locked
cargo test -p sensor_fusion --test align_api --locked
cargo test -p sim --test synthetic_gnss_ins --locked
```

## Test Coverage Map

| Area | Tests | What they protect |
| --- | --- | --- |
| Public API | `sensor_fusion/tests/fusion_api.rs`, `align_api.rs` | Facade behavior, readiness, initialization, and mount handling. |
| State operations | State-operation integration tests | Injection, reset Jacobians, prediction, update contracts, and covariance hygiene. |
| Synthetic replay | `sim/tests/synthetic_gnss_ins.rs` | Synthetic path generation, EKF convergence, visualizer trace population, and checked-in replay fixtures. |
| Generic replay | `sim/src/datasets/generic_replay.rs`, `sim/tests/synthetic_gnss_ins.rs` | Hardware-agnostic CSV schema loading and visualizer trace population. |
| Dataset packaging | `scripts/tests/test_package_dataset.py` | Hosted-data manifest generation, deterministic gzip staging, and raw binary rejection. |

## Diagnostics

Use diagnostic binaries for investigation and regression characterization, not
as replacements for focused tests. When preserving a diagnostic result, record
the exact command, scenario or dataset, mount mode, noise settings, time window,
thresholds, and artifact paths.

Use covariance and update-allocation diagnostics when a replay shows surprising
mount, attitude, residual, or NIS behavior.

For mount observability checks on synthetic roll/pitch scenarios:

```bash
cargo run --release -p sim --bin diag_mount_observability
```

Treat diagnostic columns as implementation details; compare physical quantities
only after confirming frame, units, and sign convention.

## Fixtures And Local Data

Small checked-in fixtures live under `sim/tests/fixtures/`. Synthetic replay
tests use checked-in fixtures and Rust-native path generation. External
simulator checkouts are not required for the default local loop.

## Generated-Code Changes

When changing symbolic model files or generated Rust:

```bash
python sensor_fusion/src/ekf/formulation.py --emit-rust
cargo test -p sensor_fusion --locked
cargo test -p sim --test synthetic_gnss_ins --locked
```

Review generated diffs carefully. Generated files are checked in so runtime
builds do not need Python or SymPy.

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

Use the visualizer without `--profile-only` when checking plots, maps, and
interactive trace behavior.

## Generic Replay Data

Hardware-specific conversion is intentionally outside this repository. External
converters should write `imu.csv` and `gnss.csv` using the schema documented in
the root README. Optional reference CSVs can be included for plots and
evaluation; they are not fed back into the EKF during automatic replay.

## Hosted Generic Dataset CI

Hosted replay datasets are described by
`.github/datasets/generic-datasets.json`. CI validates the manifest shape,
downloads files into `.cache/generic-datasets`, verifies checksums, checks CSV
headers and numeric rows, then runs:

```bash
node scripts/validate_generic_datasets.mjs \
  --manifest .github/datasets/generic-datasets.json \
  --cache-dir .cache/generic-datasets \
  --work-dir target/generic-datasets \
  --smoke-profile
```

Package hosted datasets with:

```bash
python3 -m unittest scripts.tests.test_package_dataset
python3 scripts/package_dataset.py /path/to/replay-dir /tmp/hosted-drive
```
