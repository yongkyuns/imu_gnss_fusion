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
cargo test -p sensor-fusion --locked
cargo test -p sim --locked
```

Run one integration test target:

```bash
cargo test -p sim --test synthetic_gnss_ins --locked
cargo test -p sim --test loose_parity --locked
cargo test -p sensor-fusion --test fusion_api --locked
```

## Test Coverage Map

| Area | Tests | What they protect |
| --- | --- | --- |
| Filter APIs | `ekf/tests/fusion_api.rs`, `align_api.rs`, `eskf_state_ops.rs` | Public API behavior, state initialization, and basic update contracts. |
| Loose observability | `ekf/tests/loose_mount_observability.rs`, `sim/tests/loose_parity.rs` | Loose mount behavior and parity with prepared seeded replay fixtures. |
| Synthetic replay | `sim/tests/synthetic_gnss_ins.rs` | Synthetic path generation, ESKF convergence, visualizer trace population, and optional `gnss-ins-sim` parity. |
| Generic replay | `sim/src/datasets/generic_replay.rs`, `sim/tests/synthetic_gnss_ins.rs` | Hardware-agnostic CSV schema loading and visualizer trace population. |
| Dataset packaging | `scripts/tests/test_package_dataset.py` | Hosted-data manifest generation, deterministic gzip staging, and raw binary rejection. |

## Fixtures And Local Data

Small checked-in fixtures live under `sim/tests/fixtures/`. The `loose_nsr_short` fixture is used by the seeded loose replay path and includes IMU/GNSS CSV inputs plus expected summary data.

Some tests and diagnostic commands can use an external `gnss-ins-sim` checkout. Those assets are not required for the default quick local loop. When a test is written to skip missing local data, treat the skip as expected unless you are specifically validating that integration.

## Generated-Code Changes

When changing symbolic model files or generated Rust under `ekf/src/generated_eskf/` or `ekf/src/generated_loose/`:

```bash
python ekf/eskf.py --emit-rust
python ekf/ins_gnss_loose.py --emit-rust
cargo test -p sensor-fusion --locked
cargo test -p sim --test loose_parity --locked
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
