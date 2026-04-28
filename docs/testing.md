# Testing

This project has three main test layers: Rust workspace tests, simulator/evaluator fixtures, and generated-model parity checks. Most contributors should start with the Rust commands below and only run data-heavy or external-reference checks when changing replay, parser, or model-generation behavior.

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
cargo test -p ublox --locked
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
| UBX parsing | `ublox/tests/*.rs` | Parser behavior, generated packet support, and proptest regressions. |

## Fixtures And Local Data

Small checked-in fixtures live under `sim/tests/fixtures/`. The `loose_nsr_short` fixture is used by the seeded loose replay path and includes IMU/GNSS CSV inputs plus expected summary data.

Some tests and diagnostic commands can use local real logs under `logger/data/` or an external `gnss-ins-sim` checkout. Those assets are not required for the default quick local loop. When a test is written to skip missing local data, treat the skip as expected unless you are specifically validating that integration.

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

For a UBX log:

```bash
cargo run --release -p sim --bin visualizer -- /path/to/ubx_raw_*.bin --profile-only
```

For synthetic data:

```bash
cargo run --release -p sim --bin visualizer -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low \
  --profile-only
```

Use the full visualizer without `--profile-only` when checking plots, maps, and interactive trace behavior.

## Parser And Fuzz-Regression Tests

The local `ublox` crate includes focused parser tests and proptest regression seeds. For parser work, run:

```bash
cargo test -p ublox --locked
```

For large binary dump parsing, see the ignored test notes in `ublox/tests/parser_binary_dump_test.rs`; those require a local `UBX_BIG_LOG_PATH`.
