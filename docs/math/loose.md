# Loose INS/GNSS Notes

The loose filter is a Rust port/reference path for a loose INS/GNSS formulation. It is used for seeded replay, visualizer comparison, and observability diagnostics beside the augmented ESKF.

## State Layout

The Rust nominal state in `ekf/src/loose.rs` contains:

```text
q0 q1 q2 q3
vn ve vd
pn pe pd
bgx bgy bgz
bax bay baz
sgx sgy sgz
sax say saz
qcs0 qcs1 qcs2 qcs3
```

The 24-dimensional error state follows the same groups without quaternion scalar components:

```text
attitude, velocity, position, gyro_bias, accel_bias, gyro_scale, accel_scale, mount
```

The process-noise vector has 21 components: gyro noise, accelerometer noise, gyro-bias random walk, accelerometer-bias random walk, gyro-scale random walk, accelerometer-scale random walk, and mount random walk.

## Measurements

The loose path uses:

- GNSS position and velocity updates in navigation coordinates.
- Non-holonomic body/vehicle velocity constraints for lateral and vertical velocity.
- IMU propagation from pre-rotated replay samples in the visualizer path.

The NHC update is gated by speed in the Rust implementation so very low-speed samples do not overconstrain lateral/vertical velocity.

## Mount Convention

The visualizer loose path uses the same coarse Align seed as ESKF. Diagnostic plots compose the full loose mount as:

```text
q_full = q_seed * inv(q_cs)
```

This matches the physical mount convention used by the ESKF plots. When debugging mount behavior, compare the seed, residual, and composed full mount separately before interpreting final roll/pitch/yaw values.

## Bias And Scale Diagnostics

Loose accelerometer bias states are additive correction states:

```text
corrected_accel = accel_scale * raw_accel + accel_bias
```

That means a physical sensor bias has the opposite sign of the plotted loose bias state. The current diagnostics keep accelerometer scale fixed by default in the visualizer loose path because gravity-dominated vertical motion makes simultaneous vertical accel scale and bias estimation underdetermined.

Useful diagnostics:

- `diag_loose_accel_z_bias`: controlled synthetic checks for vertical accelerometer bias behavior.
- `analyze_loose_bias_drift`: real-log trace sampling for loose bias drift.
- `analyze_loose_jump`: replay-oriented checks around loose state jumps.

## Generated Reference Code

The symbolic source is `ekf/ins_gnss_loose.py`. It emits Rust snippets into:

```text
ekf/src/generated_loose/reference_error_transition_generated.rs
ekf/src/generated_loose/reference_error_noise_input_generated.rs
ekf/src/generated_loose/reference_nhc_y_generated.rs
ekf/src/generated_loose/reference_nhc_z_generated.rs
```

Regenerate with:

```bash
python ekf/ins_gnss_loose.py --emit-rust
```

Then run:

```bash
cargo test -p sensor-fusion --test loose_mount_observability --locked
cargo test -p sim --test loose_parity --locked
```

## Seeded Loose Replay

The stable seeded replay binary is:

```bash
cargo run --release -p sim --bin run_loose_nsr -- --help
```

The shared parser for semicolon-delimited seeded-loose datasets lives in `sim/src/datasets/seeded_loose.rs`. The short checked-in parity fixture is under `sim/tests/fixtures/loose_nsr_short/`.
