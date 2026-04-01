# sensor_fusion C Migration Plan

This document defines the parity-first migration from the current Rust implementation in
`sensor_fusion` to a standalone POSIX-compliant C library under `sensor_fusion/c`.

The target end state is:

- all runtime implementation lives in C
- Rust becomes a thin FFI wrapper with no duplicated filter logic
- the C library is standalone, heap-free, OS-free, single-precision only
- public APIs are stable and documented with Doxygen
- internal/testing/debug APIs are exposed through a separate internal header
- parity is demonstrated against the current Rust behavior before Rust logic is removed

## Goals

1. Preserve EKF numerical behavior exactly where possible.
2. Preserve `Align` behavior exactly where possible.
3. Preserve the top-level `SensorFusion` API semantics.
4. Make the C library buildable and testable without Cargo.
5. Ensure no heap allocation, no time/syscalls, and no hidden global state.

## Non-Negotiable Constraints

- C library must be standalone and POSIX compliant.
- No heap use at runtime.
- No OS access at runtime.
- No double-precision state in filter math.
  - Temporary offline generation/test code may use higher precision if needed.
- Public API must use fixed-size structs and caller-owned memory only.
- Rust crate must eventually contain only FFI declarations and glue.

## Current Baseline

Current implementation split:

- EKF:
  - Rust: `ekf/src/ekf.rs`
  - C: `ekf/c/ekf.c`, `ekf/c/ekf.h`
  - generated math:
    - Rust: `ekf/src/ekf_generated/*.rs`
    - C: `ekf/c/generated/*.c`
- Align:
  - Rust only: `ekf/src/align.rs`, `ekf/src/stationary_mount.rs`
- Top-level fusion wrapper:
  - Rust only: `ekf/src/fusion.rs`

## Required End-State C Layout

Target layout under `ekf/c/`:

- `include/sensor_fusion.h`
  - public stable API
- `include/sensor_fusion_internal.h`
  - internal structs/helpers required by sim/tests/debugging
- `src/sensor_fusion.c`
  - top-level fusion wrapper implementation
- `src/sf_align.c`
  - Align implementation
- `src/sf_stationary_mount.c`
  - stationary bootstrap implementation
- `src/sf_ekf.c`
  - EKF implementation
- `src/generated/*.c`
  - sympy-generated EKF math
- `tests/*.c`
  - Unity test suites
- `third_party/unity/`
  - vendored Unity sources/headers
- `Makefile`
  - standalone build + test

## Public API Design

The C public API should mirror the Rust `SensorFusion` semantics, not the current fragmented EKF API.

Public object:

- `sf_sensor_fusion_t`

Public config:

- `sf_predict_noise_t`
- `sf_align_config_t`
- `sf_bootstrap_config_t`
- `sf_fusion_config_t`

Public input samples:

- `sf_imu_sample_t`
- `sf_gnss_sample_t`

Public update/status:

- `sf_update_t`

Key API:

- `void sf_fusion_init_internal(sf_sensor_fusion_t*, const sf_fusion_config_t*)`
- `void sf_fusion_init_external(sf_sensor_fusion_t*, const sf_fusion_config_t*, const float q_vb[4])`
- `void sf_fusion_set_misalignment(sf_sensor_fusion_t*, const float q_vb[4])`
- `sf_update_t sf_fusion_process_imu(sf_sensor_fusion_t*, const sf_imu_sample_t*)`
- `sf_update_t sf_fusion_process_gnss(sf_sensor_fusion_t*, const sf_gnss_sample_t*)`
- `const sf_ekf_t* sf_fusion_ekf(const sf_sensor_fusion_t*)`
- `const sf_align_t* sf_fusion_align(const sf_sensor_fusion_t*)`
- `bool sf_fusion_mount_ready(const sf_sensor_fusion_t*)`
- `void sf_fusion_mount_q_vb(const sf_sensor_fusion_t*, float out_q_vb[4])`

Rules:

- no hidden allocation
- caller owns all storage
- null-safe documented behavior

## Internal Header Scope

`sensor_fusion_internal.h` should expose only what sim/tests/debugging need:

- raw `sf_align_t`
- raw `sf_ekf_t`
- internal covariance/state accessors
- debug helper structs

It must not become a second public API.

## Migration Phases

### Phase 1: Freeze Baseline Behavior

Before moving logic:

1. Add golden parity datasets for:
   - EKF scalar outputs
   - Align covariance and mount outputs
   - `SensorFusion` mount-ready / EKF-init transitions
2. Record baseline traces from Rust for:
   - representative nominal log
   - difficult align log
   - external misalignment path
3. Add deterministic replay harnesses to compare sample-by-sample outputs.

Exit criteria:

- baseline golden outputs stored
- reproducible parity harness available

### Phase 2: Normalize Current C EKF

Current C EKF already exists, but needs to be brought into the target library structure.

Tasks:

1. Rename `ekf.c/h` to target naming/layout.
2. Keep generated sympy C path as the single EKF source of truth.
3. Verify Rust and C EKF remain bit-close through current parity tests.
4. Move existing Rust parity tests into C data-driven fixtures as well.

Exit criteria:

- `sf_ekf.c` + `generated/*.c` build standalone
- Rust FFI can call the C EKF without Rust fallback logic

### Phase 3: Port Align to C

Port:

- `ekf/src/align.rs`
- `ekf/src/stationary_mount.rs`

Approach:

1. Translate helper math first.
2. Preserve function boundaries where possible.
3. Keep all arrays fixed-size.
4. Keep same covariance update formulas and thresholds.
5. Expose enough internal state in internal header for parity tests.

Critical parity points:

- stationary bootstrap output `q_vb`
- `gravity_lp_b`
- covariance diagonals
- `coarse_alignment_ready`
- update trace semantics

Exit criteria:

- C Align matches Rust Align on recorded logs within agreed tolerance

### Phase 4: Port SensorFusion Wrapper to C

Translate Rust `SensorFusion` wrapper from `ekf/src/fusion.rs`.

Tasks:

1. Introduce `sf_sensor_fusion_t`.
2. Port bootstrap detector.
3. Port IMU/GNSS sequencing and interval aggregation.
4. Port EKF init-from-GNSS behavior.
5. Port external misalignment path.

Exit criteria:

- C wrapper reproduces Rust `SensorFusion` state transitions

### Phase 5: Rust Becomes FFI Only

After C parity is proven:

1. Replace Rust filter implementations with `extern "C"` bindings.
2. Keep Rust-side type wrappers minimal.
3. Preserve crate API shape for downstream users.
4. Remove Rust implementations of:
   - `ekf.rs`
   - `align.rs`
   - `stationary_mount.rs`
   - `fusion.rs`

Exit criteria:

- Rust runtime contains only FFI wrappers/glue
- tests pass against C backend only

### Phase 6: Final Removal of Rust Logic

1. Remove dead Rust math/codegen duplicates.
2. Remove obsolete Rust tests that duplicate C coverage.
3. Keep only FFI integration tests in Rust where useful.

## Test Strategy

### C Unit Tests with Unity

Required suites:

- `test_ekf_init.c`
- `test_ekf_predict.c`
- `test_ekf_gps_fusion.c`
- `test_ekf_body_vel.c`
- `test_align_stationary_bootstrap.c`
- `test_align_gravity_update.c`
- `test_align_heading_update.c`
- `test_align_turn_gyro_update.c`
- `test_align_readiness.c`
- `test_sensor_fusion_external_mount.c`
- `test_sensor_fusion_internal_align.c`
- `test_sensor_fusion_replay_parity.c`

### Parity Harnesses

Need deterministic fixtures comparing C output against baseline Rust outputs:

- state vectors
- covariance diagonals
- mount quaternions
- readiness booleans
- update traces

Comparison policy:

- exact where generated EKF math is identical
- tolerance-based where quaternion normalization/order introduces tiny differences

### Dataset Replay Tests

At least:

- one easy baseline log
- one difficult align observability log
- one external misalignment case
- one GNSS outage case

## Tolerance Policy

Define explicit tolerances before deleting Rust:

- EKF state: per-field abs tolerance
- EKF covariance: per-diagonal abs tolerance
- Align quaternion: angle tolerance
- Align sigma: abs tolerance
- readiness/init transitions: exact sample parity

If any tolerance must be relaxed, document why and keep the rationale in the test.

## Sympy / Code Generation

EKF:

- continue using the current sympy pipeline to generate C as the source of truth
- Rust should eventually stop carrying generated `.rs` duplicates

Align:

- initial port can be handwritten
- only introduce generation if repeated algebra proves error-prone

## Doxygen Requirements

`sensor_fusion.h` must include:

- module-level overview
- lifecycle rules
- ownership rules
- units for every field
- thread-safety statement
- determinism statement
- runtime constraints

## Risks

1. Silent numerical drift in Align covariance updates.
2. API drift between Rust wrapper and C core.
3. Generated EKF and handwritten Align ending up with inconsistent conventions.
4. Losing debug visibility needed by sim.

## Risk Mitigations

1. Baseline golden traces before porting.
2. Internal header for sim/debug.
3. Replay parity tests on real logs.
4. Migrate one layer at a time:
   - EKF core
   - Align
   - SensorFusion wrapper
   - Rust FFI-only

## Immediate Next Steps

1. Add standalone C library scaffolding:
   - headers
   - makefile
   - Unity test harness
2. Rename current C EKF files into target naming or wrap them.
3. Add parity fixture generation from current Rust implementation.
4. Port Align into C while preserving internal state structure.

