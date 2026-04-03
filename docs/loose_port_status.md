# Loose INS/GNSS Port Status

## Goal

Port the MATLAB/Python loose INS/GNSS filter from:

- `/Users/ykshin/Dev/me/open-aided-navigation`

into this repo using the same structure as the ESKF path:

- symbolic model
- generated C
- Rust wrapper
- replay harness
- visualizer integration

The target is reference-faithful parity first, then side-by-side visualization.

## Current State

There is now a separate loose filter path in this repo:

- symbolic model: [ekf/ins_gnss_loose.py](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/ins_gnss_loose.py)
- generated C: [ekf/c/generated_loose](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/c/generated_loose)
- C runtime: [ekf/c/src/sf_loose.c](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/c/src/sf_loose.c)
- wrapper/noise types: [ekf/src/c_api.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/src/c_api.rs), [ekf/src/loose.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/src/loose.rs)
- comparison harness: [sim/src/bin/run_loose_nsr.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/sim/src/bin/run_loose_nsr.rs)

The loose path is not yet committed and is still under active debugging.

## Implementation Structure

The current loose implementation is split into four practical layers:

1. Symbolic model and code generation

- source: [ekf/ins_gnss_loose.py](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/ins_gnss_loose.py)
- generated snippets: [ekf/c/generated_loose](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/c/generated_loose)

This layer defines the symbolic reference math for:

- reference error transition `F`
- reference noise-input `G`
- reference NHC observation rows

It emits C snippets that are later included by the C runtime at specific call sites.

2. C runtime implementation

- ABI/state definitions: [ekf/c/include/sf_loose.h](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/c/include/sf_loose.h)
- runtime: [ekf/c/src/sf_loose.c](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/c/src/sf_loose.c)

This is the real filter implementation. It is mostly handwritten orchestration around generated algebra blocks.

Handwritten C owns:

- nominal propagation
- covariance prediction orchestration
- GPS whitening and gating
- NHC gating and observation assembly
- Joseph batch update
- error-state injection
- ECEF/LLH/gravity helpers
- precision-sensitive shadow state

The precision-sensitive shadow state currently includes:

- `pos_e64`
- `qcs64`
- `p64`

Generated C is included into this runtime for the dense symbolic math:

- transition/noise matrices
- reference NHC rows
- scalar GPS/body-velocity update kernels
- reset Jacobian

3. Rust FFI mirror and wrapper

- Rust noise/config: [ekf/src/loose.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/src/loose.rs)
- FFI mirror/wrapper: [ekf/src/c_api.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/src/c_api.rs)

Rust mirrors the C ABI manually with `#[repr(C)]` structs and `unsafe extern "C"` declarations.

`CLooseWrapper` is a thin ergonomic layer over the C runtime. It does not reimplement the filter. It mainly provides:

- construction and initialization helpers
- exact reference-state init entry points
- safe Rust method calls for predict/update operations
- access to shadow-state helpers used by replay diagnostics

4. Replay and parity harnesses

- local replay harness: [sim/src/bin/run_loose_nsr.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/sim/src/bin/run_loose_nsr.rs)
- MATLAB exact-step helper: [tmp_loose_matlab_exact_step.m](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/tmp_loose_matlab_exact_step.m)
- MATLAB replay dump helper: [tmp_loose_matlab_replay.m](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/tmp_loose_matlab_replay.m)

These sit above the Rust wrapper and are used to verify end-to-end parity against MATLAB.

## One-Step Call Flow

For a replayed loose filter epoch, the call path is:

1. [run_loose_nsr.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/sim/src/bin/run_loose_nsr.rs) loads IMU/GNSS samples and builds a `CLooseImuDelta`.

2. Rust calls `CLooseWrapper::predict()`.

3. `CLooseWrapper::predict()` forwards to `sf_loose_predict()`.

4. `sf_loose_predict()`:

- runs `sf_loose_predict_nominal()` for RK2/Heun nominal propagation
- calls `sf_loose_compute_error_transition()`
- `sf_loose_compute_error_transition()` includes the generated symbolic `F/G` blocks
- predicts covariance using the shadow `p64`

5. If a GPS and/or NHC update is eligible, [run_loose_nsr.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/sim/src/bin/run_loose_nsr.rs) calls `CLooseWrapper::fuse_reference_batch()`.

6. `CLooseWrapper::fuse_reference_batch()` forwards to `sf_loose_fuse_reference_batch()`.

7. `sf_loose_fuse_reference_batch()`:

- prepares GPS whitening and residuals
- computes corrected IMU values for NHC gating
- includes generated reference NHC row code
- assembles accepted observation rows and variances
- runs `sf_loose_batch_update_joseph()`

8. `sf_loose_batch_update_joseph()` updates `p64`, then injects the solved error state through `sf_loose_inject_error_state()`.

9. `sf_loose_inject_error_state()` updates:

- nominal quaternion `Q_ES`
- double-shadow position `pos_e64`
- nominal velocity/bias/scale states
- double-shadow mount quaternion `qcs64`

10. The replay harness reads the updated nominal/shadow state and records output/diagnostics.

## What Already Matches

The current loose replay now matches the MATLAB loose filter on the full NSR drive for the applied-observation sequence.

Verified parity:

- full applied-observation row count matches MATLAB
- full applied-observation timestamps match MATLAB
- full applied-observation type sequence matches MATLAB

The latest full-drive comparison is:

- local: `1629` applied-observation rows
- MATLAB: `1629` applied-observation rows
- first mismatch: none

MATLAB is now the golden reference for this port.

Small-step tests and isolated blocks are in good shape.

Passing checks:

- `make -C ekf/c test`
- `cargo test -p sensor-fusion --tests`
- `cargo check -p sim --bin run_loose_nsr`
- `cargo test -p sim --test loose_parity`

Passing building-block coverage now includes:

- FFI layout for loose noise/config
- exact reference-state init
- `predict(dt=0)` is a true no-op
- two-sample Heun/RK2 nominal propagation
- covariance predict `Phi P Phi^T + Q`
- GPS-only Joseph update
- NHC-only Joseph update
- combined GPS+NHC Joseph update

There is also now a fixture-based replay regression for refactors:

- test: [sim/tests/loose_parity.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/sim/tests/loose_parity.rs)
- fixture: [sim/tests/fixtures/loose_nsr_short](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/sim/tests/fixtures/loose_nsr_short)

That regression checks:

- the applied-observation sequence over a short replay window
- several state checkpoints across the same window

So the remaining mismatch is not in the basic one-step math anymore.

## Important Bugs Already Fixed

### 1. `dt <= 0` covariance bug

`sf_loose_predict()` used to propagate covariance even for zero `dt`, which effectively corrupted `P`.

Fix:

- early return in [ekf/c/src/sf_loose.c](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/c/src/sf_loose.c)

### 2. ECEF position precision bug

Keeping ECEF position only in `f32` at `~6.4e6 m` was too coarse and caused GPS updates to kick the state away from the reference.

Fix:

- added `double` ECEF shadow state (`pos_e64`)
- nominal public state still mirrors to `float`

This removed the large early divergence.

### 3. Pre-start GNSS sequencing bug in replay harness

The local harness was dropping the last GNSS sample just before `start_ttag`, while the Python reference still uses it at the first running step.

Fix:

- [sim/src/bin/run_loose_nsr.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/sim/src/bin/run_loose_nsr.rs) now keeps the latest GNSS sample even if it arrives just before `start_ttag`

### 4. GPS whitening bug

The local C code built the GPS whitening transform with the wrong triangular orientation.

Fix:

- corrected in [ekf/c/src/sf_loose.c](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/c/src/sf_loose.c)

### 5. NHC variance scaling

The NHC variance scaling was not matching the Python reference convention.

Fix:

- aligned to the reference `GYRO_FREQ = 50 Hz` semantics in [ekf/c/src/sf_loose.c](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/c/src/sf_loose.c)

### 6. Exact ECEF init truncation

The trusted reference init dump was being parsed into Rust with `f32` ECEF position, which immediately introduced about `0.235 m` of position error before the first GPS update.

Fix:

- preserve exact ECEF init position as `f64` into the loose double shadow state in [ekf/src/c_api.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/src/c_api.rs)
- parse the replay init JSON ECEF position as `f64` in [sim/src/bin/run_loose_nsr.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/sim/src/bin/run_loose_nsr.rs)

### 7. NHC chi-squared gate variance bug

The local reference NHC path was using the time-scaled observation variance for the chi-squared gate.

MATLAB does not do that. It uses:

- unscaled variance for rejection testing
- scaled variance only when the accepted observation is inserted

This showed up late in the run at `108.780003 s`, where MATLAB accepted only `VEL_C_Y` while the local port incorrectly accepted both `VEL_C_Y` and `VEL_C_Z`.

Fix:

- corrected in [ekf/c/src/sf_loose.c](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/c/src/sf_loose.c)
- corrected the matching reference helper in [ekf/tests/loose_api.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/ekf/tests/loose_api.rs)

## What Was Ruled Out

Several earlier hypotheses turned out to be false leads.

### False lead: first GPS step mismatch

After the GPS whitening fix and corrected reference diagnostics, the first GPS step matches closely.

### False lead: early NHC gate mismatch around `2.420005 s`

At one point it looked like the local replay was rejecting an NHC update that Python accepted.

This turned out to be caused by a bad/stale reference dump.

Clean direct tracing showed:

- local rejects NHC at `2.420005 s`
- Python also rejects NHC at `2.420005 s`

Both see approximately:

- `omega_norm ~= 0.029`
- `| ||f_s|| - 9.81 | > 0.2`

So this is not the root cause.

### False lead: Joseph update implementation mismatch

Offline replay of the local pre-update NHC row through the Joseph update reproduces the local posterior almost exactly.

That means:

- the local batch Joseph update is internally consistent
- the remaining mismatch is not a simple algebra bug in the update step

## Current Best Findings

### 1. MATLAB sequence parity is now achieved

The full replay no longer has an applied-observation sequencing mismatch against MATLAB.

The two fixes that closed the remaining gap were:

- preserving the exact initial ECEF position in `f64`
- matching MATLAB's NHC chi-squared gate variance semantics

### 2. MATLAB should remain the only golden reference for loose parity

Earlier Python-based diagnostics were useful for narrowing down suspects, but the final port-closing fixes came from exact MATLAB comparison.

## Current Artifacts Used For Debugging

Local repo harness:

- [sim/src/bin/run_loose_nsr.rs](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/sim/src/bin/run_loose_nsr.rs)

Useful temporary outputs under `/tmp`:

- `/tmp/loose_long_init.json`
- `/tmp/loose_long_local.json`
- `/tmp/loose_long_ref.json`
- `/tmp/loose_long_local_diag_60s.json`
- `/tmp/loose_ref_obsdiag_post_1s.json`
- `/tmp/loose_ref_preobs_1s.json`
- `/tmp/loose_ref_preobs_60s.json`

These are scratch diagnostics, not repo artifacts.

## Recommended Next Steps

1. Keep the MATLAB exact-step script and the local replay trace support around until the loose path is committed.

2. If a future regression appears, compare against MATLAB first and normalize single-element observation arrays when diffing JSON dumps.

3. Once the loose path is committed, add a compact regression check that compares the local applied-observation sequence against a trusted MATLAB dump for this NSR drive.

The safest method is:

- drive the Python filter with explicit `prop_state -> prop_cov -> _prep_obs -> meas_update -> correct_state`
- dump pre-update and post-update data from that exact sequence

4. Do not tune yet.

Current evidence says the issue is still parity/debugging, not filter tuning.

## Short Summary

The loose port is no longer grossly broken.

What now matches:

- propagation
- covariance predict
- GPS update
- NHC update
- early pre-update state

What remains:

- a slow cumulative mismatch, mainly in mount/alignment and yaw-related behavior over long replay

The next useful work is to inspect per-update `Q_CS` correction drift, not to add more model complexity or tune noise values yet.
