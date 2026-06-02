# C API Backend

The C implementation lives under `c/` and provides C99, caller-owned context
APIs for integrations that cannot link the Rust crates directly. The public
contracts are:

- `c/sensor_fusion/include/sensor_fusion.h`
- `c/road_events/include/road_events.h`

The C API follows the same frame and unit conventions as the Rust crates:

- raw IMU samples stay in body frame `b`;
- GNSS velocity is local NED `[north, east, down]`;
- quaternions are scalar-first `[w, x, y, z]`;
- the public mount is the physical vehicle-to-body quaternion `q_bv`, with
  `R(q_bv) = C_bv` and `x_b = C_bv x_v`;
- optional input and output fields use explicit validity flags.

## Build

Build the static library and run the C API tests from the repository root:

```bash
make -C c test
```

The build uses plain Make, a C99 compiler, and the repo-local Unity-compatible
test shim in `c/third_party/unity/`. The static library artifact is written to:

```text
c/build/libimu_gnss_fusion_c.a
```

## Current Scope

The C implementation is an incremental parity target, not yet a replacement for
the Rust runtime. Current coverage is:

- C99 build and Unity test scaffold;
- Doxygen public headers for `sensor_fusion` and `road_events`;
- generated EKF fragments emitted from the same SymPy source used by Rust;
- generated-model tests for GNSS rows and the vehicle-roll-prior row;
- a private C EKF runtime layer for quaternion error-state injection, generated
  nominal prediction, sparse covariance prediction, scalar Joseph updates,
  reset-Jacobian application, GNSS position/velocity scalar wrappers, and
  Rust-style sequential batch updates for GNSS/NHC rows;
- private C GNSS outlier gating for position and velocity update groups,
  including consecutive-rejection events, gap bypass, reported-accuracy bypass,
  and the explicit next-sample gate-pass requirement;
- a manual-mount `sensor_fusion` facade path backed by the private EKF runtime
  for GNSS initialization, IMU prediction, gated GNSS updates, health/status
  snapshots, and query APIs;
- facade-level generated observations for decimated NHC, vehicle-frame speed,
  and the flat-road vehicle-roll prior in manual-mount mode;
- post-initialization GNSS scheduling that stores GNSS as pending work and
  fuses it at the following IMU epoch, optionally in the same batch as a
  same-epoch NHC update;
- manual-mount sleep/reseed handling with explicit `sensor_fusion_end_trip()`
  expected-sleep marking, short-sleep covariance aging, unexpected-gap GNSS
  reseed with attitude/calibration preservation, long-sleep GNSS reseed, and
  health-state reporting while navigation is unavailable;
- manual-mount startup policies matching Rust for yaw-observable GNSS seeding,
  current-position LLA queries, GNSS velocity standard-deviation normalization,
  and unknown-direction vehicle-speed handling;
- a C mount-alignment module ported from the Rust align logic, covering
  stationary tilt initialization, covariance prediction, gravity refinement,
  horizontal-acceleration yaw updates, turn-gyro roll/pitch updates,
  coarse-progress reporting, and focused Unity coverage;
- automatic C `sensor_fusion` facade handoff that keeps GNSS-only auto mode
  not-ready, initializes tilt from low-dynamic IMU samples, updates align from
  GNSS-to-GNSS motion windows plus averaged IMU, reports align progress before
  handoff, and seeds EKF mount covariance from align when coarse alignment
  becomes ready;
- manual-mount replay parity coverage for compact and dynamic Rust-golden
  streams that exercise IMU prediction, decimated NHC, pending GNSS fusion,
  vehicle-speed fusion, the flat-road roll prior, turn propagation, selected
  nominal states, mount state, event masks, and covariance diagonals;
- C `road_events` detectors for speed bumps, roughness/shocks, hills, reverse,
  harsh acceleration/braking/cornering presets, and trip statistics, including
  Rust-golden replay fixtures for speed bumps, roughness/shock events, hill,
  reverse, harsh accel/brake/cornering, and trip-summary outputs;
- native visualizer execution for generic replay directories and synthetic
  scenarios through `--backend c` or the native UI backend selector. The C path
  drives `sensor_fusion` and `road_events` through FFI and populates core
  position, velocity, attitude, mount, bias, covariance, map, roughness,
  bump-diagnostic, road-event, and trip-statistic traces.

Broader field replay parity remains pending. Rust remains the source of truth
for estimator behavior and for the browser/wasm visualizer until those cases
pass.

The public C API does not yet expose all Rust tuning controls. Align
configuration, align handoff delay, and explicit selection of align covariance
handoff are fixed at their current C defaults until those setters are added.

The C visualizer path does not yet expose the Rust-only EKF update inspector,
NHC correction diagnostics, or align debug-window internals. Those traces
remain empty in C-backend replay results until equivalent C diagnostics are
added.

## Parity Roadmap

The first estimator parity target was manual-mount runtime parity. The current C
branch also includes automatic alignment handoff, but replay parity must still
be expanded before the C backend should be considered interchangeable with Rust.

The implementation order is:

1. Port the private EKF state layer, process-noise configuration, GNSS gate
   state, and filter context.
2. Port quaternion and matrix helpers needed by the EKF, preserving the
   conventions `q_nv <- q_nv * dq` and `q_bv <- dq_bv * q_bv` during error-state
   injection.
3. Port GNSS initialization for manual `q_bv`, including yaw seeding,
   velocity/position initialization, covariance initialization, and residual
   mount covariance.
4. Connect IMU prediction to the generated nominal prediction, transition, and
   noise-input fragments.
5. Port scalar and batch covariance updates, error-state injection, and reset
   Jacobian handling.
6. Port GNSS, NHC, vehicle-speed, zero-velocity, and vehicle-roll-prior update
   scheduling.
7. Port the manual-mount `SensorFusion` facade flow and basic health states.
8. Port expected-sleep, unexpected-gap, and GNSS reseed policy for the
   manual-mount facade.
9. Port automatic mount alignment and EKF handoff.
10. Add native visualizer C execution for generic and synthetic replay.
11. Extend replay parity to broader field datasets and automatic mount
    alignment.
12. Expose C EKF/align/road-event diagnostics needed for full visualizer parity.

Parity tests should grow in the same order: generated rows, quaternion
injection, one-step prediction, covariance update, GNSS/NHC batches, small
manual-mount facade replays, dynamic manual-mount streams, sleep/reseed cases,
automatic align handoff, road-event replay fixtures including roughness, then
native visualizer smoke tests and broader field replays.

## Generated Model Workflow

The EKF formulation source is
`sensor_fusion/src/ekf/formulation.py`. It can emit both Rust and C generated
fragments:

```bash
python sensor_fusion/src/ekf/formulation.py --emit-all
```

C fragments are written under `c/sensor_fusion/src/ekf/generated/` and wrapped
by `c/sensor_fusion/src/ekf/generated_model.c`. The wrapper keeps the generated
rows in fixed-size C arrays so parity tests can compare the same scalar
observation rows and transition matrices against Rust.
