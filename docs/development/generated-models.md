# Generated EKF Model Code

This page explains the generated EKF model boundary. Normal Rust builds do not
run SymPy; the checked-in Rust fragments are part of the source tree. The Python
generator is used only when the mathematical structure of the EKF changes.

```{figure} ../_static/diagrams/sympy-generation-architecture-orthogonal.svg
:alt: SymPy generation architecture from formulation.py through generated Rust fragments and EKF runtime wrappers.
:class: framed

`formulation.py` is the symbolic source. The checked-in generated fragments are
normal Rust inputs to the EKF runtime.
```

## Why SymPy Is Used

The EKF combines quaternion perturbations, raw-body IMU biases, residual mount
states, local-level inertial propagation, and scalar vehicle/GNSS observations.
Those equations are easy to get subtly wrong by hand. The project keeps the
symbolic model in Python so the linearization is explicit and reviewable, while
the runtime stays small and deterministic:

- `formulation.py` defines the nominal state, 18-dimensional error state, 15
  process-noise inputs, perturbation/injection convention, propagation graph,
  reset block, and scalar observation functions.
- SymPy differentiates the perturb-propagate-linearize construction to produce
  discrete `F`, `G`, observation rows `H`, scalar gains `K`, and innovation
  variances `S`.
- Common-subexpression elimination keeps generated Rust snippets compact enough
  for `no_std` and embedded targets.
- `sensor_fusion/src/ekf/generated.rs` is the hand-written boundary that turns
  include-file assignments into typed Rust functions consumed by
  `sensor_fusion/src/ekf/mod.rs`.

The checked-in generated Rust fragments live under:

```text
sensor_fusion/src/ekf/generated/
```

The symbolic source of truth is:

```text
sensor_fusion/src/ekf/formulation.py
```

## Model ABI

The generated fragments and the hand-written wrappers share a fixed ABI. The
nominal state order is:

$$
\begin{bmatrix}
q_{nv} & v_n & p_n & b_g & b_a & q_{bv}
\end{bmatrix}.
$$

The error-state order is:

$$
\begin{aligned}
0{:}2 &: \delta\theta_v,&
3{:}5 &: \delta v_n,&
6{:}8 &: \delta p_n,\\
9{:}11 &: \delta b_g,&
12{:}14 &: \delta b_a,&
15{:}17 &: \delta\psi_{bv}.
\end{aligned}
$$

The process-noise order is three axes each of:

$$
\begin{bmatrix}
n_{\Delta\alpha_b} &
n_{\Delta v_b} &
n_{b_g} &
n_{b_a} &
n_{\psi_{bv}}
\end{bmatrix}.
$$

This ordering is the contract between `formulation.py`, generated include
files, `generated.rs`, `ekf/types.rs`, and `state_ops.rs`.

## What The Generator Emits

The generator emits only algebra that follows directly from the symbolic model.
It does not encode runtime policy.

| Generator source | Generated fragment | Wrapper boundary | Runtime consumer |
| --- | --- | --- | --- |
| `emit_nominal_prediction_rust()` | `nominal_prediction_generated.rs` | `predict_nominal_with_gravity()` | `Ekf::predict()` |
| `derive_error_dynamics()` | `error_transition_generated.rs`, `error_noise_input_generated.rs` | `error_transition_with_gravity()` | covariance prediction |
| `emit_matrix_supports()` | `error_transition_support_generated.rs` | constants included by `generated.rs` | sparse covariance prediction |
| `build_symbolic_model()` reset block | `attitude_reset_jacobian_generated.rs` | `attitude_reset_jacobian()` | attitude covariance reset |
| `derive_measurement_model()` | `gps_*_generated.rs` | `gps_*_observation()` | GNSS scalar rows and GNSS/NHC batch |
| `derive_measurement_model()` | `stationary_accel_*_generated.rs` | `stationary_accel_*_observation()` | stationary gravity updates |
| `derive_measurement_model()` | `body_vel_*_generated.rs` | `body_vel_*_observation()` | vehicle speed, NHC Y/Z, GNSS/NHC batch |

Each generated scalar observation file computes the linearized row $H$, scalar
Kalman gain $K = PH^T / S$, and innovation variance $S = HPH^T + R$ for the
current covariance. The runtime still supplies the physical innovation
$z - h(x_\mathrm{nom})$. That split is important: generated code knows the local
linearization; `ekf/mod.rs` decides whether a measurement is valid, how it is
gated, whether it is batched with other rows, and when the accumulated error is
injected into the nominal state.

## Generated/Runtime Boundary

```{figure} ../_static/diagrams/sympy-update-boundary-orthogonal.svg
:alt: Dataflow from symbolic EKF formulation through generated fragments, Rust wrappers, runtime policy, and semantic checks.
:class: framed

The generated files carry algebra. The hand-written Rust layers carry policy:
measurement validity, batching, covariance update strategy, diagnostics, and
public API behavior.
```

Regenerate EKF fragments only when changing the estimator model:

```bash
python sensor_fusion/src/ekf/formulation.py --emit-rust
```

Changes usually fall into four categories:

| Change type | Primary files | What must stay consistent |
| --- | --- | --- |
| Propagation or noise change | `propagate_nominal()`, `inject_true_state()`, `extract_error_state()`, `derive_error_dynamics()` | nominal propagation, perturbation side, $F$, $G$, support metadata, runtime $Q$ scaling |
| State or ABI change | `build_symbolic_model()`, `generated.rs`, `ekf/types.rs`, `state_ops.rs` | nominal/error/noise ordering and wrapper argument bindings |
| New or changed scalar observation | `derive_measurement_model()`, `write_observation_equations()` | predicted scalar, variance symbol, generated $H/K/S$, runtime residual sign |
| New generated function boundary | `generated.rs`, sometimes `mod.rs` | wrapper inputs, variance symbol, scalar residual sign, gravity/local-frame arguments |
| Runtime policy change | `mod.rs`, `fusion.rs` | gating, batching, covariance update form, diagnostics, public event/state reporting |

The most important semantic check is whether a generated diff is explainable
from the intended model change. For example:

- a propagation change should alter `nominal_prediction`, $F$, and often $G$;
- a noise-input change should alter $G$, support metadata, and runtime $Q$
  construction;
- an observation change should alter one scalar `H/K/S` family and the runtime
  residual should keep the same sign convention;
- a pure gating or scheduling change should not modify generated algebra.

## Implementation Checks

- State ordering in `formulation.py`, `generated.rs`, `ekf/types.rs`, and
  `state_ops.rs` must agree.
- The nominal/error convention must match [](../algorithms/frames.md): EKF
  vehicle attitude uses right injection, residual mount uses left injection.
- `F` and `G` are discrete matrices for one IMU increment; runtime process
  noise scaling must match the powers of `dt` already present in generated `G`.
- Generated row-support tables must include all nonzero entries used by sparse
  covariance prediction.
- Scalar observation wrappers expose `H`, `K`, and `S`; runtime code still owns
  innovation construction and measurement acceptance.
- Runtime EKF docs should describe behavior at the algorithm level, while this
  page explains why generated fragments changed.

## Verification Commands

For broad EKF formulation changes, the useful evidence is not just that Rust
compiles; it is that generated algebra, wrapper boundaries, and runtime policy
still agree. A typical verification set is:

```bash
cargo fmt --all -- --check
cargo test -p sensor_fusion --test ekf_state_ops --locked
cargo test -p sensor_fusion --test ekf_nhc_jacobian --locked
cargo test -p sensor_fusion --test ekf_update_diag --locked
cargo test -p sensor_fusion --locked
cargo check -p sim --bin visualizer --locked
. .venv-docs/bin/activate
sphinx-build -W --keep-going -b html docs target/docs-html
```
