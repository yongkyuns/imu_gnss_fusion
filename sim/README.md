# `sim` Tooling Map

This crate mixes interactive tooling, stable evaluators, and one-off diagnostics. The intent of this file is to make cleanup decisions easier and to keep the supported entrypoints explicit.

## Canonical bins

- `visualizer`: primary interactive viewer for UBX logs
- `align_eval`: main real-log align evaluator
- `align_eval_gnss_ins_sim`: align evaluation on `gnss-ins-sim` datasets
- `eskf_eval_gnss_ins_sim`: ESKF evaluation on `gnss-ins-sim` datasets
- `run_loose_nsr`: seeded loose replay from prepared reference-state inputs
- `seeded_loose_stress_mc`: fast Rust Monte Carlo replay for seeded loose stress
- `compare_filter_runtime`: runtime benchmark harness
- `esp32_usb_replay`: device replay / serial tooling

## Debug and exploratory bins

- `analyze_loose_jump`
- `analyze_eskf_transition`
- `analyze_eskf_stop`
- `analyze_eskf_mount_ab`
- `analyze_loose_bias_drift`
- `diag_loose_accel_z_bias`
- `check_reanchor`
- `analyze_esf_alg_behavior`
- `analyze_esf_alg_transition_speeds`
- `esf_alg_convention`
- `analyze_nav_grade`
- `sweep_mapping`

These are still useful, but they are not the stable surface area for automation or documentation.

## Visualizer mount modes

The visualizer and the main A/B analyzers use a single `--misalignment` option:

- `internal`: Align provides the initial mount seed, then ESKF estimates residual mount states.
- `external`: ESKF continuously follows Align and freezes its residual mount states.
- `ref`: ESF-ALG/reference mount angles are used.

Legacy aliases are still accepted where useful: `align`/`auto` map to `internal`,
`follow-align` maps to `external`, and `esf-alg`/`alg`/`manual` map to `ref`.
Loose keeps its historical behavior: `external` is treated like `internal`, so
loose uses the latched Align seed rather than continuously following Align.

## Loose diagnostics

Loose accel and gyro diagnostic plots use the same pre-rotated IMU stream that is
fed to the loose filter. The full loose mount plot composes the latched seed and
residual correction as `q_seed * inv(qcs)`, matching the physical mount convention
used by the ESKF plots.

The loose accel-bias states are additive correction states:

```text
corrected_accel = accel_scale * raw_accel + accel_bias
```

So a physical sensor bias has the opposite sign of the plotted loose bias state.
The loose Z-bias investigation showed that vertical accel bias is observable
through GNSS vertical position/velocity, but estimating accel scale and accel
bias together makes the Z channel underdetermined because the vehicle Z axis is
dominated by gravity. The visualizer loose path therefore keeps accel scale fixed
by default and lets accel bias absorb the remaining correction. Use
`diag_loose_accel_z_bias` for controlled synthetic checks and
`analyze_loose_bias_drift` for real-log trace sampling.

## Python scripts

Python should stay focused on:

- dataset generation
- sweeps and orchestration
- Plotly reporting

Replay-critical paths should prefer Rust. The seeded loose Monte Carlo path now follows that split.

## Shared code direction

Current shared modules worth expanding instead of duplicating in more bins:

- `src/datasets/seeded_loose.rs`: semicolon-delimited seeded-loose dataset parsing
- `src/datasets/gnss_ins_sim.rs`: `gnss-ins-sim` CSV parsing and sample loading
- `src/datasets/generic_replay.rs`: shared hardware-agnostic IMU/GNSS replay schema (`imu.csv`, `gnss.csv`)
- `src/datasets/ubx_replay.rs`: UBX-to-generic replay extraction using the same real-log timing / GNSS construction path
- `src/eval/gnss_ins.rs`: shared quaternion and simple GNSS kinematic helpers for the `gnss-ins-sim` evaluators
- `src/eval/replay.rs`: shared IMU/GNSS merge order for feeding `SensorFusion`
- `src/eval/state_summary.rs`: shared convergence/fluctuation/final-error summaries for scalar state traces with optional references
- `src/ubxlog.rs`: UBX log loading
- `src/visualizer/`: shared math and replay/pipeline pieces

## Generic replay format

The first common replay format is now:

- `imu.csv`
  - columns: `t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2`
- `gnss.csv`
  - columns: `t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,pos_std_n_m,pos_std_e_m,pos_std_d_m,vel_std_n_mps,vel_std_e_mps,vel_std_d_mps,heading_rad`
  - `heading_rad` may be `NaN` when heading is unavailable / intentionally omitted

Current producers:

- `export_gnss_ins_sim_generic`: export `gnss-ins-sim` datasets into the generic replay format
- `convert_ubx_to_generic_replay`: export UBX logs into the same generic replay format

Current consumers using the shared replay path:

- `eskf_eval_gnss_ins_sim`
- `eval_real_mount_reseed`
- `visualizer` / `ekf_compare` for the main real-log ESKF + loose injection path
- `analyze_eskf_transition`
- `analyze_loose_jump`
- `check_reanchor`
- `analyze_mount_recovery` indirectly via `build_plot_data(...)`

Still on older direct parsing / one-off replay glue:

- `align_eval`
- `esp32_usb_replay`
- `esf_alg_convention`
- `analyze_nav_grade`
- `analyze_esf_alg_behavior`
- `analyze_esf_alg_transition_speeds`
- `sweep_mapping`

This does not unify every real-log tool yet, but the primary synthetic evaluator, the main real-log visualizer path, and the main ESKF/loose debug analyzers now share one canonical injection format and one canonical IMU/GNSS merge order.

## Shared state summaries

The first shared summary path is wired into:

- `analyze_eskf_mount_ab --summary-csv ...`
- `eskf_eval_gnss_ins_sim --summary-csv ...`

For `eskf_eval_gnss_ins_sim`, the summary now covers:

- position N/E/D against truth
- velocity N/E/D against truth
- attitude roll/pitch/yaw against truth
- direct attitude quaternion-angle error against truth
- direct attitude forward-axis and down-axis errors against truth
- full mount roll/pitch/yaw against the configured truth mount
- direct mount quaternion-angle error against truth
- direct mount forward-axis and down-axis errors against truth
- full mount error magnitude
- gyro and accel bias states
- legacy seed/align/qcs mount diagnostics

The shared summary schema captures, per state:

- initial/final value
- duration and sample count
- early/tail fluctuation (`stddev`, `span`)
- tail drift
- optional reference-based error metrics (`final_error`, `MAE`, `RMSE`, `max_abs_error`, `p95_abs_error`)
- optional threshold-based settle time
