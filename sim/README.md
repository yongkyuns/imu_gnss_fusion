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
- `check_reanchor`
- `analyze_esf_alg_behavior`
- `analyze_esf_alg_transition_speeds`
- `esf_alg_convention`
- `analyze_nav_grade`
- `sweep_mapping`

These are still useful, but they are not the stable surface area for automation or documentation.

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
- `src/eval/gnss_ins.rs`: shared quaternion and simple GNSS kinematic helpers for the `gnss-ins-sim` evaluators
- `src/eval/state_summary.rs`: shared convergence/fluctuation/final-error summaries for scalar state traces with optional references
- `src/ubxlog.rs`: UBX log loading
- `src/visualizer/`: shared math and replay/pipeline pieces

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
