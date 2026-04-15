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
- `src/ubxlog.rs`: UBX log loading
- `src/visualizer/`: shared math and replay/pipeline pieces
