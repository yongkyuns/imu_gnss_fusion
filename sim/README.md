# `sim` Tooling Map

This crate contains the hardware-agnostic replay, evaluation, synthetic-data, and visualization tooling around the `sensor-fusion` library.

## Canonical Binaries

- `visualizer`: interactive viewer for generic CSV replay directories and synthetic scenarios.
- `export_gnss_ins_sim_generic`: exporter from `gnss-ins-sim` outputs into the generic replay schema.
- `run_loose_nsr`: seeded loose replay from prepared reference-state inputs.
- `seeded_loose_stress_mc`: fast Rust Monte Carlo replay for seeded loose stress.
- `compare_filter_runtime`: runtime benchmark harness.
- `diag_loose_accel_z_bias`: controlled loose Z-accelerometer-bias diagnostic.
- `synthetic_bad_basin_sweep`: synthetic early-convergence stress sweep.

## Visualizer Inputs

Run a generic replay directory containing `imu.csv` and `gnss.csv`:

```bash
cargo run --release -p sim --bin visualizer -- \
  --generic-replay-dir /path/to/replay-dir
```

Run a synthetic motion scenario:

```bash
cargo run --release -p sim --bin visualizer -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low
```

The same generic CSV parser is used by the native visualizer and the browser visualizer.

## Visualizer Mount Modes

The visualizer uses a single `--misalignment` option:

- `internal`: Align provides the initial mount seed, then ESKF estimates residual mount states.
- `external`: ESKF continuously follows Align and freezes its residual mount states.
- `ref`: Reference mount angles are used when a synthetic or converted dataset provides them.

Loose keeps its historical behavior: `external` is treated like `internal`, so loose uses the latched Align seed rather than continuously following Align.

## Generic Replay Format

The common replay format is a directory with:

- `imu.csv`
  - columns: `t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2`
- `gnss.csv`
  - columns: `t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,pos_std_n_m,pos_std_e_m,pos_std_d_m,vel_std_n_mps,vel_std_e_mps,vel_std_d_mps,heading_rad`
  - `heading_rad` may be `NaN` when heading is unavailable or intentionally omitted
- Optional `reference_attitude.csv` and `reference_mount.csv`
  - columns: `t_s,roll_deg,pitch_deg,yaw_deg`
  - used by visualizer comparison plots only; filter inputs still come from the public generic IMU/GNSS path

Hardware-specific converters should live outside this repository and emit this schema.

## Hosted Dataset Packaging

Use the repository-level packaging script to convert a generic replay directory into a static-hosting layout:

```bash
python3 scripts/package_dataset.py /path/to/replay-dir /tmp/hosted-drive
```

The output directory contains `manifest.json`, `imu.csv.gz`, `gnss.csv.gz`, and any optional reference CSVs found in the source replay. The manifest records the generic replay schema, sample counts, time span, compressed file sizes, and SHA-256 hashes.

The script also accepts `gnss-ins-sim` output directories by invoking `export_gnss_ins_sim_generic` before packaging:

```bash
python3 scripts/package_dataset.py /path/to/gnss-ins-sim/output /tmp/hosted-drive \
  --source-format gnss-ins-sim \
  --signal-source meas
```

Raw UBX or other `.bin` logs are not packaged here. Rebuilding that conversion path would reintroduce the device-specific UBX parser stack removed from this repository; external converters should emit the generic CSV schema first.

## Shared Code Direction

Prefer these shared modules instead of duplicating replay logic in new tools:

- `src/datasets/generic_replay.rs`: hardware-agnostic IMU/GNSS replay schema.
- `src/datasets/gnss_ins_sim.rs`: `gnss-ins-sim` CSV parsing and sample loading.
- `src/datasets/seeded_loose.rs`: semicolon-delimited seeded-loose dataset parsing.
- `src/eval/replay.rs`: canonical IMU/GNSS merge order for feeding `SensorFusion`.
- `src/eval/state_summary.rs`: convergence, fluctuation, and final-error summaries.
- `src/visualizer/pipeline/generic.rs`: generic replay pipeline used by native and browser visualization.
- `src/visualizer/pipeline/synthetic.rs`: synthetic replay pipeline.

## Loose Diagnostics

Loose accel and gyro diagnostic plots use the same pre-rotated IMU stream that is fed to the loose filter. The full loose mount plot composes the latched seed and residual correction as `q_seed * inv(qcs)`, matching the physical mount convention used by the ESKF plots.

The loose accel-bias states are additive correction states:

```text
corrected_accel = accel_scale * raw_accel + accel_bias
```

So a physical sensor bias has the opposite sign of the plotted loose bias state. The loose Z-bias investigation showed that vertical accel bias is observable through GNSS vertical position/velocity, but estimating accel scale and accel bias together makes the Z channel underdetermined because the vehicle Z axis is dominated by gravity. The visualizer loose path therefore keeps accel scale fixed by default and lets accel bias absorb the remaining correction.
