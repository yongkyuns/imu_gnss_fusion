# Data And Simulation

The common replay format is hardware-agnostic and uses timestamped IMU/GNSS CSV files. Synthetic scenarios and field recordings both convert into this format so the replay, evaluation, and visualizer paths stay shared.

```{figure} _static/diagrams/overall-architecture-orthogonal.svg
:alt: Architecture diagram highlighting generic replay and visualization flow.
:class: framed

Synthetic scenarios, hosted datasets, and iOS exports all enter the shared generic replay path before they reach the visualizer and `SensorFusion` runtime.
```
## Generic Replay Directory

Required files:

`imu.csv`

```text
t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2
```

`gnss.csv`

```text
t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,pos_std_n_m,pos_std_e_m,pos_std_d_m,vel_std_n_mps,vel_std_e_mps,vel_std_d_mps,heading_rad
```

IMU gyro and acceleration columns are raw body-frame samples. GNSS velocity and standard-deviation columns are local NED. `heading_rad` may be `NaN` when heading is unavailable.

Optional reference files can provide attitude, position, motion, and mount streams for plots and evaluation. Reference data is not a normal runtime input unless a tool explicitly runs manual mount mode from reference mount.

## Hosted Data

The current browser manifest contains 32 datasets. See [](data/hosted-datasets.md) for the checked-in list.

## Synthetic Scenarios

The repository includes 14 `.scenario` files under `sim/motion_profiles`. They cover city blocks, figure-eight paths, grade/stops, high-speed straight motion, roll/pitch excitation, and robustness cases.

Generate a synthetic replay directory:

```bash
cargo run --release -p sim --bin export_synthetic_replay_generic -- \
  --motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --noise low \
  --output-dir /tmp/city-blocks
```

## Dataset Packaging And Validation

Hosted datasets are static packages: `manifest.json`, `imu.csv.gz`,
`gnss.csv.gz`, and optional gzip reference streams. `scripts/package_dataset.py`
validates CSV headers, row counts, time bounds, byte counts, and SHA-256 hashes
before writing the package manifest. It accepts an existing generic replay
directory or output from the synthetic replay exporter.

iOS packaging is layered on top of the same generic format.
`scripts/package_ios_motionfusion_dataset.py` runs
`mobile/ios/scripts/export_motionfusion.py` to convert `.motionfusion` JSON into
generic `imu.csv`/`gnss.csv`, then calls `scripts/package_dataset.py` and upserts
the web and CI dataset manifests. Browser loading consumes those manifest entries
and fetches required IMU/GNSS streams plus optional references.

Detailed publication commands and CI validation behavior live in
[](development/datasets.md).
