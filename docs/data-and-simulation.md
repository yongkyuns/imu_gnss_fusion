# Data And Simulation

The common replay format is hardware-agnostic and uses timestamped IMU/GNSS CSV files. Synthetic scenarios and field recordings both convert into this format so the replay, evaluation, and visualizer paths stay shared.

```{figure} _static/diagrams/overall-architecture.png
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

## Packaging

Use `scripts/package_dataset.py` to create a deterministic per-dataset package with `manifest.json`, `imu.csv.gz`, `gnss.csv.gz`, and optional reference streams. It packages one dataset directory; it does not update the browser or CI manifest lists.

Use `scripts/package_ios_motionfusion_dataset.py` for `.motionfusion` recordings produced by the iOS app. By default that wrapper packages the dataset and updates both `web/datasets/manifest.json` and `.github/datasets/generic-datasets.json`; pass `--no-update-web-manifest` or `--no-update-ci-manifest` to disable those updates.

The hosted generic dataset CI job validates the GitHub dataset manifest, schema, checksums, and smoke profile. Browser-facing Pages validation checks safe relative/HTTPS paths and fetchability for the static artifact.

## Hosted Data

The current browser manifest contains 31 datasets. See [](data/hosted-datasets.md) for the checked-in list.

## Synthetic Scenarios

The repository includes 14 `.scenario` files under `sim/motion_profiles`. They cover city blocks, figure-eight paths, grade/stops, high-speed straight motion, roll/pitch excitation, and robustness cases.

Generate a synthetic replay directory:

```bash
cargo run --release -p sim --bin export_synthetic_replay_generic -- \
  --motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --noise low \
  --output-dir /tmp/city-blocks
```
