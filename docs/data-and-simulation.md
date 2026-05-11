# Data And Simulation

This page documents the repository boundary for replay data, synthetic
simulation, hosted datasets, and packaging. It reflects the current code in
`sim/src/datasets`, `sim/src/synthetic`, `sim/motion_profiles`,
`scripts/package_dataset.py`, `scripts/validate_generic_datasets.mjs`,
`web/datasets/manifest.json`, and the README data-format sections.

## Data Boundary

The project consumes hardware-agnostic replay data. Device-specific log parsing
and conversion should live outside this repository. External converters should
emit the generic CSV schema below, then use the packaging script when a dataset
needs to be hosted.

The two required runtime inputs are:

- `imu.csv`: raw IMU samples in the sensor/body frame.
- `gnss.csv`: WGS84 position, NED velocity, measurement standard deviations,
  and optional heading.

Optional reference CSVs are for visualization, evaluation, and manual mount
seeding. They are not filter inputs during normal automatic replay.

## Generic Replay Directory

A local generic replay directory for the native visualizer contains plain CSV
files:

```text
imu.csv
gnss.csv
reference_position.csv   # optional
reference_attitude.csv   # optional
reference_mount.csv      # optional
reference_motion.csv     # optional
```

The native `--generic-replay-dir` loader reads plain `.csv` files. Hosted
datasets use `.csv.gz` files. The browser can fetch gzip files from the hosted
manifest and can also parse dropped plain CSV text.

### Time Semantics

All `t_s` values are seconds on a common dataset timeline. The timeline may be
relative to any convenient origin; it does not need to be Unix time. Producers
should emit each CSV in nondecreasing timestamp order. Hosted validation rejects
timestamps that move backwards.

Replay merges IMU and GNSS by timestamp. When an IMU and GNSS sample have the
same `t_s`, the IMU sample is processed first. IMU timestamps are passed to
`SensorFusion::process_imu`; the replay pipeline treats each IMU sample as the
sample interval ending at `t_s`. GNSS timestamps are the measurement time of the
position/velocity fix. Reference files use the same `t_s` timeline.

The browser CSV parser sorts rows by `t_s` after parsing, but converters should
not rely on that leniency. The native directory loader does not sort before
replay.

### Required `imu.csv`

Header:

```text
t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2
```

| Column | Unit | Meaning |
| --- | --- | --- |
| `t_s` | s | Dataset timestamp. |
| `gx_radps`, `gy_radps`, `gz_radps` | rad/s | Raw body-frame angular rate about sensor axes `x_b`, `y_b`, `z_b`. |
| `ax_mps2`, `ay_mps2`, `az_mps2` | m/s^2 | Raw body-frame specific force along sensor axes `x_b`, `y_b`, `z_b`. |

Do not pre-rotate IMU samples into the vehicle frame. The runtime filter owns
the mount rotation.

### Required `gnss.csv`

Header:

```text
t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,pos_std_n_m,pos_std_e_m,pos_std_d_m,vel_std_n_mps,vel_std_e_mps,vel_std_d_mps,heading_rad
```

| Column | Unit | Meaning |
| --- | --- | --- |
| `t_s` | s | Dataset timestamp. |
| `lat_deg`, `lon_deg` | deg | WGS84 geodetic latitude and longitude. |
| `height_m` | m | WGS84 ellipsoidal height. |
| `vn_mps`, `ve_mps`, `vd_mps` | m/s | Local NED velocity: north, east, down. |
| `pos_std_n_m`, `pos_std_e_m`, `pos_std_d_m` | m | One-sigma position standard deviation in local NED axes. |
| `vel_std_n_mps`, `vel_std_e_mps`, `vel_std_d_mps` | m/s | One-sigma velocity standard deviation in local NED axes. |
| `heading_rad` | rad | Optional vehicle heading in NED, clockwise from north toward east. Use `NaN` when unavailable. |

The canonical hosted schema includes `heading_rad`. Some browser-side parsing
paths tolerate older 13-column GNSS CSVs without this column, but new datasets
should always include it.

### Optional `reference_position.csv`

Header:

```text
t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,heading_rad
```

| Column | Unit | Meaning |
| --- | --- | --- |
| `t_s` | s | Dataset timestamp. |
| `lat_deg`, `lon_deg` | deg | WGS84 reference latitude and longitude. |
| `height_m` | m | WGS84 ellipsoidal reference height. |
| `vn_mps`, `ve_mps`, `vd_mps` | m/s | Reference local NED velocity. |
| `heading_rad` | rad | Optional reference vehicle heading in NED. Use a finite value for hosted datasets. |

This file is displayed as the fused/reference path. It does not replace
`gnss.csv`; `gnss.csv` remains the GNSS-only filter input. The browser parser
can tolerate a 7-column reference-position file without `heading_rad`, but the
packaging and hosted validation path expects the 8-column header above.

### Optional `reference_attitude.csv`

Header:

```text
t_s,roll_deg,pitch_deg,yaw_deg
```

| Column | Unit | Meaning |
| --- | --- | --- |
| `t_s` | s | Dataset timestamp. |
| `roll_deg`, `pitch_deg`, `yaw_deg` | deg | Reference vehicle attitude in the local NED convention. |

### Optional `reference_mount.csv`

Header:

```text
t_s,roll_deg,pitch_deg,yaw_deg
```

| Column | Unit | Meaning |
| --- | --- | --- |
| `t_s` | s | Dataset timestamp. |
| `roll_deg`, `pitch_deg`, `yaw_deg` | deg | Physical vehicle-to-body mount, equivalent to `q_bv`, represented as RPY for plotting and manual mount seeding. |

`reference_mount.csv` is used only when the visualizer/replay is in manual
mount mode. Automatic mode uses Align to seed the mount and lets the filters
estimate mount states.

### Optional `reference_motion.csv`

Header:

```text
t_s,wx_radps,wy_radps,wz_radps,ax_mps2,ay_mps2,az_mps2
```

| Column | Unit | Meaning |
| --- | --- | --- |
| `t_s` | s | Dataset timestamp. |
| `wx_radps`, `wy_radps`, `wz_radps` | rad/s | Reference vehicle-frame angular velocity. |
| `ax_mps2`, `ay_mps2`, `az_mps2` | m/s^2 | Reference vehicle-frame gravity-compensated linear acceleration. |

This file feeds the Motion tab comparison traces. It is intentionally generic:
an external converter may derive it from any trusted reference system.

## Generic Replay Code Paths

`sim/src/datasets/generic_replay.rs` is the hardware-agnostic directory loader
and writer used by native tools. It requires exact column counts for the
canonical local CSV files, parses `NaN` for optional heading fields, and maps
rows into public `sensor_fusion::ImuSample` and `sensor_fusion::GnssSample`
values.

`sim/src/visualizer/pipeline/generic.rs` is the shared replay pipeline used by
native and browser visualization. It accepts the generic replay structs, merges
IMU/GNSS samples by timestamp, applies optional synthetic/replay GNSS outages,
and feeds only the public `SensorFusion` API. Reference files are used for
plots, map overlays, summary comparisons, and manual mount seeding.

`sim/src/datasets/synthetic_replay.rs` loads older synthetic-export directories:

```text
time.csv                 # one timestamp column for IMU-rate files
gps_time.csv             # one timestamp column for GNSS-rate files
ref_gyro.csv             # 3 columns; used with --signal-source ref
ref_accel.csv            # 3 columns; used with --signal-source ref
gyro-<data-key>.csv      # 3 columns; used with --signal-source meas
accel-<data-key>.csv     # 3 columns; used with --signal-source meas
ref_gps.csv              # 6 columns: lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps
gps-<data-key>.csv       # same 6-column schema for measured GNSS
ref_pos.csv              # truth position, 3 columns
ref_vel.csv              # truth NED velocity, 3 columns
ref_att_quat.csv         # truth quaternion, 4 columns
```

The synthetic-export loader skips the first row as a header. Gyro files are
interpreted as rad/s unless the header contains `deg/s`, in which case they are
converted to rad/s.

`sim/src/datasets/seeded_full.rs` is a diagnostic fixture loader for prepared
seeded Full EKF investigations. It is not the public generic replay boundary.
It reads semicolon-delimited files and extracts these fields:

| Loader | Rows skipped | Fields used |
| --- | ---: | --- |
| `import_gyro_data` | 3 | timestamp from column 0 divided by 1000 and floored to microseconds; angular rate from columns 1-3, rad/s. |
| `import_accel_data` | 3 | timestamp from column 0 divided by 1000 and floored to microseconds; acceleration from columns 1-3, m/s^2. |
| `import_gnss_data` | 1 | timestamp from column 0 divided by 1000 and floored; latitude column 2, longitude 3, height 4, speed 5, heading deg 6, horizontal accuracy 7, vertical accuracy 8, speed accuracy 9. |
| `import_gnss_velocity_map` | 1 | timestamp column 0; NED velocity columns 1-3; NED velocity accuracy columns 4-6. |
| `import_truth_nav` | 1 | timestamp column 0; `pitch_car_deg` from column 11. |
| `import_truth_misalignment` | n/a | row whose first field is `misalignment_deg`; mount RPY from columns 1-3. |

## Synthetic Motion DSL

Motion profile files live in `sim/motion_profiles/*.scenario` and are parsed by
`sim/src/synthetic/motion_dsl.rs`.

Comments start with `#` or `//`. Blank lines are ignored. A scenario must have
one `initial` or `init` line and at least one command. A `repeat N { ... }`
block expands its inner commands `N` times.

Initial-state syntax:

```text
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
```

Initial fields:

| Field aliases | Unit | Default | Meaning |
| --- | --- | ---: | --- |
| `lat`, `latitude` | deg | `32.0` | Initial WGS84 latitude. |
| `lon`, `longitude` | deg | `120.0` | Initial WGS84 longitude. |
| `alt`, `height`, `h` | m | `0.0` | Initial ellipsoidal height. |
| `vx`, `forward_speed`, `speed` | m/s | `0.0` | Initial body/vehicle forward velocity. |
| `vy`, `lateral_speed` | m/s | `0.0` | Initial body/vehicle lateral velocity. |
| `vz`, `vertical_speed` | m/s | `0.0` | Initial body/vehicle vertical velocity. |
| `yaw`, `pitch`, `roll` | deg | `0.0` | Initial ZYX attitude angles. |

Command duration can be written as `for 8s`, `for=8`, `duration=8`, or
`dur=8`. Numeric parsing accepts unit suffixes by reading the numeric prefix, so
`10dps`, `1.0m/s^2`, and `8s` are accepted as `10`, `1.0`, and `8`.

High-level commands:

| Command | Meaning with default `type=1` |
| --- | --- |
| `wait`, `hold`, `coast` | Zero angular-rate and acceleration command for the duration. |
| `accelerate <a>` or `accel <a>` | Forward acceleration command, m/s^2. |
| `brake <a>` or `decelerate <a>` | Negative forward acceleration command, m/s^2. |
| `turn left <r>` | Positive yaw-rate command, deg/s. |
| `turn right <r>` | Negative yaw-rate command, deg/s. |
| `yaw <r>`, `pitch <r>`, `roll <r>` | Single-axis angular-rate command, deg/s. |
| `drive yaw=<r> pitch=<r> roll=<r> ax=<a> ay=<a> az=<a>` | Combined angular-rate and body-acceleration command. |

`gps=off`, `gnss=off`, `no_gps`, and `no_gnss` are parsed into each command's
`gps_visible` flag. Current path generation stores that flag but does not use
it to remove GNSS samples. Use replay GNSS-outage options when the current
visualizer needs generated outage intervals.

Legacy command syntax can set `type` or `command_type`:

```text
command type=3 yaw=0 pitch=3 roll=0 ax=0 ay=0 az=0 for=10s gps=on
```

Command types are interpreted by `parse_motion_command` and the path generator:

| Type | Attitude command | Body-motion command |
| ---: | --- | --- |
| `1` | Angular-rate command in deg/s. | Body acceleration command in m/s^2. |
| `2` | Absolute target yaw/pitch/roll in deg. | Absolute target body velocity components, using the `ax/ay/az` fields as legacy names. |
| `3` | Relative target yaw/pitch/roll offset in deg. | Relative target body velocity offset. |
| `4` | Absolute target yaw/pitch/roll in deg. | Relative target body velocity offset. |
| `5` | Relative target yaw/pitch/roll offset in deg. | Absolute target body velocity components. |

For command types other than `1`, the generator filters target attitude and
body velocity, clamps acceleration to `max_accel_mps2`, and clamps angular
acceleration/rate to the configured limits.

Current checked-in profiles include city blocks, figure-eight variants, roll
excitation, grade/stop city driving, a reconstructed early bad-basin stress
case, high-speed straight driving, sloped-start recovery, and straight
accel/brake variants.

## Synthetic Generation

The Rust-native generator lives in `sim/src/synthetic/gnss_ins_path.rs`.
Defaults:

| Setting | Default |
| --- | ---: |
| IMU rate | `100.0 Hz` |
| GNSS rate | `2.0 Hz` |
| Simulation oversampling ratio | `1` |
| Max body acceleration clamp | `10.0 m/s^2` |
| Max angular acceleration clamp | `0.5 rad/s^2` |
| Max angular rate clamp | `1.0 rad/s` |

Generation steps:

1. Parse a `MotionProfile` from a `.scenario` file or inline DSL text.
2. Initialize WGS84 latitude, longitude, height, body velocity, and ZYX
   yaw/pitch/roll.
3. Integrate body velocity, attitude, and local geodetic position through each
   command.
4. Compute true sensor output from vehicle/body kinematics, Earth rotation,
   transport rate, WGS84 radii, and normal gravity.
5. Emit reference IMU samples and truth samples at IMU rate.
6. Emit GNSS samples at the configured GNSS period.
7. Optionally add deterministic seeded measurement noise.

Generated path structs:

| Struct | Contents |
| --- | --- |
| `GeneratedPath` | Reference `imu`, `gnss`, and `truth` vectors. |
| `GeneratedMeasurementSet` | Reference path plus noisy/measured `imu` and `gnss` vectors. |

Synthetic IMU samples are generated in the vehicle frame first. The visualizer
pipeline applies the configured truth mount `q_bv` to rotate IMU angular rate
and specific force into the raw body/sensor frame before feeding the generic
replay pipeline. Synthetic `reference_mount.csv` values are created from the
same `q_bv`.

The generator's raw IMU samples are timestamped at the start of the represented
interval. The visualizer shifts generated IMU samples and synthetic
`reference_motion.csv` samples by one IMU period before replay so the generic
pipeline sees them as interval-ending samples.

Synthetic GNSS rows use the generated WGS84 position and NED velocity. The
visualizer can add a fixed `synthetic_gnss_time_shift_ms`; shifted samples with
negative or non-finite time are dropped. It can also add an early velocity bias
vector in NED during an optional inclusive fault window.

Synthetic reference motion is derived from the noiseless reference path. Angular
velocity is vehicle-frame gyro. Acceleration is vehicle-frame linear
acceleration with gravity added back to the generated specific force, so it is
gravity-compensated.

## Synthetic Noise Modes

Visualizer noise modes map to measurement-noise presets:

| Mode | Measurement source |
| --- | --- |
| `truth` | No IMU noise and no GNSS position/velocity noise. GNSS standard-deviation columns still use the fallback display/input values `[0.5, 0.5, 0.5] m` and `[0.2, 0.2, 0.2] m/s`. |
| `low` | `ImuAccuracy::High`: low measurement noise. |
| `mid` | `ImuAccuracy::Mid`: medium measurement noise. |
| `high` | `ImuAccuracy::Low`: high measurement noise. |

IMU noise consists of zero static bias, Gauss-Markov bias drift with 100 s
correlation time, white noise density scaled by `sqrt(imu_hz)`, and optional
vibration terms. Vibration modes exist in the Rust API as random per-axis noise
or sinusoidal per-axis amplitude with frequency, but the standard visualizer
mode presets do not add vibration.

Preset values:

| Visualizer mode | Accuracy preset | Gyro drift std | Gyro white density | Accel drift std | Accel white density | GNSS pos std N/E/D | GNSS vel std N/E/D |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `low` | `High` | `1 deg/h` | `0.05 deg/sqrt(h)` | `2.0e-4 m/s^2` | `0.015 m/s/sqrt(h)` | `[0.8, 0.8, 1.2] m` | `[0.03, 0.03, 0.05] m/s` |
| `mid` | `Mid` | `10 deg/h` | `0.3 deg/sqrt(h)` | `1.0e-3 m/s^2` | `0.05 m/s/sqrt(h)` | `[3.0, 3.0, 5.0] m` | `[0.10, 0.10, 0.15] m/s` |
| `high` | `Low` | `30 deg/h` | `1.0 deg/sqrt(h)` | `5.0e-3 m/s^2` | `0.12 m/s/sqrt(h)` | `[8.0, 8.0, 12.0] m` | `[0.30, 0.30, 0.50] m/s` |

The seeded RNG is `StdRng::seed_from_u64(seed)`, so a given profile, config,
mode, and seed are repeatable.

Common native visualizer invocation:

```bash
cargo run --release -p sim --bin visualizer -- \
  --synthetic-motion-def sim/motion_profiles/city_blocks_15min.scenario \
  --synthetic-noise low
```

## Synthetic-Export Conversion

`export_synthetic_replay_generic` converts a synthetic-export directory into
the generic replay schema:

```bash
cargo run --release -p sim --bin export_synthetic_replay_generic -- \
  /path/to/synthetic-export/output /tmp/replay \
  --signal-source meas \
  --data-key 0 \
  --mount-roll-deg 0 \
  --mount-pitch-deg 0 \
  --mount-yaw-deg 0 \
  --gnss-pos-std-m 0.5 \
  --gnss-vel-std-mps 0.2
```

`--signal-source meas` reads `gyro-<data-key>.csv`, `accel-<data-key>.csv`,
and `gps-<data-key>.csv`. `--signal-source ref` reads `ref_gyro.csv`,
`ref_accel.csv`, and `ref_gps.csv`.

The converter rotates synthetic vehicle-frame IMU signals into the raw body
frame using the supplied mount RPY, writes `imu.csv` and `gnss.csv`, and fills
GNSS standard-deviation columns with the supplied scalar position and velocity
standard deviations on all axes. It does not write optional reference CSVs.

## Dataset Packaging

`scripts/package_dataset.py` stages a generic replay directory or a
synthetic-export directory into a static-hosted dataset package:

```bash
python3 scripts/package_dataset.py /path/to/replay-dir /tmp/hosted-drive
```

For synthetic-export input:

```bash
python3 scripts/package_dataset.py /path/to/synthetic-export/output /tmp/hosted-drive \
  --source-format synthetic-export \
  --signal-source meas
```

Input detection:

| `--source-format` | Behavior |
| --- | --- |
| `auto` | Prefer generic replay when `imu.csv`/`gnss.csv` or `.csv.gz` variants exist; otherwise detect synthetic-export from `time.csv` and `gps_time.csv`. |
| `generic` | Read existing generic replay CSVs. |
| `synthetic-export` | Run `export_synthetic_replay_generic` first. |

Raw `.bin` logs and arbitrary single files are rejected. Convert device logs
outside this repo first.

Package output:

```text
manifest.json
imu.csv.gz
gnss.csv.gz
reference_position.csv.gz   # optional
reference_attitude.csv.gz   # optional
reference_mount.csv.gz      # optional
reference_motion.csv.gz     # optional
```

`--keep-csv` also stages uncompressed CSVs. `--force` removes only known package
output names before rewriting a non-empty output directory.

The script validates exact headers, numeric values, nonempty files, and row
widths before writing deterministic gzip files (`mtime=0`, empty gzip filename,
compression level 9). Each file manifest records:

```text
path
media_type = "text/csv"
encoding = "gzip"
bytes
sha256
rows
first_t_s
last_t_s
columns
```

The dataset `manifest.json` records:

```text
manifest_version = 1
dataset_id
title
schema = "generic-replay-v1"
created_utc
source.format
source.input_path
files
samples
time_span_s.start
time_span_s.end
```

`dataset_id` defaults to the input directory name and can be overridden with
`--dataset-id`. `--title` sets the human-readable title.

The packaging script creates a per-dataset package. It does not automatically
update the browser index at `web/datasets/manifest.json` or the CI hosted-data
manifest at `.github/datasets/generic-datasets.json`.

## Hosted Data

Hosted datasets under `web/datasets/<dataset-id>/` are static packages using
the gzip layout above. The browser index at `web/datasets/manifest.json`
currently lists 17 hardware-agnostic datasets. Each entry has:

```json
{
  "id": "urban-short-turn-loop-nominal-001",
  "label": "Urban short turn loop 001",
  "description": "Short urban drive with repeated low-speed turns and nominal GNSS coverage.",
  "base_url": "urban-short-turn-loop-nominal-001",
  "imu_gz": "imu.csv.gz",
  "gnss_gz": "gnss.csv.gz",
  "reference_attitude_gz": "reference_attitude.csv.gz",
  "reference_mount_gz": "reference_mount.csv.gz",
  "reference_position_gz": "reference_position.csv.gz",
  "reference_motion_gz": "reference_motion.csv.gz"
}
```

Browser manifest notes:

- `base_url` may also be written as `baseUrl`.
- Plain CSV keys are `imu` and `gnss`; aliases `imu_csv` and `gnss_csv` are
  accepted.
- Gzip keys are `imu_gz` and `gnss_gz`; aliases `imu_csv_gz` and
  `gnss_csv_gz` are accepted.
- Optional references support plain and gzip keys for attitude, mount,
  position, and motion.
- Required IMU/GNSS files fall back to `imu.csv.gz`, `gnss.csv.gz`, then plain
  `imu.csv`, `gnss.csv` under `base_url` when explicit keys are omitted.
- Optional reference files are fetched only when explicitly listed.
- Dataset URLs must be relative, root-relative, or HTTP(S). Repository static
  validation rejects unsafe local paths and path traversal.

The current hosted IDs are:

```text
urban-short-turn-loop-nominal-001
urban-short-turn-loop-nominal-002
urban-medium-turn-loop-nominal-001
urban-medium-turn-loop-nominal-002
urban-long-turn-loop-nominal-001
parking-lot-figure8-nominal-001
parking-lot-circle-turns-nominal-001
urban-mixed-turns-nominal-001
urban-stop-go-tight-turns-nominal-001
mixed-road-long-drive-nominal-001
mixed-road-long-drive-large-mount-001
mixed-road-long-drive-large-mount-002
urban-stop-go-large-mount-001
urban-low-speed-low-signal-001
mixed-road-highway-low-signal-001
covered-parking-urban-drive-gnss-outage-001
covered-parking-urban-drive-data-gap-001
```

## Hosted Dataset Validation

`scripts/validate_generic_datasets.mjs` validates the CI hosted-data manifest,
downloads or stages files, checks checksums, validates CSV schema, and can run
bounded replay smoke profiles.

Default command:

```bash
node scripts/validate_generic_datasets.mjs
```

Useful options:

```bash
node scripts/validate_generic_datasets.mjs \
  --manifest .github/datasets/generic-datasets.json \
  --cache-dir .cache/generic-datasets \
  --work-dir target/generic-datasets \
  --smoke-profile
```

The manifest shape is:

```json
{
  "schema_version": 1,
  "datasets": [
    {
      "id": "urban-short-turn-loop-nominal-001",
      "version": "v1",
      "description": "Urban short turn loop 001",
      "license": "MIT",
      "replay_dir": ".",
      "files": [
        {
          "path": "imu.csv.gz",
          "url": "../../web/datasets/urban-short-turn-loop-nominal-001/imu.csv.gz",
          "sha256": "64 hex characters",
          "bytes": 578953
        }
      ],
      "smoke": {
        "enabled": false,
        "max_imu_rows": 17604,
        "max_gnss_rows": 354,
        "misalignment": "auto"
      }
    }
  ]
}
```

Manifest validation checks:

- `schema_version` is `1`.
- Every dataset has a valid `id`, nonempty `version`, and unique
  `id@version`.
- `replay_dir` and file `path` values are safe relative paths.
- File `url` values are HTTP(S) URLs or safe relative paths.
- SHA-256 values are 64-character hex digests.
- Optional `bytes` values are positive integers.
- The file list contains `imu.csv` or `imu.csv.gz` and `gnss.csv` or
  `gnss.csv.gz` under `replay_dir`.
- Smoke `misalignment`, when present, is `auto` or `manual`.

File validation checks:

- Downloaded/cached file SHA-256 matches the manifest.
- Optional byte count matches the manifest.
- Plain or gzip CSV can be assembled into the work directory.
- Required and optional CSV headers match the canonical headers exactly.
- Each CSV has at least one data row.
- Every value is finite, except `gnss.csv` `heading_rad` may be `NaN`.
- Timestamps are nondecreasing within each CSV.

Smoke validation builds a subset replay directory and runs:

```bash
cargo run -p sim --bin visualizer --locked -- \
  --generic-replay-dir <smoke-dir> \
  --profile-only \
  --misalignment auto
```

`smoke.max_imu_rows` defaults to `1000` for `imu.csv`. `smoke.max_gnss_rows`
defaults to `200` for `gnss.csv`, `reference_attitude.csv`,
`reference_mount.csv`, and `reference_position.csv`. `reference_motion.csv`
uses `smoke.max_imu_rows` when present, otherwise `2000`.

The static site validator, `scripts/validate_pages_static.mjs`, separately
checks that `web/datasets/manifest.json` is valid JSON with a `datasets` array,
that listed dataset URLs are safe, and during HTTP validation that listed core
dataset files are reachable and gzip files have a gzip header.

## External Converter Checklist

External converters should:

1. Keep device-specific parsing outside this repository.
2. Emit plain generic `imu.csv` and `gnss.csv` with the exact headers above.
3. Use one common relative `t_s` timeline in seconds and nondecreasing row
   order.
4. Leave IMU samples in the raw sensor/body frame.
5. Emit GNSS position as WGS84 latitude/longitude plus ellipsoidal height.
6. Emit GNSS velocity and standard deviations in local NED axes.
7. Use `NaN` for unavailable `gnss.csv` heading.
8. Add optional reference CSVs only when they come from a trusted generic
   reference source, not from a device-specific protocol dependency inside this
   repo.
9. Express `reference_mount.csv` as the physical vehicle-to-body mount `q_bv`
   in RPY form.
10. Express `reference_motion.csv` in the vehicle frame, with
    gravity-compensated linear acceleration.
11. Run `scripts/package_dataset.py` to create deterministic `.csv.gz` package
    files and a per-dataset manifest.
12. Add hosted entries to `web/datasets/manifest.json` for browser loading and
    `.github/datasets/generic-datasets.json` for CI validation when the dataset
    should become part of hosted test data.
