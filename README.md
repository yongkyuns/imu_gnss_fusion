# imu_gnss_fusion

Rust port of:

- `parse_pygpsdata_log.py` -> `parse_pygpsdata_log` (CLI)
- `visualize_pygpsdata_log.py` -> `visualize_pygpsdata_log` (egui app)

Tech stack:

- UBX parsing: `ublox-rs` + direct UBX frame decoding
- DataFrame/export: `polars`
- Visualization: `egui`/`eframe` + `egui_plot`

## Build

```bash
cd simulation/imu_gnss_fusion
cargo build --release
```

## Parse log

```bash
cargo run --release --bin parse_pygpsdata_log -- \
  /path/to/pygpsdata.log \
  --csv parsed.csv \
  --parquet parsed.parquet
```

## Visualize log

```bash
cargo run --release --bin visualize_pygpsdata_log -- \
  /path/to/pygpsdata.log
```

