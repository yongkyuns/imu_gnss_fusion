# Hosted Datasets

The hosted web visualizer reads its dataset list from `web/datasets/manifest.json`. The current checked-in manifest contains 32 datasets:

```text
urban-short-turn-loop-nominal-001
urban-short-turn-loop-nominal-002
urban-medium-turn-loop-nominal-001
urban-medium-turn-loop-nominal-002
parkinglot-maneuvers
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
ios-mixed-drive-reverse-parking-001
ios-stationary-smoke-001
ios-stationary-smoke-002
ios-mixed-road-drive-001
ios-mixed-road-drive-002
ios-mixed-road-loop-001
ios-evening-mixed-road-drive-001
ios-evening-mixed-road-drive-002
ios-drive-20260521-121635
ios-drive-20260521-130835
ios-drive-20260521-131420
ios-drive-20260521-164415
ios-drive-20260521-180601
ios-drive-20260526-174206
```

Dataset publishing validates manifest entries, schema, checksums, and replay
smoke behavior before a dataset is added to the hosted list. The developer
workflow is described in [](../development/datasets.md).

iOS recordings usually provide IMU/GNSS rows without reference overlays. Synthetic and curated replay datasets may include reference attitude, position, motion, and mount streams for plotting and evaluation.
