# Embedded Performance

Embedded performance numbers are maintained as budget evidence for the public `SensorFusion` runtime plus `road_events`. The benchmark harness lives outside this repository in `rustcam/apps/fusion_bench`, so these numbers are reported as external evidence rather than repo-local tooling.

Latest maintained ESP32-S3 run:

| Operation | Avg time | 100 Hz budget |
| --- | ---: | ---: |
| Runtime predict only | `950 us/step` | `9.5%` CPU |
| Runtime IMU/10 Hz NHC, roll prior off | `1230 us/step` | `12.3%` CPU |
| Runtime IMU/NHC, roll prior on | `1320 us/step` | `13.2%` CPU |
| Runtime NHC every IMU | `3690 us/step` | `36.9%` CPU |
| Runtime IMU + vehicle speed | `2620 us/step` | `26.2%` CPU |
| Runtime IMU + 2 Hz GNSS stream | `1380 us/step` | `13.8%` CPU |
| Road events update | `20 us/step` | `0.2%` CPU |
| Runtime + road events | `1400 us/step` | `14.0%` CPU |

Measured runtime breakdown:

- predict-only baseline: `950 us/tick`;
- decimated 10 Hz NHC increment: `280 us/tick` averaged over a 100 Hz stream;
- decimated vehicle-roll prior increment: `90 us/tick` averaged over a 100 Hz stream;
- NHC plus roll prior every IMU tick: `2740 us/tick` over predict-only.

The benchmark timer is coarse on the referenced embedded build, with roughly `10 ms` quantization over loops. Treat these as order-of-magnitude embedded budget numbers, not fine-grained cycle measurements.

Historical type-layout measurements reported `SensorFusion` at `8.752 kB`. Current linked-region and symbol-table numbers should be remeasured when memory footprint matters for a release.
