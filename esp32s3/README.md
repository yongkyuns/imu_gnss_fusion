# ESP32-S3 Sensor Fusion Demo

This directory contains the ESP-IDF application for running the standalone C
`sensor_fusion` library on a Freenove ESP32-S3-WROOM board.

The app receives replayed IMU and GNSS samples over the ESP32-S3 native USB
port, runs the shared `sensor_fusion` library on-device, and sends compact
status frames back to the host for parity and timing checks.

## What it does

- uses the repo's standalone C implementation in [`../ekf/c`](../ekf/c)
- receives compact IMU and GNSS samples over USB
- runs either:
  - internal align + ESKF, or
  - external misalignment + ESKF
- returns compact status snapshots back to the host

The host-side replay tool is:

- [`sim/src/bin/esp32_usb_replay.rs`](/Users/ykshin/Dev/me/ekf/simulation/imu_gnss_fusion/sim/src/bin/esp32_usb_replay.rs)

This app builds directly against the shared C sources in `../ekf/c`. The
filter implementation is not copied into this directory.

## Feasibility

Yes, this is practical on an ESP32-S3:

- replay bandwidth is small
  - 100 Hz IMU with 28-byte payload is only a few kB/s
  - 2 Hz GNSS adds very little on top
- `sensor_fusion` is already:
  - single precision
  - no heap
  - no OS dependency
- the ESP32-S3 native USB peripheral can expose a CDC device over the OTG port

## Important hardware note

Use the ESP32-S3 native USB/OTG port, not the USB-UART bridge.

On Freenove boards that expose both:

- `USB-OTG` / native USB: use this for the replay protocol
- `UART` USB bridge: optional for normal serial logs, but not required here

## Build

From this directory:

```bash
source "$IDF_PATH/export.sh"
idf.py set-target esp32s3
idf.py build
```

Flash and monitor:

```bash
idf.py -p /dev/cu.usbmodemXXXX flash monitor
```

## Known-Good Workflow

Use this path for repeatable ESP32 flash and timing runs without extra
troubleshooting.

Assumptions:

- use the native USB/OTG port, not the UART bridge
- keep the checked-in [`sdkconfig`](./sdkconfig)
  - it currently uses performance optimization
  - it currently uses a larger main task stack required by the ESKF path

Build and flash:

```bash
cd esp32s3
source "$IDF_PATH/export.sh"
idf.py build flash -p /dev/cu.usbmodemXXXX
```

For ESKF timing measurement, use external misalignment mode to bypass the
internal-align path and force the actual ESKF replay path:

```bash
cd ..
cargo run -q -p sim --bin esp32_usb_replay -- \
  --port /dev/cu.usbmodemXXXX \
  --baud 921600 \
  --serial-timeout-ms 500 \
  --settle-ms 1500 \
  --max-time-s 40 \
  --summary-only \
  --replay-speedup 1.0 \
  --external-q-vb 1,0,0,0 \
  --input logger/data/ubx_raw_20260312_170548.bin
```

Notes:

- `--serial-timeout-ms 500` avoids false host-side timeouts under replay load
- `--settle-ms 1500` gives the board time to reboot after flashing/reset
- `--external-q-vb 1,0,0,0` is the current known-good timing path
- `--summary-only` is the fastest way to get stage timing numbers
- if the device path changes, list it with `ls /dev/cu.usbmodem*`

## Host replay

Example:

```bash
cargo run -p sim --bin esp32_usb_replay -- \
  --port /dev/cu.usbmodemXXXX \
  --baud 921600 \
  --input logger/data/ubx_raw_20260328_153757.bin
```

External misalignment mode:

```bash
cargo run -p sim --bin esp32_usb_replay -- \
  --port /dev/cu.usbmodemXXXX \
  --input logger/data/ubx_raw_20260328_153757.bin \
  --external-q-vb 1,0,0,0
```

## USB protocol

All fields are little-endian.

Frame header:

- `u16 magic = 0x5346`
- `u8 version = 1`
- `u8 msg_type`
- `u16 payload_len`
- `u16 reserved`

Host to device:

- `0x01 CONFIG`
- `0x02 RESET`
- `0x10 IMU`
- `0x11 GNSS`
- `0x12 END`

Device to host:

- `0x81 STATUS`

See:

- [`main/sf_usb_protocol.h`](./main/sf_usb_protocol.h)

## Limitations

- this is a replay transport, not a live GNSS receiver stack
- transport is intentionally simple and does not include retransmission or CRC yet
- the exact TinyUSB Kconfig symbols can vary a bit across ESP-IDF versions; this scaffold targets modern ESP-IDF 5.x
- status messages are intended for validation/debug, not a final production telemetry protocol
