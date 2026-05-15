#!/usr/bin/env python3
"""Export iOS .motionfusion recordings to generic web replay CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
IMU_HEADER = [
    "t_s",
    "gx_radps",
    "gy_radps",
    "gz_radps",
    "ax_mps2",
    "ay_mps2",
    "az_mps2",
]
GNSS_HEADER = [
    "t_s",
    "lat_deg",
    "lon_deg",
    "height_m",
    "vn_mps",
    "ve_mps",
    "vd_mps",
    "pos_std_n_m",
    "pos_std_e_m",
    "pos_std_d_m",
    "vel_std_n_mps",
    "vel_std_e_mps",
    "vel_std_d_mps",
    "heading_rad",
]

DEFAULT_HORIZONTAL_ACCURACY_M = 10.0
DEFAULT_VERTICAL_ACCURACY_M = 15.0
DEFAULT_SPEED_ACCURACY_MPS = 2.0
STREAM_END_WARN_S = 1.0
IMU_LARGE_GAP_MIN_S = 1.0
GNSS_LARGE_GAP_MIN_S = 5.0
STATIONARY_SPEED_THRESHOLD_MPS = 0.25


@dataclass(frozen=True)
class ImuRow:
    t_s: float
    gyro: tuple[float, float, float]
    accel: tuple[float, float, float]


@dataclass(frozen=True)
class GnssRow:
    t_s: float
    lat_deg: float
    lon_deg: float
    height_m: float
    velocity_ned_mps: tuple[float, float, float]
    pos_std_m: tuple[float, float, float]
    vel_std_mps: tuple[float, float, float]
    heading_rad: float
    had_explicit_velocity: bool
    derived_velocity: bool


@dataclass(frozen=True)
class ExportResult:
    imu_rows: list[ImuRow]
    gnss_rows: list[GnssRow]
    source_event_count: int
    barometer_count: int
    skipped_gnss_missing_velocity: int
    session_start_s: float
    session_end_s: float
    duration_s: float
    imu_csv: Path
    gnss_csv: Path
    summary_txt: Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an iOS .motionfusion JSON recording into generic web "
            "replay imu.csv and gnss.csv files."
        )
    )
    parser.add_argument("input", type=Path, help="Input .motionfusion JSON file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input-stem>-web-export next to the input file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        result = export_motionfusion(args.input, args.output_dir)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(result.summary_txt.read_text(encoding="utf-8"), end="")
    return 0


def export_motionfusion(input_path: Path, output_dir: Path | None = None) -> ExportResult:
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise ValueError(f"input file does not exist: {input_path}")

    if output_dir is None:
        output_dir = input_path.with_name(f"{input_path.stem}-web-export")
    output_dir = output_dir.expanduser().resolve()

    log = read_motionfusion_json(input_path)
    (
        imu_rows,
        gnss_rows,
        barometer_count,
        skipped_gnss_missing_velocity,
        session_start_s,
        session_end_s,
    ) = extract_rows(log)
    duration_s = max(0.0, session_end_s - session_start_s)
    if not imu_rows:
        raise ValueError("recording contains no IMU samples")
    if not gnss_rows:
        raise ValueError("recording contains no GNSS samples")

    output_dir.mkdir(parents=True, exist_ok=True)
    imu_csv = output_dir / "imu.csv"
    gnss_csv = output_dir / "gnss.csv"
    summary_txt = output_dir / "summary.txt"
    write_imu_csv(imu_csv, imu_rows)
    write_gnss_csv(gnss_csv, gnss_rows)
    summary = format_summary(
        input_path=input_path,
        output_dir=output_dir,
        source_event_count=len(log.get("events", [])),
        barometer_count=barometer_count,
        skipped_gnss_missing_velocity=skipped_gnss_missing_velocity,
        session_start_s=session_start_s,
        session_end_s=session_end_s,
        duration_s=duration_s,
        imu_rows=imu_rows,
        gnss_rows=gnss_rows,
    )
    summary_txt.write_text(summary, encoding="utf-8")

    return ExportResult(
        imu_rows=imu_rows,
        gnss_rows=gnss_rows,
        source_event_count=len(log.get("events", [])),
        barometer_count=barometer_count,
        skipped_gnss_missing_velocity=skipped_gnss_missing_velocity,
        session_start_s=session_start_s,
        session_end_s=session_end_s,
        duration_s=duration_s,
        imu_csv=imu_csv,
        gnss_csv=gnss_csv,
        summary_txt=summary_txt,
    )


def read_motionfusion_json(input_path: Path) -> dict[str, Any]:
    try:
        log = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(log, dict):
        raise ValueError("top-level JSON value must be an object")
    schema_version = log.get("schemaVersion")
    if schema_version is not None and schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported top-level schemaVersion: {schema_version}")
    events = log.get("events")
    if not isinstance(events, list):
        raise ValueError("recording must contain an events array")
    return log


def extract_rows(
    log: dict[str, Any],
) -> tuple[list[ImuRow], list[GnssRow], int, int, float, float]:
    events = sorted(log["events"], key=event_sort_key)
    imu_rows: list[ImuRow] = []
    gnss_rows: list[GnssRow] = []
    barometer_count = 0
    skipped_gnss_missing_velocity = 0
    all_times: list[float] = []

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"events[{index}] must be an object")
        validate_event_envelope(event, index)
        kind = event["kind"]
        t_s = required_float(event, "elapsedSec", f"events[{index}]")
        all_times.append(t_s)

        if kind == "imu":
            payload = required_object(event, "imu", f"events[{index}]")
            imu_rows.append(parse_imu_row(t_s, payload, f"events[{index}].imu"))
        elif kind == "gnss":
            payload = required_object(event, "gnss", f"events[{index}]")
            row = parse_gnss_row(t_s, payload, f"events[{index}].gnss")
            if row is None:
                skipped_gnss_missing_velocity += 1
            else:
                gnss_rows.append(row)
        elif kind == "barometer":
            barometer_count += 1

    session_start_s = min(all_times) if all_times else 0.0
    session_end_s = max(all_times) if all_times else 0.0
    return (
        imu_rows,
        gnss_rows,
        barometer_count,
        skipped_gnss_missing_velocity,
        session_start_s,
        session_end_s,
    )


def event_sort_key(event: Any) -> tuple[float, int]:
    if not isinstance(event, dict):
        return (math.inf, 99)
    elapsed = event.get("elapsedSec")
    t_s = elapsed if isinstance(elapsed, (int, float)) and math.isfinite(elapsed) else math.inf
    priority = {"barometer": 0, "imu": 1, "gnss": 2}.get(event.get("kind"), 99)
    return (float(t_s), priority)


def validate_event_envelope(event: dict[str, Any], index: int) -> None:
    label = f"events[{index}]"
    schema_version = event.get("schemaVersion")
    if schema_version is not None and schema_version != SCHEMA_VERSION:
        raise ValueError(f"{label} has unsupported schemaVersion: {schema_version}")
    kind = event.get("kind")
    if kind not in {"imu", "gnss", "barometer"}:
        raise ValueError(f"{label}.kind must be imu, gnss, or barometer")
    elapsed = required_float(event, "elapsedSec", label)
    if elapsed < 0.0:
        raise ValueError(f"{label}.elapsedSec must be non-negative")
    if kind in {"imu", "gnss"} and event.get(kind) is None:
        raise ValueError(f"{label} is missing {kind} payload")


def parse_imu_row(t_s: float, payload: dict[str, Any], label: str) -> ImuRow:
    return ImuRow(
        t_s=t_s,
        gyro=(
            required_float(payload, "gyroXRadps", label),
            required_float(payload, "gyroYRadps", label),
            required_float(payload, "gyroZRadps", label),
        ),
        accel=(
            required_float(payload, "accelXMps2", label),
            required_float(payload, "accelYMps2", label),
            required_float(payload, "accelZMps2", label),
        ),
    )


def parse_gnss_row(t_s: float, payload: dict[str, Any], label: str) -> GnssRow | None:
    horizontal_accuracy = positive_optional_float(payload, "horizontalAccuracyM", label)
    vertical_accuracy = positive_optional_float(payload, "verticalAccuracyM", label)
    speed_accuracy = positive_optional_float(payload, "speedAccuracyMps", label)
    pos_std_n = horizontal_accuracy or DEFAULT_HORIZONTAL_ACCURACY_M
    pos_std_e = horizontal_accuracy or DEFAULT_HORIZONTAL_ACCURACY_M
    pos_std_d = vertical_accuracy or DEFAULT_VERTICAL_ACCURACY_M

    explicit_velocity = optional_vector(
        payload,
        ("velocityNorthMps", "velocityEastMps", "velocityDownMps"),
        label,
    )
    derived_velocity = False
    if explicit_velocity is not None:
        velocity = explicit_velocity
        vel_std = (speed_accuracy or DEFAULT_SPEED_ACCURACY_MPS,) * 3
    else:
        course_deg = optional_float(payload, "courseDeg", label)
        speed_mps = optional_float(payload, "speedMps", label)
        if speed_mps is not None and 0.0 <= speed_mps <= STATIONARY_SPEED_THRESHOLD_MPS:
            velocity = (0.0, 0.0, 0.0)
            vel_std = (speed_accuracy or DEFAULT_SPEED_ACCURACY_MPS,) * 3
            derived_velocity = True
        elif course_deg is not None and speed_mps is not None and speed_mps >= 0.0:
            heading_rad = math.radians(course_deg)
            velocity = (
                speed_mps * math.cos(heading_rad),
                speed_mps * math.sin(heading_rad),
                0.0,
            )
            vel_std = (speed_accuracy or DEFAULT_SPEED_ACCURACY_MPS,) * 3
            derived_velocity = True
        else:
            return None

    heading_rad = math.nan
    course_deg = optional_float(payload, "courseDeg", label)
    if course_deg is not None:
        heading_rad = math.radians(course_deg)

    return GnssRow(
        t_s=t_s,
        lat_deg=required_float(payload, "latitudeDeg", label),
        lon_deg=required_float(payload, "longitudeDeg", label),
        height_m=required_float(payload, "altitudeM", label),
        velocity_ned_mps=velocity,
        pos_std_m=(pos_std_n, pos_std_e, pos_std_d),
        vel_std_mps=vel_std,
        heading_rad=heading_rad,
        had_explicit_velocity=explicit_velocity is not None,
        derived_velocity=derived_velocity,
    )


def write_imu_csv(path: Path, rows: Iterable[ImuRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(IMU_HEADER)
        for row in rows:
            writer.writerow(
                [
                    csv_number(row.t_s),
                    csv_number(row.gyro[0]),
                    csv_number(row.gyro[1]),
                    csv_number(row.gyro[2]),
                    csv_number(row.accel[0]),
                    csv_number(row.accel[1]),
                    csv_number(row.accel[2]),
                ]
            )


def write_gnss_csv(path: Path, rows: Iterable[GnssRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(GNSS_HEADER)
        for row in rows:
            writer.writerow(
                [
                    csv_number(row.t_s),
                    csv_number(row.lat_deg),
                    csv_number(row.lon_deg),
                    csv_number(row.height_m),
                    csv_number(row.velocity_ned_mps[0]),
                    csv_number(row.velocity_ned_mps[1]),
                    csv_number(row.velocity_ned_mps[2]),
                    csv_number(row.pos_std_m[0]),
                    csv_number(row.pos_std_m[1]),
                    csv_number(row.pos_std_m[2]),
                    csv_number(row.vel_std_mps[0]),
                    csv_number(row.vel_std_mps[1]),
                    csv_number(row.vel_std_mps[2]),
                    "NaN" if math.isnan(row.heading_rad) else csv_number(row.heading_rad),
                ]
            )


def format_summary(
    *,
    input_path: Path,
    output_dir: Path,
    source_event_count: int,
    barometer_count: int,
    skipped_gnss_missing_velocity: int,
    session_start_s: float,
    session_end_s: float,
    duration_s: float,
    imu_rows: list[ImuRow],
    gnss_rows: list[GnssRow],
) -> str:
    missing_velocity_rows = sum(1 for row in gnss_rows if not row.had_explicit_velocity)
    derived_velocity_rows = sum(1 for row in gnss_rows if row.derived_velocity)
    accel_stats = magnitude_stats(row.accel for row in imu_rows)
    gyro_stats = magnitude_stats(row.gyro for row in imu_rows)
    imu_times = [row.t_s for row in imu_rows]
    gnss_times = [row.t_s for row in gnss_rows]
    imu_window = stream_window(imu_times)
    gnss_window = stream_window(gnss_times)
    imu_gap_stats = gap_stats(imu_times, IMU_LARGE_GAP_MIN_S)
    gnss_gap_stats = gap_stats(gnss_times, GNSS_LARGE_GAP_MIN_S)
    imu_to_gnss_end_s = max(0.0, gnss_window.end_s - imu_window.end_s)
    imu_to_session_end_s = max(0.0, session_end_s - imu_window.end_s)
    health_flags = stream_health_flags(
        imu_to_gnss_end_s=imu_to_gnss_end_s,
        imu_to_session_end_s=imu_to_session_end_s,
        imu_gap_stats=imu_gap_stats,
        gnss_gap_stats=gnss_gap_stats,
        skipped_gnss_missing_velocity=skipped_gnss_missing_velocity,
    )
    health_status = "OK" if not health_flags else "WARN"

    lines = [
        "MotionFusion export summary",
        f"input: {input_path}",
        f"output_dir: {output_dir}",
        f"source_events: {source_event_count}",
        (
            "counts: "
            f"imu={len(imu_rows)} gnss={len(gnss_rows)} barometer={barometer_count}"
        ),
        f"duration_s: {duration_s:.3f}",
        f"session_window_s: start={session_start_s:.3f} end={session_end_s:.3f}",
        format_stream_window("imu_active", imu_window),
        format_stream_window("gnss_active", gnss_window),
        f"imu_rate_hz: {imu_window.rate_hz:.3f}",
        f"gnss_rate_hz: {gnss_window.rate_hz:.3f}",
        (
            "stream_end_lag_s: "
            f"imu_to_gnss_end={imu_to_gnss_end_s:.3f}"
            f"{warn_suffix(imu_to_gnss_end_s > STREAM_END_WARN_S)}, "
            f"imu_to_session_end={imu_to_session_end_s:.3f}"
            f"{warn_suffix(imu_to_session_end_s > STREAM_END_WARN_S)}"
        ),
        format_gap_summary("imu_large_gaps", imu_gap_stats),
        format_gap_summary("gnss_large_gaps", gnss_gap_stats),
        (
            "missing_gnss_velocity_rows: "
            f"{missing_velocity_rows} "
            f"(derived_velocity={derived_velocity_rows}, "
            f"skipped={skipped_gnss_missing_velocity}"
            f"{warn_suffix(skipped_gnss_missing_velocity > 0)})"
        ),
        (
            "gnss_velocity_rows: "
            f"explicit={len(gnss_rows) - missing_velocity_rows} "
            f"derived_velocity={derived_velocity_rows} "
            f"skipped_missing_velocity={skipped_gnss_missing_velocity}"
            f"{warn_suffix(skipped_gnss_missing_velocity > 0)}"
        ),
        f"stream_health: {health_status}{(': ' + ', '.join(health_flags)) if health_flags else ''}",
        (
            "accel_magnitude_mps2: "
            f"min={accel_stats[0]:.6g} mean={accel_stats[1]:.6g} "
            f"max={accel_stats[2]:.6g} std={accel_stats[3]:.6g}"
        ),
        (
            "gyro_magnitude_radps: "
            f"min={gyro_stats[0]:.6g} mean={gyro_stats[1]:.6g} "
            f"max={gyro_stats[2]:.6g} std={gyro_stats[3]:.6g}"
        ),
        "files: imu.csv gnss.csv summary.txt",
    ]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class StreamWindow:
    start_s: float
    end_s: float
    duration_s: float
    rate_hz: float


@dataclass(frozen=True)
class GapStats:
    threshold_s: float
    count: int
    max_s: float


def stream_window(times: list[float]) -> StreamWindow:
    if not times:
        return StreamWindow(start_s=math.nan, end_s=math.nan, duration_s=0.0, rate_hz=0.0)
    start_s = min(times)
    end_s = max(times)
    duration_s = max(0.0, end_s - start_s)
    return StreamWindow(
        start_s=start_s,
        end_s=end_s,
        duration_s=duration_s,
        rate_hz=sample_rate_hz(times),
    )


def sample_rate_hz(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    span_s = max(times) - min(times)
    if span_s <= 0.0:
        return 0.0
    return (len(times) - 1) / span_s


def gap_stats(times: list[float], minimum_threshold_s: float) -> GapStats:
    if len(times) < 2:
        return GapStats(threshold_s=minimum_threshold_s, count=0, max_s=0.0)
    ordered = sorted(times)
    gaps = [right - left for left, right in zip(ordered, ordered[1:])]
    positive_gaps = [gap for gap in gaps if gap > 0.0]
    if not positive_gaps:
        return GapStats(threshold_s=minimum_threshold_s, count=0, max_s=0.0)
    median_gap_s = statistics.median(positive_gaps)
    threshold_s = max(minimum_threshold_s, median_gap_s * 5.0)
    large_gaps = [gap for gap in positive_gaps if gap > threshold_s]
    return GapStats(
        threshold_s=threshold_s,
        count=len(large_gaps),
        max_s=max(positive_gaps),
    )


def format_stream_window(label: str, window: StreamWindow) -> str:
    return (
        f"{label}: "
        f"start_s={window.start_s:.3f} "
        f"end_s={window.end_s:.3f} "
        f"duration_s={window.duration_s:.3f} "
        f"rate_hz={window.rate_hz:.3f}"
    )


def format_gap_summary(label: str, stats: GapStats) -> str:
    return (
        f"{label}: "
        f"count={stats.count}{warn_suffix(stats.count > 0)} "
        f"max_s={stats.max_s:.3f} "
        f"threshold_s={stats.threshold_s:.3f}"
    )


def stream_health_flags(
    *,
    imu_to_gnss_end_s: float,
    imu_to_session_end_s: float,
    imu_gap_stats: GapStats,
    gnss_gap_stats: GapStats,
    skipped_gnss_missing_velocity: int,
) -> list[str]:
    flags: list[str] = []
    if imu_to_gnss_end_s > STREAM_END_WARN_S:
        flags.append(f"imu_ended_before_gnss_end_by_{imu_to_gnss_end_s:.3f}s")
    if imu_to_session_end_s > STREAM_END_WARN_S:
        flags.append(f"imu_ended_before_session_end_by_{imu_to_session_end_s:.3f}s")
    if imu_gap_stats.count > 0:
        flags.append(f"imu_large_gaps_{imu_gap_stats.count}")
    if gnss_gap_stats.count > 0:
        flags.append(f"gnss_large_gaps_{gnss_gap_stats.count}")
    if skipped_gnss_missing_velocity > 0:
        flags.append(f"skipped_gnss_rows_{skipped_gnss_missing_velocity}")
    return flags


def warn_suffix(condition: bool) -> str:
    return " WARN" if condition else ""


def magnitude_stats(vectors: Iterable[tuple[float, float, float]]) -> tuple[float, float, float, float]:
    magnitudes = [math.sqrt(x * x + y * y + z * z) for x, y, z in vectors]
    if not magnitudes:
        return (math.nan, math.nan, math.nan, math.nan)
    std = statistics.pstdev(magnitudes) if len(magnitudes) > 1 else 0.0
    return (min(magnitudes), statistics.fmean(magnitudes), max(magnitudes), std)


def required_object(parent: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label}.{key} must be an object")
    return value


def required_float(parent: dict[str, Any], key: str, label: str) -> float:
    value = optional_float(parent, key, label)
    if value is None:
        raise ValueError(f"{label}.{key} must be a finite number")
    return value


def optional_float(parent: dict[str, Any], key: str, label: str) -> float | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}.{key} must be a finite number when present")
    return float(value)


def positive_optional_float(parent: dict[str, Any], key: str, label: str) -> float | None:
    value = optional_float(parent, key, label)
    if value is None or value <= 0.0:
        return None
    return value


def optional_vector(
    parent: dict[str, Any],
    keys: tuple[str, str, str],
    label: str,
) -> tuple[float, float, float] | None:
    parsed: list[float | None] = []
    for key in keys:
        parsed.append(optional_float(parent, key, label))
    if all(value is None for value in parsed):
        return None
    if any(value is None for value in parsed):
        return None
    return (parsed[0], parsed[1], parsed[2])


def csv_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"CSV value must be finite: {value}")
    return format(value, ".17g")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
