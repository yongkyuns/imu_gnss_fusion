#!/usr/bin/env python3
"""Package generic replay data for static hosting.

The package format is intentionally hardware agnostic:

  manifest.json
  imu.csv.gz
  gnss.csv.gz
  reference_attitude.csv.gz (optional)
  reference_mount.csv.gz (optional)

Input can be an existing generic replay directory containing imu.csv/gnss.csv
or a gnss-ins-sim output directory that can be converted by the existing
export_gnss_ins_sim_generic Rust tool.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

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
REFERENCE_RPY_HEADER = ["t_s", "roll_deg", "pitch_deg", "yaw_deg"]


@dataclass(frozen=True)
class CsvStats:
    rows: int
    first_t_s: float | None
    last_t_s: float | None


@dataclass(frozen=True)
class FileStats:
    path: str
    bytes: int
    sha256: str


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package generic replay data into manifest.json plus imu.csv.gz/gnss.csv.gz.",
    )
    parser.add_argument("input", type=Path, help="Generic replay dir or gnss-ins-sim output dir")
    parser.add_argument("output", type=Path, help="Output hosted-data dataset directory")
    parser.add_argument(
        "--source-format",
        choices=["auto", "generic", "gnss-ins-sim"],
        default="auto",
        help="Input format. auto detects generic replay first, then gnss-ins-sim.",
    )
    parser.add_argument(
        "--dataset-id",
        help="Dataset id stored in manifest.json. Defaults to the input directory name.",
    )
    parser.add_argument("--title", help="Optional human-readable dataset title")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--keep-csv",
        action="store_true",
        help="Also stage uncompressed imu.csv and gnss.csv next to the gzip files.",
    )

    conversion = parser.add_argument_group("gnss-ins-sim conversion options")
    conversion.add_argument(
        "--signal-source",
        choices=["meas", "ref"],
        default="meas",
        help="Forwarded to export_gnss_ins_sim_generic.",
    )
    conversion.add_argument("--data-key", type=int, default=0)
    conversion.add_argument("--mount-roll-deg", type=float, default=0.0)
    conversion.add_argument("--mount-pitch-deg", type=float, default=0.0)
    conversion.add_argument("--mount-yaw-deg", type=float, default=0.0)
    conversion.add_argument("--gnss-pos-std-m", type=float, default=0.5)
    conversion.add_argument("--gnss-vel-std-mps", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        package_dataset(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def package_dataset(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if input_path.suffix.lower() == ".bin":
        raise ValueError(
            "raw binary logs are not supported here; convert them outside this repo "
            "to generic imu.csv/gnss.csv first"
        )
    if input_path.is_file():
        raise ValueError(f"input must be a replay directory, not a file: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"input directory does not exist: {input_path}")

    ensure_output_dir(output_dir, args.force)
    source_format = detect_source_format(input_path, args.source_format)
    dataset_id = args.dataset_id or input_path.name

    with tempfile.TemporaryDirectory(prefix="imu_gnss_package_") as tmp:
        generic_dir = Path(tmp) / "generic"
        if source_format == "generic":
            generic_dir = input_path
        elif source_format == "gnss-ins-sim":
            run_gnss_ins_sim_export(input_path, generic_dir, args)
        else:
            raise AssertionError(f"unexpected source format: {source_format}")

        imu_text = read_csv_text(generic_dir, "imu.csv")
        gnss_text = read_csv_text(generic_dir, "gnss.csv")
        reference_attitude_text = read_optional_csv_text(generic_dir, "reference_attitude.csv")
        reference_mount_text = read_optional_csv_text(generic_dir, "reference_mount.csv")
        imu_stats = validate_csv(imu_text, IMU_HEADER, "imu.csv")
        gnss_stats = validate_csv(gnss_text, GNSS_HEADER, "gnss.csv")
        reference_attitude_stats = (
            validate_csv(reference_attitude_text, REFERENCE_RPY_HEADER, "reference_attitude.csv")
            if reference_attitude_text is not None
            else None
        )
        reference_mount_stats = (
            validate_csv(reference_mount_text, REFERENCE_RPY_HEADER, "reference_mount.csv")
            if reference_mount_text is not None
            else None
        )

        imu_file = write_gzip_csv(output_dir / "imu.csv.gz", imu_text)
        gnss_file = write_gzip_csv(output_dir / "gnss.csv.gz", gnss_text)
        reference_attitude_file = (
            write_gzip_csv(output_dir / "reference_attitude.csv.gz", reference_attitude_text)
            if reference_attitude_text is not None
            else None
        )
        reference_mount_file = (
            write_gzip_csv(output_dir / "reference_mount.csv.gz", reference_mount_text)
            if reference_mount_text is not None
            else None
        )
        if args.keep_csv:
            (output_dir / "imu.csv").write_text(imu_text, encoding="utf-8")
            (output_dir / "gnss.csv").write_text(gnss_text, encoding="utf-8")
            if reference_attitude_text is not None:
                (output_dir / "reference_attitude.csv").write_text(
                    reference_attitude_text, encoding="utf-8"
                )
            if reference_mount_text is not None:
                (output_dir / "reference_mount.csv").write_text(
                    reference_mount_text, encoding="utf-8"
                )

        files = {
            "imu": file_manifest(imu_file, imu_stats, IMU_HEADER),
            "gnss": file_manifest(gnss_file, gnss_stats, GNSS_HEADER),
        }
        samples = {
            "imu": imu_stats.rows,
            "gnss": gnss_stats.rows,
        }
        time_values = [
            imu_stats.first_t_s,
            gnss_stats.first_t_s,
            imu_stats.last_t_s,
            gnss_stats.last_t_s,
        ]
        if reference_attitude_file is not None and reference_attitude_stats is not None:
            files["reference_attitude"] = file_manifest(
                reference_attitude_file, reference_attitude_stats, REFERENCE_RPY_HEADER
            )
            samples["reference_attitude"] = reference_attitude_stats.rows
            time_values.extend(
                [reference_attitude_stats.first_t_s, reference_attitude_stats.last_t_s]
            )
        if reference_mount_file is not None and reference_mount_stats is not None:
            files["reference_mount"] = file_manifest(
                reference_mount_file, reference_mount_stats, REFERENCE_RPY_HEADER
            )
            samples["reference_mount"] = reference_mount_stats.rows
            time_values.extend([reference_mount_stats.first_t_s, reference_mount_stats.last_t_s])
        manifest = {
            "manifest_version": 1,
            "dataset_id": dataset_id,
            "title": args.title or dataset_id,
            "schema": "generic-replay-v1",
            "created_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "source": {
                "format": source_format,
                "input_path": str(input_path),
            },
            "files": files,
            "samples": samples,
            "time_span_s": {
                "start": min_present(time_values),
                "end": max_present(time_values),
            },
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"dataset_package_dir={output_dir}")
    print(f"manifest={output_dir / 'manifest.json'}")
    print(f"imu_samples={imu_stats.rows} gnss_samples={gnss_stats.rows}")


def ensure_output_dir(output_dir: Path, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [entry for entry in output_dir.iterdir()]
    if entries and not force:
        raise ValueError(f"output directory is not empty: {output_dir} (pass --force to overwrite)")
    if force:
        for name in [
            "manifest.json",
            "imu.csv.gz",
            "gnss.csv.gz",
            "reference_attitude.csv.gz",
            "reference_mount.csv.gz",
            "imu.csv",
            "gnss.csv",
            "reference_attitude.csv",
            "reference_mount.csv",
        ]:
            path = output_dir / name
            if path.exists():
                path.unlink()


def detect_source_format(input_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if has_generic_replay(input_dir):
        return "generic"
    if has_gnss_ins_sim(input_dir):
        return "gnss-ins-sim"
    raise ValueError(
        "could not detect input format; expected generic imu.csv/gnss.csv "
        "or gnss-ins-sim time.csv/gps_time.csv outputs"
    )


def has_generic_replay(input_dir: Path) -> bool:
    return any((input_dir / f"imu.csv{suffix}").exists() for suffix in ["", ".gz"]) and any(
        (input_dir / f"gnss.csv{suffix}").exists() for suffix in ["", ".gz"]
    )


def has_gnss_ins_sim(input_dir: Path) -> bool:
    return (input_dir / "time.csv").exists() and (input_dir / "gps_time.csv").exists()


def run_gnss_ins_sim_export(input_dir: Path, generic_dir: Path, args: argparse.Namespace) -> None:
    cmd = [
        "cargo",
        "run",
        "--release",
        "-p",
        "sim",
        "--bin",
        "export_gnss_ins_sim_generic",
        "--",
        str(input_dir),
        str(generic_dir),
        "--signal-source",
        args.signal_source,
        "--data-key",
        str(args.data_key),
        "--mount-roll-deg",
        str(args.mount_roll_deg),
        "--mount-pitch-deg",
        str(args.mount_pitch_deg),
        "--mount-yaw-deg",
        str(args.mount_yaw_deg),
        "--gnss-pos-std-m",
        str(args.gnss_pos_std_m),
        "--gnss-vel-std-mps",
        str(args.gnss_vel_std_mps),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_csv_text(input_dir: Path, name: str) -> str:
    plain = input_dir / name
    gz = input_dir / f"{name}.gz"
    if plain.exists():
        text = plain.read_text(encoding="utf-8")
    elif gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8", newline="") as handle:
            text = handle.read()
    else:
        raise ValueError(f"missing {name} in {input_dir}")
    return normalize_newline(text)


def read_optional_csv_text(input_dir: Path, name: str) -> str | None:
    plain = input_dir / name
    gz = input_dir / f"{name}.gz"
    if plain.exists() or gz.exists():
        return read_csv_text(input_dir, name)
    return None


def normalize_newline(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized if normalized.endswith("\n") else normalized + "\n"


def validate_csv(text: str, expected_header: list[str], name: str) -> CsvStats:
    rows = csv.reader(text.splitlines())
    try:
        header = next(rows)
    except StopIteration as exc:
        raise ValueError(f"{name} is empty") from exc
    if header != expected_header:
        raise ValueError(f"{name} header mismatch: expected {','.join(expected_header)}")

    count = 0
    first_t_s = None
    last_t_s = None
    for line_number, row in enumerate(rows, start=2):
        if not row:
            continue
        if len(row) != len(expected_header):
            raise ValueError(f"{name}:{line_number} expected {len(expected_header)} columns")
        try:
            values = [float(value) for value in row]
        except ValueError as exc:
            raise ValueError(f"{name}:{line_number} contains a non-numeric value") from exc
        t_s = values[0]
        if first_t_s is None:
            first_t_s = t_s
        last_t_s = t_s
        count += 1

    if count == 0:
        raise ValueError(f"{name} contains no samples")
    return CsvStats(rows=count, first_t_s=first_t_s, last_t_s=last_t_s)


def write_gzip_csv(path: Path, text: str) -> FileStats:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(text.encode("utf-8"))
    return file_stats(path)


def file_stats(path: Path) -> FileStats:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return FileStats(
        path=path.name,
        bytes=path.stat().st_size,
        sha256=digest.hexdigest(),
    )


def file_manifest(file: FileStats, csv_stats: CsvStats, columns: list[str]) -> dict[str, object]:
    return {
        "path": file.path,
        "media_type": "text/csv",
        "encoding": "gzip",
        "bytes": file.bytes,
        "sha256": file.sha256,
        "rows": csv_stats.rows,
        "first_t_s": csv_stats.first_t_s,
        "last_t_s": csv_stats.last_t_s,
        "columns": columns,
    }


def min_present(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def max_present(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
