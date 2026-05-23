#!/usr/bin/env python3
"""Convert an iOS .motionfusion log into an official web dataset package.

This script is a thin orchestrator around the existing iOS exporter and generic
dataset packager. It keeps iOS-specific parsing in mobile/ios/scripts and keeps
the web-hosted dataset format hardware-agnostic.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESCRIPTION = (
    "iOS MotionFusion drive exported to generic IMU/GNSS replay format."
)
WEB_REFERENCE_KEYS = {
    "reference_attitude": "reference_attitude_gz",
    "reference_mount": "reference_mount_gz",
    "reference_position": "reference_position_gz",
    "reference_motion": "reference_motion_gz",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an iOS .motionfusion recording, package it under "
            "web/datasets, and update dataset manifests."
        )
    )
    parser.add_argument("input", type=Path, help="Input iOS .motionfusion JSON recording")
    parser.add_argument(
        "--dataset-id",
        help="Official dataset id. Defaults to a sanitized input filename stem.",
    )
    parser.add_argument(
        "--title",
        help="Human-readable dataset title stored in the per-dataset manifest.",
    )
    parser.add_argument(
        "--label",
        help="Browser-visible label. Defaults to --title, then the dataset id.",
    )
    parser.add_argument(
        "--description",
        default=DEFAULT_DESCRIPTION,
        help="Browser/CI dataset description.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        help="Generic CSV export directory. Defaults to target/ios-web-replay/<dataset-id>.",
    )
    parser.add_argument(
        "--web-datasets-dir",
        type=Path,
        default=ROOT / "web" / "datasets",
        help="Directory that contains web-hosted datasets.",
    )
    parser.add_argument(
        "--web-manifest",
        type=Path,
        default=ROOT / "web" / "datasets" / "manifest.json",
        help="Browser dataset manifest to update.",
    )
    parser.add_argument(
        "--ci-manifest",
        type=Path,
        default=ROOT / ".github" / "datasets" / "generic-datasets.json",
        help="CI dataset manifest to update.",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="CI dataset version.",
    )
    parser.add_argument(
        "--license",
        default="MIT",
        help="CI dataset license metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output package and replace existing manifest entries.",
    )
    parser.add_argument(
        "--no-update-web-manifest",
        action="store_true",
        help="Package the dataset without updating web/datasets/manifest.json.",
    )
    parser.add_argument(
        "--no-update-ci-manifest",
        action="store_true",
        help="Package the dataset without updating .github/datasets/generic-datasets.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        package_ios_dataset(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def package_ios_dataset(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise ValueError(f"input file does not exist: {input_path}")

    dataset_id = validate_dataset_id(args.dataset_id or slugify(input_path.stem))
    title = args.title or args.label or dataset_id
    label = args.label or title
    intermediate_dir = resolve_path(
        args.intermediate_dir or ROOT / "target" / "ios-web-replay" / dataset_id
    )
    web_datasets_dir = resolve_path(args.web_datasets_dir)
    output_dir = web_datasets_dir / dataset_id

    run_command(
        [
            sys.executable,
            str(ROOT / "mobile" / "ios" / "scripts" / "export_motionfusion.py"),
            str(input_path),
            "--output-dir",
            str(intermediate_dir),
        ]
    )
    run_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_dataset.py"),
            str(intermediate_dir),
            str(output_dir),
            "--dataset-id",
            dataset_id,
            "--title",
            title,
            *(["--force"] if args.force else []),
        ]
    )

    package_manifest_path = output_dir / "manifest.json"
    package_manifest = load_json(package_manifest_path)
    sanitize_package_manifest(package_manifest)
    write_json(package_manifest_path, package_manifest)
    if not args.no_update_web_manifest:
        update_web_manifest(
            resolve_path(args.web_manifest),
            dataset_id=dataset_id,
            label=label,
            description=args.description,
            base_url=dataset_id,
            package_manifest=package_manifest,
            force=args.force,
        )
    if not args.no_update_ci_manifest:
        update_ci_manifest(
            resolve_path(args.ci_manifest),
            dataset_id=dataset_id,
            version=args.version,
            description=args.description,
            license_name=args.license,
            base_url=f"../../web/datasets/{dataset_id}",
            package_manifest=package_manifest,
            force=args.force,
        )

    print(f"dataset_id={dataset_id}")
    print(f"generic_replay_dir={intermediate_dir}")
    print(f"web_dataset_dir={output_dir}")


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "ios-motionfusion-dataset"


def validate_dataset_id(dataset_id: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", dataset_id):
        raise ValueError(
            f"dataset id must match ^[a-z0-9][a-z0-9._-]*$: {dataset_id!r}"
        )
    return dataset_id


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def upsert_dataset(
    manifest: dict[str, Any],
    entry: dict[str, Any],
    *,
    force: bool,
    match_version: str | None = None,
) -> None:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("manifest is missing a datasets array")

    for index, existing in enumerate(datasets):
        if not isinstance(existing, dict):
            continue
        same_id = existing.get("id") == entry["id"]
        same_version = match_version is None or existing.get("version") == match_version
        if same_id and same_version:
            if not force:
                raise ValueError(
                    f"dataset {entry['id']} already exists; pass --force to replace it"
                )
            datasets[index] = entry
            return

    datasets.append(entry)


def update_web_manifest(
    manifest_path: Path,
    *,
    dataset_id: str,
    label: str,
    description: str,
    base_url: str,
    package_manifest: dict[str, Any],
    force: bool,
) -> None:
    manifest = load_json(manifest_path)
    files = package_files(package_manifest)
    entry: dict[str, Any] = {
        "id": dataset_id,
        "label": label,
        "description": description,
        "base_url": base_url,
        "imu_gz": files["imu"]["path"],
        "gnss_gz": files["gnss"]["path"],
    }
    for package_key, web_key in WEB_REFERENCE_KEYS.items():
        if package_key in files:
            entry[web_key] = files[package_key]["path"]

    upsert_dataset(manifest, entry, force=force)
    write_json(manifest_path, manifest)


def update_ci_manifest(
    manifest_path: Path,
    *,
    dataset_id: str,
    version: str,
    description: str,
    license_name: str,
    base_url: str,
    package_manifest: dict[str, Any],
    force: bool,
) -> None:
    manifest = load_json(manifest_path)
    files = package_files(package_manifest)
    file_entries = []
    for key in [
        "imu",
        "gnss",
        "reference_position",
        "reference_attitude",
        "reference_mount",
        "reference_motion",
    ]:
        file_info = files.get(key)
        if file_info is None:
            continue
        path = file_info["path"]
        file_entries.append(
            {
                "path": path,
                "url": f"{base_url}/{path}",
                "sha256": file_info["sha256"],
                "bytes": file_info["bytes"],
            }
        )

    entry = {
        "id": dataset_id,
        "version": version,
        "description": description,
        "license": license_name,
        "replay_dir": ".",
        "files": file_entries,
        "smoke": {
            "enabled": False,
            "max_imu_rows": min(20_000, int(files["imu"]["rows"])),
            "max_gnss_rows": min(400, int(files["gnss"]["rows"])),
            "misalignment": "auto",
        },
    }
    upsert_dataset(manifest, entry, force=force, match_version=version)
    write_json(manifest_path, manifest)


def package_files(package_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = package_manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("package manifest is missing files object")
    for required in ["imu", "gnss"]:
        if required not in files:
            raise ValueError(f"package manifest is missing files.{required}")
    return files


def sanitize_package_manifest(package_manifest: dict[str, Any]) -> None:
    source = package_manifest.get("source")
    if isinstance(source, dict):
        source.pop("input_path", None)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
