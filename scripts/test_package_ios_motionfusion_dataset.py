#!/usr/bin/env python3
"""Tests for package_ios_motionfusion_dataset.py manifest updates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import package_ios_motionfusion_dataset as ios_package


PACKAGE_MANIFEST = {
    "files": {
        "imu": {
            "path": "imu.csv.gz",
            "sha256": "a" * 64,
            "bytes": 101,
            "rows": 12_345,
        },
        "gnss": {
            "path": "gnss.csv.gz",
            "sha256": "b" * 64,
            "bytes": 202,
            "rows": 123,
        },
        "reference_position": {
            "path": "reference_position.csv.gz",
            "sha256": "c" * 64,
            "bytes": 303,
            "rows": 123,
        },
    }
}


class PackageIosMotionFusionDatasetTests(unittest.TestCase):
    def test_update_web_manifest_appends_browser_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text('{"datasets": []}\n', encoding="utf-8")

            ios_package.update_web_manifest(
                manifest_path,
                dataset_id="ios-reverse-parking-001",
                label="iOS reverse parking 001",
                description="Drive with reverse parking.",
                base_url="ios-reverse-parking-001",
                package_manifest=PACKAGE_MANIFEST,
                force=False,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["datasets"],
                [
                    {
                        "id": "ios-reverse-parking-001",
                        "label": "iOS reverse parking 001",
                        "description": "Drive with reverse parking.",
                        "base_url": "ios-reverse-parking-001",
                        "imu_gz": "imu.csv.gz",
                        "gnss_gz": "gnss.csv.gz",
                        "reference_position_gz": "reference_position.csv.gz",
                    }
                ],
            )

    def test_update_ci_manifest_uses_package_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                '{"schema_version": 1, "datasets": []}\n',
                encoding="utf-8",
            )

            ios_package.update_ci_manifest(
                manifest_path,
                dataset_id="ios-reverse-parking-001",
                version="v1",
                description="Drive with reverse parking.",
                license_name="MIT",
                base_url="../../web/datasets/ios-reverse-parking-001",
                package_manifest=PACKAGE_MANIFEST,
                force=False,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset = manifest["datasets"][0]
            self.assertEqual(dataset["id"], "ios-reverse-parking-001")
            self.assertEqual(dataset["files"][0]["sha256"], "a" * 64)
            self.assertEqual(dataset["files"][1]["bytes"], 202)
            self.assertEqual(
                dataset["files"][2]["url"],
                "../../web/datasets/ios-reverse-parking-001/reference_position.csv.gz",
            )
            self.assertEqual(dataset["smoke"]["max_imu_rows"], 12_345)
            self.assertEqual(dataset["smoke"]["max_gnss_rows"], 123)

    def test_existing_entry_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                '{"datasets": [{"id": "ios-reverse-parking-001"}]}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "already exists"):
                ios_package.update_web_manifest(
                    manifest_path,
                    dataset_id="ios-reverse-parking-001",
                    label="iOS reverse parking 001",
                    description="Drive with reverse parking.",
                    base_url="ios-reverse-parking-001",
                    package_manifest=PACKAGE_MANIFEST,
                    force=False,
                )

    def test_sanitize_package_manifest_removes_local_input_path(self) -> None:
        manifest = {
            "source": {
                "format": "generic",
                "input_path": "/Users/example/repo/target/ios-web-replay/log",
            }
        }

        ios_package.sanitize_package_manifest(manifest)

        self.assertEqual(manifest, {"source": {"format": "generic"}})


if __name__ == "__main__":
    unittest.main()
