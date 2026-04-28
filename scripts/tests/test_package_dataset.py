#!/usr/bin/env python3

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "package_dataset.py"

IMU_CSV = """t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2
0.0,0.1,0.2,0.3,1.0,2.0,3.0
0.01,0.4,0.5,0.6,4.0,5.0,6.0
"""

GNSS_CSV = """t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,pos_std_n_m,pos_std_e_m,pos_std_d_m,vel_std_n_mps,vel_std_e_mps,vel_std_d_mps,heading_rad
0.0,32.0,120.0,5.0,1.0,2.0,0.0,0.5,0.5,0.8,0.2,0.2,0.3,nan
0.5,32.1,120.1,5.5,1.5,2.5,0.1,0.5,0.5,0.8,0.2,0.2,0.3,1.2
"""

REFERENCE_ATTITUDE_CSV = """t_s,roll_deg,pitch_deg,yaw_deg
0.0,1.0,2.0,3.0
0.5,1.5,2.5,3.5
"""


class PackageDatasetTests(unittest.TestCase):
    def test_packages_generic_replay_as_hosted_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "generic"
            source.mkdir()
            (source / "imu.csv").write_text(IMU_CSV, encoding="utf-8")
            (source / "gnss.csv").write_text(GNSS_CSV, encoding="utf-8")
            (source / "reference_attitude.csv").write_text(
                REFERENCE_ATTITUDE_CSV, encoding="utf-8"
            )
            output = tmp_path / "hosted"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    str(source),
                    str(output),
                    "--dataset-id",
                    "fixture",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("imu_samples=2 gnss_samples=2", result.stdout)
            self.assertEqual(read_gzip_text(output / "imu.csv.gz"), IMU_CSV)
            self.assertEqual(read_gzip_text(output / "gnss.csv.gz"), GNSS_CSV)
            self.assertEqual(
                read_gzip_text(output / "reference_attitude.csv.gz"),
                REFERENCE_ATTITUDE_CSV,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_id"], "fixture")
            self.assertEqual(manifest["schema"], "generic-replay-v1")
            self.assertEqual(manifest["source"]["format"], "generic")
            self.assertEqual(
                manifest["samples"], {"imu": 2, "gnss": 2, "reference_attitude": 2}
            )
            self.assertEqual(manifest["files"]["imu"]["path"], "imu.csv.gz")
            self.assertEqual(manifest["files"]["gnss"]["path"], "gnss.csv.gz")
            self.assertEqual(
                manifest["files"]["reference_attitude"]["path"],
                "reference_attitude.csv.gz",
            )
            self.assertEqual(manifest["time_span_s"], {"start": 0.0, "end": 0.5})

    def test_rejects_raw_binary_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_log = tmp_path / "drive.bin"
            raw_log.write_bytes(b"\xb5\x62")
            output = tmp_path / "hosted"

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(raw_log), str(output)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("raw binary logs are not supported", result.stderr)


def read_gzip_text(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return handle.read()


if __name__ == "__main__":
    unittest.main()
