#!/usr/bin/env python3
"""Tests for export_motionfusion.py."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_motionfusion


class ExportMotionFusionTests(unittest.TestCase):
    def test_exports_web_csvs_and_summary(self) -> None:
        log = {
            "schemaVersion": 1,
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "unit-test",
            "startTime": "2026-05-13T12:00:00.000Z",
            "appVersion": "0",
            "buildNumber": "0",
            "events": [
                {
                    "schemaVersion": 1,
                    "kind": "imu",
                    "elapsedSec": 0.0,
                    "wallTime": "2026-05-13T12:00:00.000Z",
                    "imu": {
                        "sourceUptimeSec": 100.0,
                        "accelXMps2": 0.0,
                        "accelYMps2": 0.0,
                        "accelZMps2": 9.81,
                        "gyroXRadps": 0.1,
                        "gyroYRadps": 0.0,
                        "gyroZRadps": 0.0,
                        "attitudeReferenceFrame": "xArbitraryCorrectedZVertical",
                    },
                    "gnss": None,
                    "barometer": None,
                },
                {
                    "schemaVersion": 1,
                    "kind": "gnss",
                    "elapsedSec": 0.5,
                    "wallTime": "2026-05-13T12:00:00.500Z",
                    "imu": None,
                    "gnss": {
                        "latitudeDeg": 43.0,
                        "longitudeDeg": -79.0,
                        "altitudeM": 120.0,
                        "horizontalAccuracyM": 3.0,
                        "verticalAccuracyM": 5.0,
                        "speedMps": 2.0,
                        "courseDeg": 90.0,
                        "speedAccuracyMps": 0.5,
                        "courseAccuracyDeg": 2.0,
                        "positionNorthM": None,
                        "positionEastM": None,
                        "positionDownM": None,
                        "velocityNorthMps": None,
                        "velocityEastMps": None,
                        "velocityDownMps": None,
                    },
                    "barometer": None,
                },
                {
                    "schemaVersion": 1,
                    "kind": "imu",
                    "elapsedSec": 1.0,
                    "wallTime": "2026-05-13T12:00:01.000Z",
                    "imu": {
                        "sourceUptimeSec": 101.0,
                        "accelXMps2": 1.0,
                        "accelYMps2": 2.0,
                        "accelZMps2": 3.0,
                        "gyroXRadps": 0.0,
                        "gyroYRadps": 0.2,
                        "gyroZRadps": 0.0,
                        "attitudeReferenceFrame": "xArbitraryCorrectedZVertical",
                    },
                    "gnss": None,
                    "barometer": None,
                },
                {
                    "schemaVersion": 1,
                    "kind": "gnss",
                    "elapsedSec": 1.5,
                    "wallTime": "2026-05-13T12:00:01.500Z",
                    "imu": None,
                    "gnss": {
                        "latitudeDeg": 43.0,
                        "longitudeDeg": -79.0,
                        "altitudeM": 120.0,
                        "horizontalAccuracyM": 3.0,
                        "verticalAccuracyM": 5.0,
                        "speedMps": None,
                        "courseDeg": None,
                        "speedAccuracyMps": None,
                        "courseAccuracyDeg": None,
                        "positionNorthM": None,
                        "positionEastM": None,
                        "positionDownM": None,
                        "velocityNorthMps": None,
                        "velocityEastMps": None,
                        "velocityDownMps": None,
                    },
                    "barometer": None,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "sample.motionfusion"
            output_dir = Path(tmp) / "export"
            input_path.write_text(json.dumps(log), encoding="utf-8")

            result = export_motionfusion.export_motionfusion(input_path, output_dir)

            self.assertEqual(result.imu_csv, (output_dir / "imu.csv").resolve())
            self.assertEqual(result.gnss_csv, (output_dir / "gnss.csv").resolve())
            self.assertEqual(result.summary_txt, (output_dir / "summary.txt").resolve())
            self.assertEqual(result.duration_s, 1.5)
            self.assertEqual(result.skipped_gnss_missing_velocity, 1)

            with result.imu_csv.open(newline="", encoding="utf-8") as file:
                imu_rows = list(csv.reader(file))
            self.assertEqual(imu_rows[0], export_motionfusion.IMU_HEADER)
            self.assertEqual(len(imu_rows), 3)

            with result.gnss_csv.open(newline="", encoding="utf-8") as file:
                gnss_rows = list(csv.reader(file))
            self.assertEqual(gnss_rows[0], export_motionfusion.GNSS_HEADER)
            self.assertEqual(len(gnss_rows), 2)
            self.assertAlmostEqual(float(gnss_rows[1][4]), 0.0, places=12)
            self.assertAlmostEqual(float(gnss_rows[1][5]), 2.0, places=12)
            self.assertAlmostEqual(float(gnss_rows[1][13]), 1.5707963267948966)

            summary = result.summary_txt.read_text(encoding="utf-8")
            self.assertIn("counts: imu=2 gnss=1 barometer=0", summary)
            self.assertIn("imu_active: start_s=0.000 end_s=1.000 duration_s=1.000 rate_hz=1.000", summary)
            self.assertIn("gnss_active: start_s=0.500 end_s=0.500 duration_s=0.000 rate_hz=0.000", summary)
            self.assertIn("missing_gnss_velocity_rows: 1", summary)
            self.assertIn("derived_velocity=1", summary)
            self.assertIn("skipped=1", summary)
            self.assertIn("gnss_velocity_rows: explicit=0 derived_velocity=1 skipped_missing_velocity=1 WARN", summary)
            self.assertIn("stream_health: WARN", summary)
            self.assertIn("accel_magnitude_mps2:", summary)
            self.assertIn("gyro_magnitude_radps:", summary)

    def test_summary_flags_stream_health_regressions(self) -> None:
        events = []
        for t_s in [0.0, 0.01, 0.02, 0.03, 2.5]:
            events.append(imu_event(t_s))
        for t_s in [0.5, 1.5, 2.5, 3.5, 9.0]:
            events.append(gnss_event(t_s, speed_mps=4.0, course_deg=0.0))
        events.append(gnss_event(9.5, speed_mps=None, course_deg=None))
        events.append(
            {
                "schemaVersion": 1,
                "kind": "barometer",
                "elapsedSec": 10.0,
                "wallTime": "2026-05-13T12:00:10.000Z",
                "imu": None,
                "gnss": None,
                "barometer": {"relativeAltitudeM": 0.0, "pressureKpa": 100.0},
            }
        )
        log = {
            "schemaVersion": 1,
            "id": "00000000-0000-0000-0000-000000000002",
            "name": "stream-health-test",
            "startTime": "2026-05-13T12:00:00.000Z",
            "appVersion": "0",
            "buildNumber": "0",
            "events": events,
        }

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "sample.motionfusion"
            output_dir = Path(tmp) / "export"
            input_path.write_text(json.dumps(log), encoding="utf-8")

            result = export_motionfusion.export_motionfusion(input_path, output_dir)

            self.assertEqual(result.session_start_s, 0.0)
            self.assertEqual(result.session_end_s, 10.0)
            self.assertEqual(result.duration_s, 10.0)
            self.assertEqual(result.skipped_gnss_missing_velocity, 1)

            summary = result.summary_txt.read_text(encoding="utf-8")
            self.assertIn("session_window_s: start=0.000 end=10.000", summary)
            self.assertIn("imu_active: start_s=0.000 end_s=2.500 duration_s=2.500 rate_hz=1.600", summary)
            self.assertIn("gnss_active: start_s=0.500 end_s=9.000 duration_s=8.500 rate_hz=0.471", summary)
            self.assertIn("stream_end_lag_s: imu_to_gnss_end=6.500 WARN, imu_to_session_end=7.500 WARN", summary)
            self.assertIn("imu_large_gaps: count=1 WARN max_s=2.470 threshold_s=1.000", summary)
            self.assertIn("gnss_large_gaps: count=1 WARN max_s=5.500 threshold_s=5.000", summary)
            self.assertIn("gnss_velocity_rows: explicit=0 derived_velocity=5 skipped_missing_velocity=1 WARN", summary)
            self.assertIn("stream_health: WARN", summary)
            self.assertIn("imu_ended_before_gnss_end_by_6.500s", summary)
            self.assertIn("imu_ended_before_session_end_by_7.500s", summary)
            self.assertIn("imu_large_gaps_1", summary)
            self.assertIn("gnss_large_gaps_1", summary)
            self.assertIn("skipped_gnss_rows_1", summary)

    def test_stationary_gnss_does_not_require_course(self) -> None:
        log = {
            "schemaVersion": 1,
            "id": "00000000-0000-0000-0000-000000000003",
            "name": "stationary-gnss-test",
            "startTime": "2026-05-13T12:00:00.000Z",
            "appVersion": "0",
            "buildNumber": "0",
            "events": [
                imu_event(0.0),
                gnss_event(0.5, speed_mps=0.0, course_deg=None),
                gnss_event(1.5, speed_mps=0.1, course_deg=None),
                gnss_event(2.5, speed_mps=0.5, course_deg=None),
                imu_event(3.0),
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "sample.motionfusion"
            output_dir = Path(tmp) / "export"
            input_path.write_text(json.dumps(log), encoding="utf-8")

            result = export_motionfusion.export_motionfusion(input_path, output_dir)

            self.assertEqual(result.skipped_gnss_missing_velocity, 1)
            with result.gnss_csv.open(newline="", encoding="utf-8") as file:
                gnss_rows = list(csv.reader(file))
            self.assertEqual(len(gnss_rows), 3)
            self.assertAlmostEqual(float(gnss_rows[1][4]), 0.0, places=12)
            self.assertAlmostEqual(float(gnss_rows[1][5]), 0.0, places=12)
            self.assertAlmostEqual(float(gnss_rows[2][4]), 0.0, places=12)
            self.assertAlmostEqual(float(gnss_rows[2][5]), 0.0, places=12)

            summary = result.summary_txt.read_text(encoding="utf-8")
            self.assertIn("missing_gnss_velocity_rows: 2 (derived_velocity=2, skipped=1 WARN)", summary)


def imu_event(t_s: float) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "kind": "imu",
        "elapsedSec": t_s,
        "wallTime": "2026-05-13T12:00:00.000Z",
        "imu": {
            "sourceUptimeSec": 100.0 + t_s,
            "accelXMps2": 0.0,
            "accelYMps2": 0.0,
            "accelZMps2": 9.81,
            "gyroXRadps": 0.0,
            "gyroYRadps": 0.0,
            "gyroZRadps": 0.0,
            "attitudeReferenceFrame": "xArbitraryCorrectedZVertical",
        },
        "gnss": None,
        "barometer": None,
    }


def gnss_event(t_s: float, *, speed_mps: float | None, course_deg: float | None) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "kind": "gnss",
        "elapsedSec": t_s,
        "wallTime": "2026-05-13T12:00:00.000Z",
        "imu": None,
        "gnss": {
            "latitudeDeg": 43.0,
            "longitudeDeg": -79.0,
            "altitudeM": 120.0,
            "horizontalAccuracyM": 3.0,
            "verticalAccuracyM": 5.0,
            "speedMps": speed_mps,
            "courseDeg": course_deg,
            "speedAccuracyMps": 0.5 if speed_mps is not None else None,
            "courseAccuracyDeg": 2.0 if course_deg is not None else None,
            "positionNorthM": None,
            "positionEastM": None,
            "positionDownM": None,
            "velocityNorthMps": None,
            "velocityEastMps": None,
            "velocityDownMps": None,
        },
        "barometer": None,
    }


if __name__ == "__main__":
    unittest.main()
