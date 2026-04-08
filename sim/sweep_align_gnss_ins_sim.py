#!/usr/bin/env python3

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GNSS_INS_SIM = REPO_ROOT.parents[2] / "gnss-ins-sim"
DEFAULT_MOTION_DEF = REPO_ROOT / "sim" / "motion_profiles" / "city_blocks_15min.csv"


@dataclass(frozen=True)
class SweepCase:
    label: str
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep truth misalignment cases on a gnss-ins-sim motion profile and flag wrong-way jumps."
    )
    parser.add_argument(
        "--gnss-ins-sim-dir",
        type=Path,
        default=DEFAULT_GNSS_INS_SIM,
    )
    parser.add_argument(
        "--motion-def",
        type=Path,
        default=DEFAULT_MOTION_DEF,
        help="Motion profile CSV path.",
    )
    parser.add_argument(
        "--signal-source",
        choices=["ref", "meas"],
        default="ref",
    )
    parser.add_argument("--imu-hz", type=float, default=100.0)
    parser.add_argument("--gnss-hz", type=float, default=10.0)
    parser.add_argument("--imu-accuracy", default="mid-accuracy")
    parser.add_argument("--jump-threshold-deg", type=float, default=0.5)
    parser.add_argument("--error-growth-threshold-deg", type=float, default=0.25)
    parser.add_argument("--work-dir", type=Path, help="If set, keep generated dataset and residuals here.")
    parser.add_argument("--summary-csv", type=Path, help="Optional CSV output for the sweep summary.")
    return parser.parse_args()


def default_cases() -> list[SweepCase]:
    vals_small = [-10.0, -5.0, 5.0, 10.0]
    cases: list[SweepCase] = []
    for v in vals_small:
        cases.append(SweepCase(f"roll_{v:+.0f}", v, 0.0, 0.0))
    for v in vals_small:
        cases.append(SweepCase(f"pitch_{v:+.0f}", 0.0, v, 0.0))
    for v in [-45.0, -20.0, 20.0, 45.0]:
        cases.append(SweepCase(f"yaw_{v:+.0f}", 0.0, 0.0, v))
    return cases


def generate_dataset(args: argparse.Namespace, out_dir: Path) -> None:
    gnss_ins_sim_dir = args.gnss_ins_sim_dir.resolve()
    motion_def = args.motion_def.resolve()
    code = f"""
import sys
sys.path.insert(0, {str(gnss_ins_sim_dir)!r})
from gnss_ins_sim.sim import imu_model, ins_sim
imu = imu_model.IMU(accuracy={args.imu_accuracy!r}, axis=6, gps=True)
sim = ins_sim.Sim([{args.imu_hz}, {args.gnss_hz}, {args.imu_hz}],
                  {str(motion_def)!r},
                  ref_frame=0,
                  imu=imu,
                  mode=None,
                  env=None,
                  algorithm=None)
sim.run(1)
sim.results({str(out_dir)!r})
"""
    subprocess.run([sys.executable, "-c", code], cwd=str(gnss_ins_sim_dir), check=True)


def wrap_deg180(x: float) -> float:
    while x > 180.0:
        x -= 360.0
    while x <= -180.0:
        x += 360.0
    return x


def read_csv_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if k is not None})
    return rows


def analyze_axis(rows: list[dict[str, float]], est_key: str, err_key: str, jump_threshold: float, error_growth_threshold: float):
    max_abs_step = 0.0
    wrong_way_count = 0
    max_wrong_way_step = 0.0
    max_wrong_way_t = None
    for prev, cur in zip(rows, rows[1:]):
        step = wrap_deg180(cur[est_key] - prev[est_key])
        max_abs_step = max(max_abs_step, abs(step))
        prev_abs_err = abs(prev[err_key])
        cur_abs_err = abs(cur[err_key])
        if abs(step) >= jump_threshold and cur_abs_err > prev_abs_err + error_growth_threshold:
            wrong_way_count += 1
            if abs(step) > max_wrong_way_step:
                max_wrong_way_step = abs(step)
                max_wrong_way_t = cur["t_s"]
    return {
        "max_abs_step_deg": max_abs_step,
        "wrong_way_count": wrong_way_count,
        "max_wrong_way_step_deg": max_wrong_way_step,
        "max_wrong_way_t_s": max_wrong_way_t,
    }


def run_case(args: argparse.Namespace, data_dir: Path, residual_dir: Path, case: SweepCase) -> dict[str, object]:
    residual_csv = residual_dir / f"{case.label}.csv"
    cmd = [
        "cargo", "run", "-p", "sim", "--bin", "align_eval_gnss_ins_sim", "--",
        str(data_dir),
        "--signal-source", args.signal_source,
        f"--mount-roll-deg={case.roll_deg}",
        f"--mount-pitch-deg={case.pitch_deg}",
        f"--mount-yaw-deg={case.yaw_deg}",
        "--residual-csv", str(residual_csv),
    ]
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, text=True, capture_output=True)
    stdout = completed.stdout

    rows = read_csv_rows(residual_csv)
    if not rows:
        raise RuntimeError(f"no residual rows for {case.label}")
    last = rows[-1]
    roll_stats = analyze_axis(rows, "align_roll_deg", "err_roll_deg", args.jump_threshold_deg, args.error_growth_threshold_deg)
    pitch_stats = analyze_axis(rows, "align_pitch_deg", "err_pitch_deg", args.jump_threshold_deg, args.error_growth_threshold_deg)
    yaw_stats = analyze_axis(rows, "align_yaw_deg", "err_yaw_deg", args.jump_threshold_deg, args.error_growth_threshold_deg)

    mount_ready = None
    for line in stdout.splitlines():
        if "mount_ready=" in line:
            mount_ready = line.split("mount_ready=")[-1].strip()

    return {
        "case": case.label,
        "truth_roll_deg": case.roll_deg,
        "truth_pitch_deg": case.pitch_deg,
        "truth_yaw_deg": case.yaw_deg,
        "mount_ready": mount_ready,
        "final_err_roll_deg": last["err_roll_deg"],
        "final_err_pitch_deg": last["err_pitch_deg"],
        "final_err_yaw_deg": last["err_yaw_deg"],
        "final_fwd_err_deg": last["fwd_err_deg"],
        "final_down_err_deg": last["down_err_deg"],
        "roll_max_abs_step_deg": roll_stats["max_abs_step_deg"],
        "roll_wrong_way_count": roll_stats["wrong_way_count"],
        "roll_max_wrong_way_step_deg": roll_stats["max_wrong_way_step_deg"],
        "roll_max_wrong_way_t_s": roll_stats["max_wrong_way_t_s"],
        "pitch_max_abs_step_deg": pitch_stats["max_abs_step_deg"],
        "pitch_wrong_way_count": pitch_stats["wrong_way_count"],
        "pitch_max_wrong_way_step_deg": pitch_stats["max_wrong_way_step_deg"],
        "pitch_max_wrong_way_t_s": pitch_stats["max_wrong_way_t_s"],
        "yaw_max_abs_step_deg": yaw_stats["max_abs_step_deg"],
        "yaw_wrong_way_count": yaw_stats["wrong_way_count"],
        "yaw_max_wrong_way_step_deg": yaw_stats["max_wrong_way_step_deg"],
        "yaw_max_wrong_way_t_s": yaw_stats["max_wrong_way_t_s"],
    }


def main() -> int:
    args = parse_args()
    if args.work_dir is not None:
        work_dir = args.work_dir.resolve()
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp = tempfile.TemporaryDirectory(prefix="align_sweep_")
        work_dir = Path(temp.name)

    data_dir = work_dir / "dataset"
    residual_dir = work_dir / "residuals"
    data_dir.mkdir(parents=True, exist_ok=True)
    residual_dir.mkdir(parents=True, exist_ok=True)

    generate_dataset(args, data_dir)

    results = []
    for case in default_cases():
        result = run_case(args, data_dir, residual_dir, case)
        results.append(result)
        print(
            f"{case.label}: final_err(r/p/y)=({result['final_err_roll_deg']:.3f},"
            f"{result['final_err_pitch_deg']:.3f},{result['final_err_yaw_deg']:.3f}) "
            f"wrong_way(r/p/y)=({result['roll_wrong_way_count']},"
            f"{result['pitch_wrong_way_count']},{result['yaw_wrong_way_count']})"
        )

    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"summary_csv={args.summary_csv}")

    print(f"work_dir={work_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
