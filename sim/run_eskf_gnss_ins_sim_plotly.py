#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "plotly is required. Install it with `python3 -m pip install plotly`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GNSS_INS_SIM = REPO_ROOT.parents[2] / "gnss-ins-sim"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a gnss-ins-sim dataset, run eskf_eval_gnss_ins_sim, and show Plotly dark quaternion results."
    )
    parser.add_argument(
        "--gnss-ins-sim-dir",
        type=Path,
        default=DEFAULT_GNSS_INS_SIM,
        help="Path to the gnss-ins-sim repository.",
    )
    parser.add_argument(
        "--motion-def",
        default=str(REPO_ROOT / "sim" / "motion_profiles" / "city_blocks_15min.csv"),
        help="Motion definition filename under gnss-ins-sim/demo_motion_def_files, or an arbitrary CSV path.",
    )
    parser.add_argument(
        "--signal-source",
        choices=["ref", "meas"],
        default="meas",
        help="Use reference or noisy simulated signals.",
    )
    parser.add_argument("--data-key", type=int, default=0)
    parser.add_argument(
        "--seed-source",
        choices=["internal-align", "external-truth"],
        default="internal-align",
        help="How the ESKF coarse mount is seeded before residual q_cs estimation.",
    )
    parser.add_argument("--mount-roll-deg", type=float, default=5.0)
    parser.add_argument("--mount-pitch-deg", type=float, default=-5.0)
    parser.add_argument("--mount-yaw-deg", type=float, default=5.0)
    parser.add_argument("--imu-hz", type=float, default=100.0)
    parser.add_argument("--gnss-hz", type=float, default=2.0)
    parser.add_argument(
        "--imu-accuracy",
        default="mid-accuracy",
        help="gnss-ins-sim built-in IMU accuracy preset.",
    )
    parser.add_argument(
        "--generator-env",
        choices=["conda-base", "current"],
        default="conda-base",
        help="Python environment used to run gnss-ins-sim dataset generation.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        help="If set, save Plotly HTML to this path instead of opening it interactively.",
    )
    parser.add_argument(
        "--keep-data-dir",
        type=Path,
        help="If set, keep the generated gnss-ins-sim dataset in this directory.",
    )
    parser.add_argument(
        "--keep-residual-csv",
        type=Path,
        help="If set, keep the ESKF residual CSV at this path.",
    )
    return parser.parse_args()


def generate_dataset(args: argparse.Namespace, out_dir: Path) -> None:
    gnss_ins_sim_dir = args.gnss_ins_sim_dir.resolve()
    motion_def_arg = Path(args.motion_def)
    if motion_def_arg.exists():
        motion_def = motion_def_arg.resolve()
    else:
        motion_def = (gnss_ins_sim_dir / "demo_motion_def_files" / args.motion_def).resolve()
    if not motion_def.exists():
        raise SystemExit(f"motion definition not found: {motion_def}")

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
    if args.generator_env == "conda-base":
        cmd = ["conda", "run", "-n", "base", "python", "-c", code]
    else:
        cmd = [sys.executable, "-c", code]
    subprocess.run(cmd, cwd=str(gnss_ins_sim_dir), check=True)


def run_eskf_eval(args: argparse.Namespace, data_dir: Path, residual_csv: Path) -> str:
    cmd = [
        "cargo",
        "run",
        "-p",
        "sim",
        "--bin",
        "eskf_eval_gnss_ins_sim",
        "--",
        str(data_dir),
        "--signal-source",
        args.signal_source,
        "--data-key",
        str(args.data_key),
        "--seed-source",
        args.seed_source,
        f"--mount-roll-deg={args.mount_roll_deg}",
        f"--mount-pitch-deg={args.mount_pitch_deg}",
        f"--mount-yaw-deg={args.mount_yaw_deg}",
        "--residual-csv",
        str(residual_csv),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def read_series(path: Path) -> dict[str, list[float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"no rows found in {path}")
    out: dict[str, list[float]] = {k: [] for k in rows[0].keys() if k is not None}
    for row in rows:
        for key, value in row.items():
            if key is not None:
                out[key].append(float(value))
    return out


def quat_prefix(series: dict[str, list[float]], *candidates: str) -> str:
    for prefix in candidates:
        if all(f"{prefix}_q{j}" in series for j in range(4)):
            return prefix
    raise KeyError(f"missing quaternion prefix; tried {candidates}")


def align_quat_signs(series: dict[str, list[float]], prefix: str, ref_prefix: str = "truth") -> list[list[float]]:
    out = []
    for i in range(len(series["t_s"])):
        q = [series[f"{prefix}_q{j}"][i] for j in range(4)]
        ref = [series[f"{ref_prefix}_q{j}"][i] for j in range(4)]
        if sum(a * b for a, b in zip(q, ref)) < 0.0:
            q = [-x for x in q]
        out.append(q)
    return out


def add_component_traces(fig, row: int, col: int, t, truth, seed, legacy_full, full, idx: int) -> None:
    fig.add_trace(
        go.Scatter(x=t, y=[q[idx] for q in truth], name=f"truth q{idx}", mode="lines", line={"color": "#4ea1ff", "width": 2}),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(x=t, y=[q[idx] for q in seed], name=f"seed q{idx}", mode="lines", line={"color": "#ff9f1c", "width": 1.5, "dash": "dot"}),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(x=t, y=[q[idx] for q in full], name=f"full seed*inv(qcs) q{idx}", mode="lines", line={"color": "#00d1b2", "width": 2}),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(x=t, y=[q[idx] for q in legacy_full], name=f"legacy qcs*seed q{idx}", mode="lines", line={"color": "#ff6b6b", "width": 2}),
        row=row,
        col=col,
    )


def make_plot(series: dict[str, list[float]], title: str):
    t = series["t_s"]
    truth = align_quat_signs(series, "truth", "truth")
    seed = align_quat_signs(series, "seed")
    legacy_prefix = quat_prefix(series, "legacy_full_a", "full_a")
    full_prefix = quat_prefix(series, "full_seed_inv_qcs", "full_b")
    legacy_full = align_quat_signs(series, legacy_prefix)
    full = align_quat_signs(series, full_prefix)
    legacy_err_key = "legacy_full_a_err_deg" if "legacy_full_a_err_deg" in series else "full_a_err_deg"
    full_err_key = "full_seed_inv_qcs_err_deg" if "full_seed_inv_qcs_err_deg" in series else "full_b_err_deg"

    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        subplot_titles=[
            "Quaternion Geodesic Error",
            "Residual Mount Angle",
            "q0",
            "q1",
            "q2",
            "q3",
        ],
    )

    fig.add_trace(
        go.Scatter(x=t, y=series["seed_err_deg"], name="seed err [deg]", mode="lines", line={"color": "#ff9f1c", "width": 2}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=series["align_err_deg"], name="align err [deg]", mode="lines", line={"color": "#b388eb", "width": 2, "dash": "dot"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=series[full_err_key], name="full seed*inv(qcs) err [deg]", mode="lines", line={"color": "#00d1b2", "width": 2}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=series[legacy_err_key], name="legacy qcs*seed err [deg]", mode="lines", line={"color": "#ff6b6b", "width": 2}),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=t, y=series["qcs_angle_deg"], name="|q_cs| [deg]", mode="lines", line={"color": "#ffd166", "width": 2}),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=series["speed_mps"], name="speed [m/s]", mode="lines", line={"color": "#4ea1ff", "width": 2}),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=series["course_rate_dps"], name="course-rate [deg/s]", mode="lines", line={"color": "#ab63fa", "width": 1.5, "dash": "dot"}),
        row=1,
        col=2,
    )

    add_component_traces(fig, 2, 1, t, truth, seed, legacy_full, full, 0)
    add_component_traces(fig, 2, 2, t, truth, seed, legacy_full, full, 1)
    add_component_traces(fig, 3, 1, t, truth, seed, legacy_full, full, 2)
    add_component_traces(fig, 3, 2, t, truth, seed, legacy_full, full, 3)

    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=1200,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    fig.update_xaxes(title_text="time [s]", row=3, col=1)
    fig.update_xaxes(title_text="time [s]", row=3, col=2)
    fig.update_yaxes(title_text="error [deg]", row=1, col=1)
    fig.update_yaxes(title_text="deg / m/s / deg/s", row=1, col=2)
    return fig


def main() -> None:
    args = parse_args()
    temp_dir_obj = None
    if args.keep_data_dir:
        data_dir = args.keep_data_dir.resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="eskf-gnss-ins-sim-")
        data_dir = Path(temp_dir_obj.name)

    residual_csv = args.keep_residual_csv.resolve() if args.keep_residual_csv else data_dir / "eskf_eval.csv"
    generate_dataset(args, data_dir)
    summary = run_eskf_eval(args, data_dir, residual_csv)
    series = read_series(residual_csv)
    title = (
        f"ESKF Mount Eval: {Path(args.motion_def).name} | "
        f"seed={args.seed_source} | truth=({args.mount_roll_deg:.1f}, {args.mount_pitch_deg:.1f}, {args.mount_yaw_deg:.1f}) deg"
    )
    fig = make_plot(series, title)

    if args.output_html:
        out_html = args.output_html.resolve()
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs="cdn")
        print(summary)
        print(f"wrote Plotly HTML: {out_html}")
    else:
        tmp_html = data_dir / "eskf_eval_dark.html"
        fig.write_html(tmp_html, include_plotlyjs="cdn")
        print(summary)
        webbrowser.open(tmp_html.as_uri())

    if temp_dir_obj is None and args.keep_residual_csv and residual_csv.parent != args.keep_data_dir.resolve():
        shutil.copy2(residual_csv, args.keep_residual_csv.resolve())


if __name__ == "__main__":
    main()
