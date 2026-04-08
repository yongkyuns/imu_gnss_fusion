#!/usr/bin/env python3

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "plotly is required. Install it with `python3 -m pip install plotly`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GNSS_INS_SIM = REPO_ROOT.parents[2] / "gnss-ins-sim"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a gnss-ins-sim dataset, run align_eval_gnss_ins_sim, and show Plotly dark results."
    )
    parser.add_argument(
        "--gnss-ins-sim-dir",
        type=Path,
        default=DEFAULT_GNSS_INS_SIM,
        help="Path to the gnss-ins-sim repository.",
    )
    parser.add_argument(
        "--motion-def",
        default="motion_def-long_drive.csv",
        help="Motion definition filename under gnss-ins-sim/demo_motion_def_files, or an arbitrary CSV path.",
    )
    parser.add_argument(
        "--signal-source",
        choices=["ref", "meas"],
        default="meas",
        help="Use reference or noisy simulated signals.",
    )
    parser.add_argument("--data-key", type=int, default=0)
    parser.add_argument("--mount-roll-deg", type=float, default=0.0)
    parser.add_argument("--mount-pitch-deg", type=float, default=0.0)
    parser.add_argument("--mount-yaw-deg", type=float, default=0.0)
    parser.add_argument("--imu-hz", type=float, default=100.0)
    parser.add_argument("--gnss-hz", type=float, default=10.0)
    parser.add_argument(
        "--imu-accuracy",
        default="mid-accuracy",
        help="gnss-ins-sim built-in IMU accuracy preset.",
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
        help="If set, keep the align residual CSV at this path.",
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
import os
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
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(gnss_ins_sim_dir),
        check=True,
    )


def run_align_eval(args: argparse.Namespace, data_dir: Path, residual_csv: Path) -> str:
    cmd = [
        "cargo",
        "run",
        "-p",
        "sim",
        "--bin",
        "align_eval_gnss_ins_sim",
        "--",
        str(data_dir),
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


def add_trace(fig, row: int, name: str, x, y, color: str, dash: str | None = None) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            mode="lines",
            line={"color": color, "width": 2, **({"dash": dash} if dash else {})},
        ),
        row=row,
        col=1,
    )


def make_plot(series: dict[str, list[float]], title: str):
    t = series["t_s"]
    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[
            [{}, {}],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        subplot_titles=[
            "Roll",
            "Pitch",
            "Yaw",
            "Unsigned Axis Errors",
            "Signed Axis Errors",
        ],
    )

    triples = [
        (1, "truth_roll_deg", "align_roll_deg", "err_roll_deg"),
        (2, "truth_pitch_deg", "align_pitch_deg", "err_pitch_deg"),
        (3, "truth_yaw_deg", "align_yaw_deg", "err_yaw_deg"),
    ]
    colors = {"truth": "#4ea1ff", "estimate": "#ff6b6b", "error": "#7bd88f"}
    positions = [
        (1, 1, "truth_roll_deg", "align_roll_deg", "err_roll_deg"),
        (1, 2, "truth_pitch_deg", "align_pitch_deg", "err_pitch_deg"),
        (2, 1, "truth_yaw_deg", "align_yaw_deg", "err_yaw_deg"),
    ]
    for row, col, truth_key, est_key, err_key in positions:
        fig.add_trace(
            go.Scatter(x=t, y=series[truth_key], name=truth_key, mode="lines", line={"color": colors["truth"], "width": 2}),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(x=t, y=series[est_key], name=est_key, mode="lines", line={"color": colors["estimate"], "width": 2}),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(x=t, y=series[err_key], name=err_key, mode="lines", line={"color": colors["error"], "width": 2, "dash": "dot"}),
            row=row,
            col=col,
        )

    fig.add_trace(
        go.Scatter(x=t, y=series["fwd_err_deg"], name="Forward axis", mode="lines", line={"color": "#f7b801", "width": 2}),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=series["down_err_deg"], name="Down axis", mode="lines", line={"color": "#c77dff", "width": 2}),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=t, y=series["rot_err_deg"], name="Rotation", mode="lines", line={"color": "#00d1b2", "width": 2}),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(x=t, y=series["fwd_err_signed_deg"], name="Forward signed", mode="lines", line={"color": "#f7b801", "width": 2}),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=series["down_err_signed_deg"], name="Down signed", mode="lines", line={"color": "#c77dff", "width": 2}),
        row=3,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=1200,
        width=1600,
        title=title,
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
        },
        margin={"t": 120, "r": 260},
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="deg", row=1, col=1)
    fig.update_yaxes(title_text="deg", row=1, col=2)
    fig.update_yaxes(title_text="deg", row=2, col=1)
    fig.update_yaxes(title_text="deg", row=2, col=2)
    fig.update_yaxes(title_text="deg", row=3, col=1)
    return fig


def main() -> int:
    args = parse_args()
    temp_dir_obj = None
    if args.keep_data_dir is not None:
        data_dir = args.keep_data_dir.resolve()
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="gnss_ins_sim_align_")
        data_dir = Path(temp_dir_obj.name)

    if args.keep_residual_csv is not None:
        residual_csv = args.keep_residual_csv.resolve()
        residual_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        residual_csv = data_dir / "align_residual.csv"

    generate_dataset(args, data_dir)
    stdout = run_align_eval(args, data_dir, residual_csv)
    print(stdout)
    print(f"dataset_dir={data_dir}")
    print(f"residual_csv={residual_csv}")

    series = read_series(residual_csv)
    title = (
        f"align_eval_gnss_ins_sim | {args.signal_source} | "
        f"mount=({args.mount_roll_deg:.2f},{args.mount_pitch_deg:.2f},{args.mount_yaw_deg:.2f}) deg | "
        f"{args.motion_def}"
    )
    fig = make_plot(series, title)

    if args.output_html is not None:
        output_html = args.output_html.resolve()
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_html, include_plotlyjs=True)
        print(f"plot_html={output_html}")
        webbrowser.open(output_html.as_uri())
    else:
        fig.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
