from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
EKF_ROOT = ROOT / "ekf"
OAN_ROOT = ROOT.parent.parent.parent / "open-aided-navigation"
DEFAULT_NOMINAL_MOUNT_DEG = np.array([-90.0, 0.0, 90.0], dtype=float)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EKF_ROOT) not in sys.path:
    sys.path.insert(0, str(EKF_ROOT))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return quat_normalize(
        np.array(
            [
                a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
            ],
            dtype=float,
        )
    )


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_rotmat(q) @ v


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(abs(np.dot(quat_normalize(a), quat_normalize(b))))
    dot = max(0.0, min(1.0, dot))
    return 2.0 * math.degrees(math.acos(dot))


def quat_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(0.5 * r), math.sin(0.5 * r)
    cp, sp = math.cos(0.5 * p), math.sin(0.5 * p)
    cy, sy = math.cos(0.5 * y), math.sin(0.5 * y)
    return quat_normalize(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=float,
        )
    )


def quat_to_rpy_deg(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.degrees(np.array([roll, pitch, yaw], dtype=float))


def read_csv_rows(path: Path, cols: int) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for i, row in enumerate(reader):
            if i == 0 or not row:
                continue
            vals = [float(cell.strip()) for cell in row[:cols]]
            if len(vals) != cols:
                raise ValueError(f"{path} expected {cols} columns")
            rows.append(vals)
    return rows


def csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def values_are_degrees_per_second(path: Path) -> bool:
    header = ",".join(csv_header(path)).lower()
    return "deg/s" in header


def write_semicolon_rows(path: Path, rows: Iterable[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerows(rows)


def build_loose_case(
    data_dir: Path,
    output_dir: Path,
    signal_source: str,
    mount_truth_deg: np.ndarray,
    nominal_mount_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.array([row[0] for row in read_csv_rows(data_dir / "time.csv", 1)], dtype=float)
    gps_time_s = np.array([row[0] for row in read_csv_rows(data_dir / "gps_time.csv", 1)], dtype=float)

    gyro_name = "ref_gyro.csv" if signal_source == "ref" else "gyro-0.csv"
    accel_name = "ref_accel.csv" if signal_source == "ref" else "accel-0.csv"
    gps_name = "ref_gps.csv" if signal_source == "ref" else "gps-0.csv"

    gyro_vehicle = np.asarray(read_csv_rows(data_dir / gyro_name, 3), dtype=float)
    accel_vehicle = np.asarray(read_csv_rows(data_dir / accel_name, 3), dtype=float)
    gps_rows = np.asarray(read_csv_rows(data_dir / gps_name, 6), dtype=float)

    if values_are_degrees_per_second(data_dir / gyro_name):
        gyro_vehicle = np.deg2rad(gyro_vehicle)

    q_nominal = quat_from_rpy_deg(*nominal_mount_deg.tolist())
    q_mis_truth = quat_from_rpy_deg(*mount_truth_deg.tolist())
    q_cs_true = quat_mul(q_mis_truth, q_nominal)
    q_sc_true = quat_conj(q_cs_true)

    gyro_sensor = np.vstack([quat_rotate(q_sc_true, row) for row in gyro_vehicle])
    accel_sensor = np.vstack([quat_rotate(q_sc_true, row) for row in accel_vehicle])

    prefix = output_dir.name
    acc_rows: list[list[object]] = [
        ["Sensor: Synthetic Accelerometer"],
        ["Vendor: gnss-ins-sim"],
        ["TimestampAcc [ns]", " AccX [m/s^2]", " AccY [m/s^2]", " AccZ [m/s^2]"],
    ]
    gyro_rows: list[list[object]] = [
        ["Sensor: Synthetic Gyroscope"],
        ["Vendor: gnss-ins-sim"],
        ["TimestampGyro [ns]", " GyroX [rad/s]", " GyroY [rad/s]", " GyroZ [rad/s]"],
    ]
    for t_s, acc_b, gyro_b in zip(time_s, accel_sensor, gyro_sensor, strict=True):
        t_ns = int(round(t_s * 1.0e9))
        acc_rows.append([t_ns, f"{acc_b[0]:.12f}", f"{acc_b[1]:.12f}", f"{acc_b[2]:.12f}", ""])
        gyro_rows.append([t_ns, f"{gyro_b[0]:.12f}", f"{gyro_b[1]:.12f}", f"{gyro_b[2]:.12f}", ""])

    gnss_rows: list[list[object]] = [[
        "TimestampGNSS [ns]",
        " UTC time [ms]",
        " Lat [deg]",
        " Lon [deg]",
        " Height [m]",
        " Speed [m/s]",
        " Heading [deg]",
        " Hor Acc [m]",
        " Vert Acc [m]",
        " Speed Acc [m/s]",
        " Heading Acc [deg]",
    ]]
    vel_rows: list[list[object]] = [[
        "TimeUS",
        "VelN",
        "VelE",
        "VelD",
        "VelAccN",
        "VelAccE",
        "VelAccD",
    ]]
    for t_s, gps_row in zip(gps_time_s, gps_rows, strict=True):
        lat_deg, lon_deg, height_m, v_n, v_e, v_d = gps_row.tolist()
        speed = float(math.hypot(v_n, v_e))
        heading_deg = math.degrees(math.atan2(v_e, v_n))
        if heading_deg < 0.0:
            heading_deg += 360.0
        t_ns = int(round(t_s * 1.0e9))
        t_us = int(round(t_s * 1.0e6))
        gnss_rows.append(
            [
                t_ns,
                int(round(t_s * 1.0e3)),
                f"{lat_deg:.12f}",
                f"{lon_deg:.12f}",
                f"{height_m:.6f}",
                f"{speed:.6f}",
                f"{heading_deg:.6f}",
                "3.000000",
                "5.000000",
                "0.100000",
                "5.000000",
                "",
            ]
        )
        vel_rows.append(
            [
                t_us,
                f"{v_n:.12f}",
                f"{v_e:.12f}",
                f"{v_d:.12f}",
                "0.100000",
                "0.100000",
                "0.100000",
            ]
        )

    write_semicolon_rows(output_dir / f"{prefix}_Acc.csv", acc_rows)
    write_semicolon_rows(output_dir / f"{prefix}_Gyro.csv", gyro_rows)
    write_semicolon_rows(output_dir / f"{prefix}_GNSS.csv", gnss_rows)
    write_semicolon_rows(output_dir / "gnss_velocity_meas.csv", vel_rows)
    return q_nominal, q_mis_truth


def build_plot(
    output_html: Path,
    title: str,
    series: list[dict[str, np.ndarray | str]],
    euler_truth_deg: np.ndarray,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Residual Mount Quaternion Error",
            "Residual Roll/Pitch/Yaw",
            "Quaternion Components q0/q1",
            "Quaternion Components q2/q3",
        ),
        vertical_spacing=0.12,
    )

    series_dashes = ["solid", "dash", "dot", "dashdot"]
    for series_idx, item in enumerate(series):
        dash = series_dashes[series_idx % len(series_dashes)]
        line = {"color": "#ffb000", "width": 2}
        if dash != "solid":
            line["dash"] = dash
        fig.add_trace(
            go.Scatter(
                x=item["time_s"],
                y=item["err_deg"],
                name=f"{item['name']} Geodesic Error [deg]",
                line=line,
            ),
            row=1,
            col=1,
        )

    colors = ["#4cc9f0", "#f72585", "#b8f200"]
    labels = ["Roll", "Pitch", "Yaw"]
    for idx, (label, color) in enumerate(zip(labels, colors, strict=True)):
        for series_idx, item in enumerate(series):
            dash = series_dashes[series_idx % len(series_dashes)]
            line = {"color": color, "width": 2}
            if dash != "solid":
                line["dash"] = dash
            fig.add_trace(
                go.Scatter(
                    x=item["time_s"],
                    y=item["euler_est_deg"][:, idx],
                    name=f"{item['name']} {label}",
                    line=line,
                ),
                row=1,
                col=2,
            )
        time_s = series[0]["time_s"]
        fig.add_trace(go.Scatter(x=time_s, y=np.full_like(time_s, euler_truth_deg[idx]), name=f"Truth {label}", line=dict(color=color, width=1, dash="dash")), row=1, col=2)

    q_truth = quat_from_rpy_deg(*euler_truth_deg.tolist())
    time_s = series[0]["time_s"]
    q_truth_series = np.tile(q_truth, (len(time_s), 1))
    for idx, color in enumerate(["#ffd166", "#06d6a0"], start=0):
        fig.add_trace(go.Scatter(x=time_s, y=q_truth_series[:, idx], name=f"Truth q{idx}", line=dict(color=color, width=1, dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_s, y=q_truth_series[:, idx + 2], name=f"Truth q{idx + 2}", line=dict(color=color, width=1, dash="dash")), row=2, col=2)

    for idx, color in enumerate(["#ef476f", "#118ab2"], start=0):
        for series_idx, item in enumerate(series):
            dash = series_dashes[series_idx % len(series_dashes)]
            line = {"color": color, "width": 2}
            if dash != "solid":
                line["dash"] = dash
            fig.add_trace(go.Scatter(x=item["time_s"], y=item["q_est"][:, idx], name=f"{item['name']} q{idx}", line=line), row=2, col=1)
            fig.add_trace(go.Scatter(x=item["time_s"], y=item["q_est"][:, idx + 2], name=f"{item['name']} q{idx + 2}", line=line), row=2, col=2)

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=900,
        width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
    )
    fig.update_xaxes(title_text="Time [s]")
    fig.update_yaxes(title_text="Error [deg]", row=1, col=1)
    fig.update_yaxes(title_text="Angle [deg]", row=1, col=2)
    fig.update_yaxes(title_text="Quaternion", row=2, col=1)
    fig.update_yaxes(title_text="Quaternion", row=2, col=2)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")


def build_mount_eval_series(
    name: str,
    results: dict[str, object],
    q_nominal: np.ndarray,
    q_mis_truth: np.ndarray,
) -> dict[str, np.ndarray | str]:
    hist = results["history"]
    time_s = np.asarray(hist["time_s"], dtype=float)
    q_cs_hist = np.asarray(hist["q_cs"], dtype=float)
    q_mis_est = np.vstack([quat_mul(q_cs, quat_conj(q_nominal)) for q_cs in q_cs_hist])
    err_deg = np.array([quat_angle_deg(q, q_mis_truth) for q in q_mis_est], dtype=float)
    euler_est_deg = np.vstack([quat_to_rpy_deg(q) for q in q_mis_est])
    return {
        "name": name,
        "time_s": time_s,
        "q_est": q_mis_est,
        "err_deg": err_deg,
        "euler_est_deg": euler_est_deg,
    }


def print_mount_eval_summary(
    item: dict[str, np.ndarray | str],
    mount_truth_deg: np.ndarray,
) -> None:
    name = str(item["name"])
    time_s = np.asarray(item["time_s"], dtype=float)
    err_deg = np.asarray(item["err_deg"], dtype=float)
    euler_est_deg = np.asarray(item["euler_est_deg"], dtype=float)
    tail_mask = time_s >= max(0.0, float(time_s[-1]) - 60.0)
    tail = err_deg[tail_mask]
    print(
        "{}_final_quat_err_deg={:.3f} {}_tail60_mean_deg={:.3f} {}_tail60_max_deg={:.3f}".format(
            name,
            float(err_deg[-1]),
            name,
            float(np.mean(tail)) if tail.size else float(err_deg[-1]),
            name,
            float(np.max(tail)) if tail.size else float(err_deg[-1]),
        )
    )
    final_euler = euler_est_deg[-1]
    print(
        "{}_final_residual_mount_deg=({:.3f}, {:.3f}, {:.3f}) truth=({:.3f}, {:.3f}, {:.3f})".format(
            name,
            float(final_euler[0]),
            float(final_euler[1]),
            float(final_euler[2]),
            float(mount_truth_deg[0]),
            float(mount_truth_deg[1]),
            float(mount_truth_deg[2]),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the loose filter on a gnss-ins-sim dataset and compare mount behavior.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--signal-source", choices=["ref", "meas"], default="meas")
    parser.add_argument("--backend", choices=["c", "python", "compare"], default="c")
    parser.add_argument("--mount-roll-deg", type=float, default=5.0)
    parser.add_argument("--mount-pitch-deg", type=float, default=-5.0)
    parser.add_argument("--mount-yaw-deg", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-html", type=Path, default=None)
    args = parser.parse_args()

    from ekf.ins_gnss_loose import run_c_loose_case, run_python_loose_case  # type: ignore[import-not-found]

    mount_truth_deg = np.array([args.mount_roll_deg, args.mount_pitch_deg, args.mount_yaw_deg], dtype=float)
    default_output_dir = args.data_dir / f"loose_case_{args.signal_source}"
    output_dir = args.output_dir or default_output_dir
    if args.output_html is not None:
        output_html = args.output_html
    elif args.backend == "c":
        output_html = output_dir / "loose_eval_dark.html"
    else:
        output_html = output_dir / f"loose_eval_{args.backend}_dark.html"

    q_nominal, q_mis_truth = build_loose_case(
        data_dir=args.data_dir,
        output_dir=output_dir,
        signal_source=args.signal_source,
        mount_truth_deg=mount_truth_deg,
        nominal_mount_deg=DEFAULT_NOMINAL_MOUNT_DEG,
    )

    series: list[dict[str, np.ndarray | str]] = []
    if args.backend in {"c", "compare"}:
        c_results = run_c_loose_case(output_dir, OAN_ROOT, collect_history=True)
        series.append(build_mount_eval_series("c", c_results, q_nominal, q_mis_truth))
    if args.backend in {"python", "compare"}:
        py_results = run_python_loose_case(output_dir, OAN_ROOT)
        series.append(build_mount_eval_series("python", py_results, q_nominal, q_mis_truth))

    build_plot(
        output_html=output_html,
        title=f"Loose Mount Evaluation: {args.data_dir.name} ({args.signal_source}, {args.backend})",
        series=series,
        euler_truth_deg=mount_truth_deg,
    )

    print(f"input={args.data_dir}")
    print(f"signal_source={args.signal_source}")
    print(f"backend={args.backend}")
    for item in series:
        print_mount_eval_summary(item, mount_truth_deg)
    print(f"html={output_html}")


if __name__ == "__main__":
    main()
