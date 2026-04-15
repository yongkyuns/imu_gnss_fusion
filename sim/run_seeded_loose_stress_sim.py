#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import ctypes
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # pragma: no cover
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover
    go = None
    make_subplots = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "ekf") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ekf"))

from ekf.ins_gnss_loose import (  # noqa: E402
    CLoose,
    CLooseImuDelta,
    _attach_gnss_velocity,
    _default_noise,
    _ensure_oan_imports,
    _load_c_lib,
    _load_gnss_velocity_map,
    _load_truth_nav,
    _load_truth_states,
)


DEFAULT_OAN_ROOT = Path("/Users/ykshin/Dev/me/open-aided-navigation")
DEFAULT_GNSS_INS_SIM = Path("/Users/ykshin/Dev/me/gnss-ins-sim")
DEFAULT_MOTION_DEF = REPO_ROOT / "sim" / "motion_profiles" / "seeded_highspeed_straight_10min.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sim" / "generated_seeded_loose_stress"


@dataclass(frozen=True)
class StressCase:
    label: str
    seed_mount_error_deg: tuple[float, float, float]
    gps_vd_bias_mps: float
    gps_vd_bias_duration_s: float
    vel_std_scale: float
    accel_z_bias_mps2: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic gnss-ins-sim case and replay the seeded loose C filter in the same "
            "pre-rotated-IMU mode as the real visualizer, sweeping the ingredients suspected of causing "
            "the early wrong-mount branch."
        )
    )
    parser.add_argument("--oan-root", type=Path, default=DEFAULT_OAN_ROOT)
    parser.add_argument("--gnss-ins-sim-root", type=Path, default=DEFAULT_GNSS_INS_SIM)
    parser.add_argument("--motion-def", type=Path, default=DEFAULT_MOTION_DEF)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "dataset")
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_DIR / "seeded_loose_stress_dark.html")
    parser.add_argument("--init-time-s", type=float, default=100.0)
    parser.add_argument("--imu-freq-hz", type=float, default=100.0)
    parser.add_argument("--gnss-freq-hz", type=float, default=10.0)
    parser.add_argument("--gps-horizontal-std", type=float, default=3.0)
    parser.add_argument("--gps-vertical-std", type=float, default=5.0)
    parser.add_argument("--jump-threshold-deg", type=float, default=5.0)
    parser.add_argument("--truth-misalignment-deg", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--skip-default-cases",
        action="store_true",
        help="Skip the fixed hand-picked stress cases and run only Monte Carlo, if requested.",
    )
    parser.add_argument(
        "--monte-carlo-runs",
        type=int,
        default=0,
        help="Number of randomized stress runs to execute against the same generated dataset.",
    )
    parser.add_argument(
        "--monte-carlo-seed",
        type=int,
        default=1234,
        help="RNG seed for Monte Carlo stress sampling.",
    )
    parser.add_argument("--mc-seed-roll-range-deg", type=float, nargs=2, default=[-4.0, 4.0])
    parser.add_argument("--mc-seed-pitch-range-deg", type=float, nargs=2, default=[-4.0, 4.0])
    parser.add_argument("--mc-seed-yaw-range-deg", type=float, nargs=2, default=[-6.0, 6.0])
    parser.add_argument("--mc-gps-vd-bias-range-mps", type=float, nargs=2, default=[-2.5, 0.0])
    parser.add_argument("--mc-gps-vd-bias-duration-range-s", type=float, nargs=2, default=[0.0, 20.0])
    parser.add_argument("--mc-vel-std-scale-range", type=float, nargs=2, default=[0.1, 1.0])
    parser.add_argument("--mc-accel-z-bias-range-mps2", type=float, nargs=2, default=[-0.15, 0.15])
    parser.add_argument(
        "--mc-mount-err-threshold-deg",
        type=float,
        default=5.0,
        help="Classify a run as wrong-basin if the final mount error norm exceeds this threshold.",
    )
    parser.add_argument(
        "--mc-car-pitch-err-threshold-deg",
        type=float,
        default=3.0,
        help="Classify a run as wrong-basin if the final car pitch error exceeds this threshold.",
    )
    parser.add_argument(
        "--mc-summary-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "seeded_loose_stress_mc_summary.csv",
        help="Per-run Monte Carlo summary CSV output path.",
    )
    parser.add_argument(
        "--mc-output-html",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "seeded_loose_stress_mc_dark.html",
        help="Plotly dark HTML dashboard for Monte Carlo dispersion and observability plots.",
    )
    return parser.parse_args()


def default_cases() -> list[StressCase]:
    return [
        StressCase("baseline", (0.0, 0.0, 0.0), 0.0, 0.0, 1.0),
        StressCase("seed_only", (3.0, -2.0, 3.0), 0.0, 0.0, 1.0),
        StressCase("vd_bias_only", (0.0, 0.0, 0.0), -2.0, 12.0, 0.3),
        StressCase("combined", (3.0, -2.0, 3.0), -1.5, 12.0, 0.3),
        StressCase("combined_aggressive", (3.0, -2.0, 3.0), -2.0, 20.0, 0.1),
    ]


def sample_uniform(rng: np.random.Generator, bounds: tuple[float, float] | list[float]) -> float:
    lo = float(min(bounds[0], bounds[1]))
    hi = float(max(bounds[0], bounds[1]))
    return float(rng.uniform(lo, hi))


def monte_carlo_cases(args: argparse.Namespace) -> list[StressCase]:
    rng = np.random.default_rng(args.monte_carlo_seed)
    cases: list[StressCase] = []
    for idx in range(args.monte_carlo_runs):
        cases.append(
            StressCase(
                label=f"mc_{idx:04d}",
                seed_mount_error_deg=(
                    sample_uniform(rng, args.mc_seed_roll_range_deg),
                    sample_uniform(rng, args.mc_seed_pitch_range_deg),
                    sample_uniform(rng, args.mc_seed_yaw_range_deg),
                ),
                gps_vd_bias_mps=sample_uniform(rng, args.mc_gps_vd_bias_range_mps),
                gps_vd_bias_duration_s=sample_uniform(rng, args.mc_gps_vd_bias_duration_range_s),
                vel_std_scale=sample_uniform(rng, args.mc_vel_std_scale_range),
                accel_z_bias_mps2=sample_uniform(rng, args.mc_accel_z_bias_range_mps2),
            )
        )
    return cases


def yaw_quat(yaw_rad: float) -> np.ndarray:
    half = 0.5 * yaw_rad
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def wrap_deg180(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def rotmat_from_quat(mods, q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q
    return np.array(
        [
            [1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3, 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
            [2.0 * (q1 * q2 + q0 * q3), 1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3, 2.0 * (q2 * q3 - q0 * q1)],
            [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2],
        ],
        dtype=float,
    )


def nominal_mount_quat(mods) -> np.ndarray:
    return np.asarray(
        mods["euler_to_quat"](
            mods["ParamsInsGnssFilterLoose"].ROLL_IMU2CAR,
            mods["ParamsInsGnssFilterLoose"].PITCH_IMU2CAR,
            mods["ParamsInsGnssFilterLoose"].YAW_IMU2CAR,
        ),
        dtype=float,
    )


def qcs_truth_from_truth_states(mods, truth_states: dict[str, np.ndarray]) -> np.ndarray:
    q_nom = nominal_mount_quat(mods)
    mis_rad = np.deg2rad(truth_states["misalignment_deg"])
    q_mis = np.asarray(mods["euler_to_quat"](*mis_rad), dtype=float)
    q_true = np.asarray(mods["quat_mult"](q_mis, q_nom), dtype=float)
    return q_true / np.linalg.norm(q_true)


def initial_yaw_from_gnss(gnss_sample) -> float:
    if getattr(gnss_sample, "vel_valid", False):
        speed_h = float(np.hypot(gnss_sample.v_n[0], gnss_sample.v_n[1]))
        if speed_h >= 1.0:
            return float(math.atan2(gnss_sample.v_n[1], gnss_sample.v_n[0]))
    if gnss_sample.speed > 0.0:
        return float(gnss_sample.heading)
    return 0.0


def build_default_p_diag(gnss_sample) -> np.ndarray:
    att_sigma_rad = math.radians(2.0)
    att_var = att_sigma_rad * att_sigma_rad
    vel_std = 0.2
    if getattr(gnss_sample, "vel_valid", False):
        vel_std = max(vel_std, float(np.max(gnss_sample.v_acc_n)))
    vel_var = vel_std * vel_std
    pos_n = max(float(gnss_sample.h_acc), 0.5)
    pos_e = max(float(gnss_sample.h_acc), 0.5)
    pos_d = max(float(gnss_sample.v_acc), 0.5)
    gyro_bias_sigma = math.radians(0.125)
    accel_bias_sigma = 0.075
    accel_scale_sigma = 0.02
    gyro_scale_sigma = 0.02
    p_diag = np.zeros(24, dtype=np.float32)
    p_diag[0] = pos_n * pos_n
    p_diag[1] = pos_e * pos_e
    p_diag[2] = pos_d * pos_d
    p_diag[3:6] = vel_var
    p_diag[6:9] = att_var
    p_diag[9:12] = accel_bias_sigma * accel_bias_sigma
    p_diag[12:15] = gyro_bias_sigma * gyro_bias_sigma
    p_diag[15:18] = accel_scale_sigma * accel_scale_sigma
    p_diag[18:21] = gyro_scale_sigma * gyro_scale_sigma
    p_diag[21:24] = att_var
    return p_diag


def set_loose_reference_state(
    loose: CLoose,
    q_es: np.ndarray,
    pos_ecef_m: np.ndarray,
    vel_ecef_mps: np.ndarray,
    q_cs: np.ndarray,
    p_diag: np.ndarray,
) -> None:
    loose.nominal.q0 = float(q_es[0])
    loose.nominal.q1 = float(q_es[1])
    loose.nominal.q2 = float(q_es[2])
    loose.nominal.q3 = float(q_es[3])
    loose.nominal.vn = float(vel_ecef_mps[0])
    loose.nominal.ve = float(vel_ecef_mps[1])
    loose.nominal.vd = float(vel_ecef_mps[2])
    loose.nominal.pn = float(pos_ecef_m[0])
    loose.nominal.pe = float(pos_ecef_m[1])
    loose.nominal.pd = float(pos_ecef_m[2])
    loose.nominal.bgx = 0.0
    loose.nominal.bgy = 0.0
    loose.nominal.bgz = 0.0
    loose.nominal.bax = 0.0
    loose.nominal.bay = 0.0
    loose.nominal.baz = 0.0
    loose.nominal.sgx = 1.0
    loose.nominal.sgy = 1.0
    loose.nominal.sgz = 1.0
    loose.nominal.sax = 1.0
    loose.nominal.say = 1.0
    loose.nominal.saz = 1.0
    loose.nominal.qcs0 = float(q_cs[0])
    loose.nominal.qcs1 = float(q_cs[1])
    loose.nominal.qcs2 = float(q_cs[2])
    loose.nominal.qcs3 = float(q_cs[3])
    for i in range(3):
        loose.pos_e64[i] = float(pos_ecef_m[i])
    for i in range(4):
        loose.qcs64[i] = float(q_cs[i])
    for i in range(24):
        for j in range(24):
            value = float(p_diag[i]) if i == j else 0.0
            loose.p[i][j] = value
            loose.p64[i][j] = value


def generate_dataset(args: argparse.Namespace) -> Path:
    args.case_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.oan_root))
    from open_aided_navigation.demos.generate_loose_ins_gnss_sim_case import generate_case

    generate_case(
        output_dir=args.case_dir,
        sim_root=args.gnss_ins_sim_root,
        motion_def=args.motion_def,
        imu_freq_hz=args.imu_freq_hz,
        gnss_freq_hz=args.gnss_freq_hz,
        imu_accuracy="mid-accuracy",
        gps_horizontal_std=args.gps_horizontal_std,
        gps_vertical_std=args.gps_vertical_std,
        misalignment_deg=np.asarray(args.truth_misalignment_deg, dtype=float),
        accel_bias_truth=np.array([0.03, -0.02, 0.015], dtype=float),
        gyro_bias_dps_truth=np.array([0.08, -0.05, 0.06], dtype=float),
        accel_scale_ppm=np.array([800.0, -600.0, 500.0], dtype=float),
        gyro_scale_ppm=np.array([700.0, -500.0, 400.0], dtype=float),
        seed=args.seed,
    )
    return args.case_dir


def run_seeded_case(
    input_dir: Path,
    oan_root: Path,
    case: StressCase,
    init_time_s: float,
    jump_threshold_deg: float,
) -> dict[str, np.ndarray | float | str]:
    mods = _ensure_oan_imports(oan_root)
    gyro_data, accel_data = mods["import_imu_data"](input_dir)
    gnss_data = mods["import_gnss_data"](input_dir)
    _attach_gnss_velocity(gnss_data, _load_gnss_velocity_map(input_dir))
    truth_nav = _load_truth_nav(input_dir)
    truth_states = _load_truth_states(input_dir)

    q_cs_true = qcs_truth_from_truth_states(mods, truth_states)
    q_seed_err = np.asarray(mods["euler_to_quat"](*np.deg2rad(np.asarray(case.seed_mount_error_deg, dtype=float))), dtype=float)
    q_cs_seed = np.asarray(mods["quat_mult"](q_seed_err, q_cs_true), dtype=float)
    q_cs_seed = q_cs_seed / np.linalg.norm(q_cs_seed)
    c_seed = rotmat_from_quat(mods, q_cs_seed)

    init_gyro_index = min(
        range(len(gyro_data)),
        key=lambda idx: abs(1e-6 * gyro_data[idx].ttag - init_time_s),
    )
    init_ttag = gyro_data[init_gyro_index].ttag

    gnss_init_index = 0
    while gnss_init_index + 1 < len(gnss_data) and gnss_data[gnss_init_index + 1].ttag <= init_ttag:
        gnss_init_index += 1
    gnss_init = gnss_data[gnss_init_index]

    yaw_init = initial_yaw_from_gnss(gnss_init)
    q_ne = np.asarray(mods["quat_ecef_to_ned"](gnss_init.lat, gnss_init.lon), dtype=float)
    q_es = np.asarray(mods["quat_mult"](mods["quat_conjugate"](q_ne), yaw_quat(yaw_init)), dtype=float)
    q_es = q_es / np.linalg.norm(q_es)
    pos_ecef = np.asarray(mods["llh_to_ecef"](gnss_init.lat, gnss_init.lon, gnss_init.height, mods["WGS84"]), dtype=float)
    vel_ecef = np.asarray(mods["dcm_ecef_to_ned"](gnss_init.lat, gnss_init.lon).T @ gnss_init.v_n, dtype=float)
    p_diag = build_default_p_diag(gnss_init)

    loose = CLoose()
    c_lib = _load_c_lib()
    c_lib.sf_loose_init(ctypes.byref(loose), None, ctypes.byref(_default_noise()))
    set_loose_reference_state(loose, q_es, pos_ecef, vel_ecef, np.array([1.0, 0.0, 0.0, 0.0], dtype=float), p_diag)

    gnss_index = gnss_init_index
    last_used_gnss_ttag = gnss_init.ttag
    imu_timeout_us = mods["IMU_TIMEOUT_US"]
    nominal_mount = nominal_mount_quat(mods)
    true_car_roll = np.interp(init_ttag, truth_nav["TimeUS"], truth_nav["RollCarDeg"])

    history: dict[str, list[float]] = {
        "time_s": [],
        "mount_roll_deg": [],
        "mount_pitch_deg": [],
        "mount_yaw_deg": [],
        "mount_err_roll_deg": [],
        "mount_err_pitch_deg": [],
        "mount_err_yaw_deg": [],
        "car_roll_deg": [],
        "car_pitch_deg": [],
        "car_yaw_deg": [],
        "car_pitch_truth_deg": [],
        "car_pitch_err_deg": [],
        "accel_bias_z": [],
        "vel_down_ecef": [],
        "vel_down_car": [],
        "gps_vd_used": [],
    }
    first_jump_time_s: float | None = None
    first_jump_mag_deg: float = 0.0
    prev_mount: np.ndarray | None = None

    for gyro_index in range(init_gyro_index + 1, len(gyro_data)):
        prev_gyro = gyro_data[gyro_index - 1]
        curr_gyro = gyro_data[gyro_index]
        dt_s = 1e-6 * float(curr_gyro.ttag - prev_gyro.ttag)
        accel_prev = c_seed @ accel_data[gyro_index - 1].f_b.astype(float)
        accel_curr = c_seed @ accel_data[gyro_index].f_b.astype(float)
        accel_prev[2] += case.accel_z_bias_mps2
        accel_curr[2] += case.accel_z_bias_mps2
        gyro_prev = c_seed @ prev_gyro.omega_is.astype(float)
        gyro_curr = c_seed @ curr_gyro.omega_is.astype(float)
        imu = CLooseImuDelta(
            dax_1=np.float32(gyro_prev[0] * dt_s),
            day_1=np.float32(gyro_prev[1] * dt_s),
            daz_1=np.float32(gyro_prev[2] * dt_s),
            dvx_1=np.float32(accel_prev[0] * dt_s),
            dvy_1=np.float32(accel_prev[1] * dt_s),
            dvz_1=np.float32(accel_prev[2] * dt_s),
            dax_2=np.float32(gyro_curr[0] * dt_s),
            day_2=np.float32(gyro_curr[1] * dt_s),
            daz_2=np.float32(gyro_curr[2] * dt_s),
            dvx_2=np.float32(accel_curr[0] * dt_s),
            dvy_2=np.float32(accel_curr[1] * dt_s),
            dvz_2=np.float32(accel_curr[2] * dt_s),
            dt=np.float32(dt_s),
        )
        c_lib.sf_loose_predict(ctypes.byref(loose), ctypes.byref(imu))

        while gnss_index + 1 < len(gnss_data) and gnss_data[gnss_index + 1].ttag <= curr_gyro.ttag:
            gnss_index += 1

        pos_ptr = None
        vel_ptr = None
        vel_std_ptr = None
        h_acc_m = 0.0
        gps_vd_used = 0.0
        if gnss_index < len(gnss_data):
            gnss = gnss_data[gnss_index]
            d_ttag = curr_gyro.ttag - gnss.ttag
            if d_ttag < imu_timeout_us // 2 and gnss.ttag != last_used_gnss_ttag:
                pos_ecef = np.asarray(mods["llh_to_ecef"](gnss.lat, gnss.lon, gnss.height, mods["WGS84"]), dtype=np.float64)
                pos_ptr = (ctypes.c_double * 3)(*pos_ecef.tolist())
                vel_n = gnss.v_n.astype(float).copy()
                if case.gps_vd_bias_duration_s > 0.0:
                    rel_gnss_s = 1e-6 * float(gnss.ttag - init_ttag)
                    if 0.0 <= rel_gnss_s <= case.gps_vd_bias_duration_s:
                        vel_n[2] += case.gps_vd_bias_mps
                gps_vd_used = float(vel_n[2])
                vel_ecef = np.asarray(mods["dcm_ecef_to_ned"](gnss.lat, gnss.lon).T @ vel_n, dtype=np.float32)
                vel_std = np.asarray(gnss.v_acc_n.astype(float) * case.vel_std_scale, dtype=np.float32)
                vel_ptr = (ctypes.c_float * 3)(*vel_ecef.tolist())
                vel_std_ptr = (ctypes.c_float * 3)(*vel_std.tolist())
                h_acc_m = float(gnss.h_acc)
                d_ttag_s = 1e-6 * float(curr_gyro.ttag - last_used_gnss_ttag)
                if d_ttag_s == 0.0 or d_ttag_s >= 1.0:
                    d_ttag_s = 1.0
                last_used_gnss_ttag = gnss.ttag
            else:
                d_ttag_s = 1.0
        else:
            d_ttag_s = 1.0

        gyro_ptr = (ctypes.c_float * 3)(*gyro_curr.astype(np.float32).tolist())
        accel_ptr = (ctypes.c_float * 3)(*accel_curr.astype(np.float32).tolist())
        c_lib.sf_loose_fuse_reference_batch_full(
            ctypes.byref(loose),
            pos_ptr,
            vel_ptr,
            ctypes.c_float(h_acc_m),
            vel_std_ptr,
            ctypes.c_float(d_ttag_s),
            gyro_ptr,
            accel_ptr,
            ctypes.c_float(dt_s),
        )

        q_es_hat = np.array([loose.nominal.q0, loose.nominal.q1, loose.nominal.q2, loose.nominal.q3], dtype=float)
        q_cs_resid = np.array([loose.nominal.qcs0, loose.nominal.qcs1, loose.nominal.qcs2, loose.nominal.qcs3], dtype=float)
        q_cs_full = np.asarray(mods["quat_mult"](q_cs_resid, q_cs_seed), dtype=float)
        q_cs_full /= np.linalg.norm(q_cs_full)
        lat_hat, lon_hat, _ = mods["ecef_to_llh"](np.array([loose.pos_e64[0], loose.pos_e64[1], loose.pos_e64[2]], dtype=float), mods["WGS84"])
        q_ne_hat = np.asarray(mods["quat_ecef_to_ned"](lat_hat, lon_hat), dtype=float)
        q_ns_hat = np.asarray(mods["quat_mult"](q_ne_hat, q_es_hat), dtype=float)
        q_nc_hat = np.asarray(mods["quat_mult"](q_ns_hat, mods["quat_conjugate"](q_cs_resid)), dtype=float)
        car_euler_deg = np.rad2deg(np.asarray(mods["quat_to_euler"](q_nc_hat), dtype=float))
        mount_err_deg = np.rad2deg(
            np.asarray(mods["quat_to_euler"](np.asarray(mods["quat_mult"](q_cs_full, mods["quat_conjugate"](q_cs_true)), dtype=float)), dtype=float)
        )
        full_mount_mis_deg = np.rad2deg(
            np.asarray(mods["quat_to_euler"](np.asarray(mods["quat_mult"](q_cs_full, mods["quat_conjugate"](nominal_mount)), dtype=float)), dtype=float)
        )

        vel_e = np.array([loose.nominal.vn, loose.nominal.ve, loose.nominal.vd], dtype=float)
        c_ce = rotmat_from_quat(mods, q_cs_resid) @ rotmat_from_quat(mods, q_es_hat).T
        vel_car = c_ce @ vel_e
        t_rel_s = 1e-6 * float(curr_gyro.ttag - init_ttag)

        history["time_s"].append(t_rel_s)
        history["mount_roll_deg"].append(float(full_mount_mis_deg[0]))
        history["mount_pitch_deg"].append(float(full_mount_mis_deg[1]))
        history["mount_yaw_deg"].append(float(full_mount_mis_deg[2]))
        history["mount_err_roll_deg"].append(float(mount_err_deg[0]))
        history["mount_err_pitch_deg"].append(float(mount_err_deg[1]))
        history["mount_err_yaw_deg"].append(float(mount_err_deg[2]))
        history["car_roll_deg"].append(float(car_euler_deg[0]))
        history["car_pitch_deg"].append(float(car_euler_deg[1]))
        history["car_yaw_deg"].append(float(car_euler_deg[2]))
        truth_car_pitch = float(np.interp(curr_gyro.ttag, truth_nav["TimeUS"], truth_nav["PitchCarDeg"]))
        history["car_pitch_truth_deg"].append(truth_car_pitch)
        history["car_pitch_err_deg"].append(float(car_euler_deg[1] - truth_car_pitch))
        history["accel_bias_z"].append(float(loose.nominal.baz))
        history["vel_down_ecef"].append(float(vel_e[2]))
        history["vel_down_car"].append(float(vel_car[2]))
        history["gps_vd_used"].append(float(gps_vd_used))

        if prev_mount is not None:
            step = np.linalg.norm(wrap_deg180(full_mount_mis_deg - prev_mount))
            if first_jump_time_s is None and step >= jump_threshold_deg:
                first_jump_time_s = t_rel_s
                first_jump_mag_deg = float(step)
        prev_mount = full_mount_mis_deg.copy()

    return {
        "label": case.label,
        "history": {k: np.asarray(v, dtype=float) for k, v in history.items()},
        "seed_mount_error_deg": np.asarray(case.seed_mount_error_deg, dtype=float),
        "gps_vd_bias_mps": float(case.gps_vd_bias_mps),
        "gps_vd_bias_duration_s": float(case.gps_vd_bias_duration_s),
        "vel_std_scale": float(case.vel_std_scale),
        "accel_z_bias_mps2": float(case.accel_z_bias_mps2),
        "first_jump_time_s": float("nan") if first_jump_time_s is None else first_jump_time_s,
        "first_jump_mag_deg": first_jump_mag_deg,
        "final_mount_deg": np.array(
            [history["mount_roll_deg"][-1], history["mount_pitch_deg"][-1], history["mount_yaw_deg"][-1]],
            dtype=float,
        ),
        "final_mount_err_deg": np.array(
            [history["mount_err_roll_deg"][-1], history["mount_err_pitch_deg"][-1], history["mount_err_yaw_deg"][-1]],
            dtype=float,
        ),
        "final_mount_err_norm_deg": float(
            np.linalg.norm(
                wrap_deg180(
                    np.array(
                        [
                            history["mount_err_roll_deg"][-1],
                            history["mount_err_pitch_deg"][-1],
                            history["mount_err_yaw_deg"][-1],
                        ],
                        dtype=float,
                    )
                )
            )
        ),
        "final_car_pitch_deg": float(history["car_pitch_deg"][-1]),
        "final_car_pitch_err_deg": float(history["car_pitch_err_deg"][-1]),
        "final_accel_bias_z": float(history["accel_bias_z"][-1]),
        "true_car_roll_deg_at_init": float(true_car_roll),
        "truth_mount_deg": truth_states["misalignment_deg"].astype(float),
    }


def write_plot(results: list[dict[str, np.ndarray | float | str]], output_html: Path) -> None:
    if go is None or make_subplots is None:
        print("plotly_missing=1")
        return
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=[
            "Mount Roll", "Mount Yaw",
            "Car Pitch", "Accel Bias Z",
            "Car Vertical Velocity", "GPS Down Velocity Used",
            "Mount Pitch", "Car Roll",
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )
    colors = {
        "baseline": "#4cc9f0",
        "seed_only": "#f72585",
        "vd_bias_only": "#fca311",
        "combined": "#80ed99",
        "combined_overtrust": "#c77dff",
    }
    for result in results:
        label = str(result["label"])
        hist = result["history"]
        t = hist["time_s"]
        color = colors.get(label, "#dddddd")
        for row, col, key in [
            (1, 1, "mount_roll_deg"),
            (1, 2, "mount_yaw_deg"),
            (2, 1, "car_pitch_deg"),
            (2, 2, "accel_bias_z"),
            (3, 1, "vel_down_car"),
            (3, 2, "gps_vd_used"),
            (4, 1, "mount_pitch_deg"),
            (4, 2, "car_roll_deg"),
        ]:
            fig.add_trace(go.Scatter(x=t, y=hist[key], name=f"{label} {key}", line=dict(color=color, width=2)), row=row, col=col)
    fig.update_layout(
        template="plotly_dark",
        title="Seeded Loose Stress Replay",
        height=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html)


def monte_carlo_row(result: dict[str, np.ndarray | float | str], args: argparse.Namespace) -> dict[str, float | str]:
    final_mount = np.asarray(result["final_mount_deg"], dtype=float)
    final_mount_err = np.asarray(result["final_mount_err_deg"], dtype=float)
    final_mount_err_norm_deg = float(result["final_mount_err_norm_deg"])
    final_car_pitch_err_deg = float(result["final_car_pitch_err_deg"])
    first_jump_time_s = float(result["first_jump_time_s"])
    has_jump = not math.isnan(first_jump_time_s)
    wrong_basin = (
        final_mount_err_norm_deg >= args.mc_mount_err_threshold_deg
        or abs(final_car_pitch_err_deg) >= args.mc_car_pitch_err_threshold_deg
    )
    return {
        "label": str(result["label"]),
        "seed_roll_deg": float(np.asarray(result["seed_mount_error_deg"], dtype=float)[0]),
        "seed_pitch_deg": float(np.asarray(result["seed_mount_error_deg"], dtype=float)[1]),
        "seed_yaw_deg": float(np.asarray(result["seed_mount_error_deg"], dtype=float)[2]),
        "gps_vd_bias_mps": float(result["gps_vd_bias_mps"]),
        "gps_vd_bias_duration_s": float(result["gps_vd_bias_duration_s"]),
        "vel_std_scale": float(result["vel_std_scale"]),
        "accel_z_bias_mps2": float(result["accel_z_bias_mps2"]),
        "first_jump_time_s": first_jump_time_s,
        "first_jump_mag_deg": float(result["first_jump_mag_deg"]),
        "has_jump": int(has_jump),
        "final_mount_roll_deg": float(final_mount[0]),
        "final_mount_pitch_deg": float(final_mount[1]),
        "final_mount_yaw_deg": float(final_mount[2]),
        "final_mount_err_roll_deg": float(final_mount_err[0]),
        "final_mount_err_pitch_deg": float(final_mount_err[1]),
        "final_mount_err_yaw_deg": float(final_mount_err[2]),
        "final_mount_err_norm_deg": final_mount_err_norm_deg,
        "final_car_pitch_deg": float(result["final_car_pitch_deg"]),
        "final_car_pitch_err_deg": final_car_pitch_err_deg,
        "final_accel_bias_z": float(result["final_accel_bias_z"]),
        "wrong_basin": int(wrong_basin),
    }


def write_monte_carlo_summary(rows: list[dict[str, float | str]], output_csv: Path) -> None:
    if not rows:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def write_monte_carlo_plot(rows: list[dict[str, float | str]], output_html: Path) -> None:
    if not rows or go is None or make_subplots is None:
        return

    seed_roll = np.asarray([float(row["seed_roll_deg"]) for row in rows], dtype=float)
    seed_pitch = np.asarray([float(row["seed_pitch_deg"]) for row in rows], dtype=float)
    seed_yaw = np.asarray([float(row["seed_yaw_deg"]) for row in rows], dtype=float)
    seed_norm = np.sqrt(seed_roll * seed_roll + seed_pitch * seed_pitch + seed_yaw * seed_yaw)
    gps_vd_bias = np.asarray([float(row["gps_vd_bias_mps"]) for row in rows], dtype=float)
    gps_vd_bias_duration = np.asarray([float(row["gps_vd_bias_duration_s"]) for row in rows], dtype=float)
    vel_std_scale = np.asarray([float(row["vel_std_scale"]) for row in rows], dtype=float)
    accel_z_bias = np.asarray([float(row["accel_z_bias_mps2"]) for row in rows], dtype=float)
    jump_time = np.asarray([float(row["first_jump_time_s"]) for row in rows], dtype=float)
    jump_mag = np.asarray([float(row["first_jump_mag_deg"]) for row in rows], dtype=float)
    has_jump = np.asarray([float(row["has_jump"]) for row in rows], dtype=float)
    wrong_basin = np.asarray([float(row["wrong_basin"]) for row in rows], dtype=float)
    final_mount_err_norm = np.asarray([float(row["final_mount_err_norm_deg"]) for row in rows], dtype=float)
    final_car_pitch_err = np.asarray([float(row["final_car_pitch_err_deg"]) for row in rows], dtype=float)
    final_accel_bias_z = np.asarray([float(row["final_accel_bias_z"]) for row in rows], dtype=float)
    labels = [str(row["label"]) for row in rows]
    jump_time_plot = np.where(np.isnan(jump_time), -1.0, jump_time)

    corr_inputs = {
        "seed_roll_deg": seed_roll,
        "seed_pitch_deg": seed_pitch,
        "seed_yaw_deg": seed_yaw,
        "seed_norm_deg": seed_norm,
        "gps_vd_bias_mps": gps_vd_bias,
        "gps_vd_bias_duration_s": gps_vd_bias_duration,
        "vel_std_scale": vel_std_scale,
        "accel_z_bias_mps2": accel_z_bias,
    }
    corr_to_mount = {name: safe_corrcoef(values, final_mount_err_norm) for name, values in corr_inputs.items()}
    corr_to_pitch = {name: safe_corrcoef(values, np.abs(final_car_pitch_err)) for name, values in corr_inputs.items()}

    colorscale = [
        [0.0, "#4cc9f0"],
        [0.5, "#fca311"],
        [1.0, "#f72585"],
    ]
    hover_text = [
        (
            f"{label}<br>"
            f"seed=[{sr:.2f}, {sp:.2f}, {sy:.2f}] deg<br>"
            f"gps_vd_bias={vd:.2f} m/s for {dur:.2f} s<br>"
            f"vel_std_scale={vss:.2f}<br>"
            f"accel_z_bias={az:.3f} m/s^2<br>"
            f"mount_err_norm={men:.2f} deg<br>"
            f"car_pitch_err={cpe:.2f} deg<br>"
            f"jump_time={'none' if np.isnan(jt) else f'{jt:.2f} s'}"
        )
        for label, sr, sp, sy, vd, dur, vss, az, men, cpe, jt in zip(
            labels,
            seed_roll,
            seed_pitch,
            seed_yaw,
            gps_vd_bias,
            gps_vd_bias_duration,
            vel_std_scale,
            accel_z_bias,
            final_mount_err_norm,
            final_car_pitch_err,
            jump_time,
            strict=False,
        )
    ]

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "Seed Error Dispersion vs Final Mount Error",
            "GNSS Down-Velocity Bias vs Car Pitch Error",
            "Velocity Trust vs Final Mount Error",
            "Jump Timing Dispersion",
            "Input Correlation to Final Mount Error",
            "Input Correlation to |Car Pitch Error|",
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Scatter(
            x=seed_norm,
            y=final_mount_err_norm,
            mode="markers",
            name="seed_norm vs mount_err",
            marker=dict(
                size=10,
                color=wrong_basin,
                colorscale=colorscale,
                cmin=0,
                cmax=1,
                line=dict(color="#d9d9d9", width=0.5),
                colorbar=dict(title="wrong_basin", x=0.46),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=gps_vd_bias,
            y=final_car_pitch_err,
            mode="markers",
            name="gps_vd_bias vs pitch_err",
            marker=dict(
                size=np.clip(6.0 + 0.8 * gps_vd_bias_duration, 6.0, 24.0),
                color=vel_std_scale,
                colorscale="Viridis",
                line=dict(color="#d9d9d9", width=0.5),
                colorbar=dict(title="vel_std_scale", x=1.02),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=vel_std_scale,
            y=final_mount_err_norm,
            mode="markers",
            name="vel_trust vs mount_err",
            marker=dict(
                size=10,
                color=np.abs(gps_vd_bias),
                colorscale="Turbo",
                line=dict(color="#d9d9d9", width=0.5),
                colorbar=dict(title="|gps_vd_bias|", x=0.46, y=0.36),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=jump_time_plot,
            y=jump_mag,
            mode="markers",
            name="jump timing",
            marker=dict(
                size=10,
                color=final_mount_err_norm,
                colorscale="Magma",
                line=dict(color="#d9d9d9", width=0.5),
                colorbar=dict(title="mount_err_norm", x=1.02, y=0.36),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=list(corr_to_mount.keys()),
            y=list(corr_to_mount.values()),
            name="corr -> mount_err",
            marker=dict(color="#80ed99"),
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=list(corr_to_pitch.keys()),
            y=list(corr_to_pitch.values()),
            name="corr -> |pitch_err|",
            marker=dict(color="#ff9f1c"),
        ),
        row=3,
        col=2,
    )

    fig.update_xaxes(title_text="seed error norm [deg]", row=1, col=1)
    fig.update_yaxes(title_text="final mount err norm [deg]", row=1, col=1)
    fig.update_xaxes(title_text="gps v_d bias [m/s]", row=1, col=2)
    fig.update_yaxes(title_text="final car pitch err [deg]", row=1, col=2)
    fig.update_xaxes(title_text="vel std scale", row=2, col=1)
    fig.update_yaxes(title_text="final mount err norm [deg]", row=2, col=1)
    fig.update_xaxes(title_text="first jump time [s], -1 means none", row=2, col=2)
    fig.update_yaxes(title_text="first jump magnitude [deg]", row=2, col=2)
    fig.update_xaxes(title_text="sampled input", tickangle=-30, row=3, col=1)
    fig.update_yaxes(title_text="Pearson r", range=[-1.0, 1.0], row=3, col=1)
    fig.update_xaxes(title_text="sampled input", tickangle=-30, row=3, col=2)
    fig.update_yaxes(title_text="Pearson r", range=[-1.0, 1.0], row=3, col=2)

    fig.update_layout(
        template="plotly_dark",
        title=(
            "Seeded Loose Monte Carlo Dispersion / Observability Dashboard"
            f"<br><sup>runs={len(rows)}, jump_rate={100.0 * float(np.mean(has_jump)):.1f}%, "
            f"wrong_basin_rate={100.0 * float(np.mean(wrong_basin)):.1f}%</sup>"
        ),
        height=1250,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html)


def print_monte_carlo_summary(rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    jump_flags = np.asarray([float(row["has_jump"]) for row in rows], dtype=float)
    wrong_flags = np.asarray([float(row["wrong_basin"]) for row in rows], dtype=float)
    mount_err_norm = np.asarray([float(row["final_mount_err_norm_deg"]) for row in rows], dtype=float)
    car_pitch_err = np.asarray([abs(float(row["final_car_pitch_err_deg"])) for row in rows], dtype=float)
    print(
        "mc_summary: "
        f"runs={len(rows)} "
        f"jump_rate={100.0 * float(np.mean(jump_flags)):.1f}% "
        f"wrong_basin_rate={100.0 * float(np.mean(wrong_flags)):.1f}% "
        f"mount_err_p50={float(np.percentile(mount_err_norm, 50.0)):.2f}deg "
        f"mount_err_p95={float(np.percentile(mount_err_norm, 95.0)):.2f}deg "
        f"car_pitch_err_p50={float(np.percentile(car_pitch_err, 50.0)):.2f}deg "
        f"car_pitch_err_p95={float(np.percentile(car_pitch_err, 95.0)):.2f}deg"
    )
    worst_rows = sorted(rows, key=lambda row: float(row["final_mount_err_norm_deg"]), reverse=True)[:5]
    for row in worst_rows:
        print(
            "mc_worst: "
            f"{row['label']} "
            f"mount_err_norm={float(row['final_mount_err_norm_deg']):.2f}deg "
            f"car_pitch_err={float(row['final_car_pitch_err_deg']):.2f}deg "
            f"seed=[{float(row['seed_roll_deg']):.2f},{float(row['seed_pitch_deg']):.2f},{float(row['seed_yaw_deg']):.2f}]deg "
            f"gps_vd_bias={float(row['gps_vd_bias_mps']):.2f}mps "
            f"gps_vd_bias_duration={float(row['gps_vd_bias_duration_s']):.2f}s "
            f"vel_std_scale={float(row['vel_std_scale']):.2f} "
            f"accel_z_bias={float(row['accel_z_bias_mps2']):.3f}mps2"
        )


def main() -> int:
    args = parse_args()
    case_dir = generate_dataset(args)
    results = []
    if not args.skip_default_cases:
        results = [
            run_seeded_case(case_dir, args.oan_root, case, args.init_time_s, args.jump_threshold_deg)
            for case in default_cases()
        ]
        write_plot(results, args.output_html)
    print(f"dataset={case_dir}")
    if results:
        print(f"html={args.output_html}")
    for result in results:
        jump_time = result["first_jump_time_s"]
        jump_text = "none" if math.isnan(jump_time) else f"{jump_time:.2f}s / {result['first_jump_mag_deg']:.2f}deg"
        final_mount = np.asarray(result["final_mount_deg"], dtype=float)
        print(
            f"{result['label']}: first_jump={jump_text} "
            f"final_mount_deg=[{final_mount[0]:.2f},{final_mount[1]:.2f},{final_mount[2]:.2f}] "
            f"final_car_pitch_deg={float(result['final_car_pitch_deg']):.2f} "
            f"final_baz={float(result['final_accel_bias_z']):.3f}"
        )

    mc_rows: list[dict[str, float | str]] = []
    if args.monte_carlo_runs > 0:
        mc_results = [
            run_seeded_case(case_dir, args.oan_root, case, args.init_time_s, args.jump_threshold_deg)
            for case in monte_carlo_cases(args)
        ]
        mc_rows = [monte_carlo_row(result, args) for result in mc_results]
        write_monte_carlo_summary(mc_rows, args.mc_summary_csv)
        write_monte_carlo_plot(mc_rows, args.mc_output_html)
        print(f"monte_carlo_summary_csv={args.mc_summary_csv}")
        print(f"monte_carlo_html={args.mc_output_html}")
        print_monte_carlo_summary(mc_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
