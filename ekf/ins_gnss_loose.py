import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from sympy import Matrix, Symbol, cse, symbols

from code_gen import RustCodeGenerator


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_RUST_DIR = SCRIPT_DIR / "src" / "generated_loose"
DEFAULT_OAN_ROOT = Path("/Users/ykshin/Dev/me/open-aided-navigation")
DEFAULT_SYNTH_CASE = DEFAULT_OAN_ROOT / "data" / "sim" / "generated_city_drive_case_5deg_15min"

WGS84_OMEGA_IE = 7.292115e-5

POS_E = slice(0, 3)
V_E = slice(3, 6)
PSI_EE = slice(6, 9)
B_F = slice(9, 12)
B_W = slice(12, 15)
S_F = slice(15, 18)
S_W = slice(18, 21)
PSI_CC = slice(21, 24)


def quat_to_rot(q):
    q0, q1, q2, q3 = q
    return Matrix(
        [
            [
                1 - 2 * q2**2 - 2 * q3**2,
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ],
            [
                2 * (q1 * q2 + q0 * q3),
                1 - 2 * q1**2 - 2 * q3**2,
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                1 - 2 * q1**2 - 2 * q2**2,
            ],
        ]
    )


def skew(v):
    return Matrix(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def create_reference_transition_and_noise():
    dt = Symbol("dt", real=True)
    q_es = Matrix(symbols("q0 q1 q2 q3", real=True))
    q_vb = Matrix(symbols("qcs0 qcs1 qcs2 qcs3", real=True))
    v_e = Matrix(symbols("vn ve vd", real=True))
    b_f = Matrix(symbols("bax bay baz", real=True))
    b_w = Matrix(symbols("bgx bgy bgz", real=True))
    s_f = Matrix(symbols("sax say saz", real=True))
    s_w = Matrix(symbols("sgx sgy sgz", real=True))
    accel_raw = Matrix([Symbol("dvx", real=True) / dt, Symbol("dvy", real=True) / dt, Symbol("dvz", real=True) / dt])
    gyro_raw = Matrix([Symbol("dax", real=True) / dt, Symbol("day", real=True) / dt, Symbol("daz", real=True) / dt])

    c_es = quat_to_rot(q_es)
    c_vb = quat_to_rot(q_vb).T
    f_s = Matrix(
        [
            s_f[0] * accel_raw[0] + b_f[0],
            s_f[1] * accel_raw[1] + b_f[1],
            s_f[2] * accel_raw[2] + b_f[2],
        ]
    )
    w_s = Matrix(
        [
            s_w[0] * gyro_raw[0] + b_w[0],
            s_w[1] * gyro_raw[1] + b_w[1],
            s_w[2] * gyro_raw[2] + b_w[2],
        ]
    )
    f_v = c_vb * f_s
    w_v = c_vb * w_s

    phi_cont = Matrix.zeros(24, 24)
    phi_cont[POS_E, V_E] = Matrix.eye(3)
    phi_cont[V_E, V_E] = skew(Matrix([0, 0, -2 * WGS84_OMEGA_IE]))
    phi_cont[V_E, PSI_EE] = skew(c_es * (-f_v))
    phi_cont[V_E, B_F] = c_es * c_vb
    phi_cont[V_E, S_F] = c_es * c_vb * Matrix.diag(accel_raw[0], accel_raw[1], accel_raw[2])
    phi_cont[V_E, PSI_CC] = c_es * c_vb * skew(f_s)
    phi_cont[PSI_EE, PSI_EE] = skew(Matrix([0, 0, -WGS84_OMEGA_IE]))
    phi_cont[PSI_EE, B_W] = c_es * c_vb
    phi_cont[PSI_EE, S_W] = c_es * c_vb * Matrix.diag(gyro_raw[0], gyro_raw[1], gyro_raw[2])
    phi_cont[PSI_EE, PSI_CC] = c_es * c_vb * skew(w_s)

    phi = Matrix.eye(24) + dt * phi_cont

    g = Matrix.zeros(24, 21)
    g[V_E, 0:3] = c_es * c_vb
    g[PSI_EE, 3:6] = c_es * c_vb
    g[B_F, 6:9] = Matrix.eye(3)
    g[B_W, 9:12] = Matrix.eye(3)
    g[S_F, 12:15] = Matrix.eye(3)
    g[S_W, 15:18] = Matrix.eye(3)
    g[PSI_CC, 18:21] = Matrix.eye(3)

    return phi, g


def create_reference_nhc_rows():
    q_es = Matrix(symbols("q0 q1 q2 q3", real=True))
    v_e = Matrix(symbols("vn ve vd", real=True))

    c_es = quat_to_rot(q_es)
    c_ve = c_es.T
    v_c = c_ve * v_e

    h = Matrix.zeros(3, 24)
    h[:, V_E] = c_ve
    h[:, PSI_EE] = c_ve * skew(v_e)
    return v_c, h[1, :], h[2, :]


def write_rust_nhc_row(path: Path, est_expr, h_row: Matrix, prefix: str):
    flat_values = Matrix([est_expr] + [h_row[i] for i in range(h_row.shape[1])])
    subexprs, reduced = cse(flat_values, symbols(f"{prefix}_0:2000"), optimizations="basic")
    gen = RustCodeGenerator(str(path))
    gen.print_string("Sub Expressions")
    gen.write_subexpressions(subexprs)
    values = reduced
    if len(values) == 1 and isinstance(values[0], Matrix):
        values = values[0]
    gen.print_string("Estimated Measurement")
    gen.file.write(f"vc_est = {gen.get_ccode(values[0])};\n\n")
    gen.print_string("Observation Jacobian")
    gen.write_matrix(Matrix(values[1:25]), "H")
    gen.close()


def emit_supports(path: Path, phi: Matrix, g: Matrix, h_y: Matrix, h_z: Matrix):
    with open(path, "w") as file:
        file.write("// Generated loose-filter sparsity supports\n")
        for prefix, matrix in [("F", phi), ("G", g)]:
            supports = []
            max_len = 0
            for i in range(matrix.shape[0]):
                row_support = [j for j in range(matrix.shape[1]) if matrix[i, j] != 0]
                supports.append(row_support)
                max_len = max(max_len, len(row_support))
            file.write(f"pub const {prefix}_MAX_ROW_NONZERO: usize = {max_len};\n")
            file.write(f"pub const {prefix}_ROW_COUNTS: [usize; LOOSE_ERROR_STATES] = [\n")
            file.write("    " + ", ".join(str(len(row)) for row in supports) + ",\n")
            file.write("];\n")
            file.write(
                f"pub const {prefix}_ROW_COLS: [[usize; {prefix}_MAX_ROW_NONZERO]; LOOSE_ERROR_STATES] = [\n"
            )
            for row in supports:
                padded = row + [0] * (max_len - len(row))
                file.write("    [" + ", ".join(str(col) for col in padded) + "],\n")
            file.write("];\n\n")

        for name, row in [("NHC_Y_SUPPORT", h_y), ("NHC_Z_SUPPORT", h_z)]:
            support = [j for j in range(row.shape[1]) if row[0, j] != 0]
            file.write(f"pub const {name}: [usize; {len(support)}] = [")
            file.write(", ".join(str(col) for col in support))
            file.write("];\n")


def emit_reference_rust():
    GENERATED_RUST_DIR.mkdir(parents=True, exist_ok=True)
    phi, g = create_reference_transition_and_noise()
    v_c, h_y, h_z = create_reference_nhc_rows()

    gen = RustCodeGenerator(str(GENERATED_RUST_DIR / "reference_error_transition_generated.rs"))
    gen.print_string("Sub Expressions")
    subexprs, reduced = cse(phi, symbols("tmp_F_0:2000"), optimizations="basic")
    gen.write_subexpressions(subexprs)
    gen.print_string("F")
    gen.write_matrix(Matrix(reduced), "F")
    gen.close()

    gen = RustCodeGenerator(str(GENERATED_RUST_DIR / "reference_error_noise_input_generated.rs"))
    gen.print_string("Sub Expressions")
    subexprs, reduced = cse(g, symbols("tmp_G_0:2000"), optimizations="basic")
    gen.write_subexpressions(subexprs)
    gen.print_string("G")
    gen.write_matrix(Matrix(reduced), "G")
    gen.close()

    emit_supports(GENERATED_RUST_DIR / "reference_support_generated.rs", phi, g, h_y, h_z)

    write_rust_nhc_row(GENERATED_RUST_DIR / "reference_nhc_y_generated.rs", v_c[1], h_y, "tmp_nhc_y")
    write_rust_nhc_row(GENERATED_RUST_DIR / "reference_nhc_z_generated.rs", v_c[2], h_z, "tmp_nhc_z")


def _ensure_oan_imports(oan_root: Path):
    oan_str = str(oan_root)
    if oan_str not in sys.path:
        sys.path.insert(0, oan_str)
    from open_aided_navigation.constants import IMU_TIMEOUT_US, WGS84
    from open_aided_navigation.demos.ins_gnss_loose import run_ins_gnss_loose
    from open_aided_navigation.enums import EventType, FilterMode
    from open_aided_navigation.filters import InsGnssFilterLoose, InsGnssLooseMeasDb, nav_filter_routine
    from open_aided_navigation.geo import dcm_ecef_to_ned, ecef_to_llh, llh_to_ecef, quat_ecef_to_ned
    from open_aided_navigation.io import import_gnss_data, import_imu_data
    from open_aided_navigation.math_utils import euler_to_quat, quat_conjugate, quat_mult, quat_to_euler
    from open_aided_navigation.models import ErrorStateMapInsGnssLoose, ParamsInsGnssFilterLoose, StateMapInsGnssLoose

    return {
        "IMU_TIMEOUT_US": IMU_TIMEOUT_US,
        "WGS84": WGS84,
        "run_ins_gnss_loose": run_ins_gnss_loose,
        "EventType": EventType,
        "FilterMode": FilterMode,
        "InsGnssFilterLoose": InsGnssFilterLoose,
        "InsGnssLooseMeasDb": InsGnssLooseMeasDb,
        "nav_filter_routine": nav_filter_routine,
        "dcm_ecef_to_ned": dcm_ecef_to_ned,
        "ecef_to_llh": ecef_to_llh,
        "llh_to_ecef": llh_to_ecef,
        "quat_ecef_to_ned": quat_ecef_to_ned,
        "import_gnss_data": import_gnss_data,
        "import_imu_data": import_imu_data,
        "euler_to_quat": euler_to_quat,
        "quat_conjugate": quat_conjugate,
        "quat_mult": quat_mult,
        "quat_to_euler": quat_to_euler,
        "ErrorStateMapInsGnssLoose": ErrorStateMapInsGnssLoose,
        "ParamsInsGnssFilterLoose": ParamsInsGnssFilterLoose,
        "StateMapInsGnssLoose": StateMapInsGnssLoose,
    }


def _load_gnss_velocity_map(input_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    path = input_dir / "gnss_velocity_meas.csv"
    if not path.is_file():
        return {}
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        next(reader, None)
        for row in reader:
            clean = [cell.strip() for cell in row if cell.strip()]
            if not clean:
                continue
            out[int(round(float(clean[0])))] = (
                np.array([float(clean[1]), float(clean[2]), float(clean[3])], dtype=float),
                np.array([float(clean[4]), float(clean[5]), float(clean[6])], dtype=float),
            )
    return out


def _attach_gnss_velocity(gnss_data, velocity_map: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    for sample in gnss_data:
        velocity = velocity_map.get(sample.ttag)
        if velocity is None:
            continue
        sample.vel_valid = True
        sample.v_n = velocity[0].copy()
        sample.v_acc_n = velocity[1].copy()


def _build_initialized_python_filter(input_dir: Path, oan_root: Path):
    mods = _ensure_oan_imports(oan_root)
    gyro_data, accel_data = mods["import_imu_data"](input_dir)
    gnss_data = mods["import_gnss_data"](input_dir)
    _attach_gnss_velocity(gnss_data, _load_gnss_velocity_map(input_dir))
    EventType = mods["EventType"]
    nav_filter = mods["InsGnssFilterLoose"]()
    meas_db = mods["InsGnssLooseMeasDb"](64)
    events = (
        [(sample.ttag, EventType.GYRO_DATA, index) for index, sample in enumerate(gyro_data)]
        + [(sample.ttag, EventType.ACCEL_DATA, index) for index, sample in enumerate(accel_data)]
        + [(sample.ttag, EventType.GNSS_DATA, index) for index, sample in enumerate(gnss_data)]
    )
    events.sort(key=lambda item: item[0])
    for event_idx, (_, event_type, sample_index) in enumerate(events):
        prev_mode = int(nav_filter.mode)
        if event_type == EventType.ACCEL_DATA:
            meas_db.add_accel(accel_data[sample_index])
        elif event_type == EventType.GYRO_DATA:
            meas_db.add_gyro(gyro_data[sample_index])
            nav_filter, meas_db = mods["nav_filter_routine"](nav_filter, meas_db)
            if int(nav_filter.mode) == int(mods["FilterMode"].RUNNING) and prev_mode != int(mods["FilterMode"].RUNNING):
                return {
                    "mods": mods,
                    "events": events,
                    "init_event_idx": event_idx,
                    "init_gyro_index": sample_index,
                    "gyro_data": gyro_data,
                    "accel_data": accel_data,
                    "gnss_data": gnss_data,
                    "nav_filter": nav_filter,
                }
        else:
            meas_db.add_gnss(gnss_data[sample_index])
    raise RuntimeError("Python loose filter never initialized on the provided case")


def _read_semicolon_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader)


def _load_truth_nav(input_dir: Path) -> dict[str, np.ndarray]:
    rows = _read_semicolon_csv(input_dir / "truth_nav.csv")
    out: dict[str, np.ndarray] = {}
    if not rows:
        raise ValueError(f"empty truth_nav.csv in {input_dir}")
    for key in rows[0].keys():
        out[key] = np.array([float(row[key]) for row in rows], dtype=float)
    return out


def _load_truth_states(input_dir: Path) -> dict[str, np.ndarray]:
    rows = _read_semicolon_csv(input_dir / "truth_states.csv")
    out: dict[str, np.ndarray] = {}
    for row in rows:
        out[row["State"]] = np.array([float(row["X"]), float(row["Y"]), float(row["Z"])], dtype=float)
    return out


def run_python_loose_case(
    input_dir: Path,
    oan_root: Path = DEFAULT_OAN_ROOT,
    lock_misalignment_axes: tuple[bool, bool, bool] | None = None,
) -> dict[str, np.ndarray]:
    mods = _ensure_oan_imports(oan_root)
    py_results = mods["run_ins_gnss_loose"](
        input_dir,
        lock_misalignment_axes=lock_misalignment_axes,
    )
    s_map = mods["StateMapInsGnssLoose"]
    x_out = np.asarray(py_results["x_out"], dtype=float)
    mask = np.asarray(py_results["mask"], dtype=int)
    if mask.size == 0:
        raise RuntimeError("Python loose replay produced no propagated states")

    q_es = x_out[s_map.Q_ES, :][:, mask].T
    q_cs = x_out[s_map.Q_CS, :][:, mask].T
    euler_mis_deg = np.asarray(py_results["euler_mis_deg"], dtype=float).T
    euler_ns_deg = np.asarray(py_results["euler_ns_deg"], dtype=float).T
    history = {
        "time_s": np.asarray(py_results["time"], dtype=float),
        "q_es": q_es,
        "q_cs": q_cs,
        "pos_e": x_out[s_map.POS_E, :][:, mask].T,
        "vel_e": x_out[s_map.V_E, :][:, mask].T,
        "accel_bias": x_out[s_map.B_F, :][:, mask].T,
        "gyro_bias_deg": np.rad2deg(x_out[s_map.B_W, :][:, mask].T),
        "accel_scale": x_out[s_map.S_F, :][:, mask].T,
        "gyro_scale": x_out[s_map.S_W, :][:, mask].T,
        "euler_ns_deg": euler_ns_deg,
        "euler_mis_deg": euler_mis_deg,
    }
    final = {
        "q_es": q_es[-1],
        "q_cs": q_cs[-1],
        "pos_e": history["pos_e"][-1],
        "vel_e": history["vel_e"][-1],
        "accel_bias": history["accel_bias"][-1],
        "gyro_bias_deg": history["gyro_bias_deg"][-1],
        "accel_scale": history["accel_scale"][-1],
        "gyro_scale": history["gyro_scale"][-1],
        "euler_ns_deg": euler_ns_deg[-1],
        "euler_mis_deg": euler_mis_deg[-1],
    }
    return {"final": final, "history": history, "raw": py_results}


def main():
    parser = argparse.ArgumentParser(description="Loose INS/GNSS symbolic utilities and Rust generator.")
    parser.add_argument("--emit-rust", action="store_true")
    parser.add_argument("--derive-fg", action="store_true")
    parser.add_argument("--run-python-case", type=Path)
    parser.add_argument("--oan-root", type=Path, default=DEFAULT_OAN_ROOT)
    args = parser.parse_args()

    if args.emit_rust:
        emit_reference_rust()
        return
    if args.derive_fg:
        phi, g = create_reference_transition_and_noise()
        print("Phi shape:", phi.shape)
        print("G shape:", g.shape)
        return
    if args.run_python_case is not None:
        results = run_python_loose_case(args.run_python_case, args.oan_root)
        final = results["final"]
        print("Python loose final state:")
        print("  accel_bias", np.array2string(final["accel_bias"], precision=6))
        print("  gyro_bias_deg", np.array2string(final["gyro_bias_deg"], precision=6))
        print("  accel_scale", np.array2string(final["accel_scale"], precision=6))
        print("  gyro_scale", np.array2string(final["gyro_scale"], precision=6))
        print("  mis_deg", np.array2string(final["euler_mis_deg"], precision=6))
        return
    print(
        "Usage: python3 ekf/ins_gnss_loose.py "
        "[--derive-fg|--emit-rust|--run-python-case CASE_DIR]"
    )


if __name__ == "__main__":
    main()
