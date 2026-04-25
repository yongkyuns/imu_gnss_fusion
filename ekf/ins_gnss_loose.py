import argparse
import csv
import ctypes
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from sympy import Matrix, Symbol, cse, symbols

from code_gen import CodeGenerator, RustCodeGenerator


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_C_DIR = SCRIPT_DIR / "c" / "generated_loose"
GENERATED_RUST_DIR = SCRIPT_DIR / "src" / "generated_loose"
C_DIR = SCRIPT_DIR / "c"
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
    v_e = Matrix(symbols("vn ve vd", real=True))
    b_f = Matrix(symbols("bax bay baz", real=True))
    b_w = Matrix(symbols("bgx bgy bgz", real=True))
    s_f = Matrix(symbols("sax say saz", real=True))
    s_w = Matrix(symbols("sgx sgy sgz", real=True))
    accel_raw = Matrix([Symbol("dvx", real=True) / dt, Symbol("dvy", real=True) / dt, Symbol("dvz", real=True) / dt])
    gyro_raw = Matrix([Symbol("dax", real=True) / dt, Symbol("day", real=True) / dt, Symbol("daz", real=True) / dt])

    c_es = quat_to_rot(q_es)
    f_s = Matrix(
        [
            s_f[0] * accel_raw[0] + b_f[0],
            s_f[1] * accel_raw[1] + b_f[1],
            s_f[2] * accel_raw[2] + b_f[2],
        ]
    )

    phi_cont = Matrix.zeros(24, 24)
    phi_cont[POS_E, V_E] = Matrix.eye(3)
    phi_cont[V_E, V_E] = skew(Matrix([0, 0, -2 * WGS84_OMEGA_IE]))
    phi_cont[V_E, PSI_EE] = skew(c_es * (-f_s))
    phi_cont[V_E, B_F] = c_es
    phi_cont[V_E, S_F] = c_es * Matrix.diag(accel_raw[0], accel_raw[1], accel_raw[2])
    phi_cont[PSI_EE, PSI_EE] = skew(Matrix([0, 0, -WGS84_OMEGA_IE]))
    phi_cont[PSI_EE, B_W] = c_es
    phi_cont[PSI_EE, S_W] = c_es * Matrix.diag(gyro_raw[0], gyro_raw[1], gyro_raw[2])

    phi = Matrix.eye(24) + dt * phi_cont

    g = Matrix.zeros(24, 21)
    g[V_E, 0:3] = c_es
    g[PSI_EE, 3:6] = c_es
    g[B_F, 6:9] = Matrix.eye(3)
    g[B_W, 9:12] = Matrix.eye(3)
    g[S_F, 12:15] = Matrix.eye(3)
    g[S_W, 15:18] = Matrix.eye(3)
    g[PSI_CC, 18:21] = Matrix.eye(3)

    return phi, g


def create_reference_nhc_rows():
    q_es = Matrix(symbols("q0 q1 q2 q3", real=True))
    q_cs = Matrix(symbols("qcs0 qcs1 qcs2 qcs3", real=True))
    v_e = Matrix(symbols("vn ve vd", real=True))

    c_es = quat_to_rot(q_es)
    c_cs = quat_to_rot(q_cs)
    c_ce = c_cs * c_es.T
    v_c = c_ce * v_e

    h = Matrix.zeros(3, 24)
    h[:, V_E] = c_ce
    h[:, PSI_EE] = c_ce * skew(v_e)
    h[:, PSI_CC] = skew(-v_c)
    return v_c, h[1, :], h[2, :]


def write_matrix(path: Path, exprs, var_name: str):
    subexprs, reduced = cse(exprs, symbols(f"tmp_{var_name}_0:2000"), optimizations="basic")
    gen = CodeGenerator(str(path))
    gen.print_string("Sub Expressions")
    gen.write_subexpressions(subexprs)
    gen.print_string(var_name)
    gen.write_matrix(Matrix(reduced), var_name)
    gen.close()


def write_nhc_row(path: Path, est_expr, h_row: Matrix, prefix: str):
    flat_values = Matrix([est_expr] + [h_row[i] for i in range(h_row.shape[1])])
    subexprs, reduced = cse(flat_values, symbols(f"{prefix}_0:2000"), optimizations="basic")
    gen = CodeGenerator(str(path))
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


def emit_reference_c():
    GENERATED_C_DIR.mkdir(parents=True, exist_ok=True)
    phi, g = create_reference_transition_and_noise()
    v_c, h_y, h_z = create_reference_nhc_rows()

    write_matrix(GENERATED_C_DIR / "reference_error_transition_generated.c", phi, "F")
    write_matrix(GENERATED_C_DIR / "reference_error_noise_input_generated.c", g, "G")
    write_nhc_row(GENERATED_C_DIR / "reference_nhc_y_generated.c", v_c[1], h_y, "tmp_nhc_y")
    write_nhc_row(GENERATED_C_DIR / "reference_nhc_z_generated.c", v_c[2], h_z, "tmp_nhc_z")


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

    write_rust_nhc_row(GENERATED_RUST_DIR / "reference_nhc_y_generated.rs", v_c[1], h_y, "tmp_nhc_y")
    write_rust_nhc_row(GENERATED_RUST_DIR / "reference_nhc_z_generated.rs", v_c[2], h_z, "tmp_nhc_z")


class CLoosePredictNoise(ctypes.Structure):
    _fields_ = [
        ("gyro_var", ctypes.c_float),
        ("accel_var", ctypes.c_float),
        ("gyro_bias_rw_var", ctypes.c_float),
        ("accel_bias_rw_var", ctypes.c_float),
        ("gyro_scale_rw_var", ctypes.c_float),
        ("accel_scale_rw_var", ctypes.c_float),
        ("mount_align_rw_var", ctypes.c_float),
    ]


class CLooseNominalState(ctypes.Structure):
    _fields_ = [
        ("q0", ctypes.c_float),
        ("q1", ctypes.c_float),
        ("q2", ctypes.c_float),
        ("q3", ctypes.c_float),
        ("vn", ctypes.c_float),
        ("ve", ctypes.c_float),
        ("vd", ctypes.c_float),
        ("pn", ctypes.c_float),
        ("pe", ctypes.c_float),
        ("pd", ctypes.c_float),
        ("bgx", ctypes.c_float),
        ("bgy", ctypes.c_float),
        ("bgz", ctypes.c_float),
        ("bax", ctypes.c_float),
        ("bay", ctypes.c_float),
        ("baz", ctypes.c_float),
        ("sgx", ctypes.c_float),
        ("sgy", ctypes.c_float),
        ("sgz", ctypes.c_float),
        ("sax", ctypes.c_float),
        ("say", ctypes.c_float),
        ("saz", ctypes.c_float),
        ("qcs0", ctypes.c_float),
        ("qcs1", ctypes.c_float),
        ("qcs2", ctypes.c_float),
        ("qcs3", ctypes.c_float),
    ]


class CLooseImuDelta(ctypes.Structure):
    _fields_ = [
        ("dax_1", ctypes.c_float),
        ("day_1", ctypes.c_float),
        ("daz_1", ctypes.c_float),
        ("dvx_1", ctypes.c_float),
        ("dvy_1", ctypes.c_float),
        ("dvz_1", ctypes.c_float),
        ("dax_2", ctypes.c_float),
        ("day_2", ctypes.c_float),
        ("daz_2", ctypes.c_float),
        ("dvx_2", ctypes.c_float),
        ("dvy_2", ctypes.c_float),
        ("dvz_2", ctypes.c_float),
        ("dt", ctypes.c_float),
    ]


class CLoose(ctypes.Structure):
    _fields_ = [
        ("nominal", CLooseNominalState),
        ("p", (ctypes.c_float * 24) * 24),
        ("noise", CLoosePredictNoise),
        ("pos_e64", ctypes.c_double * 3),
        ("qcs64", ctypes.c_double * 4),
        ("p64", (ctypes.c_double * 24) * 24),
        ("last_dx", ctypes.c_float * 24),
        ("last_obs_count", ctypes.c_int),
        ("last_obs_types", ctypes.c_int * 8),
    ]


def _default_noise() -> CLoosePredictNoise:
    return CLoosePredictNoise(
        gyro_var=np.float32(2.5e-5),
        accel_var=np.float32(9.0e-4),
        gyro_bias_rw_var=np.float32(1.0e-12),
        accel_bias_rw_var=np.float32(1.0e-10),
        gyro_scale_rw_var=np.float32(1.0e-10),
        accel_scale_rw_var=np.float32(1.0e-10),
        mount_align_rw_var=np.float32(1.0e-8),
    )


def _shared_lib_path() -> Path:
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return C_DIR / "build" / f"libsf_loose{suffix}"


def _compile_shared_lib() -> Path:
    lib_path = _shared_lib_path()
    source = C_DIR / "src" / "sf_loose.c"
    if lib_path.exists() and lib_path.stat().st_mtime >= source.stat().st_mtime:
        return lib_path
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Darwin":
        cmd = [
            "cc",
            "-std=c11",
            "-O2",
            "-dynamiclib",
            "-fPIC",
            "-I",
            str(C_DIR),
            "-I",
            str(C_DIR / "include"),
            str(source),
            "-lm",
            "-o",
            str(lib_path),
        ]
    else:
        cmd = [
            "cc",
            "-std=c11",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(C_DIR),
            "-I",
            str(C_DIR / "include"),
            str(source),
            "-lm",
            "-o",
            str(lib_path),
        ]
    subprocess.run(cmd, check=True)
    return lib_path


def _load_c_lib():
    lib = ctypes.CDLL(str(_compile_shared_lib()))
    lib.sf_loose_init.argtypes = [
        ctypes.POINTER(CLoose),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(CLoosePredictNoise),
    ]
    lib.sf_loose_predict.argtypes = [ctypes.POINTER(CLoose), ctypes.POINTER(CLooseImuDelta)]
    lib.sf_loose_fuse_gps_reference.argtypes = [
        ctypes.POINTER(CLoose),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.sf_loose_fuse_gps_reference_full.argtypes = [
        ctypes.POINTER(CLoose),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
    ]
    lib.sf_loose_fuse_reference_batch.argtypes = [
        ctypes.POINTER(CLoose),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
    ]
    lib.sf_loose_fuse_reference_batch_full.argtypes = [
        ctypes.POINTER(CLoose),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
    ]
    return lib


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


def _seed_c_loose_from_python(loose: CLoose, py_filter, state_map) -> None:
    x = py_filter.x
    p = py_filter.p
    loose.nominal.q0 = float(x[state_map.Q_ES][0])
    loose.nominal.q1 = float(x[state_map.Q_ES][1])
    loose.nominal.q2 = float(x[state_map.Q_ES][2])
    loose.nominal.q3 = float(x[state_map.Q_ES][3])
    loose.nominal.vn = float(x[state_map.V_E][0])
    loose.nominal.ve = float(x[state_map.V_E][1])
    loose.nominal.vd = float(x[state_map.V_E][2])
    loose.nominal.pn = float(x[state_map.POS_E][0])
    loose.nominal.pe = float(x[state_map.POS_E][1])
    loose.nominal.pd = float(x[state_map.POS_E][2])
    loose.nominal.bax = float(x[state_map.B_F][0])
    loose.nominal.bay = float(x[state_map.B_F][1])
    loose.nominal.baz = float(x[state_map.B_F][2])
    loose.nominal.bgx = float(x[state_map.B_W][0])
    loose.nominal.bgy = float(x[state_map.B_W][1])
    loose.nominal.bgz = float(x[state_map.B_W][2])
    loose.nominal.sax = float(x[state_map.S_F][0])
    loose.nominal.say = float(x[state_map.S_F][1])
    loose.nominal.saz = float(x[state_map.S_F][2])
    loose.nominal.sgx = float(x[state_map.S_W][0])
    loose.nominal.sgy = float(x[state_map.S_W][1])
    loose.nominal.sgz = float(x[state_map.S_W][2])
    loose.nominal.qcs0 = float(x[state_map.Q_CS][0])
    loose.nominal.qcs1 = float(x[state_map.Q_CS][1])
    loose.nominal.qcs2 = float(x[state_map.Q_CS][2])
    loose.nominal.qcs3 = float(x[state_map.Q_CS][3])
    for i in range(3):
        loose.pos_e64[i] = float(x[state_map.POS_E][i])
    for i in range(4):
        loose.qcs64[i] = float(x[state_map.Q_CS][i])
    for i in range(24):
        for j in range(24):
            loose.p[i][j] = float(p[i, j])
            loose.p64[i][j] = float(p[i, j])


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


def _apply_misalignment_axis_locks(loose: CLoose, mods, lock_axes) -> None:
    lock_axes_arr = np.asarray(lock_axes, dtype=bool).reshape(-1)
    if lock_axes_arr.size != 3 or not np.any(lock_axes_arr):
        return
    nominal_mount = mods["euler_to_quat"](
        mods["ParamsInsGnssFilterLoose"].ROLL_IMU2CAR,
        mods["ParamsInsGnssFilterLoose"].PITCH_IMU2CAR,
        mods["ParamsInsGnssFilterLoose"].YAW_IMU2CAR,
    )
    quat_mult = mods["quat_mult"]
    quat_conjugate = mods["quat_conjugate"]
    quat_to_euler = mods["quat_to_euler"]
    euler_to_quat = mods["euler_to_quat"]

    q_cs = np.array([loose.nominal.qcs0, loose.nominal.qcs1, loose.nominal.qcs2, loose.nominal.qcs3], dtype=float)
    q_cs_residual = quat_mult(q_cs, quat_conjugate(nominal_mount))
    euler_residual = quat_to_euler(q_cs_residual)
    euler_residual[lock_axes_arr] = 0.0
    q_cs_locked = euler_to_quat(euler_residual[0], euler_residual[1], euler_residual[2])
    q_cs_locked = quat_mult(q_cs_locked, nominal_mount)
    q_cs_locked = q_cs_locked / np.linalg.norm(q_cs_locked)

    loose.nominal.qcs0 = float(q_cs_locked[0])
    loose.nominal.qcs1 = float(q_cs_locked[1])
    loose.nominal.qcs2 = float(q_cs_locked[2])
    loose.nominal.qcs3 = float(q_cs_locked[3])
    for i in range(4):
        loose.qcs64[i] = float(q_cs_locked[i])

    locked_idx = np.array([PSI_CC.start, PSI_CC.start + 1, PSI_CC.start + 2], dtype=int)[lock_axes_arr]
    for idx in locked_idx:
        for j in range(24):
            loose.p[idx][j] = 0.0
            loose.p[j][idx] = 0.0
            loose.p64[idx][j] = 0.0
            loose.p64[j][idx] = 0.0
        loose.last_dx[idx] = 0.0


def run_c_loose_case(
    input_dir: Path,
    oan_root: Path = DEFAULT_OAN_ROOT,
    collect_history: bool = False,
    lock_misalignment_axes: tuple[bool, bool, bool] | None = None,
) -> dict[str, np.ndarray]:
    init = _build_initialized_python_filter(input_dir, oan_root)
    mods = init["mods"]
    state_map = mods["StateMapInsGnssLoose"]
    gyro_data = init["gyro_data"]
    accel_data = init["accel_data"]
    gnss_data = init["gnss_data"]
    nav_filter = init["nav_filter"]
    c_lib = _load_c_lib()
    loose = CLoose()
    noise = _default_noise()
    c_lib.sf_loose_init(ctypes.byref(loose), None, ctypes.byref(noise))
    _seed_c_loose_from_python(loose, nav_filter, state_map)
    if lock_misalignment_axes is not None:
        _apply_misalignment_axis_locks(loose, mods, lock_misalignment_axes)

    init_gyro_index = init["init_gyro_index"]
    imu_timeout_us = mods["IMU_TIMEOUT_US"]
    llh_to_ecef = mods["llh_to_ecef"]
    dcm_ecef_to_ned = mods["dcm_ecef_to_ned"]
    ecef_to_llh = mods["ecef_to_llh"]
    quat_ecef_to_ned = mods["quat_ecef_to_ned"]
    quat_mult = mods["quat_mult"]
    quat_conjugate = mods["quat_conjugate"]
    quat_to_euler = mods["quat_to_euler"]
    nominal_mount = mods["euler_to_quat"](
        mods["ParamsInsGnssFilterLoose"].ROLL_IMU2CAR,
        mods["ParamsInsGnssFilterLoose"].PITCH_IMU2CAR,
        mods["ParamsInsGnssFilterLoose"].YAW_IMU2CAR,
    )

    gnss_index = 0
    last_used_gnss_ttag = 0
    while gnss_index + 1 < len(gnss_data) and gnss_data[gnss_index + 1].ttag <= gyro_data[init_gyro_index].ttag:
        gnss_index += 1

    final_state = None
    history: dict[str, list[np.ndarray] | list[float] | list[int]] | None = None
    if collect_history:
        history = {
            "ttag": [],
            "time_s": [],
            "q_es": [],
            "q_cs": [],
            "pos_e": [],
            "vel_e": [],
            "accel_bias": [],
            "gyro_bias_deg": [],
            "accel_scale": [],
            "gyro_scale": [],
            "euler_ns_deg": [],
            "euler_nc_deg": [],
            "euler_mis_deg": [],
        }
    for gyro_index in range(init_gyro_index + 1, len(gyro_data)):
        prev_gyro = gyro_data[gyro_index - 1]
        curr_gyro = gyro_data[gyro_index]
        dt_s = 1e-6 * float(curr_gyro.ttag - prev_gyro.ttag)
        accel_prev = accel_data[gyro_index - 1].f_b.astype(float)
        accel_curr = accel_data[gyro_index].f_b.astype(float)
        imu = CLooseImuDelta(
            dax_1=np.float32(prev_gyro.omega_is[0] * dt_s),
            day_1=np.float32(prev_gyro.omega_is[1] * dt_s),
            daz_1=np.float32(prev_gyro.omega_is[2] * dt_s),
            dvx_1=np.float32(accel_prev[0] * dt_s),
            dvy_1=np.float32(accel_prev[1] * dt_s),
            dvz_1=np.float32(accel_prev[2] * dt_s),
            dax_2=np.float32(curr_gyro.omega_is[0] * dt_s),
            day_2=np.float32(curr_gyro.omega_is[1] * dt_s),
            daz_2=np.float32(curr_gyro.omega_is[2] * dt_s),
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
        speed_acc_mps = 0.0
        if gnss_index < len(gnss_data):
            gnss = gnss_data[gnss_index]
            d_ttag = curr_gyro.ttag - gnss.ttag
            if d_ttag < imu_timeout_us // 2 and gnss.ttag != last_used_gnss_ttag:
                pos_ecef = np.asarray(llh_to_ecef(gnss.lat, gnss.lon, gnss.height, mods["WGS84"]), dtype=np.float64)
                pos_ptr = (ctypes.c_double * 3)(*pos_ecef.tolist())
                if getattr(gnss, "vel_valid", False):
                    vel_ecef = np.asarray(dcm_ecef_to_ned(gnss.lat, gnss.lon).T @ gnss.v_n, dtype=np.float32)
                    vel_ptr = (ctypes.c_float * 3)(*vel_ecef.tolist())
                    vel_std_ptr = (ctypes.c_float * 3)(*gnss.v_acc_n.astype(np.float32).tolist())
                    speed_acc_mps = float(max(gnss.v_acc_n))
                elif gnss.speed > 0.0 and gnss.speed_acc > 0.0:
                    vel_n = np.array(
                        [gnss.speed * math.cos(gnss.heading), gnss.speed * math.sin(gnss.heading), 0.0],
                        dtype=np.float32,
                    )
                    vel_ecef = np.asarray(dcm_ecef_to_ned(gnss.lat, gnss.lon).T @ vel_n, dtype=np.float32)
                    vel_ptr = (ctypes.c_float * 3)(*vel_ecef.tolist())
                    speed_acc_mps = float(gnss.speed_acc)
                h_acc_m = float(gnss.h_acc)
                d_ttag_s = 1e-6 * float(curr_gyro.ttag - last_used_gnss_ttag)
                if d_ttag_s == 0.0 or d_ttag_s >= 1.0:
                    d_ttag_s = 1.0
                last_used_gnss_ttag = gnss.ttag
            else:
                d_ttag_s = 1.0
        else:
            d_ttag_s = 1.0

        gyro_ptr = (ctypes.c_float * 3)(*curr_gyro.omega_is.astype(np.float32).tolist())
        accel_ptr = (ctypes.c_float * 3)(*accel_curr.astype(np.float32).tolist())
        if vel_std_ptr is not None:
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
        else:
            c_lib.sf_loose_fuse_reference_batch(
                ctypes.byref(loose),
                pos_ptr,
                vel_ptr,
                ctypes.c_float(h_acc_m),
                ctypes.c_float(speed_acc_mps),
                ctypes.c_float(d_ttag_s),
                gyro_ptr,
                accel_ptr,
                ctypes.c_float(dt_s),
            )
        if lock_misalignment_axes is not None:
            _apply_misalignment_axis_locks(loose, mods, lock_misalignment_axes)

        final_state = {
            "ttag": curr_gyro.ttag,
            "q_es": np.array([loose.nominal.q0, loose.nominal.q1, loose.nominal.q2, loose.nominal.q3], dtype=float),
            "q_cs": np.array([loose.nominal.qcs0, loose.nominal.qcs1, loose.nominal.qcs2, loose.nominal.qcs3], dtype=float),
            "pos_e": np.array([loose.pos_e64[0], loose.pos_e64[1], loose.pos_e64[2]], dtype=float),
            "vel_e": np.array([loose.nominal.vn, loose.nominal.ve, loose.nominal.vd], dtype=float),
            "accel_bias": np.array([loose.nominal.bax, loose.nominal.bay, loose.nominal.baz], dtype=float),
            "gyro_bias_deg": np.rad2deg(np.array([loose.nominal.bgx, loose.nominal.bgy, loose.nominal.bgz], dtype=float)),
            "accel_scale": np.array([loose.nominal.sax, loose.nominal.say, loose.nominal.saz], dtype=float),
            "gyro_scale": np.array([loose.nominal.sgx, loose.nominal.sgy, loose.nominal.sgz], dtype=float),
        }
        lat, lon, _ = ecef_to_llh(final_state["pos_e"], mods["WGS84"])
        q_ne = quat_ecef_to_ned(lat, lon)
        q_ns = quat_mult(q_ne, final_state["q_es"])
        final_state["euler_ns_deg"] = np.rad2deg(quat_to_euler(q_ns))
        final_state["euler_mis_deg"] = np.rad2deg(
            quat_to_euler(quat_mult(final_state["q_cs"], quat_conjugate(nominal_mount)))
        )
        q_nc = quat_mult(q_ns, quat_conjugate(final_state["q_cs"]))
        final_state["euler_nc_deg"] = np.rad2deg(quat_to_euler(q_nc))
        if history is not None:
            history["ttag"].append(int(curr_gyro.ttag))
            history["time_s"].append(1e-6 * float(curr_gyro.ttag - gyro_data[init_gyro_index].ttag))
            history["q_es"].append(final_state["q_es"].copy())
            history["q_cs"].append(final_state["q_cs"].copy())
            history["pos_e"].append(final_state["pos_e"].copy())
            history["vel_e"].append(final_state["vel_e"].copy())
            history["accel_bias"].append(final_state["accel_bias"].copy())
            history["gyro_bias_deg"].append(final_state["gyro_bias_deg"].copy())
            history["accel_scale"].append(final_state["accel_scale"].copy())
            history["gyro_scale"].append(final_state["gyro_scale"].copy())
            history["euler_ns_deg"].append(final_state["euler_ns_deg"].copy())
            history["euler_nc_deg"].append(final_state["euler_nc_deg"].copy())
            history["euler_mis_deg"].append(final_state["euler_mis_deg"].copy())

    if final_state is None:
        raise RuntimeError("C loose replay produced no propagated states")
    result = {"final": final_state}
    if history is not None:
        result["history"] = {
            key: np.asarray(value, dtype=float if key != "ttag" else int) for key, value in history.items()
        }
    return result


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


def write_c_case_plotly_dark(input_dir: Path, output_html: Path, oan_root: Path = DEFAULT_OAN_ROOT) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    results = run_c_loose_case(input_dir, oan_root, collect_history=True)
    history = results["history"]
    truth_nav = _load_truth_nav(input_dir)
    truth_states = _load_truth_states(input_dir)

    t_hist_us = np.asarray(history["ttag"], dtype=float)
    t_hist = 1e-6 * (t_hist_us - t_hist_us[0])
    t_truth_us = np.asarray(truth_nav["TimeUS"], dtype=float)

    truth_car = np.vstack(
        [
            np.interp(t_hist_us, t_truth_us, truth_nav["RollCarDeg"]),
            np.interp(t_hist_us, t_truth_us, truth_nav["PitchCarDeg"]),
            np.interp(t_hist_us, t_truth_us, truth_nav["YawCarDeg"]),
        ]
    )
    truth_ns = np.vstack(
        [
            np.interp(t_hist_us, t_truth_us, truth_nav["RollNsDeg"]),
            np.interp(t_hist_us, t_truth_us, truth_nav["PitchNsDeg"]),
            np.interp(t_hist_us, t_truth_us, truth_nav["YawNsDeg"]),
        ]
    )
    const_truth = {
        "accel_bias": truth_states["accel_bias_mps2"],
        "gyro_bias_deg": truth_states["gyro_bias_dps"],
        "accel_scale": truth_states["accel_scale"],
        "gyro_scale": truth_states["gyro_scale"],
        "euler_mis_deg": truth_states["misalignment_deg"],
    }

    fig = make_subplots(
        rows=5,
        cols=3,
        subplot_titles=[
            "Car Roll", "Car Pitch", "Car Yaw",
            "Mis Roll", "Mis Pitch", "Mis Yaw",
            "Gyro Bias X", "Gyro Bias Y", "Gyro Bias Z",
            "Accel Bias X", "Accel Bias Y", "Accel Bias Z",
            "Gyro Scale X", "Gyro Scale Y", "Gyro Scale Z",
        ],
        vertical_spacing=0.05,
        horizontal_spacing=0.04,
    )

    axis_names = ["Roll", "Pitch", "Yaw"]
    colors = {"est": "#4cc9f0", "truth": "#f72585"}
    for idx, axis in enumerate(axis_names):
        fig.add_trace(
            go.Scatter(x=t_hist, y=history["euler_nc_deg"][:, idx], name=f"Car {axis} Est", line=dict(color=colors["est"])),
            row=1,
            col=idx + 1,
        )
        fig.add_trace(
            go.Scatter(x=t_hist, y=truth_car[idx], name=f"Car {axis} Truth", line=dict(color=colors["truth"], dash="dash")),
            row=1,
            col=idx + 1,
        )
        fig.add_trace(
            go.Scatter(x=t_hist, y=history["euler_mis_deg"][:, idx], name=f"Mis {axis} Est", line=dict(color=colors["est"])),
            row=2,
            col=idx + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_hist,
                y=np.full_like(t_hist, const_truth["euler_mis_deg"][idx]),
                name=f"Mis {axis} Truth",
                line=dict(color=colors["truth"], dash="dash"),
            ),
            row=2,
            col=idx + 1,
        )

    labels = [
        ("gyro_bias_deg", "Gyro Bias", 3),
        ("accel_bias", "Accel Bias", 4),
        ("gyro_scale", "Gyro Scale", 5),
    ]
    xyz = ["X", "Y", "Z"]
    for key, prefix, row in labels:
        for idx, axis in enumerate(xyz):
            fig.add_trace(
                go.Scatter(x=t_hist, y=history[key][:, idx], name=f"{prefix} {axis} Est", line=dict(color=colors["est"])),
                row=row,
                col=idx + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_hist,
                    y=np.full_like(t_hist, const_truth[key][idx]),
                    name=f"{prefix} {axis} Truth",
                    line=dict(color=colors["truth"], dash="dash"),
                ),
                row=row,
                col=idx + 1,
            )

    fig.update_layout(
        template="plotly_dark",
        title=f"C Loose INS/GNSS Convergence: {input_dir.name}",
        height=1800,
        width=1500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=50, r=30, t=90, b=40),
    )
    fig.update_xaxes(title_text="Time [s]")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    return output_html


def compare_c_to_open_aided(input_dir: Path, oan_root: Path = DEFAULT_OAN_ROOT) -> dict[str, np.ndarray]:
    mods = _ensure_oan_imports(oan_root)
    py_results = mods["run_ins_gnss_loose"](input_dir)
    s_map = mods["StateMapInsGnssLoose"]
    py_mask = np.asarray(py_results["mask"])
    py_x = np.asarray(py_results["x_out"])
    py_final = int(py_mask[-1])
    c_results = run_c_loose_case(input_dir, oan_root)
    c_final = c_results["final"]
    py_final_summary = {
        "accel_bias": py_x[s_map.B_F, py_final],
        "gyro_bias_deg": np.rad2deg(py_x[s_map.B_W, py_final]),
        "accel_scale": py_x[s_map.S_F, py_final],
        "gyro_scale": py_x[s_map.S_W, py_final],
        "mis_deg": np.asarray(py_results["euler_mis_deg"])[:, -1],
    }
    c_final_summary = {
        "accel_bias": c_final["accel_bias"],
        "gyro_bias_deg": c_final["gyro_bias_deg"],
        "accel_scale": c_final["accel_scale"],
        "gyro_scale": c_final["gyro_scale"],
        "mis_deg": c_final["euler_mis_deg"],
    }
    diffs = {key: c_final_summary[key] - py_final_summary[key] for key in py_final_summary}
    return {
        "python": py_final_summary,
        "c": c_final_summary,
        "diffs": diffs,
    }


def main():
    parser = argparse.ArgumentParser(description="Loose INS/GNSS symbolic utilities and C runtime harness.")
    parser.add_argument("--emit-c", action="store_true")
    parser.add_argument("--emit-rust", action="store_true")
    parser.add_argument("--derive-fg", action="store_true")
    parser.add_argument("--run-c-case", type=Path)
    parser.add_argument("--run-python-case", type=Path)
    parser.add_argument("--compare-open-aided", type=Path)
    parser.add_argument("--plot-c-dark", type=Path)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--oan-root", type=Path, default=DEFAULT_OAN_ROOT)
    args = parser.parse_args()

    if args.emit_c:
        emit_reference_c()
    if args.emit_rust:
        emit_reference_rust()
        return
    if args.derive_fg:
        phi, g = create_reference_transition_and_noise()
        print("Phi shape:", phi.shape)
        print("G shape:", g.shape)
        return
    if args.run_c_case is not None:
        results = run_c_loose_case(args.run_c_case, args.oan_root)
        final = results["final"]
        print("C loose final state:")
        print("  accel_bias", np.array2string(final["accel_bias"], precision=6))
        print("  gyro_bias_deg", np.array2string(final["gyro_bias_deg"], precision=6))
        print("  accel_scale", np.array2string(final["accel_scale"], precision=6))
        print("  gyro_scale", np.array2string(final["gyro_scale"], precision=6))
        print("  mis_deg", np.array2string(final["euler_mis_deg"], precision=6))
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
    if args.compare_open_aided is not None:
        comparison = compare_c_to_open_aided(args.compare_open_aided, args.oan_root)
        for key in ["accel_bias", "gyro_bias_deg", "accel_scale", "gyro_scale", "mis_deg"]:
            print(f"{key}:")
            print("  python", np.array2string(comparison["python"][key], precision=6))
            print("  c     ", np.array2string(comparison["c"][key], precision=6))
            print("  diff  ", np.array2string(comparison["diffs"][key], precision=6))
        return
    if args.plot_c_dark is not None:
        output_html = args.output_html
        if output_html is None:
            output_html = args.plot_c_dark / "c_loose_dark.html"
        output_path = write_c_case_plotly_dark(args.plot_c_dark, output_html, args.oan_root)
        print(f"Wrote {output_path}")
        return
        return
    print(
        "Usage: python3 ekf/ins_gnss_loose.py "
        "[--derive-fg|--emit-c|--run-c-case CASE_DIR|--run-python-case CASE_DIR|--compare-open-aided CASE_DIR]"
    )


if __name__ == "__main__":
    main()
