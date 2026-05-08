import argparse
import math
from pathlib import Path

import numpy as np
from sympy import Matrix, Symbol, cse, symbols

from code_gen import RustCodeGenerator


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_RUST_DIR = SCRIPT_DIR / "src" / "generated_full"

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
        file.write("// Generated full-filter sparsity supports\n")
        for prefix, matrix in [("F", phi), ("G", g)]:
            supports = []
            max_len = 0
            for i in range(matrix.shape[0]):
                row_support = [j for j in range(matrix.shape[1]) if matrix[i, j] != 0]
                supports.append(row_support)
                max_len = max(max_len, len(row_support))
            file.write(f"pub const {prefix}_MAX_ROW_NONZERO: usize = {max_len};\n")
            file.write(f"pub const {prefix}_ROW_COUNTS: [usize; FULL_ERROR_STATES] = [\n")
            file.write("    " + ", ".join(str(len(row)) for row in supports) + ",\n")
            file.write("];\n")
            file.write(
                f"pub const {prefix}_ROW_COLS: [[usize; {prefix}_MAX_ROW_NONZERO]; FULL_ERROR_STATES] = [\n"
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


def main():
    parser = argparse.ArgumentParser(description="Full INS/GNSS symbolic utilities and Rust generator.")
    parser.add_argument("--emit-rust", action="store_true")
    parser.add_argument("--derive-fg", action="store_true")
    args = parser.parse_args()

    if args.emit_rust:
        emit_reference_rust()
        return
    if args.derive_fg:
        phi, g = create_reference_transition_and_noise()
        print("Phi shape:", phi.shape)
        print("G shape:", g.shape)
        return
    print("Usage: python3 sensor_fusion/ins_gnss_full.py [--derive-fg|--emit-rust]")


if __name__ == "__main__":
    main()
