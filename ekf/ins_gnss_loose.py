import sys
from pathlib import Path

from sympy import Matrix, Symbol, cse, symbols

from code_gen import CodeGenerator


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_C_DIR = SCRIPT_DIR / "c" / "generated_loose"

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


def emit_reference_c():
    GENERATED_C_DIR.mkdir(parents=True, exist_ok=True)
    phi, g = create_reference_transition_and_noise()
    v_c, h_y, h_z = create_reference_nhc_rows()

    write_matrix(GENERATED_C_DIR / "reference_error_transition_generated.c", phi, "F")
    write_matrix(GENERATED_C_DIR / "reference_error_noise_input_generated.c", g, "G")
    write_nhc_row(GENERATED_C_DIR / "reference_nhc_y_generated.c", v_c[1], h_y, "tmp_nhc_y")
    write_nhc_row(GENERATED_C_DIR / "reference_nhc_z_generated.c", v_c[2], h_z, "tmp_nhc_z")


def main():
    args = set(sys.argv[1:])
    if "--emit-c" in args:
        emit_reference_c()
        return
    if "--derive-fg" in args:
        phi, g = create_reference_transition_and_noise()
        print("Phi shape:", phi.shape)
        print("G shape:", g.shape)
        return
    print("Usage: python3 ekf/ins_gnss_loose.py [--derive-fg|--emit-c]")


if __name__ == "__main__":
    main()
