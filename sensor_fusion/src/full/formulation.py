"""Symbolic model generator for the Full ECEF EKF.

This file is the source of truth for the Rust snippets included by
`sensor_fusion/src/full/generated.rs`. Normal Rust builds do not execute this
script; run it only when changing the Full mathematical model:

    python sensor_fusion/src/full/formulation.py --emit-rust

Frame and quaternion convention:

- `e`: Earth-Centered, Earth-Fixed frame.
- `v`: vehicle frame, x forward, y right, z down.
- `b`: IMU/body frame.
- `q_ev`: vehicle attitude in ECEF coordinates. `R(q_ev) = C_ev`, so
  `x_e = C_ev x_v`.
- `q_bv`: physical vehicle-to-body mount. `R(q_bv) = C_bv`, so
  `x_b = C_bv x_v`; `C_vb = C_bv.T` rotates raw IMU samples into vehicle frame.
- Quaternion/DCM composition is active: `R(q1 * q2) = R(q1) R(q2)`.

Full error state order:

    dp_e[3], dv_e[3], dpsi_ev[3],
    accel_bias_b[3], gyro_bias_b[3],
    accel_scale_b[3], gyro_scale_b[3],
    dpsi_bv[3]

Noise input order:

    accel_noise_b[3], gyro_noise_b[3],
    accel_bias_rw_b[3], gyro_bias_rw_b[3],
    accel_scale_rw_b[3], gyro_scale_rw_b[3],
    mount_rw_bv[3]

Unlike Reduced, this script generates only transition/noise matrices, NHC rows,
reset Jacobian, and sparse supports. Full nominal mechanization is implemented
directly in Rust in `sensor_fusion/src/full/mod.rs`, while this symbolic model
derives the matching first-order covariance dynamics around that state layout.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from sympy import Matrix, Symbol, cse, symbols


SCRIPT_DIR = Path(__file__).resolve().parent
CRATE_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(CRATE_DIR))

from code_gen import RustCodeGenerator


GENERATED_RUST_DIR = SCRIPT_DIR / "generated"

WGS84_OMEGA_IE = 7.292115e-5

POS_E = slice(0, 3)
V_E = slice(3, 6)
PSI_EV = slice(6, 9)
B_F = slice(9, 12)
B_W = slice(12, 15)
S_F = slice(15, 18)
S_W = slice(18, 21)
PSI_BV = slice(21, 24)


def quat_to_rot(q):
    """Return active DCM `C_ab = R(q_ab)` for quaternion `[w, x, y, z]`."""

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
    """Skew matrix satisfying `skew(a) b = a x b`."""

    return Matrix(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def create_reference_transition_and_noise():
    """Create Full discrete error transition `Phi` and noise-input `G`.

    The symbolic equations are written in continuous first-order form and then
    discretized as `Phi = I + dt * F_cont` for one IMU interval.

    Important modeling choices:

    - Raw IMU increments are converted to rates by `dv*/dt` and `da*/dt` to
      match Rust's `ImuDelta` input.
    - Bias and scale states live in body frame `b`.
    - `C_vb` rotates raw IMU terms from body into vehicle coordinates.
    - `C_ev` rotates vehicle-frame specific force/angular rate into ECEF for
      velocity and attitude error coupling.
    - Earth rotation coupling uses `WGS84_OMEGA_IE` with the same simplified
      ECEF convention as the Full Rust propagation.

    Returned matrices are consumed by `full/generated.rs::error_transition`.
    """

    dt = Symbol("dt", real=True)
    q_ev = Matrix(symbols("q0 q1 q2 q3", real=True))
    q_bv = Matrix(symbols("q_bv0 q_bv1 q_bv2 q_bv3", real=True))
    v_e = Matrix(symbols("vn ve vd", real=True))
    b_f = Matrix(symbols("bax bay baz", real=True))
    b_w = Matrix(symbols("bgx bgy bgz", real=True))
    s_f = Matrix(symbols("sax say saz", real=True))
    s_w = Matrix(symbols("sgx sgy sgz", real=True))
    accel_raw = Matrix([Symbol("dvx", real=True) / dt, Symbol("dvy", real=True) / dt, Symbol("dvz", real=True) / dt])
    gyro_raw = Matrix([Symbol("dax", real=True) / dt, Symbol("day", real=True) / dt, Symbol("daz", real=True) / dt])

    c_ev = quat_to_rot(q_ev)
    c_vb = quat_to_rot(q_bv).T
    f_b = Matrix(
        [
            s_f[0] * accel_raw[0] + b_f[0],
            s_f[1] * accel_raw[1] + b_f[1],
            s_f[2] * accel_raw[2] + b_f[2],
        ]
    )
    w_b = Matrix(
        [
            s_w[0] * gyro_raw[0] + b_w[0],
            s_w[1] * gyro_raw[1] + b_w[1],
            s_w[2] * gyro_raw[2] + b_w[2],
        ]
    )
    f_v = c_vb * f_b
    w_v = c_vb * w_b

    phi_cont = Matrix.zeros(24, 24)
    phi_cont[POS_E, V_E] = Matrix.eye(3)
    phi_cont[V_E, V_E] = skew(Matrix([0, 0, -2 * WGS84_OMEGA_IE]))
    phi_cont[V_E, PSI_EV] = skew(c_ev * (-f_v))
    phi_cont[V_E, B_F] = c_ev * c_vb
    phi_cont[V_E, S_F] = c_ev * c_vb * Matrix.diag(accel_raw[0], accel_raw[1], accel_raw[2])
    phi_cont[V_E, PSI_BV] = c_ev * c_vb * skew(f_b)
    phi_cont[PSI_EV, PSI_EV] = skew(Matrix([0, 0, -WGS84_OMEGA_IE]))
    phi_cont[PSI_EV, B_W] = c_ev * c_vb
    phi_cont[PSI_EV, S_W] = c_ev * c_vb * Matrix.diag(gyro_raw[0], gyro_raw[1], gyro_raw[2])
    phi_cont[PSI_EV, PSI_BV] = c_ev * c_vb * skew(w_b)

    phi = Matrix.eye(24) + dt * phi_cont

    g = Matrix.zeros(24, 21)
    g[V_E, 0:3] = c_ev * c_vb
    g[PSI_EV, 3:6] = c_ev * c_vb
    g[B_F, 6:9] = Matrix.eye(3)
    g[B_W, 9:12] = Matrix.eye(3)
    g[S_F, 12:15] = Matrix.eye(3)
    g[S_W, 15:18] = Matrix.eye(3)
    g[PSI_BV, 18:21] = Matrix.eye(3)

    return phi, g


def create_reference_nhc_rows():
    """Create Full vehicle-frame lateral/vertical NHC estimates and rows.

    The predicted vehicle velocity is `v_c = C_ve v_e`, where `C_ve = C_ev.T`.
    The generated rows are for the lateral (`v_c.y`) and vertical (`v_c.z`)
    constraints. They contain derivatives with respect to ECEF velocity and
    vehicle attitude error only; mount coupling reaches NHC through covariance
    generated by propagation.
    """

    q_ev = Matrix(symbols("q0 q1 q2 q3", real=True))
    v_e = Matrix(symbols("vn ve vd", real=True))

    c_ev = quat_to_rot(q_ev)
    c_ve = c_ev.T
    v_c = c_ve * v_e

    h = Matrix.zeros(3, 24)
    h[:, V_E] = c_ve
    h[:, PSI_EV] = c_ve * skew(v_e)
    return v_c, h[1, :], h[2, :]


def write_rust_nhc_row(path: Path, est_expr, h_row: Matrix, prefix: str):
    """Emit a generated Rust NHC row with estimated measurement and `H`."""

    flat_values = Matrix([est_expr] + [h_row[i] for i in range(h_row.shape[1])])
    subexprs, reduced = cse(flat_values, symbols(f"{prefix}_0:2000"), optimizations="basic")
    gen = RustCodeGenerator(str(path))
    gen.print_string("Sub Expressions")
    gen.write_subexpressions(subexprs)
    values = reduced
    if len(values) == 1 and isinstance(values[0], Matrix):
        values = values[0]
    gen.print_string("Estimated Measurement")
    gen.file.write(f"vc_evt = {gen.get_ccode(values[0])};\n\n")
    gen.print_string("Observation Jacobian")
    gen.write_matrix(Matrix(values[1:25]), "H")
    gen.close()


def emit_supports(path: Path, phi: Matrix, g: Matrix, h_y: Matrix, h_z: Matrix):
    """Emit sparse row-support metadata for Full covariance propagation/updates."""

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
            file.write(f"pub const {prefix}_ROW_COUNTS: [usize; ERROR_STATES] = [\n")
            file.write("    " + ", ".join(str(len(row)) for row in supports) + ",\n")
            file.write("];\n")
            file.write(
                f"pub const {prefix}_ROW_COLS: [[usize; {prefix}_MAX_ROW_NONZERO]; ERROR_STATES] = [\n"
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
    """Regenerate all Full Rust fragments from the symbolic model."""

    GENERATED_RUST_DIR.mkdir(parents=True, exist_ok=True)
    phi, g = create_reference_transition_and_noise()
    v_c, h_y, h_z = create_reference_nhc_rows()
    dtheta = Matrix(symbols("dtheta_x dtheta_y dtheta_z", real=True))
    reset = Matrix.eye(3) - skew(dtheta) / 2

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

    gen = RustCodeGenerator(str(GENERATED_RUST_DIR / "attitude_reset_jacobian_generated.rs"))
    gen.print_string("Generated first-order Full quaternion reset Jacobian block")
    subexprs, reduced = cse(reset, symbols("tmp_reset_0:2000"), optimizations="basic")
    gen.write_subexpressions(subexprs)
    gen.write_matrix(Matrix(reduced), "G_reset_theta")
    gen.close()

    write_rust_nhc_row(GENERATED_RUST_DIR / "reference_nhc_y_generated.rs", v_c[1], h_y, "tmp_nhc_y")
    write_rust_nhc_row(GENERATED_RUST_DIR / "reference_nhc_z_generated.rs", v_c[2], h_z, "tmp_nhc_z")


def main():
    """CLI entry point for derivation inspection and Rust generation."""

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
    print("Usage: python3 sensor_fusion/src/full/formulation.py [--derive-fg|--emit-rust]")


if __name__ == "__main__":
    main()
