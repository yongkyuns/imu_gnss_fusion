"""Symbolic model generator for the local-NED EKF.

This file is the source of truth for the Rust snippets included by
`sensor_fusion/src/ekf/generated.rs`. Normal Rust builds do not execute this
script; run it only when changing the EKF mathematical model:

    python sensor_fusion/src/ekf/formulation.py --emit-rust

Frame and quaternion convention:

- `n`: local navigation frame, North-East-Down.
- `v`: vehicle frame, x forward, y right, z down.
- `b`: IMU/body frame.
- `q`: vehicle attitude in navigation coordinates, `q_nv`.
  `R(q) = C_nv`, so `x_n = C_nv x_v`.
- `q_bv`: physical vehicle-to-body mount. `R(q_bv) = C_bv`, so
  `x_b = C_bv x_v`; `C_vb = C_bv.T` rotates raw IMU samples into vehicle frame.
- Quaternion multiplication is active composition:
  `R(q1 * q2) = R(q1) R(q2)`.

Nominal state order generated here:

    q_nv[4], v_n[3], p_n[3], gyro_bias_b[3], accel_bias_b[3], q_bv[4]

Error state order generated here:

    dtheta_nv[3], dv_n[3], dp_n[3], dbg_b[3], dba_b[3], dpsi_bv[3]

Noise input order generated here:

    gyro_delta_noise_b[3], accel_delta_noise_b[3],
    gyro_bias_rw_b[3], accel_bias_rw_b[3], mount_rw_bv[3]

The generated EKF model intentionally uses scalar observation files for
GNSS, stationary-gravity, and vehicle-frame velocity/NHC updates. Each scalar
file contains H, K, and S for a one-dimensional Kalman update so runtime code can
avoid a dense measurement inverse.
"""

import sys
from pathlib import Path

from sympy import Matrix, Symbol, cse, symbols


SCRIPT_DIR = Path(__file__).resolve().parent
CRATE_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(CRATE_DIR))

from code_gen import RustCodeGenerator


GENERATED_RUST_DIR = SCRIPT_DIR / "generated"


def create_symmetric_cov_matrix(n):
    """Create symbolic covariance matrix with one symbol per upper-triangle cell.

    The generated scalar-update expressions use this matrix to derive `H`,
    `K = P H^T / S`, and `S = H P H^T + R`. Lower-triangle entries alias the
    same upper-triangle symbols so the symbolic covariance is exactly symmetric.
    """

    def create_cov(i, j):
        if j >= i:
            return Symbol(f"P[{i}][{j}]", real=True)
        return 0

    p = Matrix(n, n, create_cov)
    for i in range(n):
        for j in range(n):
            if i > j:
                p[i, j] = p[j, i]
    return p


def generate_observation_equations(p_cov, state, observation, variance, varname, linearization_subs=None):
    """Derive one scalar EKF observation update.

    Args:
        p_cov: Symbolic covariance matrix for the EKF error state.
        state: EKF error-state vector.
        observation: Predicted scalar measurement expressed in terms of the
            perturbed/true state.
        variance: Measurement variance symbol emitted into generated Rust.
        varname: Prefix for common-subexpression temporaries.
        linearization_subs: Optional substitutions, usually zero-error
            substitutions, applied after differentiating.

    Returns:
        SymPy CSE tuple for `[H, K, S]`, where `H` is row-vector Jacobian, `K` is
        the covariance-weighted scalar gain, and `S` is innovation variance.
    """

    h = Matrix([observation]).jacobian(state)
    if linearization_subs is not None:
        h = h.subs(linearization_subs)
    innov_var = h * p_cov * h.T + Matrix([variance])
    k = p_cov * h.T / innov_var[0, 0]
    expr = cse(Matrix.vstack(h.transpose(), k, Matrix([innov_var[0, 0]])),
               symbols(f"{varname}0:1000"),
               optimizations="basic")
    return expr


def write_observation_equations(path, equations, state_dim):
    """Emit generated Rust assignments for a scalar observation update."""

    gen = RustCodeGenerator(str(path))
    gen.print_string("Sub Expressions")
    gen.write_subexpressions(equations[0])
    values = equations[1]
    if len(values) == 1 and isinstance(values[0], Matrix):
        values = values[0]
    gen.print_string("Observation Jacobians")
    gen.write_matrix(Matrix(values[0:state_dim]), "H")
    gen.print_string("Kalman gains")
    gen.write_matrix(Matrix(values[state_dim:2 * state_dim]), "K")
    gen.print_string("Innovation Variance")
    gen.file.write(f"S = {gen.get_ccode(values[2 * state_dim])};\n")
    gen.close()


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


def quat_mult(p, q):
    """Hamilton product with active-composition convention."""

    return Matrix(
        [
            p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
            p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
            p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
            p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
        ]
    )


def quat_conj(q):
    """Quaternion conjugate; inverse for unit quaternions."""

    return Matrix([q[0], -q[1], -q[2], -q[3]])


def delta_quat(dtheta):
    """First-order small-angle quaternion `[1, 0.5*dtheta]`."""

    return Matrix([1, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]])


def skew(v):
    """Skew matrix satisfying `skew(a) b = a x b`."""

    return Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def propagate_nominal(q, v, p, bg, ba, q_bv, d_ang, d_vel, dt, g_n):
    """EKF nominal mechanization used for generated prediction.

    Inputs `d_ang` and `d_vel` are raw IMU increments in body frame `b` over
    `dt`. Bias terms are body-frame rates/accelerations, so the bias increments
    are `bias * dt`. The mount inverse `C_vb` rotates the bias-corrected
    increments into the vehicle frame before attitude/velocity propagation.

    This symbolic model generates only the nominal quantities updated by the
    generated prediction snippet: attitude, velocity, and position. Bias and
    mount nominal states are constant during propagation; their uncertainty
    evolves through the generated error-state noise model.
    """

    r_v_to_n = quat_to_rot(q)
    r_v_to_b = quat_to_rot(q_bv)
    r_b_to_v = r_v_to_b.T
    d_ang_true = r_b_to_v * (d_ang - bg * dt)
    d_vel_true_v = r_b_to_v * (d_vel - ba * dt)

    q_new = quat_mult(q, delta_quat(d_ang_true))
    v_new = v + r_v_to_n * d_vel_true_v + g_n * dt
    p_new = p + v * dt
    bg_new = bg
    ba_new = ba

    return q_new, v_new, p_new, bg_new, ba_new


def inject_true_state(q, v, p, bg, ba, q_bv, dtheta, dv, dp, dbg, dba, dpsi_bv):
    """Apply EKF error-state perturbations to a nominal state.

    Attitude and mount use left small-angle perturbations:
    `q_true = q_nom * dq(dtheta)` for vehicle attitude and
    `q_bv_true = dq(dpsi_bv) * q_bv_nom` for mount. This matches the reset and
    covariance layout used by `sensor_fusion/src/ekf/mod.rs`.
    """

    q_true = quat_mult(q, delta_quat(dtheta))
    v_true = v + dv
    p_true = p + dp
    bg_true = bg + dbg
    ba_true = ba + dba
    q_bv_true = quat_mult(delta_quat(dpsi_bv), q_bv)
    return q_true, v_true, p_true, bg_true, ba_true, q_bv_true


def extract_error_state(q_nom, v_nom, p_nom, bg_nom, ba_nom, q_bv_nom, q_true, v_true, p_true, bg_true, ba_true, q_bv_true):
    """Recover first-order EKF error state from nominal and true states."""

    dq = quat_mult(quat_conj(q_nom), q_true)
    dtheta = Matrix([2 * dq[1], 2 * dq[2], 2 * dq[3]])
    dv = v_true - v_nom
    dp = p_true - p_nom
    dbg = bg_true - bg_nom
    dba = ba_true - ba_nom
    dq_bv = quat_mult(q_bv_true, quat_conj(q_bv_nom))
    dpsi_bv = Matrix([2 * dq_bv[1], 2 * dq_bv[2], 2 * dq_bv[3]])
    return Matrix.vstack(dtheta, dv, dp, dbg, dba, dpsi_bv)


def build_symbolic_model():
    """Build symbolic variables and nominal EKF propagation graph."""

    dt = Symbol("dt", real=True)
    g = Symbol("g", real=True)

    q = Matrix(symbols("q0 q1 q2 q3", real=True))
    v = Matrix(symbols("vn ve vd", real=True))
    p = Matrix(symbols("pn pe pd", real=True))
    bg = Matrix(symbols("bgx bgy bgz", real=True))
    ba = Matrix(symbols("bax bay baz", real=True))
    q_bv = Matrix(symbols("q_bv0 q_bv1 q_bv2 q_bv3", real=True))

    dtheta = Matrix(symbols("dtheta_x dtheta_y dtheta_z", real=True))
    dv = Matrix(symbols("dv_n dv_e dv_d", real=True))
    dp = Matrix(symbols("dp_n dp_e dp_d", real=True))
    dbg = Matrix(symbols("dbg_x dbg_y dbg_z", real=True))
    dba = Matrix(symbols("dba_x dba_y dba_z", real=True))
    dpsi_bv = Matrix(symbols("dpsi_bv_x dpsi_bv_y dpsi_bv_z", real=True))

    d_ang = Matrix(symbols("dax day daz", real=True))
    d_vel = Matrix(symbols("dvx dvy dvz", real=True))

    g_n = Matrix([0, 0, g])
    q_new, v_new, p_new, bg_new, ba_new = propagate_nominal(
        q, v, p, bg, ba, q_bv, d_ang, d_vel, dt, g_n
    )

    x_nom = Matrix.vstack(q, v, p, bg, ba, q_bv)
    x_nom_new = Matrix.vstack(q_new, v_new, p_new, bg_new, ba_new, q_bv)
    dx = Matrix.vstack(dtheta, dv, dp, dbg, dba, dpsi_bv)

    return {
        "dt": dt,
        "q": q,
        "v": v,
        "p": p,
        "bg": bg,
        "ba": ba,
        "q_bv": q_bv,
        "dtheta": dtheta,
        "dv": dv,
        "dp": dp,
        "dbg": dbg,
        "dba": dba,
        "dpsi_bv": dpsi_bv,
        "d_ang": d_ang,
        "d_vel": d_vel,
        "g_n": g_n,
        "x_nom": x_nom,
        "x_nom_new": x_nom_new,
        "dx": dx,
        "state_dim_nominal": x_nom.shape[0],
        "state_dim_error": dx.shape[0],
        "attitude_reset_jacobian": Matrix.eye(3) - skew(dtheta) / 2,
    }


def derive_error_dynamics():
    """Derive discrete EKF error transition `F` and noise-input `G`.

    The method is a standard perturb-propagate-linearize construction:

    1. Build a nominal state and a first-order perturbed "true" state.
    2. Propagate both through the same nominal mechanization.
    3. Add process noise to IMU increments, bias random walks, and mount random
       walk.
    4. Extract the post-propagation error state.
    5. Differentiate with respect to previous error state and noise, then
       evaluate at zero error/noise.

    The resulting `F` and `G` are discrete-time matrices for one IMU increment,
    not continuous-time matrices.
    """

    model = build_symbolic_model()

    dt = model["dt"]
    bg = model["bg"]
    ba = model["ba"]
    dx = model["dx"]
    dtheta = model["dtheta"]
    dv = model["dv"]
    dp = model["dp"]
    dbg = model["dbg"]
    dba = model["dba"]
    dpsi_bv = model["dpsi_bv"]
    d_ang = model["d_ang"]
    d_vel = model["d_vel"]
    g_n = model["g_n"]

    n_dang = Matrix(symbols("n_dax n_day n_daz", real=True))
    n_dvel = Matrix(symbols("n_dvx n_dvy n_dvz", real=True))
    n_dbg = Matrix(symbols("n_dbg_x n_dbg_y n_dbg_z", real=True))
    n_dba = Matrix(symbols("n_dba_x n_dba_y n_dba_z", real=True))
    n_mount = Matrix(symbols("n_mount_x n_mount_y n_mount_z", real=True))
    w = Matrix.vstack(n_dang, n_dvel, n_dbg, n_dba, n_mount)

    q_true, v_true, p_true, bg_true, ba_true, q_bv_true = inject_true_state(
        model["q"],
        model["v"],
        model["p"],
        bg,
        ba,
        model["q_bv"],
        dtheta,
        dv,
        dp,
        dbg,
        dba,
        dpsi_bv,
    )

    q_nom_new, v_nom_new, p_nom_new, bg_nom_new, ba_nom_new = propagate_nominal(
        model["q"], model["v"], model["p"], bg, ba, model["q_bv"], d_ang, d_vel, dt, g_n
    )
    q_true_new, v_true_new, p_true_new, bg_true_new, ba_true_new = propagate_nominal(
        q_true,
        v_true,
        p_true,
        bg_true,
        ba_true,
        q_bv_true,
        d_ang + n_dang,
        d_vel + n_dvel,
        dt,
        g_n,
    )
    bg_true_new += n_dbg * dt
    ba_true_new += n_dba * dt
    q_bv_nom_new = model["q_bv"]
    q_bv_true_new = quat_mult(delta_quat(n_mount), q_bv_true)

    dx_next = extract_error_state(
        q_nom_new,
        v_nom_new,
        p_nom_new,
        bg_nom_new,
        ba_nom_new,
        q_bv_nom_new,
        q_true_new,
        v_true_new,
        p_true_new,
        bg_true_new,
        ba_true_new,
        q_bv_true_new,
    )

    zero_subs = {symbol: 0 for symbol in list(dx) + list(w)}
    f = dx_next.jacobian(dx).subs(zero_subs)
    g = dx_next.jacobian(w).subs(zero_subs)

    return {
        **model,
        "w": w,
        "dx_next": dx_next,
        "F": f,
        "G": g,
    }


def derive_measurement_model():
    """Derive scalar EKF observation models.

    Generated scalar observations:

    - GNSS position in local NED: `gps_pos_n/e/d`.
    - GNSS velocity in local NED: `gps_vel_n/e/d`.
    - Stationary gravity cues in vehicle frame: `stationary_accel_x/y`.
    - Vehicle-frame velocity/NHC rows: `body_vel_x/y/z`.

    All rows are linearized at zero error. Runtime code passes the residual
    `z - h(x_nom)` and the generated row/gain/innovation variance to the shared
    scalar-update path.
    """

    model = build_symbolic_model()
    p_cov = create_symmetric_cov_matrix(model["state_dim_error"])
    zero_error_subs = {symbol: 0 for symbol in list(model["dx"])}
    q_true, v_true, p_true, bg_true, ba_true, q_bv_true = inject_true_state(
        model["q"],
        model["v"],
        model["p"],
        model["bg"],
        model["ba"],
        model["q_bv"],
        model["dtheta"],
        model["dv"],
        model["dp"],
        model["dbg"],
        model["dba"],
        model["dpsi_bv"],
    )
    r_true_to_n = quat_to_rot(q_true)
    v_true_v = r_true_to_n.T * v_true
    g_true_v = r_true_to_n.T * model["g_n"]
    stationary_gravity_v = -g_true_v
    return {
        **model,
        "P_cov": p_cov,
        "gps_pos_n": generate_observation_equations(p_cov, model["dx"], p_true[0], Symbol("R_POS_N", real=True), "tmp_hk_pos_n", zero_error_subs),
        "gps_pos_e": generate_observation_equations(p_cov, model["dx"], p_true[1], Symbol("R_POS_E", real=True), "tmp_hk_pos_e", zero_error_subs),
        "gps_pos_d": generate_observation_equations(p_cov, model["dx"], p_true[2], Symbol("R_POS_D", real=True), "tmp_hk_pos_d", zero_error_subs),
        "gps_vel_n": generate_observation_equations(p_cov, model["dx"], v_true[0], Symbol("R_VEL_N", real=True), "tmp_hk_vel_n", zero_error_subs),
        "gps_vel_e": generate_observation_equations(p_cov, model["dx"], v_true[1], Symbol("R_VEL_E", real=True), "tmp_hk_vel_e", zero_error_subs),
        "gps_vel_d": generate_observation_equations(p_cov, model["dx"], v_true[2], Symbol("R_VEL_D", real=True), "tmp_hk_vel_d", zero_error_subs),
        "stationary_accel_x": generate_observation_equations(p_cov, model["dx"], stationary_gravity_v[0], Symbol("R_STATIONARY_ACCEL", real=True), "tmp_hk_stat_ax", zero_error_subs),
        "stationary_accel_y": generate_observation_equations(p_cov, model["dx"], stationary_gravity_v[1], Symbol("R_STATIONARY_ACCEL", real=True), "tmp_hk_stat_ay", zero_error_subs),
        "body_vel_x": generate_observation_equations(p_cov, model["dx"], v_true_v[0], Symbol("R_BODY_VEL", real=True), "tmp_hk_body_x", zero_error_subs),
        "body_vel_y": generate_observation_equations(p_cov, model["dx"], v_true_v[1], Symbol("R_BODY_VEL", real=True), "tmp_hk_body_y", zero_error_subs),
        "body_vel_z": generate_observation_equations(p_cov, model["dx"], v_true_v[2], Symbol("R_BODY_VEL", real=True), "tmp_hk_body_z", zero_error_subs),
    }


def emit_cse_matrix_assignments(path, title, matrix, variable_name, symbol_prefix, is_symmetric=False):
    """Emit CSE-optimized Rust assignments for a matrix."""

    expr = cse(matrix, symbols(f"{symbol_prefix}0:4000"), optimizations="basic")
    gen = RustCodeGenerator(str(path))
    gen.print_string(title)
    gen.write_subexpressions(expr[0])
    values = expr[1]
    if len(values) == 1 and isinstance(values[0], Matrix):
        values = values[0]
    gen.write_matrix(Matrix(values), variable_name, is_symmetric)
    gen.close()


def emit_matrix_supports(path, matrices):
    """Emit sparse row-support metadata for generated covariance propagation."""

    with open(path, "w") as file:
        file.write("// Generated EKF transition sparsity supports\n")
        for prefix, matrix, rows in matrices:
            supports = []
            max_len = 0
            for i in range(matrix.shape[0]):
                row_support = [j for j in range(matrix.shape[1]) if matrix[i, j] != 0]
                supports.append(row_support)
                max_len = max(max_len, len(row_support))
            file.write(f"pub const {prefix}_MAX_ROW_NONZERO: usize = {max_len};\n")
            file.write(f"pub const {prefix}_ROW_COUNTS: [usize; {rows}] = [\n")
            file.write("    " + ", ".join(str(len(row)) for row in supports) + ",\n")
            file.write("];\n")
            file.write(
                f"pub const {prefix}_ROW_COLS: [[usize; {prefix}_MAX_ROW_NONZERO]; {rows}] = [\n"
            )
            for row in supports:
                padded = row + [0] * (max_len - len(row))
                file.write("    [" + ", ".join(str(col) for col in padded) + "],\n")
            file.write("];\n\n")


def emit_nominal_prediction_rust(model):
    """Emit the EKF nominal prediction snippet.

    Only attitude, velocity, and position assignments are generated here because
    bias and mount nominal states are unchanged by propagation.
    """

    q_new = model["x_nom_new"][0:4, 0]
    v_new = model["x_nom_new"][4:7, 0]
    p_new = model["x_nom_new"][7:10, 0]
    state_new = Matrix.vstack(q_new, v_new, p_new)
    expr = cse(state_new, symbols("tmp_pred0:2000"), optimizations="basic")
    pred_path = GENERATED_RUST_DIR / "nominal_prediction_generated.rs"
    gen = RustCodeGenerator(str(pred_path))
    gen.print_string("Generated EKF nominal-state prediction")
    gen.write_subexpressions(expr[0])

    values = expr[1]
    if len(values) == 1 and isinstance(values[0], Matrix):
        values = values[0]

    state_names = ["q0", "q1", "q2", "q3", "vn", "ve", "vd", "pn", "pe", "pd"]
    write_string = ""
    for i, name in enumerate(state_names):
        write_string += f"nominal.{name} = {gen.get_ccode(values[i])};\n"
    gen.file.write(write_string)
    gen.close()


def emit_generated_rust():
    """Regenerate all EKF Rust fragments from the symbolic model."""

    model = derive_error_dynamics()
    meas = derive_measurement_model()
    GENERATED_RUST_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = GENERATED_RUST_DIR / "nominal_prediction_generated.rs"
    f_path = GENERATED_RUST_DIR / "error_transition_generated.rs"
    g_path = GENERATED_RUST_DIR / "error_noise_input_generated.rs"
    support_path = GENERATED_RUST_DIR / "error_transition_support_generated.rs"
    reset_path = GENERATED_RUST_DIR / "attitude_reset_jacobian_generated.rs"
    gps_pos_n_path = GENERATED_RUST_DIR / "gps_pos_n_generated.rs"
    gps_pos_e_path = GENERATED_RUST_DIR / "gps_pos_e_generated.rs"
    gps_pos_d_path = GENERATED_RUST_DIR / "gps_pos_d_generated.rs"
    gps_vel_n_path = GENERATED_RUST_DIR / "gps_vel_n_generated.rs"
    gps_vel_e_path = GENERATED_RUST_DIR / "gps_vel_e_generated.rs"
    gps_vel_d_path = GENERATED_RUST_DIR / "gps_vel_d_generated.rs"
    stationary_accel_x_path = GENERATED_RUST_DIR / "stationary_accel_x_generated.rs"
    stationary_accel_y_path = GENERATED_RUST_DIR / "stationary_accel_y_generated.rs"
    body_vel_x_path = GENERATED_RUST_DIR / "body_vel_x_generated.rs"
    body_vel_y_path = GENERATED_RUST_DIR / "body_vel_y_generated.rs"
    body_vel_z_path = GENERATED_RUST_DIR / "body_vel_z_generated.rs"

    emit_nominal_prediction_rust(model)
    emit_cse_matrix_assignments(
        f_path,
        "Generated EKF error-state transition matrix",
        model["F"],
        "F",
        "tmp_f",
    )
    emit_cse_matrix_assignments(
        g_path,
        "Generated EKF error-state noise input matrix",
        model["G"],
        "G",
        "tmp_g",
    )
    emit_matrix_supports(
        support_path,
        [
            ("F", model["F"], "ERROR_STATES"),
            ("G", model["G"], "ERROR_STATES"),
        ],
    )
    emit_cse_matrix_assignments(
        reset_path,
        "Generated first-order EKF attitude reset Jacobian block",
        model["attitude_reset_jacobian"],
        "G_reset_theta",
        "tmp_reset",
    )
    state_dim = model["state_dim_error"]
    write_observation_equations(gps_pos_n_path, meas["gps_pos_n"], state_dim)
    write_observation_equations(gps_pos_e_path, meas["gps_pos_e"], state_dim)
    write_observation_equations(gps_pos_d_path, meas["gps_pos_d"], state_dim)
    write_observation_equations(gps_vel_n_path, meas["gps_vel_n"], state_dim)
    write_observation_equations(gps_vel_e_path, meas["gps_vel_e"], state_dim)
    write_observation_equations(gps_vel_d_path, meas["gps_vel_d"], state_dim)
    write_observation_equations(stationary_accel_x_path, meas["stationary_accel_x"], state_dim)
    write_observation_equations(stationary_accel_y_path, meas["stationary_accel_y"], state_dim)
    write_observation_equations(body_vel_x_path, meas["body_vel_x"], state_dim)
    write_observation_equations(body_vel_y_path, meas["body_vel_y"], state_dim)
    write_observation_equations(body_vel_z_path, meas["body_vel_z"], state_dim)

    for path in [
        pred_path,
        f_path,
        g_path,
        reset_path,
        gps_pos_n_path,
        gps_pos_e_path,
        gps_pos_d_path,
        gps_vel_n_path,
        gps_vel_e_path,
        gps_vel_d_path,
        stationary_accel_x_path,
        stationary_accel_y_path,
        body_vel_x_path,
        body_vel_y_path,
        body_vel_z_path,
    ]:
        print("Wrote:", path)


if __name__ == "__main__":
    if "--emit-rust" in sys.argv:
        emit_generated_rust()
    elif "--derive-fg" in sys.argv:
        model = derive_error_dynamics()
        print("F shape:", model["F"].shape)
        print("G shape:", model["G"].shape)
    else:
        model = build_symbolic_model()
        print("EKF nominal dimension:", model["state_dim_nominal"])
        print("EKF error dimension:", model["state_dim_error"])
