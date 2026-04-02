import sys
from pathlib import Path

from sympy import Matrix, Symbol, cse, symbols

from code_gen import CodeGenerator


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_C_DIR = SCRIPT_DIR / "c" / "generated_eskf"


def create_symmetric_cov_matrix(n):
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
    h = Matrix([observation]).jacobian(state)
    if linearization_subs is not None:
        h = h.subs(linearization_subs)
    innov_var = h * p_cov * h.T + Matrix([variance])
    k = p_cov * h.T / innov_var[0, 0]
    expr = cse(Matrix([h.transpose(), k]), symbols(f"{varname}0:1000"), optimizations="basic")
    return expr


def write_observation_equations(path, equations):
    gen = CodeGenerator(str(path))
    gen.print_string("Sub Expressions")
    gen.write_subexpressions(equations[0])
    gen.print_string("Observation Jacobians")
    gen.write_matrix(Matrix(equations[1][0][0:15]), "H")
    gen.print_string("Kalman gains")
    gen.write_matrix(Matrix(equations[1][0][15:]), "K")
    gen.close()


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


def quat_mult(p, q):
    return Matrix(
        [
            p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
            p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
            p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
            p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
        ]
    )


def quat_conj(q):
    return Matrix([q[0], -q[1], -q[2], -q[3]])


def delta_quat(dtheta):
    return Matrix([1, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]])


def skew(v):
    return Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def propagate_nominal(q, v, p, bg, ba, d_ang, d_vel, dt, g_n):
    r_to_n = quat_to_rot(q)
    d_ang_true = d_ang - bg * dt
    d_vel_true_b = d_vel - ba * dt + r_to_n.T * g_n * dt

    q_new = quat_mult(q, delta_quat(d_ang_true))
    v_new = v + r_to_n * d_vel_true_b
    p_new = p + v * dt
    bg_new = bg
    ba_new = ba

    return q_new, v_new, p_new, bg_new, ba_new


def inject_true_state(q, v, p, bg, ba, dtheta, dv, dp, dbg, dba):
    q_true = quat_mult(q, delta_quat(dtheta))
    v_true = v + dv
    p_true = p + dp
    bg_true = bg + dbg
    ba_true = ba + dba
    return q_true, v_true, p_true, bg_true, ba_true


def extract_error_state(q_nom, v_nom, p_nom, bg_nom, ba_nom, q_true, v_true, p_true, bg_true, ba_true):
    dq = quat_mult(quat_conj(q_nom), q_true)
    dtheta = Matrix([2 * dq[1], 2 * dq[2], 2 * dq[3]])
    dv = v_true - v_nom
    dp = p_true - p_nom
    dbg = bg_true - bg_nom
    dba = ba_true - ba_nom
    return Matrix.vstack(dtheta, dv, dp, dbg, dba)


def build_symbolic_model():
    dt = Symbol("dt", real=True)
    g = Symbol("g", real=True)

    q = Matrix(symbols("q0 q1 q2 q3", real=True))
    v = Matrix(symbols("vn ve vd", real=True))
    p = Matrix(symbols("pn pe pd", real=True))
    bg = Matrix(symbols("bgx bgy bgz", real=True))
    ba = Matrix(symbols("bax bay baz", real=True))

    dtheta = Matrix(symbols("dtheta_x dtheta_y dtheta_z", real=True))
    dv = Matrix(symbols("dv_n dv_e dv_d", real=True))
    dp = Matrix(symbols("dp_n dp_e dp_d", real=True))
    dbg = Matrix(symbols("dbg_x dbg_y dbg_z", real=True))
    dba = Matrix(symbols("dba_x dba_y dba_z", real=True))

    d_ang = Matrix(symbols("dax day daz", real=True))
    d_vel = Matrix(symbols("dvx dvy dvz", real=True))

    g_n = Matrix([0, 0, g])
    q_new, v_new, p_new, bg_new, ba_new = propagate_nominal(
        q, v, p, bg, ba, d_ang, d_vel, dt, g_n
    )

    x_nom = Matrix.vstack(q, v, p, bg, ba)
    x_nom_new = Matrix.vstack(q_new, v_new, p_new, bg_new, ba_new)
    dx = Matrix.vstack(dtheta, dv, dp, dbg, dba)

    return {
        "dt": dt,
        "q": q,
        "v": v,
        "p": p,
        "bg": bg,
        "ba": ba,
        "dtheta": dtheta,
        "dv": dv,
        "dp": dp,
        "dbg": dbg,
        "dba": dba,
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
    model = build_symbolic_model()

    dt = model["dt"]
    q = model["q"]
    v = model["v"]
    p = model["p"]
    bg = model["bg"]
    ba = model["ba"]
    dx = model["dx"]
    dtheta = model["dtheta"]
    dv = model["dv"]
    dp = model["dp"]
    dbg = model["dbg"]
    dba = model["dba"]
    d_ang = model["d_ang"]
    d_vel = model["d_vel"]
    g_n = model["g_n"]

    n_dang = Matrix(symbols("n_dax n_day n_daz", real=True))
    n_dvel = Matrix(symbols("n_dvx n_dvy n_dvz", real=True))
    n_dbg = Matrix(symbols("n_dbg_x n_dbg_y n_dbg_z", real=True))
    n_dba = Matrix(symbols("n_dba_x n_dba_y n_dba_z", real=True))
    w = Matrix.vstack(n_dang, n_dvel, n_dbg, n_dba)

    q_true, v_true, p_true, bg_true, ba_true = inject_true_state(
        q, v, p, bg, ba, dtheta, dv, dp, dbg, dba
    )

    q_nom_new, v_nom_new, p_nom_new, bg_nom_new, ba_nom_new = propagate_nominal(
        q, v, p, bg, ba, d_ang, d_vel, dt, g_n
    )
    q_true_new, v_true_new, p_true_new, bg_true_new, ba_true_new = propagate_nominal(
        q_true,
        v_true,
        p_true,
        bg_true,
        ba_true,
        d_ang + n_dang,
        d_vel + n_dvel,
        dt,
        g_n,
    )
    bg_true_new += n_dbg * dt
    ba_true_new += n_dba * dt

    dx_next = extract_error_state(
        q_nom_new,
        v_nom_new,
        p_nom_new,
        bg_nom_new,
        ba_nom_new,
        q_true_new,
        v_true_new,
        p_true_new,
        bg_true_new,
        ba_true_new,
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
    model = build_symbolic_model()
    p_cov = create_symmetric_cov_matrix(model["state_dim_error"])
    zero_error_subs = {symbol: 0 for symbol in list(model["dx"])}
    q_true, v_true, p_true, bg_true, ba_true = inject_true_state(
        model["q"],
        model["v"],
        model["p"],
        model["bg"],
        model["ba"],
        model["dtheta"],
        model["dv"],
        model["dp"],
        model["dbg"],
        model["dba"],
    )
    r_true_to_n = quat_to_rot(q_true)
    v_true_b = r_true_to_n.T * v_true
    return {
        **model,
        "P_cov": p_cov,
        "gps_pos_n": generate_observation_equations(p_cov, model["dx"], p_true[0], Symbol("R_POS_N", real=True), "ESKF_HK_POS_N", zero_error_subs),
        "gps_pos_e": generate_observation_equations(p_cov, model["dx"], p_true[1], Symbol("R_POS_E", real=True), "ESKF_HK_POS_E", zero_error_subs),
        "gps_pos_d": generate_observation_equations(p_cov, model["dx"], p_true[2], Symbol("R_POS_D", real=True), "ESKF_HK_POS_D", zero_error_subs),
        "gps_vel_n": generate_observation_equations(p_cov, model["dx"], v_true[0], Symbol("R_VEL_N", real=True), "ESKF_HK_VEL_N", zero_error_subs),
        "gps_vel_e": generate_observation_equations(p_cov, model["dx"], v_true[1], Symbol("R_VEL_E", real=True), "ESKF_HK_VEL_E", zero_error_subs),
        "gps_vel_d": generate_observation_equations(p_cov, model["dx"], v_true[2], Symbol("R_VEL_D", real=True), "ESKF_HK_VEL_D", zero_error_subs),
        "body_vel_y": generate_observation_equations(p_cov, model["dx"], v_true_b[1], Symbol("R_BODY_VEL", real=True), "ESKF_HK_BODY_Y", zero_error_subs),
        "body_vel_z": generate_observation_equations(p_cov, model["dx"], v_true_b[2], Symbol("R_BODY_VEL", real=True), "ESKF_HK_BODY_Z", zero_error_subs),
    }


def emit_cse_matrix_assignments(path, title, matrix, variable_name, symbol_prefix, is_symmetric=False):
    expr = cse(matrix, symbols(f"{symbol_prefix}0:4000"), optimizations="basic")
    gen = CodeGenerator(str(path))
    gen.print_string(title)
    gen.write_subexpressions(expr[0])
    values = expr[1]
    if len(values) == 1 and isinstance(values[0], Matrix):
        values = values[0]
    gen.write_matrix(Matrix(values), variable_name, is_symmetric)
    gen.close()


def emit_nominal_prediction_c(model):
    q_new = model["x_nom_new"][0:4, 0]
    v_new = model["x_nom_new"][4:7, 0]
    p_new = model["x_nom_new"][7:10, 0]
    state_new = Matrix.vstack(q_new, v_new, p_new)
    expr = cse(state_new, symbols("ESKF_PRED0:2000"), optimizations="basic")
    pred_path = GENERATED_C_DIR / "nominal_prediction_generated.c"
    gen = CodeGenerator(str(pred_path))
    gen.print_string("Generated ESKF nominal-state prediction")
    gen.write_subexpressions(expr[0])

    values = expr[1]
    if len(values) == 1 and isinstance(values[0], Matrix):
        values = values[0]

    state_names = ["q0", "q1", "q2", "q3", "vn", "ve", "vd", "pn", "pe", "pd"]
    write_string = ""
    for i, name in enumerate(state_names):
        write_string += f"eskf->nominal.{name} = {gen.get_ccode(values[i])};\n"
    gen.file.write(write_string)
    gen.close()


def emit_generated_c():
    model = derive_error_dynamics()
    meas = derive_measurement_model()
    GENERATED_C_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = GENERATED_C_DIR / "nominal_prediction_generated.c"
    f_path = GENERATED_C_DIR / "error_transition_generated.c"
    g_path = GENERATED_C_DIR / "error_noise_input_generated.c"
    reset_path = GENERATED_C_DIR / "attitude_reset_jacobian_generated.c"
    gps_pos_n_path = GENERATED_C_DIR / "gps_pos_n_generated.c"
    gps_pos_e_path = GENERATED_C_DIR / "gps_pos_e_generated.c"
    gps_pos_d_path = GENERATED_C_DIR / "gps_pos_d_generated.c"
    gps_vel_n_path = GENERATED_C_DIR / "gps_vel_n_generated.c"
    gps_vel_e_path = GENERATED_C_DIR / "gps_vel_e_generated.c"
    gps_vel_d_path = GENERATED_C_DIR / "gps_vel_d_generated.c"
    body_vel_y_path = GENERATED_C_DIR / "body_vel_y_generated.c"
    body_vel_z_path = GENERATED_C_DIR / "body_vel_z_generated.c"

    emit_nominal_prediction_c(model)
    emit_cse_matrix_assignments(
        f_path,
        "Generated ESKF error-state transition matrix",
        model["F"],
        "F",
        "ESKF_F",
    )
    emit_cse_matrix_assignments(
        g_path,
        "Generated ESKF error-state noise input matrix",
        model["G"],
        "G",
        "ESKF_G",
    )
    emit_cse_matrix_assignments(
        reset_path,
        "Generated first-order ESKF attitude reset Jacobian block",
        model["attitude_reset_jacobian"],
        "G_reset_theta",
        "ESKF_RESET",
    )
    write_observation_equations(gps_pos_n_path, meas["gps_pos_n"])
    write_observation_equations(gps_pos_e_path, meas["gps_pos_e"])
    write_observation_equations(gps_pos_d_path, meas["gps_pos_d"])
    write_observation_equations(gps_vel_n_path, meas["gps_vel_n"])
    write_observation_equations(gps_vel_e_path, meas["gps_vel_e"])
    write_observation_equations(gps_vel_d_path, meas["gps_vel_d"])
    write_observation_equations(body_vel_y_path, meas["body_vel_y"])
    write_observation_equations(body_vel_z_path, meas["body_vel_z"])

    print("Wrote:", pred_path)
    print("Wrote:", f_path)
    print("Wrote:", g_path)
    print("Wrote:", reset_path)
    print("Wrote:", gps_pos_n_path)
    print("Wrote:", gps_pos_e_path)
    print("Wrote:", gps_pos_d_path)
    print("Wrote:", gps_vel_n_path)
    print("Wrote:", gps_vel_e_path)
    print("Wrote:", gps_vel_d_path)
    print("Wrote:", body_vel_y_path)
    print("Wrote:", body_vel_z_path)


if __name__ == "__main__":
    if "--emit-c" in sys.argv:
        emit_generated_c()
    elif "--derive-fg" in sys.argv:
        model = derive_error_dynamics()
        print("F shape:", model["F"].shape)
        print("G shape:", model["G"].shape)
    else:
        model = build_symbolic_model()
        print("ESKF nominal dimension:", model["state_dim_nominal"])
        print("ESKF error dimension:", model["state_dim_error"])
