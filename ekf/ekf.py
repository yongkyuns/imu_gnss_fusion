from sympy import symbols, Matrix, Symbol, cse, atan2, sqrt
from code_gen import *
import os


def quat2Rot(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    Rot = Matrix(
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
    return Rot


def quat_mult(p, q):
    r = Matrix(
        [
            p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
            p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
            p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
            p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
        ]
    )
    return r


n_states = 16


def create_symmetric_cov_matrix():
    def create_cov_matrix(i, j):
        if j >= i:
            return Symbol("P[" + str(i) + "][" + str(j) + "]", real=True)
        else:
            return 0

    P = Matrix(n_states, n_states, create_cov_matrix)

    for index in range(n_states):
        for j in range(n_states):
            if index > j:
                P[index, j] = P[j, index]
    return P


def generate_observation_equations(P, state, observation, variance, varname="HK"):
    H = Matrix([observation]).jacobian(state)
    innov_var = H * P * H.T + Matrix([variance])
    assert innov_var.shape[0] == 1
    assert innov_var.shape[1] == 1
    K = P * H.T / innov_var[0, 0]
    extension = "0:1000"
    var_string = varname + extension
    HK_simple = cse(
        Matrix([H.transpose(), K]), symbols(var_string), optimizations="basic"
    )
    return HK_simple


def generate_prediction_code(state_new, state_vector_names, code_gen_obj):
    num_states_to_update = 4 + 3 + 3  # q, v, p
    state_new_subset = state_new[0:num_states_to_update]
    state_names_subset = state_vector_names[0:num_states_to_update]

    pred_simple = cse(state_new_subset, symbols("s_0:1000"), optimizations="basic")

    code_gen_obj.print_string("Equations for state prediction.")
    code_gen_obj.write_subexpressions(pred_simple[0])

    write_string = ""
    predicted_states = pred_simple[1]
    if len(predicted_states) == 1 and isinstance(predicted_states[0], Matrix):
        predicted_states = predicted_states[0]

    for i, name in enumerate(state_names_subset):
        write_string += (
            f"ekf->state.{name} = {code_gen_obj.get_ccode(predicted_states[i])};\n"
        )
    code_gen_obj.file.write(write_string)


def write_equations_to_file(equations, code_generator_id, n_obs):
    if n_obs < 1:
        return
    if n_obs == 1:
        code_generator_id.print_string("Sub Expressions")
        code_generator_id.write_subexpressions(equations[0])
        code_generator_id.print_string("Observation Jacobians")
        code_generator_id.write_matrix(Matrix(equations[1][0][0:n_states]), "Hfusion")
        code_generator_id.print_string("Kalman gains")
        code_generator_id.write_matrix(Matrix(equations[1][0][n_states:]), "Kfusion")
    else:
        pass
    return


def gps_pos_n_observation(P, state, px):
    obs_var = symbols("R_POS_N", real=True)
    observation = px
    equations = generate_observation_equations(P, state, observation, obs_var)
    cg = CodeGenerator("./c/generated/gps_pos_n_generated.c")
    write_equations_to_file(equations, cg, 1)
    cg.close()


def gps_pos_e_observation(P, state, py):
    obs_var = symbols("R_POS_E", real=True)
    observation = py
    equations = generate_observation_equations(P, state, observation, obs_var)
    cg = CodeGenerator("./c/generated/gps_pos_e_generated.c")
    write_equations_to_file(equations, cg, 1)
    cg.close()


def gps_pos_d_observation(P, state, pz):
    obs_var = symbols("R_POS_D", real=True)
    observation = pz
    equations = generate_observation_equations(P, state, observation, obs_var)
    cg = CodeGenerator("./c/generated/gps_pos_d_generated.c")
    write_equations_to_file(equations, cg, 1)
    cg.close()


def gps_speed_observation(P, state, vx, vy):
    obs_var = symbols("R_SPD", real=True)
    observation = sqrt(vx**2 + vy**2 + 1e-6)
    equations = generate_observation_equations(P, state, observation, obs_var)
    cg = CodeGenerator("./c/generated/gps_speed_generated.c")
    write_equations_to_file(equations, cg, 1)
    cg.close()


def gps_heading_observation(P, state, R_to_earth):
    obs_var = symbols("R_YAW", real=True)
    observation = atan2(R_to_earth[1, 0], R_to_earth[0, 0])
    equations = generate_observation_equations(P, state, observation, obs_var)
    cg = CodeGenerator("./c/generated/gps_heading_generated.c")
    write_equations_to_file(equations, cg, 1)
    cg.close()


def body_vel_y_observation(P, state, R_to_earth, v):
    obs_var = symbols("R_BODY_VEL", real=True)
    v_body = R_to_earth.T * v
    observation = v_body[1]
    equations = generate_observation_equations(P, state, observation, obs_var)
    cg = CodeGenerator("./c/generated/body_vel_y_generated.c")
    write_equations_to_file(equations, cg, 1)
    cg.close()


def body_vel_z_observation(P, state, R_to_earth, v):
    obs_var = symbols("R_BODY_VEL", real=True)
    v_body = R_to_earth.T * v
    observation = v_body[2]
    equations = generate_observation_equations(P, state, observation, obs_var)
    cg = CodeGenerator("./c/generated/body_vel_z_generated.c")
    write_equations_to_file(equations, cg, 1)
    cg.close()


def gps_vel_n_observation(P, state, vx):
    obs_var = symbols("R_VEL_N", real=True)
    observation = vx
    equations = generate_observation_equations(P, state, observation, obs_var)
    cg = CodeGenerator("./c/generated/gps_vel_n_generated.c")
    write_equations_to_file(equations, cg, 1)
    cg.close()


def gps_vel_e_observation(P, state, vy):
    obs_var = symbols("R_VEL_E", real=True)
    observation = vy
    equations = generate_observation_equations(P, state, observation, obs_var)
    cg = CodeGenerator("./c/generated/gps_vel_e_generated.c")
    write_equations_to_file(equations, cg, 1)
    cg.close()


def gps_vel_d_observation(P, state, vz):
    obs_var = symbols("R_VEL_D", real=True)
    observation = vz
    equations = generate_observation_equations(P, state, observation, obs_var)
    cg = CodeGenerator("./c/generated/gps_vel_d_generated.c")
    write_equations_to_file(equations, cg, 1)
    cg.close()


def generate_code():
    print("Starting ground vehicle EKF code generation:")

    if not os.path.exists("generated"):
        os.makedirs("generated")

    print("Creating symbolic variables...")

    dt = symbols("dt", real=True)
    g = symbols("g", real=True)

    d_ang_x, d_ang_y, d_ang_z = symbols("dax day daz", real=True)
    d_ang = Matrix([d_ang_x, d_ang_y, d_ang_z])

    d_v_x, d_v_y, d_v_z = symbols("dvx dvy dvz", real=True)
    d_v = Matrix([d_v_x, d_v_y, d_v_z])

    u = Matrix([d_ang, d_v])

    d_ang_x_var, d_ang_y_var, d_ang_z_var = symbols("daxVar dayVar dazVar", real=True)
    d_v_x_var, d_v_y_var, d_v_z_var = symbols("dvxVar dvyVar dvzVar", real=True)
    var_u = Matrix.diag(
        d_ang_x_var, d_ang_y_var, d_ang_z_var, d_v_x_var, d_v_y_var, d_v_z_var
    )

    d_ang_b_p_noise_var = symbols("dgb_p_noise_var", real=True)
    dvb_x_p_noise_var = symbols("dvb_x_p_noise_var", real=True)
    dvb_y_p_noise_var = symbols("dvb_y_p_noise_var", real=True)
    dvb_z_p_noise_var = symbols("dvb_z_p_noise_var", real=True)

    state_vector_names = [
        "q0",
        "q1",
        "q2",
        "q3",
        "vn",
        "ve",
        "vd",
        "pn",
        "pe",
        "pd",
        "dax_b",
        "day_b",
        "daz_b",
        "dvx_b",
        "dvy_b",
        "dvz_b",
    ]

    qw, qx, qy, qz = symbols("q0 q1 q2 q3", real=True)
    q = Matrix([qw, qx, qy, qz])
    R_to_earth = quat2Rot(q)

    vx, vy, vz = symbols("vn ve vd", real=True)
    v = Matrix([vx, vy, vz])

    px, py, pz = symbols("pn pe pd", real=True)
    p = Matrix([px, py, pz])

    d_ang_bx, d_ang_by, d_ang_bz = symbols("dax_b day_b daz_b", real=True)
    d_ang_b = Matrix([d_ang_bx, d_ang_by, d_ang_bz])
    d_ang_true = d_ang - d_ang_b

    d_vel_bx, d_vel_by, d_vel_bz = symbols("dvx_b dvy_b dvz_b", real=True)
    d_vel_b = Matrix([d_vel_bx, d_vel_by, d_vel_bz])

    state = Matrix.vstack(q, v, p, d_ang_b, d_vel_b)

    print("Defining state propagation...")

    d_vel_body = d_v - d_vel_b + R_to_earth.T * Matrix([0, 0, g]) * dt
    d_vel_accel_ned = R_to_earth * d_vel_body

    q_new = quat_mult(
        q, Matrix([1, 0.5 * d_ang_true[0], 0.5 * d_ang_true[1], 0.5 * d_ang_true[2]])
    )
    v_new = v + d_vel_accel_ned
    p_new = p + v * dt

    d_ang_b_new = d_ang_b
    d_vel_b_new = d_vel_b

    state_new = Matrix.vstack(q_new, v_new, p_new, d_ang_b_new, d_vel_b_new)

    print("Generating state prediction code...")
    pred_code_generator = CodeGenerator("./c/generated/prediction_generated.c")
    generate_prediction_code(state_new, state_vector_names, pred_code_generator)
    pred_code_generator.close()

    print("Computing state propagation jacobian...")
    A = state_new.jacobian(state)
    G = state_new.jacobian(u)

    P = create_symmetric_cov_matrix()

    print("Computing covariance propagation...")

    P_new = A * P * A.T + G * var_u * G.T

    P_new[10, 10] += d_ang_b_p_noise_var * dt**2
    P_new[11, 11] += d_ang_b_p_noise_var * dt**2
    P_new[12, 12] += d_ang_b_p_noise_var * dt**2
    P_new[13, 13] += dvb_x_p_noise_var * dt**2
    P_new[14, 14] += dvb_y_p_noise_var * dt**2
    P_new[15, 15] += dvb_z_p_noise_var * dt**2

    for index in range(n_states):
        for j in range(n_states):
            if index > j:
                P_new[index, j] = 0

    print("Simplifying covariance propagation...")

    P_new_simple = cse(P_new, symbols("PS0:2000"), optimizations="basic")

    print("Writing covariance propagation to file...")
    cov_code_generator = CodeGenerator("./c/generated/covariance_generated.c")
    cov_code_generator.print_string(
        "Equations for covariance matrix prediction, without process noise!"
    )
    cov_code_generator.write_subexpressions(P_new_simple[0])
    cov_code_generator.write_matrix(Matrix(P_new_simple[1]), "nextP", True)
    cov_code_generator.close()

    print("Generating GPS North Position observation code...")
    gps_pos_n_observation(P, state, px)

    print("Generating GPS East Position observation code...")
    gps_pos_e_observation(P, state, py)

    print("Generating GPS Down Position observation code...")
    gps_pos_d_observation(P, state, pz)

    print("Generating GPS North Velocity observation code...")
    gps_vel_n_observation(P, state, vx)

    print("Generating GPS East Velocity observation code...")
    gps_vel_e_observation(P, state, vy)

    print("Generating GPS Down Velocity observation code...")
    gps_vel_d_observation(P, state, vz)

    print("Generating Body Y-Velocity observation code...")
    body_vel_y_observation(P, state, R_to_earth, v)

    print("Generating Body Z-Velocity observation code...")
    body_vel_z_observation(P, state, R_to_earth, v)

    print("Generating GPS Heading observation code...")
    gps_heading_observation(P, state, R_to_earth)

    print("Code generation complete.")


if __name__ == "__main__":
    generate_code()
