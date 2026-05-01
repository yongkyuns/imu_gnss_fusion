use sensor_fusion::eskf_types::EskfNominalState;
use sensor_fusion::generated_eskf::{self, ERROR_STATES};
use sensor_fusion::generated_loose;
use sensor_fusion::loose::{LOOSE_ERROR_STATES, LooseNominalState};

const ESKF_NHC_Y_SUPPORT: [usize; 8] = [0, 1, 2, 3, 4, 5, 15, 17];
const ESKF_NHC_Z_SUPPORT: [usize; 8] = [0, 1, 2, 3, 4, 5, 15, 16];

#[test]
fn eskf_and_loose_nhc_jacobians_match_after_attitude_basis_transform() {
    let eskf = EskfNominalState {
        q0: 0.979_466,
        q1: 0.059_519,
        q2: -0.068_553,
        q3: 0.180_162,
        vn: 8.0,
        ve: -1.5,
        vd: 0.35,
        qcs0: 0.995_005,
        qcs1: 0.045_023,
        qcs2: -0.036_704,
        qcs3: 0.081_264,
        ..EskfNominalState::default()
    };
    let loose = LooseNominalState {
        q0: eskf.q0,
        q1: eskf.q1,
        q2: eskf.q2,
        q3: eskf.q3,
        vn: eskf.vn,
        ve: eskf.ve,
        vd: eskf.vd,
        qcs0: eskf.qcs0,
        qcs1: eskf.qcs1,
        qcs2: eskf.qcs2,
        qcs3: eskf.qcs3,
        ..LooseNominalState::default()
    };
    let p = [[0.0; ERROR_STATES]; ERROR_STATES];
    let eskf_y = generated_eskf::body_vel_y_observation(&eskf, &p, 1.0).h;
    let eskf_z = generated_eskf::body_vel_z_observation(&eskf, &p, 1.0).h;
    let (_loose_vy, loose_y) = generated_loose::nhc_y(&loose);
    let (_loose_vz, loose_z) = generated_loose::nhc_z(&loose);
    let c_es = quat_to_rot([eskf.q0, eskf.q1, eskf.q2, eskf.q3]);

    let loose_y_att_local = row_times_mat3([loose_y[6], loose_y[7], loose_y[8]], c_es);
    let loose_z_att_local = row_times_mat3([loose_z[6], loose_z[7], loose_z[8]], c_es);

    for i in 0..3 {
        assert_close("Y attitude basis", i, i, eskf_y[i], loose_y_att_local[i]);
        assert_close("Z attitude basis", i, i, eskf_z[i], loose_z_att_local[i]);
    }
    for (eskf_idx, loose_idx) in [(3, 3), (4, 4), (5, 5)] {
        assert_close(
            "Y",
            eskf_idx,
            loose_idx,
            eskf_y[eskf_idx],
            loose_y[loose_idx],
        );
        assert_close(
            "Z",
            eskf_idx,
            loose_idx,
            eskf_z[eskf_idx],
            loose_z[loose_idx],
        );
    }
    for (eskf_idx, loose_idx) in [(15, 21), (16, 22), (17, 23)] {
        assert_close(
            "Y",
            eskf_idx,
            loose_idx,
            eskf_y[eskf_idx],
            loose_y[loose_idx],
        );
        assert_close(
            "Z",
            eskf_idx,
            loose_idx,
            eskf_z[eskf_idx],
            loose_z[loose_idx],
        );
    }
}

#[test]
fn eskf_and_loose_nhc_update_match_after_covariance_basis_transform() {
    let eskf = sample_eskf_nominal();
    let loose = sample_loose_nominal(&eskf);
    let c_es = quat_to_rot([eskf.q0, eskf.q1, eskf.q2, eskf.q3]);
    let mut p_eskf = sample_eskf_covariance();
    let mut p_loose = transform_eskf_cov_to_loose(&p_eskf, c_es);

    let eskf_y = generated_eskf::body_vel_y_observation(&eskf, &p_eskf, 1.0).h;
    let eskf_z = generated_eskf::body_vel_z_observation(&eskf, &p_eskf, 1.0).h;
    let (loose_vy, loose_y) = generated_loose::nhc_y(&loose);
    let (loose_vz, loose_z) = generated_loose::nhc_z(&loose);

    let mut dx_eskf = [0.0; ERROR_STATES];
    scalar_update_eskf(
        &mut p_eskf,
        &mut dx_eskf,
        &eskf_y,
        &ESKF_NHC_Y_SUPPORT,
        -loose_vy,
        1.0,
    );
    scalar_update_eskf(
        &mut p_eskf,
        &mut dx_eskf,
        &eskf_z,
        &ESKF_NHC_Z_SUPPORT,
        -loose_vz,
        1.0,
    );

    let mut dx_loose = [0.0; LOOSE_ERROR_STATES];
    scalar_update_loose(
        &mut p_loose,
        &mut dx_loose,
        &loose_y,
        &generated_loose::NHC_Y_SUPPORT,
        -loose_vy,
        1.0,
    );
    scalar_update_loose(
        &mut p_loose,
        &mut dx_loose,
        &loose_z,
        &generated_loose::NHC_Z_SUPPORT,
        -loose_vz,
        1.0,
    );

    let dx_loose_as_eskf = transform_loose_dx_to_eskf(&dx_loose, c_es);
    for idx in [0usize, 1, 2, 3, 4, 5, 15, 16, 17] {
        assert_close("dx", idx, idx, dx_eskf[idx], dx_loose_as_eskf[idx]);
    }

    let p_loose_as_eskf = transform_loose_cov_to_eskf(&p_loose, c_es);
    for i in [0usize, 1, 2, 3, 4, 5, 15, 16, 17] {
        for j in [0usize, 1, 2, 3, 4, 5, 15, 16, 17] {
            assert_close("P", i, j, p_eskf[i][j], p_loose_as_eskf[i][j]);
        }
    }
}

fn sample_eskf_nominal() -> EskfNominalState {
    EskfNominalState {
        q0: 0.979_466,
        q1: 0.059_519,
        q2: -0.068_553,
        q3: 0.180_162,
        vn: 8.0,
        ve: -1.5,
        vd: 0.35,
        qcs0: 0.995_005,
        qcs1: 0.045_023,
        qcs2: -0.036_704,
        qcs3: 0.081_264,
        ..EskfNominalState::default()
    }
}

fn sample_loose_nominal(eskf: &EskfNominalState) -> LooseNominalState {
    LooseNominalState {
        q0: eskf.q0,
        q1: eskf.q1,
        q2: eskf.q2,
        q3: eskf.q3,
        vn: eskf.vn,
        ve: eskf.ve,
        vd: eskf.vd,
        qcs0: eskf.qcs0,
        qcs1: eskf.qcs1,
        qcs2: eskf.qcs2,
        qcs3: eskf.qcs3,
        ..LooseNominalState::default()
    }
}

fn sample_eskf_covariance() -> [[f32; ERROR_STATES]; ERROR_STATES] {
    let mut p = [[0.0; ERROR_STATES]; ERROR_STATES];
    let states = [0usize, 1, 2, 3, 4, 5, 15, 16, 17];
    for (rank, &i) in states.iter().enumerate() {
        let sigma = 0.02 + 0.015 * rank as f32;
        p[i][i] = sigma * sigma;
    }
    for (a_rank, &i) in states.iter().enumerate() {
        for (b_rank, &j) in states.iter().enumerate().skip(a_rank + 1) {
            let corr = (((a_rank + 1) * (b_rank + 3)) % 5) as f32 * 0.025 - 0.05;
            let v = corr * (p[i][i] * p[j][j]).sqrt();
            p[i][j] = v;
            p[j][i] = v;
        }
    }
    p
}

fn scalar_update_eskf(
    p: &mut [[f32; ERROR_STATES]; ERROR_STATES],
    dx: &mut [f32; ERROR_STATES],
    h: &[f32; ERROR_STATES],
    support: &[usize],
    residual: f32,
    r: f32,
) {
    let mut ph = [0.0; ERROR_STATES];
    let mut s = r;
    let mut hd = 0.0;
    for i in 0..ERROR_STATES {
        for &state in support {
            ph[i] += p[i][state] * h[state];
        }
    }
    for &state in support {
        s += h[state] * ph[state];
        hd += h[state] * dx[state];
    }
    for i in 0..ERROR_STATES {
        dx[i] += (ph[i] / s) * (residual - hd);
    }
    for i in 0..ERROR_STATES {
        for j in i..ERROR_STATES {
            let updated = p[i][j] - (ph[i] * ph[j]) / s;
            p[i][j] = updated;
            p[j][i] = updated;
        }
    }
}

fn scalar_update_loose(
    p: &mut [[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
    dx: &mut [f32; LOOSE_ERROR_STATES],
    h: &[f32; LOOSE_ERROR_STATES],
    support: &[usize],
    residual: f32,
    r: f32,
) {
    let mut ph = [0.0; LOOSE_ERROR_STATES];
    let mut s = r;
    let mut hd = 0.0;
    for i in 0..LOOSE_ERROR_STATES {
        for &state in support {
            ph[i] += p[i][state] * h[state];
        }
    }
    for &state in support {
        s += h[state] * ph[state];
        hd += h[state] * dx[state];
    }
    for i in 0..LOOSE_ERROR_STATES {
        dx[i] += (ph[i] / s) * (residual - hd);
    }
    for i in 0..LOOSE_ERROR_STATES {
        for j in i..LOOSE_ERROR_STATES {
            let updated = p[i][j] - (ph[i] * ph[j]) / s;
            p[i][j] = updated;
            p[j][i] = updated;
        }
    }
}

fn transform_eskf_cov_to_loose(
    p_eskf: &[[f32; ERROR_STATES]; ERROR_STATES],
    c_es: [[f32; 3]; 3],
) -> [[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES] {
    let mut p = [[0.0; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES];
    for loose_i in 0..LOOSE_ERROR_STATES {
        for loose_j in 0..LOOSE_ERROR_STATES {
            let mut v = 0.0;
            for eskf_i in 0..ERROR_STATES {
                for eskf_j in 0..ERROR_STATES {
                    v += loose_from_eskf_coeff(loose_i, eskf_i, c_es)
                        * p_eskf[eskf_i][eskf_j]
                        * loose_from_eskf_coeff(loose_j, eskf_j, c_es);
                }
            }
            p[loose_i][loose_j] = v;
        }
    }
    p
}

fn transform_loose_cov_to_eskf(
    p_loose: &[[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
    c_es: [[f32; 3]; 3],
) -> [[f32; ERROR_STATES]; ERROR_STATES] {
    let mut p = [[0.0; ERROR_STATES]; ERROR_STATES];
    for eskf_i in 0..ERROR_STATES {
        for eskf_j in 0..ERROR_STATES {
            let mut v = 0.0;
            for loose_i in 0..LOOSE_ERROR_STATES {
                for loose_j in 0..LOOSE_ERROR_STATES {
                    v += eskf_from_loose_coeff(eskf_i, loose_i, c_es)
                        * p_loose[loose_i][loose_j]
                        * eskf_from_loose_coeff(eskf_j, loose_j, c_es);
                }
            }
            p[eskf_i][eskf_j] = v;
        }
    }
    p
}

fn transform_loose_dx_to_eskf(
    dx_loose: &[f32; LOOSE_ERROR_STATES],
    c_es: [[f32; 3]; 3],
) -> [f32; ERROR_STATES] {
    let mut dx = [0.0; ERROR_STATES];
    for (i, out) in dx.iter_mut().enumerate() {
        for loose_i in 0..LOOSE_ERROR_STATES {
            *out += eskf_from_loose_coeff(i, loose_i, c_es) * dx_loose[loose_i];
        }
    }
    dx
}

fn loose_from_eskf_coeff(loose_idx: usize, eskf_idx: usize, c_es: [[f32; 3]; 3]) -> f32 {
    match (loose_idx, eskf_idx) {
        (6..=8, 0..=2) => c_es[loose_idx - 6][eskf_idx],
        (3, 3) | (4, 4) | (5, 5) => 1.0,
        (21, 15) | (22, 16) | (23, 17) => 1.0,
        _ => 0.0,
    }
}

fn eskf_from_loose_coeff(eskf_idx: usize, loose_idx: usize, c_es: [[f32; 3]; 3]) -> f32 {
    match (eskf_idx, loose_idx) {
        (0..=2, 6..=8) => c_es[loose_idx - 6][eskf_idx],
        (3, 3) | (4, 4) | (5, 5) => 1.0,
        (15, 21) | (16, 22) | (17, 23) => 1.0,
        _ => 0.0,
    }
}

fn quat_to_rot(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [q0, q1, q2, q3] = q;
    [
        [
            1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2,
        ],
    ]
}

fn row_times_mat3(row: [f32; 3], mat: [[f32; 3]; 3]) -> [f32; 3] {
    [
        row[0] * mat[0][0] + row[1] * mat[1][0] + row[2] * mat[2][0],
        row[0] * mat[0][1] + row[1] * mat[1][1] + row[2] * mat[2][1],
        row[0] * mat[0][2] + row[1] * mat[1][2] + row[2] * mat[2][2],
    ]
}

fn assert_close(label: &str, eskf_idx: usize, loose_idx: usize, eskf: f32, loose: f32) {
    let err = (eskf - loose).abs();
    assert!(
        err < 1.0e-3,
        "{label} H ESKF[{eskf_idx}] vs loose[{loose_idx}] mismatch: eskf={eskf} loose={loose} err={err}"
    );
}
