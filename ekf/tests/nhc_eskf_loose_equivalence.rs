use sensor_fusion::eskf_types::{EskfImuDelta, EskfNominalState};
use sensor_fusion::generated_eskf::{self, ERROR_STATES};
use sensor_fusion::generated_loose;
use sensor_fusion::loose::{LOOSE_ERROR_STATES, LooseImuDelta, LooseNominalState};

const ESKF_NHC_Y_SUPPORT: [usize; 8] = [0, 1, 2, 3, 4, 5, 15, 17];
const ESKF_NHC_Z_SUPPORT: [usize; 8] = [0, 1, 2, 3, 4, 5, 15, 16];

#[test]
fn declared_nhc_sparse_supports_cover_generated_rows() {
    let p = [[0.0; ERROR_STATES]; ERROR_STATES];
    let mut eskf_y_rows = Vec::new();
    let mut eskf_z_rows = Vec::new();
    let mut loose_y_rows = Vec::new();
    let mut loose_z_rows = Vec::new();

    for eskf in [sample_eskf_nominal(), alternate_eskf_nominal()] {
        let loose = sample_loose_nominal(&eskf);
        eskf_y_rows.push(generated_eskf::body_vel_y_observation(&eskf, &p, 1.0).h);
        eskf_z_rows.push(generated_eskf::body_vel_z_observation(&eskf, &p, 1.0).h);
        loose_y_rows.push(generated_loose::nhc_y(&loose).1);
        loose_z_rows.push(generated_loose::nhc_z(&loose).1);
    }

    assert_support_covers(
        observed_support_union::<ERROR_STATES, 5>(&eskf_y_rows),
        &ESKF_NHC_Y_SUPPORT,
    );
    assert_support_covers(
        observed_support_union::<ERROR_STATES, 5>(&eskf_z_rows),
        &ESKF_NHC_Z_SUPPORT,
    );
    assert_eq!(
        observed_support_union::<LOOSE_ERROR_STATES, 6>(&loose_y_rows),
        generated_loose::NHC_Y_SUPPORT
    );
    assert_eq!(
        observed_support_union::<LOOSE_ERROR_STATES, 6>(&loose_z_rows),
        generated_loose::NHC_Z_SUPPORT
    );
}

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
    for idx in [0usize, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17] {
        assert_close("dx", idx, idx, dx_eskf[idx], dx_loose_as_eskf[idx]);
    }

    let p_loose_as_eskf = transform_loose_cov_to_eskf(&p_loose, c_es);
    for i in [0usize, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17] {
        for j in [0usize, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17] {
            assert_close("P", i, j, p_eskf[i][j], p_loose_as_eskf[i][j]);
        }
    }
}

#[test]
fn eskf_and_loose_mount_transition_blocks_match_after_basis_transform() {
    let eskf = sample_eskf_nominal();
    let loose = sample_loose_nominal(&eskf);
    let imu_eskf = EskfImuDelta {
        dax: 0.001,
        day: -0.002,
        daz: 0.003,
        dvx: 0.02,
        dvy: -0.01,
        dvz: 0.095,
        dt: 0.01,
    };
    let imu_loose = LooseImuDelta {
        dax_1: imu_eskf.dax,
        day_1: imu_eskf.day,
        daz_1: imu_eskf.daz,
        dvx_1: imu_eskf.dvx,
        dvy_1: imu_eskf.dvy,
        dvz_1: imu_eskf.dvz,
        dax_2: imu_eskf.dax,
        day_2: imu_eskf.day,
        daz_2: imu_eskf.daz,
        dvx_2: imu_eskf.dvx,
        dvy_2: imu_eskf.dvy,
        dvz_2: imu_eskf.dvz,
        dt: imu_eskf.dt,
    };
    let (f_eskf, _) = generated_eskf::error_transition(&eskf, imu_eskf);
    let (f_loose, _) = generated_loose::error_transition(&loose, imu_loose);
    let c_es = quat_to_rot([eskf.q0, eskf.q1, eskf.q2, eskf.q3]);
    for &loose_row in &[3usize, 4, 5, 6, 7, 8] {
        for &eskf_col in &[15usize, 16, 17] {
            let mut expected = 0.0;
            for eskf_row in 0..ERROR_STATES {
                expected +=
                    loose_from_eskf_coeff(loose_row, eskf_row, c_es) * f_eskf[eskf_row][eskf_col];
            }
            let loose_col = match eskf_col {
                15 => 21,
                16 => 22,
                _ => 23,
            };
            let actual = f_loose[loose_row][loose_col];
            assert_close("F", loose_row, loose_col, expected, actual);
        }
    }
}

#[test]
fn eskf_and_loose_common_transition_blocks_are_close_after_basis_transform() {
    let eskf = sample_eskf_nominal();
    let loose = sample_loose_nominal(&eskf);
    let imu_eskf = EskfImuDelta {
        dax: 0.001,
        day: -0.002,
        daz: 0.003,
        dvx: 0.02,
        dvy: -0.01,
        dvz: 0.095,
        dt: 0.01,
    };
    let imu_loose = LooseImuDelta {
        dax_1: imu_eskf.dax,
        day_1: imu_eskf.day,
        daz_1: imu_eskf.daz,
        dvx_1: imu_eskf.dvx,
        dvy_1: imu_eskf.dvy,
        dvz_1: imu_eskf.dvz,
        dax_2: imu_eskf.dax,
        day_2: imu_eskf.day,
        daz_2: imu_eskf.daz,
        dvx_2: imu_eskf.dvx,
        dvy_2: imu_eskf.dvy,
        dvz_2: imu_eskf.dvz,
        dt: imu_eskf.dt,
    };
    let (f_eskf, _) = generated_eskf::error_transition(&eskf, imu_eskf);
    let (f_loose, _) = generated_loose::error_transition(&loose, imu_loose);
    let c_es = quat_to_rot([eskf.q0, eskf.q1, eskf.q2, eskf.q3]);
    let loose_common = [3usize, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23];
    let eskf_common = [0usize, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let mut max_err = 0.0_f32;
    let mut max_pair = (0usize, 0usize, 0.0_f32, 0.0_f32);
    for &loose_row in &loose_common {
        for &eskf_col in &eskf_common {
            let mut expected = 0.0;
            for eskf_row in 0..ERROR_STATES {
                expected +=
                    loose_from_eskf_coeff(loose_row, eskf_row, c_es) * f_eskf[eskf_row][eskf_col];
            }
            let mut actual = 0.0;
            for loose_col in 0..LOOSE_ERROR_STATES {
                actual += f_loose[loose_row][loose_col]
                    * loose_from_eskf_coeff(loose_col, eskf_col, c_es);
            }
            let err = (actual - expected).abs();
            if err > max_err {
                max_err = err;
                max_pair = (loose_row, eskf_col, actual, expected);
            }
        }
    }
    assert!(
        max_err < 5.0e-3,
        "common F mismatch loose row {} eskf col {} actual={} expected={} err={}",
        max_pair.0,
        max_pair.1,
        max_pair.2,
        max_pair.3,
        max_err
    );
}

#[test]
fn eskf_and_loose_common_prediction_covariances_are_close_after_basis_transform() {
    let eskf = sample_eskf_nominal();
    let loose = sample_loose_nominal(&eskf);
    let imu_eskf = EskfImuDelta {
        dax: 0.001,
        day: -0.002,
        daz: 0.003,
        dvx: 0.02,
        dvy: -0.01,
        dvz: 0.095,
        dt: 0.01,
    };
    let imu_loose = LooseImuDelta {
        dax_1: imu_eskf.dax,
        day_1: imu_eskf.day,
        daz_1: imu_eskf.daz,
        dvx_1: imu_eskf.dvx,
        dvy_1: imu_eskf.dvy,
        dvz_1: imu_eskf.dvz,
        dax_2: imu_eskf.dax,
        day_2: imu_eskf.day,
        daz_2: imu_eskf.daz,
        dvx_2: imu_eskf.dvx,
        dvy_2: imu_eskf.dvy,
        dvz_2: imu_eskf.dvz,
        dt: imu_eskf.dt,
    };
    let c_es = quat_to_rot([eskf.q0, eskf.q1, eskf.q2, eskf.q3]);
    let p_eskf = sample_eskf_covariance();
    let p_loose = transform_eskf_cov_to_loose(&p_eskf, c_es);
    let (f_eskf, g_eskf) = generated_eskf::error_transition(&eskf, imu_eskf);
    let (f_loose, g_loose) = generated_loose::error_transition(&loose, imu_loose);
    let q_eskf = [1.0e-8_f32; generated_eskf::NOISE_STATES];
    let mut q_loose = [0.0_f32; sensor_fusion::loose::LOOSE_NOISE_STATES];
    q_loose[0] = q_eskf[3];
    q_loose[1] = q_eskf[4];
    q_loose[2] = q_eskf[5];
    q_loose[3] = q_eskf[0];
    q_loose[4] = q_eskf[1];
    q_loose[5] = q_eskf[2];
    q_loose[6] = q_eskf[9];
    q_loose[7] = q_eskf[10];
    q_loose[8] = q_eskf[11];
    q_loose[9] = q_eskf[6];
    q_loose[10] = q_eskf[7];
    q_loose[11] = q_eskf[8];
    q_loose[18] = q_eskf[12];
    q_loose[19] = q_eskf[13];
    q_loose[20] = q_eskf[14];

    let p_eskf_next = predict_cov_eskf(&f_eskf, &g_eskf, &p_eskf, &q_eskf);
    let p_loose_next = predict_cov_loose(&f_loose, &g_loose, &p_loose, &q_loose);
    let p_loose_as_eskf = transform_loose_cov_to_eskf(&p_loose_next, c_es);
    let common = [0usize, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let mut max_err = 0.0_f32;
    let mut max_pair = (0usize, 0usize, 0.0_f32, 0.0_f32);
    for &i in &common {
        for &j in &common {
            let err = (p_eskf_next[i][j] - p_loose_as_eskf[i][j]).abs();
            if err > max_err {
                max_err = err;
                max_pair = (i, j, p_eskf_next[i][j], p_loose_as_eskf[i][j]);
            }
        }
    }
    assert!(
        max_err < 5.0e-6,
        "prediction P mismatch eskf[{}][{}]={} loose_as_eskf={} err={}",
        max_pair.0,
        max_pair.1,
        max_pair.2,
        max_pair.3,
        max_err
    );
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

fn alternate_eskf_nominal() -> EskfNominalState {
    EskfNominalState {
        q0: 0.927_361_85,
        q1: -0.162_134_74,
        q2: 0.307_685_73,
        q3: -0.130_723_83,
        vn: -3.5,
        ve: 7.25,
        vd: -1.2,
        qcs0: 0.961_256_3,
        qcs1: -0.037_259_74,
        qcs2: 0.128_494_34,
        qcs3: -0.241_015_14,
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
        sgx: 1.0,
        sgy: 1.0,
        sgz: 1.0,
        sax: 1.0,
        say: 1.0,
        saz: 1.0,
        ..LooseNominalState::default()
    }
}

fn sample_eskf_covariance() -> [[f32; ERROR_STATES]; ERROR_STATES] {
    let mut p = [[0.0; ERROR_STATES]; ERROR_STATES];
    let states = [0usize, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17];
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

fn predict_cov_eskf(
    f: &[[f32; ERROR_STATES]; ERROR_STATES],
    g: &[[f32; generated_eskf::NOISE_STATES]; ERROR_STATES],
    p: &[[f32; ERROR_STATES]; ERROR_STATES],
    q: &[f32; generated_eskf::NOISE_STATES],
) -> [[f32; ERROR_STATES]; ERROR_STATES] {
    let mut fp = [[0.0; ERROR_STATES]; ERROR_STATES];
    for i in 0..ERROR_STATES {
        for j in 0..ERROR_STATES {
            for k in 0..ERROR_STATES {
                fp[i][j] += f[i][k] * p[k][j];
            }
        }
    }
    let mut out = [[0.0; ERROR_STATES]; ERROR_STATES];
    for i in 0..ERROR_STATES {
        for j in 0..ERROR_STATES {
            for k in 0..ERROR_STATES {
                out[i][j] += fp[i][k] * f[j][k];
            }
            for (k, qk) in q.iter().enumerate() {
                out[i][j] += g[i][k] * *qk * g[j][k];
            }
        }
    }
    out
}

fn predict_cov_loose(
    f: &[[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
    g: &[[f32; sensor_fusion::loose::LOOSE_NOISE_STATES]; LOOSE_ERROR_STATES],
    p: &[[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES],
    q: &[f32; sensor_fusion::loose::LOOSE_NOISE_STATES],
) -> [[f32; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES] {
    let mut fp = [[0.0; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES];
    for i in 0..LOOSE_ERROR_STATES {
        for j in 0..LOOSE_ERROR_STATES {
            for k in 0..LOOSE_ERROR_STATES {
                fp[i][j] += f[i][k] * p[k][j];
            }
        }
    }
    let mut out = [[0.0; LOOSE_ERROR_STATES]; LOOSE_ERROR_STATES];
    for i in 0..LOOSE_ERROR_STATES {
        for j in 0..LOOSE_ERROR_STATES {
            for k in 0..LOOSE_ERROR_STATES {
                out[i][j] += fp[i][k] * f[j][k];
            }
            for (k, qk) in q.iter().enumerate() {
                out[i][j] += g[i][k] * *qk * g[j][k];
            }
        }
    }
    out
}

fn loose_from_eskf_coeff(loose_idx: usize, eskf_idx: usize, c_es: [[f32; 3]; 3]) -> f32 {
    match (loose_idx, eskf_idx) {
        (6..=8, 0..=2) => c_es[loose_idx - 6][eskf_idx],
        (3, 3) | (4, 4) | (5, 5) => 1.0,
        (12, 9) | (13, 10) | (14, 11) => -1.0,
        (9, 12) | (10, 13) | (11, 14) => -1.0,
        (21, 15) | (22, 16) | (23, 17) => 1.0,
        _ => 0.0,
    }
}

fn eskf_from_loose_coeff(eskf_idx: usize, loose_idx: usize, c_es: [[f32; 3]; 3]) -> f32 {
    match (eskf_idx, loose_idx) {
        (0..=2, 6..=8) => c_es[loose_idx - 6][eskf_idx],
        (3, 3) | (4, 4) | (5, 5) => 1.0,
        (9, 12) | (10, 13) | (11, 14) => -1.0,
        (12, 9) | (13, 10) | (14, 11) => -1.0,
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

fn observed_support_union<const N: usize, const M: usize>(rows: &[[f32; N]]) -> [usize; M] {
    let support: Vec<_> = (0..N)
        .filter(|idx| rows.iter().any(|row| row[*idx].abs() > 1.0e-6))
        .collect();
    support
        .try_into()
        .unwrap_or_else(|support: Vec<usize>| panic!("unexpected support {support:?}"))
}

fn assert_support_covers<const N: usize>(observed: [usize; N], declared: &[usize]) {
    for idx in observed {
        assert!(
            declared.contains(&idx),
            "generated row contains state {idx}, but declared support is {declared:?}"
        );
    }
}

fn assert_close(label: &str, eskf_idx: usize, loose_idx: usize, eskf: f32, loose: f32) {
    let err = (eskf - loose).abs();
    assert!(
        err < 1.0e-3,
        "{label} ESKF[{eskf_idx}] vs loose[{loose_idx}] mismatch: eskf={eskf} loose={loose} err={err}"
    );
}
