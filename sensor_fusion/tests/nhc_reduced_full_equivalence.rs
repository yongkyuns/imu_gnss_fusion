#![allow(clippy::needless_range_loop)]

use sensor_fusion::{full, generated_full, generated_reduced, reduced};

const REDUCED_STATES: usize = generated_reduced::ERROR_STATES;
const FULL_STATES: usize = full::ERROR_STATES;

const REDUCED_NHC_Y_SUPPORT: [usize; 8] = [0, 1, 2, 3, 4, 5, 15, 17];
const REDUCED_NHC_Z_SUPPORT: [usize; 8] = [0, 1, 2, 3, 4, 5, 15, 16];

#[test]
fn declared_nhc_sparse_supports_cover_generated_rows() {
    let p = [[0.0; REDUCED_STATES]; REDUCED_STATES];
    let mut reduced_y_rows = Vec::new();
    let mut reduced_z_rows = Vec::new();
    let mut full_y_rows = Vec::new();
    let mut full_z_rows = Vec::new();

    for reduced in [sample_reduced_nominal(), alternate_reduced_nominal()] {
        let full = sample_full_nominal(&reduced);
        reduced_y_rows.push(generated_reduced::body_vel_y_observation(&reduced, &p, 1.0).h);
        reduced_z_rows.push(generated_reduced::body_vel_z_observation(&reduced, &p, 1.0).h);
        full_y_rows.push(generated_full::nhc_y(&full).1);
        full_z_rows.push(generated_full::nhc_z(&full).1);
    }

    assert_support_covers(
        observed_support_union::<REDUCED_STATES, 5>(&reduced_y_rows),
        &REDUCED_NHC_Y_SUPPORT,
    );
    assert_support_covers(
        observed_support_union::<REDUCED_STATES, 5>(&reduced_z_rows),
        &REDUCED_NHC_Z_SUPPORT,
    );
    assert_eq!(
        observed_support_union::<FULL_STATES, 6>(&full_y_rows),
        generated_full::NHC_Y_SUPPORT
    );
    assert_eq!(
        observed_support_union::<FULL_STATES, 6>(&full_z_rows),
        generated_full::NHC_Z_SUPPORT
    );
}

#[test]
fn reduced_and_full_nhc_jacobians_match_after_attitude_basis_transform() {
    let reduced = reduced::NominalState {
        q0: 0.979_466,
        q1: 0.059_519,
        q2: -0.068_553,
        q3: 0.180_162,
        vn: 8.0,
        ve: -1.5,
        vd: 0.35,
        q_bv0: 0.995_005,
        q_bv1: 0.045_023,
        q_bv2: -0.036_704,
        q_bv3: 0.081_264,
        ..reduced::NominalState::default()
    };
    let full = full::NominalState {
        q0: reduced.q0,
        q1: reduced.q1,
        q2: reduced.q2,
        q3: reduced.q3,
        vn: reduced.vn,
        ve: reduced.ve,
        vd: reduced.vd,
        q_bv0: reduced.q_bv0,
        q_bv1: reduced.q_bv1,
        q_bv2: reduced.q_bv2,
        q_bv3: reduced.q_bv3,
        ..full::NominalState::default()
    };
    let p = [[0.0; REDUCED_STATES]; REDUCED_STATES];
    let reduced_y = generated_reduced::body_vel_y_observation(&reduced, &p, 1.0).h;
    let reduced_z = generated_reduced::body_vel_z_observation(&reduced, &p, 1.0).h;
    let (_full_vy, full_y) = generated_full::nhc_y(&full);
    let (_full_vz, full_z) = generated_full::nhc_z(&full);
    let c_es = quat_to_rot([reduced.q0, reduced.q1, reduced.q2, reduced.q3]);

    let full_y_att_local = row_times_mat3([full_y[6], full_y[7], full_y[8]], c_es);
    let full_z_att_local = row_times_mat3([full_z[6], full_z[7], full_z[8]], c_es);

    for i in 0..3 {
        assert_close("Y attitude basis", i, i, reduced_y[i], full_y_att_local[i]);
        assert_close("Z attitude basis", i, i, reduced_z[i], full_z_att_local[i]);
    }
    for (reduced_idx, full_idx) in [(3, 3), (4, 4), (5, 5)] {
        assert_close(
            "Y",
            reduced_idx,
            full_idx,
            reduced_y[reduced_idx],
            full_y[full_idx],
        );
        assert_close(
            "Z",
            reduced_idx,
            full_idx,
            reduced_z[reduced_idx],
            full_z[full_idx],
        );
    }
    for (reduced_idx, full_idx) in [(15, 21), (16, 22), (17, 23)] {
        assert_close(
            "Y",
            reduced_idx,
            full_idx,
            reduced_y[reduced_idx],
            full_y[full_idx],
        );
        assert_close(
            "Z",
            reduced_idx,
            full_idx,
            reduced_z[reduced_idx],
            full_z[full_idx],
        );
    }
}

#[test]
fn reduced_and_full_nhc_update_match_after_covariance_basis_transform() {
    let reduced = sample_reduced_nominal();
    let full = sample_full_nominal(&reduced);
    let c_es = quat_to_rot([reduced.q0, reduced.q1, reduced.q2, reduced.q3]);
    let mut p_reduced = sample_reduced_covariance();
    let mut p_full = transform_reduced_cov_to_full(&p_reduced, c_es);

    let reduced_y = generated_reduced::body_vel_y_observation(&reduced, &p_reduced, 1.0).h;
    let reduced_z = generated_reduced::body_vel_z_observation(&reduced, &p_reduced, 1.0).h;
    let (full_vy, full_y) = generated_full::nhc_y(&full);
    let (full_vz, full_z) = generated_full::nhc_z(&full);

    let mut dx_reduced = [0.0; REDUCED_STATES];
    scalar_update_reduced(
        &mut p_reduced,
        &mut dx_reduced,
        &reduced_y,
        &REDUCED_NHC_Y_SUPPORT,
        -full_vy,
        1.0,
    );
    scalar_update_reduced(
        &mut p_reduced,
        &mut dx_reduced,
        &reduced_z,
        &REDUCED_NHC_Z_SUPPORT,
        -full_vz,
        1.0,
    );

    let mut dx_full = [0.0; FULL_STATES];
    scalar_update_full(
        &mut p_full,
        &mut dx_full,
        &full_y,
        &generated_full::NHC_Y_SUPPORT,
        -full_vy,
        1.0,
    );
    scalar_update_full(
        &mut p_full,
        &mut dx_full,
        &full_z,
        &generated_full::NHC_Z_SUPPORT,
        -full_vz,
        1.0,
    );

    let dx_full_as_reduced = transform_full_dx_to_reduced(&dx_full, c_es);
    for idx in [0usize, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17] {
        assert_close("dx", idx, idx, dx_reduced[idx], dx_full_as_reduced[idx]);
    }

    let p_full_as_reduced = transform_full_cov_to_reduced(&p_full, c_es);
    for i in [0usize, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17] {
        for j in [0usize, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17] {
            assert_close("P", i, j, p_reduced[i][j], p_full_as_reduced[i][j]);
        }
    }
}

#[test]
fn reduced_and_full_nhc_reset_match_when_mount_covariance_is_not_reset() {
    let reduced = sample_reduced_nominal();
    let full = sample_full_nominal(&reduced);
    let c_es = quat_to_rot([reduced.q0, reduced.q1, reduced.q2, reduced.q3]);
    let mut p_reduced = sample_reduced_covariance();
    let mut p_full = transform_reduced_cov_to_full(&p_reduced, c_es);

    let reduced_y = generated_reduced::body_vel_y_observation(&reduced, &p_reduced, 1.0).h;
    let reduced_z = generated_reduced::body_vel_z_observation(&reduced, &p_reduced, 1.0).h;
    let (full_vy, full_y) = generated_full::nhc_y(&full);
    let (full_vz, full_z) = generated_full::nhc_z(&full);

    let mut dx_reduced = [0.0; REDUCED_STATES];
    scalar_update_reduced(
        &mut p_reduced,
        &mut dx_reduced,
        &reduced_y,
        &REDUCED_NHC_Y_SUPPORT,
        -full_vy,
        1.0,
    );
    scalar_update_reduced(
        &mut p_reduced,
        &mut dx_reduced,
        &reduced_z,
        &REDUCED_NHC_Z_SUPPORT,
        -full_vz,
        1.0,
    );
    apply_reduced_runtime_reset(&mut p_reduced, &dx_reduced);

    let mut dx_full = [0.0; FULL_STATES];
    scalar_update_full(
        &mut p_full,
        &mut dx_full,
        &full_y,
        &generated_full::NHC_Y_SUPPORT,
        -full_vy,
        1.0,
    );
    scalar_update_full(
        &mut p_full,
        &mut dx_full,
        &full_z,
        &generated_full::NHC_Z_SUPPORT,
        -full_vz,
        1.0,
    );
    let mut p_full_with_mount_reset = p_full;
    apply_full_runtime_reset(&mut p_full, &dx_full, false);
    apply_full_runtime_reset(&mut p_full_with_mount_reset, &dx_full, true);

    let p_full_as_reduced = transform_full_cov_to_reduced(&p_full, c_es);
    let p_full_mount_reset_as_reduced =
        transform_full_cov_to_reduced(&p_full_with_mount_reset, c_es);
    let common = [0usize, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let no_mount_reset_mismatch = max_cov_mismatch(&p_reduced, &p_full_as_reduced, &common);
    let with_mount_reset_mismatch =
        max_cov_mismatch(&p_reduced, &p_full_mount_reset_as_reduced, &common);

    assert!(
        no_mount_reset_mismatch < 5.0e-6,
        "Full attitude-only reset should match Reduced reset, mismatch={}",
        no_mount_reset_mismatch
    );
    assert!(
        with_mount_reset_mismatch > no_mount_reset_mismatch * 10.0,
        "Full mount covariance reset should be distinguishable from Reduced reset: no_mount={} with_mount={}",
        no_mount_reset_mismatch,
        with_mount_reset_mismatch
    );
}

#[test]
fn reduced_and_full_mount_transition_blocks_match_after_basis_transform() {
    let reduced = sample_reduced_nominal();
    let full = sample_full_nominal(&reduced);
    let imu_reduced = reduced::ImuDelta {
        dax: 0.001,
        day: -0.002,
        daz: 0.003,
        dvx: 0.02,
        dvy: -0.01,
        dvz: 0.095,
        dt: 0.01,
    };
    let imu_full = full::ImuDelta {
        dax_1: imu_reduced.dax,
        day_1: imu_reduced.day,
        daz_1: imu_reduced.daz,
        dvx_1: imu_reduced.dvx,
        dvy_1: imu_reduced.dvy,
        dvz_1: imu_reduced.dvz,
        dax_2: imu_reduced.dax,
        day_2: imu_reduced.day,
        daz_2: imu_reduced.daz,
        dvx_2: imu_reduced.dvx,
        dvy_2: imu_reduced.dvy,
        dvz_2: imu_reduced.dvz,
        dt: imu_reduced.dt,
    };
    let (f_reduced, _) = generated_reduced::error_transition(&reduced, imu_reduced);
    let (f_full, _) = generated_full::error_transition(&full, imu_full);
    let c_es = quat_to_rot([reduced.q0, reduced.q1, reduced.q2, reduced.q3]);
    for &full_row in &[3usize, 4, 5, 6, 7, 8] {
        for &reduced_col in &[15usize, 16, 17] {
            let mut expected = 0.0;
            for reduced_row in 0..REDUCED_STATES {
                expected += full_from_reduced_coeff(full_row, reduced_row, c_es)
                    * f_reduced[reduced_row][reduced_col];
            }
            let full_col = match reduced_col {
                15 => 21,
                16 => 22,
                _ => 23,
            };
            let actual = f_full[full_row][full_col];
            assert_close("F", full_row, full_col, expected, actual);
        }
    }
}

#[test]
fn reduced_and_full_common_transition_blocks_are_close_after_basis_transform() {
    let reduced = sample_reduced_nominal();
    let full = sample_full_nominal(&reduced);
    let imu_reduced = reduced::ImuDelta {
        dax: 0.001,
        day: -0.002,
        daz: 0.003,
        dvx: 0.02,
        dvy: -0.01,
        dvz: 0.095,
        dt: 0.01,
    };
    let imu_full = full::ImuDelta {
        dax_1: imu_reduced.dax,
        day_1: imu_reduced.day,
        daz_1: imu_reduced.daz,
        dvx_1: imu_reduced.dvx,
        dvy_1: imu_reduced.dvy,
        dvz_1: imu_reduced.dvz,
        dax_2: imu_reduced.dax,
        day_2: imu_reduced.day,
        daz_2: imu_reduced.daz,
        dvx_2: imu_reduced.dvx,
        dvy_2: imu_reduced.dvy,
        dvz_2: imu_reduced.dvz,
        dt: imu_reduced.dt,
    };
    let (f_reduced, _) = generated_reduced::error_transition(&reduced, imu_reduced);
    let (f_full, _) = generated_full::error_transition(&full, imu_full);
    let c_es = quat_to_rot([reduced.q0, reduced.q1, reduced.q2, reduced.q3]);
    let full_common = [3usize, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23];
    let reduced_common = [0usize, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let mut max_err = 0.0_f32;
    let mut max_pair = (0usize, 0usize, 0.0_f32, 0.0_f32);
    for &full_row in &full_common {
        for &reduced_col in &reduced_common {
            let mut expected = 0.0;
            for reduced_row in 0..REDUCED_STATES {
                expected += full_from_reduced_coeff(full_row, reduced_row, c_es)
                    * f_reduced[reduced_row][reduced_col];
            }
            let mut actual = 0.0;
            for full_col in 0..FULL_STATES {
                actual += f_full[full_row][full_col]
                    * full_from_reduced_coeff(full_col, reduced_col, c_es);
            }
            let err = (actual - expected).abs();
            if err > max_err {
                max_err = err;
                max_pair = (full_row, reduced_col, actual, expected);
            }
        }
    }
    assert!(
        max_err < 5.0e-3,
        "common F mismatch full row {} reduced col {} actual={} expected={} err={}",
        max_pair.0,
        max_pair.1,
        max_pair.2,
        max_pair.3,
        max_err
    );
}

#[test]
fn reduced_and_full_common_prediction_covariances_are_close_after_basis_transform() {
    let reduced = sample_reduced_nominal();
    let full = sample_full_nominal(&reduced);
    let imu_reduced = reduced::ImuDelta {
        dax: 0.001,
        day: -0.002,
        daz: 0.003,
        dvx: 0.02,
        dvy: -0.01,
        dvz: 0.095,
        dt: 0.01,
    };
    let imu_full = full::ImuDelta {
        dax_1: imu_reduced.dax,
        day_1: imu_reduced.day,
        daz_1: imu_reduced.daz,
        dvx_1: imu_reduced.dvx,
        dvy_1: imu_reduced.dvy,
        dvz_1: imu_reduced.dvz,
        dax_2: imu_reduced.dax,
        day_2: imu_reduced.day,
        daz_2: imu_reduced.daz,
        dvx_2: imu_reduced.dvx,
        dvy_2: imu_reduced.dvy,
        dvz_2: imu_reduced.dvz,
        dt: imu_reduced.dt,
    };
    let c_es = quat_to_rot([reduced.q0, reduced.q1, reduced.q2, reduced.q3]);
    let p_reduced = sample_reduced_covariance();
    let p_full = transform_reduced_cov_to_full(&p_reduced, c_es);
    let (f_reduced, g_reduced) = generated_reduced::error_transition(&reduced, imu_reduced);
    let (f_full, g_full) = generated_full::error_transition(&full, imu_full);
    let q_reduced = [1.0e-8_f32; generated_reduced::NOISE_STATES];
    let mut q_full = [0.0_f32; full::NOISE_STATES];
    q_full[0] = q_reduced[3];
    q_full[1] = q_reduced[4];
    q_full[2] = q_reduced[5];
    q_full[3] = q_reduced[0];
    q_full[4] = q_reduced[1];
    q_full[5] = q_reduced[2];
    q_full[6] = q_reduced[9];
    q_full[7] = q_reduced[10];
    q_full[8] = q_reduced[11];
    q_full[9] = q_reduced[6];
    q_full[10] = q_reduced[7];
    q_full[11] = q_reduced[8];
    q_full[18] = q_reduced[12];
    q_full[19] = q_reduced[13];
    q_full[20] = q_reduced[14];

    let p_reduced_next = predict_cov_reduced(&f_reduced, &g_reduced, &p_reduced, &q_reduced);
    let p_full_next = predict_cov_full(&f_full, &g_full, &p_full, &q_full);
    let p_full_as_reduced = transform_full_cov_to_reduced(&p_full_next, c_es);
    let common = [0usize, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let mut max_err = 0.0_f32;
    let mut max_pair = (0usize, 0usize, 0.0_f32, 0.0_f32);
    for &i in &common {
        for &j in &common {
            let err = (p_reduced_next[i][j] - p_full_as_reduced[i][j]).abs();
            if err > max_err {
                max_err = err;
                max_pair = (i, j, p_reduced_next[i][j], p_full_as_reduced[i][j]);
            }
        }
    }
    assert!(
        max_err < 5.0e-6,
        "prediction P mismatch reduced[{}][{}]={} full_as_reduced={} err={}",
        max_pair.0,
        max_pair.1,
        max_pair.2,
        max_pair.3,
        max_err
    );
}

fn sample_reduced_nominal() -> reduced::NominalState {
    reduced::NominalState {
        q0: 0.979_466,
        q1: 0.059_519,
        q2: -0.068_553,
        q3: 0.180_162,
        vn: 8.0,
        ve: -1.5,
        vd: 0.35,
        q_bv0: 0.995_005,
        q_bv1: 0.045_023,
        q_bv2: -0.036_704,
        q_bv3: 0.081_264,
        ..reduced::NominalState::default()
    }
}

fn alternate_reduced_nominal() -> reduced::NominalState {
    reduced::NominalState {
        q0: 0.927_361_85,
        q1: -0.162_134_74,
        q2: 0.307_685_73,
        q3: -0.130_723_83,
        vn: -3.5,
        ve: 7.25,
        vd: -1.2,
        q_bv0: 0.961_256_3,
        q_bv1: -0.037_259_74,
        q_bv2: 0.128_494_34,
        q_bv3: -0.241_015_14,
        ..reduced::NominalState::default()
    }
}

fn sample_full_nominal(reduced: &reduced::NominalState) -> full::NominalState {
    full::NominalState {
        q0: reduced.q0,
        q1: reduced.q1,
        q2: reduced.q2,
        q3: reduced.q3,
        vn: reduced.vn,
        ve: reduced.ve,
        vd: reduced.vd,
        q_bv0: reduced.q_bv0,
        q_bv1: reduced.q_bv1,
        q_bv2: reduced.q_bv2,
        q_bv3: reduced.q_bv3,
        sgx: 1.0,
        sgy: 1.0,
        sgz: 1.0,
        sax: 1.0,
        say: 1.0,
        saz: 1.0,
        ..full::NominalState::default()
    }
}

fn sample_reduced_covariance() -> [[f32; REDUCED_STATES]; REDUCED_STATES] {
    let mut p = [[0.0; REDUCED_STATES]; REDUCED_STATES];
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

fn scalar_update_reduced(
    p: &mut [[f32; REDUCED_STATES]; REDUCED_STATES],
    dx: &mut [f32; REDUCED_STATES],
    h: &[f32; REDUCED_STATES],
    support: &[usize],
    residual: f32,
    r: f32,
) {
    let mut ph = [0.0; REDUCED_STATES];
    let mut s = r;
    let mut hd = 0.0;
    for i in 0..REDUCED_STATES {
        for &state in support {
            ph[i] += p[i][state] * h[state];
        }
    }
    for &state in support {
        s += h[state] * ph[state];
        hd += h[state] * dx[state];
    }
    for i in 0..REDUCED_STATES {
        dx[i] += (ph[i] / s) * (residual - hd);
    }
    for i in 0..REDUCED_STATES {
        for j in i..REDUCED_STATES {
            let updated = p[i][j] - (ph[i] * ph[j]) / s;
            p[i][j] = updated;
            p[j][i] = updated;
        }
    }
}

fn scalar_update_full(
    p: &mut [[f32; FULL_STATES]; FULL_STATES],
    dx: &mut [f32; FULL_STATES],
    h: &[f32; FULL_STATES],
    support: &[usize],
    residual: f32,
    r: f32,
) {
    let mut ph = [0.0; FULL_STATES];
    let mut s = r;
    let mut hd = 0.0;
    for i in 0..FULL_STATES {
        for &state in support {
            ph[i] += p[i][state] * h[state];
        }
    }
    for &state in support {
        s += h[state] * ph[state];
        hd += h[state] * dx[state];
    }
    for i in 0..FULL_STATES {
        dx[i] += (ph[i] / s) * (residual - hd);
    }
    for i in 0..FULL_STATES {
        for j in i..FULL_STATES {
            let updated = p[i][j] - (ph[i] * ph[j]) / s;
            p[i][j] = updated;
            p[j][i] = updated;
        }
    }
}

fn transform_reduced_cov_to_full(
    p_reduced: &[[f32; REDUCED_STATES]; REDUCED_STATES],
    c_es: [[f32; 3]; 3],
) -> [[f32; FULL_STATES]; FULL_STATES] {
    let mut p = [[0.0; FULL_STATES]; FULL_STATES];
    for full_i in 0..FULL_STATES {
        for full_j in 0..FULL_STATES {
            let mut v = 0.0;
            for reduced_i in 0..REDUCED_STATES {
                for reduced_j in 0..REDUCED_STATES {
                    v += full_from_reduced_coeff(full_i, reduced_i, c_es)
                        * p_reduced[reduced_i][reduced_j]
                        * full_from_reduced_coeff(full_j, reduced_j, c_es);
                }
            }
            p[full_i][full_j] = v;
        }
    }
    p
}

fn transform_full_cov_to_reduced(
    p_full: &[[f32; FULL_STATES]; FULL_STATES],
    c_es: [[f32; 3]; 3],
) -> [[f32; REDUCED_STATES]; REDUCED_STATES] {
    let mut p = [[0.0; REDUCED_STATES]; REDUCED_STATES];
    for reduced_i in 0..REDUCED_STATES {
        for reduced_j in 0..REDUCED_STATES {
            let mut v = 0.0;
            for full_i in 0..FULL_STATES {
                for full_j in 0..FULL_STATES {
                    v += reduced_from_full_coeff(reduced_i, full_i, c_es)
                        * p_full[full_i][full_j]
                        * reduced_from_full_coeff(reduced_j, full_j, c_es);
                }
            }
            p[reduced_i][reduced_j] = v;
        }
    }
    p
}

fn transform_full_dx_to_reduced(
    dx_full: &[f32; FULL_STATES],
    c_es: [[f32; 3]; 3],
) -> [f32; REDUCED_STATES] {
    let mut dx = [0.0; REDUCED_STATES];
    for (i, out) in dx.iter_mut().enumerate() {
        for full_i in 0..FULL_STATES {
            *out += reduced_from_full_coeff(i, full_i, c_es) * dx_full[full_i];
        }
    }
    dx
}

fn predict_cov_reduced(
    f: &[[f32; REDUCED_STATES]; REDUCED_STATES],
    g: &[[f32; generated_reduced::NOISE_STATES]; REDUCED_STATES],
    p: &[[f32; REDUCED_STATES]; REDUCED_STATES],
    q: &[f32; generated_reduced::NOISE_STATES],
) -> [[f32; REDUCED_STATES]; REDUCED_STATES] {
    let mut fp = [[0.0; REDUCED_STATES]; REDUCED_STATES];
    for i in 0..REDUCED_STATES {
        for j in 0..REDUCED_STATES {
            for k in 0..REDUCED_STATES {
                fp[i][j] += f[i][k] * p[k][j];
            }
        }
    }
    let mut out = [[0.0; REDUCED_STATES]; REDUCED_STATES];
    for i in 0..REDUCED_STATES {
        for j in 0..REDUCED_STATES {
            for k in 0..REDUCED_STATES {
                out[i][j] += fp[i][k] * f[j][k];
            }
            for (k, qk) in q.iter().enumerate() {
                out[i][j] += g[i][k] * *qk * g[j][k];
            }
        }
    }
    out
}

fn predict_cov_full(
    f: &[[f32; FULL_STATES]; FULL_STATES],
    g: &[[f32; full::NOISE_STATES]; FULL_STATES],
    p: &[[f32; FULL_STATES]; FULL_STATES],
    q: &[f32; full::NOISE_STATES],
) -> [[f32; FULL_STATES]; FULL_STATES] {
    let mut fp = [[0.0; FULL_STATES]; FULL_STATES];
    for i in 0..FULL_STATES {
        for j in 0..FULL_STATES {
            for k in 0..FULL_STATES {
                fp[i][j] += f[i][k] * p[k][j];
            }
        }
    }
    let mut out = [[0.0; FULL_STATES]; FULL_STATES];
    for i in 0..FULL_STATES {
        for j in 0..FULL_STATES {
            for k in 0..FULL_STATES {
                out[i][j] += fp[i][k] * f[j][k];
            }
            for (k, qk) in q.iter().enumerate() {
                out[i][j] += g[i][k] * *qk * g[j][k];
            }
        }
    }
    out
}

fn apply_reduced_runtime_reset(
    p: &mut [[f32; REDUCED_STATES]; REDUCED_STATES],
    dx: &[f32; REDUCED_STATES],
) {
    apply_reset_block_reduced(p, 0, [dx[0], dx[1], dx[2]]);
    symmetrize(p);
}

fn apply_full_runtime_reset(
    p: &mut [[f32; FULL_STATES]; FULL_STATES],
    dx: &[f32; FULL_STATES],
    reset_mount: bool,
) {
    apply_reset_block_full(p, 6, [dx[6], dx[7], dx[8]]);
    if reset_mount {
        apply_reset_block_full(p, 21, [dx[21], dx[22], dx[23]]);
    }
    symmetrize(p);
}

fn apply_reset_block_reduced(
    p: &mut [[f32; REDUCED_STATES]; REDUCED_STATES],
    offset: usize,
    dtheta: [f32; 3],
) {
    let reset = generated_reduced::attitude_reset_jacobian(dtheta);
    apply_reset_block(p, offset, reset);
}

fn apply_reset_block_full(
    p: &mut [[f32; FULL_STATES]; FULL_STATES],
    offset: usize,
    dtheta: [f32; 3],
) {
    let reset = generated_full::reset_jacobian(dtheta);
    apply_reset_block(p, offset, reset);
}

fn apply_reset_block<const N: usize>(p: &mut [[f32; N]; N], offset: usize, reset: [[f32; 3]; 3]) {
    let mut p_aa = [[0.0; 3]; 3];
    let mut p_ab = [[0.0; N]; 3];
    let mut next_aa = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            p_aa[i][j] = p[offset + i][offset + j];
        }
        for j in 0..N {
            p_ab[i][j] = p[offset + i][j];
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                next_aa[i][j] += reset[i][k] * p_aa[k][j];
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let mut accum = 0.0;
            for k in 0..3 {
                accum += next_aa[i][k] * reset[j][k];
            }
            p[offset + i][offset + j] = accum;
            p[offset + j][offset + i] = accum;
        }
    }
    for i in 0..3 {
        for j in 0..N {
            if (offset..offset + 3).contains(&j) {
                continue;
            }
            let mut accum = 0.0;
            for k in 0..3 {
                accum += reset[i][k] * p_ab[k][j];
            }
            p[offset + i][j] = accum;
            p[j][offset + i] = accum;
        }
    }
}

fn symmetrize<const N: usize>(p: &mut [[f32; N]; N]) {
    for i in 0..N {
        for j in (i + 1)..N {
            let avg = 0.5 * (p[i][j] + p[j][i]);
            p[i][j] = avg;
            p[j][i] = avg;
        }
    }
}

fn max_cov_mismatch(
    a: &[[f32; REDUCED_STATES]; REDUCED_STATES],
    b: &[[f32; REDUCED_STATES]; REDUCED_STATES],
    states: &[usize],
) -> f32 {
    let mut max_err = 0.0;
    for &i in states {
        for &j in states {
            let err = (a[i][j] - b[i][j]).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

fn full_from_reduced_coeff(full_idx: usize, reduced_idx: usize, c_es: [[f32; 3]; 3]) -> f32 {
    match (full_idx, reduced_idx) {
        (6..=8, 0..=2) => c_es[full_idx - 6][reduced_idx],
        (3, 3) | (4, 4) | (5, 5) => 1.0,
        (12, 9) | (13, 10) | (14, 11) => -1.0,
        (9, 12) | (10, 13) | (11, 14) => -1.0,
        (21, 15) | (22, 16) | (23, 17) => 1.0,
        _ => 0.0,
    }
}

fn reduced_from_full_coeff(reduced_idx: usize, full_idx: usize, c_es: [[f32; 3]; 3]) -> f32 {
    match (reduced_idx, full_idx) {
        (0..=2, 6..=8) => c_es[full_idx - 6][reduced_idx],
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

fn assert_close(label: &str, reduced_idx: usize, full_idx: usize, reduced: f32, full: f32) {
    let err = (reduced - full).abs();
    assert!(
        err < 1.0e-3,
        "{label} Reduced[{reduced_idx}] vs full[{full_idx}] mismatch: reduced={reduced} full={full} err={err}"
    );
}
