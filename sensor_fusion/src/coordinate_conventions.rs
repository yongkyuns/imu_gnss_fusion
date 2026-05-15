use crate::math::{
    euler_to_quat_f32, mat_mul3_f32, mat_vec3_f32, quat_conj_f32, quat_multiply_f32,
    quat_to_dcm_f32, transpose3_f32,
};
use crate::{GnssSample, SensorFusion};

const EPS: f32 = 1.0e-5;

fn gnss_sample(t_s: f32) -> GnssSample {
    GnssSample {
        t_s,
        lat_deg: 42.0,
        lon_deg: -71.0,
        height_m: 12.0,
        vel_ned_mps: [5.0, 0.0, 0.0],
        pos_std_m: [1.0, 1.0, 1.5],
        vel_std_mps: [0.2, 0.2, 0.2],
        heading_rad: Some(0.0),
    }
}

#[test]
fn active_quaternion_rotation_matrix_is_c_ab() {
    let q_ab = euler_to_quat_f32(0.0, 0.0, core::f32::consts::FRAC_PI_2);
    let c_ab = quat_to_dcm_f32(q_ab);

    assert_vec_close(
        mat_vec3_f32(c_ab, [1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
        "C_ab x_b",
    );
    assert_vec_close(
        mat_vec3_f32(c_ab, [0.0, 1.0, 0.0]),
        [-1.0, 0.0, 0.0],
        "C_ab y_b",
    );

    let q_bc = euler_to_quat_f32(0.25, 0.0, 0.0);
    let c_bc = quat_to_dcm_f32(q_bc);
    let q_ac = quat_multiply_f32(q_ab, q_bc);
    assert_mat_close(
        quat_to_dcm_f32(q_ac),
        mat_mul3_f32(c_ab, c_bc),
        "R(q_ab q_bc)",
    );
}

#[test]
fn public_manual_mount_q_bv_maps_vehicle_vectors_to_body_vectors() {
    let q_bv = euler_to_quat_f32(0.0, 0.0, core::f32::consts::FRAC_PI_2);
    let mut ekf_system = SensorFusion::with_mount(q_bv);

    assert_eq!(ekf_system.mount_q_bv(), Some(q_bv));
    assert_eq!(ekf_system.ekf_mount_q_bv(), Some(q_bv));
    assert_vec_close(
        mat_vec3_f32(
            quat_to_dcm_f32(ekf_system.mount_q_bv().unwrap()),
            [1.0, 0.0, 0.0],
        ),
        [0.0, 1.0, 0.0],
        "public q_bv maps x_v into x_b",
    );

    let update = ekf_system.process_gnss(gnss_sample(1.0));
    assert_eq!(update.mount_q_bv, Some(q_bv));
    let ekf = ekf_system.ekf().unwrap();
    assert_quat_close(
        [
            ekf.nominal.q_bv0,
            ekf.nominal.q_bv1,
            ekf.nominal.q_bv2,
            ekf.nominal.q_bv3,
        ],
        q_bv,
        "EKF q_bv stores q_bv",
    );
}

#[test]
fn inverse_mount_is_the_quaternion_conjugate() {
    let q_bv = quat_multiply_f32(
        euler_to_quat_f32(0.0, 0.0, 0.6),
        euler_to_quat_f32(-0.35, 0.0, 0.0),
    );
    let q_vb = quat_conj_f32(q_bv);
    let c_bv = quat_to_dcm_f32(q_bv);
    let c_vb = quat_to_dcm_f32(q_vb);

    assert_quat_close(
        quat_multiply_f32(q_bv, q_vb),
        [1.0, 0.0, 0.0, 0.0],
        "q_bv q_vb",
    );
    assert_mat_close(c_vb, transpose3_f32(c_bv), "C_vb");

    let x_v = [0.7, -0.2, 0.4];
    let x_b = mat_vec3_f32(c_bv, x_v);
    assert_vec_close(mat_vec3_f32(c_vb, x_b), x_v, "C_vb C_bv x_v");
}

fn assert_quat_close(actual: [f32; 4], expected: [f32; 4], label: &str) {
    for i in 0..4 {
        assert!(
            (actual[i] - expected[i]).abs() < EPS,
            "{label}[{i}] actual={} expected={}",
            actual[i],
            expected[i]
        );
    }
}

fn assert_vec_close(actual: [f32; 3], expected: [f32; 3], label: &str) {
    for i in 0..3 {
        assert!(
            (actual[i] - expected[i]).abs() < EPS,
            "{label}[{i}] actual={} expected={}",
            actual[i],
            expected[i]
        );
    }
}

fn assert_mat_close(actual: [[f32; 3]; 3], expected: [[f32; 3]; 3], label: &str) {
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (actual[r][c] - expected[r][c]).abs() < EPS,
                "{label}[{r}][{c}] actual={} expected={}",
                actual[r][c],
                expected[r][c]
            );
        }
    }
}
