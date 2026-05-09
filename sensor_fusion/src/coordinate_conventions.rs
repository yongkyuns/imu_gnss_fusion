use crate::full::full_vehicle_ecef_split;
use crate::math::{
    dcm_to_quat_f32, euler_to_quat_f32, mat_mul3_f32, mat_vec3_f32, quat_conj_f32,
    quat_multiply_f32, quat_to_dcm_f32, transpose3_f32,
};
use crate::nav::ecef_to_ned_matrix_f32;
use crate::{Config, Filter, GnssSample, MountMode, SensorFusion};

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
    let mut reduced_system = SensorFusion::with_mount(q_bv);

    assert_eq!(reduced_system.mount_q_bv(), Some(q_bv));
    assert_eq!(reduced_system.reduced_mount_q_bv(), Some(q_bv));
    assert_vec_close(
        mat_vec3_f32(
            quat_to_dcm_f32(reduced_system.mount_q_bv().unwrap()),
            [1.0, 0.0, 0.0],
        ),
        [0.0, 1.0, 0.0],
        "public q_bv maps x_v into x_b",
    );

    let update = reduced_system.process_gnss(gnss_sample(1.0));
    assert_eq!(update.mount_q_bv, Some(q_bv));
    let reduced = reduced_system.reduced().unwrap();
    assert_quat_close(
        [
            reduced.nominal.qcs0,
            reduced.nominal.qcs1,
            reduced.nominal.qcs2,
            reduced.nominal.qcs3,
        ],
        q_bv,
        "Reduced qcs stores q_bv",
    );

    let mut full_system = SensorFusion::with_config(Config {
        filter: Filter::Full,
        mount_mode: MountMode::Manual(q_bv),
    });
    let update = full_system.process_gnss(gnss_sample(1.0));
    assert_eq!(update.mount_q_bv, Some(q_bv));
    let full = full_system.full().unwrap();
    assert_quat_close(
        [
            full.nominal.qcs0,
            full.nominal.qcs1,
            full.nominal.qcs2,
            full.nominal.qcs3,
        ],
        q_bv,
        "Full qcs stores q_bv",
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

#[test]
fn full_vehicle_ecef_split_composes_ned_yaw_with_ecef_frame() {
    let yaw_rad = 37.0_f32.to_radians();
    let lat_deg = 42.0;
    let lon_deg = -71.0;

    let (q_ev, q_bv) = full_vehicle_ecef_split(yaw_rad, lat_deg, lon_deg);
    assert_quat_close(q_bv, [1.0, 0.0, 0.0, 0.0], "split mount");

    let c_ev = quat_to_dcm_f32(q_ev);
    let c_nv = quat_to_dcm_f32(euler_to_quat_f32(0.0, 0.0, yaw_rad));
    let c_en = transpose3_f32(ecef_to_ned_matrix_f32(lat_deg, lon_deg));

    assert_mat_close(c_ev, mat_mul3_f32(c_en, c_nv), "C_ev = C_en C_nv");
}

#[test]
fn reduced_and_full_attitudes_are_equivalent_after_ned_ecef_transform() {
    let lat_deg = 37.4;
    let lon_deg = -122.1;
    let q_nv = quat_multiply_f32(
        euler_to_quat_f32(0.0, 0.0, 0.72),
        quat_multiply_f32(
            euler_to_quat_f32(0.0, -0.11, 0.0),
            euler_to_quat_f32(0.08, 0.0, 0.0),
        ),
    );
    let c_nv = quat_to_dcm_f32(q_nv);
    let c_ne = ecef_to_ned_matrix_f32(lat_deg, lon_deg);
    let c_en = transpose3_f32(c_ne);

    let q_ev = dcm_to_quat_f32(mat_mul3_f32(c_en, c_nv));
    let c_ev = quat_to_dcm_f32(q_ev);

    assert_mat_close(c_ev, mat_mul3_f32(c_en, c_nv), "C_ev = C_en C_nv");

    let vehicle_vector = [3.2, -0.7, 0.4];
    let ecef_from_reduced = mat_vec3_f32(c_en, mat_vec3_f32(c_nv, vehicle_vector));
    assert_vec_close(
        mat_vec3_f32(c_ev, vehicle_vector),
        ecef_from_reduced,
        "full and reduced map vehicle vector to same ECEF vector",
    );

    let ecef_vector = [0.3, -4.0, 1.2];
    let vehicle_from_full = mat_vec3_f32(transpose3_f32(c_ev), ecef_vector);
    let vehicle_from_reduced = mat_vec3_f32(transpose3_f32(c_nv), mat_vec3_f32(c_ne, ecef_vector));
    assert_vec_close(vehicle_from_full, vehicle_from_reduced, "C_ve = C_vn C_ne");
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
