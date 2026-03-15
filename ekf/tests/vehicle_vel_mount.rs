use ekf_rs::ekf::{Ekf, ekf_fuse_vehicle_vel};

fn quat_from_yaw(yaw_rad: f32) -> [f32; 4] {
    let half = 0.5 * yaw_rad;
    [half.cos(), 0.0, 0.0, half.sin()]
}

fn quat_to_rotmat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn transpose3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mat3_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn vehicle_lateral_velocity(ekf: &Ekf, q_vb: [f32; 4]) -> f32 {
    let c_n_b = quat_to_rotmat([ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3]);
    let c_b_n = transpose3(c_n_b);
    let v_b = mat3_vec(c_b_n, [ekf.state.vn, ekf.state.ve, ekf.state.vd]);
    let v_v = mat3_vec(quat_to_rotmat(q_vb), v_b);
    v_v[1]
}

#[test]
fn vehicle_vel_update_is_noop_when_mount_and_nav_are_consistent() {
    let q_vb = quat_from_yaw(20.0_f32.to_radians());
    let mut ekf = Ekf::default();
    ekf.state.q0 = q_vb[0];
    ekf.state.q1 = q_vb[1];
    ekf.state.q2 = q_vb[2];
    ekf.state.q3 = q_vb[3];
    ekf.state.vn = 12.0;

    let lat0 = vehicle_lateral_velocity(&ekf, q_vb);
    ekf_fuse_vehicle_vel(&mut ekf, q_vb, 1.0);
    let lat1 = vehicle_lateral_velocity(&ekf, q_vb);

    assert!(lat0.abs() < 1.0e-5);
    assert!(lat1.abs() < 1.0e-4);
}

#[test]
fn vehicle_vel_update_reduces_vehicle_lateral_velocity_for_wrong_nav_yaw() {
    let q_vb = quat_from_yaw(20.0_f32.to_radians());
    let mut ekf = Ekf::default();
    ekf.state.q0 = 1.0;
    ekf.state.vn = 12.0;

    let lat0 = vehicle_lateral_velocity(&ekf, q_vb).abs();
    for _ in 0..5 {
        ekf_fuse_vehicle_vel(&mut ekf, q_vb, 1.0);
    }
    let lat1 = vehicle_lateral_velocity(&ekf, q_vb).abs();

    assert!(lat1 < lat0, "vehicle lateral velocity did not decrease: {} -> {}", lat0, lat1);
}
