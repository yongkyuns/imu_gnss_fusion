use sensor_fusion::eskf::{
    ERROR_STATE_DIM, ErrorState, IDX_DBA_X, IDX_DBG_X, IDX_DPOS_N, IDX_DPSI_CS_X, IDX_DTHETA_Z,
    IDX_DVEL_N, ImuDelta, NominalState, error_reset_jacobian,
};

fn quat_norm(q: [f32; 4]) -> f32 {
    (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
}

#[test]
fn inject_zero_error_is_noop() {
    let mut x = NominalState::identity();
    x.vel_n = [1.0, 2.0, 3.0];
    x.pos_n = [4.0, 5.0, 6.0];
    x.gyro_bias_b = [0.1, 0.2, 0.3];
    x.accel_bias_b = [0.4, 0.5, 0.6];
    let before = x;

    x.inject_error(ErrorState::default());

    assert_eq!(x, before);
}

#[test]
fn inject_error_updates_nominal_translational_and_bias_states() {
    let mut x = NominalState::identity();
    x.inject_error(ErrorState {
        dtheta_b: [0.0, 0.0, 0.0],
        dvel_n: [1.0, -2.0, 3.0],
        dpos_n: [4.0, -5.0, 6.0],
        dgyro_bias_b: [0.01, -0.02, 0.03],
        daccel_bias_b: [0.04, -0.05, 0.06],
        dpsi_cs: [0.0, 0.0, 0.0],
    });

    assert_eq!(x.vel_n, [1.0, -2.0, 3.0]);
    assert_eq!(x.pos_n, [4.0, -5.0, 6.0]);
    assert_eq!(x.gyro_bias_b, [0.01, -0.02, 0.03]);
    assert_eq!(x.accel_bias_b, [0.04, -0.05, 0.06]);
}

#[test]
fn inject_small_attitude_error_rotates_quaternion_and_keeps_unit_norm() {
    let mut x = NominalState::identity();
    x.inject_error(ErrorState {
        dtheta_b: [0.0, 0.0, 0.1],
        ..ErrorState::default()
    });

    assert!(x.q_bn[3] > 0.0);
    assert!((quat_norm(x.q_bn) - 1.0).abs() < 1.0e-6);
}

#[test]
fn opposite_small_angle_injections_nearly_cancel() {
    let mut x = NominalState::identity();
    x.inject_error(ErrorState {
        dtheta_b: [0.01, -0.02, 0.03],
        ..ErrorState::default()
    });
    x.inject_error(ErrorState {
        dtheta_b: [-0.01, 0.02, -0.03],
        ..ErrorState::default()
    });

    assert!((x.q_bn[0] - 1.0).abs() < 1.0e-4);
    assert!(x.q_bn[1].abs() < 1.0e-4);
    assert!(x.q_bn[2].abs() < 1.0e-4);
    assert!(x.q_bn[3].abs() < 1.0e-4);
}

#[test]
fn error_state_array_round_trip_preserves_layout() {
    let dx = ErrorState {
        dtheta_b: [1.0, 2.0, 3.0],
        dvel_n: [4.0, 5.0, 6.0],
        dpos_n: [7.0, 8.0, 9.0],
        dgyro_bias_b: [10.0, 11.0, 12.0],
        daccel_bias_b: [13.0, 14.0, 15.0],
        dpsi_cs: [16.0, 17.0, 18.0],
    };
    let flat = dx.to_array();
    assert_eq!(flat.len(), ERROR_STATE_DIM);
    assert_eq!(flat[IDX_DTHETA_Z], 3.0);
    assert_eq!(flat[IDX_DVEL_N], 4.0);
    assert_eq!(flat[IDX_DPOS_N], 7.0);
    assert_eq!(flat[IDX_DBG_X], 10.0);
    assert_eq!(flat[IDX_DBA_X], 13.0);
    assert_eq!(flat[IDX_DPSI_CS_X], 16.0);
    assert_eq!(ErrorState::from_array(flat), dx);
}

#[test]
fn inject_small_mount_error_rotates_residual_mount_quaternion() {
    let mut x = NominalState::identity();
    x.inject_error(ErrorState {
        dpsi_cs: [0.0, 0.0, 0.1],
        ..ErrorState::default()
    });

    assert!(x.q_cs[3] > 0.0);
    assert!((quat_norm(x.q_cs) - 1.0).abs() < 1.0e-6);
    assert_eq!(x.q_bn, [1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn reset_jacobian_is_identity_at_zero_error() {
    let g = error_reset_jacobian([0.0, 0.0, 0.0]);
    for (r, row) in g.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            let expected = if r == c { 1.0 } else { 0.0 };
            assert!((*value - expected).abs() < 1.0e-6);
        }
    }
}

#[test]
fn reset_jacobian_attitude_block_matches_first_order_small_angle_form() {
    let dtheta = [0.2, -0.4, 0.6];
    let g = error_reset_jacobian(dtheta);

    let expected = [[1.0, 0.3, 0.2], [-0.3, 1.0, 0.1], [-0.2, -0.1, 1.0]];

    for r in 0..3 {
        for c in 0..3 {
            assert!((g[r][c] - expected[r][c]).abs() < 1.0e-6);
        }
    }
}

#[test]
fn predict_keeps_rest_state_with_gravity_compensated_delta_velocity() {
    let mut x = NominalState::identity();
    let dt = 0.01;
    let gravity_n = [0.0, 0.0, 9.81];

    x.predict(
        ImuDelta {
            dtheta_b: [0.0, 0.0, 0.0],
            dvel_b: [0.0, 0.0, -gravity_n[2] * dt],
            dt,
        },
        gravity_n,
    );

    assert!(x.vel_n[0].abs() < 1.0e-6);
    assert!(x.vel_n[1].abs() < 1.0e-6);
    assert!(x.vel_n[2].abs() < 1.0e-6);
}

#[test]
fn predict_gyro_bias_cancels_matching_delta_angle_input() {
    let mut x = NominalState::identity();
    x.gyro_bias_b = [0.3, -0.2, 0.1];
    let dt = 0.01;
    let imu = ImuDelta {
        dtheta_b: [
            x.gyro_bias_b[0] * dt,
            x.gyro_bias_b[1] * dt,
            x.gyro_bias_b[2] * dt,
        ],
        dvel_b: [0.0, 0.0, 0.0],
        dt,
    };

    x.predict(imu, [0.0, 0.0, 0.0]);

    assert!((x.q_bn[0] - 1.0).abs() < 1.0e-6);
    assert!(x.q_bn[1].abs() < 1.0e-6);
    assert!(x.q_bn[2].abs() < 1.0e-6);
    assert!(x.q_bn[3].abs() < 1.0e-6);
}

#[test]
fn predict_accel_bias_cancels_matching_delta_velocity_input() {
    let mut x = NominalState::identity();
    x.accel_bias_b = [0.5, -0.4, 0.3];
    let dt = 0.02;
    let imu = ImuDelta {
        dtheta_b: [0.0, 0.0, 0.0],
        dvel_b: [
            x.accel_bias_b[0] * dt,
            x.accel_bias_b[1] * dt,
            x.accel_bias_b[2] * dt,
        ],
        dt,
    };

    x.predict(imu, [0.0, 0.0, 0.0]);

    assert!(x.vel_n[0].abs() < 1.0e-6);
    assert!(x.vel_n[1].abs() < 1.0e-6);
    assert!(x.vel_n[2].abs() < 1.0e-6);
}
