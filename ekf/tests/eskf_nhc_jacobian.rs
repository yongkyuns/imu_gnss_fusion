use sensor_fusion::eskf_types::EskfNominalState;
use sensor_fusion::generated_eskf::{self, ERROR_STATES};

#[test]
fn body_velocity_yz_jacobians_match_nominal_model_finite_difference() {
    let nominal = EskfNominalState {
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
    let p = [[0.0; ERROR_STATES]; ERROR_STATES];
    let obs_y = generated_eskf::body_vel_y_observation(&nominal, &p, 1.0);
    let obs_z = generated_eskf::body_vel_z_observation(&nominal, &p, 1.0);

    for state in [0usize, 1, 2, 3, 4, 5, 15, 16, 17] {
        let dy = finite_difference(&nominal, state, 1);
        let dz = finite_difference(&nominal, state, 2);
        assert_close("body Y", state, obs_y.h[state], dy);
        assert_close("body Z", state, obs_z.h[state], dz);
    }
}

fn finite_difference(nominal: &EskfNominalState, state: usize, component: usize) -> f32 {
    let eps = 1.0e-4_f32;
    let mut plus = *nominal;
    let mut minus = *nominal;
    apply_error(&mut plus, state, eps);
    apply_error(&mut minus, state, -eps);
    (vehicle_velocity(&plus)[component] - vehicle_velocity(&minus)[component]) / (2.0 * eps)
}

fn apply_error(nominal: &mut EskfNominalState, state: usize, dx: f32) {
    match state {
        0..=2 => {
            let mut dq = [1.0, 0.0, 0.0, 0.0];
            dq[state + 1] = 0.5 * dx;
            let q = quat_mul([nominal.q0, nominal.q1, nominal.q2, nominal.q3], dq);
            [nominal.q0, nominal.q1, nominal.q2, nominal.q3] = normalize(q);
        }
        3 => nominal.vn += dx,
        4 => nominal.ve += dx,
        5 => nominal.vd += dx,
        15..=17 => {
            let mut dq = [1.0, 0.0, 0.0, 0.0];
            dq[state - 14] = 0.5 * dx;
            let q = quat_mul(dq, [nominal.qcs0, nominal.qcs1, nominal.qcs2, nominal.qcs3]);
            [nominal.qcs0, nominal.qcs1, nominal.qcs2, nominal.qcs3] = normalize(q);
        }
        _ => {}
    }
}

fn vehicle_velocity(n: &EskfNominalState) -> [f32; 3] {
    let vs0 = (1.0 - 2.0 * n.q2 * n.q2 - 2.0 * n.q3 * n.q3) * n.vn
        + 2.0 * (n.q1 * n.q2 + n.q0 * n.q3) * n.ve
        + 2.0 * (n.q1 * n.q3 - n.q0 * n.q2) * n.vd;
    let vs1 = 2.0 * (n.q1 * n.q2 - n.q0 * n.q3) * n.vn
        + (1.0 - 2.0 * n.q1 * n.q1 - 2.0 * n.q3 * n.q3) * n.ve
        + 2.0 * (n.q2 * n.q3 + n.q0 * n.q1) * n.vd;
    let vs2 = 2.0 * (n.q1 * n.q3 + n.q0 * n.q2) * n.vn
        + 2.0 * (n.q2 * n.q3 - n.q0 * n.q1) * n.ve
        + (1.0 - 2.0 * n.q1 * n.q1 - 2.0 * n.q2 * n.q2) * n.vd;
    [
        (1.0 - 2.0 * n.qcs2 * n.qcs2 - 2.0 * n.qcs3 * n.qcs3) * vs0
            + 2.0 * (n.qcs1 * n.qcs2 - n.qcs0 * n.qcs3) * vs1
            + 2.0 * (n.qcs1 * n.qcs3 + n.qcs0 * n.qcs2) * vs2,
        2.0 * (n.qcs1 * n.qcs2 + n.qcs0 * n.qcs3) * vs0
            + (1.0 - 2.0 * n.qcs1 * n.qcs1 - 2.0 * n.qcs3 * n.qcs3) * vs1
            + 2.0 * (n.qcs2 * n.qcs3 - n.qcs0 * n.qcs1) * vs2,
        2.0 * (n.qcs1 * n.qcs3 - n.qcs0 * n.qcs2) * vs0
            + 2.0 * (n.qcs2 * n.qcs3 + n.qcs0 * n.qcs1) * vs1
            + (1.0 - 2.0 * n.qcs1 * n.qcs1 - 2.0 * n.qcs2 * n.qcs2) * vs2,
    ]
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn normalize(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn assert_close(label: &str, state: usize, generated: f32, finite_diff: f32) {
    let err = (generated - finite_diff).abs();
    assert!(
        err < 2.0e-2,
        "{label} H[{state}] mismatch: generated={generated} finite_diff={finite_diff} err={err}"
    );
}
