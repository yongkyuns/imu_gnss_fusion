use align_rs::align::{
    Align, MisalignImuSample, MisalignNoise, align_fuse_velocity_forward, align_init,
    align_predict_gyro, align_q_sb, align_set_q_sb,
};

fn deg2rad(v: f32) -> f32 {
    v * std::f32::consts::PI / 180.0
}

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    let inv = n2.sqrt().recip();
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

fn quat_from_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let h = 0.5 * angle;
    let s = h.sin();
    quat_normalize([h.cos(), axis[0] * s, axis[1] * s, axis[2] * s])
}

fn quat_to_rotmat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let q = quat_normalize(q);
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
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

fn mat_vec(a: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn body_forward_in_sensor(q_sb: [f32; 4]) -> [f32; 3] {
    let r = quat_to_rotmat(q_sb);
    [r[0][0], r[1][0], r[2][0]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn gravity_sensor(r_sb: [[f32; 3]; 3]) -> [f32; 3] {
    mat_vec(r_sb, [0.0, 0.0, 9.80665])
}

fn body_acc_sensor(r_sb: [[f32; 3]; 3], ax: f32) -> [f32; 3] {
    mat_vec(r_sb, [ax, 0.0, 9.80665])
}

#[test]
fn forward_axis_update_resolves_pi_yaw_branch() {
    let mut f = Align::default();
    align_init(
        &mut f,
        [0.2, 0.2, 0.2],
        MisalignNoise {
            q_theta_rw_var: 1.0e-7,
        },
    );

    let q_true = quat_from_axis_angle([0.0, 0.0, 1.0], deg2rad(32.0));
    let q_flip = quat_mul(
        q_true,
        quat_from_axis_angle([0.0, 0.0, 1.0], std::f32::consts::PI),
    );
    let r_sb_true = quat_to_rotmat(q_true);
    let dt = 0.01_f32;

    for _ in 0..120 {
        let g_s = gravity_sensor(r_sb_true);
        let imu = MisalignImuSample {
            dt,
            f_sx: g_s[0],
            f_sy: g_s[1],
            f_sz: g_s[2],
        };
        align_predict_gyro(&mut f, &imu, 0.0, 0.0, 0.0);
    }

    align_set_q_sb(&mut f, q_flip);

    let mut speed = 0.0_f32;
    for k in 0..900 {
        speed += 0.8 * dt;
        let sensor_acc = body_acc_sensor(r_sb_true, 0.8);
        let imu = MisalignImuSample {
            dt,
            f_sx: sensor_acc[0],
            f_sy: sensor_acc[1],
            f_sz: sensor_acc[2],
        };
        align_predict_gyro(&mut f, &imu, 0.0, 0.0, 0.0);

        if k % 20 == 0 {
            align_fuse_velocity_forward(
                &mut f,
                [speed, 0.0, 0.0],
                [0.05, 0.05, 0.05],
                [0.01, 0.01, 0.01],
            );
        }
    }

    let est_forward = body_forward_in_sensor(align_q_sb(&f));
    let true_forward = body_forward_in_sensor(q_true);
    let flipped_forward = body_forward_in_sensor(q_flip);

    assert!(
        dot(est_forward, true_forward) > 0.98,
        "estimated forward axis did not converge to the true branch"
    );
    assert!(
        dot(est_forward, flipped_forward) < -0.98,
        "estimated forward axis stayed near the 180 deg flipped branch"
    );
}

#[test]
fn forward_axis_update_does_not_flip_on_single_bad_sample() {
    let mut f = Align::default();
    align_init(
        &mut f,
        [0.2, 0.2, 0.2],
        MisalignNoise {
            q_theta_rw_var: 1.0e-7,
        },
    );

    let q_true = quat_from_axis_angle([0.0, 0.0, 1.0], deg2rad(18.0));
    let r_sb_true = quat_to_rotmat(q_true);
    let dt = 0.01_f32;

    for _ in 0..120 {
        let g_s = gravity_sensor(r_sb_true);
        let imu = MisalignImuSample {
            dt,
            f_sx: g_s[0],
            f_sy: g_s[1],
            f_sz: g_s[2],
        };
        align_predict_gyro(&mut f, &imu, 0.0, 0.0, 0.0);
    }
    align_set_q_sb(&mut f, q_true);

    let mut speed = 8.0_f32;
    for _ in 0..3 {
        for _ in 0..20 {
            let sensor_acc = body_acc_sensor(r_sb_true, 0.8);
            let imu = MisalignImuSample {
                dt,
                f_sx: sensor_acc[0],
                f_sy: sensor_acc[1],
                f_sz: sensor_acc[2],
            };
            align_predict_gyro(&mut f, &imu, 0.0, 0.0, 0.0);
        }
        speed += 0.8 * 0.2;
        align_fuse_velocity_forward(
            &mut f,
            [speed, 0.0, 0.0],
            [0.05, 0.05, 0.05],
            [0.01, 0.01, 0.01],
        );
    }

    for _ in 0..6 {
        for _ in 0..20 {
            let sensor_acc = body_acc_sensor(r_sb_true, -0.8);
            let imu = MisalignImuSample {
                dt,
                f_sx: sensor_acc[0],
                f_sy: sensor_acc[1],
                f_sz: sensor_acc[2],
            };
            align_predict_gyro(&mut f, &imu, 0.0, 0.0, 0.0);
        }
        speed -= 0.8 * 0.2;
        align_fuse_velocity_forward(
            &mut f,
            [speed, 0.0, 0.0],
            [0.05, 0.05, 0.05],
            [0.01, 0.01, 0.01],
        );
    }

    let est_forward = body_forward_in_sensor(align_q_sb(&f));
    let true_forward = body_forward_in_sensor(q_true);
    assert!(
        dot(est_forward, true_forward) > 0.98,
        "a single contradictory forward sample should not flip the yaw branch"
    );
}

#[test]
fn installation_solution_freezes_after_branch_lock() {
    let mut f = Align::default();
    align_init(
        &mut f,
        [0.2, 0.2, 0.2],
        MisalignNoise {
            q_theta_rw_var: 1.0e-7,
        },
    );

    let q_true = quat_from_axis_angle([0.0, 0.0, 1.0], deg2rad(12.0));
    let r_sb_true = quat_to_rotmat(q_true);
    let dt = 0.01_f32;

    for _ in 0..120 {
        let g_s = gravity_sensor(r_sb_true);
        let imu = MisalignImuSample {
            dt,
            f_sx: g_s[0],
            f_sy: g_s[1],
            f_sz: g_s[2],
        };
        align_predict_gyro(&mut f, &imu, 0.0, 0.0, 0.0);
    }
    align_set_q_sb(&mut f, q_true);

    let mut speed = 8.0_f32;
    for _ in 0..3 {
        for _ in 0..20 {
            let sensor_acc = body_acc_sensor(r_sb_true, 0.8);
            let imu = MisalignImuSample {
                dt,
                f_sx: sensor_acc[0],
                f_sy: sensor_acc[1],
                f_sz: sensor_acc[2],
            };
            align_predict_gyro(&mut f, &imu, 0.0, 0.0, 0.0);
        }
        speed += 0.8 * 0.2;
        align_fuse_velocity_forward(
            &mut f,
            [speed, 0.0, 0.0],
            [0.05, 0.05, 0.05],
            [0.01, 0.01, 0.01],
        );
    }

    let q_locked = align_q_sb(&f);

    for _ in 0..200 {
        let sensor_acc = body_acc_sensor(r_sb_true, -1.2);
        let imu = MisalignImuSample {
            dt,
            f_sx: sensor_acc[0],
            f_sy: sensor_acc[1],
            f_sz: sensor_acc[2],
        };
        align_predict_gyro(&mut f, &imu, 0.2, -0.1, 0.3);
    }
    align_fuse_velocity_forward(
        &mut f,
        [speed - 1.2 * 2.0, 0.0, 0.0],
        [0.05, 0.05, 0.05],
        [0.01, 0.01, 0.01],
    );

    let q_after = align_q_sb(&f);
    let max_abs_delta = [
        (q_after[0] - q_locked[0]).abs(),
        (q_after[1] - q_locked[1]).abs(),
        (q_after[2] - q_locked[2]).abs(),
        (q_after[3] - q_locked[3]).abs(),
    ]
    .into_iter()
    .fold(0.0_f32, f32::max);

    assert!(
        max_abs_delta < 1.0e-6,
        "installation quaternion should stay fixed after branch lock"
    );
}
