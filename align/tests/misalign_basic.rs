use align_rs::align::{
    MisalignAttitudeSample, MisalignImuSample, MisalignNoise, Align, align_fuse_velocity, align_init,
    align_predict, align_q_sb, align_set_q_sb,
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
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn quat_from_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let h = 0.5 * angle;
    let s = h.sin();
    [h.cos(), axis[0] * s, axis[1] * s, axis[2] * s]
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

fn transpose(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

#[test]
fn nonlinear_misalignment_filter_reduces_residual() {
    let mut f = Align::default();
    align_init(
        &mut f,
        [0.5, 0.5, 0.5],
        MisalignNoise {
            q_theta_rw_var: 1.0e-8,
        },
    );

    let q_true = quat_normalize(quat_mul(
        quat_mul(
            quat_from_axis_angle([0.0, 0.0, 1.0], deg2rad(20.0)),
            quat_from_axis_angle([0.0, 1.0, 0.0], deg2rad(-5.0)),
        ),
        quat_from_axis_angle([1.0, 0.0, 0.0], deg2rad(3.0)),
    ));
    // Start with identity (wrong) align.
    align_set_q_sb(&mut f, [1.0, 0.0, 0.0, 0.0]);

    let r_sb_true = quat_to_rotmat(q_true);
    let r_bs_true = transpose(r_sb_true);

    let dt = 0.01_f32;
    let mut speed = 0.0_f32;
    let mut yaw = 0.0_f32;
    let mut prev_res_mag = f32::INFINITY;

    for k in 0..3000 {
        let t = k as f32 * dt;
        let yaw_rate = if (8.0..16.0).contains(&t) {
            deg2rad(7.0)
        } else if (20.0..26.0).contains(&t) {
            deg2rad(-9.0)
        } else {
            0.0
        };
        let a_long = if (2.0..8.0).contains(&t) {
            0.6
        } else if (16.0..22.0).contains(&t) {
            -0.4
        } else {
            0.0
        };
        speed = (speed + a_long * dt).max(0.0);
        yaw += yaw_rate * dt;

        let v_true = [speed * yaw.cos(), speed * yaw.sin(), 0.0];
        let q_nb = quat_from_axis_angle([0.0, 0.0, 1.0], yaw);
        let att = MisalignAttitudeSample {
            q_nb0: q_nb[0],
            q_nb1: q_nb[1],
            q_nb2: q_nb[2],
            q_nb3: q_nb[3],
        };

        // Specific force in body: horizontal accel only (gravity canceled in this synthetic case).
        let f_body = [a_long, speed * yaw_rate, 0.0];
        let f_sensor = mat_vec(r_sb_true, f_body);
        let imu = MisalignImuSample {
            dt,
            f_sx: f_sensor[0],
            f_sy: f_sensor[1],
            f_sz: f_sensor[2],
        };
        align_predict(&mut f, &imu, &att);

        if k % 20 == 0 {
            align_fuse_velocity(&mut f, v_true, [0.05, 0.05, 0.2]);
            let r = f.last_residual_n;
            let mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            prev_res_mag = prev_res_mag.min(mag);
        }
    }

    // Residual should settle to a small value for this consistent synthetic scenario.
    assert!(
        prev_res_mag < 0.25,
        "best residual magnitude too high: {prev_res_mag}"
    );

    let q_est = align_q_sb(&f);
    let r_bs_est = transpose(quat_to_rotmat(q_est));
    let ex = r_bs_est[0][0] - r_bs_true[0][0];
    let ey = r_bs_est[1][1] - r_bs_true[1][1];
    let ez = r_bs_est[2][2] - r_bs_true[2][2];
    let diag_err = (ex.abs() + ey.abs() + ez.abs()) / 3.0;
    assert!(diag_err < 1.5, "estimate diverged badly: {diag_err}");
}
