use proptest::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use align_rs::align::{
    MisalignImuSample, MisalignNoise, Align, align_fuse_velocity, align_init, align_predict, align_q_sb,
    align_set_q_sb,
};

fn deg2rad(v: f32) -> f32 {
    v * std::f32::consts::PI / 180.0
}

fn rad2deg(v: f32) -> f32 {
    v * 180.0 / std::f32::consts::PI
}

fn wrap180(mut v: f32) -> f32 {
    while v > 180.0 {
        v -= 360.0;
    }
    while v <= -180.0 {
        v += 360.0;
    }
    v
}

fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2 <= 1.0e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let inv = n2.sqrt().recip();
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

fn quat_from_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let h = 0.5 * angle;
    let s = h.sin();
    quat_normalize([h.cos(), axis[0] * s, axis[1] * s, axis[2] * s])
}

// ESF-ALG convention in this codebase: intrinsic ZYX (equiv. Rx * Ry * Rz composition).
fn quat_from_alg_deg(roll_deg: f32, pitch_deg: f32, yaw_deg: f32) -> [f32; 4] {
    let qx = quat_from_axis_angle([1.0, 0.0, 0.0], deg2rad(roll_deg));
    let qy = quat_from_axis_angle([0.0, 1.0, 0.0], deg2rad(pitch_deg));
    let qz = quat_from_axis_angle([0.0, 0.0, 1.0], deg2rad(yaw_deg));
    quat_normalize(quat_mul(quat_mul(qx, qy), qz))
}

fn quat_to_alg_deg(q: [f32; 4]) -> (f32, f32, f32) {
    let q = quat_normalize(q);
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r01 = 2.0 * (x * y - w * z);
    let r02 = 2.0 * (x * z + w * y);
    let r12 = 2.0 * (y * z - w * x);
    let r22 = 1.0 - 2.0 * (x * x + y * y);
    let pitch = r02.clamp(-1.0, 1.0).asin();
    let roll = (-r12).atan2(r22);
    let yaw = (-r01).atan2(r00);
    (rad2deg(roll), rad2deg(pitch), rad2deg(yaw).rem_euclid(360.0))
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

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn run_case(
    roll_true_deg: f32,
    pitch_true_deg: f32,
    yaw_true_deg: f32,
    accel_noise_std: f32,
    gyro_noise_std: f32,
    vel_noise_std: f32,
    seed: u64,
) -> (f32, f32, f32) {
    let mut f = Align::default();
    align_init(
        &mut f,
        [0.4, 0.4, 0.4],
        MisalignNoise {
            q_theta_rw_var: 1.0e-7,
            nhc_vy_var: 0.25,
            nhc_vz_var: 0.25,
        },
    );
    align_set_q_sb(&mut f, [1.0, 0.0, 0.0, 0.0]);

    let q_true = quat_from_alg_deg(roll_true_deg, pitch_true_deg, yaw_true_deg);
    let r_sb_true = quat_to_rotmat(q_true);
    let g_n = [0.0_f32, 0.0_f32, 9.80665_f32];

    let mut rng = StdRng::seed_from_u64(seed);
    let n_acc = Normal::new(0.0, accel_noise_std.max(1.0e-6) as f64).unwrap();
    let n_gyr = Normal::new(0.0, gyro_noise_std.max(1.0e-6) as f64).unwrap();
    let n_vel = Normal::new(0.0, vel_noise_std.max(1.0e-6) as f64).unwrap();

    let dt = 0.01_f32;
    let n_steps = 7000usize;
    let mut v_n = [0.0_f32, 0.0_f32, 0.0_f32];
    let mut q_nb_true = [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32];

    for k in 0..n_steps {
        let t = k as f32 * dt;
        let w_b = [
            deg2rad(8.0) * (0.19 * t).sin(),
            deg2rad(6.0) * (0.31 * t).cos(),
            deg2rad(11.0) * (0.23 * t).sin() + deg2rad(4.0) * (0.71 * t).cos(),
        ];
        let dq_nb = quat_normalize([1.0, 0.5 * w_b[0] * dt, 0.5 * w_b[1] * dt, 0.5 * w_b[2] * dt]);
        q_nb_true = quat_normalize(quat_mul(q_nb_true, dq_nb));
        let r_nb_true = quat_to_rotmat(q_nb_true);
        let r_bn_true = [
            [r_nb_true[0][0], r_nb_true[1][0], r_nb_true[2][0]],
            [r_nb_true[0][1], r_nb_true[1][1], r_nb_true[2][1]],
            [r_nb_true[0][2], r_nb_true[1][2], r_nb_true[2][2]],
        ];
        // Rich, smooth acceleration excitation in body frame.
        let a_b = [
            0.9 * (0.37 * t).sin() + 0.35 * (1.11 * t).sin(),
            0.6 * (0.29 * t).cos() + 0.25 * (0.83 * t).sin(),
            0.15 * (0.41 * t).sin(),
        ];
        let a_n = mat_vec(r_nb_true, a_b);

        v_n[0] += a_n[0] * dt;
        v_n[1] += a_n[1] * dt;
        v_n[2] += a_n[2] * dt;

        // Specific force in body: f_b = R_bn * (a_n - g_n).
        let f_b = mat_vec(
            r_bn_true,
            [a_n[0] - g_n[0], a_n[1] - g_n[1], a_n[2] - g_n[2]],
        );
        let mut f_s = mat_vec(r_sb_true, f_b);
        f_s[0] += n_acc.sample(&mut rng) as f32;
        f_s[1] += n_acc.sample(&mut rng) as f32;
        f_s[2] += n_acc.sample(&mut rng) as f32;
        let mut w_s = mat_vec(r_sb_true, w_b);
        w_s[0] += n_gyr.sample(&mut rng) as f32;
        w_s[1] += n_gyr.sample(&mut rng) as f32;
        w_s[2] += n_gyr.sample(&mut rng) as f32;

        let imu = MisalignImuSample {
            dt,
            w_sx: w_s[0],
            w_sy: w_s[1],
            w_sz: w_s[2],
            f_sx: f_s[0],
            f_sy: f_s[1],
            f_sz: f_s[2],
        };
        align_predict(&mut f, &imu);

        if k % 50 == 0 {
            let vel_meas = [
                v_n[0] + n_vel.sample(&mut rng) as f32,
                v_n[1] + n_vel.sample(&mut rng) as f32,
                v_n[2] + n_vel.sample(&mut rng) as f32,
            ];
            let r = vel_noise_std * vel_noise_std;
            align_fuse_velocity(&mut f, vel_meas, [r, r, r]);
        }
    }

    let q_est = align_q_sb(&f);
    let (r_est, p_est, y_est) = quat_to_alg_deg(q_est);
    let (r_true, p_true, y_true) = quat_to_alg_deg(q_true);
    (
        wrap180(r_est - r_true).abs(),
        wrap180(p_est - p_true).abs(),
        wrap180(y_est - y_true).abs(),
    )
}

#[test]
fn recovers_known_angles_single_case() {
    let (er, ep, ey) = run_case(7.0, -6.0, 9.0, 0.01, 0.001, 0.02, 7);
    assert!(er < 0.5, "roll error too high: {er}");
    assert!(ep < 0.5, "pitch error too high: {ep}");
    assert!(ey < 0.5, "yaw error too high: {ey}");
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 16,
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    #[test]
    fn converges_under_noisy_synthetic_inputs(
        roll in -10.0f32..10.0f32,
        pitch in -10.0f32..10.0f32,
        yaw in -15.0f32..15.0f32,
        seed in any::<u64>(),
    ) {
        let (er, ep, ey) = run_case(roll, pitch, yaw, 0.02, 0.002, 0.05, seed);
        prop_assert!(er < 0.5, "roll error: {er} for true=({roll},{pitch},{yaw})");
        prop_assert!(ep < 0.5, "pitch error: {ep} for true=({roll},{pitch},{yaw})");
        prop_assert!(ey < 0.5, "yaw error: {ey} for true=({roll},{pitch},{yaw})");
    }
}
