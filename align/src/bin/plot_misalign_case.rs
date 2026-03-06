use std::fs::File;
use std::io::Write;

use align_rs::align::{MisalignImuSample, MisalignNoise, Align, align_fuse_velocity, align_init, align_predict_gyro, align_q_sb, align_set_q_sb};

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

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_from_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let h = 0.5 * angle;
    let s = h.sin();
    quat_normalize([h.cos(), axis[0] * s, axis[1] * s, axis[2] * s])
}

fn quat_from_rpy_zyx(roll: f32, pitch: f32, yaw: f32) -> [f32; 4] {
    let (sr, cr) = (0.5 * roll).sin_cos();
    let (sp, cp) = (0.5 * pitch).sin_cos();
    let (sy, cy) = (0.5 * yaw).sin_cos();
    quat_normalize([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
}

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

#[derive(Clone, Copy)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }
    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (self.state >> 32) as u32;
        (x as f32) / (u32::MAX as f32)
    }
    fn noise(&mut self, std: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * std
    }
}

fn run_case(
    path: &str,
    roll_true_deg: f32,
    pitch_true_deg: f32,
    yaw_true_deg: f32,
    seed: u64,
    accel_noise_std: f32,
    gyro_noise_std: f32,
    vel_noise_std: f32,
) -> std::io::Result<()> {
    let mut f = Align::default();
    align_init(
        &mut f,
        [0.4, 0.4, 0.4],
        MisalignNoise {
            q_theta_rw_var: 1.0e-7,
        },
    );
    align_set_q_sb(&mut f, [1.0, 0.0, 0.0, 0.0]);

    let q_true = quat_from_alg_deg(roll_true_deg, pitch_true_deg, yaw_true_deg);
    let r_sb_true = quat_to_rotmat(q_true);
    let g_n = [0.0_f32, 0.0_f32, 9.80665_f32];

    let mut rng = Lcg::new(seed);
    let dt = 0.01_f32;
    let n_steps = 7000usize;
    let mut v_n = [0.0_f32, 0.0_f32, 0.0_f32];
    let mut v_n_prev = v_n;
    let mut speed = 0.0_f32;
    let mut yaw = 0.0_f32;
    let mut pitch = 0.0_f32;
    let mut roll = 0.0_f32;
    let mut prev_roll = 0.0_f32;
    let mut prev_pitch = 0.0_f32;
    let tau_att = 0.35_f32;

    let mut out = File::create(path)?;
    writeln!(
        out,
        "t_s,true_roll_deg,true_pitch_deg,true_yaw_deg,est_roll_deg,est_pitch_deg,est_yaw_deg,err_roll_deg,err_pitch_deg,err_yaw_deg,w_sx,w_sy,w_sz,f_sx,f_sy,f_sz,gnss_vn,gnss_ve,gnss_vd"
    )?;

    let (r_true, p_true, y_true) = quat_to_alg_deg(q_true);
    for k in 0..n_steps {
        let t = k as f32 * dt;
        let a_long_cmd = if t < 10.0 {
            0.9
        } else if t < 22.0 {
            0.0
        } else if t < 34.0 {
            -0.6
        } else if t < 45.0 {
            0.4
        } else if t < 58.0 {
            0.0
        } else if t < 70.0 {
            0.7
        } else {
            -0.5
        };
        let yaw_rate = if t < 12.0 {
            0.0
        } else if t < 28.0 {
            deg2rad(7.5) * (0.35 * t).sin()
        } else if t < 42.0 {
            deg2rad(-11.0)
        } else if t < 58.0 {
            deg2rad(5.5) * (0.45 * t).cos()
        } else {
            deg2rad(9.0) * (0.25 * t).sin()
        };
        speed = (speed + a_long_cmd * dt).clamp(0.0, 22.0);
        yaw += yaw_rate * dt;
        let a_lat = speed * yaw_rate;
        let roll_ref = (a_lat / 9.80665).atan();
        let pitch_ref = (-a_long_cmd / 9.80665).atan();
        roll += (roll_ref - roll) * (dt / tau_att);
        pitch += (pitch_ref - pitch) * (dt / tau_att);
        let roll_dot = (roll - prev_roll) / dt;
        let pitch_dot = (pitch - prev_pitch) / dt;
        prev_roll = roll;
        prev_pitch = pitch;
        let w_b = [
            roll_dot - yaw_rate * pitch.sin(),
            pitch_dot * roll.cos() + yaw_rate * roll.sin() * pitch.cos(),
            -pitch_dot * roll.sin() + yaw_rate * roll.cos() * pitch.cos(),
        ];

        let q_nb_true = quat_from_rpy_zyx(roll, pitch, yaw);
        let r_nb_true = quat_to_rotmat(q_nb_true);
        let r_bn_true = [
            [r_nb_true[0][0], r_nb_true[1][0], r_nb_true[2][0]],
            [r_nb_true[0][1], r_nb_true[1][1], r_nb_true[2][1]],
            [r_nb_true[0][2], r_nb_true[1][2], r_nb_true[2][2]],
        ];
        v_n = mat_vec(r_nb_true, [speed, 0.0, 0.0]);
        let a_n = [
            (v_n[0] - v_n_prev[0]) / dt,
            (v_n[1] - v_n_prev[1]) / dt,
            (v_n[2] - v_n_prev[2]) / dt,
        ];
        v_n_prev = v_n;

        let f_b = mat_vec(
            r_bn_true,
            [a_n[0] - g_n[0], a_n[1] - g_n[1], a_n[2] - g_n[2]],
        );
        let mut f_s = mat_vec(r_sb_true, f_b);
        f_s[0] += rng.noise(accel_noise_std);
        f_s[1] += rng.noise(accel_noise_std);
        f_s[2] += rng.noise(accel_noise_std);
        let mut w_s = mat_vec(r_sb_true, w_b);
        w_s[0] += rng.noise(gyro_noise_std);
        w_s[1] += rng.noise(gyro_noise_std);
        w_s[2] += rng.noise(gyro_noise_std);

        let imu = MisalignImuSample {
            dt,
            f_sx: f_s[0],
            f_sy: f_s[1],
            f_sz: f_s[2],
        };
        align_predict_gyro(&mut f, &imu, w_s[0], w_s[1], w_s[2]);
        let mut gnss_meas = [f32::NAN, f32::NAN, f32::NAN];
        if k % 50 == 0 {
            let r = vel_noise_std * vel_noise_std;
            gnss_meas = [
                v_n[0] + rng.noise(vel_noise_std),
                v_n[1] + rng.noise(vel_noise_std),
                v_n[2] + rng.noise(vel_noise_std),
            ];
            align_fuse_velocity(
                &mut f,
                gnss_meas,
                [r, r, r],
            );
        }

        let q_est = align_q_sb(&f);
        let (r_est, p_est, y_est) = quat_to_alg_deg(q_est);
        writeln!(
            out,
            "{t:.4},{r_true:.6},{p_true:.6},{y_true:.6},{r_est:.6},{p_est:.6},{y_est:.6},{:.6},{:.6},{:.6},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8}",
            wrap180(r_est - r_true),
            wrap180(p_est - p_true),
            wrap180(y_est - y_true),
            w_s[0],
            w_s[1],
            w_s[2],
            f_s[0],
            f_s[1],
            f_s[2],
            gnss_meas[0],
            gnss_meas[1],
            gnss_meas[2]
        )?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    run_case(
        "case_known_tiny_noise.csv",
        7.0,
        -6.0,
        9.0,
        7,
        1.0e-5,
        1.0e-6,
        1.0e-4,
    )?;
    run_case(
        "case_prop_fail_like.csv",
        -1.3056068,
        0.0,
        3.7888157,
        24110580570772,
        0.02,
        0.002,
        0.05,
    )?;
    eprintln!("wrote case_known_tiny_noise.csv and case_prop_fail_like.csv");
    Ok(())
}
