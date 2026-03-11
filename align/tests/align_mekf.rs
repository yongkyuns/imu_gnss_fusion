use align_rs::{Align, AlignConfig, AlignWindowSummary};

#[derive(Clone, Copy)]
struct Segment {
    duration_s: f32,
    a_long_mps2: f32,
    yaw_rate_radps: f32,
}

fn wrap_pi(x: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    (x + std::f32::consts::PI).rem_euclid(two_pi) - std::f32::consts::PI
}

fn euler_zyx_to_rot(roll: f32, pitch: f32, yaw: f32) -> [[f32; 3]; 3] {
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

fn mat_vec(a: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn simulate_windows(
    truth_deg: [f32; 3],
    repeat_count: usize,
) -> (Vec<[f32; 3]>, Vec<AlignWindowSummary>) {
    let truth_rad = [
        truth_deg[0].to_radians(),
        truth_deg[1].to_radians(),
        truth_deg[2].to_radians(),
    ];
    let c_v_b = euler_zyx_to_rot(truth_rad[0], truth_rad[1], truth_rad[2]);
    let segments = {
        let mut v = vec![Segment {
            duration_s: 6.0,
            a_long_mps2: 0.0,
            yaw_rate_radps: 0.0,
        }];
        let base = [
            Segment {
                duration_s: 8.0,
                a_long_mps2: 0.9,
                yaw_rate_radps: 0.0,
            },
            Segment {
                duration_s: 4.0,
                a_long_mps2: 0.0,
                yaw_rate_radps: 0.0,
            },
            Segment {
                duration_s: 8.0,
                a_long_mps2: 0.0,
                yaw_rate_radps: 10.0_f32.to_radians(),
            },
            Segment {
                duration_s: 3.0,
                a_long_mps2: -0.7,
                yaw_rate_radps: 0.0,
            },
            Segment {
                duration_s: 8.0,
                a_long_mps2: 0.0,
                yaw_rate_radps: -12.0_f32.to_radians(),
            },
            Segment {
                duration_s: 4.0,
                a_long_mps2: 0.8,
                yaw_rate_radps: 0.0,
            },
            Segment {
                duration_s: 8.0,
                a_long_mps2: 0.0,
                yaw_rate_radps: 9.0_f32.to_radians(),
            },
            Segment {
                duration_s: 5.0,
                a_long_mps2: -0.6,
                yaw_rate_radps: 0.0,
            },
        ];
        for _ in 0..repeat_count {
            v.extend(base);
        }
        v
    };

    let dt_imu = 0.01_f32;
    let steps_per_gnss = 10usize;
    let mut speed = 0.0_f32;
    let mut yaw = 0.0_f32;
    let mut stationary_accel = Vec::new();
    let mut windows = Vec::new();
    let mut accel_window = Vec::<[f32; 3]>::new();
    let mut gyro_window = Vec::<[f32; 3]>::new();
    let mut gnss_prev_v = [0.0_f32; 3];

    for seg in segments {
        let n_steps = (seg.duration_s / dt_imu).round() as usize;
        for _ in 0..n_steps {
            speed = (speed + seg.a_long_mps2 * dt_imu).max(0.0);
            yaw = wrap_pi(yaw + seg.yaw_rate_radps * dt_imu);

            let a_long = seg.a_long_mps2;
            let a_lat = speed * seg.yaw_rate_radps;
            let v_n = [speed * yaw.cos(), speed * yaw.sin(), 0.0];
            let f_v = [a_long, a_lat, -9.80665];
            let omega_v = [0.0, 0.0, seg.yaw_rate_radps];
            let accel_b = mat_vec(c_v_b, f_v);
            let gyro_b = mat_vec(c_v_b, omega_v);
            accel_window.push(accel_b);
            gyro_window.push(gyro_b);

            if speed < 0.1 && seg.yaw_rate_radps.abs() < 1.0e-4 && a_long.abs() < 1.0e-4 {
                stationary_accel.push(accel_b);
            }

            if accel_window.len() == steps_per_gnss {
                let mut mean_accel = [0.0_f32; 3];
                let mut mean_gyro = [0.0_f32; 3];
                for sample in &accel_window {
                    mean_accel[0] += sample[0];
                    mean_accel[1] += sample[1];
                    mean_accel[2] += sample[2];
                }
                for sample in &gyro_window {
                    mean_gyro[0] += sample[0];
                    mean_gyro[1] += sample[1];
                    mean_gyro[2] += sample[2];
                }
                let inv_n = 1.0 / (steps_per_gnss as f32);
                mean_accel = [
                    mean_accel[0] * inv_n,
                    mean_accel[1] * inv_n,
                    mean_accel[2] * inv_n,
                ];
                mean_gyro = [
                    mean_gyro[0] * inv_n,
                    mean_gyro[1] * inv_n,
                    mean_gyro[2] * inv_n,
                ];
                windows.push(AlignWindowSummary {
                    dt: steps_per_gnss as f32 * dt_imu,
                    mean_gyro_b: mean_gyro,
                    mean_accel_b: mean_accel,
                    gnss_vel_prev_n: gnss_prev_v,
                    gnss_vel_curr_n: v_n,
                });
                gnss_prev_v = v_n;
                accel_window.clear();
                gyro_window.clear();
            }
        }
    }

    (stationary_accel, windows)
}

fn wrap_deg180(mut v: f32) -> f32 {
    while v > 180.0 {
        v -= 360.0;
    }
    while v <= -180.0 {
        v += 360.0;
    }
    v
}

#[test]
fn align_converges_on_synthetic_case() {
    let truth = [25.0_f32, -20.0_f32, 120.0_f32];
    let (stationary_accel, windows) = simulate_windows(truth, 4);
    assert!(stationary_accel.len() >= 100);

    let mut cfg = AlignConfig::default();
    cfg.use_turn_gyro = true;
    cfg.use_course_rate = true;
    cfg.use_lateral_accel = true;
    cfg.use_longitudinal_accel = false;
    let mut filter = Align::new(cfg);
    filter
        .initialize_from_stationary(&stationary_accel, 0.0)
        .expect("stationary initialization");

    for window in windows.iter().skip(1) {
        filter.update_window(window);
    }

    let est = filter.mount_angles_deg();
    let err = [
        est[0] - truth[0],
        est[1] - truth[1],
        wrap_deg180(est[2] - truth[2]),
    ];
    let sigma = filter.sigma_deg();
    assert!(err[0].abs() < 3.0, "roll err {:?}", err);
    assert!(err[1].abs() < 4.0, "pitch err {:?}", err);
    assert!(err[2].abs() < 8.0, "yaw err {:?}", err);
    assert!(
        sigma[0] < 1.5 && sigma[1] < 1.5 && sigma[2] < 1.5,
        "sigma {:?}",
        sigma
    );
}
