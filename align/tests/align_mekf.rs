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

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

fn quat_to_rotmat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (w, x, y, z) = if n > 1.0e-8 {
        (q[0] / n, q[1] / n, q[2] / n, q[3] / n)
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };
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
fn lateral_update_preserves_tilt_and_only_corrects_heading() {
    let truth = [25.0_f32, -20.0_f32, 120.0_f32];
    let (stationary_accel, windows) = simulate_windows(truth, 4);
    assert!(stationary_accel.len() >= 100);

    let mut cfg = AlignConfig::default();
    cfg.use_lateral_accel = true;
    cfg.use_longitudinal_accel = false;
    let mut filter = Align::new(cfg);
    filter
        .initialize_from_stationary(&stationary_accel, 0.0)
        .expect("stationary initialization");

    let mut prev_q = filter.q_vb;
    let mut found = false;
    for window in windows.iter().skip(1) {
        let (_, trace) = filter.update_window_with_trace(window);
        if trace.after_lateral_accel.is_some() {
            let before_rot = quat_to_rotmat(prev_q);
            let after_rot = quat_to_rotmat(filter.q_vb);
            let before_down = [before_rot[0][2], before_rot[1][2], before_rot[2][2]];
            let after_down = [after_rot[0][2], after_rot[1][2], after_rot[2][2]];
            let before_fwd = [before_rot[0][0], before_rot[1][0], before_rot[2][0]];
            let after_fwd = [after_rot[0][0], after_rot[1][0], after_rot[2][0]];
            let down_delta = ((before_down[0] - after_down[0]).powi(2)
                + (before_down[1] - after_down[1]).powi(2)
                + (before_down[2] - after_down[2]).powi(2))
            .sqrt();
            let fwd_delta = ((before_fwd[0] - after_fwd[0]).powi(2)
                + (before_fwd[1] - after_fwd[1]).powi(2)
                + (before_fwd[2] - after_fwd[2]).powi(2))
            .sqrt();
            assert!(
                down_delta < 1.0e-3,
                "down axis changed too much: before={before_down:?} after={after_down:?}"
            );
            assert!(
                fwd_delta > 1.0e-3,
                "forward axis did not change: before={before_fwd:?} after={after_fwd:?}"
            );
            found = true;
            break;
        }
        prev_q = filter.q_vb;
    }
    assert!(found, "no lateral update occurred in synthetic replay");
}

#[test]
fn stationary_yaw_seed_preserves_gravity_down_axis() {
    let truth_deg = [18.0_f32, -27.0_f32, 115.0_f32];
    let truth_rad = [
        truth_deg[0].to_radians(),
        truth_deg[1].to_radians(),
        truth_deg[2].to_radians(),
    ];
    let c_v_b = euler_zyx_to_rot(truth_rad[0], truth_rad[1], truth_rad[2]);
    let accel_b = mat_vec(c_v_b, [0.0, 0.0, -9.80665]);
    let stationary = vec![accel_b; 300];

    let mut f0 = Align::new(AlignConfig::default());
    f0.initialize_from_stationary(&stationary, 0.0).unwrap();
    let c0 = quat_to_rotmat(f0.q_vb);
    let down0_b = [c0[0][2], c0[1][2], c0[2][2]];

    let mut f1 = Align::new(AlignConfig::default());
    f1.initialize_from_stationary(&stationary, 90.0_f32.to_radians())
        .unwrap();
    let c1 = quat_to_rotmat(f1.q_vb);
    let down1_b = [c1[0][2], c1[1][2], c1[2][2]];

    let expected_down_b = normalize3([-accel_b[0], -accel_b[1], -accel_b[2]]);
    for (name, got) in [("seed0", down0_b), ("seed90", down1_b)] {
        let got = normalize3(got);
        let err = ((got[0] - expected_down_b[0]).powi(2)
            + (got[1] - expected_down_b[1]).powi(2)
            + (got[2] - expected_down_b[2]).powi(2))
        .sqrt();
        assert!(err < 1.0e-3, "{name} down-axis mismatch: {got:?} vs {expected_down_b:?}");
    }
}
