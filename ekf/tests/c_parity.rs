use sensor_fusion::align::{Align, AlignConfig, AlignWindowSummary};

fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
    let d = (a - b).abs();
    assert!(d <= tol, "{ctx}: |{a} - {b}| = {d} > {tol}");
}

#[test]
fn align_stationary_bootstrap_and_update_are_stable() {
    let cfg = AlignConfig::default();
    let mut align = Align::new(cfg);
    let accel_samples = vec![[0.0_f32, 0.0_f32, -9.80665_f32]; 16];

    align
        .initialize_from_stationary(&accel_samples, 0.0)
        .expect("stationary init should succeed");

    assert_close(align.q_vb[0], 1.0, 1.0e-6, "bootstrap q0");
    assert_close(align.q_vb[1], 0.0, 1.0e-6, "bootstrap q1");
    assert_close(align.q_vb[2], 0.0, 1.0e-6, "bootstrap q2");
    assert_close(align.q_vb[3], 0.0, 1.0e-6, "bootstrap q3");

    let window = AlignWindowSummary {
        dt: 0.1,
        mean_gyro_b: [0.0, 0.0, 0.0],
        mean_accel_b: [1.2, 0.0, -9.80665],
        gnss_vel_prev_n: [5.0, 0.0, 0.0],
        gnss_vel_curr_n: [5.12, 0.0, 0.0],
    };
    let yaw_var_before = align.P[2][2];
    let (_score, trace) = align.update_window_with_trace(&window);

    assert!(trace.after_horiz_accel.is_some());
    assert!(align.P[2][2] < yaw_var_before);
}
