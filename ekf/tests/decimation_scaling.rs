use sensor_fusion::ekf::{
    Ekf, ImuSample, PredictNoise, ekf_fuse_body_vel, ekf_predict, ekf_set_predict_noise,
};

#[derive(Clone, Copy, Debug)]
enum ScalingMode {
    None,
    InvK,
    InvK2,
}

fn zero_imu(dt: f32) -> ImuSample {
    ImuSample {
        dax: 0.0,
        day: 0.0,
        daz: 0.0,
        dvx: 0.0,
        dvy: 0.0,
        dvz: 0.0,
        dt,
    }
}

fn tiny_constant_imu(dt: f32) -> ImuSample {
    ImuSample {
        dax: 2.0e-4 * dt,
        day: -1.0e-4 * dt,
        daz: 3.0e-4 * dt,
        dvx: 4.0e-3 * dt,
        dvy: -2.0e-3 * dt,
        dvz: 1.0e-3 * dt,
        dt,
    }
}

fn scaled_noise(base: PredictNoise, k: f32, mode: ScalingMode) -> PredictNoise {
    match mode {
        ScalingMode::None => base,
        ScalingMode::InvK => PredictNoise {
            gyro_var: base.gyro_var / k,
            accel_var: base.accel_var / k,
            gyro_bias_rw_var: base.gyro_bias_rw_var / k,
            accel_bias_rw_var: base.accel_bias_rw_var / k,
        },
        ScalingMode::InvK2 => PredictNoise {
            gyro_var: base.gyro_var / (k * k),
            accel_var: base.accel_var / (k * k),
            gyro_bias_rw_var: base.gyro_bias_rw_var / (k * k),
            accel_bias_rw_var: base.accel_bias_rw_var / (k * k),
        },
    }
}

fn run_full_rate_bias_cov(k: usize, dt: f32, noise: PredictNoise) -> Ekf {
    let mut ekf = Ekf::default();
    ekf_set_predict_noise(&mut ekf, noise);
    let imu = zero_imu(dt);
    for _ in 0..k {
        ekf_predict(&mut ekf, &imu, None);
    }
    ekf
}

fn run_decimated_bias_cov(k: usize, dt: f32, noise: PredictNoise, mode: ScalingMode) -> Ekf {
    let mut ekf = Ekf::default();
    let kf = k as f32;
    ekf_set_predict_noise(&mut ekf, scaled_noise(noise, kf, mode));
    let imu = zero_imu(dt * kf);
    ekf_predict(&mut ekf, &imu, None);
    ekf
}

fn bias_diag_abs_err(full: &Ekf, dec: &Ekf, idx: usize) -> f32 {
    (full.p[idx][idx] - dec.p[idx][idx]).abs()
}

fn run_full_rate_constant_input(k: usize, dt: f32, noise: PredictNoise) -> Ekf {
    let mut ekf = Ekf::default();
    ekf_set_predict_noise(&mut ekf, noise);
    let imu = tiny_constant_imu(dt);
    for _ in 0..k {
        ekf_predict(&mut ekf, &imu, None);
    }
    ekf
}

fn run_decimated_constant_input(k: usize, dt: f32, noise: PredictNoise, mode: ScalingMode) -> Ekf {
    let mut ekf = Ekf::default();
    let kf = k as f32;
    ekf_set_predict_noise(&mut ekf, scaled_noise(noise, kf, mode));
    let base = tiny_constant_imu(dt);
    let imu = ImuSample {
        dax: base.dax * kf,
        day: base.day * kf,
        daz: base.daz * kf,
        dvx: base.dvx * kf,
        dvy: base.dvy * kf,
        dvz: base.dvz * kf,
        dt: dt * kf,
    };
    ekf_predict(&mut ekf, &imu, None);
    ekf
}

fn ekf_state_max_abs_err(full: &Ekf, dec: &Ekf) -> f32 {
    [
        full.state.q0 - dec.state.q0,
        full.state.q1 - dec.state.q1,
        full.state.q2 - dec.state.q2,
        full.state.q3 - dec.state.q3,
        full.state.vn - dec.state.vn,
        full.state.ve - dec.state.ve,
        full.state.vd - dec.state.vd,
        full.state.pn - dec.state.pn,
        full.state.pe - dec.state.pe,
        full.state.pd - dec.state.pd,
    ]
    .into_iter()
    .map(f32::abs)
    .fold(0.0, f32::max)
}

fn init_body_vel_case() -> Ekf {
    let mut ekf = Ekf::default();
    ekf.state.q0 = 1.0;
    ekf.state.q1 = 0.0;
    ekf.state.q2 = 0.0;
    ekf.state.q3 = 0.0;
    ekf.state.vn = 12.0;
    ekf.state.ve = 0.8;
    ekf.state.vd = -0.25;
    ekf.state.pn = 0.0;
    ekf.state.pe = 0.0;
    ekf.state.pd = 0.0;
    ekf
}

fn run_full_rate_body_vel(k: usize, r_body_vel: f32) -> Ekf {
    let mut ekf = init_body_vel_case();
    for _ in 0..k {
        ekf_fuse_body_vel(&mut ekf, r_body_vel);
    }
    ekf
}

fn run_decimated_body_vel(k: usize, r_body_vel: f32, mode: ScalingMode) -> Ekf {
    let mut ekf = init_body_vel_case();
    let kf = k as f32;
    let r_scaled = match mode {
        ScalingMode::None => r_body_vel,
        ScalingMode::InvK => r_body_vel / kf,
        ScalingMode::InvK2 => r_body_vel / (kf * kf),
    };
    ekf_fuse_body_vel(&mut ekf, r_scaled);
    ekf
}

fn body_vel_cov_rms_err(full: &Ekf, dec: &Ekf) -> f32 {
    let mut sum = 0.0_f64;
    for i in 0..16 {
        let d = (full.p[i][i] - dec.p[i][i]) as f64;
        sum += d * d;
    }
    (sum / 16.0).sqrt() as f32
}

#[test]
fn inv_k_matches_gyro_bias_covariance_accumulation() {
    let k = 3usize;
    let dt = 0.01_f32;
    let noise = PredictNoise {
        gyro_var: 1.0e-4,
        accel_var: 12.0,
        gyro_bias_rw_var: 1.0e-3,
        accel_bias_rw_var: 2.0e-3,
    };

    let full = run_full_rate_bias_cov(k, dt, noise);
    let dec_none = run_decimated_bias_cov(k, dt, noise, ScalingMode::None);
    let dec_inv_k = run_decimated_bias_cov(k, dt, noise, ScalingMode::InvK);
    let dec_inv_k2 = run_decimated_bias_cov(k, dt, noise, ScalingMode::InvK2);

    for idx in [10usize, 11, 12] {
        let err_none = bias_diag_abs_err(&full, &dec_none, idx);
        let err_inv_k = bias_diag_abs_err(&full, &dec_inv_k, idx);
        let err_inv_k2 = bias_diag_abs_err(&full, &dec_inv_k2, idx);
        assert!(
            err_inv_k < 1.0e-12,
            "gyro bias P[{idx}][{idx}] should match under 1/k scaling, got {err_inv_k}"
        );
        assert!(
            err_inv_k < err_none,
            "gyro bias P[{idx}][{idx}] expected 1/k to beat no scaling: inv_k={err_inv_k} none={err_none}"
        );
        assert!(
            err_inv_k < err_inv_k2,
            "gyro bias P[{idx}][{idx}] expected 1/k to beat 1/k^2: inv_k={err_inv_k} inv_k2={err_inv_k2}"
        );
    }
}

#[test]
fn inv_k_matches_accel_bias_covariance_accumulation() {
    let k = 3usize;
    let dt = 0.01_f32;
    let noise = PredictNoise {
        gyro_var: 1.0e-4,
        accel_var: 12.0,
        gyro_bias_rw_var: 1.0e-3,
        accel_bias_rw_var: 2.0e-3,
    };

    let full = run_full_rate_bias_cov(k, dt, noise);
    let dec_none = run_decimated_bias_cov(k, dt, noise, ScalingMode::None);
    let dec_inv_k = run_decimated_bias_cov(k, dt, noise, ScalingMode::InvK);
    let dec_inv_k2 = run_decimated_bias_cov(k, dt, noise, ScalingMode::InvK2);

    for idx in [13usize, 14, 15] {
        let err_none = bias_diag_abs_err(&full, &dec_none, idx);
        let err_inv_k = bias_diag_abs_err(&full, &dec_inv_k, idx);
        let err_inv_k2 = bias_diag_abs_err(&full, &dec_inv_k2, idx);
        assert!(
            err_inv_k < 1.0e-10,
            "accel bias P[{idx}][{idx}] should match under 1/k scaling, got {err_inv_k}"
        );
        assert!(
            err_inv_k < err_none,
            "accel bias P[{idx}][{idx}] expected 1/k to beat no scaling: inv_k={err_inv_k} none={err_none}"
        );
        assert!(
            err_inv_k < err_inv_k2,
            "accel bias P[{idx}][{idx}] expected 1/k to beat 1/k^2: inv_k={err_inv_k} inv_k2={err_inv_k2}"
        );
    }
}

#[test]
fn decimated_input_uses_aggregated_small_imu_deltas_with_small_state_error() {
    let k = 3usize;
    let dt = 0.01_f32;
    let noise = PredictNoise {
        gyro_var: 0.0,
        accel_var: 0.0,
        gyro_bias_rw_var: 0.0,
        accel_bias_rw_var: 0.0,
    };

    let full = run_full_rate_constant_input(k, dt, noise);
    let dec = run_decimated_constant_input(k, dt, noise, ScalingMode::InvK);
    let err = ekf_state_max_abs_err(&full, &dec);

    assert!(
        err < 5.0e-3,
        "aggregated decimated IMU input should stay close to repeated tiny inputs, got max state err {err}"
    );
}

#[test]
fn body_vel_scaling_is_only_heuristic_in_static_case() {
    let k = 3usize;
    let r_body_vel = 5.0_f32;

    let full = run_full_rate_body_vel(k, r_body_vel);
    let dec_none = run_decimated_body_vel(k, r_body_vel, ScalingMode::None);
    let dec_inv_k = run_decimated_body_vel(k, r_body_vel, ScalingMode::InvK);
    let dec_inv_k2 = run_decimated_body_vel(k, r_body_vel, ScalingMode::InvK2);

    let state_none = ekf_state_max_abs_err(&full, &dec_none);
    let state_inv_k = ekf_state_max_abs_err(&full, &dec_inv_k);
    let state_inv_k2 = ekf_state_max_abs_err(&full, &dec_inv_k2);
    let cov_none = body_vel_cov_rms_err(&full, &dec_none);
    let cov_inv_k = body_vel_cov_rms_err(&full, &dec_inv_k);
    let cov_inv_k2 = body_vel_cov_rms_err(&full, &dec_inv_k2);

    assert!(
        state_none > 0.0 && state_inv_k > 0.0 && state_inv_k2 > 0.0,
        "body_vel decimation variants should all differ from repeated scalar updates: none={state_none} inv_k={state_inv_k} inv_k2={state_inv_k2}"
    );
    assert!(
        cov_none > 0.0 && cov_inv_k > 0.0 && cov_inv_k2 > 0.0,
        "body_vel covariance should also differ under decimation heuristics: none={cov_none} inv_k={cov_inv_k} inv_k2={cov_inv_k2}"
    );
    assert!(
        (state_none - state_inv_k).abs() > 1.0e-8 || (state_inv_k - state_inv_k2).abs() > 1.0e-8,
        "body_vel state errors unexpectedly collapsed to the same value"
    );
    assert!(
        (cov_none - cov_inv_k).abs() > 1.0e-8 || (cov_inv_k - cov_inv_k2).abs() > 1.0e-8,
        "body_vel covariance errors unexpectedly collapsed to the same value"
    );
}
