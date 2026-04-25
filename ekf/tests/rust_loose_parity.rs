#![cfg(feature = "c-reference")]

use sensor_fusion::c_api::{CLooseImuDelta, CLooseNominalState, CLooseWrapper};
use sensor_fusion::loose::{LooseFilter, LooseImuDelta, LooseNominalState, LoosePredictNoise};

const EPS: f32 = 1.0e-6;
const COV_EPS: f32 = 1.0e-6;

fn assert_close(label: &str, lhs: f32, rhs: f32, eps: f32) {
    let diff = (lhs - rhs).abs();
    assert!(
        diff <= eps,
        "{label}: |{lhs:.9} - {rhs:.9}| = {diff:.9} > {eps:.9}"
    );
}

fn assert_nominal_close(c: &CLooseNominalState, r: &LooseNominalState) {
    assert_close("q0", c.q0, r.q0, EPS);
    assert_close("q1", c.q1, r.q1, EPS);
    assert_close("q2", c.q2, r.q2, EPS);
    assert_close("q3", c.q3, r.q3, EPS);
    assert_close("vn", c.vn, r.vn, EPS);
    assert_close("ve", c.ve, r.ve, EPS);
    assert_close("vd", c.vd, r.vd, EPS);
    assert_close("pn", c.pn, r.pn, EPS);
    assert_close("pe", c.pe, r.pe, EPS);
    assert_close("pd", c.pd, r.pd, EPS);
    assert_close("bgx", c.bgx, r.bgx, EPS);
    assert_close("bgy", c.bgy, r.bgy, EPS);
    assert_close("bgz", c.bgz, r.bgz, EPS);
    assert_close("bax", c.bax, r.bax, EPS);
    assert_close("bay", c.bay, r.bay, EPS);
    assert_close("baz", c.baz, r.baz, EPS);
    assert_close("sgx", c.sgx, r.sgx, EPS);
    assert_close("sgy", c.sgy, r.sgy, EPS);
    assert_close("sgz", c.sgz, r.sgz, EPS);
    assert_close("sax", c.sax, r.sax, EPS);
    assert_close("say", c.say, r.say, EPS);
    assert_close("saz", c.saz, r.saz, EPS);
    assert_close("qcs0", c.qcs0, r.qcs0, EPS);
    assert_close("qcs1", c.qcs1, r.qcs1, EPS);
    assert_close("qcs2", c.qcs2, r.qcs2, EPS);
    assert_close("qcs3", c.qcs3, r.qcs3, EPS);
}

fn assert_covariance_close(c: &CLooseWrapper, r: &LooseFilter) {
    for i in 0..24 {
        for j in 0..24 {
            assert_close(
                &format!("p[{i}][{j}]"),
                c.covariance()[i][j],
                r.covariance()[i][j],
                COV_EPS,
            );
        }
    }
}

fn assert_filter_close(c: &CLooseWrapper, r: &LooseFilter) {
    assert_nominal_close(c.nominal(), r.nominal());
    assert_covariance_close(c, r);
    for i in 0..24 {
        assert_close(
            &format!("last_dx[{i}]"),
            c.last_dx()[i],
            r.last_dx()[i],
            EPS,
        );
    }
    assert_eq!(c.last_obs_types(), r.last_obs_types());
    let c_shadow = c.shadow_pos_ecef();
    let r_shadow = r.shadow_pos_ecef();
    for i in 0..3 {
        assert!(
            (c_shadow[i] - r_shadow[i]).abs() <= 1.0e-8,
            "shadow_pos[{i}]: |{} - {}| > 1e-8",
            c_shadow[i],
            r_shadow[i]
        );
    }
}

fn seed_filters(noise: LoosePredictNoise) -> (CLooseWrapper, LooseFilter) {
    let mut c = CLooseWrapper::new(noise);
    let mut r = LooseFilter::new(noise);
    let mut p_diag = [0.0; 24];
    for (i, v) in p_diag.iter_mut().enumerate() {
        *v = 0.01 + 0.002 * i as f32;
    }
    let q_es = [0.999_68, 0.010_3, -0.018_2, 0.011_7];
    let pos_ecef = [4_518_283.125_f64, 767_934.5_f64, 4_424_732.25_f64];
    let vel_ecef = [3.2, -1.4, 0.28];
    let gyro_bias = [0.0012, -0.0007, 0.0004];
    let accel_bias = [0.03, -0.018, 0.012];
    let gyro_scale = [1.0002, 0.9997, 1.0004];
    let accel_scale = [0.999, 1.001, 1.0005];
    let q_cs = [0.998_7, 0.014, -0.021, 0.043];
    c.init_from_reference_ecef_state(
        q_es,
        pos_ecef,
        vel_ecef,
        gyro_bias,
        accel_bias,
        gyro_scale,
        accel_scale,
        q_cs,
        Some(p_diag),
    );
    r.init_from_reference_ecef_state(
        q_es,
        pos_ecef,
        vel_ecef,
        gyro_bias,
        accel_bias,
        gyro_scale,
        accel_scale,
        q_cs,
        Some(p_diag),
    );
    (c, r)
}

fn imu(dax: f32, day: f32, daz: f32, dvx: f32, dvy: f32, dvz: f32, dt: f32) -> LooseImuDelta {
    LooseImuDelta {
        dax_1: 0.95 * dax,
        day_1: 0.95 * day,
        daz_1: 0.95 * daz,
        dvx_1: 0.95 * dvx,
        dvy_1: 0.95 * dvy,
        dvz_1: 0.95 * dvz,
        dax_2: dax,
        day_2: day,
        daz_2: daz,
        dvx_2: dvx,
        dvy_2: dvy,
        dvz_2: dvz,
        dt,
    }
}

fn c_imu(imu: LooseImuDelta) -> CLooseImuDelta {
    CLooseImuDelta {
        dax_1: imu.dax_1,
        day_1: imu.day_1,
        daz_1: imu.daz_1,
        dvx_1: imu.dvx_1,
        dvy_1: imu.dvy_1,
        dvz_1: imu.dvz_1,
        dax_2: imu.dax_2,
        day_2: imu.day_2,
        daz_2: imu.daz_2,
        dvx_2: imu.dvx_2,
        dvy_2: imu.dvy_2,
        dvz_2: imu.dvz_2,
        dt: imu.dt,
    }
}

#[test]
fn rust_loose_init_matches_c() {
    let noise = LoosePredictNoise::reference_nsr_demo();
    let c = CLooseWrapper::new(noise);
    let r = LooseFilter::new(noise);
    assert_filter_close(&c, &r);
}

#[test]
fn rust_loose_predict_matches_c() {
    let noise = LoosePredictNoise::reference_nsr_demo();
    let (mut c, mut r) = seed_filters(noise);
    for k in 0..20 {
        let step = imu(
            0.0001 + k as f32 * 1.0e-6,
            -0.0002,
            0.00015,
            0.005,
            -0.003 + k as f32 * 2.0e-5,
            -0.1962,
            0.02,
        );
        c.predict(c_imu(step));
        r.predict(step);
    }
    assert_filter_close(&c, &r);
}

#[test]
fn rust_loose_gps_position_update_match_c() {
    let noise = LoosePredictNoise::reference_nsr_demo();
    let (mut c, mut r) = seed_filters(noise);
    let pos = [4_518_283.42_f64, 767_934.21_f64, 4_424_732.39_f64];
    c.fuse_gps_reference(pos, None, 1.8, 0.25, 0.2);
    r.fuse_gps_reference(pos, None, 1.8, 0.25, 0.2);
    assert_filter_close(&c, &r);
}

#[test]
fn rust_loose_nhc_update_matches_c() {
    let noise = LoosePredictNoise::reference_nsr_demo();
    let (mut c, mut r) = seed_filters(noise);
    c.fuse_nhc_reference([0.001, -0.0005, 0.0003], [0.02, -0.01, 9.80], 0.02);
    r.fuse_nhc_reference([0.001, -0.0005, 0.0003], [0.02, -0.01, 9.80], 0.02);
    assert_filter_close(&c, &r);
}

#[test]
fn rust_loose_combined_batch_matches_c() {
    let noise = LoosePredictNoise::reference_nsr_demo();
    let (mut c, mut r) = seed_filters(noise);
    let pos = [4_518_283.38_f64, 767_934.32_f64, 4_424_732.16_f64];
    let vel = [3.1, -1.0, 0.09];
    let vel_std = [0.08, 0.09, 0.12];
    c.fuse_reference_batch_full(
        Some(pos),
        Some(vel),
        1.6,
        Some(vel_std),
        0.1,
        [0.001, -0.0005, 0.0003],
        [0.02, -0.01, 9.80],
        0.02,
    );
    r.fuse_reference_batch_full(
        Some(pos),
        Some(vel),
        1.6,
        Some(vel_std),
        0.1,
        [0.001, -0.0005, 0.0003],
        [0.02, -0.01, 9.80],
        0.02,
    );
    assert_filter_close(&c, &r);
}

#[test]
fn rust_loose_error_transition_matches_c() {
    let noise = LoosePredictNoise::reference_nsr_demo();
    let (c, r) = seed_filters(noise);
    let step = imu(0.00012, -0.00022, 0.00017, 0.005, -0.003, -0.1962, 0.02);
    let (cf, cg) = c.compute_error_transition(c_imu(step));
    let (rf, rg) = r.compute_error_transition(step);
    for i in 0..24 {
        for j in 0..24 {
            assert_close(&format!("F[{i}][{j}]"), cf[i][j], rf[i][j], EPS);
        }
        for j in 0..21 {
            assert_close(&format!("G[{i}][{j}]"), cg[i][j], rg[i][j], EPS);
        }
    }
}
