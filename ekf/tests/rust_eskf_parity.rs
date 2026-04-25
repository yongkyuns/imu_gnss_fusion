#![cfg(feature = "c-reference")]

use sensor_fusion::c_api::{CEskf, CEskfImuDelta, CEskfWrapper, EskfGnssSample as CEskfGnssSample};
use sensor_fusion::ekf::PredictNoise;
use sensor_fusion::eskf_types::{EskfGnssSample as RustEskfGnssSample, EskfImuDelta, EskfState};
use sensor_fusion::rust_eskf::RustEskf;

fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
    let d = (a - b).abs();
    assert!(d <= tol, "{ctx}: |{a} - {b}| = {d} > {tol}");
}

fn assert_state_close(c: &CEskf, r: &EskfState, tol_nominal: f32, tol_cov: f32, ctx: &str) {
    let cn = &c.nominal;
    let rn = &r.nominal;
    let nominal_pairs = [
        (cn.q0, rn.q0, "q0"),
        (cn.q1, rn.q1, "q1"),
        (cn.q2, rn.q2, "q2"),
        (cn.q3, rn.q3, "q3"),
        (cn.vn, rn.vn, "vn"),
        (cn.ve, rn.ve, "ve"),
        (cn.vd, rn.vd, "vd"),
        (cn.pn, rn.pn, "pn"),
        (cn.pe, rn.pe, "pe"),
        (cn.pd, rn.pd, "pd"),
        (cn.bgx, rn.bgx, "bgx"),
        (cn.bgy, rn.bgy, "bgy"),
        (cn.bgz, rn.bgz, "bgz"),
        (cn.bax, rn.bax, "bax"),
        (cn.bay, rn.bay, "bay"),
        (cn.baz, rn.baz, "baz"),
        (cn.qcs0, rn.qcs0, "qcs0"),
        (cn.qcs1, rn.qcs1, "qcs1"),
        (cn.qcs2, rn.qcs2, "qcs2"),
        (cn.qcs3, rn.qcs3, "qcs3"),
    ];
    for (a, b, name) in nominal_pairs {
        assert_close(a, b, tol_nominal, &format!("{ctx} nominal {name}"));
    }
    for i in 0..18 {
        for j in 0..18 {
            assert_close(c.p[i][j], r.p[i][j], tol_cov, &format!("{ctx} P[{i}][{j}]"));
        }
    }
    assert_eq!(
        c.update_diag.total_updates, r.update_diag.total_updates,
        "{ctx} total update count"
    );
    assert_eq!(
        c.update_diag.type_counts, r.update_diag.type_counts,
        "{ctx} update type counts"
    );
    let stationary_pairs = [
        (
            c.stationary_diag.innovation_x,
            r.stationary_diag.innovation_x,
            "stationary.innovation_x",
        ),
        (
            c.stationary_diag.innovation_y,
            r.stationary_diag.innovation_y,
            "stationary.innovation_y",
        ),
        (
            c.stationary_diag.k_theta_x_from_x,
            r.stationary_diag.k_theta_x_from_x,
            "stationary.k_theta_x_from_x",
        ),
        (
            c.stationary_diag.k_theta_y_from_x,
            r.stationary_diag.k_theta_y_from_x,
            "stationary.k_theta_y_from_x",
        ),
        (
            c.stationary_diag.k_bax_from_x,
            r.stationary_diag.k_bax_from_x,
            "stationary.k_bax_from_x",
        ),
        (
            c.stationary_diag.k_bay_from_x,
            r.stationary_diag.k_bay_from_x,
            "stationary.k_bay_from_x",
        ),
        (
            c.stationary_diag.k_theta_x_from_y,
            r.stationary_diag.k_theta_x_from_y,
            "stationary.k_theta_x_from_y",
        ),
        (
            c.stationary_diag.k_theta_y_from_y,
            r.stationary_diag.k_theta_y_from_y,
            "stationary.k_theta_y_from_y",
        ),
        (
            c.stationary_diag.k_bax_from_y,
            r.stationary_diag.k_bax_from_y,
            "stationary.k_bax_from_y",
        ),
        (
            c.stationary_diag.k_bay_from_y,
            r.stationary_diag.k_bay_from_y,
            "stationary.k_bay_from_y",
        ),
        (
            c.stationary_diag.p_theta_x,
            r.stationary_diag.p_theta_x,
            "stationary.p_theta_x",
        ),
        (
            c.stationary_diag.p_theta_y,
            r.stationary_diag.p_theta_y,
            "stationary.p_theta_y",
        ),
        (
            c.stationary_diag.p_bax,
            r.stationary_diag.p_bax,
            "stationary.p_bax",
        ),
        (
            c.stationary_diag.p_bay,
            r.stationary_diag.p_bay,
            "stationary.p_bay",
        ),
        (
            c.stationary_diag.p_theta_x_bax,
            r.stationary_diag.p_theta_x_bax,
            "stationary.p_theta_x_bax",
        ),
        (
            c.stationary_diag.p_theta_y_bay,
            r.stationary_diag.p_theta_y_bay,
            "stationary.p_theta_y_bay",
        ),
    ];
    for (a, b, name) in stationary_pairs {
        assert_close(a, b, tol_cov, &format!("{ctx} {name}"));
    }
    assert_eq!(
        c.stationary_diag.updates, r.stationary_diag.updates,
        "{ctx} stationary update count"
    );

    for i in 0..c.update_diag.type_counts.len() {
        assert_close(
            c.update_diag.sum_dx_pitch[i],
            r.update_diag.sum_dx_pitch[i],
            tol_nominal,
            &format!("{ctx} update.sum_dx_pitch[{i}]"),
        );
        assert_close(
            c.update_diag.sum_abs_dx_pitch[i],
            r.update_diag.sum_abs_dx_pitch[i],
            tol_nominal,
            &format!("{ctx} update.sum_abs_dx_pitch[{i}]"),
        );
        assert_close(
            c.update_diag.sum_dx_mount_yaw[i],
            r.update_diag.sum_dx_mount_yaw[i],
            tol_nominal,
            &format!("{ctx} update.sum_dx_mount_yaw[{i}]"),
        );
        assert_close(
            c.update_diag.sum_abs_dx_mount_yaw[i],
            r.update_diag.sum_abs_dx_mount_yaw[i],
            tol_nominal,
            &format!("{ctx} update.sum_abs_dx_mount_yaw[{i}]"),
        );
        assert_close(
            c.update_diag.sum_innovation[i],
            r.update_diag.sum_innovation[i],
            tol_nominal,
            &format!("{ctx} update.sum_innovation[{i}]"),
        );
        assert_close(
            c.update_diag.sum_abs_innovation[i],
            r.update_diag.sum_abs_innovation[i],
            tol_nominal,
            &format!("{ctx} update.sum_abs_innovation[{i}]"),
        );
    }
    assert_close(
        c.update_diag.last_dx_mount_yaw,
        r.update_diag.last_dx_mount_yaw,
        tol_nominal,
        &format!("{ctx} update.last_dx_mount_yaw"),
    );
    assert_close(
        c.update_diag.last_k_mount_yaw,
        r.update_diag.last_k_mount_yaw,
        tol_nominal,
        &format!("{ctx} update.last_k_mount_yaw"),
    );
    assert_close(
        c.update_diag.last_innovation,
        r.update_diag.last_innovation,
        tol_nominal,
        &format!("{ctx} update.last_innovation"),
    );
    assert_close(
        c.update_diag.last_innovation_var,
        r.update_diag.last_innovation_var,
        tol_nominal,
        &format!("{ctx} update.last_innovation_var"),
    );
    assert_eq!(
        c.update_diag.last_type, r.update_diag.last_type,
        "{ctx} update.last_type"
    );
}

fn c_seed_sample() -> CEskfGnssSample {
    CEskfGnssSample {
        t_s: 0.0,
        pos_ned_m: [12.0, -4.0, 1.5],
        vel_ned_mps: [4.5, 0.2, -0.1],
        pos_std_m: [0.8, 0.9, 1.2],
        vel_std_mps: [0.25, 0.3, 0.35],
        heading_rad: Some(0.0),
    }
}

fn rust_seed_sample() -> RustEskfGnssSample {
    RustEskfGnssSample {
        t_s: 0.0,
        pos_ned_m: [12.0, -4.0, 1.5],
        vel_ned_mps: [4.5, 0.2, -0.1],
        pos_std_m: [0.8, 0.9, 1.2],
        vel_std_mps: [0.25, 0.3, 0.35],
        heading_rad: Some(0.0),
    }
}

fn c_gps_sample(step: usize) -> CEskfGnssSample {
    let t = step as f32 * 0.01;
    CEskfGnssSample {
        t_s: t,
        pos_ned_m: [12.0 + 4.4 * t, -4.0 + 0.15 * t, 1.5 - 0.05 * t],
        vel_ned_mps: [4.45 + 0.02 * t, 0.18 - 0.01 * t, -0.08],
        pos_std_m: [0.9, 1.0, 1.4],
        vel_std_mps: [0.28, 0.32, 0.36],
        heading_rad: Some(0.0),
    }
}

fn rust_gps_sample(step: usize) -> RustEskfGnssSample {
    let t = step as f32 * 0.01;
    RustEskfGnssSample {
        t_s: t,
        pos_ned_m: [12.0 + 4.4 * t, -4.0 + 0.15 * t, 1.5 - 0.05 * t],
        vel_ned_mps: [4.45 + 0.02 * t, 0.18 - 0.01 * t, -0.08],
        pos_std_m: [0.9, 1.0, 1.4],
        vel_std_mps: [0.28, 0.32, 0.36],
        heading_rad: Some(0.0),
    }
}

fn imu_delta(step: usize) -> CEskfImuDelta {
    let t = step as f32;
    CEskfImuDelta {
        dax: 0.0008 + 0.00001 * t,
        day: -0.0005 + 0.00002 * t,
        daz: 0.0015 - 0.00001 * t,
        dvx: 0.012 + 0.0001 * t,
        dvy: 0.0015 - 0.00003 * t,
        dvz: -9.80665 * 0.01 + 0.0002,
        dt: 0.01,
    }
}

fn rust_imu_delta(step: usize) -> EskfImuDelta {
    let imu = imu_delta(step);
    EskfImuDelta {
        dax: imu.dax,
        day: imu.day,
        daz: imu.daz,
        dvx: imu.dvx,
        dvy: imu.dvy,
        dvz: imu.dvz,
        dt: imu.dt,
    }
}

#[derive(Clone, Copy)]
struct Lcg(u32);

impl Lcg {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / ((1u32 << 24) as f32)
    }

    fn range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.unit()
    }
}

fn random_c_gps(rng: &mut Lcg, t_s: f32) -> CEskfGnssSample {
    CEskfGnssSample {
        t_s,
        pos_ned_m: [
            rng.range(-25.0, 25.0),
            rng.range(-25.0, 25.0),
            rng.range(-3.0, 3.0),
        ],
        vel_ned_mps: [
            rng.range(-6.0, 8.0),
            rng.range(-2.0, 2.0),
            rng.range(-0.5, 0.5),
        ],
        pos_std_m: [
            rng.range(0.6, 2.5),
            rng.range(0.6, 2.5),
            rng.range(0.8, 3.0),
        ],
        vel_std_mps: [
            rng.range(0.2, 1.0),
            rng.range(0.2, 1.0),
            rng.range(0.3, 1.2),
        ],
        heading_rad: None,
    }
}

fn c_gps_to_rust(c: CEskfGnssSample) -> RustEskfGnssSample {
    RustEskfGnssSample {
        t_s: c.t_s,
        pos_ned_m: c.pos_ned_m,
        vel_ned_mps: c.vel_ned_mps,
        pos_std_m: c.pos_std_m,
        vel_std_mps: c.vel_std_mps,
        heading_rad: c.heading_rad,
    }
}

#[test]
fn rust_eskf_matches_c_for_predict_and_updates() {
    let noise = PredictNoise {
        gyro_var: 2.287_311_3e-7,
        accel_var: 2.450_421_4e-5,
        gyro_bias_rw_var: 0.0002e-9,
        accel_bias_rw_var: 0.002e-9,
        mount_align_rw_var: 1.0e-8,
    };
    let mut c = CEskfWrapper::new(noise);
    let mut r = RustEskf::new(noise);
    let q_bn = [0.998_750_27, 0.0, 0.0, 0.049_979_17];

    c.init_nominal_from_gnss(q_bn, c_seed_sample());
    r.init_nominal_from_gnss(q_bn, rust_seed_sample());
    assert_state_close(c.raw(), r.raw(), 1.0e-6, 1.0e-6, "after init");

    for step in 1..=60 {
        let imu = imu_delta(step);
        c.predict(imu);
        r.predict(rust_imu_delta(step));

        if step % 5 == 0 {
            c.fuse_gps(c_gps_sample(step));
            r.fuse_gps(rust_gps_sample(step));
        }
        if step % 7 == 0 {
            c.fuse_body_speed_x(4.6, 0.09);
            r.fuse_body_speed_x(4.6, 0.09);
        }
        if step % 9 == 0 {
            c.fuse_body_vel(0.35);
            r.fuse_body_vel(0.35);
        }
        if step % 13 == 0 {
            c.fuse_zero_vel(0.25);
            r.fuse_zero_vel(0.25);
        }
        if step % 17 == 0 {
            c.fuse_stationary_gravity([0.02, -0.03, -9.80], 0.20);
            r.fuse_stationary_gravity([0.02, -0.03, -9.80], 0.20);
        }

        assert_state_close(c.raw(), r.raw(), 1.0e-6, 1.0e-6, &format!("step {step}"));
    }
}

#[test]
fn generated_error_transition_matches_c_exact_path() {
    let mut c = CEskfWrapper::new(PredictNoise::default());
    let mut r = RustEskf::new(PredictNoise::default());
    let q_bn = [0.999_550_04, 0.004, -0.002, 0.029_995_5];
    c.init_nominal_from_gnss(q_bn, c_seed_sample());
    r.init_nominal_from_gnss(q_bn, rust_seed_sample());
    c.predict(imu_delta(1));
    r.predict(rust_imu_delta(1));

    let imu = imu_delta(2);
    let c_f = c.compute_error_transition(imu);
    let r_f = r.compute_error_transition(rust_imu_delta(2));
    for i in 0..18 {
        for j in 0..18 {
            assert_close(c_f[i][j], r_f[i][j], 1.0e-6, &format!("F[{i}][{j}]"));
        }
    }
}

#[test]
fn rust_eskf_matches_c_across_deterministic_random_sequences() {
    for case in 0..8 {
        let seed = 0x5eed_1000 + case;
        let mut c_rng = Lcg::new(seed);
        let noise = PredictNoise {
            gyro_var: c_rng.range(1.0e-7, 4.0e-6),
            accel_var: c_rng.range(1.0e-5, 7.0e-4),
            gyro_bias_rw_var: c_rng.range(1.0e-13, 8.0e-13),
            accel_bias_rw_var: c_rng.range(1.0e-12, 8.0e-12),
            mount_align_rw_var: c_rng.range(1.0e-9, 5.0e-8),
        };
        let mut c = CEskfWrapper::new(noise);
        let mut r = RustEskf::new(noise);
        c.init_nominal_from_gnss([1.0, 0.0, 0.0, 0.0], c_seed_sample());
        r.init_nominal_from_gnss([1.0, 0.0, 0.0, 0.0], rust_seed_sample());

        for step in 1..=120 {
            let dt = c_rng.range(0.005, 0.02);
            let c_imu = CEskfImuDelta {
                dax: c_rng.range(-0.003, 0.003),
                day: c_rng.range(-0.003, 0.003),
                daz: c_rng.range(-0.004, 0.004),
                dvx: c_rng.range(-0.04, 0.06),
                dvy: c_rng.range(-0.02, 0.02),
                dvz: -9.80665 * dt + c_rng.range(-0.01, 0.01),
                dt,
            };
            let r_imu = EskfImuDelta {
                dax: c_imu.dax,
                day: c_imu.day,
                daz: c_imu.daz,
                dvx: c_imu.dvx,
                dvy: c_imu.dvy,
                dvz: c_imu.dvz,
                dt: c_imu.dt,
            };
            c.predict(c_imu);
            r.predict(r_imu);

            let t_s = step as f32 * dt;
            if c_rng.next_u32() % 4 == 0 {
                let gps = random_c_gps(&mut c_rng, t_s);
                c.fuse_gps(gps);
                r.fuse_gps(c_gps_to_rust(gps));
            }
            if c_rng.next_u32() % 6 == 0 {
                let speed = c_rng.range(-2.0, 8.0);
                let r_speed = c_rng.range(0.05, 1.5);
                c.fuse_body_speed_x(speed, r_speed);
                r.fuse_body_speed_x(speed, r_speed);
            }
            if c_rng.next_u32() % 5 == 0 {
                let r_body = c_rng.range(0.05, 2.0);
                c.fuse_body_vel(r_body);
                r.fuse_body_vel(r_body);
            }
            if c_rng.next_u32() % 11 == 0 {
                let r_zero = c_rng.range(0.05, 1.0);
                c.fuse_zero_vel(r_zero);
                r.fuse_zero_vel(r_zero);
            }
            if c_rng.next_u32() % 13 == 0 {
                let accel = [
                    c_rng.range(-0.08, 0.08),
                    c_rng.range(-0.08, 0.08),
                    -9.80665 + c_rng.range(-0.05, 0.05),
                ];
                let r_accel = c_rng.range(0.1, 0.8);
                c.fuse_stationary_gravity(accel, r_accel);
                r.fuse_stationary_gravity(accel, r_accel);
            }

            assert_state_close(
                c.raw(),
                r.raw(),
                1.0e-6,
                1.0e-6,
                &format!("case {case} step {step}"),
            );
        }
    }
}
