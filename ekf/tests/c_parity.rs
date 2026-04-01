use sensor_fusion::ekf::{
    Ekf, EkfDebug, EkfState, GpsData, ImuSample, N_STATES, PredictNoise, ekf_fuse_gps, ekf_init,
    ekf_predict,
};

#[repr(C)]
#[derive(Clone, Copy)]
struct CEkfState {
    q0: f32,
    q1: f32,
    q2: f32,
    q3: f32,
    vn: f32,
    ve: f32,
    vd: f32,
    pn: f32,
    pe: f32,
    pd: f32,
    dax_b: f32,
    day_b: f32,
    daz_b: f32,
    dvx_b: f32,
    dvy_b: f32,
    dvz_b: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CImuSample {
    dax: f32,
    day: f32,
    daz: f32,
    dvx: f32,
    dvy: f32,
    dvz: f32,
    dt: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CEkf {
    state: CEkfState,
    p: [[f32; N_STATES]; N_STATES],
    noise: CPredictNoise,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CPredictNoise {
    gyro_var: f32,
    accel_var: f32,
    gyro_bias_rw_var: f32,
    accel_bias_rw_var: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGpsData {
    pos_n: f32,
    pos_e: f32,
    pos_d: f32,
    vel_n: f32,
    vel_e: f32,
    vel_d: f32,
    r_pos_n: f32,
    r_pos_e: f32,
    r_pos_d: f32,
    r_vel_n: f32,
    r_vel_e: f32,
    r_vel_d: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CEkfDebug {
    dvb_x: f32,
    dvb_y: f32,
    dvb_z: f32,
}

unsafe extern "C" {
    #[link_name = "ekf_init"]
    fn c_ekf_init(ekf: *mut CEkf, p_diag: *const f32, noise: *const CPredictNoise);
    #[link_name = "ekf_set_predict_noise"]
    fn c_ekf_set_predict_noise(ekf: *mut CEkf, noise: *const CPredictNoise);
    #[link_name = "ekf_predict"]
    fn c_ekf_predict(ekf: *mut CEkf, imu: *const CImuSample, debug_out: *mut CEkfDebug);
    #[link_name = "ekf_fuse_gps"]
    fn c_ekf_fuse_gps(ekf: *mut CEkf, gps: *const CGpsData);
}

fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
    let d = (a - b).abs();
    assert!(d <= tol, "{ctx}: |{a} - {b}| = {d} > {tol}");
}

fn to_c_imu(v: &ImuSample) -> CImuSample {
    CImuSample {
        dax: v.dax,
        day: v.day,
        daz: v.daz,
        dvx: v.dvx,
        dvy: v.dvy,
        dvz: v.dvz,
        dt: v.dt,
    }
}

fn to_c_gps(v: &GpsData) -> CGpsData {
    CGpsData {
        pos_n: v.pos_n,
        pos_e: v.pos_e,
        pos_d: v.pos_d,
        vel_n: v.vel_n,
        vel_e: v.vel_e,
        vel_d: v.vel_d,
        r_pos_n: v.R_POS_N,
        r_pos_e: v.R_POS_E,
        r_pos_d: v.R_POS_D,
        r_vel_n: v.R_VEL_N,
        r_vel_e: v.R_VEL_E,
        r_vel_d: v.R_VEL_D,
    }
}

fn to_c_state(v: &EkfState) -> CEkfState {
    CEkfState {
        q0: v.q0,
        q1: v.q1,
        q2: v.q2,
        q3: v.q3,
        vn: v.vn,
        ve: v.ve,
        vd: v.vd,
        pn: v.pn,
        pe: v.pe,
        pd: v.pd,
        dax_b: v.dax_b,
        day_b: v.day_b,
        daz_b: v.daz_b,
        dvx_b: v.dvx_b,
        dvy_b: v.dvy_b,
        dvz_b: v.dvz_b,
    }
}

#[test]
fn rust_matches_c_ekf_outputs() {
    let mut rust = Ekf::default();
    let c_noise = CPredictNoise {
        gyro_var: 2.5_f32,
        accel_var: 12.0_f32,
        gyro_bias_rw_var: 5.0e-7_f32,
        accel_bias_rw_var: 2.5e-6_f32,
    };
    ekf_init(
        &mut rust,
        [1.0; N_STATES],
        PredictNoise {
            gyro_var: c_noise.gyro_var,
            accel_var: c_noise.accel_var,
            gyro_bias_rw_var: c_noise.gyro_bias_rw_var,
            accel_bias_rw_var: c_noise.accel_bias_rw_var,
        },
    );
    rust.state = EkfState {
        q0: 1.0,
        q1: 0.01,
        q2: -0.02,
        q3: 0.03,
        vn: 0.4,
        ve: -0.1,
        vd: 0.05,
        pn: 5.0,
        pe: -3.0,
        pd: 1.5,
        dax_b: 0.001,
        day_b: -0.0015,
        daz_b: 0.002,
        dvx_b: 0.0002,
        dvy_b: -0.0003,
        dvz_b: 0.0001,
    };

    let mut c = CEkf {
        state: to_c_state(&rust.state),
        p: rust.p,
        noise: CPredictNoise {
            gyro_var: 0.0,
            accel_var: 0.0,
            gyro_bias_rw_var: 0.0,
            accel_bias_rw_var: 0.0,
        },
    };
    // SAFETY: `c` points to a valid `CEkf`.
    let c_p_diag = [1.0_f32; N_STATES];
    unsafe {
        c_ekf_init(
            &mut c as *mut CEkf,
            c_p_diag.as_ptr(),
            &c_noise as *const CPredictNoise,
        )
    };
    c.state = to_c_state(&rust.state);
    // SAFETY: pointers are valid and C function does not retain them.
    unsafe { c_ekf_set_predict_noise(&mut c as *mut CEkf, &c_noise as *const CPredictNoise) };

    let mut rust_dbg = EkfDebug::default();
    let mut c_dbg = CEkfDebug::default();

    for k in 0..300 {
        let t = k as f32 * 0.01;
        let imu = ImuSample {
            dax: 0.001 * (0.7 * t).sin(),
            day: 0.0012 * (0.9 * t).cos(),
            daz: -0.0008 * (0.5 * t).sin(),
            dvx: 0.02 * (0.3 * t).sin(),
            dvy: 0.015 * (0.4 * t).cos(),
            dvz: -0.01 * (0.2 * t).sin(),
            dt: 0.01,
        };

        ekf_predict(&mut rust, &imu, Some(&mut rust_dbg));

        let c_imu = to_c_imu(&imu);
        // SAFETY: pointers are valid and C function does not retain them.
        unsafe {
            c_ekf_predict(
                &mut c as *mut CEkf,
                &c_imu as *const CImuSample,
                &mut c_dbg as *mut CEkfDebug,
            );
        }

        if k % 20 == 0 {
            let gps = GpsData {
                pos_n: rust.state.pn + 0.2 * (0.11 * t).sin(),
                pos_e: rust.state.pe - 0.15 * (0.08 * t).cos(),
                pos_d: rust.state.pd + 0.1 * (0.05 * t).sin(),
                vel_n: rust.state.vn + 0.05 * (0.17 * t).sin(),
                vel_e: rust.state.ve - 0.03 * (0.07 * t).cos(),
                vel_d: rust.state.vd + 0.02 * (0.13 * t).sin(),
                R_POS_N: 0.5,
                R_POS_E: 0.5,
                R_POS_D: 0.8,
                R_VEL_N: 0.2,
                R_VEL_E: 0.2,
                R_VEL_D: 0.25,
            };
            ekf_fuse_gps(&mut rust, &gps);
            let c_gps = to_c_gps(&gps);
            // SAFETY: pointers are valid and C function does not retain them.
            unsafe {
                c_ekf_fuse_gps(&mut c as *mut CEkf, &c_gps as *const CGpsData);
            }
        }
    }

    let tol = 1e-5_f32;
    let rs = &rust.state;
    let cs = &c.state;
    assert_close(rs.q0, cs.q0, tol, "state.q0");
    assert_close(rs.q1, cs.q1, tol, "state.q1");
    assert_close(rs.q2, cs.q2, tol, "state.q2");
    assert_close(rs.q3, cs.q3, tol, "state.q3");
    assert_close(rs.vn, cs.vn, tol, "state.vn");
    assert_close(rs.ve, cs.ve, tol, "state.ve");
    assert_close(rs.vd, cs.vd, tol, "state.vd");
    assert_close(rs.pn, cs.pn, tol, "state.pn");
    assert_close(rs.pe, cs.pe, tol, "state.pe");
    assert_close(rs.pd, cs.pd, tol, "state.pd");
    assert_close(rs.dax_b, cs.dax_b, tol, "state.dax_b");
    assert_close(rs.day_b, cs.day_b, tol, "state.day_b");
    assert_close(rs.daz_b, cs.daz_b, tol, "state.daz_b");
    assert_close(rs.dvx_b, cs.dvx_b, tol, "state.dvx_b");
    assert_close(rs.dvy_b, cs.dvy_b, tol, "state.dvy_b");
    assert_close(rs.dvz_b, cs.dvz_b, tol, "state.dvz_b");

    for i in 0..N_STATES {
        for j in 0..N_STATES {
            assert_close(rust.p[i][j], c.p[i][j], tol, &format!("P[{i}][{j}]"));
        }
    }
}
