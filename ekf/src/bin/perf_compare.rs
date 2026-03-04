use ekf_rs::ekf::{
    Ekf, EkfDebug, EkfState, GpsData, ImuSample, N_STATES, PredictNoise, ekf_fuse_gps, ekf_init,
    ekf_predict,
};
use std::time::Instant;

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

#[inline]
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

#[inline]
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

fn main() {
    let n_iter = 10_000usize;
    let fuse_every = 20usize;
    let rounds = 20usize;

    let init_state = EkfState {
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

    let c_noise = CPredictNoise {
        gyro_var: 2.5_f32,
        accel_var: 12.0_f32,
        gyro_bias_rw_var: 5.0e-7_f32,
        accel_bias_rw_var: 2.5e-6_f32,
    };

    let mut imu_seq = Vec::with_capacity(n_iter);
    let mut gps_seq = Vec::with_capacity(n_iter);
    for k in 0..n_iter {
        let t = k as f32 * 0.01;
        imu_seq.push(ImuSample {
            dax: 0.001 * (0.7 * t).sin(),
            day: 0.0012 * (0.9 * t).cos(),
            daz: -0.0008 * (0.5 * t).sin(),
            dvx: 0.02 * (0.3 * t).sin(),
            dvy: 0.015 * (0.4 * t).cos(),
            dvz: -0.01 * (0.2 * t).sin(),
            dt: 0.01,
        });
        gps_seq.push(GpsData {
            pos_n: 0.2 * (0.11 * t).sin(),
            pos_e: -0.15 * (0.08 * t).cos(),
            pos_d: 0.1 * (0.05 * t).sin(),
            vel_n: 0.05 * (0.17 * t).sin(),
            vel_e: -0.03 * (0.07 * t).cos(),
            vel_d: 0.02 * (0.13 * t).sin(),
            R_POS_N: 0.5,
            R_POS_E: 0.5,
            R_POS_D: 0.8,
            R_VEL_N: 0.2,
            R_VEL_E: 0.2,
            R_VEL_D: 0.25,
        });
    }

    let run_rust = |imu_seq: &[ImuSample], gps_seq: &[GpsData]| -> f64 {
        let mut rust = Ekf::default();
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
        rust.state = init_state;
        let mut rust_dbg = EkfDebug::default();
        let t0 = Instant::now();
        for k in 0..n_iter {
            let imu = imu_seq[k];
            ekf_predict(&mut rust, &imu, Some(&mut rust_dbg));
            if k % fuse_every == 0 {
                let mut gps = gps_seq[k];
                gps.pos_n += rust.state.pn;
                gps.pos_e += rust.state.pe;
                gps.pos_d += rust.state.pd;
                gps.vel_n += rust.state.vn;
                gps.vel_e += rust.state.ve;
                gps.vel_d += rust.state.vd;
                ekf_fuse_gps(&mut rust, &gps);
            }
        }
        t0.elapsed().as_secs_f64() * 1e6
    };

    let run_c = |imu_seq: &[ImuSample], gps_seq: &[GpsData]| -> f64 {
        let mut c = CEkf {
            state: CEkfState {
                q0: init_state.q0,
                q1: init_state.q1,
                q2: init_state.q2,
                q3: init_state.q3,
                vn: init_state.vn,
                ve: init_state.ve,
                vd: init_state.vd,
                pn: init_state.pn,
                pe: init_state.pe,
                pd: init_state.pd,
                dax_b: init_state.dax_b,
                day_b: init_state.day_b,
                daz_b: init_state.daz_b,
                dvx_b: init_state.dvx_b,
                dvy_b: init_state.dvy_b,
                dvz_b: init_state.dvz_b,
            },
            p: [[0.0; N_STATES]; N_STATES],
            noise: c_noise,
        };
        // SAFETY: valid mutable pointer to C-compatible struct.
        let c_p_diag = [1.0_f32; N_STATES];
        unsafe {
            c_ekf_init(
                &mut c as *mut CEkf,
                c_p_diag.as_ptr(),
                &c_noise as *const CPredictNoise,
            )
        };
        // SAFETY: valid mutable pointer to C-compatible struct.
        unsafe { c_ekf_set_predict_noise(&mut c as *mut CEkf, &c_noise as *const CPredictNoise) };
        c.state = CEkfState {
            q0: init_state.q0,
            q1: init_state.q1,
            q2: init_state.q2,
            q3: init_state.q3,
            vn: init_state.vn,
            ve: init_state.ve,
            vd: init_state.vd,
            pn: init_state.pn,
            pe: init_state.pe,
            pd: init_state.pd,
            dax_b: init_state.dax_b,
            day_b: init_state.day_b,
            daz_b: init_state.daz_b,
            dvx_b: init_state.dvx_b,
            dvy_b: init_state.dvy_b,
            dvz_b: init_state.dvz_b,
        };
        let mut c_dbg = CEkfDebug::default();
        let t0 = Instant::now();
        for k in 0..n_iter {
            let c_imu = to_c_imu(&imu_seq[k]);
            // SAFETY: pointers are valid for call duration.
            unsafe {
                c_ekf_predict(
                    &mut c as *mut CEkf,
                    &c_imu as *const CImuSample,
                    &mut c_dbg as *mut CEkfDebug,
                );
            }
            if k % fuse_every == 0 {
                let mut gps = gps_seq[k];
                gps.pos_n += c.state.pn;
                gps.pos_e += c.state.pe;
                gps.pos_d += c.state.pd;
                gps.vel_n += c.state.vn;
                gps.vel_e += c.state.ve;
                gps.vel_d += c.state.vd;
                let c_gps = to_c_gps(&gps);
                // SAFETY: pointers are valid for call duration.
                unsafe { c_ekf_fuse_gps(&mut c as *mut CEkf, &c_gps as *const CGpsData) };
            }
        }
        t0.elapsed().as_secs_f64() * 1e6
    };

    let mut rust_runs = Vec::with_capacity(rounds);
    let mut c_runs = Vec::with_capacity(rounds);
    for r in 0..rounds {
        if r % 2 == 0 {
            rust_runs.push(run_rust(&imu_seq, &gps_seq));
            c_runs.push(run_c(&imu_seq, &gps_seq));
        } else {
            c_runs.push(run_c(&imu_seq, &gps_seq));
            rust_runs.push(run_rust(&imu_seq, &gps_seq));
        }
    }

    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let mut rust_sorted = rust_runs.clone();
    rust_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut c_sorted = c_runs.clone();
    c_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rust_med = rust_sorted[rust_sorted.len() / 2];
    let c_med = c_sorted[c_sorted.len() / 2];
    let rust_avg = avg(&rust_runs);
    let c_avg = avg(&c_runs);

    println!("iterations: {n_iter}, gps_fuse_every: {fuse_every}, rounds: {rounds}");
    println!(
        "Rust avg: {:.3} ms ({:.3} us/iter), median: {:.3} ms",
        rust_avg / 1000.0,
        rust_avg / n_iter as f64,
        rust_med / 1000.0
    );
    println!(
        "C    avg: {:.3} ms ({:.3} us/iter), median: {:.3} ms",
        c_avg / 1000.0,
        c_avg / n_iter as f64,
        c_med / 1000.0
    );
    if c_avg > 0.0 {
        println!("Rust speedup over C (avg): {:.3}x", c_avg / rust_avg);
    }
}
