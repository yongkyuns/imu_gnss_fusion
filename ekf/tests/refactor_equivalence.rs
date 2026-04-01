use sensor_fusion::ekf::{
    Ekf, GpsData, ImuSample, N_STATES, PredictNoise, ekf_fuse_body_vel, ekf_fuse_gps, ekf_init,
    ekf_predict,
};

fn fnv1a64_extend(mut h: u64, x: u64) -> u64 {
    const FNV_PRIME: u64 = 1099511628211;
    h ^= x;
    h.wrapping_mul(FNV_PRIME)
}

fn quantized_bits(v: f32) -> u64 {
    // Quantize to 1e-9 to absorb tiny float round-off while still being strict.
    let q = ((v as f64) * 1.0e9_f64).round() as i64;
    q as u64
}

fn signature_state_and_cov(ekf: &Ekf) -> (u64, u64) {
    let mut hs = 1469598103934665603_u64; // FNV-1a offset basis
    let mut hp = 1469598103934665603_u64;
    let s = &ekf.state;
    let state = [
        s.q0, s.q1, s.q2, s.q3, s.vn, s.ve, s.vd, s.pn, s.pe, s.pd, s.dax_b, s.day_b, s.daz_b,
        s.dvx_b, s.dvy_b, s.dvz_b,
    ];
    for v in state {
        hs = fnv1a64_extend(hs, quantized_bits(v));
    }
    for i in 0..N_STATES {
        for j in 0..N_STATES {
            hp = fnv1a64_extend(hp, quantized_bits(ekf.p[i][j]));
        }
    }
    (hs, hp)
}

fn run_excitation_sequence() -> Ekf {
    let mut ekf = Ekf::default();
    ekf_init(
        &mut ekf,
        [0.7; N_STATES],
        PredictNoise {
            gyro_var: 2.2,
            accel_var: 13.0,
            gyro_bias_rw_var: 4.0e-7,
            accel_bias_rw_var: 2.1e-6,
        },
    );
    ekf.state.q0 = 0.98;
    ekf.state.q1 = 0.04;
    ekf.state.q2 = -0.06;
    ekf.state.q3 = 0.10;
    ekf.state.vn = 0.5;
    ekf.state.ve = -0.3;
    ekf.state.vd = 0.2;
    ekf.state.pn = 3.0;
    ekf.state.pe = -2.0;
    ekf.state.pd = 1.0;
    ekf.state.dax_b = 2.0e-4;
    ekf.state.day_b = -1.0e-4;
    ekf.state.daz_b = 1.5e-4;
    ekf.state.dvx_b = 1.2e-4;
    ekf.state.dvy_b = -0.8e-4;
    ekf.state.dvz_b = 0.6e-4;

    for k in 0..1500 {
        let t = k as f32 * 0.01;
        let dt = 0.0095 + 0.0008 * (0.13 * t).sin();
        let imu = ImuSample {
            dax: 0.0015 * (0.53 * t).sin() + 0.0004 * (0.21 * t).cos(),
            day: 0.0011 * (0.71 * t).cos() - 0.0003 * (0.19 * t).sin(),
            daz: -0.0013 * (0.37 * t).sin() + 0.0002 * (0.43 * t).cos(),
            dvx: 0.032 * (0.31 * t).sin() + 0.004 * (0.07 * t).cos(),
            dvy: 0.027 * (0.41 * t).cos() - 0.003 * (0.29 * t).sin(),
            dvz: -0.021 * (0.23 * t).sin() + 0.002 * (0.17 * t).cos(),
            dt,
        };
        ekf_predict(&mut ekf, &imu, None);

        if k % 4 == 0 {
            let r_body = 45.0 + 25.0 * (0.17 * t).sin().abs();
            ekf_fuse_body_vel(&mut ekf, r_body);
        }

        if k % 15 == 0 {
            let gps = GpsData {
                pos_n: 8.0 * (0.05 * t).sin() + 0.3 * (0.18 * t).cos(),
                pos_e: -6.5 * (0.045 * t).cos() + 0.25 * (0.11 * t).sin(),
                pos_d: 1.8 * (0.022 * t).sin() - 0.2 * (0.09 * t).cos(),
                vel_n: 0.6 * (0.041 * t).cos() + 0.05 * (0.13 * t).sin(),
                vel_e: -0.5 * (0.039 * t).sin() + 0.06 * (0.07 * t).cos(),
                vel_d: 0.2 * (0.027 * t).cos() - 0.03 * (0.16 * t).sin(),
                R_POS_N: 0.4 + 0.2 * (0.09 * t).sin().abs(),
                R_POS_E: 0.42 + 0.2 * (0.07 * t).cos().abs(),
                R_POS_D: 0.7 + 0.15 * (0.11 * t).sin().abs(),
                R_VEL_N: 0.12 + 0.05 * (0.05 * t).cos().abs(),
                R_VEL_E: 0.11 + 0.05 * (0.06 * t).sin().abs(),
                R_VEL_D: 0.17 + 0.04 * (0.08 * t).cos().abs(),
            };
            ekf_fuse_gps(&mut ekf, &gps);
        }
    }

    ekf
}

#[test]
fn refactor_equivalence_snapshot() {
    let ekf = run_excitation_sequence();
    let (state_sig, cov_sig) = signature_state_and_cov(&ekf);

    // If this fails after intended model changes, update only with explicit approval.
    // Printed signatures make rebasing snapshot straightforward.
    println!("state_sig={state_sig:#018x} cov_sig={cov_sig:#018x}");

    const EXPECTED_STATE_SIG: u64 = 0xd5881308efe98af9;
    const EXPECTED_COV_SIG: u64 = 0xad9ce4566f0140cb;
    assert_eq!(state_sig, EXPECTED_STATE_SIG, "state signature mismatch");
    assert_eq!(cov_sig, EXPECTED_COV_SIG, "covariance signature mismatch");
}
