#![cfg(feature = "c-reference")]

use sensor_fusion::c_api::{CEskf, CSensorFusionWrapper};
use sensor_fusion::eskf_types::EskfState;
use sensor_fusion::fusion::{
    FusionGnssSample, FusionImuSample, FusionVehicleSpeedDirection, FusionVehicleSpeedSample,
    SensorFusion,
};

fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
    let d = (a - b).abs();
    assert!(d <= tol, "{ctx}: |{a} - {b}| = {d} > {tol}");
}

fn assert_fusion_state_close(c: &CEskf, r: &EskfState, ctx: &str) {
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
        assert_close(a, b, 1.0e-6, &format!("{ctx} nominal {name}"));
    }
    for i in 0..18 {
        for j in 0..18 {
            assert_close(c.p[i][j], r.p[i][j], 1.0e-6, &format!("{ctx} P[{i}][{j}]"));
        }
    }
    assert_eq!(
        c.update_diag.total_updates, r.update_diag.total_updates,
        "{ctx} update count"
    );
    assert_eq!(
        c.update_diag.type_counts, r.update_diag.type_counts,
        "{ctx} update type counts"
    );
}

fn gnss(t_s: f32, pn_m: f32, speed_mps: f32) -> FusionGnssSample {
    let lon_deg = (pn_m as f64 / 6378137.0).to_degrees() as f32;
    FusionGnssSample {
        t_s,
        lat_deg: 0.0,
        lon_deg,
        height_m: 0.0,
        vel_ned_mps: [speed_mps, 0.1, 0.0],
        pos_std_m: [0.9, 1.0, 1.5],
        vel_std_mps: [0.25, 0.3, 0.35],
        heading_rad: Some(0.02),
    }
}

#[test]
fn rust_sensor_fusion_external_path_matches_c_reference_output() {
    let q_vb = [1.0, 0.0, 0.0, 0.0];
    let mut c = CSensorFusionWrapper::new_external(q_vb);
    let mut r = SensorFusion::with_misalignment(q_vb);

    let first = gnss(1.0, 0.0, 4.0);
    let c_update = c.process_gnss(first);
    let r_update = r.process_gnss(first);
    assert_eq!(c_update.ekf_initialized_now, r_update.ekf_initialized_now);
    assert_eq!(c_update.mount_ready, r_update.mount_ready);
    assert_fusion_state_close(c.eskf().unwrap(), r.eskf().unwrap(), "init");

    for i in 1..80 {
        let t = 1.0 + i as f32 * 0.01;
        let imu = FusionImuSample {
            t_s: t,
            gyro_radps: [0.001, -0.0005, 0.002],
            accel_mps2: [0.03, 0.0, -9.80665],
        };
        let _ = c.process_imu(imu);
        let _ = r.process_imu(imu);
        if i % 20 == 0 {
            let g = gnss(t, 4.0 * (t - 1.0), 4.0 + 0.02 * i as f32);
            let _ = c.process_gnss(g);
            let _ = r.process_gnss(g);
        }
        if i % 17 == 0 {
            let speed = FusionVehicleSpeedSample {
                t_s: t,
                speed_mps: 4.1,
                direction: FusionVehicleSpeedDirection::Forward,
            };
            let _ = c.process_vehicle_speed(speed);
            let _ = r.process_vehicle_speed(speed);
        }
    }

    assert_fusion_state_close(c.eskf().unwrap(), r.eskf().unwrap(), "final");
}
