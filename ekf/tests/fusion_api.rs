use sensor_fusion::fusion::{
    FusionGnssSample, FusionImuSample, FusionVehicleSpeedDirection, FusionVehicleSpeedSample,
    SensorFusion,
};
use sensor_fusion::{ekf::PredictNoise, rust_eskf::RustEskf};

fn gnss_sample(t_s: f32) -> FusionGnssSample {
    FusionGnssSample {
        t_s,
        lat_deg: 0.0,
        lon_deg: 0.0,
        height_m: 0.0,
        vel_ned_mps: [5.0, 0.0, 0.0],
        pos_std_m: [1.0, 1.0, 1.5],
        vel_std_mps: [0.2, 0.2, 0.2],
        heading_rad: Some(0.0),
    }
}

fn stationary_gnss_sample(t_s: f32) -> FusionGnssSample {
    FusionGnssSample {
        vel_ned_mps: [0.0, 0.0, 0.0],
        heading_rad: None,
        ..gnss_sample(t_s)
    }
}

#[test]
fn external_misalignment_initializes_ekf_from_gnss() {
    let mut system = SensorFusion::with_misalignment([1.0, 0.0, 0.0, 0.0]);
    let upd = system.process_gnss(gnss_sample(1.0));
    assert!(upd.mount_ready);
    assert!(upd.ekf_initialized_now);
    assert!(system.eskf().is_some());
}

#[test]
fn vehicle_speed_sample_pulls_forward_velocity_upward() {
    let mut system = SensorFusion::with_misalignment([1.0, 0.0, 0.0, 0.0]);
    let upd = system.process_gnss(gnss_sample(1.0));
    assert!(upd.ekf_initialized_now);
    let vn_before = system.eskf().unwrap().nominal.vn;
    let _ = system.process_vehicle_speed(FusionVehicleSpeedSample {
        t_s: 1.1,
        speed_mps: 6.0,
        direction: FusionVehicleSpeedDirection::Forward,
    });
    let vn_after = system.eskf().unwrap().nominal.vn;
    assert!(vn_after > vn_before);
    assert!(vn_after < 6.0);
}

#[test]
fn freeze_misalignment_states_blocks_mount_updates() {
    let mut system = SensorFusion::with_misalignment([1.0, 0.0, 0.0, 0.0]);
    system.set_freeze_misalignment_states(true);
    system.set_mount_update_min_scale(1.0);
    system.set_mount_update_ramp_time_s(0.0);
    system.set_mount_update_innovation_gate_mps(0.0);
    system.set_r_vehicle_speed(0.001);

    let upd = system.process_gnss(gnss_sample(1.0));
    assert!(upd.ekf_initialized_now);
    let qcs_before = {
        let eskf = system.eskf().unwrap();
        [
            eskf.nominal.qcs0,
            eskf.nominal.qcs1,
            eskf.nominal.qcs2,
            eskf.nominal.qcs3,
        ]
    };

    let _ = system.process_vehicle_speed(FusionVehicleSpeedSample {
        t_s: 1.1,
        speed_mps: 7.0,
        direction: FusionVehicleSpeedDirection::Forward,
    });

    let eskf = system.eskf().unwrap();
    let qcs_after = [
        eskf.nominal.qcs0,
        eskf.nominal.qcs1,
        eskf.nominal.qcs2,
        eskf.nominal.qcs3,
    ];
    assert_eq!(qcs_after, qcs_before);
    assert_eq!(eskf.update_diag.last_dx_mount_yaw, 0.0);
    assert_eq!(eskf.update_diag.last_k_mount_yaw, 0.0);
    for i in 15..18 {
        for j in 0..18 {
            assert_eq!(eskf.p[i][j], 0.0);
            assert_eq!(eskf.p[j][i], 0.0);
        }
    }
}

#[test]
fn mount_settle_phase_releases_with_configured_covariance() {
    let mut system = SensorFusion::with_misalignment([1.0, 0.0, 0.0, 0.0]);
    system.set_mount_settle_time_s(1.0);
    system.set_mount_settle_release_sigma_rad(4.0_f32.to_radians());

    let upd = system.process_gnss(gnss_sample(1.0));
    assert!(upd.ekf_initialized_now);
    for i in 15..18 {
        assert_eq!(system.eskf().unwrap().p[i][i], 0.0);
    }

    let _ = system.process_gnss(gnss_sample(1.5));
    for i in 15..18 {
        assert_eq!(system.eskf().unwrap().p[i][i], 0.0);
    }

    let _ = system.process_gnss(gnss_sample(2.2));
    let eskf = system.eskf().unwrap();
    let release_var = 4.0_f32.to_radians().powi(2);
    for i in 15..18 {
        assert!((eskf.p[i][i] - release_var).abs() < 1.0e-8);
    }
    for i in 15..18 {
        for j in 0..15 {
            assert_eq!(eskf.p[i][j], 0.0);
            assert_eq!(eskf.p[j][i], 0.0);
        }
    }
}

#[test]
fn zero_mount_update_scale_does_not_shrink_mount_covariance() {
    let mut eskf = RustEskf::new(PredictNoise::default());
    {
        let raw = eskf.raw_mut();
        raw.nominal.vn = 8.0;
        raw.nominal.ve = 0.4;
        raw.nominal.vd = -0.2;
        raw.p = [[0.0; 18]; 18];
        for i in 0..18 {
            raw.p[i][i] = 1.0;
        }
    }

    let qcs_before = {
        let n = &eskf.raw().nominal;
        [n.qcs0, n.qcs1, n.qcs2, n.qcs3]
    };
    let mount_cov_before = [
        eskf.raw().p[15][15],
        eskf.raw().p[16][16],
        eskf.raw().p[17][17],
    ];

    eskf.fuse_body_vel_scaled(0.01, 0.0, 0.0);

    let n = &eskf.raw().nominal;
    assert_eq!([n.qcs0, n.qcs1, n.qcs2, n.qcs3], qcs_before);
    assert_eq!(
        [
            eskf.raw().p[15][15],
            eskf.raw().p[16][16],
            eskf.raw().p[17][17],
        ],
        mount_cov_before
    );
    assert_eq!(eskf.raw().update_diag.sum_abs_dx_mount_yaw[4], 0.0);
    assert_eq!(eskf.raw().update_diag.sum_abs_dx_mount_yaw[5], 0.0);
}

#[test]
fn unknown_direction_uses_predicted_sign_when_state_is_confident() {
    let mut system = SensorFusion::with_misalignment([1.0, 0.0, 0.0, 0.0]);
    let mut gnss = gnss_sample(1.0);
    gnss.vel_ned_mps = [-3.0, 0.0, 0.0];
    gnss.heading_rad = Some(core::f32::consts::PI);
    let upd = system.process_gnss(gnss);
    assert!(upd.ekf_initialized_now);
    let vn_before = system.eskf().unwrap().nominal.vn;
    let _ = system.process_vehicle_speed(FusionVehicleSpeedSample {
        t_s: 1.1,
        speed_mps: 4.0,
        direction: FusionVehicleSpeedDirection::Unknown,
    });
    let vn_after = system.eskf().unwrap().nominal.vn;
    assert!(vn_after < vn_before);
    assert!(vn_after > -4.0);
}

#[test]
fn internal_alignment_bootstraps_mount_estimate() {
    let mut system = SensorFusion::new();
    let _ = system.process_gnss(stationary_gnss_sample(0.0));
    for i in 0..120 {
        let t_s = 0.01 * i as f32;
        let _ = system.process_imu(FusionImuSample {
            t_s,
            gyro_radps: [0.0, 0.0, 0.0],
            accel_mps2: [0.0, 0.0, -9.80665],
        });
    }
    let _ = system.process_gnss(stationary_gnss_sample(1.2));
    assert!(system.mount_q_vb().is_some());
    assert!(system.align().is_some());
}
