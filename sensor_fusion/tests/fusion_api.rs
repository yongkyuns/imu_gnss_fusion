use sensor_fusion::{FusionState, ProcessNoise};
use sensor_fusion::{
    GNSS_EVENT_POSITION_REJECTED, GnssSample, ImuSample, SensorFusion, VehicleSpeedDirection,
    VehicleSpeedSample,
};

fn gnss_sample(t_s: f32) -> GnssSample {
    GnssSample {
        t_s,
        lat_deg: 0.0,
        lon_deg: 0.0,
        height_m: 0.0,
        vel_ned_mps: [6.0, 0.0, 0.0],
        pos_std_m: [1.0, 1.0, 1.5],
        vel_std_mps: [0.2, 0.2, 0.2],
        heading_rad: Some(0.0),
    }
}

fn stationary_gnss_sample(t_s: f32) -> GnssSample {
    GnssSample {
        vel_ned_mps: [0.0, 0.0, 0.0],
        heading_rad: None,
        ..gnss_sample(t_s)
    }
}

#[test]
fn manual_mount_initializes_ekf_from_gnss_with_live_mount_prior() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    let upd = system.process_gnss(gnss_sample(1.0));
    assert!(upd.mount_ready);
    assert!(upd.navigation_started);
    let ekf = system.ekf().unwrap();
    let expected_var = (3.0_f32.to_radians()).powi(2);
    for i in 15..18 {
        assert_eq!(ekf.p[i][i], expected_var);
    }
}

#[test]
fn manual_mount_waits_for_yaw_seed_before_ekf_initialization() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);

    let stationary = system.process_gnss(stationary_gnss_sample(1.0));
    assert!(stationary.mount_ready);
    assert!(!stationary.navigation_usable);
    assert!(!stationary.navigation_started);
    assert!(system.ekf().is_none());

    let mut moving = gnss_sample(2.0);
    moving.heading_rad = None;
    moving.vel_ned_mps = [-6.0, 0.0, 0.0];
    let no_heading = system.process_gnss(moving);
    assert!(!no_heading.navigation_usable);
    assert!(!no_heading.navigation_started);

    moving.heading_rad = Some(core::f32::consts::PI);
    moving.vel_ned_mps = [-5.5, 0.0, 0.0];
    let below_speed = system.process_gnss(moving);
    assert!(!below_speed.navigation_usable);
    assert!(!below_speed.navigation_started);

    moving.vel_ned_mps = [-6.0, 0.0, 0.0];
    let initialized = system.process_gnss(moving);
    assert!(initialized.navigation_started);
    let ekf = system.ekf().unwrap();
    assert!(ekf.nominal.q3.abs() > 0.99);
}

#[test]
fn manual_mount_seed_is_normalized() {
    let system = SensorFusion::with_mount([2.0, 0.0, 0.0, 0.0]);
    assert_eq!(system.mount_q_bv(), Some([1.0, 0.0, 0.0, 0.0]));
}

#[test]
fn ekf_nhc_uses_estimated_motion_during_gnss_outage() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    system.set_nhc_update_period_s(0.0);

    let update = system.process_gnss(gnss_sample(1.0));
    assert!(update.navigation_started);

    let before = system.ekf().unwrap().update_diag.total_updates;
    let _ = system.process_imu(ImuSample {
        t_s: 2.20,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let _ = system.process_imu(ImuSample {
        t_s: 2.21,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });

    assert!(system.ekf().unwrap().update_diag.total_updates > before);
}

#[test]
fn gnss_gate_events_are_returned_when_pending_gnss_is_fused() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    let update = system.process_gnss(gnss_sample(1.0));
    assert!(update.navigation_started);

    let _ = system.process_imu(ImuSample {
        t_s: 1.01,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let queued = system.process_gnss(GnssSample {
        t_s: 1.02,
        lat_deg: 0.01,
        ..gnss_sample(1.02)
    });
    assert_eq!(queued.gnss_event_mask, 0);

    let fused = system.process_imu(ImuSample {
        t_s: 1.03,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });

    assert_eq!(
        fused.gnss_event_mask & GNSS_EVENT_POSITION_REJECTED,
        GNSS_EVENT_POSITION_REJECTED
    );
}

#[test]
fn short_sleep_keeps_navigation_running() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    assert!(system.process_gnss(gnss_sample(1.0)).navigation_started);
    let _ = system.process_imu(ImuSample {
        t_s: 1.01,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let _ = system.process_imu(ImuSample {
        t_s: 1.02,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let _ = system.end_trip();

    let slept = system.process_imu(ImuSample {
        t_s: 600.0,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });

    assert!(slept.navigation_usable);
    assert_eq!(system.health().state, FusionState::Running);
    let ekf = system.ekf().unwrap();
    let pos_h_sigma_m = ekf.p[6][6].max(ekf.p[7][7]).sqrt();
    let vel_h_sigma_mps = ekf.p[3][3].max(ekf.p[4][4]).sqrt();
    assert!(
        pos_h_sigma_m < 5.0,
        "stationary short sleep should not add unbounded position uncertainty"
    );
    assert!(
        vel_h_sigma_mps < 1.0,
        "stationary short sleep should preserve usable velocity confidence"
    );
}

#[test]
fn unexpected_data_gap_reseeds_navigation_without_stale_yaw() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    assert!(system.process_gnss(gnss_sample(1.0)).navigation_started);
    let _ = system.process_imu(ImuSample {
        t_s: 1.01,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let _ = system.process_imu(ImuSample {
        t_s: 1.02,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });

    let _ = system.process_imu(ImuSample {
        t_s: 180.0,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let reseeded = system.process_gnss(GnssSample {
        t_s: 180.1,
        lat_deg: 0.001,
        vel_ned_mps: [0.02, 0.0, 0.0],
        heading_rad: Some(core::f32::consts::PI),
        ..gnss_sample(180.1)
    });

    assert!(reseeded.navigation_started);
    let ekf = system.ekf().unwrap();
    assert!(
        ekf.nominal.pn > 100.0,
        "post-gap GNSS position should reacquire instead of being rejected"
    );
    assert!(
        ekf.nominal.q3.abs() < 0.5,
        "stale low-speed GNSS heading must not flip yaw during reacquisition"
    );
    assert!(
        ekf.p[2][2].sqrt().to_degrees() >= 40.0,
        "yaw should remain uncertain when the gap contradicted stationary sleep"
    );
}

#[test]
fn declared_trip_sleep_rejects_large_gnss_position_jump() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    assert!(system.process_gnss(gnss_sample(1.0)).navigation_started);
    let _ = system.process_imu(ImuSample {
        t_s: 1.01,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let _ = system.process_imu(ImuSample {
        t_s: 1.02,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let _ = system.end_trip();

    let slept = system.process_imu(ImuSample {
        t_s: 180.0,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    assert!(slept.navigation_usable);

    let _ = system.process_gnss(GnssSample {
        t_s: 180.03,
        lat_deg: 0.001,
        vel_ned_mps: [0.02, 0.0, 0.0],
        heading_rad: Some(core::f32::consts::PI),
        ..gnss_sample(180.03)
    });
    let fused = system.process_imu(ImuSample {
        t_s: 180.04,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });

    assert_eq!(
        fused.gnss_event_mask & GNSS_EVENT_POSITION_REJECTED,
        GNSS_EVENT_POSITION_REJECTED
    );
    assert!(
        system.ekf().unwrap().nominal.pn < 50.0,
        "declared stationary sleep should not accept a large reported-accurate GNSS jump"
    );
}

#[test]
fn medium_sleep_enters_degraded_dead_reckoning_until_gnss_recovers() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    assert!(system.process_gnss(gnss_sample(1.0)).navigation_started);
    let _ = system.process_imu(ImuSample {
        t_s: 1.01,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let _ = system.process_imu(ImuSample {
        t_s: 1.02,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let _ = system.end_trip();

    let _ = system.process_imu(ImuSample {
        t_s: 1200.0,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let health = system.health();
    assert_eq!(health.state, FusionState::DegradedDeadReckoning);
    assert!(health.degraded);
    assert!(health.navigation_usable);
    let ekf = system.ekf().unwrap();
    let pos_h_sigma_m = ekf.p[6][6].max(ekf.p[7][7]).sqrt();
    let yaw_sigma_deg = ekf.p[2][2].sqrt().to_degrees();
    assert!(
        pos_h_sigma_m < 10.0,
        "stationary medium sleep should stay within bounded position aging"
    );
    assert!(
        yaw_sigma_deg < 10.0,
        "stationary medium sleep should not quickly destroy yaw confidence"
    );

    let _ = system.process_gnss(gnss_sample(1200.01));
    let _ = system.process_imu(ImuSample {
        t_s: 1200.02,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    assert_eq!(system.health().state, FusionState::Running);
}

#[test]
fn medium_sleep_with_unusable_covariance_waits_for_gnss_reseed() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    assert!(system.process_gnss(gnss_sample(1.0)).navigation_started);
    let _ = system.process_imu(ImuSample {
        t_s: 1.01,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let _ = system.process_imu(ImuSample {
        t_s: 1.02,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let _ = system.end_trip();

    system.analysis_set_ekf_attitude_roll_pitch_covariance(10.0_f32.to_radians());
    let slept = system.process_imu(ImuSample {
        t_s: 1200.0,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });

    assert!(!slept.navigation_usable);
    assert_eq!(system.health().state, FusionState::AwaitingGnssReseed);
    assert!(system.ekf().is_none());

    let reseeded = system.process_gnss(gnss_sample(1200.1));
    assert!(reseeded.navigation_usable);
    assert!(reseeded.navigation_started);
    assert_eq!(system.health().state, FusionState::Running);
}

#[test]
fn long_sleep_makes_navigation_unusable_until_gnss_reseed() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    assert!(system.process_gnss(gnss_sample(1.0)).navigation_started);
    let _ = system.process_imu(ImuSample {
        t_s: 1.01,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let _ = system.end_trip();

    let slept = system.process_imu(ImuSample {
        t_s: 4000.0,
        gyro_radps: [0.0, 0.0, 0.0],
        accel_mps2: [0.0, 0.0, 9.80665],
    });
    let health = system.health();
    assert_eq!(health.state, FusionState::AwaitingGnssReseed);
    assert!(!health.navigation_usable);
    assert!(!slept.navigation_usable);
    assert!(system.ekf().is_none());

    let reseeded = system.process_gnss(gnss_sample(4000.1));
    assert!(reseeded.navigation_usable);
    assert!(reseeded.navigation_started);
    assert_eq!(system.health().state, FusionState::Running);
    assert!(system.ekf().is_some());
}

#[test]
fn vehicle_speed_sample_pulls_forward_velocity_upward() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    let upd = system.process_gnss(gnss_sample(1.0));
    assert!(upd.navigation_started);
    let vn_before = system.ekf().unwrap().nominal.vn;
    let _ = system.process_vehicle_speed(VehicleSpeedSample {
        t_s: 1.1,
        speed_mps: 7.0,
        direction: VehicleSpeedDirection::Forward,
    });
    let vn_after = system.ekf().unwrap().nominal.vn;
    assert!(vn_after > vn_before);
    assert!(vn_after < 7.0);
}

#[test]
fn zero_velocity_update_does_not_inject_mount_error() {
    const DIAG_ZERO_VEL: usize = 2;
    const DIAG_ZERO_VEL_D: usize = 8;

    let mut ekf = sensor_fusion::ekf::Filter::new(ProcessNoise::default());
    {
        let raw = ekf.raw_mut();
        raw.nominal.vn = 0.7;
        raw.nominal.ve = -0.3;
        raw.nominal.vd = 0.2;
        raw.p = [[0.0; 18]; 18];
        for i in 0..18 {
            raw.p[i][i] = 0.1;
        }
        raw.p[3][15] = 0.02;
        raw.p[15][3] = 0.02;
        raw.p[4][16] = -0.015;
        raw.p[16][4] = -0.015;
        raw.p[5][17] = 0.01;
        raw.p[17][5] = 0.01;
    }

    let q_bv_before = {
        let n = &ekf.raw().nominal;
        [n.q_bv0, n.q_bv1, n.q_bv2, n.q_bv3]
    };
    let mount_cov_before = [
        ekf.raw().p[15][15],
        ekf.raw().p[16][16],
        ekf.raw().p[17][17],
    ];

    ekf.fuse_zero_vel(0.01);

    let raw = ekf.raw();
    let n = &raw.nominal;
    assert_ne!(n.vn, 0.7);
    assert_ne!(n.ve, -0.3);
    assert_ne!(n.vd, 0.2);
    assert_eq!([n.q_bv0, n.q_bv1, n.q_bv2, n.q_bv3], q_bv_before);
    assert_eq!(
        [raw.p[15][15], raw.p[16][16], raw.p[17][17]],
        mount_cov_before
    );
    assert_eq!(raw.update_diag.sum_abs_dx_mount_norm[DIAG_ZERO_VEL], 0.0);
    assert_eq!(raw.update_diag.sum_abs_dx_mount_norm[DIAG_ZERO_VEL_D], 0.0);
    assert_eq!(raw.update_diag.last_dx_mount_roll, 0.0);
    assert_eq!(raw.update_diag.last_dx_mount_pitch, 0.0);
    assert_eq!(raw.update_diag.last_dx_mount_yaw, 0.0);
}

#[test]
fn unknown_direction_uses_predicted_sign_when_state_is_confident() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    let mut gnss = gnss_sample(1.0);
    gnss.vel_ned_mps = [-6.0, 0.0, 0.0];
    gnss.heading_rad = Some(core::f32::consts::PI);
    let upd = system.process_gnss(gnss);
    assert!(upd.navigation_started);
    let vn_before = system.ekf().unwrap().nominal.vn;
    let _ = system.process_vehicle_speed(VehicleSpeedSample {
        t_s: 1.1,
        speed_mps: 7.0,
        direction: VehicleSpeedDirection::Unknown,
    });
    let vn_after = system.ekf().unwrap().nominal.vn;
    assert!(vn_after < vn_before);
    assert!(vn_after > -7.0);
}

#[test]
fn internal_alignment_tilt_initializes_mount_estimate() {
    let mut system = SensorFusion::new();
    let _ = system.process_gnss(stationary_gnss_sample(0.0));
    for i in 0..120 {
        let t_s = 0.01 * i as f32;
        let _ = system.process_imu(ImuSample {
            t_s,
            gyro_radps: [0.0, 0.0, 0.0],
            accel_mps2: [0.0, 0.0, -9.80665],
        });
    }
    let _ = system.process_gnss(stationary_gnss_sample(1.2));
    assert!(system.mount_q_bv().is_some());
    assert!(system.align().is_some());
}
