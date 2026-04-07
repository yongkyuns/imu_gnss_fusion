use sensor_fusion::fusion::{
    FusionGnssSample, FusionImuSample, FusionVehicleSpeedDirection, FusionVehicleSpeedSample,
    SensorFusion,
};

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
