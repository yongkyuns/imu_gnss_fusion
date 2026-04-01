use sensor_fusion::fusion::{FusionConfig, FusionGnssSample, FusionImuSample, SensorFusion};

fn gnss_sample(t_s: f32) -> FusionGnssSample {
    FusionGnssSample {
        t_s,
        pos_ned_m: [0.0, 0.0, 0.0],
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
    let mut system =
        SensorFusion::with_misalignment(FusionConfig::default(), [1.0, 0.0, 0.0, 0.0]);
    let upd = system.process_gnss(gnss_sample(1.0));
    assert!(upd.mount_ready);
    assert!(upd.ekf_initialized_now);
    assert!(system.ekf().is_some());
}

#[test]
fn internal_alignment_bootstraps_mount_estimate() {
    let mut system = SensorFusion::new(FusionConfig::default());
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
