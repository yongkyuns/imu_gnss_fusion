use sensor_fusion::ProcessNoise;
use sensor_fusion::{
    Config, Filter, GnssSample, ImuSample, MountMode, SensorFusion, VehicleSpeedDirection,
    VehicleSpeedSample,
};

fn gnss_sample(t_s: f32) -> GnssSample {
    GnssSample {
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

fn stationary_gnss_sample(t_s: f32) -> GnssSample {
    GnssSample {
        vel_ned_mps: [0.0, 0.0, 0.0],
        heading_rad: None,
        ..gnss_sample(t_s)
    }
}

#[test]
fn manual_mount_initializes_reduced_from_gnss_and_freezes_mount_states() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    let upd = system.process_gnss(gnss_sample(1.0));
    assert!(upd.mount_ready);
    assert!(upd.reduced_initialized_now);
    let reduced = system.reduced().unwrap();
    for i in 15..18 {
        for j in 0..reduced.p.len() {
            assert_eq!(reduced.p[i][j], 0.0);
            assert_eq!(reduced.p[j][i], 0.0);
        }
    }
}

#[test]
fn public_config_selects_filter_and_mount_mode() {
    let system = SensorFusion::with_config(Config {
        filter: Filter::Full,
        mount_mode: MountMode::Manual([1.0, 0.0, 0.0, 0.0]),
    });

    assert_eq!(system.filter(), Filter::Full);
}

#[test]
fn full_filter_initializes_through_public_sensor_fusion_api() {
    let mut system = SensorFusion::with_config(Config {
        filter: Filter::Full,
        mount_mode: MountMode::Manual([1.0, 0.0, 0.0, 0.0]),
    });

    let update = system.process_gnss(gnss_sample(1.0));

    assert!(update.filter_initialized);
    assert!(update.filter_initialized_now);
    let full = system.full().unwrap();
    for i in 21..24 {
        for j in 0..full.p.len() {
            assert_eq!(full.p[i][j], 0.0);
            assert_eq!(full.p[j][i], 0.0);
        }
    }
    assert!(system.reduced().is_none());
}

#[test]
fn vehicle_speed_sample_pulls_forward_velocity_upward() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    let upd = system.process_gnss(gnss_sample(1.0));
    assert!(upd.reduced_initialized_now);
    let vn_before = system.reduced().unwrap().nominal.vn;
    let _ = system.process_vehicle_speed(VehicleSpeedSample {
        t_s: 1.1,
        speed_mps: 6.0,
        direction: VehicleSpeedDirection::Forward,
    });
    let vn_after = system.reduced().unwrap().nominal.vn;
    assert!(vn_after > vn_before);
    assert!(vn_after < 6.0);
}

#[test]
fn freeze_misalignment_states_blocks_mount_updates() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    system.set_freeze_misalignment_states(true);
    system.set_r_vehicle_speed(0.001);

    let upd = system.process_gnss(gnss_sample(1.0));
    assert!(upd.reduced_initialized_now);
    let qcs_before = {
        let reduced = system.reduced().unwrap();
        [
            reduced.nominal.qcs0,
            reduced.nominal.qcs1,
            reduced.nominal.qcs2,
            reduced.nominal.qcs3,
        ]
    };

    let _ = system.process_vehicle_speed(VehicleSpeedSample {
        t_s: 1.1,
        speed_mps: 7.0,
        direction: VehicleSpeedDirection::Forward,
    });

    let reduced = system.reduced().unwrap();
    let qcs_after = [
        reduced.nominal.qcs0,
        reduced.nominal.qcs1,
        reduced.nominal.qcs2,
        reduced.nominal.qcs3,
    ];
    assert_eq!(qcs_after, qcs_before);
    assert_eq!(reduced.update_diag.last_dx_mount_yaw, 0.0);
    assert_eq!(reduced.update_diag.last_k_mount_yaw, 0.0);
    for i in 15..18 {
        for j in 0..18 {
            assert_eq!(reduced.p[i][j], 0.0);
            assert_eq!(reduced.p[j][i], 0.0);
        }
    }
}

#[test]
fn manual_mount_settle_phase_keeps_mount_states_frozen() {
    let mut system = SensorFusion::with_mount([1.0, 0.0, 0.0, 0.0]);
    system.set_mount_settle_time_s(1.0);
    system.set_mount_settle_release_sigma_rad(4.0_f32.to_radians());

    let upd = system.process_gnss(gnss_sample(1.0));
    assert!(upd.reduced_initialized_now);
    for i in 15..18 {
        assert_eq!(system.reduced().unwrap().p[i][i], 0.0);
    }

    let _ = system.process_gnss(gnss_sample(1.5));
    for i in 15..18 {
        assert_eq!(system.reduced().unwrap().p[i][i], 0.0);
    }

    let _ = system.process_gnss(gnss_sample(2.2));
    let reduced = system.reduced().unwrap();
    for i in 15..18 {
        assert_eq!(reduced.p[i][i], 0.0);
    }
    for i in 15..18 {
        for j in 0..18 {
            assert_eq!(reduced.p[i][j], 0.0);
            assert_eq!(reduced.p[j][i], 0.0);
        }
    }
}

#[test]
fn zero_velocity_update_does_not_inject_mount_error() {
    const DIAG_ZERO_VEL: usize = 2;
    const DIAG_ZERO_VEL_D: usize = 10;

    let mut reduced = sensor_fusion::reduced::Filter::new(ProcessNoise::default());
    {
        let raw = reduced.raw_mut();
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

    let qcs_before = {
        let n = &reduced.raw().nominal;
        [n.qcs0, n.qcs1, n.qcs2, n.qcs3]
    };
    let mount_cov_before = [
        reduced.raw().p[15][15],
        reduced.raw().p[16][16],
        reduced.raw().p[17][17],
    ];

    reduced.fuse_zero_vel(0.01);

    let raw = reduced.raw();
    let n = &raw.nominal;
    assert_ne!(n.vn, 0.7);
    assert_ne!(n.ve, -0.3);
    assert_ne!(n.vd, 0.2);
    assert_eq!([n.qcs0, n.qcs1, n.qcs2, n.qcs3], qcs_before);
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
    gnss.vel_ned_mps = [-3.0, 0.0, 0.0];
    gnss.heading_rad = Some(core::f32::consts::PI);
    let upd = system.process_gnss(gnss);
    assert!(upd.reduced_initialized_now);
    let vn_before = system.reduced().unwrap().nominal.vn;
    let _ = system.process_vehicle_speed(VehicleSpeedSample {
        t_s: 1.1,
        speed_mps: 4.0,
        direction: VehicleSpeedDirection::Unknown,
    });
    let vn_after = system.reduced().unwrap().nominal.vn;
    assert!(vn_after < vn_before);
    assert!(vn_after > -4.0);
}

#[test]
fn internal_alignment_bootstraps_mount_estimate() {
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
