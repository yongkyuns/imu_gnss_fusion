use sensor_fusion::ProcessNoise;
use sensor_fusion::ekf::Filter;
use sensor_fusion::ekf::GnssSample;
use sensor_fusion::ekf::ImuDelta;
use sensor_fusion::{
    GNSS_EVENT_POSITION_ACCURACY_BYPASS, GNSS_EVENT_POSITION_CONSECUTIVE_REJECTED,
    GNSS_EVENT_POSITION_GAP_BYPASS, GNSS_EVENT_POSITION_REJECTED, GNSS_EVENT_VELOCITY_REJECTED,
};

#[test]
fn gnss_nhc_batch_diag_reports_total_mount_correction() {
    let mut ekf = Filter::new(ProcessNoise::default());
    let raw = ekf.raw_mut();
    for i in 0..18 {
        for j in 0..18 {
            raw.p[i][j] = 0.0;
        }
        raw.p[i][i] = 1.0;
    }
    raw.p[3][15] = 0.2;
    raw.p[15][3] = 0.2;

    let result = ekf.fuse_gps_nhc_batch(
        GnssSample {
            t_s: 1.0,
            pos_ned_m: [0.0; 3],
            vel_ned_mps: [1.0, 0.0, 0.0],
            pos_std_m: [1.0; 3],
            vel_std_mps: [1.0; 3],
            heading_rad: None,
        },
        None,
        None,
    );

    assert!(
        ekf.raw().update_diag.last_dx_mount_roll.abs() > 0.01,
        "batch-level diagnostic should preserve the nonzero GNSS velocity mount correction"
    );
    assert_eq!(result.event_mask, 0);
}

#[test]
fn gnss_batch_records_row_diag_once_and_clears_on_predict() {
    let mut ekf = Filter::new(ProcessNoise::default());
    let result = ekf.fuse_gps_nhc_batch(
        GnssSample {
            t_s: 1.0,
            pos_ned_m: [1.0, -2.0, 0.5],
            vel_ned_mps: [0.4, -0.2, 0.1],
            pos_std_m: [1.0; 3],
            vel_std_mps: [1.0; 3],
            heading_rad: None,
        },
        None,
        None,
    );

    assert_eq!(result.event_mask, 0);
    let raw = ekf.raw();
    assert_eq!(raw.last_obs_count, 6);
    assert_eq!(&raw.last_obs_types[..6], &[0, 0, 8, 1, 1, 9]);
    assert!(
        raw.last_dx_by_obs[..6]
            .iter()
            .any(|row| row.iter().any(|v| *v != 0.0)),
        "per-row diagnostics should expose nonzero GNSS contributions"
    );

    ekf.predict(ImuDelta {
        dax: 0.0,
        day: 0.0,
        daz: 0.0,
        dvx: 0.0,
        dvy: 0.0,
        dvz: 0.0,
        dt: 0.01,
    });

    assert_eq!(ekf.raw().last_obs_count, 0);
}

#[test]
fn gnss_position_per_axis_large_jump_rejection_keeps_velocity_update() {
    let mut ekf = Filter::new(ProcessNoise::default());
    let result = ekf.fuse_gps_nhc_batch(
        GnssSample {
            t_s: 1.0,
            pos_ned_m: [100.0, 0.0, 0.0],
            vel_ned_mps: [0.4, -0.2, 0.1],
            pos_std_m: [1.0; 3],
            vel_std_mps: [1.0; 3],
            heading_rad: None,
        },
        None,
        None,
    );

    assert_eq!(result.event_mask, GNSS_EVENT_POSITION_REJECTED);
    let raw = ekf.raw();
    assert_eq!(raw.last_obs_count, 3);
    assert_eq!(&raw.last_obs_types[..3], &[1, 1, 9]);
    assert_eq!(raw.update_diag.type_counts[0], 0);
    assert_eq!(raw.update_diag.type_counts[8], 0);
    assert_eq!(raw.update_diag.type_counts[1], 2);
    assert_eq!(raw.update_diag.type_counts[9], 1);
}

#[test]
fn gnss_velocity_per_axis_large_jump_rejection_keeps_position_update() {
    let mut ekf = Filter::new(ProcessNoise::default());
    let result = ekf.fuse_gps_nhc_batch(
        GnssSample {
            t_s: 1.0,
            pos_ned_m: [1.0, -2.0, 0.5],
            vel_ned_mps: [100.0, 0.0, 0.0],
            pos_std_m: [1.0; 3],
            vel_std_mps: [1.0; 3],
            heading_rad: None,
        },
        None,
        None,
    );

    assert_eq!(result.event_mask, GNSS_EVENT_VELOCITY_REJECTED);
    let raw = ekf.raw();
    assert_eq!(raw.last_obs_count, 3);
    assert_eq!(&raw.last_obs_types[..3], &[0, 0, 8]);
    assert_eq!(raw.update_diag.type_counts[0], 2);
    assert_eq!(raw.update_diag.type_counts[8], 1);
    assert_eq!(raw.update_diag.type_counts[1], 0);
    assert_eq!(raw.update_diag.type_counts[9], 0);
}

#[test]
fn gnss_position_consecutive_rejections_emit_event_without_fusing_jump() {
    let mut ekf = Filter::new(ProcessNoise::default());
    let sample = GnssSample {
        t_s: 1.0,
        pos_ned_m: [100.0, 0.0, 0.0],
        vel_ned_mps: [0.0; 3],
        pos_std_m: [1.0; 3],
        vel_std_mps: [1.0; 3],
        heading_rad: None,
    };

    assert_eq!(
        ekf.fuse_gps_nhc_batch(sample, None, None).event_mask,
        GNSS_EVENT_POSITION_REJECTED
    );
    assert_eq!(
        ekf.fuse_gps_nhc_batch(GnssSample { t_s: 2.0, ..sample }, None, None)
            .event_mask,
        GNSS_EVENT_POSITION_REJECTED
    );
    let result = ekf.fuse_gps_nhc_batch(GnssSample { t_s: 3.0, ..sample }, None, None);

    assert_eq!(
        result.event_mask,
        GNSS_EVENT_POSITION_REJECTED | GNSS_EVENT_POSITION_CONSECUTIVE_REJECTED
    );
    assert_eq!(&ekf.raw().last_obs_types[..3], &[1, 1, 9]);
}

#[test]
fn gnss_position_gap_bypass_accepts_first_update_after_large_gap() {
    let mut ekf = Filter::new(ProcessNoise::default());
    let sample = GnssSample {
        t_s: 1.0,
        pos_ned_m: [100.0, 0.0, 0.0],
        vel_ned_mps: [0.0; 3],
        pos_std_m: [1.0; 3],
        vel_std_mps: [1.0; 3],
        heading_rad: None,
    };

    assert_eq!(
        ekf.fuse_gps_nhc_batch(sample, None, None).event_mask,
        GNSS_EVENT_POSITION_REJECTED
    );
    let result = ekf.fuse_gps_nhc_batch(GnssSample { t_s: 4.2, ..sample }, None, None);

    assert_eq!(result.event_mask, GNSS_EVENT_POSITION_GAP_BYPASS);
    assert_eq!(&ekf.raw().last_obs_types[..6], &[0, 0, 8, 1, 1, 9]);
}

#[test]
fn gnss_position_accuracy_bypass_accepts_after_reported_accuracy_improves() {
    let mut ekf = Filter::new(ProcessNoise::default());
    assert_eq!(
        ekf.fuse_gps_nhc_batch(
            GnssSample {
                t_s: 1.0,
                pos_ned_m: [1000.0, 0.0, 0.0],
                vel_ned_mps: [0.0; 3],
                pos_std_m: [10.0; 3],
                vel_std_mps: [1.0; 3],
                heading_rad: None,
            },
            None,
            None,
        )
        .event_mask,
        GNSS_EVENT_POSITION_REJECTED
    );

    let result = ekf.fuse_gps_nhc_batch(
        GnssSample {
            t_s: 2.0,
            pos_ned_m: [1000.0, 0.0, 0.0],
            vel_ned_mps: [0.0; 3],
            pos_std_m: [1.0; 3],
            vel_std_mps: [1.0; 3],
            heading_rad: None,
        },
        None,
        None,
    );

    assert_eq!(result.event_mask, GNSS_EVENT_POSITION_ACCURACY_BYPASS);
    assert_eq!(&ekf.raw().last_obs_types[..6], &[0, 0, 8, 1, 1, 9]);
}
