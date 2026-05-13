use sensor_fusion::ProcessNoise;
use sensor_fusion::reduced::Filter;
use sensor_fusion::reduced::GnssSample;
use sensor_fusion::reduced::ImuDelta;

#[test]
fn gnss_nhc_batch_diag_reports_total_mount_correction() {
    let mut reduced = Filter::new(ProcessNoise::default());
    let raw = reduced.raw_mut();
    for i in 0..18 {
        for j in 0..18 {
            raw.p[i][j] = 0.0;
        }
        raw.p[i][i] = 1.0;
    }
    raw.p[3][15] = 0.2;
    raw.p[15][3] = 0.2;

    reduced.fuse_gps_nhc_batch(
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
        reduced.raw().update_diag.last_dx_mount_roll.abs() > 0.01,
        "batch-level diagnostic should preserve the nonzero GNSS velocity mount correction"
    );
}

#[test]
fn gnss_batch_records_row_diag_once_and_clears_on_predict() {
    let mut reduced = Filter::new(ProcessNoise::default());
    reduced.fuse_gps_nhc_batch(
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

    let raw = reduced.raw();
    assert_eq!(raw.last_obs_count, 6);
    assert_eq!(&raw.last_obs_types[..6], &[0, 0, 8, 1, 1, 9]);
    assert!(
        raw.last_dx_by_obs[..6]
            .iter()
            .any(|row| row.iter().any(|v| *v != 0.0)),
        "per-row diagnostics should expose nonzero GNSS contributions"
    );

    reduced.predict(ImuDelta {
        dax: 0.0,
        day: 0.0,
        daz: 0.0,
        dvx: 0.0,
        dvy: 0.0,
        dvz: 0.0,
        dt: 0.01,
    });

    assert_eq!(reduced.raw().last_obs_count, 0);
}
