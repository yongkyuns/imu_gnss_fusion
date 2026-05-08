use sensor_fusion::ProcessNoise;
use sensor_fusion::reduced::Filter;
use sensor_fusion::reduced::GnssSample;

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
