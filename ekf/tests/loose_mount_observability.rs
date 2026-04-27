use sensor_fusion::loose::{LOOSE_ERROR_STATES, LooseFilter, LoosePredictNoise};

fn initialized_loose(vel_ecef_mps: [f32; 3]) -> LooseFilter {
    let mut p_diag = [1.0e-4; LOOSE_ERROR_STATES];
    p_diag[3] = 1.0;
    p_diag[4] = 1.0;
    p_diag[5] = 1.0;
    p_diag[21] = 0.25;
    p_diag[22] = 0.25;
    p_diag[23] = 0.25;

    let mut loose = LooseFilter::new(LoosePredictNoise::reference_nsr_demo());
    loose.init_from_reference_state(
        [1.0, 0.0, 0.0, 0.0],
        [6_378_137.0, 0.0, 0.0],
        vel_ecef_mps,
        [0.0; 3],
        [0.0; 3],
        [1.0; 3],
        [1.0; 3],
        [1.0, 0.0, 0.0, 0.0],
        Some(p_diag),
    );
    loose
}

#[test]
fn weak_nhc_direct_mount_sensitivity_does_not_move_mount() {
    let mut loose = initialized_loose([0.01, 0.005, 0.0]);

    loose.fuse_nhc_reference([0.0; 3], [0.0, 0.0, 9.81], 0.02);

    assert_eq!(loose.last_dx()[21], 0.0);
    assert_eq!(loose.last_dx()[22], 0.0);
    assert_eq!(loose.last_dx()[23], 0.0);
}

#[test]
fn moving_nhc_with_direct_mount_sensitivity_can_still_move_mount() {
    let mut loose = initialized_loose([5.0, 1.0, 0.0]);

    loose.fuse_nhc_reference([0.0; 3], [0.0, 0.0, 9.81], 0.02);

    let dx_mount = &loose.last_dx()[21..24];
    let max_abs = dx_mount
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    assert!(
        max_abs > 1.0e-6,
        "expected observable moving NHC update to move mount, dx={dx_mount:?}"
    );
}
