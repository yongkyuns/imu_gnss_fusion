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

    loose.fuse_nhc_reference_with_speed(Some(5.0), [0.0; 3], [0.0, 0.0, 9.81], 0.02);

    let dx_mount = &loose.last_dx()[21..24];
    let max_abs = dx_mount
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));
    assert!(
        max_abs > 1.0e-6,
        "expected observable moving NHC update to move mount, dx={dx_mount:?}"
    );
}

#[test]
fn batched_nhc_uses_reference_velocity_speed_by_default() {
    let mut loose = initialized_loose([5.0, 1.0, 0.0]);

    loose.fuse_reference_batch_full(
        None,
        Some([5.0, 1.0, 0.0]),
        1.0,
        None,
        0.02,
        [0.0; 3],
        [0.0, 0.0, 9.81],
        0.02,
    );

    assert!(loose.last_obs_types().contains(&7));
    assert!(loose.last_obs_types().contains(&8));
}

#[test]
fn batched_nhc_without_reference_speed_does_not_fall_back_to_state_speed() {
    let mut loose = initialized_loose([5.0, 1.0, 0.0]);

    loose.fuse_reference_batch_full(
        None,
        None,
        1.0,
        None,
        0.02,
        [0.0; 3],
        [0.0, 0.0, 9.81],
        0.02,
    );

    assert!(loose.last_obs_types().is_empty());
    assert_eq!(loose.last_dx()[21], 0.0);
    assert_eq!(loose.last_dx()[22], 0.0);
    assert_eq!(loose.last_dx()[23], 0.0);
}

#[test]
fn external_zero_speed_suppresses_batched_nhc_even_when_state_speed_is_nonzero() {
    let mut loose = initialized_loose([5.0, 1.0, 0.0]);

    loose.fuse_reference_batch_full_with_nhc_speed(
        None,
        None,
        1.0,
        None,
        0.02,
        Some(0.0),
        [0.0; 3],
        [0.0, 0.0, 9.81],
        0.02,
    );

    assert!(loose.last_obs_types().is_empty());
    assert_eq!(loose.last_dx()[21], 0.0);
    assert_eq!(loose.last_dx()[22], 0.0);
    assert_eq!(loose.last_dx()[23], 0.0);
}

#[test]
fn external_moving_speed_allows_batched_nhc() {
    let mut loose = initialized_loose([5.0, 1.0, 0.0]);

    loose.fuse_reference_batch_full_with_nhc_speed(
        None,
        None,
        1.0,
        None,
        0.02,
        Some(5.0),
        [0.0; 3],
        [0.0, 0.0, 9.81],
        0.02,
    );

    assert!(loose.last_obs_types().contains(&7));
    assert!(loose.last_obs_types().contains(&8));
}
