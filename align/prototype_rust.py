from __future__ import annotations

import numpy as np

import align_rs

from prototype import (
    CoarseAlignConfig,
    TrialHistory,
    TrialResult,
    simulate_vehicle_and_sensors,
    wrap_angle_rad,
)


def _as_tuple3(v: np.ndarray) -> tuple[float, float, float]:
    return float(v[0]), float(v[1]), float(v[2])


def _run_rust_filter(
    truth_mount_deg: np.ndarray,
    seed: int,
    repeat_count: int,
    yaw_seed_deg: float,
    cfg: CoarseAlignConfig,
) -> tuple[float, TrialResult, TrialHistory]:
    stationary_accel, windows = simulate_vehicle_and_sensors(
        truth_mount_deg=truth_mount_deg,
        seed=seed,
        repeat_count=repeat_count,
    )
    ekf = align_rs.CoarseAlignMEKF(
        use_gravity=cfg.use_gravity,
        use_turn_gyro=cfg.use_turn_gyro,
        use_course_rate=cfg.use_course_rate,
        use_lateral_accel=cfg.use_lateral_accel,
        use_longitudinal_accel=cfg.use_longitudinal_accel,
    )
    ekf.initialize_from_stationary(
        [_as_tuple3(v) for v in stationary_accel],
        yaw_seed_deg=yaw_seed_deg,
    )

    times = []
    estimates = []
    sigmas = []
    score = 0.0
    for window in windows[1:]:
        score += ekf.update_window(
            dt=float(window.dt),
            mean_gyro_b=_as_tuple3(window.mean_gyro_b),
            mean_accel_b=_as_tuple3(window.mean_accel_b),
            gnss_vel_prev_n=_as_tuple3(window.gnss_vel_prev_n),
            gnss_vel_curr_n=_as_tuple3(window.gnss_vel_curr_n),
        )
        times.append(window.t_end_s)
        estimates.append(np.asarray(ekf.mount_angles_deg()))
        sigmas.append(np.asarray(ekf.sigma_deg()))

    est_deg = np.asarray(ekf.mount_angles_deg())
    truth_deg = truth_mount_deg.astype(float)
    err_deg = est_deg - truth_deg
    err_deg[2] = np.rad2deg(wrap_angle_rad(np.deg2rad(err_deg[2])))
    return (
        TrialResult(
            label=f"rust score={score:.3f}",
            truth_deg=truth_deg,
            est_deg=est_deg,
            err_deg=err_deg,
        ),
        TrialHistory(
            t_s=np.asarray(times),
            est_deg=np.asarray(estimates),
            sigma_deg=np.asarray(sigmas),
        ),
    )


def run_rust_trial_with_history(
    truth_mount_deg: np.ndarray,
    seed: int,
    repeat_count: int = 1,
    yaw_seed_deg: float = 0.0,
    cfg: CoarseAlignConfig | None = None,
) -> tuple[TrialResult, TrialHistory]:
    cfg = cfg or CoarseAlignConfig()
    result, history = _run_rust_filter(
        truth_mount_deg=truth_mount_deg,
        seed=seed,
        repeat_count=repeat_count,
        yaw_seed_deg=yaw_seed_deg,
        cfg=cfg,
    )
    return result, history


def run_representative_sweep(seed: int = 3) -> None:
    rolls = [-25.0, 0.0, 25.0]
    pitches = [-20.0, 0.0, 20.0]
    yaws = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0]
    errs = []
    worst = None
    for roll in rolls:
        for pitch in pitches:
            for yaw in yaws:
                truth = np.array([roll, pitch, yaw], dtype=float)
                result, _history = run_rust_trial_with_history(truth, seed=seed)
                errs.append(np.abs(result.err_deg))
                if worst is None or np.max(np.abs(result.err_deg)) > np.max(np.abs(worst.err_deg)):
                    worst = result
    errs = np.asarray(errs)
    print("Rust MEKF representative sweep")
    print("  cases:", errs.shape[0])
    print("  mean abs err [deg]:", np.round(np.mean(errs, axis=0), 3))
    print("  rmse [deg]:", np.round(np.sqrt(np.mean(errs**2, axis=0)), 3))
    print("  max abs err [deg]:", np.round(np.max(errs, axis=0), 3))
    if worst is not None:
        print("  worst case truth [deg]:", np.round(worst.truth_deg, 3))
        print("  worst case err   [deg]:", np.round(worst.err_deg, 3))


def run_large_angle_checks(seed: int = 3) -> None:
    truths = [
        np.array([35.0, -30.0, 150.0]),
        np.array([-35.0, 30.0, -150.0]),
        np.array([40.0, -30.0, 170.0]),
        np.array([45.0, -35.0, 120.0]),
        np.array([-45.0, 35.0, -120.0]),
        np.array([60.0, -45.0, 90.0]),
        np.array([-60.0, 45.0, -90.0]),
        np.array([70.0, -50.0, 135.0]),
        np.array([-70.0, 50.0, -135.0]),
        np.array([75.0, -55.0, 170.0]),
        np.array([-75.0, 55.0, -170.0]),
        np.array([80.0, -60.0, 179.0]),
        np.array([-80.0, 60.0, -179.0]),
    ]
    print("Rust MEKF large-angle checks")
    for truth in truths:
        result, history = run_rust_trial_with_history(truth, seed=seed, repeat_count=4)
        tail = max(20, len(history.t_s) // 5)
        tail_err = history.est_deg[-tail:] - result.truth_deg
        tail_err[:, 2] = [np.rad2deg(wrap_angle_rad(np.deg2rad(v))) for v in tail_err[:, 2]]
        print("  truth [deg]:", np.round(truth, 3))
        print("    final err [deg]:", np.round(result.err_deg, 3))
        print("    tail mean abs err [deg]:", np.round(np.mean(np.abs(tail_err), axis=0), 3))
        print("    final sigma [deg]:", np.round(history.sigma_deg[-1], 3))


if __name__ == "__main__":
    run_representative_sweep()
    run_large_angle_checks()
