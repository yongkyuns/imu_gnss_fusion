from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GRAVITY_MPS2 = 9.80665


def wrap_angle_rad(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def euler_zyx_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_from_small_angle(dtheta: np.ndarray) -> np.ndarray:
    half = 0.5 * np.asarray(dtheta, dtype=float)
    return quat_normalize(np.array([1.0, half[0], half[1], half[2]]))


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def quat_from_rot(C: np.ndarray) -> np.ndarray:
    tr = np.trace(C)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array(
            [
                0.25 * s,
                (C[2, 1] - C[1, 2]) / s,
                (C[0, 2] - C[2, 0]) / s,
                (C[1, 0] - C[0, 1]) / s,
            ]
        )
    elif C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:
        s = np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2.0
        q = np.array(
            [
                (C[2, 1] - C[1, 2]) / s,
                0.25 * s,
                (C[0, 1] + C[1, 0]) / s,
                (C[0, 2] + C[2, 0]) / s,
            ]
        )
    elif C[1, 1] > C[2, 2]:
        s = np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2]) * 2.0
        q = np.array(
            [
                (C[0, 2] - C[2, 0]) / s,
                (C[0, 1] + C[1, 0]) / s,
                0.25 * s,
                (C[1, 2] + C[2, 1]) / s,
            ]
        )
    else:
        s = np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1]) * 2.0
        q = np.array(
            [
                (C[1, 0] - C[0, 1]) / s,
                (C[0, 2] + C[2, 0]) / s,
                (C[1, 2] + C[2, 1]) / s,
                0.25 * s,
            ]
        )
    return quat_normalize(q)


def vehicle_to_body_rotation(mount_angles: np.ndarray) -> np.ndarray:
    roll_m, pitch_m, yaw_m = mount_angles
    return euler_zyx_to_rot(roll_m, pitch_m, yaw_m)


def body_to_vehicle_rotation(mount_angles: np.ndarray) -> np.ndarray:
    return vehicle_to_body_rotation(mount_angles).T


def rot_to_euler_zyx(C: np.ndarray) -> np.ndarray:
    pitch = np.arcsin(np.clip(-C[2, 0], -1.0, 1.0))
    roll = np.arctan2(C[2, 1], C[2, 2])
    yaw = np.arctan2(C[1, 0], C[0, 0])
    return np.array([roll, pitch, wrap_angle_rad(yaw)])


def numerical_jacobian(func, x: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    y0 = func(x)
    H = np.zeros((y0.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        yp = func(x + dx)
        ym = func(x - dx)
        H[:, i] = (yp - ym) / (2.0 * eps)
    return H


@dataclass
class AlignConfig:
    q_mount_std_rad: np.ndarray = np.deg2rad(np.array([0.01, 0.01, 0.02]))
    r_gravity_std_mps2: float = 0.08
    r_turn_gyro_std_radps: float = np.deg2rad(0.2)
    r_course_rate_std_radps: float = np.deg2rad(0.35)
    r_lat_std_mps2: float = 0.10
    r_long_std_mps2: float = 0.10
    gravity_lpf_alpha: float = 0.08
    min_speed_mps: float = 4.0
    min_turn_rate_radps: float = np.deg2rad(3.0)
    min_lat_acc_mps2: float = 0.35
    min_long_acc_mps2: float = 0.25
    max_stationary_gyro_radps: float = np.deg2rad(0.8)
    max_stationary_accel_norm_err_mps2: float = 0.2
    use_gravity: bool = True
    use_turn_gyro: bool = True
    use_course_rate: bool = True
    use_lateral_accel: bool = True
    use_longitudinal_accel: bool = True


@dataclass
class WindowSummary:
    t_end_s: float
    dt: float
    mean_gyro_b: np.ndarray
    mean_accel_b: np.ndarray
    gnss_vel_prev_n: np.ndarray
    gnss_vel_curr_n: np.ndarray


@dataclass
class TrialResult:
    label: str
    truth_deg: np.ndarray
    est_deg: np.ndarray
    err_deg: np.ndarray


@dataclass
class TrialHistory:
    t_s: np.ndarray
    est_deg: np.ndarray
    sigma_deg: np.ndarray


class AlignEKF:
    def __init__(self, cfg: AlignConfig):
        self.cfg = cfg
        self.q_vb = np.array([1.0, 0.0, 0.0, 0.0])
        self.P = np.diag(np.deg2rad([20.0, 20.0, 60.0]) ** 2)
        self.gravity_lp_b = np.array([0.0, 0.0, -GRAVITY_MPS2])

    def initialize_from_stationary(
        self, accel_samples_b: np.ndarray, yaw_seed_rad: float = 0.0
    ) -> None:
        f_mean_b = np.mean(accel_samples_b, axis=0)
        n = np.linalg.norm(f_mean_b)
        if n < 1.0e-6:
            raise ValueError("stationary initialization requires nonzero accel mean")

        z_v_in_b = -f_mean_b / n
        x_ref = np.array([1.0, 0.0, 0.0])
        x_v_in_b = x_ref - z_v_in_b * np.dot(z_v_in_b, x_ref)
        if np.linalg.norm(x_v_in_b) < 1.0e-6:
            x_ref = np.array([0.0, 1.0, 0.0])
            x_v_in_b = x_ref - z_v_in_b * np.dot(z_v_in_b, x_ref)
        x_v_in_b /= np.linalg.norm(x_v_in_b)
        y_v_in_b = np.cross(z_v_in_b, x_v_in_b)
        y_v_in_b /= np.linalg.norm(y_v_in_b)
        x_v_in_b = np.cross(y_v_in_b, z_v_in_b)
        C_v_b = np.column_stack((x_v_in_b, y_v_in_b, z_v_in_b))
        init_rpy = rot_to_euler_zyx(C_v_b)
        init_rpy[2] = wrap_angle_rad(yaw_seed_rad)
        self.q_vb = quat_from_rot(euler_zyx_to_rot(*init_rpy))
        self.P = np.diag(np.deg2rad([6.0, 6.0, 20.0]) ** 2)
        self.gravity_lp_b = f_mean_b.copy()

    def predict(self, dt: float) -> None:
        self.P = self.P + np.diag(self.cfg.q_mount_std_rad**2) * max(dt, 1.0e-3)

    def _body_to_vehicle_rotation_from_quat(self, q_vb: np.ndarray) -> np.ndarray:
        return quat_to_rot(q_vb).T

    def _candidate_vehicle_specific_force(
        self, q_vb: np.ndarray, accel_b: np.ndarray
    ) -> np.ndarray:
        return self._body_to_vehicle_rotation_from_quat(q_vb) @ accel_b

    def _candidate_vehicle_gyro(self, q_vb: np.ndarray, gyro_b: np.ndarray) -> np.ndarray:
        return self._body_to_vehicle_rotation_from_quat(q_vb) @ gyro_b

    def _numerical_attitude_jacobian(self, func) -> np.ndarray:
        y0 = func(self.q_vb)
        H = np.zeros((y0.size, 3))
        eps = 1.0e-6
        for i in range(3):
            dtheta = np.zeros(3)
            dtheta[i] = eps
            qp = quat_mul(quat_from_small_angle(dtheta), self.q_vb)
            qm = quat_mul(quat_from_small_angle(-dtheta), self.q_vb)
            yp = func(quat_normalize(qp))
            ym = func(quat_normalize(qm))
            H[:, i] = (yp - ym) / (2.0 * eps)
        return H

    def _ekf_update(self, z: np.ndarray, func, R_diag: np.ndarray) -> float:
        h = func(self.q_vb)
        H = self._numerical_attitude_jacobian(func)
        y = z - h
        S = H @ self.P @ H.T + np.diag(R_diag)
        S_inv = np.linalg.inv(S)
        K = self.P @ H.T @ S_inv
        score = float(y.T @ S_inv @ y)
        dtheta = K @ y
        self.q_vb = quat_normalize(quat_mul(quat_from_small_angle(dtheta), self.q_vb))
        self.P = (np.eye(3) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return score

    def update_window(self, window: WindowSummary) -> float:
        self.predict(window.dt)
        score = 0.0

        alpha = self.cfg.gravity_lpf_alpha
        self.gravity_lp_b = (
            (1.0 - alpha) * self.gravity_lp_b + alpha * window.mean_accel_b
        )

        v_prev = window.gnss_vel_prev_n
        v_curr = window.gnss_vel_curr_n
        speed_prev = np.linalg.norm(v_prev[:2])
        speed_curr = np.linalg.norm(v_curr[:2])
        speed_mid = 0.5 * (speed_prev + speed_curr)

        course_prev = np.arctan2(v_prev[1], v_prev[0])
        course_curr = np.arctan2(v_curr[1], v_curr[0])
        course_rate = wrap_angle_rad(course_curr - course_prev) / max(window.dt, 1.0e-3)

        a_n = (v_curr - v_prev) / max(window.dt, 1.0e-3)
        v_mid_h = 0.5 * (v_prev[:2] + v_curr[:2])
        t_hat = None
        if np.linalg.norm(v_mid_h) > 1.0e-6:
            t_hat = v_mid_h / np.linalg.norm(v_mid_h)
        lat_hat = None if t_hat is None else np.array([-t_hat[1], t_hat[0]])
        a_long = 0.0 if t_hat is None else float(t_hat @ a_n[:2])
        a_lat = 0.0 if lat_hat is None else float(lat_hat @ a_n[:2])

        gyro_norm = np.linalg.norm(window.mean_gyro_b)
        accel_norm = np.linalg.norm(window.mean_accel_b)
        stationary = (
            gyro_norm <= self.cfg.max_stationary_gyro_radps
            and abs(accel_norm - GRAVITY_MPS2)
            <= self.cfg.max_stationary_accel_norm_err_mps2
            and speed_mid < 0.5
        )
        turn_valid = (
            speed_mid > self.cfg.min_speed_mps
            and abs(course_rate) > self.cfg.min_turn_rate_radps
            and abs(a_lat) > self.cfg.min_lat_acc_mps2
        )
        long_valid = (
            speed_mid > self.cfg.min_speed_mps
            and abs(a_long) > self.cfg.min_long_acc_mps2
            and abs(a_lat) < max(0.5, 0.6 * abs(a_long))
        )

        if self.cfg.use_gravity and stationary:
                z = np.array([0.0, 0.0])
                score += self._ekf_update(
                    z,
                    lambda q: self._candidate_vehicle_specific_force(q, self.gravity_lp_b)[:2],
                    np.full(2, self.cfg.r_gravity_std_mps2**2),
                )

        if turn_valid:
            if self.cfg.use_turn_gyro:
                z = np.array([0.0, 0.0])
                score += self._ekf_update(
                    z,
                    lambda q: self._candidate_vehicle_gyro(q, window.mean_gyro_b)[:2],
                    np.full(2, self.cfg.r_turn_gyro_std_radps**2),
                )
            if self.cfg.use_course_rate:
                z = np.array([course_rate])
                score += self._ekf_update(
                    z,
                    lambda q: self._candidate_vehicle_gyro(q, window.mean_gyro_b)[2:3],
                    np.array([self.cfg.r_course_rate_std_radps**2]),
                )
            if self.cfg.use_lateral_accel:
                z = np.array([a_lat])
                score += self._ekf_update(
                    z,
                    lambda q: self._candidate_vehicle_specific_force(
                        q, window.mean_accel_b
                    )[1:2],
                    np.array([self.cfg.r_lat_std_mps2**2]),
                )

        if long_valid and self.cfg.use_longitudinal_accel:
            z = np.array([a_long])
            score += self._ekf_update(
                z,
                lambda q: self._candidate_vehicle_specific_force(
                    q, window.mean_accel_b
                )[:1],
                np.array([self.cfg.r_long_std_mps2**2]),
            )
        return score

    @property
    def mount_angles_rad(self) -> np.ndarray:
        return rot_to_euler_zyx(quat_to_rot(self.q_vb))

    @property
    def mount_angles_deg(self) -> np.ndarray:
        return np.rad2deg(self.mount_angles_rad)


@dataclass
class Segment:
    duration_s: float
    a_long_mps2: float
    yaw_rate_radps: float


def simulate_vehicle_and_sensors(
    truth_mount_deg: np.ndarray,
    imu_rate_hz: float = 100.0,
    gnss_rate_hz: float = 10.0,
    gyro_noise_std_radps: float = np.deg2rad(0.03),
    accel_noise_std_mps2: float = 0.04,
    gnss_vel_noise_std_mps: float = 0.03,
    seed: int = 0,
    repeat_count: int = 1,
) -> tuple[np.ndarray, list[WindowSummary]]:
    rng = np.random.default_rng(seed)
    truth_mount_rad = np.deg2rad(truth_mount_deg)
    C_v_b_true = vehicle_to_body_rotation(truth_mount_rad)

    base_segments = [
        Segment(8.0, 0.9, 0.0),
        Segment(4.0, 0.0, 0.0),
        Segment(8.0, 0.0, np.deg2rad(10.0)),
        Segment(3.0, -0.7, 0.0),
        Segment(8.0, 0.0, np.deg2rad(-12.0)),
        Segment(4.0, 0.8, 0.0),
        Segment(8.0, 0.0, np.deg2rad(9.0)),
        Segment(5.0, -0.6, 0.0),
    ]
    segments = [Segment(6.0, 0.0, 0.0)] + base_segments * max(repeat_count, 1)

    dt_imu = 1.0 / imu_rate_hz
    steps_per_gnss = int(round(imu_rate_hz / gnss_rate_hz))
    if steps_per_gnss <= 0:
        raise ValueError("GNSS rate must be lower than IMU rate")

    v_n = np.zeros(3)
    yaw = 0.0
    speed = 0.0
    accel_window = []
    gyro_window = []
    stationary_accel = []
    windows: list[WindowSummary] = []
    gnss_prev_v = np.zeros(3)
    t_s = 0.0

    for seg in segments:
        n_steps = int(round(seg.duration_s / dt_imu))
        for _ in range(n_steps):
            t_s += dt_imu
            speed = max(0.0, speed + seg.a_long_mps2 * dt_imu)
            yaw = wrap_angle_rad(yaw + seg.yaw_rate_radps * dt_imu)

            a_long = seg.a_long_mps2
            a_lat = speed * seg.yaw_rate_radps

            v_n = np.array([speed * np.cos(yaw), speed * np.sin(yaw), 0.0])
            f_v = np.array([a_long, a_lat, -GRAVITY_MPS2])
            omega_v = np.array([0.0, 0.0, seg.yaw_rate_radps])

            accel_b = C_v_b_true @ f_v
            gyro_b = C_v_b_true @ omega_v

            accel_b = accel_b + rng.normal(scale=accel_noise_std_mps2, size=3)
            gyro_b = gyro_b + rng.normal(scale=gyro_noise_std_radps, size=3)

            accel_window.append(accel_b)
            gyro_window.append(gyro_b)

            if speed < 0.1 and abs(seg.yaw_rate_radps) < 1.0e-4 and abs(a_long) < 1.0e-4:
                stationary_accel.append(accel_b)

            if len(accel_window) == steps_per_gnss:
                gnss_v = v_n + rng.normal(scale=gnss_vel_noise_std_mps, size=3)
                windows.append(
                    WindowSummary(
                        t_end_s=t_s,
                        dt=steps_per_gnss * dt_imu,
                        mean_gyro_b=np.mean(np.asarray(gyro_window), axis=0),
                        mean_accel_b=np.mean(np.asarray(accel_window), axis=0),
                        gnss_vel_prev_n=gnss_prev_v.copy(),
                        gnss_vel_curr_n=gnss_v.copy(),
                    )
                )
                gnss_prev_v = gnss_v.copy()
                accel_window.clear()
                gyro_window.clear()

    stationary_accel_arr = np.asarray(stationary_accel)
    if stationary_accel_arr.shape[0] < 20:
        raise RuntimeError("synthetic scenario did not produce enough stationary data")
    return stationary_accel_arr, windows


def run_trial_with_history(
    truth_mount_deg: np.ndarray,
    cfg: AlignConfig,
    seed: int,
    repeat_count: int = 1,
    yaw_seed_deg: float = 0.0,
) -> tuple[TrialResult, TrialHistory]:
    stationary_accel, windows = simulate_vehicle_and_sensors(
        truth_mount_deg=truth_mount_deg,
        seed=seed,
        repeat_count=repeat_count,
    )
    ekf = AlignEKF(cfg)
    ekf.initialize_from_stationary(stationary_accel, np.deg2rad(yaw_seed_deg))
    times = []
    estimates = []
    sigmas = []
    for window in windows[1:]:
        ekf.update_window(window)
        times.append(window.t_end_s)
        estimates.append(ekf.mount_angles_deg.copy())
        sigmas.append(np.rad2deg(np.sqrt(np.maximum(np.diag(ekf.P), 0.0))))

    est_deg = ekf.mount_angles_deg
    truth_deg = truth_mount_deg.astype(float)
    err_deg = est_deg - truth_deg
    err_deg[2] = np.rad2deg(wrap_angle_rad(np.deg2rad(err_deg[2])))
    return (
        TrialResult(
            label="",
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


def run_trial(
    label: str,
    truth_mount_deg: np.ndarray,
    cfg: AlignConfig,
    seed: int,
    repeat_count: int = 1,
    yaw_seed_deg: float = 0.0,
) -> TrialResult:
    result, _history = run_trial_with_history(
        truth_mount_deg, cfg, seed, repeat_count=repeat_count, yaw_seed_deg=yaw_seed_deg
    )
    result.label = label
    return result


def summarize_results(results: list[TrialResult]) -> None:
    for result in results:
        print(result.label)
        print(f"  truth [deg] : {np.round(result.truth_deg, 3)}")
        print(f"  est   [deg] : {np.round(result.est_deg, 3)}")
        print(f"  err   [deg] : {np.round(result.err_deg, 3)}")


def monte_carlo_summary(
    label: str,
    truth_mount_deg: np.ndarray,
    cfg: AlignConfig,
    seeds: list[int],
) -> np.ndarray:
    errs = []
    for seed in seeds:
        result = run_trial(label, truth_mount_deg, cfg, seed)
        errs.append(np.abs(result.err_deg))
    errs = np.asarray(errs)
    mean_err = np.mean(errs, axis=0)
    max_err = np.max(errs, axis=0)
    print(label + " Monte Carlo")
    print(f"  mean abs err [deg] : {np.round(mean_err, 3)}")
    print(f"  max  abs err [deg] : {np.round(max_err, 3)}")
    return mean_err


def summarize_window_inputs(
    windows: list[WindowSummary],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_s = np.array([w.t_end_s for w in windows[1:]])
    gyro_dps = np.array([np.rad2deg(w.mean_gyro_b) for w in windows[1:]])
    accel = np.array([w.mean_accel_b for w in windows[1:]])
    speed = []
    course_rate_dps = []
    long_acc = []
    lat_acc = []
    for w in windows[1:]:
        v_prev = w.gnss_vel_prev_n
        v_curr = w.gnss_vel_curr_n
        v_mid_h = 0.5 * (v_prev[:2] + v_curr[:2])
        speed.append(float(np.linalg.norm(v_curr[:2])))
        course_prev = np.arctan2(v_prev[1], v_prev[0])
        course_curr = np.arctan2(v_curr[1], v_curr[0])
        course_rate_dps.append(np.rad2deg(wrap_angle_rad(course_curr - course_prev) / max(w.dt, 1.0e-3)))
        if np.linalg.norm(v_mid_h) > 1.0e-6:
            t_hat = v_mid_h / np.linalg.norm(v_mid_h)
            lat_hat = np.array([-t_hat[1], t_hat[0]])
            a_n = (v_curr - v_prev) / max(w.dt, 1.0e-3)
            long_acc.append(float(t_hat @ a_n[:2]))
            lat_acc.append(float(lat_hat @ a_n[:2]))
        else:
            long_acc.append(0.0)
            lat_acc.append(0.0)
    gnss = np.column_stack((np.asarray(speed), np.asarray(course_rate_dps), np.asarray(long_acc), np.asarray(lat_acc)))
    return t_s, gyro_dps, accel, gnss[:, 0], gnss[:, 1:]


def make_demo_plots(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_mount_deg = np.array([4.0, -3.0, 28.0])
    full_cfg = AlignConfig()
    weak_cfg = AlignConfig(
        use_gravity=True,
        use_turn_gyro=True,
        use_course_rate=True,
        use_lateral_accel=False,
        use_longitudinal_accel=False,
    )

    stationary_accel, windows = simulate_vehicle_and_sensors(
        truth_mount_deg=truth_mount_deg,
        seed=3,
    )
    _ = stationary_accel
    weak_result, weak_hist = run_trial_with_history(truth_mount_deg, weak_cfg, seed=3)
    full_result, full_hist = run_trial_with_history(truth_mount_deg, full_cfg, seed=3)
    weak_result.label = "weak"
    full_result.label = "full"

    t_s, gyro_dps, accel, speed, gnss_derived = summarize_window_inputs(windows)

    input_fig = output_dir / "align_inputs.png"
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    axes[0].plot(t_s, gyro_dps[:, 0], label="gyro x")
    axes[0].plot(t_s, gyro_dps[:, 1], label="gyro y")
    axes[0].plot(t_s, gyro_dps[:, 2], label="gyro z")
    axes[0].set_ylabel("gyro [deg/s]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_s, accel[:, 0], label="accel x")
    axes[1].plot(t_s, accel[:, 1], label="accel y")
    axes[1].plot(t_s, accel[:, 2], label="accel z")
    axes[1].set_ylabel("specific force [m/s^2]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_s, speed, label="speed")
    axes[2].set_ylabel("speed [m/s]")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    course_rate_dps = gnss_derived[:, 0].copy()
    course_rate_dps[speed < 1.0] = np.nan
    axes[3].plot(t_s, course_rate_dps, label="course rate")
    axes[3].set_ylabel("course rate [deg/s]")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(t_s, gnss_derived[:, 1], label="a_long")
    axes[4].plot(t_s, gnss_derived[:, 2], label="a_lat")
    axes[4].set_ylabel("accel [m/s^2]")
    axes[4].set_xlabel("time [s]")
    axes[4].legend(loc="upper right")
    axes[4].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(input_fig, dpi=160)
    plt.close(fig)

    angle_fig = output_dir / "align_angles.png"
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    labels = ["roll", "pitch", "yaw"]
    for i, ax in enumerate(axes):
        ax.plot(
            full_hist.t_s,
            np.full_like(full_hist.t_s, truth_mount_deg[i], dtype=float),
            label=f"truth {labels[i]}",
            linestyle="--",
            linewidth=1.5,
        )
        ax.plot(weak_hist.t_s, weak_hist.est_deg[:, i], label=f"weak est {labels[i]}")
        ax.plot(full_hist.t_s, full_hist.est_deg[:, i], label=f"full est {labels[i]}")
        ax.set_ylabel(f"{labels[i]} [deg]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(angle_fig, dpi=160)
    plt.close(fig)
    return input_fig, angle_fig


def make_long_run_plot(output_dir: Path) -> tuple[Path, TrialResult, TrialHistory]:
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_mount_deg = np.array([4.0, -3.0, 28.0])
    result, history = run_trial_with_history(
        truth_mount_deg,
        AlignConfig(),
        seed=3,
        repeat_count=4,
    )
    plot_path = output_dir / "align_long_run.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    labels = ["roll", "pitch", "yaw"]
    for i, label in enumerate(labels):
        axes[0].plot(
            history.t_s,
            history.est_deg[:, i],
            label=f"est {label}",
        )
        axes[0].plot(
            history.t_s,
            np.full_like(history.t_s, truth_mount_deg[i], dtype=float),
            linestyle="--",
            linewidth=1.0,
            label=f"truth {label}",
        )
    axes[0].set_ylabel("angles [deg]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=2)

    for i, label in enumerate(labels):
        axes[1].plot(history.t_s, history.sigma_deg[:, i], label=f"sigma {label}")
    axes[1].set_ylabel("1-sigma [deg]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    result.label = "long-run full align filter"
    return plot_path, result, history


def run_demo() -> list[TrialResult]:
    truth_mount_deg = np.array([4.0, -3.0, 28.0])
    weak_cfg = AlignConfig(
        use_gravity=True,
        use_turn_gyro=True,
        use_course_rate=True,
        use_lateral_accel=False,
        use_longitudinal_accel=False,
    )
    full_cfg = AlignConfig()

    results = [
        run_trial("weak gyro/course-rate align filter", truth_mount_deg, weak_cfg, seed=3),
        run_trial("full gravity+gyro+lat/long align filter", truth_mount_deg, full_cfg, seed=3),
        run_trial(
            "full gravity+gyro+lat/long align filter (second seed)",
            truth_mount_deg,
            full_cfg,
            seed=8,
        ),
    ]
    summarize_results(results)
    seeds = [1, 2, 3, 4, 5, 8]
    weak_mean = monte_carlo_summary(
        "weak gyro/course-rate align filter", truth_mount_deg, weak_cfg, seeds
    )
    full_mean = monte_carlo_summary(
        "full gravity+gyro+lat/long align filter", truth_mount_deg, full_cfg, seeds
    )
    print("yaw improvement factor:", round(weak_mean[2] / full_mean[2], 3))
    return results


if __name__ == "__main__":
    demo_results = run_demo()
    input_fig, angle_fig = make_demo_plots(Path(__file__).with_name("plots"))
    long_fig, long_result, long_hist = make_long_run_plot(Path(__file__).with_name("plots"))
    print(f"saved input plot: {input_fig}")
    print(f"saved angle plot: {angle_fig}")
    print(f"saved long-run plot: {long_fig}")
    tail = max(20, len(long_hist.t_s) // 5)
    tail_err = long_hist.est_deg[-tail:] - long_result.truth_deg
    tail_err[:, 2] = [np.rad2deg(wrap_angle_rad(np.deg2rad(v))) for v in tail_err[:, 2]]
    tail_mean_abs_err = np.mean(np.abs(tail_err), axis=0)
    tail_sigma_span = np.max(long_hist.sigma_deg[-tail:], axis=0) - np.min(
        long_hist.sigma_deg[-tail:], axis=0
    )
    print("long-run tail mean abs err [deg]:", np.round(tail_mean_abs_err, 3))
    print("long-run final 1-sigma [deg]:", np.round(long_hist.sigma_deg[-1], 3))
    print("long-run tail sigma span [deg]:", np.round(tail_sigma_span, 3))
    weak = demo_results[0]
    full = demo_results[1]
    if abs(full.err_deg[0]) > 1.5 or abs(full.err_deg[1]) > 1.5 or abs(full.err_deg[2]) > 3.0:
        raise SystemExit("full align filter did not converge tightly enough")
    if abs(full.err_deg[2]) >= abs(weak.err_deg[2]):
        raise SystemExit("full align filter did not improve yaw over weak baseline")
