import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from numpy import arctan2
from rls import RLS
from enum import Enum, auto


def get_stationary_period(imu_df, g_threshold):
    gyro_magnitude = np.sqrt(
        imu_df["gyro_x"] ** 2 + imu_df["gyro_y"] ** 2 + imu_df["gyro_z"] ** 2
    )
    accel_magnitude = np.sqrt(
        imu_df["accel_x"] ** 2 + imu_df["accel_y"] ** 2 + imu_df["accel_z"] ** 2
    )
    g = 9.81
    stationary_mask = np.abs(accel_magnitude - g) < g_threshold
    return stationary_mask


def estimate_roll_pitch_angles(accel_x, accel_y, accel_z):
    roll = np.arctan2(accel_y, accel_z)
    pitch = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2))
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    return roll_deg, pitch_deg


class Logger:
    def __init__(self):
        self.history_rp = {
            "timestamps": [],
            "roll_deg": [],
            "pitch_deg": [],
            "cov_trace": [],
            "instantaneous_roll_deg": [],
            "instantaneous_pitch_deg": [],
            "update_vectors": [],
        }
        self.history_raw = {
            "timestamps": [],
            "yaw_deg": [],
            "cov_trace": [],
            "instantaneous_yaw_deg": [],
            "constributing_vectors": [],
        }
        self.acc_xy_for_yaw_plot = {"x": [], "y": []}

    def log_rp_data(
        self,
        timestamp_sec,
        rls_roll_deg,
        rls_pitch_deg,
        rls_cov_trace,
        inst_roll_deg,
        inst_pitch_deg,
        update_vector,
    ):
        self.history_rp["timestamps"].append(timestamp_sec)
        self.history_rp["roll_deg"].append(rls_roll_deg)
        self.history_rp["pitch_deg"].append(rls_pitch_deg)
        self.history_rp["cov_trace"].append(rls_cov_trace)
        self.history_rp["instantaneous_roll_deg"].append(inst_roll_deg)
        self.history_rp["instantaneous_pitch_deg"].append(inst_pitch_deg)
        self.history_rp["update_vectors"].append(update_vector)

    def log_yaw_data(
        self,
        timestamp_sec,
        rls_yaw_deg,
        rls_cov_trace,
        inst_yaw_deg,
        constributing_vector,
    ):
        self.history_raw["timestamps"].append(timestamp_sec)
        self.history_raw["yaw_deg"].append(rls_yaw_deg)
        self.history_raw["cov_trace"].append(rls_cov_trace)
        self.history_raw["instantaneous_yaw_deg"].append(inst_yaw_deg)
        self.history_raw["constributing_vectors"].append(constributing_vector)

    def log_acc_xy_for_yaw_plot(self, acc_x, acc_y):
        self.acc_xy_for_yaw_plot["x"].append(acc_x)
        self.acc_xy_for_yaw_plot["y"].append(acc_y)


class AngleEstimator:
    class EstimationPhase(Enum):
        NOT_CALIBRATED = 0
        RP_CONVERGED = auto()
        YAW_CONVERGED = auto()
        ALL_CONVERGED = auto()

    def __init__(
        self,
        rls_rp_config,
        rls_yaw_config,
        rp_avg_duration_sec=3.0,
        speed_threshold_mps_staionary=5 / 9,
        accel_magnitude_threshold_stationary_ms2=(1.1 * 9.81),
        speed_threshold_mps_dynamic=2.0,
        dynamic_accel_threshold_for_yaw_ms2=2.0,
        dynamic_gyro_mag_threshold_rads=0.01,
        garvity_ms2=9.81,
        convergence_cov_trace_threshold=0.35,
        gps_acceleration_threshold_for_yaw_ms2=0.0,
    ):
        self.rls_rp_config = rls_rp_config
        self.rls_yaw_config = rls_yaw_config

        self.rp_avg_duration_sec = np.float32(rp_avg_duration_sec)
        self.speed_threshold_mps_staionary = np.float32(speed_threshold_mps_staionary)
        self.accel_magnitude_threshold_stationary_ms2 = np.float32(
            accel_magnitude_threshold_stationary_ms2
        )
        self.speed_threshold_mps_dynamic = np.float32(speed_threshold_mps_dynamic)
        self.dynamic_accel_threshold_for_yaw_ms2 = np.float32(
            dynamic_accel_threshold_for_yaw_ms2
        )
        self.dynamic_gyro_mag_threshold_rads = np.float32(
            dynamic_gyro_mag_threshold_rads
        )
        self.garvity_ms2 = np.float32(garvity_ms2)
        self.convergence_cov_trace_threshold = np.float32(
            convergence_cov_trace_threshold
        )
        self.gps_acceleration_threshold_for_yaw_ms2 = np.float32(
            gps_acceleration_threshold_for_yaw_ms2
        )

        self.rls_rp = None
        self.rls_yaw = None

        self.roll_deg = np.float32(0.0)
        self.pitch_deg = np.float32(0.0)
        self.yaw_deg = np.float32(0.0)

        self.estimation_phase = self.EstimationPhase.NOT_CALIBRATED
        self.rp_converged = False
        self.yaw_converged = False

        self._collecting_for_rp_avg = False
        self._rp_collection_start_time_current_stop = np.float32(0.0)
        self._rp_accel_sum_current_stop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._rp_accel_count_current_stop = 0
        self._stop_processed_for_rp_update = False

        self.current_speed_mps = np.float32(0.0)
        self.current_gps_acceleration_ms2 = np.float32(0.0)

        self._internal_prev_gps_speed_mps = None
        self._internal_prev_gps_timestamp_sec = None

        self.logger = Logger()

    def _is_stationary(self, current_accel_magnitude_ms2):
        is_speed_low = self.current_speed_mps < self.speed_threshold_mps_staionary
        is_accel_low = (
            current_accel_magnitude_ms2 < self.accel_magnitude_threshold_stationary_ms2
        )
        return is_speed_low and is_accel_low

    def update_speed(self, timestamp_sec, speed_mps):
        self.current_speed_mps = np.float32(speed_mps)

        if (
            self._internal_prev_gps_timestamp_sec is not None
            and self._internal_prev_gps_speed_mps is not None
        ):
            delta_t = np.float32(timestamp_sec - self._internal_prev_gps_timestamp_sec)
            if delta_t > 1e-6:
                delta_v = self.current_speed_mps - self._interanl_prev_gps_speed_mps
                self.current_gps_acceleration_ms2 = np.float32(delta_v / delta_t)
            else:
                self.current_gps_acceleration_ms2 = np.float32(0.0)

            self._internal_prev_gps_timestamp_sec = timestamp_sec
            self._internal_prev_gps_speed_mps = self.current_speed_mps

    def update_estimates(self, timestamp_sec, imu_data):
        acc_x = np.float32(imu_data["accel_x"])
        acc_y = np.float32(imu_data["accel_y"])
        acc_z = np.float32(imu_data["accel_z"])
        gyro_x = np.float32(imu_data["gyro_x"])
        gyro_y = np.float32(imu_data["gyro_y"])
        gyro_z = np.float32(imu_data["gyro_z"])

        current_accel_magnitude_ms2 = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        is_stopped_now = self._is_stationary(np.float32(current_accel_magnitude_ms2))

        if is_stopped_now:
            if (
                not self._collecting_for_rp_avg
                and not self._stop_processed_for_rp_update
            ):
                self._collecting_for_rp_avg = True
                self._rp_collection_start_time_current_stop = np.float32(timestamp_sec)
                self._rp_accel_sum_current_stop = np.array(
                    acc_x, acc_y, acc_z, dtype=np.float32
                )
                self._rp_accel_count_current_stop = 1
            elif self._collecting_for_rp_avg:
                if (
                    timestamp_sec - self._rp_collection_start_time_current_stop
                ) < self.rp_avg_duration_sec:
                    self._rp_accel_sum_current_stop += np.array(
                        [acc_x, acc_y, acc_z], dtype=np.float32
                    )
                else:
                    if self._rp_accel_count_current_stop > 0:
                        mean_measurement_accel_rp = (
                            self._rp_accel_sum_current_stop
                            / self._rp_accel_count_current_stop
                        )
                        mean_measurement_accel_rp = mean_measurement_accel_rp.astype(
                            np.float32
                        )
                        _inst_roll, _inst_pitch = estimate_roll_pitch_angles(
                            mean_measurement_accel_rp[0],
                            mean_measurement_accel_rp[1],
                            mean_measurement_accel_rp[2],
                        )
                        inst_roll = np.float32(_inst_roll)
                        inst_pitch = np.float32(_inst_pitch)

                        if self.rls_rp is None:
                            initial_state_rp = mean_measurement_accel_rp.copy()
                            self.rls_rp = RLS(
                                initial_state=initial_state_rp,
                                initial_covariance_diagonal=self.rls_rp_config[
                                    "initial_cov_diag"
                                ],
                                measurement_uncertainty_R_scalar=self.rls_rp_config[
                                    "measurement_uncertainty_R_scalar"
                                ],
                                forgetting_factor=self.rls_rp_config[
                                    "forgetting_factor"
                                ],
                            )
                        else:
                            self.rls_rp.update(mean_measurement_accel_rp)
                        _rls_roll, _rls_pitch = estimate_roll_pitch_angles(
                            self.rls_rp.x[0], self.rls_rp.x[1], self.rls_rp.x[2]
                        )
                        self.roll_deg = np.float32(_rls_roll)
                        self.pitch_deg = np.float32(_rls_pitch)

                        if (
                            np.trace(self.rls_rp.P)
                            < self.convergence_cov_trace_threshold
                        ):
                            self.rp_converged = True

                        self.logger.log_rp_data(
                            timestamp_sec=timestamp_sec,
                            rls_roll_deg=self.roll_deg,
                            rls_pitch_deg=self.pitch_deg,
                            rls_cov_trace=np.trace(self.rls_rp.P),
                            inst_roll_deg=inst_roll,
                            inst_pitch_deg=inst_pitch,
                            update_vector=mean_measurement_accel_rp.copy(),
                        )
                    self._collecting_for_rp_avg = False
                    self._stop_processed_for_rp_update = True
        else:
            if self._collecting_for_rp_avg:
                self._collecting_for_rp_avg = False
                self._rp_accel_sum_current_stop = np.array(
                    [0.0, 0.0, 0.0], dtype=np.float32
                )
                self._rp_accel_count_current_stop = 0
            self._stop_processed_for_rp_update = False

        gyro_mag_rads = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

        is_dynamic_and_ready_for_yaw = (
            self.current_speed_mps > self.speed_threshold_mps_dynamic
            and gyro_mag_rads < self.dynamic_gyro_mag_threshold_rads
            and self.rls_rp is not None
        )

        is_accelerating_via_gps = (
            self.current_gps_acceleration_ms2
            < self.gps_acceleration_threshold_for_yaw_ms2
        )

        if is_dynamic_and_ready_for_yaw and is_accelerating_via_gps:
            roll_rad_scipy = float(np.deg2rad(self.roll_deg))
            pitch_rad_scipy = float(np.deg2rad(self.pitch_deg))
            R_rp_mat = (
                R_scipy.from_euler("xyz", [roll_rad_scipy, pitch_rad_scipy, 0.0])
                .as_matrix()
                .astype(np.float32)
            )

            acc_sensor_frame = np.array([acc_x, acc_y, acc_z], dtype=np.float32)
            acc_vehicle_rp_corrected = R_rp_mat.T @ acc_sensor_frame

            horizontal_accel_magnitude_ms2 = np.float32(
                np.sqrt(
                    acc_vehicle_rp_corrected[0] ** 2 + acc_vehicle_rp_corrected[1] ** 2
                )
            )

            if (
                horizontal_accel_magnitude_ms2
                > self.dynamic_accel_threshold_for_yaw_ms2
            ):
                measurement_yaw_rls = acc_vehicle_rp_corrected

                inst_yaw = np.float32(
                    np.rad2deg(-arctan2(measurement_yaw_rls[1], measurement_yaw_rls[0]))
                )
                if inst_yaw < 0:
                    inst_yaw += 360.0
                if self.rls_yaw is None:
                    initial_state_yaw = measurement_yaw_rls.copy()
                    self.rls_yaw = RLS(
                        initial_state=initial_state_yaw,
                        initial_covariance_diagonal=self.rls_yaw_config[
                            "initial_cov_diag"
                        ],
                        measurement_uncertainty_R_scalar=self.rls_yaw_config[
                            "measurement_uncertainty_R_scalar"
                        ],
                        forgetting_factor=self.rls_yaw_config["forgetting_factor"],
                    )
                else:
                    self.rls_yaw.update(measurement_yaw_rls)

                if self.rls_yaw is not None and self.rls_yaw.x is not None:
                    self.yaw_deg = np.float32(
                        np.rad2deg(-arctan2(self.rls_yaw.x[1], self.rls_yaw.x[0]))
                    )
                    if self.yaw_deg < 0:
                        self.yaw_deg += 360.0

                    if np.trace(self.rls_yaw.P) < self.convergence_cov_trace_threshold:
                        self.yaw_converged = True
                    self.logger.log_yaw_data(
                        timestamp_sec=timestamp_sec,
                        rls_yaw_deg=self.yaw_deg,
                        rls_cov_trace=np.trace(self.rls_yaw.P),
                        inst_yaw_deg=inst_yaw,
                        constributing_vector=measurement_yaw_rls.copy(),
                    )
                self.logger.log_acc_xy_for_yaw_plot(
                    measurement_yaw_rls[0], measurement_yaw_rls[1]
                )
        if self.rp_converged and self.yaw_converged:
            self.estimation_phase = self.EstimationPhase.ALL_CONVERGED
        elif self.rp_converged:
            self.estimation_phase = self.EstimationPhase.RP_CONVERGED
        elif self.yaw_converged:
            self.estimation_phase = self.EstimationPhase.YAW_CONVERGED
        else:
            self.estimation_phase = self.EstimationPhase.NOT_CALIBRATED
        return self.roll_deg, self.pitch_deg, self.yaw_deg, self.estimation_phase
