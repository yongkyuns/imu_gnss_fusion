use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::align::{Align, AlignConfig, AlignWindowSummary};

#[pyclass(name = "Align")]
struct PyAlign {
    inner: Align,
}

#[pymethods]
impl PyAlign {
    #[new]
    #[pyo3(signature = (
        use_gravity = true,
        use_turn_gyro = true,
        use_course_rate = true,
        use_lateral_accel = true,
        use_longitudinal_accel = true,
    ))]
    fn new(
        use_gravity: bool,
        use_turn_gyro: bool,
        use_course_rate: bool,
        use_lateral_accel: bool,
        use_longitudinal_accel: bool,
    ) -> Self {
        let mut cfg = AlignConfig::default();
        cfg.use_gravity = use_gravity;
        cfg.use_turn_gyro = use_turn_gyro;
        cfg.use_course_rate = use_course_rate;
        cfg.use_lateral_accel = use_lateral_accel;
        cfg.use_longitudinal_accel = use_longitudinal_accel;
        Self {
            inner: Align::new(cfg),
        }
    }

    #[pyo3(signature = (accel_samples_b, yaw_seed_deg = 0.0))]
    fn initialize_from_stationary(
        &mut self,
        accel_samples_b: Vec<(f32, f32, f32)>,
        yaw_seed_deg: f32,
    ) -> PyResult<()> {
        let samples: Vec<[f32; 3]> = accel_samples_b
            .into_iter()
            .map(|(x, y, z)| [x, y, z])
            .collect();
        self.inner
            .initialize_from_stationary(&samples, yaw_seed_deg.to_radians())
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (dt, mean_gyro_b, mean_accel_b, gnss_vel_prev_n, gnss_vel_curr_n))]
    fn update_window(
        &mut self,
        dt: f32,
        mean_gyro_b: (f32, f32, f32),
        mean_accel_b: (f32, f32, f32),
        gnss_vel_prev_n: (f32, f32, f32),
        gnss_vel_curr_n: (f32, f32, f32),
    ) -> f32 {
        let window = AlignWindowSummary {
            dt,
            mean_gyro_b: [mean_gyro_b.0, mean_gyro_b.1, mean_gyro_b.2],
            mean_accel_b: [mean_accel_b.0, mean_accel_b.1, mean_accel_b.2],
            gnss_vel_prev_n: [gnss_vel_prev_n.0, gnss_vel_prev_n.1, gnss_vel_prev_n.2],
            gnss_vel_curr_n: [gnss_vel_curr_n.0, gnss_vel_curr_n.1, gnss_vel_curr_n.2],
        };
        self.inner.update_window(&window)
    }

    fn mount_angles_deg(&self) -> (f32, f32, f32) {
        let angles = self.inner.mount_angles_deg();
        (angles[0], angles[1], angles[2])
    }

    fn mount_angles_rad(&self) -> (f32, f32, f32) {
        let angles = self.inner.mount_angles_rad();
        (angles[0], angles[1], angles[2])
    }

    fn sigma_deg(&self) -> (f32, f32, f32) {
        let sigma = self.inner.sigma_deg();
        (sigma[0], sigma[1], sigma[2])
    }

    fn quaternion(&self) -> (f32, f32, f32, f32) {
        let q = self.inner.q_vb;
        (q[0], q[1], q[2], q[3])
    }
}

#[pymodule]
fn align_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAlign>()?;
    Ok(())
}
