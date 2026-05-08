//! Reduced, align, and full tuning panel controls.

use eframe::egui;

use crate::visualizer::model::VisualizerMountMode;
use crate::visualizer::pipeline::FilterCompareConfig;

pub(super) fn draw_reduced_tuning(
    ui: &mut egui::Ui,
    misalignment: &mut VisualizerMountMode,
    cfg: &mut FilterCompareConfig,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Mount mode");
        ui.selectable_value(misalignment, VisualizerMountMode::Auto, "auto");
        ui.selectable_value(
            misalignment,
            VisualizerMountMode::Manual,
            "manual/reference",
        );
    });
    ui.collapsing("Measurement weighting", |ui| {
        drag_f32(
            ui,
            "NHC lateral R density",
            &mut cfg.r_body_vel,
            0.01,
            0.0..=1000.0,
        );
        drag_f32(
            ui,
            "NHC vertical R density",
            &mut cfg.r_body_vel_z,
            0.01,
            0.0..=1000.0,
        );
        drag_f32(
            ui,
            "Vehicle speed R",
            &mut cfg.r_vehicle_speed,
            0.01,
            0.0..=10.0,
        );
        drag_f32(ui, "Zero velocity R", &mut cfg.r_zero_vel, 0.01, 0.0..=10.0);
        drag_f32(
            ui,
            "Stationary accel R",
            &mut cfg.r_stationary_accel,
            0.01,
            0.0..=10.0,
        );
    });
    ui.collapsing("Initialization", |ui| {
        drag_f64(
            ui,
            "Yaw init speed m/s",
            &mut cfg.yaw_init_speed_mps,
            0.1,
            0.0..=40.0,
        );
        drag_f32(
            ui,
            "Roll/pitch attitude sigma deg",
            &mut cfg.attitude_roll_pitch_init_sigma_deg,
            0.5,
            0.0..=180.0,
        );
        drag_f32(
            ui,
            "Yaw sigma deg",
            &mut cfg.yaw_init_sigma_deg,
            0.5,
            0.0..=180.0,
        );
        drag_f32(
            ui,
            "Gyro bias sigma deg/s",
            &mut cfg.gyro_bias_init_sigma_dps,
            0.01,
            0.0..=10.0,
        );
        drag_f32(
            ui,
            "Accel bias sigma m/s^2",
            &mut cfg.accel_bias_init_sigma_mps2,
            0.01,
            0.0..=10.0,
        );
        drag_f32(
            ui,
            "Mount roll/pitch sigma deg",
            &mut cfg.mount_roll_pitch_init_sigma_deg,
            0.5,
            0.0..=180.0,
        );
        drag_f32(
            ui,
            "Mount yaw sigma deg",
            &mut cfg.mount_init_sigma_deg,
            0.5,
            0.0..=180.0,
        );
    });
    ui.collapsing("Mount updates", |ui| {
        drag_rw_var_deg_per_sqrt_hr(
            ui,
            "Mount RW noise deg/sqrt(hr)",
            &mut cfg.mount_align_rw_var,
        );
        ui.checkbox(&mut cfg.freeze_misalignment_states, "Freeze mount states");
        drag_f32(
            ui,
            "Mount settle time s",
            &mut cfg.mount_settle_time_s,
            10.0,
            0.0..=1000.0,
        );
        drag_f32(
            ui,
            "Mount release sigma deg",
            &mut cfg.mount_settle_release_sigma_deg,
            0.5,
            0.0..=90.0,
        );
        ui.checkbox(
            &mut cfg.mount_settle_zero_cross_covariance,
            "Zero mount cross-covariance on release",
        );
    });
    ui.collapsing("Prediction", |ui| {
        let noise = cfg
            .noise
            .reduced
            .get_or_insert_with(|| FilterCompareConfig::default().noise.reduced.unwrap());
        drag_f32(ui, "Gyro var", &mut noise.gyro_var, 1.0e-7, 0.0..=1.0);
        drag_f32(ui, "Accel var", &mut noise.accel_var, 1.0e-5, 0.0..=100.0);
        drag_f32(
            ui,
            "Gyro bias RW var",
            &mut noise.gyro_bias_rw_var,
            1.0e-13,
            0.0..=1.0e-6,
        );
        drag_f32(
            ui,
            "Accel bias RW var",
            &mut noise.accel_bias_rw_var,
            1.0e-12,
            0.0..=1.0e-6,
        );
        drag_rw_var_deg_per_sqrt_hr(
            ui,
            "Mount RW noise deg/sqrt(hr)",
            &mut noise.mount_align_rw_var,
        );
        drag_usize(
            ui,
            "Predict IMU decimation",
            &mut cfg.predict_imu_decimation,
            1.0,
            1..=32,
        );
        let mut lpf_on = cfg.predict_imu_lpf_cutoff_hz.is_some();
        if ui.checkbox(&mut lpf_on, "Predict IMU LPF").changed() {
            cfg.predict_imu_lpf_cutoff_hz = lpf_on.then_some(150.0);
        }
        if let Some(cutoff) = cfg.predict_imu_lpf_cutoff_hz.as_mut() {
            drag_f64(ui, "Predict IMU LPF cutoff Hz", cutoff, 1.0, 1.0..=500.0);
        }
        drag_f64(
            ui,
            "Vehicle measurement LPF Hz",
            &mut cfg.vehicle_meas_lpf_cutoff_hz,
            1.0,
            1.0..=500.0,
        );
    });
}

pub(super) fn draw_align_tuning(ui: &mut egui::Ui, cfg: &mut FilterCompareConfig) {
    drag_f32(
        ui,
        "Align handoff delay s",
        &mut cfg.align_handoff_delay_s,
        1.0,
        0.0..=600.0,
    );
    let align = &mut cfg.align;
    ui.collapsing("Process and observation noise", |ui| {
        let mut q_roll = align.q_mount_std_rad[0].to_degrees();
        let mut q_pitch = align.q_mount_std_rad[1].to_degrees();
        let mut q_yaw = align.q_mount_std_rad[2].to_degrees();
        drag_f32(ui, "Mount q roll deg", &mut q_roll, 0.0001, 0.0..=1.0);
        drag_f32(ui, "Mount q pitch deg", &mut q_pitch, 0.0001, 0.0..=1.0);
        drag_f32(ui, "Mount q yaw deg", &mut q_yaw, 0.0001, 0.0..=1.0);
        align.q_mount_std_rad = [
            q_roll.to_radians(),
            q_pitch.to_radians(),
            q_yaw.to_radians(),
        ];
        drag_f32(
            ui,
            "Gravity std m/s^2",
            &mut align.r_gravity_std_mps2,
            0.01,
            0.001..=10.0,
        );
        let mut horiz = align.r_horiz_heading_std_rad.to_degrees();
        let mut turn_heading = align.r_turn_heading_std_rad.to_degrees();
        let mut turn_gyro = align.r_turn_gyro_std_radps.to_degrees();
        drag_f32(
            ui,
            "Horizontal heading std deg",
            &mut horiz,
            0.1,
            0.001..=90.0,
        );
        drag_f32(
            ui,
            "Turn heading std deg",
            &mut turn_heading,
            0.1,
            0.001..=90.0,
        );
        drag_f32(ui, "Turn gyro std deg/s", &mut turn_gyro, 0.001, 0.0..=10.0);
        align.r_horiz_heading_std_rad = horiz.to_radians();
        align.r_turn_heading_std_rad = turn_heading.to_radians();
        align.r_turn_gyro_std_radps = turn_gyro.to_radians();
        drag_f32(
            ui,
            "Turn gyro yaw scale",
            &mut align.turn_gyro_yaw_scale,
            0.01,
            0.0..=1.0,
        );
    });
    ui.collapsing("Motion gates", |ui| {
        drag_f32(
            ui,
            "Gravity LPF alpha",
            &mut align.gravity_lpf_alpha,
            0.01,
            0.0..=1.0,
        );
        drag_f32(
            ui,
            "Min speed m/s",
            &mut align.min_speed_mps,
            0.05,
            0.0..=20.0,
        );
        let mut turn_rate = align.min_turn_rate_radps.to_degrees();
        let mut stationary_gyro = align.max_stationary_gyro_radps.to_degrees();
        drag_f32(ui, "Min turn rate deg/s", &mut turn_rate, 0.1, 0.0..=90.0);
        drag_f32(
            ui,
            "Max stationary gyro deg/s",
            &mut stationary_gyro,
            0.1,
            0.0..=90.0,
        );
        align.min_turn_rate_radps = turn_rate.to_radians();
        align.max_stationary_gyro_radps = stationary_gyro.to_radians();
        drag_f32(
            ui,
            "Min lat accel m/s^2",
            &mut align.min_lat_acc_mps2,
            0.01,
            0.0..=20.0,
        );
        drag_f32(
            ui,
            "Min long accel m/s^2",
            &mut align.min_long_acc_mps2,
            0.01,
            0.0..=20.0,
        );
        drag_f32(
            ui,
            "Max stationary accel norm err",
            &mut align.max_stationary_accel_norm_err_mps2,
            0.01,
            0.0..=10.0,
        );
    });
    ui.collapsing("Turn consistency", |ui| {
        drag_usize(
            ui,
            "Min windows",
            &mut align.turn_consistency_min_windows,
            1.0,
            0..=100,
        );
        drag_f32(
            ui,
            "Min fraction",
            &mut align.turn_consistency_min_fraction,
            0.01,
            0.0..=1.0,
        );
        drag_f32(
            ui,
            "Max abs lat err m/s^2",
            &mut align.turn_consistency_max_abs_lat_err_mps2,
            0.01,
            0.0..=20.0,
        );
        drag_f32(
            ui,
            "Max rel lat err",
            &mut align.turn_consistency_max_rel_lat_err,
            0.01,
            0.0..=10.0,
        );
    });
    ui.checkbox(&mut align.use_gravity, "Use gravity updates");
    ui.checkbox(&mut align.use_turn_gyro, "Use turn gyro updates");
}

pub(super) fn draw_full_tuning(ui: &mut egui::Ui, cfg: &mut FilterCompareConfig) {
    ui.collapsing("Prediction noise", |ui| {
        let noise = cfg
            .noise
            .full
            .get_or_insert_with(|| FilterCompareConfig::default().noise.full.unwrap());
        drag_f32(ui, "Gyro var", &mut noise.gyro_var, 1.0e-7, 0.0..=1.0);
        drag_f32(ui, "Accel var", &mut noise.accel_var, 1.0e-5, 0.0..=100.0);
        drag_f32(
            ui,
            "Gyro bias RW var",
            &mut noise.gyro_bias_rw_var,
            1.0e-13,
            0.0..=1.0e-6,
        );
        drag_f32(
            ui,
            "Accel bias RW var",
            &mut noise.accel_bias_rw_var,
            1.0e-12,
            0.0..=1.0e-6,
        );
        drag_f32(
            ui,
            "Gyro scale RW var",
            &mut noise.gyro_scale_rw_var,
            1.0e-11,
            0.0..=1.0e-6,
        );
        drag_f32(
            ui,
            "Accel scale RW var",
            &mut noise.accel_scale_rw_var,
            1.0e-11,
            0.0..=1.0e-6,
        );
        drag_rw_var_deg_per_sqrt_hr(
            ui,
            "Mount RW noise deg/sqrt(hr)",
            &mut noise.mount_align_rw_var,
        );
    });
    ui.collapsing("Initial covariance", |ui| {
        let init = &mut cfg.full_init;
        drag_f32(
            ui,
            "Position min sigma m",
            &mut init.pos_min_sigma_m,
            0.1,
            0.0..=100.0,
        );
        drag_f32(
            ui,
            "Velocity min sigma m/s",
            &mut init.vel_min_sigma_mps,
            0.1,
            0.0..=50.0,
        );
        drag_f32(
            ui,
            "Attitude sigma deg",
            &mut init.attitude_sigma_deg,
            0.5,
            0.0..=180.0,
        );
        drag_f32(
            ui,
            "Gyro bias sigma deg/s",
            &mut init.gyro_bias_sigma_dps,
            0.01,
            0.0..=10.0,
        );
        drag_f32(
            ui,
            "Accel bias sigma m/s^2",
            &mut init.accel_bias_sigma_mps2,
            0.01,
            0.0..=10.0,
        );
        drag_f32(
            ui,
            "Gyro scale sigma",
            &mut init.gyro_scale_sigma,
            0.001,
            0.0..=1.0,
        );
        drag_f32(
            ui,
            "Accel scale sigma",
            &mut init.accel_scale_sigma,
            0.001,
            0.0..=1.0,
        );
        drag_f32(
            ui,
            "Mount roll/pitch sigma deg",
            &mut init.mount_sigma_deg,
            0.5,
            0.0..=180.0,
        );
        drag_f32(
            ui,
            "Mount yaw sigma deg",
            &mut init.mount_yaw_sigma_deg,
            0.5,
            0.0..=180.0,
        );
    });
}

fn drag_f32(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    speed: f64,
    range: std::ops::RangeInclusive<f64>,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(value).speed(speed).range(range));
    });
}

fn drag_rw_var_deg_per_sqrt_hr(ui: &mut egui::Ui, label: &str, value_rad2_per_s: &mut f32) {
    let mut value_deg_per_sqrt_hr = rw_var_to_deg_per_sqrt_hr(*value_rad2_per_s);
    drag_f32(ui, label, &mut value_deg_per_sqrt_hr, 0.01, 0.0..=10.0);
    *value_rad2_per_s = deg_per_sqrt_hr_to_rw_var(value_deg_per_sqrt_hr);
}

fn rw_var_to_deg_per_sqrt_hr(value_rad2_per_s: f32) -> f32 {
    if value_rad2_per_s.is_finite() && value_rad2_per_s > 0.0 {
        value_rad2_per_s.sqrt().to_degrees() * 3600.0_f32.sqrt()
    } else {
        0.0
    }
}

fn deg_per_sqrt_hr_to_rw_var(value_deg_per_sqrt_hr: f32) -> f32 {
    if value_deg_per_sqrt_hr.is_finite() && value_deg_per_sqrt_hr > 0.0 {
        let rad_per_sqrt_s = value_deg_per_sqrt_hr.to_radians() / 3600.0_f32.sqrt();
        rad_per_sqrt_s * rad_per_sqrt_s
    } else {
        0.0
    }
}

fn drag_f64(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f64,
    speed: f64,
    range: std::ops::RangeInclusive<f64>,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(value).speed(speed).range(range));
    });
}

fn drag_usize(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut usize,
    speed: f64,
    range: std::ops::RangeInclusive<usize>,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(value).speed(speed).range(range));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rw_var_ui_unit_round_trips() {
        let q = 1.0e-8;
        let ui_value = rw_var_to_deg_per_sqrt_hr(q);
        assert!((ui_value - 0.343_775).abs() < 1.0e-5);
        assert!((deg_per_sqrt_hr_to_rw_var(ui_value) - q).abs() < 1.0e-14);
    }
}
