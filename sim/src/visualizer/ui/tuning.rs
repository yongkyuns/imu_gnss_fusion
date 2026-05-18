//! EKF and align tuning panel controls.

use eframe::egui;
use sensor_fusion::ProcessNoise;

use crate::visualizer::model::VisualizerMountMode;
use crate::visualizer::pipeline::FusionTuningConfig;

const STANDARD_GRAVITY_MPS2: f32 = 9.80665;

pub(super) fn draw_ekf_tuning(
    ui: &mut egui::Ui,
    misalignment: &mut VisualizerMountMode,
    cfg: &mut FusionTuningConfig,
) {
    ui.horizontal_wrapped(|ui| {
        help_response(ui.label("Mount mode"), "Mount mode");
        help_response(
            ui.selectable_value(misalignment, VisualizerMountMode::Auto, "auto"),
            "auto",
        );
        help_response(
            ui.selectable_value(
                misalignment,
                VisualizerMountMode::Manual,
                "manual/reference",
            ),
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
        slider_f32(
            ui,
            "Mount roll sigma deg",
            &mut cfg.mount_roll_init_sigma_deg,
            0.0..=10.0,
        );
        slider_f32(
            ui,
            "Mount pitch sigma deg",
            &mut cfg.mount_pitch_init_sigma_deg,
            0.0..=10.0,
        );
        slider_f32(
            ui,
            "Mount yaw sigma deg",
            &mut cfg.mount_init_sigma_deg,
            0.0..=30.0,
        );
        let response = ui.checkbox(
            &mut cfg.use_align_mount_covariance_on_seed,
            "Use align mount covariance",
        );
        help_response(response, "Use align mount covariance");
        cfg.mount_roll_pitch_init_sigma_deg = cfg
            .mount_roll_init_sigma_deg
            .max(cfg.mount_pitch_init_sigma_deg);
    });
    ui.collapsing("Mount updates", |ui| {
        slider_rw_var_deg_per_sqrt_hr(
            ui,
            "Mount RW noise deg/sqrt(hr)",
            &mut cfg.mount_align_rw_var,
        );
    });
    ui.collapsing("Prediction", |ui| {
        let noise = cfg
            .noise
            .ekf
            .get_or_insert_with(|| FusionTuningConfig::default().noise.ekf.unwrap());
        drag_gyro_noise_density(ui, "Gyro noise deg/s/sqrt(Hz)", &mut noise.gyro_var);
        drag_accel_noise_density(ui, "Accel noise mg/sqrt(Hz)", &mut noise.accel_var);
        drag_gyro_bias_rw(
            ui,
            "Gyro bias RW deg/hr/sqrt(hr)",
            &mut noise.gyro_bias_rw_var,
        );
        drag_accel_bias_rw(
            ui,
            "Accel bias RW mg/sqrt(hr)",
            &mut noise.accel_bias_rw_var,
        );
        mount_rw_controls(ui, noise);
        drag_usize(
            ui,
            "Predict IMU decimation",
            &mut cfg.predict_imu_decimation,
            1.0,
            1..=32,
        );
        let mut lpf_on = cfg.predict_imu_lpf_cutoff_hz.is_some();
        let response = ui.checkbox(&mut lpf_on, "Predict IMU LPF");
        let changed = response.changed();
        help_response(response, "Predict IMU LPF");
        if changed {
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

pub(super) fn draw_align_tuning(ui: &mut egui::Ui, cfg: &mut FusionTuningConfig) {
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
    });
    ui.collapsing("Post-coarse refinement", |ui| {
        help_response(
            ui.checkbox(
                &mut align.refine_after_coarse_ready,
                "Smooth after coarse ready",
            ),
            "Smooth after coarse ready",
        );
        drag_f32(
            ui,
            "Process noise scale",
            &mut align.refine_process_noise_scale,
            0.01,
            0.0..=1.0,
        );
        drag_f32(
            ui,
            "Observation std scale",
            &mut align.refine_observation_std_scale,
            0.1,
            1.0..=20.0,
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
    help_response(
        ui.checkbox(&mut align.use_gravity, "Use gravity updates"),
        "Use gravity updates",
    );
    help_response(
        ui.checkbox(&mut align.use_turn_gyro, "Use turn gyro updates"),
        "Use turn gyro updates",
    );
}

fn drag_f32(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    speed: f64,
    range: std::ops::RangeInclusive<f64>,
) {
    let range = (*range.start() as f32)..=(*range.end() as f32);
    ui.horizontal(|ui| {
        help_label(ui, label);
        ui.add(
            egui::Slider::new(value, range)
                .step_by(speed)
                .clamping(egui::SliderClamping::Always),
        );
    });
}

fn slider_f32(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
) {
    ui.horizontal(|ui| {
        help_label(ui, label);
        ui.add(egui::Slider::new(value, range).clamping(egui::SliderClamping::Always));
    });
}

fn mount_rw_controls(ui: &mut egui::Ui, noise: &mut ProcessNoise) {
    slider_rw_var_deg_per_sqrt_hr(
        ui,
        "Mount RW scalar deg/sqrt(hr)",
        &mut noise.mount_align_rw_var,
    );
    help_response(
        ui.checkbox(
            &mut noise.mount_align_rw_var_axes_enabled,
            "Axis-specific mount RW",
        ),
        "Axis-specific mount RW",
    );
    if noise.mount_align_rw_var_axes_enabled {
        if noise.mount_align_rw_var_axes == [0.0; 3] && noise.mount_align_rw_var > 0.0 {
            noise.mount_align_rw_var_axes = [noise.mount_align_rw_var; 3];
        }
        slider_rw_var_deg_per_sqrt_hr(
            ui,
            "Mount roll RW deg/sqrt(hr)",
            &mut noise.mount_align_rw_var_axes[0],
        );
        slider_rw_var_deg_per_sqrt_hr(
            ui,
            "Mount pitch RW deg/sqrt(hr)",
            &mut noise.mount_align_rw_var_axes[1],
        );
        slider_rw_var_deg_per_sqrt_hr(
            ui,
            "Mount yaw RW deg/sqrt(hr)",
            &mut noise.mount_align_rw_var_axes[2],
        );
    }
}

fn slider_rw_var_deg_per_sqrt_hr(ui: &mut egui::Ui, label: &str, value_rad2_per_s: &mut f32) {
    let mut value_deg_per_sqrt_hr = rw_var_to_deg_per_sqrt_hr(*value_rad2_per_s);
    slider_f32(ui, label, &mut value_deg_per_sqrt_hr, 0.0..=5.0);
    *value_rad2_per_s = deg_per_sqrt_hr_to_rw_var(value_deg_per_sqrt_hr);
}

fn drag_gyro_noise_density(ui: &mut egui::Ui, label: &str, value_rad2_per_s: &mut f32) {
    let mut value_degps_per_sqrt_hz = gyro_var_to_degps_per_sqrt_hz(*value_rad2_per_s);
    drag_f32(ui, label, &mut value_degps_per_sqrt_hz, 0.001, 0.0..=10.0);
    *value_rad2_per_s = degps_per_sqrt_hz_to_gyro_var(value_degps_per_sqrt_hz);
}

fn drag_accel_noise_density(ui: &mut egui::Ui, label: &str, value_m2ps3: &mut f32) {
    let mut value_mg_per_sqrt_hz = accel_var_to_mg_per_sqrt_hz(*value_m2ps3);
    drag_f32(ui, label, &mut value_mg_per_sqrt_hz, 0.01, 0.0..=1000.0);
    *value_m2ps3 = mg_per_sqrt_hz_to_accel_var(value_mg_per_sqrt_hz);
}

fn drag_gyro_bias_rw(ui: &mut egui::Ui, label: &str, value_rad2ps3: &mut f32) {
    let mut value_degphr_per_sqrt_hr = gyro_bias_rw_var_to_degphr_per_sqrt_hr(*value_rad2ps3);
    drag_f32(ui, label, &mut value_degphr_per_sqrt_hr, 0.01, 0.0..=1000.0);
    *value_rad2ps3 = degphr_per_sqrt_hr_to_gyro_bias_rw_var(value_degphr_per_sqrt_hr);
}

fn drag_accel_bias_rw(ui: &mut egui::Ui, label: &str, value_m2ps5: &mut f32) {
    let mut value_mg_per_sqrt_hr = accel_bias_rw_var_to_mg_per_sqrt_hr(*value_m2ps5);
    drag_f32(ui, label, &mut value_mg_per_sqrt_hr, 0.001, 0.0..=100.0);
    *value_m2ps5 = mg_per_sqrt_hr_to_accel_bias_rw_var(value_mg_per_sqrt_hr);
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

fn gyro_var_to_degps_per_sqrt_hz(value_rad2_per_s: f32) -> f32 {
    if value_rad2_per_s.is_finite() && value_rad2_per_s > 0.0 {
        value_rad2_per_s.sqrt().to_degrees()
    } else {
        0.0
    }
}

fn degps_per_sqrt_hz_to_gyro_var(value_degps_per_sqrt_hz: f32) -> f32 {
    if value_degps_per_sqrt_hz.is_finite() && value_degps_per_sqrt_hz > 0.0 {
        let radps_per_sqrt_hz = value_degps_per_sqrt_hz.to_radians();
        radps_per_sqrt_hz * radps_per_sqrt_hz
    } else {
        0.0
    }
}

fn accel_var_to_mg_per_sqrt_hz(value_m2ps3: f32) -> f32 {
    const MG_PER_MPS2: f32 = 1000.0 / STANDARD_GRAVITY_MPS2;
    if value_m2ps3.is_finite() && value_m2ps3 > 0.0 {
        value_m2ps3.sqrt() * MG_PER_MPS2
    } else {
        0.0
    }
}

fn mg_per_sqrt_hz_to_accel_var(value_mg_per_sqrt_hz: f32) -> f32 {
    const MPS2_PER_MG: f32 = STANDARD_GRAVITY_MPS2 / 1000.0;
    if value_mg_per_sqrt_hz.is_finite() && value_mg_per_sqrt_hz > 0.0 {
        let mps2_per_sqrt_hz = value_mg_per_sqrt_hz * MPS2_PER_MG;
        mps2_per_sqrt_hz * mps2_per_sqrt_hz
    } else {
        0.0
    }
}

fn gyro_bias_rw_var_to_degphr_per_sqrt_hr(value_rad2ps3: f32) -> f32 {
    if value_rad2ps3.is_finite() && value_rad2ps3 > 0.0 {
        value_rad2ps3.sqrt().to_degrees() * 3600.0 * 3600.0_f32.sqrt()
    } else {
        0.0
    }
}

fn degphr_per_sqrt_hr_to_gyro_bias_rw_var(value_degphr_per_sqrt_hr: f32) -> f32 {
    if value_degphr_per_sqrt_hr.is_finite() && value_degphr_per_sqrt_hr > 0.0 {
        let radps_per_sqrt_s = (value_degphr_per_sqrt_hr / 3600.0).to_radians() / 3600.0_f32.sqrt();
        radps_per_sqrt_s * radps_per_sqrt_s
    } else {
        0.0
    }
}

fn accel_bias_rw_var_to_mg_per_sqrt_hr(value_m2ps5: f32) -> f32 {
    const MG_PER_MPS2: f32 = 1000.0 / STANDARD_GRAVITY_MPS2;
    if value_m2ps5.is_finite() && value_m2ps5 > 0.0 {
        value_m2ps5.sqrt() * MG_PER_MPS2 * 3600.0_f32.sqrt()
    } else {
        0.0
    }
}

fn mg_per_sqrt_hr_to_accel_bias_rw_var(value_mg_per_sqrt_hr: f32) -> f32 {
    const MPS2_PER_MG: f32 = STANDARD_GRAVITY_MPS2 / 1000.0;
    if value_mg_per_sqrt_hr.is_finite() && value_mg_per_sqrt_hr > 0.0 {
        let mps2_per_sqrt_s = value_mg_per_sqrt_hr * MPS2_PER_MG / 3600.0_f32.sqrt();
        mps2_per_sqrt_s * mps2_per_sqrt_s
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
    ui.horizontal(|ui| {
        help_label(ui, label);
        ui.add(
            egui::Slider::new(value, range)
                .step_by(speed)
                .clamping(egui::SliderClamping::Always),
        );
    });
}

fn drag_usize(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut usize,
    speed: f64,
    range: std::ops::RangeInclusive<usize>,
) {
    ui.horizontal(|ui| {
        help_label(ui, label);
        ui.add(
            egui::Slider::new(value, range)
                .step_by(speed)
                .clamping(egui::SliderClamping::Always),
        );
    });
}

fn help_label(ui: &mut egui::Ui, label: &str) {
    let response = ui.add_sized(
        [190.0, ui.spacing().interact_size.y],
        egui::Label::new(label).sense(egui::Sense::hover()),
    );
    help_response(response, label);
}

fn help_response(response: egui::Response, label: &str) {
    if let Some(text) = tuning_help(label)
        && response.hovered()
    {
        egui::Tooltip::always_open(
            response.ctx.clone(),
            response.layer_id,
            response.id,
            response.rect,
        )
        .width(420.0)
        .show(|ui| {
            ui.label(text);
        });
    }
}

fn tuning_help(label: &str) -> Option<&'static str> {
    match label {
        "Mount mode" => Some(
            "Select how the EKF obtains the vehicle-to-IMU mount.\nAuto: seed from align and estimate residual mount internally.\nManual/reference: use the supplied mount and freeze mount states.",
        ),
        "auto" => Some(
            "Automatic mount estimation.\nUse this to evaluate normal self-alignment behavior.",
        ),
        "manual/reference" => Some(
            "Fixed mount mode.\nUses the reference/manual mount and freezes EKF mount states; useful as an upper-bound comparison.",
        ),
        "NHC lateral R density" => Some(
            "Lateral nonholonomic constraint noise density.\nHigher: weaker lateral constraint, less mount/roll correction, more tolerance for slip.\nLower: stronger lateral constraint, faster roll/mount correction, more risk from aggressive maneuvers or bad velocity.",
        ),
        "NHC vertical R density" => Some(
            "Vertical nonholonomic constraint noise density.\nHigher: weaker vertical constraint, less pitch/mount correction.\nLower: stronger vertical constraint, faster pitch/mount correction, more risk from road grade or vertical motion.",
        ),
        "Vehicle speed R" => Some(
            "Vehicle forward-speed measurement variance.\nHigher: speed input nudges velocity less.\nLower: speed input pulls longitudinal velocity more strongly.",
        ),
        "Zero velocity R" => Some(
            "Zero-velocity pseudo-measurement variance when stationary.\nHigher: weaker stationary velocity correction.\nLower: pins velocity closer to zero, but can over-constrain if stationary detection is wrong.",
        ),
        "Stationary accel R" => Some(
            "Stationary gravity/accel pseudo-measurement variance.\nHigher: weaker roll/pitch/bias correction while stopped.\nLower: stronger gravity alignment while stopped, but more sensitive to vibration or non-level acceleration.",
        ),
        "Yaw init speed m/s" => Some(
            "Minimum horizontal GNSS speed used to initialize yaw from course.\nHigher: avoids noisy low-speed course yaw, but may delay reliable yaw initialization.\nLower: initializes yaw earlier, but can seed yaw from noisy course.",
        ),
        "Roll/pitch attitude sigma deg" => Some(
            "Initial roll/pitch attitude uncertainty.\nHigher: lets early measurements correct vehicle attitude more.\nLower: makes initial roll/pitch attitude stiffer.",
        ),
        "Yaw sigma deg" => Some(
            "Initial vehicle yaw uncertainty.\nHigher: lets early GNSS/NHC updates rotate yaw more.\nLower: trusts initial yaw more and can slow recovery from bad yaw seed.",
        ),
        "Gyro bias sigma deg/s" => Some(
            "Initial gyro-bias uncertainty.\nHigher: lets early residuals move gyro bias more.\nLower: makes gyro bias stiffer and pushes residuals into attitude/mount instead.",
        ),
        "Accel bias sigma m/s^2" => Some(
            "Initial accelerometer-bias uncertainty.\nHigher: lets early residuals move accel bias more.\nLower: makes accel bias stiffer and pushes residuals into velocity, attitude, or mount.",
        ),
        "Mount roll sigma deg" => Some(
            "Initial residual mount-roll uncertainty after align seed.\nHigher: lets EKF learn mount roll faster, but may absorb maneuver residuals.\nLower: trusts align seed more and makes mount roll adapt slowly.",
        ),
        "Mount pitch sigma deg" => Some(
            "Initial residual mount-pitch uncertainty after align seed.\nHigher: lets EKF learn mount pitch faster.\nLower: trusts align seed more and may leave pitch mount error if the seed is biased.",
        ),
        "Mount yaw sigma deg" => Some(
            "Initial residual mount-yaw uncertainty after align seed.\nHigher: lets turns correct mount yaw more.\nLower: trusts align yaw more and slows recovery from yaw seed error.",
        ),
        "Use align mount covariance" => Some(
            "Seed EKF residual-mount covariance from align's roll/pitch/yaw covariance at handoff.\nOff: use the hard-coded EKF mount sigma controls above.\nOn: preserve align's estimated uncertainty, including roll/pitch/yaw coupling.",
        ),
        "Mount RW noise deg/sqrt(hr)" => Some(
            "Runtime mount random-walk density for the facade-level EKF config.\nHigher: mount can keep adapting after initialization.\nLower: mount becomes stable after convergence but may stop short of the true angle.",
        ),
        "Gyro noise deg/s/sqrt(Hz)" => Some(
            "Gyro white-noise density used in prediction.\nHigher: attitude prediction is trusted less between updates.\nLower: attitude propagation is trusted more, but underestimated gyro noise can make covariance overconfident.",
        ),
        "Accel noise mg/sqrt(Hz)" => Some(
            "Accelerometer white-noise density used in prediction.\nHigher: velocity prediction is trusted less.\nLower: inertial velocity propagation is trusted more and can become overconfident.",
        ),
        "Gyro bias RW deg/hr/sqrt(hr)" => Some(
            "Gyro-bias random-walk density.\nHigher: gyro bias can drift/adapt faster.\nLower: gyro bias is more stable but may not track real bias changes.",
        ),
        "Accel bias RW mg/sqrt(hr)" => Some(
            "Accelerometer-bias random-walk density.\nHigher: accel bias can adapt faster.\nLower: accel bias is more stable but can leave residuals in velocity, attitude, or mount.",
        ),
        "Mount RW scalar deg/sqrt(hr)" => Some(
            "Scalar residual mount random-walk density applied to all mount axes unless axis-specific RW is enabled.\nHigher: mount adapts more during the drive.\nLower: mount becomes smoother and less responsive.",
        ),
        "Axis-specific mount RW" => Some(
            "Enable separate mount random-walk densities for roll, pitch, and yaw.\nOn: tune axes independently.\nOff: use one scalar mount RW for all axes.",
        ),
        "Mount roll RW deg/sqrt(hr)" => Some(
            "Residual mount-roll random-walk density.\nHigher: roll mount adapts more quickly.\nLower: roll mount is smoother but may retain steady-state error.",
        ),
        "Mount pitch RW deg/sqrt(hr)" => Some(
            "Residual mount-pitch random-walk density.\nHigher: pitch mount adapts more quickly.\nLower: pitch mount is smoother but may retain steady-state error.",
        ),
        "Mount yaw RW deg/sqrt(hr)" => Some(
            "Residual mount-yaw random-walk density.\nHigher: yaw mount adapts more through turns.\nLower: yaw mount is smoother but slower to recover.",
        ),
        "Predict IMU decimation" => Some(
            "Prediction cadence divisor for IMU samples.\nHigher: less compute, but lower effective prediction rate and more discretization error.\nLower: more faithful IMU propagation with higher compute cost.",
        ),
        "Predict IMU LPF" => Some(
            "Low-pass filter IMU samples before EKF prediction.\nOn: reduces high-frequency vibration/noise.\nOff: preserves raw dynamics but can inject more vibration into prediction.",
        ),
        "Predict IMU LPF cutoff Hz" => Some(
            "Cutoff for prediction IMU low-pass filtering.\nHigher: preserves faster motion and more noise.\nLower: smoother prediction input, but can attenuate real fast dynamics.",
        ),
        "Vehicle measurement LPF Hz" => Some(
            "Low-pass cutoff for vehicle-motion measurements used by the visualizer/replay path.\nHigher: less smoothing.\nLower: smoother measurements with more delay.",
        ),
        "Align handoff delay s" => Some(
            "How long align must remain coarse-ready before EKF starts.\nHigher: waits for a more stable seed.\nLower: initializes EKF earlier with less confirmation.",
        ),
        "Mount q roll deg" => Some(
            "Align process noise standard deviation for mount roll.\nHigher: align roll can move more between windows.\nLower: align roll is smoother/stiffer.",
        ),
        "Mount q pitch deg" => Some(
            "Align process noise standard deviation for mount pitch.\nHigher: align pitch can move more between windows.\nLower: align pitch is smoother/stiffer.",
        ),
        "Mount q yaw deg" => Some(
            "Align process noise standard deviation for mount yaw.\nHigher: align yaw can move more between windows.\nLower: align yaw is smoother/stiffer.",
        ),
        "Gravity std m/s^2" => Some(
            "Align gravity observation standard deviation.\nHigher: gravity updates are weaker.\nLower: gravity updates are stronger, but more sensitive to acceleration/vibration.",
        ),
        "Horizontal heading std deg" => Some(
            "Align horizontal-motion heading observation standard deviation.\nHigher: heading observations are weaker.\nLower: heading observations pull mount more aggressively.",
        ),
        "Turn heading std deg" => Some(
            "Align turn-derived heading observation standard deviation.\nHigher: turn heading is trusted less.\nLower: turn heading is trusted more and can create sharper corrections.",
        ),
        "Turn gyro std deg/s" => Some(
            "Align turn gyro observation standard deviation.\nHigher: yaw-rate consistency is trusted less.\nLower: turn gyro consistency pushes roll/pitch harder.",
        ),
        "Smooth after coarse ready" => Some(
            "Continue refining align after coarse readiness.\nOn: align keeps adapting after initial seed.\nOff: align behaves closer to coarse-only handoff.",
        ),
        "Process noise scale" => Some(
            "Scale for align post-coarse process noise.\nHigher: post-coarse align remains more adaptive.\nLower: post-coarse align becomes smoother/stiffer.",
        ),
        "Observation std scale" => Some(
            "Scale for align post-coarse observation standard deviations.\nHigher: post-coarse observations are weaker and smoother.\nLower: post-coarse observations cause stronger corrections.",
        ),
        "Gravity LPF alpha" => Some(
            "Align gravity low-pass filter alpha.\nHigher: gravity estimate follows samples faster but is noisier.\nLower: smoother gravity estimate with more lag.",
        ),
        "Min speed m/s" => Some(
            "Minimum speed for align motion windows.\nHigher: rejects slow/noisy motion but uses fewer windows.\nLower: accepts more data, including weaker or noisier observations.",
        ),
        "Min turn rate deg/s" => Some(
            "Minimum yaw rate for turn-based align updates.\nHigher: only strong turns contribute.\nLower: more turns contribute, including weak/noisy ones.",
        ),
        "Max stationary gyro deg/s" => Some(
            "Maximum gyro norm accepted as stationary for bootstrap.\nHigher: easier stationary detection but more risk during slight motion.\nLower: stricter stationary detection and potentially delayed bootstrap.",
        ),
        "Min lat accel m/s^2" => Some(
            "Minimum lateral acceleration for turn observability.\nHigher: only stronger turns update align.\nLower: weaker turns can update but may be noisier.",
        ),
        "Min long accel m/s^2" => Some(
            "Minimum longitudinal acceleration for accel/brake observability.\nHigher: only stronger accel/brake windows update align.\nLower: weaker longitudinal motion can update but may be noisier.",
        ),
        "Max stationary accel norm err" => Some(
            "Maximum acceleration-norm error accepted as stationary.\nHigher: easier stationary bootstrap but more contamination from motion.\nLower: stricter bootstrap and possible initialization delay.",
        ),
        "Min windows" => Some(
            "Minimum number of turn windows required for consistency gating.\nHigher: more evidence before accepting turn updates.\nLower: faster acceptance with less confirmation.",
        ),
        "Min fraction" => Some(
            "Minimum fraction of consistent turn windows.\nHigher: stricter turn consistency.\nLower: more permissive and faster, but more prone to bad updates.",
        ),
        "Max abs lat err m/s^2" => Some(
            "Maximum absolute lateral-acceleration consistency error.\nHigher: accepts less consistent turns.\nLower: stricter rejection of turns that do not match the model.",
        ),
        "Max rel lat err" => Some(
            "Maximum relative lateral-acceleration consistency error.\nHigher: more permissive turn acceptance.\nLower: stricter turn acceptance.",
        ),
        "Use gravity updates" => Some(
            "Enable align gravity observations.\nOn: roll/pitch seed uses stationary gravity.\nOff: removes gravity contribution from align.",
        ),
        "Use turn gyro updates" => Some(
            "Enable align turn gyro observations.\nOn: turn yaw-rate evidence contributes to mount alignment.\nOff: align ignores turn gyro consistency.",
        ),
        _ => None,
    }
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

    #[test]
    fn prediction_noise_ui_units_round_trip() {
        let gyro = 2.287_311_3e-6;
        let gyro_ui = gyro_var_to_degps_per_sqrt_hz(gyro);
        assert!((degps_per_sqrt_hz_to_gyro_var(gyro_ui) - gyro).abs() < 1.0e-12);

        let accel = 3.675_632e-4;
        let accel_ui = accel_var_to_mg_per_sqrt_hz(accel);
        assert!((mg_per_sqrt_hz_to_accel_var(accel_ui) - accel).abs() < 1.0e-10);

        let gyro_bias = 2.0e-13;
        let gyro_bias_ui = gyro_bias_rw_var_to_degphr_per_sqrt_hr(gyro_bias);
        assert!((degphr_per_sqrt_hr_to_gyro_bias_rw_var(gyro_bias_ui) - gyro_bias).abs() < 1.0e-18);

        let accel_bias = 2.0e-12;
        let accel_bias_ui = accel_bias_rw_var_to_mg_per_sqrt_hr(accel_bias);
        assert!((mg_per_sqrt_hr_to_accel_bias_rw_var(accel_bias_ui) - accel_bias).abs() < 1.0e-18);
    }
}
