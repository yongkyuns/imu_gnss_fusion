#![allow(clippy::excessive_precision, dead_code)]

use std::fs;
use std::path::Path;

use anyhow::{Result, bail};
use sensor_fusion::SensorFusion;
use sim::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, fusion_gnss_sample, fusion_imu_sample,
};
use sim::eval::gnss_ins::{as_q64, quat_angle_deg, quat_rotate};
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::eval::trace::{
    require_trace, require_trace_points, require_trace_schema, sample_nearest_value,
};
use sim::synthetic::gnss_ins_path::{
    AxisNoise, GpsNoiseModel, ImuAccuracy, ImuNoiseModel, MeasurementNoiseConfig, MotionProfile,
    PathGenConfig, VibrationNoise, add_measurement_noise, generate, generate_with_noise,
};
use sim::visualizer::math::{ecef_to_ned, lla_to_ecef};
use sim::visualizer::model::MountSourceMode;
use sim::visualizer::pipeline::generic::reference_mount_rpy_to_q_vb;
use sim::visualizer::pipeline::synthetic::{
    SyntheticNoiseMode, SyntheticVisualizerConfig, build_synthetic_plot_data,
};
use sim::visualizer::pipeline::{FilterCompareConfig, GnssOutageConfig};

const SHORT_PROFILE: &str = "\
initial lat=32 lon=120 alt=0 vx=0 vy=0 vz=0 yaw=0 pitch=0 roll=0
command type=1 yaw=0 pitch=0 roll=0 ax=0 ay=0 az=0 for=1s gps=on
command type=1 yaw=5 pitch=0 roll=0 ax=0.5 ay=0 az=0 for=2s gps=on
command type=1 yaw=-5 pitch=0 roll=0 ax=-0.5 ay=0 az=0 for=2s gps=on
";
const VISUALIZER_AUX_SCENARIO: &str = "\
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 5s
repeat 10 {
    accelerate 1.0m/s^2 for 4s
    hold 4s
    turn left 12dps for 4s
    hold 4s
    turn right 12dps for 4s
    brake 1.0m/s^2 for 4s
}
";

#[test]
fn rust_path_generator_matches_synthetic_replay_reference_samples() -> Result<()> {
    let profile = MotionProfile::from_dsl_str(SHORT_PROFILE)?;
    let generated = generate(&profile, PathGenConfig::default())?;

    assert_eq!(generated.imu.len(), 500);
    assert_eq!(generated.truth.len(), 500);
    assert_eq!(generated.gnss.len(), 10);

    assert_vec_close(
        "imu[0].accel",
        generated.imu[0].accel_vehicle_mps2,
        [0.0, 0.0, -9.794841972265040],
        1.0e-13,
    );
    assert_vec_close(
        "imu[0].gyro",
        generated.imu[0].gyro_vehicle_radps,
        [6.184064242703716e-05, 0.0, -3.864232215503917e-05],
        1.0e-16,
    );
    assert_vec_close(
        "imu[250].accel",
        generated.imu[250].accel_vehicle_mps2,
        [
            0.49999993839883439,
            0.061468357438283161,
            -9.7948311957013523,
        ],
        1.0e-12,
    );
    assert_vec_close(
        "imu[250].gyro",
        generated.imu[250].gyro_vehicle_radps,
        [
            6.1373023280329687e-05,
            -7.70099628080859e-06,
            0.087227801052051454,
        ],
        1.0e-15,
    );

    let truth = generated.truth[250];
    assert_close(
        "truth[250].lat_deg",
        truth.lat_deg,
        32.00000445406005,
        1.0e-12,
    );
    assert_close(
        "truth[250].lon_deg",
        truth.lon_deg,
        120.00000042658029,
        1.0e-12,
    );
    assert_vec_close(
        "truth[250].vel",
        truth.vel_ned_mps,
        [0.69966979899320469, 0.086528498557255362, 0.0],
        1.0e-12,
    );
    assert_quat_close(
        "truth[250].q_bn",
        truth.q_bn,
        [0.99810806592381152, 0.0, 0.0, 0.061484052711481649],
        1.0e-15,
    );

    let gnss = generated.gnss[5];
    assert_close("gnss[5].t_s", gnss.t_s, 2.5, 1.0e-12);
    assert_close("gnss[5].lat_deg", gnss.lat_deg, 32.00000445406005, 1.0e-12);
    assert_vec_close(
        "gnss[5].vel",
        gnss.vel_ned_mps,
        [0.69966979899320469, 0.086528498557255362, 0.0],
        1.0e-12,
    );

    Ok(())
}

#[test]
fn motion_dsl_expands_to_gnss_ins_motion_commands() -> Result<()> {
    let profile = MotionProfile::from_dsl_str(
        r#"
        initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
        wait 60s
        repeat 2 {
            accelerate 1.0m/s^2 for 8s
            hold 12s
            turn left 10dps for 9s
            brake 1.0m/s^2 for 8s
            drive yaw=-10dps ax=0 for=9s gps=off
        }
        "#,
    )?;

    assert_eq!(profile.initial.lat_deg, 32.0);
    assert_eq!(profile.commands.len(), 11);
    assert_eq!(profile.commands[1].body_cmd, [1.0, 0.0, 0.0]);
    assert_eq!(profile.commands[3].yaw_pitch_roll_cmd_deg, [10.0, 0.0, 0.0]);
    assert_eq!(profile.commands[4].body_cmd, [-1.0, 0.0, 0.0]);
    assert_eq!(
        profile.commands[5].yaw_pitch_roll_cmd_deg,
        [-10.0, 0.0, 0.0]
    );
    assert!(!profile.commands[5].gps_visible);
    Ok(())
}

#[test]
fn motion_dsl_supports_explicit_command_type_semantics() -> Result<()> {
    let profile = MotionProfile::from_dsl_str(
        r#"
        initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=3 roll=0
        command type=2 yaw=90 pitch=3 roll=0 ax=0 ay=0 az=0 for=18s gps=on
        command type=3 yaw=0 pitch=-3 roll=-3 ax=8 ay=0 az=0 for=15s gps=off
        "#,
    )?;

    assert_eq!(profile.commands.len(), 2);
    assert_eq!(profile.commands[0].command_type, 2);
    assert_eq!(profile.commands[0].yaw_pitch_roll_cmd_deg, [90.0, 3.0, 0.0]);
    assert_eq!(profile.commands[1].command_type, 3);
    assert_eq!(
        profile.commands[1].yaw_pitch_roll_cmd_deg,
        [0.0, -3.0, -3.0]
    );
    assert_eq!(profile.commands[1].body_cmd, [8.0, 0.0, 0.0]);
    assert!(!profile.commands[1].gps_visible);
    Ok(())
}

#[test]
fn measurement_noise_supports_bias_drift_white_noise_vibration_and_gps_noise() -> Result<()> {
    let profile = MotionProfile::from_dsl_str(SHORT_PROFILE)?;
    let reference = generate(&profile, PathGenConfig::default())?;

    let bias_only = add_measurement_noise(
        &reference,
        100.0,
        MeasurementNoiseConfig {
            imu: ImuNoiseModel {
                gyro: AxisNoise {
                    bias: [0.01, -0.02, 0.03],
                    ..AxisNoise::zero()
                },
                accel: AxisNoise {
                    bias: [0.1, -0.2, 0.3],
                    ..AxisNoise::zero()
                },
                gyro_vibration: None,
                accel_vibration: None,
            },
            gps: None,
        },
        7,
    );
    for (actual, expected) in bias_only.imu.iter().zip(&reference.imu) {
        assert_vec_close(
            "gyro static bias",
            sub3(actual.gyro_vehicle_radps, expected.gyro_vehicle_radps),
            [0.01, -0.02, 0.03],
            1.0e-14,
        );
        assert_vec_close(
            "accel static bias",
            sub3(actual.accel_vehicle_mps2, expected.accel_vehicle_mps2),
            [0.1, -0.2, 0.3],
            1.0e-14,
        );
    }
    assert_eq!(bias_only.gnss.len(), reference.gnss.len());
    assert_close(
        "gps unchanged without gps noise",
        bias_only.gnss[3].lat_deg,
        reference.gnss[3].lat_deg,
        0.0,
    );

    let stochastic_noise = MeasurementNoiseConfig {
        imu: ImuNoiseModel {
            gyro: AxisNoise {
                bias: [0.001, 0.0, -0.001],
                bias_drift_std: [1.0e-5; 3],
                bias_corr_time_s: [100.0; 3],
                white_noise_density: [2.0e-5; 3],
            },
            accel: AxisNoise {
                bias: [0.01, 0.0, -0.01],
                bias_drift_std: [2.0e-4; 3],
                bias_corr_time_s: [100.0; 3],
                white_noise_density: [3.0e-4; 3],
            },
            gyro_vibration: Some(VibrationNoise::Sinusoidal {
                amplitude: [1.0e-4, 2.0e-4, 3.0e-4],
                freq_hz: 3.0,
            }),
            accel_vibration: Some(VibrationNoise::Random {
                std: [0.01, 0.02, 0.03],
            }),
        },
        gps: Some(GpsNoiseModel::accuracy(ImuAccuracy::Low)),
    };
    let noisy_a = add_measurement_noise(&reference, 100.0, stochastic_noise, 42);
    let noisy_b = add_measurement_noise(&reference, 100.0, stochastic_noise, 42);
    let noisy_c = add_measurement_noise(&reference, 100.0, stochastic_noise, 43);
    assert_vec_close(
        "seeded gyro repeatability",
        noisy_a.imu[123].gyro_vehicle_radps,
        noisy_b.imu[123].gyro_vehicle_radps,
        0.0,
    );
    assert_vec_close(
        "seeded accel repeatability",
        noisy_a.imu[123].accel_vehicle_mps2,
        noisy_b.imu[123].accel_vehicle_mps2,
        0.0,
    );
    assert_close(
        "seeded gps repeatability",
        noisy_a.gnss[4].lat_deg,
        noisy_b.gnss[4].lat_deg,
        0.0,
    );
    assert!(
        norm3(sub3(
            noisy_a.imu[123].accel_vehicle_mps2,
            reference.imu[123].accel_vehicle_mps2
        )) > 1.0e-4
    );
    assert!(
        norm3(sub3(
            noisy_a.imu[123].accel_vehicle_mps2,
            noisy_c.imu[123].accel_vehicle_mps2
        )) > 1.0e-4
    );
    assert!((noisy_a.gnss[4].lat_deg - reference.gnss[4].lat_deg).abs() > 1.0e-8);
    assert!(
        MeasurementNoiseConfig::accuracy(ImuAccuracy::Low)
            .gps
            .is_some()
    );

    Ok(())
}

#[test]
fn reduced_converges_on_generated_city_blocks_truth_signals() -> Result<()> {
    assert_reduced_converges_on_profile("city_blocks_15min.scenario")
}

#[test]
fn reduced_converges_on_generated_figure8_truth_signals() -> Result<()> {
    assert_reduced_converges_on_profile("figure8_15min.scenario")
}

#[test]
fn reduced_converges_tightly_on_long_generated_figure8_truth_signals() -> Result<()> {
    let profile_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("motion_profiles/figure8_30min.scenario");
    let profile = MotionProfile::from_path(&profile_path)?;
    let generated = generate(&profile, PathGenConfig::default())?;
    let summary = run_reduced_on_generated_path(&generated, [5.0, -5.0, 5.0])?;

    assert!(
        summary.reduced_initialized,
        "Reduced did not initialize on long figure-eight truth data"
    );
    assert!(
        summary.final_mount_quat_err_deg < 0.15,
        "long figure-eight mount quaternion error too high: {summary:#?}"
    );
    assert!(
        summary.tail_mount_quat_err_mean_deg < 0.15,
        "long figure-eight tail mount quaternion mean error too high: {summary:#?}"
    );
    assert!(
        summary.final_att_quat_err_deg < 0.15,
        "long figure-eight attitude quaternion error too high: {summary:#?}"
    );
    assert!(
        summary.final_vel_err_mps < 0.35,
        "long figure-eight velocity error too high: {summary:#?}"
    );
    assert!(
        summary.final_pos_err_m < 4.0,
        "long figure-eight position error too high: {summary:#?}"
    );

    Ok(())
}

#[test]
fn reduced_converges_on_generated_city_blocks_noisy_measurements() -> Result<()> {
    let profile_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("motion_profiles/city_blocks_15min.scenario");
    let profile = MotionProfile::from_path(&profile_path)?;
    let noise = MeasurementNoiseConfig::accuracy(ImuAccuracy::Mid);
    let gps_noise = noise.gps.unwrap();
    let measured = generate_with_noise(&profile, PathGenConfig::default(), noise, 20260426)?;
    let summary = run_reduced_on_samples(
        &measured.reference,
        &measured.imu,
        &measured.gnss,
        [5.0, -5.0, 5.0],
        gps_noise.pos_std_m,
        gps_noise.vel_std_mps,
    )?;
    assert!(
        summary.final_mount_quat_err_deg < 1.5,
        "noisy mount quaternion error too high: {summary:#?}"
    );
    assert!(
        summary.tail_mount_quat_err_mean_deg < 1.5,
        "noisy tail mount quaternion mean error too high: {summary:#?}"
    );
    assert!(
        summary.final_att_quat_err_deg < 3.0,
        "noisy attitude quaternion error too high: {summary:#?}"
    );
    assert!(
        summary.final_vel_err_mps < 0.75,
        "noisy velocity error too high: {summary:#?}"
    );
    assert!(
        summary.final_pos_err_m < 12.0,
        "noisy position error too high: {summary:#?}"
    );
    Ok(())
}

#[test]
fn synthetic_early_velocity_fault_does_not_drive_reduced_mount_by_default() -> Result<()> {
    let profile_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("motion_profiles/figure8_15min.scenario");
    let profile = MotionProfile::from_path(&profile_path)?;
    let generated = generate(&profile, PathGenConfig::default())?;
    let faulted_gnss = gnss_with_early_velocity_bias(&generated.gnss, [0.5, 0.0, 0.0], 360.0);

    let clean = run_reduced_on_generated_path(&generated, [5.0, -5.0, 5.0])?;
    let faulted_default = run_reduced_on_samples(
        &generated,
        &generated.imu,
        &faulted_gnss,
        [5.0, -5.0, 5.0],
        [0.5, 0.5, 0.5],
        [0.2, 0.2, 0.2],
    )?;
    assert!(
        clean.tail_mount_quat_err_mean_deg < 0.15,
        "clean figure-eight should not create a mount basin error: {clean:#?}"
    );
    assert!(
        faulted_default.tail_mount_quat_err_mean_deg < 0.25,
        "default Reduced should not let an early GNSS velocity fault directly push mount into a wrong basin: {faulted_default:#?}"
    );
    Ok(())
}

#[test]
fn synthetic_inputs_populate_visualizer_reduced_traces() -> Result<()> {
    let profile_path = std::env::temp_dir().join(format!(
        "imu_gnss_fusion_visualizer_short_{}.scenario",
        std::process::id()
    ));
    fs::write(&profile_path, VISUALIZER_AUX_SCENARIO)?;
    let data_result = build_synthetic_plot_data(
        &SyntheticVisualizerConfig {
            motion_def: Some(profile_path.clone()),
            motion_label: profile_path.display().to_string(),
            motion_text: None,
            noise_mode: SyntheticNoiseMode::Truth,
            disable_imu_noise: false,
            disable_gnss_noise: false,
            seed: 42,
            mount_rpy_deg: [5.0, -5.0, 5.0],
            imu_hz: 100.0,
            gnss_hz: 2.0,
            gnss_time_shift_ms: 0.0,
            early_vel_bias_ned_mps: [0.0; 3],
            early_fault_window_s: None,
        },
        MountSourceMode::Ref,
        FilterCompareConfig::default(),
        GnssOutageConfig::default(),
    );
    let remove_result = fs::remove_file(&profile_path);
    let data = data_result?;
    remove_result?;

    assert!(!data.speed.is_empty());
    assert!(!data.imu_raw_gyro.is_empty());
    assert!(!data.imu_raw_accel.is_empty());
    assert!(!data.orientation.is_empty());
    assert!(!data.reduced_cmp_pos.is_empty());
    assert!(!data.reduced_cmp_vel.is_empty());
    assert!(!data.reduced_cmp_att.is_empty());
    assert!(!data.reduced_misalignment.is_empty());
    assert!(!data.reduced_map.is_empty());
    assert!(
        data.full_cmp_att
            .iter()
            .any(|trace| !trace.points.is_empty()),
        "synthetic visualizer produced no full attitude points"
    );
    assert!(
        data.full_bias_accel
            .iter()
            .any(|trace| !trace.points.is_empty()),
        "synthetic visualizer produced no full accel-bias points"
    );
    assert!(
        data.reduced_bump_pitch_speed
            .iter()
            .any(|trace| !trace.points.is_empty()),
        "synthetic visualizer produced no Reduced bump pitch/speed points"
    );
    assert!(
        data.reduced_bump_diag
            .iter()
            .any(|trace| !trace.points.is_empty()),
        "synthetic visualizer produced no Reduced bump diagnostic points"
    );
    assert!(
        data.align_cmp_att
            .iter()
            .any(|trace| !trace.points.is_empty()),
        "synthetic visualizer produced no align compare attitude points"
    );
    assert!(
        data.align_cov.iter().any(|trace| !trace.points.is_empty()),
        "synthetic visualizer produced no align covariance points"
    );
    require_trace_schema(
        "speed",
        &data.speed,
        &[
            "Synthetic truth horizontal speed [m/s]",
            "Synthetic GNSS horizontal speed [m/s]",
        ],
    )?;
    require_trace_schema(
        "imu_raw_gyro",
        &data.imu_raw_gyro,
        &[
            "Synthetic raw IMU gyro X [deg/s]",
            "Synthetic raw IMU gyro Y [deg/s]",
            "Synthetic raw IMU gyro Z [deg/s]",
        ],
    )?;
    require_trace_schema(
        "imu_raw_accel",
        &data.imu_raw_accel,
        &[
            "Synthetic raw IMU accel X [m/s^2]",
            "Synthetic raw IMU accel Y [m/s^2]",
            "Synthetic raw IMU accel Z [m/s^2]",
        ],
    )?;
    require_trace_schema(
        "orientation",
        &data.orientation,
        &[
            "Synthetic truth roll [deg]",
            "Synthetic truth pitch [deg]",
            "Synthetic truth yaw [deg]",
        ],
    )?;
    require_trace_schema(
        "reduced_cmp_pos",
        &data.reduced_cmp_pos,
        &[
            "Reduced posN [m]",
            "Synthetic truth posN [m]",
            "Reduced posE [m]",
            "Synthetic truth posE [m]",
            "Reduced posD [m]",
            "Synthetic truth posD [m]",
        ],
    )?;
    require_trace_schema(
        "reduced_cmp_vel",
        &data.reduced_cmp_vel,
        &[
            "Reduced vN [m/s]",
            "Synthetic truth vN [m/s]",
            "Reduced vE [m/s]",
            "Synthetic truth vE [m/s]",
            "Reduced vD [m/s]",
            "Synthetic truth vD [m/s]",
        ],
    )?;
    require_trace_schema(
        "reduced_cmp_att",
        &data.reduced_cmp_att,
        &[
            "Reduced roll [deg]",
            "Synthetic truth roll [deg]",
            "Reduced pitch [deg]",
            "Synthetic truth pitch [deg]",
            "Reduced yaw [deg]",
            "Synthetic truth yaw [deg]",
            "mount ready",
            "Reduced initialized",
        ],
    )?;
    require_trace_schema(
        "reduced_misalignment",
        &data.reduced_misalignment,
        &[
            "Reduced mount roll [deg]",
            "Reduced mount pitch [deg]",
            "Reduced mount yaw [deg]",
            "Reduced mount quaternion error [deg]",
            "Synthetic truth mount roll [deg]",
            "Synthetic truth mount pitch [deg]",
            "Synthetic truth mount yaw [deg]",
        ],
    )?;
    require_trace_schema(
        "reduced_map",
        &data.reduced_map,
        &[
            "Synthetic truth path (lon,lat)",
            "Synthetic GNSS path (lon,lat)",
            "Reduced path (lon,lat)",
        ],
    )?;
    require_trace_schema(
        "full_cmp_att",
        &data.full_cmp_att,
        &[
            "Full roll [deg]",
            "Full pitch [deg]",
            "Full yaw [deg]",
            "Reference roll [deg]",
            "Reference pitch [deg]",
            "Reference yaw [deg]",
        ],
    )?;
    require_trace_schema(
        "reduced_bump_pitch_speed",
        &data.reduced_bump_pitch_speed,
        &["Reduced pitch [deg]", "vehicle speed [m/s]"],
    )?;
    require_trace_schema(
        "reduced_bump_diag",
        &data.reduced_bump_diag,
        &["Pitch HPF [deg]", "Pitch RMS EMA [deg]"],
    )?;
    require_trace_schema(
        "align_cmp_att",
        &data.align_cmp_att,
        &[
            "Align roll [deg]",
            "Align pitch [deg]",
            "Align yaw [deg]",
            "Reference mount roll [deg]",
            "Reference mount pitch [deg]",
            "Reference mount yaw [deg]",
        ],
    )?;

    let pos_n = require_trace("reduced_cmp_pos", &data.reduced_cmp_pos, "Reduced posN [m]")?;
    require_trace_points("reduced_cmp_pos", pos_n)?;
    let initial_state =
        sample_nearest_value(pos_n, 0.0).expect("Reduced posN trace should be sampled");
    assert!(
        initial_state.is_finite(),
        "sampled Reduced posN should be finite, got {initial_state}"
    );
    assert!(
        data.reduced_cmp_pos
            .iter()
            .any(|trace| !trace.points.is_empty()),
        "synthetic visualizer produced no Reduced position points"
    );
    assert!(
        data.reduced_map
            .iter()
            .any(|trace| !trace.points.is_empty()),
        "synthetic visualizer produced no map points"
    );
    Ok(())
}

#[test]
fn synthetic_symmetric_figure8_does_not_create_full_mount_roll_drift() -> Result<()> {
    let scenario = "\
initial lat=32 lon=120 alt=0 speed=0 yaw=0 pitch=0 roll=0
wait 60s
accelerate 0.6m/s^2 for 20s
wait 10s
repeat 8 {
    turn left 10dps for 36s
    turn right 10dps for 36s
}
brake 0.6666667m/s^2 for 18s
";
    let data = build_synthetic_plot_data(
        &SyntheticVisualizerConfig {
            motion_def: None,
            motion_label: "symmetric_figure8.scenario".to_string(),
            motion_text: Some(scenario.to_string()),
            noise_mode: SyntheticNoiseMode::Truth,
            disable_imu_noise: false,
            disable_gnss_noise: false,
            seed: 42,
            mount_rpy_deg: [5.0, -5.0, 5.0],
            imu_hz: 25.0,
            gnss_hz: 2.0,
            gnss_time_shift_ms: 0.0,
            early_vel_bias_ned_mps: [0.0; 3],
            early_fault_window_s: None,
        },
        MountSourceMode::Internal,
        FilterCompareConfig::default(),
        GnssOutageConfig::default(),
    )?;

    let full_roll = final_trace_value(require_trace(
        "full_misalignment",
        &data.full_misalignment,
        "Full residual mount roll [deg]",
    )?)?;
    assert!(
        (full_roll - 5.0).abs() < 0.02,
        "ideal symmetric figure-eight should not drive full mount roll: final={full_roll:.6}"
    );
    Ok(())
}

#[test]
fn synthetic_roll_excitation_makes_internal_mount_observable() -> Result<()> {
    let profile_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("motion_profiles/figure8_roll_excitation_30min.scenario");
    let data = build_synthetic_plot_data(
        &SyntheticVisualizerConfig {
            motion_def: Some(profile_path),
            motion_label: "figure8_roll_excitation_30min.scenario".to_string(),
            motion_text: None,
            noise_mode: SyntheticNoiseMode::Truth,
            disable_imu_noise: false,
            disable_gnss_noise: false,
            seed: 42,
            mount_rpy_deg: [5.0, -5.0, 5.0],
            imu_hz: 25.0,
            gnss_hz: 2.0,
            gnss_time_shift_ms: 0.0,
            early_vel_bias_ned_mps: [0.5, 0.0, 0.0],
            early_fault_window_s: Some((0.0, 360.0)),
        },
        MountSourceMode::Internal,
        FilterCompareConfig::default(),
        GnssOutageConfig::default(),
    )?;

    let reduced_mount_qerr = final_trace_value(require_trace(
        "reduced_misalignment",
        &data.reduced_misalignment,
        "Reduced mount quaternion error [deg]",
    )?)?;
    assert!(
        reduced_mount_qerr < 0.25,
        "roll excitation should make Reduced mount converge; qerr={reduced_mount_qerr:.6}"
    );
    let full_mount_qerr = final_trace_value(require_trace(
        "full_misalignment",
        &data.full_misalignment,
        "Full mount quaternion error [deg]",
    )?)?;
    assert!(
        full_mount_qerr < 0.25,
        "roll excitation should make full mount converge; qerr={full_mount_qerr:.6}"
    );
    Ok(())
}

fn assert_reduced_converges_on_profile(profile_name: &str) -> Result<()> {
    let profile_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("motion_profiles/{profile_name}"));
    let profile = MotionProfile::from_path(&profile_path)?;
    let generated = generate(&profile, PathGenConfig::default())?;
    let summary = run_reduced_on_generated_path(&generated, [5.0, -5.0, 5.0])?;

    assert!(
        summary.reduced_initialized,
        "Reduced did not initialize on generated synthetic data: {profile_name}"
    );
    assert!(
        summary.final_mount_quat_err_deg < 0.75,
        "mount quaternion error too high for {profile_name}: {summary:#?}"
    );
    assert!(
        summary.tail_mount_quat_err_mean_deg < 0.75,
        "tail mount quaternion mean error too high for {profile_name}: {summary:#?}"
    );
    assert!(
        summary.final_att_quat_err_deg < 0.5,
        "attitude quaternion error too high for {profile_name}: {summary:#?}"
    );
    assert!(
        summary.final_vel_err_mps < 0.35,
        "velocity error too high for {profile_name}: {summary:#?}"
    );
    assert!(
        summary.final_pos_err_m < 4.0,
        "position error too high for {profile_name}: {summary:#?}"
    );
    assert!(
        summary.final_gyro_bias_norm_dps < 0.35,
        "gyro bias estimate diverged for {profile_name}: {summary:#?}"
    );
    assert!(
        summary.final_accel_bias_norm_mps2 < 0.8,
        "accel bias estimate diverged for {profile_name}: {summary:#?}"
    );

    Ok(())
}

fn final_trace_value(trace: &sim::visualizer::model::Trace) -> Result<f64> {
    trace
        .points
        .last()
        .map(|point| point[1])
        .ok_or_else(|| anyhow::anyhow!("trace '{}' has no points", trace.name))
}

#[derive(Clone, Copy, Debug)]
struct ReducedSyntheticSummary {
    reduced_initialized: bool,
    final_mount_quat_err_deg: f64,
    tail_mount_quat_err_mean_deg: f64,
    final_att_quat_err_deg: f64,
    final_vel_err_mps: f64,
    final_pos_err_m: f64,
    final_gyro_bias_norm_dps: f64,
    final_accel_bias_norm_mps2: f64,
}

#[derive(Clone, Copy, Debug)]
struct StateErr {
    t_s: f64,
    mount_quat_err_deg: f64,
    att_quat_err_deg: f64,
    vel_err_mps: f64,
    pos_err_m: f64,
    gyro_bias_norm_dps: f64,
    accel_bias_norm_mps2: f64,
}

fn run_reduced_on_generated_path(
    generated: &sim::synthetic::gnss_ins_path::GeneratedPath,
    mount_rpy_deg: [f64; 3],
) -> Result<ReducedSyntheticSummary> {
    run_reduced_on_samples(
        generated,
        &generated.imu,
        &generated.gnss,
        mount_rpy_deg,
        [0.5, 0.5, 0.5],
        [0.2, 0.2, 0.2],
    )
}

fn run_reduced_on_samples(
    reference: &sim::synthetic::gnss_ins_path::GeneratedPath,
    imu_samples: &[sim::datasets::synthetic_replay::ImuSample],
    gnss_samples: &[sim::datasets::synthetic_replay::GnssSample],
    mount_rpy_deg: [f64; 3],
    pos_std_m: [f64; 3],
    vel_std_mps: [f64; 3],
) -> Result<ReducedSyntheticSummary> {
    run_reduced_on_samples_configured(
        reference,
        imu_samples,
        gnss_samples,
        mount_rpy_deg,
        pos_std_m,
        vel_std_mps,
        |_| {},
    )
}

fn run_reduced_on_samples_configured(
    reference: &sim::synthetic::gnss_ins_path::GeneratedPath,
    imu_samples: &[sim::datasets::synthetic_replay::ImuSample],
    gnss_samples: &[sim::datasets::synthetic_replay::GnssSample],
    mount_rpy_deg: [f64; 3],
    pos_std_m: [f64; 3],
    vel_std_mps: [f64; 3],
    configure: impl FnOnce(&mut SensorFusion),
) -> Result<ReducedSyntheticSummary> {
    let q_truth = reference_mount_rpy_to_q_vb(mount_rpy_deg);
    let imu = imu_samples
        .iter()
        .map(|s| GenericImuSample {
            t_s: s.t_s,
            gyro_radps: quat_rotate(q_truth, s.gyro_vehicle_radps),
            accel_mps2: quat_rotate(q_truth, s.accel_vehicle_mps2),
        })
        .collect::<Vec<_>>();
    let gnss = gnss_samples
        .iter()
        .map(|s| GenericGnssSample {
            t_s: s.t_s,
            lat_deg: s.lat_deg,
            lon_deg: s.lon_deg,
            height_m: s.height_m,
            vel_ned_mps: s.vel_ned_mps,
            pos_std_m,
            vel_std_mps,
            heading_rad: None,
        })
        .collect::<Vec<_>>();

    let mut fusion = SensorFusion::new();
    // This idealized convergence test exercises the IMU/GNSS formulation without
    // runtime stationary pseudo-measurements.
    fusion.set_r_zero_vel(0.0);
    configure(&mut fusion);
    let ref_ecef = lla_to_ecef(
        reference.truth[0].lat_deg,
        reference.truth[0].lon_deg,
        reference.truth[0].height_m,
    );
    let mut errors = Vec::new();

    for_each_event(&imu, &gnss, |event| match event {
        ReplayEvent::Imu(idx, sample) => {
            let _ = fusion.process_imu(fusion_imu_sample(*sample));
            if let Some(reduced) = fusion.reduced() {
                let truth = reference.truth[idx];
                let truth_ecef = lla_to_ecef(truth.lat_deg, truth.lon_deg, truth.height_m);
                let truth_pos_ned = ecef_to_ned(
                    truth_ecef,
                    ref_ecef,
                    reference.truth[0].lat_deg,
                    reference.truth[0].lon_deg,
                );
                let q_cs = as_q64([
                    reduced.nominal.qcs0,
                    reduced.nominal.qcs1,
                    reduced.nominal.qcs2,
                    reduced.nominal.qcs3,
                ]);
                let q_est_att = as_q64([
                    reduced.nominal.q0,
                    reduced.nominal.q1,
                    reduced.nominal.q2,
                    reduced.nominal.q3,
                ]);
                errors.push(StateErr {
                    t_s: sample.t_s,
                    mount_quat_err_deg: quat_angle_deg(q_cs, q_truth),
                    att_quat_err_deg: quat_angle_deg(q_est_att, truth.q_bn),
                    vel_err_mps: norm3([
                        reduced.nominal.vn as f64 - truth.vel_ned_mps[0],
                        reduced.nominal.ve as f64 - truth.vel_ned_mps[1],
                        reduced.nominal.vd as f64 - truth.vel_ned_mps[2],
                    ]),
                    pos_err_m: norm3([
                        reduced.nominal.pn as f64 - truth_pos_ned[0],
                        reduced.nominal.pe as f64 - truth_pos_ned[1],
                        reduced.nominal.pd as f64 - truth_pos_ned[2],
                    ]),
                    gyro_bias_norm_dps: norm3([
                        reduced.nominal.bgx as f64,
                        reduced.nominal.bgy as f64,
                        reduced.nominal.bgz as f64,
                    ])
                    .to_degrees(),
                    accel_bias_norm_mps2: norm3([
                        reduced.nominal.bax as f64,
                        reduced.nominal.bay as f64,
                        reduced.nominal.baz as f64,
                    ]),
                });
            }
        }
        ReplayEvent::Gnss(_, sample) => {
            let _ = fusion.process_gnss(fusion_gnss_sample(*sample));
        }
    });

    let Some(final_err) = errors.last().copied() else {
        bail!("Reduced produced no state samples");
    };
    let tail_start = (final_err.t_s - 60.0).max(0.0);
    let tail = errors
        .iter()
        .filter(|e| e.t_s >= tail_start)
        .collect::<Vec<_>>();
    let tail_mount_quat_err_mean_deg =
        tail.iter().map(|e| e.mount_quat_err_deg).sum::<f64>() / tail.len() as f64;

    Ok(ReducedSyntheticSummary {
        reduced_initialized: fusion.reduced().is_some(),
        final_mount_quat_err_deg: final_err.mount_quat_err_deg,
        tail_mount_quat_err_mean_deg,
        final_att_quat_err_deg: final_err.att_quat_err_deg,
        final_vel_err_mps: final_err.vel_err_mps,
        final_pos_err_m: final_err.pos_err_m,
        final_gyro_bias_norm_dps: final_err.gyro_bias_norm_dps,
        final_accel_bias_norm_mps2: final_err.accel_bias_norm_mps2,
    })
}

fn gnss_with_early_velocity_bias(
    gnss: &[sim::datasets::synthetic_replay::GnssSample],
    bias_ned_mps: [f64; 3],
    end_t_s: f64,
) -> Vec<sim::datasets::synthetic_replay::GnssSample> {
    gnss.iter()
        .map(|sample| {
            let mut sample = *sample;
            if sample.t_s <= end_t_s {
                for (velocity, bias) in sample.vel_ned_mps.iter_mut().zip(bias_ned_mps) {
                    *velocity += bias;
                }
            }
            sample
        })
        .collect()
}

fn assert_vec_close(label: &str, actual: [f64; 3], expected: [f64; 3], tol: f64) {
    for i in 0..3 {
        assert_close(&format!("{label}[{i}]"), actual[i], expected[i], tol);
    }
}

fn assert_quat_close(label: &str, actual: [f64; 4], expected: [f64; 4], tol: f64) {
    let sign = if actual.iter().zip(expected).map(|(a, b)| a * b).sum::<f64>() < 0.0 {
        -1.0
    } else {
        1.0
    };
    for i in 0..4 {
        assert_close(&format!("{label}[{i}]"), actual[i] * sign, expected[i], tol);
    }
}

fn assert_close(label: &str, actual: f64, expected: f64, tol: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "{label}: actual={actual:.17e} expected={expected:.17e} diff={diff:.3e} tol={tol:.3e}"
    );
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn f64_at(row: &[serde_json::Value], idx: usize) -> f64 {
    row[idx].as_f64().expect("numeric value")
}
