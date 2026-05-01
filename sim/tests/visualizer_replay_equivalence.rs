use std::fmt::Write as _;

use anyhow::{Context, Result, bail};
use sim::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, GenericReferenceRpySample,
};
use sim::eval::gnss_ins::quat_rotate;
use sim::synthetic::gnss_ins_path::{
    GpsNoiseModel, MeasurementNoiseConfig, MotionProfile, PathGenConfig, generate_with_noise,
};
use sim::visualizer::math::quat_rpy_deg;
use sim::visualizer::model::{EkfImuSource, HeadingSample, PlotData, Trace};
use sim::visualizer::pipeline::generic::{GenericReplayInput, reference_mount_rpy_to_q_vb};
use sim::visualizer::pipeline::synthetic::{
    SyntheticNoiseMode, SyntheticVisualizerConfig, build_synthetic_plot_data,
};
use sim::visualizer::pipeline::{EkfCompareConfig, GnssOutageConfig};
use sim::visualizer::replay_job::{
    GenericReplayCsvJob, GenericReplayJobConfig, run_generic_csv_replay_job, run_generic_replay_job,
};

const EQUIVALENCE_SCENARIO: &str = "\
initial lat=32 lon=120 alt=20 speed=0 yaw=0 pitch=0 roll=0
wait 2s
accelerate 1.0m/s^2 for 4s
hold 4s
turn left 15dps for 4s
hold 4s
turn right 15dps for 4s
brake 1.0m/s^2 for 4s
hold 2s
";

const MOUNT_RPY_DEG: [f64; 3] = [3.0, -2.0, 1.5];
const IMU_HZ: f64 = 25.0;
const GNSS_HZ: f64 = 2.0;
const POINT_TOL: f64 = 1.0e-10;

#[test]
fn generic_replay_input_and_csv_replay_jobs_produce_matching_plot_data() -> Result<()> {
    let replay = generated_generic_replay()?;
    let config = replay_config();

    let direct = run_generic_replay_job(&replay, config);
    let csv = run_generic_csv_replay_job(GenericReplayCsvJob {
        imu_csv: &imu_csv(&replay.imu),
        gnss_csv: &gnss_csv(&replay.gnss),
        reference_attitude_csv: Some(&reference_rpy_csv(&replay.reference_attitude)),
        reference_mount_csv: Some(&reference_rpy_csv(&replay.reference_mount)),
        config,
    })?;

    assert_plot_data_close("direct GenericReplayInput", &direct, "CSV replay", &csv)?;
    Ok(())
}

#[test]
fn synthetic_generic_input_and_csv_replay_share_auxiliary_trace_outputs() -> Result<()> {
    let replay = generated_generic_replay()?;
    let config = replay_config();
    let direct = run_generic_replay_job(&replay, config);
    let csv = run_generic_csv_replay_job(GenericReplayCsvJob {
        imu_csv: &imu_csv(&replay.imu),
        gnss_csv: &gnss_csv(&replay.gnss),
        reference_attitude_csv: Some(&reference_rpy_csv(&replay.reference_attitude)),
        reference_mount_csv: Some(&reference_rpy_csv(&replay.reference_mount)),
        config,
    })?;
    let synthetic = build_synthetic_plot_data(
        &SyntheticVisualizerConfig {
            motion_def: None,
            motion_label: "equivalence.scenario".to_string(),
            motion_text: Some(EQUIVALENCE_SCENARIO.to_string()),
            noise_mode: SyntheticNoiseMode::Truth,
            disable_imu_noise: false,
            disable_gnss_noise: false,
            seed: 7,
            mount_rpy_deg: MOUNT_RPY_DEG,
            imu_hz: IMU_HZ,
            gnss_hz: GNSS_HZ,
            gnss_time_shift_ms: 0.0,
            early_vel_bias_ned_mps: [0.0; 3],
            early_fault_window_s: None,
        },
        EkfImuSource::Internal,
        EkfCompareConfig::default(),
        GnssOutageConfig::default(),
    )?;

    assert_shared_auxiliary_groups_close("synthetic", &synthetic, "direct", &direct)?;
    assert_shared_auxiliary_groups_close("synthetic", &synthetic, "CSV", &csv)?;
    Ok(())
}

fn generated_generic_replay() -> Result<GenericReplayInput> {
    let profile = MotionProfile::from_dsl_str(EQUIVALENCE_SCENARIO)?;
    let noise = MeasurementNoiseConfig::zero();
    let measured = generate_with_noise(
        &profile,
        PathGenConfig {
            imu_hz: IMU_HZ,
            gnss_hz: GNSS_HZ,
            ..PathGenConfig::default()
        },
        noise,
        7,
    )?;
    let q_truth_mount = reference_mount_rpy_to_q_vb(MOUNT_RPY_DEG);
    let gps_noise = noise.gps.unwrap_or(GpsNoiseModel {
        pos_std_m: [0.5, 0.5, 0.5],
        vel_std_mps: [0.2, 0.2, 0.2],
    });
    let imu = measured
        .imu
        .iter()
        .map(|sample| GenericImuSample {
            t_s: sample.t_s + 1.0 / IMU_HZ,
            gyro_radps: quat_rotate(q_truth_mount, sample.gyro_vehicle_radps),
            accel_mps2: quat_rotate(q_truth_mount, sample.accel_vehicle_mps2),
        })
        .collect::<Vec<_>>();
    let gnss = measured
        .gnss
        .iter()
        .map(|sample| GenericGnssSample {
            t_s: sample.t_s,
            lat_deg: sample.lat_deg,
            lon_deg: sample.lon_deg,
            height_m: sample.height_m,
            vel_ned_mps: sample.vel_ned_mps,
            pos_std_m: gps_noise.pos_std_m,
            vel_std_mps: gps_noise.vel_std_mps,
            heading_rad: None,
        })
        .collect::<Vec<_>>();
    let reference_attitude = measured
        .reference
        .truth
        .iter()
        .map(|truth| {
            let (roll_deg, pitch_deg, yaw_deg) = quat_rpy_deg(
                truth.q_bn[0] as f32,
                truth.q_bn[1] as f32,
                truth.q_bn[2] as f32,
                truth.q_bn[3] as f32,
            );
            GenericReferenceRpySample {
                t_s: truth.t_s,
                roll_deg,
                pitch_deg,
                yaw_deg,
            }
        })
        .collect::<Vec<_>>();
    let end_t_s = measured
        .reference
        .truth
        .last()
        .map(|sample| sample.t_s)
        .context("generated truth should not be empty")?;
    let reference_mount = vec![
        GenericReferenceRpySample {
            t_s: 0.0,
            roll_deg: MOUNT_RPY_DEG[0],
            pitch_deg: MOUNT_RPY_DEG[1],
            yaw_deg: MOUNT_RPY_DEG[2],
        },
        GenericReferenceRpySample {
            t_s: end_t_s,
            roll_deg: MOUNT_RPY_DEG[0],
            pitch_deg: MOUNT_RPY_DEG[1],
            yaw_deg: MOUNT_RPY_DEG[2],
        },
    ];

    Ok(GenericReplayInput {
        imu,
        gnss,
        reference_attitude,
        reference_mount,
    })
}

fn replay_config() -> GenericReplayJobConfig {
    GenericReplayJobConfig::full(
        EkfImuSource::Internal,
        EkfCompareConfig::default(),
        GnssOutageConfig::default(),
    )
}

fn imu_csv(samples: &[GenericImuSample]) -> String {
    let mut csv = "t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2\n".to_string();
    for sample in samples {
        writeln!(
            csv,
            "{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}",
            sample.t_s,
            sample.gyro_radps[0],
            sample.gyro_radps[1],
            sample.gyro_radps[2],
            sample.accel_mps2[0],
            sample.accel_mps2[1],
            sample.accel_mps2[2]
        )
        .expect("writing to String should not fail");
    }
    csv
}

fn gnss_csv(samples: &[GenericGnssSample]) -> String {
    let mut csv = "t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,pos_std_n_m,pos_std_e_m,pos_std_d_m,vel_std_n_mps,vel_std_e_mps,vel_std_d_mps,heading_rad\n".to_string();
    for sample in samples {
        writeln!(
            csv,
            "{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{}",
            sample.t_s,
            sample.lat_deg,
            sample.lon_deg,
            sample.height_m,
            sample.vel_ned_mps[0],
            sample.vel_ned_mps[1],
            sample.vel_ned_mps[2],
            sample.pos_std_m[0],
            sample.pos_std_m[1],
            sample.pos_std_m[2],
            sample.vel_std_mps[0],
            sample.vel_std_mps[1],
            sample.vel_std_mps[2],
            sample
                .heading_rad
                .map(|heading| format!("{heading:.17}"))
                .unwrap_or_else(|| "nan".to_string())
        )
        .expect("writing to String should not fail");
    }
    csv
}

fn reference_rpy_csv(samples: &[GenericReferenceRpySample]) -> String {
    let mut csv = "t_s,roll_deg,pitch_deg,yaw_deg\n".to_string();
    for sample in samples {
        writeln!(
            csv,
            "{:.17},{:.17},{:.17},{:.17}",
            sample.t_s, sample.roll_deg, sample.pitch_deg, sample.yaw_deg
        )
        .expect("writing to String should not fail");
    }
    csv
}

fn assert_plot_data_close(
    left_label: &str,
    left: &PlotData,
    right_label: &str,
    right: &PlotData,
) -> Result<()> {
    let trace_groups = [
        ("speed", &left.speed, &right.speed),
        ("sat_cn0", &left.sat_cn0, &right.sat_cn0),
        ("imu_raw_gyro", &left.imu_raw_gyro, &right.imu_raw_gyro),
        ("imu_raw_accel", &left.imu_raw_accel, &right.imu_raw_accel),
        ("imu_cal_gyro", &left.imu_cal_gyro, &right.imu_cal_gyro),
        ("imu_cal_accel", &left.imu_cal_accel, &right.imu_cal_accel),
        ("orientation", &left.orientation, &right.orientation),
        ("other", &left.other, &right.other),
        ("eskf_cmp_pos", &left.eskf_cmp_pos, &right.eskf_cmp_pos),
        ("eskf_cmp_vel", &left.eskf_cmp_vel, &right.eskf_cmp_vel),
        ("eskf_cmp_att", &left.eskf_cmp_att, &right.eskf_cmp_att),
        (
            "eskf_meas_gyro",
            &left.eskf_meas_gyro,
            &right.eskf_meas_gyro,
        ),
        (
            "eskf_meas_accel",
            &left.eskf_meas_accel,
            &right.eskf_meas_accel,
        ),
        (
            "eskf_bias_gyro",
            &left.eskf_bias_gyro,
            &right.eskf_bias_gyro,
        ),
        (
            "eskf_bias_accel",
            &left.eskf_bias_accel,
            &right.eskf_bias_accel,
        ),
        ("eskf_cov_bias", &left.eskf_cov_bias, &right.eskf_cov_bias),
        (
            "eskf_cov_nonbias",
            &left.eskf_cov_nonbias,
            &right.eskf_cov_nonbias,
        ),
        (
            "eskf_mount_sigma",
            &left.eskf_mount_sigma,
            &right.eskf_mount_sigma,
        ),
        ("eskf_mount_dx", &left.eskf_mount_dx, &right.eskf_mount_dx),
        (
            "eskf_nhc_mount_dx",
            &left.eskf_nhc_mount_dx,
            &right.eskf_nhc_mount_dx,
        ),
        (
            "eskf_nhc_innovation",
            &left.eskf_nhc_innovation,
            &right.eskf_nhc_innovation,
        ),
        ("eskf_nhc_nis", &left.eskf_nhc_nis, &right.eskf_nhc_nis),
        (
            "eskf_nhc_h_mount_norm",
            &left.eskf_nhc_h_mount_norm,
            &right.eskf_nhc_h_mount_norm,
        ),
        (
            "eskf_misalignment",
            &left.eskf_misalignment,
            &right.eskf_misalignment,
        ),
        (
            "eskf_stationary_diag",
            &left.eskf_stationary_diag,
            &right.eskf_stationary_diag,
        ),
        (
            "eskf_bump_pitch_speed",
            &left.eskf_bump_pitch_speed,
            &right.eskf_bump_pitch_speed,
        ),
        (
            "eskf_bump_diag",
            &left.eskf_bump_diag,
            &right.eskf_bump_diag,
        ),
        ("eskf_map", &left.eskf_map, &right.eskf_map),
        ("loose_cmp_pos", &left.loose_cmp_pos, &right.loose_cmp_pos),
        ("loose_cmp_vel", &left.loose_cmp_vel, &right.loose_cmp_vel),
        ("loose_cmp_att", &left.loose_cmp_att, &right.loose_cmp_att),
        (
            "loose_nominal_att",
            &left.loose_nominal_att,
            &right.loose_nominal_att,
        ),
        (
            "loose_residual_mount",
            &left.loose_residual_mount,
            &right.loose_residual_mount,
        ),
        (
            "loose_misalignment",
            &left.loose_misalignment,
            &right.loose_misalignment,
        ),
        (
            "loose_meas_gyro",
            &left.loose_meas_gyro,
            &right.loose_meas_gyro,
        ),
        (
            "loose_meas_accel",
            &left.loose_meas_accel,
            &right.loose_meas_accel,
        ),
        (
            "loose_bias_gyro",
            &left.loose_bias_gyro,
            &right.loose_bias_gyro,
        ),
        (
            "loose_bias_accel",
            &left.loose_bias_accel,
            &right.loose_bias_accel,
        ),
        (
            "loose_scale_gyro",
            &left.loose_scale_gyro,
            &right.loose_scale_gyro,
        ),
        (
            "loose_scale_accel",
            &left.loose_scale_accel,
            &right.loose_scale_accel,
        ),
        (
            "loose_cov_bias",
            &left.loose_cov_bias,
            &right.loose_cov_bias,
        ),
        (
            "loose_cov_nonbias",
            &left.loose_cov_nonbias,
            &right.loose_cov_nonbias,
        ),
        (
            "loose_mount_sigma",
            &left.loose_mount_sigma,
            &right.loose_mount_sigma,
        ),
        (
            "loose_mount_dx",
            &left.loose_mount_dx,
            &right.loose_mount_dx,
        ),
        ("loose_map", &left.loose_map, &right.loose_map),
        ("align_cmp_att", &left.align_cmp_att, &right.align_cmp_att),
        ("align_res_vel", &left.align_res_vel, &right.align_res_vel),
        (
            "align_axis_err",
            &left.align_axis_err,
            &right.align_axis_err,
        ),
        ("align_motion", &left.align_motion, &right.align_motion),
        ("align_flags", &left.align_flags, &right.align_flags),
        (
            "align_roll_contrib",
            &left.align_roll_contrib,
            &right.align_roll_contrib,
        ),
        (
            "align_pitch_contrib",
            &left.align_pitch_contrib,
            &right.align_pitch_contrib,
        ),
        (
            "align_yaw_contrib",
            &left.align_yaw_contrib,
            &right.align_yaw_contrib,
        ),
        ("align_cov", &left.align_cov, &right.align_cov),
    ];
    for (group, left_traces, right_traces) in trace_groups {
        assert_trace_groups_close(left_label, right_label, group, left_traces, right_traces)?;
    }
    assert_heading_samples_close(
        left_label,
        right_label,
        "eskf_map_heading",
        &left.eskf_map_heading,
        &right.eskf_map_heading,
    )?;
    assert_heading_samples_close(
        left_label,
        right_label,
        "loose_map_heading",
        &left.loose_map_heading,
        &right.loose_map_heading,
    )?;
    Ok(())
}

fn assert_shared_auxiliary_groups_close(
    left_label: &str,
    left: &PlotData,
    right_label: &str,
    right: &PlotData,
) -> Result<()> {
    let groups = [
        (
            "align_cmp_att",
            &left.align_cmp_att,
            &right.align_cmp_att,
            &[
                "Align roll [deg]",
                "Align pitch [deg]",
                "Align yaw [deg]",
                "Reference mount roll [deg]",
                "Reference mount pitch [deg]",
                "Reference mount yaw [deg]",
            ][..],
        ),
        (
            "align_axis_err",
            &left.align_axis_err,
            &right.align_axis_err,
            &[
                "Align roll error [deg]",
                "Align pitch error [deg]",
                "Align yaw error [deg]",
            ],
        ),
        (
            "align_res_vel",
            &left.align_res_vel,
            &right.align_res_vel,
            &[
                "Window speed quality proxy [m/s]",
                "Mean gyro norm [rad/s]",
                "Mean accel norm [m/s^2]",
                "Horizontal heading innovation [deg]",
                "GNSS horizontal accel norm [m/s^2]",
                "IMU horizontal accel norm [m/s^2]",
            ],
        ),
        (
            "align_flags",
            &left.align_flags,
            &right.align_flags,
            &[
                "straight window accepted",
                "turn window accepted",
                "coarse alignment ready",
            ],
        ),
        (
            "align_roll_contrib",
            &left.align_roll_contrib,
            &right.align_roll_contrib,
            &[
                "gravity roll update [deg]",
                "horizontal accel roll update [deg]",
                "turn gyro roll update [deg]",
            ],
        ),
        (
            "align_pitch_contrib",
            &left.align_pitch_contrib,
            &right.align_pitch_contrib,
            &[
                "gravity pitch update [deg]",
                "horizontal accel pitch update [deg]",
                "turn gyro pitch update [deg]",
            ],
        ),
        (
            "align_yaw_contrib",
            &left.align_yaw_contrib,
            &right.align_yaw_contrib,
            &[
                "gravity yaw update [deg]",
                "horizontal accel yaw update [deg]",
                "turn gyro yaw update [deg]",
            ],
        ),
        (
            "align_cov",
            &left.align_cov,
            &right.align_cov,
            &[
                "Align roll sigma [deg]",
                "Align pitch sigma [deg]",
                "Align yaw sigma [deg]",
            ],
        ),
        (
            "loose_cmp_pos",
            &left.loose_cmp_pos,
            &right.loose_cmp_pos,
            &["Loose posN [m]", "Loose posE [m]", "Loose posD [m]"],
        ),
        (
            "loose_cmp_vel",
            &left.loose_cmp_vel,
            &right.loose_cmp_vel,
            &["Loose velN [m/s]", "Loose velE [m/s]", "Loose velD [m/s]"],
        ),
        (
            "loose_cmp_att",
            &left.loose_cmp_att,
            &right.loose_cmp_att,
            &[
                "Loose roll [deg]",
                "Loose pitch [deg]",
                "Loose yaw [deg]",
                "Reference roll [deg]",
                "Reference pitch [deg]",
                "Reference yaw [deg]",
            ],
        ),
        (
            "loose_misalignment",
            &left.loose_misalignment,
            &right.loose_misalignment,
            &[
                "Loose residual mount roll [deg]",
                "Loose residual mount pitch [deg]",
                "Loose residual mount yaw [deg]",
            ],
        ),
        (
            "loose_meas_gyro",
            &left.loose_meas_gyro,
            &right.loose_meas_gyro,
            &[
                "Loose gyro x [deg/s]",
                "Loose gyro y [deg/s]",
                "Loose gyro z [deg/s]",
            ],
        ),
        (
            "loose_meas_accel",
            &left.loose_meas_accel,
            &right.loose_meas_accel,
            &[
                "Loose accel x [m/s^2]",
                "Loose accel y [m/s^2]",
                "Loose accel z [m/s^2]",
            ],
        ),
        (
            "loose_bias_gyro",
            &left.loose_bias_gyro,
            &right.loose_bias_gyro,
            &[
                "Loose gyro sensor bias X [deg/s]",
                "Loose gyro sensor bias Y [deg/s]",
                "Loose gyro sensor bias Z [deg/s]",
            ],
        ),
        (
            "loose_bias_accel",
            &left.loose_bias_accel,
            &right.loose_bias_accel,
            &[
                "Loose accel sensor bias X [m/s^2]",
                "Loose accel sensor bias Y [m/s^2]",
                "Loose accel sensor bias Z [m/s^2]",
            ],
        ),
        (
            "loose_scale_gyro",
            &left.loose_scale_gyro,
            &right.loose_scale_gyro,
            &["Loose sgx", "Loose sgy", "Loose sgz"],
        ),
        (
            "loose_scale_accel",
            &left.loose_scale_accel,
            &right.loose_scale_accel,
            &["Loose sax", "Loose say", "Loose saz"],
        ),
        (
            "loose_map",
            &left.loose_map,
            &right.loose_map,
            &["Loose path (lon,lat)"],
        ),
    ];

    for (group, left_traces, right_traces, names) in groups {
        assert_named_traces_close(
            left_label,
            right_label,
            group,
            left_traces,
            right_traces,
            names,
        )?;
    }
    assert_numbered_named_traces_close(
        left_label,
        right_label,
        "loose_cov_bias",
        &left.loose_cov_bias,
        &right.loose_cov_bias,
        "Loose sigma bias/scale ",
    )?;
    assert_numbered_named_traces_close(
        left_label,
        right_label,
        "loose_cov_nonbias",
        &left.loose_cov_nonbias,
        &right.loose_cov_nonbias,
        "Loose sigma state ",
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "eskf_mount_dx",
        &left.eskf_mount_dx,
        &right.eskf_mount_dx,
        &[
            "ESKF mount roll correction [deg/update]",
            "ESKF mount pitch correction [deg/update]",
            "ESKF mount yaw correction [deg/update]",
        ],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "eskf_nhc_mount_dx",
        &left.eskf_nhc_mount_dx,
        &right.eskf_nhc_mount_dx,
        &[
            "ESKF NHC Y mount roll correction [deg/update]",
            "ESKF NHC Y mount pitch correction [deg/update]",
            "ESKF NHC Y mount yaw correction [deg/update]",
            "ESKF NHC Z mount roll correction [deg/update]",
            "ESKF NHC Z mount pitch correction [deg/update]",
            "ESKF NHC Z mount yaw correction [deg/update]",
        ],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "eskf_nhc_innovation",
        &left.eskf_nhc_innovation,
        &right.eskf_nhc_innovation,
        &["ESKF NHC Y innovation [m/s]", "ESKF NHC Z innovation [m/s]"],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "eskf_nhc_nis",
        &left.eskf_nhc_nis,
        &right.eskf_nhc_nis,
        &["ESKF NHC Y NIS", "ESKF NHC Z NIS"],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "eskf_nhc_h_mount_norm",
        &left.eskf_nhc_h_mount_norm,
        &right.eskf_nhc_h_mount_norm,
        &["ESKF NHC Y mount H norm", "ESKF NHC Z mount H norm"],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "loose_mount_sigma",
        &left.loose_mount_sigma,
        &right.loose_mount_sigma,
        &[
            "Loose residual mount roll sigma [deg]",
            "Loose residual mount pitch sigma [deg]",
            "Loose residual mount yaw sigma [deg]",
        ],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "loose_mount_dx",
        &left.loose_mount_dx,
        &right.loose_mount_dx,
        &[
            "Loose residual mount roll correction [deg/update]",
            "Loose residual mount pitch correction [deg/update]",
            "Loose residual mount yaw correction [deg/update]",
        ],
    )?;
    assert_heading_samples_close(
        left_label,
        right_label,
        "loose_map_heading",
        &left.loose_map_heading,
        &right.loose_map_heading,
    )?;
    Ok(())
}

fn assert_trace_groups_close(
    left_label: &str,
    right_label: &str,
    group: &str,
    left: &[Trace],
    right: &[Trace],
) -> Result<()> {
    if left.len() != right.len() {
        bail!(
            "{group}: {left_label} produced {} traces, {right_label} produced {}",
            left.len(),
            right.len()
        );
    }
    for (idx, (left_trace, right_trace)) in left.iter().zip(right).enumerate() {
        if left_trace.name != right_trace.name {
            bail!(
                "{group}[{idx}]: {left_label} trace '{}' != {right_label} trace '{}'",
                left_trace.name,
                right_trace.name
            );
        }
        assert_trace_points_close(left_label, right_label, group, left_trace, right_trace)?;
    }
    Ok(())
}

fn assert_named_traces_close(
    left_label: &str,
    right_label: &str,
    group: &str,
    left: &[Trace],
    right: &[Trace],
    names: &[&str],
) -> Result<()> {
    for name in names {
        let left_trace = find_trace(left, name)
            .with_context(|| format!("{group}: {left_label} missing shared trace '{name}'"))?;
        let right_trace = find_trace(right, name)
            .with_context(|| format!("{group}: {right_label} missing shared trace '{name}'"))?;
        assert_trace_points_close(left_label, right_label, group, left_trace, right_trace)?;
    }
    Ok(())
}

fn assert_numbered_named_traces_close(
    left_label: &str,
    right_label: &str,
    group: &str,
    left: &[Trace],
    right: &[Trace],
    prefix: &str,
) -> Result<()> {
    let names = left
        .iter()
        .filter_map(|trace| trace.name.strip_prefix(prefix).map(|_| trace.name.as_str()))
        .collect::<Vec<_>>();
    if names.is_empty() {
        bail!("{group}: {left_label} produced no traces with prefix '{prefix}'");
    }
    assert_named_traces_close(left_label, right_label, group, left, right, &names)
}

fn assert_trace_points_close(
    left_label: &str,
    right_label: &str,
    group: &str,
    left: &Trace,
    right: &Trace,
) -> Result<()> {
    if left.points.len() != right.points.len() {
        bail!(
            "{group}/{}: {left_label} produced {} points, {right_label} produced {}",
            left.name,
            left.points.len(),
            right.points.len()
        );
    }
    for (idx, (left_point, right_point)) in left.points.iter().zip(&right.points).enumerate() {
        assert_close(
            &format!("{group}/{}[{idx}].t", left.name),
            left_point[0],
            right_point[0],
        )?;
        assert_close(
            &format!("{group}/{}[{idx}].y", left.name),
            left_point[1],
            right_point[1],
        )?;
    }
    Ok(())
}

fn assert_heading_samples_close(
    left_label: &str,
    right_label: &str,
    group: &str,
    left: &[HeadingSample],
    right: &[HeadingSample],
) -> Result<()> {
    if left.len() != right.len() {
        bail!(
            "{group}: {left_label} produced {} samples, {right_label} produced {}",
            left.len(),
            right.len()
        );
    }
    for (idx, (left_sample, right_sample)) in left.iter().zip(right).enumerate() {
        assert_close(
            &format!("{group}[{idx}].t_s"),
            left_sample.t_s,
            right_sample.t_s,
        )?;
        assert_close(
            &format!("{group}[{idx}].lon_deg"),
            left_sample.lon_deg,
            right_sample.lon_deg,
        )?;
        assert_close(
            &format!("{group}[{idx}].lat_deg"),
            left_sample.lat_deg,
            right_sample.lat_deg,
        )?;
        assert_close(
            &format!("{group}[{idx}].yaw_deg"),
            left_sample.yaw_deg,
            right_sample.yaw_deg,
        )?;
    }
    Ok(())
}

fn assert_close(label: &str, left: f64, right: f64) -> Result<()> {
    if left.is_nan() && right.is_nan() {
        return Ok(());
    }
    if (left - right).abs() <= POINT_TOL {
        return Ok(());
    }
    bail!("{label}: {left} != {right} within {POINT_TOL}");
}

fn find_trace<'a>(traces: &'a [Trace], name: &str) -> Option<&'a Trace> {
    traces.iter().find(|trace| trace.name == name)
}
