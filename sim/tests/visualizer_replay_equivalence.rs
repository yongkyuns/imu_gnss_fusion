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
use sim::visualizer::model::{HeadingSample, PlotData, Trace, VisualizerMountMode};
use sim::visualizer::pipeline::generic::{GenericReplayInput, reference_mount_rpy_to_q_bv};
use sim::visualizer::pipeline::synthetic::{
    SyntheticNoiseMode, SyntheticVisualizerConfig, build_synthetic_plot_data,
};
use sim::visualizer::pipeline::{FilterCompareConfig, GnssOutageConfig};
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
        reference_position_csv: None,
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
        reference_position_csv: None,
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
        VisualizerMountMode::Auto,
        FilterCompareConfig::default(),
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
    let q_truth_mount = reference_mount_rpy_to_q_bv(MOUNT_RPY_DEG);
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
        reference_position: Vec::new(),
    })
}

fn replay_config() -> GenericReplayJobConfig {
    GenericReplayJobConfig::full(
        VisualizerMountMode::Auto,
        FilterCompareConfig::default(),
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
        (
            "reduced_cmp_pos",
            &left.reduced_cmp_pos,
            &right.reduced_cmp_pos,
        ),
        (
            "reduced_cmp_vel",
            &left.reduced_cmp_vel,
            &right.reduced_cmp_vel,
        ),
        (
            "reduced_cmp_att",
            &left.reduced_cmp_att,
            &right.reduced_cmp_att,
        ),
        (
            "reduced_meas_gyro",
            &left.reduced_meas_gyro,
            &right.reduced_meas_gyro,
        ),
        (
            "reduced_meas_accel",
            &left.reduced_meas_accel,
            &right.reduced_meas_accel,
        ),
        (
            "reduced_bias_gyro",
            &left.reduced_bias_gyro,
            &right.reduced_bias_gyro,
        ),
        (
            "reduced_bias_accel",
            &left.reduced_bias_accel,
            &right.reduced_bias_accel,
        ),
        (
            "reduced_cov_bias",
            &left.reduced_cov_bias,
            &right.reduced_cov_bias,
        ),
        (
            "reduced_cov_nonbias",
            &left.reduced_cov_nonbias,
            &right.reduced_cov_nonbias,
        ),
        (
            "reduced_mount_sigma",
            &left.reduced_mount_sigma,
            &right.reduced_mount_sigma,
        ),
        (
            "reduced_mount_dx",
            &left.reduced_mount_dx,
            &right.reduced_mount_dx,
        ),
        (
            "reduced_nhc_mount_dx",
            &left.reduced_nhc_mount_dx,
            &right.reduced_nhc_mount_dx,
        ),
        (
            "reduced_nhc_innovation",
            &left.reduced_nhc_innovation,
            &right.reduced_nhc_innovation,
        ),
        (
            "reduced_nhc_nis",
            &left.reduced_nhc_nis,
            &right.reduced_nhc_nis,
        ),
        (
            "reduced_nhc_h_mount_norm",
            &left.reduced_nhc_h_mount_norm,
            &right.reduced_nhc_h_mount_norm,
        ),
        (
            "reduced_misalignment",
            &left.reduced_misalignment,
            &right.reduced_misalignment,
        ),
        (
            "reduced_stationary_diag",
            &left.reduced_stationary_diag,
            &right.reduced_stationary_diag,
        ),
        (
            "reduced_bump_pitch_speed",
            &left.reduced_bump_pitch_speed,
            &right.reduced_bump_pitch_speed,
        ),
        (
            "reduced_bump_diag",
            &left.reduced_bump_diag,
            &right.reduced_bump_diag,
        ),
        ("reduced_map", &left.reduced_map, &right.reduced_map),
        ("full_cmp_pos", &left.full_cmp_pos, &right.full_cmp_pos),
        ("full_cmp_vel", &left.full_cmp_vel, &right.full_cmp_vel),
        ("full_cmp_att", &left.full_cmp_att, &right.full_cmp_att),
        (
            "full_nominal_att",
            &left.full_nominal_att,
            &right.full_nominal_att,
        ),
        ("full_mount", &left.full_mount, &right.full_mount),
        (
            "full_misalignment",
            &left.full_misalignment,
            &right.full_misalignment,
        ),
        (
            "full_meas_gyro",
            &left.full_meas_gyro,
            &right.full_meas_gyro,
        ),
        (
            "full_meas_accel",
            &left.full_meas_accel,
            &right.full_meas_accel,
        ),
        (
            "full_bias_gyro",
            &left.full_bias_gyro,
            &right.full_bias_gyro,
        ),
        (
            "full_bias_accel",
            &left.full_bias_accel,
            &right.full_bias_accel,
        ),
        (
            "full_scale_gyro",
            &left.full_scale_gyro,
            &right.full_scale_gyro,
        ),
        (
            "full_scale_accel",
            &left.full_scale_accel,
            &right.full_scale_accel,
        ),
        ("full_cov_bias", &left.full_cov_bias, &right.full_cov_bias),
        (
            "full_cov_nonbias",
            &left.full_cov_nonbias,
            &right.full_cov_nonbias,
        ),
        (
            "full_mount_sigma",
            &left.full_mount_sigma,
            &right.full_mount_sigma,
        ),
        ("full_mount_dx", &left.full_mount_dx, &right.full_mount_dx),
        (
            "full_gnss_pos_gate",
            &left.full_gnss_pos_gate,
            &right.full_gnss_pos_gate,
        ),
        ("full_map", &left.full_map, &right.full_map),
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
        "reduced_map_heading",
        &left.reduced_map_heading,
        &right.reduced_map_heading,
    )?;
    assert_heading_samples_close(
        left_label,
        right_label,
        "full_map_heading",
        &left.full_map_heading,
        &right.full_map_heading,
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
            "full_cmp_pos",
            &left.full_cmp_pos,
            &right.full_cmp_pos,
            &["Full posN [m]", "Full posE [m]", "Full posD [m]"],
        ),
        (
            "full_cmp_vel",
            &left.full_cmp_vel,
            &right.full_cmp_vel,
            &["Full velN [m/s]", "Full velE [m/s]", "Full velD [m/s]"],
        ),
        (
            "full_cmp_att",
            &left.full_cmp_att,
            &right.full_cmp_att,
            &[
                "Full roll [deg]",
                "Full pitch [deg]",
                "Full yaw [deg]",
                "Reference roll [deg]",
                "Reference pitch [deg]",
                "Reference yaw [deg]",
            ],
        ),
        (
            "full_misalignment",
            &left.full_misalignment,
            &right.full_misalignment,
            &[
                "Full mount roll [deg]",
                "Full mount pitch [deg]",
                "Full mount yaw [deg]",
            ],
        ),
        (
            "full_meas_gyro",
            &left.full_meas_gyro,
            &right.full_meas_gyro,
            &[
                "Full gyro x [deg/s]",
                "Full gyro y [deg/s]",
                "Full gyro z [deg/s]",
            ],
        ),
        (
            "full_meas_accel",
            &left.full_meas_accel,
            &right.full_meas_accel,
            &[
                "Full accel x [m/s^2]",
                "Full accel y [m/s^2]",
                "Full accel z [m/s^2]",
            ],
        ),
        (
            "full_bias_gyro",
            &left.full_bias_gyro,
            &right.full_bias_gyro,
            &[
                "Full gyro sensor bias X [deg/s]",
                "Full gyro sensor bias Y [deg/s]",
                "Full gyro sensor bias Z [deg/s]",
            ],
        ),
        (
            "full_bias_accel",
            &left.full_bias_accel,
            &right.full_bias_accel,
            &[
                "Full accel sensor bias X [m/s^2]",
                "Full accel sensor bias Y [m/s^2]",
                "Full accel sensor bias Z [m/s^2]",
            ],
        ),
        (
            "full_scale_gyro",
            &left.full_scale_gyro,
            &right.full_scale_gyro,
            &["Full sgx", "Full sgy", "Full sgz"],
        ),
        (
            "full_scale_accel",
            &left.full_scale_accel,
            &right.full_scale_accel,
            &["Full sax", "Full say", "Full saz"],
        ),
        (
            "full_map",
            &left.full_map,
            &right.full_map,
            &["Full path (lon,lat)"],
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
        "full_cov_bias",
        &left.full_cov_bias,
        &right.full_cov_bias,
        "Full sigma bias/scale ",
    )?;
    assert_numbered_named_traces_close(
        left_label,
        right_label,
        "full_cov_nonbias",
        &left.full_cov_nonbias,
        &right.full_cov_nonbias,
        "Full sigma state ",
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "reduced_mount_dx",
        &left.reduced_mount_dx,
        &right.reduced_mount_dx,
        &[
            "Reduced mount roll correction [deg/update]",
            "Reduced mount pitch correction [deg/update]",
            "Reduced mount yaw correction [deg/update]",
        ],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "reduced_nhc_mount_dx",
        &left.reduced_nhc_mount_dx,
        &right.reduced_nhc_mount_dx,
        &[
            "Reduced NHC Y mount roll correction [deg/update]",
            "Reduced NHC Y mount pitch correction [deg/update]",
            "Reduced NHC Y mount yaw correction [deg/update]",
            "Reduced NHC Z mount roll correction [deg/update]",
            "Reduced NHC Z mount pitch correction [deg/update]",
            "Reduced NHC Z mount yaw correction [deg/update]",
        ],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "reduced_nhc_innovation",
        &left.reduced_nhc_innovation,
        &right.reduced_nhc_innovation,
        &[
            "Reduced NHC Y innovation [m/s]",
            "Reduced NHC Z innovation [m/s]",
        ],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "reduced_nhc_nis",
        &left.reduced_nhc_nis,
        &right.reduced_nhc_nis,
        &["Reduced NHC Y NIS", "Reduced NHC Z NIS"],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "reduced_nhc_h_mount_norm",
        &left.reduced_nhc_h_mount_norm,
        &right.reduced_nhc_h_mount_norm,
        &["Reduced NHC Y mount H norm", "Reduced NHC Z mount H norm"],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "full_mount_sigma",
        &left.full_mount_sigma,
        &right.full_mount_sigma,
        &[
            "Full mount roll sigma [deg]",
            "Full mount pitch sigma [deg]",
            "Full mount yaw sigma [deg]",
        ],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "full_mount_dx",
        &left.full_mount_dx,
        &right.full_mount_dx,
        &[
            "Full mount roll correction [deg/update]",
            "Full mount pitch correction [deg/update]",
            "Full mount yaw correction [deg/update]",
        ],
    )?;
    assert_named_traces_close(
        left_label,
        right_label,
        "full_gnss_pos_gate",
        &left.full_gnss_pos_gate,
        &right.full_gnss_pos_gate,
        &[
            "Full GNSS position gate normalized residual row 0",
            "Full GNSS position gate normalized residual row 1",
            "Full GNSS position gate normalized residual row 2",
            "Full GNSS position accepted",
        ],
    )?;
    assert_heading_samples_close(
        left_label,
        right_label,
        "full_map_heading",
        &left.full_map_heading,
        &right.full_map_heading,
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
