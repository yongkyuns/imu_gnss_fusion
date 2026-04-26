use std::path::Path;
use std::process::Command;

use anyhow::{Result, bail};
use sensor_fusion::fusion::SensorFusion;
use sim::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, fusion_gnss_sample, fusion_imu_sample,
};
use sim::eval::gnss_ins::{
    as_q64, quat_angle_deg, quat_conj, quat_from_rpy_alg_deg, quat_mul, quat_rotate,
};
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::synthetic::gnss_ins_path::{
    AxisNoise, GpsNoiseModel, ImuAccuracy, ImuNoiseModel, MeasurementNoiseConfig, MotionProfile,
    PathGenConfig, VibrationNoise, add_measurement_noise, generate, generate_with_noise,
};
use sim::visualizer::math::{ecef_to_ned, lla_to_ecef};

const LOCAL_GNSS_INS_SIM_DIR: &str = "/Users/ykshin/Dev/me/gnss-ins-sim";
const SHORT_PROFILE: &str = "\
ini lat (deg),ini lon (deg),ini alt (m),ini vx_body (m/s),ini vy_body (m/s),ini vz_body (m/s),ini yaw (deg),ini pitch (deg),ini roll (deg)
32,120,0,0,0,0,0,0,0
command type,yaw (deg),pitch (deg),roll (deg),vx_body (m/s),vy_body (m/s),vz_body (m/s),command duration (s),GPS visibility
1,0,0,0,0,0,0,1,1
1,5,0,0,0.5,0,0,2,1
1,-5,0,0,-0.5,0,0,2,1
";

#[test]
fn rust_path_generator_matches_gnss_ins_sim_reference_samples() -> Result<()> {
    let profile = MotionProfile::from_csv_str(SHORT_PROFILE)?;
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
fn rust_path_generator_matches_local_gnss_ins_sim_full_short_trajectory() -> Result<()> {
    let gnss_ins_sim_dir =
        std::env::var("GNSS_INS_SIM_DIR").unwrap_or_else(|_| LOCAL_GNSS_INS_SIM_DIR.to_string());
    if !Path::new(&gnss_ins_sim_dir).exists() {
        eprintln!("skipping local gnss-ins-sim parity test; missing {gnss_ins_sim_dir}");
        return Ok(());
    }

    let profile = MotionProfile::from_csv_str(SHORT_PROFILE)?;
    let generated = generate(&profile, PathGenConfig::default())?;
    let python = Command::new("python3")
        .arg("-c")
        .arg(format!(
            r#"
import json, math, numpy as np, sys
sys.path.insert(0, {gnss_ins_sim_dir:?})
from gnss_ins_sim.pathgen import pathgen
from gnss_ins_sim.attitude import attitude
ini = np.array([32*math.pi/180,120*math.pi/180,0,0,0,0,0,0,0.0])
motion = np.array([
 [1,0,0,0,0,0,0,1.0,1],
 [1,5*math.pi/180,0,0,0.5,0,0,2.0,1],
 [1,-5*math.pi/180,0,0,-0.5,0,0,2.0,1],
], dtype=float)
output_def=np.array([[1.0,100.0],[1.0,2.0],[-1.0,100.0]])
mobility=np.array([10.0,0.5,1.0])
r=pathgen.path_gen(ini,motion,output_def,mobility,ref_frame=0,magnet=False)
quat = [attitude.euler2quat(row[7:10]).tolist() for row in r['nav']]
print(json.dumps({{'imu': r['imu'].tolist(), 'nav': r['nav'].tolist(), 'gps': r['gps'].tolist(), 'quat': quat}}))
"#
        ))
        .output()?;
    if !python.status.success() {
        bail!(
            "gnss-ins-sim Python reference failed: {}",
            String::from_utf8_lossy(&python.stderr)
        );
    }
    let reference: serde_json::Value = serde_json::from_slice(&python.stdout)?;
    let imu_ref = reference["imu"].as_array().expect("imu array");
    let nav_ref = reference["nav"].as_array().expect("nav array");
    let gps_ref = reference["gps"].as_array().expect("gps array");
    let quat_ref = reference["quat"].as_array().expect("quat array");
    assert_eq!(imu_ref.len(), generated.imu.len());
    assert_eq!(nav_ref.len(), generated.truth.len());
    assert_eq!(gps_ref.len(), generated.gnss.len());

    for (i, row) in imu_ref.iter().enumerate() {
        let row = row.as_array().expect("imu row");
        assert_close(
            "imu t",
            generated.imu[i].t_s * 100.0,
            f64_at(row, 0),
            1.0e-10,
        );
        for axis in 0..3 {
            assert_close(
                "imu accel",
                generated.imu[i].accel_vehicle_mps2[axis],
                f64_at(row, 1 + axis),
                1.0e-12,
            );
            assert_close(
                "imu gyro",
                generated.imu[i].gyro_vehicle_radps[axis],
                f64_at(row, 4 + axis),
                1.0e-15,
            );
        }
    }
    for (i, row) in nav_ref.iter().enumerate() {
        let row = row.as_array().expect("nav row");
        assert_close(
            "nav t",
            generated.truth[i].t_s * 100.0,
            f64_at(row, 0),
            1.0e-10,
        );
        assert_close(
            "nav lat",
            generated.truth[i].lat_deg.to_radians(),
            f64_at(row, 1),
            1.0e-14,
        );
        assert_close(
            "nav lon",
            generated.truth[i].lon_deg.to_radians(),
            f64_at(row, 2),
            1.0e-14,
        );
        assert_close(
            "nav h",
            generated.truth[i].height_m,
            f64_at(row, 3),
            1.0e-12,
        );
        for axis in 0..3 {
            assert_close(
                "nav vel",
                generated.truth[i].vel_ned_mps[axis],
                f64_at(row, 4 + axis),
                1.0e-12,
            );
        }
        let q_ref = quat_ref[i].as_array().expect("quat row");
        for q_idx in 0..4 {
            assert_close(
                "nav quat",
                generated.truth[i].q_bn[q_idx],
                f64_at(q_ref, q_idx),
                1.0e-15,
            );
        }
    }
    for (i, row) in gps_ref.iter().enumerate() {
        let row = row.as_array().expect("gps row");
        assert_close(
            "gps t",
            generated.gnss[i].t_s * 100.0,
            f64_at(row, 0),
            1.0e-10,
        );
        assert_close(
            "gps lat",
            generated.gnss[i].lat_deg.to_radians(),
            f64_at(row, 1),
            1.0e-14,
        );
        assert_close(
            "gps lon",
            generated.gnss[i].lon_deg.to_radians(),
            f64_at(row, 2),
            1.0e-14,
        );
        assert_close("gps h", generated.gnss[i].height_m, f64_at(row, 3), 1.0e-12);
        for axis in 0..3 {
            assert_close(
                "gps vel",
                generated.gnss[i].vel_ned_mps[axis],
                f64_at(row, 4 + axis),
                1.0e-12,
            );
        }
    }

    Ok(())
}

#[test]
fn measurement_noise_supports_bias_drift_white_noise_vibration_and_gps_noise() -> Result<()> {
    let profile = MotionProfile::from_csv_str(SHORT_PROFILE)?;
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
        gps: Some(GpsNoiseModel::low_accuracy()),
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
    assert!(matches!(
        MeasurementNoiseConfig::accuracy(ImuAccuracy::Low).gps,
        Some(_)
    ));

    Ok(())
}

#[test]
fn eskf_converges_on_generated_city_blocks_truth_signals() -> Result<()> {
    assert_eskf_converges_on_profile("city_blocks_15min.csv")
}

#[test]
fn eskf_converges_on_generated_figure8_truth_signals() -> Result<()> {
    assert_eskf_converges_on_profile("figure8_15min.csv")
}

#[test]
fn eskf_converges_on_generated_city_blocks_noisy_measurements() -> Result<()> {
    let profile_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("motion_profiles/city_blocks_15min.csv");
    let profile = MotionProfile::from_csv(&profile_path)?;
    let measured = generate_with_noise(
        &profile,
        PathGenConfig::default(),
        MeasurementNoiseConfig::accuracy(ImuAccuracy::Mid),
        20260426,
    )?;
    let summary = run_eskf_on_samples(
        &measured.reference,
        &measured.imu,
        &measured.gnss,
        [5.0, -5.0, 5.0],
        [5.0, 5.0, 7.0],
        [0.05, 0.05, 0.05],
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

fn assert_eskf_converges_on_profile(profile_name: &str) -> Result<()> {
    let profile_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("motion_profiles/{profile_name}"));
    let profile = MotionProfile::from_csv(&profile_path)?;
    let generated = generate(&profile, PathGenConfig::default())?;
    let summary = run_eskf_on_generated_path(&generated, [5.0, -5.0, 5.0])?;

    assert!(
        summary.ekf_initialized,
        "ESKF did not initialize on generated synthetic data: {profile_name}"
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

#[derive(Clone, Copy, Debug)]
struct EskfSyntheticSummary {
    ekf_initialized: bool,
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

fn run_eskf_on_generated_path(
    generated: &sim::synthetic::gnss_ins_path::GeneratedPath,
    mount_rpy_deg: [f64; 3],
) -> Result<EskfSyntheticSummary> {
    run_eskf_on_samples(
        generated,
        &generated.imu,
        &generated.gnss,
        mount_rpy_deg,
        [0.5, 0.5, 0.5],
        [0.2, 0.2, 0.2],
    )
}

fn run_eskf_on_samples(
    reference: &sim::synthetic::gnss_ins_path::GeneratedPath,
    imu_samples: &[sim::datasets::gnss_ins_sim::ImuSample],
    gnss_samples: &[sim::datasets::gnss_ins_sim::GnssSample],
    mount_rpy_deg: [f64; 3],
    pos_std_m: [f64; 3],
    vel_std_mps: [f64; 3],
) -> Result<EskfSyntheticSummary> {
    let q_truth = quat_from_rpy_alg_deg(mount_rpy_deg[0], mount_rpy_deg[1], mount_rpy_deg[2]);
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
    let ref_ecef = lla_to_ecef(
        reference.truth[0].lat_deg,
        reference.truth[0].lon_deg,
        reference.truth[0].height_m,
    );
    let mut errors = Vec::new();

    for_each_event(&imu, &gnss, |event| match event {
        ReplayEvent::Imu(idx, sample) => {
            let _ = fusion.process_imu(fusion_imu_sample(*sample));
            if let Some(eskf) = fusion.eskf() {
                let truth = reference.truth[idx];
                let truth_ecef = lla_to_ecef(truth.lat_deg, truth.lon_deg, truth.height_m);
                let truth_pos_ned = ecef_to_ned(
                    truth_ecef,
                    ref_ecef,
                    reference.truth[0].lat_deg,
                    reference.truth[0].lon_deg,
                );
                let q_seed = fusion
                    .eskf_mount_q_vb()
                    .or_else(|| fusion.mount_q_vb())
                    .map(as_q64)
                    .unwrap_or(q_truth);
                let q_cs = as_q64([
                    eskf.nominal.qcs0,
                    eskf.nominal.qcs1,
                    eskf.nominal.qcs2,
                    eskf.nominal.qcs3,
                ]);
                let q_full_mount = quat_mul(q_seed, quat_conj(q_cs));
                let q_est_att = as_q64([
                    eskf.nominal.q0,
                    eskf.nominal.q1,
                    eskf.nominal.q2,
                    eskf.nominal.q3,
                ]);
                errors.push(StateErr {
                    t_s: sample.t_s,
                    mount_quat_err_deg: quat_angle_deg(q_full_mount, q_truth),
                    att_quat_err_deg: quat_angle_deg(q_est_att, truth.q_bn),
                    vel_err_mps: norm3([
                        eskf.nominal.vn as f64 - truth.vel_ned_mps[0],
                        eskf.nominal.ve as f64 - truth.vel_ned_mps[1],
                        eskf.nominal.vd as f64 - truth.vel_ned_mps[2],
                    ]),
                    pos_err_m: norm3([
                        eskf.nominal.pn as f64 - truth_pos_ned[0],
                        eskf.nominal.pe as f64 - truth_pos_ned[1],
                        eskf.nominal.pd as f64 - truth_pos_ned[2],
                    ]),
                    gyro_bias_norm_dps: norm3([
                        eskf.nominal.bgx as f64,
                        eskf.nominal.bgy as f64,
                        eskf.nominal.bgz as f64,
                    ])
                    .to_degrees(),
                    accel_bias_norm_mps2: norm3([
                        eskf.nominal.bax as f64,
                        eskf.nominal.bay as f64,
                        eskf.nominal.baz as f64,
                    ]),
                });
            }
        }
        ReplayEvent::Gnss(_, sample) => {
            let _ = fusion.process_gnss(fusion_gnss_sample(*sample));
        }
    });

    let Some(final_err) = errors.last().copied() else {
        bail!("ESKF produced no state samples");
    };
    let tail_start = (final_err.t_s - 60.0).max(0.0);
    let tail = errors
        .iter()
        .filter(|e| e.t_s >= tail_start)
        .collect::<Vec<_>>();
    let tail_mount_quat_err_mean_deg =
        tail.iter().map(|e| e.mount_quat_err_deg).sum::<f64>() / tail.len() as f64;

    Ok(EskfSyntheticSummary {
        ekf_initialized: fusion.eskf().is_some(),
        final_mount_quat_err_deg: final_err.mount_quat_err_deg,
        tail_mount_quat_err_mean_deg,
        final_att_quat_err_deg: final_err.att_quat_err_deg,
        final_vel_err_mps: final_err.vel_err_mps,
        final_pos_err_m: final_err.pos_err_m,
        final_gyro_bias_norm_dps: final_err.gyro_bias_norm_dps,
        final_accel_bias_norm_mps2: final_err.accel_bias_norm_mps2,
    })
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
