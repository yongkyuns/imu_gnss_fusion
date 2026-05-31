use anyhow::{Context, Result};
use fusion_tools::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, fusion_gnss_sample, fusion_imu_sample,
};
use fusion_tools::eval::gnss_ins::{as_q64, quat_angle_deg, quat_rotate};
use fusion_tools::eval::replay::{ReplayEvent, for_each_event};
use fusion_tools::synthetic::gnss_ins_path::{MotionProfile, PathGenConfig, generate};
use fusion_tools::visualizer::pipeline::generic::reference_mount_rpy_to_q_bv;
use sensor_fusion::SensorFusion;

const NOMINAL_MANEUVER: &str = "\
initial lat=32 lon=120 alt=20 speed=0 yaw=0 pitch=0 roll=0
wait 20s
accelerate 0.8m/s^2 for 16s
hold 8s
repeat 5 {
    turn left 12dps for 8s
    hold 5s
    turn right 12dps for 8s
    hold 5s
    accelerate 0.6m/s^2 for 6s
    brake 0.6m/s^2 for 6s
}
brake 0.8m/s^2 for 12s
hold 10s
";

const MOUNT_SWEEP_RPY_DEG: [[f64; 3]; 9] = [
    [0.0, 0.0, 0.0],
    [5.0, -5.0, 5.0],
    [-5.0, 5.0, -5.0],
    [8.0, -6.0, 45.0],
    [-8.0, 6.0, -45.0],
    [10.0, 0.0, 90.0],
    [0.0, -10.0, -90.0],
    [7.0, 4.0, 135.0],
    [-7.0, -4.0, 180.0],
];

#[test]
fn align_estimates_nominal_maneuver_across_mount_angle_sweep() -> Result<()> {
    let profile = MotionProfile::from_dsl_str(NOMINAL_MANEUVER)?;
    let generated = generate(
        &profile,
        PathGenConfig {
            imu_hz: 50.0,
            gnss_hz: 2.0,
            ..PathGenConfig::default()
        },
    )?;

    for mount_rpy_deg in MOUNT_SWEEP_RPY_DEG {
        let outcome = run_align_on_generated_path(&generated, mount_rpy_deg)
            .with_context(|| format!("mount_rpy_deg={mount_rpy_deg:?}"))?;

        assert!(
            outcome.coarse_ready_t_s.is_some(),
            "align never became coarse-ready for mount_rpy_deg={mount_rpy_deg:?}; outcome={outcome:?}"
        );
        assert!(
            outcome.coarse_ready_t_s.unwrap() < 130.0,
            "align seeded too late for mount_rpy_deg={mount_rpy_deg:?}; outcome={outcome:?}"
        );
        assert!(
            outcome.final_qerr_deg < 2.0,
            "align final mount error too high for mount_rpy_deg={mount_rpy_deg:?}; outcome={outcome:?}"
        );
        assert!(
            outcome.final_sigma_max_deg < 8.5,
            "align did not collapse covariance enough for mount_rpy_deg={mount_rpy_deg:?}; outcome={outcome:?}"
        );
        assert!(
            outcome.final_qerr_deg <= 1.5 * outcome.final_sigma_max_deg.max(0.5),
            "align covariance is over-confident relative to error for mount_rpy_deg={mount_rpy_deg:?}; outcome={outcome:?}"
        );
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct AlignRobustnessOutcome {
    coarse_ready_t_s: Option<f64>,
    final_qerr_deg: f64,
    final_sigma_max_deg: f64,
}

fn run_align_on_generated_path(
    generated: &fusion_tools::synthetic::gnss_ins_path::GeneratedPath,
    mount_rpy_deg: [f64; 3],
) -> Result<AlignRobustnessOutcome> {
    let q_truth = reference_mount_rpy_to_q_bv(mount_rpy_deg);
    let imu = generated
        .imu
        .iter()
        .map(|sample| GenericImuSample {
            t_s: sample.t_s,
            gyro_radps: quat_rotate(q_truth, sample.gyro_vehicle_radps),
            accel_mps2: quat_rotate(q_truth, sample.accel_vehicle_mps2),
        })
        .collect::<Vec<_>>();
    let gnss = generated
        .gnss
        .iter()
        .map(|sample| GenericGnssSample {
            t_s: sample.t_s,
            lat_deg: sample.lat_deg,
            lon_deg: sample.lon_deg,
            height_m: sample.height_m,
            vel_ned_mps: sample.vel_ned_mps,
            pos_std_m: [0.5, 0.5, 0.5],
            vel_std_mps: [0.2, 0.2, 0.2],
            heading_rad: None,
        })
        .collect::<Vec<_>>();

    let mut fusion = SensorFusion::new();
    let mut coarse_ready_t_s = None;

    for_each_event(&imu, &gnss, |event| match event {
        ReplayEvent::Imu(_, sample) => {
            let _ = fusion.process_imu(fusion_imu_sample(*sample));
        }
        ReplayEvent::Gnss(_, sample) => {
            let update = fusion.process_gnss(fusion_gnss_sample(*sample));
            if update.mount_ready && coarse_ready_t_s.is_none() {
                coarse_ready_t_s = Some(sample.t_s);
            }
        }
    });

    let align = fusion.align().context("align never initialized")?;
    let sigma = align.sigma_deg();
    Ok(AlignRobustnessOutcome {
        coarse_ready_t_s,
        final_qerr_deg: quat_angle_deg(as_q64(align.q_bv), q_truth),
        final_sigma_max_deg: sigma.iter().copied().fold(0.0_f32, f32::max) as f64,
    })
}
