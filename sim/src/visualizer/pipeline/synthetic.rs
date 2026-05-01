use anyhow::Result;

use crate::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, GenericReferenceRpySample,
};
use crate::eval::gnss_ins::{quat_angle_deg, quat_rotate};
use crate::synthetic::gnss_ins_path::{
    GpsNoiseModel, ImuAccuracy, MeasurementNoiseConfig, MotionProfile, PathGenConfig,
    generate_with_noise,
};
use crate::visualizer::math::{ecef_to_ned, lla_to_ecef, quat_rpy_deg};
use crate::visualizer::model::{EkfImuSource, PlotData, Trace};
use crate::visualizer::pipeline::generic::{
    GenericReplayInput, GenericReplayProgress, build_generic_replay_plot_data_with_eskf_mount_seed,
    build_generic_replay_plot_data_with_progress_and_eskf_mount_seed, q_vb_to_reference_mount_rpy,
    reference_mount_rpy_to_q_vb,
};
use crate::visualizer::pipeline::{EkfCompareConfig, GnssOutageConfig};

#[derive(Clone, Copy, Debug)]
pub enum SyntheticNoiseMode {
    Truth,
    Low,
    Mid,
    High,
}

#[derive(Clone, Debug)]
pub struct SyntheticVisualizerConfig {
    pub motion_def: Option<std::path::PathBuf>,
    pub motion_label: String,
    pub motion_text: Option<String>,
    pub noise_mode: SyntheticNoiseMode,
    pub seed: u64,
    pub mount_rpy_deg: [f64; 3],
    pub imu_hz: f64,
    pub gnss_hz: f64,
    pub gnss_time_shift_ms: f64,
    pub early_vel_bias_ned_mps: [f64; 3],
    pub early_fault_window_s: Option<(f64, f64)>,
}

pub fn build_synthetic_plot_data(
    synth_cfg: &SyntheticVisualizerConfig,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
) -> Result<PlotData> {
    build_synthetic_plot_data_impl(synth_cfg, ekf_imu_source, ekf_cfg, gnss_outages, None)
}

pub fn build_synthetic_plot_data_with_progress(
    synth_cfg: &SyntheticVisualizerConfig,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
    progress: &mut dyn FnMut(GenericReplayProgress),
) -> Result<PlotData> {
    build_synthetic_plot_data_impl(
        synth_cfg,
        ekf_imu_source,
        ekf_cfg,
        gnss_outages,
        Some(progress),
    )
}

fn build_synthetic_plot_data_impl(
    synth_cfg: &SyntheticVisualizerConfig,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
    progress: Option<&mut dyn FnMut(GenericReplayProgress)>,
) -> Result<PlotData> {
    let profile = match (&synth_cfg.motion_text, &synth_cfg.motion_def) {
        (Some(text), _) if synth_cfg.motion_label.ends_with(".csv") => {
            MotionProfile::from_csv_str(text)?
        }
        (Some(text), _) => MotionProfile::from_dsl_str(text)?,
        (None, Some(path)) => MotionProfile::from_path(path)?,
        (None, None) => anyhow::bail!("synthetic scenario has no path or inline text"),
    };
    let path_cfg = PathGenConfig {
        imu_hz: synth_cfg.imu_hz,
        gnss_hz: synth_cfg.gnss_hz,
        ..PathGenConfig::default()
    };
    let noise = match synth_cfg.noise_mode {
        SyntheticNoiseMode::Truth => MeasurementNoiseConfig::zero(),
        SyntheticNoiseMode::Low => MeasurementNoiseConfig::accuracy(ImuAccuracy::Low),
        SyntheticNoiseMode::Mid => MeasurementNoiseConfig::accuracy(ImuAccuracy::Mid),
        SyntheticNoiseMode::High => MeasurementNoiseConfig::accuracy(ImuAccuracy::High),
    };
    let measured = generate_with_noise(&profile, path_cfg, noise, synth_cfg.seed)?;
    let q_truth_mount = reference_mount_rpy_to_q_vb(synth_cfg.mount_rpy_deg);
    let gps_noise = noise.gps.unwrap_or(GpsNoiseModel {
        pos_std_m: [0.5, 0.5, 0.5],
        vel_std_mps: [0.2, 0.2, 0.2],
    });
    let imu_dt_s = 1.0 / synth_cfg.imu_hz;
    let imu = measured
        .imu
        .iter()
        .map(|s| GenericImuSample {
            // gnss-ins-sim timestamps IMU samples at the start of the interval
            // represented by the sample. The replay pipeline consumes IMU as
            // an interval ending at `t_s`, so shift generated-only samples by
            // one output period while preserving the generator's raw contract.
            t_s: s.t_s + imu_dt_s,
            gyro_radps: quat_rotate(q_truth_mount, s.gyro_vehicle_radps),
            accel_mps2: quat_rotate(q_truth_mount, s.accel_vehicle_mps2),
        })
        .collect::<Vec<_>>();
    let gnss = measured
        .gnss
        .iter()
        .filter_map(|s| {
            let t_s = s.t_s + synth_cfg.gnss_time_shift_ms * 1.0e-3;
            if !t_s.is_finite() || t_s < 0.0 {
                return None;
            }
            let mut vel_ned_mps = s.vel_ned_mps;
            if let Some((start_s, end_s)) = synth_cfg.early_fault_window_s
                && (start_s..=end_s).contains(&t_s)
            {
                for (v, bias) in vel_ned_mps.iter_mut().zip(synth_cfg.early_vel_bias_ned_mps) {
                    *v += bias;
                }
            }
            Some(GenericGnssSample {
                t_s,
                lat_deg: s.lat_deg,
                lon_deg: s.lon_deg,
                height_m: s.height_m,
                vel_ned_mps,
                pos_std_m: gps_noise.pos_std_m,
                vel_std_mps: gps_noise.vel_std_mps,
                heading_rad: None,
            })
        })
        .collect::<Vec<_>>();

    let ref_truth = &measured.reference.truth;
    let Some(first_truth) = ref_truth.first() else {
        anyhow::bail!("synthetic scenario produced no reference truth samples");
    };
    let ref_ecef = lla_to_ecef(
        first_truth.lat_deg,
        first_truth.lon_deg,
        first_truth.height_m,
    );
    let mut truth_pos_n = Vec::new();
    let mut truth_pos_e = Vec::new();
    let mut truth_pos_d = Vec::new();
    let mut truth_vel_n = Vec::new();
    let mut truth_vel_e = Vec::new();
    let mut truth_vel_d = Vec::new();
    let mut truth_roll = Vec::new();
    let mut truth_pitch = Vec::new();
    let mut truth_yaw = Vec::new();
    let mut truth_map = Vec::new();
    let mut truth_speed = Vec::new();
    for truth in ref_truth {
        let ecef = lla_to_ecef(truth.lat_deg, truth.lon_deg, truth.height_m);
        let ned = ecef_to_ned(ecef, ref_ecef, first_truth.lat_deg, first_truth.lon_deg);
        let (roll, pitch, yaw) = quat_rpy_deg(
            truth.q_bn[0] as f32,
            truth.q_bn[1] as f32,
            truth.q_bn[2] as f32,
            truth.q_bn[3] as f32,
        );
        truth_pos_n.push([truth.t_s, ned[0]]);
        truth_pos_e.push([truth.t_s, ned[1]]);
        truth_pos_d.push([truth.t_s, ned[2]]);
        truth_vel_n.push([truth.t_s, truth.vel_ned_mps[0]]);
        truth_vel_e.push([truth.t_s, truth.vel_ned_mps[1]]);
        truth_vel_d.push([truth.t_s, truth.vel_ned_mps[2]]);
        truth_roll.push([truth.t_s, roll]);
        truth_pitch.push([truth.t_s, pitch]);
        truth_yaw.push([truth.t_s, yaw]);
        truth_map.push([truth.lon_deg, truth.lat_deg]);
        truth_speed.push([truth.t_s, truth.vel_ned_mps[0].hypot(truth.vel_ned_mps[1])]);
    }

    let mut gnss_map = Vec::new();
    let mut gnss_speed = Vec::new();
    for sample in &gnss {
        gnss_map.push([sample.lon_deg, sample.lat_deg]);
        gnss_speed.push([
            sample.t_s,
            sample.vel_ned_mps[0].hypot(sample.vel_ned_mps[1]),
        ]);
    }

    let reference_attitude = ref_truth
        .iter()
        .zip(
            truth_roll
                .iter()
                .zip(truth_pitch.iter())
                .zip(truth_yaw.iter()),
        )
        .map(|(truth, ((roll, pitch), yaw))| GenericReferenceRpySample {
            t_s: truth.t_s,
            roll_deg: roll[1],
            pitch_deg: pitch[1],
            yaw_deg: yaw[1],
        })
        .collect::<Vec<_>>();
    let end_t_s = ref_truth.last().map(|s| s.t_s).unwrap_or(0.0);
    let (mount_roll_deg, mount_pitch_deg, mount_yaw_deg) =
        q_vb_to_reference_mount_rpy(q_truth_mount);
    let reference_mount = vec![
        GenericReferenceRpySample {
            t_s: 0.0,
            roll_deg: mount_roll_deg,
            pitch_deg: mount_pitch_deg,
            yaw_deg: mount_yaw_deg,
        },
        GenericReferenceRpySample {
            t_s: end_t_s,
            roll_deg: mount_roll_deg,
            pitch_deg: mount_pitch_deg,
            yaw_deg: mount_yaw_deg,
        },
    ];
    let replay = GenericReplayInput {
        imu,
        gnss,
        reference_attitude,
        reference_mount,
    };
    let eskf_mount_seed = ekf_imu_source.uses_ref_mount().then_some([
        q_truth_mount[0] as f32,
        q_truth_mount[1] as f32,
        q_truth_mount[2] as f32,
        q_truth_mount[3] as f32,
    ]);
    let mut data = match progress {
        Some(progress) => build_generic_replay_plot_data_with_progress_and_eskf_mount_seed(
            &replay,
            ekf_imu_source,
            ekf_cfg,
            gnss_outages,
            progress,
            eskf_mount_seed,
        ),
        None => build_generic_replay_plot_data_with_eskf_mount_seed(
            &replay,
            ekf_imu_source,
            ekf_cfg,
            gnss_outages,
            eskf_mount_seed,
        ),
    };
    add_synthetic_overlays(
        &mut data,
        SyntheticOverlayTraces {
            truth_pos_n,
            truth_pos_e,
            truth_pos_d,
            truth_vel_n,
            truth_vel_e,
            truth_vel_d,
            truth_roll,
            truth_pitch,
            truth_yaw,
            truth_map,
            truth_speed,
            gnss_map,
            gnss_speed,
            end_t_s,
            q_truth_mount,
        },
    );
    Ok(data)
}

struct SyntheticOverlayTraces {
    truth_pos_n: Vec<[f64; 2]>,
    truth_pos_e: Vec<[f64; 2]>,
    truth_pos_d: Vec<[f64; 2]>,
    truth_vel_n: Vec<[f64; 2]>,
    truth_vel_e: Vec<[f64; 2]>,
    truth_vel_d: Vec<[f64; 2]>,
    truth_roll: Vec<[f64; 2]>,
    truth_pitch: Vec<[f64; 2]>,
    truth_yaw: Vec<[f64; 2]>,
    truth_map: Vec<[f64; 2]>,
    truth_speed: Vec<[f64; 2]>,
    gnss_map: Vec<[f64; 2]>,
    gnss_speed: Vec<[f64; 2]>,
    end_t_s: f64,
    q_truth_mount: [f64; 4],
}

fn add_synthetic_overlays(data: &mut PlotData, traces: SyntheticOverlayTraces) {
    data.speed = vec![
        Trace {
            name: "Synthetic truth horizontal speed [m/s]".to_string(),
            points: traces.truth_speed,
        },
        Trace {
            name: "Synthetic GNSS horizontal speed [m/s]".to_string(),
            points: traces.gnss_speed,
        },
    ];
    rename_trace_prefix(&mut data.imu_raw_gyro, "Raw IMU", "Synthetic raw IMU");
    rename_trace_prefix(&mut data.imu_raw_accel, "Raw IMU", "Synthetic raw IMU");
    rename_trace(&mut data.eskf_cmp_vel, "ESKF velN [m/s]", "ESKF vN [m/s]");
    rename_trace(&mut data.eskf_cmp_vel, "ESKF velE [m/s]", "ESKF vE [m/s]");
    rename_trace(&mut data.eskf_cmp_vel, "ESKF velD [m/s]", "ESKF vD [m/s]");
    rename_trace(&mut data.eskf_cmp_att, "ekf initialized", "EKF initialized");
    rename_trace(
        &mut data.eskf_map,
        "GNSS path (lon,lat)",
        "Synthetic GNSS path (lon,lat)",
    );
    replace_trace_points(
        &mut data.eskf_map,
        "Synthetic GNSS path (lon,lat)",
        traces.gnss_map,
    );

    data.orientation = vec![
        Trace {
            name: "Synthetic truth roll [deg]".to_string(),
            points: traces.truth_roll.clone(),
        },
        Trace {
            name: "Synthetic truth pitch [deg]".to_string(),
            points: traces.truth_pitch.clone(),
        },
        Trace {
            name: "Synthetic truth yaw [deg]".to_string(),
            points: traces.truth_yaw.clone(),
        },
    ];
    insert_after_trace(
        &mut data.eskf_cmp_pos,
        "ESKF posN [m]",
        Trace {
            name: "Synthetic truth posN [m]".to_string(),
            points: traces.truth_pos_n,
        },
    );
    insert_after_trace(
        &mut data.eskf_cmp_pos,
        "ESKF posE [m]",
        Trace {
            name: "Synthetic truth posE [m]".to_string(),
            points: traces.truth_pos_e,
        },
    );
    insert_after_trace(
        &mut data.eskf_cmp_pos,
        "ESKF posD [m]",
        Trace {
            name: "Synthetic truth posD [m]".to_string(),
            points: traces.truth_pos_d,
        },
    );
    insert_after_trace(
        &mut data.eskf_cmp_vel,
        "ESKF vN [m/s]",
        Trace {
            name: "Synthetic truth vN [m/s]".to_string(),
            points: traces.truth_vel_n,
        },
    );
    insert_after_trace(
        &mut data.eskf_cmp_vel,
        "ESKF vE [m/s]",
        Trace {
            name: "Synthetic truth vE [m/s]".to_string(),
            points: traces.truth_vel_e,
        },
    );
    insert_after_trace(
        &mut data.eskf_cmp_vel,
        "ESKF vD [m/s]",
        Trace {
            name: "Synthetic truth vD [m/s]".to_string(),
            points: traces.truth_vel_d,
        },
    );
    insert_after_trace(
        &mut data.eskf_cmp_att,
        "ESKF roll [deg]",
        Trace {
            name: "Synthetic truth roll [deg]".to_string(),
            points: traces.truth_roll,
        },
    );
    insert_after_trace(
        &mut data.eskf_cmp_att,
        "ESKF pitch [deg]",
        Trace {
            name: "Synthetic truth pitch [deg]".to_string(),
            points: traces.truth_pitch,
        },
    );
    insert_after_trace(
        &mut data.eskf_cmp_att,
        "ESKF yaw [deg]",
        Trace {
            name: "Synthetic truth yaw [deg]".to_string(),
            points: traces.truth_yaw,
        },
    );
    data.eskf_misalignment.push(Trace {
        name: "ESKF mount quaternion error [deg]".to_string(),
        points: mount_error_points(&data.eskf_misalignment, traces.q_truth_mount),
    });
    data.eskf_misalignment
        .extend(synthetic_mount_traces(traces.q_truth_mount, traces.end_t_s));
    data.eskf_map.insert(
        0,
        Trace {
            name: "Synthetic truth path (lon,lat)".to_string(),
            points: traces.truth_map,
        },
    );
}

fn synthetic_mount_traces(q_truth_mount: [f64; 4], end_t_s: f64) -> [Trace; 3] {
    let (roll_deg, pitch_deg, yaw_deg) = q_vb_to_reference_mount_rpy(q_truth_mount);
    [
        Trace {
            name: "Synthetic truth mount roll [deg]".to_string(),
            points: vec![[0.0, roll_deg], [end_t_s, roll_deg]],
        },
        Trace {
            name: "Synthetic truth mount pitch [deg]".to_string(),
            points: vec![[0.0, pitch_deg], [end_t_s, pitch_deg]],
        },
        Trace {
            name: "Synthetic truth mount yaw [deg]".to_string(),
            points: vec![[0.0, yaw_deg], [end_t_s, yaw_deg]],
        },
    ]
}

fn mount_error_points(traces: &[Trace], q_truth_mount: [f64; 4]) -> Vec<[f64; 2]> {
    let Some(roll) = traces
        .iter()
        .find(|trace| trace.name == "ESKF mount roll [deg]")
    else {
        return Vec::new();
    };
    let Some(pitch) = traces
        .iter()
        .find(|trace| trace.name == "ESKF mount pitch [deg]")
    else {
        return Vec::new();
    };
    let Some(yaw) = traces
        .iter()
        .find(|trace| trace.name == "ESKF mount yaw [deg]")
    else {
        return Vec::new();
    };
    roll.points
        .iter()
        .filter_map(|sample| {
            let t_s = sample[0];
            let pitch_deg = sample_trace_at(pitch, t_s)?;
            let yaw_deg = sample_trace_at(yaw, t_s)?;
            let q_est = reference_mount_rpy_to_q_vb([sample[1], pitch_deg, yaw_deg]);
            Some([t_s, quat_angle_deg(q_est, q_truth_mount)])
        })
        .collect()
}

fn insert_after_trace(traces: &mut Vec<Trace>, after_name: &str, trace: Trace) {
    if traces.iter().any(|existing| existing.name == trace.name) {
        return;
    }
    let idx = traces
        .iter()
        .position(|existing| existing.name == after_name)
        .map(|idx| idx + 1)
        .unwrap_or(traces.len());
    traces.insert(idx, trace);
}

fn rename_trace(traces: &mut [Trace], from: &str, to: &str) {
    for trace in traces {
        if trace.name == from {
            trace.name = to.to_string();
        }
    }
}

fn rename_trace_prefix(traces: &mut [Trace], from: &str, to: &str) {
    for trace in traces {
        if trace.name.starts_with(from) {
            trace.name = trace.name.replacen(from, to, 1);
        }
    }
}

fn replace_trace_points(traces: &mut [Trace], name: &str, points: Vec<[f64; 2]>) {
    if let Some(trace) = traces.iter_mut().find(|trace| trace.name == name) {
        trace.points = points;
    }
}

fn sample_trace_at(trace: &Trace, t_s: f64) -> Option<f64> {
    if !t_s.is_finite() || trace.points.is_empty() {
        return None;
    }
    let points = &trace.points;
    let idx = points.partition_point(|p| p[0] < t_s);
    if idx == 0 {
        return points.first().map(|p| p[1]);
    }
    if idx >= points.len() {
        return points.last().map(|p| p[1]);
    }
    let a = points[idx - 1];
    let b = points[idx];
    let dt = b[0] - a[0];
    if dt.abs() <= f64::EPSILON {
        return Some(b[1]);
    }
    let u = ((t_s - a[0]) / dt).clamp(0.0, 1.0);
    Some(a[1] + u * (b[1] - a[1]))
}
