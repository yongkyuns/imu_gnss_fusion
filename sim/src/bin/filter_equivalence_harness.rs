use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use sensor_fusion::ProcessNoise;
use sensor_fusion::full::InitConfig;
use sensor_fusion::full::{ERROR_STATES, Filter, ImuDelta};
use sensor_fusion::reduced::State;
use sensor_fusion::{MountSource, SensorFusion};
use sim::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, GenericReferenceRpySample, fusion_gnss_sample,
    fusion_imu_sample,
};
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::visualizer::math::{ecef_to_lla, ecef_to_ned, lla_to_ecef, quat_rpy_deg};
use sim::visualizer::pipeline::FilterCompareConfig;
use sim::visualizer::pipeline::generic::{
    GenericReplayInput, parse_generic_replay_csvs_with_refs, q_vb_to_reference_mount_rpy,
    reference_mount_rpy_to_q_vb,
};
use sim::visualizer::pipeline::synthetic::{
    SyntheticNoiseMode, SyntheticVisualizerConfig, build_synthetic_replay_input,
};

const FULL_NHC_GNSS_SPEED_MAX_AGE_S: f64 = 1.0;

#[derive(Parser, Debug)]
#[command(name = "filter_equivalence_harness")]
#[command(about = "Emit Reduced/full comparison snapshots in a common physical basis")]
struct Args {
    #[arg(long, conflicts_with = "generic_replay_dir")]
    synthetic_motion_def: Option<PathBuf>,
    #[arg(long, conflicts_with = "synthetic_motion_def")]
    generic_replay_dir: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = MountMode::Ref)]
    mount_mode: MountMode,
    #[arg(long, value_enum, default_value_t = SyntheticNoiseArg::Truth)]
    synthetic_noise: SyntheticNoiseArg,
    #[arg(long, default_value_t = 100.0)]
    synthetic_imu_hz: f64,
    #[arg(long, default_value_t = 2.0)]
    synthetic_gnss_hz: f64,
    #[arg(long, default_value_t = 1)]
    sample_stride: usize,
    #[arg(long)]
    max_time_s: Option<f64>,
    #[arg(long)]
    attitude_roll_pitch_init_sigma_deg: Option<f32>,
    #[arg(long)]
    align_handoff_delay_s: Option<f32>,
    #[arg(long)]
    freeze_misalignment_states: bool,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MountMode {
    /// Use reference/true mount as the seed for both filters.
    Ref,
    /// Let Reduced and full each use the internal align handoff path.
    Internal,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SyntheticNoiseArg {
    Truth,
    Low,
    Mid,
    High,
}

#[derive(Clone, Copy, Debug, Default)]
struct VelResidual {
    valid: bool,
    value_ned_mps: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
struct CommonSnapshot {
    pos_ned_m: [f64; 3],
    vel_ned_mps: [f64; 3],
    att_rpy_deg: [f64; 3],
    mount_rpy_deg: [f64; 3],
    gyro_bias_dps: [f64; 3],
    accel_bias_mps2: [f64; 3],
    pos_sigma_m: [f64; 3],
    vel_sigma_mps: [f64; 3],
    att_sigma_deg: [f64; 3],
    mount_sigma_deg: [f64; 3],
}

struct HarnessState {
    fusion: SensorFusion,
    align_fusion: SensorFusion,
    full: Filter,
    full_ready: bool,
    last_imu: Option<GenericImuSample>,
    latest_gnss: Option<GenericGnssSample>,
    last_full_gnss_used_t_s: f64,
    ref_gnss: GenericGnssSample,
    ref_ecef: [f64; 3],
    fixed_mount_q_vb: Option<[f32; 4]>,
    last_reduced_vel_residual: VelResidual,
    last_full_vel_residual: VelResidual,
    cfg: FilterCompareConfig,
    mount_mode: MountMode,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (mut replay, fixed_mount_q_vb) = load_replay(&args)?;
    if let Some(limit) = args.max_time_s {
        let start_t_s = replay_limit_anchor_t_s(&replay)?;
        let end_t_s = start_t_s + limit;
        replay.imu.retain(|sample| sample.t_s <= end_t_s);
        replay.gnss.retain(|sample| sample.t_s <= end_t_s);
        replay
            .reference_attitude
            .retain(|sample| sample.t_s <= end_t_s);
        replay
            .reference_mount
            .retain(|sample| sample.t_s <= end_t_s);
    }
    let Some(ref_gnss) = replay.gnss.first().copied() else {
        bail!("replay has no GNSS samples");
    };
    let mut cfg = FilterCompareConfig::default();
    if let Some(sigma_deg) = args.attitude_roll_pitch_init_sigma_deg {
        cfg.attitude_roll_pitch_init_sigma_deg = sigma_deg;
    }
    if let Some(delay_s) = args.align_handoff_delay_s {
        cfg.align_handoff_delay_s = delay_s;
    }
    if args.freeze_misalignment_states {
        cfg.freeze_misalignment_states = true;
    }
    let mut state = HarnessState::new(ref_gnss, fixed_mount_q_vb, cfg, args.mount_mode);

    let mut writer: Box<dyn Write> = match &args.output {
        Some(path) => Box::new(BufWriter::new(
            fs::File::create(path)
                .with_context(|| format!("failed to create {}", path.display()))?,
        )),
        None => Box::new(BufWriter::new(io::stdout())),
    };
    write_header(&mut writer)?;

    let mut snapshot_index = 0usize;
    for_each_event(&replay.imu, &replay.gnss, |event| {
        let t_s = event_time(&event);
        let is_imu = matches!(event, ReplayEvent::Imu(_, _));
        state.process_event(&event);
        if !is_imu {
            return;
        }
        if snapshot_index % args.sample_stride.max(1) == 0 {
            let _ = write_snapshot(&mut writer, t_s, &state, &replay);
        }
        snapshot_index += 1;
    });
    Ok(())
}

fn load_replay(args: &Args) -> Result<(GenericReplayInput, Option<[f32; 4]>)> {
    if let Some(path) = &args.synthetic_motion_def {
        let synth_cfg = SyntheticVisualizerConfig {
            motion_def: Some(path.clone()),
            motion_label: path.display().to_string(),
            motion_text: None,
            noise_mode: match args.synthetic_noise {
                SyntheticNoiseArg::Truth => SyntheticNoiseMode::Truth,
                SyntheticNoiseArg::Low => SyntheticNoiseMode::Low,
                SyntheticNoiseArg::Mid => SyntheticNoiseMode::Mid,
                SyntheticNoiseArg::High => SyntheticNoiseMode::High,
            },
            disable_imu_noise: false,
            disable_gnss_noise: false,
            seed: 1,
            mount_rpy_deg: [5.0, -5.0, 5.0],
            imu_hz: args.synthetic_imu_hz,
            gnss_hz: args.synthetic_gnss_hz,
            gnss_time_shift_ms: 0.0,
            early_vel_bias_ned_mps: [0.0; 3],
            early_fault_window_s: None,
        };
        let (replay, q_mount) = build_synthetic_replay_input(&synth_cfg)?;
        return Ok((replay, Some(q_mount.map(|v| v as f32))));
    }

    if let Some(dir) = &args.generic_replay_dir {
        let imu_csv = fs::read_to_string(dir.join("imu.csv"))
            .with_context(|| format!("failed to read {}", dir.join("imu.csv").display()))?;
        let gnss_csv = fs::read_to_string(dir.join("gnss.csv"))
            .with_context(|| format!("failed to read {}", dir.join("gnss.csv").display()))?;
        let reference_attitude_csv = fs::read_to_string(dir.join("reference_attitude.csv")).ok();
        let reference_mount_csv = fs::read_to_string(dir.join("reference_mount.csv")).ok();
        let reference_position_csv = fs::read_to_string(dir.join("reference_position.csv")).ok();
        let replay = parse_generic_replay_csvs_with_refs(
            &imu_csv,
            &gnss_csv,
            reference_attitude_csv.as_deref(),
            reference_mount_csv.as_deref(),
            reference_position_csv.as_deref(),
        )?;
        let q_mount = replay
            .reference_mount
            .iter()
            .rev()
            .find(|sample| {
                sample.roll_deg.is_finite()
                    && sample.pitch_deg.is_finite()
                    && sample.yaw_deg.is_finite()
            })
            .map(|sample| {
                reference_mount_rpy_to_q_vb([sample.roll_deg, sample.pitch_deg, sample.yaw_deg])
                    .map(|v| v as f32)
            });
        return Ok((replay, q_mount));
    }

    bail!("provide either --synthetic-motion-def or --generic-replay-dir")
}

fn replay_limit_anchor_t_s(replay: &GenericReplayInput) -> Result<f64> {
    replay
        .gnss
        .first()
        .map(|sample| sample.t_s)
        .or_else(|| replay.imu.first().map(|sample| sample.t_s))
        .context("replay has no IMU or GNSS samples")
}

impl HarnessState {
    fn new(
        ref_gnss: GenericGnssSample,
        fixed_mount_q_vb: Option<[f32; 4]>,
        cfg: FilterCompareConfig,
        mount_mode: MountMode,
    ) -> Self {
        let mut fusion = SensorFusion::new();
        apply_fusion_config(&mut fusion, cfg);
        if matches!(mount_mode, MountMode::Ref)
            && let Some(q_vb) = fixed_mount_q_vb
        {
            fusion.set_misalignment(q_vb);
        }

        let mut align_fusion = SensorFusion::new();
        apply_fusion_config(&mut align_fusion, cfg);

        Self {
            fusion,
            align_fusion,
            full: Filter::new(
                cfg.noise.full
                    .unwrap_or_else(ProcessNoise::lsm6dso_104hz),
            ),
            full_ready: false,
            last_imu: None,
            latest_gnss: None,
            last_full_gnss_used_t_s: f64::NEG_INFINITY,
            ref_gnss,
            ref_ecef: lla_to_ecef(ref_gnss.lat_deg, ref_gnss.lon_deg, ref_gnss.height_m),
            fixed_mount_q_vb,
            last_reduced_vel_residual: VelResidual::default(),
            last_full_vel_residual: VelResidual::default(),
            cfg,
            mount_mode,
        }
    }

    fn process_event(&mut self, event: &ReplayEvent<'_>) {
        match event {
            ReplayEvent::Imu(_, sample) => {
                self.align_fusion.process_imu(fusion_imu_sample(**sample));
                self.fusion.process_imu(fusion_imu_sample(**sample));
                self.process_full_imu(**sample);
            }
            ReplayEvent::Gnss(_, sample) => {
                self.latest_gnss = Some(**sample);
                if let Some(reduced) = self.fusion.reduced() {
                    self.last_reduced_vel_residual = VelResidual {
                        valid: true,
                        value_ned_mps: [
                            sample.vel_ned_mps[0] - reduced.nominal.vn as f64,
                            sample.vel_ned_mps[1] - reduced.nominal.ve as f64,
                            sample.vel_ned_mps[2] - reduced.nominal.vd as f64,
                        ],
                    };
                }
                self.align_fusion.process_gnss(fusion_gnss_sample(**sample));
                self.fusion.process_gnss(fusion_gnss_sample(**sample));
                self.try_initialize_full(**sample);
            }
        }
    }

    fn try_initialize_full(&mut self, sample: GenericGnssSample) {
        if self.full_ready {
            return;
        }
        let speed = sample.vel_ned_mps[0].hypot(sample.vel_ned_mps[1]);
        if speed < 0.5 {
            return;
        }
        let q_mount = match self.mount_mode {
            MountMode::Ref => self.fixed_mount_q_vb,
            MountMode::Internal => self.align_fusion.mount_q_vb(),
        };
        let Some(q_mount) = q_mount else {
            return;
        };
        if matches!(self.mount_mode, MountMode::Internal) && !self.align_fusion.mount_ready() {
            return;
        }

        let yaw_rad = sample.vel_ned_mps[1].atan2(sample.vel_ned_mps[0]) as f32;
        let pos_ecef = lla_to_ecef(sample.lat_deg, sample.lon_deg, sample.height_m);
        let vel_ecef = ned_vector_to_ecef(sample.lat_deg, sample.lon_deg, sample.vel_ned_mps);
        self.full.init_seeded_vehicle_from_nav_ecef_state(
            yaw_rad,
            sample.lat_deg,
            sample.lon_deg,
            pos_ecef,
            vel_ecef.map(|v| v as f32),
            Some(default_full_p_diag(sample, self.cfg.full_init)),
            None,
        );
        self.full.set_mount_quat(q_mount);
        self.full_ready = true;
        self.last_full_gnss_used_t_s = sample.t_s;
    }

    fn process_full_imu(&mut self, sample: GenericImuSample) {
        let Some(prev) = self.last_imu.replace(sample) else {
            return;
        };
        if !self.full_ready {
            return;
        }
        let dt = (sample.t_s - prev.t_s).max(0.0);
        if dt <= 0.0 || dt > 1.0 {
            return;
        }
        self.full.predict(full_imu_delta(prev, sample, dt));

        let mut gps_pos = None;
        let mut gps_vel = None;
        let mut gps_pos_std = 0.0_f32;
        let mut gps_vel_std = None;
        let mut dt_since_gnss = 1.0_f32;
        if let Some(gnss) = self.latest_gnss
            && (0.0..=0.05).contains(&(sample.t_s - gnss.t_s))
            && gnss.t_s != self.last_full_gnss_used_t_s
        {
            gps_pos = Some(lla_to_ecef(gnss.lat_deg, gnss.lon_deg, gnss.height_m));
            gps_vel = Some(ned_vector_to_ecef(
                gnss.lat_deg,
                gnss.lon_deg,
                gnss.vel_ned_mps,
            ));
            gps_pos_std =
                ((gnss.pos_std_m[0] + gnss.pos_std_m[1] + gnss.pos_std_m[2]) / 3.0).max(0.1) as f32;
            gps_vel_std = Some(gnss.vel_std_mps.map(|v| v.max(0.01) as f32));
            dt_since_gnss = if self.last_full_gnss_used_t_s.is_finite() {
                (gnss.t_s - self.last_full_gnss_used_t_s).clamp(1.0e-3, 1.0) as f32
            } else {
                1.0
            };
            let n = self.full.nominal();
            let full_vel_ned = ecef_vector_to_ned(gnss.lat_deg, gnss.lon_deg, [n.vn, n.ve, n.vd]);
            self.last_full_vel_residual = VelResidual {
                valid: true,
                value_ned_mps: [
                    gnss.vel_ned_mps[0] - full_vel_ned[0],
                    gnss.vel_ned_mps[1] - full_vel_ned[1],
                    gnss.vel_ned_mps[2] - full_vel_ned[2],
                ],
            };
            self.last_full_gnss_used_t_s = gnss.t_s;
        }

        let nhc_gate_speed_mps = self.latest_gnss.and_then(|gnss| {
            let age_s = sample.t_s - gnss.t_s;
            (0.0..=FULL_NHC_GNSS_SPEED_MAX_AGE_S)
                .contains(&age_s)
                .then(|| gnss.vel_ned_mps[0].hypot(gnss.vel_ned_mps[1]) as f32)
        });
        self.full.fuse_reference_batch_full_with_nhc_speed_and_r(
            gps_pos,
            gps_vel.map(|v| v.map(|x| x as f32)),
            gps_pos_std,
            gps_vel_std.map(|std| std.map(|v| v as f32)),
            dt_since_gnss,
            nhc_gate_speed_mps,
            self.cfg.r_body_vel,
            self.cfg.r_body_vel_z,
            sample.gyro_radps.map(|v| v as f32),
            sample.accel_mps2.map(|v| v as f32),
            dt as f32,
        );
    }
}

fn write_header(writer: &mut dyn Write) -> Result<()> {
    writeln!(
        writer,
        "t_s,reduced_ready,full_ready,\
reduced_pn_m,reduced_pe_m,reduced_pd_m,full_pn_m,full_pe_m,full_pd_m,diff_pn_m,diff_pe_m,diff_pd_m,\
reduced_vn_mps,reduced_ve_mps,reduced_vd_mps,full_vn_mps,full_ve_mps,full_vd_mps,diff_vn_mps,diff_ve_mps,diff_vd_mps,\
reduced_roll_deg,reduced_pitch_deg,reduced_yaw_deg,full_roll_deg,full_pitch_deg,full_yaw_deg,att_diff_roll_deg,att_diff_pitch_deg,att_diff_yaw_deg,\
ref_roll_deg,ref_pitch_deg,ref_yaw_deg,reduced_att_err_roll_deg,reduced_att_err_pitch_deg,reduced_att_err_yaw_deg,full_att_err_roll_deg,full_att_err_pitch_deg,full_att_err_yaw_deg,\
reduced_mount_roll_deg,reduced_mount_pitch_deg,reduced_mount_yaw_deg,full_mount_roll_deg,full_mount_pitch_deg,full_mount_yaw_deg,mount_diff_roll_deg,mount_diff_pitch_deg,mount_diff_yaw_deg,\
ref_mount_roll_deg,ref_mount_pitch_deg,ref_mount_yaw_deg,reduced_mount_err_roll_deg,reduced_mount_err_pitch_deg,reduced_mount_err_yaw_deg,full_mount_err_roll_deg,full_mount_err_pitch_deg,full_mount_err_yaw_deg,\
reduced_bgx_dps,reduced_bgy_dps,reduced_bgz_dps,full_bgx_dps,full_bgy_dps,full_bgz_dps,gyro_bias_diff_x_dps,gyro_bias_diff_y_dps,gyro_bias_diff_z_dps,\
reduced_bax_mps2,reduced_bay_mps2,reduced_baz_mps2,full_bax_mps2,full_bay_mps2,full_baz_mps2,accel_bias_diff_x_mps2,accel_bias_diff_y_mps2,accel_bias_diff_z_mps2,\
reduced_pos_n_sigma_m,reduced_pos_e_sigma_m,reduced_pos_d_sigma_m,full_pos_n_sigma_m,full_pos_e_sigma_m,full_pos_d_sigma_m,\
reduced_vel_n_sigma_mps,reduced_vel_e_sigma_mps,reduced_vel_d_sigma_mps,full_vel_n_sigma_mps,full_vel_e_sigma_mps,full_vel_d_sigma_mps,\
reduced_roll_sigma_deg,reduced_pitch_sigma_deg,reduced_yaw_sigma_deg,full_roll_sigma_deg,full_pitch_sigma_deg,full_yaw_sigma_deg,\
reduced_mount_roll_sigma_deg,reduced_mount_pitch_sigma_deg,reduced_mount_yaw_sigma_deg,full_mount_roll_sigma_deg,full_mount_pitch_sigma_deg,full_mount_yaw_sigma_deg,\
align_ready,align_roll_deg,align_pitch_deg,align_yaw_deg,align_roll_err_deg,align_pitch_err_deg,align_yaw_err_deg,align_roll_sigma_deg,align_pitch_sigma_deg,align_yaw_sigma_deg,\
reduced_gnss_vel_res_n_mps,reduced_gnss_vel_res_e_mps,reduced_gnss_vel_res_d_mps,full_gnss_vel_res_n_mps,full_gnss_vel_res_e_mps,full_gnss_vel_res_d_mps"
    )?;
    Ok(())
}

fn write_snapshot(
    writer: &mut dyn Write,
    t_s: f64,
    state: &HarnessState,
    replay: &GenericReplayInput,
) -> Result<()> {
    let reduced = state.reduced_snapshot();
    let full = state.full_snapshot();
    let ref_att = reference_rpy_at(&replay.reference_attitude, t_s);
    let ref_mount = reference_rpy_at(&replay.reference_mount, t_s);
    let e = reduced.unwrap_or_else(nan_snapshot);
    let l = full.unwrap_or_else(nan_snapshot);
    let a = ref_att.unwrap_or([f64::NAN; 3]);
    let m = ref_mount.unwrap_or([f64::NAN; 3]);
    let er = state.last_reduced_vel_residual;
    let lr = state.last_full_vel_residual;
    let mut row = vec![
        format!("{t_s:.6}"),
        reduced.is_some().to_string(),
        full.is_some().to_string(),
    ];
    push3(&mut row, e.pos_ned_m);
    push3(&mut row, l.pos_ned_m);
    push3(&mut row, sub3(e.pos_ned_m, l.pos_ned_m));
    push3(&mut row, e.vel_ned_mps);
    push3(&mut row, l.vel_ned_mps);
    push3(&mut row, sub3(e.vel_ned_mps, l.vel_ned_mps));
    push3(&mut row, e.att_rpy_deg);
    push3(&mut row, l.att_rpy_deg);
    push3(&mut row, angle_diff3(e.att_rpy_deg, l.att_rpy_deg));
    push3(&mut row, a);
    push3(&mut row, angle_diff3(e.att_rpy_deg, a));
    push3(&mut row, angle_diff3(l.att_rpy_deg, a));
    push3(&mut row, e.mount_rpy_deg);
    push3(&mut row, l.mount_rpy_deg);
    push3(&mut row, angle_diff3(e.mount_rpy_deg, l.mount_rpy_deg));
    push3(&mut row, m);
    push3(&mut row, angle_diff3(e.mount_rpy_deg, m));
    push3(&mut row, angle_diff3(l.mount_rpy_deg, m));
    push3(&mut row, e.gyro_bias_dps);
    push3(&mut row, l.gyro_bias_dps);
    push3(&mut row, sub3(e.gyro_bias_dps, l.gyro_bias_dps));
    push3(&mut row, e.accel_bias_mps2);
    push3(&mut row, l.accel_bias_mps2);
    push3(&mut row, sub3(e.accel_bias_mps2, l.accel_bias_mps2));
    push3(&mut row, e.pos_sigma_m);
    push3(&mut row, l.pos_sigma_m);
    push3(&mut row, e.vel_sigma_mps);
    push3(&mut row, l.vel_sigma_mps);
    push3(&mut row, e.att_sigma_deg);
    push3(&mut row, l.att_sigma_deg);
    push3(&mut row, e.mount_sigma_deg);
    push3(&mut row, l.mount_sigma_deg);
    let align_snapshot = state.align_snapshot();
    let align_rpy = align_snapshot
        .map(|snapshot| snapshot.0)
        .unwrap_or([f64::NAN; 3]);
    let align_sigma = align_snapshot
        .map(|snapshot| snapshot.1)
        .unwrap_or([f64::NAN; 3]);
    row.push(
        align_snapshot
            .is_some_and(|_| state.align_fusion.mount_ready())
            .to_string(),
    );
    push3(&mut row, align_rpy);
    push3(&mut row, angle_diff3(align_rpy, m));
    push3(&mut row, align_sigma);
    push3(
        &mut row,
        [
            residual_or_nan(er, 0),
            residual_or_nan(er, 1),
            residual_or_nan(er, 2),
        ],
    );
    push3(
        &mut row,
        [
            residual_or_nan(lr, 0),
            residual_or_nan(lr, 1),
            residual_or_nan(lr, 2),
        ],
    );
    writeln!(writer, "{}", row.join(","))?;
    Ok(())
}

fn reference_rpy_at(samples: &[GenericReferenceRpySample], t_s: f64) -> Option<[f64; 3]> {
    if samples.is_empty() || !t_s.is_finite() {
        return None;
    }
    let idx = samples.partition_point(|sample| sample.t_s < t_s);
    let nearest = match (idx.checked_sub(1), samples.get(idx)) {
        (Some(prev_idx), Some(next)) => {
            let prev = &samples[prev_idx];
            if (t_s - prev.t_s).abs() <= (next.t_s - t_s).abs() {
                prev
            } else {
                next
            }
        }
        (Some(prev_idx), None) => &samples[prev_idx],
        (None, Some(next)) => next,
        (None, None) => return None,
    };
    Some([nearest.roll_deg, nearest.pitch_deg, nearest.yaw_deg])
}

impl HarnessState {
    fn reduced_snapshot(&self) -> Option<CommonSnapshot> {
        let reduced = self.fusion.reduced()?;
        let pos = self.reduced_display_position_ned(reduced);
        let vel = self.reduced_display_velocity_ned(reduced);
        let q_vehicle = [
            reduced.nominal.q0,
            reduced.nominal.q1,
            reduced.nominal.q2,
            reduced.nominal.q3,
        ];
        let (roll, pitch, yaw) =
            quat_rpy_deg(q_vehicle[0], q_vehicle[1], q_vehicle[2], q_vehicle[3]);
        let q_mount = [
            reduced.nominal.qcs0 as f64,
            reduced.nominal.qcs1 as f64,
            reduced.nominal.qcs2 as f64,
            reduced.nominal.qcs3 as f64,
        ];
        let (mr, mp, my) = q_vb_to_reference_mount_rpy(q_mount);
        Some(CommonSnapshot {
            pos_ned_m: pos,
            vel_ned_mps: vel,
            att_rpy_deg: [roll, pitch, yaw].map(|v| v as f64),
            mount_rpy_deg: [mr, mp, my],
            gyro_bias_dps: [
                (reduced.nominal.bgx as f64).to_degrees(),
                (reduced.nominal.bgy as f64).to_degrees(),
                (reduced.nominal.bgz as f64).to_degrees(),
            ],
            accel_bias_mps2: [
                reduced.nominal.bax as f64,
                reduced.nominal.bay as f64,
                reduced.nominal.baz as f64,
            ],
            pos_sigma_m: diag_sigma3(&reduced.p, [6, 7, 8], 1.0),
            vel_sigma_mps: diag_sigma3(&reduced.p, [3, 4, 5], 1.0),
            att_sigma_deg: diag_sigma3(&reduced.p, [0, 1, 2], 180.0 / core::f64::consts::PI),
            mount_sigma_deg: diag_sigma3(&reduced.p, [15, 16, 17], 180.0 / core::f64::consts::PI),
        })
    }

    fn full_snapshot(&self) -> Option<CommonSnapshot> {
        if !self.full_ready {
            return None;
        }
        let n = self.full.nominal();
        let pos = ecef_to_ned(
            self.full.shadow_pos_ecef(),
            self.ref_ecef,
            self.ref_gnss.lat_deg,
            self.ref_gnss.lon_deg,
        );
        let vel = ecef_vector_to_ned(
            self.ref_gnss.lat_deg,
            self.ref_gnss.lon_deg,
            [n.vn, n.ve, n.vd],
        );
        let (lat, lon, _) = ecef_to_lla(self.full.shadow_pos_ecef());
        let q_ns = quat_mul(
            quat_ecef_to_ned(lat, lon),
            [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64],
        );
        let (roll, pitch, yaw) = quat_rpy_deg(
            q_ns[0] as f32,
            q_ns[1] as f32,
            q_ns[2] as f32,
            q_ns[3] as f32,
        );
        let (mr, mp, my) = q_vb_to_reference_mount_rpy([
            n.qcs0 as f64,
            n.qcs1 as f64,
            n.qcs2 as f64,
            n.qcs3 as f64,
        ]);
        Some(CommonSnapshot {
            pos_ned_m: pos,
            vel_ned_mps: vel,
            att_rpy_deg: [roll, pitch, yaw].map(|v| v as f64),
            mount_rpy_deg: [mr, mp, my],
            gyro_bias_dps: [
                -(n.bgx as f64).to_degrees(),
                -(n.bgy as f64).to_degrees(),
                -(n.bgz as f64).to_degrees(),
            ],
            accel_bias_mps2: [-(n.bax as f64), -(n.bay as f64), -(n.baz as f64)],
            pos_sigma_m: diag_sigma3(self.full.covariance(), [0, 1, 2], 1.0),
            vel_sigma_mps: diag_sigma3(self.full.covariance(), [3, 4, 5], 1.0),
            att_sigma_deg: diag_sigma3(
                self.full.covariance(),
                [6, 7, 8],
                180.0 / core::f64::consts::PI,
            ),
            mount_sigma_deg: diag_sigma3(
                self.full.covariance(),
                [21, 22, 23],
                180.0 / core::f64::consts::PI,
            ),
        })
    }

    fn reduced_display_position_ned(&self, reduced: &State) -> [f64; 3] {
        if let Some(lla) = self.fusion.position_lla_f64() {
            return ecef_to_ned(
                lla_to_ecef(lla[0], lla[1], lla[2]),
                self.ref_ecef,
                self.ref_gnss.lat_deg,
                self.ref_gnss.lon_deg,
            );
        }
        [
            reduced.nominal.pn as f64,
            reduced.nominal.pe as f64,
            reduced.nominal.pd as f64,
        ]
    }

    fn reduced_display_velocity_ned(&self, reduced: &State) -> [f64; 3] {
        if let Some(lla) = self.fusion.position_lla_f64() {
            let vel = [reduced.nominal.vn, reduced.nominal.ve, reduced.nominal.vd];
            let c_ne_local = ecef_to_ned_matrix(lla[0], lla[1]);
            let c_en_local = transpose3(c_ne_local);
            let vel_ecef = mat3_vec(c_en_local, vel.map(|v| v as f64));
            return ecef_vector_to_ned(
                self.ref_gnss.lat_deg,
                self.ref_gnss.lon_deg,
                vel_ecef.map(|v| v as f32),
            );
        }
        [
            reduced.nominal.vn as f64,
            reduced.nominal.ve as f64,
            reduced.nominal.vd as f64,
        ]
    }

    fn align_snapshot(&self) -> Option<([f64; 3], [f64; 3])> {
        let align = self.align_fusion.align()?;
        let (roll_deg, pitch_deg, yaw_deg) = q_vb_to_reference_mount_rpy([
            align.q_vb[0] as f64,
            align.q_vb[1] as f64,
            align.q_vb[2] as f64,
            align.q_vb[3] as f64,
        ]);
        let rpy = [roll_deg, pitch_deg, yaw_deg];
        let sigma = align.sigma_deg().map(|v| v as f64);
        Some((rpy, sigma))
    }
}

fn nan_snapshot() -> CommonSnapshot {
    CommonSnapshot {
        pos_ned_m: [f64::NAN; 3],
        vel_ned_mps: [f64::NAN; 3],
        att_rpy_deg: [f64::NAN; 3],
        mount_rpy_deg: [f64::NAN; 3],
        gyro_bias_dps: [f64::NAN; 3],
        accel_bias_mps2: [f64::NAN; 3],
        pos_sigma_m: [f64::NAN; 3],
        vel_sigma_mps: [f64::NAN; 3],
        att_sigma_deg: [f64::NAN; 3],
        mount_sigma_deg: [f64::NAN; 3],
    }
}

fn apply_fusion_config(fusion: &mut SensorFusion, cfg: FilterCompareConfig) {
    fusion.set_align_config(cfg.align);
    if let Some(noise) = cfg.noise.reduced {
        fusion.set_reduced_noise(noise);
    }
    fusion.set_r_body_vel_yz(cfg.r_body_vel, cfg.r_body_vel_z);
    fusion.set_attitude_roll_pitch_init_sigma_rad(
        cfg.attitude_roll_pitch_init_sigma_deg.to_radians(),
    );
    fusion.set_yaw_init_sigma_rad(cfg.yaw_init_sigma_deg.to_radians());
    fusion.set_gyro_bias_init_sigma_radps(cfg.gyro_bias_init_sigma_dps.to_radians());
    fusion.set_accel_bias_init_sigma_mps2(cfg.accel_bias_init_sigma_mps2);
    fusion.set_mount_roll_pitch_init_sigma_rad(cfg.mount_roll_pitch_init_sigma_deg.to_radians());
    fusion.set_mount_init_sigma_rad(cfg.mount_init_sigma_deg.to_radians());
    fusion.set_r_vehicle_speed(cfg.r_vehicle_speed);
    fusion.set_r_zero_vel(cfg.r_zero_vel);
    fusion.set_r_stationary_accel(cfg.r_stationary_accel);
    fusion.set_mount_align_rw_var(cfg.mount_align_rw_var);
    fusion.set_align_handoff_delay_s(cfg.align_handoff_delay_s);
    fusion.set_freeze_misalignment_states(cfg.freeze_misalignment_states);
    fusion.set_mount_source(MountSource::LatchedSeed);
    fusion.set_mount_settle_time_s(cfg.mount_settle_time_s);
    fusion.set_mount_settle_release_sigma_rad(cfg.mount_settle_release_sigma_deg.to_radians());
    fusion.set_mount_settle_zero_cross_covariance(cfg.mount_settle_zero_cross_covariance);
}

fn default_full_p_diag(gnss: GenericGnssSample, init: InitConfig) -> [f32; ERROR_STATES] {
    let mut p = [1.0_f32; ERROR_STATES];
    let pos_n_sigma = (gnss.pos_std_m[0] as f32).max(init.pos_min_sigma_m);
    let pos_e_sigma = (gnss.pos_std_m[1] as f32).max(init.pos_min_sigma_m);
    let pos_d_sigma = (gnss.pos_std_m[2] as f32).max(init.pos_min_sigma_m);
    p[0] = pos_n_sigma * pos_n_sigma;
    p[1] = pos_e_sigma * pos_e_sigma;
    p[2] = pos_d_sigma * pos_d_sigma;
    let vel_sigma = gnss
        .vel_std_mps
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(init.vel_min_sigma_mps as f64) as f32;
    let vel_var = vel_sigma * vel_sigma;
    p[3] = vel_var;
    p[4] = vel_var;
    p[5] = vel_var;
    let attitude_var = init.attitude_sigma_deg.to_radians().powi(2);
    p[6] = attitude_var;
    p[7] = attitude_var;
    p[8] = attitude_var;
    let accel_bias_var = init.accel_bias_sigma_mps2 * init.accel_bias_sigma_mps2;
    p[9] = accel_bias_var;
    p[10] = accel_bias_var;
    p[11] = accel_bias_var;
    let gyro_bias_var = init.gyro_bias_sigma_dps.to_radians().powi(2);
    p[12] = gyro_bias_var;
    p[13] = gyro_bias_var;
    p[14] = gyro_bias_var;
    p[15] = init.accel_scale_sigma * init.accel_scale_sigma;
    p[16] = p[15];
    p[17] = p[15];
    p[18] = init.gyro_scale_sigma * init.gyro_scale_sigma;
    p[19] = p[18];
    p[20] = p[18];
    p[21] = init.mount_sigma_deg.to_radians().powi(2);
    p[22] = p[21];
    p[23] = init.mount_yaw_sigma_deg.to_radians().powi(2);
    p
}

fn full_imu_delta(prev: GenericImuSample, curr: GenericImuSample, dt: f64) -> ImuDelta {
    ImuDelta {
        dax_1: (prev.gyro_radps[0] * dt) as f32,
        day_1: (prev.gyro_radps[1] * dt) as f32,
        daz_1: (prev.gyro_radps[2] * dt) as f32,
        dvx_1: (prev.accel_mps2[0] * dt) as f32,
        dvy_1: (prev.accel_mps2[1] * dt) as f32,
        dvz_1: (prev.accel_mps2[2] * dt) as f32,
        dax_2: (curr.gyro_radps[0] * dt) as f32,
        day_2: (curr.gyro_radps[1] * dt) as f32,
        daz_2: (curr.gyro_radps[2] * dt) as f32,
        dvx_2: (curr.accel_mps2[0] * dt) as f32,
        dvy_2: (curr.accel_mps2[1] * dt) as f32,
        dvz_2: (curr.accel_mps2[2] * dt) as f32,
        dt: dt as f32,
    }
}

fn event_time(event: &ReplayEvent<'_>) -> f64 {
    match event {
        ReplayEvent::Imu(_, sample) => sample.t_s,
        ReplayEvent::Gnss(_, sample) => sample.t_s,
    }
}

fn residual_or_nan(residual: VelResidual, axis: usize) -> f64 {
    residual
        .valid
        .then_some(residual.value_ned_mps[axis])
        .unwrap_or(f64::NAN)
}

fn push3(row: &mut Vec<String>, values: [f64; 3]) {
    for value in values {
        row.push(format!("{value:.9}"));
    }
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn angle_diff3(a_deg: [f64; 3], b_deg: [f64; 3]) -> [f64; 3] {
    [
        wrap_deg(a_deg[0] - b_deg[0]),
        wrap_deg(a_deg[1] - b_deg[1]),
        wrap_deg(a_deg[2] - b_deg[2]),
    ]
}

fn wrap_deg(mut deg: f64) -> f64 {
    while deg > 180.0 {
        deg -= 360.0;
    }
    while deg < -180.0 {
        deg += 360.0;
    }
    deg
}

fn diag_sigma3<const N: usize>(p: &[[f32; N]; N], idx: [usize; 3], scale: f64) -> [f64; 3] {
    [
        (p[idx[0]][idx[0]].max(0.0) as f64).sqrt() * scale,
        (p[idx[1]][idx[1]].max(0.0) as f64).sqrt() * scale,
        (p[idx[2]][idx[2]].max(0.0) as f64).sqrt() * scale,
    ]
}

fn ned_vector_to_ecef(lat_deg: f64, lon_deg: f64, v_ned: [f64; 3]) -> [f64; 3] {
    let c_ne = ecef_to_ned_matrix(lat_deg, lon_deg);
    [
        c_ne[0][0] * v_ned[0] + c_ne[1][0] * v_ned[1] + c_ne[2][0] * v_ned[2],
        c_ne[0][1] * v_ned[0] + c_ne[1][1] * v_ned[1] + c_ne[2][1] * v_ned[2],
        c_ne[0][2] * v_ned[0] + c_ne[1][2] * v_ned[1] + c_ne[2][2] * v_ned[2],
    ]
}

fn ecef_vector_to_ned(lat_deg: f64, lon_deg: f64, v_ecef: [f32; 3]) -> [f64; 3] {
    let c_ne = ecef_to_ned_matrix(lat_deg, lon_deg);
    let v = [v_ecef[0] as f64, v_ecef[1] as f64, v_ecef[2] as f64];
    [
        c_ne[0][0] * v[0] + c_ne[0][1] * v[1] + c_ne[0][2] * v[2],
        c_ne[1][0] * v[0] + c_ne[1][1] * v[1] + c_ne[1][2] * v[2],
        c_ne[2][0] * v[0] + c_ne[2][1] * v[1] + c_ne[2][2] * v[2],
    ]
}

fn ecef_to_ned_matrix(lat_deg: f64, lon_deg: f64) -> [[f64; 3]; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    [
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [-clat * clon, -clat * slon, -slat],
    ]
}

fn quat_ecef_to_ned(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    let r = ecef_to_ned_matrix(lat_deg, lon_deg);
    rotmat_to_quat(r)
}

fn rotmat_to_quat(r: [[f64; 3]; 3]) -> [f64; 4] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        [
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        ]
    };
    quat_normalize(q)
}

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_normalize(mut q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 0.0 {
        for item in &mut q {
            *item /= n;
        }
    }
    q
}

fn transpose3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mat3_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}
