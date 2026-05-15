use anyhow::{Context, Result};
use clap::Parser;
use sensor_fusion::ProcessNoise;
use sensor_fusion::ekf::{self, Filter, GnssSample, ImuDelta};
use sim::datasets::generic_replay::{GenericGnssSample, GenericImuSample};
use sim::eval::gnss_ins::quat_angle_deg;
use sim::eval::replay::{ReplayEvent, for_each_event};
use sim::visualizer::math::{ecef_to_ned, lla_to_ecef};
use sim::visualizer::pipeline::generic::{GenericReplayInput, reference_mount_rpy_to_q_bv};
use sim::visualizer::pipeline::synthetic::{
    SyntheticNoiseMode, SyntheticVisualizerConfig, build_synthetic_replay_input,
};

const G: f64 = 9.80665;

#[derive(Parser, Debug)]
#[command(name = "diag_mount_observability")]
#[command(about = "Synthetic roll/pitch mount observability proof-of-theory diagnostic")]
struct Args {
    /// Initial injected mount roll error, degrees.
    #[arg(long, default_value_t = 3.0)]
    seed_roll_error_deg: f64,
    /// Initial injected mount pitch error, degrees.
    #[arg(long, default_value_t = 3.0)]
    seed_pitch_error_deg: f64,
    /// IMU rate used for generated scenarios.
    #[arg(long, default_value_t = 100.0)]
    imu_hz: f64,
    /// GNSS rate used for generated scenarios.
    #[arg(long, default_value_t = 2.0)]
    gnss_hz: f64,
    /// Lateral NHC variance density.
    #[arg(long, default_value_t = 0.5)]
    r_nhc_y: f64,
    /// Vertical NHC variance density.
    #[arg(long, default_value_t = 0.5)]
    r_nhc_z: f64,
    /// NHC update period, seconds. The runtime default is 0.1 s.
    #[arg(long, default_value_t = 0.1)]
    nhc_period_s: f64,
}

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    text: &'static str,
}

#[derive(Clone, Copy, Default)]
struct Observability {
    roll_joint_marginal: f64,
    pitch_joint_marginal: f64,
    ay_rms: f64,
    ax_rms: f64,
}

#[derive(Clone, Copy, Default)]
struct RunResult {
    mount_quat_error_deg: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let scenarios = scenarios();
    println!(
        "mount observability diagnostic: seed error roll={:.1} deg pitch={:.1} deg, NHC Ry={:.3}, Rz={:.3}, period={:.2}s",
        args.seed_roll_error_deg,
        args.seed_pitch_error_deg,
        args.r_nhc_y,
        args.r_nhc_z,
        args.nhc_period_s
    );
    println!(
        "{:<24} {:>7} {:>7} {:>11} {:>11} {:>12} {:>12} {:>11}",
        "scenario",
        "axRMS",
        "ayRMS",
        "Iroll|all",
        "Ipitch|all",
        "qerr_mount_flex",
        "qerr_att_flex",
        "seed_qerr"
    );
    for scenario in scenarios {
        let synth = SyntheticVisualizerConfig {
            motion_def: None,
            motion_label: scenario.name.to_string(),
            motion_text: Some(scenario.text.to_string()),
            noise_mode: SyntheticNoiseMode::Truth,
            disable_imu_noise: true,
            disable_gnss_noise: true,
            seed: 1,
            mount_rpy_deg: [0.0, 0.0, 0.0],
            imu_hz: args.imu_hz,
            gnss_hz: args.gnss_hz,
            gnss_time_shift_ms: 0.0,
            early_vel_bias_ned_mps: [0.0; 3],
            early_fault_window_s: None,
        };
        let (replay, truth_mount_q_f32) = build_synthetic_replay_input(&synth)
            .with_context(|| format!("failed to generate {}", scenario.name))?;
        let truth_mount_q = q64(truth_mount_q_f32);
        let obs = compute_observability(&replay, &args);
        let mount_flex = run_ekf_bad_seed(&replay, truth_mount_q, &args, 2.0, 6.0)
            .with_context(|| format!("failed to run EKF for {}", scenario.name))?;
        let attitude_flex = run_ekf_bad_seed(&replay, truth_mount_q, &args, 6.0, 2.0)
            .with_context(|| format!("failed to run EKF for {}", scenario.name))?;
        let seed_q =
            reference_mount_rpy_to_q_bv([args.seed_roll_error_deg, args.seed_pitch_error_deg, 0.0]);
        println!(
            "{:<24} {:>7.3} {:>7.3} {:>11.3e} {:>11.3e} {:>12.3} {:>12.3} {:>11.3}",
            scenario.name,
            obs.ax_rms,
            obs.ay_rms,
            obs.roll_joint_marginal,
            obs.pitch_joint_marginal,
            mount_flex.mount_quat_error_deg,
            attitude_flex.mount_quat_error_deg,
            quat_angle_deg(seed_q, truth_mount_q),
        );
    }
    println!();
    println!(
        "Interpretation: Iroll|all and Ipitch|all are Schur-complement information proxies after marginalizing vehicle attitude, accel bias, and velocity nuisance states."
    );
    println!(
        "qerr_mount_flex uses tight attitude covariance and loose mount covariance; qerr_att_flex swaps those priors."
    );
    println!(
        "If a maneuver is weakly observable, the final mount error should depend strongly on that covariance split. Strong unique excitation should reduce that dependence."
    );
    Ok(())
}

fn scenarios() -> [Scenario; 4] {
    [
        Scenario {
            name: "straight_constant",
            text: r#"
initial lat=32 lon=120 alt=0 vx=10 vy=0 vz=0 yaw=0 pitch=0 roll=0
wait for=480s
"#,
        },
        Scenario {
            name: "accel_brake",
            text: r#"
initial lat=32 lon=120 alt=0 vx=6 vy=0 vz=0 yaw=0 pitch=0 roll=0
repeat 12 {
  accelerate 1.0 for=15s
  coast for=10s
  brake 1.0 for=15s
  coast for=10s
}
"#,
        },
        Scenario {
            name: "turns_only",
            text: r#"
initial lat=32 lon=120 alt=0 vx=10 vy=0 vz=0 yaw=0 pitch=0 roll=0
repeat 10 {
  turn left 10 for=24s
  turn right 10 for=24s
}
"#,
        },
        Scenario {
            name: "turns_plus_accel",
            text: r#"
initial lat=32 lon=120 alt=0 vx=8 vy=0 vz=0 yaw=0 pitch=0 roll=0
repeat 8 {
  accelerate 0.8 for=10s
  turn left 10 for=20s
  brake 0.8 for=10s
  turn right 10 for=20s
}
"#,
        },
    ]
}

fn compute_observability(replay: &GenericReplayInput, args: &Args) -> Observability {
    let mut roll = Info6::default();
    let mut pitch = Info6::default();
    let mut last_t: Option<f64> = None;
    let mut last_nhc_t = None;
    let mut ax2_sum = 0.0;
    let mut ay2_sum = 0.0;
    let mut accel_count = 0.0;

    for motion in &replay.reference_motion {
        let t = motion.t_s;
        let dt = last_t
            .map(|prev| (t - prev).max(1.0e-3))
            .unwrap_or(1.0 / args.imu_hz);
        last_t = Some(t);
        let ax = motion.accel_vehicle_mps2[0];
        let ay = motion.accel_vehicle_mps2[1];
        ax2_sum += ax * ax;
        ay2_sum += ay * ay;
        accel_count += 1.0;

        let nhc_due = last_nhc_t
            .map(|prev| t - prev + 1.0e-6 >= args.nhc_period_s)
            .unwrap_or(true);
        if nhc_due {
            last_nhc_t = Some(t);
            let obs_dt = args.nhc_period_s.max(dt);
            let r_y = args.r_nhc_y / obs_dt;
            let r_z = args.r_nhc_z / obs_dt;
            roll.add_row([-G * obs_dt, G * obs_dt, obs_dt, 0.0, -1.0, 0.0], r_y);
            roll.add_row([0.0, ay * obs_dt, 0.0, obs_dt, 0.0, -1.0], r_z);
            pitch.add_row([G * obs_dt, -G * obs_dt, obs_dt, 0.0, -1.0, 0.0], r_y);
            pitch.add_row([0.0, -ax * obs_dt, 0.0, obs_dt, 0.0, -1.0], r_z);
        }
    }

    let mut prev_gnss_t: Option<f64> = None;
    for gnss in &replay.gnss {
        let dt = prev_gnss_t
            .map(|prev| (gnss.t_s - prev).clamp(1.0e-3, 1.0))
            .unwrap_or(1.0 / args.gnss_hz);
        prev_gnss_t = Some(gnss.t_s);
        let r_vx = gnss.vel_std_mps[0] * gnss.vel_std_mps[0] / dt;
        let r_vy = gnss.vel_std_mps[1] * gnss.vel_std_mps[1] / dt;
        let r_vz = gnss.vel_std_mps[2] * gnss.vel_std_mps[2] / dt;
        roll.add_row([0.0, 0.0, 0.0, 0.0, -1.0, 0.0], r_vy);
        roll.add_row([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], r_vz);
        pitch.add_row([0.0, 0.0, 0.0, 0.0, -1.0, 0.0], r_vx);
        pitch.add_row([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], r_vz);
    }

    Observability {
        roll_joint_marginal: roll.marginal_mount_info(&[0, 2, 3, 4, 5]),
        pitch_joint_marginal: pitch.marginal_mount_info(&[0, 2, 3, 4, 5]),
        ay_rms: (ay2_sum / f64::max(accel_count, 1.0)).sqrt(),
        ax_rms: (ax2_sum / f64::max(accel_count, 1.0)).sqrt(),
    }
}

fn run_ekf_bad_seed(
    replay: &GenericReplayInput,
    truth_mount_q: [f64; 4],
    args: &Args,
    attitude_sigma_deg: f64,
    mount_sigma_deg: f64,
) -> Result<RunResult> {
    let first_gnss = replay.gnss.first().context("missing GNSS")?;
    let ref_ecef = lla_to_ecef(first_gnss.lat_deg, first_gnss.lon_deg, first_gnss.height_m);
    let mut filter = Filter::new(ProcessNoise::default());
    let init = ekf_gnss(first_gnss, ref_ecef, first_gnss);
    let yaw = first_gnss.vel_ned_mps[1].atan2(first_gnss.vel_ned_mps[0]);
    filter.init_nominal_from_gnss(quat_from_yaw(yaw), init);
    let seed_q =
        reference_mount_rpy_to_q_bv([args.seed_roll_error_deg, args.seed_pitch_error_deg, 0.0]);
    {
        let raw = filter.raw_mut();
        raw.nominal.q_bv0 = seed_q[0] as f32;
        raw.nominal.q_bv1 = seed_q[1] as f32;
        raw.nominal.q_bv2 = seed_q[2] as f32;
        raw.nominal.q_bv3 = seed_q[3] as f32;
        let att_var = attitude_sigma_deg.to_radians().powi(2) as f32;
        raw.p[0][0] = att_var;
        raw.p[1][1] = att_var;
        let mount_var = mount_sigma_deg.to_radians().powi(2) as f32;
        raw.p[15][15] = mount_var;
        raw.p[16][16] = mount_var;
        raw.p[17][17] = mount_var;
    }

    let mut last_imu: Option<GenericImuSample> = None;
    let mut last_nhc_t = None;
    let mut last_gnss_t = Some(first_gnss.t_s);
    let mut skipped_first_gnss = false;
    for_each_event(&replay.imu, &replay.gnss, |event| match event {
        ReplayEvent::Imu(_, sample) => {
            if let Some(prev) = last_imu {
                let dt = (sample.t_s - prev.t_s) as f32;
                if (0.001..=0.05).contains(&dt) {
                    filter.predict(ImuDelta {
                        dax: (sample.gyro_radps[0] * dt as f64) as f32,
                        day: (sample.gyro_radps[1] * dt as f64) as f32,
                        daz: (sample.gyro_radps[2] * dt as f64) as f32,
                        dvx: (0.5 * (prev.accel_mps2[0] + sample.accel_mps2[0]) * dt as f64) as f32,
                        dvy: (0.5 * (prev.accel_mps2[1] + sample.accel_mps2[1]) * dt as f64) as f32,
                        dvz: (0.5 * (prev.accel_mps2[2] + sample.accel_mps2[2]) * dt as f64) as f32,
                        dt,
                    });
                    let t = sample.t_s;
                    let due = last_nhc_t
                        .map(|prev_t| t - prev_t + 1.0e-6 >= args.nhc_period_s)
                        .unwrap_or(true);
                    if due && speed_ned(filter.raw().nominal.vn, filter.raw().nominal.ve) > 0.05 {
                        let obs_dt = last_nhc_t.map(|prev_t| t - prev_t).unwrap_or(dt as f64);
                        last_nhc_t = Some(t);
                        filter.fuse_body_vel_yz(
                            (args.r_nhc_y / obs_dt.max(1.0e-3)) as f32,
                            (args.r_nhc_z / obs_dt.max(1.0e-3)) as f32,
                        );
                    }
                }
            }
            last_imu = Some(*sample);
        }
        ReplayEvent::Gnss(_, sample) => {
            if !skipped_first_gnss {
                skipped_first_gnss = true;
                return;
            }
            let dt = (sample.t_s - last_gnss_t.unwrap_or(sample.t_s)).clamp(1.0e-3, 1.0);
            last_gnss_t = Some(sample.t_s);
            let mut gnss = ekf_gnss(sample, ref_ecef, first_gnss);
            gnss.pos_std_m = [1.0e6, 1.0e6, 1.0e6];
            let std_scale = (1.0 / dt).sqrt() as f32;
            for std in &mut gnss.pos_std_m {
                *std *= std_scale;
            }
            filter.fuse_gps(gnss);
        }
    });

    let q = final_mount_q(filter.raw());
    Ok(RunResult {
        mount_quat_error_deg: quat_angle_deg(q, truth_mount_q),
    })
}

fn q64(q: [f32; 4]) -> [f64; 4] {
    [q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64]
}

fn ekf_gnss(
    sample: &GenericGnssSample,
    ref_ecef: [f64; 3],
    ref_sample: &GenericGnssSample,
) -> GnssSample {
    let ecef = lla_to_ecef(sample.lat_deg, sample.lon_deg, sample.height_m);
    let ned = ecef_to_ned(ecef, ref_ecef, ref_sample.lat_deg, ref_sample.lon_deg);
    GnssSample {
        t_s: sample.t_s as f32,
        pos_ned_m: [ned[0] as f32, ned[1] as f32, ned[2] as f32],
        vel_ned_mps: [
            sample.vel_ned_mps[0] as f32,
            sample.vel_ned_mps[1] as f32,
            sample.vel_ned_mps[2] as f32,
        ],
        pos_std_m: [
            sample.pos_std_m[0] as f32,
            sample.pos_std_m[1] as f32,
            sample.pos_std_m[2] as f32,
        ],
        vel_std_mps: [
            sample.vel_std_mps[0] as f32,
            sample.vel_std_mps[1] as f32,
            sample.vel_std_mps[2] as f32,
        ],
        heading_rad: sample.heading_rad.map(|v| v as f32),
    }
}

fn final_mount_q(state: &ekf::State) -> [f64; 4] {
    [
        state.nominal.q_bv0 as f64,
        state.nominal.q_bv1 as f64,
        state.nominal.q_bv2 as f64,
        state.nominal.q_bv3 as f64,
    ]
}

fn speed_ned(vn: f32, ve: f32) -> f32 {
    (vn * vn + ve * ve).sqrt()
}

fn quat_from_yaw(yaw: f64) -> [f32; 4] {
    let half = 0.5 * yaw;
    [half.cos() as f32, 0.0, 0.0, half.sin() as f32]
}

#[derive(Clone, Copy, Default)]
struct Info6 {
    m: [[f64; 6]; 6],
}

impl Info6 {
    fn add_row(&mut self, h: [f64; 6], variance: f64) {
        if !variance.is_finite() || variance <= 0.0 {
            return;
        }
        let w = 1.0 / variance;
        for r in 0..6 {
            for c in 0..6 {
                self.m[r][c] += h[r] * w * h[c];
            }
        }
    }

    fn marginal_mount_info(&self, nuisance: &[usize]) -> f64 {
        let lambda_aa = self.m[1][1];
        if nuisance.is_empty() {
            return lambda_aa.max(0.0);
        }
        let mut a = [[0.0; 5]; 5];
        let mut b = [0.0; 5];
        for (ri, &r) in nuisance.iter().enumerate() {
            b[ri] = self.m[r][1];
            for (ci, &c) in nuisance.iter().enumerate() {
                a[ri][ci] = self.m[r][c];
            }
        }
        let Some(x) = solve_linear(&mut a, &mut b, nuisance.len()) else {
            return 0.0;
        };
        let reduction = nuisance
            .iter()
            .enumerate()
            .map(|(i, &idx)| self.m[1][idx] * x[i])
            .sum::<f64>();
        (lambda_aa - reduction).max(0.0)
    }
}

fn solve_linear(a: &mut [[f64; 5]; 5], b: &mut [f64; 5], n: usize) -> Option<[f64; 5]> {
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();
        for (row, row_values) in a.iter().enumerate().take(n).skip(col + 1) {
            let v = row_values[col].abs();
            if v > pivot_abs {
                pivot = row;
                pivot_abs = v;
            }
        }
        if pivot_abs < 1.0e-12 {
            return None;
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        let diag = a[col][col];
        for c in col..n {
            a[col][c] /= diag;
        }
        b[col] /= diag;
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = a[row][col];
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                a[row][c] -= f * a[col][c];
            }
            b[row] -= f * b[col];
        }
    }
    Some(*b)
}
