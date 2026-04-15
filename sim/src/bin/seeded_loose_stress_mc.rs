use anyhow::{Context, Result, bail};
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use sensor_fusion::c_api::{CLooseImuDelta, CLooseWrapper};
use sensor_fusion::loose::LoosePredictNoise;
use sim::visualizer::math::{ecef_to_lla, lla_to_ecef};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    input_dir: PathBuf,
    #[arg(long, default_value_t = 100.0)]
    init_time_s: f64,
    #[arg(long, default_value_t = 60.0)]
    replay_duration_s: f64,
    #[arg(long, default_value_t = 24)]
    monte_carlo_runs: usize,
    #[arg(long, default_value_t = 1234)]
    monte_carlo_seed: u64,
    #[arg(long, default_value_t = 1)]
    jobs: usize,
    #[arg(long, default_value_t = -4.0)]
    mc_seed_roll_min_deg: f64,
    #[arg(long, default_value_t = 4.0)]
    mc_seed_roll_max_deg: f64,
    #[arg(long, default_value_t = -4.0)]
    mc_seed_pitch_min_deg: f64,
    #[arg(long, default_value_t = 4.0)]
    mc_seed_pitch_max_deg: f64,
    #[arg(long, default_value_t = -6.0)]
    mc_seed_yaw_min_deg: f64,
    #[arg(long, default_value_t = 6.0)]
    mc_seed_yaw_max_deg: f64,
    #[arg(long, default_value_t = -2.5)]
    mc_gps_vd_bias_min_mps: f64,
    #[arg(long, default_value_t = 0.0)]
    mc_gps_vd_bias_max_mps: f64,
    #[arg(long, default_value_t = 0.0)]
    mc_gps_vd_bias_duration_min_s: f64,
    #[arg(long, default_value_t = 20.0)]
    mc_gps_vd_bias_duration_max_s: f64,
    #[arg(long, default_value_t = 0.1)]
    mc_vel_std_scale_min: f64,
    #[arg(long, default_value_t = 1.0)]
    mc_vel_std_scale_max: f64,
    #[arg(long, default_value_t = -0.15)]
    mc_accel_z_bias_min_mps2: f64,
    #[arg(long, default_value_t = 0.15)]
    mc_accel_z_bias_max_mps2: f64,
    #[arg(long, default_value_t = 5.0)]
    mc_mount_err_threshold_deg: f64,
    #[arg(long, default_value_t = 3.0)]
    mc_car_pitch_err_threshold_deg: f64,
    #[arg(long)]
    summary_csv: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct GyroSample {
    ttag_us: i64,
    omega_radps: [f64; 3],
}

#[derive(Clone, Debug)]
struct AccelSample {
    accel_mps2: [f64; 3],
}

#[derive(Clone, Debug)]
struct GnssVelocity {
    vel_n: [f64; 3],
    vel_acc_n: [f64; 3],
}

#[derive(Clone, Debug)]
struct GnssSample {
    ttag_us: i64,
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    speed_mps: f64,
    heading_deg: f64,
    h_acc_m: f64,
    v_acc_m: f64,
    velocity: Option<GnssVelocity>,
}

#[derive(Clone, Debug)]
struct TruthNavSample {
    ttag_us: i64,
    pitch_car_deg: f64,
}

#[derive(Clone, Debug)]
struct Dataset {
    gyro: Vec<GyroSample>,
    accel: Vec<AccelSample>,
    gnss: Vec<GnssSample>,
    truth_nav: Vec<TruthNavSample>,
    truth_misalignment_deg: [f64; 3],
}

#[derive(Clone, Debug)]
struct StressCase {
    label: String,
    seed_mount_error_deg: [f64; 3],
    gps_vd_bias_mps: f64,
    gps_vd_bias_duration_s: f64,
    vel_std_scale: f64,
    accel_z_bias_mps2: f64,
}

#[derive(Clone, Debug)]
struct CaseResult {
    label: String,
    seed_mount_error_deg: [f64; 3],
    gps_vd_bias_mps: f64,
    gps_vd_bias_duration_s: f64,
    vel_std_scale: f64,
    accel_z_bias_mps2: f64,
    first_jump_time_s: f64,
    first_jump_mag_deg: f64,
    has_jump: i32,
    final_mount_roll_deg: f64,
    final_mount_pitch_deg: f64,
    final_mount_yaw_deg: f64,
    final_mount_err_roll_deg: f64,
    final_mount_err_pitch_deg: f64,
    final_mount_err_yaw_deg: f64,
    final_mount_err_norm_deg: f64,
    final_car_pitch_deg: f64,
    final_car_pitch_err_deg: f64,
    final_accel_bias_z: f64,
    wrong_basin: i32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dataset = Arc::new(load_dataset(&args.input_dir)?);
    let cases = monte_carlo_cases(&args);
    if cases.is_empty() {
        bail!("--monte-carlo-runs must be > 0");
    }

    let start = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs.max(1))
        .build()
        .context("failed to build rayon thread pool")?;
    let results = pool.install(|| {
        cases
            .par_iter()
            .map(|case| run_case(&dataset, case, &args))
            .collect::<Result<Vec<_>>>()
    })?;
    let elapsed = start.elapsed();

    if let Some(path) = &args.summary_csv {
        write_summary_csv(path, &results)?;
        println!("summary_csv={}", path.display());
    }

    print_summary(&results, elapsed);
    Ok(())
}

fn monte_carlo_cases(args: &Args) -> Vec<StressCase> {
    let mut rng = StdRng::seed_from_u64(args.monte_carlo_seed);
    (0..args.monte_carlo_runs)
        .map(|idx| StressCase {
            label: format!("mc_{idx:04}"),
            seed_mount_error_deg: [
                sample_uniform(&mut rng, args.mc_seed_roll_min_deg, args.mc_seed_roll_max_deg),
                sample_uniform(
                    &mut rng,
                    args.mc_seed_pitch_min_deg,
                    args.mc_seed_pitch_max_deg,
                ),
                sample_uniform(&mut rng, args.mc_seed_yaw_min_deg, args.mc_seed_yaw_max_deg),
            ],
            gps_vd_bias_mps: sample_uniform(
                &mut rng,
                args.mc_gps_vd_bias_min_mps,
                args.mc_gps_vd_bias_max_mps,
            ),
            gps_vd_bias_duration_s: sample_uniform(
                &mut rng,
                args.mc_gps_vd_bias_duration_min_s,
                args.mc_gps_vd_bias_duration_max_s,
            ),
            vel_std_scale: sample_uniform(
                &mut rng,
                args.mc_vel_std_scale_min,
                args.mc_vel_std_scale_max,
            ),
            accel_z_bias_mps2: sample_uniform(
                &mut rng,
                args.mc_accel_z_bias_min_mps2,
                args.mc_accel_z_bias_max_mps2,
            ),
        })
        .collect()
}

fn sample_uniform(rng: &mut StdRng, a: f64, b: f64) -> f64 {
    let lo = a.min(b);
    let hi = a.max(b);
    rng.random_range(lo..=hi)
}

fn load_dataset(input_dir: &Path) -> Result<Dataset> {
    let mut gnss = import_gnss_data(&input_dir.join("dataset_GNSS.csv"))?;
    let vel_map = import_gnss_velocity_map(&input_dir.join("gnss_velocity_meas.csv"))?;
    for sample in &mut gnss {
        sample.velocity = vel_map
            .iter()
            .find(|(ttag_us, _)| *ttag_us == sample.ttag_us)
            .map(|(_, vel)| vel.clone());
    }
    Ok(Dataset {
        gyro: import_gyro_data(&input_dir.join("dataset_Gyro.csv"))?,
        accel: import_accel_data(&input_dir.join("dataset_Acc.csv"))?,
        gnss,
        truth_nav: import_truth_nav(&input_dir.join("truth_nav.csv"))?,
        truth_misalignment_deg: import_truth_misalignment(&input_dir.join("truth_states.csv"))?,
    })
}

fn run_case(dataset: &Dataset, case: &StressCase, args: &Args) -> Result<CaseResult> {
    let q_nominal_mount = nominal_mount_quat();
    let q_mis = euler_to_quat_deg(dataset.truth_misalignment_deg);
    let q_cs_true = quat_normalize(quat_mul(q_mis, q_nominal_mount));
    let q_seed_err = euler_to_quat_deg(case.seed_mount_error_deg);
    let q_cs_seed = quat_normalize(quat_mul(q_seed_err, q_cs_true));
    let c_seed = quat_to_rotmat(q_cs_seed);

    let init_gyro_index = nearest_time_index_gyro(&dataset.gyro, args.init_time_s)
        .context("no gyro samples available")?;
    let init_ttag = dataset.gyro[init_gyro_index].ttag_us;
    let gnss_init_index = dataset
        .gnss
        .partition_point(|sample| sample.ttag_us <= init_ttag)
        .saturating_sub(1);
    let gnss_init = dataset
        .gnss
        .get(gnss_init_index)
        .context("no GNSS sample at init")?;

    let yaw_init = initial_yaw_from_gnss(gnss_init);
    let q_ne = quat_ecef_to_ned(gnss_init.lat_deg, gnss_init.lon_deg);
    let q_es = quat_normalize(quat_mul(quat_conj(q_ne), yaw_quat(yaw_init)));
    let pos_ecef = lla_to_ecef(gnss_init.lat_deg, gnss_init.lon_deg, gnss_init.height_m);
    let c_en = ecef_to_ned_matrix(gnss_init.lat_deg, gnss_init.lon_deg);
    let vel_n = gnss_init
        .velocity
        .as_ref()
        .map(|v| v.vel_n)
        .unwrap_or([0.0, 0.0, 0.0]);
    let vel_ecef = mat_vec(transpose3(c_en), vel_n);
    let p_diag = build_default_p_diag(gnss_init);

    let mut loose = CLooseWrapper::new(LoosePredictNoise::reference_nsr_demo());
    loose.init_from_reference_ecef_state(
        q_es.map(|v| v as f32),
        pos_ecef,
        vel_ecef.map(|v| v as f32),
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        Some(p_diag),
    );

    let mut gnss_index = gnss_init_index;
    let mut last_used_gnss_ttag = gnss_init.ttag_us;
    let imu_timeout_us: i64 = 500_000;
    let mut prev_mount: Option<[f64; 3]> = None;
    let mut first_jump_time_s = f64::NAN;
    let mut first_jump_mag_deg = 0.0;
    let mut final_mount_deg = [0.0; 3];
    let mut final_mount_err_deg = [0.0; 3];
    let mut final_car_pitch_deg = 0.0;
    let mut final_car_pitch_err_deg = 0.0;
    let mut final_accel_bias_z = 0.0;

    for gyro_index in (init_gyro_index + 1)..dataset.gyro.len() {
        let prev_gyro = &dataset.gyro[gyro_index - 1];
        let curr_gyro = &dataset.gyro[gyro_index];
        let dt_s = 1.0e-6 * (curr_gyro.ttag_us - prev_gyro.ttag_us) as f64;
        if !(0.0..=0.05).contains(&dt_s) || dt_s == 0.0 {
            continue;
        }
        let t_rel_s = 1.0e-6 * (curr_gyro.ttag_us - init_ttag) as f64;
        if args.replay_duration_s > 0.0 && t_rel_s > args.replay_duration_s {
            break;
        }

        let accel_prev_raw = dataset.accel[gyro_index - 1].accel_mps2;
        let accel_curr_raw = dataset.accel[gyro_index].accel_mps2;
        let gyro_prev_raw = prev_gyro.omega_radps;
        let gyro_curr_raw = curr_gyro.omega_radps;
        let mut accel_prev = mat_vec(c_seed, accel_prev_raw);
        let mut accel_curr = mat_vec(c_seed, accel_curr_raw);
        accel_prev[2] += case.accel_z_bias_mps2;
        accel_curr[2] += case.accel_z_bias_mps2;
        let gyro_prev = mat_vec(c_seed, gyro_prev_raw);
        let gyro_curr = mat_vec(c_seed, gyro_curr_raw);

        let imu = CLooseImuDelta {
            dax_1: (gyro_prev[0] * dt_s) as f32,
            day_1: (gyro_prev[1] * dt_s) as f32,
            daz_1: (gyro_prev[2] * dt_s) as f32,
            dvx_1: (accel_prev[0] * dt_s) as f32,
            dvy_1: (accel_prev[1] * dt_s) as f32,
            dvz_1: (accel_prev[2] * dt_s) as f32,
            dax_2: (gyro_curr[0] * dt_s) as f32,
            day_2: (gyro_curr[1] * dt_s) as f32,
            daz_2: (gyro_curr[2] * dt_s) as f32,
            dvx_2: (accel_curr[0] * dt_s) as f32,
            dvy_2: (accel_curr[1] * dt_s) as f32,
            dvz_2: (accel_curr[2] * dt_s) as f32,
            dt: dt_s as f32,
        };
        loose.predict(imu);

        while gnss_index + 1 < dataset.gnss.len() && dataset.gnss[gnss_index + 1].ttag_us <= curr_gyro.ttag_us {
            gnss_index += 1;
        }

        let mut pos_ecef_opt = None;
        let mut vel_ecef_opt = None;
        let mut vel_std_opt = None;
        let mut h_acc_m = 0.0_f32;
        let mut dt_since_last_gnss_s = 1.0_f32;
        if let Some(gnss) = dataset.gnss.get(gnss_index) {
            let d_ttag = curr_gyro.ttag_us - gnss.ttag_us;
            if d_ttag < imu_timeout_us / 2 && gnss.ttag_us != last_used_gnss_ttag {
                let pos_ecef = lla_to_ecef(gnss.lat_deg, gnss.lon_deg, gnss.height_m);
                pos_ecef_opt = Some(pos_ecef);
                if let Some(vel) = &gnss.velocity {
                    let mut vel_n = vel.vel_n;
                    let rel_gnss_s = 1.0e-6 * (gnss.ttag_us - init_ttag) as f64;
                    if (0.0..=case.gps_vd_bias_duration_s).contains(&rel_gnss_s) {
                        vel_n[2] += case.gps_vd_bias_mps;
                    }
                    let c_en_meas = ecef_to_ned_matrix(gnss.lat_deg, gnss.lon_deg);
                    vel_ecef_opt = Some(mat_vec(transpose3(c_en_meas), vel_n).map(|v| v as f32));
                    vel_std_opt = Some([
                        (vel.vel_acc_n[0] * case.vel_std_scale) as f32,
                        (vel.vel_acc_n[1] * case.vel_std_scale) as f32,
                        (vel.vel_acc_n[2] * case.vel_std_scale) as f32,
                    ]);
                }
                h_acc_m = gnss.h_acc_m as f32;
                let dt = 1.0e-6 * (curr_gyro.ttag_us - last_used_gnss_ttag) as f64;
                dt_since_last_gnss_s = if dt == 0.0 || dt >= 1.0 { 1.0 } else { dt as f32 };
                last_used_gnss_ttag = gnss.ttag_us;
            }
        }

        loose.fuse_reference_batch_full(
            pos_ecef_opt,
            vel_ecef_opt,
            h_acc_m,
            vel_std_opt,
            dt_since_last_gnss_s,
            gyro_curr.map(|v| v as f32),
            accel_curr.map(|v| v as f32),
            dt_s as f32,
        );

        let n = loose.nominal();
        let q_es_hat = [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64];
        let q_cs_resid = [n.qcs0 as f64, n.qcs1 as f64, n.qcs2 as f64, n.qcs3 as f64];
        let q_cs_full = quat_normalize(quat_mul(q_cs_resid, q_cs_seed));
        let q_mount_err = quat_normalize(quat_mul(q_cs_full, quat_conj(q_cs_true)));
        let q_mount_full = quat_normalize(quat_mul(q_cs_full, quat_conj(q_nominal_mount)));
        let mount_err_deg = quat_to_euler_deg(q_mount_err);
        let mount_full_deg = quat_to_euler_deg(q_mount_full);

        let pos_hat = loose.shadow_pos_ecef();
        let (lat_hat, lon_hat, _) = ecef_to_lla(pos_hat);
        let q_ne_hat = quat_ecef_to_ned(lat_hat, lon_hat);
        let q_ns_hat = quat_normalize(quat_mul(q_ne_hat, q_es_hat));
        let q_nc_hat = quat_normalize(quat_mul(q_ns_hat, quat_conj(q_cs_resid)));
        let car_euler_deg = quat_to_euler_deg(q_nc_hat);
        let truth_car_pitch = interp_truth_pitch(&dataset.truth_nav, curr_gyro.ttag_us);

        if let Some(prev) = prev_mount {
            let diff = [
                wrap_deg180(mount_full_deg[0] - prev[0]),
                wrap_deg180(mount_full_deg[1] - prev[1]),
                wrap_deg180(mount_full_deg[2] - prev[2]),
            ];
            let step = norm3(diff);
            if first_jump_time_s.is_nan() && step >= 5.0 {
                first_jump_time_s = t_rel_s;
                first_jump_mag_deg = step;
            }
        }
        prev_mount = Some(mount_full_deg);
        final_mount_deg = mount_full_deg;
        final_mount_err_deg = mount_err_deg;
        final_car_pitch_deg = car_euler_deg[1];
        final_car_pitch_err_deg = car_euler_deg[1] - truth_car_pitch;
        final_accel_bias_z = n.baz as f64;
    }

    let final_mount_err_norm_deg = norm3([
        wrap_deg180(final_mount_err_deg[0]),
        wrap_deg180(final_mount_err_deg[1]),
        wrap_deg180(final_mount_err_deg[2]),
    ]);
    let wrong_basin = (final_mount_err_norm_deg >= args.mc_mount_err_threshold_deg
        || final_car_pitch_err_deg.abs() >= args.mc_car_pitch_err_threshold_deg)
        as i32;

    Ok(CaseResult {
        label: case.label.clone(),
        seed_mount_error_deg: case.seed_mount_error_deg,
        gps_vd_bias_mps: case.gps_vd_bias_mps,
        gps_vd_bias_duration_s: case.gps_vd_bias_duration_s,
        vel_std_scale: case.vel_std_scale,
        accel_z_bias_mps2: case.accel_z_bias_mps2,
        first_jump_time_s,
        first_jump_mag_deg,
        has_jump: (!first_jump_time_s.is_nan()) as i32,
        final_mount_roll_deg: final_mount_deg[0],
        final_mount_pitch_deg: final_mount_deg[1],
        final_mount_yaw_deg: final_mount_deg[2],
        final_mount_err_roll_deg: final_mount_err_deg[0],
        final_mount_err_pitch_deg: final_mount_err_deg[1],
        final_mount_err_yaw_deg: final_mount_err_deg[2],
        final_mount_err_norm_deg,
        final_car_pitch_deg,
        final_car_pitch_err_deg,
        final_accel_bias_z,
        wrong_basin,
    })
}

fn nearest_time_index_gyro(gyro: &[GyroSample], init_time_s: f64) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (idx, sample) in gyro.iter().enumerate() {
        let err = ((sample.ttag_us as f64) * 1.0e-6 - init_time_s).abs();
        if best.is_none_or(|(_, best_err)| err < best_err) {
            best = Some((idx, err));
        }
    }
    best.map(|(idx, _)| idx)
}

fn initial_yaw_from_gnss(gnss: &GnssSample) -> f64 {
    if let Some(vel) = &gnss.velocity {
        let speed_h = vel.vel_n[0].hypot(vel.vel_n[1]);
        if speed_h >= 1.0 {
            return vel.vel_n[1].atan2(vel.vel_n[0]);
        }
    }
    if gnss.speed_mps > 0.0 {
        return gnss.heading_deg.to_radians();
    }
    0.0
}

fn build_default_p_diag(gnss: &GnssSample) -> [f32; 24] {
    let att_sigma_rad = 2.0_f64.to_radians();
    let att_var = (att_sigma_rad * att_sigma_rad) as f32;
    let mut vel_std = 0.2_f64;
    if let Some(vel) = &gnss.velocity {
        vel_std = vel_std.max(vel.vel_acc_n[0].max(vel.vel_acc_n[1]).max(vel.vel_acc_n[2]));
    }
    let vel_var = (vel_std * vel_std) as f32;
    let pos_n = gnss.h_acc_m.max(0.5) as f32;
    let pos_e = gnss.h_acc_m.max(0.5) as f32;
    let pos_d = gnss.v_acc_m.max(0.5) as f32;
    let gyro_bias_sigma = 0.125_f64.to_radians() as f32;
    let accel_bias_sigma = 0.075_f32;
    let accel_scale_sigma = 0.02_f32;
    let gyro_scale_sigma = 0.02_f32;
    let mut p = [0.0_f32; 24];
    p[0] = pos_n * pos_n;
    p[1] = pos_e * pos_e;
    p[2] = pos_d * pos_d;
    p[3] = vel_var;
    p[4] = vel_var;
    p[5] = vel_var;
    p[6] = att_var;
    p[7] = att_var;
    p[8] = att_var;
    p[9] = accel_bias_sigma * accel_bias_sigma;
    p[10] = accel_bias_sigma * accel_bias_sigma;
    p[11] = accel_bias_sigma * accel_bias_sigma;
    p[12] = gyro_bias_sigma * gyro_bias_sigma;
    p[13] = gyro_bias_sigma * gyro_bias_sigma;
    p[14] = gyro_bias_sigma * gyro_bias_sigma;
    p[15] = accel_scale_sigma * accel_scale_sigma;
    p[16] = accel_scale_sigma * accel_scale_sigma;
    p[17] = accel_scale_sigma * accel_scale_sigma;
    p[18] = gyro_scale_sigma * gyro_scale_sigma;
    p[19] = gyro_scale_sigma * gyro_scale_sigma;
    p[20] = gyro_scale_sigma * gyro_scale_sigma;
    p[21] = att_var;
    p[22] = att_var;
    p[23] = att_var;
    p
}

fn nominal_mount_quat() -> [f64; 4] {
    euler_to_quat_rad([
        -0.5 * std::f64::consts::PI,
        0.0,
        0.5 * std::f64::consts::PI,
    ])
}

fn yaw_quat(yaw_rad: f64) -> [f64; 4] {
    let half = 0.5 * yaw_rad;
    [half.cos(), 0.0, 0.0, half.sin()]
}

fn euler_to_quat_deg(rpy_deg: [f64; 3]) -> [f64; 4] {
    euler_to_quat_rad([
        rpy_deg[0].to_radians(),
        rpy_deg[1].to_radians(),
        rpy_deg[2].to_radians(),
    ])
}

fn euler_to_quat_rad(rpy_rad: [f64; 3]) -> [f64; 4] {
    let (roll, pitch, yaw) = (rpy_rad[0], rpy_rad[1], rpy_rad[2]);
    let (sr, cr) = (0.5 * roll).sin_cos();
    let (sp, cp) = (0.5 * pitch).sin_cos();
    let (sy, cy) = (0.5 * yaw).sin_cos();
    quat_normalize([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
}

fn quat_to_euler_deg(q: [f64; 4]) -> [f64; 3] {
    let q = quat_normalize(q);
    let sinr_cosp = 2.0 * (q[0] * q[1] + q[2] * q[3]);
    let cosr_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
    let roll = sinr_cosp.atan2(cosr_cosp).to_degrees();
    let sinp = 2.0 * (q[0] * q[2] - q[3] * q[1]);
    let pitch = sinp.clamp(-1.0, 1.0).asin().to_degrees();
    let siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2]);
    let cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]);
    let yaw = siny_cosp.atan2(cosy_cosp).to_degrees();
    [roll, pitch, yaw]
}

fn quat_to_rotmat(q: [f64; 4]) -> [[f64; 3]; 3] {
    let q = quat_normalize(q);
    let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);
    [
        [
            1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3,
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3,
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2,
        ],
    ]
}

fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_normalize(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 0.0 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

fn quat_ecef_to_ned(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    let lon = lon_deg.to_radians();
    let lat = lat_deg.to_radians();
    let half_lon = 0.5 * lon;
    let q1 = [half_lon.cos(), 0.0, 0.0, -half_lon.sin()];
    let half_lat = 0.5 * (lat + 0.5 * std::f64::consts::PI);
    let q2 = [half_lat.cos(), 0.0, half_lat.sin(), 0.0];
    quat_mul(q2, q1)
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

fn transpose3(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn mat_vec(a: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn wrap_deg180(v: f64) -> f64 {
    let mut out = (v + 180.0) % 360.0;
    if out < 0.0 {
        out += 360.0;
    }
    out - 180.0
}

fn interp_truth_pitch(truth_nav: &[TruthNavSample], ttag_us: i64) -> f64 {
    if truth_nav.is_empty() {
        return 0.0;
    }
    match truth_nav.binary_search_by(|s| s.ttag_us.cmp(&ttag_us)) {
        Ok(index) => truth_nav[index].pitch_car_deg,
        Err(0) => truth_nav[0].pitch_car_deg,
        Err(index) if index >= truth_nav.len() => truth_nav[truth_nav.len() - 1].pitch_car_deg,
        Err(index) => {
            let a = &truth_nav[index - 1];
            let b = &truth_nav[index];
            let span = (b.ttag_us - a.ttag_us) as f64;
            if span <= 0.0 {
                return a.pitch_car_deg;
            }
            let alpha = (ttag_us - a.ttag_us) as f64 / span;
            a.pitch_car_deg + alpha * (b.pitch_car_deg - a.pitch_car_deg)
        }
    }
}

fn write_summary_csv(path: &Path, results: &[CaseResult]) -> Result<()> {
    let mut out = String::from(
        "label,seed_roll_deg,seed_pitch_deg,seed_yaw_deg,gps_vd_bias_mps,gps_vd_bias_duration_s,vel_std_scale,accel_z_bias_mps2,first_jump_time_s,first_jump_mag_deg,has_jump,final_mount_roll_deg,final_mount_pitch_deg,final_mount_yaw_deg,final_mount_err_roll_deg,final_mount_err_pitch_deg,final_mount_err_yaw_deg,final_mount_err_norm_deg,final_car_pitch_deg,final_car_pitch_err_deg,final_accel_bias_z,wrong_basin\n",
    );
    for r in results {
        out.push_str(&format!(
            "{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{}\n",
            r.label,
            r.seed_mount_error_deg[0],
            r.seed_mount_error_deg[1],
            r.seed_mount_error_deg[2],
            r.gps_vd_bias_mps,
            r.gps_vd_bias_duration_s,
            r.vel_std_scale,
            r.accel_z_bias_mps2,
            r.first_jump_time_s,
            r.first_jump_mag_deg,
            r.has_jump,
            r.final_mount_roll_deg,
            r.final_mount_pitch_deg,
            r.final_mount_yaw_deg,
            r.final_mount_err_roll_deg,
            r.final_mount_err_pitch_deg,
            r.final_mount_err_yaw_deg,
            r.final_mount_err_norm_deg,
            r.final_car_pitch_deg,
            r.final_car_pitch_err_deg,
            r.final_accel_bias_z,
            r.wrong_basin
        ));
    }
    fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))
}

fn print_summary(results: &[CaseResult], elapsed: std::time::Duration) {
    let runs = results.len().max(1);
    let jump_rate =
        100.0 * results.iter().map(|r| r.has_jump as f64).sum::<f64>() / runs as f64;
    let wrong_rate =
        100.0 * results.iter().map(|r| r.wrong_basin as f64).sum::<f64>() / runs as f64;
    let mut mount_err: Vec<f64> = results.iter().map(|r| r.final_mount_err_norm_deg).collect();
    let mut pitch_err: Vec<f64> = results
        .iter()
        .map(|r| r.final_car_pitch_err_deg.abs())
        .collect();
    mount_err.sort_by(|a, b| a.total_cmp(b));
    pitch_err.sort_by(|a, b| a.total_cmp(b));
    let p50_mount = percentile_sorted(&mount_err, 0.50);
    let p95_mount = percentile_sorted(&mount_err, 0.95);
    let p50_pitch = percentile_sorted(&pitch_err, 0.50);
    let p95_pitch = percentile_sorted(&pitch_err, 0.95);
    println!(
        "mc_summary: runs={} jump_rate={:.1}% wrong_basin_rate={:.1}% mount_err_p50={:.2}deg mount_err_p95={:.2}deg car_pitch_err_p50={:.2}deg car_pitch_err_p95={:.2}deg elapsed_s={:.2} cases_per_s={:.2}",
        runs,
        jump_rate,
        wrong_rate,
        p50_mount,
        p95_mount,
        p50_pitch,
        p95_pitch,
        elapsed.as_secs_f64(),
        runs as f64 / elapsed.as_secs_f64().max(1.0e-9)
    );
    let mut worst = results.to_vec();
    worst.sort_by(|a, b| b.final_mount_err_norm_deg.total_cmp(&a.final_mount_err_norm_deg));
    for row in worst.into_iter().take(5) {
        println!(
            "mc_worst: {} mount_err_norm={:.2}deg car_pitch_err={:.2}deg seed=[{:.2},{:.2},{:.2}]deg gps_vd_bias={:.2}mps gps_vd_bias_duration={:.2}s vel_std_scale={:.2} accel_z_bias={:.3}mps2",
            row.label,
            row.final_mount_err_norm_deg,
            row.final_car_pitch_err_deg,
            row.seed_mount_error_deg[0],
            row.seed_mount_error_deg[1],
            row.seed_mount_error_deg[2],
            row.gps_vd_bias_mps,
            row.gps_vd_bias_duration_s,
            row.vel_std_scale,
            row.accel_z_bias_mps2
        );
    }
}

fn percentile_sorted(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let pos = ((values.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    values[pos]
}

fn import_gyro_data(path: &Path) -> Result<Vec<GyroSample>> {
    let rows = semicolon_rows(path, 3)?;
    rows.into_iter()
        .map(|row| {
            Ok(GyroSample {
                ttag_us: (parse_f64(&row[0])? / 1000.0).floor() as i64,
                omega_radps: [parse_f64(&row[1])?, parse_f64(&row[2])?, parse_f64(&row[3])?],
            })
        })
        .collect()
}

fn import_accel_data(path: &Path) -> Result<Vec<AccelSample>> {
    let rows = semicolon_rows(path, 3)?;
    rows.into_iter()
        .map(|row| {
            Ok(AccelSample {
                accel_mps2: [parse_f64(&row[1])?, parse_f64(&row[2])?, parse_f64(&row[3])?],
            })
        })
        .collect()
}

fn import_gnss_data(path: &Path) -> Result<Vec<GnssSample>> {
    let rows = semicolon_rows(path, 1)?;
    rows.into_iter()
        .map(|row| {
            Ok(GnssSample {
                ttag_us: (parse_f64(&row[0])? / 1000.0).floor() as i64,
                lat_deg: parse_f64(&row[2])?,
                lon_deg: parse_f64(&row[3])?,
                height_m: parse_f64(&row[4])?,
                speed_mps: parse_f64(&row[5])?,
                heading_deg: parse_f64(&row[6])?,
                h_acc_m: parse_f64(&row[7])?,
                v_acc_m: parse_f64(&row[8])?,
                velocity: None,
            })
        })
        .collect()
}

fn import_gnss_velocity_map(path: &Path) -> Result<Vec<(i64, GnssVelocity)>> {
    let rows = semicolon_rows(path, 1)?;
    rows.into_iter()
        .map(|row| {
            Ok((
                parse_f64(&row[0])?.round() as i64,
                GnssVelocity {
                    vel_n: [parse_f64(&row[1])?, parse_f64(&row[2])?, parse_f64(&row[3])?],
                    vel_acc_n: [parse_f64(&row[4])?, parse_f64(&row[5])?, parse_f64(&row[6])?],
                },
            ))
        })
        .collect()
}

fn import_truth_nav(path: &Path) -> Result<Vec<TruthNavSample>> {
    let rows = semicolon_rows(path, 1)?;
    rows.into_iter()
        .map(|row| {
            Ok(TruthNavSample {
                ttag_us: parse_f64(&row[0])?.round() as i64,
                pitch_car_deg: parse_f64(&row[11])?,
            })
        })
        .collect()
}

fn import_truth_misalignment(path: &Path) -> Result<[f64; 3]> {
    for row in semicolon_rows(path, 1)? {
        if row.first().is_some_and(|name| name == "misalignment_deg") {
            return Ok([parse_f64(&row[1])?, parse_f64(&row[2])?, parse_f64(&row[3])?]);
        }
    }
    bail!("misalignment_deg row not found in {}", path.display())
}

fn semicolon_rows(path: &Path, skip_rows: usize) -> Result<Vec<Vec<String>>> {
    let text = fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index < skip_rows {
            continue;
        }
        let row: Vec<String> = line
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect();
        if !row.is_empty() {
            out.push(row);
        }
    }
    Ok(out)
}

fn parse_f64(s: &str) -> Result<f64> {
    s.parse::<f64>()
        .with_context(|| format!("failed to parse float: {s}"))
}
