use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use sensor_fusion::loose::{LOOSE_ERROR_STATES, LooseFilter, LooseImuDelta, LoosePredictNoise};
use sim::eval::gnss_ins::{quat_conj, quat_mul};
use sim::synthetic::gnss_ins_path::{MotionProfile, PathGenConfig, generate};
use sim::visualizer::math::lla_to_ecef;

#[derive(Parser, Debug)]
#[command(name = "diag_loose_accel_z_bias")]
struct Args {
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "sim/motion_profiles/city_blocks_15min.scenario,sim/motion_profiles/figure8_15min.csv,sim/motion_profiles/straight_accel_brake_15min.csv,sim/motion_profiles/sloped_start_recovery_15min.csv"
    )]
    scenarios: Vec<PathBuf>,
    #[arg(long, value_delimiter = ',', default_value = "0.1")]
    accel_z_bias_mps2: Vec<f64>,
    #[arg(long, default_value_t = 100.0)]
    imu_hz: f64,
    #[arg(long, default_value_t = 2.0)]
    gnss_hz: f64,
    #[arg(long, default_value_t = 0.5)]
    accel_bias_sigma_mps2: f32,
    #[arg(long, default_value_t = 0.0)]
    accel_scale_sigma: f32,
    #[arg(long, default_value_t = 0.0)]
    gyro_scale_sigma: f32,
    #[arg(long, default_value_t = 0.0)]
    mount_sigma_deg: f32,
    #[arg(long, default_value_t = 0.0)]
    pos_std_m: f32,
    #[arg(long, default_value_t = 0.0)]
    vel_std_mps: f32,
    #[arg(long, default_value_t = true)]
    use_gnss_position: bool,
    #[arg(long, default_value_t = true)]
    use_gnss_velocity: bool,
}

#[derive(Clone, Copy, Debug)]
struct ResultRow {
    final_t_s: f64,
    final_baz_state_mps2: f64,
    estimated_sensor_bias_mps2: f64,
    final_baz_sensor_error_mps2: f64,
    final_baz_sigma_mps2: f64,
    final_saz: f64,
    final_att_err_deg: f64,
    final_vel_err_mps: f64,
    final_pos_err_m: f64,
    t_within_10pct_s: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!(
        "scenario,bias_injected_mps2,final_t_s,final_baz_state_mps2,estimated_sensor_bias_mps2,final_sensor_bias_error_mps2,final_baz_sigma_mps2,final_saz,final_att_err_deg,final_vel_err_mps,final_pos_err_m,t_within_10pct_s"
    );
    for scenario in &args.scenarios {
        for &bias in &args.accel_z_bias_mps2 {
            let row = run_case(scenario, bias, &args)
                .with_context(|| format!("scenario={} bias={bias}", scenario.display()))?;
            println!(
                "{},{:.9},{:.3},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{}",
                scenario.display(),
                bias,
                row.final_t_s,
                row.final_baz_state_mps2,
                row.estimated_sensor_bias_mps2,
                row.final_baz_sensor_error_mps2,
                row.final_baz_sigma_mps2,
                row.final_saz,
                row.final_att_err_deg,
                row.final_vel_err_mps,
                row.final_pos_err_m,
                format_optional_time(row.t_within_10pct_s),
            );
        }
    }
    Ok(())
}

fn run_case(scenario: &PathBuf, accel_z_bias_mps2: f64, args: &Args) -> Result<ResultRow> {
    let profile = MotionProfile::from_path(scenario)?;
    let generated = generate(
        &profile,
        PathGenConfig {
            imu_hz: args.imu_hz,
            gnss_hz: args.gnss_hz,
            ..PathGenConfig::default()
        },
    )?;
    let first_truth = generated
        .truth
        .first()
        .context("synthetic profile produced no truth samples")?;
    let first_gnss = generated
        .gnss
        .first()
        .context("synthetic profile produced no GNSS samples")?;

    let q_ne0 = quat_ecef_to_ned(first_truth.lat_deg, first_truth.lon_deg);
    let q_es0 = quat_mul(quat_conj(q_ne0), first_truth.q_bn);
    let pos0 = lla_to_ecef(
        first_truth.lat_deg,
        first_truth.lon_deg,
        first_truth.height_m,
    );
    let vel0 = ned_to_ecef(
        first_truth.lat_deg,
        first_truth.lon_deg,
        first_truth.vel_ned_mps,
    );
    let mut loose = LooseFilter::new(LoosePredictNoise::lsm6dso_loose_104hz());
    loose.init_from_reference_ecef_state(
        q_es0.map(|v| v as f32),
        pos0,
        vel0.map(|v| v as f32),
        [0.0; 3],
        [0.0; 3],
        [1.0; 3],
        [1.0; 3],
        [1.0, 0.0, 0.0, 0.0],
        Some(build_p_diag(args)),
    );

    let mut gnss_index = 0usize;
    let mut last_used_gnss_index = 0usize;
    let mut final_row = ResultRow {
        final_t_s: first_truth.t_s,
        final_baz_state_mps2: 0.0,
        estimated_sensor_bias_mps2: 0.0,
        final_baz_sensor_error_mps2: -accel_z_bias_mps2,
        final_baz_sigma_mps2: args.accel_bias_sigma_mps2 as f64,
        final_saz: 1.0,
        final_att_err_deg: 0.0,
        final_vel_err_mps: 0.0,
        final_pos_err_m: 0.0,
        t_within_10pct_s: f64::NAN,
    };

    for imu_index in 1..generated.imu.len() {
        let prev = generated.imu[imu_index - 1];
        let curr = generated.imu[imu_index];
        let dt_s = curr.t_s - prev.t_s;
        if !(0.0..=0.05).contains(&dt_s) {
            continue;
        }

        let accel_prev = [
            prev.accel_vehicle_mps2[0],
            prev.accel_vehicle_mps2[1],
            prev.accel_vehicle_mps2[2] + accel_z_bias_mps2,
        ];
        let accel_curr = [
            curr.accel_vehicle_mps2[0],
            curr.accel_vehicle_mps2[1],
            curr.accel_vehicle_mps2[2] + accel_z_bias_mps2,
        ];
        loose.predict(LooseImuDelta {
            dax_1: (prev.gyro_vehicle_radps[0] * dt_s) as f32,
            day_1: (prev.gyro_vehicle_radps[1] * dt_s) as f32,
            daz_1: (prev.gyro_vehicle_radps[2] * dt_s) as f32,
            dvx_1: (accel_prev[0] * dt_s) as f32,
            dvy_1: (accel_prev[1] * dt_s) as f32,
            dvz_1: (accel_prev[2] * dt_s) as f32,
            dax_2: (curr.gyro_vehicle_radps[0] * dt_s) as f32,
            day_2: (curr.gyro_vehicle_radps[1] * dt_s) as f32,
            daz_2: (curr.gyro_vehicle_radps[2] * dt_s) as f32,
            dvx_2: (accel_curr[0] * dt_s) as f32,
            dvy_2: (accel_curr[1] * dt_s) as f32,
            dvz_2: (accel_curr[2] * dt_s) as f32,
            dt: dt_s as f32,
        });

        while gnss_index + 1 < generated.gnss.len()
            && generated.gnss[gnss_index + 1].t_s <= curr.t_s + 1.0e-9
        {
            gnss_index += 1;
        }
        if gnss_index != last_used_gnss_index {
            let gnss = generated.gnss[gnss_index];
            let pos_ecef = args
                .use_gnss_position
                .then(|| lla_to_ecef(gnss.lat_deg, gnss.lon_deg, gnss.height_m));
            let vel_ecef = args.use_gnss_velocity.then(|| {
                ned_to_ecef(gnss.lat_deg, gnss.lon_deg, gnss.vel_ned_mps).map(|v| v as f32)
            });
            loose.fuse_reference_batch_full(
                pos_ecef,
                vel_ecef,
                args.pos_std_m.max(1.0e-3),
                args.use_gnss_velocity
                    .then_some([args.vel_std_mps.max(1.0e-3); 3]),
                (gnss.t_s - first_gnss.t_s).min(1.0).max(1.0e-3) as f32,
                curr.gyro_vehicle_radps.map(|v| v as f32),
                accel_curr.map(|v| v as f32),
                dt_s as f32,
            );
            last_used_gnss_index = gnss_index;
        }

        if let Some(truth) = generated.truth.get(imu_index) {
            let n = loose.nominal();
            let baz = n.baz as f64;
            let estimated_sensor_bias = -baz;
            let sensor_bias_error = estimated_sensor_bias - accel_z_bias_mps2;
            if final_row.t_within_10pct_s.is_nan()
                && sensor_bias_error.abs() <= 0.1 * accel_z_bias_mps2.abs().max(1.0e-9)
            {
                final_row.t_within_10pct_s = curr.t_s;
            }
            let pos = loose.shadow_pos_ecef();
            let pos_truth = lla_to_ecef(truth.lat_deg, truth.lon_deg, truth.height_m);
            let vel_truth = ned_to_ecef(truth.lat_deg, truth.lon_deg, truth.vel_ned_mps);
            let q_ne = quat_ecef_to_ned(truth.lat_deg, truth.lon_deg);
            let q_ns = quat_mul(q_ne, [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64]);
            final_row = ResultRow {
                final_t_s: curr.t_s,
                final_baz_state_mps2: baz,
                estimated_sensor_bias_mps2: estimated_sensor_bias,
                final_baz_sensor_error_mps2: sensor_bias_error,
                final_baz_sigma_mps2: loose.covariance()[11][11].max(0.0).sqrt() as f64,
                final_saz: n.saz as f64,
                final_att_err_deg: quat_angle_deg(q_ns, truth.q_bn),
                final_vel_err_mps: norm3([
                    n.vn as f64 - vel_truth[0],
                    n.ve as f64 - vel_truth[1],
                    n.vd as f64 - vel_truth[2],
                ]),
                final_pos_err_m: norm3([
                    pos[0] - pos_truth[0],
                    pos[1] - pos_truth[1],
                    pos[2] - pos_truth[2],
                ]),
                t_within_10pct_s: final_row.t_within_10pct_s,
            };
        }
    }

    Ok(final_row)
}

fn build_p_diag(args: &Args) -> [f32; LOOSE_ERROR_STATES] {
    let mut p = [0.0_f32; LOOSE_ERROR_STATES];
    let pos_var = args.pos_std_m.max(0.001).powi(2);
    let vel_var = args.vel_std_mps.max(0.001).powi(2);
    let att_var = 1.0e-8_f32;
    p[0] = pos_var;
    p[1] = pos_var;
    p[2] = pos_var;
    p[3] = vel_var;
    p[4] = vel_var;
    p[5] = vel_var;
    p[6] = att_var;
    p[7] = att_var;
    p[8] = att_var;
    p[9] = args.accel_bias_sigma_mps2.powi(2);
    p[10] = args.accel_bias_sigma_mps2.powi(2);
    p[11] = args.accel_bias_sigma_mps2.powi(2);
    p[12] = 1.0e-12;
    p[13] = 1.0e-12;
    p[14] = 1.0e-12;
    p[15] = args.accel_scale_sigma.powi(2);
    p[16] = args.accel_scale_sigma.powi(2);
    p[17] = args.accel_scale_sigma.powi(2);
    p[18] = args.gyro_scale_sigma.powi(2);
    p[19] = args.gyro_scale_sigma.powi(2);
    p[20] = args.gyro_scale_sigma.powi(2);
    let mount_var = args.mount_sigma_deg.to_radians().powi(2);
    p[21] = mount_var;
    p[22] = mount_var;
    p[23] = mount_var;
    p
}

fn ned_to_ecef(lat_deg: f64, lon_deg: f64, vel_ned: [f64; 3]) -> [f64; 3] {
    mat_vec(transpose3(ecef_to_ned_matrix(lat_deg, lon_deg)), vel_ned)
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
    let lon = lon_deg.to_radians();
    let lat = lat_deg.to_radians();
    let half_lon = 0.5 * lon;
    let q_lon = [half_lon.cos(), 0.0, 0.0, -half_lon.sin()];
    let half_lat = 0.5 * (lat + 0.5 * std::f64::consts::PI);
    let q_lat = [half_lat.cos(), 0.0, half_lat.sin(), 0.0];
    quat_mul(q_lat, q_lon)
}

fn quat_angle_deg(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
        .abs()
        .clamp(0.0, 1.0);
    2.0 * dot.acos().to_degrees()
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

fn format_optional_time(t_s: f64) -> String {
    if t_s.is_finite() {
        format!("{t_s:.3}")
    } else {
        "nan".to_string()
    }
}
