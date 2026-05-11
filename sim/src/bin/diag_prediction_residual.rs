use anyhow::Result;
use clap::{Parser, ValueEnum};
use sensor_fusion::{ProcessNoise, full, reduced};
use sim::eval::gnss_ins::{as_q64, quat_angle_deg, quat_conj, quat_mul};
use sim::synthetic::gnss_ins_path::{MotionProfile, PathGenConfig, generate};
use sim::visualizer::math::lla_to_ecef;
use sim::visualizer::pipeline::generic::reference_mount_rpy_to_q_bv;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "diag_prediction_residual")]
struct Args {
    #[arg(long, value_name = "SCENARIO")]
    motion_def: PathBuf,
    #[arg(long, value_enum, default_value_t = FilterArg::Both)]
    filter: FilterArg,
    #[arg(long, default_value_t = 100.0)]
    imu_hz: f64,
    #[arg(long, default_value_t = 2.0)]
    gnss_hz: f64,
    #[arg(long, default_value_t = 5.0)]
    mount_roll_deg: f64,
    #[arg(long, default_value_t = -5.0)]
    mount_pitch_deg: f64,
    #[arg(long, default_value_t = 5.0)]
    mount_yaw_deg: f64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FilterArg {
    Both,
    Reduced,
    Full,
}

#[derive(Clone, Copy, Debug)]
enum ReducedInput {
    Start,
    End,
    Trapezoid,
    LaggedTrapezoid,
}

#[derive(Clone, Copy, Debug)]
enum FullInput {
    StartStart,
    StartEnd,
    EndEnd,
    LaggedStartEnd,
}

#[derive(Clone, Copy, Debug)]
struct Summary {
    final_vel_err_ned: [f64; 3],
    rms_vel_err_mps: f64,
    final_att_err_deg: f64,
    eq_accel_drift_mps2: [f64; 3],
}

fn main() -> Result<()> {
    let args = Args::parse();
    let profile = MotionProfile::from_path(&args.motion_def)?;
    let generated = generate(
        &profile,
        PathGenConfig {
            imu_hz: args.imu_hz,
            gnss_hz: args.gnss_hz,
            ..PathGenConfig::default()
        },
    )?;
    let q_mount = reference_mount_rpy_to_q_bv([
        args.mount_roll_deg,
        args.mount_pitch_deg,
        args.mount_yaw_deg,
    ]);

    println!(
        "prediction-only residual scenario={} imu_hz={:.1} samples={} span={:.3}s",
        args.motion_def.display(),
        args.imu_hz,
        generated.imu.len(),
        generated.truth.last().map_or(0.0, |s| s.t_s)
    );
    if matches!(args.filter, FilterArg::Both | FilterArg::Reduced) {
        for mode in [
            ReducedInput::Start,
            ReducedInput::End,
            ReducedInput::Trapezoid,
            ReducedInput::LaggedTrapezoid,
        ] {
            let summary = run_reduced(&generated, q_mount, mode);
            print_summary(&format!("Reduced {mode:?}"), summary);
        }
    }
    if matches!(args.filter, FilterArg::Both | FilterArg::Full) {
        for mode in [
            FullInput::StartStart,
            FullInput::StartEnd,
            FullInput::EndEnd,
            FullInput::LaggedStartEnd,
        ] {
            let summary = run_full(&generated, q_mount, mode);
            print_summary(&format!("Full {mode:?}"), summary);
        }
    }
    Ok(())
}

fn run_reduced(
    generated: &sim::synthetic::gnss_ins_path::GeneratedPath,
    q_mount: [f64; 4],
    mode: ReducedInput,
) -> Summary {
    let first = generated.truth[0];
    let mut reduced = reduced::Filter::new(ProcessNoise::default());
    reduced.set_gravity_mss(normal_gravity_mss(first.lat_deg, first.height_m));
    reduced.init_nominal_from_gnss(
        [
            first.q_bn[0] as f32,
            first.q_bn[1] as f32,
            first.q_bn[2] as f32,
            first.q_bn[3] as f32,
        ],
        reduced::GnssSample {
            t_s: first.t_s as f32,
            pos_ned_m: [0.0; 3],
            vel_ned_mps: [
                first.vel_ned_mps[0] as f32,
                first.vel_ned_mps[1] as f32,
                first.vel_ned_mps[2] as f32,
            ],
            pos_std_m: [0.5; 3],
            vel_std_mps: [0.2; 3],
            heading_rad: None,
        },
    );
    {
        let n = &mut reduced.raw_mut().nominal;
        n.q_bv0 = q_mount[0] as f32;
        n.q_bv1 = q_mount[1] as f32;
        n.q_bv2 = q_mount[2] as f32;
        n.q_bv3 = q_mount[3] as f32;
    }

    let mut sum_vel_err2 = 0.0;
    let mut count = 0usize;
    let mut final_vel_err = [0.0; 3];
    let mut final_att_err = 0.0;
    for i in 0..generated.imu.len() - 1 {
        let dt = generated.imu[i + 1].t_s - generated.imu[i].t_s;
        if dt <= 0.0 {
            continue;
        }
        let (gyro_raw, accel_raw) = match mode {
            ReducedInput::Start => (
                mounted_gyro(generated.imu[i].gyro_vehicle_radps, q_mount),
                mounted_accel(generated.imu[i].accel_vehicle_mps2, q_mount),
            ),
            ReducedInput::End => (
                mounted_gyro(generated.imu[i + 1].gyro_vehicle_radps, q_mount),
                mounted_accel(generated.imu[i + 1].accel_vehicle_mps2, q_mount),
            ),
            ReducedInput::Trapezoid => (
                mounted_gyro(generated.imu[i + 1].gyro_vehicle_radps, q_mount),
                scale3(
                    add3(
                        mounted_accel(generated.imu[i].accel_vehicle_mps2, q_mount),
                        mounted_accel(generated.imu[i + 1].accel_vehicle_mps2, q_mount),
                    ),
                    0.5,
                ),
            ),
            ReducedInput::LaggedTrapezoid => {
                let prev_i = i.saturating_sub(1);
                (
                    mounted_gyro(generated.imu[i].gyro_vehicle_radps, q_mount),
                    scale3(
                        add3(
                            mounted_accel(generated.imu[prev_i].accel_vehicle_mps2, q_mount),
                            mounted_accel(generated.imu[i].accel_vehicle_mps2, q_mount),
                        ),
                        0.5,
                    ),
                )
            }
        };
        let truth_prev = generated.truth[i];
        let (gyro_predict, coriolis_delta_v_n) = reduced_navigation_rate_corrections(
            &reduced,
            truth_prev.lat_deg,
            truth_prev.height_m,
            dt,
            gyro_raw,
        );
        let imu = reduced_imu_delta(gyro_predict, accel_raw, dt);
        reduced.predict(imu);
        {
            let n = &mut reduced.raw_mut().nominal;
            n.vn += coriolis_delta_v_n[0] as f32;
            n.ve += coriolis_delta_v_n[1] as f32;
            n.vd += coriolis_delta_v_n[2] as f32;
        }
        let truth = generated.truth[i + 1];
        let n = &reduced.raw().nominal;
        final_vel_err = [
            n.vn as f64 - truth.vel_ned_mps[0],
            n.ve as f64 - truth.vel_ned_mps[1],
            n.vd as f64 - truth.vel_ned_mps[2],
        ];
        final_att_err = quat_angle_deg(as_q64([n.q0, n.q1, n.q2, n.q3]), truth.q_bn);
        sum_vel_err2 += norm3(final_vel_err).powi(2);
        count += 1;
    }
    summarize(
        generated.truth.last().map_or(0.0, |s| s.t_s),
        final_vel_err,
        sum_vel_err2,
        count,
        final_att_err,
    )
}

fn run_full(
    generated: &sim::synthetic::gnss_ins_path::GeneratedPath,
    q_mount: [f64; 4],
    mode: FullInput,
) -> Summary {
    let first = generated.truth[0];
    let ref_ecef = lla_to_ecef(first.lat_deg, first.lon_deg, first.height_m);
    let q_ne = quat_ecef_to_ned(first.lat_deg, first.lon_deg);
    let q_ev = quat_mul(quat_conj(q_ne), first.q_bn);
    let vel_ecef = ned_vector_to_ecef(first.lat_deg, first.lon_deg, first.vel_ned_mps);
    let mut full = full::Filter::new(ProcessNoise::default());
    full.init_from_reference_ecef_state(
        [
            q_ev[0] as f32,
            q_ev[1] as f32,
            q_ev[2] as f32,
            q_ev[3] as f32,
        ],
        ref_ecef,
        [vel_ecef[0] as f32, vel_ecef[1] as f32, vel_ecef[2] as f32],
        [0.0; 3],
        [0.0; 3],
        [1.0; 3],
        [1.0; 3],
        [
            q_mount[0] as f32,
            q_mount[1] as f32,
            q_mount[2] as f32,
            q_mount[3] as f32,
        ],
        None,
    );

    let mut sum_vel_err2 = 0.0;
    let mut count = 0usize;
    let mut final_vel_err = [0.0; 3];
    let mut final_att_err = 0.0;
    for i in 0..generated.imu.len() - 1 {
        let dt = generated.imu[i + 1].t_s - generated.imu[i].t_s;
        if dt <= 0.0 {
            continue;
        }
        let start_g = mounted_gyro(generated.imu[i].gyro_vehicle_radps, q_mount);
        let start_a = mounted_accel(generated.imu[i].accel_vehicle_mps2, q_mount);
        let end_g = mounted_gyro(generated.imu[i + 1].gyro_vehicle_radps, q_mount);
        let end_a = mounted_accel(generated.imu[i + 1].accel_vehicle_mps2, q_mount);
        let imu = match mode {
            FullInput::StartStart => full_imu_delta(start_g, start_a, start_g, start_a, dt),
            FullInput::StartEnd => full_imu_delta(start_g, start_a, end_g, end_a, dt),
            FullInput::EndEnd => full_imu_delta(end_g, end_a, end_g, end_a, dt),
            FullInput::LaggedStartEnd => {
                let prev_i = i.saturating_sub(1);
                full_imu_delta(
                    mounted_gyro(generated.imu[prev_i].gyro_vehicle_radps, q_mount),
                    mounted_accel(generated.imu[prev_i].accel_vehicle_mps2, q_mount),
                    start_g,
                    start_a,
                    dt,
                )
            }
        };
        full.predict(imu);
        let truth = generated.truth[i + 1];
        let n = full.nominal();
        let vel_ned = ecef_vector_to_ned(
            truth.lat_deg,
            truth.lon_deg,
            [n.vn as f64, n.ve as f64, n.vd as f64],
        );
        final_vel_err = [
            vel_ned[0] - truth.vel_ned_mps[0],
            vel_ned[1] - truth.vel_ned_mps[1],
            vel_ned[2] - truth.vel_ned_mps[2],
        ];
        let q_ns = quat_mul(
            quat_ecef_to_ned(truth.lat_deg, truth.lon_deg),
            [n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64],
        );
        final_att_err = quat_angle_deg(q_ns, truth.q_bn);
        sum_vel_err2 += norm3(final_vel_err).powi(2);
        count += 1;
    }
    summarize(
        generated.truth.last().map_or(0.0, |s| s.t_s),
        final_vel_err,
        sum_vel_err2,
        count,
        final_att_err,
    )
}

fn summarize(
    span_s: f64,
    final_vel_err: [f64; 3],
    sum_vel_err2: f64,
    count: usize,
    final_att_err_deg: f64,
) -> Summary {
    Summary {
        final_vel_err_ned: final_vel_err,
        rms_vel_err_mps: (sum_vel_err2 / count.max(1) as f64).sqrt(),
        final_att_err_deg,
        eq_accel_drift_mps2: if span_s > 0.0 {
            [
                final_vel_err[0] / span_s,
                final_vel_err[1] / span_s,
                final_vel_err[2] / span_s,
            ]
        } else {
            [f64::NAN; 3]
        },
    }
}

fn print_summary(label: &str, s: Summary) {
    println!(
        "{label:18} final_vel_err_ned=[{:+.6},{:+.6},{:+.6}]m/s eq_accel=[{:+.9},{:+.9},{:+.9}]m/s^2 rms_vel={:.6}m/s final_att_err={:.6}deg",
        s.final_vel_err_ned[0],
        s.final_vel_err_ned[1],
        s.final_vel_err_ned[2],
        s.eq_accel_drift_mps2[0],
        s.eq_accel_drift_mps2[1],
        s.eq_accel_drift_mps2[2],
        s.rms_vel_err_mps,
        s.final_att_err_deg,
    );
}

fn reduced_imu_delta(gyro: [f64; 3], accel: [f64; 3], dt: f64) -> reduced::ImuDelta {
    reduced::ImuDelta {
        dax: (gyro[0] * dt) as f32,
        day: (gyro[1] * dt) as f32,
        daz: (gyro[2] * dt) as f32,
        dvx: (accel[0] * dt) as f32,
        dvy: (accel[1] * dt) as f32,
        dvz: (accel[2] * dt) as f32,
        dt: dt as f32,
    }
}

fn full_imu_delta(
    gyro1: [f64; 3],
    accel1: [f64; 3],
    gyro2: [f64; 3],
    accel2: [f64; 3],
    dt: f64,
) -> full::ImuDelta {
    full::ImuDelta {
        dax_1: (gyro1[0] * dt) as f32,
        day_1: (gyro1[1] * dt) as f32,
        daz_1: (gyro1[2] * dt) as f32,
        dvx_1: (accel1[0] * dt) as f32,
        dvy_1: (accel1[1] * dt) as f32,
        dvz_1: (accel1[2] * dt) as f32,
        dax_2: (gyro2[0] * dt) as f32,
        day_2: (gyro2[1] * dt) as f32,
        daz_2: (gyro2[2] * dt) as f32,
        dvx_2: (accel2[0] * dt) as f32,
        dvy_2: (accel2[1] * dt) as f32,
        dvz_2: (accel2[2] * dt) as f32,
        dt: dt as f32,
    }
}

fn mounted_gyro(gyro_vehicle: [f64; 3], q_mount: [f64; 4]) -> [f64; 3] {
    sim::eval::gnss_ins::quat_rotate(q_mount, gyro_vehicle)
}

fn mounted_accel(accel_vehicle: [f64; 3], q_mount: [f64; 4]) -> [f64; 3] {
    sim::eval::gnss_ins::quat_rotate(q_mount, accel_vehicle)
}

fn reduced_navigation_rate_corrections(
    reduced: &reduced::Filter,
    lat_deg: f64,
    height_m: f64,
    dt: f64,
    gyro_body: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    if dt <= 0.0 {
        return (gyro_body, [0.0; 3]);
    }
    let n = &reduced.raw().nominal;
    let vel_ned = [n.vn as f64, n.ve as f64, n.vd as f64];
    let (omega_ie_n, omega_en_n) = navigation_rates_ned(lat_deg, height_m, vel_ned);
    let omega_in_n = add3(omega_ie_n, omega_en_n);
    let c_nv = quat_to_rotmat64([n.q0 as f64, n.q1 as f64, n.q2 as f64, n.q3 as f64]);
    let omega_in_v = mat3_vec64(transpose3(c_nv), omega_in_n);
    let c_bv = quat_to_rotmat64([
        n.q_bv0 as f64,
        n.q_bv1 as f64,
        n.q_bv2 as f64,
        n.q_bv3 as f64,
    ]);
    let omega_in_b = mat3_vec64(c_bv, omega_in_v);
    let gyro_predict = sub3(gyro_body, omega_in_b);
    let coriolis_rate = cross3(
        add3(scale3(omega_ie_n, 2.0), omega_en_n),
        [n.vn as f64, n.ve as f64, n.vd as f64],
    );
    (gyro_predict, scale3(coriolis_rate, -dt))
}

fn navigation_rates_ned(
    lat_deg: f64,
    height_m: f64,
    vel_ned_mps: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    const WGS84_A_M: f64 = 6_378_137.0;
    const WGS84_E2: f64 = 6.694_379_990_141_32e-3;
    const WGS84_OMEGA_IE_RADPS: f64 = 7.292_115e-5;
    let lat = lat_deg.to_radians();
    let (slat, clat) = lat.sin_cos();
    let denom = (1.0 - WGS84_E2 * slat * slat).max(1.0e-6);
    let sqrt_denom = denom.sqrt();
    let rn = WGS84_A_M / sqrt_denom;
    let rm = WGS84_A_M * (1.0 - WGS84_E2) / (denom * sqrt_denom);
    let rn_h = (rn + height_m).max(1.0);
    let rm_h = (rm + height_m).max(1.0);
    let clat_safe = if clat.abs() > 1.0e-3 {
        clat
    } else if clat.is_sign_negative() {
        -1.0e-3
    } else {
        1.0e-3
    };
    let omega_ie_n = [
        WGS84_OMEGA_IE_RADPS * clat,
        0.0,
        -WGS84_OMEGA_IE_RADPS * slat,
    ];
    let omega_en_n = [
        vel_ned_mps[1] / rn_h,
        -vel_ned_mps[0] / rm_h,
        -vel_ned_mps[1] * slat / (clat_safe * rn_h),
    ];
    (omega_ie_n, omega_en_n)
}

fn normal_gravity_mss(lat_deg: f64, height_m: f64) -> f32 {
    const WGS84_A_M: f64 = 6_378_137.0;
    const WGS84_E2: f64 = 6.694_379_990_141_32e-3;
    const WGS84_NORMAL_GRAVITY_EQUATOR: f64 = 9.780_325_335_9;
    const WGS84_NORMAL_GRAVITY_K: f64 = 0.001_931_852_652_41;
    const WGS84_NORMAL_GRAVITY_M: f64 = 0.003_449_786_506_84;
    let lat = lat_deg.to_radians();
    let slat = lat.sin();
    let slat2 = slat * slat;
    let sqrt_term = (1.0 - WGS84_E2 * slat2).sqrt();
    let surface_g =
        WGS84_NORMAL_GRAVITY_EQUATOR * (1.0 + WGS84_NORMAL_GRAVITY_K * slat2) / sqrt_term;
    let flattening = 1.0 - (1.0 - WGS84_E2).sqrt();
    let height_scale = 1.0
        - (2.0 / WGS84_A_M)
            * (1.0 + flattening + WGS84_NORMAL_GRAVITY_M - 2.0 * flattening * slat2)
            * height_m
        + 3.0 * height_m * height_m / (WGS84_A_M * WGS84_A_M);
    (surface_g * height_scale) as f32
}

fn quat_ecef_to_ned(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    dcm_to_quat(ecef_to_ned_matrix(lat_deg, lon_deg))
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

fn dcm_to_quat(c: [[f64; 3]; 3]) -> [f64; 4] {
    let trace = c[0][0] + c[1][1] + c[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (c[2][1] - c[1][2]) / s,
            (c[0][2] - c[2][0]) / s,
            (c[1][0] - c[0][1]) / s,
        ]
    } else if c[0][0] > c[1][1] && c[0][0] > c[2][2] {
        let s = (1.0 + c[0][0] - c[1][1] - c[2][2]).sqrt() * 2.0;
        [
            (c[2][1] - c[1][2]) / s,
            0.25 * s,
            (c[0][1] + c[1][0]) / s,
            (c[0][2] + c[2][0]) / s,
        ]
    } else if c[1][1] > c[2][2] {
        let s = (1.0 + c[1][1] - c[0][0] - c[2][2]).sqrt() * 2.0;
        [
            (c[0][2] - c[2][0]) / s,
            (c[0][1] + c[1][0]) / s,
            0.25 * s,
            (c[1][2] + c[2][1]) / s,
        ]
    } else {
        let s = (1.0 + c[2][2] - c[0][0] - c[1][1]).sqrt() * 2.0;
        [
            (c[1][0] - c[0][1]) / s,
            (c[0][2] + c[2][0]) / s,
            (c[1][2] + c[2][1]) / s,
            0.25 * s,
        ]
    };
    sim::eval::gnss_ins::quat_normalize(q)
}

fn ned_vector_to_ecef(lat_deg: f64, lon_deg: f64, v_ned: [f64; 3]) -> [f64; 3] {
    let c_ne = ecef_to_ned_matrix(lat_deg, lon_deg);
    [
        c_ne[0][0] * v_ned[0] + c_ne[1][0] * v_ned[1] + c_ne[2][0] * v_ned[2],
        c_ne[0][1] * v_ned[0] + c_ne[1][1] * v_ned[1] + c_ne[2][1] * v_ned[2],
        c_ne[0][2] * v_ned[0] + c_ne[1][2] * v_ned[1] + c_ne[2][2] * v_ned[2],
    ]
}

fn ecef_vector_to_ned(lat_deg: f64, lon_deg: f64, v_ecef: [f64; 3]) -> [f64; 3] {
    let c_ne = ecef_to_ned_matrix(lat_deg, lon_deg);
    [
        c_ne[0][0] * v_ecef[0] + c_ne[0][1] * v_ecef[1] + c_ne[0][2] * v_ecef[2],
        c_ne[1][0] * v_ecef[0] + c_ne[1][1] * v_ecef[1] + c_ne[1][2] * v_ecef[2],
        c_ne[2][0] * v_ecef[0] + c_ne[2][1] * v_ecef[1] + c_ne[2][2] * v_ecef[2],
    ]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn quat_to_rotmat64(q: [f64; 4]) -> [[f64; 3]; 3] {
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    let inv = if n2 > 1.0e-12 { 1.0 / n2.sqrt() } else { 1.0 };
    let q0 = q[0] * inv;
    let q1 = q[1] * inv;
    let q2 = q[2] * inv;
    let q3 = q[3] * inv;
    [
        [
            1.0 - 2.0 * (q2 * q2 + q3 * q3),
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),
        ],
        [
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * (q1 * q1 + q3 * q3),
            2.0 * (q2 * q3 - q0 * q1),
        ],
        [
            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * (q1 * q1 + q2 * q2),
        ],
    ]
}

fn transpose3(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn mat3_vec64(a: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn norm3(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
