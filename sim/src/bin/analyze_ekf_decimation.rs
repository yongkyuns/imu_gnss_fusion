use anyhow::Result;
use clap::Parser;
use sensor_fusion::ekf::{
    Ekf, GpsData, GRAVITY_MSS, ImuSample, PredictNoise, ekf_fuse_body_vel, ekf_fuse_gps,
    ekf_predict, ekf_set_predict_noise,
};

#[derive(Parser, Debug)]
#[command(name = "analyze_ekf_decimation")]
struct Args {
    #[arg(long, default_value_t = 3)]
    decimation: usize,

    #[arg(long, default_value_t = 600.0)]
    duration_s: f32,

    #[arg(long, default_value_t = 100.0)]
    imu_hz: f32,

    #[arg(long, default_value_t = 5.0)]
    r_body_vel: f32,
}

#[derive(Clone, Copy, Debug)]
struct TruthStep {
    t_s: f32,
    yaw_rad: f32,
    pos_ned_m: [f32; 3],
    vel_ned_mps: [f32; 3],
    accel_body_mps2: [f32; 3],
    gyro_body_radps: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    name: &'static str,
    use_body_vel: bool,
    gps_period_s: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
struct SampledState {
    t_s: f32,
    q_bn: [f32; 4],
    vel_ned_mps: [f32; 3],
    pos_ned_m: [f32; 3],
    p_diag: [f32; 16],
}

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    q_max_abs: f32,
    vel_rms: f32,
    pos_rms: f32,
    p_diag_rms: f32,
    worst_p_diag_idx: usize,
    worst_p_diag_rms: f32,
    accel_bias_cov_x_rms: f32,
    accel_bias_cov_z_rms: f32,
    gps_delay_avg_ms: f32,
    gps_delay_max_ms: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct GpsDelayStats {
    count: usize,
    total_s: f32,
    max_s: f32,
}

struct DecimatedRun {
    hist: Vec<SampledState>,
    gps_delay: GpsDelayStats,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let decimation = args.decimation.max(1);
    let truth = build_truth(args.duration_s, args.imu_hz);
    let base_noise = PredictNoise::default();
    let scenarios = [
        Scenario {
            name: "predict_only",
            use_body_vel: false,
            gps_period_s: None,
        },
        Scenario {
            name: "predict_plus_gps_aligned_0p60s",
            use_body_vel: false,
            gps_period_s: Some(0.60),
        },
        Scenario {
            name: "predict_plus_gps_unaligned_0p50s",
            use_body_vel: false,
            gps_period_s: Some(0.50),
        },
        Scenario {
            name: "predict_plus_body_vel",
            use_body_vel: true,
            gps_period_s: None,
        },
        Scenario {
            name: "predict_plus_body_vel_plus_gps_unaligned_0p50s",
            use_body_vel: true,
            gps_period_s: Some(0.50),
        },
    ];

    println!(
        "scenario,decimation,q_max_abs,vel_rms_mps,pos_rms_m,pdiag_rms,worst_p_diag_idx,worst_p_diag_rms,acc_bias_x_cov_rms,acc_bias_z_cov_rms,acc_bias_x_cross_s_full,acc_bias_x_cross_s_dec,acc_bias_z_cross_s_full,acc_bias_z_cross_s_dec,gps_delay_avg_ms,gps_delay_max_ms"
    );

    for scenario in scenarios {
        let full = run_full_rate(&truth, scenario, args.r_body_vel, base_noise);
        let dec = run_decimated(&truth, scenario, args.r_body_vel, base_noise, decimation);
        let metrics = compare_histories(&full, &dec.hist, decimation, dec.gps_delay);
        let full_x_cross = first_below(&full, 13, 1.0e-7);
        let dec_x_cross = first_below(&dec.hist, 13, 1.0e-7);
        let full_z_cross = first_below(&full, 15, 1.0e-7);
        let dec_z_cross = first_below(&dec.hist, 15, 1.0e-7);

        println!(
            "{},{},{:.9},{:.6},{:.6},{:.9},{},{:.9},{:.9},{:.9},{},{},{},{},{:.3},{:.3}",
            scenario.name,
            decimation,
            metrics.q_max_abs,
            metrics.vel_rms,
            metrics.pos_rms,
            metrics.p_diag_rms,
            metrics.worst_p_diag_idx,
            metrics.worst_p_diag_rms,
            metrics.accel_bias_cov_x_rms,
            metrics.accel_bias_cov_z_rms,
            format_opt(full_x_cross),
            format_opt(dec_x_cross),
            format_opt(full_z_cross),
            format_opt(dec_z_cross),
            metrics.gps_delay_avg_ms,
            metrics.gps_delay_max_ms,
        );
    }

    Ok(())
}

fn build_truth(duration_s: f32, imu_hz: f32) -> Vec<TruthStep> {
    let dt = 1.0 / imu_hz.max(1.0);
    let n_steps = (duration_s / dt).round().max(1.0) as usize + 1;
    let mut out = Vec::with_capacity(n_steps);
    let mut yaw = 0.0_f32;
    let mut pn = 0.0_f32;
    let mut pe = 0.0_f32;
    let pd = 0.0_f32;
    let mut v = 14.0_f32;

    for i in 0..n_steps {
        let t = i as f32 * dt;
        let yaw_rate = 0.03 * (0.07 * t).sin() + 0.015 * (0.021 * t).cos();
        let a_long = 0.35 * (0.05 * t).sin() + 0.15 * (0.011 * t).cos();
        if i > 0 {
            v = (v + a_long * dt).max(5.0);
            yaw += yaw_rate * dt;
        }
        let vn = v * yaw.cos();
        let ve = v * yaw.sin();
        if i > 0 {
            pn += vn * dt;
            pe += ve * dt;
        }
        let a_lat = v * yaw_rate;
        out.push(TruthStep {
            t_s: t,
            yaw_rad: yaw,
            pos_ned_m: [pn, pe, pd],
            vel_ned_mps: [vn, ve, 0.0],
            accel_body_mps2: [a_long, a_lat, -GRAVITY_MSS],
            gyro_body_radps: [0.0, 0.0, yaw_rate],
        });
    }
    out
}

fn init_ekf(truth0: TruthStep) -> Ekf {
    let mut ekf = Ekf::default();
    ekf.state.q0 = (0.5 * truth0.yaw_rad).cos();
    ekf.state.q1 = 0.0;
    ekf.state.q2 = 0.0;
    ekf.state.q3 = (0.5 * truth0.yaw_rad).sin();
    ekf.state.vn = truth0.vel_ned_mps[0];
    ekf.state.ve = truth0.vel_ned_mps[1];
    ekf.state.vd = truth0.vel_ned_mps[2];
    ekf.state.pn = truth0.pos_ned_m[0];
    ekf.state.pe = truth0.pos_ned_m[1];
    ekf.state.pd = truth0.pos_ned_m[2];
    ekf
}

fn run_full_rate(
    truth: &[TruthStep],
    scenario: Scenario,
    r_body_vel: f32,
    noise: PredictNoise,
) -> Vec<SampledState> {
    let mut ekf = init_ekf(truth[0]);
    ekf_set_predict_noise(&mut ekf, noise);
    let mut out = Vec::with_capacity(truth.len().saturating_sub(1));
    let dt0 = truth[1].t_s - truth[0].t_s;
    let gps_period_steps = scenario
        .gps_period_s
        .map(|period_s| (period_s / dt0).round().max(1.0) as usize);
    let mut next_gps_idx = gps_period_steps;

    for (step_idx, w) in truth.windows(2).enumerate() {
        let prev = w[0];
        let cur = w[1];
        let dt = cur.t_s - prev.t_s;
        let imu = ImuSample {
            dax: cur.gyro_body_radps[0] * dt,
            day: cur.gyro_body_radps[1] * dt,
            daz: cur.gyro_body_radps[2] * dt,
            dvx: cur.accel_body_mps2[0] * dt,
            dvy: cur.accel_body_mps2[1] * dt,
            dvz: cur.accel_body_mps2[2] * dt,
            dt,
        };
        ekf_predict(&mut ekf, &imu, None);
        if scenario.use_body_vel {
            ekf_fuse_body_vel(&mut ekf, r_body_vel);
        }
        while let Some(gps_idx) = next_gps_idx {
            let hist_idx = step_idx + 1;
            if gps_idx > hist_idx {
                break;
            }
            ekf_fuse_gps(&mut ekf, &gps_from_truth(truth[gps_idx]));
            next_gps_idx = gps_period_steps.map(|period| gps_idx + period);
        }
        out.push(sample_state(cur.t_s, &ekf));
    }

    out
}

fn run_decimated(
    truth: &[TruthStep],
    scenario: Scenario,
    r_body_vel: f32,
    base_noise: PredictNoise,
    decimation: usize,
) -> DecimatedRun {
    let mut ekf = init_ekf(truth[0]);
    ekf_set_predict_noise(&mut ekf, base_noise);
    let mut out = Vec::new();
    let mut gps_delay = GpsDelayStats::default();
    let dt0 = truth[1].t_s - truth[0].t_s;
    let gps_period_steps = scenario
        .gps_period_s
        .map(|period_s| (period_s / dt0).round().max(1.0) as usize);
    let mut next_gps_idx = gps_period_steps;

    let mut dax = 0.0_f32;
    let mut day = 0.0_f32;
    let mut daz = 0.0_f32;
    let mut dvx = 0.0_f32;
    let mut dvy = 0.0_f32;
    let mut dvz = 0.0_f32;
    let mut dt_accum = 0.0_f32;
    let mut block_count = 0usize;

    for (step_idx, w) in truth.windows(2).enumerate() {
        let prev = w[0];
        let cur = w[1];
        let dt = cur.t_s - prev.t_s;
        dax += cur.gyro_body_radps[0] * dt;
        day += cur.gyro_body_radps[1] * dt;
        daz += cur.gyro_body_radps[2] * dt;
        dvx += cur.accel_body_mps2[0] * dt;
        dvy += cur.accel_body_mps2[1] * dt;
        dvz += cur.accel_body_mps2[2] * dt;
        dt_accum += dt;
        block_count += 1;

        if block_count < decimation {
            continue;
        }

        let k = block_count as f32;
        let scaled_noise = PredictNoise {
            gyro_var: base_noise.gyro_var / k,
            accel_var: base_noise.accel_var / k,
            gyro_bias_rw_var: base_noise.gyro_bias_rw_var / k,
            accel_bias_rw_var: base_noise.accel_bias_rw_var / k,
        };
        ekf_set_predict_noise(&mut ekf, scaled_noise);
        let imu = ImuSample {
            dax,
            day,
            daz,
            dvx,
            dvy,
            dvz,
            dt: dt_accum,
        };
        ekf_predict(&mut ekf, &imu, None);
        if scenario.use_body_vel {
            ekf_fuse_body_vel(&mut ekf, r_body_vel / k);
        }
        ekf_set_predict_noise(&mut ekf, base_noise);

        while let Some(gps_idx) = next_gps_idx {
            let hist_idx = step_idx + 1;
            if gps_idx > hist_idx {
                break;
            }
            ekf_fuse_gps(&mut ekf, &gps_from_truth(truth[gps_idx]));
            let delay = (cur.t_s - truth[gps_idx].t_s).max(0.0);
            gps_delay.count += 1;
            gps_delay.total_s += delay;
            gps_delay.max_s = gps_delay.max_s.max(delay);
            next_gps_idx = gps_period_steps.map(|period| gps_idx + period);
        }

        out.push(sample_state(cur.t_s, &ekf));
        dax = 0.0;
        day = 0.0;
        daz = 0.0;
        dvx = 0.0;
        dvy = 0.0;
        dvz = 0.0;
        dt_accum = 0.0;
        block_count = 0;
    }

    DecimatedRun {
        hist: out,
        gps_delay,
    }
}

fn gps_from_truth(step: TruthStep) -> GpsData {
    GpsData {
        pos_n: step.pos_ned_m[0],
        pos_e: step.pos_ned_m[1],
        pos_d: step.pos_ned_m[2],
        vel_n: step.vel_ned_mps[0],
        vel_e: step.vel_ned_mps[1],
        vel_d: step.vel_ned_mps[2],
        R_POS_N: 0.05,
        R_POS_E: 0.05,
        R_POS_D: 0.05,
        R_VEL_N: 0.02,
        R_VEL_E: 0.02,
        R_VEL_D: 0.02,
    }
}

fn sample_state(t_s: f32, ekf: &Ekf) -> SampledState {
    let mut p_diag = [0.0_f32; 16];
    for (i, v) in p_diag.iter_mut().enumerate() {
        *v = ekf.p[i][i];
    }
    SampledState {
        t_s,
        q_bn: [ekf.state.q0, ekf.state.q1, ekf.state.q2, ekf.state.q3],
        vel_ned_mps: [ekf.state.vn, ekf.state.ve, ekf.state.vd],
        pos_ned_m: [ekf.state.pn, ekf.state.pe, ekf.state.pd],
        p_diag,
    }
}

fn compare_histories(
    full: &[SampledState],
    dec: &[SampledState],
    decimation: usize,
    gps_delay: GpsDelayStats,
) -> Metrics {
    let mut n = 0usize;
    let mut sum_vel2 = 0.0_f64;
    let mut sum_pos2 = 0.0_f64;
    let mut sum_p2 = 0.0_f64;
    let mut sum_p_diag2 = [0.0_f64; 16];
    let mut sum_p13_2 = 0.0_f64;
    let mut sum_p15_2 = 0.0_f64;
    let mut q_max_abs = 0.0_f32;

    for (i, d) in dec.iter().enumerate() {
        let f = full[(i + 1) * decimation - 1];
        q_max_abs = q_max_abs.max(max_abs_diff4(f.q_bn, d.q_bn));
        sum_vel2 += vec3_sq_diff(f.vel_ned_mps, d.vel_ned_mps) as f64;
        sum_pos2 += vec3_sq_diff(f.pos_ned_m, d.pos_ned_m) as f64;
        sum_p2 += diag_sq_diff(f.p_diag, d.p_diag) as f64;
        for j in 0..16 {
            sum_p_diag2[j] += ((f.p_diag[j] - d.p_diag[j]) as f64).powi(2);
        }
        sum_p13_2 += ((f.p_diag[13] - d.p_diag[13]) as f64).powi(2);
        sum_p15_2 += ((f.p_diag[15] - d.p_diag[15]) as f64).powi(2);
        n += 1;
    }

    let inv_n = if n > 0 { 1.0 / n as f64 } else { 0.0 };
    let mut worst_idx = 0usize;
    let mut worst_rms = 0.0_f32;
    for (i, s) in sum_p_diag2.into_iter().enumerate() {
        let rms = (s * inv_n).sqrt() as f32;
        if rms > worst_rms {
            worst_rms = rms;
            worst_idx = i;
        }
    }
    Metrics {
        q_max_abs,
        vel_rms: (sum_vel2 * inv_n).sqrt() as f32,
        pos_rms: (sum_pos2 * inv_n).sqrt() as f32,
        p_diag_rms: (sum_p2 * inv_n).sqrt() as f32,
        worst_p_diag_idx: worst_idx,
        worst_p_diag_rms: worst_rms,
        accel_bias_cov_x_rms: (sum_p13_2 * inv_n).sqrt() as f32,
        accel_bias_cov_z_rms: (sum_p15_2 * inv_n).sqrt() as f32,
        gps_delay_avg_ms: if gps_delay.count > 0 {
            gps_delay.total_s * 1000.0 / gps_delay.count as f32
        } else {
            0.0
        },
        gps_delay_max_ms: gps_delay.max_s * 1000.0,
    }
}

fn first_below(hist: &[SampledState], idx: usize, threshold: f32) -> Option<f32> {
    hist.iter().find(|s| s.p_diag[idx] <= threshold).map(|s| s.t_s)
}

fn format_opt(v: Option<f32>) -> String {
    match v {
        Some(x) => format!("{x:.3}"),
        None => "none".to_string(),
    }
}

fn max_abs_diff4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a.into_iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn vec3_sq_diff(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn diag_sq_diff(a: [f32; 16], b: [f32; 16]) -> f32 {
    let mut s = 0.0_f32;
    for i in 0..16 {
        s += (a[i] - b[i]).powi(2);
    }
    s
}
