#[cfg(not(feature = "c-reference"))]
fn main() {
    eprintln!(
        "compare_filter_runtime requires the sim c-reference feature: cargo run -p sim --features c-reference --bin compare_filter_runtime -- ..."
    );
}

#[cfg(feature = "c-reference")]
fn main() -> anyhow::Result<()> {
    c_reference::main()
}

#[cfg(feature = "c-reference")]
mod c_reference {
    use anyhow::{Context, Result, bail};
    use clap::Parser;
    use sensor_fusion::c_api::{
        CEskf, CEskfImuDelta, CEskfWrapper, CLooseImuDelta, CLooseNominalState, CLooseWrapper,
        EskfGnssSample as CEskfGnssSample,
    };
    use sensor_fusion::ekf::PredictNoise;
    use sensor_fusion::eskf_types::{
        EskfGnssSample as RustEskfGnssSample, EskfImuDelta as RustEskfImuDelta, EskfState,
    };
    use sensor_fusion::loose::{LooseFilter, LooseImuDelta, LoosePredictNoise};
    use sensor_fusion::rust_eskf::RustEskf;
    use serde::Deserialize;
    use sim::datasets::seeded_loose::{
        AccelSample, GnssSample, GyroSample, import_accel_data, import_gnss_data, import_gyro_data,
        resolve_single_file,
    };
    use sim::visualizer::math::{ecef_to_ned, lla_to_ecef};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long)]
        input_dir: PathBuf,
        #[arg(long)]
        init_json: PathBuf,
        #[arg(long, default_value_t = 3)]
        warmup_runs: usize,
        #[arg(long, default_value_t = 20)]
        timed_runs: usize,
        #[arg(long)]
        check_eskf_parity: bool,
        #[arg(long)]
        check_loose_parity: bool,
    }

    #[derive(Debug, Clone, Copy)]
    enum EventType {
        Accel,
        Gyro,
        Gnss,
    }

    #[derive(Debug, Clone, Copy)]
    struct Event {
        ttag_us: i64,
        event_type: EventType,
        index: usize,
    }

    #[derive(Debug, Deserialize)]
    struct RefInit {
        start_ttag_us: i64,
        ref_lat_deg: f64,
        ref_lon_deg: f64,
        ref_h_m: f64,
        q_bn: [f32; 4],
        pos_ecef_m: [f64; 3],
        vel_ecef_mps: [f32; 3],
        pos_ned_m: [f32; 3],
        vel_ned_mps: [f32; 3],
        gyro_bias_radps: [f32; 3],
        accel_bias_mps2: [f32; 3],
        gyro_scale: [f32; 3],
        accel_scale: [f32; 3],
        q_cs: [f32; 4],
        p_diag: [f32; 24],
        p_full: [[f32; 24]; 24],
    }

    #[derive(Debug, Clone, Copy)]
    struct GpsUpdate {
        eskf: CEskfGnssSample,
        pos_ecef_m: [f64; 3],
        vel_ecef_mps: [f32; 3],
        h_acc_m: f32,
        speed_acc_mps: f32,
        dt_since_last_gnss_s: f32,
    }

    #[derive(Debug, Clone, Copy)]
    struct ReplayStep {
        t_s: f64,
        eskf_imu: CEskfImuDelta,
        loose_imu: LooseImuDelta,
        gyro_radps: [f32; 3],
        accel_mps2: [f32; 3],
        gps: Option<GpsUpdate>,
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct FilterTiming {
        predict: Duration,
        gps_update: Duration,
        constrained_update: Duration,
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct FilterCounts {
        predict_steps: usize,
        gps_updates: usize,
        constrained_updates: usize,
        observation_rows: usize,
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct EskfParityDiff {
        max_nominal_abs: f32,
        max_cov_abs: f32,
        max_diag_abs: f32,
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct LooseParityDiff {
        max_nominal_abs: f32,
        max_cov_abs: f32,
        max_last_dx_abs: f32,
        max_shadow_pos_abs: f64,
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct FilterStats {
        total: Duration,
        timing: FilterTiming,
        counts: FilterCounts,
    }

    pub(super) fn main() -> Result<()> {
        let args = Args::parse();
        let init: RefInit = serde_json::from_slice(&fs::read(&args.init_json)?)?;
        let accel = import_accel_data(&resolve_single_file(&args.input_dir, "_Acc.csv")?)?;
        let gyro = import_gyro_data(&resolve_single_file(&args.input_dir, "_Gyro.csv")?)?;
        let gnss = import_gnss_data(&resolve_single_file(&args.input_dir, "_GNSS.csv")?)?;
        let steps = build_replay_steps(&init, &accel, &gyro, &gnss)?;
        if steps.is_empty() {
            bail!("no replay steps built from {}", args.input_dir.display());
        }

        if args.check_eskf_parity {
            let diff = check_eskf_parity(&init, &steps)?;
            println!(
                "ESKF C/Rust parity: max nominal={:.9e}, max covariance={:.9e}, max diagnostics={:.9e}",
                diff.max_nominal_abs, diff.max_cov_abs, diff.max_diag_abs
            );
        }
        if args.check_loose_parity {
            let diff = check_loose_parity(&init, &steps)?;
            println!(
                "Loose C/Rust parity: max nominal={:.9e}, max covariance={:.9e}, max last_dx={:.9e}, max shadow_pos={:.9e}",
                diff.max_nominal_abs,
                diff.max_cov_abs,
                diff.max_last_dx_abs,
                diff.max_shadow_pos_abs
            );
        }

        let warmup_runs = args.warmup_runs.max(1);
        let timed_runs = args.timed_runs.max(1);

        for _ in 0..warmup_runs {
            let _ = run_eskf(&init, &steps);
            let _ = run_rust_eskf(&init, &steps);
            let _ = run_loose(&init, &steps);
        }

        let mut eskf_total = FilterStats::default();
        let mut rust_eskf_total = FilterStats::default();
        let mut loose_full_total = FilterStats::default();
        for _ in 0..timed_runs {
            eskf_total += run_eskf(&init, &steps);
            rust_eskf_total += run_rust_eskf(&init, &steps);
            loose_full_total += run_loose(&init, &steps);
        }

        let data_span_s = steps.last().map(|s| s.t_s).unwrap_or(0.0);
        println!("Dataset: {}", args.input_dir.display());
        println!(
            "Steps: {}  Span: {:.3}s  Warmup: {}  Timed: {}",
            steps.len(),
            data_span_s,
            warmup_runs,
            timed_runs
        );
        println!();
        print_report("ESKF C", eskf_total, timed_runs, data_span_s);
        println!();
        print_report("ESKF Rust", rust_eskf_total, timed_runs, data_span_s);
        println!();
        print_report("Loose GPS+NHC", loose_full_total, timed_runs, data_span_s);
        Ok(())
    }

    fn build_replay_steps(
        init: &RefInit,
        accel: &[AccelSample],
        gyro: &[GyroSample],
        gnss: &[GnssSample],
    ) -> Result<Vec<ReplayStep>> {
        let mut events = Vec::with_capacity(accel.len() + gyro.len() + gnss.len());
        events.extend(gyro.iter().enumerate().map(|(index, s)| Event {
            ttag_us: s.ttag_us,
            event_type: EventType::Gyro,
            index,
        }));
        events.extend(accel.iter().enumerate().map(|(index, s)| Event {
            ttag_us: s.ttag_us,
            event_type: EventType::Accel,
            index,
        }));
        events.extend(gnss.iter().enumerate().map(|(index, s)| Event {
            ttag_us: s.ttag_us,
            event_type: EventType::Gnss,
            index,
        }));
        events.sort_by(|a, b| {
            a.ttag_us
                .cmp(&b.ttag_us)
                .then_with(|| event_rank(a.event_type).cmp(&event_rank(b.event_type)))
                .then_with(|| a.index.cmp(&b.index))
        });

        let mut steps = Vec::new();
        let mut started = false;
        let mut last_gnss_used_ttag = i64::MIN;
        let mut latest_gnss_index: Option<usize> = None;
        let mut accel_seen_count = 0usize;
        let ref_ecef = lla_to_ecef(init.ref_lat_deg, init.ref_lon_deg, init.ref_h_m);

        for event in events {
            match event.event_type {
                EventType::Accel => {
                    accel_seen_count = accel_seen_count.max(event.index + 1);
                }
                EventType::Gnss => {
                    latest_gnss_index = Some(event.index);
                }
                EventType::Gyro => {
                    let curr = &gyro[event.index];
                    if !started {
                        if curr.ttag_us < init.start_ttag_us || event.index == 0 {
                            continue;
                        }
                        started = true;
                    }
                    if event.index == 0 {
                        continue;
                    }
                    let prev = &gyro[event.index - 1];
                    if prev.ttag_us < init.start_ttag_us {
                        continue;
                    }

                    let accel_seen = &accel[..accel_seen_count];
                    let a1 = accel_at(prev.ttag_us, accel_seen)
                        .with_context(|| format!("missing accel at {}", prev.ttag_us))?;
                    let a2 = accel_at(curr.ttag_us, accel_seen)
                        .with_context(|| format!("missing accel at {}", curr.ttag_us))?;
                    let latest_accel = accel_seen
                        .last()
                        .map(|s| s.accel_mps2)
                        .with_context(|| format!("missing latest accel at {}", curr.ttag_us))?;
                    let dt = (curr.ttag_us - prev.ttag_us) as f64 * 1.0e-6;
                    if !(dt > 0.0) {
                        continue;
                    }

                    let gps = if let Some(gnss_index) = latest_gnss_index {
                        let g = &gnss[gnss_index];
                        let age_us = curr.ttag_us - g.ttag_us;
                        if age_us >= 0 && age_us < 50_000 && g.ttag_us != last_gnss_used_ttag {
                            let pos_ecef = lla_to_ecef(g.lat_deg, g.lon_deg, g.height_m);
                            let pos_ned =
                                ecef_to_ned(pos_ecef, ref_ecef, init.ref_lat_deg, init.ref_lon_deg);
                            let heading_rad = g.heading_deg.to_radians();
                            let vel_ned = [
                                (g.speed_mps * heading_rad.cos()) as f32,
                                (g.speed_mps * heading_rad.sin()) as f32,
                                0.0,
                            ];
                            let vel_ecef = mat_vec(
                                transpose3(ecef_to_ned_matrix(g.lat_deg, g.lon_deg)),
                                [vel_ned[0] as f64, vel_ned[1] as f64, vel_ned[2] as f64],
                            );
                            let dt_since_last_gnss_s = if last_gnss_used_ttag == i64::MIN {
                                1.0
                            } else {
                                ((curr.ttag_us - last_gnss_used_ttag) as f32 * 1.0e-6)
                                    .clamp(1.0e-3, 1.0)
                            };
                            last_gnss_used_ttag = g.ttag_us;
                            Some(GpsUpdate {
                                eskf: CEskfGnssSample {
                                    t_s: ((curr.ttag_us - init.start_ttag_us) as f64 * 1.0e-6)
                                        as f32,
                                    pos_ned_m: [
                                        pos_ned[0] as f32,
                                        pos_ned[1] as f32,
                                        pos_ned[2] as f32,
                                    ],
                                    vel_ned_mps: vel_ned,
                                    pos_std_m: [
                                        g.h_acc_m as f32,
                                        g.h_acc_m as f32,
                                        g.v_acc_m as f32,
                                    ],
                                    vel_std_mps: [g.speed_acc_mps as f32; 3],
                                    heading_rad: None,
                                },
                                pos_ecef_m: pos_ecef,
                                vel_ecef_mps: [
                                    vel_ecef[0] as f32,
                                    vel_ecef[1] as f32,
                                    vel_ecef[2] as f32,
                                ],
                                h_acc_m: g.h_acc_m as f32,
                                speed_acc_mps: g.speed_acc_mps as f32,
                                dt_since_last_gnss_s,
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    steps.push(ReplayStep {
                        t_s: (curr.ttag_us - init.start_ttag_us) as f64 * 1.0e-6,
                        eskf_imu: CEskfImuDelta {
                            dax: (curr.omega_radps[0] * dt) as f32,
                            day: (curr.omega_radps[1] * dt) as f32,
                            daz: (curr.omega_radps[2] * dt) as f32,
                            dvx: (a2[0] * dt) as f32,
                            dvy: (a2[1] * dt) as f32,
                            dvz: (a2[2] * dt) as f32,
                            dt: dt as f32,
                        },
                        loose_imu: LooseImuDelta {
                            dax_1: (prev.omega_radps[0] * dt) as f32,
                            day_1: (prev.omega_radps[1] * dt) as f32,
                            daz_1: (prev.omega_radps[2] * dt) as f32,
                            dvx_1: (a1[0] * dt) as f32,
                            dvy_1: (a1[1] * dt) as f32,
                            dvz_1: (a1[2] * dt) as f32,
                            dax_2: (curr.omega_radps[0] * dt) as f32,
                            day_2: (curr.omega_radps[1] * dt) as f32,
                            daz_2: (curr.omega_radps[2] * dt) as f32,
                            dvx_2: (a2[0] * dt) as f32,
                            dvy_2: (a2[1] * dt) as f32,
                            dvz_2: (a2[2] * dt) as f32,
                            dt: dt as f32,
                        },
                        gyro_radps: [
                            curr.omega_radps[0] as f32,
                            curr.omega_radps[1] as f32,
                            curr.omega_radps[2] as f32,
                        ],
                        accel_mps2: [
                            latest_accel[0] as f32,
                            latest_accel[1] as f32,
                            latest_accel[2] as f32,
                        ],
                        gps,
                    });
                }
            }
        }
        Ok(steps)
    }

    fn run_eskf(init: &RefInit, steps: &[ReplayStep]) -> FilterStats {
        run_eskf_with_state(init, steps).0
    }

    fn run_eskf_with_state(init: &RefInit, steps: &[ReplayStep]) -> (FilterStats, CEskfWrapper) {
        let mut eskf = CEskfWrapper::new(PredictNoise::default());
        let init_gnss = CEskfGnssSample {
            t_s: 0.0,
            pos_ned_m: init.pos_ned_m,
            vel_ned_mps: init.vel_ned_mps,
            pos_std_m: [2.5, 2.5, 100.0],
            vel_std_mps: [20.0, 20.0, 20.0],
            heading_rad: None,
        };
        eskf.init_nominal_from_gnss(init.q_bn, init_gnss);

        let total_start = Instant::now();
        let mut stats = FilterStats::default();
        for step in steps {
            let t0 = Instant::now();
            eskf.predict(step.eskf_imu);
            stats.timing.predict += t0.elapsed();
            stats.counts.predict_steps += 1;

            if let Some(gps) = step.gps {
                let t1 = Instant::now();
                eskf.fuse_gps(gps.eskf);
                stats.timing.gps_update += t1.elapsed();
                stats.counts.gps_updates += 1;
            }
        }
        stats.total = total_start.elapsed();
        (stats, eskf)
    }

    fn run_rust_eskf(init: &RefInit, steps: &[ReplayStep]) -> FilterStats {
        run_rust_eskf_with_state(init, steps).0
    }

    fn run_rust_eskf_with_state(init: &RefInit, steps: &[ReplayStep]) -> (FilterStats, RustEskf) {
        let mut eskf = RustEskf::new(PredictNoise::default());
        let init_gnss = RustEskfGnssSample {
            t_s: 0.0,
            pos_ned_m: init.pos_ned_m,
            vel_ned_mps: init.vel_ned_mps,
            pos_std_m: [2.5, 2.5, 100.0],
            vel_std_mps: [20.0, 20.0, 20.0],
            heading_rad: None,
        };
        eskf.init_nominal_from_gnss(init.q_bn, init_gnss);

        let total_start = Instant::now();
        let mut stats = FilterStats::default();
        for step in steps {
            let t0 = Instant::now();
            eskf.predict(rust_imu_from_c(step.eskf_imu));
            stats.timing.predict += t0.elapsed();
            stats.counts.predict_steps += 1;

            if let Some(gps) = step.gps {
                let t1 = Instant::now();
                eskf.fuse_gps(rust_gnss_from_c(gps.eskf));
                stats.timing.gps_update += t1.elapsed();
                stats.counts.gps_updates += 1;
            }
        }
        stats.total = total_start.elapsed();
        (stats, eskf)
    }

    fn run_loose(init: &RefInit, steps: &[ReplayStep]) -> FilterStats {
        run_loose_with_state(init, steps).0
    }

    fn run_loose_with_state(init: &RefInit, steps: &[ReplayStep]) -> (FilterStats, LooseFilter) {
        let mut loose = LooseFilter::new(LoosePredictNoise::reference_nsr_demo());
        loose.init_from_reference_ecef_state(
            init.q_bn,
            init.pos_ecef_m,
            init.vel_ecef_mps,
            init.gyro_bias_radps,
            init.accel_bias_mps2,
            init.gyro_scale,
            init.accel_scale,
            init.q_cs,
            Some(init.p_diag),
        );
        loose.set_covariance(init.p_full);

        let total_start = Instant::now();
        let mut stats = FilterStats::default();
        for step in steps {
            let t0 = Instant::now();
            loose.predict(step.loose_imu);
            stats.timing.predict += t0.elapsed();
            stats.counts.predict_steps += 1;

            let t1 = Instant::now();
            loose.fuse_reference_batch(
                step.gps.map(|g| g.pos_ecef_m),
                step.gps.map(|g| g.vel_ecef_mps),
                step.gps.map_or(0.0, |g| g.h_acc_m),
                step.gps.map_or(0.0, |g| g.speed_acc_mps),
                step.gps.map_or(1.0, |g| g.dt_since_last_gnss_s),
                step.gyro_radps,
                step.accel_mps2,
                step.loose_imu.dt,
            );
            stats.timing.constrained_update += t1.elapsed();
            if step.gps.is_some() {
                stats.counts.gps_updates += 1;
            }
            let obs_types = loose.last_obs_types();
            if !obs_types.is_empty() {
                stats.counts.constrained_updates += 1;
                stats.counts.observation_rows += obs_types.len();
            }
        }
        stats.total = total_start.elapsed();
        (stats, loose)
    }

    fn print_report(name: &str, total: FilterStats, timed_runs: usize, data_span_s: f64) {
        let avg = total / timed_runs as u32;
        let predict_ns = nanos_per(avg.timing.predict, avg.counts.predict_steps);
        let gps_us = nanos_per(avg.timing.gps_update, avg.counts.gps_updates) / 1_000.0;
        let constrained_us = nanos_per(
            avg.timing.constrained_update,
            avg.counts.constrained_updates,
        ) / 1_000.0;
        let obs_us =
            nanos_per(avg.timing.constrained_update, avg.counts.observation_rows) / 1_000.0;
        let realtime = if avg.total.as_secs_f64() > 0.0 {
            data_span_s / avg.total.as_secs_f64()
        } else {
            0.0
        };
        println!("{name}");
        println!(
            "  total:        {:8.3} ms   realtime: {:8.2}x",
            avg.total.as_secs_f64() * 1.0e3,
            realtime
        );
        println!(
            "  predict:      {:8} steps {:10.1} ns/step",
            avg.counts.predict_steps, predict_ns
        );
        println!(
            "  gps update:   {:8} calls {:10.3} us/call",
            avg.counts.gps_updates, gps_us
        );
        println!(
            "  constrained:  {:8} calls {:10.3} us/call",
            avg.counts.constrained_updates, constrained_us
        );
        println!(
            "  obs rows:     {:8} rows  {:10.3} us/row",
            avg.counts.observation_rows, obs_us
        );
    }

    fn nanos_per(duration: Duration, count: usize) -> f64 {
        if count == 0 {
            0.0
        } else {
            duration.as_secs_f64() * 1.0e9 / count as f64
        }
    }

    fn check_eskf_parity(init: &RefInit, steps: &[ReplayStep]) -> Result<EskfParityDiff> {
        let (_, c) = run_eskf_with_state(init, steps);
        let (_, r) = run_rust_eskf_with_state(init, steps);
        let diff = eskf_parity_diff(c.raw(), r.raw());
        const TOL: f32 = 1.0e-6;
        if diff.max_nominal_abs > TOL || diff.max_cov_abs > TOL || diff.max_diag_abs > TOL {
            bail!(
                "ESKF C/Rust output mismatch: max nominal={:.9e}, max covariance={:.9e}, max diagnostics={:.9e}, tolerance={:.9e}",
                diff.max_nominal_abs,
                diff.max_cov_abs,
                diff.max_diag_abs,
                TOL
            );
        }
        Ok(diff)
    }

    fn check_loose_parity(init: &RefInit, steps: &[ReplayStep]) -> Result<LooseParityDiff> {
        let mut c = CLooseWrapper::new(LoosePredictNoise::reference_nsr_demo());
        c.init_from_reference_ecef_state(
            init.q_bn,
            init.pos_ecef_m,
            init.vel_ecef_mps,
            init.gyro_bias_radps,
            init.accel_bias_mps2,
            init.gyro_scale,
            init.accel_scale,
            init.q_cs,
            Some(init.p_diag),
        );
        c.set_covariance(init.p_full);

        let mut r = LooseFilter::new(LoosePredictNoise::reference_nsr_demo());
        r.init_from_reference_ecef_state(
            init.q_bn,
            init.pos_ecef_m,
            init.vel_ecef_mps,
            init.gyro_bias_radps,
            init.accel_bias_mps2,
            init.gyro_scale,
            init.accel_scale,
            init.q_cs,
            Some(init.p_diag),
        );
        r.set_covariance(init.p_full);

        let mut diff = LooseParityDiff::default();
        for (index, step) in steps.iter().enumerate() {
            c.predict(c_loose_imu_from_rust(step.loose_imu));
            r.predict(step.loose_imu);
            diff = diff.max(loose_parity_diff(&c, &r));

            c.fuse_reference_batch(
                step.gps.map(|g| g.pos_ecef_m),
                step.gps.map(|g| g.vel_ecef_mps),
                step.gps.map_or(0.0, |g| g.h_acc_m),
                step.gps.map_or(0.0, |g| g.speed_acc_mps),
                step.gps.map_or(1.0, |g| g.dt_since_last_gnss_s),
                step.gyro_radps,
                step.accel_mps2,
                step.loose_imu.dt,
            );
            r.fuse_reference_batch(
                step.gps.map(|g| g.pos_ecef_m),
                step.gps.map(|g| g.vel_ecef_mps),
                step.gps.map_or(0.0, |g| g.h_acc_m),
                step.gps.map_or(0.0, |g| g.speed_acc_mps),
                step.gps.map_or(1.0, |g| g.dt_since_last_gnss_s),
                step.gyro_radps,
                step.accel_mps2,
                step.loose_imu.dt,
            );
            if c.last_obs_types() != r.last_obs_types() {
                bail!(
                    "Loose C/Rust output mismatch at step {index} t={:.6}: obs types differ: C={:?}, Rust={:?}",
                    step.t_s,
                    c.last_obs_types(),
                    r.last_obs_types()
                );
            }
            diff = diff.max(loose_parity_diff(&c, &r));
        }

        const TOL_F32: f32 = 2.0e-5;
        const TOL_F64: f64 = 5.0e-5;
        if diff.max_nominal_abs > TOL_F32
            || diff.max_cov_abs > TOL_F32
            || diff.max_last_dx_abs > TOL_F32
            || diff.max_shadow_pos_abs > TOL_F64
        {
            bail!(
                "Loose C/Rust output mismatch: max nominal={:.9e}, max covariance={:.9e}, max last_dx={:.9e}, max shadow_pos={:.9e}, tolerances f32={:.9e} f64={:.9e}",
                diff.max_nominal_abs,
                diff.max_cov_abs,
                diff.max_last_dx_abs,
                diff.max_shadow_pos_abs,
                TOL_F32,
                TOL_F64
            );
        }
        Ok(diff)
    }

    fn loose_parity_diff(c: &CLooseWrapper, r: &LooseFilter) -> LooseParityDiff {
        let mut diff = LooseParityDiff::default();
        diff_nominal(c.nominal(), r.nominal(), &mut diff);
        for i in 0..24 {
            for j in 0..24 {
                diff.max_cov_abs = diff
                    .max_cov_abs
                    .max((c.covariance()[i][j] - r.covariance()[i][j]).abs());
            }
            diff.max_last_dx_abs = diff
                .max_last_dx_abs
                .max((c.last_dx()[i] - r.last_dx()[i]).abs());
        }
        let c_pos = c.shadow_pos_ecef();
        let r_pos = r.shadow_pos_ecef();
        for i in 0..3 {
            diff.max_shadow_pos_abs = diff.max_shadow_pos_abs.max((c_pos[i] - r_pos[i]).abs());
        }
        diff
    }

    fn diff_nominal(
        c: &CLooseNominalState,
        r: &sensor_fusion::loose::LooseNominalState,
        diff: &mut LooseParityDiff,
    ) {
        let pairs = [
            (c.q0, r.q0),
            (c.q1, r.q1),
            (c.q2, r.q2),
            (c.q3, r.q3),
            (c.vn, r.vn),
            (c.ve, r.ve),
            (c.vd, r.vd),
            (c.pn, r.pn),
            (c.pe, r.pe),
            (c.pd, r.pd),
            (c.bgx, r.bgx),
            (c.bgy, r.bgy),
            (c.bgz, r.bgz),
            (c.bax, r.bax),
            (c.bay, r.bay),
            (c.baz, r.baz),
            (c.sgx, r.sgx),
            (c.sgy, r.sgy),
            (c.sgz, r.sgz),
            (c.sax, r.sax),
            (c.say, r.say),
            (c.saz, r.saz),
            (c.qcs0, r.qcs0),
            (c.qcs1, r.qcs1),
            (c.qcs2, r.qcs2),
            (c.qcs3, r.qcs3),
        ];
        for (a, b) in pairs {
            diff.max_nominal_abs = diff.max_nominal_abs.max((a - b).abs());
        }
    }

    fn eskf_parity_diff(c: &CEskf, r: &EskfState) -> EskfParityDiff {
        let cn = &c.nominal;
        let rn = &r.nominal;
        let nominal_pairs = [
            (cn.q0, rn.q0),
            (cn.q1, rn.q1),
            (cn.q2, rn.q2),
            (cn.q3, rn.q3),
            (cn.vn, rn.vn),
            (cn.ve, rn.ve),
            (cn.vd, rn.vd),
            (cn.pn, rn.pn),
            (cn.pe, rn.pe),
            (cn.pd, rn.pd),
            (cn.bgx, rn.bgx),
            (cn.bgy, rn.bgy),
            (cn.bgz, rn.bgz),
            (cn.bax, rn.bax),
            (cn.bay, rn.bay),
            (cn.baz, rn.baz),
            (cn.qcs0, rn.qcs0),
            (cn.qcs1, rn.qcs1),
            (cn.qcs2, rn.qcs2),
            (cn.qcs3, rn.qcs3),
        ];
        let mut diff = EskfParityDiff::default();
        for (a, b) in nominal_pairs {
            diff.max_nominal_abs = diff.max_nominal_abs.max((a - b).abs());
        }
        for i in 0..18 {
            for j in 0..18 {
                diff.max_cov_abs = diff.max_cov_abs.max((c.p[i][j] - r.p[i][j]).abs());
            }
        }

        let stationary_pairs = [
            (
                c.stationary_diag.innovation_x,
                r.stationary_diag.innovation_x,
            ),
            (
                c.stationary_diag.innovation_y,
                r.stationary_diag.innovation_y,
            ),
            (
                c.stationary_diag.k_theta_x_from_x,
                r.stationary_diag.k_theta_x_from_x,
            ),
            (
                c.stationary_diag.k_theta_y_from_x,
                r.stationary_diag.k_theta_y_from_x,
            ),
            (
                c.stationary_diag.k_bax_from_x,
                r.stationary_diag.k_bax_from_x,
            ),
            (
                c.stationary_diag.k_bay_from_x,
                r.stationary_diag.k_bay_from_x,
            ),
            (
                c.stationary_diag.k_theta_x_from_y,
                r.stationary_diag.k_theta_x_from_y,
            ),
            (
                c.stationary_diag.k_theta_y_from_y,
                r.stationary_diag.k_theta_y_from_y,
            ),
            (
                c.stationary_diag.k_bax_from_y,
                r.stationary_diag.k_bax_from_y,
            ),
            (
                c.stationary_diag.k_bay_from_y,
                r.stationary_diag.k_bay_from_y,
            ),
            (c.stationary_diag.p_theta_x, r.stationary_diag.p_theta_x),
            (c.stationary_diag.p_theta_y, r.stationary_diag.p_theta_y),
            (c.stationary_diag.p_bax, r.stationary_diag.p_bax),
            (c.stationary_diag.p_bay, r.stationary_diag.p_bay),
            (
                c.stationary_diag.p_theta_x_bax,
                r.stationary_diag.p_theta_x_bax,
            ),
            (
                c.stationary_diag.p_theta_y_bay,
                r.stationary_diag.p_theta_y_bay,
            ),
        ];
        for (a, b) in stationary_pairs {
            diff.max_diag_abs = diff.max_diag_abs.max((a - b).abs());
        }

        for i in 0..c.update_diag.type_counts.len() {
            diff.max_diag_abs = diff
                .max_diag_abs
                .max((c.update_diag.sum_dx_pitch[i] - r.update_diag.sum_dx_pitch[i]).abs())
                .max((c.update_diag.sum_abs_dx_pitch[i] - r.update_diag.sum_abs_dx_pitch[i]).abs())
                .max((c.update_diag.sum_dx_mount_yaw[i] - r.update_diag.sum_dx_mount_yaw[i]).abs())
                .max(
                    (c.update_diag.sum_abs_dx_mount_yaw[i] - r.update_diag.sum_abs_dx_mount_yaw[i])
                        .abs(),
                )
                .max((c.update_diag.sum_innovation[i] - r.update_diag.sum_innovation[i]).abs())
                .max(
                    (c.update_diag.sum_abs_innovation[i] - r.update_diag.sum_abs_innovation[i])
                        .abs(),
                );
        }
        diff.max_diag_abs = diff
            .max_diag_abs
            .max((c.update_diag.last_dx_mount_yaw - r.update_diag.last_dx_mount_yaw).abs())
            .max((c.update_diag.last_k_mount_yaw - r.update_diag.last_k_mount_yaw).abs())
            .max((c.update_diag.last_innovation - r.update_diag.last_innovation).abs())
            .max((c.update_diag.last_innovation_var - r.update_diag.last_innovation_var).abs());
        diff
    }

    fn rust_imu_from_c(imu: CEskfImuDelta) -> RustEskfImuDelta {
        RustEskfImuDelta {
            dax: imu.dax,
            day: imu.day,
            daz: imu.daz,
            dvx: imu.dvx,
            dvy: imu.dvy,
            dvz: imu.dvz,
            dt: imu.dt,
        }
    }

    fn rust_gnss_from_c(gnss: CEskfGnssSample) -> RustEskfGnssSample {
        RustEskfGnssSample {
            t_s: gnss.t_s,
            pos_ned_m: gnss.pos_ned_m,
            vel_ned_mps: gnss.vel_ned_mps,
            pos_std_m: gnss.pos_std_m,
            vel_std_mps: gnss.vel_std_mps,
            heading_rad: gnss.heading_rad,
        }
    }

    fn c_loose_imu_from_rust(imu: LooseImuDelta) -> CLooseImuDelta {
        CLooseImuDelta {
            dax_1: imu.dax_1,
            day_1: imu.day_1,
            daz_1: imu.daz_1,
            dvx_1: imu.dvx_1,
            dvy_1: imu.dvy_1,
            dvz_1: imu.dvz_1,
            dax_2: imu.dax_2,
            day_2: imu.day_2,
            daz_2: imu.daz_2,
            dvx_2: imu.dvx_2,
            dvy_2: imu.dvy_2,
            dvz_2: imu.dvz_2,
            dt: imu.dt,
        }
    }

    fn event_rank(event_type: EventType) -> u8 {
        match event_type {
            EventType::Gyro => 1,
            EventType::Accel => 2,
            EventType::Gnss => 4,
        }
    }

    fn accel_at(ttag_us: i64, accel: &[AccelSample]) -> Option<[f64; 3]> {
        match accel.binary_search_by(|s| s.ttag_us.cmp(&ttag_us)) {
            Ok(index) => Some(accel[index].accel_mps2),
            Err(0) | Err(_) if accel.is_empty() => None,
            Err(index) if index >= accel.len() => {
                let s = &accel[accel.len() - 1];
                ((ttag_us - s.ttag_us).abs() <= 100_000).then_some(s.accel_mps2)
            }
            Err(index) => {
                let prev = &accel[index - 1];
                let next = &accel[index];
                if ttag_us - prev.ttag_us > 100_000 || next.ttag_us - ttag_us > 100_000 {
                    return None;
                }
                let span = (next.ttag_us - prev.ttag_us) as f64;
                if span <= 0.0 {
                    return Some(prev.accel_mps2);
                }
                let a = (ttag_us - prev.ttag_us) as f64 / span;
                Some([
                    prev.accel_mps2[0] + a * (next.accel_mps2[0] - prev.accel_mps2[0]),
                    prev.accel_mps2[1] + a * (next.accel_mps2[1] - prev.accel_mps2[1]),
                    prev.accel_mps2[2] + a * (next.accel_mps2[2] - prev.accel_mps2[2]),
                ])
            }
        }
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

    impl core::ops::AddAssign for FilterStats {
        fn add_assign(&mut self, rhs: Self) {
            self.total += rhs.total;
            self.timing.predict += rhs.timing.predict;
            self.timing.gps_update += rhs.timing.gps_update;
            self.timing.constrained_update += rhs.timing.constrained_update;
            self.counts.predict_steps += rhs.counts.predict_steps;
            self.counts.gps_updates += rhs.counts.gps_updates;
            self.counts.constrained_updates += rhs.counts.constrained_updates;
            self.counts.observation_rows += rhs.counts.observation_rows;
        }
    }

    impl core::ops::Div<u32> for FilterStats {
        type Output = Self;

        fn div(self, rhs: u32) -> Self::Output {
            let div = rhs.max(1) as f64;
            Self {
                total: Duration::from_secs_f64(self.total.as_secs_f64() / div),
                timing: FilterTiming {
                    predict: Duration::from_secs_f64(self.timing.predict.as_secs_f64() / div),
                    gps_update: Duration::from_secs_f64(self.timing.gps_update.as_secs_f64() / div),
                    constrained_update: Duration::from_secs_f64(
                        self.timing.constrained_update.as_secs_f64() / div,
                    ),
                },
                counts: FilterCounts {
                    predict_steps: (self.counts.predict_steps as f64 / div).round() as usize,
                    gps_updates: (self.counts.gps_updates as f64 / div).round() as usize,
                    constrained_updates: (self.counts.constrained_updates as f64 / div).round()
                        as usize,
                    observation_rows: (self.counts.observation_rows as f64 / div).round() as usize,
                },
            }
        }
    }

    impl LooseParityDiff {
        fn max(self, rhs: Self) -> Self {
            Self {
                max_nominal_abs: self.max_nominal_abs.max(rhs.max_nominal_abs),
                max_cov_abs: self.max_cov_abs.max(rhs.max_cov_abs),
                max_last_dx_abs: self.max_last_dx_abs.max(rhs.max_last_dx_abs),
                max_shadow_pos_abs: self.max_shadow_pos_abs.max(rhs.max_shadow_pos_abs),
            }
        }
    }
}
