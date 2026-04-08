use anyhow::{Context, Result, bail};
use clap::Parser;
use sensor_fusion::c_api::{CEskfImuDelta, CEskfWrapper, CLooseImuDelta, CLooseWrapper, EskfGnssSample};
use sensor_fusion::ekf::PredictNoise;
use sensor_fusion::loose::LoosePredictNoise;
use serde::Deserialize;
use sim::visualizer::math::{ecef_to_ned, lla_to_ecef};
use std::fs;
use std::path::{Path, PathBuf};
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
}

#[derive(Debug, Clone)]
struct GyroSample {
    ttag_us: i64,
    omega_radps: [f64; 3],
}

#[derive(Debug, Clone)]
struct AccelSample {
    ttag_us: i64,
    accel_mps2: [f64; 3],
}

#[derive(Debug, Clone)]
struct GnssSample {
    ttag_us: i64,
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    speed_mps: f64,
    heading_deg: f64,
    h_acc_m: f64,
    v_acc_m: f64,
    speed_acc_mps: f64,
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
    eskf: EskfGnssSample,
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
    loose_imu: CLooseImuDelta,
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
struct FilterStats {
    total: Duration,
    timing: FilterTiming,
    counts: FilterCounts,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let init: RefInit = serde_json::from_slice(&fs::read(&args.init_json)?)?;
    let accel = import_accel_data(&resolve_single_file(&args.input_dir, "_Acc.csv")?)?;
    let gyro = import_gyro_data(&resolve_single_file(&args.input_dir, "_Gyro.csv")?)?;
    let gnss = import_gnss_data(&resolve_single_file(&args.input_dir, "_GNSS.csv")?)?;
    let steps = build_replay_steps(&init, &accel, &gyro, &gnss)?;
    if steps.is_empty() {
        bail!("no replay steps built from {}", args.input_dir.display());
    }

    let warmup_runs = args.warmup_runs.max(1);
    let timed_runs = args.timed_runs.max(1);

    for _ in 0..warmup_runs {
        let _ = run_eskf(&init, &steps);
        let _ = run_loose(&init, &steps);
    }

    let mut eskf_total = FilterStats::default();
    let mut loose_full_total = FilterStats::default();
    for _ in 0..timed_runs {
        eskf_total += run_eskf(&init, &steps);
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
    print_report("ESKF", eskf_total, timed_runs, data_span_s);
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
                        let pos_ned = ecef_to_ned(pos_ecef, ref_ecef, init.ref_lat_deg, init.ref_lon_deg);
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
                            ((curr.ttag_us - last_gnss_used_ttag) as f32 * 1.0e-6).clamp(1.0e-3, 1.0)
                        };
                        last_gnss_used_ttag = g.ttag_us;
                        Some(GpsUpdate {
                            eskf: EskfGnssSample {
                                t_s: ((curr.ttag_us - init.start_ttag_us) as f64 * 1.0e-6) as f32,
                                pos_ned_m: [pos_ned[0] as f32, pos_ned[1] as f32, pos_ned[2] as f32],
                                vel_ned_mps: vel_ned,
                                pos_std_m: [g.h_acc_m as f32, g.h_acc_m as f32, g.v_acc_m as f32],
                                vel_std_mps: [g.speed_acc_mps as f32; 3],
                                heading_rad: None,
                            },
                            pos_ecef_m: pos_ecef,
                            vel_ecef_mps: [vel_ecef[0] as f32, vel_ecef[1] as f32, vel_ecef[2] as f32],
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
                    loose_imu: CLooseImuDelta {
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
    let mut eskf = CEskfWrapper::new(PredictNoise::default());
    let init_gnss = EskfGnssSample {
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
    stats
}

fn run_loose(init: &RefInit, steps: &[ReplayStep]) -> FilterStats {
    let mut loose = CLooseWrapper::new(LoosePredictNoise::reference_nsr_demo());
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
    stats
}

fn print_report(name: &str, total: FilterStats, timed_runs: usize, data_span_s: f64) {
    let avg = total / timed_runs as u32;
    let predict_ns = nanos_per(avg.timing.predict, avg.counts.predict_steps);
    let gps_us = nanos_per(avg.timing.gps_update, avg.counts.gps_updates) / 1_000.0;
    let constrained_us = nanos_per(avg.timing.constrained_update, avg.counts.constrained_updates) / 1_000.0;
    let obs_us = nanos_per(avg.timing.constrained_update, avg.counts.observation_rows) / 1_000.0;
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
        avg.counts.predict_steps,
        predict_ns
    );
    println!(
        "  gps update:   {:8} calls {:10.3} us/call",
        avg.counts.gps_updates,
        gps_us
    );
    println!(
        "  constrained:  {:8} calls {:10.3} us/call",
        avg.counts.constrained_updates,
        constrained_us
    );
    println!(
        "  obs rows:     {:8} rows  {:10.3} us/row",
        avg.counts.observation_rows,
        obs_us
    );
}

fn nanos_per(duration: Duration, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        duration.as_secs_f64() * 1.0e9 / count as f64
    }
}

fn event_rank(event_type: EventType) -> u8 {
    match event_type {
        EventType::Gyro => 1,
        EventType::Accel => 2,
        EventType::Gnss => 4,
    }
}

fn resolve_single_file(input_dir: &Path, suffix: &str) -> Result<PathBuf> {
    let mut matches = Vec::new();
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(suffix))
        {
            matches.push(path);
        }
    }
    matches.sort();
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => bail!("missing file with suffix {suffix} in {}", input_dir.display()),
        _ => bail!("multiple files with suffix {suffix} in {}", input_dir.display()),
    }
}

fn import_gyro_data(path: &Path) -> Result<Vec<GyroSample>> {
    let rows = semicolon_rows(path, 3)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        out.push(GyroSample {
            ttag_us: (parse_f64(&row[0])? / 1000.0).floor() as i64,
            omega_radps: [parse_f64(&row[1])?, parse_f64(&row[2])?, parse_f64(&row[3])?],
        });
    }
    Ok(out)
}

fn import_accel_data(path: &Path) -> Result<Vec<AccelSample>> {
    let rows = semicolon_rows(path, 3)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        out.push(AccelSample {
            ttag_us: (parse_f64(&row[0])? / 1000.0).floor() as i64,
            accel_mps2: [parse_f64(&row[1])?, parse_f64(&row[2])?, parse_f64(&row[3])?],
        });
    }
    Ok(out)
}

fn import_gnss_data(path: &Path) -> Result<Vec<GnssSample>> {
    let rows = semicolon_rows(path, 1)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        out.push(GnssSample {
            ttag_us: (parse_f64(&row[0])? / 1000.0).floor() as i64,
            lat_deg: parse_f64(&row[2])?,
            lon_deg: parse_f64(&row[3])?,
            height_m: parse_f64(&row[4])?,
            speed_mps: parse_f64(&row[5])?,
            heading_deg: parse_f64(&row[6])?,
            h_acc_m: parse_f64(&row[7])?,
            v_acc_m: parse_f64(&row[8])?,
            speed_acc_mps: parse_f64(&row[9])?,
        });
    }
    Ok(out)
}

fn semicolon_rows(path: &Path, skip_rows: usize) -> Result<Vec<Vec<String>>> {
    let text = fs::read_to_string(path)?;
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
                constrained_updates: (self.counts.constrained_updates as f64 / div).round() as usize,
                observation_rows: (self.counts.observation_rows as f64 / div).round() as usize,
            },
        }
    }
}
