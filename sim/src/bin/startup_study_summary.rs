use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "startup_study_summary")]
struct Args {
    #[arg(long, default_value = "/tmp/startup_study_baseline.csv")]
    csv: PathBuf,
}

#[derive(Debug, Clone)]
struct Row {
    logfile: String,
    t_s: f64,
    speed_kmh: f64,
    course_rate_dps: f64,
    a_long_mps2: f64,
    a_lat_mps2: f64,
    gnss_accel_norm_mps2: f64,
    startup_gate_valid: bool,
    startup_accepted: bool,
    startup_emitted: bool,
    err_startup_theta_deg: f64,
    err_startup_theta_alt_deg: f64,
    err_final_esf_deg: f64,
    final_fwd_err_align_deg: f64,
    final_down_err_align_deg: f64,
}

#[derive(Default, Clone)]
struct Stats {
    abs_err_final_esf: Vec<f64>,
    basin_margin_deg: Vec<f64>,
    theta_better_count: usize,
}

impl Stats {
    fn push(&mut self, row: &Row) {
        if row.err_final_esf_deg.is_finite() {
            self.abs_err_final_esf.push(row.err_final_esf_deg.abs());
        }
        if row.err_startup_theta_deg.is_finite() && row.err_startup_theta_alt_deg.is_finite() {
            let margin = row.err_startup_theta_alt_deg.abs() - row.err_startup_theta_deg.abs();
            self.basin_margin_deg.push(margin);
            if margin > 0.0 {
                self.theta_better_count += 1;
            }
        }
    }

    fn count(&self) -> usize {
        self.abs_err_final_esf.len()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let rows = load_rows(&args.csv)?;
    if rows.is_empty() {
        bail!("no rows in {}", args.csv.display());
    }

    println!("CSV: {}", args.csv.display());
    println!("rows: {}", rows.len());
    println!();

    print_subset_summary("all", rows.iter());
    print_subset_summary("startup_gate_valid", rows.iter().filter(|r| r.startup_gate_valid));
    print_subset_summary("startup_accepted", rows.iter().filter(|r| r.startup_accepted));
    println!();

    println!("Per-log startup emission summary");
    println!(
        "{:<30} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "logfile", "t_emit", "theta_better", "alt_better", "mean|e_th|", "mean|e_alt|"
    );
    let mut by_log: BTreeMap<&str, Vec<&Row>> = BTreeMap::new();
    for row in &rows {
        by_log.entry(&row.logfile).or_default().push(row);
    }
    for (log, rs) in &by_log {
        let accepted: Vec<&Row> = rs.iter().copied().filter(|r| r.startup_accepted).collect();
        let emit_t = rs
            .iter()
            .find(|r| r.startup_emitted)
            .map(|r| r.t_s)
            .unwrap_or(f64::NAN);
        let theta_better = accepted
            .iter()
            .filter(|r| r.err_startup_theta_deg.abs() < r.err_startup_theta_alt_deg.abs())
            .count();
        let alt_better = accepted
            .iter()
            .filter(|r| r.err_startup_theta_alt_deg.abs() < r.err_startup_theta_deg.abs())
            .count();
        let mean_th = mean(accepted.iter().map(|r| r.err_startup_theta_deg.abs()));
        let mean_alt = mean(accepted.iter().map(|r| r.err_startup_theta_alt_deg.abs()));
        println!(
            "{:<30} {:>8.2} {:>10} {:>10} {:>10.2} {:>10.2}",
            log, emit_t, theta_better, alt_better, mean_th, mean_alt
        );
    }
    println!();

    print_binned_summary(
        "startup_accepted by speed [km/h]",
        rows.iter().filter(|r| r.startup_accepted),
        |r| r.speed_kmh,
        &[0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0],
    );
    print_binned_summary(
        "startup_accepted by |course_rate| [deg/s]",
        rows.iter().filter(|r| r.startup_accepted),
        |r| r.course_rate_dps.abs(),
        &[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0],
    );
    print_binned_summary(
        "startup_accepted by accel norm [m/s^2]",
        rows.iter().filter(|r| r.startup_accepted),
        |r| r.gnss_accel_norm_mps2,
        &[0.0, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0],
    );
    print_binned_summary(
        "startup_accepted by |a_lat|/max(|a_long|,eps)",
        rows.iter().filter(|r| r.startup_accepted),
        |r| {
            let denom = r.a_long_mps2.abs().max(1.0e-3);
            r.a_lat_mps2.abs() / denom
        },
        &[0.0, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0],
    );
    print_binned_summary(
        "all windows by speed [km/h]",
        rows.iter(),
        |r| r.speed_kmh,
        &[0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0],
    );

    Ok(())
}

fn load_rows(path: &PathBuf) -> Result<Vec<Row>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut lines = text.lines();
    let header = lines.next().context("missing CSV header")?;
    let cols: Vec<&str> = header.split(',').collect();
    let idx = |name: &str| -> Result<usize> {
        cols.iter()
            .position(|c| *c == name)
            .with_context(|| format!("missing column '{}'", name))
    };
    let logfile_i = idx("logfile")?;
    let t_s_i = idx("t_s")?;
    let speed_i = idx("speed_kmh")?;
    let course_rate_i = idx("course_rate_dps")?;
    let a_long_i = idx("a_long_mps2")?;
    let a_lat_i = idx("a_lat_mps2")?;
    let accel_norm_i = idx("gnss_accel_norm_mps2")?;
    let gate_i = idx("startup_gate_valid")?;
    let accept_i = idx("startup_accepted")?;
    let emit_i = idx("startup_emitted")?;
    let err_theta_i = idx("err_startup_theta_deg")?;
    let err_theta_alt_i = idx("err_startup_theta_alt_deg")?;
    let err_final_esf_i = idx("err_final_esf_deg")?;
    let final_fwd_i = idx("final_fwd_err_align_deg")?;

    let final_down_i = idx("final_down_err_align_deg")?;

    let mut out = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        out.push(Row {
            logfile: fields[logfile_i].to_string(),
            t_s: parse_f64(fields[t_s_i])?,
            speed_kmh: parse_f64(fields[speed_i])?,
            course_rate_dps: parse_f64(fields[course_rate_i])?,
            a_long_mps2: parse_f64(fields[a_long_i])?,
            a_lat_mps2: parse_f64(fields[a_lat_i])?,
            gnss_accel_norm_mps2: parse_f64(fields[accel_norm_i])?,
            startup_gate_valid: parse_bool01(fields[gate_i])?,
            startup_accepted: parse_bool01(fields[accept_i])?,
            startup_emitted: parse_bool01(fields[emit_i])?,
            err_startup_theta_deg: parse_f64(fields[err_theta_i])?,
            err_startup_theta_alt_deg: parse_f64(fields[err_theta_alt_i])?,
            err_final_esf_deg: parse_f64(fields[err_final_esf_i])?,
            final_fwd_err_align_deg: parse_f64(fields[final_fwd_i])?,
            final_down_err_align_deg: parse_f64(fields[final_down_i])?,
        });
    }
    Ok(out)
}

fn print_subset_summary<'a>(name: &str, rows: impl Iterator<Item = &'a Row>) {
    let mut stats = Stats::default();
    let mut count = 0usize;
    let mut fwd_errs = Vec::new();
    for row in rows {
        count += 1;
        stats.push(row);
        if row.final_fwd_err_align_deg.is_finite() {
            fwd_errs.push(row.final_fwd_err_align_deg.abs());
        }
    }
    println!(
        "{:<18} n={:<5} median|e_esf|={:>6.2} p90={:>6.2} <5={:>5.1}% <10={:>5.1}% <20={:>5.1}% theta_better={:>5.1}% med_margin={:>6.2} med|final_fwd|={:>6.2}",
        name,
        count,
        percentile(stats.abs_err_final_esf.clone(), 0.5),
        percentile(stats.abs_err_final_esf.clone(), 0.9),
        pct_below(&stats.abs_err_final_esf, 5.0),
        pct_below(&stats.abs_err_final_esf, 10.0),
        pct_below(&stats.abs_err_final_esf, 20.0),
        if stats.basin_margin_deg.is_empty() {
            f64::NAN
        } else {
            100.0 * stats.theta_better_count as f64 / stats.basin_margin_deg.len() as f64
        },
        percentile(stats.basin_margin_deg.clone(), 0.5),
        percentile(fwd_errs, 0.5),
    );
}

fn print_binned_summary<'a>(
    title: &str,
    rows: impl Iterator<Item = &'a Row>,
    value_fn: impl Fn(&Row) -> f64,
    edges: &[f64],
) {
    println!("{}", title);
    println!(
        "{:<18} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "bin", "n", "med|e_esf|", "p90|e_esf|", "theta>", "med_margin"
    );
    let mut bins: Vec<(String, Stats)> = edges
        .windows(2)
        .map(|w| (format!("[{:.2},{:.2})", w[0], w[1]), Stats::default()))
        .collect();
    bins.push((format!("[{:.2},inf)", edges[edges.len() - 1]), Stats::default()));
    for row in rows {
        let v = value_fn(row);
        let idx = bin_index(v, edges);
        bins[idx].1.push(row);
    }
    for (label, stats) in bins {
        if stats.count() == 0 {
            continue;
        }
        println!(
            "{:<18} {:>6} {:>10.2} {:>10.2} {:>9.1}% {:>10.2}",
            label,
            stats.count(),
            percentile(stats.abs_err_final_esf.clone(), 0.5),
            percentile(stats.abs_err_final_esf.clone(), 0.9),
            if stats.basin_margin_deg.is_empty() {
                f64::NAN
            } else {
                100.0 * stats.theta_better_count as f64 / stats.basin_margin_deg.len() as f64
            },
            percentile(stats.basin_margin_deg.clone(), 0.5),
        );
    }
    println!();
}

fn bin_index(v: f64, edges: &[f64]) -> usize {
    for (i, w) in edges.windows(2).enumerate() {
        if v >= w[0] && v < w[1] {
            return i;
        }
    }
    edges.len() - 1
}

fn percentile(mut xs: Vec<f64>, q: f64) -> f64 {
    xs.retain(|x| x.is_finite());
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let pos = ((xs.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    xs[pos]
}

fn pct_below(xs: &[f64], thr: f64) -> f64 {
    let finite: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    100.0 * finite.iter().filter(|x| **x < thr).count() as f64 / finite.len() as f64
}

fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for x in xs {
        if x.is_finite() {
            sum += x;
            n += 1;
        }
    }
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

fn parse_f64(s: &str) -> Result<f64> {
    s.parse::<f64>()
        .with_context(|| format!("failed to parse f64 '{}'", s))
}

fn parse_bool01(s: &str) -> Result<bool> {
    match s {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => bail!("failed to parse bool01 '{}'", s),
    }
}
