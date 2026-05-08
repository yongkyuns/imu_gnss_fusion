use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "filter_equivalence_summary")]
#[command(about = "Summarize Reduced/full deltas from filter_equivalence_harness CSV output")]
struct Args {
    #[arg(required = true)]
    csv: Vec<PathBuf>,
    #[arg(long, default_value_t = 0.5)]
    pos_threshold_m: f64,
    #[arg(long, default_value_t = 0.05)]
    vel_threshold_mps: f64,
    #[arg(long, default_value_t = 0.5)]
    attitude_threshold_deg: f64,
    #[arg(long, default_value_t = 0.5)]
    mount_threshold_deg: f64,
    #[arg(long, default_value_t = 0.02)]
    accel_bias_threshold_mps2: f64,
    #[arg(long, default_value_t = 0.01)]
    gyro_bias_threshold_dps: f64,
    #[arg(long)]
    start_time_s: Option<f64>,
    #[arg(long)]
    end_time_s: Option<f64>,
    #[arg(long)]
    markdown: bool,
}

#[derive(Clone, Copy)]
struct MetricSpec {
    label: &'static str,
    columns: [&'static str; 3],
    threshold: f64,
}

#[derive(Default, Clone, Copy)]
struct MetricStats {
    count: usize,
    sum_sq: [f64; 3],
    max_abs: [f64; 3],
    max_time_s: [f64; 3],
    first_cross_time_s: [Option<f64>; 3],
    final_value: [f64; 3],
    final_time_s: f64,
}

#[derive(Default)]
struct ReadyStats {
    total: usize,
    both_ready: usize,
    reduced_ready: usize,
    full_ready: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    for path in &args.csv {
        summarize_file(path, &args)?;
    }
    Ok(())
}

fn summarize_file(path: &PathBuf, args: &Args) -> Result<()> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut lines = text.lines();
    let Some(header_line) = lines.next() else {
        bail!("{} is empty", path.display());
    };
    let headers: Vec<_> = header_line.split(',').collect();
    let index: HashMap<&str, usize> = headers
        .iter()
        .enumerate()
        .map(|(i, name)| (*name, i))
        .collect();
    let specs = metric_specs(args);
    let mut stats = vec![MetricStats::default(); specs.len()];
    let mut ready = ReadyStats::default();
    let t_idx = required_index(&index, "t_s")?;
    let reduced_ready_idx = required_index(&index, "reduced_ready")?;
    let full_ready_idx = required_index(&index, "full_ready")?;

    for line in lines.filter(|line| !line.trim().is_empty()) {
        let row: Vec<_> = line.split(',').collect();
        if row.len() != headers.len() {
            continue;
        }
        let t_s = parse_cell(row[t_idx]);
        if !t_s.is_finite() {
            continue;
        }
        if let Some(start) = args.start_time_s
            && t_s < start
        {
            continue;
        }
        if let Some(end) = args.end_time_s
            && t_s > end
        {
            continue;
        }
        let reduced_ready = parse_bool(row[reduced_ready_idx]);
        let full_ready = parse_bool(row[full_ready_idx]);
        ready.total += 1;
        ready.reduced_ready += usize::from(reduced_ready);
        ready.full_ready += usize::from(full_ready);
        if !(reduced_ready && full_ready) {
            continue;
        }
        ready.both_ready += 1;

        for (spec, stat) in specs.iter().zip(stats.iter_mut()) {
            let values = [
                parse_cell(row[required_index(&index, spec.columns[0])?]),
                parse_cell(row[required_index(&index, spec.columns[1])?]),
                parse_cell(row[required_index(&index, spec.columns[2])?]),
            ];
            accumulate(stat, values, t_s, spec.threshold);
        }
    }

    if args.markdown {
        print_markdown(path, &specs, &stats, ready);
    } else {
        print_text(path, &specs, &stats, ready);
    }
    Ok(())
}

fn metric_specs(args: &Args) -> Vec<MetricSpec> {
    vec![
        MetricSpec {
            label: "position diff [m]",
            columns: ["diff_pn_m", "diff_pe_m", "diff_pd_m"],
            threshold: args.pos_threshold_m,
        },
        MetricSpec {
            label: "velocity diff [m/s]",
            columns: ["diff_vn_mps", "diff_ve_mps", "diff_vd_mps"],
            threshold: args.vel_threshold_mps,
        },
        MetricSpec {
            label: "attitude Reduced-full [deg]",
            columns: [
                "att_diff_roll_deg",
                "att_diff_pitch_deg",
                "att_diff_yaw_deg",
            ],
            threshold: args.attitude_threshold_deg,
        },
        MetricSpec {
            label: "Reduced attitude-reference [deg]",
            columns: [
                "reduced_att_err_roll_deg",
                "reduced_att_err_pitch_deg",
                "reduced_att_err_yaw_deg",
            ],
            threshold: args.attitude_threshold_deg,
        },
        MetricSpec {
            label: "Full attitude-reference [deg]",
            columns: [
                "full_att_err_roll_deg",
                "full_att_err_pitch_deg",
                "full_att_err_yaw_deg",
            ],
            threshold: args.attitude_threshold_deg,
        },
        MetricSpec {
            label: "mount Reduced-full [deg]",
            columns: [
                "mount_diff_roll_deg",
                "mount_diff_pitch_deg",
                "mount_diff_yaw_deg",
            ],
            threshold: args.mount_threshold_deg,
        },
        MetricSpec {
            label: "Reduced mount-reference [deg]",
            columns: [
                "reduced_mount_err_roll_deg",
                "reduced_mount_err_pitch_deg",
                "reduced_mount_err_yaw_deg",
            ],
            threshold: args.mount_threshold_deg,
        },
        MetricSpec {
            label: "Full mount-reference [deg]",
            columns: [
                "full_mount_err_roll_deg",
                "full_mount_err_pitch_deg",
                "full_mount_err_yaw_deg",
            ],
            threshold: args.mount_threshold_deg,
        },
        MetricSpec {
            label: "gyro bias Reduced-full [deg/s]",
            columns: [
                "gyro_bias_diff_x_dps",
                "gyro_bias_diff_y_dps",
                "gyro_bias_diff_z_dps",
            ],
            threshold: args.gyro_bias_threshold_dps,
        },
        MetricSpec {
            label: "accel bias Reduced-full [m/s^2]",
            columns: [
                "accel_bias_diff_x_mps2",
                "accel_bias_diff_y_mps2",
                "accel_bias_diff_z_mps2",
            ],
            threshold: args.accel_bias_threshold_mps2,
        },
        MetricSpec {
            label: "Reduced GNSS vel residual [m/s]",
            columns: [
                "reduced_gnss_vel_res_n_mps",
                "reduced_gnss_vel_res_e_mps",
                "reduced_gnss_vel_res_d_mps",
            ],
            threshold: args.vel_threshold_mps,
        },
        MetricSpec {
            label: "Full GNSS vel residual [m/s]",
            columns: [
                "full_gnss_vel_res_n_mps",
                "full_gnss_vel_res_e_mps",
                "full_gnss_vel_res_d_mps",
            ],
            threshold: args.vel_threshold_mps,
        },
    ]
}

fn accumulate(stat: &mut MetricStats, values: [f64; 3], t_s: f64, threshold: f64) {
    if !values.iter().all(|v| v.is_finite()) {
        return;
    }
    stat.count += 1;
    stat.final_value = values;
    stat.final_time_s = t_s;
    for (axis, value) in values.into_iter().enumerate() {
        stat.sum_sq[axis] += value * value;
        let abs = value.abs();
        if abs >= stat.max_abs[axis] {
            stat.max_abs[axis] = abs;
            stat.max_time_s[axis] = t_s;
        }
        if stat.first_cross_time_s[axis].is_none() && abs > threshold {
            stat.first_cross_time_s[axis] = Some(t_s);
        }
    }
}

fn print_text(path: &Path, specs: &[MetricSpec], stats: &[MetricStats], ready: ReadyStats) {
    println!("file={}", path.display());
    println!(
        "rows total={} both_ready={} reduced_ready={} full_ready={}",
        ready.total, ready.both_ready, ready.reduced_ready, ready.full_ready
    );
    for (spec, stat) in specs.iter().zip(stats) {
        if stat.count == 0 {
            println!("{} count=0", spec.label);
            continue;
        }
        let rms = [
            (stat.sum_sq[0] / stat.count as f64).sqrt(),
            (stat.sum_sq[1] / stat.count as f64).sqrt(),
            (stat.sum_sq[2] / stat.count as f64).sqrt(),
        ];
        println!(
            "{} count={} rms=[{:.6},{:.6},{:.6}] max_abs=[{:.6}@{:.2},{:.6}@{:.2},{:.6}@{:.2}] final@{:.2}=[{:.6},{:.6},{:.6}] first_cross>{:.4}=[{},{},{}]",
            spec.label,
            stat.count,
            rms[0],
            rms[1],
            rms[2],
            stat.max_abs[0],
            stat.max_time_s[0],
            stat.max_abs[1],
            stat.max_time_s[1],
            stat.max_abs[2],
            stat.max_time_s[2],
            stat.final_time_s,
            stat.final_value[0],
            stat.final_value[1],
            stat.final_value[2],
            spec.threshold,
            fmt_opt_time(stat.first_cross_time_s[0]),
            fmt_opt_time(stat.first_cross_time_s[1]),
            fmt_opt_time(stat.first_cross_time_s[2]),
        );
    }
}

fn print_markdown(path: &Path, specs: &[MetricSpec], stats: &[MetricStats], ready: ReadyStats) {
    println!("### {}", path.display());
    println!();
    println!(
        "`total={}` `both_ready={}` `reduced_ready={}` `full_ready={}`",
        ready.total, ready.both_ready, ready.reduced_ready, ready.full_ready
    );
    println!();
    println!(
        "| Metric | Count | RMS axis 0/1/2 | Max abs axis 0/1/2 | Final axis 0/1/2 | First threshold crossing |"
    );
    println!("| --- | ---: | --- | --- | --- | --- |");
    for (spec, stat) in specs.iter().zip(stats) {
        if stat.count == 0 {
            println!("| {} | 0 | n/a | n/a | n/a | n/a |", spec.label);
            continue;
        }
        let rms = [
            (stat.sum_sq[0] / stat.count as f64).sqrt(),
            (stat.sum_sq[1] / stat.count as f64).sqrt(),
            (stat.sum_sq[2] / stat.count as f64).sqrt(),
        ];
        println!(
            "| {} | {} | `{:.4}, {:.4}, {:.4}` | `{:.4}@{:.1}, {:.4}@{:.1}, {:.4}@{:.1}` | `{:.4}, {:.4}, {:.4}` | `>{:.4}: {}, {}, {}` |",
            spec.label,
            stat.count,
            rms[0],
            rms[1],
            rms[2],
            stat.max_abs[0],
            stat.max_time_s[0],
            stat.max_abs[1],
            stat.max_time_s[1],
            stat.max_abs[2],
            stat.max_time_s[2],
            stat.final_value[0],
            stat.final_value[1],
            stat.final_value[2],
            spec.threshold,
            fmt_opt_time(stat.first_cross_time_s[0]),
            fmt_opt_time(stat.first_cross_time_s[1]),
            fmt_opt_time(stat.first_cross_time_s[2]),
        );
    }
    println!();
}

fn required_index(index: &HashMap<&str, usize>, name: &str) -> Result<usize> {
    index
        .get(name)
        .copied()
        .with_context(|| format!("missing required column {name}"))
}

fn parse_cell(value: &str) -> f64 {
    value.parse::<f64>().unwrap_or(f64::NAN)
}

fn parse_bool(value: &str) -> bool {
    value.eq_ignore_ascii_case("true") || value == "1"
}

fn fmt_opt_time(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.2}"))
        .unwrap_or_else(|| "-".to_string())
}
