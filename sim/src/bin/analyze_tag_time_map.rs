use std::{collections::BTreeMap, fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::ubxlog::{
    extract_esf_raw_samples, fit_linear_map, parse_ubx_frames, sensor_meta, unwrap_counter,
};
use sim::visualizer::math::nearest_master_ms;
use sim::visualizer::pipeline::timebase::build_master_timeline;

#[derive(Parser, Debug)]
#[command(name = "analyze_tag_time_map")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long)]
    max_records: Option<usize>,
    #[arg(long, default_value_t = 60.0)]
    window_s: f64,
}

#[derive(Clone, Copy)]
struct TagAnchor {
    rel_s: f64,
    tag_u: u64,
    seq_ms: f64,
    fit_ms: f64,
    residual_ms: f64,
}

#[derive(Default)]
struct Stats {
    n: usize,
    sum: f64,
    sum_abs: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl Stats {
    fn push(&mut self, v: f64) {
        if self.n == 0 {
            self.min = v;
            self.max = v;
        }
        self.n += 1;
        self.sum += v;
        self.sum_abs += v.abs();
        self.sum_sq += v * v;
        self.min = self.min.min(v);
        self.max = self.max.max(v);
    }

    fn mean(&self) -> f64 {
        self.sum / self.n as f64
    }

    fn rms(&self) -> f64 {
        (self.sum_sq / self.n as f64).sqrt()
    }
}

fn windowed_local_fit(anchors: &[TagAnchor], start_s: f64, end_s: f64) -> Option<(f64, f64, usize)> {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for a in anchors {
        if a.rel_s >= start_s && a.rel_s < end_s {
            x.push(a.tag_u as f64);
            y.push(a.seq_ms);
        }
    }
    if x.len() < 10 {
        return None;
    }
    let (a, b) = fit_linear_map(&x, &y, 1.0e-3);
    Some((a, b, x.len()))
}

fn fit_tag_ms_map_local(
    seqs: &[u64],
    tags: &[u64],
    masters: &[(u64, f64)],
    unwrap_modulus: Option<u64>,
) -> (Vec<u64>, f64, f64) {
    let mapped_tags = match unwrap_modulus {
        Some(m) => unwrap_counter(tags, m),
        None => tags.to_vec(),
    };
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for (seq, tag_u) in seqs.iter().zip(mapped_tags.iter()) {
        if let Some(ms) = nearest_master_ms(*seq, masters) {
            x.push(*tag_u as f64);
            y.push(ms);
        }
    }
    let (a, b) = fit_linear_map(&x, &y, 1e-3);
    (mapped_tags, a, b)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut bytes = Vec::new();
    File::open(&args.logfile)
        .with_context(|| format!("failed to open {}", args.logfile.display()))?
        .read_to_end(&mut bytes)
        .context("failed to read log")?;

    let frames = parse_ubx_frames(&bytes, args.max_records);
    let tl = build_master_timeline(&frames);
    anyhow::ensure!(!tl.masters.is_empty(), "no master timeline");
    let t0_ms = tl.t0_master_ms;

    let mut raw_seq = Vec::<u64>::new();
    let mut raw_tag = Vec::<u64>::new();
    let mut raw_dtype = Vec::<u8>::new();
    for f in &frames {
        for (tag, sw) in extract_esf_raw_samples(f) {
            let (name, _, _) = sensor_meta(sw.dtype);
            if name == "gyro_x" {
                raw_seq.push(f.seq);
                raw_tag.push(tag);
                raw_dtype.push(sw.dtype);
            }
        }
    }

    let (tag_u, a_global, b_global) =
        fit_tag_ms_map_local(&raw_seq, &raw_tag, &tl.masters, Some(1 << 16));
    let _tag_u_manual = unwrap_counter(&raw_tag, 1 << 16);

    let mut anchors = Vec::<TagAnchor>::new();
    for (seq, tag) in raw_seq.iter().zip(tag_u.iter()) {
        let Some(seq_ms) = nearest_master_ms(*seq, &tl.masters) else {
            continue;
        };
        let fit_ms = a_global * (*tag as f64) + b_global;
        anchors.push(TagAnchor {
            rel_s: (seq_ms - t0_ms) * 1.0e-3,
            tag_u: *tag,
            seq_ms,
            fit_ms,
            residual_ms: fit_ms - seq_ms,
        });
    }

    let mut overall = Stats::default();
    for a in &anchors {
        overall.push(a.residual_ms);
    }

    println!(
        "global_fit: slope_ms_per_tick={:.9} intercept_ms={:.3} anchors={} residual_mean_ms={:.3} residual_mean_abs_ms={:.3} residual_rms_ms={:.3} residual_min_ms={:.3} residual_max_ms={:.3}",
        a_global,
        b_global,
        anchors.len(),
        overall.mean(),
        overall.sum_abs / overall.n as f64,
        overall.rms(),
        overall.min,
        overall.max
    );

    let mut windows = BTreeMap::<i64, Stats>::new();
    for a in &anchors {
        let key = (a.rel_s / args.window_s).floor() as i64;
        windows.entry(key).or_default().push(a.residual_ms);
    }

    println!("window_residuals:");
    for (k, stats) in &windows {
        let start_s = *k as f64 * args.window_s;
        let end_s = start_s + args.window_s;
        if let Some((a_local, b_local, n_local)) = windowed_local_fit(&anchors, start_s, end_s) {
            let center_rel_s = 0.5 * (start_s + end_s);
            let center_anchor = anchors
                .iter()
                .filter(|x| x.rel_s >= start_s && x.rel_s < end_s)
                .nth(n_local / 2)
                .copied();
            let local_minus_global_center_ms = center_anchor
                .map(|c| (a_local * c.tag_u as f64 + b_local) - (a_global * c.tag_u as f64 + b_global))
                .unwrap_or(0.0);
            println!(
                "  [{:7.1},{:7.1})s n={} residual_mean_ms={:+8.3} mean_abs_ms={:8.3} rms_ms={:8.3} min_ms={:+8.3} max_ms={:+8.3} local_slope_ms_per_tick={:.9} local_minus_global_center_ms={:+8.3}",
                start_s,
                end_s,
                stats.n,
                stats.mean(),
                stats.sum_abs / stats.n as f64,
                stats.rms(),
                stats.min,
                stats.max,
                a_local,
                local_minus_global_center_ms,
            );
            let _ = center_rel_s;
        }
    }

    for &(start_s, end_s) in &[(60.0, 120.0), (330.0, 390.0), (390.0, 450.0), (450.0, 510.0), (510.0, 570.0), (1260.0, 1320.0)] {
        if let Some((a_local, b_local, n_local)) = windowed_local_fit(&anchors, start_s, end_s) {
            let subset: Vec<_> = anchors
                .iter()
                .filter(|x| x.rel_s >= start_s && x.rel_s < end_s)
                .copied()
                .collect();
            let mut stats = Stats::default();
            for a in &subset {
                stats.push(a.residual_ms);
            }
            let center = subset[subset.len() / 2];
            let local_minus_global_center_ms =
                (a_local * center.tag_u as f64 + b_local) - (a_global * center.tag_u as f64 + b_global);
            println!(
                "focus_window [{:7.1},{:7.1})s n={} residual_mean_ms={:+8.3} mean_abs_ms={:8.3} local_slope_ms_per_tick={:.9} local_minus_global_center_ms={:+8.3}",
                start_s,
                end_s,
                n_local,
                stats.mean(),
                stats.sum_abs / stats.n as f64,
                a_local,
                local_minus_global_center_ms,
            );
        }
    }

    let mut extremes = anchors.clone();
    extremes.sort_by(|a, b| b.residual_ms.abs().total_cmp(&a.residual_ms.abs()));
    println!("largest_residuals:");
    for a in extremes.into_iter().take(10) {
        println!(
            "  t={:8.3}s tag={} seq_ms={:.3} fit_ms={:.3} residual_ms={:+8.3}",
            a.rel_s, a.tag_u, a.seq_ms, a.fit_ms, a.residual_ms
        );
    }

    Ok(())
}
