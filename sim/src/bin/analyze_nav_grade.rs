use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;

use sim::ubxlog::{extract_nav_att, extract_nav_pvt_obs, parse_ubx_frames};

#[derive(Parser, Debug)]
#[command(name = "analyze_nav_grade")]
struct Args {
    #[arg(value_name = "UBX_LOG")]
    log_path: PathBuf,

    #[arg(long, default_value_t = 10.0)]
    window_s: f64,

    #[arg(long, default_value_t = 50.0)]
    min_horiz_m: f64,
}

#[derive(Clone, Copy, Debug)]
struct NavSample {
    t_s: f64,
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
}

#[derive(Clone, Copy, Debug)]
struct AttSample {
    t_s: f64,
    pitch_deg: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let data = fs::read(&args.log_path)
        .with_context(|| format!("failed to read {}", args.log_path.display()))?;
    let frames = parse_ubx_frames(&data, None);

    let mut nav = Vec::<NavSample>::new();
    let mut att = Vec::<AttSample>::new();
    for f in &frames {
        if let Some(obs) = extract_nav_pvt_obs(f) {
            nav.push(NavSample {
                t_s: obs.itow_ms as f64 * 1.0e-3,
                lat_deg: obs.lat_deg,
                lon_deg: obs.lon_deg,
                height_m: obs.height_m,
            });
        }
        if let Some((itow_ms, _roll_deg, pitch_deg, _heading_deg)) = extract_nav_att(f) {
            att.push(AttSample {
                t_s: itow_ms as f64 * 1.0e-3,
                pitch_deg,
            });
        }
    }
    if nav.len() < 3 || att.is_empty() {
        bail!("need NAV-PVT and NAV-ATT samples");
    }

    let anchor = nav[0];
    let nav_local: Vec<(f64, f64, f64, f64)> = nav
        .iter()
        .map(|s| {
            let (n_m, e_m) = local_ne(anchor.lat_deg, anchor.lon_deg, s.lat_deg, s.lon_deg);
            (s.t_s, n_m, e_m, s.height_m)
        })
        .collect();

    let mut grades = Vec::<(f64, f64)>::new();
    let mut j0 = 0usize;
    for i in 0..nav_local.len() {
        let ti = nav_local[i].0;
        while j0 + 1 < i && nav_local[j0 + 1].0 < ti - args.window_s {
            j0 += 1;
        }
        let dt = ti - nav_local[j0].0;
        if dt < args.window_s * 0.5 {
            continue;
        }
        let dn = nav_local[i].1 - nav_local[j0].1;
        let de = nav_local[i].2 - nav_local[j0].2;
        let dh = nav_local[i].3 - nav_local[j0].3;
        let horiz = (dn * dn + de * de).sqrt();
        if horiz < args.min_horiz_m {
            continue;
        }
        let grade_deg = dh.atan2(horiz).to_degrees();
        grades.push((ti, grade_deg));
    }

    if grades.is_empty() {
        bail!("no valid grade samples");
    }

    let att_interp: Vec<(f64, f64, f64)> = grades
        .iter()
        .filter_map(|(t_s, grade_deg)| {
            interpolate_pitch(&att, *t_s).map(|pitch_deg| (*t_s, *grade_deg, pitch_deg))
        })
        .collect();
    if att_interp.is_empty() {
        bail!("could not align NAV-ATT to grade samples");
    }

    let abs_grade_mean = mean(att_interp.iter().map(|x| x.1.abs()));
    let abs_pitch_mean = mean(att_interp.iter().map(|x| x.2.abs()));
    let abs_diff_mean = mean(att_interp.iter().map(|x| (x.2 - x.1).abs()));
    let max_abs_grade = att_interp.iter().map(|x| x.1.abs()).fold(0.0_f64, f64::max);
    let max_abs_pitch = att_interp.iter().map(|x| x.2.abs()).fold(0.0_f64, f64::max);

    println!(
        "window_s={:.1} min_horiz_m={:.1} n={} mean_abs_grade_deg={:.3} mean_abs_nav_att_pitch_deg={:.3} mean_abs_pitch_minus_grade_deg={:.3} max_abs_grade_deg={:.3} max_abs_nav_att_pitch_deg={:.3}",
        args.window_s,
        args.min_horiz_m,
        att_interp.len(),
        abs_grade_mean,
        abs_pitch_mean,
        abs_diff_mean,
        max_abs_grade,
        max_abs_pitch
    );

    let mut strongest_grade = att_interp.clone();
    strongest_grade.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    println!("\nstrongest grade windows:");
    for (t_s, grade_deg, pitch_deg) in strongest_grade.into_iter().take(12) {
        println!(
            "  t={:8.3}s grade={:+6.3} deg nav_att_pitch={:+6.3} deg diff={:+6.3} deg",
            t_s,
            grade_deg,
            pitch_deg,
            pitch_deg - grade_deg
        );
    }

    let mut strongest_pitch = att_interp.clone();
    strongest_pitch.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
    println!("\nstrongest NAV-ATT pitch windows:");
    for (t_s, grade_deg, pitch_deg) in strongest_pitch.into_iter().take(12) {
        println!(
            "  t={:8.3}s nav_att_pitch={:+6.3} deg grade={:+6.3} deg diff={:+6.3} deg",
            t_s,
            pitch_deg,
            grade_deg,
            pitch_deg - grade_deg
        );
    }

    Ok(())
}

fn mean<I>(vals: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut n = 0usize;
    for v in vals {
        sum += v;
        n += 1;
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn interpolate_pitch(att: &[AttSample], t_s: f64) -> Option<f64> {
    let idx = att.partition_point(|s| s.t_s < t_s);
    if idx == 0 || idx >= att.len() {
        return None;
    }
    let a0 = att[idx - 1];
    let a1 = att[idx];
    let dt = a1.t_s - a0.t_s;
    if dt <= 1.0e-6 {
        return Some(a1.pitch_deg);
    }
    let u = ((t_s - a0.t_s) / dt).clamp(0.0, 1.0);
    Some(a0.pitch_deg + u * (a1.pitch_deg - a0.pitch_deg))
}

fn local_ne(lat0_deg: f64, lon0_deg: f64, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
    let r_earth_m = 6_378_137.0_f64;
    let lat0 = lat0_deg.to_radians();
    let dlat = (lat_deg - lat0_deg).to_radians();
    let dlon = (lon_deg - lon0_deg).to_radians();
    let north = dlat * r_earth_m;
    let east = dlon * r_earth_m * lat0.cos();
    (north, east)
}
