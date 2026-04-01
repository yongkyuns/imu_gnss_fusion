use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use sim::ubxlog::parse_ubx_frames;
use sim::visualizer::model::EkfImuSource;
use sim::visualizer::pipeline::ekf_compare::{
    EkfCompareConfig, GnssOutageConfig, build_ekf_compare_traces,
};
use sim::visualizer::pipeline::timebase::build_master_timeline;

#[derive(Parser, Debug)]
#[command(name = "compare_ekf_imu_rate")]
struct Args {
    #[arg(long, default_value = "test_files.txt")]
    file_list: PathBuf,

    #[arg(long, default_value = "Baseline")]
    section: String,

    #[arg(long, default_value = "logger/data")]
    data_dir: PathBuf,

    #[arg(long, default_value = "align", value_parser = parse_ekf_imu_source)]
    ekf_imu_source: EkfImuSource,

    #[arg(long, default_value_t = 3)]
    decimation: usize,

    #[arg(long, default_value_t = 15.0)]
    predict_lpf_cutoff_hz: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct TraceDiffMetrics {
    rms: f64,
    final_abs: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ErrorMetrics {
    pos_rms_m: f64,
    vel_rms_mps: f64,
    att_rms_deg: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let files = parse_section_files(&args.file_list, &args.section)?;
    if files.is_empty() {
        bail!("no files found in section {}", args.section);
    }

    let full_cfg = EkfCompareConfig::default();
    let mut decimated_cfg = full_cfg;
    decimated_cfg.predict_imu_decimation = args.decimation.max(1);
    decimated_cfg.predict_imu_lpf_cutoff_hz = Some(args.predict_lpf_cutoff_hz);

    println!(
        "file,full_pos_rms_m,dec_pos_rms_m,d_pos_rms_m,full_vel_rms_mps,dec_vel_rms_mps,d_vel_rms_mps,full_att_rms_deg,dec_att_rms_deg,d_att_rms_deg,state_pos_rms_m,state_vel_rms_mps,state_att_rms_deg,state_cov_rms,final_pos_delta_m,final_vel_delta_mps,final_att_delta_deg,final_cov_delta"
    );

    let mut worst_pos_reg_file = String::new();
    let mut worst_pos_reg = f64::NEG_INFINITY;
    let mut worst_state_pos_rms = 0.0_f64;
    let mut worst_state_att_rms = 0.0_f64;
    let mut worst_cov_rms = 0.0_f64;

    for file in files {
        let path = args.data_dir.join(&file);
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let frames = parse_ubx_frames(&bytes, None);
        let tl = build_master_timeline(&frames);

        let full = build_ekf_compare_traces(
            &frames,
            &tl,
            args.ekf_imu_source,
            full_cfg,
            GnssOutageConfig::default(),
        );
        let dec = build_ekf_compare_traces(
            &frames,
            &tl,
            args.ekf_imu_source,
            decimated_cfg,
            GnssOutageConfig::default(),
        );

        let full_err = compute_error_metrics(&full)?;
        let dec_err = compute_error_metrics(&dec)?;
        let state_pos = rms_vec3_diff(
            find_trace(&full.cmp_pos, "EKF posN [m]")?,
            find_trace(&full.cmp_pos, "EKF posE [m]")?,
            find_trace(&full.cmp_pos, "EKF posD [m]")?,
            find_trace(&dec.cmp_pos, "EKF posN [m]")?,
            find_trace(&dec.cmp_pos, "EKF posE [m]")?,
            find_trace(&dec.cmp_pos, "EKF posD [m]")?,
        );
        let state_vel = rms_vec3_diff(
            find_trace(&full.cmp_vel, "EKF velN [m/s]")?,
            find_trace(&full.cmp_vel, "EKF velE [m/s]")?,
            find_trace(&full.cmp_vel, "EKF velD [m/s]")?,
            find_trace(&dec.cmp_vel, "EKF velN [m/s]")?,
            find_trace(&dec.cmp_vel, "EKF velE [m/s]")?,
            find_trace(&dec.cmp_vel, "EKF velD [m/s]")?,
        );
        let state_att = rms_att_diff(
            find_trace(&full.cmp_att, "EKF roll [deg]")?,
            find_trace(&full.cmp_att, "EKF pitch [deg]")?,
            find_trace(&full.cmp_att, "EKF yaw [deg]")?,
            find_trace(&dec.cmp_att, "EKF roll [deg]")?,
            find_trace(&dec.cmp_att, "EKF pitch [deg]")?,
            find_trace(&dec.cmp_att, "EKF yaw [deg]")?,
        );
        let state_cov = rms_cov_diff(&full, &dec);

        let pos_reg = dec_err.pos_rms_m - full_err.pos_rms_m;
        if pos_reg > worst_pos_reg {
            worst_pos_reg = pos_reg;
            worst_pos_reg_file = file.clone();
        }
        worst_state_pos_rms = worst_state_pos_rms.max(state_pos.rms);
        worst_state_att_rms = worst_state_att_rms.max(state_att.rms);
        worst_cov_rms = worst_cov_rms.max(state_cov.rms);

        println!(
            "{file},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            full_err.pos_rms_m,
            dec_err.pos_rms_m,
            dec_err.pos_rms_m - full_err.pos_rms_m,
            full_err.vel_rms_mps,
            dec_err.vel_rms_mps,
            dec_err.vel_rms_mps - full_err.vel_rms_mps,
            full_err.att_rms_deg,
            dec_err.att_rms_deg,
            dec_err.att_rms_deg - full_err.att_rms_deg,
            state_pos.rms,
            state_vel.rms,
            state_att.rms,
            state_cov.rms,
            state_pos.final_abs,
            state_vel.final_abs,
            state_att.final_abs,
            state_cov.final_abs,
        );
    }

    println!();
    println!(
        "summary: decimation={} predict_lpf_cutoff_hz={:.1}",
        decimated_cfg.predict_imu_decimation,
        decimated_cfg.predict_imu_lpf_cutoff_hz.unwrap_or(0.0)
    );
    println!(
        "worst position RMS regression vs GNSS: {} ({:.3} m)",
        worst_pos_reg_file, worst_pos_reg
    );
    println!(
        "worst EKF state position RMS delta: {:.3} m",
        worst_state_pos_rms
    );
    println!(
        "worst EKF state attitude RMS delta: {:.3} deg",
        worst_state_att_rms
    );
    println!("worst covariance RMS delta: {:.6}", worst_cov_rms);

    Ok(())
}

fn compute_error_metrics(
    data: &sim::visualizer::pipeline::ekf_compare::EkfCompareData,
) -> Result<ErrorMetrics> {
    let ekf_pn = find_trace(&data.cmp_pos, "EKF posN [m]")?;
    let ekf_pe = find_trace(&data.cmp_pos, "EKF posE [m]")?;
    let ekf_pd = find_trace(&data.cmp_pos, "EKF posD [m]")?;
    let ubx_pn = find_trace(&data.cmp_pos, "UBX posN [m]")?;
    let ubx_pe = find_trace(&data.cmp_pos, "UBX posE [m]")?;
    let ubx_pd = find_trace(&data.cmp_pos, "UBX posD [m]")?;

    let ekf_vn = find_trace(&data.cmp_vel, "EKF velN [m/s]")?;
    let ekf_ve = find_trace(&data.cmp_vel, "EKF velE [m/s]")?;
    let ekf_vd = find_trace(&data.cmp_vel, "EKF velD [m/s]")?;
    let ubx_vn = find_trace(&data.cmp_vel, "UBX velN [m/s]")?;
    let ubx_ve = find_trace(&data.cmp_vel, "UBX velE [m/s]")?;
    let ubx_vd = find_trace(&data.cmp_vel, "UBX velD [m/s]")?;

    let ekf_roll = find_trace(&data.cmp_att, "EKF roll [deg]")?;
    let ekf_pitch = find_trace(&data.cmp_att, "EKF pitch [deg]")?;
    let ekf_yaw = find_trace(&data.cmp_att, "EKF yaw [deg]")?;
    let nav_roll = find_trace(&data.cmp_att, "NAV-ATT roll [deg]")?;
    let nav_pitch = find_trace(&data.cmp_att, "NAV-ATT pitch [deg]")?;
    let nav_yaw = find_trace(&data.cmp_att, "NAV-ATT heading [deg]")?;

    let mut n = 0usize;
    let mut sum_pos2 = 0.0;
    let mut sum_vel2 = 0.0;
    let mut sum_att2 = 0.0;

    for p in ekf_pn {
        let t = p[0];
        let Some(epn) = interp(ekf_pn, t) else {
            continue;
        };
        let Some(epe) = interp(ekf_pe, t) else {
            continue;
        };
        let Some(epd) = interp(ekf_pd, t) else {
            continue;
        };
        let Some(upn) = interp(ubx_pn, t) else {
            continue;
        };
        let Some(upe) = interp(ubx_pe, t) else {
            continue;
        };
        let Some(upd) = interp(ubx_pd, t) else {
            continue;
        };
        let Some(evn) = interp(ekf_vn, t) else {
            continue;
        };
        let Some(eve) = interp(ekf_ve, t) else {
            continue;
        };
        let Some(evd) = interp(ekf_vd, t) else {
            continue;
        };
        let Some(uvn) = interp(ubx_vn, t) else {
            continue;
        };
        let Some(uve) = interp(ubx_ve, t) else {
            continue;
        };
        let Some(uvd) = interp(ubx_vd, t) else {
            continue;
        };
        let Some(er) = interp(ekf_roll, t) else {
            continue;
        };
        let Some(ep) = interp(ekf_pitch, t) else {
            continue;
        };
        let Some(ey) = interp(ekf_yaw, t) else {
            continue;
        };
        let Some(nr) = interp(nav_roll, t) else {
            continue;
        };
        let Some(np) = interp(nav_pitch, t) else {
            continue;
        };
        let Some(ny) = interp(nav_yaw, t) else {
            continue;
        };

        sum_pos2 += (epn - upn).powi(2) + (epe - upe).powi(2) + (epd - upd).powi(2);
        sum_vel2 += (evn - uvn).powi(2) + (eve - uve).powi(2) + (evd - uvd).powi(2);
        sum_att2 += (er - nr).powi(2) + (ep - np).powi(2) + wrap_deg(ey - ny).powi(2);
        n += 1;
    }

    if n == 0 {
        bail!("no overlapping EKF/reference samples");
    }

    Ok(ErrorMetrics {
        pos_rms_m: (sum_pos2 / n as f64).sqrt(),
        vel_rms_mps: (sum_vel2 / n as f64).sqrt(),
        att_rms_deg: (sum_att2 / n as f64).sqrt(),
    })
}

fn rms_vec3_diff(
    ax: &[[f64; 2]],
    ay: &[[f64; 2]],
    az: &[[f64; 2]],
    bx: &[[f64; 2]],
    by: &[[f64; 2]],
    bz: &[[f64; 2]],
) -> TraceDiffMetrics {
    let mut n = 0usize;
    let mut sum2 = 0.0;
    let mut final_abs = 0.0;
    for p in ax {
        let t = p[0];
        let Some(ax_t) = interp(ax, t) else { continue };
        let Some(ay_t) = interp(ay, t) else { continue };
        let Some(az_t) = interp(az, t) else { continue };
        let Some(bx_t) = interp(bx, t) else { continue };
        let Some(by_t) = interp(by, t) else { continue };
        let Some(bz_t) = interp(bz, t) else { continue };
        let err = ((ax_t - bx_t).powi(2) + (ay_t - by_t).powi(2) + (az_t - bz_t).powi(2)).sqrt();
        sum2 += err * err;
        final_abs = err;
        n += 1;
    }
    TraceDiffMetrics {
        rms: if n > 0 {
            (sum2 / n as f64).sqrt()
        } else {
            f64::NAN
        },
        final_abs,
    }
}

fn rms_att_diff(
    ar: &[[f64; 2]],
    ap: &[[f64; 2]],
    ay: &[[f64; 2]],
    br: &[[f64; 2]],
    bp: &[[f64; 2]],
    by: &[[f64; 2]],
) -> TraceDiffMetrics {
    let mut n = 0usize;
    let mut sum2 = 0.0;
    let mut final_abs = 0.0;
    for p in ar {
        let t = p[0];
        let Some(ar_t) = interp(ar, t) else { continue };
        let Some(ap_t) = interp(ap, t) else { continue };
        let Some(ay_t) = interp(ay, t) else { continue };
        let Some(br_t) = interp(br, t) else { continue };
        let Some(bp_t) = interp(bp, t) else { continue };
        let Some(by_t) = interp(by, t) else { continue };
        let err =
            ((ar_t - br_t).powi(2) + (ap_t - bp_t).powi(2) + wrap_deg(ay_t - by_t).powi(2)).sqrt();
        sum2 += err * err;
        final_abs = err;
        n += 1;
    }
    TraceDiffMetrics {
        rms: if n > 0 {
            (sum2 / n as f64).sqrt()
        } else {
            f64::NAN
        },
        final_abs,
    }
}

fn rms_cov_diff(
    full: &sim::visualizer::pipeline::ekf_compare::EkfCompareData,
    dec: &sim::visualizer::pipeline::ekf_compare::EkfCompareData,
) -> TraceDiffMetrics {
    let mut n = 0usize;
    let mut sum2 = 0.0;
    let mut final_abs = 0.0;
    for (a, b) in full.cov_nonbias.iter().zip(dec.cov_nonbias.iter()) {
        for p in &a.points {
            let t = p[0];
            let Some(av) = interp(&a.points, t) else {
                continue;
            };
            let Some(bv) = interp(&b.points, t) else {
                continue;
            };
            let err = (av - bv).abs();
            sum2 += err * err;
            final_abs = err;
            n += 1;
        }
    }
    TraceDiffMetrics {
        rms: if n > 0 {
            (sum2 / n as f64).sqrt()
        } else {
            f64::NAN
        },
        final_abs,
    }
}

fn find_trace<'a>(
    traces: &'a [sim::visualizer::model::Trace],
    name: &str,
) -> Result<&'a [[f64; 2]]> {
    traces
        .iter()
        .find(|t| t.name == name)
        .map(|t| t.points.as_slice())
        .with_context(|| format!("missing trace {name}"))
}

fn interp(points: &[[f64; 2]], t: f64) -> Option<f64> {
    if points.is_empty() || t < points[0][0] || t > points[points.len() - 1][0] {
        return None;
    }
    match points.binary_search_by(|p| p[0].partial_cmp(&t).unwrap()) {
        Ok(i) => Some(points[i][1]),
        Err(i) => {
            if i == 0 || i >= points.len() {
                return None;
            }
            let p0 = points[i - 1];
            let p1 = points[i];
            let dt = p1[0] - p0[0];
            if dt.abs() < 1.0e-9 {
                Some(p0[1])
            } else {
                let alpha = ((t - p0[0]) / dt).clamp(0.0, 1.0);
                Some(p0[1] + alpha * (p1[1] - p0[1]))
            }
        }
    }
}

fn wrap_deg(mut d: f64) -> f64 {
    while d > 180.0 {
        d -= 360.0;
    }
    while d < -180.0 {
        d += 360.0;
    }
    d
}

fn parse_section_files(path: &Path, wanted: &str) -> Result<Vec<String>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read file list {}", path.display()))?;
    let mut out = Vec::new();
    let mut in_section = false;
    for line in text.lines() {
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        if s.ends_with(':') {
            in_section = s.trim_end_matches(':') == wanted;
            continue;
        }
        if s.starts_with('[') && s.ends_with(']') {
            in_section = &s[1..s.len() - 1] == wanted;
            continue;
        }
        if in_section {
            let file = s.split_whitespace().next().unwrap_or("");
            if file.ends_with(".bin") {
                out.push(file.to_string());
            }
        }
    }
    Ok(out)
}

fn parse_ekf_imu_source(s: &str) -> Result<EkfImuSource, String> {
    match s.to_ascii_lowercase().as_str() {
        "align" => Ok(EkfImuSource::Align),
        "esf-alg" => Ok(EkfImuSource::EsfAlg),
        other => Err(format!("invalid ekf imu source: {other}")),
    }
}
