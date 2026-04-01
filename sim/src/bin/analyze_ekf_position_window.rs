use std::{fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use sim::ubxlog::{extract_nav_pvt_obs, extract_nav2_pvt_obs, parse_ubx_frames};
use sim::visualizer::math::{ecef_to_ned, lla_to_ecef, nearest_master_ms};
use sim::visualizer::model::EkfImuSource;
use sim::visualizer::pipeline::ekf_compare::{
    EkfCompareConfig, GnssOutageConfig, build_ekf_compare_traces,
};
use sim::visualizer::pipeline::timebase::build_master_timeline;

#[derive(Parser, Debug)]
#[command(name = "analyze_ekf_position_window")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long, default_value_t = 693.0)]
    window_start_s: f64,
    #[arg(long, default_value_t = 784.0)]
    window_end_s: f64,
    #[arg(long, default_value = "align", value_parser = parse_ekf_imu_source)]
    ekf_imu_source: EkfImuSource,
    #[arg(long)]
    no_sweep: bool,
}

#[derive(Clone, Copy, Debug)]
struct WindowMetrics {
    n: usize,
    mean_2d_m: f64,
    rms_2d_m: f64,
    mean_abs_along_m: f64,
    rms_along_m: f64,
    mean_abs_cross_m: f64,
    rms_cross_m: f64,
    max_abs_cross_m: f64,
    mean_heading_deg: f64,
}

#[derive(Clone, Copy, Debug)]
struct LagMetrics {
    lag_s: f64,
    mean_2d_m: f64,
    rms_2d_m: f64,
    mean_abs_along_m: f64,
    mean_abs_cross_m: f64,
}

#[derive(Clone, Copy, Debug)]
struct PolylineMetrics {
    n: usize,
    mean_nearest_m: f64,
    rms_nearest_m: f64,
    max_nearest_m: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let frames = load_frames(&args.logfile)?;
    let tl = build_master_timeline(&frames);

    let base_cfg = EkfCompareConfig::default();
    eprintln!("baseline config: {:?}", base_cfg);

    let baseline = run_case(
        "baseline",
        &frames,
        &tl,
        args.ekf_imu_source,
        base_cfg,
        args.window_start_s,
        args.window_end_s,
    )?;
    print_case("baseline", &baseline.0, &baseline.1);
    print_nav_alignment(
        &frames,
        &tl,
        args.ekf_imu_source,
        base_cfg,
        args.window_start_s,
        args.window_end_s,
    )?;

    if !args.no_sweep {
        println!();
        println!("parameter sweeps:");
        sweep_r_body_vel(&frames, &tl, &args, base_cfg)?;
        sweep_pos_scale(&frames, &tl, &args, base_cfg)?;
        sweep_vel_scale(&frames, &tl, &args, base_cfg)?;
        sweep_accel_var(&frames, &tl, &args, base_cfg)?;
        sweep_gyro_var(&frames, &tl, &args, base_cfg)?;
    }

    Ok(())
}

fn print_nav_alignment(
    frames: &[sim::ubxlog::UbxFrame],
    tl: &sim::visualizer::pipeline::timebase::MasterTimeline,
    imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    t0_s: f64,
    t1_s: f64,
) -> Result<()> {
    let mut nav_pvt = Vec::<[f64; 5]>::new();
    let mut nav2_pvt = Vec::<[f64; 5]>::new();
    for f in frames {
        let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters) else {
            continue;
        };
        let t_s = (t_ms - tl.t0_master_ms) * 1.0e-3;
        if let Some(obs) = extract_nav2_pvt_obs(f)
            && obs.fix_ok
            && !obs.invalid_llh
        {
            nav2_pvt.push([t_s, obs.lat_deg, obs.lon_deg, obs.vel_n_mps, obs.vel_e_mps]);
        } else if let Some(obs) = extract_nav_pvt_obs(f)
            && obs.fix_ok
            && !obs.invalid_llh
        {
            nav_pvt.push([t_s, obs.lat_deg, obs.lon_deg, obs.vel_n_mps, obs.vel_e_mps]);
        }
    }
    if nav_pvt.is_empty() || nav2_pvt.is_empty() {
        println!("NAV-PVT vs NAV2-PVT: unavailable");
        return Ok(());
    }

    let ref_lat = nav2_pvt[0][1];
    let ref_lon = nav2_pvt[0][2];
    let ref_ecef = lla_to_ecef(ref_lat, ref_lon, 0.0);
    let nav_pvt_ne = nav_pvt
        .iter()
        .map(|p| {
            let ecef = lla_to_ecef(p[1], p[2], 0.0);
            let ned = ecef_to_ned(ecef, ref_ecef, ref_lat, ref_lon);
            [p[0], ned[0], ned[1], p[3], p[4]]
        })
        .collect::<Vec<_>>();
    let nav2_pvt_ne = nav2_pvt
        .iter()
        .map(|p| {
            let ecef = lla_to_ecef(p[1], p[2], 0.0);
            let ned = ecef_to_ned(ecef, ref_ecef, ref_lat, ref_lon);
            [p[0], ned[0], ned[1], p[3], p[4]]
        })
        .collect::<Vec<_>>();

    let n1 = nav_pvt_ne.iter().map(|p| [p[0], p[1]]).collect::<Vec<_>>();
    let e1 = nav_pvt_ne.iter().map(|p| [p[0], p[2]]).collect::<Vec<_>>();
    let n2 = nav2_pvt_ne.iter().map(|p| [p[0], p[1]]).collect::<Vec<_>>();
    let e2 = nav2_pvt_ne.iter().map(|p| [p[0], p[2]]).collect::<Vec<_>>();
    let vn2 = nav2_pvt_ne.iter().map(|p| [p[0], p[3]]).collect::<Vec<_>>();
    let ve2 = nav2_pvt_ne.iter().map(|p| [p[0], p[4]]).collect::<Vec<_>>();

    if let Some(m) = compute_window_metrics(&n1, &e1, &n2, &e2, &vn2, &ve2, t0_s, t1_s) {
        println!("NAV-PVT vs NAV2-PVT");
        println!(
            "{:>12}  n={:4}  mean2d={:6.2}  rms2d={:6.2}  along|={:6.2}  along_rms={:6.2}  cross|={:6.2}  cross_rms={:6.2}  cross_max={:6.2}  mean_hdg={:6.1}",
            "nav_diff",
            m.n,
            m.mean_2d_m,
            m.rms_2d_m,
            m.mean_abs_along_m,
            m.rms_along_m,
            m.mean_abs_cross_m,
            m.rms_cross_m,
            m.max_abs_cross_m,
            m.mean_heading_deg,
        );
    }

    let ekf_data =
        build_ekf_compare_traces(frames, tl, imu_source, ekf_cfg, GnssOutageConfig::default());
    let ekf_n = find_trace(&ekf_data.cmp_pos, "EKF posN [m]")?;
    let ekf_e = find_trace(&ekf_data.cmp_pos, "EKF posE [m]")?;
    if let Some(m) = compute_window_metrics(ekf_n, ekf_e, &n1, &e1, &vn2, &ve2, t0_s, t1_s) {
        println!("EKF vs NAV-PVT");
        println!(
            "{:>12}  n={:4}  mean2d={:6.2}  rms2d={:6.2}  along|={:6.2}  along_rms={:6.2}  cross|={:6.2}  cross_rms={:6.2}  cross_max={:6.2}  mean_hdg={:6.1}",
            "ekf_pvt",
            m.n,
            m.mean_2d_m,
            m.rms_2d_m,
            m.mean_abs_along_m,
            m.rms_along_m,
            m.mean_abs_cross_m,
            m.rms_cross_m,
            m.max_abs_cross_m,
            m.mean_heading_deg,
        );
    }

    if let Some(poly) = compute_polyline_metrics(ekf_n, ekf_e, &n2, &e2, t0_s, t1_s) {
        println!(
            "EKF -> NAV2 polyline   n={:4}  mean={:6.2} m  rms={:6.2} m  max={:6.2} m",
            poly.n, poly.mean_nearest_m, poly.rms_nearest_m, poly.max_nearest_m
        );
    }
    if let Some(poly) = compute_polyline_metrics(ekf_n, ekf_e, &n1, &e1, t0_s, t1_s) {
        println!(
            "EKF -> NAV-PVT polyline n={:4}  mean={:6.2} m  rms={:6.2} m  max={:6.2} m",
            poly.n, poly.mean_nearest_m, poly.rms_nearest_m, poly.max_nearest_m
        );
    }
    print_map_space_alignment(frames, tl, imu_source, ekf_cfg, t0_s, t1_s)?;

    Ok(())
}

fn print_map_space_alignment(
    frames: &[sim::ubxlog::UbxFrame],
    tl: &sim::visualizer::pipeline::timebase::MasterTimeline,
    imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    t0_s: f64,
    t1_s: f64,
) -> Result<()> {
    let data =
        build_ekf_compare_traces(frames, tl, imu_source, ekf_cfg, GnssOutageConfig::default());
    let _ekf_map = data
        .map
        .iter()
        .find(|t| t.name == "EKF path (lon,lat)")
        .context("missing EKF map trace")?;
    let _nav2_map = data
        .map
        .iter()
        .find(|t| t.name == "NAV2-PVT path (GNSS-only, lon,lat)")
        .context("missing NAV2-PVT map trace")?;
    let _nav_map = data
        .map
        .iter()
        .find(|t| t.name == "u-blox path (lon,lat)")
        .context("missing NAV-PVT map trace")?;

    let ekf_points = points_in_window_from_heading(&data.map_heading, t0_s, t1_s)
        .into_iter()
        .map(|(_, lon, lat)| [lon, lat])
        .collect::<Vec<_>>();
    let nav2_points = nav_points_in_window(frames, tl, true, t0_s, t1_s);
    let nav_points = nav_points_in_window(frames, tl, false, t0_s, t1_s);

    let ekf_nav2 = compute_lonlat_polyline_metrics(&ekf_points, &nav2_points);
    let ekf_nav = compute_lonlat_polyline_metrics(&ekf_points, &nav_points);
    let nav_nav2 = compute_lonlat_polyline_metrics(&nav_points, &nav2_points);

    if let Some(m) = ekf_nav2 {
        println!(
            "Map EKF -> NAV2       n={:4}  mean={:6.2} m  rms={:6.2} m  max={:6.2} m",
            m.n, m.mean_nearest_m, m.rms_nearest_m, m.max_nearest_m
        );
    }
    if let Some(m) = ekf_nav {
        println!(
            "Map EKF -> NAV-PVT    n={:4}  mean={:6.2} m  rms={:6.2} m  max={:6.2} m",
            m.n, m.mean_nearest_m, m.rms_nearest_m, m.max_nearest_m
        );
    }
    if let Some(m) = nav_nav2 {
        println!(
            "Map NAV-PVT -> NAV2   n={:4}  mean={:6.2} m  rms={:6.2} m  max={:6.2} m",
            m.n, m.mean_nearest_m, m.rms_nearest_m, m.max_nearest_m
        );
    }
    if let Some(m) = compute_exact_map_alignment(frames, tl, imu_source, ekf_cfg, t0_s, t1_s)? {
        println!(
            "Exact EKF -> NAV2     n={:4}  mean={:6.2} m  rms={:6.2} m  max={:6.2} m",
            m.n, m.mean_nearest_m, m.rms_nearest_m, m.max_nearest_m
        );
    }
    Ok(())
}

fn compute_exact_map_alignment(
    frames: &[sim::ubxlog::UbxFrame],
    tl: &sim::visualizer::pipeline::timebase::MasterTimeline,
    imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    t0_s: f64,
    t1_s: f64,
) -> Result<Option<PolylineMetrics>> {
    let data =
        build_ekf_compare_traces(frames, tl, imu_source, ekf_cfg, GnssOutageConfig::default());
    let ekf_n = find_trace(&data.cmp_pos, "EKF posN [m]")?;
    let ekf_e = find_trace(&data.cmp_pos, "EKF posE [m]")?;
    let ekf_d = find_trace(&data.cmp_pos, "EKF posD [m]")?;
    let nav2_events = collect_nav2_events(frames, tl, t0_s, t1_s);
    let Some(origin) = collect_nav2_origin(frames) else {
        return Ok(None);
    };
    let ref_lat = origin[0];
    let ref_lon = origin[1];
    let ref_h = origin[2];
    let ref_ecef = lla_to_ecef(ref_lat, ref_lon, ref_h);

    let mut exact_ekf = Vec::new();
    for h in data
        .map_heading
        .iter()
        .filter(|h| h.t_s >= t0_s && h.t_s <= t1_s)
    {
        let Some(n) = interp(ekf_n, h.t_s) else {
            continue;
        };
        let Some(e) = interp(ekf_e, h.t_s) else {
            continue;
        };
        let Some(d) = interp(ekf_d, h.t_s) else {
            continue;
        };
        let ecef = ned_to_ecef_exact(n, e, d, ref_ecef, ref_lat, ref_lon);
        let (lat, lon, _h) = ecef_to_lla_exact(ecef);
        exact_ekf.push([lon, lat]);
    }
    let nav2_points = nav2_events
        .into_iter()
        .map(|p| [p[2], p[1]])
        .collect::<Vec<_>>();
    Ok(compute_lonlat_polyline_metrics(&exact_ekf, &nav2_points))
}

fn collect_nav2_events(
    frames: &[sim::ubxlog::UbxFrame],
    tl: &sim::visualizer::pipeline::timebase::MasterTimeline,
    t0_s: f64,
    t1_s: f64,
) -> Vec<[f64; 4]> {
    let mut out = Vec::new();
    for f in frames {
        if !(f.class == 0x29 && f.id == 0x07) {
            continue;
        }
        let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters) else {
            continue;
        };
        let t_s = (t_ms - tl.t0_master_ms) * 1.0e-3;
        if t_s < t0_s || t_s > t1_s {
            continue;
        }
        if let Some(obs) = extract_nav2_pvt_obs(f)
            && obs.fix_ok
            && !obs.invalid_llh
        {
            out.push([t_s, obs.lat_deg, obs.lon_deg, obs.height_m]);
        }
    }
    out
}

fn collect_nav2_origin(frames: &[sim::ubxlog::UbxFrame]) -> Option<[f64; 3]> {
    for f in frames {
        if !(f.class == 0x29 && f.id == 0x07) {
            continue;
        }
        if let Some(obs) = extract_nav2_pvt_obs(f)
            && obs.fix_ok
            && !obs.invalid_llh
        {
            return Some([obs.lat_deg, obs.lon_deg, obs.height_m]);
        }
    }
    None
}

fn nav_points_in_window(
    frames: &[sim::ubxlog::UbxFrame],
    tl: &sim::visualizer::pipeline::timebase::MasterTimeline,
    nav2: bool,
    t0_s: f64,
    t1_s: f64,
) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    for f in frames {
        let Some(t_ms) = nearest_master_ms(f.seq, &tl.masters) else {
            continue;
        };
        let t_s = (t_ms - tl.t0_master_ms) * 1.0e-3;
        if t_s < t0_s || t_s > t1_s {
            continue;
        }
        if nav2 {
            if !(f.class == 0x29 && f.id == 0x07) {
                continue;
            }
        } else if !(f.class == 0x01 && f.id == 0x07) {
            continue;
        }
        let obs = if nav2 {
            extract_nav2_pvt_obs(f)
        } else {
            extract_nav_pvt_obs(f)
        };
        if let Some(obs) = obs
            && obs.fix_ok
            && !obs.invalid_llh
        {
            out.push([obs.lon_deg, obs.lat_deg]);
        }
    }
    out
}

fn points_in_window_from_heading(
    headings: &[sim::visualizer::model::HeadingSample],
    t0_s: f64,
    t1_s: f64,
) -> Vec<(f64, f64, f64)> {
    headings
        .iter()
        .filter(|h| h.t_s >= t0_s && h.t_s <= t1_s)
        .map(|h| (h.t_s, h.lon_deg, h.lat_deg))
        .collect()
}

fn compute_lonlat_polyline_metrics(
    src: &[[f64; 2]],
    reference: &[[f64; 2]],
) -> Option<PolylineMetrics> {
    if src.is_empty() || reference.len() < 2 {
        return None;
    }
    let mut n = 0usize;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut max_d = 0.0_f64;
    for p in src {
        let mut best = f64::INFINITY;
        for seg in reference.windows(2) {
            let d = lonlat_point_segment_distance_m(*p, seg[0], seg[1]);
            if d < best {
                best = d;
            }
        }
        if !best.is_finite() {
            continue;
        }
        n += 1;
        sum += best;
        sum_sq += best * best;
        max_d = max_d.max(best);
    }
    if n == 0 {
        return None;
    }
    Some(PolylineMetrics {
        n,
        mean_nearest_m: sum / n as f64,
        rms_nearest_m: (sum_sq / n as f64).sqrt(),
        max_nearest_m: max_d,
    })
}

fn lonlat_point_segment_distance_m(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let lat0 = p[1].to_radians();
    let m_per_deg_lat = 111_132.0_f64;
    let m_per_deg_lon = 111_320.0_f64 * lat0.cos().abs().max(1.0e-6);
    let to_xy = |q: [f64; 2]| [q[0] * m_per_deg_lon, q[1] * m_per_deg_lat];
    let px = to_xy(p);
    let ax = to_xy(a);
    let bx = to_xy(b);
    point_segment_distance(px, ax, bx)
}

fn ned_to_ecef_exact(
    n: f64,
    e: f64,
    d: f64,
    ref_ecef: [f64; 3],
    ref_lat_deg: f64,
    ref_lon_deg: f64,
) -> [f64; 3] {
    let lat = ref_lat_deg.to_radians();
    let lon = ref_lon_deg.to_radians();
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    let r = [
        [-slat * clon, -slon, -clat * clon],
        [-slat * slon, clon, -clat * slon],
        [clat, 0.0, -slat],
    ];
    [
        ref_ecef[0] + r[0][0] * n + r[0][1] * e + r[0][2] * d,
        ref_ecef[1] + r[1][0] * n + r[1][1] * e + r[1][2] * d,
        ref_ecef[2] + r[2][0] * n + r[2][1] * e + r[2][2] * d,
    ]
}

fn ecef_to_lla_exact(ecef: [f64; 3]) -> (f64, f64, f64) {
    let a = 6378137.0_f64;
    let e2 = 6.69437999014e-3_f64;
    let b = a * (1.0 - e2).sqrt();
    let ep2 = (a * a - b * b) / (b * b);
    let x = ecef[0];
    let y = ecef[1];
    let z = ecef[2];
    let p = x.hypot(y);
    let theta = (z * a).atan2(p * b);
    let (st, ct) = theta.sin_cos();
    let lat = (z + ep2 * b * st * st * st).atan2(p - e2 * a * ct * ct * ct);
    let lon = y.atan2(x);
    let sin_lat = lat.sin();
    let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    let h = p / lat.cos().max(1.0e-12) - n;
    (lat.to_degrees(), lon.to_degrees(), h)
}

fn load_frames(path: &PathBuf) -> Result<Vec<sim::ubxlog::UbxFrame>> {
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .read_to_end(&mut bytes)
        .context("failed to read log")?;
    Ok(parse_ubx_frames(&bytes, None))
}

fn run_case(
    _label: &str,
    frames: &[sim::ubxlog::UbxFrame],
    tl: &sim::visualizer::pipeline::timebase::MasterTimeline,
    imu_source: EkfImuSource,
    cfg: EkfCompareConfig,
    t0_s: f64,
    t1_s: f64,
) -> Result<(WindowMetrics, LagMetrics)> {
    let data = build_ekf_compare_traces(frames, tl, imu_source, cfg, GnssOutageConfig::default());
    let ekf_n = find_trace(&data.cmp_pos, "EKF posN [m]")?;
    let ubx_n = find_trace(&data.cmp_pos, "UBX posN [m]")?;
    let ekf_e = find_trace(&data.cmp_pos, "EKF posE [m]")?;
    let ubx_e = find_trace(&data.cmp_pos, "UBX posE [m]")?;
    let ubx_vn = find_trace(&data.cmp_vel, "UBX velN [m/s]")?;
    let ubx_ve = find_trace(&data.cmp_vel, "UBX velE [m/s]")?;

    let metrics = compute_window_metrics(ekf_n, ekf_e, ubx_n, ubx_e, ubx_vn, ubx_ve, t0_s, t1_s)
        .context("failed to compute baseline window metrics")?;
    let lag = compute_best_lag(ekf_n, ekf_e, ubx_n, ubx_e, ubx_vn, ubx_ve, t0_s, t1_s)
        .context("failed to compute lag metrics")?;
    Ok((metrics, lag))
}

fn sweep_r_body_vel(
    frames: &[sim::ubxlog::UbxFrame],
    tl: &sim::visualizer::pipeline::timebase::MasterTimeline,
    args: &Args,
    mut cfg: EkfCompareConfig,
) -> Result<()> {
    println!("R_BODY_VEL");
    for value in [1.0_f32, 2.0, 5.0, 10.0, 20.0] {
        cfg.r_body_vel = value;
        let (m, lag) = run_case(
            "r_body_vel",
            frames,
            tl,
            args.ekf_imu_source,
            cfg,
            args.window_start_s,
            args.window_end_s,
        )?;
        print_case(&format!("  {:>5.1}", value), &m, &lag);
    }
    Ok(())
}

fn sweep_pos_scale(
    frames: &[sim::ubxlog::UbxFrame],
    tl: &sim::visualizer::pipeline::timebase::MasterTimeline,
    args: &Args,
    mut cfg: EkfCompareConfig,
) -> Result<()> {
    println!("GNSS_POS_R_SCALE");
    for value in [1.0_f64, 0.3, 0.1, 0.03, 0.01] {
        cfg.gnss_pos_r_scale = value;
        let (m, lag) = run_case(
            "gnss_pos",
            frames,
            tl,
            args.ekf_imu_source,
            cfg,
            args.window_start_s,
            args.window_end_s,
        )?;
        print_case(&format!("  {:>5.2}", value), &m, &lag);
    }
    Ok(())
}

fn sweep_vel_scale(
    frames: &[sim::ubxlog::UbxFrame],
    tl: &sim::visualizer::pipeline::timebase::MasterTimeline,
    args: &Args,
    mut cfg: EkfCompareConfig,
) -> Result<()> {
    println!("GNSS_VEL_R_SCALE");
    for value in [1.0_f64, 0.3, 0.1, 0.03, 0.01] {
        cfg.gnss_vel_r_scale = value;
        let (m, lag) = run_case(
            "gnss_vel",
            frames,
            tl,
            args.ekf_imu_source,
            cfg,
            args.window_start_s,
            args.window_end_s,
        )?;
        print_case(&format!("  {:>5.2}", value), &m, &lag);
    }
    Ok(())
}

fn sweep_accel_var(
    frames: &[sim::ubxlog::UbxFrame],
    tl: &sim::visualizer::pipeline::timebase::MasterTimeline,
    args: &Args,
    mut cfg: EkfCompareConfig,
) -> Result<()> {
    println!("accel_var");
    for value in [3.0_f32, 6.0, 12.0, 24.0, 48.0] {
        let mut noise = cfg.predict_noise.unwrap_or_default();
        noise.accel_var = value;
        cfg.predict_noise = Some(noise);
        let (m, lag) = run_case(
            "accel_var",
            frames,
            tl,
            args.ekf_imu_source,
            cfg,
            args.window_start_s,
            args.window_end_s,
        )?;
        print_case(&format!("  {:>5.1}", value), &m, &lag);
    }
    Ok(())
}

fn sweep_gyro_var(
    frames: &[sim::ubxlog::UbxFrame],
    tl: &sim::visualizer::pipeline::timebase::MasterTimeline,
    args: &Args,
    mut cfg: EkfCompareConfig,
) -> Result<()> {
    println!("gyro_var");
    for value in [1.0e-5_f32, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3] {
        let mut noise = cfg.predict_noise.unwrap_or_default();
        noise.gyro_var = value;
        cfg.predict_noise = Some(noise);
        let (m, lag) = run_case(
            "gyro_var",
            frames,
            tl,
            args.ekf_imu_source,
            cfg,
            args.window_start_s,
            args.window_end_s,
        )?;
        print_case(&format!("  {:>8.1e}", value), &m, &lag);
    }
    Ok(())
}

fn print_case(label: &str, m: &WindowMetrics, lag: &LagMetrics) {
    println!(
        "{label:>12}  n={:4}  mean2d={:6.2}  rms2d={:6.2}  along|={:6.2}  along_rms={:6.2}  cross|={:6.2}  cross_rms={:6.2}  cross_max={:6.2}  mean_hdg={:6.1}  best_lag={:5.2}s  lag_mean2d={:6.2}  lag_rms2d={:6.2}  lag_along|={:6.2}  lag_cross|={:6.2}",
        m.n,
        m.mean_2d_m,
        m.rms_2d_m,
        m.mean_abs_along_m,
        m.rms_along_m,
        m.mean_abs_cross_m,
        m.rms_cross_m,
        m.max_abs_cross_m,
        m.mean_heading_deg,
        lag.lag_s,
        lag.mean_2d_m,
        lag.rms_2d_m,
        lag.mean_abs_along_m,
        lag.mean_abs_cross_m,
    );
}

fn compute_window_metrics(
    ekf_n: &[[f64; 2]],
    ekf_e: &[[f64; 2]],
    ubx_n: &[[f64; 2]],
    ubx_e: &[[f64; 2]],
    ubx_vn: &[[f64; 2]],
    ubx_ve: &[[f64; 2]],
    t0_s: f64,
    t1_s: f64,
) -> Option<WindowMetrics> {
    let mut n = 0usize;
    let mut sum_2d = 0.0;
    let mut sum_2d_sq = 0.0;
    let mut sum_abs_along = 0.0;
    let mut sum_along_sq = 0.0;
    let mut sum_abs_cross = 0.0;
    let mut sum_cross_sq = 0.0;
    let mut max_abs_cross = 0.0_f64;
    let mut sum_heading_deg = 0.0;

    for sample in ubx_n.iter().filter(|p| p[0] >= t0_s && p[0] <= t1_s) {
        let t = sample[0];
        let ubx_n_t = interp(ubx_n, t)?;
        let ubx_e_t = interp(ubx_e, t)?;
        let vn = interp(ubx_vn, t)?;
        let ve = interp(ubx_ve, t)?;
        let ekf_n_t = interp(ekf_n, t)?;
        let ekf_e_t = interp(ekf_e, t)?;

        let speed = vn.hypot(ve).max(1.0e-6);
        let u_n = vn / speed;
        let u_e = ve / speed;
        let left_n = -u_e;
        let left_e = u_n;
        let err_n = ekf_n_t - ubx_n_t;
        let err_e = ekf_e_t - ubx_e_t;
        let along = err_n * u_n + err_e * u_e;
        let cross = err_n * left_n + err_e * left_e;
        let err_2d = err_n.hypot(err_e);
        let hdg_deg = ve.atan2(vn).to_degrees();

        n += 1;
        sum_2d += err_2d;
        sum_2d_sq += err_2d * err_2d;
        sum_abs_along += along.abs();
        sum_along_sq += along * along;
        sum_abs_cross += cross.abs();
        sum_cross_sq += cross * cross;
        max_abs_cross = max_abs_cross.max(cross.abs());
        sum_heading_deg += hdg_deg;
    }

    if n == 0 {
        return None;
    }

    Some(WindowMetrics {
        n,
        mean_2d_m: sum_2d / n as f64,
        rms_2d_m: (sum_2d_sq / n as f64).sqrt(),
        mean_abs_along_m: sum_abs_along / n as f64,
        rms_along_m: (sum_along_sq / n as f64).sqrt(),
        mean_abs_cross_m: sum_abs_cross / n as f64,
        rms_cross_m: (sum_cross_sq / n as f64).sqrt(),
        max_abs_cross_m: max_abs_cross,
        mean_heading_deg: sum_heading_deg / n as f64,
    })
}

fn compute_best_lag(
    ekf_n: &[[f64; 2]],
    ekf_e: &[[f64; 2]],
    ubx_n: &[[f64; 2]],
    ubx_e: &[[f64; 2]],
    ubx_vn: &[[f64; 2]],
    ubx_ve: &[[f64; 2]],
    t0_s: f64,
    t1_s: f64,
) -> Option<LagMetrics> {
    let mut best: Option<LagMetrics> = None;
    for step in -80..=80 {
        let lag_s = step as f64 * 0.05;
        let mut n = 0usize;
        let mut sum_2d = 0.0;
        let mut sum_2d_sq = 0.0;
        let mut sum_abs_along = 0.0;
        let mut sum_abs_cross = 0.0;
        for sample in ubx_n.iter().filter(|p| p[0] >= t0_s && p[0] <= t1_s) {
            let t = sample[0];
            let shifted_t = t + lag_s;
            let ubx_n_t = interp(ubx_n, t)?;
            let ubx_e_t = interp(ubx_e, t)?;
            let vn = interp(ubx_vn, t)?;
            let ve = interp(ubx_ve, t)?;
            let ekf_n_t = match interp(ekf_n, shifted_t) {
                Some(v) => v,
                None => continue,
            };
            let ekf_e_t = match interp(ekf_e, shifted_t) {
                Some(v) => v,
                None => continue,
            };
            let speed = vn.hypot(ve).max(1.0e-6);
            let u_n = vn / speed;
            let u_e = ve / speed;
            let left_n = -u_e;
            let left_e = u_n;
            let err_n = ekf_n_t - ubx_n_t;
            let err_e = ekf_e_t - ubx_e_t;
            let along = err_n * u_n + err_e * u_e;
            let cross = err_n * left_n + err_e * left_e;
            let err_2d = err_n.hypot(err_e);
            n += 1;
            sum_2d += err_2d;
            sum_2d_sq += err_2d * err_2d;
            sum_abs_along += along.abs();
            sum_abs_cross += cross.abs();
        }
        if n == 0 {
            continue;
        }
        let cand = LagMetrics {
            lag_s,
            mean_2d_m: sum_2d / n as f64,
            rms_2d_m: (sum_2d_sq / n as f64).sqrt(),
            mean_abs_along_m: sum_abs_along / n as f64,
            mean_abs_cross_m: sum_abs_cross / n as f64,
        };
        match best {
            None => best = Some(cand),
            Some(prev) if cand.rms_2d_m < prev.rms_2d_m => best = Some(cand),
            _ => {}
        }
    }
    best
}

fn compute_polyline_metrics(
    src_n: &[[f64; 2]],
    src_e: &[[f64; 2]],
    ref_n: &[[f64; 2]],
    ref_e: &[[f64; 2]],
    t0_s: f64,
    t1_s: f64,
) -> Option<PolylineMetrics> {
    let src_pts = collect_ne_points(src_n, src_e, t0_s, t1_s);
    let ref_pts = collect_ne_points(ref_n, ref_e, t0_s, t1_s);
    if src_pts.is_empty() || ref_pts.len() < 2 {
        return None;
    }
    let mut n = 0usize;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut max_d = 0.0_f64;
    for p in &src_pts {
        let mut best = f64::INFINITY;
        for seg in ref_pts.windows(2) {
            let d = point_segment_distance(*p, seg[0], seg[1]);
            if d < best {
                best = d;
            }
        }
        if !best.is_finite() {
            continue;
        }
        n += 1;
        sum += best;
        sum_sq += best * best;
        max_d = max_d.max(best);
    }
    if n == 0 {
        return None;
    }
    Some(PolylineMetrics {
        n,
        mean_nearest_m: sum / n as f64,
        rms_nearest_m: (sum_sq / n as f64).sqrt(),
        max_nearest_m: max_d,
    })
}

fn collect_ne_points(
    n_trace: &[[f64; 2]],
    e_trace: &[[f64; 2]],
    t0_s: f64,
    t1_s: f64,
) -> Vec<[f64; 2]> {
    n_trace
        .iter()
        .filter(|p| p[0] >= t0_s && p[0] <= t1_s)
        .filter_map(|p| {
            let t = p[0];
            let n = interp(n_trace, t)?;
            let e = interp(e_trace, t)?;
            Some([n, e])
        })
        .collect()
}

fn point_segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let ab2 = ab[0] * ab[0] + ab[1] * ab[1];
    if ab2 <= 1.0e-12 {
        return ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt();
    }
    let u = ((ap[0] * ab[0] + ap[1] * ab[1]) / ab2).clamp(0.0, 1.0);
    let q = [a[0] + u * ab[0], a[1] + u * ab[1]];
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
}

fn find_trace<'a>(
    traces: &'a [sim::visualizer::model::Trace],
    name: &str,
) -> Result<&'a [[f64; 2]]> {
    traces
        .iter()
        .find(|tr| tr.name == name)
        .map(|tr| tr.points.as_slice())
        .with_context(|| format!("missing trace {name}"))
}

fn interp(points: &[[f64; 2]], t: f64) -> Option<f64> {
    if points.is_empty() || !t.is_finite() {
        return None;
    }
    let idx = points.partition_point(|p| p[0] < t);
    if idx == 0 {
        return (points[0][0] - t).abs().le(&1.0e-9).then_some(points[0][1]);
    }
    if idx >= points.len() {
        return (points[points.len() - 1][0] - t)
            .abs()
            .le(&1.0e-9)
            .then_some(points[points.len() - 1][1]);
    }
    let a = points[idx - 1];
    let b = points[idx];
    if !a[0].is_finite() || !b[0].is_finite() {
        return None;
    }
    let dt = b[0] - a[0];
    if dt.abs() < 1.0e-9 {
        return Some(a[1]);
    }
    let u = (t - a[0]) / dt;
    Some(a[1] + u * (b[1] - a[1]))
}

fn parse_ekf_imu_source(s: &str) -> Result<EkfImuSource, String> {
    match s.to_ascii_lowercase().as_str() {
        "align" => Ok(EkfImuSource::Align),
        "esfalg" | "esf-alg" | "alg" => Ok(EkfImuSource::EsfAlg),
        other => Err(format!("invalid ekf imu source: {other}")),
    }
}
