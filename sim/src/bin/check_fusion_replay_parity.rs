#[cfg(not(feature = "c-reference"))]
fn main() {
    eprintln!(
        "check_fusion_replay_parity requires c-reference: cargo run -p sim --features c-reference --bin check_fusion_replay_parity -- ..."
    );
}

#[cfg(feature = "c-reference")]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, bail};
    use clap::Parser;
    use sensor_fusion::c_api::CSensorFusionWrapper;
    use sensor_fusion::fusion::SensorFusion;
    use sim::datasets::generic_replay::{
        fusion_gnss_sample as to_fusion_gnss, fusion_imu_sample as to_fusion_imu,
    };
    use sim::datasets::ubx_replay::{UbxReplayConfig, build_generic_replay_from_frames};
    use sim::ubxlog::parse_ubx_frames;
    use sim::visualizer::pipeline::timebase::build_master_timeline;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(value_name = "LOGFILE")]
        logfile: std::path::PathBuf,
        #[arg(long, default_value_t = 1.0e-3)]
        tolerance: f32,
        #[arg(long, default_value_t = 1.0)]
        gnss_pos_r_scale: f64,
        #[arg(long, default_value_t = 3.0)]
        gnss_vel_r_scale: f64,
        #[arg(long)]
        dump_before_gnss_t_s: Option<f32>,
    }

    let args = Args::parse();
    let bytes = std::fs::read(&args.logfile)
        .with_context(|| format!("failed to read {}", args.logfile.display()))?;
    let frames = parse_ubx_frames(&bytes, None);
    let tl = build_master_timeline(&frames);
    if tl.masters.is_empty() {
        bail!("no master timeline");
    }
    let replay = build_generic_replay_from_frames(
        &frames,
        &tl,
        UbxReplayConfig {
            gnss_pos_r_scale: args.gnss_pos_r_scale,
            gnss_vel_r_scale: args.gnss_vel_r_scale,
            ..UbxReplayConfig::default()
        },
    )?;

    let mut c = CSensorFusionWrapper::new_internal();
    let mut r = SensorFusion::new();
    let mut gnss_idx = 0usize;
    let mut max_diff = 0.0f32;

    for imu in replay.imu_samples {
        while gnss_idx < replay.gnss_samples.len()
            && replay.gnss_samples[gnss_idx].t_s <= imu.t_s + 1.0e-9
        {
            let sample = to_fusion_gnss(replay.gnss_samples[gnss_idx]);
            if args
                .dump_before_gnss_t_s
                .is_some_and(|target| (sample.t_s - target).abs() < 0.01)
            {
                dump_state("before_gnss", &c, &r, sample.t_s);
            }
            c.process_gnss(sample);
            r.process_gnss(sample);
            if args
                .dump_before_gnss_t_s
                .is_some_and(|target| (sample.t_s - target).abs() < 0.01)
            {
                dump_state("after_gnss", &c, &r, sample.t_s);
            }
            max_diff = max_diff.max(compare(&c, &r, "GNSS", sample.t_s, args.tolerance)?);
            gnss_idx += 1;
        }

        let sample = to_fusion_imu(imu);
        c.process_imu(sample);
        r.process_imu(sample);
        max_diff = max_diff.max(compare(&c, &r, "IMU", sample.t_s, args.tolerance)?);
    }

    println!("fusion C/Rust replay parity passed max_diff={max_diff:.9e}");
    Ok(())
}

#[cfg(feature = "c-reference")]
fn dump_state(
    label: &str,
    c: &sensor_fusion::c_api::CSensorFusionWrapper,
    r: &sensor_fusion::fusion::SensorFusion,
    t_s: f32,
) {
    let (Some(c_eskf), Some(r_eskf)) = (c.eskf(), r.eskf()) else {
        println!("{label} t={t_s:.6}: one side not initialized");
        return;
    };
    let cn = &c_eskf.nominal;
    let rn = &r_eskf.nominal;
    println!(
        "{label} t={t_s:.6} C vd={:.9} pn={:.9} pe={:.9} pd={:.9} p55={:.9e} p05={:.9e} p15={:.9e} p25={:.9e}",
        cn.vd, cn.pn, cn.pe, cn.pd, c_eskf.p[5][5], c_eskf.p[0][5], c_eskf.p[1][5], c_eskf.p[2][5],
    );
    println!(
        "{label} t={t_s:.6} R vd={:.9} pn={:.9} pe={:.9} pd={:.9} p55={:.9e} p05={:.9e} p15={:.9e} p25={:.9e}",
        rn.vd, rn.pn, rn.pe, rn.pd, r_eskf.p[5][5], r_eskf.p[0][5], r_eskf.p[1][5], r_eskf.p[2][5],
    );
}

#[cfg(feature = "c-reference")]
fn compare(
    c: &sensor_fusion::c_api::CSensorFusionWrapper,
    r: &sensor_fusion::fusion::SensorFusion,
    source: &str,
    t_s: f32,
    tolerance: f32,
) -> anyhow::Result<f32> {
    let Some(c_eskf) = c.eskf() else {
        if r.eskf().is_some() {
            anyhow::bail!("{source} t={t_s:.6}: Rust initialized before C");
        }
        return Ok(0.0);
    };
    let Some(r_eskf) = r.eskf() else {
        anyhow::bail!("{source} t={t_s:.6}: C initialized before Rust");
    };
    let cn = &c_eskf.nominal;
    let rn = &r_eskf.nominal;
    let pairs = [
        ("q0", cn.q0, rn.q0),
        ("q1", cn.q1, rn.q1),
        ("q2", cn.q2, rn.q2),
        ("q3", cn.q3, rn.q3),
        ("vn", cn.vn, rn.vn),
        ("ve", cn.ve, rn.ve),
        ("vd", cn.vd, rn.vd),
        ("pn", cn.pn, rn.pn),
        ("pe", cn.pe, rn.pe),
        ("pd", cn.pd, rn.pd),
        ("qcs0", cn.qcs0, rn.qcs0),
        ("qcs1", cn.qcs1, rn.qcs1),
        ("qcs2", cn.qcs2, rn.qcs2),
        ("qcs3", cn.qcs3, rn.qcs3),
    ];
    let mut max_diff = 0.0f32;
    let mut worst = ("", 0.0, 0.0);
    for (name, a, b) in pairs {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
            worst = (name, a, b);
        }
    }
    for i in 0..18 {
        for j in 0..18 {
            let d = (c_eskf.p[i][j] - r_eskf.p[i][j]).abs();
            if d > max_diff {
                max_diff = d;
                worst = ("P", c_eskf.p[i][j], r_eskf.p[i][j]);
            }
        }
    }
    if max_diff > tolerance {
        anyhow::bail!(
            "{source} t={t_s:.6}: max_diff={max_diff:.9e} worst={} C={:.9e} Rust={:.9e} C_last_type={} Rust_last_type={} C_innov={:.9e} Rust_innov={:.9e}",
            worst.0,
            worst.1,
            worst.2,
            c_eskf.update_diag.last_type,
            r_eskf.update_diag.last_type,
            c_eskf.update_diag.last_innovation,
            r_eskf.update_diag.last_innovation,
        );
    }
    Ok(max_diff)
}
