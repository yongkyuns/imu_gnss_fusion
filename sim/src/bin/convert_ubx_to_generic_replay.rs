use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use sim::datasets::generic_replay::write_samples;
use sim::datasets::ubx_replay::{UbxReplayConfig, load_generic_replay};

#[derive(Parser, Debug)]
#[command(name = "convert_ubx_to_generic_replay")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(value_name = "OUT_DIR")]
    out_dir: PathBuf,
    #[arg(long, default_value_t = 0.1)]
    gnss_pos_r_scale: f64,
    #[arg(long, default_value_t = 3.0)]
    gnss_vel_r_scale: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (imu_samples, gnss_samples) = load_generic_replay(
        &args.logfile,
        UbxReplayConfig {
            gnss_pos_r_scale: args.gnss_pos_r_scale,
            gnss_vel_r_scale: args.gnss_vel_r_scale,
            ..UbxReplayConfig::default()
        },
    )?;
    write_samples(&args.out_dir, &imu_samples, &gnss_samples)?;
    println!("generic_replay_dir={}", args.out_dir.display());
    println!("imu_samples={} gnss_samples={}", imu_samples.len(), gnss_samples.len());
    Ok(())
}
