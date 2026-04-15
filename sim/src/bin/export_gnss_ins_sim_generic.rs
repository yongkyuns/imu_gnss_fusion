use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;
use sim::datasets::generic_replay::{GenericGnssSample, GenericImuSample, write_samples};
use sim::datasets::gnss_ins_sim::{
    load_gnss_samples as load_dataset_gnss_samples, load_imu_samples as load_dataset_imu_samples,
};
use sim::eval::gnss_ins::{SignalSource, quat_from_rpy_alg_deg, quat_rotate};

#[derive(Parser, Debug)]
#[command(name = "export_gnss_ins_sim_generic")]
struct Args {
    #[arg(value_name = "DATA_DIR")]
    data_dir: PathBuf,
    #[arg(value_name = "OUT_DIR")]
    out_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = SignalSource::Meas)]
    signal_source: SignalSource,
    #[arg(long, default_value_t = 0)]
    data_key: usize,
    #[arg(long, default_value_t = 0.0)]
    mount_roll_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    mount_pitch_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    mount_yaw_deg: f64,
    #[arg(long, default_value_t = 0.5)]
    gnss_pos_std_m: f32,
    #[arg(long, default_value_t = 0.2)]
    gnss_vel_std_mps: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let imu = load_dataset_imu_samples(
        &args.data_dir,
        args.signal_source.use_ref_signals(),
        args.data_key,
    )?;
    let gnss = load_dataset_gnss_samples(
        &args.data_dir,
        args.signal_source.use_ref_signals(),
        args.data_key,
    )?;
    if imu.is_empty() || gnss.is_empty() {
        bail!("need IMU and GNSS samples");
    }
    let q_truth = quat_from_rpy_alg_deg(
        args.mount_roll_deg,
        args.mount_pitch_deg,
        args.mount_yaw_deg,
    );
    let imu_samples = imu
        .iter()
        .map(|s| GenericImuSample {
            t_s: s.t_s,
            gyro_radps: quat_rotate(q_truth, s.gyro_vehicle_radps),
            accel_mps2: quat_rotate(q_truth, s.accel_vehicle_mps2),
        })
        .collect::<Vec<_>>();
    let gnss_samples = gnss
        .iter()
        .map(|s| GenericGnssSample {
            t_s: s.t_s,
            lat_deg: s.lat_deg,
            lon_deg: s.lon_deg,
            height_m: s.height_m,
            vel_ned_mps: s.vel_ned_mps,
            pos_std_m: [
                args.gnss_pos_std_m as f64,
                args.gnss_pos_std_m as f64,
                args.gnss_pos_std_m as f64,
            ],
            vel_std_mps: [
                args.gnss_vel_std_mps as f64,
                args.gnss_vel_std_mps as f64,
                args.gnss_vel_std_mps as f64,
            ],
            heading_rad: None,
        })
        .collect::<Vec<_>>();
    write_samples(&args.out_dir, &imu_samples, &gnss_samples)?;
    println!("generic_replay_dir={}", args.out_dir.display());
    println!("imu_samples={} gnss_samples={}", imu_samples.len(), gnss_samples.len());
    Ok(())
}
