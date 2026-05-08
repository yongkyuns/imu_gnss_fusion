use anyhow::{Context, Result, bail};
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct ImuSample {
    pub t_s: f64,
    pub gyro_vehicle_radps: [f64; 3],
    pub accel_vehicle_mps2: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct GnssSample {
    pub t_s: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
    pub vel_ned_mps: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct TruthSample {
    pub t_s: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
    pub vel_ned_mps: [f64; 3],
    pub q_bn: [f64; 4],
}

pub fn load_imu_samples(
    data_dir: &Path,
    use_ref_signals: bool,
    data_key: usize,
) -> Result<Vec<ImuSample>> {
    let time = read_time_csv(&data_dir.join("time.csv"))?;
    let gyro_name = if use_ref_signals {
        "ref_gyro.csv".to_string()
    } else {
        format!("gyro-{data_key}.csv")
    };
    let accel_name = if use_ref_signals {
        "ref_accel.csv".to_string()
    } else {
        format!("accel-{data_key}.csv")
    };
    let gyro_path = data_dir.join(&gyro_name);
    let gyro =
        read_matrix3_csv(&gyro_path).with_context(|| format!("failed to load {}", gyro_name))?;
    let accel = read_matrix3_csv(&data_dir.join(&accel_name))
        .with_context(|| format!("failed to load {}", accel_name))?;
    if time.len() != gyro.len() || time.len() != accel.len() {
        bail!("IMU files have inconsistent lengths");
    }
    let gyro_is_deg = csv_header_contains(&gyro_path, "deg/s")?;
    let mut out = Vec::with_capacity(time.len());
    for i in 0..time.len() {
        let gyro_vehicle_radps = if gyro_is_deg {
            [
                gyro[i][0].to_radians(),
                gyro[i][1].to_radians(),
                gyro[i][2].to_radians(),
            ]
        } else {
            gyro[i]
        };
        out.push(ImuSample {
            t_s: time[i],
            gyro_vehicle_radps,
            accel_vehicle_mps2: accel[i],
        });
    }
    Ok(out)
}

pub fn load_gnss_samples(
    data_dir: &Path,
    use_ref_signals: bool,
    data_key: usize,
) -> Result<Vec<GnssSample>> {
    let gps_time = read_time_csv(&data_dir.join("gps_time.csv"))?;
    let gps_name = if use_ref_signals {
        "ref_gps.csv".to_string()
    } else {
        format!("gps-{data_key}.csv")
    };
    let gps = read_matrix_csv(&data_dir.join(&gps_name), 6)
        .with_context(|| format!("failed to load {}", gps_name))?;
    if gps_time.len() != gps.len() {
        bail!("GNSS files have inconsistent lengths");
    }
    let mut out = Vec::with_capacity(gps.len());
    for i in 0..gps.len() {
        out.push(GnssSample {
            t_s: gps_time[i],
            lat_deg: gps[i][0],
            lon_deg: gps[i][1],
            height_m: gps[i][2],
            vel_ned_mps: [gps[i][3], gps[i][4], gps[i][5]],
        });
    }
    Ok(out)
}

pub fn load_truth_samples(data_dir: &Path) -> Result<Vec<TruthSample>> {
    let time = read_time_csv(&data_dir.join("time.csv"))?;
    let pos =
        read_matrix_csv(&data_dir.join("ref_pos.csv"), 3).context("failed to load ref_pos.csv")?;
    let vel =
        read_matrix_csv(&data_dir.join("ref_vel.csv"), 3).context("failed to load ref_vel.csv")?;
    let quat = read_matrix_csv(&data_dir.join("ref_att_quat.csv"), 4)
        .context("failed to load ref_att_quat.csv")?;
    if time.len() != pos.len() || time.len() != vel.len() || time.len() != quat.len() {
        bail!("truth files have inconsistent lengths");
    }
    let mut out = Vec::with_capacity(time.len());
    for i in 0..time.len() {
        out.push(TruthSample {
            t_s: time[i],
            lat_deg: pos[i][0],
            lon_deg: pos[i][1],
            height_m: pos[i][2],
            vel_ned_mps: [vel[i][0], vel[i][1], vel[i][2]],
            q_bn: [quat[i][0], quat[i][1], quat[i][2], quat[i][3]],
        });
    }
    Ok(out)
}

fn read_time_csv(path: &Path) -> Result<Vec<f64>> {
    let rows = read_csv_rows(path)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() != 1 {
            bail!("{} expected 1 column per row", path.display());
        }
        out.push(row[0]);
    }
    Ok(out)
}

fn read_matrix3_csv(path: &Path) -> Result<Vec<[f64; 3]>> {
    let rows = read_matrix_csv(path, 3)?;
    Ok(rows.into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn read_matrix_csv(path: &Path, cols: usize) -> Result<Vec<Vec<f64>>> {
    let rows = read_csv_rows(path)?;
    for row in &rows {
        if row.len() != cols {
            bail!("{} expected {} columns per row", path.display(), cols);
        }
    }
    Ok(rows)
}

fn read_csv_rows(path: &Path) -> Result<Vec<Vec<f64>>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let mut row = Vec::new();
        for part in line.split(',') {
            row.push(part.trim().parse::<f64>().with_context(|| {
                format!(
                    "failed to parse numeric field in {}: {}",
                    path.display(),
                    line
                )
            })?);
        }
        out.push(row);
    }
    Ok(out)
}

fn csv_header_contains(path: &Path, needle: &str) -> Result<bool> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let header = text.lines().next().unwrap_or_default().to_ascii_lowercase();
    Ok(header.contains(&needle.to_ascii_lowercase()))
}
