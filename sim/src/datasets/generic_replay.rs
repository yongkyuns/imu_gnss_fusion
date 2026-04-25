use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use sensor_fusion::fusion::{FusionGnssSample, FusionImuSample};

#[derive(Clone, Copy, Debug)]
pub struct GenericImuSample {
    pub t_s: f64,
    pub gyro_radps: [f64; 3],
    pub accel_mps2: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct GenericGnssSample {
    pub t_s: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
    pub vel_ned_mps: [f64; 3],
    pub pos_std_m: [f64; 3],
    pub vel_std_mps: [f64; 3],
    pub heading_rad: Option<f64>,
}

pub fn load_imu_samples(dir: &Path) -> Result<Vec<GenericImuSample>> {
    let rows = read_rows(&dir.join("imu.csv"), 7)?;
    Ok(rows
        .into_iter()
        .map(|row| GenericImuSample {
            t_s: row[0],
            gyro_radps: [row[1], row[2], row[3]],
            accel_mps2: [row[4], row[5], row[6]],
        })
        .collect())
}

pub fn load_gnss_samples(dir: &Path) -> Result<Vec<GenericGnssSample>> {
    let rows = read_rows(&dir.join("gnss.csv"), 14)?;
    Ok(rows
        .into_iter()
        .map(|row| GenericGnssSample {
            t_s: row[0],
            lat_deg: row[1],
            lon_deg: row[2],
            height_m: row[3],
            vel_ned_mps: [row[4], row[5], row[6]],
            pos_std_m: [row[7], row[8], row[9]],
            vel_std_mps: [row[10], row[11], row[12]],
            heading_rad: if row[13].is_finite() {
                Some(row[13])
            } else {
                None
            },
        })
        .collect())
}

pub fn write_samples(
    dir: &Path,
    imu_samples: &[GenericImuSample],
    gnss_samples: &[GenericGnssSample],
) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    write_imu_csv(&dir.join("imu.csv"), imu_samples)?;
    write_gnss_csv(&dir.join("gnss.csv"), gnss_samples)?;
    Ok(())
}

pub fn fusion_imu_sample(sample: GenericImuSample) -> FusionImuSample {
    FusionImuSample {
        t_s: sample.t_s as f32,
        gyro_radps: [
            sample.gyro_radps[0] as f32,
            sample.gyro_radps[1] as f32,
            sample.gyro_radps[2] as f32,
        ],
        accel_mps2: [
            sample.accel_mps2[0] as f32,
            sample.accel_mps2[1] as f32,
            sample.accel_mps2[2] as f32,
        ],
    }
}

pub fn fusion_gnss_sample(sample: GenericGnssSample) -> FusionGnssSample {
    FusionGnssSample {
        t_s: sample.t_s as f32,
        lat_deg: sample.lat_deg as f32,
        lon_deg: sample.lon_deg as f32,
        height_m: sample.height_m as f32,
        vel_ned_mps: [
            sample.vel_ned_mps[0] as f32,
            sample.vel_ned_mps[1] as f32,
            sample.vel_ned_mps[2] as f32,
        ],
        pos_std_m: [
            sample.pos_std_m[0] as f32,
            sample.pos_std_m[1] as f32,
            sample.pos_std_m[2] as f32,
        ],
        vel_std_mps: [
            sample.vel_std_mps[0] as f32,
            sample.vel_std_mps[1] as f32,
            sample.vel_std_mps[2] as f32,
        ],
        heading_rad: sample.heading_rad.map(|v| v as f32),
    }
}

fn write_imu_csv(path: &Path, samples: &[GenericImuSample]) -> Result<()> {
    let file =
        fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(w, "t_s,gx_radps,gy_radps,gz_radps,ax_mps2,ay_mps2,az_mps2")?;
    for s in samples {
        writeln!(
            w,
            "{},{},{},{},{},{},{}",
            s.t_s,
            s.gyro_radps[0],
            s.gyro_radps[1],
            s.gyro_radps[2],
            s.accel_mps2[0],
            s.accel_mps2[1],
            s.accel_mps2[2]
        )?;
    }
    Ok(())
}

fn write_gnss_csv(path: &Path, samples: &[GenericGnssSample]) -> Result<()> {
    let file =
        fs::File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "t_s,lat_deg,lon_deg,height_m,vn_mps,ve_mps,vd_mps,pos_std_n_m,pos_std_e_m,pos_std_d_m,vel_std_n_mps,vel_std_e_mps,vel_std_d_mps,heading_rad"
    )?;
    for s in samples {
        let heading = s.heading_rad.unwrap_or(f64::NAN);
        writeln!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            s.t_s,
            s.lat_deg,
            s.lon_deg,
            s.height_m,
            s.vel_ned_mps[0],
            s.vel_ned_mps[1],
            s.vel_ned_mps[2],
            s.pos_std_m[0],
            s.pos_std_m[1],
            s.pos_std_m[2],
            s.vel_std_mps[0],
            s.vel_std_mps[1],
            s.vel_std_mps[2],
            heading
        )?;
    }
    Ok(())
}

fn read_rows(path: &Path, cols: usize) -> Result<Vec<Vec<f64>>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let row = line
            .split(',')
            .map(|part| {
                let trimmed = part.trim();
                if trimmed.eq_ignore_ascii_case("nan") {
                    Ok(f64::NAN)
                } else {
                    trimmed.parse::<f64>().with_context(|| {
                        format!(
                            "failed to parse numeric field in {}: {}",
                            path.display(),
                            line
                        )
                    })
                }
            })
            .collect::<Result<Vec<_>>>()?;
        if row.len() != cols {
            bail!("{} expected {} columns per row", path.display(), cols);
        }
        out.push(row);
    }
    Ok(out)
}
