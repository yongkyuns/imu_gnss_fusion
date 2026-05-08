use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct GyroSample {
    pub ttag_us: i64,
    pub omega_radps: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct AccelSample {
    pub ttag_us: i64,
    pub accel_mps2: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct GnssVelocity {
    pub vel_n: [f64; 3],
    pub vel_acc_n: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct GnssSample {
    pub ttag_us: i64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
    pub speed_mps: f64,
    pub heading_deg: f64,
    pub h_acc_m: f64,
    pub v_acc_m: f64,
    pub speed_acc_mps: f64,
    pub velocity: Option<GnssVelocity>,
}

#[derive(Debug, Clone)]
pub struct TruthNavSample {
    pub ttag_us: i64,
    pub pitch_car_deg: f64,
}

pub fn resolve_single_file(input_dir: &Path, suffix: &str) -> Result<PathBuf> {
    let mut matches = Vec::new();
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(suffix))
        {
            matches.push(path);
        }
    }
    matches.sort();
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => bail!(
            "missing file with suffix {suffix} in {}",
            input_dir.display()
        ),
        _ => bail!(
            "multiple files with suffix {suffix} in {}",
            input_dir.display()
        ),
    }
}

pub fn import_gyro_data(path: &Path) -> Result<Vec<GyroSample>> {
    let rows = semicolon_rows(path, 3)?;
    rows.into_iter()
        .map(|row| {
            Ok(GyroSample {
                ttag_us: (parse_f64(&row[0])? / 1000.0).floor() as i64,
                omega_radps: [
                    parse_f64(&row[1])?,
                    parse_f64(&row[2])?,
                    parse_f64(&row[3])?,
                ],
            })
        })
        .collect()
}

pub fn import_accel_data(path: &Path) -> Result<Vec<AccelSample>> {
    let rows = semicolon_rows(path, 3)?;
    rows.into_iter()
        .map(|row| {
            Ok(AccelSample {
                ttag_us: (parse_f64(&row[0])? / 1000.0).floor() as i64,
                accel_mps2: [
                    parse_f64(&row[1])?,
                    parse_f64(&row[2])?,
                    parse_f64(&row[3])?,
                ],
            })
        })
        .collect()
}

pub fn import_gnss_data(path: &Path) -> Result<Vec<GnssSample>> {
    let rows = semicolon_rows(path, 1)?;
    rows.into_iter()
        .map(|row| {
            Ok(GnssSample {
                ttag_us: (parse_f64(&row[0])? / 1000.0).floor() as i64,
                lat_deg: parse_f64(&row[2])?,
                lon_deg: parse_f64(&row[3])?,
                height_m: parse_f64(&row[4])?,
                speed_mps: parse_f64(&row[5])?,
                heading_deg: parse_f64(&row[6])?,
                h_acc_m: parse_f64(&row[7])?,
                v_acc_m: parse_f64(&row[8])?,
                speed_acc_mps: parse_f64(&row[9])?,
                velocity: None,
            })
        })
        .collect()
}

pub fn import_gnss_velocity_map(path: &Path) -> Result<Vec<(i64, GnssVelocity)>> {
    let rows = semicolon_rows(path, 1)?;
    rows.into_iter()
        .map(|row| {
            Ok((
                parse_f64(&row[0])?.round() as i64,
                GnssVelocity {
                    vel_n: [
                        parse_f64(&row[1])?,
                        parse_f64(&row[2])?,
                        parse_f64(&row[3])?,
                    ],
                    vel_acc_n: [
                        parse_f64(&row[4])?,
                        parse_f64(&row[5])?,
                        parse_f64(&row[6])?,
                    ],
                },
            ))
        })
        .collect()
}

pub fn import_truth_nav(path: &Path) -> Result<Vec<TruthNavSample>> {
    let rows = semicolon_rows(path, 1)?;
    rows.into_iter()
        .map(|row| {
            Ok(TruthNavSample {
                ttag_us: parse_f64(&row[0])?.round() as i64,
                pitch_car_deg: parse_f64(&row[11])?,
            })
        })
        .collect()
}

pub fn import_truth_misalignment(path: &Path) -> Result<[f64; 3]> {
    for row in semicolon_rows(path, 1)? {
        if row.first().is_some_and(|name| name == "misalignment_deg") {
            return Ok([
                parse_f64(&row[1])?,
                parse_f64(&row[2])?,
                parse_f64(&row[3])?,
            ]);
        }
    }
    bail!("misalignment_deg row not found in {}", path.display())
}

fn semicolon_rows(path: &Path, skip_rows: usize) -> Result<Vec<Vec<String>>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index < skip_rows {
            continue;
        }
        let row: Vec<String> = line
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect();
        if !row.is_empty() {
            out.push(row);
        }
    }
    Ok(out)
}

fn parse_f64(s: &str) -> Result<f64> {
    s.parse::<f64>()
        .with_context(|| format!("failed to parse float: {s}"))
}
