use anyhow::{Context, Result, bail};
use sensor_fusion::fusion::SensorFusion;

use crate::datasets::generic_replay::{
    GenericGnssSample, GenericImuSample, fusion_gnss_sample, fusion_imu_sample,
};
use crate::eval::gnss_ins::{as_q64, quat_conj, quat_mul};
use crate::eval::replay::{ReplayEvent, for_each_event};
use crate::visualizer::math::{ecef_to_ned, lla_to_ecef, ned_to_lla_exact, quat_rpy_deg};
use crate::visualizer::model::{EkfImuSource, HeadingSample, PlotData, Trace};
use crate::visualizer::pipeline::{EkfCompareConfig, GnssOutageConfig};

pub struct GenericReplayInput {
    pub imu: Vec<GenericImuSample>,
    pub gnss: Vec<GenericGnssSample>,
}

pub fn parse_generic_replay_csvs(imu_csv: &str, gnss_csv: &str) -> Result<GenericReplayInput> {
    let mut imu = parse_imu_csv(imu_csv)?;
    let mut gnss = parse_gnss_csv(gnss_csv)?;
    imu.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    gnss.sort_by(|a, b| {
        a.t_s
            .partial_cmp(&b.t_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if imu.is_empty() {
        bail!("imu.csv contained no samples");
    }
    if gnss.is_empty() {
        bail!("gnss.csv contained no samples");
    }
    Ok(GenericReplayInput { imu, gnss })
}

pub fn build_generic_replay_plot_data(
    replay: &GenericReplayInput,
    ekf_imu_source: EkfImuSource,
    ekf_cfg: EkfCompareConfig,
    gnss_outages: GnssOutageConfig,
) -> PlotData {
    let mut fusion = SensorFusion::new();
    apply_fusion_config(&mut fusion, ekf_cfg, ekf_imu_source);

    let ref_gnss = replay.gnss.first().copied();
    let ref_ecef = ref_gnss.map(|s| lla_to_ecef(s.lat_deg, s.lon_deg, s.height_m));
    let outage_windows = sample_outage_windows(&replay.gnss, gnss_outages);

    let mut raw_gyro_x = Vec::new();
    let mut raw_gyro_y = Vec::new();
    let mut raw_gyro_z = Vec::new();
    let mut raw_accel_x = Vec::new();
    let mut raw_accel_y = Vec::new();
    let mut raw_accel_z = Vec::new();
    let mut gnss_speed = Vec::new();
    let mut gnss_pos_n = Vec::new();
    let mut gnss_pos_e = Vec::new();
    let mut gnss_pos_d = Vec::new();
    let mut gnss_vel_n = Vec::new();
    let mut gnss_vel_e = Vec::new();
    let mut gnss_vel_d = Vec::new();
    let mut gnss_map = Vec::new();

    for sample in &replay.gnss {
        gnss_speed.push([
            sample.t_s,
            sample.vel_ned_mps[0].hypot(sample.vel_ned_mps[1]),
        ]);
        gnss_vel_n.push([sample.t_s, sample.vel_ned_mps[0]]);
        gnss_vel_e.push([sample.t_s, sample.vel_ned_mps[1]]);
        gnss_vel_d.push([sample.t_s, sample.vel_ned_mps[2]]);
        gnss_map.push([sample.lon_deg, sample.lat_deg]);
        if let (Some(ref_sample), Some(ref_ecef)) = (ref_gnss, ref_ecef) {
            let ecef = lla_to_ecef(sample.lat_deg, sample.lon_deg, sample.height_m);
            let ned = ecef_to_ned(ecef, ref_ecef, ref_sample.lat_deg, ref_sample.lon_deg);
            gnss_pos_n.push([sample.t_s, ned[0]]);
            gnss_pos_e.push([sample.t_s, ned[1]]);
            gnss_pos_d.push([sample.t_s, ned[2]]);
        }
    }

    let mut eskf_pos_n = Vec::new();
    let mut eskf_pos_e = Vec::new();
    let mut eskf_pos_d = Vec::new();
    let mut eskf_vel_n = Vec::new();
    let mut eskf_vel_e = Vec::new();
    let mut eskf_vel_d = Vec::new();
    let mut eskf_roll = Vec::new();
    let mut eskf_pitch = Vec::new();
    let mut eskf_yaw = Vec::new();
    let mut eskf_mount_roll = Vec::new();
    let mut eskf_mount_pitch = Vec::new();
    let mut eskf_mount_yaw = Vec::new();
    let mut eskf_bgx = Vec::new();
    let mut eskf_bgy = Vec::new();
    let mut eskf_bgz = Vec::new();
    let mut eskf_bax = Vec::new();
    let mut eskf_bay = Vec::new();
    let mut eskf_baz = Vec::new();
    let mut eskf_cov: [Vec<[f64; 2]>; 18] = std::array::from_fn(|_| Vec::new());
    let mut eskf_map = Vec::new();
    let mut eskf_outage_map = Vec::new();
    let mut eskf_heading = Vec::new();
    let mut mount_ready_marker = Vec::new();
    let mut ekf_init_marker = Vec::new();

    for_each_event(&replay.imu, &replay.gnss, |event| match event {
        ReplayEvent::Imu(_, sample) => {
            let _ = fusion.process_imu(fusion_imu_sample(*sample));
            raw_gyro_x.push([sample.t_s, sample.gyro_radps[0].to_degrees()]);
            raw_gyro_y.push([sample.t_s, sample.gyro_radps[1].to_degrees()]);
            raw_gyro_z.push([sample.t_s, sample.gyro_radps[2].to_degrees()]);
            raw_accel_x.push([sample.t_s, sample.accel_mps2[0]]);
            raw_accel_y.push([sample.t_s, sample.accel_mps2[1]]);
            raw_accel_z.push([sample.t_s, sample.accel_mps2[2]]);
            append_eskf_sample(
                sample.t_s,
                &fusion,
                ref_gnss,
                &mut eskf_pos_n,
                &mut eskf_pos_e,
                &mut eskf_pos_d,
                &mut eskf_vel_n,
                &mut eskf_vel_e,
                &mut eskf_vel_d,
                &mut eskf_roll,
                &mut eskf_pitch,
                &mut eskf_yaw,
                &mut eskf_mount_roll,
                &mut eskf_mount_pitch,
                &mut eskf_mount_yaw,
                &mut eskf_bgx,
                &mut eskf_bgy,
                &mut eskf_bgz,
                &mut eskf_bax,
                &mut eskf_bay,
                &mut eskf_baz,
                &mut eskf_cov,
                &mut eskf_map,
                &mut eskf_outage_map,
                &mut eskf_heading,
                in_outage(sample.t_s, &outage_windows),
            );
        }
        ReplayEvent::Gnss(_, sample) => {
            if !in_outage(sample.t_s, &outage_windows) {
                let update = fusion.process_gnss(fusion_gnss_sample(*sample));
                if update.mount_ready_changed && update.mount_ready {
                    mount_ready_marker.push([sample.t_s, 0.0]);
                }
                if update.ekf_initialized_now && update.ekf_initialized {
                    ekf_init_marker.push([sample.t_s, 0.0]);
                }
            }
        }
    });

    PlotData {
        speed: vec![Trace {
            name: "GNSS speed [m/s]".to_string(),
            points: gnss_speed,
        }],
        imu_raw_gyro: vec![
            Trace {
                name: "CSV gyro X [dps]".to_string(),
                points: raw_gyro_x,
            },
            Trace {
                name: "CSV gyro Y [dps]".to_string(),
                points: raw_gyro_y,
            },
            Trace {
                name: "CSV gyro Z [dps]".to_string(),
                points: raw_gyro_z,
            },
        ],
        imu_raw_accel: vec![
            Trace {
                name: "CSV accel X [m/s^2]".to_string(),
                points: raw_accel_x,
            },
            Trace {
                name: "CSV accel Y [m/s^2]".to_string(),
                points: raw_accel_y,
            },
            Trace {
                name: "CSV accel Z [m/s^2]".to_string(),
                points: raw_accel_z,
            },
        ],
        eskf_cmp_pos: vec![
            Trace {
                name: "GNSS posN [m]".to_string(),
                points: gnss_pos_n,
            },
            Trace {
                name: "ESKF posN [m]".to_string(),
                points: eskf_pos_n,
            },
            Trace {
                name: "GNSS posE [m]".to_string(),
                points: gnss_pos_e,
            },
            Trace {
                name: "ESKF posE [m]".to_string(),
                points: eskf_pos_e,
            },
            Trace {
                name: "GNSS posD [m]".to_string(),
                points: gnss_pos_d,
            },
            Trace {
                name: "ESKF posD [m]".to_string(),
                points: eskf_pos_d,
            },
        ],
        eskf_cmp_vel: vec![
            Trace {
                name: "GNSS velN [m/s]".to_string(),
                points: gnss_vel_n,
            },
            Trace {
                name: "ESKF velN [m/s]".to_string(),
                points: eskf_vel_n,
            },
            Trace {
                name: "GNSS velE [m/s]".to_string(),
                points: gnss_vel_e,
            },
            Trace {
                name: "ESKF velE [m/s]".to_string(),
                points: eskf_vel_e,
            },
            Trace {
                name: "GNSS velD [m/s]".to_string(),
                points: gnss_vel_d,
            },
            Trace {
                name: "ESKF velD [m/s]".to_string(),
                points: eskf_vel_d,
            },
        ],
        eskf_cmp_att: vec![
            Trace {
                name: "ESKF roll [deg]".to_string(),
                points: eskf_roll,
            },
            Trace {
                name: "ESKF pitch [deg]".to_string(),
                points: eskf_pitch,
            },
            Trace {
                name: "ESKF yaw [deg]".to_string(),
                points: eskf_yaw,
            },
            Trace {
                name: "mount ready".to_string(),
                points: mount_ready_marker,
            },
            Trace {
                name: "ekf initialized".to_string(),
                points: ekf_init_marker,
            },
        ],
        eskf_bias_gyro: vec![
            Trace {
                name: "ESKF bgx [dps]".to_string(),
                points: eskf_bgx,
            },
            Trace {
                name: "ESKF bgy [dps]".to_string(),
                points: eskf_bgy,
            },
            Trace {
                name: "ESKF bgz [dps]".to_string(),
                points: eskf_bgz,
            },
        ],
        eskf_bias_accel: vec![
            Trace {
                name: "ESKF bax [m/s^2]".to_string(),
                points: eskf_bax,
            },
            Trace {
                name: "ESKF bay [m/s^2]".to_string(),
                points: eskf_bay,
            },
            Trace {
                name: "ESKF baz [m/s^2]".to_string(),
                points: eskf_baz,
            },
        ],
        eskf_cov_nonbias: eskf_cov
            .iter()
            .enumerate()
            .map(|(i, points)| Trace {
                name: format!("ESKF sigma state {i}"),
                points: points.clone(),
            })
            .collect(),
        eskf_misalignment: vec![
            Trace {
                name: "ESKF mount roll [deg]".to_string(),
                points: eskf_mount_roll,
            },
            Trace {
                name: "ESKF mount pitch [deg]".to_string(),
                points: eskf_mount_pitch,
            },
            Trace {
                name: "ESKF mount yaw [deg]".to_string(),
                points: eskf_mount_yaw,
            },
        ],
        eskf_map: vec![
            Trace {
                name: "Generic GNSS path (lon,lat)".to_string(),
                points: gnss_map,
            },
            Trace {
                name: "ESKF path (lon,lat)".to_string(),
                points: eskf_map,
            },
            Trace {
                name: "ESKF path during GNSS outage (lon,lat)".to_string(),
                points: eskf_outage_map,
            },
        ],
        eskf_map_heading: eskf_heading,
        ..PlotData::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn append_eskf_sample(
    t_s: f64,
    fusion: &SensorFusion,
    ref_gnss: Option<GenericGnssSample>,
    pos_n: &mut Vec<[f64; 2]>,
    pos_e: &mut Vec<[f64; 2]>,
    pos_d: &mut Vec<[f64; 2]>,
    vel_n: &mut Vec<[f64; 2]>,
    vel_e: &mut Vec<[f64; 2]>,
    vel_d: &mut Vec<[f64; 2]>,
    roll: &mut Vec<[f64; 2]>,
    pitch: &mut Vec<[f64; 2]>,
    yaw: &mut Vec<[f64; 2]>,
    mount_roll: &mut Vec<[f64; 2]>,
    mount_pitch: &mut Vec<[f64; 2]>,
    mount_yaw: &mut Vec<[f64; 2]>,
    bgx: &mut Vec<[f64; 2]>,
    bgy: &mut Vec<[f64; 2]>,
    bgz: &mut Vec<[f64; 2]>,
    accel_bias_x: &mut Vec<[f64; 2]>,
    accel_bias_y: &mut Vec<[f64; 2]>,
    accel_bias_z: &mut Vec<[f64; 2]>,
    cov: &mut [Vec<[f64; 2]>; 18],
    map: &mut Vec<[f64; 2]>,
    outage_map: &mut Vec<[f64; 2]>,
    headings: &mut Vec<HeadingSample>,
    outage_active: bool,
) {
    let Some(eskf) = fusion.eskf() else {
        return;
    };
    pos_n.push([t_s, eskf.nominal.pn as f64]);
    pos_e.push([t_s, eskf.nominal.pe as f64]);
    pos_d.push([t_s, eskf.nominal.pd as f64]);
    vel_n.push([t_s, eskf.nominal.vn as f64]);
    vel_e.push([t_s, eskf.nominal.ve as f64]);
    vel_d.push([t_s, eskf.nominal.vd as f64]);

    let q_vehicle = eskf_vehicle_attitude_q(eskf);
    let (r, p, y) = quat_rpy_deg(
        q_vehicle[0] as f32,
        q_vehicle[1] as f32,
        q_vehicle[2] as f32,
        q_vehicle[3] as f32,
    );
    roll.push([t_s, r]);
    pitch.push([t_s, p]);
    yaw.push([t_s, y]);

    if let Some(q) = fusion.eskf_mount_q_vb().or_else(|| fusion.mount_q_vb()) {
        let (mr, mp, my) = quat_rpy_deg(q[0], q[1], q[2], q[3]);
        mount_roll.push([t_s, mr]);
        mount_pitch.push([t_s, mp]);
        mount_yaw.push([t_s, my]);
    }

    bgx.push([t_s, (eskf.nominal.bgx as f64).to_degrees()]);
    bgy.push([t_s, (eskf.nominal.bgy as f64).to_degrees()]);
    bgz.push([t_s, (eskf.nominal.bgz as f64).to_degrees()]);
    accel_bias_x.push([t_s, eskf.nominal.bax as f64]);
    accel_bias_y.push([t_s, eskf.nominal.bay as f64]);
    accel_bias_z.push([t_s, eskf.nominal.baz as f64]);
    for (i, trace) in cov.iter_mut().enumerate() {
        trace.push([t_s, eskf.p[i][i].max(0.0).sqrt() as f64]);
    }

    if let Some(ref_sample) = ref_gnss {
        let (lat, lon, _) = ned_to_lla_exact(
            eskf.nominal.pn as f64,
            eskf.nominal.pe as f64,
            eskf.nominal.pd as f64,
            ref_sample.lat_deg,
            ref_sample.lon_deg,
            ref_sample.height_m,
        );
        map.push([lon, lat]);
        if outage_active {
            outage_map.push([lon, lat]);
        } else if outage_map
            .last()
            .map(|p| p[0].is_finite() || p[1].is_finite())
            .unwrap_or(false)
        {
            outage_map.push([f64::NAN, f64::NAN]);
        }
        headings.push(HeadingSample {
            t_s,
            lon_deg: lon,
            lat_deg: lat,
            yaw_deg: y,
        });
    }
}

fn eskf_vehicle_attitude_q(eskf: &sensor_fusion::eskf_types::EskfState) -> [f64; 4] {
    let q_seed_frame = as_q64([
        eskf.nominal.q0,
        eskf.nominal.q1,
        eskf.nominal.q2,
        eskf.nominal.q3,
    ]);
    let q_cs = as_q64([
        eskf.nominal.qcs0,
        eskf.nominal.qcs1,
        eskf.nominal.qcs2,
        eskf.nominal.qcs3,
    ]);
    quat_mul(q_seed_frame, quat_conj(q_cs))
}

fn apply_fusion_config(fusion: &mut SensorFusion, cfg: EkfCompareConfig, mode: EkfImuSource) {
    fusion.set_r_body_vel(cfg.r_body_vel);
    fusion.set_gnss_pos_mount_scale(cfg.gnss_pos_mount_scale);
    fusion.set_gnss_vel_mount_scale(cfg.gnss_vel_mount_scale);
    fusion.set_yaw_init_sigma_rad(cfg.yaw_init_sigma_deg.to_radians());
    fusion.set_gyro_bias_init_sigma_radps(cfg.gyro_bias_init_sigma_dps.to_radians());
    fusion.set_accel_bias_init_sigma_mps2(cfg.accel_bias_init_sigma_mps2);
    fusion.set_mount_init_sigma_rad(cfg.mount_init_sigma_deg.to_radians());
    fusion.set_r_vehicle_speed(cfg.r_vehicle_speed);
    fusion.set_r_zero_vel(cfg.r_zero_vel);
    fusion.set_r_stationary_accel(cfg.r_stationary_accel);
    fusion.set_mount_align_rw_var(cfg.mount_align_rw_var);
    fusion.set_mount_update_min_scale(cfg.mount_update_min_scale);
    fusion.set_mount_update_ramp_time_s(cfg.mount_update_ramp_time_s);
    fusion.set_mount_update_innovation_gate_mps(cfg.mount_update_innovation_gate_mps);
    fusion.set_mount_update_yaw_rate_gate_radps(cfg.mount_update_yaw_rate_gate_dps.to_radians());
    fusion.set_align_handoff_delay_s(cfg.align_handoff_delay_s);
    fusion.set_freeze_misalignment_states(cfg.freeze_misalignment_states);
    fusion.set_eskf_mount_source(mode.eskf_mount_source());
    fusion.set_mount_settle_time_s(cfg.mount_settle_time_s);
    fusion.set_mount_settle_release_sigma_rad(cfg.mount_settle_release_sigma_deg.to_radians());
    fusion.set_mount_settle_zero_cross_covariance(cfg.mount_settle_zero_cross_covariance);
}

fn parse_imu_csv(text: &str) -> Result<Vec<GenericImuSample>> {
    let rows = parse_numeric_rows(text, 7, "imu.csv")?;
    Ok(rows
        .into_iter()
        .map(|row| GenericImuSample {
            t_s: row[0],
            gyro_radps: [row[1], row[2], row[3]],
            accel_mps2: [row[4], row[5], row[6]],
        })
        .collect())
}

fn parse_gnss_csv(text: &str) -> Result<Vec<GenericGnssSample>> {
    let rows = parse_numeric_rows_range(text, 13..=14, "gnss.csv")?;
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
            heading_rad: row.get(13).copied().filter(|v| v.is_finite()),
        })
        .collect())
}

fn parse_numeric_rows(text: &str, cols: usize, label: &str) -> Result<Vec<Vec<f64>>> {
    parse_numeric_rows_range(text, cols..=cols, label)
}

fn parse_numeric_rows_range(
    text: &str,
    cols: std::ops::RangeInclusive<usize>,
    label: &str,
) -> Result<Vec<Vec<f64>>> {
    let mut out = Vec::new();
    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parsed = trimmed
            .split(',')
            .map(|part| {
                let value = part.trim();
                if value.eq_ignore_ascii_case("nan") {
                    Ok(f64::NAN)
                } else {
                    value
                        .parse::<f64>()
                        .with_context(|| format!("{label}: bad numeric field '{value}'"))
                }
            })
            .collect::<Result<Vec<_>>>();
        let row = match parsed {
            Ok(row) => row,
            Err(err) if out.is_empty() => {
                let _ = err;
                continue;
            }
            Err(err) => return Err(err),
        };
        if !cols.contains(&row.len()) {
            bail!(
                "{label}: line {} expected {} columns, got {}",
                line_idx + 1,
                if cols.start() == cols.end() {
                    cols.start().to_string()
                } else {
                    format!("{}..={}", cols.start(), cols.end())
                },
                row.len()
            );
        }
        out.push(row);
    }
    Ok(out)
}

fn sample_outage_windows(gnss: &[GenericGnssSample], cfg: GnssOutageConfig) -> Vec<(f64, f64)> {
    if cfg.count == 0 || cfg.duration_s <= 0.0 || gnss.len() < 2 {
        return Vec::new();
    }
    let t_min = gnss.first().map(|s| s.t_s).unwrap_or(0.0);
    let t_max = gnss.last().map(|s| s.t_s).unwrap_or(t_min);
    if t_max - t_min <= cfg.duration_s {
        return Vec::new();
    }
    let mut rng = Lcg64::new(cfg.seed);
    let mut windows = Vec::new();
    let max_attempts = cfg.count.saturating_mul(200).max(200);
    for _ in 0..max_attempts {
        if windows.len() >= cfg.count {
            break;
        }
        let start = t_min + rng.next_unit_f64() * (t_max - t_min - cfg.duration_s);
        let end = start + cfg.duration_s;
        if windows.iter().any(|(a, b)| start < *b && end > *a) {
            continue;
        }
        windows.push((start, end));
    }
    windows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    windows
}

fn in_outage(t_s: f64, windows: &[(f64, f64)]) -> bool {
    windows.iter().any(|(a, b)| t_s >= *a && t_s <= *b)
}

struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_unit_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = self.state >> 11;
        (v as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}
