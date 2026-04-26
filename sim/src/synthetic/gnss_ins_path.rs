use anyhow::{Context, Result, bail};
use std::fs;
use std::path::Path;

use crate::datasets::gnss_ins_sim::{GnssSample, ImuSample, TruthSample};

const D2R: f64 = std::f64::consts::PI / 180.0;
const RE: f64 = 6378137.0;
const FLATTENING: f64 = 1.0 / 298.257223563;
const ECCENTRICITY: f64 = 0.0818191908426215;
const E_SQR: f64 = ECCENTRICITY * ECCENTRICITY;
const W_IE: f64 = 7292115.0e-11;

#[derive(Clone, Copy, Debug)]
pub struct InitialState {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
    pub vel_body_mps: [f64; 3],
    pub yaw_pitch_roll_deg: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct MotionCommand {
    pub command_type: u8,
    pub yaw_pitch_roll_cmd_deg: [f64; 3],
    pub body_cmd: [f64; 3],
    pub duration_s: f64,
    pub gps_visible: bool,
}

#[derive(Clone, Debug)]
pub struct MotionProfile {
    pub initial: InitialState,
    pub commands: Vec<MotionCommand>,
}

#[derive(Clone, Copy, Debug)]
pub struct PathGenConfig {
    pub imu_hz: f64,
    pub gnss_hz: f64,
    pub sim_osr: u32,
    pub max_accel_mps2: f64,
    pub max_angular_accel_radps2: f64,
    pub max_angular_rate_radps: f64,
}

impl Default for PathGenConfig {
    fn default() -> Self {
        Self {
            imu_hz: 100.0,
            gnss_hz: 2.0,
            sim_osr: 1,
            max_accel_mps2: 10.0,
            max_angular_accel_radps2: 0.5,
            max_angular_rate_radps: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GeneratedPath {
    pub imu: Vec<ImuSample>,
    pub gnss: Vec<GnssSample>,
    pub truth: Vec<TruthSample>,
}

impl MotionProfile {
    pub fn from_csv(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        Self::from_csv_str(&text)
    }

    pub fn from_csv_str(text: &str) -> Result<Self> {
        let rows = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<_>>();
        if rows.len() < 4 {
            bail!("motion profile must contain initial state and at least one command");
        }
        let initial_row = parse_csv_numbers(rows[1], 9)?;
        let initial = InitialState {
            lat_deg: initial_row[0],
            lon_deg: initial_row[1],
            height_m: initial_row[2],
            vel_body_mps: [initial_row[3], initial_row[4], initial_row[5]],
            yaw_pitch_roll_deg: [initial_row[6], initial_row[7], initial_row[8]],
        };
        let mut commands = Vec::new();
        for line in rows.iter().skip(3) {
            let row = parse_csv_numbers(line, 9)?;
            commands.push(MotionCommand {
                command_type: row[0].round() as u8,
                yaw_pitch_roll_cmd_deg: [row[1], row[2], row[3]],
                body_cmd: [row[4], row[5], row[6]],
                duration_s: row[7],
                gps_visible: row[8] != 0.0,
            });
        }
        Ok(Self { initial, commands })
    }
}

pub fn generate(profile: &MotionProfile, cfg: PathGenConfig) -> Result<GeneratedPath> {
    if cfg.imu_hz <= 0.0 || cfg.gnss_hz <= 0.0 || cfg.sim_osr == 0 {
        bail!("invalid synthetic path generation frequencies");
    }
    let out_freq = cfg.imu_hz;
    let sim_osr = cfg.sim_osr as usize;
    let sim_freq = out_freq * cfg.sim_osr as f64;
    let dt = 1.0 / sim_freq;
    let gnss_period = sim_osr * (out_freq / cfg.gnss_hz).round() as usize;
    if gnss_period == 0 {
        bail!("GNSS period rounded to zero");
    }

    let alpha = 0.9;
    let kp = 5.0;
    let kd = 10.0;
    let att_converge_threshold = 1.0e-4;
    let vel_converge_threshold = 1.0e-4;

    let pos_n = [
        profile.initial.lat_deg * D2R,
        profile.initial.lon_deg * D2R,
        profile.initial.height_m,
    ];
    let mut pos_delta_n = [0.0; 3];
    let mut vel_b = profile.initial.vel_body_mps;
    let mut att = [
        profile.initial.yaw_pitch_roll_deg[0] * D2R,
        profile.initial.yaw_pitch_roll_deg[1] * D2R,
        profile.initial.yaw_pitch_roll_deg[2] * D2R,
    ];
    let mut c_nb = transpose3(euler2dcm_zyx(att));
    let mut vel_n = mat3_vec(c_nb, vel_b);

    let sample_capacity = profile
        .commands
        .iter()
        .map(|cmd| (cmd.duration_s * out_freq).ceil() as usize)
        .sum();
    let mut imu = Vec::with_capacity(sample_capacity);
    let mut truth = Vec::with_capacity(sample_capacity);
    let mut gnss = Vec::with_capacity(sample_capacity / gnss_period.max(1) + 1);

    let mut sim_count = 0usize;
    let mut att_dot = [0.0; 3];
    let mut vel_dot_b = [0.0; 3];
    let mut acc_sum = [0.0; 3];
    let mut gyro_sum = [0.0; 3];

    for command in &profile.commands {
        if command.duration_s < 0.0 {
            bail!("motion command has negative duration");
        }
        let command_sim_cycles =
            (command.duration_s * out_freq * cfg.sim_osr as f64).round() as usize;
        let sim_count_max = sim_count + command_sim_cycles;
        let motion = parse_motion_command(command, att, vel_b);
        let mut att_com_filt = att;
        let mut vel_com_b_filt = vel_b;
        let mut complete = false;

        while sim_count < sim_count_max && !complete {
            if command.command_type == 1 {
                att_dot = add3(scale3(att_dot, alpha), scale3(motion.att, 1.0 - alpha));
                vel_dot_b = add3(scale3(vel_dot_b, alpha), scale3(motion.vel, 1.0 - alpha));
            } else {
                att_com_filt = add3(scale3(att_com_filt, alpha), scale3(motion.att, 1.0 - alpha));
                vel_com_b_filt = add3(
                    scale3(vel_com_b_filt, alpha),
                    scale3(motion.vel, 1.0 - alpha),
                );
                vel_dot_b = div3(sub3(vel_com_b_filt, vel_b), dt);
                clamp3(&mut vel_dot_b, cfg.max_accel_mps2);

                let mut att_dot_dot = add3(
                    scale3(sub3(motion.att, att), kp),
                    scale3(scale3(att_dot, -1.0), kd),
                );
                clamp3(&mut att_dot_dot, cfg.max_angular_accel_radps2);
                att_dot = add3(att_dot, scale3(att_dot_dot, dt));
                clamp3(&mut att_dot, cfg.max_angular_rate_radps);

                if norm3(sub3(att, motion.att)) < att_converge_threshold
                    && norm3(sub3(vel_b, motion.vel)) < vel_converge_threshold
                {
                    complete = true;
                }
            }

            let sensor = calc_true_sensor_output(
                add3(pos_n, pos_delta_n),
                vel_b,
                att,
                c_nb,
                vel_dot_b,
                att_dot,
            );
            acc_sum = add3(acc_sum, sensor.accel_body_mps2);
            gyro_sum = add3(gyro_sum, sensor.gyro_body_radps);

            if sim_count % sim_osr == 0 {
                let t_s = sim_count as f64 / out_freq;
                let acc_avg = div3(acc_sum, sim_osr as f64);
                let gyro_avg = div3(gyro_sum, sim_osr as f64);
                let pos = add3(pos_n, pos_delta_n);
                let euler = euler_angle_range_three_axis(att);
                imu.push(ImuSample {
                    t_s,
                    gyro_vehicle_radps: gyro_avg,
                    accel_vehicle_mps2: acc_avg,
                });
                truth.push(TruthSample {
                    t_s,
                    lat_deg: pos[0] / D2R,
                    lon_deg: pos[1] / D2R,
                    height_m: pos[2],
                    vel_ned_mps: vel_n,
                    q_bn: euler2quat_zyx(euler),
                });
                acc_sum = [0.0; 3];
                gyro_sum = [0.0; 3];
            }

            if sim_count % gnss_period == 0 {
                let pos = add3(pos_n, pos_delta_n);
                gnss.push(GnssSample {
                    t_s: sim_count as f64 / out_freq,
                    lat_deg: pos[0] / D2R,
                    lon_deg: pos[1] / D2R,
                    height_m: pos[2],
                    vel_ned_mps: vel_n,
                });
            }

            pos_delta_n = add3(pos_delta_n, scale3(sensor.pos_dot_n, dt));
            vel_b = add3(vel_b, scale3(vel_dot_b, dt));
            att = add3(att, scale3(att_dot, dt));
            c_nb = transpose3(euler2dcm_zyx(att));
            vel_n = mat3_vec(c_nb, vel_b);
            sim_count += 1;
        }

        if complete {
            att_dot = [0.0; 3];
            vel_dot_b = [0.0; 3];
        }
    }

    Ok(GeneratedPath { imu, gnss, truth })
}

#[derive(Clone, Copy, Debug)]
struct ParsedMotion {
    att: [f64; 3],
    vel: [f64; 3],
}

fn parse_motion_command(command: &MotionCommand, att: [f64; 3], vel: [f64; 3]) -> ParsedMotion {
    let cmd_att = [
        command.yaw_pitch_roll_cmd_deg[0] * D2R,
        command.yaw_pitch_roll_cmd_deg[1] * D2R,
        command.yaw_pitch_roll_cmd_deg[2] * D2R,
    ];
    let cmd_vel = command.body_cmd;
    match command.command_type {
        1 | 2 => ParsedMotion {
            att: cmd_att,
            vel: cmd_vel,
        },
        3 => ParsedMotion {
            att: add3(att, cmd_att),
            vel: add3(vel, cmd_vel),
        },
        4 => ParsedMotion {
            att: cmd_att,
            vel: add3(vel, cmd_vel),
        },
        5 => ParsedMotion {
            att: add3(att, cmd_att),
            vel: cmd_vel,
        },
        _ => ParsedMotion {
            att: cmd_att,
            vel: cmd_vel,
        },
    }
}

#[derive(Clone, Copy, Debug)]
struct SensorOutput {
    accel_body_mps2: [f64; 3],
    gyro_body_radps: [f64; 3],
    pos_dot_n: [f64; 3],
}

fn calc_true_sensor_output(
    pos_n: [f64; 3],
    vel_b: [f64; 3],
    att: [f64; 3],
    c_nb: [[f64; 3]; 3],
    vel_dot_b: [f64; 3],
    att_dot: [f64; 3],
) -> SensorOutput {
    let vel_n = mat3_vec(c_nb, vel_b);
    let earth = geo_param(pos_n);
    let rm_effective = earth.rm + pos_n[2];
    let rn_effective = earth.rn + pos_n[2];
    let gravity = [0.0, 0.0, earth.g];
    let w_en_n = [
        vel_n[1] / rn_effective,
        -vel_n[0] / rm_effective,
        -vel_n[1] * earth.sl / earth.cl / rn_effective,
    ];
    let w_ie_n = [earth.w_ie * earth.cl, 0.0, -earth.w_ie * earth.sl];

    let (sh, ch) = att[0].sin_cos();
    let w_nb_n = [
        -sh * att_dot[1] + c_nb[0][0] * att_dot[2],
        ch * att_dot[1] + c_nb[1][0] * att_dot[2],
        att_dot[0] + c_nb[2][0] * att_dot[2],
    ];
    let _vel_dot_n = add3(mat3_vec(c_nb, vel_dot_b), cross3(w_nb_n, vel_n));
    let pos_dot_n = [
        vel_n[0] / rm_effective,
        vel_n[1] / rn_effective / earth.cl,
        -vel_n[2],
    ];

    let c_bn = transpose3(c_nb);
    let gyro = mat3_vec(c_bn, add3(add3(w_nb_n, w_en_n), w_ie_n));
    let w_ie_b = mat3_vec(c_bn, w_ie_n);
    let acc = sub3(
        add3(vel_dot_b, cross3(add3(w_ie_b, gyro), vel_b)),
        mat3_vec(c_bn, gravity),
    );
    SensorOutput {
        accel_body_mps2: acc,
        gyro_body_radps: gyro,
        pos_dot_n,
    }
}

#[derive(Clone, Copy, Debug)]
struct GeoParam {
    rm: f64,
    rn: f64,
    g: f64,
    sl: f64,
    cl: f64,
    w_ie: f64,
}

fn geo_param(pos: [f64; 3]) -> GeoParam {
    let normal_gravity = 9.7803253359;
    let k = 0.00193185265241;
    let m = 0.00344978650684;
    let sl = pos[0].sin();
    let cl = pos[0].cos();
    let sl_sqr = sl * sl;
    let h = pos[2];
    let sqrt_term = (1.0 - E_SQR * sl_sqr).sqrt();
    let rm = (RE * (1.0 - E_SQR)) / (sqrt_term * (1.0 - E_SQR * sl_sqr));
    let rn = RE / sqrt_term;
    let g1 = normal_gravity * (1.0 + k * sl_sqr) / sqrt_term;
    let g = g1
        * (1.0 - (2.0 / RE) * (1.0 + FLATTENING + m - 2.0 * FLATTENING * sl_sqr) * h
            + 3.0 * h * h / RE / RE);
    GeoParam {
        rm,
        rn,
        g,
        sl,
        cl,
        w_ie: W_IE,
    }
}

fn euler2dcm_zyx(angles: [f64; 3]) -> [[f64; 3]; 3] {
    let c = [angles[0].cos(), angles[1].cos(), angles[2].cos()];
    let s = [angles[0].sin(), angles[1].sin(), angles[2].sin()];
    [
        [c[1] * c[0], c[1] * s[0], -s[1]],
        [
            s[2] * s[1] * c[0] - c[2] * s[0],
            s[2] * s[1] * s[0] + c[2] * c[0],
            c[1] * s[2],
        ],
        [
            s[1] * c[2] * c[0] + s[0] * s[2],
            s[1] * c[2] * s[0] - c[0] * s[2],
            c[1] * c[2],
        ],
    ]
}

fn euler2quat_zyx(angles: [f64; 3]) -> [f64; 4] {
    let c = [
        (0.5 * angles[0]).cos(),
        (0.5 * angles[1]).cos(),
        (0.5 * angles[2]).cos(),
    ];
    let s = [
        (0.5 * angles[0]).sin(),
        (0.5 * angles[1]).sin(),
        (0.5 * angles[2]).sin(),
    ];
    [
        c[0] * c[1] * c[2] + s[0] * s[1] * s[2],
        c[0] * c[1] * s[2] - s[0] * s[1] * c[2],
        c[0] * s[1] * c[2] + s[0] * c[1] * s[2],
        s[0] * c[1] * c[2] - c[0] * s[1] * s[2],
    ]
}

fn euler_angle_range_three_axis(angles: [f64; 3]) -> [f64; 3] {
    let half_pi = 0.5 * std::f64::consts::PI;
    let mut a1 = angles[0];
    let mut a2 = angle_range_pi(angles[1]);
    let mut a3 = angles[2];
    if a2 > half_pi {
        a2 = std::f64::consts::PI - a2;
        a1 += std::f64::consts::PI;
        a3 += std::f64::consts::PI;
    } else if a2 < -half_pi {
        a2 = -std::f64::consts::PI - a2;
        a1 += std::f64::consts::PI;
        a3 += std::f64::consts::PI;
    }
    [angle_range_pi(a1), a2, angle_range_pi(a3)]
}

fn angle_range_pi(x: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut out = x % two_pi;
    if out > std::f64::consts::PI {
        out -= two_pi;
    }
    out
}

fn parse_csv_numbers(line: &str, expected: usize) -> Result<Vec<f64>> {
    let values = line
        .split(',')
        .map(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                Ok(0.0)
            } else {
                trimmed
                    .parse::<f64>()
                    .with_context(|| format!("failed to parse CSV value: {trimmed}"))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    if values.len() != expected {
        bail!("expected {expected} CSV columns, got {}", values.len());
    }
    Ok(values)
}

fn mat3_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn transpose3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn div3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] / s, v[1] / s, v[2] / s]
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn clamp3(v: &mut [f64; 3], limit: f64) {
    for value in v {
        *value = value.clamp(-limit, limit);
    }
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
