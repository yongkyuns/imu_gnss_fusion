#[derive(Debug, Clone, Copy)]
pub struct MisalignCoarseConfig {
    pub min_speed_mps: f32,
    pub min_horiz_acc_mps2: f32,
    pub min_windows: usize,
    pub min_anisotropy_ratio: f32,
}

impl Default for MisalignCoarseConfig {
    fn default() -> Self {
        Self {
            min_speed_mps: 5.0 / 3.6,
            min_horiz_acc_mps2: 0.15,
            min_windows: 10,
            min_anisotropy_ratio: 1.3,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MisalignCoarseSample {
    pub speed_mps: f32,
    pub accel_b_mps2: [f32; 3],
    pub gnss_long_mps2: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct MisalignCoarseResult {
    pub mount_yaw_rad: f32,
    pub pca_axis_rad: f32,
    pub anisotropy_ratio: f32,
    pub used_windows: usize,
}

pub fn estimate_mount_yaw_from_tilt(
    cfg: MisalignCoarseConfig,
    q_vb_tilt: [f32; 4],
    samples: &[MisalignCoarseSample],
) -> Option<MisalignCoarseResult> {
    let (roll_rad, pitch_rad, _) = rot_to_euler_zyx(quat_to_rotmat(q_vb_tilt));
    let c_level_b = rot_from_euler_zyx(roll_rad, pitch_rad, 0.0);

    let mut used = Vec::with_capacity(samples.len());
    for sample in samples {
        if sample.speed_mps < cfg.min_speed_mps {
            continue;
        }
        let accel_level = mat3_vec(c_level_b, sample.accel_b_mps2);
        let horiz_xy = [accel_level[0], accel_level[1]];
        if vec2_norm(horiz_xy) < cfg.min_horiz_acc_mps2 {
            continue;
        }
        used.push(StoredSample {
            horiz_xy,
            gnss_long_mps2: sample.gnss_long_mps2,
        });
    }
    if used.len() < cfg.min_windows {
        return None;
    }

    let (mut theta_rad, anisotropy_ratio) = principal_axis_angle(&used)?;
    if anisotropy_ratio < cfg.min_anisotropy_ratio {
        return None;
    }
    let axis = [theta_rad.cos(), theta_rad.sin()];
    let corr = used
        .iter()
        .map(|s| s.gnss_long_mps2 * dot2(s.horiz_xy, axis))
        .sum::<f32>();
    if corr < 0.0 {
        theta_rad = wrap_pi(theta_rad + std::f32::consts::PI);
    }

    Some(MisalignCoarseResult {
        mount_yaw_rad: wrap_pi(-theta_rad),
        pca_axis_rad: theta_rad,
        anisotropy_ratio,
        used_windows: used.len(),
    })
}

#[derive(Debug, Clone, Copy)]
struct StoredSample {
    horiz_xy: [f32; 2],
    gnss_long_mps2: f32,
}

fn principal_axis_angle(samples: &[StoredSample]) -> Option<(f32, f32)> {
    if samples.is_empty() {
        return None;
    }
    let mut sxx = 0.0_f32;
    let mut sxy = 0.0_f32;
    let mut syy = 0.0_f32;
    for s in samples {
        let x = s.horiz_xy[0];
        let y = s.horiz_xy[1];
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }
    let trace = sxx + syy;
    let disc = ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt();
    let lambda_max = 0.5 * (trace + disc);
    let lambda_min = 0.5 * (trace - disc).max(0.0);
    if lambda_max <= 1.0e-6 {
        return None;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let anisotropy = lambda_max / lambda_min.max(1.0e-6);
    Some((theta, anisotropy))
}

fn dot2(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn vec2_norm(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn quat_to_rotmat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = quat_normalize(q);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 1.0e-9 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

fn rot_to_euler_zyx(c: [[f32; 3]; 3]) -> (f32, f32, f32) {
    let pitch = (-c[2][0]).asin();
    let roll = c[2][1].atan2(c[2][2]);
    let yaw = c[1][0].atan2(c[0][0]);
    (roll, pitch, yaw)
}

fn rot_from_euler_zyx(roll: f32, pitch: f32, yaw: f32) -> [[f32; 3]; 3] {
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

fn mat3_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn wrap_pi(x: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    (x + std::f32::consts::PI).rem_euclid(two_pi) - std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_mount_yaw_from_leveled_accel_axis() {
        let yaw = 20.0_f32.to_radians();
        let q_vb_tilt = [1.0, 0.0, 0.0, 0.0];
        let axis = [(-yaw).cos(), (-yaw).sin()];
        let mags = [1.0_f32, -0.6, 1.2, -0.9, 0.8, -1.1, 0.7, -0.5, 1.3, -1.0];
        let samples: Vec<_> = mags
            .into_iter()
            .map(|m| MisalignCoarseSample {
                speed_mps: 8.0,
                accel_b_mps2: [m * axis[0], m * axis[1], 0.0],
                gnss_long_mps2: m,
            })
            .collect();
        let out =
            estimate_mount_yaw_from_tilt(MisalignCoarseConfig::default(), q_vb_tilt, &samples)
                .expect("coarse yaw should resolve");
        assert!((wrap_pi(out.mount_yaw_rad - yaw)).abs() < 5.0_f32.to_radians());
    }
}
