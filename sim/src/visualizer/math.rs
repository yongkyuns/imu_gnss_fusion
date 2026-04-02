pub fn normalize_heading_deg(mut deg: f64) -> f64 {
    deg %= 360.0;
    if deg < 0.0 {
        deg += 360.0;
    }
    deg
}

pub fn nearest_master_ms(seq: u64, masters: &[(u64, f64)]) -> Option<f64> {
    if masters.is_empty() {
        return None;
    }
    let idx = masters.partition_point(|(s, _)| *s < seq);
    if idx == 0 {
        return Some(masters[0].1);
    }
    if idx >= masters.len() {
        return Some(masters[masters.len() - 1].1);
    }
    let (sl, ml) = masters[idx - 1];
    let (sr, mr) = masters[idx];
    let dl = sl.abs_diff(seq);
    let dr = sr.abs_diff(seq);
    if dr < dl { Some(mr) } else { Some(ml) }
}

pub fn deg2rad(v: f64) -> f64 {
    v * std::f64::consts::PI / 180.0
}

pub fn rad2deg(v: f64) -> f64 {
    v * 180.0 / std::f64::consts::PI
}

pub fn rot_xyz(roll_rad: f64, pitch_rad: f64, yaw_rad: f64) -> [[f64; 3]; 3] {
    let (sr, cr) = roll_rad.sin_cos();
    let (sp, cp) = pitch_rad.sin_cos();
    let (sy, cy) = yaw_rad.sin_cos();
    [
        [cp * cy, -cp * sy, sp],
        [cr * sy + sr * sp * cy, cr * cy - sr * sp * sy, -sr * cp],
        [sr * sy - cr * sp * cy, sr * cy + cr * sp * sy, cr * cp],
    ]
}

pub fn rot_zyx(yaw_rad: f64, pitch_rad: f64, roll_rad: f64) -> [[f64; 3]; 3] {
    let (sy, cy) = yaw_rad.sin_cos();
    let (sp, cp) = pitch_rad.sin_cos();
    let (sr, cr) = roll_rad.sin_cos();
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

pub fn mat_vec(r: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

pub fn quat_rpy_deg(q0: f32, q1: f32, q2: f32, q3: f32) -> (f64, f64, f64) {
    let qw = q0 as f64;
    let qx = q1 as f64;
    let qy = q2 as f64;
    let qz = q3 as f64;
    let sinr_cosp = 2.0 * (qw * qx + qy * qz);
    let cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    let roll = sinr_cosp.atan2(cosr_cosp);
    let sinp = 2.0 * (qw * qy - qz * qx);
    let pitch = if sinp.abs() >= 1.0 {
        sinp.signum() * std::f64::consts::FRAC_PI_2
    } else {
        sinp.asin()
    };
    let siny_cosp = 2.0 * (qw * qz + qx * qy);
    let cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    let yaw = siny_cosp.atan2(cosy_cosp);
    (
        rad2deg(roll),
        rad2deg(pitch),
        normalize_heading_deg(rad2deg(yaw)),
    )
}

pub fn lla_to_ecef(lat_deg: f64, lon_deg: f64, h_m: f64) -> [f64; 3] {
    let a = 6378137.0_f64;
    let e2 = 6.69437999014e-3_f64;
    let lat = deg2rad(lat_deg);
    let lon = deg2rad(lon_deg);
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    let n = a / (1.0 - e2 * slat * slat).sqrt();
    [
        (n + h_m) * clat * clon,
        (n + h_m) * clat * slon,
        (n * (1.0 - e2) + h_m) * slat,
    ]
}

pub fn ecef_to_ned(
    ecef: [f64; 3],
    ref_ecef: [f64; 3],
    ref_lat_deg: f64,
    ref_lon_deg: f64,
) -> [f64; 3] {
    let lat = deg2rad(ref_lat_deg);
    let lon = deg2rad(ref_lon_deg);
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    let dx = ecef[0] - ref_ecef[0];
    let dy = ecef[1] - ref_ecef[1];
    let dz = ecef[2] - ref_ecef[2];
    [
        -slat * clon * dx - slat * slon * dy + clat * dz,
        -slon * dx + clon * dy,
        -clat * clon * dx - clat * slon * dy - slat * dz,
    ]
}

pub fn ned_to_lla_approx(
    n: f64,
    e: f64,
    d: f64,
    ref_lat_deg: f64,
    ref_lon_deg: f64,
    ref_h_m: f64,
) -> (f64, f64, f64) {
    let a = 6378137.0_f64;
    let e2 = 6.69437999014e-3_f64;
    let lat0 = deg2rad(ref_lat_deg);
    let sin_lat = lat0.sin();
    let denom = (1.0 - e2 * sin_lat * sin_lat).sqrt();
    let rn = a / denom;
    let rm = a * (1.0 - e2) / (denom * denom * denom);
    let dlat = n / (rm + ref_h_m);
    let dlon = e / ((rn + ref_h_m) * lat0.cos().max(1e-6));
    let lat = ref_lat_deg + rad2deg(dlat);
    let lon = ref_lon_deg + rad2deg(dlon);
    let h = ref_h_m - d;
    (lat, lon, h)
}

pub fn ned_to_lla_exact(
    n: f64,
    e: f64,
    d: f64,
    ref_lat_deg: f64,
    ref_lon_deg: f64,
    ref_h_m: f64,
) -> (f64, f64, f64) {
    let ref_ecef = lla_to_ecef(ref_lat_deg, ref_lon_deg, ref_h_m);
    let ecef = ned_to_ecef(n, e, d, ref_ecef, ref_lat_deg, ref_lon_deg);
    ecef_to_lla(ecef)
}

pub fn ned_to_ecef(
    n: f64,
    e: f64,
    d: f64,
    ref_ecef: [f64; 3],
    ref_lat_deg: f64,
    ref_lon_deg: f64,
) -> [f64; 3] {
    let lat = deg2rad(ref_lat_deg);
    let lon = deg2rad(ref_lon_deg);
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    let r_ned_to_ecef = [
        [-slat * clon, -slon, -clat * clon],
        [-slat * slon, clon, -clat * slon],
        [clat, 0.0, -slat],
    ];
    [
        ref_ecef[0] + r_ned_to_ecef[0][0] * n + r_ned_to_ecef[0][1] * e + r_ned_to_ecef[0][2] * d,
        ref_ecef[1] + r_ned_to_ecef[1][0] * n + r_ned_to_ecef[1][1] * e + r_ned_to_ecef[1][2] * d,
        ref_ecef[2] + r_ned_to_ecef[2][0] * n + r_ned_to_ecef[2][1] * e + r_ned_to_ecef[2][2] * d,
    ]
}

pub fn ecef_to_lla(ecef: [f64; 3]) -> (f64, f64, f64) {
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
    (rad2deg(lat), rad2deg(lon), h)
}

pub fn heading_endpoint(lat_deg: f64, lon_deg: f64, heading_deg: f64, length_m: f64) -> (f64, f64) {
    let r = 6_378_137.0_f64;
    let h = deg2rad(heading_deg);
    let d_n = length_m * h.cos();
    let d_e = length_m * h.sin();
    let d_lat = d_n / r;
    let d_lon = d_e / (r * deg2rad(lat_deg).cos().max(1e-6));
    (lat_deg + rad2deg(d_lat), lon_deg + rad2deg(d_lon))
}

pub fn unwrap_i64_counter(values: &[i64], modulus: i64) -> Vec<i64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut offset = 0i64;
    let mut prev = values[0];
    out.push(prev);
    for &v in values.iter().skip(1) {
        if v < prev && (prev - v) > (modulus / 2) {
            offset = offset.saturating_add(modulus);
        }
        out.push(v.saturating_add(offset));
        prev = v;
    }
    out
}
