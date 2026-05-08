//! Shared WGS84 and ECEF navigation helpers.

use crate::math::{atan2_f64, cos_f64, dcm_to_quat_f64, sin_cos_f64, sin_f64, sqrt_f32, sqrt_f64};

pub(crate) const WGS84_A: f32 = 6_378_137.0;
const WGS84_B: f32 = 6_356_752.314_245_18;
const WGS84_E2: f32 = 6.694_379_990_141_32e-3;
pub(crate) const WGS84_OMEGA_IE: f32 = 7.292_115e-5;
const WGS84_GM: f32 = 3.986_004_418e14;
const WGS84_J2: f32 = 1.082_629_821_368_57e-3;
const WGS84_A_F64: f64 = 6_378_137.0;
const WGS84_E2_F64: f64 = 6.694_379_990_14e-3;
const WGS84_NORMAL_GRAVITY_EQUATOR: f64 = 9.780_325_335_9;
const WGS84_NORMAL_GRAVITY_K: f64 = 0.001_931_852_652_41;
const WGS84_NORMAL_GRAVITY_M: f64 = 0.003_449_786_506_84;

pub(crate) fn dcm_ecef_to_ned_f32(lat_rad: f32, lon_rad: f32) -> [[f32; 3]; 3] {
    let sin_lat = libm::sinf(lat_rad);
    let cos_lat = libm::cosf(lat_rad);
    let sin_lon = libm::sinf(lon_rad);
    let cos_lon = libm::cosf(lon_rad);
    [
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [-sin_lon, cos_lon, 0.0],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
    ]
}

pub(crate) fn ecef_to_llh_f32(x_e: [f32; 3]) -> (f32, f32, f32) {
    let a2 = WGS84_A * WGS84_A;
    let b2 = WGS84_B * WGS84_B;
    let z2 = x_e[2] * x_e[2];
    let r2 = x_e[0] * x_e[0] + x_e[1] * x_e[1];
    let r = libm::sqrtf(r2);
    let f = 54.0 * b2 * z2;
    let g = r2 + (1.0 - WGS84_E2) * z2 - WGS84_E2 * (a2 - b2);
    let c = WGS84_E2 * WGS84_E2 * f * r2 / (g * g * g);
    let s = libm::cbrtf(1.0 + c + libm::sqrtf(c * c + 2.0 * c));
    let p = f / (3.0 * (s + 1.0 / s + 1.0) * (s + 1.0 / s + 1.0) * g * g);
    let q = libm::sqrtf(1.0 + 2.0 * WGS84_E2 * WGS84_E2 * p);
    let r0 = -p * WGS84_E2 * r / (1.0 + q)
        + libm::sqrtf(
            0.5 * a2 * (1.0 + 1.0 / q) - p * (1.0 - WGS84_E2) * z2 / (q * (1.0 + q)) - 0.5 * p * r2,
        );
    let tmp = (r - WGS84_E2 * r0) * (r - WGS84_E2 * r0);
    let u = libm::sqrtf(tmp + z2);
    let v = libm::sqrtf(tmp + (1.0 - WGS84_E2) * z2);
    let inv_av = 1.0 / (WGS84_A * v);
    let z0 = b2 * x_e[2] * inv_av;
    let height_m = u * (1.0 - b2 * inv_av);
    let lat_rad = libm::atan2f(x_e[2] + (a2 / b2 - 1.0) * z0, r);
    let lon_rad = libm::atan2f(x_e[1], x_e[0]);
    (lat_rad, lon_rad, height_m)
}

pub(crate) fn gravity_ecef_j2_f32(x_e: [f32; 3]) -> [f32; 3] {
    let r = libm::sqrtf(x_e[0] * x_e[0] + x_e[1] * x_e[1] + x_e[2] * x_e[2]);
    if r <= 0.0 {
        return [0.0; 3];
    }
    let r2 = r * r;
    let r3 = r * r2;
    let tmp1 = WGS84_GM / r3;
    let tmp2 = 1.5 * (WGS84_A * (WGS84_A * WGS84_J2)) / r2;
    let tmp3 = 5.0 * x_e[2] * x_e[2] / r2;
    [
        tmp1 * (-x_e[0] - tmp2 * (x_e[0] - tmp3 * x_e[0]))
            + WGS84_OMEGA_IE * WGS84_OMEGA_IE * x_e[0],
        tmp1 * (-x_e[1] - tmp2 * (x_e[1] - tmp3 * x_e[1]))
            + WGS84_OMEGA_IE * WGS84_OMEGA_IE * x_e[1],
        tmp1 * (-x_e[2] - tmp2 * (3.0 * x_e[2] - tmp3 * x_e[2])),
    ]
}

pub(crate) fn quat_ecef_to_ned_f64(lat_deg: f64, lon_deg: f64) -> [f64; 4] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let slat = sin_f64(lat);
    let clat = cos_f64(lat);
    let slon = sin_f64(lon);
    let clon = cos_f64(lon);
    let c_en = [
        [-slat * clon, -slat * slon, clat],
        [-slon, clon, 0.0],
        [-clat * clon, -clat * slon, -slat],
    ];
    dcm_to_quat_f64(c_en)
}

pub(crate) fn normal_gravity_mss_f64(lat_deg: f64, height_m: f64) -> f32 {
    let lat = lat_deg.to_radians();
    let slat = sin_f64(lat);
    let slat2 = slat * slat;
    let sqrt_term = sqrt_f64(1.0 - WGS84_E2_F64 * slat2);
    let surface_g =
        WGS84_NORMAL_GRAVITY_EQUATOR * (1.0 + WGS84_NORMAL_GRAVITY_K * slat2) / sqrt_term;
    let flattening = 1.0 - sqrt_f64(1.0 - WGS84_E2_F64);
    let h = height_m;
    let height_scale = 1.0
        - (2.0 / WGS84_A_F64)
            * (1.0 + flattening + WGS84_NORMAL_GRAVITY_M - 2.0 * flattening * slat2)
            * h
        + 3.0 * h * h / (WGS84_A_F64 * WGS84_A_F64);
    (surface_g * height_scale) as f32
}

pub(crate) fn lla_to_ecef_f64(lat_deg: f64, lon_deg: f64, height_m: f64) -> [f64; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (slat, clat) = sin_cos_f64(lat);
    let (slon, clon) = sin_cos_f64(lon);
    let n = WGS84_A_F64 / sqrt_f64(1.0 - WGS84_E2_F64 * slat * slat);
    [
        (n + height_m) * clat * clon,
        (n + height_m) * clat * slon,
        (n * (1.0 - WGS84_E2_F64) + height_m) * slat,
    ]
}

pub(crate) fn ecef_to_lla_f64(ecef_m: [f64; 3]) -> [f64; 3] {
    let x = ecef_m[0];
    let y = ecef_m[1];
    let z = ecef_m[2];
    let b = WGS84_A_F64 * sqrt_f64(1.0 - WGS84_E2_F64);
    let ep2 = (WGS84_A_F64 * WGS84_A_F64 - b * b) / (b * b);
    let p = sqrt_f64(x * x + y * y);
    let th = atan2_f64(WGS84_A_F64 * z, b * p);
    let lon = atan2_f64(y, x);
    let th_sin = sin_f64(th);
    let th_cos = cos_f64(th);
    let lat = atan2_f64(
        z + ep2 * b * th_sin * th_sin * th_sin,
        p - WGS84_E2_F64 * WGS84_A_F64 * th_cos * th_cos * th_cos,
    );
    let lat_sin = sin_f64(lat);
    let n = WGS84_A_F64 / sqrt_f64(1.0 - WGS84_E2_F64 * lat_sin * lat_sin);
    let h = p / cos_f64(lat) - n;
    [lat.to_degrees(), lon.to_degrees(), h]
}

pub(crate) fn ecef_to_ned_matrix_f32(lat_deg: f64, lon_deg: f64) -> [[f32; 3]; 3] {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (slat, clat) = sin_cos_f64(lat);
    let (slon, clon) = sin_cos_f64(lon);
    [
        [(-slat * clon) as f32, (-slat * slon) as f32, clat as f32],
        [-slon as f32, clon as f32, 0.0],
        [(-clat * clon) as f32, (-clat * slon) as f32, -slat as f32],
    ]
}

pub(crate) fn ned_vector_to_ecef_f32(lat_deg: f64, lon_deg: f64, v_ned: [f32; 3]) -> [f32; 3] {
    let c_ne = ecef_to_ned_matrix_f32(lat_deg, lon_deg);
    [
        c_ne[0][0] * v_ned[0] + c_ne[1][0] * v_ned[1] + c_ne[2][0] * v_ned[2],
        c_ne[0][1] * v_ned[0] + c_ne[1][1] * v_ned[1] + c_ne[2][1] * v_ned[2],
        c_ne[0][2] * v_ned[0] + c_ne[1][2] * v_ned[1] + c_ne[2][2] * v_ned[2],
    ]
}

pub(crate) fn navigation_rates_ned_f32(
    lat_deg: f32,
    height_m: f32,
    vel_ned_mps: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let lat = lat_deg.to_radians();
    let slat = libm::sinf(lat);
    let clat = libm::cosf(lat);
    let denom = (1.0 - WGS84_E2 * slat * slat).max(1.0e-6);
    let sqrt_denom = sqrt_f32(denom);
    let rn = WGS84_A / sqrt_denom;
    let rm = WGS84_A * (1.0 - WGS84_E2) / (denom * sqrt_denom);
    let rn_h = (rn + height_m).max(1.0);
    let rm_h = (rm + height_m).max(1.0);
    let clat_safe = if clat.abs() > 1.0e-3 {
        clat
    } else if clat.is_sign_negative() {
        -1.0e-3
    } else {
        1.0e-3
    };
    let omega_ie_n = [WGS84_OMEGA_IE * clat, 0.0, -WGS84_OMEGA_IE * slat];
    let omega_en_n = [
        vel_ned_mps[1] / rn_h,
        -vel_ned_mps[0] / rm_h,
        -vel_ned_mps[1] * slat / (clat_safe * rn_h),
    ];
    (omega_ie_n, omega_en_n)
}
