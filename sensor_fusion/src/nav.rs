//! Shared WGS84 and ECEF navigation helpers.
#![allow(clippy::excessive_precision)]

use crate::math::{atan2_f64, cos_f64, sin_cos_f64, sin_f64, sqrt_f32, sqrt_f64};

pub(crate) const WGS84_A: f32 = 6_378_137.0;
const WGS84_E2: f32 = 6.694_379_990_141_32e-3;
pub(crate) const WGS84_OMEGA_IE: f32 = 7.292_115e-5;
const WGS84_A_F64: f64 = 6_378_137.0;
const WGS84_E2_F64: f64 = 6.694_379_990_14e-3;
const WGS84_NORMAL_GRAVITY_EQUATOR: f64 = 9.780_325_335_9;
const WGS84_NORMAL_GRAVITY_K: f64 = 0.001_931_852_652_41;
const WGS84_NORMAL_GRAVITY_M: f64 = 0.003_449_786_506_84;

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
