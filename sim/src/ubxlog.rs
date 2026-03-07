use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use std::collections::HashMap;
use ublox::esf_alg::{EsfAlgError, EsfAlgStatus};
use ublox::nav_pvt::common::{NavPvtFlags, NavPvtValidFlags};
use ublox::proto31::PacketRef;
use ublox::{UbxProtocol, proto31::Proto31};

#[derive(Debug, Clone)]
pub struct UbxFrame {
    pub seq: u64,
    pub offset: usize,
    pub class: u8,
    pub id: u8,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SampleWord {
    pub value_i24: i32,
    pub dtype: u8,
}

pub fn parse_ubx_frames(data: &[u8], max_records: Option<usize>) -> Vec<UbxFrame> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut seq = 0u64;
    while i + 8 <= data.len() {
        if data[i] != 0xB5 || data[i + 1] != 0x62 {
            i += 1;
            continue;
        }
        let class = data[i + 2];
        let id = data[i + 3];
        let len = u16::from_le_bytes([data[i + 4], data[i + 5]]) as usize;
        let end = i + 6 + len + 2;
        if end > data.len() {
            break;
        }
        let payload = data[i + 6..i + 6 + len].to_vec();
        seq += 1;
        out.push(UbxFrame {
            seq,
            offset: i,
            class,
            id,
            payload,
        });
        if let Some(m) = max_records {
            if out.len() >= m {
                break;
            }
        }
        i = end;
    }
    out
}

pub fn u32_le_at(payload: &[u8], off: usize) -> Option<u32> {
    if off + 4 > payload.len() {
        return None;
    }
    Some(u32::from_le_bytes([
        payload[off],
        payload[off + 1],
        payload[off + 2],
        payload[off + 3],
    ]))
}

pub fn i32_le_at(payload: &[u8], off: usize) -> Option<i32> {
    if off + 4 > payload.len() {
        return None;
    }
    Some(i32::from_le_bytes([
        payload[off],
        payload[off + 1],
        payload[off + 2],
        payload[off + 3],
    ]))
}

pub fn i16_le_at(payload: &[u8], off: usize) -> Option<i16> {
    if off + 2 > payload.len() {
        return None;
    }
    Some(i16::from_le_bytes([payload[off], payload[off + 1]]))
}

pub fn u16_le_at(payload: &[u8], off: usize) -> Option<u16> {
    if off + 2 > payload.len() {
        return None;
    }
    Some(u16::from_le_bytes([payload[off], payload[off + 1]]))
}

pub fn i24_to_i32(raw_u24: u32) -> i32 {
    let mut v = (raw_u24 & 0x00FF_FFFF) as i32;
    if (v & 0x0080_0000) != 0 {
        v -= 1 << 24;
    }
    v
}

pub fn parse_sample_word(word: u32) -> SampleWord {
    let value = i24_to_i32(word & 0x00FF_FFFF);
    let dtype = ((word >> 24) & 0xFF) as u8;
    SampleWord {
        value_i24: value,
        dtype,
    }
}

pub fn identity_from_class_id(class: u8, id: u8) -> String {
    let s = match (class, id) {
        (0x01, 0x03) => "NAV-STATUS",
        (0x01, 0x05) => "NAV-ATT",
        (0x01, 0x07) => "NAV-PVT",
        (0x29, 0x07) => "NAV2-PVT",
        (0x01, 0x34) => "NAV-ORB",
        (0x01, 0x35) => "NAV-SAT",
        (0x0A, 0x04) => "MON-VER",
        (0x10, 0x02) => "ESF-MEAS",
        (0x10, 0x03) => "ESF-RAW",
        (0x10, 0x04) => "ESF-CAL",
        (0x10, 0x15) => "ESF-INS",
        (0x10, 0x14) => "ESF-ALG",
        (0x10, 0x10) => "ESF-STATUS",
        _ => "UNKNOWN",
    };
    s.to_string()
}

pub fn extract_itow_ms(frame: &UbxFrame) -> Option<i64> {
    match decode_packet(frame)? {
        PacketRef::NavPvt(pkt) => Some(pkt.itow() as i64),
        PacketRef::NavAtt(pkt) => Some(pkt.itow() as i64),
        PacketRef::NavStatus(pkt) => Some(pkt.itow() as i64),
        PacketRef::NavSat(pkt) => Some(pkt.itow() as i64),
        PacketRef::EsfIns(pkt) => Some(pkt.itow() as i64),
        PacketRef::EsfAlg(pkt) => Some(pkt.itow() as i64),
        PacketRef::EsfStatus(pkt) => Some(pkt.itow() as i64),
        _ => None,
    }
}

pub fn extract_tag_ms(frame: &UbxFrame) -> Option<i64> {
    match decode_packet(frame)? {
        PacketRef::EsfRaw(pkt) => pkt.data().next().map(|x| x.sensor_time_tag as i64),
        PacketRef::EsfCal(pkt) => Some(pkt.s_ttag() as i64),
        PacketRef::EsfMeas(pkt) => Some(pkt.itow() as i64),
        _ => None,
    }
}

pub fn extract_nav_pvt(frame: &UbxFrame) -> Option<(i64, f64, f64, f64, f64, f64, f64)> {
    match decode_nav_pvt_like(frame)? {
        PacketRef::NavPvt(pkt) => Some((
            pkt.itow() as i64,
            pkt.ground_speed_2d(),
            pkt.vel_north(),
            pkt.vel_east(),
            pkt.vel_down(),
            pkt.latitude(),
            pkt.longitude(),
        )),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NavPvtObs {
    pub itow_ms: i64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub height_m: f64,
    pub vel_n_mps: f64,
    pub vel_e_mps: f64,
    pub vel_d_mps: f64,
    pub heading_motion_deg: f64,
    pub h_acc_m: f64,
    pub v_acc_m: f64,
    pub s_acc_mps: f64,
    pub head_acc_deg: f64,
    pub fix_ok: bool,
    pub head_veh_valid: bool,
    pub invalid_llh: bool,
}

pub fn extract_nav_pvt_obs(frame: &UbxFrame) -> Option<NavPvtObs> {
    match decode_nav_pvt_like(frame)? {
        PacketRef::NavPvt(pkt) => {
            let flags = pkt.flags();
            Some(NavPvtObs {
                itow_ms: pkt.itow() as i64,
                lat_deg: pkt.latitude(),
                lon_deg: pkt.longitude(),
                height_m: pkt.height_above_ellipsoid(),
                vel_n_mps: pkt.vel_north(),
                vel_e_mps: pkt.vel_east(),
                vel_d_mps: pkt.vel_down(),
                heading_motion_deg: pkt.heading_motion(),
                h_acc_m: pkt.horizontal_accuracy(),
                v_acc_m: pkt.vertical_accuracy(),
                s_acc_mps: pkt.speed_accuracy(),
                head_acc_deg: pkt.heading_accuracy(),
                fix_ok: flags.contains(NavPvtFlags::GPS_FIX_OK),
                head_veh_valid: flags.contains(NavPvtFlags::HEAD_VEH_VALID),
                invalid_llh: pkt.flags3().invalid_llh(),
            })
        }
        _ => None,
    }
}

pub fn extract_nav2_pvt_obs(frame: &UbxFrame) -> Option<NavPvtObs> {
    if frame.class != 0x29 || frame.id != 0x07 {
        return None;
    }
    extract_nav_pvt_obs(frame)
}

pub fn extract_nav_pvt_utc(frame: &UbxFrame) -> Option<DateTime<Utc>> {
    match decode_nav_pvt_like(frame)? {
        PacketRef::NavPvt(pkt) => {
            let valid = pkt.valid();
            if !valid.contains(NavPvtValidFlags::VALID_DATE)
                || !valid.contains(NavPvtValidFlags::VALID_TIME)
                || !valid.contains(NavPvtValidFlags::FULLY_RESOLVED)
            {
                return None;
            }
            let year = pkt.year() as i32;
            let month = pkt.month() as u32;
            let day = pkt.day() as u32;
            let hour = pkt.hour() as u32;
            let minute = pkt.min() as u32;
            let second = pkt.sec() as u32;
            let date = NaiveDate::from_ymd_opt(year, month, day)?;
            let time = NaiveTime::from_hms_opt(hour, minute, second)?;
            let dt = NaiveDateTime::new(date, time);
            let base = DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc);
            Some(base + Duration::nanoseconds(pkt.nanosec() as i64))
        }
        _ => None,
    }
}

pub fn extract_nav_att(frame: &UbxFrame) -> Option<(i64, f64, f64, f64)> {
    match decode_packet(frame)? {
        PacketRef::NavAtt(pkt) => Some((
            pkt.itow() as i64,
            pkt.vehicle_roll(),
            pkt.vehicle_pitch(),
            pkt.vehicle_heading(),
        )),
        _ => None,
    }
}

pub fn extract_esf_alg(frame: &UbxFrame) -> Option<(i64, f64, f64, f64)> {
    match decode_packet(frame)? {
        PacketRef::EsfAlg(pkt) => Some((pkt.itow() as i64, pkt.roll(), pkt.pitch(), pkt.yaw())),
        _ => None,
    }
}

pub fn extract_esf_alg_status(frame: &UbxFrame) -> Option<(i64, f64, f64)> {
    match decode_packet(frame)? {
        PacketRef::EsfAlg(pkt) => {
            let status = pkt.flags().status();
            let fine = matches!(status, EsfAlgStatus::FineAlignment);
            Some((pkt.itow() as i64, status as u8 as f64, if fine { 1.0 } else { 0.0 }))
        }
        _ => None,
    }
}

pub fn extract_esf_alg_valid(frame: &UbxFrame) -> Option<(i64, f64, f64, f64)> {
    match decode_packet(frame)? {
        PacketRef::EsfAlg(pkt) => {
            let status = pkt.flags().status();
            let err = pkt.error();
            let aligned = matches!(
                status,
                EsfAlgStatus::CoarseAlignment | EsfAlgStatus::FineAlignment
            );
            let no_angle_error = !err.contains(EsfAlgError::ANGLE_ERROR);
            if aligned && no_angle_error {
                Some((pkt.itow() as i64, pkt.roll(), pkt.pitch(), pkt.yaw()))
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn extract_esf_ins(frame: &UbxFrame) -> Option<(i64, f64, f64, f64, f64, f64, f64)> {
    match decode_packet(frame)? {
        PacketRef::EsfIns(pkt) => Some((
            pkt.itow() as i64,
            pkt.x_angular_rate(),
            pkt.y_angular_rate(),
            pkt.z_angular_rate(),
            pkt.x_acceleration(),
            pkt.y_acceleration(),
            pkt.z_acceleration(),
        )),
        _ => None,
    }
}

pub fn extract_nav_sat_cn0(frame: &UbxFrame) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    if let Some(PacketRef::NavSat(pkt)) = decode_packet(frame) {
        for sv in pkt.svs() {
            out.push((
                format!("gnss{}-sv{}", sv.gnss_id(), sv.sv_id()),
                sv.cno() as f64,
            ));
        }
    }
    out
}

pub fn extract_esf_raw_samples(frame: &UbxFrame) -> Vec<(u64, SampleWord)> {
    let mut out = Vec::new();
    if let Some(PacketRef::EsfRaw(pkt)) = decode_packet(frame) {
        for d in pkt.data() {
            out.push((
                d.sensor_time_tag as u64,
                SampleWord {
                    value_i24: i24_to_i32(d.data_field),
                    dtype: d.data_type,
                },
            ));
        }
    }
    out
}

pub fn extract_esf_cal_samples(frame: &UbxFrame) -> Vec<(u64, SampleWord)> {
    let mut out = Vec::new();
    if let Some(PacketRef::EsfCal(pkt)) = decode_packet(frame) {
        let tag = pkt.s_ttag() as u64;
        for d in pkt.data() {
            out.push((
                tag,
                SampleWord {
                    value_i24: i24_to_i32(d.data_field),
                    dtype: d.data_type,
                },
            ));
        }
    }
    out
}

pub fn extract_esf_meas_samples(frame: &UbxFrame) -> Vec<(u64, SampleWord)> {
    let mut out = Vec::new();
    if let Some(PacketRef::EsfMeas(pkt)) = decode_packet(frame) {
        let tag = pkt.itow() as u64;
        for d in pkt.data() {
            out.push((
                tag,
                SampleWord {
                    value_i24: d.data_field,
                    dtype: d.data_type as u8,
                },
            ));
        }
    }
    out
}

fn decode_packet(frame: &UbxFrame) -> Option<PacketRef<'_>> {
    <Proto31 as UbxProtocol>::match_packet(frame.class, frame.id, &frame.payload).ok()
}

fn decode_nav_pvt_like(frame: &UbxFrame) -> Option<PacketRef<'_>> {
    match (frame.class, frame.id) {
        (0x01, 0x07) => <Proto31 as UbxProtocol>::match_packet(0x01, 0x07, &frame.payload).ok(),
        // NAV2-PVT has NAV-PVT-equivalent payload but class/id 0x29/0x07.
        // Decode via the crate's NAV-PVT parser to avoid manual field parsing.
        (0x29, 0x07) => <Proto31 as UbxProtocol>::match_packet(0x01, 0x07, &frame.payload).ok(),
        _ => None,
    }
}

pub fn unwrap_counter(values: &[u64], modulus: u64) -> Vec<u64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut offset = 0u64;
    let mut prev = values[0];
    out.push(prev);
    for &v in values.iter().skip(1) {
        if v < prev {
            offset = offset.saturating_add(modulus);
        }
        out.push(v.saturating_add(offset));
        prev = v;
    }
    out
}

pub fn fit_linear_map(x: &[f64], y: &[f64], fallback_s_per_tick: f64) -> (f64, f64) {
    if x.len() >= 2 {
        let n = x.len() as f64;
        let sx: f64 = x.iter().sum();
        let sy: f64 = y.iter().sum();
        let sxx: f64 = x.iter().map(|v| v * v).sum();
        let sxy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let denom = n * sxx - sx * sx;
        if denom.abs() > f64::EPSILON {
            let a = (n * sxy - sx * sy) / denom;
            let b = (sy - a * sx) / n;
            return (a, b);
        }
    }
    let a = fallback_s_per_tick * 1000.0;
    if let (Some(x0), Some(y0)) = (x.first(), y.first()) {
        return (a, y0 - a * x0);
    }
    (a, 0.0)
}

pub fn sensor_meta(dtype: u8) -> (&'static str, &'static str, f64) {
    match dtype {
        5 => ("gyro_z", "deg/s", 2f64.powi(-12)),
        13 => ("gyro_y", "deg/s", 2f64.powi(-12)),
        14 => ("gyro_x", "deg/s", 2f64.powi(-12)),
        16 => ("accel_x", "m/s^2", 2f64.powi(-10)),
        17 => ("accel_y", "m/s^2", 2f64.powi(-10)),
        18 => ("accel_z", "m/s^2", 2f64.powi(-10)),
        11 => ("speed", "m/s", 1e-3),
        12 => ("gyro_temp", "degC", 1e-2),
        _ => ("other", "raw", 1.0),
    }
}

pub fn identity_counts(frames: &[UbxFrame]) -> HashMap<String, usize> {
    let mut map: HashMap<String, usize> = HashMap::new();
    for f in frames {
        *map.entry(identity_from_class_id(f.class, f.id))
            .or_insert(0) += 1;
    }
    map
}
