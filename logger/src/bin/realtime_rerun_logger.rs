use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use rerun::{RecordingStream, RecordingStreamBuilder};
use rerun::external::re_log_types::{BlueprintActivationCommand, LogMsg, RecordingId};
use rerun::sink::SinkFlushError;
use serialport::SerialPort;
use ublox::nav_pvt::common::NavPvtFlags;
use ublox::nav_sat::NavSatRef;
use ublox::proto31::{PacketRef, Proto31};
use ublox::UbxProtocol;

const CFG_VALSET_ID: u8 = 0x8A;
const CFG_VALGET_ID: u8 = 0x8B;

#[derive(Parser, Debug, Clone)]
#[command(name = "realtime_rerun_logger")]
struct Args {
    #[arg(long, default_value = "")]
    port: String,
    #[arg(long, default_value_t = 921_600)]
    baud: u32,
    #[arg(long, default_value = "921600,460800,230400,115200")]
    detect_bauds: String,
    #[arg(long, default_value_t = 3.0)]
    config_timeout: f64,
    #[arg(long, default_value_t = false)]
    skip_startup_config: bool,
    #[arg(long, default_value_t = 0.0)]
    duration: f64,
    #[arg(long, default_value_t = 2048)]
    chunk: usize,
    #[arg(long, default_value_t = 0.01)]
    serial_timeout: f64,
    #[arg(long, default_value = "")]
    raw_log: String,
    #[arg(long, default_value = "data_logger_rerun_rs")]
    app_id: String,
    #[arg(long, default_value_t = false)]
    spawn_viewer: bool,
    #[arg(
        long,
        default_value = "pk.eyJ1IjoieW9uZ2t5dW5zODciLCJhIjoiY21tNjB5NWt6MGJmOTJzcG02MmRvN3RnYiJ9.fu_66qb1G1cgrLzAE54E0w"
    )]
    mapbox_token: String,
}

#[derive(Clone, Copy)]
struct CfgItem {
    key: u32,
    value: u64,
    size: usize,
}

#[derive(Debug, Clone)]
struct RawFrame {
    seq: u64,
    class: u8,
    id: u8,
    payload: Vec<u8>,
    elapsed_s: f64,
}

#[derive(Default)]
struct ViewerDecimator {
    last_emit_s: HashMap<&'static str, f64>,
}

impl ViewerDecimator {
    fn min_interval_s(path: &str, lag_s: f64) -> f64 {
        if path.starts_with("imu/") {
            if lag_s > 8.0 {
                0.20
            } else if lag_s > 4.0 {
                0.10
            } else if lag_s > 2.0 {
                0.05
            } else {
                0.02
            }
        } else if path.starts_with("status/") {
            0.50
        } else if path.starts_with("gnss/") || path.starts_with("fusion/") {
            if lag_s > 8.0 { 0.20 } else { 0.00 }
        } else if lag_s > 8.0 {
            0.10
        } else {
            0.00
        }
    }

    fn should_emit(&mut self, path: &'static str, t_s: f64, lag_s: f64) -> bool {
        let dt_min = Self::min_interval_s(path, lag_s);
        if dt_min <= 0.0 {
            self.last_emit_s.insert(path, t_s);
            return true;
        }
        let last = self.last_emit_s.get(path).copied();
        match last {
            Some(prev) if t_s - prev < dt_min => false,
            _ => {
                self.last_emit_s.insert(path, t_s);
                true
            }
        }
    }
}

fn cfg_items_startup(target_baud: u32) -> Vec<CfgItem> {
    vec![
        // CFG_UART1INPROT_NMEA / OUTPROT_NMEA / UART2 NMEA off
        CfgItem { key: 0x1073_0002, value: 0, size: 1 },
        CfgItem { key: 0x1074_0002, value: 0, size: 1 },
        CfgItem { key: 0x1075_0002, value: 0, size: 1 },
        CfgItem { key: 0x1076_0002, value: 0, size: 1 },
        // CFG_UART1INPROT_UBX / OUTPROT_UBX on
        CfgItem { key: 0x1073_0001, value: 1, size: 1 },
        CfgItem { key: 0x1074_0001, value: 1, size: 1 },
        // CFG_RATE_MEAS / NAV / NAV_PRIO
        CfgItem { key: 0x3021_0001, value: 500, size: 2 },
        CfgItem { key: 0x3021_0002, value: 1, size: 2 },
        CfgItem { key: 0x2021_0004, value: 30, size: 1 },
        // NAV msgout UART1
        CfgItem { key: 0x2091_0016, value: 1, size: 1 }, // NAV-SAT
        CfgItem { key: 0x2091_0007, value: 1, size: 1 }, // NAV-PVT
        CfgItem { key: 0x2091_001b, value: 1, size: 1 }, // NAV-STATUS
        CfgItem { key: 0x2091_0011, value: 1, size: 1 }, // NAV-ORB
        CfgItem { key: 0x2091_0020, value: 1, size: 1 }, // NAV-ATT
        // ESF msgout UART1
        CfgItem { key: 0x2091_02a0, value: 1, size: 1 }, // ESF-RAW
        CfgItem { key: 0x2091_06ad, value: 1, size: 1 }, // ESF-CAL
        CfgItem { key: 0x2091_0278, value: 1, size: 1 }, // ESF-MEAS
        CfgItem { key: 0x2091_0110, value: 1, size: 1 }, // ESF-ALG
        CfgItem { key: 0x2091_0115, value: 1, size: 1 }, // ESF-INS
        CfgItem { key: 0x2091_0106, value: 1, size: 1 }, // ESF-STATUS
        // UART1 baud
        CfgItem { key: 0x4052_0001, value: target_baud as u64, size: 4 },
        // CFG_NAVSPG_DYNMODEL = automotive(4)
        CfgItem { key: 0x2011_0021, value: 4, size: 1 },
    ]
}

fn parse_detect_bauds(text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for p in text.split(',') {
        if let Ok(v) = p.trim().parse::<u32>() {
            if v > 0 && !out.contains(&v) {
                out.push(v);
            }
        }
    }
    if out.is_empty() {
        vec![921_600, 460_800, 230_400, 115_200]
    } else {
        out
    }
}

fn candidate_ports(preferred: &str) -> Vec<String> {
    let mut out = Vec::new();
    if !preferred.is_empty() {
        out.push(preferred.to_owned());
    }
    if let Ok(rd) = std::fs::read_dir("/dev") {
        for entry in rd.flatten() {
            if let Ok(name) = entry.file_name().into_string() {
                let ok = name.starts_with("cu.usbmodem")
                    || name.starts_with("tty.usbmodem")
                    || name.starts_with("cu.usbserial")
                    || name.starts_with("tty.usbserial");
                if ok {
                    out.push(format!("/dev/{name}"));
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn ubx_checksum(bytes: &[u8]) -> (u8, u8) {
    let mut a: u8 = 0;
    let mut b: u8 = 0;
    for &x in bytes {
        a = a.wrapping_add(x);
        b = b.wrapping_add(a);
    }
    (a, b)
}

fn build_cfg_valset_packet(items: &[CfgItem], layers: u8) -> Vec<u8> {
    let mut payload = vec![0u8, layers, 0u8, 0u8];
    for it in items {
        payload.extend_from_slice(&it.key.to_le_bytes());
        match it.size {
            1 => payload.push(it.value as u8),
            2 => payload.extend_from_slice(&(it.value as u16).to_le_bytes()),
            4 => payload.extend_from_slice(&(it.value as u32).to_le_bytes()),
            8 => payload.extend_from_slice(&it.value.to_le_bytes()),
            _ => {}
        }
    }
    let mut pkt = Vec::with_capacity(8 + payload.len());
    pkt.extend_from_slice(&[0xB5, 0x62, 0x06, CFG_VALSET_ID]);
    pkt.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    pkt.extend_from_slice(&payload);
    let (ck_a, ck_b) = ubx_checksum(&pkt[2..]);
    pkt.push(ck_a);
    pkt.push(ck_b);
    pkt
}

fn build_cfg_valget_packet(keys: &[u32], layer: u8) -> Vec<u8> {
    let mut payload = vec![0u8, layer, 0u8, 0u8];
    for k in keys {
        payload.extend_from_slice(&k.to_le_bytes());
    }
    let mut pkt = Vec::with_capacity(8 + payload.len());
    pkt.extend_from_slice(&[0xB5, 0x62, 0x06, CFG_VALGET_ID]);
    pkt.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    pkt.extend_from_slice(&payload);
    let (ck_a, ck_b) = ubx_checksum(&pkt[2..]);
    pkt.push(ck_a);
    pkt.push(ck_b);
    pkt
}

fn extract_ubx_frames(buffer: &mut Vec<u8>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let mut sync_idx = None;
        for i in 0..buffer.len().saturating_sub(1) {
            if buffer[i] == 0xB5 && buffer[i + 1] == 0x62 {
                sync_idx = Some(i);
                break;
            }
        }
        let Some(si) = sync_idx else {
            if buffer.len() > 1 {
                let last = *buffer.last().unwrap();
                buffer.clear();
                buffer.push(last);
            }
            break;
        };
        if si > 0 {
            buffer.drain(0..si);
        }
        if buffer.len() < 6 {
            break;
        }
        let len = u16::from_le_bytes([buffer[4], buffer[5]]) as usize;
        let frame_len = len + 8;
        if buffer.len() < frame_len {
            break;
        }
        out.push(buffer.drain(0..frame_len).collect());
    }
    out
}

fn wait_cfg_valset_ack(port: &mut dyn SerialPort, timeout: Duration) -> Result<bool> {
    let deadline = Instant::now() + timeout;
    let mut tmp = [0u8; 1024];
    let mut buf = Vec::<u8>::new();
    while Instant::now() < deadline {
        match port.read(&mut tmp) {
            Ok(n) if n > 0 => {
                buf.extend_from_slice(&tmp[..n]);
                for fr in extract_ubx_frames(&mut buf) {
                    if fr.len() < 10 {
                        continue;
                    }
                    let class = fr[2];
                    let id = fr[3];
                    if class == 0x05 && id == 0x01 {
                        if fr[6] == 0x06 && fr[7] == CFG_VALSET_ID {
                            return Ok(true);
                        }
                    } else if class == 0x05 && id == 0x00 {
                        if fr[6] == 0x06 && fr[7] == CFG_VALSET_ID {
                            return Ok(false);
                        }
                    }
                }
            }
            _ => thread::sleep(Duration::from_millis(2)),
        }
    }
    Ok(false)
}

fn send_startup_config(port: &mut dyn SerialPort, target_baud: u32, timeout: Duration) -> Result<bool> {
    let pkt = build_cfg_valset_packet(&cfg_items_startup(target_baud), 0x07);
    port.write_all(&pkt)?;
    port.flush()?;
    wait_cfg_valset_ack(port, timeout)
}

fn poll_cfg_values(port: &mut dyn SerialPort, keys: &[CfgItem], timeout: Duration) -> Result<HashMap<u32, u64>> {
    let key_ids: Vec<u32> = keys.iter().map(|k| k.key).collect();
    let key_sizes: HashMap<u32, usize> = keys.iter().map(|k| (k.key, k.size)).collect();
    let pkt = build_cfg_valget_packet(&key_ids, 0);
    port.write_all(&pkt)?;
    port.flush()?;

    let mut out = HashMap::<u32, u64>::new();
    let mut buf = Vec::<u8>::new();
    let mut tmp = [0u8; 1024];
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        match port.read(&mut tmp) {
            Ok(n) if n > 0 => {
                buf.extend_from_slice(&tmp[..n]);
                for fr in extract_ubx_frames(&mut buf) {
                    if fr.len() < 8 || fr[2] != 0x06 || fr[3] != CFG_VALGET_ID {
                        continue;
                    }
                    let payload = &fr[6..fr.len() - 2];
                    if payload.len() < 4 {
                        continue;
                    }
                    let mut i = 4usize;
                    while i + 4 <= payload.len() {
                        let key = u32::from_le_bytes([payload[i], payload[i + 1], payload[i + 2], payload[i + 3]]);
                        i += 4;
                        let Some(sz) = key_sizes.get(&key).copied() else { break };
                        if i + sz > payload.len() {
                            break;
                        }
                        let val = match sz {
                            1 => payload[i] as u64,
                            2 => u16::from_le_bytes([payload[i], payload[i + 1]]) as u64,
                            4 => u32::from_le_bytes([payload[i], payload[i + 1], payload[i + 2], payload[i + 3]]) as u64,
                            8 => u64::from_le_bytes([
                                payload[i], payload[i + 1], payload[i + 2], payload[i + 3], payload[i + 4], payload[i + 5],
                                payload[i + 6], payload[i + 7],
                            ]),
                            _ => 0,
                        };
                        out.insert(key, val);
                        i += sz;
                    }
                }
            }
            _ => thread::sleep(Duration::from_millis(2)),
        }
        if out.len() >= key_ids.len() {
            break;
        }
    }
    Ok(out)
}

fn open_serial(path: &str, baud: u32, timeout_s: f64) -> Result<Box<dyn SerialPort>> {
    serialport::new(path, baud)
        .timeout(Duration::from_secs_f64(timeout_s.max(0.001)))
        .open()
        .with_context(|| format!("failed to open serial {path} @ {baud}"))
}

fn select_port_and_config(args: &Args) -> Result<(String, u32)> {
    let ports = candidate_ports(&args.port);
    if ports.is_empty() {
        anyhow::bail!("no candidate serial ports found");
    }
    let bauds = parse_detect_bauds(&args.detect_bauds);
    let mut last_err = String::new();
    for p in ports {
        for probe in &bauds {
            match open_serial(&p, *probe, 0.2) {
                Ok(mut sp) => {
                    if args.skip_startup_config {
                        return Ok((p.clone(), *probe));
                    }
                    match send_startup_config(&mut *sp, args.baud, Duration::from_secs_f64(args.config_timeout)) {
                        Ok(true) => return Ok((p.clone(), *probe)),
                        Ok(false) => {
                            last_err = format!("no ACK for CFG-VALSET on {p} @ {probe}");
                        }
                        Err(e) => last_err = e.to_string(),
                    }
                }
                Err(e) => {
                    last_err = e.to_string();
                }
            }
        }
    }
    anyhow::bail!("could not connect/configure receiver: {last_err}");
}

fn log_scalar(
    decimator: &mut ViewerDecimator,
    rec: &RecordingStream,
    t_s: f64,
    lag_s: f64,
    path: &'static str,
    value: f64,
) {
    if value.is_finite() {
        if !decimator.should_emit(path, t_s, lag_s) {
            return;
        }
        let _ = rec.log(path, &rerun::Scalars::new([value]));
    }
}

fn log_nav_sat(rec: &RecordingStream, pkt: &NavSatRef<'_>) {
    let mut rows: Vec<(&'static str, i32, f64, String)> = Vec::new();
    let mut lines = Vec::new();
    for sv in pkt.svs() {
        let cno = sv.cno() as f64;
        if cno <= 0.0 {
            continue;
        }
        let sys = match sv.gnss_id() {
            0 => "GPS",
            1 => "SBAS",
            2 => "GAL",
            3 => "BDS",
            4 => "IMES",
            5 => "QZSS",
            6 => "GLO",
            7 => "NAVIC",
            _ => "OTHER",
        };
        let flags = sv.flags();
        let used = if flags.sv_used() { "✅" } else { "❌" };
        let health = match flags.health() {
            ublox::nav_sat::NavSatSvHealth::Healthy => "✅",
            ublox::nav_sat::NavSatSvHealth::Unhealthy => "❌",
            ublox::nav_sat::NavSatSvHealth::Unknown(_) => "❔",
        };
        let quality = match flags.quality_ind() {
            ublox::nav_sat::NavSatQualityIndicator::NoSignal => "❌",
            ublox::nav_sat::NavSatQualityIndicator::Searching => "🔍",
            ublox::nav_sat::NavSatQualityIndicator::SignalAcquired => "🟡",
            ublox::nav_sat::NavSatQualityIndicator::SignalDetected => "🟡",
            ublox::nav_sat::NavSatQualityIndicator::CodeLock => "✅",
            ublox::nav_sat::NavSatQualityIndicator::CarrierLock => "✅",
            ublox::nav_sat::NavSatQualityIndicator::Invalid => "❔",
        };
        let corr = if flags.differential_correction_available()
            || flags.sbas_corr()
            || flags.rtcm_corr()
            || flags.spartn_corr()
        {
            "✅"
        } else {
            "❌"
        };
        let eph = if flags.ephemeris_available() { "✅" } else { "❌" };
        let summary = format!("use:{used} health:{health} q:{quality} corr:{corr} eph:{eph}");
        rows.push((sys, sv.sv_id() as i32, cno, summary));
    }

    if rows.is_empty() {
        return;
    }

    fn order(sys: &str) -> i32 {
        match sys {
            "GPS" => 0,
            "GAL" => 1,
            "BDS" => 2,
            "GLO" => 3,
            "QZSS" => 4,
            "SBAS" => 5,
            "NAVIC" => 6,
            "IMES" => 7,
            "OTHER" => 8,
            _ => 99,
        }
    }

    fn color(sys: &str) -> rerun::Color {
        match sys {
            "GPS" => rerun::Color::from_unmultiplied_rgba(66, 135, 245, 255),
            "GAL" => rerun::Color::from_unmultiplied_rgba(52, 168, 83, 255),
            "BDS" => rerun::Color::from_unmultiplied_rgba(251, 188, 5, 255),
            "GLO" => rerun::Color::from_unmultiplied_rgba(234, 67, 53, 255),
            "QZSS" => rerun::Color::from_unmultiplied_rgba(156, 39, 176, 255),
            "SBAS" => rerun::Color::from_unmultiplied_rgba(0, 188, 212, 255),
            "NAVIC" => rerun::Color::from_unmultiplied_rgba(255, 112, 67, 255),
            "IMES" => rerun::Color::from_unmultiplied_rgba(158, 158, 158, 255),
            _ => rerun::Color::from_unmultiplied_rgba(120, 120, 120, 255),
        }
    }

    rows.sort_by_key(|(sys, sv, _, _)| (order(sys), *sv));

    let mut grouped_cno: HashMap<&'static str, Vec<f64>> = HashMap::new();
    let mut grouped_idx: HashMap<&'static str, Vec<i64>> = HashMap::new();
    for (idx, (sys, sv, cno, summary)) in rows.into_iter().enumerate() {
        grouped_idx.entry(sys).or_default().push(idx as i64);
        grouped_cno.entry(sys).or_default().push(cno);
        lines.push(format!("{idx:02}: {sys}-{sv:02}  {cno:.1} dB-Hz  {summary}"));
    }

    for (sys, ys) in grouped_cno {
        let xs = grouped_idx.remove(sys).unwrap_or_default();
        let _ = rec.log(
            format!("gnss/nav_sat/cno/{sys}"),
            &rerun::BarChart::new(ys).with_abscissa(xs).with_color(color(sys)),
        );
    }

    let summary = lines.into_iter().take(64).collect::<Vec<_>>().join("\n");
    let _ = rec.log(
        "gnss/nav_sat/cno_summary",
        &rerun::TextDocument::from_markdown(format!(
            "Legend: use ✅/❌, health ✅/❌/❔, q ❌/🔍/🟡/✅, corr ✅/❌, eph ✅/❌\n\n{}",
            summary
        )),
    );
    let _ = rec.log(
        "gnss/nav_sat/emoji_log",
        &rerun::TextLog::new(format!("NAV-SAT\n{summary}")),
    );
}

fn safe_now_utc() -> DateTime<Utc> {
    Utc::now()
}

fn log_blueprint_view(
    rec: &RecordingStream,
    view_path: &str,
    class_identifier: &str,
    display_name: &str,
    origin: &str,
) -> Result<()> {
    use rerun::external::re_types::blueprint::archetypes::{ViewBlueprint, ViewContents};
    use rerun::external::re_types::blueprint::components::{QueryExpression, ViewClass};

    rec.log(
        format!("{view_path}/ViewContents"),
        &ViewContents::new([QueryExpression("$origin/**".into())]),
    )?;
    rec.log(
        view_path,
        &ViewBlueprint::new(ViewClass(class_identifier.into()))
            .with_display_name(display_name)
            .with_space_origin(origin),
    )?;
    Ok(())
}

fn send_startup_blueprint(app_id: &str, rec: &RecordingStream) -> Result<()> {
    use rerun::external::re_types::blueprint::archetypes::{
        ContainerBlueprint, MapBackground, MapZoom, PanelBlueprint, TimePanelBlueprint, ViewportBlueprint,
    };
    use rerun::external::re_types::blueprint::components::{
        ColumnShare, ContainerKind, IncludedContent, MapProvider, PanelState, RootContainer, RowShare,
    };
    use rerun::external::re_types::datatypes::{Float32, Uuid as DtUuid};

    let view_gnss = uuid::Uuid::new_v4();
    let view_imu_accel = uuid::Uuid::new_v4();
    let view_imu_gyro = uuid::Uuid::new_v4();
    let view_fusion = uuid::Uuid::new_v4();
    let view_track = uuid::Uuid::new_v4();
    let view_cal_esf_alg = uuid::Uuid::new_v4();
    let view_cal_esf_status = uuid::Uuid::new_v4();
    let view_nav_status = uuid::Uuid::new_v4();
    let view_cal_state = uuid::Uuid::new_v4();
    let view_nav_sat_cno = uuid::Uuid::new_v4();

    let container_left = uuid::Uuid::new_v4();
    let container_cal_tabs = uuid::Uuid::new_v4();
    let container_right = uuid::Uuid::new_v4();
    let container_main = uuid::Uuid::new_v4();

    let v_gnss = format!("view/{view_gnss}");
    let v_imu_accel = format!("view/{view_imu_accel}");
    let v_imu_gyro = format!("view/{view_imu_gyro}");
    let v_fusion = format!("view/{view_fusion}");
    let v_track = format!("view/{view_track}");
    let v_cal_esf_alg = format!("view/{view_cal_esf_alg}");
    let v_cal_esf_status = format!("view/{view_cal_esf_status}");
    let v_nav_status = format!("view/{view_nav_status}");
    let v_cal_state = format!("view/{view_cal_state}");
    let v_nav_sat_cno = format!("view/{view_nav_sat_cno}");

    let c_left = format!("container/{container_left}");
    let c_cal_tabs = format!("container/{container_cal_tabs}");
    let c_right = format!("container/{container_right}");
    let c_main = format!("container/{container_main}");

    let (bp_rec, storage) = RecordingStreamBuilder::new(app_id)
        .recording_id(RecordingId::random())
        .blueprint()
        .memory()
        .context("failed to create blueprint memory stream")?;
    bp_rec.set_time_sequence("blueprint", 0);

    // Views
    log_blueprint_view(&bp_rec, &v_gnss, "TimeSeries", "GNSS", "/gnss")?;
    log_blueprint_view(
        &bp_rec,
        &v_imu_accel,
        "TimeSeries",
        "IMU Accel (m/s^2)",
        "/imu/accel/rt",
    )?;
    log_blueprint_view(
        &bp_rec,
        &v_imu_gyro,
        "TimeSeries",
        "IMU Gyro (deg/s)",
        "/imu/gyro/rt",
    )?;
    log_blueprint_view(&bp_rec, &v_fusion, "TimeSeries", "Fusion", "/fusion")?;
    log_blueprint_view(&bp_rec, &v_track, "Map", "Track", "/map")?;
    log_blueprint_view(
        &bp_rec,
        &v_cal_esf_alg,
        "TextDocument",
        "Calibration Summary (ESF-ALG)",
        "/status/calibration/summary/esf_alg",
    )?;
    log_blueprint_view(
        &bp_rec,
        &v_cal_esf_status,
        "TextDocument",
        "Calibration Summary (ESF-STATUS)",
        "/status/calibration/summary/esf_status",
    )?;
    log_blueprint_view(
        &bp_rec,
        &v_nav_status,
        "TextDocument",
        "NAV-STATUS",
        "/status/navigation/summary/nav_status",
    )?;
    log_blueprint_view(
        &bp_rec,
        &v_cal_state,
        "TimeSeries",
        "Calibration State",
        "/status/calibration",
    )?;
    log_blueprint_view(
        &bp_rec,
        &v_nav_sat_cno,
        "BarChart",
        "Satellite C/N0",
        "/gnss/nav_sat/cno",
    )?;

    // Map-specific properties.
    bp_rec.log(
        format!("{v_track}/MapBackground"),
        &MapBackground::new(MapProvider::MapboxDark),
    )?;
    bp_rec.log(format!("{v_track}/MapZoom"), &MapZoom::new(14.0))?;

    // Containers
    bp_rec.log(
        c_left.as_str(),
        &ContainerBlueprint::new(ContainerKind::Vertical)
            .with_contents([
                IncludedContent(v_gnss.clone().into()),
                IncludedContent(v_imu_accel.clone().into()),
                IncludedContent(v_imu_gyro.clone().into()),
                IncludedContent(v_fusion.clone().into()),
            ]),
    )?;
    bp_rec.log(
        c_cal_tabs.as_str(),
        &ContainerBlueprint::new(ContainerKind::Tabs)
            .with_contents([
                IncludedContent(v_cal_esf_alg.clone().into()),
                IncludedContent(v_cal_esf_status.clone().into()),
                IncludedContent(v_nav_status.clone().into()),
                IncludedContent(v_cal_state.clone().into()),
            ])
            .with_active_tab(v_cal_esf_alg.clone()),
    )?;
    bp_rec.log(
        c_right.as_str(),
        &ContainerBlueprint::new(ContainerKind::Vertical)
            .with_contents([
                IncludedContent(v_track.clone().into()),
                IncludedContent(c_cal_tabs.clone().into()),
                IncludedContent(v_nav_sat_cno.clone().into()),
            ])
            .with_row_shares([
                RowShare(Float32(5.0)),
                RowShare(Float32(1.0)),
                RowShare(Float32(3.0)),
            ]),
    )?;
    bp_rec.log(
        c_main.as_str(),
        &ContainerBlueprint::new(ContainerKind::Horizontal)
            .with_contents([
                IncludedContent(c_left.clone().into()),
                IncludedContent(c_right.clone().into()),
            ])
            .with_col_shares([ColumnShare(Float32(1.0)), ColumnShare(Float32(1.0))]),
    )?;

    bp_rec.log(
        "viewport",
        &ViewportBlueprint::new().with_root_container(RootContainer(DtUuid::from(container_main))),
    )?;

    // Match Python's collapse_panels=True.
    bp_rec.log(
        "blueprint_panel",
        &PanelBlueprint::new().with_state(PanelState::Collapsed),
    )?;
    bp_rec.log(
        "selection_panel",
        &PanelBlueprint::new().with_state(PanelState::Collapsed),
    )?;
    bp_rec.log(
        "time_panel",
        &TimePanelBlueprint::new().with_state(PanelState::Collapsed),
    )?;

    let msgs = storage.take();
    let blueprint_id = msgs
        .iter()
        .find_map(|msg| match msg {
            LogMsg::SetStoreInfo(info) => Some(info.info.store_id.clone()),
            _ => None,
        })
        .context("blueprint stream produced no SetStoreInfo")?;

    rec.send_blueprint(
        msgs,
        BlueprintActivationCommand {
            blueprint_id,
            make_active: true,
            make_default: true,
        },
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    std::env::set_var("RERUN_MAPBOX_ACCESS_TOKEN", &args.mapbox_token);

    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    std::fs::create_dir_all(&data_dir)?;
    let raw_path = if args.raw_log.is_empty() {
        let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
        data_dir.join(format!("ubx_raw_{ts}.bin"))
    } else {
        let p = PathBuf::from(&args.raw_log);
        if p.is_absolute() { p } else { data_dir.join(p) }
    };
    if let Some(parent) = raw_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    println!("raw_log_path={}", raw_path.display());

    let (selected_port, selected_probe_baud) = select_port_and_config(&args)?;
    println!("selected_port={selected_port}");
    println!("selected_probe_baud={selected_probe_baud}");
    println!("logger_baud={}", args.baud);
    println!(
        "startup_config={}",
        if args.skip_startup_config { "disabled" } else { "enabled" }
    );

    if !args.skip_startup_config {
        if let Ok(mut sp) = open_serial(&selected_port, selected_probe_baud, 0.2) {
            let verify = cfg_items_startup(args.baud);
            if let Ok(applied) = poll_cfg_values(&mut *sp, &verify, Duration::from_secs_f64(args.config_timeout)) {
                println!("cfg_readback_ram:");
                for v in verify {
                    let got = applied.get(&v.key).copied();
                    let ok = got == Some(v.value);
                    println!(
                        "  {:#010x}: expected={} applied={:?} {}",
                        v.key,
                        v.value,
                        got,
                        if ok { "OK" } else { "DIFF" }
                    );
                }
            }
        }
    }

    let rec = if args.spawn_viewer {
        RecordingStreamBuilder::new(args.app_id.as_str())
            .spawn()
            .context("failed to spawn rerun viewer")?
    } else {
        let (rec, _storage) = RecordingStreamBuilder::new(args.app_id.as_str())
            .memory()
            .context("failed to init in-memory rerun stream")?;
        rec
    };
    if let Err(e) = send_startup_blueprint(args.app_id.as_str(), &rec) {
        eprintln!("blueprint_setup_error={e:#}");
    }

    let serial = open_serial(&selected_port, args.baud, args.serial_timeout)?;
    let mut raw_file = File::create(&raw_path)?;

    let start = Instant::now();
    let deadline = if args.duration > 0.0 {
        Some(start + Duration::from_secs_f64(args.duration))
    } else {
        None
    };

    let (tx, rx) = mpsc::channel::<RawFrame>();
    let chunk = args.chunk.max(64);
    let capture_start = start;
    let mut serial_reader = serial;
    let raw_path_clone = raw_path.clone();
    thread::spawn(move || {
        let mut seq: u64 = 0;
        let mut tmp = vec![0u8; chunk];
        let mut buf = Vec::<u8>::new();
        let mut out = File::options().append(true).open(raw_path_clone).ok();
        loop {
            match serial_reader.read(&mut tmp) {
                Ok(n) if n > 0 => {
                    if let Some(ref mut f) = out {
                        let _ = f.write_all(&tmp[..n]);
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    for fr in extract_ubx_frames(&mut buf) {
                        if fr.len() < 8 {
                            continue;
                        }
                        let class = fr[2];
                        let id = fr[3];
                        let len = u16::from_le_bytes([fr[4], fr[5]]) as usize;
                        if 6 + len + 2 > fr.len() {
                            continue;
                        }
                        let payload = fr[6..6 + len].to_vec();
                        seq += 1;
                        let elapsed = capture_start.elapsed().as_secs_f64();
                        if tx.send(RawFrame { seq, class, id, payload, elapsed_s: elapsed }).is_err() {
                            return;
                        }
                    }
                }
                _ => {
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }
    });

    let mut msg_counts: HashMap<String, usize> = HashMap::new();
    let mut frame_rate_window: VecDeque<f64> = VecDeque::new();
    let mut last_rate_log = 0.0f64;
    let mut track_points: Vec<(f64, f64)> = Vec::new();
    let mut ubx_frames = 0usize;
    let mut parse_errors = 0usize;
    let mut last_raw_emit_s: Option<f64> = None;
    let mut raw_gap_ms: Vec<f64> = Vec::new();
    let mut last_viewer_probe_s = 0.0f64;
    let mut viewer_seen_connected = false;
    let mut viewer_failed_probes = 0usize;
    let mut viewer_decimator = ViewerDecimator::default();
    let mut last_stream_text_log_s = -1.0f64;
    let mut last_lag_log_s = -1.0f64;

    while deadline.map(|d| Instant::now() < d).unwrap_or(true) {
        let Ok(frame) = rx.recv_timeout(Duration::from_millis(100)) else {
            continue;
        };
        ubx_frames += 1;
        let wall_elapsed_s = start.elapsed().as_secs_f64();
        let viewer_lag_s = (wall_elapsed_s - frame.elapsed_s).max(0.0);
        rec.set_duration_secs("time", frame.elapsed_s);
        let now_ts = safe_now_utc().timestamp() as f64;
        log_scalar(
            &mut viewer_decimator,
            &rec,
            frame.elapsed_s,
            viewer_lag_s,
            "status/timing/utc_seconds",
            now_ts,
        );
        log_scalar(
            &mut viewer_decimator,
            &rec,
            frame.elapsed_s,
            viewer_lag_s,
            "status/stream/seq",
            frame.seq as f64,
        );
        if frame.elapsed_s - last_lag_log_s >= 1.0 {
            last_lag_log_s = frame.elapsed_s;
            let _ = rec.log("status/stream/viewer_lag_s", &rerun::Scalars::new([viewer_lag_s]));
        }
        let ident = match (frame.class, frame.id) {
            (0x01, 0x07) => "NAV-PVT",
            (0x01, 0x35) => "NAV-SAT",
            (0x01, 0x05) => "NAV-ATT",
            (0x01, 0x03) => "NAV-STATUS",
            (0x01, 0x34) => "NAV-ORB",
            (0x10, 0x03) => "ESF-RAW",
            (0x10, 0x04) => "ESF-CAL",
            (0x10, 0x02) => "ESF-MEAS",
            (0x10, 0x14) => "ESF-ALG",
            (0x10, 0x15) => "ESF-INS",
            (0x10, 0x10) => "ESF-STATUS",
            _ => "UNKNOWN",
        };
        *msg_counts.entry(ident.to_string()).or_default() += 1;
        if frame.elapsed_s - last_stream_text_log_s >= 1.0 {
            last_stream_text_log_s = frame.elapsed_s;
            let _ = rec.log(
                "status/stream",
                &rerun::TextLog::new(format!("seq={} {ident}", frame.seq)),
            );
        }

        frame_rate_window.push_back(frame.elapsed_s);
        while frame_rate_window.front().map(|t| frame.elapsed_s - *t > 5.0).unwrap_or(false) {
            frame_rate_window.pop_front();
        }
        if frame.elapsed_s - last_rate_log >= 1.0 {
            let hz = frame_rate_window.len() as f64 / 5.0;
            log_scalar(
                &mut viewer_decimator,
                &rec,
                frame.elapsed_s,
                viewer_lag_s,
                "status/rates/frame_hz",
                hz,
            );
            last_rate_log = frame.elapsed_s;
        }
        if args.spawn_viewer && (frame.elapsed_s - last_viewer_probe_s) >= 1.0 {
            last_viewer_probe_s = frame.elapsed_s;
            match rec.flush_with_timeout(Duration::from_millis(50)) {
                Ok(()) => {
                    viewer_seen_connected = true;
                    viewer_failed_probes = 0;
                }
                Err(SinkFlushError::Timeout) => {
                    // Sink is busy; don't treat as disconnect.
                }
                Err(SinkFlushError::Failed { message }) => {
                    // Avoid false positives during initial viewer startup race.
                    if frame.elapsed_s > 5.0 {
                        viewer_failed_probes += 1;
                        if viewer_seen_connected && viewer_failed_probes >= 3 {
                            println!(
                                "viewer_disconnected=true reason={message} action=auto_exit"
                            );
                            break;
                        }
                    }
                }
            }
        }

        match <Proto31 as UbxProtocol>::match_packet(frame.class, frame.id, &frame.payload) {
            Ok(pkt) => match pkt {
                PacketRef::NavPvt(p) => {
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "gnss/nav_pvt/speed_g_mps", p.ground_speed_2d());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "gnss/nav_pvt/speed_n_mps", p.vel_north());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "gnss/nav_pvt/speed_e_mps", p.vel_east());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "gnss/nav_pvt/speed_d_mps", p.vel_down());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "gnss/nav_pvt/heading_deg", p.heading_motion());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "gnss/nav_pvt/hacc_m", p.horizontal_accuracy());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "gnss/nav_pvt/vacc_m", p.vertical_accuracy());
                    // Same robustness as Python logger:
                    // 1) Gate map updates on valid GNSS fix.
                    // 2) Accept both already-scaled deg and raw 1e-7 UBX units.
                    // 3) Ignore anything outside sane lat/lon bounds.
                    let fix_ok =
                        p.flags().contains(NavPvtFlags::GPS_FIX_OK) && !p.flags3().invalid_llh();
                    if fix_ok {
                        let lat_v = p.latitude();
                        let lon_v = p.longitude();
                        let (lat, lon) = if lat_v.abs() <= 90.0 && lon_v.abs() <= 180.0 {
                            (lat_v, lon_v)
                        } else {
                            (lat_v * 1e-7, lon_v * 1e-7)
                        };
                        if lat.abs() <= 90.0 && lon.abs() <= 180.0 {
                            track_points.push((lat, lon));
                            if track_points.len() > 10_000 {
                                track_points.remove(0);
                            }
                            if track_points.len() >= 2 {
                                let _ = rec.log(
                                    "map/track_line",
                                    &rerun::GeoLineStrings::from_lat_lon([track_points.clone()]),
                                );
                            }
                            let _ = rec.log(
                                "map/track_current",
                                &rerun::GeoPoints::from_lat_lon([(lat, lon)])
                                    .with_radii([rerun::Radius::new_ui_points(4.0)]),
                            );
                        }
                    }
                }
                PacketRef::NavSat(p) => {
                    log_nav_sat(&rec, &p);
                }
                PacketRef::NavStatus(p) => {
                    let fix_ok = p
                        .flags()
                        .contains(ublox::nav_status::NavStatusFlags::GPS_FIX_OK);
                    let diff_soln = p
                        .flags()
                        .contains(ublox::nav_status::NavStatusFlags::DIFF_SOLN);
                    let wkn_set = p
                        .flags()
                        .contains(ublox::nav_status::NavStatusFlags::WKN_SET);
                    let tow_set = p
                        .flags()
                        .contains(ublox::nav_status::NavStatusFlags::TOW_SET);
                    let e_fix = if fix_ok { "✅" } else { "❌" };
                    let e_diff = if diff_soln { "✅" } else { "❌" };
                    let e_wkn = if wkn_set { "✅" } else { "❌" };
                    let e_tow = if tow_set { "✅" } else { "❌" };
                    let has_pr = p.fix_stat().has_pr_prr_correction();
                    let e_pr = if has_pr { "✅" } else { "❌" };
                    let map_match = p.fix_stat().map_matching();
                    let e_map = match map_match {
                        ublox::nav_status::MapMatchingStatus::None => "❌",
                        ublox::nav_status::MapMatchingStatus::Valid => "🟡",
                        ublox::nav_status::MapMatchingStatus::Used => "✅",
                        ublox::nav_status::MapMatchingStatus::Dr => "🧭",
                    };
                    let lines = [
                        format!("- iTOW: {}", p.itow()),
                        format!("- fixType: {:?}", p.fix_type()),
                        format!("- fixOk: {} {}", e_fix, fix_ok),
                        format!("- diffSoln: {} {}", e_diff, diff_soln),
                        format!("- wknSet: {} {}", e_wkn, wkn_set),
                        format!("- towSet: {} {}", e_tow, tow_set),
                        format!("- hasPrPrrCorrection: {} {}", e_pr, has_pr),
                        format!("- mapMatching: {} {:?}", e_map, map_match),
                        format!("- flags2: {:?}", p.flags2()),
                        format!("- ttff_ms: {}", p.time_to_first_fix()),
                        format!("- uptime_ms: {}", p.uptime_ms()),
                    ];
                    let _ = rec.log(
                        "status/navigation/summary/nav_status",
                        &rerun::TextDocument::from_markdown(lines.join("\n")),
                    );
                }
                PacketRef::NavAtt(p) => {
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "fusion/nav_att/roll_deg", p.vehicle_roll());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "fusion/nav_att/pitch_deg", p.vehicle_pitch());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "fusion/nav_att/heading_deg", p.vehicle_heading());
                }
                PacketRef::EsfAlg(p) => {
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "fusion/esf_alg/yaw_deg", p.yaw());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "fusion/esf_alg/roll_deg", p.roll());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "fusion/esf_alg/pitch_deg", p.pitch());
                    let flags = p.flags();
                    let err = p.error();
                    let _ = rec.log(
                        "status/calibration/summary/esf_alg",
                        &rerun::TextDocument::from_markdown(
                            [
                                format!("- iTOW: {}", p.itow()),
                                format!("- version: {}", p.version()),
                                format!("- yaw_deg: {:.3}", p.yaw()),
                                format!("- roll_deg: {:.3}", p.roll()),
                                format!("- pitch_deg: {:.3}", p.pitch()),
                                format!("- autoImuMountAlgOn: {}", flags.auto_imu_mount_alg_on()),
                                format!("- algStatus: {:?}", flags.status()),
                                format!("- flagsRaw: {}", flags.flags_raw()),
                                format!("- errorRaw: {:?}", err),
                                format!(
                                    "- tiltAlgError: {}",
                                    err.contains(ublox::esf_alg::EsfAlgError::TILT_ALG_ERROR)
                                ),
                                format!(
                                    "- yawAlgError: {}",
                                    err.contains(ublox::esf_alg::EsfAlgError::YAW_ALG_ERROR)
                                ),
                                format!(
                                    "- angleError: {}",
                                    err.contains(ublox::esf_alg::EsfAlgError::ANGLE_ERROR)
                                ),
                            ]
                            .join("\n"),
                        ),
                    );
                }
                PacketRef::EsfIns(p) => {
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "imu/gyro/esf_ins/xAngRate_dps", p.x_angular_rate());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "imu/gyro/esf_ins/yAngRate_dps", p.y_angular_rate());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "imu/gyro/esf_ins/zAngRate_dps", p.z_angular_rate());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "imu/accel/esf_ins/xAccel_mps2", p.x_acceleration());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "imu/accel/esf_ins/yAccel_mps2", p.y_acceleration());
                    log_scalar(&mut viewer_decimator, &rec, frame.elapsed_s, viewer_lag_s, "imu/accel/esf_ins/zAccel_mps2", p.z_acceleration());
                }
                PacketRef::EsfStatus(p) => {
                    let init1 = p.init_status1();
                    let init2 = p.init_status2();
                    let mut lines = vec![
                        format!("- iTOW: {}", p.itow()),
                        format!("- version: {}", p.version()),
                        format!("- fusionMode: {:?}", p.fusion_mode()),
                        format!("- numSens: {}", p.num_sens()),
                        format!("- wtInitStatus: {:?}", init1.wheel_tick_init_status()),
                        format!("- mntAlgStatus: {:?}", init1.mounting_angle_status()),
                        format!("- insInitStatus: {:?}", init1.ins_initialization_status()),
                        format!("- imuInitStatus: {:?}", init2.imu_init_status()),
                    ];
                    for (i, s) in p.data().enumerate() {
                        lines.push(format!(
                            "- sensor[{i}]: type={:?} used={} ready={} calib={:?} time={:?} freq={} faults={:?}",
                            s.sensor_type(),
                            s.sensor_used(),
                            s.sensor_ready(),
                            s.calibration_status(),
                            s.time_status(),
                            s.freq(),
                            s.faults()
                        ));
                    }
                    let _ = rec.log(
                        "status/calibration/summary/esf_status",
                        &rerun::TextDocument::from_markdown(lines.join("\n")),
                    );
                }
                PacketRef::EsfRaw(p) => {
                    if let Some(last) = last_raw_emit_s {
                        raw_gap_ms.push((frame.elapsed_s - last) * 1000.0);
                    }
                    last_raw_emit_s = Some(frame.elapsed_s);
                    for d in p.data() {
                        let dtype = d.data_type;
                        let raw_i24 = {
                            let mut v = (d.data_field & 0x00FF_FFFF) as i32;
                            if (v & 0x0080_0000) != 0 {
                                v -= 1 << 24;
                            }
                            v
                        };
                        let (name, scale) = match dtype {
                            14 => ("imu/gyro/rt/x_dps", 2f64.powi(-12)),
                            13 => ("imu/gyro/rt/y_dps", 2f64.powi(-12)),
                            5 => ("imu/gyro/rt/z_dps", 2f64.powi(-12)),
                            16 => ("imu/accel/rt/x_mps2", 2f64.powi(-10)),
                            17 => ("imu/accel/rt/y_mps2", 2f64.powi(-10)),
                            18 => ("imu/accel/rt/z_mps2", 2f64.powi(-10)),
                            _ => continue,
                        };
                        log_scalar(
                            &mut viewer_decimator,
                            &rec,
                            frame.elapsed_s,
                            viewer_lag_s,
                            name,
                            raw_i24 as f64 * scale,
                        );
                    }
                }
                PacketRef::EsfCal(p) => {
                    for d in p.data() {
                        let mut v = (d.data_field & 0x00FF_FFFF) as i32;
                        if (v & 0x0080_0000) != 0 {
                            v -= 1 << 24;
                        }
                        let (name, scale) = match d.data_type {
                            14 => ("imu/gyro/src/esf_cal/gyro_x_dps", 2f64.powi(-12)),
                            13 => ("imu/gyro/src/esf_cal/gyro_y_dps", 2f64.powi(-12)),
                            5 => ("imu/gyro/src/esf_cal/gyro_z_dps", 2f64.powi(-12)),
                            16 => ("imu/accel/src/esf_cal/accel_x_mps2", 2f64.powi(-10)),
                            17 => ("imu/accel/src/esf_cal/accel_y_mps2", 2f64.powi(-10)),
                            18 => ("imu/accel/src/esf_cal/accel_z_mps2", 2f64.powi(-10)),
                            _ => continue,
                        };
                        log_scalar(
                            &mut viewer_decimator,
                            &rec,
                            frame.elapsed_s,
                            viewer_lag_s,
                            name,
                            v as f64 * scale,
                        );
                    }
                }
                PacketRef::EsfMeas(p) => {
                    for d in p.data() {
                        let v = d.data_field;
                        let (name, scale) = match d.data_type as u8 {
                            14 => ("imu/gyro/src/esf_meas/gyro_x_dps", 2f64.powi(-12)),
                            13 => ("imu/gyro/src/esf_meas/gyro_y_dps", 2f64.powi(-12)),
                            5 => ("imu/gyro/src/esf_meas/gyro_z_dps", 2f64.powi(-12)),
                            16 => ("imu/accel/src/esf_meas/accel_x_mps2", 2f64.powi(-10)),
                            17 => ("imu/accel/src/esf_meas/accel_y_mps2", 2f64.powi(-10)),
                            18 => ("imu/accel/src/esf_meas/accel_z_mps2", 2f64.powi(-10)),
                            _ => continue,
                        };
                        log_scalar(
                            &mut viewer_decimator,
                            &rec,
                            frame.elapsed_s,
                            viewer_lag_s,
                            name,
                            v as f64 * scale,
                        );
                    }
                }
                _ => {}
            },
            Err(_) => {
                parse_errors += 1;
            }
        }
    }

    // Stop capture thread by dropping receiver and allow append-buffered writes to flush.
    drop(rx);
    thread::sleep(Duration::from_millis(20));

    let total_bytes = std::fs::metadata(&raw_path).map(|m| m.len()).unwrap_or(0);
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    println!("raw_log={}", raw_path.display());
    println!("bytes={total_bytes}");
    println!("ubx_frames={ubx_frames}");
    println!("parse_errors={parse_errors}");
    println!("duration_s={elapsed:.3}");
    println!("byte_rate={:.1}", total_bytes as f64 / elapsed);
    println!("frame_rate={:.1}", ubx_frames as f64 / elapsed);
    if !raw_gap_ms.is_empty() {
        let mut sorted = raw_gap_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_idx = ((0.95 * sorted.len() as f64) as usize).saturating_sub(1);
        let mean = raw_gap_ms.iter().sum::<f64>() / raw_gap_ms.len() as f64;
        println!("raw_gap_ms_mean={mean:.3}");
        println!("raw_gap_ms_p95={:.3}", sorted[p95_idx]);
        println!("raw_gap_ms_max={:.3}", sorted[sorted.len() - 1]);
    }
    if !msg_counts.is_empty() {
        println!("message_counts:");
        let mut keys: Vec<_> = msg_counts.keys().cloned().collect();
        keys.sort();
        for k in keys {
            if let Some(v) = msg_counts.get(&k) {
                println!("  {k}: {v}");
            }
        }
    }

    // Ensure flush before exit.
    let _ = rec.flush_blocking();
    let _ = raw_file.flush();
    let _ = SystemTime::now().duration_since(UNIX_EPOCH);
    Ok(())
}
