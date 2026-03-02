use std::{collections::BTreeMap, fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use polars::prelude::*;
use pygps_rs::ubxlog::{
    extract_esf_alg, extract_esf_cal_samples, extract_esf_meas_samples, extract_esf_raw_samples, extract_itow_ms,
    extract_nav_att, extract_nav_pvt, extract_nav_sat_cn0, extract_tag_ms, identity_counts, identity_from_class_id,
    parse_ubx_frames, sensor_meta,
};
use serde_json::json;

#[derive(Parser, Debug)]
#[command(name = "parse_pygpsdata_log")]
struct Args {
    #[arg(value_name = "LOGFILE")]
    logfile: PathBuf,
    #[arg(long)]
    parquet: Option<PathBuf>,
    #[arg(long)]
    signals_parquet: Option<PathBuf>,
    #[arg(long)]
    max_records: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut bytes = Vec::new();
    File::open(&args.logfile)
        .with_context(|| format!("failed to open {}", args.logfile.display()))?
        .read_to_end(&mut bytes)
        .context("failed to read log")?;

    let frames = parse_ubx_frames(&bytes, args.max_records);
    let shape_rows = frames.len();
    println!("Parsed UBX records: {}", shape_rows);

    let mut seq = Vec::with_capacity(shape_rows);
    let mut offset = Vec::with_capacity(shape_rows);
    let mut msg_class = Vec::with_capacity(shape_rows);
    let mut msg_id = Vec::with_capacity(shape_rows);
    let mut identity = Vec::with_capacity(shape_rows);
    let mut payload_len = Vec::with_capacity(shape_rows);
    let mut itow_ms: Vec<Option<i64>> = Vec::with_capacity(shape_rows);
    let mut tag_ms: Vec<Option<i64>> = Vec::with_capacity(shape_rows);
    let mut fields_json = Vec::with_capacity(shape_rows);

    // Long-format signal rows for fast downstream visualization in Python.
    let mut s_seq: Vec<i64> = Vec::new();
    let mut s_source: Vec<String> = Vec::new();
    let mut s_signal: Vec<String> = Vec::new();
    let mut s_unit: Vec<String> = Vec::new();
    let mut s_value: Vec<f64> = Vec::new();
    let mut s_itow_ms: Vec<Option<i64>> = Vec::new();
    let mut s_tag_ms: Vec<Option<i64>> = Vec::new();

    for f in &frames {
        seq.push(f.seq as i64);
        offset.push(f.offset as i64);
        msg_class.push(format!("0x{:02X}", f.class));
        msg_id.push(format!("0x{:02X}", f.id));
        identity.push(identity_from_class_id(f.class, f.id));
        payload_len.push(f.payload.len() as i64);
        let it = extract_itow_ms(f);
        let tg = extract_tag_ms(f);
        itow_ms.push(it);
        tag_ms.push(tg);
        fields_json.push(
            json!({
                "payload_len": f.payload.len(),
                "itow_ms": it,
                "tag_ms": tg
            })
            .to_string(),
        );

        if let Some((itow, gspeed, vel_n, vel_e, vel_d, _lat, _lon)) = extract_nav_pvt(f) {
            for (name, val) in [("gSpeed", gspeed), ("velN", vel_n), ("velE", vel_e), ("velD", vel_d)] {
                s_seq.push(f.seq as i64);
                s_source.push("NAV-PVT".to_string());
                s_signal.push(name.to_string());
                s_unit.push("m/s".to_string());
                s_value.push(val);
                s_itow_ms.push(Some(itow));
                s_tag_ms.push(None);
            }
        }
        if let Some((itow, roll, pitch, heading)) = extract_nav_att(f) {
            for (name, val) in [("roll", roll), ("pitch", pitch), ("heading", heading)] {
                s_seq.push(f.seq as i64);
                s_source.push("NAV-ATT".to_string());
                s_signal.push(name.to_string());
                s_unit.push("deg".to_string());
                s_value.push(val);
                s_itow_ms.push(Some(itow));
                s_tag_ms.push(None);
            }
        }
        if let Some((itow, roll, pitch, yaw)) = extract_esf_alg(f) {
            for (name, val) in [("roll", roll), ("pitch", pitch), ("yaw", yaw)] {
                s_seq.push(f.seq as i64);
                s_source.push("ESF-ALG".to_string());
                s_signal.push(name.to_string());
                s_unit.push("deg".to_string());
                s_value.push(val);
                s_itow_ms.push(Some(itow));
                s_tag_ms.push(None);
            }
        }
        let itow = extract_itow_ms(f);
        for (sat, cno) in extract_nav_sat_cn0(f) {
            s_seq.push(f.seq as i64);
            s_source.push("NAV-SAT".to_string());
            s_signal.push(format!("cno_{sat}"));
            s_unit.push("dB-Hz".to_string());
            s_value.push(cno);
            s_itow_ms.push(itow);
            s_tag_ms.push(None);
        }
        for (tag, sw) in extract_esf_raw_samples(f) {
            let (sig, unit, scale) = sensor_meta(sw.dtype);
            s_seq.push(f.seq as i64);
            s_source.push("ESF-RAW".to_string());
            s_signal.push(sig.to_string());
            s_unit.push(unit.to_string());
            s_value.push(sw.value_i24 as f64 * scale);
            s_itow_ms.push(None);
            s_tag_ms.push(Some(tag as i64));
        }
        for (tag, sw) in extract_esf_cal_samples(f) {
            let (sig, unit, scale) = sensor_meta(sw.dtype);
            s_seq.push(f.seq as i64);
            s_source.push("ESF-CAL".to_string());
            s_signal.push(sig.to_string());
            s_unit.push(unit.to_string());
            s_value.push(sw.value_i24 as f64 * scale);
            s_itow_ms.push(None);
            s_tag_ms.push(Some(tag as i64));
        }
        for (tag, sw) in extract_esf_meas_samples(f) {
            let (sig, unit, scale) = sensor_meta(sw.dtype);
            s_seq.push(f.seq as i64);
            s_source.push("ESF-MEAS".to_string());
            s_signal.push(sig.to_string());
            s_unit.push(unit.to_string());
            s_value.push(sw.value_i24 as f64 * scale);
            s_itow_ms.push(None);
            s_tag_ms.push(Some(tag as i64));
        }
    }

    let mut df = DataFrame::new(shape_rows, vec![
        Series::new("seq".into(), seq).into(),
        Series::new("offset".into(), offset).into(),
        Series::new("msg_class".into(), msg_class).into(),
        Series::new("msg_id".into(), msg_id).into(),
        Series::new("identity".into(), identity).into(),
        Series::new("payload_len".into(), payload_len).into(),
        Series::new("itow_ms".into(), itow_ms).into(),
        Series::new("tag_ms".into(), tag_ms).into(),
        Series::new("fields_json".into(), fields_json).into(),
    ])?;

    let mut counts = BTreeMap::new();
    for (k, v) in identity_counts(&frames) {
        counts.insert(k, v);
    }
    println!("Record counts by message identity:");
    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (k, c) in sorted {
        println!("{:>12} {}", k, c);
    }

    if let Some(path) = args.parquet {
        let mut file = File::create(&path)?;
        ParquetWriter::new(&mut file).finish(&mut df)?;
        println!("Saved Parquet: {}", path.display());
    }
    if let Some(path) = args.signals_parquet {
        let n = s_seq.len();
        let mut sdf = DataFrame::new(
            n,
            vec![
                Series::new("seq".into(), s_seq).into(),
                Series::new("source".into(), s_source).into(),
                Series::new("signal".into(), s_signal).into(),
                Series::new("unit".into(), s_unit).into(),
                Series::new("value".into(), s_value).into(),
                Series::new("itow_ms".into(), s_itow_ms).into(),
                Series::new("tag_ms".into(), s_tag_ms).into(),
            ],
        )?;
        let mut file = File::create(&path)?;
        ParquetWriter::new(&mut file).finish(&mut sdf)?;
        println!("Saved signals Parquet: {} (rows={})", path.display(), n);
    }
    Ok(())
}
