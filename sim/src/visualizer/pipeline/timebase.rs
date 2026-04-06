use crate::ubxlog::{UbxFrame, extract_itow_ms};

use super::super::math::{nearest_master_ms, unwrap_i64_counter};

#[derive(Clone)]
pub struct MasterTimeline {
    pub masters: Vec<(u64, f64)>,
    pub has_itow: bool,
    pub t0_master_ms: f64,
    pub master_min: f64,
    pub master_max: f64,
}

impl MasterTimeline {
    pub fn master_ms_to_rel_s(&self, master_ms: f64) -> Option<f64> {
        if !self.has_itow || master_ms < self.t0_master_ms {
            return None;
        }
        Some((master_ms - self.t0_master_ms) * 1e-3)
    }

    pub fn seq_to_rel_s(&self, seq: u64) -> Option<f64> {
        let master_ms = nearest_master_ms(seq, &self.masters)?;
        self.master_ms_to_rel_s(master_ms)
    }

    pub fn map_tag_ms(&self, a: f64, b: f64, tag: f64, seq: u64) -> Option<f64> {
        let seq_ms = nearest_master_ms(seq, &self.masters)?;
        let mut ms = a * tag + b;
        if !ms.is_finite()
            || ms < self.master_min - 1000.0
            || ms > self.master_max + 1000.0
            || (ms - seq_ms).abs() > 2000.0
        {
            ms = seq_ms;
        }
        Some(ms)
    }
}

pub fn build_master_timeline(frames: &[UbxFrame]) -> MasterTimeline {
    let mut masters: Vec<(u64, f64)> = Vec::new();
    for f in frames {
        if let Some(itow) = extract_itow_ms(f) {
            if (0..604_800_000).contains(&itow) {
                masters.push((f.seq, itow as f64));
            }
        }
    }
    masters.sort_by_key(|x| x.0);

    if !masters.is_empty() {
        let raw: Vec<i64> = masters.iter().map(|(_, ms)| *ms as i64).collect();
        let unwrapped = unwrap_i64_counter(&raw, 604_800_000);
        for (m, msu) in masters.iter_mut().zip(unwrapped.into_iter()) {
            m.1 = msu as f64;
        }

        let mut sorted_ms: Vec<f64> = masters.iter().map(|(_, ms)| *ms).collect();
        sorted_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_ms = sorted_ms[sorted_ms.len() / 2];
        let dominant_window_ms = 4.0 * 3600.0 * 1000.0;
        let filtered: Vec<(u64, f64)> = masters
            .iter()
            .copied()
            .filter(|(_, ms)| (*ms - median_ms).abs() <= dominant_window_ms)
            .collect();
        if filtered.len() >= 10 {
            masters = filtered;
        }
    }

    let has_itow = !masters.is_empty();
    let t0_master_ms = masters
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::INFINITY, f64::min);
    let t0_master_ms = if t0_master_ms.is_finite() {
        t0_master_ms
    } else {
        0.0
    };

    let master_min = masters
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::INFINITY, f64::min);
    let master_min = if master_min.is_finite() {
        master_min
    } else {
        0.0
    };
    let master_max = masters
        .iter()
        .map(|(_, ms)| *ms)
        .fold(f64::NEG_INFINITY, f64::max);
    let master_max = if master_max.is_finite() {
        master_max
    } else {
        master_min
    };

    MasterTimeline {
        masters,
        has_itow,
        t0_master_ms,
        master_min,
        master_max,
    }
}
