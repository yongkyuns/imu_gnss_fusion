use crate::ubxlog::{fit_linear_map, unwrap_counter};

use super::super::math::nearest_master_ms;

pub fn fit_tag_ms_map(
    seqs: &[u64],
    tags: &[u64],
    masters: &[(u64, f64)],
    unwrap_modulus: Option<u64>,
) -> (Vec<u64>, f64, f64) {
    let mapped_tags = match unwrap_modulus {
        Some(m) => unwrap_counter(tags, m),
        None => tags.to_vec(),
    };
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for (seq, tag_u) in seqs.iter().zip(mapped_tags.iter()) {
        if let Some(ms) = nearest_master_ms(*seq, masters) {
            x.push(*tag_u as f64);
            y.push(ms);
        }
    }
    let (a, b) = fit_linear_map(&x, &y, 1e-3);
    (mapped_tags, a, b)
}
