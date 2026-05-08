//! Reference mount and attitude helpers shared by replay pipelines.

use crate::datasets::generic_replay::GenericReferenceRpySample;
use crate::eval::gnss_ins::{quat_conj, quat_from_rpy_alg_deg, quat_mul};
use crate::visualizer::model::MountSourceMode;
use crate::visualizer::pipeline::generic::GenericReplayInput;

/// Converts a vehicle-to-body mount quaternion into reference mount RPY angles in degrees.
pub fn q_vb_to_reference_mount_rpy(q_vb: [f64; 4]) -> (f64, f64, f64) {
    let q_x_180 = [0.0, 1.0, 0.0, 0.0];
    let q_flu = quat_mul(q_x_180, quat_conj(q_vb));
    quat_rpy_alg_deg(q_flu)
}

/// Converts reference mount RPY angles in degrees into a vehicle-to-body mount quaternion.
pub fn reference_mount_rpy_to_q_vb(rpy_deg: [f64; 3]) -> [f64; 4] {
    let q_x_180 = [0.0, 1.0, 0.0, 0.0];
    let q_flu = quat_from_rpy_alg_deg(rpy_deg[0], rpy_deg[1], rpy_deg[2]);
    quat_mul(quat_conj(q_flu), q_x_180)
}

pub(super) fn reference_mount_seed_q_vb(
    replay: &GenericReplayInput,
    mount_source: MountSourceMode,
) -> Option<[f32; 4]> {
    if !mount_source.uses_ref_mount() {
        return None;
    }
    replay
        .reference_mount
        .iter()
        .rev()
        .find(|sample| {
            sample.roll_deg.is_finite()
                && sample.pitch_deg.is_finite()
                && sample.yaw_deg.is_finite()
        })
        .map(|sample| {
            let q =
                reference_mount_rpy_to_q_vb([sample.roll_deg, sample.pitch_deg, sample.yaw_deg]);
            [q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32]
        })
}

pub(super) fn rpy_series_from_samples(
    samples: &[GenericReferenceRpySample],
) -> Option<[Vec<[f64; 2]>; 3]> {
    if samples.is_empty() {
        return None;
    }
    let mut roll = Vec::with_capacity(samples.len());
    let mut pitch = Vec::with_capacity(samples.len());
    let mut yaw = Vec::with_capacity(samples.len());
    for sample in samples {
        roll.push([sample.t_s, sample.roll_deg]);
        pitch.push([sample.t_s, sample.pitch_deg]);
        yaw.push([sample.t_s, sample.yaw_deg]);
    }
    Some([roll, pitch, yaw])
}

pub(super) fn reference_rpy_at(
    samples: &[GenericReferenceRpySample],
    t_s: f64,
) -> Option<[f64; 3]> {
    if samples.is_empty() {
        return None;
    }
    let idx = samples.partition_point(|sample| sample.t_s < t_s);
    let nearest = match (idx.checked_sub(1), samples.get(idx)) {
        (Some(prev_idx), Some(next)) => {
            let prev = samples[prev_idx];
            if (t_s - prev.t_s).abs() <= (next.t_s - t_s).abs() {
                prev
            } else {
                *next
            }
        }
        (Some(prev_idx), None) => samples[prev_idx],
        (None, Some(next)) => *next,
        (None, None) => return None,
    };
    Some([nearest.roll_deg, nearest.pitch_deg, nearest.yaw_deg])
}

fn quat_rpy_alg_deg(q: [f64; 4]) -> (f64, f64, f64) {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (w, x, y, z) = if n > 1.0e-12 {
        (q[0] / n, q[1] / n, q[2] / n, q[3] / n)
    } else {
        (1.0, 0.0, 0.0, 0.0)
    };
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r10 = 2.0 * (x * y + w * z);
    let r20 = 2.0 * (x * z - w * y);
    let r21 = 2.0 * (y * z + w * x);
    let r22 = 1.0 - 2.0 * (x * x + y * y);
    let pitch = (-r20).clamp(-1.0, 1.0).asin();
    let roll = r21.atan2(r22);
    let yaw = r10.atan2(r00);
    (
        roll.to_degrees(),
        pitch.to_degrees(),
        crate::visualizer::math::normalize_heading_deg(yaw.to_degrees()),
    )
}
