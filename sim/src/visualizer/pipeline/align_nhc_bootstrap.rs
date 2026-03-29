use align_rs::align::{Align, AlignConfig};

use crate::ubxlog::NavPvtObs;
use crate::visualizer::model::ImuPacket;

#[derive(Clone, Copy, Debug)]
pub struct AlignNhcBootstrapSeed {
    pub q_vb: [f32; 4],
    pub sigma_rad: [f32; 3],
}

pub fn resolve_align_nhc_bootstrap_q_vb_seed(
    stationary_accel: &[[f32; 3]],
    _prev_nav: Option<(f64, NavPvtObs)>,
    _nav_events: &[(f64, NavPvtObs)],
    _current_nav_idx: usize,
    _interval_start_idx: usize,
    _scan_idx: usize,
    _imu_packets: &[ImuPacket],
    align_cfg: AlignConfig,
    _lookahead_windows: usize,
) -> AlignNhcBootstrapSeed {
    let mut align = Align::new(align_cfg);
    if align
        .initialize_from_stationary(stationary_accel, 0.0)
        .is_err()
    {
        return AlignNhcBootstrapSeed {
            q_vb: [1.0, 0.0, 0.0, 0.0],
            sigma_rad: [10.0_f32.to_radians(); 3],
        };
    }

    AlignNhcBootstrapSeed {
        q_vb: align.q_vb,
        sigma_rad: clipped_mount_sigma_rad(align.P),
    }
}

fn clipped_mount_sigma_rad(p: [[f32; 3]; 3]) -> [f32; 3] {
    [
        p[0][0].sqrt().clamp(0.5_f32.to_radians(), 3.0_f32.to_radians()),
        p[1][1].sqrt().clamp(0.5_f32.to_radians(), 3.0_f32.to_radians()),
        p[2][2].sqrt().clamp(1.0_f32.to_radians(), 5.0_f32.to_radians()),
    ]
}
