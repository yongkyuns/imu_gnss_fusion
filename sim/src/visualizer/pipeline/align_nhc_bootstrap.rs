use align_rs::align::{Align, AlignConfig, AlignWindowSummary};

use crate::ubxlog::NavPvtObs;
use crate::visualizer::model::ImuPacket;

pub fn resolve_align_nhc_bootstrap_q_vb_seed(
    stationary_accel: &[[f32; 3]],
    prev_nav: Option<(f64, NavPvtObs)>,
    nav_events: &[(f64, NavPvtObs)],
    current_nav_idx: usize,
    interval_start_idx: usize,
    scan_idx: usize,
    imu_packets: &[ImuPacket],
    align_cfg: AlignConfig,
    lookahead_windows: usize,
) -> [f32; 4] {
    let mut align = Align::new(align_cfg);
    if align
        .initialize_from_stationary(stationary_accel, 0.0)
        .is_err()
    {
        return [1.0, 0.0, 0.0, 0.0];
    }

    let Some(prev_nav) = prev_nav else {
        return align.q_vb;
    };

    let end_nav_idx = (current_nav_idx + lookahead_windows).min(nav_events.len());
    let mut local_prev_nav = Some(prev_nav);
    let mut local_interval_start_idx = interval_start_idx;
    let mut local_scan_idx = scan_idx;
    let mut prev_q = align.q_vb;
    let mut heading_evidence_seen = false;
    let mut stable_count = 0usize;
    let mut best_q = align.q_vb;
    let mut best_cost = f32::INFINITY;

    for nav_idx in current_nav_idx..end_nav_idx {
        let (tn, nav) = nav_events[nav_idx];
        if nav_idx > current_nav_idx {
            while local_scan_idx < imu_packets.len() && imu_packets[local_scan_idx].t_ms <= tn {
                local_scan_idx += 1;
            }
        }

        if let Some((t_prev, nav_prev)) = local_prev_nav {
            let dt = ((tn - t_prev) * 1.0e-3) as f32;
            let interval_packets = &imu_packets[local_interval_start_idx..local_scan_idx];
            if dt > 0.0 && !interval_packets.is_empty() {
                let mut gyro_sum = [0.0_f32; 3];
                let mut accel_sum = [0.0_f32; 3];
                for pkt in interval_packets {
                    gyro_sum[0] += pkt.gx_dps.to_radians() as f32;
                    gyro_sum[1] += pkt.gy_dps.to_radians() as f32;
                    gyro_sum[2] += pkt.gz_dps.to_radians() as f32;
                    accel_sum[0] += pkt.ax_mps2 as f32;
                    accel_sum[1] += pkt.ay_mps2 as f32;
                    accel_sum[2] += pkt.az_mps2 as f32;
                }
                let inv_n = 1.0 / interval_packets.len() as f32;
                let window = AlignWindowSummary {
                    dt,
                    mean_gyro_b: [
                        gyro_sum[0] * inv_n,
                        gyro_sum[1] * inv_n,
                        gyro_sum[2] * inv_n,
                    ],
                    mean_accel_b: [
                        accel_sum[0] * inv_n,
                        accel_sum[1] * inv_n,
                        accel_sum[2] * inv_n,
                    ],
                    gnss_vel_prev_n: [
                        nav_prev.vel_n_mps as f32,
                        nav_prev.vel_e_mps as f32,
                        nav_prev.vel_d_mps as f32,
                    ],
                    gnss_vel_curr_n: [
                        nav.vel_n_mps as f32,
                        nav.vel_e_mps as f32,
                        nav.vel_d_mps as f32,
                    ],
                };
                let (_, trace) = align.update_window_with_trace(&window);

                let heading_update = trace.after_pca_yaw_seed.is_some()
                    || trace.after_course_rate.is_some()
                    || trace.after_lateral_accel.is_some()
                    || trace.after_longitudinal_accel.is_some();
                heading_evidence_seen |= heading_update;

                let sigma_roll_deg = align.P[0][0].sqrt().to_degrees();
                let sigma_pitch_deg = align.P[1][1].sqrt().to_degrees();
                let sigma_yaw_deg = align.P[2][2].sqrt().to_degrees();
                let delta_deg = quat_angle_deg(prev_q, align.q_vb);
                prev_q = align.q_vb;

                if heading_evidence_seen {
                    let cost =
                        sigma_yaw_deg + 0.25 * (sigma_roll_deg + sigma_pitch_deg) + 0.5 * delta_deg;
                    if cost < best_cost {
                        best_cost = cost;
                        best_q = align.q_vb;
                    }

                    let stable = sigma_roll_deg <= 1.0
                        && sigma_pitch_deg <= 1.0
                        && sigma_yaw_deg <= 3.0
                        && delta_deg <= 1.0;
                    stable_count = if stable { stable_count + 1 } else { 0 };
                    if stable_count >= 20 {
                        return align.q_vb;
                    }
                }
            }
        }

        local_prev_nav = Some((tn, nav));
        local_interval_start_idx = local_scan_idx;
    }

    if best_cost.is_finite() {
        best_q
    } else {
        align.q_vb
    }
}

fn quat_angle_deg(a: [f32; 4], b: [f32; 4]) -> f32 {
    let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
        .abs()
        .clamp(0.0, 1.0);
    (2.0 * dot.acos()).to_degrees()
}
