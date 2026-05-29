use super::{
    HarshAccelConfig, HarshAccelDetector, HarshBrakeConfig, HarshBrakeDetector, HarshCornerConfig,
    HarshCornerDetector, HarshCornerSample, HarshLongitudinalSample, HillConfig, HillDetector,
    HillKind, HillSample, ReverseConfig, ReverseDetector, ReverseSample, RoadRoughnessAnalyzer,
    RoadRoughnessConfig, RoadRoughnessLevel, RoadRoughnessSample, SpeedBumpConfig,
    SpeedBumpDetector, SpeedBumpSample, TripConfig, TripEventKind, TripSample, TripStats,
};
use std::vec::Vec;

#[test]
fn detects_accel_double_peak_with_pitch_confirmation() {
    let mut detector = SpeedBumpDetector::new(SpeedBumpConfig {
        trigger_confidence: 0.12,
        ..SpeedBumpConfig::default()
    });
    let mut events = 0;
    for i in 0..600 {
        let t = i as f32 * 0.01;
        let accel =
            gaussian(t, 2.00, 0.12) - 1.2 * gaussian(t, 2.30, 0.12) + gaussian(t, 2.60, 0.12);
        let pitch = 0.90 * gaussian(t, 2.10, 0.14) - 1.10 * gaussian(t, 2.42, 0.14);
        let (_, event) = detector.update(SpeedBumpSample {
            t_s: t,
            speed_mps: 4.0,
            pitch_deg: pitch,
            vertical_accel_mps2: 4.0 * accel,
        });
        if event.is_some() {
            events += 1;
        }
    }
    assert_eq!(events, 1);
}

#[test]
fn ignores_slow_pitch_drift() {
    let mut detector = SpeedBumpDetector::new(SpeedBumpConfig::default());
    let mut events = 0;
    for i in 0..700 {
        let t = i as f32 * 0.01;
        let (_, event) = detector.update(SpeedBumpSample {
            t_s: t,
            speed_mps: 5.0,
            pitch_deg: 0.8 * (0.2 * t).sin(),
            vertical_accel_mps2: 0.2 * (0.3 * t).sin(),
        });
        if event.is_some() {
            events += 1;
        }
    }
    assert_eq!(events, 0);
}

#[test]
fn ignores_vertical_impulse_without_pitch_confirmation() {
    let mut detector = SpeedBumpDetector::new(SpeedBumpConfig {
        trigger_confidence: 0.25,
        ..SpeedBumpConfig::default()
    });
    let mut events = 0;
    for i in 0..600 {
        let t = i as f32 * 0.01;
        let accel =
            gaussian(t, 2.00, 0.12) - 1.2 * gaussian(t, 2.30, 0.12) + gaussian(t, 2.60, 0.12);
        let (_, event) = detector.update(SpeedBumpSample {
            t_s: t,
            speed_mps: 4.0,
            pitch_deg: 0.02 * (13.0 * t).sin(),
            vertical_accel_mps2: 4.0 * accel,
        });
        if event.is_some() {
            events += 1;
        }
    }
    assert_eq!(events, 0);
}

#[test]
fn detects_sustained_uphill_and_downhill_intervals() {
    let mut detector = HillDetector::new(HillConfig::default());
    let mut events = Vec::new();
    for i in 0..160 {
        let t = i as f32 * 0.1;
        let pitch_deg = if (2.0..=6.0).contains(&t) {
            4.5
        } else if (9.0..=12.4).contains(&t) {
            -4.8
        } else {
            1.0
        };
        if let Some(event) = detector.update(HillSample {
            t_s: t,
            speed_mps: 5.0,
            pitch_deg,
        }) {
            events.push(event);
        }
    }
    if let Some(event) = detector.finish() {
        events.push(event);
    }

    assert_eq!(events.len(), 2);
    assert_eq!(events[0].kind, HillKind::Uphill);
    assert!(events[0].duration_s >= 1.0);
    assert_eq!(events[1].kind, HillKind::Downhill);
    assert!(events[1].duration_s >= 1.0);
}

#[test]
fn emits_hill_once_when_confirmed_instead_of_waiting_for_exit() {
    let mut detector = HillDetector::new(HillConfig::default());
    let mut events = Vec::new();
    for i in 0..180 {
        let t = i as f32 * 0.1;
        let pitch_deg = if (2.0..=14.0).contains(&t) { 4.5 } else { 0.0 };
        if let Some(event) = detector.update(HillSample {
            t_s: t,
            speed_mps: 5.0,
            pitch_deg,
        }) {
            events.push(event);
        }
    }
    if let Some(event) = detector.finish() {
        events.push(event);
    }

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, HillKind::Uphill);
    assert_close(events[0].start_t_s, 2.0, 1.0e-5);
    assert!(events[0].end_t_s <= 3.1);
    assert!(events[0].duration_s >= 1.0);
}

#[test]
fn ignores_short_hill_pitch_excursion() {
    let mut detector = HillDetector::new(HillConfig::default());
    let mut events = 0;
    for i in 0..80 {
        let t = i as f32 * 0.1;
        let pitch_deg = if (2.0..2.8).contains(&t) { 5.0 } else { 0.0 };
        if detector
            .update(HillSample {
                t_s: t,
                speed_mps: 4.0,
                pitch_deg,
            })
            .is_some()
        {
            events += 1;
        }
    }
    if detector.finish().is_some() {
        events += 1;
    }
    assert_eq!(events, 0);
}

#[test]
fn detects_reverse_interval_with_hysteresis() {
    let mut detector = ReverseDetector::new(ReverseConfig::default());
    let mut events = Vec::new();
    for i in 0..80 {
        let t = i as f32 * 0.1;
        let forward_velocity_mps = if (1.0..=3.0).contains(&t) {
            -0.8
        } else if (3.1..=3.3).contains(&t) {
            -0.15
        } else if (3.4..=4.0).contains(&t) {
            -0.3
        } else {
            0.0
        };
        if let Some(event) = detector.update(ReverseSample {
            t_s: t,
            forward_velocity_mps,
        }) {
            events.push(event);
        }
    }
    if let Some(event) = detector.finish() {
        events.push(event);
    }

    assert_eq!(events.len(), 1);
    assert!(events[0].duration_s >= 1.0);
    assert!(events[0].mean_reverse_speed_mps > 0.4);
    assert!(events[0].peak_reverse_speed_mps >= 0.8);
}

#[test]
fn ignores_short_reverse_velocity_blip() {
    let mut detector = ReverseDetector::new(ReverseConfig::default());
    let mut events = 0;
    for i in 0..30 {
        let t = i as f32 * 0.1;
        let forward_velocity_mps = if (1.0..1.3).contains(&t) { -0.9 } else { 0.0 };
        if detector
            .update(ReverseSample {
                t_s: t,
                forward_velocity_mps,
            })
            .is_some()
        {
            events += 1;
        }
    }
    if detector.finish().is_some() {
        events += 1;
    }
    assert_eq!(events, 0);
}

#[test]
fn detects_harsh_accel_from_velocity_derivative_ema() {
    let mut detector = HarshAccelDetector::new(HarshAccelConfig::default());
    let mut events = Vec::new();
    for i in 0..250 {
        let t = i as f32 * 0.02;
        let forward_velocity_mps = if t < 1.0 {
            2.0
        } else if t <= 3.0 {
            2.0 + 3.0 * (t - 1.0)
        } else {
            8.0
        };
        if let Some(event) = detector.update(HarshLongitudinalSample {
            t_s: t,
            forward_velocity_mps,
        }) {
            events.push(event);
        }
    }
    if let Some(event) = detector.finish() {
        events.push(event);
    }

    assert_eq!(events.len(), 1);
    assert!(events[0].peak_accel_mps2 >= 2.5);
    assert!(events[0].delta_velocity_mps > 0.0);
}

#[test]
fn detects_harsh_brake_from_velocity_derivative_ema() {
    let mut detector = HarshBrakeDetector::new(HarshBrakeConfig::default());
    let mut events = Vec::new();
    for i in 0..250 {
        let t = i as f32 * 0.02;
        let forward_velocity_mps = if t < 1.0 {
            10.0
        } else if t <= 2.5 {
            10.0 - 4.0 * (t - 1.0)
        } else {
            4.0
        };
        if let Some(event) = detector.update(HarshLongitudinalSample {
            t_s: t,
            forward_velocity_mps,
        }) {
            events.push(event);
        }
    }
    if let Some(event) = detector.finish() {
        events.push(event);
    }

    assert_eq!(events.len(), 1);
    assert!(events[0].peak_accel_mps2 >= 3.0);
    assert!(events[0].delta_velocity_mps < 0.0);
}

#[test]
fn detects_harsh_cornering_from_lateral_specific_force_jerk() {
    let mut detector = HarshCornerDetector::new(HarshCornerConfig::default());
    let mut events = Vec::new();
    for i in 0..220 {
        let t = i as f32 * 0.02;
        let lateral_accel_mps2 = if t < 1.0 {
            0.0
        } else if t < 1.22 {
            3.8 * (t - 1.0) / 0.22
        } else if t <= 2.2 {
            3.8
        } else {
            0.0
        };
        if let Some(event) = detector.update(HarshCornerSample {
            t_s: t,
            speed_mps: 12.0,
            lateral_accel_mps2,
        }) {
            events.push(event);
        }
    }
    if let Some(event) = detector.finish() {
        events.push(event);
    }

    assert_eq!(events.len(), 1);
    assert!(events[0].peak_lateral_accel_mps2 >= 3.0);
}

#[test]
fn harsh_cornering_presets_change_detection_behavior_near_threshold() {
    let sensitive = corner_events_for_profile(
        super::HarshBehaviorPreset::Sensitive.configs().corner,
        2.7,
        12.0,
    );
    let balanced = corner_events_for_profile(
        super::HarshBehaviorPreset::Balanced.configs().corner,
        2.7,
        12.0,
    );
    let conservative = corner_events_for_profile(
        super::HarshBehaviorPreset::Conservative.configs().corner,
        3.5,
        12.0,
    );

    assert_eq!(sensitive, 1);
    assert_eq!(balanced, 0);
    assert_eq!(conservative, 0);
}

#[test]
fn harsh_cornering_suppresses_low_speed_and_is_sign_symmetric() {
    assert_eq!(
        corner_events_for_profile(HarshCornerConfig::default(), 4.2, 2.0),
        0
    );
    assert_eq!(
        corner_events_for_signed_profile(HarshCornerConfig::default(), 4.2, 12.0, 1.0),
        1
    );
    assert_eq!(
        corner_events_for_signed_profile(HarshCornerConfig::default(), 4.2, 12.0, -1.0),
        1
    );
}

#[test]
fn ignores_smooth_steady_high_lateral_load() {
    let mut detector = HarshCornerDetector::new(HarshCornerConfig::default());
    let mut events = Vec::new();
    for i in 0..360 {
        let t = i as f32 * 0.02;
        let lateral_accel_mps2 = if t < 1.0 {
            0.0
        } else if t < 3.5 {
            3.8 * (t - 1.0) / 2.5
        } else {
            3.8
        };
        if let Some(event) = detector.update(HarshCornerSample {
            t_s: t,
            speed_mps: 12.0,
            lateral_accel_mps2,
        }) {
            events.push(event);
        }
    }
    if let Some(event) = detector.finish() {
        events.push(event);
    }

    assert_eq!(events.len(), 0);
}

#[test]
fn ignores_invalid_lateral_specific_force() {
    let mut detector = HarshCornerDetector::new(HarshCornerConfig::default());
    let mut events = Vec::new();
    for i in 0..180 {
        let t = i as f32 * 0.02;
        if let Some(event) = detector.update(HarshCornerSample {
            t_s: t,
            speed_mps: 10.0,
            lateral_accel_mps2: f32::NAN,
        }) {
            events.push(event);
        }
    }
    if let Some(event) = detector.finish() {
        events.push(event);
    }

    assert_eq!(events.len(), 0);
}

#[test]
fn harsh_behavior_presets_adjust_only_harsh_thresholds() {
    let sensitive = super::HarshBehaviorPreset::Sensitive.configs();
    let balanced = super::HarshBehaviorPreset::Balanced.configs();
    let conservative = super::HarshBehaviorPreset::Conservative.configs();

    assert_eq!(sensitive.accel.accel_tau_s, balanced.accel.accel_tau_s);
    assert_eq!(conservative.accel.accel_tau_s, balanced.accel.accel_tau_s);
    assert_eq!(
        sensitive.corner.lateral_accel_tau_s,
        balanced.corner.lateral_accel_tau_s
    );
    assert_eq!(
        conservative.corner.lateral_jerk_tau_s,
        balanced.corner.lateral_jerk_tau_s
    );
    assert_close(balanced.corner.lateral_accel_threshold_mps2, 3.4, 1.0e-6);
    assert_close(
        balanced.corner.exit_lateral_accel_threshold_mps2,
        2.9,
        1.0e-6,
    );
    assert_close(balanced.corner.lateral_jerk_threshold_mps3, 5.0, 1.0e-6);

    assert!(sensitive.accel.accel_threshold_mps2 < balanced.accel.accel_threshold_mps2);
    assert!(balanced.accel.accel_threshold_mps2 < conservative.accel.accel_threshold_mps2);
    assert!(sensitive.brake.decel_threshold_mps2 < balanced.brake.decel_threshold_mps2);
    assert!(balanced.brake.decel_threshold_mps2 < conservative.brake.decel_threshold_mps2);
    assert!(
        sensitive.corner.lateral_accel_threshold_mps2
            < balanced.corner.lateral_accel_threshold_mps2
    );
    assert!(
        balanced.corner.lateral_accel_threshold_mps2
            < conservative.corner.lateral_accel_threshold_mps2
    );
    assert!(
        sensitive.corner.lateral_jerk_threshold_mps3 < balanced.corner.lateral_jerk_threshold_mps3
    );
    assert!(
        balanced.corner.lateral_jerk_threshold_mps3
            < conservative.corner.lateral_jerk_threshold_mps3
    );
}

#[test]
fn accumulates_trip_distance_duration_and_speed_stats() {
    let mut stats = TripStats::new(TripConfig {
        moving_speed_threshold_mps: 0.5,
        reverse_speed_threshold_mps: 0.2,
        rolling_tau_s: 2.0,
        max_integrated_dt_s: 1.0,
    });

    for i in 0..=10 {
        stats.update_motion(TripSample {
            t_s: i as f32,
            speed_mps: if i < 3 { 0.0 } else { 10.0 },
            forward_velocity_mps: if i < 3 { 0.0 } else { 10.0 },
            height_m: None,
            height_frame_id: 0,
            longitudinal_accel_mps2: if i == 4 { 2.5 } else { 0.0 },
            lateral_accel_mps2: if i == 6 { -3.2 } else { 0.0 },
        });
    }

    let summary = stats.summary();
    assert_eq!(summary.sample_count, 11);
    assert_close(summary.duration_s, 10.0, 1.0e-5);
    assert_close(summary.moving_duration_s, 8.0, 1.0e-5);
    assert_close(summary.stationary_duration_s, 2.0, 1.0e-5);
    assert_close(summary.distance_m, 75.0, 1.0e-5);
    assert_close(summary.mean_speed_mps, 7.5, 1.0e-5);
    assert_close(summary.moving_mean_speed_mps, 9.375, 1.0e-5);
    assert_close(summary.peak_speed_mps, 10.0, 1.0e-5);
    assert_close(summary.peak_accel_mps2, 2.5, 1.0e-5);
    assert_close(summary.peak_lateral_accel_mps2, 3.2, 1.0e-5);
}

#[test]
fn accumulates_reverse_hill_and_event_rates_without_buffers() {
    let mut stats = TripStats::default();

    for i in 0..=10 {
        let t = i as f32;
        let forward_velocity_mps: f32 = if (2..=5).contains(&i) { -2.0 } else { 4.0 };
        stats.update_motion(TripSample {
            t_s: t,
            speed_mps: forward_velocity_mps.abs(),
            forward_velocity_mps,
            height_m: None,
            height_frame_id: 0,
            longitudinal_accel_mps2: if i == 7 { -3.5 } else { 0.0 },
            lateral_accel_mps2: 0.0,
        });
    }
    stats.record_event(TripEventKind::SpeedBump);
    stats.record_event(TripEventKind::HarshAcceleration);
    stats.record_event(TripEventKind::HarshBraking);
    stats.record_event(TripEventKind::Reverse);

    let summary = stats.summary();
    assert_close(summary.reverse_duration_s, 3.0, 1.0e-5);
    assert_close(summary.reverse_distance_m, 6.0, 1.0e-5);
    assert!(!summary.elevation_valid);
    assert_close(summary.elevation_gain_m, 0.0, 1.0e-5);
    assert_close(summary.elevation_loss_m, 0.0, 1.0e-5);
    assert_eq!(summary.events.speed_bumps, 1);
    assert_eq!(summary.events.harsh_acceleration, 1);
    assert_eq!(summary.events.harsh_braking, 1);
    assert_eq!(summary.events.reverse, 1);
    assert!(summary.speed_bumps_per_km > 0.0);
    assert!(summary.harsh_events_per_km > summary.speed_bumps_per_km);
    assert_close(summary.peak_decel_mps2, 3.5, 1.0e-5);
}

#[test]
fn clips_large_time_gaps_and_counts_invalid_samples() {
    let mut stats = TripStats::new(TripConfig {
        max_integrated_dt_s: 1.0,
        ..TripConfig::default()
    });

    stats.update_motion(TripSample {
        t_s: 0.0,
        speed_mps: 5.0,
        forward_velocity_mps: 5.0,
        height_m: None,
        height_frame_id: 0,
        longitudinal_accel_mps2: 0.0,
        lateral_accel_mps2: 0.0,
    });
    stats.update_motion(TripSample {
        t_s: 5.0,
        speed_mps: 5.0,
        forward_velocity_mps: 5.0,
        height_m: None,
        height_frame_id: 0,
        longitudinal_accel_mps2: 0.0,
        lateral_accel_mps2: 0.0,
    });
    stats.update_motion(TripSample {
        t_s: f32::NAN,
        speed_mps: 5.0,
        forward_velocity_mps: 5.0,
        height_m: None,
        height_frame_id: 0,
        longitudinal_accel_mps2: 0.0,
        lateral_accel_mps2: 0.0,
    });

    let summary = stats.summary();
    assert_eq!(summary.sample_count, 2);
    assert_eq!(summary.invalid_sample_count, 1);
    assert_eq!(summary.data_gap_count, 1);
    assert_close(summary.max_sample_gap_s, 5.0, 1.0e-5);
    assert_close(summary.total_gap_duration_s, 4.0, 1.0e-5);
    assert_close(summary.distance_m, 5.0, 1.0e-5);
}

#[test]
fn accumulates_position_derived_elevation() {
    let mut stats = TripStats::default();
    let heights = [100.0, 101.0, 103.0, 102.0, 104.0];
    for (i, height_m) in heights.into_iter().enumerate() {
        stats.update_motion(TripSample {
            t_s: i as f32,
            speed_mps: 10.0,
            forward_velocity_mps: 10.0,
            height_m: Some(height_m),
            height_frame_id: 0,
            longitudinal_accel_mps2: 0.0,
            lateral_accel_mps2: 0.0,
        });
    }

    let summary = stats.summary();
    assert!(summary.elevation_valid);
    assert_close(summary.elevation_gain_m, 5.0, 1.0e-5);
    assert_close(summary.elevation_loss_m, 1.0, 1.0e-5);
}

#[test]
fn skips_vertical_elevation_across_frame_changes() {
    let mut stats = TripStats::default();
    for (t_s, height_m, height_frame_id) in [(0.0, 100.0, 0), (1.0, 110.0, 0), (2.0, 50.0, 1)] {
        stats.update_motion(TripSample {
            t_s,
            speed_mps: 10.0,
            forward_velocity_mps: 10.0,
            height_m: Some(height_m),
            height_frame_id,
            longitudinal_accel_mps2: 0.0,
            lateral_accel_mps2: 0.0,
        });
    }

    let summary = stats.summary();
    assert!(summary.elevation_valid);
    assert_close(summary.elevation_gain_m, 10.0, 1.0e-5);
    assert_close(summary.elevation_loss_m, 0.0, 1.0e-5);
}

#[test]
fn roughness_updates_over_distance_not_time() {
    let cfg = RoadRoughnessConfig {
        high_pass_cutoff_hz: 0.05,
        low_pass_cutoff_hz: 20.0,
        distance_tau_m: 70.0,
        min_speed_mps: 0.5,
        clip_mps2: 3.0,
        ..RoadRoughnessConfig::default()
    };
    let slow = roughness_for_distance(cfg, 5.0, 160.0, 1.0);
    let fast = roughness_for_distance(cfg, 20.0, 160.0, 1.0);
    assert!((slow - fast).abs() < 0.08, "slow {slow} fast {fast}");
}

#[test]
fn roughness_holds_below_minimum_speed() {
    let mut analyzer = RoadRoughnessAnalyzer::new(RoadRoughnessConfig {
        min_speed_mps: 2.0,
        ..RoadRoughnessConfig::default()
    });
    let mut last = None;
    for i in 0..200 {
        let t = i as f32 * 0.02;
        last = analyzer.update(RoadRoughnessSample {
            t_s: t,
            speed_mps: 0.5,
            vertical_accel_mps2: 2.0 * (12.0 * t).sin(),
        });
    }
    let estimate = last.expect("valid roughness estimate");
    assert!(!estimate.updated);
    assert_close(estimate.distance_m, 0.0, 1.0e-5);
    assert_close(estimate.roughness_rms_mps2, 0.0, 1.0e-5);
}

#[test]
fn roughness_clips_isolated_impulses() {
    let cfg = RoadRoughnessConfig {
        high_pass_cutoff_hz: 0.05,
        low_pass_cutoff_hz: 20.0,
        distance_tau_m: 70.0,
        min_speed_mps: 0.5,
        clip_mps2: 2.0,
        ..RoadRoughnessConfig::default()
    };
    let mut analyzer = RoadRoughnessAnalyzer::new(cfg);
    let mut peak: f32 = 0.0;
    for i in 0..600 {
        let t = i as f32 * 0.02;
        let accel = if (2.0..2.06).contains(&t) { 20.0 } else { 0.15 };
        if let Some(estimate) = analyzer.update(RoadRoughnessSample {
            t_s: t,
            speed_mps: 10.0,
            vertical_accel_mps2: accel,
        }) {
            peak = peak.max(estimate.roughness_rms_mps2);
        }
    }
    assert!(peak <= cfg.clip_mps2 + 1.0e-4, "peak {peak}");
}

#[test]
fn roughness_keeps_discrete_bumps_out_of_rough_level() {
    let cfg = RoadRoughnessConfig::default();
    let mut analyzer = RoadRoughnessAnalyzer::new(cfg);
    let mut peak_rms = 0.0;
    let mut peak_level = RoadRoughnessLevel::VerySmooth;
    for i in 0..4000 {
        let t = i as f32 * 0.02;
        let bump = [8.0, 11.0, 12.5, 14.0]
            .iter()
            .any(|center| (t - center).abs() < 0.015);
        let accel = 0.25 * (8.0 * t).sin() + if bump { 12.0 } else { 0.0 };
        if let Some(estimate) = analyzer.update(RoadRoughnessSample {
            t_s: t,
            speed_mps: 10.0,
            vertical_accel_mps2: accel,
        }) && estimate.roughness_rms_mps2 > peak_rms
        {
            peak_rms = estimate.roughness_rms_mps2;
            peak_level = estimate.level;
        }
    }

    assert_ne!(peak_level, RoadRoughnessLevel::Rough);
    assert_ne!(peak_level, RoadRoughnessLevel::VeryRough);
    assert_ne!(peak_level, RoadRoughnessLevel::Severe);
    assert!(
        peak_rms < cfg.moderate_threshold_mps2,
        "peak roughness {peak_rms}"
    );
}

#[test]
fn roughness_emits_shocks_without_inflating_ambient_texture() {
    let cfg = RoadRoughnessConfig {
        high_pass_cutoff_hz: 0.05,
        low_pass_cutoff_hz: 20.0,
        distance_tau_m: 30.0,
        min_speed_mps: 0.5,
        shock_min_peak_mps2: 2.0,
        ..RoadRoughnessConfig::default()
    };
    let mut analyzer = RoadRoughnessAnalyzer::new(cfg);
    let mut shock_events = 0;
    let mut rough_events = 0;
    let mut peak_rms = 0.0;
    for i in 0..5000 {
        let t = i as f32 * 0.02;
        let shock = [8.0, 11.0, 12.5, 14.0, 25.0, 31.0]
            .iter()
            .any(|center| (t - center).abs() < 0.03);
        let accel = 0.25 * (8.0 * t).sin() + if shock { 10.0 } else { 0.0 };
        let update = analyzer
            .update_with_events(RoadRoughnessSample {
                t_s: t,
                speed_mps: 10.0,
                vertical_accel_mps2: accel,
            })
            .expect("valid roughness sample");
        peak_rms = update.estimate.roughness_rms_mps2.max(peak_rms);
        shock_events += u32::from(update.shock_event.is_some());
        rough_events += u32::from(update.roughness_event.is_some());
    }

    assert!(shock_events >= 3, "shock_events {shock_events}");
    assert_eq!(rough_events, 0);
    assert!(
        peak_rms < cfg.rough_event_enter_mps2,
        "peak roughness {peak_rms}"
    );
}

#[test]
fn roughness_emits_rough_road_event_without_shocks_for_sustained_texture() {
    let cfg = RoadRoughnessConfig {
        high_pass_cutoff_hz: 0.05,
        low_pass_cutoff_hz: 20.0,
        min_speed_mps: 0.5,
        shock_min_peak_mps2: 2.5,
        ..RoadRoughnessConfig::default()
    };
    let mut analyzer = RoadRoughnessAnalyzer::new(cfg);
    let speed_mps = 10.0;
    let dt = 0.02;
    let spatial_wavelength_m = 5.0;
    let mut rough_events = 0;
    let mut shock_events = 0;
    for i in 0..5000 {
        let t = i as f32 * dt;
        let x = speed_mps * t;
        let phase = core::f32::consts::TAU * x / spatial_wavelength_m;
        let accel = if (20.0..90.0).contains(&x) {
            1.15 * phase.sin()
        } else {
            0.08 * phase.sin()
        };
        let update = analyzer
            .update_with_events(RoadRoughnessSample {
                t_s: t,
                speed_mps,
                vertical_accel_mps2: accel,
            })
            .expect("valid roughness sample");
        rough_events += u32::from(update.roughness_event.is_some());
        shock_events += u32::from(update.shock_event.is_some());
    }

    assert!(rough_events >= 1, "rough_events {rough_events}");
    assert_eq!(shock_events, 0);
}

#[test]
fn roughness_live_notification_and_completed_interval_are_separate() {
    let cfg = RoadRoughnessConfig {
        high_pass_cutoff_hz: 0.05,
        low_pass_cutoff_hz: 20.0,
        min_speed_mps: 0.5,
        shock_min_peak_mps2: 2.5,
        ..RoadRoughnessConfig::default()
    };
    let mut analyzer = RoadRoughnessAnalyzer::new(cfg);
    let speed_mps = 10.0;
    let dt = 0.02;
    let spatial_wavelength_m = 5.0;
    let mut live_event = None;
    let mut completed_event = None;
    let mut live_events = 0;
    let mut completed_events = 0;

    for i in 0..5000 {
        let t = i as f32 * dt;
        let x = speed_mps * t;
        let phase = core::f32::consts::TAU * x / spatial_wavelength_m;
        let accel = if (20.0..90.0).contains(&x) {
            1.15 * phase.sin()
        } else {
            0.08 * phase.sin()
        };
        let update = analyzer
            .update_with_events(RoadRoughnessSample {
                t_s: t,
                speed_mps,
                vertical_accel_mps2: accel,
            })
            .expect("valid roughness sample");
        if let Some(event) = update.roughness_event {
            live_events += 1;
            live_event = Some(event);
        }
        if let Some(event) = update.completed_roughness_event {
            completed_events += 1;
            completed_event = Some(event);
        }
    }

    let live_event = live_event.expect("live rough-road notification");
    let completed_event = completed_event.expect("completed rough-road interval");
    assert_eq!(live_events, 1);
    assert_eq!(completed_events, 1);
    assert_eq!(live_event.start_t_s, completed_event.start_t_s);
    assert!(completed_event.end_t_s > live_event.end_t_s);
    assert!(completed_event.duration_s > live_event.duration_s);
}

#[test]
fn roughness_dissipates_after_clean_road_distance() {
    let cfg = RoadRoughnessConfig {
        high_pass_cutoff_hz: 0.05,
        low_pass_cutoff_hz: 20.0,
        min_speed_mps: 0.5,
        clip_mps2: 3.0,
        ..RoadRoughnessConfig::default()
    };
    assert_close(cfg.distance_tau_m, 10.0, 1.0e-5);

    let mut analyzer = RoadRoughnessAnalyzer::new(cfg);
    let speed_mps = 10.0;
    let dt = 0.02;
    let spatial_wavelength_m = 5.0;
    let mut rough_end = 0.0;
    let mut clean_end = 0.0;
    for i in 0..1500 {
        let t = i as f32 * dt;
        let x = speed_mps * t;
        let rough_segment = x < 20.0;
        let phase = core::f32::consts::TAU * x / spatial_wavelength_m;
        let accel = if rough_segment {
            1.2 * phase.sin()
        } else {
            0.02 * phase.sin()
        };
        let estimate = analyzer
            .update(RoadRoughnessSample {
                t_s: t,
                speed_mps,
                vertical_accel_mps2: accel,
            })
            .expect("valid roughness sample");
        if rough_segment {
            rough_end = estimate.roughness_rms_mps2;
        }
        clean_end = estimate.roughness_rms_mps2;
    }

    assert!(
        rough_end > cfg.moderate_threshold_mps2,
        "rough_end {rough_end}"
    );
    assert!(
        clean_end < cfg.smooth_threshold_mps2,
        "rough_end {rough_end} clean_end {clean_end}"
    );
}

#[test]
fn roughness_classifies_levels_from_rms_thresholds() {
    let cfg = RoadRoughnessConfig {
        high_pass_cutoff_hz: 0.05,
        low_pass_cutoff_hz: 20.0,
        distance_tau_m: 30.0,
        min_speed_mps: 0.5,
        clip_mps2: 3.0,
        ..RoadRoughnessConfig::default()
    };
    let very_smooth = roughness_estimate_for_spatial_sine(cfg, 0.12);
    let smooth = roughness_estimate_for_spatial_sine(cfg, 0.25);
    let light_texture = roughness_estimate_for_spatial_sine(cfg, 0.50);
    let moderate = roughness_estimate_for_spatial_sine(cfg, 0.80);
    let rough = roughness_estimate_for_spatial_sine(cfg, 1.10);
    let very_rough = roughness_estimate_for_spatial_sine(cfg, 1.55);
    let severe = roughness_estimate_for_spatial_sine(cfg, 2.20);
    assert_eq!(very_smooth.level, RoadRoughnessLevel::VerySmooth);
    assert_eq!(smooth.level, RoadRoughnessLevel::Smooth);
    assert_eq!(light_texture.level, RoadRoughnessLevel::LightTexture);
    assert_eq!(moderate.level, RoadRoughnessLevel::Moderate);
    assert_eq!(rough.level, RoadRoughnessLevel::Rough);
    assert_eq!(very_rough.level, RoadRoughnessLevel::VeryRough);
    assert_eq!(severe.level, RoadRoughnessLevel::Severe);
}

fn gaussian(t: f32, center: f32, sigma: f32) -> f32 {
    let z = (t - center) / sigma;
    (-0.5 * z * z).exp()
}

fn roughness_for_distance(
    cfg: RoadRoughnessConfig,
    speed_mps: f32,
    distance_m: f32,
    accel_amplitude_mps2: f32,
) -> f32 {
    let mut analyzer = RoadRoughnessAnalyzer::new(cfg);
    let dt = 0.01;
    let samples = (distance_m / (speed_mps * dt)) as usize;
    let spatial_wavelength_m = 8.0;
    let mut estimate = analyzer.estimate();
    for i in 0..samples {
        let t = i as f32 * dt;
        let x = speed_mps * t;
        let phase = core::f32::consts::TAU * x / spatial_wavelength_m;
        estimate = analyzer
            .update(RoadRoughnessSample {
                t_s: t,
                speed_mps,
                vertical_accel_mps2: accel_amplitude_mps2 * phase.sin(),
            })
            .expect("valid roughness sample");
    }
    estimate.roughness_rms_mps2
}

fn roughness_estimate_for_spatial_sine(
    cfg: RoadRoughnessConfig,
    accel_amplitude_mps2: f32,
) -> super::RoadRoughnessEstimate {
    let mut analyzer = RoadRoughnessAnalyzer::new(cfg);
    let mut estimate = analyzer.estimate();
    let speed_mps = 10.0;
    let dt = 0.02;
    let spatial_wavelength_m = 8.0;
    for i in 0..3000 {
        let t = i as f32 * dt;
        let x = speed_mps * t;
        let phase = core::f32::consts::TAU * x / spatial_wavelength_m;
        estimate = analyzer
            .update(RoadRoughnessSample {
                t_s: t,
                speed_mps,
                vertical_accel_mps2: accel_amplitude_mps2 * phase.sin(),
            })
            .expect("valid roughness sample");
    }
    estimate
}

fn corner_events_for_profile(
    cfg: HarshCornerConfig,
    peak_lateral_accel_mps2: f32,
    speed_mps: f32,
) -> usize {
    corner_events_for_signed_profile(cfg, peak_lateral_accel_mps2, speed_mps, 1.0)
}

fn corner_events_for_signed_profile(
    cfg: HarshCornerConfig,
    peak_lateral_accel_mps2: f32,
    speed_mps: f32,
    sign: f32,
) -> usize {
    let mut detector = HarshCornerDetector::new(cfg);
    let mut events = 0;
    for i in 0..220 {
        let t = i as f32 * 0.02;
        let lateral_accel_mps2 = if t < 1.0 {
            0.0
        } else if t < 1.22 {
            peak_lateral_accel_mps2 * (t - 1.0) / 0.22
        } else if t <= 2.2 {
            peak_lateral_accel_mps2
        } else {
            0.0
        };
        events += detector
            .update(HarshCornerSample {
                t_s: t,
                speed_mps,
                lateral_accel_mps2: sign * lateral_accel_mps2,
            })
            .is_some() as usize;
    }
    events + detector.finish().is_some() as usize
}

fn assert_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "actual {actual} expected {expected}"
    );
}
