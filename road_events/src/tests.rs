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
    assert!(events[0].duration_s >= 3.0);
    assert_eq!(events[1].kind, HillKind::Downhill);
    assert!(events[1].duration_s >= 3.0);
}

#[test]
fn ignores_short_hill_pitch_excursion() {
    let mut detector = HillDetector::new(HillConfig::default());
    let mut events = 0;
    for i in 0..80 {
        let t = i as f32 * 0.1;
        let pitch_deg = if (2.0..4.0).contains(&t) { 5.0 } else { 0.0 };
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
fn detects_harsh_cornering_from_yaw_rate_and_speed() {
    let mut detector = HarshCornerDetector::new(HarshCornerConfig::default());
    let mut events = Vec::new();
    for i in 0..180 {
        let t = i as f32 * 0.02;
        let yaw_rate_radps = if (1.0..=2.0).contains(&t) { 0.35 } else { 0.0 };
        if let Some(event) = detector.update(HarshCornerSample {
            t_s: t,
            speed_mps: 10.0,
            yaw_rate_radps,
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
        }) {
            if estimate.roughness_rms_mps2 > peak_rms {
                peak_rms = estimate.roughness_rms_mps2;
                peak_level = estimate.level;
            }
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

fn assert_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "actual {actual} expected {expected}"
    );
}
