use super::{
    HarshAccelConfig, HarshAccelDetector, HarshBrakeConfig, HarshBrakeDetector, HarshCornerConfig,
    HarshCornerDetector, HarshCornerSample, HarshLongitudinalSample, HillConfig, HillDetector,
    HillKind, HillSample, ReverseConfig, ReverseDetector, ReverseSample, SpeedBumpConfig,
    SpeedBumpDetector, SpeedBumpSample,
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

fn gaussian(t: f32, center: f32, sigma: f32) -> f32 {
    let z = (t - center) / sigma;
    (-0.5 * z * z).exp()
}
