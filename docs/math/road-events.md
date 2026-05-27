# Road Event Detector Math

`road_events` is a `#![no_std]` crate of independent streaming detectors. Each detector keeps small state, accepts timestamped vehicle-motion samples, and emits event intervals or point events.

## Signals And Units

Inputs are vehicle-frame or scalar vehicle-motion quantities:

| Signal | Unit | Meaning |
| --- | --- | --- |
| `speed_mps` | m/s | nonnegative speed magnitude |
| `forward_velocity_mps` | m/s | signed vehicle-frame forward velocity |
| `vertical_accel_mps2` | m/s^2 | gravity-compensated vertical acceleration |
| `lateral_accel_mps2` | m/s^2 | vehicle-frame lateral specific force / side-load |
| `pitch_deg` | deg | vehicle pitch estimate |

All detectors reject invalid timestamps or non-finite inputs. Large sample gaps are clipped or reset according to detector-specific configuration.

## Streaming Primitives

### High-Pass Filter

The vertical impulse detectors remove slow grade and suspension drift with a first-order high-pass filter:

$$
y_k = \alpha (y_{k-1} + x_k - x_{k-1}),
\qquad
\alpha = \frac{\tau}{\tau + dt}.
$$

### Exponential Moving Average

Velocity-derivative and jerk metrics use EMA smoothing:

$$
\bar{x}_k = \bar{x}_{k-1} + \alpha (x_k - \bar{x}_{k-1}),
\qquad
\alpha = 1 - e^{-dt/\tau}.
$$

### Distance-Domain EMA

Road roughness is normalized by distance rather than wall-clock time:

$$
\alpha_d = 1 - e^{-ds/\tau_d}.
$$

This prevents the same road patch from scoring differently only because the vehicle traversed it at a different speed.

## Road Roughness

Roughness estimates ambient vertical vibration energy:

1. band-pass gravity-compensated vertical acceleration;
2. maintain a robust baseline;
3. clip isolated impulses before integrating ambient energy;
4. update a distance-domain RMS estimate.

The output estimate includes:

- roughness RMS;
- qualitative level;
- raw band-passed acceleration;
- clipped acceleration;
- integrated valid distance;
- whether the current sample updated the estimate.

Roughness deliberately excludes short shocks so a single pothole or speed bump does not dominate ambient road-noise classification.

## Rough-Road And Shock Events

`RoadRoughnessAnalyzer::update_with_events` can emit three event channels:

- live rough-road notification once sustained roughness is confirmed;
- completed rough-road interval when the rough patch exits or is flushed;
- short `RoadShockEvent` for isolated vertical impacts.

Shock detection uses raw band-passed vertical acceleration and an ambient-baseline-scaled threshold. Rough-road detection uses sustained roughness RMS with enter/exit hysteresis and duration/refractory gates.

## Speed Bumps

Speed-bump detection looks for a front/rear-axle vertical impulse pattern combined with pitch response:

- high-pass pitch and vertical acceleration;
- collect positive/negative extrema;
- enforce candidate timing consistent with vehicle speed;
- score pitch and vertical impulse consistency;
- emit confidence after the score clears threshold.

The confidence value is a normalized UI-facing score, not a calibrated probability.

## Hills

Hill detection uses pitch and speed:

- pitch above `+threshold` for uphill;
- pitch below `-threshold` for downhill;
- minimum duration gate.

The default threshold is `4.0 deg` with `1.0 s` confirmation.

## Reverse

Reverse detection uses signed forward velocity with hysteresis:

```text
enter: forward_velocity < -0.5 m/s
exit:  forward_velocity > -0.2 m/s
```

Debounce and minimum-duration gates prevent short sign flips from becoming events.

## Harsh Acceleration And Braking

Harsh acceleration and braking are based on smoothed velocity derivative:

$$
a_k = \frac{v_k - v_{k-1}}{dt}.
$$

Raw derivative values are clamped, then EMA-smoothed. Separate enter/exit thresholds and duration/refractory gates emit acceleration and braking intervals.

## Harsh Cornering

Harsh cornering uses jerk-gated lateral side-load, not `yaw_rate * speed`.

The input lateral acceleration is bias-corrected vehicle-frame specific force:

$$
a_\text{lat} = f_y .
$$

This is the lateral load passengers feel, including bank effects. The detector smooths lateral load, computes and smooths absolute lateral jerk, and only arms a cornering interval when jerk has recently exceeded its threshold. This avoids treating smooth steady-state cornering as harsh by itself.

Balanced preset thresholds are:

```text
enter lateral load = 3.4 m/s^2
exit lateral load  = 2.9 m/s^2
jerk gate          = 5.0 m/s^3
```

## Default Configuration Summary

The crate exposes `Sensitive`, `Balanced`, and `Conservative` harsh behavior presets. Presets adjust harsh acceleration, braking, and cornering thresholds while leaving the detector structure unchanged.

Road roughness default thresholds classify RMS levels from very smooth through severe. Shock defaults are separate from rough-road defaults so impulse artifacts do not inflate ambient roughness.

## Trip Statistics

`TripStats` accumulates constant-memory trip summaries:

- sample counts and invalid samples;
- data gap counts;
- duration and moving/stationary time;
- distance and mean/peak speed;
- reverse duration and distance;
- optional elevation gain/loss;
- event counts and rates per kilometer.

It does not retain sample history.

## Visualizer Interpretation

The visualizer treats point events and segment events separately:

- speed bumps and shocks are point-like map annotations;
- rough-road, hill, reverse, and harsh behavior events are intervals/segments;
- hover trigger traces expose detector context;
- the Events page shows detector-specific plots and trip summary panels.

This mirrors the runtime distinction between instantaneous impacts, sustained roughness, and sustained driver/vehicle behavior intervals.
