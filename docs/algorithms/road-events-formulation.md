# Road Event Detector Formulation

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
\bar{x}_k = (1-\alpha_d)\bar{x}_{k-1} + \alpha_d x_k,
\qquad
\alpha_d = \frac{ds}{\tau_d + ds}.
$$

This first-order form is used instead of a rolling distance buffer, so memory stays constant. It also prevents the same road patch from scoring differently only because the vehicle traversed it at a different speed.

## Road Roughness

Roughness estimates ambient vertical vibration energy:

1. high-pass gravity-compensated vertical acceleration at `0.7 Hz`;
2. low-pass the result at `10 Hz`;
3. maintain a distance-domain robust baseline over `20 m`;
4. clip isolated impulses before integrating ambient energy;
5. update a distance-domain RMS estimate over `10 m`.

The implementation is:

$$
\begin{aligned}
a_\mathrm{bp,k} &= \operatorname{LPF}_{10Hz}(\operatorname{HPF}_{0.7Hz}(a_{z,k})),\\
c_k &= \max(10^{-3}, \min(1.5,\max(0.60,3\,b_k))),\\
a_\mathrm{rob,k} &= \operatorname{clip}(a_\mathrm{bp,k}, -c_k, c_k),\\
e_k &= (1-\alpha_d)e_{k-1}+\alpha_d a_{\mathrm{rob},k}^2,\\
r_k &= \sqrt{\max(e_k,0)}.
\end{aligned}
$$

Here $b_k$ is the distance-domain EMA of $|a_\mathrm{rob}|$. Samples only update the roughness energy when `speed_mps` is at least $2.0$ and traveled distance in the clipped sample interval is positive.

The output estimate includes:

- roughness RMS;
- qualitative level;
- raw band-passed acceleration;
- clipped acceleration;
- integrated valid distance;
- whether the current sample updated the estimate.

Roughness deliberately excludes short shocks so a single pothole or speed bump does not dominate ambient road-noise classification.

Default roughness levels are thresholded on $r_k$:

| Level | RMS range |
| --- | --- |
| `VerySmooth` | $< 0.15\,\mathrm{m/s^2}$ |
| `Smooth` | $< 0.25\,\mathrm{m/s^2}$ |
| `LightTexture` | $< 0.40\,\mathrm{m/s^2}$ |
| `Moderate` | $< 0.60\,\mathrm{m/s^2}$ |
| `Rough` | $< 0.90\,\mathrm{m/s^2}$ |
| `VeryRough` | $< 1.20\,\mathrm{m/s^2}$ |
| `Severe` | $\ge 1.20\,\mathrm{m/s^2}$ |

## Rough-Road And Shock Events

`RoadRoughnessAnalyzer::update_with_events` can emit three event channels:

- live rough-road notification once sustained roughness is confirmed;
- completed rough-road interval when the rough patch exits or is flushed;
- short `RoadShockEvent` for isolated vertical impacts.

Shock detection uses raw band-passed vertical acceleration and an ambient-baseline-scaled threshold. Rough-road detection uses sustained roughness RMS with enter/exit hysteresis and duration/refractory gates.

The shock enter threshold is:

$$
T_\mathrm{shock}=\max(2.5,\;6b_k).
$$

A shock remains active until $|a_\mathrm{bp}| < 0.45T_\mathrm{shock}$. It emits only if the duration is between $0.02\,\mathrm{s}$ and $0.65\,\mathrm{s}$, with a $0.50\,\mathrm{s}$ refractory period. The rough-road interval uses $r_k \ge 0.60\,\mathrm{m/s^2}$ to enter, $r_k \ge 0.42\,\mathrm{m/s^2}$ to remain active, $1.0\,\mathrm{s}$ minimum duration, and $8.0\,\mathrm{s}$ refractory.

## Speed Bumps

Speed-bump detection looks for a front/rear-axle vertical impulse pattern combined with pitch response:

- high-pass pitch and vertical acceleration;
- collect positive/negative extrema;
- enforce candidate timing consistent with vehicle speed;
- score pitch and vertical impulse consistency;
- emit confidence after the score clears threshold.

The detector keeps the latest extrema of high-pass vertical acceleration and evaluates the newest three extrema. A candidate must have alternating signs:

$$
\operatorname{sign}(a_1)=\operatorname{sign}(a_3),
\qquad
\operatorname{sign}(a_1)\ne\operatorname{sign}(a_2).
$$

The pattern duration must fit both fixed and speed-derived bounds:

$$
\begin{aligned}
t_\min &= \max\left(0.18,\frac{0.35\,l_\min}{v}\right),\\
t_\max &= \min\left(1.8,\max\left(\frac{2.5\,l_\max}{v},t_\min\right)\right),
\end{aligned}
$$

with $l_\min = 1.8\,\mathrm{m}$, $l_\max = 3.6\,\mathrm{m}$, and $v \ge 1.5\,\mathrm{m/s}$. This is a loose wheelbase-scale timing model: the vertical impulse pair should compress in time as speed increases, but the constants stay broad enough for different vehicles and suspension responses.

Vertical acceleration also has to be active for at least $25\%$ of the candidate duration and at least $0.25\,\mathrm{s}$. The adaptive thresholds are:

$$
\begin{aligned}
T_a &= \max(3.5\,n_a,\;1.5\,\mathrm{m/s^2}),\\
T_\theta &= \max(3.0\,n_\theta,\;0.25^\circ),
\end{aligned}
$$

where $n_a$ and $n_\theta$ are $6\,\mathrm{s}$ absolute-EMA noise estimates for vertical acceleration and pitch.

The internal pattern score is:

$$
s = s_\theta \operatorname{clamp}
\left(0.45s_a + 0.25s_t + 0.15s_b + 0.15s_\theta,\;0,\;1\right).
$$

The score components are clipped peak-over-threshold terms for acceleration and pitch, a centered timing score, and front/rear balance. The event emits when $s \ge 0.12$, then maps the internal score to a UI confidence near $0.90\ldots0.98$. Confidence is therefore a display score, not a calibrated probability.

## Hills

Hill detection uses pitch and speed:

- pitch above `+threshold` for uphill;
- pitch below `-threshold` for downhill;
- minimum duration gate.

The default threshold is $4.0^\circ$ with $1.0\,\mathrm{s}$ confirmation.

## Reverse

Reverse detection uses signed forward velocity with hysteresis:

$$
\begin{aligned}
\text{enter:}\quad v_x &< -0.5\,\mathrm{m/s},\\
\text{exit:}\quad v_x &> -0.2\,\mathrm{m/s}.
\end{aligned}
$$

Debounce and minimum-duration gates prevent short sign flips from becoming events.

## Harsh Acceleration And Braking

Harsh acceleration and braking are based on smoothed velocity derivative:

$$
a_k = \frac{v_k - v_{k-1}}{dt}.
$$

Raw derivative values are clamped, then EMA-smoothed. Separate enter/exit thresholds and duration/refractory gates emit acceleration and braking intervals.

## Harsh Cornering

Harsh cornering uses jerk-gated lateral side-load, not $\dot{\psi}v$.

The input lateral acceleration is bias-corrected vehicle-frame specific force:

$$
a_\text{lat} = f_y .
$$

This is the lateral load passengers feel, including bank effects. The detector smooths lateral load, computes and smooths absolute lateral jerk, and only arms a cornering interval when jerk has recently exceeded its threshold. This avoids treating smooth steady-state cornering as harsh by itself.

Implementation details:

$$
\begin{aligned}
\bar{a}_{y,k} &= \operatorname{EMA}_{0.15s}(a_{y,k}),\\
j_{y,k} &= \operatorname{clip}\left(\frac{\bar{a}_{y,k}-\bar{a}_{y,k-1}}{dt}, -80, 80\right),\\
\bar{j}_{y,k} &= \operatorname{EMA}_{0.20s}(|j_{y,k}|).
\end{aligned}
$$

The lateral-load interval can start only when a jerk trigger occurred in the previous $0.50\,\mathrm{s}$, the speed is at least $3.0\,\mathrm{m/s}$, and the smoothed lateral load exceeds the preset enter threshold. The interval exits with hysteresis at the preset exit threshold. A confirmed interval must last $0.5\,\mathrm{s}$ and then observes a $2.0\,\mathrm{s}$ refractory period.

Balanced preset thresholds are:

$$
\begin{aligned}
a_{\mathrm{lat,enter}} &= 3.4\,\mathrm{m/s^2},\\
a_{\mathrm{lat,exit}} &= 2.9\,\mathrm{m/s^2},\\
j_{\mathrm{lat,gate}} &= 5.0\,\mathrm{m/s^3}.
\end{aligned}
$$

## Default Configuration Summary

The crate exposes `Sensitive`, `Balanced`, and `Conservative` harsh behavior presets. Presets adjust harsh acceleration, braking, and cornering thresholds while leaving the detector structure unchanged.

| Preset | accel enter/exit | brake enter/exit | corner enter/exit | jerk gate |
| --- | --- | --- | --- | --- |
| `Sensitive` | $2.0 / 1.6\,\mathrm{m/s^2}$ | $2.5 / 2.0\,\mathrm{m/s^2}$ | $2.3 / 1.84\,\mathrm{m/s^2}$ | $3.0\,\mathrm{m/s^3}$ |
| `Balanced` | $2.5 / 2.0\,\mathrm{m/s^2}$ | $3.0 / 2.4\,\mathrm{m/s^2}$ | $3.4 / 2.9\,\mathrm{m/s^2}$ | $5.0\,\mathrm{m/s^3}$ |
| `Conservative` | $3.2 / 2.56\,\mathrm{m/s^2}$ | $4.0 / 3.2\,\mathrm{m/s^2}$ | $3.8 / 3.04\,\mathrm{m/s^2}$ | $6.0\,\mathrm{m/s^3}$ |

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
